"""Resumable local-only construction, deterministic repairs, no solver filtering."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
from compile_scale09 import build
from run_generation import ROOT, read, dump, digest, run_lock, NoRedirect
from scale09_prompt import messages, schema
from scale09_rules import validate
from token_count import count_chat_tokens

WORK = ROOT / 'construction/scale09_dev120'
CODE = ('run_scale09.py', 'scale09_prompt.py', 'scale09_rules.py', 'compile_scale09.py', 'token_count.py')


def load_sources():
    path = WORK / 'development_sources.private.jsonl'
    expected = read(WORK / 'development_sources_integrity.json')
    if digest(path) != expected['sha256']: raise ValueError('Development source transfer changed')
    sources = {s['source_id']: s for s in (json.loads(line) for line in path.read_text(encoding='utf-8').splitlines())}
    assert len(sources) == 200 and all(s['split'] == 'development' for s in sources.values())
    return sources


def one(source, proto, tokenizer, port):
    sid = source['source_id']; folder = WORK / 'sources' / sid
    if (folder / 'result.json').exists(): return read(folder / 'result.json')
    started = time.monotonic(); errors = []; history = []
    for attempt in range(1 + proto['construction_repairs']['maximum']):
        dest = folder / ('candidate_' + str(attempt + 1).zfill(2))
        reqfile, rawfile = dest / 'request.private.json', dest / 'response.private.json'
        if reqfile.exists(): body = read(reqfile)
        else:
            msg = messages(source, errors)
            n = count_chat_tokens(tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, enable_thinking=False))
            if n + proto['teacher_settings']['output_tokens'] > proto['teacher_settings']['context']:
                history.append({'candidate': attempt + 1, 'status': 'context_preflight_rejected_no_truncation', 'input_tokens': n})
                break
            body = {'model': proto['teacher_model'], 'messages': msg, 'max_tokens': proto['teacher_settings']['output_tokens'],
                    'temperature': proto['teacher_settings']['temperature'], 'stream': False,
                    'chat_template_kwargs': {'enable_thinking': False},
                    'response_format': {'type': 'json_schema', 'json_schema': {'name': 'source_rules', 'strict': True, 'schema': schema()}}}
            dump(reqfile, body); dump(dest / 'token_preflight.json', {'input_tokens': n, 'max_output_tokens': body['max_tokens'], 'context': 65536, 'truncated': False})
        transport = []
        if rawfile.exists(): raw = read(rawfile)
        else:
            raw = None
            for tx in range(proto['construction_repairs']['transport_attempts_per_candidate']):
                try:
                    req = urllib.request.Request('http://127.0.0.1:' + str(port) + '/v1/chat/completions',
                                                 data=json.dumps(body, ensure_ascii=False).encode(), headers={'Content-Type': 'application/json'}, method='POST')
                    with urllib.request.build_opener(NoRedirect()).open(req, timeout=1800) as response: raw = json.load(response)
                    dump(rawfile, raw); transport.append({'attempt': tx + 1, 'status': 'response_saved'}); break
                except urllib.error.HTTPError as e:
                    dump(dest / ('transport_' + str(tx + 1) + '.private.json'), {'status': e.code, 'body': e.read(5000).decode(errors='replace')})
                    transport.append({'attempt': tx + 1, 'status': 'http_' + str(e.code)})
                except Exception as e:
                    transport.append({'attempt': tx + 1, 'status': type(e).__name__, 'error': str(e)[:180]})
                dump(dest / 'transport.json', transport)
            if raw is None:
                history.append({'candidate': attempt + 1, 'status': 'transport_failed', 'transport': transport}); break
        dump(dest / 'transport.json', transport)
        try:
            choice = raw['choices'][0]
            if choice['finish_reason'] == 'length':
                errors = ['output_limit_incomplete_candidate']
                clean = None
            else:
                candidate = json.loads(choice['message']['content']); dump(dest / 'candidate.private.json', candidate)
                clean, errors = validate(candidate, source)
        except Exception as e:
            clean, errors = None, ['candidate_parse_' + type(e).__name__ + ': ' + str(e)[:180]]
        report = {'candidate': attempt + 1, 'machine_valid': clean is not None, 'errors': errors,
                  'not_clinical_semantic_validation': True, 'finish_reason': raw.get('choices', [{}])[0].get('finish_reason'),
                  'usage': raw.get('usage')}
        dump(dest / 'validation.json', report); history.append(report)
        if clean is not None:
            dump(folder / 'accepted_spec.private.json', clean)
            row = {'source_id': sid, 'source_cluster': source['source_cluster'], 'modality': source['modality'],
                   'length_band': source['length_band'], 'status': 'machine_valid_silver', 'machine_valid': True,
                   'selected_candidate': attempt + 1, 'first_candidate_valid': attempt == 0,
                   'families': len(clean), 'balanced_triplets': len(clean) - 1,
                   'multi_step': sum(i['family_type'] == 'multi_step' for i in clean),
                   'history': history, 'seconds': time.monotonic() - started}
            dump(folder / 'result.json', row); return row
        print(json.dumps({'event': 'construction_repair', 'source': sid, 'candidate': attempt + 1, 'machine_errors': errors[:8]}, ensure_ascii=False), flush=True)
    row = {'source_id': sid, 'source_cluster': source['source_cluster'], 'modality': source['modality'],
           'length_band': source['length_band'], 'status': 'rejected_preserved', 'machine_valid': False,
           'history': history, 'seconds': time.monotonic() - started}
    dump(folder / 'result.json', row); return row


def gather():
    return [read(p) for p in sorted((WORK / 'sources').glob('*/result.json'))]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--ports', nargs='+', type=int, required=True)
    p.add_argument('--pilot', action='store_true'); p.add_argument('--allow-private-data', action='store_true'); a = p.parse_args()
    if not a.allow_private_data or not a.ports or any(p not in (8071, 8072) for p in a.ports):
        raise ValueError('Explicit user-server local-only construction ports required')
    proto = read(WORK / 'protocol.json'); sources = load_sources()
    if digest(WORK / 'source_pool_build_integrity.json') != proto['source_pool_integrity_sha256']: raise ValueError('Source pool snapshot changed')
    plan = {'protocol_sha256': digest(WORK / 'protocol.json'), 'code_sha256': {n: digest(ROOT / n) for n in CODE},
            'teacher_model': proto['teacher_model'], 'not_solver_evaluation': True,
            'development_input_sha256': digest(WORK / 'development_sources.private.jsonl')}
    with run_lock(WORK / 'run.lock'):
        if (WORK / 'run_plan.json').exists() and read(WORK / 'run_plan.json') != plan: raise ValueError('Frozen construction code changed')
        dump(WORK / 'run_plan.json', plan)
        from transformers import AutoTokenizer
        path = Path(proto['teacher_path'])
        if digest(path / 'config.json') != proto['teacher_config_sha256']: raise ValueError('Teacher identity changed')
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        for port in a.ports:
            with urllib.request.build_opener(NoRedirect()).open('http://127.0.0.1:' + str(port) + '/v1/models', timeout=15) as response:
                if proto['teacher_model'] not in {m['id'] for m in json.load(response)['data']}: raise ValueError('Wrong local model')
        reuse = set(proto['reused_source_ids']); queues = proto['stratum_queue']
        # First finish the complete fixed-primary queue, not a teacher-selected
        # easy subset. The pilot is a deterministic prefix and is retained.
        primary = [s for q in queues for s in q['primary_ids'] if s not in reuse]
        if a.pilot:
            primary = [next(s for s in q['primary_ids'] if s not in reuse) for q in queues][:4]
        def batch(ids):
            with ThreadPoolExecutor(max_workers=4 * len(a.ports)) as ex:
                futures = [ex.submit(one, sources[sid], proto, tokenizer, a.ports[k % len(a.ports)]) for k, sid in enumerate(ids)]
                for future in as_completed(futures):
                    r = future.result(); rows = gather()
                    dump(WORK / 'progress.json', {'attempted_sources': len(rows), 'machine_valid': sum(r['machine_valid'] for r in rows),
                                                  'rejected': sum(not r['machine_valid'] for r in rows), 'pilot': a.pilot,
                                                  'old_8_reused_separately': 8, 'target_total': 120, 'solver_calls': 0})
                    print(json.dumps({'event': 'source_constructed', 'source': r['source_id'], 'status': r['status'],
                                      'families': r.get('families'), 'recorded_sources': len(rows)}, ensure_ascii=False), flush=True)
        batch(primary)
        if a.pilot:
            rows = [read(WORK / 'sources' / sid / 'result.json') for sid in primary]
            dump(WORK / 'pilot.json', {'selected_source_ids': primary, 'machine_valid': sum(r['machine_valid'] for r in rows), 'all_attempts_retained': True})
            return
        accepted = {}; selection = []
        for q in queues:
            kept = [s for s in q['primary_ids'] if s in reuse or read(WORK / 'sources' / s / 'result.json')['machine_valid']]
            reserve_used = []
            for sid in q['reserve_ids']:
                if len(kept) >= q['target']: break
                batch([sid]); reserve_used.append(sid)
                if read(WORK / 'sources' / sid / 'result.json')['machine_valid']: kept.append(sid)
            for sid in kept:
                if sid not in reuse: accepted[sid] = read(WORK / 'sources' / sid / 'accepted_spec.private.json')
            selection.append({**q, 'selected_ids': kept, 'reserve_attempted_ids': reserve_used, 'unfilled_sources': q['target'] - len(kept)})
        dump(WORK / 'selection.json', selection)
        output = ROOT / 'datasets/medical09_dev120_silver'
        audit = read(output / 'construction_audit.json') if output.exists() else build(proto, sources, accepted, output)
        rows = gather()
        final = {'status': 'target_constructed_machine_silver' if audit['source_clusters'] == 120 else 'partial_target_not_met',
                 'dataset': str(output), 'dataset_integrity_sha256': digest(output / 'build_integrity.json'),
                 'audit': audit, 'attempted_new_sources': len(rows), 'first_candidate_valid': sum(r.get('first_candidate_valid', False) for r in rows),
                 'machine_valid_candidates': sum(r['machine_valid'] for r in rows), 'rejected_sources_preserved': sum(not r['machine_valid'] for r in rows),
                 'expert_clinical_validation_completed': False, 'solver_experiments_completed': 0,
                 'old_gate_status_unchanged': True}
        dump(WORK / 'complete.json', final)
        print(json.dumps({'event': 'construction_complete', **final}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
