"""Resumable native-agent branches; only actor bundle is loaded on server."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
from run_generation import ROOT, read, dump, digest, run_lock, NoRedirect
from pilot_runtime import fingerprint
from typed_client03 import tokenizer_messages
from token_count import count_chat_tokens
from workflow11_runtime import World, POLICY, fork, export_observations

CODE = ('workflow11_runtime.py', 'run_workflow11.py', 'medical_runtime07.py', 'paired_runtime06.py',
        'pilot_runtime.py', 'medical_runtime04.py', 'typed_client03.py', 'token_count.py', 'run_generation.py')
ARMS = ('evidence_constraint', 'labels_only', 'fresh')


def snapshot(w):
    data = copy.deepcopy(w.__dict__)
    for k in ('observed', 'seen_documents'):
        data[k] = sorted(data[k])
    if data['saved_observations'] is not None:
        data['saved_observations'] = [sorted(x) for x in data['saved_observations']]
    return data


def restore(data):
    w = World(data['public'])
    w.__dict__.update(copy.deepcopy(data))
    for k in ('observed', 'seen_documents'):
        setattr(w, k, set(getattr(w, k)))
    if w.saved_observations is not None:
        w.saved_observations = tuple(set(x) for x in w.saved_observations)
    return w


def request_body(messages, tools, tokenizer, model, settings):
    count = count_chat_tokens(tokenizer.apply_chat_template(tokenizer_messages(messages), tools=tools,
                tokenize=True, add_generation_prompt=True, enable_thinking=False))
    if count + settings['max_output_tokens'] > 65536:
        raise ValueError('context_limit_no_silent_truncation')
    wire = copy.deepcopy(messages)
    for msg in wire:
        if msg['role'] == 'assistant' and msg.get('content') is None:
            msg['content'] = ''
    return {'model': model, 'messages': wire, 'tools': tools, 'tool_choice': 'auto',
            'parallel_tool_calls': False, 'temperature': settings['temperature'], 'seed': settings['seed'],
            'max_tokens': settings['max_output_tokens'], 'stream': False,
            'chat_template_kwargs': {'enable_thinking': False}}, count


def phase(w, folder, port, model, tokenizer, settings):
    started = time.monotonic()
    start_hash = fingerprint(w.state)
    public_hash = fingerprint(w.public)
    ck = folder / 'checkpoint.private.json'
    calls, elapsed, next_round, status = 0, 0, 1, 'round_limit'
    if ck.exists():
        saved = read(ck)
        if saved['start_hash'] != start_hash or saved['public_hash'] != public_hash:
            raise ValueError('Checkpoint start differs')
        w, messages = restore(saved['world']), saved['messages']
        calls, elapsed, next_round, status = saved['requests'], saved['seconds'], saved['next_round'], saved['status']
        if saved['terminal']:
            return w, status, calls, elapsed
    else:
        messages = [{'role': 'system', 'content': POLICY},
                    {'role': 'user', 'content': json.dumps(w.initial_input(), ensure_ascii=False)}]
    def save(terminal=False):
        dump(ck, {'start_hash': start_hash, 'public_hash': public_hash, 'world': snapshot(w), 'messages': messages,
                  'requests': calls, 'seconds': elapsed + time.monotonic() - started,
                  'next_round': next_round, 'status': status, 'terminal': terminal})
    save()
    silent = 0
    for number in range(next_round, settings['max_rounds'] + 1):
        response = folder / ('response_%03d.private.json' % number)
        pending = folder / ('pending_%03d.json' % number)
        try:
            if response.exists():
                raw = read(response)
                calls = max(calls, number)
            elif pending.exists():
                status = 'indeterminate_inflight_no_resample'
                break
            else:
                body, ntokens = request_body(messages, w.tools(), tokenizer, model, settings)
                dump(folder / ('request_%03d.private.json' % number), body)
                dump(pending, {'input_tokens': ntokens, 'started': time.time(), 'request_sha256': fingerprint(body)})
                calls += 1
                req = urllib.request.Request('http://127.0.0.1:' + str(port) + '/v1/chat/completions',
                    data=json.dumps(body, ensure_ascii=False).encode(), headers={'Content-Type': 'application/json'}, method='POST')
                with urllib.request.build_opener(NoRedirect()).open(req, timeout=1800) as res:
                    raw = json.load(res)
                dump(response, raw)
            choice = raw['choices'][0]
            if choice['finish_reason'] == 'length':
                status = 'output_limit_not_content_error'
                break
            if choice['finish_reason'] not in ('stop', 'tool_calls'):
                status = 'response_not_completed'
                break
            message = {k: v for k, v in choice['message'].items() if k in ('role', 'content', 'tool_calls') and v is not None}
            if message.get('role') != 'assistant':
                raise ValueError('Invalid assistant role')
            messages.append(message)
            tool_calls = message.get('tool_calls', [])
            if not tool_calls:
                silent += 1
                if silent >= 2:
                    status = 'no_submission'
                    break
                messages.append({'role': 'user', 'content': '请通过公开工具完成任务，最终调用 submit_answer 提交本次全部项目；不返回评分。'})
            else:
                silent = 0
                for tc in tool_calls:
                    try:
                        args = json.loads(tc['function']['arguments'])
                    except (ValueError, TypeError):
                        args = None
                    result = w.call(tc['function']['name'], args)
                    messages.append({'role': 'tool', 'tool_call_id': tc['id'], 'content': json.dumps(result, ensure_ascii=False)})
            next_round = number + 1
            if w.done:
                status = 'submitted'
                break
            save()
        except urllib.error.HTTPError as e:
            # Store error text only inside the private server trace, not reports.
            dump(folder / ('http_error_%03d.private.json' % number), {'status': e.code, 'body': e.read(12000).decode(errors='replace')})
            status = 'http_' + str(e.code)
            break
        except Exception as e:
            status = str(e) if isinstance(e, ValueError) and str(e) == 'context_limit_no_silent_truncation' else 'runner_or_transport_' + type(e).__name__
            break
    save(True)
    return w, status, calls, elapsed + time.monotonic() - started


def one(actor, event, arm, out, port, model, tokenizer, settings, base=None):
    sid = actor['source_id']
    eid = 'initial' if event is None else event['event_id']
    folder = out / 'phases' / sid / (eid + '__' + arm)
    if base is not None and not base.done:
        row = {'source_id': sid, 'event_id': eid, 'arm': arm, 'submitted': False,
               'status': 'blocked_actual_initial_not_submitted', 'requests': 0, 'seconds': 0}
        dump(out / 'rows' / sid / (eid + '__' + arm + '.json'), row)
        return row, None
    w = World(actor['initial']) if event is None else fork(base, event['controller'], eid, arm)
    start_hash = fingerprint(w.state)
    w, status, calls, seconds = phase(w, folder, port, model, tokenizer, settings)
    row = {'source_id': sid, 'event_id': eid, 'arm': arm, 'submitted': w.done, 'status': status,
           'requests': calls, 'seconds': seconds, 'pre_state_hash': start_hash,
           'initial_checkpoint_hash': fingerprint(snapshot(base)) if base is not None else None,
           'public_hash': fingerprint(w.public), 'environment': w.public['environment'],
           **export_observations(w)}
    dump(out / 'rows' / sid / (eid + '__' + arm + '.json'), row)
    return row, w


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=('Qwen3.5-9B', 'Qwen3.8-27B'))
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--allow-private-data', action='store_true')
    a = p.parse_args()
    if not a.allow_private_data:
        raise ValueError('Explicit private source permission required')
    protocol = read(ROOT / 'workflow11_protocol.json')
    inputs = ROOT / 'workflow11_inputs_private'
    am = read(inputs / 'actor_manifest.json')
    if am['dataset_integrity_sha256'] != protocol['dataset_integrity_sha256']:
        raise ValueError('Actor dataset mismatch')
    for name, sha in am['files'].items():
        if name != 'actors.private.jsonl' or digest(inputs / name) != sha:
            raise ValueError('Actor bundle mismatch')
    actors = [json.loads(s) for s in (inputs / 'actors.private.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(actors) != 120 or any(len(x['events']) != 4 for x in actors):
        raise ValueError('Need full fixed120 source matrix')
    slug = 'q9b' if a.model == 'Qwen3.5-9B' else 'q27b'
    out = ROOT / 'runs' / ('workflow11_' + slug + '_01')
    plan = {'model': a.model, 'port': a.port, 'harness': 'native_workflow11',
            'protocol_sha256': digest(ROOT / 'workflow11_protocol.json'),
            'actor_manifest_sha256': digest(inputs / 'actor_manifest.json'),
            'code_sha256': {n: digest(ROOT / n) for n in CODE},
            'planned': 1560, 'decoding': protocol['decoding'], 'gold_loaded_on_server': False}
    with run_lock(out / 'run.lock'):
        if (out / 'plan.json').exists() and read(out / 'plan.json') != plan:
            raise ValueError('Preserve frozen workflow run')
        dump(out / 'plan.json', plan)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('/data/zhanchen/models/' + a.model, local_files_only=True, trust_remote_code=False)
        rows = []
        with ThreadPoolExecutor(max_workers=protocol['decoding']['workers_per_model']) as ex:
            # Phase-level scheduling lets branches run concurrently; order is a
            # score-blind hash and never selected by initial correctness.
            futures = {ex.submit(one, actor, None, 'initial', out, a.port, a.model, tokenizer, protocol['decoding']): (actor, None)
                       for actor in sorted(actors, key=lambda x: fingerprint(x['source_id']))}
            from concurrent.futures import wait, FIRST_COMPLETED
            while futures:
                ready, _ = wait(futures, return_when=FIRST_COMPLETED)
                for f in ready:
                    actor, ev = futures.pop(f)
                    row, world = f.result()
                    rows.append(row)
                    dump(out / 'results.json', rows)
                    statuses = {}
                    for r in rows:
                        statuses[r['status']] = statuses.get(r['status'], 0) + 1
                    dump(out / 'progress.json', {'model': a.model, 'planned': 1560, 'recorded': len(rows),
                         'submitted': sum(r['submitted'] for r in rows), 'statuses': statuses, 'updated': time.time()})
                    if ev is None:
                        jobs = sorted([(event, arm) for event in actor['events'] for arm in ARMS],
                                      key=lambda t: fingerprint(actor['source_id'] + '|' + t[0]['event_id'] + '|' + t[1]))
                        for event, arm in jobs:
                            futures[ex.submit(one, actor, event, arm, out, a.port, a.model, tokenizer,
                                              protocol['decoding'], copy.deepcopy(world))] = actor, event
                    if len(rows) % 20 == 0:
                        print(json.dumps({'event': 'workflow11_progress', 'model': a.model, 'recorded': len(rows), 'submitted': sum(r['submitted'] for r in rows)}), flush=True)
        if len(rows) != 1560:
            raise ValueError('Not all phases accounted')
        dump(out / 'complete.json', {'model': a.model, 'planned': 1560, 'recorded': len(rows),
              'submitted': sum(r['submitted'] for r in rows), 'finished': time.time(), 'all_phases_accounted': True,
              'gold_loaded_on_server': False})
        print(json.dumps({'event': 'workflow11_model_complete', 'model': a.model, 'recorded': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
