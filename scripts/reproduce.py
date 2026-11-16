#!/usr/bin/env python3
"""Re-score recorded predictions with the unchanged official evaluator. No model calls."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import re
import socket
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'benchmark/medical_scale_v06'
sys.dont_write_bytecode = True
sys.path[:0] = [str(RUNTIME), str(ROOT / 'code/analysis')]
SETTINGS = ['fresh', 'labels', 'index', 'guard', 'index_guard']
DISPLAY = ['Fresh', 'Labels', 'Labels+Index', 'Labels+Guard', 'Index+Guard']
MODELS = {'deepseek': 'deepseek-flash', 'qwen': 'Qwen3.8-27B'}


def read(path):
    with (gzip.open(path, 'rt', encoding='utf-8') if str(path).endswith('.gz') else Path(path).open(encoding='utf-8')) as f:
        return json.load(f)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def same(actual, expected, where=''):
    """Compare regenerated numeric results, allowing only float rounding noise."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), where
        for k in expected: same(actual[k], expected[k], where + '/' + k)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), where
        for i, (a, e) in enumerate(zip(actual, expected)): same(a, e, where + '/' + str(i))
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-10), (where, actual, expected)
    else:
        assert actual == expected, (where, actual, expected)


def verify_manifest():
    path = ROOT / 'manifest.json'
    if not path.exists(): return {'manifest': 'not yet generated'}
    manifest = read(path)
    for entry in manifest['files']:
        file = ROOT / entry['path']
        assert hashlib.sha256(file.read_bytes()).hexdigest() == entry['sha256'], entry['path']
    return {'files_checked': len(manifest['files']), 'passed': True}


def load_inputs(source_ids, helpers):
    actors = helpers.selected_jsonl(RUNTIME / 'confirmation15_inputs_private/actors.private.jsonl', source_ids)
    targets = helpers.selected_jsonl(RUNTIME / 'datasets/medical15_confirmation120_silver_v1/evaluator_targets.LOCAL_ONLY.private.jsonl', source_ids)
    return actors, targets


def rescore(rows, pairing, helpers, scorer):
    actors, targets = load_inputs({r['source_id'] for r in rows}, helpers)
    # Remove previously derived evaluator fields: scoring starts from actual outputs.
    derived = {'score', 'gold_transition_local', 'planned_changed', 'planned_stable',
               'construction_subset', 'source_cluster', 'modality', 'items_planned'}
    raw = [{k: v for k, v in row.items() if k not in derived} for row in rows]
    result = scorer.enrich(raw, actors, targets, pairing)
    for old, new in zip(rows, result):
        if 'score' in old: same(new['score'], old['score'], 'per-phase score')
    return result


def validate(rows, settings, helpers, scorer, initial_archive=None, pairing=None):
    from evaluate_methods import validate_matrix
    from pilot_runtime import fingerprint
    ids = sorted({r['source_id'] for r in rows})
    validate_matrix(rows, ids, ('key_mixed', 'single_definite', 'single_uncertainty', 'neutral'),
                    tuple(s+'__'+v for s in settings for v in ('baseline', 'formalization')), 'reproduction')
    by_key = {(r['source_id'], r['event_id'], r['arm']): r for r in rows}
    pairs = 0
    for row in rows:
        assert row['requests'] <= 40, ('request budget', row['source_id'], row['arm'])
        actions = (row.get('method_analysis') or {}).get('action_count_including_method_boundary')
        if actions is not None: assert actions <= 40
        if row['event_id'] == 'initial' or not row['arm'].endswith('__baseline'): continue
        other = by_key[(row['source_id'], row['event_id'], row['arm'].replace('__baseline', '__formalization'))]
        for key in ('initial_checkpoint_hash', 'pre_state_hash', 'public_hash'):
            assert row.get(key) == other.get(key), (row['source_id'], row['arm'], key)
        pairs += 1
    checkpoints = 0
    if initial_archive:
        with tarfile.open(initial_archive, 'r:gz') as tar:
            for member in tar.getmembers():
                if not member.isfile(): continue
                cp = json.load(tar.extractfile(member))
                sid = Path(member.name).parts[1]
                assert fingerprint(cp['world']) == pairing[sid]['initial_world_sha256'], sid
                checkpoints += 1
        assert checkpoints == len(ids)
    return {'sources': len(ids), 'rows': len(rows), 'matched_update_pairs': pairs,
            'initial_checkpoints_verified': checkpoints, 'request_budget': 40, 'passed': True}


def conditions_for(rows, settings, helpers, scorer):
    conditions = [helpers.condition([r for r in rows if r['event_id']=='initial'], scorer, 'initial', 'initial')]
    for s in settings:
        for v in ('baseline', 'formalization'):
            arm = s+'__'+v
            conditions.append(helpers.condition([r for r in rows if r['arm']==arm], scorer, 'all_update_events', arm))
    return conditions


def paper_tables(all_conditions, out):
    fields = ['backbone', 'setting', 'variant', 'M', 'E', 'J', 'submitted', 'planned', 'requests', 'median_seconds']
    buf = io.StringIO(); writer = csv.DictWriter(buf, fieldnames=fields); writer.writeheader()
    lookup = {}
    for backbone, conditions in all_conditions.items():
        for c in conditions:
            if c['arm']=='initial': continue
            setting, variant = c['arm'].split('__')
            lookup[(backbone, setting, variant)] = c
            writer.writerow(dict(backbone=MODELS[backbone], setting=DISPLAY[SETTINGS.index(setting)], variant=variant,
                                 **{f:c[f] for f in ('M','E','J','submitted','planned','requests')},
                                 median_seconds=c['phase_seconds']['median']))
    (out/'table1.csv').write_text(buf.getvalue(), encoding='utf-8')
    table = ['| Setting | DS M | DS +ScopeForm M | DS E | DS +ScopeForm E | DS J | DS +ScopeForm J | Qwen M | Qwen +ScopeForm M | Qwen E | Qwen +ScopeForm E | Qwen J | Qwen +ScopeForm J |',
             '|---|'+'---:|'*12]
    tex = []
    for s, label in zip(SETTINGS, DISPLAY):
        cells = []; md = []
        for backbone in MODELS:
            for v in ('baseline','formalization'):
                c = lookup[(backbone,s,v)]
                for key in ('M','E','J'):
                    value = f"{c[key]:.2f}" if key=='M' else str(c[key])
                    cells.append(value)
            for key in ('M','E','J'):
                for v in ('baseline','formalization'):
                    c=lookup[(backbone,s,v)]; md.append(f"{c[key]:.2f}" if key=='M' else str(c[key]))
        table.append('| '+label+' | '+' | '.join(md)+' |')
        tex.append(label+' & '+' & '.join(cells)+r' \\')
    (out/'table1.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    (out/'table1_rows.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    # Verify every displayed number against the manuscript's actual LaTeX table.
    original = (ROOT/'expected/merged_results.paper.tex').read_text()
    checked=0
    for line in tex:
        label, *values = line.removesuffix(r' \\').split(' & ')
        old = next(x for x in original.splitlines() if x.startswith(label+' & '))
        old = re.sub(r'\\textbf\{([^}]+)\}',r'\1',old).removesuffix(r' \\')
        assert old.split(' & ')[1:]==values, label
        checked += len(values)
    return checked


def repair_audit(out):
    audit = module('recorded_repair_audit', ROOT/'code/analysis/analyze_repair_traces.py')
    trace_root = out/'repair_inputs'
    archive = ROOT/'artifacts/deepseek/repair_inputs.tar.gz'
    print('Reconstructing 2,626 recorded repair trajectories...', flush=True)
    trace_root.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz') as tar:
        for m in tar.getmembers():
            if m.isfile():
                dest=(trace_root/m.name).resolve()
                dest.relative_to(trace_root.resolve())
                dest.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(m) as src, dest.open('wb') as dst: dst.write(src.read())
    audit.RUN=trace_root
    audit.OUT=out/'repair_audit'
    audit.SCORED=out/'deepseek/scored_rows.json'
    audit.OUT.mkdir(parents=True,exist_ok=True)
    audit.main()
    actual=read(audit.OUT/'summary.json');expected=read(ROOT/'expected/repair_audit.json')
    for key in ('observed_first_pairs','unavailable_first_pairs','unavailable_reasons','first_attempt',
                'first_attempt_by_setting','later_guard_attempts_separate'):
        same(actual[key],expected[key],key)
    return {'first_pairs':actual['observed_first_pairs'], 'passed':True,
            'counts':actual['first_attempt']['counts']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'output/reproduced')
    parser.add_argument('--skip-repair-audit',action='store_true',help='Only rescore main results, intervals and costs.')
    parser.add_argument('--run-dir',type=Path,help='Score a NEW run from run_experiment.py instead of the recorded paper runs.')
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    # Enforce that the reproduction path cannot issue a network request.
    def no_network(*a,**kw): raise RuntimeError('Offline reproduction does not use the network')
    socket.create_connection=no_network;socket.socket.connect=no_network
    import score_study15_local as scorer
    import evaluate_five_settings as paired
    helpers=module('original_score_helpers',ROOT/'code/baseline/score.py')
    if args.run_dir:
        run=args.run_dir.resolve();config=read(run/'config.json')
        rows=rescore(read(run/'main/results.json'),read(run/'main/pairing_audit.json'),helpers,scorer)
        settings=config['selected_settings']
        verification=validate(rows,settings,helpers,scorer)
        write(out/'scored_rows.json',rows)
        write(out/'metrics.json',conditions_for(rows,settings,helpers,scorer))
        paired.SETTINGS={s:s for s in settings}
        write(out/'paired_effects.json',paired.paired_effects(rows,scorer,helpers))
        write(out/'verification.json',verification)
        print(json.dumps(verification));return
    report={'offline':True,'model_calls':0,'manifest':verify_manifest(),'backbones':{}}
    all_conditions={}
    for backbone in MODELS:
        print('Re-scoring '+MODELS[backbone]+' (4,920 phases)...',flush=True)
        folder=ROOT/'artifacts'/backbone;pairing=read(folder/'pairing_audit.json')
        rows=rescore(read(folder/'scored_rows.recorded.json.gz'),pairing,helpers,scorer)
        verification=validate(rows,SETTINGS,helpers,scorer,folder/'shared_initials.tar.gz',pairing)
        dest=out/backbone
        write(dest/'scored_rows.json',rows)
        conditions=conditions_for(rows,SETTINGS,helpers,scorer);all_conditions[backbone]=conditions
        expected=read(ROOT/'expected'/f'{backbone}_metrics.json')
        expected=expected['core']['conditions'] if backbone=='deepseek' else expected['conditions']
        for c in conditions:
            previous=next(e for e in expected if e['event']==c['event'] and e['arm']==c['arm'])
            same(c, {k:previous[k] for k in c}, backbone+'/'+c['arm'])
        intervals=paired.paired_effects(rows,scorer,helpers)
        same(intervals,read(ROOT/'expected'/f'{backbone}_paired.json'),backbone+'/paired')
        write(dest/'metrics.json',conditions);write(dest/'paired_effects.json',intervals)
        costs={v:helpers.condition([r for r in rows if r['event_id']!='initial' and r['variant']==v],scorer,'all_settings',v)
               for v in ('baseline','formalization')}
        write(dest/'aggregate_costs.json',costs)
        report['backbones'][backbone]={**verification,'per_phase_scores_match':True,'summary_metrics_match':True,'paired_intervals_match':True}
    report['paper_numeric_cells_verified']=paper_tables(all_conditions,out)
    if not args.skip_repair_audit:report['repair_audit']=repair_audit(out)
    report['passed']=True
    write(out/'verification.json',report)
    print(json.dumps({'passed':True,'model_calls':0,'phases_rescored':9840,'paper_table_cells':report['paper_numeric_cells_verified'],'output':str(out)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
