#!/usr/bin/env python3
"""Score a local agent run with the unchanged benchmark evaluator. No model calls."""
from __future__ import annotations
import argparse
import gzip
import importlib.util
import json
import math
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'benchmark/medical_scale_v06'
sys.dont_write_bytecode = True
sys.path[:0] = [str(RUNTIME), str(ROOT / 'code/analysis')]



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


def validate(rows, settings):
    from evaluate_methods import validate_matrix
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
    return {'sources': len(ids), 'rows': len(rows), 'matched_update_pairs': pairs,
            'request_budget': 40, 'passed': True}


def conditions_for(rows, settings, helpers, scorer):
    conditions = [helpers.condition([r for r in rows if r['event_id']=='initial'], scorer, 'initial', 'initial')]
    for s in settings:
        for v in ('baseline', 'formalization'):
            arm = s+'__'+v
            conditions.append(helpers.condition([r for r in rows if r['arm']==arm], scorer, 'all_update_events', arm))
    return conditions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True,help='Run directory created by run_experiment.py.')
    parser.add_argument('--out',type=Path,default=ROOT/'output/reproduced')
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    # Offline scoring cannot issue a network request.
    def no_network(*a,**kw): raise RuntimeError('Offline scoring does not use the network')
    socket.create_connection=no_network;socket.socket.connect=no_network
    import score_study15_local as scorer
    import evaluate_five_settings as paired
    helpers=module('original_score_helpers',ROOT/'code/baseline/score.py')
    run=args.run_dir.resolve();config=read(run/'config.json')
    rows=rescore(read(run/'main/results.json'),read(run/'main/pairing_audit.json'),helpers,scorer)
    settings=config['selected_settings']
    verification=validate(rows,settings)
    write(out/'scored_rows.json',rows)
    write(out/'metrics.json',conditions_for(rows,settings,helpers,scorer))
    paired.SETTINGS={s:s for s in settings}
    write(out/'paired_effects.json',paired.paired_effects(rows,scorer,helpers))
    write(out/'verification.json',verification)
    print(json.dumps(verification))

if __name__=='__main__':main()
