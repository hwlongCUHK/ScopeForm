"""Read-only verification and append-only audit outputs for confirmation13.

May run while authoring proceeds. Does not invoke a model or alter a seal.
"""
from collections import Counter
import argparse
import json

import confirmation13 as c


def oracle_replay(bundle):
    from workflow11_runtime import World, fork, export_observations
    from validate_workflow11_offline import complete
    from score_workflow11_local import score_row
    from pilot_runtime import fingerprint
    initial = bundle['initial']
    base = complete(World(initial['controller']), initial['items'])
    base_row = {'submitted': True, 'event_id': 'initial', **export_observations(base)}
    score = score_row(base_row, initial['items'], initial['controller'], initial['items'], base_row)
    if not score['strict']:
        raise ValueError('Known-answer initial mechanical replay failed')
    phases = 1
    for event in bundle['events']:
        for arm in ('evidence_constraint', 'labels_only', 'fresh'):
            world = fork(base, event['controller'], event['event_id'], arm)
            if fingerprint(world.state) != fingerprint(base.state):
                raise ValueError('Mechanical branch did not clone its actual base')
            complete(world, event['items'])
            row = {'submitted': True, 'event_id': event['event_id'], **export_observations(world)}
            score = score_row(row, event['items'], event['controller'], initial['items'], base_row)
            if not score['strict']:
                raise ValueError('Known-answer event mechanical replay failed')
            phases += 1
    return phases


def verify(oracle=False):
    c.verify_plan()
    snapshot = c.WORK / 'code_snapshot'
    for name, expected in c.read(c.WORK / 'protocol.json')['code_sha256'].items():
        raw = (c.ROOT / name).read_bytes()
        if c.digest(c.ROOT / name) != expected:
            raise ValueError('Builder hash mismatch')
        c.frozen(snapshot / name, raw)
    selected = c.load_reserved()
    directions = c.read(c.WORK / 'direction_assignment.LOCAL_ONLY.json')
    true_directions = Counter()
    status = Counter()
    kinds = Counter()
    independent_checks = 0
    source_spans = 0
    mechanical_phases = 0
    for source in selected:
        sid = source['source_id']
        path = c.WORK / 'sources' / sid / 'result.json'
        if not path.exists():
            status['pending'] += 1
            continue
        row = c.read(path)
        status['qualified' if row['machine_valid'] else 'rejected'] += 1
        if not row['machine_valid']:
            continue
        items = c.read(path.parent / 'accepted_spec.private.json')
        okay, errors = c.check_items(items, source)
        if not okay:
            raise ValueError('Accepted specification failed replay: ' + sid)
        kinds[row['construction_kind']] += 1
        blocks = {b['anchor_id']: b['text'] for d in source['documents'] for b in d['blocks']}
        for item in items:
            for reference in item['references']:
                if reference['quote'] not in blocks[reference['anchor_id']]:
                    raise ValueError('Source quotation not exact')
                source_spans += 1
            for state in item['state_values']:
                c.independent_label(item, state)
                independent_checks += 1
        bundle = c.compile_source(source, items, directions[sid])
        c.verify_compiled(source, bundle)
        if oracle:
            mechanical_phases += oracle_replay(bundle)
        base = {g['item_id']: g['label'] for g in bundle['initial']['items']}
        for event in bundle['events']:
            for item in event['items']:
                if base[item['item_id']] != item['label']:
                    true_directions[(event['event_id'], base[item['item_id']], item['label'])] += 1
    if c.DATASET.exists():
        c.verify_tree(c.DATASET)
    report = {'source_inputs_frozen': 120, 'source_seal_valid': True,
        'reserved_source_status': dict(status), 'qualified_subsets': dict(kinds),
        'independent_rule_state_checks': independent_checks, 'original_quote_roundtrips': source_spans,
        'verified_qualified_direction_counts': [{'event': event,'from': a,'to': b,'count': n}
                                               for (event,a,b),n in sorted(true_directions.items())],
        'tasks_frozen': c.DATASET.exists(), 'solver_calls': 0,
        'mechanical_oracle_phases_not_model_results': mechanical_phases,
        'clinical_validation': False, 'old_artifacts_unchanged': True}
    print(json.dumps(report, ensure_ascii=True))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--oracle', action='store_true')
    args = parser.parse_args()
    verify(args.oracle)
