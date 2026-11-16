"""Local mechanical oracle replay, not model or medical-semantic validation."""
import json
from run_generation import ROOT, read, dump, digest
from workflow11_runtime import World, fork, export_observations
from score_workflow11_local import score_row
from pilot_runtime import fingerprint


def complete(w, items):
    for d in w.documents():
        w.call('read_source', {'document_id': d['document_id']})
    pred = [{'item_id': g['item_id'], 'label': g['label'], 'evidence_ids':
             list(g['sufficient_evidence_groups'][0]) if g['label'] != 'not_determined' else []} for g in items]
    state = w.call('read_state', {})
    if w.permission():
        receipt = w.call('save_summary', {'items': pred, 'expected_version': state['version'],
                         'idempotency_key': 'oracle_' + w.event_kind})
        if not receipt.get('ok'):
            raise ValueError('Oracle save rejected')
        w.call('read_state', {})
    accepted = w.call('submit_answer', {'items': pred, 'action_status': w.action()})
    if not accepted.get('accepted'):
        raise ValueError('Oracle submit rejected')
    return w


def main():
    folder = ROOT / 'workflow11_inputs_private'
    actors = [json.loads(line) for line in (folder / 'actors.private.jsonl').read_text(encoding='utf-8').splitlines()]
    targets = {t['source_id']: t for t in map(json.loads, (folder / 'evaluator_targets.LOCAL_ONLY.private.jsonl').read_text(encoding='utf-8').splitlines())}
    cases, passed = 0, 0
    for actor in actors:
        t = targets[actor['source_id']]
        b = complete(World(actor['initial']), t['initial'])
        br = {'submitted': True, 'event_id': 'initial', **export_observations(b)}
        score = score_row(br, t['initial'], actor['initial'], t['initial'], br)
        cases += 1
        passed += score['strict']
        for event in actor['events']:
            expected = next(e['items'] for e in t['events'] if e['event_id'] == event['event_id'])
            for arm in ('evidence_constraint', 'labels_only', 'fresh'):
                w = fork(b, event['controller'], event['event_id'], arm)
                if fingerprint(w.state) != fingerprint(b.state):
                    raise ValueError('Oracle branch base changed')
                complete(w, expected)
                row = {'submitted': True, 'event_id': event['event_id'], **export_observations(w)}
                s = score_row(row, expected, event['controller'], t['initial'], br)
                cases += 1
                passed += s['strict']
    report = {'kind': 'offline_local_mechanical_oracle_NOT_model_result', 'source_clusters': len(actors),
              'unique_workflow_phases': cases, 'strict_mechanical_pass': passed, 'all_passed': passed == cases == 1560,
              'model_calls': 0, 'not_clinical_semantic_verification': True,
              'protocol_sha256': digest(ROOT / 'workflow11_protocol.json'), 'validator_sha256': digest(__file__ and __import__('pathlib').Path(__file__))}
    dump(ROOT.parent / 'output/medical11_workflow_20261021/offline_mechanical_validation.json', report)
    print(json.dumps(report))
    if not report['all_passed']:
        raise ValueError('Mechanical protocol validation failed')


if __name__ == '__main__':
    main()
