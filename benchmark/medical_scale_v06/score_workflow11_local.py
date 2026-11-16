"""Independent LOCAL scoring of prediction-only workflow exports.

Never imported by the remote actor or included in its deployment archive.
"""
import argparse
import collections
import json
from pathlib import Path
import random
import statistics
from run_generation import ROOT, read, dump, digest
from pilot_runtime import fingerprint
from workflow11_runtime import World

EVENTS = ('key_mixed', 'single_definite', 'single_uncertainty', 'neutral')
ARMS = ('evidence_constraint', 'labels_only', 'fresh')
INPUTS = ROOT / 'workflow11_inputs_private'


def shape(pred, expected):
    return (isinstance(pred, list) and len(pred) == len(expected) and
            len({x.get('item_id') for x in pred}) == len(expected) and
            {x.get('item_id') for x in pred} == {x['item_id'] for x in expected})


def score_row(row, expected, controller, initial_expected, base_row):
    if not row['submitted']:
        return None
    pred = row['prediction']['items']
    by = {p['item_id']: p for p in pred}
    stage = 'initial' if row['event_id'] in ('initial', 'neutral') else 'updated'
    w = World(controller, stage)
    docs = w.documents()
    active = {b['anchor_id'] for d in docs for b in d['blocks']}
    observed = set(row['observed'])
    if not observed <= active or not set(row['seen_documents']) <= {d['document_id'] for d in docs}:
        raise ValueError('Exported acquisition is not current')
    initial = {i['item_id']: i['label'] for i in initial_expected}
    base = {i['item_id']: i['label'] for i in (base_row.get('prediction') or {}).get('items', [])}
    rows = []
    for g in expected:
        p = by.get(g['item_id'], {})
        refs = p.get('evidence_ids', [])
        acquired = len(refs) == len(set(refs)) and set(refs) <= observed
        enough = (any(set(group) <= set(refs) for group in g['sufficient_evidence_groups'])
                  if g['label'] != 'not_determined' else
                  {d['document_id'] for d in docs} <= set(row['seen_documents']))
        changed = initial[g['item_id']] != g['label']
        previously_correct = base.get(g['item_id']) == initial[g['item_id']]
        rows.append({'item_id': g['item_id'], 'gold': g['label'], 'prediction': p.get('label'),
                     'content': p.get('label') == g['label'], 'evidence': bool(acquired and enough),
                     'changed': changed, 'stable': not changed,
                     'definite_flip': changed and 'not_determined' not in (initial[g['item_id']], g['label']),
                     'to_unknown': changed and g['label'] == 'not_determined',
                     'from_unknown': changed and initial[g['item_id']] == 'not_determined',
                     'protected_initial_correct': not changed and previously_correct,
                     'initial_wrong_stable': not changed and not previously_correct,
                     'old_prediction': base.get(g['item_id']), 'initial_gold': initial[g['item_id']]})
    shaped = shape(pred, expected)
    content = shaped and all(r['content'] for r in rows)
    evidence = shaped and all(r['evidence'] for r in rows)
    actual_state = None
    readback = None
    if controller['environment'] == 'E3':
        if row['action_expected'] == 'saved':
            actual_state = (row['state']['summary'] is not None and
                {i['item_id']: i['label'] for i in row['state']['summary']} == {g['item_id']: g['label'] for g in expected} and
                row['state']['protected'] == row['phase_start_state']['protected'])
        else:
            actual_state = row['state'] == row['phase_start_state']
        readback = row['last_readback'] == row['state']
    else:
        actual_state = row['state'] == row['phase_start_state']
    action = row['prediction']['action_status'] == row['action_expected']
    unauthorized = row['tool_errors'].count('not_authorized')
    result = {'content': content, 'structure': shaped, 'evidence': evidence,
              'action_report': action, 'actual_state': actual_state, 'readback': readback,
              'strict': bool(content and evidence and action and actual_state and readback is not False and unauthorized == 0),
              'items': rows, 'unauthorized_attempts': unauthorized}
    for name in ('changed', 'stable', 'definite_flip', 'to_unknown', 'from_unknown', 'protected_initial_correct', 'initial_wrong_stable'):
        sub = [r for r in rows if r[name]]
        result[name] = {'correct': sum(r['content'] for r in sub), 'total': len(sub)}
    return result


def ratio(n, d):
    return n / d if d else None


def source_values(rows, category='all', metric='content', coverage_adjusted=False):
    groups = collections.defaultdict(list)
    for r in rows:
        if category == 'new112' and r['adaptive_calibration_reused']:
            continue
        if category == 'old8' and not r['adaptive_calibration_reused']:
            continue
        s = r['score']
        if metric == 'content':
            n = sum(i['content'] for i in s['items']) if s else 0
            d = r['items_planned'] if s or coverage_adjusted else 0
        elif metric in ('changed', 'stable', 'definite_flip', 'to_unknown', 'from_unknown', 'protected_initial_correct', 'initial_wrong_stable'):
            n = s[metric]['correct'] if s else 0
            d = s[metric]['total'] if s else 0
            if not s and coverage_adjusted:
                d = r.get('planned_' + metric, 0)
        else:
            n = int(s[metric] is True) if s else 0
            d = int(s is not None and s[metric] is not None)
            if coverage_adjusted and not s:
                d = int(metric not in ('readback',) or r['environment'] == 'E3')
        if d:
            groups[r['source_id']].append((n, d))
    return {sid: sum(n for n, d in values) / sum(d for n, d in values) for sid, values in groups.items()}


def mean_ci(values, seed=20261021):
    vals = list(values.values())
    if not vals:
        return {'macro': None, 'sources': 0, 'ci95': None}
    rng = random.Random(seed)
    boots = sorted(statistics.fmean(rng.choices(vals, k=len(vals))) for _ in range(2000))
    return {'macro': statistics.fmean(vals), 'sources': len(vals), 'ci95': [boots[49], boots[1949]]}


def aggregate(rows):
    done = [r for r in rows if r['submitted']]
    result = {'planned': len(rows), 'submitted': len(done), 'statuses': dict(collections.Counter(r['status'] for r in rows)),
        'item_correct': sum(sum(i['content'] for i in r['score']['items']) for r in done),
        'items_submitted': sum(r['items_planned'] for r in done), 'items_planned': sum(r['items_planned'] for r in rows),
        'whole_packet_correct': sum(r['score']['content'] for r in done),
        'evidence_packets_correct': sum(r['score']['evidence'] for r in done),
        'strict_packets_correct': sum(r['score']['strict'] for r in done),
        'actual_state_correct': sum(r['score']['actual_state'] is True for r in done),
        'readback_correct': sum(r['score']['readback'] is True for r in done),
        'readback_applicable': sum(r['score']['readback'] is not None for r in done),
        'tool_calls': sum(r.get('tool_calls', 0) for r in rows), 'requests': sum(r['requests'] for r in rows),
        'guard_rejections': sum(r.get('guard_rejections', 0) for r in rows),
        'unauthorized_attempts': sum(r['score']['unauthorized_attempts'] for r in done)}
    for metric in ('changed', 'stable', 'definite_flip', 'to_unknown', 'from_unknown', 'protected_initial_correct', 'initial_wrong_stable'):
        result[metric] = {'correct': sum(r['score'][metric]['correct'] for r in done),
                          'total': sum(r['score'][metric]['total'] for r in done),
                          'planned': sum(r['planned_' + metric] for r in rows)}
    result['by_split'] = {split: {metric + ('_coverage_adjusted' if cov else ''):
        mean_ci(source_values(rows, split, metric, cov)) for metric in ('content', 'changed', 'stable', 'protected_initial_correct', 'evidence', 'strict') for cov in (False, True)}
        for split in ('all', 'new112', 'old8')}
    return result


def label_references(rows, targets, bases):
    """Supplementary zero-inference label references, NOT agent submissions."""
    output = []
    for event in EVENTS:
        chosen = [r for r in rows if r['event_id'] == event and r['arm'] == 'fresh']
        for reference in ('copy_actual_initial_labels', 'always_supported', 'always_refuted', 'always_unknown'):
            values = []
            for r in chosen:
                sid = r['source_id']
                t = targets[sid]
                expected = next(e['items'] for e in t['events'] if e['event_id'] == event)
                initial = {i['item_id']: i['label'] for i in t['initial']}
                base = bases[sid]
                previous = {i['item_id']: i['label'] for i in (base.get('prediction') or {}).get('items', [])}
                available = base['submitted'] if reference == 'copy_actual_initial_labels' else True
                constant = {'always_supported': 'supported', 'always_refuted': 'refuted', 'always_unknown': 'not_determined'}.get(reference)
                correct = [available and (previous.get(g['item_id']) if constant is None else constant) == g['label'] for g in expected]
                changed = [i for i, g in enumerate(expected) if g['label'] != initial[g['item_id']]]
                values.append({'source_id': sid, 'old8': t['adaptive_calibration_reused'], 'available': available,
                    'content': sum(correct) / len(expected), 'changed': sum(correct[i] for i in changed) / len(changed) if changed else None})
            for split in ('all', 'new112', 'old8'):
                vs = [v for v in values if split == 'all' or (split == 'old8') == v['old8']]
                available = [v for v in vs if v['available']]
                changed = [v for v in available if v['changed'] is not None]
                output.append({'event_id': event, 'reference': reference, 'split': split,
                    'sources_planned': len(vs), 'sources_with_reference': len(available),
                    'content_source_macro': statistics.fmean(v['content'] for v in available) if available else None,
                    'content_source_macro_coverage_adjusted': statistics.fmean(v['content'] for v in vs) if vs else None,
                    'changed_source_macro': statistics.fmean(v['changed'] for v in changed) if changed else None,
                    'not_an_agent_submission_or_evidence_or_state_score': True})
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--downloaded', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--attempt', choices=('01', 'xml02'), default='01')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve completed or partial readout')
    protocol = read(ROOT / 'workflow11_protocol.json')
    local_manifest = read(INPUTS / 'local_evaluator_manifest.json')
    if digest(INPUTS / 'evaluator_targets.LOCAL_ONLY.private.jsonl') != local_manifest['target_sha256']:
        raise ValueError('Evaluator target changed')
    actors = {r['source_id']: r for r in map(json.loads, (INPUTS / 'actors.private.jsonl').read_text(encoding='utf-8').splitlines())}
    targets = {r['source_id']: r for r in map(json.loads, (INPUTS / 'evaluator_targets.LOCAL_ONLY.private.jsonl').read_text(encoding='utf-8').splitlines())}
    reports, all_rows, audits = [], [], []
    for model, name in [('Qwen3.5-9B', 'workflow11_q9b_' + a.attempt), ('Qwen3.8-27B', 'workflow11_q27b_' + a.attempt)]:
        folder = a.downloaded / 'runs' / name
        if not (folder / 'complete.json').exists():
            raise ValueError('Wait for both full runs; do not publish partial readout')
        plan = read(folder / 'plan.json')
        if plan['protocol_sha256'] != digest(ROOT / 'workflow11_protocol.json') or plan['gold_loaded_on_server']:
            raise ValueError('Protocol/isolation mismatch')
        if a.attempt == 'xml02':
            if plan.get('tool_call_parser') != 'qwen3_coder' or plan.get('transport_erratum_sha256') != digest(ROOT / 'workflow11_transport_erratum.json'):
                raise ValueError('Corrected parser provenance mismatch')
            port = 18510 if model == 'Qwen3.5-9B' else 18511
            check = read(a.downloaded / 'runs/workflow11_queue_xml02' / ('preflight_' + str(port) + '.json'))
            if not check['native_auto_tool_and_nested_arguments_pass'] or not check['synthetic_nonmedical']:
                raise ValueError('Native auto-tool interface not validated')
        raw = read(folder / 'results.json')
        if len(raw) != 1560 or len({(r['source_id'], r['event_id'], r['arm']) for r in raw}) != 1560:
            raise ValueError('Incomplete/duplicate phase matrix')
        bases = {r['source_id']: r for r in raw if r['event_id'] == 'initial'}
        for row in raw:
            sid, event = row['source_id'], row['event_id']
            target, actor = targets[sid], actors[sid]
            expected = target['initial'] if event == 'initial' else next(e['items'] for e in target['events'] if e['event_id'] == event)
            controller = actor['initial'] if event == 'initial' else next(e['controller'] for e in actor['events'] if e['event_id'] == event)
            if row.get('public_hash') and row['public_hash'] != fingerprint(controller):
                raise ValueError('Actor/evaluator public state mismatch')
            old = {g['item_id']: g['label'] for g in target['initial']}
            previous = {i['item_id']: i['label'] for i in (bases[sid].get('prediction') or {}).get('items', [])}
            masks = {'changed': lambda g: g['label'] != old[g['item_id']],
                     'stable': lambda g: g['label'] == old[g['item_id']],
                     'definite_flip': lambda g: g['label'] != old[g['item_id']] and 'not_determined' not in (g['label'], old[g['item_id']]),
                     'to_unknown': lambda g: g['label'] != old[g['item_id']] and g['label'] == 'not_determined',
                     'from_unknown': lambda g: g['label'] != old[g['item_id']] and old[g['item_id']] == 'not_determined',
                     'protected_initial_correct': lambda g: g['label'] == old[g['item_id']] and previous.get(g['item_id']) == old[g['item_id']],
                     'initial_wrong_stable': lambda g: g['label'] == old[g['item_id']] and previous.get(g['item_id']) != old[g['item_id']]}
            row.update(model=model, items_planned=len(expected), source_cluster=target['source_cluster'],
                adaptive_calibration_reused=target['adaptive_calibration_reused'], environment=target['environment'],
                construction_subset=target['construction_subset'],
                **{'planned_' + k: sum(fn(g) for g in expected) for k, fn in masks.items()})
            row['score'] = score_row(row, expected, controller, target['initial'], bases[sid])
        model_audits = []
        for sid, b in bases.items():
            branches = [r for r in raw if r['source_id'] == sid and r['event_id'] != 'initial']
            if len(branches) != 12:
                raise ValueError('Twelve branches required for each source')
            if b['submitted']:
                hashes = {r.get('pre_state_hash') for r in branches}
                ckhashes = {r.get('initial_checkpoint_hash') for r in branches}
                if hashes != {fingerprint(b['state'])} or len(ckhashes) != 1 or None in ckhashes:
                    raise ValueError('Branches did not use same actual initial checkpoint')
                good = True
            else:
                good = all(r['status'] == 'blocked_actual_initial_not_submitted' for r in branches)
            model_audits.append({'source_id': sid, 'base_submitted': b['submitted'], 'all_branches_same_actual_start_or_blocked': good})
        summaries = [{'event_id': 'initial', 'arm': 'initial', **aggregate([r for r in raw if r['event_id'] == 'initial'])}]
        summaries += [{'event_id': e, 'arm': arm, **aggregate([r for r in raw if r['event_id'] == e and r['arm'] == arm])}
                      for e in EVENTS for arm in ARMS]
        paired = []
        for e in EVENTS:
            for split in ('all', 'new112'):
                for metric in ('content', 'changed', 'stable', 'protected_initial_correct', 'strict'):
                    for left, right in [('evidence_constraint', 'labels_only'), ('evidence_constraint', 'fresh'), ('labels_only', 'fresh')]:
                        lv = source_values([r for r in raw if r['event_id'] == e and r['arm'] == left], split, metric, True)
                        rv = source_values([r for r in raw if r['event_id'] == e and r['arm'] == right], split, metric, True)
                        common = sorted(lv.keys() & rv.keys())
                        paired.append({'event_id': e, 'split': split, 'metric': metric, 'left': left, 'right': right,
                                       'coverage_adjusted': True, **mean_ci({sid: lv[sid] - rv[sid] for sid in common})})
        reports.append({'model': model, 'run': name, 'complete': read(folder / 'complete.json'), 'by_condition': summaries,
                        'supplementary_zero_inference_label_references': label_references(raw, targets, bases),
                        'paired_source_cluster_differences': paired})
        audits.append({'model': model, 'sources': model_audits})
        all_rows.extend(raw)
    result = {'kind': 'actual_complete_workflow11_local_scoring', 'protocol': protocol,
        'dataset_integrity_sha256': protocol['dataset_integrity_sha256'], 'models': reports,
        'planned_phases': 3120, 'recorded_phases': len(all_rows),
        'submitted_phases': sum(r['submitted'] for r in all_rows), 'evaluator_target_transferred': False,
        'clinical_gold_expert_validated': False, 'locked_test': False, 'independent_sources': 120,
        'attempt': a.attempt}
    if a.attempt == 'xml02':
        result['transport_erratum'] = read(a.downloaded / 'workflow11_transport_erratum.json')
        result['resource_amendment'] = read(a.downloaded / 'workflow11_resource_amendment.json')
        result['actual_deployments'] = read(a.downloaded / 'resource/deployment.json')
        result['broken_interface_attempt_not_content_score'] = read(a.downloaded / 'resource/broken_interface_attempt.json')
        fault = read(a.downloaded / 'runs/workflow11_queue_xml02/technical_fault_audit.json')
        if fault['dataset_integrity_sha256'] != protocol['dataset_integrity_sha256'] or fault['actor_bundle_sha256'] != digest(INPUTS / 'actors.private.jsonl'):
            raise ValueError('Technical-fault audit belongs to another dataset/input')
        result['technical_fault_audit'] = fault
    dump(a.output / 'results.json', result)
    dump(a.output / 'scored_rows.private.json', all_rows)
    dump(a.output / 'checkpoint_pairing_audit.json', audits)
    dump(a.output / 'verification.json', {'all_3120_accounted': len(all_rows) == 3120,
        'all_branch_starts_verified': all(all(s['all_branches_same_actual_start_or_blocked'] for s in m['sources']) for m in audits),
        'evaluator_target_transferred': False, 'protocol_sha256': digest(ROOT / 'workflow11_protocol.json'),
        'dataset_integrity_sha256': protocol['dataset_integrity_sha256'],
        'scorer_sha256': digest(Path(__file__)), 'local_target_sha256': local_manifest['target_sha256'],
        'attempt': a.attempt, 'synthetic_native_tool_parser_preflights_passed': a.attempt == 'xml02'})
    print(json.dumps({'event': 'workflow11_complete_scored_locally', 'phases': len(all_rows),
                      'submitted': result['submitted_phases'], 'models': len(reports)}))


if __name__ == '__main__':
    main()
