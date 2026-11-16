"""Frozen evaluation utilities; actor payloads never contain evaluator targets."""
import collections
import hashlib
import json
from pathlib import Path
import re
from run_generation import ROOT, read, dump, digest

DS = ROOT / 'datasets/medical09_dev120_silver_v1'
INPUTS = ROOT / 'evaluation_inputs_private10'
LABELS = ('supported', 'refuted', 'not_determined')
CODE = ('baseline10_common.py', 'run_baseline10.py', 'service_baseline10.py',
        'queue_baseline10.py', 'run_generation.py', 'token_count.py')


def slug(model):
    return re.sub('[^a-z0-9]+', '_', model.lower()).strip('_')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def verify(check_dataset_files=False):
    p = read(ROOT / 'baseline10_protocol.json')
    if digest(DS / 'build_integrity.json') != p['dataset_integrity_sha256']:
        raise ValueError('Frozen120-source dataset integrity changed')
    if check_dataset_files:
        for relative, sha in read(DS / 'build_integrity.json').items():
            if digest(DS / relative) != sha:
                raise ValueError('Frozen data changed: ' + relative)
    inputs = read(INPUTS / 'actor_manifest.json')
    if inputs['dataset_integrity_sha256'] != p['dataset_integrity_sha256']:
        raise ValueError('Inputs belong to a different dataset')
    for name, sha in inputs['files'].items():
        if digest(INPUTS / name) != sha:
            raise ValueError('Matched input changed: ' + name)
    return p


def load_views(with_targets=True):
    # The caller must not serialize targets into an actor request.
    targets = {t['case_id']: t for t in jsonl(INPUTS / 'evaluator_targets.private.jsonl')} if with_targets else None
    views = {v: jsonl(INPUTS / (v + '.private.jsonl'))
             for v in ('full_context', 'source_withheld', 'stem_only')}
    for v, rows in views.items():
        if len(rows) != 404 or len({r['case_id'] for r in rows}) != 404 or (targets is not None and {r['case_id'] for r in rows} != set(targets)):
            raise ValueError('Incomplete matched inputs')
        if targets is not None:
            for r in rows:
                if {c['item_id'] for c in r['input']['claims']} != {i['item_id'] for i in targets[r['case_id']]['items']}:
                    raise ValueError('Claim/target alignment failed')
    return views, targets


def context_for(model):
    conf = read(Path('/data/zhanchen/models') / model / 'config.json')
    configured = 32768 if conf['model_type'] in ('qwen3', 'glm4') else 65536
    return min(configured, conf.get('max_position_embeddings', conf.get('text_config', {}).get('max_position_embeddings', configured)))


def payload(record):
    # Explicit allowlist, not exclusion of a few private keys.
    return {'system_prompt': record['system_prompt'], 'input': record['input']}


def grouped_jobs(views, mode):
    jobs = {}
    for condition, records in views.items():
        for record in records:
            actor = payload(record)
            parts = [actor] if mode == 'generative' else [
                {'system_prompt': actor['system_prompt'],
                 'input': {**{k: v for k, v in actor['input'].items() if k != 'claims'}, 'claims': [claim]}}
                for claim in actor['input']['claims']]
            for public in parts:
                key = condition + '__' + fingerprint(public)
                if key not in jobs:
                    jobs[key] = {'key': key, 'condition': condition, 'actor': public, 'targets': []}
                ids = [c['item_id'] for c in public['input']['claims']]
                jobs[key]['targets'].append({'case_id': record['case_id'], 'item_ids': ids})
    return sorted(jobs.values(), key=lambda j: fingerprint(j['key']))


def expand_predictions(jobs, rows, targets):
    predictions = []
    for job in jobs:
        row = rows.get(job['key'])
        if row is None:
            continue
        for ref in job['targets']:
            target = targets[ref['case_id']]
            for item in target['items']:
                if item['item_id'] not in ref['item_ids']:
                    continue
                pred = row.get('predictions', {}).get(item['item_id']) if row.get('submitted') else None
                predictions.append({
                    'case_id': ref['case_id'], 'source_id': target['source_id'],
                    'source_cluster': target['source_cluster'], 'case_kind': target['case_kind'],
                    'adaptive_calibration_reused': target['adaptive_calibration_reused'],
                    'construction_subset': target['construction_subset'], 'condition': job['condition'],
                    'item_id': item['item_id'], 'gold': item['label'], 'prediction': pred,
                    'stable_insufficiency': item['stable_insufficiency'],
                    'new_primary_content_rule': item['new_primary_content_rule'],
                    'submitted': bool(row.get('submitted')), 'correct': pred == item['label'],
                    'job_key': job['key'], 'status': row['status']})
    return predictions


def summarize(rows, jobs, targets, final=False):
    expanded = expand_predictions(jobs, rows, targets)
    subsets = {
        'all_primary_warning_majority_unknown': lambda x: x['case_kind'] == 'primary_three_state',
        'new112_balanced': lambda x: x['case_kind'] == 'primary_three_state' and not x['adaptive_calibration_reused'] and not x['stable_insufficiency'],
        'new_primary_structural_balanced': lambda x: x['case_kind'] == 'primary_three_state' and x['new_primary_content_rule'] and not x['stable_insufficiency'],
        'persistent_insufficiency': lambda x: x['case_kind'] == 'primary_three_state' and x['stable_insufficiency'],
        'partial_control': lambda x: x['case_kind'] == 'partial_information_control',
        'old8_adaptive': lambda x: x['adaptive_calibration_reused'],
        'new_free_graph_balanced': lambda x: x['case_kind'] == 'primary_three_state' and not x['adaptive_calibration_reused'] and not x['stable_insufficiency'] and 'quantitative' not in x['construction_subset'],
        'new_quantitative_balanced': lambda x: x['case_kind'] == 'primary_three_state' and not x['adaptive_calibration_reused'] and not x['stable_insufficiency'] and 'quantitative' in x['construction_subset'],
    }
    report = {}
    for condition in ('full_context', 'source_withheld', 'stem_only'):
        report[condition] = {}
        for name, select in subsets.items():
            rr = [r for r in expanded if r['condition'] == condition and select(r)]
            ss = [r for r in rr if r['submitted']]
            groups = collections.defaultdict(list)
            for r in ss:
                groups[r['source_cluster']].append(r['correct'])
            values = [sum(v) / len(v) for v in groups.values()]
            d = {'recorded_judgments': len(rr), 'submitted_judgments': len(ss),
                 'correct': sum(r['correct'] for r in ss),
                 'submitted_accuracy': sum(r['correct'] for r in ss) / len(ss) if ss else None,
                 'coverage_adjusted_accuracy': sum(r['correct'] for r in ss) / len(rr) if rr else None,
                 'source_clusters_submitted': len(values),
                 'source_macro_accuracy': sum(values) / len(values) if values else None,
                 'fixed_always_unknown_accuracy': sum(r['gold'] == 'not_determined' for r in rr) / len(rr) if rr else None,
                 'confusion': {g: dict(collections.Counter(r['prediction'] for r in ss if r['gold'] == g)) for g in LABELS}}
            if final and values:
                import numpy as np
                rng = np.random.default_rng(20261020)
                means = np.asarray(values)[rng.integers(0, len(values), (2000, len(values)))].mean(axis=1)
                d['source_cluster_bootstrap95'] = np.quantile(means, [.025, .975]).tolist()
            report[condition][name] = d
    if final:
        # All-three-state metric counts families, not three independent patients.
        families = collections.defaultdict(list)
        for r in expanded:
            if r['case_kind'] == 'primary_three_state' and not r['stable_insufficiency']:
                families[(r['condition'], r['source_id'], r['item_id'])].append(r)
        report['balanced_family_triplets'] = {
            condition: {'families': sum(k[0] == condition for k in families),
                        'all_three_correct': sum(k[0] == condition and len(rr) == 3 and all(r['correct'] for r in rr) for k, rr in families.items()),
                        'complete_three_predictions': sum(k[0] == condition and len(rr) == 3 and all(r['submitted'] for r in rr) for k, rr in families.items())}
            for condition in ('full_context', 'source_withheld', 'stem_only')}
        pairs = {}
        index = {(r['condition'], r['case_id'], r['item_id']): r for r in expanded}
        for probe in ('source_withheld', 'stem_only'):
            clusters = collections.defaultdict(list)
            for r in expanded:
                if r['condition'] != 'full_context' or r['adaptive_calibration_reused'] or r['stable_insufficiency'] or r['case_kind'] != 'primary_three_state':
                    continue
                other = index.get((probe, r['case_id'], r['item_id']))
                if r['submitted'] and other and other['submitted']:
                    clusters[r['source_cluster']].append(int(r['correct']) - int(other['correct']))
            vals = [sum(v) / len(v) for v in clusters.values()]
            d = {'paired_source_clusters': len(vals), 'macro_full_minus_probe': sum(vals) / len(vals) if vals else None,
                 'meaning': 'Matched development predictability contrast, not a causal clinical validation or a heldout generalization estimate.'}
            if vals:
                import numpy as np
                rng = np.random.default_rng(20261020)
                means = np.asarray(vals)[rng.integers(0, len(vals), (2000, len(vals)))].mean(axis=1)
                d['paired_source_cluster_bootstrap95'] = np.quantile(means, [.025, .975]).tolist()
            pairs[probe] = d
        report['paired_new_balanced_contrasts'] = pairs
    return {'planned_unique_actor_requests': len(jobs), 'recorded_unique_actor_requests': len(rows),
            'submitted_unique_actor_requests': sum(r.get('submitted', False) for r in rows.values()),
            'statuses': dict(collections.Counter(r['status'] for r in rows.values())),
            'actual_request_attempts': sum(r.get('requests', 0) for r in rows.values()),
            'report': report, 'complete': len(rows) == len(jobs),
            'development_silver_not_clinical_gold': True, 'quality_retry': False}
