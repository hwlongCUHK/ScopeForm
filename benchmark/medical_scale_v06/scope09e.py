"""Post-construction trace closure: preserve short replies WITH their context."""
from collections import Counter
import shutil
import scale09d as prior
import scale09_rules_r4 as rules
from run_generation import ROOT, read, dump, digest
from compile_scale09 import compile_source

def build(protocol, sources, accepted, ignored_output):
    output = ROOT / 'datasets/medical09_dev120_silver_r4'
    if output.exists(): raise FileExistsError('Never overwrite a dataset freeze')
    old = ROOT / 'datasets/medical07_dev8_r3'; m0 = read(old / 'manifest.json'); cases = []
    assert digest(old / 'build_integrity.json') == protocol['old_dataset_integrity_sha256']
    for c in m0['cases']:
        shutil.copytree(old / 'cases' / c['task_id'], output / 'cases' / c['task_id'])
        cases.append({**c, 'adaptive_calibration_reused': True})
    for sid in m0['source_ids']:
        dst = output / 'episodes' / (sid + '.private.json'); dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old / 'episodes' / (sid + '.private.json'), dst)
    for sid, clean in accepted.items(): cases += compile_source(sources[sid], clean, output)
    ids = m0['source_ids'] + list(accepted)
    assert len(ids) == len(set(ids)) and all(sources[s]['split'] == 'development' for s in ids)
    counts = Counter(); families = 0
    for c in cases:
        gs = read(output / 'cases' / c['task_id'] / 'gold.private.json')['initial']; counts.update(g['label'] for g in gs)
        if c['configuration_order_private'] == 0: families += len(gs)
    audit = {'source_clusters': len(ids), 'new_source_clusters': len(accepted), 'adaptive_calibration_reused': 8,
             'state_packets': len(cases), 'question_families': families, 'judgments': sum(counts.values()), 'labels': dict(counts),
             'balanced_triplets': counts[S], 'stable_insufficiency_families': len(ids),
             'original_source_records': sum(len(sources[s]['documents']) for s in ids), 'all_original_block_texts_preserved': True,
             'clinical_expert_gold': False, 'solver_experiments_completed': 0, 'previous_gates_not_overridden': True, 'not_independent_patients': True}
    dump(output / 'construction_audit.json', audit)
    dump(output / 'manifest.json', {'kind': 'scale09-r4-development-source-specific-silver-triplets', 'source_ids': ids, 'cases': cases,
                                  'clinical_expert_gold': False, 'protocol_sha256': digest(WORK / 'protocol.json'),
                                  'old_dataset_integrity_sha256': protocol['old_dataset_integrity_sha256'],
                                  'construction_revision': 'flat_source_rule_graphs_programmatic_state_search'})
    dump(output / 'build_integrity.json', {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob('*.json'))})
    return audit





WORK = ROOT / 'construction/scale09_dev120_r4'
_normalize = prior.normalize_graph


def normalize_graph(item, source):
    item = _normalize(item, source)
    context = {}
    for document in source['documents']:
        for pos, block in enumerate(document['blocks']):
            context[block['anchor_id']] = (block, document['blocks'][max(0, pos - 3):pos])
    refs = list(item['references'])
    for aid in list(refs) + [c['anchor_id'] for c in item['constants']]:
        block, preceding = context[aid]
        if aid not in refs: refs.append(aid)
        if len(block['text']) < 6:
            # Never concatenate a paraphrased quote or change the original
            # short reply. Keep distinct original anchors for the antecedent.
            for previous in reversed(preceding):
                if previous['anchor_id'] not in refs: refs.append(previous['anchor_id'])
                if len(previous['text']) >= 6: break
    item['references'] = refs
    return item


def bank(source):
    old_normalize, old_rules = prior.normalize_graph, prior.rules
    prior.normalize_graph, prior.rules = normalize_graph, rules
    try: return prior.bank(source)
    finally: prior.normalize_graph, prior.rules = old_normalize, old_rules


def extract(candidate, source):
    old_normalize, old_rules = prior.normalize_graph, prior.rules
    prior.normalize_graph, prior.rules = normalize_graph, rules
    try: return prior.extract(candidate, source)
    finally: prior.normalize_graph, prior.rules = old_normalize, old_rules

