"""Compile validated source-specific DSLs into the existing variable-size sandbox."""
from collections import Counter
import copy
import hashlib
import json
import shutil
from run_generation import ROOT, read, dump, digest
from scale09_rules import outcome, S, R, U

SCENES = '甲乙丙丁戊己庚'


def key(text): return hashlib.sha256(text.encode()).hexdigest()


def render(value):
    if value is None: return '本轮值未记录；不能用历史值替代'
    if type(value) is bool: return '是' if value else '否'
    return str(value)


def compile_source(source, clean, target):
    sid = source['source_id']
    order = sorted(range(len(clean)), key=lambda j: key(sid + '|scene|' + str(j)))
    iid_by = {j: 'C' + str(k + 1) for k, j in enumerate(order)}
    scene_by = {j: SCENES[k] for k, j in enumerate(order)}
    perms = {j: sorted(range(3), key=lambda k: key(sid + '|state|' + str(j) + '|' + str(k))) for j in range(len(clean))}
    cases, variants = [], []
    for packet in range(3):
        tid = sid + '__scale09_' + key(sid + '|packet|' + str(packet))[:10]
        claims, chart, gold, proofs = [], [], [], []
        for j in order:
            item = clean[j]; state = item['state_values'][perms[j][packet]]
            iid = iid_by[j]; aid = sid + '-SIM-' + str(j).zfill(2)
            statement = '场景' + scene_by[j] + '：' + item['statement']
            claims.append({'item_id': iid, 'claim_type': 'bounded_clinical_assertion', 'statement': statement})
            chart.append({'anchor_id': aid, 'label': '独立模拟场景' + scene_by[j], 'origin_kind': 'explicit_synthetic_chartlet',
                          'text': '独立模拟场景' + scene_by[j] + '。\n' + '\n'.join(item['field_map'][f]['label'] + ('（' + item['field_map'][f]['unit'] + '）' if item['field_map'][f]['unit'] else '') + '：' + render(v) for f, v in state.items())})
            label = outcome(item['expression'], state, item['constant_map'])
            refs = list(dict.fromkeys(r['anchor_id'] for r in item['references']))
            refs += [c['anchor_id'] for c in item['constant_map'].values() if c['anchor_id'] not in refs]
            refs = list(dict.fromkeys(refs))
            gold.append({'item_id': iid, 'label': label, 'support_anchor_ids': [aid] + refs, 'sufficient_evidence_groups': [[aid] + refs]})
            proofs.append({'item_id': iid, 'author_index': j, 'configuration_index_private': perms[j][packet], 'state': state,
                           'expression': item['expression'], 'constants': item['constant_map'], 'fields': item['field_map'],
                           'label': label, 'stable_rule_gap': item['family_type'] == 'stable_insufficiency',
                           'mechanism': item['family_type'], 'source_required_stratum': item['family_type'] != 'stable_insufficiency',
                           'clinical_scope': item['clinical_scope'], 'source_quotes': item['references'],
                           'reasoning_steps_private': item['reasoning_steps'], 'missing_information_private': item['missing_information'],
                           'not_expert_clinical_gold': True})
        original = copy.deepcopy(source['documents'])
        chart_doc = {'document_id': sid + '-SIMULATED-CHART', 'title': '明确模拟：独立临床交接状态（不是原始随访）',
                     'origin_kind': 'explicit_synthetic_chartlets', 'blocks': chart}
        visible = [b['anchor_id'] for d in original + [chart_doc] for b in d['blocks']]
        public = {'source_id': sid, 'modality': source['modality'], 'environment': 'E2' if source['modality'] == 'dialogue' else 'E3',
                  'key_revokes_write': False, 'task_id': sid + '__scale09_clinical_state', 'task_version': 'scale09',
                  'documents': original + [chart_doc], 'claims': claims,
                  'scenario_contract': '每个字母场景是不同的独立模拟患者。当前患者参数只来自相应模拟状态；原文仅供核对规则或命名历史参照。缺失不是阴性。仅核对陈述的有界含义，不据此向真实患者给治疗建议。',
                  'visibility': {'initial': visible, 'updated': visible}}
        meta = {'source_id': sid, 'source_cluster': source['source_cluster'], 'split': 'development',
                'source_characters': source['characters'], 'task_id': tid, 'configuration_order_private': packet,
                'items': len(clean), 'variant_type': 'source_specific_typed_clinical_counterfactuals',
                'not_independent_patient': True, 'modality': source['modality'], 'adaptive_calibration_reused': False,
                'source_required_item_ids_private': sorted(p['item_id'] for p in proofs if p['source_required_stratum'])}
        dest = target / 'cases' / tid
        for name, value in [('controller.private.json', public), ('gold.private.json', {'initial': gold, 'updated': gold}),
                            ('proofs.private.json', proofs), ('metadata.json', meta)]: dump(dest / name, value)
        variants.append({'public': public, 'gold': gold, 'proofs': proofs}); cases.append(meta)
    assert variants[0]['public']['claims'] == variants[1]['public']['claims'] == variants[2]['public']['claims']
    # Select an initial state before solver access, with at least two known
    # balanced claims. Flip one definite label and one known/unknown boundary.
    base_idx = next(k for k, v in enumerate(variants) if sum(g['label'] != U for g in v['gold']) >= 2)
    base = variants[base_idx]; gb = {g['item_id']: g for g in base['gold']}
    balanced = [iid_by[j] for j in order if clean[j]['family_type'] != 'stable_insufficiency']
    first = next(i for i in balanced if gb[i]['label'] != U)
    second = next((i for i in balanced if i != first and gb[i]['label'] == U), next(i for i in balanced if i != first))
    desired = {first: R if gb[first]['label'] == S else S, second: S if gb[second]['label'] == U else U}
    updated = copy.deepcopy(base['public']); ug = copy.deepcopy(gb); changes = []
    for iid, label in desired.items():
        v = next(v for v in variants if next(g for g in v['gold'] if g['item_id'] == iid)['label'] == label)
        p = next(p for p in v['proofs'] if p['item_id'] == iid)
        aid = sid + '-SIM-' + str(p['author_index']).zfill(2)
        b = next(b for b in v['public']['documents'][-1]['blocks'] if b['anchor_id'] == aid)
        updated['documents'][-1]['blocks'] = [copy.deepcopy(b) if x['anchor_id'] == aid else x for x in updated['documents'][-1]['blocks']]
        ug[iid] = copy.deepcopy(next(g for g in v['gold'] if g['item_id'] == iid))
        changes.append({'item_id': iid, 'from': gb[iid]['label'], 'to': label, 'updated_source_state_proof': p})
    ep = {'source_id': sid, 'source_cluster': source['source_cluster'], 'initial_task_id': cases[base_idx]['task_id'],
          'updated_controller': updated, 'gold': {'initial': base['gold'], 'updated': [ug[g['item_id']] for g in base['gold']]},
          'changes_private': changes, 'changed_count': 2, 'stable_count': len(clean) - 2,
          'definite_flip_count': 1, 'uncertainty_transition_count': 1, 'initial_choice_uses_solver_output': False,
          'not_actor_visible': ['gold', 'changes_private', 'changed_count', 'initial_choice_criterion']}
    dump(target / 'episodes' / (sid + '.private.json'), ep)
    return cases


def build(protocol, sources, accepted, output):
    if output.exists(): raise FileExistsError('Dataset freeze is append-only; never overwrite')
    old = ROOT / 'datasets/medical07_dev8_r3'
    assert digest(old / 'build_integrity.json') == protocol['old_dataset_integrity_sha256']
    m0 = read(old / 'manifest.json'); cases = []
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
        g = read(output / 'cases' / c['task_id'] / 'gold.private.json')['initial']
        counts.update(x['label'] for x in g)
        if c['configuration_order_private'] == 0: families += len(g)
    audit = {'source_clusters': len(ids), 'new_source_clusters': len(accepted), 'adaptive_calibration_reused': 8,
             'state_packets': len(cases), 'question_families': families, 'judgments': sum(counts.values()), 'labels': dict(counts),
             'balanced_triplets': counts[S], 'stable_insufficiency_families': len(ids),
             'original_source_records': sum(len(sources[s]['documents']) for s in ids),
             'all_original_block_texts_preserved': True, 'clinical_expert_gold': False, 'solver_experiments_completed': 0,
             'previous_gates_not_overridden': True, 'not_independent_patients': True}
    dump(output / 'construction_audit.json', audit)
    dump(output / 'manifest.json', {'kind': 'scale09-development-source-specific-silver-triplets', 'source_ids': ids, 'cases': cases,
                                  'clinical_expert_gold': False, 'protocol_sha256': digest(ROOT / 'construction/scale09_dev120/protocol.json'),
                                  'old_dataset_integrity_sha256': protocol['old_dataset_integrity_sha256']})
    dump(output / 'build_integrity.json', {str(p.relative_to(output)).replace('\\', '/'): digest(p) for p in sorted(output.rglob('*.json'))})
    return audit
