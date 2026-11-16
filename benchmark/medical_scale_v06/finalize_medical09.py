"""Materialize the 120-source development target without using solver results."""
from collections import Counter
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
from run_generation import ROOT, read, dump, digest, run_lock
import scope09e as scoped
import rich_tasks09 as rich
from clinical_contrasts09 import refine
import scale09_rules_r4 as rules
import compile_scale09 as compiler

WORK = ROOT / 'construction/medical09_target120'
DATASET = ROOT / 'datasets/medical09_dev120_silver_v1'
CODE = ('finalize_medical09.py', 'clinical_contrasts09.py', 'scope09e.py', 'rich_tasks09.py',
        'scale09d.py', 'scale09c.py', 'scale09_rules_r4.py', 'scale09_rules_r3.py',
        'compile_scale09.py', 'scale09_rules.py', 'run_generation.py')


def freeze_plan():
    plan = {'protocol_sha256': digest(WORK / 'protocol.json'),
            'code_sha256': {n: digest(ROOT / n) for n in CODE},
            'new_solver_results_used': False}
    path = WORK / 'postconstruction_plan.json'
    if path.exists() and read(path) != plan:
        raise ValueError('Postconstruction qualification changed; use a new version, not stale caches')
    if not path.exists(): dump(path, plan)
    return digest(path)


def load_sources():
    original = ROOT / 'construction/scale09_dev120_r3'; reserve = ROOT / 'construction/rule_rich_reserve09'
    p = original / 'development_sources.private.jsonl'; m = read(original / 'development_sources_integrity.json')
    assert digest(p) == m['sha256']
    old = {s['source_id']: s for s in (json.loads(x) for x in p.read_text(encoding='utf-8').splitlines())}
    p = reserve / 'sources.private.jsonl'; m = read(reserve / 'manifest.json'); assert digest(p) == m['sha256']
    extra = {s['source_id']: s for s in (json.loads(x) for x in p.read_text(encoding='utf-8').splitlines())}
    assert not set(old) & set(extra)
    assert not {s['source_cluster'] for s in old.values()} & {s['source_cluster'] for s in extra.values()}
    combined = {**old, **extra}; assert all(s['split'] == 'development' for s in combined.values())
    return combined, set(old), set(extra)


def operator_count(node):
    def simplify(n):
        if not isinstance(n, dict) or 'op' not in n: return n
        args = [simplify(a) for a in n['args']]
        if n['op'] == 'not' and isinstance(args[0], dict) and args[0].get('op') == 'not': return args[0]['args'][0]
        if n['op'] in ('eq', 'ne') and args[0] == args[1]: return n['op'] == 'eq'
        return {'op': n['op'], 'args': args}
    node = simplify(node)
    if not isinstance(node, dict): return 0
    nodes = {json.dumps(n, sort_keys=True, ensure_ascii=False) for n in rules.walk(node) if isinstance(n, dict) and 'op' in n and not
        (n['op'] in ('eq', 'ne') and n['args'][0] == n['args'][1])}
    return len(nodes)


def qualify(sid, source, origin):
    folder = WORK / 'source_checks' / sid
    resultfile = folder / 'result.json'
    if resultfile.exists():
        row = read(resultfile)
        assert row['postconstruction_plan_sha256'] == digest(WORK / 'postconstruction_plan.json')
        return (read(folder / 'accepted_spec.private.json') if row['qualified'] else None), row
    selected, errors = rich.bank(source) if origin == 'rich' else scoped.bank(source)
    if not selected:
        row = {'source_id': sid, 'modality': source['modality'], 'origin': origin, 'qualified': False,
               'reason': 'no_valid_source_rule_bank', 'errors': errors,
               'postconstruction_plan_sha256': digest(WORK / 'postconstruction_plan.json')}
        dump(resultfile, row); return None, row
    improved = []; provenance = []
    for clean, canonical, proof in selected:
        item = refine(clean, canonical, proof, source)
        item['construction_provenance_private'] = proof
        item['nonredundant_operator_count_private'] = operator_count(item['expression'])
        improved.append(item); provenance.append(proof)
    candidates = [i for i in improved if i['family_type'] != 'stable_insufficiency' and
                  i['counterfactual_design_private'].get('reference_hidden_from_stem') and
                  i['counterfactual_design_private'].get('primary_content_challenge') and
                  i['nonredundant_operator_count_private'] >= 3]
    checks, failure = rules.validate({'source_id': sid, 'items': improved}, source)
    qualified = bool(candidates) and not failure
    row = {'source_id': sid, 'source_cluster': source['source_cluster'], 'modality': source['modality'], 'origin': origin,
           'qualified': qualified, 'families': len(improved), 'hidden_reference_primary_families': len(candidates),
           'errors': failure, 'reason': None if qualified else 'no_nontrivial_hidden_reference_content_contrast_or_invalid_refinement',
           'clinical_expert_gold': False, 'new_solver_results_used': False}
    row['postconstruction_plan_sha256'] = digest(WORK / 'postconstruction_plan.json')
    if qualified: dump(folder / 'accepted_spec.private.json', improved)
    dump(resultfile, row); return (improved if qualified else None), row


def decorate_cases(target, sid, specs, cases):
    for case in cases:
        folder = target / 'cases' / case['task_id']; proofs = read(folder / 'proofs.private.json')
        hidden = []; balanced = []
        for proof in proofs:
            spec = specs[proof['author_index']]; design = spec['counterfactual_design_private']
            proof['counterfactual_design_private'] = design
            proof['construction_provenance_private'] = spec['construction_provenance_private']
            proof['task_scope_private'] = spec.get('task_scope_private')
            proof['source_required_stratum'] = bool(design.get('reference_hidden_from_stem'))
            if not proof['stable_rule_gap']: balanced.append(proof['item_id'])
            if proof['source_required_stratum']: hidden.append(proof['item_id'])
        case['source_required_item_ids_private'] = sorted(hidden); case['balanced_item_ids_private'] = sorted(balanced)
        case['source_origin_private'] = 'rule_rich_new_development' if sid.startswith('R09D') else 'original_development_pool'
        dump(folder / 'proofs.private.json', proofs); dump(folder / 'metadata.json', case)


def selective_episodes(target, sid):
    ep = read(target / 'episodes' / (sid + '.private.json')); base = read(target / 'cases' / ep['initial_task_id'] / 'controller.private.json')
    gb = {g['item_id']: g for g in ep['gold']['initial']}; gu = {g['item_id']: g for g in ep['gold']['updated']}
    for changed in ep['changes_private']:
        iid = changed['item_id']; proof = changed['updated_source_state_proof']; aid = sid + '-SIM-' + str(proof['author_index']).zfill(2)
        updated = copy.deepcopy(base)
        replacement = next(b for b in ep['updated_controller']['documents'][-1]['blocks'] if b['anchor_id'] == aid)
        updated['documents'][-1]['blocks'] = [copy.deepcopy(replacement) if b['anchor_id'] == aid else b for b in updated['documents'][-1]['blocks']]
        label_type = 'definite_only' if {gb[iid]['label'], gu[iid]['label']} == {rules.S, rules.R} else 'uncertainty_only'
        gs = [copy.deepcopy(gu[g['item_id']] if g['item_id'] == iid else g) for g in ep['gold']['initial']]
        stable_known = sum(gb[g['item_id']]['label'] == g['label'] and g['label'] != rules.U for g in gs)
        assert stable_known >= 1
        value = {**ep, 'updated_controller': updated, 'gold': {'initial': ep['gold']['initial'], 'updated': gs},
                 'changes_private': [changed], 'changed_count': 1, 'stable_count': len(gs) - 1,
                 'definite_flip_count': int(label_type == 'definite_only'), 'uncertainty_transition_count': int(label_type == 'uncertainty_only'),
                 'known_stable_count': stable_known, 'update_subtype': label_type, 'initial_choice_uses_solver_output': False}
        dump(target / 'episodes_selective' / (sid + '__' + label_type + '.private.json'), value)


def build(sources, accepted, selection, protocol):
    if DATASET.exists(): raise FileExistsError('Never overwrite a frozen dataset')
    old = ROOT / 'datasets/medical07_dev8_r3'; m0 = read(old / 'manifest.json')
    assert digest(old / 'build_integrity.json') == protocol['preserve_old_dataset_integrity_sha256']
    cases = []
    for case in m0['cases']:
        shutil.copytree(old / 'cases' / case['task_id'], DATASET / 'cases' / case['task_id'])
        cases.append({**case, 'adaptive_calibration_reused': True})
    for sid in m0['source_ids']:
        destination = DATASET / 'episodes' / (sid + '.private.json'); destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old / 'episodes' / (sid + '.private.json'), destination)
    compiler.outcome = rules.outcome
    for sid, specs in accepted.items():
        new = compiler.compile_source(sources[sid], specs, DATASET); decorate_cases(DATASET, sid, specs, new); cases += new
    ids = m0['source_ids'] + list(accepted); assert len(ids) == 120 and len(set(ids)) == 120
    for sid in ids: selective_episodes(DATASET, sid)
    counts = Counter(); families = hidden = 0
    for case in cases:
        gs = read(DATASET / 'cases' / case['task_id'] / 'gold.private.json')['initial']; counts.update(g['label'] for g in gs)
        if case['configuration_order_private'] == 0:
            families += len(gs); hidden += len(case['source_required_item_ids_private'])
    audit = {'source_clusters': 120, 'new_source_clusters': 112, 'reused_adaptive_sources': 8,
        'modalities': dict(Counter(sources[s]['modality'] for s in ids)), 'original_source_records': sum(len(sources[s]['documents']) for s in ids),
        'state_packets': len(cases), 'question_families': families, 'judgments': sum(counts.values()), 'labels': dict(counts),
        'balanced_families': counts[rules.S], 'stable_insufficiency_families': 120, 'hidden_reference_families_including_old_author_stratum': hidden,
        'mixed_update_episodes': 120, 'selective_update_episodes': 240, 'new_rule_rich_sources_selected': sum(s.startswith('R09D') for s in ids),
        'clinical_expert_gold': False, 'solver_evaluations_completed': 0, 'old_results_and_gates_unchanged': True}
    assert audit['modalities'] == protocol['target_modalities']
    dump(DATASET / 'construction_audit.json', audit)
    dump(DATASET / 'manifest.json', {'kind': 'medical09-dev120-source-grounded-synthetic-state-silver-v1', 'source_ids': ids, 'cases': cases,
         'protocol_sha256': digest(WORK / 'protocol.json'), 'old_dataset_integrity_sha256': digest(old / 'build_integrity.json'),
         'clinical_expert_gold': False, 'source_selection': selection, 'source_split': 'development_only',
         'not_real_patient_count': True, 'no_source_text_public_release_authorized': True})
    dump(DATASET / 'build_integrity.json', {p.relative_to(DATASET).as_posix(): digest(p) for p in sorted(DATASET.rglob('*.json'))})
    return audit


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--check-only', action='store_true'); args = parser.parse_args()
    freeze_plan()
    protocol = read(WORK / 'protocol.json'); sources, original, extra = load_sources()
    m0 = read(ROOT / 'datasets/medical07_dev8_r3/manifest.json'); reused = set(m0['source_ids'])
    baseline = ROOT / 'construction/scale09_dev120_r3'
    # The frozen upstream runner completed its source queue, then failed while
    # exporting a partial dataset (an undefined label constant). Preserve that
    # failure; consume only terminal proposals and its saved selection here.
    if not (baseline / 'selection.json').exists():
        print(json.dumps({'status': 'waiting_original_construction'})); return 3
    upstream = read(baseline / 'selection.json')
    expected = {s for q in upstream for s in q['primary_ids'] + q['reserve_attempted_ids']} - reused
    assert all((baseline / 'sources' / s / 'result.json').exists() for s in expected)
    source_order = read(baseline / 'protocol.json')['stratum_queue']
    priorities = []
    for group in source_order: priorities += group['primary_ids']
    for group in source_order: priorities += group['reserve_ids']
    priorities = list(dict.fromkeys(priorities))
    accepted = {}; selected = {m: [s for s in m0['source_ids'] if sources[s]['modality'] == m] for m in ('dialogue', 'drug')}
    history = []
    for sid in priorities:
        if sid in reused: continue
        if not (baseline / 'sources' / sid / 'result.json').exists(): continue
        specs, row = qualify(sid, sources[sid], 'original'); history.append(row)
        if len(history) % 10 == 0: print(json.dumps({'qualified_progress': len(history), 'selected': {k: len(v) for k, v in selected.items()}}), flush=True)
        kind = sources[sid]['modality']
        if specs and len(selected[kind]) < protocol['target_modalities'][kind]: selected[kind].append(sid); accepted[sid] = specs
    reserve_order = [s['source_id'] for s in read(ROOT / 'construction/rule_rich_reserve09/manifest.json')['source_catalog']]
    prefix = 0
    for sid in reserve_order:
        if len(selected['dialogue']) == protocol['target_modalities']['dialogue']: break
        if not (rich.WORK / 'sources' / sid / 'result.json').exists(): break
        prefix += 1
        specs, row = qualify(sid, sources[sid], 'rich'); history.append(row)
        if specs: selected['dialogue'].append(sid); accepted[sid] = specs
    progress = {'selected_counts': {m: len(v) for m, v in selected.items()}, 'target': protocol['target_modalities'],
                'completed_rich_prefix_considered': prefix, 'qualified_new_sources': len(accepted), 'old_reused': 8,
                'new_solver_results_used': False}
    dump(WORK / 'selection_progress.json', progress)
    if args.check_only:
        print(json.dumps({'status': 'qualification_only', **progress}, ensure_ascii=False)); return 0
    if any(len(selected[m]) != protocol['target_modalities'][m] for m in selected):
        print(json.dumps({'status': 'waiting_or_source_shortfall', **progress}, ensure_ascii=False)); return 2
    selection = {'selected_ids_by_modality': selected, 'rich_prefix': prefix, 'qualification_history': history,
                 'no_completion_speed_or_solver_accuracy_selection': True, 'source_length_quotas_not_filled_with_invented_rules': True}
    dump(WORK / 'selection.json', selection)
    # Preserve all admitted rule banks next to the final construction plan.
    for sid, specs in accepted.items(): dump(WORK / 'sources' / sid / 'accepted_spec.private.json', specs)
    with (WORK / 'combined_development_sources.private.jsonl').open('x', encoding='utf-8', newline='\n') as f:
        for sid in sorted(sources): f.write(json.dumps(sources[sid], ensure_ascii=False, separators=(',', ':')) + '\n')
    dump(WORK / 'combined_source_integrity.json', {'sha256': digest(WORK / 'combined_development_sources.private.jsonl'),
         'sources': len(sources), 'original_development_sources': len(original), 'extra_rule_rich_development_sources': len(extra), 'validation_or_locked_texts_included': False})
    audit = build(sources, accepted, selection, protocol)
    dump(WORK / 'complete.json', {'status': 'target_120_source_development_silver_constructed', 'audit': audit,
         'dataset_integrity_sha256': digest(DATASET / 'build_integrity.json'), 'clinical_expert_gold': False,
         'old_failed_scientific_gates_overridden': False, 'solver_evaluations_completed': 0})
    print(json.dumps(read(WORK / 'complete.json'), ensure_ascii=False)); return 0


if __name__ == '__main__': raise SystemExit(main())
