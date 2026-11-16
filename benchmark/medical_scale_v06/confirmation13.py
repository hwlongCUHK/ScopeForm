"""Append-only, local-only construction of the reserved120 confirmation sources.

No solver calls, accuracy filtering, clinical gold claims or writes to old data.
Author proposals are untrusted: typed rules and labels are compiled locally.
"""
import argparse
from collections import Counter, defaultdict
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import statistics
import unicodedata

from run_generation import ROOT, read, digest
import scale09_rules_r4 as rules
import scope09e as scoped
import scale09d as bank_selector
from clinical_contrasts09 import refine, OPERATIONAL
from finalize_medical09 import operator_count
import quantitative_target09 as quantitative
from workflow11_runtime import clean_controller

RESERVATION = ROOT.parent / 'output/medical12_www_gaps_20261022/confirmation_reservation.json'
POOL = ROOT / 'datasets/source_pool_v06_1000'
WORK = ROOT / 'construction/confirmation13_120_v1'
SOURCES = ROOT / 'datasets/medical13_confirmation120_sources_v1'
DATASET = ROOT / 'datasets/medical13_confirmation120_silver_v1'
OUT = ROOT.parent / 'output/medical13_confirmation_20261022'
LABELS = (rules.S, rules.R, rules.U)
EVENTS = ('key_mixed', 'single_definite', 'single_uncertainty', 'neutral')
CODE = ('confirmation13.py', 'construct_confirmation13.py', 'test_confirmation13.py',
        'scale09_rules_r4.py', 'scale09c.py', 'scale09d.py', 'scope09e.py',
        'clinical_contrasts09.py', 'quantitative_target09.py', 'workflow11_runtime.py')


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode('utf-8')


def hash_value(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def frozen(path, value):
    """Idempotent only for byte-identical output; never replace an old artifact."""
    raw = value if isinstance(value, bytes) else value.encode('utf-8') if isinstance(value, str) else encoded(value)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('Immutable artifact differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(raw)


def status(**fields):
    from run_generation import dump
    prior = read(OUT / 'RUN_STATUS.json') if (OUT / 'RUN_STATUS.json').exists() else {}
    prior.update(fields)
    prior.update(solver_calls=0, clinical_expert_gold=False)
    dump(OUT / 'RUN_STATUS.json', prior)


def jsonl(rows):
    return ''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in rows).encode('utf-8')


def verify_tree(folder, integrity='build_integrity.json'):
    manifest = read(folder / integrity)
    for rel, expected in manifest.items():
        path = (folder / rel).resolve()
        if not path.is_relative_to(folder.resolve()) or digest(path) != expected:
            raise ValueError('Frozen input hash mismatch: ' + rel)
    return True


def source_text(source):
    return '\n'.join(b['text'] for d in source['documents'] for b in d['blocks'])


def normalize(text):
    text = unicodedata.normalize('NFKC', text).lower()
    return ''.join(c for c in text if c.isalnum())


def near_duplicates(selected, exposed):
    """Exact comparison of normalized7-grams, not a global contamination test."""
    all_rows = {s['source_cluster']: s for s in exposed + selected}
    names = {k: normalize(source_text(s)) for k, s in all_rows.items()}
    grams = {k: {t[i:i+7] for i in range(len(t)-6)} for k, t in names.items()}
    selected_keys = {s['source_cluster'] for s in selected}
    exact, close, seen = [], [], set()
    for source in selected:
        a = source['source_cluster']
        for b, other in all_rows.items():
            if a == b or source['modality'] != other['modality']:
                continue
            pair = tuple(sorted((a, b)))
            if pair in seen:
                continue
            seen.add(pair)
            # Brief source IDs only; never put text or quotations in the report.
            record = {'source_id_a': source['source_id'], 'source_id_b': other['source_id'],
                      'both_reserved': b in selected_keys, 'modality': source['modality']}
            if names[a] == names[b]:
                exact.append(record)
                continue
            aa, bb = grams[a], grams[b]
            if not aa or not bb:
                continue
            shared = len(aa & bb)
            jac = shared / (len(aa) + len(bb) - shared)
            containment = shared / min(len(aa), len(bb))
            if jac >= .80 or (containment >= .92 and min(len(names[a]), len(names[b])) >= 500):
                close.append({**record, 'jaccard_7gram': round(jac, 6),
                              'shorter_containment_7gram': round(containment, 6)})
    return {'normalization': 'NFKC, lowercase, retain Unicode alphanumeric; complete original source body',
            'method': 'exact character7-gram Jaccard and containment; same-modality source comparisons',
            'comparisons': len(seen), 'exposed_clusters_compared': len({s['source_cluster'] for s in exposed}),
            'exact_normalized_duplicate_pairs': exact, 'near_duplicate_flags': close,
            'near_duplicates_need_contextual_review': bool(close),
            'not_patient_identity_or_training_contamination_certificate': True}


def load_reserved():
    return [json.loads(x) for x in (SOURCES / 'sources.private.jsonl').read_text(encoding='utf-8').splitlines()]


def load_exposed_sources():
    """Use locally available original snapshots plus the actual old actor inputs."""
    rows = {}
    for relative in ('construction/scale09_dev120_r3/development_sources.private.jsonl',
                     'construction/rule_rich_reserve09/sources.private.jsonl',
                     'construction/explicit_regimen_reserve09/sources.private.jsonl'):
        path = ROOT / relative
        if path.exists():
            for line in path.read_text(encoding='utf-8').splitlines():
                source = json.loads(line)
                rows[source['source_cluster']] = source
    targets = {r['source_id']: r for r in map(json.loads, (ROOT / 'workflow11_inputs_private/evaluator_targets.LOCAL_ONLY.private.jsonl').read_text(encoding='utf-8').splitlines())}
    for actor in map(json.loads, (ROOT / 'workflow11_inputs_private/actors.private.jsonl').read_text(encoding='utf-8').splitlines()):
        sid = actor['source_id']
        docs = [d for d in actor['initial']['documents'] if d.get('origin_kind') != 'explicit_synthetic_chartlets']
        cluster = targets[sid]['source_cluster']
        if cluster not in rows:
            rows[cluster] = {'source_id': sid, 'source_cluster': cluster,
                             'modality': actor['initial']['modality'], 'documents': docs}
    return list(rows.values())


def protocol():
    return {'version': 'confirmation13-source-holdout-silver-v1',
            'reservation_sha256': digest(RESERVATION),
            'source_seal_sha256': digest(SOURCES / 'build_integrity.json'),
            'code_sha256': {name: digest(ROOT / name) for name in CODE},
            'sources': 120, 'modalities': {'dialogue': 72, 'drug': 48},
            'candidate_selection': 'Exactly the120 reserved IDs, no replacements, no solver score access.',
            'construction': 'Exact-regimen parser first, otherwise at most3 source-grounded typed-graph author proposals; retain construction failures.',
            'construction_model': 'Installed Qwen3.8-27B directory; author only, not an independent clinical judge.',
            'author_runtime_profile': {'max_model_len': 65536, 'max_output_tokens': 8192,
                'temperature': .2, 'enable_thinking': False, 'workers': 4,
                'max_proposals_per_source': 3, 'local_loopback_port': 18630,
                'no_remote_commercial_api_fallback': True},
            'labels': 'Local exact-rational, typed three-valued DSL plus a separately implemented evaluator.',
            'primary_family_requirement': 'At least2 distinct balanced source-dependent families, at least1 source-reference clinical-parameter contrast with >=3 nonredundant operators and2 effective inputs, plus1 persistent-insufficiency control.',
            'protected_probe': 'If needed reuse one valid family in a separately simulated scene; never count that as a new independent rule family.',
            'directions': {'single_definite': {'S_to_R': 60, 'R_to_S': 60},
                           'single_uncertainty': {'S_to_U': 30, 'R_to_U': 30, 'U_to_S': 30, 'U_to_R': 30}},
            'direction_assignment': 'Fixed SHA256 ordering within modality x length strata before authoring or solver access.',
            'question_only_control': 'Every balanced family has the identical stem in all3 truth states; full-information and evidence-removal views use identical claims.',
            'source_holdout_not_rule_family_holdout': True,
            'same_builder_templates_may_be_shared_with_development': True,
            'no_performance_based_filters_or_target_accuracy': True,
            'solver_calls': 0, 'clinical_expert_gold': False,
            'inference_authorized_by_this_build': False,
            'public_release_authorized': False,
            'clinical_review_status': 'pending; a mechanical silver seal is not clinical approval',
            'full_task_freeze_requires': ['all120 reserved sources machine-qualified', 'all source roundtrips exact',
                'independent evaluator agrees', 'same-stem triplets balanced', 'all six transition directions',
                '>=1 unchanged known claim and1 unchanged insufficiency claim per mixed update',
                'current evidence anchors exist', 'actor/target separation', 'old artifacts unchanged']}


def assignment(catalog):
    by = defaultdict(list)
    for row in catalog:
        by[(row['modality'], row['length_band'])].append(row['source_id'])
    out = {}
    uncertainty = [(rules.S, rules.U), (rules.R, rules.U), (rules.U, rules.S), (rules.U, rules.R)]
    for stratum, ids in sorted(by.items()):
        ids.sort(key=lambda sid: hash_value(['confirmation13-direction', sid]))
        for k, sid in enumerate(ids):
            out[sid] = {'definite': [rules.S, rules.R] if k % 2 == 0 else [rules.R, rules.S],
                        'uncertainty': list(uncertainty[k % 4]), 'stratum': list(stratum)}
    return out


def prepare():
    verify_tree(POOL)
    reservation = read(RESERVATION)
    if digest(POOL / 'manifest.json') != reservation['source_pool_manifest_sha256']:
        raise ValueError('Reserved pool was changed')
    wanted = reservation['candidates']
    ids = [r['source_id'] for r in wanted]
    if len(set(ids)) != 120:
        raise ValueError('Need original120 reserved IDs')
    all_sources = {r['source_id']: r for r in map(json.loads, (POOL / 'sources.private.jsonl').read_text(encoding='utf-8').splitlines())}
    selected = [all_sources[sid] for sid in ids]
    for row, source in zip(wanted, selected):
        if any(row[k] != source[k] for k in ('source_id', 'source_cluster', 'modality', 'length_band', 'split')):
            raise ValueError('Reservation metadata mismatch')
        if source['split'] != 'locked_test':
            raise ValueError('Not previously reserved holdout source')
    exposed = load_exposed_sources()
    if {s['source_cluster'] for s in selected} & {s['source_cluster'] for s in exposed}:
        raise ValueError('Reserved source cluster was already in development inputs')
    audit = near_duplicates(selected, exposed)
    if audit['exact_normalized_duplicate_pairs']:
        frozen(OUT / 'source_overlap_blocker.json', audit)
        raise ValueError('Normalized source duplicate found; do not silently replace reserved IDs')
    frozen(SOURCES / 'sources.private.jsonl', jsonl(selected))
    frozen(SOURCES / 'near_duplicate_audit.json', audit)
    manifest = {'kind': 'sealed-source-inputs-NOT-completed-executable-tasks',
                'source_clusters': 120, 'source_catalog': wanted,
                'modalities': dict(Counter(s['modality'] for s in selected)),
                'original_source_records': sum(len(s['documents']) for s in selected),
                'source_characters': {'minimum': min(s['characters'] for s in selected),
                                      'maximum': max(s['characters'] for s in selected)},
                'reservation_sha256': digest(RESERVATION),
                'pool_integrity_sha256': digest(POOL / 'build_integrity.json'),
                'sources_sha256': digest(SOURCES / 'sources.private.jsonl'),
                'current120_cluster_overlap': 0, 'completed_tasks': 0, 'solver_calls': 0,
                'clinical_expert_gold': False, 'public_release_authorized': False,
                'near_duplicate_flags': len(audit['near_duplicate_flags']),
                'source_holdout_not_rule_family_holdout': True}
    frozen(SOURCES / 'manifest.json', manifest)
    frozen(SOURCES / 'build_integrity.json', {p.name: digest(p) for p in sorted(SOURCES.glob('*')) if p.name != 'build_integrity.json'})
    frozen(WORK / 'protocol.json', protocol())
    frozen(WORK / 'direction_assignment.LOCAL_ONLY.json', assignment(wanted))
    old = ROOT.parent / 'output/medical12_www_gaps_20261022'
    preserved = {str((old / name).relative_to(ROOT.parent)).replace('\\', '/'): digest(old / name)
                 for name in ['final/delivery_manifest.json', 'final/results.json', 'RUN_STATUS.json']}
    preserved.update({'medical_scale_v06/datasets/medical09_dev120_silver_v1/build_integrity.json':
                      digest(ROOT / 'datasets/medical09_dev120_silver_v1/build_integrity.json'),
                      'medical_scale_v06/workflow11_inputs_private/actor_manifest.json':
                      digest(ROOT / 'workflow11_inputs_private/actor_manifest.json')})
    frozen(WORK / 'preserved_old_hashes.json', preserved)
    status(stage='sources_frozen_construction_pending', reserved_sources=120, source_inputs_frozen=True,
           tasks_frozen=False, near_duplicate_flags=len(audit['near_duplicate_flags']))
    print(json.dumps({'source_inputs_frozen': 120, 'completed_task_sources': 0,
                      'near_duplicate_flags': len(audit['near_duplicate_flags']), 'solver_calls': 0}))


def verify_plan():
    verify_tree(SOURCES)
    if read(WORK / 'protocol.json') != protocol():
        raise ValueError('Construction protocol/code changed; use a new version')
    for rel, expected in read(WORK / 'preserved_old_hashes.json').items():
        if digest(ROOT.parent / rel) != expected:
            raise ValueError('Previously sealed artifact changed')


def exact_evaluate(node, state, constants):
    """Separate implementation; does not call the construction evaluator."""
    def scalar(v):
        return Fraction(str(v)) if type(v) in (int, float) else v
    if isinstance(node, bool):
        return node
    if 'field' in node:
        return scalar(state[node['field']])
    if 'constant' in node:
        return scalar(constants[node['constant']]['value'])
    op = node['op']
    xs = [exact_evaluate(a, state, constants) for a in node['args']]
    if op == 'and':
        return False if any(x is False for x in xs) else None if any(x is None for x in xs) else True
    if op == 'or':
        return True if any(x is True for x in xs) else None if any(x is None for x in xs) else False
    if op == 'not':
        return None if xs[0] is None else not xs[0]
    if op == 'if':
        return xs[1] if xs[0] is True else xs[2] if xs[0] is False else xs[1] if xs[1] == xs[2] else None
    if any(x is None for x in xs):
        return None
    conversion = {'week_to_day': 7, 'day_to_hour': 24, 'g_to_mg': 1000, 'mg_to_ug': 1000, 'l_to_ml': 1000}
    if op in conversion:
        return xs[0] * conversion[op]
    if op == 'add':
        return sum(xs)
    if op == 'sub':
        return xs[0] - xs[1]
    if op == 'mul':
        value = Fraction(1)
        for x in xs:
            value *= x
        return value
    if op == 'div':
        if xs[1] == 0:
            raise ValueError('division by zero')
        return xs[0] / xs[1]
    if op == 'eq': return xs[0] == xs[1]
    if op == 'ne': return xs[0] != xs[1]
    if op == 'lt': return xs[0] < xs[1]
    if op == 'le': return xs[0] <= xs[1]
    if op == 'gt': return xs[0] > xs[1]
    if op == 'ge': return xs[0] >= xs[1]
    raise ValueError('Unsupported independent operation')


def independent_label(item, state):
    v = exact_evaluate(item['expression'], state, item['constant_map'])
    if v is not None and type(v) is not bool:
        raise ValueError('Nonboolean claim')
    return rules.U if v is None else rules.S if v else rules.R


def check_items(items, source):
    _, errors = rules.validate({'source_id': source['source_id'], 'items': items}, source)
    if errors:
        return False, errors
    balanced = [i for i in items if i['family_type'] != 'stable_insufficiency']
    if len(balanced) < 2:
        return False, ['need_two_distinct_balanced_rules']
    primary = [i for i in balanced if i.get('counterfactual_design_private', {}).get('primary_content_challenge')
               and i.get('counterfactual_design_private', {}).get('reference_hidden_from_stem')
               and operator_count(i['expression']) >= 3
               and sum(not OPERATIONAL.search(f['label']) for f in i['field_map'].values()) >= 2]
    if not primary:
        return False, ['no_nontrivial_source_reference_clinical_parameter_rule']
    for item in items:
        for state in item['state_values']:
            if independent_label(item, state) != rules.outcome(item['expression'], state, item['constant_map']):
                return False, ['independent_evaluator_disagreement']
    return True, []


def accept_quantitative(source):
    folder = WORK / 'sources' / source['source_id']
    if (folder / 'result.json').exists():
        return read(folder / 'result.json')
    items, report = quantitative.construct(source)
    if items is None:
        return None
    for item in items:
        item['construction_provenance_private'] = {'kind': 'deterministic_source_exact_regimen_parser',
            'source_id': source['source_id'], 'rule_index': items.index(item)}
    valid, errors = check_items(items, source)
    if not valid:
        frozen(folder / 'parser_rejection.json', {'errors': errors})
        return None
    frozen(folder / 'parsed_regimen.private.json', report['parsed'])
    frozen(folder / 'accepted_spec.private.json', items)
    row = {'source_id': source['source_id'], 'machine_valid': True,
           'construction_kind': 'deterministic_source_exact_quantitative',
           'family_count': len(items), 'author_requests': 0, 'solver_calls': 0, 'clinical_expert_gold': False}
    frozen(folder / 'result.json', row)
    return row


def qualify_proposals(source):
    folder = WORK / 'sources' / source['source_id']
    entries, failures = [], []
    for path in sorted(folder.glob('candidate_*/candidate.private.json')):
        raw = read(path)
        try:
            selected, errors = scoped.extract(raw, source)
            failures.extend(errors)
        except (ValueError, KeyError, TypeError) as error:
            failures.append({'error': str(error)[:220]})
            continue
        for clean, canonical in selected:
            proof = {'candidate_path': path.relative_to(ROOT).as_posix(), 'item_statement': canonical['statement']}
            try:
                item = refine(clean, canonical, proof, source)
                item['construction_provenance_private'] = proof
                item['nonredundant_operator_count_private'] = operator_count(item['expression'])
                entries.append((item, canonical, proof))
            except (ValueError, KeyError, TypeError, ArithmeticError) as error:
                failures.append({'error': str(error)[:220]})
    selected = bank_selector.select_bank(entries)
    if not selected:
        return None, failures[-12:] + [{'error': 'need_two_valid_balanced_families_one_multistep_and_one_persistent_gap'}]
    items = [i[0] for i in selected]
    valid, errors = check_items(items, source)
    return (items if valid else None), failures[-12:] + [{'error': e} for e in errors]


def compile_source(source, items, direction):
    """Compile matched static triplets and four workflow events from one base."""
    sid = source['source_id']
    instances = [copy.deepcopy(i) for i in items]
    balanced = [j for j, i in enumerate(instances) if i['family_type'] != 'stable_insufficiency']
    # Two changed families alone would leave only the gap. A distinct scene
    # reuses one valid rule as a protection probe; it is NOT a new rule family.
    protected = copy.deepcopy(instances[balanced[0]])
    protected['reused_family_index_private'] = balanced[0]
    protected_index = len(instances)
    instances.append(protected)
    order = sorted(range(len(instances)), key=lambda j: hash_value([sid, 'claim-order', j]))
    ids = {j: 'C' + str(k+1) for k, j in enumerate(order)}
    scenes = {j: '场景' + chr(65+k) for k, j in enumerate(order)}
    aids = {j: sid + '-SIM-' + hashlib.sha256((sid + '|scene|' + str(j)).encode()).hexdigest()[:10] for j in order}
    claims = [{'item_id': ids[j], 'claim_type': 'bounded_clinical_assertion',
               'statement': scenes[j] + '：' + instances[j]['statement']} for j in order]
    for claim in claims:
        j = next(j for j in order if ids[j] == claim['item_id'])
        if 'reference_locator_public' in instances[j]:
            claim['reference_locator'] = copy.deepcopy(instances[j]['reference_locator_public'])

    def state_for(j, label):
        item = instances[j]
        return copy.deepcopy(next(s for s in item['state_values'] if independent_label(item, s) == label))

    def materialize(states, token):
        chart, gold, proof = [], [], []
        for j in order:
            item, state = instances[j], states[j]
            lines = []
            for key, v in state.items():
                f = item['field_map'][key]
                rendered = '本轮值未记录；不能用历史值替代' if v is None else '是' if v is True else '否' if v is False else str(v)
                lines.append(f['label'] + ('（' + f['unit'] + '）' if f['unit'] else '') + '：' + rendered)
            chart.append({'anchor_id': aids[j], 'label': '独立模拟' + scenes[j],
                          'origin_kind': 'explicit_synthetic_chartlet',
                          'text': '独立模拟' + scenes[j] + '。\n' + '\n'.join(lines)})
            refs = list(dict.fromkeys([r['anchor_id'] for r in item['references']] + [c['anchor_id'] for c in item['constant_map'].values()]))
            label = independent_label(item, state)
            gold.append({'item_id': ids[j], 'label': label, 'support_anchor_ids': [aids[j]] + refs,
                         'sufficient_evidence_groups': [[aids[j]] + refs]})
            proof.append({'item_id': ids[j], 'author_index': j, 'state': state, 'expression': item['expression'],
                          'constants': item['constant_map'], 'fields': item['field_map'], 'label': label,
                          'persistent_insufficiency': item['family_type'] == 'stable_insufficiency',
                          'reused_family_index_private': item.get('reused_family_index_private'),
                          'source_quotes': item['references'], 'independent_evaluator_agrees': True})
        docs = copy.deepcopy(source['documents']) + [{'document_id': sid + '-SIMULATED-CHART',
                'title': '明确模拟：独立临床交接状态（不是原始随访）',
                'origin_kind': 'explicit_synthetic_chartlets', 'blocks': chart}]
        visible = [b['anchor_id'] for d in docs for b in d['blocks']]
        controller = {'source_id': sid, 'modality': source['modality'],
            'environment': 'E2' if source['modality'] == 'dialogue' else 'E3', 'key_revokes_write': False,
            'task_id': sid + '__confirmation13_clinical_state', 'task_version': 'confirmation13',
            'documents': docs, 'claims': copy.deepcopy(claims),
            'scenario_contract': '每个字母场景是不同的独立模拟患者。当前患者参数只来自相应模拟状态；原文仅供核对规则或命名历史参照。缺失不是阴性。仅核对陈述的有界含义，不据此向真实患者给治疗建议。',
            'visibility': {'initial': visible, 'updated': visible}}
        return {'packet_id': sid + '__c13_' + hash_value([sid, token])[:12],
                'controller': clean_controller(controller), 'items': gold, 'proofs_private': proof}

    packets = []
    for packet in range(3):
        states = {}
        for j in order:
            item = instances[j]
            if item['family_type'] == 'stable_insufficiency':
                states[j] = copy.deepcopy(item['state_values'][packet])
            else:
                permutation = sorted(range(3), key=lambda k: hash_value([sid, 'packet-state', j, k]))
                states[j] = state_for(j, LABELS[permutation[packet]])
        packets.append(materialize(states, ['static', packet]))
    initial_states = {}
    for j in order:
        initial_states[j] = (copy.deepcopy(instances[j]['state_values'][0]) if instances[j]['family_type'] == 'stable_insufficiency'
                             else state_for(j, LABELS[int(hash_value([sid, j, 'initial'])[:8], 16) % 2]))
    first, second = balanced[:2]
    initial_states[first] = state_for(first, direction['definite'][0])
    initial_states[second] = state_for(second, direction['uncertainty'][0])
    # The protector stays a known S/R value, never a known-false named flag.
    initial_states[protected_index] = state_for(protected_index, LABELS[int(hash_value([sid, 'protected'])[:8], 16) % 2])
    initial = materialize(initial_states, 'initial')
    events = []
    for event in EVENTS:
        states = copy.deepcopy(initial_states)
        if event in ('key_mixed', 'single_definite'):
            states[first] = state_for(first, direction['definite'][1])
        if event in ('key_mixed', 'single_uncertainty'):
            states[second] = state_for(second, direction['uncertainty'][1])
        events.append({'event_id': event, **materialize(states, ['event', event])})
    return {'source_id': sid, 'source_cluster': source['source_cluster'], 'modality': source['modality'],
            'unique_authored_families': len(items), 'claim_instances_per_packet': len(instances),
            'protection_probe_is_reused_family_not_new_independent_case': True,
            'packets': packets, 'initial': initial, 'events': events}


def verify_compiled(source, bundle):
    controllers = bundle['packets'] + [bundle['initial']] + bundle['events']
    stems = [p['controller']['claims'] for p in controllers]
    if not all(s == stems[0] for s in stems):
        raise ValueError('Claim changed across truth states')
    originals = source['documents']
    for p in controllers:
        c = p['controller']
        if c['documents'][:-1] != originals:
            raise ValueError('Original source roundtrip failed')
        if any(k in c for k in ('gold', 'proofs_private', 'changes_private', 'labels')):
            raise ValueError('Target in actor controller')
        aids = {b['anchor_id'] for d in c['documents'] for b in d['blocks']}
        if len(aids) != sum(len(d['blocks']) for d in c['documents']):
            raise ValueError('Anchor ID collision')
        for g, proof in zip(p['items'], p['proofs_private']):
            if independent_label({'expression': proof['expression'], 'constant_map': proof['constants']}, proof['state']) != g['label']:
                raise ValueError('Compiled independent label mismatch')
            if not all(set(group) <= aids for group in g['sufficient_evidence_groups']):
                raise ValueError('Missing current-state evidence anchor')
    for j, proof in enumerate(bundle['packets'][0]['proofs_private']):
        labels = [p['items'][j]['label'] for p in bundle['packets']]
        if proof['persistent_insufficiency']:
            if labels != [rules.U] * 3:
                raise ValueError('Persistent-gap control determined')
        elif Counter(labels) != Counter(LABELS):
            raise ValueError('Same-stem triplet not balanced')
    initial = {g['item_id']: g['label'] for g in bundle['initial']['items']}
    for event in bundle['events']:
        end = {g['item_id']: g['label'] for g in event['items']}
        changed = [iid for iid in initial if initial[iid] != end[iid]]
        expected = 2 if event['event_id'] == 'key_mixed' else 0 if event['event_id'] == 'neutral' else 1
        if len(changed) != expected:
            raise ValueError('Wrong selective-change count')
        if event['event_id'] == 'key_mixed' and not any(initial[i] == end[i] != rules.U for i in initial):
            raise ValueError('No unchanged known claim')
    return True


def freeze_tasks():
    verify_plan()
    sources = load_reserved()
    results = {s['source_id']: read(WORK / 'sources' / s['source_id'] / 'result.json')
               for s in sources if (WORK / 'sources' / s['source_id'] / 'result.json').exists()}
    qualified = [sid for sid, row in results.items() if row['machine_valid']]
    if len(qualified) != 120:
        status(stage='construction_incomplete_not_task_frozen', reserved_sources=120,
               construction_accounted=len(results), qualified_sources=len(qualified), tasks_frozen=False)
        return {'tasks_frozen': False, 'qualified_sources': len(qualified), 'accounted': len(results), 'reserved': 120}
    if DATASET.exists():
        verify_tree(DATASET)
        return read(DATASET / 'construction_audit.json')
    directions = read(WORK / 'direction_assignment.LOCAL_ONLY.json')
    actors, targets, totals, transition = [], [], Counter(), Counter()
    templates, families = Counter(), Counter()
    def rule_shape(expression, fields, constants):
        def shape(node):
            if type(node) is bool: return ['bool']
            if 'field' in node: return ['field', fields[node['field']]['type']]
            if 'constant' in node: return ['source_constant', type(constants[node['constant']]['value']).__name__]
            return [node['op'], [shape(a) for a in node['args']]]
        return hash_value(shape(expression))
    old_shapes = set()
    old_dataset = ROOT / 'datasets/medical09_dev120_silver_v1'
    for case in read(old_dataset / 'manifest.json')['cases']:
        if case['configuration_order_private'] == 0:
            for proof in read(old_dataset / 'cases' / case['task_id'] / 'proofs.private.json'):
                old_shapes.add(rule_shape(proof['expression'], proof['fields'], proof['constants']))
    for source in sources:
        sid = source['source_id']
        items = read(WORK / 'sources' / sid / 'accepted_spec.private.json')
        valid, errors = check_items(items, source)
        if not valid:
            raise ValueError('Qualified rule mutated: ' + sid + ': ' + str(errors))
        bundle = compile_source(source, items, directions[sid])
        verify_compiled(source, bundle)
        frozen(DATASET / 'local_private' / (sid + '.private.json'), bundle)
        initial = bundle['initial']
        actors.append({'source_id': sid, 'initial': initial['controller'],
                       'events': [{'event_id': e['event_id'], 'controller': e['controller']} for e in bundle['events']]})
        targets.append({'source_id': sid, 'source_cluster': source['source_cluster'],
                        'adaptive_calibration_reused': False, 'source_split': 'confirmation_source_holdout',
                        'modality': source['modality'], 'environment': initial['controller']['environment'],
                        'construction_subset': results[sid]['construction_kind'], 'initial': initial['items'],
                        'events': [{'event_id': e['event_id'], 'items': e['items']} for e in bundle['events']]})
        b = {g['item_id']: g['label'] for g in initial['items']}
        for e in bundle['events']:
            for g in e['items']:
                if b[g['item_id']] != g['label']:
                    transition[(e['event_id'], b[g['item_id']], g['label'])] += 1
        totals['unique_authored_families'] += len(items)
        totals['claim_instances'] += bundle['claim_instances_per_packet']
        totals['primary_static_judgments'] += sum(len(p['items']) for p in bundle['packets'])
        totals['original_source_records'] += len(source['documents'])
        families[results[sid]['construction_kind']] += 1
        for item in items:
            templates[rule_shape(item['expression'], item['field_map'], item['constant_map'])] += 1
    frozen(DATASET / 'actors.private.jsonl', jsonl(actors))
    frozen(DATASET / 'evaluator_targets.LOCAL_ONLY.private.jsonl', jsonl(targets))
    frozen(DATASET / 'protocol.json', read(WORK / 'protocol.json'))
    frozen(DATASET / 'source_snapshot_manifest.json', read(SOURCES / 'manifest.json'))
    actor_manifest = {'kind': 'confirmation13-gold-free-actor-bundle', 'sources': 120, 'events': 480,
                      'files': {'actors.private.jsonl': digest(DATASET / 'actors.private.jsonl')},
                      'protocol_sha256': digest(DATASET / 'protocol.json'), 'private_source_text': True,
                      'no_evaluator_target_in_actor_manifest': True, 'public_release_authorized': False}
    frozen(DATASET / 'actor_manifest.json', actor_manifest)
    audit = {'kind': 'confirmation13-reserved120-mechanical-silver-seal', 'task_sources': 120,
             'source_clusters': 120, 'primary_static_packets': 360, 'workflow_initial_states': 120,
             'workflow_event_states': 480, **totals, 'construction_subsets': dict(families),
             'transition_counts': [{'event': e, 'from': a, 'to': b, 'count': n} for (e,a,b), n in sorted(transition.items())],
             'distinct_anonymous_rule_shapes': len(templates),
             'rule_shapes_also_seen_in_development': len(set(templates) & old_shapes),
             'authored_families_with_seen_rule_shape': sum(n for shape,n in templates.items() if shape in old_shapes),
             'source_holdout_not_rule_family_holdout': True, 'clinical_expert_gold': False,
             'solver_calls': 0, 'tasks_frozen': True, 'private_data_public_release_authorized': False,
             'not_independent_patient_or_hospital_count': True}
    frozen(DATASET / 'construction_audit.json', audit)
    frozen(DATASET / 'rule_shape_overlap_audit.json', {'anonymous_shapes': dict(templates),
        'previous_development_shapes': sorted(old_shapes), 'shared_shapes': sorted(set(templates) & old_shapes),
        'not_a_semantic_rule_novelty_or_training_contamination_certificate': True})
    frozen(DATASET / 'verification.json', {'source_roundtrips': True, 'balanced_same_stem_triplets': True,
        'independent_evaluator_agrees': True, 'all_six_transition_directions': True,
        'mixed_update_has_unchanged_known_and_unknown': True, 'all_current_evidence_anchors_exist': True,
        'all120_reserved_sources_used_no_replacements': True, 'no_solver_calls': True,
        'clinical_validation': False, 'rule_family_disjoint': False})
    files = {p.relative_to(DATASET).as_posix(): digest(p) for p in sorted(DATASET.rglob('*')) if p.is_file()}
    frozen(DATASET / 'build_integrity.json', files)
    frozen(OUT / 'delivery_manifest.json', {'dataset_path': str(DATASET),
        'dataset_integrity_sha256': digest(DATASET / 'build_integrity.json'),
        'source_integrity_sha256': digest(SOURCES / 'build_integrity.json'),
        'protocol_sha256': digest(WORK / 'protocol.json'), 'tasks_frozen': True, 'solver_calls': 0})
    status(stage='confirmation_tasks_frozen_no_solver_evaluation', tasks_frozen=True,
           qualified_sources=120, construction_accounted=120, reserved_sources=120)
    frozen(OUT / 'RESULTS.md', '# 新120来源确认集构建与封存\n\n'
        + '已按原预留清单构建120个来源簇；旧开发集和结果不变。生成规则草案不等于求解评测，求解调用为0。\n\n'
        + '这是来源隔离、程序核验的模拟状态银标，不是医学专家金标，不代表独立患者数，也不代表规则族隔离。医学有效性复核尚未完成。\n\n'
        + '包括360份同问法三态静态资料包、120个真实求解前初始状态定义、480个事件状态定义。工作流起点尚未由模型提交；后续必须从真实初始提交复制分支。\n\n'
        + '原文与标签仅保留在内部目录。仅actors.private.jsonl及actor_manifest.json可作为后续内部求解输入，evaluator_targets.LOCAL_ONLY.private.jsonl和local_private不得上传求解端。\n\n'
        + '构建统计：\n\n```json\n' + json.dumps(audit, ensure_ascii=False, indent=2) + '\n```\n')
    verify_tree(DATASET)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'quantitative', 'freeze', 'verify'])
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'quantitative':
        verify_plan()
        for source in load_reserved():
            accept_quantitative(source)
        rows = [read(p) for p in sorted((WORK / 'sources').glob('*/result.json'))]
        print(json.dumps({'deterministic_qualified_sources': sum(r['machine_valid'] for r in rows), 'reserved': 120}))
    elif args.action == 'freeze':
        print(json.dumps(freeze_tasks(), ensure_ascii=True))
    else:
        verify_plan()
        if DATASET.exists(): verify_tree(DATASET)
        print(json.dumps({'source_seal_valid': True, 'task_seal_valid': DATASET.exists(), 'solver_calls': 0}))


if __name__ == '__main__':
    main()
