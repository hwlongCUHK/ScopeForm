"""Rule-level construction: retain valid source-specific families, not bad packs."""
import copy
from collections import Counter
from itertools import product
import json
import re
import shutil
import scale09c as flat
import scale09_rules_r3 as rules
from run_generation import ROOT, read, dump, digest
import compile_scale09 as compiler
compiler.outcome = rules.outcome
from compile_scale09 import compile_source

WORK = ROOT / 'construction/scale09_dev120_r3'


def normalize_graph(item, source):
    item = copy.deepcopy(item)
    blocks = {b['anchor_id']: b['text'] for d in source['documents'] for b in d['blocks']}
    existing = {n['id'] for n in item['nodes']}
    if len(existing) != len(item['nodes']): raise ValueError('duplicate_node_id')
    conversions = []
    # Exact calendar-invariant / SI conversions only. Months, years, vague
    # "half a month", age group thresholds and inferred medical cutoffs are NOT
    # converted. Every recovered native value still needs its actual witness.
    conversions_allowed = [({'天', 'day', 'days', 'd'}, r'(\d+(?:\.\d+)?|[一二两三四五六七八九十]+)\s*(?:周|星期|weeks?)', 7, 'week_to_day', '周'),
        ({'mg', '毫克'}, r'(\d+(?:\.\d+)?)\s*(?:g|克)(?!/)', 1000, 'g_to_mg', 'g'),
        ({'ug', 'μg', '微克'}, r'(\d+(?:\.\d+)?)\s*(?:mg|毫克)', 1000, 'mg_to_ug', 'mg'),
        ({'ml', 'mL', '毫升'}, r'(\d+(?:\.\d+)?)\s*(?:L|升)', 1000, 'l_to_ml', 'L')]
    for ci, c in enumerate(item['constants']):
        if type(c['value']) not in (int, float): continue
        text = blocks[c['anchor_id']]
        direct = [float(n) for n in re.findall(r'(?<![\d.])-?\d+(?:\.\d+)?', text.replace(',', ''))] + rules.chinese_cardinals(text)
        if float(c['value']) in direct: continue
        found = []
        for units, pattern, factor, op, native_unit in conversions_allowed:
            if c['unit'] not in units: continue
            for match in re.finditer(pattern, text):
                raw = match.group(1); values = rules.chinese_cardinals(raw) if not raw[0].isdigit() else [float(raw)]
                if len(values) == 1 and abs(values[0] * factor - c['value']) < 1e-9:
                    found.append((values[0], native_unit, op))
        found = sorted(set(found))
        if len(found) == 1:
            value, unit, op = found[0]; nid = 'unit_conversion_' + str(ci)
            if nid in existing: raise ValueError('conversion_node_id_collision')
            for node in item['nodes']:
                node['args'] = ['node:' + nid if a == 'const:' + c['key'] else a for a in node['args']]
            conversions.append({'id': nid, 'op': op, 'args': ['const:' + c['key']]})
            c['value'], c['unit'] = value, unit
    nodes = {n['id']: n for n in conversions + item['nodes']}
    fields = {f['key'] for f in item['fields']}; consts = {c['key'] for c in item['constants']}
    for node in nodes.values():
        for i, ref in enumerate(node['args']):
            if ':' not in ref: continue
            prefix, name = ref.split(':', 1)
            # Repair a namespace tag only when the EXACT symbol exists in a
            # single namespace. Never infer a different medical variable name.
            namespaces = [p for p, names in [('field', fields), ('const', consts), ('node', set(nodes))] if name in names]
            if len(namespaces) == 1 and prefix != namespaces[0]: node['args'][i] = namespaces[0] + ':' + name
    ordered, active, visiting = [], set(), set()
    def visit(nid):
        if nid in visiting: raise ValueError('cyclic_rule_graph')
        if nid in active: return
        visiting.add(nid)
        for arg in nodes[nid]['args']:
            if arg.startswith('node:'): visit(arg[5:])
        visiting.remove(nid); active.add(nid); ordered.append(nodes[nid])
    visit(item['root']); item['nodes'] = ordered
    return item


def choose_triplet(converted, native):
    expression = json.loads(converted['expression_json'])
    fields = {f['key']: f for f in converted['fields']}
    constants = {c['key']: {**c, 'value': json.loads(c['value_json'])} for c in converted['constants']}
    used = {n['field'] for n in rules.walk(expression) if isinstance(n, dict) and 'field' in n}
    used_c = {n['constant'] for n in rules.walk(expression) if isinstance(n, dict) and 'constant' in n}
    if not used <= set(fields): raise ValueError('undefined_field_symbol')
    converted['fields'] = [f for f in converted['fields'] if f['key'] in used]
    converted['constants'] = [c for c in converted['constants'] if c['key'] in used_c]
    constants = {k: c for k, c in constants.items() if k in used_c}
    if converted['family_type'] == 'stable_insufficiency':
        converted['states'] = [[v for v in state if v['key'] in used] for state in converted['states']]
        return converted
    keys = [f['key'] for f in converted['fields']]
    domains = {f['key']: list(dict.fromkeys(json.dumps(v, ensure_ascii=False) for v in f['values'])) + ['null'] for f in native['fields'] if f['key'] in used}
    groups = {rules.S: [], rules.R: [], rules.U: []}
    states = []; labels = []
    for values in product(*(domains[k] for k in keys)):
        state = {k: json.loads(v) for k, v in zip(keys, values)}
        try: label = rules.outcome(expression, state, constants)
        except (ValueError, ArithmeticError): continue
        idx = len(states); states.append(state); labels.append(label); groups[label].append(idx)
    def rank(i): return (sum(v is None for v in states[i].values()), json.dumps(states[i], sort_keys=True, ensure_ascii=False))
    candidates = {}
    for label, indices in groups.items():
        if not indices: raise ValueError('no_' + label + '_in_proposed_domains')
        ordered = sorted(indices, key=rank)
        candidates[label] = ordered[:12] + [i for i in ordered[12:] if i in ordered[::max(1, len(ordered) // 12)]][:12]
    # Find a compact, actual causal contrast rather than choosing lexically
    # unrelated S and R states, which can hide the role of a valid OR branch.
    pairs = sorted(product(candidates[rules.S], candidates[rules.R]), key=lambda ab: (sum(states[ab[0]][k] != states[ab[1]][k] for k in keys), rank(ab[0]), rank(ab[1])))
    changes = {}
    for i in set(j for xs in candidates.values() for j in xs):
        changes[i] = {}
        for k in keys:
            targets = set()
            for raw in domains[k]:
                try:
                    if rules.outcome(expression, {**states[i], k: json.loads(raw)}, constants) != labels[i]: targets.add(raw)
                except (ValueError, ArithmeticError): pass
            changes[i][k] = targets
    for a, b in pairs:
        for c in candidates[rules.U]:
            selected = [a, b, c]
            if all(any(changes[i][k] & {json.dumps(states[j][k], ensure_ascii=False) for j in selected} for i in selected) for k in keys):
                converted['states'] = [[{'key': k, 'value_json': json.dumps(v, ensure_ascii=False)} for k, v in states[i].items()] for i in selected]
                return converted
    raise ValueError('no_triplet_exercises_every_clinical_input')


def extract(candidate, source):
    # Only this process's module bindings change; earlier version code/files
    # and running processes are untouched.
    if candidate.get('source_id') != source['source_id']: return [], [{'error': 'source_id_mismatch'}]
    flat.infer = rules.infer; flat.outcome = rules.outcome
    selected, errors = [], []
    for index, raw in enumerate(candidate.get('items', [])):
        try:
            native = normalize_graph(raw, source)
            converted = flat.canonical({'source_id': source['source_id'], 'items': [native]}, source)['items'][0]
            converted = choose_triplet(converted, native)
            clean, failures = rules.validate({'source_id': source['source_id'], 'items': [converted]}, source, single=True)
            if failures: raise ValueError('; '.join(failures))
            selected.append((clean[0], converted))
        except (KeyError, TypeError, ValueError, ArithmeticError) as e: errors.append({'proposed_item': index, 'error': str(e)[:240]})
    return selected, errors


def select_bank(entries):
    seen, stems, balanced, gaps = set(), set(), [], []
    for clean, canonical, provenance in entries:
        def semantic_shape(node):
            if type(node) is bool: return node
            if 'field' in node:
                f = clean['field_map'][node['field']]
                return ['clinical_input', f['type'], f['unit']]
            if 'constant' in node:
                c = clean['constant_map'][node['constant']]
                return ['source_constant', c['anchor_id'], c['value'], c['unit']]
            args = [semantic_shape(x) for x in node['args']]
            if node['op'] in ('and', 'or', 'add', 'mul', 'eq', 'ne'): args.sort(key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False))
            return [node['op'], args]
        # Conservative within-source deduplication: renaming fields or
        # paraphrasing the same rule cannot create another clinical family.
        identity = json.dumps(semantic_shape(clean['expression']), ensure_ascii=False, sort_keys=True)
        if identity in seen or clean['statement'] in stems: continue
        seen.add(identity); stems.add(clean['statement'])
        if clean['family_type'] == 'stable_insufficiency': gaps.append((clean, canonical, provenance))
        else: balanced.append((clean, canonical, provenance))
    # Source-richness is allowed to vary; no weak or invented third rule is
    # included just to make every source provide an identical question count.
    if len(balanced) < 2 or not gaps or not any(x[0]['family_type'] == 'multi_step' for x in balanced): return None
    first_multi = next(x for x in balanced if x[0]['family_type'] == 'multi_step')
    kept = [first_multi] + [x for x in balanced if x is not first_multi][:3] + gaps[:1]
    if not any(x[0]['family_type'] == 'multi_step' for x in kept): return None
    return kept


def messages(source, errors=None):
    flat.WORK = WORK
    msg = flat.messages(source, None)
    msg[0]['content'] += '''\n补充构建约束：原文“年轻/一般/可能”不能凭常识变成40岁等数值阈值。“两周”是2周，优先让字段和常量使用同一原生单位；不得把“半个月”硬定成14或15天。未知控制可以是一个真实关键检查缺失，不必为它添加不影响结论的装饰参数。程序会分别保留合格规则，整个提案不要求每一项同时合格；但每个保留的balanced都必须来源依赖，且至少有两个实际影响结论的临床参数。'''
    if errors:
        old = sorted((WORK / 'sources' / source['source_id']).glob('candidate_*/candidate.private.json'))
        msg[0]['content'] += '\n上一提案构建错误：' + json.dumps(errors, ensure_ascii=False)
        if old: msg[0]['content'] += '\n上一提案：' + json.dumps(read(old[-1]), ensure_ascii=False)
    return msg


schema = flat.schema

def build(protocol, sources, accepted, ignored_output):
    output = ROOT / 'datasets/medical09_dev120_silver_r3'
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
    dump(output / 'manifest.json', {'kind': 'scale09-r3-development-source-specific-silver-triplets', 'source_ids': ids, 'cases': cases,
                                  'clinical_expert_gold': False, 'protocol_sha256': digest(WORK / 'protocol.json'),
                                  'old_dataset_integrity_sha256': protocol['old_dataset_integrity_sha256'],
                                  'construction_revision': 'flat_source_rule_graphs_programmatic_state_search'})
    dump(output / 'build_integrity.json', {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob('*.json'))})
    return audit




def bank(source):
    paths = []
    for revision in ('scale09_dev120_r2', 'scale09_dev120_r3'):
        paths += sorted((ROOT / 'construction' / revision / 'sources' / source['source_id']).glob('candidate_*/candidate.private.json'))
    entries, errors = [], []
    for path in paths:
        proposed = read(path)
        selected, failures = extract(proposed, source)
        errors += [{'proposal': path.parent.name, **e} for e in failures]
        for clean, canonical in selected:
            entries.append((clean, canonical, {'candidate_path': path.relative_to(ROOT).as_posix(), 'item_statement': canonical['statement']}))
    return select_bank(entries), errors


def validate(candidate, source):
    selected, errors = bank(source)
    if selected:
        clean = [x[0] for x in selected]
        for item, _, provenance in selected: item['construction_provenance_private'] = provenance
        return clean, []
    return None, ['Need >=2 source-dependent balanced families, >=1 actual multi-step, and one stable insufficiency control.'] + [json.dumps(e, ensure_ascii=False) for e in errors[-10:]]


def adopt(source):
    from run_generation import dump
    folder = WORK / 'sources' / source['source_id']
    if (folder / 'result.json').exists(): return read(folder / 'result.json')
    clean, errors = validate(None, source)
    if clean is None: return None
    dump(folder / 'accepted_spec.private.json', clean)
    row = {'source_id': source['source_id'], 'source_cluster': source['source_cluster'], 'modality': source['modality'],
           'length_band': source['length_band'], 'status': 'machine_valid_silver_adopted_prior_construction', 'machine_valid': True,
           'first_candidate_valid': False, 'selected_candidate': 'prior_rule_bank', 'families': len(clean),
           'balanced_triplets': len(clean) - 1, 'multi_step': sum(x['family_type'] == 'multi_step' for x in clean),
           'history': [], 'seconds': 0, 'new_model_requests_for_adoption': 0, 'not_clinical_gold': True}
    dump(folder / 'result.json', row)
    return row
