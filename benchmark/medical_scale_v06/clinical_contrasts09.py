"""Favor reference-dependent clinical counterfactuals, not execution-flag flips."""
import copy
from itertools import product, combinations
import json
import math
import re
from run_generation import ROOT, read
import scale09_rules_r4 as rules

OPERATIONAL = re.compile(r'是否(?:已|已经)?(?:实际)?(?:服用|用药|执行|实施|完成|授权|保存|提交|落实)|是否.*(?:符合|一致|达到标准)|(?:实际|已)(?:服药|用药|执行|实施|完成)')
MISSING_WORDS = {'未知', '未记录', '未提供', '不清楚', '不知道', '缺失', '其他', '其他药物', '另一种药物', '不符合', '符合'}


def unit(value):
    value = value.strip().lower().replace(' ', '')
    return {'毫克': 'mg', '克': 'g', '毫升': 'ml', 'days': '天', 'day': '天', 'd': '天', 'weeks': '周', 'week': '周', '小时': 'h'}.get(value, value)


def refine(clean, canonical, provenance, source):
    if clean['family_type'] == 'stable_insufficiency':
        out = copy.deepcopy(clean); out['counterfactual_design_private'] = {'stratum': 'persistent_insufficiency', 'primary_content_challenge': False}
        return out
    raw = read(ROOT / provenance['candidate_path'])
    native = next(i for i in raw['items'] if i['statement'] == provenance['item_statement'])
    fields = clean['field_map']; constants = clean['constant_map']; expr = clean['expression']
    domains = {f['key']: [v for v in f['values'] if type(v) is not str or v not in MISSING_WORDS] for f in native['fields'] if f['key'] in fields}
    if any(not values for values in domains.values()):
        out = copy.deepcopy(clean); out['counterfactual_design_private'] = {'stratum': 'unresolved_domain_placeholder', 'primary_content_challenge': False}; return out
    stem_numbers = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', clean['statement'])] + rules.chinese_cardinals(clean['statement'])
    hidden = set()
    for key, c in constants.items():
        v = c['value']
        if type(v) in (int, float) and float(v) not in stem_numbers: hidden.add(key)
        if type(v) is str and v not in clean['statement'] and (len(v) >= 2 or v.isascii()): hidden.add(key)
    associated, hidden_fields = {}, set()
    for node in rules.walk(expr):
        if not isinstance(node, dict) or node.get('op') not in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'): continue
        ns = list(rules.walk(node)); cs = {n['constant'] for n in ns if isinstance(n, dict) and 'constant' in n and type(constants[n['constant']]['value']) is not bool}
        if not cs: continue
        fs = {n['field'] for n in ns if isinstance(n, dict) and 'field' in n}
        for f in fs:
            if OPERATIONAL.search(fields[f]['label']): continue
            associated.setdefault(f, set()).update(cs)
            if cs & hidden: hidden_fields.add(f)
    keys = list(fields); states = []
    for values in product(*(domains[k] for k in keys)):
        state = dict(zip(keys, values))
        try:
            if rules.outcome(expr, state, constants) == rules.S: states.append(state)
        except (ValueError, ArithmeticError): pass
    states.sort(key=lambda s: json.dumps(s, ensure_ascii=False, sort_keys=True))
    seeds = states[:8] + states[8:][::max(1, len(states) // 8)][:8]
    best = None
    for base in seeds:
        for f in sorted(associated, key=lambda k: (k not in hidden_fields, fields[k]['type'] != 'number', k)):
            cs = associated[f]
            if base[f] is None: continue
            options = list(domains[f])
            if fields[f]['type'] == 'number':
                discrete = bool(re.search(r'片|粒|次|瓶|支|天|日|岁', fields[f]['unit'])) and all(type(v) is int for v in domains[f])
                delta = max(1 if discrete else .01, abs(float(base[f])) * .10)
                options += [float(base[f]) - delta, float(base[f]) + delta]
                for c in cs:
                    if type(constants[c]['value']) in (int, float) and unit(constants[c]['unit']) == unit(fields[f]['unit']):
                        center = float(constants[c]['value']); step = max(1 if discrete else .01, abs(center) * .10)
                        options += [center, center - step, center + step]
                options = [round(v) if discrete else float(f'{v:.8g}') for v in options if math.isfinite(v)]
                # Relative dates may legitimately precede a reference event;
                # physical negative quantities are not added by this design.
                if not re.search(r'相对|偏移|差值|前后', fields[f]['label']): options = [v for v in options if v >= 0]
            options = list(dict.fromkeys(json.dumps(v, ensure_ascii=False) for v in options))
            local = []
            for value in options:
                state = {**base, f: json.loads(value)}
                try: label = rules.outcome(expr, state, constants)
                except (ValueError, ArithmeticError): continue
                if label in (rules.S, rules.R): local.append((label, state))
            pairs = [(s[1], r[1]) for s in local if s[0] == rules.S for r in local if r[0] == rules.R]
            def pair_distance(pair):
                a, b = pair
                return abs(float(a[f]) - float(b[f])) / max(1, abs(float(a[f])), abs(float(b[f]))) if fields[f]['type'] == 'number' else 1
            pairs.sort(key=lambda p: (pair_distance(p), json.dumps(p, ensure_ascii=False, sort_keys=True)))
            for supported, refuted in pairs[:24]:
                    if supported[f] == refuted[f]: continue
                    unknowns = []
                    order = [f] + [k for k in associated if k != f] + [k for k in keys if k not in associated]
                    for count in (1, 2):
                        for missing in combinations(order, count):
                            state = {**supported, **{k: None for k in missing}}
                            if rules.outcome(expr, state, constants) == rules.U: unknowns.append(state)
                        if unknowns: break
                    unknowns += [s for s in clean['state_values'] if rules.outcome(expr, s, constants) == rules.U]
                    for unknown in unknowns[:16]:
                        version = copy.deepcopy(canonical)
                        version['states'] = [[{'key': k, 'value_json': json.dumps(v, ensure_ascii=False)} for k, v in s.items()] for s in (supported, refuted, unknown)]
                        checked, errors = rules.validate({'source_id': source['source_id'], 'items': [version]}, source, single=True)
                        if errors: continue
                        distance = (abs(float(supported[f]) - float(refuted[f])) / max(1, abs(float(supported[f])), abs(float(refuted[f])))) if fields[f]['type'] == 'number' else 1
                        priority = (f not in hidden_fields, fields[f]['type'] != 'number', distance, json.dumps(version['states'], sort_keys=True, ensure_ascii=False))
                        if best is None or priority < best[0]:
                            item = checked[0]
                            for meta in ('task_scope_private', 'construction_provenance_private'):
                                if meta in clean: item[meta] = clean[meta]
                            item['counterfactual_design_private'] = {'stratum': 'hidden_reference_clinical_parameter' if f in hidden_fields else 'provided_constraint_clinical_parameter',
                                'primary_content_challenge': True, 'reference_hidden_from_stem': f in hidden_fields,
                                'known_pair_changed_fields': [f], 'changed_field_type': fields[f]['type'],
                                'execution_flags_held_fixed': True, 'same_medical_rule_and_statement': True,
                                'simulated_numeric_neighbors_may_be_added': fields[f]['type'] == 'number', 'chosen_using_solver_results': False}
                            best = (priority, item)
                            if f in hidden_fields: return item
    if best: return best[1]
    out = copy.deepcopy(clean)
    out['counterfactual_design_private'] = {'stratum': 'boolean_or_unresolved_contrast_control', 'primary_content_challenge': False,
        'reason': 'No validated source-reference parameter contrast found under the fixed construction search; not a primary model-content challenge.'}
    return out
