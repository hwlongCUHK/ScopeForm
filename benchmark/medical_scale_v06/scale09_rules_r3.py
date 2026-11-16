"""Revision 3: preserve strict balanced-claim checks, exact unit arithmetic and singleton stable-gap controls. Earlier rule files stay frozen."""
"""Closed, typed, three-valued construction DSL. Never evaluates Python code."""
import json
import math
import re
from fractions import Fraction

S, R, U = 'supported', 'refuted', 'not_determined'
TYPES = {'number', 'boolean', 'text'}
ARITY = {'not': (1, 1), 'if': (3, 3), 'and': (2, 6), 'or': (2, 6),
         'add': (2, 6), 'mul': (2, 6), 'sub': (2, 2), 'div': (2, 2),
         'eq': (2, 2), 'ne': (2, 2), 'lt': (2, 2), 'le': (2, 2),
         'gt': (2, 2), 'ge': (2, 2)}


UNIT_FACTORS = {'week_to_day': 7, 'day_to_hour': 24, 'g_to_mg': 1000, 'mg_to_ug': 1000, 'l_to_ml': 1000}
ARITY.update({name: (1, 1) for name in UNIT_FACTORS})


def chinese_cardinals(text):
    values = []
    digits = {'零': 0, '〇': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
    for raw in re.findall(r'[零〇一二两三四五六七八九十]{1,3}', text):
        if '十' in raw:
            a, b = raw.split('十', 1)
            if len(a) <= 1 and len(b) <= 1:
                values.append((digits.get(a, 1) if a else 1) * 10 + (digits.get(b, 0) if b else 0))
        elif len(raw) == 1:
            values.append(digits[raw])
    return values


def scalar(value):
    if value is None or type(value) in (str, bool):
        return value
    if type(value) in (int, float) and math.isfinite(value):
        return Fraction(str(value))
    raise ValueError('Only finite numbers, booleans, strings or null are allowed')


def walk(node):
    yield node
    if isinstance(node, dict):
        for child in node.get('args', []):
            yield from walk(child)


def infer(node, fields, constants, depth=0):
    if depth > 12:
        raise ValueError('Expression depth exceeds 12')
    if type(node) is bool:
        return 'boolean'
    if not isinstance(node, dict):
        raise ValueError('Literal medical thresholds must be cited constants')
    if set(node) == {'field'}:
        return fields[node['field']]['type']
    if set(node) == {'constant'}:
        value = constants[node['constant']]['value']
        return 'boolean' if type(value) is bool else 'number' if type(value) in (int, float) else 'text'
    if set(node) != {'op', 'args'} or node['op'] not in ARITY or not isinstance(node['args'], list):
        raise ValueError('Unknown expression operator or node shape')
    lo, hi = ARITY[node['op']]
    if not lo <= len(node['args']) <= hi:
        raise ValueError('Wrong operator arity')
    ts = [infer(x, fields, constants, depth + 1) for x in node['args']]
    op = node['op']
    if op in ('and', 'or', 'not'):
        if set(ts) != {'boolean'}:
            raise ValueError('Logical operators require booleans')
        return 'boolean'
    if op == 'if':
        if ts[0] != 'boolean' or ts[1] != ts[2]:
            raise ValueError('Conditional branches need a common type')
        return ts[1]
    if op in ('eq', 'ne'):
        if ts[0] != ts[1]:
            raise ValueError('Equality type mismatch')
        return 'boolean'
    if set(ts) != {'number'}:
        raise ValueError('Arithmetic and ordering require numeric inputs')
    return 'boolean' if op in ('lt', 'le', 'gt', 'ge') else 'number'


def evaluate(node, state, constants):
    if type(node) is bool:
        return node
    if 'field' in node:
        return scalar(state[node['field']])
    if 'constant' in node:
        return scalar(constants[node['constant']]['value'])
    op = node['op']
    xs = [evaluate(x, state, constants) for x in node['args']]
    if op == 'and':
        return False if False in xs else None if None in xs else True
    if op == 'or':
        return True if True in xs else None if None in xs else False
    if op == 'not':
        return None if xs[0] is None else not xs[0]
    if op == 'if':
        return xs[1] if xs[0] is True else xs[2] if xs[0] is False else (xs[1] if xs[1] == xs[2] else None)
    if None in xs:
        return None
    if op in UNIT_FACTORS: return xs[0] * UNIT_FACTORS[op]
    if op == 'eq': return xs[0] == xs[1]
    if op == 'ne': return xs[0] != xs[1]
    if op == 'lt': return xs[0] < xs[1]
    if op == 'le': return xs[0] <= xs[1]
    if op == 'gt': return xs[0] > xs[1]
    if op == 'ge': return xs[0] >= xs[1]
    if op == 'add': return sum(xs, Fraction(0))
    if op == 'sub': return xs[0] - xs[1]
    if op == 'mul':
        result = Fraction(1)
        for value in xs: result *= value
        return result
    if op == 'div':
        if xs[1] == 0: raise ValueError('Undefined division is not evidence insufficiency')
        return xs[0] / xs[1]
    raise ValueError('Unknown operator')


def outcome(expression, state, constants):
    value = evaluate(expression, state, constants)
    if value is not None and type(value) is not bool:
        raise ValueError('A claim must evaluate to boolean or unknown')
    return U if value is None else S if value else R


def validate(candidate, source, single=False):
    """Machine provenance/type/coverage checks, NOT clinical entailment validation."""
    errors, clean = [], []
    blocks = {b['anchor_id']: b['text'] for d in source['documents'] for b in d['blocks']}
    if not isinstance(candidate, dict) or set(candidate) != {'source_id', 'items'}:
        return None, ['candidate_shape']
    if candidate['source_id'] != source['source_id']:
        errors.append('source_id_mismatch')
    items = candidate.get('items')
    if not isinstance(items, list) or not (1 if single else 3) <= len(items) <= 7:
        return None, errors + ['need_3_to_7_source_specific_families']
    for index, item in enumerate(items):
        try:
            if not isinstance(item['statement'], str) or not 12 <= len(item['statement']) <= 220:
                raise ValueError('statement_length')
            if any(word in item['statement'].lower() for word in ('supported', 'refuted', 'not_determined', '资料不足', '无法判断', '未提供', '未记录')):
                raise ValueError('answer_or_missingness_hint_in_stem')
            fields = {f['key']: f for f in item['fields']}
            if len(fields) != len(item['fields']) or not (1 if item['family_type'] == 'stable_insufficiency' else 2) <= len(fields) <= 6:
                raise ValueError('need_2_to_6_distinct_clinical_fields')
            for key, f in fields.items():
                if not re.fullmatch('[a-z][a-z0-9_]{0,39}', key) or f['type'] not in TYPES:
                    raise ValueError('invalid_field_key_or_type')
                if not f['label'] or len(f['label']) > 100 or any(w in f['label'] for w in ('答案', '判定结果', '结论是否', '标准阈值', '规则阈值')):
                    raise ValueError('label_encoded_state')
                if f['type'] == 'number' and not f['unit']:
                    raise ValueError('numeric_field_needs_explicit_unit')
            refs = item['references']
            if not 1 <= len(refs) <= 6:
                raise ValueError('need_original_quote_references')
            for r in refs:
                if r['anchor_id'] not in blocks or len(r['quote']) < 6 or r['quote'] not in blocks[r['anchor_id']]:
                    raise ValueError('quote_not_exact_original_substring')
            constants = {}
            for c in item['constants']:
                if c['key'] in constants or not re.fullmatch('[a-z][a-z0-9_]{0,39}', c['key']):
                    raise ValueError('invalid_constant_key')
                v = json.loads(c['value_json'])
                if v is None: raise ValueError('null_is_not_a_rule_constant')
                scalar(v)
                if c['anchor_id'] not in blocks or len(c['quote']) < 6 or c['quote'] not in blocks[c['anchor_id']]:
                    raise ValueError('constant_quote_not_exact')
                if type(v) is str and v not in c['quote']:
                    raise ValueError('text_constant_not_in_quoted_rule')
                if type(v) in (int, float):
                    numbers = [float(x) for x in re.findall(r'(?<![\d.])-?\d+(?:\.\d+)?', c['quote'].replace(',', '').replace('，', ''))]
                    if float(v) not in numbers + chinese_cardinals(c['quote']):
                        raise ValueError('numeric_constant_not_explicit_in_quote_use_source_native_units')
                    if not c['unit']: raise ValueError('numeric_constant_unit_missing')
                constants[c['key']] = {**c, 'value': v}
            if item['family_type'] not in ('multi_step', 'clinical_relation', 'stable_insufficiency'):
                raise ValueError('unknown_family_type')
            expression = json.loads(item['expression_json'])
            if infer(expression, fields, constants) != 'boolean':
                raise ValueError('claim_root_not_boolean')
            nodes = list(walk(expression))
            used_fields = {n['field'] for n in nodes if isinstance(n, dict) and 'field' in n}
            used_constants = {n['constant'] for n in nodes if isinstance(n, dict) and 'constant' in n}
            if used_fields != set(fields) or used_constants != set(constants):
                raise ValueError('unused_or_undefined_rule_inputs')
            states = []
            for state in item['states']:
                values = {v['key']: json.loads(v['value_json']) for v in state}
                if len(values) != len(state) or set(values) != set(fields):
                    raise ValueError('state_field_set_mismatch')
                for key, value in values.items():
                    scalar(value)
                    if value is not None:
                        actual = 'boolean' if type(value) is bool else 'number' if type(value) in (int, float) else 'text'
                        if actual != fields[key]['type']: raise ValueError('state_value_type_mismatch')
                states.append(values)
            if len(states) != 3 or (item['family_type'] != 'stable_insufficiency' and len({json.dumps(s, sort_keys=True) for s in states}) != 3):
                raise ValueError('three_distinct_states_required')
            labels = [outcome(expression, s, constants) for s in states]
            stable = item['family_type'] == 'stable_insufficiency'
            if stable:
                if labels != [U, U, U] or len(item['missing_information']) < 8:
                    raise ValueError('gap_control_not_unknown_in_all_states')
            else:
                if set(labels) != {S, R, U}:
                    raise ValueError('balanced_family_needs_one_S_one_R_one_U')
                # Each exposed clinical field must affect this actual rule, not
                # just be a decorative field making a one-hop template look long.
                for key in fields:
                    sensitive = False
                    choices = [s[key] for s in states] + [None]
                    for state in states:
                        before = outcome(expression, state, constants)
                        for value in choices:
                            changed = {**state, key: value}
                            try: sensitive |= outcome(expression, changed, constants) != before
                            except ValueError: pass
                    if not sensitive: raise ValueError('decorative_noncausal_field_' + key)
                source_sensitive = False
                for key, c in constants.items():
                    v = c['value']
                    alternatives = [not v] if type(v) is bool else [-1000000, 0, 1, 1000000, float(v) + 1] if type(v) in (int, float) else ['__different_source_identity__']
                    for alternative in alternatives:
                        altered = {**constants, key: {**c, 'value': alternative}}
                        for state in states:
                            try:
                                source_sensitive |= outcome(expression, state, constants) != outcome(expression, state, altered)
                            except ValueError: pass
                if not source_sensitive:
                    raise ValueError('source_constants_cancel_or_do_not_affect_any_state')
            ops = sum(isinstance(n, dict) and 'op' in n for n in nodes)
            multistep = ops >= 3 and len(used_fields) >= 2
            if item['family_type'] == 'multi_step' and not multistep:
                raise ValueError('multi_step_needs_3_operators_and_2_effective_inputs')
            if not stable and not used_constants:
                raise ValueError('balanced_claim_requires_an_original_source_constant')
            if len(item['clinical_scope']) < 12 or len(item['reasoning_steps']) < 2:
                raise ValueError('need_bounded_clinical_scope_and_explicit_derivation')
            clean.append({**item, 'expression': expression, 'field_map': fields,
                          'constant_map': constants, 'state_values': states,
                          'labels': labels, 'machine_multistep': multistep})
        except (KeyError, TypeError, ValueError, ArithmeticError, json.JSONDecodeError) as e:
            errors.append(f'item_{index}: {str(e)[:220]}')
    if not errors and not single:
        if sum(i['family_type'] == 'stable_insufficiency' for i in clean) != 1:
            errors.append('exactly_one_stable_insufficiency_family_required')
        if sum(i['family_type'] == 'multi_step' for i in clean) < 1:
            errors.append('at_least_one_multistep_family_required')
        if len({i['statement'] for i in clean}) != len(clean): errors.append('duplicate_question_stems')
    return (clean if not errors else None), errors


