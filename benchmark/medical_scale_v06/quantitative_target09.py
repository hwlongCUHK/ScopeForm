"""Source-exact quantitative medication reconciliation, not prescribing advice.

Supplemental construction only. Regimen structure is parsed from original text;
no new model calls, no inferred medical thresholds, and no solver-score filter.
"""
from collections import Counter
import copy
import json
from pathlib import Path
import re
from run_generation import ROOT, read, dump, digest
import scale09_rules_r4 as rules
import finalize_medical09 as prior

WORK = ROOT / 'construction/quantitative_target09'
NUM = r'(?:\d+(?:\.\d+)?|[一二两三四五六七八九十]+)'
UNIT = r'(?:毫克|毫升|微克|mg|mL|ml|μg|ug|克|g|片|粒|袋|支|丸)'
SINGLE = re.compile(r'(?:每次|一次|单次)\s*(' + NUM + r')\s*(?:[-—~～至到]\s*(' + NUM + r'))?\s*(' + UNIT + r')', re.I)
DAILY = re.compile(r'(?:每日|每天|一日|一天)\s*(' + NUM + r')\s*(?:[-—~～至到]\s*(' + NUM + r'))?\s*次')
BAD = re.compile(r'大概|大约|左右|至少|最多|不超过|不少于|以上|以下|最少|按需|必要时|每周|每月|隔日|隔天|[<>≤≥]')


def number(text):
    if re.fullmatch(r'\d+(?:\.\d+)?', text): return float(text) if '.' in text else int(text)
    values = rules.chinese_cardinals(text)
    if len(values) != 1: raise ValueError('Ambiguous Chinese quantity')
    return values[0]


def parse(source):
    """Choose the first unambiguous complete regimen in an original block.

    Only doctor turns or drug instruction blocks. Each bound stays in original
    units; ranges are ranges, never a single sampled endpoint called a rule.
    """
    for document in source['documents']:
        for block in document['blocks']:
            if source['modality'] == 'dialogue' and block.get('role') != '医生一方': continue
            text = block['text']; singles = list(SINGLE.finditer(text))
            if not singles: continue
            # The statement names this FIRST complete regimen in the block,
            # not an undisclosed later subgroup chosen by a teacher.
            for dose in singles:
                candidates = [m for m in DAILY.finditer(text) if 0 <= m.start() - dose.end() <= 70]
                if len(candidates) != 1: continue
                freq = candidates[0]
                if any(dose.end() <= s.start() < freq.start() for s in singles): continue
                snippet = text[dose.start():freq.end()]
                if BAD.search(snippet): continue
                trailing = text[freq.end():freq.end() + 8]
                if re.match(r'\s*[?？吗么]', trailing): continue
                if re.match(r'\s*(?:或|至|到|[-—~～])\s*' + NUM, trailing): continue
                # Do not accidentally read "每日1次" out of "每日1次/周".
                if re.match(r'\s*[/／]\s*[周月]', trailing): continue
                lo, hi = number(dose[1]), number(dose[2] or dose[1])
                fl, fh = number(freq[1]), number(freq[2] or freq[1])
                if not (0 < lo <= hi < 100000 and 0 < fl <= fh <= 24): continue
                return {'anchor_id': block['anchor_id'], 'document_id': document['document_id'], 'unit': dose[3],
                    'single_min': lo, 'single_max': hi, 'daily_min': fl, 'daily_max': fh,
                    'original_contiguous_quote': snippet, 'character_span_private': [dose.start(), freq.end()],
                    'source_block_sha256': digest_text(text), 'doctor_or_instruction_only': True,
                    'medical_scope': 'Numerical reconciliation with this explicitly cited recorded regimen, not treatment eligibility or clinical safety.'}
    return None


def digest_text(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def construct(source):
    parsed = parse(source)
    if parsed is None: return None, {'reason': 'no_unambiguous_explicit_single_dose_and_daily_frequency'}
    aid = parsed['anchor_id']; block = next(b for d in source['documents'] for b in d['blocks'] if b['anchor_id'] == aid)
    refs = [{'anchor_id': aid, 'quote': block['text']}]
    a, z, b, y = [parsed[k] for k in ('single_min', 'single_max', 'daily_min', 'daily_max')]
    constants = [{'key': key, 'value_json': json.dumps(value), 'unit': unit, 'anchor_id': aid, 'quote': block['text']}
        for key, value, unit in [('a_min', a, parsed['unit']), ('a_max', z, parsed['unit']), ('b_min', b, '次/日'), ('b_max', y, '次/日')]]
    fields = [{'key': 'actual_single', 'label': '本轮实际每次使用量', 'type': 'number', 'unit': parsed['unit']},
              {'key': 'actual_daily_frequency', 'label': '本轮实际每日次数', 'type': 'number', 'unit': '次/日'}]
    f = lambda k: {'field': k}; c = lambda k: {'constant': k}; op = lambda k, *args: {'op': k, 'args': list(args)}
    actual = op('mul', f('actual_single'), f('actual_daily_frequency'))
    expressions = [op('and', op('ge', actual, op('mul', c('a_min'), c('b_min'))), op('le', actual, op('mul', c('a_max'), c('b_max')))),
        op('and', op('ge', f('actual_single'), c('a_min')), op('le', f('actual_single'), c('a_max')),
           op('ge', f('actual_daily_frequency'), c('b_min')), op('le', f('actual_daily_frequency'), c('b_max')))]
    scope = '参照原文锚点“' + aid + '”中由题目reference_locator定位的那条单次量与每日次数记载。'
    statements = [scope + '本轮实际日合计量位于原文单次量与日次数两区间相乘得到的名义日总量区间内。',
        scope + '本轮实际单次使用量与每日次数，分别位于原文该方案明确记载的单次量区间和日次数区间内。']
    s = {'actual_single': a, 'actual_daily_frequency': b}
    delta = max(.01, z * .1)
    bad_single = z * y / b + delta
    states = [[s, {'actual_single': bad_single, 'actual_daily_frequency': b}, {'actual_single': a, 'actual_daily_frequency': None}],
              [s, {'actual_single': z + delta, 'actual_daily_frequency': b}, {'actual_single': None, 'actual_daily_frequency': b}]]
    native = []
    for index, (statement, expression, values) in enumerate(zip(statements, expressions, states)):
        used = {n['constant'] for n in rules.walk(expression) if isinstance(n, dict) and 'constant' in n}
        native.append({'statement': statement, 'family_type': 'multi_step', 'references': refs, 'fields': fields,
            'constants': [cc for cc in constants if cc['key'] in used], 'expression_json': json.dumps(expression),
            'states': [[{'key': k, 'value_json': json.dumps(v)} for k, v in state.items()] for state in values],
            'clinical_scope': '仅核对指定原文的数值关系；日总量在区间不代表单次方案符合、总体处方适宜或对真实患者安全有效。',
            'reasoning_steps': ['从指定原文确定原生单位及单次用量、每日次数的上下界。', '分别计算日总量区间或逐项约束；严格区别总量约束与分次方案约束。'],
            'missing_information': ''})
    native.append({'statement': scope + '本轮实际单次使用量已达到原文该方案的单次量下界。',
        'family_type': 'stable_insufficiency', 'references': refs, 'fields': [fields[0]], 'constants': [constants[0]],
        'expression_json': json.dumps(op('ge', f('actual_single'), c('a_min'))),
        'states': [[{'key': 'actual_single', 'value_json': 'null'}] for _ in range(3)],
        'clinical_scope': '全文有历史方案也不能替代本轮患者实际暴露值，不据此推断真实患者治疗情况。',
        'reasoning_steps': ['读取原文明确方案中的单次量。', '当前患者单次实际用量没有记录，历史方案不能充当实际执行。'],
        'missing_information': '缺少本轮独立模拟患者的实际单次使用量。'})
    clean, errors = rules.validate({'source_id': source['source_id'], 'items': native}, source)
    if errors: return None, {'reason': 'typed_rule_validation', 'errors': errors}
    for index, item in enumerate(clean):
        gap = item['family_type'] == 'stable_insufficiency'
        item['counterfactual_design_private'] = {'stratum': 'persistent_insufficiency' if gap else 'hidden_reference_clinical_parameter',
            'primary_content_challenge': not gap, 'reference_hidden_from_stem': not gap,
            'execution_flags_held_fixed': True, 'no_operational_boolean_in_rule': True,
            'same_medical_rule_and_statement': True, 'chosen_using_solver_results': False}
        item['construction_provenance_private'] = {'candidate_path': (WORK / 'sources' / source['source_id'] / 'parsed_regimen.private.json').relative_to(ROOT).as_posix(),
            'kind': 'deterministic_exact_original_regimen_parser', 'rule_index': index, 'item_statement': item['statement']}
        item['nonredundant_operator_count_private'] = prior.operator_count(item['expression'])
        item['task_scope_private'] = 'bounded_quantitative_regimen_reconciliation_not_prescribing'
        item['parsed_regimen_private'] = parsed
        item['reference_locator_public'] = {'anchor_id': aid, 'start_character': parsed['character_span_private'][0],
            'end_character': parsed['character_span_private'][1], 'indexing': 'zero-based Unicode codepoint, half-open',
            'purpose': 'Identify the exact original clause; no clinical reference values or labels are repeated here.'}
    return clean, {'parsed': parsed, 'clinical_expert_gold': False, 'new_model_calls': 0}


def load_sources():
    sources, _, _ = prior.load_sources()
    base = ROOT / 'construction/explicit_regimen_reserve09'; m = read(base / 'manifest.json'); p = base / 'sources.private.jsonl'
    assert digest(p) == m['sha256']
    extra = {s['source_id']: s for s in (json.loads(line) for line in p.read_text(encoding='utf-8').splitlines())}
    assert len(extra) == m['sources'] and not set(extra) & set(sources)
    assert not {s['source_cluster'] for s in extra.values()} & {s['source_cluster'] for s in sources.values()}
    assert all(s['split'] == 'development' for s in extra.values())
    return {**sources, **extra}, set(extra)


def main():
    sources, extra = load_sources(); plan0 = read(ROOT / 'construction/clinical_repair09/run_plan.json')
    # New development consultation candidates were screened only on original
    # doctor regimen evidence before inference (this stage makes no calls).
    new_order = [s['source_id'] for s in read(ROOT / 'construction/explicit_regimen_reserve09/manifest.json')['source_catalog']]
    old_drugs = [s for s in plan0['source_order'] if sources[s]['modality'] == 'drug']
    plan = {'version': 'source-exact-quantitative-supplement09-1', 'prior_qualified_ids': plan0['prior_qualified_ids'],
        'new_dialogue_source_order': new_order, 'old_development_drug_source_order': old_drugs,
        'source_profile_manifest_sha256': digest(ROOT / 'construction/explicit_regimen_reserve09/manifest.json'),
        'code_sha256': {n: digest(ROOT / n) for n in ('quantitative_target09.py', 'scale09_rules_r4.py', 'compile_scale09.py')},
        'target_modalities': {'dialogue': 72, 'drug': 48}, 'new_model_calls': 0, 'solver_accuracy_selection': False,
        'clinical_expert_gold': False, 'primary_content_meaning': 'Bounded medication information reconciliation, not diagnostic reasoning or treatment safety.',
        'related_families': 'Daily-total interval and per-administration/frequency constraints are different but correlated checks on one original regimen. Cluster statistics by source, never independent patients.',
        'sampling_bias': 'Additional dialogue sources require an explicit doctor regimen; this is a medication-information development subset, not representative general clinical encounters.'}
    if (WORK / 'run_plan.json').exists(): assert read(WORK / 'run_plan.json') == plan
    else: dump(WORK / 'run_plan.json', plan)
    reused = read(ROOT / 'datasets/medical07_dev8_r3/manifest.json')['source_ids']
    counts = Counter(sources[s]['modality'] for s in reused + plan['prior_qualified_ids']); selected = []; history = []
    for modality, order in (('drug', old_drugs), ('dialogue', new_order)):
        for sid in order:
            if counts[modality] == plan['target_modalities'][modality]: break
            items, report = construct(sources[sid]); folder = WORK / 'sources' / sid
            row = {'source_id': sid, 'modality': modality, 'machine_valid': items is not None, **report}
            dump(folder / 'result.json', row); history.append(row)
            if items:
                dump(folder / 'accepted_spec.private.json', items); dump(folder / 'parsed_regimen.private.json', report['parsed'])
                selected.append(sid); counts[modality] += 1
    value = {'target_reached': dict(counts) == plan['target_modalities'], 'source_counts': dict(counts),
        'selected_additional_ids': selected, 'attempted_ids': [r['source_id'] for r in history],
        'new_model_calls': 0, 'new_solver_evaluations': 0, 'clinical_expert_gold': False,
        'construction_subsets_must_be_reported_separately': True}
    dump(WORK / 'complete.json', value); print(json.dumps(value), flush=True)


if __name__ == '__main__': main()
