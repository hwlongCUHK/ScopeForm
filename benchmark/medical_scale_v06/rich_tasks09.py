"""Source-rich supplemental construction, same executable contrast invariants."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import threading
from pathlib import Path
import urllib.request
import run_scale09 as transport
import scale09c as flat
import scale09d as core
import scope09e as scoped
from run_generation import ROOT, read, dump, digest, run_lock, NoRedirect

WORK = ROOT / 'construction/rich_tasks09'
CODE = ('rich_tasks09.py', 'scope09e.py', 'scale09d.py', 'scale09c.py', 'scale09_rules_r4.py', 'scale09_rules_r3.py',
        'run_scale09.py', 'scale09_prompt.py', 'scale09_rules.py', 'compile_scale09.py', 'token_count.py')
LOCAL = threading.local()


def schema():
    result = flat.schema(); result['properties']['items']['minItems'] = 3; result['properties']['items']['maxItems'] = 3
    item = result['properties']['items']['items']
    item['properties']['task_scope'] = {'type': 'string', 'enum': ['source_stated_clinical_rule', 'patient_record_reconciliation', 'persistent_insufficiency']}
    item['required'].append('task_scope')
    fields = ['f' + str(i) for i in range(1, 6)]; constants = ['c' + str(i) for i in range(1, 7)]; nodes = ['n' + str(i) for i in range(1, 11)]
    item['properties']['fields']['items']['properties']['key'] = {'type': 'string', 'enum': fields}
    item['properties']['constants']['items']['properties']['key'] = {'type': 'string', 'enum': constants}
    item['properties']['nodes']['items']['properties']['id'] = {'type': 'string', 'enum': nodes}
    item['properties']['nodes']['items']['properties']['args']['items'] = {'type': 'string', 'enum':
        ['field:' + f for f in fields] + ['const:' + c for c in constants] + ['node:' + n for n in nodes] + ['bool:true', 'bool:false']}
    item['properties']['root'] = {'type': 'string', 'enum': nodes}
    item['properties']['gap_field'] = {'type': 'string', 'enum': [''] + fields}
    source = getattr(LOCAL, 'source', None)
    if source:
        ids = [b['anchor_id'] for d in source['documents'] for b in d['blocks']]
        item['properties']['references']['items'] = {'type': 'string', 'enum': ids}
        item['properties']['constants']['items']['properties']['anchor_id'] = {'type': 'string', 'enum': ids}
        result['properties']['source_id'] = {'type': 'string', 'enum': [source['source_id']]}
    return result


def messages(source, errors=None):
    LOCAL.source = source
    policy = flat.POLICY.replace('4个独立', '3个独立').replace('3个balanced', '2个balanced').replace('至少两个balanced', '至少一个balanced')
    policy += '''\n计算图内部键统一使用f1至f5（字段）、c1至c6（常量）、n1至n10（节点）；临床含义放label、unit和解释里。所有引用使用相应field:f1、const:c1、node:n1形式，不能把常量写成field或省略符号。真实anchor_id只能从当前来源选择。内部计算符号不会给被测模型。
补充：只从医生原文明确给出的条件/剂量关系构建临床条件题。不能把原患者的年龄、体重或数值，偷换成对所有人的临床阈值；“年轻/一般”不能变成40岁等数值界限。
原文若提供的是个体事实/医嘱，可做patient_record_reconciliation：例如结合本轮原始药名、真实单次用量、真实频次/实际执行，核对原文明确记载的既往方案；或用实际开始/结束时间计算持续区间，再核对既定复查时序和是否真的实施。只判“文内信息一致/指定条件”，不判总体处方适宜、疗效/因果、药物临床等效。
每个balanced至少两个原始临床输入都要影响结论。至少一个经三步以上整合（例如原文单次剂量×原文频次作为参照，本轮实际量比较，再结合真实暴露；或日期差、原文规定间隔、实际到诊；或原始药名与用法分别比较，再结合实际用药）。不得把常量与它自己比较，也不得通过增加无用参数或重复not凑步骤。
fields必须给原始可观察量，不可给“是否符合原文”“是否与历史一致”“是否达到阈值”等答案等价字段。参照值仅在原文，不能重述到题干或本轮状态。错误态也用真实可理解的药名、剂量、时段，避免“其他药”“不符合标准”等答案提示值。
先选本来源有足够证据的两个不同信息核对点，再写计算图；不必把同一临床概念换词写成多个家族。stable_insufficiency选缺本轮真实检查/执行结果的有界陈述，不把缺乏因果证据包装为确定诊断。
task_scope仅用于私有分层审查，按source_stated_clinical_rule、patient_record_reconciliation、persistent_insufficiency填写。值不表示已获独立医学验证。字符串常量按原文逐字复制，不能改成医学同义名或自行增加“类/级”等字。'''
    text = policy + '\n完整来源（未截断）：' + json.dumps({k: source[k] for k in ('source_id', 'modality', 'documents')}, ensure_ascii=False)
    if errors:
        paths = sorted((WORK / 'sources' / source['source_id']).glob('candidate_*/candidate.private.json'))
        text += '\n确定性构建错误：' + json.dumps(errors, ensure_ascii=False)
        if paths: text += '\n上一提案原样保留，针对错误修复：' + json.dumps(read(paths[-1]), ensure_ascii=False)
    return [{'role': 'user', 'content': text}]


def bank(source):
    entries, failures = [], []
    for path in sorted((WORK / 'sources' / source['source_id']).glob('candidate_*/candidate.private.json')):
        candidate = read(path); selected, errors = scoped.extract(candidate, source)
        failures += [{'proposal': path.parent.name, **e} for e in errors]
        for clean, canonical in selected:
            raw = next(i for i in candidate['items'] if i['statement'] == clean['statement'])
            clean['task_scope_private'] = raw.get('task_scope')
            entries.append((clean, canonical, {'candidate_path': path.relative_to(ROOT).as_posix(), 'item_statement': clean['statement']}))
    return core.select_bank(entries), failures


def validate(candidate, source):
    selected, errors = bank(source)
    if not selected: return None, ['Need 2 source-grounded balanced families, >=1 real multistep, one stable gap.'] + [json.dumps(e, ensure_ascii=False) for e in errors[-8:]]
    for item, _, provenance in selected: item['construction_provenance_private'] = provenance
    return [x[0] for x in selected], []


def main():
    p = argparse.ArgumentParser(); p.add_argument('--pilot', action='store_true'); p.add_argument('--allow-private-data', action='store_true'); a = p.parse_args()
    if not a.allow_private_data: raise ValueError('Explicit local private-source construction required')
    proto = read(WORK / 'protocol.json'); reserve = ROOT / 'construction/rule_rich_reserve09'
    metadata = read(reserve / 'manifest.json'); path = reserve / 'sources.private.jsonl'
    assert digest(path) == metadata['sha256']
    sources = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
    assert len(sources) == metadata['sources'] and all(s['split'] == 'development' for s in sources)
    plan = {'protocol_sha256': digest(WORK / 'protocol.json'), 'source_manifest_sha256': digest(reserve / 'manifest.json'),
            'source_input_sha256': digest(path), 'code_sha256': {n: digest(ROOT / n) for n in CODE},
            'source_order': [s['source_id'] for s in sources], 'not_solver_evaluation': True}
    with run_lock(WORK / 'run.lock'):
        if (WORK / 'run_plan.json').exists() and read(WORK / 'run_plan.json') != plan: raise ValueError('Frozen supplemental plan changed')
        dump(WORK / 'run_plan.json', plan)
        from transformers import AutoTokenizer
        modelpath = Path(proto['teacher_path']); assert digest(modelpath / 'config.json') == proto['teacher_config_sha256']
        tok = AutoTokenizer.from_pretrained(modelpath, local_files_only=True, trust_remote_code=False)
        with urllib.request.build_opener(NoRedirect()).open('http://127.0.0.1:8073/v1/models', timeout=15) as r:
            assert proto['teacher_model'] in {m['id'] for m in json.load(r)['data']}
        transport.WORK = WORK; transport.messages = messages; transport.schema = schema; transport.validate = validate
        jobs = sources[:4] if a.pilot else sources
        rows = []
        with ThreadPoolExecutor(max_workers=4) as ex:
            for f in as_completed([ex.submit(transport.one, s, proto, tok, 8073) for s in jobs]):
                row = f.result(); rows.append(row)
                dump(WORK / 'progress.json', {'planned_sources': len(jobs), 'recorded': len(rows), 'machine_valid': sum(r['machine_valid'] for r in rows),
                     'source_pool_not_task_inflation': True, 'solver_calls': 0})
                print(json.dumps({'event': 'rich_source_constructed', 'source': row['source_id'], 'status': row['status'], 'recorded': len(rows)}, ensure_ascii=False), flush=True)
        oracle = 0
        if a.pilot:
            import tempfile
            from compile_scale09 import compile_source
            from medical_runtime07 import World, score
            by = {s['source_id']: s for s in sources}
            for row in rows:
                if not row['machine_valid']: continue
                with tempfile.TemporaryDirectory(prefix='rich09_oracle_') as temp:
                    temp = Path(temp)
                    cases = compile_source(by[row['source_id']], read(WORK / 'sources' / row['source_id'] / 'accepted_spec.private.json'), temp)
                    for case in cases:
                        folder = temp / 'cases' / case['task_id']; public = read(folder / 'controller.private.json'); gold = read(folder / 'gold.private.json')
                        w = World(public, 'initial', True); w.initial_input()
                        w.call('submit_answer', {'items': [{'item_id': g['item_id'], 'label': g['label'], 'evidence_ids': g['support_anchor_ids']} for g in gold['initial']], 'action_status': 'not_requested'})
                        assert score(w, gold)['content'] and score(w, gold)['evidence']; oracle += 1
        dump(WORK / ('pilot.json' if a.pilot else 'complete.json'), {'recorded': len(rows), 'machine_valid': sum(r['machine_valid'] for r in rows),
            'planned_sources': len(jobs), 'oracle_packets_passed': oracle, 'new_solver_evaluations': 0, 'expert_clinical_gold': False})


if __name__ == '__main__': main()
