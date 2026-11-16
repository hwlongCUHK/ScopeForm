"""One extraction call; source records remain fixed while query references bind."""
import json
from pathlib import Path

from common import fingerprint
from methods import record_solver, ssv


PROMPT = r'''将公开原文转换成“固定来源记录 + 待绑定的查询引用”。只返回JSON。
每条记录表示一个可识别的方案、事件或独立模拟状态。记录同时存在，查询选哪条
都不能改变记录中的值。一个段落可以有多个方案；同一方案明确分散在多个文档时
可以合成一条记录并保存各字段的来源。不能仅因同一患者、同段建议而合并事件。

使用这5个字段：properties、records、references、rules、queries。
以下是一个完整的非医疗例子：
{
 "properties":[{"name":"count","type":"Real"},{"name":"weight","type":"Real"}],
 "records":[
  {"id":"R1","kind":"plan","entity":"箱甲方案","time":"reference","modality":"recommended",
   "scenario":"","document_id":"方案文档","anchor_id":"P1",
   "fields":{"count":3,"weight":2},"sources":[{"location":"P1","quote":"甲方案3件，每件2千克"}]},
  {"id":"R2","kind":"plan","entity":"箱乙方案","time":"reference","modality":"recommended",
   "scenario":"","document_id":"方案文档","anchor_id":"P1",
   "fields":{"count":1,"weight":5},"sources":[{"location":"P1","quote":"乙方案1件，每件5千克"}]}],
 "references":[{"id":"r","text":"P1中的该方案",
   "where":{"kind":"plan","document_id":"方案文档","anchor_id":"P1"},
   "scores":{"count":{"R1":2,"R2":2},"weight":{"R1":2,"R2":2}}}],
 "rules":[],
 "queries":[{"item_id":"示例Q","text":"该方案总重量超过5千克",
   "expr":["gt",["mul",["ref","r","count"],["ref","r","weight"]],5]}]
}

properties 为全部可能使用的字段声明统一类型 Bool/Int/Real/String。取值的未知
不改变类型。属性名可自定，同一种含义尽量共用字段名，单位在字段名中明确。

records:
- 一个具体方案的剂量、频次、疗程在同一record内；两个不同药品方案分为两条。
  原文显式列出的匹配方案应全部保留，不能预选一个再丢掉其他方案。
- 每个record含 id、kind、entity、time、modality、scenario、document_id、anchor_id、fields。
  kind使用简短一致类别（例如regimen/state/history）；time标明current/historical/reference；
  modality标明actual/recommended/conditional/reported等，引用处使用同样拼法。
  来源锚点不同的记录保留各自位置；多文档同对象可用sources保留全部位置。
- fields只含来源明确给出的数字、布尔、字符串或null。字段未记录填null或省略。
  本轮未记录不设成0或false，不回填历史值。所有字段值都是所在record内部的陈述。
  建议剂量不等于当前实际剂量；独立模拟场景的参数在相应场景内是实际值。
- 固定值作为区间参照时，例如明确单次量5，则dose_min与dose_max均为5；查询所用
  来源上下界必须有对应来源字段。实际日总量、每次量、频次是不同字段。
- 不把整条待判命题的真假直接写成一个字段。允许对应原文具体临床状态的布尔字段。

references:
- 每个引用指向一种记录。where只写引用原文明确指定的指称条件，可用字段仅有
  kind/entity/time/modality/scenario/document_id/anchor_id。按这些条件程序会枚举
  所有匹配记录；你不返回候选白名单。未知的定位条件省略，不能自行猜测补上。
- 同一“该方案”的单次量、频次、疗程通过同一ref读取。不同独立场景分别引用。
  显式共指可以复用ref。当前实际值和建议参照是两个不同引用。
- where绝不能写剂量、结局、是否执行等待判属性，也不能依据实际用量来选择更
  容易符合的参照。缺少reference_locator就不能假造其指向某药品。
- scores为每个被读取字段对各兼容record的0至4匹配分，仅依据原文指称、身份、
  时间与上下文，不看计算后能否得到某个答案。同样合理时同分。遗漏分数默认为0，
  不会从候选中删除该记录。不要把一个共同偏好故意拆成冲突的字段偏好。
- 记录确实匹配但某字段null/缺失时仍是候选；查询使用该字段时保持未知。

表达式（禁止Python）：
["ref",reference_id,field]：查询从选中的record读字段。
["record",record_id,field]：明确固定记录的字段，供rules使用。
字面量为数字、字符串、true/false。
["eq",a,b]及ne/lt/le/gt/ge；["and",...]、["or",...]、["not",x]；
["implies",条件,结果]；["add",...]、["sub",a,b]、["mul",...]、["div",a,b]；
["ite",条件,真值,假值]。
rules为[{id,text,expr,sources}]，仅在需要表达来源条件规则或派生关系时填写。
规则只能使用固定record地址，不能出现ref；规则不能把问题内容当成事实。
没有规则就用[]。具体临床适用条件未知时，不能直接断言结论。

queries必须包含全部原始item_id，text保留原命题，expr保留全部限定、否定、
身份、时间、实际执行和适用条件。查询只能用ref读取字段，record只供rules使用。
计算剂量/日总量要用算术，
不能用一个“符合要求”的布尔代替。缺失证据的条件仍保留在查询中。

只利用task_input和实际已取得的observations，不依赖外部医疗常识。
保留原文定位和简短引文，输出紧凑JSON；不输出标签、不校验、不修复、不加新步骤。
'''


def generate(ctx):
    raw = ctx.llm([
        {"role": "system", "content": "你将中文多文档内容抽取为固定来源记录及可组合查询，输出JSON。"},
        {"role": "user", "content": PROMPT + "\n" + json.dumps(ssv._public_problem(ctx.problem), ensure_ascii=False)}
    ], stage="record_generate", temperature=0, json_mode=True)
    ctx.record("record_generation_response", {"raw": raw})
    data = json.loads(raw)
    ctx.record("record_graph", data)
    ids = ssv._claim_ids(ctx.problem)
    query_ids = [q["item_id"] for q in data["queries"]]
    if len(query_ids) != len(set(query_ids)) or set(query_ids) != set(ids):
        raise ValueError("record_query_ids_differ_from_task")
    return data


def finish(ctx, data, selection, mode, model_calls=1):
    ids = ssv._claim_ids(ctx.problem)
    ctx.record("record_selection", selection)
    diagnostics = {"graph_sha256": fingerprint(data), "selection_mode": mode,
                   "record_count": len(data["records"]), "reference_count": len(data["references"]),
                   "multi_candidate_references": sum(len(v) > 1 for v in selection["candidate_domains"].values()),
                   "changed_fields": selection["changed_fields"],
                   "mixed_references": selection["mixed_references"],
                   "empty_domains": selection["empty_domains"],
                   "verified": False, "model_calls": model_calls,
                   "implementation": "CPM_immutable_records_late_binding"}
    if selection["empty_domains"]:
        result = {"method": "record_" + mode, "status": "failed", "items": [],
                  "diagnostics": diagnostics, "error": "unresolved_reference"}
    else:
        program = record_solver.compile_program(data, selection)
        diagnostics["evidence_sha256"] = fingerprint({k: program[k] for k in ("scopes", "properties", "facts")})
        ctx.record("record_compiled", program)
        compiler = Path(__file__).with_name("scope_solver.py").read_text(encoding="utf-8")
        execution = ctx.execute(compiler + "\nresult = evaluate(" + repr(program) + ", {})\n",
                                stage="record_" + mode, timeout=30)
        outcome = execution.get("result") or {}
        ctx.record("record_evaluation", outcome)
        items = outcome.get("items", [])
        complete = execution.get("ok") and outcome.get("base_status") == "sat" and len(items) == len(ids)
        diagnostics["solver_base_status"] = outcome.get("base_status")
        result = {"method": "record_" + mode, "status": "completed" if complete else "failed",
                  "items": items if complete else [], "diagnostics": diagnostics}
        if not complete:
            result["error"] = execution.get("stderr") or "inconsistent_or_incomplete_formalization"
    ctx.record("result", result)
    return result


def run_mode(ctx, mode):
    data = generate(ctx)
    return finish(ctx, data, record_solver.select(data, mode), mode)


def run(ctx):
    return run_mode(ctx, "shared")
