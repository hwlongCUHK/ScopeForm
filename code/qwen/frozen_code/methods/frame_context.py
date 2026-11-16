"""One public-evidence formalization call using the executable frame catalog.

Both pilot arms receive the identical generation; only role context in lowering
differs. Source records stay fixed and the original native agent continues.
"""
from collections import Counter
import json

from common import fingerprint
from methods import definition_library, frame_library, record_shared, record_solver, ssv


PROMPT = r'''将本轮公开材料和待判命题转换为固定来源记录、引用和可执行语义框架。
只返回JSON对象，包含properties、records、references、rules、queries。只使用本轮
task_input和已取得的observations，不输出答案标签，不把待判命题或旧答案当作事实。

每个来源方案、事件状态或独立模拟状态各自成为record。不同方案不能合并；明确
属于同一方案且分散在多文档的字段可以合并并保留各来源。每个record包含：
id、kind、subject、time、modality、scenario、document_id、anchor_id、fields、sources。
kind规范为regimen（用药方案）、event（事件状态）、state（其他状态）。
modality规范为actual、recommendation、plan、order，表达原文实际状态、推荐、
计划、医嘱意图；不要为迁就框架把plan/order改成recommendation。来源是患者自述
不妨碍它描述actual。主体subject与药品/检查object_id分开。场景按task_input契约
保持独立scenario；subject按原文指称确定，明确为不同模拟患者时分别表示，
不单凭场景字母虚构主体。当前值不能从历史或推荐回填。
time按所述状态解释，不按文档发布日期解释。通用推荐不虚构具体患者subject。
fields只保存明确的数值、字符串、布尔或null；缺失不等于false/0。不把整条
查询的真假或“符合要求”抽取为字段。具体的执行、并用、症状状态可用布尔字段。
null只出现在fields，查询读取有类型变量，不写x!=null或字面量null运算。
sources保存实际取得原文的document_id、location和简短quote。

properties声明每个字段的name与底层type：Bool/Int/Real/String。框架读取字段
还要声明semantic_type：Identity、Dose、Frequency、Completion。Dose声明unit，
同一比较两侧统一单位；Frequency单位固定1/d（每日次数）；Identity为String，
Completion为Bool。临床数值只从公开原文取得。未使用框架的字段可保留普通类型。
框架的语义槽默认使用同名字段：object_id、dose、frequency、dose_lower、dose_upper、
frequency_lower、frequency_upper、completed。字段名可自定；不同单位可用不同字段。
固定推荐值作为区间时，上下界均为该值；区间两端不能取自不同方案。

references每项含id、text、where、slots、scores：
- text描述引用原句；where只放原句明确的指称条件，可用kind/entity/subject/target/
  time/modality/scenario/document_id/anchor_id。未知定位条件省略。where.target读取
  该引用object_id槽指向的字段，与对象比较使用同一个身份值。
- 框架会添加其角色固定的kind/modality。引用中这些条件仍须忠实原句，不能冲突。
  主体、时间、场景、来源定位在各角色单独确定，不把实际患者条件复制给通用推荐。
- where不能包含completed、剂量、频次、结局等待判内容。问“实际药品是否与推荐
  相同”时，不能先把实际候选限定成推荐药品；问“已经完成”不能只找完成记录。
- slots将框架语义槽映射为实际字段名，例如{"dose":"dose_mg"}；同名可省略映射。
  同一引用的全部字段共同选择一条record，跨命题确实共指则复用引用。实际与推荐
  使用不同引用。一个引用的候选域不是多个允许选项的集合。
- scores为具体字段名到候选record的0至4语义匹配分，依据身份、定位、时间与上下文；
  不能依据计算结果更容易支持命题来评分。用reference_locator识别原文所指方案，
  共享锚点中的不同方案仍须保留。记录字段未知仍是候选，不要为获得数值换记录。

queries覆盖全部原始item_id，包含item_id、text、expr，text保留原命题。
适用库中含义时，用{"frame":框架ID,"roles":{角色名:引用ID}}；allowed角色为
完整允许集合的引用ID列表。通过core的all_of/any_of/negate组合多个子命题。
框架只覆盖它声明的条件，疗程、并用、监测、症状、单侧界限等额外条件仍要用core
或原表达式保留。不要把超出框架的含义硬塞进框架。无合适框架时可直接用core或AST。
实际日总量直接有记录时读该字段，不用未知的单次量/频次代替它重算。历史陈述
与计划时限也各自按原意表示；支持的是本轮命题的有界含义。

core调用为{"definition":定义ID,"arguments":{参数名:表达式}}，可嵌套框架。
原表达式：数字、字符串、true/false；["ref",引用ID,字段名]；
eq/ne/lt/le/gt/ge（二参数），and/or（多参数）、not（一参数）、implies（二参数）、
add/mul（多参数）、sub/div（二参数）、ite（条件、真值、假值）。
rules仅表达来源明确的条件规则，格式[{id,text,expr,sources}]，无则[]。
规则只能用["record",固定记录ID,字段名]，不能用ref，不能把查询当来源规则。

非医疗结构例子（示意字段单位为kg）：
{"properties":[{"name":"dose","type":"Real","semantic_type":"Dose","unit":"kg"},
 {"name":"dose_lower","type":"Real","semantic_type":"Dose","unit":"kg"},
 {"name":"dose_upper","type":"Real","semantic_type":"Dose","unit":"kg"}],
 "records":[
 {"id":"actual_box","kind":"regimen","subject":"box_A","time":"current","modality":"actual",
  "scenario":"A","document_id":"measurement","anchor_id":"M1","fields":{"dose":2},
  "sources":[{"location":"M1","quote":"箱A本次每件重2千克"}]},
 {"id":"advised_box","kind":"regimen","subject":"generic_box","time":"reference","modality":"recommendation",
  "scenario":"spec","document_id":"spec","anchor_id":"S1","fields":{"dose_lower":3,"dose_upper":3},
  "sources":[{"location":"S1","quote":"建议每件3千克"}]}],
 "references":[
 {"id":"actual","text":"箱A本次实际值","where":{"subject":"box_A","time":"current","scenario":"A"},"slots":{},"scores":{}},
 {"id":"advice","text":"S1的建议","where":{"document_id":"spec","anchor_id":"S1"},"slots":{},"scores":{}}],
 "rules":[],"queries":[{"item_id":"Q1","text":"本次每件重量符合建议",
 "expr":{"frame":"single_dose_conforms","roles":{"observed":"actual","reference":"advice"}}}]}

下面是可用库定义，紧接其后的用户消息是本轮公开问题。输出紧凑JSON，不加核对、
修复、重采样步骤，不输出私有标签。'''


def catalog_for_prompt():
    core = definition_library.load_catalog()
    frames = frame_library.load_catalog()
    return {
        "semantic_types": frames["semantic_types"],
        "frames": [{key: row[key] for key in (
            "id", "meaning", "roles", "body", "contrast", "boundary")}
            for row in frames["frames"]],
        "definitions": [{key: row[key] for key in (
            "id", "meaning", "parameters", "returns", "formula", "selection_hint", "assumptions")}
            for row in core["definitions"]],
    }


def usage(graph):
    def walk(node, counts):
        if isinstance(node, dict):
            if "frame" in node:
                counts[node["frame"]] += 1
            for value in node.values():
                walk(value, counts)
        elif isinstance(node, list):
            for value in node:
                walk(value, counts)

    by_item, total = {}, Counter()
    for query in graph["queries"]:
        counts = Counter()
        walk(query["expr"], counts)
        by_item[query["item_id"]] = dict(counts)
        total.update(counts)
    return {"frames_by_item": by_item, "frame_counts": dict(total),
            "framed_claims": sum(bool(value) for value in by_item.values()),
            "total_claims": len(by_item)}


def generate(ctx):
    catalog = catalog_for_prompt()
    raw = ctx.llm([
        {"role": "system", "content": PROMPT + "\n" + json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))},
        {"role": "user", "content": json.dumps(ssv._public_problem(ctx.problem), ensure_ascii=False)},
    ], stage="frame_generate", temperature=0, json_mode=True)
    ctx.record("frame_generation_response", {"raw": raw, "catalog_sha256": fingerprint(catalog)})
    data = json.loads(raw)
    ctx.record("frame_graph", data)
    ids = [query["item_id"] for query in data["queries"]]
    if len(set(ids)) != len(ids) or set(ids) != set(ssv._claim_ids(ctx.problem)):
        raise ValueError("frame_query_ids_differ_from_task")
    ctx.record("frame_usage", usage(data))
    return data


def run_mode(ctx, use_role_context):
    raw_graph = generate(ctx)
    graph = frame_library.lower(raw_graph, use_role_context=use_role_context)
    ctx.record("record_graph", graph)
    selection = record_solver.select(graph, mode="shared")
    arm = "frame_context" if use_role_context else "frame_plain"
    result = record_shared.finish(ctx, graph, selection, mode=arm, model_calls=1)
    result["method"] = arm
    result["diagnostics"].update(
        implementation="CPM_executable_role_frames", use_role_context=use_role_context,
        raw_frame_graph_sha256=fingerprint(raw_graph), **usage(raw_graph))
    ctx.record("result", result)
    return result


def run(ctx):
    return run_mode(ctx, True)
