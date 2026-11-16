"""Single model-facing expression syntax for the existing executable frames.

The model emits op/args calls. This adapter translates them once into the
unchanged frame/record language; it neither repairs output nor calls a checker.
"""
from copy import deepcopy
import json

from common import fingerprint
from methods import definition_library, frame_context, frame_library, record_shared, record_solver, ssv


PRIMITIVES = {
    "ref": ["reference_id", "field"],
    "record": ["record_id", "field"],
    "eq": ["left", "right"], "ne": ["left", "right"],
    "lt": ["left", "right"], "le": ["left", "right"],
    "gt": ["left", "right"], "ge": ["left", "right"],
    "and": ["condition", "..."], "or": ["condition", "..."],
    "not": ["condition"], "implies": ["premise", "conclusion"],
    "ite": ["condition", "if_true", "if_false"],
    "add": ["number", "..."], "mul": ["number", "..."],
    "sub": ["left", "right"], "div": ["numerator", "denominator"],
}

PROMPT = r'''将本轮公开材料与待判命题转成可执行语义图。返回紧凑JSON对象，只有
properties、records、references、rules、queries五个顶层字段。只根据task_input与
observations抽取来源事实，不输出答案标签，不把查询中的断言当成来源事实。

表达式只有一种写法：{"op":"操作名","args":[按签名顺序给出的参数]}。
数字、字符串、true/false直接写字面值。op只能选下方目录列出的名字；目录中的
语义框架也是同一种调用，参数是已声明reference的ID，程序负责展开其内部逻辑。
ref读取待绑定引用的字段，record读取明确指定的固定记录字段，二者均可用于查询。
来源rules只用record，避免来源事实随查询绑定变化。框架allowed参数是
引用ID数组。不要输出其他表达式结构，也不要输出内部模板。

properties: [{"name":字段名,"type":"Bool|Int|Real|String",...}]。
框架所读字段需有semantic_type：Identity为String，Dose为Int/Real并注明unit，
Frequency为Int/Real且unit="1/d"，Completion为Bool。其他字段声明底层type即可，
数量字段注明unit。同一数量比较必须量纲一致；按体重/面积归一化的量需显式计算，
未知分母保留未知字段，不能省掉分母。只能做由来源数值支持的单位换算。
NOT/AND/OR等明确布尔位置的字段若漏写类型，程序可推导为Bool，但不会补充真假值。

records: [{"id":记录ID,"kind":"regimen|event|state","subject":主体,
"time":状态时间,"modality":"actual|recommendation|plan|order",
"scenario":场景,"document_id":文档ID,"anchor_id":锚点ID,
"fields":{字段名:字面值或null},"sources":[{"location":位置,"quote":原文}]}]。
每个独立方案、模拟场景、事件状态分别建记录；明确同一状态的多文档字段可合并。
实际值与推荐、历史、计划各自保留。actual包括模拟场景给定的实际状态和患者自述。
time是所述状态的时间，不是文档发布日期。主体与药品/检查object_id分开；通用
推荐不继承患者身份。计划/医嘱不改写为推荐，计划完成不等于实际完成。
只保存来源明确的字段；缺失或未记录用null，不写false/0。查询读null字段时得到
有类型的未知量，表达式中不用null。可计算的“符合/相同/全部正确”必须展开比较，
不能抽成不明真假的整体布尔。原文明示的症状、执行、并用状态可直接是布尔字段。

references: [{"id":引用ID,"text":原文指称,"where":{定位条件},
"slots":{语义槽:实际字段名},"scores":{候选记录ID:0到4的语义匹配分}}]。
where仅用kind/entity/subject/target/time/modality/scenario/document_id/anchor_id，
只放指称已明确的条件，不放剂量、频次、completed等查询要判断的内容。target读取
object_id槽；问实际对象是否与推荐相同时，不先按推荐药品限制实际候选。
按照reference_locator及上下文找到所指方案；同一锚点的不同方案仍分别保留。
slots把dose/frequency/dose_lower/dose_upper/frequency_lower/frequency_upper/
object_id/completed这些框架槽映射到properties字段；字段同名可省略映射。
scores按指称、来源定位、时间、身份评分，不按能否使命题成立评分；没有偏好可{}。
同一引用在所有查询中选择同一记录。框架为实际角色补充actual，比较参照仅补充
kind=regimen，不指定其modality；医嘱order、推荐recommendation、计划plan按原文
保留并在引用where中定位。主体、时间、场景、来源各自指定，不跨角色复制。
每个引用ID都要在references中声明，记录ID不是引用ID。
一个含糊引用的多个候选不是允许集合：一个reference负责选定一个来源；只有
原文明确允许多个备选时才用allowed或OR，不把来源其他方案擅自加入允许范围。

queries: [{"item_id":原ID,"text":原命题,"expr":表达式}]，覆盖全部原ID。
选用符合原意的框架，额外的疗程、并用、症状、严格或单侧界限用原子比较与AND/OR
组合保留。单次量、频次分别符合与日总量符合不同；直接给出的日总量直接读取。
全方/整组一致要逐项比较并AND。指向明确来源记录时直接用record，需要消歧或
跨字段共享绑定时用ref。不能通过rules假定查询为真。

rules区分两种来源关系，没有则[]：
1. 事实性条件关系：{"id":ID,"text":原文,"expr":表达式,"sources":[来源]}。
   expr只用record读取，表示事实之间的蕴涵，不表示禁用、推荐、医嘱或慎用。
2. 条件性规范：{"id":ID,"text":原文,"modality":"prohibition|recommendation|order|caution|permission",
   "action":原文规范的行为/对象,"when":条件表达式,"sources":[来源]}。
   禁用用prohibition，建议用recommendation，要求用order，慎用用caution，明确允许
   用permission。when用record读取所指当前场景的条件；无条件规范才写true。
   同一规范用于不同患者/场景时分别实例化when，不能把A的条件拿给B。
   queries用{"op":"directive_applies","args":[规则ID,"prohibition"]}询问该条
   禁令是否适用；第二参数也可为其他意图类型。程序只计算条件是否成立，不把规范
   写成事实蕴涵，更不会推出患者已执行/未执行。实际行为另读actual记录字段。
   not(directive_applies(...))仅表示该条禁令不适用，不表示对所有风险都允许使用。
规则提到“满足P时禁用X”不等于P已经发生；不能在规则记录里填P=true，也不能写
P→NOT(P)或P→NOT(实际执行X)。未知条件保持null；未记录不等于阴性。
例如“潮湿时禁止启动机器”，规则用modality=prohibition、action="启动机器"、
when={"op":"record","args":["当前设备","潮湿"]}。即使实际已启动，禁令适用和
实际启动也可同时为真，事实不会矛盾。若潮湿未知，禁令是否适用也未知。

一个完整的非医疗例子（协议示例中的记录和值不属于本轮事实）：
{"properties":[{"name":"dose","type":"Real","semantic_type":"Dose","unit":"kg"},
{"name":"dose_lower","type":"Real","semantic_type":"Dose","unit":"kg"},
{"name":"dose_upper","type":"Real","semantic_type":"Dose","unit":"kg"},
{"name":"area","type":"Real","unit":"m2"},
{"name":"surface_limit","type":"Real","unit":"kg/m2"}],
"records":[{"id":"measured","kind":"regimen","subject":"boxA","time":"current",
"modality":"actual","scenario":"A","document_id":"measure","anchor_id":"M1",
"fields":{"dose":8,"area":null},"sources":[{"location":"M1","quote":"A本件8kg，面积未记录"}]},
{"id":"spec","kind":"regimen","subject":"generic_box","time":"reference",
"modality":"recommendation","scenario":"spec","document_id":"specification","anchor_id":"R1",
"fields":{"dose_lower":7,"dose_upper":9,"surface_limit":4},
"sources":[{"location":"R1","quote":"建议每件7至9kg且每平方米不超过4kg"}]}],
"references":[{"id":"actual","text":"A本件","where":{"subject":"boxA","scenario":"A"},
"slots":{},"scores":{"measured":4}},
{"id":"recommended","text":"R1建议","where":{"document_id":"specification","anchor_id":"R1"},
"slots":{},"scores":{"spec":4}}],"rules":[],
"queries":[{"item_id":"Q1","text":"重量和单位面积重量均符合R1建议",
"expr":{"op":"and","args":[{"op":"single_dose_conforms","args":["actual","recommended"]},
{"op":"le","args":[{"op":"div","args":[{"op":"ref","args":["actual","dose"]},
{"op":"ref","args":["actual","area"]}]},{"op":"ref","args":["recommended","surface_limit"]}]}]}}]}

下面是操作目录；签名中的args顺序固定。只输出一次JSON。'''


def catalog_for_prompt():
    library = frame_library.load_catalog()
    frames = []
    for frame in library["frames"]:
        boundary = frame["boundary"]
        if frame["id"] == "dose_frequency_conforms":
            boundary = "量与频次来自同一方案。仅原文明示允许的备选方案可以分别构造框架再OR；指称候选不代表允许备选。"
        frames.append({"op": frame["id"], "args": list(frame["roles"]),
                       "meaning": frame["meaning"], "roles": frame["roles"],
                       "boundary": boundary})
    directive = next(row for row in definition_library.load_catalog()["definitions"]
                     if row["id"] == "directive_active")
    return {"frames": frames,
            "directives": [{"op": "directive_applies", "args": ["rule_id", "modality"],
                            "meaning": directive["meaning"],
                            "modalities": directive["parameters"]["queried_modality"]["values"],
                            "boundary": directive["assumptions"]}],
            "primitives": [{"op": op, "args": args} for op, args in PRIMITIVES.items()]}


def decode(graph):
    """Compile the single external syntax. No aliases or output repair."""
    frames = {row["id"]: row for row in frame_library.load_catalog()["frames"]}
    result = deepcopy(graph)

    def expression(node):
        if node is None or isinstance(node, list):
            raise ValueError("expression_requires_literal_or_op_args")
        if not isinstance(node, dict):
            return node
        if set(node) != {"op", "args"} or not isinstance(node["args"], list):
            raise ValueError("expression_requires_op_args")
        op, args = node["op"], node["args"]
        if op == "directive_applies":
            if len(args) != 2:
                raise ValueError("directive_argument_count")
            return {"directive": args[0], "modality": args[1]}
        if op in frames:
            roles = list(frames[op]["roles"])
            if len(args) != len(roles):
                raise ValueError("frame_argument_count: " + op)
            return {"frame": op, "roles": dict(zip(roles, args))}
        if op not in PRIMITIVES:
            raise ValueError("unknown_protocol_operation: " + str(op))
        signature = PRIMITIVES[op]
        if "..." not in signature and len(args) != len(signature):
            raise ValueError("primitive_argument_count: " + op)
        return [op] + [expression(arg) for arg in args]

    for row in result["queries"]:
        row["expr"] = expression(row["expr"])
    for row in result.get("rules", []):
        field = "when" if "modality" in row else "expr"
        row[field] = expression(row[field])
    # A reference denotes one record; its semantic match score is common to
    # every field. The unchanged record solver consumes field-indexed scores.
    fields = [prop["name"] for prop in result["properties"]]
    for reference in result["references"]:
        scores = reference.get("scores", {})
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               for value in scores.values()):
            raise ValueError("reference_scores_require_record_numbers")
        reference["scores"] = {field: dict(scores) for field in fields}
    return result


def generate(ctx):
    catalog = catalog_for_prompt()
    raw = ctx.llm([
        {"role": "system", "content": PROMPT + "\n" + json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))},
        {"role": "user", "content": json.dumps(ssv._public_problem(ctx.problem), ensure_ascii=False)},
    ], stage="frame_generate", temperature=0, json_mode=True)
    ctx.record("frame_generation_response", {"raw": raw, "catalog_sha256": fingerprint(catalog)})
    return accept_graph(ctx, json.loads(raw))


def accept_graph(ctx, data):
    """Accept either an independent generation or the native formalization tool."""
    ctx.record("frame_protocol_graph", data)
    graph = decode(data)
    ctx.record("frame_graph", graph)
    ids = [query["item_id"] for query in graph["queries"]]
    if len(set(ids)) != len(ids) or set(ids) != set(ssv._claim_ids(ctx.problem)):
        raise ValueError("frame_query_ids_differ_from_task")
    ctx.record("frame_usage", frame_context.usage(graph))
    return graph


def run_mode(ctx, use_role_context):
    return solve_graph(ctx, generate(ctx), use_role_context)


def solve_graph(ctx, raw_graph, use_role_context=True, arm=None):
    """Run an already generated graph without any model request."""
    graph = frame_library.lower(raw_graph, use_role_context=use_role_context)
    ctx.record("record_graph", graph)
    selection = record_solver.select(graph, mode="shared")
    arm = arm or ("frame_protocol_context" if use_role_context else "frame_protocol_plain")
    result = record_shared.finish(ctx, graph, selection, mode=arm, model_calls=1)
    result["method"] = arm
    result["diagnostics"].update(
        implementation="CPM_single_protocol_role_frames", use_role_context=use_role_context,
        raw_frame_graph_sha256=fingerprint(raw_graph), **frame_context.usage(raw_graph))
    ctx.record("result", result)
    return result
