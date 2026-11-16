"""One-call scoped candidate generation, joint binding selection, exact queries.

This is a CPM integration inspired by compositional scope parsing and global
structured prediction, not a reproduction of any one source paper.
"""
import json
from pathlib import Path

from common import fingerprint
from methods import ssv


PROMPT = r'''把已取得的原文和待核对命题转换成一个可执行的作用域连接图。
只使用 task_input 与 observations，不输出最终标签。一次生成全部内容与连接；
本地程序先选择连接，再求解公式。所有表达式都通过 get 引用连接，不能绕过连接。

JSON 格式（下方给出一个完整、与当前任务无关的仓库示例）：
{
 "scopes":[
  {"id":"old_box","entity":"箱A","time":"上轮","modality":"实际"},
  {"id":"new_box","entity":"箱A","time":"本轮","modality":"实际"}],
 "properties":[{"name":"count","type":"Real"},{"name":"humidity","type":"Real"}],
 "choices":[
  {"id":"old_record","kind":"context","text":"旧记录语境",
   "candidates":[{"id":"old","value":"old_box","score":4}]},
  {"id":"current_record","kind":"context","text":"本轮记录与后续指代的共同语境",
   "candidates":[{"id":"now","value":"new_box","score":4},{"id":"old","value":"old_box","score":0}]},
  {"id":"query_count","kind":"binding","text":"本轮说明中的‘它的数量’",
   "candidates":[{"id":"old","value":"old_box","score":2},{"id":"now","value":"new_box","score":2}]},
  {"id":"query_humidity","kind":"binding","text":"同段的‘它的湿度’",
   "candidates":[{"id":"old","value":"old_box","score":2},{"id":"now","value":"new_box","score":2}]}],
 "factors":[{"id":"shared_current","relation":"same_scope",
   "choices":["current_record","query_count","query_humidity"],"score":4,
   "reason":"两个它均延续本轮箱A这一共同语境",
   "sources":[{"observation_index":0,"location":"例子","quote":"本轮箱A有4件；它的湿度未测"}]}],
 "facts":[
  {"id":"K1","text":"旧记录10件","expr":["eq",["get","old_record","count"],10]},
  {"id":"K2","text":"本轮4件","expr":["eq",["get","current_record","count"],4]},
  {"id":"K3","text":"本轮湿度未测","expr":null}],
 "queries":[{"item_id":"示例Q","text":"它的数量超过5件而且湿度低于70",
   "expr":["and",["gt",["get","query_count","count"],5],["lt",["get","query_humidity","humidity"],70]]}]
}

当前任务的图也使用以上6个字段。sources 尽量保留原文短引文和锚点。
scopes 描述具体主体/事件/时间/模态命名空间。不同药品或不同处方方案必须分别
建 scope，即使它们出现在同一段；历史建议、历史观察、当前实际状态也要区分。
properties 声明字段名称与类型：Bool、Int、Real、String。
choices 是所有引用经过的连接：
- context 连接表示一段原文内容的共同语境；同段多个事实通过它共享归属。
- binding 连接表示查询或其他内容提及的对象/时间/方案，value 只能是 scope ID。
- 每个连接给1至3个合理候选；确实只有一个解释时就给1个。数值不同但属于同一
  完整方案的字段共享该方案 scope，不能分别挑到不同药品的剂量和频次。
- 局部分数 score 为0至4，只比较提及本身/直接锚点的匹配，跨句证据放入 factor。
  对局部同样合理的候选用相同分数，不要提前将全局结论写入局部分数。

表达式语言只有这些操作，禁止生成 Python、直接访问 scope 或最终标签：
["get",choice_id,property_name]：先选出该连接的 scope，再读取其字段。
数字、true/false、字符串为常量。
比较：["eq",a,b]，以及 ne、lt、le、gt、ge。
布尔：["and",...]、["or",...]、["not",x]、["implies",条件,结果]。
算术：["add",...]、["sub",a,b]、["mul",...]、["div",a,b]。
条件值：["ite",条件,真值,假值]。
例如每次量区间固定为5时，dose_min和dose_max均应有证据等于5，不能只定义
另一个dose字段却让查询使用未赋值的上下界。数字比较和日总量使用算术。

共享上下文因素：
- 同一个主体/时间/模态范围内的多个引用，生成一个 relation=same_scope 的
  group factor，同时列出来源context与相关binding，score为0至4。
  它只在所有这些连接选中同一scope时加分。别把历史推荐与当前实际强行同域。
- 对其他具体兼容关系，可用 {id,when:{choice_id:candidate_id,...},score,reason,sources}，
  score为-4至4。when至少包含两个连接，表示它们联合解释上下文的证据。
- 每条原文线索记一次，不拆成重复两两奖励；没有跨连接证据就留空，不臆造。
- 因素只依据原文的身份、时间、版本、模态、延续/转换线索；不能依据最终答案，
  不能通过偏好公式无矛盾来决定解释。选完以后才求解。

语义要求：
1. facts固定为来源事实/规则；queries固定为待核对命题，不能互换。每条原命题
   恰好一个query，保留原文全部条件，包括缺少证据的条件、否定条件和适用条件。
2. 明确模拟场景里的参数在该场景内是真实值，各字母场景独立。历史参照是比较
   标准，不是本轮实际参数。医生建议某操作不等于已经执行某操作。
3. 未记录/未知字段用facts的expr:null表示，或者不添加fact。它的get仍可在query
   中使用并保持未赋值。显式‘否’才约束为false。未知不是x!=x，也不是x=0。
4. 条件规则用implies；条件未知不能自动推出结果。不用外部医疗常识补全。
5. 若题目声称有定位信息、实际输入却没有，不得假造唯一匹配。保留合理的独立
   方案候选，所有候选必须指向不同的实际scope，而不是同一个表达式换名字。
6. 将必要的实体、时间、模态和执行状态分别体现在绑定与字段中；不要用代表
   整条命题是否正确的单个布尔值。不要加入核对器或修复流程。

只返回当前任务所需的紧凑JSON图。所有facts/queries的字段引用都必须使用get；
context-kind选择通过facts中的get使用，共指/参照binding通过queries或规则中的get使用。
'''


def run_mode(ctx, mode):
    raw = ctx.llm([
        {"role": "system", "content": "You perform source-grounded compositional semantic parsing. Return a JSON graph."},
        {"role": "user", "content": PROMPT + "\n" + json.dumps(ssv._public_problem(ctx.problem), ensure_ascii=False)}
    ], stage="scope_generate", temperature=0, json_mode=True)
    ctx.record("scope_generation_response", {"raw": raw})
    graph = json.loads(raw)
    ctx.record("scope_graph", graph)
    ctx.record("scope_prepared", {"logical_requests": ctx.logical_index,
                                  "graph_sha256": fingerprint(graph)})
    wanted = ssv._claim_ids(ctx.problem)
    actual = [q["item_id"] for q in graph["queries"]]
    if len(actual) != len(set(actual)) or set(actual) != set(wanted):
        raise ValueError("scope_query_ids_differ_from_task")
    scopes = {s["id"] for s in graph["scopes"]}
    if any(c["value"] not in scopes for choice in graph["choices"] for c in choice["candidates"]):
        raise ValueError("scope_candidates_must_target_scopes")
    def check_references(expr):
        if not isinstance(expr, list):
            return
        if expr[0] in ("var", "choose"):
            raise ValueError("scope_content_must_reference_bindings_via_get")
        for child in expr[1:]:
            check_references(child)
    for node in graph["facts"] + graph["queries"]:
        check_references(node["expr"])
    # The model generates only data. The executable compiler is fixed source.
    compiler = Path(__file__).with_name("scope_solver.py").read_text(encoding="utf-8")
    execution = ctx.execute(compiler + "\nresult = solve(" + repr(graph) + ", " + repr(mode) + ")\n",
                            stage="scope_" + mode, timeout=30)
    outcome = execution.get("result") or {}
    ctx.record("scope_selection", outcome.get("selection", {}))
    ctx.record("scope_evaluation", {k: v for k, v in outcome.items() if k != "selection"})
    items = outcome.get("items", [])
    complete = execution.get("ok") and outcome.get("base_status") == "sat" and len(items) == len(wanted)
    result = {"method": "scope_" + mode, "status": "completed" if complete else "failed",
              "items": items if complete else [], "diagnostics": {
                  "graph_sha256": fingerprint(graph), "selection_mode": mode,
                  "choice_count": len(graph.get("choices", [])),
                  "context_choice_count": sum(c.get("kind") == "context" for c in graph["choices"]),
                  "factor_count": len(graph.get("factors", [])),
                  "coupled_factor_count": sum(len(f.get("choices", f.get("when", {}))) > 1 for f in graph.get("factors", [])),
                  "changed_choices": (outcome.get("selection") or {}).get("changed_choices", []),
                  "solver_base_status": outcome.get("base_status"),
                  "verified": False, "model_calls": 1, "implementation": "CPM_scoped_joint_binding"}}
    if not complete:
        result["error"] = execution.get("stderr") or "inconsistent_or_incomplete_formalization"
    ctx.record("result", result)
    return result


def run(ctx):
    return run_mode(ctx, "joint")
