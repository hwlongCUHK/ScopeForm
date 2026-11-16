"""Source-grounded diagnosis, followed by execution-error feedback if needed.

The model classifies semantic violations; exception names do not determine the
category. The existing compiler and solver execute the edited graph. No trained
verifier, answer oracle, or automatic fallback is added. Every feedback call
uses the same Context and phase budget as reading and initial diagnosis.
"""
from copy import deepcopy
import json
from pathlib import Path

from common import fingerprint, read
from methods import frame_direct, frame_protocol, ssv

TAXONOMY = Path(__file__).resolve().parents[1] / "definitions/error_taxonomy.json"
TARGET_KEYS = {"properties": "name", "records": "id", "references": "id",
               "rules": "id", "queries": "item_id"}

PROMPT = r'''先判断当前形式化图是否存在具体错误；只有发现错误，才为该错误给出局部修复。
分类按被违反的语义要求决定，不能把异常名直接当成语义类别。
检查四类要求，包括能编译但与原文不符的图。保留正确内容；没有发现问题时issues=[]。
每条reason需说明图中错误位置、当前实际表达的含义，以及它与原文或语言约束的具体冲突。
不能仅以“更明确”“更规范”或可能出错为由修改图；无法指出具体错误就不输出issue。
compiler_feedback来自程序，不是标准答案；编译成功也不证明与原文一致。
其中reference_resolution是程序已经完成的实际引用绑定，不是尚待模型选择的候选。
候选集合包含多个记录本身不是错误；要对照实际选中的记录判断是否错绑。
不能仅因where没有限定成单候选或没有target字段就报告引用错误。
只使用本次公开任务与实际已读取的observations，不使用外部知识补事实。
原文与查询分开：待判命题不是来源事实；未知也不是失败，不为消除未知而编造数值。
按原文选择关系、参数角色和来源，不按期望答案调整。修复共享引用时检查受影响的全部查询。
库接口不支持某种记录时，可用语义等价的原子表达式表达原意，不能伪造记录类别或意图。
原图里有明确抽取错误时可以依据原文修正记录；不要为了编译通过改变正确的主体、时间或事实。
需要多个字段共同修复时放在同一issue的edits中。所有修改按数组顺序应用。

只输出JSON：{"issues":[{"category":"分类ID","reason":"违反哪项要求及为什么",
"evidence":[{"document_id":"已读取文档ID","anchor_id":"原文锚点","quote":"简短原文"}],
"edits":[{"collection":"references","id":"要修改的引用ID",
"path":["slots","frequency"],"value":"daily_frequency"}]}]}。
每个issue必须给出具体修改。collection为properties/records/references/rules/queries之一。
目标标识字段与原对象一致：records、references、rules写id，queries写item_id，properties写name。
例如修改查询C3：{"collection":"queries","item_id":"C3","path":["expr"],"value":新表达式}。
queries、properties也兼容统一的id字段，填写同一个目标标识值。
程序按该标识查找对象，不用数组下标定位记录或命题。path只写该对象内部的字段路径，
例如改记录类别用["kind"]，改查询表达式用["expr"]，补槽映射用["slots","frequency"]。
可以添加末级字段；替换表达式时使用下述op/args协议。
新增或整体替换一个对象时path=[]、value为完整对象，其标识必须与补丁指定的目标标识相同；
删除一个对象时path=[]、value=null。对象内部增删数组项或字典键时替换其所在容器。
已有对象只需修改相关字段。queries的item_id与text保持原样。
evidence_ids若需要调整，只能选择已读取的原文锚点。不要输出答案标签或额外的通过判定。
纯表示问题的evidence可以为空；涉及来源含义的修改请给出原文依据。

下面是当前形式化协议和语义目录，它们也约束修复输出：
'''


def load_taxonomy():
    return read(TAXONOMY)


def prompt_taxonomy(taxonomy):
    return [{k: row[k] for k in ("id", "name", "requirement", "question", "boundary")}
            for row in taxonomy["categories"]]


def edit_target_id(edit):
    """Use the collection's own identity field; keep the earlier id spelling."""
    collection = edit["collection"]
    identity = TARGET_KEYS[collection]
    if identity in edit:
        target = edit[identity]
        if "id" in edit and edit["id"] != target:
            raise ValueError(f"repair_conflicting_target: {collection}.{identity}={target!r} "
                             f"differs from id={edit['id']!r}")
        return target
    if "id" in edit:
        return edit["id"]
    raise ValueError(f"repair_missing_target: collection={collection}; "
                     f"provide {identity!r} with the target object's identifier")


def apply_repairs(graph, diagnosis):
    """Locate named objects by ID, then apply paths relative to those objects."""
    categories = {c["id"] for c in load_taxonomy()["categories"]}
    result = deepcopy(graph)
    for issue in diagnosis["issues"]:
        if issue["category"] not in categories:
            raise ValueError("repair_unknown_category")
        if not issue["reason"] or not isinstance(issue["evidence"], list) or not isinstance(issue["edits"], list):
            raise ValueError("repair_issue_requires_reason_evidence_edits")
        for edit in issue["edits"]:
            collection, target, path = edit["collection"], edit_target_id(edit), edit["path"]
            identity = TARGET_KEYS[collection]
            rows = result[collection]
            positions = {row[identity]: i for i, row in enumerate(rows)}
            if not isinstance(path, list):
                raise ValueError("repair_path_must_be_relative_list")
            if not path:
                value = deepcopy(edit["value"])
                if value is None:
                    del rows[positions[target]]
                else:
                    if value[identity] != target:
                        raise ValueError("repair_object_identity_mismatch")
                    if target in positions:
                        rows[positions[target]] = value
                    else:
                        rows.append(value)
                continue
            parent = rows[positions[target]]
            for part in path[:-1]:
                parent = parent[part]
            parent[path[-1]] = deepcopy(edit["value"])
    original_queries = [(q["item_id"], q["text"]) for q in graph["queries"]]
    if [(q["item_id"], q["text"]) for q in result["queries"]] != original_queries:
        raise ValueError("repair_changed_task_queries")
    return result


def solve_candidate(ctx, graph):
    try:
        return frame_direct.solve(ctx, graph)
    except (KeyError, ValueError, TypeError, IndexError) as error:
        return {"method": "frame_direct", "status": "implementation_error", "items": [],
                "error_type": type(error).__name__, "error": str(error)}


def compiler_feedback(ctx, result):
    diagnostics = result.get("diagnostics", {})
    feedback = {k: result[k] for k in ("status", "error", "error_type", "error_stage") if k in result}
    feedback.update({k: diagnostics[k] for k in ("empty_domains", "solver_base_status") if k in diagnostics})
    if result["status"] == "completed":
        selection = read(ctx.folder / "method" / "record_selection.json")
        feedback["reference_resolution"] = {k: selection[k] for k in (
            "record_assignments", "field_assignments", "candidate_domains", "tied_references")}
    return feedback


def run(ctx, graph):
    """Stop after successful execution; unresolved errors consume the shared budget."""
    taxonomy = load_taxonomy()
    before = solve_candidate(ctx, graph)
    ctx.record("repair_original_graph", graph)
    ctx.record("repair_before_result", before)
    feedback = compiler_feedback(ctx, before)
    payload = {"problem": ssv._public_problem(ctx.problem), "graph": graph,
               "compiler_feedback": feedback, "taxonomy": prompt_taxonomy(taxonomy)}
    ctx.record("repair_input", payload)
    requests_before = ctx.requests
    messages = [
        {"role": "system", "content": PROMPT + frame_protocol.PROMPT + "\n" +
         json.dumps(frame_protocol.catalog_for_prompt(), ensure_ascii=False, separators=(",", ":"))},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
    ]
    current, result = deepcopy(graph), deepcopy(before)
    rounds, issues = 0, []
    while True:
        # Context.llm enforces the one shared physical-request cap. No local cap
        # or budget reset: even an unchanged failed graph needs a real repair.
        raw = ctx.llm(messages, stage="frame_diagnose_repair" if rounds == 0 else "frame_repair_feedback",
                      temperature=0, json_mode=True)
        rounds += 1
        if rounds == 1:
            ctx.record("repair_response", {"raw": raw, "taxonomy_sha256": fingerprint(taxonomy)})
        diagnosis = None
        try:
            diagnosis = json.loads(raw)
            if rounds == 1:
                ctx.record("repair_diagnosis", diagnosis)
            repaired = apply_repairs(current, diagnosis) if diagnosis["issues"] else deepcopy(current)
            issues.extend(diagnosis["issues"])
            result = (solve_candidate(ctx, repaired)
                      if repaired != current or result.get("error_stage") == "repair_output"
                      else deepcopy(result))
            current = repaired
        except (KeyError, ValueError, TypeError, IndexError) as error:
            result = {"method": "frame_direct_repair", "status": "implementation_error", "items": [],
                      "error_stage": "repair_output", "error_type": type(error).__name__, "error": str(error)}
        result["method"] = "frame_direct_repair"
        result.setdefault("diagnostics", {}).update(
            extra_model_calls=ctx.requests - requests_before, repair_rounds=rounds,
            repair_issue_count=len(issues), repair_changed_graph=current != graph,
            repair_categories=[issue["category"] for issue in issues],
            taxonomy_sha256=fingerprint(taxonomy))
        ctx.record(f"repair_round_{rounds:03d}", {"round": rounds, "request": ctx.requests,
                   "compiler_feedback": feedback, "diagnosis": diagnosis,
                   "graph_after": current, "result_after": result})
        ctx.record("repair_final_graph", current)
        ctx.record("repair_after_result", result)
        ctx.record("result", result)
        if result["status"] == "completed":
            return current, result
        feedback = compiler_feedback(ctx, result)
        messages += [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": json.dumps({
                "instruction": "修复后的图仍无法执行。根据本次具体报错继续局部修复，按ID指定要修改的对象。"
                               "只修复报错及其依赖，不重新全面改写；原文、任务与协议沿用上文。",
                "graph": current, "compiler_feedback": feedback,
            }, ensure_ascii=False, separators=(",", ":"))},
        ]
