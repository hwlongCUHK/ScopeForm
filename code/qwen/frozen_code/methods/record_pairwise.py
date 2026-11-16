"""Batched pairwise referent comparisons over fixed source records.

All unordered pairs are compared once. A win contributes 1, a tie 0.5;
the highest total selects one complete record. Tied totals retain source order.
Only ambiguous references trigger a single model call for the whole phase.
"""
from copy import deepcopy
from itertools import combinations
import json

from methods import record_shared, record_solver, ssv


PROMPT = """你只做中文原文中的指代消解。给定原文上下文、一个待解析引用以及两个候选记录，
判断A或B哪一个更符合这个引用实际指向的对象/事件/方案。若原文不足以区分，返回tie。
依据原文的共指、参与者身份、时间、版本、建议/实际/条件语义和段落上下文比较。
两个候选都可能与话题相关；比较的是该引用的指向。不能依据哪个候选使待判命题
成立、数值更接近模拟参数或答案更容易计算来选择。原文缺失的定位信息不猜造。
候选记录、字段值和查询表达式已固定，不修改它们，不求命题真假，不输出答案标签。
每个pair独立比较；全部pair在同一个JSON中返回，不省略、不增加pair。
输出格式：{"comparisons":[{"pair_id":"给定ID","winner":"A或B或tie",
"reason":"一句话说明区分线索；无法区分时说明缺少的线索"}]}。
"""


def make_pairs(data, selection, reverse=False):
    records = {r["id"]: r for r in data["records"]}
    refs = {r["id"]: r for r in data["references"]}
    pairs = []
    for rid, domain in selection["candidate_domains"].items():
        for index, (a, b) in enumerate(combinations(domain, 2)):
            if reverse:
                a, b = b, a
            pairs.append({"pair_id": rid + ":" + str(index), "reference_id": rid,
                "reference": {k: v for k, v in refs[rid].items() if k != "scores"},
                "A": records[a], "B": records[b]})
    return pairs


def choose(selection, pairs, comparisons):
    expected = {p["pair_id"] for p in pairs}
    actual = [c["pair_id"] for c in comparisons]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("pairwise_comparisons_differ_from_requested_pairs")
    by_id = {c["pair_id"]: c for c in comparisons}
    scores = {rid: {record: 0.0 for record in domain}
              for rid, domain in selection["candidate_domains"].items() if len(domain) > 1}
    for pair in pairs:
        winner = by_id[pair["pair_id"]]["winner"]
        tally = scores[pair["reference_id"]]
        a, b = pair["A"]["id"], pair["B"]["id"]
        if winner == "tie":
            tally[a] += 0.5
            tally[b] += 0.5
        elif winner in ("A", "B"):
            tally[pair[winner]["id"]] += 1
        else:
            raise ValueError("pairwise_winner_must_be_A_B_or_tie")
    result = deepcopy(selection)
    result["mode"] = "pairwise"
    result["previous_shared_assignments"] = deepcopy(selection["assignments"])
    result["pairwise_scores"] = scores
    result["tied_references"] = [r for r in selection["tied_references"] if r not in scores]
    for rid, tally in scores.items():
        selected = max(tally, key=tally.get)
        if sum(v == tally[selected] for v in tally.values()) > 1:
            result["tied_references"].append(rid)
        result["record_assignments"][rid] = selected
        for field in selection["field_scores"][rid]:
            result["assignments"][rid + "." + field] = selected
    result["field_assignments"] = dict(result["assignments"])
    result["changed_fields"] = [f for f, value in result["assignments"].items()
                                if value != selection["assignments"][f]]
    result["mixed_references"] = []
    return result


def rerank(ctx, data, selection, reverse=False):
    pairs = make_pairs(data, selection, reverse)
    ctx.record("record_numeric_selection", selection)
    if not pairs or selection["empty_domains"]:
        ctx.record("record_pairwise_comparison", {"status": "skipped", "pairs": [],
                   "comparisons": [], "reason": "no_resolvable_ambiguous_reference"})
        return selection, 0
    payload = {"public_problem": ssv._public_problem(ctx.problem), "pairs": pairs}
    ctx.record("record_pairwise_input", payload)
    raw = ctx.llm([
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
    ], stage="record_pairwise_compare", temperature=0, json_mode=True)
    ctx.record("record_pairwise_response", {"raw": raw})
    comparisons = json.loads(raw)["comparisons"]
    result = choose(selection, pairs, comparisons)
    result["pairwise_display_order"] = "reversed" if reverse else "forward"
    ctx.record("record_pairwise_comparison", {"status": "completed", "pairs": pairs,
               "comparisons": comparisons, "scores": result["pairwise_scores"]})
    return result, 1


def run_order(ctx, reverse=False):
    data = record_shared.generate(ctx)
    selection = record_solver.select(data, "shared")
    selection, calls = rerank(ctx, data, selection, reverse)
    mode = "pairwise_reverse" if reverse else "pairwise"
    return record_shared.finish(ctx, data, selection, mode, model_calls=1 + calls)


def run(ctx):
    return run_order(ctx)
