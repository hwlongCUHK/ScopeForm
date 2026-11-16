"""Kev's pointer choices on the same fixed pairs used by record_pairwise."""
from methods import record_pairwise, record_shared, record_solver, ssv


MODEL = "jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52"
INSTRUCTIONS = record_pairwise.PROMPT.split("每个pair独立比较")[0]


def decision_request(problem, pairs):
    return {"model": "kev-latest", "state": ssv._public_problem(problem),
        "questions": {p["pair_id"]: {"type": "choice",
            "instructions": {"task": INSTRUCTIONS, "reference": p["reference"]},
            "criteria": {"A": p["A"], "B": p["B"],
                         "tie": "原文不足以区分A和B，无法确定该引用指向哪一条记录。"}}
            for p in pairs}}


def rerank(ctx, data, selection, reverse=False):
    pairs = record_pairwise.make_pairs(data, selection, reverse)
    ctx.record("record_numeric_selection", selection)
    if not pairs or selection["empty_domains"]:
        ctx.record("record_pairwise_comparison", {"status": "skipped", "pairs": [],
            "comparisons": [], "reason": "no_resolvable_ambiguous_reference"})
        return selection, 0
    request = decision_request(ctx.problem, pairs)
    ctx.record("record_kev_input", request)
    raw = ctx.chat({"model": MODEL, "kev_request": request}, stage="record_kev_compare")
    decision = raw["decision"]
    ctx.record("record_kev_response", decision)
    comparisons = [{"pair_id": pid, "winner": answer["choice"],
                    "probabilities": answer["probabilities"],
                    "confidence": answer["confidence"]}
                   for pid, answer in decision["answers"].items()]
    result = record_pairwise.choose(selection, pairs, comparisons)
    result["mode"] = "kev"
    result["pairwise_display_order"] = "reversed" if reverse else "forward"
    ctx.record("record_pairwise_comparison", {"status": "completed", "backend": MODEL,
        "pairs": pairs, "comparisons": comparisons, "scores": result["pairwise_scores"],
        "model_latency_ms": decision["latency_ms"], "http_seconds": raw["http_seconds"]})
    return result, 1


def run_order(ctx, reverse=False):
    data = record_shared.generate(ctx)
    selection, calls = rerank(ctx, data, record_solver.select(data, "shared"), reverse)
    return record_shared.finish(ctx, data, selection, "kev_reverse" if reverse else "kev", 1 + calls)


def run(ctx):
    return run_order(ctx)
