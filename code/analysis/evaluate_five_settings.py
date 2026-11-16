"""Original offline scoring, paired effects and input audits for five settings."""
import argparse
from collections import defaultdict
import importlib
import json
from pathlib import Path
import random
import statistics
import sys

from evaluate_methods import digest, item_map, load_helpers, metric_values, phase_folder, read, score_run
from run_five_settings import SETTINGS, GUARDED, dump


def paired_effects(rows, scorer, helpers):
    comparisons = []
    for setting in SETTINGS:
        left = [r for r in rows if r["arm"] == setting + "__formalization"]
        right = [r for r in rows if r["arm"] == setting + "__baseline"]
        comparisons.append({"setting": setting})
        for metric in ("M", "J", "E"):
            lv, rv = metric_values(left, scorer, metric), metric_values(right, scorer, metric)
            assert lv.keys() == rv.keys()
            diffs = [100 * (lv[sid] - rv[sid]) for sid in sorted(lv)]
            rng = random.Random(20261114)
            means = [statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(20000)]
            comparisons[-1][metric] = {
                "delta_pp": statistics.fmean(diffs),
                "ci95_pp": [helpers.percentile(means, p) for p in (.025, .975)],
                "sources": len(diffs),
                "source_wins": sum(d > 0 for d in diffs),
                "source_losses": sum(d < 0 for d in diffs),
                "source_ties": sum(d == 0 for d in diffs),
            }
            if metric == "M":
                # Five two-sided 99% intervals give >=95% simultaneous coverage
                # under the bootstrap approximation; no independence assumption.
                comparisons[-1][metric]["ci99_bonferroni5_pp"] = [
                    helpers.percentile(means, p) for p in (.005, .995)]
    return comparisons


def audit(out, config, rows, root):
    sys.path.insert(0, str(root))
    workflow = importlib.import_module("confirmation14_workflow")
    fingerprint = importlib.import_module("pilot_runtime").fingerprint
    helpers = load_helpers(config)
    actors = helpers.selected_jsonl(root / "confirmation15_inputs_private/actors.private.jsonl",
                                   set(config["source_ids"]))
    errors = defaultdict(list)
    pairs = defaultdict(list)
    for origin, hashes in ((out / "frozen_code", config["code_sha256"]),
                           (root, config["runtime_code_sha256"]),
                           (Path(config["baseline_root"]), config["original_baseline_sha256"])):
        for name, expected in hashes.items():
            if digest(origin / name) != expected:
                errors["code"].append(str(origin / name))
    if digest(out / "run_five_settings.py") != config["harness_sha256"]:
        errors["code"].append("experiment wrapper")
    for row in rows:
        key = [row["source_id"], row["event_id"], row["arm"]]
        analysis = row["method_analysis"]
        ledger = analysis["request_accounting"]
        if not ledger["requests_match_row"] or (ledger["known_tokens_match_row"] and
                                               not all(ledger["known_tokens_match_row"].values())):
            errors["ledger"].append(key)
        if row["requests"] > 40 or (analysis["action_count_including_method_boundary"] or 0) > 40:
            errors["budget"].append(key)
        if row["event_id"] == "initial":
            continue
        setting = row["paper_setting"]
        pairs[tuple(key[:2]) + (setting,)].append(row)
        if row["status"] == "blocked_actual_initial_not_submitted":
            continue
        phase = phase_folder(out / "main", row)
        cp = read(phase / "checkpoint.private.json")
        base_cp = read(out / "main/shared/phases" / row["source_id"] / "initial__initial/checkpoint.private.json")
        base = workflow.restore(base_cp["world"])
        target = next(e["controller"] for e in actors[row["source_id"]]["events"]
                      if e["event_id"] == row["event_id"])
        expected = workflow.branch(base, target, row["event_id"], SETTINGS[setting])
        first_request = read(phase / "call_001.request.private.json")
        if not first_request["messages"][0]["content"].endswith(
                json.dumps(expected.initial_input(), ensure_ascii=False)):
            errors["task_input"].append(key)
        if cp["world"]["support"] != expected.support:
            errors["history"].append(key)
        actual_guard = cp["world"].get("factorial_guard", cp["world"]["arm"] == "evidence_constraint")
        if actual_guard != (setting in GUARDED):
            errors["guard"].append(key)
        if row["variant"] == "formalization" and row["submitted"] and (
                item_map(cp["method_result"]) != item_map(row["prediction"])):
            errors["submitted_solver_labels"].append(key)
        if row["initial_checkpoint_hash"] != fingerprint(base_cp["world"]):
            errors["initial_pairing"].append(key)
    for key, pair in pairs.items():
        if len(pair) != 2 or any(pair[0].get(f) != pair[1].get(f) for f in
                                ("initial_checkpoint_hash", "pre_state_hash", "public_hash")):
            errors["pairing"].append(list(key))
    return {"passed": not errors, "errors": dict(errors), "matched_pairs": len(pairs),
            "original_scoring": True, "budget_includes_all_helpers": True,
            "history_is_actual_initial_prediction_and_acquisition": True}


def evaluate(out):
    report = score_run(out)
    config = read(out / "config.json")
    rows = read(out / "evaluation/scored_rows.LOCAL_ONLY.private.json")
    helpers = load_helpers(config)
    root = helpers.package_root(config, out)
    scorer = importlib.import_module("score_study15_local")
    comparisons = paired_effects(rows, scorer, helpers)
    verification = audit(out, config, rows, root)
    conclusions = {
        "primary_endpoint": "M", "scope": config["scope"],
        "evaluator_sha256": digest(__file__),
        "all_five_M_estimates_positive": all(c["M"]["delta_pp"] > 0 for c in comparisons),
        "all_five_M_simultaneous_intervals_positive": all(
            c["M"]["ci99_bonferroni5_pp"][0] > 0 for c in comparisons),
        "all_five_J_estimates_positive": all(c["J"]["delta_pp"] > 0 for c in comparisons),
        "pilot_cannot_establish_benchmark_wide_claim": config["scope"] == "pilot",
        "verification_passed": verification["passed"],
    }
    conclusions["full_benchmark_all_five_M_gain_supported"] = (
        config["scope"] == "full" and verification["passed"] and
        conclusions["all_five_M_simultaneous_intervals_positive"])
    dump(out / "evaluation/paired_five_settings.json", comparisons)
    dump(out / "evaluation/verification.json", verification)
    dump(out / "evaluation/conclusions.json", conclusions)
    conditions = {c["arm"]: c for c in report["core"]["conditions"] if c["event"] == "all_update_events"}
    labels = dict(zip(SETTINGS, ("Fresh", "Labels", "Labels + Index", "Labels + Guard", "Index + Guard")))
    lines = ["# 五种论文设置：原 Agent 与形式化的配对比较", "",
             f"版本：{config['formalization_version']}；范围：{config['scope']}；来源：{len(config['source_ids'])}。",
             "每对共享实际初始状态、旧标签和索引；所有更新阶段新调用。五组形式化代码相同。",
             "M 为来源平均答案分，未提交记零；J、E 和提交均使用全部计划阶段作分母。", "",
             "| 设置 | 原版 M | 形式化 M | ΔM [95% CI] | ΔM [五重校正区间] | 原版 J | 形式化 J |",
             "|---|---:|---:|---|---|---:|---:|"]
    for c in comparisons:
        setting = c["setting"]
        a, b = conditions[setting + "__baseline"], conditions[setting + "__formalization"]
        effect = c["M"]
        lo, hi = effect["ci95_pp"]
        blo, bhi = effect["ci99_bonferroni5_pp"]
        lines.append(f"| {labels[setting]} | {a['M']:.2f}% | {b['M']:.2f}% | {effect['delta_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}] | [{blo:+.2f}, {bhi:+.2f}] | {a['J']}/{a['planned']} | {b['J']}/{b['planned']} |")
    lines += ["", "按来源配对重采样 20,000 次；同一来源的四个事件一起保留。M 的五重校正使用每组 99% 区间，近似控制五组同时覆盖率为 95%；J/E 的 95% 区间为描述性结果。",
              "", "| 设置 | 版本 | 提交 | E | 请求 | 中位秒 |", "|---|---|---:|---:|---:|---:|"]
    for setting in SETTINGS:
        for variant in ("baseline", "formalization"):
            c = conditions[setting + "__" + variant]
            latency = c["phase_seconds"]["median"]
            lines.append(f"| {labels[setting]} | {variant} | {c['submitted']}/{c['planned']} | {c['E']}/{c['planned']} | {c['requests']} | {latency:.2f} |")
    lines += ["", f"五组 ΔM 点估计全为正：{conclusions['all_five_M_estimates_positive']}。",
              f"五组 M 同时区间下界全高于零：{conclusions['all_five_M_simultaneous_intervals_positive']}。",
              f"输入、原版历史索引/Guard、代码、预算和最终求解标签核查通过：{verification['passed']}。",
              "", "小批只用于接入验证和探索。整个基准此前参与过方法开发，完整结果也不能称为未见数据泛化；不能按设置挑版本。",
              "形式化版本包含既有自动保存/回读流程；J 的变化评估整套接入效果，不能全部归因于逻辑表达本身。"]
    (out / "evaluation/report.md").write_text("\n".join(lines) + "\n")
    return conclusions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(evaluate(args.run_dir.resolve()))
