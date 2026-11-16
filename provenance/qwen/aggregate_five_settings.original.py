"""Combine completed, independently scored Guard and remaining-four Qwen runs.

This offline script reads existing scored rows; it never calls inference, reloads
private targets, rescores predictions, or modifies either source run. The output
is an aggregate of two runs, not a newly executed five-setting run.
The original source-hash assignment is shared. Remaining-four endpoint overrides
move 12 untouched whole sources from replica 18228 to 18227 without interrupting
active model requests; each source-setting comparison retains one endpoint.
The existing Guard run, label scorer, and inference settings remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
SETTING_LABELS = {"fresh": "Fresh", "labels": "Labels", "index": "Labels + Index",
                  "guard": "Labels + Guard", "index_guard": "Index + Guard"}
VARIANTS = ("baseline", "formalization")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def phase_key(row):
    return row["source_id"], row["event_id"], row["arm"]


def load_run(path, settings, ev):
    config = ev.read(path / "config.json")
    rows = ev.read(path / "evaluation/scored_rows.LOCAL_ONLY.private.json")
    summary = ev.read(path / "evaluation/summary.json")
    verification = ev.read(path / "evaluation/verification.json")
    arms = {setting + "__" + variant for setting in settings for variant in VARIANTS}
    require(ev.read(path / "run_status.json")["status"] == "complete", f"Incomplete run: {path}")
    require(config["scope"] == "full" and len(config["source_ids"]) == 120, f"Not the full cohort: {path}")
    require(len(set(config["source_ids"])) == 120, f"Duplicate sources: {path}")
    require(set(config["experimental_arms"]) == arms, f"Unexpected setting matrix: {path}")
    ev.validate_matrix(rows, config["source_ids"], ev.CORE_EVENTS, arms, str(path))
    complete = ev.read(path / "main/complete.json")
    require(complete == {"planned": len(rows), "recorded": len(rows)}, f"Completion counts differ: {path}")
    require(summary["recorded_phases"] == len(rows), f"Scored summary is incomplete: {path}")
    require(verification["passed"] and not verification["errors"], f"Input scoring audit failed: {path}")
    transport = verification["transport"]
    require(transport["passed"] and transport["replica_routing"]["passed"], f"Transport audit failed: {path}")
    require(transport["replica_routing"]["enabled"], f"Source routing was not audited: {path}")
    ports = config["replica_ports"]
    expected_routes = {
        sid: f"http://127.0.0.1:{ports[int(hashlib.sha256(sid.encode()).hexdigest(), 16) % len(ports)]}/v1"
        for sid in config["source_ids"]}
    require(config["source_endpoint_map"] == expected_routes, f"Source-hash routing differs: {path}")
    require(not config.get("reused_baseline") and not config.get("reused_native_prefixes"),
            f"This aggregate expects all update requests to have run live: {path}")
    update_calls = sum(row["requests"] for row in rows if row["event_id"] != "initial")
    require(update_calls == transport["physical_transport_requests"] == transport["charged_requests"],
            f"Physical update request accounting differs: {path}")
    for row in rows:
        analysis = row["method_analysis"]
        ledger = analysis["request_accounting"]
        require(row["requests"] <= 40 and (analysis["action_count_including_method_boundary"] or 0) <= 40,
                f"Budget exceeded: {phase_key(row)}")
        require(ledger["requests_match_row"], f"Request ledger mismatch: {phase_key(row)}")
        require(not ledger["known_tokens_match_row"] or all(ledger["known_tokens_match_row"].values()),
                f"Known token ledger mismatch: {phase_key(row)}")
    files = ("config.json", "run_status.json", "main/complete.json", "main/results.json",
             "main/pairing_audit.json", "evaluation/scored_rows.LOCAL_ONLY.private.json",
             "evaluation/summary.json", "evaluation/verification.json", "run_qwen.py", "run_five_settings.py")
    hashes = {name: ev.digest(path / name) for name in files}
    require(hashes["run_qwen.py"] == config["harness_sha256"], f"Runner wrapper changed: {path}")
    require(hashes["run_five_settings.py"] == verification["wrapper_hashes"]["expected_five_wrapper"],
            f"Five-setting environment wrapper changed: {path}")
    return {"path": path, "config": config, "rows": rows, "summary": summary,
            "verification": verification, "file_sha256": hashes}


def aggregate(guard_path, remaining_path, output, evaluator_dir):
    sys.path.insert(0, str(evaluator_dir))
    ev = importlib.import_module("evaluate_methods")
    five = importlib.import_module("evaluate_five_settings")
    settings = importlib.import_module("run_five_settings").SETTINGS
    require(set(settings) == set(SETTING_LABELS), "Unexpected five-setting definitions")
    guard = load_run(guard_path, {"guard"}, ev)
    remaining = load_run(remaining_path, set(settings) - {"guard"}, ev)
    inputs = (guard, remaining)
    a, b = guard["config"], remaining["config"]
    equal_fields = ("model", "formalization_version", "actors_sha256", "code_sha256", "runtime_code_sha256",
                    "original_baseline_sha256", "study_protocol_sha256", "dataset_integrity_sha256",
                    "native_tool_choice", "replica_ports", "source_endpoint_map", "method_temperatures")
    for field in equal_fields:
        require(a[field] == b[field], f"Runs differ in {field}")
    inference_fields = ("temperature", "max_output_tokens", "max_actions", "max_requests", "thinking")
    for field in inference_fields:
        require(a["settings"][field] == b["settings"][field], f"Inference setting differs: {field}")
    require(a["settings"]["max_requests"] == a["settings"]["max_actions"] == 40, "Expected shared 40-call/action budget")
    ids = sorted(a["source_ids"])
    require(ids == sorted(b["source_ids"]), "Source cohorts differ")
    require(guard["file_sha256"]["run_five_settings.py"] == remaining["file_sha256"]["run_five_settings.py"],
            "Setting implementations differ")
    provenance_fields = ("actors_sha256", "private_targets_sha256", "scoring_code_sha256",
                         "original_helpers_sha256", "experiment_scorer_sha256")
    for field in provenance_fields:
        require(guard["summary"]["provenance"][field] == remaining["summary"]["provenance"][field],
                f"Original scoring provenance differs: {field}")

    helpers = ev.load_helpers(a)
    root = helpers.package_root(a, guard_path)
    sys.path.insert(0, str(root))
    scorer = importlib.import_module("score_study15_local")
    fingerprint = importlib.import_module("pilot_runtime").fingerprint
    require(Path(scorer.__file__).resolve().parent == root, "Unexpected original scoring module")
    initial_hashes = {}
    initial_rows = []
    first_initials = {r["source_id"]: r for r in guard["rows"] if r["event_id"] == "initial"}
    second_initials = {r["source_id"]: r for r in remaining["rows"] if r["event_id"] == "initial"}
    initial_fields = ("submitted", "status", "requests", "usage", "pre_state_hash", "public_hash", "prediction", "score")
    for sid in ids:
        relative = Path("main/shared/phases") / sid / "initial__initial/checkpoint.private.json"
        left, right = ev.read(guard_path / relative), ev.read(remaining_path / relative)
        require(left["terminal"] and right["terminal"] and left["world"] == right["world"],
                f"Initial worlds differ: {sid}")
        initial_hashes[sid] = fingerprint(left["world"])
        require(all(first_initials[sid].get(f) == second_initials[sid].get(f) for f in initial_fields),
                f"Reused initial result differs: {sid}")
        initial_rows.append(first_initials[sid])

    for item in inputs:
        config, path = item["config"], item["path"]
        package_root = ev.load_helpers(config).package_root(config, path)
        for origin, hashes in ((path / "frozen_code", config["code_sha256"]),
                               (package_root, config["runtime_code_sha256"]),
                               (Path(config["baseline_root"]), config["original_baseline_sha256"])):
            for name, expected in hashes.items():
                require(ev.digest(origin / name) == expected, f"Frozen code changed: {origin / name}")
        for row in item["rows"]:
            if row["event_id"] != "initial":
                require(row["initial_checkpoint_hash"] == initial_hashes[row["source_id"]],
                        f"Update does not use the shared actual initial: {phase_key(row)}")

    updates = [r for item in inputs for r in item["rows"] if r["event_id"] != "initial"]
    rows = initial_rows + sorted(updates, key=phase_key)
    arms = tuple(setting + "__" + variant for setting in settings for variant in VARIANTS)
    ev.validate_matrix(rows, ids, ev.CORE_EVENTS, arms, "combined five-setting aggregate")
    require(len(rows) == 4920 and len(updates) == 4800, "Expected 120 shared initials plus 4800 updates")
    for setting in settings:
        for sid in ids:
            for event in ev.CORE_EVENTS:
                pair = [r for r in updates if r["source_id"] == sid and r["event_id"] == event
                        and r["arm"].startswith(setting + "__")]
                require(len(pair) == 2 and all(pair[0].get(f) == pair[1].get(f)
                                              for f in ("initial_checkpoint_hash", "pre_state_hash", "public_hash")),
                        f"Matched task input differs: {sid}/{event}/{setting}")

    termination = ev.read(guard_path / "evaluation/termination_audit.json")
    stop_key = (termination["source_id"], termination["event_id"], termination["arm"])
    stopped = next(row for row in rows if phase_key(row) == stop_key)
    require(termination["termination_cause"] == "user_requested_stop", "Unexpected Guard termination audit")
    require(not stopped["submitted"] and stopped["status"] == termination["raw_terminal_status_preserved"]
            and stopped["requests"] == termination["final_actual_request_count"], "User-stopped row was changed")
    require(scorer.content(stopped) == 0, "User-stopped row must retain zero coverage-adjusted score")
    guard["file_sha256"]["evaluation/termination_audit.json"] = ev.digest(guard_path / "evaluation/termination_audit.json")
    guard["file_sha256"]["user_requested_stop.json"] = ev.digest(guard_path / "user_requested_stop.json")

    conditions = [helpers.condition(initial_rows, scorer, "initial", "initial")]
    for event in ("all_update_events",) + tuple(ev.CORE_EVENTS):
        for arm in arms:
            conditions.append(helpers.condition(ev.select(rows, event, arm), scorer, event, arm))
    by_condition = {(c["event"], c["arm"]): c for c in conditions}
    for item in inputs:
        for old in item["summary"]["core"]["conditions"]:
            new = by_condition[(old["event"], old["arm"])]
            require(math.isclose(new["M"], old["M"], abs_tol=1e-10), f"Aggregation changed M: {old['arm']}")
            require(all(new[f] == old[f] for f in ("planned", "submitted", "E", "J", "requests", "usage")),
                    f"Aggregation changed counts/costs: {old['arm']}")
    five.SETTINGS = settings
    comparisons = five.paired_effects(rows, scorer, helpers)
    sources = [{"run_dir": str(item["path"]), "settings": sorted({arm.split("__")[0] for arm in item["config"]["experimental_arms"]}),
                "recorded_phases": len(item["rows"]), "input_sha256": item["file_sha256"],
                "wrapper_sha256": item["config"]["harness_sha256"], "scheduling": item["config"].get("scheduling", {"kind": "shared_executor"}),
                "scheduling_rebalance": item["config"].get("scheduling_rebalance")}
               for item in inputs]
    audit = {"passed": True, "kind": "two_completed_runs_offline_aggregation", "model": a["model"], "input_runs": sources,
             "same_frozen_method_runtime_actors_and_scorer": True, "same_actual_initial_worlds": initial_hashes,
              "initial_source_endpoint_map": a["source_endpoint_map"],
              "effective_source_endpoint_maps_by_run": {
                  str(item["path"]): {**item["config"]["source_endpoint_map"], **item["config"].get("source_endpoint_overrides", {})}
                  for item in inputs},
              "source_counts_by_endpoint_by_run": {
                  str(item["path"]): dict(Counter({**item["config"]["source_endpoint_map"], **item["config"].get("source_endpoint_overrides", {})}.values()))
                  for item in inputs},
              "same_replica_within_each_source_setting_comparison": True,
             "same_inference_settings": {field: a["settings"][field] for field in inference_fields},
             "matched_source_event_setting_pairs": 2400, "unique_initial_phases": 120,
             "duplicate_initial_rows_removed": 120, "update_phases": 4800, "total_rows": 4920,
             "all_failed_and_blocked_rows_retained": True, "all_existing_condition_metrics_preserved": True,
             "all_physical_calls_share_40_request_budget": True, "original_scoring_verifications_passed": True,
             "guard_termination": termination, "standard_uninterrupted_completion": False,
             "aggregate_does_not_relabel_user_stop_as_budget_exhaustion": True,
             "aggregator_sha256": ev.digest(__file__), "paired_effects_code_sha256": ev.digest(Path(five.__file__))}
    conclusions = {"primary_endpoint": "M", "scope": "full", "verification_passed": True,
                   "all_five_M_estimates_positive": all(c["M"]["delta_pp"] > 0 for c in comparisons),
                   "all_five_M_simultaneous_intervals_positive": all(c["M"]["ci99_bonferroni5_pp"][0] > 0 for c in comparisons),
                   "all_five_J_estimates_positive": all(c["J"]["delta_pp"] > 0 for c in comparisons),
                   "aggregate_of_two_runs": True, "guard_user_stop_retained_as_failure": True}
    metrics = {"kind": "qwen_five_setting_aggregate_of_two_frozen_runs", "model": a["model"], "sources": 120,
               "recorded_phases": len(rows), "unique_initial_phases": 120, "update_phases": 4800,
               "input_runs": sources, "conditions": conditions,
               "rows": [{"setting": c["arm"].split("__")[0], "variant": c["arm"].split("__")[1], **c}
                        for c in conditions if c["event"] == "all_update_events"],
               "intervals": "20,000 source-paired bootstrap resamples, seed 20261114; 95% for M/J/E; per-model five-comparison Bonferroni 99% for primary M",
               "J_E_intervals": "Descriptive 95%; not multiplicity-adjusted", "initial_cost_counted_once": True,
               "M_unit": "source-macro coverage-adjusted percent; unsubmitted phases score zero",
               "E_J_unit": "counts / 480 planned updates per arm", "guard_termination": termination,
               "standard_uninterrupted_completion": False, "verification_passed": True}
    lines = ["# Qwen 五设置配对汇总", "", f"模型：{a['model']}。合并两个冻结方法运行；120 个共同实际初始仅计一次，合计 4800 个更新阶段。",
             "", f"| 设置 | {a['model']} M | +ScopeForm M | ΔM [95% CI] | ΔM [五重校正 99% CI] | {a['model']} J | +ScopeForm J |",
             "|---|---:|---:|---|---|---:|---:|"]
    for comparison in comparisons:
        setting = comparison["setting"]
        left = by_condition[("all_update_events", setting + "__baseline")]
        right = by_condition[("all_update_events", setting + "__formalization")]
        effect = comparison["M"]
        lo, hi = effect["ci95_pp"]
        blo, bhi = effect["ci99_bonferroni5_pp"]
        lines.append(f"| {SETTING_LABELS[setting]} | {left['M']:.2f} | {right['M']:.2f} | {effect['delta_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}] | [{blo:+.2f}, {bhi:+.2f}] | {left['J']}/480 | {right['J']}/480 |")
    lines += ["", "按来源簇配对重采样 20,000 次，同一来源的四个事件共同保留。主指标 M 的每组 99% 区间用于每模型五组比较的 Bonferroni 校正；J/E 仅报告描述性 95% 区间。",
              "", f"旧 Guard 的 {stop_key[0]}/{stop_key[1]} 为用户终止失败，保留 {stopped['requests']} 次实际请求和零分；不是 40 次预算耗尽。两次运行的调度 wrapper 可不同；冻结方法、任务输入、模型、推理设置及原标签评分器保持不变。原始来源哈希分配相同，但补跑四设置通过 config.source_endpoint_overrides 将 12 个尚未开始的完整来源从慢副本 18228 迁至快副本 18227。每个来源和设置内，baseline 与 ScopeForm 的四个更新事件仍使用同一端点。旧 Guard 不变，迁移未中断任何活动模型请求。",
              "", "初始成本只计一次；所有失败和 blocked 阶段保留在分母。此目录是两次完成运行的离线汇总，不声称是一次连续全量执行。"]
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("combined_metrics.json", metrics), ("paired_five_settings.json", comparisons),
                        ("input_audit.json", audit), ("conclusions.json", conclusions),
                        ("combined_scored_rows.LOCAL_ONLY.private.json", rows)):
        helpers.write(output / name, value)
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"output_dir": str(output), "recorded_phases": len(rows), **conclusions}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guard-run", type=Path, required=True)
    parser.add_argument("--remaining-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluator-dir", type=Path, required=True,
                        help="Directory containing the unchanged evaluate_methods.py and evaluate_five_settings.py")
    args = parser.parse_args()
    print(json.dumps(aggregate(args.guard_run.resolve(), args.remaining_run.resolve(),
                               args.output_dir.resolve(), args.evaluator_dir.resolve()), ensure_ascii=False))
