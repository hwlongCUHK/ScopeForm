"""Offline, original-CPM scoring and failure diagnostics for method comparison.

This module is never imported by the runner. Private targets are read only inside
score_run, after the complete planned matrix has been collected. Main metrics
retain all failed and blocked phases; intermediate method labels are diagnostics.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
CORE_EVENTS = ("key_mixed", "single_definite", "single_uncertainty", "neutral")
LABELS = {"supported", "refuted", "not_determined"}
TOKEN_FIELDS = {"input_tokens": "prompt_tokens", "output_tokens": "completion_tokens",
                "prompt_cache_hit_tokens": "prompt_cache_hit_tokens",
                "prompt_cache_miss_tokens": "prompt_cache_miss_tokens"}
DEFAULT_BASELINE = Path("/Users/darrenlong/Desktop/长江/Drug Agent Project/sandbox/drug agent")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_helpers(config):
    path = Path(config.get("baseline_root", DEFAULT_BASELINE)).resolve() / "score.py"
    spec = importlib.util.spec_from_file_location("_cpm_original_score_helpers", path)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    return helpers


def validate_matrix(rows, ids, events, arms, part):
    expected = {(sid, "initial", "initial") for sid in ids}
    expected.update((sid, event, arm) for sid in ids for event in events for arm in arms)
    if not isinstance(rows, list):
        raise ValueError(f"{part}: results must be a list")
    keys = [(r["source_id"], r["event_id"], r["arm"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{part}: duplicate phases")
    if set(keys) != expected:
        raise ValueError(f"{part}: phase matrix differs; missing={len(expected - set(keys))}, extra={len(set(keys) - expected)}")
    for row in rows:
        if not isinstance(row.get("submitted"), bool) or not isinstance(row.get("status"), str):
            raise ValueError(f"{part}: invalid submitted/status")
        if isinstance(row.get("requests"), bool) or not isinstance(row.get("requests"), int) or row["requests"] < 0:
            raise ValueError(f"{part}: invalid request count")


def select(rows, event, arm):
    return [r for r in rows if r["arm"] == arm and
            (r["event_id"] != "initial" if event == "all_update_events" else r["event_id"] == event)]


def metric_values(rows, scorer, metric):
    grouped = defaultdict(list)
    for row in rows:
        if metric == "M":
            value = scorer.content(row)
        else:
            field = {"E": "evidence", "J": "strict"}[metric]
            value = int(bool(row.get("submitted") and (row.get("score") or {}).get(field)))
        grouped[row["source_id"]].append(value)
    return {sid: statistics.fmean(values) for sid, values in sorted(grouped.items())}


def paired(left, right, scorer, **metadata):
    result = {**metadata, "unit": "percentage points", "coverage_adjusted": True,
              "bootstrap": "original mean_ci: 2000 source resamples, seed 20261021",
              "multiple_comparison_adjustment": "none; descriptive paired intervals"}
    for metric in ("M", "E", "J"):
        lv, rv = metric_values(left, scorer, metric), metric_values(right, scorer, metric)
        if set(lv) != set(rv):
            raise ValueError("Paired source sets differ")
        estimate = scorer.mean_ci({sid: lv[sid] - rv[sid] for sid in sorted(lv)})
        result["sources"] = estimate["sources"]
        result["delta_" + metric] = estimate["macro"] * 100 if estimate["macro"] is not None else None
        result["delta_" + metric + "_ci95"] = [v * 100 for v in estimate["ci95"]] if estimate["ci95"] else None
    return result


def phase_folder(folder, row):
    if row.get("phase_relative_path"):
        return folder / row["phase_relative_path"]
    initial = row["event_id"] == "initial"
    return folder / ("shared" if initial else row["arm"]) / "phases" / row["source_id"] / (
        row["event_id"] + "__" + ("initial" if initial else "fresh"))


def item_map(value):
    if isinstance(value, dict):
        value = value.get("items", [])
    if not isinstance(value, list):
        return {}
    return {x["item_id"]: x["label"] for x in value if isinstance(x, dict)
            and isinstance(x.get("item_id"), str) and x.get("label") in LABELS}


def stage_family(stage):
    if stage == "native":
        return "native"
    if stage.startswith("nl2logic."):
        return "nl2logic." + stage.rsplit(".", 1)[-1]
    if stage.startswith("ssr_"):
        for kind in ("formalization", "chain_verbalization", "missing_premise_check", "cot_fallback", "official_llm_call"):
            if kind in stage:
                return "ssr." + kind
    return re.sub(r"_\d+", "", stage)


def request_diagnostics(folder, row):
    accounting_path = folder / "request_accounting.json"
    accounting = read(accounting_path) if accounting_path.exists() else {}
    result = {"accounting_present": bool(accounting), "native_requests": 0,
              "helper_requests": 0, "unclassified_requests": 0,
              "logical_native_calls": 0, "logical_helper_calls": 0,
              "rejected_requests": 0, "unresolved_dispatches": 0,
              "stage_requests": {}, "last_stage": None,
              "native_tokens": {k: 0 for k in TOKEN_FIELDS},
              "helper_tokens": {k: 0 for k in TOKEN_FIELDS},
              "token_fields_unreported": {k: 0 for k in TOKEN_FIELDS}}
    counts = Counter()
    # Guard repair calls live inside guard_rounds, but share the native phase's
    # logical request numbering and physical-call budget.
    attempts_paths = sorted(folder.rglob("call_*.attempts.json"),
                            key=lambda p: int(p.name.split("_")[1].split(".")[0])) if folder.exists() else []
    for path in attempts_paths:
        attempts = read(path)
        stem = path.name[:-len(".attempts.json")]
        response_path = path.parent / (stem + ".response.private.json")
        stage = attempts[-1].get("stage", "unknown") if attempts else "unknown"
        category = "native" if stage == "native" else "helper"
        result["last_stage"] = stage
        result["logical_" + category + "_calls"] += 1
        for attempt in attempts:
            kind = "native" if attempt.get("stage") == "native" else "helper"
            result[kind + "_requests"] += 1
            counts[stage_family(attempt.get("stage", "unknown"))] += 1
            if attempt.get("error", "").startswith("api_http_"):
                result["rejected_requests"] += 1
            elif not attempt.get("error") and not attempt.get("completed_epoch") and not response_path.exists():
                result["unresolved_dispatches"] += 1
        if response_path.exists():
            usage = (read(response_path).get("usage") or {})
            for key, remote in TOKEN_FIELDS.items():
                if usage.get(remote) is None:
                    result["token_fields_unreported"][key] += 1
                else:
                    result[category + "_tokens"][key] += usage[remote]
            unreported = max(0, len(attempts) - 1)
        else:
            unreported = len(attempts)
        for key in TOKEN_FIELDS:
            result["token_fields_unreported"][key] += unreported
    classified = result["native_requests"] + result["helper_requests"]
    result["unclassified_requests"] = max(0, row["requests"] - classified)
    result["physical_requests_from_attempt_logs"] = classified
    result["row_requests"] = row["requests"]
    result["requests_match_row"] = classified == row["requests"] if attempts_paths else row["requests"] == 0
    result["stage_requests"] = dict(counts)
    result["known_tokens_match_row"] = {
        key: result["native_tokens"][key] + result["helper_tokens"][key] == (row.get("usage") or {}).get(key, 0)
        for key in TOKEN_FIELDS
    } if attempts_paths else None
    return result


def method_artifacts(folder, arm):
    method = folder / "method"
    detail = {"artifact_files": len(list(method.glob("*.json"))) if method.exists() else 0,
              "partial_items": {}, "failure_kinds": {}, "solver_failures": 0,
              "solver_executions": 0, "solver_seconds": 0.0}
    failures = Counter()
    for path in sorted(method.glob("solver_*_output.json")):
        value = read(path)
        detail["solver_executions"] += 1
        detail["solver_seconds"] += value.get("seconds", 0) or 0
        candidate = value.get("result") or {}
        if value.get("ok") and isinstance(candidate, dict) and candidate.get("base_status") == "sat":
            labels = item_map(candidate)
            if labels and not detail["partial_items"]:
                # SSV retains the first executable answer while trying later
                # temperatures/verification. It may exhaust its request budget
                # before returning that answer to the native agent.
                detail["partial_items"] = labels
                detail["partial_items_artifact"] = path.name
        if not value.get("ok"):
            detail["solver_failures"] += 1
            text = str(value.get("stderr", ""))
            kind = "timeout" if "timeout" in text.lower() else "solver_execution_error"
            failures[kind] += 1
    if arm == "nl2logic":
        for name in ("nl2logic_progress", "nl2logic_failure"):
            if (method / (name + ".json")).exists():
                detail[name] = read(method / (name + ".json"))
        if "nl2logic_failure" in detail:
            failures["nl2logic." + detail["nl2logic_failure"].get("error_type", "formalization_failed")] += 1
    elif arm == "ssr":
        branches = Counter()
        detail["ssr_completed_claim_records"] = 0
        detail["ssr_parser_true_returns"] = 0
        detail["ssr_conflict_unknowns"] = 0
        detail["ssr_claim_seconds"] = 0.0
        for path in sorted(method.glob("claim_*.json")):
            value = read(path)
            detail["ssr_completed_claim_records"] += 1
            detail["ssr_claim_seconds"] += value.get("seconds", 0) or 0
            branches[value.get("branch", value.get("status", "unknown"))] += 1
            detail["ssr_parser_true_returns"] += len((value.get("execution_trace") or {}).get("parser_true_returns", []))
            detail["ssr_conflict_unknowns"] += int(bool(value.get("official_unknown_from_conflict")))
            label = {"A": "supported", "B": "refuted", "C": "not_determined"}.get((value.get("raw") or {}).get("selected_answer"))
            if label and value.get("item_id"):
                detail["partial_items"][value["item_id"]] = label
            if value.get("status") != "completed":
                failures["ssr." + value.get("status", "unknown")] += 1
        detail["ssr_branches"] = dict(branches)
    elif arm == "ssv":
        temperatures = set()
        detail["ssv_verification_failures"] = 0
        detail["ssv_verification_checks"] = 0
        detail["ssv_verification_passes"] = 0
        for path in sorted(method.glob("*.json")):
            match = re.match(r"temperature_(\d+)_", path.name)
            if match:
                temperatures.add(int(match.group(1)))
            if path.name.endswith("_verification_failure.json"):
                value = read(path)
                detail["ssv_verification_failures"] += 1
                failures["ssv." + value.get("kind", "unknown")] += 1
            elif re.search(r"_verify_\d+\.json$", path.name):
                value = read(path)
                detail["ssv_verification_checks"] += 1
                execution = value.get("execution") or {}
                detail["ssv_verification_passes"] += int(bool(execution.get("ok") and (execution.get("result") or {}).get("passed")))
        detail["ssv_temperature_indices_attempted"] = sorted(temperatures)
    detail["failure_kinds"] = dict(failures)
    return detail


def phase_analysis(folder, row):
    phase = phase_folder(folder, row)
    cp_path = phase / "checkpoint.private.json"
    cp = read(cp_path) if cp_path.exists() else {}
    input_path = phase / "method" / "method_input.json"
    method_input = read(input_path) if input_path.exists() else {}
    result_path = phase / "method" / "method_result.json"
    method_result = cp.get("method_result") or (read(result_path) if result_path.exists() else {})
    proposed = method_input.get("proposed_action") or {}
    pre = item_map(proposed.get("arguments", proposed))
    returned = item_map(method_result)
    final = item_map(row.get("prediction") or {}) if row["submitted"] else {}
    gold = {x["item_id"]: x["to"] for x in row["gold_transition_local"]}
    artifacts = method_artifacts(phase, row["arm"])
    partial = artifacts.pop("partial_items")
    started = cp.get("method_started_at_request", row.get("method_started_at_request")) is not None
    did_return = bool(cp.get("method_completed", row.get("method_completed", False)))
    status = method_result.get("status", row.get("method_status"))
    counts = {"planned_claims": len(gold), "proposed_labels": len(set(pre) & set(gold)),
              "method_labels": len(set(returned) & set(gold)),
              "partial_method_labels": len(set(partial) & set(gold)),
              "submitted_labels": len(set(final) & set(gold)),
              "proposed_correct": 0, "method_correct": 0, "partial_method_correct": 0,
              "submitted_correct": 0, "proposed_wrong_method_right": 0,
              "proposed_right_method_wrong": 0, "method_wrong_final_right": 0,
              "method_right_final_wrong": 0, "method_final_agreement": 0,
              "method_final_comparable": 0}
    transitions = []
    for item_id, expected in gold.items():
        p, m, f, partial_label = pre.get(item_id), returned.get(item_id), final.get(item_id), partial.get(item_id)
        for key, value in (("proposed", p), ("method", m), ("partial_method", partial_label), ("submitted", f)):
            counts[key + "_correct"] += int(value == expected)
        if p is not None and m is not None:
            counts["proposed_wrong_method_right"] += int(p != expected and m == expected)
            counts["proposed_right_method_wrong"] += int(p == expected and m != expected)
        if m is not None and f is not None:
            counts["method_wrong_final_right"] += int(m != expected and f == expected)
            counts["method_right_final_wrong"] += int(m == expected and f != expected)
            counts["method_final_agreement"] += int(m == f)
            counts["method_final_comparable"] += 1
        transitions.append({"item_id": item_id, "gold": expected, "proposed": p,
                            "method": m, "partial_method": partial_label, "submitted": f})
    error_path = phase / "error.json"
    error = read(error_path) if error_path.exists() else {}
    diagnostics = {"phase_folder": str(phase), "checkpoint_present": bool(cp),
                   "method_started": started, "method_returned": did_return,
                   "method_status": status, "method_successfully_completed": status == "completed",
                   "method_has_all_labels": bool(returned) and set(returned) == set(gold),
                   "method_started_at_request": cp.get("method_started_at_request"),
                   "action_count_including_method_boundary": cp.get("action_count"),
                   "native_format_failure": row["status"].startswith("invalid_native_") or
                       row["status"] == "runner_or_transport_JSONDecodeError",
                   "error": error, "label_counts": counts,
                   "label_comparison_is_diagnostic_only": True,
                   "request_accounting": request_diagnostics(phase, row),
                   "method_artifacts": artifacts}
    # The only per-claim gold/prediction comparison goes into the private file.
    row["method_analysis"] = {**diagnostics, "label_transitions_LOCAL_ONLY": transitions}
    return {"source_id": row["source_id"], "event_id": row["event_id"], "arm": row["arm"],
            "submitted": row["submitted"], "status": row["status"], **diagnostics}


def aggregate_analysis(phases):
    statuses, methods, stages, failure_kinds, ssr_branches = Counter(), Counter(), Counter(), Counter(), Counter()
    result = {"phases": len(phases), "submitted": 0, "method_started": 0,
              "method_returned": 0, "method_successfully_completed": 0,
              "method_has_all_labels": 0, "native_format_failures": 0,
              "total_request_limit": 0, "native_requests": 0, "helper_requests": 0,
              "unclassified_requests": 0, "rejected_requests": 0, "unresolved_dispatches": 0,
              "accounting_mismatch_phases": 0, "accounting_missing_nonzero_request_phases": 0,
              "native_tokens": {key: 0 for key in TOKEN_FIELDS},
              "helper_tokens": {key: 0 for key in TOKEN_FIELDS},
              "label_counts": {}, "solver_executions": 0, "solver_failures": 0, "solver_seconds": 0.0,
              "ssr_parser_true_returns": 0, "ssr_conflict_unknowns": 0,
              "ssv_verification_checks": 0, "ssv_verification_passes": 0,
              "ssv_verification_failures": 0}
    label_counts = Counter()
    for phase in phases:
        statuses[phase["status"]] += 1
        methods[phase["method_status"] or ("not_returned" if phase["method_started"] else "not_started")] += 1
        for key in ("submitted", "method_started", "method_returned", "method_successfully_completed", "method_has_all_labels"):
            result[key] += int(phase[key])
        result["native_format_failures"] += int(phase["native_format_failure"])
        result["total_request_limit"] += int(phase["status"] == "total_request_limit")
        label_counts.update(phase["label_counts"])
        accounting = phase["request_accounting"]
        for key in ("native_requests", "helper_requests", "unclassified_requests", "rejected_requests", "unresolved_dispatches"):
            result[key] += accounting[key]
        result["accounting_mismatch_phases"] += int(not accounting["requests_match_row"] and accounting["accounting_present"])
        result["accounting_missing_nonzero_request_phases"] += int(not accounting["accounting_present"] and accounting["row_requests"] > 0)
        stages.update(accounting["stage_requests"])
        for category in ("native_tokens", "helper_tokens"):
            for key in TOKEN_FIELDS:
                result[category][key] += accounting[category][key]
        artifacts = phase["method_artifacts"]
        failure_kinds.update(artifacts["failure_kinds"])
        ssr_branches.update(artifacts.get("ssr_branches", {}))
        for key in ("solver_executions", "solver_failures", "solver_seconds", "ssr_parser_true_returns",
                    "ssr_conflict_unknowns", "ssv_verification_checks", "ssv_verification_passes", "ssv_verification_failures"):
            result[key] += artifacts.get(key, 0)
    result.update(statuses=dict(statuses), method_statuses=dict(methods), stage_requests=dict(stages),
                  method_failure_kinds=dict(failure_kinds), ssr_branches=dict(ssr_branches),
                  label_counts=dict(label_counts))
    result["intermediate_label_metrics_not_main_scores"] = True
    return result


def evaluate_part(folder, ids, events, arms, actors, targets, scorer, helpers, part, layout=None):
    rows = read(folder / "results.json")
    validate_matrix(rows, ids, events, arms, layout or part)
    pairing = read(folder / "pairing_audit.json")
    if any(not isinstance(pairing.get(sid), dict) or not pairing[sid].get("initial_world_sha256") for sid in ids):
        raise ValueError("Actual-initial audit missing")
    enriched = scorer.enrich(rows, {sid: actors[sid] for sid in ids}, {sid: targets[sid] for sid in ids}, pairing)
    phases = [phase_analysis(folder, row) for row in enriched]
    conditions = [helpers.condition(select(enriched, "initial", "initial"), scorer, "initial", "initial")]
    for event in ("all_update_events",) + tuple(events):
        for arm in arms:
            condition = helpers.condition(select(enriched, event, arm), scorer, event, arm)
            subset = [p for p in phases if p["arm"] == arm and (p["event_id"] != "initial" if event == "all_update_events" else p["event_id"] == event)]
            condition["method_analysis"] = aggregate_analysis(subset)
            conditions.append(condition)
    comparisons = []
    if "baseline" in arms:
        for event in ("all_update_events",) + tuple(events):
            for arm in arms:
                if arm != "baseline":
                    comparisons.append(paired(select(enriched, event, arm), select(enriched, event, "baseline"),
                                              scorer, event=event, arm=arm, contrast=arm + " minus baseline"))
    report = {"part": part, "layout": layout, "sources": len(ids), "source_ids": ids,
              "events": list(events), "arms": list(arms), "planned": len(rows), "recorded": len(rows),
              "all_actual_initial_pairs_verified": True, "initial_cost_shared_between_arms": True,
              "conditions": conditions, "paired_methods_minus_baseline": comparisons}
    return report, [{**r, "part": part, "layout": layout} for r in enriched], {
        "part": part, "layout": layout,
        "by_arm": {arm: aggregate_analysis([p for p in phases if p["arm"] == arm]) for arm in ("initial",) + tuple(arms)},
        "phases": phases}


def report_markdown(result, helpers):
    f = helpers.format_number
    interval = helpers.metric_text
    lines = ["# CPM 原版与完整推理方法对照", "",
             "模型：`" + result["metadata"].get("model", "unknown") + "`。所有方法从原版 Fresh agent 接入，不使用笔记组件。", "",
             "M 是原评分器的来源平均答案得分；E 是整包证据通过数；J 是答案、证据、动作状态、权限及回读全部通过数。"
             "未提交、超限和 blocked 均保留在计划分母，主指标不按成功完成的子集重算。", "",
             "每阶段全部原生和方法内部模型调用合计最多 40 次。方法中途产生的标签仅用于诊断，不能代替实际提交。"
             "初始阶段共享并单列成本。配对差值按来源 bootstrap（2000 次），95% 区间未作多重比较校正。", ""]
    for part in [result["core"], *result["layouts"]]:
        name = part["layout"] or "主实验"
        lines += ["## " + name, "", f"来源 {part['sources']}；阶段 {part['recorded']}/{part['planned']}。", "",
                  "| Event | Arm | M [95% CI] | E | J | 提交/计划 | 请求 | 输入 token | 输出 token | 耗时中位数/p95 秒 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for c in part["conditions"]:
            usage, timing = c["usage"], c["phase_seconds"]
            tokens = [("≥" if c["usage_rows_missing"][k] else "") + str(usage[k]) for k in ("input_tokens", "output_tokens")]
            lines.append("| " + " | ".join([c["event"], c["arm"], interval(c["M"], c["M_ci95"]), str(c["E"]), str(c["J"]),
                f"{c['submitted']}/{c['planned']}", str(c["requests"]), *tokens, f"{f(timing['median'])}/{f(timing['p95'])}"]) + " |")
        lines += ["", "配对差值：方法 − 原版，单位为百分点。E/J 在每个来源内平均事件成功率后配对。", "",
                  "| Event | Method | ΔM [95% CI] | ΔE [95% CI] | ΔJ [95% CI] |", "|---|---|---:|---:|---:|"]
        for c in part["paired_methods_minus_baseline"]:
            lines.append("| " + " | ".join([c["event"], c["arm"], *[interval(c["delta_" + m], c["delta_" + m + "_ci95"]) for m in ("M", "E", "J")]]) + " |")
        lines += ["", "方法执行与费用分解（全部更新阶段）：", "",
                  "| Arm | 启动 | 方法返回 | completed | 全部标签 | 总请求超限 | 原生/辅助请求 | 辅助输入/输出 token |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for c in part["conditions"]:
            if c["event"] != "all_update_events":
                continue
            a = c["method_analysis"]
            lines.append("| " + " | ".join([c["arm"], str(a["method_started"]), str(a["method_returned"]),
                str(a["method_successfully_completed"]), str(a["method_has_all_labels"]), str(a["total_request_limit"]),
                f"{a['native_requests']}/{a['helper_requests']}",
                f"{a['helper_tokens']['input_tokens']}/{a['helper_tokens']['output_tokens']}"]) + " |")
        lines += ["", "`方法返回`包括失败或未验证的返回；`completed`仅按方法状态计数，仍不代表答案正确。"
                  "SSR 的官方 commonsense/soft 分支、NL2Logic 的数值表示限制、SSV 的验证与修复失败均在 analysis.json 中单列。", ""]
    if result["paired_layouts"]:
        lines += ["## 分散布局减集中布局", "", "两种布局各有自己的实际初始状态，不假定跨布局初始相同。", "",
                  "| Event | Arm | ΔM [95% CI] | ΔE [95% CI] | ΔJ [95% CI] |", "|---|---|---:|---:|---:|"]
        for c in result["paired_layouts"]:
            lines.append("| " + " | ".join([c["event"], c["arm"], *[interval(c["delta_" + m], c["delta_" + m + "_ci95"]) for m in ("M", "E", "J")]]) + " |")
    lines += ["", "调用阶段、部分执行、求解错误和方法前后标签变化见 `analysis.json`。"
              "逐项 gold 与预测只保存在 `scored_rows.LOCAL_ONLY.private.json`，没有传回模型。", ""]
    return "\n".join(lines)


def score_run(run_dir, output_dir=None):
    run_dir = Path(run_dir).expanduser().resolve()
    config = read(run_dir / "config.json")
    helpers = load_helpers(config)
    root = helpers.package_root(config, run_dir)
    arms = tuple(config["experimental_arms"])
    if not arms or len(set(arms)) != len(arms) or "initial" in arms:
        raise ValueError("experimental_arms must be unique, nonempty update arms")
    ids = helpers.source_ids(config.get("source_ids"), "source_ids")
    layout_ids = helpers.source_ids(config["layout_source_ids"], "layout_source_ids") if config.get("layout_source_ids") else []
    if config.get("study_protocol_sha256") != digest(root / "study15_protocol.json"):
        raise ValueError("Original study protocol hash differs")
    if config.get("dataset_integrity_sha256") != digest(root / "datasets/medical15_confirmation120_silver_v1/build_integrity.json"):
        raise ValueError("Original dataset integrity hash differs")
    sys.path.insert(0, str(root))
    scorer = importlib.import_module("score_study15_local")
    workflow = importlib.import_module("score_workflow11_local")
    if Path(scorer.__file__).resolve().parent != root or scorer.score_row is not workflow.score_row:
        raise ValueError("Original scoring function identity differs")
    dataset = Path(scorer.DATASET).resolve()
    if not dataset.is_relative_to(root):
        raise ValueError("Private evaluator dataset outside original package")
    actors_path = root / "confirmation15_inputs_private/actors.private.jsonl"
    targets_path = dataset / "evaluator_targets.LOCAL_ONLY.private.jsonl"
    selected = set(ids) | set(layout_ids)
    actors = helpers.selected_jsonl(actors_path, selected)
    targets = helpers.selected_jsonl(targets_path, selected)
    core, scored, analysis_core = evaluate_part(run_dir / "main", ids, CORE_EVENTS, arms, actors, targets, scorer, helpers, "core")
    layouts, layout_rows, layout_analysis, paired_layouts = [], {}, [], []
    if layout_ids:
        web = importlib.import_module("web15_runtime")
        web_scorer = importlib.import_module("score_web15_local")
        if not set(layout_ids) <= set(read(root / "web15_protocol.json")["source_ids"]):
            raise ValueError("Layout cohort differs from original protocol")
        for layout in web.LAYOUTS:
            aa = {sid: web.actor(actors[sid], layout) for sid in layout_ids}
            tt = {sid: web_scorer.target(targets[sid], actors[sid], layout) for sid in layout_ids}
            report, rows, diagnostics = evaluate_part(run_dir / "layouts" / layout, layout_ids, tuple(web.EVENTS),
                arms, aa, tt, scorer, helpers, "layout", layout)
            layouts.append(report)
            layout_rows[layout] = rows
            layout_analysis.append(diagnostics)
            scored.extend(rows)
        centralized, distributed = web.LAYOUTS
        for event in ("all_update_events",) + tuple(web.EVENTS):
            for arm in arms:
                paired_layouts.append(paired(select(layout_rows[distributed], event, arm), select(layout_rows[centralized], event, arm),
                    scorer, event=event, arm=arm, contrast="distributed minus centralized", different_actual_initials_per_layout=True))
    result = {"kind": "cpm_full_method_comparison_offline", "run_dir": str(run_dir), "package_root": str(root),
              "metadata": {k: config[k] for k in ("model", "endpoint", "settings", "experimental_arms", "budget", "integration", "method_temperatures") if k in config},
              "core": core, "layouts": layouts, "paired_layouts": paired_layouts,
              "planned_phases": len(scored), "recorded_phases": len(scored),
              "all_actual_initial_pairs_verified": True, "initial_cost_reported_separately": True,
              "private_targets_loaded_only_by_local_scorer": True, "static_and_robustness_evaluated": False,
              "scoring_unit": "source cluster; events and claims within a source are correlated",
              "provenance": {"actors_path": str(actors_path), "actors_sha256": digest(actors_path),
                  "private_targets_path": str(targets_path), "private_targets_sha256": digest(targets_path),
                  "scoring_code_sha256": {name: digest(root / name) for name in
                      ("score_study15_local.py", "score_workflow11_local.py", "score_web15_local.py")},
                  "original_helpers_sha256": digest(Path(config.get("baseline_root", DEFAULT_BASELINE)) / "score.py"),
                  "experiment_scorer_sha256": digest(__file__)}}
    analysis = {"kind": "offline_method_failure_and_cost_diagnostics", "run_dir": str(run_dir),
                "not_a_replacement_for_coverage_adjusted_M_E_J": True,
                "core": analysis_core, "layouts": layout_analysis,
                "reading_notes": ["method_completed in checkpoints means the method returned, not correctness",
                    "partial method labels and proposed labels are not achieved agent submissions",
                    "request counts include explicitly rejected physical attempts; missing usage is unknown",
                    "method failure kind counts can exceed phase counts because one phase may contain several failures",
                    "SSR parser True returns are audit signals, not automatic proof of semantic failure"]}
    output = Path(output_dir).expanduser().resolve() if output_dir else run_dir / "evaluation"
    helpers.write(output / "scored_rows.LOCAL_ONLY.private.json", scored)
    helpers.write(output / "summary.json", result)
    helpers.write(output / "analysis.json", analysis)
    (output / "report.md").write_text(report_markdown(result, helpers), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        result = score_run(args.run_dir, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        parser.exit(1, f"Scoring failed: {error}\n")
    print(json.dumps({"scored_phases": result["recorded_phases"], "core_sources": result["core"]["sources"],
                      "arms": result["metadata"]["experimental_arms"],
                      "output": str(args.output_dir or args.run_dir / "evaluation")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
