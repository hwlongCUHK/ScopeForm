"""Offline scoring for the paired CPM versus minimal-notes experiment.

Only this process loads the original private evaluator targets. It imports the
frozen scoring functions without writing bytecode into the original package.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


sys.dont_write_bytecode = True
ARMS = ("baseline", "notes")
CORE_EVENTS = ("key_mixed", "single_definite", "single_uncertainty", "neutral")
TOKEN_KEYS = (
    "input_tokens", "output_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
)
PACKAGE_NAME = "CPM_Sandbox_collaborator_complete_20261112"


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def package_root(config: dict, run_dir: Path) -> Path:
    configured = config.get("package_root")
    if configured:
        root = Path(configured).expanduser()
        if not root.is_absolute():
            root = run_dir / root
    else:
        root = Path(__file__).resolve().parent.parent / PACKAGE_NAME / "medical_scale_v06"
    root = root.resolve()
    if (root / "medical_scale_v06").is_dir():
        root = root / "medical_scale_v06"
    if not (root / "score_study15_local.py").is_file():
        raise ValueError("Original scorer not found; set config.json package_root to medical_scale_v06")
    return root


def source_ids(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not value or any(not isinstance(sid, str) or not sid for sid in value):
        raise ValueError(f"{name} must be a nonempty list of source IDs")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} contains duplicate source IDs")
    return value


def selected_jsonl(path: Path, selected: set[str]) -> dict[str, dict]:
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["source_id"] in selected:
                if row["source_id"] in records:
                    raise ValueError(f"Duplicate source ID in {path.name}")
                records[row["source_id"]] = row
    if set(records) != selected:
        raise ValueError(f"Planned source IDs missing from {path.name}")
    return records


def matrix(rows: Any, ids: list[str], events: tuple[str, ...], part: str) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"{part}: results.json must contain a row list")
    expected = {(sid, "initial", "initial") for sid in ids}
    expected.update((sid, event, arm) for sid in ids for event in events for arm in ARMS)
    keys = [(r["source_id"], r["event_id"], r["arm"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{part}: duplicate phase rows")
    if set(keys) - expected:
        raise ValueError(f"{part}: results contain phases outside the configured source/event/arm matrix")
    if expected - set(keys):
        raise ValueError(f"{part}: run is incomplete ({len(expected - set(keys))} planned phase rows missing)")
    for row in rows:
        if not isinstance(row.get("submitted"), bool) or not isinstance(row.get("status"), str):
            raise ValueError(f"{part}: invalid submitted/status fields")
        requests = row.get("requests")
        if isinstance(requests, bool) or not isinstance(requests, int) or requests < 0:
            raise ValueError(f"{part}: invalid request count")


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def costs(rows: list[dict]) -> dict:
    totals = {key: 0 for key in TOKEN_KEYS}
    missing = {key: 0 for key in TOKEN_KEYS}
    missing_requests = {key: 0 for key in TOKEN_KEYS}
    times = []
    for row in rows:
        usage = row.get("usage") or {}
        if not isinstance(usage, dict):
            raise ValueError("Row usage must be a JSON object")
        unreported = usage.get("unreported_requests") or {}
        for key in TOKEN_KEYS:
            value = usage.get(key)
            if row["requests"] > 0 and (value is None or unreported.get(key, 0) > 0):
                missing[key] += 1
                missing_requests[key] += unreported.get(key, 0) or row["requests"]
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {key} usage")
            totals[key] += value
        seconds = row.get("seconds")
        if seconds is not None:
            if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
                raise ValueError("Invalid phase seconds")
            times.append(seconds)
    return {
        "usage": totals,
        "usage_rows_missing": missing,
        "usage_requests_missing": missing_requests,
        "usage_complete": not any(missing.values()),
        "phase_seconds": {
            "median": statistics.median(times) if times else None,
            "p95": percentile(times, 0.95),
            "observed_phases": len(times),
            "missing_phases": len(rows) - len(times),
            "population": "all recorded phases, including failed and blocked phases",
            "percentile_method": "linear interpolation at (n - 1) * p",
        },
    }


def condition(rows: list[dict], scorer: Any, event: str, arm: str) -> dict:
    result = scorer.summary(rows)
    metric = result["content_source_macro_coverage_adjusted"]
    return {
        "event": event, "arm": arm,
        **result,
        "M": metric["macro"] * 100 if metric["macro"] is not None else None,
        "M_ci95": [value * 100 for value in metric["ci95"]] if metric["ci95"] else None,
        "E": result["evidence_packets"], "J": result["strict_packets"],
        "sources": metric["sources"],
        **costs(rows),
    }


def source_means(rows: list[dict], scorer: Any) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row in rows:
        values.setdefault(row["source_id"], []).append(scorer.content(row))
    return {sid: statistics.fmean(scores) for sid, scores in values.items()}


def comparison(left: list[dict], right: list[dict], scorer: Any, **metadata: Any) -> dict:
    lv, rv = source_means(left, scorer), source_means(right, scorer)
    if set(lv) != set(rv):
        raise ValueError("Paired comparison source sets differ")
    result = scorer.mean_ci({sid: lv[sid] - rv[sid] for sid in sorted(lv)})
    return {
        **metadata, **result,
        "delta_M": result["macro"] * 100 if result["macro"] is not None else None,
        "delta_M_ci95": [value * 100 for value in result["ci95"]] if result["ci95"] else None,
        "coverage_adjusted": True, "unit": "percentage points",
        "bootstrap": "original mean_ci: 2000 source resamples, seed 20261021",
    }


def event_rows(rows: list[dict], event: str, arm: str) -> list[dict]:
    return [row for row in rows if row["arm"] == arm and
            (row["event_id"] != "initial" if event == "all_update_events" else row["event_id"] == event)]


def evaluate_part(folder: Path, ids: list[str], events: tuple[str, ...], actors: dict,
                  targets: dict, scorer: Any, part: str, layout: str | None = None) -> tuple[dict, list[dict]]:
    if not (folder / "results.json").is_file():
        raise ValueError(f"{part}: run is incomplete (results.json missing)")
    rows = read(folder / "results.json")
    matrix(rows, ids, events, part)
    if not (folder / "pairing_audit.json").is_file():
        raise ValueError(f"{part}: run is incomplete (pairing_audit.json missing)")
    pairing = read(folder / "pairing_audit.json")
    if any(not isinstance(pairing.get(sid), dict) or not pairing[sid].get("initial_world_sha256") for sid in ids):
        raise ValueError(f"{part}: actual initial checkpoint audit is incomplete")
    # Only the planned subset is passed to enrich, including every failed/blocked row.
    enriched = scorer.enrich(rows, {sid: actors[sid] for sid in ids},
                             {sid: targets[sid] for sid in ids}, pairing)
    conditions = [condition(event_rows(enriched, "initial", "initial"), scorer, "initial", "initial")]
    for event in ("all_update_events",) + events:
        for arm in ARMS:
            conditions.append(condition(event_rows(enriched, event, arm), scorer, event, arm))
    paired = [comparison(event_rows(enriched, event, "notes"), event_rows(enriched, event, "baseline"),
                         scorer, event=event, contrast="notes minus baseline")
              for event in ("all_update_events",) + events]
    report = {"part": part, "layout": layout, "sources": len(ids), "source_ids": ids,
              "events": list(events), "planned": len(rows), "recorded": len(rows),
              "all_actual_initial_pairs_verified": True,
              "initial_cost_shared_between_arms": True,
              "conditions": conditions, "paired_notes_minus_baseline": paired}
    return report, [{**row, "part": part, "layout": layout} for row in enriched]


def format_number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def metric_text(point: float | None, interval: list | None) -> str:
    return format_number(point) + (f" [{format_number(interval[0])}, {format_number(interval[1])}]" if interval else "")


def report_markdown(result: dict) -> str:
    lines = ["# CPM / minimal notes 本地评测", "",
             f"模型：`{result['metadata'].get('model', 'unknown')}`。原包：`{result['package_root']}`。", "",
             "M = source macro coverage-adjusted × 100；E = evidence_packets；J = strict_packets。"
             "未提交与 blocked 行保留在计划分母中，其 coverage-adjusted 内容得分为 0，"
             "不将接口失败解释为医学答案错误。括号区间为原 mean_ci 的 source bootstrap 95% CI。", "",
             "初始阶段单列：每个 source 的实际初始只计费一次，两臂更新成本均不重复包含初始。"
             "延迟覆盖所有阶段（含失败和 blocked）；token 数带 ≥ 表示有请求缺少该 usage 字段，已知部分仅为下界。", ""]
    for part in [result["core"], *result["layouts"]]:
        label = part["layout"] or "core / 4 events"
        lines += [f"## {label}", "", f"Sources：{part['sources']}；recorded/planned：{part['recorded']}/{part['planned']}。", "",
                  "| Event | Arm | M [95% CI] | E | J | Submitted/planned | Requests | Input | Output | Cache hit | Cache miss | Seconds median/p95 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for entry in part["conditions"]:
            tokens = [("≥" if entry["usage_rows_missing"][key] else "") + str(entry["usage"][key]) for key in TOKEN_KEYS]
            timing = entry["phase_seconds"]
            cells = [entry["event"], entry["arm"], metric_text(entry["M"], entry["M_ci95"]),
                     str(entry["E"]), str(entry["J"]), f"{entry['submitted']}/{entry['planned']}",
                     str(entry["requests"]), *tokens,
                     f"{format_number(timing['median'])}/{format_number(timing['p95'])}"]
            lines.append("| " + " | ".join(cells) + " |")
        lines += ["", "配对 ΔM = notes − baseline（百分点），按 source 配对；总体先在每个 source 内平均各事件：", "",
                  "| Event | ΔM [95% CI] | Sources |", "|---|---:|---:|"]
        for entry in part["paired_notes_minus_baseline"]:
            lines.append(f"| {entry['event']} | {metric_text(entry['delta_M'], entry['delta_M_ci95'])} | {entry['sources']} |")
        lines.append("")
    if result["paired_layouts"]:
        lines += ["## Layout 比较", "", "ΔM = distributed − centralized（百分点）。"
                  "同一 source 在两种 layout 下有各自实际初始；该比较不假设两个 layout 共用初始状态。", "",
                  "| Event | Arm | ΔM [95% CI] | Sources |", "|---|---|---:|---:|"]
        for entry in result["paired_layouts"]:
            lines.append(f"| {entry['event']} | {entry['arm']} | {metric_text(entry['delta_M'], entry['delta_M_ci95'])} | {entry['sources']} |")
        lines.append("")
    lines += ["只描述本次结果，不预设显著性或优劣结论。标签与逐项评分仅写入本地 private 文件。", ""]
    return "\n".join(lines)


def score_run(run_dir: Path) -> dict:
    run_dir = run_dir.expanduser().resolve()
    config = read(run_dir / "config.json")
    root = package_root(config, run_dir)
    if tuple(config.get("experimental_arms", ARMS)) != ARMS:
        raise ValueError("This scorer expects experimental_arms=['baseline', 'notes']")
    if tuple(config.get("events", CORE_EVENTS)) != CORE_EVENTS:
        raise ValueError("This scorer expects the original four core events")
    ids = source_ids(config.get("source_ids"), "source_ids")
    layout_ids = source_ids(config["layout_source_ids"], "layout_source_ids") if config.get("layout_source_ids") else []
    sys.path.insert(0, str(root))
    scorer = importlib.import_module("score_study15_local")
    workflow_scorer = importlib.import_module("score_workflow11_local")
    if Path(scorer.__file__).resolve().parent != root or Path(workflow_scorer.__file__).resolve().parent != root:
        raise ValueError("Scorer imports do not refer to config.json package_root")
    if scorer.score_row is not workflow_scorer.score_row:
        raise ValueError("Original score_row identity mismatch")
    dataset = Path(scorer.DATASET).resolve()
    if not dataset.is_relative_to(root):
        raise ValueError("Private evaluator dataset is outside the original package")
    actors_path = root / "confirmation15_inputs_private" / "actors.private.jsonl"
    targets_path = dataset / "evaluator_targets.LOCAL_ONLY.private.jsonl"
    selected = set(ids) | set(layout_ids)
    actors = selected_jsonl(actors_path, selected)
    targets = selected_jsonl(targets_path, selected)
    core, scored = evaluate_part(run_dir / "main", ids, CORE_EVENTS, actors, targets, scorer, "core")
    layout_reports, layout_rows, paired_layouts = [], {}, []
    if layout_ids:
        web = importlib.import_module("web15_runtime")
        web_scorer = importlib.import_module("score_web15_local")
        protocol_ids = set(read(root / "web15_protocol.json")["source_ids"])
        if not set(layout_ids) <= protocol_ids:
            raise ValueError("layout_source_ids must come from the original web15_protocol.json")
        for layout in web.LAYOUTS:
            aa = {sid: web.actor(actors[sid], layout) for sid in layout_ids}
            tt = {sid: web_scorer.target(targets[sid], actors[sid], layout) for sid in layout_ids}
            report, rr = evaluate_part(run_dir / "layouts" / layout, layout_ids, tuple(web.EVENTS),
                                       aa, tt, scorer, "layout", layout)
            layout_reports.append(report)
            layout_rows[layout] = rr
            scored.extend(rr)
        centralized, distributed = web.LAYOUTS
        for event in ("all_update_events",) + tuple(web.EVENTS):
            for arm in ARMS:
                paired_layouts.append(comparison(
                    event_rows(layout_rows[distributed], event, arm),
                    event_rows(layout_rows[centralized], event, arm), scorer,
                    event=event, arm=arm, contrast="distributed minus centralized",
                    different_actual_initials_per_layout=True))
    result = {
        "kind": "cpm_vs_minimal_notes_local_scoring", "package_root": str(root),
        "metadata": {key: config[key] for key in ("model", "endpoint", "settings", "experimental_arms", "events") if key in config},
        "core": core, "layouts": layout_reports, "paired_layouts": paired_layouts,
        "provenance": {
            "actors_path": str(actors_path), "actors_sha256": digest(actors_path),
            "private_targets_path": str(targets_path), "private_targets_sha256": digest(targets_path),
            "scoring_code_sha256": {name: digest(root / name) for name in
                                    ("score_study15_local.py", "score_workflow11_local.py", "score_web15_local.py")},
            "experiment_scorer_sha256": digest(Path(__file__)),
        },
        "scoring_unit": "source; events and claims within a source are correlated",
        "initial_cost_reported_separately": True,
        "static_and_robustness_evaluated": False,
        "private_targets_loaded_only_by_local_scorer": True,
        "clinical_validation": False,
    }
    output = run_dir / "evaluation"
    write(output / "scored_rows.LOCAL_ONLY.private.json", scored)
    write(output / "summary.json", result)
    (output / "report.md").write_text(report_markdown(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = score_run(args.run_dir)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        parser.exit(1, f"Scoring failed: {error}\n")
    print(f"Scored {result['core']['sources']} core sources; outputs: {args.run_dir.resolve() / 'evaluation'}")


if __name__ == "__main__":
    main()
