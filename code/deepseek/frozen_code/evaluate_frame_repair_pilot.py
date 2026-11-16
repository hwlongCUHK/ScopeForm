"""Score actual pilot submissions offline and audit matched-prefix repair changes."""
import argparse
from collections import Counter
import importlib
from pathlib import Path
import sys

from common import TOKEN_FIELDS, dump, fingerprint, read
from evaluate_frame_direct import metrics
from evaluate_methods import load_helpers
from run_ssv_pilot import phase_path

ARM = "frame_direct_guard_repair"
OLD_ARM = "frame_direct_guard"


def evaluate(root):
    config, rows = read(root / "config.json"), read(root / "results.json")
    cases = config["cases"]
    keys = {(c["source_id"], c["event_id"]) for c in cases}
    if len(rows) != len(cases) or {(r["source_id"], r["event_id"]) for r in rows} != keys:
        raise ValueError("Incomplete repair pilot")
    source = Path(config["source_run"])
    helper = load_helpers(config)
    package = helper.package_root(config, root)
    sys.path.insert(0, str(package))
    scorer = importlib.import_module("score_study15_local")
    ids = {c["source_id"] for c in cases}
    actors = helper.selected_jsonl(package / "confirmation15_inputs_private/actors.private.jsonl", ids)
    targets = helper.selected_jsonl(Path(scorer.DATASET) / "evaluator_targets.LOCAL_ONLY.private.jsonl", ids)
    raw = read(source / "main/results.json")
    initial = [r for r in raw if r["event_id"] == "initial" and r["source_id"] in ids]
    controls = [r for r in raw if r["arm"] == OLD_ARM and (r["source_id"], r["event_id"]) in keys]
    pairing = read(source / "main/pairing_audit.json")
    all_scored = scorer.enrich(initial + controls + rows, actors, targets, {sid: pairing[sid] for sid in ids})
    groups = {arm: [r for r in all_scored if r["arm"] == arm] for arm in (OLD_ARM, ARM)}
    counts, live, cases_out = Counter(), Counter(), []
    checks = []
    for case in cases:
        pair = {arm: next(r for r in rr if r["source_id"] == case["source_id"] and r["event_id"] == case["event_id"])
                for arm, rr in groups.items()}
        old, new = pair[OLD_ARM], pair[ARM]
        folder = phase_path(root, ARM, case)
        cp = read(folder / "checkpoint.private.json")
        manifest = read(folder / "replay_manifest.json")
        first = folder / "guard_rounds/001/method"
        matched = (fingerprint(read(first / "method_input.json")) == manifest["public_method_input_sha256"]
                   and fingerprint(read(first / "repair_original_graph.json")) == manifest["first_graph_sha256"])
        prior_path = folder / "prior_discarded_calls.json"
        prior_count = read(prior_path)["requests"] if prior_path.exists() else 0
        request_count, phase_live = prior_count, 0
        live["prior_failed_requests"] += prior_count
        for path in folder.rglob("call_*.attempts.json"):
            attempts = read(path)
            request_count += len(attempts)
            number = int(path.name.split("_")[1].split(".")[0])
            if number > manifest["copied_through_logical_request"] and not config.get("replay_run"):
                phase_live += len(attempts)
                response = path.with_name(path.name.replace("attempts.json", "response.private.json"))
                usage = read(response).get("usage", {}) if response.exists() else {}
                for field, remote in TOKEN_FIELDS.items():
                    live[field] += usage.get(remote, 0)
        live["physical_requests"] += phase_live
        issues = []
        repair_rounds = 0
        for method_folder in sorted(folder.glob("guard_rounds/*/method")):
            rounds = sorted(method_folder.glob("repair_round_[0-9][0-9][0-9].json"))
            if rounds:
                repair_rounds += len(rounds)
                for path in rounds:
                    diagnosis = read(path).get("diagnosis")
                    if isinstance(diagnosis, dict):
                        issues += diagnosis.get("issues", [])
            else:
                issues += read(method_folder / "repair_diagnosis.json")["issues"]
                repair_rounds += 1
        counts.update(i["category"] for i in issues)
        valid = (matched and request_count == new["requests"] and request_count <= 40
                 and cp["action_count"] <= 40
                 and all(old.get(k) == new.get(k) for k in ("initial_checkpoint_hash", "pre_state_hash", "public_hash")))
        if new["submitted"]:
            valid &= {i["item_id"]: i["label"] for i in cp["method_result"]["items"]} == {
                i["item_id"]: i["label"] for i in new["prediction"]["items"]}
        checks.append(bool(valid))
        cases_out.append({**case, "matched_first_graph_and_input": matched, "new_live_requests": phase_live,
            "repair_rounds": repair_rounds,
            "diagnosed_categories": [i["category"] for i in issues],
            **{arm: {"status": r["status"], "submitted": r["submitted"], "claims": r["items_planned"],
                     "correct": sum(i["content"] for i in (r.get("score") or {}).get("items", [])) if r["submitted"] else 0,
                     "exact": bool((r.get("score") or {}).get("content")),
                     "E": bool((r.get("score") or {}).get("evidence")),
                     "J": bool((r.get("score") or {}).get("strict")), "requests": r["requests"]}
               for arm, r in pair.items()}})
    comparison = config["comparison"]
    replay_audit = None
    if config.get("replay_run"):
        previous_run = Path(config["replay_run"])
        previous_calls = {str(p.relative_to(previous_run)): fingerprint(read(p))
                          for p in previous_run.rglob("call_*.json")}
        current_calls = {str(p.relative_to(root)): fingerprint(read(p))
                         for p in root.rglob("call_*.json")}
        replay_audit = {"source_run": str(previous_run), "call_artifacts": len(current_calls),
                        "identical_all_call_artifacts": previous_calls == current_calls}
        checks.append(replay_audit["identical_all_call_artifacts"])
        comparison = ("Same actual initial worlds, native read prefix and first graph; all model responses "
                      "replayed unchanged from the recorded repair run. Only the local edit application changed. "
                      "Failure-enriched development pilot, not full-benchmark or held-out evidence.")
    summary = {"metrics": {arm: metrics(rr) for arm, rr in groups.items()}, "cases": cases_out,
               "diagnosed_categories_not_gold": dict(counts), "live_usage": dict(live),
               "verification_passed": all(checks), "comparison": comparison,
               "response_replay_audit": replay_audit}
    dest = root / "evaluation"
    dump(dest / "scored_rows.LOCAL_ONLY.private.json", all_scored)
    dump(dest / "summary.json", summary)
    execution = ("所有模型回包沿用已完成的在线纠错运行，未重新调用模型；本次仅调整局部修改接口，允许edits=[]的条目不产生修改。"
                 if config.get("replay_run") else "此后纠错及Guard补读在线运行。")
    if config.get("repair_prefix_run"):
        execution = "首轮诊断与修改也复用上一版回包；只有修复后仍失败时的反馈循环及Guard继续读取会新增API调用。"
    lines = ["# 理论分类纠错：固定8例开发验证", "", comparison, "",
             "两组共享历史实际初始状态、首次读取与形式化图。" + execution,
             "这不是代表性抽样或全量结果；类别是模型诊断，未经人工金标认证。", "",
             "| 方法 | 提交 | 正确命题 | 整包全对 | E | J | 计入预算请求 | 含历史回包耗时的中位秒 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, m in summary["metrics"].items():
        lines.append(f"| {arm} | {m['submitted']}/{m['packets']} | {m['correct']}/{m['claims']} | {m['all_correct']} | {m['E']} | {m['J']} | {m['requests']} | {m['median_seconds']:.2f} |")
    lines += ["", "| 来源／事件 | 选择原因 | 旧正确 | 新正确 | 新状态 | 模型诊断类别 |",
              "|---|---|---:|---:|---|---|"]
    for case in cases_out:
        a, b = case[OLD_ARM], case[ARM]
        lines.append(f"| {case['source_id']} / {case['event_id']} | {case['selection_group']} | {a['correct']}/{a['claims']} | {b['correct']}/{b['claims']} | {b['status']} | {', '.join(case['diagnosed_categories']) or '无修改'} |")
    lines += ["", f"本次新增实际API请求：{live['physical_requests']}；重放请求同样计入每阶段40次上限。",
              f"预算还包含先前网络失败请求{live['prior_failed_requests']}次，其未报告token量不当作零消费。",
              "时间包含历史回包记录的模型耗时与本次本地执行耗时，不是一次从头在线运行的延迟测量。",
              "首轮诊断后，仅形式化执行失败触发继续修复；成功即停止，全部调用共用每阶段40次额度。"
              "不依据评分挑选修改，评分目标仅由本离线脚本读取。",
              f"共享输入、预算记账、初始世界及求解标签提交核验：{all(checks)}。"]
    (dest / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    import json
    result = evaluate(args.run_dir.resolve())
    print(json.dumps({k: result[k] for k in ("metrics", "live_usage", "verification_passed")}, ensure_ascii=False, indent=2))
