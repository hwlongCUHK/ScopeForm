"""Replay matched prefixes, then run error-feedback repair and the original Guard.

The fixed selection is development data, not a representative benchmark sample.
No private scoring targets are imported by inference. Cached physical attempts
and all new repair/continuation requests share the original 40-call phase cap.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import shutil
import time

import run_methods as run
from common import TOKEN_FIELDS, dump, fingerprint, read
from run_ssv_pilot import PilotContext, PilotRunner, copy_calls, phase_path

ARM = "frame_direct_guard_repair"
REFERENCE_ARM = "frame_direct_guard"


def prepare(out, source, selection, workers, prior_failed_run=None, replay_run=None, repair_prefix_run=None):
    previous = read(source / "config.json")
    cases = read(selection)["cases"]
    names = set(previous["code_sha256"]) | {
        "run_frame_repair_pilot.py", "evaluate_frame_repair_pilot.py", "methods/frame_repair.py",
        "definitions/error_taxonomy.json", "run_ssv_pilot.py", "recover_methods.py"}
    config = {**{k: previous[k] for k in ("package_root", "baseline_root", "model", "endpoint",
               "runtime_patch", "runtime_code_sha256")},
        "kind": "theory_classified_local_repair_pilot", "source_run": str(source), "reference_run": str(source),
        "experimental_arms": [ARM], "cases": cases,
        "prior_failed_run": str(prior_failed_run) if prior_failed_run else None,
        "replay_run": str(replay_run) if replay_run else None,
        "repair_prefix_run": str(repair_prefix_run) if repair_prefix_run else None,
        "settings": {**previous["settings"], "workers_per_model": workers},
        "comparison": "Same actual initial worlds, native read prefix and first graph; subsequent repair/Guard reads run live. Failure-enriched development pilot, not full-benchmark or held-out evidence.",
        "repair": "ID-addressed local edits; one initial diagnosis per graph; execution-error feedback to the same model until executable or the shared phase budget is exhausted; no training or fallback",
        "budget": "All copied and new physical requests together <=40; actions <=40; guard rounds do not reset either cap",
        "code_sha256": {name: run.sha(run.HERE / name) for name in sorted(names)}}
    if repair_prefix_run:
        config["comparison"] = ("Same actual initial worlds, native read prefix, first graph and first repair response; "
                                "only subsequent error-feedback and Guard continuation calls run live. "
                                "Failure-enriched development pilot, not full-benchmark or held-out evidence.")
    for name, expected in config["runtime_code_sha256"].items():
        if run.sha(run.baseline.RUNTIME / name) != expected:
            raise ValueError("Runtime differs: " + name)
    if (out / "config.json").exists() and read(out / "config.json") != config:
        raise ValueError("Run config differs; use a new directory")
    dump(out / "config.json", config)
    shutil.copy2(selection, out / "selection.json")
    for name in names:
        target = out / "frozen_code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(run.HERE / name, target)
    for case in cases:
        old = phase_path(source, REFERENCE_ARM, case)
        target = phase_path(out, ARM, case)
        checkpoint = read(old / "checkpoint.private.json")
        through = checkpoint["direct_attempts"][0]["request"]
        copy_calls(old, target, through)
        original = read(old / "guard_rounds/001/method/repair_original_graph.json") if (
            old / "guard_rounds/001/method/repair_original_graph.json").exists() else read(
            old / "guard_rounds/001/method/frame_generation_response.json")
        if "raw" in original:
            original = json.loads(original["raw"])
        manifest = {"source": str(old), "copied_through_logical_request": through,
            "copied_native_through_logical_request": through,
            "public_method_input_sha256": fingerprint(read(old / "guard_rounds/001/method/method_input.json")),
            "first_graph_sha256": fingerprint(original)}
        if repair_prefix_run:
            previous_phase = phase_path(repair_prefix_run, ARM, case)
            first = previous_phase / "guard_rounds/001"
            if fingerprint(read(first / "method/repair_original_graph.json")) != manifest["first_graph_sha256"]:
                raise ValueError("First repair must use the same original graph")
            for suffix in ("request.private.json", "response.private.json", "attempts.json"):
                name = f"call_{through + 1:03d}.{suffix}"
                dest = target / "guard_rounds/001" / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.copy2(first / name, dest)
            manifest.update(copied_through_logical_request=through + 1, repair_prefix_source=str(first))
        dump(target / "replay_manifest.json", manifest)
        if prior_failed_run:
            failed = phase_path(prior_failed_run, ARM, case)
            if read(failed / "checkpoint.private.json")["status"] != "indeterminate_transport_URLError":
                raise ValueError("Expected the prior network failure, not a content-result retry")
            entries = []
            for path in failed.rglob("call_*.attempts.json"):
                number = int(path.name.split("_")[1].split(".")[0])
                if number <= through:
                    continue
                response = path.with_name(path.name.replace("attempts.json", "response.private.json"))
                if response.exists():
                    raise ValueError("Do not discard successful model responses")
                entries.append({"source": str(path), "logical_request": number,
                                "attempts": read(path), "usage": {}})
            count = sum(len(e["attempts"]) for e in entries)
            dump(target / "prior_discarded_calls.json", {"requests": count, "entries": entries,
                "usage": {**dict.fromkeys(TOKEN_FIELDS, 0), "unreported_requests": dict.fromkeys(TOKEN_FIELDS, count)},
                "reason": "Prior sandbox network failures retained and charged; retry in network-enabled execution environment"})
        if replay_run:
            previous_phase = phase_path(replay_run, ARM, case)
            for path in previous_phase.rglob("call_*.json"):
                dest = target / path.relative_to(previous_phase)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.copy2(path, dest)
    return config


def run_case(case, source, out, runner, actors):
    sid = case["source_id"]
    actor = actors[sid]
    event = next(e for e in actor["events"] if e["event_id"] == case["event_id"])
    initial = read(source / "main/shared/phases" / sid / "initial__initial/checkpoint.private.json")
    base = run.h.restore(initial["world"])
    row, _ = runner.one(actor, event, ARM, out / "main", base=base)
    row.update(part="core", layout=None)
    dump(out / "main" / ARM / "rows" / sid / (case["event_id"] + "__fresh.json"), row)
    print(json.dumps({"source": sid, "event": case["event_id"], "status": row["status"],
                      "requests": row["requests"]}), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=run.HERE / "runs/frame_guard_core_20261114")
    parser.add_argument("--selection", type=Path, default=run.HERE / "frame_repair_selection.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prior-failed-run", type=Path)
    parser.add_argument("--replay-run", type=Path, help="Replay exact requests/responses; never call the API")
    parser.add_argument("--repair-prefix-run", type=Path,
                        help="Replay this run's first repair response, then call the API only for error feedback")
    args = parser.parse_args()
    source, out = args.source_run.resolve(), args.run_dir.resolve()
    config = prepare(out, source, args.selection, args.workers,
                     args.prior_failed_run.resolve() if args.prior_failed_run else None,
                     args.replay_run.resolve() if args.replay_run else None,
                     args.repair_prefix_run.resolve() if args.repair_prefix_run else None)
    if args.prepare_only:
        print(json.dumps({"cases": len(config["cases"]), "run": str(out), "api_calls": 0}))
        return
    if args.replay_run:
        class ReplayOnlyClient:
            models, fingerprints = set(), set()
            def chat(self, payload):
                raise RuntimeError("replay_missing_response_no_live_call")
        client = ReplayOnlyClient()
    else:
        key = os.environ.get("DEEPSEEK_API_KEY") or getpass.getpass("DeepSeek API key (hidden): ")
        client = run.baseline.Client(key, config["endpoint"])
        del key
    runner = PilotRunner(client, config["model"], config["settings"], (ARM,))
    run.Context, run.h.phase = PilotContext, runner.phase
    actors = {a["source_id"]: a for a in (json.loads(line) for line in
        (run.baseline.RUNTIME / "confirmation15_inputs_private/actors.private.jsonl").read_text().splitlines() if line)}
    started, rows = time.monotonic(), []
    dump(out / "run_status.json", {"status": "running", "planned": len(config["cases"]),
                                  "started_utc": datetime.now(timezone.utc).isoformat()})
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_case, case, source, out, runner, actors) for case in config["cases"]]
        for future in as_completed(futures):
            rows.append(future.result())
            dump(out / "results.json", rows)
    dump(out / "provider_metadata.json", {"requested_model": config["model"],
         "returned_models": sorted(client.models), "system_fingerprints": sorted(client.fingerprints)})
    dump(out / "run_status.json", {"status": "complete", "planned": len(config["cases"]), "recorded": len(rows),
         "wall_seconds": time.monotonic() - started, "finished_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
