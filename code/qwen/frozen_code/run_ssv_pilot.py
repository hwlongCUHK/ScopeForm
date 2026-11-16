"""Paired small SSV ablation from previously acquired public evidence.

Replay the original native read prefix and share one SSV-lite program across
three arms. Then actually run each arm and the original save/readback/submit
workflow. Replayed calls remain charged against each phase's 40-call cap.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import getpass
import importlib
import json
import os
from pathlib import Path
import shutil
import threading
import time

import run_methods as run
from common import dump, read, fingerprint, TOKEN_FIELDS
from recover_methods import ReplayContext

ARMS = ("ssv_lite", "ssv_align", "ssv_roundtrip")
CHANGED_STAGES = {"ssv_align_scoped_repair", "roundtrip_source_repair"}


def suite_path(case):
    return Path("main") if case["part"] == "core" else Path("layouts") / case["layout"]


def phase_path(out, arm, case):
    return out / suite_path(case) / arm / "phases" / case["source_id"] / (case["event_id"] + "__fresh")


def copy_calls(source, target, last):
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for index in range(1, last + 1):
        for suffix in ("request.private.json", "response.private.json", "attempts.json"):
            name = "call_%03d.%s" % (index, suffix)
            dest = target / name
            if not dest.exists():
                shutil.copy2(source / name, dest)
            copied.append(name)
    return copied


def reuse_previous(source, target, shared_ledger=None):
    """Retain valid pre-fix completions; charge discarded/interrupted calls too."""
    last = 0
    if source.exists():
        for request in sorted(source.glob("call_*.request.private.json")):
            index = int(request.name.split("_")[1].split(".")[0])
            attempts = read(source / ("call_%03d.attempts.json" % index))
            if attempts[0]["stage"] in CHANGED_STAGES or not (source / ("call_%03d.response.private.json" % index)).exists():
                break
            last = index
        copy_calls(source, target, last)
        for inp in sorted((source / "method").glob("solver_*_input.json")):
            value = read(inp)
            if any(stage in value["stage"] for stage in CHANGED_STAGES):
                break
            output = inp.with_name(inp.name.replace("_input", "_output"))
            if not output.exists():
                break
            (target / "method").mkdir(exist_ok=True)
            for item in (inp, output):
                if not (target / "method" / item.name).exists():
                    shutil.copy2(item, target / "method" / item.name)
    entries = []
    usage = {k: 0 for k in TOKEN_FIELDS}
    usage["unreported_requests"] = {k: 0 for k in TOKEN_FIELDS}
    for attempts_path in sorted(source.glob("call_*.attempts.json")):
        index = int(attempts_path.name.split("_")[1].split(".")[0])
        if index <= last:
            continue
        attempts = read(attempts_path)
        response = source / ("call_%03d.response.private.json" % index)
        reported = (read(response).get("usage") or {}) if response.exists() else {}
        entries.append({"source": str(attempts_path), "logical_request": index,
                        "attempts": attempts, "usage": reported})
        for key, remote in TOKEN_FIELDS.items():
            usage[key] += reported.get(remote, 0)
            usage["unreported_requests"][key] += len(attempts) - int(remote in reported)
    ledger = {"requests": sum(len(e["attempts"]) for e in entries), "usage": usage,
              "entries": entries, "reason": "repair-format implementation correction; interrupted calls remain charged"}
    # An interrupted shared generation is also charged to arms using its retry.
    if shared_ledger:
        shared = read(shared_ledger)
        for entry in shared["entries"]:
            if all(a["stage"].startswith("ssv_lite_") for a in entry["attempts"]):
                ledger["entries"].append(entry)
                ledger["requests"] += len(entry["attempts"])
                for key, remote in TOKEN_FIELDS.items():
                    ledger["usage"][key] += entry["usage"].get(remote, 0)
                    ledger["usage"]["unreported_requests"][key] += len(entry["attempts"]) - int(remote in entry["usage"])
    dump(target / "prior_discarded_calls.json", ledger)
    return last


class PilotContext(ReplayContext):
    current = threading.local()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replayed_seconds = 0.0
        self.replayed_logical_calls = 0
        self.live_calls = 0
        ledger_path = self.folder / "prior_discarded_calls.json"
        if ledger_path.exists():
            ledger = read(ledger_path)
            self.requests += ledger["requests"]
            for key in TOKEN_FIELDS:
                self.usage[key] += ledger["usage"][key]
                self.usage["unreported_requests"][key] += ledger["usage"]["unreported_requests"][key]
        PilotContext.current.value = self

    def chat(self, payload, *, stage):
        number = self.logical_index + 1
        cached = (self.folder / ("call_%03d.response.private.json" % number)).exists()
        raw = super().chat(payload, stage=stage)
        if cached:
            self.replayed_logical_calls += 1
            attempts = read(self.folder / ("call_%03d.attempts.json" % number))
            self.replayed_seconds += sum(max(0, a.get("completed_epoch", a["dispatch_epoch"]) - a["dispatch_epoch"])
                                         for a in attempts)
        else:
            self.live_calls += 1
        return raw

    def execute(self, code, *, stage, timeout=20):
        cached = (self.folder / "method" / ("solver_%03d_output.json" % (self.executions + 1))).exists()
        value = super().execute(code, stage=stage, timeout=timeout)
        if cached:
            self.replayed_seconds += value.get("seconds", 0)
        return value


class TimedMethod:
    def __init__(self, name):
        self.method = importlib.import_module("methods." + name)

    def run(self, ctx):
        expected = read(ctx.folder / "replay_manifest.json")["public_method_input_sha256"]
        if fingerprint(ctx.problem) != expected:
            raise RuntimeError("replay_public_method_input_mismatch")
        start, replayed, calls = time.monotonic(), ctx.replayed_seconds, ctx.requests
        try:
            return self.method.run(ctx)
        finally:
            wall = time.monotonic() - start
            ctx.record("pilot_method_timing", {"live_wall_seconds": wall,
                       "replayed_seconds": ctx.replayed_seconds - replayed,
                       "replay_adjusted_seconds": wall + ctx.replayed_seconds - replayed,
                       "charged_requests": ctx.requests - calls})


class PilotRunner(run.Runner):
    def phase(self, world, folder, *args):
        path = Path(folder) / "checkpoint.private.json"
        already_done = path.exists() and read(path).get("terminal")
        result = super().phase(world, folder, *args)
        if already_done:
            return result
        ctx = PilotContext.current.value
        world, status, requests, wall, usage = result
        elapsed = wall + ctx.replayed_seconds
        cp = read(path)
        cp.update(seconds=elapsed, live_wall_seconds=wall,
                  replayed_seconds=ctx.replayed_seconds,
                  replayed_logical_calls=ctx.replayed_logical_calls,
                  new_live_calls=ctx.live_calls)
        dump(path, cp)
        return world, status, requests, elapsed, usage


def native_prefix(source):
    stages = read(source / "request_accounting.json")["stages"]
    native = []
    for stage in stages:
        if stage["stage"] != "native":
            break
        native.append(stage["logical_request"])
    if not native or not (source / "method/method_input.json").exists():
        raise ValueError("Selected original phase did not reach a method boundary")
    return native[-1]


def run_case(case, source_run, out, runner, actors, port=0, reuse_run=None):
    sid = case["source_id"]
    actor = actors[sid]
    layout = case.get("layout")
    if layout:
        actor = run.web.actor(actor, layout)
    event = next(e for e in actor["events"] if e["event_id"] == case["event_id"])
    suite = suite_path(case)
    initial_cp = read(source_run / suite / "shared/phases" / sid / "initial__initial/checkpoint.private.json")
    base = run.web.restore_proxy(initial_cp["world"]) if layout else run.h.restore(initial_cp["world"])
    if not base.done:
        raise ValueError("Pilot requires an actually submitted shared initial")
    source = phase_path(source_run, "ssv", case)
    last = native_prefix(source)
    input_hash = fingerprint(read(source / "method/method_input.json"))
    rows = []
    for arm in ARMS:
        target = phase_path(out, arm, case)
        if arm == "ssv_lite":
            copy_calls(source, target, last)
            parent = str(source)
            through = last
        else:
            parent_path = phase_path(out, "ssv_lite", case)
            prepared = read(parent_path / "method/ssv_lite_prepared.json")
            through = prepared["logical_requests"]
            copy_calls(parent_path, target, through)
            (target / "method").mkdir(exist_ok=True)
            for index in range(1, prepared["executions"] + 1):
                for suffix in ("input", "output"):
                    name = "solver_%03d_%s.json" % (index, suffix)
                    dest = target / "method" / name
                    if not dest.exists():
                        shutil.copy2(parent_path / "method" / name, dest)
            parent = str(parent_path)
        if reuse_run:
            shared_ledger = (phase_path(out, "ssv_lite", case) / "prior_discarded_calls.json") if arm != "ssv_lite" else None
            through = max(through, reuse_previous(phase_path(reuse_run, arm, case), target, shared_ledger))
        dump(target / "replay_manifest.json", {"replay_source": parent,
             "copied_through_logical_request": through,
             "original_native_prefix_last_logical_request": last,
             "public_method_input_sha256": input_hash,
             "shared_initial_checkpoint": str(source_run / suite / "shared/phases" / sid / "initial__initial/checkpoint.private.json")})
        row, _ = runner.one(actor, event, arm, out / suite, base=base, layout=layout, tool_port=port)
        row.update(part=case["part"], layout=layout, selection_group=case.get("group", "selected"))
        dump(out / suite / arm / "rows" / sid / (case["event_id"] + "__fresh.json"), row)
        rows.append(row)
        print(json.dumps({"case": sid, "event": case["event_id"], "layout": layout,
                          "arm": arm, "status": row["status"], "requests": row["requests"]}), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=run.HERE / "runs/full_methods_recovered_20261113")
    parser.add_argument("--selection", type=Path, default=run.HERE / "pilot_selection.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reuse-run", type=Path)
    args = parser.parse_args()
    source, out = args.source_run.resolve(), args.run_dir.resolve()
    selection = read(args.selection)
    cases = selection["cases"] if isinstance(selection, dict) else selection
    # Only routing fields enter the inference harness; offline selection details
    # and old labels/gold never enter method inputs.
    cases = [{k: c[k] for k in ("part", "layout", "source_id", "event_id", "group") if k in c} for c in cases]
    out.mkdir(parents=True, exist_ok=True)
    old_config = read(source / "config.json")
    code = [Path(__file__), run.HERE / "run_methods.py", run.HERE / "common.py",
            run.HERE / "recover_methods.py", run.HERE / "solver_worker.py", run.HERE / "methods/ssv.py"]
    code += [run.HERE / "methods" / (name + ".py") for name in ARMS]
    config = {"kind": "failure_enriched_small_pilot", "source_run": str(source),
              "package_root": old_config["package_root"], "baseline_root": old_config["baseline_root"],
              "model": old_config["model"], "endpoint": old_config["endpoint"],
              "settings": {**old_config["settings"], "workers_per_model": args.workers},
              "experimental_arms": list(ARMS), "cases": cases,
              "reuse_run": str(args.reuse_run.resolve()) if args.reuse_run else None,
              "paired_design": "same actual initial, original native read prefix, and first lite program; live method and native continuation",
              "budget": "40 physical requests per phase, including all replayed prefix and shared generation attempts",
              "latency": "live wall time plus recorded replayed provider/solver durations; excludes original tool execution overhead",
              "code_sha256": {str(p.relative_to(run.HERE)): run.sha(p) for p in code}}
    if (out / "config.json").exists() and read(out / "config.json") != config:
        raise ValueError("Pilot config differs; use a new output directory")
    dump(out / "config.json", config)
    for path in code:
        dest = out / "frozen_code" / path.relative_to(run.HERE)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(path, dest)
    key = os.environ.get("DEEPSEEK_API_KEY") or getpass.getpass("DeepSeek API key (hidden): ")
    client = run.baseline.Client(key, config["endpoint"])
    del key
    run.Context = PilotContext
    run.METHODS.update({name: TimedMethod(name) for name in ARMS})
    runner = PilotRunner(client, config["model"], config["settings"], ARMS)
    run.h.phase = runner.phase
    actors = {a["source_id"]: a for a in [json.loads(line) for line in
              (run.baseline.RUNTIME / "confirmation15_inputs_private/actors.private.jsonl").read_text().splitlines() if line]}
    started = time.monotonic()
    dump(out / "run_status.json", {"status": "running", "planned": len(cases) * len(ARMS),
                                   "started_utc": datetime.now(timezone.utc).isoformat()})
    rows = []
    with run.web.ToolService() as service:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_case, case, source, out, runner, actors, service.port,
                                   args.reuse_run.resolve() if args.reuse_run else None) for case in cases]
            for future in as_completed(futures):
                rows.extend(future.result())
                dump(out / "results.json", rows)
    dump(out / "provider_metadata.json", {"requested_model": config["model"],
         "returned_models": sorted(client.models), "system_fingerprints": sorted(client.fingerprints)})
    dump(out / "run_status.json", {"status": "complete", "recorded": len(rows),
         "planned": len(cases) * len(ARMS), "wall_seconds": time.monotonic() - started,
         "finished_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"complete": str(out), "phases": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
