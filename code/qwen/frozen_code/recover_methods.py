"""Resume explicit HTTP rejections in a copied run, retaining the 40-call cap.

Only transport replay changes: frozen inference, prompts and scoring stay intact.
The original run remains the immutable record of the first attempt.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import shutil
import threading
import time
import urllib.error
import urllib.request

import run_methods as run
from common import Context, dump, read

REJECTED = {"api_http_402", "api_http_429"}
LAYOUT = "distributed_versioned_services"


class ReplayContext(Context):
    def execute(self, code, *, stage, timeout=20):
        number = self.executions + 1
        folder = self.folder / "method"
        source = folder / ("solver_%03d_input.json" % number)
        result = folder / ("solver_%03d_output.json" % number)
        if result.exists():
            if read(source) != {"stage": stage, "code": code}:
                raise RuntimeError("replay_solver_input_mismatch")
            value = read(result)
            if value.pop("stage") != stage:
                raise RuntimeError("replay_solver_stage_mismatch")
            self.executions = number
            return value
        return super().execute(code, stage=stage, timeout=timeout)


class RecoveryClient(run.baseline.Client):
    def __init__(self, key, endpoint, folder):
        super().__init__(key, endpoint)
        self.stopped = threading.Event()
        self.folder = folder

    def request(self, path, payload=None):
        data = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
        request = urllib.request.Request(self.endpoint + path, data=data, headers={
            "Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        try:
            with urllib.request.build_opener(run.baseline.NoRedirect()).open(request, timeout=300) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            status = "api_http_" + str(error.code)
            if status in REJECTED:
                self.stopped.set()
            detail = error.read(4096).decode("utf-8", errors="replace").replace(self.key, "[REDACTED]")
            with self.lock:
                with (self.folder / "recovery_http_errors.private.jsonl").open("a") as handle:
                    handle.write(json.dumps({"epoch": time.time(), "status": status, "response_body": detail}) + "\n")
            raise RuntimeError(status) from None


class RecoveryRunner(run.Runner):
    def phase(self, world, folder, *args):
        previous = self.local.previous_seconds
        result = super().phase(world, folder, *args)
        world, status, requests, seconds, usage = result
        path = Path(folder) / "checkpoint.private.json"
        checkpoint = read(path)
        checkpoint["recovery_elapsed_seconds"] = checkpoint["seconds"]
        checkpoint["previous_attempt_seconds"] = previous
        checkpoint["seconds"] += previous
        dump(path, checkpoint)
        return world, status, requests, seconds + previous, usage

    def recover_one(self, actor, event, arm, out, base, tool_port):
        stem = event["event_id"] + "__fresh"
        row_path = out / arm / "rows" / actor["source_id"] / (stem + ".json")
        checkpoint_path = out / arm / "phases" / actor["source_id"] / stem / "checkpoint.private.json"
        old_row, checkpoint = read(row_path), read(checkpoint_path)
        if old_row["status"] not in REJECTED or checkpoint["status"] not in REJECTED:
            raise ValueError("Only explicit HTTP-rejected phases may be reopened")
        self.local.previous_seconds = old_row["seconds"]
        checkpoint["terminal"] = False
        dump(checkpoint_path, checkpoint)
        row_path.unlink()
        return self.one(actor, event, arm, out, base, LAYOUT, tool_port)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    source, destination = args.source_run.resolve(), args.run_dir.resolve()
    if source == destination:
        raise ValueError("Recovery requires a separate copied run")
    config = read(destination / "config.json")
    if config != read(source / "config.json"):
        raise ValueError("Copied configuration differs")
    for relative, expected in config["code_sha256"].items():
        if run.sha(run.HERE / relative) != expected:
            raise ValueError("Frozen inference code changed: " + relative)
    out = destination / "layouts" / LAYOUT
    rows = read(out / "results.json")
    indexed = {(r["source_id"], r["event_id"], r["arm"]): r for r in rows}
    selected = [r for r in rows if r["status"] in REJECTED]
    if any(r["event_id"] == "initial" for r in selected):
        raise ValueError("This recovery is scoped to rejected updates, not changed initial states")
    if not selected:
        raise ValueError("No known HTTP-rejected update remains")
    metadata_path = destination / "recovery_config.json"
    if not metadata_path.exists():
        dump(metadata_path, {"source_run": str(source), "eligible_phases": len(selected),
            "settings_unchanged": True, "workers": config["settings"]["workers_per_model"],
            "recovery_code_sha256": run.sha(Path(__file__)),
            "policy": "Only known HTTP402/429 retries; accepted responses and local solver outputs replayed exactly; every physical attempt counts toward40",
            "latency": "prior active phase seconds plus replay/resume active seconds; excludes recharge downtime",
            "local_method_artifact_timing": "SSR local traces may be recomputed during replay; prior artifacts remain in source run"})
        shutil.copy2(__file__, destination / "frozen_recovery.py")
        if (destination / "evaluation").exists():
            (destination / "evaluation").rename(destination / "evaluation_first_attempt")
    key = os.environ.get("DEEPSEEK_API_KEY") or getpass.getpass("DeepSeek API key (hidden): ")
    client = RecoveryClient(key, config["endpoint"], destination)
    del key
    runner = RecoveryRunner(client, config["model"], config["settings"], config["experimental_arms"], True)
    run.Context = ReplayContext
    run.h.phase = runner.phase
    actors = [json.loads(line) for line in (run.baseline.RUNTIME / "confirmation15_inputs_private/actors.private.jsonl").read_text().splitlines() if line]
    actors = {a["source_id"]: run.web.actor(a, LAYOUT) for a in actors if a["source_id"] in config["layout_source_ids"]}
    began = time.monotonic()
    dump(destination / "run_status.json", {"status": "recovery_running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(source), "eligible_phases": len(selected)})
    completed = 0
    with run.web.ToolService() as service:
        bases = {}
        for sid in sorted({r["source_id"] for r in selected}):
            _, base = runner.one(actors[sid], None, "baseline", out, layout=LAYOUT, tool_port=service.port)
            if not base.done:
                raise ValueError("Rejected update does not have a submitted shared initial")
            bases[sid] = base
        jobs = deque(sorted(selected, key=lambda r: run.baseline.fingerprint([r["source_id"], r["event_id"], r["arm"]])))
        with ThreadPoolExecutor(max_workers=config["settings"]["workers_per_model"]) as pool:
            futures = {}
            while jobs or futures:
                while jobs and len(futures) < config["settings"]["workers_per_model"] and not client.stopped.is_set():
                    old = jobs.popleft()
                    actor = actors[old["source_id"]]
                    event = next(e for e in actor["events"] if e["event_id"] == old["event_id"])
                    future = pool.submit(runner.recover_one, actor, event, old["arm"], out, bases[old["source_id"]], service.port)
                    futures[future] = old
                if not futures:
                    break
                done, _ = wait(futures, timeout=20, return_when=FIRST_COMPLETED)
                for future in done:
                    futures.pop(future)
                    row, _ = future.result()
                    indexed[(row["source_id"], row["event_id"], row["arm"])] = row
                    completed += 1
                current = list(indexed.values())
                dump(out / "results.json", current)
                progress = {"suite": LAYOUT, "planned": len(rows), "recorded": len(current),
                    "recovery_completed": completed, "recovery_selected": len(selected),
                    "recovery_pending": len(jobs), "submitted": sum(r["submitted"] for r in current),
                    "statuses": dict(Counter(r["status"] for r in current))}
                dump(out / "progress.json", progress)
                if done:
                    print(json.dumps(progress), flush=True)
    status = "complete" if completed == len(selected) and not client.stopped.is_set() else "recovery_stopped"
    dump(destination / "recovery_provider_metadata.json", {"returned_models": sorted(client.models), "system_fingerprints": sorted(client.fingerprints)})
    dump(destination / "run_status.json", {"status": status, "recovery_wall_seconds": time.monotonic() - began,
        "recovered_phases": completed, "selected_phases": len(selected), "finished_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(source)})
    print(json.dumps({"status": status, "completed": completed, "selected": len(selected)}), flush=True)


if __name__ == "__main__":
    main()
