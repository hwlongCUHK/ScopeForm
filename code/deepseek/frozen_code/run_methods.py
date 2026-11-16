"""Original CPM baseline plus three complete, independent reasoning methods.

No notes. Every native and method-internal LLM request shares a 40-call cap.
Only acquired public observations are passed to reasoning implementations.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
from datetime import datetime, timezone
import getpass
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASELINE_ROOT = Path(os.environ.get("CPM_BASELINE_ROOT",
    "/Users/darrenlong/Desktop/长江/Drug Agent Project/sandbox/drug agent"))
sys.path.insert(0, str(BASELINE_ROOT))
import run_eval as baseline
sys.path.insert(0, str(HERE))
from common import Context, BudgetExceeded, ModelOutputLimit, dump, read

h, web, rw = baseline.h, baseline.web, baseline.rw
ARMS = ("baseline", "nl2logic", "ssr", "ssv")
METHODS = {name: importlib.import_module("methods." + name) for name in ARMS[1:]}
SETTINGS = {**baseline.SETTINGS, "max_requests": 40}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Runner(baseline.Runner):
    def __init__(self, client, model, settings, arms=ARMS, retry_transport=False):
        super().__init__(client, model, settings)
        self.arms = tuple(arms)
        self.retry_transport = retry_transport

    def phase(self, world, folder, port, model, tokenizer, settings):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        arm = getattr(self.local, "arm", "baseline")
        repairing = arm in ("frame_direct_repair", "frame_direct_guard_repair")
        guarded = arm in ("frame_direct_guard", "frame_direct_guard_repair")
        direct = repairing or arm in ("frame_direct", "frame_direct_guard")
        if direct:
            from methods import frame_direct
        if repairing:
            from methods import frame_repair
        if guarded:
            # Enable the paper's World.call guard, keeping Fresh's support=None.
            world.arm = "evidence_constraint"
        start_hash, public_hash = baseline.fingerprint(world.state), baseline.fingerprint(world.public)
        checkpoint = folder / "checkpoint.private.json"
        if checkpoint.exists():
            old = read(checkpoint)
            if old["start_hash"] != start_hash or old["public_hash"] != public_hash:
                raise ValueError("Actual initial/checkpoint differs")
            if old["terminal"]:
                restored = web.restore_proxy(old["world"]) if isinstance(world, web.WebWorld) else h.restore(old["world"])
                if isinstance(world, web.WebWorld):
                    restored.web_http_port = world.web_http_port
                return restored, old["status"], old["requests"], old["seconds"], old["usage"]
        started = time.monotonic()
        ctx = Context(self.client, model, folder, settings, retry_transport=self.retry_transport)
        instruction = frame_direct.instruction(h.POLICY, guarded=guarded) if direct else h.POLICY + baseline.NATIVE_INSTRUCTION
        messages = [{"role": "user", "content": instruction +
                     json.dumps(world.initial_input(), ensure_ascii=False)}]
        observations = []
        method_done = False
        method_result = None
        method_started_at = None
        action_count = 0
        status = "action_limit"
        direct_finished = False
        direct_attempts = []
        def actual_call(name, args):
            nonlocal action_count
            if action_count >= settings["max_actions"]:
                raise RuntimeError("direct_action_limit")
            action_count += 1
            value = world.call(name, args)
            observations.append({"tool": name, "arguments": copy.deepcopy(args), "result": copy.deepcopy(value)})
            return value
        def save(terminal=False):
            dump(checkpoint, {"start_hash": start_hash, "public_hash": public_hash,
                "world": h.prior.snapshot(world), "messages": messages,
                "requests": ctx.requests, "seconds": time.monotonic() - started,
                "action_count": action_count, "status": status, "terminal": terminal,
                "usage": ctx.usage, "harness": "cpm_full_method_comparison",
                "experimental_arm": arm, "notes_enabled": False,
                "method_started_at_request": method_started_at,
                "method_completed": method_done, "method_result": method_result,
                "guard_enabled": guarded, "direct_attempts": direct_attempts})
        save()
        try:
            while action_count < settings["max_actions"] and ctx.requests < settings["max_requests"]:
                request = {"model": model, "messages": copy.deepcopy(messages),
                    "tools": frame_direct.tools(world, guarded=guarded) if direct else world.tools(), "tool_choice": "auto", "parallel_tool_calls": False,
                    "temperature": 0, "thinking": {"type": "disabled"},
                    "max_tokens": settings["max_output_tokens"], "stream": False}
                raw = ctx.chat(request, stage="native")
                choice = raw["choices"][0]
                calls = choice["message"].get("tool_calls", [])
                if not calls:
                    status = "invalid_native_no_tool_call"
                    break
                parsed = []
                for call in calls:
                    if call.get("type") != "function" or not isinstance(call.get("id"), str):
                        raise ValueError("invalid_native_call_shape")
                    args = json.loads(call["function"]["arguments"])
                    if not isinstance(args, dict):
                        raise ValueError("invalid_native_call_arguments")
                    parsed.append((call, call["function"]["name"], args))
                messages.append({"role": "assistant", "content": choice["message"].get("content") or "",
                                 "tool_calls": copy.deepcopy(calls)})
                deferred = False
                for call, name, args in parsed:
                    if world.done:
                        result = {"ok": False, "error": "not_executed_after_terminal_submission"}
                    elif action_count >= settings["max_actions"]:
                        result = {"ok": False, "error": "not_executed_action_cap"}
                    elif direct and direct_finished:
                        result = {"ok": False, "error": "direct_formalization_already_finished"}
                    elif direct and name == frame_direct.TOOL_NAME:
                        action_count += 1
                        direct_finished = True
                        method_started_at = ctx.requests
                        attempt = {"request": ctx.requests, "action_count_before": action_count - 1}
                        if guarded:
                            direct_attempts.append(attempt)
                            ctx.folder = folder / "guard_rounds" / f"{len(direct_attempts):03d}"
                            ctx.folder.mkdir(parents=True, exist_ok=True)
                        ctx.problem = {"task_input": world.initial_input(),
                                       "observations": copy.deepcopy(observations)}
                        ctx.record("method_input", ctx.problem)
                        try:
                            submitted_graph = args["graph"]
                            if repairing:
                                submitted_graph, method_result = frame_repair.run(ctx, submitted_graph)
                            else:
                                method_result = frame_direct.solve(ctx, submitted_graph)
                        except (BudgetExceeded, ModelOutputLimit):
                            raise
                        except Exception as error:
                            if repairing and str(error).startswith(("api_http_", "indeterminate_", "replay_")):
                                raise
                            method_result = {"method": arm, "status": "implementation_error", "items": [],
                                             "error_type": type(error).__name__, "error": str(error)}
                        method_done = True
                        ctx.record("method_result", method_result)
                        try:
                            if method_result["status"] == "completed":
                                result = frame_direct.submit(ctx, submitted_graph, method_result, actual_call)
                            else:
                                status = "direct_formalization_failed"
                                result = {"ok": False, "method_result": method_result}
                        except frame_direct.GuardRejected as rejection:
                            if not guarded:
                                raise
                            result = rejection.response
                            attempt["guard_feedback"] = copy.deepcopy(result)
                            direct_finished = False
                        finally:
                            if guarded:
                                attempt.update(action_count_after=action_count, submitted=world.done,
                                               method_status=method_result["status"])
                                ctx.record("guard_round", attempt)
                                shutil.copytree(ctx.folder / "method", folder / "method", dirs_exist_ok=True)
                                ctx.folder = folder
                    elif direct and name not in frame_direct.READ_TOOLS:
                        raise ValueError("direct_tool_not_available: " + name)
                    elif deferred:
                        action_count += 1
                        result = {"ok": False, "error": "deferred_until_method_result_processed"}
                    elif arm != "baseline" and not method_done and name in ("save_summary", "submit_answer"):
                        action_count += 1
                        method_started_at = ctx.requests
                        ctx.problem = {"task_input": world.initial_input(),
                            "observations": copy.deepcopy(observations),
                            "proposed_action": {"name": name, "arguments": copy.deepcopy(args)}}
                        ctx.record("method_input", ctx.problem)
                        save()
                        try:
                            method_result = METHODS[arm].run(ctx)
                        except (BudgetExceeded, ModelOutputLimit):
                            raise
                        except RuntimeError as error:
                            if str(error).startswith(("api_http_", "indeterminate_", "replay_")):
                                raise
                            method_result = {"method": arm, "status": "implementation_error", "items": [],
                                             "error_type": type(error).__name__, "error": str(error)}
                        except Exception as error:
                            method_result = {"method": arm, "status": "implementation_error", "items": [],
                                             "error_type": type(error).__name__, "error": str(error)}
                        ctx.record("method_result", method_result)
                        method_done = True
                        deferred = True
                        result = {"ok": False, "requested_action_executed": False,
                            "reason": "reasoning_method_result_ready_before_first_write_or_submit",
                            "reasoning_method": arm, "method_result": method_result,
                            "instruction": "本次保存或提交尚未执行。以上是本实验指定推理方法的输出，不是标准答案。"
                            "使用其逐项判断继续完成原任务，引用必须来自你实际取得的当前原文。"
                            "若方法未产生某项判断，按原任务继续核对。需要保存时实际保存并读回，再实际提交。"}
                    else:
                        action_count += 1
                        result = world.call(name, args)
                        observations.append({"tool": name, "arguments": copy.deepcopy(args),
                                             "result": copy.deepcopy(result)})
                    messages.append({"role": "tool", "tool_call_id": call["id"],
                                     "content": json.dumps(result, ensure_ascii=False)})
                if world.done:
                    status = "submitted"
                    break
                if direct_finished:
                    break
                save()
            if not world.done and ctx.requests >= settings["max_requests"]:
                status = "total_request_limit"
        except BudgetExceeded:
            status = "total_request_limit"
            ctx.record("budget_exhausted", {"requests": ctx.requests,
                "method_started_at_request": method_started_at, "method_completed": method_done})
        except ModelOutputLimit:
            status = "output_limit_not_content_error"
        except Exception as error:
            message = str(error)
            status = message if message.startswith(("api_http_", "invalid_native_", "indeterminate_", "replay_")) else "runner_or_transport_" + type(error).__name__
            dump(folder / "error.json", {"status": status, "type": type(error).__name__, "message": message})
        save(True)
        return world, status, ctx.requests, time.monotonic() - started, ctx.usage

    def one(self, actor, event, arm, out, base=None, layout=None, tool_port=0):
        self.local.arm = arm
        original_method = "initial" if event is None else "fresh"
        folder = Path(out) / ("shared" if event is None else arm)
        if layout is None:
            row, world = h.one(actor, event, original_method, folder, 0, self.model,
                               None, self.settings, base=base)
        else:
            row, world = rw.one(actor, event, original_method, layout, folder, 0, tool_port,
                                self.model, None, self.settings, base=base)
        row = {**row, "arm": "initial" if event is None else arm,
               "harness": "cpm_full_method_comparison", "environment_arm": original_method}
        cp = folder / "phases" / actor["source_id"] / ((event["event_id"] if event else "initial") + "__" + original_method) / "checkpoint.private.json"
        if cp.exists():
            state = read(cp)
            row["method_started_at_request"] = state.get("method_started_at_request")
            row["method_completed"] = state.get("method_completed", False)
            row["method_status"] = (state.get("method_result") or {}).get("status")
        return row, world

    def workflow(self, actors, out, layout=None, tool_port=0):
        out = Path(out)
        rows, pairing = [], {}
        planned = sum(1 + len(a["events"]) * len(self.arms) for a in actors)
        with ThreadPoolExecutor(max_workers=self.settings["workers_per_model"]) as pool:
            futures = {pool.submit(self.one, actor, None, "baseline", out,
                layout=layout, tool_port=tool_port): (actor, True)
                for actor in sorted(actors, key=lambda x: baseline.fingerprint(x["source_id"]))}
            while futures:
                done, _ = wait(futures, timeout=20, return_when=FIRST_COMPLETED)
                for future in done:
                    actor, initial = futures.pop(future)
                    row, world = future.result()
                    rows.append(row)
                    if initial:
                        pairing[actor["source_id"]] = {
                            "initial_world_sha256": web.checkpoint_hash(world) if layout else baseline.fingerprint(h.prior.snapshot(world)),
                            "actual_submitted": world.done}
                        jobs = [(event, arm) for event in actor["events"] for arm in self.arms]
                        for event, arm in sorted(jobs, key=lambda x: baseline.fingerprint([actor["source_id"], x[0]["event_id"], x[1]])):
                            futures[pool.submit(self.one, actor, event, arm, out, world, layout, tool_port)] = (actor, False)
                    dump(out / "results.json", rows)
                    dump(out / "pairing_audit.json", pairing)
                progress = {"suite": layout or "core", "planned": planned, "recorded": len(rows),
                    "submitted": sum(r["submitted"] for r in rows),
                    "statuses": dict(Counter(r["status"] for r in rows)),
                    "by_arm": {arm: dict(Counter(r["status"] for r in rows if r["arm"] == arm)) for arm in self.arms}}
                dump(out / "progress.json", progress)
                if done:
                    print(json.dumps(progress), flush=True)
        dump(out / "complete.json", {"planned": planned, "recorded": len(rows)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit-sources", type=int)
    parser.add_argument("--suite", choices=("core", "both"), default="both")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--retry-transport", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    out = args.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    key = os.environ.get("DEEPSEEK_API_KEY") or getpass.getpass("DeepSeek API key (hidden): ")
    client = baseline.Client(key)
    del key
    if not (out / "probe_response.json").exists():
        baseline.probe(client, args.model, out)
    if args.probe_only:
        return
    actors = [json.loads(line) for line in (baseline.RUNTIME / "confirmation15_inputs_private/actors.private.jsonl").read_text().splitlines() if line]
    actors.sort(key=lambda a: baseline.fingerprint(a["source_id"]))
    if args.limit_sources:
        actors = actors[:args.limit_sources]
    settings = {**SETTINGS, "workers_per_model": args.workers}
    layout_ids = read(baseline.RUNTIME / "web15_protocol.json")["source_ids"] if args.suite == "both" else []
    if args.limit_sources:
        layout_ids = [a["source_id"] for a in actors if a["source_id"] in layout_ids]
    codefiles = [HERE / name for name in ("run_methods.py", "common.py", "solver_worker.py")]
    codefiles += list((HERE / "methods").glob("*.py"))
    vendorfiles = list((HERE / "vendor").rglob("*.py"))
    config = {"package_root": str(baseline.PACKAGE_ROOT), "baseline_root": str(BASELINE_ROOT),
        "model": args.model, "endpoint": client.endpoint, "settings": settings,
        "source_ids": [a["source_id"] for a in actors], "layout_source_ids": layout_ids,
        "experimental_arms": args.arms, "environment_arm": "fresh", "suite": args.suite,
        "shared_initials": True, "notes_enabled": False,
        "budget": "all native and method-internal model requests combined <= 40; no hidden verifier budget",
        "integration": "one full method at first save/submit boundary; pending action explicitly deferred",
        "method_temperatures": "native=0; source-method search temperatures retained and charged",
        "study_protocol_sha256": sha(baseline.RUNTIME / "study15_protocol.json"),
        "dataset_integrity_sha256": sha(baseline.RUNTIME / "datasets/medical15_confirmation120_silver_v1/build_integrity.json"),
        "code_sha256": {str(p.relative_to(HERE)): sha(p) for p in sorted(codefiles)},
        "vendor_code_sha256": {str(p.relative_to(HERE)): sha(p) for p in sorted(vendorfiles)},
        "original_baseline_sha256": {name: sha(BASELINE_ROOT / name) for name in ("run_eval.py", "original.py")}}
    if (out / "config.json").exists() and read(out / "config.json") != config:
        raise ValueError("Existing run configuration differs; use a new run directory")
    dump(out / "config.json", config)
    for source in codefiles + vendorfiles + list((HERE / "methods").glob("*_METHOD.md")):
        target = out / "frozen_code" / source.relative_to(HERE)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    dump(out / "run_status.json", {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat()})
    runner = Runner(client, args.model, settings, args.arms, args.retry_transport)
    h.phase = runner.phase
    started = time.monotonic()
    runner.workflow(actors, out / "main")
    if args.suite == "both" and layout_ids:
        selected = [a for a in actors if a["source_id"] in layout_ids]
        with web.ToolService() as service:
            for layout in web.LAYOUTS:
                runner.workflow([web.actor(a, layout) for a in selected], out / "layouts" / layout,
                                layout=layout, tool_port=service.port)
    dump(out / "provider_metadata.json", {"requested_model": args.model,
        "returned_models": sorted(client.models), "system_fingerprints": sorted(client.fingerprints)})
    dump(out / "run_status.json", {"status": "complete", "wall_seconds": time.monotonic() - started,
        "finished_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"complete": str(out), "seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main()
