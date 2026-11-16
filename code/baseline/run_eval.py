"""Original CPM native tool loop, with optional phase-local evidence notes.

The baseline prompt, World, branching and actual-initial pairing are reused.
The native loop follows deepseek19_native_v2.NativeAdapter.phase. Changes are
the official endpoint, a new output directory, and optional note parameters.
No evaluator targets are loaded by this process.
"""
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.request

from original import PACKAGE_ROOT, RUNTIME
import confirmation14_workflow as h
import run_web15 as rw
import web15_runtime as web
from pilot_runtime import fingerprint
from run_generation import NoRedirect, read, dump, digest
from evidence_notes import Notes

ARMS = ("baseline", "notes")
EVENTS = ("key_mixed", "single_definite", "single_uncertainty", "neutral")
SETTINGS = {"temperature": 0, "max_output_tokens": 8192, "max_actions": 40,
            "workers_per_model": 24, "thinking": "disabled"}
NATIVE_INSTRUCTION = (
    "\n使用当前公开的原生工具执行任务。优先每次调用一个工具；多个调用将按返回顺序执行，"
    "逐个计入40动作上限。不要输出推理过程，必须实际调用submit_answer。\n"
)
NOTES_INSTRUCTION = (
    "\n你可以在正常工具调用的 evidence_patch 参数中更新简短证据笔记，不需要单独调用工具。"
    "读到与某个item有关的原文后，用实际document_id和anchor_id将其关联到item_id。"
    "只填写影响该题判断的字段：scope区分对象/独立场景和当前/历史；"
    "assertion区分肯定/否定/不确定；action_stage区分建议/计划/下单/执行/完成；"
    "conditions记录来源适用条件及当前是否满足。"
    "只写新增或变化的关联，不抄原文、不写长解释、不必填写无关字段。"
    "这些笔记是你的工作记录，不是额外事实或正确答案；下一轮会提供当前精简记录。"
)


class Client:
    def __init__(self, key, endpoint="https://api.deepseek.com"):
        self.key = key
        self.endpoint = endpoint.rstrip("/")
        self.models = set()
        self.fingerprints = set()
        self.lock = threading.Lock()

    def request(self, path, payload=None):
        data = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
        request = urllib.request.Request(self.endpoint + path, data=data, headers={
            "Authorization": "Bearer " + self.key,
            "Content-Type": "application/json",
        })
        try:
            with urllib.request.build_opener(NoRedirect()).open(request, timeout=300) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError("api_http_" + str(error.code)) from None

    def chat(self, payload):
        result = self.request("/chat/completions", payload)
        with self.lock:
            if result.get("model"):
                self.models.add(result["model"])
            if result.get("system_fingerprint"):
                self.fingerprints.add(result["system_fingerprint"])
        return result


class Runner:
    def __init__(self, client, model, settings):
        self.client, self.model, self.settings = client, model, settings
        self.local = threading.local()

    def phase(self, world, folder, port, model, tokenizer, settings):
        """Serial native execution with the original 40-action/request caps."""
        folder = Path(folder)
        use_notes = getattr(self.local, "arm", "baseline") == "notes"
        notes = Notes()
        start_hash, public_hash = fingerprint(world.state), fingerprint(world.public)
        started, elapsed, requests, action_count, next_request = time.monotonic(), 0., 0, 0, 1
        token_fields = {"input_tokens": "prompt_tokens", "output_tokens": "completion_tokens",
                        "prompt_cache_hit_tokens": "prompt_cache_hit_tokens", "prompt_cache_miss_tokens": "prompt_cache_miss_tokens"}
        usage = {key: 0 for key in token_fields}
        usage["unreported_requests"] = {key: 0 for key in token_fields}
        instruction = h.POLICY + NATIVE_INSTRUCTION
        if use_notes:
            instruction += NOTES_INSTRUCTION + "\n"
        messages = [{"role": "user", "content": instruction + json.dumps(world.initial_input(), ensure_ascii=False)}]
        status = "action_limit"
        cpfile = folder / "checkpoint.private.json"
        if cpfile.exists():
            old = read(cpfile)
            if old["start_hash"] != start_hash or old["public_hash"] != public_hash:
                raise ValueError("Actual initial/checkpoint differs")
            if isinstance(world, web.WebWorld):
                tool_port = world.web_http_port
                world = web.restore_proxy(old["world"])
                world.web_http_port = tool_port
            else:
                world = h.restore(old["world"])
            messages, usage, status = old["messages"], old["usage"], old["status"]
            requests, elapsed = old["requests"], old["seconds"]
            action_count, next_request = old["action_count"], old["next_request"]
            notes = Notes.from_dict(old.get("evidence_notes", {}))
            if old["terminal"]:
                return world, status, requests, elapsed, usage

        def save(terminal=False):
            dump(cpfile, {"start_hash": start_hash, "public_hash": public_hash,
                "world": h.prior.snapshot(world), "messages": messages,
                "requests": requests, "seconds": elapsed + time.monotonic() - started,
                "next_request": next_request, "action_count": action_count,
                "status": status, "terminal": terminal, "usage": usage,
                "evidence_notes": notes.to_dict() if use_notes else {},
                "harness": "cpm_native_evidence_notes", "notes_enabled": use_notes})

        save()
        while action_count < settings["max_actions"] and requests < settings["max_actions"]:
            number = next_request
            usage_recorded = False
            response = folder / ("response_%03d.private.json" % number)
            dispatch = folder / ("dispatch_%03d.json" % number)
            try:
                if response.exists():
                    raw = read(response)
                    requests = max(requests, number)
                elif dispatch.exists():
                    requests = max(requests, number)
                    for key in token_fields:
                        usage.setdefault("unreported_requests", {}).setdefault(key, 0)
                        usage["unreported_requests"][key] += 1
                    status = "indeterminate_inflight_no_resample"
                    break
                else:
                    wire = copy.deepcopy(messages)
                    if use_notes and notes.context()["claims"]:
                        # One current view per request, never accumulated in history.
                        wire.append({"role": "user", "content": "当前证据笔记（你此前的记录）：\n"
                                     + json.dumps(notes.context(), ensure_ascii=False, separators=(",", ":"))})
                    request = {"model": model, "messages": wire,
                        "tools": notes.tool_specs(world.tools()) if use_notes else world.tools(),
                        "tool_choice": "auto", "parallel_tool_calls": False,
                        "temperature": 0, "thinking": {"type": "disabled"},
                        "max_tokens": settings["max_output_tokens"], "stream": False}
                    dump(folder / ("request_%03d.private.json" % number), request)
                    dump(dispatch, {"request_sha256": fingerprint(request), "dispatch_epoch": time.time()})
                    requests += 1
                    raw = self.client.chat(request)
                    dump(response, raw)
                u = raw.get("usage") or {}
                for key, remote_key in token_fields.items():
                    if u.get(remote_key) is None:
                        usage.setdefault("unreported_requests", {}).setdefault(key, 0)
                        usage["unreported_requests"][key] += 1
                    else:
                        usage[key] += u[remote_key]
                usage_recorded = True
                choice = raw["choices"][0]
                if choice.get("finish_reason") == "length":
                    status = "output_limit_not_content_error"
                    break
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
                for call, name, args in parsed:
                    if world.done:
                        result = {"ok": False, "error": "not_executed_after_terminal_submission"}
                    elif action_count >= settings["max_actions"]:
                        result = {"ok": False, "error": "not_executed_action_cap"}
                    else:
                        action_count += 1
                        if use_notes:
                            notes.update(args.pop("evidence_patch", None))
                        result = world.call(name, args)
                        if use_notes:
                            notes.capture(name, result)
                    messages.append({"role": "tool", "tool_call_id": call["id"],
                                     "content": json.dumps(result, ensure_ascii=False)})
                next_request = number + 1
                if world.done:
                    status = "submitted"
                    break
                save()
            except Exception as error:
                if not usage_recorded and requests >= number:
                    for key in token_fields:
                        usage.setdefault("unreported_requests", {}).setdefault(key, 0)
                        usage["unreported_requests"][key] += 1
                status = str(error) if str(error).startswith(("api_http_", "invalid_native_")) else "runner_or_transport_" + type(error).__name__
                dump(folder / ("error_%03d.json" % number), {"status": status})
                break
        save(True)
        return world, status, requests, elapsed + time.monotonic() - started, usage

    def one(self, actor, event, arm, out, base=None, layout=None, tool_port=0):
        self.local.arm = arm
        original_method = "initial" if event is None else "fresh"
        folder = out / ("shared" if event is None else arm)
        if layout is None:
            row, world = h.one(actor, event, original_method, folder, 0, self.model, None, self.settings, base=base)
        else:
            row, world = rw.one(actor, event, original_method, layout, folder, 0, tool_port,
                                self.model, None, self.settings, base=base)
        row = {**row, "arm": "initial" if event is None else arm,
               "harness": "cpm_native_evidence_notes", "environment_arm": original_method}
        return row, world

    def workflow(self, actors, out, layout=None, tool_port=0):
        out = Path(out)
        rows, pairing = [], {}
        planned = sum(1 + len(a["events"]) * len(ARMS) for a in actors)
        with ThreadPoolExecutor(max_workers=self.settings["workers_per_model"]) as pool:
            futures = {pool.submit(self.one, actor, None, "baseline", out,
                                   layout=layout, tool_port=tool_port): (actor, True)
                       for actor in sorted(actors, key=lambda x: fingerprint(x["source_id"]))}
            while futures:
                done, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    actor, initial = futures.pop(future)
                    row, world = future.result()
                    rows.append(row)
                    if initial:
                        pairing[actor["source_id"]] = {
                            "initial_world_sha256": web.checkpoint_hash(world) if layout else fingerprint(h.prior.snapshot(world)),
                            "actual_submitted": world.done}
                        jobs = [(event, arm) for event in actor["events"] for arm in ARMS]
                        for event, arm in sorted(jobs, key=lambda x: fingerprint([actor["source_id"], x[0]["event_id"], x[1]])):
                            futures[pool.submit(self.one, actor, event, arm, out, world, layout, tool_port)] = (actor, False)
                    dump(out / "results.json", rows)
                    dump(out / "pairing_audit.json", pairing)
                progress = {"suite": layout or "core", "planned": planned, "recorded": len(rows),
                    "submitted": sum(r["submitted"] for r in rows),
                    "statuses": dict(Counter(r["status"] for r in rows))}
                dump(out / "progress.json", progress)
                if done:
                    print(json.dumps(progress), flush=True)
        dump(out / "complete.json", {"planned": planned, "recorded": len(rows)})


def probe(client, model, out):
    models = client.request("/models")
    ids = [m["id"] for m in models.get("data", [])]
    dump(out / "api_models.json", {"requested": model, "available_models": ids})
    if model not in ids:
        raise ValueError("Requested model unavailable: " + model)
    response = client.chat({"model": model, "temperature": 0, "thinking": {"type": "disabled"},
        "max_tokens": 256, "stream": False, "messages": [{"role": "user", "content":
            "Nonmedical interface check. Call echo exactly once with value='ok'."}],
        "tools": [{"type": "function", "function": {"name": "echo", "description": "Echo a value",
            "parameters": {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}}}],
        "tool_choice": "auto", "parallel_tool_calls": False})
    dump(out / "probe_response.json", response)
    calls = response["choices"][0]["message"].get("tool_calls", [])
    if not calls or calls[0]["function"]["name"] != "echo":
        raise ValueError("Native tool probe failed")
    print(json.dumps({"probe": "passed", "returned_model": response.get("model"), "usage": response.get("usage")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit-sources", type=int, help="Source-blind smoke run; omit for the frozen full cohort")
    parser.add_argument("--suite", choices=("core", "both"), default="both")
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    out = args.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    key = os.environ.get("DEEPSEEK_API_KEY") or getpass.getpass("DeepSeek API key (hidden): ")
    client = Client(key)
    del key
    if not (out / "probe_response.json").exists():
        probe(client, args.model, out)
    if args.probe_only:
        return
    actors = [json.loads(line) for line in (RUNTIME / "confirmation15_inputs_private/actors.private.jsonl").read_text().splitlines() if line]
    actors.sort(key=lambda a: fingerprint(a["source_id"]))
    if args.limit_sources:
        actors = actors[:args.limit_sources]
    settings = {**SETTINGS, "workers_per_model": args.workers}
    layout_ids = read(RUNTIME / "web15_protocol.json")["source_ids"] if args.suite == "both" else []
    if args.limit_sources:
        layout_ids = [a["source_id"] for a in actors if a["source_id"] in layout_ids]
    config = {"package_root": str(PACKAGE_ROOT), "model": args.model, "endpoint": client.endpoint,
        "paper_model": "deepseek-v4.1-flash", "paper_endpoint": "https://www.micuapi.ai/v1",
        "settings": settings, "source_ids": [a["source_id"] for a in actors], "events": list(EVENTS),
        "layout_source_ids": layout_ids, "experimental_arms": list(ARMS), "environment_arm": "fresh",
        "suite": args.suite, "shared_initials": True, "notes_reset_each_phase": True,
        "source_selection": "sha256(source_id), no target-label selection",
        "dataset_integrity_sha256": digest(RUNTIME / "datasets/medical15_confirmation120_silver_v1/build_integrity.json"),
        "study_protocol_sha256": digest(RUNTIME / "study15_protocol.json"),
        "code_sha256": {name: digest(Path(__file__).parent / name) for name in ("run_eval.py", "original.py", "evidence_notes.py")}}
    if (out / "config.json").exists() and read(out / "config.json") != config:
        raise ValueError("Existing run configuration differs; use a new run directory")
    dump(out / "config.json", config)
    dump(out / "run_status.json", {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat()})
    runner = Runner(client, args.model, settings)
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
