"""Charged model calls and local solver execution for the three method arms."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


TOKEN_FIELDS = {"input_tokens": "prompt_tokens", "output_tokens": "completion_tokens",
                "prompt_cache_hit_tokens": "prompt_cache_hit_tokens",
                "prompt_cache_miss_tokens": "prompt_cache_miss_tokens"}


class BudgetExceeded(RuntimeError):
    pass


class ModelOutputLimit(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


class Context:
    def __init__(self, client, model, folder, settings, *, retry_transport=False):
        self.client, self.model = client, model
        self.folder, self.settings = Path(folder), settings
        self.retry_transport = retry_transport
        self.problem = {}
        self.requests = self.logical_index = self.executions = 0
        self.usage = {key: 0 for key in TOKEN_FIELDS}
        self.usage["unreported_requests"] = {key: 0 for key in TOKEN_FIELDS}
        self.stages = []

    def record(self, name, value):
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        dump(self.folder / "method" / (name + ".json"), value)

    def chat(self, payload, *, stage):
        """Replay recorded completions; never resample an ambiguous dispatch.

        Explicit HTTP rejections may be resumed when requested. All physical
        attempts, including rejected ones, consume the same 40-call budget.
        """
        self.logical_index += 1
        number = self.logical_index
        prefix = self.folder / ("call_%03d" % number)
        reqfile = prefix.with_suffix(".request.private.json")
        resfile = prefix.with_suffix(".response.private.json")
        attemptsfile = prefix.with_suffix(".attempts.json")
        if reqfile.exists() and fingerprint(read(reqfile)) != fingerprint(payload):
            raise RuntimeError("replay_request_mismatch")
        attempts = read(attemptsfile) if attemptsfile.exists() else []
        rejected = sum(a.get("error", "").startswith("api_http_") for a in attempts)
        if resfile.exists():
            raw = read(resfile)
            self.requests += len(attempts)
        else:
            self.requests += len(attempts)
            if attempts:
                last = attempts[-1]
                if not last.get("error", "").startswith("api_http_"):
                    raise RuntimeError("indeterminate_inflight_no_resample")
                if not self.retry_transport:
                    raise RuntimeError(last["error"])
            if self.requests >= self.settings["max_requests"]:
                raise BudgetExceeded("total_request_limit")
            dump(reqfile, payload)
            attempts.append({"stage": stage, "dispatch_epoch": time.time(),
                             "request_sha256": fingerprint(payload)})
            dump(attemptsfile, attempts)
            self.requests += 1
            try:
                raw = self.client.chat(payload)
                dump(resfile, raw)
                attempts[-1]["completed_epoch"] = time.time()
                dump(attemptsfile, attempts)
            except Exception as error:
                transport_status = (str(error) if str(error).startswith("api_http_")
                                    else "indeterminate_transport_" + type(error).__name__)
                attempts[-1]["error"] = transport_status
                dump(attemptsfile, attempts)
                for key in TOKEN_FIELDS:
                    self.usage["unreported_requests"][key] += rejected + 1
                raise RuntimeError(transport_status) from None
        for key, remote in TOKEN_FIELDS.items():
            self.usage["unreported_requests"][key] += rejected
            value = (raw.get("usage") or {}).get(remote)
            if value is None:
                self.usage["unreported_requests"][key] += 1
            else:
                self.usage[key] += value
        self.stages.append({"logical_request": number, "stage": stage,
                            "physical_requests": len(attempts), "usage": raw.get("usage", {}),
                            "temperature": payload.get("temperature", 0),
                            "finish_reason": raw["choices"][0].get("finish_reason")})
        dump(self.folder / "request_accounting.json", {"requests": self.requests,
             "logical_requests": number, "usage": self.usage, "stages": self.stages})
        # Native output truncation is terminal in the unchanged CPM adapter.
        # Source methods instead receive the content exactly as their original
        # provider wrapper does, so their published parser/repair can handle it.
        if stage == "native" and raw["choices"][0].get("finish_reason") == "length":
            raise ModelOutputLimit("output_limit_not_content_error")
        return raw

    def llm(self, messages, *, stage, temperature=0, json_mode=False, max_tokens=None):
        request = {"model": self.model, "messages": messages, "temperature": temperature,
                   "thinking": {"type": "disabled"}, "stream": False,
                   "max_tokens": min(max_tokens or self.settings["max_output_tokens"],
                                     self.settings["max_output_tokens"])}
        if json_mode:
            request["response_format"] = {"type": "json_object"}
        raw = self.chat(request, stage=stage)
        content = raw["choices"][0]["message"].get("content")
        if not isinstance(content, str):
            raise RuntimeError("method_empty_assistant_content")
        return content

    def execute(self, code, *, stage, timeout=20):
        self.executions += 1
        self.record("solver_%03d_input" % self.executions, {"stage": stage, "code": code})
        started = time.monotonic()
        # Generated reasoning programs receive no API credential or filesystem
        # helpers. A child process also separates Z3 contexts between workers.
        env = {k: v for k, v in os.environ.items()
               if not any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
        try:
            process = subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("solver_worker.py"))],
                input=json.dumps({"code": code}), text=True, capture_output=True,
                timeout=timeout, env=env, cwd=self.folder)
            value = json.loads(process.stdout) if process.stdout.strip() else {
                "ok": False, "stderr": process.stderr, "stdout": "", "result": None}
        except subprocess.TimeoutExpired:
            value = {"ok": False, "stderr": "solver_process_timeout", "stdout": "", "result": None}
        except Exception as error:
            value = {"ok": False, "stderr": type(error).__name__ + ": " + str(error),
                     "stdout": "", "result": None}
        value["seconds"] = time.monotonic() - started
        self.record("solver_%03d_output" % self.executions, {"stage": stage, **value})
        return value
