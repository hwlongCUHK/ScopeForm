"""Full released SSR inference pipeline, adapted to CPM through an RPC backend.

The official definitions are compiled directly from the pinned vendor source.
Only import-time dataset/client setup is excluded. No solver algorithm, parser,
repair, fallback, or branch is replaced by a custom approximation.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import re
import selectors
import subprocess
import sys
import time
from types import SimpleNamespace
import typing


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "ssr" / "SSR"
OFFICIAL_COMMIT = "c4a2d9c3f813adcff8931f19498e94141654e899"
LABELS = {"A": "supported", "B": "refuted", "C": "not_determined"}
LOCAL_WAIT_SECONDS = 60


def _problem_parts(problem):
    task = problem.get("task_input", {})
    claims = task.get("claims", [])
    if not isinstance(claims, list) or not claims:
        raise ValueError("ssr_missing_public_claims")
    for claim in claims:
        if not isinstance(claim, dict) or not claim.get("item_id") or not isinstance(claim.get("statement"), str):
            raise ValueError("ssr_invalid_public_claim")
    # A target statement is the question, never an asserted context premise.
    # Proposed native labels are intentionally not fed to the formalizer.
    scope = {key: task[key] for key in (
        "task_id", "stage", "event", "source_notice", "scenario_contract",
        "simulated_environment_notice", "evidence_rule",
    ) if key in task}
    context = (
        "The following is a source-conditioned evidence task. Distinct simulated "
        "scenes are independent. Original historical records provide reference "
        "rules or historical facts, not missing current scene values. Missing "
        "information is not negation. Recommendations are not execution. "
        "Only actually acquired public tool results below are evidence. "
        "Task metadata is scope information, not a clinical finding.\n\n"
        "Public scope metadata:\n" + json.dumps(scope, ensure_ascii=False)
        + "\n\nActually acquired public tool observations:\n"
        + json.dumps(problem.get("observations", []), ensure_ascii=False)
    )
    return claims, context


def _load_vendor_namespace(create):
    """Load unchanged official function/class ASTs, with no client side effects."""
    import z3

    chain_name = "_cpm_official_ssr_chain"
    spec = importlib.util.spec_from_file_location(chain_name, VENDOR / "chain.py")
    chain = importlib.util.module_from_spec(spec)
    sys.modules[chain_name] = chain
    spec.loader.exec_module(chain)
    namespace = {name: getattr(z3, name) for name in dir(z3) if not name.startswith("__")}
    namespace.update({name: getattr(typing, name) for name in dir(typing) if not name.startswith("__")})
    namespace.update({
        "__name__": "_cpm_official_ssr", "re": re, "json": json,
        "itertools": itertools, "time": time, "Path": Path,
        "DEFAULT_MODEL": "deepseek-flash", "GENERATE_SYMBOLIC_CHAINS": True,
        "TRANSLATE_CHAINS_TO_NL": True, "CHAIN_MODE": "both",
        "generate_reasoning_chains": chain.generate_reasoning_chains,
        "llm_client": SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    })
    source_path = VENDOR / "run.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                   and getattr(node, "name", "") not in ("parse_args",)]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace


def _call_kind(messages):
    text = "\n".join(str(row.get("content", "")) for row in messages)
    if "strict translator" in text:
        return "formalization"
    if "faithfully verbalize" in text:
        return "chain_verbalization"
    if "atomic condition" in text and "supported" in text:
        return "missing_premise_check"
    if "#### <Letter>" in text:
        return "cot_fallback"
    return "official_llm_call"


def _send(value):
    sys.stdout.write(json.dumps(value, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _worker():
    problem = json.loads(sys.stdin.readline())
    claims, context = _problem_parts(problem)
    item_id = ""
    call_number = 0
    per_claim_calls = []

    def create(**kwargs):
        nonlocal call_number
        call_number += 1
        kind = _call_kind(kwargs["messages"])
        per_claim_calls.append({"kind": kind, "max_tokens": kwargs.get("max_tokens"),
                                "temperature": kwargs.get("temperature")})
        _send({"type": "llm", "messages": kwargs["messages"],
               "stage": f"ssr_{item_id}_{kind}_{call_number:03d}",
               "temperature": kwargs.get("temperature", 0),
               "max_tokens": kwargs.get("max_tokens", 1000)})
        response = json.loads(sys.stdin.readline())
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response["content"]))])

    namespace = _load_vendor_namespace(create)
    pipeline = namespace["run_pipeline"]
    vendor_file = str(VENDOR / "run.py")
    items, records = [], []
    for claim in claims:
        item_id = claim["item_id"]
        per_claim_calls.clear()
        trace_data = {"parser_true_returns": []}

        def trace(frame, event, arg):
            if frame.f_code.co_filename != vendor_file:
                return None
            if event == "return":
                local = frame.f_locals
                if frame.f_code.co_name == "parse_fact" and namespace["is_true"](arg):
                    trace_data["parser_true_returns"].append({"input": local.get("s"),
                                                               "line": frame.f_lineno})
                elif frame.f_code.co_name == "run_pipeline":
                    for key in ("sat_known", "sat_query", "sat_not_query", "weights",
                                "constraints_texts", "q_cost", "nq_cost",
                                "discriminator_support", "pre_time"):
                        if key in local:
                            trace_data[key] = local[key]
                    for key in ("constraints_exprs",):
                        if key in local:
                            trace_data[key] = [str(x) for x in local[key]]
                    for key in ("query_expr", "discriminator", "current_query"):
                        if key in local:
                            trace_data[key] = str(local[key])
                    if "suspicious" in local:
                        trace_data["suspicious"] = sorted(local["suspicious"])
            return trace

        started = time.monotonic()
        try:
            sys.settrace(trace)
            raw = pipeline(context, claim["statement"], generate_chains=True,
                           translate_chains=True, chain_mode="both", model="deepseek-flash")
            sys.settrace(None)
            # The context is already saved once in the method input. Avoid
            # duplicating it in the method result that the native agent sees.
            raw.pop("context", None)
            selected = raw.get("selected_answer")
            status = "completed" if selected in LABELS else "official_no_answer"
            branch = ("cot_fallback" if "llm_cot" in raw else
                      "soft_conflict" if trace_data.get("sat_known") is False else
                      "missing_premise" if raw.get("sat_query") and raw.get("sat_not_query") else
                      "hard_entailment")
            record = {"item_id": item_id, "status": status, "branch": branch,
                      "raw": raw, "execution_trace": trace_data,
                      "calls": list(per_claim_calls), "seconds": time.monotonic() - started,
                      "official_unknown_from_conflict": branch == "soft_conflict" and selected == "C"}
            if selected in LABELS:
                items.append({"item_id": item_id, "label": LABELS[selected]})
        except Exception as exc:
            sys.settrace(None)
            record = {"item_id": item_id, "status": "official_pipeline_error",
                      "error_type": type(exc).__name__, "error": str(exc),
                      "execution_trace": trace_data, "calls": list(per_claim_calls),
                      "seconds": time.monotonic() - started}
        records.append(record)
        _send({"type": "record", "name": f"claim_{item_id}", "value": record})
        # Keep one outstanding protocol message at a time. This also avoids
        # selector/readline interaction with prefetched pipe data.
        json.loads(sys.stdin.readline())
    summary = [{key: row[key] for key in ("item_id", "status", "branch",
                                         "official_unknown_from_conflict", "error_type", "error") if key in row}
               for row in records]
    _send({"type": "result", "value": {"method": "ssr", "items": items,
           "status": "completed" if len(items) == len(claims) else "partial_method_failure",
           "diagnostics": {"official_commit": OFFICIAL_COMMIT, "claims": summary,
                           "generated_chains": True, "translated_chains": True,
                           "representation": "official unary grounded Boolean parser"}}})


def run(ctx):
    """Run the released per-query algorithm for every bounded CPM claim."""
    claims, _ = _problem_parts(ctx.problem)
    ctx.record("provenance", {"official_repository": "https://github.com/1240148048/SSR",
        "official_commit": OFFICIAL_COMMIT,
        "source_sha256": {name: hashlib.sha256((VENDOR / name).read_bytes()).hexdigest()
                          for name in ("run.py", "chain.py")},
        "settings": {"generate_chains": True, "translate_chains": True,
                     "chain_mode": "both", "translation_attempts": 3,
                     "translation_max_tokens": 1000, "discriminator_max_tokens": 50,
                     "temperature": 0, "python_hash_seed": 0,
                     "local_compute_wait_seconds": LOCAL_WAIT_SECONDS},
        "claims": [claim["item_id"] for claim in claims]})
    # Child runs only the trusted upstream parser/solver. API credentials and
    # transports remain in the parent; every request passes through ctx.llm.
    env = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "TMPDIR") if key in os.environ}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "--worker"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    recorded = []
    try:
        process.stdin.write(json.dumps(ctx.problem, ensure_ascii=False) + "\n")
        process.stdin.flush()
        while True:
            if not selector.select(LOCAL_WAIT_SECONDS):
                return {"method": "ssr", "items": [], "status": "ssr_local_compute_timeout",
                        "diagnostics": {"completed_claims": recorded,
                                        "timeout_seconds": LOCAL_WAIT_SECONDS}}
            line = process.stdout.readline()
            if not line:
                error = process.stderr.read()
                return {"method": "ssr", "items": [], "status": "ssr_worker_failed",
                        "diagnostics": {"returncode": process.poll(), "stderr": error,
                                        "completed_claims": recorded}}
            message = json.loads(line)
            if message["type"] == "llm":
                # In particular, do not intercept BudgetExceeded or transport
                # exceptions here. They must terminate the outer phase.
                content = ctx.llm(message["messages"], stage=message["stage"],
                                  temperature=message["temperature"],
                                  max_tokens=message["max_tokens"])
                process.stdin.write(json.dumps({"content": content}, ensure_ascii=False) + "\n")
                process.stdin.flush()
            elif message["type"] == "record":
                ctx.record(message["name"], message["value"])
                recorded.append(message["value"]["item_id"])
                process.stdin.write('{"ack":true}\n')
                process.stdin.flush()
            elif message["type"] == "result":
                return message["value"]
            else:
                raise ValueError("ssr_invalid_worker_protocol")
    finally:
        selector.close()
        if process.poll() is None:
            process.kill()
        process.communicate()


if __name__ == "__main__" and sys.argv[1:] == ["--worker"]:
    _worker()
