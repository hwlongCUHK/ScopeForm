"""Full official NL2Logic parser/compiler, adapted to acquired CPM evidence.

The official recursive Pipeline body and AST implementation are loaded unchanged.
Only the provider wrapper, the document-to-sentences input bridge, and multi-claim
entailment queries are supplied here. No notes or private evaluator data are used.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import re
import sys
import threading

from pydantic import ValidationError


VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "nl2logic"
OFFICIAL_COMMIT = "3d625c5b33ae8ae390a2156639662d3e67063303"
_VENDOR_LOCK = threading.Lock()
_VENDOR_CLASSES = None

BRIDGE_INSTRUCTION = """Prepare the INPUT to the published NL2Logic semantic parser.
The source task is a bounded evidence-checking task, not a request for medical
advice. Use only the task and original tool observations supplied below. Old
saved labels and a claim's own wording are not premises proving that claim.

Return JSON with exactly two arrays:
  premises: [{"text": "one English factual or conditional sentence",
              "evidence_ids": ["actual acquired anchor IDs"]}]
  queries: [{"item_id": "original claim ID", "text": "English claim sentence"}]

Translate the acquired relevant source statements and every requested claim into
English sentences for an English FOL parser. Include the factual premises and
applicable conditional rules needed to evaluate every claim; preserve explicit
negation, quantifiers, conditions, the identity of each independent scene, and
current versus historical time. Keep a consistent name for the same entity in
all sentences. A rule and its applicability conditions must remain together.
Do not treat unobserved information as false. Do not fill missing current values
from historical records or other scenes. Do not promote a recommendation to an
executed event. Do not introduce outside knowledge or make up numerical limits.
Do not decide labels, evaluate numerical comparisons, calculate daily totals,
or add derived conclusions: those are the reasoning system's job. Preserve
numeric statements in natural language even though the published FOL AST may
not be able to express their arithmetic. Do not simplify them to a guessed
supported/refuted predicate. Do not truncate the required premises or omit
claims to fit a request budget. Output JSON only.
"""


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _official_classes():
    """Import the unchanged algorithm without unused provider dependencies.

    The upstream module eagerly imports ollama/openai and loads .env. Extracting
    its one Pipeline class avoids those provider side effects; no class methods
    or prompts are rewritten, including their original fallbacks.
    """
    global _VENDOR_CLASSES
    with _VENDOR_LOCK:
        if _VENDOR_CLASSES is None:
            nodes = _load_module("_cpm_nl2logic_ast", VENDOR / "ast_rl.py")
            schemas = _load_module("_cpm_nl2logic_schemas", VENDOR / "structured_output.py")
            source = ast.parse((VENDOR / "pipeline.py").read_text(encoding="utf-8"))
            pipeline = next(n for n in source.body
                            if isinstance(n, ast.ClassDef) and n.name == "Pipeline")
            namespace = {**vars(nodes), **vars(schemas)}
            module = ast.Module(body=[pipeline], type_ignores=[])
            exec(compile(module, str(VENDOR / "pipeline.py"), "exec"), namespace)
            _VENDOR_CLASSES = (namespace["Pipeline"], nodes)
    return _VENDOR_CLASSES


class _ContextWrapper:
    """Upstream generate(text, PydanticModel) using the shared charged client."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.calls = 0

    def generate(self, text, fmt):
        self.calls += 1
        schema = json.dumps(fmt.model_json_schema(), ensure_ascii=False)
        raw = self.ctx.llm(
            [{"role": "user", "content": text +
              "\nReturn only a JSON object satisfying this output schema:\n" + schema}],
            stage=f"nl2logic.{self.calls:03d}.{fmt.__name__}",
            temperature=0,
            json_mode=True,
        )
        return fmt.model_validate_json(raw)


def _bridge(ctx):
    # The proposed action may contain the native agent's provisional labels.
    # It is deliberately excluded from the parser's premise-construction input.
    problem = {"task_input": ctx.problem.get("task_input", {}),
               "observations": ctx.problem.get("observations", [])}
    raw = ctx.llm(
        [{"role": "system", "content": BRIDGE_INSTRUCTION},
         {"role": "user", "content": json.dumps(problem, ensure_ascii=False)}],
        stage="nl2logic.input_bridge", temperature=0, json_mode=True,
    )
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("input_bridge_not_object")
    premises, queries = value.get("premises"), value.get("queries")
    if not isinstance(premises, list) or not isinstance(queries, list) or not queries:
        raise ValueError("input_bridge_missing_premises_or_queries")
    for premise in premises:
        if not isinstance(premise, dict) or not isinstance(premise.get("text"), str) or not premise["text"].strip():
            raise ValueError("input_bridge_invalid_premise")
    ids = []
    for query in queries:
        if not isinstance(query, dict) or not isinstance(query.get("text"), str) or not query["text"].strip() or not isinstance(query.get("item_id"), str):
            raise ValueError("input_bridge_invalid_query")
        ids.append(query["item_id"])
    if len(set(ids)) != len(ids):
        raise ValueError("input_bridge_duplicate_query")
    claims = problem["task_input"].get("claims", [])
    expected = {c["item_id"] for c in claims if isinstance(c, dict) and "item_id" in c}
    if expected and set(ids) != expected:
        raise ValueError("input_bridge_query_ids_differ_from_task")
    return value


def compile_entailment(nodes, premises, queries):
    """Use upstream declaration/expression passes; add standard T/F/U queries.

    This does not add numeric operators or fix upstream predicate naming,
    implication direction, coreference, or cross-sentence arity behavior.
    """
    all_nodes = [n for _, n in premises] + [n for _, n in queries]
    logic = nodes.RelationalLogic(original_sentence="CPM acquired evidence", sentences=all_nodes)
    lines = ["from z3 import *", logic.z3declaration_pass(), "s = Solver()"]
    lines += ["s.add(" + n.z3expression_pass() + ")" for _, n in premises]
    lines += ["base_status = str(s.check())",
              "result = {'base_status': base_status, 'items': [], 'queries': []}"]
    for item, node in queries:
        lines += [
            "if base_status == 'sat':",
            "    query = " + node.z3expression_pass(),
            "    s.push()",
            "    s.add(query)",
            "    positive = str(s.check())",
            "    s.pop()",
            "    s.push()",
            "    s.add(Not(query))",
            "    negative = str(s.check())",
            "    s.pop()",
            "    label = None",
            "    if positive == 'sat' and negative == 'unsat': label = 'supported'",
            "    elif positive == 'unsat' and negative == 'sat': label = 'refuted'",
            "    elif positive == 'sat' and negative == 'sat': label = 'not_determined'",
            "    result['queries'].append({'item_id': " + repr(item["item_id"]) +
            ", 'positive': positive, 'negative': negative})",
            "    if label is not None: result['items'].append({'item_id': " +
            repr(item["item_id"]) + ", 'label': label})",
        ]
    return "\n".join(lines) + "\n"


def run(ctx):
    pipeline_class, nodes = _official_classes()
    wrapper = _ContextWrapper(ctx)
    pipeline = pipeline_class.__new__(pipeline_class)
    pipeline.llm = wrapper
    pipeline.logging = False
    diagnostics = {
        "official_commit": OFFICIAL_COMMIT,
        "implementation": "official_recursive_pipeline_and_ast_with_cpm_adapter",
        "numeric_arithmetic_supported": False,
        "predicate_coreference_repair_added": False,
        "parser_calls": 0,
        "parsed_premises": 0,
        "parsed_queries": 0,
    }
    parsed_premises, parsed_queries = [], []
    try:
        bridge = _bridge(ctx)
        ctx.record("nl2logic_input_bridge", bridge)
        diagnostics.update(premises=len(bridge["premises"]), queries=len(bridge["queries"]),
                           sentences_with_numbers=sum(bool(re.search(r"\d", x["text"]))
                               for x in bridge["premises"] + bridge["queries"]))
        for kind, entries, target in (("premise", bridge["premises"], parsed_premises),
                                      ("query", bridge["queries"], parsed_queries)):
            for index, item in enumerate(entries):
                diagnostics.update(active_kind=kind, active_index=index, parser_calls=wrapper.calls)
                ctx.record("nl2logic_progress", diagnostics)
                node = pipeline.rephrase_and_parse(item["text"])
                target.append((item, node))
                diagnostics["parsed_premises" if kind == "premise" else "parsed_queries"] = len(target)
                ctx.record(f"nl2logic_{kind}_{index:03d}", {
                    "input": item, "ast": node.model_dump(), "formula": str(node),
                    "declarations": node.z3declaration_pass(),
                    "expression": node.z3expression_pass(),
                })
        diagnostics["parser_calls"] = wrapper.calls
        code = compile_entailment(nodes, parsed_premises, parsed_queries)
        ctx.record("nl2logic_compiled_program", {"code": code})
        execution = ctx.execute(code, stage="nl2logic.entailment", timeout=20)
        ctx.record("nl2logic_execution", execution)
        if not execution.get("ok") or not isinstance(execution.get("result"), dict):
            return {"method": "nl2logic", "status": "solver_execution_failed", "items": [],
                    "diagnostics": {**diagnostics, "execution": execution}}
        result = execution["result"]
        diagnostics.update(solver_queries=result.get("queries", []), base_status=result.get("base_status"))
        if result.get("base_status") != "sat":
            return {"method": "nl2logic", "status": "inconsistent_formalization" if result.get("base_status") == "unsat"
                    else "solver_unknown", "items": [], "diagnostics": diagnostics}
        items = result.get("items", [])
        return {"method": "nl2logic", "status": "completed" if len(items) == len(parsed_queries)
                else "partial_solver_unknown", "items": items, "diagnostics": diagnostics}
    except (json.JSONDecodeError, ValidationError, ValueError, RecursionError) as error:
        # The shared client's BudgetExceeded and transport failures must propagate.
        diagnostics.update(parser_calls=wrapper.calls, error_type=type(error).__name__, error=str(error))
        ctx.record("nl2logic_failure", diagnostics)
        return {"method": "nl2logic", "status": "formalization_failed", "items": [], "diagnostics": diagnostics}
