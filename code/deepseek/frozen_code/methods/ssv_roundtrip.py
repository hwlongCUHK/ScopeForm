"""Small CPM roundtrip/source-check adaptation, not a paper reproduction.

One shared lite candidate -> code-only backtranslation -> re-formalization ->
Z3 equivalence -> one source-grounded diagnosis -> at most one repair of A.
Equivalence measures roundtrip consistency, not faithfulness to the source.
"""
from __future__ import annotations

import ast

from methods import ssv, ssv_lite as lite


BACKTRANSLATE = """Read the executable Z3 program and explain exactly the model it
encodes in natural language, including initial restrictions, every premise and
every query. Interpret expressions; do not repair or infer missing constraints.
Separate unknown/unconstrained values from explicit false values, historical and
current scenarios, and recommendations from observed execution. A declared
variable has no known value unless code constrains it. Preserve IDs.
Return JSON {initial_context: string, constraints: [{id, text}],
queries: [{item_id, text}]}. Include each supplied entry once. No code or labels.
"""

REFORMALIZE = """Translate the supplied natural-language backtranslation into Z3,
using exactly the existing initialization symbols and domains. The initialization
will be reused unchanged. Do not infer omitted source facts or create new symbols.
Each constraint code appends its formula to pre_conditions. Each query code assigns
a Boolean Z3 expression to check_property. Preserve every constraint/query ID.
Return JSON {constraints: [{id, text, code}], queries: [{item_id, text, code}]}.
The backtranslation is your only statement of the premises and queries.
"""

DIAGNOSE = """Compare the original acquired evidence and requested claims with
program A and its code-derived backtranslation. Diagnose a translation defect,
not whether the solver's final labels seem plausible. Check scenario/person,
historical/current time, recommendation vs actual execution, absence vs negation,
applicability conditions, units/relations, and the full requested query meaning.
Inspect the original evidence itself, not only A's rewritten descriptions.

Formal A/B equivalence only tests roundtrip consistency: both may omit the same
source requirement. A SAT difference identifies disagreement, not which side is
right. Distinguish A's source drift from a backtranslation error or B translation
error. Do not change A merely to make it equal to B. Unknown evidence can correctly
leave a variable unconstrained; solver uncertainty is not evidence uncertainty.
Also identify whether shared initialization already contains source observations;
if so, B has inherited those facts rather than independently re-formalizing them.

Return JSON {
  source_drift: [{entry_id: "init|constraint:K1|query:C1", source_quote: string,
                 reason: string, correction: string}],
  roundtrip_errors: [{stage: "backtranslation|reformalization", reason: string}],
  undetermined: [{entry_id: string, reason: string}],
  initialization_contains_source_facts: boolean
}.
source_drift contains ONLY concrete defects of A against original evidence or
claim wording, with a targeted correction. For unsupported additions explain
which restriction lacks support. Context-dependent ambiguities belong in
undetermined. Do not invent missing facts or private target rules.
"""


def _executable_only(code: str) -> str:
    """Comments and docstrings must not tell the backtranslator the intended text."""
    class StripDocstrings(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return None
            return self.generic_visit(node)

    return ast.unparse(StripDocstrings().visit(ast.parse(code)))


def code_only(program: dict) -> dict:
    return {
        "init": _executable_only(program["init"]),
        "constraints": [{"id": c["id"], "code": _executable_only(c["code"])}
                        for c in program["constraints"]],
        "queries": [{"item_id": q["item_id"], "code": _executable_only(q["code"])}
                    for q in program["queries"]],
    }


def _validate_backtranslation(value: dict, program: dict) -> None:
    if not isinstance(value.get("initial_context"), str):
        raise ValueError("roundtrip_missing_initial_context")
    for name, key in (("constraints", "id"), ("queries", "item_id")):
        entries = value.get(name)
        if not isinstance(entries, list) or any(
            not isinstance(e, dict) or not isinstance(e.get(key), str)
            or not isinstance(e.get("text"), str) for e in entries
        ):
            raise ValueError("roundtrip_bad_" + name)
        if sorted(e[key] for e in entries) != sorted(e[key] for e in program[name]):
            raise ValueError("roundtrip_changed_" + name + "_ids")


def equivalence_code(a: dict, b: dict) -> str:
    """Compare premises and query meanings under the same domain initialization.

Queries are compared under domain constraints, not all observed facts: otherwise
two different questions can appear equivalent just because today's values happen
to satisfy both. This deliberately does not claim original-text equivalence.
"""
    if a["init"] != b["init"]:
        raise ValueError("roundtrip_initializations_differ")
    lines = [a["init"], "rt_initial = list(pre_conditions)"]
    lines.extend(c["code"] for c in a["constraints"])
    lines.extend(["rt_a_premises = And(pre_conditions)", "rt_a_queries = {}"])
    for query in a["queries"]:
        lines.extend([query["code"], f"rt_a_queries[{query['item_id']!r}] = check_property"])
    lines.append("pre_conditions = list(rt_initial)")
    lines.extend(c["code"] for c in b["constraints"])
    lines.extend(["rt_b_premises = And(pre_conditions)", "rt_b_queries = {}"])
    for query in b["queries"]:
        lines.extend([query["code"], f"rt_b_queries[{query['item_id']!r}] = check_property"])
    lines.append('''
def rt_difference(left, right):
    solver = Solver()
    solver.set(timeout=10000)
    solver.add(rt_initial)
    solver.add(Xor(left, right))
    status = str(solver.check())
    return {"status": status,
            "counterexample": str(solver.model()) if status == "sat" else None}

rt_domain_solver = Solver()
rt_domain_solver.set(timeout=10000)
rt_domain_solver.add(rt_initial)
rt_domain_status = str(rt_domain_solver.check())
rt_checks = [{"kind": "premises", **rt_difference(rt_a_premises, rt_b_premises)}]
for item_id in rt_a_queries:
    rt_checks.append({"kind": "query", "item_id": item_id,
                      **rt_difference(rt_a_queries[item_id], rt_b_queries[item_id])})
result = {"domain_status": rt_domain_status,
          "initial_conditions": [str(c) for c in rt_initial], "checks": rt_checks,
          "equivalent": rt_domain_status == "sat" and all(c["status"] == "unsat" for c in rt_checks)}
''')
    return "\n".join(lines)


def run(ctx) -> dict:
    prepared = lite.prepare(ctx)
    program, execution, items = prepared["program"], prepared["execution"], prepared["items"]
    diagnostics = {"implementation": "roundtrip_source_check_pilot",
                   "base_status": prepared["status"], "initial_items": items,
                   "repair_attempted": False, "review_status": "not_started"}
    if items is None:
        return lite.finish(ctx, "ssv_roundtrip", program, execution, items, diagnostics)

    # API/budget exceptions propagate and remain counted by the runner.
    try:
        executable = code_only(program)
        backtranslation = lite.ask(ctx, BACKTRANSLATE, executable, "roundtrip_backtranslate")
        _validate_backtranslation(backtranslation, program)
        ctx.record("roundtrip_backtranslation", backtranslation)
        regenerated = lite.ask(ctx, REFORMALIZE,
            {"initialization_code": executable["init"], "backtranslation": backtranslation},
            "roundtrip_reformalize")
        b = {"initial_context": backtranslation["initial_context"], "init": program["init"],
             "constraints": regenerated["constraints"], "queries": regenerated["queries"]}
        ssv._validate(b, [q["item_id"] for q in program["queries"]])
        ctx.record("roundtrip_program_b", b)
        b_execution, b_items = lite.execute_program(ctx, b, "roundtrip_program_b")
        equivalent = ctx.execute(equivalence_code(program, b), stage="roundtrip_equivalence")
        ctx.record("roundtrip_equivalence", equivalent)
        diagnostics.update({"roundtrip_items": b_items,
                            "roundtrip_equivalence": equivalent.get("result") if equivalent.get("ok") else None,
                            "roundtrip_execution_ok": b_execution.get("ok", False)})
        diagnosis = lite.ask(ctx, DIAGNOSE,
            {"original_problem": lite.public_problem(ctx), "program_a": program,
             "backtranslation": backtranslation, "program_b": b,
             "equivalence_check": equivalent}, "roundtrip_diagnose")
        if not all(isinstance(diagnosis.get(k), list) for k in
                   ("source_drift", "roundtrip_errors", "undetermined")):
            raise ValueError("roundtrip_bad_diagnosis")
        if any(not isinstance(d, dict) or not all(isinstance(d.get(k), str) for k in
               ("entry_id", "source_quote", "reason", "correction")) for d in diagnosis["source_drift"]):
            raise ValueError("roundtrip_bad_source_drift")
        ctx.record("roundtrip_diagnosis", diagnosis)
        diagnostics.update({"review_status": "completed", "diagnosis": diagnosis})
        if diagnosis["source_drift"]:
            diagnostics["repair_attempted"] = True
            repaired, repaired_execution, repaired_items = lite.repair(ctx, program,
                {"source_drift": diagnosis["source_drift"],
                 "instruction": "Repair only the localized source defects of A. Preserve unaffected meanings."},
                "roundtrip_source_repair")
            diagnostics["repaired_items"] = repaired_items
            diagnostics["repair_applied"] = repaired_items is not None
            program, execution, items = repaired, repaired_execution, repaired_items
    except (ValueError, KeyError, TypeError, SyntaxError) as error:
        diagnostics.update({"review_status": "output_error", "review_error": str(error)})
        ctx.record("roundtrip_output_error", {"type": type(error).__name__, "error": str(error)})
        items = None
    return lite.finish(ctx, "ssv_roundtrip", program, execution, items, diagnostics)
