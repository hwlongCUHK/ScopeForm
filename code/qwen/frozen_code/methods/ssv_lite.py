"""Small CPM-specific SSV ablation: one formalization and the original solver.

No instance generation, temperature search, compositional generation or extra
direct-answer fallback. A program that cannot execute gets one repair attempt.
This is an adaptation, not the full published SSV algorithm.
"""
from __future__ import annotations

import copy
import json

from methods import ssv


FORMAT = ssv.PROGRAM_FORMAT + """
CPM adaptation:
- init declares symbols and general domains ONLY. Put observed values, source
  recommendations and applicable rules in constraints, never also in init.
- Missing information means a declared but unconstrained value. Do not create
  a constraint whose code is True merely to represent a missing observation.
- Keep actual observed events, recommendations, historical facts, and separate
  current scenarios distinct. Preserve units and source conditions in queries.
- Each constraint may include sources:[{observation_index:int, location:string,
  quote:string}], pointing to the supplied original observation, not a rewritten
  explanation. Use the original claim wording in query text.
- Describe the entire requested claim, including temporal, identity, execution
  and applicability requirements. Claims themselves are never evidence.
"""


def public_problem(ctx):
    return ssv._public_problem(ctx.problem)


def ask(ctx, prompt, data, stage):
    raw = ctx.llm(
        [{"role": "system", "content": ssv.RULES},
         {"role": "user", "content": prompt + "\n" + json.dumps(data, ensure_ascii=False)}],
        stage=stage, temperature=0, json_mode=True,
    )
    ctx.record(stage + "_response", {"raw": raw})
    return ssv._json(raw)


def execute_program(ctx, program, stage):
    try:
        ssv._validate(program, ssv._claim_ids(ctx.problem))
    except (ValueError, TypeError, KeyError) as error:
        execution = {"ok": False, "error": str(error), "result": None}
        ctx.record(stage + "_execution", execution)
        return execution, None
    ctx.record(stage + "_program", program)
    execution = ssv._execute(ctx, program, stage)
    return execution, ssv._complete_answer(execution, program)


def prepare(ctx):
    program, items = None, None
    try:
        program = ask(ctx, "Translate the acquired public evidence and claims into one Z3 program.\n" + FORMAT,
                      public_problem(ctx), "ssv_lite_generate")
        execution, items = execute_program(ctx, program, "ssv_lite_initial")
    except (ValueError, TypeError, KeyError) as error:
        execution = {"ok": False, "error": str(error), "result": None}
    # Repair only execution/format failures or inconsistent/unknown solver
    # results, not valid evidence-level U answers.
    if items is None:
        feedback = {key: value for key, value in execution.items() if key != "seconds"}
        try:
            program = ask(ctx, "Repair this program once so it executes with consistent evidence. "
                          "Correct a translation or code error; do not invent facts or discard true source facts "
                          "to force an answer. Return the complete program.\n" + FORMAT,
                          {"problem": public_problem(ctx), "program": program, "execution": feedback},
                          "ssv_lite_execution_repair")
            execution, items = execute_program(ctx, program, "ssv_lite_repaired")
        except (ValueError, TypeError, KeyError) as error:
            execution = {"ok": False, "error": str(error), "result": None}
    result = {"program": program, "execution": execution, "items": items,
              "status": "completed" if items is not None else "failed"}
    ctx.record("ssv_lite_prepared", {**result,
        "logical_requests": getattr(ctx, "logical_index", None),
        "requests": getattr(ctx, "requests", None),
        "executions": getattr(ctx, "executions", None)})
    return result


def repair(ctx, program, feedback, stage):
    patch = ask(ctx, "Repair only the source-to-formula defects identified below. Keep unaffected "
                "entries unchanged. Return a JSON patch: {init:null or corrected initialization string, "
                "replace_constraints:[{id,text,code,sources}], remove_constraint_ids:[id], "
                "replace_queries:[{item_id,text,code,sources}]}. A new constraint id adds an entry. "
                "Use init only if shared declarations/domain setup must change; no observed facts in init. "
                "Retain every original claim id. Do not put the desired answer in the premises. "
                "Constraint patches append a Z3 expression to pre_conditions; query patches assign "
                "check_property. Entry fields match the original program. Return ONLY the patch "
                "fields above, not a complete program. Omit entries that do not need changes.",
                {"problem": public_problem(ctx), "program": program, "feedback": feedback}, stage)
    if "constraints" in patch or "queries" in patch:
        raise ValueError("ssv_lite_expected_patch_not_complete_program")
    revised = copy.deepcopy(program)
    if patch.get("init") is not None:
        revised["init"] = patch["init"]
    removed = set(patch.get("remove_constraint_ids", []))
    constraints = {row["id"]: row for row in revised["constraints"] if row["id"] not in removed}
    constraints.update({row["id"]: row for row in patch.get("replace_constraints", [])})
    revised["constraints"] = list(constraints.values())
    queries = {row["item_id"]: row for row in revised["queries"]}
    queries.update({row["item_id"]: row for row in patch.get("replace_queries", [])})
    revised["queries"] = list(queries.values())
    ctx.record(stage + "_patch", patch)
    execution, items = execute_program(ctx, revised, stage)
    return revised, execution, items


def finish(ctx, method, program, execution, items, diagnostics):
    ctx.record(method + "_selected_program", program)
    result = {"method": method, "status": "completed" if items is not None else "failed",
              "items": items if items is not None else [],
              "diagnostics": {**diagnostics, "verified": False,
                  "solver_base_status": (execution.get("result") or {}).get("base_status"),
                  "implementation": "CPM_small_ablation"}}
    ctx.record("result", result)
    return result


def run(ctx):
    prepared = prepare(ctx)
    return finish(ctx, "ssv_lite", prepared["program"], prepared["execution"],
                  prepared["items"], {"temperature": 0, "instance_validation": False,
                                      "temperature_search": False})
