"""SSV-lite plus one source-to-formula alignment pass and one scoped repair.

Inspired by FaithAlign's context-aware align-then-label formulation
(arXiv:2610.18498), not a reproduction of its mathematical benchmark.
The alignment verdict is a model judgment, never a correctness certificate.
"""
from __future__ import annotations

from methods import ssv_lite as lite


ALIGNMENT_PROMPT = """Compare this segmented Z3 formalization with the original public
task and actual tool observations. Align source phrases to the exact expressions
implementing them, then judge each alignment using its original context.
The program's natural-language descriptions are proposals, not source evidence.
Read the original observations, including the context resolving references.

Cover every query and its material premises. Prioritize scenario/person identity,
historical versus current facts, recommendation/planned versus executed actions,
missing versus explicitly negated information, applicability conditions, units,
numerical dependencies, and whether the query expresses the claim actually asked.
Claims are questions, not facts. A current recommendation does not establish an
actual action; an old observation does not fix a missing current value. Do not
infer outside medical facts or invent assumptions to make a query decidable.

Include a source-only link for a material condition/relation omitted from the
formalization, and a formal-only link for an unsupported added premise or query
condition. Omitting irrelevant prose or representing missing information by an
unconstrained variable is legitimate; do not require a fact for every sentence.
Symbol declarations alone do not encode a numerical or temporal relationship.
Use undetermined when the supplied source cannot settle an interpretation, not
unfaithful merely because a query can have either truth value.

Return JSON {"links":[
 {"source":{"location":"document/turn or observation locator","quote":"exact source phrase"},
  "target":{"entry_id":"init, constraint:K1, or query:C1","fragment":"exact code fragment"},
  "verdict":"faithful|unfaithful|undetermined", "reason":"specific correspondence or defect"}
]}. Either source or target may be null for a missing counterpart, but not both.
Report separable defects as separate links. Do not return S/R/U labels or a
replacement program. Do not give an overall verified certificate.
"""


def run(ctx) -> dict:
    prepared = lite.prepare(ctx)
    program, execution, items = (prepared[k] for k in ("program", "execution", "items"))
    diagnostics = {"adaptation": "source_formula_align_then_label",
                   "verified": False, "initial_items": items,
                   "alignment_attempted": False, "repair_attempted": False}
    if items is None:
        return lite.finish(ctx, "ssv_align", program, execution, items, diagnostics)

    alignment = lite.ask(ctx, ALIGNMENT_PROMPT,
                         {"original_public_problem": lite.public_problem(ctx),
                          "program": program}, "ssv_align_alignment")
    links = alignment["links"]
    if not isinstance(links, list) or any(
        not isinstance(link, dict)
        or link.get("verdict") not in ("faithful", "unfaithful", "undetermined")
        or not (link.get("source") or link.get("target"))
        for link in links
    ):
        raise ValueError("ssv_align_invalid_links")
    ctx.record("ssv_align_alignment", alignment)
    defects = [link for link in links if link["verdict"] == "unfaithful"]
    diagnostics.update(alignment_attempted=True, alignment_links=len(links),
                       unfaithful_links=len(defects),
                       undetermined_links=sum(link["verdict"] == "undetermined" for link in links))
    if defects:
        diagnostics["repair_attempted"] = True
        program, execution, items = lite.repair(ctx, program, {
            "instruction": "Repair only the concrete unfaithful alignments listed below and any "
                           "directly dependent symbols/expressions. Preserve unaffected entries. "
                           "Use the original source to decide the correction; the critique is not "
                           "new evidence. Do not turn undetermined interpretations into facts.",
            "unfaithful_alignments": defects,
        }, "ssv_align_scoped_repair")
    diagnostics["final_items"] = items
    return lite.finish(ctx, "ssv_align", program, execution, items, diagnostics)
