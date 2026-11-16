"""Lower role-bearing frames to the existing immutable-record language.

The catalog supplies role context and typed field access; the source supplies
values and reference descriptors. All fields of a reference share one record.
No model calls or truth-dependent selection happen in this module.
"""
from copy import deepcopy
import json
from pathlib import Path

from methods import definition_library, record_solver, scope_solver

CATALOG = Path(__file__).resolve().parents[1] / "definitions/frames.json"


def load_catalog():
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def lower(graph, use_role_context=True):
    """Expand frames/core calls and refine their existing references in a copy.

    `False` is the ablation: identical fields and formulas, without the frame's
    kind/modality requirements. Explicit reference descriptors are always kept.
    Missing field values are never used to filter a candidate record.
    """
    catalog = load_catalog()
    frames = {frame["id"]: frame for frame in catalog["frames"]}
    result = deepcopy(graph)
    references = {ref["id"]: ref for ref in result["references"]}
    properties = {prop["name"]: prop for prop in result["properties"]}
    # A directive is a source norm, not a material implication about events.
    # Keep its evidence in the graph, but never assert it as a factual premise.
    directives = {rule["id"]: rule for rule in result.get("rules", []) if "modality" in rule}
    result["directives"] = list(directives.values())
    result["rules"] = [rule for rule in result.get("rules", []) if "modality" not in rule]

    def property_for(rid, slot):
        return references[rid].get("slots", {}).get(slot, slot)

    def field_type(field, expected):
        prop = properties[field]
        spec = catalog["semantic_types"][expected]
        if prop.get("semantic_type") != expected or prop["type"] not in spec["types"]:
            raise ValueError("frame_field_type_mismatch: " + field)
        if spec.get("unit_required") and not prop.get("unit"):
            raise ValueError("frame_quantity_needs_unit: " + field)
        if "unit" in spec and prop.get("unit") != spec["unit"]:
            raise ValueError("frame_quantity_unit_mismatch: " + field)
        return prop.get("unit")

    def expand_frame(node):
        frame = frames[node["frame"]]
        bindings = node["roles"]
        if set(bindings) != set(frame["roles"]):
            raise ValueError("frame_roles_differ: " + frame["id"])
        role_refs, units = {}, {}
        for role, spec in frame["roles"].items():
            value = bindings[role]
            if spec.get("many"):
                if not isinstance(value, list):
                    raise ValueError("frame_role_needs_reference_list: " + role)
                ids = value
            else:
                if not isinstance(value, str):
                    raise ValueError("frame_role_needs_reference: " + role)
                ids = [value]
            role_refs[role] = ids
            for rid in ids:
                reference = references[rid]
                if use_role_context:
                    for key, required in spec["where"].items():
                        where = reference["where"]
                        if key in where and where[key] != required:
                            raise ValueError("frame_reference_context_conflict: " + rid + "." + key)
                        where[key] = required
                for slot, expected in spec["fields"].items():
                    unit = field_type(property_for(rid, slot), expected)
                    if unit is not None:
                        units.setdefault(expected, set()).add(unit)
        if any(len(values) > 1 for values in units.values()):
            raise ValueError("frame_comparison_units_differ: " + frame["id"])

        def instantiate(template):
            if isinstance(template, dict):
                if "field" in template:
                    role, slot = template["field"]
                    rid = role_refs[role][0]
                    return ["ref", rid, property_for(rid, slot)]
                if "fields" in template:
                    role, slot = template["fields"]
                    return [["ref", rid, property_for(rid, slot)] for rid in role_refs[role]]
                return {key: instantiate(value) for key, value in template.items()}
            if isinstance(template, list):
                return [instantiate(child) for child in template]
            return template

        return instantiate(frame["body"])

    def visit(node):
        if isinstance(node, dict):
            if "directive" in node:
                rule = directives[node["directive"]]
                return {"definition": "directive_active", "arguments": {
                    "condition": visit(rule["when"]), "source_modality": rule["modality"],
                    "queried_modality": node["modality"]}}
            if "frame" in node:
                return expand_frame(node)
            return {key: visit(value) for key, value in node.items()}
        if isinstance(node, list):
            return [visit(child) for child in node]
        return node

    for query in result["queries"]:
        query["expr"] = definition_library.expand(visit(query["expr"]))
    return result


def solve(graph, use_role_context=True):
    """Run the lowered graph through the unchanged shared-record/Z3 pipeline."""
    lowered = lower(graph, use_role_context=use_role_context)
    selection = record_solver.select(lowered, mode="shared")
    compiled = record_solver.compile_program(lowered, selection)
    return {"graph": lowered, "selection": selection, "compiled": compiled,
            **scope_solver.evaluate(compiled, {})}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=CATALOG.with_name("frame_example.json"))
    parser.add_argument("--without-role-context", action="store_true", help="same formulas without frame kind/modality binding")
    args = parser.parse_args()
    graph = json.loads(args.path.read_text(encoding="utf-8"))
    outcome = solve(graph, use_role_context=not args.without_role_context)
    print(json.dumps({"base_status": outcome["base_status"],
                      "candidate_domains": outcome["selection"]["candidate_domains"],
                      "record_assignments": outcome["selection"]["record_assignments"],
                      "items": outcome["items"]}, ensure_ascii=False, indent=2))
