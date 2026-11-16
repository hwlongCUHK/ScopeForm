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


class FrameCompileError(ValueError):
    """A compiler lookup/type error with its source-expression location."""

    def __init__(self, code, **details):
        self.diagnostic = {"code": code, **details}
        super().__init__(code + ": " + str(details.get("symbol", details.get("field", ""))))


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
    records = {record["id"]: record for record in result["records"]}
    properties = {prop["name"]: prop for prop in result["properties"]}
    # A directive is a source norm, not a material implication about events.
    # Keep its evidence in the graph, but never assert it as a factual premise.
    directives = {rule["id"]: rule for rule in result.get("rules", []) if "modality" in rule}
    result["directives"] = list(directives.values())
    result["rules"] = [rule for rule in result.get("rules", []) if "modality" not in rule]

    def property_for(rid, slot):
        return references[rid].get("slots", {}).get(slot, slot)

    def record_occurrences(field):
        # This is an inventory, not a binding suggestion or candidate filter.
        # In particular, an unused declaration may have no recorded value.
        return [record["id"] for record in graph["records"] if field in record["fields"]]

    def field_type(field, expected, origin):
        spec = catalog["semantic_types"][expected]
        prop = properties.get(field)
        if field not in properties:
            code = "undeclared_property"
        elif prop.get("semantic_type") != expected or prop["type"] not in spec["types"]:
            code = "frame_field_type_mismatch"
        elif spec.get("unit_required") and not prop.get("unit"):
            code = "frame_quantity_needs_unit"
        elif "unit" in spec and prop.get("unit") != spec["unit"]:
            code = "frame_quantity_unit_mismatch"
        else:
            return prop.get("unit")
        # Produce symbol-table context only on the compiler's existing failure.
        compatible = [{**decl, "record_ids": record_occurrences(name)}
                      for name, decl in properties.items()
                      if decl.get("semantic_type") == expected and decl.get("type") in spec["types"]
                      and (not spec.get("unit_required") or decl.get("unit"))
                      and ("unit" not in spec or decl.get("unit") == spec["unit"])]
        raise FrameCompileError(code, **origin, namespace="properties", symbol=field,
            expected_type={"semantic_type": expected, **spec}, actual_declaration=prop,
            declared_symbols=list(properties), type_compatible_properties=compatible,
            requested_field_record_ids=record_occurrences(field))

    def expand_frame(node, query_id, path):
        frame = frames[node["frame"]]
        bindings = node["roles"]
        if set(bindings) != set(frame["roles"]):
            raise ValueError("frame_roles_differ: " + frame["id"])
        role_refs, units = {}, {}
        for role_index, (role, spec) in enumerate(frame["roles"].items()):
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
            for member_index, rid in enumerate(ids):
                argument_path = path + ["args", role_index]
                if spec.get("many"):
                    argument_path += [member_index]
                origin = {"query_id": query_id, "expression_path": path,
                          "argument_path": argument_path, "frame": frame["id"],
                          "role": role, "reference_id": rid}
                if rid not in references:
                    raise FrameCompileError("undeclared_reference", **origin,
                        namespace="references", symbol=rid, declared_symbols=list(references),
                        found_in_other_namespaces=["records"] if any(r["id"] == rid for r in graph["records"]) else [],
                        expected_role={"where": spec["where"], "fields": spec["fields"]})
                reference = references[rid]
                if use_role_context:
                    for key, required in spec["where"].items():
                        where = reference["where"]
                        if key in where and where[key] != required:
                            raise ValueError("frame_reference_context_conflict: " + rid + "." + key)
                        where[key] = required
                for slot, expected in spec["fields"].items():
                    field = property_for(rid, slot)
                    unit = field_type(field, expected, {**origin, "slot": slot,
                        "slot_mapping_path": {"collection": "references", "id": rid, "path": ["slots", slot]}})
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

    def source_address(query_id, path):
        if path and path[0] == "rules":
            return {"collection": "rules", "id": path[1], "path": path[2:]}
        return {"collection": "queries", "item_id": query_id, "path": path}

    def address_lookup(node, query_id, path):
        # Same name-resolution obligation as record_solver; no field lookup.
        # Undeclared Bool fields can still be inferred by the existing compiler.
        if len(node) < 2 or node[0] not in ("ref", "record") or not isinstance(node[1], str):
            return
        namespace, symbols, other_name, other_symbols = (
            ("references", references, "records", records) if node[0] == "ref" else
            ("records", records, "references", references))
        if node[1] not in symbols:
            raise FrameCompileError("undeclared_reference" if node[0] == "ref" else "undeclared_record",
                namespace=namespace, symbol=node[1], operation=node[0], query_id=query_id,
                expression_path=path, argument_path=path + ["args", 0],
                source_address=source_address(query_id, path + ["args", 0]),
                declared_symbols=list(symbols),
                found_in_other_namespaces=[other_name] if node[1] in other_symbols else [])

    def visit(node, query_id, path, lower_calls=True):
        if isinstance(node, dict):
            if lower_calls and "directive" in node:
                rule = directives[node["directive"]]
                return {"definition": "directive_active", "arguments": {
                    "condition": visit(rule["when"], query_id, ["rules", rule["id"], "when"]),
                    "source_modality": rule["modality"], "queried_modality": node["modality"]}}
            if lower_calls and "frame" in node:
                return expand_frame(node, query_id, path)
            return {key: visit(value, query_id, path + [key], lower_calls) for key, value in node.items()}
        if isinstance(node, list):
            # frame_protocol.decode represents an op/args call as [op, *args].
            # Report paths in the model-facing syntax, not the lowered array.
            if node and isinstance(node[0], str):
                address_lookup(node, query_id, path)
                return [node[0]] + [visit(child, query_id, path + ["args", index], lower_calls)
                                   for index, child in enumerate(node[1:])]
            return [visit(child, query_id, path + [index], lower_calls) for index, child in enumerate(node)]
        return node

    for rule in result["rules"]:
        # Factual source rules are not frame-expanded. Only expose their
        # existing identifier failures with the truthful source-rule address.
        visit(rule["expr"], None, ["rules", rule["id"], "expr"], lower_calls=False)
    for query in result["queries"]:
        query["expr"] = definition_library.expand(visit(query["expr"], query["item_id"], ["expr"]))
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
