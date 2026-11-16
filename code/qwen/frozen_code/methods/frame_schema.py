"""Executable JSON grammar for the existing external formalization language.

This constrains representation only. Source truth, scope and entailment remain
the responsibilities of formalization, diagnosis and the unchanged compiler.
The definitions live at the function-parameter root because vLLM hoists them
when combining selectable tools for required function calling.
"""
from methods import frame_protocol, record_solver


def _object(properties, required=(), additional=False):
    return {"type": "object", "properties": properties, "required": list(required),
            "additionalProperties": additional}


def parameters():
    string = {"type": "string"}
    expr = {"$ref": "#/$defs/scopeform_expression"}
    literal = {"type": ["string", "number", "boolean", "null"]}
    source = _object({"location": string, "quote": string,
                      "document_id": string, "anchor_id": string}, additional=True)
    sources = {"type": "array", "items": source}

    def call(op, arguments=None, variadic=False):
        if variadic:
            args = {"type": "array", "items": expr}
        else:
            args = {"type": "array", "prefixItems": arguments,
                    "minItems": len(arguments), "maxItems": len(arguments), "items": False}
        return _object({"op": {"const": op}, "args": args}, ("op", "args"))

    alternatives = [{"type": ["string", "number", "boolean"]}]
    for op, signature in frame_protocol.PRIMITIVES.items():
        arguments = [string, string] if op in ("ref", "record") else [expr] * len(signature)
        alternatives.append(call(op, arguments, "..." in signature))
    catalog = frame_protocol.catalog_for_prompt()
    for frame in catalog["frames"]:
        arguments = [{"type": "array", "items": string} if role.get("many") else string
                     for role in frame["roles"].values()]
        alternatives.append(call(frame["op"], arguments))
    directive = catalog["directives"][0]
    alternatives.append(call("directive_applies", [string, {"enum": directive["modalities"]}]))

    property_schema = _object({
        "name": string, "type": {"enum": ["Bool", "Int", "Real", "String"]},
        "semantic_type": {"enum": ["Identity", "Dose", "Frequency", "Completion"]},
        "unit": string}, ("name", "type"))
    record = _object({
        "id": string, "kind": {"enum": ["regimen", "event", "state"]},
        "subject": string, "entity": string, "time": string,
        "modality": {"enum": ["actual", "recommendation", "plan", "order"]},
        "scenario": string, "document_id": string, "anchor_id": string,
        "fields": {"type": "object", "additionalProperties": literal}, "sources": sources,
    }, ("id", "fields"))
    reference = _object({
        "id": string, "text": string,
        "where": _object({name: string for name in sorted(record_solver.SELECTORS)}),
        "slots": {"type": "object", "additionalProperties": string},
        "scores": {"type": "object", "additionalProperties": {"type": "number"}},
    }, ("id", "where"))
    fact_rule = _object({"id": string, "text": string, "expr": expr, "sources": sources},
                        ("id", "expr"))
    directive_rule = _object({
        "id": string, "text": string, "modality": {"enum": directive["modalities"]},
        "action": string, "when": expr, "sources": sources,
    }, ("id", "modality", "action", "when"))
    query = _object({"item_id": string, "text": string, "expr": expr,
                     "evidence_ids": {"type": "array", "items": string}},
                    ("item_id", "text", "expr", "evidence_ids"))
    rows = {"properties": property_schema, "records": record, "references": reference,
            "rules": {"anyOf": [fact_rule, directive_rule]}, "queries": query}
    graph = _object({key: {"type": "array", "items": value} for key, value in rows.items()}, rows)
    result = _object({"graph": graph}, ("graph",))
    result["$defs"] = {"scopeform_expression": {"anyOf": alternatives}}
    return result


def graph_schema():
    """The same graph grammar as a direct JSON response, without a tool wrapper."""
    wrapped = parameters()
    return {**wrapped["properties"]["graph"], "$defs": wrapped["$defs"]}
