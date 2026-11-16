"""Constrain the repair envelope, leaving each proposed value to the model.

This is the existing ID-edit interface as an executable output grammar. It
does not fill missing fields, infer evidence, select an edit or validate its
semantics. Every generation is charged through the phase's existing Context.
"""
import json
from pathlib import Path


TARGET_KEYS = {"properties": "name", "records": "id", "references": "id",
               "rules": "id", "queries": "item_id"}


def annotation_hole(feedback):
    """Locate an undeclared field already present in source records.

    The compiler supplies the address and expected type. An absent source
    field or an unresolved reference is not an annotation hole.
    """
    for diagnostic in (feedback or {}).get("compiler_diagnostics", []):
        expected = diagnostic.get("expected_type", {})
        if (diagnostic.get("code") == "undeclared_property"
                and diagnostic.get("namespace") == "properties"
                and diagnostic.get("requested_field_record_ids")
                and diagnostic.get("symbol")
                and expected.get("semantic_type") and expected.get("types")):
            return diagnostic
    return None


def schema(feedback=None):
    taxonomy = json.loads((Path(__file__).resolve().parents[1] /
                           "definitions/error_taxonomy.json").read_text(encoding="utf-8"))
    categories = [row["id"] for row in taxonomy["categories"]]
    text = {"type": "string"}
    identifier = {"type": "string", "minLength": 1}
    value = {"$ref": "#/$defs/repair_json_value"}

    def obj(properties):
        return {"type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False}

    edits = [obj({
        "collection": {"const": collection}, key: identifier,
        "path": {"type": "array", "items": {"type": ["string", "integer"]}},
        "value": value,
    }) for collection, key in TARGET_KEYS.items()]
    evidence = obj({"document_id": text, "anchor_id": text, "quote": text})
    issue = obj({
        "category": {"enum": categories}, "reason": {"type": "string", "minLength": 1},
        "evidence": {"type": "array", "items": evidence},
        "edits": {"type": "array", "items": {"anyOf": edits}, "minItems": 1},
    })
    result = obj({"issues": {"type": "array", "items": issue}})
    result["$defs"] = {"repair_json_value": {"anyOf": [
        {"type": ["string", "number", "boolean", "null"]},
        {"type": "array", "items": value},
        {"type": "object", "additionalProperties": value},
    ]}}
    hole = annotation_hole(feedback)
    if hole is not None:
        expected, name = hole["expected_type"], hole["symbol"]
        declaration = {"name": {"const": name}, "type": {"enum": expected["types"]},
                       "semantic_type": {"const": expected["semantic_type"]}}
        if "unit" in expected:
            declaration["unit"] = {"const": expected["unit"]}
        elif expected.get("unit_required"):
            declaration["unit"] = {"type": "string", "minLength": 1}
        edit = obj({"collection": {"const": "properties"}, "name": {"const": name},
                    "path": {"const": []}, "value": obj(declaration)})
        issue = obj({"category": {"const": "representation_validity"},
                     "reason": {"type": "string", "minLength": 1},
                     "evidence": {"type": "array", "items": evidence},
                     "edits": {"type": "array", "items": edit, "minItems": 1, "maxItems": 1}})
        result = obj({"issues": {"type": "array", "items": issue,
                                  "minItems": 1, "maxItems": 1}})
    return result


def generate(ctx, messages, stage, feedback=None):
    request = {
        "model": ctx.model, "messages": messages, "temperature": 0,
        "thinking": {"type": "disabled"}, "stream": False,
        "max_tokens": min(ctx.settings["max_output_tokens"], 8192),
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "scopeform_repair", "strict": True, "schema": schema(feedback)}},
    }
    response = ctx.chat(request, stage=stage)
    content = response["choices"][0]["message"].get("content")
    if not isinstance(content, str):
        raise RuntimeError("method_empty_assistant_content")
    return content
