"""Immutable source records, mechanically enumerated references, late binding.

Both ablations use identical candidate domains and per-field scores. Sharing a
reference chooses one complete record; it never relocates source facts. This
module performs no model calls and never inspects a query's truth value.
"""

SELECTORS = {"kind", "entity", "subject", "target", "time", "modality", "scenario", "document_id", "anchor_id"}


def reference_fields(expr, fields):
    if not isinstance(expr, list):
        return
    if expr[0] == "ref":
        fields.setdefault(expr[1], set()).add(expr[2])
        return
    for child in expr[1:]:
        reference_fields(child, fields)


def select(data, mode="shared"):
    if mode not in ("fields", "shared"):
        raise ValueError("unknown_record_selection_mode")
    records = {r["id"]: r for r in data["records"]}
    refs = {r["id"]: r for r in data["references"]}
    if len(records) != len(data["records"]) or len(refs) != len(data["references"]):
        raise ValueError("duplicate_record_or_reference_id")
    fields = {}
    for query in data["queries"]:
        reference_fields(query["expr"], fields)
    domains, independent, assignments, ties, empty = {}, {}, {}, [], []
    per_reference = {}
    scores_used = {}
    for rid, needed in fields.items():
        ref = refs[rid]
        where = ref["where"]
        if not set(where) <= SELECTORS:
            raise ValueError("reference_selector_must_be_identity_not_query_criterion")
        # Missing values do not disqualify a record. Only explicit reference
        # descriptors filter the domain, never a dose/outcome/answer condition.
        # The optional target selector reads the same identity that queries read;
        # it never copies another role's target or creates a second identity field.
        identity_field = ref.get("slots", {}).get("object_id", "object_id")
        domain = [key for key, record in records.items()
                  if all((record["fields"].get(identity_field) if k == "target" else record.get(k)) == v
                         for k, v in where.items())]
        domains[rid] = domain
        if not domain:
            empty.append(rid)
            continue
        field_scores = {field: {key: ref.get("scores", {}).get(field, {}).get(key, 0)
                                 for key in domain} for field in sorted(needed)}
        scores_used[rid] = field_scores
        for field in sorted(needed):
            independent[rid + "." + field] = max(domain, key=field_scores[field].get)
        # Count each referenced field once, irrespective of repeated AST uses.
        total = {key: sum(field_scores[f][key] for f in sorted(needed)) for key in domain}
        selected = max(domain, key=total.get)
        if sum(value == total[selected] for value in total.values()) > 1:
            ties.append(rid)
        per_reference[rid] = selected
        for field in sorted(needed):
            address = rid + "." + field
            assignments[address] = selected if mode == "shared" else independent[address]
    mixed = [rid for rid, needed in fields.items() if rid not in empty
             and len({assignments[rid + "." + f] for f in needed}) > 1]
    return {"mode": mode, "assignments": assignments, "field_assignments": assignments,
            "independent_assignments": independent, "record_assignments": per_reference,
            "changed_fields": [a for a in assignments if assignments[a] != independent[a]],
            "candidate_domains": domains, "mixed_references": mixed,
            "tied_references": ties, "empty_domains": empty,
            "field_scores": scores_used}


def compile_program(data, selection):
    """Translate source fields/rules and the selected query into the fixed AST."""
    if selection["empty_domains"]:
        raise ValueError("unresolved_record_reference: " + ",".join(selection["empty_domains"]))
    records = {r["id"]: r for r in data["records"]}
    assignments = selection["assignments"]

    def resolve(expr, evidence=False):
        if not isinstance(expr, list):
            return expr
        if expr[0] == "record":
            # An explicit source address is already bound, in either a rule
            # or a query. Only ambiguous/shared references need selection.
            if expr[1] not in records:
                raise ValueError("unknown_fixed_record")
            return ["var", expr[1], expr[2]]
        if expr[0] == "ref":
            if evidence:
                raise ValueError("query_reference_cannot_own_source_evidence")
            return ["var", assignments[expr[1] + "." + expr[2]], expr[2]]
        if expr[0] in ("var", "get", "choose"):
            raise ValueError("use_record_or_ref_addressing")
        return [expr[0]] + [resolve(child, evidence) for child in expr[1:]]

    facts = []
    for record in data["records"]:
        for name, value in record["fields"].items():
            if isinstance(value, (dict, list)):
                raise ValueError("record_fields_are_literal_or_null")
            if value is not None:
                facts.append({"id": record["id"] + "." + name,
                              "text": record["id"] + "." + name,
                              "expr": ["eq", ["var", record["id"], name], value],
                              "sources": record.get("sources", [])})
    for rule in data.get("rules", []):
        facts.append({**rule, "expr": resolve(rule["expr"], evidence=True)})
    queries = [{**q, "expr": resolve(q["expr"])} for q in data["queries"]]
    properties = list(data["properties"])
    declared = {p["name"] for p in properties}

    def boolean_types(expr, boolean_position):
        """A logical operand's Bool type follows from syntax, not its truth."""
        if not isinstance(expr, list):
            return
        op, args = expr[0], expr[1:]
        if op == "var":
            field = args[1]
            if boolean_position and field not in declared:
                properties.append({"name": field, "type": "Bool"})
                declared.add(field)
            return
        for index, child in enumerate(args):
            required = op in ("and", "or", "not", "implies")
            if op == "ite":
                required = index == 0 or boolean_position
            boolean_types(child, required)

    for row in facts + queries:
        boolean_types(row["expr"], True)
    return {"scopes": [{"id": r["id"]} for r in data["records"]],
            "properties": properties, "facts": facts,
            "queries": queries,
            "choices": [], "factors": []}
