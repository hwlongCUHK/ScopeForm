"""Finite scoped expressions and scored binding selection, independent of gold.

Selection sees only candidate/factor scores. Evidence satisfiability and query
answers are computed afterwards, and cannot change the selected interpretation.
This module also runs inside the existing isolated solver worker.
"""
from z3 import (And, Or, Not, Implies, If, Sum, Bool, Int, Real, String,
                BoolVal, RealVal, StringVal, Solver, Optimize, ToReal, is_int, sat, unsat)


def select(data, mode="joint"):
    choices = data.get("choices", [])
    ids = [c["id"] for c in choices]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate_choice_id")
    by_id = {c["id"]: c for c in choices}
    candidates = {}
    for choice in choices:
        rows = choice["candidates"]
        if not rows or len({r["id"] for r in rows}) != len(rows):
            raise ValueError("empty_or_duplicate_candidates")
        candidates[choice["id"]] = {r["id"]: i for i, r in enumerate(rows)}
    for factor in data.get("factors", []):
        if factor.get("relation") == "same_scope":
            if not factor["choices"] or any(k not in by_id for k in factor["choices"]):
                raise ValueError("invalid_shared_scope_factor")
            continue
        if not factor["when"]:
            raise ValueError("empty_factor")
        for key, candidate in factor["when"].items():
            if key not in candidates or candidate not in candidates[key]:
                raise ValueError("unknown_factor_assignment")
    local = {c["id"]: max(c["candidates"], key=lambda r: r["score"])["id"] for c in choices}
    if mode == "local" or not choices:
        assignments = dict(local)
    elif mode == "joint":
        optimizer = Optimize()
        optimizer.set(timeout=5000)
        variables = {c["id"]: Int("choice_" + str(i)) for i, c in enumerate(choices)}
        terms = []
        for c in choices:
            variable = variables[c["id"]]
            optimizer.add(variable >= 0, variable < len(c["candidates"]))
            terms.extend(If(variable == i, RealVal(str(r["score"])), RealVal(0))
                         for i, r in enumerate(c["candidates"]))
        for f in data.get("factors", []):
            if f.get("relation") == "same_scope":
                keys = f["choices"]
                def chosen_scope(key):
                    return Sum(*[If(variables[key] == i, scope_numbers[r["value"]], 0)
                                 for i, r in enumerate(by_id[key]["candidates"])])
                scope_numbers = {s["id"]: i + 1 for i, s in enumerate(data["scopes"])}
                active = And(*[chosen_scope(k) == chosen_scope(keys[0]) for k in keys[1:]])
            else:
                active = And(*[variables[k] == candidates[k][v] for k, v in f["when"].items()])
            terms.append(If(active, RealVal(str(f["score"])), RealVal(0)))
        optimizer.maximize(Sum(*terms))
        # Identical scores have a stable first-candidate tie break in both arms.
        for c in choices:
            optimizer.minimize(variables[c["id"]])
        if optimizer.check() != sat:
            raise ValueError("binding_optimization_incomplete")
        model = optimizer.model()
        assignments = {c["id"]: c["candidates"][model.eval(variables[c["id"]]).as_long()]["id"]
                       for c in choices}
    else:
        raise ValueError("unknown_selection_mode")
    unary = sum(by_id[k]["candidates"][candidates[k][v]]["score"] for k, v in assignments.items())
    def factor_active(f):
        if f.get("relation") == "same_scope":
            values = [by_id[k]["candidates"][candidates[k][assignments[k]]]["value"] for k in f["choices"]]
            return all(v == values[0] for v in values[1:])
        return all(assignments[k] == v for k, v in f["when"].items())
    active_factors = [f for f in data.get("factors", []) if factor_active(f)]
    coupling = sum(f["score"] for f in active_factors)
    return {"assignments": assignments, "local_assignments": local,
            "changed_choices": [k for k in assignments if assignments[k] != local[k]],
            "local_score": unary, "factor_score": coupling,
            "objective": unary + (coupling if mode == "joint" else 0),
            "active_factors": [f["id"] for f in active_factors], "mode": mode}


def evaluate(data, assignments):
    scopes = {s["id"] for s in data["scopes"]}
    properties = {p["name"]: p["type"] for p in data["properties"]}
    constructors = {"Bool": Bool, "Int": Int, "Real": Real, "String": String}
    if len(properties) != len(data["properties"]) or not all(t in constructors for t in properties.values()):
        raise ValueError("invalid_property_declarations")
    choices = {c["id"]: {r["id"]: r["value"] for r in c["candidates"]}
               for c in data.get("choices", [])}
    symbols = {}

    def expand(node, trail=()):
        if isinstance(node, list) and node and node[0] == "get":
            return ["var", expand(["choose", node[1]], trail), node[2]]
        if isinstance(node, list) and node and node[0] == "choose":
            key = node[1]
            if key in trail:
                raise ValueError("cyclic_binding")
            return expand(choices[key][assignments[key]], trail + (key,))
        if isinstance(node, list):
            return [expand(part, trail) for part in node]
        return node

    def expression(node):
        if isinstance(node, bool):
            return BoolVal(node)
        if isinstance(node, (int, float)):
            return RealVal(str(node))
        if isinstance(node, str):
            return StringVal(node)
        op, args = node[0], node[1:]
        if op == "var":
            scope, prop = args
            if scope not in scopes or prop not in properties:
                raise ValueError("undeclared_scoped_variable")
            key = (scope, prop)
            if key not in symbols:
                symbols[key] = constructors[properties[prop]]("value_" + str(len(symbols)))
            return symbols[key]
        values = [expression(arg) for arg in args]
        if op == "and":
            return And(*values)
        if op == "or":
            return Or(*values)
        if op == "not":
            return Not(values[0])
        if op == "implies":
            return Implies(*values)
        if op == "ite":
            return If(*values)
        if op == "add":
            return Sum(*values)
        if op == "mul":
            product = RealVal(1)
            for value in values:
                product = product * value
            return product
        a, b = values
        if op == "eq":
            return a == b
        if op == "ne":
            return a != b
        if op == "lt":
            return a < b
        if op == "le":
            return a <= b
        if op == "gt":
            return a > b
        if op == "ge":
            return a >= b
        if op == "sub":
            return a - b
        if op == "div":
            if is_int(a):
                a = ToReal(a)
            if is_int(b):
                b = ToReal(b)
            return a / b
        raise ValueError("unknown_expression_operator: " + str(op))

    solver = Solver()
    solver.set(timeout=10000)
    for fact in data["facts"]:
        if fact["expr"] is not None:
            solver.add(expression(expand(fact["expr"])))
    base_status = str(solver.check())
    result = {"base_status": base_status, "items": [], "queries": [],
              "resolved_facts": [{**f, "expr": expand(f["expr"])} for f in data["facts"]]}
    for query in data["queries"]:
        resolved = expand(query["expr"])
        term = expression(resolved)
        solver.push()
        solver.add(term)
        positive = str(solver.check())
        solver.pop()
        solver.push()
        solver.add(Not(term))
        negative = str(solver.check())
        solver.pop()
        label = None
        if base_status == "sat":
            label = {("sat", "unsat"): "supported", ("unsat", "sat"): "refuted",
                     ("sat", "sat"): "not_determined"}.get((positive, negative))
        result["queries"].append({"item_id": query["item_id"], "expr": resolved,
                                  "positive": positive, "negative": negative})
        if label is not None:
            result["items"].append({"item_id": query["item_id"], "label": label})
    return result


def solve(data, mode="joint"):
    selection = select(data, mode)
    return {"selection": selection, **evaluate(data, selection["assignments"])}
