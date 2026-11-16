"""Expand the declarative semantic catalog into the existing record AST.

No model calls, record selection or facts are added here. Parameter types and
semantic roles live in the catalog; binding supplies existing typed references.
"""
from copy import deepcopy
import json
from pathlib import Path

CATALOG = Path(__file__).resolve().parents[1] / "definitions/core.json"


def load_catalog():
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def expand(expression):
    """Expand nested {definition, arguments} calls, retaining ref/record leaves."""
    definitions = {d["id"]: d for d in load_catalog()["definitions"]}

    def instantiate(template, arguments):
        if isinstance(template, dict):
            if "param" in template:
                return deepcopy(arguments[template["param"]])
            return {k: instantiate(v, arguments) for k, v in template.items()}
        if isinstance(template, list):
            result = []
            for child in template:
                if isinstance(child, dict) and "spread" in child:
                    result.extend(deepcopy(arguments[child["spread"]]))
                elif isinstance(child, dict) and "each" in child:
                    result.extend(instantiate(child["body"], {**arguments, child["as"]: value})
                                  for value in arguments[child["each"]])
                else:
                    result.append(instantiate(child, arguments))
            return result
        return template

    def visit(node):
        if isinstance(node, dict):
            definition = definitions[node["definition"]]
            arguments = {k: visit(v) for k, v in node["arguments"].items()}
            if set(arguments) != set(definition["parameters"]):
                raise ValueError("definition_arguments_differ: " + definition["id"])
            for key, parameter in definition["parameters"].items():
                if "values" in parameter and arguments[key] not in parameter["values"]:
                    raise ValueError("invalid_definition_enum: " + key)
            return visit(instantiate(definition["expansion"], arguments))
        if isinstance(node, list):
            return [visit(part) for part in node]
        if node is None:
            raise ValueError("unknown_values_belong_in_record_fields_bind_a_ref")
        return node

    return visit(expression)


def expand_queries(graph):
    """Return a graph ready for record_solver.select/compile_program.

    Source records, source rules and reference descriptions stay as supplied.
    Calling this function does not activate a new experiment arm.
    """
    result = deepcopy(graph)
    for query in result["queries"]:
        query["expr"] = expand(query["expr"])
    return result


if __name__ == "__main__":
    import sys
    from methods import record_solver, scope_solver

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else CATALOG.with_name("example.json")
    graph = expand_queries(json.loads(path.read_text(encoding="utf-8")))
    selection = record_solver.select(graph, mode="shared")
    compiled = record_solver.compile_program(graph, selection)
    result = scope_solver.evaluate(compiled, {})
    print(json.dumps({"base_status": result["base_status"], "items": result["items"]},
                     ensure_ascii=False, indent=2))
