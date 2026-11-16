"""Local execution boundary for generated Z3 expressions, without I/O APIs."""
import ast
import builtins
import contextlib
import io
import json
import traceback
import sys


def main():
    code = json.load(sys.stdin)["code"]
    output = io.StringIO()
    try:
        tree = ast.parse(code)
        modules = {"z3", "math", "itertools", "functools", "collections", "fractions", "decimal"}
        denied = {"open", "eval", "exec", "compile", "input", "globals", "locals", "vars",
                  "getattr", "setattr", "delattr", "breakpoint", "help"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(a.name.split(".")[0] not in modules for a in node.names):
                    raise ValueError("solver_import_not_allowed")
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] not in modules:
                raise ValueError("solver_import_not_allowed")
            if isinstance(node, ast.Name) and (node.id in denied or node.id.startswith("__")):
                raise ValueError("solver_io_or_introspection_not_allowed")
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise ValueError("solver_introspection_not_allowed")
        allowed = {k: v for k, v in vars(builtins).items()
                   if k not in denied and not k.startswith("__")}
        def importer(name, globals=None, locals=None, fromlist=(), level=0):
            if name.split(".")[0] not in modules or level:
                raise ImportError("solver_import_not_allowed")
            return builtins.__import__(name, globals, locals, fromlist, level)
        allowed["__import__"] = importer
        scope = {"__builtins__": allowed, "__name__": "cpm_generated_solver"}
        with contextlib.redirect_stdout(output):
            exec(compile(tree, "<generated-z3>", "exec"), scope)
        print(json.dumps({"ok": True, "result": scope.get("result"),
                          "stdout": output.getvalue(), "stderr": ""}, default=str))
    except Exception:
        print(json.dumps({"ok": False, "result": None, "stdout": output.getvalue(),
                          "stderr": traceback.format_exc()}))


if __name__ == "__main__":
    main()
