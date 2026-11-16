"""Same scoped candidate graph, with independent highest-score selection."""
from methods.scope_joint import run_mode


def run(ctx):
    return run_mode(ctx, "local")
