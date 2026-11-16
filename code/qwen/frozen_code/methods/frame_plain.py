"""Same generated frames and formulas, with role context disabled at lowering."""
from methods.frame_context import run_mode


def run(ctx):
    return run_mode(ctx, False)
