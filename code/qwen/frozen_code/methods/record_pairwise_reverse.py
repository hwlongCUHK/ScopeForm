"""Order-sensitivity diagnostic; the same method with A/B interchanged."""
from methods.record_pairwise import run_order


def run(ctx):
    return run_order(ctx, reverse=True)
