"""Same immutable records and candidate domains, independent field selection."""
from methods.record_shared import run_mode


def run(ctx):
    return run_mode(ctx, "fields")
