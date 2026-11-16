# Release validation

The package was extracted into a separate directory and executed with a new Python 3.10 environment installed from `requirements.txt`. No model API calls were made.

- `offline_reproduction.json`: all 9,840 phases re-scored; 60 manuscript numeric cells, both paired-bootstrap analyses, actual initial checkpoints and repair-audit statistics matched.
- `recorded_case_replay.json`: the original baseline and ScopeForm episode were re-executed through the full agent/runtime/solver using actual saved model responses. All nine generated requests and the final outputs matched.
- `subset_scoring.json`: the new-run scoring entry point was exercised on a recorded 9-phase subset.
- `deepseek_solver_demo.json` / `qwen_solver_demo.json`: both frozen solvers executed the illustrative context-binding, repair, and unknown-value cases.
- Both launchers were imported/prepared from the extracted package; no provider request was sent. Live API availability and a new GPU service were not tested during packaging.

The manifest referenced by the offline validation was the validation archive's manifest. The final release manifest also includes these newly generated validation documents; method code and experimental artifacts are unchanged.
