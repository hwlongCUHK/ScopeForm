# Agent and method code

| Component | Entry point | Role |
|---|---|---|
| Native LLM Agent baseline | `baseline/run_eval.py`, `baseline/original.py` | Reads sources and submits labels with the original benchmark tools |
| ScopeForm Agent | `<backbone>/frozen_code/run_methods.py` | Native reading plus formalization, diagnostics/repair, solver and submission |
| Formalization tool | `<backbone>/frozen_code/methods/frame_direct.py` | `solve_and_submit`; no provisional label submission |
| Semantic library | `<backbone>/frozen_code/definitions/frames.json`, `core.json` | Executable definitions and typed argument roles |
| Context binding and solving | `methods/frame_protocol.py`, `frame_context.py`, `frame_library.py`, `record_solver.py` | Source records, shared references and SMT entailment |
| Repair | `methods/frame_repair.py` | Theory-organized diagnosis, ID-addressed edits and error feedback |
| Five-setting harness | `<backbone>/run_five_settings.py` | Identical setting and actual initial checkpoint within each pair |
| Portable launcher | `../scripts/run_experiment.py` | Paths, endpoint, cohort, budget and resume handling |

`<backbone>` is `deepseek` or `qwen`. Method paths in the last rows are relative to the selected `frozen_code/`. These are the Python agents used in the paper experiments. The earlier unrelated application UI and development experiments are not part of this evaluation implementation.

Both frozen snapshots preserve their original dependency modules. The launcher selects the integrated `frame_direct[_guard]_repair` method; it does not enable the old notes agent or historical NL2Logic/SSR/SSV experiments.
