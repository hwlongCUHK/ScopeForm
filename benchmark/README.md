# CPM-Sandbox main-cohort benchmark

120 source clusters (72 consultation/dialogue sources, 48 drug sources), 519 initial claims, four update events per source. The bundled original runtime supports the five settings evaluated in ScopeForm.

- `medical_scale_v06/confirmation15_inputs_private/actors.private.jsonl`: actor inputs, source documents, claims, event controllers.
- `medical_scale_v06/study15_protocol.json`: protocol.
- `medical_scale_v06/datasets/medical15_confirmation120_silver_v1/evaluator_targets.LOCAL_ONLY.private.jsonl`: independent offline scoring targets.
- `medical_scale_v06/score_study15_local.py` and `score_workflow11_local.py`: unchanged official scoring.

The runnable package includes the local import dependency closure of the original harness/scorer, not unrelated historical construction runs. All files listed in the recorded runtime hash manifests are preserved. Use `../scripts/run_experiment.py` to run the agents and `../scripts/reproduce.py` to score. The inference process does not load evaluator targets.
