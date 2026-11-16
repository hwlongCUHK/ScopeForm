# ScopeForm reproducibility bundle

Code and recorded results for **Context Before Calculation: Context-Faithful Formalization for Medical Evidence Agents**.

See [README_中文.md](README_中文.md) for the full setup, code map, protocol, and data documentation.

- [Benchmark and data](benchmark/README.md): 120 source clusters, 519 claims, four updates, five settings.
- [Agent and method code](code/README.md): native LLM Agent, ScopeForm, semantic library, ID-addressed repair.
- [Results table](results/table1.md) · [CSV](results/table1.csv) · [Validation](validation/README.md).

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/reproduce.py
```

This invokes the unchanged evaluator on 9,840 recorded phases, reproduces all 60 numerical cells of the paper's five-setting/two-backbone table, recomputes paired bootstrap intervals and cost summaries, and reconstructs the recorded repair audit. No API, GPU, or model weights are needed. `--skip-repair-audit` skips extraction of the detailed repair traces. Results go to `output/reproduced/`.

```bash
python scripts/demo.py
python scripts/replay_case.py
python scripts/run_experiment.py --backbone deepseek --scope pilot --out runs/ds --prepare-only
python scripts/run_experiment.py --backbone deepseek --scope pilot --out runs/ds
python scripts/reproduce.py --run-dir runs/ds --out runs/ds/evaluation
```

The runner reads `DEEPSEEK_API_KEY` from the environment or a hidden prompt. For Qwen use `--backbone qwen --endpoint http://127.0.0.1:18227/v1`. `--initials recorded` is the default; `--initials fresh` reruns initial phases. Every source is paired across the native baseline and ScopeForm. All native, formalization and repair calls share a 40-request/40-action limit per phase.

The archived DeepSeek and Qwen snapshots are the actual versions used, including Qwen's schema/parse adaptations. All five settings within each backbone use the same snapshot. Guard is an experimental evidence-access condition, not a ScopeForm component. A recorded Qwen interruption is retained as failure, with its original audit.

This is an author-authorized public reproducibility release containing actor data, evaluator targets, predictions, and traces. Historical `private` / `LOCAL_ONLY` filenames describe separation from the evaluated agent. Inference does not read evaluator targets. Credentials and model weights are excluded. See [PUBLICATION.md](PUBLICATION.md). Recorded offline scoring is exact; a new model run can differ because of inference nondeterminism or service changes.
