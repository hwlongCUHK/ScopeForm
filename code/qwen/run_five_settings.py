"""Matched original-agent/formalization comparisons in all five paper settings.

Reuse the frozen, fully evaluated inference and actual initial worlds. Only the
experiment wrapper changes: each pair receives the original paper branch,
including its actual prior labels/index and Guard. No targets are loaded here.
"""
import argparse
from datetime import datetime, timezone
import getpass
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "versions/frame_repair_id_fields_20261114"
SETTINGS = {
    "fresh": "fresh",
    "labels": "labels_only",
    "index": "index_only",
    "guard": "labels_guard",
    "index_guard": "evidence_constraint",
}
GUARDED = {"guard", "index_guard"}
ARMS = tuple(f"{setting}__{variant}" for setting in SETTINGS
             for variant in ("baseline", "formalization"))


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_frozen(source=DEFAULT_SOURCE):
    """Import the selected, already evaluated version, not development files."""
    frozen = source / "frozen_code"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(frozen))
    run = importlib.import_module("run_methods")
    if run.HERE.resolve() != frozen.resolve():
        raise ValueError("Another run_methods version is already imported")
    return run


def runner_class(run):
    class FiveSettingsRunner(run.Runner):
        def __init__(self, client, model, settings, method):
            super().__init__(client, model, settings, ARMS)
            self.method = method

        def one(self, actor, event, arm, out, base=None, layout=None, tool_port=0):
            if layout is not None:
                raise ValueError("This experiment uses the paper's main cohort")
            setting, variant = arm.split("__") if event else ("initial", "baseline")
            environment = SETTINGS[setting] if event else "initial"
            inference = "baseline"
            if variant == "formalization":
                inference = "frame_direct"
                if setting in GUARDED:
                    inference += "_guard"
                if self.method == "repair":
                    inference += "_repair"
            self.local.arm = inference
            folder = Path(out) / (arm if event else "shared")
            row, world = run.h.one(actor, event, environment, folder, 0, self.model,
                                   None, self.settings, base=base)
            relative = Path(arm if event else "shared") / "phases" / actor["source_id"] / (
                (event["event_id"] if event else "initial") + "__" + environment)
            row = {**row, "arm": arm if event else "initial",
                   "paper_setting": setting, "variant": variant,
                   "inference_arm": inference, "environment_arm": environment,
                   "phase_relative_path": str(relative),
                   "harness": "five_paper_settings_matched_formalization"}
            checkpoint = Path(out) / relative / "checkpoint.private.json"
            if checkpoint.exists():
                cp = read(checkpoint)
                row.update(method_started_at_request=cp.get("method_started_at_request"),
                           method_completed=cp.get("method_completed", False),
                           method_status=(cp.get("method_result") or {}).get("status"))
            return row, world
    return FiveSettingsRunner


def prepare(out, source, method, scope, workers):
    old = read(source / "config.json")
    run = load_frozen(source)
    for name, digest in old["code_sha256"].items():
        assert run.sha(source / "frozen_code" / name) == digest, name
    for root, hashes in ((run.baseline.RUNTIME, old["runtime_code_sha256"]),
                         (run.BASELINE_ROOT, old["original_baseline_sha256"])):
        for name, digest in hashes.items():
            assert run.sha(root / name) == digest, name
    actors_file = run.baseline.RUNTIME / "confirmation15_inputs_private/actors.private.jsonl"
    assert run.sha(actors_file) == old["actors_sha256"]
    actors = [json.loads(line) for line in actors_file.read_text().splitlines() if line]
    assert {a["source_id"] for a in actors} == set(old["source_ids"])
    if scope == "pilot":
        # Preserve 72:48 proportions; select without looking at outcomes.
        actors = [a for modality, count in (("dialogue", 6), ("drug", 4))
                  for a in sorted((a for a in actors if a["initial"]["modality"] == modality),
                                  key=lambda a: run.baseline.fingerprint(a["source_id"]))[:count]]
    actors.sort(key=lambda a: a["source_id"])
    total = len(actors) * 4 * len(ARMS)
    config = {
        **{k: old[k] for k in ("package_root", "baseline_root", "model", "endpoint",
            "events", "study_protocol_sha256", "actors_sha256", "actor_manifest_sha256",
            "dataset_integrity_sha256", "runtime_code_sha256", "original_baseline_sha256",
            "runtime_patch", "code_sha256")},
        "kind": "five_paper_settings_matched_formalization", "suite": "core", "scope": scope,
        "source_ids": [a["source_id"] for a in actors], "layout_source_ids": [],
        "settings": {**old["settings"], "workers_per_model": workers},
        "experimental_arms": list(ARMS), "paper_settings": SETTINGS,
        "formalization_version": method, "frozen_inference_source": str(source / "frozen_code"),
        "source_run": str(source), "source_config_sha256": run.sha(source / "config.json"),
        "reused": "Actual initial worlds only; every update arm runs new native requests",
        "source_selection": "Full protocol cohort" if scope == "full" else
            "Lowest sha256(source_id) within modality: 6 dialogue, 4 drug; no outcome selection",
        "planned_phases": len(actors) + total, "planned_new_phases": total,
        "planned_shared_initials": len(actors), "planned_updates_per_arm": len(actors) * 4,
        "shared_initials": True, "notes_enabled": False,
        "budget": "All physical native/helper requests <=40 and actions <=40 per phase; failures stay in denominator",
        "integration": "Original paper branch -> native labels or frozen formalization -> original submission/Guard",
        "method_temperatures": "temperature=0, thinking disabled; identical formalization version in all five settings",
        "primary_endpoint": "Within-setting source-macro coverage-adjusted answer score M",
        "secondary_endpoints": ["J", "E", "submission", "requests", "tokens", "latency"],
        "analysis_plan": "Pair by source, keep all four events together; 20000 source bootstrap resamples, seed 20261114; report 95% intervals and Bonferroni 99% M intervals for the five simultaneous contrasts",
        "interpretation": "Pilot is integration/development evidence; full run tests benchmark-wide effects, not unseen-data generalization; no setting-specific method selection",
        "harness_sha256": run.sha(__file__),
    }
    if (out / "config.json").exists():
        assert read(out / "config.json") == config, "Existing run configuration differs"
    dump(out / "config.json", config)
    shutil.copy2(__file__, out / Path(__file__).name)
    shared = out / "main/shared"
    shared.parent.mkdir(parents=True, exist_ok=True)
    if not shared.exists():
        shared.symlink_to((source / "main/shared").resolve(), target_is_directory=True)
    assert shared.resolve() == (source / "main/shared").resolve()
    frozen = out / "frozen_code"
    if not frozen.exists():
        frozen.symlink_to((source / "frozen_code").resolve(), target_is_directory=True)
    assert frozen.resolve() == (source / "frozen_code").resolve()
    return run, config, actors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--method", choices=("formalization", "repair"), default="formalization")
    parser.add_argument("--scope", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out = args.run_dir.resolve()
    run, config, actors = prepare(out, args.source_run.resolve(), args.method, args.scope, args.workers)
    if args.prepare_only:
        print(json.dumps({"prepared": str(out), "sources": len(actors),
                          "planned_new_phases": config["planned_new_phases"], "api_calls": 0}))
        return
    key = os.environ.get("DEEPSEEK_API_KEY") or getpass.getpass("DeepSeek API key (hidden): ")
    client = run.baseline.Client(key, config["endpoint"])
    del key
    runner = runner_class(run)(client, config["model"], config["settings"], args.method)
    run.h.phase = runner.phase
    start = time.monotonic()
    dump(out / "run_status.json", {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat()})
    runner.workflow(actors, out / "main")
    dump(out / "provider_metadata.json", {"requested_model": config["model"],
         "returned_models": sorted(client.models), "system_fingerprints": sorted(client.fingerprints)})
    dump(out / "run_status.json", {"status": "complete", "wall_seconds": time.monotonic() - start,
         "finished_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
