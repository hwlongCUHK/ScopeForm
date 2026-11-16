"""Add the original acquisition guard to semantic v2 on all 120 main sources.

Reuse the completed actual initials and both controls; only the 480 planned
guard-arm updates make new API calls. No private scoring targets are loaded.
"""
import argparse
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import shutil
import time

import run_methods as run
from common import dump, read

ARMS = ("baseline", "frame_direct", "frame_direct_guard")
SOURCE = run.HERE / "runs/frame_semantic_v2_core_20261114"


def prepare(out, source, workers):
    old = read(source / "config.json")
    assert read(source / "run_status.json")["status"] == "complete"
    assert read(source / "main/complete.json") == {"planned": 1080, "recorded": 1080}
    runtime = run.baseline.RUNTIME
    protocol = read(runtime / "study15_protocol.json")
    assert old["source_ids"] == protocol["source_ids"] and len(old["source_ids"]) == 120
    actors = [json.loads(line) for line in
              (runtime / "confirmation15_inputs_private/actors.private.jsonl").read_text().splitlines() if line]
    assert {a["source_id"] for a in actors} == set(old["source_ids"])
    for name, expected in old["code_sha256"].items():
        if name not in ("run_core_reproduction.py", "run_methods.py", "methods/frame_direct.py"):
            assert run.sha(run.HERE / name) == expected, name
        assert run.sha(source / "frozen_code" / name) == expected, name
    for name, expected in old["runtime_code_sha256"].items():
        assert run.sha(runtime / name) == expected, name
    assert run.sha(runtime / "confirmation15_inputs_private/actors.private.jsonl") == old["actors_sha256"]
    code_names = (set(old["code_sha256"]) - {"run_core_reproduction.py"}) | {"run_guard_core.py"}
    config = {
        **{k: old[k] for k in ("package_root", "baseline_root", "model", "endpoint", "source_ids",
            "layout_source_ids", "events", "study_protocol_sha256", "actors_sha256", "actor_manifest_sha256",
            "dataset_integrity_sha256", "runtime_code_sha256", "original_baseline_sha256", "runtime_patch")},
        "kind": "semantic_v2_original_guard_main120", "suite": "core",
        "settings": {**old["settings"], "workers_per_model": workers},
        "experimental_arms": list(ARMS), "environment_arm": "fresh",
        "guard_arm": "frame_direct_guard", "guard_implementation": "Original workflow11_runtime.World.call, unchanged",
        "guard_feedback": "The same native agent receives the original violation result, reads sources and submits a new graph",
        "prior_labels_or_index_added": False,
        "source_run": str(source), "reused_arms": ["initial", "baseline", "frame_direct"],
        "source_config_sha256": run.sha(source / "config.json"),
        "source_results_sha256": run.sha(source / "main/results.json"),
        "source_pairing_sha256": run.sha(source / "main/pairing_audit.json"),
        "source_provider_metadata": read(source / "provider_metadata.json"),
        "planned_phases": 1560, "planned_reused_phases": 1080, "planned_new_phases": 480,
        "planned_shared_initials": 120, "planned_updates_per_arm": 480, "shared_initials": True,
        "new_api_calls_for_all_phases": False, "notes_enabled": False,
        "budget": "All physical model requests <=40 and all actions <=40 per phase, including rejected saves/submissions and feedback rounds",
        "integration": "Native reads -> graph/solver -> actual original guarded save/submit -> original violations -> same agent reads/updates graph; final solver labels are submitted",
        "method_temperatures": "temperature=0, thinking disabled; no extra semantic checker, fallback or formalization-error repair",
        "comparison": "Guard minus semantic v2 using identical actual initials/public tasks and reused prior controls; separate API run, not a shared-generation ablation",
        "code_sha256": {name: run.sha(run.HERE / name) for name in sorted(code_names)},
    }
    if (out / "config.json").exists() and read(out / "config.json") != config:
        raise ValueError("Run configuration changed; use a new directory")
    dump(out / "config.json", config)
    for name in code_names:
        target = out / "frozen_code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(run.HERE / name, target)
        assert run.sha(target) == config["code_sha256"][name]
    for prefix, origin, hashes in (
            ("runtime", runtime, config["runtime_code_sha256"]),
            ("baseline", run.BASELINE_ROOT, config["original_baseline_sha256"])):
        for name, expected in hashes.items():
            assert run.sha(origin / name) == expected
            target = out / "frozen_code" / prefix / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(origin / name, target)
            assert run.sha(target) == expected
    # Cached rows/checkpoints are read-only inputs to the unchanged Runner.one.
    # Symlinks avoid duplicating all historical API request/response files.
    for name in ("shared", "baseline", "frame_direct"):
        target = out / "main" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(source / "main" / name, target_is_directory=True)
        assert target.is_symlink() and target.resolve() == source / "main" / name
    return config, actors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, default=SOURCE)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out, source = args.run_dir.resolve(), args.source_run.resolve()
    config, actors = prepare(out, source, args.workers)
    if args.prepare_only:
        print(json.dumps({"prepared": str(out), "sources": len(actors), "planned_new": 480, "reused": 1080, "api_calls": 0}))
        return
    key = os.environ.get("DEEPSEEK_API_KEY") or getpass.getpass("DeepSeek API key (hidden): ")
    client = run.baseline.Client(key, config["endpoint"])
    del key
    runner = run.Runner(client, config["model"], config["settings"], ARMS)
    run.h.phase = runner.phase
    started = time.monotonic()
    dump(out / "run_status.json", {"status": "running", "planned": 1560, "planned_new": 480,
                                  "started_utc": datetime.now(timezone.utc).isoformat()})
    runner.workflow(actors, out / "main")
    dump(out / "provider_metadata.json", {"requested_model": config["model"],
         "returned_models": sorted(client.models), "system_fingerprints": sorted(client.fingerprints),
         "scope": "new guard-arm requests only; reused controls have separate source_provider_metadata"})
    dump(out / "run_status.json", {"status": "complete", "planned": 1560, "planned_new": 480,
         "wall_seconds": time.monotonic() - started, "finished_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
