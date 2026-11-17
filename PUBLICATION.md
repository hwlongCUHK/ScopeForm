# Public code and benchmark release

Released with the repository owner's authorization on 2026-11-16. This repository contains the ScopeForm benchmark main cohort, the LLM Agent baseline and ScopeForm implementations, runtime configurations, and constructed demonstration inputs. Recorded experimental results, predictions, model responses, execution traces, and result tables are excluded from the current release.

The benchmark's original filenames and manifests are retained for exact data/code hash verification. In particular, historical `public_release_authorized: false` fields describe the earlier collaborator snapshot; this repository is the subsequent author-authorized public release. `private` and `LOCAL_ONLY` in evaluator filenames distinguish scoring information from actor-accessible evidence. The inference entry point never loads those targets.

The frozen method/runtime files retain their recorded hashes. Portable launchers support fresh experiment runs and scoring of locally generated outputs. Model weights and credentials are not distributed. Third-party model and source-material rights remain with their respective owners; this release does not assign them a new license.
