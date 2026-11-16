# First-formalization diagnosis and repair: recorded-trajectory audit

The fixed primary slice contains 2372 first-attempt pairs from 119 sources across five settings. Of 2,400 planned phases, 20 were blocked by the shared initial state and 8 had no first graph because of native/transport/action failures. No unobserved before/after output is imputed.

Before and after use the same public evidence within one formalization attempt. Later Guard rereads are excluded. Categories below are model-reported issue types, not manually verified error labels. This is a trace audit, not a component ablation.

| Quantity | Before diagnosis | After repair |
|---|---:|---:|
| Executable packets / 2372 | 2262 (95.36%) | 2351 (99.11%) |
| Exact packets / 2372 | 1669 | 1659 |
| Correct labels / 10259 | 8807 | 9047 |
| Observed-slice phase-macro score | 86.9323% | 89.1224% |
| Observed-slice source-macro score | 86.8513% | 89.0876% |

These scores use the same observed pair denominator and assign zero to non-executable outputs; they do not replace the full benchmark M, E, or J.

## Decomposition

- Execution rescue: 91 packets, adding 356 correct labels and 34 exact packets.
- Execution breakage: 2 packets, losing 7 correct labels and 1 exact packets.
- Both executable: 2260 pairs; 157 labels become correct and 266 become incorrect, a net -109. Exact packets change 1668 → 1625 (54 gained; 97 lost).
- Total correct-label change: 356 − 7 + 157 − 266 = +240.
- 19 pairs remain non-executable. 3238 diagnostic calls are logged, including 866 feedback calls after the initial diagnosis.

The observed net benefit comes from recovering executable coverage, while semantic edits to already executable graphs have a net negative label effect.

## First diagnosis categories

| Obligation | Reported issues | Phases reporting category |
|---|---:|---:|
| representation_validity | 133 | 113 |
| concept_role_fidelity | 263 | 198 |
| context_fidelity | 521 | 335 |
| logic_knowledge_fidelity | 834 | 526 |

## Fixed-pair breakdown by setting

| Setting | Pairs | Executable before → after | Exact before → after | Phase score before → after | Source score before → after |
|---|---:|---:|---:|---:|---:|
| fresh | 475 | 449 → 472 | 335 → 329 | 86.112 → 88.575 | 86.106 → 88.564 |
| labels | 475 | 455 → 471 | 339 → 330 | 87.635 → 89.200 | 87.577 → 89.153 |
| index | 473 | 457 → 469 | 332 → 324 | 87.748 → 88.795 | 87.609 → 88.676 |
| guard | 475 | 450 → 470 | 334 → 339 | 86.691 → 89.737 | 86.667 → 89.723 |
| index_guard | 474 | 451 → 469 | 329 → 337 | 86.477 → 89.304 | 86.475 → 89.279 |

## Audit

All model/solver results are pre-existing files. The analysis reads private scoring labels only offline. It verifies that every repair payload contains exactly the public task and the same observations as its formalization input, scans that payload for private answer-field names, and checks the final round against the saved after-result. The runtime evidence_sha256 hashes compiled facts, not retrieved observations, and can change when a record is corrected. No run files are written and no model/solver calls are made.

There are 254 additional Guard-attempt pairs, reported separately in summary.json. The root method/ copy in guarded phases is deliberately ignored because it is an overwritten convenience copy of later attempts.
