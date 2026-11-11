# docs/plans/

Designs, the verification source dossier, and implementation plans for
platform work (platform implementation only, never research strategy:
CORRECTIONS 2026-08-12). A plan is written before its work and is not
maintained afterwards. Once executed it is a historical record, not
instructions; what the code does now is described in `docs/ARCHITECTURE.md`,
`docs/DATA_GENERATION.md` and the ADRs.

Tests and commit messages cite plans by short name ("plan 2b, Task 7"). The
table maps those names to files.

| File | Kind | Short name | Status |
|---|---|---|---|
| `2026-07-05-readme-roadmap-todo-design.md` | design | | executed: the README Roadmap (ADR-0009) |
| `2026-09-21-reference-data-verification-design.md` | design | | the design behind ADR-0066 |
| `2026-09-21-reference-data-verification-sources.md` | source dossier | | current: cited by ADR-0066, the criteria and the published records; nothing in it is ratified; an index of claims, sources and locators since 2026-11-11 (the sources' own sentences removed before the first archived release) |
| `2026-09-24-abaqus-data-pipeline-design.md` | design | | the design behind ADR-0069 |
| `2026-09-24-abaqus-data-pipeline-plan-1.md` | plan | plan 1 (ADR-0069) | executed, merged 2026-09-24 (`a59075f`) |
| `2026-09-24-abaqus-data-pipeline-plan-2.md` | plan | plan 2 | executed, merged 2026-09-25 (`4b30201`) |
| `2026-09-27-abaqus-datagen-platform-design.md` | design | | the design behind ADR-0071 |
| `2026-09-27-datagen-platform-plan-1.md` | plan | part one | executed, merged 2026-09-27 (`602ab0d`) |
| `2026-09-27-datagen-platform-plan-2a.md` | plan | plan 2a | executed, merged 2026-09-27 (`486d9be`) |
| `2026-09-27-datagen-platform-plan-2b.md` | plan | plan 2b | executed, merged 2026-09-28 (`8032ac3`) |
| `2026-09-27-validation-against-experiments-design.md` | design | | the design behind ADR-0072 |
| `2026-09-27-validation-plan-1.md` | plan | validation plan 1 | executed, merged 2026-09-27 (`3208612`) |
| `2026-09-28-datagen-platform-plan-3a.md` | plan | plan 3a | executed, merged 2026-09-28 (`ca70020`) |
| `2026-11-06-benchmark-by-path-design.md` | design | benchmark by path | executed on `feat/benchmark-by-path` (2026-11-06); recorded as notes on ADR-0032 and ADR-0039 |
| `2026-11-07-two-site-state-design.md` | design | two-site state | the design behind ADR-0075 (Accepted 2026-11-07); 3a built on `feat/two-site-state-3a` (2026-11-07), 3c on `feat/two-site-state-3c` and 3b on `feat/two-site-state-3b` (2026-11-09) |
| `2026-11-07-two-site-state-plan-3a.md` | plan | plan 3a | executed on `feat/two-site-state-3a` (2026-11-07): flow-map scheme only; the AR scheme and the pushforward chains are follow-ups; note 2026-11-08: the `return_map` arm does not train (stiffness, not scale); note (2) of the same date: the admissible arm built in two forms, `admissible_coords` and `projected_trial` (ADR-0075 note 2026-11-08) |
| `2026-11-08-open-evaluation-plan.md` | plan | open evaluation | the plan behind ADR-0074's mechanics (2026-11-08); executed the same day: task 1 with the plan, tasks 2–7 on `feat/open-evaluation-2-3` (case parameters and coverage, registry fields and the results registry, `withheld` and `grid`, the scoring and verification tools, `CONTRIBUTING.md`); the first blind set and the first external result wait on data |
| `2026-11-08-gradient-task-family-design.md` | design | gradient family | the design behind ADR-0076 (Accepted 2026-11-08); built on `feat/gradient-task-family` the same day: the free-flight finding, `eval/sensitivity.py`, Taylor's declaration, the committed reference record (`docs/gradients/`), `--mode gradient`; the first registered model's rows wait on a run |
| `2026-11-09-two-site-state-plan-3c.md` | plan | plan 3c | executed on `feat/two-site-state-3c` (2026-11-09): `eval/properties.py`, the spec's `property_tolerances`, `matched_initial_state_pairs` and `defined_properties`, `evaluate` writing the `prop_*` keys, the Properties tier first on every landing page with empty cells, `prop_error_growth` in the scoring tool; the deviator-trace kernel deferred, the semigroup row with 3b; the shipped rows' numbers wait on a DUG re-evaluation |
| `2026-11-09-two-site-state-plan-3b.md` | plan | plan 3b | executed on `feat/two-site-state-3b` (2026-11-09): `eval/restart.py` (the window, the keys, the semigroup error), `flow_map_rollout(frame_offset=)`, the spec's `restart_anchors` / `restart_horizons` / `restart_compositions` and the row's `restart_kind`, `evaluate_restart` and `--mode restart`, the Restart tier; declaration only, no shipped benchmark declares anchors |
| `2026-11-09-checkpoint-publishing.md` | plan | checkpoint publishing | executed on `feat/checkpoint-publishing` (2026-11-09): `tools/publish_checkpoints.py` stages a release's blessed bundles for one Hugging Face model repo per benchmark; the upload and the pointer rewrite happen at release from the `models/` mirror |
| `2026-11-08-card-fields-and-compliance-table.md` | plan | card fields and compliance table | executed 2026-11-08 (ADR-0065 follow-up 2, a plan not an ADR by the maintainer's ruling): the V&V card fields, the compliance table, the four shipped cards filled, both generated views; notes on ADR-0027, 0065 and 0074 |
| `2026-11-11-release-outline-to-v1.md` | outline | releases v0.5 to v1.0 | a long-term reference, not a schedule (maintainer, 2026-11-11): each release's closing criteria; the README Roadmap draws a release's checklist from it when the release is picked up |

No plan is in flight. What ADR-0071's part three still owes has no plan yet
(README Roadmap).
