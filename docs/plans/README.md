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
| `2026-09-21-reference-data-verification-sources.md` | source dossier | | current: cited by ADR-0066, the criteria and the published records; nothing in it is ratified |
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
| `2026-11-07-two-site-state-design.md` | design | two-site state | the design behind ADR-0075 (Proposed); 3a built on `feat/two-site-state-3a` (2026-11-07), 3b and 3c not started |
| `2026-11-07-two-site-state-plan-3a.md` | plan | plan 3a | executed on `feat/two-site-state-3a` (2026-11-07): flow-map scheme only; the AR scheme and the pushforward chains are follow-ups; note 2026-11-08: the `return_map` arm does not train (stiffness, not scale); note (2) of the same date: the admissible arm built in two forms, `admissible_coords` and `projected_trial` (ADR-0075 note 2026-11-08) |
| `2026-11-08-gradient-task-family-design.md` | design | gradient family | the design behind ADR-0076 (Accepted 2026-11-08); built on `feat/gradient-task-family` the same day: the free-flight finding, `eval/sensitivity.py`, Taylor's declaration, the committed reference record (`docs/gradients/`), `--mode gradient`; the first registered model's rows wait on a run |
| `2026-11-08-card-fields-and-compliance-table.md` | plan | card fields and compliance table | executed 2026-11-08 (ADR-0065 follow-up 2, a plan not an ADR by the maintainer's ruling): the V&V card fields, the compliance table, the four shipped cards filled, both generated views; notes on ADR-0027, 0065 and 0074 |

No plan is in flight. What ADR-0071's part three still owes has no plan yet
(README Roadmap).
