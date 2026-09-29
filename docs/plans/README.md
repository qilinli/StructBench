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

No plan is in flight. What ADR-0071's part three still owes has no plan yet
(README Roadmap).
