# 0077 — Checkpoint selection on the reported metric: validation relative L2 of displacement

**Status**: Accepted (maintainer, in-session 2026-11-10)
**Type**: Ephemeral
**Date**: 2026-11-10
**Amends**: 0028 (supersedes its selection clause and the last sentence of its 2026-07-10 note), 0062 (supersedes the selection sentence of its in-training validation paragraph); 0055 stands as written; 0025 stands (see clause 3)

**Your call**:
1. Every training loop selects `model-best` on the mean over validation cases of the pooled relative L2 of displacement, the number ADR-0055 reports. Not millimetres, and not a composite with stress or state. Reversing is a small code change, but runs trained in between were selected differently from those before and after.
2. The 1D wave benchmark follows the same rule, although ADR-0025 names axial stress as its physics. Reversing selects that benchmark alone on stress relative L2 and breaks the uniform rule.
3. No existing run is re-selected or re-evaluated. Registered rows stand as selected on millimetres. Reversing means re-scoring the periodic checkpoints of each registered run, where they still exist.

## Context

ADR-0028 set selection to validation rollout position RMSE in millimetres, after a millimetre + megapascal sum had picked a step-4,000 checkpoint of a 100k run. ADR-0062 restated it for the flow map. ADR-0055 then made the pooled relative L2 of displacement the headline on every benchmark and did not touch selection. Since then a run is selected on one number and reported on another.

The two numbers differ in more than units. The millimetre mean weights each validation case by its displacement amplitude; the relative form weights cases equally. The effect measured so far is small: on 110 evaluator-scored periodic checkpoints of seven runs (measured outside this repository, 2026-11-10), the two selectors chose the same checkpoint in five runs and differed by 0.014 and 0.007 relative L2 in the other two. A composite with the state error was tried there and rejected: state errors are several times larger, dominate a sum, and on an autoregressive run selected the 10k checkpoint of a 300k run, the ADR-0028 failure again. The change is one of principle and of comparability across benchmarks.

## Decision

1. **The selector.** In every training loop, `model-best` is the checkpoint with the lowest mean over validation cases of `relative_l2_pooled` on displacement: the evaluator's `rollout_rel_l2_displacement`, on the same scored span, kinematic mask and rollout protocol the evaluator uses for that scheme. For the flow map that is the self-anchored rollout at the canonical interval.
2. **One rule for every benchmark and family.** No benchmark or model family elects another selector. A new benchmark inherits the rule.
3. **The 1D wave benchmark.** Displacement relative L2 selects there too. ADR-0025 rejected position as the headline because displacements are small in absolute terms; the relative form is not degenerate on this data. The two validation cases have a displacement RMS of 2.5 and 6.7 mm over the scored span, above the 2 mm lattice spacing, and no scored frame is below 0.001 mm. The six registered rows span 0.11 to 1.57 in displacement relative L2 and order the methods exactly as stress relative L2 does. Selection there was already on displacement, in millimetres; this clause changes its form, not its quantity.
4. **What stays in the log.** The validation line keeps position RMSE in millimetres and every auxiliary or state number it carries today, unselected, so old and new logs compare.
5. **The record.** `config.json` names the selector (`run.checkpoint_selector`). A run record without the field was selected on position RMSE in millimetres.
6. **Existing runs.** Runs before this record were selected on millimetres. None is re-selected or re-evaluated; registered results and blessed checkpoints stand.
7. **The pin.** A test holds, for one checkpoint in each training loop, that the loop's selection number equals the evaluator's mean `rollout_rel_l2_displacement` on the validation split.

## Alternatives considered

- **Keep millimetres.** Selection and report stay different numbers, and the validation mean stays weighted toward large-amplitude cases.
- **A composite with stress or state relative L2.** Rejected by the maintainer on the evidence above: the larger term dominates and selects early checkpoints.
- **Each benchmark selects on its own headline field.** Stress for the wave benchmark. Rejected: ADR-0055 already made displacement the headline there, and one rule keeps selection comparable across benchmarks.

## Consequences

- A run after this record and a run before it can differ in which checkpoint was kept, by about 0.01 relative L2 where measured. `checkpoint_selector` tells them apart.
- Recipe conclusions drawn from millimetre-selected fleets stand; a fleet that mixes both selectors says so.
- The divergence guard of by-path runs keeps reading millimetres against the domain scale: it is a sanity bound, not a selector.
- Validation costs the same: the relative form comes from the rollout already run.
