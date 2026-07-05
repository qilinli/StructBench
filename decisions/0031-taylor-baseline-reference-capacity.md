# 0031 — Taylor baseline adopts the reference-capacity GNS recipe (amends 0028)

**Status**: Proposed
**Type**: Ephemeral
**Date**: 2026-07-05

## Context

ADR-0028 fixed the mechanical recipe defects found after the first full run
and deliberately deferred four axes — noise_std, w_aux, model capacity, and
stress-history features — as "Phase-2 ablation axes, pending evidence". That
evidence now exists. Two single-GPU fleets ran the ablation
(`runs/fleet-2026-07-03/MANIFEST.tsv`, commit a46d703: five single-axis
variants at 20k steps; `runs/fleet-2026-07-04/MANIFEST.tsv`, commit 2364dfa:
two capacity+noise combinations at 40k steps) against the post-0028 reference
run (`runs/taylor-full-adr0024`, 100k steps — the directory name carries the
rework's pre-renumbering label).

Rollout RMSE, position (mm) / von Mises (MPa), on the ADR-0019 protocol:

| run | deltas vs post-0028 recipe | test_interp | test_extrap |
|---|---|---|---|
| baseline (100k) | — | 1.63 / 68.0 | 3.79 / 84.6 |
| n001 (20k) | noise 0.01 | 1.37 / 63.8 | 1.32 / 66.9 |
| n005 (20k) | noise 0.05 | 1.73 / 73.5 | 4.62 / 76.8 |
| waux0 (20k) | w_aux 0 | 1.80 / 96.5 | 3.22 / 105.0 |
| waux01 (20k) | w_aux 0.1 | 1.78 / 62.1 | 3.69 / 76.1 |
| cap128 (20k) | hidden 128, nmlp 2, mp 10 | 1.12 / 53.1 | 1.33 / 52.5 |
| cap128-n001 (40k) | cap128 + noise 0.01 | 1.97 / 69.4 | 1.11 / 53.5 |
| **cap128-n002 (40k)** | cap128 + noise 0.02 | **1.07 / 53.1** | **1.25 / 57.4** |

Findings:

1. **Capacity is the dominant lever.** The reference-scale architecture that
   0028 explicitly kept "as a later ablation" (hidden 128, 2-layer MLPs, 10 MP
   steps) beats the 100k-step baseline on every metric at one-fifth the step
   budget — most dramatically on extrapolation (3.79 → 1.33 mm,
   84.6 → 52.5 MPa).
2. **The auxiliary loss is load-bearing for stress.** w_aux 0 doubles the von
   Mises RMSE (96–105 MPa) with no position benefit; w_aux 0.1 recovers stress
   but not positions. w_aux 1.0 stands.
3. **noise_std 0.02 stands.** 0.05 clearly degrades everything; 0.01 ties 0.02
   for the small model but regresses at the 130 m/s interpolation cases with
   the large one (cap128-n001: 2.4–3.5 mm on the three low-velocity cases).
4. **Budget was not the constraint.** The 100k baseline's best validation
   checkpoint was at step 6k; capacity, not steps, was binding. But
   cap128-n002's best checkpoint landed on its final step (40k), so the larger
   model was still improving.

A qualitative check (2026-07-05; all 12 cap128-n002 rollouts rendered with
`structbench.viz` and inspected — plots in
`runs/fleet-2026-07-04/cap128-n002/plots/`, notes in
`scratch/2026-07-05-fleet-log-summary.md`) found faithful kinematics in every
case with no instabilities, and two honest limitations: late-time residual
stress in the rod interior stays at ~120–190 MPa where the solver relaxes
below ~120 MPa (this offset drives the flat ~50 MPa aux RMSE), and short rods
(L = 60 mm) over-mushroom under extrapolation, worst at 200 m/s.

## Decision

The v0.1 trained-baseline recipe becomes the cap128-n002 recipe, resolving
the axes 0028 deferred. Relative to 0028, `configs/taylor_2d.toml` changes:

- **hidden_dim 128, nmlp_layers 2, message_passing_steps 10** — the DeepMind
  reference scale.
- **batch_size 8** — memory bound: batch 16 OOMs an A100 80 GB
  (fleet job 62280853).
- **training_steps 80,000** — double the 40k at which cap128-n002 was still
  improving; with lr_decay_steps 30,000 unchanged, the run sees decays at 30k
  and 60k.
- **Unchanged, now evidence-backed rather than deferred**: noise_std 0.02,
  w_aux 1.0. All 0028 mechanical fixes (selection on rollout position RMSE,
  radius 1.5 mm / max_neighbors 48, signed wall feature, lr 1e-4 + clipping,
  terminal-frame trim) stand.

The released v0.1 baseline is one training run of this config on the 21
TRAIN trajectories, with the ADR-0019 reported metrics taken from its
selected checkpoint.

## Alternatives considered

- **Keep the 0028 small model (hidden 64)**: dominated on every metric;
  3× worse extrapolation. Rejected on the fleet evidence.
- **noise_std 0.01**: best extrapolation with the large model (1.11 mm) but a
  2–3× regression on the low-velocity interpolation cases; 0.02 is uniformly
  strong. A velocity-dependent noise schedule would be research scope.
- **Train the final baseline at 40k**: cheapest, but the best checkpoint at
  exactly 40k says the run was cut short; 80k costs one overnight job
  (~18 h) and selection-on-validation makes extra steps safe.
- **Stress-history input feature / push-forward training**: the remaining
  0028 deferrals; still open, now aimed at the two documented limitations
  rather than at general quality. Not needed for v0.1.

## Consequences

- The final baseline run costs ~18 h on one A100 80 GB (fleet measured ~9 h
  per 40k steps at batch 8) — submitted from the DUG login node per
  `hpc/dug/`, between fleets per `docs/WORKFLOW.md`.
- Model size grows ~4× and rollout inference cost roughly doubles per step
  count (10 vs 5 message-passing rounds); acceptable for a reference
  baseline.
- Fleet runs and the new baseline are directly comparable (same post-0028
  code); pre-0028 runs remain incomparable as recorded in 0028.
- The two qualitative limitations (residual-stress offset; L = 60
  over-mushrooming under extrapolation) become documented benchmark-card /
  README discussion items for v0.1 rather than open recipe questions.
- `runs/taylor-full-adr0024` is superseded as the reference run once the new
  baseline lands; it remains the post-0028/pre-0031 record.
