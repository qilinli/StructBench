# 0028 — GNS baseline training-recipe rework after the first full run

**Status**: Accepted
**Type**: Ephemeral
**Date**: 2026-07-03

## Context

The first full Taylor 2D baseline run (100k steps, 2026-07-03; record in
`runs/taylor-baseline/SUMMARY.md`) produced excellent one-step accuracy
(0.004 mm) but qualitatively wrong long rollouts: scrambled particles at the
impact face and near-mean stress. A root-cause investigation (two independent
code audits plus targeted experiments on the trained checkpoint) found no bug
in the noise implementation, normalization, windowing, or integrator — all
faithful to the DeepMind GNS reference — but a compounding set of recipe
defects:

1. **Checkpoint selection summed mm + MPa** (47:1 stress-dominated): the
   shipped model was the step-4,000 checkpoint of a 100k run, selected almost
   purely for rollout-stress RMSE, which a near-mean stress predictor wins
   early. Position quality never influenced selection.
2. **connectivity_radius 0.6 mm = 1.2× the 0.5 mm lattice spacing**: degree
   ~4, no diagonal edges, 0.1 mm edge-break margin. Rollouts produced 30+
   fully isolated particles (ballistic, message-free) and 0.006 mm particle
   overlaps by frame 100 — the direct scrambling mechanism. The
   `max_num_neighbors=20` cap then truncates asymmetrically once clumping
   starts.
3. **Wall feature `clamp(x − wall, 0, R)` erased penetration**: identically
   zero for a particle resting on the wall and one driven through it (ground
   truth itself penetrates to −0.34 mm with ~100 particles at feature ≡ 0), so
   rollouts got no restoring signal at the exact failure region.
4. **Optimization hygiene**: lr 1e-3 (10× the reference) at batch 32 over only
   2,961 unique windows ≈ 1,080 epochs; unclipped 5× loss spikes.
5. **Terminal dt artifact**: LS-DYNA writes its termination state 0.077 µs
   (not ~2 µs) after the previous dump in every case, corrupting one training
   target per trajectory and biasing final-frame metrics (~0.03 mm — minor).

## Decision

Recipe and code changes, applied together as the v0.1 baseline recipe:

- **Model selection on rollout position RMSE alone**; the validation log
  reports position (mm) and von Mises (MPa) channels separately. The ADR-0019
  *reported* metrics are unchanged.
- **connectivity_radius 1.5 mm (3× spacing) with `max_neighbors = 48`**, now a
  `GNSConfig` field recorded in `config.json`.
- **Signed, radius-normalized wall feature** `clamp((x − wall)/R, −1, 1)`.
- **lr_init 1e-4** and **gradient clipping** (global norm 1.0).
- **Terminal-frame trim** in the canonical trajectory loader
  (`n_valid_frames`): the final frame is dropped when the last interval is
  under half the median output interval; the viz loader applies the same rule
  so ground truth stays aligned with rollout artifacts.

Deliberately *not* changed here (Phase-2 ablation axes, pending evidence):
noise_std, w_aux / a separate aux head, model capacity, and any
stress-history input feature.

## Alternatives considered

- **Fix only the selection metric and rerun**: cheapest, but the isolated
  particles and penetration blindness are mechanical defects visible in the
  rollouts themselves; rerunning without them wastes a GPU-day.
- **Jump straight to the reference architecture (1.6M params, 10 MP steps)**:
  confounds capacity with the mechanical fixes; kept as a later ablation.
- **Rollout-in-training (push-forward) methods**: promising for the residual
  accumulation, but a research-scale change; premature before the recipe
  defects are removed.

## Consequences

- Prior runs are not comparable to post-0024 runs (different wall feature,
  graph, trajectory length 151 vs 152). `runs/taylor-baseline` remains the
  pre-0024 record.
- Edge count roughly triples (degree ~4 → ~19): training throughput drops
  accordingly; the measured 14k steps/h on an A100 is expected to fall to
  roughly a third of that.
- The trimmed final frame moves the QoI evaluation point ~2 µs earlier
  (bar essentially at rest; negligible physically).
- `GNSConfig` gains `max_neighbors`; older `config.json` files without it
  load with the (new) default — `evaluate()` on pre-0024 run dirs still works
  because architecture is rebuilt from the run's own record.

---

*2026-07-05 — the Phase-2 ablation this ADR deferred (noise_std, w_aux,
model capacity) has run (fleets in `runs/fleet-2026-07-03` and
`runs/fleet-2026-07-04`, manifests + metrics per run). Maintainer decision:
baseline recipes are config, not ADR material — a drafted recipe ADR was
withdrawn before merge (its candidate number was since claimed by the
data-archive ADR-0031). Recipe values live
in `configs/<benchmark>/<family>.toml`; the fleet evidence also predates the
ADR-0032 protocol change (init 11 → 3), so recipe conclusions must be
re-established under the new protocol before the final baseline. Stress-history
features and push-forward training remain open.*

---

*2026-07-10 — checkpoint tooling addendum after the wave-1D round-1 fleet
(drafted by Claude Code, maintainer to confirm). That fleet showed the
validation selection signal is volatile at 2-case scale (adjacent-eval swings
of 14–66%, 8–18% same-weights CUDA eval spread; one seed's outcome was set by
a single transient eval whose better later states were never written to
disk). The trainer therefore now also snapshots `ckpt-<step>.pt` every 10k
steps — deliberately outside the `model-*.pt` selection glob, ignored by
default evaluation — and the eval modes accept `--checkpoint` to score one
explicitly (metrics land in a `metrics-<split>@<stem>.json` side file; the
canonical artifacts of the selected checkpoint are never overwritten). This
enables post-hoc SMOOTHED selection applied identically across recipe-fleet
arms. The in-run selection rule this ADR set — best checkpoint on rollout
position RMSE alone — is unchanged.*
