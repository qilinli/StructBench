# 0043 — DeformingPlate benchmark protocol: task, split, eval, and the MGN blessing gate

**Status**: Accepted
**Type**: Durable
**Date**: 2026-08-08

## Context

ADR-0041 made `DeformingPlate` (MeshGraphNets, Pfaff et al. 2021) the v0.3
benchmark; ADR-0042 landed its canonical ingestion (schema 0.2.0, per-node
fields). This ADR fixes the benchmark protocol — the ADR-0019/0025/0026
counterpart — and, because MGN is the *validatable anchor* of the v0.3
method comparison, the acceptance gate for blessing the native MGN baseline
against published numbers.

The protocol below is grounded in a verified research pass (2026-08-08,
78/78 claims adversarially confirmed against sources): the paper's per-dataset
tables (ar5iv 2010.03409), the official `meta.json` and DeepMind framework
code, MGN-baseline numbers from four later papers (M4GN, MAVEN, Unisoma,
MGN-Transformer), and the repo's own protocol machinery (card/registry
validation, `scored_frames`, `kinematic_types`, ADR-0035). Two facts shape
everything:

1. **The paper's DeformingPlate model uses no velocity history (h = 0)** —
   node inputs are the one-hot node type only; scripted (actuator) nodes
   additionally receive the next-step world-space velocity. Dynamics enter
   through relative edge features and one-step integration.
2. **The published headline number is a pooled RMSE** — root of the mean
   squared position error pooled over coordinates × nodes × all rollout
   steps × the 100 test trajectories: 15.1 ± 4.0 (×10⁻³, dataset-native
   length units), corroborated independently at 14.7 (M4GN) and 15.1
   (MGN-Transformer). This is a *different statistic* from StructBench's
   leaderboard mean-of-per-step-RMSE; conflating conventions destroys
   comparability (MAVEN's per-step-averaged 50-step MGN number differs from
   the paper's by ~8× for exactly this reason).

## Decision

1. **Task.** Quasi-static load-stepping autoregressive rollout. Each of the
   400 frames is a load step; `response/time/t` is the frame index
   (`dt = 0` in the source; pseudo-time). The model advances node positions
   frame to frame and co-predicts the nodal auxiliary field. The card's
   `output_dt_ms = 1.0` is **nominal pseudo-time** (one unit per frame),
   stated as such in the card; no physical milliseconds are implied.

2. **Split.** The published split verbatim: 1000 train / 100 val / 100 test
   trajectories (`n_cases = 1200`, `n_frames = 400`). Case ids follow the
   converter's naming: `train_0000…train_0999`, `val_0000…val_0099`,
   `test_0000…test_0099`. Split keys are `train`/`val`/`test` (the published
   names; `eval_splits = ("val", "test")` — no interpolation/extrapolation
   distinction exists for this dataset).

3. **Input window: `input_frames = 2`.** The paper uses h = 0 history;
   StructBench's floor is 2 (a velocity requires two frames). 2 is the
   unique faithful value; the scored span [2, 400) leaves the rollout
   maximally comparable to the paper's whole-trajectory convention. Per
   ADR-0035 the window is the rollout init — no history backfill — and the
   card pins it (`config` equality check applies).

4. **Kinematic prescription: `kinematic_types = (1, 3)`.** Node-type codes
   OBSTACLE = 1 (scripted actuator) and HANDLE = 3 (fixed) are prescribed
   from ground-truth positions at every rollout step and masked from all
   metrics; only NORMAL = 0 plate nodes are predicted and scored. This ADR
   explicitly blesses reading `kinematic_types` as **node-type codes**
   (sourced from `nodes.node_type`, schema 0.2.0) — on SPH benchmarks the
   same field carries part-ids; the mechanism (`np.isin` on
   `particle_type`) is identical, the vocabulary differs. Training mirrors
   the official framework: noise injection and the loss are masked to
   NORMAL nodes only.

5. **Aux field and metrics.** `aux_field = "von_mises_stress"` — the nodal
   field stored directly at `response/node/von_mises_stress` (T, N, 1); the
   mesh-aware loader reads it verbatim (no Voigt reconstruction — the
   element-stress extractor path does not apply). Leaderboard metrics are
   the standard set (ADR-0019 §5 / ADR-0035): per-step and mean position
   RMSE, per-step and mean aux (von Mises) RMSE, teacher-forced one-step
   position and aux RMSE — all over NORMAL nodes only. **The stress RMSE is
   a StructBench-native metric**: the paper trains its stress output but
   publishes no quantitative rollout stress error; there is no reproduction
   target for it.

6. **Scored horizon: full.** `scored_frames = None`, card
   `horizon = "full"` — the scored span is [2, 400), matching the paper's
   headline whole-trajectory convention. The dataset is quasi-static with
   no known late-horizon pathology (the ADR-0039 situation does not arise);
   the per-frame arrays remain the long-horizon diagnostic. Notation
   convention, stated once for all successors: scored spans are
   **exclusive-end**, `[input_frames, scored_end)` — ADR-0039's
   inclusive-looking prose is not inherited.

7. **QoIs (declared StructBench definitions — no published QoI exists;
   excluded from the blessing gate).** Two:
   - `peak_vm_stress` — the maximum nodal von Mises stress over NORMAL
     nodes across the scored span.
   - `terminal_peak_deflection` — the maximum displacement magnitude
     (L2 norm of displacement from the initial frame) over NORMAL nodes at
     the final scored frame. (The late-trajectory deformed state is the one
     externally precedented evaluation point on this dataset — Unisoma's
     one-shot task at the frame of largest actuator travel.)
   Errors report as pred − true per case; registries aggregate MAE
   (existing convention).

8. **The native-MGN blessing recipe and gate.**
   - *Recipe (pinned by primary sources; the reference configuration for
     the blessing run):* node features = one-hot node type (+ next-step
     world-space velocity on OBSTACLE nodes); mesh-edge features
     (u_ij, |u_ij|, x_ij, |x_ij|); world edges (x_ij, |x_ij|) between
     non-mesh-connected pairs within r_W = 0.03 (= the dataset's
     `collision_radius`), undirected, no node-type filtering; outputs =
     (velocity, von Mises stress), velocity integrated once (forward Euler,
     Δt = 1), stress direct; 15 message-passing steps, latent 128,
     two-hidden-layer MLPs; training noise N(0, 3e-3) on world positions of
     NORMAL nodes with first-order target adjustment; Adam, batch 2, 10M
     steps, lr 1e-4 → 1e-6 exponential decay over 5M steps, normalizer
     warmup. Deviations (e.g. shorter training for compute reasons) are
     recorded in the training ledger; the gate below applies regardless.
   - *Gate:* the pooled rollout-all position RMSE on the 100-trajectory
     test split — computed under the **paper's pooled convention** (root of
     the mean squared error over coords × NORMAL nodes × scored steps ×
     trajectories), in the dataset's **native position units** (not the mm
     working frame) — must fall within **15.1 ± 4.0 ×10⁻³**
     (11.1–19.1 ×10⁻³). A teacher-forced 1-step sanity value within
     0.20–0.31 ×10⁻³ is reported alongside; pooled rollout-50 is recorded
     as informational (published band 1.3–2.4 ×10⁻³). 14–16 ×10⁻³ is the
     expected landing zone (the two independent corroborations), but the
     gate is the primary source's own spread — a reimplementation is not
     held tighter than the paper's trajectory-level variability. The pooled
     validation aggregate is blessing-only; the leaderboard keeps the
     standard per-step-mean statistics of §5.

9. **Declared choices (no primary source exists; recorded as StructBench
   protocol, not reproduction).**
   - *Stress supervision:* a single L2 loss on the concatenated
     (velocity, stress) decoder output, per-component normalized, equal
     weight, NORMAL-masked. (The official `deform_model.py` was never
     released — confirmed 404; the paper specifies a single L2 on the
     output vector and no per-component weights.)
   - *Actuator prescription at rollout:* ground-truth next positions
     (consistent with the reproduction lineage and with StructBench's
     kinematic mechanism). Whether DeepMind's unreleased evaluation did
     exactly this cannot be confirmed; the gap is recorded here.

## Alternatives considered

- **Pin `scored_frames = 50`** to target the better-conditioned rollout-50
  number. Rejected: the headline published convention is whole-trajectory;
  no late-horizon pathology motivates a pin; rollout-50 survives as an
  informational blessing checkpoint.
- **CGN-style history window (`input_frames = 6`).** Rejected: unfaithful
  to the paper's h = 0 and pointless for a quasi-static system; it would
  also shrink the scored span for no benefit.
- **No QoIs at v1.** Rejected in favour of the minimal engineering pair;
  they are declared definitions kept out of the blessing gate, so they cost
  nothing in reproduction rigor.
- **Tighter (14–16) or multi-horizon blessing gates.** Rejected: a
  reimplementation should not be held to a band tighter than the primary
  source's own reported spread; multi-horizon all-must-pass gating
  multiplies false-failure risk on run-to-run variance.
- **One-shot terminal-state task (Unisoma's shape).** Rejected: ADR-0041
  fixed the autoregressive-rollout family; the terminal state enters as a
  QoI instead.

## Consequences

- **Benchmark module** `benchmarks/deforming_plate/` (registry entry,
  frozen split lists over the converter's case ids, card with
  `discretisation = "FEM"`, `solver = "COMSOL"`, `n_frames = 400`,
  `input_frames = 2`, `horizon = "full"`, nominal `output_dt_ms = 1.0`, and
  a `data_license` string reflecting ADR-0042's no-redistribution posture:
  the data is downloaded from source, not rehosted).
- **Mesh-aware trajectory loader is a hard prerequisite** (flag-first
  `datasets/` change, its own plan): nodes-as-particles from a tetra mesh,
  `particle_type` from `nodes.node_type`, aux read directly from
  `response/node/von_mises_stress`, tetra connectivity carried through for
  MGN's mesh edges. The current loader is SPH-only on all three counts.
- **`models/mgn`** implements the §8 recipe; `cli/train.py` gains
  model-family dispatch (currently CGN-hardwired); both are next plans.
- **Card finalisation is gated on Task 8** (real-data measurement): physical
  units of positions/stress (the ×10⁻³ scale's physical meaning), node-count
  distribution and per-type counts, confirmation that node-type codes
  {0, 1, 3} are exhaustive and HANDLE nodes are stationary, trajectory-count
  and length uniformity, and the ADR-0032 §5 ground-truth timeline analysis
  behind the card's `protocol_rationale`. The protocol above does not move
  with these; the card's numeric fields and rationale text do.
- **The blessing run's compute cost is real** (batch 2 × 10M steps in the
  reference recipe); scheduling it is a maintainer decision, and any recipe
  deviation is recorded in the training ledger with the gate unchanged.
- The registries/leaderboard render this benchmark with the standard
  statistics; the pooled blessing aggregate appears in the blessing record
  (ADR-0033 registry `notes`/metrics), not as a leaderboard column.

---

**Dated note (2026-08-08, maintainer) — three measurement/primary-source
findings.**

1. **The hosted `train.tfrecord` carries 1,200 trajectories**, not the
   paper's stated 1,000 (paper A.1: "1000 training, 100 validation and 100
   test"; byte-size arithmetic independently corroborates 1,200). The §2
   protocol split is unchanged: **the protocol's 1,000 train trajectories are
   the first 1,000 in file order**, the converter caps train at 1,000 by
   default, and the extra 200 exist upstream but sit outside the protocol
   (they may serve as explicitly-extra-protocol data in future work).
2. **HANDLE (type-3) nodes are not strictly stationary in the data** (max
   drift ≈ 0.02 m in train, ≈ 0.06 m in valid/test), contrary to the paper's
   idealised description. §4 is unaffected — both kinematic types are
   GT-prescribed and excluded from scoring — and the §8/§9 scripted-velocity
   input stays OBSTACLE-only per the paper's own A.2 wording.
3. **§8 gate pooling correction** (from the primary PDF, A.5.2): the paper's
   rollout RMSE pools over "all spatial coordinates, **all mesh nodes**, all
   steps in each trajectory, and all 100 trajectories", with the ± being the
   standard error across trajectories. The §8 gate's aggregate is therefore
   computed over **all nodes** (kinematic rows are GT-prescribed and
   contribute zero error), NOT over NORMAL nodes only as §8's original
   wording said — NORMAL-only pooling would hold a reimplementation to a
   materially stricter bar than the published 15.1±4.0. The §5 leaderboard
   metrics keep StructBench's own NORMAL-only masking, which is unaffected.

---

**Narrowed by ADR-0046 (2026-08-09).** This ADR's Consequences said the pooled
blessing aggregate "appears in the blessing record (ADR-0033 registry
`notes`/metrics), not as a leaderboard column." ADR-0046's comparison renderer
surfaces **every** `metrics` key as a table row, so the pooled number's
containment is narrowed to **`notes` free-text only** — it must never be a
`metrics` key, or it would leak into the method-comparison columns and destroy
the comparability this ADR (§8/context) warns against. A deliberate tightening
of the letter above, not a reversal.
