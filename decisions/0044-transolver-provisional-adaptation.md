# 0044 — Transolver provisional adaptation: native Physics-Attention on the DeformingPlate rollout

**Status**: Proposed
**Type**: Durable
**Date**: 2026-08-09

## Context

ADR-0041 (clause 7) sets the v0.3 build order — ① ingestion + `DeformingPlate`
+ blessed MGN, ② Transolver provisional, ③ GeoFLARE provisional — and rules
that Transolver ships *native* under `models/` (no PhysicsNeMo runtime
dependency) and *provisional* (fidelity-vs-published deferred, flagged in the
results registry). ADR-0043 fixed the DeformingPlate protocol — the pins every
family obeys — and the MGN blessing gate. This ADR records the adaptation
decisions for the native Transolver family (`models/transolver`), which are
now implemented on `feat/transolver-provisional`; it is the ADR-0041 step ②
counterpart of ADR-0043's blessing recipe, but for a method with *no*
reproduction target.

The situation that shapes every decision below:

1. **Transolver is off-native here, and there is nothing to reproduce.**
   Transolver (Wu et al., ICML 2024, arXiv:2402.02366) is a steady-state /
   parametric operator (physics-attention over slice tokens). Its own paper is
   single-shot (Elasticity regression; Plasticity time-*query*, not rollout),
   and the official `thuml/Transolver` repo contains **zero** autoregressive /
   rollout / recurrent code (verified empty set). There is **no published
   vanilla-Transolver `deforming_plate` rollout number** to reproduce — every
   deforming_plate transformer number in the literature is a hybrid
   (MPNN-wrapped attention) or a non-physics-attention transformer. ADR-0041
   deliberately kept StructBench's autoregressive-rollout family rather than
   adding an operator-learning modality where Transolver is native, accepting
   off-native operation as the price of a contained release. This ADR
   therefore *declares* an adaptation; it does not certify a reproduction.

2. **The model math is a transcription job; the adaptation is the decision
   surface.** Physics-Attention (Eqs 1–4: slice, token aggregation, token
   attention, deslice) and the pre-LN block (Eq 6) are fully pinned against
   both the paper and the thuml irregular-mesh code, down to every code-only
   detail. What is *not* given by any source — featurization for a rollout
   task, ragged-batch handling through global attention, output/integration,
   stabilization, optimizer, and the definition of "done" — is what this ADR
   fixes.

Everything below is grounded in a verified research pass
(`scratch/2026-08-08-transolver-grounding.md` — maintainer-local scratch
record, gitignored and absent from clones; produced by verification workflow
wf_47ff0633-b84: 147 extracted claims, 136 adversarially confirmed, 10
refuted-with-corrections; §-references point there). Numeric values are the
defaults now shipped in
`config.py` `TransolverConfig` and `configs/deforming_plate/transolver.toml`.

## Decision

1. **Featurization: 18 channels, information-parity with MGN.** The per-node
   input is `[one_hot(node_type, 9), scripted_velocity (3), x_t (3),
   reference_coords (3)]` — `node_type_size + 3 * dim` channels, i.e. 18 for
   the ADR-0043 recipe (`node_type_size = 9`, `dim = 3`). The information-parity
   argument: Transolver receives the *same* one-hot node type and
   scripted-actuator velocity features MGN uses; the geometry MGN obtains
   through its dynamically rebuilt edges instead enters as absolute current
   position `x_t` plus material `reference_coords` — Transolver's native
   convention (its Elasticity variant regresses from raw coordinates,
   `fun_dim = 0`). `reference_coords` comes from schema 0.2.0's per-node field
   (ADR-0042). No history velocity is fed: the source task is `h = 0`
   (grounding §5.4/C21), so `x_last = position_seq[:, -1]` exactly as MGN. The
   cost of edge-free featurization — contact learned from coordinates alone —
   is a declared fidelity risk (clause 13).

2. **Output: `(P, dim+1)` velocity-then-stress, forward-Euler.** The network
   co-predicts nodal velocity and von Mises stress in MGN's exact output
   layout; velocity is integrated once by forward Euler (`next = x_t +
   velocity`, no Δt, quasi-static), stress is taken directly. The target
   normalizer is inverted on the full `(P, dim+1)` vector *before* slicing
   (denorm-before-slice, grounding c32). Matching MGN's parametrization is
   deliberate: the ADR-0043 §4 loss (NORMAL-masked
   `w_pos·‖Δv‖² + w_aux·Δaux²`), the integration step, and the target
   normalizer are then reused verbatim.

3. **Stabilization: GNS noise σ = 3e-3, NORMAL-masked, γ = 1 by
   construction.** Zero-mean Gaussian noise of `noise_std = 0.003` is added to
   the last input position of NORMAL nodes only — the field's revealed
   preference on deforming_plate (grounding §5.3/C24), not pushforward, not
   AR-RT. The noise is applied to `x_last` by the caller, and the velocity
   target is measured from that noisy position, so the first-order target
   adjustment (grounding C22) *falls out* as γ = 1 with no separate correction
   term (matching the MGN recipe, grounding c34). `_train_transolver` mirrors
   `_train_mgn`'s noise block line-for-line (an independent copy, config
   substitution `mgn.noise_std` → `cfg.noise_std` only); the semantics are
   identical — NORMAL-masked, γ = 1 by construction.

4. **Ragged-N batching: per-example segment computation, mathematically
   identical to thuml batch = 1.** This is the single largest correctness risk
   of the port: physics-attention's slice softmax and token pooling sum over
   the whole `N` axis, so a naive flat `(ΣP, C)` batch (StructBench's
   `collate_mesh_samples` layout) would silently pool across trajectories.
   thuml has no masking and ran the irregular-mesh variant at batch = 1. The
   solution: slice-weight computation is pointwise and safe on the flat tensor
   (softmax over the slice axis does not mix rows); the parts that *do* sum
   across points — token aggregation (Eq 2), token attention (Eq 3), and
   deslice (Eq 4) — run per contiguous example segment, keyed on
   `n_particles_per_example`, in a Python loop (B ≤ 8, M = 64 — loop cost
   negligible). The batched result equals the concatenation of per-example
   forwards *exactly* (no padding, no masks); the killer test
   (`test_batched_forward_matches_per_example`) enforces it.

5. **Normalization: `OnlineNormalizer`, not thuml's precomputed
   `UnitTransformer`.** thuml fits a `UnitTransformer` (per-channel mean/std
   over the train split, computed once up front) and decodes before the loss.
   StructBench instead uses the harness's streaming `OnlineNormalizer` (with a
   `normalizer_warmup_steps = 1000` accumulation window) for both node features
   and targets. This is a deliberate, harness-consistent deviation from the
   released code: it keeps checkpoints self-contained — the four normalizer
   buffers live in the `state_dict`, so evaluation needs no separate
   `normalization_stats.npz` (grounding c23) — and gives Transolver the same
   warmup semantics MGN already has. Recorded here as a departure from
   released-code fidelity, made for harness uniformity.

6. **Optimizer recipe on `TransolverConfig`, not `TrainConfig`.** The
   method-native recipe is **AdamW(weight_decay = 1e-5) + global-norm gradient
   clip 0.1 + cosine anneal** of the learning rate from `lr_init` down to
   `LR_SCHEDULE_FLOOR` over `training_steps` — a steps-port of the Transolver
   Elasticity reference's per-epoch `CosineAnnealingLR` (grounding §4.1),
   implemented as `_lr_at_cosine`. `weight_decay` and `max_grad_norm` are
   fields of `TransolverConfig`, **not** `TrainConfig`, and this placement is a
   decision, not an accident: the `[train]` section has a strict,
   family-uniform schema, so adding optimizer fields there would (a) break
   every existing TOML the strict loader validates and (b) impose Transolver's
   knobs on families that do not use them. Family-recipe knobs belong on the
   model config — the precedent is `MGNConfig.noise_std` and
   `MGNConfig.normalizer_warmup_steps`. `[train].lr_decay` is
   present-but-unused for this family (it drives the exponential `_lr_at`
   schedule of CGN/MGN, not the cosine one) and is commented as such in the
   config; the schema stays family-uniform.

7. **Matched budget, deliberately not matched optimizer.** The reference
   budget is matched to MGN for comparability — **batch 2, 10 M steps**
   (ADR-0043 §8) — *budget only*. `lr_init = 1e-3` is Transolver's own
   Elasticity-reference learning rate, deliberately **not** matched to MGN's
   `1e-4`: each method keeps its native optimizer recipe (Transolver's
   AdamW + cosine vs MGN's Adam + exponential decay). The cross-method
   comparison is therefore **same-task / same-data / same-budget, not
   same-optimizer**. Holding the LR or optimizer common would advantage
   whichever method the shared recipe happened to suit and misrepresent each
   method's own reported practice.

8. **Fidelity target: faithful to released code, not to apparent intent.** The
   port replicates the thuml irregular-mesh variant exactly, including its
   quirks (grounding §2.3/§3.3):
   - `in_project_slice` receives an `orthogonal_` init that is then
     **overwritten** by the global `trunc_normal_(std = 0.02)` + zero-bias pass
     in `TransolverNet._initialize_weights` (called at the end of `__init__`).
     The
     `orthogonal_` call is *kept and documented*, not silently dropped — the
     released behaviour is the trunc-normal weight, and reproducing the
     ordering quirk is the point.
   - Learnable per-head slice **temperature init 0.5, unclamped** (the
     irregular-mesh variant; the structured-mesh variants clamp to [0.1, 5]).
   - The learned **placeholder** vector is added **unconditionally** after the
     preprocess MLP (irregular-mesh behaviour; the structured variant's
     `else`-nested placeholder — never applied when a feature field is passed —
     is a known upstream inconsistency, deliberately not copied).
   - `mlp_ratio = 1` (no FFN expansion), **GELU** activation, **dropout 0.0**.
   - **No `Time_Input` path** (the rollout task carries no absolute-time
     query; time enters through the autoregressive step, not a sinusoidal
     embedding).
   "Faithful to released code" ≠ "faithful to apparent intent"; where they
   diverge (the init overwrite, the unclamped temperature) the released
   behaviour wins, so a later fidelity check compares against what thuml
   actually runs.

9. **Pure `torch` — no `einops`, no `timm`.** The reference's sole `einops`
   call is a head-merge `rearrange('b h n d -> b n (h d)')`, reproduced here by
   choosing the head-attention `einsum` output subscript order followed by a
   plain `reshape`; `timm`'s `trunc_normal_` is `torch.nn.init.trunc_normal_`
   (native since torch 1.10). No new dependency and no `pyproject` mypy
   override are introduced. (This is a no-new-dependency choice, not a
   prohibition — ADR-0041 does not ban `einops`; grounding §7.1.)

10. **Provisional means no numeric gate.** Because no published
    vanilla-Transolver deforming_plate rollout number exists (grounding
    §5.1/C18 — the verified empty set: no autoregressive code upstream, no
    reproduction target), there is **no acceptance gate** for Transolver
    analogous to ADR-0043 §8's MGN blessing gate. "Done" for the
    implementation is: all tasks review-clean, the full test suite green, and
    the smoke config training and evaluating end-to-end. The eventual training
    run is maintainer compute; its number is recorded **provisional**, flagged
    in the registry, and never read as a blessed baseline.

11. **Comparison statistic: the ADR-0043 §5 leaderboard metrics; the pooled
    tool is informational.** The ADR-0041 cross-method comparison uses the
    §5 per-step-mean leaderboard metrics that `evaluate()` already emits for
    every family — one statistic held consistent across blessed MGN and
    provisional Transolver (ADR-0043 warns that conflating conventions destroys
    comparability). The pooled blessing-convention aggregate
    (`tools/blessing_pooled_rmse.py`) runs on any run directory and is recorded
    as **informational** for Transolver, never as a gate.

12. **Shared base: `models/common/CaseBoundSimulator`.** The method-agnostic
    per-case state machinery — `bind_case` / `reset_rollout`, the GT tripwire,
    the scripted-velocity helpers, `save` / `load` — was extracted **move-only**
    from `MeshSimulator` into `models/common/simulator_base.py`;
    `MeshSimulator` and `TransolverSimulator` both subclass it (MGN behaviour
    bit-identical, its full suite unchanged). `predict_positions` and
    `forward_train` stay per-family (the graph-vs-point-set logic differs; the
    shared risk was the state machinery). The `evaluate()` gate widens from
    `isinstance(MeshSimulator)` to `isinstance(CaseBoundSimulator)`. GeoFLARE
    (step ③) reuses this base.

13. **Two documented fidelity risks.** These are the concrete content of the
    fidelity debt ADR-0041 anticipated:
    - **Slice-weight collapse.** Transolver++ frames adaptive temperature +
      Gumbel-Softmax as a *defect correction* of the original, not merely a
      scaling upgrade: the original's slice weights can collapse toward uniform
      → average pooling → "homogeneous physical states" (grounding §7.3/C9).
      The faithful original therefore *may* degenerate on this task; whether it
      matters at deforming_plate's node scale (~672–2189 nodes, mean ~1270) is
      unverified.
    - **Contact learned from absolute coordinates.** DeformingPlate
      deformation is entirely contact-driven, but vanilla Transolver has no
      edge or proximity mechanism (ADR-0043 §8 notes MGN's dynamics enter
      through relative edge features and world edges within `r_W = 0.03`).
      Actuator–plate contact and the OBSTACLE's next-step motion must be
      learned from absolute coordinates and the scripted-velocity feature
      alone. An accepted, documented off-native cost of the provisional
      adaptation.

14. **Attribution: MIT credit at docstring level; a repo-root NOTICE is an
    open question for the maintainer.** The port credits Wu et al., ICML 2024
    (arXiv:2402.02366) and the reference implementation
    (github.com/thuml/Transolver, MIT License, Copyright (c) 2024 THUML @
    Tsinghua University) in the `models/transolver/network.py` module docstring.
    Whether a repo-root `NOTICE` file is *also* required — the StructBench repo
    is Apache-2.0 and has no NOTICE / attribution precedent yet, and MIT
    requires its notice travel with substantial ported portions — is routed to
    the maintainer: a repo-root file is flag-first, and LICENSE-adjacent
    changes are out-of-session per the CLAUDE.md forbidden tier. No such file
    exists at the repo root today.

## Alternatives considered

- **Acceleration + Verlet integration** (the crash-paper precedent, Nabian et
  al. 2026, the one vanilla-Transolver mesh-rollout precedent). Rejected:
  deforming_plate is quasi-static, and MGN's own recipe predicts velocity
  integrated once. Velocity + forward-Euler maximizes harness reuse and matches
  the blessed baseline's parametrization on the *same* task.
- **AR-RT / BPTT training** (the crash precedent's stabilization: multi-step
  rollout of the model's own predictions, gradients through the whole rollout,
  per-step gradient checkpointing). Rejected for v0.3: it demands new
  training-loop machinery outside provisional scope, and the field's
  deforming_plate revealed preference is plain GNS noise (grounding §5.3), not
  AR-RT.
- **Padding + attention/slice masks for ragged batching.** Rejected: no masking
  exists upstream, so it would be authored from scratch and carry its own
  correctness burden; segment computation is mathematically exact, needs no
  padding, and reuses `collate_mesh_samples` unchanged.
- **Optimizer fields on `TrainConfig`.** Rejected: it breaks the strict,
  family-uniform `[train]` schema (the loader rejects unknown keys, so every
  existing TOML would fail) and imposes Transolver's knobs on other families.
  Family-recipe knobs belong on the model config (the `MGNConfig` precedent).
- **Matching MGN's optimizer and learning rate** (Adam + exponential decay,
  `lr = 1e-4`) for a same-optimizer comparison. Rejected: each method should
  keep its own reported recipe; a shared optimizer would advantage whichever
  method it suited. The comparison is held at same-budget, not same-optimizer.
- **Faithful-to-apparent-intent init** (keep the `orthogonal_` init on
  `in_project_slice` rather than letting `trunc_normal_` overwrite it).
  Rejected: released-code fidelity is the reproducibility target; the overwrite
  is the behaviour thuml actually runs, so it is reproduced and documented, not
  quietly corrected.
- **Adding `einops`** for the head-merge. Rejected: the single `rearrange` is
  trivially native, so no dependency or mypy override is warranted (a
  legitimate option per ADR-0041, declined on cost).
- **Validating Transolver now** against a published number. Rejected upstream
  by ADR-0041: no vanilla-Transolver deforming_plate rollout number exists, so
  there is nothing to validate against; the `provisional` flag manages the
  fidelity debt until a scheduled fidelity check.

## Consequences

- **New code surface:** `models/common/` (the `CaseBoundSimulator` base),
  `models/transolver/` (`network.py` — Physics-Attention with segment-exact
  ragged batching; `simulator.py` — the stateful rollout wrapper),
  `TransolverConfig` + the `"transolver"` `MODEL_FAMILIES` entry in
  `config.py`, and `_train_transolver` / `_lr_at_cosine` /
  `build_transolver_simulator` / the `evaluate()` transolver arm in
  `cli/train.py`, plus the reference and smoke configs under
  `configs/deforming_plate/`.
- **MGN is unchanged.** The base extraction is move-only; the full existing
  suite passes and `MeshSimulator`'s public API is frozen.
- **No new runtime dependency.** The family is pure `torch`; `pyproject`'s
  dependency set and mypy overrides are untouched.
- **The comparison view and the `provisional` registry flag are a separate
  plan.** This ADR fixes the method adaptation, not the results-registry schema
  or the landing-page comparison table; a training run is maintainer compute.
- **The fidelity debt is now explicit and localized.** Clause 13's two risks
  are the first things a future fidelity check (against a Transolver number
  established elsewhere, or a PhysicsNeMo cross-check) should probe.
- **Attribution is settled at docstring level; the NOTICE-file question is
  pending the maintainer.**
- **GeoFLARE (step ③) inherits the seams established here** — the
  `CaseBoundSimulator` base and the point-set featurization / segment-batching
  patterns — rather than re-deriving them.
