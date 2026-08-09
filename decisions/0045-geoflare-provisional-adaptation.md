# 0045 — GeoFLARE provisional adaptation: native GALE_FA (GeoTransolver + FLARE) on the DeformingPlate rollout

**Status**: Proposed
**Type**: Durable
**Date**: 2026-08-09

## Context

ADR-0041 (clause 7) sets the v0.3 build order — ① ingestion + `DeformingPlate`
+ blessed MGN, ② Transolver provisional, ③ GeoFLARE provisional — and rules
that every method ships *native* under `models/` (no PhysicsNeMo runtime
dependency) and that Transolver and GeoFLARE ship *provisional*
(fidelity-vs-published deferred, flagged in the results registry). ADR-0043
fixed the DeformingPlate protocol — the pins every family obeys — and the MGN
blessing gate. ADR-0044 recorded the native Transolver adaptation and, in its
clause 12, established the seams GeoFLARE now inherits: the
`models/common/CaseBoundSimulator` base, and the point-set featurization /
per-example segment-batching patterns. This ADR is the ADR-0041 step ③
counterpart of ADR-0044 — the adaptation decisions for the native GeoFLARE
family (`models/geoflare`), now implemented on `feat/geoflare-provisional`.

**Identity.** "GeoFLARE" is not a separate model. It is **`GeoTransolver`
instantiated with `attention_type="GALE_FA"`** — GeoTransolver's
geometry-aware architecture (GALE = *Geometry-Aware Latent Embeddings*: a
multi-scale ball-query context pathway feeding a persistent cross-attention)
with its self-attention backend swapped from vanilla physics-attention to
**FLARE** (*Fast Low-rank Attention Routing Engine*): `M` learnable global
queries encode all `N` tokens, then each token's own key decodes them back
out — two attention passes, `O(NM)` instead of `O(N²)`. Three source papers
define it: GeoTransolver (Adams et al., NVIDIA, arXiv:2512.20399), FLARE
(Puri et al., arXiv:2508.12594), and the crash companion that formalizes the
FLARE-backend variant (Akhare, Nabian, Adams et al., arXiv:2605.27758).

**The naming landscape is inconsistent upstream, so the port fixes its own.**
The core paper string is "GeoTransolver with FLARE"; the crash paper's tables
call it "GeoTS-FLARE"; the physicsnemo code enum is `attention_type="GALE_FA"`
(the only accepted values are `"GALE"` and `"GALE_FA"` — anything else raises
`ValueError`); the README prose says "GeoFlare". **Upstream `GALE_FE` comment
bug:** the crash config sets `attention_type: "GALE_FA"` but the bumper config
sets `attention_type: "GALE_FE"`, and *both* carry the comment "GALE_FE for
GeoFLARE" — yet `GALE_FE` is not an accepted enum value and would raise
`ValueError` at construction. StructBench does not copy `GALE_FE`; it adopts a
single native name (`geoflare`) and records the mapping here.

The situation that shapes every decision below mirrors Transolver's:

1. **There is nothing to reproduce.** No published GeoFLARE (or vanilla
   GeoTransolver) `deforming_plate` rollout number exists. The reference
   validated GeoFlare only under **one-shot** prediction; its **autoregressive
   rollout was reported "unstable"** (bumper T=50, divergent past T=25). A
   next-step-trained model rolled out over DeformingPlate's long horizon
   inherits that drift risk, and no reference number anchors the comparison.
   This ADR therefore *declares* an adaptation; it does not certify a
   reproduction.

2. **The model math is a transcription job; the adaptation is the decision
   surface.** GALE_FA (FLARE encode/decode, GALE cross-attention, the mixing
   gate), the multi-scale ball-query context pathway, the pre-LN block, and the
   context tokenizer are all pinned against the physicsnemo code as shipped.
   What no source gives — featurization for a rollout task, ragged-batch
   handling, geometry frame and coordinate scaling, output/integration,
   stabilization, optimizer, and the definition of "done" — is what this ADR
   fixes. Where the physicsnemo code and the papers disagree, the **code is the
   fidelity target** (per-family fidelity: each port is faithful to its *own*
   upstream reference, which is why some pins differ numerically from the
   thuml-faithful Transolver port).

Everything below is grounded in a verified research pass
(`scratch/2026-08-09-geoflare-grounding.md` — maintainer-local scratch record,
gitignored and absent from clones; produced by verification workflow
wf_8ed06010-0d0: 180 extracted claims, 157 adversarially confirmed, 20
refuted-with-corrections; the §10 addendum was verified by a follow-up
raw-file-quoting agent; §-references point there). Numeric values are the
defaults now shipped in `config.py` `GeoFlareConfig` and
`configs/deforming_plate/geoflare.toml`.

## Decision

1. **Identity, native name, and the `GALE_FE` bug.** The family is
   `"geoflare"` (`MODEL_FAMILIES` key), `GeoFlareConfig`, `models/geoflare/`,
   classes `GeoFlareNet` / `GaleFlareAttention` / `GeoFlareSimulator` /
   `MultiScaleContext`. It *is* GeoTransolver + `attention_type="GALE_FA"`. The
   upstream `GALE_FE` string is a comment/config bug (not an accepted enum
   value) and is not reproduced. ADR-0041 carries a dated note pointing here.

2. **Scope: the full shipped-GeoFlare configuration, ball-query context
   pathway included.** The shipped crash/bumper configs set
   `include_local_features: true` (grounding §10), so ball query genuinely
   runs; a context-free FLARE-only port would leave the "Geo" vestigial and
   weaken the ADR-0041 cross-method headline. The port keeps the geometry
   tokenizer plus the two-scale ball-query context/local pathways, and drops
   two pieces the shipped config does not use: the **global tokenizer**
   (`global_dim=None` — DeformingPlate has no per-run design scalars) and the
   reference's **multi-stream input tuple** (StructBench always has exactly one
   functional stream — a declared single-stream simplification). Block
   structure is code-faithful: pre-LN residual attention, an FFN with its own
   `LayerNorm → Linear → GELU → Linear`, and a single `LayerNorm → Linear`
   decoder external to every block (no last-block decoder fold, no thuml-style
   `placeholder` vector — that is Transolver-only).

3. **Featurization: 18 channels, information-parity with MGN and Transolver.**
   The functional per-node input is `[one_hot(node_type, 9),
   scripted_velocity (3), x_t (3), reference_coords (3)]` — `node_type_size +
   3 * dim` = 18 for the ADR-0043 recipe — **byte-identical to Transolver's**,
   so the cross-method comparison is apples-to-apples. The geometry the
   ball-query pathway consumes is threaded to the network as a **separate**
   coordinate argument, not mixed into the functional tensor.

4. **Output: `(P, dim+1)` velocity-then-stress, forward-Euler.** The network
   co-predicts nodal velocity and von Mises stress in MGN's exact output
   layout; velocity is integrated once by forward Euler (`next = x_t +
   velocity`, quasi-static), stress is taken directly. The target normalizer is
   inverted on the full `(P, dim+1)` vector *before* slicing. The ADR-0043 §4
   NORMAL-masked loss (`w_pos·‖Δv‖² + w_aux·Δaux²`), the integration step, and
   the target normalizer are reused verbatim from the shared harness.

5. **GALE_FA attention, pinned to the shipped code.** Implemented as
   `GaleFlareAttention` (single-example; ragged batches handled by the caller,
   clause 10):
   - **Manual attention, scale = 1.0 at all three sites** (both FLARE passes
     *and* the GALE cross-attention). The reference hardcodes `scale = 1.0`
     with the comment that the FLARE authors recommend `dim_head**-0.5` for
     `dim_head > 8` "but we use 1.0 because the recommended scaling is not
     tested yet" — a known, flagged upstream deviation from FLARE's own
     guidance, reproduced faithfully. It is written as manual
     `softmax(q @ kᵀ · scale) @ v` (the house style of `models/transolver`) and
     **not** `F.scaled_dot_product_attention`: SDPA's `scale=` keyword requires
     torch ≥ 2.1, above the repo's declared `torch >= 2.0` floor, and SDPA's
     default scale (`dim_head**-0.5`) would silently produce the wrong math on
     a torch-2.0 install. Manual softmax has no version gate.
   - **`q_global = Parameter(randn(H, M, dim_head))`, std ~1.0, no init pass.**
     The reference uses plain PyTorch defaults everywhere and a plain
     `torch.randn` for the global queries — GALE_FA has no slice-projection
     layer to orthogonal-init and runs no global weight-init pass. GeoFLARE
     therefore does **not** reuse `TransolverNet`'s `_initialize_weights`
     (`trunc_normal_(0.02)`) pass, which would rescale `q_global` by ~50×.
   - **Parallel self- and cross-attention** from the same pre-attention stream.
     The paper prose reads *sequential* (self → then cross from the self
     output); the code computes both from the same projected `x`, then mixes.
     The port matches the code.
   - **Weighted mix `w·self + (1−w)·cross`, `state_mixing = Parameter(0.0)`,
     so σ(0) = 0.5 — a balanced 50/50 init.**

6. **Context tokenizer, pinned to physicsnemo's `ContextProjector` — and
   deliberately *different* from our Transolver port.** The classic
   Transolver-style slice tokenizer is reused purely as a tokenizer (no
   attention among its tokens). Two numeric details follow physicsnemo, not
   thuml, and this divergence is the per-family fidelity principle at work
   (each port faithful to its own reference): the slice **temperature is
   clamped to [0.5, 5]** (our thuml-faithful Transolver leaves it unclamped)
   and the **slice-norm epsilon is 1e-2** (Transolver's is 1e-5). The
   temperature parameter has shape `(heads, 1, 1)` — the per-head *semantics*
   is the pin; the shape follows this port's head-first `(H, N, S)` logits
   layout, where upstream's `(1, 1, H, 1)` belongs to its own batch-first
   `(B, N, H, S)` layout. The weighted token aggregation is written as an
   `einsum` that is **mathematically equivalent to the reference's
   matmul-with-permutes** — recorded as einsum-equivalent, never as a
   byte-level port of that matmul. (Grounding is firmly clear that GALE_FA runs
   no explicit init; it is *ambiguous* specifically on whether the
   `ContextProjector`'s own slice-projection carries an orthogonal init. The
   port pins **no** orthogonal init there too. This is functionally immaterial
   — the weights are learned — but is noted honestly rather than claimed as
   settled.)

7. **Ball query: deterministic nearest-first, absolute coordinates,
   zero-padded.** Per example, distances via chunked `torch.cdist`, then per
   query row the `k` smallest within `radius` (`torch.topk(largest=False)`,
   which returns ascending — nearest first), with the radius applied as a hard
   `<=` cutoff on the top-`k` selection (a point just outside `radius` is
   zero-padded even when a slot remains). A point is its own distance-0
   neighbour (no self-exclusion). Fewer than `k` qualifying neighbours →
   coordinate rows `(0, 0, 0)`. Gathered features are **absolute** neighbour
   coordinates (no offsets/centering). physicsnemo's accelerated Warp backend
   returns neighbours in an **arbitrary** order (hash-grid traversal, early
   break; upstream disclaims relying on the order); StructBench pins the
   deterministic torch-fallback's nearest-first order as its own declared
   semantics. Each `GeometricFeatureProcessor` flattens the `(N, k, 3)` result
   and runs the reference MLP **`[3K, 32, 16, 32]`** (GELU between layers, none
   after the last) with **`tanh` applied *outside* the MLP stack**.

8. **Two `dim_head` values — the likeliest porting bug, pinned apart.** At the
   defaults (`n_hidden=256`, `n_heads=8`, `n_hidden_local=32`, 2 scales): the
   block/attention `dim_head = effective_hidden // n_heads = 320 // 8 = 40`
   (where `effective_hidden = n_hidden + n_hidden_local·2 = 320`), but the
   context-tokenizer `dim_head = n_hidden // n_heads = 256 // 8 = 32`. These are
   different numbers; conflating them silently mis-sizes the whole stack. The
   context tensor is `3 · 32 = 96` wide (geometry tokenizer + 2 ball-query
   scale parts, no global part), and `cross_k`/`cross_v` are `Linear(96, 40)`.

9. **Context part order is code-faithful (scales first, geometry last), with a
   paper-vs-code discrepancy noted.** The raw `build_context` runs its
   local-extractor loop (`context_parts.extend(...)`) before appending the
   geometry context (`context_parts.append(...)`), giving `[scale_1, scale_2,
   geometry]`. The paper's Eq (10) lists geometry *first*
   (`concat([Z_geo, Z_glob, Z_1..Z_Ns])`). Paper and code disagree; the port's
   fidelity target is the code. The order is functionally absorbed by the
   `cross_k`/`cross_v` `Linear` over the full context width either way, so the
   choice is invisible downstream — recorded for honesty, not because it
   changes results.

10. **Ragged-N batching: one per-example segment loop wrapping the *whole*
    stack.** GeoFLARE reuses Transolver's `_segments` primitive but, unlike
    `TransolverNet` (which loops per attention call), loops once around the
    entire `preprocess → blocks → decoder` pipeline. The reason is the geometry
    context: it is a per-example ball-query construction over *one* example's
    own coordinate cloud (a ball query across concatenated meshes would find
    spurious cross-example "neighbours"), it dominates a GALE_FA block in cost
    at DeformingPlate's ~1.3k-node scale, and it is reused unchanged by every
    block — so the simplest correct shape loops segments once and rebuilds the
    context per example. The batched forward equals the concatenation of
    per-example forwards exactly (no padding, no masks); the killer test
    enforces it, and a segment-leak mutation pin guards the encode softmax over
    `N`.

11. **Declared adaptation — per-example coordinate standardization; radii in
    standardized units.** The reference radii `[0.05, 0.25]` live in
    position-*normalized* space (the crash pipeline normalizes positions with
    fixed train-split stats before all rollout math). This port instead
    standardizes the geometry coordinates **per example** — `g = (x − mean(x))
    / rms(x)`, a scalar-isotropic scale to exact zero mean and unit population
    RMS — and keeps the radii `0.05`/`0.25` in that standardized frame, with
    neighbour caps `[8, 32]` bounding the effect. Deviation from the reference's
    dataset-stats normalization, chosen because per-example standardization is
    self-contained, scale-free, and needs no stats plumbing (checkpoints stay
    self-contained). The same standardized coordinates feed both the geometry
    tokenizer and the ball-query MLPs. Ledgered caveat: under deformation the
    per-example scale shifts, so the *effective* physical radius adapts
    frame-to-frame — a documented consequence of the choice.

12. **Declared adaptation — current-frame geometry, rebuilt every forward.**
    The ball-query geometry is the example's **current** positions (train: the
    noised `x_last`; eval: the rolled-out `x_t`), rebuilt every forward — never
    a fixed reference frame. This matches every reference call site (the crash
    AR rebuilds context from live predicted coordinates each step,
    `geometry=coords`). Cost at ~1.3k nodes is negligible. The coordinates are
    threaded to the network **raw** (working-frame mm); standardization
    (clause 11) happens inside `MultiScaleContext`, per example — the
    coordinates are *not* run through the `OnlineNormalizer`.

13. **Declared adaptation — GNS noise `σ = 3e-3`, NORMAL-masked, γ = 1;
    `OnlineNormalizer` for features and targets.** The reference example
    injects **no** noise; StructBench applies the ADR-0043 zero-mean Gaussian
    noise to the last input position of NORMAL nodes only, measuring the
    velocity target from the noisy position so the first-order correction falls
    out as γ = 1 (MGN/Transolver parity). Node features and targets use the
    harness's streaming `OnlineNormalizer` (with `normalizer_warmup_steps`),
    keeping checkpoints self-contained — a deliberate, harness-consistent
    deviation from the reference's offline train-split stats.

14. **Declared adaptation — harness optimizer, deliberately not Muon; matched
    budget.** The recipe is **AdamW + `_lr_at_cosine`** (cosine anneal from
    `lr_init = 1e-4` — the reference crash-example `start_lr` — down to
    `LR_SCHEDULE_FLOOR = 1e-6`; the reference ends at `3e-7`, a declared
    difference), with **`weight_decay = 1e-4`** (the reference's AdamW-arm
    value) and gradient clipping **off by default** (`max_grad_norm = 0.0`,
    the knob kept for parity with `TransolverConfig`; the reference example
    applies no clipping). The reference's headline rows used **Muon** (2-D
    params) via a combined optimizer — deliberately **not** adopted, because
    `torch.optim.Muon` requires PyTorch ≥ 2.9, a flag-first dependency bump
    above the repo's `torch >= 2.0` floor; provisional latitude permits matching
    the harness and skipping the bump. `weight_decay` and `max_grad_norm` are
    fields of `GeoFlareConfig`, **not** `TrainConfig` (the `MGNConfig.noise_std`
    precedent — family-recipe knobs stay off the strict, family-uniform
    `[train]` schema; `[train].lr_decay` is present-but-unused for this cosine
    family). The reference budget is matched to MGN/Transolver for
    comparability — **batch 2, 10 M steps, `val_every = 50k`** — budget only, not
    optimizer: the comparison is same-task / same-data / same-budget, not
    same-optimizer.

15. **Provisional means no numeric gate; the comparison statistic is the
    ADR-0043 §5 leaderboard; drift risk is a standing note.** Because no
    published GeoFLARE `deforming_plate` rollout number exists, there is **no
    acceptance gate** analogous to the MGN blessing gate. "Done" for the
    implementation is: all tasks review-clean, the full suite green, and the
    smoke config training and evaluating end-to-end. The eventual training run
    is maintainer compute; its number is recorded **provisional**, flagged in
    the registry, and never read as blessed. Cross-method comparison uses the
    ADR-0043 §5 per-step-mean leaderboard metrics `evaluate()` already emits for
    every family. **Drift-risk note:** the reference validated GeoFlare *only*
    one-shot; its AR rollout was "unstable" at T=50 with no reported MSE, and
    divergent past T=25. A next-step-trained GeoFLARE rolled out over
    DeformingPlate's long horizon inherits that risk; our mitigations (GNS
    noise, the quasi-static task) are declared, but no reference number bounds
    the long-rollout behaviour.

16. **Shared base: `models/common/CaseBoundSimulator` (ADR-0044 clause 12).**
    `GeoFlareSimulator` subclasses the same per-case state base as
    `MeshSimulator` and `TransolverSimulator`; `predict_positions` and
    `forward_train` stay per-family. The `evaluate()` gate already generalizes
    on `isinstance(CaseBoundSimulator)`. GeoFLARE inherits the point-set
    featurization and segment-batching patterns rather than re-deriving them.

17. **Attribution: Apache-2.0 credit at docstring level.** StructBench is
    Apache-2.0 and the physicsnemo reference is Apache-2.0, so the licences are
    compatible with no `NOTICE`-file obligation. The three modules that carry
    ported math — `models/geoflare/network.py` (GALE_FA / FLARE), `context.py`
    (the GALE context pathway), and `geo_ops.py` (the ball query) — each credit
    "NVIDIA PhysicsNeMo (Apache-2.0 License, Copyright (c) NVIDIA CORPORATION &
    AFFILIATES)" in their module docstrings, with the FLARE paper
    (arXiv:2508.12594) cited in `network.py`. `simulator.py` and `__init__.py`
    are StructBench-original harness/wiring and carry no such credit.

18. **Two process notes recorded here.**
    - **Cross-module private import of `_segments`.** `network.py` imports
      `_segments` and `build_mlp_2layer` from
      `structbench.models.transolver.network`. `_segments` is private-by-
      convention to its own module (the PRINCIPLES.md `_`-prefix boundary); this
      cross-module import is a deliberate, brief-directed reuse exception (the
      ragged-batch primitive is genuinely shared cross-family), not an
      oversight, and this ADR is its place of record.
    - **`test_geoflare_network.py` filename.** The plan named the network test
      `test_network.py`, but that basename already exists under
      `tests/models/transolver/`, and with no `tests/__init__.py` anywhere in
      the tree pytest's default import mode collides on duplicate basenames. It
      was renamed `test_geoflare_network.py`, matching the existing
      `test_mgn_network.py` precedent — family-prefixed `test_<family>_network.py`
      is worth adopting as the house convention for every family after MGN.

## Alternatives considered

- **FLARE-only, context-free port** (`geometry_dim=None`, `global_dim=None`,
  `include_local_features=False` ⇒ `context=None` ⇒ the cross-attention branch
  and all ball-query/Warp code never run). Trivial to port, but the shipped
  GeoFlare config sets `include_local_features: true`, so a context-free
  variant would not *be* "GeoFLARE" — the "Geo" would be vestigial and the
  ADR-0041 cross-method headline weaker. Rejected in favour of the full port.
- **Muon + a torch ≥ 2.9 bump** to match the reference's headline optimizer.
  Rejected: `torch.optim.Muon` needs PyTorch ≥ 2.9, a flag-first dependency
  bump above the repo's `torch >= 2.0` floor; provisional latitude lets the port
  match the AdamW+cosine harness and skip the bump. The declared cost: the
  reference's best numbers were Muon rows.
- **Dataset-stats coordinate normalization** (fixed train-split position
  mean/std, the reference's scheme). Rejected: per-example standardization is
  self-contained, scale-free, and needs no stats plumbing, keeping checkpoints
  self-contained. The effective-radius-under-deformation caveat is the accepted
  cost (clause 11).
- **Padding + attention/slice masks for ragged batching.** Rejected: no masking
  exists upstream (it ran batch = 1), so it would be authored from scratch with
  its own correctness burden; the per-example segment loop is mathematically
  exact, needs no padding, and reuses `collate_mesh_samples` unchanged.
- **`F.scaled_dot_product_attention`** instead of manual softmax. Rejected: its
  `scale=` keyword requires torch ≥ 2.1, and the non-default `scale = 1.0` pin
  would silently be wrong under SDPA's default scale on a torch-2.0 install.
  Manual attention has no version gate (clause 5).
- **Acceleration output + symplectic-Euler / Verlet double integration** (the
  crash example's parametrization). Rejected: DeformingPlate is quasi-static,
  and MGN's blessed recipe predicts velocity integrated once. Velocity +
  forward-Euler maximizes harness reuse and matches the blessed baseline's
  parametrization on the same task.
- **The reference's multi-stream input tuple machinery.** Rejected: StructBench
  always has exactly one functional stream, so the tuple plumbing would be dead
  weight — a declared single-stream simplification (clause 2).

## Consequences

- **New code surface:** `models/geoflare/` (`geo_ops.py` — deterministic ball
  query + per-example standardization; `context.py` — multi-scale ball-query
  context + slice tokenizers; `network.py` — `GaleFlareAttention` /
  `GeoFlareBlock` / `GeoFlareNet` with the segment-exact driver; `simulator.py`
  — the stateful rollout wrapper), `GeoFlareConfig` + the `"geoflare"`
  `MODEL_FAMILIES` entry in `config.py`, and `_train_geoflare` /
  `build_geoflare_simulator` / the `evaluate()` geoflare arm (plus the
  per-family type widenings) in `cli/train.py`, with the reference and smoke
  configs under `configs/deforming_plate/`.
- **MGN and Transolver are unchanged.** GeoFLARE reuses their utilities —
  `OnlineNormalizer` (from `models/mgn`), `build_mlp_2layer`/`_segments` (from
  `models/transolver`), `collate_mesh_samples`/`mesh_static_from_trajectory`
  (from `models/mgn`), and the `CaseBoundSimulator` base — without modifying
  them. The harness noise block and cosine schedule are independent per-family
  clones, not shared code paths.
- **No new runtime dependency.** The family is pure `torch`; `pyproject`'s
  dependency set and mypy overrides are untouched (no `einops`, `timm`,
  `jaxtyping`, `transformer_engine`, or `warp`).
- **The comparison view and the `provisional` registry flag are a separate
  plan.** This ADR fixes the method adaptation, not the results-registry schema
  or the landing-page comparison table; a training run is maintainer compute.
- **The fidelity debt is explicit and localized.** The first things a future
  fidelity check should probe: the long-rollout drift risk (clause 15,
  one-shot-only reference validation), and the two per-family divergences from
  Transolver (clamped temperature + eps 1e-2, clause 6) — each faithful to its
  own upstream, but the pair is where the two ports differ.
- **Attribution is settled at docstring level** (Apache-2.0 ↔ Apache-2.0, no
  `NOTICE` obligation).
- **Naming is recorded on ADR-0041** via a dated note, and in this ADR's
  clause 1.
