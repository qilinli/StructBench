# 0049 — Taylor native recipe repair: noise rescale, velocity history, MGN stretch gate

**Status**: Accepted (maintainer, in-session 2026-08-13)
**Type**: Durable
**Date**: 2026-08-13

## Context

The ADR-0047 provisional runs completed 2026-08-13. Two of three landed
useful results (val/test_interp/test_extrap rollout position RMSE, mm):
Transolver 0.64/2.02/5.53 and GeoFLARE 0.70/1.45/6.87 against blessed CGN
0.95/1.27/7.65 — the natives win the stress field everywhere and the
extrapolation RMSE, CGN keeps the in-distribution position lead and the
extrapolated geometry QoIs. MGN failed to train: best checkpoint at step
4k of 100k, training-val rollout drifting from 5.2 to ~7 mm while the
one-step train loss kept falling (2.76 → ~0.2, no NaN, no clipping) —
one-step learning continued while rollouts destabilized.

A same-day diagnosis session established three mechanisms, two measured:

1. **The synthesized lattice is destroyed by the physics.** Measured on
   T-20-80-150 ground truth over the ADR-0047 mesh: by 300 µs, 10% of mesh
   edges exceed 2× rest length, p99 stretch is 5.8×, max 31× (0.5 mm →
   15.6 mm) in the mushroom foot. The notch-impact beam mesh — where the
   same MGN recipe trains normally (val 0.22 mm at ~100k as of this ADR) —
   maxes at 1.46×. MGN is the only family whose messages consume the mesh
   (edge features `[u_ij, |u|, x_ij, |x|]`); the edge-free Transolver and
   GeoFLARE trained fine on identical data and noise. A fixed-connectivity
   mesh over an SPH flow is the wrong inductive bias at Taylor deformation
   levels, and grossly-stretched edges feed far-out-of-distribution features
   into message passing on exactly the material that matters.
2. **Noise is under-scaled ~7×.** The native configs ported DP's
   `noise_std = 0.003` (quasi-static); Taylor moves ~0.3 mm/frame at
   150 m/s, so that is ~1% of a step, against the blessed Taylor CGN's
   0.02 (~7%, random-walk). The config comments already flagged this as an
   open ledger item. Under-noised one-step training never teaches recovery
   from rollout drift — the classic MGN/GNS stabilizer, disabled in
   practice.
3. **The native adaptation is Markovian in position.** The ADR-0044/0045
   feature builder consumes only the window's last frame
   (`[one_hot, scripted_velocity, x_t, reference_coords]`);
   `input_frames = 6` affects windowing and rollout seeding only. CGN, by
   contrast, consumes the full 5-velocity GNS history. A model that cannot
   see momentum has no direct signal that a 200 m/s bar carries more of it
   than the 100–160 m/s training band — consistent with the observed
   extrapolation under-commitment (Transolver's best-of-field extrap RMSE
   comes from under-mushrooming; its mushroom-width QoI is 2–3× worse than
   CGN's). This was survivable on DeformingPlate (quasi-static, scripted
   loading) and on Taylor in-distribution (scoring starts at first wall
   contact, frame ~7, after which the deformed state encodes the velocity),
   but it is the prime suspect for the extrapolation gap.

## Decision

1. **Three recipe knobs, all config-gated, all default-off/reference.** No
   protocol change, no card change, no canonical-data change; every ADR-0047
   run record re-evaluates identically (old records lack the new keys and
   dataclass defaults preserve behaviour; default-off state dicts are
   width-identical).

   - **Noise rescale** (config-only): Taylor native arms adopt the blessed
     CGN Taylor `noise_std = 0.02`. Closes the ledger item.
   - **`velocity_history`** (all three native families): when true, the
     window's `input_frames − 1` finite-difference velocities are appended
     to the node features (`history_velocities · dim` extra channels), and
     the training noise switches from single-frame Gaussian to the CGN
     random-walk over the full window, so the velocity features are
     consistently noisy (`_mesh_family_noise`, shared by the three loops).
     Rollout builds the same feature from the sliding prediction window.
     This is CGN-parity momentum awareness, not an architecture change.
     **Target convention (2026-08-13 review):** the velocity-history path
     also adopts the GNS adjusted-next target (`next + noise[:, −1]`), so
     the target is the *clean* next velocity — the model de-noises the
     velocity exactly and is *not* asked to undo the random walk's
     accumulated position offset (~3.3 σ at C = 5), a partially
     unobservable component that would inflate the irreducible loss and
     the online target-normalizer std and bias rollouts toward
     over-contraction. The reference single-frame path keeps its MGN
     γ = 1 convention untouched — each noise scheme is paired with its
     own reference's target.
   - **`mesh_edge_max_stretch`** (MGN only): when > 0, mesh-edge messages
     whose current length exceeds the threshold × rest length are dropped —
     in training and rollout alike, inside the shared `_graph_features` —
     and the dropped (torn) pairs regain world-edge eligibility, so
     separated material interacts by proximity like any contact. A
     stretch-torn edge is physically meaningless in a lattice synthesized
     from an SPH flow; the gate makes the mesh a *breakable* inductive bias
     instead of a wrong one.

2. **The repair fleet: six arms, one ablation ladder per family.**
   `configs/taylor_impact_2d/`: `transolver-n02`, `transolver-n02-vh`,
   `geoflare-n02`, `geoflare-n02-vh`, `mgn-n02`, `mgn-n02-vh-sg2` — noise
   rescale alone per family (attribution), then +velocity-history
   (Transolver/GeoFLARE), then the full MGN repair (+stretch gate 2.0×; the
   threshold clears every physically coherent edge — notch max 1.46× — and
   tears the measured 5.8–31× tail). Everything else pins to the ADR-0047
   arm values (100k steps, batch 2, seed 1, measured radii). All results
   **provisional** per ADR-0046, recorded in the Taylor registry alongside —
   not replacing — the ADR-0047 numbers; the ADR-0047 MGN failure is
   recorded as-is with this ADR's diagnosis as the note.
   `hpc/dug/train_taylor_adr0049.slurm` parameterizes submission by arm
   config; scheduling is maintainer compute (flag-first).

   **Attention-family tuning round (maintainer-requested, 2026-08-13):**
   four further arms extend the ladder above the `n02-vh` base, each a
   single evidence-motivated knob (or matched pair):

   - `transolver-n02-vh-big` — hidden 128→256, slice_num 64→128 (~0.72M →
     ~2.9M params). Transolver's interp deficit concentrates on the
     8000-particle 100 mm bars (2.88/2.98 mm, its two worst cases) while
     128-slice GeoFLARE is best-in-class there (0.67–0.89 mm): a
     slice/width bottleneck test.
   - `transolver-n02-vh-250k`, `geoflare-n02-vh-250k` — 2.5× budget; both
     ADR-0047 runs selected checkpoints at 70k/78k of 100k (unsaturated),
     and the notch 200k→250k extension cut test rollout RMSE 21%
     (ADR-0039). The cosine anneal stretches with `training_steps`.
   - `geoflare-n02-vh-rad2x` — ball radii ×2 (standardized coords).
     GeoFLARE's one interp blowup is the SHORTEST bar (T-20-60-130,
     2.81 mm), where per-example standardization maps the DP-inherited
     radii to the smallest physical neighbourhood in the dataset.

   Tuning arms stack on `n02-vh` before that base is validated — a
   deliberate wall-clock trade the ladder still resolves: every tuning
   arm differs from `n02-vh` by exactly one knob, and `n02-vh` differs
   from `n02` by exactly one.

3. **Scope.** The knobs land benchmark-agnostic (any mesh-family config may
   set them; the 18 existing family TOMLs state the reference values
   explicitly per the strict-schema rule) but only the Taylor arms above
   enable them in this ADR. Porting to notch-impact/DeformingPlate arms, a
   velocity-history DP rerun, and any blessing decision are future ADRs on
   their own evidence.

## Alternatives considered

- **Per-step mesh reconnection (re-triangulating current positions)** —
  fixes the stretch pathology but abandons the reference-space edge
  features that define MGN, and rebuilds an O(P log P) mesh every rollout
  step; the stretch gate keeps the method's identity and costs one norm
  compare per edge.
- **World-edges-only MGN (drop the mesh entirely)** — degenerates MGN into
  a slower CGN; scientifically uninteresting as an "MGN" number.
- **Larger noise without the other repairs (noise-only for MGN)** — kept,
  as the `mgn-n02` control arm: if noise alone rescues MGN, the stretch
  gate's contribution is attributable by comparison with `mgn-n02-vh-sg2`.
- **Velocity history via wider input windows into the network (stacked
  frames)** — equivalent information, but velocities are the
  translation-invariant, GNS-reference encoding and keep the feature scale
  uniform.
- **Retuning `input_frames`** — protocol-pinned by the card (ADR-0035);
  not a knob.

## Consequences

- The three native families gain momentum awareness behind one flag,
  closing the declared gap between the natives' Markovian adaptation and
  CGN's GNS-reference history — with the DP-blessed reference behaviour
  untouched by default.
- MGN gets a physically-motivated contact mechanism for SPH-synthesized
  meshes; if `mgn-n02-vh-sg2` trains, Taylor's method comparison becomes a
  genuine four-way grid, and the stretch gate becomes a candidate for any
  future large-deformation mesh benchmark.
- Six more single-GPU runs (~41 GPU-h estimated from the ADR-0047 wall
  times) before the Taylor registry entries are final; the registry gains
  the ablation ladder, which is itself publishable comparison content.
- The strict config schema forces the new keys into every existing
  mesh-family TOML — noisy diff, but every config remains self-documenting
  about which recipe generation it belongs to.
