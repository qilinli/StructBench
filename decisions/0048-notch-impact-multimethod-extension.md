# 0048 — Notch-impact multi-method extension: native MGN/Transolver/GeoFLARE on the notched-beam SPH benchmark

**Status**: Proposed
**Type**: Durable
**Date**: 2026-08-12

## Context

ADR-0047 built the mesh-native path for SPH benchmarks — loader-level
lattice-mesh synthesis behind `BenchmarkSpec.mesh_transform`, kinematic
boundary bodies contacting through world edges, the ADR-0044/0045
autoregressive one-step scheme — and applied it to Taylor 2D.
`notch_beam_2d_impact` (ADR-0026/0029, blessed CGN h250c-s1 under the
ADR-0039 250 µs scored horizon) is the natural second application and
differs from Taylor in three ways that the ADR-0047 machinery must absorb:

1. **The lattice is incomplete.** The beam (part 1) sits on an exact 2.5 mm
   generation lattice at ~99.8% occupancy — the notch removes sites.
   ADR-0047's synthesizer requires a complete lattice and rejects it.
2. **The boundary bodies are particles, not an analytic plane.** The pin
   (part 2, bullet or sphere shaped, rigid-material) and the two support
   blocks (part 3, constrained) are SPH particles already present in every
   case, already declared kinematic (ADR-0026); the beam (part 1) is the
   concrete. Nothing needs to be injected — but unlike Taylor's static
   wall, BOTH bodies move in the data (maintainer clarification + measured
   2026-08-12): the pin's trajectory is *dynamic* — rigid in deformation but
   decelerating on contact (−120 → ≈−15 m/s over the window), so its motion
   is response-coupled, not exogenous — and the supports, though
   constrained, displace up to ≈3–6.6 mm.
3. **The blessed recipe carries aux-channel knobs the native families lack**:
   h250c trains max principal strain in asinh space with tail weight 3
   (ADR-0038); the native families have no such knobs.

Data facts (measured 2026-08-12 over all 110 cases): parts {1: beam,
2: pin, 3: supports}; beam lattice recoverable in every case (unique sites,
uniform 2.5 mm spacing); zero particle deletion over the 502-frame horizon;
initial pin–beam gap 2.5–2.7 mm; case sizes 4 264–8 360 particles across the
{320, 480, 640} mm spans.

## Decision

1. **Scope and status.** The three native families run on
   `notch_beam_2d_impact` as **provisional** baselines (ADR-0046 pattern, no
   numeric gate — no published number exists for any of them on this task).
   CGN h250c-s1 remains the blessed baseline. The benchmark protocol —
   ADR-0026 splits, ADR-0029 aux (`max_principal_strain`, 0.01 threshold),
   ADR-0039 scored horizon (`scored_frames = 250`) and truncated-training
   recipe (`train_frames = 250`) — is unchanged. All three families run the
   ADR-0044/0045 autoregressive one-step scheme (ADR-0047 clause 1); the
   scored rollout is 244 predicted steps from the card's 6-frame seed.

2. **Beam-only partial-lattice mesh.** `synthesize_lattice_mesh` gains two
   parameters, both defaulting to the strict ADR-0047 behaviour:
   `part` restricts meshing to one particle type (cells index the FULL
   particle array; `reference_coords` covers every particle), and
   `allow_missing` permits vacant lattice sites — a quad contributes its
   two triangles only when all four corners exist, stair-stepping the notch
   boundary. The notch-impact transform is
   `synthesize_lattice_mesh(traj, part=CONCRETE_TYPE, allow_missing=True)`.
   **No nodes are appended**: the particle set is identical to the cgn
   path's, so the ADR-0026/0029 QoIs (`midspan_deflection_peak`,
   `cracked_fraction`) apply verbatim — no wrapper, unlike Taylor.

3. **Pin and supports stay unmeshed, kinematic and scripted.** The spec
   gains `mesh_transform` and `scripted_types = (PIN_TYPE, SUPPORT_TYPE)`
   (= its existing `kinematic_types`). Both bodies interact with the beam
   through world edges exactly as DeformingPlate's OBSTACLE nodes, and both
   feed their real ground-truth next-step velocity as the scripted input —
   real motion for each (clause 2 of the Context). Two properties are
   declared rather than redesigned: (a) playing back GT motion for a
   *dynamic* pin injects response-coupled information into the rollout —
   this is the established ADR-0026 kinematic protocol, identical for the
   cgn baseline, so the cross-family comparison is like-for-like; (b) the
   DP scripted-velocity recipe feeds the *next-step* GT velocity, giving
   the native families a one-frame look-ahead of pin/support motion that
   the cgn input schema (current-state history) does not have —
   family-faithful to ADR-0043, noted in the registry entries. The cgn
   path is untouched (it never applies the transform, and its kinematic
   handling is unchanged). The pin/support rows carry no learnable aux:
   the ADR-0026 kinematic protocol masks them from the training loss
   (velocity AND strain terms), rollout zeroes their predicted aux, the
   strain metrics score beam rows only, and both QoIs are constructed
   ``concrete_type``-restricted. The single reference-faithful exception
   is the online target normalizer, which — as in the official MGN
   framework — accumulates over all rows (~4% kinematic, near-zero
   strain), mildly diluting the strain target scale; declared, not
   redesigned.

4. **`input_frames = 6`, h = 0 preserved** — identical to ADR-0047 clause 4.

5. **Working-frame scales.** `world_edge_radius = 7.5` mm — the CGN notch
   connectivity radius (3× the 2.5 mm particle spacing, corrections log
   2026-07-06), which sees the pin from frame 0 (measured initial gap
   2.5–2.7 mm). GeoFLARE ball radii stay per-example standardized
   (scale-free, ADR-0045).

6. **Budget matched to the blessed h250c recipe**: 250k steps,
   `train_frames = 250`, `val_every = 2000`, seeds ≥ 1. **Declared
   deviations** (config comments + training ledger): the native families
   train the strain channel in **raw space with plain MSE** — no
   `aux_transform="asinh"`, no `aux_tail_weight=3` (porting the ADR-0038
   knobs into three families is out of provisional scope); and they keep
   family-native batch 2 / noise 0.003 (h250c: batch 4 / noise 0.01).

7. **Configs.** `configs/notch_beam_2d_impact/{mgn,transolver,geoflare}.toml`
   plus `_smoke` siblings; `node_type_size = 4` (raw ids 1–3).

## Alternatives considered

- **Meshing the pin and supports too** — rejected: the sphere pin is a
  ~78%-occupancy disc (not lattice-complete in a meaningful sense), the
  supports are two disconnected blocks, and DP's obstacle mechanism —
  unmeshed kinematic nodes through world edges — is the reference-faithful
  treatment of ground-truth-driven bodies anyway.
- **Treating the pin as a free dynamic body (predicted, not scripted)** —
  rejected for this ADR: physically truer (its deceleration is
  response-coupled) but a *protocol* change to ADR-0026's kinematic
  declaration, which would break comparability with the blessed cgn
  baseline; open to a future benchmark-version ADR if the look-ahead
  concern proves material.
- **Porting asinh/tail-weight into the native families** — deferred, not
  taken: it adds ADR-0038's knobs to three architectures for a provisional
  comparison; if the natives' raw-space `cracked_fraction` numbers are
  distorted enough to matter, a follow-up ADR can add the knobs then.
- **Full-window (502-frame) training** — rejected: ADR-0039's truncated
  recipe is the blessed protocol and the gate/scoring window; matching it
  is the comparison.
- **A Taylor-style wall/boundary injection** — not applicable: every
  boundary body already exists as particles.

## Consequences

- The cross-method grid gains its third benchmark (after DeformingPlate and
  Taylor), now covering a moving scripted contact body and a
  fracture-proxy aux channel.
- `synthesize_lattice_mesh` becomes partial-lattice capable behind
  default-off parameters — ADR-0047's Taylor contract is untouched (strict
  complete-lattice mode remains the default and its tests still pass).
- The aux-channel comparison carries a declared asymmetry: CGN h250c trains
  strain in asinh+tail space, the natives in raw space. Rollout/one-step
  strain RMSE and `cracked_fraction` must be read with that in mind; the
  registry entries say so.
- Memory: up to 8 360 nodes per case (vs Taylor's 8 000) — inside what the
  Taylor runs already handle with the activation-checkpointed MGN processor.
- The runs are maintainer compute (three ~250k-step single-GPU jobs,
  roughly 2.5× the Taylor arms by steps).
