# 0047 — Taylor 2D multi-method extension: native MGN/Transolver/GeoFLARE on the SPH benchmark

**Status**: Proposed
**Type**: Durable
**Date**: 2026-08-12

## Context

v0.3 built three native model families — MGN (ADR-0043), Transolver (ADR-0044),
and GeoFLARE (ADR-0045) — on `deforming_plate`, with cross-method comparison as
the headline (ADR-0041, results protocol ADR-0046). Extending the same three
families to `taylor_impact_2d` (protocol ADR-0019/0035, blessed single-scale
CGN baseline s1) would produce the platform's first cross-benchmark ×
cross-method grid and exercise the families on a benchmark that differs from
DeformingPlate on every axis: 2D not 3D, SPH particles not nodal FE, an
analytic rigid wall not scripted obstacle nodes, and a 100k-step CGN recipe
not a 10M-step paper reproduction.

Today none of the three can run on Taylor. A 2026-08-12 review found four
gaps and established the data facts that shape the fixes:

1. **No mesh connectivity.** The canonical loader's SPH path returns
   `cells=None, reference_coords=None`; all three trainers and `evaluate`
   raise `ValueError` on that. It is not only MGN's mesh edges — Transolver
   and GeoFLARE consume `reference_coords` as input channels (18-ch schema,
   ADR-0044 §inputs).
2. **The wall is invisible to the native input schemas.** Taylor's rigid wall
   is analytic (`WALL_X_MM = -2.0`), fed to CGN as a wall-distance input
   channel via `spec.boundary_feature_fn` — a channel only the CGN path
   consumes. The native families' node features are
   `[one_hot, scripted_velocity(, x, ref)]`; in DeformingPlate, contact is
   learned from kinematic OBSTACLE nodes through world edges. Taylor's spec
   has no kinematic types and its particle set contains no wall nodes (the
   4-node visualization shell, part id 2, is dropped by the SPH loader).
3. **`input_frames` clash.** The Taylor card pins `input_frames = 6` (GNS
   C = 5 reference); under ADR-0035 the config loader rejects any
   `[model].input_frames` that disagrees. The native families' configs
   default to 2. Mechanically, however, all three consume only the window's
   last frame (`position_seq[:, -1]`; the reference MGN history is h = 0), so
   window length affects only sampling and rollout seeding.
4. **No configs.** `configs/taylor_impact_2d/` has only the CGN pair, and at
   least `world_edge_radius` cannot be copied from DP (30.0 mm was measured
   for DP's working frame, ADR-0042 §2b; Taylor's scale is 0.5 mm particle
   spacing).

Data facts (measured 2026-08-12 on `T-20-100-100`; structure shared by all
cases): 8000 SPH particles, all part id 1, on an exact regular 0.5 mm lattice
(40 × 200, the 20 × 100 mm bar); zero particle deletion over the 152-frame
horizon; the wall shell is part id 2, a degenerate visualization quad on the
x = −2 mm plane.

## Decision

1. **Scope and status.** The three native families run on `taylor_impact_2d`
   as **provisional** baselines under the ADR-0046 pattern: no numeric gate
   (no published number exists for any of the three on this task), results
   recorded in the Taylor results registry with the method-comparison
   rendering. CGN remains the blessed baseline. The ADR-0019/0035 benchmark
   protocol — splits, 151-frame horizon, scored span, QoIs, aux
   `von_mises_stress` — is unchanged.

2. **Synthesized mesh, loader-level, opt-in per benchmark.** For SPH cases of
   a benchmark that opts in (a new `BenchmarkSpec` field), the loader
   synthesizes mesh connectivity from the initial particle lattice:
   `reference_coords` = initial coordinates; `cells` = **triangles**, each
   lattice quad split into two (39 × 199 quads → 15 522 triangles for
   Taylor). Triangles, not quads, because `cells_to_edges` expands every
   intra-cell vertex pair — quad cells would emit both diagonals, triangles
   give the standard simplicial mesh (lattice edges + one diagonal per quad),
   matching the 2D meshes of the MGN reference tasks. The canonical `.h5`
   files are untouched — no schema change, ADR-0042 stands — and the CGN SPH
   path is byte-identical (synthesis feeds only the `cells` /
   `reference_coords` fields CGN never reads; zero-deletion is asserted at
   synthesis time since a fixed mesh cannot represent eroded particles).

3. **Wall as synthesized kinematic nodes.** The analytic wall enters the
   native graph as static nodes on the x = `WALL_X_MM` plane at the particle
   spacing (0.5 mm), node type **2** (the wall's raw LS-DYNA part id — no
   remapping, consistent with the CGN `max(part_id)+1` embedding convention),
   zero velocity throughout, aux = 0. The Taylor spec gains
   `kinematic_types = (2,)`: wall rows are excluded from noise, loss, and
   rollout scoring exactly as DP's OBSTACLE nodes, and contact is learned
   through world edges (the ADR-0043 mechanism, no architecture change). CGN
   is unaffected: its loaded particle set contains no type-2 rows, so the
   kinematic mask matches nothing, and its wall-distance channel stays as-is.
   The wall segment's lateral span must generously cover the mushroomed bar's
   contact footprint; the value is **measured over the dataset's maximum
   lateral spread at the wall and recorded in the training ledger** (open
   numeric, implementation-time).

4. **`input_frames = 6`, h = 0 preserved.** Native Taylor configs set
   `input_frames = 6` to satisfy the card check; the families still read only
   the last frame, so the extra frames affect windowing and the ADR-0035
   rollout seed count only. The scored span stays `[6, 151]`, identical to
   CGN — no card amendment, full score comparability. Any latent
   `input_frames == 2` assumption found at implementation is relaxed to
   card-matching, not hardcoded.

5. **Working-frame scales.** `world_edge_radius` for Taylor is a measured
   value per the ADR-0042 §2b procedure (wall-gap / contact-distance
   distributions), expected O(1 mm); measured before configs are finalized
   and recorded in the config comment. GeoFLARE's ball radii stay
   `[0.05, 0.25]` in per-example standardized space (scale-free, ADR-0045);
   Transolver/GeoFLARE channel widths derive from `dim = 2` automatically
   (`node_type_size + 3·dim`).

6. **Budget matched to the blessed CGN Taylor recipe, not DP.** Batch 2,
   100k steps, CGN-cadence validation, seeds ≥ 1 (corrections log
   2026-07-10). The comparison that matters lives on this benchmark against
   the CGN baseline at equal budget; DP's 10M-step budget is a
   paper-reproduction constraint with no counterpart here. A longer-budget
   rerun stays open to the maintainer. Scheduling the runs is maintainer
   compute (flag-first), with the ≥2-concurrent-jobs DUG convention.

7. **Configs.** `configs/taylor_impact_2d/{mgn,transolver,geoflare}.toml`
   plus `_smoke` siblings; every deviation from the DP reference values
   (budget, radii, `dim`, `input_frames`) is carried in config comments and
   the training ledger at run time.

## Alternatives considered

- **Delaunay triangulation of initial positions** — general, but Taylor's
  lattice is exact and dependency-free to recover; Delaunay adds a scipy
  dependency (flag-first) for no benefit here. Reconsider if a future SPH
  benchmark is non-lattice.
- **Wall-distance input channel on the native families** (CGN-style) —
  rejected: modifies the three architectures' input schemas away from their
  DP-reference implementations (per-family fidelity principle, ADR-0044/0045)
  when DP's kinematic-node mechanism already covers rigid boundaries.
- **Re-running Taylor as Lagrangian FE for a true mesh** — rejected: new data
  generation breaks identity with the released benchmark; the benchmark *is*
  the SPH dataset.
- **Amending the card to `input_frames = 2`** — rejected: a protocol change
  that would re-open the blessed CGN scoring span for zero benefit given the
  families are h = 0.
- **DP's 10M-step budget** — rejected as the default: ~100× the compute of
  the CGN-matched budget with no published-number gate to satisfy, and it
  would break equal-budget comparability with the blessed baseline.
- **Remapping node types to a compact 0/1** — rejected: raw part ids
  (1 = bar, 2 = wall) are already compact and match the existing embedding
  convention.

## Consequences

- The platform gains its first cross-benchmark × cross-method grid, and the
  native families become 2D- and SPH-capable through one reusable
  lattice-synthesis path (notch-impact, also SPH, could follow with the same
  mechanism).
- The synthesized mesh is a **declared modeling choice, not data**: the
  canonical files stay pure SPH, and the benchmark card/landing page must say
  the native baselines run on a synthesized lattice mesh.
- Two numerics are deliberately left open for implementation-time
  measurement: the wall segment span and the Taylor `world_edge_radius`;
  both land in config comments and the training ledger, not this ADR.
- Memory is the run-time risk to watch: 8000 nodes × 15 MP steps is ~6× DP's
  node count, and world edges densify at the wall during mushrooming; the
  ADR-0043 activation-checkpointed processor is already in place, and the
  radius choice bounds degree.
- Implementation surface: `datasets/canonical.py` (synthesis),
  `benchmarks/taylor_impact_2d` (spec fields), three configs + smokes, tests
  (synthetic lattice synthesis, wall-node injection, config load); expected
  zero changes inside `models/`.
