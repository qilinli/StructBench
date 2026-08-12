# EMI26IC study plan — cross-method impact comparison + token-scale mechanism study

*Written 2026-08-12 (home-PC session; maintainer-approved direction). Target: EMI26IC talk (early Sep,
abstract: "Multi-Scale Structural Graph Neural Networks for Efficient Prediction of Structural Responses" —
drift approved by maintainer) + seed of a CMAME mechanism paper. This doc is the sync point between venues:
implementation/runs on DUG, diagnostics/analysis/slides on the home machine. Session-strategy context lives
in the maintainer's research-notes vault and `scratch/` (absent from clones by design) — this plan is
self-contained for implementation purposes.*

## Study in one paragraph

Two coupled deliverables. (1) **Cross-method comparison on impact**: native MGN, Transolver, and GeoFLARE
run on Taylor-impact 2D and notch-beam 2D against the blessed CGN anchors, under the pinned per-benchmark
protocols, provisional-flagged per ADR-0046 — StructBench's first method×benchmark grid on its own impact
data. (2) **Mechanism study**: when/why does N→M token reduction break autoregressive solids rollout —
token count M as a learned low-pass cutoff. Hypothesis: token-bottleneck models' one-step error
concentrates in high graph-frequency, contact-local bands whose per-step amplification exceeds 1 under
rollout; MGN/CGN (M=∞, local MP) stay spectrally flatter; accuracy-vs-M is a dose-response curve with a
knee (the "safe compression budget"); slice-assignment entropy measured ALONG the rollout predicts
error-growth onset. Pre-registered alternative outcome: flat/M-independent bands ⇒ decoder spectral bias
(PDE-Refiner/ROBIN account), not the bottleneck — publishable either way. The impact runs double as the
cross-regime test: the effect should be stronger on IC-driven Taylor/notch than boundary-driven
DeformingPlate.

## Experiments

### E0 — Diagnostics (inference only; home machine, as checkpoints land)

| ID | Substrate | Measured |
|----|-----------|----------|
| E0a | Taylor blessed CGN (available) | Band-resolved rollout error growth (graph-Laplacian bands on the reference-configuration graph), elastoplastic front-position error, drift shape (extends the 2026-08-07 diagnostic) |
| E0b | DP Transolver (job 62801985, ETA ~Aug 13) | Pinned rollouts + hooks: per-layer slice entropy and token occupancy vs rollout step, per-band error growth, divergence horizon, contact-zone vs bulk error split |
| E0c | DP GeoFLARE (job 62801986, ETA ~Aug 16) | Same suite + query-attention rank; does upstream's AR instability reproduce under our recipe? |
| E0d | DP MGN blessed (running, ETA ~Aug 19–21) | Same suite = M=∞ control; blessing transcription → the ADR-0046 comparison table |
| E0e | Notch blessed CGN (optional) | Crack-localization figure (IoU/band count) — the "concrete, local behaviour" slide |

### E1 — Transolver M-sweep on DeformingPlate (the causal axis; DUG)

Retrain the native Transolver at **M ∈ {8, 32, 128, 512} slices**, all else pinned to the reference recipe,
at reduced budget **2–4M steps** (justified: the full 10M run's best val is at 4.0M with no improvement in
the following 3M — plateaued), 1 seed each, then **+2 seeds at the knee M** (tuning-stability check).
Report pinned metrics + E0 diagnostic suite + memory and steps/sec per M.
≈ 6 runs × 18–36 h (measured ~109k steps/h) ≈ 300–400 A100-h.

### E2 — Impact cross-method matrix (implementation on DUG, then runs)

**Matrix: {MGN, Transolver, GeoFLARE} × {taylor_impact_2d, notch_beam_2d_impact}**, vs the blessed CGN
anchors (CGN stays the incumbent; result rows land provisional per ADR-0046). ≈ 6 runs + seeds at
Taylor/notch scale (5–13k particles, runs ~1–2 days) ≈ 500–700 A100-h.

Adaptation pins settled so far (each family gets a declared-adaptation ADR in the 0044/0045 pattern;
maintainer finalizes):

- **MGN mesh synthesis (Taylor):** the SPH lattice is exact (8,000 particles on a perfect 0.5 mm 40×200
  grid, zero deletion over the horizon) → synthesize the reference-configuration mesh directly, **as
  triangles via quad-split (15,522 cells)** because `cells_to_edges` expands every vertex pair per cell and
  quad cells would emit both diagonals as spurious edges. Mesh-space edges from reference coords + dynamic
  world edges for contact = full MGN semantics. *A DUG-side ADR draft covering this exists — push it from
  DUG; it is the source of truth for these pins.*
- **Task shape (all three families):** pinned Taylor/notch protocol — velocity-history input window per
  ADR-0035 (input_frames=6), **acceleration target + GNS-style integration** (matches the blessed task;
  MGN's own dynamic-domain convention), per-particle von Mises co-prediction, GNS noise recipe, 2D
  featurization (point-set families: the DP 18-channel layout re-derived for dim=2 + velocity window).
- **Notch impactor:** ground-truth-prescribed → map onto the existing kinematic/OBSTACLE masking pattern
  (prescribed channels legal over the horizon; masked from loss/metrics). Never derive impactor-contact
  QoIs (label-leakage rule).

Open decisions to settle during implementation (record in the ADRs):
1. **Notch lattice verification** — confirm mesh recoverability (regular lattice minus notch geometry, no
   deletion over the ADR-0039 250 µs horizon) against canonical data before writing the notch configs.
2. **Taylor wall representation for MGN** — boundary-distance node feature (CGN's pattern) vs synthesized
   wall nodes + world edges (MGN's pattern). Whichever is chosen, document information-parity vs CGN.
3. **Budget parity per benchmark** — define "same budget" against the blessed CGN recipes (samples-seen
   basis; declare the cost basis with every comparison, wall-clock-vs-samples confound is a known trap).
4. **Transolver/GeoFLARE 2D featurization details** — channel layout, normalizers, per-example
   standardization radii (GeoFLARE geo_ops) re-checked for 2D.

**Expectation set deliberately:** provisional token models on severe-deformation contact may perform badly
or diverge (GeoFLARE's upstream AR instability). Under the mechanism framing that outcome is a finding with
diagnostics attached, not a failure; provisional flags keep the registry honest.

## Timeline (talk early Sep)

| Dates | Work |
|---|---|
| Aug 12–17 | E0a now; E2 adaptation implementation (DUG); E0b when Transolver lands (~Aug 13); launch E1 wave 1 |
| Aug 16–21 | E0c (~Aug 16); E2 impact runs launch (~Aug 18–19); E0d + comparison table when MGN lands (~Aug 19–21); E1 wave 2 (knee seeds) |
| Aug 22–25 | E2 runs land; full diagnostic suite over the impact grid |
| Aug 25–31 | Analysis consolidation; talk figures (dose-response w/ cost overlay, per-band growth, entropy-vs-time, comparison tables, contact/bulk split, notch localization); slides |

Fleet discipline: ≥2 concurrent single-GPU jobs throughout (E1 + E2 arms interleave); seeds ≥1; fresh
`--out` per attempt; no training resume.

## Roadmap beyond the talk (CMAME phase, not pre-EMI)

Matched-budget stabilizer factorial (spectrum-shaped vs isotropic noise, pushforward), band-scoped
corrector intervention, GeoFLARE rank sweep, mechanism-transfer analysis across the full method×benchmark
grid, and the talk's roadmap slide: part-aware carrier placement (the abstract's element/component/object
hierarchy, rehabilitated as placement policy) + deforming Lagrangian carrier.
