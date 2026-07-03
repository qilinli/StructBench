# ROADMAP.md

*Where StructBench is going, in what order, and what is currently parked.
**Ephemeral** (docs/HARNESS.md tenet 3): revised in place as priorities move; the
git history is the record. Decisions and their rationale live in `decisions/`
— this file only sequences them and tracks status. Read at session start when
the session is about planning or scoping (CLAUDE.md).*

*Last revised: 2026-07-03 — ADRs 0022–0027 **Accepted**: v0.2 = wave-1d +
notch-beam pair with retrained GNS baselines and benchmark cards; RC beam
(erosion) → v0.3; dataset hosting still open and now gating v0.2 as well.*

---

## Scope rule

Everything on this roadmap is **substrate-layer** work per ADR-0014's litmus
test: reusable benchmark infrastructure, canonical format, baselines,
evaluation protocols, general tooling. The research program's brain/body
layers (foundation models as paper contributions, agentic deployment on real
assets — `RESEARCH-PROGRAM.md`) are deliberately **not** roadmap items here;
they live in separate repositories and consume StructBench artefacts.

---

## Milestone: v0.1 — the Taylor substrate proof (current focus)

Per ADR-0021 (amending ADR-0015's release sequencing): **v0.1 is a running,
provable pipeline for the Taylor 2D impact benchmark**, released as a public
GitHub repository. No paper. The RC beam and segmented-beam benchmarks remain
committed (ADR-0015) but move to v0.2+.

### Definition of done (per ADR-0021)

- [x] **Canonical case format** (ADR-0011/0012/0013) with round-trip-tested
      HDF5 I/O
- [x] **General LS-DYNA adapter** (ADR-0016) — proven on two dataset families
      (Taylor 2D sweep; NB concrete-beam case ingested unchanged 2026-07-02)
- [x] **Taylor 2D benchmark** (ADR-0019): fixed split, eval protocol, QoIs —
      code + protocol complete, 66 tests green
- [x] **Config-driven pipeline** (`structbench-train`: train/valid/rollout,
      run-dir contract, metrics artifacts)
- [x] **radius_graph batch-partition fix** (landed 2026-07-02: search now
      grouped per example — 50.9 s → 0.22 s per batch-32 graph build,
      identical edges, same interface)
- [ ] **Trained single-scale GNS baseline** with checkpoint + recorded
      ADR-0019 metrics (DUG run; human-gated SSH-side steps)
- [x] **Public GitHub repository** — live at
      github.com/qilinli/StructBench (2026-07-02, pre-release)
- [ ] **v0.1 release proper**: baseline metrics into the README table,
      prediction-vs-truth hero GIF, dataset link, version tag (human action)

### Explicitly not in v0.1

MS-GNS second baseline (Proposed spec stands on its own); RC beam (v0.3,
ADR-0024) and segmented-beam benchmarks; a paper; PyPI packaging (undecided,
not required).

---

## Milestone: v0.2 — the ladder grows (next)

Per ADR-0024: v0.2 ships three benchmarks — `wave_propagation_1d`
(ADR-0025) and the pair `notch_beam_2d_bend` / `notch_beam_2d_impact`
(ADR-0026) — each with the v0.1 single-scale GNS retrained as its
baseline, plus the benchmark-card convention (ADR-0027) across all four
benchmarks. The platform reads as a difficulty ladder: 1D elastic wave →
Taylor 2D plasticity → concrete fracture (no erosion) → v0.3 erosion.

### Definition of done

- [ ] **Ingestion**: 16 wave runs + the notch-beam spec's 216 cases + 5
      generalisation probes to canonical HDF5 (batched OneDrive
      hydration, specific paths only; extras in the raw tree flagged,
      not ingested)
      (wave-1d done 2026-07-03; notch-beam batch in progress)
- [ ] **Three benchmark modules** with frozen split lists (ADR-0025's
      table; ADR-0026's stratified rule) and QoIs
      (all three v0.2 benchmark modules done 2026-07-03)
- [ ] **Benchmark cards** (ADR-0027): `BenchmarkCard` type, per-module
      `card.py` (Taylor retrofitted), generated `docs/benchmarks.md`
      index + per-archive README/`card.json`
- [ ] **Benchmark-selection mechanism** in the config pipeline
      (`cli/train.py` currently hard-imports Taylor)
- [ ] **Three trained GNS baselines** with checkpoints + recorded metrics
- [ ] **Dataset hosting settled** (open question below — ~7× v0.1 data)

---

## Near horizon (v0.3 candidates)

Sequenced roughly; each becomes its own ADR/spec when picked up.

- **RC beam benchmark** (v0.3 headline per ADR-0024): erosion is the open
  problem, twice — numerically for the FEM data, and structurally for the
  autoregressive surrogate (deleted elements = particles vanishing
  mid-rollout). Dataset designation narrows to
  `Concrete-Beam/Concrete_simulation_constantV1-16` (the `2DNotchBeam`
  half of the old evidence became its own benchmark pair, ADR-0026);
  prior model likely `code/gns-errosion`.
- **Segmented beam benchmark** (parked 2026-07-03 as more complex than
  the next release should carry): candidate folder
  `../data/Segmental_Beam` spotted in the archive root; dataset and
  prior model still to be designated.
- **MS-GNS second Taylor baseline**: spec + plan committed (Proposed),
  awaiting approval — confirmed out of v0.1.
- **Data-generation autonomy** (ADR-0015 §3 deferral): deck-templating with
  collaborator buy-in for scripted batch runs, or a Python-native solver.
  Unblocks dataset expansion beyond what collaborators have produced.
- **Training robustness**: resume support (optimizer state + `--resume`;
  turns the DUG walltime into a soft limit); part-id→embedding-index remap
  (needed before any dataset with large/sparse LS-DYNA part ids — a design
  call because it changes the Taylor embedding; checked against the
  notch-beam decks at v0.2 ingestion, ADR-0024 — spec IDs look
  small/contiguous).
- **Checkpoint publishing**: a baseline-checkpoint release workflow
  (docs/ARCHITECTURE.md promises "a published checkpoint" per model; spec marked
  it out of the first slice).
- **Second auxiliary target**: `effective_plastic_strain` (ADR-0019 noted;
  data already ingested).
- **Eval growth**: leaderboard submission validator; per-region probe
  metrics (ADR-0017); optional Convergence mesh-resolution check (ADR-0019).
- **RC slab under close-in blast** as a harder benchmark (deferred from
  ADR-0003; coupled ALE+SPH+Lagrangian).
- **Benchmark ergonomics**: `benchmark.py` owns dataset-root resolution;
  normalization-stats cache shipped 2026-07-02, pattern extends per dataset.

## Later horizons (post-v0.1, still substrate)

- **Scale**: spatial-grid / cell-list `radius_graph` backend (O(N), same
  interface — ADR-0020's designed seam) when a ≥10⁶-node dataset lands.
- **Other solvers**: sibling adapters (`core/io/kratos.py`, OpenSees,
  OpenRadioss…) per ADR-0004; community contributions become realistic once
  v0.1 is public.
- **SHM expansion** (VISION's second phase): sensor schema slots
  (`response/sensor`, reserved in ADR-0012/0013) get field-level definitions;
  multi-modal SHM benchmark datasets; the reserved `vision/` and `sensing/`
  namespaces (docs/ARCHITECTURE.md) open when this begins.
- **Deployment tools** (`deploy/` namespace): asset-onboarding workflows —
  the substrate side of the program's body layer.
- **Packaging options**: `structbench[train]` extra for a data-only light
  install (ADR-0018 flags this as a natural future superseding ADR);
  selective-extraction opt-in for the adapter (ADR-0016).
- **Interop**: PhysicsNeMo model-edge compatibility, opt-in per baseline
  (ADR-0017).

---

## Parked — waiting on a human decision or action

| Item | Waiting on |
|---|---|
| Taylor baseline DUG run | SSH-side steps (partition/gres via `sinfo`, placeholders, rclone/rsync, smoke, sbatch) |
| v0.1 tag + release | trained baseline metrics landing first |
| MS-GNS implementation | approval (or amendment) of the Proposed spec |
| RC beam benchmark (v0.3, ADR-0024) | dataset + prior-model confirmation; erosion task design |
| Segmented beam benchmark (parked) | dataset + prior-model designation (candidate: `../data/Segmental_Beam`) |
| ADR-0012 Voigt-component prose reconciliation | docs/CORRECTIONS.md distillation pass |

## Open scoping questions

1. **Dataset hosting for released benchmarks.** All data currently lives in
   the maintainer's OneDrive; the canonical HDF5 sets are GB-scale. How the
   public release points at data (Zenodo DOIs, HuggingFace datasets,
   institutional storage, on-request) is an open discussion — to be settled
   before the GitHub release, since a benchmark repo without reachable data
   is not usable by others. Now also a v0.2 definition-of-done item
   (ADR-0024): v0.2 multiplies the case count roughly sevenfold.

*Release-history note (2026-07-02): `RESEARCH-PROGRAM.md` is untracked from
here on, but it remains in commits before this date — so if this repository
is ever made public directly, publish a fresh clean-cut repo from the release
state instead of flipping this one's visibility.*

*(Resolved 2026-07-02: v0.1 release form — GitHub repo, no paper, Taylor-only
running pipeline → ADR-0021. MS-GNS out of v0.1 → ADR-0021 §4.)*
