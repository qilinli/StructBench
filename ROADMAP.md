# ROADMAP.md

*Where StructBench is going, in what order, and what is currently parked.
**Ephemeral** (HARNESS.md tenet 3): revised in place as priorities move; the
git history is the record. Decisions and their rationale live in `decisions/`
— this file only sequences them and tracks status. Read at session start when
the session is about planning or scoping (CLAUDE.md).*

*Last revised: 2026-07-02 — v0.1 scope settled by the maintainer (Taylor-only,
GitHub release, no paper; ADR-0021 Proposed); MS-GNS confirmed out of v0.1;
dataset hosting still open.*

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
- [ ] **Public GitHub release** (out-of-session human action)

### Explicitly not in v0.1

MS-GNS second baseline (Proposed spec stands on its own); RC beam and
segmented-beam benchmarks (v0.2+); a paper; PyPI packaging (undecided,
not required).

---

## Near horizon (v0.1 → v0.2 candidates)

Sequenced roughly; each becomes its own ADR/spec when picked up.

- **RC beam benchmark** (moved from v0.1 by ADR-0021): designate the dataset
  + prior-model repo (evidence so far:
  `Concrete-Beam/Concrete_simulation_constantV1-16` + `2DNotchBeam`, prior
  model likely `code/gns-errosion`); the adapter already ingests the data
  unchanged, so glue + benchmark ADR + baseline port is the work.
- **Segmented beam benchmark** (moved from v0.1 by ADR-0021): dataset not
  yet identified in the archive.
- **MS-GNS second Taylor baseline**: spec + plan committed (Proposed),
  awaiting approval — confirmed out of v0.1.
- **Data-generation autonomy** (ADR-0015 §3 deferral): deck-templating with
  collaborator buy-in for scripted batch runs, or a Python-native solver.
  Unblocks dataset expansion beyond what collaborators have produced.
- **Training robustness**: resume support (optimizer state + `--resume`;
  turns the DUG walltime into a soft limit); part-id→embedding-index remap
  (needed before any dataset with large/sparse LS-DYNA part ids — a design
  call because it changes the Taylor embedding).
- **Checkpoint publishing**: a baseline-checkpoint release workflow
  (ARCHITECTURE.md promises "a published checkpoint" per model; spec marked
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
  namespaces (ARCHITECTURE.md) open when this begins.
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
| ADR-0021 finalisation | human accepts (or amends) the Proposed draft |
| Taylor baseline DUG run | SSH-side steps (partition/gres via `sinfo`, placeholders, rclone/rsync, smoke, sbatch) |
| v0.1 GitHub publication | out-of-session human action (after the trained baseline lands) |
| MS-GNS implementation | approval (or amendment) of the Proposed spec |
| RC beam benchmark (v0.2) | dataset + prior-model designation |
| Segmented beam benchmark (v0.2) | dataset identification in `../data/` |
| ADR-0012 Voigt-component prose reconciliation | CORRECTIONS.md distillation pass |

## Open scoping questions

1. **Dataset hosting for released benchmarks.** All data currently lives in
   the maintainer's OneDrive; the canonical HDF5 sets are GB-scale. How the
   public release points at data (Zenodo DOIs, HuggingFace datasets,
   institutional storage, on-request) is an open discussion — to be settled
   before the GitHub release, since a benchmark repo without reachable data
   is not usable by others.

*(Resolved 2026-07-02: v0.1 release form — GitHub repo, no paper, Taylor-only
running pipeline → ADR-0021. MS-GNS out of v0.1 → ADR-0021 §4.)*
