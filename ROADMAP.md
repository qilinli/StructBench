# ROADMAP.md

*Where StructBench is going, in what order, and what is currently parked.
**Ephemeral** (HARNESS.md tenet 3): revised in place as priorities move; the
git history is the record. Decisions and their rationale live in `decisions/`
— this file only sequences them and tracks status. Read at session start when
the session is about planning or scoping (CLAUDE.md).*

*Last revised: 2026-07-02 (drafted by Claude Code; pending human review).*

---

## Scope rule

Everything on this roadmap is **substrate-layer** work per ADR-0014's litmus
test: reusable benchmark infrastructure, canonical format, baselines,
evaluation protocols, general tooling. The research program's brain/body
layers (foundation models as paper contributions, agentic deployment on real
assets — `RESEARCH-PROGRAM.md`) are deliberately **not** roadmap items here;
they live in separate repositories and consume StructBench artefacts.

---

## Milestone: v0.1 — the substrate proof (current focus)

Per ADR-0015: a portfolio of existing LS-DYNA datasets shipped as benchmarks,
each paired with a prior-paper GNN baseline, ingested through the general
adapter into the canonical format.

### Definition of done (proposed — confirm or amend)

- [ ] **Three benchmarks** with fixed splits, eval protocols, and QoIs:
  - [x] Taylor 2D (ADR-0019; code + protocol complete, 65 tests green)
  - [ ] RC beam — *data located, designation pending (see Parked)*
  - [ ] Segmented beam — *dataset not yet identified in the archive*
- [ ] **One trained baseline per benchmark**, with training code, config,
  checkpoint, and reported ADR-0019-style metrics:
  - [ ] Taylor 2D single-scale GNS — code complete; full training run parked
        (DUG); **blocked by the radius_graph batch-partition fix** (identified
        2026-07-02: current op is O((B·N)²) across the concatenated batch,
        224× slower than per-example at batch 32 — must land first)
  - [ ] RC beam baseline (port of its prior-paper GNN — repo TBC)
  - [ ] Segmented beam baseline (ditto)
- [x] **Canonical case format** (ADR-0011/0012/0013) with round-trip-tested
      HDF5 I/O
- [x] **General LS-DYNA adapter** (ADR-0016) — proven on two datasets
      (Taylor 2D sweep; NB concrete-beam case ingested unchanged 2026-07-02)
- [ ] **Batch-conversion glue** per dataset under `data_generation/lsdyna/`
      (Taylor done; others follow designation)
- [ ] **Release form** — *open question below: what does "ship" mean for
      v0.1 (public repo? PyPI? checkpoint hosting? a paper?)*

### In-flight / proposed within v0.1

| Item | Status | Gate |
|---|---|---|
| radius_graph batch-partition fix | identified, ~20 lines, TDD | none — next code slice |
| Taylor full baseline run on DUG | recipe ready (`deploy/dug/`) | human: SSH, placeholders, data transfer, smoke, sbatch |
| MS-GNS second Taylor baseline | spec + plan committed (Proposed) | human approval of `docs/specs/2026-07-02-taylor-multiscale-gns.md` |
| RC beam ingestion + benchmark ADR | adapter proven; glue is quick | human: designate dataset + prior-model repo |
| Segmented beam | not started | human: identify dataset |

---

## Near horizon (v0.1 → v0.2 candidates)

Sequenced roughly; each becomes its own ADR/spec when picked up.

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
| Taylor baseline DUG run | SSH-side steps (partition/gres via `sinfo`, placeholders, rclone/rsync, smoke, sbatch) |
| MS-GNS implementation | approval (or amendment) of the Proposed spec |
| RC beam benchmark | which dataset + which prior-model repo (evidence so far: `Concrete-Beam/Concrete_simulation_constantV1-16` + `2DNotchBeam`, prior model likely `code/gns-errosion`) |
| Segmented beam benchmark | dataset identification in `../data/` |
| v0.1 release form | open questions below |
| ADR-0012 Voigt-component prose reconciliation | CORRECTIONS.md distillation pass |

## Open scoping questions (nobody has decided these yet)

1. **What does "ship v0.1" concretely mean?** Public GitHub repo? PyPI
   package? Zenodo dataset DOIs? Hosted checkpoints? A datasets/benchmarks
   paper (ADR-0015 hints the paper positions StructBench as substrate)?
2. **Dataset hosting**: the canonical HDF5 sets are GB-scale — where do
   released benchmarks live (Zenodo, HuggingFace datasets, institutional)?
3. **Is MS-GNS in or out of v0.1?** ADR-0015 requires one baseline per
   benchmark; a second Taylor baseline is enrichment, not a requirement.
