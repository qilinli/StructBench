# 0017 — Relationship to NVIDIA PhysicsNeMo: independent substrate, opt-in interop at the model edge

**Status**: Accepted
**Type**: Durable
**Date**: 2026-06-28

## Context

NVIDIA PhysicsNeMo (the open-source, Apache-2.0, PyTorch physics-ML
framework formerly named Modulus) overlaps StructBench's near-term scope.
Since 2025 it has added a `structural_mechanics` example suite
(`deforming_plate`, `crash`, `drop_test`, `openradioss_dataset_gen`) whose
newer examples ingest LS-DYNA / OpenRadioss explicit-FE data and train
GNN/transformer surrogates of transient structural response — the same
territory as StructBench v0.1 (ADR-0015, ADR-0016). This prompted the
question: should StructBench adopt PhysicsNeMo's data and processing
abstractions (build "on" / "compatible with" PhysicsNeMo), or build
independently? The motivation was twofold: avoid reinventing wheels, and
avoid competing with a large NVIDIA team.

A primary-source-verified review this session established the relevant
facts:

- **PhysicsNeMo is a framework, not a benchmark.** It is a model zoo
  (~23 architectures) plus training/inference pipelines. Its only
  benchmark harness, `physicsnemo-cfd`, is automotive-aerodynamics only
  (DrivAerML), with no leaderboard and no structural coverage. Its
  structural pieces are framework *examples*, not a benchmark.
- **It standardizes no canonical case format.** Each structural example
  ships its own bespoke reader (`crash`: VTP + Zarr; `drop_test`: VTU;
  `deforming_plate`: TFRecord). There is nothing on disk to "conform to."
  It also prescribes no units convention — StructBench's strict-SI
  canonical (ADR-0012) is *more* disciplined than anything PhysicsNeMo
  offers.
- **Its data layer is mid-refactor.** Two generations coexist (legacy
  `torch.utils.data.Dataset` subclasses vs. a newer
  `Reader → Transform → Dataset` system on `tensordict`); the GNN
  datapipes are not migrated. The only stable, blessed interop surfaces
  are `physicsnemo.Module` (checkpoint contract),
  `torch_geometric.data.Data`/`HeteroData` (graph interface), and
  `tensordict` (in-memory container). DGL is no longer used.
- **It occupies none of the broader program's white space.** No
  structural health monitoring, no real-sensor assimilation, no
  online/Bayesian updating, no agentic asset-management layer; its civil
  "digital twin" story is Omniverse visualization, not management
  decisioning.

Under the ADR-0014 substrate litmus test, PhysicsNeMo overlaps StructBench
only at the model/framework level *inside* the substrate — the layer where
StructBench's reference baselines (ADR-0015) live. It does not touch
StructBench's benchmark identity (canonical format, fixed splits,
model-agnostic evaluation, leaderboard), nor the **brain** (foundation
models) and **body** (agentic SHM / asset management) layers of the
broader program. The "complete digital twin + agentic system" ambition is
the program's brain+body, which by ADR-0014 lives outside StructBench — so
no PhysicsNeMo-compatibility question arises there at all.

The decisive observation: adopting PhysicsNeMo would not solve the problem
StructBench exists to solve. StructBench's purpose is to unify previously
ad-hoc, per-paper datasets and models behind one consistent
train/evaluate substrate. PhysicsNeMo's answer to "unify my datasets" is
"write a reader per dataset" — which is the ad-hoc situation StructBench is
curing, merely relocated. The unifier is a canonical schema, which
StructBench has (ADR-0011/0012/0013) and PhysicsNeMo lacks.

## Decision

1. **StructBench remains an independent, framework-neutral substrate.**
   It does not adopt PhysicsNeMo (or any ML framework) as its foundation.
   `core/` takes no dependency on PhysicsNeMo. The canonical case schema
   (ADR-0011/0012/0013) and the d3plot ingestion path (ADR-0016) are
   sovereign and are not restructured around PhysicsNeMo's abstractions.
   Framework-neutrality is treated as load-bearing: it is what lets the
   benchmark fairly evaluate a PhysicsNeMo-trained model and a
   non-PhysicsNeMo model on the same canonical data and metrics.

2. **PhysicsNeMo is a bounded design reference and an opt-in interop
   target at the model edge only.**
   - *Design reference* — the model and evaluation *patterns* in its
     structural examples (the rollout-scheme taxonomy; mesh- and
     contact-edge graph construction; next-step delta targets;
     train-only normalization-stats discipline) inform the design of
     `datasets/`, `models/`, and `eval/`, reimplemented in StructBench's
     own framework-neutral code.
   - *Interop target* — exactly three PhysicsNeMo surfaces may be
     conformed to, optionally, and only in `datasets/` and `models/`:
     `physicsnemo.Module` (so a baseline may publish a
     physicsnemo-loadable checkpoint), `torch_geometric.data.Data` /
     `HeteroData` (the graph interface for GNN baselines), and
     `tensordict`. A `case → PyG Data` adapter belongs in `datasets/`
     and is added when the first GNN baseline needs it, never
     speculatively.

3. **Taking on `physicsnemo` (or PyTorch Geometric / `tensordict`) as a
   runtime or optional dependency is a separate, deferred, flag-first
   decision** — its own ADR plus a PRINCIPLES.md entry — triggered only
   when a concrete baseline requires it. v0.1's prior-paper GNN baselines
   (ADR-0015) ship as their existing implementations; they are not
   ported onto PhysicsNeMo for v0.1.

4. **StructBench does not inherit PhysicsNeMo's coupling anti-patterns.**
   Normalization and any physics integrator stay out of model `forward`
   (they live in `datasets/` and `eval/`); models remain pure
   tensor→tensor; the rollout/evaluation protocol is model-agnostic and
   lives in `eval/`. (PhysicsNeMo's structural examples bake normalization
   and integration into the model's `forward(sample, data_stats=...)`,
   which would break the model-agnostic boundary `eval/` depends on.)

## Alternatives considered

- **Adopt PhysicsNeMo's data/processing abstractions as StructBench's
  foundation** ("a structural domain on top of PhysicsNeMo"). Rejected.
  There is no canonical PhysicsNeMo schema to conform to; the abstractions
  are mid-refactor; and subordinating the substrate would forfeit
  framework-neutrality (the property that lets StructBench evaluate
  PhysicsNeMo and non-PhysicsNeMo models alike) while diluting the
  benchmark identity that is StructBench's actual differentiator. It would
  relocate, not cure, the ad-hoc per-dataset reader problem.

- **Build all v0.1 baselines on PhysicsNeMo now** (adopt it as a
  model/training dependency immediately). Rejected for v0.1. The
  prior-paper GNNs are already implemented; porting buys little, imports a
  heavy CUDA/PyG dependency into the first release, and couples the
  baselines to a framework mid-refactor. Left open as a per-baseline
  opt-in later (point 3).

- **Treat PhysicsNeMo purely as a competitor and ignore it.** Rejected.
  Its model implementations (notably `HybridMeshGraphNet` with separate
  mesh and contact edges) and structural example patterns are genuinely
  reusable, and a benchmark gains from being able to evaluate
  PhysicsNeMo-trained models. Ignoring it would reinvent wheels for no
  benefit.

- **Adopt Hydra/OmegaConf as the configuration/orchestration layer**
  (mirroring the examples). Rejected as a foundational choice — it is a
  heavyweight dependency, and a lightweight name-keyed registry preserves
  neutrality. May be revisited for `cli/` ergonomics as a separate
  decision.

## Consequences

- `core/` stays clean of PhysicsNeMo. The only PhysicsNeMo-facing code is
  an opt-in `case → PyG Data` adapter in `datasets/` and optional
  `physicsnemo.Module`-conforming baselines in `models/`, both gated
  behind the deferred dependency decision (point 3).

- The benchmark-defining work is confirmed StructBench-unique, because
  PhysicsNeMo provides no template for it: a canonical, hash/seed-pinned
  train/val/test split (PhysicsNeMo uses arbitrary runtime directories);
  model-agnostic metrics in physical units, including the Voigt
  stress/strain fields PhysicsNeMo's `crash` metric skips; a leaderboard
  submission validator; and a formal parametric-space + data-generation
  protocol per benchmark. These are built from first principles.

- The broader program's brain and body layers remain entirely outside
  PhysicsNeMo's footprint; no compatibility constraint applies there.

- This ADR sets posture only. It adds no dependency and changes no public
  API. Concrete adapters/baselines and the dependency decision follow in
  their own changes.

### Operational guidance: PhysicsNeMo → StructBench layer map

Derived from a verified read of the `crash`, `drop_test`, and
`deforming_plate` examples. Verdicts: **borrow** (reimplement the pattern
in our own code) · **adapt** (use the idea but it needs real changes to
fit our schema/neutrality) · **interop** (an opt-in PhysicsNeMo type at the
model edge) · **skip** (PhysicsNeMo-specific, not useful) · **gap**
(PhysicsNeMo lacks it; a benchmark must build it). Future sessions
building these layers consult this map rather than re-deriving it.

**`core/` (schema + io)**
- *skip* the storage formats (VTP/VTU/TFRecord readers); StructBench reads
  d3plot directly (ADR-0016), which is earlier and lossless versus
  PhysicsNeMo's already-Curator-processed VTU.
- *borrow* the extraction logic into ingestion adapters: "absolute
  position = coords + displacement" reconstruction; cell→point promotion
  of Von Mises stress.

**`datasets/` (case → model-ready tensors) — most borrowing lives here**
- *borrow*: the in-memory sample shape (per-node feature dict + `[N, T,
  Fo]` targets + global scalars + optional graph); mesh-edges-from-element
  cells (per-element-type edge templates, undirected + coalesce);
  relative-displacement edge features (`disp ‖ ‖disp‖`); one-hot
  node-type / BC encoding (derive from BC tags / `part_id`); next-step
  delta targets (forward difference); `log1p`/`expm1` transform for
  wide-range stress (paired forward/inverse, applied here — never written
  into the canonical HDF5); train-only normalization stats persisted and
  reused; preprocess-once / cache-tensors two-stage pipeline.
- *adapt*: the split-aware `Dataset → torch_geometric.data.Data` adapter
  (re-point from TFRecord to the canonical HDF5 Case);
  **world/contact edges via radius search** — the key idea for RC-beam
  (rebar↔concrete) and segmented-beam contact, but derive contact edges
  from the Case's contact/part definitions, not only a geometric radius.
- *interop*: `torch_geometric.data.Data`/`HeteroData` as the graph type the
  adapter emits.

**`models/` (reference baselines)**
- *interop*: `physicsnemo.Module` + `save_checkpoint`/`load_checkpoint`;
  `HybridMeshGraphNet` (dual mesh + contact-edge GNN — the canonical
  contact-aware baseline for RC-beam / segmented-beam); the `MeshGraphNet`
  I/O contract.
- *borrow*: the rollout-scheme taxonomy (one-shot / autoregressive-BPTT /
  teacher-forced one-step / time-conditional) as named, swappable
  prediction modes; one-shot output packing; MSE as the default
  *training* objective; Adam+cosine or `LambdaLR` exp-decay + AMP/DDP as a
  per-baseline protocol.
- *skip*: Muon / `CombinedOptimizer` (physicsnemo.optim-specific, needs
  torch ≥ 2.9); the `model(sample, data_stats=...)` signature that leaks
  normalization into the model.

**`eval/` (model-agnostic metrics + protocols)**
- *adapt*: relative-L2 per-field metric, reimplemented over canonical
  HDF5 response fields including Voigt stress/strain, in physical units;
  per-field error decomposition (position vs. stress) as reduced scalars
  on the held-out test split; probe/region-of-interest kinematics
  (declare probe regions in the benchmark definition; emit a scalar).
- *borrow*: autoregressive rollout *as an evaluation protocol* driving any
  baseline over the response time axis (PhysicsNeMo's `deforming_plate`
  `inference.py` `predict()` loop is a working reference); the
  inference-emits-artifacts / separate-evaluator-scores decoupling, but
  round-tripping predictions into canonical HDF5 rather than `.vtp`.
- *gap*: leaderboard submission validator; single comparable aggregate
  scalar; protocol object binding metric + split + seed; strict-SI at the
  evaluation boundary.

**`benchmarks/` (problem definitions)**
- *borrow*: the experiment description that pins one problem → one dataset
  → its reference baselines.
- *adapt*: per-run scenario scalars (PhysicsNeMo's `global_features.json`:
  material scales, rigid-wall angles) are the benchmark's parametric-space
  sample → fold into `Case.metadata` and declare the varied parameters in
  the benchmark definition, not a loose JSON sidecar keyed by filename.
- *gap*: fixed/versioned/hashed canonical split; formal parametric space +
  data-generation protocol; target definition (and any transform) pinned
  as part of the problem so all submissions are scored in the same space.

**`cli/`**
- *adapt*: config-driven baseline/dataset selection and the
  model↔datapipe compatibility guard are good ideas, but Hydra/OmegaConf
  is a heavy dependency; use a lightweight name-keyed registry to stay
  framework-neutral.
