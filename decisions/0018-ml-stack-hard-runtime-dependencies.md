# 0018 — PyTorch + PyG are hard runtime dependencies of the ML layer

**Status**: Accepted
**Type**: Durable
**Date**: 2026-06-29

## Context

ADR-0015 commits v0.1 to shipping prior-paper GNNs as reference baselines, and
ARCHITECTURE.md places those reference models inside the package (`models/`),
with data preparation in `datasets/` and evaluation in `eval/`. The first such
baseline is a Graph Network Simulator (GNS) for the Taylor 2D benchmark, ported
from the user's existing `sgnn` code. It is built on **PyTorch** and
**PyTorch Geometric** (`torch_geometric`), which provide the autograd engine,
the `MessagePassing` primitive, and `radius_graph` the model relies on.

The open question was how these heavy, GPU-oriented libraries relate to the
package's dependency surface. Until now the package has held a deliberately
light core (`numpy`, `h5py`, `lasso-python`) so that someone who only wants to
read benchmark data carries no ML stack. Adding a training framework is a
posture shift and a long-term commitment, so it is recorded here.

## Decision

1. **`torch` and `torch_geometric` are hard runtime dependencies** of
   `structbench`, added to the approved runtime list in PRINCIPLES.md and to
   `[project.dependencies]` in `pyproject.toml`. `pip install structbench`
   pulls the full ML stack.

2. **Layer boundary:** the dependency is used only in the ML-facing layers —
   `datasets/` (turning canonical cases into model-ready graphs/tensors),
   `models/`, and `eval/`. **`core/` stays `torch`-free** (schema, validation,
   HDF5 I/O, and the LS-DYNA adapter remain a pure `numpy`/`h5py` substrate).
   This preserves the option to split the ML stack back out later without
   touching the substrate.

3. **CI installs the CPU build** of `torch`/`torch_geometric`; tests must not
   require a GPU (consistent with PRINCIPLES.md: deterministic, no special
   hardware). GPU is used only for actual training/evaluation runs, which are
   not part of the test suite.

## Alternatives considered

- **Optional extra (`structbench[train]`).** Keep the core light; the ML stack
  installs only on opt-in. Rejected by the maintainer in favour of a single,
  simpler install story. It remains the natural target of a future superseding
  ADR if a data-only light install is needed.

- **Training entirely outside the package** (a separate repo / the existing
  `sgnn` environment), with `structbench` providing only data loading.
  Rejected: it contradicts ADR-0015's commitment that `models/` ships reference
  models with published checkpoints.

- **Hard dependency, but allow `core/` to use `torch`.** Rejected: keeping the
  substrate `torch`-free is cheap now and keeps the door open to re-splitting.

## Consequences

- Every install of `structbench` carries `torch` + `torch_geometric` (hundreds
  of MB, GPU libraries included), even for pure data consumers. This couples
  the substrate to a training framework — an accepted trade for install
  simplicity, revisitable via a superseding ADR.

- PRINCIPLES.md's approved-dependency list and `pyproject.toml` gain both
  packages with loose lower bounds; the exact versions are pinned in the `uv`
  lockfile for reproducible training.

- The interface discipline gains a rule: `core/` must not import `torch`. A
  lint/review check enforces it.

- Version coupling: `torch_geometric` is sensitive to the `torch` version. The
  approved lower bounds and the lockfile must be kept compatible; bumping one
  is a deliberate, tested change.
