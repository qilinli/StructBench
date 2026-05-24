# 0016 — LS-DYNA d3plot is the canonical ingestion path; general adapter on lasso-python

**Status**: Accepted
**Type**: Durable
**Date**: 2026-05-24

## Context

ADR-0015 commits v0.1 to shipping a portfolio of existing LS-DYNA datasets
as benchmarks. ADR-0011/0012/0013 commit the substrate to a single
canonical case format (HDF5, schema fixed). The missing piece is the
ingestion path that fills the gap between the two: how does LS-DYNA output
become canonical HDF5?

The user's existing practice is ad-hoc per-paper post-processing scripts
that extract whichever response quantities a given paper needed. This
scatters extraction logic across scripts, re-derives common quantities for
each new dataset, and means the choice of "what's available" is fixed at
extraction time — if a future analysis needs a field the original script
didn't extract, the script must be re-run. This pattern is exactly what
the substrate-layer commitment in ADR-0014 is meant to escape.

The substrate-layer move is a single general adapter that handles any
LS-DYNA d3plot uniformly, extracts everything once, and writes the
canonical HDF5. Downstream analysis is then downstream's job; ingestion
loses no information.

`lasso-python` (`lasso.dyna.D3plot`, MIT-licensed, actively maintained) is
the de-facto pure-Python d3plot reader and is already in use in the user's
ad-hoc scripts. Building a parser from scratch is not on the table.

## Decision

1. **Canonical ingestion path:** LS-DYNA d3plot binary + paired deck `.k`
   text → canonical HDF5 (per ADR-0013).

2. **Adapter location:** `src/structbench/core/io/lsdyna.py`, exposing
   general primitives (`read_d3plot`, `extract_geometry`,
   `extract_response`, …) and a top-level `lsdyna_to_case()` that takes a
   d3plot path and a deck path and returns a `Case`. The adapter writes
   via the existing `write_case()` in `core/io/`.

3. **Implementation basis:** d3plot parsing wraps
   `lasso.dyna.D3plot` from `lasso-python`. Deck parsing is local custom
   code reading the handful of `*`-card kinds the schema needs:
   `*PART`, `*MAT_*`, `*SECTION_*`, `*BOUNDARY_*`, `*LOAD_*`,
   `*INITIAL_*`, `*CONSTRAINED_*`, `*CONTACT_*`, `*DEFINE_CURVE`,
   `*SET_*`. Cards not in this list are ignored on first read; the
   adapter may grow card coverage as new datasets surface needs.

4. **Extraction policy:** extract everything d3plot contains by default —
   all available response fields, all frames, all element types — and
   write them through the canonical schema. No feature engineering at
   ingestion; downstream analysis is downstream's job. A selective mode
   may be added later as an opt-in, but the default and the public-API
   contract is extract-everything.

5. **Unit handling:** the adapter applies unit conversion at the
   write boundary, enforcing the strict-SI canonical form mandated by
   ADR-0012. The source-deck unit convention is read from the
   `*CONTROL_UNITS` card when present; when absent, the per-dataset
   glue specifies it. The original convention is preserved in
   `metadata/source_units` per the schema.

6. **Per-dataset glue lives outside the package** under
   `data_generation/lsdyna/<dataset>/`, per ARCHITECTURE.md and ADR-0010.
   Glue knows where the d3plot and deck live, the dataset's unit
   convention if not in the deck, and any dataset-specific naming. Glue
   must not manipulate response data — that bypasses the canonical
   extraction and reintroduces the ad-hoc pattern this ADR exists to
   end.

7. **Dependency:** `lasso-python` is added to the approved runtime
   dependency list in PRINCIPLES.md.

## Alternatives considered

- **Build a d3plot parser from scratch.** Rejected. Months of work to
  reproduce what `lasso-python` already does; no compensating benefit.

- **Ingest from post-processed forms** (npz, CSV — whatever the prior
  papers' scripts produced) **rather than directly from d3plot.**
  Rejected. Those forms are derived and information-lossy; canonical
  extraction from the source format avoids losing information that
  downstream may later need. The whole point is to escape the
  "re-run the script when a new field is needed" loop.

- **Per-dataset adapters with no shared general code.** Rejected — that
  is the pattern this ADR exists to escape. Shared code in
  `core/io/lsdyna.py`; per-dataset glue stays thin.

- **Selective extraction as the default** (user passes a list of fields
  to extract). Rejected as default behaviour because it reintroduces the
  "I trained on data missing field X" surprise. Extract-everything is
  the default; selectivity may be added later as an opt-in.

- **Bake unit conversion into per-dataset glue rather than the
  adapter.** Rejected. ADR-0012 mandates strict SI in the canonical
  form, so converting at the adapter boundary is the right enforcement
  point. Glue is for dataset-specific *inputs* to the adapter, not for
  manipulating the canonical *output*.

## Consequences

- `lasso-python` becomes a runtime dependency. Added to the approved
  list in PRINCIPLES.md as a downstream edit on acceptance.

- The first concrete `core/io/lsdyna.py` content is a generalised
  reimplementation of `scratch/adapt_taylor.py`'s logic, reading
  directly from d3plot rather than the post-processed npz. The scratch
  file is retired once the adapter handles the Taylor 2D case
  end-to-end.

- File sizes grow relative to selective extraction (because everything is
  extracted), but stay bounded by the gzip+chunking choices in ADR-0013.
  Re-examine only if size becomes a practical problem.

- The user's ad-hoc post-processing scripts can be retired in favour of
  analysis-on-canonical-HDF5 once trust in the adapter is established.
  Migrating those scripts is not in scope for this ADR but is a natural
  consequence.

- Schema diversity across the v0.1 datasets (SPH for Taylor; solid +
  beam + discrete with rebar coupling for RC; segmented-beam structure)
  will pressure-test the schema and the deck-card coverage. Gaps are
  handled in their own amendment ADRs as they surface — this ADR does
  not pre-solve them.

- The architecture extends solver-agnostic per ADR-0004: future non-
  LS-DYNA adapters live as siblings — `core/io/kratos.py`,
  `core/io/openradioss.py`, etc. — each free to choose its own parsing
  library. This ADR is about LS-DYNA specifically.
