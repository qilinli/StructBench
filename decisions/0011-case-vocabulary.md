# 0011 — Case vocabulary for the data record

**Status**: Accepted
**Type**: Durable
**Date**: 2026-04-27

## Context

The earlier name "asset model schema" for StructBench's central data record collided with two existing meanings of "asset":

- In structural engineering, "asset" refers to a physical structure — a bridge, a building, a wharf. ARCHITECTURE.md's reserved `deploy/` namespace already uses the word in exactly this sense ("asset onboarding"). Continuing to call the data record an "asset" forced the term to do two jobs in our own documents.
- In existing ML practice in this group, the unit being trained on is one simulation run (one `.npz` per run, hundreds per dataset, several datasets). That unit is more naturally a *case* than an *asset*.

We needed a record-level term that:

- Reads naturally to a structural engineer (does not silently rename "asset"),
- Reads naturally to an ML practitioner (one case = one training example),
- Carries through to post-v0.1 SHM, where one observed event on a real asset is a record,
- Composes well with whatever sub-structure is chosen at field level,
- Does not pre-commit the persistence layout.

## Decision

The data record is called a **case**. One case = one file = one ML data example.

Its conceptual content is a triple:

- **Specimen** — the structure being studied (geometry, topology, materials, boundary conditions, sensor placements where present).
- **Scenario** — the loading or event applied to it (impact, blast, observed earthquake/wind, etc.).
- **Response** — the resulting temporal evolution under that scenario (per-node / per-element / global state, sensor readings).

This `case = specimen + scenario + response` triple is **conceptual vocabulary, not structural layout**. The on-disk schema's groups will follow the data's natural shape — driven by what makes solver-output adapters and ML readers cleanest — not by the conceptual triple. Solver inputs (LS-DYNA `.k`, ANSYS `.imp`) interleave specimen and scenario across the same set of cards, so forcing a specimen/scenario split in storage would be an artificial cut.

The temporal axis inside a `response` uses two further terms:

- **Frame** — a single time slice of the response (one image in the "video" of state evolution).
- **Transition** — a pair of consecutive frames `(frame_t, frame_{t+1})`, the natural unit for auto-regressive ML training.

The word **asset** is reserved for the physical-structure / deployment meaning. It survives in `deploy/`'s "asset onboarding" and may appear as an `asset_id` metadata field on cases that came from real-world observation, so that many cases observed on the same physical structure can be linked.

## Alternatives considered

- **Keep "asset model"**: rejected. Collides with `deploy/`'s "asset onboarding" and with the engineering meaning of "asset" as a physical structure.
- **Sample**: clean ML term. Rejected because it has weaker engineering connotation (a "sample of concrete", a statistical sample) and "case" reads more naturally inside engineering documents while remaining intelligible to ML readers.
- **Realisation**: precise in stochastic mechanics, but more academic and awkward at code sites; less familiar to the broader audience.
- **Run / Simulation**: too solver-specific; breaks for SHM observations of real assets.
- **Trial / Instance**: weaker fit on either axis.
- **Snapshot for the consecutive-frame pair**: rejected in favour of **transition**, which is the conventional term in autoregressive / dynamics / RL contexts.
- **Structurally enforcing the specimen/scenario/response hierarchy as top-level folders or HDF5 groups**: rejected. Specimen and scenario are not separable in solver inputs; reifying the conceptual split as layout would force adapters to do an artificial cut and would not save anything downstream.

## Consequences

- `ARCHITECTURE.md`, `CLAUDE.md`, and ADR-0009 are updated to use "case schema" in place of "asset model schema". `VISION.md` and `deploy/`'s "asset onboarding" are unaffected.
- The vocabulary is now stable enough to support Stage 2 (field-level definitions) and Stage 3 (HDF5 persistence) without rename churn.
- Future references — in code, docs, ADRs, dataset metadata — should use `case`, `specimen`, `scenario`, `response`, `frame`, `transition` consistently. Deviation should be flagged.
- The on-disk layout is deliberately left undefined here; that is a separate decision (likely its own ADR when settled).
- Whether a stored case must roundtrip back to its original solver deck (LS-DYNA `.k`, etc.) is left open and will be addressed in a future ADR if it becomes a binding requirement.
- Sub-questions still open at field level: where initial conditions, sensor placements, and solver/observation settings live in the schema (Specimen vs Scenario vs Metadata). These are field-level decisions and don't affect the vocabulary established here.
