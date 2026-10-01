# 0004 — Platform is solver-agnostic; LS-DYNA for v0.1 data generation

**Status**: Accepted
**Type**: Durable
**Date**: 2026-04-24

## Context

A platform positioning itself as "open" must clarify what openness means. Two interpretations were considered:

1. Openness of the tools used to produce ground truth data (requiring open-source solvers).
2. Openness of the artifacts the community consumes (data, schema, evaluation code, reference models — regardless of how ground truth was produced).

Interpretation 1 would force a migration to an open-source solver (Kratos, Code_Aster, FEniCS, etc.), incurring significant ramp-up cost for the author and risking delayed v0.1 delivery. Interpretation 2 matches the pattern of established benchmarks: ImageNet's cameras are not open; MoleculeNet's DFT calculations use proprietary tools (VASP, Gaussian); WeatherBench uses ECMWF-generated data.

## Decision

**Openness applies to what users consume, not to what the author uses to produce ground truth.** The platform is designed as solver-agnostic: the data schema, evaluation protocol, and reference models carry no solver-specific assumptions. Users do not need any commercial software to use StructBench.

For v0.1, ground truth data is generated using **LS-DYNA**, because the author has validated LS-DYNA workflows for RC under impact/blast and existing input decks that can be parameterised quickly.

Contributions of benchmark-compatible datasets from other solvers (Kratos, OpenSees, Abaqus, etc.) are explicitly welcomed and will be incorporated into future releases, subject to schema conformance and validation checks.

## Alternatives considered

- **Open-solver-only** (Kratos or similar for v0.1): rejected because the author's lack of solver experience would push v0.1 from ~3 months to ~5+ months, and initial ground truth quality would be uncertain.
- **Single-solver commitment**: rejected because it artificially limits community contribution and fails to reflect the reality that the field uses many solvers.
- **Waiting for open solvers to mature in this problem class**: rejected because it indefinitely delays v0.1 for a limited benefit.

## Consequences

- Datasets and reference models are freely downloadable and usable without any licence dependency.
- The author can leverage existing LS-DYNA expertise and input decks for v0.1, keeping the timeline realistic.
- The schema design must be especially careful to avoid leaking LS-DYNA-specific assumptions (element formulations, material model parameters, output conventions).
- The v0.1 paper must frame this choice honestly, stating that reference data uses LS-DYNA while the platform itself is solver-agnostic.
- Community growth pathway: other groups contribute datasets from their own preferred solvers as the platform matures.
