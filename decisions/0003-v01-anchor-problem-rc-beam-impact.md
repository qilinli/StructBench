# 0003 — v0.1 anchor problem is impact on RC beams

**Status**: Accepted
**Type**: Durable
**Date**: 2026-04-24

## Context

v0.1 requires a single well-defined benchmark problem. The problem choice shapes the initial dataset, the reference model, and the first paper. Changing the anchor problem later is expensive because data generation and schema design cascade from it.

Constraints on the choice:
- Must be in the author's domain of strongest expertise (published work, validated simulation experience).
- Must have clean loading parameterisation (few parameters, easy to vary).
- Must have sufficient nonlinearity to make surrogate modelling non-trivial.
- Must have a clear upgrade path to harder problems in v0.2+.
- Must have experimental literature available for validation of ground truth.

## Decision

v0.1 targets **drop-weight impact on simply-supported RC beams**. The benchmark is named `RCBeam-DropImpact-v1`. Initial validation targets experimental data from Fujikake et al. 2009 (a widely-cited dataset for drop-weight RC beam tests).

Parametric variation covers: beam span (1.0–3.0 m), cross-section, reinforcement ratio, concrete/steel material strengths, drop mass (50–500 kg), drop height (0.5–3.0 m), and impactor nose geometry.

## Alternatives considered

- **2D RC frame under seismic** (OpenSees): initially proposed. Rejected because the author has no OpenSees or seismic experience — the ramp-up cost was disproportionate.
- **RC slab under close-in TNT blast**: aligns with the author's published work. Rejected for v0.1 because the coupled physics (ALE + SPH + Lagrangian) is too complex for v0.1 and the fragmentation output is harder to standardise. Deferred to v0.2.
- **Taylor impact test**: too narrow; less useful as a community benchmark.
- **Linear elastic RC beam**: too simple; would not differentiate surrogate methods meaningfully.

## Consequences

- Starts in the author's strongest domain (impact/blast on RC via LS-DYNA).
- Clear upgrade path: v0.2 extends to RC slabs under blast; later versions add 3D geometry, beam-column joints, columns, etc.
- Dataset is generated from simulation; ground truth quality depends on careful material model validation against Fujikake or equivalent experiments.
- The benchmark's community positioning is impact/blast engineering (journals: *Engineering Structures*, *International Journal of Impact Engineering*, *International Journal of Protective Structures*) rather than seismic — a less crowded space with strong author credibility.
