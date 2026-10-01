# 0002 — Project name is StructBench

**Status**: Accepted
**Type**: Durable
**Date**: 2026-04-24

## Context

The project needs a name that will serve from v0.1 through at least a 3–5 year trajectory, during which its scope is expected to expand from benchmarks for structural simulation to include multi-modal SHM and eventually reusable deployment tools for applying ML-based monitoring to new assets.

A concern was raised that "StructBench" might sound too narrow once the scope expands beyond benchmarks. Successful platforms (Hugging Face, PyTorch, OGB) routinely outgrow their original literal meaning; what matters is citation continuity and community recognition, not literal descriptive accuracy.

## Decision

The project is named **StructBench**. Scope expansion is absorbed under this name rather than triggering a rebrand. The tagline — *"An open platform for data-driven structural engineering — benchmarks, models, and deployment tools"* — declares the broader scope without renaming.

Sub-brands (e.g., `StructBench-SHM`, `StructBench-Deploy`) may be used for specific releases or papers but remain under the unified name.

## Alternatives considered

- **StructTwin / AssetTwin / LiveTwin**: ride the digital twin vocabulary. Rejected because "digital twin" is crowded and partly discredited by vendor hype.
- **OpenSHM / OpenStruct**: ROS-style descriptive naming. Rejected because SHM is a narrower framing than the platform's eventual scope.
- **Umbrella brand with sub-brands** (StructBench as one component of a larger named platform): rejected because umbrella branding dilutes citation density and is harder to make iconic.
- **Rename as scope expands**: rejected because rebranding breaks citation continuity and sacrifices the compounding value of an established name.

## Consequences

- Citation continuity is preserved across versions and scope expansions.
- The tagline and package namespace structure must be designed from day one to accommodate scope growth.
- Some new readers in later phases may be mildly confused that "StructBench" does more than benchmarks — mitigated by the tagline and by the visibility of the wider scope in README and VISION.
- Renaming is available as an option later, but requires a deliberate decision and would be a superseding ADR.
