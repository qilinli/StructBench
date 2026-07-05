# 0034 — The reference baseline is CGN (Concrete Graph Network)

**Status**: Accepted
**Type**: Durable
**Date**: 2026-07-05

## Context

Since the first implementation, the reference baseline has been named "GNS"
after its architectural origin — the Graph Network Simulator of
Sanchez-Gonzalez et al. (2020). But the implementation in `models/` is the
lineage of the maintainer's own published model, ported from its code
(`../code/sgnn`), and that model was developed and validated on exactly the
datasets StructBench ships as benchmarks: the Taylor-bar impact sweep and
the notch-beam bend/impact families. The platform's baseline should carry
that model's name and citation, not the name of the architecture it builds
on.

## Decision

1. **The reference baseline is CGN — Concrete Graph Network**:
   Li, Q., Wang, Z., Li, L., Hao, H., Chen, W., & Shao, Y. (2023).
   *Machine learning prediction of structural dynamic responses using graph
   neural networks.* Computers & Structures, 289, 107188.
   https://doi.org/10.1016/j.compstruc.2023.107188
2. **GNS (Sanchez-Gonzalez et al. 2020) is credited as the architectural
   origin** — in the `models/cgn` docstring, ARCHITECTURE.md, and here —
   and is not a separate StructBench baseline.
3. **Renames**: `models/gns` → `models/cgn`; `GNSConfig` → `CGNConfig`;
   `configs/<benchmark>/gns*.toml` → `cgn*.toml` with
   `[model].family = "cgn"`; docs, roadmap, and HPC scripts follow.
4. **`"gns"` remains a deprecated model-family alias** resolving to the
   same config class and builder, so pre-rename run directories (whose
   `config.json` records `family = "gns"`, including the 2026-07-03/04 DUG
   fleet) stay loadable and re-evaluable. New runs record `cgn`.
5. **Recipe and architecture are untouched.** The ADR-0028/0032 training
   recipe stands; the paper's reference capacity (hidden 128) remains a
   Phase-2 ablation axis, not adopted here. The Proposed multi-scale
   second-baseline spec keeps its MS-GNS filename until picked up, and is
   referred to as multi-scale CGN going forward.

## Alternatives considered

- **Keep the GNS name**: rejected — it under-credits the platform's own
  published lineage and misattributes the baseline to a paper trained on
  different physics (granular/fluid systems).
- **Carry both names as separate baselines**: rejected — there is one
  implementation; presenting it twice would fake a comparison.
- **Hard rename with no legacy alias**: rejected — it would orphan the
  existing fleet's run directories for one dict entry's worth of cost.

## Consequences

- Results recorded via ADR-0033 registries use `family="cgn"` and labels
  like "CGN baseline"; archive READMEs and `docs/benchmarks.md` render the
  new name; the citation gives external users a paper to cite for the
  baseline.
- Historical ADRs (0018, 0019, 0028, 0032) and dated plans/specs keep
  their GNS wording as records of their time (ADR-0009 precedent).
- The DUG seat should pull before its next fleet: job scripts and config
  paths changed (`configs/<benchmark>/cgn.toml`, job name `taylor-cgn`).
