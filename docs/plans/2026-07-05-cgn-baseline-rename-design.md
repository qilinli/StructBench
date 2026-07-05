# Design: the reference baseline is CGN (Concrete Graph Network)

**Date**: 2026-07-05
**Status**: Approved (maintainer, in-session)
**Scope**: rename + attribution only — no architecture or recipe change, no
retraining. Touches `src/structbench/{models,config,cli}`, `tests/`,
`configs/`, `hpc/dug/`, README, ARCHITECTURE, generated views, ADR-0034.

---

## Problem

The reference baseline has been named "GNS" after its architectural origin
(Sanchez-Gonzalez et al. 2020), but the implementation is the lineage of the
maintainer's own published model — developed and validated on exactly the
Taylor-bar and notch-beam datasets StructBench ships — and the platform
should name and cite that model as its baseline.

## Identity (maintainer decision, 2026-07-05)

- The baseline is **CGN — Concrete Graph Network**:
  Li, Q., Wang, Z., Li, L., Hao, H., Chen, W., & Shao, Y. (2023).
  *Machine learning prediction of structural dynamic responses using graph
  neural networks.* Computers & Structures, 289, 107188.
  https://doi.org/10.1016/j.compstruc.2023.107188
  (volume/article verified 2026-07-05 via the ACM DL record).
- **GNS (Sanchez-Gonzalez et al. 2020) is the architectural origin**, kept
  as attribution — in the CGN module docstring, ARCHITECTURE.md, and
  ADR-0034 — not as a separate baseline.
- Recipe untouched: the ADR-0028/0032 values stand (hidden 64, 5 MP steps,
  window 11, radius 1.5, lr 1e-4, grad clip). The paper's reference
  capacity (hidden 128) remains a Phase-2 ablation axis.

## Changes

### Code (clean renames — pre-v0.1, no public consumers)

| From | To |
|---|---|
| `src/structbench/models/gns/` | `src/structbench/models/cgn/` (docstring gains the citation + GNS-origin note) |
| `GNSConfig` (in `structbench.config`) | `CGNConfig` |
| `tests/models/gns/` | `tests/models/cgn/` |
| family registry `{"gns": GNSConfig}` | `{"cgn": CGNConfig, "gns": CGNConfig}` — `"gns"` stays as a **deprecated legacy alias** so the 2026-07-03/04 DUG fleet run dirs (config.json `family = "gns"`) remain re-evaluable; new runs record `cgn` |
| internal references in `cli/train.py`, `config.py`, `datasets/particle.py`, `viz/__main__.py`, `benchmarks/render.py` + `results.py` examples | renamed |

### Configs

`configs/<benchmark>/{gns,gns_smoke}.toml` → `{cgn,cgn_smoke}.toml`;
`[model] family = "cgn"`; header comments re-attributed (CGN, cite ADR-0034).

### Docs

- README: quickstart config paths, repository-layout line
  (`models/cgn/  # reference CGN baseline (Li et al. 2023)`), roadmap items
  ("Trained CGN baseline…", "Three trained CGN baselines", "multi-scale
  CGN (spec Proposed as MS-GNS)"). Genre prose ("GNS-style codebases" in
  Why) stays — it describes the ecosystem, not our baseline.
- docs/ARCHITECTURE.md `models/` section: CGN named + cited; GNS-origin
  note.
- Archive-README usage snippet renders `configs/<name>/cgn.toml`;
  `docs/benchmarks.md` + four archive READMEs regenerated.
- `hpc/dug/`: slurm scripts + README config paths and run-dir names.
- Historical ADRs (0018/0019/0028/0032) and dated plans/specs keep their
  GNS wording — records of their time (ADR-0009 precedent). The Proposed
  MS-GNS spec is renamed only when picked up.

### ADR-0034 (Durable)

Baseline identity: CGN named/cited as the official reference baseline; GNS
2020 recorded as origin; family-key rename with legacy alias; recipe
explicitly out of scope.

## Verification

Full pytest (renamed test tree), ruff + format, mypy, `gen_benchmark_docs
--check` after regeneration, and a final repo grep proving remaining
"GNS"/"gns" mentions are only: the legacy alias + its comment, the
origin-attribution lines, genre prose, and historical documents.

## Out of scope

Recipe/architecture changes; retraining; MS-GNS spec rewrite; any data
movement.
