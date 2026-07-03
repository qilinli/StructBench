# 0022 — FEM-convention visualization harness (`viz/`, matplotlib as optional extra)

**Status**: Proposed
**Type**: Durable
**Date**: 2026-07-03

## Context

The first trained baseline (Taylor 2D GNS) produced rollout figures ad hoc, with
a generic scientific-plotting style. The project's audience is structural
engineers whose visual vocabulary comes from FEM postprocessors — LS-PrePost,
Abaqus/CAE: the jet rainbow fringe (blue = low, red = high), a fringe bar with
evenly spaced labelled levels, physical units, deformed geometry at equal
aspect. The README's rollout GIF already uses this style. Ad hoc styling per
session drifts; physics-quantity figures need one project-level home so every
run, document, and release renders identically. Separately, inspecting the
baseline showed *why* this matters: aggregate RMSE hid qualitatively wrong
deformation that a fringe snapshot exposes immediately.

## Decision

1. A new package module `src/structbench/viz/` is the single rendering path for
   figures that show a physics quantity (von Mises stress, effective plastic
   strain, pressure, ...). It follows FEM-postprocessor conventions: jet color
   code, optionally banded (Abaqus-style) fringes, a labelled fringe bar with
   evenly spaced levels, working-frame units (mm, MPa), equal-aspect axes, the
   rigid wall drawn at its plane.
2. A field registry (`FIELDS: dict[str, FieldSpec]`) carries each quantity's
   display contract (label, unit, tick format). New quantities are added to the
   registry, not styled inline.
3. **matplotlib joins the approved list as the optional `viz` extra** — never a
   hard runtime dependency. `import structbench.viz` succeeds without it;
   plotting calls raise `ImportError` with the install instruction. This keeps
   ADR-0018's minimal hard-dependency posture intact.
4. In the dependency graph, `viz/` sits beside `benchmarks/`/`models/`/`eval/`:
   it may depend on `core/` and `datasets/` only. It plots arrays, not models.
5. `python -m structbench.viz --run <run> --data-root <data>` regenerates a
   run's standard figures (ground-truth vs prediction fringe grids per rollout
   file, optional GIFs).

## Alternatives considered

- **Keep ad hoc per-run plotting scripts.** Rejected: style drift across
  sessions is guaranteed (it already happened once), and figures are part of
  the benchmark's public face.
- **Perceptually uniform colormaps (viridis) instead of jet.** Rejected for
  physics-quantity fringes despite being the general-purpose best practice:
  matching the audience's FEM-postprocessor convention is worth more here than
  perceptual uniformity, and the README already set the expectation. Non-physics
  figures (loss curves, error accumulation) are out of this ADR's scope and do
  not use the fringe style.
- **matplotlib as a hard runtime dependency.** Rejected: training/eval on
  headless clusters must not drag in a plotting stack (ADR-0018 posture).
- **A separate plotting repo/scripts folder outside the package.** Rejected:
  ADR-0010 keeps *solver* code out of the package because it is not importable
  functionality; visualization is importable functionality with a public API,
  so it belongs in the package with tests and typing.

## Consequences

- Physics-quantity figures are consistent everywhere; a session cannot
  accidentally restyle them.
- The README GIF becomes reproducible from the package
  (`structbench.viz.animate_rollout`).
- Documentation and run summaries gain a one-command figure refresh.
- The `viz` extra must be installed where figures are produced
  (`pip install structbench[viz]`; the dev extra includes it for CI).
- Jet's known perceptual caveats are accepted knowingly for fringe plots;
  quantitative judgements should rest on the metrics, not on hue reading.
