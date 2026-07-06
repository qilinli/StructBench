# data_generation/

Solver-specific scripts that turn raw FEM-solver output into canonical
StructBench cases. **Not importable** as part of `structbench` (ADR-0010): these
are standalone scripts, run with the project environment, that import the
installed `structbench` package.

Layout is `<solver>/<dataset>/`. Each per-dataset folder holds thin *glue*
(ADR-0016 §6): it knows where that dataset's files live, its source unit
convention, its dimensionality, and its case-id naming — and delegates **all**
extraction to `structbench.core.io`. Glue must not manipulate response data;
doing so would bypass the canonical extraction and reintroduce the ad-hoc
per-paper post-processing the substrate layer exists to end (ADR-0014, ADR-0016).

## LS-DYNA

- `lsdyna/2D-Copper-Bar-Taylor-Impact/convert.py` — batch-convert the Taylor 2D
  copper-bar SPH impact sweep to canonical HDF5 via
  `structbench.core.io.lsdyna.lsdyna_to_case`. Start with
  `python .../convert.py --dry-run`, which lists the discovered cases without
  reading (and therefore without hydrating) any d3plot.
- `lsdyna/1DWavePropagation/convert.py` — batch-convert the 1D wave-propagation
  sweep (16 runs, 4 lengths × 4 velocities) to canonical HDF5. Source units
  `kg-mm-ms` (ADR-0030). Same `--dry-run` first.
- `lsdyna/2DNotchBeam/convert.py` — batch-convert the notch-beam SPH cases
  (221 runs feeding the bend + impact benchmarks) to canonical HDF5. Source
  units `kg-mm-ms` (ADR-0030). Same `--dry-run` first.
- `lsdyna/2DNotchBeam/freeze_splits.py` — deterministically derive and freeze
  the notch-beam train/val/test split (ADR-0026); run once, its output is the
  frozen split the benchmark modules read.
- `lsdyna/Concrete-Beam-unit-patch/patch_units.py` — one-off, idempotent
  in-place fix that rescales the mass-derived fields of the already-ingested
  Concrete-Beam family (wave + notch) by ×1000 to correct the g-vs-kg mass-unit
  error (ADR-0030). Not part of normal ingestion.
