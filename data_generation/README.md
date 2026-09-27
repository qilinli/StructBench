# data_generation/

Solver-specific **glue that is not importable**: per-dataset collectors and
converters for LS-DYNA archives that predate the package's pipeline. The
Abaqus data-generation pipeline lives in the package as `structbench.datagen`
(`structbench-datagen`, ADR-0071); its conformance record is
`docs/datagen/abaqus-conformance.md` and its guide `docs/DATA_GENERATION.md`.

## LS-DYNA

- `lsdyna/STANDARD_INPUT_BLOCK.md` — what every generated LS-DYNA input switches
  on so that a run supplies the run evidence E1–E10 of ADR-0066 (energy ledger
  with every term, per-part and reaction output, integration-point fields, one
  sampling clock), and what to keep from the run folder. Draft until one
  conformance run has exercised it.
- `lsdyna/2D-Copper-Bar-Taylor-Impact/collect_run_evidence.py` — read each
  Taylor run's message file and global statistics into one whitelisted
  run-evidence record for `structbench.cli.datacheck measure --run-evidence`.
  Paths are built from case ids; nothing of the raw text is kept.
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

## Abaqus

Nothing here any more: the Abaqus pipeline is `structbench.datagen` (ADR-0071),
run as `structbench-datagen new | check | generate | run | export | convert |
validate | archive`. Its conformance record is `docs/datagen/abaqus-conformance.md`,
its guide `docs/DATA_GENERATION.md`, and the example definition ships in the
package (`structbench/datagen/examples/abaqus_conformance`).

## MeshGraphNets

- `meshgraphnets/deforming_plate/convert.py` — download-and-convert driver
  for the MeshGraphNets `deforming_plate` dataset (ADR-0042). Unlike the
  LS-DYNA sweeps, this dataset is not held on OneDrive: it carries no
  redistribution licence, so StructBench does not rehost it. See the
  per-dataset `README.md` alongside the script for the throwaway TensorFlow
  environment, the source download, and a `--limit 2` smoke invocation.
