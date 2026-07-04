"""Apply the g-mm-ms → kg-mm-ms unit correction to Concrete-Beam canonical HDF5s.

Background (ADR-0030)
---------------------
The Concrete-Beam family decks (2DNotchBeam + 1DWavePropagation) use **kg-mm-ms**
units, not g-mm-ms as recorded at conversion time.  Evidence:

- K&C material UCF = 145 000 ⟹ stress in GPa only under kg-mm-ms.
- Steel reinforcement E = 200 GPa matches kg-mm-ms.
- Measured wavefront speed ≈ 3 100 m/s matches concrete's elastic wave speed only
  when time is in ms.

Consequence: in every canonical HDF5 produced by the existing convert.py scripts,
every *mass-derived* SI quantity is stored 1 000× too small (a factor of
g/kg = 1/1000).  Kinematic fields (time, displacement, velocity, acceleration,
strain, strain_rate, effective_plastic_strain, radius) are dimensionless or depend
only on mm and ms, so they are **identical** in both systems and are left untouched.

Patch
-----
For each affected HDF5:

1. **GATE** on ``metadata.attrs["source_units"]``:
   - ``"g-mm-ms"``  → patch (multiply affected datasets by 1 000, update attr).
   - ``"kg-mm-ms"`` → SKIP (already correct; idempotent re-run).
   - anything else  → FAIL loudly (unknown provenance).

2. **Multiply by 1 000** in-place:

   ``response/element/sph/{stress, pressure, density, mass, internal_energy}``
   ``response/global/{kinetic_energy, internal_energy, total_energy}``

3. **Set** ``metadata.attrs["source_units"] = "kg-mm-ms"``.

4. After the full batch, **delete stale normalisation caches** under
   ``<h5-root>/derived/`` for both roots (caches key on field names, not values,
   so they are stale by construction after any value change).

Usage::

    # Dry-run — list files and would-patch/skip status; no writes
    python patch_units.py --dry-run

    # Real run — patch in place (expect ~10–20 min for 237 files)
    python patch_units.py

    # Override default roots (e.g. for testing on a copy)
    python patch_units.py --notch-root /alt/path/2DNotchBeam/h5_canonical \\
                          --wave-root  /alt/path/1DWavePropagation/h5_canonical

This script is not part of the importable package (ADR-0010).  Run with the
project conda/venv Python (h5py required).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

#: <repo>/data_generation/lsdyna/Concrete-Beam-unit-patch/patch_units.py
#: → repo root is parents[3], data lives at <repo-parent>/data/…
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_ROOT = _REPO_ROOT.parent / "data" / "Concrete-Beam"

_DEFAULT_NOTCH_ROOT = _DATA_ROOT / "2DNotchBeam" / "h5_canonical"
_DEFAULT_WAVE_ROOT = _DATA_ROOT / "1DWavePropagation" / "h5_canonical"

# ---------------------------------------------------------------------------
# Dataset paths to scale (all under the HDF5 root)
# ---------------------------------------------------------------------------

#: SPH element fields that are mass-derived and must be multiplied by 1 000.
_SPH_FIELDS: tuple[str, ...] = (
    "response/element/sph/stress",
    "response/element/sph/pressure",
    "response/element/sph/density",
    "response/element/sph/mass",
    "response/element/sph/internal_energy",
)

#: Global energy fields that are mass-derived and must be multiplied by 1 000.
_GLOBAL_FIELDS: tuple[str, ...] = (
    "response/global/kinetic_energy",
    "response/global/internal_energy",
    "response/global/total_energy",
)

_SCALE_FACTOR = 1_000.0

_LOG = logging.getLogger("patch_units")


# ---------------------------------------------------------------------------
# Per-file logic
# ---------------------------------------------------------------------------


def _patch_file(h5_path: Path, *, dry_run: bool) -> str:
    """Open *h5_path* and apply (or preview) the unit patch.

    Returns one of ``"PATCHED"``, ``"SKIP"`` (already correct), or raises on
    unknown ``source_units``.
    """
    with h5py.File(h5_path, "r" if dry_run else "r+") as f:
        source_units: str = f["metadata"].attrs.get("source_units", "")

        if source_units == "kg-mm-ms":
            return "SKIP"

        if source_units != "g-mm-ms":
            raise ValueError(
                f"{h5_path.name}: unexpected source_units={source_units!r}; "
                "expected 'g-mm-ms' or 'kg-mm-ms'. Refusing to patch."
            )

        # source_units == "g-mm-ms" → needs patching
        if dry_run:
            return "WOULD-PATCH"

        for ds_path in (*_SPH_FIELDS, *_GLOBAL_FIELDS):
            if ds_path not in f:
                _LOG.warning(
                    "%s: dataset %s not found — skipping", h5_path.name, ds_path
                )
                continue
            ds = f[ds_path]
            # Read, scale, write back in full (h5py direct slice assignment)
            data = ds[...].astype(np.float64) * _SCALE_FACTOR
            ds[...] = data.astype(ds.dtype)

        # Update provenance attribute
        f["metadata"].attrs["source_units"] = "kg-mm-ms"

    return "PATCHED"


# ---------------------------------------------------------------------------
# Derived-cache cleanup
# ---------------------------------------------------------------------------


def _delete_derived(h5_root: Path, *, dry_run: bool) -> int:
    """Delete every file under *h5_root/derived/* and return the count.

    The derived/ caches key on field-name hashes and shapes, not on values, so
    any value-level patch invalidates them unconditionally.
    """
    derived_dir = h5_root / "derived"
    if not derived_dir.exists():
        _LOG.info("  derived/ not found under %s — nothing to delete", h5_root)
        return 0

    deleted = 0
    for item in sorted(derived_dir.iterdir()):
        if item.is_file():
            if dry_run:
                _LOG.info("  DRY-RUN: would delete derived cache %s", item.name)
            else:
                item.unlink()
                _LOG.info("  Deleted stale cache: %s", item)
            deleted += 1

    return deleted


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------


def _run_batch(roots: list[Path], *, dry_run: bool) -> int:
    """Iterate over all .h5 files under *roots*, patch, then clean derived/.

    Returns 0 on success, 1 if any file raised an error.
    """
    all_files: list[tuple[Path, Path]] = []  # (h5_root, h5_file)
    for root in roots:
        if not root.exists():
            _LOG.error("Root not found: %s", root)
            return 1
        for f in sorted(root.glob("*.h5")):
            all_files.append((root, f))

    total = len(all_files)
    _LOG.info("Found %d .h5 file(s) across %d root(s)", total, len(roots))

    counts: dict[str, int] = {"PATCHED": 0, "WOULD-PATCH": 0, "SKIP": 0, "FAIL": 0}
    failures: list[tuple[str, str]] = []

    for _h5_root, h5_path in all_files:
        try:
            status = _patch_file(h5_path, dry_run=dry_run)
            counts[status] = counts.get(status, 0) + 1
            _LOG.info("  %-12s %s", status, h5_path.name)
        except Exception as exc:  # noqa: BLE001 — batch keeps going
            counts["FAIL"] += 1
            failures.append((h5_path.name, f"{type(exc).__name__}: {exc}"))
            _LOG.error(
                "  FAIL         %s — %s: %s", h5_path.name, type(exc).__name__, exc
            )

    # --- Derived-cache cleanup ---
    total_deleted = 0
    for root in roots:
        n = _delete_derived(root, dry_run=dry_run)
        if n:
            action = "would delete" if dry_run else "deleted"
            _LOG.info(
                "%s %d stale cache file(s) under %s/derived/",
                action.capitalize(),
                n,
                root,
            )
        total_deleted += n

    # --- Summary ---
    print()
    if dry_run:
        print(
            f"DRY RUN summary: {counts['WOULD-PATCH']} would-patch, "
            f"{counts['SKIP']} skip, {counts['FAIL']} fail "
            f"(total {total}); {total_deleted} derived cache(s) would be deleted"
        )
    else:
        print(
            f"Batch complete: {counts['PATCHED']} patched, "
            f"{counts['SKIP']} skipped, {counts['FAIL']} failed "
            f"(total {total}); {total_deleted} derived cache(s) deleted"
        )

    if failures:
        print(f"\nFailures ({len(failures)}):")
        for name, err in failures:
            print(f"  {name}: {err}")
        return 1

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the g-mm-ms → kg-mm-ms unit correction to Concrete-Beam canonical "
            "HDF5s (ADR-0030). Idempotent: already-corrected files are skipped."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "List files and would-patch/skip status without writing anything. "
            "Also previews derived-cache deletions."
        ),
    )
    parser.add_argument(
        "--notch-root",
        type=Path,
        default=_DEFAULT_NOTCH_ROOT,
        help=f"2DNotchBeam h5_canonical directory (default: {_DEFAULT_NOTCH_ROOT})",
    )
    parser.add_argument(
        "--wave-root",
        type=Path,
        default=_DEFAULT_WAVE_ROOT,
        help=(
            f"1DWavePropagation h5_canonical directory (default: {_DEFAULT_WAVE_ROOT})"
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.dry_run:
        _LOG.info("=== DRY RUN — no files will be modified ===")

    roots = [args.notch_root, args.wave_root]
    return _run_batch(roots, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
