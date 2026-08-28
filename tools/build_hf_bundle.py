"""Build the Hugging Face upload staging bundle for one benchmark.

Emits a small staging folder (metadata only — the case ``.h5`` files are
uploaded straight from the canonical archive, never copied):

- ``README.md`` — the generated archive README with HF front-matter and a
  Download section (``hf_hub_download`` snippets) prepended;
- ``card.json`` — machine-readable card metadata (ADR-0027);
- ``cases.csv`` — one row per case: split, parsed loading/geometry
  parameters, node/frame counts, file size, SHA-256 (feeds the HF dataset
  viewer and doubles as an integrity manifest). Case files the archive
  ships outside the protocol splits (e.g. Taylor's held-aside Convergence
  run) get a row with ``split=held_aside`` and blank parameter columns;
- ``decks/`` — the LS-DYNA input decks copied from the raw tree when
  ``--raw-root`` is given (full provenance at ~zero size cost).

Not part of the importable ``structbench`` package (ADR-0010) — a standalone
script that imports the installed package. DeformingPlate is refused: its
source states no data licence, so StructBench points to the DeepMind bucket
rather than rehosting (ADR-0042).

Usage (see scratch/2026-08-28-hf-upload-runbook-v2.md for the full sequence):

    python tools/build_hf_bundle.py --benchmark taylor_impact_2d \
        --data-root <...>/canonical/taylor_impact_2d \
        --raw-root  <...>/raw/taylor_impact_2d \
        --out scratch/hf-upload

Reading each ``.h5`` (counts + SHA-256) hydrates OneDrive placeholders —
run on a fast connection. ``--no-sha256`` and ``--allow-missing`` exist for
smoke runs against partial or synthetic data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import shutil
import sys
from pathlib import Path

import h5py

from structbench.benchmarks.registry import get_benchmark
from structbench.benchmarks.render import card_json, render_archive_readme

_LOG = logging.getLogger("build_hf_bundle")

#: HF dataset repo name per registry benchmark (the ``StructBench/<repo>`` id).
HF_REPOS: dict[str, str] = {
    "taylor_impact_2d": "taylor-impact-2d",
    "wave_propagation_1d": "wave-propagation-1d",
    "notch_beam_2d_impact": "notch-beam-2d-impact",
}

#: Dataset impactor names -> the published cross-section terms (paper H/W
#: notation; verified against canonical frame-0 extents, 2026-08-27).
_NOTCH_CROSS_SECTIONS = {
    "Rectangular": "plate",
    "Sphere": "disk",
    "Bullet": "rod",
}


def _case_params(benchmark: str, case_id: str) -> dict[str, object]:
    """Loading/geometry parameters parsed from a case id (per-benchmark)."""
    if benchmark == "taylor_impact_2d":
        # T-20-<L>-<V>
        _, width, length, speed = case_id.split("-")
        return {
            "bar_width_mm": int(width),
            "bar_length_mm": int(length),
            "impact_speed_ms": int(speed),
        }
    if benchmark == "wave_propagation_1d":
        # W1D-<L>-<V>
        _, length, speed = case_id.split("-")
        return {"bar_length_mm": int(length), "initial_speed_ms": int(speed)}
    if benchmark == "notch_beam_2d_impact":
        if case_id.startswith("NB-I-"):
            # NB-I-<W>-<Shape>-<n>-<V>
            _, _, width, shape, notch, speed = case_id.split("-")
            return {
                "beam_height_mm": 80,
                "beam_width_mm": int(width),
                "impactor": shape,
                "impactor_cross_section": _NOTCH_CROSS_SECTIONS[shape],
                "notch_position": notch,
                "impact_speed_ms": int(speed),
            }
        # Probe: S_<H>_<W>_V<V>_<label> (off-grid, off-centre; ADR-0026)
        _, height, width, v_token, label = case_id.split("_")
        return {
            "beam_height_mm": int(height),
            "beam_width_mm": int(width),
            "impactor": "Sphere",
            "impactor_cross_section": "disk",
            "notch_position": "",
            "impact_speed_ms": int(v_token.removeprefix("V")),
            "probe_label": label,
        }
    raise ValueError(f"no case-id parser for benchmark {benchmark!r}")


def _deck_path(benchmark: str, case_id: str, raw_root: Path) -> Path:
    """The LS-DYNA input deck for ``case_id`` in the raw tree.

    Layouts mirror each family's ``data_generation`` converter (the
    authoritative raw-tree walkers); deck filenames per converter constants.
    """
    if benchmark == "taylor_impact_2d":
        _, width, length, speed = case_id.split("-")
        return raw_root / "lsdyna" / f"{width}{length}" / speed / "Taylor.k"
    if benchmark == "wave_propagation_1d":
        _, length, speed = case_id.split("-")
        return raw_root / f"{length}_{speed}" / "WavePropagation.k"
    if benchmark == "notch_beam_2d_impact":
        if case_id.startswith("NB-I-"):
            _, _, width, shape, notch, speed = case_id.split("-")
            return (
                raw_root
                / "InitialVelocity"
                / shape
                / f"80{width}"
                / f"A{notch}{speed}"
                / "Beam1.k"
            )
        return raw_root / "2DGeneralizibility" / case_id / "Beam1.k"
    raise ValueError(f"no deck locator for benchmark {benchmark!r}")


def _h5_stats(path: Path, *, hash_files: bool) -> dict[str, object]:
    """Node/frame counts, byte size, and (optionally) SHA-256 of one case."""
    with h5py.File(path, "r") as f:
        n_nodes = int(f["nodes/coords"].shape[0])
        n_frames = int(f["response/time/t"].shape[0])
    stats: dict[str, object] = {
        "n_nodes": n_nodes,
        "n_frames": n_frames,
        "file_bytes": path.stat().st_size,
    }
    if hash_files:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                digest.update(chunk)
        stats["sha256"] = digest.hexdigest()
    return stats


def _hf_readme(spec, name: str, repo: str) -> str:
    """The archive README with HF front-matter + a Download section."""
    size_category = "n<1K" if spec.card.n_cases < 1_000 else "1K<n<10K"
    front_matter = "\n".join(
        [
            "---",
            "license: cc-by-4.0",
            f"pretty_name: {spec.card.name} (StructBench)",
            "tags:",
            "- structural-engineering",
            "- physics",
            "- simulation",
            "- lsdyna",
            "- sph",
            "- benchmark",
            "size_categories:",
            f"- {size_category}",
            # Point the Dataset Viewer at the manifest only: the .h5 case
            # files are not a viewer format, so without an explicit config the
            # viewer would guess (or fail) over the mixed tree. One subset,
            # one split, the CSV.
            "configs:",
            "- config_name: manifest",
            "  data_files:",
            "  - split: cases",
            "    path: cases.csv",
            "---",
            "",
        ]
    )
    download = "\n".join(
        [
            "## Download",
            "",
            "One case, one file — fetch exactly what you need"
            " (`pip install huggingface_hub`):",
            "",
            "```python",
            "from huggingface_hub import hf_hub_download, snapshot_download",
            "",
            "# one case",
            f'path = hf_hub_download("StructBench/{repo}",',
            '                       filename="<case_id>.h5", repo_type="dataset")',
            "",
            "# the full archive (resumable; cached under HF_HOME)",
            f'root = snapshot_download("StructBench/{repo}", repo_type="dataset")',
            "```",
            "",
            "`cases.csv` lists every case with its split and loading/geometry",
            "parameters plus a SHA-256 manifest; pin a git revision",
            '(`revision="v0.1.0"`) for reproducible pipelines. Point',
            "`structbench-train --data-root` at the snapshot directory.",
            "Code, benchmark protocol, and leaderboards:",
            "<https://github.com/qilinli/StructBench>.",
            "",
        ]
    )
    readme = render_archive_readme(spec, name)
    title, _, rest = readme.partition("\n")
    # ``download`` ends in a single newline; the extra one keeps the archive
    # README's first paragraph from merging into the Download section.
    return f"{front_matter}{title}\n\n{download}\n{rest.lstrip()}"


def build(args: argparse.Namespace) -> int:
    if args.benchmark == "deforming_plate":
        _LOG.error(
            "deforming_plate is not rehosted (ADR-0042): the source states "
            "no data licence — point users to the DeepMind bucket and the "
            "download-and-convert script instead."
        )
        return 2
    if args.benchmark not in HF_REPOS:
        _LOG.error("no HF repo mapping for %r (add it to HF_REPOS)", args.benchmark)
        return 2

    spec = get_benchmark(args.benchmark)
    repo = HF_REPOS[args.benchmark]
    data_root = Path(args.data_root)
    out = Path(args.out) / args.benchmark
    out.mkdir(parents=True, exist_ok=True)

    expected = [cid for ids in spec.splits.values() for cid in ids]
    split_of = {cid: s for s, ids in spec.splits.items() for cid in ids}
    missing = [cid for cid in expected if not (data_root / f"{cid}.h5").exists()]
    if missing:
        level = _LOG.warning if args.allow_missing else _LOG.error
        level(
            "%d/%d case files missing under %s (first: %s)",
            len(missing),
            len(expected),
            data_root,
            missing[:3],
        )
        if not args.allow_missing:
            return 1
    # Case files outside the protocol splits (e.g. Taylor's held-aside
    # Convergence run, card provenance) ship with the archive, so they get a
    # manifest row too (split=held_aside) — every .h5 in the repo has a SHA.
    extras = sorted(p.stem for p in data_root.glob("*.h5") if p.stem not in split_of)
    if extras:
        _LOG.info(
            "%d .h5 outside the protocol splits, shipped as split=held_aside: %s",
            len(extras),
            extras,
        )

    # cases.csv — first-seen column union across rows keeps benchmarks with
    # heterogeneous params (notch grid vs probe) in one flat table.
    rows: list[dict[str, object]] = []
    for cid in expected + extras:
        path = data_root / f"{cid}.h5"
        if not path.exists():
            continue
        row: dict[str, object] = {
            "case_id": cid,
            "split": split_of.get(cid, "held_aside"),
        }
        try:
            row.update(_case_params(args.benchmark, cid))
        except (ValueError, KeyError):
            # Held-aside ids sit outside the family's naming grid; the row
            # keeps counts + hash with blank parameter columns.
            pass
        row.update(_h5_stats(path, hash_files=not args.no_sha256))
        rows.append(row)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with (out / "cases.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)

    (out / "README.md").write_text(
        _hf_readme(spec, args.benchmark, repo), encoding="utf-8"
    )
    (out / "card.json").write_text(card_json(spec.card), encoding="utf-8")

    copied = 0
    if args.raw_root:
        deck_dir = out / "decks"
        deck_dir.mkdir(exist_ok=True)
        for cid in expected:
            deck = _deck_path(args.benchmark, cid, Path(args.raw_root))
            if deck.exists():
                shutil.copyfile(deck, deck_dir / f"{cid}.k")
                copied += 1
            else:
                _LOG.warning("deck missing for %s: %s", cid, deck)
        (deck_dir / "README.md").write_text(
            "LS-DYNA input decks, one per case (`<case_id>.k`), in the deck's\n"
            "native kg-mm-ms unit system — the provenance root: re-running a\n"
            "deck in LS-DYNA regenerates the case's raw output, which the\n"
            "repository's adapter converts to the canonical HDF5 shipped here\n"
            "(strict SI, ADR-0012/0016).\n",
            encoding="utf-8",
        )

    total = sum(int(r["file_bytes"]) for r in rows)
    n_protocol = sum(r["split"] != "held_aside" for r in rows)
    print(
        f"{args.benchmark}: staged {out} — cases.csv ({n_protocol}/{len(expected)} "
        f"protocol cases + {len(rows) - n_protocol} held aside, "
        f"{total / 1e9:.2f} GB in archive), README.md, card.json, "
        f"{copied} decks. Upload the canonical dir first, this staging dir "
        f"second (see the runbook)."
    )
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmark", required=True)
    parser.add_argument(
        "--data-root",
        required=True,
        help="the benchmark's canonical archive directory (<...>/canonical/<name>)",
    )
    parser.add_argument(
        "--raw-root",
        default=None,
        help="the benchmark's raw family directory (<...>/raw/<family>); "
        "when given, input decks are copied into the bundle",
    )
    parser.add_argument("--out", default="scratch/hf-upload")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="tolerate missing case files (smoke runs on partial data)",
    )
    parser.add_argument(
        "--no-sha256",
        action="store_true",
        help="skip per-file hashing (fast smoke runs)",
    )
    return build(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
