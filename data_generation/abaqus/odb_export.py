"""Export Abaqus output databases to the neutral ``abaqus-npz/1`` intermediate.

Runs under Abaqus's own interpreter (Python 3.10, numpy, odbAccess)::

    abaqus python data_generation/abaqus/odb_export.py --sweep <work-root>/<name>
        [--cases ID ...]

Dataset-blind: it writes whatever the ODB holds, for cases whose ``run.json``
says ``completed`` and that have no ``<case_id>.npz`` yet. It must stay parseable
as Python 3.10 and import ``odbAccess`` only inside :func:`export` (ADR-0069).

Layout of ``<case_id>.npz`` (``np.savez_compressed``, no pickles)::

    manifest                                        0-d str: JSON (format, odb_sha256,
                                                    abaqus_release, precision,
                                                    materials, sections, fields,
                                                    skipped)
    mesh/<instance>/node_labels                     (n,) int64
    mesh/<instance>/node_coords                     (n, d) float64
    mesh/<instance>/elements/<type>/labels          (e,) int64
    mesh/<instance>/elements/<type>/connectivity    (e, k) int64 node labels
    step/<step>/frame_times                         (T,) float64, step time
    field/<step>/<name>/<instance>/data             (T, m, c) as stored
    field/<step>/<name>/<instance>/node_labels      (m,) int64, or element_labels
    field/<step>/<name>/<instance>/integration_points  (m,) int64, element fields
    history/<step>/<region>/<output>                (s, 2) float64 (time, value)

Units are the deck's own; ids are Abaqus labels, never minted here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

FORMAT = "abaqus-npz/1"
ASSEMBLY = "ASSEMBLY"  # instance key for assembly-level (instance-less) data


def selected_cases(sweep: Path, cases: list[str] | None = None) -> list[Path]:
    """Case folders whose run completed and that have no export yet."""
    out = []
    for case_dir in sorted(p for p in sweep.iterdir() if p.is_dir()):
        if cases and case_dir.name not in cases:
            continue
        run = case_dir / "run.json"
        if not run.is_file() or (case_dir / f"{case_dir.name}.npz").exists():
            continue
        if json.loads(run.read_text(encoding="utf-8")).get("status") != "completed":
            continue
        out.append(case_dir)
    return out


def _labels(values) -> np.ndarray:
    """A block's label array; odbAccess gives ``None`` for labels it lacks."""
    return np.asarray(() if values is None else values, dtype=np.int64).reshape(-1)


def stack_blocks(blocks: list) -> dict:
    """Concatenate one frame's bulk-data blocks for one instance."""
    data = [np.asarray(b.data) for b in blocks]
    data = [d.reshape(-1, 1) if d.ndim == 1 else d for d in data]
    elements = [_labels(b.elementLabels) for b in blocks]
    if all(e.size for e in elements):
        return {
            "data": np.concatenate(data),
            "labels": np.concatenate(elements),
            "label_kind": "element",
            "integration_points": np.concatenate(
                [_labels(b.integrationPoints) for b in blocks]
            ),
        }
    nodes = [_labels(b.nodeLabels) for b in blocks]
    return {
        "data": np.concatenate(data),
        "labels": np.concatenate(nodes).astype(np.int64),
        "label_kind": "node",
        "integration_points": None,
    }


def time_series(per_frame: list) -> dict:
    """Stack frames to ``(T, m, c)``; the labels must not change between frames."""
    first = per_frame[0]
    for frame in per_frame[1:]:
        if not np.array_equal(frame["labels"], first["labels"]):
            raise ValueError("labels change between frames")
    return dict(first, data=np.stack([f["data"] for f in per_frame]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mesh(odb, arrays: dict) -> None:
    for iname, inst in odb.rootAssembly.instances.items():
        base = f"mesh/{iname}"
        arrays[f"{base}/node_labels"] = np.array(
            [n.label for n in inst.nodes], dtype=np.int64
        )
        arrays[f"{base}/node_coords"] = np.array(
            [n.coordinates for n in inst.nodes], dtype=np.float64
        )
        by_type = {}
        for element in inst.elements:
            by_type.setdefault(element.type, []).append(element)
        for etype, elements in by_type.items():
            arrays[f"{base}/elements/{etype}/labels"] = np.array(
                [e.label for e in elements], dtype=np.int64
            )
            arrays[f"{base}/elements/{etype}/connectivity"] = np.array(
                [e.connectivity for e in elements], dtype=np.int64
            )


def _fields(sname: str, frames, arrays: dict, manifest: dict) -> None:
    for fname in sorted(frames[0].fieldOutputs.keys()):
        if any(fname not in f.fieldOutputs.keys() for f in frames):
            manifest["skipped"].append(
                {"step": sname, "field": fname, "reason": "absent_in_some_frames"}
            )
            continue
        per_instance = {}
        positions = {}
        for frame in frames:
            groups = {}
            for block in frame.fieldOutputs[fname].bulkDataBlocks:
                key = block.instance.name if block.instance is not None else ASSEMBLY
                groups.setdefault(key, []).append(block)
                positions[key] = str(block.position)
            for key, blocks in groups.items():
                per_instance.setdefault(key, []).append(stack_blocks(blocks))
        components = [str(c) for c in frames[0].fieldOutputs[fname].componentLabels]
        for key, series in per_instance.items():
            if len(series) != len(frames):
                manifest["skipped"].append(
                    {
                        "step": sname,
                        "field": fname,
                        "instance": key,
                        "reason": "absent_in_some_frames",
                    }
                )
                continue
            stacked = time_series(series)
            base = f"field/{sname}/{fname}/{key}"
            arrays[f"{base}/data"] = stacked["data"]
            arrays[f"{base}/{stacked['label_kind']}_labels"] = stacked["labels"]
            if stacked["integration_points"] is not None:
                arrays[f"{base}/integration_points"] = stacked["integration_points"]
            manifest["fields"][base] = {
                "component_labels": components,
                "position": positions[key],
                "label_kind": stacked["label_kind"],
            }


def export(odb_path: Path, out_path: Path) -> dict:
    """Write ``out_path`` from ``odb_path``; return the manifest."""
    from odbAccess import openOdb  # Abaqus interpreter only

    arrays = {}
    manifest = {
        "format": FORMAT,
        "odb_sha256": _sha256(odb_path),
        "fields": {},
        "skipped": [],
    }
    odb = openOdb(str(odb_path), readOnly=True)
    try:
        manifest["abaqus_release"] = str(odb.jobData.version)
        manifest["precision"] = str(odb.jobData.precision)
        manifest["materials"] = sorted(odb.materials.keys())
        manifest["sections"] = sorted(odb.sections.keys())
        _mesh(odb, arrays)
        for sname, step in odb.steps.items():
            frames = step.frames
            arrays[f"step/{sname}/frame_times"] = np.array(
                [f.frameValue for f in frames], dtype=np.float64
            )
            _fields(sname, frames, arrays, manifest)
            for rname, region in step.historyRegions.items():
                for oname, output in region.historyOutputs.items():
                    arrays[f"history/{sname}/{rname}/{oname}"] = np.array(
                        output.data, dtype=np.float64
                    )
    finally:
        odb.close()
    arrays["manifest"] = np.array(json.dumps(manifest, sort_keys=True))
    partial = out_path.with_name(out_path.stem + ".partial.npz")
    np.savez_compressed(partial, **arrays)
    partial.replace(out_path)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args(argv)
    todo = selected_cases(args.sweep, args.cases)
    failures = 0
    for k, case_dir in enumerate(todo, 1):
        name = case_dir.name
        tag = f"[{k}/{len(todo)}] {name}"
        try:
            manifest = export(case_dir / f"{name}.odb", case_dir / f"{name}.npz")
            n_fields, n_skipped = len(manifest["fields"]), len(manifest["skipped"])
            print(f"{tag} exported: {n_fields} fields, {n_skipped} skipped")
        except Exception as exc:  # report, keep going; no partial .npz survives
            failures += 1
            print(f"{tag} FAILED: {type(exc).__name__}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
