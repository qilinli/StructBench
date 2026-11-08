"""Build a benchmark's reference-sensitivity record (ADR-0076).

Reads every case of the benchmark's sweep from a canonical archive, computes
the quantities of interest on the stored trajectories, takes the reference
sensitivities along each sweep line with their declared errors, and writes
``docs/gradients/<benchmark>.json`` and its Markdown rendering. The record
is what a reader sees before any model's number; the drift test recomputes
the sensitivities from the stored quantities and re-renders the page.

Usage:
    python tools/gradient_reference.py taylor_impact_2d --data-root DIR
    python tools/gradient_reference.py taylor_impact_2d --check  # re-render only
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from structbench.benchmarks import get_benchmark
from structbench.datasets import load_case_trajectory
from structbench.eval import QoiInputs
from structbench.eval.sensitivity import reference_record, render_reference_markdown

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "gradients"


def build(name: str, data_root: Path, revision: str) -> dict:
    spec = get_benchmark(name)
    family = spec.gradient_family
    if family is None:
        raise SystemExit(f"{name} declares no gradient family (ADR-0076)")
    quantities: dict[str, dict[str, float]] = {}
    for split in spec.splits.values():
        for case_id in split:
            traj = load_case_trajectory(
                data_root / f"{case_id}.h5", aux_field=spec.aux_field
            )
            inputs = QoiInputs(
                time=traj.time,
                positions=traj.positions,
                aux=traj.aux[..., 0],
                particle_type=traj.particle_type,
                init=spec.card.input_frames,
            )
            quantities[case_id] = {
                qoi: float(spec.qois[qoi](inputs)) for qoi in family.noise
            }
            print(case_id, {k: round(v, 4) for k, v in quantities[case_id].items()})
    return reference_record(
        family,
        benchmark=name,
        quantities=quantities,
        source={
            "archive": f"canonical/{name}",
            "revision": revision,
            "generated": dt.date.today().isoformat(),
            "tool": "tools/gradient_reference.py",
            "input_frames": spec.card.input_frames,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--revision", default="v0.1.0")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    json_path = OUT_DIR / f"{args.benchmark}.json"
    md_path = OUT_DIR / f"{args.benchmark}.md"
    if args.check:
        record = json.loads(json_path.read_text(encoding="utf-8"))
        rendered = render_reference_markdown(record)
        if md_path.read_text(encoding="utf-8") != rendered:
            print(f"stale: {md_path}")
            return 1
        print("gradient record up to date")
        return 0
    if args.data_root is None:
        print("error: --data-root is required to build the record")
        return 2
    record = build(args.benchmark, args.data_root, args.revision)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    md_path.write_text(
        render_reference_markdown(record), encoding="utf-8", newline="\n"
    )
    print(f"wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
