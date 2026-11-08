"""Run a submitter's container on the published blind inputs and compare its
predictions with the submitted ones: the verified tier (ADR-0074 clause 6;
open-evaluation plan §7).

Usage:
    python tools/verify_submission.py --image <oci image> --cases <dir> \
        --submission <dir> --out verification.json [--budget-seconds 3600] \
        [--submitted-scores scores.json --reproduced-scores scores2.json] \
        [--dry-run]

The container contract: the image's entry point is
``structbench-submit predict --cases <dir> --out <dir>``; it reads the
published inputs from ``/cases`` (read-only), writes prediction files in the
scoring tool's format to ``/out``, has no network, and finishes within the
budget (one GPU-hour or twenty-four CPU-hours; ``--budget-seconds`` is the
wall clock the steward grants). A result is *verified* when the container's
predictions reproduce the submitted ones to a pooled relative L2 of 1e-3 per
response field on every case and, when both score files are given, its
public-test metrics reproduce the submitted ones to the same relative
tolerance. Otherwise the result stays self-reported, and this file says
where it departed.

Only the steward runs this, on DUG or the maintainer's machine; the
withheld reference outputs are not involved — the comparison is between two
sets of predictions and two score files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from structbench.core import read_case

TOLERANCE = 1e-3
Runner = Callable[..., Any]


def container_command(image: str, cases: Path, out: Path) -> list[str]:
    """The ``docker run`` line of the container contract."""
    return [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "-v",
        f"{cases.resolve()}:/cases:ro",
        "-v",
        f"{out.resolve()}:/out",
        image,
        "structbench-submit",
        "predict",
        "--cases",
        "/cases",
        "--out",
        "/out",
    ]


def run_container(
    image: str,
    cases: Path,
    out: Path,
    *,
    budget_seconds: float,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Run the container within the budget; the record of the run.

    ``runner`` is ``subprocess.run`` or a stand-in with its signature (tests
    inject one). A run past the budget is recorded as ``timed_out`` with
    return code ``None``.
    """
    out.mkdir(parents=True, exist_ok=True)
    command = container_command(image, cases, out)
    started = time.perf_counter()
    try:
        proc = runner(
            command, capture_output=True, text=True, timeout=budget_seconds, check=False
        )
        returncode: int | None = int(proc.returncode)
        tail = (proc.stdout or "")[-2000:] + (proc.stderr or "")[-2000:]
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        returncode, timed_out = None, True
        tail = str(exc)[-2000:]
    return {
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_s": time.perf_counter() - started,
        "budget_s": budget_seconds,
        "output_tail": tail,
    }


def _fields(case: Any) -> dict[str, np.ndarray]:
    r = case.response
    if r is None:
        return {}
    out = {"time": np.asarray(r.time, dtype=float)}
    out.update({f"node/{k}": np.asarray(v, dtype=float) for k, v in r.node.items()})
    for block, fields in r.element.items():
        out.update(
            {
                f"element/{block}/{k}": np.asarray(v, dtype=float)
                for k, v in fields.items()
            }
        )
    out.update(
        {f"global/{k}": np.asarray(v, dtype=float) for k, v in r.globals_.items()}
    )
    return out


def compare_predictions(
    submitted: Path, produced: Path, *, tolerance: float = TOLERANCE
) -> dict[str, Any]:
    """Pooled relative L2 of each produced field against the submitted one.

    One row per submitted prediction file: per field the ratio
    ``‖produced − submitted‖₂ / max(‖submitted‖₂, 1e-12)``, ``"missing"``
    when the container wrote no file or no such field, ``"shape"`` when the
    shapes differ. ``verified`` is true when every ratio is at most
    ``tolerance`` and nothing is missing.
    """
    rows: dict[str, dict[str, Any]] = {}
    verified = True
    for path in sorted(submitted.glob("*.h5")):
        other = produced / path.name
        if not other.is_file():
            rows[path.stem] = {"status": "missing"}
            verified = False
            continue
        a, b = _fields(read_case(path)), _fields(read_case(other))
        row: dict[str, Any] = {}
        for key, ref in a.items():
            got = b.get(key)
            if got is None:
                row[key] = "missing"
                verified = False
            elif got.shape != ref.shape:
                row[key] = "shape"
                verified = False
            else:
                ratio = float(
                    np.linalg.norm((got - ref).ravel())
                    / max(np.linalg.norm(ref.ravel()), 1e-12)
                )
                row[key] = ratio
                if ratio > tolerance:
                    verified = False
        rows[path.stem] = row
    if not rows:
        verified = False
    return {"tolerance": tolerance, "cases": rows, "verified": verified}


def compare_metrics(
    submitted: dict[str, Any],
    reproduced: dict[str, Any],
    *,
    tolerance: float = TOLERANCE,
) -> dict[str, Any]:
    """Relative difference of the two score files' means, key by key."""
    a, b = submitted.get("mean", {}), reproduced.get("mean", {})
    rows: dict[str, Any] = {}
    verified = True
    for key in sorted(set(a) & set(b)):
        diff = abs(float(a[key]) - float(b[key])) / max(abs(float(a[key])), 1e-12)
        rows[key] = diff
        if diff > tolerance:
            verified = False
    for key in sorted(set(a) ^ set(b)):
        rows[key] = "missing"
        verified = False
    if not rows:
        verified = False
    return {"tolerance": tolerance, "metrics": rows, "verified": verified}


def main(argv: list[str] | None = None, *, runner: Runner = subprocess.run) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--image", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=None)
    parser.add_argument("--budget-seconds", type=float, default=3600.0)
    parser.add_argument("--submitted-scores", type=Path, default=None)
    parser.add_argument("--reproduced-scores", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    work = args.work or args.out.parent / "reproduced"
    if args.dry_run:
        print(" ".join(container_command(args.image, args.cases, work)))
        return 0
    run = run_container(
        args.image, args.cases, work, budget_seconds=args.budget_seconds, runner=runner
    )
    predictions = compare_predictions(args.submission, work)
    report: dict[str, Any] = {
        "image": args.image,
        "run": run,
        "predictions": predictions,
        "verified": predictions["verified"] and run["returncode"] == 0,
    }
    if args.submitted_scores and args.reproduced_scores:
        metrics = compare_metrics(
            json.loads(args.submitted_scores.read_text(encoding="utf-8")),
            json.loads(args.reproduced_scores.read_text(encoding="utf-8")),
        )
        report["metrics"] = metrics
        report["verified"] = report["verified"] and metrics["verified"]
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
    print(f"verified={report['verified']} -> {args.out}")
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    sys.exit(main())
