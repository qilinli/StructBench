"""Score a submission's predictions against withheld or public reference cases
(ADR-0074 clause 4; open-evaluation plan §5 and §9).

Usage:
    python tools/score_submission.py --benchmark taylor_impact_2d \
        --submission <dir> --reference <dir> --out scores.json \
        [--ledger ledger.json] [--blind-tag blind-v1]

The submission directory holds ``submission.toml`` (the provenance ADR-0074
clause 4 lists) and one canonical-format HDF5 per predicted case, named
``<case_id>.h5``. The reference directory holds the full reference cases
(the private withheld tree for a blind set, or a public archive for the
public sets); it never leaves the steward's machine.

Prediction format (plan §5, made exact here). A prediction file is a case
file whose ``response`` covers the *scored span of the working trajectory*:
frames ``[input_frames, T')`` of the case as ``load_case_trajectory`` reads
it (``T'`` drops the terminal solver-output frame of an SPH case, ADR-0028),
with ``response/time/t`` equal to those times. It carries
``response/node/displacement`` for every node, in metres, with the published
case's node order, and the benchmark's aux field as a scalar per particle in
SI — ``response/element/sph/<aux_field>`` ``(n, P)`` for an SPH benchmark,
``response/node/<aux_field>`` ``(n, N, 1)`` for a mesh benchmark. Nothing
else is read. Metrics follow the registry's: pooled relative L2 of the
displacement (frame-0 reference) and of the aux field, mean per-frame RMSEs,
the quantities of interest, each over the benchmark's scored span with its
kinematic particles masked (ADR-0035, ADR-0039, ADR-0055), and the one
property of ADR-0075's block a prediction file supports, the error growth
(``prop_error_growth``; closure needs a one-step sweep and the constraint
properties an element-site submission, which no benchmark has yet).

The regeneration guard (clause 5) flags a set whose mean pooled relative L2
on a field lies below the floor the card measured for it
(``card.ground_truth.floors``); the rate limit (plan §5: three per method
per blind tag per thirty days) reads and extends the ledger when one is
given. Flags never change the numbers; they are returned beside them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from structbench.benchmarks import BenchmarkSpec, get_benchmark
from structbench.core import read_case
from structbench.datasets import CaseTrajectory, load_case_trajectory
from structbench.eval import (
    QoiInputs,
    coverage_bin,
    error_growth,
    field_rmse,
    position_rmse,
    relative_l2_pooled,
)

REQUIRED_KEYS = (
    "method",
    "family",
    "scheme",
    "division",
    "submitter",
    "commit",
    "config_sha256",
    "seeds",
    "checkpoint_sha256",
    "writeup",
)
RATE_LIMIT = 3
RATE_WINDOW_DAYS = 30


def read_submission(path: Path) -> dict[str, Any]:
    """``submission.toml`` with every required key present, or ``ValueError``."""
    try:
        meta = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"submission.toml: {exc}") from exc
    missing = [k for k in REQUIRED_KEYS if k not in meta]
    if missing:
        raise ValueError(f"submission.toml lacks {missing}")
    if meta["division"] not in ("closed", "open"):
        raise ValueError("submission.toml: division must be closed or open")
    if meta["division"] == "open" and not str(meta.get("data", "")).strip():
        raise ValueError(
            "submission.toml: an open-division submission declares its data"
        )
    return meta


def load_prediction(
    path: Path, spec: BenchmarkSpec, reference: CaseTrajectory, input_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """The predicted ``(positions, aux)`` over the scored span, working frame.

    Raises ``ValueError`` when the file does not match the reference case's
    span, times or shapes.
    """
    case = read_case(path)
    if case.metadata.case_id != reference.case_id:
        raise ValueError(
            f"{path.name}: case id {case.metadata.case_id!r} is not "
            f"{reference.case_id!r}"
        )
    r = case.response
    if r is None:
        raise ValueError(f"{path.name}: no response")
    t_ref = reference.time[input_frames:]
    t = np.asarray(r.time, dtype=float)
    if t.shape != t_ref.shape or not np.allclose(t, t_ref, rtol=1e-6, atol=1e-12):
        raise ValueError(
            f"{path.name}: time axis must equal the reference's frames "
            f"[{input_frames}, {reference.time.shape[0]}) ({t_ref.shape[0]} frames)"
        )
    scale = 1e-6 if spec.card.aux_unit.strip() == "MPa" else 1.0
    dim = case.metadata.dimension
    if "sph" in case.elements:
        idx = case.elements["sph"].connectivity[:, 0]
        coords0 = case.nodes.coords[idx][:, :dim]
        disp = np.asarray(r.node["displacement"])[:, idx, :]
        block = r.element.get("sph", {})
        if spec.aux_field not in block:
            raise ValueError(
                f"{path.name}: response/element/sph/{spec.aux_field} is missing"
            )
        aux = np.asarray(block[spec.aux_field], dtype=float) * scale
    else:
        coords0 = case.nodes.coords[:, :dim]
        disp = np.asarray(r.node["displacement"])
        if spec.aux_field not in r.node:
            raise ValueError(f"{path.name}: response/node/{spec.aux_field} is missing")
        aux = np.asarray(r.node[spec.aux_field], dtype=float) * scale
        if aux.ndim == 3:
            aux = aux[..., 0]
    positions = (coords0[None] + disp) * 1e3
    want_pos = reference.positions[input_frames:].shape
    want_aux = reference.aux[input_frames:, :, 0].shape
    if positions.shape != want_pos or aux.shape != want_aux:
        raise ValueError(
            f"{path.name}: shapes {positions.shape}/{aux.shape} do not match the "
            f"reference's {want_pos}/{want_aux}"
        )
    return positions.astype(np.float32), aux.astype(np.float32)


def score_case(
    spec: BenchmarkSpec,
    reference: CaseTrajectory,
    pred_pos: np.ndarray,
    pred_aux: np.ndarray,
    input_frames: int,
) -> dict[str, float | None]:
    """The registry's metrics for one case (mirrors the rollout's scoring),
    with ``prop_error_growth`` (ADR-0075 clause 6), ``None`` where undefined."""
    T = reference.time.shape[0]
    keep = ~np.isin(reference.particle_type, np.asarray(spec.kinematic_types))
    bound = spec.scored_frames_for(reference.case_id)
    scored_end = T if bound is None else min(bound, T)
    n_scored = scored_end - input_frames
    full_pos = np.concatenate([reference.positions[:input_frames], pred_pos], axis=0)
    full_aux = np.concatenate([reference.aux[:input_frames, :, 0], pred_aux], axis=0)
    true_pos = reference.positions
    true_aux = reference.aux[:, :, 0]
    u_pred = (full_pos - full_pos[0])[input_frames:][:n_scored]
    u_true = (true_pos - true_pos[0])[input_frames:][:n_scored]
    pos_rmse = position_rmse(
        full_pos[input_frames:][:n_scored],
        true_pos[input_frames:][:n_scored],
        keep=keep,
    )
    growth = error_growth(pos_rmse)
    out: dict[str, float | None] = {
        "prop_error_growth": growth if np.isfinite(growth) else None,
        "rollout_rel_l2_disp": float(relative_l2_pooled(u_pred, u_true, mask=keep)),
        "rollout_rel_l2_aux": float(
            relative_l2_pooled(
                full_aux[input_frames:][:n_scored],
                true_aux[input_frames:][:n_scored],
                mask=keep,
            )
        ),
        "rollout_pos_rmse_mm": float(pos_rmse.mean()),
        "rollout_aux_rmse": float(
            field_rmse(
                full_aux[input_frames:][:n_scored],
                true_aux[input_frames:][:n_scored],
                keep=keep,
            ).mean()
        ),
    }
    for name, fn in spec.qois.items():
        pred = fn(
            QoiInputs(
                time=reference.time,
                positions=full_pos,
                aux=full_aux,
                particle_type=reference.particle_type,
                init=input_frames,
            )
        )
        true = fn(
            QoiInputs(
                time=reference.time,
                positions=true_pos,
                aux=true_aux,
                particle_type=reference.particle_type,
                init=input_frames,
            )
        )
        out[f"qoi_{name}"] = float(pred)
        out[f"qoi_{name}_true"] = float(true)
        out[f"qoi_{name}_error"] = float(pred - true)
    return out


def guard_flags(spec: BenchmarkSpec, mean: dict[str, float]) -> list[dict[str, Any]]:
    """The regeneration guard: a set scoring below a measured floor."""
    gt = spec.card.ground_truth
    if gt is None or not gt.floors:
        return []
    metric_of = {
        "displacement": "rollout_rel_l2_disp",
        spec.aux_field: "rollout_rel_l2_aux",
    }
    flags = []
    for field_name, floor in gt.floors.items():
        key = metric_of.get(field_name)
        if key is None or key not in mean:
            continue
        if mean[key] < floor:
            flags.append(
                {
                    "type": "below_noise_floor",
                    "field": field_name,
                    "metric": key,
                    "value": mean[key],
                    "floor": floor,
                    "says": "closer to the reference than its measured noise floor "
                    "(ADR-0074 clause 5); a container may be asked for",
                }
            )
    return flags


def rate_limit_flag(
    ledger_path: Path | None, meta: dict[str, Any], blind_tag: str, now: dt.datetime
) -> list[dict[str, Any]]:
    """Read the ledger, flag the fourth submission in the window, append."""
    if ledger_path is None:
        return []
    entries: list[dict[str, Any]] = []
    if ledger_path.is_file():
        entries = json.loads(ledger_path.read_text(encoding="utf-8"))
    window = now - dt.timedelta(days=RATE_WINDOW_DAYS)
    recent = [
        e
        for e in entries
        if e.get("submitter") == meta["submitter"]
        and e.get("method") == meta["method"]
        and e.get("blind_tag", "") == blind_tag
        and dt.datetime.fromisoformat(e["scored_at"]) >= window
    ]
    flags = []
    if len(recent) >= RATE_LIMIT:
        flags.append(
            {
                "type": "rate_limited",
                "recent": len(recent),
                "limit": RATE_LIMIT,
                "window_days": RATE_WINDOW_DAYS,
                "says": "held until the window passes (plan of 2026-11-08, §5)",
            }
        )
    entries.append(
        {
            "submitter": meta["submitter"],
            "method": meta["method"],
            "blind_tag": blind_tag,
            "scored_at": now.isoformat(),
        }
    )
    ledger_path.write_text(
        json.dumps(entries, indent=1), encoding="utf-8", newline="\n"
    )
    return flags


def score(
    spec: BenchmarkSpec,
    submission: Path,
    reference: Path,
    *,
    blind_tag: str = "",
    ledger: Path | None = None,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Score every prediction in ``submission`` against ``reference``."""
    now = now or dt.datetime.now(dt.UTC)
    meta = read_submission(submission / "submission.toml")
    input_frames = spec.card.input_frames
    set_of = {c: s for s, cs in spec.splits.items() for c in cs}
    try:
        cov = spec.coverage()
    except ValueError:
        cov = {}
    cases: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for path in sorted(submission.glob("*.h5")):
        cid = path.stem
        if cid not in set_of or set_of[cid] in ("train", "val"):
            problems.append(f"{path.name}: not a scored case of {spec.card.name}")
            continue
        ref_path = reference / f"{cid}.h5"
        if not ref_path.is_file():
            problems.append(f"{path.name}: no reference case at {ref_path}")
            continue
        ref = load_case_trajectory(ref_path, aux_field=spec.aux_field)
        try:
            pos, aux = load_prediction(path, spec, ref, input_frames)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        row: dict[str, Any] = {"set": set_of[cid]}
        row["coverage"] = coverage_bin(cov[cid]) if cid in cov else None
        row.update(score_case(spec, ref, pos, aux, input_frames))
        cases[cid] = row
    metric_keys = sorted(
        {k for r in cases.values() for k in r if isinstance(r[k], float)}
    )
    # A null (an undefined property) is left out of its mean, as evaluate does.
    mean = {
        k: float(np.mean([r[k] for r in cases.values() if r.get(k) is not None]))
        for k in metric_keys
    }
    by_set: dict[str, dict[str, dict[str, float]]] = {}
    for row in cases.values():
        label = row["coverage"] or "unbinned"
        bucket = by_set.setdefault(row["set"], {}).setdefault(
            label, {"n": 0.0, "sum": 0.0}
        )
        bucket["n"] += 1
        bucket["sum"] += row["rollout_rel_l2_disp"]
    by_set_out = {
        s: {
            label: {"n": int(b["n"]), "rollout_rel_l2_disp": b["sum"] / b["n"]}
            for label, b in bins.items()
        }
        for s, bins in by_set.items()
    }
    flags = guard_flags(spec, mean) if cases else []
    flags += rate_limit_flag(ledger, meta, blind_tag, now)
    return {
        "benchmark": spec.card.name,
        "blind_tag": blind_tag,
        "submission": meta,
        "scored_at": now.isoformat(),
        "n_cases": len(cases),
        "cases": cases,
        "mean": mean,
        "by_set": by_set_out,
        "flags": flags,
        "problems": problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blind-tag", default="")
    parser.add_argument("--ledger", type=Path, default=None)
    args = parser.parse_args()
    spec = get_benchmark(args.benchmark)
    try:
        report = score(
            spec,
            args.submission,
            args.reference,
            blind_tag=args.blind_tag,
            ledger=args.ledger,
        )
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
    print(
        f"scored {report['n_cases']} cases of {spec.card.name}; "
        f"{len(report['flags'])} flags, {len(report['problems'])} problems "
        f"-> {args.out}"
    )
    for key in ("rollout_rel_l2_disp", "rollout_rel_l2_aux"):
        if key in report["mean"]:
            print(f"  mean {key} {report['mean'][key]:.5f}")
    for flag in report["flags"]:
        print(f"  flag: {flag['type']}")
    for problem in report["problems"]:
        print(f"  problem: {problem}")
    return 0 if not report["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
