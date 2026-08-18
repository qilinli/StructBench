#!/usr/bin/env python
"""Fast idea-screening for the transolver family: reduced-budget train + verdict.

The trial loop for a new slice/temperature idea:
  1. add the knob (tools/add_transolver_knob.py) + thread it in code
  2. `screen.py gen`  -> a reduced-budget screening config (few k steps)
  3. sbatch it (train_arm.slurm does train+valid+rollout)
  4. `screen.py verdict` -> per-run val + slice KL-from-uniform, idea vs baseline

Screening is TRIAGE, not truth: a reduced budget gives a directional read (does
val track/beat baseline, does slice-KL rise) before a full fleet. Screen on the
benchmark with headroom (notch for phi ideas), not Taylor (SAROS Cause-3).

Examples:
  python tools/screen.py gen --bench notch_beam_2d_impact \
      --base transolver-timecond-iv-s1 --name screen-phitau \
      --set temperature_phi=true --steps 25000 --val-every 2000
  python tools/screen.py verdict --bench notch_beam_2d_impact \
      --runs runs/screen-phitau runs/screen-baseline
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = {
    "taylor_impact_2d": "taylor_impact",
    "notch_beam_2d_impact": "notch_beam_2d_impact",
    "deforming_plate": "deforming_plate",
}


def cmd_gen(a: argparse.Namespace) -> int:
    """Write a reduced-budget screening config = base + overrides."""
    base = REPO / "configs" / a.bench / f"{a.base}.toml"
    sets = dict(kv.split("=", 1) for kv in a.set or [])
    overrides = {**sets, "training_steps": str(a.steps), "val_every": str(a.val_every)}
    applied: set[str] = set()
    out: list[str] = []
    for line in base.read_text().splitlines():
        key = line.split("=", 1)[0].strip() if "=" in line else None
        if key in overrides:
            tag = "screen override" if key in sets else "screen budget"
            out.append(f"{key} = {overrides[key]}  # {tag}")
            applied.add(key)
        else:
            out.append(line)
    missing = set(overrides) - applied
    if missing:
        print(
            f"ERROR: keys absent from {a.base}: {sorted(missing)} "
            "(add the field to the schema + this base TOML first)"
        )
        return 1
    dest = REPO / "configs" / a.bench / f"{a.name}.toml"
    dest.write_text("\n".join(out) + "\n")
    data = DATA_DIR[a.bench]
    print(f"wrote {dest.relative_to(REPO)}  (steps={a.steps}, overrides={sets})")
    print("launch:")
    print(
        f"  sbatch --job-name={a.name} "
        f"--export=ALL,BENCH={a.bench},ARM={a.name},"
        f"DATA=/data/curtin_eecms/curtin_qilin/data/{data},"
        f"OUT=$PWD/runs/{a.name} hpc/dug/train_arm.slurm"
    )
    return 0


def _best_ckpt(run: Path) -> Path | None:
    cks = sorted(
        glob.glob(str(run / "model-best-*.pt")),
        key=lambda p: int(p.split("-")[-1].split(".")[0]),
    )
    return Path(cks[-1]) if cks else None


def _kl(bench: str, run: Path) -> float | None:
    """Mean normalized slice KL-from-uniform on the run's best checkpoint."""
    import torch

    from structbench.benchmarks import get_benchmark
    from structbench.cli.train import (
        _model_config_from_record,
        _tc_time_ref_frames,
        build_transolver_simulator,
    )
    from structbench.config import read_run_record
    from structbench.datasets import load_case_trajectory
    from structbench.models.transolver.network import (
        PhysicsAttentionIrregularMesh as PA,
    )

    ck = _best_ckpt(run)
    if ck is None or not (run / "config.json").exists():
        return None
    rec = read_run_record(run / "config.json")
    cfg = _model_config_from_record(rec)
    spec = get_benchmark(bench)
    sim = build_transolver_simulator(
        cfg,
        kinematic_types=spec.kinematic_types,
        scripted_types=spec.scripted_types,
        device="cpu",
    )
    sim.load(str(ck))
    sim.eval()
    log_m = math.log(cfg.slice_num)
    tot = [0.0, 0.0]
    orig = PA._slice_weights

    def hook(self, x, phi=None):  # noqa: ANN001
        w = orig(self, x, phi)
        with torch.no_grad():
            kl = log_m - (-(w * torch.log(w.clamp_min(1e-12))).sum(-1))
            tot[0] += float(kl.sum())
            tot[1] += float(kl.numel())
        return w

    PA._slice_weights = hook
    try:
        dr = Path(rec.get("data_root") or rec["train"]["data_root"])
        for cid in list(spec.splits["val"])[:2]:
            tr = load_case_trajectory(dr / f"{cid}.h5", aux_field=spec.aux_field)
            if spec.mesh_transform:
                tr = spec.mesh_transform(tr)
            has_iv = getattr(cfg, "impact_velocity_feature", False)
            load = spec.loading_scalar(cid) if has_iv and spec.loading_scalar else None
            sim.bind_case(
                torch.as_tensor(tr.cells, dtype=torch.int64),
                torch.as_tensor(tr.reference_coords, dtype=torch.float32),
                torch.as_tensor(tr.particle_type, dtype=torch.int64),
                torch.as_tensor(tr.positions, dtype=torch.float32),
                loading_scalar=load,
            )
            sim.reset_rollout()
            tref = _tc_time_ref_frames(
                spec.scored_frames, rec["train"].get("train_frames", 0), len(tr.time)
            )
            n = tr.positions.shape[0]
            hi = min(spec.scored_frames or n, n)
            lo = spec.card.input_frames
            for f in range(lo, hi, max(1, (hi - lo) // 10)):
                sim.predict_state_at(int(f), f / (tref - 1))
    finally:
        PA._slice_weights = orig
    return (tot[0] / tot[1] / log_m) if tot[1] else None


def _split_metric(run: Path, split: str, key: str) -> float | None:
    fp = run / f"metrics-{split}.json"
    if not fp.exists():
        return None
    return json.loads(fp.read_text()).get("mean", {}).get(key)


def cmd_verdict(a: argparse.Namespace) -> int:
    """Per-run displacement + AUX rel-L2 across splits (rowless = missing)."""
    from structbench.benchmarks import get_benchmark

    splits = list(get_benchmark(a.bench).eval_splits)
    cols = [f"{s[:6]}:{m}" for s in splits for m in ("disp", "aux")]
    print(f"{'run':32s} " + " ".join(f"{c:>11s}" for c in cols))
    for r in a.runs:
        run = Path(r).expanduser().resolve()
        vals = []
        for s in splits:
            for k in ("rollout_rel_l2_displacement", "rollout_rel_l2_aux"):
                v = _split_metric(run, s, k)
                vals.append("—" if v is None else f"{v:.4f}")
        print(f"{run.name:32s} " + " ".join(f"{v:>11s}" for v in vals))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen")
    g.set_defaults(fn=cmd_gen)
    g.add_argument("--bench", required=True, choices=list(DATA_DIR))
    g.add_argument("--base", required=True, help="base config stem")
    g.add_argument("--name", required=True, help="screening config/run name")
    g.add_argument(
        "--set", action="append", help="knob override key=value (repeatable)"
    )
    g.add_argument("--steps", type=int, default=25000)
    g.add_argument("--val-every", type=int, default=2000)

    v = sub.add_parser("verdict")
    v.set_defaults(fn=cmd_verdict)
    v.add_argument("--bench", required=True, choices=list(DATA_DIR))
    v.add_argument("--runs", nargs="+", required=True)

    a = p.parse_args()
    return int(a.fn(a))


if __name__ == "__main__":
    raise SystemExit(main())
