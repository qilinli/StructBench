"""Stage-1 state-sufficiency probe for Taylor 2D.

Question: does the one-step operator need the hidden state (``s``, ``peeq``,
``E``, ``rho``), or do kinematics alone suffice?

Trains ``F: z_n -> dz`` teacher-forced on ground-truth pairs -- no rollout, no
error accumulation, no stability work -- under two input arms:

  ``--arm full``       node features = v, s, peeq, E, rho   (8 channels)
  ``--arm kinematic``  node features = v                     (2 channels)

Both predict the same eight increments, so the per-field errors are directly
comparable. If ``kinematic`` matches ``full``, the hidden variables carry
nothing and the complete-state direction is dead. If it does not, the gap is
what they carry.

Two trunks (``--model``): ``mp`` (default) is the local message-passing
neighbourhood operator; ``transolver`` is the package's native Physics-
Attention net (the Stage-3 family), positions as input channels since it has
no edges. Same arms, targets, and budget either way.

Eval additionally reports ``vm``: relative L2 of the one-step von Mises
COMPOSED from the predicted deviator increment, ``vm(s_n + ds_pred)`` against
``vm(s_{n+1})`` -- stress computed, not regressed (plan C1). Derived, so it
costs nothing and stays consistent with the predicted tensor by construction.

Usage
-----
    python probe.py --smoke                      # synthetic, no data needed
    python probe.py --arm full      --data-root <...>/canonical/taylor_impact_2d
    python probe.py --arm kinematic --data-root <...>/canonical/taylor_impact_2d

Exploratory scratch code -- not part of the ``structbench`` package.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from model import StateOperator, edge_features
from state import (
    UNIT_SCALE,
    X_SCALE,
    CaseState,
    load_state,
    normalise,
    von_mises_from_deviatoric,
)
from torch import Tensor

from structbench.benchmarks.taylor_impact_2d.benchmark import TEST_INTERP, TRAIN, VAL
from structbench.models.cgn.graph_ops import radius_graph

# Taylor recipe: 3x particle spacing, project-wide 32-neighbour backstop cap
# (ADR-0028 + the 2026-07-06/07 corrections). Metres, since the state is SI.
RADIUS = 1.5e-3
MAX_NEIGHBORS = 32

#: Feature channels, in packed order. These are also the predicted increment
#: channels: the operator maps the state to its own one-step change. Position
#: is integrated from velocity, never predicted directly.
LAYOUT: tuple[tuple[str, int], ...] = (
    ("v", 2), ("s", 3), ("peeq", 1), ("E", 1), ("rho", 1),
)
FEAT_DIM = sum(n for _, n in LAYOUT)  # 8
_S_OFF = LAYOUT[0][1]  # deviator slice within the packed layout
S_SLICE = slice(_S_OFF, _S_OFF + LAYOUT[1][1])

#: Input channels per ablation arm. Both arms predict all eight increments.
ARM_SLICE: dict[str, slice] = {
    "full": slice(0, FEAT_DIM),
    "kinematic": slice(0, 2),
}


@dataclass
class Case:
    """One case as memory-mapped position and normalised feature arrays."""

    case_id: str
    pos: np.ndarray  # (T, P, 2) metres -- graph construction stays in SI
    feat: np.ndarray  # (T, P, 8) normalised

    @property
    def n_pairs(self) -> int:
        return int(self.pos.shape[0]) - 1


def _as_2d(a: np.ndarray) -> np.ndarray:
    return a[..., None] if a.ndim == 2 else a


def pack(st: CaseState) -> tuple[np.ndarray, np.ndarray]:
    """Normalise a state into ``(pos, feat)`` arrays in ``LAYOUT`` order."""
    norm = {k: _as_2d(v) for k, v in normalise(st).items()}
    feat = np.concatenate([norm[name] for name, _ in LAYOUT], axis=-1)
    return st.x.astype(np.float32), feat.astype(np.float32)


def load_case(
    case_id: str, data_root: Path, cache_root: Path, *, rebuild: bool = False
) -> Case:
    """Load a case from the prepared cache, building it from HDF5 if absent.

    The canonical files are ~96 MB each and live on the OneDrive-backed data
    root; reading all 30 takes tens of minutes. The prepared arrays are ~29 MB
    per case and memory-map, so the cost is paid once and random-frame
    sampling pages in only the frames it touches.
    """
    dest = cache_root / case_id
    pos_p, feat_p = dest / "pos.npy", dest / "feat.npy"
    if rebuild or not (pos_p.exists() and feat_p.exists()):
        dest.mkdir(parents=True, exist_ok=True)
        pos, feat = pack(load_state(data_root / f"{case_id}.h5", check=True))
        np.save(pos_p, pos)
        np.save(feat_p, feat)
    return Case(
        case_id,
        np.load(pos_p, mmap_mode="r"),
        np.load(feat_p, mmap_mode="r"),
    )


def load_cases(
    ids: list[str], data_root: Path, cache_root: Path, *, rebuild: bool = False
) -> list[Case]:
    cases = []
    for i, cid in enumerate(ids, 1):
        t0 = time.time()
        cases.append(load_case(cid, data_root, cache_root, rebuild=rebuild))
        print(f"  [{i:>2}/{len(ids)}] {cid}  ({time.time() - t0:.1f}s)", flush=True)
    return cases


def target_scales(cases: list[Case], stride: int = 5) -> np.ndarray:
    """Per-channel increment std over the training cases.

    Increments are far smaller than the states they update and differ by
    orders of magnitude between fields; without this one field dominates the
    gradient and the others are never learned. Reported errors are converted
    back to physical units, so this choice does not flatter any arm.
    Subsampled by ``stride`` frames -- a normaliser does not need every frame.
    """
    chunks = []
    for c in cases:
        f = np.asarray(c.feat[::stride])
        chunks.append((f[1:] - f[:-1]).reshape(-1, FEAT_DIM))
    scale = np.concatenate(chunks, axis=0).std(axis=0)
    return np.where(scale > 0, scale, 1.0).astype(np.float32)


class Batcher:
    """Samples ``(case, frame)`` pairs and assembles model-ready tensors.

    ``assemble`` always returns ``(a, b, c, tgt)`` with ``model(a, b, c)``
    the forward call: for the mp trunk ``(feat, edge_feat, edge_index)``,
    for the transolver trunk ``(feat, pos_norm, n_per_example)`` -- no graph
    is built there, Physics-Attention has no edges.
    """

    def __init__(
        self,
        cases: list[Case],
        arm: str,
        scales: np.ndarray,
        device: torch.device,
        model_kind: str = "mp",
    ) -> None:
        self.cases = cases
        self.sl = ARM_SLICE[arm]
        self.scales = torch.from_numpy(scales).to(device)
        self.device = device
        self.model_kind = model_kind
        self.node_in = self.sl.stop - self.sl.start

    def sample(self, rng: np.random.Generator, batch_size: int) -> tuple[Tensor, ...]:
        picks = []
        for _ in range(batch_size):
            case = self.cases[rng.integers(len(self.cases))]
            picks.append((case, int(rng.integers(case.n_pairs))))
        return self.assemble(picks)

    def assemble(self, picks: list[tuple[Case, int]]) -> tuple[Tensor, ...]:
        pos, feat, tgt, batch = [], [], [], []
        for k, (case, t) in enumerate(picks):
            p = np.ascontiguousarray(case.pos[t])
            f0 = np.ascontiguousarray(case.feat[t])
            f1 = np.ascontiguousarray(case.feat[t + 1])
            pos.append(torch.from_numpy(p))
            feat.append(torch.from_numpy(f0[:, self.sl]))
            tgt.append(torch.from_numpy(f1 - f0))
            batch.append(torch.full((p.shape[0],), k, dtype=torch.long))

        dev = self.device
        pos_t = torch.cat(pos).to(dev)
        feat_t = torch.cat(feat).to(dev)
        tgt_t = torch.cat(tgt).to(dev) / self.scales
        if self.model_kind == "transolver":
            npp = torch.tensor([p.shape[0] for p in pos], device=dev)
            return feat_t, pos_t / X_SCALE, npp, tgt_t
        edge_index = radius_graph(
            pos_t, RADIUS, torch.cat(batch).to(dev),
            max_num_neighbors=MAX_NEIGHBORS, loop=True,
        )
        return feat_t, edge_features(pos_t, edge_index, RADIUS), edge_index, tgt_t


def evaluate(
    model: torch.nn.Module, batcher: Batcher, cases: list[Case], stride: int = 10
) -> dict[str, float]:
    """Per-field increment relative L2 in physical units, plus derived ``vm``.

    ``vm`` scores the composed one-step von Mises -- ``vm(s_n + ds_pred)``
    against ``vm(s_n + ds_true)`` via :func:`von_mises_from_deviatoric` -- so
    stress accuracy is measured the way a complete-state simulator would
    report it: computed from the tensor, never regressed (plan C1). ``s_n``
    is the ground-truth deviator (teacher forcing), read from the case
    regardless of the input arm.

    Restricted to frames >= 7 -- the rod is in free flight before first wall
    contact, and including it inflates every score.
    """
    model.eval()
    names = [n for n, _ in LAYOUT] + ["vm"]
    num = dict.fromkeys(names, 0.0)
    den = dict.fromkeys(names, 0.0)
    s_unit = UNIT_SCALE["s"]
    with torch.no_grad():
        for case in cases:
            for t in range(7, case.n_pairs, stride):
                a, b, c, tgt = batcher.assemble([(case, t)])
                pred = model(a, b, c)
                p = (pred * batcher.scales).cpu().numpy()
                g = (tgt * batcher.scales).cpu().numpy()
                off = 0
                for name, width in LAYOUT:
                    unit = UNIT_SCALE[name]
                    pe = p[:, off : off + width] * unit
                    ge = g[:, off : off + width] * unit
                    num[name] += float(((pe - ge) ** 2).sum())
                    den[name] += float((ge**2).sum())
                    off += width
                s0 = np.ascontiguousarray(case.feat[t][:, S_SLICE]) * s_unit
                vm_p = von_mises_from_deviatoric(s0 + p[:, S_SLICE] * s_unit)
                vm_g = von_mises_from_deviatoric(s0 + g[:, S_SLICE] * s_unit)
                num["vm"] += float(((vm_p - vm_g) ** 2).sum())
                den["vm"] += float((vm_g**2).sum())
                del a, b, c, tgt, pred
    model.train()
    return {
        n: float(np.sqrt(num[n] / den[n])) if den[n] > 0 else float("nan")
        for n in names
    }


def synthetic(n_frames: int = 40, n_particles: int = 600, seed: int = 1) -> Case:
    """A plausible-shaped case for smoke-testing without touching the data."""
    rng = np.random.default_rng(seed)
    grid = np.stack(
        np.meshgrid(np.linspace(0, 0.02, 20), np.linspace(0, 0.06, n_particles // 20)),
        -1,
    ).reshape(-1, 2)
    p = grid.shape[0]
    drift = np.linspace(0, -5e-3, n_frames)[:, None, None]
    st = CaseState(
        case_id=f"SYNTH-{seed}",
        x=(grid[None] + drift).astype(np.float32),
        v=(rng.normal(0, 20, (n_frames, p, 2)) - 100).astype(np.float32),
        s=(rng.normal(0, 5e7, (n_frames, p, 3))).astype(np.float32),
        peeq=np.abs(rng.normal(0.3, 0.1, (n_frames, p))).astype(np.float32),
        E=np.abs(rng.normal(0.01, 0.003, (n_frames, p))).astype(np.float32),
        rho=(8900 + rng.normal(0, 50, (n_frames, p))).astype(np.float32),
        time=np.arange(n_frames) * 2e-6,
    )
    pos, feat = pack(st)
    return Case(st.case_id, pos, feat)


def free_cache(device: torch.device) -> None:
    """Release the allocator's cached blocks.

    Edge counts vary every step (124k-127k on Taylor), so the MPS caching
    allocator fragments and keeps claiming new blocks instead of reusing
    them. On unified memory that grows against system RAM until the machine
    stalls, so the cache is dropped periodically.
    """
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def memory_gb(device: torch.device) -> float:
    """Driver-allocated memory in GiB, or 0 where the backend cannot report."""
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**30
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**30
    return 0.0


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(args: argparse.Namespace, node_in: int) -> torch.nn.Module:
    """Construct the trunk. ``--n-steps`` is MP blocks or Transolver layers."""
    if args.model == "transolver":
        from model_transolver import TransolverOperator

        return TransolverOperator(
            node_in,
            hidden=args.hidden,
            n_layers=args.n_steps,
            slice_num=args.slice_num,
            out_dim=FEAT_DIM,
        )
    return StateOperator(
        node_in, hidden=args.hidden, n_steps=args.n_steps, out_dim=FEAT_DIM
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=sorted(ARM_SLICE), default="full")
    ap.add_argument("--model", choices=("mp", "transolver"), default="mp")
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--cache-dir", type=Path, default=Path("cache"))
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n-steps", type=int, default=4)
    ap.add_argument("--slice-num", type=int, default=32, help="transolver only")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)  # never 0 (CORRECTIONS 2026-07-10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-every", type=int, default=2_000)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--empty-cache-every", type=int, default=25)
    ap.add_argument("--smoke", action="store_true", help="synthetic data, 50 steps")
    ap.add_argument(
        "--prepare-cache",
        action="store_true",
        help="build the memmap cache for every case and exit (run once "
        "before submitting a fleet; concurrent jobs would race on it)",
    )
    args = ap.parse_args()

    if args.prepare_cache:
        if args.data_root is None:
            ap.error("--data-root is required with --prepare-cache")
        ids = TRAIN + VAL + TEST_INTERP
        print(f"building cache for {len(ids)} cases -> {args.cache_dir}", flush=True)
        load_cases(ids, args.data_root, args.cache_dir, rebuild=args.rebuild_cache)
        print("cache ready", flush=True)
        return

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)

    if args.smoke:
        train_cases = [synthetic(seed=1), synthetic(seed=2)]
        val_cases = [synthetic(seed=3)]
        args.steps, args.eval_every = 50, 25
    else:
        if args.data_root is None:
            ap.error("--data-root is required unless --smoke")
        print(f"loading {len(TRAIN)} train cases (cache: {args.cache_dir})", flush=True)
        train_cases = load_cases(
            TRAIN, args.data_root, args.cache_dir, rebuild=args.rebuild_cache
        )
        held = VAL + TEST_INTERP
        print(f"loading {len(held)} val/test cases", flush=True)
        val_cases = load_cases(
            held, args.data_root, args.cache_dir, rebuild=args.rebuild_cache
        )

    scales = target_scales(train_cases)
    batcher = Batcher(train_cases, args.arm, scales, device, args.model)
    val_batcher = Batcher(val_cases, args.arm, scales, device, args.model)
    model = build_model(args, batcher.node_in).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())

    print(
        f"model={args.model} arm={args.arm} node_in={batcher.node_in} "
        f"params={n_params:,} device={device} "
        f"train={len(train_cases)} val={len(val_cases)}",
        flush=True,
    )

    history = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        a, b, c, tgt = batcher.sample(rng, args.batch_size)
        pred = model(a, b, c)
        loss = torch.nn.functional.mse_loss(pred, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if args.empty_cache_every and step % args.empty_cache_every == 0:
            free_cache(device)

        if step % args.eval_every == 0 or step == args.steps:
            rel = evaluate(model, val_batcher, val_cases)
            free_cache(device)
            elapsed = time.time() - t0
            mem = memory_gb(device)
            history.append(
                {
                    "step": step, "loss": loss.item(), "elapsed_s": elapsed,
                    "mem_gb": mem, **rel,
                }
            )
            fields = "  ".join(f"{k}={v:.4f}" for k, v in rel.items())
            print(
                f"[{step:>6}] loss={loss.item():.5f}  {fields}  "
                f"({step / elapsed:.1f} steps/s, {mem:.1f} GB)",
                flush=True,
            )

    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / f"probe-{args.arm}-s{args.seed}.json"
    meta = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    dest.write_text(json.dumps({"args": meta, "history": history}, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
