# State-sufficiency probe (Taylor 2D)

**Exploratory. Not part of the `structbench` platform** — no public API, no
registry entry, no ADR, no compatibility guarantee. It exists to answer one
question cheaply before any pipeline change is proposed.

## What it does

Trains a one-step neighbourhood operator on ground-truth pairs from the Taylor
2D SPH benchmark:

    F: z_n -> dz     teacher-forced, no rollout, no error accumulation

under two input arms:

| arm | node features | channels |
|---|---|---|
| `full` | `v, s, peeq, E, rho` | 8 |
| `kinematic` | `v` | 2 |

Both arms predict the same eight increments and get identical budget, so the
per-field errors compare directly. The measured quantity is **per-field
increment relative L2 in physical units**, on `val` + `test_interp`.

## The state

Ten scalars per particle, derived from the deck rather than chosen — the
closure of `*MAT_ELASTIC_PLASTIC_HYDRO` + `*EOS_GRUNEISEN` under SPH `FORM=12`:

| | dim | why it must be carried |
|---|---|---|
| `x` | 2 | neighbour search, kernel gradients |
| `v` | 2 | momentum |
| `s` | 3 | the deviatoric update is a *rate* law — needs the current deviator |
| `peeq` | 1 | sets `sigma_y` through the hardening table; path-dependent |
| `E` | 1 | Gruneisen needs it: `p = p(rho, E)` |
| `rho` | 1 | integrated from continuity; **not** recoverable from positions |

Mass and smoothing length are invariant on this benchmark and excluded. `s` is
three components, not four: plane strain gives `s_yz = s_zx = 0` and
`s_zz = -(s_xx + s_yy)`.

The dynamics are **not** per-particle — `z_{n+1}(i) = F(z_n over the
neighbourhood of i)` — so `model.py` is a neighbourhood operator throughout. A
per-particle MLP has no spatial gradients and would fail for reasons unrelated
to state sufficiency.

## Trunks and the vm metric

`--model mp` (default) is the message-passing neighbourhood operator below;
`--model transolver` swaps in the package's native `TransolverNet` (the
ADR-0044 Physics-Attention family) behind the same contract — arms, targets,
budget unchanged. Physics-Attention has no edges, so normalised position is
prepended to the node features (native Transolver treatment; both arms get
it, so the ablation stays fair) at the cost of the MP trunk's structural
translation invariance. `--slice-num` sets the slice tokens (default 32).

Eval also reports `vm`: relative L2 of the composed one-step von Mises,
`vm(s_n + ds_pred)` vs `vm(s_{n+1})` — stress computed from the predicted
tensor, never regressed (plan C1). Teacher forcing anchors the base state,
so absolute `vm` values run far below the increment errors; read it across
arms/trunks, not against the other rows.

## Verification

`load_state(..., check=True)` gates every case on:

- plane strain — `sigma_yz` and `sigma_zx` identically zero
- tracelessness of the extracted deviator
- von Mises round-trip — `sqrt(3/2 s:s)` from the three stored components
  against `von_mises_from_voigt` on the full six-component tensor, `< 1e-12`
  relative

Measured on `T-20-60-100` (2026-09-01): T=151 after the ADR-0028 artifact-frame
drop, P=4800, dt ~2 us uniform. Zero yield violations in 724,800 samples (max
`vm/sigma_y` = 1.000353, float32 + linear-interp round-off) and zero `peeq`
monotonicity violations in 720,000.

## Design notes

- **Increments, not absolute state.** The operator predicts `dv, ds, dpeeq,
  dE, drho`; position is integrated from velocity. The solver integrates rates.
- **Inputs normalised by physical constants** (`state.py`), not dataset
  statistics — dataset stats move when cases are added. `rho` is centred on
  8900; a raw divide puts every value near 1.0.
- **Targets normalised by per-field increment std** on the training split.
  Increments differ by orders of magnitude between fields. Errors are reported
  back in SI, so this flatters neither arm.
- **Graph**: radius 1.5 mm (3x particle spacing), 32-neighbour cap — ADR-0028
  and the 2026-07-06/07 corrections. Measured degree ~26. Edges carry relative
  displacement only, so the operator is translation invariant.
- **Data loading** reads the eight datasets it needs directly rather than via
  `read_case`, which materialises every field (~96 MB per case against ~38 MB
  used). Prepared arrays are cached as memmapped `.npy` (~29 MB per case).

## Running locally

    python probe.py --smoke      # synthetic, no data, ~10 s

    # real data on Apple silicon: ALWAYS set the caps (see Memory)
    export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.0
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.25
    python probe.py --arm full      --data-root <...>/canonical/taylor_impact_2d
    python probe.py --arm kinematic --data-root <...>/canonical/taylor_impact_2d

~5.5 steps/s and 2–3 GB at the laptop defaults (batch 1, hidden 64, 4 blocks)
on an M5 Pro. Put `--cache-dir` outside any synced folder.

## Running on DUG

See `hpc/dug/probe_state_sufficiency.slurm`, which uses the A100 configuration
(batch 2, hidden 128, 6 blocks — measured to need ~12 GB, ample on an 80 GB
card). Build the cache once before submitting a fleet; concurrent jobs race on
it.

## Memory

An early laptop run consumed 48 GB and stalled the machine. Three causes:

1. **The MPS caching allocator does not bound itself.** Edge counts vary every
   step (124k–127k), so freed blocks rarely fit the next request and the
   allocator keeps claiming new ones. On unified memory that grows against
   system RAM. `--empty-cache-every` (default 25 steps) drops the cache, and
   GB is reported at every eval so drift is visible.
2. **Activation storage scales with edges × width × depth.** At batch 2 /
   hidden 128 / 6 blocks the edge MLPs alone hold ~1.2 GB each.
3. **No hard ceiling.** The two `PYTORCH_MPS_*_WATERMARK_RATIO` variables cap
   MPS at a fraction of system RAM, so a runaway raises instead of taking the
   machine down. They must be set *before* torch is imported, hence the shell
   export; `LOW` must be below `HIGH` or torch refuses to start.

CUDA is not affected by (1) or (3) in the same way; the SLURM script sets
neither.
