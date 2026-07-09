# Running CGN baselines on DUG McCloud

Taylor 2D is the worked example below; the **Wave-1D** deltas (data, config,
batch script) are in the [last section](#wave-1d-baseline-v02). DUG uses **SLURM**. Access is SSH (with a JupyterLab-over-localhost option).
GPU nodes offer 4×V100 or 2×A100 80 GB. The training loop is **single-GPU**, so
we request **one** A100 80 GB — the full `configs/taylor_impact_2d/cgn.toml` (batch 8,
~50–60 GB) fits on the 80 GB card (batch 16 OOMs). The extra GPUs would sit idle
unless the model is extended with DistributedDataParallel (a separate change).

`train_taylor.slurm` carries live values verified 2026-07-03: partition
`curtin_eecms` (one 2× A100-80GB node, `TIMELIMIT infinite`), `gpu:a100:1`,
data at `/data/curtin_eecms/curtin_qilin/data/taylor_impact`. Two operational
facts: **`sbatch` only works from a login node** (the JupyterHub container has
no munge socket), and the partition's A100 node is shared with JupyterHub
sessions — a submitted GPU job PENDs until any session holding the GPUs ends.

## 1. Copy code + data up (from your Windows machine, Git Bash)

```bash
REPO="<path-to-local>/StructBench"
DATA="<path-to-local>/data/StructBench/canonical/taylor_impact_2d"

rsync -avP "$REPO/" <user>@<dug-host>:<proj>/structbench/ \
  --exclude '.venv' --exclude 'runs' --exclude '.git' \
  --exclude '__pycache__' --exclude 'scratch'
rsync -avP "$DATA/" <user>@<dug-host>:<proj>/data/taylor_impact/    # 2.4 GB, 34 files (the path train_taylor.slurm reads)
```

For the data, `rclone` on the DUG side is the alternative that avoids hydrating
the OneDrive files locally first — with a OneDrive remote configured there:

```bash
rclone copy onedrive:"<path-to>/data/StructBench/canonical/taylor_impact_2d" \
  <proj>/data/taylor_impact --progress --transfers=8
```

## 2. Build the env (once, on the DUG login node)

```bash
cd <proj>/structbench
bash hpc/dug/setup_env.sh
```

Only `torch` + `torch_geometric.nn.MessagePassing` are needed — no compiled
`torch-cluster`/`pyg-lib` — so this is a clean install (native `radius_graph`,
ADR-0020).

## 3. Smoke-check on a GPU node first (~5 min), then the full run

```bash
# quick sanity that CUDA + the pipeline work on DUG's GPU, using the tiny config:
srun --partition=curtin_eecms --gres=gpu:a100:1 --time=00:15:00 --pty bash -lc '
  source .venv/bin/activate && export PYTHONPATH=src
  python -m structbench.cli.train --mode train --config configs/taylor_impact_2d/cgn_smoke.toml \
    --data-root /data/curtin_eecms/curtin_qilin/data/taylor_impact --out runs/smoke'

# full baseline as a batch job (from a login node; OUT defaults to
# runs/taylor-cgn-v01 and must be fresh per attempt):
sbatch hpc/dug/train_taylor.slurm
squeue --me                     # watch the queue
tail -f logs/slurm-taylor-*.out # progress: val_pos (mm) / val_aux (MPa) each val_every
```

The full run is 80k steps at batch 8 (ADR-0028 reference recipe, radius 1.5).
Throughput was **~4.6k steps/h → ~22 h** when last measured (2026-07-03), but
that was at the earlier batch-32/100k recipe — re-measure for batch 8. Either
way the run stays inside the 36 h ceiling in the script (the partition's time
limit is `infinite`). The
best checkpoint is written whenever the validation **position** RMSE improves
(ADR-0028; checked every 2000 steps), so an early stop still leaves a usable
model — but **training has no resume**, so a killed run restarts from scratch.

Two rules the entry point now enforces / expects:

- **Fresh `--out` per training attempt.** `--mode train` refuses a directory
  that already holds `model-*.pt` (eval picks the newest checkpoint by mtime,
  so a from-scratch rerun would shadow the better earlier model).
- **If train is killed by walltime, the eval steps never ran** (`set -e`).
  Rerun just the two eval lines from the slurm script against the same
  `--out` — they rebuild the model from the run's own `config.json`.

## 4. Bring results back

```bash
rsync -avP <user>@<dug-host>:<proj>/structbench/runs/taylor-cgn-v01/ ./runs/taylor-cgn-v01/
# -> model-best-*.pt, config.json, normalization_stats.npz,
#    metrics-val.json / metrics-test_interp.json / metrics-test_extrap.json
#    (ADR-0019 per-case + split-mean metrics), and rollouts/*.npz (predicted
#    trajectories). The slurm log carries the split means and per-case RMSEs;
#    per-case QoI values/errors are only in the metrics JSON.
```

Use the JupyterLab-over-localhost option for interactive inspection — load a
checkpoint and plot rollouts — rather than for the long training run itself.

---

## Wave-1D baseline (v0.2)

Same machine, env build (§2), and result bring-back (§4) as Taylor — only the
data, config, and batch script differ. Wave is a much smaller model (hidden 64,
5 MP hops, ~500–1250 particles/case vs Taylor's 128/10), so it runs with wide
headroom; `train_wave.slurm` keeps the A100 request for a guaranteed fit, but
you can drop to `--gres=gpu:v100:1` and trim `--mem`/`--time` after the smoke
test measures the real footprint.

**1. Copy the data up** — rclone on the DUG login node, straight from a OneDrive
remote (0.23 GB, 16 cases; avoids hydrating the cloud files locally). Adjust the
`onedrive:` remote name and source path to your rclone config:

```bash
rclone copy \
  onedrive:"research/civil_engineering/data/StructBench/canonical/wave_propagation_1d" \
  /data/curtin_eecms/curtin_qilin/data/wave_propagation_1d \
  --progress --transfers=8
# sanity: the 16 .h5 cases + generated README.md / card.json
rclone ls /data/curtin_eecms/curtin_qilin/data/wave_propagation_1d | wc -l
```

The dest uses the full archive name `wave_propagation_1d`, not the truncated
form Taylor's dir (`data/taylor_impact`) still carries.

**2. Smoke-check on a GPU node (~2 min), then submit the full run:**

```bash
srun --partition=curtin_eecms --gres=gpu:a100:1 --time=00:10:00 --pty bash -lc '
  source .venv/bin/activate && export PYTHONPATH=src
  python -m structbench.cli.train --mode train --config configs/wave_propagation_1d/cgn_smoke.toml \
    --data-root /data/curtin_eecms/curtin_qilin/data/wave_propagation_1d --out runs/wave-smoke'

sbatch hpc/dug/train_wave.slurm   # OUT defaults to runs/wave-cgn-v02
squeue --me
tail -f logs/slurm-wave-*.out     # val_pos (mm) / val_aux (MPa) each val_every
```

The full run is 50k steps at batch 32 (half the Taylor budget; ADR-0028 recipe
at wave capacity). Eval covers `val` and `test_interp` only — wave has no
extrapolation split (ADR-0025) — landing as `metrics-val.json` /
`metrics-test_interp.json` + `rollouts/*.npz`. Bring the run back as in §4,
swapping the run dir for `runs/wave-cgn-v02`.
