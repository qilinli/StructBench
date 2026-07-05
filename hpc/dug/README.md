# Running the Taylor 2D GNS baseline on DUG McCloud

DUG uses **SLURM**. Access is SSH (with a JupyterLab-over-localhost option).
GPU nodes offer 4×V100 or 2×A100 80 GB. The training loop is **single-GPU**, so
we request **one** A100 80 GB — the full `configs/taylor_2d.toml` (batch 32,
~14 GB) fits with room to spare. The extra GPUs would sit idle unless the model
is extended with DistributedDataParallel (a separate change).

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
  python -m structbench.cli.train --mode train --config configs/taylor_2d_smoke.toml \
    --data-root /data/curtin_eecms/curtin_qilin/data/taylor_impact --out runs/smoke'

# full baseline as a batch job (from a login node; OUT defaults to
# runs/taylor-full-adr0024 and must be fresh per attempt):
sbatch hpc/dug/train_taylor.slurm
squeue --me                     # watch the queue
tail -f slurm-taylor-*.out      # progress: val_pos (mm) / val_aux (MPa) each val_every
```

The full run is 100k steps at batch 32. Measured 2026-07-03 with the ADR-0028
recipe (radius 1.5): **~4.6k steps/h on one A100 → ~22 h**, well inside the
36 h ceiling in the script (the partition's time limit is `infinite`). The
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
rsync -avP <user>@<dug-host>:<proj>/structbench/runs/taylor-baseline/ ./runs/taylor-baseline/
# -> model-best-*.pt, config.json, normalization_stats.npz,
#    metrics-val.json / metrics-test_interp.json / metrics-test_extrap.json
#    (ADR-0019 per-case + split-mean metrics), and rollouts/*.npz (predicted
#    trajectories). The slurm log carries the split means and per-case RMSEs;
#    per-case QoI values/errors are only in the metrics JSON.
```

Use the JupyterLab-over-localhost option for interactive inspection — load a
checkpoint and plot rollouts — rather than for the long training run itself.
