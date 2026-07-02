# Running the Taylor 2D GNS baseline on DUG McCloud

DUG uses **SLURM**. Access is SSH (with a JupyterLab-over-localhost option).
GPU nodes offer 4×V100 or 2×A100 80 GB. The training loop is **single-GPU**, so
we request **one** A100 80 GB — the full `configs/taylor_2d.toml` (batch 32,
~14 GB) fits with room to spare. The extra GPUs would sit idle unless the model
is extended with DistributedDataParallel (a separate change).

Placeholders to fill in `train_taylor.slurm`: `<GPU_PARTITION>`, `<GPU_GRES>`
(e.g. `gpu:a100:1` — request the typed A100 unless the partition is A100-only),
`<PATH_TO_h5_canonical>`, and `<PROJECT>` (only if your DUG project requires an
account). Find the partition + GPU gres string with `sinfo -o "%P %G"`.

## 1. Copy code + data up (from your Windows machine, Git Bash)

```bash
REPO="C:/Users/kylin/OneDrive - Curtin/research/civil_engineering/StructBench"
DATA="C:/Users/kylin/OneDrive - Curtin/research/civil_engineering/data/2D-Copper-Bar-Taylor-Impact/h5_canonical"

rsync -avP "$REPO/" <user>@<dug-host>:<proj>/structbench/ \
  --exclude '.venv' --exclude 'runs' --exclude '.git' \
  --exclude '__pycache__' --exclude 'scratch'
rsync -avP "$DATA/" <user>@<dug-host>:<proj>/data/h5_canonical/     # 2.4 GB, 34 files
```

For the data, `rclone` on the DUG side is the alternative that avoids hydrating
the OneDrive files locally first — with a OneDrive remote configured there:

```bash
rclone copy onedrive:"research/civil_engineering/data/2D-Copper-Bar-Taylor-Impact/h5_canonical" \
  <proj>/data/h5_canonical --progress --transfers=8
```

## 2. Build the env (once, on the DUG login node)

```bash
cd <proj>/structbench
bash deploy/dug/setup_env.sh
```

Only `torch` + `torch_geometric.nn.MessagePassing` are needed — no compiled
`torch-cluster`/`pyg-lib` — so this is a clean install (native `radius_graph`,
ADR-0020).

## 3. Smoke-check on a GPU node first (~5 min), then the full run

```bash
# quick sanity that CUDA + the pipeline work on DUG's GPU, using the tiny config:
srun --partition=<GPU_PARTITION> --gres=gpu:1 --time=00:15:00 --pty bash -lc '
  source .venv/bin/activate && export PYTHONPATH=src
  python -m structbench.cli.train --mode train --config configs/taylor_2d_smoke.toml \
    --data-root <proj>/data/h5_canonical --out runs/smoke'

# full baseline as a batch job:
mkdir -p runs/taylor-baseline
sbatch deploy/dug/train_taylor.slurm
squeue --me                     # watch the queue
tail -f slurm-taylor-*.out      # watch progress; logs train_loss / val_loss each val_every
```

The full run is 100k steps at batch 32. Runtime is uncertain until measured
(the per-example graph build is O(N²) over 4800 particles); expect on the order
of hours-to-half-a-day on one A100. The best checkpoint is written whenever
validation improves (checked every 2000 steps, so the first save lands at step
2000), meaning a timeout or early stop still leaves a usable model. To get a
real ETA before committing the walltime, watch the first few `val_every`
intervals in the smoke/full log and extrapolate — if 100k steps clearly won't
fit in 24 h, `scancel` early: **training has no resume**, so the walltime is a
hard compute budget.

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
