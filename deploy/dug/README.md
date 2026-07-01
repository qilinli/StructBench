# Running the Taylor 2D GNS baseline on DUG McCloud

DUG uses **SLURM**. Access is SSH (with a JupyterLab-over-localhost option).
GPU nodes offer 4×V100 or 2×A100 80 GB. The training loop is **single-GPU**, so
we request **one** A100 80 GB — the full `configs/taylor_2d.toml` (batch 32,
~14 GB) fits with room to spare. The extra GPUs would sit idle unless the model
is extended with DistributedDataParallel (a separate change).

Placeholders to fill in `train_taylor.slurm`: `<GPU_PARTITION>`,
`<PATH_TO_h5_canonical>`, and `<PROJECT>` (only if your DUG project requires an
account). Find the partition + GPU gres string with `sinfo -o "%P %G"`.

## 1. Copy code + data up (from your Windows machine, Git Bash)

```bash
REPO="C:/Users/kylin/OneDrive - Curtin/research/civil_engineering/StructBench"
DATA="C:/Users/kylin/OneDrive - Curtin/research/civil_engineering/data/2D-Copper-Bar-Taylor-Impact/h5_canonical"

rsync -avP "$REPO/" <user>@<dug-host>:<proj>/structbench/ --exclude '.venv' --exclude 'runs'
rsync -avP "$DATA/" <user>@<dug-host>:<proj>/data/h5_canonical/     # 2.4 GB, 34 files
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
of hours-to-half-a-day on one A100. The best checkpoint is written every 2000
steps, so a timeout or early stop still leaves a usable model. To get a real ETA
before committing the walltime, watch the first few `val_every` intervals in the
smoke/full log and extrapolate.

## 4. Bring results back

```bash
rsync -avP <user>@<dug-host>:<proj>/structbench/runs/taylor-baseline/ ./runs/taylor-baseline/
# -> model-best-*.pt, config.json, normalization_stats.npz; eval RMSEs are in the slurm log
```

Use the JupyterLab-over-localhost option for interactive inspection — load a
checkpoint and plot rollouts — rather than for the long training run itself.
