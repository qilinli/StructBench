#!/bin/bash
# Full Wave-1D CGN baseline on DUG (SLURM). Submit from the repo root ON A
# LOGIN NODE (sbatch does not work inside the JupyterHub container — no munge):
#   sbatch hpc/dug/train_wave.slurm
#
# The wave model is small (hidden 64, 5 MP hops, ~500-1250 particles/case vs
# Taylor's 128/10), so it has wide headroom on the A100 and should finish well
# inside the ceiling below. We keep the A100 request for parity with Taylor and
# a guaranteed fit; after the smoke test measures the real footprint, try
# --gres=gpu:v100:1 and trim --mem/--time so wave stops contending for the
# shared A100 node.

#SBATCH --job-name=wave-cgn
#SBATCH --partition=curtin_eecms
#SBATCH --gres=gpu:a100:1                   # ONE A100 80G (single-GPU loop, no DDP).
                                            # Wave fits with large headroom; a V100 may
                                            # suffice once the smoke test confirms usage.
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                           # generous; wave's host-RAM need is modest
#SBATCH --time=24:00:00                     # generous ceiling: 50k steps at wave capacity
                                            # should finish in a few hours — re-measure and trim.
#SBATCH --output=logs/slurm-wave-%j.out     # job stdout in logs/ (gitignored), not the repo root
# #SBATCH --account=<PROJECT>              # uncomment if submission demands an account

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

OUT=${OUT:-runs/wave-cgn-v02}               # override: OUT=runs/xyz sbatch ...
                                            # (train refuses an OUT that already has checkpoints)
mkdir -p "$OUT"
source .venv/bin/activate
export PYTHONPATH=src

DATA=/data/curtin_eecms/curtin_qilin/data/wave_propagation_1d

nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# train -> best checkpoint (selected on rollout position RMSE, ADR-0028)
# + config.json + normalization_stats.npz. Logs tee into the run dir so
# summarization never depends on slurm-* files.
python -m structbench.cli.train --mode train   --config configs/wave_propagation_1d/cgn.toml --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/train.log"
# eval on the held-out splits (val, test_interp — wave has no extrapolation
# split, ADR-0025): metrics go to $OUT/metrics-*.json + rollouts/*.npz. The
# architecture is rebuilt from the run's own config.json, so no --config here.
# If the train step is killed these never run -- rerun just these two lines.
python -m structbench.cli.train --mode valid   --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/eval.log"
python -m structbench.cli.train --mode rollout --data-root "$DATA" --out "$OUT" 2>&1 | tee -a "$OUT/eval.log"
