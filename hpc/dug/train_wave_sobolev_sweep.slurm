#!/bin/bash
# Wave-1D Transolver-TC + Sobolev gradient supervision — RATIO SWEEP (F-007
# open residual: wave was only tested at the transferred notch ratio 0.35,
# ~3x harmful; this sweeps the useful band to bound the smooth-regime harm).
# Seed-matched to runs/wave-transolver-tc-s{1,2} (50k budget, commit 6615d2e).
# Submit from THIS worktree root ON A LOGIN NODE, one job per (seed, ratio):
#   mkdir -p scratch/logs
#   SEED=1 RATIO=0.1 sbatch hpc/dug/train_wave_sobolev_sweep.slurm
#   SEED=1 RATIO=0.2 sbatch hpc/dug/train_wave_sobolev_sweep.slurm
#   SEED=2 RATIO=0.1 sbatch hpc/dug/train_wave_sobolev_sweep.slurm
#   SEED=2 RATIO=0.2 sbatch hpc/dug/train_wave_sobolev_sweep.slurm

#SBATCH --job-name=wave-sobolev
#SBATCH --partition=curtin_eecms
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=scratch/logs/slurm-wave-sobolev-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

SEED=${SEED:?set SEED=1 or SEED=2}
RATIO=${RATIO:?set RATIO=0.1 or RATIO=0.2}
RTAG=$(echo "$RATIO" | tr -d '.')
OUT=runs/wave-transolver-tc-sobolev-s${SEED}-r${RTAG}
CONFIG=configs/wave_propagation_1d/transolver-timecond-s${SEED}.toml
mkdir -p "$OUT"
source .venv/bin/activate
export PYTHONPATH=src
export STRUCTBENCH_H1_RATIO=$RATIO
export STRUCTBENCH_H1_NEIGHBORS=12

DATA=/data/curtin_eecms/curtin_qilin/data/wave_propagation_1d

echo "code commit: $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "sobolev ratio: $STRUCTBENCH_H1_RATIO  neighbors: $STRUCTBENCH_H1_NEIGHBORS"
nvidia-smi

python -m structbench.cli.train --mode train   --config "$CONFIG" --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/train.log"
python -m structbench.cli.train --mode valid   --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/eval.log"
python -m structbench.cli.train --mode rollout --data-root "$DATA" --out "$OUT" 2>&1 | tee -a "$OUT/eval.log"
