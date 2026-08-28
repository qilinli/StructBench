#!/bin/bash
# One wave-1D multi-method fleet arm as a single-GPU SLURM job (2026-08-27).
# The config FILE is the arm — no override knobs here (ADR-0032 discipline);
# the arms live in configs/wave_propagation_1d/{mgn,transolver,geoflare}-*.
# Submit from the repo root ON A LOGIN NODE, one job per arm:
#
#   mkdir -p scratch/logs
#   sbatch --job-name=wave-mgn-s1 --export=ALL,ARM=mgn-s1 \
#       hpc/dug/train_wave_fleet.slurm
#
# Recognised variables:
#   ARM   config stem under configs/wave_propagation_1d/   (required)
#   DATA  data root   (default /data/curtin_eecms/curtin_qilin/data/wave_propagation_1d)
#   OUT   output dir  (default runs/wave-<ARM>)
#
# Wave is small (~500-1250 particles/case, 50k steps): transolver/geoflare
# arms should finish in ~1-2 h, MGN similar at this particle count. The 12 h
# ceiling is generous headroom, not a forecast. NO numeric gate; metrics
# recorded provisional per ADR-0046.
#
# Single-GPU loop; ONE A100 80G per job — nodes carry gpu:a100:2, so jobs
# pack two-per-node automatically (no exclusive flags).

#SBATCH --partition=curtin_eecms
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=scratch/logs/slurm-%x-%j.out       # scratch/logs/ must exist BEFORE sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

: "${ARM:?set ARM=<config stem>, e.g. --export=ALL,ARM=mgn-s1}"
CONFIG="configs/wave_propagation_1d/${ARM}.toml"
[ -f "$CONFIG" ] || { echo "no such arm config: $CONFIG" >&2; exit 1; }

OUT=${OUT:-runs/wave-${ARM}}
mkdir -p "$OUT"
source .venv/bin/activate
export PYTHONPATH=src

DATA=${DATA:-/data/curtin_eecms/curtin_qilin/data/wave_propagation_1d}

echo "code commit: $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "arm: $ARM  config: $CONFIG  out: $OUT"
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# train -> model-best-<step>.pt (val rollout position RMSE, every 2k) +
# periodic ckpt-<step>.pt + config.json. Self-contained checkpoints
# (normalizer buffers in state_dict).
python -m structbench.cli.train --mode train   --config "$CONFIG" --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/train.log"
# eval on val + test_interp (wave has no extrapolation split, ADR-0025):
# metrics-*.json + rollouts/*.npz (architecture rebuilt from the run's
# config.json). If train is killed these never ran (set -e) — rerun just
# these lines.
python -m structbench.cli.train --mode valid   --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/eval.log"
python -m structbench.cli.train --mode rollout --data-root "$DATA" --out "$OUT" 2>&1 | tee -a "$OUT/eval.log"
