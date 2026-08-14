#!/bin/bash
# Generic single-GPU arm launcher: the config FILE is the arm (ADR-0032
# discipline; no override knobs). Works for any benchmark — resolves
# configs/$BENCH/$ARM.toml. Submit from the repo root ON A LOGIN NODE, one job
# per arm, e.g.:
#
#   mkdir -p logs
#   sbatch --job-name=os-taylor-s1 \
#       --export=ALL,BENCH=taylor_impact_2d,ARM=transolver-oneshot-s1,\
#OUT=runs/oneshot-taylor-transolver-s1,DATA=/data/curtin_eecms/curtin_qilin/data/taylor_impact \
#       hpc/dug/train_arm.slurm
#
# Recognised variables:
#   BENCH  benchmark dir under configs/            (required)
#   ARM    config stem under configs/$BENCH/       (required)
#   DATA   canonical data root                     (required)
#   OUT    output dir                              (default runs/$ARM)
#
# ONE A100 80G per job; nodes carry gpu:a100:2 so jobs pack two-per-node
# automatically (no exclusive flags). 2-day ceiling covers the notch 250k /
# one-shot budgets; override --time / --mem via sbatch if needed.

#SBATCH --partition=curtin_eecms
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm-%x-%j.out       # logs/ must exist BEFORE sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

: "${BENCH:?set BENCH=<benchmark dir under configs/>}"
: "${ARM:?set ARM=<config stem under configs/$BENCH/>}"
: "${DATA:?set DATA=<canonical data root>}"
CONFIG="configs/${BENCH}/${ARM}.toml"
[ -f "$CONFIG" ] || { echo "no such arm config: $CONFIG" >&2; exit 1; }

OUT=${OUT:-runs/${ARM}}
mkdir -p "$OUT"
source .venv/bin/activate
export PYTHONPATH=src

echo "code commit: $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "bench: $BENCH  arm: $ARM  config: $CONFIG  out: $OUT  data: $DATA"
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# train -> checkpoints + config.json (self-contained; normalizer buffers in
# state_dict). eval rebuilds the architecture from the run's config.json, so
# the two eval lines are re-runnable standalone if train is killed (set -e).
python -m structbench.cli.train --mode train   --config "$CONFIG" --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/train.log"
python -m structbench.cli.train --mode valid   --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/eval.log"
python -m structbench.cli.train --mode rollout --data-root "$DATA" --out "$OUT" 2>&1 | tee -a "$OUT/eval.log"
