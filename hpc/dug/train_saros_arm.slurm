#!/bin/bash
# SAROS batch arm: one single-GPU job for benchmark BENCH, config stem ARM.
#   sbatch --job-name=<name> --export=ALL,BENCH=<taylor_impact_2d|notch_beam_2d_impact>,ARM=<stem>,OUT=<dir> hpc/dug/train_saros_arm.slurm
#SBATCH --partition=curtin_eecms
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${BENCH:?set BENCH}"; : "${ARM:?set ARM}"
CONFIG="configs/${BENCH}/${ARM}.toml"; [ -f "$CONFIG" ] || { echo "no config $CONFIG"; exit 1; }
declare -A DATAMAP=( [taylor_impact_2d]=taylor_impact [notch_beam_2d_impact]=notch_beam_2d_impact )
DATA=${DATA:-/data/curtin_eecms/curtin_qilin/data/${DATAMAP[$BENCH]}}
OUT=${OUT:-runs/${ARM}}
mkdir -p "$OUT"; source .venv/bin/activate; export PYTHONPATH=src
echo "commit $(git rev-parse --short HEAD) | $CONFIG | $DATA"
python -m structbench.cli.train --mode train   --config "$CONFIG" --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/train.log"
python -m structbench.cli.train --mode valid   --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/eval.log"
python -m structbench.cli.train --mode rollout --data-root "$DATA" --out "$OUT" 2>&1 | tee -a "$OUT/eval.log"
