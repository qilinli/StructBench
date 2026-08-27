#!/bin/bash
# Taylor 2D Transolver-TC + Sobolev (H1) gradient supervision — completes the
# F-007 benchmark coverage (notch/DP/wave done; taylor untested). Seed-matched
# to the blessed TC controls (runs/taylor-transolver-tc-s{1,2}, 100k budget):
# same config, only STRUCTBENCH_H1_RATIO=0.35 added (the notch working point).
# Submit from the repo root ON A LOGIN NODE, one job per seed:
#   mkdir -p scratch/logs
#   SEED=1 sbatch hpc/dug/train_taylor_sobolev.slurm
#   SEED=2 sbatch hpc/dug/train_taylor_sobolev.slurm

#SBATCH --job-name=taylor-sobolev
#SBATCH --partition=curtin_eecms
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00                   # ~6 h expected for 100k (17k steps/h)
#SBATCH --output=scratch/logs/slurm-taylor-sobolev-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

SEED=${SEED:?set SEED=1 or SEED=2}
OUT=runs/taylor-transolver-tc-sobolev-s${SEED}
CONFIG=configs/taylor_impact_2d/transolver-timecond-iv-s${SEED}.toml
mkdir -p "$OUT"
source .venv/bin/activate
export PYTHONPATH=src
export STRUCTBENCH_H1_RATIO=0.35
export STRUCTBENCH_H1_NEIGHBORS=12

DATA=${DATA:-/data/curtin_eecms/curtin_qilin/data/taylor_impact}

echo "code commit: $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "sobolev ratio: $STRUCTBENCH_H1_RATIO  neighbors: $STRUCTBENCH_H1_NEIGHBORS"
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

python -m structbench.cli.train --mode train   --config "$CONFIG" --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/train.log"
python -m structbench.cli.train --mode valid   --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/eval.log"
python -m structbench.cli.train --mode rollout --data-root "$DATA" --out "$OUT" 2>&1 | tee -a "$OUT/eval.log"
