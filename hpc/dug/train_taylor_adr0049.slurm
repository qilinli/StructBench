#!/bin/bash
# One Taylor 2D ADR-0049 repair-fleet arm as a single-GPU SLURM job.
# The config FILE is the arm — no override knobs here (ADR-0032 discipline);
# the six arms live in configs/taylor_impact_2d/{transolver,geoflare,mgn}-n02*
# (noise rescale / +velocity history / +MGN stretch gate; ADR-0049).
# Submit from the repo root ON A LOGIN NODE, one job per arm:
#
#   mkdir -p logs
#   sbatch --job-name=taylor-transolver-n02 --export=ALL,ARM=transolver-n02 \
#       hpc/dug/train_taylor_adr0049.slurm
#
# Recognised variables:
#   ARM   config stem under configs/taylor_impact_2d/   (required)
#   DATA  data root            (default /data/curtin_eecms/curtin_qilin/data/taylor_impact)
#   OUT   output dir           (default runs/taylor-<ARM>-adr0049)
#
# Ceiling covers the slowest family (ADR-0047 mgn: 12.7 h at 100k steps);
# transolver arms finish in ~2.5 h, geoflare in ~6 h — override with
# sbatch --time=... if queueing tight. NO numeric gate; metrics recorded
# provisional per ADR-0046.
#
# Single-GPU loop; ONE A100 80G per job — nodes carry gpu:a100:2, so jobs
# pack two-per-node automatically (no exclusive flags).

#SBATCH --partition=curtin_eecms
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm-%x-%j.out       # logs/ must exist BEFORE sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

: "${ARM:?set ARM=<config stem>, e.g. --export=ALL,ARM=transolver-n02}"
CONFIG="configs/taylor_impact_2d/${ARM}.toml"
[ -f "$CONFIG" ] || { echo "no such arm config: $CONFIG" >&2; exit 1; }

OUT=${OUT:-runs/taylor-${ARM}-adr0049}
mkdir -p "$OUT"
source .venv/bin/activate
export PYTHONPATH=src

DATA=${DATA:-/data/curtin_eecms/curtin_qilin/data/taylor_impact}

echo "code commit: $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "arm: $ARM  config: $CONFIG  out: $OUT"
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# train -> model-best-<step>.pt (val rollout position RMSE, every 2k) +
# periodic ckpt-<step>.pt + config.json. Self-contained checkpoints
# (normalizer buffers in state_dict).
python -m structbench.cli.train --mode train   --config "$CONFIG" --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/train.log"
# eval on val + test_interp + test_extrap: metrics-*.json + rollouts/*.npz
# (architecture rebuilt from the run's config.json). If train is killed these
# never ran (set -e) — rerun just these lines.
python -m structbench.cli.train --mode valid   --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/eval.log"
python -m structbench.cli.train --mode rollout --data-root "$DATA" --out "$OUT" 2>&1 | tee -a "$OUT/eval.log"
