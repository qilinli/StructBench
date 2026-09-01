#!/bin/bash
# Stage-1 state-sufficiency probe on Taylor 2D as a single-GPU SLURM job.
#
# Trains a one-step neighbourhood operator F: z_n -> dz teacher-forced on
# ground-truth pairs, under two input arms:
#
#   ARM=full        node features = v, s, peeq, E, rho   (8 channels)
#   ARM=kinematic   node features = v                     (2 channels)
#
# Both arms predict the same eight increments, so per-field errors compare
# directly. The ARM is the experiment; everything else is held fixed.
#
# Submit from the repo root ON A LOGIN NODE. Build the cache ONCE first --
# four jobs starting together would race on it:
#
#   mkdir -p scratch/logs
#   source .venv/bin/activate && export PYTHONPATH=src:tools/state_probe
#   python tools/state_probe/probe.py --prepare-cache \
#       --data-root /data/curtin_eecms/curtin_qilin/data/taylor_impact \
#       --cache-dir scratch/probe-cache
#
# then the four-job fleet (2 arms x 2 seeds; >=2 concurrent per the 2026-07-10
# correction, and seeds start at 1 -- never 0):
#
#   for ARM in full kinematic; do for S in 1 2; do
#     sbatch --job-name=probe-$ARM-s$S --export=ALL,ARM=$ARM,SEED=$S \
#         hpc/dug/probe_state_sufficiency.slurm
#   done; done
#
# Recognised variables:
#   ARM        full | kinematic                  (required)
#   SEED       training seed                     (default 1; never 0)
#   STEPS      training steps                    (default 30000)
#   DATA       data root  (default /data/curtin_eecms/curtin_qilin/data/taylor_impact)
#   CACHE      memmap cache dir                  (default scratch/probe-cache)
#   PROBE_DIR  probe source dir                  (default tools/state_probe)
#   OUT        results dir                       (default runs/probe-<ARM>-s<SEED>)
#
# Model size is the A100 configuration (hidden 128, 6 blocks, batch 2), NOT
# the reduced laptop defaults in probe.py -- an 80 GB card has no reason to
# run the MPS-constrained shape. Both arms get identical budget, which is what
# keeps the ablation fair.
#
# No numeric gate. This probe decides whether the complete-state direction is
# worth a pipeline change, not whether a model is blessed.
#
# Single-GPU loop; ONE A100 80G per job -- nodes carry gpu:a100:2, so jobs
# pack two-per-node automatically (no exclusive flags).

#SBATCH --partition=curtin_eecms
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=scratch/logs/slurm-%x-%j.out       # scratch/logs/ must exist BEFORE sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

: "${ARM:?set ARM=full or ARM=kinematic, e.g. --export=ALL,ARM=full}"
case "$ARM" in
  full|kinematic) ;;
  *) echo "ARM must be 'full' or 'kinematic', got: $ARM" >&2; exit 1 ;;
esac

SEED=${SEED:-1}
[ "$SEED" -ge 1 ] || { echo "SEED must be >= 1 (2026-07-10 correction)" >&2; exit 1; }

STEPS=${STEPS:-30000}
DATA=${DATA:-/data/curtin_eecms/curtin_qilin/data/taylor_impact}
CACHE=${CACHE:-scratch/probe-cache}
PROBE_DIR=${PROBE_DIR:-tools/state_probe}
OUT=${OUT:-runs/probe-${ARM}-s${SEED}}

[ -f "$PROBE_DIR/probe.py" ] || { echo "no probe at $PROBE_DIR/probe.py" >&2; exit 1; }
[ -d "$CACHE" ] || { echo "cache missing at $CACHE -- run --prepare-cache first" >&2; exit 1; }

mkdir -p "$OUT"
source .venv/bin/activate
export PYTHONPATH="src:$PROBE_DIR"

echo "code commit: $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "arm: $ARM  seed: $SEED  steps: $STEPS  out: $OUT"
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Per-field increment relative L2 on val + test_interp, logged every eval and
# written to $OUT/probe-<arm>-s<seed>.json. No checkpoints: the probe answers a
# comparison, and nothing downstream loads its weights.
python "$PROBE_DIR/probe.py" \
    --arm "$ARM" \
    --seed "$SEED" \
    --steps "$STEPS" \
    --data-root "$DATA" \
    --cache-dir "$CACHE" \
    --out "$OUT" \
    --batch-size 2 \
    --hidden 128 \
    --n-steps 6 \
    --eval-every 2000 \
    2>&1 | tee "$OUT/probe.log"
