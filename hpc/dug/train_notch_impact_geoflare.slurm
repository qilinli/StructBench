#!/bin/bash
# Notch-impact GeoFLARE PROVISIONAL run on DUG (SLURM) — ADR-0048 recipe (ADR-0045
# family recipe on the beam-only synthesized lattice mesh; pin + supports
# unmeshed kinematic+scripted; h250c-matched 250k budget, ADR-0039
# train_frames=250). NO numeric gate; metrics recorded provisional per
# ADR-0046. Submit from the repo root ON A LOGIN NODE:
#   mkdir -p logs && sbatch hpc/dug/train_notch_impact_geoflare.slurm
#
# Single-GPU loop; ONE A100 80G per job — nodes carry gpu:a100:2, so jobs
# pack two-per-node automatically (no exclusive flags).

#SBATCH --job-name=notch-geoflare
#SBATCH --partition=curtin_eecms
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00                   # ceiling. The taylor GeoFLARE arm measured ~17.5k
                                            # steps/h at ~8.2k nodes -> ~14-20 h for 250k;
                                            # 2 d is a generous ceiling.
#SBATCH --output=scratch/logs/slurm-notch-geoflare-%j.out    # scratch/logs/ must exist BEFORE sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

OUT=${OUT:-runs/notch-geoflare-adr0048}      # override: OUT=runs/xyz sbatch ...
mkdir -p "$OUT"
source .venv/bin/activate
export PYTHONPATH=src

DATA=${DATA:-/data/curtin_eecms/curtin_qilin/data/notch_beam_2d_impact}

echo "code commit: $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# train -> model-best-<step>.pt (val rollout position RMSE over the 8 val
# cases' 244-step scored span, every 2k) + periodic ckpt-<step>.pt +
# config.json. Self-contained checkpoints (normalizer buffers in state_dict).
python -m structbench.cli.train --mode train   --config configs/notch_beam_2d_impact/geoflare.toml --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/train.log"
# eval on val + test_interp + probe: metrics-*.json + rollouts/*.npz
# (architecture rebuilt from the run's config.json). If train is killed these
# never ran (set -e) — rerun just these lines.
python -m structbench.cli.train --mode valid   --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/eval.log"
python -m structbench.cli.train --mode rollout --data-root "$DATA" --out "$OUT" 2>&1 | tee -a "$OUT/eval.log"
