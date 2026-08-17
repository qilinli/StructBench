#!/bin/bash
# Taylor 2D GeoFLARE PROVISIONAL run on DUG (SLURM) — ADR-0047 recipe (ADR-0045
# family recipe on the synthesized lattice mesh + wall nodes; CGN-matched
# 100k budget). NO numeric gate; metrics recorded provisional per ADR-0046.
# Submit from the repo root ON A LOGIN NODE:
#   mkdir -p logs && sbatch hpc/dug/train_taylor_geoflare.slurm
#
# Single-GPU loop; ONE A100 80G per job — nodes carry gpu:a100:2, so jobs
# pack two-per-node automatically (no exclusive flags).

#SBATCH --job-name=taylor-geoflare
#SBATCH --partition=curtin_eecms
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00                   # ceiling. Naive scaling from the DP GeoFLARE run
                                            # (~66k steps/h at ~1.3k nodes) puts taylor's
                                            # ~8.2k nodes at ~10k steps/h -> ~10 h for 100k;
                                            # 2 d covers a large misestimate.
#SBATCH --output=scratch/logs/slurm-taylor-geoflare-%j.out    # scratch/logs/ must exist BEFORE sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

OUT=${OUT:-runs/taylor-geoflare-adr0047}         # override: OUT=runs/xyz sbatch ...
mkdir -p "$OUT"
source .venv/bin/activate
export PYTHONPATH=src

DATA=${DATA:-/data/curtin_eecms/curtin_qilin/data/taylor_impact}

echo "code commit: $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# train -> model-best-<step>.pt (val rollout position RMSE, every 2k; only 3
# val cases so validation is cheap) + periodic ckpt-<step>.pt + config.json.
# Self-contained checkpoints (normalizer buffers in state_dict).
python -m structbench.cli.train --mode train   --config configs/taylor_impact_2d/geoflare.toml --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/train.log"
# eval on val + test_interp + test_extrap: metrics-*.json + rollouts/*.npz
# (architecture rebuilt from the run's config.json). If train is killed these
# never ran (set -e) — rerun just these lines.
python -m structbench.cli.train --mode valid   --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/eval.log"
python -m structbench.cli.train --mode rollout --data-root "$DATA" --out "$OUT" 2>&1 | tee -a "$OUT/eval.log"
