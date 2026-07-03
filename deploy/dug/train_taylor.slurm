#!/bin/bash
# Full Taylor 2D GNS baseline on DUG (SLURM). Submit from the repo root ON A
# LOGIN NODE (sbatch does not work inside the JupyterHub container — no munge):
#   sbatch deploy/dug/train_taylor.slurm
#
# Values below are live for the curtin_eecms partition (verified 2026-07-03:
# one 2x A100-80GB node, TIMELIMIT infinite). The partition's A100 node is
# shared with JupyterHub sessions — a submitted job PENDs until the session
# releases the GPUs.

#SBATCH --job-name=taylor-gns
#SBATCH --partition=curtin_eecms
#SBATCH --gres=gpu:a100:1                   # ONE A100 80G (single-GPU loop, no DDP).
                                            # The V100 node is too small for the
                                            # radius-1.5 memory footprint (~50-60 GB);
                                            # the mi50 node is AMD (no CUDA build).
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00                     # generous ceiling (partition allows more);
                                            # ~22h expected for 100k steps at the
                                            # measured 4.6k steps/h (radius 1.5).
#SBATCH --output=slurm-taylor-%j.out        # written to the submit dir (always exists)
# #SBATCH --account=<PROJECT>              # uncomment if submission demands an account

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

OUT=${OUT:-runs/taylor-full-adr0024}        # override: OUT=runs/xyz sbatch ...
mkdir -p "$OUT"
source .venv/bin/activate
export PYTHONPATH=src

DATA=/data/curtin_eecms/curtin_qilin/data/taylor_impact

nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# train -> best checkpoint (selected on rollout position RMSE, ADR-0024)
# + config.json + normalization_stats.npz. Logs tee into the run dir so
# summarization never depends on slurm-* files.
python -m structbench.cli.train --mode train   --config configs/taylor_2d.toml --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/train.log"
# eval on the held-out splits: ADR-0019 metrics go to $OUT/metrics-*.json +
# rollouts/*.npz. The architecture is rebuilt from the run's own config.json,
# so no --config here. If the train step is killed these never run -- rerun
# just these two lines.
python -m structbench.cli.train --mode valid   --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/eval.log"
python -m structbench.cli.train --mode rollout --data-root "$DATA" --out "$OUT" 2>&1 | tee -a "$OUT/eval.log"
