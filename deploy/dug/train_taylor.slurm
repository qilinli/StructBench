#!/bin/bash
# Full Taylor 2D GNS baseline on DUG (SLURM). Submit from the repo root:
#   mkdir -p runs/taylor-baseline
#   sbatch deploy/dug/train_taylor.slurm
#
# Fill the <...> placeholders first (partition, GPU gres, data path, and
# account if your DUG project needs one). Confirm partition + gres with:
#   sinfo -o "%P %G"      # partitions and their gres (e.g. gpu:a100:2, gpu:v100:4)

#SBATCH --job-name=taylor-gns
#SBATCH --partition=<GPU_PARTITION>        # <-- from `sinfo`
#SBATCH --gres=<GPU_GRES>                   # <-- e.g. gpu:a100:1 -- ONE typed A100 80G (single-GPU
                                            # loop, no DDP). Use untyped gpu:1 ONLY on an A100-only
                                            # partition: a mixed partition can hand you a 16 GB V100,
                                            # which is marginal for the batch-32 run (~14 GB) and slow.
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00                     # hard budget: training has NO resume. The best checkpoint
                                            # (saved whenever validation improves; first at step 2000)
                                            # survives a timeout, but a rerun needs a FRESH --out
                                            # (train refuses a dir that already holds checkpoints).
#SBATCH --output=slurm-taylor-%j.out        # written to the submit dir (always exists)
# #SBATCH --account=<PROJECT>              # uncomment if your DUG project requires it

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/taylor-baseline
source .venv/bin/activate
export PYTHONPATH=src

DATA=<PATH_TO_h5_canonical>                 # <-- where you rsync'd the 34 .h5 files

nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# train -> best checkpoint + config.json + normalization_stats.npz
python -m structbench.cli.train --mode train   --config configs/taylor_2d.toml --data-root "$DATA" --out runs/taylor-baseline
# eval on the held-out splits: ADR-0019 metrics go to the slurm log AND to
# runs/taylor-baseline/metrics-*.json + rollouts/*.npz. The architecture is
# rebuilt from the run's own config.json, so no --config here. If the train
# step is killed by walltime these never run -- rerun just these two lines.
python -m structbench.cli.train --mode valid   --data-root "$DATA" --out runs/taylor-baseline
python -m structbench.cli.train --mode rollout --data-root "$DATA" --out runs/taylor-baseline
