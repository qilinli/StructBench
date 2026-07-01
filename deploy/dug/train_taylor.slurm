#!/bin/bash
# Full Taylor 2D GNS baseline on DUG (SLURM). Submit from the repo root:
#   mkdir -p runs/taylor-baseline
#   sbatch deploy/dug/train_taylor.slurm
#
# Fill the three <...> placeholders first (partition, data path, and account if
# your DUG project needs one). Confirm the partition + GPU gres string with:
#   sinfo -o "%P %G"      # partitions and their gres (e.g. gpu:a100:2, gpu:v100:4)

#SBATCH --job-name=taylor-gns
#SBATCH --partition=<GPU_PARTITION>        # <-- from `sinfo`
#SBATCH --gres=gpu:1                        # ONE GPU: the training loop is single-GPU (no DDP).
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00                     # generous; best checkpoint is saved every 2000 steps,
                                            # so an early stop / timeout still yields a usable model.
#SBATCH --output=slurm-taylor-%j.out        # written to the submit dir (always exists)
# #SBATCH --account=<PROJECT>              # uncomment if your DUG project requires it
# #SBATCH --gres=gpu:a100:1                # prefer the A100 80G if GPUs are typed (confirm the string)

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
# eval on the held-out splits (numbers go to the slurm log)
python -m structbench.cli.train --mode valid   --config configs/taylor_2d.toml --data-root "$DATA" --out runs/taylor-baseline
python -m structbench.cli.train --mode rollout --config configs/taylor_2d.toml --data-root "$DATA" --out runs/taylor-baseline
