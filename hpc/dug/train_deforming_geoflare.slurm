#!/bin/bash
# DeformingPlate Geoflare PROVISIONAL run on DUG (SLURM) — ADR-0045 recipe.
# NO numeric gate applies (no published rollout anchor); the run's §5 metrics
# are recorded provisional=True per ADR-0046 clause 7. Submit from the repo
# root ON A LOGIN NODE:
#   mkdir -p logs && sbatch hpc/dug/train_deforming_geoflare.slurm
#
# Single-GPU loop; ONE A100 80G per job — nodes carry gpu:a100:2, so two of
# these jobs pack per node automatically (no exclusive flags).

#SBATCH --job-name=dp-geoflare
#SBATCH --partition=curtin_eecms
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14-00:00:00                   # ceiling only (partition allows more).
                                             # Budget matched to MGN (10M steps, batch 2);
                                             # NO RESUME — project wall-clock from the
                                             # measured steps/s early and scancel+re-budget
                                             # if unacceptable (ledger deviation; no gate).
#SBATCH --output=logs/slurm-dp-geoflare-%j.out    # logs/ must exist BEFORE sbatch
# #SBATCH --account=<PROJECT>                # uncomment if submission demands an account

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

OUT=${OUT:-runs/deforming-geoflare-v03}                       # override: OUT=runs/xyz sbatch ...
mkdir -p "$OUT"
source .venv/bin/activate
export PYTHONPATH=src

DATA=${DATA:-/data/curtin_eecms/curtin_qilin/data/deforming_plate}

nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# train -> model-best-<step>.pt (val rollout position RMSE, every 50k) +
# periodic ckpt-<step>.pt every 10k + config.json. Self-contained checkpoints
# (normalizer buffers in state_dict; no stats file).
python -m structbench.cli.train --mode train   --config configs/deforming_plate/geoflare.toml --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/train.log"
# eval: ADR-0043 §5 metrics -> metrics-val.json / metrics-test.json +
# rollouts/*.npz (architecture rebuilt from the run's config.json). If train
# is killed these never ran (set -e) — rerun just these lines.
python -m structbench.cli.train --mode valid   --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/eval.log"
python -m structbench.cli.train --mode rollout --data-root "$DATA" --out "$OUT" 2>&1 | tee -a "$OUT/eval.log"
# Pooled aggregate: INFORMATIONAL ONLY for provisional families (ADR-0045;
# the 11.1-19.1 mm band is the MGN blessing gate, not a gate here) — hence
# "|| true" so an out-of-band value cannot fail the job.
python tools/blessing_pooled_rmse.py --run-dir "$OUT" --data-root "$DATA" --split test 2>&1 | tee "$OUT/pooled-informational.log" || true
