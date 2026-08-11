#!/bin/bash
# Full DeformingPlate MGN blessing run on DUG (SLURM) — ADR-0043 §8 recipe.
# Submit from the repo root ON A LOGIN NODE (sbatch does not work inside the
# JupyterHub container — no munge):
#   mkdir -p logs && sbatch hpc/dug/train_deforming_mgn.slurm
#
# Values live for the curtin_eecms partition (TIMELIMIT infinite; ~21 nodes
# with gpu:a100:2 as of 2026-07-10). Single-GPU loop; ONE A100 80G. The MGN
# footprint (~1,270-node tet meshes, batch 2, latent 128, 15 MP steps) is far
# below Taylor's radius-graph footprint — memory headroom is wide.

#SBATCH --job-name=dp-mgn
#SBATCH --partition=curtin_eecms
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14-00:00:00                   # ceiling only (partition allows more).
                                             # 10M steps at batch 2: project from the
                                             # measured steps/s in the first log lines
                                             # (e.g. ~20 steps/s ≈ 5.8 days). NO RESUME:
                                             # a killed run restarts from scratch — an
                                             # early model-best is usable for analysis,
                                             # but a partial run is a training-ledger
                                             # deviation, not the §8 blessing recipe.
#SBATCH --output=logs/slurm-dp-mgn-%j.out    # logs/ must exist BEFORE sbatch
# #SBATCH --account=<PROJECT>                # uncomment if submission demands an account

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

OUT=${OUT:-runs/deforming-mgn-v03}           # override: OUT=runs/xyz sbatch ...
                                             # (train refuses an OUT that already has checkpoints)
mkdir -p "$OUT"
source .venv/bin/activate
export PYTHONPATH=src

# World edges are dynamic contact edges (data-dependent count); a contact-heavy
# frame's processor activations OOM'd the first v03 attempt at ~760k steps (see
# hpc/dug/README.md incident note). Primary fix is MGNet activation
# checkpointing (default-on, output-identical). This complements it by letting
# the allocator reclaim the reserved-but-fragmented pool instead of failing.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA=${DATA:-/data/curtin_eecms/curtin_qilin/data/deforming_plate}

nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# train -> model-best-<step>.pt (selected on val rollout position RMSE every
# val_every=50k) + periodic ckpt-<step>.pt every 10k (analysis-only) +
# config.json. MGN checkpoints are self-contained (normalizer buffers in the
# state_dict; no normalization_stats.npz — ADR-0043 §8).
python -m structbench.cli.train --mode train   --config configs/deforming_plate/mgn.toml --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/train.log"
# eval: ADR-0043 §5 metrics -> $OUT/metrics-val.json / metrics-test.json +
# rollouts/*.npz. Architecture rebuilt from the run's own config.json (no
# --config). If train is killed these never run (set -e) — rerun just these
# lines against the same --out.
python -m structbench.cli.train --mode valid   --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/eval.log"
python -m structbench.cli.train --mode rollout --data-root "$DATA" --out "$OUT" 2>&1 | tee -a "$OUT/eval.log"
# ADR-0043 §8 blessing gate: pooled rollout-all position RMSE over ALL mesh
# nodes (dated-note convention), band 15.1±4.0e-3 native (11.1–19.1 mm);
# exits 1 on FAIL. Writes $OUT/blessing-pooled-test.json.
python tools/blessing_pooled_rmse.py --run-dir "$OUT" --data-root "$DATA" --split test 2>&1 | tee "$OUT/blessing.log"
