#!/bin/bash
# One Taylor 2D GNS ablation as a single-GPU SLURM job (ADR-0028 Phase 2).
# Submit from the repo root ON A LOGIN NODE, parameterized via --export:
#
#   sbatch --job-name=abl-n005 --export=ALL,NAME=n005,NOISE=0.05 \
#       hpc/dug/ablate_taylor.slurm
#
# Recognised variables (all optional except NAME):
#   NAME    run name -> output dir runs/fleet-<date>/<NAME>   (required)
#   NOISE   noise_std override            (default: keep configs/taylor_impact_2d/gns.toml)
#   WAUX    w_aux override
#   HIDDEN  hidden_dim override
#   NMLP    nmlp_layers override
#   MP      message_passing_steps override
#   BATCH   batch_size override
#   SEED    run seed override
#   STEPS   training_steps override       (default 20000)
#   FLEET   fleet root                    (default runs/fleet-<today>)
#
# Every run writes: <out>/config.toml (derived), <out>/config.json (resolved by
# the trainer), train.log (val_pos/val_aux every 2000 steps), metrics-*.json,
# rollouts/*.npz, job-info.txt; and appends one TSV line to <fleet>/MANIFEST.tsv
# (name, jobid, status, config deltas, timings) for post-hoc summarization.

#SBATCH --partition=curtin_eecms
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-ablate-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

: "${NAME:?set NAME via --export=ALL,NAME=...}"
STEPS=${STEPS:-20000}
FLEET=${FLEET:-runs/fleet-$(date +%F)}
OUT="$FLEET/$NAME"
mkdir -p "$OUT"
MANIFEST="$FLEET/MANIFEST.tsv"
[ -f "$MANIFEST" ] || echo -e "name\tjobid\tstatus\tdeltas\tstart\tend\tcommit" > "$MANIFEST"

DELTAS="steps=$STEPS"
for v in NOISE WAUX HIDDEN NMLP MP BATCH SEED; do
  [ -n "${!v:-}" ] && DELTAS="$DELTAS,$(echo $v | tr 'A-Z' 'a-z')=${!v}"
done
COMMIT=$(PATH=/d/sw/git/latest/bin:$PATH git rev-parse --short HEAD 2>/dev/null || echo unknown)
START=$(date +%FT%T)

finish() {
  local status=$?
  [ $status -eq 0 ] && s=DONE || s="FAILED($status)"
  echo -e "$NAME\t${SLURM_JOB_ID:-?}\t$s\t$DELTAS\t$START\t$(date +%FT%T)\t$COMMIT" >> "$MANIFEST"
}
trap finish EXIT

# Derive the run config from the canonical one, overriding only exported keys.
python3 - "$OUT/config.toml" <<'PY'
import os, re, sys

text = open("configs/taylor_impact_2d/gns.toml").read()
overrides = {
    "noise_std": os.environ.get("NOISE"),
    "w_aux": os.environ.get("WAUX"),
    "hidden_dim": os.environ.get("HIDDEN"),
    "nmlp_layers": os.environ.get("NMLP"),
    "message_passing_steps": os.environ.get("MP"),
    "batch_size": os.environ.get("BATCH"),
    "seed": os.environ.get("SEED"),
    "training_steps": os.environ.get("STEPS", "20000"),
}
for key, value in overrides.items():
    if value is None:
        continue
    pattern = rf"(?m)^{key} = \S+"
    if not re.search(pattern, text):
        sys.exit(f"key {key} not found in base config")
    text = re.sub(pattern, f"{key} = {value}", text)
open(sys.argv[1], "w").write(text)
print(f"wrote {sys.argv[1]}")
PY

source .venv/bin/activate
export PYTHONPATH=src

{
  echo "name=$NAME jobid=${SLURM_JOB_ID:-?} node=$(hostname) commit=$COMMIT"
  echo "deltas=$DELTAS"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
} | tee "$OUT/job-info.txt"

DATA=/data/curtin_eecms/curtin_qilin/data/taylor_impact
# tee into the run dir: train.log / eval.log live beside the artifacts they
# describe, so post-hoc summarization never depends on slurm-* files.
python -m structbench.cli.train --mode train   --config "$OUT/config.toml" --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/train.log"
python -m structbench.cli.train --mode valid   --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/eval.log"
python -m structbench.cli.train --mode rollout --data-root "$DATA" --out "$OUT" 2>&1 | tee -a "$OUT/eval.log"
