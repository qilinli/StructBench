#!/bin/bash
# One NotchBeam2D-Impact CGN ablation as a single-GPU SLURM job (round-1
# recipe fleet: capacity cell fixed at 128/10/2, noise arms at seed 1; the
# winner rewrites configs/notch_beam_2d_impact/cgn.toml at bless time, wave
# round-2 precedent). Cloned from ablate_wave.slurm — only the base config,
# data root, and walltime differ. Submit from the repo root ON A LOGIN NODE,
# parameterized via --export:
#
#   sbatch --job-name=ni-n002-s1 \
#       --export=ALL,NAME=n002-s1,HIDDEN=128,MP=10,NMLP=2,BATCH=8,SEED=1,NOISE=0.02,STEPS=50000,FLEET=runs/fleet-<date> \
#       hpc/dug/ablate_notch_impact.slurm
#
# The base config still carries the pre-round-1 small recipe (64/5/1), so
# round-1 arms pass the capacity overrides explicitly, exactly as wave
# round-2 did against its small base. Avoid seed 0 (CORRECTIONS 2026-07-10).
#
# Recognised variables (all optional except NAME):
#   NAME    run name -> output dir runs/fleet-<date>/<NAME>   (required)
#   NOISE   noise_std override            (default: keep configs/notch_beam_2d_impact/cgn.toml)
#   WAUX    w_aux override
#   HIDDEN  hidden_dim override
#   NMLP    nmlp_layers override
#   MP      message_passing_steps override
#   BATCH   batch_size override
#   SEED    run seed override             (avoid seed 0 — CORRECTIONS 2026-07-10)
#   STEPS   training_steps override       (default 100000, the baseline budget)
#   FLEET   fleet root                    (default runs/fleet-<today>; PIN THIS
#           when submitting a multi-job fleet, e.g. FLEET=runs/fleet-2026-07-12
#           — the default is evaluated at each job's START, so queued jobs that
#           cross midnight would fragment into two roots/MANIFESTs)
#
# Every run writes: <out>/config.toml (derived), <out>/config.json (resolved by
# the trainer), train.log (val_pos/val_aux every val_every), periodic
# ckpt-<step>.pt snapshots every 10k steps (for post-hoc smoothed selection;
# ignored by the default eval), metrics-*.json, rollouts/*.npz, job-info.txt;
# and appends one TSV line to <fleet>/MANIFEST.tsv (name, jobid, status,
# config deltas, timings) for post-hoc summarization.

#SBATCH --partition=curtin_eecms
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00                        # notch cases are several times Taylor's
                                               # particle count; Taylor's 100k took ~22 h,
                                               # so give capacity arms real headroom
#SBATCH --output=logs/slurm-ablate-%x-%j.out   # job stdout in logs/ (gitignored), not the repo root

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

: "${NAME:?set NAME via --export=ALL,NAME=...}"
STEPS=${STEPS:-100000}
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

text = open("configs/notch_beam_2d_impact/cgn.toml").read()
overrides = {
    "noise_std": os.environ.get("NOISE"),
    "w_aux": os.environ.get("WAUX"),
    "hidden_dim": os.environ.get("HIDDEN"),
    "nmlp_layers": os.environ.get("NMLP"),
    "message_passing_steps": os.environ.get("MP"),
    "batch_size": os.environ.get("BATCH"),
    "seed": os.environ.get("SEED"),
    "training_steps": os.environ.get("STEPS", "100000"),
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

DATA=/data/curtin_eecms/curtin_qilin/data/notch_beam_2d_impact
# tee into the run dir: train.log / eval.log live beside the artifacts they
# describe, so post-hoc summarization never depends on slurm-* files.
python -m structbench.cli.train --mode train   --config "$OUT/config.toml" --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/train.log"
# eval covers val, test_interp, and probe (ADR-0026/0029 splits); the
# architecture is rebuilt from the run's own config.json, so no --config here.
python -m structbench.cli.train --mode valid   --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/eval.log"
python -m structbench.cli.train --mode rollout --data-root "$DATA" --out "$OUT" 2>&1 | tee -a "$OUT/eval.log"
