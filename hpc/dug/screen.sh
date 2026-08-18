#!/bin/bash
# One-shot idea screen: reduced-budget train -> auto verdict. The edit->trial->
# verdict loop as a single fire-and-forget command.
#
#   hpc/dug/screen.sh <bench> <base> <name> <steps> [key=val ...]
#
# Does: (1) tools/screen.py gen  -> configs/<bench>/<name>.toml (base + overrides
# + reduced steps); (2) sbatch train_arm.slurm -> runs/<name> (train+valid+
# rollout); (3) a dependent (afterany) verdict job -> scratch/screen-<name>.txt
# with val + slice-KL, compared against $VS if set. ~35 min on 1 GPU @ 25k.
#
# Compare against a cached baseline screen (run once) via VS:
#   VS=$PWD/runs/screen-baseline hpc/dug/screen.sh notch_beam_2d_impact \
#       transolver-timecond-iv-s1 screen-phitau 25000 \
#       adaptive_temperature=true temperature_phi=true
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate
export PYTHONPATH=src

BENCH=$1 BASE=$2 NAME=$3 STEPS=$4; shift 4
# force the screen- prefix so the generated config is git-ignored (.gitignore)
[[ $NAME == screen-* ]] || NAME="screen-$NAME"
SETS=(); for kv in "$@"; do SETS+=(--set "$kv"); done
declare -A DATADIR=(
  [taylor_impact_2d]=taylor_impact
  [notch_beam_2d_impact]=notch_beam_2d_impact
  [deforming_plate]=deforming_plate
)
DATA=/data/curtin_eecms/curtin_qilin/data/${DATADIR[$BENCH]}
OUT=$PWD/runs/$NAME
mkdir -p scratch/logs

# 1) generate the reduced-budget screening config
python tools/screen.py gen --bench "$BENCH" --base "$BASE" --name "$NAME" \
  --steps "$STEPS" "${SETS[@]}"

# 2) train (train_arm.slurm resolves configs/<bench>/<name>.toml)
TRAIN_ID=$(sbatch --parsable --job-name="scr-$NAME" \
  --export=ALL,BENCH="$BENCH",ARM="$NAME",DATA="$DATA",OUT="$OUT" \
  hpc/dug/train_arm.slurm)
echo "train job: $TRAIN_ID -> $OUT"

# 3) dependent verdict (runs whether train succeeds or fails, reports what landed)
VS_ARG=""; [ -n "${VS:-}" ] && VS_ARG="$VS"
VERDICT_ID=$(sbatch --parsable --job-name="scrv-$NAME" \
  --dependency=afterany:"$TRAIN_ID" \
  --partition=curtin_eecms --gres=gpu:a100:1 --cpus-per-task=4 --mem=32G \
  --time=00:20:00 --output="scratch/logs/scrv-$NAME-%j.out" \
  --wrap="cd $PWD && source .venv/bin/activate && export PYTHONPATH=src && \
    python tools/screen.py verdict --bench $BENCH --runs $OUT $VS_ARG \
    | tee scratch/screen-$NAME.txt")
echo "verdict job: $VERDICT_ID (afterany:$TRAIN_ID) -> scratch/screen-$NAME.txt"
