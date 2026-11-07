#!/bin/bash
# Generic single-GPU arm launcher: the config FILE is the arm (ADR-0032
# discipline; no override knobs). Works for any benchmark — resolves
# configs/$BENCH/$ARM.toml. Submit from the repo root ON A LOGIN NODE, one job
# per arm, e.g.:
#
#   mkdir -p scratch/logs
#   sbatch --job-name=os-taylor-s1 \
#       --export=ALL,BENCH=taylor_impact_2d,ARM=transolver-oneshot-s1,\
#OUT=runs/oneshot-taylor-transolver-s1,DATA=/data/curtin_eecms/curtin_qilin/data/taylor_impact \
#       tools/hpc/dug/train_arm.slurm
#
# A benchmark loaded by path (CONFIG=) belongs to the tree that owns its
# config, so OUT is required and absolute, and the slurm log is sent beside
# the run with --output= instead of into scratch/logs/:
#
#   sbatch --job-name=<arm> --output=/abs/path/to/<study>/runs/slurm-%x-%j.out \
#       --export=ALL,CONFIG=/abs/path/to/<study>/configs/<arm>.toml,\
#OUT=/abs/path/to/<study>/runs/<arm>,DATA=/abs/path/to/<study>/data/canonical \
#       tools/hpc/dug/train_arm.slurm
#
# Recognised variables:
#   BENCH  benchmark dir under configs/            (required unless CONFIG)
#   ARM    config stem under configs/$BENCH/       (required unless CONFIG)
#   CONFIG path to the arm's config file, for a benchmark loaded by path
#          (ADR-0032 note, 2026-11-06) whose configs live outside this
#          checkout; BENCH/ARM are then optional (ARM defaults to the stem)
#   DATA   canonical data root                     (required)
#   OUT    output dir: default runs/$ARM under the checkout for BENCH/ARM;
#          required and absolute with CONFIG (the job exits otherwise)
#
# Every finished run ends with MANIFEST.sha256 in $OUT: every file under the
# run with its SHA-256 (sha256sum format, relative paths, sorted, the manifest
# itself excluded), so `sha256sum -c MANIFEST.sha256` verifies a copy of the
# run on another machine.
#
# ONE A100 80G per job; nodes carry gpu:a100:2 so jobs pack two-per-node
# automatically (no exclusive flags). 2-day ceiling covers the notch 250k /
# one-shot budgets; override --time / --mem via sbatch if needed.

#SBATCH --partition=curtin_eecms
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=scratch/logs/slurm-%x-%j.out       # scratch/logs/ must exist BEFORE sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

: "${DATA:?set DATA=<canonical data root>}"
if [ -n "${CONFIG:-}" ]; then
    # A config by path: the arm is the file's stem unless ARM says otherwise,
    # and its outputs belong to the tree that owns the config, never to this
    # checkout, so OUT must be given and absolute.
    ARM=${ARM:-$(basename "${CONFIG%.toml}")}
    BENCH=${BENCH:-by-path}
    case "${OUT:-}" in
        /*) ;;
        *) echo "CONFIG= needs an absolute OUT outside this checkout (got OUT='${OUT:-}')" >&2; exit 1 ;;
    esac
else
    : "${BENCH:?set BENCH=<benchmark dir under configs/> (or CONFIG=<path>)}"
    : "${ARM:?set ARM=<config stem under configs/$BENCH/>}"
    CONFIG="configs/${BENCH}/${ARM}.toml"
fi
[ -f "$CONFIG" ] || { echo "no such arm config: $CONFIG" >&2; exit 1; }

OUT=${OUT:-runs/${ARM}}
mkdir -p "$OUT"
source .venv/bin/activate
export PYTHONPATH=src

echo "code commit: $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "bench: $BENCH  arm: $ARM  config: $CONFIG  out: $OUT  data: $DATA"
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# train -> checkpoints + config.json (self-contained; normalizer buffers in
# state_dict). eval rebuilds the architecture from the run's config.json, so
# the two eval lines are re-runnable standalone if train is killed (set -e).
python -m structbench.cli.train --mode train   --config "$CONFIG" --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/train.log"
python -m structbench.cli.train --mode valid   --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/eval.log"
python -m structbench.cli.train --mode rollout --data-root "$DATA" --out "$OUT" 2>&1 | tee -a "$OUT/eval.log"

# The run's manifest, written last so it covers the logs above: sha256sum
# lines for every file under $OUT, relative paths, sorted, the manifest itself
# (and its temporary) excluded. Verify a copy with `sha256sum -c MANIFEST.sha256`
# from inside the run.
(
    cd "$OUT"
    find . -type f ! -name 'MANIFEST.*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > MANIFEST.tmp
    mv -f MANIFEST.tmp MANIFEST.sha256
)
echo "manifest: $(wc -l < "$OUT/MANIFEST.sha256" | tr -d ' ') files in $OUT/MANIFEST.sha256"
