# Running CGN baselines on DUG McCloud

Taylor 2D is the worked example below; the **Wave-1D** deltas (data, config,
batch script) are in the [last section](#wave-1d-baseline-v02). DUG uses **SLURM**. Access is SSH (with a JupyterLab-over-localhost option).
GPU nodes offer 4×V100 or 2×A100 80 GB. The training loop is **single-GPU**, so
we request **one** A100 80 GB — the full `configs/taylor_impact_2d/cgn.toml` (batch 8,
~50–60 GB) fits on the 80 GB card (batch 16 OOMs). The extra GPUs would sit idle
unless the model is extended with DistributedDataParallel (a separate change).

`train_taylor.slurm` carries live values, re-verified 2026-07-10: partition
`curtin_eecms` (`TIMELIMIT infinite`; now ~21 nodes with `gpu:a100:2` plus
v100:4 nodes — the 2026-07-03 "one shared A100 node" note is obsolete, and
batch GPU jobs generally start within seconds), `gpu:a100:1`, data at
`/data/curtin_eecms/curtin_qilin/data/taylor_impact`. Three operational
facts: **`sbatch` only works from a login node** (the JupyterHub container has
no munge socket); a freshly submitted job can transiently show
`ReqNodeNotAvail` with an alarming everything-unavailable node list for one
scheduler cycle (~1 min) before starting — don't diagnose it early; and a
big-memory JupyterHub session (`mem=1019000M`) monopolizes its node's RAM, so
no batch job co-resides with one — budget a node per live session when
planning fleets.

## 1. Copy code + data up (from your Windows machine, Git Bash)

```bash
REPO="<path-to-local>/StructBench"
DATA="<path-to-local>/data/StructBench/canonical/taylor_impact_2d"

rsync -avP "$REPO/" <user>@<dug-host>:<proj>/structbench/ \
  --exclude '.venv' --exclude 'runs' --exclude '.git' \
  --exclude '__pycache__' --exclude 'scratch'
rsync -avP "$DATA/" <user>@<dug-host>:<proj>/data/taylor_impact/    # 2.4 GB, 34 files (the path train_taylor.slurm reads)
```

For the data, `rclone` on the DUG side is the alternative that avoids hydrating
the OneDrive files locally first — with a OneDrive remote configured there:

```bash
rclone copy onedrive:"<path-to>/data/StructBench/canonical/taylor_impact_2d" \
  <proj>/data/taylor_impact --progress --transfers=8
```

## 2. Build the env (once, on the DUG login node)

```bash
cd <proj>/structbench
bash hpc/dug/setup_env.sh
```

Only `torch` + `torch_geometric.nn.MessagePassing` are needed — no compiled
`torch-cluster`/`pyg-lib` — so this is a clean install (native `radius_graph`,
ADR-0020).

## 3. Smoke-check on a GPU node first (~5 min), then the full run

```bash
# quick sanity that CUDA + the pipeline work on DUG's GPU, using the tiny config:
srun --partition=curtin_eecms --gres=gpu:a100:1 --time=00:15:00 --pty bash -lc '
  source .venv/bin/activate && export PYTHONPATH=src
  python -m structbench.cli.train --mode train --config configs/taylor_impact_2d/cgn_smoke.toml \
    --data-root /data/curtin_eecms/curtin_qilin/data/taylor_impact --out runs/smoke'

# full baseline as a batch job (from a login node; OUT defaults to
# runs/taylor-cgn-v01 and must be fresh per attempt):
sbatch hpc/dug/train_taylor.slurm
squeue --me                     # watch the queue
tail -f logs/slurm-taylor-*.out # progress: val_pos (mm) / val_aux (MPa) each val_every
```

The full run is 80k steps at batch 8 (ADR-0028 reference recipe, radius 1.5).
Throughput was **~4.6k steps/h → ~22 h** when last measured (2026-07-03), but
that was at the earlier batch-32/100k recipe — re-measure for batch 8. Either
way the run stays inside the 36 h ceiling in the script (the partition's time
limit is `infinite`). The
best checkpoint is written whenever the validation **position** RMSE improves
(ADR-0028; checked every 2000 steps), so an early stop still leaves a usable
model — but **training has no resume**, so a killed run restarts from scratch.

Three rules the entry point now enforces / expects:

- **Fresh `--out` per training attempt.** `--mode train` refuses a directory
  that already holds `model-*.pt` or `ckpt-*.pt` (eval picks the highest-step
  `model-*.pt`, so a from-scratch rerun would shadow the better earlier model).
- **If train is killed by walltime, the eval steps never ran** (`set -e`).
  Rerun just the two eval lines from the slurm script against the same
  `--out` — they rebuild the model from the run's own `config.json`.
- **Periodic snapshots are analysis-only.** Training also writes `ckpt-<step>.pt`
  every 10k steps (ADR-0028, 2026-07-10 note). The default eval ignores them;
  score one explicitly with `--mode valid --checkpoint ckpt-050000.pt` — its
  metrics land in `metrics-val@ckpt-050000.json` (rollout `.npz` skipped), so
  the selected checkpoint's canonical artifacts are never overwritten.

## 4. Bring results back

```bash
rsync -avP <user>@<dug-host>:<proj>/structbench/runs/taylor-cgn-v01/ ./runs/taylor-cgn-v01/
# -> model-best-*.pt, config.json, normalization_stats.npz,
#    metrics-val.json / metrics-test_interp.json / metrics-test_extrap.json
#    (ADR-0019 per-case + split-mean metrics), and rollouts/*.npz (predicted
#    trajectories). The slurm log carries the split means and per-case RMSEs;
#    per-case QoI values/errors are only in the metrics JSON.
```

Use the JupyterLab-over-localhost option for interactive inspection — load a
checkpoint and plot rollouts — rather than for the long training run itself.

## 5. Bless and archive

When the maintainer blesses a run, the full mechanical checklist lives here
(ADR-0033 defines what blessing *is*; ADR-0037 adds the archive steps):

1. **Rewrite the grouped config** `configs/<benchmark>/<family>.toml` to
   match the blessed run's `config.toml` — seed included. The committed
   config at HEAD *is* the blessed recipe; the landing page's "Reproducing
   the baseline" section sends users to it, so any drift (a fleet-override
   winner, a different seed) breaks reproduction silently.
2. **Transcribe** the run's `metrics-*.json` split means into a
   `BaselineResult` in the benchmark's `RESULTS` registry (ADR-0033), with
   the run commit and date from its `config.json` / fleet `MANIFEST.tsv`.
3. **Regenerate the views**: `python tools/gen_benchmark_docs.py`.
4. **Assemble the bundle** in a `scratch/` staging folder (ADR-0037): the
   blessed run directory verbatim minus editor droppings (hidden dirs such
   as `.ipynb_checkpoints/`), plus `provenance/` holding the fleet ledger
   and selection record (MANIFEST/SUMMARY/REPORT, disambiguated names when
   several MANIFESTs apply):

   ```bash
   B=scratch/models-staging/<benchmark>/<family>-<run_commit>
   rsync -a --exclude='.*' runs/<blessed-run>/ "$B/"
   mkdir "$B/provenance"   # then cp the fleet MANIFEST/SUMMARY/REPORT in
   ```

5. **Checksum** from the bundle root:

   ```bash
   (cd "$B" && find . -type f ! -name SHA256SUMS -print0 | sort -z \
      | xargs -0 sha256sum > SHA256SUMS)
   ```

6. **Sync to the OneDrive private master** — the `models/` mirror beside
   `canonical/` and `raw/` (adjust remote name/path to your rclone config):

   ```bash
   rclone copy scratch/models-staging/<benchmark> \
     onedrive:"<path-to>/data/StructBench/models/<benchmark>" --progress
   ```

7. **Point the registry at it**: set `checkpoint=` (archive-relative path,
   `models/<benchmark>/<family>-<commit>/<file>.pt`) and
   `checkpoint_sha256=` on the blessed entry, then regenerate the views
   again (the pointer renders with a private-archive marker until it is a
   public URL).

---

## Wave-1D baseline (v0.2)

Same machine, env build (§2), and result bring-back (§4) as Taylor — only the
data, config, and batch script differ. Wave is a much smaller model (hidden 64,
5 MP hops, ~500–1250 particles/case vs Taylor's 128/10), so it runs with wide
headroom; `train_wave.slurm` keeps the A100 request for a guaranteed fit, but
you can drop to `--gres=gpu:v100:1` and trim `--mem`/`--time` after the smoke
test measures the real footprint.

**1. Copy the data up** — rclone on the DUG login node, straight from a OneDrive
remote (0.23 GB, 16 cases; avoids hydrating the cloud files locally). Adjust the
`onedrive:` remote name and source path to your rclone config:

```bash
rclone copy \
  onedrive:"research/civil_engineering/data/StructBench/canonical/wave_propagation_1d" \
  /data/curtin_eecms/curtin_qilin/data/wave_propagation_1d \
  --progress --transfers=8
# sanity: the 16 .h5 cases + generated README.md / card.json
rclone ls /data/curtin_eecms/curtin_qilin/data/wave_propagation_1d | wc -l
```

The dest uses the full archive name `wave_propagation_1d`, not the truncated
form Taylor's dir (`data/taylor_impact`) still carries.

**2. Smoke-check on a GPU node (~2 min), then submit the full run:**

```bash
srun --partition=curtin_eecms --gres=gpu:a100:1 --time=00:10:00 --pty bash -lc '
  source .venv/bin/activate && export PYTHONPATH=src
  python -m structbench.cli.train --mode train --config configs/wave_propagation_1d/cgn_smoke.toml \
    --data-root /data/curtin_eecms/curtin_qilin/data/wave_propagation_1d --out runs/wave-smoke'

sbatch hpc/dug/train_wave.slurm   # OUT defaults to runs/wave-cgn-v02
squeue --me
tail -f logs/slurm-wave-*.out     # val_pos (mm) / val_aux (MPa) each val_every
```

The full run is 50k steps at batch 32 (half the Taylor budget; ADR-0028 recipe
at wave capacity). Eval covers `val` and `test_interp` only — wave has no
extrapolation split (ADR-0025) — landing as `metrics-val.json` /
`metrics-test_interp.json` + `rollouts/*.npz`. Bring the run back as in §4,
swapping the run dir for `runs/wave-cgn-v02`.

Measured throughput (2026-07-10, A100, hidden 64 / 5 MP / batch 32):
**~40k steps/h — 50k steps train in ~65 min, whole job ~1 h 10 m** including
both eval passes. The round-1 4-seed fleet (`runs/wave-cgn-v02*`) ran at this
rate; budget ~2.5 h for a 100k-step run at this capacity.

**Round-2 recipe fleet**: `ablate_wave.slurm` runs one ablation arm per job
(same override mechanism as `ablate_taylor.slurm`; STEPS defaults to 100000,
walltime 24 h). Arms, seeds, and the pre-registered analysis protocol live in
the maintainer's round-2 proposal; smoke-test cap-128 memory on a worst-case
composed batch before submitting capacity arms, and avoid seed 0
(CORRECTIONS 2026-07-10).

---

## Notch-impact baseline (v0.2)

Same machine, env build (§2), result bring-back (§4), and bless flow (§5) as
the others. The dataset is already staged and rclone-verified on DUG:
`/data/curtin_eecms/curtin_qilin/data/notch_beam_2d_impact` (113 files,
25 GB). Notch beams (H80 × span {320, 480, 640} mm at 2.5 mm spacing) carry
several times Taylor's particle count, so expect Taylor-like or slower
steps/h at the capacity cell and a small memory-bound batch — smoke-test
memory and throughput before any fleet (30-step capacity smoke against a
`scratch/` config; the first run also builds the dataset's `derived/`
normalization cache, a one-off ~25 GB read).

**Round-1 recipe fleet (recipe first, seeds later)**: the base
`configs/notch_beam_2d_impact/cgn.toml` still carries the pre-round-1 small
recipe (64/5/1, seed 0), so round-1 arms pass the capacity cell and seed
explicitly via `ablate_notch_impact.slurm` — one arm per job, noise_std as
the primary sweep axis, everything at `SEED=1`:

```bash
FLEET=runs/fleet-$(date +%F)
sbatch --job-name=ni-n002-s1 \
  --export=ALL,NAME=n002-s1,HIDDEN=128,MP=10,NMLP=2,BATCH=8,SEED=1,NOISE=0.02,STEPS=50000,FLEET=$FLEET \
  hpc/dug/ablate_notch_impact.slurm
```

Rank arms on val rollout metrics at the reduced step budget; the winner gets
the full budget (plus a seed fleet if the maintainer wants spread evidence)
and rewrites `cgn.toml` at bless time — the wave round-2 precedent. Keep ≥2
jobs running concurrently (CORRECTIONS 2026-07-10) and avoid seed 0.
