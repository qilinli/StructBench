# 0037 — Blessed runs archive: `models/` mirror and registry checkpoint pointers (amends 0031, 0033)

**Status**: Accepted
**Type**: Durable
**Date**: 2026-07-12

## Context

Both trained baselines are now blessed (Taylor: run 2026-07-08, commit
`7be9d4b`, blessed 2026-07-09; wave: run and bless 2026-07-10, commit
`48046ea`), and the README's
reproducibility contract names the run directory as "the complete,
portable evidence for its numbers". Yet those two run directories —
`runs/taylor-cgn-baseline/s1` (189 MB) and `runs/fleet-2026-07-10/x1-s1`
(105 MB) — exist as exactly one copy each, in the gitignored `runs/` tree
of the DUG execution checkout. No document prescribes preserving them:
ADR-0031's archive tree covers datasets only, the ADR-0033 bless
procedure ends at transcription + regeneration, `BaselineResult`'s
forward-looking `checkpoint` field is unset for both entries, and no
checksum of any checkpoint is recorded anywhere. Training has no resume
(DUG runbook), so losing a blessed run dir means re-running a GPU fleet —
and, until then, published numbers with no artifact behind them. Part of
the fleet provenance (the wave round-1 ledger and round-2 selection
report) sits in `scratch/`, which is ephemeral by definition.

VISION.md promises "reference models with released checkpoints" and the
v0.2 roadmap requires "checkpoint + metrics each"; public hosting and the
checkpoint-publishing workflow are parked (2026-07-09 correction,
README Roadmap → Later). Preservation cannot wait on publication.

## Decision

1. **A third mirror in the ADR-0031 tree**: `models/<benchmark>/` under
   `../data/StructBench/`, beside `canonical/` and `raw/`. One bundle
   folder per blessed registry entry, named `<family>-<run_commit>/`
   (e.g. `models/wave_propagation_1d/cgn-48046ea/`).
2. **Bundle = the blessed run directory** (checkpoints,
   `config.json`/`.toml`, `normalization_stats.npz`, `metrics-*.json`,
   `job-info.txt`, logs, `rollouts/` — the evidence unit the README
   contract names), copied verbatim minus editor droppings
   (`.ipynb_checkpoints/` and other hidden dirs), plus a `provenance/`
   subfolder holding the fleet-level record (fleet `MANIFEST.tsv`,
   selection `SUMMARY.md`/`REPORT.md`, proposal notes), plus a
   `SHA256SUMS` file at the bundle root covering every bundled file.
   Sibling seeds are not archived; the fleet spread they establish is
   recorded in the provenance documents.
3. **OneDrive stays the private master**, exactly as for datasets:
   bundles are assembled on the training machine and synced to the
   OneDrive `models/` mirror. This amends ADR-0031's aside that
   "only `canonical/` is ever uploaded": when the parked
   checkpoint-publishing workflow unparks, `models/` bundles become
   uploadable archives too; `raw/` remains private-only.
4. **The registry points at the archive**: `BaselineResult.checkpoint`
   holds the archive-relative path of the blessed checkpoint
   (e.g. `models/taylor_impact_2d/cgn-7be9d4b/model-best-096000.pt`),
   rewritten to a public URL if and when checkpoints publish. This
   amends the field's documented contract ("pointer/URL to the
   published checkpoint"): archive-relative path into the private
   master until publication, public URL after — the `results.py`
   docstring changes with it. A new optional field `checkpoint_sha256`
   records the file's digest, making the pointer verifiable; when set
   it must be a 64-hex digest and requires `checkpoint` to be set.
   (Both are public-API changes, flagged by this ADR.) The two
   generated views — archive README and ADR-0036 landing page — render
   the pointer with an explicit private-archive marker until it is a
   public URL, honest rather than silent (ADR-0033). Both blessed
   entries gain pointer + digest; generated views regenerate.
5. **Blessing gains one final mechanical step, with one operational
   home**: `hpc/dug/README.md` gains a blessing-and-archiving section
   holding the full checklist (rewrite the grouped config to the blessed
   recipe → transcribe metrics → regenerate docs → assemble the bundle
   in a `scratch/` staging folder → write `SHA256SUMS` → `rclone copy`
   to the OneDrive `models/` mirror → set registry pointer + digest).
   The config-rewrite step (added 2026-07-12) keeps the committed
   grouped config equal to the blessed recipe verbatim — seed included —
   the invariant the landing pages' reproduction guidance relies on.
   ADR-0033 defines what blessing *is*; the runbook holds the *how*.

## Alternatives considered

- **Track checkpoints in git (plain or LFS)**: the blessed checkpoints
  are only ~6.4 MB each, but the evidence unit is the run directory
  (hundreds of MB, growing per release); LFS adds a dev dependency and
  couples artifact storage to GitHub hosting. Binaries stay out of the
  repo; pointers go in.
- **GitHub Releases as the checkpoint home**: release actions are the
  forbidden tier (deliberate human action) and would couple preservation
  to publication timing. Releases may still become the *publication*
  channel later; this ADR only secures the master copy.
- **Wait for Zenodo/dataset hosting**: hosting is parked with no
  near-term plan, which leaves the single-copy risk open indefinitely.
- **Archive an "essential subset" (checkpoint + config + stats only,
  <10 MB)**: reopens "which files count as evidence" case by case; the
  README contract already answers it — the run directory — and the full
  bundles are small enough (~300 MB total today) that trimming buys
  nothing.

## Consequences

- Preservation is decoupled from the parked publication question; when
  hosting unparks, publishing a checkpoint becomes "upload the existing
  bundle and rewrite the pointer" — the same move ADR-0031 set up for
  datasets.
- `BaselineResult` gains `checkpoint_sha256` and the `checkpoint`
  contract changes (public-API changes, flagged here); both generated
  views show the marked pointer they previously suppressed for `None`.
- The OneDrive master grows by ~300 MB now; each future blessed baseline
  adds its bundle (the notch pair is next, in v0.2).
- Backfill is the first act under this ADR: assemble
  `models/taylor_impact_2d/cgn-7be9d4b/` from
  `runs/taylor-cgn-baseline/s1` (+ that fleet's `MANIFEST.tsv` and
  `SUMMARY.md`) and `models/wave_propagation_1d/cgn-48046ea/` from
  `runs/fleet-2026-07-10/x1-s1` (+ `runs/fleet-2026-07-10/MANIFEST.tsv`,
  `scratch/wave-fleet/MANIFEST.tsv`,
  `scratch/wave-fleet/round2-proposal.md`, and
  `scratch/wave-fleet/round2-analysis/REPORT.md`), rescuing the wave
  provenance out of `scratch/`.
- Non-blessed fleet runs remain unarchived and deletable; the archive
  boundary — blessed entry in, siblings out — is now explicit.
