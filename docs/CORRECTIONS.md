# CORRECTIONS.md

*Lightweight log of small corrections that don't rise to ADR level. Active entries inform Claude Code's behaviour from session start; durable corrections are eventually promoted into `CLAUDE.md` or `PRINCIPLES.md`. See ADR-0007 for the rationale behind this mechanism.*

---

## Format

Append-only log, one entry per line:

```
- YYYY-MM-DD | status | one-line correction
```

**Statuses**:

- `active` — currently informs Claude Code's behaviour; read at session start.
- `resolved` — no longer applies; retained for history.
- `promoted` — distilled into `CLAUDE.md` or `PRINCIPLES.md` as a durable rule; retained for history.

## Workflow

- When the human corrects something that could plausibly recur, Claude Code asks: *"should I log this to `CORRECTIONS.md`?"*. On confirmation, an entry is added before continuing.
- The human may also add corrections directly.
- At session start, Claude Code reads all `active` entries.
- Every few weeks, a distillation pass: durable corrections are promoted; resolved ones are marked; one-offs are deleted.

---

## Entries

- 2026-06-29 | active | Don't run recursive filesystem scans (`find`, `Get-ChildItem -Recurse`, broad globs) over the OneDrive `../data` tree — they force OneDrive to hydrate/download cloud-only files (d3plots are 100+ MB each). Access only specific, known paths.
- 2026-06-29 | promoted | Ingestion keeps the solver's full tensor component count (6-component Voigt stress/strain) even for 2D cases — extract-everything (ADR-0016 §4) overrides ADR-0012's "4 in 2D" prose. *(2026-07-06: reconciled — ADR-0012's tensor-component line and ARCHITECTURE.md's schema section now both record the 6-component-verbatim rule; the durable statement lives there.)*
- 2026-07-03 | active | Physics-quantity figures (von Mises, plastic strain, …) always follow FEM-postprocessor conventions — jet fringe, labelled levels, working-frame units, per the README rollout GIF — never generic scientific styling. Render via `structbench.viz` (ADR-0022), don't restyle inline.
- 2026-07-03 | active | "Harness" in requests means the project's behavioral harness (rules in HARNESS.md/CORRECTIONS.md/CLAUDE.md) unless code is explicitly meant — confirm scope before building modules; prefer the smallest artifact that encodes the behavior.
- 2026-07-03 | promoted | Git operations on protected state (merge to main, push, branch deletion) execute on explicit in-session human confirmation - formalized as ADR-0023 (git authority on instruction, amends 0006).
- 2026-07-06 | active | CGN uses **small directional neighbourhoods** and relies on message-passing steps for range — never a large radius. **Convention: `connectivity_radius` = 2-3× the particle spacing** (Taylor & notch use 3× ≈ ~28 neighbours; wave ~2.4×), which keeps the physical degree low. `max_neighbors` is a **project-wide backstop cap of 32** (all configs + `CGNConfig` default), sized above that degree so it does not truncate — this resolves review finding M-B: when the cap never binds, the send→receiver truncation direction is moot. Recipe values live in config, not an ADR (ADR-0028's 2026-07-05 maintainer note). (Notch `connectivity_radius` was corrected 15→7.5 here; the 15 was stale from the old repo.)
- 2026-07-07 | active | Never write files to the repo root — job stdout, temp/analysis outputs, and scratch go in a subdir, not root. SLURM `--output` → `scratch/logs/` (under the gitignored `scratch/`; the root-level `logs/` dir was removed 2026-08-17); scratch/one-off work → `scratch/`. The root holds only tracked project files.
- 2026-07-07 | active | Corrects the 2026-07-06 `max_neighbors` note above: Taylor's radius-1.5 degree reaches **~53** in the compressed mushroom (measured across the 200/190 m/s cases; ~28 only undeformed), so the project-wide cap of **32 DOES truncate for Taylor** — the "sized above the degree / cap never binds" claim was undeformed-only. Kept at 32 regardless: truncation is benign because range comes from the 10 message-passing hops, not the 1-hop cap (maintainer, 2026-07-07), so the send→receiver asymmetry under truncation is accepted as negligible. `configs/taylor_impact_2d/cgn.toml` comment corrected to match.
- 2026-07-09 | resolved | Public dataset hosting is deferred with no near-term plan — not a v0.1/v0.2 release gate. Benchmarks ship code + protocol + baseline; OneDrive stays the private master; the Zenodo direction (one record per benchmark, on the ADR-0031 archive layout) is parked under README Roadmap → Later. The README "Data availability" paragraph still reads "Hosting is being finalised for the v0.1 release" — knowingly left as-is for now (maintainer, 2026-07-09). *(2026-08-06: settled by ADR-0040 — OneDrive stays the master, archives shared on request, Zenodo dropped; README wording updated.)*
- 2026-07-10 | active | DUG GPU training: keep **≥2 single-GPU jobs running concurrently** (seeds, ablation arms, other benchmarks) — one lone job wastes the 2×A100 nodes; the training loop is single-GPU by design, so parallelism comes from concurrent jobs (maintainer, 2026-07-10).
- 2026-07-10 | active | **Avoid seed 0 for training runs**; use seeds ≥1 (maintainer, 2026-07-10). In the wave-1D round-1 fleet, seed 0 was worst on every metric and its selected checkpoint was a single lucky val eval at step 8k (best-of-fleet was seed 1; Taylor's blessed seed was also s1). Fleet seed sets start at 1, e.g. {1,2} pairs for ablation arms, {1,2,3,4} for baseline fleets.
- 2026-07-20 | active | In notch-beam case IDs (`NB-I-320-Sphere-a-120`, …) the number after the benchmark tag is the **beam span in mm** (spans {320,480,640} per the benchmark card), not a frame or rollout count — all notch cases have `n_frames=502` uniformly. (Was misread in-session as "320-step rollout", 2026-07-20.)
- 2026-08-12 | active | Research-strategy content (hypotheses, pre-registrations, study designs, conference/journal plans) never enters the public repo — `docs/plans/` is for platform implementation plans only; durable strategy lives in the maintainer's vault research repo, dated operational plans and session records in `scratch/`. (Origin: EMI26 study plan briefly pushed public 2026-08-12, removed same day; remains in git history.)
- 2026-08-17 | active | **`noise_std` must be expressed in the model's WORKING frame, not the data-native frame** — when a benchmark loads with `length_scale ≠ 1` (DeformingPlate: `load_case_trajectory` scales positions ×1000, m→mm), the position noise must be scaled by the same factor. The DP baselines (blessed MGN + AR Transolver/GeoFLARE) trained with `noise_std=0.003` — the paper's native-frame `3e-3` copied UNSCALED into the mm frame, so ~1000× too weak (faithful value is `3.0` mm). The tell: `world_edge_radius` WAS scaled (0.03 m→30 mm) in the same config but the noise was not. Symptom = the under-noising signature: MGN one-step 4× *better* than the paper (0.059 vs 0.25 mm) but rollout at the *high* (worse) edge of the band (16.98 vs 15.1 mm pooled). No `noise_std × length_scale` factor exists anywhere in the pipeline (`train.py:237`). Retraining all DP baselines with `noise_std=3.0`; new DP AR runs (CGN + native-scheme ablations) use the working-frame value from the start. Recorded in ADR-0043 dated note.
