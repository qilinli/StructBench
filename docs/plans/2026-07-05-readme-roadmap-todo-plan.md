# README Roadmap Todo List Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire ROADMAP.md into a living, repo-verified todo list rendered as the `## Roadmap` section of README.md, with every cross-reference repointed.

**Architecture:** Docs-only, three commits in dependency order: (1) README gains the new section while ROADMAP.md still exists, (2) all references repoint to the README section, (3) ROADMAP.md is deleted and its publication note rehomes to docs/WORKFLOW.md — so no commit ever contains a dangling reference.

**Tech Stack:** Markdown (GitHub-flavored: task-list checkboxes + strikethrough), git.

**Spec:** `docs/plans/2026-07-05-readme-roadmap-todo-design.md` (approved 2026-07-05).

## Global Constraints

- Docs-only: no file under `src/`, `tests/`, `configs/`, or `tools/` is touched.
- Historical documents are **not** edited: `decisions/0001-*.md`, `decisions/0021-*.md`, `decisions/0024-*.md`, `docs/plans/2026-07-03-*.md` keep their ROADMAP.md mentions as records of their time (ADR-0009 precedent).
- Work on branch `docs/readme-roadmap-todo`; never commit to `main` (ADR-0023).
- Commits follow Conventional Commits with the agent trailer (docs/PRINCIPLES.md).
- Wrap Markdown near 80 columns, matching the files being edited.
- All verification commands run from the repo root; the dev interpreter is `.venv\Scripts\python.exe` (Windows clone).

---

### Task 1: README.md — the living Roadmap section

**Files:**
- Modify: `README.md` (status-blurb link line 12; `## Roadmap` section lines 173–183; stale ADR counts lines 170 and 188)

**Interfaces:**
- Produces: a `## Roadmap` heading in README.md (GitHub anchor `#roadmap`) that Task 2's repointed references and this plan's grep checks rely on.

- [ ] **Step 1: Replace the stale `## Roadmap` section body**

In `README.md`, replace exactly this block:

```markdown
## Roadmap

- **v0.1** — Taylor 2D end to end: data, baseline checkpoint, recorded
  metrics, public release.
- **v0.2+** — RC beam and segmented-beam benchmarks (data exists from prior
  published work); a multi-scale GNS second baseline (spec drafted); plastic
  strain as a second target.
- **Later** — more solver adapters, larger-scale graph backends, multi-modal
  SHM benchmarks, deployment tooling.

Sequencing in [ROADMAP.md](ROADMAP.md); reasoning in [`decisions/`](decisions/).
```

with:

```markdown
## Roadmap

<!-- Living todo list (the single planning home; ROADMAP.md is retired).
     Conventions: done = [x] + strikethrough + (date); ad-hoc additions land
     in Inbox and get triaged into a milestone; when a milestone ships, its
     crossed-out block may be compressed to one line. Reasoning lives in
     decisions/, not here. Substrate-layer work only (ADR-0014). -->

*Last revised: 2026-07-05.*

### v0.1 — Taylor 2D substrate proof

- [x] ~~Canonical case format + round-trip-tested HDF5 I/O (ADR-0011..0013)~~
- [x] ~~General LS-DYNA adapter on lasso-python (ADR-0016)~~
- [x] ~~Taylor 2D benchmark: fixed split, eval protocol, QoIs (ADR-0019)~~
- [x] ~~Config-driven pipeline `structbench-train` (train/valid/rollout)~~
- [x] ~~`radius_graph` batch-partition fix: 50.9 s → 0.22 s per batch~~ (2026-07-02)
- [x] ~~Public GitHub repository~~ (2026-07-02)
- [x] ~~First full baseline run → training-recipe rework (ADR-0028)~~ (2026-07-03)
- [ ] Trained GNS baseline with the ADR-0028 recipe (DUG A100; SSH-side
      steps are human-gated)
  - [ ] full retrain (~⅓ of the first run's 14k steps/h — plan walltime
        accordingly)
  - [ ] checkpoint + recorded ADR-0019 metrics
- [ ] Release: baseline metrics into the README table, prediction-vs-truth
      hero GIF, dataset link, version tag (human action)

### v0.2 — wave-1d + notch-beam pair

- [x] ~~Ingestion: 16 wave runs + 221 notch-beam cases to canonical HDF5~~ (2026-07-04)
- [x] ~~Three benchmark modules: frozen splits + QoIs (ADR-0025/0026)~~ (2026-07-03)
- [x] ~~Benchmark cards + generated views (ADR-0027), Taylor retrofitted~~ (2026-07-03)
- [x] ~~Benchmark-selection registry in `structbench-train`~~ (2026-07-03)
- [x] ~~Notch aux → max principal strain; damaged→cracked fraction (ADR-0029)~~ (2026-07-04)
- [ ] ADR-0030 unit-fix follow-through (Concrete-Beam decks are kg-mm-ms)
  - [ ] write + index the ADR (decision already live in `patch_units.py`)
  - [ ] confirm `patch_units.py` ran over all 237 canonical HDF5s
        (idempotent)
  - [ ] fix `SOURCE_UNITS` in `1DWavePropagation/convert.py` +
        `2DNotchBeam/convert.py`
  - [ ] correct `source_units` in the three cards; regenerate
        `docs/benchmarks.md`
- [ ] Three trained GNS baselines (checkpoint + metrics each)
  - [ ] `wave_propagation_1d`
  - [ ] `notch_beam_2d_bend`
  - [ ] `notch_beam_2d_impact`
- [ ] Validate the provisional `cracked_fraction` threshold 0.01 (ADR-0029;
      version bump if revised)
- [ ] Dataset hosting decision (Zenodo / HuggingFace / institutional) —
      gates the v0.1 release too; v0.2 is ~7× the data
- [ ] Archive packaging: measure `size_gb` per benchmark, generate
      per-archive README + card.json into the hosted archives

### Inbox — untriaged, add freely

- [ ] `lr_init` code default still 1e-3; ADR-0028's 1e-4 lives only in the
      TOML
- [ ] confirm the Taylor deck genuinely is g-mm-ms (sanity check alongside
      ADR-0030)
- [ ] reconcile ADR-0012's "4 Voigt components in 2D" prose
      (CORRECTIONS.md item)

### Later (each becomes an ADR/spec when picked up)

- **v0.3 — RC beam benchmark**: erosion, twice (numerically for the FEM
  data; structurally for the surrogate — particles vanishing mid-rollout)
- Segmented beam benchmark (parked) · MS-GNS second Taylor baseline (spec
  Proposed)
- Training: resume support (optimizer state + `--resume`) ·
  part-id→embedding remap · ADR-0028 Phase-2 ablations (noise_std, aux
  head, capacity, stress-history)
- Eval: leaderboard submission validator · cross-benchmark utilities ·
  per-region probe metrics · convergence check
- Checkpoint-publishing workflow · second aux target (effective plastic
  strain)
- Data-generation autonomy (deck templating or a Python-native solver)
- Scale: cell-list `radius_graph` backend when a ≥10⁶-node dataset lands
- Other solvers (Kratos, OpenSees, OpenRadioss) · SHM expansion ·
  deployment tools · packaging extras · PhysicsNeMo interop

Rationale for every item lives in [`decisions/`](decisions/).
```

- [ ] **Step 2: Repoint the status-blurb link**

In `README.md`, replace:

```markdown
> [roadmap](ROADMAP.md).
```

with:

```markdown
> [roadmap](#roadmap).
```

- [ ] **Step 3: Make the two stale ADR counts count-free**

In `README.md`, replace:

```
  decisions/         # 21 architecture decision records
```

with:

```
  decisions/         # architecture decision records
```

and replace:

```
explicit written harness: a decision log (21 ADRs), tiered agent authority,
```

with:

```
explicit written harness: a decision log of ADRs, tiered agent authority,
```

- [ ] **Step 4: Verify README self-consistency**

Run: `grep -n "ROADMAP" README.md`
Expected: exactly one hit — the HTML comment line `(the single planning home; ROADMAP.md is retired)`. No remaining `ROADMAP.md` links.

Run: `grep -c "^- \[x\]" README.md`
Expected: `12` (7 crossed-out v0.1 items + 5 crossed-out v0.2 items).

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: README Roadmap section is the living todo list (repo-verified status)"
```

---

### Task 2: Repoint every planning reference to README § Roadmap

**Files:**
- Modify: `CLAUDE.md` (lines 19, 42, 132)
- Modify: `decisions/0009-session-start-reading-list.md` (line 27 + revision log)
- Modify: `docs/ARCHITECTURE.md` (line 73)

**Interfaces:**
- Consumes: the `## Roadmap` heading in README.md from Task 1.
- Produces: a repo where no *live* document references ROADMAP.md, so Task 3 can delete it.

- [ ] **Step 1: CLAUDE.md — project snapshot pointer**

Replace:

```markdown
See ROADMAP.md for sequencing.
```

with:

```markdown
See the Roadmap section of README.md for sequencing.
```

- [ ] **Step 2: CLAUDE.md — conditional session-start reading item**

Replace:

```markdown
- `ROADMAP.md` — if the session is about planning or scoping.
```

with:

```markdown
- The Roadmap section of `README.md` — if the session is about planning or scoping.
```

- [ ] **Step 3: CLAUDE.md — "Where other rules live" entry**

Replace:

```markdown
- **Long-term trajectory**: `ROADMAP.md`.
```

with:

```markdown
- **Long-term trajectory**: the Roadmap section of `README.md`.
```

- [ ] **Step 4: ADR-0009 — amend the conditional-read item (Ephemeral ADR, in-place revision)**

In `decisions/0009-session-start-reading-list.md`, replace:

```markdown
- `ROADMAP.md` — if the session is about planning or scoping.
```

with:

```markdown
- The Roadmap section of `README.md` — if the session is about planning or scoping.
```

and append to the `## Revision log` list:

```markdown
- 2026-07-05 — ROADMAP.md retired; the living todo list is now the Roadmap section of `README.md` (design: `docs/plans/2026-07-05-readme-roadmap-todo-design.md`). Conditional-read item updated accordingly.
```

- [ ] **Step 5: ARCHITECTURE.md — repoint the eval/ forward reference**

In `docs/ARCHITECTURE.md`, replace:

```markdown
Metrics and evaluation protocols. Each benchmark declares its own evaluation metrics; this module implements them in a model-agnostic way. A leaderboard submission validator and cross-benchmark evaluation utilities are planned here (see ROADMAP.md) but do not exist yet.
```

with:

```markdown
Metrics and evaluation protocols. Each benchmark declares its own evaluation metrics; this module implements them in a model-agnostic way. A leaderboard submission validator and cross-benchmark evaluation utilities are planned here (see the Roadmap section of README.md) but do not exist yet.
```

- [ ] **Step 6: Verify no live document still points at ROADMAP.md**

Run: `grep -rn --exclude-dir=.git "ROADMAP" CLAUDE.md docs/ARCHITECTURE.md decisions/0009-session-start-reading-list.md`
Expected: only the ADR-0009 revision-log entry added in Step 4 (which *describes* the retirement) — no conditional-read or pointer lines.

- [ ] **Step 7: Commit**

```bash
git add CLAUDE.md decisions/0009-session-start-reading-list.md docs/ARCHITECTURE.md
git commit -m "docs: repoint planning references to README's Roadmap section"
```

---

### Task 3: Retire ROADMAP.md; rehome the publication note

**Files:**
- Delete: `ROADMAP.md`
- Modify: `docs/WORKFLOW.md` (append one section at end of file)

**Interfaces:**
- Consumes: Task 2's guarantee that no live document references ROADMAP.md.

- [ ] **Step 1: Append the publication note to docs/WORKFLOW.md**

At the end of `docs/WORKFLOW.md` (after the "Git rules (all venues)" section's last bullet), append:

```markdown

## Publication note

`RESEARCH-PROGRAM.md` (private program strategy) is untracked from
2026-07-02 onward but exists in commits before that date. If this repository
is ever made public directly, publish a fresh clean-cut repo from the
release state instead of flipping this one's visibility. *(Moved from
ROADMAP.md at its retirement, 2026-07-05.)*
```

- [ ] **Step 2: Delete ROADMAP.md**

```bash
git rm ROADMAP.md
```

- [ ] **Step 3: Full-repo reference sweep**

Run: `grep -rln --exclude-dir=.git "ROADMAP"`
Expected file list, exactly:
- `README.md` (HTML-comment convention line)
- `docs/WORKFLOW.md` (publication-note attribution)
- `decisions/0009-session-start-reading-list.md` (revision-log entry)
- `decisions/0001-adopt-harness-engineering.md`, `decisions/0021-v01-narrows-to-taylor.md`, `decisions/0024-v02-wave-and-notch-beam-scope.md` (historical, untouched)
- `docs/plans/2026-07-03-wave-1d-benchmark.md`, `docs/plans/2026-07-03-notch-beam-pair.md` (historical, untouched)
- `docs/plans/2026-07-05-readme-roadmap-todo-design.md`, `docs/plans/2026-07-05-readme-roadmap-todo-plan.md` (this change's own records)

Any other file in the list is a missed reference — fix it before committing.

- [ ] **Step 4: Commit**

```bash
git add docs/WORKFLOW.md
git commit -m "docs: retire ROADMAP.md; publication note moves to WORKFLOW.md"
```

---

### Task 4: Final verification

**Files:** none modified — checks only.

- [ ] **Step 1: Generated docs are untouched and current**

Run: `.venv\Scripts\python.exe tools\gen_benchmark_docs.py --check`
Expected: exit 0 (docs/benchmarks.md up to date — this change must not touch generated views).

- [ ] **Step 2: Test suite still green (docs-only sanity)**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: all tests pass (66+), no collection errors.

- [ ] **Step 3: Render sanity**

Open README.md in a Markdown preview (or push the branch and view on GitHub): the `#roadmap` anchor resolves from the status blurb, checkboxes render, strikethrough renders on `[x]` items, the HTML comment is invisible.

- [ ] **Step 4: Report**

Working tree clean, three commits on `docs/readme-roadmap-todo` (after the two design/plan commits). Merge to `main` and push remain on the human's explicit instruction (ADR-0023).
