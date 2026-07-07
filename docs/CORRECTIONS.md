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
- 2026-07-07 | active | Never write files to the repo root — job stdout, temp/analysis outputs, and scratch go in a subdir, not root. SLURM `--output` → `logs/` (gitignored, dir tracked via `.gitkeep`); scratch/one-off work → `scratch/`. The root holds only tracked project files.
