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
- 2026-06-29 | active | Ingestion keeps the solver's full tensor component count (6-component Voigt stress/strain) even for 2D cases — extract-everything (ADR-0016 §4) overrides ADR-0012's "4 in 2D" prose. Reconcile the ADR-0012 text in a later distillation pass.
