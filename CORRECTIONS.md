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

*(none yet)*
