# WORKFLOW.md

*Multi-machine session venues and git workflow. Read at session start; identify
your venue before making any change.*

---

## Why this exists

The project is worked on by multiple Claude Code sessions (and the human) from
different places at once. A git branch checkout is a property of a *folder*,
not of a session: two writers in one folder silently overwrite each other, and
on DUG the execution folder is also what SLURM jobs run. Venue separation makes
those conflicts structurally impossible instead of a matter of discipline.

## Venues

Identify the venue at session start and adopt its role:

| Venue | Detect by | Folder | Role |
|---|---|---|---|
| **Operations** | DUG login node: hostname `prud*`, no `SINGULARITY_NAME` in env | `/data/curtin_eecms/curtin_qilin/structbench` (*execution checkout*) | Submit/monitor SLURM jobs, merge/push on instruction, summarize results. **Never edit code here.** |
| **Debug** | JupyterHub container: `SINGULARITY_NAME=jupyterlab.sif` | `/data/curtin_eecms/curtin_qilin/structbench-dev` (git worktree of the same repo) | Interactive GPU debugging and small fixes, on feature branches created here; commit and merge promptly. |
| **Development** | Windows or any other separate clone | that clone | Major feature work on feature branches, synced through GitHub. |

## The execution checkout is special

`structbench` is the tree SLURM jobs execute. It moves only by deliberate
`git merge` / `git pull --ff-only` **between** job fleets — never while jobs
are queued or running, and never by editing files in place. Every training run
records its commit (`config.json`, fleet `MANIFEST.tsv`), which is the link
between gitignored `runs/` artifacts and code history.

## Git rules (all venues)

- **One branch has one writer at a time.** Parallel work streams get parallel
  branches, even when the same human drives both sessions. (The two DUG
  folders are worktrees of one repository, so git itself refuses to check the
  same branch out in both.)
- **Sync ritual:** `git pull --ff-only` before starting work in a separate
  clone; push your branch when stopping. Between the two DUG worktrees no push
  is needed — they share one history.
- **Code moves between machines only through git.** Never copy files sideways
  (a stale CRLF snapshot from exactly that mistake cost a day; on Windows set
  `git config core.autocrlf input`).
- **`main` moves only on the human's explicit in-session instruction**
  (authority tiers in CLAUDE.md, ADR-0023), and release actions remain fully
  human (forbidden tier).
- Branches are short-lived: merge within days, delete after merging.
