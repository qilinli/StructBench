# The standard Abaqus input block

*How an Abaqus job supplies the run evidence E1–E10 of ADR-0066, so that the
reference-data verification instrument can measure it. Every generated Abaqus
input in this repository carries this block. A contributed dataset is measured
against the same requirement; this file tells a contributor what to switch on.*

**Status: partial, and deliberately so.** Its sibling
[`../lsdyna/STANDARD_INPUT_BLOCK.md`](../lsdyna/STANDARD_INPUT_BLOCK.md) cites
a claim id from a sourced dossier for every setting. This one cannot yet: the
Abaqus Keywords Reference has not been read, and ADR-0068 clause 8 defers
every keyword-level question until a sourced dossier exists.

So the file is split. **What is established** was derived by reading a real
Abaqus/Standard 2025 job's own output — a successful run, the same job
re-run batch from its `.inp`, and a deliberately rejected job — and each
statement names the file it came from. **What is not established** is listed
as open, and no keyword is recommended on recall. Nothing here is a reference
level, and ADR-0066's rule stands: a run's evidence is read, never assumed.

---

## What is established

### The job's own record: which files exist is itself evidence

A job that completes writes `.sta`, `.msg` and `.dat`. **A job the
pre-processor rejects writes only `.dat`** — no `.sta`, no `.msg`. So the
reader takes three optional texts, and a collector must not treat a missing
status file as a missing run.

| Item | File | The text the reader keys on |
|---|---|---|
| E2 solver identity | `.sta`, `.msg`, `.dat` | `Abaqus/Standard 2025` / `Abaqus 2025`; the release year is the version token |
| E3 termination, normal | `.sta` | `THE ANALYSIS HAS COMPLETED SUCCESSFULLY` |
| E3 termination, rejected | `.dat` | `THE PROGRAM HAS DISCOVERED     n FATAL ERRORS` |
| E3 diagnostics | `.dat` | `***ERROR:` and `***WARNING:` markers |
| E4 increments | `.sta` | the `STEP INC ATT …` table, one row per increment |

Two properties of that reading are worth stating because a collector must not
undo them:

- **The solver prints a fatal-error count only when it has some.** A clean run
  is therefore counted from the `***ERROR:` markers, of which it has none.
  Zero markers *is* zero diagnostics — a reading, not an assumption.
- **A wrapped diagnostic repeats its marker on every line**, so marker counts
  over-report. Where the solver states a count, that count wins.

An unrecognised termination banner is refused, not classified: the reader
records `termination_wording` and reports status `none`. A banner it has not
seen must never read as a clean run.

### Nothing of the licence line may be published

The `.msg` header carries a line naming the licensed seat. It is the direct
analogue of the LS-DYNA licence number, and the same rule applies: the
whitelisted record holds numbers, enum values and version tokens only, and no
raw solver text is ever published. This is under test.

### Units

Abaqus carries no units — a deck is dimensionally consistent by the author's
discipline alone. The dataset therefore **declares** its convention, in the
platform's mass-length-time form. `t-mm-s` — tonne, millimetre, second — is
the consistent set that yields N and MPa, and is what the observed job uses.

### The input must be readable, attributable, and stably named

- **No `*Include`.** The input reader refuses a deck that hides content:
  it records `include` and asserts nothing absent. Ship a resolved deck.
- **Declare the solver.** ADR-0068 clause 4: a case that stores a solver
  input must carry `Provenance.solver_name`, or the input is not parsed at
  all. An unattributable deck is not guessed at.
- **Part and material names are load-bearing.** Abaqus names them where
  LS-DYNA numbers them, so the platform mints integer ids by order of first
  appearance (`mint_ids`, ADR-0068). Renaming or reordering a part between
  cases of one sweep changes its id. Keep them stable across the sweep.

### A step's "time" is not always time

A `*Static` step's time period is a dimensionless load parameter, not a
duration. The reader deliberately leaves `end_time` unset for such a step
rather than record a number in a field documented as seconds. A dynamic step's
period is a duration and can be read as one.

## After the run

Keep, per run folder: the resolved `.inp` and everything it references, the
`.sta`, `.msg` and `.dat`, the `.odb`, and — once it exists — the neutral
intermediate the extractor writes beside the `.odb` (ADR-0068 clause 6),
because an `.odb` alone cannot be opened by a recipient without an Abaqus
licence and so does not satisfy ADR-0040's sharing promise.

Then the dataset's glue reads the three text files through
`structbench.core.io.abaqus_run.read_abaqus_run_evidence` and writes one
whitelisted JSON record, exactly as the LS-DYNA glue does:

    python data_generation/abaqus/<dataset>/collect_run_evidence.py --out runs/datachecks/<name>_run_evidence.json
    python -m structbench.cli.datacheck measure --benchmark <name> --data-root <canonical dir> \
        --run-evidence runs/datachecks/<name>_run_evidence.json --out runs/datachecks/<name>.json

## Open points

These are **not established**. None is a recommendation; each is a question a
sourced dossier and one conformance run must settle (ADR-0068 clause 8).

1. **Everything about energy output (E5).** The observed job requested none
   and wrote none, so the reader returns no ledger. Which keyword requests a
   global energy history, what the printed terms are called, whether their sum
   reproduces a printed total, and whether any term is uncomputed by default
   as `*CONTROL_ENERGY`'s are in LS-DYNA — all unknown here.
2. **What `variable=PRESELECT` actually selects**, for both `*Output, field`
   and `*Output, history`. The observed job used it for both. Whether it
   yields integration-point data, or averages, is exactly the distinction
   LS-DYNA's `NINTSLD` open point turns on, and it is unread.
3. **How to request field output at the constitutive points** (E8), and which
   state variables a material writes.
4. **Per-part and per-interface output (E6) and reaction resultants (E7).**
   No observed job requested either.
5. **The time-integration record (E4) as a series.** The `.sta` increment
   table is read only as a count; whether a usable time/step series can be
   recovered from it, and what a `*Dynamic, Explicit` job writes instead, is
   unread.
6. **Whether `abaqus python` consumes a licence token.** The interpreter runs
   without one — `abaqus python -c "import sys"` succeeds — but whether
   `odbAccess` checks one out is unverified, and it decides whether a
   recipient needs a seat merely to read an archive.
7. **Element-code semantics.** The reader classifies only the `C3D` prefix it
   observed and refuses the rest by name. Which codes are reduced-integration,
   and therefore whether `under_integrated` can ever be established, is
   unread — so the hourglass rows cannot yet apply to an Abaqus run.
8. **Whether a rejected job's `.dat` always carries the fatal-error count**,
   or only for pre-processing failures. A job that fails during the analysis
   may end differently, and no such run has been seen.
