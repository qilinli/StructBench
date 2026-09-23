# 0068 — Abaqus is the second solver; the deferred abstraction question is answered

**Status**: Proposed
**Type**: Durable
**Date**: 2026-09-24

## Context

ADR-0004 commits the platform to being solver-agnostic. ADR-0010 put
solver-specific scripts in `data_generation/<solver>/` and solver-output
adapters in `core/io/`, and explicitly **deferred** a solver-abstraction layer
as premature: *"A multi-solver abstraction needs at least two concrete solvers
to be designed correctly."* ADR-0066 clause 3 then wrote the deferral into the
verification module: *"`core/io/lsdyna_run.py` is the single extractor. There is
no protocol, registry, or plugin seam (ADR-0010 stands)."*

Abaqus is that second solver. A 2025 installation is available (scripting
interpreter Python 3.10.5), with a worked cantilever example carrying a
successful run, a batch-only run, and a deliberately failing job. The README
Roadmap already names Abaqus as the first candidate for agentic data
generation. So the deferred question's trigger has fired, and the honest thing
is to answer it rather than let a second parallel path grow by accident.

**The single extractor was already silently wrong, and that is what forced
this ADR now.** `measure_dataset` handed *any* stored deck to
`read_input_facts` — the LS-DYNA keyword parser. Measured on the example's
real `Cantilever.inp` (1743 lines, `*End Part` at 1683, `*Static` present):

    time_integration    = 'explicit'      <- the deck is *Static
    databases_requested = frozenset()
    materials = 0   parts = 0

It does not fail; it **asserts a false fact** about the run and then reports
zero requested databases, so `input_requests_required_evidence` — a row whose
`bears_on` is `input` — FAILS, accusing a correct Abaqus deck of omitting
outputs it never needed. This is the species ADR-0066's claim-audit note
names, in its most severe form: not a misattributed absence but an invented
positive claim, used to fail a contributor.

## Decision

1. **Abaqus is admitted as the platform's second solver.** Solver-side code
   lives in `data_generation/abaqus/`, never importable (ADR-0010 unchanged).

2. **ADR-0010's deferred abstraction question is answered: no abstraction.**
   Each solver contributes plain reader functions producing the existing
   solver-neutral records (`InputFacts`, `RunEvidence`, `Case`). Those record
   types *are* the whole interface. There is no protocol, no plugin seam, and
   no per-solver class hierarchy. What changes from ADR-0066 clause 3 is only
   that "the single extractor" becomes "one extractor per solver, selected by
   a lookup" — recorded here so the repository does not lose a deliberately
   deferred decision by never noting that its trigger fired.

3. **A stored solver input is read only by the reader for its own solver, and
   dispatch fails closed.** `input_facts_for(case)` selects on
   `Provenance.solver_name`, normalised so `"LS-DYNA"`, `"ls dyna"` and
   `"LSDYNA"` agree. Three outcomes, and they are three different claims:

   - no deck stored → no facts, and the E1 rows report `source_missing`, the
     dataset's gap, as before;
   - deck stored, solver recognised → parsed by that solver's reader;
   - **deck stored, solver absent or unrecognised → not parsed**, and the E1
     rows report `unsupported`, which is in `PLATFORM_REASONS` and so never
     counts against the contributor.

   The third outcome may not be implemented by passing `facts=None`: `gate()`
   then returns `source_missing`, which is *not* a platform reason, so the
   misreport would be relocated rather than closed. The reason is threaded
   through `measure_case` into `gate()`.

4. **Provenance becomes required for any case that stores a solver input.**
   `Provenance` is optional in the schema and `solver_name` is free text; a
   deck we cannot attribute is therefore not parsed at all. Guessing from the
   deck's own text would reintroduce exactly the bug this ADR exists to close.
   All four registered benchmarks already declare a solver (three `LS-DYNA`,
   `DeformingPlate` `COMSOL`), so no published record changes.

5. **The Abaqus path is two processes, and that is forced, not chosen.** The
   package requires Python `>=3.12`; the Abaqus scripting interpreter is
   3.10.5. An `.odb` is readable only through that interpreter, whereas
   `lasso-python` reads a d3plot with no LS-DYNA installed. So an
   Abaqus-side exporter under `data_generation/abaqus/` writes a neutral
   intermediate, and the package adapter reads only that. `odbAccess` must
   stay unreachable from `import structbench`, or ADR-0004's promise that a
   user needs no solver is broken.

6. **The neutral intermediate is a long-lived archived artefact, not a
   scratch file.** It is stored in the raw archive beside the `.odb` and
   shared on request with it. ADR-0040 promises archives shared on request;
   with LS-DYNA that hands a recipient a d3plot they can convert for free,
   while a bare `.odb` hands most recipients a file they cannot open. The
   intermediate restores that parity, so it carries a version and a
   documented layout. It is **not** published to Hugging Face: it is
   near-duplicate mass beside the canonical `.h5` and gives a downloader
   nothing the `.h5` does not already give them.

7. **The Abaqus deck requirement is written before any Abaqus data is
   generated.** `data_generation/abaqus/STANDARD_INPUT_BLOCK.md` states what
   a job must request for a run to supply E1–E10, sourced against the
   Keywords Reference. Every current benchmark is retrofitted, and notch
   carries thirteen permanently unanswerable rows because nobody asked for
   `glstat` at deck-writing time. Building to the requirement is the only way
   the first Abaqus dataset is the first one that meets the platform's own
   standard.

8. **This ADR discharges none of ADR-0065's four follow-ups**, and settles no
   keyword-level question. Which Abaqus keywords realise E5, and the
   output-request vocabulary the conformance row should check, are deferred
   to a dated amendment once the claim dossier exists and one conformance run
   has been read — so that no unverified external claim enters an accepted
   record.

## Alternatives considered

- **A solver protocol or plugin registry** (`SolverAdapter` with `read_input`,
  `read_run`, `read_case`). Rejected. Two solvers is the minimum evidence for
  an abstraction and the minimum at which over-fitting is likely; the neutral
  *record types* already carry everything shared, and a protocol would add a
  seam with one implementation per method and no third caller to justify it.
  ADR-0010's reasoning survives its own trigger.

- **Sniff the solver from the deck text** (`*KEYWORD` vs `*Heading`).
  Rejected. It is a guess, it silently prefers whichever pattern is tested
  first, and a deck that matches neither would fall back to the LS-DYNA
  parser — which is the present defect restated. Declared attribution can be
  wrong, but it is wrong *visibly*.

- **Keep `facts=None` for a foreign deck.** Rejected on measurement: it
  produces `source_missing`, a contributor-owned reason, across the E1 rows.
  It reads as "your input is missing" to someone who supplied it in full.

- **Read the `.odb` directly from the package** via `odbAccess`. Rejected: it
  would make `import structbench` require an Abaqus installation and licence,
  breaking ADR-0004. The two-process split is not a style preference.

- **Publish the neutral intermediate to Hugging Face.** Rejected as
  near-duplicate mass — notch alone is 24.9 GB — for an artefact a downloader
  gains nothing from. The on-request channel is where the asymmetry bites and
  where clause 6 puts the remedy.

- **Defer all of this until Abaqus data exists.** Rejected. The dispatch
  defect is live *now* and would meet the first Abaqus dataset by accusing it;
  fixing the instrument after it has misreported is worse than before.

## Consequences

- A second solver can be added without touching `benchmarks/`, `models/` or
  `eval/` — ADR-0010's stated consequence, now exercised rather than assumed.
- `core/io/lsdyna_run.py` is no longer "the single extractor"; ADR-0066
  clause 3 gains a dated note. LS-DYNA vocabulary still stays inside
  `core/io/lsdyna*.py`, and Abaqus vocabulary will stay inside
  `core/io/abaqus*.py`.
- A case that stores a deck without provenance is newly unparsed. No
  published record is affected, but a future contributor who omits
  provenance gets `unsupported` rather than silent mis-parsing.
- The Abaqus reader work — both text readers, their tests, the standard input
  block, the dossier — is authorable and reviewable with **no Abaqus
  invocation at all**, using the example files as a read-only layout test
  bed. Only the conformance run, the `.odb` extractor and the first benchmark
  need a licensed installation.
- **Risk.** `Provenance.solver_name` is free text that nothing has validated
  until now, so its values are whatever past ingestion happened to write. The
  four current values were checked by hand for this ADR. A misspelling in a
  future dataset degrades safely — to `unsupported`, a visible platform gap —
  rather than to a wrong parse, which is the property clause 3 is chosen for.
- **Not decided here.** Whether `solver_name` should become a closed
  vocabulary; whether the intermediate's format is HDF5 or something else;
  every Abaqus keyword question in clause 8; and whether a `linear_elastic`
  material class is added, which is a separate decision on its own merits.
