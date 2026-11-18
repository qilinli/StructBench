# 0079 — How a further solver is admitted to data generation

**Status**: Proposed
**Type**: Ephemeral
**Date**: 2026-11-18
**Amends**: 0068 (clause 1's admission of one solver becomes a rule for any; clause 2 stands), 0071 (`check`'s solver list gains a second state)

**Your call**:
1. A solver's code may be merged as *experimental* before the solver is admitted, on evidence a reviewer can read without the solver. Reversing means a contributor brings readers, a conformance page and a run in one pull request, or nothing.
2. Admission rests on one conformance run that you read through the readers but do not re-run; its files reach you privately. Reversing means every admitted solver needs a licence and an installation of yours.
3. Admission is your ratification of the solver's conformance page, with no decision record per solver, and ADR-0068's "no abstraction" answer holds for a third solver. Reversing the first costs one record per solver; reversing the second is a redesign of the readers.

## Context

ADR-0068 admitted Abaqus as the second solver by name and answered the abstraction question with plain reader functions. `structbench.datagen.definition.SOLVERS` is `("abaqus",)`, so `check` refuses a definition that names any other solver. `docs/datagen/solver-requirements.md` says what the verification instrument requires of any solver, but nothing says what a contributor must bring for a further one, or what its data may then be used for. Without a rule, each solver needs the maintainer's answer before a pull request can be opened.

## Decision

1. **Two states.** A solver known to data generation is *experimental* or *admitted*. `definition.py` holds one tuple for each. `check` accepts a definition naming either; for an experimental solver it says so in one line, and the data card's provenance part says so too.
2. **Entering as experimental** takes a pull request that the suite can test with no solver installed:
   - the reader pair in `core/io/<solver>.py` and `core/io/<solver>_run.py`, producing `Case`, `InputFacts` and `RunEvidence`, obeying the three reader rules of `solver-requirements.md`, and registered in the input-reader lookup;
   - tests on fixture files the solver wrote, with licence text removed (`CONTRIBUTING.md`);
   - a draft conformance page under `docs/datagen/`, on the template of `solver-requirements.md`, with what is not established listed as open;
   - an example definition under `datagen/examples/` that passes `check`.
3. **Admission** takes, in addition:
   - one conformance run of the example definition, whose files the contributor gives the maintainer privately. They are not published;
   - the verification record of that run, produced by the maintainer from those files, in which every catalogue row is measured or is absent for a reason the conformance page lists;
   - the conformance page with its open points either closed or accepted by the maintainer.
   The maintainer ratifies the page, dated, and moves the solver's name to the admitted tuple. No decision record is written unless the solver needs something this record or ADR-0068 does not allow.
4. **What experimental withholds.** A dataset generated with an experimental solver is not offered as a benchmark's reference data until its solver is admitted. Its cases can still be converted, measured and judged.
5. **Where the code lives** is ADR-0068's pattern, unchanged: readers in `core/io/`, the deck writer and any exporter in `structbench.datagen.<solver>`. The package imports no solver. A result database that only the solver's own interpreter opens is read through a neutral intermediate (ADR-0068 clauses 5 and 6). A library a reader needs is a dependency, with its own record.
6. **No abstraction.** ADR-0068 clause 2 stands with three solvers: the neutral record types are the interface, and a further reader is one more entry in the lookup.
7. **Not decided here: how the stages after `check` choose a solver's code.** They are Abaqus's today. Until that is decided, each of them refuses a definition whose solver is not Abaqus, naming the solver, with exit code 2. An experimental solver's conformance run is therefore launched by the contributor outside the stages and converted by its reader.

## Alternatives considered

- **Admit each solver by its own record, as ADR-0068 did.** The judgments would be the same each time.
- **No experimental state: `check` accepts only admitted solvers.** Nothing would be mergeable until a conformance run had been read.
- **The maintainer re-runs the conformance case.** It ties admission to the licences the maintainer holds; the instrument judges a run from its files (ADR-0066).
- **A solver protocol, now that there is a third.** The readers share record types, not behaviour. The stage question (clause 7) is where a seam may be needed, and it is left open.
- **Experimental code outside the package.** The suite would not test it.

## Consequences

- A contributor can open a pull request for a solver without asking first; `CONTRIBUTING.md` gains a section stating clauses 2 and 3.
- To build on acceptance: the experimental tuple, the notice in `check` and the card, the refusal in the later stages. No solver is added by this record.
- The maintainer holds private conformance files for each admitted solver, as the raw archives are held (ADR-0040).
- LS-DYNA has readers and a conformance page but no deck writer; it is not a data-generation solver and this record does not make it one.
