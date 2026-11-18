# What the instrument requires of a solver

*The solver-neutral statement behind the two conformance documents
([LS-DYNA](lsdyna-conformance.md), [Abaqus](abaqus-conformance.md)): what a
run must supply for the reference-data verification instrument (ADR-0066) to
measure it, whatever solver produced it. A further solver's conformance page
fills in the template at the end. Nothing here is a new requirement: every
item is ADR-0066's, and the code that holds it is named beside it.*

## The shape of the requirement

The instrument never reads a solver's files. It reads three neutral records,
and a solver is supported exactly as far as plain reader functions can build
them from its output (ADR-0068 clause 2: the record types are the whole
interface):

| Record | Built from | Holds | Defined in |
|---|---|---|---|
| `InputFacts` | the solver input, as stored with the case | parts, materials, the time integration, what the model has (contact, erosion, damping, mass scaling, rigid planes), which energy terms the input computes and which outputs it requests | `structbench.core.evidence` |
| `RunEvidence` | the files the run itself writes | solver identity, termination, diagnostic counts, the time step, the energy ledger | `structbench.core.evidence` |
| `Case` | the result database | the mesh, the stored fields and globals over time, in SI | `structbench.core` |

Three rules hold for every reader, and they are where a new one most easily
goes wrong:

1. **A reader fails closed.** What it does not recognise it records as
   unparsable and asserts nothing about: an unrecognised termination banner
   is not a clean run, an input that hides content (an include, a parameter
   file) is not an input with nothing in it. A missing item is `None`, never
   a default.
2. **An absence is a claim about the data.** When a row cannot be measured,
   the reason says whose gap it is: the run did not keep the evidence
   (`source_missing`), or the platform cannot read it yet (`unsupported`).
   The second never counts against a contributor, so it must not be used
   where the first is true (ADR-0068 clause 3).
3. **A stored input is read only by its own solver's reader**, chosen by the
   case's declared `Provenance.solver_name`. A case that stores an input and
   declares no solver is not parsed at all (ADR-0068 clause 4).

## The evidence items

ADR-0066 names ten items of run evidence (`EvidenceItem`). For each: what the
instrument needs, in solver-neutral terms, and where the two existing pages
say how their solver supplies it.

| Item | What a run must supply | Goes into |
|---|---|---|
| **E1** the complete input | the input the solver read, resolved (nothing hidden behind an include), stored with the case and attributable to its solver | `InputFacts` |
| **E2** solver identity | the solver's name, version and revision, the precision it ran in, the parallel layout | `RunEvidence.identity` |
| **E3** termination and diagnostics | how each phase ended (`normal`, `error`, or not established), the final time, and the run's own counts of errors and warnings, from wherever the solver prints *all* of them | `RunEvidence.termination`, `n_errors`, `n_warnings` |
| **E4** the time-integration record | for explicit integration, the time step over time | `RunEvidence.timestep` |
| **E5** the global energy ledger | every energy term the model can have, each as its own series, with the solver's own total | `RunEvidence.ledger` |
| **E6** per part and per interface | energy and mass per part and per contact interface | no field yet: no run has supplied it in a form that was read |
| **E7** loads and reactions | applied-load resultants and reactions | no field yet |
| **E8** fields at the constitutive points | stress, strain and state at the integration points, not averaged over them | `Case.response` |
| **E9** one sampling clock | the ledger sampled at the instants the fields are stored at | read from the ledger's and the case's times |
| **E10** units | the declared unit system (mass-length-time), and three SI anchors of independent dimension | the dataset's declaration |

### The ledger and its identity (E5)

The ledger is where solvers differ most and where a reader's choices matter.

- **Terms.** The neutral names are `kinetic`, `internal`, `external_work`,
  `zero_energy_mode` (hourglass or artificial strain energy), `contact`,
  `rigid_surface`, `damping`, `discrete_element`, and the three eroded terms
  (`LEDGER_TERMS`). A reader maps the solver's own terms onto these and
  invents none.
- **The identity is declared, then checked.** The reader states which terms
  sum to the total, each with its sign. A term that is contained in another
  (an eroded energy inside the kinetic term, a dissipation inside the
  internal term) or that is the input side (external work) stays out of the
  identity. The reader keeps the solver's own printed total and the
  instrument checks the identity against it, so a term the reader missed
  cannot pass silently.
- **A term the solver does not compute unless asked is part of the input
  requirement.** Both solvers have them: LS-DYNA computes no hourglass or
  damping energy by default, and an Abaqus ledger does not close on a
  contact model without the penalty-work term. The conformance page says
  which requests the model's features make necessary, and
  `input_requests_required_evidence` reads them back off the input and counts
  the ones it omits: a deck that could never have supplied its evidence is
  told apart from a run whose files were lost.

### The clock (E9)

The instrument compares the ledger with the stored fields only at shared
instants. The requirement on the input is that there are some: the history
(ledger) output and the field output are requested on one clock, with frames
at exact multiples of the interval. Which relations between the two intervals
a solver's reader can place is the conformance page's to say (LS-DYNA: the
field interval a whole multiple of the ledger's; Abaqus: equal intervals).
Solvers add frames of their own (a frame at the termination time, a
duplicated end-of-step frame); the page says which, and what the adapter does
with them.

### What is declared, not read (E10, and the material)

Units are never read from a solver: the dataset declares its system, and the
adapter converts to SI. What a history variable holds, and which material
class a model belongs to, are likewise declared once per dataset; a material
class the platform does not have is a decision record of its own (ADR-0067,
ADR-0070).

## What a conformance page must establish

A page is written from the solver's own output, read, never from recall of
its manual: each statement names the file it was observed in, or cites a
claim in a sourced dossier, and what is not established is listed as open
and not recommended (ADR-0068 clauses 7 and 8). The template:

1. **Status.** What the page rests on: which runs were read, which release,
   what is sourced and what is observed.
2. **The job's own record.** Which files a run writes, and which exist when
   a run is rejected before it starts or fails during the analysis. Per
   item E2 to E4: the file and the exact text the reader keys on. Where the
   diagnostics are counted from, and what a count of zero rests on.
3. **Licence text.** Which files carry a licensee, a seat, a host or a path.
   The published record holds numbers, enum values and version tokens only;
   no raw solver text is published.
4. **Units.** Whether the solver carries units, and the convention the
   examples declare.
5. **The input.** What makes it unreadable (includes, parameters), how parts
   and materials are identified, and how the reader mints stable ids where
   the solver uses names.
6. **The energy output (E5).** The solver's terms and their neutral names;
   the identity, with the evidence that it closes on a real run; which terms
   are computed only on request, and for which model features.
7. **Field and history output (E8, E9).** The request that puts frames on an
   exact clock, the precision the fields are stored in, the extra frames the
   solver writes, and how integration-point values are obtained without
   averaging.
8. **What the instrument requires of an input.** The table
   `input_requests_required_evidence` enforces: each request, when it is
   required, and why. What the platform cannot confirm is not counted as
   missing, and the page says which cases those are.
9. **After the run.** What to keep per run folder, and how the result
   database is read: directly by the package, or through a neutral
   intermediate written by the solver's own tools when the database can be
   opened no other way (ADR-0068 clauses 5 and 6).
10. **Open points.** Everything not established, each as a question a run or
    a source would settle.

A page is a draft until one unpublished *conformance run* with its input
block has been read through the readers; the maintainer ratifies it. What a
contributor brings for a further solver, and when it is admitted, is ADR-0079
(`CONTRIBUTING.md`, *Contributing a solver*).
