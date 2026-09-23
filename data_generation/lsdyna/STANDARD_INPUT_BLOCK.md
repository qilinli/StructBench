# The standard LS-DYNA input block

*How an LS-DYNA run supplies the run evidence E1–E10 of ADR-0066, so that the
reference-data verification instrument can measure it. Every generated input
deck in this repository carries this block. A contributed dataset is measured
against the same requirement; this file tells a contributor what to switch on.*

**Status: draft, never yet exercised by a run.** It is assembled from the
keyword manual (R13 and R15 were read; the legacy sweeps ran R12) and from
what was observed in one legacy run folder. Since 2026-09-23 the instrument
**reads this block back off the deck**: `input_requests_required_evidence`
counts the requests below that an input omits, for the features the model
actually has, so a deck that could never have supplied the evidence is
distinguished from a run whose files were lost. Both legacy sweeps fail it. Each setting cites its claim in
[`docs/plans/2026-09-21-reference-data-verification-sources.md`](../../docs/plans/2026-09-21-reference-data-verification-sources.md)
(`L-…` energy ledger and time integration, `F-…` fields, echo and
diagnostics). One unpublished *conformance run* with this block is what turns
the draft into a tested realisation; the [open points](#open-points) are what
that run has to settle.

Settings are given **by variable name, not by column**. Card layouts differ
between releases — `*CONTROL_ENERGY` gained two fields in R15 (L-C07) — so
write the cards with LS-PrePost for the release you run, or from that
release's manual, and keep the `$#` variable header line above each card: the
input reader in `structbench.core.io.lsdyna_run` is fixed-width and fails
closed on what it does not recognise.

---

## What to switch on

### E5 — the energy ledger, every term separate

`*CONTROL_ENERGY`: by default three dissipation terms are **not computed**, so
they are missing from the total and from the ledger.

| Variable | Set | Why |
|---|---|---|
| `HGEN` | 2 | default 1 computes no hourglass energy; 2 includes it in the balance and reports it (L-C02). Costs about ten percent. |
| `RWEN` | 2 | rigid-wall energy; 2 is the default and is stated so the input says it (L-C03) |
| `SLNTEN` | 2 | sliding-interface (contact) energy; the manual says it is forced to 2 when contact is active — state it anyway (L-C04) |
| `RYLEN` | 2 | default 1 computes no Rayleigh-damping dissipation (L-C05) |
| `IRGEN`, `MATEN` | leave at defaults (2, 1) | L-C06 |
| `DRLEN`, `DISEN` | R15 only, implicit-related; 2 for an implicit run | L-C07 |

`*DATABASE_GLSTAT` with a non-zero `DT`: the global statistics. No file is
written without its card, and `DT = 0` means no output (L-C27).

The balance identity the reader declares for this ledger is

    total = kinetic + internal + hourglass + rigid-wall + system damping + sliding interface

(L-C11, from the vendor's support site — the manual gives no formula, L-C10).
Eroded energies are *contained in* the kinetic and internal terms (L-C13), and
external work is the input side: the printed "total energy / initial energy"
is total / (initial + external work) (L-C12). Stiffness damping and solid bulk
viscosity are inside internal energy (L-C14, L-C15, L-C17). The reader checks
this identity against the solver's own printed total and refuses the ledger if
it does not reproduce it, so a term this list has missed cannot pass silently.

### E6 — per part and per contact interface

| Keyword | Gives | Claim |
|---|---|---|
| `*DATABASE_MATSUM` | per part: kinetic, internal and hourglass energy, momentum, added mass | L-C20, L-C21 |
| `*CONTROL_OUTPUT` `IERODE = 1` | adds per-part eroded internal and kinetic energy to `matsum` (default 0 writes none) | L-C23 |
| `*DATABASE_SLEOUT` | per contact interface: slave, master and frictional energy | L-C20, L-C21 |

Part-summary and global kinetic energy legitimately differ (L-C22); they are
not expected to sum.

### E7 — loads and reactions

| Keyword | Gives | Claim |
|---|---|---|
| `*DATABASE_RWFORC` | rigid-wall normal and x, y, z force | L-C21 |
| `*DATABASE_RCFORC` | contact resultants — **averaged over the preceding output interval**, not instantaneous | L-C21, L-C24 |
| `*DATABASE_SPCFORC` | single-point-constraint reaction forces and moments | L-C21 |
| `*DATABASE_BNDOUT` | boundary-condition forces and energy; its `OPTION1–4` left at 0 so that nodal force groups, concentrated forces, pressures and prescribed motions are all included | L-C25 |
| `*DATABASE_NODFOR` + `*DATABASE_NODAL_FORCE_GROUP` | reaction resultants of a node set, and the external work they do | L-C26 |

Request the ones the model has: a wall needs `RWFORC`, a contact `RCFORC` and
`SLEOUT`, a prescribed motion `BNDOUT`.

### E4 — the time-integration record

Explicit runs need nothing beyond `*DATABASE_GLSTAT`: it carries the time
step and the element and part controlling it (L-C34). Added mass appears in
`glstat` only when mass scaling is on (`DT2MS < 0`, L-C09), and per part in
`matsum` (L-C33). The controlling element is in the text file but LS-PrePost
does not read it (L-C34) — the glue reads the text.

### E3 — termination and diagnostics

| Setting | Why | Claim |
|---|---|---|
| `*CONTROL_TERMINATION` stated in full (`ENDTIM`, `ENDCYC`, `DTMIN`, `ENDENG`, `ENDMAS`) | every criterion that can end a run "normally" short of `ENDTIM` is then in the input, where the reader finds it | L-C36 |
| `*CONTROL_OUTPUT` `MSGMAX` > 0 (the default 50 is fine) | a positive cap limits the screen only; **every** warning and error still reaches the message file. Zero or negative truncates the file too | F-C23 |
| `*CONTROL_SOLUTION` `ISNAN = 1` | checks the assembled force arrays for NaN; default is no check. About two percent | F-C25 |
| `*CONTROL_CONTACT` `IGNORE = 2` (automatic contacts) | initial penetrations are tracked **and printed**, instead of nodes being moved silently | F-C28 |

### E8 — fields at the constitutive points

`*DATABASE_EXTENT_BINARY`:

| Variable | Set | Why | Claim |
|---|---|---|---|
| `NINTSLD` | 8 | any other value writes the **average** over a solid's integration points — von Mises of an average is below the average von Mises, so yield checks need the points | F-C10 |
| `MAXINT` | negative, `-n` for `n` through-thickness points | writes every in-plane integration point of a shell with no averaging | F-C04, F-C05 |
| `STRFLG` | 1 (11 to add the plastic-strain tensor) | strain tensors; read digit-wise | F-C06 |
| `NEIPH`, `NEIPS` | the number of history variables the material's admissibility function needs | extra history variables for solids **and SPH** (`NEIPH`) and for shells (`NEIPS`) | F-C02, F-C03 |
| `DELERES` | 1 where elements are deleted | 0 writes deleted elements as all zero; 1 keeps their last state | F-C12 |

The strain written is a time-integrated rate of deformation (F-C08); declare
it as such. Which history variable holds what is material-specific and is
declared once per dataset (ADR-0066 clause 2, E8).

### E9 — one sampling clock

Give every `*DATABASE_` card above the same `DT`, and make the
`*DATABASE_BINARY_D3PLOT` interval a whole multiple of it, so that ledger
samples and stored states fall on shared instants. The instrument *finds*
shared instants (within a hundredth of a stored interval) and compares ledger
and fields only there. The legacy Taylor sweep used 0.002 ms for both and
matches on every frame but the one written at the termination time.

### E2 — solver identity

Nothing to switch on: the message file's header carries the version, revision,
precision and parallel layout, and the glue reads those. Binary output is
64-bit by default from a double-precision executable; `*DATABASE_FORMAT`
`IBINARY = 1` would make it 32-bit — leave it at 0 and say so (F-C13).

### E1 — the input as the solver read it

Keep the keyword deck with the run, together with every file it includes.
`d3hsp` is the solver's echo of the input (F-C14, F-C17), and
`outdeck = s` on the execution line writes a resolved structured-format deck,
`dyna.str` (F-C21). The input reader refuses `*INCLUDE` and `*PARAMETER`
rather than guess at what they hide, so a deck that uses them needs the
resolved form.

### E10 — units

Not a solver matter. Declare the unit system (`g-mm-ms`, …) and three SI
anchors of independent dimension — a density, a stress-dimension constant, a
length — each naming an input quantity and its SI value (ADR-0066 clause 2).
The K&C concrete model's `RSIZE` and `UCF` are built-in unit anchors (F-C36).

---

## After the run

**MPP writes these databases in binary only.** Whatever `BINARY` says, an MPP
executable produces `binout*`; the text files come from the converter
afterwards (L-C28, L-C29):

    l2a binout*

Keep, per run folder: the deck and its includes, `d3hsp`, the message file of
rank 0 (`mes0000`; `messag` for SMP, F-C22), the converted `glstat`, `matsum`
and whichever of `rwforc`, `rcforc`, `sleout`, `spcforc`, `bndout`, `nodfor`
the model requested, and the `d3plot` family.

Then the dataset's glue (for Taylor,
`2D-Copper-Bar-Taylor-Impact/collect_run_evidence.py`) builds each run
folder's path from its case id, reads the text files through
`structbench.core.read_run_evidence`, and writes **one** whitelisted JSON
record. `mes0000` and `d3hsp` contain a licence number, a host name and local
paths; the record holds numbers, enum values and version tokens only, and
raw solver files are never published.

    python data_generation/lsdyna/<dataset>/collect_run_evidence.py --out runs/datachecks/<name>_run_evidence.json
    python -m structbench.cli.datacheck measure --benchmark <name> --data-root <canonical dir> \
        --run-evidence runs/datachecks/<name>_run_evidence.json --out runs/datachecks/<name>.json

## What the instrument reads today

| Item | Read from | Status |
|---|---|---|
| E1 | the deck stored in the canonical case | built, including `*CONTROL_ENERGY` and which `*DATABASE_` cards the input requests |
| E2, E3 | `mes0000` | built |
| E4, E5 | `glstat` | built (time step; ledger with its identity) |
| E6 | `matsum`, `sleout` | **not built** — no run has supplied them in a form that was read |
| E7 | `rwforc`, `rcforc`, `spcforc`, `bndout`, `nodfor` | **not built** |
| E8 | the canonical case | built |
| E9 | shared instants of `glstat` and the stored states | built for the kinetic-energy closure and for stored globals against the ledger |
| E10 | benchmark declarations | unit label built; anchors live on the card (`BenchmarkCard.units_anchors`), declared for Taylor only |

A reader is added when a run first supplies its file (ADR-0066 clause 3); the
conformance run is what triggers E6 and E7.

## Open points

These are **not established** and are not to be assumed; the conformance run
is how each is settled.

1. **Text output from MPP.** The manual says MPP writes binary only, yet one
   legacy MPP run folder holds both `binout0000` and a text `glstat`, written
   with `BINARY = 0` on the card. Unexplained.
2. **Per-part added mass under MPP.** An old release note says SMP only;
   current status unverified.
3. **Whether `d3hsp` echoes the content of `*INCLUDE` files.** Not established
   (F-C14, F-C17).
4. **SPH artificial viscosity in the energy balance.** No statement found on
   which ledger term, if any, holds it.
5. **Whether "spring and damper energy" is contained in internal energy.** Not
   established; the reader leaves it out of the identity and relies on the
   check against the printed total.
6. **Message-file wording — settled, and it varies within one release.**
   Taylor's R12.0.0 build prints `SVN Version: 148978`; the notch sweep's
   R12.1-190 build prints `Revision: R12.1-190-gadfcdf9018` and no SVN line
   at all. The reader now takes the SVN number where both appear and the
   describe string otherwise. Treat any *other* banner wording as unread
   until a run shows it: the reader reports `unparsable` rather than guessing.
   A related assumption is **not** sourced and should be: diagnostics are
   counted by the lines that *raise* one — the severity word at the head of
   the line — because a warning's own explanatory text may name an "error"
   it is reporting. Observed, not documented.
7. **The printed label of the hourglass term** is taken from the manual's
   table (L-C08); no run with `HGEN = 2` has been read.
8. **`d3plot`'s global total energy is not `glstat`'s total.** On the Taylor
   sweep the stored `global_total_energy` reproduces the ledger's kinetic
   plus internal energy to 2e-6, while the solver's printed total also
   carries the rigid-wall term — so the two disagree by 0.6 to 1.2 % of the
   peak, growing as wall work accumulates. Both come from the same run.
   Which definition `d3plot` writes, and whether it varies with the terms
   `*CONTROL_ENERGY` switches on, is unestablished; a run with `HGEN = 2`
   and contact would settle it. Until then the stored channel named
   `total_energy` should not be read as the run's total energy.
9. **`d3hsp` is kept but never read.** It is on the keep list above as the
   solver's echo of the input and as the fallback if the deck is lost;
   nothing in `structbench` opens it. Keep it for provenance, not because
   the instrument needs it.
