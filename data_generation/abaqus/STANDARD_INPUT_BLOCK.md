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
re-run batch from its `.inp`, and a deliberately rejected job — and, since
2026-09-24, seven Abaqus/Explicit conformance jobs. Each statement names the
file it came from. **What is not established** is listed
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
| E3 diagnostics | `.msg`, else `.dat` | `n ERROR MESSAGES` and `n WARNING MESSAGES` in the `.msg` ANALYSIS SUMMARY are the run's own totals and are authoritative; the `.dat` carries no such summary, and its `***ERROR:` markers are the fallback for a rejected job that has no `.msg` |
| E4 increments | `.sta` | the `STEP INC ATT …` table, one row per increment |

Two properties of that reading are worth stating because a collector must not
undo them:

- **The `.msg` ANALYSIS SUMMARY is where the counts are**, and it separates
  warnings raised during input processing from those raised during the
  analysis, so both are summed. Reading only the `.dat` — which carries no
  such summary at all — let a run with analysis errors report zero and pass a
  ratified must-be-zero requirement.
- **A rejected job has no `.msg`**, so its `.dat` fatal-error count, and
  failing that its `***ERROR:` markers, are the fallback. Zero markers *is*
  zero diagnostics — a reading, not an assumption.

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

### Abaqus/Explicit (conformance run, 2026-09-24)

Seven `*Dynamic, Explicit` jobs of an axisymmetric CAX4R impact case against an
analytical rigid wall, run with `double=both cpus=1`. Three completed. Four
stopped early on excessive distortion, which is a physical limit of those
cases, not a deck fault. The files cited are each job's own. "npz" means the
job's ODB exported through `odb_export.py`.

**The job's own record**

| Item | File | What it says |
|---|---|---|
| E2 identity | `.sta` header | `Abaqus/Explicit 2025` |
| E2 identity | `.dat` line 3 | `Abaqus 2025` |
| E2 precision | `.sta`, `NUMERICAL PRECISION USED` block | `Double precision package and explicit executables will be used in this analysis.` |
| E3, normal end | `.sta`, last line | `THE ANALYSIS HAS COMPLETED SUCCESSFULLY` (the Standard wording) |
| E3, analysis-phase failure | `.sta`, last line | `THE ANALYSIS HAS NOT BEEN COMPLETED` |
| E3, diagnostics | `.sta` | Explicit writes its `***ERROR` and `***WARNING` blocks here, for example `***ERROR: Excessive distortion of element number N` |
| E3, diagnostics | `.msg` | holds only `STEP 1 ORIGIN` markers; it has **no** ANALYSIS SUMMARY with message counts |
| E4, time step | `.sta` increment table | one row per printed increment, with columns `INCREMENT, TOTAL TIME, STEP TIME, CPU TIME, STABLE INCREMENT, CRITICAL ELEMENT, KINETIC ENERGY, TOTAL ENERGY`. A distortion failure shows the stable increment collapsing (to `1.00000E-14` in one run) |

Two consequences for the readers:
- Under Explicit, diagnostics must be counted from the `.sta`. Reading only the
  `.msg` and `.dat` finds the `.dat` pre-processor warnings and misses every
  analysis-phase error.
- A failure during the analysis leaves **no** fatal-error count in the `.dat`.
  Its only record is the `.sta`.

**Output requests and what they produce**

- **Field output.** `*Output, field, time interval=Δ, time marks=YES` wrote
  frames at `kΔ` for `k = 0 … N`: frame times on the grid to float32 rounding,
  which is float32 rounding (npz `step/<step>/frame_times`). It **also wrote an
  extra end-of-step frame (N + 1)** at the same time as frame N (`.sta`: an
  `Output Field Frame Number N+1` line after `Restart Number 1`, at the
  step's end time). Its U, V, S, PEEQ and every history output repeat
  frame N's to float32 storage (in a few percent of runs a value or two
  differs by a few ulp, at most 1.8e-7 of the field's peak). **Its A does
  not:** in every run of a 2026-09-24 sweep it differed from frame N's, by up
  to 64 % of the field's peak, sign changes included. Why is not
  established. Consumers must expect that frame; the canonical adapter keeps
  frame N (`core/io/abaqus.py`).
- **Field data are float32 under `double=both`** (npz dtypes; manifest
  precision `DOUBLE_PRECISION`). The analysis runs in double precision; the
  stored fields do not.
- **Stress and state at the integration point.** `*Element Output` with `S` and
  `PEEQ` gave position `INTEGRATION_POINT`, one integration point per CAX4R
  element (npz `integration_points` are all 1). The stress components are
  `S11, S22, S33, S12`, where S33 is the hoop stress.
- **Energy (E5).** `*Output, history, time interval=Δ` followed by
  `*Energy Output` naming `ALLAE, ALLCD, ALLFD, ALLIE, ALLKE, ALLPD, ALLSE,
  ALLVD, ALLWK, ETOTAL` wrote all ten terms under the history region
  `Assembly Assembly-1`. They were sampled at the same instants as the field
  frames, including the duplicate end frame (npz `history/…`).
- **Reaction (E7, partly).** `*Node Output` of `RF2` on the rigid body's
  reference node, in the history request, wrote region
  `Node <instance>.<label>`. That is the wall's reaction resultant.
- **Initial conditions reach frame 0 exactly** (npz frame 0 against the deck):
  - `*Initial Conditions, type=VELOCITY` data lines `<nset>, 2, <value>` gave V2
    equal to the value on every node, with V1 = 0 and zero stress.
  - `*Initial Conditions, type=HARDENING` data lines `<element>, <PEEQ>` gave
    frame-0 PEEQ equal to the stated values within float32 rounding (6e-9).

**Element code.** CAX4R is reduced-integration. The `.dat` warns that
`HOURGLASS` on `*Section Controls` "is relevant for … elements with reduced
integration", the npz stores one integration point per element, and ALLAE,
the artificial strain energy, is non-zero. So `under_integrated` is
established for CAX4R from the run's own output. It is not established for
any other code.

**Rigid contact.** A 2D `*Surface, type=SEGMENTS` from `START (x0, 0)` to
`LINE (x1, 0)` with `x1 > x0`, on `*Rigid Body, analytical surface=…`, kept
every node at `y ≥ −4e-15` over the whole run. So its contact side faced `+y`
(npz node coordinates plus U2). A node that carries both a `*Boundary` and
kinematic `*Contact Pair` constraint gets a `.sta` warning
(`WarnNodeBcIntersectKinCon`) saying that the boundary condition overrides
contact on that degree of freedom.

## After the run

Keep, per run folder: the resolved `.inp` and everything it references, the
`.sta`, `.msg` and `.dat`, the `.odb`, and — once it exists — the neutral
intermediate the extractor writes beside the `.odb` (ADR-0068 clause 6),
because an `.odb` alone cannot be opened by a recipient without an Abaqus
licence and so does not satisfy ADR-0040's sharing promise.

### The abaqus-npz/1 intermediate

`odb_export.py` writes `<case_id>.npz` (`np.savez_compressed`, no pickles)
beside the `.odb`. It runs under `abaqus python` and needs no StructBench
install. Its layout (module docstring of `odb_export.py`):

    manifest                                        0-d str: JSON (format, odb_sha256,
                                                    abaqus_release, precision,
                                                    materials, sections, fields, skipped)
    mesh/<instance>/node_labels                     (n,) int64
    mesh/<instance>/node_coords                     (n, 3) float64
    mesh/<instance>/elements/<type>/labels          (e,) int64
    mesh/<instance>/elements/<type>/connectivity    (e, k) int64 node labels
    step/<step>/frame_times                         (T,) float64, step time
    field/<step>/<name>/<instance>/data             (T, m, c) as stored
    field/<step>/<name>/<instance>/node_labels      (m,) int64, or element_labels
    field/<step>/<name>/<instance>/integration_points  (m,) int64, element fields
    history/<step>/<region>/<output>                (s, 2) float64 (time, value)

Established from the conformance exports:
- 2D instances store three coordinate columns, the third zero.
- An analytical rigid surface appears as its own instance (e.g. `WALL`) with
  no field blocks.
- A rigid body's reference node carries no field output unless it is in the
  requested node set.
- The file keeps the end-of-step frame: the last two records share a time,
  and all but the acceleration hold the same values to float32 storage.
- Units are the deck's own; ids are Abaqus labels, never minted by the
  exporter.

Then the dataset's glue reads the three text files through
`structbench.core.io.abaqus_run.read_abaqus_run_evidence` and writes one
whitelisted JSON record, exactly as the LS-DYNA glue does:

    python data_generation/abaqus/<dataset>/collect_run_evidence.py --out runs/datachecks/<name>_run_evidence.json
    python -m structbench.cli.datacheck measure --benchmark <name> --data-root <canonical dir> \
        --run-evidence runs/datachecks/<name>_run_evidence.json --out runs/datachecks/<name>.json

## Open points

These are **not established**. None is a recommendation; each is a question a
sourced dossier and one conformance run must settle (ADR-0068 clause 8).

1. **Energy output (E5), partly settled.** Which keyword requests the global
   energy history, and what the terms are called, is now established (see
   *Abaqus/Explicit* above). Two things are not. First, whether the terms'
   sum reproduces `ETOTAL`: the identity has not been measured. Second,
   whether any term goes uncomputed unless asked for, the way LS-DYNA's
   `*CONTROL_ENERGY` terms do.
2. **What `variable=PRESELECT` actually selects**, for both `*Output, field`
   and `*Output, history`. The Standard job used it for both. The Explicit jobs
   named their variables explicitly and did not use it. Whether it yields
   integration-point data or averages is exactly the distinction LS-DYNA's
   `NINTSLD` open point turns on, and it is still unread.
3. **Field output at the constitutive points (E8), partly settled.** `S` and
   `PEEQ` at the integration point are established for CAX4R. Which state
   variables other materials write is not.
4. **Per-part and per-interface output (E6).** No job has requested it. The
   E7 reaction resultant for a rigid body is established (see above); nodal
   reactions on a `*Boundary` are not.
5. ~~The time-integration record (E4) as a series~~ — settled for Explicit:
   the `.sta` increment table carries the stable increment (see above).
6. **Whether `abaqus python` consumes a licence token.** The interpreter runs
   without one: `abaqus python -c "import sys"` succeeds. Whether `odbAccess`
   checks one out is still unverified; the conformance export did not look.
   It decides whether a recipient needs a seat merely to read an archive.
7. **Element-code semantics, partly settled.** CAX4R is established as
   reduced-integration (see above). `CAX*` and `CPE*` codes are read as
   solid continua (axisymmetric and plane strain), with `under_integrated`
   left unset for every code but CAX4R. Any other family is refused by name.
8. ~~Whether a rejected job's `.dat` always carries the fatal-error count~~ —
   settled: it does not. An Explicit job that fails during the analysis leaves
   no count in the `.dat`; its record is the `.sta` (see above).
9. **What the end-of-step frame's acceleration is.** It differs from frame N's
   at the same instant (see above). Which one is the state at the step's end,
   and whether a field frame's A at a node in kinematic contact is taken
   before or after the contact correction, is not established. Nothing in
   verification reads A.
