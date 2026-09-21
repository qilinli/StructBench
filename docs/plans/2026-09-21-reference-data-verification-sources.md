# Source dossier: published limits and the LS-DYNA realisation of E1–E10

**Date**: 2026-09-21
**Status**: Evidence for maintainer ratification — nothing here is a decision
**Serves**: ADR-0066 clause 7 (an indicator's reference levels cite a
verified external source, as the source states it, scoped by problem class
and formulation) and clause 2 (each solver's way of meeting E1–E10 is checked
against its manual before it is written down).

---

## How this was produced, and how far to trust it

Four research passes fetched primary sources; each was followed by an
independent verification pass that re-fetched every cited source and looked
for the quoted text. **119 claims: 118 confirmed, 1 wording mismatch** (the
HTML title of a web page — the document title itself is confirmed). No quote
was found to be fabricated, paraphrased, or misattributed.

Limits of the reading, to weigh before ratifying anything:

- **NCHRP Web-Only Document 179** was readable only as page *images*
  (no text layer). Every quote from it is a visual transcription, confirmed
  by a second visual reading. Spot-check the page image before ratifying a
  number from it. Its title is "*Procedures* for Verification and Validation
  of Computer Simulations Used for Roadside Safety Applications" — it is
  often mis-cited as "Guidelines for…".
- **Belytschko, Liu, Moran, Elkhodary (2nd ed., Wiley 2014)** was read only
  through Google Books search-inside snippets. Adequate for the quoted
  sentences; not adequate for claims of absence.
- **Abaqus** text comes from university-hosted copies of the vendor's
  *Getting Started* guide (v6.6 and 2016); the vendor's own help site
  refused access.
- **LS-DYNA**: Keyword Manual Vol I R13 and R15, Vol II R15, and Theory
  Manual R16 were downloaded from the vendor and text-searched; support-site
  pages were read from archived captures. Releases R11, R12, and R14 were
  **not** read, and the legacy runs used R12 — defaults can differ by
  release.
- **Not accessible (paywalled or refused)**: ASME V&V 10 / 10.1 / 20, NAFEMS
  guidance, EN 16303 / CEN/TR 16303, the LSTC `energy_balance` FAQ file, the
  Abaqus Analysis User's Guide. Nothing is claimed about their content.

## What the sources say about limits

The sources do **not** agree, and the differences are the finding. They are
kept separate below and must not be merged into a number no source states.

| Source | Energy balance | Zero-energy-mode (hourglass) energy | Added mass |
|---|---|---|---|
| NCHRP W179, Table E-1 (roadside-safety crash FE) | total energy "must not vary more than 10 percent from the beginning of the run to the end of the run"; end-of-run; no external-work term | whole model: < 5 % of total *initial* energy **and** < 10 % of total *internal* energy, both at end of run; worst part: < 10 % of that part's internal energy at end of run | < 5 % of total model mass; worst part < 10 % of its own mass; moving parts < 5 % of moving mass |
| Belytschko et al., eq. (6.2.18) | \|W_kin + W_int − W_ext\| ≤ ε · max(W_ext, W_int, W_kin), ε "generally on the order of 10⁻²"; checked at time steps during the run; purpose: detecting (arrested) instability | ratio to *total* energy; "large" = "on the order of 3 % or 5 %" | no number |
| Abaqus *Getting Started* | total energy "only approximately constant, generally with an error of less than 1 %"; no denominator or evaluation time stated | no universal number in the pages read (worked examples: ~2 % of internal energy "not a problem", ~15 % "substantial") | — (quasi-static: kinetic energy of deforming material typically ≤ 5–10 % of its internal energy) |
| LS-DYNA support site | ratio = total energy / (initial energy + external work), 1.0 = perfect; **no tolerance stated**; Theory Manual R16 states none either | "< 10 % as a rule-of-thumb" of *peak* internal energy, **for each part**; contact energy without friction: 10 % of peak internal energy "might be considered acceptable" | acceptable "if the mass increases remain insignificant"; no number |
| Altair Radioss help | error relative to energy at the start of the run plus external work; positive error "+1 % or +2 % is acceptable", > 2 % must be explained; about −10 % to −15 % "normal" when hourglass energy is excluded from the sum | excluded from the energy error by construction | "recommended to keep the amount of mass added to less than 5 %" |
| Euro NCAP virtual-testing protocols (2023, 2025) | none | *maximum* hourglass energy < 10 % of *maximum* internal energy, whole setup and per dummy | < 5 % of initial model mass (2.5 % for separate models) |

Things the maintainer should know before choosing:

1. **W179 contradicts itself on the per-part hourglass row.** Table E-1 says
   end of run, < 10 % of the part's own internal energy. The report's worked
   example (Table 22) says *at any time during the run*, < 5 % of the model's
   total initial energy. The later NCHRP Report 894 uses the Table 22 wording
   and drops one row. A ratified criterion has to name the variant.
2. **W179's limits are community practice, not derivations.** The hourglass
   ratio is cited to unpublished course notes ("Du Bois, Paul, NOTES, 1998");
   the energy threshold is introduced as "e.g., say 5 or 10 percent". Its
   domain is full-vehicle roadside crash with finite elements.
3. **End-of-run and over-the-run criteria are different quantities.** W179's
   energy row compares the start with the end. Belytschko's is checked
   through the run. A run whose energy rises and then relaxes can pass one
   and fail the other.
4. **Sign matters in some sources and not others.** Radioss treats a small
   *positive* error as the alarming direction (> +2 % must be explained);
   W179 and Belytschko are sign-agnostic.

**How ADR-0066 (Proposed) uses this.** On the maintainer's direction the
energy criterion is a fixed general indicator rather than a single hard
limit: a signed residual in Belytschko's normalised form with the energy
present at the start counted as input, reported as its largest gain, largest
loss, and final value. A source's number becomes a *reference level* only
where its statistic, normalisation, and evaluation time match the quantity:
B-BLM-1 attaches to the over-the-run gain and loss for explicit time
integration; W-W179-02 attaches to the final value for runs with initial
energy and negligible external work; the two vendor statements attach to
nothing and are context. Exceeding a level yields `review`, and the call is
a person's. Extending the same treatment to the hourglass, added-mass, and
contact-energy limits in the table is the drafter's proposal, to be
confirmed.

## Findings that bear on the design

- **MPP LS-DYNA writes these databases in binary only.** Manual Appendix O:
  ASCII output is obtained after the run with the vendor's `l2a` converter;
  the `BINARY` field defaults to 1 (SMP, ASCII) and 2 (MPP, binary), and the
  manual documents no way to force ASCII from MPP. *Unexplained observation*:
  the legacy Taylor MPP run folder holds both an ASCII global-statistics file
  and a binary database. The standard input block should specify `l2a`
  conversion as a post-run step, which keeps the text parsers and avoids a
  binary reader; the conformance run settles what actually happens.
- **Hourglass energy is not computed by default** (`HGEN` default 1;
  2 = "computed and included in the energy balance"). Rigid-wall energy
  defaults to computed; sliding-interface energy is forced on when contact is
  active; Rayleigh damping energy defaults to not computed and, when
  computed, is lumped into internal energy.
- **Solid-element stress in the field file is the element average by
  default** (`NINTSLD` default 1; 8 is required for integration-point
  values). Shell output is averaged in-plane at the element centre unless
  `MAXINT` is negative. This confirms why E8 asks for constitutive-point data
  or a declared reduction.
- **The same output slot can mean four different things.** For the K&C
  concrete model the quantity labelled "plastic strain" is selected by the
  input flag `NOUT` (2 = the scaled damage measure, range 0–2); the default
  was not found. This confirms why E8 asks for the meaning of each state
  variable to be declared rather than inferred.
- **Contact resultant forces are interval-averaged**, not instantaneous —
  relevant to E9 and to any impulse balance.
- **`MAT_ELASTIC_PLASTIC_HYDRO`**: the pressure-hardening card (A1, A2,
  SPALL) is present *only* with the `_SPALL` keyword option; the hardening
  table sits on the following four cards. With the table defined the manual
  gives σ_y = f(ε_p) with no pressure term (the manual does not say in so
  many words that A1/A2 are then ignored).
- **SPH**: `FORM=12` is the moving-least-squares formulation (MPP only);
  `IDIM` 2 = plane strain, −2 = axisymmetric; the `*HOURGLASS` card's Q1/Q2
  *are* the Monaghan artificial-viscosity constants for SPH parts. Whether
  SPH artificial-viscosity work is inside the reported internal energy was
  **not found**.
- **Field-file precision**: per the R15 manual a double-precision executable
  writes 64-bit binary output by default; 32-bit is the opt-in.

## Draft LS-DYNA realisation of E1–E10

Each line cites claim ids in the tables below. "Observed" means seen in a
legacy run file, not in the manual. This is a draft for the standard input
block, to be finalised against the release actually used.

| Item | Setting or source | Status |
|---|---|---|
| E1 | the keyword input, plus `d3hsp` (input printed during the second input phase; `*PARAMETER` values echoed unless `_NOECHO`) [F-C14, F-C17]; `outdeck = s` writes a structured-format resolved deck [F-C21] | whether `d3hsp` echoes `*INCLUDE` content: not established |
| E2 | message-file header (observed); `*DATABASE_FORMAT IBINARY` for output precision [F-C13] | header fields not documented in the manual pages read |
| E3 | `*CONTROL_TERMINATION` ENDTIM / ENDCYC / DTMIN / ENDENG / ENDMAS [L-C36]; message files `messag` (SMP) or one per processor (MPP) [F-C22]; `MSGMAX` so that every message reaches the files [F-C23]; `ISNAN=1` [F-C25]; contact `IGNORE=2` to print penetration warnings [F-C28] | termination-banner text and any warning-count summary: not in the manual (banner observed) |
| E4 | global statistics carry "time step" and "element & part ID controlling time step" [L-C34]; "added mass" appears only if `DT2MS<0` [L-C09]; per part in the part summary [L-C33] | per-part added mass under MPP: old release note says SMP only; current status unverified |
| E5 | `*CONTROL_ENERGY` HGEN=2, RWEN=2, SLNTEN=2, RYLEN=2 (R15 adds DRLEN, DISEN) [L-C02–C07]; `*DATABASE_GLSTAT` [L-C08]; total = internal + kinetic + contact + hourglass + system damping + rigid-wall [L-C11]; ratio = total / (initial + external work) [L-C12]; stiffness damping and solid bulk viscosity are inside internal energy [L-C14, L-C15, L-C17] | balance identity is from the support site, not the manual; SPH artificial viscosity: not found |
| E6 | `*DATABASE_MATSUM`, with `*CONTROL_OUTPUT IERODE=1` for eroded energies per part [L-C21, L-C23]; `*DATABASE_SLEOUT` for per-interface contact energy [L-C20]; part-summary and global kinetic energies legitimately differ [L-C22] | — |
| E7 | `*DATABASE_RWFORC`, `RCFORC` (interval-averaged [L-C24]), `BNDOUT`, `SPCFORC`, `NODFOR` with a nodal force group [L-C20, L-C25, L-C26] | — |
| E8 | `*DATABASE_EXTENT_BINARY`: `NINTSLD=8` [F-C10], negative `MAXINT` [F-C04, F-C05], `STRFLG` for strain tensors [F-C06], `NEIPH`/`NEIPS` for history variables, also SPH [F-C02, F-C03]; the strain written is a time-integrated rate of deformation [F-C08]; `DELERES` for deleted elements [F-C12] | SPH-specific output flags beyond `NEIPH`: none found |
| E9 | the `DT` field of each `*DATABASE_` card; no file without its card; `DT=0` means no output [L-C27] | — |
| E10 | not a solver matter, except that the K&C model's `RSIZE` and `UCF` are built-in unit anchors [F-C36] | — |
| MPP | binary database only; `l2a binout*` after the run for ASCII [L-C28, L-C29] | legacy observation unexplained |

Claim-id prefixes: **W** crash-simulation V&V report and related, **B**
energy-balance criteria, **L** LS-DYNA energy ledger and time integration,
**F** LS-DYNA fields, echo, and diagnostics.

---

## W — Crash-simulation V&V report and related protocols

### W-W179-01 — Document identity

**Claim.** NCHRP Web-Only Document 179 (Project 22-24) is titled 'Procedures for Verification and Validation of Computer Simulations Used for Roadside Safety Applications' (not 'Guidelines for...'); NAP record 17647.

> Procedures for Verification and Validation of Computer Simulations Used for Roadside Safety Applications | The National Academies Press

- Source: *NCHRP Web-Only Document 179: Procedures for Verification and Validation of Computer Simulations Used for Roadside Safety Applications (Ray, Mongiardini, Plaxico, Anghileri)* — National Academies Press record 17647; 2010/2011 (year not read from a title page in this session)
- Locator: HTML page title of NAP reader; onlinepubs.trb.org/onlinepubs/nchrp/nchrp_w179.pdf redirects to this NAP record
- URL: <https://www.nationalacademies.org/read/17647/chapter/5>
- Kind / confidence: primary / high · Independent check: **quote-differs**
- Verifier: The document title is confirmed, but the quoted page-title string is not what the URL serves. I fetched https://www.nationalacademies.org/read/17647/chapter/5 with curl. Its HTML <title> reads: 'Read "Procedures for Verification and Validation of Computer Simulations Used for Roadside Safety Applications" at NAP.edu'. It does not read '... | The National Academies Press'. Page metadata gives citation_title 'Procedures for Verification and Validation of Computer Simulations Used for Roadside Saf…
- Corrected locator: HTML <title> and embedded metadata JSON of https://www.nationalacademies.org/read/17647/chapter/5; publication date 2011-10-05 per NAP metadata; TRB PDF served directly at https://onlinepubs.trb.org/onlinepubs/nchrp/nchrp_w179.pdf
- Caveats: Authors and publication year were not read from a title page (front-matter images were not examined); they come from the task brief/search listing.

### W-W179-02 — Table E-1 row 1: total energy

**Claim.** Total-energy criterion: total energy must not vary more than 10 percent from the beginning of the run to the end of the run. It is worded as beginning-versus-end, not explicitly as a maximum over the run. The energy types are listed as 'kinetic, potential, contact, etc.' and external work is not mentioned.

> Total energy of the analysis solution (i.e., kinetic, potential, contact, etc.) must not vary more than 10 percent from the beginning of the run to the end of the run.

- Source: *NCHRP Web-Only Document 179, Appendix E Validation/Verification Report Forms* — 2010/2011
- Locator: Appendix E, Part II: Analysis Solution Verification, Table E-1 'Analysis Solution Verification Table', printed page E-3 (NAP page image 551), row 1
- URL: <https://nap.nationalacademies.org/books/17647/gif/551.gif>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: Transcribed visually from a page image (no text layer). The table's columns are 'Verification Evaluation Criteria | Change (%) | Pass?'.

### W-W179-03 — Table E-1 row 2: global hourglass vs initial energy

**Claim.** Whole-model hourglass energy at the end of the run must be less than 5 percent of the total initial energy at the beginning of the run.

> Hourglass Energy of the analysis solution at the end of the run is less than five percent of the total initial energy at the beginning of the run.

- Source: *NCHRP Web-Only Document 179, Appendix E* — 2010/2011
- Locator: Table E-1, p. E-3, row 2
- URL: <https://nap.nationalacademies.org/books/17647/gif/551.gif>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: Visual transcription.

### W-W179-04 — Table E-1 row 3: global hourglass vs internal energy

**Claim.** Whole-model hourglass energy at the end of the run must be less than 10 percent of the total internal energy at the end of the run.

> Hourglass Energy of the analysis solution at the end of the run is less than ten percent of the total internal energy at the end of the run.

- Source: *NCHRP Web-Only Document 179, Appendix E* — 2010/2011
- Locator: Table E-1, p. E-3, row 3
- URL: <https://nap.nationalacademies.org/books/17647/gif/551.gif>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: Visual transcription.

### W-W179-05 — Table E-1 row 4: per-part hourglass

**Claim.** In Table E-1 the per-part hourglass criterion is: the part/material with the highest hourglass energy at the end of the run must have it below 10 percent of that part/material's own total internal energy at the end of the run.

> The part/material with the highest amount of hourglass energy at the end of the run is less than ten percent of the total internal energy of the part/material at the end of the run.

- Source: *NCHRP Web-Only Document 179, Appendix E* — 2010/2011
- Locator: Table E-1, p. E-3, row 4
- URL: <https://nap.nationalacademies.org/books/17647/gif/551.gif>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: Conflicts with the same report's Table 22 (see W179-11), which uses a different evaluation time, denominator and limit.

### W-W179-06 — Table E-1 row 5: total added mass

**Claim.** Mass added to the total model must be less than 5 percent of the total model mass at the beginning of the run.

> Mass added to the total model is less than five percent of the total model mass at the beginning of the run.

- Source: *NCHRP Web-Only Document 179, Appendix E* — 2010/2011
- Locator: Table E-1, p. E-3, row 5
- URL: <https://nap.nationalacademies.org/books/17647/gif/551.gif>
- Kind / confidence: primary / high · Independent check: **confirmed**

### W-W179-07 — Table E-1 row 6: per-part added mass

**Claim.** The part/material with the most added mass must have less than 10 percent of its own initial mass added.

> The part/material with the most mass added had less than 10 percent of its initial mass added.

- Source: *NCHRP Web-Only Document 179, Appendix E* — 2010/2011
- Locator: Table E-1, p. E-3, row 6
- URL: <https://nap.nationalacademies.org/books/17647/gif/551.gif>
- Kind / confidence: primary / high · Independent check: **confirmed**

### W-W179-08 — Table E-1 row 7: moving-part added mass

**Claim.** Moving parts/materials must have less than 5 percent of mass added relative to the initial moving mass of the model.

> The moving parts/materials in the model have less than five percent of mass added to the initial moving mass of the model.

- Source: *NCHRP Web-Only Document 179, Appendix E* — 2010/2011
- Locator: Table E-1, p. E-3, row 7
- URL: <https://nap.nationalacademies.org/books/17647/gif/551.gif>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: The rationale is given on p. 100: 'Too much mass added to moving parts will result in a non-physical increase in the initial kinetic energy of the system.'

### W-W179-09 — Table E-1 rows 8-9: shooting nodes, negative volumes

**Claim.** The last two rows are yes/no checks: no shooting nodes, and no solid elements with negative volumes. Table E-1 has exactly nine rows.

> There are no shooting nodes in the solution? / There are no solid elements with negative volumes?

- Source: *NCHRP Web-Only Document 179, Appendix E* — 2010/2011
- Locator: Table E-1, p. E-3, rows 8 and 9
- URL: <https://nap.nationalacademies.org/books/17647/gif/551.gif>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: The quote joins two separate rows with '/'. In the worked example (Table 22) the wording is 'There are shooting nodes in the solution?', answered 'No'.

### W-W179-10 — Mandatory status and exceptions

**Claim.** All Table E-1 criteria must be satisfied, and a failed criterion means the solution cannot be used for verification/validation. Analyst exceptions may be footnoted and explained, and the form has a 'with/without exceptions as noted' checkbox.

> All the criteria listed in Table E-1 must be satisfied. [p.120] ... If any criterion in Table E-1 does not pass one of the verification criterion listed in Table E-1, the analysis solution cannot be used to verify or validate the known solution. If there are exceptions that the analyst things are relevant these should be footnoted in the table and explained below the table. [p. E-3]

- Source: *NCHRP Web-Only Document 179, Chapter 4 Procedures and Appendix E* — 2010/2011
- Locator: Chapter 4, printed p. 120 (image 122); Appendix E p. E-3 (image 551, https://nap.nationalacademies.org/books/17647/gif/551.gif)
- URL: <https://nap.nationalacademies.org/books/17647/gif/122.gif>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: 'things' is a typo in the original ('thinks'). P. 120 also says the table's purpose is to show the solution 'obeyed basic physical laws (e.g., conservation of energy, mass and momentum) and that the solution is numerically stable' and that the analyst 'can certainly add to the list'.

### W-W179-11 — Internal inconsistency: Table 22 vs Table E-1

**Claim.** The report's worked example (Table 22, test case 1) words the per-part hourglass row differently from Table E-1: highest per-part hourglass energy 'at any time during the run' must be below 5 percent of the total initial energy at the beginning of the run. The other eight rows match Table E-1. The example reports total energy change 1.3%, hourglass 0, added mass 0.

> The part/material with the highest amount of hourglass energy at any time during the run is less than five percent of the total initial energy at the beginning of the run.

- Source: *NCHRP Web-Only Document 179, Chapter 6 Benchmark Cases* — 2010/2011
- Locator: Table 22 'Analysis solution verification table for test case 1', printed p. 162 (image 164), row 4
- URL: <https://nap.nationalacademies.org/books/17647/gif/164.gif>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: Visual transcription. The report does not explain the discrepancy, and I did not check the Appendix C filled-in forms to see which variant they use.

### W-W179-12 — Total energy: evaluation window, denominator, external work

**Claim.** The narrative says the maximum change in energy over the run can be reported and frames the threshold as change as a percent of the initial energy and momentum, 'e.g., say 5 or 10 percent'. The worked example treats the model as a closed system with no external work, where total energy should equal the vehicle's initial kinetic energy, and names gravity as a minimal exception. No correction for external work is given.

> Checking the total energy, kinetic energy and momentum of a simulation is quite straightforward in LSDYNA so the maximum change in energy over the simulation run can be reported. Ideally, there should be no change in total energy but as a practical matter, total energy sometimes varies due to a variety of computation affects including hourglass energy, as discussed above, as well as contact and frictional forces. If the change in energy and momentum as a percent of the initial energy and momentum is below some threshold value (e.g., say 5 or 10 percent) then the non-physical errors in the sim…

- Source: *NCHRP Web-Only Document 179, Chapter 2 (verification discussion) and Chapter 6* — 2010/2011
- Locator: Section 'Energy Balance', printed p. 99 (image 101); Chapter 6 'Part II - Solution Verification', printed p. 160 (image 162, https://nap.nationalacademies.org/books/17647/gif/162.gif)
- URL: <https://nap.nationalacademies.org/books/17647/gif/101.gif>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: The Table E-1 row itself says only 'from the beginning of the run to the end of the run', so whether the binding test is end-versus-start or maximum over the run is ambiguous in the source. Figure 54 plots 'Energy / Initial Energy'. 'affects' is as in the original.

### W-W179-13 — Provenance of the hourglass limit

**Claim.** The ratio of hourglass energy to internal energy of about 0.1 is attributed to reference 128, listed as 'Du Bois, Paul, NOTES, 1998' (unpublished course notes). The report also warns that a per-part ratio is not fool-proof, because a large part ID dilutes local hourglassing.

> the energy resulting from these forces should be low compared to the internal energy of the element to ensure reasonable accuracy of the solution, say: h_e / i_e <= lambda ~ 0.1 where h_e is hourglass energy and i_e is internal energy.(128) ... A metric may be developed that defines an acceptable limit of hourglass energy based on the amount of internal energy computed for the overall model and for each individual part. This, however, is not necessarily a fool-proof means of quantifying that hourglass energy is below acceptable limits. ... [Reference list] 128 Du Bois, Paul, NOTES, 1998

- Source: *NCHRP Web-Only Document 179, Chapter 2 and References* — 2010/2011
- Locator: Section 'Energy Balance', printed pp. 98-99 (images 100-101); reference 128 on printed p. 251 (image 253, https://nap.nationalacademies.org/books/17647/gif/253.gif)
- URL: <https://nap.nationalacademies.org/books/17647/gif/100.gif>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: The equation is rendered in ASCII from the typeset original (h_e/i_e ≤ λ ≈ 0.1). The example on p. 99 is a guardrail with the whole w-beam rail in one part ID, giving a small λ even with very high local hourglass forces.

### W-W179-14 — Provenance: community practice and practitioner survey

**Claim.** The report presents its procedures as a formalisation of informal community practice and gives no derivation of the limits. A practitioner survey (Appendix D, Q27) records tolerance bands. For total energy variation 'during the run', 51.6% chose <5% and 41.9% chose 5-10%. For hourglass energy relative to total internal energy, 41.9% chose <5% and 41.9% chose 5-10%. For added mass relative to total physical mass, 64.5% chose <5%.

> The procedures in this document formalize what many in the roadside safety community have been doing informally for a number of years. [p.113] ... Survey results show that as long as the variation in total energy and added mass are less than five percent practitioners (i.e., 52 percent) are generally not concerned about the energy balance and mass increase. Similarly, as long as the ratio between the hourglass energy and total energy is less than 10 percent practitioners (i.e., 42 percent) are generally not concerned. [p.108]

- Source: *NCHRP Web-Only Document 179, Chapter 4 Introduction; Chapter 3 Survey of Modeling Best Practices; Appendix D survey* — 2010/2011
- Locator: Printed p. 108 (image 110) 'Energy Balance and Comparisons'; printed p. 113 (image 115, https://nap.nationalacademies.org/books/17647/gif/115.gif); survey Q27 on survey page 27 (image 542, https://nap.nationalacademies.org/books/17647/gif/542.gif)
- URL: <https://nap.nationalacademies.org/books/17647/gif/110.gif>
- Kind / confidence: primary / medium · Independent check: **confirmed**
- Caveats: The report never states in so many words that the Table E-1 limits were chosen from the survey or from engineering judgment; that link is my inference from adjacent text. The p. 108 summary says 'hourglass energy and total energy' while survey Q27 asks about 'total internal energy', an inconsistency in the source. The survey respondent count is truncated in the page image (about 31). The survey's fourth row covers mass added to a part stationary…

### W-W179-15 — Mass-scaling metric rationale

**Claim.** The report proposes three bases for limiting added mass: percentage of total model mass, percentage of moving mass, and percentage added to individual elements. It requires any exceedance to be reported and justified.

> A metric could be established that limits the amount of mass added in the analysis that is based on: 1) Percentage of the total mass of the model – Typically, the amount of mass added should be small in comparison to the overall mass of the model. 2) Percentage of the "moving" mass of the model – Too much mass added to moving parts will result in a non-physical increase in the initial kinetic energy of the system. 3) Percentage of mass added to individual elements of the model – Abrupt density changes in a mesh due to mass added to individual elements will influence transmission and reflectio…

- Source: *NCHRP Web-Only Document 179, Chapter 2, 'Mass Scaling'* — 2010/2011
- Locator: Printed p. 100 (image 102)
- URL: <https://nap.nationalacademies.org/books/17647/gif/102.gif>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: No numerical values are given in this passage; the 5/10/5 percent values appear only in Table E-1 and Table 22.

### W-FHWA-01 — FHWA adoption

**Claim.** FHWA's roadside-hardware FAQ adopts the W179 / 22-24 verification and validation report by reference for finite element analysis submitted for Federal-aid eligibility. The page states no numerical limits of its own.

> In addition, any Finite Element Analyses that are submitted to FHWA as part of the documentation package for determining eligibility for reimbursement under the Federal-aid highway program should be accompanied by a Verification and Validation Report as detailed in NCHRP Report 22-24.

- Source: *FHWA, 'FAQs: Barriers, Terminals, Transitions, Attenuators, and Bridge Railings'* — Wayback Machine capture 2025-01-10
- Locator: FAQ text; sentences containing 'Finite Element'
- URL: <https://web.archive.org/web/20250110153738id_/https://highways.dot.gov/safety/rwd/reduce-crash-severity/faqs-barriers-terminals-transitions-attenuators-and-bridge-railings>
- Kind / confidence: primary / medium · Independent check: **confirmed**
- Corrected locator: FAQ question 'MAY COMPUTER MODELING BE USED AS A SUBSTITUTE FOR FULL-SCALE CRASH TESTING OF NEW DEVICES?' (FAQs are unnumbered in the capture)
- Caveats: The live highways.dot.gov page returned HTTP 403 to both WebFetch and curl, so I read the archived copy. The same page also says: 'Manufacturers developing new hardware are encouraged to use Finite Element Analysis (ie: LS-DYNA) ... using the Verification and Validation process as detailed in NCHRP Web-Only Document 179'. The sentences were extracted by keyword; I did not record the surrounding FAQ question number.

### W-RSVVP-01 — RSVVP documentation

**Claim.** The RSVVP User's Manual Rev. 1.4 (38 pages) contains no solution-verification criteria. A full-text search for 'hourglass', 'added mass', 'total energy' and 'shooting' returned no hits. RSVVP covers curve-comparison metrics only.

> Roadside Safety Verification and Validation Program (RSVVP) User's Manual Worcester Polytechnic Institute (WPI) December 2008 (Rev. 1.4) Malcolm H. Ray Mario Mongiardini

- Source: *Roadside Safety Verification and Validation Program (RSVVP) User's Manual* — Rev. 1.4, December 2008
- Locator: Title page; whole-document keyword search of the text layer
- URL: <https://roadsafellc.com/NCHRP22-24/QPR/AttachmentD-7.pdf>
- Kind / confidence: primary / medium · Independent check: **confirmed**
- Caveats: This is a negative finding for the one version I checked, a project-attachment copy. Later manual revisions were not checked.

### W-R894-01 — Later NCHRP use of the table (variant)

**Claim.** NCHRP Report 894 Appendix C ('CCSA Validation/Verification Report') uses an 8-row 'Analysis Solution Verification Summary'. It has the same 10% total-energy row and the 5%-of-initial-energy global hourglass row. It omits the row for global hourglass below 10% of internal energy. For the per-part row it uses the Table-22 wording (any time during the run, below 5% of total initial energy). The added-mass rows (5% / 10% / 5%) and the shooting-node and negative-volume rows are unchanged.

> Total energy of the analysis solution (i.e., kinetic, potential, contact, etc.) must not vary more than 10 percent from the beginning of the run to the end of the run. ... Hourglass Energy of the analysis solution at the end of the run is less than 5 % of the total initial energy at the beginning of the run ... The part/material with the highest amount of hourglass energy at any time during the run is less than 5 % of the total initial energy at the beginning of the run. ... Mass added to the total model is less than 5 % the total model mass at the start of the run. ... The part/material with…

- Source: *NCHRP Report 894, Appendix C: Finite Element Model Validations (CCSA Longitudinal Barriers on Curved, Superelevated Roadway Sections)* — PDF created 2017-02-28 (file metadata); report publication year not verified
- Locator: PDF page 4 (printed C-4), 'Table C – Analysis Solution Verification Summary'; repeated on p. C-17 and five other pages (7 instances)
- URL: <https://onlinepubs.trb.org/onlinepubs/nchrp/nchrp_rpt_894AppendixC.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: Extracted from the PDF text layer. This is a user's application of W179, not an official revision of it.

### W-ENCAP-01 — Euro NCAP virtual testing quality criteria (frontal/crash protection)

**Claim.** Euro NCAP requires, for each simulation: maximum hourglass energy of the full setup below 10% of maximum internal energy; maximum hourglass energy of all dummy components below 10% of maximum internal energy for each dummy; and maximum mass added by mass scaling below 5% of total model mass at the beginning of the run (2.5% for separate models). There is no total-energy-change criterion. The hourglass/internal ratio at the time of maximum head excursion and the per-subsystem added mass are monitored with no limit defined.

> The simulation results should meet the following quality criteria for each simulation: − Max. Hourglass Energy of full setup must be < 10% of max. internal energy. − Max. Hourglass Energy of all dummy components must be < 10% of max. internal energy for each dummy. − Max. mass added due to mass scaling to the total model is less than 5 % (2.5% in case of separate models) of the total model mass at the beginning of the run. − Less than 10 mm H-point z-displacement recorded in first 5 ms of the simulation (5 ms after t0). The following parameters are monitored and therefore calculated on the VT…

- Source: *Euro NCAP Crash Protection Virtual Testing Protocol* — Version 1.0.2, December 2025 (implementation January 2026)
- Locator: Section 5 'Qualification of Simulation Model', 'Quality criteria of the simulation set-up', printed p. 13 (PDF p. 14)
- URL: <https://cdn.euroncap.com/cars/assets/euro_ncap_supporting_protocol_crash_protection_virtual_testing_v102_ca63155dc4.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: The definition is maximum over maximum (maximum hourglass divided by maximum internal energy over the run), which differs from the end-of-run ratios in W179. The scope is sled/occupant virtual testing, not full-vehicle structural crash.

### W-ENCAP-02 — Euro NCAP virtual far side protocol quality criteria

**Claim.** The 2023 far-side protocol sets the same limits: maximum hourglass energy of the full setup below 10% of maximum internal energy; WSID dummy components below 10% of the maximum internal energy of the WSID; added mass below 5% of total model mass at the beginning of the run. It also requires simulation time beyond the time of maximum head y-displacement plus 20%. All quality criteria must pass or the dossier is rejected.

> The simulation set-ups are only accepted for virtual testing if all quality criteria and validation criteria for all load cases are fulfilled. ... 6.1.2 Quality Criteria • Max. Hourglass Energy of full setup must be < 10% of max. internal energy. • Max. Hourglass Energy of all WSID components must be < 10% of max. internal energy of WSID. • Max. mass added due to mass scaling to the total model is less than 5 % of the total model mass at the beginning of the run. • Less than 10 mm H-point z-displacement recorded in first 5 ms of the simulation (5 ms after t0). • The simulation time needs to e…

- Source: *Euro NCAP Virtual Far Side Simulation & Assessment Protocol* — Version 1.0, 15 June 2023 (implementation 2024)
- Locator: Section 6.1.1-6.1.3, printed p. 12 (PDF p. 16)
- URL: <https://cdn.euroncap.com/cars/assets/euro_ncap_vtc_simulation_and_assessment_protocol_v10_f13aa41e72.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**

**Verifier summary.** 18 of the 19 claims are confirmed against the primary sources; the one exception is a wording mismatch in W179-01, and no quote was fabricated or misattributed. I re-fetched every source into the scratchpad folder `verify-crash-vv-report`. I read the eleven NAP page images for W179 visually, since they have no text layer. For the RSVVP manual, NCHRP Report 894 Appendix C and the two Euro NCAP protocols I extracted the PDF text with pypdf. The FHWA FAQ came from the Wayback capture, which returned HTTP 429 once and then 200. I did not try the live FHWA page. **W179-01 (quote-differs).** The report title is confirmed, and NAP metadata gives a publication date of 2011-10-05 and DOI 10.17226/17647. Two details in the claim are wrong: - The page's HTML title is 'Read "Procedures for Verification and Validation of Computer Simulations Used for Roadside Safety Applications" at NAP.edu'. It is not '... | The National Academies Press'. - The TRB URL `nchrp_w179.pdf` does not redirect to NAP. It served the PDF directly (HTTP 200, application/pdf, 10.7 MB). **Points to carry into ratification:** - **Table E-1 vs Table 22 (W179-05, W179-11).** Table E-1 row 4 says per-part hourglass energy at end of run must be below 10% of that part's internal energy. Table 22 says per-part hourglass energy at any time during the run must be below 5% of total initial energy. Both wordings are in the report. Table 22 also drops the 'no' from the two yes/no rows. Report 894 uses the Table 22 variant and omits the row for global hourglass below 10% of internal energy, so it has 8 rows across 7 table instances. - **Hourglass ratio provenance (W179-13).** The source states the ratio of about 0.1 per element ('internal energy of the element') and cites 'Du Bois, Paul, NOTES, 1998'. 'Unpublished course notes' is the researcher's gloss, not the source's words. - **Added-mass bases (W179-15).** The third basis on p. 100 is per individual element. Table E-1 row 6 is per part/material. - **Negative claims.** 'No derivation of the limits' (W179-14) and 'no total-energy criterion' (ENCAP-01) hold for the pages I read and for a keyword search of the full 17-page text. They are not whole-report readings…

**Not found or not accessible.**

- A text-layer PDF of NCHRP W179: COULD NOT ACCESS. onlinepubs.trb.org/onlinepubs/nchrp/nchrp_w179.pdf redirects to the NAP catalog page (HTML), and the NAP PDF download sits behind a login or challenge. No OCR tool was available locally (no tesseract, pytesseract or easyocr). I therefore read the NAP per-page GIF images visually, so all W179 quotes are visual transcriptions and should be spot-checked against the imag…
- Whether the Appendix C filled-in validation forms in W179 (test cases 1-4) use the Table E-1 wording or the Table 22 wording for the per-part hourglass row: NOT CHECKED. Only Table E-1 (p. E-3) and Table 22 (p. 162) were read.
- An explicit statement in W179 that the specific Table E-1 numbers (10 / 5 / 10 / 10 / 5 / 10 / 5 percent) were set by engineering judgment, panel consensus or the survey: NOT FOUND in the pages read (98-100, 108, 113, 120-121, 160, 162, E-3, survey p. 27). I found only indirect provenance: 'formalize what many ... have been doing informally', 'e.g., say 5 or 10 percent', the Du Bois 1998 notes for the 0.1 ratio, and…
- An AASHTO document adopting or reproducing the table: NOT FOUND. A search turned up only FHWA pages. The FHWA eligibility memo page (highways.dot.gov/.../memorandum-federal-aid-reimbursement-eligibility-process-safety) returned HTTP 403 live, and its archived copy contained no finite-element or 22-24 sentence by keyword search. Only the FHWA FAQ page, read via the Wayback Machine, references W179 / 22-24, and it giv…
- EN 16303:2020 and CEN/TR 16303-1..4 (2012), covering energy, hourglass and added-mass limits for virtual testing of vehicle restraint systems: COULD NOT ACCESS. These are paywalled standards (BSI, SIS, iTeh and GlobalSpec listings only, and the iTeh page returned no preview content). A search-engine summary attributed limits of 'hourglass <5% of total initial energy, <10% of internal energy at end, total energy chan…
- IIHS or NHTSA published numerical simulation-quality checklists with energy, hourglass or mass-scaling limits: NOT FOUND. The ESV paper on Euro NCAP virtual testing (www-esv.nhtsa.dot.gov/Proceedings/27/27ESV-000284.pdf) returned Access Denied (403). Euro NCAP Technical Bulletins CP 510/520 (dummy model qualification) and the v1.0 (March 2025) protocol URL (404) were not read.
- W179 publication year, authors and report number as printed on its title page: not read in this session, because the front-matter images R1-R9 were not examined. Treat '2010/2011' as unverified.

## B — Energy-balance, hourglass, and mass-scaling criteria in other authoritative sources

### B-BLM-1 — Belytschko energy balance inequality and tolerance

**Claim.** The book's energy balance check for explicit integration is |Wkin + Wint - Wext| <= eps * max(Wext, Wint, Wkin), numbered (6.2.18) in the 2nd edition, with eps a small tolerance 'generally on the order of 10^-2'. Normalisation is the maximum of external work, internal energy and kinetic energy. For very large systems (order 10^5 nodes or more) the balance should be done on subdomains.

> Energy conservation requires that | Wkin + Wint − Wext | ≤ε max. (. Wext , Wint , Wkin. ) (6.2.18) where e is a small tolerance, generally on the order of 10–2. If the system is very large, on the order of 105 nodes or larger, the energy ... [second snippet:] ... tolerance, generally on the order of 10–2. If the system is very large, on the order of 105 nodes or larger, the energy balance should be performed on subdomains of the model. The internal forces from adjacent subdomains are then ...

- Source: *Belytschko, Liu, Moran, Elkhodary - Nonlinear Finite Elements for Continua and Structures, 2nd edition (Wiley)* — 2nd ed., 2014 (page proofs dated 10/5/2013 in snippet)
- Locator: p. 336, Section 6.2.3 'Energy Balance', eq. (6.2.18)
- URL: <https://books.google.com/books?id=e_w8AgAAQBAJ&q=%22energy+conservation+requires%22&jscmd=SearchWithinVolume2>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: Read as Google Books search-inside SNIPPETS of the actual book text, not the full page. OCR artifacts: '10–2' is 10^-2, '105' is 10^5, 'e' is epsilon, stray '. (' punctuation. The snippet does not say whether the check is at end of run or every step; see BLM-2 for the flowchart. Equation number may differ in the 1st edition (not checked).

### B-BLM-2 — Belytschko energy balance - when evaluated and purpose

**Claim.** The energy balance is checked each time step within the explicit flowchart (Box 6.1), and its stated purpose is detecting instabilities (incl. 'arrested instabilities') that are not visible in results; instability shows up as spurious energy generation.

> Check energy balance at time step n + 1 : see ( 6.2.14–18 ) ... [p.335:] arrested instabilities can lead to a large overprediction of displacements, but they are not detectable by perusing the results. However, they can easily be detected by an energy balance check. Any instability results in the spurious ... [p.336:] ... spurious generation of energy which leads to a violation of the conservation of energy. Therefore, whether stability was maintained during a nonlinear computation can be established by checking energy balance. In low-order methods like the central difference method, the ener…

- Source: *Belytschko, Liu, Moran, Elkhodary - Nonlinear Finite Elements for Continua and Structures, 2nd edition (Wiley)* — 2nd ed., 2014
- Locator: p. 333 (Box 6.1 flowchart, step 11), pp. 335-336 (Section 6.2.3)
- URL: <https://books.google.com/books?id=e_w8AgAAQBAJ&q=check+on+energy&jscmd=SearchWithinVolume2>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Corrected locator: p. 333 Box 6.1 step 11; p. 335 (end) to p. 336, Section 6.2.3
- Caveats: Snippets only; three separate snippets concatenated with '...' markers. The book frames this as a stability check, not an accuracy acceptance criterion.

### B-BLM-3 — Belytschko hourglass energy fraction

**Claim.** The book says the ratio of hourglass energy to TOTAL energy should be monitored; 'large' means on the order of 3% or 5%, in which case results are in error by the same order. For large problems, monitor on subdomains.

> The ratio of the hourglass energy to the total energy should be monitored. If it is large (on the order of 3% or 5%), the results will be in error by the same order of magnitude. If the problem is large, the hourglass energy should be monitored on subdomains, just as we recommended for the energy balance in Section 6.2.3 ...

- Source: *Belytschko, Liu, Moran, Elkhodary - Nonlinear Finite Elements for Continua and Structures, 2nd edition (Wiley)* — 2nd ed., 2014
- Locator: p. 523, Section 8.7 (hourglass/stabilization), near eq. (8.7.14)
- URL: <https://books.google.com/books?id=e_w8AgAAQBAJ&q=%22energy+to+the+total+energy+should+be+monitored%22&jscmd=SearchWithinVolume2>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Corrected locator: p. 523 (Section 8.7 inferred from eq. number (8.7.10) cited in the adjacent sentence; '(8.7.14)' not verified)
- Caveats: Snippets only (two overlapping snippets joined). Denominator is 'total energy' (not internal energy, not peak); timing (end vs over run) not stated.

### B-BLM-4 — Belytschko mass scaling

**Claim.** The book gives no numeric added-mass limit; it says mass scaling should be used where high-frequency effects are unimportant and is not recommended where high-frequency response matters.

> Mass scaling should be used for problems where high frequency effects are not important. For example, in sheet-metal forming, which is essentially a static process, it causes no difficulties. On the other hand, if high frequency response is important, mass scaling is not recommended ...

- Source: *Belytschko, Liu, Moran, Elkhodary - Nonlinear Finite Elements for Continua and Structures, 2nd edition (Wiley)* — 2nd ed., 2014
- Locator: p. 337, Section 6.2.x 'Mass Scaling, Subcycling and Dynamic Relaxation'
- URL: <https://books.google.com/books?id=e_w8AgAAQBAJ&q=%22Mass+scaling+should+be+used+for+problems%22&jscmd=SearchWithinVolume2>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Corrected locator: p. 337, Section 6.2.5 'Mass Scaling, Subcycling and Dynamic Relaxation'
- Caveats: Snippet only.

### B-ABQ-1 — Abaqus/Explicit ETOTAL constancy

**Claim.** Abaqus Getting Started guide states the whole-model energy balance ETOTAL should be constant and in the numerical model is only approximately constant, 'generally with an error of less than 1%'. No denominator and no end-of-run vs over-run qualifier is given. Identical sentence in v6.6 (Explicit keywords guide 3.7.1) and in 2016 (Abaqus/CAE guide 9.6 and 13.4).

> An energy balance for the entire model can be written as ... The sum of these energy components is , which should be constant. In the numerical model is only approximately constant, generally with an error of less than 1%.

- Source: *Getting Started with ABAQUS/Explicit: Keywords Version (v6.6), Section 3.7.1 'Statement of energy balance'; same text in Getting Started with Abaqus/CAE (2016) Sections 9.6 and 13.4* — Abaqus 6.6 (2006) and Abaqus 2016
- Locator: Section 3.7.1 (v6.6); 2016 copy: https://ceae-server.colorado.edu/v2016/books/gsa/ch09s06.html Section 9.6
- URL: <https://web.archive.org/web/20170614214554id_/http://classes.engineering.wustl.edu/2009/spring/mase5513/abaqus/docs/v6.6/books/gsx/ch03s07.html>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Corrected locator: v6.6 GSX Section 3.7.1; 2016 GSA Section 9.6.1 (within 9.6) and Section 13.4
- Caveats: University-hosted copies of the vendor documentation (WUSTL via Wayback; Univ. Colorado), not help.3ds.com. Equation symbols (E_I, E_total etc.) are images and dropped from the extracted text, hence the gaps in the quote. The '1%' is descriptive ('generally'), not phrased as an acceptance criterion; normalisation not stated.

### B-ABQ-2 — Abaqus artificial strain energy (ALLAE) vs internal energy (ALLIE)

**Claim.** In the pages read, Abaqus gives no universal numeric threshold. General text: large artificial strain energy means mesh changes are needed. Worked examples compare whole-model ALLAE to whole-model ALLIE: ~2% is judged 'not a problem' (blast-loaded plate); ~15% is 'a substantial fraction' whose reduction would improve the solution (circuit board drop).

> [9.6] Large values of artificial strain energy indicate that mesh refinement or other changes to the mesh are necessary. [10.5] Such a variable is the total internal energy, ALLIE, which is a summation of all internal energy quantities. The artificial strain energy is approximately 2% of the total internal energy, indicating that hourglassing is not a problem. [12.11] Another important energy output variable is the artificial energy, which is a substantial fraction (approximately 15%) of the internal energy in this analysis. By now you should know that the quality of the solution would improv…

- Source: *Getting Started with Abaqus/CAE (2016)* — Abaqus 2016
- Locator: Section 10.5 (blast loading on a stiffened plate); Section 12.11 (circuit board drop test) at .../gsa/ch12s11.html; Section 9.6 at .../gsa/ch09s06.html
- URL: <https://ceae-server.colorado.edu/v2016/books/gsa/ch10s05.html>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: University-hosted copy. Denominator = ALLIE, whole model; comparison made on history plots (over the run), no explicit end/peak rule. The widely repeated 'ALLAE < 1-5-10% of ALLIE' rule was only seen in secondary web results (forums, a search-engine summary) and NOT in vendor text I read - treat as unverified.

### B-ABQ-3 — Abaqus quasi-static KE/IE criterion (relevant to mass scaling acceptance)

**Claim.** For quasi-static explicit analyses Abaqus says the kinetic energy of the deforming material should not exceed typically 5% to 10% of its internal energy throughout most of the process; the forming example tightens this to 'a few percent'. Mass scaling acceptability is judged by the same approach as load-rate scaling; no percent-added-mass limit is given.

> As a general rule the kinetic energy of the deforming material should not exceed a small fraction (typically 5% to 10%) of its internal energy throughout most of the process. [13.5] To determine whether an acceptable quasi-static solution has been obtained, the kinetic energy of the blank should be no greater than a few percent of its internal energy. [13.3] Therefore, excessive mass scaling, just like excessive loading rates, can lead to erroneous solutions. The suggested approach to determining an acceptable mass scaling factor, then, is similar to the approach to determining an acceptable …

- Source: *Getting Started with Abaqus/CAE (2016)* — Abaqus 2016
- Locator: Section 13.4 'Energy balance'; Section 13.5 (.../ch13s05.html) 'Strategy for evaluating the results'; Section 13.3 (.../ch13s03.html) 'Mass scaling'
- URL: <https://ceae-server.colorado.edu/v2016/books/gsa/ch13s04.html>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Corrected locator: 13.4 'Energy balance' (ch13s04.html); 13.5 'Example: forming a channel in Abaqus/Explicit' (ch13s05.html); 13.3 mass scaling (ch13s03.html)
- Caveats: Applies to QUASI-STATIC use of explicit, not to true transient dynamics. 'Deforming material' = per deforming part (the blank), over the run ('throughout most of the process').

### B-LSD-1 — LS-DYNA energy ratio definition

**Claim.** LS-DYNA support defines the glstat energy ratio as total energy / (initial energy + external work), whole model, equal to 1.0 for perfect balance. No tolerance band around 1.0 is stated on this page.

> The energy balance is perfect if total energy = initial total energy + external work, or in other words if the energy ratio (referred to in GLSTAT as total energy / initial energy although it actually is total energy / (initial energy + external work)) is equal to 1.0.

- Source: *LS-DYNA support knowledge base - 'Total energy' (Ansys/DYNAmore)* — web page, fetched 2026-09-21
- Locator: paragraph beginning 'The energy balance is perfect if'
- URL: <https://lsdyna.ansys.com/total-energy-2/>
- Kind / confidence: vendor-support / high · Independent check: **confirmed**
- Caveats: Same text as dynasupport.com/howtos/general/total-energy (confirmed via WebFetch summary; direct download from dynasupport.com timed out). Total energy here includes hourglass, contact, damping and rigidwall energies only if HGEN/RWEN/RYLEN=2.

### B-LSD-2 — LS-DYNA hourglass energy rule of thumb

**Claim.** LS-DYNA support: hourglass energy should be <10% (rule of thumb) of PEAK internal energy, judged FOR EACH PART (matsum), with HGEN=2.

> To evaluate hourglass energy, set HGEN to 2 in *CONTROL_ENERGY and use *DATABASE_GLSTAT and *DATABASE_MATSUM to report the HG energy for the system and for each part, respectively. The point is to confirm that the nonphysical HG energy is small relative to peak internal energy for each part (<10% as a rule-of-thumb).

- Source: *LS-DYNA support knowledge base - 'Hourglass' (Ansys/DYNAmore)* — web page, fetched 2026-09-21
- Locator: Section 'Notice', paragraph beginning 'To evaluate hourglass energy'
- URL: <https://lsdyna.ansys.com/hourglass/>
- Kind / confidence: vendor-support / high · Independent check: **confirmed**
- Caveats: Denominator = peak internal energy of that part over the run; numerator timing (peak vs final HG energy) not specified. Written in first person by a support engineer; it is a rule of thumb, not a manual requirement.

### B-LSD-3 — LS-DYNA contact (sliding) energy without friction

**Claim.** LS-DYNA support: absent friction, net contact energy of about 10% of peak internal energy 'might be considered acceptable'; 'small is a matter of judgement'.

> In the absence of friction, you would hope to see a small net contact energy (net = sum of slave side energy and master side energy). Small is a matter of judgement – 10% of peak internal energy might be considered acceptable for contact energy in the absence of contact friction.

- Source: *LS-DYNA support knowledge base - 'Total energy' (Ansys/DYNAmore)* — web page, fetched 2026-09-21
- Locator: Section 'Positive contact energy:'
- URL: <https://lsdyna.ansys.com/total-energy-2/>
- Kind / confidence: vendor-support / high · Independent check: **confirmed**
- Caveats: Quotation marks/dashes normalised from mis-encoded characters in the page. Whole-model vs per-contact not stated (sleout gives per-contact).

### B-LSD-4 — LS-DYNA energy equation and interpretation (User's Guide tutorial)

**Claim.** The LS-DYNA user's-guide tutorial says the energy equation should hold at all times during the analysis (i.e. over the run), and that deviation indicates an error; it gives no numeric tolerance. The energy ratio can be used as a termination criterion via ENDENG.

> The following equation should hold at all times during an analysis. ... If the equation does not hold the user should suspect an error. If the left hand side of the equation rises above the right hand side, energy is being introduced artificially – for example, by numerical instability, or the sudden detection of artificial penetration through a contact surface ... If the left hand side falls below the right hand side, energy is being absorbed artificially, perhaps by excessive hourglassing or by stonewalls or over-compliant contact surfaces. ... This energy ratio may be used as a criterium f…

- Source: *LS-DYNA support - Tutorial / LS-DYNA User's Guide - 'Energy data' (Ansys/DYNAmore)* — web page, fetched 2026-09-21
- Locator: whole page
- URL: <https://lsdyna.ansys.com/energy-data-2/>
- Kind / confidence: vendor-support / medium · Independent check: **confirmed**
- Caveats: The equation itself and the energy-ratio formula are IMAGES on the page and could not be read; only the surrounding text was read.

### B-LSD-5 — LS-DYNA Theory Manual - energy balance

**Claim.** The LS-DYNA Theory Manual R16 contains NO numerical energy-ratio/energy-balance tolerance for explicit analyses. A full-text search for 'energy ratio' returned no hits; 'energy balance' appears only in passing and in Section 38.4.2 (Implicit), which gives qualitative checks only.

> In sum, there are three checks that can be made 1.|Dk| ≪ Ek → Time integration scheme is energy conserving 2.|Di| ≪ Ei → Internal forces are trustworthy 3.Ek + Dk + Ei + Di + Wd ≈ We → Numerical problem solved accurately enough

- Source: *ANSYS LS-DYNA Theory Manual R16* — R16@e545952c7 (03/21/25)
- Locator: Section 38.4.2 'Energy balance', p. 38-24 (PDF page 796)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2025/04/LS-DYNA_Manual_Theory_R16.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Corrected locator: Section 38.4.2 starts p. 38-23 (PDF p. 795); the three checks are on p. 38-24 (PDF p. 796)
- Caveats: The quoted checks are for IMPLICIT dynamics (Newmark) and use '≪' / '≈' with no numbers. Absence claim is based on text search of the 902-page PDF with PyMuPDF for 'energy ratio', 'energy balance', 'hourglass energy'.

### B-LSD-6 — LS-DYNA termination controls: ENDENG, ENDMAS, DTMIN (time-step collapse)

**Claim.** The keyword manual provides user-set terminations but recommends no values: ENDENG = percent change in energy ratio (inactive if undefined, default 0.0); ENDMAS = percent change in total mass (only with DT2MS mass scaling); DTMIN sets tsmin = dtstart x DTMIN and LS-DYNA terminates with a restart dump when the time step drops to tsmin (or erodes elements if ERODE=1).

> DTMIN Reduction (or scale) factor to determine minimum time step, tsmin, where tsmin = dtstart × DTMIN and dtstart is the initial step size determined by LS-DYNA. When the time step drops to tsmin, LS-DYNA terminates with a restart dump. ... ENDENG Percent change in energy ratio for termination of calculation. If undefined, this option is inactive. ENDMAS Percent change in the total mass for termination of calculation. This option is relevant if and only if mass scaling is used to limit the minimum time step size; see *CONTROL_TIMESTEP field DT2MS.

- Source: *ANSYS LS-DYNA Keyword User's Manual Volume I, R16* — R16@431ab7b9b (10/29/25)
- Locator: *CONTROL_TERMINATION, p. 12-553 to 12-554 (PDF pages 1867-1868); Remark 2 'Erosion'
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Vol_I_R16.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: Defaults in the card table: DTMIN 0.0, ENDENG 0.0, ENDMAS printed as '108' in extracted text (almost certainly 10^8 with lost superscript - not verified visually). These are mechanisms, not acceptance criteria.

### B-LSD-7 — LS-DYNA mass scaling acceptability

**Claim.** LS-DYNA gives no numeric added-mass limit. Keyword manual: negative DT2MS 'can be used in transient analyses if the mass increases remain insignificant'. Support page: justified e.g. for a few small elements in a noncritical area or quasi-static runs with KE very small relative to peak internal energy; otherwise analyst judgement and a sensitivity re-run. Added mass is reported whole-model (glstat) and per part (matsum).

> [Keyword manual, DT2MS LT.0.0] This option can be used in transient analyses if the mass increases remain insignificant. See also the variable MS1ST below and the *CONTROL_TERMINATION variable ENDMAS. [Support page] Anytime you add nonphysical mass to increase the timestep in a dynamicanalysis, you affect the results (think of F=m*a). Sometimes the effect is insignificant and in those cases adding nonphysical mass is justifiable. Examples of such cases may include the addition of mass to just a few small elements in a noncritical area or quasi-static simulations where the velocity is low and …

- Source: *ANSYS LS-DYNA Keyword User's Manual Vol I R16 (*CONTROL_TIMESTEP, DT2MS) and LS-DYNA support knowledge base 'Mass scaling'* — R16 (10/29/25); web page fetched 2026-09-21
- Locator: Keyword manual: *CONTROL_TIMESTEP variable DT2MS, PDF page 1894 (https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Vol_I_R16.pdf); support page: first two paragraphs
- URL: <https://lsdyna.ansys.com/mass-scaling/>
- Kind / confidence: vendor-support / high · Independent check: **confirmed**
- Caveats: Two sources in one claim, both quoted separately and labelled. Typos ('dynamicanalysis', 'gage the affect') are in the original.

### B-RAD-1 — Radioss energy error definition

**Claim.** Radioss %Error = 100 * ( (Ek + Ekr + Ei) / (Ek,1 + Ekr,1 + Ei,1 + Ewk - Ewk,1) - 1 ), whole model, printed every output cycle, referenced to energies at the beginning of the RUN (not t=0), bounded to +-99%, reset at each restart. Hourglass energy (and contact energy) is NOT counted, so a negative error is normal with under-integrated elements.

> E , 1 Energy at beginning of the RUN (not at time t=0) The Hourglass energy is not counted in this energy balance, so that a negative energy error generally occurs (except if using Ishell=24 (QEPH) or Ishell=12 (BATOZ) shells, and fully integrated solids, for which there is no Hourglass). %Error is bounded to ± 99%. The energy error is reset after each RESTART.

- Source: *Altair Radioss Help - FAQ 'Results Checking'* — current online help (help.altair.com/hwsolvers), fetched 2026-09-21
- Locator: 'Energy Error calculation.'
- URL: <https://help.altair.com/hwsolvers/rad/topics/solvers/rad/faq_rad_results_checking_r.htm>
- Kind / confidence: vendor-support / medium · Independent check: **confirmed**
- Caveats: The formula is MathML; I reconstructed it from the token stream '% Error = 100 ( E k + E k r + E i E k,1 + E k,1 r + E i,1 + E w k − E w k,1 − 1 )' - the fraction bar position is inferred (numerator Ek+Ekr+Ei; denominator the rest). Verify visually before quoting the formula. The prose quote is verbatim.

### B-RAD-2 — Radioss acceptable energy error magnitudes

**Claim.** Radioss guidance: negative error from hourglass 'normal' at about -10% to -15%; positive error of +1% or +2% acceptable; positive error >2% must have its source identified; error growing to +-99% can indicate divergence. Exceptions: low initial energy early in run, a single diverging part, large frictional contact energy.

> The normal amount of Hourglass energy is about -10% to -15%. If the error is positive, there is an energy creation.In case of using QEPH shell formulation (Ishell=24) or fully integrated elements, the Energy Error can be slightly positive since there is no Hourglass energy and the computation is much more accurate. An error of +1% or +2% is acceptable If the positive energy error is greater than 2%, the source of this energy has to be identified. Incompatible kinematic conditions can lead to such a situation. An increasing Energy Error that reaches ±99% can indicate the simulation has diverge…

- Source: *Altair Radioss Help - FAQ 'Results Checking'* — current online help, fetched 2026-09-21
- Locator: 'Recommendation of Energy Error.'
- URL: <https://help.altair.com/hwsolvers/rad/topics/solvers/rad/faq_rad_results_checking_r.htm>
- Kind / confidence: vendor-support / high · Independent check: **confirmed**
- Caveats: Whole model; the -10% to -15% is expressed as an energy-ERROR percentage (relative to the Radioss denominator in RAD-1), not as HG/internal energy. Page also says a large end-of-run error can be caused by 'only one part diverging'. Missing full stop after 'acceptable' is in the extracted text. OpenRadioss shares this Altair documentation; no separate OpenRadioss statement was found.

### B-RAD-3 — Radioss added mass limit

**Claim.** Radioss recommends keeping added mass below 5% in general (larger may be acceptable e.g. quasi-static), measured as mass error DM/M0 = (M - M0)/M0 for the whole model (MAS.ERR column) with M0 reset at each Engine file; local nodal added mass should also be checked (/ANIM/NODA/DMAS). Added mass that adds KE shows as positive energy error.

> Good engineering judgement must be used to determine how much mass is an acceptable amount to be added to a model. ... In general, it is recommended to keep the amount of mass added to less than 5%. However, larger mass increases may be acceptable in some types of simulation. For example, in quasi-static simulations the velocities are usually small, so adding mass does not greatly increase the kinetic energy. For those reasons, it is recommended to check the mass increase in the model by running a simulation without or with reduced mass scaling and comparing the results. If added mass results…

- Source: *Altair Radioss Help - 'Nodal Time Step Control'* — current online help, fetched 2026-09-21
- Locator: introductory paragraphs and 'Check for Mass Increase'
- URL: <https://help.altair.com/hwsolvers/rad/topics/solvers/rad/time_step_nodal_time_step_control_r.htm>
- Kind / confidence: vendor-support / high · Independent check: **confirmed**
- Caveats: The companion 'Results Checking' FAQ (RAD-1 URL) gives no number: 'it is necessary to check if it is not too important with respect to the total mass of the model (see the DM/M value ...)' and 'It is also important to post-process this added mass in order to check that it is not too large locally'.

**Verifier summary.** All 17 claims checked against the cited sources, re-fetched independently (raw HTML/JSON via curl, PDFs via PyMuPDF; files in <local scratch folder>). No hallucinated or misattributed quotes found; all 17 are 'confirmed'. Corrections and residual gaps to carry forward: (1) ABQ-3 locator: Abaqus 2016 GSA Section 13.5 is titled 'Example: forming a channel in Abaqus/Explicit', not 'Strategy for evaluating the results'. (2) RAD-1: the detail 'printed every output cycle' is NOT in the cited page and should be dropped; the MathML fraction structure was verified (numerator Ek+Ekr+Ei; denominator Ek,1+Ekr,1+Ei,1+Ewk-Ewk,1), and the page adds a /CHKPT exception to the reset-on-restart rule. (3) BLM-3: 'near eq. (8.7.14)' not verified; only (8.7.10) is visible in the snippet. BLM-4 section is 6.2.5. (4) BLM-2: the bridging words 'which leads to a' between pp.335/336 were not returned in any snippet I retrieved; everything around them was. All Belytschko evidence is Google Books search-inside snippets of the 2nd edition (with OCR artifacts), never a full page - adequate for the quotes, not for absence claims (e.g. 'no numeric mass limit'). (5) LSD-5: section 38.4.2 heading is on PDF p.795 (38-23), the quoted checks on p.796 (38-24); 'energy ratio' has zero text hits in the Theory Manual R16, confirmed. (6) LSD-6: ENDMAS default verified as 10^8 (superscript span), resolving the researcher's caveat. (7) LSD-4: the energy equation and ratio formula are images and remain unread; page is legacy user's-guide text. (8) Abaqus sources are university-hosted copies of vendor docs (WUSTL via Wayback, CU Boulder), not help.3ds.com; equation symbols are images. Definitions as found: Belytschko - |Wkin+Wint-Wext| <= eps*max(Wext,Wint,Wkin), eps ~1e-2, checked each step, a stability check; HG/total energy 3-5% 'large'. LS-DYNA support - HG < 10% of PEAK internal energy PER PART (rule of thumb); frictionless contact energy ~10% of peak internal energy 'might be considered acceptable'; energy ratio = total/(initial+external work), no tolerance stated. Abaqus…

**Not found or not accessible.**

- ASME V&V 10-2006/2019, V&V 10.1-2012, V&V 20: COULD NOT ACCESS (paywalled standards; only ASME/ANSI webstore and NAFEMS review landing pages found via search). No statement, numerical or otherwise, about energy balance as a solution-verification indicator can be attributed to them from this work.
- NAFEMS guidance on explicit-dynamics energy balance / hourglass fractions: NOT FOUND in readable form. Search surfaced only course/landing pages ('10 Steps to Successful Explicit Dynamic Analysis', 'Getting started with Explicit FEA'); members-only content not read.
- LSTC FAQ text file 'energy_balance' (https://ftp.lstc.com/anonymous/outgoing/support/FAQ/energy_balance and the jday/faq copy): COULD NOT ACCESS - HTTP 403 from both curl (browser UA) and WebFetch. The lsdyna.ansys.com 'Total energy' page appears to carry the same content but I did not verify equivalence.
- www.dynasupport.com originals: direct download timed out from this machine (curl and PowerShell). WebFetch reached them but returns model-summarised text, so verbatim quotes were taken from the lsdyna.ansys.com copies instead. The equation and energy-ratio formula on the 'Energy data' page are images and were not read.
- Abaqus Analysis User's Guide (mass scaling section; Abaqus/Explicit 'Explicit dynamic analysis' section) on help.3ds.com or the docs.software.vt.edu 2024 mirror: COULD NOT ACCESS (HTTP 403; WUSTL v6.6 mirror gives TLS certificate error/403 and only one page was in the Wayback Machine). So no Abaqus statement on acceptable percent mass change was read; only the Getting Started guide's KE/IE approach (ABQ-3).
- An Abaqus vendor statement of the form 'ALLAE should be less than 1-5% / 5-10% of ALLIE': NOT FOUND in vendor text. It appears only in forum posts and search-engine summaries (eng-tips, ResearchGate, iMechanica). The Getting Started 'hourglassing in a rubber block' example (gsx v6.5 ch04s03), which may contain such a number, could not be fetched (cert error; not archived). FROM MEMORY - UNVERIFIED: I recall a ~1% fi…
- Belytschko 1st edition (2000) wording and equation number: not checked; only the 2nd edition (2014) was read, via Google Books search-inside snippets (Google Books API quota was exhausted; the books.google.com jscmd=SearchWithinVolume2 endpoint was used). Full-page text of Section 6.2.3 (including eqs 6.2.14-6.2.17 defining Wint, Wext, Wkin) was not read beyond fragmentary snippets.
- LS-DYNA: any vendor-stated numeric tolerance on the glstat energy ratio (e.g. '0.9-1.1' or '+-5%'): NOT FOUND in Theory Manual R16, Keyword Manual Vol I R16 (*CONTROL_TERMINATION/*CONTROL_TIMESTEP pages), or the support pages read. *CONTROL_ENERGY page of the keyword manual was not separately inspected for remarks.
- OpenRadioss-specific documentation on energy error: nothing separate found; OpenRadioss relies on the Altair Radioss help pages cited.

## L — LS-DYNA: energy ledger and time-integration record

### L-C01 — 1 CONTROL_ENERGY card layout and defaults (R13)

**Claim.** In R13, *CONTROL_ENERGY Card 1 has six fields HGEN, RWEN, SLNTEN, RYLEN, IRGEN, MATEN (all integer) with table defaults 1, 2, 1, 1, 2, 1 respectively. DRLEN and DISEN are not present in R13.

> Purpose: Provide controls for energy dissipation options. ... Variable HGEN RWEN SLNTEN RYLEN IRGEN MATEN ... Default 1 2 1 1 2 1

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_ENERGY, p. 12-103 (PDF page 1333)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: Quote is the card table flattened from PDF text extraction (cells appear one per line in the PDF). Absence of DRLEN/DISEN in R13 verified by full-text search of the R13 PDF (zero hits).

### L-C02 — 1 CONTROL_ENERGY HGEN

**Claim.** HGEN=1: hourglass energy not computed (default). HGEN=2: computed and included in the energy balance, reported in glstat and matsum. Costs about ten percent.

> Hourglass energy calculation option. This option requires significant additional storage and increases cost by ten percent: EQ.1: Hourglass energy is not computed (default). EQ.2: Hourglass energy is computed and included in the energy balance. The hourglass energies are reported in the ASCII files glstat and matsum, see *DATABASE_OPTION. For implicit, or if the DRCPSID is active on *CONTROL_SHELL, the drilling energy is included here.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_ENERGY, p. 12-103 (PDF page 1333)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: Line-break hyphens and the 'fi' ligature from the PDF were normalised in the quote. Identical wording in R15 p. 12-106.

### L-C03 — 1 CONTROL_ENERGY RWEN

**Claim.** RWEN=1: rigidwall (stonewall) energy dissipation not computed. RWEN=2 (default): computed and included in the energy balance, reported in glstat.

> Rigidwall energy (a.k.a. stonewall energy) dissipation option: EQ.1: Energy dissipation is not computed. EQ.2: Energy dissipation is computed and included in the energy balance (default). The rigidwall energy dissipation is reported in the ASCII file glstat; see *DATABASE_OPTION.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_ENERGY, p. 12-103 (PDF page 1333)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**

### L-C04 — 1 CONTROL_ENERGY SLNTEN

**Claim.** SLNTEN table default is 1, but the manual states it is always set to 2 if contact is active and that SLNTEN=1 is not available; with 2 the sliding interface energy is included in the energy balance and reported in glstat and sleout.

> Sliding interface energy dissipation option (This parameter is always set to 2 if contact is active. The option SLNTEN = 1 is not available.): EQ.1: Energy dissipation is not computed. EQ.2: Energy dissipation is computed and included in the energy balance. The sliding interface energy is reported in ASCII files glstat and sleout; see *DATABASE_OPTION.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_ENERGY, p. 12-103 (PDF page 1333)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: The card table lists default 1 while the description says it is forced to 2 when contact is active; both are in the source.

### L-C05 — 1 CONTROL_ENERGY RYLEN

**Claim.** RYLEN=1 (default): Rayleigh damping energy dissipation not computed. RYLEN=2: computed and included in the energy balance; reported in glstat and matsum.

> Rayleigh energy dissipation option (damping energy dissipation): EQ.1: Energy dissipation is not computed (default). EQ.2: Energy dissipation is computed and included in the energy balance. The damping energy is reported in ASCII file glstat and matsum; see *DATABASE_OPTION.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_ENERGY, pp. 12-103 to 12-104 (PDF pages 1333-1334)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**

### L-C06 — 1 CONTROL_ENERGY IRGEN and MATEN

**Claim.** IRGEN=2 (default) includes initial reference geometry energy in the energy balance as part of internal energy; IRGEN=1 does not compute it. MATEN=1 (default) no detailed material energies; MATEN=2 splits internal energy into elastic, plastic, damage portions for a listed set of materials, reported in glstat and matsum.

> Initial reference geometry energy option (included in internal energy, resulting from *INITIAL_FOAM_REFERENCE_GEOMETRY): EQ.1: Initial reference geometry energy is not computed. EQ.2: Initial reference geometry energy is computed and included in the energy balance as part of the internal energy (default). MATEN Detailed material energies option. For a choice of material models (currently supported are 3, 4, 15, 19, 24, 63, 81, 82, 98, 104, 105, 106, 107, 123, 124, 188, 224, 225, 240, and 251 for shell and solid elements), internal energy is additionally split into elastic, plastic, and damage…

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_ENERGY, p. 12-104 (PDF page 1334)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**

### L-C07 — 1 CONTROL_ENERGY DRLEN and DISEN (R15 only)

**Claim.** R15 adds fields 7 and 8, DRLEN and DISEN, both default 1. DRLEN=2 computes drilling energy (implicit / DRCPSID-DRCPRM) into the energy balance, reported in glstat. DISEN=2 computes dissipated kinetic and internal energy for implicit into the energy balance, reported in glstat. Both are described as implicit-related.

> DRLEN Drilling energy calculation option, for implicit and with use of DRCPSID/DRCPRM on *CONTROL_SHELL: EQ.1: Drilling energy is not computed (default). EQ.2: Drilling energy is computed and included in the energy balance. The drilling energies are reported in the ASCII file glstat, see *DATABASE_OPTION. DISEN Dissipation energy calculation option, for implicit: EQ.1: Dissipated energy is not computed (default). EQ.2: Dissipated kinetic and internal energy is computed and included in the energy balance. The dissipation energies are reported in the ASCII file glstat, see *DATABASE_OPTION.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (R15@7355dd456, 02/28/24)
- Locator: *CONTROL_ENERGY, pp. 12-106 to 12-107 (PDF pages 1384-1385); card table defaults row '1 2 1 1 2 1 1 1'
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: R14 was not checked, so the release in which DRLEN/DISEN first appear (R14 or R15) is not established. No MPP-specific default is stated anywhere on the *CONTROL_ENERGY card in R13 or R15.

### L-C08 — 2 glstat contents

**Claim.** The manual's GLSTAT output-component table lists: time step, kinetic energy, internal energy, spring and damper energy (printed 'sprint'), hourglass energy, system damping energy, sliding interface energy, eroded kinetic energy, eroded internal energy, eroded hourglass energy, added mass, total energy, external work, total and initial energy, energy ratio without eroded energy, element & part ID controlling time step, global x,y,z velocity, time per zone cycle, joint internal energy, stonewall energy, rigid body stopper energy, percentage [mass] increase.

> GLSTAT time step total energy kinetic energy external work internal energy total and initial energy sprint and damper energy energy ratio without eroded energy hourglass energy element & part ID controlling time step system damping energy global x, y, z velocity sliding interface energy time per zone cycle eroded kinetic energy† joint internal energy eroded internal energy† stonewall energy eroded hourglass energy rigid body stopper energy added mass percentage [mass] increase

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *DATABASE_OPTION, 'Output Components for ASCII Files', GLSTAT table, p. 16-11 (PDF page 1833)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: Two-column table; quote is the raw extraction order (alternating left/right columns). I also rendered the page as an image and confirmed the items. 'sprint' is the manual's typo for 'spring'. The table does not list an item literally named 'total energy / initial energy'; it lists 'total and initial energy' and 'energy ratio without eroded energy'.

### L-C09 — 2 glstat contents are deck-dependent

**Claim.** Items in glstat appear only when relevant: hourglass energy only if HGEN=2; added mass only if DT2MS<0.

> Contents of "glstat." The glstat table above includes all items that may appear in the glstat data. The items that are actually written depend on the contents of the input deck. For example, hourglass energy appears only if HGEN = 2 in *CONTROL_ENERGY and added mass only appears if DT2MS < 0 in *CONTROL_TIMESTEP.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *DATABASE_OPTION, Remark 10, p. 16-15 (PDF page 1837)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: Same remark present in R15.

### L-C10 — 2 definition of total energy (manual)

**Claim.** The only explicit total-energy formula found in Volume I is in a release-notes bullet: total energy = kinetic + internal + hourglass + rigidwall energy (stated in the context of including eroded hourglass energy in glstat hourglass energy). The *DATABASE section itself gives no formula.

> Include eroded hourglass energy in hourglass energy in glstat file to be consistent with KE & IE calculations so that the total energy = kinetic energy + internal energy + hourglass energy + rigidwall energy.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: INTRODUCTION, release-notes bullet list (PDF page 110)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / medium · Independent check: **confirmed**
- Corrected locator: INTRODUCTION release notes, PDF p.110 (R7.1 section) and PDF p.118 (R8.0 section)
- Caveats: This is a change-log bullet, not a definition; it omits sliding-interface and damping energy, which the vendor support page (C11) includes. It does imply glstat KE, IE and hourglass energy INCLUDE eroded contributions. Which release the bullet belongs to was not determined.

### L-C11 — 2 definition of total energy (support site)

**Claim.** Per the vendor support site, glstat total energy is the sum of six terms: internal, kinetic, contact (sliding), hourglass, system damping, and rigidwall energy. So hourglass, sliding-interface, rigidwall and system-damping energies are INSIDE total energy and are separate from internal energy.

> Total energy reported in GLSTAT (see *DATABASE_GLSTAT ) is the sum of internal energy kinetic energy contact (sliding) energy hourglass energy system damping energy rigidwall energy

- Source: *Total energy - LS-DYNA support site (dynasupport.com howto)* — Wayback capture 2024-10-05 of live page (page text references v.970/971-era revisions)
- Locator: First paragraph and bullet list
- URL: <https://web.archive.org/web/20241005194851id_/https://www.dynasupport.com/howtos/general/total-energy>
- Kind / confidence: vendor-support / high · Independent check: **confirmed**
- Caveats: Original URL https://www.dynasupport.com/howtos/general/total-energy; read via Wayback raw capture because direct curl hung and WebFetch would not return verbatim text. Bullets are concatenated in the quote. External work is not in the sum.

### L-C12 — 2 energy ratio definition

**Claim.** The glstat 'total energy / initial energy' ratio is actually total energy / (initial energy + external work); external work IS in the denominator. Perfect balance means ratio = 1.0.

> The energy balance is perfect if total energy = initial total energy + external work, or in other words if the energy ratio (referred to in GLSTAT as total energy / initial energy although it actually is total energy / (initial energy + external work)) is equal to 1.0.

- Source: *Total energy - LS-DYNA support site (dynasupport.com howto)* — Wayback capture 2024-10-05
- Locator: Paragraph beginning 'The energy balance is perfect if'
- URL: <https://web.archive.org/web/20241005194851id_/https://www.dynasupport.com/howtos/general/total-energy>
- Kind / confidence: vendor-support / high · Independent check: **confirmed**
- Caveats: The Keyword Manual Vol I does not state this formula anywhere I could find (full-text search for 'energy ratio' and 'external work'). The ratio is reported per output state over time; the source says nothing about end-of-run vs maximum - that is an acceptance-criterion choice, not a solver definition. It is a whole-model quantity (glstat), not per part.

### L-C13 — 2 eroded energies

**Claim.** glstat energies include eroded-element contributions; eroded energy = internal energy of deleted elements and kinetic energy of deleted nodes. The 'energy ratio w/o eroded energy' is 1 with no deletions and <1 if elements have been deleted; deletions should not affect the total energy/initial energy ratio.

> The History > Global energies do not include the contributions of eroded elements whereas the GLSTAT energies do include those contributions. ... Eroded energy is the energy associated with deleted elements (internal energy) and deleted nodes (kinetic energy). Typically, the energy ratio w/o eroded energy would be equal to 1 if no elements have been deleted or less than one if elements have been deleted. The deleted elements should have no bearing on the total energy / initial energy ratio.

- Source: *Total energy - LS-DYNA support site (dynasupport.com howto)* — Wayback capture 2024-10-05
- Locator: Paragraphs beginning 'The History > Global energies' and 'Eroded energy is'
- URL: <https://web.archive.org/web/20241005194851id_/https://www.dynasupport.com/howtos/general/total-energy>
- Kind / confidence: vendor-support / high · Independent check: **confirmed**
- Caveats: Ellipsis joins two adjacent paragraphs. For implicit runs the manual footnote (p. 16-11) says 'lost' discretization energy is also accumulated in the eroded kinetic/internal energies.

### L-C14 — 2 damping energy placement

**Claim.** Stiffness damping energy is lumped into internal energy; mass damping energy is the separate glstat item 'system damping energy'. Hourglass, rigidwall and system damping energies are computed/written only when HGEN, RWEN, RYLEN = 2.

> Hourglass energy is computed and written only if HGEN is set to 2 in *CONTROL_ENERGY. Likewise, rigidwall energy and system damping energy are computed and written only if RWEN and RYLEN, respectively, are set to 2. Stiffness damping energy is lumped into internal energy. Mass damping energy appears as a separate line item system damping energy.

- Source: *Total energy - LS-DYNA support site (dynasupport.com howto)* — Wayback capture 2024-10-05
- Locator: Paragraphs beginning 'Hourglass energy is computed' and 'Stiffness damping energy'
- URL: <https://web.archive.org/web/20241005194851id_/https://www.dynasupport.com/howtos/general/total-energy>
- Kind / confidence: vendor-support / high · Independent check: **confirmed**

### L-C15 — 2 Rayleigh stiffness damping energy in internal energy (manual)

**Claim.** The manual confirms energy dissipated by Rayleigh (stiffness) damping is computed only if RYLEN=2, is accumulated as element internal energy, and is lumped in with internal energy in glstat.

> Energy dissipated by Rayleigh damping is computed only if the RYLEN on *CONTROL_ENERGY is set to 2. This energy is accumulated as element internal energy and is included in the energy balance. In the glstat file this energy will be lumped in with the internal energy.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *DAMPING section, stiffness-damping keyword remarks (PDF page 1819)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Corrected locator: *DAMPING_PART_STIFFNESS, Remarks, p. 15-11 (PDF page 1819)
- Caveats: I located the passage by line/page but did not confirm the exact keyword-card header on that page (it is the COEF stiffness-damping discussion, consistent with *DAMPING_PART_STIFFNESS).

### L-C16 — 2 spring and damper energy

**Claim.** 'Spring and damper energy' in glstat is a subset of 'internal energy' (discrete elements, seatbelts, joint stiffness).

> Spring and Damper Energy. "Spring and damper energy" reported in glstat is a subset of "internal energy". The "spring and damper energy" includes internal energy of discrete elements, seatbelt elements, and that associated with joint stiffness (see *CONSTRAINED_JOINT_STIFFNESS_…).

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *DATABASE_OPTION, Remark 6, p. 16-14 (PDF page 1836)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**

### L-C17 — 2 bulk viscosity energy

**Claim.** With the default bulk viscosity TYPE=1 (solids only) the internal energy from bulk viscosity is always computed and included in the overall energy balance; same for TYPE=2 (Richards-Wilkins, 2D plane strain and axisymmetric solids only). For shells it is included only with TYPE=-2 (TYPE=-1 applies viscosity to shells without computing the energy). Defaults: Q1=1.5, Q2=.06, TYPE=1.

> TYPE Default bulk viscosity type, IBQ (default = 1): EQ.-2: same as -1 but the internal energy dissipated by the viscosity in the shell elements is computed and included in the overall energy balance. EQ.-1: same as 1 but also includes viscosity in shell formulations 2, 4, 10, 16, and 17. The internal energy is not computed in the shell elements. EQ.1: standard bulk viscosity. Solid elements only and internal energy is always computed and included in the overall energy balance. EQ.2: Richards-Wilkins bulk viscosity. Two-dimensional plane strain and axisymmetric solid elements only. Internal e…

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_BULK_VISCOSITY, p. 12-60 (PDF page 1290)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: The text calls the bulk-viscosity dissipation 'internal energy' that is 'included in the overall energy balance'; it does not literally say 'reported under the glstat internal energy line'. That the q-work lands in the internal-energy line is the natural reading but is an inference. The support page adds: 'Energy dissipated due to shell bulk viscosity was not calculated prior to revision 4748 of v. 970.'

### L-C18 — 2 internal energy definition

**Claim.** Internal energy is accumulated incrementally per element as stress times incremental strain times volume over six components, summed over elements; stiffness damping dissipation is added if RYLEN=2.

> (IE)new = (IE)old + sum over all six directions of (stress * incremental strain * volume) The internal energies of all the elements is summed to give the total internal energy. In addition, energy dissipated due to stiffness damping (*DAMPING_PART_STIFFNESS) is added to the internal energy if RYLEN=2 in *CONTROL_ENERGY.

- Source: *Internal Energy - LS-DYNA support site (dynasupport.com howto)* — Wayback capture 2025-05-29
- Locator: Opening paragraphs
- URL: <https://web.archive.org/web/20250529110029id_/https://www.dynasupport.com/howtos/general/internal-energy>
- Kind / confidence: vendor-support / high · Independent check: **confirmed**
- Caveats: Original URL https://www.dynasupport.com/howtos/general/internal-energy. The page does not mention bulk viscosity.

### L-C19 — 2 external work content

**Claim.** External work includes work by applied forces and pressures and by velocity, displacement or acceleration boundary conditions; internal energy includes elastic strain energy and work done in permanent deformation. Hourglass energy is excluded by default; Rayleigh damping dissipation (RYLEN=2) is added to internal energy. The energy ratio can be a termination criterion via ENDENG.

> Internal energy includes elastic strain energy and work done in permanent deformation. External work includes work done by applied forces and pressures as well as work done by velocity, displacement or acceleration boundary conditions. Energy associated with hourglassing is excluded by default, but can be included by setting HGEN to 2 on *CONTROL_ENERGY (Control Card 19, Column 5).

- Source: *Energy data - LS-DYNA support site (LS-DYNA User's Guide tutorial)* — Wayback capture 2024-10-11
- Locator: Paragraph after 'Where'
- URL: <https://web.archive.org/web/20241011173004id_/https://www.dynasupport.com/tutorial/ls-dyna-users-guide/energy-data>
- Kind / confidence: vendor-support / high · Independent check: **confirmed**
- Caveats: The page's balance equation and its energy-ratio formula are embedded as IMAGES and could not be read; only the surrounding prose was read. The 'Control Card 19' references indicate an old fixed-format-era text.

### L-C20 — 3 what each ASCII database provides

**Claim.** Manual one-line descriptions: BNDOUT 'Boundary condition forces and energy'; GLSTAT 'Global statistics and energies (recommended)'; MATSUM 'Part energies'; NODFOR 'Nodal force groups'; RCFORC 'Resultant contact interface forces'; RWFORC 'Rigidwall forces'; SLEOUT 'Contact interface energies'; SPCFORC 'SPC reaction forces'.

> BNDOUT Boundary condition forces and energy. ... GLSTAT Global statistics and energies (recommended). See *CONTROL_ENERGY. ... MATSUM Part energies. See Remarks 1 and 2 below. ... NODFOR Nodal force groups. See *DATABASE_NODAL_FORCE_GROUP. ... RCFORC Resultant contact interface forces. To output in a local coordinate system, see *CONTACT, Optional Card C. RWFORC Rigidwall forces. See *RIGIDWALL_OPTION. ... SLEOUT Contact interface energies. See *CONTACT_OPTION. SPCFORC SPC reaction forces. See *BOUNDARY_SPC_OPTION.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *DATABASE_OPTION, OPTION1 list, pp. 16-3 to 16-4 (PDF pages 1825-1826)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: Ellipses skip intervening list entries.

### L-C21 — 3 output components per file

**Claim.** Output components per the manual tables: MATSUM = kinetic energy, internal energy, hourglass energy, x,y,z momentum, x,y,z rigid body velocity, eroded internal energy, eroded kinetic energy, added mass. SLEOUT = slave energy, master energy, frictional energy. RWFORC = normal, x,y,z force. RCFORC = x,y,z force, mass of nodes in contact. SPCFORC = x,y,z force, x,y,z moment. BNDOUT = x,y,z force, x,y,z moment (listed), energies. NODFOR = x,y,z force.

> MATSUM kinetic energy x, y, z rigid body velocity internal energy eroded internal energy hourglass energy eroded kinetic energy x, y, z momentum added mass

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *DATABASE_OPTION, 'Output Components for ASCII Files' tables, pp. 16-10 to 16-13 (PDF pages 1832-1835)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / medium · Independent check: **confirmed**
- Caveats: Quote given for MATSUM only. The BNDOUT/DCFAIL/DEFORC, NCFORC/NODOUT/NODFOR, PLLYOUT/RBDOUT/RCFORC and RWFORC/SECFORC/SLEOUT tables are multi-column and text extraction interleaves columns; my column assignment for BNDOUT, RCFORC, RWFORC, SLEOUT, SPCFORC was inferred from extraction order, not checked against a rendered image (only the GLSTAT page was rendered). Verify these columns visually on PDF pages 1832, 1834, 1835 before ratifying. The ma…

### L-C22 — 3 matsum vs glstat kinetic energy discrepancy

**Claim.** matsum and glstat kinetic energies can differ: added-mass energy is included in glstat but not matsum; matsum is element-by-element with midpoint velocities, glstat uses nodal velocities.

> Discrepancies between "matsum" and "glstat" Output. The kinetic energy quantities in the matsum and glstat files may differ slightly in values for several reasons. First, the energy associated with added mass (from mass-scaling) is included in the glstat calculation but is not included in matsum. Secondly, the energies are computed element by element in matsum for the deformable materials and, consequently, nodes which are merged with rigid bodies will also have their kinetic energy included in the rigid body total. Furthermore, kinetic energy is computed from nodal velocities in glstat and f…

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *DATABASE_OPTION, Remark 1, p. 16-13 (PDF page 1835)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: Relevant to any acceptance check that sums matsum part energies and compares with glstat.

### L-C23 — 3 matsum eroded energies (IERODE)

**Claim.** Eroded internal and kinetic energy per part are written to matsum only if IERODE=1 on *CONTROL_OUTPUT (default 0 = no extra data); also adds part ID 0 (KE of nonstructural/lumped mass and inertia) and part ID -1 (*ELEMENT_MASS_PART).

> IERODE Output eroded internal and kinetic energy into the matsum file. Also, (1) under the heading of part ID 0 in matsum, output the kinetic energy from nonstructural mass, lumped mass elements, and lumped inertia elements; and (2) under the heading of part ID -1in matsum, output the kinetic energy associated with distributed mass from *ELEMENT_MASS_PART. EQ.0: Do not output extra data. EQ.1: Output the eroded internal and kinetic energy.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_OUTPUT, IERODE (PDF page 1648)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**

### L-C24 — 3 rcforc averaging

**Claim.** rcforc resultant contact forces are averaged over the preceding output interval (not instantaneous).

> The "rcforc" File. Resultant contact forces reported in rcforc are averaged over the preceding output interval.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *DATABASE_OPTION, Remark 5, p. 16-14 (PDF page 1836)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**

### L-C25 — 3 bndout categories

**Claim.** bndout OPTION1-4 control inclusion of nodal force group, concentrated force, pressure BC, and velocity/displacement/acceleration nodal BC output; 0/blank = included (default), 1 = excluded.

> OPTIONn Field for "bndout." For the bndout file, OPTION1 controls the nodal force group output, OPTION2 controls the concentrated force output, OPTION3 controls the pressure boundary condition output, and OPTION4 controls the velocity/displacement/acceleration nodal boundary conditions. If the value is 0 or left blank, the category is included (the default), and if it is 1, the category is not included in the bndout file.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *DATABASE_OPTION, Remark 9, pp. 16-14 to 16-15 (PDF pages 1836-1837)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: So prescribed-motion forces/energy come from bndout (velocity/displacement/acceleration category), SPC reactions from spcforc.

### L-C26 — 3 nodfor

**Claim.** nodfor requires *DATABASE_NODAL_FORCE_GROUP (NSID, optional CID) plus *DATABASE_NODFOR for the interval; it writes group resultant reaction forces, the external work done by them, and per-node forces.

> The reaction forces in the global x, y, and z directions (and local x, y, and z directions if CID is defined above) for the nodal force group are written to the nodfor file (see *DATABASE_NODFOR) along with the external work done by these reaction forces. The reaction forces in the global x, y, and z directions for each node in the nodal force group are also written to nodfor.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *DATABASE_NODAL_FORCE_GROUP, Remark 1, p. 16-111 (PDF page 1933)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**

### L-C27 — 3/4 DT field and file creation

**Claim.** No ASCII database is created unless its *DATABASE_OPTION1 card is present; DT is the output interval, default 0., and DT=0 means no output.

> LS-DYNA will not create an ASCII database unless the corresponding *DATABASE_OPTION1 card is included in the input deck. ... DT Time interval between outputs. If DT is zero, no output is printed.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *DATABASE_OPTION, pp. 16-3 and 16-5 (PDF pages 1825, 1827); Card 1 defaults row: DT 0., BINARY '1 or 2', LCUR none, IOOPT 0.
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**

### L-C28 — 4 BINARY field values and defaults

**Claim.** BINARY: 1 = ASCII file written (SMP default); 2 = data written to binout, ASCII not created (MPP default); 3 = both ASCII and binout, but MPP executables only produce the binary database. Card default is listed as '1 or 2'.

> BINARY Flag for binary output. See remarks under "Output Files and Post-Processing" in Appendix O, "LS-DYNA MPP User Guide." EQ.1: ASCII file is written. This is the default for shared memory parallel (SMP) LS-DYNA executables. EQ.2: Data written to a binary database binout, which contains data that would otherwise be output to the ASCII file. The ASCII file in this case is not created. This is the default for MPP LS-DYNA executables. EQ.3: ASCII file is written, and the data is also written to the binary database (NOTE: MPP LS-DYNA executables will only produce the binary database).

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *DATABASE_OPTION, BINARY, pp. 16-5 to 16-6 (PDF pages 1827-1828)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: Same wording confirmed present in R15 (text lines for EQ.1/EQ.2/EQ.3).

### L-C29 — 4 MPP ASCII output

**Claim.** Per Appendix O, MPP writes these databases only in binary (binout) regardless of BINARY; ASCII files are obtained after the run by the l2a converter ('l2a binout*'). The manual documents no input setting that forces direct ASCII output from MPP. Files may be one per processor, named binoutnnnn.

> For performance reasons, many of the ASCII output files normally created by LS-DYNA have been combined into a new binary format used by MPP/LS-DYNA. When running MPP/LS-DYNA, this output is written only in binary format, irrespective of the setting of the variable BINARY in *DATABASE_OPTION. There is a post-processing program l2a, which reads the binary database of files and produces as output the corresponding ASCII files. ... The files (up to one per processor) are named binoutnnnn, where nnnn is replaced by the four-digit processor number. To convert these files to ASCII, simply feed them …

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: Appendix O 'LS-DYNA MPP User Guide', section 'Output Files and Post-Processing', p. 63-2 (PDF page 3762); supported-file list continues p. 63-3 and includes GLSTAT, MATSUM, RCFORC, SPCFORC, RWFORC, BNDOUT, SLEOUT, SPHOUT, 'NODOFR' [sic]
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: The same sentence ('irrespective of the setting...') is present in R15 Appendix O (PDF page 3968). Practitioner reports that recent MPP builds do honour BINARY=1/3 are FROM MEMORY - UNVERIFIED and contradict the manual text; I did not find a primary source for that. The ASCII glstat converted by l2a is where the controlling element ID can be read (see C34).

### L-C30 — 5 DT2MS semantics

**Claim.** DT2MS<0: TSSFAC x |DT2MS| is the minimum permitted time step and mass is added only where needed to meet the Courant criterion; DT2MS>0 is for quasi-static / inertia-insignificant analyses. Default 0.0 (no mass scaling).

> DT2MS Time step size for mass scaled solutions. (Default = 0.0) GT.0.0: Positive values are for quasi-static analyses or time history analyses where the inertial effects are insignificant. LT.0.0: TSSFAC × |DT2MS| is the minimum time step size permitted and mass scaling is done if and only if it is necessary to meet the Courant time step size criterion. This option can be used in transient analyses if the mass increases remain insignificant. See also the variable MS1ST below and the *CONTROL_TERMINATION variable ENDMAS.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_TIMESTEP, DT2MS, p. 12-551 (PDF page 1781)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: Support site adds for positive DT2MS: 'Mass is added or taken away from elements so that the timestep of every element is the same.' (dynasupport Mass scaling howto).

### L-C31 — 5 TSSFAC semantics and default

**Claim.** Time step = TSSFAC x min over elements of element time steps. Default TSSFAC: 0.90 normally; 0.667 with *INITIAL_DETONATION or *BOUNDARY_NON_REFLECTING; with contacts and SLSFAC specified, 0.333 for SLSFAC>=9 and 0.667 for 1<=SLSFAC<9. Values above 0.90 often unstable.

> During the solution we loop through the elements and determine a new time step size by taking the minimum value over all elements: Δt n+1 = TSSFAC × min{Δt1, Δt2, . . . , ΔtN} where N is the number of elements. ... For stability reasons the scale factor TSSFAC should be set to a value less than 1.0. The default value of TSSFAC is as follows: a) If contacts are present, and SLSFAC is specified in CONTROL_CONTACT, then TSSFAC = {0.333 for SLSFAC≥9. 0.667 for 1. ≤SLSFAC< 9. b) TSSFAC = 0.667 if either of the following cards is used: *INITIAL_DETONATION *BOUNDARY_NON_REFLECTING c) TSSFAC = 0.90 o…

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_TIMESTEP, Remark 1, pp. 12-553 to 12-554 (PDF pages 1783-1784)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: Math glyphs were transliterated from the PDF's math font. Note the *BOUNDARY_NON_REFLECTING rule: relevant to wave-propagation decks.

### L-C32 — 5 MS1ST and monotonic added mass

**Claim.** With DT2MS<0 and MS1ST=0 (default) mass scaling is applied throughout; added mass may increase with time but never decrease. MS1ST=1 computes added mass once at the first step.

> MS1ST Option for mass scaling that applies when DT2MS < 0. EQ.0: mass scaling is considered throughout the analysis to ensure that the minimum time step cannot drop below TSSFAC × |DT2MS|. Added mass may increase with time, but it will never decrease. (default) EQ.1: added mass is calculated at the first time step and remains unchanged thereafter.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_TIMESTEP, MS1ST, p. 12-552 (PDF page 1782)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: Because added mass is non-decreasing, end-of-run added mass equals the maximum over the run (for MS1ST=0).

### L-C33 — 5 where added mass is reported

**Claim.** Added mass vs time is available for the whole model in glstat and per part in matsum; per-element added mass for shells can be written to d3plot with STSSZ=3. d3hsp also reports 'added mass' and 'percentage increase'.

> To determine where and when mass is automatically added, write GLSTAT and MATSUM files. These files will allow you to plot added mass vs. time for the complete model and for individual parts, respectively. To produce fringe plots of added mass in parts comprised of shell elements (DT2MS negative), set STSSZ=3 in *DATABASE_EXTENT_BINARY.

- Source: *Mass scaling - LS-DYNA support site (dynasupport.com howto)* — Wayback capture 2024-12-03
- Locator: Paragraph beginning 'To determine where and when mass is automatically added'
- URL: <https://web.archive.org/web/20241203084725id_/https://www.dynasupport.com/howtos/general/mass-scaling>
- Kind / confidence: vendor-support / high · Independent check: **confirmed**
- Caveats: Original URL https://www.dynasupport.com/howtos/general/mass-scaling. d3hsp reporting is evidenced on the same page only in the spot-weld discussion: '"added mass" and "percentage increase" in d3hsp AFTER the first time step "added mass" in glstat and matsum'. The manual's matsum component table lists 'added mass' (C21) and the glstat table lists 'added mass' and 'percentage [mass] increase' (C08). A manual release note says per-part added mass …

### L-C34 — 6 element/part controlling the time step

**Claim.** glstat reports 'time step' and 'element & part ID controlling time step'; the controlling element ID is in the glstat data but is not read by LS-PrePost, so the ASCII glstat must be opened in a text editor.

> Element ID Controlling the Time Step. The element ID controlling the time step is included in the glstat data but is not read by LS-PrePost. If the element ID is of interest to the user, the ASCII version of the glstat file can be opened with a text editor.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *DATABASE_OPTION, Remark 11, p. 16-15 (PDF page 1837); GLSTAT table p. 16-11
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: Time-step history resolution is limited to the glstat DT output interval. The manual does not document the exact glstat line text (e.g. 'dt of cycle N is controlled by ...') - see not_found.

### L-C35 — 6 initial element time steps in d3hsp

**Claim.** *CONTROL_OUTPUT IPNINT controls printing of initial element time step sizes to d3hsp: 0 = the 100 smallest, 1 = all elements, >1 = IPNINT smallest.

> IPNINT Flag controlling output of initial time step sizes for elements to d3hsp: EQ.0: 100 elements with the smallest time step sizes are printed. EQ.1: Time step sizes for all elements are printed. GT.1: IPNINT elements with the smallest time step sizes are printed.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_OUTPUT, IPNINT, pp. 12-416 to 12-417 (PDF pages 1646-1647)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**

### L-C36 — 7 CONTROL_TERMINATION fields

**Claim.** Fields and defaults: ENDTIM 0.0 (mandatory termination time); ENDCYC 0 (termination cycle, used if reached before ENDTIM; cycle = time step number); DTMIN 0.0 (tsmin = dtstart x DTMIN; terminates with a restart dump when step drops to tsmin, unless ERODE); ENDENG 0.0 (percent change in energy ratio for termination; inactive if undefined); ENDMAS 1.0E+08 (percent change in total mass for termination; relevant only with DT2MS mass scaling); NOSOL 0.

> ENDTIM Termination time. Mandatory. ENDCYC Termination cycle. The termination cycle is optional and will be used if the specified cycle is reached before the termination time. Cycle number is identical with the time step number. DTMIN Reduction (or scale) factor to determine minimum time step, tsmin, where tsmin = dtstart × DTMIN and dtstart is the initial step size determined by LS-DYNA. When the time step drops to tsmin, LS-DYNA terminates with a restart dump. ... ENDENG Percent change in energy ratio for termination of calculation. If undefined, this option is inactive. ENDMAS Percent chan…

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_TERMINATION, pp. 12-524 to 12-525 (PDF pages 1754-1755); defaults row '0.0 0 0.0 0.0 1.0E+08 0'
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: The manual does not say whether ENDENG's 'percent change' is signed or absolute, nor which ratio (with/without eroded energy) it uses.

### L-C37 — 7 error-termination reporting (MPP)

**Claim.** For MPP, on error termination each processor writes a 'last known location' line to its own message file beginning 'When error termination was triggered, this processor was'.

> For MPP, set a "last known location" flag to give some indication of where the processors were if an error termination happens. Each writes a message to their own message file. Look for a line that says "When error termination was triggered, this processor was".

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: INTRODUCTION, release-notes bullets (text line ~5550 of extraction; PDF page ~128)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / medium · Independent check: **confirmed**
- Corrected locator: INTRODUCTION, R8.0 release-notes bullets, PDF page 131
- Caveats: Release-note bullet only. The exact 'N o r m a l t e r m i n a t i o n' / 'E r r o r t e r m i n a t i o n' banner strings are NOT documented in Volume I (see not_found). PDF page number is approximate; I did not compute it for this line.

### L-C38 — 8 CONTROL_SPH IDIM

**Claim.** IDIM: 3 = 3D; 2 = 2D plane strain; -2 = 2D axisymmetric. Card default listed as 'none'.

> IDIM Space dimension for SPH particles: EQ.3: 3D problems EQ.2: 2D plane strain problems EQ.-2: 2D axisymmetric problems (see Remark 2)

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_SPH, IDIM, p. 12-502 (PDF page 1732)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**

### L-C39 — 8 CONTROL_SPH FORM=12

**Claim.** FORM=12 is the moving least-squares based formulation, available for MPP only, for improved accuracy and tensile stability in extremely large deformation at significant cost; constant smoothing length (HMIN=HMAX=1.0) strongly recommended. ISTAB and QL (default 0.01) apply only to it. FORM default is 0.

> EQ.12: Moving least-squares based formulation (MPP only, see Remark 2e) ... e) For improved accuracy and tensile stability, a formulation based on moving least-squares (FORM = 12) is available. This formulation can be used for extremely large deformation applications but entails a significant computational cost. FORM = 12 is available for MPP simulations only. It is strongly recommended to keep a constant smoothing length for this formulation by setting HMIN = 1.0 and HMAX = 1.0 in *SECTION_SPH.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_SPH, FORM list p. 12-502 and Remark 2(e) p. 12-506 (PDF pages 1732, 1736)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: Consequence: a FORM=12 run is necessarily MPP, hence binout-only output per Appendix O (C29). R15 lists the same 'EQ.12 ... (MPP only' text.

### L-C40 — 8 SPH formulations valid in axisymmetry

**Claim.** Only FORM 0, 1, 15 and 16 are implemented for 2D axisymmetric SPH (IDIM=-2); hence FORM=12 is not available with IDIM=-2 per the manual. No analogous restriction is stated for plane strain (IDIM=2).

> f) Only formulations 0, 1, 15 and 16 are implemented for 2D axisymmetric problems (IDIM = -2).

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_SPH, Remark 2(f), p. 12-506 (PDF page 1736)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: The manual neither confirms nor denies FORM=12 support for IDIM=2; absence of a restriction is not confirmation.

### L-C41 — 8 *HOURGLASS IBQ/Q1/Q2

**Claim.** On *HOURGLASS, IBQ is 'Not used' (bulk viscosity always on for solids; beams/shells only via *CONTROL_BULK_VISCOSITY TYPE), but Q1 (quadratic, default 1.5) and Q2 (linear, default 0.06) set the coefficients per part and, when the *HOURGLASS ID is referenced by a part, override *CONTROL_BULK_VISCOSITY.

> Purpose: Define hourglass and bulk viscosity properties which are referenced using HGID in the *PART command. Properties specified here, when invoked for a particular part, override those in *CONTROL_HOURGLASS and *CONTROL_BULK_VISCOSITY. ... IBQ Not used. Bulk viscosity is always on for solids. Bulk viscosity for beams and shells can only be turned on using the variable TYPE in *CONTROL_BULK_VISCOSITY; however, the coefficients can be set using Q1 and Q2 below. Q1 Quadratic bulk viscosity coefficient. See Remark 3. Q2 Linear bulk viscosity coefficient. See Remark 3.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *HOURGLASS, pp. 24-1 to 24-2 (PDF pages 2627-2628); defaults row: IBQ 'not used', Q1 1.5, Q2 0.06
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: Identical IBQ text in R15.

### L-C42 — 8 bulk viscosity coefficients for SPH

**Claim.** For SPH, artificial viscosity type is chosen by IAVIS on *CONTROL_SPH (0 = Monaghan, default; 1 = standard solid-element form, not supported in 2D/2D-axisymmetric SPH). The Q1 and Q2 constants used by SPH come from *CONTROL_BULK_VISCOSITY or *HOURGLASS, and the manual recommends Q1=Q2=1.0 with Monaghan viscosity. So the Q1/Q2 fields of *HOURGLASS do apply to SPH parts; IBQ does not (it is unused).

> and Q1 and Q2 are input constants. When using Monaghan type artificial viscosity, it is recommended that the user set both Q1 and Q2 to 1.0 on either the *CONTROL_BULK_VISCOSITY or *HOURGLASS keywords; see for example G. R. Liu.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_SPH, Remark 4 'Artificial Viscosity', pp. 12-507 to 12-508 (PDF pages 1737-1738); IAVIS field p. 12-504: 'EQ.0: Monaghan type artificial viscosity formulation is used. EQ.1: Standard type artificial viscosity formulation from solid element is used (this option is not supported in SPH 2D and 2D axisymmetric elements).'
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: The manual does not state whether SPH artificial-viscosity work is accumulated into the glstat internal energy; for the solid-type form it says only that it 'has better energy balance for SPH elements'. The remark text writes 'AVIS' where the card field is 'IAVIS'.

### L-C43 — hourglass energy acceptance heuristic (context)

**Claim.** Support-site rule of thumb: hourglass energy should be small relative to PEAK internal energy FOR EACH PART, <10%, evaluated with HGEN=2 via glstat (system) and matsum (per part).

> To evaluate hourglass energy, set HGEN to 2 in *CONTROL_ENERGY and use *DATABASE_GLSTAT and *DATABASE_MATSUM to report the HG energy for the system and for each part, respectively. The point is to confirm that the nonphysical HG energy is small relative to peak internal energy for each part (<10% as a rule-of-thumb).

- Source: *Hourglass - LS-DYNA support site (dynasupport.com howto)* — Wayback capture 2025-06-16
- Locator: Paragraph beginning 'To evaluate hourglass energy'
- URL: <https://web.archive.org/web/20250616182522id_/https://www.dynasupport.com/howtos/element/hourglass>
- Kind / confidence: vendor-support / high · Independent check: **confirmed**
- Caveats: Original URL https://www.dynasupport.com/howtos/element/hourglass. Normaliser is peak internal energy per part, not total energy. Explicitly a rule of thumb. Hourglass modes do not exist in SPH; this applies to under-integrated solids/shells.

### L-C44 — contact energy acceptance heuristic (context)

**Claim.** Support-site judgement: without friction, net contact energy (slave + master) should be small; 10% of peak internal energy 'might be considered acceptable'. sleout reports contact energies per contact.

> In the absence of friction, you would hope to see a small net contact energy (net = sum of slave side energy and master side energy). Small is a matter of judgement -- 10% of peak internal energy might be considered acceptable for contact energy in the absence of contact friction. ... If you have more than one contact defined, the sleout file (*DATABASE_SLEOUT) will report contact energies for each contact

- Source: *Total energy - LS-DYNA support site (dynasupport.com howto)* — Wayback capture 2024-10-05
- Locator: Sections 'Positive contact energy' and 'Negative contact energy'
- URL: <https://web.archive.org/web/20241005194851id_/https://www.dynasupport.com/howtos/general/total-energy>
- Kind / confidence: vendor-support / high · Independent check: **confirmed**
- Caveats: Hedged wording in the source ('might be considered acceptable'); not a standard.

**Verifier summary.** All 44 claims check out against sources I fetched myself. No quote was hallucinated, misattributed or taken from the wrong edition. **What I read** - The R13 and R15 Volume I PDFs downloaded with HTTP 200 from the cited lsdyna.ansys.com URLs. The title pages read "R13@bbf149e1c (12/16/25)" and "R15@7355dd456 (02/28/24)", matching the claimed editions. - I extracted the text of every page and string-matched each quote. The only differences are whitespace, line-break hyphens, ligatures (fi, ffi, ff) and math-italic glyphs. - The five dynasupport pages came from the exact Wayback URLs cited. The internal-energy capture arrived gzip-compressed and had to be decompressed before it could be read. - The table claims that depend on column layout (C08 and C21) were checked on rendered page images of PDF pages 1832 to 1835. The researcher's column assignments for BNDOUT, RCFORC, RWFORC, SLEOUT, SPCFORC and NODFOR are correct. - The claim that R13 has no DRLEN or DISEN field holds: both names return zero hits in the R13 text. **Corrections to locators and caveats (no verdict changes)** - **C37:** the bullet is on PDF page 131, not about 128, inside the R8.0 release-notes section. - **C10:** the bullet appears twice, on PDF page 110 (R7.1 release notes) and page 118 (R8.0 release notes). The release the researcher left undetermined is therefore R7.1/R8.0. - **C15:** the page header is *DAMPING_PART_STIFFNESS, page 15-11, which settles the researcher's open question about the keyword. - **C29:** the caveat is wrong that R15 Appendix O carries the same sentence. R15 rewords it as "MPP/LS-DYNA writes this output only in binary format, irrespective of the setting of the variable BINARY in *DATABASE_OPTION." The meaning is the same, but it should not be quoted as identical across editions. - **C39:** the manual's ISTAB text says "only used when IFORM = 12", which is the manual's own inconsistency with the field name FORM. **Inferences still flagged as not stated in the sources** - The manual does not say the bulk-viscosity dissipation lands in the glstat internal-energy line (C17). - The statement that *HOURGLASS Q1/Q2 apply to SPH parts rests on the Remark 4 recommendation, no…

**Not found or not accessible.**

- Exact formula for glstat 'total energy' and for the energy ratio inside the Keyword Manual Vol I itself: NOT FOUND. Full-text searched R13 for 'total energy', 'energy ratio', 'external work', 'energy balance'; only the glstat item list, a release-note bullet (C10) and ENDENG exist. The definitions in the claims come from dynasupport.com (vendor-support). The Theory Manual was not read.
- The balance equation and energy-ratio equation on dynasupport 'Energy data' page: COULD NOT ACCESS - they are embedded images; only the surrounding prose was read.
- Whether the energy-ratio acceptance should be evaluated at end of run or as max over the run: no source read says; sources define the ratio as a time history in glstat only.
- Literal glstat / message-file line formats ('dt of cycle N is controlled by ...', 'N o r m a l t e r m i n a t i o n', 'E r r o r t e r m i n a t i o n'): NOT FOUND in Vol I R13 (grep for 'controlled by', 'N o r m a l', 'normal termination', 'error termination'). Any such strings are FROM MEMORY - UNVERIFIED and should be confirmed against an actual messag/glstat/d3hsp file from the project's own runs.
- A documented way to force direct ASCII output from an MPP executable: NOT FOUND. Manual R13 and R15 Appendix O both say MPP output is binary only irrespective of BINARY; the only documented route is l2a conversion. Claims that newer MPP builds honour BINARY=1/3 are FROM MEMORY - UNVERIFIED.
- MPP-specific defaults for *CONTROL_ENERGY fields: none stated on the card in R13 or R15 (the only MPP/SMP-dependent default found is BINARY on *DATABASE_OPTION).
- Whether per-part 'added mass' in matsum is available from MPP today: only an old release note '(SMP version only)' was found (PDF page 53); current status not established.
- Whether matsum carries a per-part damping energy column: RYLEN text says damping energy is reported in 'glstat and matsum' but the MATSUM component table does not list it; not resolved.
- Whether SPH artificial-viscosity work is included in the internal energy reported in glstat/matsum for SPH parts: NOT FOUND in *CONTROL_SPH, *CONTROL_BULK_VISCOSITY or *HOURGLASS.
- Whether FORM=12 is supported with IDIM=2 (plane strain): manual silent; only the axisymmetric restriction (FORM 0,1,15,16) is stated.
- R11, R12, R14 editions were not read; release in which DRLEN/DISEN were introduced (R14 vs R15) not established.
- dynasupport pages 'Negative contact energy' (howtos/contact/negative-contact-energy), a bulk-viscosity howto, and an MPP-ASCII howto: Wayback returned 404 for the URLs guessed; the negative-contact-energy content that was read is the section inside the 'Total energy' page. Direct access to www.dynasupport.com via curl/Python failed (TLS hang / ASN1 error); WebFetch reached it but refused verbatim reproduction, so Wa…
- Column assignment of the multi-column BNDOUT / RCFORC / RWFORC / SLEOUT / SPCFORC output-component tables was inferred from text-extraction order and not visually verified (only the GLSTAT page was rendered to an image).

## F — LS-DYNA: field output, input echo, and diagnostics

### F-C01 — EXTENT_BINARY card 1b defaults

**Claim.** In R15, Card 1b of *DATABASE_EXTENT_BINARY carries NEIPH, NEIPS, MAXINT, STRFLG, SIGFLG, EPSFLG, RLTFLG, ENGFLG with defaults 0, 0, 3, 0, 1, 1, 1, 1 respectively.

> Variable NEIPH NEIPS MAXINT STRFLG SIGFLG EPSFLG RLTFLG ENGFLG Type I I I I I I I I Default 0 0 3 0 1 1 1 1

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (R15@7355dd456, 02/28/24)
- Locator: *DATABASE_EXTENT_BINARY, Card 1b, p. 16-60 (PDF p. 1954)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: Table flattened by text extraction; variable/default order read column-wise. Only R15 checked.

### F-C02 — NEIPH (solids and SPH)

**Claim.** NEIPH (default 0) is the number of extra integration-point history variables written to d3plot/d3part/d3drlf for solid elements AND SPH particles, in memory storage order.

> Number of additional integration point history variables written to the binary databases (d3plot, d3part, d3drlf) for solid elements and SPH particles. The integration point data is written in the same order that it is stored in memory; each material model has its own history variables that are stored.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *DATABASE_EXTENT_BINARY, NEIPH, p. 16-60 (PDF p. 1954)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: This is the only SPH-specific statement found on the EXTENT_BINARY card; no separate SPH output flag was found on this card.

### F-C03 — NEIPS

**Claim.** NEIPS (default 0) is the number of extra history variables written per integration point for shells and thick shells.

> Number of additional integration point history variables written to the binary databases (d3plot, d3part, d3drlf) for both shell and thick shell elements for each integration point; see NEIPH above and *DEFINE_MATERIAL_HISTORIES.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *DATABASE_EXTENT_BINARY, NEIPS, p. 16-60 (PDF p. 1954)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**

### F-C04 — MAXINT (shell through-thickness output)

**Claim.** MAXINT (default 3) is the number of shell/tshell through-thickness integration points written to d3plot; it does not govern STRFLG strain output. With MAXINT=3 and more than 3 points, top, bottom and neutral-axis results are written. Negative MAXINT writes |MAXINT| points at every in-plane integration point with no averaging.

> Number of shell and thick shell through-thickness integration points for which output is written to d3plot. This does not apply to the strain tensor output flagged by STRFLG. [...] results are output for the outermost (top) and innermost (bottom) integration points together with results for the neutral axis. [...] < 0 Any MAXINT integration points are output for each in plane integration point location and no averaging is used. This can greatly increase the size of the binary databases d3plot, d3thdt, and d3part.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *DATABASE_EXTENT_BINARY, MAXINT table, p. 16-61 (PDF p. 1955)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: Quote joins three cells of the MAXINT table; '[...]' marks omitted table cells.

### F-C05 — Shell output is in-plane averaged at element centre by default

**Claim.** For shells, with MAXINT=3 mid/inner/outer surface stresses are output at the element centre; in-plane integration points are averaged; for an even number of points the mid-surface value is the average of the two closest points. For MAXINT not equal to 3, stresses are output in stored order, after in-plane averaging.

> If MAXINT is set to 3, then mid-surface, inner-surface and outer-surface stresses are output at the center of the element. For an even number of integration points, the points closest to the center are averaged to obtain the midsurface values. If multiple integration points are used in the shell plane, the stresses at the center of the element are found by computing the average of these points. [...] If MAXINT is not equal to 3, then the stresses at the center of the element are output in the order that they are stored for the selected integration rule. If multiple points are used in plane, t…

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *DATABASE_EXTENT_BINARY, Remark 1 (MAXINT Field), p. 16-68 (PDF p. 1962)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**

### F-C06 — STRFLG digits

**Claim.** STRFLG (default 0) is read digit-wise as STRFLG = L + M*10 + N*100: L=1 writes strain tensors to d3plot, elout and dynain (shells: innermost and outermost point; solids: a single tensor); M=1 writes plastic strain tensor data to d3plot; N=1 writes thermal strain tensors. STRFLG=11 gives strain + plastic strain tensors.

> L.EQ.1: Write strain tensor data to d3plot, elout, and dynain. For shell and thick shell elements two tensors are written, one at the innermost and one at the outermost integration point. For solid elements a single strain tensor is written. M.EQ.1: Write plastic strain data to d3plot. N.EQ.1: Write thermal strain data to d3plot. [...] For STRFLG = 11 (011) LS-DYNA will write both strain and plastic strain tensors, but no thermal strain tensors.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *DATABASE_EXTENT_BINARY, STRFLG, pp. 16-61 to 16-62 (PDF pp. 1955-1956)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**

### F-C07 — Plastic/thermal strain tensor: averaging and support

**Claim.** The plastic and thermal strain tensors written under STRFLG are, for solids, the element-average strain (6 components, global system unless CMPFLG set); for shells the plastic strain is 3 plane-averaged tensors (bottom, middle, top). Support is limited to listed element/material combinations (plastic strain tensors: shells 2, 16, 23; solids 1, 2; materials 24, 255).

> a) For solids the element average strain in the global system having 6 components is written (local system if CMPFLG is set). b) For shells both plastic and thermal strains have 6 components. The thermal strain is written as a single tensor as in the solid case. The plastic strain output consists of 3 plane-averaged tensors: one for the bottom, one for the middle, and one for the top. [...] Currently, only the following element/materials combinations are supported but others will be added upon request.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *DATABASE_EXTENT_BINARY, Remark 11, p. 16-70 (PDF p. 1964)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: The support table is badly flattened by extraction; my reading is plastic-strain tensors: shells 2,16,23 / solids 1,2 / materials 24,255; thermal: same elements, materials 'Add thermal expansion, 255'. Verify against the PDF page before relying on the material list. Note MAT_010, MAT_003 and MAT_072R3 are NOT in the listed materials.

### F-C08 — Strain tensor definition

**Claim.** The d3plot strain tensor is a time-integrated rate-of-deformation (Jaumann rate for solids, co-rotational rate for shells), and the manual warns it is not invariant to element formulation.

> The strain tensors that are output to the d3plot database are calculated using proper time integration of the rate-of-deformation tensor D. [...] This should be kept in mind when interpreting the results since they are not invariant to changes in element formulations and possibly nodal connectivities.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *DATABASE_EXTENT_BINARY, Remark 10, p. 16-70 (PDF p. 1964)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: Math symbols (epsilon, D) normalised from PDF glyphs.

### F-C09 — SIGFLG / EPSFLG / RLTFLG / ENGFLG

**Claim.** SIGFLG (default 1 = include stress tensor; 2 = exclude for shells; 3 = exclude for shells and solids). EPSFLG same pattern for effective plastic strain. RLTFLG (1 include shell stress resultants, 2 exclude). ENGFLG (1 include shell/tshell/beam internal energy density and shell thickness, 2 exclude). All defaults are 'include'.

> SIGFLG Flag for including the stress tensor for shells and solids. EQ.1: Include (default), EQ.2: Exclude for shells, include for solids. EQ.3: Exclude for shells and solids. EPSFLG Flag for including the effective plastic strains for shells and solids: EQ.1: Include (default), EQ.2: Exclude for shells, include for solids. EQ.3: Exclude for shells and solids. RLTFLG Flag for including stress resultants in the shell LS-DYNA database: EQ.1: Include (default), EQ.2: Exclude. ENGFLG Flag for including shell, tshell, and beam internal energy density and shell thickness: EQ.1: Include (default), EQ…

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *DATABASE_EXTENT_BINARY, p. 16-62 (PDF p. 1956)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**

### F-C10 — NINTSLD: solid stress is element average by default

**Claim.** NINTSLD default is 1. For any value other than 8, solid integration-point values are averaged and only the averages are written; NINTSLD=8 is required to get individual integration-point values, even for solids with fewer than 8 points.

> Number of solid element integration points written to the LS-DYNA database. When NINTSLD is set to 1 (default) or to any value other than 8, integration point values are averaged and only those averages are written output. To obtain values for individual integration points, set NINTSLD to 8, even if the multi-integration point solid has fewer than 8 integration points.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *DATABASE_EXTENT_BINARY, Card 3, NINTSLD, p. 16-64 (PDF p. 1958)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: Card 3 is optional. Integration point ordering is given in Remark 8 of the same keyword (point #1 closest to node #1, etc.).

### F-C11 — IEVERP / DCOMP / N3THDT

**Claim.** Card 2 defaults: CMPFLG 0, IEVERP 0, BEAMIP 0, DCOMP 1, SHGE 1, STSSZ 1, N3THDT 2, IALEMAT 1. IEVERP=1 writes one state per d3plot file. DCOMP=1 (default) no rigid-body data compression; values 3, 4 and 6 eliminate ALL nodal velocities and accelerations from the database. N3THDT=2 (default) writes material energy to d3thdt; 1 turns it off.

> DCOMP Data compression to eliminate rigid body data: EQ.1: Off (default), no rigid body data compression, EQ.2: On, rigid body data compression active, EQ.3: Off, no rigid body data compression, but all nodal velocities and accelerations are eliminated from the database. [...] N3THDT Flag for including material energy in d3thdt database: EQ.1: Off, energy is not written to d3thdt database, EQ.2: On (default), energy is written to d3thdt database.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *DATABASE_EXTENT_BINARY, Card 2, pp. 16-62 to 16-64 (PDF pp. 1956-1958)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: IEVERP text: 'EQ.0: More than one state can be on each plot file, EQ.1: One state only on each plot file.'

### F-C12 — DELERES and HYDRO

**Claim.** DELERES default 0 means results of deleted elements are written as all zero; DELERES=1 writes the last available results. HYDRO (default 0) adds 3, 5 or 7 shock-physics history variables as the LAST history variables in d3plot (HYDRO=1: internal energy per reference volume, reference volume, bulk-viscosity pressure; =2 adds relative volume and current density; =4 adds volumetric strain and hourglass energy per unit initial volume).

> DELERES Output flag for results of deleted elements: EQ.0: No results output (all zero) EQ.1: Last available results, such as stresses and history variables, are written to d3plot and d3part.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *DATABASE_EXTENT_BINARY, Card 4 DELERES p. 16-68 (PDF p. 1962); HYDRO p. 16-65 (PDF p. 1959)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: HYDRO quote (p. 16-65): 'Either 3, 5 or 7 additional history variables useful to shock physics are output as the last history variables to d3plot (does not apply to elout).'

### F-C13 — d3plot precision

**Claim.** Per the R15 manual, a double-precision executable writes binary output (d3plot, d3drlf) in 64-bit format BY DEFAULT (IBINARY=0); 32-bit IEEE output is the non-default option IBINARY=1 on *DATABASE_FORMAT, or environment variable LSTC_BINARY=32ieee. IBINARY applies only to double-precision executables.

> IBINARY Flag to control the word size in the binary output files, such as d3plot and d3drlf. This variable applies only to double precision LS-DYNA executables. EQ.0: 64 bit format for binary output (default for double precision), EQ.1: 32 bit IEEE format for binary output. This reduces the volume of binary output from double precision executables by a factor of two. [...] As an alternative to setting IBINARY = 1, the user may set the system environment variable LSTC_BINARY to 32ieee.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *DATABASE_FORMAT, IBINARY and Remark 2, pp. 16-84 to 16-85 (PDF pp. 1978-1979)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: This contradicts the common assumption that d3plot is always single precision. The manual does not state the word size written by a single-precision executable (implied 32-bit, not quoted). A site/launcher may set LSTC_BINARY=32ieee, so the actual word size of a given archive must be checked from the file, not assumed.

### F-C14 — d3hsp definition

**Claim.** d3hsp is the default name of the 'high speed printer file' (command-line O=). The input data is printed into it during the second input phase, where most input checking happens; the manual tells users to check d3hsp or messag for the word 'Error'.

> Considerably more checking is done during the second phase where the input data is printed out. Since LS-DYNA has retained the option of reading older non-keyword input files, we print out the data into the output file d3hsp (default name) as in previous versions of LS-DYNA. [...] The user should always check either output file, d3hsp or messag, for the word "Error".

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: Getting Started, p. 2-3 (PDF p. 365); execution syntax 'otf = High speed printer file (default = d3hsp)' p. 2-12 (PDF p. 374)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: The manual does not use the phrase 'echo with defaults applied'; it says the input data 'is printed out' into d3hsp. Whether every defaulted field appears with its resolved value was not found stated.

### F-C15 — d3hsp verbosity: NPOPT, NEECHO, echo file

**Claim.** *CONTROL_OUTPUT NPOPT (default 0 = no suppression) controls d3hsp printing; NPOPT=1 suppresses nodal coordinates, element connectivities, rigid walls, nodal SPCs, initial velocities, initial strains, etc. NEECHO (default 0 = all data printed) controls a SEPARATE echo file (command line E=efl), not d3hsp.

> NPOPT Print suppression during input phase flag for the d3hsp file: EQ.0: No suppression, EQ.1: Nodal coordinates, element connectivities, rigid wall definitions, nodal SPCs, initial velocities, initial strains, adaptive constraints, and SPR2/SPR3 constraints are not printed. NEECHO Print suppression during input phase flag for echo file: EQ.0: All data printed, EQ.1: Nodal printing is suppressed, EQ.2: Element printing is suppressed, EQ.3: Both nodal and element printing is suppressed.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *CONTROL_OUTPUT, Card 1, pp. 12-425 to 12-426 (PDF pp. 1703-1704); echo file: Getting Started p. 2-13 (PDF p. 375) 'efl = Echo file containing optional input echo with or without node/element data'
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**

### F-C16 — Other d3hsp content controls

**Claim.** IPNINT controls how many element initial time steps are printed to d3hsp (default: 100 smallest; 1 = all). IKEDIT (default 100) is the status-report interval to d3hsp, ignored if glstat is written. HISNOUT>=1 writes history-variable names per part to d3hsp (2/3 also write hisnames.xml / d3labels.xml). IPCURV=1 writes digitized curve data to messag and d3hsp.

> IPNINT Flag controlling output of initial time step sizes for elements to d3hsp: EQ.0: 100 elements with the smallest time step sizes are printed. EQ.1: Time step sizes for all elements are printed. GT.1: IPNINT elements with the smallest time step sizes are printed.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *CONTROL_OUTPUT, IPNINT/IKEDIT p. 12-427 (PDF p. 1705); IPCURV p. 12-428; HISNOUT pp. 12-433 to 12-434 (PDF pp. 1711-1712)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: HISNOUT quote: 'this option allows the output of the history variable names, listed for each part separately, to d3hsp.' and 'EQ.0: No output (default) EQ.1: Information written to d3hsp'.

### F-C17 — *PARAMETER echo to d3hsp

**Claim.** Defined parameters are echoed to d3hsp by default; the _NOECHO keyword option suppresses that echo.

> NOECHO option. The NOECHO keyword option tells LS-DYNA to not echo the defined parameters to the d3hsp file. This feature is useful for indicating which parameters in an encrypted file should not be echoed.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *PARAMETER, Remark 7, p. 35-4 (PDF p. 3396)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: That parameters ARE echoed by default is an inference from this remark; the manual does not show the d3hsp format of the echo.

### F-C18 — *PARAMETER reference syntax

**Claim.** A parameter is referenced anywhere in the input by '&' immediately before its name; '-&' flips the sign; a character parameter embedded in a larger string starts with '&' and ends with '^'. Names are up to 9 characters (letters, numbers, '_', not starting with a number), with a type prefix R/I/C in the definition.

> Parameters can be referenced anywhere in the input by placing an "&" immediately preceding the parameter name. If a minus sign "-" is placed directly before "&", i.e., "-&", with no space the sign of the numerical value will be switched. [...] The included character parameter should start with '&' and end with '^'.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *PARAMETER, Remarks 1-2, p. 35-3 (PDF p. 3395)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: A detector for unresolved constructs must not treat every '&' as a parameter: the manual also uses '&' in column 1 to flag some optional cards (e.g. 'The keyword reader identifies this card by an "&" in the first column', Vol I text near *DEFINE/CONTACT cards). *PARAMETER_EXPRESSION was not reviewed.

### F-C19 — *PARAMETER scoping across *INCLUDE

**Claim.** A non-LOCAL parameter is visible at any later point in input processing including include files; *PARAMETER_LOCAL parameters disappear when the parser finishes the file in which they appear and can mask non-LOCAL ones; include files can redefine values (manual example: VAL1 redefined to 10.0 in file1 remains 10.0 after returning to main.k).

> Parameters defined with the LOCAL versions disappear when the input parser finishes reading the file in which they appear. LOCAL variables can temporarily mask non-LOCAL variables.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *PARAMETER, Remark 5, p. 35-4 (PDF p. 3396)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**

### F-C20 — *INCLUDE

**Claim.** *INCLUDE includes independent input files; options are <BLANK>, BINARY, NASTRAN, TRANSFORM, TRANSFORM_BINARY; TRANSFORM variants offset IDs and transform/scale coordinates and constitutive parameters, so the resolved model can differ from the literal include-file content.

> Purpose: Include independent input files containing model data. [...] The TRANSFORM and TRANSFORM_BINARY options allow for node, element, and set IDs to be offset and for coordinates and constitutive parameters to be transformed and scaled.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *INCLUDE_{OPTION}, p. 26-3 (PDF p. 2837)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**

### F-C21 — Solver-written resolved deck (dyna.str)

**Claim.** *CONTROL_STRUCTURED (or 'outdeck = s' on the execution line) makes the solver write a STRUCTURED-format (not keyword-format) input deck named dyna.str that is 'largely or wholly equivalent' to the keyword deck; not all features are supported and some IDs (e.g. load curve numbers) are in internal numbering. _TERM terminates after writing it.

> Purpose: Write out an LS-DYNA structured input deck that is largely or wholly equivalent to the keyword input deck. [...] The name of the structured input deck is "dyna.str". Not all LS-DYNA features are supported in structured input format. Some data such as load curve numbers will be output in an internal numbering system. If the TERM option is activated, termination will occur after the structured input deck is written. Adding "outdeck = s" to the LS-DYNA execution line serves the same purpose as including *CONTROL_STRUCTURED in the keyword input deck.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *CONTROL_STRUCTURED, p. 12-539 (PDF p. 1817)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: This is the only solver-side 'resolved deck' option I found in Vol I. It is not a flattened KEYWORD file. No solver option writing a single resolved keyword-format file was found.

### F-C22 — Message file names SMP vs MPP

**Claim.** The message file is 'messag' in SMP; in MPP there is one per processor, named mesnnnn (written 'mesxxxx' and 'mes****' elsewhere in the same manual), normally in the pfile's local directory unless the pfile option global_message_files is set.

> Some of the normal LS-DYNA files will have corresponding collections of files produced by MPP/LS-DYNA, with one per processor. These include the d3dump files (new names = d3dump.nnnn), the messag files (now mesnnnn) and others. Most of these will be found in the local directory specified in the pfile.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: Appendix O (MPP), p. 64-3 (PDF p. 3969); also Appendix P p. 65-5 (PDF p. 3987): 'The message files, messag in SMP and mesxxxx in MPP'
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Corrected locator: Add: global_message_files at Appendix O, PDF p. 3971 (approx. p. 64-5)
- Caveats: The literal name 'mes0000' appears in the R15 manual only in release-note bullets (e.g. 'Add output of performance statistics for the MPP implicit eigensolver to mes0000'), not in a definition.

### F-C23 — MSGMAX: warning/error message truncation

**Claim.** *CONTROL_OUTPUT MSGMAX (default 50) caps the number of each error/warning message: if >0 the cap applies to screen output only and ALL messages go to d3hsp/messag; if <=0 the cap also applies to d3hsp/messag. So with default settings every occurrence is in d3hsp/messag.

> MSGMAX Maximum number of each error/warning message: GT.0: Number of messages to screen output; all messages written to d3hsp/messag LE.0: Number of messages to screen output and d3hsp/messag EQ.0: Default, 50

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *CONTROL_OUTPUT, Card 2, MSGMAX, p. 12-428 (PDF p. 1706)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: The text is ambiguous for EQ.0 (it is both 'LE.0' and 'Default, 50'); whether MSGMAX=0 truncates d3hsp/messag at 50 is not resolved by the manual text. Setting an explicit positive MSGMAX avoids the ambiguity.

### F-C24 — MSGFLG / d3msg

**Claim.** Standard-length error/warning messages are written to messag or mes****; MSGFLG=1 additionally writes detailed messages to a file d3msg at the end of the run, each only once.

> MSGFLG Flag for writing detailed error/warning message to d3msg. MSGFLG has no effect on output of standard length error/warning messages; such messages are written to messag or mes****. NOTE: Most errors/warnings offer only standard length messages. Only a few also offer optional, detailed messages. EQ.0: Do not write detailed messages to d3msg. EQ.1: Write detailed messages to d3msg at the conclusion of the run. Each detailed message is written only once even in cases where the associated error or warning occurs multiple times.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *CONTROL_OUTPUT, MSGFLG, p. 12-430 (PDF p. 1708)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**

### F-C25 — NaN checking (ISNAN)

**Claim.** *CONTROL_SOLUTION ISNAN=1 activates a NaN check on the assembled force and moment arrays (about 2% cost); default 0 = no checking.

> ISNAN Flag to check for a NaN in the force and moment arrays after the assembly of these arrays is completed. This option can be useful for debugging purposes. A cost overhead of approximately 2% is incurred when this option is active. EQ.0: No checking EQ.1: Checking is active.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *CONTROL_SOLUTION, ISNAN, p. 12-516 (PDF p. 1794)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**

### F-C26 — NaN / out-of-range message text

**Claim.** Per the LS-DYNA support site FAQ, with ISNAN=1 (or command-line checknan=1) the screen shows '*** NaN detected. Please check message file from processor for detail.' and the per-processor message file contains '*** termination due to out-of-range forces' followed by the number of nodes and a node list.

> Output to messag-files (e.g. grep 'out-of-range' mes0030): *** termination due to out-of-range forces number of nodes has out-of-range forces 364 Node list:

- Source: *LS-DYNA runs into Segmentation Violation (SIGSEGV) error. What can I do? (LS-DYNA support site FAQ, dated 'gp 04/14')* — 2014
- Locator: Section 'Checknan option' / 'Which output is produced by checknan option?'
- URL: <https://www.dynasupport.com/faq/general/SIGSEGV>
- Kind / confidence: vendor-support / medium · Independent check: **confirmed**
- Caveats: Read via WebFetch (an intermediary model transcribes the page); I asked for verbatim reproduction but could not fetch the raw HTML (curl to dynasupport.com failed). Exact whitespace not guaranteed; message wording may differ by solver release (page is from 2014). The companion page https://www.dynasupport.com/howtos/general/not-a-number-nan-1 says: 'by activating ISNAN (*CONTROL_SOLUTION), nodes with force or moment arrays are reported (out-of-r…

### F-C27 — Element failure/deletion messages

**Claim.** Beam, shell and solid element failure messages are written to d3hsp and the message files by default; *CONTROL_MPP_IO_NOBEAMOUT (pfile: nobeamout) suppresses them.

> Purpose: Suppress beam, shell, and solid element failure messages in the d3hsp and message files. There are no parameters for this keyword.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *CONTROL_MPP_IO_NOBEAMOUT, p. 12-414 (PDF p. 1692)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: The exact text of the failure message is not given in the manual.

### F-C28 — Initial penetrations: IGNORE

**Claim.** *CONTROL_CONTACT IGNORE (default 0) for *CONTACT_AUTOMATIC options: 0 = move nodes to eliminate initial penetrations; 1 = allow and track them; 2 = allow and track, and print penetration warning messages with original and recommended coordinates of each penetrating node. 'Initial' means the first time step a penetration is encountered. Not implemented in SMP for AUTOMATIC_GENERAL. Can be overridden per contact on *CONTACT optional card C.

> EQ.0: Move nodes to eliminate initial penetrations in the model definition. EQ.1: Allow initial penetrations to exist by tracking the initial penetrations. EQ.2: Allow initial penetrations to exist by tracking the initial penetrations. However, penetration warning messages are printed with the original coordinates and the recommended coordinates of each penetrating node given.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *CONTROL_CONTACT, Card 4, IGNORE, pp. 12-72 to 12-73 (PDF pp. 1351-1352)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Corrected locator: *CONTROL_CONTACT Card 4, IGNORE: p. 12-73 only (PDF p. 1351); Card header 'This card is optional.'
- Caveats: Important consequence for V&V: with IGNORE=0 the solver silently MOVES nodes, so the solved geometry differs from the input deck. For Mortar contact only, Appendix P (p. 65-20/21, PDF p. 4002) states 'initial penetrations are always reported in the message file(s), including the maximum penetration and how initial penetrations are to be handled'; I found no equivalent 'always reported' statement for non-Mortar contacts.

### F-C29 — Negative volume handling

**Claim.** Per the LS-DYNA support site, a negative volume causes the calculation to terminate unless ERODE=1 in *CONTROL_TIMESTEP and DTMIN in *CONTROL_TERMINATION is nonzero, in which case the offending element is deleted and the run usually continues. The R15 manual likewise describes PSFAIL on *CONTROL_SOLID as limiting negative-volume erosion to a part set, 'similar to setting ERODE = 1 in *CONTROL_TIMESTEP'.

> A negative volume calculation in LS-DYNA will cause the calculation to terminate unless ERODE in *CONTROL_TIMESTEP is set to 1 and DTMIN in *CONTROL_TERMINATION is set to any nonzero value

- Source: *Negative volumes in brick elements (LS-DYNA support site how-to)* — undated web page, fetched 2026-09-21
- Locator: second paragraph
- URL: <https://www.dynasupport.com/howtos/element/negative-volumes-in-brick-elements>
- Kind / confidence: vendor-support / medium · Independent check: **confirmed**
- Corrected locator: PSFAIL corroboration: Vol I R15 *CONTROL_SOLID, PDF p. 1788 (approx. p. 12-510), not ~1790
- Caveats: Read via WebFetch, not raw HTML. The exact wording of the negative-volume message in messag/d3hsp was NOT found. Manual corroboration (Vol I R15, *CONTROL_SOLID PSFAIL, PDF p. ~1790): 'Solid element erosion from negative volume is limited only to solid elements in the part set indicated by PSFAIL. This is similar to setting ERODE = 1 in *CONTROL_TIMESTEP, except that it is not global.'

### F-C30 — Restart: categories and command lines

**Claim.** Three restart categories: simple restart (R=rtf, no input deck), small restart (I=restartinput R=D3DUMPnn, deck contains only restart keywords such as *CHANGE_OPTION), full restart (full model plus *STRESS_INITIALIZATION; SMP: R=D3DUMPnn, MPP: N=D3FULLnn). File-family members are numbered consecutively from the last member before termination. d3dump is written at the end of every run and as requested by *DATABASE_BINARY_D3DUMP; runrsf is the running restart file.

> All modifications to the problem made with the restart input deck will be reflected in subsequent restart dumps. All the members of the file families are consecutively numbered beginning from the last member prior to termination. For a small restart run a small input deck replaces the standard input deck on the execution line which must have at least the following command line arguments: LS-DYNA I=restartinput R=D3DUMPnn

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: RESTART INPUT DATA, p. 48-1 (PDF p. 3797); Getting Started 'RESTART ANALYSIS' pp. 2-20 to 2-22 (PDF pp. 382-384); execution syntax p. 2-12/2-13 (PDF pp. 374-375): 'dpf = Dump file to write for purposes of restarting (default = d3dump). This file is written at the end of every run and during the run as requested by *DATABASE_BINARY_D3DUMP.'
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**

### F-C31 — Restart: traces in outputs

**Claim.** In a full restart, time and output intervals continue from the first job (time is not reset) and 'completely new databases will be generated with the time offset'; by default a full restart OVERWRITES existing ASCII output (*CHANGE/…IASCII=0) and appends only if IASCII=1. A small restart's d3hsp carries the heading given by the restart *TITLE (default 'LS-DYNA USER INPUT'), which is written to no other file.

> Time and output intervals are continuous with job1; that is, the time is not reset to zero. [...] Completely new databases will be generated with the time offset.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: Getting Started, RESTART ANALYSIS, p. 2-22 (PDF p. 384); IASCII: RESTART INPUT DATA p. 48-6 (PDF p. 3802) 'EQ.0: Full restart overwrites existing ASCII output (default), EQ.1: Full restart appends to existing ASCII output.'; restart *TITLE p. 48-40 (PDF p. 3836) 'Heading to appear in a small restart's d3hsp file. This heading is not written to any other output file.'
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Corrected locator: IASCII is on *CHANGE_OUTPUT ('ASCII Output Overwrite Card. This format applies to the OUTPUT keyword option.'), p. 48-6 (PDF 3802)
- Caveats: I did not identify which restart keyword carries IASCII (it is in the *CHANGE/*CONTROL section near p. 48-6; check the card name before citing). No statement was found describing a specific 'restart' banner line in d3hsp/messag by which segments can be identified.

### F-C32 — MAT_010 card layout

**Claim.** *MAT_ELASTIC_PLASTIC_HYDRO: Card 1 = MID, RO, G, SIG0, EH, PC, FS, CHARL. Card 2 = A1, A2, SPALL and is 'included if and only if the SPALL keyword option is used'. Cards 3-4 = EPS1-EPS16, Cards 5-6 = ES1-ES16, all four required. So in a plain *MAT_ELASTIC_PLASTIC_HYDRO (no _SPALL) there is no A1/A2/SPALL card at all.

> Card 2. This card is included if and only if the SPALL keyword option is used. A1 A2 SPALL Card 3. This card is required. EPS1 EPS2 EPS3 EPS4 EPS5 EPS6 EPS7 EPS8

- Source: *LS-DYNA Keyword User's Manual Volume II, Material Models* — R15 (R15@d71677e2e, 02/29/24)
- Locator: *MAT_010 Card Summary, p. 2-175 (PDF p. 243)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_II_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: Card 1 defaults: SIG0 0.0, EH 0.0, PC -infinity ('If zero, a cutoff of -inf is assumed'), FS 0.0, CHARL 0.0. SPALL EQ.0.0 defaults to 1.0. An archive deck that places A1/A2/SPALL under the non-SPALL keyword would be mis-read (the card would be taken as EPS1-8).

### F-C33 — MAT_010 pressure hardening vs EPS/ES table

**Claim.** The manual gives the pressure-hardening term only in the SIG0/EH branch: sigma_y = sigma_0 + E_h*eps_p + (a1 + p*a2)*max[p,0], used 'If ES and EPS values are undefined'. When ES and EPS are specified, the manual says EH is ignored and gives the yield stress as sigma_y = f(eps_p), interpolated from the curve, with no a1/a2 term shown. A1 = 'Linear pressure hardening coefficient', A2 = 'Quadratic pressure hardening coefficient'. Pressure p positive in compression.

> If ES and EPS values are undefined, the yield stress and plastic hardening modulus are taken from SIG0 and EH. [...] If ES and EPS are specified, a curve like that shown in Figure M10-1 may be defined. [...] In this case the plastic hardening modulus on Card 1 is ignored and the yield stress is given as [sigma_y = f(eps_p)], where the value for f(eps_p) is found by interpolating the data curve.

- Source: *LS-DYNA Keyword User's Manual Volume II, Material Models* — R15 (02/29/24)
- Locator: *MAT_010 Remark 2 (Yield Stress and Plastic Hardening Modulus), p. 2-179 (PDF p. 247)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_II_R15.pdf>
- Kind / confidence: primary / medium · Independent check: **confirmed**
- Caveats: The manual does NOT explicitly say 'A1 and A2 are ignored when the table is used'; that is an inference from the absence of the pressure term in the tabulated-branch formula. Equations were reconstructed from garbled PDF glyphs (bracketed part is my transcription); check PDF p. 247 visually. The Theory Manual was not consulted and might state the behaviour explicitly.

### F-C34 — MAT_010 effective plastic strain definition

**Claim.** For MAT_010 effective stress is sqrt(3/2 s_ij s_ij) and effective plastic strain is the time integral of sqrt(2/3 Dp_ij Dp_ij); EPSi are 'Effective plastic strain (true)' with linear extrapolation beyond the last point; FS is the effective plastic strain at which erosion occurs.

> EPSi Effective plastic strain (true). Define up to 16 values. Care must be taken that the full range of strains expected in the analysis is covered. Linear extrapolation is used if the strain values exceed the maximum input value.

- Source: *LS-DYNA Keyword User's Manual Volume II, Material Models* — R15 (02/29/24)
- Locator: *MAT_010 EPSi p. 2-177 (PDF p. 245); definitions in Remark 2 p. 2-179 (PDF p. 247)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_II_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: The R15 MAT_010 section contains NO history-variable table (section ends after Remark 3 on spall models, p. 2-180).

### F-C35 — MAT_072R3 'plastic strain' slot / NOUT

**Claim.** For *MAT_CONCRETE_DAMAGE_REL3 the quantity LS-PrePost labels 'plastic strain' is selected by NOUT (Card 3): 1 = current shear failure surface radius; 2 = scaled damage measure delta = 2*lambda/(lambda + lambda_m); 3 = strain energy (rate); 4 = plastic strain energy (rate). The scaled damage measure ranges 0 to 1 from yield to maximum failure surface and 1 to 2 from maximum to residual surface.

> The "scaled damage measure" ranges from 0 to 1 as the material transitions from the yield failure surface to the maximum failure surface, and thereafter ranges from 1 to 2 as the material ranges from the maximum failure surface to the residual failure surface. [...] The quantity labeled as "plastic strain" by LS-PrePost is actually the quantity described in Table M72-1, in accordance with the input value of NOUT (see Card 3 above).

- Source: *LS-DYNA Keyword User's Manual Volume II, Material Models* — R15 (02/29/24)
- Locator: *MAT_072R3, 'Output of Selected Variables' and Table M72-1, pp. 2-518 to 2-519 (PDF pp. 586-587)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_II_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: The slot holds the scaled damage measure ONLY if NOUT=2. The Card 3 default row lists NOUT default as 'none'; what is output when NOUT is blank/0 (as in the manual's own sample input) is not stated in the manual - do not assume it is the damage measure. Formula delta = 2*lambda/(lambda+lambda_m) transcribed from PDF glyphs.

### F-C36 — MAT_072R3 RSIZE / UCF

**Claim.** RSIZE = unit conversion factor for length in inches per user length unit (39.37 for metres); UCF = unit conversion factor for stress in psi per user stress unit (145 for MPa). The manual's mm-ms-g-MPa sample uses RSIZE 3.94E-2 and UCF 145.0.

> RSIZE Unit conversion factor for length (inches/user-unit). For example, set to 39.37 if user length unit in meters. UCF Unit conversion factor for stress (psi/user-unit). For instance set to 145 if f'c in MPa.

- Source: *LS-DYNA Keyword User's Manual Volume II, Material Models* — R15 (02/29/24)
- Locator: *MAT_072R3, Card 3 variables, p. 2-516 (PDF p. 584); sample input p. 2-519/2-520 (PDF pp. 587-588)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_II_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: Defaults for RSIZE and UCF are listed as 'none'.

### F-C37 — MAT_072R3 extra history variables and d3hsp echo

**Claim.** Six extra history variables are available with NEIPH=6: #1 internal energy, #2 pressure from bulk viscosity, #3 volume in previous time step, #4 plastic volumetric strain, #5 slope of damage evolution (eta vs lambda) curve, #6 'modified' effective plastic strain (lambda). Auto-generated model parameters and the generated *EOS_TABULATED_COMPACTION parameters are written to d3hsp.

> An additional six extra history variables as shown in Table M72-2 may be written by setting NEIPH = 6 on the keyword *DATABASE_EXTENT_BINARY. [...] These generated material parameters, along with the generated parameters for *EOS_TABULATED_COMPACTION, are written to the d3hsp file.

- Source: *LS-DYNA Keyword User's Manual Volume II, Material Models* — R15 (02/29/24)
- Locator: *MAT_072R3, Table M72-2 p. 2-519 (PDF p. 587); intro p. 2-514 (PDF p. 582)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_II_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: This means d3hsp is the ONLY record of the actual K&C parameters when auto-generation (from f'c alone) is used - directly relevant to the 'echo of resolved input' requirement.

### F-C38 — MAT_003 BETA, SRC, SRP, VP

**Claim.** *MAT_PLASTIC_KINEMATIC: Card 1 = MID, RO, E, PR, SIGY, ETAN, BETA (ETAN, BETA default 0.0); Card 2 = SRC, SRP, FS, VP (defaults 0.0, 0.0, 1e20, 0.0). BETA is the hardening parameter, 0 = kinematic, 1 = isotropic. SRC and SRP are Cowper-Symonds C and p; the yield stress is scaled by 1 + (eps_dot/C)^(1/p); set both to zero to ignore rate effects. VP=0 scales yield stress (default), VP=1 viscoplastic formulation (recommended).

> Strain rate is accounted for using the Cowper and Symonds model which scales the yield stress with the factor [1 + (eps_dot/C)^(1/p)] where eps_dot is the strain rate. A fully viscoplastic formulation is optional which incorporates the Cowper and Symonds formulation within the yield surface. To ignore strain rate effects set both SRC and SRP to zero. [...] Kinematic, isotropic, or a combination of kinematic and isotropic hardening may be specified by varying beta' between 0 and 1.

- Source: *LS-DYNA Keyword User's Manual Volume II, Material Models* — R15 (02/29/24)
- Locator: *MAT_003, Cards 1-2 p. 2-156, Remarks 1-2 pp. 2-157 to 2-158 (PDF pp. 224-226)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_II_R15.pdf>
- Kind / confidence: primary / high · Independent check: **confirmed**
- Caveats: Bracketed formula transcribed from PDF glyphs. Variable table says 'BETA Hardening parameter, 0 < beta' < 1'. FS default printed as '1020' in extraction = 10^20. History variables (Remark 3): back stress components xx, yy, xy, yz, zx (5 extra history variables). Note MAT_010's Remark 2 separately says its bilinear curve corresponds to 'hardening parameter, beta = 1' (isotropic).

**Verifier summary.** All 38 claims checked out; none was hallucinated or misattributed. I checked 36 of them against the two R15 manual PDFs downloaded from the cited lsdyna.ansys.com URLs. The other two (C26, C29) I could only re-read through WebFetch. - **Volume I:** R15@7355dd456, 02/28/24, 4084 pages. - **Volume II:** R15@d71677e2e, 02/29/24, 2153 pages. - **Method:** I extracted the full text page by page with PyMuPDF and read every cited page. Table column order and equations were read from that extraction, not by visual inspection. **Corrections and nuances to carry forward** - **C04:** "Any" in the quoted "`< 0` Any MAXINT integration points" is the table's number-of-points column cell, not part of the sentence. The manual writes "MAXINT integration points", not |MAXINT|. - **C28:** IGNORE is entirely on p. 12-73 (PDF 1351). The manual says "third optional card" under `*CONTACT`, not "card C". - **C29:** the PSFAIL corroboration is at Vol I PDF p. 1788, not ~1790. The support page also says a negative volume "may still occur and cause a premature termination" even with ERODE and DTMIN set. - **C31:** IASCII belongs to `*CHANGE_OUTPUT` (p. 48-6, PDF 3802). - **C22:** the `global_message_files` pfile option is at PDF p. 3971. - **C32:** MAT_010 also has a STOCHASTIC option, which the claim omits. - **C30:** "d3dump written at the end of every run" is qualified by `d=nodump` and `*CONTROL_MPP_IO_NOD3DUMP`. **Statements the claimants inferred that the manual does not make** - Parameters are echoed to d3hsp by default (C17). - Element failure messages are written "by default" (C27). - A1/A2 are ignored when the EPS/ES table is used (C33). - d3hsp is the "only" record of the generated K&C parameters (C37). - With the default MSGMAX, every message occurrence reaches d3hsp/messag (C23). The text is ambiguous for a literal 0. - What NOUT outputs when left blank (C35). **C26 and C29 (dynasupport.com)** Raw curl to dynasupport.com hung, so I re-read both pages only through WebFetch. WebFetch is an intermediary-model transcription, not raw HTML. The message texts matched the claims, but exact whitespace and wording are not verified and the FAQ dates from 04/2014. Treat both at medium c…

**Not found or not accessible.**

- Termination banner text ('N o r m a l t e r m i n a t i o n' / 'E r r o r t e r m i n a t i o n'): NOT FOUND in any source I could read. grep of R15 Vol I and Vol II text for 't e r m', 'N o r', 'E r r' returned nothing; dynasupport 'Getting Started' tutorial page has no such text; web searches returned only forum paraphrases. FROM MEMORY - UNVERIFIED: the banners are letter-spaced as in the question. Must be confir…
- Warning/error COUNT summary at the end of d3hsp/messag: NOT FOUND. The R15 manual only tells the user to search d3hsp or messag for the word 'Error' (p. 2-3) and describes MSGMAX/MSGFLG. No statement that a count summary is printed. Confirm from a real run's output.
- Exact text of the negative-volume message (e.g. 'negative volume in solid element #') and of element-deletion/failure messages: NOT FOUND in the manual or the dynasupport page text I could obtain; only the behaviour is documented (C27, C29).
- 'Out-of-range velocities' wording: NOT FOUND. The only message text obtained is 'termination due to out-of-range forces' (dynasupport SIGSEGV FAQ, 2014, read via WebFetch); an Arup conference paper (Newlands, 15th Int. LS-DYNA Conf. 2018, dynalook.com) mentions termination 'due to out of range moments' in prose only (secondary).
- Whether d3hsp contains the content of *INCLUDE files and fully resolved values of every defaulted field: NOT FOUND as an explicit statement. Established only: input data is 'printed out' into d3hsp in the second input phase (p. 2-3), parameters are echoed unless _NOECHO, NPOPT=1 suppresses bulk data. Needs checking against a real d3hsp.
- An LS-PrePost (or solver) option to write a single fully resolved KEYWORD-format file with includes merged and parameters substituted: NOT FOUND / not researched in LS-PrePost documentation. Only the solver's structured-format dyna.str (*CONTROL_STRUCTURED, 'outdeck = s') was found. FROM MEMORY - UNVERIFIED: LS-PrePost can save a keyword file with includes merged; no source read.
- How a restart is recorded inside d3hsp/messag (a restart banner/line identifying the segment): NOT FOUND. Only found: restart *TITLE appears as heading of a small restart's d3hsp; file families are numbered consecutively; full restart generates new databases with a time offset; IASCII overwrite/append.
- MAT_010 history variables and what the 'effective plastic strain' output slot holds: the R15 Vol II MAT_010 section has NO history-variable table and no statement about the output slot. The dynasupport history-variables page (www.dynasupport.com/howtos/material/history-variables, referenced by the manual under HISNOUT) was not fetched. HISNOUT>=1 on *CONTROL_OUTPUT would make the solver print the names into d3hsp.
- Whether A1/A2 act when the EPS/ES table is used in MAT_010: only inferable (C33); no explicit statement found in Vol II R15. LS-DYNA Theory Manual not consulted.
- MAT_072R3 NOUT default behaviour (what the plastic-strain slot contains when NOUT is blank or 0): NOT FOUND; the manual lists default 'none'.
- Single-precision executable d3plot word size: not explicitly stated in the passages read (IBINARY 'applies only to double precision'); 32-bit is implied but not quoted.
- Releases R11-R14: not read. All quotes are from R15 only. The ftp.lstc.com manual directory returned HTTP 403; dynasupport.com PDFs could not be downloaded with curl from this machine. Note: a background curl I started for the R13 PDFs into <local scratch folder> reported exit 0 after I had moved on; I did not use or inspect its output. That folder also contains files (vol1_r13.pdf, vol1_r15.pdf, .txt extracts, etc.) wr…
- SPH-specific d3plot output flags beyond NEIPH: none found on *DATABASE_EXTENT_BINARY in R15; *CONTROL_SPH and *DATABASE_SPHOUT/ *DATABASE_HISTORY_SPH were not reviewed for output content.

