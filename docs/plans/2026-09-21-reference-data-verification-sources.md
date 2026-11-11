# Source dossier: published limits and the LS-DYNA realisation of E1–E10

**Date**: 2026-09-21
**Status**: Evidence for maintainer ratification — nothing here is a decision
**Serves**: ADR-0066 clause 7 (an indicator's reference levels cite a
verified external source, as the source states it, scoped by problem class
and formulation) and clause 2 (each solver's way of meeting E1–E10 is checked
against its manual before it is written down).

---

## How this was produced, and how far to trust it

Four research passes read primary sources; each was followed by an independent
pass that returned to every cited source and checked the claim at its locator.
**119 claims: 118 confirmed, 1 wording mismatch** (the title string of a web
page; the document title itself is confirmed).

This file is an index: each entry states a claim in the platform's words and
gives the source, the locator and the address where it can be checked. The
sources' own sentences are not reproduced here (removed 2026-11-11, before the
first archived release); read them at the locator before ratifying a level.

Limits of the reading, to weigh before ratifying anything:

- **NCHRP Web-Only Document 179** has no text layer; its entries were read
  from page images, twice. Check the page before ratifying a number from it.
- **Belytschko, Liu, Moran, Elkhodary (2nd ed., Wiley 2014)** was read only in
  part. Adequate for the claims indexed; not adequate for claims of absence.
- **Abaqus** entries come from the vendor's *Getting Started* guide (v6.6 and
  2016); the Analysis User's Guide was not read.
- **LS-DYNA**: Keyword Manual Vol I R13 and R15, Vol II R15, and Theory Manual
  R16 were read. Releases R11, R12 and R14 were **not**, and the legacy runs
  used R12: defaults can differ by release.
- **Not read**: ASME V&V 10 / 10.1 / 20, NAFEMS guidance, EN 16303 /
  CEN/TR 16303. Nothing is claimed about their content.

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

**How ADR-0066 uses this.** On the maintainer's direction the
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

- Source: *NCHRP Web-Only Document 179: Procedures for Verification and Validation of Computer Simulations Used for Roadside Safety Applications (Ray, Mongiardini, Plaxico, Anghileri)* — National Academies Press record 17647; 2010/2011 (year not read from a title page in this session)
- Locator: HTML page title of NAP reader; onlinepubs.trb.org/onlinepubs/nchrp/nchrp_w179.pdf redirects to this NAP record
- URL: <https://www.nationalacademies.org/read/17647/chapter/5>
- Kind / confidence: primary / high
- Corrected locator: HTML <title> and embedded metadata JSON of https://www.nationalacademies.org/read/17647/chapter/5; publication date 2011-10-05 per NAP metadata; TRB PDF served directly at https://onlinepubs.trb.org/onlinepubs/nchrp/nchrp_w179.pdf

### W-W179-02 — Table E-1 row 1: total energy

**Claim.** Total-energy criterion: total energy must not vary more than 10 percent from the beginning of the run to the end of the run. It is worded as beginning-versus-end, not explicitly as a maximum over the run. The energy types are listed as 'kinetic, potential, contact, etc.' and external work is not mentioned.

- Source: *NCHRP Web-Only Document 179, Appendix E Validation/Verification Report Forms* — 2010/2011
- Locator: Appendix E, Part II: Analysis Solution Verification, Table E-1 'Analysis Solution Verification Table', printed page E-3 (NAP page image 551), row 1
- URL: <https://nap.nationalacademies.org/books/17647/gif/551.gif>
- Kind / confidence: primary / high

### W-W179-03 — Table E-1 row 2: global hourglass vs initial energy

**Claim.** Whole-model hourglass energy at the end of the run must be less than 5 percent of the total initial energy at the beginning of the run.

- Source: *NCHRP Web-Only Document 179, Appendix E* — 2010/2011
- Locator: Table E-1, p. E-3, row 2
- URL: <https://nap.nationalacademies.org/books/17647/gif/551.gif>
- Kind / confidence: primary / high

### W-W179-04 — Table E-1 row 3: global hourglass vs internal energy

**Claim.** Whole-model hourglass energy at the end of the run must be less than 10 percent of the total internal energy at the end of the run.

- Source: *NCHRP Web-Only Document 179, Appendix E* — 2010/2011
- Locator: Table E-1, p. E-3, row 3
- URL: <https://nap.nationalacademies.org/books/17647/gif/551.gif>
- Kind / confidence: primary / high

### W-W179-05 — Table E-1 row 4: per-part hourglass

**Claim.** In Table E-1 the per-part hourglass criterion is: the part/material with the highest hourglass energy at the end of the run must have it below 10 percent of that part/material's own total internal energy at the end of the run.

- Source: *NCHRP Web-Only Document 179, Appendix E* — 2010/2011
- Locator: Table E-1, p. E-3, row 4
- URL: <https://nap.nationalacademies.org/books/17647/gif/551.gif>
- Kind / confidence: primary / high

### W-W179-06 — Table E-1 row 5: total added mass

**Claim.** Mass added to the total model must be less than 5 percent of the total model mass at the beginning of the run.

- Source: *NCHRP Web-Only Document 179, Appendix E* — 2010/2011
- Locator: Table E-1, p. E-3, row 5
- URL: <https://nap.nationalacademies.org/books/17647/gif/551.gif>
- Kind / confidence: primary / high

### W-W179-07 — Table E-1 row 6: per-part added mass

**Claim.** The part/material with the most added mass must have less than 10 percent of its own initial mass added.

- Source: *NCHRP Web-Only Document 179, Appendix E* — 2010/2011
- Locator: Table E-1, p. E-3, row 6
- URL: <https://nap.nationalacademies.org/books/17647/gif/551.gif>
- Kind / confidence: primary / high

### W-W179-08 — Table E-1 row 7: moving-part added mass

**Claim.** Moving parts/materials must have less than 5 percent of mass added relative to the initial moving mass of the model.

- Source: *NCHRP Web-Only Document 179, Appendix E* — 2010/2011
- Locator: Table E-1, p. E-3, row 7
- URL: <https://nap.nationalacademies.org/books/17647/gif/551.gif>
- Kind / confidence: primary / high

### W-W179-09 — Table E-1 rows 8-9: shooting nodes, negative volumes

**Claim.** The last two rows are yes/no checks: no shooting nodes, and no solid elements with negative volumes. Table E-1 has exactly nine rows.

- Source: *NCHRP Web-Only Document 179, Appendix E* — 2010/2011
- Locator: Table E-1, p. E-3, rows 8 and 9
- URL: <https://nap.nationalacademies.org/books/17647/gif/551.gif>
- Kind / confidence: primary / high

### W-W179-10 — Mandatory status and exceptions

**Claim.** All Table E-1 criteria must be satisfied, and a failed criterion means the solution cannot be used for verification/validation. Analyst exceptions may be footnoted and explained, and the form has a 'with/without exceptions as noted' checkbox.

- Source: *NCHRP Web-Only Document 179, Chapter 4 Procedures and Appendix E* — 2010/2011
- Locator: Chapter 4, printed p. 120 (image 122); Appendix E p. E-3 (image 551, https://nap.nationalacademies.org/books/17647/gif/551.gif)
- URL: <https://nap.nationalacademies.org/books/17647/gif/122.gif>
- Kind / confidence: primary / high

### W-W179-11 — Internal inconsistency: Table 22 vs Table E-1

**Claim.** The report's worked example (Table 22, test case 1) words the per-part hourglass row differently from Table E-1: highest per-part hourglass energy 'at any time during the run' must be below 5 percent of the total initial energy at the beginning of the run. The other eight rows match Table E-1. The example reports total energy change 1.3%, hourglass 0, added mass 0.

- Source: *NCHRP Web-Only Document 179, Chapter 6 Benchmark Cases* — 2010/2011
- Locator: Table 22 'Analysis solution verification table for test case 1', printed p. 162 (image 164), row 4
- URL: <https://nap.nationalacademies.org/books/17647/gif/164.gif>
- Kind / confidence: primary / high

### W-W179-12 — Total energy: evaluation window, denominator, external work

**Claim.** The narrative says the maximum change in energy over the run can be reported and frames the threshold as change as a percent of the initial energy and momentum, 'e.g., say 5 or 10 percent'. The worked example treats the model as a closed system with no external work, where total energy should equal the vehicle's initial kinetic energy, and names gravity as a minimal exception. No correction for external work is given.

- Source: *NCHRP Web-Only Document 179, Chapter 2 (verification discussion) and Chapter 6* — 2010/2011
- Locator: Section 'Energy Balance', printed p. 99 (image 101); Chapter 6 'Part II - Solution Verification', printed p. 160 (image 162, https://nap.nationalacademies.org/books/17647/gif/162.gif)
- URL: <https://nap.nationalacademies.org/books/17647/gif/101.gif>
- Kind / confidence: primary / high

### W-W179-13 — Provenance of the hourglass limit

**Claim.** The ratio of hourglass energy to internal energy of about 0.1 is attributed to reference 128, listed as 'Du Bois, Paul, NOTES, 1998' (unpublished course notes). The report also warns that a per-part ratio is not fool-proof, because a large part ID dilutes local hourglassing.

- Source: *NCHRP Web-Only Document 179, Chapter 2 and References* — 2010/2011
- Locator: Section 'Energy Balance', printed pp. 98-99 (images 100-101); reference 128 on printed p. 251 (image 253, https://nap.nationalacademies.org/books/17647/gif/253.gif)
- URL: <https://nap.nationalacademies.org/books/17647/gif/100.gif>
- Kind / confidence: primary / high

### W-W179-14 — Provenance: community practice and practitioner survey

**Claim.** The report presents its procedures as a formalisation of informal community practice and gives no derivation of the limits. A practitioner survey (Appendix D, Q27) records tolerance bands. For total energy variation 'during the run', 51.6% chose <5% and 41.9% chose 5-10%. For hourglass energy relative to total internal energy, 41.9% chose <5% and 41.9% chose 5-10%. For added mass relative to total physical mass, 64.5% chose <5%.

- Source: *NCHRP Web-Only Document 179, Chapter 4 Introduction; Chapter 3 Survey of Modeling Best Practices; Appendix D survey* — 2010/2011
- Locator: Printed p. 108 (image 110) 'Energy Balance and Comparisons'; printed p. 113 (image 115, https://nap.nationalacademies.org/books/17647/gif/115.gif); survey Q27 on survey page 27 (image 542, https://nap.nationalacademies.org/books/17647/gif/542.gif)
- URL: <https://nap.nationalacademies.org/books/17647/gif/110.gif>
- Kind / confidence: primary / medium

### W-W179-15 — Mass-scaling metric rationale

**Claim.** The report proposes three bases for limiting added mass: percentage of total model mass, percentage of moving mass, and percentage added to individual elements. It requires any exceedance to be reported and justified.

- Source: *NCHRP Web-Only Document 179, Chapter 2, 'Mass Scaling'* — 2010/2011
- Locator: Printed p. 100 (image 102)
- URL: <https://nap.nationalacademies.org/books/17647/gif/102.gif>
- Kind / confidence: primary / high

### W-FHWA-01 — FHWA adoption

**Claim.** FHWA's roadside-hardware FAQ adopts the W179 / 22-24 verification and validation report by reference for finite element analysis submitted for Federal-aid eligibility. The page states no numerical limits of its own.

- Source: *FHWA, 'FAQs: Barriers, Terminals, Transitions, Attenuators, and Bridge Railings'* — Wayback Machine capture 2025-01-10
- Locator: FAQ text; sentences containing 'Finite Element'
- URL: <https://web.archive.org/web/20250110153738id_/https://highways.dot.gov/safety/rwd/reduce-crash-severity/faqs-barriers-terminals-transitions-attenuators-and-bridge-railings>
- Kind / confidence: primary / medium
- Corrected locator: FAQ question 'MAY COMPUTER MODELING BE USED AS A SUBSTITUTE FOR FULL-SCALE CRASH TESTING OF NEW DEVICES?' (FAQs are unnumbered in the capture)

### W-RSVVP-01 — RSVVP documentation

**Claim.** The RSVVP User's Manual Rev. 1.4 (38 pages) contains no solution-verification criteria. A full-text search for 'hourglass', 'added mass', 'total energy' and 'shooting' returned no hits. RSVVP covers curve-comparison metrics only.

- Source: *Roadside Safety Verification and Validation Program (RSVVP) User's Manual* — Rev. 1.4, December 2008
- Locator: Title page; whole-document keyword search of the text layer
- URL: <https://roadsafellc.com/NCHRP22-24/QPR/AttachmentD-7.pdf>
- Kind / confidence: primary / medium

### W-R894-01 — Later NCHRP use of the table (variant)

**Claim.** NCHRP Report 894 Appendix C ('CCSA Validation/Verification Report') uses an 8-row 'Analysis Solution Verification Summary'. It has the same 10% total-energy row and the 5%-of-initial-energy global hourglass row. It omits the row for global hourglass below 10% of internal energy. For the per-part row it uses the Table-22 wording (any time during the run, below 5% of total initial energy). The added-mass rows (5% / 10% / 5%) and the shooting-node and negative-volume rows are unchanged.

- Source: *NCHRP Report 894, Appendix C: Finite Element Model Validations (CCSA Longitudinal Barriers on Curved, Superelevated Roadway Sections)* — PDF created 2017-02-28 (file metadata); report publication year not verified
- Locator: PDF page 4 (printed C-4), 'Table C – Analysis Solution Verification Summary'; repeated on p. C-17 and five other pages (7 instances)
- URL: <https://onlinepubs.trb.org/onlinepubs/nchrp/nchrp_rpt_894AppendixC.pdf>
- Kind / confidence: primary / high

### W-ENCAP-01 — Euro NCAP virtual testing quality criteria (frontal/crash protection)

**Claim.** Euro NCAP requires, for each simulation: maximum hourglass energy of the full setup below 10% of maximum internal energy; maximum hourglass energy of all dummy components below 10% of maximum internal energy for each dummy; and maximum mass added by mass scaling below 5% of total model mass at the beginning of the run (2.5% for separate models). There is no total-energy-change criterion. The hourglass/internal ratio at the time of maximum head excursion and the per-subsystem added mass are monitored with no limit defined.

- Source: *Euro NCAP Crash Protection Virtual Testing Protocol* — Version 1.0.2, December 2025 (implementation January 2026)
- Locator: Section 5 'Qualification of Simulation Model', 'Quality criteria of the simulation set-up', printed p. 13 (PDF p. 14)
- URL: <https://cdn.euroncap.com/cars/assets/euro_ncap_supporting_protocol_crash_protection_virtual_testing_v102_ca63155dc4.pdf>
- Kind / confidence: primary / high

### W-ENCAP-02 — Euro NCAP virtual far side protocol quality criteria

**Claim.** The 2023 far-side protocol sets the same limits: maximum hourglass energy of the full setup below 10% of maximum internal energy; WSID dummy components below 10% of the maximum internal energy of the WSID; added mass below 5% of total model mass at the beginning of the run. It also requires simulation time beyond the time of maximum head y-displacement plus 20%. All quality criteria must pass or the dossier is rejected.

- Source: *Euro NCAP Virtual Far Side Simulation & Assessment Protocol* — Version 1.0, 15 June 2023 (implementation 2024)
- Locator: Section 6.1.1-6.1.3, printed p. 12 (PDF p. 16)
- URL: <https://cdn.euroncap.com/cars/assets/euro_ncap_vtc_simulation_and_assessment_protocol_v10_f13aa41e72.pdf>
- Kind / confidence: primary / high

## B — Energy-balance, hourglass, and mass-scaling criteria in other authoritative sources

### B-BLM-1 — Belytschko energy balance inequality and tolerance

**Claim.** The book's energy balance check for explicit integration is |Wkin + Wint - Wext| <= eps * max(Wext, Wint, Wkin), numbered (6.2.18) in the 2nd edition, with eps a small tolerance 'generally on the order of 10^-2'. Normalisation is the maximum of external work, internal energy and kinetic energy. For very large systems (order 10^5 nodes or more) the balance should be done on subdomains.

- Source: *Belytschko, Liu, Moran, Elkhodary - Nonlinear Finite Elements for Continua and Structures, 2nd edition (Wiley)* — 2nd ed., 2014 (page proofs dated 10/5/2013 in snippet)
- Locator: p. 336, Section 6.2.3 'Energy Balance', eq. (6.2.18)
- URL: <https://books.google.com/books?id=e_w8AgAAQBAJ&q=%22energy+conservation+requires%22&jscmd=SearchWithinVolume2>
- Kind / confidence: primary / high

### B-BLM-2 — Belytschko energy balance - when evaluated and purpose

**Claim.** The energy balance is checked each time step within the explicit flowchart (Box 6.1), and its stated purpose is detecting instabilities (incl. 'arrested instabilities') that are not visible in results; instability shows up as spurious energy generation.

- Source: *Belytschko, Liu, Moran, Elkhodary - Nonlinear Finite Elements for Continua and Structures, 2nd edition (Wiley)* — 2nd ed., 2014
- Locator: p. 333 (Box 6.1 flowchart, step 11), pp. 335-336 (Section 6.2.3)
- URL: <https://books.google.com/books?id=e_w8AgAAQBAJ&q=check+on+energy&jscmd=SearchWithinVolume2>
- Kind / confidence: primary / high
- Corrected locator: p. 333 Box 6.1 step 11; p. 335 (end) to p. 336, Section 6.2.3

### B-BLM-3 — Belytschko hourglass energy fraction

**Claim.** The book says the ratio of hourglass energy to TOTAL energy should be monitored; 'large' means on the order of 3% or 5%, in which case results are in error by the same order. For large problems, monitor on subdomains.

- Source: *Belytschko, Liu, Moran, Elkhodary - Nonlinear Finite Elements for Continua and Structures, 2nd edition (Wiley)* — 2nd ed., 2014
- Locator: p. 523, Section 8.7 (hourglass/stabilization), near eq. (8.7.14)
- URL: <https://books.google.com/books?id=e_w8AgAAQBAJ&q=%22energy+to+the+total+energy+should+be+monitored%22&jscmd=SearchWithinVolume2>
- Kind / confidence: primary / high
- Corrected locator: p. 523 (Section 8.7 inferred from eq. number (8.7.10) cited in the adjacent sentence; '(8.7.14)' not verified)

### B-BLM-4 — Belytschko mass scaling

**Claim.** The book gives no numeric added-mass limit; it says mass scaling should be used where high-frequency effects are unimportant and is not recommended where high-frequency response matters.

- Source: *Belytschko, Liu, Moran, Elkhodary - Nonlinear Finite Elements for Continua and Structures, 2nd edition (Wiley)* — 2nd ed., 2014
- Locator: p. 337, Section 6.2.x 'Mass Scaling, Subcycling and Dynamic Relaxation'
- URL: <https://books.google.com/books?id=e_w8AgAAQBAJ&q=%22Mass+scaling+should+be+used+for+problems%22&jscmd=SearchWithinVolume2>
- Kind / confidence: primary / high
- Corrected locator: p. 337, Section 6.2.5 'Mass Scaling, Subcycling and Dynamic Relaxation'

### B-ABQ-1 — Abaqus/Explicit ETOTAL constancy

**Claim.** Abaqus Getting Started guide states the whole-model energy balance ETOTAL should be constant and in the numerical model is only approximately constant, 'generally with an error of less than 1%'. No denominator and no end-of-run vs over-run qualifier is given. Identical sentence in v6.6 (Explicit keywords guide 3.7.1) and in 2016 (Abaqus/CAE guide 9.6 and 13.4).

- Source: *Getting Started with ABAQUS/Explicit: Keywords Version (v6.6), Section 3.7.1 'Statement of energy balance'; same text in Getting Started with Abaqus/CAE (2016) Sections 9.6 and 13.4* — Abaqus 6.6 (2006) and Abaqus 2016
- Locator: Section 3.7.1 (v6.6); 2016 copy: https://ceae-server.colorado.edu/v2016/books/gsa/ch09s06.html Section 9.6
- URL: <https://web.archive.org/web/20170614214554id_/http://classes.engineering.wustl.edu/2009/spring/mase5513/abaqus/docs/v6.6/books/gsx/ch03s07.html>
- Kind / confidence: primary / high
- Corrected locator: v6.6 GSX Section 3.7.1; 2016 GSA Section 9.6.1 (within 9.6) and Section 13.4

### B-ABQ-2 — Abaqus artificial strain energy (ALLAE) vs internal energy (ALLIE)

**Claim.** In the pages read, Abaqus gives no universal numeric threshold. General text: large artificial strain energy means mesh changes are needed. Worked examples compare whole-model ALLAE to whole-model ALLIE: ~2% is judged 'not a problem' (blast-loaded plate); ~15% is 'a substantial fraction' whose reduction would improve the solution (circuit board drop).

- Source: *Getting Started with Abaqus/CAE (2016)* — Abaqus 2016
- Locator: Section 10.5 (blast loading on a stiffened plate); Section 12.11 (circuit board drop test) at .../gsa/ch12s11.html; Section 9.6 at .../gsa/ch09s06.html
- URL: <https://ceae-server.colorado.edu/v2016/books/gsa/ch10s05.html>
- Kind / confidence: primary / high

### B-ABQ-3 — Abaqus quasi-static KE/IE criterion (relevant to mass scaling acceptance)

**Claim.** For quasi-static explicit analyses Abaqus says the kinetic energy of the deforming material should not exceed typically 5% to 10% of its internal energy throughout most of the process; the forming example tightens this to 'a few percent'. Mass scaling acceptability is judged by the same approach as load-rate scaling; no percent-added-mass limit is given.

- Source: *Getting Started with Abaqus/CAE (2016)* — Abaqus 2016
- Locator: Section 13.4 'Energy balance'; Section 13.5 (.../ch13s05.html) 'Strategy for evaluating the results'; Section 13.3 (.../ch13s03.html) 'Mass scaling'
- URL: <https://ceae-server.colorado.edu/v2016/books/gsa/ch13s04.html>
- Kind / confidence: primary / high
- Corrected locator: 13.4 'Energy balance' (ch13s04.html); 13.5 'Example: forming a channel in Abaqus/Explicit' (ch13s05.html); 13.3 mass scaling (ch13s03.html)

### B-LSD-1 — LS-DYNA energy ratio definition

**Claim.** LS-DYNA support defines the glstat energy ratio as total energy / (initial energy + external work), whole model, equal to 1.0 for perfect balance. No tolerance band around 1.0 is stated on this page.

- Source: *LS-DYNA support knowledge base - 'Total energy' (Ansys/DYNAmore)* — web page, fetched 2026-09-21
- Locator: paragraph beginning 'The energy balance is perfect if'
- URL: <https://lsdyna.ansys.com/total-energy-2/>
- Kind / confidence: vendor-support / high

### B-LSD-2 — LS-DYNA hourglass energy rule of thumb

**Claim.** LS-DYNA support: hourglass energy should be <10% (rule of thumb) of PEAK internal energy, judged FOR EACH PART (matsum), with HGEN=2.

- Source: *LS-DYNA support knowledge base - 'Hourglass' (Ansys/DYNAmore)* — web page, fetched 2026-09-21
- Locator: Section 'Notice', paragraph beginning 'To evaluate hourglass energy'
- URL: <https://lsdyna.ansys.com/hourglass/>
- Kind / confidence: vendor-support / high

### B-LSD-3 — LS-DYNA contact (sliding) energy without friction

**Claim.** LS-DYNA support: absent friction, net contact energy of about 10% of peak internal energy 'might be considered acceptable'; 'small is a matter of judgement'.

- Source: *LS-DYNA support knowledge base - 'Total energy' (Ansys/DYNAmore)* — web page, fetched 2026-09-21
- Locator: Section 'Positive contact energy:'
- URL: <https://lsdyna.ansys.com/total-energy-2/>
- Kind / confidence: vendor-support / high

### B-LSD-4 — LS-DYNA energy equation and interpretation (User's Guide tutorial)

**Claim.** The LS-DYNA user's-guide tutorial says the energy equation should hold at all times during the analysis (i.e. over the run), and that deviation indicates an error; it gives no numeric tolerance. The energy ratio can be used as a termination criterion via ENDENG.

- Source: *LS-DYNA support - Tutorial / LS-DYNA User's Guide - 'Energy data' (Ansys/DYNAmore)* — web page, fetched 2026-09-21
- Locator: whole page
- URL: <https://lsdyna.ansys.com/energy-data-2/>
- Kind / confidence: vendor-support / medium

### B-LSD-5 — LS-DYNA Theory Manual - energy balance

**Claim.** The LS-DYNA Theory Manual R16 contains NO numerical energy-ratio/energy-balance tolerance for explicit analyses. A full-text search for 'energy ratio' returned no hits; 'energy balance' appears only in passing and in Section 38.4.2 (Implicit), which gives qualitative checks only.

- Source: *ANSYS LS-DYNA Theory Manual R16* — R16@e545952c7 (03/21/25)
- Locator: Section 38.4.2 'Energy balance', p. 38-24 (PDF page 796)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2025/04/LS-DYNA_Manual_Theory_R16.pdf>
- Kind / confidence: primary / high
- Corrected locator: Section 38.4.2 starts p. 38-23 (PDF p. 795); the three checks are on p. 38-24 (PDF p. 796)

### B-LSD-6 — LS-DYNA termination controls: ENDENG, ENDMAS, DTMIN (time-step collapse)

**Claim.** The keyword manual provides user-set terminations but recommends no values: ENDENG = percent change in energy ratio (inactive if undefined, default 0.0); ENDMAS = percent change in total mass (only with DT2MS mass scaling); DTMIN sets tsmin = dtstart x DTMIN and LS-DYNA terminates with a restart dump when the time step drops to tsmin (or erodes elements if ERODE=1).

- Source: *ANSYS LS-DYNA Keyword User's Manual Volume I, R16* — R16@431ab7b9b (10/29/25)
- Locator: *CONTROL_TERMINATION, p. 12-553 to 12-554 (PDF pages 1867-1868); Remark 2 'Erosion'
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Vol_I_R16.pdf>
- Kind / confidence: primary / high

### B-LSD-7 — LS-DYNA mass scaling acceptability

**Claim.** LS-DYNA gives no numeric added-mass limit. Keyword manual: negative DT2MS 'can be used in transient analyses if the mass increases remain insignificant'. Support page: justified e.g. for a few small elements in a noncritical area or quasi-static runs with KE very small relative to peak internal energy; otherwise analyst judgement and a sensitivity re-run. Added mass is reported whole-model (glstat) and per part (matsum).

- Source: *ANSYS LS-DYNA Keyword User's Manual Vol I R16 (*CONTROL_TIMESTEP, DT2MS) and LS-DYNA support knowledge base 'Mass scaling'* — R16 (10/29/25); web page fetched 2026-09-21
- Locator: Keyword manual: *CONTROL_TIMESTEP variable DT2MS, PDF page 1894 (https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Vol_I_R16.pdf); support page: first two paragraphs
- URL: <https://lsdyna.ansys.com/mass-scaling/>
- Kind / confidence: vendor-support / high

### B-RAD-1 — Radioss energy error definition

**Claim.** Radioss %Error = 100 * ( (Ek + Ekr + Ei) / (Ek,1 + Ekr,1 + Ei,1 + Ewk - Ewk,1) - 1 ), whole model, printed every output cycle, referenced to energies at the beginning of the RUN (not t=0), bounded to +-99%, reset at each restart. Hourglass energy (and contact energy) is NOT counted, so a negative error is normal with under-integrated elements.

- Source: *Altair Radioss Help - FAQ 'Results Checking'* — current online help (help.altair.com/hwsolvers), fetched 2026-09-21
- Locator: 'Energy Error calculation.'
- URL: <https://help.altair.com/hwsolvers/rad/topics/solvers/rad/faq_rad_results_checking_r.htm>
- Kind / confidence: vendor-support / medium

### B-RAD-2 — Radioss acceptable energy error magnitudes

**Claim.** Radioss guidance: negative error from hourglass 'normal' at about -10% to -15%; positive error of +1% or +2% acceptable; positive error >2% must have its source identified; error growing to +-99% can indicate divergence. Exceptions: low initial energy early in run, a single diverging part, large frictional contact energy.

- Source: *Altair Radioss Help - FAQ 'Results Checking'* — current online help, fetched 2026-09-21
- Locator: 'Recommendation of Energy Error.'
- URL: <https://help.altair.com/hwsolvers/rad/topics/solvers/rad/faq_rad_results_checking_r.htm>
- Kind / confidence: vendor-support / high

### B-RAD-3 — Radioss added mass limit

**Claim.** Radioss recommends keeping added mass below 5% in general (larger may be acceptable e.g. quasi-static), measured as mass error DM/M0 = (M - M0)/M0 for the whole model (MAS.ERR column) with M0 reset at each Engine file; local nodal added mass should also be checked (/ANIM/NODA/DMAS). Added mass that adds KE shows as positive energy error.

- Source: *Altair Radioss Help - 'Nodal Time Step Control'* — current online help, fetched 2026-09-21
- Locator: introductory paragraphs and 'Check for Mass Increase'
- URL: <https://help.altair.com/hwsolvers/rad/topics/solvers/rad/time_step_nodal_time_step_control_r.htm>
- Kind / confidence: vendor-support / high

## L — LS-DYNA: energy ledger and time-integration record

### L-C01 — 1 CONTROL_ENERGY card layout and defaults (R13)

**Claim.** In R13, *CONTROL_ENERGY Card 1 has six fields HGEN, RWEN, SLNTEN, RYLEN, IRGEN, MATEN (all integer) with table defaults 1, 2, 1, 1, 2, 1 respectively. DRLEN and DISEN are not present in R13.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_ENERGY, p. 12-103 (PDF page 1333)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C02 — 1 CONTROL_ENERGY HGEN

**Claim.** HGEN=1: hourglass energy not computed (default). HGEN=2: computed and included in the energy balance, reported in glstat and matsum. Costs about ten percent.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_ENERGY, p. 12-103 (PDF page 1333)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C03 — 1 CONTROL_ENERGY RWEN

**Claim.** RWEN=1: rigidwall (stonewall) energy dissipation not computed. RWEN=2 (default): computed and included in the energy balance, reported in glstat.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_ENERGY, p. 12-103 (PDF page 1333)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C04 — 1 CONTROL_ENERGY SLNTEN

**Claim.** SLNTEN table default is 1, but the manual states it is always set to 2 if contact is active and that SLNTEN=1 is not available; with 2 the sliding interface energy is included in the energy balance and reported in glstat and sleout.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_ENERGY, p. 12-103 (PDF page 1333)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C05 — 1 CONTROL_ENERGY RYLEN

**Claim.** RYLEN=1 (default): Rayleigh damping energy dissipation not computed. RYLEN=2: computed and included in the energy balance; reported in glstat and matsum.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_ENERGY, pp. 12-103 to 12-104 (PDF pages 1333-1334)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C06 — 1 CONTROL_ENERGY IRGEN and MATEN

**Claim.** IRGEN=2 (default) includes initial reference geometry energy in the energy balance as part of internal energy; IRGEN=1 does not compute it. MATEN=1 (default) no detailed material energies; MATEN=2 splits internal energy into elastic, plastic, damage portions for a listed set of materials, reported in glstat and matsum.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_ENERGY, p. 12-104 (PDF page 1334)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C07 — 1 CONTROL_ENERGY DRLEN and DISEN (R15 only)

**Claim.** R15 adds fields 7 and 8, DRLEN and DISEN, both default 1. DRLEN=2 computes drilling energy (implicit / DRCPSID-DRCPRM) into the energy balance, reported in glstat. DISEN=2 computes dissipated kinetic and internal energy for implicit into the energy balance, reported in glstat. Both are described as implicit-related.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (R15@7355dd456, 02/28/24)
- Locator: *CONTROL_ENERGY, pp. 12-106 to 12-107 (PDF pages 1384-1385); card table defaults row '1 2 1 1 2 1 1 1'
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high

### L-C08 — 2 glstat contents

**Claim.** The manual's GLSTAT output-component table lists: time step, kinetic energy, internal energy, spring and damper energy (printed 'sprint'), hourglass energy, system damping energy, sliding interface energy, eroded kinetic energy, eroded internal energy, eroded hourglass energy, added mass, total energy, external work, total and initial energy, energy ratio without eroded energy, element & part ID controlling time step, global x,y,z velocity, time per zone cycle, joint internal energy, stonewall energy, rigid body stopper energy, percentage [mass] increase.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *DATABASE_OPTION, 'Output Components for ASCII Files', GLSTAT table, p. 16-11 (PDF page 1833)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C09 — 2 glstat contents are deck-dependent

**Claim.** Items in glstat appear only when relevant: hourglass energy only if HGEN=2; added mass only if DT2MS<0.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *DATABASE_OPTION, Remark 10, p. 16-15 (PDF page 1837)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C10 — 2 definition of total energy (manual)

**Claim.** The only explicit total-energy formula found in Volume I is in a release-notes bullet: total energy = kinetic + internal + hourglass + rigidwall energy (stated in the context of including eroded hourglass energy in glstat hourglass energy). The *DATABASE section itself gives no formula.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: INTRODUCTION, release-notes bullet list (PDF page 110)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / medium
- Corrected locator: INTRODUCTION release notes, PDF p.110 (R7.1 section) and PDF p.118 (R8.0 section)

### L-C11 — 2 definition of total energy (support site)

**Claim.** Per the vendor support site, glstat total energy is the sum of six terms: internal, kinetic, contact (sliding), hourglass, system damping, and rigidwall energy. So hourglass, sliding-interface, rigidwall and system-damping energies are INSIDE total energy and are separate from internal energy.

- Source: *Total energy - LS-DYNA support site (dynasupport.com howto)* — Wayback capture 2024-10-05 of live page (page text references v.970/971-era revisions)
- Locator: First paragraph and bullet list
- URL: <https://web.archive.org/web/20241005194851id_/https://www.dynasupport.com/howtos/general/total-energy>
- Kind / confidence: vendor-support / high

### L-C12 — 2 energy ratio definition

**Claim.** The glstat 'total energy / initial energy' ratio is actually total energy / (initial energy + external work); external work IS in the denominator. Perfect balance means ratio = 1.0.

- Source: *Total energy - LS-DYNA support site (dynasupport.com howto)* — Wayback capture 2024-10-05
- Locator: Paragraph beginning 'The energy balance is perfect if'
- URL: <https://web.archive.org/web/20241005194851id_/https://www.dynasupport.com/howtos/general/total-energy>
- Kind / confidence: vendor-support / high

### L-C13 — 2 eroded energies

**Claim.** glstat energies include eroded-element contributions; eroded energy = internal energy of deleted elements and kinetic energy of deleted nodes. The 'energy ratio w/o eroded energy' is 1 with no deletions and <1 if elements have been deleted; deletions should not affect the total energy/initial energy ratio.

- Source: *Total energy - LS-DYNA support site (dynasupport.com howto)* — Wayback capture 2024-10-05
- Locator: Paragraphs beginning 'The History > Global energies' and 'Eroded energy is'
- URL: <https://web.archive.org/web/20241005194851id_/https://www.dynasupport.com/howtos/general/total-energy>
- Kind / confidence: vendor-support / high

### L-C14 — 2 damping energy placement

**Claim.** Stiffness damping energy is lumped into internal energy; mass damping energy is the separate glstat item 'system damping energy'. Hourglass, rigidwall and system damping energies are computed/written only when HGEN, RWEN, RYLEN = 2.

- Source: *Total energy - LS-DYNA support site (dynasupport.com howto)* — Wayback capture 2024-10-05
- Locator: Paragraphs beginning 'Hourglass energy is computed' and 'Stiffness damping energy'
- URL: <https://web.archive.org/web/20241005194851id_/https://www.dynasupport.com/howtos/general/total-energy>
- Kind / confidence: vendor-support / high

### L-C15 — 2 Rayleigh stiffness damping energy in internal energy (manual)

**Claim.** The manual confirms energy dissipated by Rayleigh (stiffness) damping is computed only if RYLEN=2, is accumulated as element internal energy, and is lumped in with internal energy in glstat.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *DAMPING section, stiffness-damping keyword remarks (PDF page 1819)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high
- Corrected locator: *DAMPING_PART_STIFFNESS, Remarks, p. 15-11 (PDF page 1819)

### L-C16 — 2 spring and damper energy

**Claim.** 'Spring and damper energy' in glstat is a subset of 'internal energy' (discrete elements, seatbelts, joint stiffness).

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *DATABASE_OPTION, Remark 6, p. 16-14 (PDF page 1836)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C17 — 2 bulk viscosity energy

**Claim.** With the default bulk viscosity TYPE=1 (solids only) the internal energy from bulk viscosity is always computed and included in the overall energy balance; same for TYPE=2 (Richards-Wilkins, 2D plane strain and axisymmetric solids only). For shells it is included only with TYPE=-2 (TYPE=-1 applies viscosity to shells without computing the energy). Defaults: Q1=1.5, Q2=.06, TYPE=1.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_BULK_VISCOSITY, p. 12-60 (PDF page 1290)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C18 — 2 internal energy definition

**Claim.** Internal energy is accumulated incrementally per element as stress times incremental strain times volume over six components, summed over elements; stiffness damping dissipation is added if RYLEN=2.

- Source: *Internal Energy - LS-DYNA support site (dynasupport.com howto)* — Wayback capture 2025-05-29
- Locator: Opening paragraphs
- URL: <https://web.archive.org/web/20250529110029id_/https://www.dynasupport.com/howtos/general/internal-energy>
- Kind / confidence: vendor-support / high

### L-C19 — 2 external work content

**Claim.** External work includes work by applied forces and pressures and by velocity, displacement or acceleration boundary conditions; internal energy includes elastic strain energy and work done in permanent deformation. Hourglass energy is excluded by default; Rayleigh damping dissipation (RYLEN=2) is added to internal energy. The energy ratio can be a termination criterion via ENDENG.

- Source: *Energy data - LS-DYNA support site (LS-DYNA User's Guide tutorial)* — Wayback capture 2024-10-11
- Locator: Paragraph after 'Where'
- URL: <https://web.archive.org/web/20241011173004id_/https://www.dynasupport.com/tutorial/ls-dyna-users-guide/energy-data>
- Kind / confidence: vendor-support / high

### L-C20 — 3 what each ASCII database provides

**Claim.** Manual one-line descriptions: BNDOUT 'Boundary condition forces and energy'; GLSTAT 'Global statistics and energies (recommended)'; MATSUM 'Part energies'; NODFOR 'Nodal force groups'; RCFORC 'Resultant contact interface forces'; RWFORC 'Rigidwall forces'; SLEOUT 'Contact interface energies'; SPCFORC 'SPC reaction forces'.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *DATABASE_OPTION, OPTION1 list, pp. 16-3 to 16-4 (PDF pages 1825-1826)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C21 — 3 output components per file

**Claim.** Output components per the manual tables: MATSUM = kinetic energy, internal energy, hourglass energy, x,y,z momentum, x,y,z rigid body velocity, eroded internal energy, eroded kinetic energy, added mass. SLEOUT = slave energy, master energy, frictional energy. RWFORC = normal, x,y,z force. RCFORC = x,y,z force, mass of nodes in contact. SPCFORC = x,y,z force, x,y,z moment. BNDOUT = x,y,z force, x,y,z moment (listed), energies. NODFOR = x,y,z force.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *DATABASE_OPTION, 'Output Components for ASCII Files' tables, pp. 16-10 to 16-13 (PDF pages 1832-1835)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / medium

### L-C22 — 3 matsum vs glstat kinetic energy discrepancy

**Claim.** matsum and glstat kinetic energies can differ: added-mass energy is included in glstat but not matsum; matsum is element-by-element with midpoint velocities, glstat uses nodal velocities.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *DATABASE_OPTION, Remark 1, p. 16-13 (PDF page 1835)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C23 — 3 matsum eroded energies (IERODE)

**Claim.** Eroded internal and kinetic energy per part are written to matsum only if IERODE=1 on *CONTROL_OUTPUT (default 0 = no extra data); also adds part ID 0 (KE of nonstructural/lumped mass and inertia) and part ID -1 (*ELEMENT_MASS_PART).

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_OUTPUT, IERODE (PDF page 1648)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C24 — 3 rcforc averaging

**Claim.** rcforc resultant contact forces are averaged over the preceding output interval (not instantaneous).

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *DATABASE_OPTION, Remark 5, p. 16-14 (PDF page 1836)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C25 — 3 bndout categories

**Claim.** bndout OPTION1-4 control inclusion of nodal force group, concentrated force, pressure BC, and velocity/displacement/acceleration nodal BC output; 0/blank = included (default), 1 = excluded.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *DATABASE_OPTION, Remark 9, pp. 16-14 to 16-15 (PDF pages 1836-1837)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C26 — 3 nodfor

**Claim.** nodfor requires *DATABASE_NODAL_FORCE_GROUP (NSID, optional CID) plus *DATABASE_NODFOR for the interval; it writes group resultant reaction forces, the external work done by them, and per-node forces.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *DATABASE_NODAL_FORCE_GROUP, Remark 1, p. 16-111 (PDF page 1933)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C27 — 3/4 DT field and file creation

**Claim.** No ASCII database is created unless its *DATABASE_OPTION1 card is present; DT is the output interval, default 0., and DT=0 means no output.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *DATABASE_OPTION, pp. 16-3 and 16-5 (PDF pages 1825, 1827); Card 1 defaults row: DT 0., BINARY '1 or 2', LCUR none, IOOPT 0.
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C28 — 4 BINARY field values and defaults

**Claim.** BINARY: 1 = ASCII file written (SMP default); 2 = data written to binout, ASCII not created (MPP default); 3 = both ASCII and binout, but MPP executables only produce the binary database. Card default is listed as '1 or 2'.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *DATABASE_OPTION, BINARY, pp. 16-5 to 16-6 (PDF pages 1827-1828)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C29 — 4 MPP ASCII output

**Claim.** Per Appendix O, MPP writes these databases only in binary (binout) regardless of BINARY; ASCII files are obtained after the run by the l2a converter ('l2a binout*'). The manual documents no input setting that forces direct ASCII output from MPP. Files may be one per processor, named binoutnnnn.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: Appendix O 'LS-DYNA MPP User Guide', section 'Output Files and Post-Processing', p. 63-2 (PDF page 3762); supported-file list continues p. 63-3 and includes GLSTAT, MATSUM, RCFORC, SPCFORC, RWFORC, BNDOUT, SLEOUT, SPHOUT, 'NODOFR' [sic]
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C30 — 5 DT2MS semantics

**Claim.** DT2MS<0: TSSFAC x |DT2MS| is the minimum permitted time step and mass is added only where needed to meet the Courant criterion; DT2MS>0 is for quasi-static / inertia-insignificant analyses. Default 0.0 (no mass scaling).

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_TIMESTEP, DT2MS, p. 12-551 (PDF page 1781)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C31 — 5 TSSFAC semantics and default

**Claim.** Time step = TSSFAC x min over elements of element time steps. Default TSSFAC: 0.90 normally; 0.667 with *INITIAL_DETONATION or *BOUNDARY_NON_REFLECTING; with contacts and SLSFAC specified, 0.333 for SLSFAC>=9 and 0.667 for 1<=SLSFAC<9. Values above 0.90 often unstable.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_TIMESTEP, Remark 1, pp. 12-553 to 12-554 (PDF pages 1783-1784)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C32 — 5 MS1ST and monotonic added mass

**Claim.** With DT2MS<0 and MS1ST=0 (default) mass scaling is applied throughout; added mass may increase with time but never decrease. MS1ST=1 computes added mass once at the first step.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_TIMESTEP, MS1ST, p. 12-552 (PDF page 1782)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C33 — 5 where added mass is reported

**Claim.** Added mass vs time is available for the whole model in glstat and per part in matsum; per-element added mass for shells can be written to d3plot with STSSZ=3. d3hsp also reports 'added mass' and 'percentage increase'.

- Source: *Mass scaling - LS-DYNA support site (dynasupport.com howto)* — Wayback capture 2024-12-03
- Locator: Paragraph beginning 'To determine where and when mass is automatically added'
- URL: <https://web.archive.org/web/20241203084725id_/https://www.dynasupport.com/howtos/general/mass-scaling>
- Kind / confidence: vendor-support / high

### L-C34 — 6 element/part controlling the time step

**Claim.** glstat reports 'time step' and 'element & part ID controlling time step'; the controlling element ID is in the glstat data but is not read by LS-PrePost, so the ASCII glstat must be opened in a text editor.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *DATABASE_OPTION, Remark 11, p. 16-15 (PDF page 1837); GLSTAT table p. 16-11
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C35 — 6 initial element time steps in d3hsp

**Claim.** *CONTROL_OUTPUT IPNINT controls printing of initial element time step sizes to d3hsp: 0 = the 100 smallest, 1 = all elements, >1 = IPNINT smallest.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_OUTPUT, IPNINT, pp. 12-416 to 12-417 (PDF pages 1646-1647)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C36 — 7 CONTROL_TERMINATION fields

**Claim.** Fields and defaults: ENDTIM 0.0 (mandatory termination time); ENDCYC 0 (termination cycle, used if reached before ENDTIM; cycle = time step number); DTMIN 0.0 (tsmin = dtstart x DTMIN; terminates with a restart dump when step drops to tsmin, unless ERODE); ENDENG 0.0 (percent change in energy ratio for termination; inactive if undefined); ENDMAS 1.0E+08 (percent change in total mass for termination; relevant only with DT2MS mass scaling); NOSOL 0.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_TERMINATION, pp. 12-524 to 12-525 (PDF pages 1754-1755); defaults row '0.0 0 0.0 0.0 1.0E+08 0'
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C37 — 7 error-termination reporting (MPP)

**Claim.** For MPP, on error termination each processor writes a 'last known location' line to its own message file beginning 'When error termination was triggered, this processor was'.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: INTRODUCTION, release-notes bullets (text line ~5550 of extraction; PDF page ~128)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / medium
- Corrected locator: INTRODUCTION, R8.0 release-notes bullets, PDF page 131

### L-C38 — 8 CONTROL_SPH IDIM

**Claim.** IDIM: 3 = 3D; 2 = 2D plane strain; -2 = 2D axisymmetric. Card default listed as 'none'.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_SPH, IDIM, p. 12-502 (PDF page 1732)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C39 — 8 CONTROL_SPH FORM=12

**Claim.** FORM=12 is the moving least-squares based formulation, available for MPP only, for improved accuracy and tensile stability in extremely large deformation at significant cost; constant smoothing length (HMIN=HMAX=1.0) strongly recommended. ISTAB and QL (default 0.01) apply only to it. FORM default is 0.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_SPH, FORM list p. 12-502 and Remark 2(e) p. 12-506 (PDF pages 1732, 1736)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C40 — 8 SPH formulations valid in axisymmetry

**Claim.** Only FORM 0, 1, 15 and 16 are implemented for 2D axisymmetric SPH (IDIM=-2); hence FORM=12 is not available with IDIM=-2 per the manual. No analogous restriction is stated for plane strain (IDIM=2).

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_SPH, Remark 2(f), p. 12-506 (PDF page 1736)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C41 — 8 *HOURGLASS IBQ/Q1/Q2

**Claim.** On *HOURGLASS, IBQ is 'Not used' (bulk viscosity always on for solids; beams/shells only via *CONTROL_BULK_VISCOSITY TYPE), but Q1 (quadratic, default 1.5) and Q2 (linear, default 0.06) set the coefficients per part and, when the *HOURGLASS ID is referenced by a part, override *CONTROL_BULK_VISCOSITY.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *HOURGLASS, pp. 24-1 to 24-2 (PDF pages 2627-2628); defaults row: IBQ 'not used', Q1 1.5, Q2 0.06
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C42 — 8 bulk viscosity coefficients for SPH

**Claim.** For SPH, artificial viscosity type is chosen by IAVIS on *CONTROL_SPH (0 = Monaghan, default; 1 = standard solid-element form, not supported in 2D/2D-axisymmetric SPH). The Q1 and Q2 constants used by SPH come from *CONTROL_BULK_VISCOSITY or *HOURGLASS, and the manual recommends Q1=Q2=1.0 with Monaghan viscosity. So the Q1/Q2 fields of *HOURGLASS do apply to SPH parts; IBQ does not (it is unused).

- Source: *LS-DYNA Keyword User's Manual Volume I* — R13 (R13@bbf149e1c, 12/16/25)
- Locator: *CONTROL_SPH, Remark 4 'Artificial Viscosity', pp. 12-507 to 12-508 (PDF pages 1737-1738); IAVIS field p. 12-504: 'EQ.0: Monaghan type artificial viscosity formulation is used. EQ.1: Standard type artificial viscosity formulation from solid element is used (this option is not supported in SPH 2D and 2D axisymmetric elements).'
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R13.pdf>
- Kind / confidence: primary / high

### L-C43 — hourglass energy acceptance heuristic (context)

**Claim.** Support-site rule of thumb: hourglass energy should be small relative to PEAK internal energy FOR EACH PART, <10%, evaluated with HGEN=2 via glstat (system) and matsum (per part).

- Source: *Hourglass - LS-DYNA support site (dynasupport.com howto)* — Wayback capture 2025-06-16
- Locator: Paragraph beginning 'To evaluate hourglass energy'
- URL: <https://web.archive.org/web/20250616182522id_/https://www.dynasupport.com/howtos/element/hourglass>
- Kind / confidence: vendor-support / high

### L-C44 — contact energy acceptance heuristic (context)

**Claim.** Support-site judgement: without friction, net contact energy (slave + master) should be small; 10% of peak internal energy 'might be considered acceptable'. sleout reports contact energies per contact.

- Source: *Total energy - LS-DYNA support site (dynasupport.com howto)* — Wayback capture 2024-10-05
- Locator: Sections 'Positive contact energy' and 'Negative contact energy'
- URL: <https://web.archive.org/web/20241005194851id_/https://www.dynasupport.com/howtos/general/total-energy>
- Kind / confidence: vendor-support / high

## F — LS-DYNA: field output, input echo, and diagnostics

### F-C01 — EXTENT_BINARY card 1b defaults

**Claim.** In R15, Card 1b of *DATABASE_EXTENT_BINARY carries NEIPH, NEIPS, MAXINT, STRFLG, SIGFLG, EPSFLG, RLTFLG, ENGFLG with defaults 0, 0, 3, 0, 1, 1, 1, 1 respectively.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (R15@7355dd456, 02/28/24)
- Locator: *DATABASE_EXTENT_BINARY, Card 1b, p. 16-60 (PDF p. 1954)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high

### F-C02 — NEIPH (solids and SPH)

**Claim.** NEIPH (default 0) is the number of extra integration-point history variables written to d3plot/d3part/d3drlf for solid elements AND SPH particles, in memory storage order.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *DATABASE_EXTENT_BINARY, NEIPH, p. 16-60 (PDF p. 1954)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high

### F-C03 — NEIPS

**Claim.** NEIPS (default 0) is the number of extra history variables written per integration point for shells and thick shells.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *DATABASE_EXTENT_BINARY, NEIPS, p. 16-60 (PDF p. 1954)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high

### F-C04 — MAXINT (shell through-thickness output)

**Claim.** MAXINT (default 3) is the number of shell/tshell through-thickness integration points written to d3plot; it does not govern STRFLG strain output. With MAXINT=3 and more than 3 points, top, bottom and neutral-axis results are written. Negative MAXINT writes |MAXINT| points at every in-plane integration point with no averaging.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *DATABASE_EXTENT_BINARY, MAXINT table, p. 16-61 (PDF p. 1955)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high

### F-C05 — Shell output is in-plane averaged at element centre by default

**Claim.** For shells, with MAXINT=3 mid/inner/outer surface stresses are output at the element centre; in-plane integration points are averaged; for an even number of points the mid-surface value is the average of the two closest points. For MAXINT not equal to 3, stresses are output in stored order, after in-plane averaging.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *DATABASE_EXTENT_BINARY, Remark 1 (MAXINT Field), p. 16-68 (PDF p. 1962)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high

### F-C06 — STRFLG digits

**Claim.** STRFLG (default 0) is read digit-wise as STRFLG = L + M*10 + N*100: L=1 writes strain tensors to d3plot, elout and dynain (shells: innermost and outermost point; solids: a single tensor); M=1 writes plastic strain tensor data to d3plot; N=1 writes thermal strain tensors. STRFLG=11 gives strain + plastic strain tensors.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *DATABASE_EXTENT_BINARY, STRFLG, pp. 16-61 to 16-62 (PDF pp. 1955-1956)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high

### F-C07 — Plastic/thermal strain tensor: averaging and support

**Claim.** The plastic and thermal strain tensors written under STRFLG are, for solids, the element-average strain (6 components, global system unless CMPFLG set); for shells the plastic strain is 3 plane-averaged tensors (bottom, middle, top). Support is limited to listed element/material combinations (plastic strain tensors: shells 2, 16, 23; solids 1, 2; materials 24, 255).

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *DATABASE_EXTENT_BINARY, Remark 11, p. 16-70 (PDF p. 1964)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high

### F-C08 — Strain tensor definition

**Claim.** The d3plot strain tensor is a time-integrated rate-of-deformation (Jaumann rate for solids, co-rotational rate for shells), and the manual warns it is not invariant to element formulation.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *DATABASE_EXTENT_BINARY, Remark 10, p. 16-70 (PDF p. 1964)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high

### F-C09 — SIGFLG / EPSFLG / RLTFLG / ENGFLG

**Claim.** SIGFLG (default 1 = include stress tensor; 2 = exclude for shells; 3 = exclude for shells and solids). EPSFLG same pattern for effective plastic strain. RLTFLG (1 include shell stress resultants, 2 exclude). ENGFLG (1 include shell/tshell/beam internal energy density and shell thickness, 2 exclude). All defaults are 'include'.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *DATABASE_EXTENT_BINARY, p. 16-62 (PDF p. 1956)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high

### F-C10 — NINTSLD: solid stress is element average by default

**Claim.** NINTSLD default is 1. For any value other than 8, solid integration-point values are averaged and only the averages are written; NINTSLD=8 is required to get individual integration-point values, even for solids with fewer than 8 points.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *DATABASE_EXTENT_BINARY, Card 3, NINTSLD, p. 16-64 (PDF p. 1958)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high

### F-C11 — IEVERP / DCOMP / N3THDT

**Claim.** Card 2 defaults: CMPFLG 0, IEVERP 0, BEAMIP 0, DCOMP 1, SHGE 1, STSSZ 1, N3THDT 2, IALEMAT 1. IEVERP=1 writes one state per d3plot file. DCOMP=1 (default) no rigid-body data compression; values 3, 4 and 6 eliminate ALL nodal velocities and accelerations from the database. N3THDT=2 (default) writes material energy to d3thdt; 1 turns it off.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *DATABASE_EXTENT_BINARY, Card 2, pp. 16-62 to 16-64 (PDF pp. 1956-1958)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high

### F-C12 — DELERES and HYDRO

**Claim.** DELERES default 0 means results of deleted elements are written as all zero; DELERES=1 writes the last available results. HYDRO (default 0) adds 3, 5 or 7 shock-physics history variables as the LAST history variables in d3plot (HYDRO=1: internal energy per reference volume, reference volume, bulk-viscosity pressure; =2 adds relative volume and current density; =4 adds volumetric strain and hourglass energy per unit initial volume).

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *DATABASE_EXTENT_BINARY, Card 4 DELERES p. 16-68 (PDF p. 1962); HYDRO p. 16-65 (PDF p. 1959)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high

### F-C13 — d3plot precision

**Claim.** Per the R15 manual, a double-precision executable writes binary output (d3plot, d3drlf) in 64-bit format BY DEFAULT (IBINARY=0); 32-bit IEEE output is the non-default option IBINARY=1 on *DATABASE_FORMAT, or environment variable LSTC_BINARY=32ieee. IBINARY applies only to double-precision executables.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *DATABASE_FORMAT, IBINARY and Remark 2, pp. 16-84 to 16-85 (PDF pp. 1978-1979)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high

### F-C14 — d3hsp definition

**Claim.** d3hsp is the default name of the 'high speed printer file' (command-line O=). The input data is printed into it during the second input phase, where most input checking happens; the manual tells users to check d3hsp or messag for the word 'Error'.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: Getting Started, p. 2-3 (PDF p. 365); execution syntax 'otf = High speed printer file (default = d3hsp)' p. 2-12 (PDF p. 374)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high

### F-C15 — d3hsp verbosity: NPOPT, NEECHO, echo file

**Claim.** *CONTROL_OUTPUT NPOPT (default 0 = no suppression) controls d3hsp printing; NPOPT=1 suppresses nodal coordinates, element connectivities, rigid walls, nodal SPCs, initial velocities, initial strains, etc. NEECHO (default 0 = all data printed) controls a SEPARATE echo file (command line E=efl), not d3hsp.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *CONTROL_OUTPUT, Card 1, pp. 12-425 to 12-426 (PDF pp. 1703-1704); echo file: Getting Started p. 2-13 (PDF p. 375) 'efl = Echo file containing optional input echo with or without node/element data'
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high

### F-C16 — Other d3hsp content controls

**Claim.** IPNINT controls how many element initial time steps are printed to d3hsp (default: 100 smallest; 1 = all). IKEDIT (default 100) is the status-report interval to d3hsp, ignored if glstat is written. HISNOUT>=1 writes history-variable names per part to d3hsp (2/3 also write hisnames.xml / d3labels.xml). IPCURV=1 writes digitized curve data to messag and d3hsp.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *CONTROL_OUTPUT, IPNINT/IKEDIT p. 12-427 (PDF p. 1705); IPCURV p. 12-428; HISNOUT pp. 12-433 to 12-434 (PDF pp. 1711-1712)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high

### F-C17 — *PARAMETER echo to d3hsp

**Claim.** Defined parameters are echoed to d3hsp by default; the _NOECHO keyword option suppresses that echo.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *PARAMETER, Remark 7, p. 35-4 (PDF p. 3396)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high

### F-C18 — *PARAMETER reference syntax

**Claim.** A parameter is referenced anywhere in the input by '&' immediately before its name; '-&' flips the sign; a character parameter embedded in a larger string starts with '&' and ends with '^'. Names are up to 9 characters (letters, numbers, '_', not starting with a number), with a type prefix R/I/C in the definition.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *PARAMETER, Remarks 1-2, p. 35-3 (PDF p. 3395)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high

### F-C19 — *PARAMETER scoping across *INCLUDE

**Claim.** A non-LOCAL parameter is visible at any later point in input processing including include files; *PARAMETER_LOCAL parameters disappear when the parser finishes the file in which they appear and can mask non-LOCAL ones; include files can redefine values (manual example: VAL1 redefined to 10.0 in file1 remains 10.0 after returning to main.k).

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *PARAMETER, Remark 5, p. 35-4 (PDF p. 3396)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high

### F-C20 — *INCLUDE

**Claim.** *INCLUDE includes independent input files; options are <BLANK>, BINARY, NASTRAN, TRANSFORM, TRANSFORM_BINARY; TRANSFORM variants offset IDs and transform/scale coordinates and constitutive parameters, so the resolved model can differ from the literal include-file content.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *INCLUDE_{OPTION}, p. 26-3 (PDF p. 2837)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high

### F-C21 — Solver-written resolved deck (dyna.str)

**Claim.** *CONTROL_STRUCTURED (or 'outdeck = s' on the execution line) makes the solver write a STRUCTURED-format (not keyword-format) input deck named dyna.str that is 'largely or wholly equivalent' to the keyword deck; not all features are supported and some IDs (e.g. load curve numbers) are in internal numbering. _TERM terminates after writing it.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *CONTROL_STRUCTURED, p. 12-539 (PDF p. 1817)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high

### F-C22 — Message file names SMP vs MPP

**Claim.** The message file is 'messag' in SMP; in MPP there is one per processor, named mesnnnn (written 'mesxxxx' and 'mes****' elsewhere in the same manual), normally in the pfile's local directory unless the pfile option global_message_files is set.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: Appendix O (MPP), p. 64-3 (PDF p. 3969); also Appendix P p. 65-5 (PDF p. 3987): 'The message files, messag in SMP and mesxxxx in MPP'
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high
- Corrected locator: Add: global_message_files at Appendix O, PDF p. 3971 (approx. p. 64-5)

### F-C23 — MSGMAX: warning/error message truncation

**Claim.** *CONTROL_OUTPUT MSGMAX (default 50) caps the number of each error/warning message: if >0 the cap applies to screen output only and ALL messages go to d3hsp/messag; if <=0 the cap also applies to d3hsp/messag. So with default settings every occurrence is in d3hsp/messag.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *CONTROL_OUTPUT, Card 2, MSGMAX, p. 12-428 (PDF p. 1706)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high

### F-C24 — MSGFLG / d3msg

**Claim.** Standard-length error/warning messages are written to messag or mes****; MSGFLG=1 additionally writes detailed messages to a file d3msg at the end of the run, each only once.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *CONTROL_OUTPUT, MSGFLG, p. 12-430 (PDF p. 1708)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high

### F-C25 — NaN checking (ISNAN)

**Claim.** *CONTROL_SOLUTION ISNAN=1 activates a NaN check on the assembled force and moment arrays (about 2% cost); default 0 = no checking.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *CONTROL_SOLUTION, ISNAN, p. 12-516 (PDF p. 1794)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high

### F-C26 — NaN / out-of-range message text

**Claim.** Per the LS-DYNA support site FAQ, with ISNAN=1 (or command-line checknan=1) the screen shows '*** NaN detected. Please check message file from processor for detail.' and the per-processor message file contains '*** termination due to out-of-range forces' followed by the number of nodes and a node list.

- Source: *LS-DYNA runs into Segmentation Violation (SIGSEGV) error. What can I do? (LS-DYNA support site FAQ, dated 'gp 04/14')* — 2014
- Locator: Section 'Checknan option' / 'Which output is produced by checknan option?'
- URL: <https://www.dynasupport.com/faq/general/SIGSEGV>
- Kind / confidence: vendor-support / medium

### F-C27 — Element failure/deletion messages

**Claim.** Beam, shell and solid element failure messages are written to d3hsp and the message files by default; *CONTROL_MPP_IO_NOBEAMOUT (pfile: nobeamout) suppresses them.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *CONTROL_MPP_IO_NOBEAMOUT, p. 12-414 (PDF p. 1692)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high

### F-C28 — Initial penetrations: IGNORE

**Claim.** *CONTROL_CONTACT IGNORE (default 0) for *CONTACT_AUTOMATIC options: 0 = move nodes to eliminate initial penetrations; 1 = allow and track them; 2 = allow and track, and print penetration warning messages with original and recommended coordinates of each penetrating node. 'Initial' means the first time step a penetration is encountered. Not implemented in SMP for AUTOMATIC_GENERAL. Can be overridden per contact on *CONTACT optional card C.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: *CONTROL_CONTACT, Card 4, IGNORE, pp. 12-72 to 12-73 (PDF pp. 1351-1352)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high
- Corrected locator: *CONTROL_CONTACT Card 4, IGNORE: p. 12-73 only (PDF p. 1351); Card header 'This card is optional.'

### F-C29 — Negative volume handling

**Claim.** Per the LS-DYNA support site, a negative volume causes the calculation to terminate unless ERODE=1 in *CONTROL_TIMESTEP and DTMIN in *CONTROL_TERMINATION is nonzero, in which case the offending element is deleted and the run usually continues. The R15 manual likewise describes PSFAIL on *CONTROL_SOLID as limiting negative-volume erosion to a part set, 'similar to setting ERODE = 1 in *CONTROL_TIMESTEP'.

- Source: *Negative volumes in brick elements (LS-DYNA support site how-to)* — undated web page, fetched 2026-09-21
- Locator: second paragraph
- URL: <https://www.dynasupport.com/howtos/element/negative-volumes-in-brick-elements>
- Kind / confidence: vendor-support / medium
- Corrected locator: PSFAIL corroboration: Vol I R15 *CONTROL_SOLID, PDF p. 1788 (approx. p. 12-510), not ~1790

### F-C30 — Restart: categories and command lines

**Claim.** Three restart categories: simple restart (R=rtf, no input deck), small restart (I=restartinput R=D3DUMPnn, deck contains only restart keywords such as *CHANGE_OPTION), full restart (full model plus *STRESS_INITIALIZATION; SMP: R=D3DUMPnn, MPP: N=D3FULLnn). File-family members are numbered consecutively from the last member before termination. d3dump is written at the end of every run and as requested by *DATABASE_BINARY_D3DUMP; runrsf is the running restart file.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: RESTART INPUT DATA, p. 48-1 (PDF p. 3797); Getting Started 'RESTART ANALYSIS' pp. 2-20 to 2-22 (PDF pp. 382-384); execution syntax p. 2-12/2-13 (PDF pp. 374-375): 'dpf = Dump file to write for purposes of restarting (default = d3dump). This file is written at the end of every run and during the run as requested by *DATABASE_BINARY_D3DUMP.'
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high

### F-C31 — Restart: traces in outputs

**Claim.** In a full restart, time and output intervals continue from the first job (time is not reset) and 'completely new databases will be generated with the time offset'; by default a full restart OVERWRITES existing ASCII output (*CHANGE/…IASCII=0) and appends only if IASCII=1. A small restart's d3hsp carries the heading given by the restart *TITLE (default 'LS-DYNA USER INPUT'), which is written to no other file.

- Source: *LS-DYNA Keyword User's Manual Volume I* — R15 (02/28/24)
- Locator: Getting Started, RESTART ANALYSIS, p. 2-22 (PDF p. 384); IASCII: RESTART INPUT DATA p. 48-6 (PDF p. 3802) 'EQ.0: Full restart overwrites existing ASCII output (default), EQ.1: Full restart appends to existing ASCII output.'; restart *TITLE p. 48-40 (PDF p. 3836) 'Heading to appear in a small restart's d3hsp file. This heading is not written to any other output file.'
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_I_R15.pdf>
- Kind / confidence: primary / high
- Corrected locator: IASCII is on *CHANGE_OUTPUT ('ASCII Output Overwrite Card. This format applies to the OUTPUT keyword option.'), p. 48-6 (PDF 3802)

### F-C32 — MAT_010 card layout

**Claim.** *MAT_ELASTIC_PLASTIC_HYDRO: Card 1 = MID, RO, G, SIG0, EH, PC, FS, CHARL. Card 2 = A1, A2, SPALL and is 'included if and only if the SPALL keyword option is used'. Cards 3-4 = EPS1-EPS16, Cards 5-6 = ES1-ES16, all four required. So in a plain *MAT_ELASTIC_PLASTIC_HYDRO (no _SPALL) there is no A1/A2/SPALL card at all.

- Source: *LS-DYNA Keyword User's Manual Volume II, Material Models* — R15 (R15@d71677e2e, 02/29/24)
- Locator: *MAT_010 Card Summary, p. 2-175 (PDF p. 243)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_II_R15.pdf>
- Kind / confidence: primary / high

### F-C33 — MAT_010 pressure hardening vs EPS/ES table

**Claim.** The manual gives the pressure-hardening term only in the SIG0/EH branch: sigma_y = sigma_0 + E_h*eps_p + (a1 + p*a2)*max[p,0], used 'If ES and EPS values are undefined'. When ES and EPS are specified, the manual says EH is ignored and gives the yield stress as sigma_y = f(eps_p), interpolated from the curve, with no a1/a2 term shown. A1 = 'Linear pressure hardening coefficient', A2 = 'Quadratic pressure hardening coefficient'. Pressure p positive in compression.

- Source: *LS-DYNA Keyword User's Manual Volume II, Material Models* — R15 (02/29/24)
- Locator: *MAT_010 Remark 2 (Yield Stress and Plastic Hardening Modulus), p. 2-179 (PDF p. 247)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_II_R15.pdf>
- Kind / confidence: primary / medium

### F-C34 — MAT_010 effective plastic strain definition

**Claim.** For MAT_010 effective stress is sqrt(3/2 s_ij s_ij) and effective plastic strain is the time integral of sqrt(2/3 Dp_ij Dp_ij); EPSi are 'Effective plastic strain (true)' with linear extrapolation beyond the last point; FS is the effective plastic strain at which erosion occurs.

- Source: *LS-DYNA Keyword User's Manual Volume II, Material Models* — R15 (02/29/24)
- Locator: *MAT_010 EPSi p. 2-177 (PDF p. 245); definitions in Remark 2 p. 2-179 (PDF p. 247)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_II_R15.pdf>
- Kind / confidence: primary / high

### F-C35 — MAT_072R3 'plastic strain' slot / NOUT

**Claim.** For *MAT_CONCRETE_DAMAGE_REL3 the quantity LS-PrePost labels 'plastic strain' is selected by NOUT (Card 3): 1 = current shear failure surface radius; 2 = scaled damage measure delta = 2*lambda/(lambda + lambda_m); 3 = strain energy (rate); 4 = plastic strain energy (rate). The scaled damage measure ranges 0 to 1 from yield to maximum failure surface and 1 to 2 from maximum to residual surface.

- Source: *LS-DYNA Keyword User's Manual Volume II, Material Models* — R15 (02/29/24)
- Locator: *MAT_072R3, 'Output of Selected Variables' and Table M72-1, pp. 2-518 to 2-519 (PDF pp. 586-587)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_II_R15.pdf>
- Kind / confidence: primary / high

### F-C36 — MAT_072R3 RSIZE / UCF

**Claim.** RSIZE = unit conversion factor for length in inches per user length unit (39.37 for metres); UCF = unit conversion factor for stress in psi per user stress unit (145 for MPa). The manual's mm-ms-g-MPa sample uses RSIZE 3.94E-2 and UCF 145.0.

- Source: *LS-DYNA Keyword User's Manual Volume II, Material Models* — R15 (02/29/24)
- Locator: *MAT_072R3, Card 3 variables, p. 2-516 (PDF p. 584); sample input p. 2-519/2-520 (PDF pp. 587-588)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_II_R15.pdf>
- Kind / confidence: primary / high

### F-C37 — MAT_072R3 extra history variables and d3hsp echo

**Claim.** Six extra history variables are available with NEIPH=6: #1 internal energy, #2 pressure from bulk viscosity, #3 volume in previous time step, #4 plastic volumetric strain, #5 slope of damage evolution (eta vs lambda) curve, #6 'modified' effective plastic strain (lambda). Auto-generated model parameters and the generated *EOS_TABULATED_COMPACTION parameters are written to d3hsp.

- Source: *LS-DYNA Keyword User's Manual Volume II, Material Models* — R15 (02/29/24)
- Locator: *MAT_072R3, Table M72-2 p. 2-519 (PDF p. 587); intro p. 2-514 (PDF p. 582)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_II_R15.pdf>
- Kind / confidence: primary / high

### F-C38 — MAT_003 BETA, SRC, SRP, VP

**Claim.** *MAT_PLASTIC_KINEMATIC: Card 1 = MID, RO, E, PR, SIGY, ETAN, BETA (ETAN, BETA default 0.0); Card 2 = SRC, SRP, FS, VP (defaults 0.0, 0.0, 1e20, 0.0). BETA is the hardening parameter, 0 = kinematic, 1 = isotropic. SRC and SRP are Cowper-Symonds C and p; the yield stress is scaled by 1 + (eps_dot/C)^(1/p); set both to zero to ignore rate effects. VP=0 scales yield stress (default), VP=1 viscoplastic formulation (recommended).

- Source: *LS-DYNA Keyword User's Manual Volume II, Material Models* — R15 (02/29/24)
- Locator: *MAT_003, Cards 1-2 p. 2-156, Remarks 1-2 pp. 2-157 to 2-158 (PDF pp. 224-226)
- URL: <https://lsdyna.ansys.com/wp-content/uploads/2026/08/LS-DYNA_Manual_Volume_II_R15.pdf>
- Kind / confidence: primary / high

---

## M — Material property ranges (for the plausibility screens)

Surveyed 2026-09-21 for ADR-0066 clause 2 (units are verified from outside the label). Same method as above: sources read, rows indexed, independent re-check. Primary sources read: the Cambridge University Engineering Department *Materials Data Book* (2003 and 2011 editions), Ashby's *Materials Selection in Mechanical Design* (2nd ed.) and Granta chart booklet, Kaye & Laby online (archived), EN 1992-1-1:2004, Greaves et al. (Nature Materials 2011), Trachenko et al. (Science Advances 2020). Not accessible: CRC and ASM handbooks, fib Model Code, ACI 318 (quoted only through a reproduction), Kolsky, Meyers.

### Density and elastic constants

**Claim.** Cast irons 7.05-7.25; high/medium/low carbon steels 7.8-7.9; low alloy steels 7.8-7.9; stainless steels 7.6-8.1 Mg/m3

- Source: *Materials Data Book, Cambridge University Engineering Department* — 2003 Edition
- Locator: Section II.2 DENSITY, printed p.10 (PDF p.12)
- URL: <https://www-mdp.eng.cam.ac.uk/web/library/enginfo/cueddatabooks/materials.pdf>
- Kind / confidence: primary / high

**Claim.** Aluminium alloys 2.5-2.9; copper alloys 8.93-8.94; lead alloys 10-11.4; magnesium alloys 1.74-1.95; nickel alloys 8.83-8.95; titanium alloys 4.4-4.8; zinc alloys 4.95-7 Mg/m3. Family min-max in this table: 1.74 to 11.4 Mg/m3.

- Source: *Materials Data Book, Cambridge University Engineering Department* — 2003 Edition
- Locator: Section II.2 DENSITY, printed p.10 (PDF p.12)
- URL: <https://www-mdp.eng.cam.ac.uk/web/library/enginfo/cueddatabooks/materials.pdf>
- Kind / confidence: primary / high

**Claim.** Glasses 2.17-2.8; brick 1.9-2.1; concrete (typical) 2.2-2.6; stone 2.5-3; technical ceramics 2.3 (silicon) to 15.9 (tungsten carbide) Mg/m3

- Source: *Materials Data Book, Cambridge University Engineering Department* — 2003 Edition
- Locator: Section II.2 DENSITY, printed p.10 (PDF p.12)
- URL: <https://www-mdp.eng.cam.ac.uk/web/library/enginfo/cueddatabooks/materials.pdf>
- Kind / confidence: primary / high

**Claim.** Al/SiC 2.66-2.9; CFRP 1.5-1.6; GFRP 1.75-1.97; bamboo 0.6-0.8; cork 0.12-0.24; leather 0.81-1.05; wood (typical) 0.6-0.8 Mg/m3

- Source: *Materials Data Book, Cambridge University Engineering Department* — 2003 Edition
- Locator: Section II.2 DENSITY, printed p.10 (PDF p.12)
- URL: <https://www-mdp.eng.cam.ac.uk/web/library/enginfo/cueddatabooks/materials.pdf>
- Kind / confidence: primary / high

**Claim.** Elastomers 0.9-1.8; thermoplastics 0.89 (PP) to 2.2 (PTFE); thermosets 1.04-1.4 Mg/m3

- Source: *Materials Data Book, Cambridge University Engineering Department* — 2003 Edition
- Locator: Section II.2 DENSITY, printed p.10 (PDF p.12)
- URL: <https://www-mdp.eng.cam.ac.uk/web/library/enginfo/cueddatabooks/materials.pdf>
- Kind / confidence: primary / high

**Claim.** Flexible polymer foams 0.016-0.115 Mg/m3; rigid polymer foams 0.036-0.47 Mg/m3. Lowest tabulated engineering-solid density: 0.016 Mg/m3 (16 kg/m3).

- Source: *Materials Data Book, Cambridge University Engineering Department* — 2003 Edition
- Locator: Section II.2 DENSITY, printed p.10 (PDF p.12)
- URL: <https://www-mdp.eng.cam.ac.uk/web/library/enginfo/cueddatabooks/materials.pdf>
- Kind / confidence: primary / high

**Claim.** Young's modulus of engineering materials spans 0.01 GPa (low-density foams) to 1000 GPa (diamond); density spans a factor of 2000, from less than 0.1 to 20 Mg/m3. The lower limit for true (non-foam, non-elastomer) solids is E ~ 1 GPa and density ~ 1 Mg/m3.

- Source: *M. F. Ashby, Materials Selection in Mechanical Design* — Second edition, Butterworth-Heinemann, 1999
- Locator: Ch. 4, Section 4.3 'The modulus-density chart (Chart 1, Figure 4.3)', printed pp. 36-38 (PDF pp. 47-49)
- URL: <https://www.utc.fr/~hagegebe/UV/MQ12/CORRECTIONS_TD/%5BASHBY99%5D%20-%20Materials%20Selection%20In%20Mechanical%20Design%202Ed.pdf>
- Kind / confidence: primary / medium

**Claim.** Approximate densities, g/cm3: carbon steel (<1% C) 7.8; mild steel 7.9; stainless steel 7.8; cast iron 7.0-7.4; brass (60/40) 8.4, (70/30) 8.5; phosphor-bronze 8.9; 'heavy alloy' 16.8-18.0; platinum-iridium (90/10) 21.5; tungsten carbide (6% Co) 15.0; granite 2.7; marble 2.7; slate 2.6-3.3; glass (soda) 2.5, (Pyrex) 2.23, (lead) 3-4; balsa wood 0.1-0.2; cork 0.25; lignum vitae 1.3; ice 0.92; sea water 1.03; petroleum 0.8; glycerine 1.3

- Source: *Kaye & Laby, Tables of Physical & Chemical Constants (online), NPL - Section 2.2.1 Densities* — Online Version 1.0/1.1 (page updated 21 Nov 2008; site archived 2017); Wayback capture 2019-04-22
- Locator: Table 'Approximate densities of commonly used materials'
- URL: <https://web.archive.org/web/20190422110321id_/http://www.kayelaby.npl.co.uk/general_physics/2_2/2_2_1.html>
- Kind / confidence: primary / medium

**Claim.** Water 998.20 kg/m3 at 20 C (999.97 at 4 C, 958.34 at 100 C); mercury 13 545.85 kg/m3 at 20 C (13 595.08 at 0 C); heavy water 1105.34 kg/m3 at 20 C

- Source: *Kaye & Laby, Tables of Physical & Chemical Constants (online), NPL - Section 2.2.1 Densities* — Online Version 1.1 (21 Nov 2008); Wayback capture 2019-04-22
- Locator: Tables 'Density of water', 'Density of heavy water', 'Density of mercury'
- URL: <https://web.archive.org/web/20190422110321id_/http://www.kayelaby.npl.co.uk/general_physics/2_2/2_2_1.html>
- Kind / confidence: primary / high

**Claim.** Osmium: density 22.5872 g/cm3; stated to be the densest of all the elements.

- Source: *Royal Society of Chemistry, Periodic Table - Osmium* — Web page as fetched 2026-09-21
- Locator: 'Fact box' (density) and 'Appearance' paragraph
- URL: <https://periodic-table.rsc.org/element/76/osmium>
- Kind / confidence: secondary / medium

**Claim.** Metallic foam density is adjustable over 0.05 to 10 g/cm3 depending on porosity and matrix metal.

- Source: *F. García-Moreno, 'Commercial Applications of Metal Foams: Their Properties and Production', Materials (MDPI), Review* — Published 29 January 2016
- Locator: PDF p.9 (Section 3, functional properties)
- URL: <https://pdfs.semanticscholar.org/e6c1/2c07e154fbfa2d105dfb0777facfa669fe0a.pdf>
- Kind / confidence: primary / medium

**Claim.** Cast irons 165-180; carbon steels 200-216; low alloy steels 201-217; stainless 189-210; aluminium alloys 68-82; copper alloys 112-148; lead alloys 12.5-15; magnesium alloys 42-47; nickel alloys 190-220; titanium alloys 90-120; zinc alloys 68-95 GPa. Family min-max in this table: 12.5 to 220 GPa.

- Source: *Materials Data Book, Cambridge University Engineering Department* — 2003 Edition
- Locator: Section II.3 YOUNG'S MODULUS, printed p.11 (PDF p.13)
- URL: <https://www-mdp.eng.cam.ac.uk/web/library/enginfo/cueddatabooks/materials.pdf>
- Kind / confidence: primary / high

**Claim.** Glasses 61-110; brick 10-50; concrete (typical) 25-38; stone 6.9-21; technical ceramics 140 (silicon) to 720 (tungsten carbide) GPa

- Source: *Materials Data Book, Cambridge University Engineering Department* — 2003 Edition
- Locator: Section II.3 YOUNG'S MODULUS, printed p.11 (PDF p.13)
- URL: <https://www-mdp.eng.cam.ac.uk/web/library/enginfo/cueddatabooks/materials.pdf>
- Kind / confidence: primary / high

**Claim.** Al/SiC 81-100; CFRP 69-150; GFRP 15-28; bamboo 15-20; cork 0.013-0.05; leather 0.1-0.5; wood longitudinal 6-20, transverse 0.5-3 GPa

- Source: *Materials Data Book, Cambridge University Engineering Department* — 2003 Edition
- Locator: Section II.3 YOUNG'S MODULUS, printed p.11 (PDF p.13)
- URL: <https://www-mdp.eng.cam.ac.uk/web/library/enginfo/cueddatabooks/materials.pdf>
- Kind / confidence: primary / high

**Claim.** Elastomers 0.0007-0.04 GPa; thermoplastics 0.2 (ionomer) to 5 (POM) GPa; thermosets 2.07-4.83 GPa; flexible polymer foams 0.0003-0.012 GPa; rigid polymer foams 0.023-0.48 GPa

- Source: *Materials Data Book, Cambridge University Engineering Department* — 2003 Edition
- Locator: Section II.3 YOUNG'S MODULUS, printed p.11 (PDF p.13)
- URL: <https://www-mdp.eng.cam.ac.uk/web/library/enginfo/cueddatabooks/materials.pdf>
- Kind / confidence: primary / high

**Claim.** At 20 C (E GPa / G GPa / nu / K GPa): Aluminium 70.3/26.1/0.345/75.5; Copper 129.8/48.3/0.343/137.8; Chromium 279.1/115.4/0.210/160.1; Gold 78.0/27.0/0.44/217.0; Iron (soft) 211.4/81.6/0.293/169.8; Iron (cast) 152.3/60.0/0.27/109.5; Lead 16.1/5.59/0.44/45.8; Magnesium 44.7/17.3/0.291/35.6; Titanium 115.7/43.8/0.321/107.7; Tungsten 411.0/160.6/0.280/311.0; Zinc 108.4/43.4/0.249/72.0; Brass (70/30) 100.6/37.3/0.350/111.8; Steel (mild) 211.9/82.2/0.291/169.2; Steel stainless 215.3/83.9/0.293/166.0; Tungsten carbide 534.4/219.0/0.22/319.0. Glasses: heavy flint 80.1/31.5/0.27/57.6; crown 71.3/29.2/0.22/41.2; fused quartz 73.1/31.2/0.17/36.9. Across the metals table nu runs 0.210 (Cr) to 0.44 (Au, Pb); G 5.59-160.6 GPa; K 31.3-311.0 GPa.

- Source: *Kaye & Laby, Tables of Physical & Chemical Constants (online), NPL - Section 2.2.2 Elasticities and strengths* — Online Version 1.0 (site archived 2017); Wayback capture 2019-04-22
- Locator: Tables 'Elasticities of metals and alloys' and 'Elasticities of glasses'
- URL: <https://web.archive.org/web/20190422003653/http://www.kayelaby.npl.co.uk/general_physics/2_2/2_2_2.html>
- Kind / confidence: primary / high

**Claim.** Woods: EL 6 (balsa, rel. density 0.2) to 16 GPa (ash/birch/Douglas fir), ER 0.3-2.2 GPa, ET 0.1-1.1 GPa. Plastics E (20 C, slow strain rate): 0.4 (PTFE, HDPE lower bound) to 4.2 GPa (polystyrene upper bound). Unidirectional high-modulus carbon fibre/epoxy: E11 287, E22 7.80, E33 7.75, G12 6.7, G13 6.7, G23 2.5 GPa, v12 0.30, v21 0.01, v23 0.55. Liquid bulk moduli: water 2.05 GPa (15 C, 1-25 atm), mercury 26.2 GPa, glycerine 4.03 GPa, pentane 0.318 GPa.

- Source: *Kaye & Laby, Tables of Physical & Chemical Constants (online), NPL - Section 2.2.2 Elasticities and strengths* — Online Version 1.0; Wayback capture 2019-04-22
- Locator: Tables 'Elasticities of woods', 'Elasticities of plastics', 'Elasticities of fibre-reinforced plastics – full set', 'Bulk moduli of liquids'
- URL: <https://web.archive.org/web/20190422003653/http://www.kayelaby.npl.co.uk/general_physics/2_2/2_2_2.html>
- Kind / confidence: primary / high

**Claim.** For isotropic materials -1 <= nu <= 1/2; all stable isotropic materials lie between these limits.

- Source: *G. N. Greaves, A. L. Greer, R. S. Lakes and T. Rouxel, 'Poisson's ratio and modern materials', Nature Materials 10, 823-837* — November 2011, DOI 10.1038/NMAT3134 (author-hosted PDF)
- Locator: Abstract and p.823, section 'Definition and physical significance'; Fig. 1a caption ('Numerical window of Poisson's ratio ν, from –1 to ½')
- URL: <https://silver.neep.wisc.edu/~lakes/PoiNatMat11.pdf>
- Kind / confidence: primary / high

**Claim.** Isotropic linear elasticity allows nu from -1 to 1/2 for an unconstrained object: positive bulk modulus implies nu > -1, positive shear modulus implies nu < 1/2. Relations: nu = (3K - 2G)/(6K + 2G); E = 2G(1 + nu); E = 3K(1 - 2nu). In anisotropic solids the range is unbounded.

- Source: *R. S. Lakes (University of Wisconsin-Madison), 'Poisson's ratio range in linear isotropic classical elasticity' and 'Meaning of Poisson's ratio'* — Web pages as fetched 2026-09-21
- Locator: Section 'Range for Poisson's ratio' and 'Anisotropy'; relations from https://silver.neep.wisc.edu/~lakes/PoissonIntro.html section 'Poisson's ratio: relation to elastic moduli in isotropic solids'
- URL: <https://silver.neep.wisc.edu/~lakes/PoissonElas.html>
- Kind / confidence: secondary / high

**Claim.** G = E / (2(1 + nu)); K = E / (3(1 - 2 nu)); K = EG / (3(3G - E)); M = K + (4/3)G. Rough guide for polycrystalline solids: nu ~ 1/3, G ~ 3E/8, K ~ E (breaks down for rubber and porous solids). Elastomers: nu = 1/2, G = E/3, K >> E.

- Source: *Kaye & Laby online Section 2.2.2; CUED Materials Data Book 2003; Ashby, Materials Selection in Mechanical Design 2nd ed. 1999* — K&L Version 1.0 (archived capture 2019-04-22); CUED 2003 Edition; Ashby 1999
- Locator: K&L: 'Elastic properties – isotropic materials'. CUED: Section I 'ELASTIC MODULI', printed p.4 (PDF p.6), https://www-mdp.eng.cam.ac.uk/web/library/enginfo/cueddatabooks/materials.pdf. Ashby: eqs (3.1)-(3.2b) printed p.23 and text printed p.38.
- URL: <https://web.archive.org/web/20190422003653/http://www.kayelaby.npl.co.uk/general_physics/2_2/2_2_2.html>
- Kind / confidence: primary / high

**Claim.** Metals: nu from 0.25 to 0.42 (B/G 1.7-5.6). Polymers: nu ~ 0.33. Ceramics, glasses and semiconductors: nu -> 1/4. Most well-known solids (metals, polymers, ceramics): 0.25 < nu < 0.35. Liquids and rubbers: nu -> 1/2. Cork: nu ~ 0. Re-entrant foams and some networks: nu < 0. Hard metals such as steel: nu ~ 1/3. Many glasses: 0.2; silica glass rises from 0.19 to 0.33 with densification.

- Source: *Greaves, Greer, Lakes and Rouxel, 'Poisson's ratio and modern materials', Nature Materials 10, 823-837* — November 2011, DOI 10.1038/NMAT3134 (author-hosted PDF)
- Locator: pp. 823-824, sections 'Definition and physical significance' and 'Poisson's ratio and packing density'
- URL: <https://silver.neep.wisc.edu/~lakes/PoiNatMat11.pdf>
- Kind / confidence: primary / high

**Claim.** Rubber 0.48-~0.5; indium 0.45; gold 0.42; lead 0.44; copper 0.37 / 0.35; aluminum 0.34; polystyrene 0.34; brass 0.33; ice 0.33; polystyrene foam 0.3; stainless steel 0.30; steel 0.29; tungsten 0.30 / 0.28; zinc 0.25; fused quartz 0.17; boron 0.08; beryllium 0.03; re-entrant foam -0.7. Beryllium has the lowest Poisson's ratio of any polycrystalline metal. Glassy polymers ~1/3, approaching 1/2 when rubbery.

- Source: *R. S. Lakes (University of Wisconsin-Madison), 'Meaning of Poisson's ratio'* — Web page as fetched 2026-09-21
- Locator: Section 'Poisson's ratio in various materials: table of Poisson's ratio'; section 'Poisson's ratio in viscoelastic materials'
- URL: <https://silver.neep.wisc.edu/~lakes/PoissonIntro.html>
- Kind / confidence: secondary / medium

**Claim.** Poisson's ratio is usually positive, lying between 0 and 1/2, but may be negative in some cases; it can exceed 0.5 in angle-ply or multidirectional laminates.

- Source: *Kaye & Laby online, NPL - Section 2.2.2 Elasticities and strengths* — Online Version 1.0; Wayback capture 2019-04-22
- Locator: 'Elastic properties – isotropic materials' and 'Elastic properties – anisotropic materials'
- URL: <https://web.archive.org/web/20190422003653/http://www.kayelaby.npl.co.uk/general_physics/2_2/2_2_2.html>
- Kind / confidence: primary / medium

**Claim.** Ecm (secant, 0 to 0.4 fcm, quartzite aggregate) = 27, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 41, 42, 44 GPa for fck = 12, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90 MPa. Range 27-44 GPa. Ecm = 22[(fcm)/10]^0.3 (fcm in MPa), fcm = fck + 8 MPa. Aggregate corrections: limestone -10%, sandstone -30%, basalt +20%. Recommended maximum class C90/105.

- Source: *EN 1992-1-1:2004 Eurocode 2: Design of concrete structures - Part 1-1: General rules and rules for buildings (BS EN 1992-1-1:2004)* — 2004 (E), incorporating corrigendum markers; public.resource.org scan
- Locator: Table 3.1, printed p.29 (PDF p.31); clause 3.1.3(2) printed p.28; clause 3.1.2(2) Note (Cmax = C90/105) printed p.27
- URL: <https://www.phd.eng.br/wp-content/uploads/2015/12/en.1992.1.1.2004.pdf>
- Kind / confidence: primary / high

**Claim.** nu = 0.2 for uncracked concrete and 0 for cracked concrete.

- Source: *EN 1992-1-1:2004 Eurocode 2: Design of concrete structures - Part 1-1* — 2004 (E)
- Locator: Clause 3.1.3(4), printed p.30 (PDF p.32)
- URL: <https://www.phd.eng.br/wp-content/uploads/2015/12/en.1992.1.1.2004.pdf>
- Kind / confidence: primary / high

**Claim.** LWAC is defined as density not more than 2200 kg/m3. EN 206-1 density classes 1,0 / 1,2 / 1,4 / 1,6 / 1,8 / 2,0 = 801-1000 / 1001-1200 / 1201-1400 / 1401-1600 / 1601-1800 / 1801-2000 kg/m3; design densities plain 1050-2050, reinforced 1150-2150 kg/m3. Modulus: Elcm = Ecm x eta_E, eta_E = (rho/2200)^2.

- Source: *EN 1992-1-1:2004 Eurocode 2: Design of concrete structures - Part 1-1* — 2004 (E)
- Locator: Section 11: clause 11.1.1(4)P printed p.185 (PDF p.187); Table 11.1 and clause 11.3.2(1) printed p.186 (PDF p.188)
- URL: <https://www.phd.eng.br/wp-content/uploads/2015/12/en.1992.1.1.2004.pdf>
- Kind / confidence: primary / high

**Claim.** ACI 318-19 19.2.2.1: for wc between 90 and 160 lb/ft3 (1440 and 2560 kg/m3), Ec = wc^1.5 x 33 sqrt(f'c) psi [= wc^1.5 x 0.043 sqrt(f'c) MPa]; for normalweight concrete Ec = 57,000 sqrt(f'c) psi [= 4700 sqrt(f'c) MPa]. Lightweight concrete defined as equilibrium density between 90 and 135 lb/ft3 (1440 and 2160 kg/m3). Default normalweight density 150 lb/ft3 (2400 kg/m3).

- Source: *ESCSI Information Sheet 7900.095-2019, 'Structural Lightweight Concrete in the Building Code Requirements for Structural Concrete (ACI 318-19)' - reproduces ACI 318-19 code text* — ESCSI 7900.095-2019, revised March 2024 (page footers: 'Revised July 2022')
- Locator: Sheet p.17 (19.2.2 Modulus of elasticity); p.3 (2.3 Terminology); p.31 (27.4.6.5)
- URL: <https://www.escsi.org/wp-content/uploads/2024/08/ESCSI_Technical-Sheet_7900.095-2019_FINAL.pdf>
- Kind / confidence: secondary / medium

**Claim.** Reinforcing steel: mean density 7850 kg/m3; design modulus of elasticity Es = 200 GPa.

- Source: *EN 1992-1-1:2004 Eurocode 2: Design of concrete structures - Part 1-1* — 2004 (E)
- Locator: Clause 3.2.7(3) and (4), printed pp.40-41 (PDF pp.42-43)
- URL: <https://www.phd.eng.br/wp-content/uploads/2015/12/en.1992.1.1.2004.pdf>
- Kind / confidence: primary / high

### Strength, yield strain, wave speed, impact velocity

**Claim.** Yield stress sigma_y and tensile strength sigma_ts, MPa, min-max per class. Across the listed metals, sigma_y spans 8 MPa (lead alloys) to 1245 MPa (titanium alloys), and sigma_ts spans 12 MPa (lead alloys) to 2240 MPa (stainless steels). Low carbon steels: sigma_y 250-395, sigma_ts 345-580. Aluminium alloys: sigma_y 30-500, sigma_ts 58-550. Copper alloys: sigma_y 30-500, sigma_ts 100-550.

- Source: *Materials Data Book, 2011 Edition (Revised 2019), Cambridge University Engineering Department* — 2011 Edition, revised 2019
- Locator: Section II.4 'Yield stress, sigma_y, and tensile strength, sigma_ts', printed page 12 (PDF page 14), left table, Metals block
- URL: <https://teaching.eng.cam.ac.uk/sites/teaching.eng.cam.ac.uk/files/Documents/Databooks/MATERIALS%20DATABOOK%20(2011)%20version%20for%20Moodle.pdf>
- Kind / confidence: primary / high

**Claim.** For ceramics the sigma_y column holds COMPRESSIVE strength, MPa. Concrete, typical: 32-60 compressive, 2-6 tensile. Stone: 34-248 compressive, 5-17 tensile. Brick: 50-140 compressive, 7-14 tensile. Glasses: 264-2129 compressive, 22-177 tensile. Technical ceramics: compressive 524 (silicon nitride minimum) to 6833 (tungsten carbide maximum), tensile 160-800. The source states that ceramics are of the order of 10 times stronger in compression than in tension.

- Source: *Materials Data Book, 2011 Edition (Revised 2019), Cambridge University Engineering Department* — 2011 Edition, revised 2019
- Locator: Section II.4, printed page 12 (PDF page 14), Ceramics block and footnote (*)
- URL: <https://teaching.eng.cam.ac.uk/sites/teaching.eng.cam.ac.uk/files/Documents/Databooks/MATERIALS%20DATABOOK%20(2011)%20version%20for%20Moodle.pdf>
- Kind / confidence: primary / high

**Claim.** All values in MPa, as sigma_y | sigma_ts. Elastomers: 2-51 | 2.4-51. Thermoplastics: 8.3-95 | 17.2-165. Thermosets: 27.6-71.7 | 34.5-89.6. Polymer foams: 0.01-12 | 0.24-12.4. CFRP: 550-1050 | 550-1050. GFRP: 110-192 | 138-241. Al/SiC: 280-324 | 290-365. Wood, longitudinal: 30-70 | 60-100. Wood, transverse: 2-6 | 4-9. Cork: 0.3-1.5 | 0.5-2.5. Young's modulus from Table II.3 of the same book, in GPa: steels 189-217, aluminium alloys 68-82, copper alloys 112-148, concrete 15-25, stone 20-60, elastomers 0.0007-0.04, thermoplastics 0.2-5, thermosets 2.07-4.83, polymer foams 0.0003-0.48, CFRP 69-150, GFRP 15-28, wood (longitudinal) 6-20, tungsten carbide 600-720.

- Source: *Materials Data Book, 2011 Edition (Revised 2019), Cambridge University Engineering Department* — 2011 Edition, revised 2019
- Locator: Section II.4, printed page 12 (PDF page 14); E values from Section II.3 'Young's modulus, E', printed page 11 (PDF page 13)
- URL: <https://teaching.eng.cam.ac.uk/sites/teaching.eng.cam.ac.uk/files/Documents/Databooks/MATERIALS%20DATABOOK%20(2011)%20version%20for%20Moodle.pdf>
- Kind / confidence: primary / high

**Claim.** On Ashby's modulus-strength chart the plotted 'strength' differs by family: 0.2% offset yield for metals, 1% yield strength for polymers, compressive crushing strength for ceramics and glasses (roughly 15 times the tensile strength), tensile strength for composites, and tear strength for elastomers. The chart contours show the failure strain sigma_f/E. The contour labels printed on the chart are 10^-4, 10^-3, 10^-2 and 10^-1 (dimensionless), under the heading 'Yield strain sigma_f/E = 10^-4'.

- Source: *M. F. Ashby, 'Material and Process Selection Charts' (The CES EduPack Resource Booklet 2), Granta Design* — Version 1, (c) Granta Design, January 2010
- Locator: Chart 3 'Young's modulus, E, against Strength, sigma_f', printed page 7 (PDF page 9). Chart 4, printed page 8: 'The contours show the yield strain, sigma_f/E.' Chart 2, printed page 6, says for polymers 'typically, a strain of abut 1%'.
- URL: <http://users.fs.cvut.cz/libor.benes/vyuka/engineeringmaterials/2-Materials-Charts-2010.pdf>
- Kind / confidence: primary / medium

**Claim.** Across the 14 strength classes: fck (cylinder) 12-90 MPa, fck,cube 15-105 MPa, fcm 20-98 MPa, fctm 1,6-5,0 MPa, fctk,0,05 1,1-3,5 MPa, fctk,0,95 2,0-6,6 MPa, Ecm 27-44 GPa, eps_c1 (strain at peak stress) 1,8-2,8 per mille, eps_cu1 3,5-2,8 per mille. The recommended maximum class is C90/105. Reinforcing steel: the application rules are valid for fyk = 400 to 600 MPa.

- Source: *EN 1992-1-1:2004 Eurocode 2: Design of concrete structures - Part 1-1: General rules and rules for buildings (BS EN 1992-1-1:2004 print incorporating corrigendum AC2)* — EN 1992-1-1:2004 (E)
- Locator: Table 3.1, printed page 29 (PDF page 31, rotated page, read from the rendered image); clause 3.1.2(2)P printed page 27 (PDF page 29); clause 3.2.2(3)P printed page 38 (PDF page 40); Annex C Table C.1 also gives 'Characteristic yield strength fyk or f0,2k (MPa) 400 to 600'
- URL: <https://www.phd.eng.br/wp-content/uploads/2015/12/en.1992.1.1.2004.pdf>
- Kind / confidence: primary / high

**Claim.** The contours on Ashby's E-rho chart are the longitudinal wave speed in m/s. The contours drawn and labelled on the chart are 10^2 m/s, 10^3 m/s and 10^4 m/s. The source also states that moduli of engineering materials span a range of 10^7 and densities a range of 3000.

- Source: *M. F. Ashby, 'Material and Process Selection Charts' (The CES EduPack Resource Booklet 2), Granta Design* — Version 1, (c) Granta Design, January 2010
- Locator: Chart 1 'Young's modulus, E and Density, rho', printed page 5 (PDF page 7). The contour labels '10^4 m/s', '10^3 m/s' and '10^2 m/s' were read from the figure under the heading 'Longitudinal wave speed'.
- URL: <http://users.fs.cvut.cz/libor.benes/vyuka/engineeringmaterials/2-Materials-Charts-2010.pdf>
- Kind / confidence: primary / medium

**Claim.** VL / VS in mm/us (= km/s). Beryllium 12.89 / 8.88; silicon carbide 13.06 / 7.27; sapphire 11.1 / 6.04; boron carbide 11.0; alumina 10.52; aluminium (rolled) 6.42 / 3.04; mild steel 5.9 / 3.2; stainless 347 5.79 / 3.10; iron 5.9 / 3.2; cast iron 4.6 / 2.6; copper (rolled) 5.01 / 2.27; titanium 6.1 / 3.1; tungsten 5.2 / 2.9; lead 2.2 / 0.7. VL only: granite 6.5; marble 3.8; brick 4.3; concrete 3.1. Glasses: fused silica 5.70 / 3.75; pyrex 5.64 / 3.28; soda lime VL 6.00. VL only: oak 4.0; pine 3.5; balsa 0.80; cork 0.5. Plastics VL about 1.59 (Bakelite) to 2.90 (Melopas), for example ABS 2.23, acrylic 2.75, nylon 6/6 2.6 (VS 1.1), polycarbonate 2.30. Rubbers VL: butyl 1.80, neoprene 1.6, silicone (Silastic) 1.02-1.04.

- Source: *Onda Corporation, 'Acoustic Properties of Solids', 'Acoustic Properties of Plastics' and 'Acoustic Properties of Rubbers' tables* — Updated April 11, 2003
- Locator: Solids.pdf pages 1-7 (alphabetical). Plastics: https://www.ondacorp.com/wp-content/uploads/2020/09/Plastics.pdf page 1. Rubbers: https://www.ondacorp.com/wp-content/uploads/2020/09/Rubbers.pdf page 1.
- URL: <https://www.ondacorp.com/wp-content/uploads/2020/09/Solids.pdf>
- Kind / confidence: secondary / medium

**Claim.** The theoretical upper bound for the speed of sound in condensed phases is vu of about 36,100 m/s. Experimental longitudinal speeds in 133 systems all lie below it. vu is about twice the speed in diamond, which the paper calls the highest speed of sound measured at ambient conditions; the in-plane speed in graphite is slightly above that in diamond. A fit to the elemental data gives an intercept of 37,350 m/s.

- Source: *K. Trachenko, B. Monserrat, C. J. Pickard, V. V. Brazhkin, 'Speed of sound from fundamental physical constants', Science Advances 6, eabc8662 (2020)* — arXiv:2004.04818 (version downloaded 2026-09-21)
- Locator: Abstract (PDF page 1); Eq. (9) and the discussion of Figs. 1-2 (PDF pages 3-4)
- URL: <https://arxiv.org/pdf/2004.04818>
- Kind / confidence: primary / high

**Claim.** NASA: hypervelocity is 'greater than 3 km/s'; about 3 km/s is described as 'low velocity' against 7 km/s. ESA: in hypervelocity impacts the projectile velocity exceeds the speed of sound within the target material; beyond 4 km/s (depending on materials) the projectile breaks up and melts completely; impact velocities can reach 15 km/s for space debris and 72 km/s for meteoroids, with about 14 km/s typical for debris; 'At low velocities, plastic deformation normally prevails.'

- Source: *NASA JSC ARES HyperVelocity Impact Technology, 'Impact Physics' web page; ESA 'Hypervelocity impacts and protecting spacecraft' web page* — Web pages as fetched 2026-09-21 (undated)
- Locator: NASA: section 'The Debris Cloud' / 'Ballistic Limits'. ESA: https://www.esa.int/Space_Safety/Space_Debris/Hypervelocity_impacts_and_protecting_spacecraft, sections 'Effects of hypervelocity impacts' and 'Hypervelocity impact testing'.
- URL: <https://hvit.jsc.nasa.gov/hypervelocity-testing/impact-physics.html>
- Kind / confidence: primary / medium

---

## V — Level derivation record

How each reference level in the design doc was derived from the claims above under ADR-0066 clause 7's attachment rule, and what an adversarial pass said about it. The design doc's *Reference levels* table is the adjudicated result; this is the working record. "Proposed" is the first pass; "Adversary" is the second.

**Adversary's overall finding.** Read-only adversarial review of each proposed level against its sources at their locators. 12 proposed levels: 1 sound, 8 sound with edits, 1 not attachable, 1 with hidden discretion, 1 whose scope was influenced by the known test-bed value. None of the numbers misreads its source. The defects are in scopes, in two evaluation-time readings, and in a doctrine the proposer invented. **Cross-cutting findings** 1. **The "one-directional difference" doctrine does not exist.** - The proposer tolerates statistic mismatches "on the precedent clause 7 sets for B-BLM-1". Clause 7 never names B-BLM-1. It says only that sampled extremes "are lower bounds on the solver-step extremes". - The B-BLM-1 attachment is asserted in design-doc physics note 4 and in the dossier header, and both are subordinate to the ADR. - The rule gives two routes: identical, or equal under a condition the scope records. A reading is not a scope-recordable condition. - Three attachments lean on the doctrine: W-W179-02 on `r_final`, B-LSD-2, and W-W179-06's evaluation time. Each must stand on the rule itself or go. 2. **The `{explicit}` energy scope carries a documented influence from the known value.** - The design doc ("Calls made in drafting") justifies including non-conservative particle formulations with "would leave the test bed's energy rise unflagged". - The proposer deletes that sentence, keeps the scope, and says "B-BLM-2's 'central difference method' wording already carries the scope". That wording supports `explicit` only. The source is a finite-element text and says nothing about particle or advecting formulations. - Clause 7 concedes that for those formulations the ledger "is not built from the work of its internal forces". By the attachment rule's own test, the quantity is then not the source's statistic there. - Deleting the sentence would erase the record of the influence and leave the scope it produced in place. The call has to be re-decided on source grounds and its history recorded. 3. **Domain generalisation of W-W179-02.** - W179's 10 % level is proposed at `{explicit, initial_energy_driven}` with no spatial trait. - A finite-element roadside-crash practice limit would therefore issue verdicts on particle runs, including the known test-bed run. - No ADR sentence extends W179 across discretisations. The "general level" sentence concerns the explicit-integration level. 4. **The physics-note-4 finding is right, but its example is wrong.** - With zero external work, the kinetic branch of r's denominator needs `E_kin(t) > E_tot(t0)`. That does not happen early in an impact. - The branch that matters is the non-kinetic one, late in an impact that comes to rest after a net gain. That is the normal end state for this platform's impact benchmarks, not a rare one. - Two consequences: - W179's start-to-end statistic and `r_final` differ systematically in that regime. This strengthens the case for a separate quantity. - The design doc's stage-2 acceptance sentence ("the measure must return ...") cannot be asserted from the solver ratio. It has to be recomputed from r's definition or restricted to the first-argument branch. - Neither consequence touches a level. 5. **Strictness conflicts with clause 7.** - Every proposed bound is written `<`. Clause 7 says "At or below the level the verdict is `pass`". - A `hi_strict` flag therefore needs an edit to clause 7, not only to `Criterion`. The alternative is for every "less than" rationale to state that equality passes here, contrary to the source. 6. **Evidence hygiene.** - The W-R894-01 quote is truncated at "The part/material with…". "Unchanged in Report 894" for the per-part and moving-part added-mass rows therefore rests on claim text, not a quote. - The proposer completes the truncated word "reflectio…" in W-W179-15. - W-W179-12 is listed as context-only although it is the only evidence for the energy denominator. 7. **Internal-energy composition is an undisclosed normalisation choice on four of the five rows that divide by internal energy.** - The sources' solver books stiffness-damping dissipation into internal energy (L-C14, L-C15). L-C17 extends this to bulk viscosity, flagged in the dossier as an inference…

### V — `energy_gain_max`  <= 0.01

- Statistic: Whole model. Maximum over all ledger samples t of r(t), counted where r > 0. r(t) = {[E_tot(t) - E_tot(t0)] - [W_ext(t) - W_ext(t0)]} / max(|E_tot(t0) + W_ext(t) - W_ext(t0)|, E_kin(t), E_tot(t) - E_kin(t)). E_tot is the sum of every term of the declared balance identity. t0 is the first ledger sample. Evaluation time: the whole run.
- Scope proposed: {explicit}
- Sources: B-BLM-1, B-BLM-2 · match: identical
- Discretion remaining: (1) 'On the order of 10^-2' is rendered as 0.01, the only number the source prints. The source does not delimit the band that 'order of' admits, so any other figure would be the platform's. (2) The source is sign-agnostic. Applying 0.01 separately to gain and loss is equivalent to |r| <= 0.01 at every sample. (3) The source checks every step, and the quantity is a sampled maximum, which is a lower bound. `review` is therefore sound and `pass` is weaker than the source's check. (4) For systems 'on the order of 10^5 nodes or larger' the source balances subdomains, but the quantity is whole-model. The scope vocabulary cannot record this condition. Publish the node count as a scope fact and state the point in the rationale. (5) Equations 6.2.14-17, which define the three symbols, were read only as snippets. Spot-check the full page before ratifying.
- **Adversary: influenced-by-known-value.** 1. **The number stands.** - 0.01 is the only figure printed. - It is not sensitive to the known value under any half-decade reading of "order of". - Quote "generally" as well as "on the order of" in the hedge. 2. **Change `match` from "identical" to "equal under the reading clause 7 fixes".** - The reading: W_int is all non-kinetic ledger terms, and energy at t0 counts as input. - The match is literal only when E_tot(t0) = 0. No scope token records that. - Set `Criterion.provisional=True`. - Make reading the full p. 336 with eqs 6.2.14-17 a precondition of ratification, not a suggestion. 3. **Scope.** - Strike "B-BLM-2's 'central difference method' wording already carries the scope". That wording supports `explicit` only. - The only recorded argument for covering `particle_nonconservative` and `advecting_mesh` is the design-doc line "would leave the test bed's energy rise unflagged". That argues a scope from a measurement. - Do not just delete the line. Re-decide the call on source grounds and record that it was first argued from the known gain. - **Rule-derived scope:** two levels with the same bound, `{explicit, lagrangian_mesh}` and `{explicit, particle_conservative}`. - These are disjoint because every part must fall inside the scope. - `particle_nonconservative` and `advecting_mesh` get no level and read `no_ratified_criterion`, because clause 7 itself says their ledger "…

### V — `energy_loss_max`  <= 0.01

- Statistic: Whole model. Maximum over all ledger samples t of -r(t), counted where r < 0. r(t) is defined exactly as for `energy_gain_max`. Reported as a positive magnitude. Evaluation time: the whole run.
- Scope proposed: {explicit}
- Sources: B-BLM-1, B-BLM-2 · match: identical
- Discretion remaining: Same five points as `energy_gain_max`. One addition: the source's stated purpose is detecting 'spurious generation of energy', which is the gain direction. Applying the same number to losses follows the absolute value in the inequality itself, not the prose.
- **Adversary: sound-with-edit.** - Apply edits 2-5 of `energy_gain_max` unchanged. Same source, same scope. - The known test-bed value is a gain, so neither the number nor the direction here is touched. - The scope is shared with the gain, so it inherits the same re-decision. - Keep the disclosed point that applying 0.01 to losses follows the inequality's absolute value, not the prose purpose.

### V — `energy_residual_final`  abs <= 0.10

- Statistic: Whole model. r(t_end), signed. r(t) is defined as for `energy_gain_max`. Evaluation time: the last ledger sample, measured against the first.
- Scope proposed: {explicit, initial_energy_driven}
- Sources: W-W179-02, W-W179-12, W-R894-01 · match: equal-under-condition
- Discretion remaining: (1) The number 10 % involves no discretion. (2) The denominator 'initial energy' is not in the table row. It comes from the narrative quote W-W179-12: 'change in energy ... as a percent of the initial energy'. (3) Condition (ii) is tolerated on the precedent of B-BLM-1: the difference is one-directional, bounded and stated. The strict-letter alternative is a separate quantity, `total_energy_change_final` = [E_tot(t_end) - E_tot(t0)] / E_tot(t0), gated on `initial_energy_driven`. That would leave r_final with no level. (4) Evaluating at end-of-run follows the row. The narrative's 'maximum change ... can be reported' and its 'e.g., say 5 or 10 percent' are illustration and yield nothing. (5) Including `explicit` is a domain choice. Implicit runs get no level rather than one taken out of domain.
- **Adversary: discretion-hidden.** - **(a) Move the level off `r_final`.** - Add a catalogue row `total_energy_change_final` = [E_tot(t_end) − E_tot(t0)] / E_tot(t0), signed, with bound abs <= 0.10. - `r_final` has a different denominator whenever a bucket exceeds E_tot(t0). - That is the normal end state of an impact that comes to rest after a net gain. - The condition is a fact of the measurement that no scope token records. The rule's clause "each statistic given a level is its own catalogue quantity" therefore applies. - The proposer split hourglass energy six ways on the same ground. - `energy_residual_final` stays measured with no level. Do not tolerate the (0.10, 0.111] band on the invented "precedent". - **(b) Disclose the evaluation-time ambiguity.** - The proposer takes the denominator from W-W179-12 but dismisses the same passage's "maximum change ... over the simulation run" as illustration. It cannot have both. - The rationale must say: - The row is read as end versus start, as the ADR and dossier header finding 3 read it. - The narrative and the dossier caveat also admit a maximum-over-run reading. - Under that reading the source's statistic is >= this one, so `pass` is not guaranteed under both readings. - If the maintainer treats this as a hedge on "when", the proposer's own blocking rule makes the source context-only. - **(c) Add `lagrangian_mesh` to the scope: `{explicit, lagrangian_mesh, init…

### V — `zero_energy_mode_final_over_initial_total`  < 0.05

- Statistic: Whole model. Zero-energy-mode (hourglass) ledger term at the last ledger sample, divided by E_tot(t0), the sum of all balance-identity terms at the first sample.
- Scope proposed: {explicit, initial_energy_driven}
- Sources: W-W179-03, W-R894-01 · match: identical
- Discretion remaining: None in the number, body, denominator or time. 'Less than' is strict, while ADR clause 7 passes at equality (see the catalogue changes). The spatial class is left out of the scope on purpose: the gate already requires under-integrated parts, and the whole-model denominator is the same whatever else the model contains.
- **Adversary: sound-with-edit.** - The statistic, the normalisation and the evaluation time are identical. Keep the level. - **Provenance.** Strike "The hourglass limits trace to Du Bois, NOTES, 1998" from this rationale. - Du Bois is cited only for the ~0.1 per-element ratio against internal energy. - Replace with: "no provenance is stated for the 5 %-of-initial-energy figure in the pages read". - **Trait gate.** Remove `initial_energy_driven` from the gate and keep it in the scope. - The quantity exists whenever under-integrated parts exist and E_tot(t0) > 0. - An externally driven run must read `no_ratified_criterion`, which the platform owns. - It must not read `not_applicable`, which clause 5 reserves for a quantity that does not exist for the formulation. - **Scope.** Add `lagrangian_mesh`, or state in the rationale that applying the level to mixed mesh-and-particle models is the platform's extension of a finite-element source. - **Strictness.** State that equality passes here although the source says "less than", or amend clause 7.

### V — `zero_energy_mode_final_over_internal_final`  < 0.10

- Statistic: Whole model. Zero-energy-mode ledger term at the last sample, divided by the ledger's internal-energy term at the last sample.
- Scope proposed: {explicit}
- Sources: W-W179-04 · match: identical
- Discretion remaining: None in the number or the statistic. 'Internal energy' is the ledger's internal term as the balance identity declares it. The source does not define its composition. In the source's own solver it includes stiffness-damping and bulk-viscosity dissipation (L-C14, L-C15, L-C17).
- **Adversary: sound-with-edit.** - The statistic and the evaluation time are identical. Keep the level. - **Fix the denominator token once in the catalogue.** Define "internal energy" in the sources' sense: - It is the work of element internal forces. - It includes dissipation the sources' solver books as internal: stiffness damping and bulk or artificial viscosity. - It excludes zero-energy-mode, contact, rigid-surface and mass-damping terms. - **Why the proposer's definition falls short.** - It excludes only the last group. - E5 requires damping and artificial viscosity as separate ledger terms. - A ledger that follows E5 literally would therefore give a smaller denominator than the source's, and `review` would be stricter than the source. - State this as a platform choice in the rationale of every row normalised by internal energy, not only this one. - Note that L-C17's placement of bulk viscosity is an inference.

### V — `zero_energy_mode_top_part_final_over_internal_final`  < 0.10

- Statistic: One part only. p* is the part with the largest zero-energy-mode energy at the last sample. The value is HG_p*(t_end) / IE_p*(t_end). This is NOT the maximum of the ratio over all parts.
- Scope proposed: {explicit}
- Sources: W-W179-05 · match: identical
- Discretion remaining: None, provided the part is selected as the source selects it, by absolute hourglass energy. Judging every part against 0.10 end-over-end would be a stricter statistic that no source states. That reading yields no level.
- **Adversary: sound.** - None of substance. - The proposer's literal reading is the least discretionary one. - It is correctly distinguished from a maximum of the ratio over all parts. - **Minor.** - Say in the rationale that the tie-break (larger ratio) is a platform choice. - Carry the internal-energy composition sentence. - Add the strictness note.

### V — `zero_energy_mode_top_part_peak_over_initial_total`  < 0.05

- Statistic: Maximum over parts and over all ledger samples of the per-part zero-energy-mode energy, divided by the whole model's E_tot(t0).
- Scope proposed: {explicit, initial_energy_driven}
- Sources: W-W179-11, W-R894-01 · match: identical
- Discretion remaining: None in the number or the statistic. The report does not explain why Table 22 differs from Table E-1 (W-W179-11). Both wordings are verified quotes, so both are kept on separate quantities.
- **Adversary: sound-with-edit.** - The statistic matches the wording. - **Present levelling both W179 variants as a decision for the maintainer, not as settled.** The dossier explicitly asks for the variant to be named. - **Rationale changes.** - Say that this wording comes from a worked example and from one user's application (Report 894), not from the normative table. - Say that W179's "must be satisfied" sentence does not cover it. - Replace "shows the variant is in practical use" with "one later application uses it (7 table instances)". - Replace "Domain and authority are as for the other W179 rows". No provenance is stated for the 5 % figure. - **Sampling.** Add the sampled-maximum caveat. - "At any time during the run" evaluated on E6 samples is a lower bound. - **Trait gate.** Make the same change as for `zero_energy_mode_final_over_initial_total`: `initial_energy_driven` belongs in the scope, not in the gate.

### V — `zero_energy_mode_peak_over_internal_peak`  < 0.10

- Statistic: Whole model. (Maximum over the run of the zero-energy-mode ledger term) divided by (maximum over the run of the internal-energy term). The two maxima are taken independently in time.
- Scope proposed: {explicit}
- Sources: W-ENCAP-01, W-ENCAP-02 · match: identical
- Discretion remaining: None in the number or the statistic. Scoping to `explicit` is inferred from the same criteria list, which carries a mass-scaling limit. That limit is a device of explicit integration.
- **Adversary: sound-with-edit.** - Keep the level. - **Change "None in the number or the statistic" to list three discretions.** 1. "Max. internal energy" is read as that of the full setup, by parallel with the per-dummy bullet. The quote does not say so. 2. The composition of internal energy (see the row above). 3. The maxima are sampled maxima. - **Domain.** - Keep the rationale's domain sentence. - Add that no run trait can record "occupant / sled virtual testing". The extension to all explicit meshed runs is therefore the platform's. - Consider adding `lagrangian_mesh`. - Add the strictness note.

### V — `zero_energy_mode_part_peak_over_internal_peak`  < 0.10

- Statistic: Each under-integrated part p. (Maximum over the run of HG_p) divided by (maximum over the run of IE_p). Judged for every part, with the worst part reported.
- Scope proposed: {explicit}
- Sources: B-LSD-2, L-C43 · match: equal-under-condition
- Discretion remaining: The evaluation time of the numerator is unstated, and the dossier caveat says so. The denominator (the part's peak internal energy), the body (each part) and the number are all explicit. 'Rule-of-thumb' hedges authority, not the statistic, and is quoted. This is a borderline attachment. It rests on the one-directional-difference precedent that clause 7 sets for B-BLM-1.
- **Adversary: not-attachable.** - **Demote to context-only.** - **Why it cannot attach.** - The numerator's evaluation time is not stated, so identity cannot be established. - The proposer's "equal-under-condition" rests on a reading, not on a condition a scope can record. - The proposer concedes this: "This condition is a reading, not a run trait". - Its only support is the "one-directional precedent", which clause 7 does not contain. - The proposer's own rule says a hedge on "when" blocks a level. It applies that rule to B-LSD-3 on the same page family. Apply it here too. - **Catalogue.** Drop this row from the six-way split. - Every-part peak over peak then has no level and reads `no_ratified_criterion`. - **Context use.** Mention B-LSD-2 in the Euro NCAP row's rationale as agreeing on the number and on peak normalisation. - **If the maintainer still wants it**, two things are needed: - An explicit amendment to clause 7 admitting "bounding readings". - `provisional=True` on the level.

### V — `added_mass_fraction`  < 0.05

- Statistic: Whole model. (Maximum over the run of non-physical mass added by mass scaling) divided by (physical model mass at the first sample).
- Scope proposed: {explicit, mass_scaled}
- Sources: W-ENCAP-01, W-ENCAP-02, W-W179-06, W-R894-01 · match: identical
- Discretion remaining: None in the number, body or denominator. The level applies whatever the declared intent. Excluding quasi-static runs would let a dataset shed the level by labelling itself, and the Scope type has no negation anyway. '2.5% in case of separate models' is not a derivable trait and is context only.
- **Adversary: sound-with-edit.** - **Keep the level on Euro NCAP alone as the attached source.** It says "Max." and is identical. - Re-label W-W179-06 as corroborating under L-C32's condition (non-decreasing added mass, which no scope token records). It is not "identical". - Change "the one level with two independent sources stating the same statistic" to "same number and body; W179 states no evaluation time". - **Change "None in the number, body or denominator" to two disclosures.** 1. The source prints two numbers, 5 % and 2.5 %. The platform takes the one for a single complete model, because "separate models" is not derivable. 2. "Total model mass at the beginning of the run" is read as physical mass before scaling. - If it meant mass after first-step scaling, the statistic would be a/(1+a). - Verdicts would then differ only for a in (0.05, 0.0526]. - Add the sampled-maximum caveat and the strictness note. - Keep the refusal to let declared quasi-static intent shed the level.

### V — `added_mass_top_part_fraction`  < 0.10

- Statistic: One part only. p* is the part with the most added mass in absolute terms. The value is (maximum over the run of mass added to p*) / (physical mass of p* at the first sample). This is NOT the maximum of the ratio over all parts.
- Scope proposed: {explicit, mass_scaled}
- Sources: W-W179-07, W-R894-01 · match: identical
- Discretion remaining: None, provided the part is selected as the source selects it. A level for every part has no source. W-W179-15's third basis is per element, not per part, and carries no number.
- **Adversary: sound-with-edit.** - **(1) Evaluate at the last sample, not as a maximum over the run.** - Evaluate both the selection of p* and the value there. That is what "had ... added" says. - The proposer's maximum-over-run form equals it only when added mass is non-decreasing (L-C32), which no scope token records. - At the least, state the time at which p* is selected. The proposal leaves it unspecified. - **(2) Cite W-R894-01 as "claim text only; the verbatim quote is truncated before this row", or drop it.** - **(3) Quote W-W179-15 only as far as the dossier prints it ("...transmission and reflectio…").** Do not complete the word. - **(4) Disclose that "initial mass" is read as physical mass before scaling.** - **(5) Add the strictness note.** - The literal "one part only" reading is correct. Keep it.

### V — `added_mass_moving_fraction`  < 0.05

- Statistic: (Maximum over the run of mass added to the moving parts) / (physical mass of the moving parts at the first sample). 'Moving' means a part with non-zero initial kinetic energy.
- Scope proposed: {explicit, mass_scaled, initial_energy_driven}
- Sources: W-W179-08, W-W179-15, W-R894-01 · match: equal-under-condition
- Discretion remaining: The source does not define 'moving'. The rendering is taken from its stated rationale, not from the table row. The alternative reading, 'moves at any time', is degenerate after impact and was rejected. If the maintainer will not accept the rendering, this row is context only.
- **Adversary: sound-with-edit.** - **Keep the level, with the discretion stated as the proposer states it.** - Strengthen the anchor. - The row's own words "initial moving mass" fix the denominator as the mass moving at the start. - The rendering therefore does not rest only on the narrative rationale. - **Define "moving" from the input, not from the ledger.** - A part counts as moving if it has any node assigned a non-zero initial velocity. - Traits are input-derived, and this avoids a kinetic-energy threshold. - **Evaluate at the last sample, as for the top-part row.** - **Change `match` from "equal-under-condition" to "identical under a stated rendering of an undefined source term".** - The condition is a reading, not a scope token. - The maintainer must accept it explicitly. Otherwise the row is context-only, as the proposer says. - Cite W-R894-01 as claim text only. - Add the strictness note. - Disclose the physical-mass reading of "initial".

### V — Context only (no level)

- B-RAD-1, B-RAD-2 (Energy error: positive +1 % or +2 % is acceptable, above 2 % must be explained, and -10 % to -15 % is called normal.): The Radioss statistic is (E_kin + E_rot + E_int) / (the same sum at the start + external work) - 1. It omits zero-energy-mode and contact energy by construction, its denominator differs from r's, and the quote states no evaluation time. It coincides with r only for a ledger that has no other terms, and the scope vocabulary cannot express that with positive tokens. Its scope would also overlap B-BLM-1's {explicit} on `energy_gain_max`, which the disjointness rule forbids. It could be made its ow…
- B-ABQ-1 (Total energy 'generally with an error of less than 1%'.): It states no denominator and no evaluation time. It is descriptive ('generally'), not an acceptance statement.
- B-LSD-1, L-C12, B-LSD-4, B-LSD-5, L-C11 (The energy ratio total / (initial + external work), the statement that the balance should hold at all times, and the qualitative implicit checks.): These give definitions and direction-of-error prose only, and no tolerance is stated anywhere. The implicit checks use '≪' and '≈'. Use: a cross-check of the measure, and support for evaluating over the run. Note that r = ratio - 1 holds only while r's denominator takes its first argument.
- W-W179-12, W-W179-14 (Energy-change narrative ('e.g., say 5 or 10 percent') and practitioner survey bands.): The narrative's numbers are illustrative ('some threshold value (e.g., say ...)'), and it mixes energy with momentum. The survey reports the shares of respondents who picked each band, not a limit. Use: the denominator 'initial energy' for W-W179-02, and the community-practice provenance.
- B-BLM-3 (Hourglass energy over total energy; 'large (on the order of 3% or 5%)'.): It states an error estimate ('the results will be in error by the same order of magnitude'), not an acceptability limit. The evaluation time is unstated, the composition of 'total energy' is unstated, it gives two numbers under 'on the order of', and it says large problems should be monitored on subdomains. Use: a rationale sentence for every zero-energy-mode level.
- W-W179-13 (h_e / i_e <= lambda ~ 0.1, per element.): It is stated per element ('internal energy of the element'), is prefixed with 'say', and gives no evaluation time. E6 supplies per-part energies, not per-element energies. Use: provenance of the 10 % figure ('Du Bois, Paul, NOTES, 1998'), and the dilution warning for part-level ratios.
- B-ABQ-2 (Artificial strain energy at about 2 % of internal energy is 'not a problem'; at about 15 % it is 'substantial'.): These are two worked examples read off history plots. They state no rule, no limit, and no end-or-peak convention.
- W-ENCAP-01, W-ENCAP-02 (Per-dummy hourglass ratio; 2.5 % added mass for separate models; items monitored with no limit.): A dummy is a declared group of parts. 'Separate models' is a declared setup. Neither is a trait derived from the input or the ledger, so a dataset would be selecting its own level. The monitored items state no limit.
- B-LSD-3, L-C44 (Frictionless net contact energy; '10% of peak internal energy might be considered acceptable'.): The hedge on authority would not block a level. The statistic does. The quote leaves three things unstated: whole-model or per-interface (the dossier caveat says so, and sleout reports per contact), whose internal energy is meant, and the numerator's time and sign. The whole-model and per-interface readings are not nested. Interface energies of opposite sign can cancel, and several small ones can sum to more than 10 %. So neither verdict is sound under both readings. It would attach only to run…
- B-ABQ-3 (Quasi-static: kinetic energy of the deforming material 'typically 5% to 10%' of its internal energy 'throughout most of the process'; 'a few percent' in the forming example.): 'Throughout most of the process' fixes no evaluation time. The outermost reading, the maximum over the whole run, is degenerate because internal energy starts at zero. Any non-degenerate statistic requires the platform to choose the portion of the run to exclude. 'Deforming material' needs a rule for choosing parts. 'A few percent' is not a number. If a statistic were ever fixed, the least discretionary reading of the range is its lower end, 0.05, because `pass` would then hold under both figur…
- B-RAD-3 (Added mass 'less than 5%'.): The quoted sentence names no denominator and no evaluation time. The denominator DM/M0 appears only in the claim text and in a caveat. It corroborates the 5 % whole-model level in its rationale and is not cited as an attached source. Its quasi-static exception is also rationale text.
- B-BLM-4, B-LSD-7, L-C30, L-C09, L-C32, L-C33, W-W179-15 (Added mass: qualitative acceptability and reporting mechanics.): None of these gives a number ('if the mass increases remain insignificant'). L-C32 (added mass is non-decreasing) is the fact that makes the end-of-run and maximum readings coincide for the source's own solver. L-C09 and L-C33 say where the evidence is reported.
- W-W179-09 (No shooting nodes; no negative volumes.): These are yes/no checks with a definitional bound of zero. They belong to the requirement row `solver_error_count` (inversions; out-of-range kinematics), not to an indicator. 'Shooting nodes' has no operational definition in the quote.
- W-W179-10, W-W179-01, W-FHWA-01, W-RSVVP-01 (Mandatory status and footnoted exceptions; document identity; FHWA adoption; the RSVVP manual.): These are not statistics. W-W179-10 is rationale text: the source treats its rows as pass/fail with exceptions explained. The platform's `review` plus a disposition mirrors that.
- B-LSD-6, L-C36, L-C31 (Time-step collapse: DTMIN, ENDENG, ENDMAS, TSSFAC.): These are termination mechanisms and defaults. The manual 'recommends no values'.
- L-C39, L-C38, L-C40, L-C41, L-C42 (Particle-method settings.): These are settings and recommendations, such as a constant smoothing length for the moving-least-squares form. No neighbour-count or smoothing-length-saturation statistic or number is quoted.

### V — Indicators left without a level

- `contact_energy_ratio`: The only source (B-LSD-3 / L-C44) leaves the body, the denominator's body and the numerator's time unstated, and the possible readings are not nested. The row stays `ind` and reads `no_ratified_criterion`. The row's own definition must be fixed first.
- `contact_energy_negative_ratio`: No dossier quote gives a number for negative contact energy. The 'Negative contact energy' section was read and nothing quantitative was quoted.
- `quasi_static_kinetic_ratio`: B-ABQ-3 has no determinate evaluation time or part selection. B-LSD-7 gives no number ('very small relative to the peak internal energy').
- `timestep_min_ratio`: No published limit was found. B-LSD-6 and L-C36 describe a termination mechanism with no recommended value.
- `solver_warning_count`: No source was surveyed. F-C23 concerns message truncation, not an acceptable count.
- `particle_neighbors_min`: The dossier has no sourced number. The d + 1 floor is a definitional requirement, not an indicator level, and needs its own row.
- `particle_neighbors_growth`: No source.
- `smoothing_length_at_bound_fraction`: No source. L-C39's recommendation of a constant smoothing length removes the quantity (its gate is a variable smoothing length). It does not bound it.
- `rigid_surface_penetration_max (penalty enforcement, normalised by local spacing)`: No source.
- `input_density_plausible / input_constants_plausible / input_dimensionless_groups_plausible / response_magnitudes_plausible`: No materials reference was surveyed, and the dossier contains none.
- `energy_gain_max / energy_loss_max, implicit scope`: B-BLM-1 is an explicit-integration check. B-LSD-5's implicit checks carry no numbers.
- `energy_residual_final, externally driven scope`: W179 has no external-work term. B-BLM-1 is already judged through the gain and loss, and a second level here would overlap scopes.
- `zero-energy-mode and added-mass statistics judged for every part, end-over-end`: W179's rows concern only the part with the largest absolute hourglass energy or added mass. No source states an every-part end-of-run ratio.

### V — Levels the adversary said were missed

- **`total_energy_change_final`** = [E_tot(t_end) − E_tot(t0)] / E_tot(t0), signed. - Bound: abs <= 0.10. - Scope: `{explicit, lagrangian_mesh, initial_energy_driven}`. - Source: W-W179-02. The denominator comes from W-W179-12. - The proposer listed this only as a "strict-letter alternative". Under the attachment rule's clause "each statistic given a level is its own catalogue quantity", it is the correct home for the W179 level. `energy_residual_final` is then left unlevelled.
- **A second B-BLM-1 level for `energy_gain_max` and `energy_loss_max`, if the scope is restricted as the rule requires.** - Scope: `{explicit, particle_conservative}`. Same bound, 0.01. - It is disjoint from `{explicit, lagrangian_mesh}`. - A pairwise-conservative particle ledger is built from the work of its internal forces, so the statistic identity holds there. - The rationale must say that the source, a finite-element text, does not itself cover particle methods.
- **No other attachable level exists in the dossier.** - Every F-claim and the remaining L-claims were checked for limits. - They hold settings, defaults and reporting mechanics only. - Examples: DTMIN, ENDENG, ENDMAS, TSSFAC defaults; the smoothing-length recommendation for FORM=12. - No published number was found for time-step collapse, warning counts, neighbour counts, smoothing-length saturation, negative contact energy, or plausibility ranges. - Those rows correctly stay `ind (no level yet)`.

### V — Exclusions the adversary commented on

- **W-W179-12** - It is listed as context-only ("illustrative ... yields nothing"). Yet it is the sole evidence for the denominator of the 10 % energy level and is cited in that level's `source_claim_ids`. - The exclusion also drops two things that bear on the evaluation time: - The sentence "the maximum change in energy over the simulation run can be reported". - The dossier caveat that end-versus-start or maximum-over-run "is ambiguous in the source". - Mark it as load-bearing for the denominator, with the ambiguity disclosed.
- **B-RAD-1 / B-RAD-2** - The exclusion is right, because no evaluation time appears in the quote. - Strike the reason "its scope would also overlap B-BLM-1's {explicit} ... which the disjointness rule forbids". - Disjointness applies within one (quantity, statistic). - A Radioss statistic would be its own quantity, so there is no overlap.
- **B-ABQ-3** - The exclusion is right. "Throughout most of the process" fixes no evaluation time. - Strike the aside that pre-selects 0.05 as "the least discretionary reading of the range". - It fixes a number for a statistic nobody has defined. - That is a level decided outside the record.
- **B-LSD-3 / L-C44** - The exclusion is right. - It is consistent only if B-LSD-2 is excluded too. Both leave the numerator's evaluation time unstated, and the proposer's blocking rule was applied to one and waived for the other.
- **Dossier header finding 1** ("A ratified criterion has to name the variant") - It is not a source claim, but it was sidestepped by levelling both W179 per-part hourglass wordings. - Put it to the maintainer as a decision.
