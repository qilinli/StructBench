# 0067 — Material classes for the notch sweep: K&C concrete and bilinear steel

**Status**: Accepted (maintainer, in-session 2026-09-23); built 2026-09-23
**Type**: Durable
**Date**: 2026-09-23

## Context

`verification/materials.py` (ADR-0066 clause 6) holds the platform's material
knowledge that is solver-neutral: keyed on ADR-0012's `canonical_model`, each
entry says what a class's stored internal variable *means*, how it is bounded,
whether it can ever decrease, and which yield law can be assessed from what is
stored. The same stored field is plastic strain for one class and a damage
measure for another; only the class can say which.

It carries three classes today — `elastic_plastic_hydro` (Taylor's copper),
plus `null` and `rigid`, which carry no load. **One structural class in the
whole platform.** The module's own docstring defers the rest: *"A class is
added when the first dataset that uses it arrives, with the material-class ADR
that ADR-0012 anticipates."* This is that ADR.

The notch-impact sweep's two materials have no class, so ten of its rows read
`not_assessable / unsupported`. That is honest but imprecise: it reports a gap
in the platform where, for five of those rows, the sharper truth is a fact
about the data.

Two things narrow this decision to its real content. First, the **card
layouts** landed separately on 2026-09-23 with no ADR, on the principle that
reading a layout says what numbers a card holds while only a class says what
they mean: `*MAT_CONCRETE_DAMAGE_REL3` and `*MAT_PLASTIC_KINEMATIC` already
give up their density, Poisson ratio and (for the steel) Young's modulus, and
`canonical_model` stays `None`. Second, verification keys on
`facts.materials`, parsed fresh from the stored deck at measure time, in both
`gate()` and `class_mask`. The archive's stale `canonical_model` is irrelevant
to it, **so adding a class needs no re-conversion of the 110 case files.**

What remained was meaning, and it was measured rather than recalled. Evidence
below is from this repository's own data, per CORRECTIONS 2026-09-21: the
archive is a test bed for the instrument, and these classes are stated as
platform standards that future contributions are measured against, not shaped
around what these files happen to contain.

## Decision

Add two classes to `MATERIAL_CLASSES`, and the two keyword mappings to
`_CANONICAL_MAT` in `core/io/lsdyna.py` (stripping the `_TITLE` option, which
the deck carries and the verification-side parser already handles):

| field | `concrete_damage` | `elastic_plastic_kinematic` |
|---|---|---|
| `state_variable` | `damage` | `plastic_strain` |
| `state_bounds` | `(0.0, 2.0)` | `(0.0, None)` |
| `monotone` | `True` | `True` |
| `yield_law` | `not_assessable` | `not_assessable` |
| `has_equation_of_state` | `False` | `False` |
| `structural` | `True` | `True` |

**`state_variable` and `state_bounds`.** For `*MAT_CONCRETE_DAMAGE_REL3` the
d3plot effective-plastic-strain slot records the K&C scaled damage measure,
unitless, on `0..2` — already recorded in `datasets/canonical.py`,
`benchmarks/render.py` and ADR-0026, and measured again here. Across 22 notch
cases sampled every fifth across the grid, the concrete's stored scalar spans
exactly `[0.000000, 2.000000]`.

**`monotone`.** Measured, not assumed: over those same 22 cases — roughly
66 million per-particle-per-step samples — the worst per-step decrease is
exactly `0.000e+00` and the number of decreasing samples is **zero**.

**`yield_law = not_assessable` for the concrete**, on two independent grounds.

*Empirically, von Mises stress is not a function of the stored scalar.*
Binning one case's 6,136 concrete particles over all 502 frames:

```
  damage bin         n   vM min   vM med    vM max     p min     p max
[0.00,0.01)    266397     0.00     0.00    122.19      -2.5     155.8
[0.01,0.50)    154027     0.02    10.30    395.93      -5.7     486.0
[0.50,1.00)     77100     0.02     5.14    429.29      -5.4     431.3
[1.00,1.50)    115621     0.03     4.64    456.36      -5.2     462.4
[1.50,1.90)    601460     0.02     7.50    445.51      -5.8     486.0
[1.90,2.00)   1865667     0.00     1.84    413.24      -5.6     422.5
```

At every damage level von Mises spans zero to roughly 400 MPa or more. Fully
damaged concrete still reaches 413 MPa in a C50 mix whose unconfined strength
is 50 MPa, because strength tracks **confinement**, not damage: in that band
pressure reaches 422 MPa. Confirmed on three further cases, where fully
damaged von Mises reaches 500, 340 and 274 MPa against pressures of 607, 301
and 220 MPa. No tabulated function of the state variable can bound this;
concrete's yield is a surface in pressure, not a curve in plastic strain.

*And the surface's coefficients are not in the input.* The deck reads
`a0 = -0.05, a1 = 0.0, a2 = 0.0`: a negative `a0` directs the solver to
generate the surface internally from the unconfined compressive strength
(0.05 GPa = 50 MPa, the C50 mix), and `a1`, `a2` are products of that
generation rather than inputs to it. The numbers that define the surface never
appear in the input the instrument reads, and the stored `0..2` scalar is a
remapped damage measure rather than the internal variable the surface
consumes. `not_assessable` is exactly the enum value for this: a yield surface
exists, and its arguments are not exported.

**`yield_law = not_assessable` for the steel** for a narrower reason: its law
is bilinear, stated by `sigy` and `etan` rather than by knots, so there is no
tabulated curve to compare against and `yield_table` stays `None` (its
contract is knots a deck states verbatim; two would have to be invented).

**`has_equation_of_state = False`** for both: the notch deck carries no
`*EOS_*` card, only the two `*MAT_` cards. Contrast Taylor, whose
`elastic_plastic_hydro` is paired with `*EOS_GRUNEISEN`.

## Alternatives considered

- **Leave both unsupported (status quo).** Honest, and it was the right
  default until a dataset arrived. Rejected now because two rows are
  measurable today and are not being measured, and because for five more the
  report blames the platform where the truth is a specific fact about the
  data. An instrument that cannot distinguish "we have no class for this" from
  "this material's surface cannot be evaluated from what was stored" is less
  useful than one that can.

- **Declare `yield_law = tabulated_j2` and synthesise a curve from f'c.**
  Rejected on the evidence above: no curve in the state variable bounds the
  von Mises stress, so the check would fail every concrete case for a reason
  that is the instrument's error, not the data's. This is the failure mode
  worth most avoiding — a confident false verdict reads as assurance.

- **Add a pressure-dependent yield law and evaluate the surface.** The stored
  fields do carry pressure, so the shape of such a check is imaginable.
  Rejected for now because the surface's coefficients are auto-generated and
  absent from the input, and the internal damage variable the surface
  interpolates on is not stored. Revisit if a future deck states `a0`, `a1`,
  `a2` explicitly — which the standard input block could require — and if the
  internal variable is exported through `NEIPH`.

- **Put the steel's `sigy` into `yield_table` as two knots.** Rejected:
  `yield_table` holds knots a deck states verbatim, and a bilinear law states
  none. It would also feed rows that compare against a declared hardening
  curve, which notch does not have.

- **Add the classes without an ADR, as a patch.** Rejected. Three of the five
  fields are claims about physics that the instrument then *enforces*; a wrong
  `monotone` alone would fail every notch case. `materials.py` names the ADR
  requirement for exactly this reason.

## Consequences

- Two rows become measurable on notch, on concrete particles:
  `state_variable_min` and `state_variable_decrease_max`. Both are already
  implemented and already pass on Taylor, so no new measure is needed.
- `eos_closure` becomes `not_applicable` rather than `not_assessable /
  unsupported` — more accurate, since neither material has an equation of
  state.
- The five yield rows keep no verdict, but their reason changes from "the
  platform has no class for this material" to a true statement about the data.
- Both published records must be re-measured. Taylor is unaffected in
  substance (its class is untouched) but should be re-measured to confirm it.
- No re-conversion of the canonical archives is required, per the Context.
- Run traits do not change: `_load_carrying_kinds` already includes a part
  whose class is unknown, so notch's traits stay `{dim2, explicit,
  externally_driven}` and no existing verdict shifts.
- **Revisability, and the main risk.** `monotone=True` rests on 22 cases of
  one sweep under impact loading. Load reversal or a different K&C
  configuration could in principle produce a decrease, which would then read
  as a `fail` in the stored response when the fault is this claim. The
  evidence is strong (zero decreases in ~66 million samples) but it is one
  sweep; if a future dataset contradicts it, the honest response is a dated
  note narrowing the claim, not a tolerance bolted onto the check.
- **Left open, deliberately.** The steel card carries `e = 200.0` beside
  `sigy = 337.0` in a unit system whose stress unit is 1 GPa — a first-yield
  strain of 1.7. `input_dimensionless_groups_plausible` is the row built to
  see this and cannot, because it reads `yield_table`. A scalar `yield_stress`
  on `MaterialInput` would close it and would change what two units rows
  measure, so it is a separate decision. Those parts are protocol-kinematic
  (ADR-0026) — driven by ground truth, excluded from loss and metrics — so
  nothing a user trains on depends on it.
- This ADR discharges none of ADR-0065's four follow-ups, and does not touch
  the 13 notch rows that read `source_missing`: no material class reaches the
  ledger, the time-step history or the load resultants those runs never wrote.

## Build note (2026-09-23)

Built as decided. `MATERIAL_CLASSES` gains the two classes;
`_CANONICAL_MAT` gains the two keyword mappings; and the option-stripping
the Decision called for became `canonical_model_for()`, shared by the
converter and the verification reader, because the converter was looking up
the verbatim source-model name and a `_TITLE` option was enough to hide a
material the platform knows.

**The Consequences section was slightly wrong, and the build showed it.** It
predicted the five yield rows would "keep no verdict, but their reason
changes". They did not: the trait gate asks whether any class offers
`tabulated_j2`, and a class that offers `not_assessable` answered the same
as a class with no yield surface at all, so the rows came back
`not_applicable` — reading, to anyone scanning the report, as though yield
admissibility were irrelevant to a concrete impact benchmark. It is not
irrelevant; it is unevaluable, which is a different claim and the honest one.

`MaterialClass` already draws that distinction (`"none"` = no surface;
`"not_assessable"` = a surface whose arguments are not exported) and the
gate was collapsing it. The gate now separates them, and the rows read
`not_assessable / not_available_from_solver` — the reason whose own
definition is "closed by nobody, for this solver", which is exactly the
situation. A material with no surface at all, such as `rigid`, still reads
`not_applicable`. Both are under test.

This is the same misreport this session found twice before — a
`not_applicable` asserting something about the data that was not true — and
it is worth stating as a rule rather than a third coincidence: **`not_applicable`
is a claim, and a claim has to be checked like any other.**

Effect on the published records. Notch: eight rows move off
`not_assessable / unsupported` — `state_variable_min` and
`state_variable_decrease_max` to **pass**, `eos_closure` to
`not_applicable`, and the five yield rows to
`not_assessable / not_available_from_solver`. The two passes are the
monotonicity claim of this ADR holding on all 110 cases rather than on the
22 it was measured from. Taylor is re-measured and byte-identical: its class
was untouched, and its yield law is tabulated, so the gate change does not
reach it.
