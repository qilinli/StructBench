# 0070 — Material class `elastic_plastic_isotropic` (Abaqus `*PLASTIC`, isotropic)

**Status**: Proposed
**Type**: Durable
**Date**: 2026-09-25

## Context

`verification/materials.py` (ADR-0066 clause 6, ADR-0067) holds solver-neutral
knowledge of each material class, keyed on ADR-0012's `canonical_model`. The
first Abaqus dataset (ADR-0068, ADR-0069) uses `*ELASTIC` + `*PLASTIC` with
isotropic hardening given as a table of (yield stress, equivalent plastic
strain), with no equation of state. No registered class describes it:
- `elastic_plastic_hydro` carries an EOS, which Abaqus `*PLASTIC` does not;
- leaving the material unclassified makes the monotonicity and yield rows
  report `unsupported` for every case.

## Decision

Register one class:

| Field | Value |
|---|---|
| `canonical_model` | `elastic_plastic_isotropic` |
| `state_variable` | `plastic_strain` (Abaqus PEEQ, the schema's `effective_plastic_strain`) |
| `state_bounds` | `(0, None)` |
| `monotone` | `True` |
| `yield_law` | `tabulated_j2` |
| `has_equation_of_state` | `False` |
| `structural` | `True` |

The Abaqus input reader assigns this class to a material whose `*PLASTIC` card
has no `HARDENING=` option or has `HARDENING=ISOTROPIC`, and which carries no
other inelastic card.

- **`monotone`, `state_bounds`.** Equivalent plastic strain under isotropic
  hardening accumulates and never decreases. Measured on a 2026-09-24
  sweep: no decrease in any case, and PEEQ ≥ 0 throughout.
- **`yield_law = tabulated_j2`.** The von Mises stress is bounded by the
  table's yield stress at the current PEEQ. A one-element check (2026-09-24,
  initial PEEQ 0.2, σ₀ = 100 MPa, H = 500 MPa) yielded at 199.7–200.0 MPa and
  then followed σ₀ + H·PEEQ exactly. Past the table's final knot the stress
  is taken as held at its last value: the reader stores a one-row table as a
  flat two-knot table, and verification interpolates with end-clamping. No
  run has yet been checked beyond a final knot, so that is an assumption of
  this instrument, not an observed fact about the solver.
- **The yield table is the case's own.** A sweep whose hardening varies from
  case to case declares no single curve, so the yield rows read the per-case
  table from the stored deck when the benchmark declares none. A declared
  table still takes precedence (ADR-0066 note, 2026-09-25).

## Alternatives considered

- **Reuse `elastic_plastic_hydro`.** Rejected: it declares an equation of
  state, and would gate EOS rows on a material that has none.
- **Leave it unclassified.** Rejected: the rows would read `unsupported`,
  blaming the platform for evidence the input fully states.
- **One class per Abaqus hardening option (isotropic, kinematic, combined)
  now.** Rejected: only isotropic is exercised. Combined hardening, with
  backstress state, is decided when a dataset first needs it.

## Consequences

- The Abaqus reader's `*PLASTIC` parsing (plan 2, Task 4) has a class to
  assign.
- `state_variable_min`, `state_variable_decrease_max`, `yield_ratio_max` and
  `yield_saturation_min` can apply to FE `solid` blocks carrying this class
  (plan 2, Task 7).
- Not decided here: combined or kinematic hardening classes; rate- or
  temperature-dependent tables.
