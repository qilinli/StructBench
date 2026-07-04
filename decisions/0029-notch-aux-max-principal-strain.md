# 0029 — Notch-beam aux is max principal strain, not K&C damage

**Status**: Accepted
**Type**: Durable
**Date**: 2026-07-04

## Context

ADR-0026 chose the **K&C scaled damage measure** as the notch-beam
auxiliary field, on the reasoning that `MAT_CONCRETE_DAMAGE_REL3`
writes its scaled damage variable to the d3plot effective-plastic-strain
slot. This premise was empirically wrong.

An inspection of a representative case (`NB-B-320-Aa-8`, 2026-07-04)
showed:

| Field | Median | p90 | Max |
|-------|--------|-----|-----|
| `effective_plastic_strain` slot (labelled `damage`) | 1.78 | ~2.0 | ~2.0 |
| `strain` slot (6-component Voigt) — max principal | ~3e-4 | ~1e-2 | ~0.61 |

The `damage` slot is saturating (most particles at or near 2, the K&C
ceiling); it carries almost no spatial information about the crack
pattern. The `strain` slot is the actual field the prior study called
"strain": its distribution is clearly separated (elastic band ≈ 1e-4,
damaged particles up to 0.6), and Pearson correlation between the two
fields is 0.17. The maintainer confirmed that the prior-study extracts
are the max principal strain derived from the Voigt strain tensor.

## Decision

1. **Aux field for both notch benchmarks is `max_principal_strain`**, not
   `damage`. The extractor (`_aux_max_principal_strain`) builds the
   symmetric 3×3 strain tensor from the 6-component Voigt array
   (engineering shear components halved) and returns the largest
   eigenvalue via `np.linalg.eigvalsh`. Dimensionless; `stress_scale`
   is ignored.

2. **`cracked_fraction` QoI** replaces `damaged_fraction`. It computes
   the final-frame fraction of particles whose max principal strain
   exceeds a threshold. The **default threshold is 0.01** (1% principal
   strain), which sits clearly beyond the elastic band in the ingested
   data. This threshold is **provisional**: it has not been validated
   against the prior study's crack-pattern figures and may be revised
   before the first trained leaderboard entries. Any revision is a
   benchmark version change (ADR-0019 precedent).

3. **`"damage"` extractor remains registered** in `_AUX_EXTRACTORS` as
   an available option. It still reads the effective-plastic-strain slot
   correctly for material models where that slot genuinely carries
   damage; it is simply not the right choice for these cases.

4. **`"strain"` is added to both cards' `fields` tuple** to declare the
   Voigt strain array as a delivered field.

## Alternatives considered

- **Keep `damage` as the aux field.** The saturation problem means it
  cannot serve as a useful training target or leaderboard metric: a
  model predicting all-2.0 would score near-perfectly on aux RMSE
  while being useless.

- **Use effective plastic strain directly as the aux field (no
  renaming).** The slot is mislabelled; calling it `damage` in the
  registry was the source of the confusion. Removing the `damage`
  extractor outright would break code that legitimately uses it for
  other material models.

- **Validate the threshold against prior-study figures before
  committing.** Preferred — but the prior-study figures are not yet
  accessible. The 0.01 threshold is physically motivated (1% strain is
  a reasonable crack-initiation proxy for concrete) and the distribution
  data support it; the uncertainty is flagged explicitly in the
  docstring and here.

## Consequences

- Notch-benchmark training targets change (aux RMSE now measures strain
  rather than a saturated damage scalar). No leaderboard baseline had
  been trained yet, so there is no leaderboard impact.

- The `cracked_fraction` QoI threshold may require a version bump when
  validated; the docstring and this ADR both flag it as provisional.

- `decisions/0026-notch-beam-2d-benchmark-pair.md` is amended by this
  ADR (index updated). The text of ADR-0026 is kept for history.

- Cards and generated docs (`docs/benchmarks.md`) are regenerated.
