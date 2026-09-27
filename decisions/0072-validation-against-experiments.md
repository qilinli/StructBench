# 0072 — Validation against experiments: `structbench.validation`, reference sets, and the stage rename

**Status**: Proposed
**Type**: Durable
**Date**: 2026-09-27
**Relates to**: ADR-0065 (the V&V identity), ADR-0066 (verification of reference runs), ADR-0071 (the data-generation pipeline; its `validate` stage is renamed here)

## Context

ADR-0065 names the platform's purpose as verification *and* validation of
learned surrogates, and ADR-0066 built verification: deterministic checks
that judge a reference run from its own evidence. Validation — whether a
benchmark's reference setup reproduces a physical measurement — has no home
in the repository. The first Abaqus dataset performed it privately in
2026-09-25: the measured outlines of five copper Taylor tests were extracted
exactly from the vector figures of an open-access compilation, the dataset's
own setup was run at the tests' conditions with a literature copper model,
and final length, largest radius and lateral profile were compared, with
mesh, contact, increment and rate dependence varied to show what the
comparison is sensitive to. That work sits in a script bound to one
dataset's case ids, and the reference measurements are not public.

The word is also taken: the data-generation stage `structbench-datagen
validate` runs the verification instrument. The maintainer chose (in
session, 2026-09-27) to rename that stage rather than give validation a
second name. The design is
`docs/plans/2026-09-27-validation-against-experiments-design.md`.

## Decision

1. **A `structbench.validation` package**, beside `verification`, holds
   reference-experiment sets as package data, measures that apply the same
   function to a measured outline and to a canonical case, a comparison that
   reports deviations, and a byte-stable record with a Markdown rendering,
   behind one command, `structbench-validate`. It depends on `core` only.
2. **Validation reports and does not judge.** The record states each
   test's measured and simulated values, the relative deviations, and their
   minimum, median and maximum per variant; no acceptance level is ratified,
   for the reason ADR-0066 gives for reference levels: a tolerance chosen to
   pass our own runs would be fitted to the test bed. A person writes the
   sentence, from the numbers.
3. **A reference set carries its provenance**: sources with DOIs, whether
   each was consulted directly or taken as another source reports it, the
   licence under which redistributed curves travel, the extraction tool and
   its checks, and caveats. The first set, `taylor_copper`, is the six tests
   compiled by Zelepugin, Cherepanov & Pakhnutova (2023, CC BY 4.0), with
   outlines read from the paper's vector drawings by a digitiser kept in
   `tools/validation/`. The source's own simulations are not redistributed,
   and no PDF enters the repository.
4. **The measures are the source's** — final length, largest radius and the
   lateral radius at stated fractions of the final length — computed from an
   outline; the simulation's outline is the deformed boundary of the
   canonical mesh at the last frame, so the measures work on any
   axisymmetric case, including a surrogate's prediction.
5. **The data-generation stage `validate` is renamed `verify`**, its module
   with it, and the old name is refused with a pointer. After this, *verify*
   means the ADR-0066 instrument and *validate* means comparison with
   experiment, in code, commands and prose. ADR-0071 is amended by a dated
   note.
6. **A dataset's validation record stays with the dataset until admission**,
   produced by the public command from a `pairs.toml` and a `setup.toml`
   in its own folder. The card field and landing-page link for a public
   record come with the first admitted one.

## Alternatives considered

- **Keep the private script and publish only a table.** Rejected: a table
  without the measurements and the measure functions cannot be checked or
  reused, and the next benchmark would write the script again.
- **Put validation inside `verification`.** Rejected: ADR-0065's vocabulary
  separates them for a reason, and a reader looking for "did the setup match
  the experiment" should not have to look under "is the run internally
  sound".
- **Keep the stage name `validate` and call the new command something
  else.** Rejected by the maintainer: two meanings for one word in one
  tool.
- **Ratify a tolerance (say 5 % in final length).** Rejected, as in ADR-0066
  clause 7: the only data to choose it from are our own runs.
- **Redistribute the source's simulated curves as context.** Rejected: they
  are not measurements, and the comparison the record makes is with
  experiment.

## Consequences

- The repository gains public reference measurements with provenance, and a
  second use of the canonical case schema: an experiment's outline and a
  simulation's are measured alike.
- Every `structbench-datagen validate` in docs, recipes and the ADR-0071
  design becomes `verify`; the private dataset's recipe changes in the same
  pass.
- The first Abaqus dataset's validation record is regenerated with the
  public command from re-run probe cases (its earlier exports were deleted
  on 2026-09-27); differences from the earlier private README beyond the
  stated length band are explained there.
- The LS-DYNA Taylor benchmark could carry a public record once runs at the
  experiments' conditions exist; that is the maintainer's decision and not
  part of this ADR.
