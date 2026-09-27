# Validation against experiments — design

*Design for ADR-0072 (Proposed). Written 2026-09-27; the maintainer approved
the shape in session before this document was written.*

## Problem

ADR-0065 makes StructBench a verification-and-validation platform, and
ADR-0066 built the verification half: an instrument that judges whether a
reference *run* is trustworthy from its own evidence. Nothing in the
repository yet says whether a benchmark's reference *setup* — solver,
element, mesh, contact, increment, the class of material law — reproduces
what was measured in a physical experiment. The first Abaqus dataset did
that check privately: measured copper Taylor tests were extracted exactly
from a source's vector figures, the dataset's setup was run at their
conditions with a literature copper model, and the final length, largest
radius and lateral profile were compared. The check lives in a private
script tied to that dataset's run folders and case ids, so no other
benchmark can use it, and the reference measurements exist nowhere public.

Two words are also crossed: the data-generation stage called `validate` runs
the verification instrument.

## Principles

- **Validation is comparison with experiment; verification is the
  instrument.** The stage is renamed `verify`; `validate` is used for nothing
  else.
- **Both sides are measured by the same function.** The measures take an
  outline; the experiment's outline comes from the reference set and the
  simulation's from the canonical case, and neither is read by eye.
- **Deviations are reported, never judged.** As with ADR-0066's reference
  levels, no acceptance threshold is ratified; the record states the numbers
  and their spread, and a person writes the sentence.
- **Provenance travels with the numbers.** A reference set carries its
  sources with DOIs, the licence under which its curves are redistributed,
  the extraction method and its checks, and what was not consulted directly.
- **Nothing of a private dataset enters the repository.** The reference set
  is public science; a dataset's validation record stays with the dataset
  until admission.

## The reference set

One JSON file per set under `structbench/validation/references/`, format
`validation-reference/1`, with a Markdown companion for readers:

| Field | Content |
|---|---|
| `format`, `name`, `family`, `title` | `validation-reference/1`; `taylor_copper`; `taylor_rod`; one line |
| `units` | `length: mm`, `velocity: m/s`, `temperature: K` |
| `sources[]` | `id`, `citation`, `doi`, `licence` (where known), `role`, `consulted` (true if read directly, false if taken as another source reports it) |
| `extraction` | `tool` (the repository path of the digitiser), `method`, `checks` (the tick-map residual, the two-figure agreement), `date` |
| `measures` | the fractions for W_f and the definitions of L_f, R_f, W_f |
| `tests[]` | `id`, `source`, `via`, `material`, `L0_mm`, `D0_mm`, `v0_ms`, `T0_K`, `outline_rz_mm` (the measured half profile, z from the impact face), `Lf_mm`, `Rf_mm`, `Wf_mm[]` |
| `caveats[]` | plain sentences |

`taylor_copper` holds the six tests of Zelepugin, Cherepanov & Pakhnutova
(2023, Materials 16, 5452, CC BY 4.0), whose Table 1 gives the conditions and
whose Figures 4 and 5 draw the measured outlines as vector paths: test 1 from
Wilkins & Guinan (1973) and tests 3–6 from Zelepugin et al. (2022, Metals 12,
2186), both as the 2023 paper reports them; test 2 (ETP copper at 718 K, Gust
1982) is included with its conditions so the set is complete, and a
comparison chooses the tests it uses. The source's own simulated curves are
not published: they are context, not experiment. The PDF is not in the
repository. The values `Lf_mm`, `Rf_mm`, `Wf_mm` in the file are what the
public measure functions return on `outline_rz_mm`, and a test holds them to
that.

The digitiser (`tools/validation/digitize_zelepugin2023.py`) takes the PDF's
path as its argument, reads the black polylines from the figure form
XObjects, maps them to length with a least-squares fit to the tick marks, and
checks that each test's curve is identical in both figures; it writes the
reference JSON. It is the provenance of the numbers and is not run by the
test suite.

## Measures (`structbench.validation.measures.taylor`)

- `outline(case, frame=-1)`: the closed boundary of an axisymmetric quad mesh
  at a frame — edges that belong to exactly one element, chained into a loop,
  deformed by the frame's displacement and shifted so the impact face sits at
  z = 0. Nodes no element uses are ignored.
- `final_length(outline)` = max z; `largest_radius(outline)` = max r;
  `lateral_radii(outline, Lf, fractions)` = the largest r where the outline
  crosses the height f·L_f, for each fraction. The reference set's fractions
  are the source's: 0.2, 0.25, 1/3, 1/2, 2/3.
- `taylor_outcome(case)`: the three measures at the last frame, plus the
  half peak-to-peak band of the length over the last quarter of the stored
  frames (the elastic vibration after the rod has left the wall), reported
  with the length and never subtracted from it.

L_f is taken at the last frame, the same instant as the datasets' QoI
`final_length`; the earlier private script averaged over the last 100 µs,
and the two differ by less than the band.

## Comparison (`structbench.validation.compare`)

Input: a reference set and a list of pairs. A pair names a test, a variant
key, and either a canonical case file or a status (`aborted`, with a reason)
for a run that did not complete. Variants are the setups compared side by
side (the dataset's mesh, a coarser mesh, a rate-free law, a fitted law),
each with a label.

Output, per pair: the measured and simulated L_f, R_f and W_f, the relative
deviation of L_f and R_f, the RMS relative deviation of W_f over the
fractions, the length band, the case id, and the status. Per variant and
measure: minimum, median and maximum deviation and the number of tests. A
missing case file is `missing`, never silently skipped.

## Record and command

`structbench.validation.report` writes `validation-record/1`: the reference's
name and sha256, the setup (a table of strings the dataset supplies — solver,
element, mesh, wall, increment, elastic constants, material — and its
caveats), the variants, the results and the summary, sorted keys, no
timestamps, so a regeneration is byte-identical. `render_markdown` produces
the reader's document: sources with DOIs, the tests used, one results table
per measure with deviations in brackets, the summary, the setup, the
caveats. A data-free test keeps a committed `.md` equal to what its `.json`
renders, as the datachecks do.

Command: `structbench-validate --reference taylor_copper --pairs pairs.toml
--setup setup.toml --out <dir>` writes `<dir>/<name>.json` and `.md`;
`structbench-validate --list` names the shipped reference sets and their
tests. `pairs.toml`:

```toml
reference = "taylor_copper"

[variants]
variant_a = "<label: the setup at the production mesh>"
variant_b = "<label: the same setup at a coarser mesh>"

[[pair]]
variant = "variant_a"
test = "3"
case = "<canonical case>.h5"      # relative to this file

[[pair]]
variant = "variant_a"
test = "6"
status = "aborted"
reason = "<why the run did not complete>"
```

`setup.toml` is a `[setup]` table of strings and a `caveats` list.

## The rename

`structbench/datagen/validate.py` becomes `verify.py`; the stage is
`structbench-datagen verify`; the old name is refused with one line naming
the new one. Tests, `docs/DATA_GENERATION.md`, `docs/datagen/
abaqus-conformance.md` (whose command was already stale), `docs/ARCHITECTURE.md`,
the design document of ADR-0071 and `CLAUDE.md` follow; ADR-0071 gets a dated
note. The private recipe is updated in the same pass.

## Package layout and layering

```
structbench/validation/
    __init__.py
    references/taylor_copper.json, taylor_copper.md      package data
    measures/__init__.py, taylor.py
    compare.py
    report.py
    cli.py                                                structbench-validate
tools/validation/digitize_zelepugin2023.py
```

`validation` depends on `core` (the case schema and readers) and nothing
else in the package; `cli` sits above it. It joins the layering beside
`verification`: `core ← datasets ← {verification, validation} ← {eval,
benchmarks, datagen} ← cli`. No torch, no scipy.

## Migration of the private record

The dataset's `compare.py` retires. Its validation folder keeps `pairs.toml`
and `setup.toml`, and the record is regenerated with `structbench-validate`
from canonical cases. Because the probe runs' exports were deleted on
2026-09-27, the cases the record uses are re-run from their decks
(`generate → run → export → convert`, byte-identical decks by provenance),
which also exercises the pipeline end to end under the new stage name. The
regenerated numbers are compared with the earlier README's and any
difference beyond the length band is explained there.

## Testing

- The reference JSON: format and required fields; every test's `Lf_mm`,
  `Rf_mm`, `Wf_mm` equal the measure functions on its outline; every source
  has a DOI; the licence of the redistributed curves is stated; the Markdown
  companion names every source.
- Measures: a synthetic axisymmetric case whose last frame is a known
  deformed outline (a rectangle with a flared foot) gives L_f, R_f and W_f
  exactly; nodes no element uses are ignored; the band is zero for a frozen
  tail and equals the imposed oscillation otherwise.
- Comparison: deviation arithmetic on synthetic pairs; `aborted` and
  `missing` appear in the results with no numbers; the summary's n counts
  completed pairs only.
- Record: byte-identical on regeneration; `render_markdown(from_json(x)) ==
  render_markdown(record)`; the CLI writes both files and `--list` names
  `taylor_copper`.
- The rename: `structbench-datagen verify --help` works; `validate` exits 2
  with the pointer; no `validate.py` remains under `datagen`.

## Out of scope

Acceptance levels or verdicts; a `BenchmarkCard` field and a landing-page
link (they come with the first public record); other reference sets (the
format is ready); validation of the LS-DYNA Taylor benchmark (it needs
LS-DYNA runs at the experiments' conditions — the maintainer's call);
validating a surrogate's predictions against experiment, which the measures
already allow because they take a `Case`.

## Open points

- Whether `taylor_copper.md` should also carry the outlines as a figure; the
  first version does not (no matplotlib dependency in the package).
- The reference-set format is `/1`; a set with time-resolved measurements
  (rear-surface velocity) would need a second family and possibly a `/2`.
