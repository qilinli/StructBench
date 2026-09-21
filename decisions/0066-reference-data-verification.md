# 0066 — Reference-data verification: the `verification/` module

**Status**: Proposed
**Type**: Durable
**Date**: 2026-09-21

## Context

ADR-0065 gives StructBench "the data-standard side of the data layer" and
records that none of the four shipped benchmarks would be admitted under the
data standards it names, a gap to be made visible rather than hidden. Today
nothing in the repository can measure whether a reference simulation run is
trustworthy:

- `core/validation.py` checks structure and shapes only — no NaN scan, no
  physics, no non-fatal finding channel.
- The constitutive checks on reference data (plane strain, von Mises
  identity, yield surface, plastic-strain monotonicity) exist only as
  `assert`s in `tools/state_probe/`, which is explicitly exploratory, and
  their outcomes are prose in its README. The model-side admissibility
  diagnostics ADR-0064 names (D2/D3) live in untracked `scratch/`.
- Provenance is `("LS-DYNA", "unknown", <d3plot mtime>)` on every ingested
  case. No reader exists for any solver output other than d3plot.

Two facts about the evidence shape the design. **Public evidence is richer
than it looks**: every canonical `<case_id>.h5` already stores the input deck
verbatim (`metadata/source_deck`), the three global energies, and the SPH
fields a constitutive or particle-health check needs, so a large part of an
audit is reproducible from the public archives alone. **Private evidence is
uneven**: the maintainer-held run folders all keep the solver message files,
but only the Taylor runs wrote a global-statistics file — the wave-1D and
notch decks never requested one, so their energy decomposition is not
assessable without a re-run. A first measurement on one Taylor run shows why
this matters: normal termination, no mass scaling, zero external work, and a
total-to-initial energy ratio that peaks at 1.089 and ends at 1.070 — net
energy creation in a closed system, recorded nowhere.

The design was reviewed in-session (2026-09-21) as three independent module
designs attacked by three adversarial reviews (repository governance,
computational mechanics, buildability). The reviews changed the design
materially; the rejected alternatives below record what they overturned.

Binding constraints: ADR-0004 (no solver vocabulary outside adapters),
ADR-0010 (a solver abstraction layer was rejected as premature — "needs at
least two concrete solvers to be designed correctly"), ADR-0016 §4/§6 (no
feature engineering at ingestion; glue never manipulates response data),
ADR-0027 (per-benchmark declarations are typed dataclasses; derivable numbers
are computed, not declared), ADR-0031 (the archive content list is closed),
ADR-0033 (results never bump a benchmark version), ADR-0064 (one
authoritative hardening table), the dependency graph in
`docs/ARCHITECTURE.md`, and the active correction of 2026-07-03 (prefer the
smallest artifact that encodes the behaviour).

## Decision

1. **A new top-level module `src/structbench/verification/`** holds the
   deterministic checks: array-level kernels, material-class semantics, the
   quantity catalogue, `measure`, the criteria with `judge`, and the report.
   It sits between `datasets/` and the peer modules:

   ```
   core ← datasets ← verification ← {eval, benchmarks, models, viz} ← cli
   ```

   It imports `core` and `datasets` only. No code moves; `datasets` gains one
   export (`n_valid_frames`). The name follows ADR-0065 clause 4
   ("verification is reported before accuracy is compared") and leaves
   *validation* free for the reality layer; it is unrelated to
   `core/validation.py`, which stays the schema-validity check. The kernels
   are written on plain arrays (e.g. `(von_mises, yield_stress)`), so `eval/`
   can later call the same functions on predicted fields with no new edge.

2. **Evidence is a record type plus one concrete extractor — not a solver
   abstraction.** `core/evidence.py` defines small solver-neutral, SI-valued
   records (`Absence`, `PartTraits`, `DeckFacts`, `RunEvidence`), re-exported
   through `structbench.core`. `core/io/lsdyna_run.py` is the single
   extractor. There is no protocol, registry, or plugin seam (ADR-0010
   stands), records carry only fields a shipped check consumes, and LS-DYNA
   vocabulary stays inside `core/io/lsdyna*.py`. Three evidence tiers exist,
   and every measurement states which one produced it:

   | Tier | Source | Availability |
   |---|---|---|
   | `case` | canonical `.h5` response and globals | public |
   | `deck` | `metadata/source_deck` in the same file | public |
   | `run` | solver message and global-statistics files | maintainer-held |

   Parsers take *text*, never directories. Per-dataset glue under
   `data_generation/` builds explicit paths from case ids (no globbing),
   calls the extractor, and writes one whitelisted `run_evidence.json`; the
   library and CLI never receive a run directory. Evidence records have no
   free-text field, so licence numbers, host names, and local paths cannot
   reach an artefact. Deck facts are read by a new faithful, **fail-closed**
   card reader (blank data lines and fixed field positions preserved;
   `*INCLUDE`, `*PARAMETER`, `&` references, and free-format rows yield
   `unparsable`); the existing `_card_blocks` drops blank and non-numeric
   rows and stays private to material ingestion. Run-tier measurement works
   with no `Case`, so a run that never became a case can still be measured.

3. **Measuring and judging are separate functions.** `measure` turns
   evidence into threshold-free measurements (quantity, value, unit, tier,
   sample count, detail) or a typed absence. `judge` turns measurements plus
   criteria into verdicts and never touches data. Verdicts are therefore a
   pure function of committed measurements and tracked criteria, and no
   command-line flag alters a criterion.

4. **Four verdicts, one owner each, evaluated in a fixed order.** Every
   catalogue quantity answers for every case.

   1. Formulation traits own `not_applicable` — the quantity does not exist
      for this formulation (hourglass energy on an SPH part).
   2. Evidence absence owns `not_assessable`, always with a typed reason:
      `not_requested`, `not_computed`, `not_ingested`, `source_missing`,
      `unparsable`, `source_unreadable`, `unsupported`; and, from `judge`,
      `no_ratified_criterion`.
   3. Measurement against a criterion owns `pass` and `fail`.

   The test between the middle two is whether better evidence could change
   the answer. Traits are evaluated first, and unknown traits yield
   `not_assessable`, never `not_applicable`. A check that has not been built
   is a library gap listed in documentation; it is never rendered as a
   verdict about a dataset.

5. **Traits are per part and deck-derived; declarations are claims.** A case
   may carry several materials, so the extractor reads the part-to-material
   map from the deck. A canonical file may also carry element blocks that
   have no part in the deck — the Taylor files hold a four-node shell that
   is visualisation geometry written into the d3plot, while the deck defines
   a single SPH part. Such blocks are non-structural: they are excluded from
   every check and counted by an integrity row, so they cannot make an SPH
   case read as "coupled". Traits are therefore never inferred from a case
   file's element keys. Card declarations
   (`discretisation`, `erosion`, `fields`) are cross-checked against derived
   traits and are the fallback only for deck-less data. Material knowledge
   is two tables: *solver keyword → material class* in `core/io`, and *class
   → meaning of the internal-variable slot, its bounds, and which yield law
   is assessable* in `verification/`. `BenchmarkSpec.hardening_curve` stays
   the only operative yield table (ADR-0064); the deck's table is read only
   to verify that the two agree.

6. **Criteria are the platform's standard, of two kinds that are never
   mixed.** *Acceptance criteria* (energy balance, added mass, hourglass
   ratio, time-step collapse) are set a priori from an external source the
   maintainer has verified, and are committed before the cases they judge
   are measured; until ratified the verdict is
   `not_assessable / no_ratified_criterion`. *Instrument tolerances*
   (float32 closure sums, discretised-return yield excess, end-time match)
   derive from storage precision or a named algorithmic mechanism, may be
   calibrated from measurement, and are flagged `provisional` until pinned.
   Every criterion is a record with a required, rendered rationale.
   Benchmarks cannot loosen a criterion; they may declare non-derivable
   facts and, when first needed, dated waivers that acknowledge a known
   deviation. A grandfathered benchmark may visibly fail. Gates, severities,
   and scores are not introduced without a further ADR.

7. **Measurements are the committed record; verdicts are generated.** The
   JSON report carries a schema id, the package version, a per-quantity
   `definition_version`, the SHA-256 of each measured file, and values
   rounded to fixed significant digits; it carries no timestamp, host, or
   path, and a re-run is byte-identical. Working output goes to gitignored
   `runs/`. The published record is `docs/datachecks/<benchmark>.json` with a
   generated `.md`, pinned by a data-free test that the markdown equals
   `render(judge(json, criteria))`. A stale `definition_version` renders as
   "stale — re-measure" for that row only. The CLI exits `0` on completion
   and `2` on usage or I/O errors; failing verdicts are reported in the
   output, not through the exit code, and an unreadable input becomes a
   recorded row rather than an abort. Changing a criterion never bumps a
   benchmark version (the ADR-0033 rule); removing a case because of a
   verdict is a split change and does (ADR-0019).

8. **Scope.** This ADR discharges none of ADR-0065's four follow-ups. It
   supplies the measurements follow-up 2's compliance table will render and
   the kernels follow-up 1's properties block will call. Benchmark cards,
   results registries, `render.py`, the generated benchmark pages, the case
   schema, and the archive contents are untouched.

## Alternatives considered

- **No new module — measures in `datasets/`, verdicts and catalogue in
  `benchmarks/`** (the `timeline.py` precedent). Rejected.
  `benchmarks/registry.py` imports `eval`, so `eval` could never import the
  result types; `datasets/` is chartered for data loading; a run with no
  registered benchmark and no case could not be judged; and the catalogue is
  several times `timeline.py`'s size. The smaller footprint buys a forced
  public-API move later.
- **Host the checks in `eval/`.** Rejected: everything in `eval/` scores a
  model's prediction against ground truth, whereas these checks judge the
  ground truth itself, and `benchmarks → eval` is already a flagged edge.
- **Move the tensor kernels into `core/` for a torch-free import path.**
  Rejected for now. It moves public functions for an import-time benefit
  only (torch is a hard dependency, ADR-0018). Revisit if per-call import
  cost becomes a measured problem.
- **A solver-neutral evidence vocabulary with a plugin registry.** Rejected
  under ADR-0010: shaped from one solver it would encode that solver's
  output list.
- **Per-benchmark bounds ratified after measuring the benchmark.** Rejected:
  the acceptance criterion becomes a function of the data it judges, and no
  common standard remains.
- **Commit verdicts rather than measurements.** Rejected: every criterion
  change would need the private run files, and a verdict stale against the
  code would be invisible.
- **Store verdicts in the case file, or as per-case sidecar files.**
  Rejected: the first is a schema change (ADR-0012/0013), the second breaks
  ADR-0031's closed archive list. Any archive artefact belongs to ADR-0065
  follow-up 2.
- **Profiles or criteria in TOML/YAML.** Rejected for ADR-0027's reasons.
- **A hand-committed markdown report** (the `docs/timelines/` precedent).
  Rejected: no drift gate, so a criterion could change while the published
  table stays old.
- **Reuse `_card_blocks` for named deck parameters.** Rejected: it silently
  drops blank and non-numeric rows, so positional extraction can return a
  plausible but wrong table.
- **Adopt ADR-0064's 1.001 as the reference-data yield tolerance.**
  Rejected: that number bounds a model head's float32 saturation error
  (~2e-7). The excess measured on reference data (3.5e-4, one case) is three
  orders larger and has a different, as yet unconfirmed, cause.

## Consequences

- **Flag-first items approved with this ADR**: the new module; the two new
  `core` files and their re-exports; the `n_valid_frames` export; a new
  `cli/datacheck.py` (module entry point, no console script); glue scripts
  under `data_generation/lsdyna/<dataset>/`; the `docs/datachecks/` tree.
  `docs/ARCHITECTURE.md` (package layout, module responsibilities,
  dependency graph) is updated in the change that lands the module, and the
  README Roadmap gains the corresponding entries.
- **Delivery is staged on Taylor**: public tiers first, then the run tier,
  then ratification and publication; wave-1D and notch-impact follow. The
  implementation plan lives in `docs/plans/`.
- **`tools/state_probe` is the acceptance oracle**, untouched in the first
  slice: the port must reproduce its recorded numbers before anything
  migrates. Retiring its copies is a later change with a regression first.
- **Tests stay synthetic-only.** Real data enters through environment-gated
  tests that skip when unset. Parser fixtures use invented numbers on the
  real file layout and name the solver version they mimic; unknown labels
  fail closed. A new import-boundary test resolves relative imports (the
  existing torch-free test does not).
- **Shipped benchmarks will show honest gaps.** Most wave-1D and notch
  energy rows will read `not_assessable / not_requested`; DeformingPlate
  will be almost entirely `not_assessable`; Taylor may fail a ratified
  criterion. None of this withdraws or re-scores anything (ADR-0065).
- **Deferred, each with its trigger**: a selective or streamed reader in
  `core/io` (notch-sized cases); material classes beyond tabulated J2 (the
  wave and notch widenings, with the material-class ADR that ADR-0012
  anticipates); a per-benchmark facts field on `BenchmarkSpec` (the first
  non-derivable fact, e.g. a units or wave-speed anchor); waiver records
  (the first ratified criterion a shipped benchmark fails); a deck-derived
  operative yield table (the first dataset with no `BenchmarkSpec`); binary
  solver-output reading (it would make `pandas` a dependency); a dated note
  on ADR-0016 for the d3plot arrays the adapter currently discards.
