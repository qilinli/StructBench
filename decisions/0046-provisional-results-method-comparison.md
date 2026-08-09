# 0046 — Provisional results and the method-comparison table (closes ADR-0041 clause 4)

**Status**: Proposed
**Type**: Durable
**Date**: 2026-08-09

## Context

ADR-0041 clause 4 named the reusable substrate v0.3 delivers in one sentence:
the results registry (ADR-0033) "extends from per-benchmark to
**per-(benchmark × method)** with a `provisional` flag; the generated landing
page (ADR-0036) renders a **method-comparison table** distinguishing blessed
from provisional." No shipped ADR had claimed it. ADR-0044 (Transolver) and
ADR-0045 (GeoFLARE) both explicitly deferred it — each closes with "The
comparison view and the `provisional` registry flag are a separate plan." This
ADR is that plan, and the last agent-side deliverable of the v0.3 build order
(ADR-0041 clause 7). It records the schema extension, the comparison renderer,
and the display-key contract; the training runs the table will hold are
maintainer compute.

Three facts shaped every decision below:

1. **"Blessed" was convention only, with no way to record a provisional
   result.** `BaselineResult` carried `family, label, run_commit, run_date,
   metrics, checkpoint=None, checkpoint_sha256=None, notes=""` and no status
   field; an entry was "blessed" merely by appearing in a `RESULTS` tuple.
   There was **exactly three** blessed entries — taylor (`cgn`/`7be9d4b`),
   wave-1d (`cgn`/`48046ea`), notch-impact (`cgn`/`5956d81`); `notch_beam_2d_bend`
   and `deforming_plate` both ship `RESULTS = ()`. (The earlier plan framing's
   "four existing entries" was wrong — it is three; the backward-compat blast
   radius is those three tuples.) ADR-0041 mandates provisional Transolver and
   GeoFLARE entries that must record as present-but-not-blessed, which no
   mechanism supported.

2. **There was no side-by-side renderer — the comparison table is new code,
   not a tweak.** All result rendering was strictly sequential over
   `spec.results`: `_numbers_to_beat` emits one-or-two tables *per*
   `BaselineResult`, never a method × metric matrix. The natural home is
   `render_benchmark_page` / `render_archive_readme`, which own
   "## Numbers to beat". Because a shared renderer serves every page and the
   drift tests byte-match all of them, the choice to make the section uniform
   (below) means every committed page regenerates once.

3. **A live Quickstart bug was entangled with the plan.** Both renderers picked
   `family = spec.results[0].family if spec.results else "cgn"` and interpolated
   into `configs/{name}/{family}.toml`. `deforming_plate` has `RESULTS = ()`, so
   the fallback always resolved `"cgn"` — but `configs/deforming_plate/` holds
   only `mgn`/`transolver`/`geoflare` pairs, no `cgn.toml`, so the committed page
   pointed at a nonexistent config. `deforming_plate` is the first benchmark
   where the hardcoded-`cgn` premise is false; the fix ships here.

Everything below is grounded in a verified research pass
(`scratch/2026-08-09-comparison-table-grounding.md` — maintainer-local scratch
record, gitignored and absent from clones; produced by verification workflow
wf_7f62939a-0ae: 93 extracted claims, 85 adversarially confirmed, 8
refuted-with-corrections; §-references point there). Two controller corrections
were applied on top of that pass and are load-bearing here: (a) the grounding's
own refutation of claim c25 was itself overturned — the ADR-0043 §8 pooling
convention is over **all** mesh nodes per the governing 2026-08-08 dated note,
not NORMAL nodes only; and (b) the plan review falsified the "import cycle"
rationale first offered for omitting family-vocabulary validation (clause 3) —
`config`'s `benchmarks` import is function-local, so no cycle exists; the real
reasons are YAGNI and coupling avoidance. Every literal below (field names,
defaults, footnote text, display keys, the empty-state line) matches the code
as implemented on `feat/provisional-comparison-table`; where this ADR and the
code could disagree, **the code governs**.

## Decision

1. **Schema: `provisional: bool = False`, appended after `notes` on
   `BaselineResult`.** A plain boolean, not an enum or a separate collection:
   two states are all ADR-0041 names, and `False = blessed` keeps every existing
   entry blessed with **zero call-site edits**. All trailing `BaselineResult`
   fields already carry defaults and all three call sites use keyword arguments,
   so a trailing defaulted field is a pure additive extension — the exact
   precedent is ADR-0037 clause 4's `checkpoint_sha256`, which was flagged as a
   public-API change though `BaselineResult` is not in `benchmarks.__all__`.
   This ADR flags the field addition the same way. The field's meaning is fixed
   in the docstring: `False` (default) = blessed, validated against its published
   anchor or protocol gate; `True` = best-effort implementation recorded for
   comparison, fidelity unvalidated (ADR-0044/0045/0046) — never read as a
   blessed baseline.

2. **`BenchmarkSpec.blessed_results` is the one canonical blessed predicate.**
   A new property returning `tuple(r for r in self.results if not r.provisional)`
   in declaration order. After this ADR, "is this blessed?" is answered *only*
   by this predicate — nothing re-derives blessedness from mere presence in
   `results`. Both the Quickstart selection (clause 4) and the reworked
   blessed-set identity test consume it. `results` remains the full tuple
   (blessed + provisional); `blessed_results` is the filtered view, and the
   docstrings on both point readers to `blessed_results` wherever "the blessed
   baseline" is meant.

3. **Duplicate-family rejection; deliberately no family-vocabulary check.**
   `BenchmarkSpec.__post_init__` rejects two entries sharing a `family` for one
   benchmark (`ValueError` naming both the duplicated family and the benchmark
   via `self.card.name` — the only identity available there), because a family
   is the comparison table's column key and duplicate columns are meaningless.
   It does **not** validate `family` against `config.MODEL_FAMILIES`. This
   omission is a decision, not an oversight, and its rationale is corrected from
   an earlier draft: there is **no import cycle** (`config`'s `benchmarks`
   import is function-local, so `benchmarks → config` would import cleanly).
   The real reasons are YAGNI for a small maintainer-controlled vocabulary that
   `config.MODEL_FAMILIES` already governs at config-load time, and avoiding a
   new `benchmarks → config` coupling for a check that cannot fail in practice
   (a maintainer transcribing a blessed run reads the family off `config.json`).

4. **Quickstart: an explicit `quickstart_family` default plus a blessed-first
   selection chain.** `BenchmarkSpec` gains `quickstart_family: str = "cgn"`
   (validated non-blank). The default preserves the four pre-v0.3 benchmarks
   unchanged; `deforming_plate` sets `quickstart_family="mgn"` — its reference
   method (ADR-0043) with a committed `configs/deforming_plate/mgn.toml` — which
   fixes the nonexistent-`cgn.toml` bug immediately, before any run is recorded.
   The renderer's `_quickstart_family(spec) -> (family, provenance)` selection
   chain is:
   - first **blessed** result's family (declaration order) → `provenance = "blessed"`;
   - else the first result's family, blessed or not → `"provisional"`;
   - else `spec.quickstart_family` → `"default"`.
   A provisional entry listed before a later blessed one still loses to the
   blessed one, because `blessed_results` preserves declaration order among
   blessed entries only. Both renderers use the selected family for the **config
   path**; the selection-aware **prose** exists only in `render_benchmark_page`
   (the archive README has a config path but no recipe prose): the current
   blessed-recipe wording when the chosen result is blessed, a distinct
   provisional wording ("the provisional `<family>` recipe … never read it as a
   blessed baseline", citing ADR-0044/0045) when provisional, and the current
   generic wording when there are no results. A **config-path-exists regression
   guard** is added to the test suite — for every registered benchmark it parses
   the Quickstart config path out of the rendered page and asserts
   `configs/<name>/<family>.toml` exists on disk — so this class of bug cannot
   recur silently.

5. **The method-comparison table: a new `## Method comparison` section on every
   page.** Additive to (and immediately before) the existing `## Numbers to
   beat` blocks, which keep their split-complete detail and checkpoint pointers
   unchanged. It is uniform across all benchmarks — a single-method benchmark
   renders a one-column table, no renderer branch — so all pages regenerate once
   and the empty/partial states are exercised from day one. Layout and encoding:
   - **Columns = methods.** One column per `spec.results` entry in declaration
     order; header `**<family>**`, with a ` (provisional)` suffix when the entry
     is flagged. Column `| Metric | … |`.
   - **Rows = (split · metric) pairs.** Split axis: iterate `spec.splits` in
     declaration order and include a split iff at least one result carries a
     metric for it (exactly `_numbers_to_beat`'s existing rule; `eval_splits` is
     not involved). Metric axis (**first-seen union rule**, mirroring the
     renderer's existing convention): iterate results in declaration order,
     within each iterate its splits in card order, and collect not-yet-seen
     metric keys into **one** global ordered list; then per split-row-group,
     filter that global list to keys present in **any** entry for that split, and
     stable-partition non-`qoi_` keys before `qoi_` keys. Row label
     `<split> · <metric>` (a middle dot). Missing cells render `—`.
   - **Provisional encoding is unmissable (honest-not-silent, ADR-0033):** the
     ` (provisional)` header suffix, plus a footnote appended whenever any column
     is provisional, verbatim:

     > `*Provisional entries are best-effort implementations whose fidelity is not validated against published numbers (ADR-0044/0045) — never read them as blessed baselines.*`

     Blessed columns carry no tag. The `" — private archive; publication parked"`
     checkpoint marker is the style precedent for this.
   - **Empty state**, verbatim, and deliberately distinct from
     `_numbers_to_beat`'s ADR-0033 "no official baseline yet" placeholder —
     this section is method-oriented and provisional entries land here too, so a
     separate wording is a recorded choice, not drift:

     > `*No results yet — method entries land here as runs are recorded (blessed or provisional).*`

     Partial state shows only the recorded methods (the registry is the source
     of truth; no skeleton columns for pending methods).
   - **`_numbers_to_beat` headings are tagged too, not hidden.** A provisional
     entry keeps its full per-result detail block below `## Numbers to beat` —
     the per-split tables and checkpoint pointer matter for a provisional run
     just as much as a blessed one — but its heading gains the same
     ` (provisional)` suffix as the comparison-table header and `_baseline_line`,
     so a provisional entry's detail block can never be read as a blessed
     "number to beat" (closing a finding from the final whole-branch review,
     2026-08-09).

6. **Comparison-statistic containment: the ADR-0043 §8 pooled number is
   notes-only.** The comparison renderer surfaces **every** `metrics` key as a
   row — there is no "non-comparison metrics key" slot. So the MGN blessing-gate
   pooled aggregate (ADR-0043 §8: root-mean-square over coords × all mesh nodes ×
   scored steps × trajectories, in native units — a different statistic from the
   §5 per-step-mean leaderboard set) must live in `notes` free-text, never a
   `metrics` key, or it would leak into the comparison columns and destroy the
   comparability ADR-0043 itself warns about. This **refines** ADR-0043's
   Consequences wording ("the blessing record — ADR-0033 registry
   `notes`/metrics") to **notes-only**. A dated pointer note on ADR-0043 records
   the narrowing; it is a deliberate tightening of that ADR's letter, not silent
   drift.

7. **`deforming_plate` display-key contract, pre-declared here.** The schema and
   renderer are built ahead of any run, so the display keys a future MGN /
   Transolver / GeoFLARE `BaselineResult` will use are fixed now, transcribed
   from `metrics-<split>.json["mean"]` (all three mesh families emit the same
   family-agnostic block) into the mm / MPa working frame with precedent naming:

   | `BaselineResult` metric key | `metrics-<split>.json["mean"]` source |
   |---|---|
   | `one_step_pos_rmse_mm` | `one_step_position_rmse` |
   | `one_step_vm_rmse_mpa` | `one_step_aux_rmse` |
   | `rollout_pos_rmse_mm` | `rollout_position_rmse` |
   | `rollout_vm_rmse_mpa` | `rollout_aux_rmse` |
   | `qoi_peak_vm_stress_mae_mpa` | `qoi_abs_error["peak_vm_stress"]` |
   | `qoi_terminal_peak_deflection_mae_mm` | `qoi_abs_error["terminal_peak_deflection"]` |

   The two QoI keys keep the `qoi_` prefix the renderer partitions on. Recorded
   split: `test` only (per the existing precedent, `val` is omitted — it selects
   checkpoints). This is the transcription recipe; the actual entries land with
   their runs.

8. **Provisional archive discipline is identical to blessed (ADR-0037).** A
   provisional entry carries a `checkpoint` pointer and `checkpoint_sha256`, and
   is bundled to the private `models/` OneDrive mirror, exactly as a blessed run
   is; the `hpc/dug/README.md` §5 blessing-and-archiving checklist applies to
   both. The `provisional` flag changes how an entry is *interpreted* and
   *rendered*, not how it is stored — a provisional checkpoint pointer still
   renders with the private-archive marker until publication.

9. **Ordering convention: declaration order is column order, blessed first.**
   The order of entries in a benchmark's `RESULTS` tuple is the comparison-table
   column order (deterministic; the drift test and the Quickstart tie-break both
   depend on it). By convention blessed entries are declared before provisional
   ones. This is a convention, not validation — nothing rejects a provisional-
   first declaration; `blessed_results` and `_quickstart_family` remain correct
   regardless.

10. **Index tagging keeps the cross-benchmark page consistent with the landing
    pages.** `_baseline_line` (the `docs/benchmarks.md` one-line summary) gains a
    ` (provisional)` suffix on each provisional entry's phrase, so the index
    never quietly reads a provisional method as blessed once `deforming_plate`'s
    methods land. (No committed entry is provisional today, so this is
    output-neutral until the first provisional run is recorded.)

11. **Test migration.** The blessed-set identity test that filtered on truthy
    `.results` is reworked to filter on `blessed_results` — asserting the **same
    three** blessed benchmarks as today, proving zero behaviour change for the
    existing entries. `test_deforming_plate_split`'s `spec.results == ()`
    assertion **stays** (still true — no entry exists yet) and its
    `quickstart_family == "mgn"` is pinned; the `results == ()` assertion is
    noted to change at the first recorded entry, blessed or provisional.

## Alternatives considered

- **A `status: Literal["blessed", "provisional"]` field (or a separate
  provisional collection).** Rejected. An enum is YAGNI for two states and would
  force every consumer that today reads `results` to branch on a string; a
  separate collection duplicates the container, splits the single source the
  renderers iterate, and complicates ordering. A trailing `bool = False` is
  minimal, keeps existing entries blessed with zero edits, and matches ADR-0041's
  literal "provisional flag".
- **A multi-method-only table (render the section only when >1 result).** Rejected.
  It needs a renderer branch, leaves the single-baseline pages inconsistent with
  the multi-method one, and hides the empty/partial-state behaviour that the
  `deforming_plate` drift test should pin from day one. A uniform one-column
  table costs one regeneration of the existing pages and no branch.
- **A filesystem-probing Quickstart (detect which `configs/<name>/*.toml`
  exist).** Rejected. It breaks renderer purity (functions stay pure over specs,
  no disk probing), is non-deterministic against the committed configs, and hides
  the real fix. The default family is spec data (`quickstart_family`); the
  config-path-exists guard is a *test*, not a renderer responsibility.
- **Family-vocabulary validation via a `MODEL_FAMILIES` import in
  `benchmarks.registry`.** Rejected — but not for the reason first offered. There
  is no import cycle (the `config → benchmarks` import is function-local). It is
  declined on YAGNI (a small maintainer-controlled vocabulary `config` already
  governs at load time) and on not adding a `benchmarks → config` coupling for a
  check that cannot fail in practice (clause 3).
- **The pooled blessing number as a `metrics` key with an exclusion rule.**
  Rejected. The renderer surfaces every `metrics` key as a row, so a "don't show
  this one" exclusion rule would be a second, fragile source of truth; free-text
  `notes` keeps the pooled aggregate out of the comparison columns by
  construction (clause 6).

## Consequences

- **Every committed benchmark page regenerates once.** The uniform section adds
  `## Method comparison` to all five `docs/benchmarks/<name>.md` pages; the
  cross-benchmark index (`docs/benchmarks.md`) is byte-identical (it has no
  comparison section and no committed entry is provisional). The drift tests
  govern, and the section insertion + regeneration are atomic in one commit.
- **`blessed_results` is now the single blessed predicate.** Any future consumer
  asking "is this blessed?" uses it; presence in `results` no longer implies
  blessed.
- **The recording workflow for the three upcoming `deforming_plate` runs is
  fixed.** Blessing MGN = transcribe `metrics-test.json` into a `BaselineResult`
  (display keys per clause 7) with `provisional=False`, archive per ADR-0037,
  regenerate. Recording provisional Transolver / GeoFLARE = the same transcription
  with `provisional=True`; the `hpc/dug/README.md` §5 checklist applies to both,
  and the flag alone changes interpretation and rendering. Each recorded entry
  adds a column to the `deforming_plate` comparison table and (for the first one)
  flips its Quickstart from the `quickstart_family="mgn"` default to the selected
  result's family.
- **The card schema is untouched.** Method identity lives in `BaselineResult.family`;
  `BenchmarkCard` gains no results/method/default-family field.
- **No new dependency.** The change is stdlib-only Python across
  `benchmarks/results.py`, `benchmarks/registry.py`,
  `benchmarks/render.py`, and `benchmarks/deforming_plate/__init__.py`.
- **ADR-0033 and ADR-0036 are extended, ADR-0043 refined**, each by a dated
  pointer note (in-place dated notes have precedent), and this ADR closes
  ADR-0041 clause 4 — the v0.3 cross-method infrastructure is delivered; the
  numbers it will hold are maintainer compute.
