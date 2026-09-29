# Datagen platform, part three (a): a gate a real dataset can pass — implementation plan

> **Executed** — merged 2026-09-28 (`ca70020`). A historical record, not instructions: do not re-run it. See `docs/plans/README.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (the maintainer chose native execution for this series). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the preflight passable on a real dataset's merits. The dataset declares which stored quantities the frame step reports rather than judges, each with its reason. The stamp records a person's acceptance of a `review`, with its reason. The budget estimates what production still has to generate. A re-judge mode re-evaluates the steps on existing runs when only the judging changed and every deck is byte-identical.

**Why now:** the first real preflight (2026-09-28, the maintainer's first Abaqus dataset) could not pass whatever the data's quality:
- the frame step judges every stored global, and a rigid-wall reaction force or a strain energy is never resolved at a practical clock;
- a `review` (an unobserved order on a quantity that barely moves) blocks forever, because nothing records the person's call;
- the budget estimated a production sweep that already existed, from pilots placed at the box's corners.

Re-running a 5.5-hour preflight to change a tolerance is waste the evidence does not require.

**Architecture:** Two new `[pilot]` tables carry the dataset's declarations with reasons, beside `accepted_gaps`:
- `frame_reported` maps a stored field key to the reason it is reported, not judged.
- `accepted_reviews` maps `<step>.<name>` to the reason a person accepts that review.

`Step` gains the keys of its reviews. The stamp computes which reviews are accepted and passes a review step only when all its keys are. The budget step sizes production from `problem.mesh()` for the cases not yet completed. `preflight --rejudge` relaxes the stale-runs refusal only where the runs are provably the same evidence: every deck is byte-identical to the current definition's and the units are unchanged. The stamp then records both hash sets.

**Tech Stack:** Python 3.14 (venv at `<venv>`), numpy, pytest, the fake solver of plan 2b; no new dependencies.

**Spec:** `docs/plans/2026-09-27-abaqus-datagen-platform-design.md` ("`preflight`", "Open points"), ADR-0071 (the notes of 2026-09-27 and 2026-09-28), the plan 2b review's deferred minor F10 (no way to record a person's call on a review), and the first real preflight's record (private; its conclusions are in ADR-0071's 2026-09-28 note and this plan's Goal).

## Global Constraints

- Work on branch `feat/datagen-platform-3a` of `<repo>`, cut from `main` at 8032ac3. `main` moves only on the maintainer's instruction; never push.
- Gates before every commit, in one unbroken `&&` chain with `set -o pipefail`: `<venv>/ruff check .`, `<venv>/ruff format --check .`, `<venv>/mypy src`, and the FULL suite `<venv>/python -m pytest -q -p no:cacheprovider`.
  - A heredoc ends a chain, so commit with `-m "$(cat <<'EOF' … EOF)"`.
  - Check new test basenames for duplicates with `git ls-files tests | xargs -n1 basename | sort | uniq -d`.
- TDD for every production change: the failing test first, watched to fail, then the minimal code.
- JSON records are written sorted, LF, `ensure_ascii=False`. The stamp carries no absolute path.
- **No private dataset detail enters the repository.** Task 7 works in the private repository and is written with placeholders.
- Layering is unchanged: `datagen` imports `core`, `datasets`, `verification`, `validation` and itself. `verification` imports `core` and `datasets`.
- Existing definitions keep loading: both new tables default to empty. An empty `frame_reported` means every stored field is judged, which is stricter than today's built-in acceleration exception. The shipped example declares acceleration explicitly.
- An accepted review never turns `fail` or `not_assessable` into a pass. Nothing in this plan deletes a run, a deck or a record.

## Review Focus

1. `--rejudge` after a units change, a deck change or a new case: refused with the first differing case named, nothing run or rewritten. After a tolerance-, QoI- or declaration-only change, nothing is re-run, the stamp records both hash sets, and `generate`'s gate reads the current ones. Task 5 tests each.
2. A review step with one accepted key and one unaccepted key stays blocking. A `fail` or `not_assessable` step is never rescued by `accepted_reviews`. An accepted key that never occurred is listed as unused, not an error. Task 3 tests each.
3. A `frame_reported` key naming a field that is not stored is noted in the step's detail, not silently accepted. An empty reason, or a key outside `node/`, `<block>/`, `global/`, is refused at load. Tasks 1 and 2 test it.
4. The budget with every production case already completed: nothing remains, and the step passes. With some completed, only the rest is estimated. A pilot mesh with no nodes or no elements makes the step `not_assessable`, never a division by zero. Task 4 tests it.
5. The stamp's `passed` is recomputable from the stamp alone: the report renders every accepted review and every reported field with its reason, and the gate reads `passed` only. Tasks 3 and 6 test it.

---

### Task 1: The contract — `[pilot].frame_reported` and `[pilot].accepted_reviews`

**Files:**
- Modify: `src/structbench/datagen/definition.py` (`PILOT_KEYS`, `Pilot`, `_probe_fields` or a sibling helper).
- Modify: `src/structbench/datagen/examples/abaqus_conformance/dataset.toml`.
- Test: `tests/datagen/test_definition.py` (add), `tests/datagen/test_template.py` (the example test).

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class Pilot:
      ...                                                     # as before, plus
      frame_reported: Mapping[str, str] = field(default_factory=dict)    # field key -> reason
      accepted_reviews: Mapping[str, str] = field(default_factory=dict)  # "<step>.<name>" -> reason
  FIELD_KEY = re.compile(r"^(node|global|[A-Za-z_][A-Za-z0-9_]*)/[A-Za-z_][A-Za-z0-9_]*$")
  REVIEW_KEY = re.compile(r"^[a-z_]+\.[A-Za-z_][A-Za-z0-9_]*$")
  ```
- Rules (each a `DefinitionError` naming the field):
  - both tables must be TOML tables of string to non-empty string;
  - a `frame_reported` key must match `FIELD_KEY`;
  - an `accepted_reviews` key must match `REVIEW_KEY`, and its step part must be one of the preflight's step names (`space`, `increment`, `frame`, `duration`, `energy`, `budget`, `verification`, `conformance`, `feasibility`, `deck_regression`), listed in `definition.py` as `REVIEWABLE_STEPS` so `definition` does not import `preflight`.
- The example declares `frame_reported = { "node/acceleration" = "the second time derivative of a frame-sampled explicit response; reported, not scored" }`.

- [ ] **Step 1: Write the failing tests.** Defaults are empty. A valid table of each kind loads. Refusals: an empty reason; a key `"stress"` with no slash; a review key `"spaces.final_length"` (unknown step); a non-string reason; a non-table value. The example's `frame_reported` is the acceleration entry.
- [ ] **Step 2: Run them.** Expected: FAIL, `Pilot` has no `frame_reported`.
- [ ] **Step 3: Implement.** Add both fields to `PILOT_KEYS` and parse them with a `_reasons(pt, key, pattern, allowed_steps=None)` helper.
- [ ] **Step 4: Run** `tests/datagen/test_definition.py tests/datagen/test_template.py`. Expected: PASS.
- [ ] **Step 5: Gates and commit.** Message: `feat(datagen): [pilot].frame_reported and [pilot].accepted_reviews -- the dataset's declarations with their reasons`.

---

### Task 2: The frame step judges every stored field but those the dataset reports

**Files:**
- Modify: `src/structbench/datagen/preflight.py` (`step_frame`).
- Test: `tests/datagen/test_preflight.py`: the acceleration test becomes a declared-report test; add the tests below.

**Interfaces:**
- Consumes: `defn.pilot.frame_reported`.
- Produces: `step_frame`'s detail with
  - `judged`: every interpolated key not in `frame_reported`;
  - `reported`: `{key: {"error": value, "reason": reason}}` for the declared keys that are stored;
  - `reported_absent`: the declared keys not stored, sorted.

  The built-in acceleration exception is removed; the dataset now carries it. The summary names how many fields were judged and lists the reported ones.

- [ ] **Step 1: Write the failing tests.**
  - With `frame_reported = {}`, a jagged `node/acceleration` fails the step.
  - With `frame_reported = {"node/acceleration": "…"}`, the same case passes, `reported["node/acceleration"]["reason"]` is the declared text, and `node/acceleration` is not in `judged`.
  - An element field (`solid/stress`) is judged by default and fails when jagged.
  - A declared key the case does not store lands in `reported_absent` and changes nothing else.
  - A non-finite value in a *reported* field does not block. A non-finite value in a judged field reads `not_assessable`, as before.
- [ ] **Step 2: Run them.** Expected: FAIL. `solid/stress` is not judged today, and acceleration is excepted by code.
- [ ] **Step 3: Implement.** Replace the `judged` / `reported` computation. Keep `interpolation`, `common_instants`, `rise_frames` and `stride` as they are.
- [ ] **Step 4: Run** `tests/datagen/test_preflight.py`, then the e2e file. The fake solver's fields are smooth, and the example now declares acceleration. Expected: PASS.
- [ ] **Step 5: Gates and commit.** Message: `feat(datagen): the frame step judges every stored field but those the dataset reports, each with its reason`.

---

### Task 3: A person's call on a review, recorded — `Step.reviews` and the stamp's `accepted`

**Files:**
- Modify: `src/structbench/datagen/preflight.py` (`Step`, `step_space`, `passed`, `stamp_record`, `render_report`).
- Test: `tests/datagen/test_preflight.py` (add).

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class Step:
      name: str
      verdict: str
      summary: str
      detail: dict[str, Any]
      reviews: tuple[str, ...] = ()     # "<step>.<name>" keys behind a review verdict

  def accepted(steps: Sequence[Step], acceptances: Mapping[str, str]) -> dict[str, str]
      # the review keys that occurred and are accepted, with reasons
  def passed(steps: Sequence[Step], acceptances: Mapping[str, str] = {}) -> bool
      # every step pass/not_applicable, or review with every key accepted
  ```
  - `step_space` fills `reviews` with `space.<qoi>` for every reviewed quantity, de-duplicated and sorted.
  - `stamp_record` gains three keys:
    - `"accepted_reviews"`: the keys that occurred and are accepted, each `{key: reason}`;
    - `"unaccepted_reviews"`: a sorted list;
    - `"unused_acceptances"`: the declared keys that never occurred, sorted.
  - Each step entry gains `"reviews"`.
  - `render_report` prints an "Accepted by the dataset" section: each accepted review and each reported frame field with its reason.
  - A stamp passes on accepted reviews only when no step is `fail` or `not_assessable`.

- [ ] **Step 1: Write the failing tests.**
  - `step_space` on a record with one oscillatory QoI: `reviews == ("space.final_length",)`.
  - `passed` with that key accepted: True. With one of two review keys accepted: False.
  - A `fail` step plus accepted reviews: False. A `not_assessable` step: False.
  - `stamp_record` lists accepted, unaccepted and unused keys correctly.
  - The report contains every reason verbatim.
  - The stamp's `passed` equals `passed()` recomputed from the stamp's `steps` and `accepted_reviews`.
- [ ] **Step 2: Run them.** Expected: FAIL, `Step` takes no `reviews`.
- [ ] **Step 3: Implement.** The driver passes `defn.pilot.accepted_reviews` into `stamp_record`, whose signature becomes `stamp_record(defn, dataset_dir, steps, specs, *, created_utc, siblings_sha256)`. It reads the acceptances from `defn` itself, so callers do not change.
- [ ] **Step 4: Run** `tests/datagen/test_preflight.py tests/datagen/test_generate.py`. Expected: PASS.
- [ ] **Step 5: Gates and commit.** Message: `feat(datagen): the stamp records a person's call on a review -- accepted with its reason, never over a fail or a missing measurement`.

---

### Task 4: The budget estimates what production still has to generate

**Files:**
- Modify: `src/structbench/datagen/preflight.py` (`step_budget`, the driver's call).
- Test: `tests/datagen/test_preflight.py` (the budget test rewritten, plus two more).

**Interfaces:**
- Produces:
  ```python
  def step_budget(defn, problem, specs, runs, sizes, free, *, completed: Collection[str] = ()) -> Step
  ```
  - `completed` holds the production case ids whose `run.json` says completed in the sweep directory. The driver reads them from `<work-root>/<name>/*/run.json`.
  - The remaining production cases are those planned, not probes, and not completed.
  - **Disk:** `bytes_per_node` is the median over production-level pilots of `size / nodes(mesh(pilot params))`. The estimate is `Σ nodes(mesh(case params)) × bytes_per_node` over the remaining cases.
  - **Wall:** `wall_per_element` is the median of `wall_s / elements(mesh)` over the same pilots. The estimate is `Σ elements × wall_per_element`, serial. The summary says it ignores the increment count and the case's severity.
  - Detail keys: `production_cases`, `completed_cases`, `remaining_cases`, `bytes_per_node`, `wall_s_per_element`, `estimated_gb`, `estimated_wall_h`, `free_gb`, `min_free_gb`, `per_level`.
  - `run.estimate_line` reads `bytes_per_case` and `wall_s_median`. The detail keeps both, as the median per pilot, so the runner's line stays meaningful.
  - With no remaining case the step passes with "nothing left to generate".
  - A zero-node or zero-element pilot mesh reads `not_assessable`.

- [ ] **Step 1: Write the failing tests.**
  - With the toy definition and pilots of two sizes (L 1 and 2), the estimate scales with the production meshes' node counts, not the pilots' median bytes. Derive the literals from `problem.mesh` by hand in the test.
  - With every production id completed, nothing remains and the step passes.
  - With half completed, the estimate is half.
  - A problem whose `mesh()` returns no nodes reads `not_assessable`.
- [ ] **Step 2: Run them.** Expected: FAIL (signature).
- [ ] **Step 3: Implement.** Add the driver's `completed` scan.
- [ ] **Step 4: Run** `tests/datagen/test_preflight.py tests/datagen/test_run_budget.py`. Expected: PASS.
- [ ] **Step 5: Gates and commit.** Message: `feat(datagen): the budget estimates what production still has to generate, sized by the production meshes`.

---

### Task 5: `preflight --rejudge` — the steps re-evaluated on the same evidence

**Files:**
- Modify: `src/structbench/datagen/preflight.py` (`preflight`, `main`, `_stale_cases`, `stamp_record`).
- Test: `tests/datagen/test_preflight_e2e.py` (add).

**Interfaces:**
- Produces:
  - `preflight(..., rejudge: bool = False)` and a `--rejudge` flag.
  - With `rejudge` and stale cases present, each stale case must satisfy both conditions below. The first case that does not is named in the refusal (exit 2):
    - its deck on disk is byte-identical to `deck_for(spec, problem)` under the current definition;
    - its provenance `units` equals `defn.units`.
  - Every current spec must already have its case folder; a new case (a changed pilot list) is refused, because re-judging runs nothing.
  - When all hold, the driver skips `materialise` and every run and export. It re-converts only what is missing, re-judges everything, and writes the stamp with `"runs_generated_under"`: the distinct `{definition_sha256, problem_sha256, siblings_sha256}` sets found in the cases' provenance.
  - Without `--rejudge`, behaviour is unchanged: stale runs are refused.

- [ ] **Step 1: Write the failing e2e tests.**
  - Pass once; loosen `[qoi].tolerance` and commit; `--rejudge` exits 0. Nothing is re-run (every `run.json`'s bytes are unchanged), and the stamp's `definition_sha256` is the new one while `runs_generated_under` holds the old one.
  - Change the deck (the heading) and commit; `--rejudge` exits 2 naming a case.
  - Change `[dataset].units` in a definition whose decks do not depend on it and commit; `--rejudge` exits 2 naming the units.
  - Add a pilot point; `--rejudge` exits 2 with "new case".
- [ ] **Step 2: Run them.** Expected: FAIL, unknown flag.
- [ ] **Step 3: Implement.** Split the driver's "evaluate" half, from convert to finish, into a function both paths call.
- [ ] **Step 4: Run** the e2e file and `tests/datagen/test_preflight.py`. Expected: PASS.
- [ ] **Step 5: Gates and commit.** Message: `feat(datagen): preflight --rejudge re-evaluates the steps on runs whose decks and units the current definition reproduces`.

---

### Task 6: Documentation, the example, the ADR note, the snapshot

**Files:**
- Modify: `docs/DATA_GENERATION.md`:
  - the `[pilot]` table gains both fields;
  - the frame row loses the built-in acceleration clause;
  - a paragraph on accepted reviews;
  - the budget's new semantics;
  - `--rejudge`.
- Modify: `decisions/0071-datagen-platform.md`: a note "part three (a) built", with the reasons the first real preflight gave.
- Modify: `docs/plans/2026-09-27-abaqus-datagen-platform-design.md`: the `[pilot]` row, and the open point on `review`.
- Modify: `CLAUDE.md`:
  - the snapshot's part-three line;
  - replace "acceleration reported but not judged (a ruling the maintainer may reverse)" with the dataset-declared mechanism.
- Modify: `docs/datagen/abaqus-lessons.md`: checklist item 10 mentions declaring the reported fields and accepted reviews.
- Test: `tests/datagen/test_template.py` (the example); the full suite.

- [ ] **Step 1:** Write the documents. Scan the diff for private detail.
- [ ] **Step 2:** Full gates. Commit with message `docs: frame_reported, accepted_reviews, the remaining-production budget and --rejudge in the guide, the design, ADR-0071 and the snapshot`.

---

### Task 7: The private dataset under the new contract, re-judged

**Files (the private dataset repository; `<dataset dir>` its dataset folder, `<runs>` its run folder):**
- Modify: `<dataset dir>/dataset.toml`:
  - `frame_reported` for the fields its card already says the clock does not resolve, each with the card's reason;
  - `accepted_reviews` for any review the re-judged stamp shows, each with a reason the maintainer stands behind, proposed from the card's numbers and flagged in NOTES as a proposal.
- Modify: `<dataset dir>/NOTES.md`, `<dataset dir>/DATA_CARD.md` (the "Resolution in time" record line), `<dataset dir>/preflight/` (the new stamp and report).

- [ ] **Step 1:** `check` the definition, then run `preflight --rejudge` on the existing preflight runs. Expected: no solver run.
- [ ] **Step 2:** Read the stamp. Every step that still does not pass is reported to the maintainer with its numbers; nothing is accepted that the card does not already state.
- [ ] **Step 3:** Run the private suite and commit there. Nothing is pushed.

---

## Self-review

- **Spec coverage:**
  - The two blockers the first real preflight exposed: Task 2 (reported fields) and Task 3 (accepted reviews).
  - The budget's two biases: Task 4. Corner pilots are handled by sizing from the production meshes; already-generated production by counting only the remaining cases.
  - The cost of re-running unchanged evidence: Task 5.
  - Docs: Task 6. The private dataset: Task 7.
  - Not in this plan: the other part-three items (the Abaqus evidence and hourglass rows, the card generator, the full guide).
- **Placeholders:** the tests of Tasks 1–5 are named with their intent and expected values rather than written out. The executor writes each from the interfaces and derives the literals by hand before running. Task 7 uses placeholders by the global constraint.
- **Type consistency:**
  - `Step.reviews` is filled by `step_space` and read by `accepted` and `passed`.
  - `Pilot.accepted_reviews` is read by `stamp_record`; `Pilot.frame_reported` by `step_frame`.
  - `step_budget(..., completed=)` is called by the driver, and its detail keeps `bytes_per_case` and `wall_s_median` for `run.estimate_line`.
  - `stamp_record` gains `accepted_reviews`, `unaccepted_reviews`, `unused_acceptances` and `runs_generated_under`, all rendered by `render_report`.
- **Review Focus:** 1 → Task 5; 2 → Task 3; 3 → Tasks 1 and 2; 4 → Task 4; 5 → Tasks 3 and 6.
