# Datagen platform, part two (b): the preflight and its stamp, the gate, the runner's budget, `follow` — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (the maintainer chose native execution). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the pipeline its preparation stage: `structbench-datagen preflight` runs the pilot split through the three resolutions (space, time, duration), the energy account, the budget and the verification instrument, writes a report and a stamp bound to the definition's hashes; `generate` refuses production splits without a passing stamp; `run` stops cleanly when the disk runs low and prints the sweep's estimate; `follow` exports and converts finished cases while a run proceeds.

**Architecture:** The temporal measures (settling, separation, rise time, midpoint-interpolation and common-instant errors) are solution verification and live in `verification/temporal.py`, on `Case` objects, beside `convergence.py`. `datagen/preflight.py` does what only the pipeline knows: it plans the preflight case set from the pilot points (levels, increment factors, a frame probe, a conformance run), materialises the decks through `generate`'s machinery into `<work-root>/<name>/preflight/` — a sweep of its own — drives `run`, `export`, `convert`, `verify` and `converge` over it, turns their records into ten step verdicts, and writes `report.md` and `stamp.json`. The stamp is the contract between `preflight` and `generate`; the runner reads the same stamp for its margin and its estimate. `export` moves out of `cli.py` into `datagen/export.py` so `preflight` and `follow` can call it without importing the entry layer.

**Tech Stack:** Python 3.14 (venv at `<venv>`), numpy, h5py through `core`, scipy (the `datagen` extra, already required by `converge`), tomllib, pytest; no new dependencies. Tests run the whole chain against a fake solver script (no Abaqus, no `odbAccess`).

**Spec:** `docs/plans/2026-09-27-abaqus-datagen-platform-design.md` (ADR-0071 Proposed, `decisions/0071-datagen-platform.md`), sections "Stages", "`preflight`", "Testing", "Open points"; the ADR-0071 note of 2026-09-27 (three resolutions); plan 2a's deferred minor "probe-only groups dropped silently" (the anchor split).

## Global Constraints

- Work on branch `feat/datagen-platform-2b` of `<repo>` (cut from `main` at 486d9be); `main` moves only on the maintainer's instruction; never push.
- Gates before every commit, in one unbroken `&&` chain with `set -o pipefail` (a heredoc ends a chain — commit with `-m "$(cat <<'EOF' … EOF)"`): `<venv>/ruff check .`, `<venv>/ruff format --check .`, `<venv>/mypy src`, and the FULL suite `<venv>/python -m pytest -q -p no:cacheprovider` (duplicate test basenames break partial collection; check new basenames with `git ls-files tests | xargs -n1 basename | sort | uniq -d`).
- TDD for every production change: the failing test first, watched to fail, then the minimal code.
- Files written by scripts are bytes with LF endings; JSON records are `json.dumps(..., indent=2, sort_keys=True, ensure_ascii=False) + "\n"`; the stamp carries a `created_utc` by design (the spec names the date) and no absolute path; every other record has no timestamp.
- **No private dataset detail enters the repository**: no dataset name, split name, case id, constant, count, run path or private repository path. Task 10 works in the private repository and is written here with placeholders.
- Layering, held by the AST boundary tests: `verification` imports `core` and `datasets` only; `datagen` imports `core`, `datasets`, `verification`, `validation` and itself — never `cli`, `benchmarks`, `eval`, `models`. `datagen/cli.py` imports the stages; no stage imports `datagen/cli.py`.
- Existing definitions keep loading: every new `[pilot]` and `[qoi]` field has a default. Existing sweeps keep working: readers accept `abaqus-provenance/2` and `/3` alike.
- Nothing in this plan deletes a run folder, a deck or a record; a preflight directory that holds another definition's runs is refused, not cleared.

## Review Focus

1. A preflight re-run after `dataset.toml` or `problem.py` changed while `<sweep>/preflight/` holds runs of the old definition: refused with exit 2 and one sentence, nothing deleted or overwritten — Task 7 tests it.
2. `generate` against a stamp that did not pass, or whose hashes match only one of the two files: refused (exit 2) naming the guarded splits; `--no-preflight` generates and writes `"preflight": {"skipped": true}` into every guarded case's provenance; probe splits are never gated — Task 4 tests each.
3. The runner's disk stop: jobs already running finish and are recorded `completed`; jobs not launched get no `run.json` and stay `pending`; the lock is released; the exit code is 3 — Task 5 tests it.
4. A frame probe that cannot align: `fixed.n_intervals / frame_factor` not a whole number is refused at load; a probe key absent from `[fixed]` makes the step `not_assessable` naming the key; never a silently misaligned comparison — Tasks 1 and 6 test it.
5. A probe key the deck does not read (`dt_scale` set but `input_deck` ignores it): the deck-regression step fails naming the key, so the increment step cannot pass trivially on identical runs — Task 6 tests it.

---

### Task 1: The contract — `[pilot]`'s probe fields, `[qoi].tolerance`, the conformance deck, the ledger's public names

**Files:**
- Modify: `src/structbench/datagen/definition.py` (`Pilot`, `QoiSpec`, `load_definition`), `src/structbench/datagen/abaqus/deck.py` (`with_all_energy`), `src/structbench/core/io/abaqus.py` (`LEDGER_CLOSED_TERMS`, `assembly_history`)
- Test: `tests/datagen/test_definition.py` (add), `tests/datagen/test_deck.py` (add), `tests/core/test_abaqus_adapter.py` (add)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  ```python
  # definition.py
  PILOT_KEYS = {"split", "fine_cases", "min_free_gb", "accepted_gaps", "increment_key",
                "increment_factors", "frame_key", "frame_count_key", "frame_factor",
                "frame_tolerance", "settling_margin", "contact_force_global"}
  QOI_KEYS = {"names", "units", "tolerance"}

  @dataclass(frozen=True)
  class Pilot:
      split: str
      fine_cases: tuple[str, ...]
      min_free_gb: float
      accepted_gaps: tuple[str, ...]
      increment_key: str = "dt_scale"
      increment_factors: tuple[float, ...] = (0.5,)   # production value × factor; () disables the probe
      frame_key: str = "frame_interval"
      frame_count_key: str = "n_intervals"
      frame_factor: float = 0.5                        # 0 disables the frame probe
      frame_tolerance: float = 0.05
      settling_margin: float = 0.25
      contact_force_global: str | None = None          # a stored global's name, "global/" prefix stripped

  @dataclass(frozen=True)
  class QoiSpec:
      names: tuple[str, ...]
      units: tuple[str, ...]
      tolerance: tuple[float, ...]                     # one per name; default 0.01 each

  # deck.py
  def with_all_energy(text: str) -> str               # ValueError when standard_output's block is absent

  # core/io/abaqus.py
  LEDGER_CLOSED_TERMS: frozenset[str]                  # the old _LEDGER_CLOSED, kept as an alias
  def assembly_history(export: AbaqusExport) -> dict[str, NDArray[np.float64]]   # term -> values on the frame clock
  ```

Validation rules in `load_definition` (each a `DefinitionError` naming the field):
- unknown keys in `[pilot]` or `[qoi]` are refused (`pilot.<key>: unknown field`);
- `increment_factors`: every factor `> 0` and `!= 1.0`;
- `0 <= frame_factor < 1`; when `frame_factor > 0` and both frame keys are in `[fixed]`, `fixed[frame_count_key] / frame_factor` must be a whole number within `1e-9` (`pilot.frame_factor: fixed.n_intervals / 0.3 is not a whole number of frames`);
- `increment_key`, `frame_key`, `frame_count_key` may not be sampled (in `[variables]` or any split's `extra`): `pilot.increment_key: 'dt_scale' is sampled; the probe needs a constant`;
- `frame_tolerance > 0`, `0 <= settling_margin < 1`;
- `qoi.tolerance`: one positive number per name; absent → `0.01` per name;
- `contact_force_global`: a non-empty string; a leading `global/` is stripped; the empty string means `None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/datagen/test_definition.py`:

```python
def test_pilot_probe_fields_have_defaults(definition_dir):
    p = definition.load_definition(definition_dir).pilot
    assert (p.increment_key, p.increment_factors) == ("dt_scale", (0.5,))
    assert (p.frame_key, p.frame_count_key, p.frame_factor) == ("frame_interval", "n_intervals", 0.5)
    assert (p.frame_tolerance, p.settling_margin, p.contact_force_global) == (0.05, 0.25, None)
    assert definition.load_definition(definition_dir).qoi.tolerance == (0.01,)


@pytest.mark.parametrize(
    "extra, message",
    [
        ("increment_factors = [1.0]", "increment_factors"),
        ("frame_factor = 1.0", "frame_factor"),
        ("frame_factor = 0.3", "whole number"),  # fixed.n_intervals = 20 below
        ("frame_key = \"L\"", "sampled"),
        ("nonsense = 1", "unknown field"),
        ("contact_force_global = \"\"", "contact_force_global"),
    ],
)
def test_bad_pilot_probe_fields_are_refused(tmp_path, extra, message):
    toml = MINIMAL_TOML.replace("[fixed]\nE = 1000.0", "[fixed]\nE = 1000.0\nframe_interval = 1.0\nn_intervals = 20")
    toml = toml.replace('accepted_gaps = ["solver_identity_complete"]', f'accepted_gaps = ["solver_identity_complete"]\n{extra}')
    with pytest.raises(definition.DefinitionError, match=message):
        definition.load_definition(write_definition(tmp_path / "d", toml=toml))


def test_qoi_tolerance_is_one_positive_number_per_name(tmp_path):
    toml = MINIMAL_TOML.replace('units = ["m"]', 'units = ["m"]\ntolerance = [0.02, 0.02]')
    with pytest.raises(definition.DefinitionError, match="qoi.tolerance"):
        definition.load_definition(write_definition(tmp_path / "d", toml=toml))
    toml = MINIMAL_TOML.replace('units = ["m"]', 'units = ["m"]\ntolerance = [0.02]')
    assert definition.load_definition(write_definition(tmp_path / "e", toml=toml)).qoi.tolerance == (0.02,)


def test_contact_force_global_drops_the_global_prefix(tmp_path):
    toml = MINIMAL_TOML.replace(
        'accepted_gaps = ["solver_identity_complete"]',
        'accepted_gaps = ["solver_identity_complete"]\ncontact_force_global = "global/reaction_force_2_reference_node"',
    )
    p = definition.load_definition(write_definition(tmp_path / "d", toml=toml)).pilot
    assert p.contact_force_global == "reaction_force_2_reference_node"
```

Append to `tests/datagen/test_deck.py`:

```python
def test_with_all_energy_widens_the_standard_request_and_nothing_else():
    body = deck.standard_output(1e-6, node_set="N", node_vars=("U",), element_set="E", element_vars=("S",))
    text = deck.heading("x") + deck.explicit_step("S", 1e-3, body)
    widened = deck.with_all_energy(text)
    assert "*ENERGY OUTPUT, VARIABLE=ALL\n" in widened
    assert ", ".join(deck.ENERGY_TERMS) not in widened
    assert widened.replace("*ENERGY OUTPUT, VARIABLE=ALL\n", "*ENERGY OUTPUT\n" + ", ".join(deck.ENERGY_TERMS) + "\n") == text
    with pytest.raises(ValueError, match="ENERGY OUTPUT"):
        deck.with_all_energy(deck.heading("no step"))
```

Append to `tests/core/test_abaqus_adapter.py`:

```python
def test_assembly_history_names_every_term_on_the_frame_clock(tmp_path):
    from structbench.core.io.abaqus import LEDGER_CLOSED_TERMS, assembly_history, read_abaqus_export

    path = tmp_path / "toy.npz"
    np.savez(path, **_arrays())
    history = assembly_history(read_abaqus_export(path))
    assert set(history) == set(_TERMS) and history["ALLKE"].shape == (2,)  # the duplicate end frame dropped
    assert {"ALLKE", "ALLIE", "ALLVD", "ALLWK", "ETOTAL"} <= LEDGER_CLOSED_TERMS
```

- [ ] **Step 2: Run them to see them fail**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/datagen/test_definition.py tests/datagen/test_deck.py tests/core/test_abaqus_adapter.py 2>&1 | tail -5`
Expected: FAIL — `Pilot` has no `increment_key`; `deck` has no `with_all_energy`; `ImportError: assembly_history`.

- [ ] **Step 3: Implement**

`definition.py`: the dataclass fields above; in `load_definition`, after `pt = _table(raw, "pilot")`: refuse unknown keys; read the optional fields with `pt.get(...)` through `_field` when present; apply the rules; `sampled = set(variables) | {k for s in splits for k in s.extra}` for the constant check; `[qoi]`: `tolerance = tuple(float(x) for x in q.get("tolerance", [0.01] * len(names)))`, refuse a length mismatch or a non-positive value; `contact_force_global`: `None` when the key is absent, refused when empty (a dataset that means "none" omits the key), and a value starting with `global/` loses the prefix.

`deck.py`:
```python
STANDARD_ENERGY_REQUEST = "*ENERGY OUTPUT\n" + ", ".join(ENERGY_TERMS) + "\n"
ALL_ENERGY_REQUEST = "*ENERGY OUTPUT, VARIABLE=ALL\n"


def with_all_energy(text: str) -> str:
    """The deck with ``standard_output``'s energy request widened to every term.

    The conformance run of the preflight: the ledger identity must close with
    the standard terms and no other term may be non-zero (design, step 1).
    """
    if STANDARD_ENERGY_REQUEST not in text:
        raise ValueError("no *ENERGY OUTPUT block written by standard_output in the deck")
    return text.replace(STANDARD_ENERGY_REQUEST, ALL_ENERGY_REQUEST, 1)
```
and `standard_output` uses `STANDARD_ENERGY_REQUEST`.

`core/io/abaqus.py`: rename `_LEDGER_CLOSED` to `LEDGER_CLOSED_TERMS` (keep `_LEDGER_CLOSED = LEDGER_CLOSED_TERMS`), add to `__all__`; extract from `abaqus_ledger` the dict comprehension over `history/<step>/Assembly*/<term>` into `assembly_history(export)`; `abaqus_ledger` calls it.

- [ ] **Step 4: Run the tests** — Run: same command. Expected: PASS.

- [ ] **Step 5: Gates and commit**

```bash
cd <repo> && set -o pipefail && <venv>/ruff check . && <venv>/ruff format --check . && <venv>/mypy src && <venv>/python -m pytest -q -p no:cacheprovider 2>&1 | tail -3 && git add -A src tests && git commit -q -m "$(cat <<'EOF'
feat(datagen): the preflight's contract -- [pilot] probe fields with defaults, [qoi].tolerance, the conformance deck, the ledger's public names

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `verification/temporal.py` — the time and duration measures on `Case` objects

**Files:**
- Create: `src/structbench/verification/temporal.py`
- Test: `tests/verification/test_temporal.py`

**Interfaces:**
- Consumes: `structbench.core.Case`, `Response`; `verification.kernels.relative_l2_pooled`.
- Produces:
  ```python
  def response_until(case: Case, frame: int) -> Case
      # the case with frames [0, frame] of every response array; ValueError outside range
  def qoi_history(case: Case, qoi: Callable[[Case], Mapping[str, float]]) -> dict[str, NDArray[np.float64]]
      # qoi() on response_until(case, k) for every k; one array (T,) per name
  def settling_frame(history: Mapping[str, ArrayLike], tolerance: Mapping[str, float]) -> int
      # the last frame k with |q[k] - q[-1]| > tol * max(|q[-1]|, floor) for any q; 0 if none; floor = 1e-300
  def separation_frame(force: ArrayLike, *, threshold: float = 1e-3) -> int | None
      # the last frame with |f| > threshold * max|f|; None when the peak is 0
  def rise_time_frames(series: ArrayLike) -> int | None
      # frames from the first crossing of 10 % to the first crossing of 90 % of max|value|; None for a flat or a falling series
  def midpoint_interpolation_errors(case: Case) -> dict[str, float]
      # a case of 2n+1 frames: linear interpolation of the even frames at the odd frames vs the stored odd frames,
      # relative_l2_pooled per field; keys "node/<f>", "<block>/<f>", "global/<g>"; ValueError on an even frame count
  def common_instant_errors(coarse: Case, fine: Case) -> dict[str, float]
      # stride = (len(fine.time)-1) / (len(coarse.time)-1) must be a whole number >= 1; fine[::stride] vs coarse per field,
      # same keys; fields missing on either side are skipped; ValueError when the clocks do not align
  ```

- [ ] **Step 1: Write the failing tests**

`tests/verification/test_temporal.py` builds cases with the same `Case` constructor pattern as `tests/datagen/test_converge.py::_case` (one `Material`, a 2×2 quad mesh, float32 response arrays). Tests:

```python
def test_response_until_keeps_frames_up_to_and_including_the_one_named():
    c = _smooth_case(frames=5)
    t = temporal.response_until(c, 2)
    assert t.response.time.shape == (3,) and t.response.node["displacement"].shape[0] == 3
    with pytest.raises(ValueError):
        temporal.response_until(c, 5)


def test_settling_frame_is_the_last_frame_outside_tolerance():
    hist = {"q": np.array([0.0, 0.5, 0.9, 0.99, 0.999, 1.0])}
    assert temporal.settling_frame(hist, {"q": 0.05}) == 2   # 0.9 is 10 % off, 0.99 is 1 % off
    assert temporal.settling_frame(hist, {"q": 0.5}) == 1
    assert temporal.settling_frame({"q": np.ones(4)}, {"q": 0.01}) == 0


def test_qoi_history_evaluates_the_hook_frame_by_frame():
    c = _smooth_case(frames=5)  # displacement x grows 0, 1, 2, 3, 4
    hist = temporal.qoi_history(c, lambda case: {"ux": float(case.response.node["displacement"][-1, 0, 0])})
    assert hist["ux"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_separation_frame_and_rise_time():
    force = np.array([0.0, 4.0, 10.0, 6.0, 0.005, 0.0, 0.0])
    # threshold 1e-3 * 10 = 0.01; frames above it are 1, 2, 3; 0.005 at frame 4 is below
    assert temporal.separation_frame(force) == 3
    assert temporal.separation_frame(np.zeros(3)) is None
    # 10 % of the peak (1.0) is first crossed at frame 2, 90 % (9.0) at frame 3
    assert temporal.rise_time_frames(np.array([0.0, 0.5, 2.0, 9.5, 10.0])) == 1
    assert temporal.rise_time_frames(np.array([10.0, 5.0, 0.0])) is None
```

```python
def test_midpoint_interpolation_error_is_zero_for_a_linear_field_and_positive_for_a_curved_one():
    linear = _case_with_displacement(lambda t: t)              # 5 frames, u = t
    curved = _case_with_displacement(lambda t: t * t)
    assert temporal.midpoint_interpolation_errors(linear)["node/displacement"] == pytest.approx(0.0, abs=1e-6)
    assert temporal.midpoint_interpolation_errors(curved)["node/displacement"] > 0.01
    with pytest.raises(ValueError, match="odd"):
        temporal.midpoint_interpolation_errors(_smooth_case(frames=4))


def test_common_instant_errors_align_by_stride_and_refuse_misaligned_clocks():
    coarse = _case_with_displacement(lambda t: t, frames=3)     # t = 0, .5, 1
    fine = _case_with_displacement(lambda t: t, frames=5)       # t = 0, .25, .5, .75, 1
    assert temporal.common_instant_errors(coarse, fine)["node/displacement"] == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError, match="stride"):
        temporal.common_instant_errors(coarse, _case_with_displacement(lambda t: t, frames=4))
```

- [ ] **Step 2: Run to see them fail** — `ModuleNotFoundError: structbench.verification.temporal`.

- [ ] **Step 3: Implement** `temporal.py` per the interfaces; `response_until` uses `dataclasses.replace` on `Response` with every array sliced `[: frame + 1]` (node, element per block, globals, time); `midpoint_interpolation_errors` computes `pred = 0.5 * (a[0:-1:2] + a[2::2])`, `gt = a[1::2]` per array; `common_instant_errors` computes `stride` from the time lengths and checks `np.allclose(fine.time[::stride], coarse.time)`.

- [ ] **Step 4: Run the tests plus the boundary test** — Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/verification/test_temporal.py tests/test_verification_import_boundary.py 2>&1 | tail -3`. Expected: PASS.

- [ ] **Step 5: Gates and commit** — as Task 1, message `feat(verification): temporal measures -- settling, separation, rise time, midpoint-interpolation and common-instant errors on cases`.

---

### Task 3: `converge` compares probe-only groups when told which split anchors them

**Files:**
- Modify: `src/structbench/datagen/converge.py` (`pair_levels`, `converge`, `main`, `render_markdown`)
- Test: `tests/datagen/test_converge_anchor.py`

**Interfaces:**
- Consumes: `pair_levels`, `converge`, `LevelRun`, the `Sweep` helper of `tests/datagen/test_converge.py`.
- Produces:
  ```python
  def pair_levels(defn, roots, *, anchor: str | None = None, cases: Collection[str] | None = None)
  def converge(defn, problem, roots, *, anchor: str | None = None, cases: Collection[str] | None = None) -> dict
  # CLI: --anchor SPLIT, --cases ID [ID ...]; the record gains "anchor": anchor
  ```
  Semantics: a run of split `anchor` at `[levels].production` stands as the production run of a group that has no true production run (a true one still wins); the group's case id is that run's; no "probe standing at the production level" note for it. Without `anchor`, groups with runs but no production run are counted into one note: `"<n> group(s) have runs from probe splits only and no production run; name a split with --anchor to compare them"`. `cases` keeps only the listed case ids.

- [ ] **Step 1: Write the failing tests**

```python
"""converge: a probe split may anchor groups that have no production run (plan 2b)."""

import pytest
from conftest import write_definition
from test_converge import PROBLEM, TOML, Sweep

pytest.importorskip("scipy")

from structbench.datagen import converge, definition  # noqa: E402


@pytest.fixture
def toy(tmp_path):
    ds = write_definition(tmp_path / "ds", toml=TOML, problem=PROBLEM)
    return definition.load_definition(ds), definition.load_problem(ds)


def _probe_only(tmp_path, problem):
    s = Sweep(tmp_path / "runs", problem)
    s.run("TOY-conv-0000-L1", "conv", "1").run("TOY-conv-0000-L2", "conv", "2")
    s.run("TOY-conv-0000-L4", "conv", "4")
    return s


def test_without_an_anchor_probe_only_groups_are_counted_not_compared(tmp_path, toy):
    defn, problem = toy
    record = converge.converge(defn, problem, [_probe_only(tmp_path, problem).root])
    assert record["cases"] == []
    assert any("no production run" in n and "--anchor" in n for n in record["notes"])


def test_the_anchor_splits_run_at_the_production_level_stands_for_production(tmp_path, toy):
    defn, problem = toy
    record = converge.converge(defn, problem, [_probe_only(tmp_path, problem).root], anchor="conv")
    (entry,) = record["cases"]
    assert entry["case_id"] == "TOY-conv-0000-L2" and entry["production_level"] == "2"
    assert entry["extrapolation"]["length"]["status"] == "monotone"
    assert not any("probe" in n for n in entry["notes"])
    assert record["anchor"] == "conv"


def test_a_true_production_run_still_wins_over_the_anchor(tmp_path, toy):
    defn, problem = toy
    s = _probe_only(tmp_path, problem).run("TOY-train-0000", "train", "2")
    (entry,) = converge.converge(defn, problem, [s.root], anchor="conv")["cases"]
    assert entry["case_id"] == "TOY-train-0000" and entry["levels"]["2"] == "TOY-train-0000"


def test_cases_restricts_the_pairing(tmp_path, toy):
    defn, problem = toy
    s = _probe_only(tmp_path, problem)
    record = converge.converge(defn, problem, [s.root], anchor="conv", cases={"TOY-conv-0000-L1", "TOY-conv-0000-L2"})
    (entry,) = record["cases"]
    assert set(entry["levels"]) == {"1", "2"} and entry["extrapolation"]["length"] is None


def test_the_cli_takes_the_anchor(tmp_path, toy, capsys):
    defn, problem = toy
    s = _probe_only(tmp_path, problem)
    ds = tmp_path / "ds"
    assert converge.main(["--dataset", str(ds), "--sweep", str(s.root), "--anchor", "conv"]) == 0
```

- [ ] **Step 2: Run to see them fail** — `TypeError: unexpected keyword argument 'anchor'`.

- [ ] **Step 3: Implement** — in `pair_levels`: skip runs whose id is not in `cases` when given; after grouping, `production = [r for r in runs if r.production]`; if none and `anchor`: `stand_ins = sorted((r for r in runs if r.split == anchor and r.level == defn.levels.production), key=(root_index, case_id))`; if a stand-in exists, treat it as production (`dataclasses.replace(r, production=True)` before the per-level selection) and remember its id for the note filter; if none, `skipped += 1`. `_one_case`'s "probe standing at production level" note: `_one_case` receives `by_level` only; pass the anchored id through `LevelRun.production=True`, so the existing note logic (which keys on `production`) stays silent for it. The record gains `"anchor"`, `render_markdown` prints `Anchor split: <name>` when set.

- [ ] **Step 4: Run** the new file with `tests/datagen/test_converge.py tests/datagen/test_converge_review.py`. Expected: PASS.

- [ ] **Step 5: Gates and commit** — message `feat(datagen): converge --anchor names the probe split whose production-level run anchors probe-only groups; --cases restricts the pairing`.

---

### Task 4: `generate` — the shared materialiser, provenance 3 with the preflight record, the gate; `export` becomes a module

**Files:**
- Create: `src/structbench/datagen/export.py`
- Modify: `src/structbench/datagen/generate.py`, `src/structbench/datagen/cli.py`, `src/structbench/datagen/convert.py` (`dataset_id`)
- Test: `tests/datagen/test_generate.py` (add, and the format literal), `tests/datagen/test_cli.py` (unchanged, must pass), `tests/datagen/test_convert.py` (add)

**Interfaces:**
- Consumes: `definition.file_sha256`, `Definition`.
- Produces:
  ```python
  # generate.py
  PROVENANCE_FORMAT = "abaqus-provenance/3"
  STAMP_FILE = "preflight/stamp.json"        # relative to the sweep directory

  @dataclass(frozen=True)
  class CaseSpec:
      ...                                    # as before, plus
      probe: dict[str, Any] | None = None    # preflight cases: {"role": "level"|"increment"|"frame"|"conformance", ...}

  def case_id_for(defn: Definition, split: str, index: int, variant: str | None, suffix: str = "") -> str
      # f"{prefix}-{split}-{index:04d}" + (f"-{variant}" if variant) + suffix; ValueError unless _CASE_ID matches

  def materialise(
      sweep_dir: Path, specs: Sequence[CaseSpec], *, defn: Definition, problem: ModuleType, dataset_dir: Path,
      force: bool = False, preflight: dict[str, Any] | None = None,
      decks: Mapping[str, str] | None = None,           # case_id -> deck text; built from problem.input_deck when absent
  ) -> tuple[Counter[str], list[str]]                   # outcomes, problems ("<id>: conflict|locked")
      # writes provenance with "format", ..., "probe": spec.probe, "preflight": preflight if the split is not a probe else None

  def read_stamp(sweep_dir: Path) -> dict[str, Any] | None
  def stamp_refusal(stamp: dict | None, definition_sha: str, problem_sha: str) -> str | None
      # None when the stamp passed for both hashes; else "no preflight stamp" | "the preflight stamp did not pass"
      # | "the preflight stamp is for another dataset.toml" | "... another problem.py"
  # main: --no-preflight; the gate before any deck is written

  # export.py
  EXPORTER
  def export_command(abaqus: str, sweep: Path, cases: Sequence[str] | None) -> list[str]
  def export_cases(sweep: Path, abaqus: str, *, cases: Sequence[str] | None = None,
                   exporter_args: Sequence[str] | None = None) -> int    # 2 when the executable is not on PATH
  def main(argv: list[str] | None) -> int

  # convert.py
  def convert_sweep(sweep, *, splits=None, out=None, dataset_id: str | None = None) -> ConvertReport
  ```
  `cli.py` keeps `export_command` as a re-export (`from structbench.datagen.export import export_command as export_command`) and dispatches `"export": export.main`.

The gate in `generate.main`, after `selected` and the dry-run branch:
```python
probe_of = {s.name: s.probe for s in defn.splits}
guarded = sorted({s.split for s in selected if not probe_of[s.split]})
preflight: dict[str, Any] | None = None
if guarded:
    if args.no_preflight:
        preflight = {"skipped": True}
    else:
        why = stamp_refusal(read_stamp(sweep_dir), definition_sha, problem_sha)
        if why:
            print(f"{', '.join(guarded)}: {why}; run `structbench-datagen preflight --dataset {dataset_dir.name} --work-root ...`, or pass --no-preflight to record the omission", file=sys.stderr)
            return 2
        stamp_path = sweep_dir / STAMP_FILE
        preflight = {"stamp_sha256": file_sha256(stamp_path), "created_utc": read_stamp(sweep_dir)["created_utc"]}
```
Every existing `generate` test that runs a non-probe split gains `"--no-preflight"` in its arguments (the toy sweeps have no stamp) — that change is part of Step 1, and one new test keeps the default path refusing.

- [ ] **Step 1: Write the failing tests**

In `tests/datagen/test_generate.py`: `_run(ds, work, *extra)` passes `"--no-preflight"` by default (`_run(ds, work, *extra, gate=False)`); the format literal becomes `"abaqus-provenance/3"`; add:

```python
def _stamp(work, ds, *, passed=True, definition_sha=None, problem_sha=None):
    sweep = work / "toy"
    (sweep / "preflight").mkdir(parents=True)
    stamp = {
        "format": "preflight-stamp/1", "passed": passed, "created_utc": "2026-09-27T00:00:00+00:00",
        "definition_sha256": definition_sha or definition.load_definition(ds).sha256(),
        "problem_sha256": problem_sha or definition.problem_sha256(ds),
    }
    (sweep / "preflight" / "stamp.json").write_text(json.dumps(stamp), encoding="utf-8")


def test_production_splits_are_refused_without_a_passing_stamp(tmp_path, capsys):
    ds, work = _dataset(tmp_path), tmp_path / "work"
    assert _run(ds, work, gate=True) == 2
    err = capsys.readouterr().err
    assert "no preflight stamp" in err and "main" in err and "--no-preflight" in err
    assert not (work / "toy" / "TOY-main-0000").exists()
    _stamp(work, ds, passed=False)
    assert _run(ds, work, gate=True) == 2 and "did not pass" in capsys.readouterr().err
    _stamp(work, ds, problem_sha="0" * 64)
    assert _run(ds, work, gate=True) == 2 and "another problem.py" in capsys.readouterr().err


def test_probe_splits_are_never_gated(tmp_path):
    ds, work = _dataset(tmp_path), tmp_path / "work"
    assert _run(ds, work, "--split", "pilot", gate=True) == 0
    prov = json.loads((work / "toy/TOY-pilot-0000/provenance.json").read_text())
    assert prov["preflight"] is None and prov["probe"] is None


def test_a_passing_stamp_is_recorded_in_every_guarded_case(tmp_path):
    ds, work = _dataset(tmp_path), tmp_path / "work"
    _stamp(work, ds)
    assert _run(ds, work, gate=True) == 0
    prov = json.loads((work / "toy/TOY-main-0000/provenance.json").read_text())
    assert set(prov["preflight"]) == {"stamp_sha256", "created_utc"}
    assert prov["preflight"]["stamp_sha256"] == definition.file_sha256(work / "toy/preflight/stamp.json")


def test_no_preflight_records_the_omission(tmp_path):
    ds, work = _dataset(tmp_path), tmp_path / "work"
    assert _run(ds, work) == 0
    prov = json.loads((work / "toy/TOY-main-0000/provenance.json").read_text())
    assert prov["preflight"] == {"skipped": True} and prov["format"] == "abaqus-provenance/3"


def test_case_id_for_appends_a_suffix_within_the_job_name_rule(tmp_path):
    defn = definition.load_definition(_dataset(tmp_path))
    assert generate.case_id_for(defn, "pilot", 3, None, "-L2") == "TOY-pilot-0003-L2"
    assert generate.case_id_for(defn, "pilot", 3, "x", "-T0p5") == "TOY-pilot-0003-x-T0p5"
    with pytest.raises(ValueError, match="job name"):
        generate.case_id_for(defn, "pilot", 3, None, "-" + "x" * 40)
```

In `tests/datagen/test_cli.py`, the three export tests stay as they are (they exercise the re-export and the dispatch). In `tests/datagen/test_convert.py`:

```python
def test_dataset_id_defaults_to_the_sweep_name_and_can_be_given(tmp_path):
    sweep = tmp_path / "toy_sweep"
    _case(sweep, "T-0000")
    convert.convert_sweep(sweep, dataset_id="toy")
    assert read_case(sweep / "canonical" / "T-0000.h5").metadata.dataset_id == "toy"
```
(Check the metadata attribute's real name in `core` before writing this assertion.)

- [ ] **Step 2: Run to see them fail** — `unrecognized arguments: --no-preflight`; `AttributeError: case_id_for`.

- [ ] **Step 3: Implement** — extract the loop of `generate.main` into `materialise` (it computes `dataset_repo`, `package_state()`, the hashes, and returns the counts and problems; `main` prints the dirty warning and the counts); `plan_cases` uses `case_id_for`; the gate as above; `export.py` from `cli.py`'s `EXPORTER`, `export_command`, `_export`; `cli.py` imports them. `convert_sweep`'s `dataset_id` flows into `abaqus_export_to_case`.

- [ ] **Step 4: Run** `tests/datagen/test_generate.py tests/datagen/test_cli.py tests/datagen/test_convert.py tests/test_datagen_import_boundary.py`. Expected: PASS.

- [ ] **Step 5: Gates and commit** — message `feat(datagen): generate is gated on the preflight stamp (--no-preflight records the omission), provenance 3, the shared materialiser; export is a module`.

---

### Task 5: The runner's budget — a free-space margin, a clean stop with exit 3, the estimate

**Files:**
- Modify: `src/structbench/datagen/run.py`
- Test: `tests/datagen/test_run_budget.py`

**Interfaces:**
- Consumes: `generate.read_stamp` (import inside `main` only — `run.py` stays importable without `generate`? No: `datagen` may import itself; import at module top is fine).
- Produces:
  ```python
  def free_gb(path: Path) -> float                       # shutil.disk_usage(path).free / 1e9
  def estimate_line(stamp: dict[str, Any], n_cases: int, workers: int, free: float) -> str
      # "estimate from the preflight: 500 cases ≈ 3.2 h at 6 workers, ≈ 41.7 GB; free 170.3 GB (margin 20 GB)"
      # from stamp["budget"]: wall_s_median, bytes_per_case, min_free_gb
  def run_sweep(..., min_free_gb: float | None = None, echo=print) -> list[JobResult]
      # before each launch: free_gb(sweep) < min_free_gb -> no more launches; that job's result is
      # JobResult(id, "not_launched", 0.0, None) with no run.json and no run_log row; running jobs finish normally
  # main: --min-free-gb FLOAT (default: the stamp's budget.min_free_gb when <sweep>/preflight/stamp.json exists);
  #       prints estimate_line when a stamp exists; exit 3 when any result is not_launched
  ```
  `_Control` gains `halted: threading.Event`; `_run_one` takes `min_free_gb` and checks it under `control.guard` before `Popen`.

- [ ] **Step 1: Write the failing tests**

```python
"""The runner's budget (plan 2b): the free-space margin, the clean stop, the estimate."""

import json
import sys

import pytest
from test_run import FAKE, _case

from structbench.datagen import run as run_jobs


@pytest.fixture
def abaqus(tmp_path):
    fake = tmp_path / "fake_abaqus.py"
    fake.write_text(FAKE, encoding="utf-8")
    return [sys.executable, str(fake)]


def test_a_low_disk_stops_launching_and_leaves_pending_cases_pending(tmp_path, abaqus, monkeypatch):
    sweep = tmp_path / "sweep"
    for k in range(3):
        _case(sweep, f"J-ok-{k}")
    free = iter([100.0, 0.1, 0.1])  # the first launch sees room, the next two do not
    monkeypatch.setattr(run_jobs, "free_gb", lambda path: next(free))
    results = run_jobs.run_sweep(sweep, abaqus, workers=1, min_free_gb=5.0, echo=lambda s: None)
    by = {r.case_id: r.status for r in results}
    assert list(by.values()).count("completed") == 1 and list(by.values()).count("not_launched") == 2
    pending = [d.name for d in sorted(sweep.iterdir()) if d.is_dir() and run_jobs.case_state(d) == "pending"]
    assert len(pending) == 2 and not (sweep / run_jobs.LOCK_NAME).exists()
    rows = (sweep / "run_log.csv").read_text().splitlines()
    assert len(rows) == 2  # header + the one run


def test_main_exits_three_on_a_disk_stop_and_reads_the_margin_from_the_stamp(tmp_path, abaqus, monkeypatch, capsys):
    sweep = tmp_path / "sweep"
    _case(sweep, "J-ok-0")
    (sweep / "preflight").mkdir()
    stamp = {"budget": {"min_free_gb": 5.0, "wall_s_median": 30.0, "bytes_per_case": 2e8, "production_cases": 10}}
    (sweep / "preflight" / "stamp.json").write_text(json.dumps(stamp))
    monkeypatch.setattr(run_jobs, "free_gb", lambda path: 0.1)
    monkeypatch.setattr(run_jobs.shutil, "which", lambda name: abaqus[1])
    monkeypatch.setattr(run_jobs, "_solver", lambda exe: abaqus)  # see Step 3: main builds the command through _solver
    assert run_jobs.main(["--sweep", str(sweep), "--workers", "1"]) == 3
    out = capsys.readouterr().out
    assert "estimate from the preflight" in out and "margin 5" in out and "below 5" in out


def test_estimate_line_reads_the_budget():
    stamp = {"budget": {"min_free_gb": 20.0, "wall_s_median": 36.0, "bytes_per_case": 5e7}}
    line = run_jobs.estimate_line(stamp, 500, 6, 170.3)
    assert "500 cases" in line and "0.8 h at 6 workers" in line and "25.0 GB" in line and "170.3 GB" in line
```
(`main` currently runs `[exe]`; the fake solver needs `[python, script]`, so `main` gets a one-line seam `_solver(exe) -> list[str]` returning `[exe]`, which the test replaces. Alternative without a seam: write a `.bat`/shell wrapper — brittle on Windows; the seam is the smaller change.)

- [ ] **Step 2: Run to see them fail** — `TypeError: unexpected keyword argument 'min_free_gb'`.

- [ ] **Step 3: Implement** per the interfaces: in `_run_one`, inside `with control.guard:` after the `stopping` check:
```python
if control.halted.is_set():
    return _held(case_dir, case_id)
if min_free_gb is not None and free_gb(case_dir) < min_free_gb:
    control.halted.set()
    return _held(case_dir, case_id)
```
where `_held` closes and removes `runner.log` like the `stopping` branch (restructure that branch into `_held`). In the result loop, on the first `not_launched`: `echo(f"free space below {min_free_gb:g} GB: launching nothing more; running jobs finish")`; `_append_log` is skipped for `not_launched`. `main`: `--min-free-gb`, the stamp read (`generate.read_stamp(args.sweep)`), the estimate line printed after `chosen` is known — `run_sweep` gets an `estimate: str | None` to echo before the lock? Simpler: `main` counts the candidates with `dry_run=True` first? No — double work. `run_sweep` gains `stamp: dict | None = None` and echoes `estimate_line(stamp, len(chosen), workers, free_gb(sweep))` before launching. Exit: `3 if any(r.status == "not_launched") else 1 if any not completed else 0`.

- [ ] **Step 4: Run** `tests/datagen/test_run_budget.py tests/datagen/test_run.py`. Expected: PASS.

- [ ] **Step 5: Gates and commit** — message `feat(datagen): run stops launching below the free-space margin and exits 3; the estimate from the preflight stamp`.

---

### Task 6: `preflight.py` — the case set, the ten steps as pure functions, the stamp and the report

**Files:**
- Create: `src/structbench/datagen/preflight.py`
- Modify: `src/structbench/datagen/verify.py` (`judge_sweep`)
- Test: `tests/datagen/test_preflight.py`

**Interfaces:**
- Consumes: Task 1's `Pilot`/`QoiSpec`/`with_all_energy`/`assembly_history`/`LEDGER_CLOSED_TERMS`; Task 2's measures; Task 3's `converge(..., anchor, cases)`; Task 4's `CaseSpec.probe`, `case_id_for`, `materialise`, `export_cases`; Task 5's `run_sweep(min_free_gb=...)`; `template.deck_sha256_in_fresh_interpreter`, `template.check_definition`; `verification.criteria.judge`, `CheckResult`, `Verdict`.
- Produces:
  ```python
  STAMP_FORMAT = "preflight-stamp/1"
  PREFLIGHT_DIR = "preflight"
  ROLES = ("level", "increment", "frame", "conformance")
  STEPS = ("deck_regression", "feasibility", "conformance", "space", "increment", "frame",
           "duration", "energy", "budget", "verification")           # the order the report prints
  ENERGY_ROWS = ("energy_gain_max", "energy_loss_max", "energy_residual_final", "kinetic_energy_closure",
                 "plastic_dissipation_excess_max", "stored_globals_match_ledger")

  @dataclass(frozen=True)
  class Step:
      name: str
      verdict: str            # pass | fail | review | not_applicable | not_assessable
      summary: str            # one sentence
      detail: dict[str, Any]  # numbers, per case; JSON-safe

  def label_suffix(role: str, label: str | float) -> str
      # "-L2", "-T0p5", "-F", "-E": str(label).replace(".", "p"); "-" and "." are the only characters translated
  def preflight_cases(defn: Definition) -> list[CaseSpec]
      # for each pilot point × variant: role "level" at every [levels].pilot level except the finest, always at
      # production, and at the finest too when the plain id is in fine_cases; role "increment" at production for
      # each factor with params[increment_key] = production_value * factor (production value = fixed[increment_key]
      # or 1.0); role "frame" (one case: the first fine case, else the first pilot) at production with
      # params[frame_key] *= frame_factor and params[frame_count_key] = round(count / frame_factor), only when
      # frame_factor > 0 and both keys are in [fixed]; role "conformance" (the first pilot) at production.
      # spec.probe = {"role", "level", "factor"} (factor None except increment/frame); ids via case_id_for(..., suffix)
  def production_value(defn: Definition) -> float
  def deck_for(spec: CaseSpec, problem: ModuleType) -> str          # with_all_energy for role conformance
  def by_role(specs: Sequence[CaseSpec], role: str) -> list[CaseSpec]
  def batch_a(specs) -> list[str]     # ids of the production-level "level" cases and the conformance case

  def step_deck_regression(dataset_dir: Path, specs, decks: Mapping[str, str], defn) -> Step
      # (a) every spec: deck_sha256_in_fresh_interpreter(dataset_dir, spec.params) == sha256(problem deck);
      #     conformance case compared on the plain deck (the transformation is the writer's);
      # (b) probe keys are read: for one pilot, the level decks differ pairwise, the increment decks differ from
      #     the production deck, the frame deck differs from it -> else fail: "'dt_scale' leaves the deck unchanged: problem.input_deck does not read it"
  def step_feasibility(specs, states: Mapping[str, str], defn, problem) -> Step
      # every batch-A case "done"; detail: the [limits] values and feasible(params) per pilot
  def step_conformance(npz_path: Path | None, units: str) -> Step
      # None -> not_assessable "the conformance run has no export"; present: assembly_history: missing required
      # terms -> fail; terms outside LEDGER_CLOSED_TERMS with any non-zero value -> fail naming them; identity
      # residual max|Σ s_k T_k − total| / max|total| > 1e-5 -> fail; else pass; detail: terms, residual
  def step_space(record: Mapping[str, Any], defn) -> Step
      # over record["cases"]: per QoI with an extrapolation: monotone and error_vs_extrapolated[production] <= tol -> ok;
      # monotone above tol -> fail; another status -> review; no case with an extrapolation -> not_assessable
      # "no pilot ran at three levels"; detail: per case per QoI (status, order, error at production), the field summary
  def step_increment(qois: Mapping[str, Mapping[str, float]], fields: Mapping[str, Mapping[str, float]], specs, defn, report) -> Step
      # qois: case_id -> QoI values; per pilot per factor: |q_T − q_P| / max(|q_P|, floor) <= tol -> ok else fail;
      # energy rows of the increment cases in the verify report: any FAIL -> fail; no factors -> not_applicable;
      # detail per pilot per factor per QoI, the field errors (common_instant_errors, stride 1)
  def step_frame(frame_case: Case | None, production_case: Case | None, defn) -> Step
      # frame_factor == 0 -> not_applicable "disabled by the dataset"; keys absent -> not_assessable naming them;
      # midpoint_interpolation_errors(frame_case): node fields and globals <= frame_tolerance -> pass else fail;
      # detail: every field's error, common_instant_errors(production_case, frame_case), the shortest rise time
      # among the globals in production frames (rise_time_frames on frame_case[::stride])
  def step_duration(cases: Mapping[str, Case], specs, defn, problem) -> Step
      # for each production-level "level" case: settling_frame(qoi_history(case, problem.qoi), tol by name);
      # contact_force_global -> separation_frame(case.response.globals_[name]); share_after = 1 − settle/(T−1);
      # pass when max settle <= (1 − settling_margin) * (T − 1); detail per pilot, horizon (frames, time span)
  def step_energy(report, case_ids: Collection[str]) -> Step
      # ENERGY_ROWS over the named cases: any FAIL -> fail; else pass; detail: per row verdict counts and value spreads
  def step_budget(defn, problem, specs, runs: Mapping[str, Mapping[str, Any]], sizes: Mapping[str, int], free: float) -> Step
      # wall_s median/max per level from run.json["wall_s"]; bytes_per_case = median size of the production-level cases
      # (case folder + canonical file); production_cases = len([s for s in plan_cases(defn, feasible) if not probe]);
      # estimated_wall_h = production_cases * wall_s_median / 3600 (serial); estimated_gb = production_cases * bytes / 1e9;
      # pass when free >= estimated_gb + min_free_gb; detail is the stamp's "budget" table (plus min_free_gb)
  def step_verification(report, defn) -> Step
      # rows with verdict FAIL whose quantity is not in accepted_gaps -> fail listing "case: row"; else pass;
      # detail: the accepted gaps with their verdicts across cases
  def not_run(name: str, why: str) -> Step                # not_assessable
  def passed(steps: Sequence[Step]) -> bool               # all in {"pass", "not_applicable"}

  def stamp_record(defn, dataset_dir, steps: Sequence[Step], specs, *, created_utc: str) -> dict[str, Any]
      # {"format", "dataset", "definition_sha256", "problem_sha256", "structbench": package_state() without "dirty",
      #  "created_utc", "passed", "steps": {name: {verdict, summary, detail}}, "budget": steps["budget"].detail,
      #  "cases": {role: [ids]}}
  def render_report(stamp: Mapping[str, Any]) -> str      # Markdown; deterministic from the stamp
  def write_outputs(pre: Path, stamp: dict) -> None       # stamp.json (sorted, LF), report.md

  # verify.py
  def judge_sweep(sweep, dataset, *, splits=None, data_root=None, out=None) -> tuple[DatasetMeasurements, Any]
      # the record and judge(record); writes run_evidence.json, measurements.json, report.md into out
  def validate_sweep(...) -> int                          # unchanged behaviour, built on judge_sweep
  ```

- [ ] **Step 1: Write the failing tests** (`tests/datagen/test_preflight.py`; uses `conftest.MINIMAL_TOML`/`write_definition`, the `Case` builder pattern from `test_converge.py`, and a fake verify report built from `CheckResult` objects)

Tests to write, each with literal expected values derived by hand:
- `test_preflight_cases_cover_levels_factors_frame_and_conformance`: MINIMAL_TOML with `pilot = ["1","2","4"]`, `production = "2"`, `fine_cases = ["TOY-pilot-0000"]`, `[fixed]` gaining `dt_scale = 0.5`, `frame_interval = 1.0`, `n_intervals = 20`: ids are exactly `TOY-pilot-0000-L1, -L2, -L4, TOY-pilot-0001-L1, -L2, TOY-pilot-0000-T0p25, TOY-pilot-0001-T0p25, TOY-pilot-0000-F, TOY-pilot-0000-E`; the F case has `frame_interval == 0.5` and `n_intervals == 40`; every case's `refine` is a string level; `probe["role"]` as named.
- `test_preflight_cases_without_frame_keys_skip_the_frame_probe`: no `-F` case when `[fixed]` lacks the keys; `step_frame(None, None, defn)` is `not_assessable` naming `frame_interval`.
- `test_frame_probe_disabled_is_not_applicable`: `frame_factor = 0` → no `-F` case and `step_frame` → `not_applicable`.
- `test_production_value_defaults_to_one`: `[fixed]` without `dt_scale` → the T case has `dt_scale == 0.5` (1.0 × 0.5).
- `test_batch_a_is_the_production_level_cases_and_the_conformance_run`.
- `test_deck_regression_fails_when_a_probe_key_is_ignored`: MINIMAL_PROBLEM ignores `dt_scale` → `step_deck_regression` fails with `'dt_scale' leaves the deck unchanged`; a problem that writes `dt_scale` into the heading passes (the deck differs).
- `test_step_space_reads_the_convergence_record`: a hand-written record dict with one case, `status monotone`, `error_vs_extrapolated {"2": 0.004}` and tol 0.01 → pass; 0.02 → fail; `status oscillatory` → review; no cases → not_assessable.
- `test_step_increment_compares_qois_within_tolerance`: qois `{"...-L2": {"length": 1.0}, "...-T0p25": {"length": 1.004}}` → pass; 1.02 → fail; an empty `increment_factors` → not_applicable.
- `test_step_duration_measures_settling_and_separation`: a case whose displacement follows `1 − exp(−t/τ)` on 21 frames with τ = period/10 and a global `reaction_force_2_reference_node` that is a sine pulse over the first 30 %: settling frame ≈ 5, separation frame 6, share after ≈ 0.75 → pass at margin 0.25; margin 0.9 → fail.
- `test_step_frame_passes_a_smooth_response_and_fails_a_jagged_one`.
- `test_step_conformance_names_missing_and_extra_terms` (npz built from `tests/core/test_abaqus_adapter._arrays()` with a non-zero `ALLDMD` added → fail naming it; with the required terms only, and ETOTAL equal to KE+IE → pass).
- `test_step_energy_and_verification_read_the_report`: a fake report with one FAIL on `plastic_dissipation_excess_max` → energy fails and verification fails; the same FAIL on `solver_identity_complete` (accepted) → verification passes; energy passes.
- `test_step_budget_estimates_from_the_pilots`: runs with `wall_s` 30/40/50 at production, sizes 1e8 each, 10 production cases, free 100 → pass, estimated_gb 1.0, wall_s_median 40; free 1.5 with min_free_gb 5 → fail.
- `test_stamp_record_and_report_are_deterministic`: two renders of one record are equal; the JSON is sorted and ends with LF; `passed` false when any step is `review`.
- `test_judge_sweep_returns_the_report_validate_sweep_prints`: over the `_sweep` of `tests/datagen/test_verify.py` (import its helper the way that file imports the adapter fixture).

- [ ] **Step 2: Run to see them fail** — `ModuleNotFoundError: structbench.datagen.preflight`.

- [ ] **Step 3: Implement** `preflight.py` per the interfaces (the driver and `main` come in Task 7; keep this task to the pure functions, the stamp and the report); `judge_sweep` extracted from `validate_sweep`.

- [ ] **Step 4: Run** `tests/datagen/test_preflight.py tests/datagen/test_verify.py tests/test_datagen_import_boundary.py`. Expected: PASS.

- [ ] **Step 5: Gates and commit** — message `feat(datagen): the preflight's case set, its ten steps as pure functions on records, the stamp and the report`.

---

### Task 7: The preflight driver and CLI against a fake solver, end to end

**Files:**
- Create: `tests/datagen/fake_solver.py` (a script, not collected), `tests/datagen/test_preflight_e2e.py`
- Modify: `src/structbench/datagen/preflight.py` (`preflight`, `main`), `src/structbench/datagen/cli.py` (stage `preflight`)

**Interfaces:**
- Consumes: everything of Task 6; `run.run_sweep`, `export.export_cases`, `convert.convert_sweep`, `verify.judge_sweep`, `converge.converge`, `core.io.read_case`.
- Produces:
  ```python
  def preflight(dataset_dir: Path, work_root: Path, *, abaqus: str = "abaqus", workers: int = 2,
                timeout: float | None = None, exporter_args: Sequence[str] | None = None,
                echo: Callable[[str], None] = print) -> int
  def main(argv) -> int
  # structbench-datagen preflight --dataset <dir> --work-root <runs> [--abaqus EXE] [--workers N] [--timeout S]
  #                               [--exporter-args ...]   (the last hidden; tests)
  # exit 0 passed | 1 a step failed or needs review (stamp written, passed false) | 2 refused | 3 environment
  ```
  The driver, in order: load and `check_definition` (findings → print, 2) → `specs`, `decks` → `materialise(pre, specs, decks=decks, preflight=None)` (any `conflict`/`locked` → `"<pre> holds decks of another definition; move it aside (nothing is deleted)"`, 2) → `step_deck_regression` (fail → outputs with the rest `not_run`, 1) → `shutil.which(abaqus)` (None → 2) → `run_sweep(pre, solver, cases=batch_a, workers, timeout, min_free_gb=defn.pilot.min_free_gb, echo)` (any `not_launched` → 3) → `step_feasibility` → `export_cases(pre, exe, cases=batch_a, exporter_args)` → `step_conformance` (either fails → outputs, 1) → `run_sweep` on the rest (3 on a halt) → `export_cases` on the rest → `convert_sweep(pre, dataset_id=defn.name)` → `judge_sweep(pre, dataset_dir)` → `converge.converge(defn, problem, [pre], anchor=pilot split, cases=level ids)` written to `pre/converge/` → `read_case` for the production-level level cases, the increment cases and the frame case → steps space, increment, frame, duration, energy, budget, verification → `write_outputs` → 0 if `passed` else 1. Re-running after an interruption resumes: `run_sweep` skips done cases, `export` skips exported ones, `convert` skips converted ones.

**The fake solver** (`tests/datagen/fake_solver.py`, run as `python fake_solver.py job=ID input=ID.inp double=both cpus=1 interactive` from the case folder, and as `python fake_solver.py python <exporter> --sweep <dir> [--cases ...]`, which exits 0 because the fake writes the export itself):
- reads `provenance.json` (`params`: `refine`, `L`, `D`, `v0`, `frame_interval`, `n_intervals`) and the deck (`*NODE` blocks → labels and `(r, z)`; `*ELEMENT, TYPE=CAX4R` → connectivity; `VARIABLE=ALL` present or not);
- writes `.sta` (`" Abaqus/Explicit 2025\n THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n"`), `.msg` (`" Abaqus 2025\n      0 ERROR MESSAGES\n      0 WARNING MESSAGES\n"`), `.dat` (`" Abaqus 2025\n"`), an empty `.odb`, and `<id>.npz` in the `abaqus-npz/1` layout of `tests/core/test_abaqus_adapter._arrays()` (one instance `ROD-1`, the reference node in the mesh but not in the fields, the duplicate end frame);
- the response, with `k = int(refine)`, `bias = 1 + 0.02 / k**2`, `s(t) = 1 − exp(−t/τ)`, `τ = period / 10`, `period = frame_interval × n_intervals`, `R = D/2`, `δ = 0.1·L·(v0/1e5)·bias`, `ε = 0.1·R·bias`: `u_z = −δ·(z/L)·s(t)`, `u_r = ε·(1 − z/L)·s(t)`, `V` and `A` its analytic time derivatives, `S` (e, 4) = `200·bias·s(t)` in every component, `PEEQ` (e, 1) = `0.01·s(t)`; Assembly history: `ALLKE = K0·exp(−2t/τ)`, `ALLIE = K0 − ALLKE`, `ALLPD = 0.8·ALLIE`, `ALLSE = 0.2·ALLIE`, `ALLAE = ALLCD = ALLFD = ALLPW = ALLVD = ALLWK = 0`, `ETOTAL = K0`; when the deck says `VARIABLE=ALL`, also `ALLDMD = 0`; `Node ROD-1.<rp>/RF2 = F0·sin(π t / t_c)` for `t < t_c = 0.3·period`, else 0.
  So: QoIs converge at second order in `1/k` (levels 1, 2, 4 give `error_vs_extrapolated["2"]` ≈ 0.06 % of the final length), the response settles by ≈ 23 % of the horizon, contact ends at 30 %, the ledger closes exactly, no term outside the identity is non-zero, the increment factor changes nothing, and the half-interval frame probe interpolates an exponential to well under 5 %.

**The end-to-end test** copies the shipped example definition (`structbench.datagen.template.EXAMPLE_DIR`) into `tmp_path / "ds"`, edits its TOML (`name = "e2e"`, `L = 10.0`, `n_intervals = 20`, two pilot points, `pilot = ["1", "2", "4"]`, `production = "2"`, `fine_cases = ["ACX-pilot-0000"]`, `min_free_gb = 0.5`, `contact_force_global = "reaction_force_2_reference_node"`), `git init`s and commits it (as `test_generate._dataset` does), then:

```python
def test_the_preflight_passes_on_the_fake_solver_and_writes_a_stamp_generate_accepts(tmp_path, capsys):
    ds, work = _dataset(tmp_path), tmp_path / "work"
    rc = preflight.main(["--dataset", str(ds), "--work-root", str(work), "--abaqus", sys.executable,
                         "--exporter-args", str(FAKE), "--workers", "2"])
    pre = work / "e2e" / "preflight"
    stamp = json.loads((pre / "stamp.json").read_text(encoding="utf-8"))
    assert {n: s["verdict"] for n, s in stamp["steps"].items()} == {
        "deck_regression": "pass", "feasibility": "pass", "conformance": "pass", "space": "pass",
        "increment": "pass", "frame": "pass", "duration": "pass", "energy": "pass", "budget": "pass",
        "verification": "pass",
    }, (pre / "report.md").read_text(encoding="utf-8")
    assert rc == 0 and stamp["passed"]
    assert sorted(stamp["cases"]["level"]) == ["ACX-pilot-0000-L1", "ACX-pilot-0000-L2", "ACX-pilot-0000-L4",
                                              "ACX-pilot-0001-L1", "ACX-pilot-0001-L2"]
    assert (pre / "converge" / "convergence.json").is_file() and (pre / "datacheck" / "report.md").is_file()
    # the gate opens
    assert generate.main(["--dataset", str(ds), "--work-root", str(work), "--split", "train"]) == 0
    prov = json.loads((work / "e2e" / "ACX-train-0000" / "provenance.json").read_text())
    assert prov["preflight"]["stamp_sha256"] == definition.file_sha256(pre / "stamp.json")


def test_a_second_run_resumes_and_rewrites_the_same_verdicts(tmp_path):
    ... run once, delete pre / "stamp.json", run again: no new run.json mtimes, same verdicts, rc 0


def test_a_changed_definition_is_refused_and_nothing_is_deleted(tmp_path, capsys):
    ... run once (rc 0); append a comment-free change to problem.py (a different HEADING text) and commit;
    rc == 2, "move it aside" in stderr, every earlier case folder still has its run.json


def test_a_failing_step_writes_the_stamp_as_not_passed_and_the_gate_stays_shut(tmp_path):
    ... the fake solver is told (an environment variable FAKE_SOLVER_SETTLE="slow") to use τ = period, so the
    response never settles: duration fails, rc == 1, stamp["passed"] is False,
    generate on train exits 2 with "did not pass"
```

- [ ] **Step 1: Write the fake solver and the failing e2e tests** (the fake is test infrastructure, no TDD cycle of its own; each test above is watched to fail: `preflight` has no `main`).

- [ ] **Step 2: Run** `tests/datagen/test_preflight_e2e.py`. Expected: FAIL `AttributeError: main`.

- [ ] **Step 3: Implement** the driver and `main`; `cli.py` gets `"preflight": preflight.main` and a usage line. If a verification row fails on the fake's data for a reason that is the fake's (a field the declaration lists that the fake does not write, say), fix the fake; if it fails for an instrument reason unrelated to the data (a row that needs evidence no synthetic run has), add the row to the test TOML's `accepted_gaps` and ledger the ruling with the row's name and reason.

- [ ] **Step 4: Run** `tests/datagen/test_preflight_e2e.py tests/datagen/test_preflight.py tests/datagen/test_cli.py` — Expected: PASS, the four e2e tests within about a minute together.

- [ ] **Step 5: Gates and commit** — message `feat(datagen): structbench-datagen preflight -- the driver over run, export, convert, verify and converge; tested end to end against a fake solver`.

---

### Task 8: `follow` — export and convert finished cases while a run proceeds

**Files:**
- Create: `src/structbench/datagen/follow.py`
- Modify: `src/structbench/datagen/cli.py` (stage `follow`)
- Test: `tests/datagen/test_follow.py`

**Interfaces:**
- Consumes: `export.export_cases`, `convert.convert_sweep`, `run.LOCK_NAME`, `run.case_state`.
- Produces:
  ```python
  @dataclass
  class FollowReport:
      rounds: int = 0
      exported: list[str] = field(default_factory=list)      # ids handed to the exporter
      converted: list[str] = field(default_factory=list)
      failed: dict[str, str] = field(default_factory=dict)   # convert failures, and "export: rc N" once per round

  def to_export(sweep: Path, splits: Sequence[str] | None) -> list[str]
      # cases whose run.json says completed and that have no <id>.npz
  def follow(sweep: Path, abaqus: str, *, interval: float = 60.0, once: bool = False,
             splits: Sequence[str] | None = None, exporter_args: Sequence[str] | None = None,
             echo=print, sleep=time.sleep) -> FollowReport
      # loop: export what to_export lists (one exporter call), convert_sweep(sweep, splits=splits);
      # stop when once, or when the runner's lock is absent and nothing is left to export or convert;
      # else sleep(interval)
  def main(argv) -> int     # --sweep --abaqus --interval --once --split ...; hidden --exporter-args; 0 clean, 1 any failure, 2 no solver
  ```

- [ ] **Step 1: Write the failing tests** — a sweep built with `tests/datagen/test_convert._case(..., npz=False)`; a stub exporter script that, for each `--cases` id, writes the adapter fixture's arrays as `<id>.npz`; a fake `sleep` that, on its first call, writes a second case's `run.json` and on the second removes the lock:

```python
def test_follow_exports_then_converts_and_stops_when_the_runner_is_gone(tmp_path):
    sweep = _sweep_with(tmp_path, done=["T-0000"], running=["T-0001"])
    (sweep / run_jobs.LOCK_NAME).write_text("1")
    events = []
    def sleep(_):
        events.append("sleep")
        if len(events) == 1:
            _finish(sweep / "T-0001")            # its run.json appears
        else:
            (sweep / run_jobs.LOCK_NAME).unlink()
    report = follow.follow(sweep, sys.executable, interval=0, exporter_args=[str(STUB)], echo=lambda s: None, sleep=sleep)
    assert report.exported == ["T-0000", "T-0001"] and report.converted == ["T-0000", "T-0001"]
    assert report.rounds == 3 and report.failed == {}
    assert (sweep / "canonical" / "T-0001.h5").is_file()


def test_once_does_one_round_and_reports_failures(tmp_path):
    ... a case whose deck is unparsable -> convert fails -> report.failed names it, main returns 1


def test_main_without_a_solver_exits_two(tmp_path, capsys): ...
```

- [ ] **Step 2: Run to see them fail** — `ModuleNotFoundError: structbench.datagen.follow`.

- [ ] **Step 3: Implement**; `cli.py` gets `"follow": follow.main` and its usage line.

- [ ] **Step 4: Run** `tests/datagen/test_follow.py tests/datagen/test_cli.py`. Expected: PASS.

- [ ] **Step 5: Gates and commit** — message `feat(datagen): follow exports and converts finished cases while a run proceeds`.

---

### Task 9: Documentation, the example definition, the ADR note, the snapshot

**Files:**
- Modify: `docs/DATA_GENERATION.md` (the preflight section replaces "Until the stage exists, do these three by hand"; the `[pilot]` fields table with defaults and what each probe measures; the stamp and `--no-preflight`; `run`'s margin, estimate and exit 3; `follow`; the stage list in order `new check preflight generate run follow export convert verify converge archive`), `docs/ARCHITECTURE.md` (`verification/temporal.py` beside `convergence.py`; `datagen`'s stage list gains `preflight`, `follow`, `export`), `docs/plans/2026-09-27-abaqus-datagen-platform-design.md` (stage table: `preflight`, `generate`'s gate, `run`, `follow` built 2026-09-27; the `[pilot]` row lists the settled fields; the open point on the `[pilot]` fields closed with a dated line naming this plan), `decisions/0071-datagen-platform.md` (note "part two (b) built": the settled `[pilot]` fields and their defaults, the stamp format, exit code 3, the anchor split; what part three still owes), `src/structbench/datagen/examples/abaqus_conformance/dataset.toml` (the new `[pilot]` fields written out with comments, `contact_force_global = "reaction_force_2_reference_node"`, `[qoi].tolerance`), `docs/datagen/abaqus-lessons.md` (the checklist's resolution items point at `preflight`), `CLAUDE.md` (snapshot: part two (b) built; owed: part three only), `README.md` only if its stage list names stages.
- Test: `tests/test_docs.py` (existing checks must pass; add one: the example TOML loads and its pilot fields equal the documented defaults except `contact_force_global`), `tests/datagen/test_cli.py::test_every_stage_answers_help` covers the new stages by iterating `STAGES`.

- [ ] **Step 1: Write the failing doc test**, run it, see it fail (the example TOML lacks `contact_force_global`).
- [ ] **Step 2: Write the documents and the example**; scan for private detail (`git diff --cached | grep -i -E "<private name patterns>"` against the names listed in the maintainer's memory, never against the repo).
- [ ] **Step 3: Run** the full suite; gates; commit — message `docs: the preflight, its stamp and [pilot] fields, run's budget and follow in the guide, ARCHITECTURE, the design, ADR-0071 (part two (b) built) and the snapshot; the example declares its probes`.

---

### Task 10: The private definition under the new contract

**Files (the private dataset repository; `<dataset dir>` its dataset folder):**
- Modify: `<dataset dir>/dataset.toml` (`[qoi].tolerance` with values the maintainer stands behind — propose the tolerances the study used for its convergence target; `contact_force_global` naming the stored reaction-force global; the other `[pilot]` fields left to their defaults, which match the study's production increment scale and frame convention), `<dataset dir>/NOTES.md` (dated entry: the contract fields, and that the preflight was not run on the finished dataset — running it is a decision for the maintainer, with its cost written down)
- Test: the private suite (the definition loads; `structbench-datagen check` clean; the deck regression against provenance unchanged, since no deck changes)

- [ ] **Step 1:** run `check`; expect it clean before any edit (defaults).
- [ ] **Step 2:** add the fields; `check` again; the private suite; commit there with the plan's commit style.
- [ ] **Step 3:** nothing is pushed.

---

## Self-review

- **Spec coverage:** design "`preflight`" steps 1–7 → Task 6 (steps as functions) and Task 7 (the driver; step 1's Abaqus `input_requests_required_evidence` is part three and the report says so); the stamp and `generate`'s gate → Tasks 4, 6, 7; `run`'s free-space check, exit 3 and estimate → Task 5; `follow` → Task 8; the three resolutions of the ADR-0071 note → Tasks 1, 2, 6; the `[pilot]` fields' names and defaults (open point 4) → Task 1; the design's testing section (a stub solver through the whole chain) → Task 7; docs → Task 9; the private definition → Task 10. Not in this plan: the `card` stage and the Abaqus verification additions (part three).
- **Placeholders:** Task 6's and Task 7's later tests are named with their intent and their expected numbers rather than written out in full; the executor writes each from the interfaces and derives the literals by hand before running. Task 10 uses placeholders by the global constraint.
- **Type consistency:** `CaseSpec.probe` written by `preflight_cases`, read by `materialise` (provenance) and `by_role`; `Step` fields read by `stamp_record` and `render_report`; `stamp["budget"]` keys (`min_free_gb`, `wall_s_median`, `bytes_per_case`, `production_cases`) read by `run.estimate_line` and `run.main`; `stamp_refusal` reads `passed`, `definition_sha256`, `problem_sha256`; `converge(..., anchor, cases)` called by the driver with the pilot split and the level ids; `judge_sweep` returns what `step_energy` and `step_verification` read (`report.cases[i].case_id`, `.results[j].quantity/.verdict/.value`).
- **Review Focus:** 1 → Task 7; 2 → Task 4; 3 → Task 5; 4 → Tasks 1 and 6; 5 → Task 6.
