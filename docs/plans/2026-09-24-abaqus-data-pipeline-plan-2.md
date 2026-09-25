# Abaqus Data Pipeline — Plan 2 of 2: readers, adapter, verification, validate, archive

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn finished Abaqus/Explicit runs into verified canonical cases:
- read Explicit run records and 2D decks correctly;
- convert `abaqus-npz/1` exports into the canonical schema;
- measure them with the verification module, including sweeps that are not registered benchmarks;
- archive finished cases into the data tree.

**Architecture:**
- Package side: `core/io/abaqus.py` (the adapter), the extended readers in `core/io/abaqus_run.py`, one new material class, and three verification rows.
- Script side, in `data_generation/abaqus/`: `convert.py`, `collect_run_evidence.py`, `validate.py` and `archive.py`, all dataset-blind like Plan 1's scripts.
- Everything written here was either established by a real run (the Plan 1 conformance and pilot runs, 2026-09-24) or is pinned by a test on invented text in the same format.

**Tech Stack:** Python 3.12+, numpy, h5py (package); scipy (the `datagen` extra, already approved in ADR-0069); pytest, ruff, mypy.

**Spec:** `docs/plans/2026-09-24-abaqus-data-pipeline-design.md` (approved). Facts it relies on:
- `data_generation/abaqus/STANDARD_INPUT_BLOCK.md` → "Abaqus/Explicit (conformance run)";
- the private handoff note `scratch/2026-09-24-abaqus-plan-2-handoff.md`.

## Global Constraints

- **Branch** `feat/abaqus-pipeline-2`. **Interpreter** `PY=<venv>/Scripts/python.exe`.
- **Gates** before every commit (full suite; no CI):
  ```bash
  set -o pipefail; $PY -m ruff format --check . && $PY -m ruff check . && $PY -m mypy src && $PY -m pytest -q && $PY tools/gen_benchmark_docs.py --check
  ```
- **Public-repo rule** (CORRECTIONS 2026-08-12): fixtures use invented text and values in the real format. No study values, split names, dataset names or licence text.
- **ADR-0068 clause 3:** a stored deck is parsed only by its own solver's reader. Nothing is sniffed.
- **ADR-0066:**
  - `measure` stays threshold-free.
  - New verdicts come only from definitions or instrument tolerances.
  - Energy rows stay unratified (the maintainer's 2026-09-21 decision; the Plan 1 spec, Decision 5).
- **Existing records.** Published LS-DYNA records (`docs/datachecks/*.json`) must be unchanged by any measure change. Where Taylor or notch canonical data is reachable, check that with the env-gated acceptance tests. Otherwise rule and ledger.
- **CORRECTIONS 2026-09-21:** specify from requirements, and implement only what the Abaqus dataset exercises. Legacy archives are never a design input.
- **Commits:** Conventional Commits plus `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Decisions this plan takes (confirm at review)

1. **Duplicate end frame.** The adapter drops the last frame only when its time equals the previous frame's and every field and history value is identical. It raises otherwise, so the drop never loses information. Without it, `time_axis_monotone` (a hard requirement) fails on every Abaqus case.
2. **Rigid-body reference node.** It has no field output, so it is left out of `Nodes`. Each history `RF<k>` it writes becomes a global `reaction_force_<k>_node_<label>` in N.
3. **Flat decks (no `*PART`).** These are one implicit part named `PART-1`, Abaqus's default; the conformance ODB instance was `PART-1-1`.
4. **Yield table source.** Yield rows use the declared table when a benchmark declares one, and otherwise the case's own `*PLASTIC` table. Sweeps with per-case hardening declare none. The declared rule is unchanged for LS-DYNA.
5. **Constitutive rows** run on `solid` element blocks as well as `sph`.
6. **Energy ledger identity** is established numerically on the conformance exports (Task 6) before `collect_run_evidence.py` relies on it.

---

## File Structure

```
src/structbench/core/evidence.py            MODIFY: InputFacts.initial_velocity / initial_hardening (defaults None)
src/structbench/core/io/abaqus_run.py       MODIFY: Explicit run record (Task 2); 2D decks, *DENSITY, *PLASTIC,
                                             *INITIAL CONDITIONS, implicit part, end time (Task 4); energy terms (Task 6)
src/structbench/core/io/abaqus.py           CREATE: read_abaqus_export, abaqus_export_to_case, abaqus_ledger
src/structbench/core/io/__init__.py         MODIFY: export the three
src/structbench/verification/materials.py   MODIFY: class elastic_plastic_isotropic
src/structbench/verification/measures/*.py  MODIFY: constitutive rows on solid blocks; new rows
src/structbench/verification/quantities.py  MODIFY: rows initial_state_matches_input, plastic_dissipation_late_growth;
                                             build sampling_clock_consistent; yield rows lose declared=("yield_table",)
src/structbench/verification/criteria.py    MODIFY: criteria for the new definitional rows
src/structbench/cli/datacheck.py            MODIFY: measure --declaration; measure_cases()
decisions/0070-material-class-isotropic-tabulated.md   CREATE (Proposed)
decisions/0066-reference-data-verification.md          MODIFY: dated note
decisions/README.md                                    MODIFY: index row 0070
data_generation/abaqus/{convert,collect_run_evidence,validate,archive}.py  CREATE
data_generation/abaqus/STANDARD_INPUT_BLOCK.md         MODIFY: npz layout; energy identity
docs/plans/2026-09-24-abaqus-data-pipeline-design.md   MODIFY: Decision 6 work-root wording
tests/core/test_abaqus_run_record.py, tests/core/test_abaqus_input.py   MODIFY
tests/core/test_abaqus_adapter.py                      CREATE
tests/verification/test_measures_solid.py              CREATE
tests/cli/test_datacheck_declaration.py                CREATE
tests/tools/test_abaqus_{convert,collect,archive}.py   CREATE
```

---

### Task 1: Documentation debts from Plan 1

**Files:** `docs/plans/2026-09-24-abaqus-data-pipeline-design.md`, `data_generation/abaqus/STANDARD_INPUT_BLOCK.md`.

- [ ] **Step 1: Spec Decision 6.** Replace "(default `C:\structbench-runs\<name>\` on the maintainer's machine)" with "(on the maintainer's machine, a gitignored folder beside the private dataset definitions)". Do not name the private repository.
- [ ] **Step 2: Document the npz layout.** In `STANDARD_INPUT_BLOCK.md` → "After the run", add a subsection `### The abaqus-npz/1 intermediate` that reproduces the key layout from `odb_export.py`'s module docstring. Add three facts:
  - 2D instances store 3 coordinate columns, with z = 0;
  - an analytical rigid surface appears as an instance with no field blocks;
  - the file holds the duplicate end-of-step frame.
- [ ] **Step 3: Gates, then commit** `docs(abaqus): npz layout; spec work-root wording`.

---

### Task 2: Explicit run record (`read_abaqus_run_evidence`)

**Files:** Modify `src/structbench/core/io/abaqus_run.py` (`_termination`, `_diagnostics`, `read_abaqus_run_evidence`). Test: `tests/core/test_abaqus_run_record.py`.

**Interfaces:** `read_abaqus_run_evidence` keeps its signature. What changes:
- `termination`: `status="error"` with criterion `"analysis_not_completed"` for an analysis-phase failure, and `final_time` in s from the last increment row;
- `n_errors` and `n_warnings` now also count `.sta` markers when the `.msg` has no ANALYSIS SUMMARY;
- `timestep` gets `(times_s, stable_increments_s)` from the Explicit increment table;
- `identity.precision` becomes `"double"` or `"single"` from the `.sta` precision line.

- [ ] **Step 1: Failing tests.** Append to `tests/core/test_abaqus_run_record.py`. The fixtures are invented but match the real Explicit format:

```python
_EXPLICIT_HEAD = (
    "Abaqus/Explicit 2025                             DATE 01-Jan-2026  TIME 00:00:00\n"
    " NUMERICAL PRECISION USED FOR THIS Abaqus/Explicit ANALYSIS\n"
    "Double precision package and explicit executables will be used in this analysis.\n"
    "***WARNING: There are 1 warning messages in the data (.dat) file.  Please\n"
    "            check the data file for possible errors in the input file.\n"
    "***WARNING: Each of the nodes listed below participates in a boundary condition\n"
    "  STEP  TOTAL      STEP      CPU       STABLE       CRITICAL    KINETIC    TOTAL\n"
    "INCREMENT     TIME      TIME      TIME     INCREMENT     ELEMENT     ENERGY     ENERGY\n"
    "        0  0.000E+00 0.000E+00  00:00:00 5.00000E-08          12  1.000E+04  1.000E+04\n"
    "       20  1.000E-06 1.000E-06  00:00:00 4.00000E-08          12  9.500E+03  9.990E+03\n"
)
_COMPLETED = _EXPLICIT_HEAD + (
    "       40  2.000E-06 2.000E-06  00:00:01 3.00000E-08           7  9.000E+03  9.990E+03\n"
    "\n  THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n"
)
_ABORTED = _EXPLICIT_HEAD + (
    "       30  1.500E-06 1.500E-06  00:00:01 1.00000E-14           7  9.100E+03  9.990E+03\n"
    "***ERROR: Excessive distortion of element number 7\n"
    "\n  THE ANALYSIS HAS NOT BEEN COMPLETED\n"
)
_EXPLICIT_MSG = "\n STEP 1  ORIGIN 0.0000\n"
_EXPLICIT_DAT = "   Abaqus 2025\n ***WARNING: THE PARAMETER HOURGLASS ON THE *SECTION CONTROLS\n"


def test_explicit_completed_run_reads_series_precision_and_counts():
    ev = _read(_COMPLETED, _EXPLICIT_MSG, _EXPLICIT_DAT)
    assert ev.termination[0].status == "normal"
    assert ev.termination[0].final_time == pytest.approx(2.0e-6)
    assert ev.identity.precision == "double"
    times, steps = ev.timestep
    assert times == pytest.approx((0.0, 1.0e-6, 2.0e-6))
    assert steps == pytest.approx((5.0e-8, 4.0e-8, 3.0e-8))
    # .dat hourglass warning + .sta boundary/contact warning; the .sta pointer
    # to the .dat warning is not a second warning.
    assert (ev.n_errors, ev.n_warnings) == (0, 2)


def test_explicit_analysis_failure_is_an_error_with_its_markers_counted():
    ev = _read(_ABORTED, _EXPLICIT_MSG, _EXPLICIT_DAT)
    assert ev.termination[0].status == "error"
    assert ev.termination[0].criterion == "analysis_not_completed"
    assert ev.n_errors == 1
    assert "termination_wording" not in ev.unparsable
```

Adapt `_read(sta, msg, dat)` (existing helper, test line 79) if its argument order differs. Keep the Standard-format tests unchanged: they must still pass.

- [ ] **Step 2: Run** `$PY -m pytest tests/core/test_abaqus_run_record.py -q`. Expected: the two new tests fail on termination status, missing precision and the `timestep` of None.

- [ ] **Step 3: Implement** in `abaqus_run.py`:

```python
#: The banner an Explicit job writes when the analysis phase stops early.
_NOT_COMPLETED = re.compile(r"THE ANALYSIS HAS NOT BEEN COMPLETED")
#: An Explicit increment row: increment, total time, step time, CPU hh:mm:ss,
#: stable increment, critical element, kinetic energy, total energy.
_EXPLICIT_ROW = re.compile(
    r"^\s+(\d+)\s+(\S+)\s+(\S+)\s+\d+:\d\d:\d\d\s+(\S+)\s+(\d+)\s+(\S+)\s+(\S+)\s*$",
    re.MULTILINE,
)
_PRECISION = re.compile(r"(Double|Single) precision package and explicit", re.I)
#: The .sta's pointer to the .dat's own warnings -- not a warning of its own.
_DAT_POINTER = re.compile(r"\*{3}WARNING: There (?:is|are) \d+ warning messages? in the data")
```

- **`_termination(status_text, printed_text, tokens, time_factor)`:**
  - Parse `rows = _EXPLICIT_ROW.findall(status_text or "")`.
  - If there are Explicit rows: `n_steps = len(rows)` and `final_time = float(rows[-1][2]) * time_factor`. Otherwise keep the Standard `_INCREMENT` count.
  - Order: `_COMPLETED` gives normal. Then `_NOT_COMPLETED` gives `TerminationRecord("error", final_time, n_steps, "analysis_not_completed")`. Then `_FATAL`. Then `none`.
- **`_diagnostics(messages_text, printed_text, status_text)`:** keep the `.msg` summary first. When the `.msg` states no summary:
  - errors = `.dat` fatal count, or else `.dat` + `.sta` `***ERROR` markers;
  - warnings = `.dat` markers + `.sta` markers − `_DAT_POINTER` matches.
- **`read_abaqus_run_evidence`:**
  - `timestep = (tuple(t), tuple(dt))` with t = step time × `f["time"]` and dt = stable increment × `f["time"]`, when Explicit rows exist;
  - `precision` from `_PRECISION` (lower-cased) goes into `SolverIdentity("abaqus", version, precision=...)`;
  - `f = unit_factors(source_units)` replaces the bare validation call.

- [ ] **Step 4: Run** the file's tests, then the full suite. Expected: all pass.
- [ ] **Step 5: Gates, then commit** `feat(core): read Abaqus/Explicit run records (termination, diagnostics, time step)`.

---

### Task 3: Material class `elastic_plastic_isotropic` (ADR-0070)

**Files:** Create `decisions/0070-material-class-isotropic-tabulated.md`. Modify `decisions/README.md`, `src/structbench/verification/materials.py`, `tests/verification/test_materials.py`.

- [ ] **Step 1: Failing test** (append to `test_materials.py`):

```python
def test_isotropic_tabulated_class():
    cls = material_class("elastic_plastic_isotropic")
    assert cls is not None
    assert (cls.state_variable, cls.state_bounds, cls.monotone) == ("plastic_strain", (0.0, None), True)
    assert cls.yield_law == "tabulated_j2" and not cls.has_equation_of_state
```

- [ ] **Step 2: Run.** Expected: fails with `cls is None`.
- [ ] **Step 3: Register** it in `_CLASSES` (materials.py), after `elastic_plastic_hydro`:

```python
    # ADR-0070. Abaqus *ELASTIC + *PLASTIC (isotropic hardening, tabulated,
    # no EOS): Mises yield bounded by the table, held at its last value.
    MaterialClass(
        "elastic_plastic_isotropic",
        state_variable="plastic_strain",
        state_bounds=(0.0, None),
        monotone=True,
        yield_law="tabulated_j2",
    ),
```

- [ ] **Step 4: Write ADR-0070**, following ADR-0067's headings:
  - **Context:** Abaqus `*PLASTIC` with isotropic hardening and no EOS, as exercised by the first Abaqus dataset.
  - **Decision:** the field table above, with a justification per field:
    - monotone PEEQ was measured on the conformance runs;
    - the yield bound comes from the deck's own table, flat past the last knot, which is Abaqus's documented behaviour. Do not assert it: cite the one-element conformance check, where σ followed σ₀ + Hε̄ᵖ exactly.
  - **Alternatives:** reuse `elastic_plastic_hydro` (rejected: it implies an EOS); leave it unclassified (rejected: the yield and monotonicity rows would read `unsupported`).
  - **Consequences.**

  Add the index row `| 0070 | Material class elastic_plastic_isotropic (Abaqus *PLASTIC, isotropic) | Durable | Proposed |`.
- [ ] **Step 5: Gates, then commit** `feat(verification): material class elastic_plastic_isotropic (ADR-0070)`.

---

### Task 4: 2D Abaqus decks in `read_abaqus_input_facts`

**Files:**
- Modify `src/structbench/core/evidence.py` (InputFacts gains two defaulted fields);
- Modify `src/structbench/core/io/abaqus_run.py`;
- Test: `tests/core/test_abaqus_input.py`.

**Interfaces** (`core.evidence`, public API):

```python
    #: E1: `*INITIAL CONDITIONS, TYPE=VELOCITY` as (node ids, dof 1-based, value m/s).
    initial_velocity: tuple[tuple[frozenset[int], int, float], ...] | None = None
    #: E1: `*INITIAL CONDITIONS, TYPE=HARDENING` as (element id, equivalent plastic strain).
    initial_hardening: tuple[tuple[int, float], ...] | None = None
```

They are appended after `solver`, with defaults, so the LS-DYNA reader and all tests keep constructing `InputFacts` unchanged.

- [ ] **Step 1: Failing tests** (append to `test_abaqus_input.py`). This invented deck builds with the Plan 1 writers' output format:

```python
_FLAT_2D = """*HEADING
toy
*NODE
1, 0.0, 0.0
2, 1.0, 0.0
3, 0.0, 1.0
4, 1.0, 1.0
*ELEMENT, TYPE=CAX4R, ELSET=E
1, 1, 2, 4, 3
*NSET, NSET=ALLN
1, 2, 3, 4
*MATERIAL, NAME=M
*DENSITY
7e-09
*ELASTIC
200000.0, 0.3
*PLASTIC
250.0, 0.0
1250.0, 10.0
*SECTION CONTROLS, NAME=HG_ENHANCED, HOURGLASS=ENHANCED
*SOLID SECTION, ELSET=E, MATERIAL=M, CONTROLS=HG_ENHANCED
*INITIAL CONDITIONS, TYPE=VELOCITY
ALLN, 2, -30000.0
*INITIAL CONDITIONS, TYPE=HARDENING
1, 0.15
*STEP, NAME=S, NLGEOM=YES
*DYNAMIC, EXPLICIT
, 0.001
*END STEP
"""


def test_flat_axisymmetric_deck():
    f = read_abaqus_input_facts(_FLAT_2D, source_units="t-mm-s")
    assert f.dimension == 2 and f.plane_strain is False
    assert f.time_integration == "explicit" and f.end_time == pytest.approx(0.001)
    (part,) = f.parts
    assert (part.discretisation, part.under_integrated) == ("solid", True)
    (m,) = f.materials
    assert m.canonical_model == "elastic_plastic_isotropic"
    assert m.density == pytest.approx(7000.0)  # t/mm^3 -> kg/m^3
    assert m.yield_table == ((0.0, 10.0), pytest.approx((250e6, 1250e6)))
    assert f.initial_velocity == ((frozenset({1, 2, 3, 4}), 2, pytest.approx(-30.0)),)
    assert f.initial_hardening == ((1, pytest.approx(0.15)),)
    assert not {t for t in f.unparsable if t.startswith("unread_card")}


def test_single_row_plastic_is_flat_and_cpe_is_plane_strain():
    deck = _FLAT_2D.replace("CAX4R", "CPE4R").replace("1250.0, 10.0\n", "")
    f = read_abaqus_input_facts(deck, source_units="t-mm-s")
    assert f.plane_strain is True
    assert f.materials[0].yield_table == ((0.0, 1.0), pytest.approx((250e6, 250e6)))
    assert f.parts[0].under_integrated is None  # only CAX4R is established
```

- [ ] **Step 2: Run.** Expected: failures on the missing part, `plane_strain` None, `end_time` None, etc.
- [ ] **Step 3: Implement** in `abaqus_run.py`:
  1. **Element codes:**
     - `_ELEMENT_KINDS = {"C3D": "solid", "CAX": "solid", "CPE": "solid"}`;
     - `_UNDER_INTEGRATED = frozenset({"CAX4R"})`, with the comment "established by the conformance run: one integration point per element, non-zero ALLAE";
     - record `under_integrated` True for those codes and `None` otherwise;
     - collect the codes seen: `plane_strain` is True if every code starts with `CPE`, False if every code starts with `CAX`, otherwise None.
  2. **Implicit part:** `has_parts = any(l.upper().startswith("*PART") for l in lines)`. When there are no parts and an `ELEMENT` or `SOLID SECTION` card arrives with `current_part is None`, set `current_part = "PART-1"` and register it once.
  3. **Data rows:** add `_data_rows(lines, index) -> list[str]`, the consecutive non-keyword lines after `index`, for multi-row tables.
  4. **`DENSITY`** (in a material): `density = float(first value) * f["density"]`.
  5. **`PLASTIC`** (in a material):
     - with no `HARDENING=` or `HARDENING=ISOTROPIC`: knots from `(σ, ε)` rows; σ × `f["stress"]`;
     - one row becomes `((0.0, 1.0), (σ, σ))` (flat; the comment cites Task 3's ADR);
     - the material gets `canonical_model = "elastic_plastic_isotropic"` unless another inelastic card appears;
     - any other `HARDENING=` keeps today's `unread_card:PLASTIC` and inelastic path.
  6. **`NSET`:** record explicit label lists; `GENERATE` adds the token `unread_card:NSET_GENERATE`.
  7. **`INITIAL CONDITIONS`:**
     - `TYPE=VELOCITY` rows `(target, dof, v)`: the target is an NSET name or a node label; v × `f["velocity"]`;
     - `TYPE=HARDENING` rows `(element label, peeq)`: an elset name adds the token `unread_card:HARDENING_ELSET`;
     - add `INITIAL CONDITIONS` to `_ENDS_MATERIAL`, so it no longer lands as an unread material card.
  8. **End time:** after `*DYNAMIC, EXPLICIT`, the data row `, period` gives `end_time = period * f["time"]`. Keep `None` for `*Static`.
  9. **Material construction:** `MaterialInput(mid, canonical, density, None, youngs, poisson, yield_table)`. `canonical` is `"elastic_plastic_isotropic"` if a table was read and nothing is inelastic; else `"linear_elastic"` if elastic only; else None.
- [ ] **Step 4: Run** the file's tests and the full suite, including `tests/cli/test_datacheck_solver_gate.py`. Expected: all pass. The existing C3D cantilever test keeps its facts.
- [ ] **Step 5: Gates, then commit** `feat(core): read 2D Abaqus decks -- parts, density, plastic table, initial conditions`.

---

### Task 5: The adapter `core/io/abaqus.py` and `convert.py`

**Files:**
- Create `src/structbench/core/io/abaqus.py` and `data_generation/abaqus/convert.py`;
- Modify `src/structbench/core/io/__init__.py`;
- Tests: `tests/core/test_abaqus_adapter.py`, `tests/tools/test_abaqus_convert.py`.

**Interfaces:**

```python
ABAQUS_NPZ_FORMAT = "abaqus-npz/1"

def abaqus_export_to_case(
    npz_path: str | Path,
    deck_text: str,
    *,
    source_units: str,
    dimension: int,
    case_id: str,
    dataset_id: str | None = None,
    generation_date: str = "unknown",
) -> Case: ...
```

- [ ] **Step 1: Failing tests.** `tests/core/test_abaqus_adapter.py` builds a tiny synthetic npz: 1 CAX4R element, 4 field nodes, 1 RP node without field data, 3 frames, the last a byte duplicate. It is written with `np.savez` in exactly the Plan 1 layout. The tests:
  1. **Units and shapes:** mm→m on coords and displacement (×1e-3), MPa→Pa on stress, mJ→J on energies. Stress becomes 6-Voigt with `yz = zx = 0`.
  2. **Duplicate frame:** the duplicate end frame is dropped (T = 2). Changing any value of the last frame raises `ValueError` matching "duplicate".
  3. **RP node:** it is absent from `Nodes`, and its `RF2` becomes `globals_["reaction_force_2_node_5"]`.
  4. **Clock:** a history clock that differs from the frame clock raises `ValueError` matching "clock".
  5. **Unknown format:** an unknown `format` raises `ValueError` matching "abaqus-npz".
  6. **Metadata:** provenance solver is "Abaqus" and version "2025"; `source_deck` is the deck; `validate(case)` passes; materials come from the deck (canonical `elastic_plastic_isotropic`).

  Use the Task 4 `_FLAT_2D` deck (import it or copy it; keep the basenames unique).
- [ ] **Step 2: Run.** Expected: ImportError.
- [ ] **Step 3: Implement** `abaqus.py`. Its module docstring states Decisions 1–3.
  - **Loading:** `np.load(allow_pickle=False)`; the manifest format must equal `ABAQUS_NPZ_FORMAT`.
  - **Mesh:** exactly one instance carries `mesh/<inst>/elements/<type>`, otherwise `NotImplementedError("multi-instance decks")`.
    - Keep only nodes present in `field/<step>/U/<inst>/node_labels`.
    - Coordinates are `node_coords[:, :dimension]`; the remaining columns must be 0.
    - Connectivity maps labels to 0-based indices; a label outside the kept nodes raises.
  - **Frames:** exactly one step; times × `f["time"]`.
    - Duplicate rule: if `t[-1] == t[-2]`, every field and history array's last two samples must be equal, then drop the last frame. Otherwise raise.
  - **Node fields:** U→displacement, V→velocity (`f["velocity"]`), A→acceleration, reordered to the `Nodes` order.
  - **Element fields:**
    - one integration point per element, else `NotImplementedError`;
    - reorder to `ElementBlock` order;
    - S (4 components S11, S22, S33, S12) → `np.stack([S11, S22, S33, S12, 0, 0], -1)` × `f["stress"]`;
    - PEEQ → `(T, E)`.
  - **Globals:**
    - the history region starting `Assembly` must satisfy `times == frame times` exactly (after the duplicate drop), else `ValueError("… clock …")`;
    - map `ALLKE→kinetic_energy`, `ALLIE→internal_energy`, `ETOTAL→total_energy`, `ALLAE→hourglass_energy`, `ALLPD→plastic_dissipation`, `ALLVD→viscous_dissipation`, `ALLWK→external_work`, `ALLFD→frictional_dissipation`, `ALLSE→strain_energy`, `ALLCD→creep_dissipation`, all × `f["energy"]`, float32;
    - `Node <inst>.<label>` regions with `RF<k>` become `reaction_force_<k>_node_<label>` × `f["force"]`.
  - **Parts and materials:** `facts = read_abaqus_input_facts(deck_text, source_units=...)`. Exactly one part is required; `part_id` = that part's id for every element.
    - `Material(m.material_id, "ABAQUS", {"density": …, "youngs_modulus": …, "poisson_ratio": …, "yield_table": [list(k), list(s)]}, m.canonical_model)`, in SI.
  - **Metadata:** `Metadata(case_id, dimension, provenance=Provenance("Abaqus", <year from manifest abaqus_release>, generation_date), source_units=..., source_deck=deck_text, dataset_id=...)`, then `validate(case)`.
- [ ] **Step 4: `convert.py`** (script): `--sweep <dir> [--split …] [--out <dir>]`, default `<sweep>/canonical`.
  - For each case with `run.json` completed, an `<id>.npz`, and no `<out>/<id>.h5`, write `abaqus_export_to_case(...)` with `dimension=2`.
    - `generation_date` = `run.json["end_utc"][:10]`;
    - `dataset_id` = the sweep folder name;
    - `source_units` from `provenance.json`.
  - Idempotent; prints `written=/skipped=/failed=` and the failure reasons.
  - **Test:** `tests/tools/test_abaqus_convert.py` uses the synthetic npz plus a fake `run.json` and `provenance.json`.
- [ ] **Step 5: Real data.** Run `convert.py` on the conformance cases in the private work root. Then:
  - `read_case` round-trips;
  - the frame count is 401;
  - `validate` passes;
  - check by eye the S33 hoop component against the npz.
- [ ] **Step 6: Gates, then commit** `feat(core): Abaqus adapter from abaqus-npz/1 to the canonical case`.

---

### Task 6: Energy ledger and the run-evidence collector

**Files:**
- Modify `src/structbench/core/io/abaqus.py` (add `abaqus_ledger`) and `abaqus_run.py` (`energy_terms_computed`);
- Create `data_generation/abaqus/collect_run_evidence.py`;
- Test: `tests/core/test_abaqus_adapter.py`, `tests/tools/test_abaqus_collect.py`.

- [ ] **Step 1: Establish the identity on real data.**
  - On the conformance exports, check numerically which sum reproduces ETOTAL to float precision.
  - Candidate: `ALLKE + ALLIE + ALLVD + ALLFD + ALLCD − ALLWK`. ALLAE, ALLPD and ALLSE are parts of ALLIE, not addends.
  - Record the residual and the cases in the Task 1 section of `STANDARD_INPUT_BLOCK.md`, naming the files.
  - **If no candidate closes, stop and report: the ledger is then not established, and the energy rows stay `source_missing` for Abaqus.**
- [ ] **Step 2: Failing tests.**
  - `abaqus_ledger(npz_path, *, source_units) -> EnergyLedger` returns:
    - terms `kinetic`, `internal`, `damping` (ALLVD), `contact` (ALLFD + ALLCD, if Step 1 put them in the sum), `external_work`, `zero_energy_mode` (ALLAE);
    - an identity equal to the Step 1 sum, with ±1 signs and without `external_work` (that term enters the residual separately, as for LS-DYNA);
    - `solver_total = ETOTAL + ALLWK` (in J).
  - `read_abaqus_input_facts` sets `energy_terms_computed` from `*ENERGY OUTPUT` data rows mapped through the same name table, and `databases_requested` stays None.
- [ ] **Step 3: Implement** both, then **`collect_run_evidence.py`**: `--sweep --split … --out <json>`.
  - Per case with `run.json`: `read_abaqus_run_evidence(...)` from `.sta/.msg/.dat`, plus `ledger=abaqus_ledger(npz)` if the npz exists.
  - Write `dump_run_evidence({case_id: ev})`.
  - No raw text leaves the reader. A test asserts that no invented licence line in the `.dat` fixture reaches the JSON.
- [ ] **Step 4: Run** on the conformance cases. `energy_residual_final` must now measure, not come back absent.
- [ ] **Step 5: Gates, then commit** `feat(core): Abaqus energy ledger and run-evidence collector`.

---

### Task 7: Verification on solid blocks, and three rows

**Files:**
- `src/structbench/verification/measures/{_common,constitutive,__init__,integrity}.py`, `quantities.py`, `criteria.py`;
- `decisions/0066-reference-data-verification.md` (dated note);
- Test: `tests/verification/test_measures_solid.py`.

- [ ] **Step 1: Failing tests** in `test_measures_solid.py`, using a synthetic 2D case with one `solid` block (2 elements), `elastic_plastic_isotropic` facts and a per-case input table (no declared table):
  1. `yield_ratio_max` measures from the input table: σ_vm / σ_y(PEEQ), flat beyond the last knot.
  2. `state_variable_decrease_max` counts a planted decrease; `state_variable_min` reads the minimum.
  3. `initial_state_matches_input`:
     - 0 when frame-0 velocity equals the input's initial velocity on the listed nodes and PEEQ(0) equals the initial hardening;
     - a planted 1 % deviation measures 0.01;
     - with no initial conditions in the input: `not_applicable`;
     - for an LS-DYNA deck: `unsupported` (reader gap, a platform reason).
  4. `plastic_dissipation_late_growth`:
     - `(P[-1] − P[⌊0.9 T⌋]) / P[-1]` from `globals_["plastic_dissipation"]`;
     - `not_applicable` when `P[-1] == 0`;
     - `source_missing` without the global.
  5. `sampling_clock_consistent`: the count of field frames with no ledger sample at the same instant (relative tolerance 1e-9); 0 on the synthetic case.
  6. A pure-SPH case measures exactly as before for all four constitutive rows. Copy one existing SPH fixture expectation.
- [ ] **Step 2: Run.** Expected: failures (the rows are `unsupported` or missing).
- [ ] **Step 3: Implement.**
  - **`_common`:**
    - `class_mask(case, facts, wanted, block=PARTICLES)` and `element_field(case, block, name, keep)`;
    - `STATE_BLOCKS = (PARTICLES, "solid")`;
    - keep `particle_field` as a thin wrapper, so other callers are untouched.
  - **`constitutive`:**
    - `_yield_inputs` and `_state` concatenate over `STATE_BLOCKS` present in the case, along the element axis;
    - add `_yield_table(declared, facts)`: the declared table if set, otherwise the single material's input table, otherwise None (which returns `input_gap`).
  - **`__init__`:** remove `yield_ratio_max`, `yield_saturation_min`, `state_variable_min` and `state_variable_decrease_max` from `_PARTICLE_ONLY`.
  - **`quantities`:**
    - drop `declared=("yield_table",)` from `yield_ratio_max` and `yield_saturation_min` only (the matches and covers rows keep it);
    - add the rows `initial_state_matches_input` (`data_integrity`, requires E1 and E8, bears_on `response`) and `plastic_dissipation_late_growth` (`conservation`, requires E8, bears_on `response`), each with `_TITLES` and `_BEARS_ON` entries;
    - mark `sampling_clock_consistent` implemented, with its measure in `CLOSURE_MEASURES`.
  - **`criteria`:**
    - `initial_state_matches_input`: an INSTRUMENT tolerance of ≤ 1e-5 (provisional; float32 storage);
    - `sampling_clock_consistent`: a REQUIREMENT of 0;
    - `plastic_dissipation_late_growth`: **no criterion** (measured only, per the spec).
  - **ADR-0066:** a dated note naming the three rows and the yield-table source rule (Decision 4).
- [ ] **Step 4: LS-DYNA records unchanged.**
  - Run the full suite. `test_published_records.py` must stay green.
  - If Taylor or notch canonical data is reachable (`STRUCTBENCH_DATA_ROOT`), run the env-gated acceptance tests and diff a re-measured record against `docs/datachecks/*.json`. Only the two new rows may differ, as not-checked rows.
  - If the data is not reachable, rule and ledger it.
- [ ] **Step 5: Gates, then commit** `feat(verification): constitutive rows on solid blocks; initial-state, late-plasticity and sampling-clock rows`.

---

### Task 8: `datacheck measure --declaration` and `validate.py`

**Files:** `src/structbench/cli/datacheck.py`, `data_generation/abaqus/validate.py`. Tests: `tests/cli/test_datacheck_declaration.py`, `tests/tools/test_abaqus_validate.py`.

**Interfaces:**

```python
def declared_from_toml(path: Path) -> tuple[str, DeclaredFacts]:
    """([dataset].name, DeclaredFacts from [declaration]) of a sweep.toml."""

def measure_cases(
    declared: DeclaredFacts, name: str | None, data_root: Path,
    case_ids: Sequence[str] | None = None, *,
    dataset_revision: str | None = None,
    run_evidence: Mapping[str, RunEvidence] | None = None,
) -> DatasetMeasurements: ...
```

`measure_dataset` becomes a thin call to `measure_cases(declared_from_spec(spec), spec.card.name, …)` with its default case list.

- **`[declaration]` keys:** `unit_system` (required), `fields` (list), `discretisation`, `erosion`, `material_family`.
- **Unknown keys** raise.
- **CLI:** `--benchmark` and `--declaration` are mutually exclusive, and exactly one is required. With `--declaration`, the default case list is every `*.h5` under `--data-root`.

- [ ] **Step 1: Failing tests.**
  - A toy TOML plus two synthetic canonical `.h5` files. `measure --declaration` writes a record whose `benchmark` is `None`, with 2 cases.
  - Both flags at once, or neither, exits 2.
  - An unknown key in `[declaration]` raises, naming the key.
- [ ] **Step 2: Run.** Expected: failures. **Step 3: Implement.** **Step 4: Run.**
- [ ] **Step 5: `validate.py`** (script): `--sweep --dataset <dir> --split … [--data-root <sweep>/canonical]`. It:
  1. runs `collect_run_evidence.py`;
  2. runs `python -m structbench.cli.datacheck measure --declaration <dataset>/sweep.toml`;
  3. runs `judge`;
  4. writes into `<sweep>/datacheck/`;
  5. prints failing rows by case and the lowest, median and highest of each measured-only row;
  6. exits 1 on any `fail`.

  The test uses the synthetic sweep from the Task 5 tests.
- [ ] **Step 6: Private dataset.** Add a `[declaration]` block to the private `sweep.toml`:
  - `unit_system = "t-mm-s"`;
  - `fields` = the canonical names the adapter writes;
  - `discretisation = "solid"`, `erosion = false`.

  Commit in the private repository.
- [ ] **Step 7: Gates, then commit** `feat(cli): datacheck --declaration for unregistered sweeps; validate.py`.

---

### Task 9: Validate the production sweep

**Files:** none in the repository; output goes to `<work-root>/<name>/datacheck/`.

- [ ] **Step 1: Convert.** Run `convert.py` on the production splits. Record the `.h5` count, total size and any failures.
- [ ] **Step 2: Validate.** Run `validate.py` on the production splits.
- [ ] **Step 3: Report to the maintainer:**
  - verdict counts per row;
  - every `fail`, with the case and the reason;
  - the spread of each measured-only row: energy drift after impact, hourglass ratio, time-step drop, late plastic-dissipation growth;
  - anything `unsupported` or `source_missing`.

  **Stop there.** Deciding what a failure means for the dataset is the maintainer's call.

---

### Task 10: `archive.py` (dry run by default)

**Files:** `data_generation/abaqus/archive.py`, `tests/tools/test_abaqus_archive.py`.

**Behaviour:**
- **Command:** `--sweep --split … --data-root <OneDrive data tree> [--prune-odb] [--yes]`.
- **Retention** comes from `sweep.toml [retention]`:
  - `odb_fraction` (default 0.05);
  - `odb_seed`;
  - `odb_cases` (named cases to keep, e.g. convergence cases).
- **Plan:** for each validated case (case id present in `<sweep>/datacheck/measurements.json`):
  - copy `<id>.inp`, `provenance.json`, `run.json`, `.sta/.msg/.dat`, `<id>.npz` and, if retained, `<id>.odb` to `<data-root>/raw/<name>/abaqus/<id>/`;
  - copy `<id>.h5` to `<data-root>/canonical/<name>/`;
  - **never copy `runner.log`**, which holds licence text.
- **Paths:** writes only to known paths and never lists the data tree (CORRECTIONS 2026-06-29).
- **Execution:**
  - without `--yes` it prints the plan (counts, bytes) and does nothing;
  - with `--yes` it copies, verifies each file by size and sha256, and never deletes the local copy.
- **`--prune-odb --yes`** deletes local ODBs that are not retained, only for cases whose npz manifest's `odb_sha256` matches the ODB. It prints what it deleted.

- [ ] **Step 1: Failing tests** with tmp folders:
  - the dry run writes nothing;
  - `--yes` copies the expected files and never `runner.log`;
  - retention is deterministic for a seed;
  - prune refuses an ODB whose sha256 differs from the manifest.
- [ ] **Step 2: Run. Step 3: Implement. Step 4: Run.**
- [ ] **Step 5: Gates, then commit** `feat(datagen): archive.py -- copy finished cases to the data tree; ODB retention`.
- [ ] **Step 6: Dry run on production.** Run `archive.py` in dry-run mode and show the maintainer the plan. The real copy into OneDrive (several GB) and any pruning wait for the maintainer's explicit yes.

---

## Review Focus

1. **An aborted case in a converted set.** `convert.py` skips cases whose `run.json` isn't completed, and `validate.py` still reports them from the run evidence as `terminated_normally = fail`. They must not disappear silently. (Task 8 test.)
2. **An ODB exported before the duplicate-frame rule existed, or an export with two equal-time frames that differ.** The adapter raises; it never picks one. (Task 5 test.)
3. **A deck whose initial velocity targets a node set the reader could not resolve** (GENERATE). `initial_state_matches_input` returns `not_assessable/unparsable`, not 0. (Task 7 test.)
4. **A sweep with a declared yield table and per-case input tables that disagree.** The declared table wins (Decision 4) and `yield_table_matches_input` reports the disagreement. (Task 7 test.)
5. **`archive.py` re-run after a partial copy.** Files already copied and verified are skipped; a size or hash mismatch on the destination aborts that case with a message and never overwrites silently. (Task 10 test.)
