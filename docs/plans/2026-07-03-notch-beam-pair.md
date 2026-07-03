# Notch-Beam Benchmark Pair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the two notch-beam benchmarks (`notch_beam_2d_bend`, `notch_beam_2d_impact`) end-to-end per ADR-0026 — damage aux channel, kinematic-particle prescription for the driven pin, 221-case ingestion, frozen stratified splits, cards/configs/docs — plus the Plan-2 carry-overs.

**Architecture:** Substrate extensions first (carry-overs; `damage` extractor; particle-aware QoIs/metrics; the kinematic-prescription mechanism — the plan's one genuinely new capability, needed because the bend pin is velocity-driven and a surrogate cannot infer prescribed motion). Then the two-family conversion script (modeled on the wave `convert.py`), with the long 221-case batch run in the background by the controller while benchmark definition proceeds; splits are frozen from the spec grid (no data needed), and cards take their data facts from the first converted case per family. E2E smokes close both benchmarks.

**Tech Stack:** Python 3.11+, existing deps only. No new dependencies.

**Plan 3 of 3 for v0.2** (ADR-0024). After this: trained baselines (human-gated) + dataset hosting are the only v0.2 items left.

## Global Constraints

- Python floor **3.11**; ruff line length **88** + `ruff format`; mypy `disallow_untyped_defs = true`; NumPy docstrings on public APIs; `_`-prefix private across modules.
- **No new dependencies.**
- **Pre-approved API changes for this plan** (scoped): `QoiInputs` gains optional `particle_type`; `position_rmse`/`field_rmse` gain an optional particle mask; `BenchmarkSpec` gains `kinematic_types: tuple[int, ...] = ()`. Taylor and wave behavior must be bit-identical when these defaults are in play (their existing tests prove it).
- **ADR-0026 frozen contract**: two flat sibling benchmarks; aux = K&C `damage` (d3plot effective-plastic-strain slot, unitless); per benchmark train 88 / val 8 / test_interp 12 built by a fixed stratified rule (this plan: seeded generator script, seed **26**, constraints below), exact id lists frozen as module literals; probes = `2DGeneralizibility` cases (`C_*` ×3 → Bend, `S_*` ×2 → Impact) as a separately-reported `probe` split; eval_splits `("val", "test_interp", "probe")`.
- **Split constraints (enforced by test)**: sizes exactly 88/8/12; val∪test disjoint, union with train = all 108; val+test cells use INTERIOR velocities only (Bend {12,16}; Impact {80,120}); every factor level (span, velocity, loading-point/impactor-shape, notch) appears in train.
- **Verified dataset facts (2026-07-03)**: raw tree matches the spec sheet exactly — Bend: `ConstantVelocity/80{320,480,640}/<L><n><v>/Beam1.k` with L∈{A,B,C}, n∈{a,b,c}, v∈{8,12,16,20} → 36 run dirs per span, 108 total (the other folder entries are `.txt` extracts — ignore); Impact: `InitialVelocity/{Bullet,Rectangular,Sphere}/80{320,480,640}/A<n><v>/` with v∈{40,80,120,160} → 108; probes: 5 dirs under `2DGeneralizibility/`, each holding a run directly. SPH, `*MAT_CONCRETE_DAMAGE_REL3` + `*MAT_PLASTIC_KINEMATIC`, no erosion; spec sheet says 500 frames at 1 ms (ACTUAL count read at ingestion and frozen into cards); concrete deck density reads 2.4e-6 g/mm³ — 1000× light, apparently the same toy scaling as the wave set: VERIFY at ingestion and record on the cards.
- **Case ids**: Bend `NB-B-<span>-<L><n>-<v>` (e.g. `NB-B-320-Aa-8`); Impact `NB-I-<span>-<Shape>-<n>-<v>` (e.g. `NB-I-320-Bullet-a-40`); probes keep their folder names verbatim (e.g. `C_60_240_V22_extrapolation`).
- **Data caution (CORRECTIONS.md)**: enumerate/read ONLY under `..\data\Concrete-Beam\2DNotchBeam\`; the 221-family hydration is expected and approved but LARGE (tens of GB) — the batch must be resumable (skip-existing) and is launched by the controller in the background, never inside a task subagent. Conversion output: `..\data\Concrete-Beam\2DNotchBeam\h5_canonical\`.
- Interpreters: tests/training via the conda env (`C:\Users\272766h\AppData\Local\miniconda3\envs\structbench\python.exe`, ruff at `...\Scripts\ruff.exe`); **conversion via `uv run python`** (lasso-python lives in the project `.venv`, not the conda env).
- Env-gated data tests: notch shares ONE root var `STRUCTBENCH_NOTCH_DATA_ROOT` for both benchmarks.
- Commits: Conventional Commits + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer. Branch `feat/notch-beam-pair` off `main`. Merge/push are confirm-gated (ADR-0023).
- After any card/registry change: regenerate `docs/benchmarks.md` (drift test).

## File Structure

```
tests/benchmarks/test_card_data.py       # MODIFY T1: fields-vs-h5 validation; T8: notch env-var entries
src/structbench/benchmarks/wave_propagation_1d/card.py  # MODIFY T1 only if fields prove wrong
src/structbench/viz/__main__.py          # MODIFY T2: resolve benchmark from run config; fail fast
src/structbench/datasets/canonical.py    # MODIFY T3: +"damage" extractor
src/structbench/eval/metrics.py          # MODIFY T4: QoiInputs.particle_type; masked rmse. T6: notch QoIs
src/structbench/eval/rollout.py          # MODIFY T4: pass particle_type; T5: kinematic prescription + masked metrics
src/structbench/benchmarks/registry.py   # MODIFY T5: kinematic_types field; T8: two _MODULES entries
src/structbench/cli/train.py             # MODIFY T5: kinematic loss mask
src/structbench/benchmarks/notch_beam_2d_bend/{__init__,benchmark,card}.py    # CREATE T8
src/structbench/benchmarks/notch_beam_2d_impact/{__init__,benchmark,card}.py  # CREATE T8
data_generation/lsdyna/2DNotchBeam/convert.py        # CREATE T7
data_generation/lsdyna/2DNotchBeam/freeze_splits.py  # CREATE T8
configs/notch_bend{,_smoke}.toml, configs/notch_impact{,_smoke}.toml  # CREATE T9
README.md                                # MODIFY T9: Benchmarks section
ROADMAP.md                               # MODIFY T9: v0.2 DoD annotations
docs/benchmarks.md                       # REGENERATE T8
tests/... (per task below)
```

---

### Task 1: card-fields validation (Plan-2 carry-over)

**Files:**
- Modify: `tests/benchmarks/test_card_data.py`
- Modify (only if the data disagrees): `src/structbench/benchmarks/wave_propagation_1d/card.py` + regenerate `docs/benchmarks.md`

**Interfaces:**
- Consumes: `read_case`, `get_benchmark`, the existing `_BENCHMARK_ROOTS` parametrization.
- Produces: every env-gated card-data case now ALSO asserts each `card.fields` entry exists in the canonical h5 (`"positions"` maps to `response/node/displacement`; other names must be keys of `response.node` or `response.element["sph"]`).

- [ ] **Step 1: Add the check to the existing parametrized test** (inside `test_card_matches_one_canonical_case`, after the n_frames assertion)

```python
    available = set(case.response.node) | set(case.response.element["sph"])
    available.add("positions")  # derived: coords + response/node/displacement
    missing = [f for f in spec.card.fields if f not in available]
    assert not missing, f"card fields absent from canonical data: {missing}"
```

- [ ] **Step 2: Run with the wave data present**

Run (PowerShell): `$env:STRUCTBENCH_WAVE1D_DATA_ROOT = "..\data\Concrete-Beam\1DWavePropagation\h5_canonical"; & $py -m pytest tests/benchmarks/test_card_data.py -v; Remove-Item Env:STRUCTBENCH_WAVE1D_DATA_ROOT`
Expected: wave case PASSES if `effective_plastic_strain` exists in the h5; if it FAILS, remove that entry from the wave card's `fields`, regenerate docs (`& $py tools/gen_benchmark_docs.py`), and re-run. Taylor skips (unset var).

- [ ] **Step 3: Full suite** (`& $py -m pytest -q` — all pass), **then commit**

```bash
git add tests/benchmarks/test_card_data.py src/structbench/benchmarks/wave_propagation_1d/card.py docs/benchmarks.md
git commit -m "test: card fields validated against canonical data (carry-over)"
```

(If the card didn't change, commit only the test file.)

---

### Task 2: viz resolves the benchmark from the run config (Plan-2 carry-over)

**Files:**
- Modify: `src/structbench/viz/__main__.py` (currently hard-codes `"von_mises_stress"` ground truth and "von Mises" titles — written for Taylor)
- Test: `tests/viz/test_fringe.py` (add one test; module skips without matplotlib — keep that pattern)

**Interfaces:**
- Consumes: `get_benchmark`; the run dir's `config.json` `"benchmark"` key (default `"taylor_impact_2d"`, matching `evaluate()`).
- Produces: viz loads ground truth with `spec.aux_field` and titles/labels with `spec.aux_field` + `spec.card.aux_unit`. Unknown benchmark name → the registry's KeyError propagates with its available-names message (fail fast, no silent Taylor fallback beyond the absent-key default).

- [ ] **Step 1: Read `src/structbench/viz/__main__.py` in full** (it is ~125 lines; find where the run dir/config is read and where `load_case_trajectory`/titles use von Mises).

- [ ] **Step 2: Write the failing test** (in `tests/viz/test_fringe.py`, following its existing skip-without-matplotlib and tmp-run-dir patterns — read the file first; if it has no run-dir fixture, test the resolver in isolation):

```python
def test_viz_resolves_spec_from_run_config(tmp_path):
    import json

    from structbench.viz.__main__ import _resolve_run_spec  # name per Step 3

    (tmp_path / "config.json").write_text(
        json.dumps({"benchmark": "wave_propagation_1d"}), encoding="utf-8"
    )
    spec, resolved = _resolve_run_spec(tmp_path)
    assert spec.aux_field == "axial_stress"
```

- [ ] **Step 3: Implement** — add a module-level helper mirroring `cli/train.py`'s pattern (import `get_benchmark` from `..benchmarks`):

```python
def _resolve_run_spec(out_dir: Path) -> tuple[BenchmarkSpec, dict[str, Any]]:
    """Resolve the run directory's benchmark spec from its config.json."""
    config_path = out_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing run config: {config_path}")
    resolved = json.loads(config_path.read_text(encoding="utf-8"))
    return get_benchmark(resolved.get("benchmark", "taylor_impact_2d")), resolved
```

Then replace the hard-coded `"von_mises_stress"` load with `spec.aux_field`, and the "von Mises" title/colorbar strings with `f"{spec.aux_field} ({spec.card.aux_unit})"`. Keep everything else (fringe styling per ADR-0022) untouched.

- [ ] **Step 4: Run** `& $py -m pytest tests/viz -q` (skips cleanly without matplotlib — the new test must ALSO skip under the module's matplotlib gate if it imports fringe; if `_resolve_run_spec` is importable without matplotlib, place the test so it runs anyway — prefer that). Full suite green. **Commit:**

```bash
git add src/structbench/viz/__main__.py tests/viz/test_fringe.py
git commit -m "fix(viz): resolve benchmark spec from run config (carry-over)"
```

---

### Task 3: `damage` aux extractor

**Files:**
- Modify: `src/structbench/datasets/canonical.py`
- Test: `tests/datasets/test_canonical.py`

**Interfaces:**
- Consumes: `_AUX_EXTRACTORS`, `AuxExtractor` (mapping of SPH response fields + stress_scale → `(T, P)` float32).
- Produces: `available_aux_fields()` contains `"damage"`; the extractor reads `effective_plastic_strain` (already `(T, P)`, unitless K&C scaled damage measure for `*MAT_CONCRETE_DAMAGE_REL3`) and IGNORES `stress_scale`.

- [ ] **Step 1: Failing tests** (add to `tests/datasets/test_canonical.py`; `_sph_case(tmp_path)` writes an `effective_plastic_strain` dataset — verify by reading the helper; if it doesn't, extend the helper's response dict with a known `(T, P)` array first):

```python
def test_damage_extractor_reads_eff_plastic_strain_unscaled(tmp_path):
    h5_path = _sph_case(tmp_path)
    tr = load_case_trajectory(h5_path, aux_field="damage")
    import h5py

    with h5py.File(h5_path) as f:
        expected = f["response/element/sph/effective_plastic_strain"][...]
    np.testing.assert_allclose(tr.aux, expected, rtol=1e-6)  # NO stress scaling


def test_available_aux_fields_lists_damage():
    assert "damage" in available_aux_fields()
```

- [ ] **Step 2: Verify fail → implement** (next to the other extractors):

```python
def _aux_damage(
    sph: Mapping[str, NDArray[np.floating]], stress_scale: float
) -> NDArray[np.float32]:
    """K&C scaled damage measure from the effective-plastic-strain slot.

    For ``*MAT_CONCRETE_DAMAGE_REL3`` the d3plot effective-plastic-strain
    slot records the scaled damage measure (0..2), unitless — so
    ``stress_scale`` is ignored (ADR-0026).

    Parameters
    ----------
    sph:
        Mapping of SPH response fields with an
        ``"effective_plastic_strain"`` key holding a ``(T, P)`` array.
    stress_scale:
        Unused; present for the :data:`AuxExtractor` signature.

    Returns
    -------
    numpy.ndarray
        Shape ``(T, P)``, float32, unitless.
    """
    del stress_scale
    return sph["effective_plastic_strain"][...].astype(np.float32)
```

Add `"damage": _aux_damage` to `_AUX_EXTRACTORS`.

- [ ] **Step 3: Tests pass → full suite → commit**

```bash
git add src/structbench/datasets/canonical.py tests/datasets/test_canonical.py
git commit -m "feat: damage aux extractor (ADR-0026)"
```

---

### Task 4: particle-aware QoIs and metrics

**Files:**
- Modify: `src/structbench/eval/metrics.py` (QoiInputs field; masked rmse)
- Modify: `src/structbench/eval/rollout.py` (pass particle_type into QoiInputs)
- Test: `tests/eval/test_metrics.py`, `tests/eval/test_rollout.py`

**Interfaces:**
- Consumes: existing `QoiInputs`, `position_rmse`, `field_rmse`, `rollout`.
- Produces:
  - `QoiInputs` gains `particle_type: NDArray[np.int64] | None = None` (last field, default None — every existing construction stays valid).
  - `position_rmse(pred, true, keep: NDArray[np.bool_] | None = None)` and `field_rmse(pred, true, keep=None)` — when `keep` (shape `(P,)`) is given, the mean runs over kept particles only.
  - `rollout` fills `particle_type=trajectory.particle_type` in BOTH pred and true QoiInputs.

- [ ] **Step 1: Failing tests**

In `tests/eval/test_metrics.py`:

```python
def test_position_rmse_keep_mask_excludes_particles():
    pred = np.zeros((2, 3, 2), np.float32)
    true = np.zeros((2, 3, 2), np.float32)
    true[:, 2, :] = 10.0  # particle 2 is wildly wrong
    keep = np.array([True, True, False])
    full = position_rmse(pred, true)
    masked = position_rmse(pred, true, keep=keep)
    assert full[0] > 0 and np.allclose(masked, 0.0)


def test_field_rmse_keep_mask():
    pred = np.zeros((2, 3), np.float32)
    true = np.zeros((2, 3), np.float32)
    true[:, 0] = 4.0
    assert np.allclose(field_rmse(pred, true, keep=np.array([False, True, True])), 0.0)
```

In `tests/eval/test_rollout.py`:

```python
def test_rollout_qoi_inputs_carry_particle_type():
    traj = _const_vel_traj()
    sim = _ConstVelSim()

    def type_checker(inputs: QoiInputs) -> float:
        assert inputs.particle_type is not None
        return float(inputs.particle_type.sum())

    result = rollout(sim, traj, window=2, qois={"tc": type_checker})
    assert result.qoi_true["tc"] == float(traj.particle_type.sum())
```

- [ ] **Step 2: Verify fail → implement.** `QoiInputs`: append the optional field + one Attributes line ("particle part-ids, when the caller provides them"). Masked rmse:

```python
def position_rmse(
    pred: NDArray, true: NDArray, keep: NDArray[np.bool_] | None = None
) -> NDArray[np.float64]:
    """Per-frame position RMSE over particles and dimensions.

    Parameters
    ----------
    pred, true:
        Arrays of shape ``(T, P, dim)``.
    keep:
        Optional boolean particle mask ``(P,)``; when given, the mean runs
        over kept particles only (e.g. excluding kinematically prescribed
        particles, ADR-0026).

    Returns
    -------
    numpy.ndarray
        Shape ``(T,)``.
    """
    d = (np.asarray(pred, float) - np.asarray(true, float)) ** 2
    if keep is not None:
        d = d[:, keep, :]
    return np.sqrt(d.mean(axis=(1, 2)))
```

(`field_rmse` analogous with `d = d[:, keep]`.) In `rollout`, add `particle_type=trajectory.particle_type` to both `QoiInputs(...)` constructions.

- [ ] **Step 3: Full suite → commit**

```bash
git add src/structbench/eval tests/eval
git commit -m "feat: particle-aware QoI inputs and maskable rmse metrics"
```

---

### Task 5: kinematic-particle prescription (the driven pin)

The bend pin moves at a prescribed constant velocity and the supports are fixed — a surrogate cannot infer that from history; standard GNS practice prescribes such particles. Behavior with `kinematic_types=()` (Taylor, wave) must be identical to today.

**Files:**
- Modify: `src/structbench/benchmarks/registry.py` (`BenchmarkSpec.kinematic_types: tuple[int, ...] = ()`, documented)
- Modify: `src/structbench/eval/rollout.py` (`rollout` and both one-step functions gain `kinematic_types: tuple[int, ...] = ()`)
- Modify: `src/structbench/cli/train.py` (loss mask; thread `spec.kinematic_types` into rollout/one-step calls in `_validate` and `evaluate`)
- Test: `tests/eval/test_rollout.py`, `tests/benchmarks/test_registry.py`, `tests/cli/test_train_eval.py` (ripple only)

**Interfaces:**
- Produces:
  - In `rollout`: particles whose `trajectory.particle_type` is in `kinematic_types` are OVERWRITTEN with ground-truth positions after each predicted step (they follow their prescription; their aux prediction is left as-is), and both `position_rmse`/`field_rmse` calls pass `keep=~kinematic_mask` so reported errors cover free particles only. Same `keep` in `one_step_position_rmse`/`one_step_aux_rmse` (teacher-forced positions need no overwrite — history is ground truth already — but metrics mask).
  - In the training loop: the per-particle loss terms are multiplied by the free-particle mask (kinematic particles contribute zero loss), normalized by the free count.
  - `train()`/`evaluate()` pass `spec.kinematic_types` everywhere.

- [ ] **Step 1: Failing tests**

```python
# tests/eval/test_rollout.py
def test_rollout_prescribes_kinematic_particles():
    traj = _const_vel_traj()  # give particle 0 type 7 in a local copy
    ptype = traj.particle_type.copy()
    ptype[0] = 7
    traj = dataclasses.replace(traj, particle_type=ptype)
    sim = _ZeroSim()  # a stub predicting zeros everywhere (add if absent)
    result = rollout(sim, traj, window=2, kinematic_types=(7,))
    # particle 0 follows ground truth exactly despite the zero predictor
    np.testing.assert_allclose(
        result.predicted_positions[:, 0, :], traj.positions[:, 0, :]
    )
    # and the reported rmse excludes it: with all-zero predictions on the
    # free particles, rmse reflects only their error
    assert result.position_rmse.shape[0] == traj.positions.shape[0] - 2


def test_rollout_metrics_exclude_kinematic_particles():
    traj = _const_vel_traj()
    ptype = traj.particle_type.copy()
    ptype[0] = 7
    traj = dataclasses.replace(traj, particle_type=ptype)
    sim = _PerfectSim(traj)  # stub returning ground-truth next positions/aux (add if absent)
    result = rollout(sim, traj, window=2, kinematic_types=(7,))
    assert np.allclose(result.position_rmse, 0.0)
```

(`CaseTrajectory` may not be a dataclass with replace-support for tests — check; if `dataclasses.replace` doesn't apply, construct a fresh `CaseTrajectory` with the modified `particle_type`. `_ZeroSim`/`_PerfectSim`: small local stubs mirroring `_ConstVelSim`'s interface.)

```python
# tests/benchmarks/test_registry.py
def test_spec_kinematic_types_default_empty():
    spec = get_benchmark("taylor_impact_2d")
    assert spec.kinematic_types == ()
```

- [ ] **Step 2: Verify fail → implement.**

`registry.py`: add the field after `dataset_id` with docstring "Particle part-ids whose motion is prescribed (kinematic loaders, fixed supports); excluded from training loss and rollout metrics, and driven by ground truth during rollout (ADR-0026)."

`rollout.py` — in `rollout(simulator, trajectory, window, device="cpu", qois=None, kinematic_types=())`:

```python
    kin_mask_np = np.isin(trajectory.particle_type, np.asarray(kinematic_types))
    keep = ~kin_mask_np if kin_mask_np.any() else None
    kin_idx = torch.from_numpy(np.nonzero(kin_mask_np)[0]).to(device)
```

and inside the autoregressive loop, right after `next_pos, aux = simulator.predict_positions(...)`:

```python
            if kin_idx.numel():
                next_pos = next_pos.clone()
                next_pos[kin_idx] = pos[t][kin_idx]
```

(the loop must expose the frame index: change `for _ in range(window, n_frames)` to `for t in range(window, n_frames)`). Pass `keep=keep` to the `position_rmse`/`field_rmse` calls. Same `keep` computation + metric masking in `one_step_position_rmse` and `one_step_aux_rmse` (add the same `kinematic_types=()` parameter; no overwrite needed there). Update docstrings.

`cli/train.py`: in the training loop, after the batch tensors are loaded, build the free mask and apply it to both loss terms:

```python
            free = ~torch.isin(
                particle_type,
                torch.as_tensor(spec.kinematic_types, device=device),
            ) if spec.kinematic_types else None
            ...
            per_particle = train_cfg.w_pos * loss_pos + train_cfg.w_aux * loss_aux
            loss = (
                per_particle[free].mean() if free is not None else per_particle.mean()
            )
```

(adapt to the actual loss-block shape at HEAD — ADR-0028's rework may have restructured it; preserve its semantics, only excluding kinematic particles from the mean). Thread `spec.kinematic_types` into `_validate`'s rollout call and `evaluate`'s rollout/one-step calls.

- [ ] **Step 3: Full suite (Taylor/wave tests prove default-path identity) → gates → commit**

```bash
git add src/structbench/benchmarks/registry.py src/structbench/eval src/structbench/cli/train.py tests
git commit -m "feat: kinematic-particle prescription in training and rollout (ADR-0026)"
```

---

### Task 6: notch QoIs — `midspan_deflection_peak` and `damaged_fraction`

**Files:**
- Modify: `src/structbench/eval/metrics.py`; export both from `src/structbench/eval/__init__.py`
- Test: `tests/eval/test_metrics.py`

**Interfaces:**
- Produces:
  - `midspan_deflection_peak(gauge_halfwidth: float = 5.0, concrete_type: int | None = None) -> QoiFn` — gauge = particles within `gauge_halfwidth` (mm) of the frame-0 x-midspan (restricted to `concrete_type` particles when given and `inputs.particle_type` is present); returns the peak mean downward y-deflection over time, in mm: `max_t(-(mean_y(t) - mean_y(0)))`.
  - `damaged_fraction(threshold: float = 1.9, concrete_type: int | None = None) -> QoiFn` — final-frame fraction of (concrete) particles with `aux >= threshold` (K&C damage saturates at 2; ≥1.9 = fully damaged, the crack pattern's scalar proxy per ADR-0026).
  - ADR-0026's "deflection history" and "damage-field" errors are carried by the rollout position/aux RMSE metrics; these two QoIs are the scalar physical checks.

- [ ] **Step 1: Failing tests**

```python
def _beam_inputs():
    """Static 3-particle 'beam' on x in {0, 50, 100}; middle particle sags."""
    t = np.linspace(0.0, 0.5, 6)
    positions = np.zeros((6, 3, 2), np.float32)
    positions[:, :, 0] = np.array([0.0, 50.0, 100.0], np.float32)
    positions[:, 1, 1] = -np.array([0, 1, 2, 4, 3, 2], np.float32)  # sag, peak 4mm
    aux = np.zeros((6, 3), np.float32)
    aux[-1] = np.array([2.0, 0.5, 2.0], np.float32)  # 2 of 3 damaged at end
    ptype = np.array([1, 1, 2], np.int64)  # particle 2 is not concrete
    return QoiInputs(time=t, positions=positions, aux=aux, particle_type=ptype)


def test_midspan_deflection_peak_reads_the_sag():
    assert midspan_deflection_peak(gauge_halfwidth=5.0)(_beam_inputs()) == pytest.approx(4.0)


def test_damaged_fraction_final_frame():
    inputs = _beam_inputs()
    assert damaged_fraction(threshold=1.9)(inputs) == pytest.approx(2.0 / 3.0)
    assert damaged_fraction(threshold=1.9, concrete_type=1)(inputs) == pytest.approx(0.5)
```

- [ ] **Step 2: Verify fail → implement**

```python
def midspan_deflection_peak(
    gauge_halfwidth: float = 5.0, concrete_type: int | None = None
) -> QoiFn:
    """QoI factory: peak downward mid-span deflection, mm (ADR-0026).

    The gauge is the set of particles within ``gauge_halfwidth`` of the
    frame-0 x-midspan (optionally restricted to ``concrete_type``
    particles). Deflection is the gauge's mean y-displacement from frame 0;
    the QoI is its peak downward excursion over the trajectory.

    Parameters
    ----------
    gauge_halfwidth:
        Half-width of the mid-span gauge window, mm.
    concrete_type:
        When given and ``inputs.particle_type`` is present, only particles
        of this part-id form the gauge.

    Returns
    -------
    QoiFn
        Maps :class:`QoiInputs` to the peak downward deflection (mm).
    """

    def qoi(inputs: QoiInputs) -> float:
        pos = np.asarray(inputs.positions, float)
        x0 = pos[0, :, 0]
        mid = 0.5 * (x0.min() + x0.max())
        gauge = np.abs(x0 - mid) <= gauge_halfwidth
        if concrete_type is not None and inputs.particle_type is not None:
            gauge &= inputs.particle_type == concrete_type
        y = pos[:, gauge, 1].mean(axis=1)
        return float(np.max(y[0] - y))

    return qoi


def damaged_fraction(
    threshold: float = 1.9, concrete_type: int | None = None
) -> QoiFn:
    """QoI factory: final-frame fraction of particles at full damage.

    The K&C scaled damage measure saturates at 2; particles with
    ``aux >= threshold`` in the final frame count as fully damaged — a
    scalar proxy for the crack pattern (ADR-0026).

    Parameters
    ----------
    threshold:
        Damage level counted as fully damaged.
    concrete_type:
        When given and ``inputs.particle_type`` is present, the fraction
        runs over that part-id's particles only.

    Returns
    -------
    QoiFn
        Maps :class:`QoiInputs` to a fraction in ``[0, 1]``.
    """

    def qoi(inputs: QoiInputs) -> float:
        damage = np.asarray(inputs.aux, float)[-1]
        if concrete_type is not None and inputs.particle_type is not None:
            damage = damage[inputs.particle_type == concrete_type]
        if damage.size == 0:
            return 0.0
        return float((damage >= threshold).mean())

    return qoi
```

- [ ] **Step 3: Tests pass → full suite → commit**

```bash
git add src/structbench/eval tests/eval/test_metrics.py
git commit -m "feat: notch-beam QoIs — midspan deflection peak, damaged fraction (ADR-0026)"
```

---

### Task 7: two-family conversion script + first cases + background batch handoff

**Files:**
- Create: `data_generation/lsdyna/2DNotchBeam/convert.py` (model: `data_generation/lsdyna/1DWavePropagation/convert.py` — read it first and mirror structure/flags exactly)

**Interfaces:**
- Produces: `<data-root>/h5_canonical/<case_id>.h5` for all 221 cases; constants `DATASET_ID = "2D-Notched-Beam"`, `SOURCE_UNITS = "g-mm-ms"` (verify no `*CONTROL_UNITS` first, Taylor precedent), `DIMENSION = 2`, `DECK_NAME = "Beam1.k"`; default data root `<repo-parent>/data/Concrete-Beam/2DNotchBeam`.
- Enumeration (three sources, one `Run` list; `discover_runs` does existence checks only, NO d3plot reads):
  - Bend: `ConstantVelocity/80{320,480,640}/<L><n><v>` for L in ABC, n in abc, v in {8,12,16,20} → id `NB-B-<span>-<L><n>-<v>` (span = folder name minus the `80` prefix).
  - Impact: `InitialVelocity/{Bullet,Rectangular,Sphere}/80{320,480,640}/A<n><v>` for n in abc, v in {40,80,120,160} → id `NB-I-<span>-<Shape>-<n>-<v>`.
  - Probes: every directory under `2DGeneralizibility/` containing `Beam1.k` → id = folder name verbatim.
  - Enumerate the SPEC grid explicitly (loop the value tuples, check `run_dir.exists()`) rather than globbing letters — anything on disk beyond the grid is thereby ignored; log a count of grid cells whose directory is MISSING (expect 0).
- `--case` filter matches the case id exactly. `--dry-run` lists 221 and reads nothing.

- [ ] **Step 1: Check the unit card**: `grep -c "CONTROL_UNITS" "..\data\Concrete-Beam\2DNotchBeam\ConstantVelocity\80320\Aa8\Beam1.k"` → expect 0 (else STOP and report).

- [ ] **Step 2: Write the script** per the interfaces above (three enumeration loops building `Run(case_id, run_dir)` — a simpler Run shape than wave's is fine since ids are precomputed; keep the batch loop, SKIP/OK/FAIL/MiB reporting, and the module docstring's hydration warning: "a full batch hydrates 221 d3plot families (tens of GB) from OneDrive; use --case for spot conversions; re-runs skip existing outputs").

- [ ] **Step 3: Dry-run**: `uv run python data_generation/lsdyna/2DNotchBeam/convert.py --dry-run` → exactly 221 cases (108 NB-B, 108 NB-I, 5 probes), 0 missing grid cells.

- [ ] **Step 4: Convert ONE case per family and inspect** (`--case NB-B-320-Aa-8`, then `--case NB-I-320-Bullet-a-40`, then `--case C_60_240_V22_extrapolation`). For each, print via a snippet like wave's Task 5: sph count, frame count, `sorted(set(part_id))`, source_units, and the density from `materials/source_params` (confirm the 2.4e-6 toy-scaling reading). Record ALL of it in your report — Task 8's cards and `kinematic_types` depend on the part-id assignment (expect concrete/pin/support as distinct ids; identify WHICH id is which by count: concrete ≈ 4096, pin 32 (bend) / 112 (impact), support 64).

- [ ] **Step 5: STOP-and-report conditions**: `*CONTROL_UNITS` present; part-id counts don't match the spec-sheet roles; frame counts differ between the two family samples; any conversion failure.

- [ ] **Step 6: Commit the script** (script only — no data):

```bash
git add data_generation/lsdyna/2DNotchBeam/convert.py
git commit -m "feat: 2D notch-beam conversion script (two families + probes, 221 cases)"
```

- [ ] **Step 7: HAND THE BATCH TO THE CONTROLLER.** Do NOT run the full batch yourself — report DONE with the recorded facts; the controller launches `uv run python data_generation/lsdyna/2DNotchBeam/convert.py` as a background process (resumable; skip-existing) and later tasks proceed in parallel. (Controller: log the launch + completion in the ledger; Task 10 gates on completion.)

---

### Task 8: split freezing + the two benchmark modules

**Files:**
- Create: `data_generation/lsdyna/2DNotchBeam/freeze_splits.py`
- Create: `src/structbench/benchmarks/notch_beam_2d_bend/{__init__,benchmark,card}.py`
- Create: `src/structbench/benchmarks/notch_beam_2d_impact/{__init__,benchmark,card}.py`
- Modify: `src/structbench/benchmarks/registry.py` (two `_MODULES` entries), `tests/benchmarks/test_card_data.py` (`_BENCHMARK_ROOTS` gains both notch benchmarks → `STRUCTBENCH_NOTCH_DATA_ROOT` for each)
- Regenerate: `docs/benchmarks.md`
- Test: `tests/benchmarks/test_notch_splits.py`

**Interfaces:**
- Consumes: Task 7's recorded facts (frame count, particle counts/ranges, part-id roles, density note); `damage` extractor; the notch QoIs; `kinematic_types`.
- Produces: `get_benchmark("notch_beam_2d_bend")` / `get_benchmark("notch_beam_2d_impact")` with: aux_field `"damage"`, aux_unit `"-"`, eval_splits `("val", "test_interp", "probe")`, probe splits (`C_*`×3 bend / `S_*`×2 impact), `kinematic_types=(<pin_id>, <support_id>)` from Task 7's facts, `boundary_feature_fn=None`, dataset_id `"2D-Notched-Beam"`, QOIS `{"midspan_deflection_peak": midspan_deflection_peak(concrete_type=<concrete_id>), "damaged_fraction": damaged_fraction(concrete_type=<concrete_id>)}`.

- [ ] **Step 1: Write `freeze_splits.py`** — deterministic generator, run ONCE, output pasted into the modules as literals:

```python
"""Generate the frozen ADR-0026 splits for the two notch-beam benchmarks.

Run once; paste the printed TRAIN/VAL/TEST_INTERP lists into the benchmark
modules as immutable literals. Seed 26; constraints per ADR-0026: sizes
88/8/12, val+test drawn from interior velocities only, every factor level
present in train. Provenance script — not part of the package (ADR-0010).
"""

from __future__ import annotations

import random

SPANS = (320, 480, 640)
BEND_V, BEND_INTERIOR = (8, 12, 16, 20), (12, 16)
IMPACT_V, IMPACT_INTERIOR = (40, 80, 120, 160), (80, 120)
LOADS, NOTCHES = ("A", "B", "C"), ("a", "b", "c")
SHAPES = ("Bullet", "Rectangular", "Sphere")


def freeze(name: str, cases: list[str], interior: list[str]) -> None:
    rng = random.Random(26)
    held = rng.sample(sorted(interior), 20)
    val, test = sorted(held[:8]), sorted(held[8:])
    train = sorted(c for c in cases if c not in held)
    assert len(train) == 88 and len(val) == 8 and len(test) == 12
    # every factor token of every case-id appears among the train ids
    train_tokens = {tok for c in train for tok in c.split("-")}
    for case in cases:
        assert set(case.split("-")) <= train_tokens or case in held
    print(f"# {name}\nTRAIN = {train!r}\nVAL = {val!r}\nTEST_INTERP = {test!r}\n")


bend = [
    f"NB-B-{s}-{ln}-{v}"
    for s in SPANS
    for v in BEND_V
    for ln in (lo + n for lo in LOADS for n in NOTCHES)
]
bend_interior = [c for c in bend if int(c.rsplit("-", 1)[1]) in BEND_INTERIOR]
freeze("notch_beam_2d_bend", bend, bend_interior)

impact = [
    f"NB-I-{s}-{sh}-{n}-{v}"
    for s in SPANS
    for sh in SHAPES
    for n in NOTCHES
    for v in IMPACT_V
]
impact_interior = [c for c in impact if int(c.rsplit("-", 1)[1]) in IMPACT_INTERIOR]
freeze("notch_beam_2d_impact", impact, impact_interior)
```

Run it (`& $py data_generation/lsdyna/2DNotchBeam/freeze_splits.py`), capture the six printed lists. NOTE: the factor-coverage assertion is nearly always satisfied with 88 train cases; if it ever fails, bump the seed by 1, document the seed used in the module docstring, and report the change.

- [ ] **Step 2: Failing tests** (`tests/benchmarks/test_notch_splits.py`):

```python
"""ADR-0026 notch-beam splits: frozen 88/8/12, interior holdout, probes."""

import pytest

from structbench.benchmarks import get_benchmark


@pytest.mark.parametrize(
    ("name", "interior", "n_probes"),
    [
        ("notch_beam_2d_bend", {"12", "16"}, 3),
        ("notch_beam_2d_impact", {"80", "120"}, 2),
    ],
)
def test_frozen_split_honours_adr_0026(name, interior, n_probes):
    spec = get_benchmark(name)
    train = spec.splits["train"]
    val, test = spec.splits["val"], spec.splits["test_interp"]
    assert (len(train), len(val), len(test)) == (88, 8, 12)
    all_ids = set(train) | set(val) | set(test)
    assert len(all_ids) == 108
    for case in list(val) + list(test):
        assert case.rsplit("-", 1)[1] in interior
    train_tokens = {tok for c in train for tok in c.split("-")}
    for case in all_ids:
        assert set(case.split("-")) <= train_tokens  # every factor level in train
    assert len(spec.splits["probe"]) == n_probes
    assert spec.eval_splits == ("val", "test_interp", "probe")
    assert spec.aux_field == "damage"
    assert spec.kinematic_types  # non-empty: pin + support prescribed
```

- [ ] **Step 3: Build the modules.** Each `benchmark.py` mirrors `wave_propagation_1d/benchmark.py`: module docstring citing ADR-0026 + the freeze seed, the pasted `TRAIN/VAL/TEST_INTERP` literals, `PROBE` list (bend: the three `C_*` folder names; impact: the two `S_*`), `AUX_FIELD = "damage"`, part-id constants from Task 7 (`CONCRETE_TYPE`, `PIN_TYPE`, `SUPPORT_TYPE`), and

```python
QOIS: dict[str, QoiFn] = {
    "midspan_deflection_peak": midspan_deflection_peak(concrete_type=CONCRETE_TYPE),
    "damaged_fraction": damaged_fraction(concrete_type=CONCRETE_TYPE),
}
```

Each `card.py` mirrors the wave card: names `"NotchBeam2D-Bend"` / `"NotchBeam2D-Impact"`; materials `("*MAT_CONCRETE_DAMAGE_REL3 (K&C)", "*MAT_PLASTIC_KINEMATIC")` plus the density-scaling note from Task 7 if confirmed; `erosion=False`; loading strings — bend: `"constant-velocity pin, 3-point bend, 8-20 mm/s"`, impact: `"drop-weight impact, initial velocity 40-160 mm/s, impactor shapes Bullet/Rectangular/Sphere"`; `source_units="g-mm-ms"`; geometry `"2D SPH notched beam, H80 x span {320,480,640} mm"`; splits computed from `len()` INCLUDING `"probe": len(PROBE)` (so `n_cases` = 111 bend / 110 impact — the card counts every case the benchmark touches); `task="autoregressive transition (ADR-0026)"`; aux `"damage"`/`"-"`; `qois=tuple(QOIS)`; fields from what Task 7 found in the h5; particles/n_frames/output_dt_ms from Task 7's facts. Each `__init__.py` mirrors the wave one, with `splits` including `"probe"`, `eval_splits=("val", "test_interp", "probe")`, `kinematic_types=(PIN_TYPE, SUPPORT_TYPE)`, `dataset_id="2D-Notched-Beam"`.

- [ ] **Step 4: Registry + env roots.** `_MODULES` gains both names. `_BENCHMARK_ROOTS` in `tests/benchmarks/test_card_data.py` gains both benchmarks mapped to `os.environ.get("STRUCTBENCH_NOTCH_DATA_ROOT")`.

- [ ] **Step 5: Regenerate docs** (`& $py tools/gen_benchmark_docs.py`) → 4 rows. Run the with-data card check once the batch has produced the train[0] cases (coordinate with the controller; if the batch is still running, note it and let Task 10 cover the with-data validation):

`$env:STRUCTBENCH_NOTCH_DATA_ROOT = "..\data\Concrete-Beam\2DNotchBeam\h5_canonical"; & $py -m pytest tests/benchmarks/test_card_data.py -v; Remove-Item Env:STRUCTBENCH_NOTCH_DATA_ROOT`

- [ ] **Step 6: Full suite + gates → commit**

```bash
git add src/structbench/benchmarks data_generation/lsdyna/2DNotchBeam/freeze_splits.py tests/benchmarks docs/benchmarks.md
git commit -m "feat: notch-beam benchmark pair — frozen splits, cards, registry (ADR-0026)"
```

---

### Task 9: configs, README benchmarks section, ROADMAP

**Files:**
- Create: `configs/notch_bend.toml`, `configs/notch_bend_smoke.toml`, `configs/notch_impact.toml`, `configs/notch_impact_smoke.toml`
- Modify: `README.md`, `ROADMAP.md`

- [ ] **Step 1: Configs.** Particle spacing is 2.5 mm (vs Taylor 0.25 → post-ADR-0028 radius 1.5, i.e. 6× spacing); start both benchmarks at the same multiplier with a comment that training tuning is a human call:

```toml
# configs/notch_bend.toml — NotchBeam2D-Bend GNS baseline (ADR-0026)
benchmark = "notch_beam_2d_bend"
dim = 2
connectivity_radius = 15.0  # 6x the 2.5 mm particle spacing (ADR-0028 convention)
training_steps = 100000
```

(`notch_impact.toml` identical but `benchmark = "notch_beam_2d_impact"`. Smoke variants mirror `wave_1d_smoke.toml`: `training_steps = 10`, `val_every = 5`, `batch_size = 2`.)

- [ ] **Step 2: README Benchmarks section** (the ADR-0027 summary row, deferred since Plan 1). Read `README.md` first; insert a short section after the project intro, matching its tone:

```markdown
## Benchmarks

| Benchmark | Problem | Cases |
|---|---|---|
| Taylor2D-Impact | copper bar impact (SPH, plasticity) | 33 |
| Wave1D-Propagation | elastic wave in a bar (entry tier) | 16 |
| NotchBeam2D-Bend | notched concrete beam, 3-point bend | 111 |
| NotchBeam2D-Impact | notched concrete beam, drop-weight impact | 110 |

Full cards (solver, materials, splits, QoIs): [docs/benchmarks.md](docs/benchmarks.md).
```

- [ ] **Step 3: ROADMAP** v0.2 definition-of-done: mark the ingestion item `(wave-1d + notch-beam done 2026-07-03)` — only if the background batch has completed; otherwise `(notch-beam in progress)` — and the modules item `(all three v0.2 benchmarks done)`.

- [ ] **Step 4: Full suite → commit**

```bash
git add configs README.md ROADMAP.md
git commit -m "feat: notch-beam configs; README benchmarks table; ROADMAP status"
```

---

### Task 10: E2E smokes (both benchmarks) + full verification

**Prerequisite:** the controller confirms the background batch finished 221/221 (and the Task 8 with-data card check has been run if it was deferred).

- [ ] **Step 1: Bend smoke** — train/valid/rollout with `configs/notch_bend_smoke.toml`, `--data-root "..\data\Concrete-Beam\2DNotchBeam\h5_canonical"`, `--out scratch\notch-bend-smoke`. Expected: `metrics-val.json`, `metrics-test_interp.json`, `metrics-probe.json` — each with finite `rollout_aux_rmse`, `one_step_aux_rmse`, `"aux_field": "damage"`, `"aux_unit": "-"`, and both QoIs finite; `damaged_fraction` qoi_true in [0, 1].

- [ ] **Step 2: Impact smoke** — same with `configs/notch_impact_smoke.toml`, `--out scratch\notch-impact-smoke`.

- [ ] **Step 3: Kinematic sanity on real data** (one PowerShell snippet, bend val case): load the rollout npz from `scratch\notch-bend-smoke\rollouts\val-<first val case>.npz`, load the same case's trajectory, and assert the pin particles' predicted positions equal ground truth (`np.allclose`) — the prescription working on real data, not just synthetic.

- [ ] **Step 4: Full gates**: `& $py -m pytest -q` (expect all pass; skips = env-var cases without vars + matplotlib viz); ruff check + format --check; mypy. With-data card check for notch if not yet run.

- [ ] **Step 5: Commit** (only if any file changed — otherwise report clean): report the metric mean-blocks for all six metrics files, and hand off to the controller for the final whole-branch review.

---

## Post-plan notes

- **The background batch is the long pole** (221 d3plot families, tens of GB OneDrive hydration). It runs from Task 7 onward while Tasks 8-9 proceed (they need only the three spot-converted cases). Task 10 gates on completion. If the session ends mid-batch, re-running the script resumes (skip-existing).
- **Training-quality caveats for the human training runs** (not plumbing): bend per-frame displacements are tiny (0.008-0.02 mm at dt=1ms) vs `noise_std=0.02` — signal-to-noise may need config tuning; the toy-scaled density (if confirmed) slows dynamics similarly to the wave set. Record both in the training-run notes.
- **Deliberately not here**: load-capacity QoI (ADR-0026 lists it as a possible later addition — needs reaction data confirmation); leaderboard validator; RC-beam 3D (v0.3, erosion).
- v0.2 remaining after this plan: trained baselines (human-gated), dataset hosting decision.
