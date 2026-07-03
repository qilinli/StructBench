# Wave-1D Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the `wave_propagation_1d` benchmark end-to-end per ADR-0025 — ingestion of the 16 LS-DYNA runs, aux-aware QoIs, benchmark module + card + registry entry + config — on the registry/cards substrate that Plan 1 merged.

**Architecture:** Three substrate extensions come first (the human-approved `cached_compute_stats` cache-key fix; the `axial_stress` extractor; a `QoiInputs`-based QoI signature so quantities can read the aux field and time, not just positions). Then the per-dataset conversion script (modeled on the Taylor `convert.py`) ingests the 16 runs, and the benchmark module freezes the ADR-0025 splits with wave QoIs. An end-to-end CPU smoke run on the real ingested data closes the loop.

**Tech Stack:** Python 3.11+, existing deps only (numpy/h5py/lasso-python/torch/torch-geometric). No new dependencies.

**Plan 2 of 3 for v0.2** (ADR-0024). Plan 3 (notch-beam pair, ADR-0026) follows.

## Global Constraints

- Python floor **3.11**; ruff line length **88** + `ruff format`; mypy `disallow_untyped_defs = true`; NumPy-style docstrings on every public API; `_`-prefix symbols private across modules.
- **No new dependencies.**
- **Pre-approved public-API changes in this plan** (human sign-off already given, 2026-07-03): (a) `cached_compute_stats` gains a required `aux_field` keyword and includes it in the cache key; (b) the QoI signature changes from `Callable[[NDArray], float]` over positions to `Callable[[QoiInputs], float]` — ADR-0019's QoI *values* (final length, mushroom width) are frozen; their plumbing is not. No OTHER public API changes without flagging.
- **ADR-0025 frozen contract**: splits are exactly — train = all of L200 and L500 plus (300,1), (300,8), (400,1), (400,8) [12 cases]; val = (300,2), (400,4); test_interp = (300,4), (400,2); **no extrapolation split**. Aux field `axial_stress` (headline); position RMSE is the sanity metric.
- **Data caution (CORRECTIONS.md)**: never run recursive scans/globs over `..\data` beyond the dataset's own folder; the conversion may enumerate and read exactly the 16 run dirs under `..\data\Concrete-Beam\1DWavePropagation\` (one d3plot family each — this hydration is expected and approved).
- Verified dataset facts (checked in the decks/spec 2026-07-03): 2D SPH (`*CONTROL_SPH` IDIM=2), bar along **x** (5 particle rows at y = −4..4 mm), `*MAT_ELASTIC`, `*INITIAL_VELOCITY` loading, no erosion; sweep = length {200,300,400,500} mm × initial velocity {1,2,4,8} mm/ms; deck name `WavePropagation.k`; g-mm-ms units expected (verify no `*CONTROL_UNITS` card, Taylor precedent); ~500/750/1000/1250 particles; spec says 300 outputs at 0.1 ms (actual frame count read at ingestion and frozen into the card).
- Tests: pytest, synthetic-only except the env-gated card-data pattern; deterministic; run via the `structbench` conda env interpreter (`& "C:\Users\272766h\AppData\Local\miniconda3\envs\structbench\Scripts\..\python.exe"` — i.e. the env's `python.exe`).
- Commits: Conventional Commits + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer. Branch: `feat/wave-1d-benchmark` off `main`. Never commit to `main`; merge/push are confirm-gated human calls (ADR-0023).
- After any card or registry change: regenerate `docs/benchmarks.md` (`python tools/gen_benchmark_docs.py`) — the drift test fails otherwise.

## File Structure

```
src/structbench/datasets/
  normalization.py               # MODIFY: cached_compute_stats(+aux_field, keyed)
  canonical.py                   # MODIFY: +"axial_stress" extractor entry
src/structbench/eval/
  metrics.py                     # MODIFY: QoiInputs + QoiFn move here; wave QoIs;
                                 #         final_length/mushroom_width take QoiInputs
  rollout.py                     # MODIFY: import QoiFn/QoiInputs; build QoiInputs;
                                 #         seed predicted_aux with ground truth
  __init__.py                    # MODIFY: export QoiInputs, arrival_time, peak_stress
src/structbench/cli/train.py     # MODIFY: pass aux_field to cached_compute_stats
src/structbench/benchmarks/
  registry.py                    # MODIFY: +wave_propagation_1d in _MODULES
  wave_propagation_1d/
    __init__.py                  # CREATE: SPEC
    benchmark.py                 # CREATE: splits, AUX_FIELD, QOIS
    card.py                      # CREATE: CARD
data_generation/lsdyna/1DWavePropagation/
  convert.py                     # CREATE: 16-run conversion script
configs/
  wave_1d.toml                   # CREATE: full config
  wave_1d_smoke.toml             # CREATE: 10-step smoke config
docs/benchmarks.md               # REGENERATE (Task 6)
tests/
  datasets/test_normalization.py # MODIFY: cache-key tests
  datasets/test_canonical.py     # MODIFY: axial_stress tests
  eval/test_metrics.py           # MODIFY: QoiInputs migration + wave QoI tests
  eval/test_rollout.py           # MODIFY: QoiInputs migration + aux-seed test
  benchmarks/test_wave_split.py  # CREATE
  benchmarks/test_card_data.py   # MODIFY: per-benchmark env-var parametrization
```

---

### Task 1: `aux_field` in `cached_compute_stats` (opening task — approved API change)

**Files:**
- Modify: `src/structbench/datasets/normalization.py:106-136` (signature, key, docstring)
- Modify: `src/structbench/cli/train.py` (the single call site in `train()`: `stats = cached_compute_stats(train_trajs, dataset_root=data_root)`)
- Test: `tests/datasets/test_normalization.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `cached_compute_stats(trajectories: list[CaseTrajectory], *, dataset_root: str | Path, aux_field: str) -> NormalizationStats` — `aux_field` REQUIRED keyword, folded into the cache key so different aux channels never share a cache file.

- [ ] **Step 1: Write the failing tests** (add to `tests/datasets/test_normalization.py`, reusing its existing trajectory-builder helper for inputs)

```python
def test_cache_key_separates_aux_fields(tmp_path):
    trajs = [_const_accel_traj()]  # use this file's existing helper
    s1 = cached_compute_stats(trajs, dataset_root=tmp_path, aux_field="von_mises_stress")
    s2 = cached_compute_stats(trajs, dataset_root=tmp_path, aux_field="axial_stress")
    caches = sorted((tmp_path / "derived").glob("norm_*.npz"))
    assert len(caches) == 2  # one file per aux field, same case ids
    np.testing.assert_allclose(s1.aux_mean, s2.aux_mean)  # same trajs -> same stats


def test_cache_hit_same_aux_field(tmp_path):
    trajs = [_const_accel_traj()]
    cached_compute_stats(trajs, dataset_root=tmp_path, aux_field="von_mises_stress")
    cached_compute_stats(trajs, dataset_root=tmp_path, aux_field="von_mises_stress")
    caches = list((tmp_path / "derived").glob("norm_*.npz"))
    assert len(caches) == 1
```

Also update this file's existing `cached_compute_stats` tests to pass `aux_field="von_mises_stress"` (the parameter is now required).

- [ ] **Step 2: Run to verify failure**

Run: `python.exe -m pytest tests/datasets/test_normalization.py -v`
Expected: FAIL — unexpected keyword argument `aux_field`

- [ ] **Step 3: Implement**

In `normalization.py`, change the signature and key (docstring updated to say the key hashes the aux-field name plus the case-id list):

```python
def cached_compute_stats(
    trajectories: list[CaseTrajectory],
    *,
    dataset_root: str | Path,
    aux_field: str,
) -> NormalizationStats:
```

```python
    case_ids = [trajectory.case_id for trajectory in trajectories]
    key_material = "\n".join([aux_field, *case_ids])
    key = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:12]
```

In `cli/train.py`'s `train()`, the call becomes:

```python
    stats = cached_compute_stats(
        train_trajs, dataset_root=data_root, aux_field=spec.aux_field
    )
```

- [ ] **Step 4: Run the full suite**

Run: `python.exe -m pytest -q`
Expected: all pass (any other `cached_compute_stats` caller in tests updated in Step 1).

- [ ] **Step 5: Commit**

```bash
git add src/structbench/datasets/normalization.py src/structbench/cli/train.py tests/datasets/test_normalization.py
git commit -m "fix: aux_field in cached_compute_stats signature + cache key"
```

---

### Task 2: `axial_stress` extractor

**Files:**
- Modify: `src/structbench/datasets/canonical.py` (one dict entry + one function)
- Test: `tests/datasets/test_canonical.py`

**Interfaces:**
- Consumes: the `_AUX_EXTRACTORS` registry and `AuxExtractor` alias (mapping of SPH response fields + stress_scale → `(T, P)` float32).
- Produces: `available_aux_fields()` now contains `"axial_stress"`; `load_case_trajectory(..., aux_field="axial_stress")` returns `aux` = Voigt component 0 (σ_xx) scaled by `stress_scale`.

- [ ] **Step 1: Write the failing test** (add to `tests/datasets/test_canonical.py`; `_sph_case(tmp_path)` is this file's existing synthetic-case helper — its stress array is `(T, P, 6)` with known values)

```python
def test_axial_stress_extractor_takes_voigt_xx(tmp_path):
    h5_path = _sph_case(tmp_path)
    tr_axial = load_case_trajectory(h5_path, aux_field="axial_stress")
    import h5py

    with h5py.File(h5_path) as f:
        sxx_pa = f["response/element/sph/stress"][...][..., 0]
    np.testing.assert_allclose(tr_axial.aux, sxx_pa * 1e-6, rtol=1e-6)  # Pa -> MPa


def test_available_aux_fields_lists_axial_stress():
    assert "axial_stress" in available_aux_fields()
```

- [ ] **Step 2: Run to verify failure**

Run: `python.exe -m pytest tests/datasets/test_canonical.py -v`
Expected: the two new tests FAIL with `KeyError: ... axial_stress`

- [ ] **Step 3: Implement** (in `canonical.py`, next to `_aux_von_mises`)

```python
def _aux_axial_stress(
    sph: Mapping[str, NDArray[np.floating]], stress_scale: float
) -> NDArray[np.float32]:
    """Axial stress: Voigt component 0 (sigma_xx), scaled to the working frame.

    Parameters
    ----------
    sph:
        Mapping of SPH response fields with a ``"stress"`` key holding a
        ``(T, P, 6)`` Voigt array (Pa).
    stress_scale:
        Multiplier to the working stress unit (1e-6 for Pa -> MPa).

    Returns
    -------
    numpy.ndarray
        Shape ``(T, P)``, float32, working-frame units (MPa by default).
    """
    return (sph["stress"][...][..., 0] * stress_scale).astype(np.float32)


_AUX_EXTRACTORS: dict[str, AuxExtractor] = {
    "von_mises_stress": _aux_von_mises,
    "axial_stress": _aux_axial_stress,
}
```

- [ ] **Step 4: Run tests, then commit**

Run: `python.exe -m pytest tests/datasets -q` then `python.exe -m pytest -q` — all pass.

```bash
git add src/structbench/datasets/canonical.py tests/datasets/test_canonical.py
git commit -m "feat: axial_stress aux extractor (ADR-0025)"
```

---

### Task 3: `QoiInputs` — aux-aware QoI signature + ground-truth aux seeding

The current `QoiFn` receives only positions (`rollout.py:107-110`), and the seeded frames of `predicted_aux` are zero stubs (`rollout.py:92`). Wave QoIs need time + aux, and QoIs over aux must not see fabricated zeros. `QoiInputs`/`QoiFn` move to `metrics.py` (rollout already imports metrics; the reverse import would cycle).

**Files:**
- Modify: `src/structbench/eval/metrics.py` (add `QoiInputs`, `QoiFn`; migrate `final_length`/`mushroom_width`)
- Modify: `src/structbench/eval/rollout.py` (import them; delete its own `QoiFn`; build `QoiInputs`; seed aux with ground truth)
- Modify: `src/structbench/eval/__init__.py` (export `QoiInputs`; `QoiFn` export must keep working — `registry.py` imports it from `..eval`)
- Modify: `src/structbench/benchmarks/taylor_impact_2d/benchmark.py` — NO content change required (it imports `QoiFn, final_length, mushroom_width` from `...eval`; verify the import path still resolves)
- Test: `tests/eval/test_metrics.py`, `tests/eval/test_rollout.py`

**Interfaces:**
- Consumes: `CaseTrajectory.aux` and `.time`.
- Produces (Tasks 4/6 rely on these):

```python
@dataclass(frozen=True)
class QoiInputs:
    time: NDArray[np.float64]       # (T,) seconds
    positions: NDArray[np.float32]  # (T, P, dim) working frame (mm)
    aux: NDArray[np.float32]        # (T, P) working frame (card's aux unit)

QoiFn = Callable[[QoiInputs], float]
```

`rollout(..., qois=...)` evaluates each fn on predicted inputs (`QoiInputs(trajectory.time, pred_pos, pred_aux)`) and true inputs (`QoiInputs(trajectory.time, trajectory.positions, trajectory.aux)`). `final_length(inputs)`/`mushroom_width(inputs)` compute the identical ADR-0019 values from `inputs.positions`. Seeded frames of `predicted_aux` now carry ground-truth aux (mirroring the seeded positions), not zeros.

Additionally (ADR-0025 §4 promises one-step **axial-stress** RMSE, which the pipeline lacks): a new additive function `one_step_aux_rmse(simulator, trajectory, window, device="cpu") -> NDArray[np.float64]` in `rollout.py` — teacher-forced like `one_step_position_rmse` but collecting the aux channel (`aux[:, 0]`, de-normalized by the simulator as in `rollout`) and returning per-frame `field_rmse` against `trajectory.aux[window:]`. `evaluate()` in `cli/train.py` gains per-case and mean `"one_step_aux_rmse"` metric keys next to the existing one-step position key. Exported from `eval/__init__.py`.

- [ ] **Step 1: Write the failing tests**

In `tests/eval/test_metrics.py`, migrate the existing `final_length`/`mushroom_width` tests to wrap their positions arrays (values asserted stay identical):

```python
def _inputs(positions):
    t = positions.shape[0]
    return QoiInputs(
        time=np.arange(t, dtype=float),
        positions=np.asarray(positions, np.float32),
        aux=np.zeros(positions.shape[:2], np.float32),
    )
# existing assertions become e.g.: final_length(_inputs(positions)) == expected
```

In `tests/eval/test_rollout.py`, add:

```python
def test_rollout_qois_receive_aux_and_time():
    traj = _traj()  # this file's existing synthetic-trajectory helper
    sim = _ConstantSimulator()  # this file's existing stub simulator

    def aux_peak(inputs: QoiInputs) -> float:
        assert inputs.time.shape[0] == inputs.positions.shape[0]
        return float(np.abs(inputs.aux).max())

    result = rollout(sim, traj, window=2, qois={"aux_peak": aux_peak})
    assert np.isfinite(result.qoi_true["aux_peak"])
    assert result.qoi_true["aux_peak"] == float(np.abs(traj.aux).max())


def test_rollout_seeds_predicted_aux_with_ground_truth():
    traj = _traj()
    sim = _ConstantSimulator()
    result = rollout(sim, traj, window=2)
    np.testing.assert_allclose(result.predicted_aux[:2], traj.aux[:2])


def test_one_step_aux_rmse_shape_and_finiteness():
    traj = _traj()
    sim = _ConstantSimulator()
    per_frame = one_step_aux_rmse(sim, traj, window=2)
    assert per_frame.shape == (traj.positions.shape[0] - 2,)
    assert np.all(np.isfinite(per_frame))
```

(Adapt helper names to the file's actual builders — read the file first; if its simulator stub is named differently, use that name. If no aux values are non-zero in the helper trajectory, give the helper's aux a non-trivial fill, e.g. `np.arange`, so the seed test is meaningful.)

- [ ] **Step 2: Run to verify failure**

Run: `python.exe -m pytest tests/eval -v`
Expected: FAIL — `QoiInputs` not importable / seeded aux is zeros.

- [ ] **Step 3: Implement**

`metrics.py` — module docstring becomes "Rollout metrics and quantity-of-interest inputs." Add at top (imports: `dataclass`, `Callable`):

```python
@dataclass(frozen=True)
class QoiInputs:
    """Arrays a quantity of interest may read (predicted or ground truth).

    Attributes
    ----------
    time:
        ``(T,)`` global time axis, seconds.
    positions:
        ``(T, P, dim)`` particle positions, working frame (mm).
    aux:
        ``(T, P)`` auxiliary field, working frame (the card's aux unit).
    """

    time: NDArray[np.float64]
    positions: NDArray[np.float32]
    aux: NDArray[np.float32]


#: A quantity of interest maps rollout arrays to one scalar.
QoiFn = Callable[[QoiInputs], float]
```

Migrate the two ADR-0019 QoIs (identical values, new signature):

```python
def final_length(inputs: QoiInputs) -> float:
    """x-extent of the final frame (ADR-0019 QoI; value unchanged)."""
    last = np.asarray(inputs.positions, float)[-1]
    x = last[:, 0]
    return float(x.max() - x.min())


def mushroom_width(inputs: QoiInputs) -> float:
    """y-extent of the final frame (ADR-0019 QoI; value unchanged)."""
    last = np.asarray(inputs.positions, float)[-1]
    y = last[:, 1]
    return float(y.max() - y.min())
```

`rollout.py` — remove its `QoiFn` definition; `from .metrics import QoiFn, QoiInputs, field_rmse, position_rmse`. Replace the zero-stub aux seed (line 92) with:

```python
    aux_true = torch.from_numpy(trajectory.aux).to(device)
    aux_pred = [aux_true[i] for i in range(window)]
```

Replace the QoI evaluation block (lines 107-110) with:

```python
    pred_inputs = QoiInputs(time=trajectory.time, positions=pred_pos, aux=pred_aux)
    true_inputs = QoiInputs(
        time=trajectory.time, positions=trajectory.positions, aux=trajectory.aux
    )
    qoi_pred = {name: float(fn(pred_inputs)) for name, fn in (qois or {}).items()}
    qoi_true = {name: float(fn(true_inputs)) for name, fn in (qois or {}).items()}
```

Update the `qois` parameter docstring ("function of a :class:`QoiInputs`") and the `RolloutResult` docstring line about seeded aux frames (now ground truth, mirroring the seeded positions). `eval/__init__.py`: add `QoiInputs` and `one_step_aux_rmse` to the re-exports (keep `QoiFn` exported — now sourced from `metrics`).

Add `one_step_aux_rmse` to `rollout.py`, mirroring `one_step_position_rmse` (same loop and guards) but keeping the aux channel:

```python
def one_step_aux_rmse(
    simulator: _SimulatorLike,
    trajectory: CaseTrajectory,
    window: int,
    device: str = "cpu",
) -> NDArray[np.float64]:
    """Teacher-forced next-step aux-field RMSE per predicted frame (ADR-0025).

    Same protocol as :func:`one_step_position_rmse`, reading the auxiliary
    prediction instead: each frame from ``window`` onward is predicted from
    its ground-truth history, isolating single-step accuracy of the aux head.

    Returns
    -------
    numpy.ndarray
        Shape ``(T - window,)``, in the trajectory's working aux unit.
    """
    pos = torch.from_numpy(trajectory.positions).to(device)
    n_frames, n_particles, _ = pos.shape
    if n_frames <= window:
        raise ValueError(f"trajectory has {n_frames} frames; window={window}")
    ptype = torch.from_numpy(trajectory.particle_type).to(device)
    npp = torch.tensor([n_particles], device=device)

    predicted = []
    with torch.no_grad():
        for t in range(window, n_frames):
            seq_pw = pos[t - window : t].permute(1, 0, 2).contiguous()
            _, aux = simulator.predict_positions(seq_pw, npp, ptype)
            predicted.append(aux[:, 0])

    pred = torch.stack(predicted, dim=0).cpu().numpy().astype(np.float32)
    return field_rmse(pred, trajectory.aux[window:])
```

In `cli/train.py`'s `evaluate()`: compute `one_step_aux = one_step_aux_rmse(simulator, trajectory, gns.window, device)` next to the existing one-step call; add per-case key `"one_step_aux_rmse": float(one_step_aux.mean())` and the corresponding `"one_step_aux_rmse"` entry in the mean block; include the array in the saved rollout `.npz`.

- [ ] **Step 4: Run the full suite**

Run: `python.exe -m pytest -q`
Expected: all pass — including Taylor split/QoI tests and `tests/cli` (which exercise `rollout` with `spec.qois`).

- [ ] **Step 5: Commit**

```bash
git add src/structbench/eval tests/eval src/structbench/benchmarks/taylor_impact_2d/benchmark.py
git commit -m "refactor: QoiInputs signature for QoIs; ground-truth aux seeding in rollout"
```

---

### Task 4: wave QoIs — `arrival_time` and `peak_stress`

**Files:**
- Modify: `src/structbench/eval/metrics.py`
- Modify: `src/structbench/eval/__init__.py` (export both)
- Test: `tests/eval/test_metrics.py`

**Interfaces:**
- Consumes: `QoiInputs`, `QoiFn` (Task 3).
- Produces: `arrival_time(station_frac: float, *, threshold_frac: float = 0.1) -> QoiFn` (returns milliseconds) and `peak_stress(inputs: QoiInputs) -> float` (working stress unit). Task 6's `QOIS` dict uses them.

- [ ] **Step 1: Write the failing tests** (add to `tests/eval/test_metrics.py`)

```python
def _plane_wave_inputs():
    """A |stress| front moving +x at 1 station per frame; bar x in [0, 100]."""
    t = np.linspace(0.0, 0.01, 11)  # 11 frames, seconds
    x = np.linspace(0.0, 100.0, 101, dtype=np.float32)
    positions = np.zeros((11, 101, 2), np.float32)
    positions[:, :, 0] = x  # static bar
    aux = np.zeros((11, 101), np.float32)
    for frame in range(11):
        front = frame * 10.0  # front position in mm at this frame
        aux[frame, x <= front] = 5.0  # 5 MPa behind the front
    return QoiInputs(time=t, positions=positions, aux=aux)


def test_arrival_time_reads_the_front_crossing():
    inputs = _plane_wave_inputs()
    # station 0.5 -> x = 50 mm; front reaches it at frame 5 -> t = 0.005 s = 5 ms
    assert arrival_time(0.5)(inputs) == pytest.approx(5.0)
    assert arrival_time(0.25)(inputs) == pytest.approx(2.5, abs=0.51)  # frame 3


def test_arrival_time_saturates_when_no_crossing():
    inputs = _plane_wave_inputs()
    quiet = QoiInputs(
        time=inputs.time, positions=inputs.positions, aux=np.zeros_like(inputs.aux)
    )
    assert arrival_time(0.5)(quiet) == pytest.approx(inputs.time[-1] * 1e3)


def test_peak_stress_is_global_abs_max():
    inputs = _plane_wave_inputs()
    assert peak_stress(inputs) == pytest.approx(5.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `python.exe -m pytest tests/eval/test_metrics.py -v`
Expected: FAIL — `arrival_time` not defined.

- [ ] **Step 3: Implement** (in `metrics.py`)

```python
def arrival_time(station_frac: float, *, threshold_frac: float = 0.1) -> QoiFn:
    """QoI factory: wave-front arrival time at a gauge station, milliseconds.

    The gauge is the particle nearest to the fractional position
    ``station_frac`` along the frame-0 x-extent of the bar. Arrival is the
    first frame where the gauge's ``|aux|`` reaches ``threshold_frac`` of
    that trajectory's own peak ``|aux|`` (self-referenced so predicted and
    ground-truth trajectories are judged by the same rule). If the signal
    never crosses (e.g. an all-zero field), the final time is returned —
    a saturating "never arrived" value rather than NaN.

    Parameters
    ----------
    station_frac:
        Fractional gauge position along the bar, in ``[0, 1]``.
    threshold_frac:
        Arrival threshold as a fraction of the trajectory's peak ``|aux|``.

    Returns
    -------
    QoiFn
        Maps :class:`QoiInputs` to the arrival time in milliseconds.
    """

    def qoi(inputs: QoiInputs) -> float:
        x0 = np.asarray(inputs.positions, float)[0, :, 0]
        gauge_x = x0.min() + station_frac * (x0.max() - x0.min())
        gauge = int(np.argmin(np.abs(x0 - gauge_x)))
        signal = np.abs(np.asarray(inputs.aux, float)[:, gauge])
        peak = float(np.abs(np.asarray(inputs.aux, float)).max())
        if peak == 0.0:
            return float(inputs.time[-1] * 1e3)
        hits = np.nonzero(signal >= threshold_frac * peak)[0]
        frame = int(hits[0]) if hits.size else -1
        return float(inputs.time[frame] * 1e3)

    return qoi


def peak_stress(inputs: QoiInputs) -> float:
    """Global peak ``|aux|`` over all frames and particles (working unit)."""
    return float(np.abs(np.asarray(inputs.aux, float)).max())
```

Export `arrival_time` and `peak_stress` from `eval/__init__.py`.

- [ ] **Step 4: Run tests, then commit**

Run: `python.exe -m pytest tests/eval -q` then `python.exe -m pytest -q` — all pass.

```bash
git add src/structbench/eval tests/eval/test_metrics.py
git commit -m "feat: wave QoIs — gauge arrival time and peak stress (ADR-0025)"
```

---

### Task 5: conversion script + ingestion of the 16 runs

**Files:**
- Create: `data_generation/lsdyna/1DWavePropagation/convert.py`
- No unit tests (per ADR-0010 the script is outside the package; the Taylor `convert.py` precedent has none — verification is the `--dry-run` and the real conversion below)

**Interfaces:**
- Consumes: `lsdyna_to_case(d3plot_path, deck_path, *, source_units, dimension, case_id, dataset_id)` and `write_case(case, path)` from `structbench.core`.
- Produces: `<data-root>/h5_canonical/W1D-<L>-<v>.h5` × 16, case ids `W1D-200-1` … `W1D-500-8` (Task 6's splits use exactly these ids). Reports the ACTUAL frame count and particle counts for Task 6's card.

- [ ] **Step 1: Verify the unit convention** (one deck, no hydration — the .k is small text)

Run: `grep -c "CONTROL_UNITS" "..\data\Concrete-Beam\1DWavePropagation\200_1\WavePropagation.k"` (Git Bash from repo root)
Expected: `0` → the convention is supplied by the script as `g-mm-ms` (Taylor precedent, ADR-0016 §5). If nonzero, STOP and report — the script must then read it instead.

- [ ] **Step 2: Write the script** — model: `data_generation/lsdyna/2D-Copper-Bar-Taylor-Impact/convert.py` (same structure, flags `--data-root/--out/--case/--dry-run/--overwrite`, same batch loop with per-case OK/FAIL and MiB reporting; reuse its code with these dataset specifics):

```python
DATASET_ID = "1D-Wave-Propagation"
SOURCE_UNITS = "g-mm-ms"  # no *CONTROL_UNITS in the deck (ADR-0016 par. 5)
DIMENSION = 2  # *CONTROL_SPH IDIM=2; thin strip, bar along x
DECK_NAME = "WavePropagation.k"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATA_ROOT = (
    _REPO_ROOT.parent / "data" / "Concrete-Beam" / "1DWavePropagation"
)
```

Run layout and naming (replaces Taylor's `Run` dataclass fields `geom`/`vel`):

```python
@dataclass(frozen=True)
class Run:
    """One LS-DYNA run: <data-root>/<length>_<velocity>/."""

    length: str
    velocity: str
    run_dir: Path

    @property
    def case_id(self) -> str:
        return f"W1D-{self.length}-{self.velocity}"

    @property
    def d3plot(self) -> Path:
        return self.run_dir / "d3plot"

    @property
    def deck(self) -> Path:
        return self.run_dir / DECK_NAME


def discover_runs(data_root: Path) -> list[Run]:
    """Enumerate <L>_<v> run dirs holding a d3plot + deck (no hydration)."""
    runs: list[Run] = []
    for run_dir in sorted(p for p in data_root.glob("[0-9]*_[0-9]*") if p.is_dir()):
        length, _, velocity = run_dir.name.partition("_")
        run = Run(length=length, velocity=velocity, run_dir=run_dir)
        if run.d3plot.exists() and run.deck.exists():
            runs.append(run)
    return runs
```

The `--case` filter matches `f"{r.length}/{r.velocity}"` (e.g. `--case 200/1`). Module docstring: adapt Taylor's, noting the 16-run sweep, the `[0-9]*_[0-9]*` layout, and that a full batch hydrates 16 d3plot families from OneDrive.

- [ ] **Step 3: Dry-run**

Run: `python.exe data_generation/lsdyna/1DWavePropagation/convert.py --dry-run`
Expected: exactly 16 cases listed, `W1D-200-1` … `W1D-500-8`, nothing read.

- [ ] **Step 4: Convert one case, inspect it**

Run: `python.exe data_generation/lsdyna/1DWavePropagation/convert.py --case 200/1`
Then inspect (records the card facts):

```powershell
& $py -c "
from structbench.core.io import read_case
c = read_case(r'..\data\Concrete-Beam\1DWavePropagation\h5_canonical\W1D-200-1.h5')
print('sph:', c.elements['sph'].element_id.shape[0])
print('frames:', c.response.time.shape[0])
print('t_end (s):', float(c.response.time[-1]))
print('source_units:', c.metadata.source_units)
"
```

Expected: sph = 500; frames ≈ 300–301 (record the EXACT value); t_end ≈ 0.030; source_units g-mm-ms. If sph ≠ 500, STOP and report.

- [ ] **Step 5: Convert the batch**

Run: `python.exe data_generation/lsdyna/1DWavePropagation/convert.py`
Expected: `16/16 done, 0 failed`. Record the particle counts across lengths (expect 500/750/1000/1250) and the frame count — Task 6 needs both.

- [ ] **Step 6: Commit** (script only — data stays outside the repo)

```bash
git add data_generation/lsdyna/1DWavePropagation/convert.py
git commit -m "feat: 1D wave propagation conversion script (16-run sweep)"
```

---

### Task 6: benchmark module, card, registry entry, configs, docs regen

**Files:**
- Create: `src/structbench/benchmarks/wave_propagation_1d/{__init__.py,benchmark.py,card.py}`
- Modify: `src/structbench/benchmarks/registry.py` (`_MODULES` entry)
- Create: `configs/wave_1d.toml`, `configs/wave_1d_smoke.toml`
- Modify: `tests/benchmarks/test_card_data.py` (per-benchmark env vars)
- Regenerate: `docs/benchmarks.md`
- Test: `tests/benchmarks/test_wave_split.py`

**Interfaces:**
- Consumes: `BenchmarkCard`, `BenchmarkSpec`, `arrival_time`, `peak_stress`, `QoiFn`; case ids `W1D-<L>-<v>` from Task 5.
- Produces: `get_benchmark("wave_propagation_1d")` resolves a validated SPEC; `configs/wave_1d.toml` trains it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/benchmarks/test_wave_split.py
"""ADR-0025 wave-1d split: 12/2/2 interior holdout, no extrapolation."""

from structbench.benchmarks import get_benchmark
from structbench.benchmarks.wave_propagation_1d.benchmark import (
    TEST_INTERP,
    TRAIN,
    VAL,
)


def test_split_partitions_the_16_cases():
    all_ids = TRAIN + VAL + TEST_INTERP
    assert len(TRAIN) == 12 and len(VAL) == 2 and len(TEST_INTERP) == 2
    assert len(set(all_ids)) == 16


def test_split_cells_match_adr_0025():
    assert set(VAL) == {"W1D-300-2", "W1D-400-4"}
    assert set(TEST_INTERP) == {"W1D-300-4", "W1D-400-2"}
    assert all(c.startswith("W1D-200-") or c.startswith("W1D-500-")
               or c in {"W1D-300-1", "W1D-300-8", "W1D-400-1", "W1D-400-8"}
               for c in TRAIN)


def test_spec_resolves_with_no_extrapolation_split():
    spec = get_benchmark("wave_propagation_1d")
    assert spec.eval_splits == ("val", "test_interp")
    assert "test_extrap" not in spec.splits
    assert spec.aux_field == "axial_stress"
    assert spec.boundary_feature_fn is None
    assert set(spec.qois) == {
        "arrival_time_25", "arrival_time_50", "arrival_time_75", "peak_stress",
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `python.exe -m pytest tests/benchmarks/test_wave_split.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `benchmark.py`**

```python
"""The wave-1d benchmark: split, aux field, QoIs (ADR-0025)."""

from __future__ import annotations

from ...eval import QoiFn, arrival_time, peak_stress

_LENGTHS = (200, 300, 400, 500)
_VELOCITIES = (1, 2, 4, 8)


def _case(length: int, velocity: int) -> str:
    return f"W1D-{length}-{velocity}"


#: Fixed, immutable split (ADR-0025). Changing it is a new benchmark version.
VAL: list[str] = [_case(300, 2), _case(400, 4)]
TEST_INTERP: list[str] = [_case(300, 4), _case(400, 2)]
TRAIN: list[str] = [
    _case(length, velocity)
    for length in _LENGTHS
    for velocity in _VELOCITIES
    if _case(length, velocity) not in VAL + TEST_INTERP
]
ALL_BENCHMARK_CASES: list[str] = TRAIN + VAL + TEST_INTERP

#: Auxiliary per-particle target: the travelling stress wave IS the signal.
AUX_FIELD = "axial_stress"

#: ADR-0025 QoIs: gauge arrival times (ms) and global peak stress (MPa).
QOIS: dict[str, QoiFn] = {
    "arrival_time_25": arrival_time(0.25),
    "arrival_time_50": arrival_time(0.50),
    "arrival_time_75": arrival_time(0.75),
    "peak_stress": peak_stress,
}
```

- [ ] **Step 4: Implement `card.py`** (use the ACTUAL frame count from Task 5 for `n_frames` — the value below assumes the spec sheet's 300; correct it to what ingestion reported)

```python
"""Benchmark card for the wave-1d benchmark (ADR-0027)."""

from ..card import BenchmarkCard
from .benchmark import AUX_FIELD, QOIS, TEST_INTERP, TRAIN, VAL

CARD = BenchmarkCard(
    name="Wave1D-Propagation",
    version="0.1",
    description=(
        "Autoregressive next-step surrogate of an elastic stress wave in a "
        "2D SPH bar strip under initial-velocity excitation (ADR-0025). "
        "Entry tier: onboarding, tutorial, and fast CI."
    ),
    provenance=(
        "LS-DYNA parametric sweep (4 bar lengths x 4 initial velocities) "
        "produced by Curtin collaborators; benchmark protocol per ADR-0025."
    ),
    data_license="CC BY 4.0",
    solver="LS-DYNA",
    discretisation="SPH",
    materials=("*MAT_ELASTIC",),
    erosion=False,
    loading="initial velocity 1-8 mm/ms; elastic wave propagation",
    source_units="g-mm-ms",
    geometry="2D strip, 5 particle rows, {200, 300, 400, 500} mm x 8 mm",
    n_cases=len(TRAIN) + len(VAL) + len(TEST_INTERP),
    splits={
        "train": len(TRAIN),
        "val": len(VAL),
        "test_interp": len(TEST_INTERP),
    },
    task="autoregressive transition (ADR-0025)",
    aux_field=AUX_FIELD,
    aux_unit="MPa",
    qois=tuple(QOIS),
    fields=(
        "positions",
        "velocity",
        "acceleration",
        "stress",
        "effective_plastic_strain",
    ),
    particles_per_case="500-1250",
    n_frames=300,
    output_dt_ms=0.1,
)
```

- [ ] **Step 5: Implement `__init__.py` + registry entry**

```python
"""Wave-1d benchmark: entry-tier elastic wave propagation (ADR-0025)."""

from ..registry import BenchmarkSpec
from .benchmark import (
    ALL_BENCHMARK_CASES,
    AUX_FIELD,
    QOIS,
    TEST_INTERP,
    TRAIN,
    VAL,
)
from .card import CARD

__all__ = [
    "ALL_BENCHMARK_CASES",
    "AUX_FIELD",
    "CARD",
    "QOIS",
    "SPEC",
    "TEST_INTERP",
    "TRAIN",
    "VAL",
]

SPEC = BenchmarkSpec(
    card=CARD,
    splits={
        "train": tuple(TRAIN),
        "val": tuple(VAL),
        "test_interp": tuple(TEST_INTERP),
    },
    eval_splits=("val", "test_interp"),
    aux_field=AUX_FIELD,
    qois=dict(QOIS),
    boundary_feature_fn=None,
    dataset_id="1D-Wave-Propagation",
)
```

In `registry.py`:

```python
_MODULES: dict[str, str] = {
    "taylor_impact_2d": "structbench.benchmarks.taylor_impact_2d",
    "wave_propagation_1d": "structbench.benchmarks.wave_propagation_1d",
}
```

- [ ] **Step 6: Configs** (particle spacing is 2.0 mm — vs Taylor's 0.25 mm — so the connectivity radius scales accordingly; smoke mirrors `taylor_2d_smoke.toml`'s step count)

```toml
# configs/wave_1d.toml — wave-1d GNS baseline (ADR-0025)
benchmark = "wave_propagation_1d"
dim = 2
connectivity_radius = 4.8   # ~2.4x the 2.0 mm particle spacing
training_steps = 50000      # 16 small cases; half the Taylor budget
```

```toml
# configs/wave_1d_smoke.toml — 10-step smoke of the wave-1d pipeline
benchmark = "wave_propagation_1d"
dim = 2
connectivity_radius = 4.8
training_steps = 10
val_every = 5
batch_size = 2
```

(Any GNSConfig/TrainConfig keys not set fall back to defaults — check `taylor_2d_smoke.toml` for which keys it sets and mirror that shape.)

- [ ] **Step 7: Per-benchmark env vars in the card-data test** — rework `tests/benchmarks/test_card_data.py` to parametrize: keep `STRUCTBENCH_DATA_ROOT` for Taylor and add `STRUCTBENCH_WAVE1D_DATA_ROOT` for wave-1d; each case skips independently when its var is unset:

```python
_BENCHMARK_ROOTS = {
    "taylor_impact_2d": os.environ.get("STRUCTBENCH_DATA_ROOT"),
    "wave_propagation_1d": os.environ.get("STRUCTBENCH_WAVE1D_DATA_ROOT"),
}


@pytest.mark.parametrize("name", sorted(_BENCHMARK_ROOTS))
def test_card_matches_one_canonical_case(name):
    root = _BENCHMARK_ROOTS[name]
    if root is None:
        pytest.skip(f"data root env var for {name} not set")
    spec = get_benchmark(name)
    case = read_case(Path(root) / f"{spec.splits['train'][0]}.h5")
    lo, hi = (int(x) for x in spec.card.particles_per_case.split("-"))
    assert lo <= case.elements["sph"].element_id.shape[0] <= hi
    assert case.response is not None
    assert case.response.time.shape[0] == spec.card.n_frames
```

(The module-level `pytestmark` skipif is replaced by the per-case skip above.)

- [ ] **Step 8: Regenerate docs and run everything**

Run: `python.exe tools/gen_benchmark_docs.py` → docs/benchmarks.md gains the Wave1D row.
Run with data: `$env:STRUCTBENCH_WAVE1D_DATA_ROOT = "..\data\Concrete-Beam\1DWavePropagation\h5_canonical"; python.exe -m pytest tests/benchmarks -v; Remove-Item Env:STRUCTBENCH_WAVE1D_DATA_ROOT` — the wave card-data case must PASS (fix the CARD, not the test, if `n_frames`/particles disagree — then regenerate docs again).
Run: `python.exe -m pytest -q` — all pass.

- [ ] **Step 9: Commit**

```bash
git add src/structbench/benchmarks configs docs/benchmarks.md tests/benchmarks
git commit -m "feat: wave_propagation_1d benchmark — splits, card, registry, configs (ADR-0025)"
```

---

### Task 7: end-to-end smoke on the real data + full verification

**Files:** none created (scratch outputs only, under `scratch/` which is gitignored)

- [ ] **Step 1: Smoke-train on the ingested data (CPU, ~minutes)**

```powershell
& $py -m structbench.cli.train --mode train --config configs/wave_1d_smoke.toml `
  --data-root "..\data\Concrete-Beam\1DWavePropagation\h5_canonical" `
  --out scratch\wave1d-smoke
```

Expected: completes, writes `config.json` (with `"benchmark": "wave_propagation_1d"`), `normalization_stats.npz`, one `model-*.pt`.

- [ ] **Step 2: Validate + rollout**

```powershell
& $py -m structbench.cli.train --mode valid --data-root "..\data\Concrete-Beam\1DWavePropagation\h5_canonical" --out scratch\wave1d-smoke
& $py -m structbench.cli.train --mode rollout --data-root "..\data\Concrete-Beam\1DWavePropagation\h5_canonical" --out scratch\wave1d-smoke
```

Expected: `metrics-val.json` and `metrics-test_interp.json` written; each contains finite `rollout_aux_rmse` AND `one_step_aux_rmse`, `"aux_field": "axial_stress"`, and all four QoIs (`arrival_time_25/50/75`, `peak_stress`) with finite values. NO `test_extrap` file (spec has none). Quality of the 10-step model is irrelevant — this validates plumbing.

- [ ] **Step 3: Full verification**

Run, all clean: `python.exe -m pytest -q` (expect all pass; skips only for unset env vars + matplotlib-less viz) · `ruff check src tests tools` · `ruff format --check src tests tools` · `python.exe -m mypy src`.

- [ ] **Step 4: Update ROADMAP ingestion line** — in `ROADMAP.md`'s v0.2 definition of done, annotate the ingestion item: `(wave-1d: 16 cases ingested 2026-07-03; notch-beam pending)` and the benchmark-modules item: `(wave_propagation_1d done; notch pair pending)`. Commit:

```bash
git add ROADMAP.md
git commit -m "docs: ROADMAP marks wave-1d ingestion + module done"
```

- [ ] **Step 5: Hand off** — leave the branch for the confirm-gated merge; report metrics-file contents (the smoke QoI values), test counts, and any deviations. The REAL wave-1d training run (full `wave_1d.toml`) is a human-gated compute decision, like the Taylor DUG run.

---

## Post-plan notes

- **Deliberately deferred to Plan 3 (notch pair)**: the `damage` extractor; the README summary row (per Plan 1's note); the notch-beam conversion + stratified split freezing (ADR-0026).
- **Known interaction**: changing `QoiFn`'s signature (Task 3) touches the ADR-0019 QoI *plumbing* while preserving its frozen *values* — the Taylor split/QoI-value tests prove non-regression.
- **`_prepared_run` fixture** in `tests/cli/test_train_eval.py` still builds stats without an explicit `aux_field`; after Task 1 it must pass one (the required keyword makes this mechanical and self-announcing — the suite fails loudly at Task 1 Step 4 if missed).
- The wave dataset's hosting (like Taylor's) remains the open ROADMAP question — conversion writes to the local OneDrive tree only.
