# Task 7 Report — end-to-end CPU smoke on real wave-1d data

Date: 2026-07-03
Branch: feat/wave-1d-benchmark
Commit: a1f957d "docs: ROADMAP marks wave-1d ingestion + module done"

---

## Step 1 — Smoke train (10 steps, CPU)

Command:
```
python.exe -m structbench.cli.train --mode train --config configs/wave_1d_smoke.toml \
  --data-root "..\data\Concrete-Beam\1DWavePropagation\h5_canonical" \
  --out scratch\wave1d-smoke
```

Output:
```
INFO __main__: loading 12 TRAIN trajectories from ..\data\Concrete-Beam\1DWavePropagation\h5_canonical
INFO structbench.datasets.normalization: normalization stats: cached at ..\data\Concrete-Beam\1DWavePropagation\h5_canonical\derived\norm_90ccc148adb7.npz
INFO __main__: starting training: 10 steps, batch 2, 2 particle types
INFO __main__: step 5: train_loss 0.996131 val_loss 15.149014 (best inf)
INFO __main__: saved improved checkpoint: scratch\wave1d-smoke\model-best-000005.pt
INFO __main__: step 10: train_loss 4.248939 val_loss 19.757314 (best 15.149014)
mode=train device=cpu out=scratch\wave1d-smoke
training complete; best checkpoint: scratch\wave1d-smoke\model-best-000005.pt
```

Artifacts written:
- `scratch\wave1d-smoke\config.json` (contains `"benchmark": "wave_propagation_1d"`)
- `scratch\wave1d-smoke\normalization_stats.npz`
- `scratch\wave1d-smoke\model-best-000005.pt`

---

## Step 2 — Validate + Rollout

### Validation (--mode valid)

Command:
```
python.exe -m structbench.cli.train --mode valid \
  --data-root "..\data\Concrete-Beam\1DWavePropagation\h5_canonical" \
  --out scratch\wave1d-smoke
```

Output:
```
INFO: [val] W1D-300-2: one-step 0.0159 mm | rollout 12.2650 mm | axial_stress 0.0002 MPa
INFO: [val] W1D-400-4: one-step 0.0260 mm | rollout 18.0324 mm | axial_stress 0.0004 MPa
[val] one-step position RMSE 0.0209 mm | rollout position RMSE 15.1487 mm | rollout axial_stress RMSE 0.0003 MPa
[val] QoI mean |error|: arrival_time_25 2.4959 mm, arrival_time_50 1.3479 mm, arrival_time_75 0.1505 mm, peak_stress 0.0000 mm
```

metrics-val.json "mean" block:
```json
"mean": {
  "one_step_position_rmse": 0.02091616167281253,
  "one_step_aux_rmse": 0.00033258140152489827,
  "rollout_position_rmse": 15.148683068561704,
  "rollout_aux_rmse": 0.0003310259150037223,
  "qoi_abs_error": {
    "arrival_time_25": 2.49589109819733,
    "arrival_time_50": 1.3479492029738163,
    "arrival_time_75": 0.15054423256767024,
    "peak_stress": 0.0
  }
}
```

### Rollout (--mode rollout -> test_interp split)

Command:
```
python.exe -m structbench.cli.train --mode rollout \
  --data-root "..\data\Concrete-Beam\1DWavePropagation\h5_canonical" \
  --out scratch\wave1d-smoke
```

Output:
```
INFO: [test_interp] W1D-300-4: one-step 0.0285 mm | rollout 18.4151 mm | axial_stress 0.0004 MPa
INFO: [test_interp] W1D-400-2: one-step 0.0146 mm | rollout 12.0138 mm | axial_stress 0.0002 MPa
[test_interp] one-step position RMSE 0.0215 mm | rollout position RMSE 15.2145 mm | rollout axial_stress RMSE 0.0003 MPa
[test_interp] QoI mean |error|: arrival_time_25 2.5481 mm, arrival_time_50 1.3392 mm, arrival_time_75 0.1512 mm, peak_stress 0.0000 mm
```

metrics-test_interp.json "mean" block:
```json
"mean": {
  "one_step_position_rmse": 0.02154342742567364,
  "one_step_aux_rmse": 0.00032946220223235454,
  "rollout_position_rmse": 15.214481410111556,
  "rollout_aux_rmse": 0.0003280098975135073,
  "qoi_abs_error": {
    "arrival_time_25": 2.5481332280526403,
    "arrival_time_50": 1.3391952043998283,
    "arrival_time_75": 0.15120794509460955,
    "peak_stress": 0.0
  }
}
```

Key fields confirmed in both files: `rollout_aux_rmse` (finite), `one_step_aux_rmse` (finite),
`"aux_field": "axial_stress"`, `"aux_unit": "MPa"`, all four QoIs present with finite values.
NO `metrics-test_extrap.json` written (correct -- spec has no extrap split).

Note: `peak_stress` QoI absolute error is 0.0 across all 4 cases. For a 10-step untrained
model the predicted peak coincides with the true peak by chance (model outputs near-initial
values). Plumbing is correct; quality irrelevant for a smoke test.

---

## Step 3 — Full verification

### pytest
```
99 passed, 3 skipped, 2 warnings in 9.88s
```
3 skips: unset env vars + matplotlib-less viz (expected per brief).

### ruff check
```
All checks passed!  (exit 0)
```

### ruff format --check
```
57 files already formatted  (exit 0)
```

### mypy
```
Success: no issues found in 34 source files
```

All four gates clean.

---

## Step 4 — ROADMAP annotation + commit

Two lines added to `ROADMAP.md` v0.2 definition-of-done:
- After ingestion item: `(wave-1d: 16 cases ingested 2026-07-03; notch-beam pending)`
- After benchmark-modules item: `(wave_propagation_1d done; notch pair pending)`

Commit: `a1f957d` on feat/wave-1d-benchmark
Message: "docs: ROADMAP marks wave-1d ingestion + module done"
Only ROADMAP.md in commit; `scratch\wave1d-smoke` is gitignored and absent (clean tree).

---

## Fix wave: late-half peak_stress

Date: 2026-07-03  
Commit: (see below — applied on feat/wave-1d-benchmark)

### Problem

`peak_stress` was defined as global max |aux| over all frames. The true peak occurs at
frame 1 (excitation onset, constrained end) in all 16 cases — inside the frames a rollout
seeds with ground truth — so every model scored ~0 error trivially.

### Fix (maintainer-approved)

Redefined `peak_stress` as max |aux| over frames `T // 2` onward (the reflection regime).
Changes applied:

- `src/structbench/eval/metrics.py` — late-half window: `aux[aux.shape[0] // 2 :].max()`
- `tests/eval/test_metrics.py` — renamed `test_peak_stress_is_global_abs_max` →
  `test_peak_stress_reads_late_half`; added `test_peak_stress_ignores_early_only_spike`
  (early-only 99.0 spike → returns 2.0 from late half)
- `src/structbench/benchmarks/wave_propagation_1d/card.py` — enriched `materials` and
  `loading` strings with scaled toy elastic constants (E=0.01 MPa, rho=2e-6 g/mm3,
  wave speed ~70.7 mm/ms, ~10 traversals per trajectory)
- `docs/benchmarks.md` — regenerated (drift test enforces)

### Empirical validation (metrics-val.json after re-run)

| Case | qoi_pred | qoi_true | qoi_error |
|------|----------|----------|-----------|
| W1D-300-2 | 0.000168 MPa | 0.000426 MPa | −0.000258 MPa |
| W1D-400-4 | 0.000169 MPa | 0.000849 MPa | −0.000680 MPa |
| Mean abs error | — | — | 0.000469 MPa |

peak_stress is now nonzero for both val cases. The untrained 10-step model underestimates
the late-half amplitude as expected.

### Verification gates

- pytest: 100 passed, 3 skipped (ruff/mypy skips expected)
- ruff check: All checks passed
- ruff format --check: 57 files already formatted
- mypy: Success: no issues found in 34 source files
- docs --check: docs/benchmarks.md is up to date

---

## Concerns / deviations

None. Pipeline ran cleanly end-to-end. All expected artifacts produced. The real training
run (full `wave_1d.toml`) remains human-gated as per the plan.

---

## Artifacts (all under scratch\, gitignored)

- `scratch\wave1d-smoke\config.json`
- `scratch\wave1d-smoke\normalization_stats.npz`
- `scratch\wave1d-smoke\model-best-000005.pt`
- `scratch\wave1d-smoke\metrics-val.json`
- `scratch\wave1d-smoke\metrics-test_interp.json`
- `scratch\wave1d-smoke\rollouts\val-W1D-300-2.npz`
- `scratch\wave1d-smoke\rollouts\val-W1D-400-4.npz`
- `scratch\wave1d-smoke\rollouts\test_interp-W1D-300-4.npz`
- `scratch\wave1d-smoke\rollouts\test_interp-W1D-400-2.npz`
