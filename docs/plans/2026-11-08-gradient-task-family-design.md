# Design: the gradient task family on the Taylor sweep (ADR-0076)

*The design behind ADR-0076 (Proposed 2026-11-08). Written before the
build; the record holds the judgments, this note holds the detail and what
the build must verify before the family is scored. Platform implementation
only.*

## What is measured

For a scalar design parameter `p` the sweep varies (Taylor: impact velocity
`v`, eleven values 100–200 m/s at 10 m/s steps, for each bar length) and a
quantity of interest `Q` (final length, mushroom width, peak mean von Mises,
its time), the *sensitivity* is `dQ/dp` at a grid value.

**Reference.** From the archive's reference trajectories:

- interior values: `s_ref(v) = (Q(v+h) − Q(v−h)) / 2h`, `h = 10 m/s`;
- ends of the line (100, 200): one-sided, `(Q(v) − Q(v∓h)) / ±h`;
- truncation error: `e_trunc(v) = |s_h(v) − s_2h(v)|`, the same central
  difference at `2h = 20 m/s` (one-sided ends use the second one-sided
  estimate at `2h`); this is a Richardson-style estimate of the leading
  error term, not a bound;
- measurement noise: `e_meas(v) = √2 · σ_Q / 2h`, where `σ_Q` is the
  quantity's own discretisation noise; for the von Mises quantities the
  card's declared floor (~0.12 relative L2 on the field) propagated to the
  particle-mean peak, for the geometric quantities the particle spacing
  (half a spacing, 0.5 mm at the benchmark resolution) — both stated as
  assumptions in the record and revisited when the convergence case is
  measured (README Roadmap);
- a value is *judged* when `max(e_trunc, e_meas) ≤ ½ |s_ref|`, otherwise
  *reported*: its numbers render, it is excluded from the pooled headline
  and marked.

The reference values, both error terms and the judged/reported flag are
computed once by a tool (`tools/gradient_reference.py`) and committed as
`docs/gradients/taylor_impact_2d.json` with a generated Markdown rendering
beside it, like the datacheck records (ADR-0066): a reader sees the
reference's own uncertainty before any model's number. The tool is
re-runnable against the archive and a drift test compares.

**Surrogate.** For a model under the benchmark's own protocol (6 seed
frames, full horizon, native output times):

1. construct seeds at `v`, `v+δ`, `v−δ`: the stored frame-0 lattice of the
   case at `v`, translated by `v'·t_k` for frames `k = 0…5`, velocities set
   to `v'`, the wall and every other kinematic feature as in the stored
   case. **To verify in the build, before anything is scored**: that the
   stored seed frames of every Taylor case are free flight to float32
   (positions linear in time, velocities constant, zero stress) — the
   timeline analysis says contact is near frame 7, and the construction is
   exact only if frames 0–5 carry no contact in any case;
2. roll out from each seed with the model's registered recipe, compute the
   four quantities of interest with the benchmark's `QOIS`;
3. `s_model(v) = (Q(v+δ) − Q(v−δ)) / 2δ` at `δ = 1 m/s` and `δ = 2 m/s`;
   the registered value is the `δ = 1` one, and `|s_1 − s_2|` is the
   model's own gradient floor, reported beside it;
4. optionally, a model whose parameter path is differentiable (the
   time-conditioned Transolver's `impact_velocity_feature`, ADR-0051) also
   reports autodiff `dQ/dv` through the rollout; stored under a separate
   key, never the registered value.

Seeds at ±δ are off the grid, so no reference exists there and none is
needed: the model is differenced against itself.

**Metrics** (keys in the ADR-0055 style, per quantity of interest):

- `gradient_rel_l2_<qoi>` — pooled relative L2 over the judged scored
  values: `‖s_model − s_ref‖₂ / ‖s_ref‖₂`;
- `gradient_sign_<qoi>` — fraction of judged values where
  `sign(s_model) = sign(s_ref)`;
- `gradient_floor_<qoi>` — the model's own floor, pooled the same way
  against `s_ref`;
- per-value rows: `v`, set, coverage (ADR-0074: 200 m/s lies beyond the
  training box 100–190 and 100 m/s on its edge; how the edge reads is the
  coverage kernel's call, not this note's), `s_ref ± e_trunc, e_meas`,
  judged flag, `s_model`, floor, autodiff.

Scored values: all eleven per length; the headline pools the values of
`test_interp` and `test_extrap` (130, 170; 200); train and val values are
reported as diagnostics. The 200 m/s value is one-sided and beyond
coverage: shown in its own column, never pooled with the interior.

## Surface

- `benchmarks/registry.py`: `BenchmarkSpec.gradient_family: GradientFamily
  | None` — the parameter name, how it reads from a case id or `cases.csv`
  (Taylor: the velocity token of `T-20-<L>-<v>`), the grid values, the
  seed constructor, the δ pair. Taylor declares it; the other three are
  `None`.
- `eval/sensitivity.py`: the difference estimators with their errors, the
  judged rule, the metrics; pure functions over arrays of `Q(v)`.
- `eval/rollout.py` or a sibling: the perturbed-seed rollout driver, which
  reuses the existing rollout path unchanged.
- `cli/train.py --mode gradient`: runs a registered run directory's
  checkpoint through the driver and writes `metrics-gradient.json` and the
  per-value rows.
- `tools/gradient_reference.py`: the reference record under
  `docs/gradients/`.
- Results registries: `BaselineResult.metrics` already keys by split; the
  gradient group goes under a `gradient` key with the metrics above; the
  renderer adds a *Sensitivity* tier after the quantities of interest and
  before the properties block's place (ADR-0075 part 3c decides the final
  order).

## Not decided here

The coverage kernel's reading of the 100 m/s edge (ADR-0074's plan); the
rendering order against the properties block (ADR-0075's plan 3c); whether
the gradient family ever gates anything (no, until a note says otherwise);
a second parameter or benchmark.

## Build plan (after the record is Accepted)

1. The free-flight verification of the stored seeds, as a test over the
   archive (environment-gated like the card-data test) and a one-line
   finding in this note.
2. `eval/sensitivity.py` with unit tests on analytic `Q(v)` (a quadratic:
   central difference exact to rounding, truncation estimate zero; a cubic:
   the Richardson estimate recovers the known error).
3. `tools/gradient_reference.py` and the committed Taylor record.
4. The perturbed-seed driver and `--mode gradient`, smoke-tested on a
   synthetic case with a linear response.
5. One registered model run through it (the blessed CGN), its rows in the
   registry under the new key, the renderer tier, pages regenerated.
6. Gate: full suite green; the record's reference values reproduce from
   the archive; the first model's numbers reported, whatever they say.
