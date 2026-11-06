# Design: the two-site state, the restart family and the properties block

*The design behind ADR-0075 (Proposed, 2026-11-07). Platform implementation
only. Written before the work; once built, `docs/ARCHITECTURE.md` and the
ADR describe what the code does.*

## 1. What changes, in one paragraph

Mesh cases from the Abaqus pipeline store the solver's constitutive state per
element (`response/element/<type>`: `stress` as Voigt components, `peeq`).
Today the loader projects those to nodes and every model family predicts,
and is scored on, nodal fields. After this design a benchmark definition may
declare an **element site**; the trajectory then carries the element state
verbatim, the simulator carries it between frames or anchors, an admissible
arm derives it by the material class's return map from the carried state and
the predicted kinematics, metrics and property kernels act on it directly,
and a restart task family starts rollouts from stored two-site states. The
nodal projection stays as the cheap baseline representation and as the
like-for-like row.

## 2. Data contract

`CaseTrajectory` gains two optional fields, populated only for mesh cases
whose benchmark declares an element site:

- `state: (T, E, S) float32` — the element-site channels in the working
  units (MPa for stress), read verbatim from the element block in declaration
  order; `E` = cells, `S` = declared channels.
- `state_channels: tuple[str, ...]` — the declared names, e.g.
  `("s_rr", "s_zz", "s_tt", "s_rz", "peeq")` for an axisymmetric solid.

`cells` already links elements to nodes. The nodal `aux` block is unchanged
(ADR-0059); when a benchmark declares an element site, its nodal `aux` is the
projection of `state` by the existing area-weighted mean, so the baseline
representation is derived from the same source and the two never disagree.

The case schema and HDF5 layout are untouched (ADR-0012, 0013, 0042).

## 3. Benchmark definition surface (`BenchmarkSpec`)

- `state_site: Literal["node", "element"] = "node"`.
- `state_channels: tuple[str, ...] = ()` — element-site channels; validated
  against the element block's extractors.
- `constitutive_law: Callable[[str], ConstitutiveLaw] | None` — the case's
  own law for the admissible decoder: for `elastic_plastic_isotropic`
  (ADR-0070) the elastic constants and the `(peeq, sigma_y)` knots. Replaces
  ADR-0064's single `hardening_curve` table; `hardening_curve` stays for the
  existing SPH heads and is rejected together with `constitutive_law`.
- `restart_anchors: Callable[[str], tuple[int, ...]] | tuple[int, ...]` and
  `restart_horizons: tuple[int, ...]` — declared like `scored_frames`, fixed
  or per case; anchors must lie inside `[input_frames - 1, scored_bound)`.
- `matched_initial_state_pairs: tuple[tuple[str, str], ...] = ()` — case-id
  pairs with identical kinematic seeds and different stored state, for the
  state-sufficiency property.

`BenchmarkSource` and `run.definition` (2026-11-06) already record the
definition's hashes, so a result points at the law and the anchors it used.

## 4. The admissible decoder (state carried by the simulator)

Per element `e`, with carried state `(σ_e, ε̄ᵖ_e)` at the anchor frame `a`
and predicted nodal positions at the query frame `t` (under an
autoregressive scheme `a = t − 1`; under the anchored flow map, ADR-0062,
`a` is the segment anchor):

1. **Kinematic increment.** The incremental deformation gradient of the
   element from its corner displacements between `a` and `t`, through the
   element's shape-function gradients in the configuration at `a`; the
   axisymmetric element adds the hoop stretch `1 + Δu_r / r` at its
   integration point. Polar decomposition gives the rotation increment `ΔR`
   and the logarithmic strain increment `Δε` (small-increment limit of the
   solver's corotational update).
2. **Trial state.** `σ_trial = ΔR σ_e ΔRᵀ + C : Δε` with the case's elastic
   constants.
3. **Return map.** `f = ‖dev σ_trial‖_vm − σ_y(ε̄ᵖ_e)`; if `f > 0`,
   `Δγ = f / (3G + H)` with `H` the local hardening slope from the knots,
   `σ = σ_trial − 2G Δγ n`, `ε̄ᵖ = ε̄ᵖ_e + Δγ`; else the trial state is the
   state. All operations are differentiable; the loss on the stored element
   state flows through them into the network.
4. **Carry.** `(σ, ε̄ᵖ)` at `t` become the carried state for the next frame
   or, at a re-anchor, the anchor state; the projected nodal view is
   recomputed for the network's next input.

What holds by construction: von Mises at or below `σ_y(ε̄ᵖ)` at every
emission; `ε̄ᵖ` non-decreasing along the carried chain; deviator traceless.
What does not: path dependence inside one increment (a single return map
over a frame or a segment is exact for proportional loading within it and
approximate otherwise). The composition sweep (§6) measures that cost.

The unconstrained two-site arm emits the element state from a head of width
`S` per element, fed by the corner nodes' latents (mean of the four), and is
scored identically; it isolates the decoder's effect.

## 5. Metrics at the state site

Per channel, ADR-0055 forms on the element block: `state_rel_l2_<ch>`,
`state_rmse_<ch>`, pooled `state_rel_l2`. The like-for-like row:
`state_proj_rel_l2_<ch>`, the prediction projected to nodes against the
projected reference, which is what nodal arms report under the existing
`rollout_rel_l2_aux` keys. Rollout `.npz` artifacts gain `predicted_state`
`(T, E, S)` when an element site is declared.

## 6. The restart family

For each declared anchor `a` and horizon `h`: load the stored two-site state
at `a` (nodal frames `a − input_frames + 1 .. a`, element state at `a`),
roll out to `a + h`, score on `(a, a + h]`. Keys: `restart_a{a}_h{h}_*`.
Held-out interval pairs: anchors and horizons not seen in training (the
flow map trains on sampled pairs; the AR scheme sees every unit step).
Composition sweep: a fixed interval `T` reached in `k = 1, 2, 4, …`
compositions, keys `compose_T{T}_k{k}_*`. Semigroup error, as in the
flow-map literature: `‖Φ_t(Φ_s(x)) − Φ_{s+t}(x)‖ / ‖Φ_{s+t}(x)‖` over
sampled `(s, t)`, key `semigroup_err`. A kinematics-only scheme runs the
family from the kinematic anchor only and its rows carry
`restart_kind = "kinematic"`.

## 7. Properties block

Computed in `eval/properties.py` from the rollout artifacts, each by one
kernel; where `verification/kernels.py` already has the kernel (yield
admissibility, monotonicity, trace) the eval module calls it, so the
reference data and the predictions are judged by the same function.

| Property | Key | Kernel | Tolerance |
|---|---|---|---|
| Yield bound | `prop_yield_violation_rate` | fraction of (frame, element) with `vm > (1 + tol) σ_y(peeq)` | `tol` = the reference's own measured excess from its datacheck record, never tighter |
| Plastic-strain monotonicity | `prop_peeq_decrease_rate` | fraction of decrements below `−tol` | same source |
| Deviator trace | `prop_trace_max` | max `|tr s| / σ_y0` | same source |
| State sufficiency | `prop_state_sufficiency` | fraction of the true divergence between a matched pair recovered from the stored state, pooled over the scored span | reported |
| Closure | `prop_closure_ratio` | free-rollout error over teacher-forced one-step error at the same frame | reported |
| Error growth | `prop_error_growth` | slope of log error against frame over the scored span | reported |
| Semigroup | `semigroup_err` | §6 | reported |

No property gates (ADR-0075 clause 6). Registries gain the block as a
column group rendered before the accuracy columns; landing pages the same;
results without the block show it empty.

## 8. Build order

- **3a** — data contract (§2), spec surface (§3), the decoder in the
  Transolver family for the anchored flow map and the AR scheme (§4), the
  unconstrained two-site head, state-site metrics (§5). Gate: on a mesh
  benchmark the admissible arm's by-construction properties read zero at the
  reference tolerance; the projected row of a nodal arm is byte-identical to
  its pre-3a metrics.
- **3b** — the restart family and the composition sweep (§6) in
  `eval/rollout.py` and the CLI; `restart_kind` on kinematics-only schemes.
- **3c** — the properties block (§7), the shared kernels, registries and
  landing pages.

Each part is a feature branch with the full suite as its gate; the ADR is
accepted or withdrawn in the session that merges 3a.
