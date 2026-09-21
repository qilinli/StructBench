# Reference-data verification: Taylor2D-Impact

- Dataset revision: v0.1.0
- Cases: 33
- Instrument: structbench 0.3.0, `structbench.verification/1`

Generated from the committed measurements and the platform criteria; no data was read to produce it. A `pass` on numerical health or conservation is a necessary solution-verification indicator, not evidence of accuracy. Verdicts come from definitional requirements and instrument tolerances only. Published reference levels are shown beside the measurement for context: none has been confirmed against its source by the maintainer, so none is this platform's standard and none gives a verdict.

## Verdicts

Quantities measured on at least one case.

| Quantity | Unit | Measured | pass | fail | review | n/a | not assessable | Criterion |
|---|---|---|---|---|---|---|---|---|
| `active_mass_drift` | 1 | 0 | 33 |  |  |  |  | <= 1e-06 any run (provisional) |
| `declared_traits_match_input` | 1 | 0 | 33 |  |  |  |  | <= 0 any run |
| `density_slot_matches_input` | 1 | 0 | 33 |  |  |  |  | <= 1e-05 any run (provisional) |
| `elements_without_input_part` | 1 | 1 |  | 33 |  |  |  | <= 0 any run |
| `energy_gain_max` | 1 | 0.0816209 … 0.186831 |  |  |  |  | 33 | none — out of scope: <= 0.01 {explicit, lagrangian_mesh} [B-BLM-1, B-BLM-2]; <= 0.01 {explicit, particle_conservative} [B-BLM-1, B-BLM-2] |
| `energy_loss_max` | 1 | 0 |  |  |  |  | 33 | none — out of scope: <= 0.01 {explicit, lagrangian_mesh} [B-BLM-1, B-BLM-2]; <= 0.01 {explicit, particle_conservative} [B-BLM-1, B-BLM-2] |
| `energy_residual_final` | 1 | 0.0536692 … 0.127847 |  |  |  |  | 33 | none |
| `fields_match_declaration` | 1 | 0 | 33 |  |  |  |  | <= 0 any run |
| `input_density_plausible` | kg/m^3 | 8900 |  |  |  |  | 33 | none — published level, not ratified: 16 <= x <= 22590 any run [M-D6, M-D5, M-D10] |
| `input_dimensionless_groups_plausible` | 1 | 0.00176731 |  |  |  |  | 33 | none |
| `input_strength_plausible` | Pa | 4.222e+08 |  |  |  |  | 33 | none — published level, not ratified: 10000 <= x <= 6.8e+09 any run [M-S1, M-S2, M-S3] |
| `kinetic_energy_closure` | 1 | 0.000379198 … 0.00304207 |  |  |  |  | 33 | none |
| `nonfinite_count` | 1 | 0 | 33 |  |  |  |  | <= 0 any run |
| `out_of_plane_shear_max` | Pa | 0 | 33 |  |  |  |  | <= 0 any run (provisional) |
| `particle_deactivated_count` | 1 | 0 | 33 |  |  |  |  | <= 0 any run |
| `particle_neighbors_growth` | 1 | 1.33333 … 2.125 |  |  |  |  | 33 | none |
| `particle_neighbors_min` | 1 | 5 … 6 |  |  |  |  | 33 | none |
| `plane_strain_ezz_max` | 1 | 0 | 33 |  |  |  |  | <= 1e-06 any run (provisional) |
| `pressure_trace_residual` | 1 | 4.38983e-08 … 7.56961e-08 |  |  |  |  | 33 | none |
| `reached_end_time` | 1 | 1 … 1.00026 | 33 |  |  |  |  | >= 0.999999 any run (provisional) |
| `response_magnitudes_plausible` | m/s | 184.449 … 415.648 |  |  |  |  | 33 | none — published level, not ratified: <= 3000 any run [M-S9] |
| `rigid_surface_penetration_max` | m | 0.000272232 … 0.000389357 |  |  |  |  | 33 | none |
| `smoothing_length_within_input_bounds` | 1 | 0 | 33 |  |  |  |  | <= 1e-05 any run (provisional) |
| `solver_error_count` | 1 | 0 | 33 |  |  |  |  | <= 0 any run |
| `solver_identity_complete` | 1 | 0 | 33 |  |  |  |  | <= 0 any run |
| `solver_warning_count` | 1 | 0 |  |  |  |  | 33 | none |
| `state_variable_decrease_max` | 1 | 0 | 33 |  |  |  |  | <= 0 any run |
| `state_variable_min` | 1 | 0 | 33 |  |  |  |  | >= 0 any run |
| `terminal_artifact_frames` | 1 | 1 |  |  |  |  | 33 | none |
| `terminated_normally` | 1 | 1 | 33 |  |  |  |  | >= 1 any run |
| `time_axis_monotone` | 1 | 0 | 33 |  |  |  |  | <= 0 any run |
| `timestep_min_ratio` | 1 | 0.912567 … 0.958889 |  |  |  |  | 33 | none |
| `total_energy_change_final` | 1 | 0.0536951 … 0.146223 |  |  |  |  | 33 | none — out of scope: -0.1 <= x <= 0.1 {explicit, initial_energy_driven, lagrangian_mesh} [W-W179-02, W-W179-12] |
| `yield_ratio_max` | 1 | 1.00032 … 1.00153 |  |  |  |  | 33 | none |
| `yield_saturation_min` | 1 | 1.00032 … 1.00153 | 33 |  |  |  |  | >= 0.5 any run (provisional) |
| `yield_table_covers_range` | 1 | 0.0530745 … 0.150588 | 33 |  |  |  |  | <= 1 any run |
| `yield_table_matches_input` | 1 | 0 | 33 |  |  |  |  | <= 1e-09 any run |
| `yield_table_monotone` | 1 | 1 |  | 33 |  |  |  | <= 0 any run |

## Findings

- **fail** `elements_without_input_part` = 1 — `T-20-100-100`, `T-20-100-110`, `T-20-100-120`, `T-20-100-130`, `T-20-100-140`, `T-20-100-150`, … (33 cases)
- **fail** `yield_table_monotone` = 1 — `T-20-100-100`, `T-20-100-110`, `T-20-100-120`, `T-20-100-130`, `T-20-100-140`, `T-20-100-150`, … (33 cases)

## Not applicable to these runs

`added_mass_fraction`, `added_mass_moving_fraction`, `added_mass_top_part_fraction`, `contact_energy_negative_ratio`, `contact_energy_ratio`, `implicit_convergence`, `input_constants_plausible`, `mass_closure`, `prescribed_motion_realised`, `quasi_static_kinetic_ratio`, `smoothing_length_at_bound_fraction`, `zero_energy_mode_final_over_initial_total`, `zero_energy_mode_final_over_internal_final`, `zero_energy_mode_peak_over_internal_peak`, `zero_energy_mode_top_part_final_over_internal_final`

## Evidence the runs did not supply

- `source_missing` (E7), 33 cases: `external_work_closure`, `momentum_impulse_balance`

## Not yet checked by this instrument

Gaps in the platform, not in the data: a quantity the instrument cannot measure yet, or a declaration it has no home for.

- `eos_closure` — unsupported
- `input_unit_declaration_consistent` — unsupported
- `internal_energy_closure` — unsupported
- `sampling_clock_consistent` — unsupported
- `stored_globals_match_ledger` — unsupported
- `timestep_vs_stability_estimate` — unsupported
- `units_anchors_consistent` — no_declaration_home

## Criteria

- `active_mass_drift` <= 1e-06 any run (instrument, provisional). With mass scaling and deletion off, every stored mass is one constant rounded the same way each frame; 1e-6 is a decade above float32 resolution.
- `declared_traits_match_input` <= 0 any run (requirement). What the benchmark declares about the run agrees with the solver input.
- `density_slot_matches_input` <= 1e-05 any run (instrument, provisional). At the first stored state an unloaded part has its input density, to float32 resolution (1.2e-7); a wrong slot is off by orders of magnitude. A preloaded first state needs its own bound.
- `elements_without_input_part` <= 0 any run (requirement). Every stored element belongs to a part the solver input defines.
- `fields_match_declaration` <= 0 any run (requirement). The stored fields are exactly the fields the benchmark declares.
- `nonfinite_count` <= 0 any run (requirement). A stored response contains no NaN or infinity.
- `out_of_plane_shear_max` <= 0 any run (instrument, provisional). A two-dimensional formulation carries no out-of-plane shear; the slots hold exact zeros, so any other value is a slot mix-up.
- `particle_deactivated_count` <= 0 any run (requirement). With erosion off, no particle is ever deactivated.
- `plane_strain_ezz_max` <= 1e-06 any run (instrument, provisional). Plane strain sets the out-of-plane normal strain to zero; 1e-6 is a decade above float32 resolution of a strain of order one, and three decades below the smallest hoop strain of an axisymmetric run.
- `reached_end_time` >= 0.999999 any run (instrument, provisional). The last stored time reaches the requested end time. float32 time stamps resolve 1.2e-7 of their value; 1e-6 leaves a decade. No upper bound: an explicit run may overshoot by one step.
- `smoothing_length_within_input_bounds` <= 1e-05 any run (instrument, provisional). The scale is a ratio of two float32 values, resolved to about 2.4e-7; 1e-5 leaves over a decade.
- `solver_error_count` <= 0 any run (requirement). The solver's record mentions no error.
- `solver_identity_complete` <= 0 any run (requirement). The record names the solver's version, revision, precision and parallel layout; none is missing.
- `state_variable_decrease_max` <= 0 any run (requirement). The material class says its state variable never decreases; rounding to float32 is monotone, so storage adds no tolerance.
- `state_variable_min` >= 0 any run (requirement). The material class bounds its state variable below by zero.
- `terminated_normally` >= 1 any run (requirement). Every phase and restart segment ends with the solver's normal-termination statement.
- `time_axis_monotone` <= 0 any run (requirement). Stored times strictly increase.
- `yield_saturation_min` >= 0.5 any run (requirement, provisional). A scale screen, not a return-mapping tolerance: among points loading through a stored frame, one must sit near the yield surface. One half is far below one and above 0.145, the reciprocal of the smallest stress factor between common consistent unit systems (psi against kPa, 6.9). Chosen after the test-bed value (about one) was known, not from it.
- `yield_table_covers_range` <= 1 any run (requirement). The largest state reached lies within the table, so no yield stress was extrapolated.
- `yield_table_matches_input` <= 1e-09 any run (requirement). The declared table equals the input's. Both reach SI through their own float64 conversion, so equality is taken to the nine significant digits the report stores.
- `yield_table_monotone` <= 0 any run (requirement). A tabulated yield stress does not fall with its state variable; a dip is an input defect.
