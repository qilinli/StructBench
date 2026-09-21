# Taylor2D-Impact — reference-data verification

Dataset revision v0.1.0 · 33 cases · 62 checks per case · structbench 0.3.0

This report says what was checked about the simulation runs behind this dataset, what was found, and what could not be checked. It is generated from a committed record of measurements; no verdict here rests on a number fitted to this dataset. A pass is a necessary sign of a healthy run, not evidence that the simulation matches reality.

## Summary

- **19 checks pass** wherever they apply.
- **2 findings**: dips in the input's hardening table; stored elements that no input part owns.
- **17 quantities are measured but not judged**: the platform has no confirmed criterion. The values are below for you to weigh.
- **9 checks could not be made**, because the runs did not keep the evidence or the instrument cannot do it yet.
- 15 checks do not apply to these runs.

| | Pass | Finding | Measured, not judged | Not checked | Not applicable |
|---|---|---|---|---|---|
| Data integrity | 6 | 2 | 1 | 2 |  |
| Numerical health of the runs | 6 |  | 5 | 1 | 10 |
| Energy and mass conservation | 1 |  | 5 | 4 | 4 |
| Material behaviour | 5 |  | 2 |  |  |
| Units and magnitudes | 1 |  | 4 | 2 | 1 |

## Findings

### Dips in the input's hardening table — fail

- Found: 1 in all 33 cases.
- Required: must be zero.
- What it means: the input's hardening table is not monotone (input defect).

### Stored elements that no input part owns — fail

- Found: 1 in all 33 cases.
- Required: must be zero.
- What it means: the case holds elements the input does not define.

## Results by category

### Data integrity

| Check | Result | Worst case | Verdict | Basis |
|---|---|---|---|---|
| Dips in the input's hardening table | 1 | all cases | **fail** (33 of 33) | must be zero |
| Stored elements that no input part owns | 1 | all cases | **fail** (33 of 33) | must be zero |
| Declared traits against the solver input | 0 | all cases | pass | must be zero |
| Declared yield table against the input's | 0 | all cases | pass | at most 1e-09 |
| Non-finite values (NaN, infinity) | 0 | all cases | pass | must be zero |
| Plastic strain reached, relative to the table's range | 0.0530745 to 0.150588 | `T-20-80-200` | pass | at most 1 |
| Stored fields against the declared field list | 0 | all cases | pass | must be zero |
| Time steps that go backward | 0 | all cases | pass | must be zero |
| Frames written off the sampling interval | 1 | all cases | not judged | no criterion |
| Energy ledger sampled on the field-output clock | — |  | not checked | this instrument cannot measure it yet |
| Stored global energies against the solver's ledger | — |  | not checked | this instrument cannot measure it yet |

### Numerical health of the runs

| Check | Result | Worst case | Verdict | Basis |
|---|---|---|---|---|
| Particles deactivated without erosion | 0 | all cases | pass | must be zero |
| Run reached its requested end time | 1 to 1.00026 | `T-20-60-140` | pass | at least 0.999999 (provisional tolerance) |
| Smoothing length within input bounds | 0 | all cases | pass | at most 1e-05 (provisional tolerance) |
| Solver errors | 0 | all cases | pass | must be zero |
| Solver terminated normally | yes | all cases | pass | must be yes |
| Solver version and precision recorded | 0 | all cases | pass | must be zero |
| Deepest penetration of a rigid surface | 272.2 µm to 389.4 µm | `T-20-100-140` | not judged | no criterion |
| Fewest neighbours of any particle | 5 to 6 | `T-20-100-100` | not judged | no criterion |
| Growth of the largest neighbour count | 1.33333 to 2.125 | `T-20-80-200` | not judged | no criterion |
| Smallest time step, relative to the first | 0.912567 to 0.958889 | `T-20-100-200` | not judged | no criterion |
| Solver warnings | 0 | all cases | not judged | no criterion |
| Time step against a stability estimate | — |  | not checked | this instrument cannot measure it yet |

Does not apply to these runs: hourglass energy vs initial energy; hourglass energy vs internal energy; hourglass energy, worst part; implicit increments accepted without convergence; mass added by mass scaling, moving parts; mass added by mass scaling, whole model; mass added by mass scaling, worst part; particles with smoothing length at a bound; peak hourglass vs peak internal energy; prescribed motion achieved.

### Energy and mass conservation

| Check | Result | Worst case | Verdict | Basis |
|---|---|---|---|---|
| Drift of the total mass | 0 % | all cases | pass | at most 0.0001 % (provisional tolerance) |
| Change in total energy, start to end | 5.37 % to 14.6 % | `T-20-60-200` | not judged | published level shown below, for context |
| Energy imbalance at the end of the run | 5.37 % to 12.8 % | `T-20-60-200` | not judged | no criterion |
| Kinetic energy: stored fields vs solver ledger | 0.0379 % to 0.304 % | `T-20-60-190` | not judged | no criterion |
| Largest energy gain during the run | 8.16 % to 18.7 % | `T-20-60-200` | not judged | published level shown below, for context |
| Largest energy loss during the run | 0 % | all cases | not judged | published level shown below, for context |
| External work against the applied loads | — |  | not checked | the runs did not supply applied loads and reaction forces over time |
| Internal energy: stored fields vs solver ledger | — |  | not checked | this instrument cannot measure it yet |
| Momentum change against applied impulse | — |  | not checked | the runs did not supply applied loads and reaction forces over time |
| Pressure against the equation of state | — |  | not checked | this instrument cannot measure it yet |

Does not apply to these runs: contact energy against internal energy; kinetic energy in a quasi-static run; mass bookkeeping with scaling or deletion; negative contact energy.

### Material behaviour

| Check | Result | Worst case | Verdict | Basis |
|---|---|---|---|---|
| Out-of-plane shear in a two-dimensional run | 0 Pa | all cases | pass | must be zero (provisional tolerance) |
| Out-of-plane strain in a plane-strain run | 0 | all cases | pass | at most 1e-06 (provisional tolerance) |
| Plastic state never decreases | 0 | all cases | pass | must be zero |
| Plastic state never negative | 0 | all cases | pass | at least 0 |
| Yielding material sits on the yield surface | 1.00032 to 1.00153 | `T-20-80-130` | pass | at least 0.5 (provisional tolerance) |
| Largest stress relative to the yield surface | 1.00032 to 1.00153 | `T-20-100-200` | not judged | no criterion |
| Stored pressure against the stress trace | 4.39e-06 % to 7.57e-06 % | `T-20-100-100` | not judged | no criterion |

### Units and magnitudes

| Check | Result | Worst case | Verdict | Basis |
|---|---|---|---|---|
| Stored density against the input's density | 0 % | all cases | pass | at most 0.001 % (provisional tolerance) |
| Largest first-yield strain in the input | 0.00176731 | all cases | not judged | no criterion |
| Largest speed in the response | 184.4 m/s to 415.6 m/s | `T-20-100-200` | not judged | published level shown below, for context |
| Most extreme input density | 8900 kg/m^3 | all cases | not judged | published level shown below, for context |
| Most extreme yield stress in the input | 422.2 MPa | all cases | not judged | published level shown below, for context |
| Declared unit anchors agree with the input | — |  | not checked | the benchmark has nowhere to declare it yet |
| The input's own unit declaration | — |  | not checked | this instrument cannot measure it yet |

Does not apply to these runs: most extreme young's modulus in the input.

## Measured, not judged

Numbers the platform reports without a verdict. Where a published limit exists it is shown for context only: it has not been confirmed against its source by the maintainer, so it is not this platform's standard.

- **Change in total energy, start to end**: 5.37 % to 14.6 % (worst: `T-20-60-200`). A problem here would mean that total energy changed between the start and the end of the run. *Published levels exist only for other kinds of run: between -10 % and 10 %, for explicit, initial energy driven, lagrangian mesh runs [W-W179-02, W-W179-12].*
- **Deepest penetration of a rigid surface**: 272.2 µm to 389.4 µm (worst: `T-20-100-140`). A problem here would mean that material passed through a rigid surface. *No criterion.*
- **Energy imbalance at the end of the run**: 5.37 % to 12.8 % (worst: `T-20-60-200`). A problem here would mean that the energy balance does not close. *No criterion.*
- **Fewest neighbours of any particle**: 5 to 6 (worst: `T-20-100-100`). A problem here would mean that particles ran short of neighbours (kernel starvation). *No criterion.*
- **Frames written off the sampling interval**: 1. A problem here would mean that frames were written off the sampling interval - a fact, not a defect: a solver writes its state at the termination time and loaders drop it (ADR-0028). *No criterion.*
- **Growth of the largest neighbour count**: 1.33333 to 2.125 (worst: `T-20-80-200`). A problem here would mean that particles clumped (neighbour counts grew). *No criterion.*
- **Kinetic energy: stored fields vs solver ledger**: 0.0379 % to 0.304 % (worst: `T-20-60-190`). A problem here would mean that stored velocities and masses do not reproduce the ledger's kinetic energy. *No criterion.*
- **Largest energy gain during the run**: 8.16 % to 18.7 % (worst: `T-20-60-200`). A problem here would mean that energy was created during the run. *Published levels exist only for other kinds of run: at most 1 %, for explicit, lagrangian mesh runs [B-BLM-1, B-BLM-2]; at most 1 %, for explicit, particle conservative runs [B-BLM-1, B-BLM-2].*
- **Largest energy loss during the run**: 0 %. A problem here would mean that energy went unaccounted for. *Published levels exist only for other kinds of run: at most 1 %, for explicit, lagrangian mesh runs [B-BLM-1, B-BLM-2]; at most 1 %, for explicit, particle conservative runs [B-BLM-1, B-BLM-2].*
- **Largest first-yield strain in the input**: 0.00176731. A problem here would mean that the input mixes units within a material definition. *No criterion.*
- **Largest speed in the response**: 184.4 m/s to 415.6 m/s (worst: `T-20-100-200`). A problem here would mean that a response speed is outside the structural-impact regime. *A published level exists, not confirmed by this platform: at most 3000 m/s [M-S9].*
- **Largest stress relative to the yield surface**: 1.00032 to 1.00153 (worst: `T-20-100-200`). A problem here would mean that stress lies outside the yield surface (export or post-processing error). *No criterion.*
- **Most extreme input density**: 8900 kg/m^3. A problem here would mean that density is outside the range of condensed matter (mass-unit error). *A published level exists, not confirmed by this platform: between 16 kg/m^3 and 22590 kg/m^3 [M-D6, M-D5, M-D10].*
- **Most extreme yield stress in the input**: 422.2 MPa. A problem here would mean that a yield or compressive strength is outside its plausible range. *A published level exists, not confirmed by this platform: between 10 kPa and 6.8 GPa [M-S1, M-S2, M-S3].*
- **Smallest time step, relative to the first**: 0.912567 to 0.958889 (worst: `T-20-100-200`). A problem here would mean that the time step collapsed during the run. *No criterion.*
- **Solver warnings**: 0. A problem here would mean that the solver reported warnings that may affect the result. *No criterion.*
- **Stored pressure against the stress trace**: 4.39e-06 % to 7.57e-06 % (worst: `T-20-100-100`). A problem here would mean that stored pressure disagrees with the stress trace. *No criterion.*

## What could not be checked

- Because the benchmark has nowhere to declare it yet: declared unit anchors agree with the input.
- Because the runs did not supply applied loads and reaction forces over time: external work against the applied loads; momentum change against applied impulse.
- Because this instrument cannot measure it yet: energy ledger sampled on the field-output clock; internal energy: stored fields vs solver ledger; pressure against the equation of state; stored global energies against the solver's ledger; the input's own unit declaration; time step against a stability estimate.

## Per-case values

The unjudged quantities that differ between cases, one row per case.

| Case | Change in total energy, start to end | Deepest penetration of a rigid surface | Energy imbalance at the end of the run | Fewest neighbours of any particle | Growth of the largest neighbour count | Kinetic energy: stored fields vs solver ledger | Largest energy gain during the run | Largest speed in the response | Largest stress relative to the yield surface | Smallest time step, relative to the first | Stored pressure against the stress trace |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `T-20-100-100` | 7.02 % | 294.5 µm | 6.57 % | 5 | 1.5 | 0.0658 % | 8.93 % | 184.4 m/s | 1.00039 | 0.958889 | 7.57e-06 % |
| `T-20-100-110` | 5.86 % | 297 µm | 5.59 % | 5 | 1.41667 | 0.0593 % | 8.16 % | 205.2 m/s | 1.00039 | 0.956154 | 7.13e-06 % |
| `T-20-100-120` | 5.54 % | 304.1 µm | 5.37 % | 5 | 1.41667 | 0.0794 % | 8.35 % | 222.6 m/s | 1.00041 | 0.952859 | 6.37e-06 % |
| `T-20-100-130` | 6.28 % | 357.7 µm | 6.11 % | 5 | 1.625 | 0.115 % | 9.5 % | 254.8 m/s | 1.00049 | 0.948203 | 5.74e-06 % |
| `T-20-100-140` | 6.28 % | 389.4 µm | 6.18 % | 5 | 1.70833 | 0.04 % | 9.73 % | 274.8 m/s | 1.00053 | 0.942126 | 5.37e-06 % |
| `T-20-100-150` | 5.37 % | 342.6 µm | 5.37 % | 5 | 1.58333 | 0.091 % | 8.93 % | 297.1 m/s | 1.0009 | 0.940069 | 5.31e-06 % |
| `T-20-100-160` | 5.65 % | 349.4 µm | 5.65 % | 5 | 1.70833 | 0.0379 % | 9.18 % | 318.2 m/s | 1.00074 | 0.936068 | 4.69e-06 % |
| `T-20-100-170` | 6.76 % | 370.2 µm | 6.76 % | 5 | 1.75 | 0.119 % | 10.5 % | 346.7 m/s | 1.00093 | 0.926888 | 4.44e-06 % |
| `T-20-100-180` | 6.42 % | 362.8 µm | 6.42 % | 6 | 1.79167 | 0.0611 % | 10.1 % | 360.6 m/s | 1.00084 | 0.930186 | 4.39e-06 % |
| `T-20-100-190` | 6.92 % | 335.3 µm | 6.92 % | 5 | 1.875 | 0.182 % | 11 % | 392.7 m/s | 1.00089 | 0.923667 | 5.7e-06 % |
| `T-20-100-200` | 6.8 % | 348 µm | 6.8 % | 5 | 2.08333 | 0.119 % | 11.1 % | 415.6 m/s | 1.00153 | 0.912567 | 6.76e-06 % |
| `T-20-60-100` | 13.6 % | 279.5 µm | 12 % | 5 | 1.33333 | 0.11 % | 14.4 % | 184.4 m/s | 1.00035 | 0.958889 | 7.57e-06 % |
| `T-20-60-110` | 12 % | 275.6 µm | 10.7 % | 5 | 1.33333 | 0.0989 % | 13.3 % | 205.2 m/s | 1.00034 | 0.956154 | 7.14e-06 % |
| `T-20-60-120` | 12 % | 272.2 µm | 10.8 % | 5 | 1.33333 | 0.132 % | 13.7 % | 222.6 m/s | 1.00046 | 0.952859 | 6.37e-06 % |
| `T-20-60-130` | 13.6 % | 333.2 µm | 12 % | 5 | 1.375 | 0.191 % | 15.7 % | 254.8 m/s | 1.00038 | 0.948203 | 5.74e-06 % |
| `T-20-60-140` | 13.7 % | 334.4 µm | 12 % | 5 | 1.375 | 0.0667 % | 16.2 % | 274.8 m/s | 1.00047 | 0.942126 | 5.37e-06 % |
| `T-20-60-150` | 12 % | 333 µm | 10.7 % | 5 | 1.41667 | 0.152 % | 14.9 % | 297.1 m/s | 1.0009 | 0.940069 | 5.31e-06 % |
| `T-20-60-160` | 12.1 % | 331.4 µm | 10.8 % | 5 | 1.41667 | 0.0626 % | 15.3 % | 318.2 m/s | 1.00066 | 0.936068 | 4.69e-06 % |
| `T-20-60-170` | 14 % | 351.1 µm | 12.3 % | 5 | 1.625 | 0.198 % | 17.6 % | 346.7 m/s | 1.00093 | 0.926888 | 5.55e-06 % |
| `T-20-60-180` | 13.5 % | 301.2 µm | 11.9 % | 5 | 1.66667 | 0.102 % | 17.1 % | 360.6 m/s | 1.00084 | 0.930186 | 4.39e-06 % |
| `T-20-60-190` | 14.6 % | 298.7 µm | 12.7 % | 5 | 1.75 | 0.304 % | 18.4 % | 392.7 m/s | 1.00066 | 0.923667 | 5.7e-06 % |
| `T-20-60-200` | 14.6 % | 316.9 µm | 12.8 % | 5 | 1.83333 | 0.198 % | 18.7 % | 415.6 m/s | 1.00153 | 0.912567 | 6.76e-06 % |
| `T-20-80-100` | 9.55 % | 288.4 µm | 8.73 % | 5 | 1.33333 | 0.0822 % | 11 % | 184.4 m/s | 1.00037 | 0.958889 | 7.57e-06 % |
| `T-20-80-110` | 8.24 % | 289.8 µm | 7.63 % | 5 | 1.375 | 0.0742 % | 10.1 % | 205.2 m/s | 1.00036 | 0.956154 | 7.14e-06 % |
| `T-20-80-120` | 7.91 % | 297.6 µm | 7.33 % | 5 | 1.33333 | 0.0992 % | 10.3 % | 222.6 m/s | 1.00042 | 0.952859 | 6.37e-06 % |
| `T-20-80-130` | 8.88 % | 331.6 µm | 8.18 % | 5 | 1.5 | 0.143 % | 11.8 % | 254.8 m/s | 1.00032 | 0.948203 | 5.74e-06 % |
| `T-20-80-140` | 8.85 % | 326.2 µm | 8.17 % | 5 | 1.54167 | 0.05 % | 12.1 % | 274.8 m/s | 1.00053 | 0.942126 | 5.01e-06 % |
| `T-20-80-150` | 7.45 % | 343 µm | 7 % | 5 | 1.58333 | 0.114 % | 11.1 % | 297.1 m/s | 1.0009 | 0.940069 | 5.31e-06 % |
| `T-20-80-160` | 7.81 % | 313.9 µm | 7.34 % | 5 | 1.66667 | 0.047 % | 11.5 % | 318.2 m/s | 1.00072 | 0.936068 | 4.69e-06 % |
| `T-20-80-170` | 9.31 % | 323.1 µm | 8.68 % | 5 | 1.75 | 0.149 % | 13.1 % | 346.7 m/s | 1.00093 | 0.926888 | 4.44e-06 % |
| `T-20-80-180` | 8.83 % | 317.5 µm | 8.32 % | 6 | 1.79167 | 0.0764 % | 12.7 % | 360.6 m/s | 1.00084 | 0.930186 | 4.39e-06 % |
| `T-20-80-190` | 9.75 % | 323.4 µm | 9.17 % | 5 | 1.75 | 0.228 % | 13.7 % | 392.7 m/s | 1.00103 | 0.923667 | 5.7e-06 % |
| `T-20-80-200` | 9.52 % | 361.6 µm | 9.05 % | 6 | 2.125 | 0.148 % | 14 % | 415.6 m/s | 1.00153 | 0.912567 | 6.76e-06 % |

## How to read this report

- **pass / fail** — the value was compared with a bound that is definitional (zero, equality, the input's own value) or computed from storage precision. None was fitted to this dataset.
- **not judged** — the value is reported, and no bound is applied. Sourced limits from the literature are shown for context until the maintainer has confirmed them against their sources; none has been.
- **not checked** — the check needs evidence the runs did not keep, or the instrument cannot make it yet. The reason is given.
- **not applicable** — the quantity does not exist for this kind of run (hourglass energy in a particle model, for instance).

Bounds applied:

- *Declared traits against the solver input* — must be zero. What the benchmark declares about the run agrees with the solver input.
- *Declared yield table against the input's* — at most 1e-09. The declared table equals the input's. Both reach SI through their own float64 conversion, so equality is taken to the nine significant digits the report stores.
- *Dips in the input's hardening table* — must be zero. A tabulated yield stress does not fall with its state variable; a dip is an input defect.
- *Drift of the total mass* — at most 0.0001 % (provisional). With mass scaling and deletion off, every stored mass is one constant rounded the same way each frame; 1e-6 is a decade above float32 resolution.
- *Non-finite values (NaN, infinity)* — must be zero. A stored response contains no NaN or infinity.
- *Out-of-plane shear in a two-dimensional run* — must be zero (provisional). A two-dimensional formulation carries no out-of-plane shear; the slots hold exact zeros, so any other value is a slot mix-up.
- *Out-of-plane strain in a plane-strain run* — at most 1e-06 (provisional). Plane strain sets the out-of-plane normal strain to zero; 1e-6 is a decade above float32 resolution of a strain of order one, and three decades below the smallest hoop strain of an axisymmetric run.
- *Particles deactivated without erosion* — must be zero. With erosion off, no particle is ever deactivated.
- *Plastic state never decreases* — must be zero. The material class says its state variable never decreases; rounding to float32 is monotone, so storage adds no tolerance.
- *Plastic state never negative* — at least 0. The material class bounds its state variable below by zero.
- *Plastic strain reached, relative to the table's range* — at most 1. The largest state reached lies within the table, so no yield stress was extrapolated.
- *Run reached its requested end time* — at least 0.999999 (provisional). The last stored time reaches the requested end time. float32 time stamps resolve 1.2e-7 of their value; 1e-6 leaves a decade. No upper bound: an explicit run may overshoot by one step.
- *Smoothing length within input bounds* — at most 1e-05 (provisional). The scale is a ratio of two float32 values, resolved to about 2.4e-7; 1e-5 leaves over a decade.
- *Solver errors* — must be zero. The solver's record mentions no error.
- *Solver terminated normally* — must be yes. Every phase and restart segment ends with the solver's normal-termination statement.
- *Solver version and precision recorded* — must be zero. The record names the solver's version, revision, precision and parallel layout; none is missing.
- *Stored density against the input's density* — at most 0.001 % (provisional). At the first stored state an unloaded part has its input density, to float32 resolution (1.2e-7); a wrong slot is off by orders of magnitude. A preloaded first state needs its own bound.
- *Stored elements that no input part owns* — must be zero. Every stored element belongs to a part the solver input defines.
- *Stored fields against the declared field list* — must be zero. The stored fields are exactly the fields the benchmark declares.
- *Time steps that go backward* — must be zero. Stored times strictly increase.
- *Yielding material sits on the yield surface* — at least 0.5 (provisional). A scale screen, not a return-mapping tolerance: among points loading through a stored frame, one must sit near the yield surface. One half is far below one and above 0.145, the reciprocal of the smallest stress factor between common consistent unit systems (psi against kPa, 6.9). Chosen after the test-bed value (about one) was known, not from it.

Published levels shown for context (not applied). The bracketed ids are claims in the source dossier, `docs/plans/2026-09-21-reference-data-verification-sources.md`:

- *Change in total energy, start to end* — between -10 % and 10 %, for explicit, initial energy driven, lagrangian mesh runs [W-W179-02, W-W179-12]. Total energy 'must not vary more than 10 percent from the beginning of the run to the end'; the denominator is the initial total and the source has no external-work term. Roadside-crash practice with finite elements. Fixed from the source alone; one test-bed value (a legacy particle run whose total energy rises by several percent) was known beforehand.
- *Largest energy gain during the run* — at most 1 %, for explicit, lagrangian mesh runs [B-BLM-1, B-BLM-2]. Over-the-run energy residual 'generally on the order of 10^-2', rendered as the one number the source prints. A stability check, not an accuracy limit; the source checks every step, so a sampled pass is weaker. Read as snippets only. Fixed from the source alone; one test-bed value (a legacy particle run whose total energy rises by several percent) was known beforehand.
- *Largest energy gain during the run* — at most 1 %, for explicit, particle conservative runs [B-BLM-1, B-BLM-2]. Over-the-run energy residual 'generally on the order of 10^-2', rendered as the one number the source prints. A stability check, not an accuracy limit; the source checks every step, so a sampled pass is weaker. Read as snippets only. Fixed from the source alone; one test-bed value (a legacy particle run whose total energy rises by several percent) was known beforehand.
- *Largest energy loss during the run* — at most 1 %, for explicit, lagrangian mesh runs [B-BLM-1, B-BLM-2]. Over-the-run energy residual 'generally on the order of 10^-2', rendered as the one number the source prints. A stability check, not an accuracy limit; the source checks every step, so a sampled pass is weaker. Read as snippets only. Fixed from the source alone; one test-bed value (a legacy particle run whose total energy rises by several percent) was known beforehand.
- *Largest energy loss during the run* — at most 1 %, for explicit, particle conservative runs [B-BLM-1, B-BLM-2]. Over-the-run energy residual 'generally on the order of 10^-2', rendered as the one number the source prints. A stability check, not an accuracy limit; the source checks every step, so a sampled pass is weaker. Read as snippets only. Fixed from the source alone; one test-bed value (a legacy particle run whose total energy rises by several percent) was known beforehand.
- *Largest speed in the response* — at most 3000 m/s [M-S9]. Above 3 km/s is the hypervelocity regime, outside structural impact. Blind to the mass unit.
- *Most extreme input density* — between 16 kg/m^3 and 22590 kg/m^3 [M-D6, M-D5, M-D10]. Flexible polymer foam to osmium. The only family-free screen that sees a mass-unit error.
- *Most extreme yield stress in the input* — between 10 kPa and 6.8 GPa [M-S1, M-S2, M-S3]. Polymer foam to tungsten carbide in compression: sees gross stress-unit errors only, and needs no declared material family.
