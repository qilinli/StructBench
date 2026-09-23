# NotchBeam2D-Impact — reference-data verification

Dataset revision v0.1.0 · 110 cases · 62 checks per case · structbench 0.3.0

This report says what was checked about the simulation runs behind this dataset, what was found, and what could not be checked. It is generated from a committed record of measurements; no verdict here rests on a number fitted to this dataset. A pass is a necessary sign of a healthy run, not evidence that the simulation matches reality.

## Summary

- **14 checks pass** wherever they apply.
- **No findings.**
- **6 quantities are measured but not judged**: the platform has no confirmed criterion. The values are below for you to weigh.
- **26 checks could not be made**, because the runs did not keep the evidence or the instrument cannot do it yet.
- 16 checks do not apply to these runs.

| | Pass | Finding | Measured, not judged | Not checked | Not applicable |
|---|---|---|---|---|---|
| Data integrity | 5 |  | 1 | 5 |  |
| Numerical health of the runs | 6 |  | 3 | 3 | 10 |
| Energy and mass conservation | 1 |  |  | 9 | 4 |
| Material behaviour | 2 |  | 1 | 4 |  |
| Units and magnitudes |  |  | 1 | 5 | 2 |

## Findings

None.

## Results by category

### Data integrity

| Check | Result | Worst case | Verdict | Basis |
|---|---|---|---|---|
| Declared traits against the solver input | 0 |  | pass | must be zero |
| Non-finite values (NaN, infinity) | 0 |  | pass | must be zero |
| Stored elements that no input part owns | 0 |  | pass | must be zero |
| Stored fields against the declared field list | 0 |  | pass | must be zero |
| Time steps that go backward | 0 |  | pass | must be zero |
| Frames written off the sampling interval | 1 |  | not judged | no criterion |
| Declared yield table against the input's | — |  | not checked | this instrument cannot measure it yet |
| Dips in the input's hardening table | — |  | not checked | this instrument cannot measure it yet |
| Energy ledger sampled on the field-output clock | — |  | not checked | the runs did not supply the global energy ledger and a sampling clock shared by the energy ledger and the fields |
| Plastic strain reached, relative to the table's range | — |  | not checked | this instrument cannot measure it yet |
| Stored global energies against the solver's ledger | — |  | not checked | the runs did not supply the global energy ledger and a sampling clock shared by the energy ledger and the fields |

### Numerical health of the runs

| Check | Result | Worst case | Verdict | Basis |
|---|---|---|---|---|
| Particles deactivated without erosion | 0 |  | pass | must be zero |
| Run reached its requested end time | 1.00026 to 1.00033 (+0.0259 % to +0.0335 %) | `NB-I-640-Sphere-b-160` | pass | at least 0.999999 (provisional tolerance) |
| Smoothing length within input bounds | 0 |  | pass | at most 1e-05 (provisional tolerance) |
| Solver errors | 0 |  | pass | must be zero |
| Solver terminated normally | yes |  | pass | must be yes |
| Solver version and precision recorded | 0 |  | pass | must be zero |
| Fewest neighbours of any particle | 3 to 8 | `NB-I-320-Bullet-a-160` | not judged | no criterion |
| Growth of the largest neighbour count | 1 to 1.125 | `NB-I-320-Bullet-a-160` | not judged | no criterion |
| Solver warnings | 1 |  | not judged | no criterion |
| Prescribed motion achieved | — |  | not checked | the runs did not supply applied loads and reaction forces over time |
| Smallest time step, relative to the first | — |  | not checked | the runs did not supply the time-integration record (time step, added mass) |
| Time step against a stability estimate | — |  | not checked | the runs did not supply the time-integration record (time step, added mass) |

Does not apply to these runs: deepest penetration of a rigid surface; hourglass energy vs initial energy; hourglass energy vs internal energy; hourglass energy, worst part; implicit increments accepted without convergence; mass added by mass scaling, moving parts; mass added by mass scaling, whole model; mass added by mass scaling, worst part; particles with smoothing length at a bound; peak hourglass vs peak internal energy.

### Energy and mass conservation

| Check | Result | Worst case | Verdict | Basis |
|---|---|---|---|---|
| Drift of the total mass | 0 % |  | pass | at most 0.0001 % (provisional tolerance) |
| Change in total energy, start to end | — |  | not checked | the runs did not supply the global energy ledger |
| Energy imbalance at the end of the run | — |  | not checked | the runs did not supply the global energy ledger |
| External work against the applied loads | — |  | not checked | the runs did not supply the global energy ledger and applied loads and reaction forces over time |
| Internal energy: stored fields vs solver ledger | — |  | not checked | the runs did not supply the global energy ledger and a sampling clock shared by the energy ledger and the fields |
| Kinetic energy: stored fields vs solver ledger | — |  | not checked | the runs did not supply the global energy ledger and a sampling clock shared by the energy ledger and the fields |
| Largest energy gain during the run | — |  | not checked | the runs did not supply the global energy ledger |
| Largest energy loss during the run | — |  | not checked | the runs did not supply the global energy ledger |
| Momentum change against applied impulse | — |  | not checked | the runs did not supply applied loads and reaction forces over time |
| Pressure against the equation of state | — |  | not checked | this instrument cannot measure it yet |

Does not apply to these runs: contact energy against internal energy; kinetic energy in a quasi-static run; mass bookkeeping with scaling or deletion; negative contact energy.

### Material behaviour

| Check | Result | Worst case | Verdict | Basis |
|---|---|---|---|---|
| Out-of-plane shear in a two-dimensional run | 0 Pa |  | pass | must be zero (provisional tolerance) |
| Out-of-plane strain in a plane-strain run | 0 |  | pass | at most 1e-06 (provisional tolerance) |
| Stored pressure against the stress trace | < 0.001 % | `NB-I-320-Bullet-a-80` | not judged | no criterion |
| Largest stress relative to the yield surface | — |  | not checked | this instrument cannot measure it yet |
| Plastic state never decreases | — |  | not checked | this instrument cannot measure it yet |
| Plastic state never negative | — |  | not checked | this instrument cannot measure it yet |
| Yielding material sits on the yield surface | — |  | not checked | this instrument cannot measure it yet |

### Units and magnitudes

| Check | Result | Worst case | Verdict | Basis |
|---|---|---|---|---|
| Largest speed in the response | 41.32 m/s to 184.5 m/s | `NB-I-320-Sphere-a-160` | not judged | published level shown below, for context |
| Declared unit anchors agree with the input | — |  | not checked | the benchmark has nowhere to declare it yet |
| Largest first-yield strain in the input | — |  | not checked | this instrument cannot measure it yet |
| Most extreme input density | — |  | not checked | the instrument could not read what the run supplied |
| Stored density against the input's density | — |  | not checked | the instrument could not read what the run supplied |
| The input's own unit declaration | — |  | not checked | this instrument cannot measure it yet |

Does not apply to these runs: most extreme young's modulus in the input; most extreme yield stress in the input.

## Measured, not judged

What the numbers above would mean, for the quantities no bound is applied to. Where a published limit exists it is shown for context only: it has not been confirmed against its source by the maintainer, so it is not this platform's standard.

- **Fewest neighbours of any particle** — a problem here would mean that particles ran short of neighbours (kernel starvation). *No criterion.*
- **Frames written off the sampling interval** — a problem here would mean that frames were written off the sampling interval - a fact, not a defect: a solver writes its state at the termination time and loaders drop it (ADR-0028). *No criterion.*
- **Growth of the largest neighbour count** — a problem here would mean that particles clumped (neighbour counts grew). *No criterion.*
- **Largest speed in the response** — a problem here would mean that a response speed is outside the structural-impact regime. *A published level exists, not confirmed by this platform: at most 3000 m/s [M-S9].*
- **Solver warnings** — a problem here would mean that the solver reported warnings that may affect the result. *No criterion.*
- **Stored pressure against the stress trace** — a problem here would mean that stored pressure disagrees with the stress trace. *No criterion.*

## Spread across cases

The unjudged quantities that differ between cases. The median is a measured value, never an interpolated one, and the worst case is the end of the range that matters for the quantity.

| Quantity | Lowest | Median | Highest | Worst case |
|---|---|---|---|---|
| Fewest neighbours of any particle | 3 | 8 | 8 | `NB-I-320-Bullet-a-160` |
| Growth of the largest neighbour count | 1 | 1 | 1.125 | `NB-I-320-Bullet-a-160` |
| Largest speed in the response | 41.32 m/s | 91.73 m/s | 184.5 m/s | `NB-I-320-Sphere-a-160` |
| Stored pressure against the stress trace | < 0.001 % | < 0.001 % | < 0.001 % | `NB-I-320-Bullet-a-80` |

## How to read this report

- **pass / fail** — the value was compared with a bound that is definitional (zero, equality, the input's own value) or computed from storage precision. None was fitted to this dataset.
- **not judged** — the value is reported, and no bound is applied. Sourced limits from the literature are shown for context until the maintainer has confirmed them against their sources; none has been.
- **not checked** — the check needs evidence the runs did not keep, or the instrument cannot make it yet. The reason is given.
- **not applicable** — the quantity does not exist for this kind of run (hourglass energy in a particle model, for instance).

Every check names the artefact a finding would condemn, and so who would have to act:

- **the stored response** — the fields a user loads and trains on.
- **the solver input** — the input deck; the fix is a corrected deck and a new run.
- **the run record** — how the run behaved and what the solver reported; the stored arrays may be intact.
- **the benchmark's declaration** — what this repository claims about the runs; the fix is a corrected benchmark card.

Bounds applied:

- *Declared traits against the solver input* — must be zero. What the benchmark declares about the run agrees with the solver input.
- *Drift of the total mass* — at most 0.0001 % (provisional). With mass scaling and deletion off, every stored mass is one constant rounded the same way each frame; 1e-6 is a decade above float32 resolution.
- *Non-finite values (NaN, infinity)* — must be zero. A stored response contains no NaN or infinity.
- *Out-of-plane shear in a two-dimensional run* — must be zero (provisional). A two-dimensional formulation carries no out-of-plane shear; the slots hold exact zeros, so any other value is a slot mix-up.
- *Out-of-plane strain in a plane-strain run* — at most 1e-06 (provisional). Plane strain sets the out-of-plane normal strain to zero; 1e-6 is a decade above float32 resolution of a strain of order one, and three decades below the smallest hoop strain of an axisymmetric run.
- *Particles deactivated without erosion* — must be zero. With erosion off, no particle is ever deactivated.
- *Run reached its requested end time* — at least 0.999999 (provisional). The last stored time reaches the requested end time. float32 time stamps resolve 1.2e-7 of their value; 1e-6 leaves a decade. No upper bound: an explicit run may overshoot by one step.
- *Smoothing length within input bounds* — at most 1e-05 (provisional). The scale is a ratio of two float32 values, resolved to about 2.4e-7; 1e-5 leaves over a decade.
- *Solver errors* — must be zero. The solver's record mentions no error.
- *Solver terminated normally* — must be yes. Every phase and restart segment ends with the solver's normal-termination statement.
- *Solver version and precision recorded* — must be zero. The record names the solver's version, revision, precision and parallel layout; none is missing.
- *Stored elements that no input part owns* — must be zero. Every stored element belongs to a part the solver input defines.
- *Stored fields against the declared field list* — must be zero. The stored fields are exactly the fields the benchmark declares.
- *Time steps that go backward* — must be zero. Stored times strictly increase.

Published levels shown for context (not applied). The bracketed ids are claims in the source dossier, `docs/plans/2026-09-21-reference-data-verification-sources.md`:

- *Largest speed in the response* — at most 3000 m/s [M-S9]. Above 3 km/s is the hypervelocity regime, outside structural impact. Blind to the mass unit.
