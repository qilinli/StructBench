# `taylor_copper` — copper Taylor impact tests

*A reference-experiment set for validation against experiments (ADR-0072),
format `validation-reference/1`. The numbers live in `taylor_copper.json`;
this page is their provenance.*

## What it is

Six Taylor impact tests of copper cylinders on a rigid anvil, as compiled in
Table 1 of Zelepugin, Cherepanov & Pakhnutova (2023): the initial length,
diameter, impact speed and temperature of each specimen, and the measured
outline of the recovered specimen (its half profile, r against z from the
impact face). From each outline the shared measures give the final length
L_f, the largest radius R_f and the lateral radius W_f at fixed fractions of
L_f. A simulation run at a test's conditions is measured by the same
functions on its own deformed boundary, and the comparison reports the
deviations; no acceptance level is attached to them.

## Sources

| Id | Citation | DOI | Consulted directly | Licence | Role |
|---|---|---|---|---|---|
| S1 | Zelepugin S.A., Cherepanov R.O., Pakhnutova N.V. *Optimization of Johnson–Cook constitutive model parameters using the Nesterov gradient-descent method.* Materials 2023, 16, 5452. | [10.3390/ma16155452](https://doi.org/10.3390/ma16155452) | yes | CC BY 4.0 | test conditions (Table 1); measured outlines (the black curves of Figures 4 and 5) |
| S2 | Zelepugin S.A., Pakhnutova N.V., Shkoda O.A., Boyangin E.N. *Experimental study of the microhardness and microstructure of a copper specimen using the Taylor impact test.* Metals 2022, 12, 2186. | [10.3390/met12122186](https://doi.org/10.3390/met12122186) | no | — | the original source of tests 3–6, as S1 reports them |
| S3 | Wilkins M.L., Guinan M.W. *Impact of cylinders on a rigid boundary.* J. Appl. Phys. 1973, 44, 1200–1206. | [10.1063/1.1662328](https://doi.org/10.1063/1.1662328) | no | — | the original source of test 1, as S1 reports it |
| S4 | Gust W.H. *High impact deformation of metal cylinders at elevated temperatures.* J. Appl. Phys. 1982, 53, 3566–3575. | [10.1063/1.331136](https://doi.org/10.1063/1.331136) | no | — | the original source of test 2, as S1 reports it |

The outlines are redistributed from S1's figures under its CC BY 4.0
licence, with this attribution. The conditions are S1's Table 1. S1's own
simulated curves (the red curves of the same figures) are not part of this
set: they are a simulation, not a measurement. The PDF is not in the
repository.

## The tests

| Id | Material | L₀ (mm) | D₀ (mm) | v₀ (m/s) | T₀ (K) | Original source | L_f/L₀ | R_f/R₀ |
|---|---|---|---|---|---|---|---|---|
| 1 | OFHC Cu | 23.47 | 7.62 | 210 | 298 | S3 | 0.630 | 2.204 |
| 2 | ETP Cu | 30.0 | 6.0 | 188 | 718 | S4 | 0.553 | 1.865 |
| 3 | OFHC Cu M1 | 34.5 | 7.8 | 162 | 298 | S2 | 0.763 | 1.701 |
| 4 | OFHC Cu M1 | 34.5 | 7.8 | 167 | 298 | S2 | 0.743 | 1.772 |
| 5 | OFHC Cu M1 | 34.5 | 7.8 | 225 | 298 | S2 | 0.645 | 2.056 |
| 6 | OFHC Cu M1 | 34.5 | 7.8 | 316 | 298 | S2 | 0.464 | 2.724 |

The ratios are rounded here; the JSON holds L_f, R_f and W_f in millimetres
to the precision of the extraction.

## How the outlines were extracted

S1's Figures 4 and 5 are vector drawings, one panel per test, captioned
"calculated (red) and experimental (black) profiles of the external surfaces
of the cylinders". `tools/validation/digitize_zelepugin2023.py` reads the
black polylines directly from the PDF's drawing commands — nothing is read by
eye — and maps them to centimetres by a least-squares fit to the major tick
marks and their labels. Checks: the worst tick-map residual is 6.8 × 10⁻⁴ cm
(the tool refuses above 0.005 cm), and every test's experimental curve is
drawn in both figures and the two agree to 0.001 cm (test 6's Figure 5 copy
has one extra vertex, with the same top and the same largest radius; the
Figure 4 copy is kept). The outline is converted to millimetres, shifted so
that its lowest point is z = 0, and rounded to 10⁻⁴ mm.

To regenerate: `python tools/validation/digitize_zelepugin2023.py <S1.pdf>
src/structbench/validation/references/taylor_copper.json`. The test suite
holds the stored L_f, R_f and W_f to what the measure functions return on the
stored outline.

## Measures

- **L_f** — final length: the outline's extent along the axis.
- **R_f** — the largest radius anywhere on the outline.
- **W_f** — the lateral radius: the largest r where the outline crosses the
  height f·L_f, at f = 0.2, 0.25, 1/3, 1/2 and 2/3 (S1's Figure 2).

## Caveats

- The measured values reach this set through S1's figures; S1 does not
  tabulate them, and S2, S3 and S4 were not consulted directly.
- Test 1's measured top edge is drawn with a slight slope; L_f is its highest
  point.
- Test 2 is ETP copper at 718 K; a room-temperature comparison leaves it out.
- Test 6's Figure 5 copy has one extra vertex, with the same top and the same
  largest radius as its Figure 4 copy.
