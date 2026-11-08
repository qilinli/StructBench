# 0076 — The gradient task family: a surrogate's sensitivities scored against the sweep's own differences

**Status**: Accepted (maintainer, in-session 2026-11-08)
**Type**: Durable
**Date**: 2026-11-08
**Amends**: 0019 (a task family is added; the task stands), 0032 and 0033 (a metric group), 0055 (the relative form extends to sensitivities), 0075 (a second task family); constrains 0074 (gradient rows are read by coverage)

**Your call**:
1. A surrogate's *sensitivity* — the derivative of each quantity of interest with respect to a design parameter — is a registered task family, scored on the shipped sweep against reference sensitivities taken by central differences across the sweep's own grid, each with its declared error. No new dataset. Reversing withdraws the gradient rows and nothing else.
2. The surrogate's sensitivity is measured family-agnostically, by finite differences of its own predictions from seeds the protocol constructs at perturbed parameter values. Reversing makes the family a model-interface change every family must implement.
3. The first instance is Taylor's impact velocity alone; a reference point whose error exceeds half its sensitivity is reported and not judged. Reversing is cheap: a note names the next parameter or benchmark.

## Context

Every shipped benchmark scores the predicted response. Nothing scores how the response *changes* with a design parameter, which is what an optimiser consumes and what ADR-0065 names among the properties that make a surrogate fit for a decision. Accuracy does not stand in for it: ADR-0065's note records the accuracy leader becoming the worst model beyond coverage, where an optimiser walks. A model can match every trajectory and have its gradients point the wrong way, undetectably.

The data to test this exists. Taylor's sweep is a line of eleven impact velocities at 10 m/s steps per bar length (ADR-0019), so a central difference across neighbouring cases is a reference sensitivity at every interior velocity, with its error measurable at a doubled step. The seed frames are free flight (timeline analysis, 2026-07-05), so a seed at any velocity is the initial lattice translated, and a model can be queried off the grid without a solver. The maintainer chose this over a generated dataset on 2026-11-08: it asks first whether any shipped model yields a gradient worth trusting.

## Decision

1. **The family.** A gradient task declares a benchmark, a scalar parameter the sweep varies, the quantities of interest scored and the parameter values scored. It adds rows to the results registry and changes no task, split or protocol of its host benchmark, so it never bumps a version (ADR-0032 §4).
2. **Reference sensitivities.** At each grid value, the central difference of the reference quantity across the neighbouring cases, one-sided at the ends. Its truncation error is the same difference at twice the step; the quantity's discretisation noise, the floor the card declares, is a second term. Both are stored with the value; nothing is fitted.
3. **The surrogate's sensitivity.** The protocol constructs seeds at the scored value and at ±δ, rolls the model out under the benchmark's protocol, computes the quantities of interest and takes the central difference; two δ are run and their disagreement is the model's own floor. A model with a differentiable parameter path may report an autodiff value beside it; the finite-difference value is the registered one.
4. **Scoring.** Per quantity, the pooled relative L2 of the sensitivity over the scored values (the ADR-0055 form), keyed `gradient_rel_l2_<qoi>`, with the fraction of sign agreements beside it. A value whose reference error exceeds half the reference sensitivity renders with its numbers and leaves the pooled headline, marked so (the measure/judge separation of ADR-0066). Rows are read by coverage (ADR-0074); the sweep's ends are shown apart.
5. **Scored values.** Every value on the line is scored and reported by set; the headline pools the scored sets. References use every split's data: the family tests the model's derivative, not its memory.
6. **First instance.** Taylor 2D, impact velocity, the four quantities of ADR-0019. Bar length is excluded: three levels at 20 mm and a changing particle count. The other sweeps are too coarse (notch-impact 40 m/s steps, wave doubling steps); an instance there takes a dated note.
7. **Where it lives.** `structbench.eval`, a declaration on `BenchmarkSpec`, a `gradient` mode of `structbench-train`, a metric tier on the registries and pages under ADR-0075's rendering plan. Design: `docs/plans/2026-11-08-gradient-task-family-design.md`.

## Alternatives considered

- **A generated dataset with paired perturbation runs or solver sensitivities.** Clean gradients in any parameter; a dataset project. The route if this instance shows the question is worth it.
- **Autodiff as the required measurement.** A model-interface change for every family, meaningless where the parameter enters only through the seed. Kept optional.
- **Scoring the gradient field.** Dominated by von Mises discretisation noise; an optimiser reads the quantities of interest.

## Consequences

- No data is generated or published. The family is a protocol, a metric group and registry rows.
- Reference sensitivities and their errors are computed once from the archive and committed as a generated record, so a reader sees the reference's uncertainty before any model's number.
- Legacy rows carry no gradient metrics until re-evaluated; a row with them is read beside its accuracy and properties block (ADR-0075), never ranked (ADR-0074).
- The quantity-of-interest functions gain a contract: smooth enough in the parameter for a difference to mean anything. Where one is not (a time of peak read at the output interval), the declared error says so.
