# 0080 — Force is read from the predicted state

**Status**: Proposed
**Type**: Durable
**Date**: 2026-11-27
**Amends**: 0075 (clause 6 gains two properties; the element site of clause 1 is what a quantity of interest may now read), 0032 (what a benchmark's quantity of interest is given)

**Your call**:
1. A force is a derived quantity: the predicted stress integrated on the predicted mesh by the finite-element operation, never a model output and never a second difference of predicted positions. Reversing means re-registering every force quantity a benchmark has declared.
2. The reference force is always computed from the stored integration-point stress. A model that predicts stress only at the nodes is scored on its stress averaged to the elements, and the record says so. Reversing re-scores the force quantities of every nodal result.
3. The force balance of a quasi-static benchmark and the penetration of a declared rigid body join the properties block, reported and never gating; a loss on force, a body advanced from force and a stress field in equilibrium by construction stay out of the platform. Reversing the first re-renders the registries; the second is a scope change.

## Context

Peak force, mean force and impulse are what a structural engineer reads from an impact or crush simulation, and no benchmark could register one. A quantity of interest was a function of positions and one auxiliary channel (ADR-0032), so the only force available was mass times the second difference of a predicted position, which amplifies position error without bound. A simulator that predicts positions and stress has already predicted every force: the nodal internal force is the stress integrated against the shape-function gradients, the operation a solver performs each increment. ADR-0075 made the integration-point state a scored site; this record says what is read from it.

## Decision

1. **The operator.** `structbench.eval.force.internal_force` maps positions, element stress and connectivity to nodal forces on the deformed mesh, with one implementation per element formulation a benchmark can declare (ADR-0075's `element_formulation`). An element type without an implementation is refused by name; none is approximated by another. Each implementation is tested against closed forms.
2. **What a quantity is given.** Beside positions, the auxiliary channel and the case id, a quantity of interest is given the element stress, the connectivity, the declared formulation and the definition's named node sets. A benchmark without an element site gives none of these, and its quantities are unchanged.
3. **Which stress.** The reference side is the stored integration-point stress. The predicted side is the model's own element state where it carries one (ADR-0075 clause 3), and otherwise its nodal stress averaged over each element's nodes. Every metrics record names which (`qoi_stress_site`). A result that carries no stress reports such a quantity as undefined, never as a number.
4. **Force quantities.** The platform provides the reaction at a named node set, the force a declared rigid body applies through the nodes on it, and the force through a section. Each is a series over the frames, stored with the rollout for the prediction and the reference, so that the scalar a benchmark registers (peak, mean, impulse) is a declared reduction of a stored series.
5. **Two properties.** Clause 6 of ADR-0075 gains: the normalised out-of-balance force at the unloaded nodes, for a benchmark whose definition declares the problem quasi-static; and the largest depth by which a node passes a rigid body the definition declares. Both follow that clause's rules: one kernel, reported where defined, no gate. A dynamic benchmark reports no balance until a result carries a predicted acceleration.
6. **Before a force is registered** on a benchmark, its landing page records the check that the operator, applied to the stored stress, reproduces a second reference from the same data: a stored reaction or contact history, or a rigid body's mass times its stored acceleration. What a dataset must store for this is in `docs/DATA_GENERATION.md`.
7. **Outside the platform:** a training loss on a derived force, a rigid body whose motion is integrated from the force on it, and a stress field constrained to equilibrium. Each is a method, and a study's to propose.

## Alternatives considered

- **Force as mass times the second difference of position.** Available without any stress, and wrong by the full value where the position error is a few percent.
- **A force output head, scored against a stored reaction history.** It adds a prediction that need not agree with the model's own stress, and it needs the history in every dataset.
- **Score every model at the nodes.** It would discard the site ADR-0075 made authoritative and smooth the reference.
- **Leave force to each study.** Then no two results compute it the same way and none can be registered.

## Consequences

- A benchmark can register a force without any change to a model.
- A result that predicts no stress tensor cannot report a force. This is deliberate: the quantity tests the state.
- Each new element formulation or element type needs its own operator and closed-form tests before a benchmark on it can register a force.
- Built on `feat/force-from-state`; no shipped benchmark declares a force quantity, a node set or either property, so no registered number moves.
- Not built: the dynamic balance, and element types other than the four-node quadrilateral with one integration point.
