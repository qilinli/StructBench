"""The instrument must not blame a dataset for its own particle assumption.

Every measure that reads the SPH block indexes ``case.elements["sph"]``
directly. On a mesh-only case that is a ``KeyError``, which ``measure_case``
catches and stamps ``source_unreadable`` — a contributor-owned reason — for a
case that is perfectly readable. Nine such rows carry no particle trait gate,
so they only escape today because ``facts=None`` gates them out first; an
Abaqus reader supplying input facts would reach all of them.
"""

from __future__ import annotations

import numpy as np

from structbench.core import (
    AbsenceReason,
    Case,
    ElementBlock,
    InputFacts,
    MaterialInput,
    Metadata,
    Nodes,
    PartTraits,
    Response,
    RunEvidence,
    SolverIdentity,
    TerminationRecord,
)
from structbench.verification.measures import measure_case


def _mesh_case() -> Case:
    """A solid-element case with no ``sph`` block at all."""
    n = 8
    return Case(
        metadata=Metadata(case_id="m", dimension=3, source_units="kg-mm-ms"),
        nodes=Nodes(
            coords=np.zeros((n, 3)), node_id=np.arange(1, n + 1, dtype=np.int64)
        ),
        elements={
            "solid": ElementBlock(
                connectivity=np.arange(n, dtype=np.int64).reshape(1, 8),
                element_id=np.array([1], dtype=np.int64),
                part_id=np.array([1], dtype=np.int64),
            )
        },
        materials=[],
        response=Response(
            time=np.array([0.0, 1.0e-6, 2.0e-6]),
            node={"displacement": np.zeros((3, n, 3), dtype=np.float32)},
        ),
    )


def _mesh_facts() -> InputFacts:
    """Input facts for that case: one solid part, one supported material."""
    return InputFacts(
        parts=(PartTraits(1, 2, "solid", False),),
        materials=(
            MaterialInput(2, "elastic_plastic_hydro", 7850.0, 8.0e10, None, None, None),
        ),
        time_integration="implicit",
        dimension=3,
        plane_strain=False,
        end_time=1.0,
        other_termination_criteria=frozenset(),
        mass_scaling_enabled=False,
        erosion_enabled=False,
        contact_defined=False,
        prescribed_motion_defined=False,
        damping_defined=False,
        rigid_planes=(),
        particle_pairwise_conservative=None,
        smoothing_length_scale_bounds=None,
        energy_terms_computed=None,
        databases_requested=None,
        unparsable=frozenset(),
        solver="lsdyna",
    )


def _unreadable(result) -> list[str]:
    return [
        m.quantity
        for m in result.measurements
        if m.absence is not None and m.absence.reason is AbsenceReason.SOURCE_UNREADABLE
    ]


def test_a_mesh_only_case_is_never_called_unreadable() -> None:
    """The regression that makes this a class fix rather than an instance one."""
    result = measure_case(_mesh_case(), None, None, case_id="m")
    assert _unreadable(result) == []


def test_a_mesh_case_with_input_facts_is_never_called_unreadable() -> None:
    """With facts supplied, the trait gates stop shielding the ungated rows."""
    result = measure_case(_mesh_case(), _mesh_facts(), None, case_id="m")
    assert _unreadable(result) == []


def test_a_particle_only_row_on_a_mesh_case_is_the_platforms_gap() -> None:
    """`unsupported` is in PLATFORM_REASONS; `source_unreadable` is not."""
    result = measure_case(_mesh_case(), _mesh_facts(), None, case_id="m")
    row = next(
        m for m in result.measurements if m.quantity == "pressure_trace_residual"
    )
    assert row.absence is not None
    assert row.absence.reason is AbsenceReason.UNSUPPORTED


def test_a_run_that_printed_no_warning_count_is_not_an_unreadable_source() -> None:
    """The Abaqus failure shape: two fatal errors, no warning count anywhere.

    `_supplied` admits E3 on termination + n_errors, so the warning row is
    measured and then asserts on a field the record never carried. The record
    is fine; what is missing is one datum, and the row should say so.
    """
    run = RunEvidence(
        identity=SolverIdentity("abaqus", "2025", None, "double", None),
        termination=(TerminationRecord("error", None, None, None),),
        n_errors=2,
        n_warnings=None,
    )
    result = measure_case(_mesh_case(), None, None, case_id="m", run=run)
    warn = next(m for m in result.measurements if m.quantity == "solver_warning_count")
    errors = next(m for m in result.measurements if m.quantity == "solver_error_count")
    assert errors.value == 2.0  # still measured
    assert warn.absence is not None
    assert warn.absence.reason is AbsenceReason.SOURCE_MISSING
    assert _unreadable(result) == []
