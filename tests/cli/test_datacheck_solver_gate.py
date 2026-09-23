"""The measure path must not hand a foreign deck to the LS-DYNA parser.

Today `measure_dataset` calls `read_input_facts` — the LS-DYNA *keyword*
parser — on whatever `metadata.source_deck` holds. Given an Abaqus `.inp` it
does not merely fail: it reports `time_integration='explicit'` for a `*Static`
job and `databases_requested=frozenset()`, so the conformance row fails with
`bears_on="input"` and accuses a correct deck of omitting outputs.

The naive repair — pass `facts=None` — relocates the misreport rather than
closing it: `gate()` then returns `SOURCE_MISSING`, which is NOT in
`PLATFORM_REASONS`, so a contributor who supplied a complete parseable deck
is told their input is missing. The gate has to report a *platform* reason.
"""

from __future__ import annotations

import numpy as np

from structbench.cli.datacheck import input_facts_for
from structbench.core import (
    AbsenceReason,
    Case,
    ElementBlock,
    Metadata,
    Nodes,
    Provenance,
    Response,
)
from structbench.core.evidence import PLATFORM_REASONS

_LSDYNA_DECK = "\n".join(
    [
        "*KEYWORD",
        "*CONTROL_TERMINATION",
        "$#  endtim",
        "      0.25",
        "*END",
    ]
)
_ABAQUS_DECK = "\n".join(["*Heading", "** Job name: Cantilever", "*Static", "1., 1."])


def _case(deck: str | None, solver: str | None) -> Case:
    provenance = None if solver is None else Provenance(solver, "unknown", "2026-09-24")
    return Case(
        metadata=Metadata(
            case_id="c",
            dimension=3,
            source_units="g-mm-ms",
            source_deck=deck,
            provenance=provenance,
        ),
        nodes=Nodes(coords=np.zeros((2, 3)), node_id=np.array([1, 2], dtype=np.int64)),
        elements={
            "solid": ElementBlock(
                connectivity=np.array([[0, 1]], dtype=np.int64),
                element_id=np.array([1], dtype=np.int64),
                part_id=np.array([1], dtype=np.int64),
            )
        },
        materials=[],
        response=Response(time=np.array([0.0, 1.0]), node={}),
    )


def test_an_lsdyna_deck_is_still_read() -> None:
    facts, reason = input_facts_for(_case(_LSDYNA_DECK, "LS-DYNA"))
    assert reason is None
    assert facts is not None and facts.end_time is not None


def test_the_solver_name_is_matched_without_punctuation_or_case() -> None:
    for spelling in ("ls-dyna", "LSDYNA", "  LS-Dyna  "):
        facts, reason = input_facts_for(_case(_LSDYNA_DECK, spelling))
        assert reason is None, spelling
        assert facts is not None, spelling


def test_an_abaqus_deck_now_goes_to_the_abaqus_reader() -> None:
    """Registered since ADR-0068 step 7; it is never handed to LS-DYNA's."""
    facts, reason = input_facts_for(_case(_ABAQUS_DECK, "Abaqus"))
    assert reason is None
    assert facts is not None and facts.solver == "abaqus"
    assert facts.time_integration == "implicit"  # *Static, not LS-DYNA's default


def test_a_solver_with_no_reader_is_the_platforms_gap() -> None:
    """DeformingPlate already declares COMSOL, for which there is no reader."""
    facts, reason = input_facts_for(_case(_ABAQUS_DECK, "COMSOL"))
    assert facts is None  # not handed to whichever parser happens to be first
    assert reason is AbsenceReason.UNSUPPORTED
    assert reason in PLATFORM_REASONS  # so it never counts against the dataset


def test_an_unnamed_solver_fails_closed_rather_than_guessing() -> None:
    """Provenance is optional. An input we cannot attribute is not parsed."""
    facts, reason = input_facts_for(_case(_LSDYNA_DECK, None))
    assert facts is None
    assert reason is AbsenceReason.UNSUPPORTED


def test_a_case_that_stores_no_input_is_the_datasets_gap_not_the_platforms() -> None:
    """No deck is a different claim from a deck we cannot read."""
    facts, reason = input_facts_for(_case(None, "LS-DYNA"))
    assert facts is None
    assert reason is None  # falls through to SOURCE_MISSING / E1, as before


def test_deforming_plate_shape_is_unchanged() -> None:
    """It declares COMSOL and stores no deck; nothing should newly fire."""
    facts, reason = input_facts_for(_case(None, "COMSOL"))
    assert facts is None and reason is None
