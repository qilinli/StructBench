"""Every registered benchmark declares its V&V identity and covers the data
standard (ADR-0065 follow-up 2, built by the plan of 2026-11-08).

The dataclass leaves the fields optional so a fixture or a definition loaded
by path is not forced to declare what it has not assessed; a *registered*
benchmark is, here. The statuses pinned below are the honest ones the plan
fixed: Taylor alone has a convergence run and a measured noise floor;
DeformingPlate can never carry a blind set (no generator, ADR-0074); the
LS-DYNA benchmarks get theirs when solver access allows.
"""

import pytest

from structbench.benchmarks import available_benchmarks, get_benchmark
from structbench.benchmarks.card import REQUIRED_ITEMS, STANDARD_ITEMS


def _card(name):
    return get_benchmark(name).card


def _statuses(name):
    return {r.item: r.status for r in _card(name).compliance}


@pytest.mark.parametrize("name", available_benchmarks())
def test_every_registered_card_covers_the_standard(name):
    card = _card(name)
    assert card.compliance_covers_standard(), (
        f"{name}: compliance rows {sorted(r.item for r in card.compliance)} "
        f"do not cover {sorted(STANDARD_ITEMS)}"
    )


@pytest.mark.parametrize("name", available_benchmarks())
def test_every_registered_card_declares_its_vv_identity(name):
    card = _card(name)
    assert card.structural_scale, f"{name}: no structural_scale"
    assert card.load_type.strip(), f"{name}: no load_type"
    assert card.autonomy.strip(), f"{name}: no autonomy"
    assert card.parameter_space is not None, f"{name}: no parameter_space"
    assert card.ground_truth is not None, f"{name}: no ground_truth"


@pytest.mark.parametrize("name", available_benchmarks())
def test_shipped_benchmarks_are_grandfathered_not_compliant(name):
    # ADR-0065: none of the four would be admitted under the standard today;
    # the table makes that visible rather than hiding it.
    statuses = _statuses(name)
    assert any(statuses[item] != "pass" for item in REQUIRED_ITEMS), name


@pytest.mark.parametrize("name", available_benchmarks())
def test_taylor_alone_has_a_convergence_run_and_a_noise_floor(name):
    statuses = _statuses(name)
    expected = "pass" if name == "taylor_impact_2d" else "missing"
    assert statuses["convergence_run"] == expected
    assert statuses["noise_floor"] == expected


@pytest.mark.parametrize("name", available_benchmarks())
def test_blind_rows_follow_adr_0074(name):
    status = _statuses(name)["blind_evaluation"]
    if name == "deforming_plate":
        assert status == "never"
    elif _card(name).solver == "LS-DYNA":
        assert status == "missing"
