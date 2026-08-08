"""CaseBoundSimulator base contract (extracted from MeshSimulator; ADR-0044)."""

import pytest

from structbench.models.common import CaseBoundSimulator
from structbench.models.mgn import MeshSimulator


def test_mesh_simulator_is_case_bound() -> None:
    assert issubclass(MeshSimulator, CaseBoundSimulator)


def test_scripted_subset_validation() -> None:
    with pytest.raises(ValueError, match="scripted_types"):
        CaseBoundSimulator(
            dim=3, node_type_size=9, kinematic_types=(3,), scripted_types=(1,)
        )


def test_reset_rollout_clears_pointer() -> None:
    sim = CaseBoundSimulator(
        dim=3, node_type_size=9, kinematic_types=(1, 3), scripted_types=(1,)
    )
    sim._t = 7
    sim.reset_rollout()
    assert sim._t is None
