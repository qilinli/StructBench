"""Card-vs-data validation (ADR-0027) — runs only when data is present.

Each benchmark has its own env var, pointing at its canonical archive
directory (ADR-0031 layout: <data>/StructBench/canonical/<benchmark>/):
  STRUCTBENCH_DATA_ROOT              — taylor_impact_2d
  STRUCTBENCH_WAVE1D_DATA_ROOT       — wave_propagation_1d
  STRUCTBENCH_NOTCH_IMPACT_DATA_ROOT — notch_beam_2d_impact
  STRUCTBENCH_DEFORMING_PLATE_DATA_ROOT — deforming_plate

Each parametrized case skips independently when its var is unset.
"""

import os
from pathlib import Path

import pytest

from structbench.benchmarks import get_benchmark
from structbench.core.io import read_case

_BENCHMARK_ROOTS = {
    "notch_beam_2d_impact": os.environ.get("STRUCTBENCH_NOTCH_IMPACT_DATA_ROOT"),
    "taylor_impact_2d": os.environ.get("STRUCTBENCH_DATA_ROOT"),
    "wave_propagation_1d": os.environ.get("STRUCTBENCH_WAVE1D_DATA_ROOT"),
    "deforming_plate": os.environ.get("STRUCTBENCH_DEFORMING_PLATE_DATA_ROOT"),
}


@pytest.mark.parametrize("name", sorted(_BENCHMARK_ROOTS))
def test_card_matches_one_canonical_case(name: str) -> None:
    root = _BENCHMARK_ROOTS[name]
    if root is None:
        pytest.skip(f"data root env var for {name} not set")
    spec = get_benchmark(name)
    case = read_case(Path(root) / f"{spec.splits['train'][0]}.h5")
    lo, hi = (int(x) for x in spec.card.particles_per_case.split("-"))
    # Particles: SPH benchmarks count sph elements; mesh benchmarks count nodes
    # (nodes-as-particles, ADR-0043).
    if "sph" in case.elements:
        n_particles = case.elements["sph"].element_id.shape[0]
    else:
        n_particles = case.nodes.coords.shape[0]
    assert lo <= n_particles <= hi
    assert case.response is not None
    assert case.response.time.shape[0] == spec.card.n_frames
    available = (
        {f"node/{k}" for k in case.response.node}
        | {
            f"{etype}/{k}"
            for etype, fields in case.response.element.items()
            for k in fields
        }
        | {f"global/{k}" for k in case.response.globals_}
    )
    assert set(spec.card.fields) == available, (
        f"card fields != canonical data fields; "
        f"missing_from_data={set(spec.card.fields) - available}, "
        f"undeclared={available - set(spec.card.fields)}"
    )
