"""The committed reference-sensitivity record (ADR-0076) is checkable without
the archive: its sensitivities recompute from its stored quantities, its
page re-renders byte for byte, and it covers every case of the sweep with
the family's declared noise."""

import json
from pathlib import Path

import pytest

from structbench.benchmarks import available_benchmarks, get_benchmark
from structbench.eval.sensitivity import (
    render_reference_markdown,
    sensitivities_from_record,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORDS = REPO_ROOT / "docs" / "gradients"

_WITH_FAMILY = [n for n in available_benchmarks() if get_benchmark(n).gradient_family]


def test_taylor_declares_the_first_instance():
    assert "taylor_impact_2d" in _WITH_FAMILY


@pytest.mark.parametrize("name", _WITH_FAMILY)
def test_every_declared_family_has_a_committed_record(name):
    assert (RECORDS / f"{name}.json").is_file(), name
    assert (RECORDS / f"{name}.md").is_file(), name


@pytest.mark.parametrize("name", _WITH_FAMILY)
def test_record_covers_the_sweep_with_the_declared_noise(name):
    spec = get_benchmark(name)
    fam = spec.gradient_family
    record = json.loads((RECORDS / f"{name}.json").read_text(encoding="utf-8"))
    assert record["benchmark"] == name
    assert record["parameter"] == fam.parameter
    assert record["deltas"] == list(fam.deltas)
    cases = sorted(c for body in record["lines"].values() for c in body["cases"])
    assert cases == sorted(c for s in spec.splits.values() for c in s)
    for qoi, noise in fam.noise.items():
        assert record["noise"][qoi]["absolute"] == noise.absolute
        assert record["noise"][qoi]["relative"] == noise.relative
        assert record["noise"][qoi]["basis"] == noise.basis


@pytest.mark.parametrize("name", _WITH_FAMILY)
def test_record_sensitivities_recompute_and_page_rerenders(name):
    record = json.loads((RECORDS / f"{name}.json").read_text(encoding="utf-8"))
    again = sensitivities_from_record(record)
    for line, body in record["lines"].items():
        for qoi, stored in body["sensitivities"].items():
            for s, p in zip(stored, again[line][qoi], strict=True):
                assert s["value"] == p.value
                assert s["sensitivity"] == pytest.approx(p.sensitivity, rel=1e-12)
                assert s["truncation_error"] == pytest.approx(
                    p.truncation_error, rel=1e-12, abs=1e-15
                )
                assert s["measurement_error"] == pytest.approx(
                    p.measurement_error, rel=1e-12, abs=1e-15
                )
                assert s["judged"] == p.judged
    page = (RECORDS / f"{name}.md").read_text(encoding="utf-8")
    assert page == render_reference_markdown(record)
