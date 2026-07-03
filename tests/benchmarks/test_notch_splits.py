"""ADR-0026 notch-beam splits: frozen 88/8/12, interior holdout, probes."""

import pytest

from structbench.benchmarks import get_benchmark


@pytest.mark.parametrize(
    ("name", "interior", "n_probes"),
    [
        ("notch_beam_2d_bend", {"12", "16"}, 3),
        ("notch_beam_2d_impact", {"80", "120"}, 2),
    ],
)
def test_frozen_split_honours_adr_0026(name, interior, n_probes):
    spec = get_benchmark(name)
    train = spec.splits["train"]
    val, test = spec.splits["val"], spec.splits["test_interp"]
    assert (len(train), len(val), len(test)) == (88, 8, 12)
    all_ids = set(train) | set(val) | set(test)
    assert len(all_ids) == 108
    for case in list(val) + list(test):
        assert case.rsplit("-", 1)[1] in interior
    train_tokens = {tok for c in train for tok in c.split("-")}
    for case in all_ids:
        assert set(case.split("-")) <= train_tokens  # every factor level in train
    assert len(spec.splits["probe"]) == n_probes
    assert spec.eval_splits == ("val", "test_interp", "probe")
    assert spec.aux_field == "damage"
    assert spec.kinematic_types  # non-empty: pin + support prescribed
