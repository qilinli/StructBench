"""ADR-0043 deforming_plate split and protocol pins."""

from structbench.benchmarks import get_benchmark


def test_split_sizes_and_ids():
    spec = get_benchmark("deforming_plate")
    assert len(spec.splits["train"]) == 1000
    assert len(spec.splits["val"]) == 100
    assert len(spec.splits["test"]) == 100
    assert spec.splits["train"][0] == "train_0000"
    assert spec.splits["train"][-1] == "train_0999"
    assert spec.splits["val"][0] == "val_0000"
    assert spec.splits["test"][-1] == "test_0099"
    all_ids = [i for s in ("train", "val", "test") for i in spec.splits[s]]
    assert len(set(all_ids)) == 1200  # disjoint


def test_protocol_pins():
    spec = get_benchmark("deforming_plate")
    assert spec.card.input_frames == 2
    assert spec.kinematic_types == (1, 3)
    assert spec.scored_frames is None
    assert spec.card.horizon == "full"
    assert spec.aux_field == "von_mises_stress"
    assert set(spec.qois) == {"peak_vm_stress", "terminal_peak_deflection"}
    assert spec.eval_splits == ("val", "test")
    # Four baselines (ADR-0043/0044/0045/0057): blessed MGN (autoregressive) +
    # provisional Transolver and Transolver++ (time-conditioned, ADR-0054/0057)
    # and GeoFLARE (autoregressive), scored on the single `test` split. The
    # Transolver row switched from the autoregressive 2M-step run to the
    # time-conditioned 250k run (decisively better; the faithful native scheme).
    # Scheme matrix (2026-08-21): one row per (family, scheme); blessed MGN first.
    assert tuple((r.family, r.scheme) for r in spec.results) == (
        ("mgn", "autoregressive"),
        ("transolver", "autoregressive"),
        ("transolver", "time-conditioned"),
        ("transolver_plus", "time-conditioned"),
        ("geoflare", "autoregressive"),
        ("geoflare", "time-conditioned"),
    )
    assert [r.provisional for r in spec.results] == [False] + [True] * 5
    assert all(set(r.metrics) == {"test"} for r in spec.results)
    assert spec.quickstart_family == "mgn"
