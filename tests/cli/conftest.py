"""Shared checks for the training entry-point tests.

``scored_span_check`` is the ADR-0078 contract, run by each family's smoke
test on its own tiny benchmark with a per-case scored bound: one declared
span governs the training pool, the time reference and the evaluator.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest


def _short_bound(case_id: str) -> int:
    """A scored bound that differs per case and is shorter than the 8 stored
    frames for every case: TRAIN 5, 6, 7, 5; VAL 5, 6."""
    index = int(case_id.rsplit("_", 1)[1])
    return 5 + index % 3 if case_id.startswith("train") else 5 + index


@pytest.fixture
def short_bound():
    """The per-case scored bound of the ADR-0078 tests (a function of the
    case id), shorter than the 8 stored frames for every case."""
    return _short_bound


def _last_frame(sample: dict[str, Any]) -> int:
    """The largest frame index a training sample reads, inputs and targets."""
    if "anchor_frame" in sample:  # a flow-map pair or chain
        frames = [int(np.max(np.asarray(v))) for k, v in sample.items() if "frame" in k]
        return max(frames)
    target = sample["next_position"]
    span = target.shape[1] if target.ndim == 3 else 1
    return int(sample["target_frame"]) + span - 1


_NOT_SCORED = {"rollout_position_rmse_full", "unscored_tail"}


def _scored(block: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in block.items() if k not in _NOT_SCORED}


@pytest.fixture
def scored_span_check(tmp_path, monkeypatch):
    """Return ``check(run, timed=...)``.

    ``run(directory, training_span)`` trains one tiny run on a benchmark
    whose bound is the ``short_bound`` fixture and returns
    ``(spec, data_root, out_dir, ids)``. ``timed`` says whether the scheme
    normalizes a query time (time-conditioned, flow map).
    """
    import structbench.cli.train as cli_train

    calls: list[tuple[Any, ...]] = []
    original = cli_train._record_training_pool

    def spy(out_dir, spec, train_cfg, train_trajs, make_dataset, n_samples, **kw):
        calls.append((list(train_trajs), make_dataset, n_samples, kw.get("time_ref")))
        return original(
            out_dir, spec, train_cfg, train_trajs, make_dataset, n_samples, **kw
        )

    monkeypatch.setattr(cli_train, "_record_training_pool", spy)

    def check(run, *, timed: bool) -> dict[str, Any]:
        scored_dir = tmp_path / "scored"
        scored_dir.mkdir()
        spec, data_root, out, ids = run(scored_dir, "scored")
        stored = spec.card.n_frames

        # 1. The pool: every trajectory the loop draws from ends at its
        #    case's bound, and no sample reads a frame at or beyond it.
        ((trajs, make_dataset, n_samples, time_ref),) = calls
        for tr in trajs:
            assert len(tr.time) == spec.scored_frames_for(tr.case_id) < stored
        dataset = make_dataset(trajs)
        assert len(dataset) == n_samples > 0
        for i in range(len(dataset)):
            sample = dataset[i]
            bound = spec.scored_frames_for(trajs[sample["traj_idx"]].case_id)
            assert _last_frame(sample) < bound

        # 2. The record: span, pool size, scored share, passes, reference.
        record = json.loads((out / "config.json").read_text(encoding="utf-8"))
        pool = record["run"]["training_pool"]
        assert record["train"]["training_span"] == "scored"
        assert pool["training_span"] == "scored"
        assert pool["scored_frames"] == "per_case"
        assert pool["samples"] == n_samples
        assert pool["scored_share"] == 1.0
        steps, batch = record["train"]["training_steps"], record["train"]["batch_size"]
        assert pool["passes"] == pytest.approx(steps * batch / n_samples)
        if timed:
            # One clock per benchmark: the card's stored length, not the
            # shortest cut trajectory (5 here).
            assert time_ref == stored
            assert record["run"]["time_ref_frames"] == stored
        else:
            assert time_ref is None
            assert "time_ref_frames" not in record["run"]

        # 3. The evaluator stops at the bound, and its scored numbers do not
        #    change when it rolls on.
        monkeypatch.setattr(cli_train, "get_benchmark", lambda name: spec)
        stop = cli_train.evaluate(ids["val"], data_root, out, "cpu", split_name="val")
        saved = {
            cid: dict(np.load(out / "rollouts" / f"val-{cid}.npz"))
            for cid in ids["val"]
        }
        on = cli_train.evaluate(
            ids["val"], data_root, out, "cpu", split_name="val", tail=True
        )
        assert stop["tail"] is False and on["tail"] is True
        assert stop["training_span"] == on["training_span"] == "scored"
        assert _scored(stop["mean"]) == _scored(on["mean"])
        assert "unscored_tail" not in stop["mean"]
        assert stop["mean"]["rollout_position_rmse_full"] is None
        for cid in ids["val"]:
            bound = spec.scored_frames_for(cid)
            assert _scored(stop["cases"][cid]) == _scored(on["cases"][cid])
            assert "unscored_tail" not in stop["cases"][cid]
            tail = on["cases"][cid]["unscored_tail"]
            assert tail["frames"] == [bound, stored]
            assert math.isfinite(tail["rollout_position_rmse"])
            assert math.isfinite(tail["rollout_rel_l2_displacement"])
            assert math.isfinite(on["cases"][cid]["rollout_position_rmse_full"])
            # Each saved rollout ends at its bound and carries it.
            arrays = saved[cid]
            assert arrays["predicted_positions"].shape[0] == bound
            assert int(arrays["scored_frames"]) == bound
            assert int(arrays["stored_frames"]) == stored
            assert int(arrays["input_frames"]) == spec.card.input_frames
            rolled_on = np.load(out / "rollouts" / f"val-{cid}.npz")
            assert rolled_on["predicted_positions"].shape[0] == stored
            assert int(rolled_on["scored_frames"]) == bound
            assert np.array_equal(
                rolled_on["predicted_positions"][:bound],
                arrays["predicted_positions"],
            )
        assert math.isfinite(on["mean"]["unscored_tail"]["rollout_position_rmse"])

        # 4. training_span = "full": the whole stored record, the same clock.
        calls.clear()
        full_dir = tmp_path / "full"
        full_dir.mkdir()
        _spec, _root, out_full, _ids = run(full_dir, "full")
        ((trajs_full, _make, n_full, time_ref_full),) = calls
        assert all(len(tr.time) == stored for tr in trajs_full)
        record_full = json.loads((out_full / "config.json").read_text(encoding="utf-8"))
        pool_full = record_full["run"]["training_pool"]
        assert pool_full["training_span"] == "full"
        assert pool_full["samples"] == n_full > n_samples
        assert pool_full["scored_share"] == pytest.approx(n_samples / n_full)
        assert time_ref_full == time_ref
        assert record_full["run"].get("time_ref_frames") == record["run"].get(
            "time_ref_frames"
        )
        return {"spec": spec, "data_root": data_root, "out": out, "ids": ids}

    return check
