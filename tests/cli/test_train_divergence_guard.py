"""The first-validation divergence guard for benchmarks loaded by path: the
domain scale, the one-line reason, and the CLI's exit status."""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import structbench.cli.train as cli_train
from structbench.cli.train import (
    DIVERGENCE_LOSS_CEILING,
    TrainingDiverged,
    _domain_scale,
    _early_divergence,
)


def _traj(reference):
    return SimpleNamespace(
        reference_coords=np.asarray(reference, dtype=np.float32),
        positions=np.asarray(reference, dtype=np.float32)[None],
    )


def test_domain_scale_is_the_largest_bounding_box_diagonal():
    small = _traj([[0.0, 0.0], [3.0, 4.0]])  # diagonal 5
    big = _traj([[0.0, 0.0], [6.0, 8.0], [1.0, 1.0]])  # diagonal 10
    assert _domain_scale([small, big]) == pytest.approx(10.0)
    assert _domain_scale([]) == 0.0
    # Without reference coordinates the first frame's positions stand in.
    first = SimpleNamespace(
        reference_coords=None, positions=np.zeros((2, 3, 2), dtype=np.float32)
    )
    assert _domain_scale([first]) == 0.0


def test_early_divergence_reasons():
    assert _early_divergence(0.2, 2.5, 60.0) is None
    # A healthy loss but a rollout larger than the body.
    reason = _early_divergence(0.2, 61.0, 60.0)
    assert reason is not None and "domain scale" in reason
    # A rollout inside the body but a loss orders of magnitude above O(1).
    reason = _early_divergence(9459.0, 38.5, 60.0)
    assert reason is not None and "train_loss" in reason and "domain" not in reason
    assert _early_divergence(DIVERGENCE_LOSS_CEILING, 1.0, 60.0) is None
    # Non-finite values never pass.
    assert _early_divergence(math.nan, 1.0, 60.0) is not None
    assert _early_divergence(0.1, math.inf, 60.0) is not None
    # No domain scale (no geometry): only the loss is judged.
    assert _early_divergence(0.1, 1e9, 0.0) is None
    assert "\n" not in (_early_divergence(1e4, 1e4, 1.0) or "")


def test_main_exits_3_with_the_one_line_message(monkeypatch, tmp_path, capsys):
    run_config = SimpleNamespace(
        train=SimpleNamespace(benchmark="x"), model=None, family="transolver"
    )
    monkeypatch.setattr(cli_train, "load_run_config", lambda path: run_config)
    monkeypatch.setattr(cli_train, "get_benchmark", lambda name: object())

    def diverge(*args, **kwargs):
        raise TrainingDiverged(
            "step 2000: first validation shows the run is not training"
        )

    monkeypatch.setattr(cli_train, "train", diverge)
    code = cli_train.main(
        [
            "--mode",
            "train",
            "--config",
            str(Path(tmp_path, "arm.toml")),
            "--data-root",
            str(tmp_path),
            "--out",
            str(tmp_path / "run"),
        ]
    )
    assert code == 3
    out = capsys.readouterr().out
    assert "error: step 2000: first validation shows the run is not training" in out
