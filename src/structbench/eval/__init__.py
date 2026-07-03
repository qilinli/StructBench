"""Evaluation: autoregressive rollout, one-step eval, and benchmark metrics."""

from .metrics import (
    QoiFn,
    QoiInputs,
    arrival_time,
    field_rmse,
    final_length,
    mushroom_width,
    peak_stress,
    position_rmse,
)
from .rollout import (
    RolloutResult,
    one_step_aux_rmse,
    one_step_position_rmse,
    rollout,
)

__all__ = [
    "position_rmse",
    "field_rmse",
    "final_length",
    "mushroom_width",
    "arrival_time",
    "peak_stress",
    "QoiFn",
    "QoiInputs",
    "RolloutResult",
    "rollout",
    "one_step_position_rmse",
    "one_step_aux_rmse",
]
