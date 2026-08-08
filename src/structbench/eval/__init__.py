"""Evaluation: autoregressive rollout, one-step eval, and benchmark metrics."""

from .metrics import (
    QoiFn,
    QoiInputs,
    arrival_time,
    cracked_fraction,
    field_rmse,
    final_length,
    midspan_deflection_peak,
    mushroom_width,
    peak_mean_aux,
    peak_nodal_aux,
    peak_stress,
    position_rmse,
    t_peak_mean_aux,
    terminal_peak_displacement,
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
    "peak_mean_aux",
    "t_peak_mean_aux",
    "midspan_deflection_peak",
    "cracked_fraction",
    "peak_nodal_aux",
    "terminal_peak_displacement",
    "QoiFn",
    "QoiInputs",
    "RolloutResult",
    "rollout",
    "one_step_position_rmse",
    "one_step_aux_rmse",
]
