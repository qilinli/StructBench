"""Data loading: canonical cases -> model-ready trajectories and samples."""

from .canonical import (
    CaseTrajectory,
    as_aux_fields,
    aux_channel_count,
    aux_channel_labels,
    aux_channel_units,
    available_aux_fields,
    load_case_trajectory,
    von_mises_from_voigt,
)
from .normalization import (
    NormalizationStats,
    aux_forward_transform,
    aux_forward_transform_channels,
    aux_inverse_transform,
    aux_inverse_transform_channels,
    cached_compute_stats,
    compute_stats,
    expand_aux_knob,
)
from .particle import WindowDataset, collate_samples

__all__ = [
    "CaseTrajectory",
    "as_aux_fields",
    "aux_channel_count",
    "aux_channel_labels",
    "aux_channel_units",
    "available_aux_fields",
    "load_case_trajectory",
    "von_mises_from_voigt",
    "NormalizationStats",
    "aux_forward_transform",
    "aux_forward_transform_channels",
    "aux_inverse_transform",
    "aux_inverse_transform_channels",
    "expand_aux_knob",
    "compute_stats",
    "cached_compute_stats",
    "WindowDataset",
    "collate_samples",
]
