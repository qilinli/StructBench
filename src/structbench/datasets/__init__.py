"""Data loading: canonical cases -> model-ready trajectories and samples."""

from .canonical import (
    CaseTrajectory,
    available_aux_fields,
    load_case_trajectory,
    von_mises_from_voigt,
)
from .normalization import (
    NormalizationStats,
    aux_forward_transform,
    aux_inverse_transform,
    cached_compute_stats,
    compute_stats,
)
from .particle import WindowDataset, collate_samples

__all__ = [
    "CaseTrajectory",
    "available_aux_fields",
    "load_case_trajectory",
    "von_mises_from_voigt",
    "NormalizationStats",
    "aux_forward_transform",
    "aux_inverse_transform",
    "compute_stats",
    "cached_compute_stats",
    "WindowDataset",
    "collate_samples",
]
