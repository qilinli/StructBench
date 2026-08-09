"""Native GeoFLARE (GeoTransolver + GALE_FA) family (ADR-0041 step 3; ADR-0045).

Names beyond ``ball_query``/``standardize_coords`` land in later tasks of
this plan; re-exporting them before they exist would trip ruff's F822
(undefined name in ``__all__``).
"""

from .geo_ops import ball_query, standardize_coords

__all__ = [
    "ball_query",
    "standardize_coords",
]
