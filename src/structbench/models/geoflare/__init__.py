"""Native GeoFLARE (GeoTransolver + GALE_FA) family (ADR-0041 step 3; ADR-0045).

``ContextTokenizer``/``GeometricFeatureProcessor`` stay module-internal to
``context.py`` (imported directly where needed, e.g. in tests); only the
top-level ``MultiScaleContext`` assembly is re-exported here. Names beyond
these land in later tasks of this plan; re-exporting them before they
exist would trip ruff's F822 (undefined name in ``__all__``).
"""

from .context import MultiScaleContext
from .geo_ops import ball_query, standardize_coords

__all__ = [
    "MultiScaleContext",
    "ball_query",
    "standardize_coords",
]
