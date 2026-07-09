"""FEM-postprocessor-style visualization of particle physics fields.

Any figure that shows a physics quantity (von Mises stress, plastic
strain, ...) renders through this module so the color code and fringe-bar
conventions match what structural engineers know from LS-PrePost and
Abaqus/CAE (ADR-0022). Requires the ``viz`` extra (matplotlib).
"""

from .fringe import (
    FIELDS,
    CaseField,
    FieldSpec,
    animate_comparison,
    animate_rollout,
    compare_rollout,
    fringe_scatter,
    load_case_field,
    snapshot,
)

__all__ = [
    "FIELDS",
    "CaseField",
    "FieldSpec",
    "animate_comparison",
    "animate_rollout",
    "compare_rollout",
    "fringe_scatter",
    "load_case_field",
    "snapshot",
]
