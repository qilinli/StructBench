"""Material-class semantics for the constitutive checks (ADR-0066 clause 6).

Material knowledge has two homes. *Solver keyword -> material class* is the
adapter's job (ADR-0012's ``canonical_model``, stored with each case). *Class
-> what its stored state variable means, how it is bounded, and which yield
law can be assessed* is solver-neutral and lives here, keyed on
``canonical_model``. The same stored field is plastic strain for one class
and a damage measure for another; only the class can say which.

A class is added when the first dataset that uses it arrives, with the
material-class ADR that ADR-0012 anticipates. A class that is not listed is
unsupported: its constitutive rows read ``not_assessable / unsupported``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

__all__ = ["MATERIAL_CLASSES", "MaterialClass", "material_class"]


@dataclass(frozen=True)
class MaterialClass:
    """What the platform knows about one material class.

    Parameters
    ----------
    canonical_model : str
        ADR-0012 class name.
    state_variable : {"plastic_strain", "damage", "none"}
        Meaning of the stored scalar internal variable.
    state_bounds : (float, float or None) or None
        Admissible range of that variable; ``None`` upper bound = unbounded.
    monotone : bool
        Whether the variable can never decrease.
    yield_law : {"tabulated_j2", "none", "not_assessable"}
        ``"tabulated_j2"``: von Mises stress bounded by a tabulated function
        of plastic strain. ``"not_assessable"``: a yield surface exists but
        its arguments are not exported. ``"none"``: no yield surface.
    has_equation_of_state : bool
        Pressure, density and internal energy are independent state.
    structural : bool
        ``False`` for classes that carry no load (a null or rigid part).
    """

    canonical_model: str
    state_variable: Literal["plastic_strain", "damage", "none"]
    state_bounds: tuple[float, float | None] | None
    monotone: bool
    yield_law: Literal["tabulated_j2", "none", "not_assessable"]
    has_equation_of_state: bool = False
    structural: bool = True


_CLASSES = (
    MaterialClass(
        "elastic_plastic_hydro",
        state_variable="plastic_strain",
        state_bounds=(0.0, None),
        monotone=True,
        yield_law="tabulated_j2",
        has_equation_of_state=True,
    ),
    MaterialClass("null", "none", None, False, "none", structural=False),
    MaterialClass("rigid", "none", None, False, "none", structural=False),
)

#: Supported classes, keyed on ``canonical_model``.
MATERIAL_CLASSES: Mapping[str, MaterialClass] = MappingProxyType(
    {cls.canonical_model: cls for cls in _CLASSES}
)


def material_class(canonical_model: str | None) -> MaterialClass | None:
    """Return the class record, or ``None`` when the class is unsupported.

    Parameters
    ----------
    canonical_model : str or None
        ``Material.canonical_model`` / ``MaterialInput.canonical_model``.
    """
    if canonical_model is None:
        return None
    return MATERIAL_CLASSES.get(canonical_model)
