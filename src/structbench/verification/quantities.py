"""The quantity catalogue and its gates (ADR-0066 clauses 2 and 5).

Every quantity the platform *specifies* has a row here — data, not a code
path — naming the evidence it requires, when it exists at all, and what a
violation means. :func:`gate` answers for every row: traits first
(``not_applicable``), then evidence (``not_assessable`` with the missing
items named). Only *measures* are built incrementally, when a run first
supplies the evidence; a specified row that passes both gates is an
instrument gap, never a verdict about a dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..core import (
    Absence,
    AbsenceReason,
    DeclaredFacts,
    EvidenceItem,
    InputFacts,
)
from .materials import MaterialClass, material_class
from .results import Category, Measurement, Status

__all__ = ["CATALOGUE", "Quantity", "TraitGate", "gate", "get_quantity"]

E = EvidenceItem
_UNSUPPORTED = "unsupported"

#: Which artefact a violation condemns; see :attr:`Quantity.bears_on`.
BearsOn = Literal["input", "response", "run", "declared"]


@dataclass(frozen=True)
class TraitGate:
    """When a quantity exists at all. Every set flag must hold."""

    needs_particle_part: bool = False
    needs_variable_smoothing_length: bool = False
    needs_under_integrated_part: bool = False
    needs_explicit: bool = False
    needs_implicit: bool = False
    needs_mass_scaling: bool = False
    needs_mass_scaling_or_erosion: bool = False
    needs_no_mass_scaling_and_no_erosion: bool = False
    needs_erosion_off: bool = False
    needs_contact: bool = False
    needs_rigid_plane: bool = False
    needs_prescribed_motion: bool = False
    needs_two_dimensional: bool = False
    needs_plane_strain: bool = False
    needs_quasi_static_intent: bool = False
    needs_yield_law: bool = False
    needs_monotone_state_variable: bool = False
    needs_eos_material: bool = False


@dataclass(frozen=True)
class Quantity:
    """One catalogue row.

    Parameters
    ----------
    name : str
    category : Category
    unit : str
        SI unit of the measured value; ``"1"`` for dimensionless.
    requires : frozenset of EvidenceItem
        Evidence the run must supply (ADR-0066 clause 2).
    declared_fields : tuple of str
        ``DeclaredFacts`` attributes that must be set.
    gate : TraitGate
    meaning : str
        What a violation means, in one line; rendered in reports.
    bears_on : {"input", "response", "run", "declared"}
        Which artefact a violation condemns, and so who must fix it: the
        solver input deck, the stored fields a user loads, the run's
        numerical conduct and the solver's own record, or the benchmark's
        own declaration. A definition, never a measurement: it says where a
        finding of this kind lands, not where its evidence was read
        (:class:`~structbench.verification.results.Location` does that).
    title : str
        The short phrase a report shows instead of ``name``.
    display : {"plain", "percent", "count", "flag"}
        How a report formats the value.
    lower_is_worse : bool
        Whether the smallest value across cases is the one to look at.
    status : Status
    definition_version : int or None
        ``None`` while the row is only specified.
    """

    name: str
    category: Category
    unit: str
    requires: frozenset[EvidenceItem]
    declared_fields: tuple[str, ...]
    gate: TraitGate
    meaning: str
    status: Status
    definition_version: int | None
    bears_on: BearsOn
    title: str = ""
    display: Literal["plain", "percent", "count", "flag"] = "plain"
    lower_is_worse: bool = False


#: What a reader sees instead of the quantity's name: a short noun phrase.
_TITLES: dict[str, str] = {
    # numerical health
    "terminated_normally": "Solver terminated normally",
    "reached_end_time": "Run reached its requested end time",
    "solver_error_count": "Solver errors",
    "solver_warning_count": "Solver warnings",
    "solver_identity_complete": "Solver version and precision recorded",
    "timestep_min_ratio": "Smallest time step, relative to the first",
    "timestep_vs_stability_estimate": "Time step against a stability estimate",
    "added_mass_fraction": "Mass added by mass scaling, whole model",
    "added_mass_top_part_fraction": "Mass added by mass scaling, worst part",
    "added_mass_moving_fraction": "Mass added by mass scaling, moving parts",
    "implicit_convergence": "Implicit increments accepted without convergence",
    "zero_energy_mode_final_over_initial_total": "Hourglass energy vs initial energy",
    "zero_energy_mode_final_over_internal_final": "Hourglass energy vs internal energy",
    "zero_energy_mode_peak_over_internal_peak": (
        "Peak hourglass vs peak internal energy"
    ),
    "zero_energy_mode_top_part_final_over_internal_final": (
        "Hourglass energy, worst part"
    ),
    "particle_deactivated_count": "Particles deactivated without erosion",
    "particle_neighbors_min": "Fewest neighbours of any particle",
    "particle_neighbors_growth": "Growth of the largest neighbour count",
    "smoothing_length_within_input_bounds": "Smoothing length within input bounds",
    "smoothing_length_at_bound_fraction": "Particles with smoothing length at a bound",
    "rigid_surface_penetration_max": "Deepest penetration of a rigid surface",
    "prescribed_motion_realised": "Prescribed motion achieved",
    # conservation
    "energy_gain_max": "Largest energy gain during the run",
    "energy_loss_max": "Largest energy loss during the run",
    "energy_residual_final": "Energy imbalance at the end of the run",
    "total_energy_change_final": "Change in total energy, start to end",
    "quasi_static_kinetic_ratio": "Kinetic energy in a quasi-static run",
    "kinetic_energy_closure": "Kinetic energy: stored fields vs solver ledger",
    "internal_energy_closure": "Internal energy: stored fields vs solver ledger",
    "contact_energy_ratio": "Contact energy against internal energy",
    "contact_energy_negative_ratio": "Negative contact energy",
    "external_work_closure": "External work against the applied loads",
    "momentum_impulse_balance": "Momentum change against applied impulse",
    "active_mass_drift": "Drift of the total mass",
    "mass_closure": "Mass bookkeeping with scaling or deletion",
    "eos_closure": "Pressure against the equation of state",
    # constitutive
    "yield_ratio_max": "Largest stress relative to the yield surface",
    "yield_saturation_min": "Yielding material sits on the yield surface",
    "state_variable_decrease_max": "Plastic state never decreases",
    "state_variable_min": "Plastic state never negative",
    "plane_strain_ezz_max": "Out-of-plane strain in a plane-strain run",
    "out_of_plane_shear_max": "Out-of-plane shear in a two-dimensional run",
    "pressure_trace_residual": "Stored pressure against the stress trace",
    # units
    "units_anchors_consistent": "Declared unit anchors agree with the input",
    "input_density_plausible": "Most extreme input density",
    "input_constants_plausible": "Most extreme Young's modulus in the input",
    "input_strength_plausible": "Most extreme yield stress in the input",
    "input_dimensionless_groups_plausible": "Largest first-yield strain in the input",
    "response_magnitudes_plausible": "Largest speed in the response",
    "input_unit_declaration_consistent": "The input's own unit declaration",
    "density_slot_matches_input": "Stored density against the input's density",
    # data integrity
    "nonfinite_count": "Non-finite values (NaN, infinity)",
    "time_axis_monotone": "Time steps that go backward",
    "terminal_artifact_frames": "Frames written off the sampling interval",
    "elements_without_input_part": "Stored elements that no input part owns",
    "fields_match_declaration": "Stored fields against the declared field list",
    "declared_traits_match_input": "Declared traits against the solver input",
    "yield_table_matches_input": "Declared yield table against the input's",
    "yield_table_monotone": "Dips in the input's hardening table",
    "yield_table_covers_range": "Plastic strain reached, relative to the table's range",
    "input_requests_required_evidence": "Output the input asks the solver to write",
    "sampling_clock_consistent": "Energy ledger sampled on the field-output clock",
    "stored_globals_match_ledger": "Stored global energies against the solver's ledger",
}

#: Ratios a reader takes in as a percentage.
_PERCENT = frozenset(
    {
        "added_mass_fraction",
        "added_mass_top_part_fraction",
        "added_mass_moving_fraction",
        "zero_energy_mode_final_over_initial_total",
        "zero_energy_mode_final_over_internal_final",
        "zero_energy_mode_peak_over_internal_peak",
        "zero_energy_mode_top_part_final_over_internal_final",
        "smoothing_length_at_bound_fraction",
        "energy_gain_max",
        "energy_loss_max",
        "energy_residual_final",
        "total_energy_change_final",
        "quasi_static_kinetic_ratio",
        "kinetic_energy_closure",
        "internal_energy_closure",
        "contact_energy_ratio",
        "contact_energy_negative_ratio",
        "external_work_closure",
        "momentum_impulse_balance",
        "active_mass_drift",
        "mass_closure",
        "eos_closure",
        "density_slot_matches_input",
        "pressure_trace_residual",
    }
)
#: Whole numbers.
_COUNTS = frozenset(
    {
        "solver_error_count",
        "solver_warning_count",
        "solver_identity_complete",
        "implicit_convergence",
        "particle_deactivated_count",
        "particle_neighbors_min",
        "units_anchors_consistent",
        "nonfinite_count",
        "time_axis_monotone",
        "terminal_artifact_frames",
        "elements_without_input_part",
        "fields_match_declaration",
        "declared_traits_match_input",
        "yield_table_monotone",
    }
)
#: One means yes.
_FLAGS = frozenset({"terminated_normally"})
#: Quantities whose *smallest* value is the one to look at.
_LOWER_IS_WORSE = frozenset(
    {
        "terminated_normally",
        "reached_end_time",
        "timestep_min_ratio",
        "particle_neighbors_min",
        "yield_saturation_min",
        "state_variable_min",
    }
)
#: Which artefact a violation condemns — the definition, not the evidence read.
_BEARS_ON: dict[str, BearsOn] = {
    # The solver input deck: the fix is a corrected deck and a new run.
    "input_constants_plausible": "input",
    "input_density_plausible": "input",
    "input_dimensionless_groups_plausible": "input",
    "input_strength_plausible": "input",
    "input_unit_declaration_consistent": "input",
    "yield_table_covers_range": "input",
    "yield_table_monotone": "input",
    # The benchmark's own declaration: the fix is a corrected card.
    "declared_traits_match_input": "declared",
    "fields_match_declaration": "declared",
    "units_anchors_consistent": "declared",
    "yield_table_matches_input": "declared",
    # The stored fields a user loads: what is trained on is wrong or short.
    "active_mass_drift": "response",
    "density_slot_matches_input": "response",
    "elements_without_input_part": "response",
    "eos_closure": "response",
    "internal_energy_closure": "response",
    "kinetic_energy_closure": "response",
    "nonfinite_count": "response",
    "out_of_plane_shear_max": "response",
    "particle_deactivated_count": "response",
    "plane_strain_ezz_max": "response",
    "pressure_trace_residual": "response",
    "reached_end_time": "response",
    "response_magnitudes_plausible": "response",
    "state_variable_decrease_max": "response",
    "state_variable_min": "response",
    "input_requests_required_evidence": "input",
    "stored_globals_match_ledger": "response",
    "terminal_artifact_frames": "response",
    "time_axis_monotone": "response",
    "yield_ratio_max": "response",
    "yield_saturation_min": "response",
    # The run's numerical conduct and the solver's record: the arrays may be
    # intact, but the physics behind them is in question.
    "added_mass_fraction": "run",
    "added_mass_moving_fraction": "run",
    "added_mass_top_part_fraction": "run",
    "contact_energy_negative_ratio": "run",
    "contact_energy_ratio": "run",
    "energy_gain_max": "run",
    "energy_loss_max": "run",
    "energy_residual_final": "run",
    "external_work_closure": "run",
    "implicit_convergence": "run",
    "mass_closure": "run",
    "momentum_impulse_balance": "run",
    "particle_neighbors_growth": "run",
    "particle_neighbors_min": "run",
    "prescribed_motion_realised": "run",
    "quasi_static_kinetic_ratio": "run",
    "rigid_surface_penetration_max": "run",
    "sampling_clock_consistent": "run",
    "smoothing_length_at_bound_fraction": "run",
    "smoothing_length_within_input_bounds": "run",
    "solver_error_count": "run",
    "solver_identity_complete": "run",
    "solver_warning_count": "run",
    "terminated_normally": "run",
    "timestep_min_ratio": "run",
    "timestep_vs_stability_estimate": "run",
    "total_energy_change_final": "run",
    "zero_energy_mode_final_over_initial_total": "run",
    "zero_energy_mode_final_over_internal_final": "run",
    "zero_energy_mode_peak_over_internal_peak": "run",
    "zero_energy_mode_top_part_final_over_internal_final": "run",
}


def _display(name: str) -> Literal["plain", "percent", "count", "flag"]:
    if name in _PERCENT:
        return "percent"
    if name in _COUNTS:
        return "count"
    return "flag" if name in _FLAGS else "plain"


def _row(
    name: str,
    category: Category,
    unit: str,
    requires: set[EvidenceItem],
    meaning: str,
    *,
    gate: TraitGate | None = None,
    declared: tuple[str, ...] = (),
    implemented: bool = False,
) -> Quantity:
    return Quantity(
        name,
        category,
        unit,
        frozenset(requires),
        declared,
        gate or TraitGate(),
        meaning,
        Status.IMPLEMENTED if implemented else Status.SPECIFIED,
        1 if implemented else None,
        _BEARS_ON[name],  # an unclassified row is a KeyError at import
        _TITLES[name],  # a row without a title is a KeyError at import
        _display(name),
        name in _LOWER_IS_WORSE,
    )


_H, _C, _K, _U, _D = (
    Category.NUMERICAL_HEALTH,
    Category.CONSERVATION,
    Category.CONSTITUTIVE,
    Category.UNITS,
    Category.DATA_INTEGRITY,
)
_PARTICLE = TraitGate(needs_particle_part=True)
_EXPLICIT = TraitGate(needs_explicit=True)
_MASS_SCALED = TraitGate(needs_explicit=True, needs_mass_scaling=True)
_ZERO_ENERGY = TraitGate(needs_under_integrated_part=True)
_CONTACT = TraitGate(needs_contact=True)
_YIELD = TraitGate(needs_yield_law=True)
_STATE = TraitGate(needs_monotone_state_variable=True)

_ROWS = (
    # ------------------------------------------------------ numerical health
    _row(
        "terminated_normally",
        _H,
        "1",
        {E.E3},
        "the solver did not finish the run",
        implemented=True,
    ),
    _row(
        "reached_end_time",
        _H,
        "1",
        {E.E1, E.E8},
        "the stored response stops short of the requested end time",
        implemented=True,
    ),
    _row(
        "solver_error_count",
        _H,
        "1",
        {E.E3},
        "the solver reported errors",
        implemented=True,
    ),
    _row(
        "solver_warning_count",
        _H,
        "1",
        {E.E3},
        "the solver reported warnings that may affect the result",
        implemented=True,
    ),
    _row(
        "solver_identity_complete",
        _H,
        "1",
        {E.E2},
        "the run cannot be attributed to a solver version and precision",
        implemented=True,
    ),
    _row(
        "timestep_min_ratio",
        _H,
        "1",
        {E.E4},
        "the time step collapsed during the run",
        gate=_EXPLICIT,
        implemented=True,
    ),
    _row(
        "timestep_vs_stability_estimate",
        _H,
        "1",
        {E.E1, E.E4, E.E8},
        "the step is governed by something other than size and wave speed",
        gate=_EXPLICIT,
    ),
    _row(
        "added_mass_fraction",
        _H,
        "1",
        {E.E4},
        "mass scaling changed the inertia of the model",
        gate=_MASS_SCALED,
    ),
    _row(
        "added_mass_top_part_fraction",
        _H,
        "1",
        {E.E4},
        "mass scaling changed the inertia of one part",
        gate=_MASS_SCALED,
    ),
    _row(
        "added_mass_moving_fraction",
        _H,
        "1",
        {E.E4},
        "mass scaling changed the inertia of the moving parts",
        gate=_MASS_SCALED,
    ),
    _row(
        "implicit_convergence",
        _H,
        "1",
        {E.E4},
        "an increment was accepted without equilibrium",
        gate=TraitGate(needs_implicit=True),
    ),
    _row(
        "zero_energy_mode_final_over_initial_total",
        _H,
        "1",
        {E.E5},
        "zero-energy modes absorbed a share of the run's energy",
        gate=_ZERO_ENERGY,
    ),
    _row(
        "zero_energy_mode_final_over_internal_final",
        _H,
        "1",
        {E.E5},
        "zero-energy modes rival the internal energy",
        gate=_ZERO_ENERGY,
    ),
    _row(
        "zero_energy_mode_peak_over_internal_peak",
        _H,
        "1",
        {E.E5},
        "zero-energy modes rival the internal energy at their peak",
        gate=_ZERO_ENERGY,
    ),
    _row(
        "zero_energy_mode_top_part_final_over_internal_final",
        _H,
        "1",
        {E.E5, E.E6},
        "zero-energy modes rival the internal energy of one part",
        gate=_ZERO_ENERGY,
    ),
    _row(
        "particle_deactivated_count",
        _H,
        "1",
        {E.E8},
        "particles were removed although the input enables no erosion",
        gate=TraitGate(needs_particle_part=True, needs_erosion_off=True),
        implemented=True,
    ),
    _row(
        "particle_neighbors_min",
        _H,
        "1",
        {E.E8},
        "particles ran short of neighbours (kernel starvation)",
        gate=_PARTICLE,
        implemented=True,
    ),
    _row(
        "particle_neighbors_growth",
        _H,
        "1",
        {E.E8},
        "particles clumped (neighbour counts grew)",
        gate=_PARTICLE,
        implemented=True,
    ),
    _row(
        "smoothing_length_within_input_bounds",
        _H,
        "1",
        {E.E1, E.E8},
        "stored smoothing lengths leave the input's own bounds (mapping error)",
        gate=_PARTICLE,
        implemented=True,
    ),
    _row(
        "smoothing_length_at_bound_fraction",
        _H,
        "1",
        {E.E1, E.E8},
        "smoothing lengths saturate at a bound (starvation or clumping)",
        gate=TraitGate(needs_particle_part=True, needs_variable_smoothing_length=True),
        implemented=True,
    ),
    _row(
        "rigid_surface_penetration_max",
        _H,
        "m",
        {E.E1, E.E8},
        "material passed through a rigid surface",
        gate=TraitGate(needs_rigid_plane=True),
        implemented=True,
    ),
    _row(
        "prescribed_motion_realised",
        _H,
        "1",
        {E.E1, E.E7, E.E8},
        "a prescribed motion was not achieved",
        gate=TraitGate(needs_prescribed_motion=True),
    ),
    # --------------------------------------------------------- conservation
    _row(
        "energy_gain_max",
        _C,
        "1",
        {E.E5},
        "energy was created during the run",
        implemented=True,
    ),
    _row(
        "energy_loss_max",
        _C,
        "1",
        {E.E5},
        "energy went unaccounted for",
        implemented=True,
    ),
    _row(
        "energy_residual_final",
        _C,
        "1",
        {E.E5},
        "the energy balance does not close",
        implemented=True,
    ),
    _row(
        "total_energy_change_final",
        _C,
        "1",
        {E.E5},
        "total energy changed between the start and the end of the run",
        implemented=True,
    ),
    _row(
        "quasi_static_kinetic_ratio",
        _C,
        "1",
        {E.E5},
        "inertia is not negligible in a run declared quasi-static",
        gate=TraitGate(needs_quasi_static_intent=True),
    ),
    _row(
        "kinetic_energy_closure",
        _C,
        "1",
        {E.E5, E.E8, E.E9},
        "stored velocities and masses do not reproduce the ledger's kinetic energy",
        implemented=True,
    ),
    _row(
        "internal_energy_closure",
        _C,
        "1",
        {E.E5, E.E8, E.E9},
        "stored internal energy does not reproduce the ledger's",
    ),
    _row(
        "contact_energy_ratio",
        _C,
        "1",
        {E.E5, E.E6},
        "contact stored or released a share of the run's energy",
        gate=_CONTACT,
    ),
    _row(
        "contact_energy_negative_ratio",
        _C,
        "1",
        {E.E5, E.E6},
        "a contact interface created energy",
        gate=_CONTACT,
    ),
    _row(
        "external_work_closure",
        _C,
        "1",
        {E.E5, E.E7},
        "the ledger's external work disagrees with the applied loads",
    ),
    _row(
        "momentum_impulse_balance",
        _C,
        "1",
        {E.E7, E.E8},
        "momentum change disagrees with the applied impulse",
    ),
    _row(
        "active_mass_drift",
        _C,
        "1",
        {E.E8},
        "mass changed although nothing adds or removes it",
        gate=TraitGate(needs_no_mass_scaling_and_no_erosion=True),
        implemented=True,
    ),
    _row(
        "mass_closure",
        _C,
        "1",
        {E.E4, E.E6, E.E8},
        "active, deleted and added mass do not sum to the initial mass",
        gate=TraitGate(needs_mass_scaling_or_erosion=True),
    ),
    _row(
        "eos_closure",
        _C,
        "1",
        {E.E1, E.E8},
        "stored pressure disagrees with the equation of state",
        gate=TraitGate(needs_eos_material=True),
    ),
    # --------------------------------------------------------- constitutive
    _row(
        "yield_ratio_max",
        _K,
        "1",
        {E.E1, E.E8},
        "stress lies outside the yield surface (export or post-processing error)",
        gate=_YIELD,
        declared=("yield_table",),
        implemented=True,
    ),
    _row(
        "yield_saturation_min",
        _K,
        "1",
        {E.E1, E.E8},
        "yielding material sits far inside the surface (stress or table scale error)",
        gate=_YIELD,
        declared=("yield_table",),
        implemented=True,
    ),
    _row(
        "state_variable_decrease_max",
        _K,
        "1",
        {E.E1, E.E8},
        "an irreversible state variable decreased",
        gate=_STATE,
        implemented=True,
    ),
    _row(
        "state_variable_min",
        _K,
        "1",
        {E.E1, E.E8},
        "a state variable left its admissible range",
        gate=_STATE,
        implemented=True,
    ),
    _row(
        "plane_strain_ezz_max",
        _K,
        "1",
        {E.E1, E.E8},
        "out-of-plane strain is non-zero in a plane-strain run",
        gate=TraitGate(needs_plane_strain=True),
        implemented=True,
    ),
    _row(
        "out_of_plane_shear_max",
        _K,
        "Pa",
        {E.E1, E.E8},
        "out-of-plane shear is non-zero in a two-dimensional run",
        gate=TraitGate(needs_two_dimensional=True),
        implemented=True,
    ),
    _row(
        "pressure_trace_residual",
        _K,
        "1",
        {E.E8},
        "stored pressure disagrees with the stress trace",
        implemented=True,
    ),
    # ---------------------------------------------------------------- units
    _row(
        "units_anchors_consistent",
        _U,
        "1",
        {E.E1, E.E10A, E.E10B},
        "the declared unit system contradicts a declared SI anchor",
        implemented=True,
    ),
    _row(
        "input_density_plausible",
        _U,
        "kg/m^3",
        {E.E1, E.E10A},
        "density is outside the range of condensed matter (mass-unit error)",
        implemented=True,
    ),
    _row(
        "input_constants_plausible",
        _U,
        "Pa",
        {E.E1, E.E10A},
        "a Young's modulus is outside its plausible range",
        implemented=True,
    ),
    _row(
        "input_strength_plausible",
        _U,
        "Pa",
        {E.E1, E.E10A},
        "a yield or compressive strength is outside its plausible range",
        implemented=True,
    ),
    _row(
        "input_dimensionless_groups_plausible",
        _U,
        "1",
        {E.E1},
        "the input mixes units within a material definition",
        implemented=True,
    ),
    _row(
        "response_magnitudes_plausible",
        _U,
        "m/s",
        {E.E8, E.E10A},
        "a response speed is outside the structural-impact regime",
        implemented=True,
    ),
    _row(
        "input_unit_declaration_consistent",
        _U,
        "1",
        {E.E1, E.E10A},
        "the input's own unit declaration contradicts the declared units",
    ),
    _row(
        "density_slot_matches_input",
        _U,
        "1",
        {E.E1, E.E8},
        "stored density disagrees with the input's (ingestion mapping error)",
        implemented=True,
    ),
    # ------------------------------------------------------- data integrity
    _row(
        "nonfinite_count",
        _D,
        "1",
        {E.E8},
        "the response contains non-finite values",
        implemented=True,
    ),
    _row(
        "time_axis_monotone",
        _D,
        "1",
        {E.E8},
        "the time axis does not increase strictly",
        implemented=True,
    ),
    _row(
        "terminal_artifact_frames",
        _D,
        "1",
        {E.E8},
        "frames were written off the sampling interval - a fact, not a defect: a"
        " solver writes its state at the termination time and loaders drop it"
        " (ADR-0028)",
        implemented=True,
    ),
    _row(
        "elements_without_input_part",
        _D,
        "1",
        {E.E1, E.E8},
        "the case holds elements the input does not define",
        implemented=True,
    ),
    _row(
        "fields_match_declaration",
        _D,
        "1",
        {E.E8},
        "stored fields differ from the declared field list",
        declared=("fields",),
        implemented=True,
    ),
    _row(
        "declared_traits_match_input",
        _D,
        "1",
        {E.E1},
        "a declared trait contradicts the solver input",
        declared=("discretisation", "erosion"),
        implemented=True,
    ),
    _row(
        "yield_table_matches_input",
        _D,
        "1",
        {E.E1},
        "the declared yield table differs from the input's",
        gate=_YIELD,
        declared=("yield_table",),
        implemented=True,
    ),
    _row(
        "yield_table_monotone",
        _D,
        "1",
        {E.E1},
        "the input's hardening table is not monotone (input defect)",
        gate=_YIELD,
        implemented=True,
    ),
    _row(
        "yield_table_covers_range",
        _D,
        "1",
        {E.E1, E.E8},
        "plastic strain exceeds the last knot of the hardening table",
        gate=_YIELD,
        declared=("yield_table",),
        implemented=True,
    ),
    _row(
        "input_requests_required_evidence",
        _D,
        "count",
        {E.E1},
        "the input does not ask the solver for evidence the platform requires",
        implemented=True,
    ),
    _row(
        "sampling_clock_consistent",
        _D,
        "1",
        {E.E5, E.E8, E.E9},
        "the ledger is not sampled on the field-output clock",
    ),
    _row(
        "stored_globals_match_ledger",
        _D,
        "1",
        {E.E5, E.E8, E.E9},
        "globals stored with the case disagree with the ledger (ingestion error)",
        implemented=True,
    ),
)

#: Every specified quantity, sorted by name.
CATALOGUE: tuple[Quantity, ...] = tuple(sorted(_ROWS, key=lambda q: q.name))
_BY_NAME = {q.name: q for q in CATALOGUE}


def get_quantity(name: str) -> Quantity:
    """Return the catalogue row called ``name``.

    Raises
    ------
    KeyError
        Listing the valid names.
    """
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(
            f"unknown quantity {name!r}; valid: {sorted(_BY_NAME)}"
        ) from None


def _classes(facts: InputFacts) -> list[MaterialClass | None]:
    """Material class of every part's material (``None`` = unsupported)."""
    by_id = {m.material_id: m for m in facts.materials}
    out: list[MaterialClass | None] = []
    for part in facts.parts:
        material = by_id.get(part.material_id)
        out.append(material_class(material.canonical_model) if material else None)
    return out


def _any_class(facts: InputFacts, attribute: str, wanted: object) -> bool | str | None:
    classes = _classes(facts)
    if not classes:
        return None
    if any(c is not None and getattr(c, attribute) == wanted for c in classes):
        return True
    structural = [c for c in classes if c is None or c.structural]
    return _UNSUPPORTED if any(c is None for c in structural) else False


def _trait(
    flag: str, facts: InputFacts, declared: DeclaredFacts | None
) -> bool | str | None:
    """Evaluate one ``TraitGate`` flag: True, False, None (unknown), or unsupported."""
    scaled, eroding = facts.mass_scaling_enabled, facts.erosion_enabled
    match flag:
        case "needs_particle_part":
            if not facts.parts:
                return None
            return any(p.discretisation == "particle" for p in facts.parts)
        case "needs_variable_smoothing_length":
            bounds = facts.smoothing_length_scale_bounds
            return None if bounds is None else bounds[0] != bounds[1]
        case "needs_under_integrated_part":
            meshed = [
                p.under_integrated
                for p in facts.parts
                if p.discretisation != "particle"
            ]
            if not facts.parts:
                return None
            if any(v is True for v in meshed):
                return True
            return False if all(v is False for v in meshed) else None
        case "needs_explicit" | "needs_implicit":
            if facts.time_integration is None:
                return None
            return facts.time_integration == flag.removeprefix("needs_")
        case "needs_mass_scaling":
            return scaled
        case "needs_erosion_off":
            return None if eroding is None else not eroding
        case "needs_mass_scaling_or_erosion":
            if scaled or eroding:
                return True
            return False if scaled is False and eroding is False else None
        case "needs_no_mass_scaling_and_no_erosion":
            if scaled or eroding:
                return False
            return True if scaled is False and eroding is False else None
        case "needs_contact":
            return facts.contact_defined
        case "needs_rigid_plane":
            return bool(facts.rigid_planes)
        case "needs_prescribed_motion":
            return facts.prescribed_motion_defined
        case "needs_two_dimensional":
            return None if facts.dimension is None else facts.dimension == 2
        case "needs_plane_strain":
            return False if facts.dimension == 3 else facts.plane_strain
        case "needs_quasi_static_intent":
            return bool(declared and declared.quasi_static)
        case "needs_yield_law":
            return _any_class(facts, "yield_law", "tabulated_j2")
        case "needs_monotone_state_variable":
            return _any_class(facts, "monotone", True)
        case "needs_eos_material":
            return _any_class(facts, "has_equation_of_state", True)
    raise ValueError(f"unknown trait flag {flag!r}")


def _absent(q: Quantity, reason: AbsenceReason, *missing: EvidenceItem) -> Measurement:
    return Measurement(
        q.name, None, q.unit, absence=Absence(reason, frozenset(missing))
    )


def gate(
    q: Quantity,
    facts: InputFacts | None,
    declared: DeclaredFacts | None,
    supplied: frozenset[EvidenceItem],
) -> Measurement | None:
    """Decide whether ``q`` can be measured on this run.

    Traits are evaluated first, then evidence (ADR-0066 clause 5). Unknown
    traits never yield ``not_applicable``.

    Parameters
    ----------
    q : Quantity
    facts : InputFacts or None
        ``None`` when the run supplied no solver input (E1).
    declared : DeclaredFacts or None
    supplied : frozenset of EvidenceItem
        The evidence items the run supplied.

    Returns
    -------
    Measurement or None
        A ``not_applicable`` or absent measurement, or ``None`` when the
        quantity is applicable, assessable and implemented.
    """
    flags = [name for name, on in vars(q.gate).items() if on]
    if flags and facts is None:
        return _absent(q, AbsenceReason.SOURCE_MISSING, E.E1)
    if facts is not None:
        results = [_trait(flag, facts, declared) for flag in flags]
        if any(r is False for r in results):
            return Measurement(q.name, None, q.unit, not_applicable=True)
        if _UNSUPPORTED in results:
            return _absent(q, AbsenceReason.UNSUPPORTED)
        if any(r is None for r in results):
            reason = (
                AbsenceReason.UNPARSABLE
                if facts.unparsable
                else AbsenceReason.SOURCE_MISSING
            )
            return _absent(q, reason, E.E1)

    missing = q.requires - supplied
    if missing - {E.E10B}:
        return _absent(q, AbsenceReason.SOURCE_MISSING, *sorted(missing))
    if missing:
        return _absent(q, AbsenceReason.NO_DECLARATION_HOME, E.E10B)
    for name in q.declared_fields:
        if declared is None or getattr(declared, name) in (None, (), frozenset()):
            return _absent(q, AbsenceReason.SOURCE_MISSING)

    if q.status is Status.SPECIFIED:
        return _absent(q, AbsenceReason.UNSUPPORTED)
    return None
