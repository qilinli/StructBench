"""Notch-beam drop-weight impact benchmark (ADR-0026)."""

from ..registry import BenchmarkSpec
from ..results import BaselineResult
from .benchmark import (
    AUX_FIELD,
    CONCRETE_TYPE,
    PIN_TYPE,
    PROBE,
    QOIS,
    SUPPORT_TYPE,
    TEST_INTERP,
    TRAIN,
    VAL,
    native_mesh_transform,
)
from .card import CARD

__all__ = [
    "AUX_FIELD",
    "native_mesh_transform",
    "CARD",
    "CONCRETE_TYPE",
    "PIN_TYPE",
    "PROBE",
    "QOIS",
    "SPEC",
    "SUPPORT_TYPE",
    "TEST_INTERP",
    "TRAIN",
    "VAL",
]

#: Official baseline results (ADR-0033). Transcribed from the ``mean`` block
#: of the blessed run's held-out ``metrics-test_interp.json`` and
#: ``metrics-probe.json`` at 4 significant figures; full precision, per-case
#: numbers and the fleet spread stay in the run directory. ``val`` selects the
#: checkpoint, so it is not a number to beat and is omitted here. All scored
#: metrics use the ADR-0039 horizon (frames [6, 250) of 502).
RESULTS: tuple[BaselineResult, ...] = (
    BaselineResult(
        family="cgn",
        label="CGN baseline",
        scheme="autoregressive",
        run_commit="5956d81",
        run_date="2026-07-24",
        metrics={
            "test_interp": {
                "rollout_rel_l2_disp": 0.2827,
                "rollout_rel_l2_aux": 0.5876,
                "rollout_pos_rmse_mm": 0.2497,
                "rollout_strain_rmse": 0.01697,
                "one_step_pos_rmse_mm": 0.0006992,
                "one_step_strain_rmse": 0.0006181,
                "qoi_midspan_deflection_peak_mae_mm": 0.5843,
                "qoi_cracked_fraction_mae": 0.1892,
            },
            "probe": {
                "rollout_rel_l2_disp": 0.5905,
                "rollout_rel_l2_aux": 0.8535,
                "rollout_pos_rmse_mm": 0.3951,
                "rollout_strain_rmse": 0.01931,
                "one_step_pos_rmse_mm": 0.0006437,
                "one_step_strain_rmse": 0.0009397,
                "qoi_midspan_deflection_peak_mae_mm": 1.337,
                "qoi_cracked_fraction_mae": 0.1860,
            },
        },
        checkpoint=("models/notch_beam_2d_impact/cgn-5956d81/model-best-186000.pt"),
        checkpoint_sha256=(
            "a1d75cfaa643ee5d3a09aa2de8eb8338c675a59118057a5fcb0ff5337cb310c8"
        ),
        notes=(
            "Single-scale CGN (ADR-0034) on the ADR-0039 §4 truncated recipe "
            "with the ADR-0038 strain knobs (train_frames 250, "
            "aux_tail_weight 3, asinh aux transform at scale 0.01; hidden "
            "192 / 15 MP steps / 2-layer node MLP, noise_std 0.01, batch 4) "
            "at 250k steps; seed 1 of the 2026-07-24 h250c pair (seeds 1-2), "
            "val-selected checkpoint model-best-186000.pt (186k), one "
            "A100-80GB, ~80 h. Extending the same recipe from 200k to 250k "
            "steps cut seed-mean test rollout position RMSE 21% and "
            "deflection MAE 30% while validation strain RMSE stayed flat "
            "(0.0173 -> 0.0163): the extra budget buys kinematics, not "
            "damage-field quality. Caveats: the model over-predicts cracked "
            "fraction on the reviewed cases (crack MAE 0.19 vs sibling seed "
            "s2's 0.13, the one metric s2 wins); the off-grid probe case "
            "S_80_400_V140 is this seed's worst rollout (0.59 mm scored vs "
            "0.40 for s2); predictions break the mirror symmetry of "
            "centered-notch cases while the ground truth stays symmetric "
            "(2026-07-24 finding); full-horizon (502-frame) rollout position "
            "RMSE is 0.87 mm on test_interp - diagnostic only, not scored. "
            "Relative L2 (rollout_rel_l2_disp/aux) is the pooled space+time "
            "headline (ADR-0055), added 2026-08-16 from a re-eval on this "
            "checkpoint; RMSE reproduced to <1%, so the blessed RMSE/QoI values "
            "are unchanged."
        ),
    ),
)


def _impact_velocity(case_id: str) -> float:
    """Impact speed (m/s) from a notch case id.

    Two id formats coexist: the main grid ``NB-I-<span>-<proj>-<cfg>-<V>``
    (velocity is the last ``-`` field) and the off-grid probe cases
    ``S_<w>_<h>_V<V>_<label>`` (velocity is the ``V`` token).
    """
    if case_id.startswith("NB-"):
        return float(case_id.rsplit("-", 1)[1])
    for token in case_id.split("_"):
        if token.startswith("V") and token[1:].isdigit():
            return float(token[1:])
    raise ValueError(f"cannot parse impact velocity from notch case id {case_id!r}")


SPEC = BenchmarkSpec(
    card=CARD,
    results=RESULTS,
    splits={
        "train": tuple(TRAIN),
        "val": tuple(VAL),
        "test_interp": tuple(TEST_INTERP),
        "probe": tuple(PROBE),
    },
    eval_splits=("val", "test_interp", "probe"),
    aux_field=AUX_FIELD,
    qois=dict(QOIS),
    boundary_feature_fn=None,
    dataset_id="2D-Notched-Beam",
    kinematic_types=(PIN_TYPE, SUPPORT_TYPE),
    scored_frames=250,
    mesh_transform=native_mesh_transform,
    scripted_types=(PIN_TYPE, SUPPORT_TYPE),
    loading_scalar=_impact_velocity,
)
