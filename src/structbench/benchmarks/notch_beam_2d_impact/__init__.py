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
        family="mgn",
        label="MGN",
        scheme="autoregressive",
        reference=(
            "Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., & Battaglia, P. W. "
            "(2021). Learning Mesh-Based Simulation with Graph Networks. *ICLR*. "
            "https://arxiv.org/abs/2010.03409"
        ),
        provisional=True,
        run_commit="59d5786",
        run_date="2026-08-17",
        metrics={
            "test_interp": {
                "rollout_rel_l2_disp": 0.2245,
                "rollout_rel_l2_aux": 0.6988,
                "rollout_pos_rmse_mm": 0.1984,
                "rollout_strain_rmse": 0.01997,
                "one_step_pos_rmse_mm": 0.001448,
                "one_step_strain_rmse": 0.0009101,
                "qoi_midspan_deflection_peak_mae_mm": 0.5842,
                "qoi_cracked_fraction_mae": 0.1081,
            },
            "probe": {
                "rollout_rel_l2_disp": 0.7672,
                "rollout_rel_l2_aux": 1.255,
                "rollout_pos_rmse_mm": 0.2834,
                "rollout_strain_rmse": 0.02077,
                "one_step_pos_rmse_mm": 0.002156,
                "one_step_strain_rmse": 0.001003,
                "qoi_midspan_deflection_peak_mae_mm": 0.7027,
                "qoi_cracked_fraction_mae": 0.2225,
            },
        },
        notes=(
            "Native MeshGraphNets (ADR-0047 baseline, ADR-0049 repair): "
            "autoregressive next-step on the SPH particle set as a mesh, "
            "val-selected model-best-212000.pt, seed 2 of the s1-s2 pair, run "
            "notch-mgn-base-s2. PROVISIONAL (ADR-0044/0045). On test_interp it "
            "roughly matches CGN (displacement relative L2 0.22 vs 0.28) and "
            "both trail Transolver ~6x; on the off-centre triple-OOD PROBE the "
            "relative-position message passing degrades more gracefully than the "
            "global-attention operator (disp 0.77 vs Transolver 1.05), though "
            "CGN is best there (0.59). Pooled relative L2 headline (ADR-0055) "
            "from the 2026-08-17 re-eval."
        ),
    ),
    BaselineResult(
        family="cgn",
        label="CGN",
        scheme="autoregressive",
        reference=(
            "Li, Q., Wang, Z., Li, L., Hao, H., Chen, W., & Shao, Y. (2023). "
            "Machine learning prediction of structural dynamic responses using "
            "graph neural networks. *Computers & Structures*, 289, 107188. "
            "https://doi.org/10.1016/j.compstruc.2023.107188"
        ),
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
    BaselineResult(
        family="transolver",
        label="Transolver",
        scheme="time-conditioned",
        reference=(
            "Wu, H., Luo, H., Wang, H., Wang, J., & Long, M. (2024). Transolver: "
            "A Fast Transformer Solver for PDEs on General Geometries. *ICML*. "
            "https://arxiv.org/abs/2402.02366"
        ),
        provisional=True,
        run_commit="59d5786",
        run_date="2026-08-16",
        metrics={
            "test_interp": {
                "rollout_rel_l2_disp": 0.03517,
                "rollout_rel_l2_aux": 0.2294,
                "rollout_pos_rmse_mm": 0.03365,
                "rollout_strain_rmse": 0.006467,
                "qoi_midspan_deflection_peak_mae_mm": 0.04067,
                "qoi_cracked_fraction_mae": 0.0212,
            },
            "probe": {
                "rollout_rel_l2_disp": 1.047,
                "rollout_rel_l2_aux": 1.236,
                "rollout_pos_rmse_mm": 0.6319,
                "rollout_strain_rmse": 0.02965,
                "qoi_midspan_deflection_peak_mae_mm": 3.405,
                "qoi_cracked_fraction_mae": 0.04823,
            },
        },
        notes=(
            "Native time-conditioned Transolver (ADR-0054): history-free "
            "independent-time-query, no rollout accumulation, so one-step is "
            "N/A. Seed 2 of the s1-s2 pair, val-selected model-best-230000.pt, "
            "run notch-transolver-tc-s2. PROVISIONAL (ADR-0044/0045). On "
            "test_interp it is the strongest baseline (~8x lower displacement "
            "relative L2 than CGN); on the PROBE it fails hard (relative L2 > 1) "
            "- the off-centre triple-OOD case (ADR-0026 amendment) where the "
            "global-attention operator mis-localises the response to the learned "
            "midspan prior, while CGN's relative-position message passing "
            "degrades more gracefully (probe disp 0.59 vs 1.05). Pooled "
            "relative L2 headline (ADR-0055) from the 2026-08-16 re-eval."
        ),
    ),
    BaselineResult(
        family="transolver_plus",
        label="Transolver++",
        scheme="time-conditioned",
        reference=(
            "Luo, H., Wu, H., Zhou, H., Wang, J., & Long, M. (2025). "
            "Transolver++: An Accurate Neural Solver for PDEs on Million-Scale "
            "Geometries. https://arxiv.org/abs/2502.02414. Adapted per ADR-0057 "
            "(thuml reference implementation github.com/thuml/Transolver_plus)."
        ),
        provisional=True,
        run_commit="5b84119",
        run_date="2026-08-18",
        metrics={
            "test_interp": {
                "rollout_rel_l2_disp": 0.03699,
                "rollout_rel_l2_aux": 0.2396,
                "rollout_pos_rmse_mm": 0.03509,
                "rollout_strain_rmse": 0.006714,
                "qoi_midspan_deflection_peak_mae_mm": 0.04696,
                "qoi_cracked_fraction_mae": 0.02438,
            },
            "probe": {
                "rollout_rel_l2_disp": 1.010,
                "rollout_rel_l2_aux": 1.263,
                "rollout_pos_rmse_mm": 0.6148,
                "rollout_strain_rmse": 0.03025,
                "qoi_midspan_deflection_peak_mae_mm": 3.329,
                "qoi_cracked_fraction_mae": 0.04194,
            },
        },
        notes=(
            "Transolver++ (ADR-0057): the native time-conditioned Transolver "
            "with both eidetic-state knobs ON - per-point adaptive slice "
            "temperature + train-only Gumbel Rep-Slice reparameterisation. Seed "
            "2 of the s1-s2 pair (val-selected model-best-250000.pt, run "
            "notch-transolver-tcpp-s2), seed-matched to the plain Transolver "
            "entry above (also seed 2). PROVISIONAL method comparison (ADR-0046), "
            "not a blessed baseline. It is neutral-to-worse than plain "
            "Transolver: test_interp displacement is ~5% worse (0.03699 vs "
            "0.03517) and strain ~4% worse; on the off-centre triple-OOD PROBE "
            "displacement is marginally better (1.01 vs 1.05) but strain is "
            "worse (1.263 vs 1.236) and both still fail hard (relative L2 > 1). "
            "No robust generalisation gain. Consistent with Transolver++ being "
            "designed for million-scale geometries: its eidetic-state edits do "
            "not help these O(10^4)-particle impact meshes. Pooled relative L2 "
            "headline (ADR-0055)."
        ),
    ),
)


def _impact_velocity(case_id: str) -> float:
    """Impact speed (m/s) from a notch case id.

    Two id formats coexist: the main grid ``NB-I-<W>-<shape>-<notch>-<V>``
    (velocity is the last ``-`` field) and the off-grid probe cases
    ``S_<H>_<W>_V<V>_<label>`` (velocity is the ``V`` token).
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
