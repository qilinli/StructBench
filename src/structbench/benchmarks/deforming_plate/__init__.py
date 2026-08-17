"""DeformingPlate benchmark: MeshGraphNets quasi-static rollout (ADR-0043)."""

from ..registry import BenchmarkSpec
from ..results import BaselineResult
from .benchmark import AUX_FIELD, KINEMATIC_TYPES, QOIS, TEST, TRAIN, VAL
from .card import CARD

__all__ = [
    "AUX_FIELD",
    "CARD",
    "KINEMATIC_TYPES",
    "QOIS",
    "SPEC",
    "TEST",
    "TRAIN",
    "VAL",
]

#: Official baseline results (ADR-0033). Transcribed from the ``mean`` block of
#: each run's held-out ``metrics-test.json`` at 4 significant figures; full
#: precision and per-case numbers stay in the run directory. DeformingPlate has
#: a single scored ``test`` split (``val`` only selects the checkpoint). All
#: three baselines are autoregressive (next-step rollout). MGN is the *blessed*
#: reference (ADR-0043); Transolver and GeoFLARE are provisional native
#: adaptations (ADR-0044/0045). Cross-method comparison is the v0.3 headline
#: (ADR-0041): on this smooth quasi-static task the operators outrun the
#: mesh-based reference ~3-4x on displacement relative L2.
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
        run_commit="0f103b5",
        run_date="2026-08-15",
        metrics={
            "test": {
                "rollout_rel_l2_disp": 0.8092,
                "rollout_rel_l2_aux": 4.209,
                "rollout_pos_rmse_mm": 11.25,
                "rollout_vm_rmse_mpa": 0.1168,
                "one_step_pos_rmse_mm": 0.05900,
                "one_step_vm_rmse_mpa": 0.008093,
                "qoi_peak_vm_mae_mpa": 2.940,
                "qoi_terminal_deflection_mae_mm": 47.39,
            },
        },
        checkpoint=(
            "models/deforming_plate/mgn-0f103b5/model-best-2200000.pt"
        ),
        checkpoint_sha256=(
            "89a1054af4dd6bab9aebe3fa0b508fac740203d0f43c579ebf52fccdc760133b"
        ),
        notes=(
            "Native MeshGraphNets (ADR-0042/0043): autoregressive next-step on "
            "the 3D tetrahedral mesh with world edges (activation-checkpointed, "
            "commit 0f103b5). BLESSED: its pooled test position RMSE, 11.25 mm "
            "(x10^-3 dataset-native length units), falls inside the published "
            "band 15.1 +/- 4.0 that four later papers corroborate (ADR-0043), so "
            "it reproduces the reference deformation result the field trusts - at "
            "the optimistic edge of the band. The 10M-step blessing budget was "
            "cut at the val-selected 2.2M checkpoint (model-best-2200000.pt), "
            "in-band and plateaued (ADR-0043 dated note). Position is strong, but "
            "the von Mises FIELD collapses under rollout: relative L2 4.21 (worse "
            "than a zero baseline on 96 of 100 cases) and rollout von Mises RMSE "
            "0.117 MPa, ~13x the operators - a pure autoregressive-accumulation "
            "artifact (one-step von Mises RMSE 0.0081 MPa is fine). The published "
            "DeformingPlate result gates on position only; stress is a StructBench "
            "secondary. Pooled relative L2 headline (ADR-0055) from the 2026-08-17 "
            "re-eval."
        ),
    ),
    BaselineResult(
        family="transolver",
        label="Transolver",
        scheme="autoregressive",
        reference=(
            "Wu, H., Luo, H., Wang, H., Wang, J., & Long, M. (2024). Transolver: "
            "A Fast Transformer Solver for PDEs on General Geometries. *ICML*. "
            "https://arxiv.org/abs/2402.02366"
        ),
        provisional=True,
        run_commit="84df162",
        run_date="2026-08-11",
        metrics={
            "test": {
                "rollout_rel_l2_disp": 0.2681,
                "rollout_rel_l2_aux": 0.2400,
                "rollout_pos_rmse_mm": 2.929,
                "rollout_vm_rmse_mpa": 0.009137,
                "one_step_pos_rmse_mm": 0.01412,
                "one_step_vm_rmse_mpa": 0.005993,
                "qoi_peak_vm_mae_mpa": 0.01915,
                "qoi_terminal_deflection_mae_mm": 6.172,
            },
        },
        notes=(
            "Native Transolver (Physics-Attention), provisional autoregressive "
            "adaptation on the DeformingPlate rollout (ADR-0044): a best-effort "
            "native implementation, not validated against a published "
            "Transolver-DeformingPlate number. Val-selected model-best-4000000.pt. "
            "It is the strongest baseline on this benchmark by a wide margin - "
            "~3x lower displacement relative L2 than the blessed MGN (0.268 vs "
            "0.809) and ~4x lower position RMSE (2.93 vs 11.25 mm) - because its "
            "global attention is not tied to a fixed mesh graph and it accumulates "
            "little rollout error on smooth quasi-static deformation (von Mises "
            "relative L2 0.240 vs MGN 4.21). Pooled relative L2 headline "
            "(ADR-0055) from the 2026-08-17 re-eval."
        ),
    ),
    BaselineResult(
        family="geoflare",
        label="GeoFLARE",
        scheme="autoregressive",
        reference=(
            "Adams, R., et al. (NVIDIA). GeoTransolver. arXiv:2512.20399; with "
            "Puri, R., et al. FLARE: Fast Low-rank Attention Routing Engine. "
            "arXiv:2508.12594. GeoFLARE is GeoTransolver with the FLARE attention "
            "backend (attention_type GALE_FA; ADR-0045)."
        ),
        provisional=True,
        run_commit="84df162",
        run_date="2026-08-14",
        metrics={
            "test": {
                "rollout_rel_l2_disp": 0.5729,
                "rollout_rel_l2_aux": 0.3513,
                "rollout_pos_rmse_mm": 4.876,
                "rollout_vm_rmse_mpa": 0.01360,
                "one_step_pos_rmse_mm": 0.02778,
                "one_step_vm_rmse_mpa": 0.01054,
                "qoi_peak_vm_mae_mpa": 0.03846,
                "qoi_terminal_deflection_mae_mm": 10.26,
            },
        },
        notes=(
            "Native GeoFLARE - GeoTransolver with the FLARE low-rank attention "
            "backend (attention_type GALE_FA) - provisional autoregressive "
            "adaptation on the DeformingPlate rollout (ADR-0045): a best-effort "
            "native implementation, not validated against a published number. "
            "Val-selected model-best-6600000.pt. It sits between Transolver and "
            "MGN: displacement relative L2 0.573 (vs Transolver 0.268, MGN 0.809) "
            "and position RMSE 4.88 mm - clearly beating the blessed reference but "
            "trailing the plain Physics-Attention operator on this task. Pooled "
            "relative L2 headline (ADR-0055) from the 2026-08-17 re-eval."
        ),
    ),
)

SPEC = BenchmarkSpec(
    card=CARD,
    results=RESULTS,
    splits={"train": tuple(TRAIN), "val": tuple(VAL), "test": tuple(TEST)},
    eval_splits=("val", "test"),
    aux_field=AUX_FIELD,
    qois=dict(QOIS),
    boundary_feature_fn=None,
    dataset_id="deforming_plate",
    kinematic_types=KINEMATIC_TYPES,
    # MGN is the blessed family (ADR-0043); _quickstart_family (ADR-0046) now
    # resolves it from the blessed result above, and this explicit default keeps
    # the quickstart pinned to configs/deforming_plate/mgn.toml regardless (the
    # base-class default "cgn" has no grouped config here).
    quickstart_family="mgn",
)
