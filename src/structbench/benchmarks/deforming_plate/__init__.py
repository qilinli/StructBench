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
#:
#: RMSE CONVENTION (ADR-0043): DeformingPlate's ``rollout_pos_rmse_mm`` /
#: ``rollout_vm_rmse_mpa`` are the **pooled** space+time RMSE (root of the mean
#: squared error pooled over coordinates x nodes x steps x trajectories) - the
#: same statistic as the published DeformingPlate number, computed by
#: ``tools/blessing_pooled_rmse.py`` for position. This is DELIBERATELY a
#: different statistic from the mean-of-per-step-RMSE the evaluator reports as
#: ``rollout_position_rmse`` and that the other benchmarks' leaderboards use
#: (ADR-0019 SS5); conflating the two destroys comparability with the paper
#: (ADR-0043 SS8), so DP pins the pooled convention to keep the MGN blessing
#: gate honest. One-step and QoI columns are as the evaluator reports them.
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
        run_commit="eb39994",
        run_date="2026-08-17",
        metrics={
            "test": {
                "rollout_rel_l2_disp": 0.5013,
                "rollout_rel_l2_aux": 0.3630,
                "rollout_pos_rmse_mm": 15.45,
                "rollout_vm_rmse_mpa": 0.01505,
                "one_step_pos_rmse_mm": 0.2593,
                "one_step_vm_rmse_mpa": 0.005785,
                "qoi_peak_vm_mae_mpa": 0.03978,
                "qoi_terminal_deflection_mae_mm": 48.42,
            },
        },
        checkpoint=(
            "models/deforming_plate/mgn-eb39994/model-best-1750000.pt"
        ),
        checkpoint_sha256=(
            "5309491a0595e90678eed3bf0a8063e76754d6f52a94228c7c58fb1bc700bbf7"
        ),
        notes=(
            "Native MeshGraphNets (ADR-0042/0043): autoregressive next-step on "
            "the 3D tetrahedral mesh with world edges (activation-checkpointed). "
            "BLESSED: its POOLED test position RMSE, 15.45 mm (x10^-3 dataset-native "
            "length units; ADR-0043 paper convention, from "
            "tools/blessing_pooled_rmse.py), falls inside the published band "
            "15.1 +/- 4.0 = [11.1, 19.1] that four later papers corroborate "
            "(ADR-0043), reproducing the reference deformation result the field "
            "trusts. The pooled RMSE is dominated by a handful of divergent "
            "trajectories (per-case mean 7.38 mm) - the honest whole-dataset "
            "statistic. NOISE-FIX RETRAIN (2026-08-20): the prior blessing trained "
            "with noise_std 3e-3 in the WRONG frame (dataset-native x10^-3), ~1000x "
            "too weak - effectively noise-free, which let rollout mesh distortion "
            "run away and COLLAPSE the von Mises field (relative L2 4.21, worse than "
            "a zero baseline on 96/100 cases; pooled vm RMSE 0.169 MPa). Correcting "
            "the training noise to the working frame (noise_std 3.0 mm = the paper's "
            "3e-3 m; CORRECTIONS 2026-08-17) REPAIRS it: von Mises relative L2 "
            "4.21 -> 0.363 (-91%), pooled vm RMSE 0.169 -> 0.0151 MPa, and rollout "
            "displacement also improves (relative L2 0.809 -> 0.501, pooled position "
            "RMSE 16.98 -> 15.45 mm, still in-band, closer to the 15.1 centre). The "
            "classic noise-injection trade holds - one-step position RMSE rises "
            "(0.059 -> 0.259 mm) while rollout stability improves - so the noise is "
            "doing its job. Val-selected model-best-1750000.pt (retrain stopped at "
            "~1.75M steps; the mgn.toml default is now 1M for a fairer cross-method "
            "budget, whose in-band status at 1M is not yet verified). The published "
            "DeformingPlate result gates on position only and reports no stress "
            "number; stress is a StructBench secondary. Pooled relative L2 headline "
            "(ADR-0055) from the 2026-08-20 noise-fix eval."
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
        run_commit="5b84119",
        run_date="2026-08-18",
        metrics={
            "test": {
                "rollout_rel_l2_disp": 0.1538,
                "rollout_rel_l2_aux": 0.1993,
                "rollout_pos_rmse_mm": 1.996,
                "rollout_vm_rmse_mpa": 0.007373,
                "qoi_peak_vm_mae_mpa": 0.01388,
                "qoi_terminal_deflection_mae_mm": 0.5615,
            },
        },
        notes=(
            "Native time-conditioned Transolver (ADR-0054): history-free, each "
            "frame predicted independently from the reference mesh + normalized "
            "query time, with no rollout accumulation, so one-step is N/A. This "
            "REPLACES the earlier provisional autoregressive DP Transolver entry "
            "(2M steps, run 84df162): time-conditioned is the faithful native "
            "Transolver scheme (ADR-0054, matching Taylor/notch) and is also "
            "decisively better here at a fraction of the budget - at 250k steps "
            "it beats the 2M autoregressive run on every metric (displacement "
            "relative L2 0.154 vs 0.268, pooled position RMSE 2.00 vs 4.28 mm, "
            "terminal-deflection MAE 0.56 vs 6.17 mm), because avoiding rollout "
            "accumulation dominates on this smooth quasi-static deformation. It "
            "is now the strongest baseline on this benchmark by a wide margin - "
            "~3.3x lower displacement relative L2 than the blessed (noise-fixed) "
            "MGN (0.154 vs 0.501), ~7.7x lower pooled position RMSE (2.00 vs 15.45 "
            "mm) - and tracks the von Mises field better (relative L2 0.199 vs MGN "
            "0.363; the MGN noise fix closed the earlier huge stress gap, was vs "
            "4.21). Seed 1 of the s1-s2 pair, val-selected model-best-235000.pt, "
            "run deforming-transolver-tc-s1. PROVISIONAL (ADR-0044/0046): a "
            "best-effort native implementation, not validated against a "
            "published Transolver-DeformingPlate number. Pooled relative L2 "
            "headline (ADR-0055)."
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
            "test": {
                "rollout_rel_l2_disp": 0.158,
                "rollout_rel_l2_aux": 0.2054,
                "rollout_pos_rmse_mm": 2.046,
                "rollout_vm_rmse_mpa": 0.007728,
                "qoi_peak_vm_mae_mpa": 0.01895,
                "qoi_terminal_deflection_mae_mm": 0.8748,
            },
        },
        notes=(
            "Transolver++ (ADR-0057): the native time-conditioned Transolver "
            "with both eidetic-state knobs ON - per-point adaptive slice "
            "temperature + train-only Gumbel Rep-Slice reparameterisation. Seed "
            "1 of the s1-s2 pair (val-selected model-best-215000.pt, run "
            "deforming-transolver-tcpp-s1), seed-matched to the plain "
            "time-conditioned Transolver entry above (also seed 1). PROVISIONAL "
            "method comparison (ADR-0046). It is slightly WORSE than the plain "
            "time-conditioned Transolver on every metric (displacement relative "
            "L2 0.158 vs 0.154, von Mises 0.205 vs 0.199, terminal-deflection "
            "MAE 0.87 vs 0.56 mm) - the same neutral-to-worse result seen on "
            "Taylor and notch-impact, consistent with Transolver++'s "
            "eidetic-state edits being designed for million-scale geometries "
            "rather than this benchmark's small tetrahedral meshes. Pooled "
            "relative L2 headline (ADR-0055)."
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
                "rollout_pos_rmse_mm": 5.144,
                "rollout_vm_rmse_mpa": 0.01493,
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
            "Val-selected model-best-6600000.pt. This GeoFLARE run is "
            "autoregressive. Its standing is now METRIC-DEPENDENT against the "
            "noise-fixed MGN: on pooled position RMSE it still beats MGN (5.14 vs "
            "15.45 mm, MGN's pooled RMSE inflated by a few divergent trajectories), "
            "but on per-case-mean displacement relative L2 it now TRAILS MGN "
            "(0.573 vs 0.501; before the MGN noise fix it led, 0.573 vs 0.809). It "
            "stays behind the time-conditioned Transolver (0.154) throughout. "
            "Pooled relative L2 headline (ADR-0055)."
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
