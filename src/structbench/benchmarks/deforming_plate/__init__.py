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

#: Official baseline results (ADR-0033). Relative-L2 and QoI values are
#: transcribed from the ``mean`` block of each run's held-out
#: ``metrics-test.json`` at 4 significant figures; full precision and per-case
#: numbers stay in the run directory. DeformingPlate has a single scored
#: ``test`` split (``val`` only selects the checkpoint).
#:
#: SCHEME MATRIX (2026-08-21, maintainer-approved; extends the ADR-0046
#: one-entry-per-family convention — dated ADR note pending): this benchmark
#: tables one row per family x prediction scheme, because the scheme axis
#: (autoregressive vs time-conditioned, ADR-0054) is itself a v0.3 finding
#: worth reading off the leaderboard. All autoregressive rows are the
#: working-frame NOISE-FIXED refits (noise_std 3.0 mm, CORRECTIONS
#: 2026-08-17; runs deforming-*-n3) — the earlier pre-fix operator entries
#: (run 84df162) trained with the same ~1000x-too-weak noise as the pre-fix
#: MGN and were not comparable to the noise-fixed MGN they sat beside. MGN is
#: the *blessed* reference (ADR-0043); every operator row is a provisional
#: native adaptation (ADR-0044/0045/0054/0057). A CGN control run exists
#: (deforming-cgn-n3) but fails this mesh benchmark outright (displacement
#: relative L2 1.33, worse than predicting zero) and is not tabled
#: (maintainer decision, 2026-08-21).
#:
#: RMSE CONVENTION (ADR-0043; uniformly enforced 2026-08-21): DeformingPlate's
#: ``rollout_pos_rmse_mm`` / ``rollout_vm_rmse_mpa`` are the **pooled**
#: space+time RMSE (root of the mean squared error pooled over coordinates x
#: nodes x steps x trajectories) — the same statistic as the published
#: DeformingPlate number, computed by ``tools/blessing_pooled_rmse.py`` for
#: position and by the same pooling for von Mises with kinematic rows
#: excluded (they carry no aux prediction; ADR-0026 masking). This is
#: DELIBERATELY a different statistic from the mean-of-per-step-RMSE the
#: evaluator reports as ``rollout_position_rmse`` and that the other
#: benchmarks' leaderboards use (ADR-0019 SS5); conflating the two destroys
#: comparability with the paper (ADR-0043 SS8). The 2026-08-21 pass found the
#: earlier time-conditioned rows had been transcribed under the evaluator's
#: mean-of-per-step statistic in violation of this header (Transolver TC
#: 1.996 -> pooled 3.454; T++ 2.046 -> 3.322; vm analogously) and recomputed
#: every row under the pooled convention from the runs' saved rollouts (a
#: maintainer-local verification script; the MGN position value reproduced
#: exactly, its vm value moved <0.3%).
#: One-step and QoI columns are as the evaluator reports them.
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
                "rollout_vm_rmse_mpa": 0.01501,
                "one_step_pos_rmse_mm": 0.2593,
                "one_step_vm_rmse_mpa": 0.005785,
                "qoi_peak_vm_mae_mpa": 0.03978,
                "qoi_terminal_deflection_mae_mm": 48.42,
            },
        },
        checkpoint=("models/deforming_plate/mgn-eb39994/model-best-1750000.pt"),
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
            "4.21 -> 0.363 (-91%), pooled vm RMSE 0.169 -> 0.0150 MPa, and rollout "
            "displacement also improves (relative L2 0.809 -> 0.501, pooled position "
            "RMSE 16.98 -> 15.45 mm, still in-band, closer to the 15.1 centre). The "
            "classic noise-injection trade holds - one-step position RMSE rises "
            "(0.059 -> 0.259 mm) while rollout stability improves - so the noise is "
            "doing its job. Val-selected model-best-1750000.pt (retrain stopped at "
            "~1.75M steps; the mgn.toml default is now 1M for a fairer cross-method "
            "budget, whose in-band status at 1M is not yet verified). The published "
            "DeformingPlate result gates on position only and reports no stress "
            "number; stress is a StructBench secondary. Pooled relative L2 headline "
            "(ADR-0055) from the 2026-08-20 noise-fix eval. Pooled vm recomputed "
            "0.01505 -> 0.01501 under the uniform kinematic-excluded pooling "
            "(2026-08-21 convention pass, <0.3% move)."
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
        run_commit="eb39994",
        run_date="2026-08-18",
        metrics={
            "test": {
                "rollout_rel_l2_disp": 0.1437,
                "rollout_rel_l2_aux": 0.1777,
                "rollout_pos_rmse_mm": 3.018,
                "rollout_vm_rmse_mpa": 0.008139,
                "one_step_pos_rmse_mm": 0.2788,
                "one_step_vm_rmse_mpa": 0.005460,
                "qoi_peak_vm_mae_mpa": 0.02108,
                "qoi_terminal_deflection_mae_mm": 4.244,
            },
        },
        notes=(
            "Native autoregressive Transolver (ADR-0044) under the CORRECT "
            "working-frame training noise (noise_std 3.0 mm; run "
            "deforming-transolver-n3, 2M steps, val-selected "
            "model-best-1850000.pt, single seed). RESTORED to the table by the "
            "2026-08-21 scheme-matrix pass: the original AR entry (run 84df162, "
            "displacement relative L2 0.2681, pooled position RMSE 4.282 mm) "
            "was first replaced by the time-conditioned entry and then found to "
            "be a PRE-noise-fix run; the noise-fixed refit is much stronger "
            "(0.268 -> 0.144). Standing: statistically TIES the time-conditioned "
            "Transolver on the field metrics (displacement relative L2 0.1437 vs "
            "0.1538, von Mises 0.1777 vs 0.1993, pooled position RMSE 3.02 vs "
            "3.45 mm) at 8x the training budget (2M vs 250k), while the "
            "time-conditioned run wins the engineering QoIs decisively "
            "(terminal-deflection MAE 4.24 vs 0.56 mm, peak-vm 0.0211 vs 0.0139 "
            "MPa) - on this benchmark the scheme trade is QoIs-and-budget, not "
            "field accuracy. One-step position RMSE 0.279 mm carries the "
            "noise-injection trade like MGN's. PROVISIONAL (ADR-0044/0046). "
            "Pooled relative L2 headline (ADR-0055)."
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
                "rollout_pos_rmse_mm": 3.454,
                "rollout_vm_rmse_mpa": 0.008906,
                "qoi_peak_vm_mae_mpa": 0.01388,
                "qoi_terminal_deflection_mae_mm": 0.5615,
            },
        },
        notes=(
            "Native time-conditioned Transolver (ADR-0054): history-free, each "
            "frame predicted independently from the reference mesh + normalized "
            "query time, with no rollout accumulation, so one-step is N/A. "
            "CONVENTION REPAIR (2026-08-21): the previous entry transcribed the "
            "evaluator's mean-of-per-step statistics (1.996 mm / 0.007373 MPa) "
            "in violation of this module's pooled-RMSE header; the pooled "
            "values are 3.454 mm / 0.008906 MPa (relative-L2 and QoI values "
            "unchanged). Standing against the noise-fixed autoregressive "
            "Transolver above: the FIELD metrics tie (0.1538 vs 0.1437 "
            "displacement relative L2) - the earlier 'beats AR on every metric' "
            "reading compared against the pre-noise-fix AR run (0.268) - but "
            "time-conditioning wins the engineering QoIs decisively "
            "(terminal-deflection MAE 0.56 vs 4.24 mm, peak-vm 0.0139 vs 0.0211 "
            "MPa) at 1/8 the budget (250k vs 2M steps), because avoiding "
            "rollout accumulation protects exactly the late-horizon quantities "
            "the QoIs read. Against the blessed noise-fixed MGN it leads "
            "~3.3x on displacement relative L2 (0.154 vs 0.501) and ~4.5x on "
            "pooled position RMSE (3.45 vs 15.45 mm). Seed 1 of the s1-s2 pair, "
            "val-selected model-best-235000.pt (val relative L2 0.174 vs seed "
            "2's 0.189; seed 2 test displacement 0.1471), run "
            "deforming-transolver-tc-s1. PROVISIONAL (ADR-0044/0046): a "
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
                "rollout_pos_rmse_mm": 3.322,
                "rollout_vm_rmse_mpa": 0.009332,
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
            "pos/vm RMSE restated under the uniform pooled convention "
            "(2026-08-21; were 2.046 mm / 0.007728 MPa mean-of-per-step). "
            "Pooled relative L2 headline (ADR-0055)."
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
        run_commit="eb39994",
        run_date="2026-08-18",
        metrics={
            "test": {
                "rollout_rel_l2_disp": 0.3828,
                "rollout_rel_l2_aux": 0.2935,
                "rollout_pos_rmse_mm": 4.064,
                "rollout_vm_rmse_mpa": 0.01326,
                "one_step_pos_rmse_mm": 0.2086,
                "one_step_vm_rmse_mpa": 0.006014,
                "qoi_peak_vm_mae_mpa": 0.03595,
                "qoi_terminal_deflection_mae_mm": 6.196,
            },
        },
        notes=(
            "Native autoregressive GeoFLARE - GeoTransolver with the FLARE "
            "low-rank attention backend (attention_type GALE_FA; ADR-0045) - "
            "under the CORRECT working-frame training noise (noise_std 3.0 mm; "
            "run deforming-geoflare-n3, 2M steps, val-selected "
            "model-best-1900000.pt, single seed). REPLACES the pre-noise-fix "
            "entry (run 84df162, val-selected at 6.6M steps: displacement "
            "relative L2 0.5729, pooled position RMSE 5.144 mm) - that run "
            "trained with the same ~1000x-too-weak noise as the pre-fix MGN, so "
            "its numbers were not comparable to the noise-fixed MGN beside it; "
            "the fix improves GeoFLARE-AR sharply (0.573 -> 0.383). Standing: "
            "ahead of the blessed MGN on the fields (0.383 vs 0.501 "
            "displacement relative L2; pooled position 4.06 vs 15.45 mm) and on "
            "terminal deflection (6.20 vs 48.42 mm), behind both Transolver "
            "rows on everything. PROVISIONAL (ADR-0045/0046). Pooled relative "
            "L2 headline (ADR-0055)."
        ),
    ),
    BaselineResult(
        family="geoflare",
        label="GeoFLARE",
        scheme="time-conditioned",
        reference=(
            "Adams, R., et al. (NVIDIA). GeoTransolver. arXiv:2512.20399; with "
            "Puri, R., et al. FLARE: Fast Low-rank Attention Routing Engine. "
            "arXiv:2508.12594. GeoFLARE is GeoTransolver with the FLARE attention "
            "backend (attention_type GALE_FA; ADR-0045)."
        ),
        provisional=True,
        run_commit="7919060",
        run_date="2026-08-21",
        metrics={
            "test": {
                "rollout_rel_l2_disp": 0.2464,
                "rollout_rel_l2_aux": 0.2804,
                "rollout_pos_rmse_mm": 4.369,
                "rollout_vm_rmse_mpa": 0.01219,
                "qoi_peak_vm_mae_mpa": 0.04545,
                "qoi_terminal_deflection_mae_mm": 3.669,
            },
        },
        notes=(
            "Native GeoFLARE under the time-conditioned scheme (ADR-0054): "
            "history-free, each frame predicted independently from the "
            "reference mesh + normalized query time, no rollout accumulation, "
            "so one-step is N/A. The scheme effect first measured on Transolver "
            "replicates on a second operator family: against the noise-fixed "
            "autoregressive GeoFLARE above it wins displacement relative L2 "
            "(0.246 vs 0.383) and terminal deflection (3.67 vs 6.20 mm) at 1/8 "
            "the budget (250k vs 2M steps), while the AR row edges it on pooled "
            "position RMSE (4.06 vs 4.37 mm) and peak-vm (0.0360 vs 0.0455 MPa) "
            "- a milder version of the Transolver scheme trade. It stays behind "
            "both Transolver rows on every metric at ~1.5x the time-conditioned "
            "Transolver's measured wall-clock (7.1 vs 4.7 h A100 at the same "
            "250k budget) - the persistent geometry cross-attention buys "
            "nothing over plain Physics-Attention here, now shown "
            "scheme-matched. Seed 1 of the s1-s2 pair, val-selected "
            "model-best-225000.pt (val relative L2 0.2546 vs seed 2's 0.2945; "
            "seed 2 test displacement 0.2767), run deforming-geoflare-tc-s1 "
            "(branch feat/geoflare-tc). PROVISIONAL (ADR-0045/0046): a "
            "best-effort native implementation, not validated against a "
            "published number. Pooled relative L2 headline (ADR-0055)."
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
