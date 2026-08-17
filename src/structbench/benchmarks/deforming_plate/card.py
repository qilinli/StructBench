"""Benchmark card for the deforming_plate benchmark (ADR-0027, ADR-0043)."""

from ..card import BenchmarkCard, BenchmarkFigure
from .benchmark import AUX_FIELD, QOIS, TEST, TRAIN, VAL

# Landing-page narrative (ADR-0036). Non-derivable prose; the structured facts,
# splits and baseline numbers are rendered from the card + results registry.
_OVERVIEW = """\
## The problem

A rigid actuator presses into a soft hyperelastic plate, indenting it and
wrapping the sheet around the tool while a stress field builds through the
material. **DeformingPlate** is the 3D quasi-static benchmark from
MeshGraphNets (Pfaff et al. 2021): a tetrahedral plate held along one edge
(HANDLE nodes) and pushed by a scripted rigid obstacle, solved to static
equilibrium at each load step in COMSOL. It is StructBench's first 3D
benchmark and its first with an irregular, per-case mesh (672-2189 nodes) - a
test of whether a surrogate can cope with genuine 3D geometry, contact with a
moving tool, and a ragged node count, not just a fixed 2D lattice.

The task is an **autoregressive load-stepping rollout**: from a two-frame
prefix the model advances the mesh one pseudo-time step at a time to the end
of the trajectory (400 steps), predicting both the nodal displacement and the
per-node von Mises stress. Time here is a load-step index, not milliseconds -
the source solve is quasi-static (dt = 0 in the data).

## Cross-method comparison

DeformingPlate is where StructBench's headline is **method against method**.
MeshGraphNets is the *blessed* reference: its rollout reproduces the published
number (pooled position RMSE inside the reported band, ADR-0043), anchoring the
benchmark to a result the field already trusts. Against it run two provisional
native operators - **Transolver** (Physics-Attention) and **GeoFLARE**
(geometry-aware attention with a low-rank routing engine) - both adapted here
to autoregressive rollout (ADR-0044/0045). Everything is scored in physical
units - displacement RMSE in mm, the von Mises field in MPa - with two
quantities of interest reading the engineering outcome: peak von Mises stress
and terminal peak deflection. On this smooth, quasi-static task the operators'
freedom from a fixed mesh graph tells: both outrun the mesh-based reference on
displacement (Transolver by ~3x on relative L2), while MGN's stress field
degrades under rollout. The leaderboard is below."""

_FIGURES = (
    BenchmarkFigure(
        path="assets/deforming_plate_rollout_methods.gif",
        caption=(
            "Ground truth vs the three baselines (MGN, Transolver, GeoFLARE) on "
            "test_0028, the deformable plate coloured by von Mises stress over "
            "the press-release rollout — the rigid actuator indents the sheet to "
            "~65 mm and then retracts. Transolver tracks the stress "
            "concentration and the bowl geometry most closely; GeoFLARE captures "
            "the shape but diffuses the stress; MGN's von Mises saturates far "
            "above ground truth and its mesh distorts. Transolver and GeoFLARE "
            "are provisional native baselines (ADR-0044/0045)."
        ),
        alt=(
            "Four-panel animation: ground truth and MGN, Transolver, GeoFLARE "
            "predictions of the deforming plate's von Mises stress."
        ),
    ),
    BenchmarkFigure(
        path="assets/deforming_plate_vm_methods.png",
        caption=(
            "von Mises stress at peak deflection (test_0028, step 360/400): "
            "ground truth vs MGN, Transolver, GeoFLARE. Transolver reproduces "
            "the fold-line stress concentration most faithfully (rollout "
            "displacement relative L2 0.27, von Mises 0.24); GeoFLARE "
            "under-concentrates the stress (0.57 / 0.35); MGN over-predicts the "
            "field almost everywhere — von Mises relative L2 4.21, worse than a "
            "zero baseline on 96 of 100 cases — and tears the mesh, even though "
            "its pooled position error stays inside the blessed band. This is "
            "the ranking the leaderboard reports."
        ),
        alt=(
            "Grid of von Mises snapshots: ground truth and MGN, Transolver, "
            "GeoFLARE predictions of the deforming plate at peak deflection."
        ),
    ),
)

CARD = BenchmarkCard(
    name="DeformingPlate",
    version="0.1",
    description=(
        "Quasi-static deformation of a hyperelastic 3D plate pressed by a "
        "scripted rigid actuator; the MeshGraphNets deforming_plate dataset "
        "(Pfaff et al. 2021) under the ADR-0043 rollout protocol."
    ),
    provenance=(
        "MeshGraphNets dataset (Pfaff et al., ICLR 2021; COMSOL ground "
        "truth), downloaded from the DeepMind source bucket and converted "
        "locally to canonical HDF5 (ADR-0042; not redistributed)."
    ),
    data_license=(
        "None stated by the source; downloaded from source, not "
        "redistributed (ADR-0042)"
    ),
    solver="COMSOL",
    discretisation="FEM",
    materials=("Hyperelastic (constants not published with the dataset)",),
    erosion=False,
    loading=("Scripted rigid actuator (OBSTACLE nodes, kinematic); HANDLE nodes fixed"),
    source_units="kg-m-s (SI; measured 2026-08-08, ADR-0042 §2b)",
    geometry=(
        "3D tetrahedral mesh, ~0.5 m plate + rigid actuator; 672-2189 nodes "
        "per case (mean ~1270, ragged)"
    ),
    n_cases=len(TRAIN) + len(VAL) + len(TEST),
    splits={"train": len(TRAIN), "val": len(VAL), "test": len(TEST)},
    task="quasi-static load-stepping autoregressive rollout (ADR-0043)",
    aux_field=AUX_FIELD,
    aux_unit="MPa",
    qois=tuple(QOIS),
    fields=("node/displacement", "node/von_mises_stress"),
    particles_per_case="672-2189",
    n_frames=400,
    output_dt_ms=1.0,
    input_frames=2,
    horizon="full",
    protocol_rationale=(
        "input_frames=2 is the floor (a velocity needs two frames) and the "
        "faithful value: the source model uses h=0 history — node inputs are "
        "the one-hot node type only — so no window tuning question exists "
        "and no ground-truth timeline analysis can move it (ADR-0043 §3). "
        "Pseudo-time: dt=0 in the source (quasi-static); time is the frame "
        "index and output_dt_ms=1.0 is nominal, not milliseconds. Units "
        "MEASURED on the full dataset (2026-08-08, ADR-0042 §2b): positions "
        "are metres, stress is Pa — so aux_unit MPa holds and kg-m-s is the "
        "identity source convention. Two measured caveats (ADR-0043 dated "
        "note): the hosted train.tfrecord carries 1,200 trajectories — the "
        "protocol's 1,000 are the first 1,000 in file order; and HANDLE "
        "(type-3) nodes are not strictly stationary in the data (max drift "
        "~0.02 m train / ~0.06 m valid+test) — both kinematic types are "
        "GT-prescribed and excluded from scoring either way. Scored span is "
        "[2, 400), exclusive end (ADR-0043 §6)."
    ),
    size_gb=6.1,
    overview=_OVERVIEW,
    figures=_FIGURES,
)
