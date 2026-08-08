"""Benchmark card for the deforming_plate benchmark (ADR-0027, ADR-0043)."""

from ..card import BenchmarkCard
from .benchmark import AUX_FIELD, QOIS, TEST, TRAIN, VAL

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
)
