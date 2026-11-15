"""The isotropic paths are byte-identical across the combined-hardening work.

Every digest below was computed by this file on the commit before combined
hardening was added (``main`` at ``4f06dd9``) and pinned. The test uses only
what existed then, so it runs unchanged on both sides: the decks of the first
shipped example, the isotropic deck writers, the canonical file written from an
isotropic export, the registered channels and benchmark declarations, the
isotropic law's description and the constraint properties under it.
"""

from __future__ import annotations

import hashlib
import json
from importlib import resources

import h5py
import numpy as np
import pytest

pytest.importorskip("scipy")

from structbench.benchmarks import (  # noqa: E402
    ConstitutiveLaw,
    available_benchmarks,
    get_benchmark,
)
from structbench.cli.train import _state_constraint_properties  # noqa: E402
from structbench.core.io import write_case  # noqa: E402
from structbench.core.io.abaqus import (  # noqa: E402
    ABAQUS_NPZ_FORMAT,
    abaqus_export_to_case,
)
from structbench.datagen import definition, sampling  # noqa: E402
from structbench.datagen.abaqus import deck  # noqa: E402
from structbench.datasets.canonical import aux_channel_labels  # noqa: E402

PINNED = {
    "example_decks": "1dd7b3cbcc4fddefc9956816e136b7c6207fd6a9ab24145927acadf0511d12f5",
    "deck_writers": "7aa4875470e6407e6c81c9e77717ac452db5ea38269611825e852577485a3633",
    "canonical": "86263d1c8c1b9faf5ef39ac859cdaad0a2f655056a1889c12f1ef8f65b5f5ece",
    "declarations": "5e9ded3d9c087bae754d99deaf955ae7ebdccacac876a1a406775e647897bbf7",
    "law": "8a08d294f3eb6a017d678b807c2687a16a07522c005e8d86108c32a23d5ec28b",
    "properties": "851dece7191db30d809dd8d1fe67e653c91ed8feb8387ec1c169b67526980c51",
}

_DECK = """*HEADING
toy
*NODE
1, 0.0, 0.0
2, 1.0, 0.0
3, 0.0, 1.0
4, 1.0, 1.0
*ELEMENT, TYPE=CAX4R, ELSET=E
1, 1, 2, 4, 3
*NSET, NSET=ALLN
1, 2, 3, 4
*MATERIAL, NAME=M
*DENSITY
7e-09
*ELASTIC
200000.0, 0.3
*PLASTIC
250.0, 0.0
1250.0, 10.0
*SECTION CONTROLS, NAME=HG_ENHANCED, HOURGLASS=ENHANCED
*SOLID SECTION, ELSET=E, MATERIAL=M, CONTROLS=HG_ENHANCED
*INITIAL CONDITIONS, TYPE=VELOCITY
ALLN, 2, -30000.0
*INITIAL CONDITIONS, TYPE=HARDENING
1, 0.15
*STEP, NAME=S, NLGEOM=YES
*DYNAMIC, EXPLICIT
, 0.001
*END STEP
"""
_INST = "PART-1-1"
_TERMS = ("ALLAE", "ALLCD", "ALLFD", "ALLIE", "ALLKE", "ALLPD", "ALLSE", "ALLVD")
_TERMS += ("ALLWK", "ETOTAL")
_OLD_CHANNELS = (
    "von_mises_stress",
    "axial_stress",
    "damage",
    "max_principal_strain",
    "deviatoric_stress_2d",
    "effective_plastic_strain",
    "internal_energy",
    "density",
    "stress_axisymmetric",
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _isotropic_export() -> dict[str, np.ndarray]:
    """An isotropic export in the ``abaqus-npz/1`` layout: S and PEEQ only."""
    t = np.array([0.0, 0.0005, 0.001])
    rng = np.random.default_rng(7)
    manifest = {
        "format": ABAQUS_NPZ_FORMAT,
        "odb_sha256": "0" * 64,
        "abaqus_release": "Abaqus/Explicit 2025",
        "precision": "DOUBLE_PRECISION",
        "materials": ["M"],
        "sections": ["Section-E"],
        "fields": {},
        "skipped": [],
    }
    out = {
        "manifest": np.array(json.dumps(manifest)),
        f"mesh/{_INST}/node_labels": np.array([1, 2, 3, 4]),
        f"mesh/{_INST}/node_coords": np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float
        ),
        f"mesh/{_INST}/elements/CAX4R/labels": np.array([1]),
        f"mesh/{_INST}/elements/CAX4R/connectivity": np.array([[1, 2, 4, 3]]),
        "step/S/frame_times": t,
    }
    for name in ("U", "V", "A"):
        out[f"field/S/{name}/{_INST}/data"] = rng.normal(size=(3, 4, 2)).astype("f4")
        out[f"field/S/{name}/{_INST}/node_labels"] = np.array([1, 2, 3, 4])
    for name, width in (("S", 4), ("PEEQ", 1)):
        out[f"field/S/{name}/{_INST}/data"] = rng.normal(size=(3, 1, width)).astype(
            "f4"
        )
        out[f"field/S/{name}/{_INST}/element_labels"] = np.array([1])
        out[f"field/S/{name}/{_INST}/integration_points"] = np.array([1])
    for k, term in enumerate(_TERMS):
        values = np.array([0.0, 1.0 + k, 2.0 + k])
        out[f"history/S/Assembly Assembly-1/{term}"] = np.stack([t, values], -1)
    return out


def _h5_digest(path) -> str:
    """Every group, dataset and attribute of an HDF5 file, in name order."""
    digest = hashlib.sha256()

    def attrs(obj) -> None:
        for key in sorted(obj.attrs):
            digest.update(f"@{key}={obj.attrs[key]!r}".encode())

    def visit(name: str, obj) -> None:
        digest.update(name.encode())
        attrs(obj)
        if isinstance(obj, h5py.Dataset):
            data = obj[()]
            digest.update(f"{obj.dtype}{obj.shape}".encode())
            # Numeric arrays by their bytes; strings by value (an object array's
            # bytes are addresses, which differ from run to run).
            numeric = isinstance(data, np.ndarray) and data.dtype.kind in "biuf"
            digest.update(
                data.tobytes()
                if numeric
                else repr(
                    data.tolist() if isinstance(data, np.ndarray) else data
                ).encode()
            )

    with h5py.File(path, "r") as handle:
        attrs(handle)
        handle.visititems(visit)
    return digest.hexdigest()


def _measure(tmp_path) -> dict[str, str]:
    example = resources.files("structbench.datagen") / "examples" / "abaqus_conformance"
    with resources.as_file(example) as folder:
        d, problem = definition.load_definition(folder), definition.load_problem(folder)
        decks = [
            problem.input_deck(
                {**d.fixed, **p.params, d.levels.refine_key: d.levels.production}, None
            )
            for p in sampling.sample_split(d.variables, d.regions, d.split("pilot"))
        ]
    writers = deck.material(
        "M", density=7.0e-9, youngs=2.0e5, poisson=0.3, plastic=[(250.0, 0.0)]
    ) + deck.initial_hardening([1, 2, 17], np.array([0.0, 0.15, 1.0e-3]))

    npz = tmp_path / "iso.npz"
    np.savez(npz, **_isotropic_export())
    case = abaqus_export_to_case(
        npz, _DECK, source_units="t-mm-s", dimension=2, case_id="iso"
    )
    assert case.response is not None
    assert sorted(case.response.element["solid"]) == [
        "effective_plastic_strain",
        "stress",
    ]
    write_case(case, tmp_path / "iso.h5")

    declarations: list[tuple] = [
        (name, aux_channel_labels(name)) for name in _OLD_CHANNELS
    ]
    for name in sorted(available_benchmarks()):
        spec = get_benchmark(name)
        declarations.append(
            (
                name,
                spec.state_site,
                tuple(spec.state_channels),
                spec.constitutive_law is None,
                spec.defined_properties,
            )
        )

    law = ConstitutiveLaw(124000.0, 0.34, ((0.0, 10.0), (200.0, 5200.0)))
    rng = np.random.default_rng(3)
    state = np.concatenate(
        [rng.normal(0, 150, (12, 9, 4)), np.abs(rng.normal(0, 0.02, (12, 9, 1)))], -1
    )
    labels = ("s_rr", "s_zz", "s_tt", "s_rz", "effective_plastic_strain")
    properties = _state_constraint_properties(
        state, labels, law, {"yield": 0.01, "peeq": 1e-4}, 2, None
    )
    return {
        "example_decks": _sha("\x00".join(decks)),
        "deck_writers": _sha(writers),
        "canonical": _h5_digest(tmp_path / "iso.h5"),
        "declarations": _sha(repr(declarations)),
        "law": _sha(repr(law) + repr((law.shear_modulus, law.bulk_modulus))),
        "properties": _sha(repr(sorted(properties.items()))),
    }


def test_isotropic_paths_are_byte_identical(tmp_path):
    assert _measure(tmp_path) == PINNED
