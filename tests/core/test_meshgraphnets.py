from __future__ import annotations

import pytest

from structbench.core.io.meshgraphnets import parse_meta

_DEFORMING_PLATE_META = {
    "simulator": "comsol",
    "dt": 0,
    "trajectory_length": 400,
    "field_names": ["cells", "node_type", "mesh_pos", "world_pos", "stress"],
    "features": {
        "cells": {"type": "static", "dtype": "int32", "shape": [1, -1, 4]},
        "node_type": {"type": "static", "dtype": "int32", "shape": [1, -1, 1]},
        "mesh_pos": {"type": "static", "dtype": "float32", "shape": [1, -1, 3]},
        "world_pos": {"type": "dynamic", "dtype": "float32", "shape": [400, -1, 3]},
        "stress": {"type": "dynamic", "dtype": "float32", "shape": [400, -1, 1]},
    },
}


def test_parse_meta_reads_all_fields():
    specs = parse_meta(_DEFORMING_PLATE_META)
    assert set(specs) == set(_DEFORMING_PLATE_META["field_names"])
    assert specs["cells"].ftype == "static" and specs["cells"].shape == (1, -1, 4)
    assert specs["world_pos"].ftype == "dynamic"
    assert specs["world_pos"].dtype == "float32"


def test_parse_meta_rejects_unknown_type():
    bad = {
        "field_names": ["x"],
        "trajectory_length": 1,
        "features": {"x": {"type": "bogus", "dtype": "float32", "shape": [1]}},
    }
    with pytest.raises(ValueError, match="bogus"):
        parse_meta(bad)
