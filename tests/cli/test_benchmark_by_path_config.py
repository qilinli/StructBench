"""A run config may name a benchmark definition directory (ADR-0032 note,
2026-11-06): the card's protocol is enforced from it, the config carries its
absolute path, and the run record names the definition and its hashes."""

import textwrap
from dataclasses import replace

import pytest

from structbench.benchmarks import get_benchmark
from structbench.config import (
    ConfigError,
    TrainConfig,
    load_run_config,
    read_run_record,
    resolved_config_dict,
)

DEFINITION = textwrap.dedent(
    """
    from structbench.benchmarks import BenchmarkSpec
    from structbench.benchmarks.card import BenchmarkCard

    SPLITS = {"train": ("c-0", "c-1"), "val": ("c-2",)}
    SPEC = BenchmarkSpec(
        card=BenchmarkCard(
            name="ConfigByPath",
            version="0.0",
            description="fixture",
            provenance="synthetic",
            data_license="n/a",
            solver="none",
            discretisation="FEM",
            materials=("synthetic",),
            erosion=False,
            loading="none",
            source_units="kg-m-s",
            geometry="synthetic",
            n_cases=3,
            splits={k: len(v) for k, v in SPLITS.items()},
            task="fixture",
            aux_field="von_mises_stress",
            aux_unit="MPa",
            qois=(),
            fields=("node/displacement",),
            particles_per_case="4",
            n_frames=8,
            input_frames=INPUT_FRAMES,
            output_dt_ms=1.0,
            protocol_rationale="fixture",
        ),
        splits=SPLITS,
        eval_splits=("val",),
        aux_field="von_mises_stress",
        dataset_id="config-by-path",
    )
    """
)

CONFIG = """\
[run]
benchmark = "{benchmark}"
seed = 7

[model]
family = "cgn"
input_frames = 6
connectivity_radius = 1.5
hidden_dim = 64
message_passing_steps = 5
nmlp_layers = 1
particle_type_embedding_size = 9
noise_std = 0.02
dim = 2
max_neighbors = 48
aux_transform = "none"
aux_transform_scale = 0.01

[train]
batch_size = 8
lr_init = 1e-4
lr_decay = 0.1
training_steps = 100
val_every = 50
w_pos = 1.0
w_aux = 1.0
aux_tail_weight = 0.0
train_frames = 0
"""


def _definition(tmp_path, input_frames=6):
    d = tmp_path / "defn"
    d.mkdir()
    (d / "benchmark.py").write_text(
        f"INPUT_FRAMES = {input_frames}\n" + DEFINITION, encoding="utf-8"
    )
    return d


def _config(tmp_path, benchmark):
    p = tmp_path / "run.toml"
    p.write_text(
        CONFIG.format(benchmark=benchmark.replace("\\", "/")), encoding="utf-8"
    )
    return p


def test_config_names_the_definition_by_absolute_path(tmp_path, monkeypatch):
    d = _definition(tmp_path)
    monkeypatch.chdir(tmp_path)
    rc = load_run_config(_config(tmp_path, "defn"))  # relative to the cwd
    assert rc.train.benchmark == str(d.resolve())
    spec = get_benchmark(rc.train.benchmark)
    assert spec.card.name == "ConfigByPath"
    assert spec.source is not None and spec.source.path == d.resolve()


def test_protocol_is_enforced_from_the_definition_card(tmp_path):
    d = _definition(tmp_path, input_frames=4)
    with pytest.raises(ConfigError, match="must equal benchmark"):
        load_run_config(_config(tmp_path, str(d)))


def test_run_record_carries_the_definition(tmp_path):
    d = _definition(tmp_path)
    rc = load_run_config(_config(tmp_path, str(d)))
    spec = get_benchmark(rc.train.benchmark)
    assert spec.source is not None
    record = resolved_config_dict(
        "cgn",
        rc.model,
        rc.train,
        n_particle_types=1,
        data_root=tmp_path,
        definition=spec.source.to_json_dict(),
    )
    assert record["run"]["benchmark"] == str(d.resolve())
    assert record["run"]["definition"]["path"] == str(d.resolve())
    assert set(record["run"]["definition"]["files"]) == {"benchmark.py"}
    # A registered benchmark writes no definition key.
    registered = resolved_config_dict(
        "cgn",
        rc.model,
        replace(rc.train, benchmark="taylor_impact_2d"),
        n_particle_types=1,
        data_root=tmp_path,
    )
    assert "definition" not in registered["run"]
    # The record round-trips through the reader unchanged.
    out = tmp_path / "config.json"
    import json

    out.write_text(json.dumps(record), encoding="utf-8")
    assert read_run_record(out)["run"]["definition"] == record["run"]["definition"]


def test_unresolvable_name_still_loads_without_protocol_checks(tmp_path):
    # Programmatic configs around unregistered specs keep working (the smoke
    # tests' pattern): no registry name, no directory, no checks.
    rc = load_run_config(_config(tmp_path, "NotARegisteredBenchmark"))
    assert isinstance(rc.train, TrainConfig)
    assert rc.train.benchmark == "NotARegisteredBenchmark"
