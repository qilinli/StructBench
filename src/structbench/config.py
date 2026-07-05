"""Grouped run configuration: typed sections, strict loading (ADR-0032).

A run config is a TOML file with sections mirroring ownership::

    [run]       benchmark, seed                     (orchestration)
    [model]     family + every field of the family  (architecture)
    [train]     every schedule field                (optimization)
    [protocol]  optional research override          (ADR-0032 §4)

Loading is strict: unknown sections or keys are errors, and ``[model]`` /
``[train]`` must be complete — a missing key is an error, never a silent
fallback to a dataclass default. The dataclass defaults below exist only for
programmatic construction (tests, notebooks); TOML-driven runs state every
value explicitly.

Benchmark *protocol* (``init_frames``, horizon, eval times) is not run
configuration: it lives on the benchmark card, pinned per ADR-0032 §4. A
``[protocol]`` section here overrides it for research runs only, and the run
records ``protocol.standard = false``.
"""

from __future__ import annotations

import json
import subprocess
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class GNSConfig:
    """Architecture and noise hyperparameters of the learned simulator.

    Attributes
    ----------
    window : int
        Number of consecutive input frames per sample (history length).
        A model-family choice, not benchmark protocol: at rollout it is
        warm-started from the protocol's ``init_frames`` observed frames
        (ADR-0032 §4).
    connectivity_radius : float
        Graph connectivity radius in the mm working frame.
    hidden_dim : int
        Latent/MLP hidden width.
    message_passing_steps : int
        Number of interaction-network message-passing steps.
    nmlp_layers : int
        Number of hidden layers in each MLP.
    particle_type_embedding_size : int
        Embedding width for the particle-type lookup (used only when more than
        one particle type is present).
    noise_std : float
        Standard deviation of the random-walk training noise at the last step.
    dim : int
        Spatial dimensionality (2 for the Taylor 2D benchmark).
    max_neighbors : int
        Per-node neighbour cap for the radius graph. Size it above the true
        maximum degree at ``connectivity_radius`` so it never binds on
        physical configurations (ADR-0028).
    """

    window: int = 11
    connectivity_radius: float = 1.5
    hidden_dim: int = 64
    message_passing_steps: int = 5
    nmlp_layers: int = 1
    particle_type_embedding_size: int = 9
    noise_std: float = 0.02
    dim: int = 2
    max_neighbors: int = 48


@dataclass
class TrainConfig:
    """Optimization schedule and loss weights for training.

    Attributes
    ----------
    benchmark : str
        Registry name resolved via :func:`structbench.benchmarks.get_benchmark`.
        Sourced from the ``[run]`` section of a grouped config.
    batch_size : int
        Number of trajectory windows per optimizer step.
    lr_init : float
        Initial Adam learning rate.
    lr_decay : float
        Multiplicative decay base for the exponential schedule.
    lr_decay_steps : int
        Step interval over which ``lr_decay`` is applied once.
    training_steps : int
        Total number of optimizer steps.
    val_every : int
        Validation/checkpoint interval in steps.
    w_pos : float
        Weight on the acceleration (position) loss term.
    w_aux : float
        Weight on the auxiliary loss term.
    seed : int
        Torch RNG seed set at the start of training; fixes weight
        initialization, training-noise draws, and shuffle order. Sourced from
        the ``[run]`` section of a grouped config. Bitwise GPU reproducibility
        would additionally require deterministic kernels (scatter-add is
        nondeterministic on CUDA), which are not enabled.
    """

    benchmark: str = "taylor_impact_2d"
    batch_size: int = 32
    lr_init: float = 1e-3
    lr_decay: float = 0.1
    lr_decay_steps: int = 30000
    training_steps: int = 100000
    val_every: int = 2000
    w_pos: float = 1.0
    w_aux: float = 1.0
    seed: int = 0


#: Model families dispatchable from ``[model].family`` (ADR-0032 §2).
MODEL_FAMILIES: dict[str, type] = {"gns": GNSConfig}

#: ``[run]`` keys — exactly these, no more, no fewer.
_RUN_KEYS = {"benchmark", "seed"}

#: ``TrainConfig`` fields sourced from ``[run]`` rather than ``[train]``.
_RUN_SOURCED = {"benchmark", "seed"}


@dataclass(frozen=True)
class ProtocolOverride:
    """Research-only protocol override (ADR-0032 §4); marks the run non-standard."""

    init_frames: int


@dataclass
class ResolvedRunConfig:
    """A fully-loaded grouped run config."""

    family: str
    model: Any  # instance of MODEL_FAMILIES[family]
    train: TrainConfig
    protocol_override: ProtocolOverride | None = None


class ConfigError(ValueError):
    """A run config failed strict validation; the message says how to fix it."""


#: TOML value types acceptable per annotated field type (bools are not ints).
_TYPE_OK: dict[str, tuple[type, ...]] = {
    "int": (int,),
    "float": (int, float),
    "str": (str,),
    "bool": (bool,),
}


def _check_value_types(section: str, table: dict[str, Any], cls: type) -> None:
    """Reject wrong-typed values so strict validation fails at load, not mid-run."""
    annotations = {f.name: f.type for f in fields(cls)}
    for key, value in table.items():
        expected = annotations.get(key)
        allowed = _TYPE_OK.get(str(expected))
        if allowed is None:
            continue
        if isinstance(value, bool) and expected != "bool":
            raise ConfigError(
                f"[{section}] {key} must be {expected}, got bool ({value!r})"
            )
        if not isinstance(value, allowed):
            raise ConfigError(
                f"[{section}] {key} must be {expected}, "
                f"got {type(value).__name__} ({value!r})"
            )


def _require_keys(section: str, given: set[str], required: set[str]) -> None:
    missing = sorted(required - given)
    unknown = sorted(given - required)
    problems = []
    if missing:
        problems.append(f"missing keys: {', '.join(missing)}")
    if unknown:
        problems.append(f"unknown keys: {', '.join(unknown)}")
    if problems:
        raise ConfigError(f"[{section}] {'; '.join(problems)}")


def load_run_config(path: str | Path) -> ResolvedRunConfig:
    """Load and strictly validate a grouped run config (ADR-0032).

    Parameters
    ----------
    path : str or pathlib.Path
        TOML file with ``[run]``, ``[model]``, ``[train]`` and optionally
        ``[protocol]`` sections.

    Returns
    -------
    ResolvedRunConfig

    Raises
    ------
    ConfigError
        On any unknown section or key, any missing required key, a flat
        (pre-0032) config, or an unregistered model family.
    """
    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    arrays = sorted(k for k, v in data.items() if isinstance(v, list))
    if arrays:
        raise ConfigError(
            f"sections {', '.join(arrays)} use [[...]] (array of tables); "
            f"each must be a single table — write [{arrays[0]}], not [[{arrays[0]}]]"
        )
    flat = sorted(k for k, v in data.items() if not isinstance(v, dict))
    if flat:
        raise ConfigError(
            f"top-level keys {', '.join(flat)} — flat configs are no longer "
            "supported; use [run]/[model]/[train] sections (ADR-0032)"
        )
    sections = set(data)
    unknown_sections = sorted(sections - {"run", "model", "train", "protocol"})
    if unknown_sections:
        raise ConfigError(f"unknown sections: {', '.join(unknown_sections)}")
    missing_sections = sorted({"run", "model", "train"} - sections)
    if missing_sections:
        raise ConfigError(f"missing sections: {', '.join(missing_sections)}")

    run = data["run"]
    _require_keys("run", set(run), _RUN_KEYS)
    if not isinstance(run["benchmark"], str):
        raise ConfigError(f"[run] benchmark must be str, got {run['benchmark']!r}")
    if isinstance(run["seed"], bool) or not isinstance(run["seed"], int):
        raise ConfigError(f"[run] seed must be int, got {run['seed']!r}")

    model_table = dict(data["model"])
    family = model_table.pop("family", None)
    if family is None:
        raise ConfigError("[model] missing key: family")
    if family not in MODEL_FAMILIES:
        raise ConfigError(
            f"[model] unknown family {family!r}; registered: "
            f"{', '.join(sorted(MODEL_FAMILIES))}"
        )
    model_cls = MODEL_FAMILIES[family]
    _require_keys("model", set(model_table), {f.name for f in fields(model_cls)})
    _check_value_types("model", model_table, model_cls)

    train_table = data["train"]
    misplaced = sorted(_RUN_SOURCED & set(train_table))
    if misplaced:
        raise ConfigError(f"[train] keys {', '.join(misplaced)} belong in [run]")
    train_fields = {f.name for f in fields(TrainConfig)} - _RUN_SOURCED
    _require_keys("train", set(train_table), train_fields)
    _check_value_types("train", train_table, TrainConfig)

    override = None
    if "protocol" in data:
        protocol_table = data["protocol"]
        _require_keys("protocol", set(protocol_table), {"init_frames"})
        init = protocol_table["init_frames"]
        if not isinstance(init, int) or init < 2:
            raise ConfigError(
                f"[protocol] init_frames must be an int >= 2, got {init!r}"
            )
        override = ProtocolOverride(init_frames=init)

    return ResolvedRunConfig(
        family=family,
        model=model_cls(**model_table),
        train=TrainConfig(benchmark=run["benchmark"], seed=run["seed"], **train_table),
        protocol_override=override,
    )


def _git_commit() -> str:
    """Short commit hash of the source tree this module runs from, or "unknown"."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def resolved_config_dict(
    family: str,
    model: Any,
    train: TrainConfig,
    *,
    init_frames: int,
    horizon: str = "full",
    eval_times: str = "native",
    standard: bool,
    n_particle_types: int,
    data_root: Path,
) -> dict[str, Any]:
    """The nested ``config.json`` payload for one run (ADR-0032 §3).

    Parameters
    ----------
    family : str
        Model-family registry key.
    model :
        The family's config instance (dataclass).
    train : TrainConfig
        Schedule and loss weights; ``benchmark``/``seed`` are emitted under
        ``"run"``.
    init_frames : int
        The protocol init actually used by this run (card value or override).
    horizon, eval_times : str
        The card's protocol values (ADR-0032 §4), recorded verbatim.
    standard : bool
        False when a ``[protocol]`` override was applied; such runs are
        ineligible for official card metrics (ADR-0032 §4).
    n_particle_types : int
        As computed from the training trajectories.
    data_root : pathlib.Path
        Directory of canonical cases.
    """
    return {
        "run": {
            "benchmark": train.benchmark,
            "seed": train.seed,
            "commit": _git_commit(),
        },
        "model": {"family": family, **asdict(model)},
        "train": {k: v for k, v in asdict(train).items() if k not in _RUN_SOURCED},
        "protocol": {
            "init_frames": init_frames,
            "horizon": horizon,
            "eval_times": eval_times,
            "standard": standard,
        },
        "n_particle_types": n_particle_types,
        "data_root": str(data_root),
    }


def read_run_record(config_path: Path) -> dict[str, Any]:
    """Read a run directory's ``config.json``, normalizing pre-0032 records.

    Pre-0032 run directories store ``{"benchmark", "gns", "train", ...}``
    with no protocol block; they are normalized to the nested shape with
    ``family = "gns"`` and ``init_frames = window`` (their historical
    protocol), so evaluation of fleet-era checkpoints keeps working.

    Raises
    ------
    FileNotFoundError
        If ``config_path`` does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"missing run config: {config_path}")
    record = json.loads(config_path.read_text(encoding="utf-8"))
    if "model" in record:
        return record
    # Minimal legacy records (e.g. viz-only) may lack the "gns" section; real
    # pre-0032 run dirs always carry it. Window 11 was the only pre-0032
    # default in use.
    gns = dict(record.get("gns") or {})
    train = dict(record.get("train", {}))
    return {
        "run": {
            "benchmark": record.get("benchmark", "taylor_impact_2d"),
            "seed": train.pop("seed", 0),
            "commit": "pre-0032",
        },
        "model": {"family": "gns", **gns},
        "train": train,
        "protocol": {
            "init_frames": gns.get("window", 11),
            "horizon": "full",
            "eval_times": "native",
            "standard": True,
        },
        "n_particle_types": record.get("n_particle_types", 1),
        "data_root": record.get("data_root", ""),
    }
