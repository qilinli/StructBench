"""Simulator construction for each model family."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from ...config import (
    CGNConfig,
    GeoFlareConfig,
    MGNConfig,
    TransolverConfig,
)
from ...models.cgn import LearnedSimulator
from ...models.geoflare import GeoFlareSimulator
from ...models.mgn import (
    MeshSimulator,
)
from ...models.transolver import (
    TransolverSimulator,
)


def build_simulator(
    stats: dict[str, dict[str, Tensor]],
    cgn: CGNConfig,
    *,
    n_particle_types: int,
    boundary_feature_fn: Callable[[Tensor], Tensor] | None,
    device: str,
) -> LearnedSimulator:
    """Construct a :class:`LearnedSimulator` from stats and architecture config.

    The node-input width is computed as
    ``(input_frames - 1) * dim + n_boundary + embedding`` where ``n_boundary``
    is 1 when ``boundary_feature_fn`` is given (else 0) and ``embedding`` is
    ``particle_type_embedding_size`` when ``n_particle_types > 1`` (else 0). The
    edge-input width is ``dim + 1``. Each normalization std is inflated by the
    training noise as ``sqrt(std**2 + noise_std**2)``, matching the source.

    Parameters
    ----------
    stats : dict
        Mapping ``{"velocity": ..., "acceleration": ..., "aux": ...}`` where
        each value is ``{"mean": Tensor, "std": Tensor}``. Velocity and
        acceleration stats are per-dimension (shape ``(dim,)``); the ``"aux"``
        stats are per-channel (shape ``(C,)``; ADR-0059 — the decoder's aux
        width ``n_aux`` is derived from them). Velocity/acceleration std is
        inflated by the training noise; the auxiliary stats are passed through
        unchanged (the auxiliary target carries no input noise).
    cgn : CGNConfig
        Architecture and noise configuration.
    n_particle_types : int
        Number of distinct particle types; controls the embedding.
    boundary_feature_fn : Callable or None
        Maps the most-recent positions ``(P, dim)`` to a boundary feature block
        ``(P, 1)``; ``None`` adds no boundary feature.
    device : str
        Torch device string for the stats tensors and batch-id construction.

    Returns
    -------
    LearnedSimulator
    """
    n_boundary = 1 if boundary_feature_fn is not None else 0
    embedding = cgn.particle_type_embedding_size if n_particle_types > 1 else 0
    nnode_in = (cgn.input_frames - 1) * cgn.dim + n_boundary + embedding
    nedge_in = cgn.dim + 1

    noise_var = cgn.noise_std**2
    normalization_stats: dict[str, dict[str, Tensor]] = {}
    for key in ("velocity", "acceleration"):
        mean = stats[key]["mean"].to(device)
        std = torch.sqrt(stats[key]["std"].to(device) ** 2 + noise_var)
        normalization_stats[key] = {"mean": mean, "std": std}

    # The auxiliary target carries no input noise, so its stats are
    # passed through without the sqrt(std^2 + noise^2) inflation applied above.
    normalization_stats["aux"] = {
        "mean": stats["aux"]["mean"].to(device),
        "std": stats["aux"]["std"].to(device),
    }

    return LearnedSimulator(
        particle_dimensions=cgn.dim,
        nnode_in=nnode_in,
        nedge_in=nedge_in,
        latent_dim=cgn.hidden_dim,
        nmessage_passing_steps=cgn.message_passing_steps,
        nmlp_layers=cgn.nmlp_layers,
        mlp_hidden_dim=cgn.hidden_dim,
        connectivity_radius=cgn.connectivity_radius,
        normalization_stats=normalization_stats,
        nparticle_types=n_particle_types,
        particle_type_embedding_size=cgn.particle_type_embedding_size,
        # ADR-0059: the aux decoder width is the stats' channel count.
        n_aux=int(stats["aux"]["mean"].numel()),
        aux_transform=cgn.aux_transform,
        aux_transform_scale=cgn.aux_transform_scale,
        max_neighbors=cgn.max_neighbors,
        boundary_feature_fn=boundary_feature_fn,
        device=device,
    )


def build_mgn_simulator(
    mgn: MGNConfig,
    *,
    kinematic_types: tuple[int, ...],
    scripted_types: tuple[int, ...] | None = None,
    n_aux: int = 1,
    device: str,
) -> MeshSimulator:
    """Construct a :class:`MeshSimulator` from an :class:`MGNConfig`.

    Unlike :func:`build_simulator`, no normalization stats are needed at
    construction time: ``MeshSimulator``'s four
    :class:`~structbench.models.mgn.normalizers.OnlineNormalizer` buffers are
    part of its own ``state_dict`` (ADR-0043 §8), so a checkpoint carries its
    normalizer state and a fresh instance starts with the online normalizers'
    own (untrained) defaults until ``.load()`` restores them.

    Parameters
    ----------
    mgn : MGNConfig
        Architecture hyperparameters.
    kinematic_types : tuple of int
        Node-type codes whose motion is prescribed by ground truth (the
        benchmark spec's ``kinematic_types``); forwarded verbatim.
    device : str
        Torch device string.

    Returns
    -------
    MeshSimulator
        ``scripted_types=None`` leaves the class default ``(1,)`` (the
        ADR-0043 recipe scripts only the OBSTACLE node type); a benchmark
        overrides it via ``spec.scripted_types`` (ADR-0047).
    """
    return MeshSimulator(
        dim=mgn.dim,
        latent=mgn.hidden_dim,
        mp_steps=mgn.message_passing_steps,
        n_hidden=mgn.nmlp_layers,
        node_type_size=mgn.node_type_size,
        world_edge_radius=mgn.world_edge_radius,
        history_velocities=mgn.history_frames,
        mesh_edge_max_stretch=mgn.mesh_edge_max_stretch,
        kinematic_types=kinematic_types,
        **({} if scripted_types is None else {"scripted_types": scripted_types}),
        n_aux=n_aux,
        device=device,
    )


def build_transolver_simulator(
    cfg: TransolverConfig,
    *,
    kinematic_types: tuple[int, ...],
    scripted_types: tuple[int, ...] | None = None,
    n_aux: int = 1,
    hardening_curve: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
    element_formulation: str | None = None,
    device: str,
) -> TransolverSimulator:
    """Construct a :class:`TransolverSimulator` from a :class:`TransolverConfig`.

    Mirrors :func:`build_mgn_simulator`: no normalization stats are needed at
    construction time, since ``TransolverSimulator``'s two
    :class:`~structbench.models.mgn.normalizers.OnlineNormalizer` buffers
    (node and target) are part of its own ``state_dict``, matching MGN's
    self-contained-checkpoint pattern (ADR-0043 §8) — a checkpoint carries
    its normalizer state and a fresh instance starts with the online
    normalizers' own (untrained) defaults until ``.load()`` restores them.

    Parameters
    ----------
    cfg : TransolverConfig
        Architecture hyperparameters.
    kinematic_types : tuple of int
        Node-type codes whose motion is prescribed by ground truth (the
        benchmark spec's ``kinematic_types``); forwarded verbatim.
    device : str
        Torch device string.

    Returns
    -------
    TransolverSimulator
        ``scripted_types=None`` leaves the class default ``(1,)`` (the
        ADR-0043 recipe scripts only the OBSTACLE node type); a benchmark
        overrides it via ``spec.scripted_types`` (ADR-0047).
    """
    return TransolverSimulator(
        dim=cfg.dim,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        slice_num=cfg.slice_num,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        node_type_size=cfg.node_type_size,
        history_velocities=cfg.history_frames,
        # ADR-0050/0051: cfg.frames_per_call is the resolved k. The k=T sentinel
        # (0) is resolved to a concrete horizon upstream in _train_transolver
        # (train) / evaluate reads the resolved integer from config.json, so a
        # 0 never reaches here.
        frames_per_call=cfg.frames_per_call,
        impact_velocity_feature=cfg.impact_velocity_feature,
        time_conditioned=cfg.time_conditioned,
        adaptive_temperature=cfg.adaptive_temperature,
        slice_reparam=cfg.slice_reparam,
        aux_input=cfg.aux_input,
        flow_map=cfg.flow_map,
        flow_map_anchor_time=cfg.flow_map_anchor_time,
        # ADR-0064: the benchmark's sigma_y(peeq) table; consumed by the
        # simulator exactly when flow_map_structured_heads is on.
        flow_map_structured_heads=cfg.flow_map_structured_heads,
        hardening_curve=hardening_curve,
        # ADR-0075: the two-site decoder carries the element state block,
        # which has the nodal aux block's channels (its projection).
        two_site_decoder=cfg.two_site_decoder,
        n_state=n_aux if cfg.two_site_decoder != "off" else 0,
        element_formulation=element_formulation,
        kinematic_types=kinematic_types,
        **({} if scripted_types is None else {"scripted_types": scripted_types}),
        n_aux=n_aux,
        device=device,
    )


def build_geoflare_simulator(
    cfg: GeoFlareConfig,
    *,
    kinematic_types: tuple[int, ...],
    scripted_types: tuple[int, ...] | None = None,
    n_aux: int = 1,
    device: str,
) -> GeoFlareSimulator:
    """Construct a :class:`GeoFlareSimulator` from a :class:`GeoFlareConfig`.

    Mirrors :func:`build_transolver_simulator`: no normalization stats are
    needed at construction time, since ``GeoFlareSimulator``'s two
    :class:`~structbench.models.mgn.normalizers.OnlineNormalizer` buffers
    (node and target) are part of its own ``state_dict``, matching the
    MGN/Transolver self-contained-checkpoint pattern (ADR-0043 §8) — a
    checkpoint carries its normalizer state and a fresh instance starts
    with the online normalizers' own (untrained) defaults until ``.load()``
    restores them.

    The config's four ball-query scalars are assembled into the
    simulator's ``(near, far)`` tuples here — this scalar-to-tuple mapping
    lives ONLY in this function: ``radii=(cfg.radius_near, cfg.radius_far)``,
    ``neighbors=(cfg.neighbors_near, cfg.neighbors_far)``.

    Parameters
    ----------
    cfg : GeoFlareConfig
        Architecture hyperparameters.
    kinematic_types : tuple of int
        Node-type codes whose motion is prescribed by ground truth (the
        benchmark spec's ``kinematic_types``); forwarded verbatim.
    device : str
        Torch device string.

    Returns
    -------
    GeoFlareSimulator
        ``scripted_types=None`` leaves the class default ``(1,)`` (the
        ADR-0043 recipe scripts only the OBSTACLE node type); a benchmark
        overrides it via ``spec.scripted_types`` (ADR-0047).
    """
    return GeoFlareSimulator(
        dim=cfg.dim,
        n_hidden=cfg.n_hidden,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        slice_num=cfg.slice_num,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        n_hidden_local=cfg.n_hidden_local,
        radii=(cfg.radius_near, cfg.radius_far),
        neighbors=(cfg.neighbors_near, cfg.neighbors_far),
        node_type_size=cfg.node_type_size,
        history_velocities=cfg.history_frames,
        impact_velocity_feature=cfg.impact_velocity_feature,
        time_conditioned=cfg.time_conditioned,
        kinematic_types=kinematic_types,
        **({} if scripted_types is None else {"scripted_types": scripted_types}),
        n_aux=n_aux,
        device=device,
    )
