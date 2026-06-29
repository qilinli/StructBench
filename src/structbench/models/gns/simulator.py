"""LearnedSimulator: the GNS position-prediction wrapper.

Ported from the sgnn reference implementation
(``sgnn/single_scale/learned_simulator.py``) for the learned simulator of
Sanchez-Gonzalez et al., *Learning to Simulate Complex Physics with Graph
Networks* (https://arxiv.org/abs/2002.09405).

The port preserves the reference behaviour -- graph connectivity
(``radius_graph`` with ``max_num_neighbors=20`` and self-loops),
velocity-history normalisation, the Euler integrator of the decoder
post-processor, its inverse, the GNS noise handling, and ``save``/``load`` --
with three deliberate generalisations relative to the source:

1. The hardcoded left-wall distance feature is replaced by an injected
   ``boundary_feature_fn`` callable (``None`` adds no boundary feature).
2. The auxiliary decoder width is made explicit via ``n_aux`` (the reference
   hardcoded a single auxiliary output), and the auxiliary prediction is kept
   2-D with shape ``(nparticles, n_aux)``.
3. The reference debug/``print`` helpers are dropped; library code does not
   print.

Indentation is four-space; type hints and docstrings target Python 3.11+ and
mypy.  Positions are expressed in the millimetre working units of the ML layer
(see ADR-0019); this module is unit-agnostic and simply propagates whatever
units the caller supplies.
"""

import logging
from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor

from .graph_network import EncodeProcessDecode
from .graph_ops import radius_graph

logger = logging.getLogger(__name__)


class LearnedSimulator(nn.Module):
    """Learned graph-network simulator (Sanchez-Gonzalez et al., 2020).

    The simulator wraps an :class:`EncodeProcessDecode` network: it turns a
    short history of particle positions into a graph, predicts normalised
    accelerations (plus ``n_aux`` auxiliary channels), and integrates them
    forward one step with an Euler update.

    Parameters
    ----------
    particle_dimensions : int
        Spatial dimensionality of the problem (2 or 3).
    nnode_in : int
        Number of node input features expected by the encoder. The caller is
        responsible for computing this as
        ``(window - 1) * particle_dimensions`` plus the width of any boundary
        feature plus ``particle_type_embedding_size`` when
        ``nparticle_types > 1``.
    nedge_in : int
        Number of edge input features (``particle_dimensions + 1``: the
        normalised relative displacement plus its norm).
    latent_dim : int
        Size of the latent node and edge embeddings.
    nmessage_passing_steps : int
        Number of interaction-network message-passing steps.
    nmlp_layers : int
        Number of hidden layers in each MLP.
    mlp_hidden_dim : int
        Width of each MLP hidden layer.
    connectivity_radius : float
        Radius (in working units) within which particles are connected by an
        edge.
    normalization_stats : dict[str, dict[str, Tensor]]
        Mapping with keys ``"velocity"`` and ``"acceleration"``, each mapping
        to ``{"mean": Tensor, "std": Tensor}`` of shape
        ``(particle_dimensions,)``.
    nparticle_types : int
        Number of distinct particle types.
    particle_type_embedding_size : int
        Embedding size for the particle-type lookup.
    n_aux : int, optional
        Number of auxiliary output channels predicted alongside the
        acceleration (default ``1``). The decoder produces
        ``particle_dimensions + n_aux`` outputs.
    boundary_feature_fn : Callable[[Tensor], Tensor] | None, optional
        Maps the most-recent position ``(nparticles, particle_dimensions)`` to
        a node feature block ``(nparticles, n_boundary)`` appended after the
        velocity history and before the particle-type embedding. ``None``
        (default) adds no boundary feature.
    device : str, optional
        Runtime device for the batch-id construction (default ``"cpu"``).
    """

    def __init__(
        self,
        particle_dimensions: int,
        nnode_in: int,
        nedge_in: int,
        latent_dim: int,
        nmessage_passing_steps: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
        connectivity_radius: float,
        normalization_stats: dict[str, dict[str, Tensor]],
        nparticle_types: int,
        particle_type_embedding_size: int,
        *,
        n_aux: int = 1,
        boundary_feature_fn: Callable[[Tensor], Tensor] | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self._connectivity_radius = connectivity_radius
        self._normalization_stats = normalization_stats
        self._nparticle_types = nparticle_types
        self._particle_dimensions = particle_dimensions
        self._n_aux = n_aux
        self._boundary_feature_fn = boundary_feature_fn

        # Particle-type embedding lookup.
        self._particle_type_embedding = nn.Embedding(
            nparticle_types, particle_type_embedding_size
        )

        # Initialise the EncodeProcessDecode. The decoder emits the
        # ``particle_dimensions`` acceleration channels plus ``n_aux``
        # auxiliary channels.
        self._encode_process_decode = EncodeProcessDecode(
            nnode_in_features=nnode_in,
            nnode_out_features=particle_dimensions + n_aux,
            nedge_in_features=nedge_in,
            latent_dim=latent_dim,
            nmessage_passing_steps=nmessage_passing_steps,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )

        self._device = device

    def forward(self) -> None:
        """No-op forward hook; prediction goes through ``predict_positions``."""
        return None

    def _compute_graph_connectivity(
        self,
        positions: Tensor,
        nparticles_per_example: Tensor,
        radius: float,
        add_self_edges: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Build graph edges between particles within ``radius``.

        Parameters
        ----------
        positions : Tensor
            Particle positions with shape ``(nparticles, particle_dimensions)``.
        nparticles_per_example : Tensor
            Number of particles in each example of the batch.
        radius : float
            Threshold radius for connecting particles.
        add_self_edges : bool, optional
            Whether to include self-loops (default ``True``).

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(receivers, senders)`` node-index tensors, each shape
            ``(nedges,)``.
        """
        # Validate inputs
        if len(positions.shape) != 2:
            raise ValueError(
                f"Expected 2D positions tensor, got shape {positions.shape}"
            )

        if not isinstance(nparticles_per_example, torch.Tensor):
            nparticles_per_example = torch.tensor(nparticles_per_example)

        # Ensure nparticles_per_example is 1D
        if len(nparticles_per_example.shape) > 1:
            nparticles_per_example = nparticles_per_example.flatten()

        # Validate that total particles matches
        total_particles = nparticles_per_example.sum().item()
        if total_particles != positions.shape[0]:
            logger.warning(
                "Total particles mismatch: %s vs %s (nparticles_per_example=%s)",
                total_particles,
                positions.shape[0],
                nparticles_per_example,
            )

        # Specify examples id for particles
        batch_ids = torch.cat(
            [
                torch.LongTensor([i for _ in range(n)])
                for i, n in enumerate(nparticles_per_example)
            ]
        ).to(positions.device)

        # Native radius graph: edge_index[0] is the central node i, edge_index[1]
        # its neighbour j (distance <= radius); shape (2, nedges).
        edge_index = radius_graph(
            positions,
            r=radius,
            batch=batch_ids,
            loop=add_self_edges,
            max_num_neighbors=20,
        )

        # The flow direction when combined with message passing is
        # "source_to_target".
        receivers = edge_index[0, :]
        senders = edge_index[1, :]

        return receivers, senders

    def _encoder_preprocessor(
        self,
        position_sequence: Tensor,
        nparticles_per_example: Tensor,
        particle_types: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Extract node and edge features from a position sequence.

        Parameters
        ----------
        position_sequence : Tensor
            Particle positions with shape
            ``(nparticles, window, particle_dimensions)`` (current plus history).
        nparticles_per_example : Tensor
            Number of particles in each example of the batch.
        particle_types : Tensor
            Particle types with shape ``(nparticles,)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(node_features, edge_index, edge_features)`` where
            ``node_features`` has shape ``(nparticles, nnode_in)``,
            ``edge_index`` has shape ``(2, nedges)``, and ``edge_features`` has
            shape ``(nedges, particle_dimensions + 1)``.
        """
        # Ensure input tensor is contiguous for reliable operations.
        position_sequence = position_sequence.contiguous()

        # Validate input dimensions.
        if len(position_sequence.shape) != 3:
            raise ValueError(
                "Expected position_sequence to have 3 dimensions, got "
                f"{len(position_sequence.shape)}"
            )
        if position_sequence.shape[1] < 2:
            raise ValueError(
                f"Expected at least 2 timesteps, got {position_sequence.shape[1]}"
            )

        nparticles = position_sequence.shape[0]
        most_recent_position = position_sequence[:, -1].contiguous()
        velocity_sequence = time_diff(position_sequence).contiguous()

        # Graph connectivity from the most-recent positions. Note the reference
        # binds (senders, receivers) to the returned (receivers, senders); this
        # naming swap is preserved for behavioural fidelity.
        senders, receivers = self._compute_graph_connectivity(
            most_recent_position, nparticles_per_example, self._connectivity_radius
        )

        node_features: list[Tensor] = []

        # Normalised velocity sequence, merging the spatial and time axes.
        velocity_stats = self._normalization_stats["velocity"]
        normalized_velocity_sequence = (
            (velocity_sequence - velocity_stats["mean"]) / velocity_stats["std"]
        ).contiguous()
        flat_velocity_sequence = normalized_velocity_sequence.reshape(nparticles, -1)
        node_features.append(flat_velocity_sequence)

        # Optional boundary feature (generalisation of the reference's hardcoded
        # left-wall distance). Appended after the velocity block and before the
        # particle-type embedding, matching the reference feature ordering.
        if self._boundary_feature_fn is not None:
            node_features.append(self._boundary_feature_fn(most_recent_position))

        # Particle type.
        if self._nparticle_types > 1:
            particle_type_embeddings = self._particle_type_embedding(particle_types)
            node_features.append(particle_type_embeddings)

        # Collect edge features.
        edge_features: list[Tensor] = []

        # Relative displacement normalised to the connectivity radius,
        # shape (nedges, particle_dimensions).
        normalized_relative_displacements = (
            most_recent_position[senders, :] - most_recent_position[receivers, :]
        ) / self._connectivity_radius
        edge_features.append(normalized_relative_displacements)

        # Relative distance with shape (nedges, 1); edge features end up shaped
        # (nedges, particle_dimensions + 1).
        normalized_relative_distances = torch.norm(
            normalized_relative_displacements, dim=-1, keepdim=True
        )
        edge_features.append(normalized_relative_distances)

        return (
            torch.cat(node_features, dim=-1),
            torch.stack([senders, receivers]),
            torch.cat(edge_features, dim=-1),
        )

    def _decoder_postprocessor(
        self,
        normalized_acceleration: Tensor,
        position_sequence: Tensor,
    ) -> Tensor:
        """Integrate normalised acceleration into a new position.

        The model output is in normalised space, so inverse normalisation is
        applied before an Euler integration step (assuming ``dt = 1``, the size
        of the finite difference).

        Parameters
        ----------
        normalized_acceleration : Tensor
            Normalised acceleration with shape
            ``(nparticles, particle_dimensions)``.
        position_sequence : Tensor
            Position sequence with shape
            ``(nparticles, window, particle_dimensions)``.

        Returns
        -------
        Tensor
            New particle positions with shape
            ``(nparticles, particle_dimensions)``.
        """
        # Recover real acceleration from normalised values.
        acceleration_stats = self._normalization_stats["acceleration"]
        acceleration = (
            normalized_acceleration * acceleration_stats["std"]
        ) + acceleration_stats["mean"]

        # Euler integrator from acceleration to position, assuming dt = 1.
        most_recent_position = position_sequence[:, -1]
        most_recent_velocity = most_recent_position - position_sequence[:, -2]

        new_velocity = most_recent_velocity + acceleration  # * dt = 1
        new_position = most_recent_position + new_velocity  # * dt = 1
        return new_position

    def predict_positions(
        self,
        current_positions: Tensor,
        nparticles_per_example: Tensor,
        particle_types: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Predict the next positions and auxiliary outputs.

        Parameters
        ----------
        current_positions : Tensor
            Position sequence with shape
            ``(nparticles, window, particle_dimensions)``.
        nparticles_per_example : Tensor
            Number of particles in each example of the batch.
        particle_types : Tensor
            Particle types with shape ``(nparticles,)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(next_positions, predicted_aux)`` with shapes
            ``(nparticles, particle_dimensions)`` and ``(nparticles, n_aux)``.
        """
        node_features, edge_index, edge_features = self._encoder_preprocessor(
            current_positions, nparticles_per_example, particle_types
        )
        pred = self._encode_process_decode(node_features, edge_index, edge_features)
        # The first ``particle_dimensions`` channels are accelerations; the
        # remaining ``n_aux`` channels are auxiliary outputs (kept 2-D).
        predicted_normalized_acceleration = pred[:, : self._particle_dimensions]
        predicted_aux = pred[:, self._particle_dimensions :]
        next_positions = self._decoder_postprocessor(
            predicted_normalized_acceleration, current_positions
        )

        return next_positions, predicted_aux

    def predict_accelerations(
        self,
        next_positions: Tensor,
        position_sequence_noise: Tensor,
        position_sequence: Tensor,
        nparticles_per_example: Tensor,
        particle_types: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Produce predicted and target normalised accelerations (training).

        Parameters
        ----------
        next_positions : Tensor
            Ground-truth next positions with shape
            ``(nparticles, particle_dimensions)``.
        position_sequence_noise : Tensor
            Noise to add to the input positions, same shape as
            ``position_sequence``.
        position_sequence : Tensor
            Position sequence with shape
            ``(nparticles, window, particle_dimensions)``.
        nparticles_per_example : Tensor
            Number of particles in each example of the batch.
        particle_types : Tensor
            Particle types with shape ``(nparticles,)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(predicted_normalized_acceleration, target_normalized_acceleration,
            predicted_aux)`` with shapes ``(nparticles, particle_dimensions)``,
            ``(nparticles, particle_dimensions)``, and ``(nparticles, n_aux)``.
        """
        # Add noise to the input position sequence.
        noisy_position_sequence = position_sequence + position_sequence_noise

        # Forward pass with the noisy position sequence.
        node_features, edge_index, edge_features = self._encoder_preprocessor(
            noisy_position_sequence, nparticles_per_example, particle_types
        )
        pred = self._encode_process_decode(node_features, edge_index, edge_features)
        predicted_normalized_acceleration = pred[:, : self._particle_dimensions]
        predicted_aux = pred[:, self._particle_dimensions :]

        # Compute the target acceleration using an `adjusted_next_position` that
        # is shifted by the noise in the last input position.
        next_position_adjusted = next_positions + position_sequence_noise[:, -1]
        target_normalized_acceleration = self._inverse_decoder_postprocessor(
            next_position_adjusted, noisy_position_sequence
        )
        # The inverted Euler update in `_inverse_decoder_postprocessor` produces:
        # * A target acceleration that does not explicitly correct for the noise
        #   in the input positions, as `next_position_adjusted` differs from the
        #   true `next_position`.
        # * A target acceleration that exactly corrects noise in the input
        #   velocity, since the target next velocity computed by the inverse
        #   Euler update as `next_position_adjusted - noisy_position_sequence[:,
        #   -1]` matches the ground-truth next velocity (noise cancels out).

        return (
            predicted_normalized_acceleration,
            target_normalized_acceleration,
            predicted_aux,
        )

    def _inverse_decoder_postprocessor(
        self,
        next_position: Tensor,
        position_sequence: Tensor,
    ) -> Tensor:
        """Invert :meth:`_decoder_postprocessor`.

        Parameters
        ----------
        next_position : Tensor
            Next positions with shape ``(nparticles, particle_dimensions)``.
        position_sequence : Tensor
            Position sequence with shape
            ``(nparticles, window, particle_dimensions)``.

        Returns
        -------
        Tensor
            Normalised acceleration with shape
            ``(nparticles, particle_dimensions)``.
        """
        previous_position = position_sequence[:, -1]
        previous_velocity = previous_position - position_sequence[:, -2]
        next_velocity = next_position - previous_position
        acceleration = next_velocity - previous_velocity

        acceleration_stats = self._normalization_stats["acceleration"]
        normalized_acceleration = (
            acceleration - acceleration_stats["mean"]
        ) / acceleration_stats["std"]
        return normalized_acceleration

    def save(self, path: str = "model.pt") -> None:
        """Save the model state dict.

        Parameters
        ----------
        path : str, optional
            Destination path (default ``"model.pt"``).
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """Load the model state dict from ``path`` (mapped to CPU).

        Parameters
        ----------
        path : str
            Source path of the saved state dict.
        """
        self.load_state_dict(torch.load(path, map_location=torch.device("cpu")))


def time_diff(position_sequence: Tensor) -> Tensor:
    """Finite difference of a position sequence along the time axis.

    Parameters
    ----------
    position_sequence : Tensor
        Position sequence with shape
        ``(nparticles, window, particle_dimensions)``.

    Returns
    -------
    Tensor
        Velocity sequence with shape
        ``(nparticles, window - 1, particle_dimensions)``.
    """
    return (position_sequence[:, 1:] - position_sequence[:, :-1]).contiguous()
