"""``CaseBoundSimulator``: shared per-case-binding state base (ADR-0043 §8).

Extracted (move-only) from MGN's ``MeshSimulator``
(:mod:`structbench.models.mgn.simulator`) as the state and control-flow every
case-bound simulator family shares — MGN today, the Transolver family next
(ADR-0041's multi-method plan). A subclass supplies its own network and
feature-building (graph-specific for MGN, point-set-specific for Transolver);
this base never touches either.

**STATEFULNESS CONTRACT — READ BEFORE USE.**

A ``CaseBoundSimulator`` subclass carries two kinds of state that must not be
confused:

1. Model parameters and any per-family normalizer buffers (e.g. MGN's four
   :class:`~structbench.models.mgn.normalizers.OnlineNormalizer` buffers).
   These live in ``state_dict()`` and travel with a checkpoint via
   :meth:`CaseBoundSimulator.save`/:meth:`CaseBoundSimulator.load`. They are
   **case-independent**.
2. Per-case binding, set by :meth:`CaseBoundSimulator.bind_case` and held as
   PLAIN (non-buffer, non-parameter) attributes, so a checkpoint never
   carries case data. Binding a case caches: the mesh-space reference
   coordinates, the one-hot node types, the kinematic-particle mask, and the
   bound case's FULL ground-truth position trajectory ``(T, P, dim)`` — of
   which only the KINEMATIC rows are ever read (the scripted-actuator
   velocity input, and the tripwire below). A subclass may cache additional
   per-case geometry (e.g. MGN's mesh edge index) via the
   :meth:`CaseBoundSimulator._on_bind_case` hook. Binding also resets the
   autoregressive step pointer.

A subclass's ``predict_positions`` (MGN's is the reference implementation) is
the ONLY method ``structbench.eval`` calls
(:func:`~structbench.eval.rollout.rollout`,
:func:`~structbench.eval.rollout.one_step_position_rmse`,
:func:`~structbench.eval.rollout.one_step_aux_rmse`) — none of them pass any
per-case context. Consequently:

* Call :meth:`CaseBoundSimulator.bind_case` before the first
  ``predict_positions`` call of an eval pass on a given trajectory.
* Call :meth:`CaseBoundSimulator.reset_rollout` before **EACH** separate eval
  pass over the SAME bound case. A fresh rollout and a fresh one-step sweep
  are two separate eval passes, and each needs its own ``reset_rollout()`` —
  the step pointer has no way to know an eval pass has ended on its own.

**Step pointer — deterministic, never search-based.** The first
``predict_positions`` call after ``bind_case``/``reset_rollout`` anchors the
pointer at ``t = F`` (``F`` = the input window's frame count), because every
eval entry point — rollout's autoregressive loop and both one-step
teacher-forced sweeps — makes its first call with a window whose last frame
is ground-truth frame ``F - 1``. Every subsequent call advances the pointer
by 1. :meth:`CaseBoundSimulator._advance_pointer` implements this.

**Tripwire — verification only, never anchoring.** At every call (when
kinematic particles are bound), the input window's kinematic rows are
compared against the bound ground truth at frame ``t - 1`` (``atol=1e-4``).
A mismatch raises ``RuntimeError`` naming both likely causes: a stale
rollout pointer (call ``reset_rollout()`` before each eval pass) or a
case/trajectory mismatch (``bind_case`` was not (re)bound to the trajectory
under evaluation). This check stays exact even when a kinematic particle is
stationary across consecutive frames (e.g. a HANDLE node paused
mid-actuation) — the exact failure mode a search-based "find this window in
the GT" anchor would silently mishandle. With no kinematic particles bound,
the pointer/tripwire logic is skipped entirely and the scripted-velocity
input is always zero.

**Scripted-velocity feature.** :meth:`CaseBoundSimulator._eval_scripted_velocity`
(eval path) and :meth:`CaseBoundSimulator._train_scripted_velocity` (training
path) both compute the scripted-actuator velocity node feature —
GT-next-frame minus current position on ``scripted_types`` rows, zero
elsewhere — from different sources (the bound ground truth vs. the caller's
``next_positions``) but the same semantics; see each method's docstring.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CaseBoundSimulator(nn.Module):
    """Per-case GT-binding state base shared by case-bound simulators.

    See the module docstring for the full statefulness contract (bind per
    case, reset before each eval pass, tripwire semantics) — it is not
    repeated here. A subclass builds its own network and normalizers, and
    overrides :meth:`_on_bind_case` for any per-case geometry it needs
    beyond what is cached here (e.g. MGN's mesh edge index).

    Parameters
    ----------
    dim:
        Spatial dimensionality of node positions.
    node_type_size:
        Width of the one-hot node-type encoding (``NodeType.SIZE`` in the
        source MeshGraphNets framework).
    kinematic_types:
        Node-type codes whose motion is prescribed by ground truth; their
        rows anchor and verify the step pointer (the tripwire).
    scripted_types:
        Node-type codes whose next-step ground-truth velocity is fed as a
        node input feature (a subset of ``kinematic_types`` in the ADR-0043
        recipe: OBSTACLE is scripted, HANDLE is not).
    history_velocities:
        Number of finite-difference window velocities appended to the node
        features (ADR-0049). ``0`` (the reference recipe) keeps the feature
        builder Markovian in position; a velocity-history run sets it to
        ``input_frames - 1``, and :meth:`_window_velocity_history` builds
        the feature from the rollout window.
    device:
        Device the module is moved to at construction time. A subclass that
        registers parameters/buffers after calling ``super().__init__()``
        must call ``self.to(device)`` again once those exist.
    """

    def __init__(
        self,
        dim: int = 3,
        node_type_size: int = 9,
        kinematic_types: tuple[int, ...] = (1, 3),
        scripted_types: tuple[int, ...] = (1,),
        history_velocities: int = 0,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self._dim = dim
        self._node_type_size = node_type_size
        self._kinematic_types = kinematic_types
        self._scripted_types = scripted_types
        self._history_velocities = history_velocities

        if not set(scripted_types) <= set(kinematic_types):
            raise ValueError(
                "scripted_types must be a subset of kinematic_types (the "
                "NORMAL-only noise mask relies on it): "
                f"scripted_types={scripted_types!r}, "
                f"kinematic_types={kinematic_types!r}"
            )

        # Per-case binding (populated by bind_case). These are PLAIN
        # attributes -- never passed through register_buffer/register_
        # parameter -- so they never enter state_dict(): a saved checkpoint
        # is case-independent, and predict_positions before bind_case fails
        # loudly instead of running on stale/absent data.
        self._reference_coords: Tensor | None = None
        self._node_type_onehot: Tensor | None = None
        self._kin_mask: Tensor | None = None
        self._scripted_mask: Tensor | None = None
        self._gt_positions: Tensor | None = None
        self._has_kinematic: bool = False
        self._n_gt_frames: int = 0
        self._t: int | None = None

        self.to(device)

    def bind_case(
        self,
        cells: Tensor,
        reference_coords: Tensor,
        particle_types: Tensor,
        kinematic_positions: Tensor,
    ) -> None:
        """Bind one case's static geometry and GT trajectory; reset pointer.

        Parameters
        ----------
        cells:
            ``(n_cells, nodes_per_cell)`` int64 element connectivity. Not
            interpreted here — forwarded verbatim to :meth:`_on_bind_case`,
            which a subclass overrides for its own per-case geometry setup
            (e.g. MGN builds its mesh edge index there).
        reference_coords:
            ``(P, dim)`` mesh-space (rest/reference) coordinates.
        particle_types:
            ``(P,)`` int64 node-type codes.
        kinematic_positions:
            ``(T, P, dim)`` full ground-truth world-space trajectory of this
            case. Only rows whose type is in ``kinematic_types`` are ever
            read (the scripted-velocity node feature, restricted further to
            ``scripted_types``, and the pointer tripwire).
        """
        self._reference_coords = reference_coords
        self._node_type_onehot = F.one_hot(
            particle_types, num_classes=self._node_type_size
        ).to(torch.float32)

        dtype, device = particle_types.dtype, particle_types.device
        kinematic = torch.tensor(self._kinematic_types, dtype=dtype, device=device)
        scripted = torch.tensor(self._scripted_types, dtype=dtype, device=device)
        self._kin_mask = torch.isin(particle_types, kinematic)
        self._scripted_mask = torch.isin(particle_types, scripted)
        self._has_kinematic = bool(self._kin_mask.any())

        self._gt_positions = kinematic_positions
        self._n_gt_frames = kinematic_positions.shape[0]
        self._t = None

        self._on_bind_case(cells)

    def _on_bind_case(self, cells: Tensor) -> None:
        """Subclass hook for per-case geometry setup; default no-op.

        Called at the end of :meth:`bind_case`, after all method-agnostic
        bind state above has been cached, with the same ``cells`` that call
        received. MGN's :class:`~structbench.models.mgn.simulator.MeshSimulator`
        overrides this to build its mesh edge index
        (``cells_to_edges(cells)``); a subclass with no static connectivity
        to derive from ``cells`` may leave the default no-op.

        Parameters
        ----------
        cells:
            ``(n_cells, nodes_per_cell)`` int64 element connectivity, as
            passed to :meth:`bind_case`.
        """

    def reset_rollout(self) -> None:
        """Reset the step pointer; the next call re-anchors at ``t = F``."""
        self._t = None

    def _window_velocity_history(self, window: Tensor) -> Tensor:
        """Flattened last ``history_velocities`` finite differences of a window.

        Parameters
        ----------
        window:
            ``(P, F, dim)`` position window, most recent frame last — the
            same tensor :meth:`predict_positions` receives.

        Returns
        -------
        Tensor
            ``(P, history_velocities * dim)`` float32: the window's last
            ``history_velocities`` per-frame position differences, oldest
            first, flattened.

        Raises
        ------
        ValueError
            If the window carries fewer than ``history_velocities + 1``
            frames — the rollout seed must supply the full history.
        """
        h = self._history_velocities
        if window.shape[1] < h + 1:
            raise ValueError(
                f"velocity history needs {h + 1} window frames, got "
                f"{window.shape[1]}; the rollout seed must supply the full "
                "input_frames window (ADR-0035/ADR-0049)"
            )
        velocities = window[:, 1:] - window[:, :-1]
        return velocities[:, -h:].flatten(1)

    def _advance_pointer(self, x_t: Tensor, n_frames: int, step: int = 1) -> None:
        """Advance the autoregressive step pointer and verify it (tripwire).

        No-op when the bound case has no kinematic particles: ``self._t`` is
        then never touched and stays ``None`` for the whole rollout, which
        is what :meth:`_eval_scripted_velocity` reads as "no pointer, zero
        scripted-velocity feature".

        Parameters
        ----------
        x_t:
            ``(P, dim)`` current (most recent) world positions from the
            input window.
        n_frames:
            The input window's frame count ``F``; anchors the pointer at
            ``t = F`` on the first call after :meth:`bind_case`/
            :meth:`reset_rollout`.
        step:
            Frames the previous call consumed — the pointer advance per call
            (ADR-0050/0051). ``1`` (default) is the autoregressive scheme and
            every ``k=1`` family (MGN/GeoFLARE call this positionally); a
            ``k``-frame bundle advances by ``k`` so the tripwire's
            ``gt_positions[t-1]`` aligns with the re-seeded window's last
            predicted frame. Only the LAST (possibly short) bundle of a rollout
            advances by a value other than ``k``, and no call follows it, so a
            fixed ``step=k`` never misleads a successor.

        Raises
        ------
        RuntimeError
            If the input window's kinematic rows do not match the bound
            ground truth at frame ``t - 1`` (see the module docstring's
            tripwire section).
        """
        if not self._has_kinematic:
            return
        kin_mask = self._kin_mask
        gt_positions = self._gt_positions
        # has_kinematic is only ever True after a bind_case() that set both.
        assert kin_mask is not None and gt_positions is not None

        t = n_frames if self._t is None else self._t + step
        gt_prev = gt_positions[t - 1]
        if not torch.allclose(x_t[kin_mask], gt_prev[kin_mask], atol=1e-4):
            raise RuntimeError(
                "the bound case's kinematic input rows are out of sync "
                f"with the bound ground-truth trajectory at frame {t - 1}. "
                "This means either: call reset_rollout() before each "
                "eval pass (rollout / one_step_position_rmse / "
                "one_step_aux_rmse), or bind_case() was not (re)bound to "
                "the trajectory currently being evaluated."
            )
        self._t = t

    def _eval_scripted_velocity(self, x_t: Tensor) -> Tensor:
        """Compute the eval-path scripted-actuator velocity node feature.

        Reads the pointer already advanced by :meth:`_advance_pointer`
        (advances nothing itself): ``GT[t][scripted] - x_t[scripted]`` on
        ``scripted_types`` rows, zero elsewhere and zero past the bound
        trajectory's final frame (the final-frame guard) or whenever the
        pointer is unset (no kinematic particles bound).

        Parameters
        ----------
        x_t:
            ``(P, dim)`` current (most recent) world positions.

        Returns
        -------
        Tensor
            ``(P, dim)`` scripted velocity node feature.
        """
        scripted_mask = self._scripted_mask
        # Set by bind_case(); predict_positions() checks this before calling.
        assert scripted_mask is not None

        scripted_velocity = torch.zeros(
            x_t.shape[0], self._dim, dtype=x_t.dtype, device=x_t.device
        )
        t = self._t
        if t is not None and t < self._n_gt_frames:
            gt_positions = self._gt_positions
            assert gt_positions is not None
            gt_t = gt_positions[t]
            scripted_velocity[scripted_mask] = gt_t[scripted_mask] - x_t[scripted_mask]
        return scripted_velocity

    def _train_scripted_velocity(
        self, x_last: Tensor, next_positions: Tensor, particle_types: Tensor
    ) -> Tensor:
        """Compute the training-path scripted-actuator velocity feature.

        Same semantics as :meth:`_eval_scripted_velocity` but computed from
        the caller's ``next_positions`` (ground truth for the current
        training sample) rather than the bound case's cached trajectory —
        these differ in source, not meaning, since a training step is not
        bound to a case via :meth:`bind_case`.

        Parameters
        ----------
        x_last:
            ``(P, dim)`` current world positions (post-noise, if any).
        next_positions:
            ``(P, dim)`` ground-truth next-frame world positions.
        particle_types:
            ``(P,)`` int64 node-type codes.

        Returns
        -------
        Tensor
            ``(P, dim)`` scripted velocity node feature.
        """
        dtype, device = particle_types.dtype, particle_types.device
        scripted = torch.tensor(self._scripted_types, dtype=dtype, device=device)
        scripted_mask = torch.isin(particle_types, scripted)

        scripted_velocity = torch.zeros_like(x_last)
        scripted_velocity[scripted_mask] = (next_positions - x_last)[scripted_mask]
        return scripted_velocity

    def save(self, path: str | Path) -> None:
        """Save the model's ``state_dict`` (network + normalizer buffers).

        Parameters
        ----------
        path:
            Destination path.
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
        """Load a ``state_dict`` saved by :meth:`save` (mapped to CPU).

        Parameters
        ----------
        path:
            Source path of the saved state dict.
        """
        self.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
