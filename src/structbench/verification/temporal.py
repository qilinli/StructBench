"""Temporal measures of a stored trajectory (ADR-0071, part two (b)).

Solution verification in time, beside ``convergence.py``'s in space: when a
response settles, when contact ends, how fast its globals move, and whether
the stored frame interval resolves it. Everything works on ``Case`` objects
and is threshold-free; the preflight applies the dataset's declared
tolerances to what is measured here.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from structbench.core import Case, Response
from structbench.verification.kernels import relative_l2_pooled

#: A quantity that ends near zero is judged against this fraction of its peak,
#: so float noise about zero does not read as motion.
_ZERO_FLOOR = 1e-9


def _response(case: Case) -> Response:
    if case.response is None:
        raise ValueError("the case stores no response")
    return case.response


def _arrays(case: Case) -> dict[str, NDArray]:
    """Every stored field keyed ``node/<f>``, ``<block>/<f>``, ``global/<g>``."""
    r = _response(case)
    out: dict[str, NDArray] = {f"node/{f}": a for f, a in r.node.items()}
    for block, fields in r.element.items():
        out.update({f"{block}/{f}": a for f, a in fields.items()})
    out.update({f"global/{g}": a for g, a in r.globals_.items()})
    return out


def response_until(case: Case, frame: int) -> Case:
    """The case with frames ``0..frame`` of every response array (a copy)."""
    r = _response(case)
    n = len(r.time)
    if not 0 <= frame < n:
        raise ValueError(f"frame {frame} is outside the stored range 0..{n - 1}")
    k = frame + 1
    cut = dataclasses.replace(
        r,
        time=r.time[:k],
        node={f: a[:k] for f, a in r.node.items()},
        element={
            b: {f: a[:k] for f, a in fields.items()} for b, fields in r.element.items()
        },
        globals_={g: a[:k] for g, a in r.globals_.items()},
    )
    return dataclasses.replace(case, response=cut)


def qoi_history(
    case: Case, qoi: Callable[[Case], Mapping[str, float]]
) -> dict[str, NDArray[np.float64]]:
    """``qoi()`` evaluated on the trajectory truncated at every frame."""
    r = _response(case)
    rows = [qoi(response_until(case, k)) for k in range(len(r.time))]
    names = list(rows[0]) if rows else []
    return {name: np.array([float(row[name]) for row in rows]) for name in names}


def settling_frame(
    history: Mapping[str, ArrayLike], tolerance: Mapping[str, float]
) -> int:
    """The last frame at which any quantity is outside its tolerance of its
    final value (relative to that value); 0 when none ever is."""
    last = 0
    for name, values in history.items():
        q = np.asarray(values, dtype=np.float64).ravel()
        if q.size == 0:
            continue
        final = q[-1]
        scale = max(abs(final), _ZERO_FLOOR * float(np.max(np.abs(q))), 1e-300)
        outside = np.flatnonzero(np.abs(q - final) > tolerance[name] * scale)
        if outside.size:
            last = max(last, int(outside[-1]))
    return last


def separation_frame(force: ArrayLike, *, threshold: float = 1e-3) -> int | None:
    """The last frame at which ``|force|`` exceeds ``threshold`` of its peak;
    None when the series has no peak (all zero)."""
    f = np.abs(np.asarray(force, dtype=np.float64).ravel())
    peak = float(f.max()) if f.size else 0.0
    if peak <= 0.0:
        return None
    above = np.flatnonzero(f > threshold * peak)
    return int(above[-1]) if above.size else None


def rise_time_frames(series: ArrayLike) -> int | None:
    """Frames from the first crossing of 10 % of the peak magnitude to the
    first crossing of 90 %; None for a flat series or one that does not rise
    from below 10 % (a falling series, or one already up at its first frame)."""
    s = np.abs(np.asarray(series, dtype=np.float64).ravel())
    if s.size == 0 or not bool(np.all(np.isfinite(s))):
        return None
    peak = float(s.max())
    if peak <= 0.0 or s[0] >= 0.1 * peak:
        return None
    low = np.flatnonzero(s >= 0.1 * peak)
    high = np.flatnonzero(s >= 0.9 * peak)
    return int(high[0] - low[0])


def interpolation_errors(case: Case, stride: int) -> dict[str, float]:
    """How well every ``stride``-th frame predicts the frames between by linear
    interpolation.

    For a trajectory exported at ``1 / stride`` of the intended frame
    interval, the frames at multiples of ``stride`` are the intended clock and
    the others sit between them: the pooled relative L2 (ADR-0055) of the
    interpolated in-between frames against the stored ones, per field, is the
    resolution error of the intended clock. Uses the stored times, so a
    non-uniform clock is exact. The frame count must be a whole number of
    strides plus one.
    """
    r = _response(case)
    n = len(r.time)
    if stride < 2:
        raise ValueError("the frame probe needs a stride of at least 2")
    if n < stride + 1 or (n - 1) % stride != 0:
        raise ValueError(
            f"the frame probe's {n} frames are not a whole number of strides of "
            f"{stride} plus one"
        )
    t = np.asarray(r.time, dtype=np.float64)
    between = np.array([j for j in range(n) if j % stride], dtype=np.int64)
    lo = (between // stride) * stride
    hi = lo + stride
    weight = (t[between] - t[lo]) / (t[hi] - t[lo])
    out: dict[str, float] = {}
    for key, stored in _arrays(case).items():
        a = np.asarray(stored, dtype=np.float64)
        w = weight.reshape((-1,) + (1,) * (a.ndim - 1))
        pred = a[lo] + w * (a[hi] - a[lo])
        out[key] = relative_l2_pooled(pred, a[between])
    return out


def midpoint_interpolation_errors(case: Case) -> dict[str, float]:
    """``interpolation_errors`` at stride 2: a half-interval export."""
    return interpolation_errors(case, 2)


def common_instant_errors(coarse: Case, fine: Case) -> dict[str, float]:
    """The coarse-clock trajectory against the fine one at their common instants.

    The fine clock must be a whole stride of the coarse one (``(T_f - 1) /
    (T_c - 1)`` a whole number) with the same instants; the pooled relative L2
    of the coarse arrays against the subsampled fine ones is returned per field
    stored on both sides. Stride 1 compares two runs on one clock.
    """
    tc = np.asarray(_response(coarse).time, dtype=np.float64)
    tf = np.asarray(_response(fine).time, dtype=np.float64)
    if len(tc) < 2 or len(tf) < 2:
        raise ValueError("a trajectory needs at least two frames")
    ratio = (len(tf) - 1) / (len(tc) - 1)
    stride = round(ratio)
    if stride < 1 or abs(ratio - stride) > 1e-9:
        raise ValueError(
            f"the fine clock's {len(tf)} frames are not a whole stride of the "
            f"coarse clock's {len(tc)}"
        )
    span = max(abs(float(tc[-1])), 1e-300)
    if not np.allclose(tf[::stride], tc, rtol=1e-6, atol=1e-9 * span):
        raise ValueError("the common instants of the two clocks differ")
    fine_arrays, coarse_arrays = _arrays(fine), _arrays(coarse)
    out: dict[str, float] = {}
    for key, stored in coarse_arrays.items():
        if key not in fine_arrays:
            continue
        sampled = np.asarray(fine_arrays[key], dtype=np.float64)[::stride]
        a = np.asarray(stored, dtype=np.float64)
        if a.shape != sampled.shape:
            raise ValueError(
                f"{key}: {a.shape[1:]} on the coarse clock, {sampled.shape[1:]} on "
                "the fine one; the two runs must share a mesh"
            )
        out[key] = relative_l2_pooled(a, sampled)
    return out
