"""``ball_query`` / ``standardize_coords`` (ADR-0041 step 3; ADR-0045).

Semantics pinned to NVIDIA PhysicsNeMo's torch-fallback ball query (see
``scratch/2026-08-09-geoflare-grounding.md`` S10): nearest-first top-k,
absolute coordinates, zero-padded rows past however many neighbours
qualify within ``radius``.
"""

import torch

from structbench.models.geoflare.geo_ops import ball_query, standardize_coords

# Five collinear points, spaced so every pairwise distance is unique (powers
# of two: 1, 2, 4, 8 between successive points) -- this makes nearest-first
# ORDER unambiguous from every query row (no ties to obscure a mutation that
# breaks ordering).
_LINE = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [7.0, 0.0, 0.0],
        [15.0, 0.0, 0.0],
    ]
)


def test_ball_query_nearest_first_order_pinned() -> None:
    # (a) Exact neighbor sets and ORDER, radius large enough to include
    # everyone so only the top-k selection (not the radius cutoff) is
    # exercised.
    out = ball_query(_LINE, radius=100.0, k=3)

    # Query index 0 (x=0): distances to [0,1,3,7,15] = 0,1,3,7,15 ->
    # nearest three (ascending) are self, x=1, x=3.
    assert torch.allclose(out[0], torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]]))
    # Query index 2 (x=3): distances to [0,1,3,7,15] = 3,2,0,4,12 ->
    # nearest three (ascending) are self(0), x=1(2), x=0(3).
    assert torch.allclose(out[2], torch.tensor([[3.0, 0, 0], [1.0, 0, 0], [0.0, 0, 0]]))
    # Query index 4 (x=15): distances = 15,14,12,8,0 -> nearest three are
    # self(0), x=7(8), x=3(12).
    assert torch.allclose(
        out[4], torch.tensor([[15.0, 0, 0], [7.0, 0, 0], [3.0, 0, 0]])
    )


def test_ball_query_radius_cutoff_zero_pads_even_with_k_slots_remaining() -> None:
    # (b) Query index 3 (x=7): distances to [0,1,3,7,15] = 7,6,4,0,8.
    # With radius=5, only self (0) and x=3 (4) qualify; x=1 (6) is JUST
    # outside radius even though a plain top-3-by-distance would include it
    # (a top-k-ignoring-radius bug would put x=1's coords, [1,0,0], in the
    # third slot instead of a zero row).
    out = ball_query(_LINE, radius=5.0, k=3)
    assert torch.allclose(out[3], torch.tensor([[7.0, 0, 0], [3.0, 0, 0], [0.0, 0, 0]]))


def test_ball_query_fewer_than_k_padding_is_exact_zero_rows() -> None:
    # (c) Query index 2 (x=3, deliberately NOT at the origin -- a "pad with
    # the query's own coords" bug would be indistinguishable from "pad with
    # zero" for a query point that happens to sit at the origin), radius=0.5:
    # nearest other point is x=1 at distance 2, so only self qualifies; the
    # remaining k-1 slots must be EXACTLY (0, 0, 0), not some other sentinel.
    out = ball_query(_LINE, radius=0.5, k=4)
    assert torch.equal(out[2, 0], torch.tensor([3.0, 0.0, 0.0]))
    assert torch.equal(out[2, 1], torch.zeros(3))
    assert torch.equal(out[2, 2], torch.zeros(3))
    assert torch.equal(out[2, 3], torch.zeros(3))


def test_ball_query_returns_absolute_coords_not_offsets() -> None:
    # (d) A line shifted well away from the origin: if the implementation
    # mistakenly returned (neighbor - query) offsets, the non-self neighbor
    # row would read [-1, 0, 0] instead of the neighbor's absolute [10, 0, 0].
    coords = torch.tensor([[11.0, 0.0, 0.0], [10.0, 0.0, 0.0], [13.0, 0.0, 0.0]])
    out = ball_query(coords, radius=100.0, k=2)
    # Query index 0 (x=11): nearest two are self (x=11) then x=10 (dist 1).
    assert torch.allclose(out[0, 0], torch.tensor([11.0, 0.0, 0.0]))
    assert torch.allclose(out[0, 1], torch.tensor([10.0, 0.0, 0.0]))


def test_ball_query_self_inclusion_at_distance_zero() -> None:
    # (e) Every query point is its own nearest neighbor (distance 0), so
    # slot 0 must always equal the query point's own coordinates.
    out = ball_query(_LINE, radius=100.0, k=1)
    assert torch.allclose(out[:, 0, :], _LINE)


def test_standardize_coords_exact_zero_mean_and_unit_rms() -> None:
    # (f) Per-axis zero mean AND exact unit RMS by construction (population
    # RMS pooled over ALL elements, a single scalar divisor).
    torch.manual_seed(0)
    coords = torch.randn(20, 3) * 5.0 + torch.tensor([100.0, -50.0, 3.0])
    g = standardize_coords(coords)
    assert torch.allclose(g.mean(dim=0), torch.zeros(3), atol=1e-5)
    assert torch.isclose(g.pow(2).mean().sqrt(), torch.tensor(1.0), atol=1e-5)


def test_standardize_coords_scalar_divisor_preserves_axis_ratio() -> None:
    # (f) The divisor must be a single SCALAR pooled over every element,
    # not a per-axis RMS: a per-axis divisor would equalize spread across
    # axes and distort the point cloud's shape. Note this is NOT caught by
    # the pooled-unit-RMS check above -- a per-axis-normalized result also
    # has pooled RMS exactly 1 (each axis individually hits unit RMS, so
    # the pool of three axis-each-unit-RMS values is still 1); axis-ratio
    # preservation is the property that actually distinguishes the two.
    coords = torch.tensor([[-10.0, -1.0, 0.0], [10.0, 1.0, 0.0]])
    g = standardize_coords(coords)
    # Raw x has 10x the spread of raw y; a scalar divisor preserves that
    # ratio exactly.
    assert torch.isclose(g[1, 0] / g[1, 1], torch.tensor(10.0), atol=1e-4)


def test_standardize_coords_translation_and_scale_invariant() -> None:
    # (f) A per-axis-centered, scalar-isotropic-divisor standardization is
    # invariant to translating or uniformly scaling the raw input.
    torch.manual_seed(1)
    coords = torch.randn(15, 3)
    shifted = coords + torch.tensor([7.0, -3.0, 2.0])
    scaled = coords * 4.0
    g0 = standardize_coords(coords)
    g1 = standardize_coords(shifted)
    g2 = standardize_coords(scaled)
    assert torch.allclose(g0, g1, atol=1e-5)
    assert torch.allclose(g0, g2, atol=1e-5)


def test_standardize_coords_degenerate_all_identical_points_no_nan() -> None:
    # (f) All-identical points -> zero RMS before the clamp; the clamp_min
    # must prevent a 0/0 NaN.
    coords = torch.full((10, 3), 5.0)
    g = standardize_coords(coords)
    assert torch.isfinite(g).all()
    assert torch.equal(g, torch.zeros(10, 3))
