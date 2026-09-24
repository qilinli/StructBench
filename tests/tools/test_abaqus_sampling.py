"""Tests for the shared Sobol sampler (ADR-0069). Toy variables only."""

import abaqus_paths  # noqa: F401
import pytest

pytest.importorskip("scipy")

import sampling  # noqa: E402

VARIABLES = {"a": (0.0, 1.0), "b": (10.0, 20.0)}
REGIONS = {"corner": {"a": (0.8, 1.0), "b": (18.0, 20.0)}}


def _split(**raw):
    return sampling.parse_splits({"splits": {"s": raw}})[0]


def test_same_seed_same_points_and_nested_prefixes():
    small = sampling.sample_split(VARIABLES, REGIONS, _split(n=10, seed=3))
    large = sampling.sample_split(VARIABLES, REGIONS, _split(n=20, seed=3))
    again = sampling.sample_split(VARIABLES, REGIONS, _split(n=10, seed=3))
    assert [p.params for p in small] == [p.params for p in again]
    assert [p.params for p in small] == [p.params for p in large[:10]]
    assert [p.index for p in large] == list(range(20))


def test_exclude_filters_the_region_out():
    pts = sampling.sample_split(
        VARIABLES, REGIONS, _split(n=50, seed=1, exclude=["corner"])
    )
    assert not any(p.params["a"] >= 0.8 and p.params["b"] >= 18.0 for p in pts)


def test_within_draws_inside_the_box():
    pts = sampling.sample_split(
        VARIABLES, REGIONS, _split(n=60, seed=2, within="corner")
    )
    assert len(pts) == 60
    assert all(
        0.8 <= p.params["a"] <= 1.0 and 18.0 <= p.params["b"] <= 20.0 for p in pts
    )


def test_extra_categorical_and_explicit_points():
    pts = sampling.sample_split(
        VARIABLES,
        REGIONS,
        _split(
            points=[{"a": 0.5, "b": 15.0, "c": 2.0}, {"a": 0.1, "b": 11.0, "c": 3.0}],
            extra={"c": [1.0, 5.0]},
            categorical={"kind": ["x", "y"]},
        ),
    )
    assert [p.params["kind"] for p in pts] == ["x", "y"]
    assert pts[1].params == {"a": 0.1, "b": 11.0, "c": 3.0, "kind": "y"}


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({"n": 5, "seed": 1, "exclude": ["nowhere"]}, "unknown region"),
        ({"points": [{"a": 0.5}]}, "missing"),
        ({"n": 5}, "needs a seed"),
        ({"n": 5, "seed": 1, "exclude": ["corner"], "within": "corner"}, "both"),
        ({"points": [{"a": 0.9, "b": 19.0}], "exclude": ["corner"]}, "only 0 of 1"),
    ],
)
def test_misconfigured_splits_raise_naming_the_split(raw, match):
    with pytest.raises((KeyError, ValueError), match=match):
        sampling.sample_split(VARIABLES, REGIONS, _split(**raw))


def test_region_naming_unknown_variable_raises():
    regions = {"bad": {"zz": (0.0, 1.0)}}
    with pytest.raises(KeyError, match="unknown variables"):
        sampling.sample_split(VARIABLES, regions, _split(n=2, seed=1, exclude=["bad"]))


def test_bounds_must_be_increasing():
    with pytest.raises(ValueError, match="low < high"):
        sampling.parse_bounds({"a": [1.0, 1.0]}, "variables")
