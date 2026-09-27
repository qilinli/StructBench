"""Scrambled-Sobol sampling of a sweep's splits, shared by every Abaqus dataset.

A split draws 1024 points from its own seeded, scrambled Sobol engine and keeps
the first ``n`` that survive its ``exclude`` regions, so the first ``k`` cases
of a split are nested subsets. A ``within`` split draws inside its region's box
rather than filtering, because a small box filtered from a global draw leaves
too few points. Explicit ``points`` bypass the engine (pilots, probes), and
also the dataset's optional feasibility limit, which filters Sobol points the
way ``exclude`` does.
``categorical`` values are assigned by cycling through the list by index;
``variants`` are expanded by the generator, not here (ADR-0069).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import qmc

N_DRAW_LOG2 = 10  # 1024 points per split

Bounds = tuple[float, float]


@dataclass(frozen=True)
class Split:
    name: str
    n: int
    seed: int | None = None
    exclude: tuple[str, ...] = ()
    within: str | None = None
    extra: dict[str, Bounds] = field(default_factory=dict)
    categorical: dict[str, tuple[str, ...]] = field(default_factory=dict)
    variants: tuple[str, ...] = ()
    points: tuple[dict[str, float], ...] = ()


@dataclass(frozen=True)
class Point:
    split: str
    index: int
    params: dict[str, float | str]


def parse_bounds(raw: Mapping[str, Any], where: str) -> dict[str, Bounds]:
    """``{"a": [0, 1]}`` -> ``{"a": (0.0, 1.0)}``; each range must increase."""
    out: dict[str, Bounds] = {}
    for name, pair in raw.items():
        low, high = (float(v) for v in pair)
        if not low < high:
            raise ValueError(f"{where}.{name}: need low < high, got {pair!r}")
        out[name] = (low, high)
    return out


def parse_splits(sweep: Mapping[str, Any]) -> list[Split]:
    """The ``[splits.*]`` tables of a parsed sweep.toml, in file order."""
    splits = []
    for name, raw in sweep["splits"].items():
        points = tuple(
            {k: float(v) for k, v in p.items()} for p in raw.get("points", ())
        )
        if points and "seed" in raw:
            raise ValueError(f"splits.{name}: explicit points take no seed")
        if not points and "seed" not in raw:
            raise ValueError(f"splits.{name}: a sampled split needs a seed")
        if raw.get("exclude") and raw.get("within"):
            raise ValueError(f"splits.{name}: set exclude or within, not both")
        n = len(points) if points else int(raw["n"])
        if not 0 < n <= 2**N_DRAW_LOG2:
            raise ValueError(f"splits.{name}: n must be in 1..{2**N_DRAW_LOG2}")
        splits.append(
            Split(
                name=name,
                n=n,
                seed=raw.get("seed"),
                exclude=tuple(raw.get("exclude", ())),
                within=raw.get("within"),
                extra=parse_bounds(raw.get("extra", {}), f"splits.{name}.extra"),
                categorical={
                    k: tuple(v) for k, v in raw.get("categorical", {}).items()
                },
                variants=tuple(raw.get("variants", ())),
                points=points,
            )
        )
    return splits


def _inside(params: Mapping[str, float | str], region: Mapping[str, Bounds]) -> bool:
    return all(low <= float(params[k]) <= high for k, (low, high) in region.items())


def sample_split(
    variables: Mapping[str, Bounds],
    regions: Mapping[str, Mapping[str, Bounds]],
    split: Split,
    feasible: Callable[[dict[str, float | str]], bool] | None = None,
) -> list[Point]:
    """The split's points, in Sobol (or listed) order, indexed from 0.

    ``feasible`` is the dataset's declared solver-feasibility limit: a Sobol
    point it rejects is skipped like an excluded one, so prefixes stay nested.
    Explicit points bypass it -- a listed point is deliberate.
    """
    box = {**variables, **split.extra}
    named = (*split.exclude, *((split.within,) if split.within else ()))
    for region in named:
        if region not in regions:
            raise KeyError(f"splits.{split.name}: unknown region {region!r}")
        unknown = set(regions[region]) - set(box)
        if unknown:
            raise KeyError(
                f"region {region!r} names unknown variables {sorted(unknown)}"
            )
    if split.points:
        rows = [dict(p) for p in split.points]
        for row in rows:
            missing = set(box) - set(row)
            if missing:
                raise ValueError(
                    f"splits.{split.name}: point missing {sorted(missing)}"
                )
    else:
        if split.within:
            box.update(regions[split.within])
        names = list(box)
        engine = qmc.Sobol(
            d=len(names), scramble=True, rng=np.random.default_rng(split.seed)
        )
        scaled = qmc.scale(
            engine.random_base2(m=N_DRAW_LOG2),
            [box[k][0] for k in names],
            [box[k][1] for k in names],
        )
        rows = [{k: float(v) for k, v in zip(names, r, strict=True)} for r in scaled]
    points: list[Point] = []
    for row in rows:
        if any(_inside(row, regions[r]) for r in split.exclude):
            continue
        index = len(points)
        params: dict[str, float | str] = dict(row)
        for key, values in split.categorical.items():
            params[key] = values[index % len(values)]
        if feasible is not None and not split.points and not feasible(params):
            continue
        points.append(Point(split.name, index, params))
        if len(points) == split.n:
            break
    if len(points) < split.n:
        raise ValueError(
            f"splits.{split.name}: only {len(points)} of {split.n} points "
            "survive its regions"
        )
    return points
