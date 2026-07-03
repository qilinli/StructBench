"""Generate the frozen ADR-0026 splits for the two notch-beam benchmarks.

Run once; paste the printed TRAIN/VAL/TEST_INTERP lists into the benchmark
modules as immutable literals. Seed 26; constraints per ADR-0026: sizes
88/8/12, val+test drawn from interior velocities only, every factor level
present in train. Provenance script — not part of the package (ADR-0010).
"""

from __future__ import annotations

import random

SPANS = (320, 480, 640)
BEND_V, BEND_INTERIOR = (8, 12, 16, 20), (12, 16)
IMPACT_V, IMPACT_INTERIOR = (40, 80, 120, 160), (80, 120)
LOADS, NOTCHES = ("A", "B", "C"), ("a", "b", "c")
SHAPES = ("Bullet", "Rectangular", "Sphere")


def freeze(name: str, cases: list[str], interior: list[str]) -> None:
    rng = random.Random(26)
    held = rng.sample(sorted(interior), 20)
    val, test = sorted(held[:8]), sorted(held[8:])
    train = sorted(c for c in cases if c not in held)
    assert len(train) == 88 and len(val) == 8 and len(test) == 12
    # every factor token of every case-id appears among the train ids
    train_tokens = {tok for c in train for tok in c.split("-")}
    for case in cases:
        assert set(case.split("-")) <= train_tokens or case in held
    print(f"# {name}\nTRAIN = {train!r}\nVAL = {val!r}\nTEST_INTERP = {test!r}\n")


bend = [
    f"NB-B-{s}-{ln}-{v}"
    for s in SPANS
    for v in BEND_V
    for ln in (lo + n for lo in LOADS for n in NOTCHES)
]
bend_interior = [c for c in bend if int(c.rsplit("-", 1)[1]) in BEND_INTERIOR]
freeze("notch_beam_2d_bend", bend, bend_interior)

impact = [
    f"NB-I-{s}-{sh}-{n}-{v}"
    for s in SPANS
    for sh in SHAPES
    for n in NOTCHES
    for v in IMPACT_V
]
impact_interior = [c for c in impact if int(c.rsplit("-", 1)[1]) in IMPACT_INTERIOR]
freeze("notch_beam_2d_impact", impact, impact_interior)
