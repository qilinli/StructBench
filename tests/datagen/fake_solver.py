"""A fake Abaqus for the preflight's end-to-end tests: no solver, no odbAccess.

It answers the three command lines the pipeline issues, like the executable:

    python fake_solver.py job=<id> input=<id>.inp double=both cpus=1 interactive
        reads provenance.json and the deck in the working directory, writes
        <id>.sta, <id>.msg, <id>.dat, an empty <id>.odb and <id>.npz in the
        abaqus-npz/1 layout (the export the real exporter would write);
    python fake_solver.py --sweep <dir> [--cases ...]
        the exporter's place: exit 0, the export already exists;
    python fake_solver.py terminate job=<id>
        exit 0.

The response is analytic and chosen so that a sound preflight passes: with
k the refinement factor and bias = 1 + 0.02 / k**2, the rod shortens by
delta = 0.1 L (v0 / 1e5) bias and its face widens by 0.1 R bias, following
s(t) = 1 - exp(-t / tau) with tau = period / 10 (second-order convergence in
1/k, settled by a quarter of the horizon); kinetic energy decays into
internal energy with the total constant, no term outside the ledger
identity is non-zero, and the wall reaction is a half-sine pulse over the
first 30 % of the horizon. FAKE_SOLVER_SETTLE=slow sets tau = period, so
the response never settles and the duration step fails.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

STA = " Abaqus/Explicit 2025\n THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n"
MSG = " Abaqus 2025\n      0 ERROR MESSAGES\n      0 WARNING MESSAGES\n"
DAT = " Abaqus 2025\n"
INSTANCE = "ROD-1"
STEP = "IMPACT"
TERMS = ("ALLAE", "ALLCD", "ALLFD", "ALLIE", "ALLKE", "ALLPD", "ALLPW", "ALLSE")
TERMS += ("ALLVD", "ALLWK", "ETOTAL")


def parse_deck(text: str):
    nodes: dict[int, tuple[float, float]] = {}
    elements: list[tuple[int, list[int]]] = []
    interval = period = None
    block = None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("*"):
            up = line.upper()
            block = None
            if up.startswith(("*NODE OUTPUT", "*ELEMENT OUTPUT")):
                continue
            if up.startswith("*NODE"):
                block = "node"
            elif up.startswith("*ELEMENT"):
                block = "element"
            elif up.startswith("*OUTPUT, FIELD"):
                m = re.search(r"TIME INTERVAL=([0-9.eE+-]+)", up)
                interval = float(m.group(1)) if m else None
            elif up.startswith("*DYNAMIC, EXPLICIT"):
                period = float(lines[i + 1].split(",")[1])
            continue
        if block == "node":
            parts = [p.strip() for p in line.split(",")]
            nodes[int(parts[0])] = (float(parts[1]), float(parts[2]))
        elif block == "element":
            parts = [int(p) for p in line.split(",")]
            elements.append((parts[0], parts[1:]))
    return nodes, elements, interval, period, "VARIABLE=ALL" in text.upper()


def solve(job: str, here: Path) -> None:
    prov = json.loads((here / "provenance.json").read_text(encoding="utf-8"))
    params = prov["params"]
    nodes, elements, interval, period, all_energy = parse_deck(
        (here / f"{job}.inp").read_text(encoding="utf-8")
    )
    assert interval is not None and period is not None
    labels = np.array(sorted(nodes), dtype=np.int64)
    coords = np.array([nodes[int(lab)] for lab in labels], dtype=float)
    elem_labels = np.array([e[0] for e in elements], dtype=np.int64)
    conn = np.array([e[1] for e in elements], dtype=np.int64)
    field_nodes = np.unique(conn)
    (rp,) = [int(lab) for lab in labels if int(lab) not in set(field_nodes.tolist())]
    n_intervals = int(round(period / interval))
    t = np.arange(n_intervals + 1) * interval
    t = np.append(t, t[-1])  # the solver's duplicate end frame

    k = int(params.get("refine", 1))
    length, radius = float(params["L"]), float(params["D"]) / 2.0
    speed = float(params["v0"]) / 1e5
    bias = 1.0 + 0.02 / k**2
    slow = os.environ.get("FAKE_SOLVER_SETTLE") == "slow"
    tau = period if slow else period / 10.0
    s = 1.0 - np.exp(-t / tau)
    sd = np.exp(-t / tau) / tau
    sdd = -np.exp(-t / tau) / tau**2
    delta, eps = 0.1 * length * speed * bias, 0.1 * radius * bias
    rz = coords[np.searchsorted(labels, field_nodes)]
    z = rz[:, 1]
    shape = np.stack([eps * (1.0 - z / length), -delta * (z / length)], -1)
    # The velocity starts at the deck's initial condition (-v0 along z on every
    # rod node) and decays with the motion; it is not the displacement's time
    # derivative, which no judged row requires.
    v0 = float(params["v0"])
    axial = np.stack([np.zeros_like(z), np.ones_like(z)], -1)
    fields = {
        "U": (s[:, None, None] * shape[None]).astype(np.float32),
        "V": (-v0 * tau * sd[:, None, None] * axial[None]).astype(np.float32),
        "A": (-v0 * tau * sdd[:, None, None] * axial[None]).astype(np.float32),
    }
    e = len(elem_labels)
    stress = (200.0 * bias * s)[:, None, None] * np.ones((1, e, 4))
    peeq = (0.01 * s)[:, None, None] * np.ones((1, e, 1))
    k0 = 1000.0 * speed**2
    ke = k0 * np.exp(-t / tau)  # the same time constant as the motion
    ie = k0 - ke
    zero = np.zeros_like(t)
    terms = {
        "ALLKE": ke,
        "ALLIE": ie,
        "ALLPD": 0.8 * ie,
        "ALLSE": 0.2 * ie,
        "ALLAE": zero,
        "ALLCD": zero,
        "ALLFD": zero,
        "ALLPW": zero,
        "ALLVD": zero,
        "ALLWK": zero,
        "ETOTAL": np.full_like(t, k0),
    }
    if all_energy:
        terms["ALLDMD"] = zero  # a term VARIABLE=ALL adds; zero, as it must be
    t_contact = 0.3 * period
    force = np.where(t < t_contact, np.sin(np.pi * t / t_contact), 0.0) * 1e4 * speed

    manifest = {
        "format": "abaqus-npz/1",
        "odb_sha256": "0" * 64,
        "abaqus_release": "Abaqus/Explicit 2025 (fake)",
        "precision": "DOUBLE_PRECISION",
        "materials": ["ROD_MATERIAL"],
        "sections": ["ROD"],
        "fields": {},
        "skipped": [],
    }
    out = {
        "manifest": np.array(json.dumps(manifest)),
        f"mesh/{INSTANCE}/node_labels": labels,
        f"mesh/{INSTANCE}/node_coords": np.column_stack(
            [coords, np.zeros(len(coords))]
        ),
        f"mesh/{INSTANCE}/elements/CAX4R/labels": elem_labels,
        f"mesh/{INSTANCE}/elements/CAX4R/connectivity": conn,
        f"step/{STEP}/frame_times": t,
    }
    for name, data in fields.items():
        out[f"field/{STEP}/{name}/{INSTANCE}/data"] = data
        out[f"field/{STEP}/{name}/{INSTANCE}/node_labels"] = field_nodes
    for name, data in (("S", stress), ("PEEQ", peeq)):
        out[f"field/{STEP}/{name}/{INSTANCE}/data"] = data.astype(np.float32)
        out[f"field/{STEP}/{name}/{INSTANCE}/element_labels"] = elem_labels
        out[f"field/{STEP}/{name}/{INSTANCE}/integration_points"] = np.ones(
            e, dtype=np.int64
        )
    for term, series in terms.items():
        out[f"history/{STEP}/Assembly ASSEMBLY/{term}"] = np.stack([t, series], -1)
    out[f"history/{STEP}/Node {INSTANCE}.{rp}/RF2"] = np.stack([t, force], -1)
    np.savez_compressed(here / f"{job}.npz", **out)
    (here / f"{job}.odb").write_bytes(b"")
    (here / f"{job}.sta").write_text(STA, encoding="utf-8")
    (here / f"{job}.msg").write_text(MSG, encoding="utf-8")
    (here / f"{job}.dat").write_text(DAT, encoding="utf-8")


def main(argv: list[str]) -> int:
    if not argv or argv[0] == "terminate" or argv[0].startswith("--"):
        return 0  # terminate, or the exporter's call: nothing to do
    job = next(a.split("=", 1)[1] for a in argv if a.startswith("job="))
    solve(job, Path.cwd())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
