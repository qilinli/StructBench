"""Extract the measured Taylor-test outlines from the vector figures of the source.

    python tools/validation/digitize_zelepugin2023.py <source.pdf> <out.json>

Source: Zelepugin S.A., Cherepanov R.O., Pakhnutova N.V. (2023), Materials 16,
5452, doi:10.3390/ma16155452 (CC BY 4.0). Its Figures 4 and 5 plot
"Calculated (red) and experimental (black) profiles of the external surfaces of
the cylinders" for tests 1-6 of its Table 1. Each panel is a vector form
XObject, so the experimental outline is read exactly from the drawing
commands: tick marks and tick labels give a linear map to cm (least squares;
the worst residual is recorded), and the black polyline is the measured
profile. The red (calculated) curves are not kept: they are the source's
simulation, not a measurement.

Checks: every test's black polyline must be identical in Figures 4 and 5 (the
same measurement drawn twice), and the mapping residual must be below
0.005 cm. The output is a ``validation-reference/1`` set (ADR-0072) whose
``Lf_mm``, ``Rf_mm`` and ``Wf_mm`` are what ``structbench.validation.measures
.taylor`` returns on the stored outline. The PDF is not in the repository.
"""

from __future__ import annotations

import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np

from structbench.validation.measures import taylor

NUM = r"-?[\d.]+"
#: Form XObject -> (figure, test number in the source's Table 1). Figure 4 uses
#: the original JC constants, Figure 5 the optimised ones; the black
#: (experimental) curve is the same in both.
PANELS = {
    363: ("4", 1),
    368: ("4", 2),
    370: ("4", 3),
    480: ("4", 4),
    482: ("4", 5),
    484: ("4", 6),
    546: ("5", 1),
    551: ("5", 2),
    553: ("5", 3),
    555: ("5", 4),
    557: ("5", 5),
    559: ("5", 6),
}
#: The source's Table 1 (initial conditions), with the original source of each test.
TESTS = {
    1: {
        "material": "OFHC Cu",
        "L0_mm": 23.47,
        "D0_mm": 7.62,
        "v0_ms": 210,
        "T0_K": 298,
        "source": "S3",
    },
    2: {
        "material": "ETP Cu",
        "L0_mm": 30.0,
        "D0_mm": 6.0,
        "v0_ms": 188,
        "T0_K": 718,
        "source": "S4",
    },
    3: {
        "material": "OFHC Cu M1",
        "L0_mm": 34.5,
        "D0_mm": 7.8,
        "v0_ms": 162,
        "T0_K": 298,
        "source": "S2",
    },
    4: {
        "material": "OFHC Cu M1",
        "L0_mm": 34.5,
        "D0_mm": 7.8,
        "v0_ms": 167,
        "T0_K": 298,
        "source": "S2",
    },
    5: {
        "material": "OFHC Cu M1",
        "L0_mm": 34.5,
        "D0_mm": 7.8,
        "v0_ms": 225,
        "T0_K": 298,
        "source": "S2",
    },
    6: {
        "material": "OFHC Cu M1",
        "L0_mm": 34.5,
        "D0_mm": 7.8,
        "v0_ms": 316,
        "T0_K": 298,
        "source": "S2",
    },
}
SOURCES = [
    {
        "id": "S1",
        "citation": (
            "Zelepugin S.A., Cherepanov R.O., Pakhnutova N.V. Optimization of "
            "Johnson-Cook constitutive model parameters using the Nesterov "
            "gradient-descent method. Materials 2023, 16, 5452."
        ),
        "doi": "10.3390/ma16155452",
        "licence": "CC BY 4.0",
        "consulted": True,
        "role": (
            "test conditions (Table 1); measured outlines (the black curves of "
            "Figures 4 and 5)"
        ),
    },
    {
        "id": "S2",
        "citation": (
            "Zelepugin S.A., Pakhnutova N.V., Shkoda O.A., Boyangin E.N. Experimental "
            "study of the microhardness and microstructure of a copper specimen using "
            "the Taylor impact test. Metals 2022, 12, 2186."
        ),
        "doi": "10.3390/met12122186",
        "licence": None,
        "consulted": False,
        "role": "the original source of tests 3-6, as S1 reports them",
    },
    {
        "id": "S3",
        "citation": (
            "Wilkins M.L., Guinan M.W. Impact of cylinders on a rigid boundary. "
            "J. Appl. Phys. 1973, 44, 1200-1206."
        ),
        "doi": "10.1063/1.1662328",
        "licence": None,
        "consulted": False,
        "role": "the original source of test 1, as S1 reports it",
    },
    {
        "id": "S4",
        "citation": (
            "Gust W.H. High impact deformation of metal cylinders at elevated "
            "temperatures. J. Appl. Phys. 1982, 53, 3566-3575."
        ),
        "doi": "10.1063/1.331136",
        "licence": None,
        "consulted": False,
        "role": "the original source of test 2, as S1 reports it",
    },
]
CAVEATS = [
    (
        "The measured values reach this set through S1's figures; S1 does not "
        "tabulate them, and S2, S3 and S4 were not consulted directly."
    ),
    (
        "Test 1's measured top edge is drawn with a slight slope; L_f is its "
        "highest point."
    ),
    "Test 2 is ETP copper at 718 K; a room-temperature comparison leaves it out.",
    (
        "Test 6's Figure 5 copy has one extra vertex, with the same top and the same "
        "largest radius as its Figure 4 copy."
    ),
]


def form_streams(pdf: bytes) -> dict[int, bytes]:
    out = {}
    for m in re.finditer(rb"(\d+) 0 obj", pdf):
        obj = int(m.group(1))
        if obj not in PANELS:
            continue
        start = pdf.find(b"stream", m.end())
        body = start + 6 + (2 if pdf[start + 6 : start + 8] == b"\r\n" else 1)
        out[obj] = zlib.decompress(pdf[body : pdf.find(b"endstream", body)])
    return out


def parse(stream: bytes) -> dict:
    tokens = re.findall(
        r"\[\([^)]*\)\]|\([^)]*\)|/?[A-Za-z*\"']+|" + NUM, stream.decode("latin-1")
    )
    stack: list[str] = []
    color, current, tm = "black", [], (0.0, 0.0)
    segments, paths, labels = [], [], []
    for tok in tokens:
        if re.fullmatch(NUM, tok):
            stack.append(tok)
            continue
        if tok == "m":
            current = [(float(stack[-2]), float(stack[-1]))]
        elif tok == "l":
            current.append((float(stack[-2]), float(stack[-1])))
        elif tok == "S":
            (segments if len(current) == 2 else paths).append((color, current))
            current = []
        elif tok == "RG":
            color = "red" if [float(s) for s in stack[-3:]] == [1, 0, 0] else "other"
        elif tok == "G":
            color = "black" if float(stack[-1]) == 0.0 else "other"
        elif tok == "Q":
            color = "black"
        elif tok == "Tm":
            tm = (float(stack[-2]), float(stack[-1]))
        elif tok[0] in "[(":
            try:
                labels.append((tm, float(tok.strip("[]()"))))
            except ValueError:
                pass
        stack = []
    return {
        "segments": segments,
        "paths": [p for p in paths if len(p[1]) > 2],
        "labels": labels,
    }


def axis_map(seg: list, labels: list) -> dict[str, tuple[float, float, float]]:
    horiz = [p for _, p in seg if abs(p[0][1] - p[1][1]) < 1e-6]
    vert = [p for _, p in seg if abs(p[0][0] - p[1][0]) < 1e-6]
    xaxis = max(horiz, key=lambda p: abs(p[1][0] - p[0][0]))
    yaxis = max(vert, key=lambda p: abs(p[1][1] - p[0][1]))
    y0, x0 = xaxis[0][1], yaxis[0][0]
    xt = [p for p in vert if abs(p[0][1] - y0) < 1e-6 and p is not yaxis]
    yt = [p for p in horiz if abs(p[0][0] - x0) < 1e-6 and p is not xaxis]
    xl = max(abs(p[1][1] - p[0][1]) for p in xt)
    yl = max(abs(p[1][0] - p[0][0]) for p in yt)
    xmaj = sorted(p[0][0] for p in xt if abs(abs(p[1][1] - p[0][1]) - xl) < 1e-3)
    ymaj = sorted(p[0][1] for p in yt if abs(abs(p[1][0] - p[0][0]) - yl) < 1e-3)

    def row(items, key):
        counts: dict[float, int] = {}
        for t in items:
            counts[round(key(t), 1)] = counts.get(round(key(t), 1), 0) + 1
        best = max(counts, key=lambda k: counts[k])
        return [t for t in items if round(key(t), 1) == best]

    below = row([t for t in labels if t[0][1] < y0], lambda t: t[0][1])
    left = row([t for t in labels if t[0][0] < x0], lambda t: t[0][0])
    xv = [v for _, v in sorted(below, key=lambda t: t[0][0])]
    yv = [v for _, v in sorted(left, key=lambda t: t[0][1])]
    out = {}
    for name, pos, val in (("x", xmaj, xv), ("y", ymaj, yv)):
        assert len(pos) == len(val), (name, len(pos), len(val))
        a, b = np.polyfit(pos, val, 1)
        residual = float(np.abs(np.polyval([a, b], pos) - np.array(val)).max())
        out[name] = (float(a), float(b), residual)
    return out


def main(pdf_path: Path, out_path: Path) -> None:
    streams = form_streams(pdf_path.read_bytes())
    curves_cm: dict[int, list[list[float]]] = {}
    worst = 0.0
    for obj, (_fig, test) in sorted(PANELS.items()):
        parsed = parse(streams[obj])
        amap = axis_map(parsed["segments"], parsed["labels"])
        (ax, bx, rx), (ay, by, ry) = amap["x"], amap["y"]
        worst = max(worst, rx, ry)
        black = [
            [round(ax * x + bx, 5), round(ay * y + by, 5)]
            for c, pts in parsed["paths"]
            if c == "black"
            for x, y in pts
        ]
        if test in curves_cm:
            # Same measurement drawn twice: equal to 1e-3 cm (test 6's Fig. 5
            # copy has one extra vertex, with the same top and largest radius).
            a, b = np.array(curves_cm[test]), np.array(black)
            assert abs(a[:, 1].max() - b[:, 1].max()) < 1e-3, f"test {test}: Lf differs"
            assert abs(a[:, 0].max() - b[:, 0].max()) < 1e-3, f"test {test}: Rf differs"
            if a.shape == b.shape:
                assert np.abs(a - b).max() < 1e-3, f"test {test}: Figs 4/5 differ"
            continue  # keep the Figure 4 copy
        curves_cm[test] = black
    assert worst < 0.005, worst

    tests = []
    for number, cond in TESTS.items():
        rz = np.array(curves_cm[number], dtype=np.float64) * 10.0  # cm -> mm
        rz[:, 1] -= rz[:, 1].min()  # z from the impact face
        rz = np.round(rz, 4)
        lf = taylor.final_length(rz)
        tests.append(
            {
                "id": str(number),
                "source": cond["source"],
                "via": "S1",
                "material": cond["material"],
                "L0_mm": cond["L0_mm"],
                "D0_mm": cond["D0_mm"],
                "v0_ms": cond["v0_ms"],
                "T0_K": cond["T0_K"],
                "outline_rz_mm": rz.tolist(),
                "Lf_mm": lf,
                "Rf_mm": taylor.largest_radius(rz),
                "Wf_mm": taylor.lateral_radii(rz, lf, taylor.FRACTIONS),
            }
        )
    data = {
        "format": "validation-reference/1",
        "name": "taylor_copper",
        "family": "taylor_rod",
        "title": (
            "Copper Taylor impact tests (Wilkins & Guinan 1973; Gust 1982; Zelepugin "
            "et al. 2022), as compiled by Zelepugin, Cherepanov & Pakhnutova 2023"
        ),
        "units": {"length": "mm", "velocity": "m/s", "temperature": "K"},
        "sources": SOURCES,
        "extraction": {
            "tool": "tools/validation/digitize_zelepugin2023.py",
            "method": (
                "The black polylines of S1's Figures 4 and 5 (vector XObjects) are "
                "read from the PDF drawing commands and mapped to length by a "
                "least-squares fit to the major tick marks and their labels; the "
                "outline is shifted so that its lowest point is z = 0 and rounded to "
                "1e-4 mm."
            ),
            "checks": [
                f"worst tick-map residual {worst:.1e} cm (limit 0.005 cm)",
                (
                    "each test's curve is drawn in both figures and the two agree to "
                    "0.001 cm"
                ),
            ],
            "date": "2026-09-25",
        },
        "measures": {
            "fractions": list(taylor.FRACTIONS),
            "definitions": {
                "Lf_mm": "final length: the outline's extent along the axis",
                "Rf_mm": "largest radius anywhere on the outline",
                "Wf_mm": (
                    "lateral radius: the largest r where the outline crosses height "
                    "f * Lf, for each fraction f"
                ),
            },
        },
        "tests": tests,
        "caveats": CAVEATS,
    }
    text = json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    out_path.write_bytes(text.encode("utf-8"))
    print(f"tick-map residual (worst) {worst:.1e} cm; Figs 4 and 5 agree")
    for t in tests:
        print(
            f"test {t['id']}: L0 {t['L0_mm']} v0 {t['v0_ms']}: Lf {t['Lf_mm']:.3f} mm "
            f"(Lf/L0 {t['Lf_mm'] / t['L0_mm']:.4f}), Rf {t['Rf_mm']:.3f} mm "
            f"(Rf/R0 {2 * t['Rf_mm'] / t['D0_mm']:.3f})"
        )
    print(f"-> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
