#!/usr/bin/env python
"""Add a new ``[model]`` knob to every transolver config + the test fixture.

Kills the strict-loader tax: adding a ``TransolverConfig`` field requires the key
in ALL ``configs/*/transolver*.toml`` and the ``VALID_TRANSOLVER`` fixture, or
the loader rejects them. This inserts ``<name> = <default>`` at the end of each
``[model]`` table (idempotent). It does NOT touch the dataclass — add the field
to ``TransolverConfig`` yourself first.

Usage:
    python tools/add_transolver_knob.py temperature_phi false
    python tools/add_transolver_knob.py phi_neighbors 16
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "cli" / "test_train_config.py"


def _has_key(lines: list[str], key: str) -> bool:
    return any(
        ln.strip().startswith(f"{key} ") or ln.strip().startswith(f"{key}=")
        for ln in lines
    )


def insert_into_model_table(text: str, line: str, key: str) -> str | None:
    """Insert ``line`` at the end of the ``[model]`` table; None if key present."""
    lines = text.splitlines()
    if _has_key(lines, key):
        return None
    out: list[str] = []
    in_model = False
    inserted = False
    for ln in lines:
        stripped = ln.strip()
        if in_model and stripped.startswith("[") and not inserted:
            out.append(line)  # end of [model] -> insert before the next section
            inserted = True
        if stripped.startswith("[model]"):
            in_model = True
        out.append(ln)
    if in_model and not inserted:  # [model] ran to EOF
        out.append(line)
    tail = "\n" if text.endswith("\n") else ""
    return "\n".join(out) + tail


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    name, default = sys.argv[1], sys.argv[2]
    line = f"{name} = {default}"
    n = 0
    for fp in sorted(glob.glob(str(REPO / "configs" / "*" / "transolver*.toml"))):
        p = Path(fp)
        new = insert_into_model_table(p.read_text(), line, name)
        if new is not None:
            p.write_text(new)
            n += 1
    if FIXTURE.exists():
        s = FIXTURE.read_text()
        marker = "time_conditioned = false\n"
        if line not in s and marker in s:
            FIXTURE.write_text(s.replace(marker, marker + line + "\n", 1))
            print(f"fixture: added {name}")
    print(f"added `{line}` to {n} transolver configs")
    print("REMINDER: add the field to TransolverConfig (config.py) and thread it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
