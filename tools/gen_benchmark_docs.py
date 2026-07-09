"""Regenerate docs/benchmarks.md + the per-benchmark landing pages
(and, optionally, archive card files).

Usage:
    python tools/gen_benchmark_docs.py                 # rewrite index + pages
    python tools/gen_benchmark_docs.py --check         # exit 1 if stale
    python tools/gen_benchmark_docs.py --archive taylor_impact_2d --out DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from structbench.benchmarks import available_benchmarks, get_benchmark
from structbench.benchmarks.render import (
    card_json,
    render_archive_readme,
    render_benchmark_page,
    render_index,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX = REPO_ROOT / "docs" / "benchmarks.md"
PAGES_DIR = REPO_ROOT / "docs" / "benchmarks"


def _targets() -> dict[Path, str]:
    """Every generated markdown file mapped to its expected content."""
    specs = {n: get_benchmark(n) for n in available_benchmarks()}
    out = {INDEX: render_index(list(specs.values()))}
    for name, spec in specs.items():
        out[PAGES_DIR / f"{name}.md"] = render_benchmark_page(spec, name)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--archive", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if args.archive:
        if not args.out:
            print("error: --archive requires --out")
            return 2
        spec = get_benchmark(args.archive)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "README.md").write_text(
            render_archive_readme(spec, args.archive), encoding="utf-8", newline="\n"
        )
        (out / "card.json").write_text(
            card_json(spec.card), encoding="utf-8", newline="\n"
        )
        print(f"wrote {out / 'README.md'} and {out / 'card.json'}")
        return 0

    targets = _targets()
    if args.check:
        stale = [
            p.relative_to(REPO_ROOT)
            for p, text in targets.items()
            if not p.exists() or p.read_text(encoding="utf-8") != text
        ]
        if stale:
            listed = ", ".join(str(p) for p in stale)
            print(f"stale, run tools/gen_benchmark_docs.py: {listed}")
            return 1
        print(f"benchmark docs up to date ({len(targets)} files)")
        return 0
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    for path, text in targets.items():
        path.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {len(targets)} files: {INDEX.name} + {len(targets) - 1} pages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
