"""Regenerate docs/benchmarks.md (and, optionally, archive card files).

Usage:
    python tools/gen_benchmark_docs.py                 # rewrite docs/benchmarks.md
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
    render_index,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX = REPO_ROOT / "docs" / "benchmarks.md"


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
            render_archive_readme(spec), encoding="utf-8", newline="\n"
        )
        (out / "card.json").write_text(
            card_json(spec.card), encoding="utf-8", newline="\n"
        )
        print(f"wrote {out / 'README.md'} and {out / 'card.json'}")
        return 0

    text = render_index([get_benchmark(n) for n in available_benchmarks()])
    if args.check:
        current = INDEX.read_text(encoding="utf-8") if INDEX.exists() else ""
        if current != text:
            print("docs/benchmarks.md is stale; run tools/gen_benchmark_docs.py")
            return 1
        print("docs/benchmarks.md is up to date")
        return 0
    INDEX.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {INDEX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
