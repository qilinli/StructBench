"""``structbench-validate``: canonical cases against a reference-experiment set.

    structbench-validate --list
    structbench-validate --reference NAME --pairs pairs.toml --setup setup.toml
                         --out DIR

Writes ``DIR/<reference>.json`` (the byte-stable record) and ``DIR/<reference>.md``.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

from structbench.validation import compare, reference, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="structbench-validate", description=(__doc__ or "").splitlines()[0]
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="name the shipped reference sets and their tests",
    )
    parser.add_argument("--reference", help="a shipped reference set")
    parser.add_argument(
        "--pairs", type=Path, help="pairs.toml: variants and (test, case) pairs"
    )
    parser.add_argument(
        "--setup", type=Path, help="setup.toml: [setup] strings and caveats"
    )
    parser.add_argument("--out", type=Path, help="where the record and its Markdown go")
    args = parser.parse_args(argv)
    if args.list:
        for name in reference.list_references():
            ref = reference.load_reference(name)
            print(f"{name}: {ref.title}")
            for t in ref.tests:
                print(
                    f"  test {t.id}: {t.material}, L0 {t.L0_mm:g} mm, "
                    f"D0 {t.D0_mm:g} mm, v0 {t.v0_ms:g} m/s, T0 {t.T0_K:g} K"
                )
        return 0
    if not (args.reference and args.pairs and args.setup and args.out):
        parser.error("--reference, --pairs, --setup and --out are required (or --list)")
    try:
        ref = reference.load_reference(args.reference)
        name, variants, pairs = compare.load_pairs(args.pairs)
        if name != args.reference:
            raise compare.PairsError(
                f"{args.pairs.name} names reference {name!r}, not {args.reference!r}"
            )
        setup_raw = tomllib.loads(args.setup.read_text(encoding="utf-8"))
        setup = {str(k): str(v) for k, v in setup_raw.get("setup", {}).items()}
        caveats = tuple(str(c) for c in setup_raw.get("caveats", ()))
        comparison = compare.compare(ref, pairs, variants)
    except (
        reference.ReferenceError,
        compare.PairsError,
        OSError,
        tomllib.TOMLDecodeError,
    ) as exc:
        print(exc, file=sys.stderr)
        return 2
    record = report.build_record(ref, comparison, variants, setup, caveats)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{ref.name}.json").write_bytes(report.to_json(record))
    (args.out / f"{ref.name}.md").write_bytes(
        report.render_markdown(record, ref).encode("utf-8")
    )
    done = sum(r.status == "completed" for r in comparison.results)
    print(
        f"{ref.name}: {done} of {len(comparison.results)} pairs measured -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
