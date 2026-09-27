"""Copy a validated sweep into the data tree; keep a sample of its ODBs.

    python data_generation/abaqus/archive.py --sweep <work-root>/<name>
        --dataset <dataset-dir> --data-root <data tree>
        [--split NAME ...] [--prune-odb] [--yes]

The cases are those ``validate.py`` recorded in ``<sweep>/datacheck/
measurements.json``. For each, the run record and the export go to
``<data-root>/raw/<name>/abaqus/<id>/`` -- ``<id>.inp``, ``provenance.json``,
``run.json``, ``<id>.sta/.msg/.dat``, ``<id>.npz``, and ``<id>.odb`` when
retained -- and the canonical case to ``<data-root>/canonical/<name>/<id>.h5``.
``runner.log`` is never copied: it holds the licence text. The ``.dat`` and
``.msg`` are copied with their licence header and any licence line removed
(``redact``), and verified against the redacted bytes.

Retention comes from ``[retention]`` in ``<dataset-dir>/sweep.toml``:
about ``odb_fraction`` (default 0.05) of the cases, each decided by a hash of
``odb_seed`` and its id, plus the named ``odb_cases``. A named case that is
not in the sweep, and a retained case whose ODB is gone, are reported.

Without ``--yes`` this prints the plan and writes nothing. With it, each file
is copied through a ``.partial`` name and verified by size and sha256; a file
already at its destination is skipped when identical, and a different one
stops that case with a message -- nothing is ever overwritten, and no local
copy is ever deleted by the copy. ``--prune-odb --yes`` then deletes the local
ODBs that are not retained, only for archived cases whose export manifest
names that ODB's sha256. The data tree is written at the paths above and
never listed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_RUN_FILES = (
    "{id}.inp",
    "provenance.json",
    "run.json",
    "{id}.sta",
    "{id}.msg",
    "{id}.dat",
)


#: Files whose header names the licensee, host, seat or site. They are copied
#: redacted, like ``runner.log`` is not copied at all.
_REDACTED = frozenset({".dat", ".msg"})
#: A line that says something about the licence, wherever it stands.
_LICENCE_LINE = re.compile(
    r"licen[cs]|flexnet|\bseat\b|site id|for use by|on machine|"
    r"authori[sz]ed to run|\btokens?\b|\buntil \d",
    re.IGNORECASE,
)
#: The version line the run-record reader keys on (``Abaqus 2025``).
_VERSION_LINE = re.compile(r"^\s*Abaqus(?:/\w+)?\s+\d{4}\b")
#: The asterisk box that ends the printed file's header.
_BANNER = re.compile(r"^\s*\*(\s|$)")
_MARK = "   [licence header removed by archive.py]\n"


def redact(text: str) -> str:
    """``text`` without its licence header or any licence line.

    Before the first asterisk banner only version lines are kept; after it,
    only lines that do not match ``_LICENCE_LINE``. What the run-record
    reader needs -- the version, the banners, the counts, the markers --
    survives.
    """
    lines = text.splitlines(keepends=True)
    banner = next((i for i, ln in enumerate(lines[:200]) if _BANNER.match(ln)), 0)
    kept = [
        ln
        for i, ln in enumerate(lines)
        if not _LICENCE_LINE.search(ln)
        and (i >= banner or not ln.strip() or _VERSION_LINE.match(ln))
    ]
    return _MARK + "".join(kept)


def _payload(src: Path) -> bytes | None:
    """The bytes to archive for ``src`` when they are not the file's own."""
    if src.suffix not in _REDACTED:
        return None
    return redact(src.read_text(encoding="utf-8", errors="replace")).encode("utf-8")


@dataclass(frozen=True)
class Copy:
    case_id: str
    kind: str  # "run", "npz", "odb" or "h5"
    src: Path
    dst: Path


def retained(
    case_ids: list[str], *, fraction: float, seed: int, named: list[str]
) -> set[str]:
    """Cases whose ODB is kept: about ``fraction`` of them, plus ``named``.

    Each case is decided on its own, by where ``sha256("<seed>:<case id>")``
    falls in [0, 1). The answer for a case therefore never depends on which
    other cases one invocation lists (a ``--split``, a partial record) nor on
    a random-number stream, so a later prune cannot delete an ODB an earlier
    archive meant to keep.
    """
    cut = fraction * 2**256
    drawn = {
        c
        for c in case_ids
        if int(hashlib.sha256(f"{seed}:{c}".encode()).hexdigest(), 16) < cut
    }
    return drawn | (set(named) & set(case_ids))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _case_ids(sweep: Path, splits: list[str] | None) -> list[str]:
    record = json.loads((sweep / "datacheck" / "measurements.json").read_text("utf-8"))
    ids = [c["case_id"] for c in record["cases"]]
    if not splits:
        return ids
    return [
        cid
        for cid in ids
        if json.loads((sweep / cid / "provenance.json").read_text("utf-8")).get("split")
        in splits
    ]


def plan(
    sweep: Path, name: str, data_root: Path, case_ids: list[str], keep: set[str]
) -> list[Copy]:
    """Every copy the archive makes, in case order; missing sources are left out."""
    copies = []
    for cid in case_ids:
        raw = data_root / "raw" / name / "abaqus" / cid
        sources = [("run", sweep / cid / f.format(id=cid)) for f in _RUN_FILES]
        sources.append(("npz", sweep / cid / f"{cid}.npz"))
        if cid in keep:
            sources.append(("odb", sweep / cid / f"{cid}.odb"))
        copies += [Copy(cid, kind, src, raw / src.name) for kind, src in sources]
        h5 = sweep / "canonical" / f"{cid}.h5"
        copies.append(Copy(cid, "h5", h5, data_root / "canonical" / name / h5.name))
    return [c for c in copies if c.src.is_file()]


def _copy_case(copies: list[Copy], counts: Counter[str]) -> str | None:
    """Copy one case's files; the reason it stopped, or ``None``."""
    for c in copies:
        payload = _payload(c.src)
        if payload is None:
            want, size = _sha256(c.src), c.src.stat().st_size
        else:
            want, size = hashlib.sha256(payload).hexdigest(), len(payload)
        if c.dst.exists():
            if c.dst.stat().st_size == size and _sha256(c.dst) == want:
                counts["verified"] += 1
                continue
            return f"mismatch at {c.dst.name}: a different file is there; left as is"
        c.dst.parent.mkdir(parents=True, exist_ok=True)
        partial = c.dst.with_name(c.dst.name + ".partial")
        if payload is None:
            shutil.copyfile(c.src, partial)
        else:
            partial.write_bytes(payload)
        if partial.stat().st_size != size or _sha256(partial) != want:
            partial.unlink()
            return f"copy of {c.src.name} did not verify"
        os.replace(partial, c.dst)
        counts["copied"] += 1
    return None


def _odb_vouched(sweep: Path, cid: str) -> bool:
    """Whether the case's export manifest names this ODB's sha256."""
    odb, npz = sweep / cid / f"{cid}.odb", sweep / cid / f"{cid}.npz"
    if not npz.is_file():
        return False
    with np.load(npz, allow_pickle=False) as z:
        manifest = json.loads(str(z["manifest"][()]))
    return manifest.get("odb_sha256") == _sha256(odb)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", action="append")
    parser.add_argument("--prune-odb", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)

    toml = tomllib.loads((args.dataset / "sweep.toml").read_text(encoding="utf-8"))
    name = toml["dataset"]["name"]
    rules = toml.get("retention", {})
    case_ids = _case_ids(args.sweep, args.split)
    keep = retained(
        case_ids,
        fraction=float(rules.get("odb_fraction", 0.05)),
        seed=int(rules.get("odb_seed", 0)),
        named=list(rules.get("odb_cases", [])),
    )
    copies = plan(args.sweep, name, args.data_root, case_ids, keep)
    unknown = sorted(set(rules.get("odb_cases", [])) - set(case_ids))
    if unknown:
        print(f"odb_cases not among these cases: {', '.join(unknown)}")
    gone = sorted(c for c in keep if not (args.sweep / c / f"{c}.odb").is_file())
    if gone:
        print(f"retained but no local ODB to archive: {', '.join(gone)}")
    prunable = [
        cid
        for cid in case_ids
        if cid not in keep and (args.sweep / cid / f"{cid}.odb").is_file()
    ]

    if not args.yes:
        size: Counter[str] = Counter()
        for c in copies:
            size[c.kind] += c.src.stat().st_size
        print(f"dry run: {len(case_ids)} cases of {name}, {len(keep)} ODBs retained")
        print(f"  -> {args.data_root / 'raw' / name / 'abaqus'}")
        print(f"  -> {args.data_root / 'canonical' / name}")
        for kind in ("run", "npz", "odb", "h5"):
            n = sum(c.kind == kind for c in copies)
            print(f"  {kind}: {n} files, {size[kind] / 1e9:.2f} GB")
        if args.prune_odb:
            freed = sum((args.sweep / c / f"{c}.odb").stat().st_size for c in prunable)
            print(f"  prune: {len(prunable)} local ODBs, {freed / 1e9:.2f} GB")
        print("nothing written; re-run with --yes to copy")
        return 0

    counts: Counter[str] = Counter()
    stopped: dict[str, str] = {}
    by_case: dict[str, list[Copy]] = {}
    for c in copies:
        by_case.setdefault(c.case_id, []).append(c)
    for cid in case_ids:
        reason = _copy_case(by_case.get(cid, []), counts)
        if reason is not None:
            stopped[cid] = reason
    print(
        f"copied={counts['copied']} verified={counts['verified']} "
        f"stopped={len(stopped)}"
    )
    for cid, reason in sorted(stopped.items()):
        print(f"  {cid}: {reason}")
    refused = []
    if args.prune_odb:
        for cid in prunable:
            if cid in stopped:
                continue  # not archived, so not pruned
            if _odb_vouched(args.sweep, cid):
                (args.sweep / cid / f"{cid}.odb").unlink()
                print(f"  pruned {cid}.odb")
            else:
                refused.append(cid)
                print(f"  refused {cid}.odb: its sha256 is not the export manifest's")
    return 1 if stopped or refused else 0


if __name__ == "__main__":
    sys.exit(main())
