"""Stage a release's blessed checkpoints for publication (ADR-0074 clause 8;
ADR-0037 clause 4; plan ``docs/plans/2026-11-09-checkpoint-publishing.md``).

For one benchmark, every blessed registry row whose ``checkpoint`` still
points into the private ``models/`` mirror is verified against the mirror
(the bundle exists, ``SHA256SUMS`` lists the checkpoint, the file's digest
equals the listed one and the registry's ``checkpoint_sha256``) and staged
for upload to the benchmark's Hugging Face *model* repository
``StructBench/<slug>``. What is published is the minimum that runs and
checks the model, not the private bundle (maintainer, 2026-11-10): the
registered checkpoint, ``config.json``, ``normalization_stats.npz`` when the
run has one, the ``metrics-<split>.json`` of the row's scored splits, and a
``SHA256SUMS`` of those files. Logs, ``provenance/``, rollouts and the
unregistered checkpoints stay private, and local paths are removed from the
JSON files. Staging writes that folder per row, a generated model-card
``README.md``, a ``publish.sh`` holding the ``hf`` commands (create the
repo, upload each staged folder, upload the README, flip public, tag with
the code release), and ``pointers.json`` mapping each row to its public URL
``https://huggingface.co/StructBench/<slug>/resolve/<tag>/<bundle>/<file>``.
The release session edits the registries from ``pointers.json`` and
regenerates the pages; the landing pages then drop the private-archive
marker by themselves (ADR-0037).

The upload is the maintainer's: ADR-0023's on-instruction tier names the
dataset repos, not model repos, so this tool writes the commands and runs
nothing against Hugging Face. Not part of the importable package
(ADR-0010).

Usage::

    python tools/publish_checkpoints.py --benchmark taylor_impact_2d \
        --models-root <...>/data/StructBench/models --out scratch/hf-models \
        --tag v0.4.0 [--include-provisional] [--check] \
        [--run-dir <bundle>=<run directory>]
    bash scratch/hf-models/taylor_impact_2d/publish.sh
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from structbench.benchmarks.registry import BenchmarkSpec, get_benchmark
from structbench.benchmarks.results import BaselineResult

#: Hugging Face model repo per benchmark (``StructBench/<slug>``): the dataset
#: repos' slugs (``tools/build_hf_bundle.py``) plus DeformingPlate, whose data
#: is not rehosted (ADR-0042) but whose checkpoint is StructBench's own.
MODEL_REPOS: dict[str, str] = {
    "taylor_impact_2d": "taylor-impact-2d",
    "wave_propagation_1d": "wave-propagation-1d",
    "notch_beam_2d_impact": "notch-beam-2d-impact",
    "deforming_plate": "deforming-plate",
}
HF_ORG = "StructBench"
_REPO_URL = "https://github.com/qilinli/StructBench"


@dataclass(frozen=True)
class BundleCheck:
    """What the mirror says about one registry row's checkpoint."""

    bundle_dir: Path
    checkpoint: Path
    digest: str | None
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass(frozen=True)
class ReleaseRow:
    """One row staged for publication."""

    label: str
    result: BaselineResult
    bundle: str
    filename: str
    check: BundleCheck
    url: str


@dataclass(frozen=True)
class ReleasePlan:
    """Every row of a benchmark the release stages, and the ones it skips."""

    benchmark: str
    tag: str
    models_root: Path
    rows: tuple[ReleaseRow, ...]
    skipped: dict[str, str]

    @property
    def ok(self) -> bool:
        return all(r.check.ok for r in self.rows)


def parse_pointer(pointer: str) -> tuple[str, str, str]:
    """``models/<benchmark>/<bundle>/<file>`` -> its three parts.

    Raises ``ValueError`` for a URL or any other shape: only an
    archive-relative pointer (ADR-0037 clause 4) is publishable.
    """
    parts = pointer.split("/")
    if len(parts) != 4 or parts[0] != "models":
        raise ValueError(
            f"{pointer!r} is not an archive-relative checkpoint pointer "
            "(models/<benchmark>/<bundle>/<file>)"
        )
    return parts[1], parts[2], parts[3]


def read_sha256sums(bundle_dir: Path) -> dict[str, str]:
    """The bundle's ``SHA256SUMS`` (``sha256sum`` format, ``./``-relative
    paths) as ``{path: digest}``; empty when the file is missing."""
    path = bundle_dir / "SHA256SUMS"
    if not path.is_file():
        return {}
    sums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        sums[name.strip().removeprefix("./")] = digest.strip()
    return sums


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_bundle(
    models_root: Path,
    result: BaselineResult,
    run_dirs: dict[str, Path] | None = None,
) -> BundleCheck:
    """Check a row's bundle on the mirror: present, listed, digest consistent.

    Every failure is a named problem; a registry row without a digest is one
    (the pointer would be unverifiable in public). ``run_dirs`` maps a bundle
    name to the run directory to read in place of the mirror, for a row whose
    private bundle is not assembled: such a directory needs no
    ``SHA256SUMS``, and the registry digest alone identifies the checkpoint.
    """
    assert result.checkpoint is not None
    benchmark, bundle, filename = parse_pointer(result.checkpoint)
    run_dir = (run_dirs or {}).get(bundle)
    from_run_dir = run_dir is not None
    bundle_dir = run_dir if run_dir is not None else models_root / benchmark / bundle
    checkpoint = bundle_dir / filename
    problems: list[str] = []
    if not bundle_dir.is_dir():
        problems.append(f"bundle {bundle_dir} is not on the mirror")
        return BundleCheck(bundle_dir, checkpoint, None, problems)
    if not checkpoint.is_file():
        problems.append(f"checkpoint {filename} is not in bundle {bundle}")
        return BundleCheck(bundle_dir, checkpoint, None, problems)
    digest = _sha256(checkpoint)
    listed = read_sha256sums(bundle_dir)
    if not (bundle_dir / "config.json").is_file():
        problems.append(f"bundle {bundle} has no config.json")
    if from_run_dir and not listed:
        pass
    elif not listed:
        problems.append(f"bundle {bundle} has no SHA256SUMS")
    elif filename not in listed:
        problems.append(f"SHA256SUMS of bundle {bundle} does not list {filename}")
    elif listed[filename] != digest:
        problems.append(
            f"SHA256SUMS of bundle {bundle} lists {listed[filename][:12]}… for "
            f"{filename}, the file's digest is {digest[:12]}…"
        )
    if result.checkpoint_sha256 is None:
        problems.append(f"registry row {result.label!r} has no checkpoint_sha256")
    elif result.checkpoint_sha256 != digest:
        problems.append(
            f"registry digest {result.checkpoint_sha256[:12]}… for "
            f"{result.label!r} differs from the file's {digest[:12]}…"
        )
    return BundleCheck(bundle_dir, checkpoint, digest, problems)


def public_url(benchmark: str, tag: str, bundle: str, filename: str) -> str:
    """The immutable public pointer of a published checkpoint."""
    return (
        f"https://huggingface.co/{HF_ORG}/{MODEL_REPOS[benchmark]}/resolve/{tag}/"
        f"{bundle}/{filename}"
    )


def plan_release(
    benchmark: str,
    spec: BenchmarkSpec,
    models_root: Path,
    tag: str,
    *,
    include_provisional: bool = False,
    run_dirs: dict[str, Path] | None = None,
) -> ReleasePlan:
    """The rows of ``spec`` (registered as ``benchmark``) to publish under
    ``tag``, verified on the mirror.

    Blessed rows with an archive-relative pointer are staged; provisional
    rows only with ``include_provisional``; a row whose pointer is already a
    URL, or has none, is skipped and said so.
    """
    if benchmark not in MODEL_REPOS:
        raise ValueError(f"no model repo for benchmark {benchmark!r}")
    name = benchmark
    rows: list[ReleaseRow] = []
    skipped: dict[str, str] = {}
    for result in spec.results:
        if result.checkpoint is None:
            skipped[result.label] = "no checkpoint pointer"
            continue
        if result.checkpoint.startswith(("http://", "https://")):
            skipped[result.label] = "already public"
            continue
        if result.provisional and not include_provisional:
            skipped[result.label] = "provisional"
            continue
        benchmark, bundle, filename = parse_pointer(result.checkpoint)
        rows.append(
            ReleaseRow(
                label=result.label,
                result=result,
                bundle=bundle,
                filename=filename,
                check=verify_bundle(models_root, result, run_dirs),
                url=public_url(benchmark, tag, bundle, filename),
            )
        )
    return ReleasePlan(name, tag, models_root, tuple(rows), skipped)


def _headline(result: BaselineResult) -> str:
    for split, values in result.metrics.items():
        for key, value in values.items():
            if "rel_l2" in key and "disp" in key:
                return f"{split} {key} {value:.4g}"
    return "—"


def citation_section() -> list[str]:
    """The model card's Citation section, read from the repo's ``CITATION.cff``.

    The same entry the dataset cards carry (``tools/build_hf_bundle.py``):
    the software is what is cited, with the archive's all-versions DOI when
    the file has one. Parsed line-wise, without a YAML dependency.
    """
    text = (Path(__file__).resolve().parents[1] / "CITATION.cff").read_text(
        encoding="utf-8"
    )

    def field(pattern: str, default: str) -> str:
        m = re.search(pattern, text, re.M)
        return m.group(1).strip('"') if m else default

    names = re.findall(r"family-names:\s*(.+)\n\s*given-names:\s*(.+)", text)
    authors = " and ".join(f"{f.strip()}, {g.strip()}" for f, g in names)
    doi = field(r"^doi:\s*(\S+)", "")
    year = field(r'^date-released:\s*"?(\d{4})', "2026")
    version = field(r"^version:\s*(\S+)", "")
    return [
        "## Citation",
        "",
        "The checkpoints and the code are released together — cite the "
        "software (`CITATION.cff` in the code repository):",
        "",
        "```bibtex",
        "@software{structbench,",
        f"  author  = {{{authors or 'Li, Qilin'}}},",
        "  title   = {StructBench: verification and validation benchmarks "
        "for learned surrogates of structural response},",
        f"  year    = {{{year}}},",
        f"  version = {{{version}}},",
        *([f"  doi     = {{{doi}}},"] if doi else []),
        f"  url     = {{{_REPO_URL}}},",
        "}",
        "```",
        "",
    ]


def render_model_readme(plan: ReleasePlan) -> str:
    """The Hugging Face model card of one benchmark's checkpoints."""
    slug = MODEL_REPOS[plan.benchmark]
    lines = [
        "---",
        "license: apache-2.0",
        "library_name: pytorch",
        "tags:",
        "- structbench",
        "- structural-engineering",
        "- surrogate-model",
        f"- {plan.benchmark}",
        "---",
        "",
        f"# StructBench checkpoints: `{plan.benchmark}`",
        "",
        f"Blessed reference checkpoints of the StructBench benchmark "
        f"`{plan.benchmark}`, published with code release `{plan.tag}` "
        "(ADR-0074 clause 8). Each folder holds what running and checking "
        "the model needs: the registered checkpoint, the resolved "
        "`config.json`, the normalisation statistics when the model uses "
        "them, the `metrics-<split>.json` of the scored splits, and a "
        "`SHA256SUMS` covering those files. The results "
        "registry that reads these numbers is the benchmark's landing page, "
        f"[`docs/benchmarks/{plan.benchmark}.md`]({_REPO_URL}/blob/{plan.tag}/"
        f"docs/benchmarks/{plan.benchmark}.md).",
        "",
        "| Row | Family | Scheme | Run commit | Run date | Bundle | Checkpoint | "
        "SHA-256 | Headline |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in plan.rows:
        r = row.result
        lines.append(
            f"| {r.label} | {r.family} | {r.scheme or '—'} | `{r.run_commit}` | "
            f"{r.run_date} | `{row.bundle}/` | `{row.filename}` | "
            f"`{row.check.digest or '—'}` | {_headline(r)} |"
        )
    lines += [
        "",
        "## Use",
        "",
        "Download a folder and evaluate it with the StructBench CLI, which "
        "rebuilds the simulator from the folder's own `config.json` (the "
        "evaluation overwrites the downloaded `metrics-<split>.json`; keep a "
        "copy to compare):",
        "",
        "```bash",
        f"hf download {HF_ORG}/{slug} --revision {plan.tag} --local-dir bundles",
        "structbench-train --mode rollout --out bundles/<bundle> "
        "--data-root <canonical cases>",
        "```",
        "",
        "Verify a file against `SHA256SUMS` before trusting it. Code and "
        f"benchmark definitions: {_REPO_URL} (Apache-2.0).",
        "",
        *citation_section(),
    ]
    return "\n".join(lines)


def render_publish_script(plan: ReleasePlan) -> str:
    """The ``hf`` commands that publish the plan, for the maintainer to run."""
    repo = f"{HF_ORG}/{MODEL_REPOS[plan.benchmark]}"
    lines = [
        "#!/usr/bin/env bash",
        f"# Publish the blessed checkpoints of {plan.benchmark} under {plan.tag}",
        "# (ADR-0074 clause 8). Generated by tools/publish_checkpoints.py; the",
        "# maintainer runs it: Hugging Face actions stay the human's (ADR-0023).",
        "set -euo pipefail",
        f"hf repos create {repo} --type model --private || true",
    ]
    for row in plan.rows:
        lines.append(
            f'hf upload {repo} "$(dirname "$0")/{row.bundle}" {row.bundle} --type model'
        )
    lines += [
        f'hf upload {repo} "$(dirname "$0")/README.md" README.md --type model',
        f"hf repos settings {repo} --type model --public",
        f"hf repos tag create {repo} {plan.tag} --type model",
        "",
    ]
    return "\n".join(lines)


def _is_local_path(text: str) -> bool:
    return text.startswith(("/", "~", "\\")) or text[1:3] in (":\\", ":/")


def _scrub(value: object, where: str) -> object:
    """``value`` without local paths: ``data_root`` is dropped, a
    ``checkpoint_path`` keeps its file name, and any other absolute path is
    an error (a published file names no machine)."""
    if isinstance(value, dict):
        out: dict[str, object] = {}
        for key, item in value.items():
            if key == "data_root":
                continue
            if key == "checkpoint_path" and isinstance(item, str):
                out[key] = item.replace("\\", "/").rsplit("/", 1)[-1]
                continue
            out[key] = _scrub(item, f"{where}/{key}")
        return out
    if isinstance(value, list):
        return [_scrub(item, f"{where}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, str) and _is_local_path(value):
        raise ValueError(f"{where} holds a local path: {value!r}")
    return value


def published_files(row: ReleaseRow) -> list[str]:
    """The files of a row's bundle that are published, in upload order."""
    names = [row.filename, "config.json"]
    if (row.check.bundle_dir / "normalization_stats.npz").is_file():
        names.append("normalization_stats.npz")
    for split in row.result.metrics:
        name = f"metrics-{split}.json"
        if (row.check.bundle_dir / name).is_file():
            names.append(name)
    return names


def stage_bundle(row: ReleaseRow, folder: Path) -> Path:
    """Write the published subset of ``row``'s bundle to
    ``folder/<bundle>/`` with its own ``SHA256SUMS``; returns that folder.
    JSON files are rewritten without local paths; binaries are copied."""
    dest = folder / row.bundle
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for name in published_files(row):
        source = row.check.bundle_dir / name
        if name.endswith(".json"):
            data = _scrub(json.loads(source.read_text(encoding="utf-8")), name)
            text = json.dumps(data, indent=2) + "\n"
            (dest / name).write_text(text, encoding="utf-8")
        else:
            shutil.copyfile(source, dest / name)
    sums = "".join(
        f"{_sha256(path)}  ./{path.name}\n" for path in sorted(dest.iterdir())
    )
    (dest / "SHA256SUMS").write_text(sums, encoding="utf-8")
    return dest


def stage(plan: ReleasePlan, out: Path) -> Path:
    """Write each row's published folder, the README, the publish script and
    ``pointers.json`` for the plan under ``out/<benchmark>/``; returns that
    folder."""
    folder = out / plan.benchmark
    folder.mkdir(parents=True, exist_ok=True)
    for row in plan.rows:
        stage_bundle(row, folder)
    (folder / "README.md").write_text(render_model_readme(plan), encoding="utf-8")
    (folder / "publish.sh").write_text(render_publish_script(plan), encoding="utf-8")
    pointers = {
        row.label: {
            "checkpoint": row.result.checkpoint,
            "url": row.url,
            "checkpoint_sha256": row.check.digest,
        }
        for row in plan.rows
    }
    (folder / "pointers.json").write_text(
        json.dumps(pointers, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return folder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmark", required=True, choices=sorted(MODEL_REPOS))
    parser.add_argument("--models-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tag", required=True, help="the code release tag")
    parser.add_argument("--include-provisional", action="store_true")
    parser.add_argument(
        "--check", action="store_true", help="verify the mirror; stage nothing"
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        metavar="BUNDLE=DIR",
        help="read this bundle from a run directory instead of the mirror",
    )
    args = parser.parse_args(argv)
    run_dirs: dict[str, Path] = {}
    for item in args.run_dir:
        bundle, sep, directory = item.partition("=")
        if not sep or not bundle or not directory:
            parser.error(f"--run-dir takes BUNDLE=DIR, got {item!r}")
        run_dirs[bundle] = Path(directory)
    spec = get_benchmark(args.benchmark)
    plan = plan_release(
        args.benchmark,
        spec,
        args.models_root,
        args.tag,
        include_provisional=args.include_provisional,
        run_dirs=run_dirs,
    )
    for label, why in plan.skipped.items():
        print(f"skip {label}: {why}")
    for row in plan.rows:
        state = "ok" if row.check.ok else "FAIL"
        print(f"{state} {row.label}: {row.result.checkpoint} -> {row.url}")
        for problem in row.check.problems:
            print(f"  problem: {problem}")
    if not plan.ok:
        print("error: a row failed verification; nothing staged")
        return 1
    if not plan.rows:
        print("nothing to publish")
        return 0
    if args.check:
        print("check passed; nothing staged (--check)")
        return 0
    folder = stage(plan, args.out)
    for row in plan.rows:
        print(f"staged {row.bundle}/: {', '.join(published_files(row))}, SHA256SUMS")
    print(f"staged {len(plan.rows)} row(s) under {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
