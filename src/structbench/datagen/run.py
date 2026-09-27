"""Run a sweep's Abaqus decks, N at a time, and record each run.

    structbench-datagen run --sweep <work-root>/<name>
        [--split NAME ...] [--cases ID ...] [--limit N] [--workers 6]
        [--timeout S] [--retry-failed] [--abaqus EXE] [--dry-run]
        [--min-free-gb GB]

Solver-generic: it knows only the case-folder layout the generator writes. A
case is done once its ``run.json`` exists; nothing is deleted -- an interrupted
or retried attempt moves to ``attempts/<n>/``. Status comes from
``read_abaqus_run_evidence``, the reader the validator uses, so "did it finish"
has one answer everywhere (ADR-0069).

The budget (ADR-0071): before each launch the free space is compared with the
margin (``--min-free-gb``, or the preflight stamp's); below it nothing more is
launched, running jobs finish and are recorded, held cases stay pending, and
the exit code is 3. When a stamp exists, its estimate of the chosen cases'
time and disk is printed before the first launch. Exit codes: 0 clean, 1 a
job did not complete, 2 refused, 3 stopped by the environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from structbench.core.io import unit_factors
from structbench.core.io.abaqus_run import read_abaqus_run_evidence
from structbench.datagen.generate import read_stamp

LOCK_NAME = ".runner.lock"
_KEEP = {"provenance.json", "attempts"}
#: A job held back because the free space fell below the margin: no
#: ``run.json``, no log row; the case stays pending.
NOT_LAUNCHED = "not_launched"


@dataclass(frozen=True)
class JobResult:
    case_id: str
    status: str
    wall_s: float
    return_code: int | None


@dataclass
class _Control:
    """Shared between the result loop and the workers, so a stop reaches both.

    ``guard`` makes "is the sweep stopping?" and "register my process" one step,
    so no job can launch after the loop has decided to stop.
    """

    running: dict[str, subprocess.Popen[bytes]] = field(default_factory=dict)
    stopping: threading.Event = field(default_factory=threading.Event)
    #: Set when the disk margin is breached: no more launches, running jobs
    #: finish normally (unlike ``stopping``, which kills them).
    halted: threading.Event = field(default_factory=threading.Event)
    guard: threading.Lock = field(default_factory=threading.Lock)


def free_gb(path: Path) -> float:
    """Free space on the volume holding ``path``, in GB (1e9 bytes)."""
    return shutil.disk_usage(path).free / 1e9


def estimate_line(
    stamp: dict[str, Any], n_cases: int, workers: int, free: float
) -> str:
    """One line: what the preflight's pilots say the chosen cases will cost."""
    budget = stamp.get("budget") or {}
    wall, per_case = budget.get("wall_s_median"), budget.get("bytes_per_case")
    margin = budget.get("min_free_gb")
    hours = (
        "?" if wall is None else f"{n_cases * float(wall) / max(workers, 1) / 3600:.1f}"
    )
    disk = "?" if per_case is None else f"{n_cases * float(per_case) / 1e9:.1f}"
    margin_text = "no margin" if margin is None else f"margin {float(margin):g} GB"
    return (
        f"estimate from the preflight: {n_cases} cases ~ {hours} h at {workers} "
        f"workers, ~ {disk} GB; free {free:.1f} GB ({margin_text})"
    )


def _cases(sweep: Path) -> list[Path]:
    return sorted(
        p for p in sweep.iterdir() if p.is_dir() and (p / f"{p.name}.inp").is_file()
    )


def case_state(case_dir: Path) -> str:
    """``pending``, ``done``, ``failed`` or ``interrupted``, read from the folder."""
    run = case_dir / "run.json"
    if run.is_file():
        status = json.loads(run.read_text(encoding="utf-8"))["status"]
        if status == "stopped":  # the runner stopped it; not the job's failure
            return "interrupted"
        return "done" if status == "completed" else "failed"
    keep = _KEEP | {f"{case_dir.name}.inp"}
    launched = any(p.name not in keep for p in case_dir.iterdir())
    return "interrupted" if launched else "pending"


def _move_attempt(case_dir: Path) -> None:
    root = case_dir / "attempts"
    n = 1 + (sum(1 for p in root.iterdir() if p.is_dir()) if root.is_dir() else 0)
    target = root / str(n)
    target.mkdir(parents=True)
    keep = _KEEP | {f"{case_dir.name}.inp"}
    for item in list(case_dir.iterdir()):
        if item.name not in keep:
            shutil.move(str(item), str(target / item.name))


def _classify(case_dir: Path, units: str) -> tuple[str, dict[str, Any]]:
    def text(ext: str) -> str | None:
        path = case_dir / f"{case_dir.name}{ext}"
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8", errors="replace")

    evidence = read_abaqus_run_evidence(
        status_text=text(".sta"),
        messages_text=text(".msg"),
        printed_text=text(".dat"),
        source_units=units,
    )
    records = evidence.termination or ()
    if any(r.status == "error" for r in records):
        status = "error"
    elif records and all(r.status == "normal" for r in records):
        status = "completed"
    else:
        status = "unrecognised"
    return status, {
        "abaqus_version": evidence.identity.version if evidence.identity else None,
        "n_errors": evidence.n_errors,
        "n_warnings": evidence.n_warnings,
        "termination": [r.status for r in records],
        "unparsable": sorted(evidence.unparsable),
    }


def _stop(
    proc: subprocess.Popen[bytes], abaqus: list[str], case_id: str, cwd: Path
) -> None:
    try:
        subprocess.run(
            [*abaqus, "terminate", f"job={case_id}"],
            cwd=cwd,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass
    if proc.poll() is None:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True
            )
        else:
            proc.kill()
    proc.wait()


def _run_one(
    case_dir: Path,
    abaqus: list[str],
    timeout: float | None,
    control: _Control,
    min_free_gb: float | None = None,
) -> JobResult:
    case_id = case_dir.name
    provenance = json.loads((case_dir / "provenance.json").read_text(encoding="utf-8"))
    command = [
        *abaqus,
        f"job={case_id}",
        f"input={case_id}.inp",
        "double=both",
        "cpus=1",
        "interactive",
    ]
    start, t0 = datetime.now(UTC), time.monotonic()
    return_code: int | None = None
    facts: dict[str, Any] = {}
    try:
        with (case_dir / "runner.log").open("wb") as log:
            with control.guard:
                held = None
                if control.stopping.is_set():  # the sweep stopped before this job
                    held = "stopped"
                elif control.halted.is_set():
                    held = NOT_LAUNCHED
                elif min_free_gb is not None and free_gb(case_dir) < min_free_gb:
                    control.halted.set()
                    held = NOT_LAUNCHED
                if held is not None:
                    log.close()
                    (case_dir / "runner.log").unlink()
                    return JobResult(case_id, held, 0.0, None)
                proc = subprocess.Popen(
                    command, cwd=case_dir, stdout=log, stderr=subprocess.STDOUT
                )
                control.running[case_id] = proc
            try:
                return_code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _stop(proc, abaqus, case_id, case_dir)
                status = "timeout"
            else:
                if control.stopping.is_set():
                    status = "stopped"
                else:
                    try:
                        status, facts = _classify(case_dir, provenance["units"])
                    except Exception as exc:  # never lose the record of a real run
                        status = "unrecognised"
                        facts = {"error": f"{type(exc).__name__}: {exc}"}
            finally:
                control.running.pop(case_id, None)
    except OSError as exc:
        status, facts = "launch_error", {"error": type(exc).__name__}
    wall = time.monotonic() - t0
    record = {
        "case_id": case_id,
        "command": [Path(command[0]).name, *command[1:]],
        "start_utc": start.isoformat(timespec="seconds"),
        "end_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "wall_s": round(wall, 1),
        "return_code": return_code,
        "status": status,
        **facts,
    }
    (case_dir / "run.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return JobResult(case_id, status, wall, return_code)


def _append_log(sweep: Path, result: JobResult) -> None:
    path = sweep / "run_log.csv"
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(
                ["case_id", "status", "wall_s", "return_code", "logged_utc"]
            )
        writer.writerow(
            [
                result.case_id,
                result.status,
                round(result.wall_s, 1),
                result.return_code,
                datetime.now(UTC).isoformat(timespec="seconds"),
            ]
        )


def run_sweep(
    sweep: Path,
    abaqus: list[str],
    *,
    cases: list[str] | None = None,
    splits: list[str] | None = None,
    limit: int | None = None,
    workers: int = 6,
    timeout: float | None = None,
    retry_failed: bool = False,
    dry_run: bool = False,
    min_free_gb: float | None = None,
    stamp: dict[str, Any] | None = None,
    echo: Callable[[str], None] = print,
) -> list[JobResult]:
    """Run the sweep's pending (and, on request, failed) cases.

    ``min_free_gb`` is the disk margin checked before each launch; below it
    the remaining jobs come back ``not_launched`` and stay pending. ``stamp``
    is the preflight stamp whose budget is printed as an estimate.
    """
    if any(ch.isspace() for ch in str(sweep.resolve())):
        raise ValueError(
            "the sweep path contains whitespace, which Abaqus job folders "
            f"must not: {sweep.name}"
        )
    chosen: list[tuple[Path, str]] = []
    for case_dir in _cases(sweep):
        if cases and case_dir.name not in cases:
            continue
        if splits:
            prov = json.loads(
                (case_dir / "provenance.json").read_text(encoding="utf-8")
            )
            if prov["split"] not in splits:
                continue
        state = case_state(case_dir)
        if state == "done" or (state == "failed" and not retry_failed):
            continue
        chosen.append((case_dir, state))
    chosen = chosen[:limit] if limit else chosen
    for case_dir, _ in chosen:  # a bad unit label must fail here, not after a solve
        prov = json.loads((case_dir / "provenance.json").read_text(encoding="utf-8"))
        try:
            unit_factors(prov["units"])
        except ValueError as exc:
            raise ValueError(f"{case_dir.name}: {exc}") from None
    if stamp is not None:
        echo(estimate_line(stamp, len(chosen), workers, free_gb(sweep)))
    if dry_run:
        for case_dir, state in chosen:
            echo(f"{case_dir.name} {state}")
        return []
    lock = sweep / LOCK_NAME
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(
            f"{LOCK_NAME} exists: another runner holds this sweep, or one crashed. "
            "Check, then delete it."
        ) from None
    with os.fdopen(fd, "w") as fh:
        fh.write(str(os.getpid()))
    results: list[JobResult] = []
    control = _Control()
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        for case_dir, state in chosen:
            if state in ("interrupted", "failed"):
                _move_attempt(case_dir)
        futures = [
            pool.submit(_run_one, d, abaqus, timeout, control, min_free_gb)
            for d, _ in chosen
        ]
        announced = False
        for k, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            if result.status == NOT_LAUNCHED:
                if not announced:
                    echo(
                        f"free space below {min_free_gb} GB: launching nothing "
                        "more; running jobs finish"
                    )
                    announced = True
                continue
            _append_log(sweep, result)  # only this thread writes the log
            progress = f"[{k}/{len(chosen)}] {result.case_id}"
            echo(f"{progress} {result.status} {result.wall_s:.0f} s")
    except BaseException:
        # Ctrl+C or any failure here: launch nothing more, stop what runs.
        with control.guard:
            control.stopping.set()
            running = list(control.running.items())
        pool.shutdown(wait=False, cancel_futures=True)
        for case_id, proc in running:
            _stop(proc, abaqus, case_id, sweep / case_id)
        raise
    finally:
        pool.shutdown(wait=True)  # the lock outlives every worker
        lock.unlink(missing_ok=True)
    return results


def _summarise(results: list[JobResult]) -> None:
    counts = Counter(r.status for r in results)
    print("summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    ran = [r for r in results if r.status != NOT_LAUNCHED]
    for r in sorted(ran, key=lambda r: -r.wall_s)[:3]:
        print(f"slowest: {r.case_id} {r.wall_s:.0f} s")
    for r in ran:
        if r.status != "completed":
            print(f"FAILED: {r.case_id} {r.status}")
    held = counts.get(NOT_LAUNCHED, 0)
    if held:
        print(
            f"not launched: {held} case{'s' if held != 1 else ''} stay pending "
            "(free space below the margin)"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="structbench-datagen run", description=(__doc__ or "").splitlines()[0]
    )
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--split", action="append")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--abaqus", default="abaqus")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--min-free-gb",
        type=float,
        help="launch nothing more below this free space (default: the preflight "
        "stamp's margin, when the sweep has one)",
    )
    args = parser.parse_args(argv)
    exe = shutil.which(args.abaqus)
    if exe is None:
        print(f"abaqus executable {args.abaqus!r} not found", file=sys.stderr)
        return 2
    stamp = read_stamp(args.sweep)
    min_free = args.min_free_gb
    if min_free is None and stamp is not None:
        margin = (stamp.get("budget") or {}).get("min_free_gb")
        min_free = None if margin is None else float(margin)
    try:
        results = run_sweep(
            args.sweep,
            [exe],
            cases=args.cases,
            splits=args.split,
            limit=args.limit,
            workers=args.workers,
            timeout=args.timeout,
            retry_failed=args.retry_failed,
            dry_run=args.dry_run,
            min_free_gb=min_free,
            stamp=stamp,
        )
    except (RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    _summarise(results)
    if any(r.status == NOT_LAUNCHED for r in results):
        return 3
    return 1 if any(r.status != "completed" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
