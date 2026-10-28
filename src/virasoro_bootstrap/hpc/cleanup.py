"""Clean generated SDP inputs of finished, unchanged cluster tasks.

    python -m virasoro_bootstrap.hpc.cleanup RUNS_ROOT [--execute]

The default is a dry run. Unfinished/stale tasks and scratch symlinks are skipped.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import os

from .storage import cleanup_task_solver_inputs, regenerate_sdp_inputs, keep_solver_inputs, _OWNED_FILE, _has_finished_solver_output
from .worker import result_doc_is_final, spec_hash, _task_output_lock, _TaskBusy


def _active_jobs():
    """The user's queued/running job ids: each job's own id (%A) and its array base id (%F).

    A run manifest records the base id of its array, which %A alone never shows
    for array elements.
    """
    if shutil.which("squeue") is None:
        return set()
    text = subprocess.check_output(["squeue", "-u", str(os.getuid()), "-h", "-o", "%A %F"], text=True)
    return set(text.split())


def _read_document(path):
    doc = json.loads(path.read_text())
    if not isinstance(doc, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return doc


def _clean_finished(spec_path, task):
    # Recheck after queuing: another process may have edited or restarted it.
    try:
        with _task_output_lock(task):
            spec = _read_document(spec_path)
            status = _read_document(task / "status.json")
            result = _read_document(task / "result.json")
            if status.get("state") != "done" or not result_doc_is_final(
                    result, spec_hash(spec["task"], spec.get("params", {}))):
                return {"removed_files": 0, "removed_bytes": 0, "retained_bytes": 0}
            return cleanup_task_solver_inputs(task)
    except _TaskBusy:
        return {"removed_files": 0, "removed_bytes": 0, "retained_bytes": 0}


def cleanup_runs(runs_root, *, execute=False, workers=1):
    from virasoro_blocks.requests import _integer_field

    if not isinstance(execute, bool):
        raise ValueError("execute must be bool")
    workers = _integer_field("workers", workers)
    if workers < 1:
        raise ValueError("workers must be >= 1")
    totals = {"removed_files": 0, "removed_bytes": 0, "retained_bytes": 0,
              "completed_tasks": 0, "errors": []}
    active = _active_jobs()
    ready = []
    for run in sorted(Path(runs_root).expanduser().iterdir()):
        if not run.is_dir() or run.is_symlink() or not (run / "manifest.json").is_file():
            continue
        try:
            manifest = _read_document(run / "manifest.json")
            cluster = manifest.get("cluster", {})
            if not isinstance(cluster, dict) or not isinstance(cluster.get("env", {}), dict):
                raise ValueError(f"invalid saved cluster environment: {run}")
            retain = keep_solver_inputs(cluster.get("env", {}).get("VB_KEEP_SOLVER_INPUTS", ""))
        except (OSError, ValueError) as exc:
            totals["errors"].append({"run": str(run), "error": str(exc)})
            continue
        if retain:
            continue
        # One squeue snapshot: each task is rechecked under its lock before cleanup.
        if str(manifest.get("slurm_job_id")) in active:
            continue
        for spec_path in sorted((run / "specs").glob("*.json")):
            try:
                spec = _read_document(spec_path)
                task = Path(spec["output_dir"])
                if not task.is_absolute():
                    task = run / task
                # Historic absolute paths may use either the home symlink or
                # physical project path. Refuse outputs outside this run.
                if task.is_symlink() or not task.is_dir() or not task.resolve().is_relative_to(run.resolve()):
                    continue
                if not (task / "status.json").is_file() or not (task / "result.json").is_file():
                    continue
                status = _read_document(task / "status.json")
                result = _read_document(task / "result.json")
                if (status.get("state") != "done" or not result_doc_is_final(
                        result, spec_hash(spec["task"], spec.get("params", {})))):
                    continue
                totals["completed_tasks"] += 1
                if execute:
                    ready.append((spec_path, task))
                else:
                    count = 0
                    for base, dirs, files in os.walk(task, followlinks=False):
                        dirs[:] = [d for d in dirs if d not in {"blocks", "sdpb_out"}
                                   and not d.endswith(".ck") and not (Path(base) / d).is_symlink()]
                        if "control.json" in files and "normalization.json" in files:
                            if not _has_finished_solver_output(base):
                                dirs.clear()
                                continue
                            control = _read_document(Path(base) / "control.json")
                            if control.get("command") == "virasoro_bootstrap.sdp_writer":
                                count += sum(bool(_OWNED_FILE.fullmatch(name)) for name in files)
                            dirs.clear()
                    print(json.dumps({"task": str(task), "candidate_files": count}), flush=True)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                totals["errors"].append({"spec": str(spec_path), "error": str(exc)})
                print(json.dumps(totals["errors"][-1]), flush=True)
    if ready:
        # Large decimal-array encoding/decoding holds the Python GIL. Separate
        # processes let a multi-core allocation do that work concurrently.
        executor = ThreadPoolExecutor if workers == 1 else ProcessPoolExecutor
        with executor(max_workers=workers) as pool:
            futures = {pool.submit(_clean_finished, spec, task): (spec, task)
                       for spec, task in ready}
            for future in as_completed(futures):
                spec_path, task = futures[future]
                try:
                    stats = future.result()
                    for name in ("removed_files", "removed_bytes", "retained_bytes"):
                        totals[name] += stats[name]
                    print(json.dumps({"task": str(task), **stats}), flush=True)
                except Exception as exc:
                    error = {"spec": str(spec_path), "error": str(exc)}
                    totals["errors"].append(error)
                    print(json.dumps(error), flush=True)
    return totals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_root", type=Path, nargs="?")
    parser.add_argument("--execute", action="store_true", help="remove verified generated inputs")
    parser.add_argument("--workers", type=int, default=1, help="concurrent completed tasks (default 1)")
    parser.add_argument("--restore", type=Path, metavar="SOLVE_DIR", help="regenerate one cleaned solve's SDP inputs")
    args = parser.parse_args(argv)
    if args.restore:
        if args.runs_root or args.execute or args.workers != 1:
            parser.error("--restore cannot be combined with runs_root, --execute, or --workers")
        print(regenerate_sdp_inputs(args.restore))
        return 0
    if args.runs_root is None:
        parser.error("runs_root is required unless --restore is given")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    totals = cleanup_runs(args.runs_root, execute=args.execute, workers=args.workers)
    print(json.dumps({"summary": totals}), flush=True)
    return bool(totals["errors"])


if __name__ == "__main__":
    raise SystemExit(main())
