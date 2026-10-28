"""vb-status / vb-collect / vb-list: inspect, aggregate, and enumerate runs.

    vb-status  20260101-120000-gap_vs_c_l7            # a run id, as vb-list prints it
    vb-status  artifacts/runs/20260101-120000-gap_vs_c_l7   # or the run directory
    vb-status  12345678                              # bare SLURM job id also works
    vb-collect 20260101-120000-gap_vs_c_l7 [-o results.json]
    vb-list                                          # "what runs exist and what are their IDs"

    # standing on the cluster itself: --cluster (no --host -- no SSH needed)
    # sources the profile's runs_root instead of a bare ./artifacts/runs. Set
    # $VB_CLUSTER (e.g. in ~/.bashrc) and even --cluster becomes optional:
    vb-list

    # from the laptop: run the same inspection over SSH, no local run dir needed
    vb-status  <run-id> --host mycluster --cluster <cluster-profile.toml>
    vb-collect <run-id> --host mycluster --cluster <cluster-profile.toml> [-o results.json]
    vb-list --host mycluster --cluster <cluster-profile.toml>

``vb-list`` exists because a run's directory name (the thing every other
command needs) is a generated timestamp+sweep-name -- it's printed once at
submit time and otherwise easy to lose. Its default runs_root is, in order:
``--runs-root`` if given, else ``--cluster``'s ``runs_root`` if given (no SSH
needed -- just reads the profile file), else ``artifacts/runs/`` in the current
directory (the laptop local-docker convention); with ``--host`` the same
precedence applies on the remote machine, over SSH. It lists that directory newest-first
with each run's task/n_tasks/SLURM job id, so a bare directory listing is
never necessary.

``vb-status``/``vb-collect`` accept a bare SLURM job id (with or without an
array-task suffix, e.g. ``12345678`` or ``12345678_0`` -- squeue shows the
latter) instead of a run directory: it's resolved by scanning ``runs_root``
(the same resolution order as ``vb-list``, above) for the manifest whose
``slurm_job_id`` matches. A literal existing directory always wins first, so
this never changes behavior for the original run-directory usage.

Both ``vb-status``/``vb-collect`` work purely from the run directory's files (specs, status.json,
result.json, progress.json) -- the filesystem is the source of truth, so they
give the same answer on the cluster, over an rsync'd copy, or mid-run.
``vb-collect`` writes the per-task ``result`` dicts as one JSON list in task
order (for ``scalar_gap_bisection`` sweeps this is exactly the ``bounds.json``
format the plotting scripts already consume); tasks without a result yet are
emitted as ``{"reason": "pending", ...params...}`` placeholders.

``--host`` runs the same code over SSH on the given alias (from
``~/.ssh/config``), using ``remote_repo``/``python`` from the ``--cluster``
profile; ``<remote-run-dir>`` is interpreted relative to ``remote_repo``
(absolute paths work too). ``vb-collect --host`` additionally rsyncs the
aggregate JSON it writes back to the local machine for plotting.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .worker import (result_doc_convention, result_doc_convention_matches, result_doc_failure, spec_hash,
                     _task_result_path, _task_output_dir)
from .inputs import (
    HpcInputError, cli_input_errors, load_run_manifest, nonnegative_integer,
    path_argument, read_job_spec, resolve_local_runs_root,
)
from .journal import progress_counter, progress_mapping
from ..paths import (
    resolve_research_path,
    result_file,
)


@dataclass
class TaskView:
    index: int
    params: dict
    state: str          # done | stale | failed | interrupted | running | pending
    starts: int
    detail: str
    result: dict | None


def _result_brief(result: dict, limit: int = 100) -> str:
    """One-line summary of a result dict: its scalar fields, logs/nests elided."""
    parts = [
        f"{k}={v}"
        for k, v in result.items()
        if isinstance(v, (str, int, float, bool)) or v is None
    ]
    brief = ", ".join(parts) or "ok"
    return brief[: limit - 3] + "..." if len(brief) > limit else brief


def _read_json(path: Path) -> dict | None:
    try:
        document = json.loads(path.read_text())
        return document if isinstance(document, dict) else None
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _read_status(path: Path) -> dict:
    """Keep damaged diagnostic fields readable without trusting them as states."""
    status = _read_json(path) or {}
    state = status.get("state", "pending")
    if not isinstance(state, str) or state not in {
            "done", "stale", "failed", "interrupted", "running", "pending"}:
        status = {**status, "state": "failed", "detail": f"invalid status state in {path}"}
    elif "detail" in status and not isinstance(status["detail"], str):
        status = {**status, "detail": f"invalid status detail in {path}"}
    return status


def _result_is_stale(result_doc: dict, spec: dict) -> bool:
    """True when ``result.json`` was produced by a different spec than the current one.

    Only a hash in the worker's own format (16 hex digits) is judged: a result
    without one (pre-hash, or hand-written) is taken as it stands.
    """
    from .worker import spec_hash

    stored = result_doc.get("spec_hash")
    if (
        not isinstance(stored, str)
        or re.fullmatch(r"[0-9a-f]{16}", stored) is None
        or "task" not in spec
    ):
        return False
    return stored != spec_hash(spec["task"], spec.get("params", {}))


def _newer_attempt(first: dict, second: dict) -> bool:
    """Compare ordinals across spec edits, falling back to legacy start times."""
    a, b = first.get('attempt'), second.get('attempt')
    if type(a) is int and a > 0 and type(b) is int and b > 0:
        return a > b
    try:
        return (datetime.datetime.fromisoformat(first.get('started_at'))
                > datetime.datetime.fromisoformat(second.get('started_at')))
    except (TypeError, ValueError):
        return False


def _result_view(index: int, spec: dict, starts: int, result_doc: dict, status=None) -> TaskView:
    """The view of a task with a current ``result.json``: done, or failed.

    A task that caught its own failure left an error payload; it is still worth
    collecting (its logs are the diagnosis) but it is not done, and
    ``vb-submit --retry`` reruns it.
    """
    result = result_doc.get("result")
    failure = result_doc_failure(result_doc)
    current = result_doc
    if failure is not None:
        # An exception in a later attempt can leave the last task payload
        # intact. Retain its logs, but report the latest worker failure.
        if (status and status.get('state') == 'failed'
                and not _newer_attempt(result_doc, status)
                and isinstance(status.get('detail'), str) and status['detail']):
            failure = status['detail']
            current = status
        if type(current.get('starts')) is int and current['starts'] >= 0:
            starts = current['starts']
        detail = _result_brief({"error": failure}, limit=300)
        if not isinstance(result, dict):
            result = {"reason": "invalid_result", "error": failure, "raw_result": result}
        return TaskView(index, spec.get("params", {}), "failed", starts, detail, result)
    detail = _result_brief(result) if isinstance(result, dict) else "ok"
    if type(result_doc.get('starts')) is int and result_doc['starts'] >= 0:
        starts = result_doc['starts']
    return TaskView(index, spec.get("params", {}), "done", starts, detail, result)


def load_run(run_dir: Path) -> tuple[dict, list[TaskView]]:
    run_dir = run_dir.resolve()
    manifest, n_tasks = load_run_manifest(run_dir)
    views: list[TaskView] = []
    for i in range(n_tasks):
        spec_path = run_dir / "specs" / f"{i:04d}.json"
        saved_spec = read_job_spec(spec_path, allow_missing=True, require_output=False)
        spec = saved_spec or {}
        task_dir = _task_output_dir(run_dir, i, spec.get("output_dir"))
        result_doc = _read_json(_task_result_path(run_dir, i, spec.get("output_dir")))
        status = _read_status(task_dir / "status.json")
        if saved_spec is None and (result_doc is not None or status.get("state") == "done"):
            raise HpcInputError(f"no job spec at {spec_path}; cannot verify the saved task result")
        journal = _read_json(task_dir / "progress.json") or {}
        stale_journal = _result_is_stale(journal, spec)
        starts = 0 if stale_journal else progress_counter(journal, "starts")
        current_status = (status if 'task' in spec and status.get('spec_hash')
                          == spec_hash(spec['task'], spec.get('params', {})) else {})
        if type(current_status.get('starts')) is int and current_status['starts'] >= 0:
            starts = current_status['starts']
        if result_doc is not None and _result_is_stale(result_doc, spec):
            if current_status.get('state') not in {'running', 'interrupted', 'failed', 'done'}:
                # An edit without a new attempt is stale. After a retry starts,
                # its current status takes precedence over the old payload.
                views.append(TaskView(
                    i, spec.get("params", {}), "stale", starts,
                    "result.json is for a different spec (edited since); vb-submit --retry reruns it",
                    None,
                ))
                continue
            result_doc = None
        if result_doc is not None and not result_doc_convention_matches(result_doc):
            if current_status.get('state') not in {'running', 'interrupted', 'failed'}:
                from virasoro_blocks._conventions import scalar_G_spin0_convention

                views.append(TaskView(
                    i, spec.get("params", {}), "stale", starts,
                    f"result.json was computed with scalar_G_spin0 = {result_doc_convention(result_doc)!r}, not "
                    f"this process's {scalar_G_spin0_convention()!r}; vb-submit --retry reruns it",
                    None,
                ))
                continue
            result_doc = None
        if (result_doc is not None and result_doc_failure(result_doc) is not None
                and current_status.get('state') in {'running', 'interrupted', 'done'}
                and _newer_attempt(current_status, result_doc)):
            result_doc = None
        if result_doc is not None:
            views.append(_result_view(i, spec, starts, result_doc, current_status))
            continue
        state = status.get("state", "pending")
        if _result_is_stale(status, spec) or (stale_journal and not current_status):
            state, detail = 'stale', 'saved status or progress is for a different spec; vb-submit --retry reruns it'
        elif state == "done":
            # The worker writes result.json THEN status.json (in that order,
            # in one process), but a *different* reading client -- e.g. the
            # login node under --host, one NFS hop away from the compute node
            # that wrote both files -- can observe that ordering out of turn:
            # status.json's rename becomes visible before result.json's does.
            # Observed live on a cluster. A genuine failure never resolves; this
            # race does within tens of milliseconds, so retry briefly before
            # concluding the result is actually missing.
            for _ in range(5):
                time.sleep(0.05)
                result_doc = _read_json(_task_result_path(run_dir, i, spec.get("output_dir")))
                if (result_doc is not None and not _result_is_stale(result_doc, spec)
                        and not (result_doc_failure(result_doc) is not None
                                 and _newer_attempt(current_status, result_doc))):
                    break
            else:
                result_doc = None
            if result_doc is not None:
                views.append(_result_view(i, spec, starts, result_doc, current_status))
                continue
            state, detail = "failed", "status=done but no readable result.json"
        else:
            detail = status.get("detail", "")
        n_probes = 0 if stale_journal else len(progress_mapping(journal, "probes"))
        if n_probes:
            detail = (detail + f" [{n_probes} probe(s) journaled]").strip()
        views.append(TaskView(i, spec.get("params", {}), state, starts, detail, None))
    return manifest, views


def _summarize(views: list[TaskView]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in views:
        counts[v.state] = counts.get(v.state, 0) + 1
    return counts


def _add_host_args(parser: argparse.ArgumentParser) -> None:
    from .submit import ENV_CLUSTER, _cluster_env_default

    parser.add_argument(
        "--host", default=None, metavar="SSH_ALIAS",
        help="inspect on this SSH host instead of locally; needs --cluster",
    )
    parser.add_argument(
        "--cluster", type=Path, default=_cluster_env_default(),
        help=f"cluster profile TOML (remote_repo/python/runs_root; default: ${ENV_CLUSTER})",
    )


def _local_runs_root(args: argparse.Namespace) -> Path:
    """Local run root: explicit, profile-provided, or the artifact default."""
    cluster = {}
    if getattr(args, "cluster", None):
        from .submit import load_toml, _validate_cluster_profile

        cluster = load_toml(args.cluster)
        _validate_cluster_profile(cluster)
    explicit = getattr(args, "runs_root", None)
    raw_root = explicit if explicit is not None else cluster.get("runs_root")
    return resolve_local_runs_root(raw_root)


def _remote_runs_root_arg(args: argparse.Namespace) -> str:
    """Forward explicit roots; the remote process resolves its own defaults."""
    from .submit import load_toml, _validate_cluster_profile

    cluster = load_toml(args.cluster)
    _validate_cluster_profile(cluster)
    root = args.runs_root if args.runs_root is not None else cluster.get("runs_root")
    return f" --runs-root {shlex.quote(str(root))}" if root is not None else ""


_JOB_ID_RE = re.compile(r"^\d+(_\d+)?$")


def _created_at_sort_key(manifest: dict) -> str:
    """Unknown or malformed display dates sort after runs with readable dates."""
    value = manifest.get("created_at")
    return value if isinstance(value, str) else ""


def _looks_like_run_dir(value: str) -> bool:
    p = Path(value)
    return p.is_dir() and (p / "manifest.json").exists()


def _find_run_by_slurm_job_id(job_id: str, runs_root: Path) -> Path | None:
    """The run that submitted *job_id*: its original array or any retry."""
    from .submit import _run_job_ids

    normalized = re.sub(r"_\d+$", "", job_id)  # squeue shows "<job>_<array-idx>"
    if not runs_root.is_dir():
        return None
    candidates = []
    for d in runs_root.iterdir():
        if not d.is_dir():
            continue
        manifest = _read_json(d / "manifest.json")
        if manifest is not None and normalized in _run_job_ids(manifest):
            candidates.append((d, manifest))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: _created_at_sort_key(pair[1]), reverse=True)
    return candidates[0][0]


def resolve_run_dir_arg(value: str, runs_root: Path) -> Path:
    """Resolve a run argument: a run directory, a run id, or a SLURM job id.

    In order: a literal run directory; a run id (the name ``vb-list`` and
    ``vb-submit`` print) under *runs_root*; a bare SLURM job id, matched against
    the manifests under *runs_root*. Anything else is an error naming what was
    tried, rather than a bare "no readable manifest.json".
    """
    if _looks_like_run_dir(value):
        return Path(value)
    by_id = Path(runs_root) / value
    if _looks_like_run_dir(str(by_id)):
        return by_id
    if _JOB_ID_RE.match(value):
        found = _find_run_by_slurm_job_id(value, runs_root)
        if found is not None:
            return found
        raise SystemExit(
            f"no run under {runs_root} has SLURM job id {value}, and no run there "
            "is named that either (try vb-list)"
        )
    literal = Path(value)
    if literal.is_dir() and not (literal / "manifest.json").exists():
        raise SystemExit(f"{literal} is a directory but has no manifest.json; is it a run directory?")
    raise SystemExit(
        f"{value!r} is neither a run directory nor a run id under {runs_root} "
        "(vb-list shows the run ids; --runs-root or --cluster picks the runs root)"
    )


def _ssh_module_call(host: str, cluster_path: Path, module_args: str) -> subprocess.CompletedProcess:
    # Local import: submit.py already has load_toml + the SSH plumbing, and
    # importing it here (rather than duplicating) keeps one source of truth.
    from .submit import (
        load_toml, _shell_profile_python, _ssh_exec, _ssh_run, _validate_cluster_profile,
    )

    cluster = load_toml(cluster_path)
    _validate_cluster_profile(cluster)
    remote_repo = cluster.get("remote_repo")
    python_exe = _shell_profile_python(cluster)
    remote_cmd = f'{python_exe} -m virasoro_bootstrap.hpc.collect {module_args}'
    return (_ssh_run(host, remote_repo, remote_cmd) if remote_repo
            else _ssh_exec(host, remote_cmd))


def _add_runs_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runs-root", type=path_argument, default=None,
        help="where to look for runs (default: --cluster's runs_root, else ./artifacts/runs)",
    )


def _normalize_cluster_arg(args: argparse.Namespace) -> None:
    if getattr(args, "cluster", None) is not None:
        args.cluster = resolve_research_path(args.cluster)


_RUN_ARG_HELP = "run id (as vb-list prints it), run directory, or SLURM job id"

_WHERE_RUNS_ARE = """\
RUN is looked up under --runs-root, else the --cluster profile's runs_root
($VB_CLUSTER sets the default profile), else ./artifacts/runs. With --host the
same command runs over SSH on that machine, using the profile's remote_repo and
python.
"""

STATUS_DESCRIPTION = """\
Show the state of every task of a run: done, failed, stale (its spec was
edited since), interrupted, running or pending, with a one-line result or
error for each.

examples:
  vb-status 20260101-120000-gap_vs_c
  vb-status 12345678                      # the SLURM job id
  vb-status 20260101-120000-gap_vs_c --host mycluster --cluster mycluster.toml

""" + _WHERE_RUNS_ARE

COLLECT_DESCRIPTION = """\
Gather the per-task results of a run into one JSON list, in task order. Tasks
without a result appear as {"reason": "pending", ...params} placeholders;
failed tasks keep their error payload. Writes <run-dir>/results.json unless -o
is given; with --host the file is copied back here.

examples:
  vb-collect 20260101-120000-gap_vs_c -o results.json
  vb-collect 20260101-120000-gap_vs_c --host mycluster --cluster mycluster.toml

""" + _WHERE_RUNS_ARE

LIST_DESCRIPTION = """\
List the runs under a runs root, newest first, with each run's id, task, task
count and SLURM job id. The run id is what vb-status, vb-collect and
vb-certify take.

examples:
  vb-list
  vb-list --cluster mycluster.toml -n 50
  vb-list --host mycluster --cluster mycluster.toml

Runs are looked up under --runs-root, else the --cluster profile's runs_root
($VB_CLUSTER sets the default profile), else ./artifacts/runs.
"""


def _parser(prog: str, description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog=prog, description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


@cli_input_errors("vb-status")
def status_main(argv: list[str] | None = None) -> int:
    parser = _parser("vb-status", STATUS_DESCRIPTION)
    parser.add_argument("run_dir", type=str, metavar="RUN", help=_RUN_ARG_HELP)
    parser.add_argument("--requeue-warn", type=nonnegative_integer, default=5,
                        help="flag tasks restarted at least this many times")
    _add_runs_root_arg(parser)
    _add_host_args(parser)
    args = parser.parse_args(argv)
    _normalize_cluster_arg(args)

    if args.host:
        if args.cluster is None:
            parser.error("--host needs --cluster")
        module_args = (
            f"status {shlex.quote(args.run_dir)} --requeue-warn {args.requeue_warn}"
            + _remote_runs_root_arg(args)
        )
        result = _ssh_module_call(args.host, args.cluster, module_args)
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        return result.returncode

    run_dir = resolve_run_dir_arg(args.run_dir, _local_runs_root(args))
    manifest, views = load_run(run_dir)
    for key in ("run_id", "task"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise HpcInputError(f"{run_dir / 'manifest.json'}: {key} must be a nonempty string")
    print(f"run {manifest['run_id']}  task={manifest['task']}  "
          f"n_tasks={manifest['n_tasks']}  slurm_job={manifest.get('slurm_job_id')}")
    counts = _summarize(views)
    print("  " + "  ".join(f"{state}={n}" for state, n in sorted(counts.items())))
    for v in views:
        flag = "  <-- requeue loop?" if v.starts >= args.requeue_warn else ""
        # Show the swept params compactly (c=..., etc.), not the full dict.
        brief = ", ".join(
            f"{k}={v.params[k]}" for k in ("c", "lam", "delta_ext") if k in v.params
        )
        print(f"  {v.index:04d}  {v.state:<12} starts={v.starts}  {brief}  {v.detail}{flag}")
    return 0


@cli_input_errors("vb-collect")
def collect_main(argv: list[str] | None = None) -> int:
    parser = _parser("vb-collect", COLLECT_DESCRIPTION)
    parser.add_argument("run_dir", type=str, metavar="RUN", help=_RUN_ARG_HELP)
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="output JSON path (default <run_dir>/results.json; with --host, "
                             "a local artifacts/results/<RUN>-results.json, under "
                             "$VB_RESULTS_ROOT or $VB_ARTIFACT_ROOT when set)")
    _add_runs_root_arg(parser)
    _add_host_args(parser)
    args = parser.parse_args(argv)
    _normalize_cluster_arg(args)

    if args.host:
        if args.cluster is None:
            parser.error("--host needs --cluster")
        # Collect remotely into the remote run_dir's default results.json, then
        # rsync that one small file back -- the bulky per-task solve output
        # never has to cross the network.
        module_args = (
            f"collect {shlex.quote(args.run_dir)}" + _remote_runs_root_arg(args)
        )
        result = _ssh_module_call(args.host, args.cluster, module_args)
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        if result.returncode != 0:
            return result.returncode

        # Parse the resolved path from the remote process's own "wrote ..."
        # line rather than reconstructing it locally: remote_repo may contain
        # a literal $HOME, which scp -- unlike the remote python process that
        # already resolved it -- has no reliable way to expand.
        match = re.search(
            r"^wrote (.+) \(\d+/\d+ task\(s\) complete\)$",
            result.stdout, re.MULTILINE,
        )
        if match is None:
            print("warning: could not parse the remote results path from output",
                  file=sys.stderr)
            return 1
        remote_results = match.group(1)
        local_out = args.output or result_file(Path(args.run_dir).name)
        local_out.parent.mkdir(parents=True, exist_ok=True)
        scp = subprocess.run(
            ["scp", "-q", f"{args.host}:{remote_results}", str(local_out)],
            capture_output=True, text=True,
        )
        if scp.returncode != 0:
            print(f"warning: could not copy results back: {scp.stderr.strip()}", file=sys.stderr)
            return scp.returncode
        print(f"copied to {local_out}")
        return 0

    run_dir = resolve_run_dir_arg(args.run_dir, _local_runs_root(args))
    manifest, views = load_run(run_dir)
    aggregate = []
    for v in views:
        if v.result is not None:
            aggregate.append(v.result)
        else:
            aggregate.append({"reason": "pending", **v.params})
    out_path = args.output or (run_dir.resolve() / "results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Collectors can overlap on the same run. Each stages its own complete
    # document on this filesystem, so publication cannot consume another
    # invocation's temporary file. The context also cleans up a failed write.
    with tempfile.TemporaryDirectory(prefix=f".{out_path.name}-", dir=out_path.parent) as temporary:
        tmp = Path(temporary) / out_path.name
        tmp.write_text(json.dumps(aggregate, indent=2))
        tmp.replace(out_path)
    done = sum(1 for v in views if v.state == "done")
    print(f"wrote {out_path} ({done}/{len(views)} task(s) complete)")
    return 0


@cli_input_errors("vb-list")
def list_main(argv: list[str] | None = None) -> int:
    parser = _parser("vb-list", LIST_DESCRIPTION)
    _add_runs_root_arg(parser)
    parser.add_argument("-n", "--limit", type=nonnegative_integer, default=15,
                        help="show at most this many most-recent runs (default 15)")
    _add_host_args(parser)
    args = parser.parse_args(argv)
    _normalize_cluster_arg(args)

    if args.host:
        if args.cluster is None:
            parser.error("--host needs --cluster")
        module_args = f"list{_remote_runs_root_arg(args)} --limit {args.limit}"
        result = _ssh_module_call(args.host, args.cluster, module_args)
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        return result.returncode

    # --cluster without --host: standing directly on the cluster already (no
    # SSH hop needed), but still want the profile's runs_root as the default
    # rather than a bare ./artifacts/runs that only matters for the local-docker
    # workflow. This is exactly the case of running vb-list on the cluster itself.
    root = _local_runs_root(args)
    if not root.is_dir():
        print(f"no runs under {root} (nothing submitted from here yet?)")
        return 0

    entries = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        manifest = _read_json(d / "manifest.json")
        if manifest is not None:
            entries.append((d, manifest))
    if not entries:
        print(f"no runs with a manifest.json under {root}")
        return 0

    entries.sort(key=lambda pair: _created_at_sort_key(pair[1]), reverse=True)
    print(f"runs under {root} (newest first):")
    for d, manifest in entries[: args.limit]:
        print(
            f"  {d.name}  task={manifest.get('task')}  n_tasks={manifest.get('n_tasks')}  "
            f"slurm_job={manifest.get('slurm_job_id')}  created={manifest.get('created_at')}"
        )
    if len(entries) > args.limit:
        print(f"  ... and {len(entries) - args.limit} more (raise -n/--limit to see them)")
    return 0


if __name__ == "__main__":
    dispatch = {"status": status_main, "collect": collect_main, "list": list_main}
    if len(sys.argv) > 1 and sys.argv[1] in dispatch:
        mode, rest = sys.argv[1], sys.argv[2:]
    else:
        mode, rest = "status", sys.argv[1:]
    sys.exit(dispatch[mode](rest))
