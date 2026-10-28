"""vb-submit: expand a sweep config into task specs and submit a SLURM array.

    vb-submit sweep.toml --cluster <cluster-profile.toml> [--dry-run]
    vb-submit --retry artifacts/runs/20260101-120000-gap_vs_c_l7

    # from the laptop: same commands, plus --host <ssh alias from ~/.ssh/config>
    vb-submit sweep.toml --cluster <cluster-profile.toml> --host mycluster
    vb-submit --retry <remote-run-dir> --cluster <cluster-profile.toml> --host mycluster

Set ``$VB_CLUSTER`` (e.g. in ``~/.bashrc`` on the cluster itself) to the
profile path and every ``--cluster`` above becomes optional -- meant for the
"I'm always on this one machine" case (a laptop that only ever submits to one
cluster, or a session on the cluster itself).

``--host`` copies the sweep and cluster TOML to the remote run directory over
ssh and runs the identical ``vb-submit`` invocation there.  The files can live
anywhere -- a home directory, a private repo, anywhere readable.  They do not
need to be tracked, or even inside a checkout.

The remote resulting ``manifest.json`` is copied back to a local mirror under
``artifacts/runs/`` (or ``--runs-root``), so the laptop keeps a record of every
run it launched without needing the bulky remote output.  Nothing stays running
locally: the submit call returns as soon as sbatch (or the local run) is done.

Two ways to supply the code on the far side:

``remote_repo`` set (a checkout's path on that machine)
    If the local library is itself a git checkout, ``--host`` fetches and
    checks out that exact commit remotely, so the cluster runs a known
    revision.  The local tree must then be clean and pushed -- the remote
    fetches from ``origin``, not from your working tree.  This is the strongest
    provenance and what the manifest's ``git_commit`` records.

    ``--allow-dirty`` downgrades the clean-tree requirement to a warning, for
    the very common case where the uncommitted work is somewhere the run does
    not touch.  It does NOT ship your edits: the cluster still runs HEAD.  The
    dirty file list is written to ``<run-dir>/inputs/dirty_at_submit.txt`` so a
    later reader can see that the tree it was launched from did not match the
    commit that ran.  HEAD must still be pushed either way -- the remote
    ``git checkout`` would just fail on a commit origin has never seen.

    CAVEAT, and the reason --allow-dirty is only a partial answer: the remote
    checkout is SHARED by every run.  ``git checkout <commit>`` pins the code at
    SUBMIT time, but tasks import it at RUN time (the sbatch runs
    ``python -m virasoro_bootstrap.hpc.worker`` against an editable install of
    that one directory).  So submitting a second run at a different commit while
    a first is still queued silently moves the first run's remaining tasks onto
    the new code.  Giving each run its own worktree would fix that, and would
    also let --allow-dirty ship the edits for real.

``remote_repo`` unset
    The remote simply runs its installed ``virasoro_blocks``.  Nothing is
    checked out and no commit is pinned, which is the ordinary arrangement for
    a ``pip install`` on a cluster you do not develop in.  The manifest still
    records both input files by value, as it always has.

Creates ``<runs-root>/<run-id>/`` containing::

    manifest.json   what was submitted: sweep, cluster profile, git commit, job id
    specs/NNNN.json one job spec per task (cartesian product of the [sweep] lists)
    tasks/NNNN/     per-task output (result.json, status.json, progress.json, ...)
    job.sbatch      the rendered array script
    logs/           SLURM stdout per array element

The sweep config (TOML)::

    name = "gap_vs_c_l7"
    task = "scalar_gap_bisection"

    [resources]                  # optional; sbatch sizing
    cpus_per_task = 8
    time = "23:00:00"
    mem_per_cpu = "8G"

    [params]                     # shared by every task
    lam = 7
    tol = "0.05"

    [params.sdpb_options]        # merged over the task's default SDPB flags
    "--precision" = 768
    "--mpiProcs" = 8             # see docs/CLUSTER.md on MPI ranks in a job

    [sweep]                      # every list-valued key is swept (cartesian)
    c = ["2", "4", "8", "16", "32"]

Some parameters need to move together rather than as an independent cartesian
axis -- e.g. a (q_order, npoles) convergence ladder, where only specific pairs
are meaningful. Use ``[[sweep_rows]]`` for that: each entry is one correlated
combo (merged over ``[params]``), and the rows are then crossed with any
ordinary ``[sweep]`` keys (e.g. a shared ``delta_ext`` grid run at every row)::

    [[sweep_rows]]
    q_order = 60
    npoles = 20

    [[sweep_rows]]
    q_order = 80
    npoles = 60

    [sweep]
    delta_ext = ["1.5", "2.0", "2.5"]   # crossed with EVERY row above

Omitting ``[[sweep_rows]]`` behaves exactly as before (one implicit empty row).

The cluster profile (TOML) carries scheduler and environment facts.
``kind = "local"`` runs the specs in-process
instead of submitting (same specs, same worker), so the laptop is just
another profile.
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

from .worker import result_doc_is_final, run_spec, spec_hash, _task_result_path
from .inputs import (
    HpcInputError, cli_input_errors, input_integer, load_run_manifest, path_argument,
    read_job_spec, resolve_local_runs_root,
)
from ..paths import (
    canonical_research_relative_path,
    resolve_research_path,
)

DEFAULT_RESOURCES = {
    "cpus_per_task": 8,
    "time": "23:00:00",
    # Allow headroom above the measured 8.7 GiB total peak for a Lambda=19
    # moment-bound run with 16 CPUs. Sweeps can override this request.
    "mem_per_cpu": "4G",
}
# Send USR1 to the batch step this long before the walltime so the worker can
# stop at a safe point, flush its journal, and requeue itself.
SIGNAL_LEAD_S = 600


def _toml_float(token: str) -> float | int | str:
    """Keep a decimal's exact value through task JSON and numeric parsing."""
    token = token.replace("_", "")
    exact = Decimal(token)
    native = float(token)
    if not exact.is_finite():
        return native  # Numerical-field validation rejects TOML inf/nan.
    spelling = Decimal(str(native))
    if not spelling.is_finite():
        return token  # Retain large exponents without expanding a JSON integer.
    # The real-input layer reads floats via their shortest decimal spelling;
    # integer controls instead read the binary value. Preserve both contracts.
    if spelling == exact and (
            not native.is_integer() or Decimal.from_float(native) == exact):
        return native
    if exact == exact.to_integral_value():
        return int(exact)
    return token


def load_toml(path: Path) -> dict:
    try:
        with open(path, "rb") as f:
            document = tomllib.load(f, parse_float=_toml_float)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, InvalidOperation) as exc:
        raise HpcInputError(f"cannot read {path}: {exc}") from exc
    # All original values go into the JSON manifest and task specs. TOML's
    # date/time objects need quoting at the input, before any remote staging.
    try:
        json.dumps(document)
    except TypeError as exc:
        raise HpcInputError(
            f"{path}: input cannot be preserved in JSON ({exc}); "
            "quote TOML dates and times as strings"
        ) from exc
    return document


ENV_CLUSTER = "VB_CLUSTER"


def _cluster_env_default() -> Path | None:
    """Default for every --cluster flag: $VB_CLUSTER if set, else None.

    Meant to be set once per machine (e.g. in ~/.bashrc on the cluster
    itself, alongside the PATH/conda setup already needed there), so none of
    vb-submit/-status/-collect/-list need --cluster repeated on every call.
    """
    value = os.environ.get(ENV_CLUSTER)
    return Path(value) if value else None


def expand_sweep(config: dict) -> list[dict]:
    """Return one params dict per task: base [params] x [[sweep_rows]] x cartesian [sweep].

    ``[[sweep_rows]]`` holds params that must move together (e.g. a
    (q_order, npoles) ladder); each row is crossed with the ordinary
    cartesian ``[sweep]`` keys. No ``[[sweep_rows]]`` is exactly one implicit
    empty row, so plain cartesian-only configs are unaffected. An explicitly
    empty row list produces no tasks.
    """
    base = dict(config.get("params", {}))
    rows = config.get("sweep_rows", [{}])
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise ValueError("[[sweep_rows]] must be an array of tables")

    sweep = config.get("sweep", {})
    keys = list(sweep)
    for key in keys:
        if not isinstance(sweep[key], list):
            raise ValueError(f"[sweep] {key} must be a list, got {type(sweep[key]).__name__}")
    combos = list(itertools.product(*(sweep[k] for k in keys))) if keys else [()]

    return [
        {**base, **row, **dict(zip(keys, combo))}
        for row in rows
        for combo in combos
    ]


def check_sweep(config: dict, sweep_path: Path | str = "sweep") -> None:
    """Exit with every problem :func:`~.params.validate_sweep` finds, if any."""
    from .params import validate_sweep

    errors = validate_sweep(config, expand=expand_sweep)
    if errors:
        raise SystemExit(
            f"{sweep_path}: not submitted:\n" + "\n".join(f"  - {e}" for e in errors)
        )


def _git_commit(repo_hint: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_hint), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _array_throttle(cluster: dict) -> int | None:
    throttle = cluster.get("max_array_throttle")
    if throttle is None or throttle is False:
        return None
    return input_integer("max_array_throttle", throttle, minimum=0)


def _shell_expanding_value(value) -> str:
    """Quote literal characters while retaining shell variable expansion."""
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _shell_expanding_path(raw: str) -> str:
    """Quote a path while retaining shell home and variable expansion."""
    home, slash, remainder = raw.partition("/")
    # Bash expands a tilde only when it is unquoted. Quote the remainder so
    # spaces and literal quotation marks stay inside this one path argument.
    is_home = re.fullmatch(r"~[A-Za-z0-9_.-]*", home) is not None
    if is_home and not slash:
        return home
    prefix = home + "/" if slash and is_home else ""
    body = remainder if prefix else raw
    return prefix + _shell_expanding_value(body)


def _shell_profile_python(cluster: dict) -> str:
    """Quote an interpreter path while retaining home and variable expansion."""
    raw = cluster.get("python", "python")
    if not isinstance(raw, str) or not raw.strip() or "\0" in raw:
        raise HpcInputError("profile python must be a nonempty interpreter path")
    return _shell_expanding_path(raw)


#: Every key a cluster profile may set (docs/CLUSTER.md describes each one).
#: ``host`` is read only by the development tree's cluster helpers.
PROFILE_KEYS = frozenset({
    "kind", "partition", "account", "runs_root", "python", "modules", "setup",
    "sbatch_extra", "env", "remote_repo", "max_array_throttle", "host",
})


def _validate_cluster_profile(cluster: dict) -> None:
    """Reject profile shapes that cannot be rendered or dispatched as written."""
    if not isinstance(cluster, dict):
        raise HpcInputError("cluster profile must be a table")
    unknown = sorted(str(key) for key in cluster if key not in PROFILE_KEYS)
    if unknown:
        # A misspelled key (`partiton`, `acount`) would otherwise be dropped
        # silently and the job submitted without its directive.
        import difflib

        hints = []
        for key in unknown:
            close = difflib.get_close_matches(key, sorted(PROFILE_KEYS), n=1)
            hints.append(f"{key!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        raise HpcInputError(
            f"unknown cluster profile key(s) {', '.join(hints)}; "
            f"valid keys: {', '.join(sorted(PROFILE_KEYS))}"
        )
    if cluster.get("kind", "slurm") not in ("slurm", "local"):
        raise HpcInputError("profile kind must be 'slurm' or 'local'")
    for key in ("modules", "setup", "sbatch_extra"):
        values = cluster.get(key, [])
        if (not isinstance(values, (list, tuple))
                or any(not isinstance(value, str) or "\0" in value for value in values)):
            raise HpcInputError(f"profile {key} must be a list of strings without NUL")
    env = cluster.get("env", {})
    if not isinstance(env, dict):
        raise HpcInputError("profile [env] must be a table")
    for key, value in env.items():
        if not isinstance(key, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None:
            raise HpcInputError(f"profile [env] key {key!r} is not a shell variable name")
        if not isinstance(value, (str, int, float, bool)) or "\0" in str(value):
            raise HpcInputError(f"profile [env] {key} must be a scalar value without NUL")
    if "VB_SDPB_MODE" in env:
        from .runner import SDPB_MODES

        if env["VB_SDPB_MODE"] not in SDPB_MODES:
            raise HpcInputError(
                f"profile [env] VB_SDPB_MODE = {env['VB_SDPB_MODE']!r} is not one of "
                f"{', '.join(repr(mode) for mode in SDPB_MODES)}"
            )
    for key in ("runs_root", "remote_repo"):
        value = cluster.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip() or "\0" in value):
            raise HpcInputError(f"profile {key} must be a nonempty path string without NUL")
    _shell_profile_python(cluster)
    _array_throttle(cluster)


def _sbatch_value(name: str, value) -> str:
    """Keep a generated directive value inside one Slurm argument."""
    text = str(value)
    if any(char in text for char in ("\n", "\r", "\0")):
        raise HpcInputError(f"{name} must not contain newlines or NUL")
    if any(char.isspace() or char in "\"'\\#" for char in text):
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return text


def render_sbatch(
    *,
    name: str,
    run_dir: Path,
    n_tasks: int,
    resources: dict,
    cluster: dict,
) -> str:
    _validate_cluster_profile(cluster)
    run_dir = Path(run_dir).resolve()
    # No throttle by default: SLURM already does fair-share, backfill and
    # resource limits, so an array cap just makes our own run slower without
    # helping anyone else. Set max_array_throttle only for a resource SLURM
    # cannot see -- e.g. shared-filesystem capacity, since each task here writes
    # ~670 MB of SDP intermediates (a 654-task run is ~440 GB).
    n_tasks = input_integer("n_tasks", n_tasks, minimum=1)
    cpus = input_integer("cpus_per_task", resources.get("cpus_per_task", DEFAULT_RESOURCES["cpus_per_task"]),
                         minimum=1)
    throttle = _array_throttle(cluster)
    partition = resources.get("partition") or cluster.get("partition")
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={_sbatch_value('job name', name)}",
    ]
    if partition:
        lines.append(f"#SBATCH --partition={_sbatch_value('partition', partition)}")
    if cluster.get("account"):
        lines.append(f"#SBATCH --account={_sbatch_value('account', cluster['account'])}")
    lines += [
        f"#SBATCH --time={_sbatch_value('time', resources.get('time', DEFAULT_RESOURCES['time']))}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem-per-cpu={_sbatch_value('mem_per_cpu', resources.get('mem_per_cpu', DEFAULT_RESOURCES['mem_per_cpu']))}",
        f"#SBATCH --array=0-{n_tasks - 1}"
        + (f"%{throttle}" if throttle else ""),
        # Slurm treats percent signs and backslashes as filename controls.
        # Keep the run directory literal, outside the stdout filename pattern.
        f"#SBATCH --chdir={_sbatch_value('working directory', run_dir)}",
        "#SBATCH --output=logs/task-%a.out",
        # Requeued elements append to their log instead of truncating it.
        "#SBATCH --open-mode=append",
        # Interruption is the normal case: allow requeue and give the worker a
        # pre-wall warning (B: = deliver to the batch step, which IS the python
        # process because the last line exec's it).
        "#SBATCH --requeue",
        f"#SBATCH --signal=B:USR1@{SIGNAL_LEAD_S}",
    ]
    for extra in cluster.get("sbatch_extra", []):
        lines.append(f"#SBATCH {extra}")
    lines.append("")
    for module in cluster.get("modules", []):
        lines.append(f"module load {module}")
    for setup in cluster.get("setup", []):
        lines.append(setup)
    for key, value in cluster.get("env", {}).items():
        # Double quotes: let $PROJECT-style references expand on the cluster.
        lines.append(f'export {key}={_shell_expanding_value(value)}')
    # Absolute interpreter path (profile key `python`) beats environment
    # activation in batch scripts: conda's shell hooks are fragile in
    # non-interactive shells, while <env>/bin/python needs no activation.
    python_exe = _shell_profile_python(cluster)
    lines += [
        "",
        f"cd {shlex.quote(str(run_dir))} || exit",
        f'exec {python_exe} -m virasoro_bootstrap.hpc.worker '
        '"specs/$(printf \'%04d\' "$SLURM_ARRAY_TASK_ID").json"',
        "",
    ]
    return "\n".join(lines)


# --- --host: SSH transport for laptop submission -----------------------------

def _git(repo_hint: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo_hint), *args], capture_output=True, text=True
    )


def _require_clean_and_pushed(repo_root: Path, *, allow_dirty: bool = False) -> str:
    """Return HEAD's commit hash, or exit if it's dirty or not on the pushed upstream.

    ``--host`` runs whatever the remote can ``git fetch``, so an uncommitted or
    unpushed change to a *tracked* file would silently run stale code on the
    cluster. Untracked files are exempt: ``git checkout <sha>`` on the remote
    can never see them either way, so an unrelated scratch file sitting in the
    working tree shouldn't block submission.

    ``allow_dirty`` downgrades the dirty check to a warning, for the common case
    where the uncommitted work is in a part of the tree the run does not touch.
    It does NOT ship those edits: the cluster still runs HEAD. The unpushed
    check is never downgraded, because the remote ``git checkout`` would simply
    fail on a commit origin has never seen.
    """
    dirty = _git(repo_root, "status", "--porcelain", "--untracked-files=no")
    if dirty.stdout.strip():
        if not allow_dirty:
            raise SystemExit(
                "--host needs a clean working tree (uncommitted changes to tracked "
                "files found); commit (and push) first, or pass --allow-dirty to "
                "submit HEAD anyway:\n" + dirty.stdout
            )
        print(
            "warning: --allow-dirty: submitting HEAD with a dirty working tree.\n"
            "  The cluster runs HEAD. The changes below are NOT in the run:\n"
            + "".join(f"  {line}\n" for line in dirty.stdout.strip().splitlines()),
            file=sys.stderr,
        )
    head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    upstream = _git(repo_root, "rev-parse", "@{u}")
    if upstream.returncode != 0:
        raise SystemExit(
            "--host needs the current branch to have a pushed upstream "
            "(the remote fetches from origin, not from your working tree)"
        )
    if upstream.stdout.strip() != head:
        raise SystemExit(
            "local HEAD is ahead of (or diverged from) the pushed upstream; "
            "`git push` first -- --host submits whatever origin has"
        )
    return head


def _ssh_run(host: str, remote_repo: str, remote_shell_cmd: str) -> subprocess.CompletedProcess:
    # Profile paths expand home/variables remotely, while literal quotes and
    # backslashes remain part of the one directory argument.
    inner = f'cd -- {_shell_expanding_path(remote_repo)} && {{\n{remote_shell_cmd}\n}}'
    # ssh joins ALL trailing argv elements with a space each and hands that ONE
    # string to the remote shell to re-parse from scratch -- so passing
    # ["bash", "-lc", inner] as 3 separate argv elements is not safe: the
    # remote re-splits "-lc"'s argument at the first space, so only a bare
    # `cd` (no path) becomes the actual script and everything else -- the
    # fetch, checkout, worker invocation -- runs as later top-level commands
    # in the wrong directory. Collapsing to one already-quoted argv element
    # leaves ssh nothing to rejoin.
    outer = f"bash -lc {shlex.quote(inner)}"
    return subprocess.run(["ssh", host, outer], capture_output=True, text=True)


def _ssh_exec(host: str, remote_shell_cmd: str) -> subprocess.CompletedProcess:
    """Run a command on the host without assuming a checkout to cd into."""
    outer = f"bash -lc {shlex.quote(remote_shell_cmd)}"
    return subprocess.run(["ssh", host, outer], capture_output=True, text=True)


def _remote_expand(host: str, remote_path: str, *, remote_repo: str | None = None) -> str:
    """Resolve a runs root in the remote submit process's working directory.

    Profile values may use either, and only the remote shell knows what they
    mean -- expanding ``$HOME`` locally would
    give the laptop's.

    A leading ``~`` is rewritten to ``$HOME`` first, because tilde expansion
    does not happen inside the double quotes that make ``$VARS`` expand and
    that keep a path with spaces in one piece.  ``~otheruser`` is not
    supported; profiles name a directory the submitting account owns.
    """
    if remote_path == "~":
        remote_path = "$HOME"
    elif remote_path.startswith("~/"):
        remote_path = "$HOME/" + remote_path[2:]
    command = (
        f'vb_runs_path={_shell_expanding_value(remote_path)}; '
        'test -n "$vb_runs_path" || exit 1; '
        'case "$vb_runs_path" in /*) ;; *) vb_runs_path="$PWD/$vb_runs_path";; esac; '
        'printf %s "$vb_runs_path"'
    )
    result = (_ssh_run(host, remote_repo, command) if remote_repo
              else _ssh_exec(host, command))
    if result.returncode != 0 or not result.stdout:
        raise SystemExit(
            f"could not expand {remote_path!r} on {host}: "
            f"{result.stderr.strip() or 'no output'}"
        )
    return result.stdout


def _scp_to(host: str, local: Path, remote_dir: str) -> None:
    result = subprocess.run(
        ["scp", "-q", str(local.resolve()), f"{host}:{remote_dir}/{local.name}"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"could not copy {local} to {host}:{remote_dir}: {result.stderr.strip()}"
        )


def _library_checkout() -> Path | None:
    """Return the git checkout this package is running from, if any.

    ``--host`` checks out a commit remotely to pin the *code*.  That only means
    something when the local library is itself a checkout; an ordinary
    ``pip install`` has no commit to pin, and that is a supported way to run.
    """
    from ..paths import _source_checkout_root

    # Not repository_root(): outside a source checkout that falls back to the
    # working directory, which may be the user's own, unrelated repository.
    root = _source_checkout_root()
    if root is None or not (root / ".git").exists():
        return None
    return root if _git(root, "rev-parse", "HEAD").returncode == 0 else None


def _print_ssh_result(result: subprocess.CompletedProcess, *, what: str) -> None:
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode != 0:
        raise SystemExit(f"remote {what} failed (ssh exit {result.returncode})")


def _validate_run_id(value: str) -> None:
    if (not isinstance(value, str) or not value.strip() or "\0" in value
            or value in (".", "..") or Path(value).name != value):
        raise HpcInputError("run_id must be a nonempty directory name without path separators")


def submit_remote(
    sweep_path: Path,
    cluster_path: Path,
    *,
    host: str,
    runs_root: Path | None = None,
    run_id: str | None = None,
    dry_run: bool = False,
    allow_dirty: bool = False,
) -> Path | None:
    cluster = load_toml(cluster_path)
    _validate_cluster_profile(cluster)
    local_root = resolve_local_runs_root(runs_root)
    if cluster.get("kind") == "local":
        raise SystemExit(
            "cluster profile kind='local' isn't for --host: it would run tasks "
            "in-process on the remote login node instead of submitting to SLURM"
        )
    remote_repo = cluster.get("remote_repo")

    # Pin the CODE, not the inputs.  The remote checkout is what makes the
    # cluster run a known commit; it only means something when the local
    # library is itself a checkout.  A pip-installed library has no commit to
    # pin, which is a supported way to run -- the manifest then records the
    # inputs by value, as it always has, and the remote runs whatever version
    # is installed there.
    checkout = _library_checkout() if remote_repo else None
    commit = (
        _require_clean_and_pushed(checkout, allow_dirty=allow_dirty)
        if checkout
        else None
    )
    dirty_files = (
        [
            line.strip()
            for line in _git(
                checkout, "status", "--porcelain", "--untracked-files=no"
            ).stdout.strip().splitlines()
        ]
        if checkout
        else []
    )

    sweep_cfg = load_toml(sweep_path)
    # Validate here, before anything is copied or checked out remotely; the
    # remote vb-submit validates again against the code it actually runs.
    check_sweep(sweep_cfg, sweep_path)
    name = sweep_cfg.get("name") or sweep_path.stem
    resolved_run_id = (run_id if run_id is not None
                       else f"{datetime.datetime.now():%Y%m%d-%H%M%S}-{name}")
    _validate_run_id(resolved_run_id)
    python_exe = _shell_profile_python(cluster)

    # Copy inputs over SSH and record their contents in the manifest.
    # Sweep and profile files can live outside a git checkout.
    profile_runs_root = cluster.get("runs_root")
    if not profile_runs_root:
        raise SystemExit(f"{cluster_path} has no runs_root; required for --host")
    # Validate the profile before SSH creates the remote input directory. The
    # actual script is rendered remotely using its expanded run path.
    render_sbatch(
        name=name, run_dir=Path(str(profile_runs_root)) / resolved_run_id,
        n_tasks=len(expand_sweep(sweep_cfg)),
        resources={**DEFAULT_RESOURCES, **sweep_cfg.get("resources", {})},
        cluster=cluster,
    )
    remote_root = _remote_expand(host, str(profile_runs_root), remote_repo=remote_repo)
    remote_run = f"{remote_root}/{resolved_run_id}"
    remote_inputs = f"{remote_run}/inputs"
    remote_sweeps = f"{remote_inputs}/sweep"
    remote_profiles = f"{remote_inputs}/cluster"
    # Reserve a fresh run before any copies. Existing inputs belong to that
    # run even when its submission failed or was only a dry run.
    mk = _ssh_exec(host, f"mkdir -p -- {shlex.quote(remote_root)} && "
                   f"mkdir -- {shlex.quote(remote_run)} && mkdir -- {shlex.quote(remote_inputs)} "
                   f"{shlex.quote(remote_sweeps)} {shlex.quote(remote_profiles)}")
    if mk.returncode != 0:
        raise SystemExit(
            f"could not create run {remote_run} on {host} (use a new --run-id if it exists): "
            f"{mk.stderr.strip()}"
        )
    _scp_to(host, sweep_path, remote_sweeps)
    _scp_to(host, cluster_path, remote_profiles)
    if dirty_files:
        # --allow-dirty. The cluster runs HEAD, so these edits are NOT in the
        # run; record them beside it so a later reader can tell that the tree
        # this was launched from did not match the commit that ran.
        import tempfile

        note = (
            f"Submitted with --allow-dirty at {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}\n"
            f"The run executes commit {commit}. The following tracked files had\n"
            "uncommitted changes at submit time and are NOT part of this run:\n\n"
            + "\n".join(f"  {line}" for line in dirty_files)
            + "\n"
        )
        with tempfile.TemporaryDirectory() as td:
            note_path = Path(td) / "dirty_at_submit.txt"
            note_path.write_text(note)
            _scp_to(host, note_path, remote_inputs)

    remote_sweep = f"{remote_sweeps}/{sweep_path.name}"
    remote_cluster = f"{remote_profiles}/{cluster_path.name}"
    remote_submit = (
        f'{python_exe} -m virasoro_bootstrap.hpc.submit {shlex.quote(remote_sweep)} '
        f"--cluster {shlex.quote(remote_cluster)} --run-id {shlex.quote(resolved_run_id)}"
        + (" --dry-run" if dry_run else "")
    )

    if commit is not None:
        remote_cmd = " && ".join(
            ["git fetch origin", f"git checkout {commit}", remote_submit]
        )
        print(
            f"--host {host}: checking out {commit[:12]} on {remote_repo} "
            "and submitting..."
        )
        result = _ssh_run(host, remote_repo, remote_cmd)
    else:
        where = "the installed package" if remote_repo is None else remote_repo
        print(f"--host {host}: submitting with {where} (no commit pinned)...")
        result = (
            _ssh_run(host, remote_repo, remote_submit)
            if remote_repo
            else _ssh_exec(host, remote_submit)
        )
    _print_ssh_result(result, what="submission")
    if dry_run:
        return None

    match = re.search(r"^run .*: .* under (.+)$", result.stdout, re.MULTILINE)
    remote_run_dir = match.group(1) if match else None

    local_run_dir = (local_root / resolved_run_id).resolve()
    local_run_dir.mkdir(parents=True, exist_ok=True)
    if remote_run_dir is None:
        print(
            "warning: could not parse the remote run directory from output; "
            "manifest not copied back", file=sys.stderr,
        )
        return local_run_dir

    scp = subprocess.run(
        ["scp", "-q", f"{host}:{remote_run_dir}/manifest.json",
         str(local_run_dir / "manifest.json")],
        capture_output=True, text=True,
    )
    if scp.returncode == 0:
        print(f"copied manifest to {local_run_dir / 'manifest.json'}")
    else:
        print(f"warning: could not copy manifest back: {scp.stderr.strip()}", file=sys.stderr)
    return local_run_dir


def retry_remote(remote_run_dir: str, cluster_path: Path, *, host: str, dry_run: bool = False) -> None:
    """Resubmit a remote run's pending tasks without re-fetching (retry uses
    whatever commit is already checked out there -- it should not silently
    change code versions underneath an in-flight run)."""
    cluster = load_toml(cluster_path)
    _validate_cluster_profile(cluster)
    remote_repo = cluster.get("remote_repo")
    python_exe = _shell_profile_python(cluster)
    remote_cmd = (
        f'{python_exe} -m virasoro_bootstrap.hpc.submit --retry {shlex.quote(remote_run_dir)}'
        + (" --dry-run" if dry_run else "")
    )
    result = (_ssh_run(host, remote_repo, remote_cmd) if remote_repo
              else _ssh_exec(host, remote_cmd))
    _print_ssh_result(result, what="retry")


def _apply_local_env(cluster: dict) -> None:
    """kind=local runs in-process; the profile's [env] applies to us directly."""
    _validate_cluster_profile(cluster)
    for key, value in cluster.get("env", {}).items():
        os.environ[key] = os.path.expandvars(str(value))


def _pending_indices(run_dir: Path, n_tasks: int) -> list[int]:
    """Array indices without a valid result: none, a stale hash, or a recorded failure."""
    from .provenance import execution_settings

    pending = []
    for i in range(n_tasks):
        spec_path = run_dir / "specs" / f"{i:04d}.json"
        spec = read_job_spec(spec_path)
        result_path = _task_result_path(run_dir, i, spec.get("output_dir"))
        try:
            existing = json.loads(result_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            existing = None
        params = spec.get("params", {})
        if result_doc_is_final(existing, spec_hash(spec["task"], params),
                               execution=execution_settings(spec["task"], params)):
            continue
        pending.append(i)
    return pending


def _run_job_ids(manifest: dict) -> list[str]:
    """Every SLURM array submitted for a run: the original and each retry (older manifests: the latest)."""
    jobs = manifest.get("slurm_job_ids")
    ids = [str(job) for job in jobs] if isinstance(jobs, list) else []
    latest = manifest.get("slurm_job_id")
    if latest is not None and str(latest) not in ids:
        ids.append(str(latest))
    return ids


def _queued_array_indices(manifest: dict) -> set[int]:
    """Task indices of a run whose array element is still queued or running (empty without squeue).

    Array indices are task indices for the original submission and every retry.
    """
    ids = set(_run_job_ids(manifest))
    if not ids or shutil.which("squeue") is None:
        return set()
    try:
        text = subprocess.run(["squeue", "-h", "-r", "-u", str(os.getuid()), "-o", "%F %K"],
                              check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"warning: squeue failed ({exc}); tasks still queued may be resubmitted", file=sys.stderr)
        return set()
    queued = set()
    for line in text.splitlines():
        base, _, index = line.strip().partition(" ")
        if base in ids and index.isdigit():
            queued.add(int(index))
    return queued


def _sbatch(args: list[str], run_dir: Path) -> str:
    out = subprocess.run(
        ["sbatch", *args], check=True, capture_output=True, text=True, cwd=run_dir
    )
    # "Submitted batch job 12345"
    return out.stdout.strip().split()[-1]


def _update_manifest(run_dir: Path, **updates) -> dict:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(updates)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def python_preflight(cluster: dict) -> list[str]:
    """Warn if the profile's interpreter cannot import the package.

    ``python`` defaults to bare ``python``, which on a cluster is often a
    system interpreter with nothing installed in it.  The resulting failure --
    ``No module named virasoro_bootstrap`` -- surfaces inside a SLURM log after
    the task has been queued and scheduled, which is an expensive place to
    learn it.
    """
    if cluster.get("modules") or cluster.get("setup"):
        return []
    raw = str(cluster.get("python", "python"))
    try:
        python_exe = str(Path(os.path.expandvars(raw)).expanduser())
    except RuntimeError:
        return [f"profile python {raw!r} has a home directory that cannot be resolved here"]
    resolved = shutil.which(python_exe)
    if resolved is None:
        return [
            f"profile python {raw!r} not found here; tasks will fail unless "
            "the cluster provides it"
        ]
    probe = subprocess.run(
        [resolved, "-c", "import virasoro_bootstrap"],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        return [
            f"profile python {raw!r} cannot import virasoro_bootstrap; "
            f"install the package into that interpreter (pip install -e .)"
        ]
    return []


def sdpb_preflight(cluster: dict) -> list[str]:
    """Warn about an SDPB toolchain that looks unreachable, before submitting.

    Otherwise a profile pointing at a missing ``.sif`` or an unbuilt ``sdpb``
    is only discovered once tasks have been queued and scheduled -- possibly
    hundreds of them, hours later.

    Warnings rather than errors: this runs where the submission happens, and a
    cluster may legitimately provide the toolchain only on compute nodes (via
    ``modules``/``setup``), so an unreachable-looking tool here is suggestive,
    not conclusive.
    """
    from .runner import ENV_IMAGE, missing_sdpb_tools, resolve_sdpb_runner

    env = {str(k): str(v) for k, v in cluster.get("env", {}).items()}
    if cluster.get("modules") or cluster.get("setup"):
        # The job's environment is built by steps we cannot evaluate here.
        return []

    saved = dict(os.environ)
    try:
        os.environ.update({k: os.path.expandvars(v) for k, v in env.items()})
        try:
            runner = resolve_sdpb_runner()
        except HpcInputError as exc:
            # An invalid VB_SDPB_MODE in the profile is refused earlier, by
            # _validate_cluster_profile. What remains (e.g. apptainer with no
            # image here) may still be supplied on the worker.
            return [f"{exc}; tasks will fail unless the cluster provides it"]
        missing = missing_sdpb_tools(runner)
        if missing:
            where = runner.bin_dir or "PATH"
            return [
                f"SDPB mode is 'native' but {', '.join(missing)} not found in "
                f"{where}; tasks will fail unless the cluster provides it"
            ]
        if runner.mode == "apptainer":
            image = runner.image
            if image and not Path(image).exists():
                return [
                    f"{ENV_IMAGE} points at {image}, which does not exist here; "
                    "tasks will fail unless it exists on the worker"
                ]
    finally:
        os.environ.clear()
        os.environ.update(saved)
    return []


def _mpi_procs(params: dict) -> int | None:
    """The ``--mpiProcs`` a task's ``sdpb_options`` requests, if any."""
    options = params.get("sdpb_options")
    if not isinstance(options, dict):
        return None
    procs = None
    for flag, value in options.items():
        name, equals, inline = str(flag).partition("=")
        if name != "--mpiProcs" or value is False or value is None:
            continue
        try:
            procs = int(str(inline if equals else value))
        except ValueError:
            return None  # check_sweep reports the malformed value
    return procs


def mpi_preflight(cluster: dict, all_params: list[dict], resources: dict) -> list[str]:
    """Warn about ``--mpiProcs`` settings that a SLURM allocation will not run as written.

    Each array element is ONE SLURM task with ``cpus_per_task`` CPUs, and the
    worker launches ``sdpb`` inside it. Open MPI's ``mpirun`` counts allocated
    tasks, not CPUs, so the default ``mpirun -n {n}`` refuses more than one rank
    ("not enough slots") unless the launcher allows oversubscription. Which
    flag does that depends on the MPI implementation, so the profile sets it.
    """
    if cluster.get("kind", "slurm") == "local":
        return []
    procs = max((n for params in all_params if (n := _mpi_procs(params)) is not None), default=None)
    if procs is None or procs <= 1:
        return []
    warnings = []
    cpus = resources.get("cpus_per_task", DEFAULT_RESOURCES["cpus_per_task"])
    try:
        if procs > int(cpus):
            warnings.append(
                f"sdpb_options --mpiProcs {procs} exceeds cpus_per_task {cpus}; "
                "the ranks will share the task's CPUs"
            )
    except (TypeError, ValueError):
        pass
    env = cluster.get("env", {})
    launcher_set = "VB_MPI_LAUNCHER" in env or any(
        "VB_MPI_LAUNCHER" in line for line in cluster.get("setup", []))
    multi_task = any(re.match(r"\s*(--ntasks\b|-n\b|--ntasks-per-node\b)", extra)
                     for extra in cluster.get("sbatch_extra", []))
    if not launcher_set and not multi_task and env.get("VB_SDPB_MODE") != "docker":
        warnings.append(
            f"sdpb_options --mpiProcs {procs} uses the default launcher 'mpirun -n {{n}}' inside a "
            "one-task SLURM allocation, which Open MPI refuses ('not enough slots'); set "
            "VB_MPI_LAUNCHER in the profile's [env], e.g. \"mpirun --oversubscribe -n {n}\" "
            "for Open MPI (docs/CLUSTER.md, 'MPI ranks inside a job')"
        )
    return warnings


def create_run(
    sweep_path: Path,
    cluster_path: Path,
    *,
    runs_root: Path | None = None,
    run_id: str | None = None,
    dry_run: bool = False,
) -> Path:
    sweep = load_toml(sweep_path)
    cluster = load_toml(cluster_path)
    _validate_cluster_profile(cluster)
    check_sweep(sweep, sweep_path)
    name = sweep.get("name") or sweep_path.stem
    task_name = sweep["task"]
    all_params = expand_sweep(sweep)
    resources = {**DEFAULT_RESOURCES, **sweep.get("resources", {})}

    # expanduser + expandvars: profile paths use ~/... or $HOME/... so the same
    # profile works for any group member.
    raw_root = runs_root if runs_root is not None else cluster.get("runs_root")
    root = resolve_local_runs_root(raw_root)
    if run_id is None:
        run_id = f"{datetime.datetime.now():%Y%m%d-%H%M%S}-{name}"
    _validate_run_id(run_id)
    run_dir = (root / run_id).resolve()
    if run_dir.exists():
        # --host stages the sweep and profile into <run-dir>/inputs before
        # invoking this, so that alone is not a previous run. Anything else is.
        stray = sorted(p.name for p in run_dir.iterdir() if p.name != "inputs")
        if stray:
            raise SystemExit(
                f"run dir {run_dir} already exists (contains {', '.join(stray)})"
            )
    # Validate/render the complete batch script before leaving task specs or
    # directories behind for a profile that cannot be submitted.
    script = render_sbatch(
        name=name, run_dir=run_dir, n_tasks=len(all_params),
        resources=resources, cluster=cluster,
    )
    warnings = (*python_preflight(cluster), *sdpb_preflight(cluster),
                *mpi_preflight(cluster, all_params, resources))
    manifest = {
        "run_id": run_id,
        "name": name,
        "task": task_name,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "n_tasks": len(all_params),
        "sweep_file": str(sweep_path.resolve()),
        "sweep": sweep,
        "cluster_file": str(cluster_path.resolve()),
        "cluster": cluster,
        # The commit of this package's own checkout; None for an installed
        # package, even one inside some other project's git repository.
        "git_commit": _git_commit(checkout) if (checkout := _library_checkout()) else None,
        "slurm_job_id": None,
    }
    # The manifest contains every original parameter/profile value. Encoding
    # it first also checks that task specs can preserve those values in JSON.
    manifest_text = json.dumps(manifest, indent=2)
    (run_dir / "specs").mkdir(parents=True)
    (run_dir / "tasks").mkdir()
    (run_dir / "logs").mkdir()

    for i, params in enumerate(all_params):
        (run_dir / "specs" / f"{i:04d}.json").write_text(
            json.dumps(
                {"task": task_name, "params": params, "output_dir": f"tasks/{i:04d}"},
                indent=2,
            )
        )

    sbatch_path = run_dir / "job.sbatch"
    sbatch_path.write_text(script)

    (run_dir / "manifest.json").write_text(manifest_text)

    kind = cluster.get("kind", "slurm")
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"run {run_id}: {len(all_params)} task(s) under {run_dir}")
    if dry_run:
        print(f"dry run: wrote {sbatch_path}, not submitting")
        return run_dir

    if kind == "local":
        _apply_local_env(cluster)
        for i in range(len(all_params)):
            rc = run_spec(run_dir / "specs" / f"{i:04d}.json")
            if rc not in (0, 3):
                print(f"task {i:04d} exited {rc}", file=sys.stderr)
        return run_dir

    job_id = _sbatch([str(sbatch_path)], run_dir)
    _update_manifest(run_dir, slurm_job_id=job_id)
    print(f"submitted SLURM job {job_id}")
    return run_dir


def _check_retry_profile(manifest: dict, run_dir: Path, cluster_path: Path) -> None:
    """Refuse a --cluster profile that differs from the one the run recorded.

    A retry resubmits the run's saved ``job.sbatch``, which was rendered from
    the profile embedded in ``manifest.json``. Accepting a different profile
    and then ignoring it would silently submit to the old partition, account,
    and environment.
    """
    given = json.loads(json.dumps(load_toml(cluster_path)))
    if given != manifest.get("cluster", {}):
        raise HpcInputError(
            f"--retry resubmits {run_dir / 'job.sbatch'}, rendered from the profile recorded in "
            f"its manifest.json; --cluster {cluster_path} differs from that profile. Omit --cluster "
            "to retry with the recorded profile, or submit the sweep again as a new run with the "
            "new profile"
        )


def retry_run(run_dir: Path, *, dry_run: bool = False, cluster_path: Path | None = None) -> None:
    """Resubmit a run's unfinished tasks with its saved job script and profile.

    *cluster_path*, when given, must hold the profile the run recorded.
    """
    run_dir = run_dir.resolve()
    manifest, n_tasks = load_run_manifest(run_dir)
    cluster = manifest.get("cluster", {})
    _validate_cluster_profile(cluster)
    if cluster_path is not None:
        _check_retry_profile(manifest, run_dir, cluster_path)
    pending = _pending_indices(run_dir, n_tasks)
    if not pending:
        print("all tasks have valid results; nothing to retry")
        return
    kind = cluster.get("kind", "slurm")
    if kind != "local":
        # A task of this run that is still queued or running has no result yet, but
        # resubmitting it starts a duplicate that finds its output busy.
        queued = _queued_array_indices(manifest)
        active = [i for i in pending if i in queued]
        if active:
            print(f"skipping {len(active)} task(s) still queued or running: {active}")
            pending = [i for i in pending if i not in queued]
        if not pending:
            print("every unfinished task is still queued or running; nothing to retry")
            return
    print(f"retrying {len(pending)} task(s): {pending}")
    if kind == "local":
        if dry_run:
            return
        _apply_local_env(cluster)
        for i in pending:
            rc = run_spec(run_dir / "specs" / f"{i:04d}.json")
            if rc not in (0, 3):
                print(f"task {i:04d} exited {rc}", file=sys.stderr)
        return
    throttle = _array_throttle(cluster)
    array_arg = "--array=" + ",".join(map(str, pending))
    if throttle:
        array_arg += f"%{throttle}"
    if dry_run:
        print(f"dry run: sbatch {array_arg} job.sbatch")
        return
    job_id = _sbatch([array_arg, str(run_dir / "job.sbatch")], run_dir)
    _update_manifest(run_dir, slurm_job_id=job_id, slurm_job_ids=[*_run_job_ids(manifest), job_id])
    print(f"submitted SLURM job {job_id}")


SUBMIT_DESCRIPTION = """\
Expand a sweep file into one job spec per task and submit them as a SLURM
array (or run them in-process with a kind = "local" profile).

examples:
  vb-submit sweep.toml --cluster mycluster.toml --dry-run   # validate + render only
  vb-submit sweep.toml --cluster mycluster.toml
  vb-submit --retry <run-dir>                # rerun failed/missing tasks (recorded profile)
  vb-submit sweep.toml --cluster mycluster.toml --host <ssh-alias>
  vb-submit --list-tasks                    # tasks and the parameters they accept
  vb-submit --list-tasks scalar_gap_bisection

The sweep is validated before anything is written: unknown tasks or parameter
keys, missing required parameters, malformed values and an empty [sweep] are
errors. Set $VB_CLUSTER to make --cluster optional. Afterwards: vb-status,
vb-collect and vb-certify take the run id that vb-submit and vb-list print.
See docs/CLUSTER.md for the sweep and profile formats.
"""


@cli_input_errors("vb-submit")
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vb-submit", description=SUBMIT_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("sweep", nargs="?", type=Path, help="sweep config TOML")
    parser.add_argument("--cluster", type=Path, default=None,
                        help=f"cluster profile TOML (default: ${ENV_CLUSTER})")
    parser.add_argument("--retry", type=Path, metavar="RUN_DIR",
                        help="resubmit a run's tasks that have no successful result, using the "
                             "run's recorded profile and job.sbatch (with --host, --cluster "
                             "only says how to reach the cluster)")
    parser.add_argument("--runs-root", type=path_argument, default=None,
                        help="where run directories go (default: the profile's runs_root)")
    parser.add_argument("--run-id", default=None,
                        help="name of the run directory (default: <timestamp>-<sweep name>)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate, write the specs and job.sbatch, but submit nothing")
    parser.add_argument(
        "--list-tasks", nargs="*", metavar="TASK", default=None,
        help="list the available tasks and their parameters (with a TASK: its full description) and exit",
    )
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="with --host and remote_repo: submit even with uncommitted changes. "
             "The cluster still runs HEAD, not your edits; the run records which "
             "files differed. HEAD must still be pushed.",
    )
    parser.add_argument(
        "--host", default=None, metavar="SSH_ALIAS",
        help="submit on this SSH host (from ~/.ssh/config): copies the sweep and "
             "profile there and runs vb-submit remotely",
    )
    args = parser.parse_args(argv)
    cluster_given = args.cluster is not None
    if not cluster_given:
        args.cluster = _cluster_env_default()

    if args.list_tasks is not None:
        from .params import describe_tasks

        try:
            print(describe_tasks(args.list_tasks or None), end="")
        except KeyError as exc:
            parser.error(str(exc.args[0]) if exc.args else str(exc))
        return 0

    # Keep old top-level ``sweeps/`` and ``clusters/`` command lines working while
    # making the canonical paths recorded in manifests and remote commands
    # point into the relocated tree. This is now the only compatibility layer
    # that still accepts the older spellings -- existing
    # manifests and cluster handoffs depend on it.
    if args.sweep is not None:
        args.sweep = resolve_research_path(args.sweep)
    if args.cluster is not None:
        args.cluster = resolve_research_path(args.cluster)

    if args.retry is not None:
        # A retry reuses the run's directory, id, and job script; these options
        # name a new run and would otherwise be silently ignored.
        ignored = [flag for flag, given in (
            ("a sweep file", args.sweep is not None),
            ("--runs-root", args.runs_root is not None),
            ("--run-id", args.run_id is not None),
            ("--allow-dirty", args.allow_dirty),
        ) if given]
        if ignored:
            parser.error(f"--retry reuses the run's directory, id, and job script; "
                         f"it does not take {', '.join(ignored)}")

    if args.host:
        if args.cluster is None:
            parser.error("--host needs --cluster")
        if args.retry is not None:
            retry_remote(str(args.retry), args.cluster, host=args.host, dry_run=args.dry_run)
            return 0
        if args.sweep is None:
            parser.error("need a sweep config for --host submission")
        submit_remote(
            args.sweep, args.cluster, host=args.host,
            runs_root=args.runs_root, run_id=args.run_id, dry_run=args.dry_run,
            allow_dirty=args.allow_dirty,
        )
        return 0

    if args.retry is not None:
        # Only an explicit --cluster is checked against the recorded profile:
        # $VB_CLUSTER names this machine's current profile, which may have
        # changed since the run was submitted.
        retry_run(args.retry, dry_run=args.dry_run,
                  cluster_path=args.cluster if cluster_given else None)
        return 0
    if args.sweep is None or args.cluster is None:
        parser.error("need a sweep config and --cluster profile (or --retry RUN_DIR)")
    create_run(
        args.sweep, args.cluster,
        runs_root=args.runs_root, run_id=args.run_id, dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
