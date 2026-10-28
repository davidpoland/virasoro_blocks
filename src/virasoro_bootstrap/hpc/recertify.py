"""vb-certify: rebuild a finished run's problem and certify its SDPB solution.

    vb-certify <run-dir> --task 0 --probe gap_3p0
    vb-certify <run-dir> --task 0                        # every probe (same as --all-probes)
    vb-certify <run-dir> --all-tasks --last-probe        # the largest-gap probe of each task

Why this exists
---------------
:func:`virasoro_bootstrap.certification.certify_dual_from_sdpb_output` is
already generic over problems and checks the functional against the true blocks
rather than the SDP's own nodes: a gap problem is evaluated on a finite Delta
grid of the exact blocks, while normalized-moment PMP paths also decide
positivity of the solver's represented polynomials by root isolation.  What was missing
was the other half: a finished cluster run leaves behind an ``sdpb_out`` directory
and a problem manifest, but not the in-memory problem the certificate needs.  Without a way to rebuild
it, every verdict has to be taken on the solver's word -- which is how a whole
investigation can end up arguing about SDPB terminateReason strings instead of
checking a functional.

This module reloads each probe's manifest and hash-checked block files, then
hands the saved problem to the existing certifier. Task parameters and rounded
probe names are not sufficient to reconstruct the original SDP reliably.

Supported tasks
---------------
:data:`PROBLEM_BUILDERS` lists the supported scalar-gap task names. Its
parameter-based builders remain available for standalone callers, but
:func:`certify_probe` always reloads the saved manifest and its hash-checked
blocks. Adding support for a different problem kind requires an appropriate
manifest loader and certification route; task parameters or probe names do
not establish which problem was solved.

The verdict
-----------
``passed`` per channel is the certifier's verdict: the finite-grid true-block
check for a gap problem, and on the normalized-moment PMP paths the
root-isolation check as well. Missing data and invalid numerical evaluations
also prevent a check from proving positivity. A negative verdict requires a
finite real witness or a negative polynomial certificate; an incomplete check
is reported as NOT PROVEN. A normalization violation or nonfinite dual vector
is an INVALID functional, without implying that a channel went negative.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mpmath as mp

from ..defaults import (
    DEFAULT_LARGE_SPIN_POLE_SELECTION,
    DEFAULT_POLE_SELECTION,
)
from .collect import _local_runs_root, _normalize_cluster_arg, resolve_run_dir_arg
from .inputs import (
    HpcInputError, cli_input_errors, load_run_manifest, nonnegative_integer,
    path_argument, positive_integer, read_job_spec,
)
from .submit import _cluster_env_default
from .worker import _task_output_dir


def _lambda_preset(lam: int) -> dict:
    from .tasks import LAMBDA_PRESETS

    return LAMBDA_PRESETS[lam]


def build_scalar_gap_problem(params: dict, probe_value: mp.mpf | None):
    """Rebuild one ``scalar_gap_bisection`` probe's problem.

    ``probe_value`` is the bisected gap for that probe. Mirrors the task's own
    ``make_spec`` -- if that changes, this must follow, which is why both live
    in this package rather than in a diagnostics script.
    """
    from ..identical_scalar import (
        IdenticalScalarGapSpec,
        build_identical_scalar_gap_problem,
    )
    from ..large_spin import double_twist_set, twists_above_gap
    from .tasks import (
        _central_charge, _task_integer, _task_boolean, _task_spins, _task_real_input,
    )

    lam = _task_integer("lam", params.get("lam", 7))
    preset = _lambda_preset(lam)
    dps = _task_integer("dps", params.get("dps", preset["dps"]))
    q_order = _task_integer("q_order", params.get("q_order", preset["q_order"]))
    npoles = _task_integer("npoles", params.get("npoles", preset["npoles"]))
    num_threads = _task_integer("num_threads", params.get("num_threads", 4))
    gap = _task_real_input("gap", probe_value if probe_value is not None else params.get("gap_lo", "0.5"))

    self_consistent = _task_boolean("self_consistent_h_ext", params.get("self_consistent_h_ext", False))
    h_ext = gap / 2 if self_consistent else _task_real_input("h_ext", params.get("h_ext", "0.0625"))
    spins = _task_spins(params.get("spins", (0, 2, 4, 6, 8, 10)))
    lmax = max(spins) + 2
    n_tw = _task_integer("large_spin_double_twists", params.get(
        "large_spin_double_twists",
        5 if preset["large_spin_double_twists"] is None else preset["large_spin_double_twists"]))
    tower_npoles = _task_integer("large_spin_npoles", params.get(
        "large_spin_npoles",
        preset["npoles"] if preset["large_spin_npoles"] is None else preset["large_spin_npoles"]))

    tower_kwargs: dict = {}
    if _task_boolean("tower", params.get("tower", True)):
        mode = str(params.get("twist_gap", "none"))
        if mode == "none":
            twists = tuple(double_twist_set(h_ext, n_tw))
        elif mode == "match_gap":
            twists = tuple(twists_above_gap(double_twist_set(h_ext, n_tw), gap))
        else:
            twists = tuple(twists_above_gap(double_twist_set(h_ext, n_tw),
                                          _task_real_input("twist_gap", params["twist_gap"])))
        tower_kwargs = dict(
            large_spin_lmin=lmax,
            large_spin_twists=twists,
            large_spin_pole_selection=str(
                params.get("large_spin_pole_selection", DEFAULT_LARGE_SPIN_POLE_SELECTION)
            ),
            large_spin_npoles=tower_npoles,
        )

    delta_min_by_spin = None
    if _task_boolean("all_spins_twist_gap", params.get("all_spins_twist_gap", False)):
        delta_min_by_spin = {ell: ell + gap for ell in spins if ell != 0}

    spec = IdenticalScalarGapSpec(
        c=_central_charge(params["c"]),
        h_ext=h_ext,
        delta_gap=gap,
        delta_min_by_spin=delta_min_by_spin,
        lam=lam,
        spin_list=spins,
        q_order=q_order,
        dps=dps,
        npoles=npoles,
        pole_selection=str(params.get("pole_selection", DEFAULT_POLE_SELECTION)),
        num_threads=num_threads,
        **tower_kwargs,
    )
    return build_identical_scalar_gap_problem(spec)


#: Supported scalar-gap tasks and their standalone parameter-based builders.
#: certify_probe uses the names as an allowlist and loads the saved manifest.
PROBLEM_BUILDERS = {
    "scalar_gap_bisection": build_scalar_gap_problem,
    # Modular gap problems require a separate builder. Matching vector
    # lengths alone does not make their functionals compatible with sphere blocks.
}


def load_spec(run_dir: Path, task_index: int) -> dict:
    path = Path(run_dir) / "specs" / f"{task_index:04d}.json"
    try:
        return read_job_spec(path, require_output=False)
    except HpcInputError as exc:
        raise SystemExit(str(exc)) from exc


def probe_dirs(run_dir: Path, task_index: int) -> list[Path]:
    spec = load_spec(run_dir, task_index)
    task_dir = _task_output_dir(Path(run_dir), task_index, spec.get("output_dir"))
    if not task_dir.is_dir():
        return []
    return sorted(p for p in task_dir.iterdir() if (p / "sdpb_out").is_dir())


def incomplete_probe_reason(probe_dir: Path) -> str | None:
    """Why a probe has no finished SDPB solve to certify, or None when it has one.

    The solve helpers record the source (state ``running``) before SDPB starts,
    so ``sdpb_out`` alone does not mean the solve finished.
    """
    from .storage import recorded_sdp_state

    sdpb_out = Path(probe_dir) / "sdpb_out"
    try:
        state = recorded_sdp_state(sdpb_out)
    except (OSError, ValueError) as exc:
        return f"unreadable solver record: {exc}"
    if state not in (None, "completed"):
        return f"SDPB attempt {state}, not completed"
    if not (sdpb_out / "y.txt").is_file():
        return "no y.txt (no dual solution was written)"
    return None


def _probe_gap_from_manifest(probe_dir: Path):
    """Read the exact saved gap for ordering probes; certification checks hashes."""
    from virasoro_blocks.numeric_input import exact_real

    path = probe_dir / "manifest.json"
    try:
        doc = json.loads(path.read_text())
        raw = doc["spec"]["delta_gap"]
        gap = None if isinstance(raw, bool) else exact_real(raw)
        if gap is None or gap < 0:
            raise ValueError("delta_gap must be a nonnegative real number")
        return gap
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"cannot read probe gap from manifest {path}: {exc}") from exc


def certify_probe(run_dir, task_index: int, probe: str, **cert_kwargs):
    """Reload one probe's saved problem and certify its SDPB output.

    Returns ``(ok, details, info)``; ``info`` carries what was rebuilt so a
    caller can report it without re-deriving.
    """
    from ..certification import certify_dual_from_sdpb_output

    run_dir = Path(run_dir)
    spec = load_spec(run_dir, task_index)
    task = spec.get("task")
    builder = PROBLEM_BUILDERS.get(task)
    if builder is None:
        raise SystemExit(
            f"no problem builder registered for task {task!r}; "
            f"known: {sorted(PROBLEM_BUILDERS)}"
        )
    probe_dir = _task_output_dir(run_dir, task_index, spec.get("output_dir")) / probe
    sdpb_out = probe_dir / "sdpb_out"
    if not sdpb_out.is_dir():
        raise SystemExit(f"no sdpb_out at {sdpb_out}")
    reason = incomplete_probe_reason(probe_dir)
    if reason is not None:
        raise SystemExit(f"unfinished probe: {reason}")
    manifest_path = probe_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(
            f"no manifest at {manifest_path}; the original solved problem cannot "
            "be recovered reliably from task parameters and a rounded probe name"
        )
    from ..manifest import build_identical_scalar_problem_from_manifest

    try:
        problem = build_identical_scalar_problem_from_manifest(manifest_path)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"cannot load probe manifest {manifest_path}: {exc}") from exc
    if problem.problem_kind != "gap":
        raise SystemExit(f"probe manifest describes {problem.problem_kind!r}, not a scalar gap problem")
    dps = int(problem.spec.dps)
    value = problem.spec.delta_gap
    with mp.workdps(dps):
        try:
            ok, details = certify_dual_from_sdpb_output(problem, sdpb_out, **cert_kwargs)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"cannot certify {sdpb_out}: {exc}") from exc
    return ok, details, {
        "task": task, "probe": probe, "probe_value": value, "dps": dps,
        "manifest": str(manifest_path),
    }


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _channel(detail: dict) -> str:
    """``ell=2`` for a fixed-spin channel, ``ell=12 twist=0.25`` for a tower slice."""
    parts = [f"ell={detail.get('ell')}"]
    if detail.get("twist") is not None:
        parts.append(f"twist={_fmt(detail['twist'])}")
    return " ".join(parts)


def _negative_where(detail: dict) -> str:
    """Where a failing channel goes negative, from whichever keys its path carries.

    The root-isolation path reports the first negative point
    (``first_negative_delta`` / ``first_negative_x``); the finite-grid gap path
    reports the grid minimum (``min_delta`` / ``min_x`` with ``min_value``); a
    structural failure (missing data, normalization) carries only a ``reason``.
    """
    parts = []
    if detail.get("first_negative_delta") is not None:
        parts.append(f"first negative at Delta~{_fmt(detail['first_negative_delta'])}")
    elif detail.get("first_negative_x") is not None:
        parts.append(f"first negative at x~{_fmt(detail['first_negative_x'])}")
    elif detail.get("first_negative_sample_index") is not None:
        parts.append(f"first negative at sample {detail['first_negative_sample_index']}")
    elif detail.get("min_delta") is not None:
        parts.append(f"min at Delta~{_fmt(detail['min_delta'])}")
    elif detail.get("min_x") is not None:
        parts.append(f"min at x~{_fmt(detail['min_x'])}")
    value = detail.get("first_negative_value")
    if value is None:
        value = detail.get("min_value")
    if value is not None:
        parts.append(f"value {_fmt(value)}")
    if detail.get("n_negative"):
        parts.append(f"{detail['n_negative']}/{detail.get('n_test_points', '?')} points failed")
    if detail.get("reason"):
        parts.append(f"({detail['reason']})")
    return ", ".join(parts) if parts else "(no location reported)"


def _has_negative_witness(detail: dict) -> bool:
    if (detail.get("exact_positivity") or {}).get("passed") is False:
        return True
    # Sampled checks separate finite real witnesses from their failure counts.
    # Polynomial diagnostics without a separate witness carry a real minimum.
    key = "first_negative_value" if "first_negative_value" in detail else "min_value"
    if detail.get(key) is None:
        return False
    try:
        value = mp.mpf(detail[key])
    except (TypeError, ValueError):
        return False
    return mp.isfinite(value) and value < 0


def _summarize(ok, details, info) -> bool:
    n = len(details)
    passed = 0
    invalid, negative, undecided = [], [], []
    for detail in details:
        if detail.get("reason") in {"normalization_violated", "non_finite_dual_vector"}:
            invalid.append(detail)
        elif _has_negative_witness(detail):
            negative.append(detail)
        elif (detail.get("passed") is True and not detail.get("exact_positivity_undecided")
              and ("exact_positivity" not in detail
                   or detail["exact_positivity"].get("passed") is True)):
            passed += 1
        else:
            undecided.append(detail)
    valid = bool(ok and n and passed == n)
    val = "-" if info["probe_value"] is None else mp.nstr(info["probe_value"], 6)
    print(f"  {info['probe']:<16} value={val:<10} {passed}/{n} channels positive")
    for d in negative[:6]:
        print(f"      NEGATIVE  {_channel(d)}  {_negative_where(d)}")
    for d in invalid[:6]:
        reason = d["reason"]
        if d.get("normalization_value") is not None:
            reason += f" (n.y={_fmt(d['normalization_value'])}, expected 1)"
        print(f"      INVALID functional  {reason}")
    for d in undecided[:6]:
        reason = d.get("exact_positivity_undecided") or d.get("reason") or "no positivity proof"
        if reason == "no positivity proof" and d.get("n_negative") and "first_negative_value" in d:
            reason = "nonfinite or nonreal sample values"
        print(f"      UNDECIDED {_channel(d)}  ({reason})")
    verdict = ("INVALID (negative channel)" if negative else "INVALID functional" if invalid
               else "VALID functional" if valid else "NOT PROVEN (incomplete checks)")
    print(f"      -> {verdict}")
    return valid


CLI_DESCRIPTION = """\
Reload a finished run's saved problem manifest and check that each
probe's SDPB dual functional is non-negative on the true blocks, rather than
taking the solver's terminateReason on trust.

examples:
  vb-certify <run-id> --task 0 --probe gap_3p0
  vb-certify <run-id> --task 0                   # every probe of the task
  vb-certify <run-id> --all-tasks --last-probe   # the largest-gap probe of each task

Each probe prints "k/n channels positive" and a verdict: VALID functional,
INVALID (a negative channel or invalid functional), or NOT PROVEN (missing data
or a channel whose positivity could not be decided). Exit status 0 only when
every checked probe is VALID. Supported tasks: {tasks}.
"""


@cli_input_errors("vb-certify")
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="vb-certify",
        description=CLI_DESCRIPTION.format(tasks=", ".join(sorted(PROBLEM_BUILDERS))),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("run_dir", type=str, metavar="RUN",
                    help="run id (as vb-list prints it), run directory, or SLURM job id")
    task_selection = ap.add_mutually_exclusive_group(required=True)
    task_selection.add_argument("--task", type=nonnegative_integer, default=None,
                                help="task index (the NNNN of specs/NNNN.json)")
    task_selection.add_argument("--all-tasks", action="store_true", help="certify every task of the run")
    probe_selection = ap.add_mutually_exclusive_group()
    probe_selection.add_argument("--probe", default=None, help="probe directory name, e.g. gap_3p0")
    probe_selection.add_argument("--all-probes", action="store_true",
                                 help="certify every probe of each task (the default)")
    probe_selection.add_argument("--last-probe", action="store_true",
                    help="only the numerically largest saved gap; this need not be the final bisection bracket")
    ap.add_argument("--extra-spins", type=nonnegative_integer, default=0,
                    help="also check this many even spins beyond the solved spin list")
    ap.add_argument("--extra-twists", type=nonnegative_integer, default=0,
                    help="also check this many twists between each pair of tower slices")
    ap.add_argument("--num-threads", type=positive_integer, default=1, help="worker processes")
    ap.add_argument("--runs-root", type=path_argument, default=None,
                    help="where to look up a run id (default: --cluster's runs_root, else ./artifacts/runs)")
    ap.add_argument("--cluster", type=Path, default=_cluster_env_default(),
                    help="cluster profile whose runs_root to search (default: $VB_CLUSTER)")
    args = ap.parse_args(argv)
    _normalize_cluster_arg(args)
    args.run_dir = resolve_run_dir_arg(args.run_dir, _local_runs_root(args))
    _manifest, n_tasks = load_run_manifest(args.run_dir)

    if args.all_tasks:
        tasks = range(n_tasks)
    else:
        if args.task >= n_tasks:
            raise HpcInputError(f"task index {args.task} is outside this run's {n_tasks} task(s)")
        tasks = [args.task]

    cert_kwargs = dict(
        extra_spins=args.extra_spins,
        extra_twists=args.extra_twists,
        num_threads=args.num_threads,
        warn=False,
    )
    all_ok = True
    for t in tasks:
        try:
            directories = probe_dirs(args.run_dir, t)
        except SystemExit as exc:
            print(f"task {t:04d}: SKIPPED: {exc}")
            all_ok = False
            continue
        probes = [p.name for p in directories]
        if not probes:
            print(f"task {t:04d}: no probes with sdpb_out")
            all_ok = False
            continue
        if args.probe:
            probes = [args.probe]
        elif args.last_probe:
            finished = [p for p in directories if incomplete_probe_reason(p) is None]
            if not finished:
                print(f"task {t:04d}: SKIPPED: no probe has a finished SDPB solve")
                all_ok = False
                continue
            try:
                valued = [(_probe_gap_from_manifest(p), p.name) for p in finished]
            except SystemExit as exc:
                print(f"task {t:04d}: SKIPPED: {exc}")
                all_ok = False
                continue
            probes = [max(valued)[1]]
        # Otherwise (--all-probes, or no probe flag) every probe is checked:
        # a default of one arbitrary probe would report VALID for a run whose
        # other functionals were never looked at.
        print(f"task {t:04d}:")
        for probe in probes:
            try:
                ok, details, info = certify_probe(args.run_dir, t, probe, **cert_kwargs)
            except SystemExit as exc:
                print(f"  {probe:<16} SKIPPED: {exc}")
                all_ok = False
                continue
            all_ok = _summarize(ok, details, info) and all_ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
