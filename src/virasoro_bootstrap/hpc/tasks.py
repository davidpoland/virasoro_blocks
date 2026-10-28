"""Task registry: named, JSON-parameterized units of cluster work.

A *task* is the unit that runs inside one SLURM array element (or one local
process): self-contained, parameterized by a plain JSON-serializable dict
(mpmath values as strings, matching the manifest conventions), writing
everything under its own output directory, and journaling enough progress to
resume mid-chain after an interruption.

Task functions receive a :class:`TaskContext` and return a JSON-serializable
result dict. They must (a) consult ``ctx.journal`` on start and record
completed probes as they go, and (b) poll ``ctx.should_stop()`` between
solver calls, raising :class:`TaskInterrupted` when set -- the worker turns
that into a clean self-requeue instead of a failure.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import mpmath as mp

from virasoro_blocks.requests import _integer_field

from .journal import Journal, progress_mapping
from ..defaults import (
    DEFAULT_LARGE_SPIN_POLE_SELECTION,
    DEFAULT_POLE_SELECTION,
    load_default_params,
    _profile_from_snapshot,
    profile as _profile,
)
from ..paths import research_config_path


class TaskInterrupted(Exception):
    """The task stopped at a safe point in response to should_stop()."""


@dataclass
class TaskContext:
    params: dict
    output_dir: Path
    journal: Journal
    should_stop: Callable[[], bool] = field(default=lambda: False)
    execution: dict | None = None

    def profile(self, lam: int, *, moment: bool = False) -> dict:
        """Use the snapshot checked by the worker, or the direct-call defaults."""
        if self.execution is not None:
            return self.execution["source"]["profile"]
        return _load_default_params_profile(lam) if moment else LAMBDA_PRESETS[lam]

    def selector(self, name: str) -> str:
        if self.execution is not None:
            return self.execution["source"]["selectors"][name]
        return globals()["DEFAULT_" + name.upper()]


TASKS: dict[str, Callable[[TaskContext], dict]] = {}

#: The module constant holding each task's default SDPB flags, by name so that
#: reuse checks read the same value the task body does.
TASK_SDPB_DEFAULTS: dict[str, str] = {}


def task(name: str, *, sdpb_defaults: str | None = None):
    """Register a task function under *name* (its job-spec ``task`` key)."""

    def register(fn: Callable[[TaskContext], dict]):
        if name in TASKS:
            raise ValueError(f"duplicate task name {name!r}")
        TASKS[name] = fn
        if sdpb_defaults is not None:
            TASK_SDPB_DEFAULTS[name] = sdpb_defaults
        return fn

    return register


#: A task reports a failure it caught itself (so the result can still carry
#: its logs) by returning a result whose ``reason`` starts with one of these.
#: The worker records such a result as ``failed`` and ``vb-submit --retry``
#: reruns it; every other ``reason`` (``lo_excluded``, ``hi_allowed_at_cap``,
#: ...) is an answer, not a failure.
ERROR_REASON_PREFIXES = ("solve_error", "prepare_error")


def result_error(result) -> str | None:
    """The failure a task's result reports, or None for a genuine answer."""
    if not isinstance(result, dict):
        return "missing or malformed task result (expected a JSON object)"
    reason = result.get("reason")
    if isinstance(reason, str) and reason.startswith(ERROR_REASON_PREFIXES):
        error = result.get("error")
        return f"{reason}: {error}" if error else reason
    # Moment tasks report one outcome per identity convention and direction.
    # A completed but uncertified solve is a result; an execution/source error
    # is unfinished work and must remain eligible for worker/submission retry.
    for mode in ("include", "exclude"):
        outcomes = result.get(mode)
        if not isinstance(outcomes, dict):
            continue
        for direction in ("lower", "upper"):
            status = outcomes.get(f"{direction}_status")
            if isinstance(status, str) and status.startswith(ERROR_REASON_PREFIXES):
                return f"{mode}_{direction}: {status}"
            entry = outcomes.get(direction)
            if isinstance(entry, dict) and entry.get("error"):
                return f"{mode}_{direction}: {entry['error']}"
    return None


def get_task(name: str) -> Callable[[TaskContext], dict]:
    try:
        return TASKS[name]
    except KeyError:
        raise KeyError(
            f"unknown task {name!r}; registered tasks: {sorted(TASKS)}"
        ) from None


def _task_integer_value(name, value) -> int:
    """Parse an integral task value, including integer strings from job files."""
    if isinstance(value, str):
        if re.fullmatch(r"[+-]?\d+", value.strip()) is None:
            raise ValueError(f"{name} must be an integer")
        value = int(value)
    return _integer_field(name, value)


def _task_integer(name, value, *, minimum: int | None = None) -> int:
    """Validate task integers while retaining integer strings from job files."""
    integer = _task_integer_value(name, value)
    if minimum is None:
        minimum = {"dps": 10, "modular_dps": 1, "four_point_dps": 10,
                   "lam": 1, "q_order": 1, "precision": 1,
                   "num_threads": 1, "cert_n_test_points": 1}.get(name, 0)
    if integer < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return integer


def _task_optional_integer(name, value) -> int | None:
    return None if value is None else _task_integer(name, value)


def _task_boolean(name, value) -> bool:
    """Match submission validation; quoted booleans must not enable flags."""
    if isinstance(value, bool) or (type(value) is int and value in (0, 1)):
        return bool(value)
    raise ValueError(f"{name} must be true/false (without quotes), or 0/1")


def _task_spins(values) -> tuple[int, ...]:
    from ..identical_scalar import _validate_even_spins

    if isinstance(values, (str, bytes)):
        raise ValueError("spins must be a sequence of integers")
    try:
        values = tuple(values)
    except TypeError:
        raise ValueError("spins must be a sequence of integers") from None
    return _validate_even_spins(tuple(_task_integer("spin", s) for s in values))


def _task_modular_spins(values) -> tuple[int, ...]:
    from ..modular_crossing import _validate_spins

    if isinstance(values, (str, bytes)):
        raise ValueError("modular_spins must be a sequence of integers")
    try:
        values = tuple(values)
    except TypeError:
        raise ValueError("modular_spins must be a sequence of integers") from None
    return _validate_spins(tuple(_task_integer("spin", s) for s in values))


def _task_exchange_spin(value) -> int:
    spin = _task_integer("spin_star", value)
    if spin % 2:
        raise ValueError("spin_star must be a nonnegative even integer")
    return spin


def _task_choice_list(name, values, choices) -> tuple:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence chosen from {choices}")
    try:
        values = tuple(values)
    except TypeError:
        raise ValueError(f"{name} must be a sequence chosen from {choices}") from None
    if any(value not in choices for value in values):
        raise ValueError(f"{name} must be a sequence chosen from {choices}")
    return values


def _task_real_list(name, values, *, positive=False, ordered=False) -> tuple:
    from collections.abc import Mapping

    if isinstance(values, (str, bytes, Mapping)) or values is None:
        raise ValueError(f"{name} must be a nonempty sequence of real numbers")
    try:
        values = tuple(values)
    except TypeError:
        raise ValueError(f"{name} must be a nonempty sequence of real numbers") from None
    if not values:
        raise ValueError(f"{name} must be a nonempty sequence of real numbers")
    try:
        labels = tuple(_task_real_input(f"{name} entry", value) for value in values)
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise ValueError(f"{name} entries must be finite real numbers: {exc}") from exc
    if any(value <= 0 if positive else value < 0 for value in labels):
        raise ValueError(f"{name} entries must be {'positive' if positive else 'nonnegative'}")
    if ordered and any(b <= a for a, b in zip(labels, labels[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return labels


def _preset_from_profile(prof: dict) -> dict[str, int | None]:
    """The block-table knobs the task layer reads, pulled off a lambdaN profile."""

    def _opt(key: str) -> int | None:
        value = prof.get(key)
        return None if value is None else _task_integer(key, value)

    return {
        "q_order": _task_integer("q_order", prof["q_order"]),
        "npoles": _task_integer("npoles", prof["npoles"]),
        "dps": _task_integer("dps", prof["dps"]),
        "large_spin_npoles": _opt("large_spin_npoles"),
        "large_spin_double_twists": _opt("large_spin_double_twists"),
    }


def _presets_from_snapshot(data: dict) -> dict[int, dict[str, int | None]]:
    return {
        lam: _preset_from_profile(_profile_from_snapshot(data, lam))
        for lam in sorted(
            int(m.group(1))
            for m in (re.fullmatch(r"lambda(\d+)", key) for key in data)
            if m is not None
        )
    }


#: Block-table parameters per derivative order Lambda, read from the packaged
#: ``default_params.json`` so every task uses the same profiles. Per-profile
#: ``_comment`` fields record provenance.
#:
#: Keys are exactly what the task layer consumes. Anything else in a profile
#: (spin_list, sdpb_precision, cert coverage, ...) is deliberately not surfaced
#: through here -- tasks that want those read the profile directly.
LAMBDA_PRESETS = _presets_from_snapshot(load_default_params())


# Feasibility-verdict solver defaults (a binary search only needs the
# allowed/excluded verdict, so stop at the first feasible point on either side
# or at a primal or dual feasible jump, whichever comes first).
DEFAULT_FEASIBILITY_SDPB_OPTIONS: dict[str, str | bool | int] = {
    "--precision": 768,
    "--findPrimalFeasible": True,
    "--findDualFeasible": True,
    "--detectPrimalFeasibleJump": True,
    "--detectDualFeasibleJump": True,
    # the objective is zero, so a solve must never end as "optimal" before a verdict
    "--dualityGapThreshold": "1e-200",
    "--writeSolution": "x,y",
    "--noFinalCheckpoint": True,
}

# OPE-max runs to OPTIMALITY: no find*Feasible, no jump detectors. The objective
# is real, so SDPB converges and `optimal` carries a usable dualObjective; a
# runaway shows up as max_complementarity, which is the SIGNAL that delta_star
# is above the gap bound and never itself a bound.
DEFAULT_OPE_MAX_SDPB_OPTIONS: dict[str, str | bool | int] = {
    "--precision": 768,
    "--maxIterations": 1000,
    "--dualityGapThreshold": "1e-10",
    "--writeSolution": "x,y",
    "--noFinalCheckpoint": True,
}

DEFAULT_GRID_FEASIBILITY_SDPB_OPTIONS: dict[str, str | bool | int] = {
    "--precision": 768,
    "--findPrimalFeasible": True,
    "--findDualFeasible": True,
    "--detectPrimalFeasibleJump": True,
    "--detectDualFeasibleJump": True,
    "--dualityGapThreshold": "1e-200",
    "--writeSolution": "x,y",
    "--noFinalCheckpoint": True,
}

def _merged_sdpb_options(defaults, options, *, execution=None) -> dict[str, str | bool | int]:
    """A task's SDPB flags: the sweep's ``sdpb_options`` merged over its defaults.

    Sweep values win, flag by flag; every default the sweep does not mention
    is kept. A flag written in the equals form (``"--maxIterations=500"``)
    replaces the default of the same flag, and ``false`` drops a default
    (``"--findPrimalFeasible" = false``).
    """
    if execution is not None:
        return dict(execution["effective"]["sdpb_options"])
    merged = dict(defaults)
    if not options:
        return merged
    if not isinstance(options, dict):
        raise ValueError("sdpb_options must be a table of SDPB flags")
    for flag, value in options.items():
        name = str(flag).partition("=")[0]
        for existing in [key for key in merged if str(key).partition("=")[0] == name]:
            del merged[existing]
        if value is not False and value is not None:
            merged[flag] = value
    return merged


# Exit the solver this many seconds before the SLURM allocation ends, leaving
# room to write the checkpoint and for the worker to flush its journal.
_WALLTIME_MARGIN_S = 900


def _slurm_max_runtime_seconds() -> int | None:
    """Remaining allocation seconds (minus margin), or None off-SLURM.

    Derived from ``SLURM_JOB_END_TIME`` (epoch seconds) so every SDPB solve can
    carry ``--maxRuntime`` and always exits cleanly -- checkpoint written --
    before the wall, instead of being SIGKILLed mid-iteration.
    """
    end_text = os.environ.get("SLURM_JOB_END_TIME")
    if not end_text:
        return None
    try:
        remaining = int(end_text) - int(time.time()) - _WALLTIME_MARGIN_S
    except ValueError:
        return None
    return max(remaining, 60)


def _central_charge(value):
    """Keep the exact input label until the block builder chooses its precision."""
    from ..identical_scalar import _central_charge_input

    return _central_charge_input(value)


def _task_real_input(name, value):
    """Validate a real task input while retaining its exact source label."""
    from ..identical_scalar import _as_real_label

    return _as_real_label(value, name=name)


def _validate_bisection_tolerance(tol, name="tol"):
    message = f"{name} must be finite and positive"
    try:
        tol = _task_real_input(name, tol)
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise ValueError(message) from exc
    if tol <= 0:
        raise ValueError(message)
    return tol


def _validate_gap_bracket(lo, hi):
    message = "gap bracket must be finite with 0 <= gap_lo < gap_hi"
    try:
        lo = _task_real_input("gap_lo", lo)
        hi = _task_real_input("gap_hi", hi)
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise ValueError(message) from exc
    if not 0 <= lo < hi:
        raise ValueError(message)
    return lo, hi


def _task_twist_gap(value):
    """A tower gap is either a named mode or an exact real label."""
    if isinstance(value, str) and value in ("none", "match_gap"):
        return value
    return _task_real_input("twist_gap", value)


def _bisection_midpoint(lo, hi):
    from virasoro_blocks.numeric_input import literal_number

    mid = (mp.mpf(mp.convert(lo)) + mp.mpf(mp.convert(hi))) / 2
    if not literal_number(lo) < literal_number(mid) < literal_number(hi):
        raise ValueError("bisection cannot resolve the requested tolerance; increase dps")
    return mid


def _result_number(value) -> str:
    """Serialize a task result without losing its stored numerical precision."""
    from fractions import Fraction

    if isinstance(value, Fraction):
        from virasoro_blocks.writers import _metadata_num_str

        return _metadata_num_str(value, mp.mp.dps + 5)
    parts = value._mpc_ if isinstance(value, mp.mpc) else (
        (value._mpf_,) if isinstance(value, mp.mpf) else ()
    )
    bits = max((part[3] for part in parts), default=0)
    digits = max(mp.mp.dps + 5, (bits * 30103 + 99999) // 100000 + 5)
    return mp.nstr(value, digits)


def _label(value: mp.mpf) -> str:
    """A readable, bounded probe key that does not merge rounded values."""
    import hashlib
    from fractions import Fraction
    from virasoro_blocks.numeric_input import literal_number

    if not isinstance(value, mp.mpf):
        value = literal_number(value)
        if not isinstance(value, Fraction):
            raise ValueError("probe values must be finite real numbers")
        if value.denominator & (value.denominator - 1):
            with mp.workdps(25):
                text = mp.nstr(mp.mpf(mp.convert(value)), 17)
            if Fraction(text) == value:
                return text.replace(".", "p")
            payload = f"rational:{value.numerator:x}:{value.denominator:x}".encode("ascii")
            return text.replace(".", "p") + "~" + hashlib.sha256(payload).hexdigest()
        # Build the exact binary payload without rounding at ambient precision;
        # this retains existing keys for equivalent supplied mpf values.
        numerator = abs(value.numerator)
        trailing = (numerator & -numerator).bit_length() - 1 if numerator else 0
        mantissa = numerator >> trailing
        exponent = trailing - (value.denominator.bit_length() - 1) if numerator else 0
        value = mp.make_mpf((int(value < 0), mantissa, exponent, mantissa.bit_length()))
    if not mp.isfinite(value):
        raise ValueError("probe values must be finite")
    text = mp.nstr(value, 17)
    sign, mantissa, exponent, _ = value._mpf_
    # Keep the familiar short spelling only when it denotes this exact value.
    # Bound the rational conversion for extreme exponents; those use a digest.
    if abs(exponent) <= 4096:
        exact = Fraction((-1 if sign else 1) * mantissa) * Fraction(2) ** exponent
        if Fraction(text) == exact:
            return text.replace(".", "p")
    # The normalized mpf tuple is independent of ambient precision. A digest
    # keeps very high precision inputs below filesystem component limits.
    payload = f"{sign}:{mantissa:x}:{exponent}".encode("ascii")
    digest = hashlib.sha256(payload).hexdigest()
    return text.replace(".", "p") + "~" + digest


def sdp_structure_signature(problem) -> str:
    """A short digest of the SDP block structure a problem solves.

    Two problems with the same signature have the same number of positivity
    blocks with the same matrix sizes, polynomial degrees and sample counts, and
    the same functional length -- exactly what an SDPB checkpoint must match to
    seed a solve (``--initialCheckpointDir``); SDPB aborts on a mismatch.
    """
    import hashlib

    doc = problem.pmp_build.doc
    blocks = []
    for block in doc["PositiveMatrixWithPrefactorArray"]:
        polys = block["polynomials"]
        blocks.append((
            len(polys),
            len(polys[0]) if polys else 0,
            len(polys[0][0]) if polys and polys[0] else 0,
            max((len(v) for row in polys for entry in row for v in entry), default=0),
            len(block.get("samplePoints") or ()),
        ))
    shape = repr((len(doc.get("objective") or ()), tuple(blocks)))
    return hashlib.sha256(shape.encode()).hexdigest()[:16]


def _allowed_from_status(status: str | None) -> bool | None:
    if status == "primal_feasible":
        return True
    if status == "dual_feasible":
        return False
    return None


#: SDPB outcomes that say nothing about feasibility. A solve that stopped on
#: one of these is retried or reported as inconclusive, never scored as a
#: verdict. ``max_runtime`` never reaches a verdict either: see
#: :func:`_checked_sdpb_status`.
_INCONCLUSIVE_SDPB_STATUSES = frozenset({
    None,
    "solver_error",
    "max_complementarity",
    "max_iterations",
    "max_runtime",
    "primal_step_too_small",
    "dual_step_too_small",
})


def _checked_sdpb_status(run, where: str) -> str | None:
    """Return a solve's SDPB status, raising TaskInterrupted for ``--maxRuntime``.

    SDPB's ``--maxRuntime`` is the clean pre-wall exit this layer requests
    (SDPB checkpoints and quits). It is not a verdict, so it surfaces as an
    interruption: the worker requeues and the next allocation repeats the
    solve. The log-text checks cover a classifier that predates the explicit
    ``max_runtime`` and ``max_iterations`` statuses.
    """
    status = run.sdpb_status
    log_text = ""
    log_path = getattr(run, "sdpb_log_path", None)
    if status is None and log_path is not None and Path(log_path).exists():
        log_text = Path(log_path).read_text(errors="replace")
    if status == "max_runtime" or (status is None and "maxRuntime exceeded" in log_text):
        raise TaskInterrupted(f"SDPB hit --maxRuntime at {where}")
    if status is None and "maxIterations exceeded" in log_text:
        status = "max_iterations"
    return status


def _resume_probes(journal: Journal) -> tuple[dict, dict[str, bool]]:
    """Reuse only explicit boolean verdicts; invalid saved work is recomputed."""
    probes = {
        key: entry for key, entry in progress_mapping(journal.data, "probes").items()
        if "allowed" in entry
        and (entry["allowed"] is None or type(entry["allowed"]) is bool)
    }
    cache = {key: entry["allowed"] for key, entry in probes.items()
             if type(entry["allowed"]) is bool}
    return probes, cache


def _has_finite_ope_bound(entry) -> bool:
    if not isinstance(entry, dict) or entry.get("status") != "optimal":
        return False
    try:
        return mp.isfinite(mp.mpf(entry["ope_squared_bound"]))
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return False


def _has_saved_moment_bound(entry) -> bool:
    """A completed solve has explicit outcome fields and consistent finite values."""
    if not isinstance(entry, dict) or not all(
            key in entry for key in ("value", "solved_value", "status", "solve_seconds")):
        return False
    status = entry["status"]
    if (status is not None and not isinstance(status, str)) or status == "solve_error":
        return False
    try:
        if _task_real_input("saved solve_seconds", entry["solve_seconds"]) < 0:
            return False
        if status in ("optimal", "optimal_uncertified"):
            solved = _task_real_input("saved solved_value", entry["solved_value"])
            if status == "optimal":
                return _task_real_input("saved value", entry["value"]) == solved
            return entry["value"] is None
    except ValueError:
        return False
    return entry["value"] is None and entry["solved_value"] is None


_NON_GLOBAL_MOMENT_SCOPES = frozenset({
    "finite_true_block_grid", "interpolated_pmp_polynomial",
})


def _has_saved_moment_certificate(entry) -> bool:
    """Require an explicit certificate verdict and valid channel counts on resume."""
    if not isinstance(entry, dict) or "error" in entry:
        return False
    certified, finite_grid = entry.get("certified"), entry.get("finite_grid_passed")
    n_pass, n_total = entry.get("n_pass"), entry.get("n_total")
    if (type(certified) is not bool or type(finite_grid) is not bool
            or type(n_pass) is not int or type(n_total) is not int
            or not 0 <= n_pass <= n_total):
        return False
    scopes = entry.get("certificate_scope", [])
    if not isinstance(scopes, list) or any(not isinstance(scope, str) for scope in scopes):
        return False
    if certified and any(scope in _NON_GLOBAL_MOMENT_SCOPES for scope in scopes):
        return False
    return not certified or (finite_grid and n_total > 0 and n_pass == n_total)


def _optimal_dual_objective(objectives, *, where: str):
    """An optimal status is usable only with a finite objective, at its stored precision."""
    try:
        value = objectives["dual"]
        value = value if isinstance(value, mp.mpf) else mp.mpf(value)
        if not mp.isfinite(value):
            raise ValueError("nonfinite objective")
        return value
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        raise ValueError(f"optimal SDPB solve {where} has no finite dual objective") from exc


@task("isolated_relevant_scalar_grid_point", sdpb_defaults="DEFAULT_GRID_FEASIBILITY_SDPB_OPTIONS")
def isolated_relevant_scalar_grid_point(ctx: TaskContext) -> dict:
    """Allowed/excluded test at one ``(c, Delta_phi)`` grid point.

    This is the coarse-region task for the first Z + four-point experiment:
    both sectors share the light data (one isolated scalar ``phi`` and no other
    relevant primary below ``delta_gap``), while their heavy measures remain
    independent. The relaxed coupled-light verdict is therefore the conjunction
    of the modular and four-point verdicts at the same point.

    params:
      c, delta_phi          grid coordinates (strings recommended)
      delta_gap             next scalar/primary gap (default "2")
      lam                   Lambda for both sectors (default 11)
      modular_scalar_gap_only  bool: impose delta_gap only in spin 0 for Z
      modular_phi_mode      "fixed" (d_phi fixed) or "allowed" (d_phi >= 0)
      chiral_currents       bool: irreducible chiral-current characters at Delta = j in Z (default true)
      run_four_point        bool: also solve the four-point half (default true)
      modular_spins         all integer spins for Z, or modular_spin_max
      four_point_spins      even spins for phi^4 (default [0,2,4,6,8,10])
      backend               four-point backend, "sdp" by default
      sdpb_options          SDPB flags; defaults to feasibility jump options
    """
    from ..identical_scalar import run_identical_scalar_gap_problem
    from ..isolated_relevant_scalar import (
        IsolatedRelevantScalarSpec,
        four_point_isolated_spec,
        modular_isolated_spec,
    )
    from ..modular_crossing import (
        build_modular_gap_problem,
        run_modular_gap_problem,
    )
    from ..identical_scalar import build_identical_scalar_gap_problem
    from ..large_spin import double_twist_set

    p = ctx.params
    lam = _task_integer("lam", p.get("lam", 11))
    preset = ctx.profile(lam)
    modular_dps = _task_integer("modular_dps", p.get("modular_dps", p.get("dps", preset["dps"])))
    four_point_dps = _task_integer("four_point_dps", p.get("four_point_dps", p.get("dps", preset["dps"])))
    sdpb_options = _merged_sdpb_options(DEFAULT_GRID_FEASIBILITY_SDPB_OPTIONS, p.get("sdpb_options"), execution=ctx.execution)
    max_runtime = _slurm_max_runtime_seconds()
    if max_runtime is not None and "--maxRuntime" not in sdpb_options:
        sdpb_options["--maxRuntime"] = max_runtime

    modular_spins_param = p.get("modular_spins")
    if modular_spins_param is None:
        modular_spin_max = _task_integer("modular_spin_max", p.get("modular_spin_max", 20))
        modular_spins = tuple(range(modular_spin_max + 1))
    else:
        modular_spins = _task_modular_spins(modular_spins_param)

    four_point_spins = _task_spins(p.get("four_point_spins", (0, 2, 4, 6, 8, 10)))
    run_four_point = _task_boolean("run_four_point", p.get("run_four_point", True))
    four_point_large_spin = _task_boolean("four_point_large_spin", p.get("four_point_large_spin", True))

    with mp.workdps(max(modular_dps, four_point_dps)):
        c = _central_charge(p["c"])
        delta_phi = _task_real_input("delta_phi", p["delta_phi"])
        delta_gap = _task_real_input("delta_gap", p.get("delta_gap", "2"))
        spec = IsolatedRelevantScalarSpec(
            c=c,
            delta_phi=delta_phi,
            delta_gap=delta_gap,
            degeneracy_phi=_task_real_input("degeneracy_phi", p.get("degeneracy_phi", "1")),
            modular_lam=lam,
            four_point_lam=lam,
            modular_spin_list=modular_spins,
            four_point_spin_list=four_point_spins,
            modular_dps=modular_dps,
            four_point_dps=four_point_dps,
            modular_scalar_gap_only=_task_boolean("modular_scalar_gap_only", p.get("modular_scalar_gap_only", False)),
            modular_phi_mode=str(p.get("modular_phi_mode", "fixed")),
            modular_chiral_currents=_task_boolean("chiral_currents", p.get("chiral_currents", True)),
            q_order=_task_integer("q_order", p.get("q_order", preset["q_order"])),
            npoles=_task_integer("npoles", p.get("npoles", preset["npoles"])),
            chiral_norm=p.get("chiral_norm"),
        )

        results = {
            key: entry for key, entry in progress_mapping(ctx.journal.data, "results").items()
            if "status" in entry and "allowed" in entry
            and (entry["status"] is None or isinstance(entry["status"], str))
            and entry["allowed"] is _allowed_from_status(entry["status"])
        }

        if "modular" not in results:
            if ctx.should_stop():
                raise TaskInterrupted("stop requested before modular solve")
            modular_problem = build_modular_gap_problem(modular_isolated_spec(spec))
            modular_run = run_modular_gap_problem(
                modular_problem,
                ctx.output_dir / "modular",
                sdpb_options=sdpb_options,
            )
            modular_status = _checked_sdpb_status(modular_run, "the modular solve")
            results["modular"] = {
                "status": modular_status,
                "allowed": _allowed_from_status(modular_status),
                "log": str(modular_run.sdpb_log_path),
            }
            ctx.journal.update(results=results)

        if run_four_point and "four_point" not in results:
            if ctx.should_stop():
                raise TaskInterrupted("stop requested before four-point solve")
            fp_spec = replace(four_point_isolated_spec(spec), pole_selection=ctx.selector("pole_selection"))
            if four_point_large_spin:
                fp_spec = replace(
                    fp_spec,
                    large_spin_lmin=max(four_point_spins) + 2,
                    large_spin_twists=tuple(double_twist_set(delta_phi / 2, 5)),
                    large_spin_pole_selection=ctx.selector("large_spin_pole_selection"),
                    large_spin_npoles=preset["large_spin_npoles"],
                )
            four_point_problem = build_identical_scalar_gap_problem(
                fp_spec
            )
            four_point_run = run_identical_scalar_gap_problem(
                four_point_problem,
                ctx.output_dir / "four_point",
                backend=str(p.get("backend", "sdp")),
                sdpb_options=sdpb_options,
                pmp2sdp_log_name="pmp2sdp.log",
                sdpb_log_name="sdpb.log",
            )
            four_point_status = _checked_sdpb_status(four_point_run, "the four-point solve")
            results["four_point"] = {
                "status": four_point_status,
                "allowed": _allowed_from_status(four_point_status),
                "log": str(four_point_run.sdpb_log_path),
            }
            ctx.journal.update(results=results)

        modular_allowed = results["modular"]["allowed"]
        four_point_allowed = (
            results["four_point"]["allowed"]
            if "four_point" in results
            else None
        )
        relaxed_allowed = (
            modular_allowed and four_point_allowed
            if modular_allowed is not None and four_point_allowed is not None
            else None
        )
        return {
            "c": _result_number(c),
            "delta_phi": _result_number(delta_phi),
            "delta_gap": _result_number(delta_gap),
            "lam": lam,
            "modular_scalar_gap_only": spec.modular_scalar_gap_only,
            "modular_phi_mode": spec.modular_phi_mode,
            "modular_allowed": modular_allowed,
            "four_point_allowed": four_point_allowed,
            "relaxed_coupled_light_allowed": relaxed_allowed,
            "modular_status": results["modular"]["status"],
            "four_point_status": (
                results["four_point"]["status"]
                if "four_point" in results
                else "not_run"
            ),
        }


@task("isolated_scalar_four_point_ope_bounds", sdpb_defaults="DEFAULT_OPE_MAX_SDPB_OPTIONS")
def isolated_scalar_four_point_ope_bounds(ctx: TaskContext) -> dict:
    """Upper/lower four-point bounds on an isolated operator's p = N^Delta lambda^2 (`ope_squared_bound`).

    The identical external scalar has dimension ``delta_ext``. A distinguished
    exchanged operator of spin ``spin_star`` at ``delta_star`` is separated from
    the positivity cone, while the rest of the scalar sector starts at
    ``delta_gap`` and non-scalar even spins start at their configured lower
    edges.

    ``ope_squared_bound`` is the block coefficient ``p = N^Delta lambda^2`` of
    the distinguished operator, in the same normalization at every spin (the
    spin-0 block is ``B(h, h)``).  Before 0.1.0 a spinning operator's value
    (``spin_star > 0``) was twice this; scalar values are unchanged.
    """
    from ..identical_scalar import (
        IdenticalScalarGapSpec,
        build_identical_scalar_ope_bound_problem,
        ope_squared_bound_from_objective,
        ope_squared_lower_bound_from_objective,
        prepare_identical_scalar_gap_build,
        read_sdpb_objective,
        run_identical_scalar_gap_problem,
    )
    from ..large_spin import double_twist_set

    p = ctx.params
    lam = _task_integer("lam", p.get("lam", 11))
    preset = ctx.profile(lam)
    dps = _task_integer("dps", p.get("dps", preset["dps"]))
    backend = str(p.get("backend", "sdp"))
    directions = _task_choice_list("directions", p.get("directions", ("lower", "upper")), ("lower", "upper"))
    sdpb_options = _merged_sdpb_options(DEFAULT_OPE_MAX_SDPB_OPTIONS, p.get("sdpb_options"), execution=ctx.execution)
    max_runtime = _slurm_max_runtime_seconds()
    if max_runtime is not None and "--maxRuntime" not in sdpb_options:
        sdpb_options["--maxRuntime"] = max_runtime

    with mp.workdps(dps):
        c = _central_charge(p["c"])
        delta_ext = _task_real_input("delta_ext", p["delta_ext"])
        delta_star = _task_real_input("delta_star", p.get("delta_star", delta_ext))
        delta_gap = _task_real_input("delta_gap", p.get("delta_gap", "2"))
        h_ext = delta_ext / 2
        spin_star = _task_exchange_spin(p.get("spin_star", 0))
        spins = _task_spins(p.get("four_point_spins", (0, 2, 4, 6, 8, 10)))

        tower_kwargs: dict = {}
        if _task_boolean("four_point_large_spin", p.get("four_point_large_spin", True)):
            tower_kwargs = dict(
                large_spin_lmin=max(spins) + 2,
                large_spin_twists=tuple(
                    double_twist_set(
                        h_ext,
                        _task_integer("large_spin_double_twists", p.get("large_spin_double_twists", 5)),
                    )
                ),
                large_spin_pole_selection=ctx.selector("large_spin_pole_selection"),
                large_spin_npoles=preset["large_spin_npoles"],
            )

        spec = IdenticalScalarGapSpec(
            c=c,
            h_ext=h_ext,
            delta_gap=delta_gap,
            lam=lam,
            spin_list=spins,
            q_order=_task_integer("q_order", p.get("q_order", preset["q_order"])),
            dps=dps,
            npoles=_task_integer("npoles", p.get("npoles", preset["npoles"])),
            pole_selection=str(p.get("pole_selection", ctx.selector("pole_selection"))),
            chiral_norm=p.get("chiral_norm"),
            num_threads=_task_integer("num_threads", p.get("num_threads", 4)),
            **tower_kwargs,
        )

        solved: dict = progress_mapping(ctx.journal.data, "solved")
        result = {
            "c": _result_number(c),
            "delta_ext": _result_number(delta_ext),
            "delta_star": _result_number(delta_star),
            "delta_gap": _result_number(delta_gap),
            "lam": lam,
            "four_point_spins": spins,
            "backend": backend,
        }

        prepared = prepare_identical_scalar_gap_build(spec)

        for direction in directions:
            if direction not in {"lower", "upper"}:
                raise ValueError("directions must contain only 'lower'/'upper'")
            previous = solved.get(direction)
            if isinstance(previous, dict) and (
                previous.get("status") != "optimal" or _has_finite_ope_bound(previous)
            ):
                result[direction] = solved[direction]
                continue
            if ctx.should_stop():
                raise TaskInterrupted(f"stop requested before {direction} OPE bound")

            problem = build_identical_scalar_ope_bound_problem(
                spec,
                delta_star=delta_star,
                spin_star=spin_star,
                direction=direction,
                prepared=prepared,
            )
            run = run_identical_scalar_gap_problem(
                problem,
                ctx.output_dir / direction,
                backend=backend,
                sdpb_options=sdpb_options,
                pmp2sdp_log_name="pmp2sdp.log",
                sdpb_log_name="sdpb.log",
            )
            obj = read_sdpb_objective(run.sdpb_out_dir) if run.sdpb_out_dir else None
            value = None
            status = _checked_sdpb_status(run, f"the {direction} OPE bound")
            if status == "optimal":
                dual = _optimal_dual_objective(obj, where=f"for the {direction} OPE bound")
                if direction == "upper":
                    value = ope_squared_bound_from_objective(dual)
                else:
                    value = ope_squared_lower_bound_from_objective(dual)
            solved[direction] = {
                "status": status,
                "dual_objective": _result_number(obj["dual"]) if obj and "dual" in obj else None,
                "primal_objective": _result_number(obj["primal"]) if obj and "primal" in obj else None,
                "ope_squared_bound": _result_number(value) if value is not None else None,
                "log": str(run.sdpb_log_path),
            }
            ctx.journal.update(solved=solved)
            result[direction] = solved[direction]

        return result


@task("isolated_scalar_four_point_fixed_ope", sdpb_defaults="DEFAULT_GRID_FEASIBILITY_SDPB_OPTIONS")
def isolated_scalar_four_point_fixed_ope(ctx: TaskContext) -> dict:
    """Feasibility test with an isolated operator's p = N^Delta lambda^2 fixed to `ope_squared`.

    ``ope_squared`` is the block coefficient ``p = N^Delta lambda^2`` of the
    distinguished operator (spin ``spin_star`` at ``delta_star``), in the same
    normalization at every spin (the spin-0 block is ``B(h, h)``).  Before
    0.1.0 a spinning operator's value (``spin_star > 0``) was twice this, so an
    old sweep with ``spin_star > 0`` must halve its ``ope_squared`` to pose
    the same problem; scalar values are unchanged.
    """
    from ..identical_scalar import (
        IdenticalScalarGapSpec,
        build_identical_scalar_fixed_ope_problem,
        prepare_identical_scalar_gap_build,
        run_identical_scalar_gap_problem,
    )
    from ..large_spin import double_twist_set

    p = ctx.params
    lam = _task_integer("lam", p.get("lam", 11))
    preset = ctx.profile(lam)
    dps = _task_integer("dps", p.get("dps", preset["dps"]))
    backend = str(p.get("backend", "sdp"))
    sdpb_options = _merged_sdpb_options(DEFAULT_GRID_FEASIBILITY_SDPB_OPTIONS, p.get("sdpb_options"), execution=ctx.execution)
    max_runtime = _slurm_max_runtime_seconds()
    if max_runtime is not None and "--maxRuntime" not in sdpb_options:
        sdpb_options["--maxRuntime"] = max_runtime

    with mp.workdps(dps):
        c = _central_charge(p["c"])
        delta_ext = _task_real_input("delta_ext", p["delta_ext"])
        delta_star = _task_real_input("delta_star", p.get("delta_star", delta_ext))
        delta_gap = _task_real_input("delta_gap", p.get("delta_gap", "2"))
        ope_squared = _task_real_input("ope_squared", p["ope_squared"])
        h_ext = delta_ext / 2
        spin_star = _task_exchange_spin(p.get("spin_star", 0))
        spins = _task_spins(p.get("four_point_spins", (0, 2, 4, 6, 8, 10)))

        tower_kwargs: dict = {}
        if _task_boolean("four_point_large_spin", p.get("four_point_large_spin", True)):
            tower_kwargs = dict(
                large_spin_lmin=max(spins) + 2,
                large_spin_twists=tuple(
                    double_twist_set(
                        h_ext,
                        _task_integer("large_spin_double_twists", p.get("large_spin_double_twists", 5)),
                    )
                ),
                large_spin_pole_selection=ctx.selector("large_spin_pole_selection"),
                large_spin_npoles=preset["large_spin_npoles"],
            )

        spec = IdenticalScalarGapSpec(
            c=c,
            h_ext=h_ext,
            delta_gap=delta_gap,
            lam=lam,
            spin_list=spins,
            q_order=_task_integer("q_order", p.get("q_order", preset["q_order"])),
            dps=dps,
            npoles=_task_integer("npoles", p.get("npoles", preset["npoles"])),
            pole_selection=str(p.get("pole_selection", ctx.selector("pole_selection"))),
            chiral_norm=p.get("chiral_norm"),
            num_threads=_task_integer("num_threads", p.get("num_threads", 4)),
            **tower_kwargs,
        )
        prepared = prepare_identical_scalar_gap_build(spec)
        problem = build_identical_scalar_fixed_ope_problem(
            spec,
            delta_star=delta_star,
            ope_squared=ope_squared,
            spin_star=spin_star,
            prepared=prepared,
        )
        run = run_identical_scalar_gap_problem(
            problem,
            ctx.output_dir / "fixed_ope",
            backend=backend,
            sdpb_options=sdpb_options,
            pmp2sdp_log_name="pmp2sdp.log",
            sdpb_log_name="sdpb.log",
        )
        status = _checked_sdpb_status(run, "the fixed-OPE solve")
        allowed = _allowed_from_status(status)
        return {
            "c": _result_number(c),
            "delta_ext": _result_number(delta_ext),
            "delta_star": _result_number(delta_star),
            "delta_gap": _result_number(delta_gap),
            "ope_squared": _result_number(ope_squared),
            "lam": lam,
            "four_point_spins": spins,
            "backend": backend,
            "status": status,
            "allowed": allowed,
            "log": str(run.sdpb_log_path),
        }


@task("modular_scalar_gap_bisection", sdpb_defaults="DEFAULT_GRID_FEASIBILITY_SDPB_OPTIONS")
def modular_scalar_gap_bisection(ctx: TaskContext) -> dict:
    """Modular bootstrap bound on the scalar gap at one central charge.

    This is the scalar-gap analogue of the Phase-1 modular gap scan: only the
    spin-0 sector is gapped, while spinning primaries are allowed from their
    unitarity edges. It is the setup to compare against Fig. 8 of
    Collier-Lin-Yin (1608.06241).

    params:
      c                 central charge (string)
      gap_lo/gap_hi     bisection bracket (defaults 0.05 and c/6+2)
      tol               bracket width (default 0.05)
      lam               derivative order (default 11)
      modular_spin_max  include integer spins 0..N (default 20)
      chiral_currents   impose the irreducible chiral-current characters at Delta = j (default true;
                        false assumes no currents besides the stress tensor)
      sdpb_options      SDPB flags
    """
    from ..modular_crossing import (
        ModularGapSpec,
        build_modular_gap_problem,
        prepare_modular_gap_build,
        run_modular_gap_problem,
    )

    p = ctx.params
    lam = _task_integer("lam", p.get("lam", 11))
    preset = ctx.profile(lam)
    dps = _task_integer("dps", p.get("dps", preset["dps"]), minimum=1)
    with mp.workdps(dps):
        c = _central_charge(p["c"])
        gap_lo, gap_hi = _validate_gap_bracket(p.get("gap_lo", "0.05"), p.get("gap_hi", c / 6 + 2))
        tol = _validate_bisection_tolerance(p.get("tol", "0.05"))
        spin_max = _task_integer("modular_spin_max", p.get("modular_spin_max", 20))
        spins = tuple(range(spin_max + 1))
        sdpb_options = _merged_sdpb_options(DEFAULT_GRID_FEASIBILITY_SDPB_OPTIONS, p.get("sdpb_options"), execution=ctx.execution)

        max_runtime = _slurm_max_runtime_seconds()
        if max_runtime is not None and "--maxRuntime" not in sdpb_options:
            sdpb_options["--maxRuntime"] = max_runtime

        base_spec_kwargs = dict(c=c, delta_gap=0, lam=lam, spin_list=spins, dps=dps,
                                chiral_currents=_task_boolean("chiral_currents", p.get("chiral_currents", True)))
        prepared = prepare_modular_gap_build(ModularGapSpec(**base_spec_kwargs))

        probes, cache = _resume_probes(ctx.journal)

        def make_spec(gap: mp.mpf) -> ModularGapSpec:
            return ModularGapSpec(
                **base_spec_kwargs,
                delta_min_by_spin={0: gap},
            )

        def is_allowed(gap: mp.mpf) -> bool:
            key = _label(gap)
            if key not in cache:
                if ctx.should_stop():
                    raise TaskInterrupted(
                        f"stop requested before scalar gap={mp.nstr(mp.mpf(gap), 6)}"
                    )
                problem = build_modular_gap_problem(make_spec(gap), prepared=prepared)
                run = run_modular_gap_problem(
                    problem,
                    ctx.output_dir / f"gap_{key}",
                    sdpb_options=sdpb_options,
                )
                status = _checked_sdpb_status(run, f"scalar gap={mp.nstr(mp.mpf(gap), 6)}")
                allowed = _allowed_from_status(status)
                probes[key] = {
                    "gap": _result_number(gap),
                    "status": status,
                    "allowed": allowed,
                    "log": str(run.sdpb_log_path),
                }
                ctx.journal.update(probes=probes)
                if allowed is None:
                    raise RuntimeError(
                        f"unexpected SDPB status {status!r} "
                        f"at scalar gap {mp.nstr(mp.mpf(gap), 8)}"
                    )
                cache[key] = allowed
            return cache[key]

        if not is_allowed(gap_lo):
            return {
                "c": _result_number(c),
                "lam": lam,
                "reason": "lower_bracket_excluded",
                "gap_lo": _result_number(gap_lo),
                "gap_hi": _result_number(gap_hi),
                "n_probes": len(probes),
            }

        expansions = 0
        while is_allowed(gap_hi):
            gap_lo = gap_hi
            gap_hi = gap_hi * 3 / 2
            expansions += 1
            if gap_hi > max(mp.mpf(20), c * 10):
                return {
                    "c": _result_number(c),
                    "lam": lam,
                    "reason": "no_excluded_upper_bracket",
                    "gap_lo": _result_number(gap_lo),
                    "gap_hi": _result_number(gap_hi),
                    "n_probes": len(probes),
                }

        while gap_hi - gap_lo > tol:
            mid = _bisection_midpoint(gap_lo, gap_hi)
            if is_allowed(mid):
                gap_lo = mid
            else:
                gap_hi = mid

        return {
            "c": _result_number(c),
            "lam": lam,
            "scalar_gap_lo": _result_number(gap_lo),
            "scalar_gap_hi": _result_number(gap_hi),
            "scalar_gap_mid": _result_number((gap_lo + gap_hi) / 2),
            "tol": _result_number(tol),
            "modular_spin_max": spin_max,
            "n_probes": len(probes),
            "bracket_expansions": expansions,
        }


@task("isolated_relevant_scalar_delta_phi_intervals", sdpb_defaults="DEFAULT_GRID_FEASIBILITY_SDPB_OPTIONS")
def isolated_relevant_scalar_delta_phi_intervals(ctx: TaskContext) -> dict:
    """Refine the modular allowed intervals in ``Delta_phi`` at fixed ``c``.

    The task first probes a coarse list of ``delta_phi_grid`` values. Whenever
    adjacent coarse points have opposite verdicts, it bisects that transition
    to ``delta_phi_tol``. This deliberately does not assume monotonicity in
    ``Delta_phi``; a fixed-c slice may contain several allowed intervals.

    params:
      c                  fixed central charge
      delta_phi_grid     ordered coarse probe list
      delta_phi_tol      transition refinement tolerance
      delta_gap          scalar gap above the isolated point (default "2")
      lam                derivative order (default 11)
      modular_spin_max   include integer spins 0..N (default 20)
      chiral_currents    impose the irreducible chiral-current characters at Delta = j (default true;
                         false assumes no currents besides the stress tensor)
      sdpb_options       SDPB flags
    """
    from ..modular_crossing import (
        ModularGapSpec,
        build_modular_gap_problem,
        prepare_modular_gap_build,
        run_modular_gap_problem,
    )

    p = ctx.params
    lam = _task_integer("lam", p.get("lam", 11))
    preset = ctx.profile(lam)
    dps = _task_integer("modular_dps", p.get("modular_dps", p.get("dps", preset["dps"])))
    sdpb_options = _merged_sdpb_options(DEFAULT_GRID_FEASIBILITY_SDPB_OPTIONS, p.get("sdpb_options"), execution=ctx.execution)
    max_runtime = _slurm_max_runtime_seconds()
    if max_runtime is not None and "--maxRuntime" not in sdpb_options:
        sdpb_options["--maxRuntime"] = max_runtime

    with mp.workdps(dps):
        c = _central_charge(p["c"])
        delta_gap = _task_real_input("delta_gap", p.get("delta_gap", "2"))
        tol = _validate_bisection_tolerance(p.get("delta_phi_tol", "0.025"), "delta_phi_tol")
        grid = _task_real_list("delta_phi_grid", p["delta_phi_grid"], positive=True, ordered=True)

        spin_max = _task_integer("modular_spin_max", p.get("modular_spin_max", 20))
        spins = tuple(range(spin_max + 1))
        base_spec_kwargs = dict(
            c=c,
            delta_gap=0,
            lam=lam,
            spin_list=spins,
            dps=dps,
            delta_min_by_spin={0: delta_gap},
            chiral_currents=_task_boolean("chiral_currents", p.get("chiral_currents", True)),
        )
        prepared = prepare_modular_gap_build(ModularGapSpec(**base_spec_kwargs))

        probes, cache = _resume_probes(ctx.journal)

        def make_spec(delta_phi: mp.mpf) -> ModularGapSpec:
            return ModularGapSpec(
                **base_spec_kwargs,
                allowed_isolated_scalars=(delta_phi,),
            )

        def is_allowed(delta_phi: mp.mpf) -> bool:
            key = _label(delta_phi)
            if key not in cache:
                if ctx.should_stop():
                    raise TaskInterrupted(
                        f"stop requested before Delta_phi={mp.nstr(delta_phi, 8)}"
                    )
                problem = build_modular_gap_problem(
                    make_spec(delta_phi),
                    prepared=prepared,
                )
                run = run_modular_gap_problem(
                    problem,
                    ctx.output_dir / f"delta_phi_{key}",
                    sdpb_options=sdpb_options,
                )
                status = _checked_sdpb_status(run, f"Delta_phi={mp.nstr(delta_phi, 8)}")
                allowed = _allowed_from_status(status)
                probes[key] = {
                    "delta_phi": _result_number(delta_phi),
                    "status": status,
                    "allowed": allowed,
                    "log": str(run.sdpb_log_path),
                }
                ctx.journal.update(probes=probes)
                if allowed is None:
                    raise RuntimeError(
                        f"unexpected SDPB status {status!r} "
                        f"at Delta_phi {mp.nstr(delta_phi, 8)}"
                    )
                cache[key] = allowed
            return cache[key]

        coarse = []
        for delta_phi in grid:
            coarse.append((delta_phi, is_allowed(delta_phi)))

        transitions = []
        for (left, left_allowed), (right, right_allowed) in zip(
            coarse, coarse[1:]
        ):
            if left_allowed == right_allowed:
                continue
            lo = left
            hi = right
            lo_allowed = left_allowed
            hi_allowed = right_allowed
            while hi - lo > tol:
                mid = _bisection_midpoint(lo, hi)
                mid_allowed = is_allowed(mid)
                if mid_allowed == lo_allowed:
                    lo = mid
                    lo_allowed = mid_allowed
                else:
                    hi = mid
                    hi_allowed = mid_allowed
            transitions.append(
                {
                    "left": _result_number(lo),
                    "right": _result_number(hi),
                    "mid": _result_number((lo + hi) / 2),
                    "left_allowed": lo_allowed,
                    "right_allowed": hi_allowed,
                }
            )

        boundaries = [
            (mp.mpf(item["mid"]), item["left_allowed"], item["right_allowed"])
            for item in transitions
        ]
        interval_edges = [grid[0]]
        interval_edges.extend(boundary for boundary, _, _ in boundaries)
        interval_edges.append(grid[-1])
        intervals = []
        for left, right in zip(interval_edges, interval_edges[1:]):
            sample = (left + right) / 2
            allowed = is_allowed(sample)
            if allowed:
                intervals.append(
                    {
                        "left": _result_number(left),
                        "right": _result_number(right),
                        "sample": _result_number(sample),
                    }
                )

        return {
            "c": _result_number(c),
            "delta_gap": _result_number(delta_gap),
            "lam": lam,
            "modular_phi_mode": "allowed",
            "modular_scalar_gap_only": True,
            "modular_spin_max": spin_max,
            "delta_phi_tol": _result_number(tol),
            "coarse": [
                {
                    "delta_phi": _result_number(delta_phi),
                    "allowed": allowed,
                }
                for delta_phi, allowed in coarse
            ],
            "transitions": transitions,
            "allowed_intervals": intervals,
            "n_probes": len(probes),
        }


@task("isolated_relevant_scalar_degeneracy_intervals", sdpb_defaults="DEFAULT_GRID_FEASIBILITY_SDPB_OPTIONS")
def isolated_relevant_scalar_degeneracy_intervals(ctx: TaskContext) -> dict:
    """Refine modular allowed intervals in fixed isolated-scalar degeneracy.

    This tests whether modular crossing is feasible with a scalar contribution
    ``d_phi * chi_Delta_phi`` moved to the known side, while the rest of the
    scalar sector is gapped above ``delta_gap`` and spinning sectors start at
    unitarity. As with the Delta_phi interval task, the scan refines every
    transition it sees and does not assume a single interval.

    params:
      c                  central charge
      delta_phi          isolated scalar dimension
      degeneracy_grid    ordered coarse d_phi probe list
      degeneracy_tol     transition refinement tolerance
      delta_gap          scalar gap above the isolated point (default "2")
      lam                derivative order (default 11)
      modular_spin_max   include integer spins 0..N (default 20)
      chiral_currents    impose the irreducible chiral-current characters at Delta = j (default true;
                         false assumes no currents besides the stress tensor)
      sdpb_options       SDPB flags
    """
    from ..modular_crossing import (
        ModularGapSpec,
        build_modular_gap_problem,
        prepare_modular_gap_build,
        run_modular_gap_problem,
    )

    p = ctx.params
    lam = _task_integer("lam", p.get("lam", 11))
    preset = ctx.profile(lam)
    dps = _task_integer("modular_dps", p.get("modular_dps", p.get("dps", preset["dps"])))
    sdpb_options = _merged_sdpb_options(DEFAULT_GRID_FEASIBILITY_SDPB_OPTIONS, p.get("sdpb_options"), execution=ctx.execution)
    max_runtime = _slurm_max_runtime_seconds()
    if max_runtime is not None and "--maxRuntime" not in sdpb_options:
        sdpb_options["--maxRuntime"] = max_runtime

    with mp.workdps(dps):
        c = _central_charge(p["c"])
        delta_phi = _task_real_input("delta_phi", p["delta_phi"])
        delta_gap = _task_real_input("delta_gap", p.get("delta_gap", "2"))
        tol = _validate_bisection_tolerance(p.get("degeneracy_tol", "0.125"), "degeneracy_tol")
        grid = _task_real_list("degeneracy_grid", p["degeneracy_grid"], ordered=True)

        spin_max = _task_integer("modular_spin_max", p.get("modular_spin_max", 20))
        spins = tuple(range(spin_max + 1))
        base_spec_kwargs = dict(
            c=c,
            delta_gap=0,
            lam=lam,
            spin_list=spins,
            dps=dps,
            delta_min_by_spin={0: delta_gap},
            chiral_currents=_task_boolean("chiral_currents", p.get("chiral_currents", True)),
        )
        prepared = prepare_modular_gap_build(ModularGapSpec(**base_spec_kwargs))

        probes, cache = _resume_probes(ctx.journal)

        def make_spec(degeneracy: mp.mpf) -> ModularGapSpec:
            return ModularGapSpec(
                **base_spec_kwargs,
                isolated_scalars=((delta_phi, degeneracy),),
            )

        def is_allowed(degeneracy: mp.mpf) -> bool:
            key = _label(degeneracy)
            if key not in cache:
                if ctx.should_stop():
                    raise TaskInterrupted(
                        f"stop requested before d_phi={mp.nstr(degeneracy, 8)}"
                    )
                problem = build_modular_gap_problem(
                    make_spec(degeneracy),
                    prepared=prepared,
                )
                run = run_modular_gap_problem(
                    problem,
                    ctx.output_dir / f"degeneracy_{key}",
                    sdpb_options=sdpb_options,
                )
                status = _checked_sdpb_status(run, f"d_phi={mp.nstr(degeneracy, 8)}")
                allowed = _allowed_from_status(status)
                probes[key] = {
                    "degeneracy": _result_number(degeneracy),
                    "status": status,
                    "allowed": allowed,
                    "log": str(run.sdpb_log_path),
                }
                ctx.journal.update(probes=probes)
                if allowed is None:
                    raise RuntimeError(
                        f"unexpected SDPB status {status!r} "
                        f"at d_phi {mp.nstr(degeneracy, 8)}"
                    )
                cache[key] = allowed
            return cache[key]

        coarse = []
        for degeneracy in grid:
            coarse.append((degeneracy, is_allowed(degeneracy)))

        transitions = []
        for (left, left_allowed), (right, right_allowed) in zip(
            coarse, coarse[1:]
        ):
            if left_allowed == right_allowed:
                continue
            lo = left
            hi = right
            lo_allowed = left_allowed
            hi_allowed = right_allowed
            while hi - lo > tol:
                mid = _bisection_midpoint(lo, hi)
                mid_allowed = is_allowed(mid)
                if mid_allowed == lo_allowed:
                    lo = mid
                    lo_allowed = mid_allowed
                else:
                    hi = mid
                    hi_allowed = mid_allowed
            transitions.append(
                {
                    "left": _result_number(lo),
                    "right": _result_number(hi),
                    "mid": _result_number((lo + hi) / 2),
                    "left_allowed": lo_allowed,
                    "right_allowed": hi_allowed,
                }
            )

        boundaries = [
            (mp.mpf(item["mid"]), item["left_allowed"], item["right_allowed"])
            for item in transitions
        ]
        interval_edges = [grid[0]]
        interval_edges.extend(boundary for boundary, _, _ in boundaries)
        interval_edges.append(grid[-1])
        intervals = []
        for left, right in zip(interval_edges, interval_edges[1:]):
            sample = (left + right) / 2
            allowed = is_allowed(sample)
            if allowed:
                intervals.append(
                    {
                        "left": _result_number(left),
                        "right": _result_number(right),
                        "sample": _result_number(sample),
                    }
                )

        return {
            "c": _result_number(c),
            "delta_phi": _result_number(delta_phi),
            "delta_gap": _result_number(delta_gap),
            "lam": lam,
            "modular_phi_mode": "fixed",
            "modular_scalar_gap_only": True,
            "modular_spin_max": spin_max,
            "degeneracy_tol": _result_number(tol),
            "coarse": [
                {
                    "degeneracy": _result_number(degeneracy),
                    "allowed": allowed,
                }
                for degeneracy, allowed in coarse
            ],
            "transitions": transitions,
            "allowed_intervals": intervals,
            "n_probes": len(probes),
        }


@task("scalar_gap_bisection", sdpb_defaults="DEFAULT_FEASIBILITY_SDPB_OPTIONS")
def scalar_gap_bisection(ctx: TaskContext) -> dict:
    """Critical scalar-gap bound Delta*(0) at one central charge, by bisection.

    The cluster-task form of the per-central-charge gap worker: block tables
    are prepared once and reused across probes (except under
    ``twist_gap="match_gap"``, ``all_spins_twist_gap`` or
    ``self_consistent_h_ext``, where the blocks move with the probed gap and
    every probe rebuilds them), and each probe
    optionally hotstarts SDPB from the previous probe's checkpoint, and every
    completed probe verdict is journaled so an interrupted bisection resumes
    from its last probe (the bracket replay from cached verdicts is free).

    params (all optional except ``c``; numbers as strings where noted):
      c            central charge (string)
      gap_lo       lower bracket, expected ALLOWED   (default "0.5")
      gap_hi       upper bracket, expected EXCLUDED  (default "3.0")
      max_hi       cap on auto-expanding gap_hi when it's still ALLOWED
                   (default "64"; lower this -- e.g. "8" -- when
                   self_consistent_h_ext=True so the search doesn't chase
                   past the block-table's trusted Delta_ext window)
      tol          bisection stop width               (default "0.05")
      h_ext        external weight h                  (default "0.0625")
      spins        even spin list                     (default [0,2,4,6,8,10])
      lam          derivative order; keys LAMBDA_PRESETS (default 7)
      npoles       kept Kac poles per block (default: the lam preset's)
      large_spin_pole_selection  tower kept-pole selector (default
                   profile's "contribution_optimal")
      pole_selection  fixed-spin kept-pole selector (default profile's
                   "contribution_optimal")
      q_order      q-series truncation order  (default: the lam preset's)
      num_threads  block-build workers                (default 4)
      sdpb_options mapping of SDPB flags              (default DEFAULT_FEASIBILITY_SDPB_OPTIONS)
      twist_gap    "none", "match_gap" (track the current bisected gap
                   dynamically -- for the uniform TWIST-gap bound, where the
                   tower's lowest twist slice must equal the probed gap at
                   every probe), or a fixed numeric twist string
                                                        (default "none")
      all_spins_twist_gap  bool: coupled per-spin gaps ell+gap (default False)
      self_consistent_h_ext  bool: h_ext = gap/2 at every probe instead of
                   the fixed ``h_ext`` param -- the external operator sits
                   exactly on the gap being bisected. Forces a fresh block-table build every probe
                   (like all_spins_twist_gap) since the blocks depend on
                   h_ext.                                (default False)
      tower        bool: include the large-spin double-twist tower. Set
                   False for a pure fixed-spin (or scalar-only, with
                   spins=[0]) comparison with no asymptotic-spin channel at
                   all (default True)
      large_spin_double_twists  number of double-twist slices in the tower
                   (default: the lam profile's, 16 at Lambda=11)
      large_spin_npoles  kept Kac poles per tower block (default: the lam
                   profile's, falling back to the fixed-spin npoles)
      bound_direction  "upper" (default): gap_lo expected ALLOWED, gap_hi
                   expected EXCLUDED, standard "how large can the gap be"
                   search. "lower": mirror image -- gap_lo expected
                   EXCLUDED, gap_hi expected ALLOWED, for setups (observed
                   so far only in the scalar-only case) where a gap SMALLER
                   than some critical value is also forbidden. Auto-expand
                   and failure-reason semantics mirror accordingly
                   (lo_allowed / hi_excluded_at_cap replace lo_excluded /
                   hi_allowed_at_cap).
      backend      "sdp" | "pmp2sdp"                  (default "sdp")
      hotstart     chain SDPB checkpoints across probes (default True)
      decide_by    "status" (default) reads SDPB's terminateReason:
                   primal_feasible -> ALLOWED, anything else conclusive ->
                   EXCLUDED. "certificate" IGNORES the status and decides on
                   the functional -- a normalized y positive on every channel
                   excludes the gap, absence of one is "allowed".
                   Use "certificate" when the status cannot be trusted, which
                   at c=8/7 it demonstrably cannot: a Delta_ext=0.1 bisection
                   scored gaps 0.64375/0.73438/0.77969 as primal_feasible while
                   their own y certifies VALID on all 29 channels, so it
                   reported 0.819 for a bound below 0.64. Under "certificate" a
                   maxIterations exit is not a failure -- SDPB still writes y.
      cert_extra_spins / cert_extra_twists  coverage beyond the SDP's own
                   channels when decide_by="certificate" (default 0)
    """
    # Imported here so importing the registry stays cheap (flint/mpmath block
    # machinery only loads inside a running task).
    from ..certification import certify_dual_from_sdpb_output
    from ..identical_scalar import (
        IdenticalScalarGapSpec,
        build_identical_scalar_gap_problem,
        prepare_identical_scalar_gap_build,
        run_identical_scalar_gap_problem,
    )
    from ..large_spin import double_twist_set, twists_above_gap

    p = ctx.params
    lam = _task_integer("lam", p.get("lam", 7))
    preset = ctx.profile(lam)
    dps = _task_integer("dps", p.get("dps", preset["dps"]))
    # Block-accuracy knobs, overridable so a verdict can be convergence-tested
    # against block accuracy (the scalar-only false-exclusion mechanism: a dual
    # functional certified >=0 at the sample points can go negative off-sample
    # when the interpolation carries fewer digits than the functional's
    # cancellation depth).
    npoles = _task_integer("npoles", p.get("npoles", preset["npoles"]))
    q_order = _task_integer("q_order", p.get("q_order", preset["q_order"]))
    with mp.workdps(dps):
        c = _central_charge(p["c"])
        gap_lo, gap_hi = _validate_gap_bracket(p.get("gap_lo", "0.5"), p.get("gap_hi", "3.0"))
        tol = _validate_bisection_tolerance(p.get("tol", "0.05"))
        h_ext = _task_real_input("h_ext", p.get("h_ext", "0.0625"))
        spins = _task_spins(p.get("spins", (0, 2, 4, 6, 8, 10)))
        num_threads = _task_integer("num_threads", p.get("num_threads", 4))
        sdpb_options = _merged_sdpb_options(DEFAULT_FEASIBILITY_SDPB_OPTIONS, p.get("sdpb_options"), execution=ctx.execution)
        twist_gap_input = _task_twist_gap(p.get("twist_gap", "none"))
        twist_gap_mode = str(twist_gap_input)
        all_spins_twist_gap = _task_boolean("all_spins_twist_gap", p.get("all_spins_twist_gap", False))
        self_consistent_h_ext = _task_boolean("self_consistent_h_ext", p.get("self_consistent_h_ext", False))
        tower = _task_boolean("tower", p.get("tower", True))
        bound_direction = str(p.get("bound_direction", "upper"))
        if bound_direction not in ("upper", "lower"):
            raise ValueError(f"bound_direction must be 'upper' or 'lower', got {bound_direction!r}")
        backend = str(p.get("backend", "sdp"))
        hotstart = _task_boolean("hotstart", p.get("hotstart", True))
        decide_by = str(p.get("decide_by", "status"))
        if decide_by not in ("status", "certificate"):
            raise ValueError(
                f"decide_by must be 'status' or 'certificate', got {decide_by!r}"
            )
        cert_extra_spins = _task_integer("cert_extra_spins", p.get("cert_extra_spins", 0))
        cert_extra_twists = _task_integer("cert_extra_twists", p.get("cert_extra_twists", 0))
        n_twists = _task_integer("large_spin_double_twists", p.get("large_spin_double_twists", 5 if preset["large_spin_double_twists"] is None else preset["large_spin_double_twists"]))
        large_spin_npoles = _task_integer("large_spin_npoles", p.get("large_spin_npoles", preset["npoles"] if preset["large_spin_npoles"] is None else preset["large_spin_npoles"]))
        # documented params; vb-certify (hpc/recertify.py) rebuilds with them too,
        # so hard-coding the defaults here made it certify a different problem
        pole_selection = str(p.get("pole_selection", ctx.selector("pole_selection")))
        large_spin_pole_selection = str(
            p.get("large_spin_pole_selection", ctx.selector("large_spin_pole_selection"))
        )

        lmax = max(spins) + 2
        logs = [f"c={mp.nstr(c, 5)}:"]

        def make_spec(gap: object) -> IdenticalScalarGapSpec:
            # Midpoints are numerical mpf values; preserve their stored bits
            # before adding spins or deriving the external weight.
            gap = _task_real_input("gap", gap)
            h_ext_local = gap / 2 if self_consistent_h_ext else h_ext

            tower_kwargs: dict = {}
            if tower:
                if twist_gap_mode == "none":
                    twists = tuple(double_twist_set(h_ext_local, n_twists))
                elif twist_gap_mode == "match_gap":
                    twists = tuple(twists_above_gap(double_twist_set(h_ext_local, n_twists), gap))
                else:
                    twists = tuple(
                        twists_above_gap(double_twist_set(h_ext_local, n_twists),
                                        twist_gap_input)
                    )
                tower_kwargs = dict(
                    large_spin_lmin=lmax,
                    large_spin_twists=twists,
                    large_spin_pole_selection=large_spin_pole_selection,
                    large_spin_npoles=large_spin_npoles,
                )

            delta_min_by_spin = None
            if all_spins_twist_gap:
                delta_min_by_spin = {ell: ell + gap for ell in spins if ell != 0}

            return IdenticalScalarGapSpec(
                c=c,
                h_ext=h_ext_local,
                delta_gap=gap,
                delta_min_by_spin=delta_min_by_spin,
                lam=lam,
                spin_list=spins,
                q_order=q_order,
                dps=dps,
                npoles=npoles,
                pole_selection=pole_selection,
                num_threads=num_threads,
                **tower_kwargs,
            )

        # Resume state: verdicts of every completed probe, plus the newest usable
        # hotstart checkpoint. The bracket itself is NOT journaled -- replaying the
        # bisection against the cached verdicts reconstructs it for free.
        probes, cache = _resume_probes(ctx.journal)
        if cache:
            logs.append(f"  resumed with {len(cache)} journaled probe(s)")
        # A checkpoint only seeds a solve with the same SDP block structure:
        # SDPB aborts (``local_height == block.LocalHeight()``) otherwise. Under
        # twist_gap="match_gap" the number of tower slices changes whenever the
        # probed gap crosses a double-twist value, so each checkpoint carries the
        # signature of the SDP that wrote it and is chained only on a match.
        prev_checkpoint: Path | None = None
        prev_signature: str | None = None
        if hotstart:
            last_ck = ctx.journal.data.get("last_checkpoint")
            if last_ck and Path(last_ck).is_dir():
                prev_checkpoint = Path(last_ck)
                prev_signature = ctx.journal.data.get("last_checkpoint_signature")

        # A shared block-table cache across probes is only valid when the spec
        # is unchanged apart from delta_gap itself: self_consistent_h_ext moves
        # h_ext (hence the blocks) with every probe, exactly like
        # all_spins_twist_gap moves delta_min_by_spin.
        prepared = None
        # (twist_gap="match_gap" moves the large-spin twists with every probe, too)
        if not all_spins_twist_gap and not self_consistent_h_ext and twist_gap_mode != "match_gap":
            try:
                prepared = prepare_identical_scalar_gap_build(make_spec(gap_lo))
            except Exception as exc:
                logs.append(f"  -> prepare failed: {exc.__class__.__name__}: {exc}")
                return {
                    "c": _result_number(c),
                    "delta_gap": None,
                    "reason": f"prepare_error:{exc.__class__.__name__}",
                    "logs": logs,
                }

        def is_allowed(gap: mp.mpf) -> bool:
            nonlocal prev_checkpoint, prev_signature
            key = _label(gap)
            if key not in cache:
                if ctx.should_stop():
                    raise TaskInterrupted(f"stop requested before gap={mp.nstr(mp.mpf(gap), 6)}")
                last_status = None
                last_summary = None
                seed_allowed = True
                for attempt in range(3):
                    try:
                        if prepared is None:
                            problem = build_identical_scalar_gap_problem(make_spec(gap))
                        else:
                            problem = build_identical_scalar_gap_problem(
                                make_spec(gap), prepared=prepared
                            )
                        signature = sdp_structure_signature(problem)
                        seed = (
                            prev_checkpoint
                            if hotstart
                            and seed_allowed
                            and prev_checkpoint is not None
                            and prev_signature == signature
                            else None
                        )
                        if hotstart and prev_checkpoint is not None and seed is None and attempt == 0:
                            logs.append(
                                f"  gap={mp.nstr(mp.mpf(gap), 6)}: SDP structure differs from the "
                                "previous checkpoint's; solving cold"
                            )
                        run_options = dict(sdpb_options)
                        if attempt > 0:
                            # Escalate from whatever the sweep configured, not from
                            # a hardcoded 500: a sweep that deliberately raises
                            # --maxIterations (e.g. jump-only termination, where the
                            # dual side needs a long run to jump at all) would
                            # otherwise have its first retry LOWERED below its own
                            # setting.
                            base = int(run_options.get("--maxIterations", 500))
                            run_options["--maxIterations"] = base * (2**attempt)
                        if hotstart:
                            # Keep the final checkpoint so the next probe can chain
                            # from it (drop --noFinalCheckpoint if present).
                            run_options.pop("--noFinalCheckpoint", None)
                        max_runtime = _slurm_max_runtime_seconds()
                        if max_runtime is not None and "--maxRuntime" not in run_options:
                            run_options["--maxRuntime"] = max_runtime
                        run = run_identical_scalar_gap_problem(
                            problem,
                            ctx.output_dir / f"gap_{key}",
                            backend=backend,
                            sdpb_options=run_options,
                            pmp2sdp_log_name="pmp2sdp.log",
                            sdpb_log_name="sdpb.log",
                            initial_checkpoint_dir=seed,
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            f"is_allowed failed at gap={mp.nstr(mp.mpf(gap), 6)}: "
                            f"{exc.__class__.__name__}: {exc}"
                        ) from exc

                    # A --maxRuntime exit raises TaskInterrupted: the worker
                    # requeues and the next allocation resumes this probe.
                    status = _checked_sdpb_status(run, f"gap={mp.nstr(mp.mpf(gap), 6)}")

                    last_status = status
                    last_summary = run.sdpb_summary
                    if status == "solver_error" and seed is not None:
                        # The seed may be what broke the solve; never retry on it.
                        seed_allowed = False

                    if decide_by == "certificate":
                        # Decide from the functional: a normalized y that is
                        # positive on every channel excludes the gap, regardless
                        # of the solver's termination status. This also applies
                        # when the solver reaches maxIterations. Retry if no
                        # output is available to certify.
                        if run.sdpb_out_dir is None or not Path(run.sdpb_out_dir).is_dir():
                            if attempt < 2:
                                logs.append(
                                    f"  gap={mp.nstr(mp.mpf(gap), 6)}  ->  no sdpb_out; retrying"
                                )
                                continue
                            raise RuntimeError(
                                f"no SDPB output to certify at gap={mp.nstr(mp.mpf(gap), 6)}"
                            )
                        certified, details = certify_dual_from_sdpb_output(
                            problem,
                            run.sdpb_out_dir,
                            extra_spins=cert_extra_spins,
                            extra_twists=cert_extra_twists,
                            num_threads=num_threads,
                            warn=False,
                        )
                        undecided = sum(
                            1 for d in details if d.get("exact_positivity_undecided")
                        )
                        # A valid certificate EXCLUDES. Undecided channels are not
                        # evidence of exclusion, so they fall to "allowed" -- the
                        # conservative direction for an upper bound.
                        excluded = bool(certified) and undecided == 0
                        allowed = not excluded
                        n_pos = sum(1 for d in details if d.get("passed") is True)
                        cache[key] = allowed
                        probes[key] = {
                            "allowed": allowed,
                            "status": status,
                            "decided_by": "certificate",
                            "channels_positive": n_pos,
                            "channels_total": len(details),
                            "undecided": undecided,
                        }
                        journal_updates = {"probes": probes}
                        if hotstart and run.checkpoint_dir is not None:
                            prev_checkpoint, prev_signature = run.checkpoint_dir, signature
                            journal_updates["last_checkpoint"] = str(run.checkpoint_dir)
                            journal_updates["last_checkpoint_signature"] = signature
                        ctx.journal.update(**journal_updates)
                        verdict = "ALLOWED" if allowed else "EXCLUDED"
                        logs.append(
                            f"  gap={mp.nstr(mp.mpf(gap), 6)}  ->  {verdict}"
                            f"  [cert {n_pos}/{len(details)} positive"
                            f"{f', {undecided} undecided' if undecided else ''}"
                            f"; sdpb said {status}]"
                        )
                        break

                    if status not in _INCONCLUSIVE_SDPB_STATUSES:
                        allowed = status == "primal_feasible"
                        cache[key] = allowed
                        probes[key] = {"allowed": allowed, "status": status}
                        journal_updates: dict = {"probes": probes}
                        # Chain the checkpoint to the next probe (this task only).
                        if hotstart and run.checkpoint_dir is not None:
                            prev_checkpoint, prev_signature = run.checkpoint_dir, signature
                            journal_updates["last_checkpoint"] = str(run.checkpoint_dir)
                            journal_updates["last_checkpoint_signature"] = signature
                        ctx.journal.update(**journal_updates)
                        verdict = "ALLOWED" if allowed else f"EXCLUDED ({status})"
                        logs.append(f"  gap={mp.nstr(mp.mpf(gap), 6)}  ->  {verdict}")
                        break

                    cause = f": {last_summary}" if last_summary else ""
                    if attempt < 2:
                        logs.append(
                            f"  gap={mp.nstr(mp.mpf(gap), 6)}  ->  INCONCLUSIVE ({status}{cause}); retrying"
                            + (" without the hotstart checkpoint" if seed is not None and not seed_allowed else "")
                        )
                        continue

                    raise RuntimeError(
                        f"inconclusive SDPB status after retries at gap={mp.nstr(mp.mpf(gap), 6)}: "
                        f"{last_status}{cause}"
                        + (f" (see {run.sdpb_log_path})" if run.sdpb_log_path is not None else "")
                    )
            return cache[key]

        max_hi = _task_real_input("max_hi", p.get("max_hi", "64"))
        try:
            if bound_direction == "upper":
                if not is_allowed(gap_lo):
                    logs.append(f"  -> lower bracket {mp.nstr(mp.mpf(gap_lo), 5)} already EXCLUDED; skipping")
                    return {"c": _result_number(c), "delta_gap": None, "reason": "lo_excluded", "logs": logs}
                # Auto-expand the upper bracket so dense scans don't fail on ranges where
                # the fixed initial hi is still allowed. Default 64 is fine when h_ext is
                # fixed (delta_gap never pushes the block-table's trusted Delta_ext
                # window); with self_consistent_h_ext, delta_gap IS Delta_ext, so a huge
                # cap can chase the search into the untrustworthy large-Delta_ext regime
                # (fixed q_order/npoles/dps calibrated for small Delta_ext) and return a
                # spurious "still allowed" verdict; use the conservative
                # MAX_HI=8 default for the block-trust window.
                while is_allowed(gap_hi) and gap_hi < max_hi:
                    gap_hi = min(gap_hi * 2, max_hi)
                    logs.append(f"  -> upper bracket still ALLOWED, expanding hi to {mp.nstr(mp.mpf(gap_hi), 6)}")

                if is_allowed(gap_hi):
                    logs.append(
                        f"  -> upper bracket {mp.nstr(mp.mpf(gap_hi), 5)} still ALLOWED at cap {mp.nstr(mp.mpf(max_hi), 5)}; skipping"
                    )
                    return {"c": _result_number(c), "delta_gap": None, "reason": "hi_allowed_at_cap", "logs": logs}

                lo, hi = gap_lo, gap_hi
                while hi - lo > tol:
                    mid = _bisection_midpoint(lo, hi)
                    if is_allowed(mid):
                        lo = mid
                    else:
                        hi = mid
            else:  # bound_direction == "lower": mirror image search.
                # gap_lo is expected EXCLUDED (a small/zero gap already forbidden) and
                # gap_hi is expected ALLOWED (some larger gap is consistent) -- the
                # LOWER critical gap is the threshold in between. If gap_lo is
                # unexpectedly already ALLOWED, there is no nontrivial lower bound in
                # this bracket.
                if is_allowed(gap_lo):
                    logs.append(f"  -> lower bracket {mp.nstr(mp.mpf(gap_lo), 5)} already ALLOWED; no lower bound here")
                    return {"c": _result_number(c), "delta_gap": None, "reason": "lo_allowed", "logs": logs}
                while not is_allowed(gap_hi) and gap_hi < max_hi:
                    gap_hi = min(gap_hi * 2, max_hi)
                    logs.append(f"  -> upper bracket still EXCLUDED, expanding hi to {mp.nstr(mp.mpf(gap_hi), 6)}")

                if not is_allowed(gap_hi):
                    logs.append(
                        f"  -> upper bracket {mp.nstr(mp.mpf(gap_hi), 5)} still EXCLUDED at cap {mp.nstr(mp.mpf(max_hi), 5)}; skipping"
                    )
                    return {"c": _result_number(c), "delta_gap": None, "reason": "hi_excluded_at_cap", "logs": logs}

                lo, hi = gap_lo, gap_hi
                while hi - lo > tol:
                    mid = _bisection_midpoint(lo, hi)
                    if is_allowed(mid):
                        hi = mid
                    else:
                        lo = mid
        except TaskInterrupted:
            raise
        except Exception as exc:
            logs.append(f"  -> solve failed: {exc}")
            return {
                "c": _result_number(c),
                "delta_gap": None,
                "reason": "solve_error",
                "error": str(exc),
                "logs": logs,
            }

        bound = (lo + hi) / 2
        logs.append(f"  -> Delta*(0) ~= {mp.nstr(mp.mpf(bound), 6)}  in [{mp.nstr(mp.mpf(lo), 8)}, {mp.nstr(mp.mpf(hi), 8)}]")
        return {
            "c": _result_number(c),
            "delta_gap": _result_number(bound),
            "delta_gap_lo": _result_number(lo),
            "delta_gap_hi": _result_number(hi),
            "bound_direction": bound_direction,
            "logs": logs,
        }


@task("scalar_ope_max_scan", sdpb_defaults="DEFAULT_OPE_MAX_SDPB_OPTIONS")
def scalar_ope_max_scan(ctx: TaskContext) -> dict:
    """Locate the scalar gap bound by maximizing an OPE coefficient.

    At each trial gap, maximize the block coefficient of the distinguished
    operator. With normalization V_* and objective V_id, an ``optimal`` result
    gives the upper bound ``-dualObjective`` on p = N^Delta lambda^2. See
    ``build_identical_scalar_ope_max_problem`` and ``examples/ope_max_scalar.py``.

    The task estimates a gap bracket from the largest trial value with a finite
    optimal bound and the smallest with ``max_complementarity``, which it treats
    as a runaway objective. Other statuses are inconclusive. A
    ``max_complementarity`` result supplies no usable OPE bound. If the two
    endpoints are reversed or equal, the task reports ``inconsistent_scan``
    without a bracket.

    Block tables are reused across the scalar-gap scan. With
    ``all_spins_twist_gap``, each trial changes every channel's lower edge, so
    the tables are rebuilt at each point.

    params:
      c            central charge (string, required)
      h_ext        external weight h                  (default "0.0625")
      delta_stars  list of Delta* values to scan (strings). Each is used as both
                   the imposed gap and the distinguished operator's location.
      spin_star    spin of the distinguished operator (default 0). Its reported
                   ``ope_squared_bound`` is p = N^Delta lambda^2, the same
                   normalization at every spin
      all_spins_twist_gap  bool: read each delta_star as a uniform twist gap tau
                   -- every spin gets delta_min = ell + tau and the tower's
                   lowest slice tracks tau -- instead of gapping the scalar
                   channel alone (default False). Note this forces a block
                   rebuild per point, since delta_min moves on every channel.
                   The distinguished operator has Delta = spin_star + tau.
      lam, npoles, q_order, dps, spins, tower knobs -- as scalar_gap_bisection
      sdpb_options mapping of SDPB flags. Run to optimality: do not pass
                   find*Feasible or detect*Jump here.
    """
    from ..identical_scalar import (
        IdenticalScalarGapSpec,
        build_identical_scalar_ope_max_problem,
        ope_squared_bound_from_objective,
        prepare_identical_scalar_gap_build,
        read_sdpb_objective,
        run_identical_scalar_gap_problem,
    )
    from ..large_spin import double_twist_set, twists_above_gap

    p = ctx.params
    lam = _task_integer("lam", p.get("lam", 11))
    preset = ctx.profile(lam)
    dps = _task_integer("dps", p.get("dps", preset["dps"]))
    with mp.workdps(dps):
        c = _central_charge(p["c"])
        h_ext = _task_real_input("h_ext", p.get("h_ext", "0.0625"))
        spin_star = _task_exchange_spin(p.get("spin_star", 0))
        spins = _task_spins(p.get("spins", (0, 2, 4, 6, 8, 10)))
        num_threads = _task_integer("num_threads", p.get("num_threads", 4))
        sdpb_options = _merged_sdpb_options(DEFAULT_OPE_MAX_SDPB_OPTIONS, p.get("sdpb_options"), execution=ctx.execution)
        delta_stars = _task_real_list("delta_stars", p.get("delta_stars", ["0.5", "1.0"]))
        lmax = max(spins) + 2
        tower = _task_boolean("tower", p.get("tower", True))
        q_order = _task_integer("q_order", p.get("q_order", preset["q_order"]))
        npoles = _task_integer("npoles", p.get("npoles", preset["npoles"]))
        n_tw = _task_integer("large_spin_double_twists", p.get(
            "large_spin_double_twists",
            5 if preset["large_spin_double_twists"] is None else preset["large_spin_double_twists"]))
        large_spin_npoles = _task_integer("large_spin_npoles", p.get(
            "large_spin_npoles",
            preset["npoles"] if preset["large_spin_npoles"] is None else preset["large_spin_npoles"]))
        # TWIST gap: delta_star is read as a uniform twist tau, so every spin
        # gets delta_min = ell + tau instead of the scalar channel alone getting
        # delta_min = delta_star. The tower's lowest slice tracks tau too, or the
        # asymptotic channel would sit below the gap it is meant to respect.
        all_spins_twist_gap = _task_boolean("all_spins_twist_gap", p.get("all_spins_twist_gap", False))

        def make_spec(delta_star: object) -> IdenticalScalarGapSpec:
            tower_kwargs: dict = {}
            if tower:
                twists = (
                    twists_above_gap(double_twist_set(h_ext, n_tw), delta_star)
                    if all_spins_twist_gap
                    else double_twist_set(h_ext, n_tw)
                )
                tower_kwargs = dict(
                    large_spin_lmin=lmax,
                    large_spin_twists=tuple(twists),
                    large_spin_pole_selection=ctx.selector("large_spin_pole_selection"),
                    large_spin_npoles=large_spin_npoles,
                )
            return IdenticalScalarGapSpec(
                c=c,
                h_ext=h_ext,
                delta_gap=delta_star,
                delta_min_by_spin=(
                    {ell: ell + delta_star for ell in spins if ell != 0}
                    if all_spins_twist_gap
                    else None
                ),
                lam=lam,
                spin_list=spins,
                q_order=q_order,
                dps=dps,
                npoles=npoles,
                pole_selection=ctx.selector("pole_selection"),
                num_threads=num_threads,
                **tower_kwargs,
            )

        if spin_star not in spins:
            raise ValueError(f"spin_star={spin_star} is not among the fixed spins {spins}")
        if not all_spins_twist_gap and any(delta_star < spin_star for delta_star in delta_stars):
            raise ValueError(f"delta_stars entries must be >= spin_star={spin_star} (unitarity)")

        logs = [f"c={mp.nstr(c, 5)}  Delta_ext={mp.nstr(2 * h_ext, 5)}  Lambda={lam}"]
        # One prepare serves the whole scan ONLY when delta_star moves nothing but
        # the scalar channel's delta_min. Under a twist gap it also moves every
        # other spin's delta_min and the tower's lowest slice, so the block
        # tables must be rebuilt per point (same rule as scalar_gap_bisection).
        prepared = None
        try:
            if not all_spins_twist_gap:
                prepared = prepare_identical_scalar_gap_build(make_spec(delta_stars[0]))
        except Exception as exc:
            return {
                "c": _result_number(c),
                "reason": f"prepare_error:{exc.__class__.__name__}",
                "logs": logs + [f"  -> prepare failed: {exc}"],
            }

        points = progress_mapping(ctx.journal.data, "points")
        for delta_star in delta_stars:
            key = _label(delta_star)
            operator_delta = spin_star + delta_star if all_spins_twist_gap else delta_star
            previous = points.get(key)
            # Older nonzero-spin twist-gap probes used tau as Delta. Their
            # verdicts cannot be reused for the operator at its proper edge.
            matches_operator = True
            if all_spins_twist_gap and spin_star:
                from virasoro_blocks.numeric_input import literal_number
                try:
                    matches_operator = literal_number(previous["operator_delta"]) == operator_delta
                except (KeyError, TypeError, ValueError, ArithmeticError):
                    matches_operator = False
            if isinstance(previous, dict) and (
                previous.get("status") != "optimal" or _has_finite_ope_bound(previous)
            ) and matches_operator:
                continue
            if ctx.should_stop():
                raise TaskInterrupted(f"stop requested before Delta*={key}")
            problem = build_identical_scalar_ope_max_problem(
                make_spec(delta_star), delta_star=operator_delta,
                spin_star=spin_star, prepared=prepared,
            )
            run = run_identical_scalar_gap_problem(
                problem,
                ctx.output_dir / f"dstar_{key}",
                backend=str(p.get("backend", "sdp")),
                sdpb_options=sdpb_options,
                pmp2sdp_log_name="pmp2sdp.log",
                sdpb_log_name="sdpb.log",
            )
            status = _checked_sdpb_status(run, f"Delta*={key}")
            obj = read_sdpb_objective(run.sdpb_out_dir) if run.sdpb_out_dir else None
            entry: dict = {"status": status, "delta_star": _result_number(delta_star),
                           "operator_delta": _result_number(operator_delta)}
            if status == "optimal":
                dual = _optimal_dual_objective(obj, where=f"at Delta*={key}")
                bound = ope_squared_bound_from_objective(dual)
                entry["ope_squared_bound"] = _result_number(bound)
                logs.append(
                    f"  Delta*={mp.nstr(mp.mpf(delta_star), 6)}  ->  OPE^2 <= "
                    f"{mp.nstr(mp.mpf(bound), 10)}   [BELOW the gap bound]"
                )
            elif status == "max_complementarity":
                logs.append(
                    f"  Delta*={mp.nstr(mp.mpf(delta_star), 6)}  ->  RUNAWAY "
                    f"(max_complementarity)   [ABOVE the gap bound]"
                )
            else:
                logs.append(
                    f"  Delta*={mp.nstr(mp.mpf(delta_star), 6)}  ->  INCONCLUSIVE ({status})"
                )
            points[key] = entry
            ctx.journal.update(points=points)

        # The bound is bracketed by the last finite Delta* and the first runaway.
        finite = [d for d in delta_stars if _has_finite_ope_bound(points.get(_label(d)))]
        runaway = [
            d for d in delta_stars
            if points.get(_label(d), {}).get("status") == "max_complementarity"
        ]
        lo = max(finite) if finite else None
        hi = min(runaway) if runaway else None
        inconsistent = lo is not None and hi is not None and lo >= hi
        if inconsistent:
            logs.append(
                f"  -> INCONSISTENT scan: finite OPE bound at Delta*={mp.nstr(mp.mpf(lo), 8)} "
                f"but runaway at Delta*={mp.nstr(mp.mpf(hi), 8)}; no gap bracket reported"
            )
            lo = hi = None
        if lo is not None and hi is not None:
            logs.append(
                f"  -> Delta*(0) in [{mp.nstr(mp.mpf(lo), 8)}, {mp.nstr(mp.mpf(hi), 8)}]"
            )
        return {
            "c": _result_number(c),
            "delta_ext": _result_number(2 * h_ext),
            "lam": lam,
            "points": {k: v for k, v in points.items()},
            "gap_bound_lo": None if lo is None else _result_number(lo),
            "gap_bound_hi": None if hi is None else _result_number(hi),
            "logs": logs,
            **({"reason": "inconsistent_scan"} if inconsistent else {}),
        }


@task("scalar_only_gap_probe", sdpb_defaults="DEFAULT_FEASIBILITY_SDPB_OPTIONS")
def scalar_only_gap_probe(ctx: TaskContext) -> dict:
    """Single-point ALLOWED/EXCLUDED classification for the scalar-only gap.

    The HPC-task form of the scalar-only per-point worker
    (spin_list=(0,), no large-spin tower, Delta_ext=delta_gap self-consistent):
    one solve, no bisection. Meant to coarsely map the shape of the allowed
    region -- which can be a bounded INTERVAL in this scalar-only setup, not
    just the usual half-line -- before committing to precise bisections for
    the upper and/or lower critical gap via
    scalar_gap_bisection(spins=[0], tower=False, self_consistent_h_ext=True,
    bound_direction=...).

    params (all optional except c, delta):
      c            central charge (string)
      delta        the gap value to classify (string); Delta_ext = delta,
                   h_ext = delta / 2
      lam          derivative order; keys LAMBDA_PRESETS (default 11)
      num_threads  block-build workers                (default 4)
      sdpb_options mapping of SDPB flags              (default DEFAULT_FEASIBILITY_SDPB_OPTIONS)
      backend      "sdp" | "pmp2sdp"                  (default "sdp")
    """
    from ..identical_scalar import (
        IdenticalScalarGapSpec,
        build_identical_scalar_gap_problem,
        run_identical_scalar_gap_problem,
    )

    p = ctx.params
    lam = _task_integer("lam", p.get("lam", 11))
    preset = ctx.profile(lam)
    dps = _task_integer("dps", p.get("dps", preset["dps"]))
    with mp.workdps(dps):
        c = _central_charge(p["c"])
        delta = _task_real_input("delta", p["delta"])
        h_ext = delta / 2
        num_threads = _task_integer("num_threads", p.get("num_threads", 4))
        sdpb_options = _merged_sdpb_options(DEFAULT_FEASIBILITY_SDPB_OPTIONS, p.get("sdpb_options"), execution=ctx.execution)
        backend = str(p.get("backend", "sdp"))

        spec = IdenticalScalarGapSpec(
            c=c, h_ext=h_ext, delta_gap=delta, delta_min_by_spin={},
            lam=lam, spin_list=(0,),
            q_order=int(preset["q_order"]), dps=dps, npoles=int(preset["npoles"]),
            pole_selection=ctx.selector("pole_selection"), num_threads=num_threads,
        )

        logs = [f"c={mp.nstr(c, 6)}  delta={mp.nstr(delta, 6)}:"]
        try:
            problem = build_identical_scalar_gap_problem(spec)
            run_options = dict(sdpb_options)
            max_runtime = _slurm_max_runtime_seconds()
            if max_runtime is not None and "--maxRuntime" not in run_options:
                run_options["--maxRuntime"] = max_runtime
            run = run_identical_scalar_gap_problem(
                problem, ctx.output_dir / "solve", backend=backend,
                sdpb_options=run_options,
                pmp2sdp_log_name="pmp2sdp.log", sdpb_log_name="sdpb.log",
            )
        except Exception as exc:
            logs.append(f"  -> solve failed: {exc.__class__.__name__}: {exc}")
            return {
                "c": _result_number(c), "delta": _result_number(delta),
                "reason": f"solve_error:{exc.__class__.__name__}", "error": str(exc),
                "logs": logs,
            }

        status = _checked_sdpb_status(run, f"delta={mp.nstr(delta, 6)}")

        if status not in _INCONCLUSIVE_SDPB_STATUSES:
            allowed = status == "primal_feasible"
            verdict = "ALLOWED" if allowed else f"EXCLUDED ({status})"
            logs.append(f"  -> {verdict}")
            return {
                "c": _result_number(c), "delta": _result_number(delta),
                "allowed": allowed, "sdpb_status": status, "logs": logs,
            }

        if status == "solver_error":
            # SDPB (or the container around it) failed to run: a failure to
            # retry, not a verdict. The cause is only in sdpb.log otherwise.
            logs.append(f"  -> solver error: {run.sdpb_summary}")
            return {
                "c": _result_number(c), "delta": _result_number(delta),
                "reason": "solve_error", "sdpb_status": status,
                "error": f"{run.sdpb_summary} (see {run.sdpb_log_path})", "logs": logs,
            }
        logs.append(f"  -> inconclusive: {status}")
        return {
            "c": _result_number(c), "delta": _result_number(delta),
            "reason": "inconclusive", "sdpb_status": status, "logs": logs,
        }


_DEFAULT_PARAMS_PATH = research_config_path()


def _load_default_params_profile(lam: int) -> dict:
    """Read the packaged ``default_params.json``'s ``lambdaN`` profile directly.

    Returns the full profile; :data:`LAMBDA_PRESETS` exposes only the
    block-table keys of the same config.

    Delegates to :func:`virasoro_bootstrap.defaults.profile`, which merges the
    config's cross-Lambda ``defaults`` block under the ``lambdaN`` profile. The
    selector keys therefore arrive here even though no ``lambdaN`` object states
    them -- they are declared once, in ``defaults``.
    """
    from ..defaults import profile as _profile

    return _profile(lam)


# Default normalized-moment certification coverage when the profile doesn't
# set these explicitly (only lambda7 sets them) to match the production
# fallback values.
_DEFAULT_CERT_EXTRA_SPINS = 8
_DEFAULT_CERT_EXTRA_TWISTS = 2
_DEFAULT_ADAPTIVE_CERT_EXTRA_SPINS = 20
_DEFAULT_ADAPTIVE_CERT_EXTRA_TWISTS = 2

_DEFAULT_MOMENT_SDPB_OPTIONS: dict[str, str | bool | int] = {
    "--writeSolution": "x,y",
    "--maxIterations": 1000,
    # Balance objective convergence against the iteration cost at Lambda=19.
    "--dualityGapThreshold": "1e-10",
    "--noFinalCheckpoint": True,
}


def _moment_certification_controls(p: dict, profile: dict, *, q_order: int | None) -> dict:
    """Use the same validated certification settings for solves and rechecks.

    Keep literal parameter reads so the task signature and submit validation
    can discover every control from this shared helper.
    """
    delta_range = _task_real_input("cert_delta_range", p.get("cert_delta_range", 30.0))
    if delta_range < 0:
        raise ValueError("cert_delta_range must be finite and nonnegative")
    return dict(
        extra_spins=_task_integer("cert_extra_spins", p.get(
            "cert_extra_spins", profile.get("cert_extra_spins", _DEFAULT_CERT_EXTRA_SPINS))),
        extra_twists=_task_integer("cert_extra_twists", p.get(
            "cert_extra_twists", profile.get("cert_extra_twists", _DEFAULT_CERT_EXTRA_TWISTS))),
        adaptive_moment_extra_spins=_task_integer("adaptive_cert_extra_spins", p.get(
            "adaptive_cert_extra_spins", profile.get("adaptive_cert_extra_spins", _DEFAULT_ADAPTIVE_CERT_EXTRA_SPINS))),
        adaptive_moment_extra_twists=_task_integer("adaptive_cert_extra_twists", p.get(
            "adaptive_cert_extra_twists", profile.get("adaptive_cert_extra_twists", _DEFAULT_ADAPTIVE_CERT_EXTRA_TWISTS))),
        interstitial_twists=_task_integer("interstitial_twists", p.get("interstitial_twists", 0)),
        n_test_points=_task_integer("cert_n_test_points", p.get("cert_n_test_points", 100)),
        log_near_points=_task_integer("cert_log_near_points", p.get("cert_log_near_points", 0)),
        moment_cert_q_order=q_order, delta_range=delta_range, adaptive_moment=True,
        physical_moment_blocks=_task_boolean("physical_moment_blocks", p.get("physical_moment_blocks", True)),
        force_exact_included=_task_boolean("force_exact_included", p.get("force_exact_included", False)),
    )


def _bind_moment_settings(ctx: TaskContext, spec, cert_controls: dict, sdpb_options=None) -> dict:
    """Record resolved settings, including for callers that bypass the worker."""
    from ..manifest import _spec_to_json
    from .provenance import settings_digest

    def encode(value):
        if isinstance(value, dict):
            return {str(k): encode(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [encode(v) for v in value]
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        return _result_number(value)

    settings = encode({"spec": _spec_to_json(spec), "certification": cert_controls,
                       "sdpb_options": sdpb_options, "params": ctx.params})
    previous = ctx.journal.data.get("resolved_settings")
    if progress_mapping(ctx.journal.data, "solved") and previous != settings:
        suffix = settings_digest(previous) if isinstance(previous, dict) else "unrecorded"
        path = ctx.journal.path
        archive = path.with_name(f"{path.stem}.settings-{suffix}{path.suffix}")
        path.replace(archive)
        print(f"{path}: resolved moment settings differ or were not recorded; saved {archive.name}, starting afresh")
        ctx.journal.update(solved={}, resolved_settings=settings)
    elif previous != settings:
        ctx.journal.update(resolved_settings=settings)
    return settings


def _moment_nu1_over_nu0_spec(p: dict, profile: dict, *, lam: int, dps: int, num_threads: int):
    """The ``IdenticalScalarGapSpec`` a ``moment_nu1_over_nu0`` task solves.

    Shared with ``moment_recertify``, which must rebuild exactly the solved
    problem: its own copy had drifted (it dropped ``delta_gap``,
    ``all_spins_twist_gap``, the pole-selector overrides and
    ``shared_sample_points``) and so certified a different problem.  Call it
    inside ``mp.workdps(dps)``.
    """
    from ..identical_scalar import IdenticalScalarGapSpec
    from ..large_spin import double_twist_set, twists_above_gap

    c = _central_charge(p["c"])
    h_ext = _task_real_input("delta_ext", p["delta_ext"]) / 2
    q_order = _task_integer("q_order", p.get("q_order", profile["q_order"]))
    npoles = _task_integer("npoles", p.get("npoles", profile["npoles"]))
    # Kept-pole selector. The profile default is contribution_optimal;
    # individual tasks may override it for an explicit comparison.
    pole_selection = str(p.get("pole_selection", profile.get("pole_selection", DEFAULT_POLE_SELECTION)))
    # Tower selector, separate from the fixed-spin one. Its inverted
    # geometry makes large_spin_npoles an independent convergence control.
    large_spin_pole_selection = str(p.get(
        "large_spin_pole_selection",
        profile.get("large_spin_pole_selection", DEFAULT_LARGE_SPIN_POLE_SELECTION),
    ))
    spins = _task_spins(p.get("spins", profile["spin_list"]))
    double_twists = _task_integer("large_spin_double_twists", p.get("large_spin_double_twists", profile.get("large_spin_double_twists", 0)))
    shared_sample_points = _task_boolean("shared_sample_points", p.get("shared_sample_points", False))
    delta_gap = _task_real_input("delta_gap", p.get("delta_gap", "0"))
    # TWIST gap: read delta_gap as a uniform twist tau, so every spin gets
    # delta_min = ell + tau and the tower's lowest slice tracks tau, rather
    # than the scalar channel alone being gapped. Liouville is the reason to
    # want this: it has NO spinning operators, so it satisfies every ell>0
    # constraint vacuously and only its scalar threshold (c-1)/12 bites --
    # imposing a twist gap therefore drives the problem toward the
    # scalar-only/Liouville corner.
    all_spins_twist_gap = _task_boolean("all_spins_twist_gap", p.get("all_spins_twist_gap", False))

    custom_twists = p.get("large_spin_twists")
    if isinstance(custom_twists, (str, bytes)):
        raise ValueError("large_spin_twists must be a sequence of real numbers")
    tower_kwargs: dict = {}
    if double_twists or custom_twists:
        twists = (
            tuple(_task_real_input("large_spin_twists entry", t) for t in custom_twists)
            if custom_twists
            else tuple(
                twists_above_gap(double_twist_set(h_ext, double_twists), delta_gap)
                if all_spins_twist_gap
                else double_twist_set(h_ext, double_twists)
            )
        )
        # Slices below the lowest one, every `large_spin_low_twist_step` from the twist threshold (0, or the twist
        # gap) up to it.  The double twists start at 2 Delta_ext, and unitarity allows any twist >= 0 at large spin:
        # without these the SDP leaves L >= large_spin_lmin with twist under 2 Delta_ext unconstrained (Lambda 19,
        # c = 1.00001, Delta_ext 1.0 and 1.3: the functional negative at twist 0 on spins 26-40, where the c = 1
        # theories have twist-0 primaries).
        low_step = p.get("large_spin_low_twist_step")
        if low_step is not None and twists:
            step = _task_real_input("large_spin_low_twist_step", low_step)
            if not step > 0:
                raise ValueError(f"large_spin_low_twist_step must be > 0, got {low_step!r}")
            start = delta_gap if all_spins_twist_gap else 0
            lowest = min(twists)
            low = []
            k = 0
            while start + k * step < lowest - step / 2:
                low.append(start + k * step)
                k += 1
            twists = tuple(low) + tuple(twists)
        if not custom_twists and not all_spins_twist_gap and twists and min(twists) > 0:
            # With no twist gap, include the twist-zero unitarity threshold.
            # An explicit large_spin_twists list is used as supplied.
            twists = (0,) + tuple(twists)
        tower_kwargs = dict(
            # Start above the fixed spins by default. A lower floor extends the continuous range and leads to
            # some near-degenerate constraints, but generally leaves bounds unchanged.
            large_spin_lmin=_task_integer("large_spin_lmin", p.get("large_spin_lmin", max(spins) + 2)),
            large_spin_twists=twists,
            large_spin_pole_selection=large_spin_pole_selection,
            large_spin_npoles=_task_optional_integer("large_spin_npoles", p.get("large_spin_npoles", profile.get("large_spin_npoles"))),
        )
    return IdenticalScalarGapSpec(
        # delta_gap defaults to 0 -- the moment bound is over the FULL
        # spectrum, so any nonzero value changes the physics and is a
        # diagnostic rather than a production setting. It exists because
        # moving the positivity threshold off zero localizes numerical
        # trouble: if an npoles discrepancy closes once the near-threshold
        # region is excluded, the difficulty lives at x -> 0, which is
        # exactly where the interpolation packs its nodes.
        c=c, h_ext=h_ext, delta_gap=delta_gap, lam=lam, spin_list=spins,
        delta_min_by_spin=(
            {ell: ell + delta_gap for ell in spins if ell != 0}
            if all_spins_twist_gap
            else None
        ),
        q_order=q_order, dps=dps, npoles=npoles,
        pole_selection=pole_selection,
        # Off by default. Collapsing every derivative entry onto the single
        # max-degree node set makes each one EXACT at the points the SDP
        # evaluates (a low-order entry otherwise carries ~1e-12 error there
        # at npoles=16) and slightly improves in-span accuracy, at the cost
        # of raising the low-order polynomial degrees, which breaks the
        # large-Delta tail power (ratio off by 5.6e14 by Delta=400). Whether
        # that tail matters to the BOUND is an empirical question -- it is
        # exponentially suppressed -- so this exists to be A/B'd rather than
        # argued about.
        shared_sample_points=shared_sample_points,
        num_threads=num_threads, **tower_kwargs,
    )


@task("moment_nu1_over_nu0", sdpb_defaults="_DEFAULT_MOMENT_SDPB_OPTIONS")
def moment_nu1_over_nu0(ctx: TaskContext) -> dict:
    """nu_1/nu_0 normalized-moment upper/lower bounds at one (c, Delta_ext).

    Solves up to 4 independent direct-SDP problems -- {lower, upper} bounds
    crossed with identity {include, exclude} -- sharing one block-table
    ``prepare`` (the expensive, gap/objective-independent part). Built for two
    uses at once: (a) a normal sweep point (vary c/Delta_ext, fixed params),
    and (b) a parameter-convergence or mpiProcs timing probe (fixed c/Delta_ext,
    vary q_order/npoles/sdpb_options across tasks) -- every block-table knob
    the profile sets can be overridden per task, and both the prepare phase
    and each SDPB solve are individually timed in the result for exactly that
    purpose.

    Each of the 4 solves is independent (no hotstart between them -- lower and
    upper bounds are different objectives), so ``ctx.journal`` records each
    completed ``{identity_mode}_{direction}`` result as it lands: an
    interruption only re-solves whichever of the 4 hadn't finished yet.

    params (``c`` required; everything else optional):
      c, delta_ext     central charge / external dimension (strings; "8/7" ok)
      lam              derivative order; profile = virasoro_bootstrap/default_params.json
                        lambda{lam}                              (default 19)
      q_order, npoles, dps, spins, large_spin_double_twists
                        override the profile's values             (optional)
      large_spin_lmin
                        first spin covered by the tower (profile default);
                        cannot leave a gap beyond the fixed spins
      large_spin_twists
                        explicit tower twist list (e.g. a denser, non-
                        double-twist-spaced grid), overriding
                        large_spin_double_twists's double_twist_set(...)
                        entirely when given                       (optional)
                        Without it, and without a twist gap, the tower is
                        twist 0 (the unitarity threshold) plus the double
                        twists
      large_spin_low_twist_step
                        add tower slices every this much from twist 0
                        (the twist gap with all_spins_twist_gap) up to the
                        lowest slice: the double twists start at
                        2 Delta_ext, and without these the SDP leaves the
                        large-spin twists below it unconstrained (optional)
      large_spin_npoles
                        tower kept-pole count per twist channel
                        (profile default: 8). Controls how tightly SDPB's
                        own polynomial approximation of the tower block
                        tracks the true block -- too low and SDPB can
                        certify positivity against its OWN approximation
                        while the true block (which the certificate
                        re-derives exactly) is still slightly negative
                        nearby. Raising it costs SDP conditioning (the
                        tower's inverted-geometry pole coefficients grow
                        fast), so watch the solved VALUE for distortion,
                        not just whether certification clears    (optional)
      identity_modes   subset of ["include","exclude"]  (default both)

    Result fields per {identity_mode}: ``{direction}`` is the certified bound
    (null unless status == "optimal"), ``{direction}_solved_value`` is the
    solved objective recorded REGARDLESS of certification, and
    ``{direction}_status`` the verdict. Use ``_solved_value`` only for
    convergence diagnostics -- an uncertified value is not a bound (it is
    spuriously STRONG, since the functional is not positive everywhere).
    A normalized-moment solve whose details are only an interpolant certificate
    or a finite true-block grid is deliberately reported as
    ``optimal_uncertified``; the scope is recorded separately when spot checks
    are retained.
      directions       subset of ["lower","upper"]       (default both)
      backend          "sdp" | "pmp2sdp"                 (default "sdp")
      precision        SDPB --precision bit count     (default profile's)
      sdpb_options     extra/overriding SDPB flags (merged over the defaults
                        above; put --mpiProcs here for an mpiProcs benchmark)
      cert_extra_spins, cert_extra_twists, adaptive_cert_extra_spins,
      adaptive_cert_extra_twists, moment_cert_q_order
                        certification coverage (profile, else the
                        lambda7-derived fallbacks above)
      physical_moment_blocks
                        bool: evaluate normalized-moment certification on
                        freshly evaluated physical blocks rather than the
                        serialized SDP interpolants (default True)
      force_exact_included
                        bool: rebuild the INCLUDED normalized-moment channels
                        for the certification pass as well (default False;
                        the legacy fast path checks the SDP interpolant)
      interstitial_twists
                        int: add physical tower checks between the discrete
                        twists used by the SDP (default 0)
      cert_n_test_points, cert_delta_range, cert_log_near_points
                        override the physical-block certification grid; the
                        last option adds logarithmic points immediately above
                        each threshold (defaults 100, 30, 0)
      spot_check       bool: widen to the adaptive coverage and record
                        per-channel certify detail regardless of result
                        (default False)
      keep_checkpoint  bool: write a real, hotstart-usable SDPB checkpoint
                        for each solve (drops --noFinalCheckpoint, the
                        default). Off by default since checkpoints cost disk
                        and most sweeps only need the verdict; turn on when
                        you might want to chain a later re-solve (e.g. at a
                        tighter --dualityGapThreshold) from this one.
      initial_checkpoint_source_dir
                        path to a PRIOR task's output_dir (its
                        ``{identity_mode}_{direction}/sdp.ck``) to hotstart
                        each of THIS task's solves from -- only valid when
                        the source was run with keep_checkpoint=true AND the
                        PMP block structure matches (same spin_list/tower/
                        npoles/q_order; changing --dualityGapThreshold or
                        --maxIterations is fine, that's the point).
    """
    from ..certification import (
        _is_normalized_moment_problem,
        certify_dual_from_sdpb_output,
    )
    from ..identical_scalar import (
        prepare_identical_scalar_gap_build,
        read_sdpb_objective,
        run_identical_scalar_gap_problem,
    )
    from ..moments import (
        MomentWeight,
        build_identical_scalar_normalized_moment_lower_problem,
        build_identical_scalar_normalized_moment_upper_problem,
        normalized_moment_lower_bound_from_objective,
        normalized_moment_upper_bound_from_objective,
    )

    p = ctx.params
    lam = _task_integer("lam", p.get("lam", 19))
    profile = ctx.profile(lam, moment=True)
    dps = _task_integer("dps", p.get("dps", profile["dps"]))
    with mp.workdps(dps):
        c = _central_charge(p["c"])
        delta_ext = _task_real_input("delta_ext", p["delta_ext"])
        q_order = _task_integer("q_order", p.get("q_order", profile["q_order"]))
        npoles = _task_integer("npoles", p.get("npoles", profile["npoles"]))
        identity_modes = _task_choice_list("identity_modes", p.get("identity_modes", ("include", "exclude")), ("include", "exclude"))
        directions = _task_choice_list("directions", p.get("directions", ("lower", "upper")), ("lower", "upper"))
        backend = str(p.get("backend", "sdp"))
        precision = _task_integer("precision", p.get("precision", profile["sdpb_precision"]))
        spot_check = _task_boolean("spot_check", p.get("spot_check", False))
        shared_sample_points = _task_boolean("shared_sample_points", p.get("shared_sample_points", False))
        # (the spec itself -- selectors, spins, tower, gaps -- is parsed by
        # _moment_nu1_over_nu0_spec, shared with moment_recertify)
        delta_gap = _task_real_input("delta_gap", p.get("delta_gap", "0"))
        all_spins_twist_gap = _task_boolean("all_spins_twist_gap", p.get("all_spins_twist_gap", False))

        cert_q_order = _task_optional_integer("moment_cert_q_order", p.get(
            "moment_cert_q_order", profile.get("moment_cert_q_order")))
        cert_controls = _moment_certification_controls(p, profile, q_order=cert_q_order)
        num_threads = _task_integer("num_threads", p.get("num_threads", 4))
        keep_checkpoint = _task_boolean("keep_checkpoint", p.get("keep_checkpoint", False))
        checkpoint_source = p.get("initial_checkpoint_source_dir")
        initial_checkpoint_source_dir = Path(checkpoint_source) if checkpoint_source else None

        moment_sdpb_defaults = {**_DEFAULT_MOMENT_SDPB_OPTIONS, "--precision": precision}
        sdpb_options = _merged_sdpb_options(moment_sdpb_defaults, p.get("sdpb_options"), execution=ctx.execution)
        if keep_checkpoint:
            sdpb_options.pop("--noFinalCheckpoint", None)

        spec = _moment_nu1_over_nu0_spec(p, profile, lam=lam, dps=dps, num_threads=num_threads)

        resolved_settings = _bind_moment_settings(ctx, spec, cert_controls, sdpb_options)

        logs = [
            f"c={mp.nstr(c, 6)}  Delta_ext={mp.nstr(delta_ext, 6)}  Lambda={lam}  "
            f"q_order={q_order}  npoles={npoles}  dps={dps}"
            + ("  shared_sample_points" if shared_sample_points else "")
            + (
                f"  {'twist_gap' if all_spins_twist_gap else 'delta_gap'}"
                f"={mp.nstr(delta_gap, 6)}"
                if delta_gap != 0 else ""
            )
        ]
        result: dict = {
            "c": _result_number(c), "delta_ext": _result_number(delta_ext), "lam": lam,
            "q_order": q_order, "npoles": npoles, "dps": dps,
            "large_spin_lmin": spec.large_spin_lmin, "resolved_settings": resolved_settings,
        }

        t0 = time.monotonic()
        try:
            prepared = prepare_identical_scalar_gap_build(spec)
        except Exception as exc:
            logs.append(f"  -> prepare failed: {exc.__class__.__name__}: {exc}")
            result["reason"] = f"prepare_error:{exc.__class__.__name__}"
            result["logs"] = logs
            return result
        result["prepare_seconds"] = round(time.monotonic() - t0, 2)

        numerator = MomentWeight.monomial(Delta=1)
        denominator = MomentWeight.one()
        solved: dict = progress_mapping(ctx.journal.data, "solved")

        for identity_mode in identity_modes:
            result[identity_mode] = {}
            for direction in directions:
                key = f"{identity_mode}_{direction}"
                if _has_saved_moment_bound(solved.get(key)):
                    result[identity_mode][direction] = solved[key].get("value")
                    result[identity_mode][f"{direction}_solved_value"] = solved[key].get("solved_value")
                    result[identity_mode][f"{direction}_status"] = solved[key].get("status")
                    result[identity_mode][f"{direction}_solve_seconds"] = solved[key].get("solve_seconds")
                    for diagnostic in ("certificate_scope", "spot_check"):
                        if diagnostic in solved[key]:
                            result[identity_mode][f"{direction}_{diagnostic}"] = solved[key][diagnostic]
                    logs.append(f"  {key}: resumed from journal ({solved[key].get('status')})")
                    continue
                if ctx.should_stop():
                    raise TaskInterrupted(f"stop requested before {key}")

                if direction == "lower":
                    problem = build_identical_scalar_normalized_moment_lower_problem(
                        spec, numerator, denominator, identity_mode=identity_mode, prepared=prepared,
                    )
                    convert = normalized_moment_lower_bound_from_objective
                else:
                    problem = build_identical_scalar_normalized_moment_upper_problem(
                        spec, numerator, denominator, identity_mode=identity_mode, prepared=prepared,
                    )
                    convert = normalized_moment_upper_bound_from_objective

                run_options = dict(sdpb_options)
                max_runtime = _slurm_max_runtime_seconds()
                if max_runtime is not None and "--maxRuntime" not in run_options:
                    run_options["--maxRuntime"] = max_runtime

                initial_checkpoint_dir = None
                if initial_checkpoint_source_dir is not None:
                    candidate = initial_checkpoint_source_dir / key / "sdp.ck"
                    if not candidate.is_dir():
                        raise FileNotFoundError(
                            f"initial_checkpoint_source_dir has no checkpoint at {candidate} "
                            "(source run must have used keep_checkpoint=true)"
                        )
                    initial_checkpoint_dir = candidate

                t_solve = time.monotonic()
                try:
                    run = run_identical_scalar_gap_problem(
                        problem, ctx.output_dir / key, backend=backend,
                        sdpb_options=run_options,
                        pmp2sdp_log_name="pmp2sdp.log", sdpb_log_name="sdpb.log",
                        initial_checkpoint_dir=initial_checkpoint_dir,
                    )
                except Exception as exc:
                    logs.append(f"  {key}: solve failed: {exc.__class__.__name__}: {exc}")
                    solved[key] = {
                        "value": None, "solved_value": None,
                        "status": "solve_error", "solve_seconds": None,
                    }
                    ctx.journal.update(solved=solved)
                    result[identity_mode][direction] = None
                    result[identity_mode][f"{direction}_solved_value"] = None
                    result[identity_mode][f"{direction}_status"] = "solve_error"
                    continue
                solve_seconds = round(time.monotonic() - t_solve, 2)

                obj = read_sdpb_objective(run.sdpb_out_dir) if run.sdpb_out_dir else None
                value: str | None = None
                solved_value: str | None = None
                # A --maxRuntime exit is not a verdict: don't journal it as
                # one. TaskInterrupted makes the worker requeue, and the next
                # allocation retries this exact solve from scratch.
                status = _checked_sdpb_status(run, key)
                details = None
                if status == "optimal" and obj and "dual" in obj:
                    certified, details = certify_dual_from_sdpb_output(
                        problem, run.sdpb_out_dir,
                        **cert_controls,
                        num_threads=num_threads,
                    )
                    # Retain the solved objective for coverage-convergence
                    # comparisons regardless of certification. The separate
                    # `value` field is populated only for certified bounds.
                    solved_value = _result_number(convert(obj["dual"]))
                    # ``certify_dual_from_sdpb_output`` returns the finite-grid
                    # verdict for physical numeric blocks.  That is useful
                    # diagnostics, but it is not a global certificate for a
                    # continuous large-spin tower.  Likewise the direct-SDP
                    # root isolation is global only for the serialized
                    # interpolation polynomial, not the physical block.  Do
                    # not publish either as an ordinary bound.
                    non_global_scope = any(
                        d.get("certificate_scope") in _NON_GLOBAL_MOMENT_SCOPES
                        or d.get("basis") == "physical_numeric_blocks"
                        for d in (details or [])
                    )
                    if non_global_scope and _is_normalized_moment_problem(problem):
                        status = "optimal_uncertified"
                    elif not certified:
                        status = "optimal_uncertified"
                    else:
                        value = solved_value

                solved[key] = {
                    "value": value,
                    "solved_value": solved_value,
                    "status": status,
                    "solve_seconds": solve_seconds,
                }
                result[identity_mode][direction] = value
                result[identity_mode][f"{direction}_solved_value"] = solved_value
                result[identity_mode][f"{direction}_status"] = status
                result[identity_mode][f"{direction}_solve_seconds"] = solve_seconds
                if details:
                    scopes = sorted({
                        str(d.get("certificate_scope"))
                        for d in details
                        if d.get("certificate_scope") is not None
                    })
                    if scopes:
                        result[identity_mode][f"{direction}_certificate_scope"] = scopes
                        solved[key]["certificate_scope"] = scopes
                verdict = value if value is not None else f"no bound ({status})"
                logs.append(f"  {key}: {verdict}  [{solve_seconds}s]")
                if spot_check and details is not None:
                    n_pass = sum(1 for d in details if d.get("passed"))
                    logs.append(f"    spot-check: {n_pass}/{len(details)} channels passed")
                    result[identity_mode][f"{direction}_spot_check"] = details
                    solved[key]["spot_check"] = details
                ctx.journal.update(solved=solved)

        result["logs"] = logs
        return result


@task("moment_recertify")
def moment_recertify(ctx: TaskContext) -> dict:
    """Recheck a saved moment_nu1_over_nu0 result without another SDPB solve.

    Use ``moment_cert_q_order`` to change the certification's q-series order.

    Raising this order tests whether an ``optimal_uncertified`` verdict changes
    as the certification's q-series converges. The task loads the saved
    manifest and hash-checked block files, verifies that the requested physics
    and numerical settings match the original solve, and calls
    :func:`~virasoro_bootstrap.certification.certify_dual_from_sdpb_output`
    on the existing ``sdpb_out`` directories under ``source_output_dir``.
    Each selected solve must have a manifest. Certification may evaluate new
    blocks, but it does not rebuild or solve the SDP.

    params:
      c, delta_ext, lam, q_order, npoles, dps, spins, large_spin_double_twists,
      large_spin_lmin, large_spin_twists, large_spin_npoles, pole_selection,
      large_spin_pole_selection, delta_gap, all_spins_twist_gap,
      shared_sample_points
                        must match the original solve's spec (see above); the
                        spec is built by the same helper as moment_nu1_over_nu0
      source_output_dir  path to the original task's output dir (containing
                        ``{identity_mode}_{direction}/sdpb_out`` subdirs,
                        e.g. the original run's ``tasks/0000``)
      moment_cert_q_order  the q-series truncation to test certification at
                        (required)
      identity_modes, directions  which of the original solves to recheck
                        (default both x both, matching moment_nu1_over_nu0)
      cert_extra_spins, cert_extra_twists, adaptive_cert_extra_spins,
      adaptive_cert_extra_twists  certification coverage (profile, else the
                        lambda7-derived fallbacks, same as moment_nu1_over_nu0)
      spot_check        include per-channel detail in the result (default
                        True)
      interstitial_twists  check this many tower twist lines between each pair
                        of double-twist slices (default 0). The SDP constrains
                        positivity only on the slices, while unitarity permits
                        operators at any tau >= 0 at large spin, so a dip in
                        between is a real gap in the relaxation. The extra-twist
                        knob alone cannot see it: it extends the double-twist
                        count, which only adds lines above the top slice.
      physical_moment_blocks, cert_n_test_points, cert_delta_range,
      cert_log_near_points  same physical certification grid controls as the
                        solve task (defaults True, 100, 30, 0)
      force_exact_included  with physical_moment_blocks=True, evaluate the
                        included blocks directly instead of using the saved
                        SDP samples (default False). The default check uses the
                        solve's npoles interpolant and cannot detect its block
                        interpolation error. Use this option to check that error.
    """
    from ..certification import (
        _is_normalized_moment_problem,
        certify_dual_from_sdpb_output,
    )
    from ..manifest import build_identical_scalar_problem_from_manifest
    from ..moments import MomentWeight, _resolve_base_problem

    p = ctx.params
    lam = _task_integer("lam", p.get("lam", 19))
    profile = ctx.profile(lam, moment=True)
    dps = _task_integer("dps", p.get("dps", profile["dps"]))
    with mp.workdps(dps):
        c = _central_charge(p["c"])
        delta_ext = _task_real_input("delta_ext", p["delta_ext"])
        q_order = _task_integer("q_order", p.get("q_order", profile["q_order"]))
        npoles = _task_integer("npoles", p.get("npoles", profile["npoles"]))
        identity_modes = _task_choice_list("identity_modes", p.get("identity_modes", ("include", "exclude")), ("include", "exclude"))
        directions = _task_choice_list("directions", p.get("directions", ("lower", "upper")), ("lower", "upper"))
        source_output_dir = Path(p["source_output_dir"])
        moment_cert_q_order = _task_integer("moment_cert_q_order", p["moment_cert_q_order"])
        cert_controls = _moment_certification_controls(p, profile, q_order=moment_cert_q_order)
        num_threads = _task_integer("num_threads", p.get("num_threads", 4))
        spot_check = _task_boolean("spot_check", p.get("spot_check", True))

        # the same spec builder as moment_nu1_over_nu0: certification is checked
        # against that solve's y, so every spec knob (delta_gap, the twist gap,
        # pole selectors, shared_sample_points, ...) must match it exactly
        spec = _moment_nu1_over_nu0_spec(p, profile, lam=lam, dps=dps, num_threads=num_threads)
        resolved_settings = _bind_moment_settings(ctx, spec, cert_controls)

        logs = [
            f"recertify c={mp.nstr(c, 6)}  Delta_ext={mp.nstr(delta_ext, 6)}  "
            f"solve q_order={q_order} npoles={npoles}  moment_cert_q_order={moment_cert_q_order}"
        ]
        result: dict = {
            "c": _result_number(c), "delta_ext": _result_number(delta_ext), "lam": lam,
            "q_order": q_order, "npoles": npoles, "moment_cert_q_order": moment_cert_q_order,
            "large_spin_lmin": spec.large_spin_lmin, "resolved_settings": resolved_settings,
        }

        numerator = MomentWeight.monomial(Delta=1)
        denominator = MomentWeight.one()
        solved: dict = progress_mapping(ctx.journal.data, "solved")

        for identity_mode in identity_modes:
            result[identity_mode] = {}
            for direction in directions:
                key = f"{identity_mode}_{direction}"
                if _has_saved_moment_certificate(solved.get(key)):
                    result[identity_mode][direction] = solved[key]
                    logs.append(f"  {key}: resumed from journal")
                    continue
                if ctx.should_stop():
                    raise TaskInterrupted(f"stop requested before {key}")

                sdpb_out_dir = source_output_dir / key / "sdpb_out"
                if not sdpb_out_dir.is_dir():
                    entry = {"error": f"no sdpb_out at {sdpb_out_dir}"}
                    result[identity_mode][direction] = entry
                    logs.append(f"  {key}: {entry['error']}")
                    continue

                manifest_path = sdpb_out_dir.parent / "manifest.json"
                try:
                    problem = build_identical_scalar_problem_from_manifest(manifest_path)
                    _resolve_base_problem(spec, None, problem)
                    moment = (problem.pmp_metadata or {}).get("moment", {})
                    if (
                        problem.problem_kind != f"normalized_moment_{direction}"
                        or moment.get("identity_mode") != identity_mode
                        or MomentWeight.from_json(moment["numerator"]) != numerator
                        or MomentWeight.from_json(moment["denominator"]) != denominator
                        or mp.mpf(moment.get("alpha", 0)) != 0
                        or mp.mpf(moment.get("beta", 0)) != 0
                    ):
                        raise ValueError("manifest does not describe the requested nu_1/nu_0 moment problem")
                except (OSError, ValueError, KeyError) as exc:
                    entry = {"error": f"cannot load matching moment manifest {manifest_path}: {exc}"}
                    result[identity_mode][direction] = entry
                    logs.append(f"  {key}: {entry['error']}")
                    # A repaired source can be retried; failed loads are not completed checks.
                    continue

                t_cert = time.monotonic()
                certified, details = certify_dual_from_sdpb_output(
                    problem, sdpb_out_dir,
                    **cert_controls, num_threads=num_threads,
                )
                cert_seconds = round(time.monotonic() - t_cert, 2)
                n_pass = sum(1 for d in details if d.get("passed")) if details else None
                n_total = len(details) if details else None
                scopes = sorted({
                    str(d.get("certificate_scope"))
                    for d in (details or [])
                    if d.get("certificate_scope") is not None
                })
                non_global_scope = any(
                    d.get("certificate_scope") in _NON_GLOBAL_MOMENT_SCOPES
                    or d.get("basis") == "physical_numeric_blocks"
                    for d in (details or [])
                )
                # A recertification task must use the same strict semantics as
                # the original moment task. Keep the raw finite-grid verdict
                # separately: it is valuable convergence evidence, but it is
                # not a publishable bound while the continuous tower remains
                # uncertified.
                globally_certified = bool(certified)
                if _is_normalized_moment_problem(problem):
                    globally_certified = bool(certified) and bool(details) and not non_global_scope
                entry = {
                    "certified": globally_certified,
                    "finite_grid_passed": bool(certified),
                    "n_pass": n_pass, "n_total": n_total,
                    "cert_seconds": cert_seconds,
                }
                if scopes:
                    entry["certificate_scope"] = scopes
                if spot_check and details is not None:
                    entry["details"] = details
                solved[key] = entry
                ctx.journal.update(solved=solved)
                result[identity_mode][direction] = entry
                logs.append(
                    f"  {key}: certified={globally_certified} "
                    f"finite_grid={bool(certified)}  "
                    f"({n_pass}/{n_total} channels)  [{cert_seconds}s]"
                )

        result["logs"] = logs
        return result
