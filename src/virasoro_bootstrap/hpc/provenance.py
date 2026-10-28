"""Parameter snapshots used for task execution and reuse checks."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re


_DEFAULT_LAM = {
    "isolated_relevant_scalar_grid_point": 11,
    "isolated_scalar_four_point_ope_bounds": 11,
    "isolated_scalar_four_point_fixed_ope": 11,
    "modular_scalar_gap_bisection": 11,
    "isolated_relevant_scalar_delta_phi_intervals": 11,
    "isolated_relevant_scalar_degeneracy_intervals": 11,
    "scalar_gap_bisection": 7,
    "scalar_ope_max_scan": 11,
    "scalar_only_gap_probe": 11,
    "moment_nu1_over_nu0": 19,
    "moment_recertify": 19,
}
_MOMENT_TASKS = {"moment_nu1_over_nu0", "moment_recertify"}


def _without_comments(value):
    if isinstance(value, dict):
        return {str(k): _without_comments(v) for k, v in value.items() if not str(k).startswith("_comment")}
    if isinstance(value, (list, tuple)):
        return [_without_comments(v) for v in value]
    return value


def _snapshot(task_name: str, params: dict, recorded: dict | None = None) -> dict:
    """Read the sources the task uses, or recover them from an older record."""
    from .. import defaults
    from . import tasks

    lam = params.get("lam", _DEFAULT_LAM.get(task_name))
    lam = None if lam is None else tasks._task_integer("lam", lam)
    flags = tasks.TASK_SDPB_DEFAULTS.get(task_name)
    selectors = {key: getattr(tasks, "DEFAULT_" + key.upper())
                 for key in ("pole_selection", "large_spin_pole_selection")}
    if recorded is None:
        if task_name in _MOMENT_TASKS:
            profile = tasks._load_default_params_profile(lam)
        elif task_name in _DEFAULT_LAM:
            # These tasks use import-time presets, not a later file snapshot.
            profile = tasks.LAMBDA_PRESETS[lam]
        else:
            profile = _custom_profiles(defaults.load_default_params(), lam)
        solver = None if flags is None else getattr(tasks, flags)
    elif recorded["schema"] == 1:
        config = recorded["defaults"]
        constants = recorded["task_defaults"]
        selectors = {key: constants["DEFAULT_" + key.upper()] for key in selectors}
        if task_name in _MOMENT_TASKS:
            profile = dict(defaults._profile_from_snapshot(config, lam))
            for key in ("cert_extra_spins", "cert_extra_twists",
                        "adaptive_cert_extra_spins", "adaptive_cert_extra_twists"):
                profile.setdefault(key, constants["_DEFAULT_" + key.upper()])
        elif task_name in _DEFAULT_LAM:
            presets = recorded["presets"]
            profile = presets[str(lam)] if str(lam) in presets else presets[lam]
        else:
            profile = _custom_profiles(config, lam)
        solver = None if flags is None else constants[flags]
    elif recorded["schema"] == 2:
        profiles = recorded["profiles"]
        profile = profiles if lam is None else profiles[f"lambda{lam}"]
        if task_name in _DEFAULT_LAM:
            selectors = {key: profile[key] for key in selectors}
        solver = recorded["sdpb_defaults"]
    else:
        raise ValueError("unsupported task settings schema")
    if task_name in _MOMENT_TASKS:
        profile = dict(profile)
        for key, value in selectors.items():
            profile.setdefault(key, value)
        for key in ("cert_extra_spins", "cert_extra_twists",
                    "adaptive_cert_extra_spins", "adaptive_cert_extra_twists"):
            profile.setdefault(key, getattr(tasks, "_DEFAULT_" + key.upper()))
    return deepcopy({"lam": lam, "profile": profile, "selectors": selectors,
                     "sdpb_defaults": solver})


def _custom_profiles(config: dict, lam) -> dict:
    from ..defaults import _profile_from_snapshot

    if lam is not None:
        return _profile_from_snapshot(config, lam)
    return {key: _profile_from_snapshot(config, int(match.group(1)))
            for key in config if (match := re.fullmatch(r"lambda(\d+)", key))}


def _effective(task_name: str, p: dict, source: dict) -> dict:
    """Select consumed defaults and apply overrides before comparing them."""
    from . import tasks

    profile = source["profile"]
    selectors = source["selectors"]
    flags = source["sdpb_defaults"]
    solver = None if flags is None else tasks._merged_sdpb_options(flags, p.get("sdpb_options"))
    values = {"lam": source["lam"]}

    if task_name in _MOMENT_TASKS:
        import mpmath as mp
        from ..manifest import _spec_to_json

        dps = tasks._task_integer("dps", p.get("dps", profile["dps"]))
        with mp.workdps(dps):
            spec = tasks._moment_nu1_over_nu0_spec(
                p, profile, lam=source["lam"], dps=dps,
                num_threads=tasks._task_integer("num_threads", p.get("num_threads", 4)))
            q_order = (p["moment_cert_q_order"] if task_name == "moment_recertify"
                       else p.get("moment_cert_q_order", profile.get("moment_cert_q_order")))
            q_order = tasks._task_optional_integer("moment_cert_q_order", q_order)
            controls = tasks._moment_certification_controls(p, profile, q_order=q_order)
            controls["delta_range"] = tasks._result_number(controls["delta_range"])
            values.update(spec=_spec_to_json(spec), certification=controls)
        if flags is not None:
            flags = {**flags, "--precision": tasks._task_integer(
                "precision", p.get("precision", profile["sdpb_precision"]))}
            solver = tasks._merged_sdpb_options(flags, p.get("sdpb_options"))
            if p.get("keep_checkpoint", False):
                solver.pop("--noFinalCheckpoint", None)
    elif task_name in _DEFAULT_LAM:
        def value(key, fallback=None):
            return p.get(key, profile.get(key, fallback))

        grid = task_name == "isolated_relevant_scalar_grid_point"
        intervals = task_name in {"isolated_relevant_scalar_delta_phi_intervals",
                                 "isolated_relevant_scalar_degeneracy_intervals"}
        modular = task_name == "modular_scalar_gap_bisection" or intervals
        if grid or intervals:
            values["modular_dps"] = p.get("modular_dps", value("dps"))
            if grid:
                values["four_point_dps"] = p.get("four_point_dps", value("dps"))
        else:
            values["dps"] = value("dps")
        four_point = not modular and (not grid or p.get("run_four_point", True))
        if four_point:
            scalar_only = task_name == "scalar_only_gap_probe"
            for key in ("q_order", "npoles"):
                values[key] = profile[key] if scalar_only else value(key)
            selectable = task_name in {"scalar_gap_bisection", "isolated_scalar_four_point_ope_bounds",
                                       "isolated_scalar_four_point_fixed_ope"}
            values["pole_selection"] = (p.get("pole_selection", selectors["pole_selection"])
                                        if selectable else selectors["pole_selection"])
            tower = (not scalar_only and p.get("four_point_large_spin" if task_name.startswith("isolated_")
                                               else "tower", True))
            if tower:
                values["large_spin_pole_selection"] = (
                    p.get("large_spin_pole_selection", selectors["large_spin_pole_selection"])
                    if task_name == "scalar_gap_bisection" else selectors["large_spin_pole_selection"])
                if task_name.startswith("isolated_"):
                    values["large_spin_npoles"] = profile["large_spin_npoles"]
                else:
                    values["large_spin_npoles"] = p.get("large_spin_npoles", profile["npoles"]
                        if profile["large_spin_npoles"] is None else profile["large_spin_npoles"])
                    values["large_spin_double_twists"] = p.get("large_spin_double_twists", 5
                        if profile["large_spin_double_twists"] is None else profile["large_spin_double_twists"])
    else:
        # Keep conservative profile checks for externally registered tasks.
        values["profile"] = profile
    for key in ("dps", "modular_dps", "four_point_dps", "q_order", "npoles",
                "large_spin_npoles", "large_spin_double_twists"):
        if key in values and values[key] is not None:
            values[key] = tasks._task_integer(key, values[key], minimum=0)
    values["sdpb_options"] = solver
    return _without_comments(values)


def execution_settings(task_name: str, params: dict) -> dict:
    """Freeze defaults for execution; compare only their effective values.

    The spec hash covers explicit parameters. Source snapshots let the task
    execute with exactly the defaults inspected here, even if the defaults
    file changes afterward. Code itself is not fingerprinted.
    """
    try:
        source = _snapshot(task_name, params)
        return {"schema": 3, "task": task_name, "params": deepcopy(params),
                "source": source, "effective": _effective(task_name, params, source)}
    except (KeyError, TypeError) as exc:
        raise ValueError(f"cannot resolve parameters for {task_name}: {exc}") from exc


def execution_matches(recorded, execution: dict | None, *, resolved_settings=None) -> bool:
    if execution is None:
        return True
    if execution["task"] in _MOMENT_TASKS and resolved_settings is not None:
        if not isinstance(resolved_settings, dict) or any(
            resolved_settings.get(key) != execution["effective"][key]
            for key in ("spec", "certification", "sdpb_options")
        ):
            return False
    if recorded is None:
        # Preserve adoption of work from before parameter recording existed.
        return True
    if not isinstance(recorded, dict):
        return False
    if recorded.get("schema") == 3:
        return (recorded.get("task") == execution["task"]
                and recorded.get("effective") == execution["effective"])
    try:
        source = _snapshot(execution["task"], execution["params"], recorded)
        return _effective(execution["task"], execution["params"], source) == execution["effective"]
    except (KeyError, TypeError, ValueError, AttributeError):
        # An incomplete or unknown record cannot establish a parameter match.
        return False


def settings_digest(settings: dict) -> str:
    if settings.get("schema") == 3:
        settings = {"task": settings["task"], "effective": settings["effective"]}
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
