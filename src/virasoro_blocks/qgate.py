"""q-order convergence gate for the Zamolodchikov h-recursion.

Low q_order silently corrupts blocks: the truncation tail of the q-expansion
grows with the derivative order and with central charge, so a q_order that is
adequate at one (c, Lambda) is garbage at another, and raising dps cannot fix
it. This module measures the actual truncation tail of the chiral coefficient
series and turns "silently corrupted" into a loud, quantified report.

Method
------
The H-recursion is triangular in the q-power k: ``H_k`` is independent of the
truncation order, so the difference of two evaluator builds at q_orders
``N`` and ``N - w`` is *exactly* the sum of the tail terms ``k in (N-w, N]``
(and likewise for ``N - w`` vs ``N - 2w``). For each Taylor-coefficient order
``i`` of the chiral series at a probe weight ``h``:

    S1_i = F_N,i - F_{N-w},i        (last window)
    S2_i = F_{N-w},i - F_{N-2w},i   (previous window)

the window-sum ratio ``x_i = |S1_i| / |S2_i|`` estimates the geometric decay
of the series, and the beyond-N tail is ``|S1_i| * x_i / (1 - x_i)``. If
``x_i >= 1`` the series is not yet in its geometric regime at this ``c`` (the
large-c trap, where H_k grows before decaying) and the block is reported
unconverged outright.

Each 2d product entry is built from chiral coefficients; for small relative
errors in nonzero factors, their relative errors add to first order. The
gate checks every chiral coefficient order and reports the worst estimate,
without assuming that the highest order is always worst. Extrapolating
finite tail windows at selected weights does not provide a rigorous bound
on the omitted series or on all exchanged weights. Parity folds can also
amplify errors through cancellation between product entries.

The three evaluator builds reuse the process-wide memoized
``_chiral_series_evaluator`` and the shared H-context cache, so the gate costs
a few evaluator calls once per (c, externals, q_order, precision) and is free
across the spin family.
"""

from __future__ import annotations

from copy import deepcopy
from functools import wraps

import sys

import mpmath as mp

__all__ = [
    "QConvergenceError",
    "q_tail_report",
    "point_q_tail_report",
    "super_point_q_tail_report",
    "super_q_tail_report",
    "gate_q_convergence",
    "DEFAULT_MIN_DIGITS",
]

# Default warn threshold, in decimal digits of estimated block accuracy.
# Production-validated configs sit at ~30-55 digits (q_order - Lambda of
# ~20-40 at ~1.4 digits/order at small c). A 20-digit threshold leaves a
# margin below those configurations.
DEFAULT_MIN_DIGITS = 20.0


class QConvergenceError(ValueError):
    """Raised by the strict gate when the q-expansion is inadequate."""


# Chiral evaluator factory per coordinate family (the parent single-chiral
# coordinate of each 2d / folded coordinate).
_COORD_FAMILY = {
    "zzb": "z", "xt": "z", "xt_radial": "z", "z": "z",
    "yyb": "y", "ws": "y", "ws_radial": "y", "y": "y",
    "qqb": "q", "qqb_ws": "q", "qqb_ws_radial": "q", "q": "q",
}

_REPORT_CACHE = {}
_REPORT_CACHE_MAXSIZE = 256

# Probe weights: small h is the slow-convergence end (H(h) -> 1 at large h),
# and the physical chiral arguments at the unitarity threshold are h in
# {0, ell}. Offsets dodge Kac weights; the evaluator is re-probed at a nudged
# point if a probe still lands on a pole.
_DEFAULT_PROBES = ("0.1371", "1.6371", "7.1371")
_PROBE_NUDGE = "0.0587"


def _scalar_key(v):
    from .derivs import _chiral_eval_memo_scalar_key

    return _chiral_eval_memo_scalar_key(v)


def _probe_central_charge(c, q_order, *, family=None):
    """Keep the supplied charge exact before probing.

    Singular charges use the merged-pole evaluator at that charge. Generic
    charges remain generic, including deliberately tiny shifts from a locus.
    Native mpmath charges use the block engine's working-precision recognition
    before their supplied bits are normalized to a literal label.
    """
    from .numeric_input import literal_number

    if isinstance(c, (mp.mpf, mp.mpc)):
        if family is None:
            from .singularity import exact_b_squared

            beta = exact_b_squared(c, q_order=q_order)
            if beta is not None:
                c = 13 + 6 * (beta + 1 / beta)
        else:
            from .super_laurent import exact_beta_squared, exact_c_of

            beta = exact_beta_squared(c)
            if beta is not None:
                c = exact_c_of(beta)
    if isinstance(c, float):
        from .singularity import snap_float_c

        c = snap_float_c(c)    # a float typed on a singular locus is that locus (0.7 is 7/10)
    c = literal_number(c)
    if isinstance(c, (complex, mp.mpc)) and c.imag == 0:
        c = literal_number(c.real)
    return c


def _exact_probe_inputs(function):
    """Keep point/table probe inputs exact until evaluation."""
    @wraps(function)
    def report(**kwargs):
        from .numeric_input import literal_number

        if isinstance(kwargs.get("c"), (mp.mpf, mp.mpc)) and "q_order" in kwargs:
            order = _gate_integer("q_order", kwargs["q_order"], 0)
            family = kwargs.get("family")
            if family is not None:
                from .superconformal_blocks import _resolve_family

                family = _resolve_family(family)
            kwargs["c"] = _probe_central_charge(kwargs["c"], order, family=family)
        for name in ("c", "h", "z", "center", "chiral_norm", "z_shift"):
            if name in kwargs:
                kwargs[name] = literal_number(kwargs[name])
        if "h_externals" in kwargs:
            kwargs["h_externals"] = tuple(literal_number(value) for value in kwargs["h_externals"])
        return function(**kwargs)
    return report


def _evaluator_factory(coordinate):
    from . import derivs as _derivs

    family = _COORD_FAMILY.get(coordinate)
    if family == "z":
        return _derivs.chiral_z_series_evaluator
    if family == "y":
        return _derivs.chiral_y_series_evaluator
    if family == "q":
        return _derivs.chiral_q_series_evaluator
    return None


def _finite_row(row):
    if row is None:
        return None
    out = []
    for v in row:
        z = mp.mpc(v)
        if not (mp.isfinite(z.real) and mp.isfinite(z.imag)):
            return None
        out.append(z)
    return out


def _rows_digits(F, G, Hrow, n_terms, noise_digits):
    """``(digits, window_ratio, unconverged)`` of a coefficient row from builds at q_orders N, N - w, N - 2w (None
    digits for an all-zero row); the worst coefficient order decides."""
    maxF = max((abs(v) for v in F), default=mp.mpf(0))
    if maxF == 0:
        if any(value != 0 for row in (G, Hrow) for value in row):
            # Cancellation at the final truncation does not establish
            # convergence; skipping it could hide the worst probe entirely.
            return mp.mpf(0), None, True
        return None, None, False
    floor = maxF * mp.mpf("1e-8")
    probe_digits, probe_x, unconverged = mp.inf, None, False
    for i in range(n_terms):
        scale = max(abs(F[i]), floor)
        s1 = abs(F[i] - G[i])
        s2 = abs(G[i] - Hrow[i])
        if s1 <= scale * mp.mpf(10) ** (-noise_digits):
            continue  # tail below the noise floor: converged at this i
        if s2 <= s1:
            unconverged = True
            d = -mp.log10(max(s1 / scale, mp.mpf("1e-300")))
            if d < probe_digits:
                probe_digits, probe_x = d, None
            continue
        x = s1 / s2
        tail = s1 * x / (1 - x)
        d = -mp.log10(max(tail / scale, mp.mpf("1e-300")))
        if d < probe_digits:
            probe_digits, probe_x = d, x
    return probe_digits, probe_x, unconverged


def _gate_integer(name, value, minimum):
    from .requests import _integer_field

    value = _integer_field(name, value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def q_tail_report(
    *,
    c,
    h_externals,
    q_order: int,
    n_terms: int,
    coordinate: str = "zzb",
    chiral_norm=None,
    z_shift=None,
    h_probes=None,
    window: int = 4,
    center=None,
):
    """Estimate the q-truncation accuracy of the chiral block series.

    ``center`` (a cross ratio z0, default the symmetric point 1/2) is the expansion point of the table; pass a
    sequence to probe several (an off-diagonal table's z0 and zbar0) and report the worst.
    A center sequence must be nonempty. ``h_probes`` accepts finite real
    labels under the exact-input policy; offsets are retained before evaluation.
    Genuine probe poles are nudged off and retried.

    Returns a dict:

    - ``status``: ``"ok"`` (geometric tail measured), ``"unconverged"``
      (window ratio >= 1 somewhere: the q-series has not entered its geometric
      regime at this ``c`` -- the block is not trustworthy at this q_order),
      or ``"unavailable"`` (unsupported coordinate: nothing measured; the
      gate returns the report without warning or raising).
    - ``digits``: estimated decimal digits of relative block accuracy (min
      over probes and coefficient orders); +inf when the tail is below the
      working-precision noise floor everywhere.
    - ``digits_per_order``: geometric decay rate (digits gained per unit of
      q_order), for sizing a fix.
    - ``suggested_q_order(target)`` is precomputed for the default threshold
      as ``suggested_q_order`` (None when already adequate or unmeasurable).
    - ``probe_c``: the exact central charge probed. Singular charges use
      merged poles at that charge; deliberate offsets are retained.

    Work starts at the caller's precision, raised when needed to resolve exact
    input labels. The chiral evaluators apply their numerical accuracy guards.
    """
    q_order = _gate_integer("q_order", q_order, 0)
    n_terms = _gate_integer("n_terms", n_terms, 1)
    window = _gate_integer("window", window, 1)
    h_externals = tuple(h_externals)
    probe_c = _probe_central_charge(c, q_order)
    if q_order < 6:
        # Too short to split into three windows: report unconverged rather
        # than guessing (any production run is far above this).
        return {
            "status": "unconverged", "digits": 0.0, "digits_per_order": None,
            "suggested_q_order": None, "q_order": q_order,
            "probe_c": str(probe_c), "reason": "q_order too small to probe",
        }
    window = max(2, min(window, (q_order - 2) // 2))

    factory = _evaluator_factory(coordinate)
    if factory is None:
        return {
            "status": "unavailable", "digits": None, "digits_per_order": None,
            "suggested_q_order": None, "q_order": q_order,
            "probe_c": str(probe_c),
            "reason": f"unsupported coordinate {coordinate!r}",
        }


    from .numeric_input import parse_number, real_input_label

    # Probe labels reach the evaluator before numerical conversion: a
    # written offset could otherwise round onto a genuine Kac pole here.
    h_probes = tuple(real_input_label(p, name='h_probes')
                     for p in (_DEFAULT_PROBES if h_probes is None else h_probes))

    centers = [None] if center is None else ([center] if not isinstance(center, (list, tuple)) else list(center))
    if not centers:
        raise ValueError('center must contain at least one probe point')
    if len(centers) > 1:
        reports = [q_tail_report(c=c, h_externals=h_externals, q_order=q_order, n_terms=n_terms,
                                 coordinate=coordinate, chiral_norm=chiral_norm, z_shift=z_shift, h_probes=h_probes,
                                 window=window, center=point) for point in centers]
        measured = [r for r in reports if r["status"] != "unavailable"] or reports
        return min(measured, key=lambda r: (r["status"] != "unconverged", r["digits"] if r["digits"] is not None
                                            else float("inf")))
    center = centers[0]
    if center is not None:
        from .numeric_input import literal_number

        center = literal_number(center)
        if center == literal_number(".5"):
            center = None
    cache_key = (
        _scalar_key(center),
        tuple(_scalar_key(hi) for hi in h_externals),
        n_terms, _scalar_key(probe_c), q_order,
        _scalar_key(chiral_norm), _scalar_key(z_shift),
        _COORD_FAMILY.get(coordinate), window,
        tuple(_scalar_key(p) for p in h_probes),
        mp.mp.prec,
    )
    hit = _REPORT_CACHE.get(cache_key)
    if hit is not None:
        return deepcopy(hit)

    if center is not None:
        from . import derivs as _derivs
        from .coordinates import COORDINATES

        chiral = COORDINATES[_COORD_FAMILY[coordinate]]

        def factory(ext, n, *, c, q_order, chiral_norm, z_shift):
            return _derivs._chiral_series_evaluator(
                ext, n, c=c, q_order=q_order, chiral_norm=chiral_norm, z_shift=z_shift,
                z_of_u=chiral.z_of_u, u_symmetric=chiral.u_of_z(mp.mpf(center)), coord_key=chiral.chiral_var,
            )
    from virasoro_blocks.singularity import merged_singular_evaluators

    with merged_singular_evaluators():
        evaluators = [
            factory(
                list(h_externals), n_terms, c=probe_c, q_order=q,
                chiral_norm=chiral_norm, z_shift=z_shift,
            )
            for q in (q_order, q_order - window, q_order - 2 * window)
        ]
    if any(ev is None for ev in evaluators):
        return {
            "status": "unavailable", "digits": None, "digits_per_order": None,
            "suggested_q_order": None, "q_order": q_order,
            "probe_c": str(probe_c), "reason": "evaluator unavailable",
        }

    noise_digits = mp.mp.dps - 6
    nudge = parse_number(_PROBE_NUDGE)
    from .requests import BlockPoleError

    worst_digits = mp.inf
    worst_x = None
    unconverged = False
    per_probe = []

    for h0 in h_probes:
        rows = None
        h = h0
        for _attempt in range(4):
            try:
                rows = [_finite_row(ev(h)) for ev in evaluators]
            except (BlockPoleError, ZeroDivisionError):
                rows = None
            if rows is not None and all(r is not None for r in rows):
                break
            h = h + nudge  # Kac-pole hit: nudge off and retry
            rows = None
        if rows is None:
            continue
        probe_digits, probe_x, probe_unconverged = _rows_digits(*rows, n_terms, noise_digits)
        if probe_digits is None:
            continue
        unconverged = unconverged or probe_unconverged
        per_probe.append({"h": str(h), "digits": float(probe_digits)})
        if probe_digits < worst_digits:
            worst_digits = probe_digits
            worst_x = probe_x

    report = _probes_report(per_probe, worst_digits, worst_x, unconverged, q_order=q_order, window=window,
                            probe_c=probe_c)

    if len(_REPORT_CACHE) >= _REPORT_CACHE_MAXSIZE:
        try:
            _REPORT_CACHE.pop(next(iter(_REPORT_CACHE)))
        except (StopIteration, KeyError):
            pass
    _REPORT_CACHE[cache_key] = deepcopy(report)
    return report


def _probes_report(per_probe, worst_digits, worst_x, unconverged, *, q_order, window, probe_c):
    """The report dict from the worst probe (see :func:`q_tail_report`)."""
    if not per_probe:
        return {"status": "unavailable", "digits": None, "digits_per_order": None, "suggested_q_order": None,
                "q_order": q_order, "probe_c": str(probe_c), "reason": "no valid probe points"}
    digits = float(worst_digits) if worst_digits != mp.inf else float("inf")
    # per-order decay rate from the window ratio: r = x^(1/w)
    dpo = float(-mp.log10(worst_x) / window) if worst_x is not None and worst_x > 0 else None
    suggested = None
    if unconverged:
        status = "unconverged"
        digits = min(digits, 0.0) if digits != float("inf") else 0.0
    else:
        status = "ok"
        if dpo is not None and dpo > 0 and digits < DEFAULT_MIN_DIGITS:
            suggested = q_order + int(mp.ceil((DEFAULT_MIN_DIGITS - digits) / dpo))
    return {"status": status, "digits": digits, "digits_per_order": dpo, "suggested_q_order": suggested,
            "q_order": q_order, "probe_c": str(probe_c), "per_probe": per_probe}


@_exact_probe_inputs
def super_q_tail_report(*, family, c, h_externals, q_order: int, n_terms: int, coordinate: str = "zzb",
                        sector="even", branch="++", center=None, chiral_norm=None, z_shift=None, window: int = 4):
    """:func:`q_tail_report` for NS/Ramond tables: the chiral Taylor coefficients at the table's center from the
    numeric super sampler at q_orders N, N - w, N - 2w (their tables are verified against the same-order sampler,
    which cannot see truncation: Ramond at order 10 recorded 53 verified digits for a 12-15 digit table)."""
    from .superconformal_blocks import (
        SuperconformalPoleError, _resolve_family, normalize_branch, normalize_sector, numeric_chiral_table,
    )

    family = _resolve_family(family)
    sector = normalize_sector(sector)
    branch = normalize_branch(branch)
    q_order = _gate_integer("q_order", q_order, 0)
    window = _gate_integer("window", window, 1)
    n_terms = _gate_integer("n_terms", n_terms, 1)
    if q_order < 6:
        return _short_order_report(q_order, c)
    window = max(2, min(window, (q_order - 2) // 2))
    chiral = _COORD_FAMILY.get(coordinate)
    if chiral is None:
        return {**_short_order_report(q_order, c), "status": "unavailable", "digits": None,
                "reason": f"unsupported coordinate {coordinate!r}"}
    noise_digits = mp.mp.dps - 6
    per_probe, worst_digits, worst_x, unconverged = [], mp.inf, None, False
    for h0 in (mp.mpf(p) for p in _DEFAULT_PROBES):
        rows = None
        h = h0
        for _attempt in range(4):
            try:
                rows = [_finite_row([row[0] for row in numeric_chiral_table(
                    family, h, chiral, list(h_externals), c=c, q_order=order, m_max=n_terms - 1,
                    chiral_norm=chiral_norm, z_shift=z_shift, sector=sector, branch=branch, center=center)])
                    for order in (q_order, q_order - window, q_order - 2 * window)]
            except SuperconformalPoleError:
                rows = None       # an exchanged-pole hit: nudge off it
            if rows is not None and all(r is not None for r in rows):
                break
            h, rows = h + mp.mpf(_PROBE_NUDGE), None
        if rows is None:
            continue
        probe_digits, probe_x, probe_unconverged = _rows_digits(*rows, n_terms, noise_digits)
        if probe_digits is None:
            continue
        unconverged = unconverged or probe_unconverged
        per_probe.append({"h": str(h), "digits": float(probe_digits)})
        if probe_digits < worst_digits:
            worst_digits, worst_x = probe_digits, probe_x
    return _probes_report(per_probe, worst_digits, worst_x, unconverged, q_order=q_order, window=window,
                          probe_c=c)


@_exact_probe_inputs
def point_q_tail_report(*, h, z, h_externals, c, q_order: int, window: int = 4):
    """The q-truncation accuracy of one chiral block value ``F(h, z)`` (what ``evaluate`` returns).

    The same window-ratio estimate as :func:`q_tail_report`, applied to the terms ``H_k q(z)^k`` of the
    H-series at the evaluation point itself (the recursion is triangular in ``k``, so the windows are exact tail
    sums). At large ``c`` the terms grow before they decay and the order needed grows with ``c`` (c = 1e5 needs
    q_order > 80 at z = 0.4, and order 30 gave 1e-10 for 0.03 at every precision): that is ``"unconverged"``.
    Returns the :func:`q_tail_report` keys (``status``, ``digits``, ``digits_per_order``, ``suggested_q_order``).
    """
    from .recursion import H_q_series, q_from_z

    q_order = _gate_integer("q_order", q_order, 0)
    window = _gate_integer("window", window, 1)
    if q_order < 6:
        return _short_order_report(q_order, c)
    window = max(2, min(window, (q_order - 2) // 2))
    probe_c = _probe_central_charge(c, q_order)
    q = q_from_z(mp.convert(z))
    from .singularity import recognized_singular_locus

    at_locus = recognized_singular_locus(probe_c, q_order=q_order) is not None

    def series(weight):
        if at_locus:
            # exactly at the singular c: the merged-pole H of the one Laurent table (on a Kac weight the block
            # kac_limit_mode selects, the irreducible module's by default)
            from .polynomial import _acb_to_mpc
            from .recursion import build_merged_H_context, eval_merged_H

            ctx = build_merged_H_context(q_order, probe_c, list(h_externals))
            values = eval_merged_H(weight, ctx, q_order, complex_out=True)
            return None if values is None else [_acb_to_mpc(v) for v in values]   # None: on a pole, nudged
        # at large c or near a locus the recursion cancels (c = 1000, order 60: the high terms were 1e-10
        # garbage once); the coefficients are certified from their own balls, with the Kac data formed from c
        return H_q_series(weight, list(h_externals), c=probe_c, q_order=q_order)

    coeffs = _finite_coefficients(series, h)
    if coeffs is None:
        return {**_short_order_report(q_order, c), "status": "unavailable", "digits": None,
                "reason": "no finite probe"}
    terms = [coeffs[k] * q ** k for k in range(q_order + 1)]
    return _window_report(terms, window, q_order=q_order, per_order=1, probe_c=probe_c)


@_exact_probe_inputs
def super_point_q_tail_report(*, family, h, z, h_externals, c, q_order: int, sector, branch=None, window: int = 4):
    """:func:`point_q_tail_report` for an NS or Ramond chiral block, whose H-series runs in ``t = q^(1/2)``.

    Their order needed grows with c too (NS at c = 3e4, order 20 gave 0.0034 for 0.0319).  At an N=1 singular c
    the series is that of c itself (the exact Laurent recursion).
    """
    from .recursion import q_from_z
    from .recursion_ns import _resolve_ns_order
    from .super_laurent import SuperLaurentPoleError
    from .superconformal_blocks import FAMILY_NS, _resolve_family, normalize_branch, normalize_sector

    family = _resolve_family(family)
    sector = normalize_sector(sector)
    branch = None if branch is None else normalize_branch(branch)

    q_order = _gate_integer("q_order", q_order, 0)
    window = _gate_integer("window", window, 1)
    if q_order < 6:
        return _short_order_report(q_order, c)
    window = max(2, min(window, (q_order - 2) // 2))
    # at an N=1 singular c the series is the exact one of c itself (super_laurent, verified by its own balls), and
    # elsewhere the certified recursion: no offset and no guard
    probe_c = c
    half_order = _resolve_ns_order(None, q_order)
    if family == FAMILY_NS:
        from .recursion_ns import H_q_series_ns

        def evaluate(weight):
            return H_q_series_ns(weight, list(h_externals), c=probe_c, ns_order=half_order, sector=sector)
    else:
        from .recursion_r import H_q_series_r

        def evaluate(weight):
            return H_q_series_r(weight, list(h_externals), c=probe_c, r_order=half_order, sector=sector,
                                branch=branch)

    def series(weight):
        try:
            return evaluate(weight)
        except SuperLaurentPoleError:
            return None                     # (on a genuine pole of the singular c: nudged off it)

    coeffs = _finite_coefficients(series, h)
    if coeffs is None:
        return {**_short_order_report(q_order, c), "status": "unavailable", "digits": None,
                "reason": "no finite probe"}
    t = mp.sqrt(q_from_z(mp.convert(z)))
    terms = [coeffs[k] * t ** k for k in range(len(coeffs))]
    return _window_report(terms, 2 * window, q_order=q_order, per_order=2, probe_c=probe_c)


def _short_order_report(q_order, c=None):
    return {"q_order": q_order, "digits_per_order": None, "suggested_q_order": None, "status": "unconverged",
            "digits": 0.0, "reason": "q_order too small to probe", "probe_c": str(c)}


def _finite_coefficients(series, h):
    """``series(h)`` as finite numbers, nudging ``h`` off an exact Kac pole (the tail is continuous in h)."""
    for _attempt in range(4):
        coeffs = _finite_row(series(h))
        if coeffs is not None:
            return coeffs
        h += mp.mpf(10) ** (-(mp.mp.dps // 3))
    return None


def _window_report(terms, window, *, q_order, per_order, probe_c):
    """The report from the last two windows of ``terms`` (``per_order`` terms per unit of q_order)."""
    n = len(terms)
    total = abs(mp.fsum(terms))
    s1 = abs(mp.fsum(terms[n - window:]))
    s2 = abs(mp.fsum(terms[n - 2 * window:n - window]))
    base = {"q_order": q_order, "digits_per_order": None, "suggested_q_order": None, "probe_c": str(probe_c)}
    # The noise floor is set by the largest term, not the sum: at large c the terms grow and cancel, and their
    # rounding (the recursion's own) left the tail windows at noise level, read as "unconverged" at orders 80-240 of
    # a converged c = 1000 value.
    largest = max(abs(term) for term in terms)
    if total == 0 and largest != 0:
        return {**base, "status": "unconverged", "digits": 0.0,
                "reason": "zero truncated sum with nonzero terms; relative tail error cannot be estimated"}
    noise = largest * mp.mpf(10) ** (-(mp.mp.dps - 6))
    if total == 0 or s1 <= max(noise, total * mp.mpf(10) ** (-(mp.mp.dps - 6))):
        return {**base, "status": "ok", "digits": float("inf")}
    if s2 <= s1 and s2 > noise:
        return {**base, "status": "unconverged", "digits": 0.0}
    if s2 <= noise:
        # the previous window is noise too: the tail is at the noise floor
        return {**base, "status": "ok", "digits": float(-mp.log10(max(noise, s1) / total))}
    x = s1 / s2
    digits = float(-mp.log10(max(s1 * x / (1 - x) / total, mp.mpf("1e-300"))))
    dpo = float(-mp.log10(x) * per_order / window) if x > 0 else None
    suggested = (q_order + int(mp.ceil((DEFAULT_MIN_DIGITS - digits) / dpo))
                 if dpo and dpo > 0 and digits < DEFAULT_MIN_DIGITS else None)
    return {**base, "status": "ok", "digits": digits, "digits_per_order": dpo, "suggested_q_order": suggested}


def _q_gate_threshold(value):
    try:
        threshold = float(value)
        valid = not isinstance(value, bool) and mp.isfinite(threshold)
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"min_digits must be a finite real number, got {value!r}")
    return threshold


def act_on_q_report(report, *, mode: str = "warn", min_digits: float = DEFAULT_MIN_DIGITS, label: str = "",
                    stream=None):
    """Warn (``mode="warn"``) or raise :class:`QConvergenceError` (``"strict"``) on a bad q-tail report."""
    if mode == "off":
        return
    if mode not in ("warn", "strict"):
        raise ValueError(f"q-gate mode must be off/warn/strict; got {mode!r}")
    threshold = _q_gate_threshold(min_digits)
    if report is None or report.get("status") == "unavailable":
        return
    if report.get("status") not in ("ok", "unconverged"):
        raise ValueError(f"invalid q-gate report status {report.get('status')!r}")
    digits = report.get("digits")
    if report["status"] == "ok" or digits is not None:
        try:
            measured = float(digits)
            valid = not isinstance(digits, bool) and not mp.isnan(measured)
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError(f"invalid q-gate report digits {digits!r}")
        digits = measured
    bad = report["status"] == "unconverged" or digits < threshold
    if bad:
        message = _format_report({**report, "digits": digits}, label)
        if mode == "strict":
            raise QConvergenceError(message)
        print(f"warning: {message}", file=stream or sys.stderr)


def _format_report(report, label):
    where = f" [{label}]" if label else ""
    if report["status"] == "unconverged":
        if report.get("reason"):
            return (f"q-order gate{where}: {report['reason']} at q_order={report['q_order']} "
                    f"(c={report.get('probe_c')}). Raise --order substantially.")
        return (
            f"q-order gate{where}: the q-expansion has NOT entered its "
            f"geometric regime at q_order={report['q_order']} "
            f"(c={report.get('probe_c')}); the block values are untrustworthy. "
            "Raise --order substantially."
        )
    digits = report["digits"]
    tail = (
        f"estimated block accuracy ~{digits:.1f} digits at "
        f"q_order={report['q_order']}"
    )
    if report.get("digits_per_order"):
        tail += f" (~{report['digits_per_order']:.2f} digits/order)"
    if report.get("suggested_q_order"):
        tail += f"; suggest q_order >= {report['suggested_q_order']}"
    return f"q-order gate{where}: {tail}"


def gate_q_convergence(
    *,
    c,
    h_externals,
    q_order: int,
    n_terms: int,
    coordinate: str = "zzb",
    chiral_norm=None,
    z_shift=None,
    h_probes=None,
    center=None,
    mode: str = "warn",
    min_digits: float = DEFAULT_MIN_DIGITS,
    label: str = "",
    stream=None,
):
    """Run the q-order gate and warn or raise per ``mode``.

    ``mode``: ``"off"`` (skip entirely), ``"warn"`` (print a one-line stderr
    warning when the estimated accuracy is below ``min_digits`` or the series
    is unconverged), ``"strict"`` (raise :class:`QConvergenceError` instead).
    Returns the report dict (or None when ``mode="off"``). An
    ``"unavailable"`` report (unsupported coordinate) never warns or
    raises -- there is nothing measured to act on.
    """
    if mode == "off":
        return None
    if mode not in ("warn", "strict"):
        raise ValueError(f"q-gate mode must be off/warn/strict; got {mode!r}")
    min_digits = _q_gate_threshold(min_digits)
    report = q_tail_report(
        c=c, h_externals=h_externals, q_order=q_order, n_terms=n_terms,
        coordinate=coordinate, chiral_norm=chiral_norm, z_shift=z_shift,
        h_probes=h_probes, center=center,
    )
    act_on_q_report(report, mode=mode, min_digits=min_digits, label=label, stream=stream)
    return report
