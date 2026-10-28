"""Decide functional positivity exactly, by root isolation rather than sampling.

The SDP constraint on a functional is not "``alpha . F(Delta) >= 0`` at the
points we happened to evaluate".  It is

    q(x) >= 0   for ALL x in [0, infinity)

with ``q`` a polynomial.  Sampling cannot decide that, and the gap is not
academic: at c=8/7, Delta_ext=8/5, Lambda=11 the npoles=16 functional dips
negative on the npoles=32 blocks over an interval only ~0.004 wide (ell=0, first
root at x = 1.2789), and in ell=2/ell=4 at x < 5.7e-8 and < 4.9e-10.  It was
missed for a full day because the direct-SDP certification path evaluated only at
the SDP's own ~44 sample points -- precisely the points the degree-43 polynomial
was fitted through, where it is positive by construction.

:class:`~virasoro_blocks.serialization.SerializedDerivTable` stores a block as

    base**x * N(x) / prod_j (x - pole_j),        x = Delta - shift

so with ``base**x > 0`` and the pole product positive on ``x >= 0``, positivity
of ``alpha . F`` is equivalent to positivity of the scalar polynomial

    q(x) = sum_j y_j * N_j(x)

whose coefficients the PMP doc carries explicitly.  Isolate the positive real
roots of ``q`` and read its sign on each resulting interval, plus the leading
coefficient for ``x -> infinity``.  That is a decision procedure.

The pole-product hypothesis is CHECKED, not assumed (:func:`real_poles_are_negative`):
the DampedRational lists only the real poles -- complex pairs are dropped from it
by :mod:`.pmp_writer` -- but each dropped pair contributes ``|x - p|^2 > 0`` and
so cannot flip the sign.
"""

from __future__ import annotations

import mpmath as mp

__all__ = [
    "combine_polynomial",
    "positive_real_roots",
    "real_poles_are_negative",
    "block_positivity",
]


def _exact_real(value):
    """The supplied real number, without working-precision rounding."""
    from fractions import Fraction
    from flint import fmpq
    from virasoro_blocks.numeric_input import literal_number

    if isinstance(value, (complex, mp.mpc)):
        if value.imag != 0:
            raise ValueError("polynomial certification requires real numbers")
        value = value.real
    if isinstance(value, mp.mpf):
        # Native binary data need no intermediate Fraction allocation. This
        # is also independent of caller precision, including stored guard bits.
        sign, mantissa, exponent, _ = value._mpf_
        if not mantissa and exponent:
            raise ValueError("numeric inputs must be finite")
        numerator = -int(mantissa) if sign else int(mantissa)
        return (fmpq(numerator << exponent) if exponent >= 0
                else fmpq(numerator, 1 << -exponent))
    value = literal_number(value)
    if not isinstance(value, Fraction):
        value = literal_number(mp.mpf(value))
    return fmpq(value.numerator, value.denominator)


def _combined_exact(polys, y):
    from flint import fmpq_poly

    if len(polys) != len(y):
        raise ValueError("polynomial and functional dimensions disagree")
    result = fmpq_poly([])
    for coefficients, weight in zip(polys, y):
        result += fmpq_poly([_exact_real(c) for c in coefficients]) * _exact_real(weight)
    return result


def _as_mpf(value):
    return mp.mpf(int(value.p)) / int(value.q)


def combine_polynomial(polys, y):
    """Combine coefficients exactly, then round the result to caller precision."""
    return [_as_mpf(c) for c in _combined_exact(polys, y).coeffs()] or [mp.mpf(0)]


def _root_controls(dps=None, maxsteps=None):
    from virasoro_blocks.requests import _integer_field

    work = _integer_field("dps", mp.mp.dps if dps is None else dps)
    if work < 1:
        raise ValueError("dps must be positive")
    attempts = 6 if maxsteps is None else _integer_field("maxsteps", maxsteps)
    if attempts < 1:
        raise ValueError("maxsteps must be positive")
    return max(30, work), min(6, attempts)


def _certified_intervals(polynomial, dps, maxsteps=None):
    """Disjoint positive-real root enclosures of an exact rational polynomial."""
    from flint import ctx

    work, attempts = _root_controls(dps, maxsteps)
    for _ in range(attempts):
        with ctx.workdps(work):
            roots = polynomial.complex_roots()
            if sum(multiplicity for _, multiplicity in roots) != polynomial.degree():
                raise RuntimeError("root isolation returned an incomplete root set")
            positive = []
            unresolved = False
            for root, multiplicity in roots:
                if not root.imag.is_zero():
                    unresolved |= root.imag.contains(0)
                    continue
                real = root.real
                if real.is_zero() or real < 0:
                    continue
                if not real > 0:
                    unresolved = True
                    continue
                positive.append((real.lower().fmpq(), real.upper().fmpq(), multiplicity))
            positive.sort(key=lambda item: item[0])
            if not unresolved and all(a[1] < b[0] for a, b in zip(positive, positive[1:])):
                return positive, work
        work *= 2
    raise RuntimeError("real-root intervals could not be separated")


def _evaluate(coeffs, x):
    acc = mp.mpf(0)
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc


def positive_real_roots(coeffs, *, dps=None, maxsteps=None):
    """Positive real roots, with multiplicity, from certified root enclosures.

    The coefficients are retained exactly. ``dps`` controls the initial root
    enclosure precision; ambiguous enclosures trigger precision doubling.
    ``maxsteps`` optionally caps these refinement attempts (at most six).
    Failure raises instead of returning an incomplete root list.
    """
    work, attempts = _root_controls(dps, maxsteps)
    q = _combined_exact([coeffs], [1])
    if q.degree() <= 0:
        return []
    intervals, work = _certified_intervals(q, work, attempts)
    with mp.workdps(work):
        return [_as_mpf((left + right) / 2)
                for left, right, multiplicity in intervals for _ in range(multiplicity)]


def _prefactor_number(value):
    """Accept exact real labels and retain nonfinite diagnostics for refusal."""
    from virasoro_blocks.numeric_input import real_input_label

    try:
        return mp.convert(real_input_label(value, name="PMP prefactor"))
    except ValueError:
        # Nonfinite metadata is reported as an undecided check, with the pole
        # value retained. Other invalid inputs must not bypass label validation.
        number = mp.mpf(value)
        if mp.isfinite(number):
            raise
        return number


def real_poles_are_negative(entry):
    """Poles breaking numerator positivity on the open ray: positive or non-finite.

    The historical name is retained. A finite pole at zero is allowed by the
    PMP writer: its factor x is positive for x > 0, with any multiplicity.
    The numerator's continuity also detects a negative limit at the boundary.
    """
    prefactor = entry.get("DampedRational")
    if prefactor is None:
        prefactor = {}
    if not isinstance(prefactor, dict):
        raise ValueError("DampedRational must be an object")
    pole_values = prefactor.get("poles", [])
    if not isinstance(pole_values, (list, tuple)):
        raise ValueError("DampedRational poles must be an array")
    poles = [_prefactor_number(p) for p in pole_values]
    return tuple(p for p in poles if not mp.isfinite(p) or p > 0)


def block_positivity(entry, y, *, dps=None, shift=None):
    """Decide positivity of one PMP block under functional ``y``.

    Returns a dict with ``passed``, ``degree``, ``leading``, ``roots``, and --
    when negative -- ``first_negative_x``/``first_negative_value`` and the
    corresponding ``first_negative_delta`` if ``shift`` is supplied (recall
    ``x = Delta - shift``, so a bare ``x`` is not a physical dimension).

    ``passed`` is None with a ``reason`` when the block cannot be decided (a
    non-positive/non-finite base or constant, a positive/non-finite pole,
    a length mismatch, or a non-finite functional or combined polynomial), so a caller can never mistake "not
    checked" for "checked and fine".
    """
    dps, _ = _root_controls(dps)
    prefactor = entry.get("DampedRational")
    if prefactor is None:
        prefactor = {}
    if not isinstance(prefactor, dict):
        return {"passed": None, "reason": "invalid_damped_rational_prefactor"}
    for name in ("base", "constant"):
        try:
            value = _prefactor_number(prefactor.get(name, "1"))
        except (TypeError, ValueError, ArithmeticError):
            return {"passed": None, "reason": f"damped_rational_{name}_not_positive"}
        if not mp.isfinite(value) or value <= 0:
            return {"passed": None, "reason": f"damped_rational_{name}_not_positive"}
    try:
        bad = real_poles_are_negative(entry)
    except (TypeError, ValueError, ArithmeticError):
        return {"passed": None, "reason": "invalid_damped_rational_poles"}
    if bad:
        return {
            "passed": None,
            "reason": "damped_rational_pole_not_negative",
            "poles": bad,
        }
    matrix = entry["polynomials"]
    if len(matrix) != 1 or len(matrix[0]) != 1:
        return {"passed": None, "reason": "non_scalar_polynomial_matrix"}
    polys = matrix[0][0]
    # ``v < 0`` is False for NaN, so a NaN (or inf) functional would sail through
    # every sign test below and report a pass.  It certifies nothing.
    if not all(mp.isfinite(yj) for yj in y):
        return {
            "passed": None,
            "reason": "non_finite_functional",
            "y_length": len(y),
        }
    if len(polys) != len(y):
        return {
            "passed": None,
            "reason": "dimension_mismatch",
            "n_polynomials": len(polys),
            "y_length": len(y),
        }
    try:
        q = _combined_exact(polys, y)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError):
        return {"passed": None, "reason": "non_finite_or_non_real_polynomial"}
    if not q:
        return {"passed": True, "degree": 0, "leading": mp.mpf(0),
                "roots": (), "n_positive_roots": 0}
    try:
        intervals, work = _certified_intervals(q, dps) if q.degree() > 0 else ([], dps)
    except (ValueError, RuntimeError) as exc:
        return {"passed": None, "reason": "root_isolation_failed", "error": str(exc)}

    # Every interval between distinct real roots has constant sign. Probe
    # rational points outside the certified root enclosures and evaluate the
    # exact polynomial there; no floating-point sign decision is involved.
    from flint import fmpq

    probes = [fmpq(0)]
    lower = fmpq(0)
    for left, right, _ in intervals:
        probes.append((lower + left) / 2)
        lower = right
    probes.append(lower + 1)
    negative = [(x, q(x)) for x in probes if q(x) < 0]
    with mp.workdps(work):
        roots = tuple(_as_mpf((left + right) / 2)
                      for left, right, multiplicity in intervals for _ in range(multiplicity))
        detail = {"passed": not negative, "degree": q.degree(),
                  "leading": _as_mpf(q[q.degree()]), "roots": roots,
                  "n_positive_roots": len(roots)}
        if negative:
            x, value = negative[0]
            detail["first_negative_x"] = _as_mpf(x)
            detail["first_negative_value"] = _as_mpf(value)
            if shift is not None:
                detail["first_negative_delta"] = _as_mpf(x + _exact_real(shift))
    return detail
