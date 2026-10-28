"""Library-wide precision controls.

`mp.mp.dps` is the single knob that governs the decimal precision of every
mpmath operation in the library. The helpers here let internal tolerances and
iteration caps scale off it instead of being hardcoded at fixed values.

Conventions:
- `set_precision(dps)` sets `mp.mp.dps`. It exists for symmetry with the rest
  of the library API; it is equivalent to writing `mp.mp.dps = dps` directly.
- `tolerance(guard=g)` returns `10 ** -(dps - g)`. The `guard` is the number
  of digits sacrificed to accumulated roundoff; pick it for the call site:
    * `guard ~ 5`     for input-validation checks ("this value should be
      exact real / a small integer"), where the value has not been routed
      through any mpmath arithmetic.
    * `guard = 10`    (default) for typical "did these two computed values
      agree" checks. Most internal sites use this.
    * `guard = 15-20` when the values being compared sit several Polynomial
      evaluations or transcendental-function calls downstream of the inputs.
- `iteration_cap(base=b, per_dps=k)` returns `b + k * dps`, useful as a hard
  iteration limit for bisection-style loops whose convergence count scales
  with precision.
- Internal domain and real-output checks use `_validation_tolerance`, which
  caps the allowance at 1e-5 even when the caller lowers precision.

These helpers read `mp.mp.dps` lazily, so changing the precision after a
function has been called still affects subsequent calls. `q_order` is a
separate concern: it sets the Zamolodchikov recursion truncation order, which
is governed by the q-radius (q(1/2) ~ 0.043), not by digits of precision.
"""

from __future__ import annotations

import sys

import mpmath as mp

from .numeric_input import _integer_control

# High-precision mpmath routinely converts very large integers to/from strings
# (e.g. the flint q-series in `recursion._q_from_z_flint` renders a mantissa
# with ~one digit per bit of precision). Python 3.11+ caps int<->str at 4300
# digits by default, which a regularization that escalates to a few thousand
# dps would trip. Raise the cap once at import to a value far beyond any
# physical precision so the library never hits the limit mid-computation.
if hasattr(sys, "set_int_max_str_digits"):
    _int_string_limit = sys.get_int_max_str_digits()
    # Zero means unlimited; replacing it with a finite cap would lower the
    # caller's explicitly chosen limit rather than making high precision safer.
    if 0 < _int_string_limit < 1_000_000:
        sys.set_int_max_str_digits(1_000_000)


def set_precision(dps: int) -> None:
    """Set the global mpmath decimal precision."""
    mp.mp.dps = _integer_control('dps', dps, 5)


def tolerance(*, guard: int = 10) -> mp.mpf:
    """Return a tolerance scaled to the current `mp.mp.dps`.

    The result is `10 ** -(dps - guard)`. Tight guards reflect "the inputs
    should be exact"; looser guards reflect accumulated roundoff.
    """
    guard = _integer_control('guard', guard)
    return mp.mpf(10) ** (-(mp.mp.dps - guard))


_VALIDATION_TOLERANCES: dict[tuple[int, int], mp.mpf] = {}


def _validation_tolerance(*, guard=10):
    """Roundoff allowance capped at 1e-5, including at low precision.

    Preserve the usual guard at higher precision. Its uncapped form can
    exceed one at low precision and admit material domain violations.
    Cache by binary precision: real-output checks can run millions of times.
    """
    key = (mp.mp.prec, guard)
    value = _VALIDATION_TOLERANCES.get(key)
    if value is None:
        value = mp.mpf(10) ** -max(5, mp.mp.dps - guard)
        _VALIDATION_TOLERANCES[key] = value
    return value


def iteration_cap(*, base: int = 30, per_dps: int = 4) -> int:
    """Iteration cap for bisection-style loops, scaled by `mp.mp.dps`.

    Bisection needs `log2(10) ~ 3.32` iterations per decimal digit, so a
    linear scaling with `dps` plus a small `base` constant covers the
    typical convergence horizon with headroom.
    """
    base = _integer_control('base', base)
    per_dps = _integer_control('per_dps', per_dps)
    return base + per_dps * mp.mp.dps


def convert_exact(value):
    """Convert exact input to mpmath at the current working precision.

    Rationals such as ``7/10`` and ``3/80`` have no finite binary expansion.
    Computing ``mp.mpf(7)/10`` rounds at the active precision; increasing
    ``dps`` later cannot recover the lost digits. Pass ``Fraction(7, 10)``
    to defer rounding until conversion.

    This wrapper around :func:`mpmath.convert` accepts ``Fraction``, ``str``,
    ``int``, and mpmath types. It preserves complex values, which are needed
    when Taylor coefficients are extracted by Cauchy integration.
    """
    return mp.convert(value)


def taylor_coefficients(f, x, n: int, **options) -> list:
    """Taylor coefficients ``f^(i)(x) / i!``, ``i = 0..n``: :func:`mpmath.taylor` without its default chop.

    ``mp.taylor`` chops every derivative with an ABSOLUTE ``100 eps`` cut, so a block of large weight (``F ~
    0.04^h``) has every coefficient set to exactly zero once ``h`` exceeds ~0.7 dps, and the imaginary part of
    any complex coefficient below ``100 eps`` is dropped however large it is relative to the value.  Here only
    an imaginary part negligible RELATIVE to the value is dropped (a real function at a real point stays
    real); nothing is set to zero.
    """
    tol = min(100 * mp.eps, _validation_tolerance(guard=5))
    out = []
    for i, d in enumerate(mp.diffs(f, x, n, **options)):
        v = d / mp.factorial(i)
        if isinstance(v, mp.mpc) and abs(v.imag) <= tol * abs(v):
            v = v.real
        out.append(v)
    return out
