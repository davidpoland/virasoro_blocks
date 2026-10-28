"""Numerical (u, ubar)-derivative tables for B and G blocks at the
crossing-symmetric point u = ubar = u_*.

This module handles the non-folded coordinate options:

  zzb  - (z, zbar), u_* = 1/2
  yyb  - (y, ybar), y = (2z - 1)/(1 + 2 sqrt(z(1-z))),   u_* = 0
         (the blocks_3d coordinate, z = (1+y)^2/(2(1+y^2)); crossing is y -> -y.
         This is NOT the radial coordinate rho = (1 - sqrt(1-z))/(1 + sqrt(1-z)).)
  qqb  - (q, qbar), q = exp(-pi K(1-z)/K(z)),           u_* = exp(-pi)

Each coordinate is built from a chiral transform u(z); the table entry

    table[m][n] = (1 / m! n!) * (d/du)^m (d/dubar)^n B(u, ubar) at u = ubar = u_*

so locally  B = sum_{m, n} table[m][n] (u - u_*)^m (ubar - u_*)^n.

B factorizes as F(h, z; {h_i}) * F(hbar, zbar; {hbar_i}), so each table is the
outer product of two chiral Taylor series of F at u_*. Since the chiral
transform z(u) is one-variable, the chain rule from (z, zbar) to (u, ubar) is
applied independently on each side; we obtain the u-Taylor coefficients of
F(h, z(u)) directly via `precision.taylor_coefficients` (mpmath's numerical
differentiation, without `mp.taylor`'s absolute chop) at u_*, bypassing manual series
composition.

The (Delta, ell) variant G adds the parity-symmetrized B(hbar, h) term and is
the right object for scalar externals.

Performance note: each Taylor coefficient still samples the block on mpmath's
contour, but the fixed-h q-recursion is now built once per chiral Taylor
series and reused across those z-samples.
"""

from __future__ import annotations

from typing import Callable, Sequence
from contextvars import ContextVar
from functools import lru_cache

import mpmath as mp
from .precision import taylor_coefficients, _validation_tolerance

from .coordinates import (
    COORDINATES,
    q_symmetric,
    y_symmetric,
    z_of_q,
    z_of_y,
    z_symmetric,
)
from .numeric_input import ExactRational, _exchange_weight, _integer_control, literal_number, _signed_half_integer as _check_ell_signed, _integer_spin as _check_ell
from .params import b_from_c, c_from_b, c_origin_of_b, _c_input_of_b
from .singularity import bare_singular_locus, reject_singular_c
from .parity import parity_reversal_phase
from ._conventions import scalar_G_fold_parity, scalar_G_swap_coefficient
from .polynomial import (
    Polynomial,
    _acb_to_mpc,
    _arb_to_mpf,
    _mpf_to_acb,
    _mpf_to_arb,
)
from .recursion import (
    H_q_series,
    _build_H_context,
    _eval_H_at_acb_nonstrip_complex,
    _eval_H_at_acb_strip_complex,
    _eval_H_at_acb_strip_real,
    _eval_H_at_arb_real,
    _fixed_h_block_evaluator,
    _kac_input_values,
    _maybe_real_scalar,
    _resolve_chiral_norm,
    _theta3,
    q_from_z,
)


_FOLD_PLAN_CACHE: dict[tuple[int, int, int, int, int], tuple[list, int]] = {}


def _resolve_b(b, c):
    """Resolve the Liouville b.

    (The singular-c guard sits where the recursion is built -- `_chiral_taylor`
    regularizes, `_chiral_series_evaluator` fails closed -- since only there are
    the externals known that decide whether a singular term is live.)
    """
    if (b is None) == (c is None):
        raise ValueError("specify exactly one of b, c")
    if b is not None:
        origin = c_origin_of_b(b)
        if origin is None:
            return b
        c = origin
    from .singularity import snap_float_c

    # b is formed with extra digits: at large c the block is so sensitive to b that its rounding at the working
    # precision alone cost ~9 digits at c = 1e5 (a guarded evaluation uses the extra digits, an unguarded one rounds
    # them away); a Fraction has no mpc(); a float 0.7 means 7/10
    c = snap_float_c(c)
    with mp.workdps(2 * mp.mp.dps):
        return b_from_c(c)


def _check_externals(h_externals: Sequence) -> None:
    if len(h_externals) != 4:
        raise ValueError("h_externals must be a length-4 sequence")


def _check_table_orders(m_max: int, n_max: int) -> tuple[int, int]:
    return _integer_control('m_max', m_max), _integer_control('n_max', n_max)


@lru_cache(maxsize=32)
def _rational_kac_weights(beta, q_order):
    """Exact Kac locations at rational b², shared across weights and externals."""
    return frozenset(
        ((1-r*r)*beta + 2*(1-r*s) + (1-s*s)/beta)/4
        for r in range(1,q_order+1) for s in range(1,q_order//r+1)
    )


def _chiral_taylor(
    h,
    h_externals: Sequence,
    n_terms: int,
    *,
    b=None,
    c=None,
    q_order: int,
    z_of_u: Callable,
    u_symmetric,
    chiral_norm=None,
    z_shift=None,
):
    """Taylor coefficients of F(h, z(u); {h_i}; c) around u = u_symmetric.

    At a recognized recursion-singular c the exact-c Laurent regularizer
    handles structural recursion singularities and exact exchanged Kac
    weights, including those below the first structural singularity. Every
    per-h table builder on top of this follows the same pole policy as F_block.
    """
    n_terms = _integer_control('n_terms', n_terms, 1)
    q_order = _integer_control('q_order', q_order)
    _check_externals(h_externals)
    from .numeric_input import literal_number, parse_number
    from .singularity import exact_b_squared, exact_locus_c
    from .recursion import _kac_weights
    import math

    h = _coerce_delta_scalar(h)
    h_externals = tuple(literal_number(value) for value in h_externals)
    if c is None and b is not None:
        origin = c_origin_of_b(b)
        if origin is not None:
            b, c = None, origin
    c = parse_number(c)
    # Freeze the low-level mpmath locus convention before raising precision.
    if c is not None:
        locus = exact_locus_c(c, q_order=q_order)
        if locus is not None:
            c = parse_number(locus)
    resolved_b = _resolve_b(b, c)
    on_kac = False
    try:
        h64 = complex(h)
    except OverflowError:
        h64 = complex(math.inf)
    if math.isfinite(h64.real) and math.isfinite(h64.imag):
        _, poles = _kac_weights(resolved_b, q_order)
        if any(abs(pole-h64) <= 1e-8*max(1, abs(h64), abs(pole)) for pole in poles):
            # The exact label decides whether h sits on a Kac weight.
            from fractions import Fraction
            beta = exact_b_squared(c if c is not None else c_from_b(resolved_b), q_order=q_order)
            on_kac = isinstance(h, Fraction) and (
                h == 0 or beta is not None and h in _rational_kac_weights(beta, int(q_order))
            )
    b = resolved_b
    # structural check (externals-blind): a singular term whose residue these externals kill is
    # harmless in exact arithmetic but can still divide 0/0 in the bare build, so any singular term
    # up to q_order routes to the (always correct, just slower) regularizer
    singular = bare_singular_locus(b=b, q_order=q_order)
    if singular is None and on_kac:
        # An outer h - d_rs denominator can vanish before any inner
        # recursion denominator does (Ising h=0 at q_order=6).
        singular = bare_singular_locus(b=b)
    if singular is not None:
        from .regularization import _regularized_chiral_taylor

        return _regularized_chiral_taylor(
            h, h_externals, n_terms, c=singular[1], q_order=q_order,
            z_of_u=z_of_u, u_symmetric=u_symmetric, chiral_norm=chiral_norm, z_shift=z_shift,
        )
    from .singularity import near_singular_compute

    # exact labels reach the certified block evaluation, which resolves their offsets; a rounded mpmath input
    # (complex decimal text is parsed at the working precision) cannot carry one past it
    if (not on_kac and isinstance(h, (mp.mpf, mp.mpc))
            and math.isfinite(h64.real) and math.isfinite(h64.imag)):
        _, kac64 = _kac_weights(b, q_order)
        if any(abs(pole-h64) <= 1e-8*max(abs(h64), abs(pole)) for pole in kac64):
            from .conditioning import pole_loss, report
            from .polyblock import chiral_denominator_roots

            genuine = chiral_denominator_roots(h_externals, b=b, q_order=q_order)
            report(pole_loss(h, genuine), what="chiral Taylor", source=h)

    def bare():
        evaluate_block = _fixed_h_block_evaluator(
            h,
            h_externals,
            b=b,
            q_order=q_order,
            chiral_norm=chiral_norm,
            z_shift=z_shift,
        )
        return taylor_coefficients(
            lambda u: evaluate_block(z_of_u(u)),
            u_symmetric,
            n_terms - 1,
        )

    # near (not on) a singular locus the colliding Kac poles cancel pairwise: verified guard digits
    # (G_zzb_table at 7/10 + 1e-16, dps 30, lost 15 of 30 without them); b is kept as given
    coefficients = near_singular_compute(_c_input_of_b(b), f"chiral_taylor:{n_terms}", bare, h_externals=h_externals,
                                         q_order=q_order,
                                         cache_context=(h, z_of_u, u_symmetric, chiral_norm, z_shift, b))
    return [+value for value in coefficients]


def chiral_z_taylor(
    h,
    h_externals: Sequence,
    n_terms: int,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
):
    """Taylor coefficients of F(h, z; {h_i}; c) around z = 1/2."""
    return _chiral_taylor(
        h,
        h_externals,
        n_terms,
        b=b, c=c,
        q_order=q_order,
        z_of_u=COORDINATES['z'].z_of_u,
        u_symmetric=z_symmetric(),
        chiral_norm=chiral_norm,
        z_shift=z_shift,
    )


def _flint_prec_bits() -> int:
    return int(mp.mp.dps * 3.33) + 20


def _arb_to_mpf_checked(a, ndig, *, what="block coefficient"):
    """Convert an Arb midpoint to ``mp.mpf``, rejecting unresolved intervals.

    If ``arb.str(radius=False)`` returns an interval, raise an error asking
    for more precision. Otherwise parse the midpoint, even when only its
    leading digits are accurate: the contribution scorer ranks poles by
    magnitude and can use these values at low working precision.
    """
    s = a.str(ndig, radius=False)
    if "[" in s or "/" in s:
        raise ValueError(
            f"flint precision collapsed extracting a {what} "
            f"(arb={s!r}); raise mp.dps (current dps={mp.mp.dps})."
        )
    return mp.mpf(s)


from collections import OrderedDict as _OrderedDict

_GEOMETRY_CACHE_MAXSIZE = 64
_GEOMETRY_CACHE: _OrderedDict = _OrderedDict()


def _geometry_scalar_key(value):
    """Hashable precision-exact key fragment for an mpf/mpc scalar, or None."""
    return _chiral_eval_memo_scalar_key(value)


def _flint_geometry_series_coeffs(
    n_terms,
    sigma,
    exp_z,
    exp_omz,
    exp_theta,
    *,
    z_of_u,
    u_symmetric,
):
    """Return ``(q_coeffs, base_coeffs)`` via flint power-series arithmetic.

    Mirrors the mpmath geometry in `_chiral_geometry_series_coeffs`, but builds
    the whole coordinate geometry as truncated `acb_series` objects instead of
    asking `mp.taylor` to Richardson-extrapolate two separate scalar functions.
    Returns ``None`` when the balls cannot be certified (e.g. real z < 0, where
    the nome's ball straddles the cut of the log); the caller then uses the
    mpmath geometry.
    """
    from flint import arb, acb, acb_series, ctx as flint_ctx

    old_cap = flint_ctx.cap
    flint_ctx.cap = n_terms
    try:
        flint_ctx.prec = _flint_prec_bits()
        ndigits = mp.mp.dps + 5

        def _to_acb(x):
            # exact: an mpf is a binary float, so it enters the ball with radius 0 at any precision
            x = mp.mpc(x)
            return acb(_mpf_to_arb(x.real), _mpf_to_arb(x.imag))

        u0 = _to_acb(u_symmetric)
        sigma_f = _to_acb(sigma)
        exp_z_f = _to_acb(exp_z)
        exp_omz_f = _to_acb(exp_omz)
        exp_theta_f = _to_acb(exp_theta)

        def negligible(term, total):
            """Whether every coefficient of ``term`` is below 2^-work_bits of ``total`` (coefficients far below the
            series' scale: of 2^-64 times it)."""
            tot = total.coeffs()[:n_terms]
            scale = max((abs(c.mid()) for c in tot), default=arb(0))
            tol = arb(2) ** -(work_bits + 8)
            for k, t in enumerate(term.coeffs()[:n_terms]):
                size = abs(tot[k].mid()) if k < len(tot) else arb(0)
                size = size if size > scale * arb(2) ** -64 else scale * arb(2) ** -64
                mag = abs(t)
                if not mag.mid() + mag.rad() <= tol * size:
                    return False
            return True

        def theta_sums(q_series, full):
            """``(theta2_reduced, theta3, theta4)`` as series in u (theta2/theta4 only with ``full``), summed until
            two consecutive terms are below the working precision in every coefficient. A fixed term count ignored
            |q0|: towards z -> 1 the nome approaches 1 and the geometry lost up to 25 digits."""
            two_s = acb_series([acb(2)])
            if not abs(q_series.coeffs()[0]) < 1:
                raise _Unresolved
            theta2_reduced = theta3 = theta4 = acb_series([acb(1)])
            quiet, nn = 0, 0
            while quiet < 2:
                nn += 1
                qn = q_series ** (nn * nn)
                theta3 = theta3 + two_s * qn
                if full:
                    theta2_reduced = theta2_reduced + q_series ** (nn * (nn + 1))
                    theta4 = theta4 + acb_series([acb(2 * (-1) ** nn)]) * qn
                quiet = quiet + 1 if negligible(qn, theta3) else 0
            return theta2_reduced, theta3, theta4

        def build_geometry():
            is_q_coord = z_of_u is z_of_q
            is_y_coord = z_of_u is z_of_y

            if is_q_coord:
                # Theta-native geometry: q is the expansion variable, so the Jacobi
                # identities z = (theta2/theta3)^4, 1 - z = (theta4/theta3)^4 let the
                # whole prefactor be assembled from theta series of q directly --
                # no z = 16q theta2^4/theta3^4 ratio, no 1 - z subtraction, no log of
                # a ratio (each theta is 1 + O(q), so its log is well conditioned).
                # The combined exponents collapse factor1..4 (16q)^{-sigma}
                # z^{exp_z} (1-z)^{exp_omz} theta3^{exp_theta} to
                #   (16q)^{exp_z - sigma} theta2r^{4 exp_z} theta4^{4 exp_omz}
                #   theta3^{exp_theta - 4 exp_z - 4 exp_omz},
                # with theta2r the reduced theta2 (its q^{1/4} folded into 16q).
                # Share one theta3 series across the factors.
                q_series = acb_series([u0, acb(1)])
                theta2_reduced, theta3, theta4 = theta_sums(q_series, full=True)
                sixteen_q = acb_series([acb(16)]) * q_series

                def _pow(series, exp_f):
                    return (exp_f * series.log()).exp()

                base_series = (
                    _pow(sixteen_q, exp_z_f - sigma_f)
                    * _pow(theta2_reduced, 4 * exp_z_f)
                    * _pow(theta4, 4 * exp_omz_f)
                    * _pow(theta3, exp_theta_f - 4 * exp_z_f - 4 * exp_omz_f)
                )
            else:
                # zzb / yyb: z (or y) is the variable, so z^{exp_z}, (1-z)^{exp_omz}
                # are cheap; the cost is the elliptic nome q(z), which has no theta
                # shortcut (so the theta-native simplification does not apply here).
                if is_y_coord:
                    one = acb(1)
                    one_plus_y = acb_series([one + u0, 1])
                    one_plus_y_sq = one_plus_y * one_plus_y
                    denom = acb_series([one + u0 * u0, 2 * u0, one])
                    z_series = one_plus_y_sq / (acb_series([acb(2)]) * denom)
                else:
                    z_series = acb_series([u0, 1])

                z_coeffs = z_series.coeffs()
                one_minus_z = acb_series(
                    [acb(1) - z_coeffs[0]] + [-c for c in z_coeffs[1:]]
                )
                K_series = z_series.elliptic_k()
                Kp_series = one_minus_z.elliptic_k()
                pi_s = acb_series([acb(arb.pi())])
                q_series = (-pi_s * Kp_series / K_series).exp()

                sixteen_q = acb_series([acb(16)]) * q_series
                factor1 = ((-sigma_f) * sixteen_q.log()).exp()
                factor2 = (exp_z_f * z_series.log()).exp()
                factor3 = (exp_omz_f * one_minus_z.log()).exp()
                theta3 = theta_sums(q_series, full=False)[1]
                factor4 = (exp_theta_f * theta3.log()).exp()
                base_series = factor1 * factor2 * factor3 * factor4

            return q_series, base_series

        work_bits = _flint_prec_bits()
        im_tol = arb(2) ** (-(work_bits - 10))

        class _Unresolved(Exception):
            pass

        def _deficit_bits(coeffs_list):
            """Bits by which the balls miss the working precision: each coefficient must be certified to
            2^-work_bits relative, or -- one far below the series' scale -- to 2^-work_bits of 2^-64 times it."""
            scale = max((abs(c.mid()) for c in coeffs_list), default=arb(0))
            floor = scale * arb(2) ** -64
            worst = 0
            for coeff in coeffs_list:
                # a ball that is not finite (at real z < 0 the nome's ball straddles the cut of the log, and
                # log(16 q) comes out indeterminate) must not be read as its midpoint (it printed as 0)
                if not coeff.is_finite():
                    raise _Unresolved
                size = abs(coeff.mid())
                size = size if size > floor else floor
                rad = coeff.rad()
                if size == 0:
                    # the whole series has zero midpoints: exact zeros certify, a ball around 0 certifies nothing
                    # (zzb at real z < 0, c > 25: an indeterminate phase gave [+/- 3] balls and a block of exactly 0)
                    if rad != 0:
                        raise _Unresolved
                    continue
                if rad > size * arb(2) ** -work_bits:
                    worst = max(worst, int(float((rad / size).log() / arb(2).log())) + work_bits + 1)
            return worst

        def _to_mpmath(coeffs_list):
            result = []
            for coeff in coeffs_list:
                re_val = +_arb_to_mpf(coeff.real)            # the exact midpoint, rounded to working precision
                im_val = coeff.imag
                if abs(im_val) <= im_tol:
                    result.append(re_val)
                else:
                    result.append(mp.mpc(re_val, +_arb_to_mpf(im_val)))
            while len(result) < n_terms:
                result.append(mp.mpf(0))
            return result

        # The coordinate composition cancels (the z-series coefficients grow like 2^n and combine to O(1) ones in
        # y; 20 terms lost ~13 digits in y, ~7 in z), so the geometry is built with guard bits, widened by the
        # measured deficit until every coefficient's ball certifies the working precision.
        guard = 32 + 2 * n_terms
        try:
            previous = None
            for _attempt in range(6):
                flint_ctx.prec = work_bits + guard
                q_series, base_series = build_geometry()
                q_list = q_series.coeffs()[:n_terms]
                base_list = base_series.coeffs()[:n_terms]
                deficit = max(_deficit_bits(q_list), _deficit_bits(base_list))
                if previous is not None and deficit >= previous:
                    raise _Unresolved          # more guard bits do not help (a branch cut, not rounding)
                previous = deficit
                if deficit == 0:
                    flint_ctx.prec = work_bits
                    return _to_mpmath(q_list), _to_mpmath(base_list)
                guard += deficit + 32
            raise _Unresolved
        except _Unresolved:
            return None               # the mpmath geometry (taylor_coefficients) takes over
    finally:
        flint_ctx.cap = old_cap
        flint_ctx.prec = _flint_prec_bits()


def _chiral_geometry_series_coeffs(h_externals, n_terms, b, z_shift, sigma,
                                   exp_z, exp_omz, exp_theta,
                                   z_of_u=None, u_symmetric=None,
                                   coord_key="z"):
    """Delta-independent coordinate-Taylor coeffs (q(z(u)) and the prefactor
    base, expanded around ``u = u_symmetric``), cached by
    (precision, n_terms, coordinate, b, h_ext, z_shift).

    These are the only transcendental work in the series block evaluator
    (`precision.taylor_coefficients` of the smooth nome/theta geometry composed with the chiral
    coordinate map ``z(u)``) and depend only on the geometry -- not on the
    exchanged h -- so every sample Delta and every per-target interpolation
    group reuses one computation.

    With the default ``z_of_u`` (identity around z = 1/2) this reproduces the
    z-coordinate geometry bit-for-bit; passing ``z_of_u = z_of_y`` /
    ``u_symmetric = 0`` gives the same fast geometry for the y coordinate.
    """
    if z_of_u is None:
        z_of_u = lambda u: u  # noqa: E731 - identity z(z)
        u_symmetric = mp.mpf(1) / 2
    key = (
        mp.mp.prec,
        n_terms,
        coord_key,
        _geometry_scalar_key(u_symmetric),
        _geometry_scalar_key(b),
        tuple(_geometry_scalar_key(hi) for hi in h_externals),
        _geometry_scalar_key(z_shift),
    )
    cached = _GEOMETRY_CACHE.get(key)
    if cached is not None:
        _GEOMETRY_CACHE.move_to_end(key)
        return cached

    flint_coeffs = _flint_geometry_series_coeffs(
        n_terms,
        sigma,
        exp_z,
        exp_omz,
        exp_theta,
        z_of_u=z_of_u,
        u_symmetric=u_symmetric,
    )
    if flint_coeffs is not None:
        q_coeffs, base_coeffs = flint_coeffs
    else:
        q_coeffs = taylor_coefficients(lambda u: q_from_z(z_of_u(u)), u_symmetric, n_terms - 1)

        def base_fn(u):
            z = z_of_u(u)
            q = q_from_z(z)
            return (
                (16 * q) ** (-sigma)
                * z ** exp_z
                * (1 - z) ** exp_omz
                * _theta3(q) ** exp_theta
            )

        base_coeffs = taylor_coefficients(base_fn, u_symmetric, n_terms - 1)

    _GEOMETRY_CACHE[key] = (q_coeffs, base_coeffs)
    _GEOMETRY_CACHE.move_to_end(key)
    if len(_GEOMETRY_CACHE) > _GEOMETRY_CACHE_MAXSIZE:
        _GEOMETRY_CACHE.popitem(last=False)
    return q_coeffs, base_coeffs


def _acb(x):
    """mpmath scalar -> flint acb. Fast path via _mpf_to_acb."""
    return _mpf_to_acb(mp.mpf(x) if not isinstance(x, mp.mpc) else x)


def _acb_to_scalar(value, ndigits=None):
    """flint acb -> mp.mpf when numerically real, else mp.mpc. Fast path."""
    re = _arb_to_mpf(value.real)
    im = _arb_to_mpf(value.imag)
    if im == 0:
        return re
    return mp.make_mpc((re._mpf_, im._mpf_))


# Memo for _chiral_series_evaluator. The evaluator is a pure per-``h``
# function of (externals, geometry, central charge, precision) -- notably NOT
# of the spin -- so every spin table of a bound rebuild asks for an identical
# evaluator. Sharing one lets the whole spin family reuse the per-evaluator
# ``pole_laurent`` scaffolding and the analytic scorer's chiral-factor cache,
# whose singular-side arguments (the Kac weights h_{r,s}) are spin-independent.
# Bounded: oldest entry evicted (insertion-ordered dict), matching the other
# module caches; an evicted evaluator still works for holders of a reference,
# it just stops being shared.
_CHIRAL_EVAL_MEMO = {}
_CHIRAL_EVAL_MEMO_MAXSIZE = 16


def _chiral_eval_memo_scalar_key(v):
    """Hashable value-key for a memo scalar (None / str / real / complex)."""
    from fractions import Fraction

    if v is None or isinstance(v, (str, int)):
        return v
    if isinstance(v, Fraction):
        return ("rational", v.numerator, v.denominator)
    if isinstance(v, mp.mpf):
        return (v._mpf_, mp.mpf(0)._mpf_)
    if isinstance(v, mp.mpc):
        return v._mpc_
    z = mp.mpc(v)
    return (mp.mpf(z.real)._mpf_, mp.mpf(z.imag)._mpf_)


def _chiral_series_evaluator(
    h_externals: Sequence,
    n_terms: int,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
    z_of_u=None,
    u_symmetric=None,
    coord_key="z",
):
    """Memoized front for :func:`_chiral_series_evaluator_uncached`.

    Keyed by every construction argument plus the working precision (the
    evaluator's flint data is built at the precision current at construction).
    ``z_of_u`` enters the key as the function object itself (identity hash;
    keeping it in the key also keeps it alive, so the id can never be reused
    by a different function). Unkeyable arguments (e.g. non-mpmath scalars)
    simply skip the memo.

    Refuses a recognized singular c before the memo lookup, so an evaluator
    built there under `singularity.bare_recursion_probes` is never handed out.
    """
    from .singularity import merged_singular_active, recognized_singular_locus, exact_locus_c
    from .numeric_input import parse_number

    n_terms = _integer_control('n_terms', n_terms, 1)
    q_order = _integer_control('q_order', q_order)
    if c is None and b is not None:
        origin = c_origin_of_b(b)
        if origin is not None:
            b, c = None, origin
    c = parse_number(c)
    # The low-level mpmath convention recognizes a locus at the precision c
    # carries. Freeze that choice before an external label raises precision.
    if merged_singular_active() and c is not None:
        locus_c = exact_locus_c(c, q_order=q_order)
        if locus_c is not None:
            c = parse_number(locus_c)
    h_externals = tuple(parse_number(value) for value in h_externals)
    from .recursion import kac_limit_mode

    kac_limit = kac_limit_mode()        # the block on a Kac weight (the CLI's --kac-limit), fixed per evaluator

    merged_c0 = None
    if merged_singular_active():
        c_check = c if c is not None else _c_input_of_b(b)
        if recognized_singular_locus(c_check, q_order=q_order) is not None:
            merged_c0 = c_check
    if merged_c0 is None:
        reject_singular_c(
            c, b=b, q_order=q_order, h_externals=h_externals, where="chiral series evaluator",
        )
    try:
        key = (
            "merged" if merged_c0 is not None else "bare",
            tuple(_chiral_eval_memo_scalar_key(hi) for hi in h_externals),
            int(n_terms),
            _chiral_eval_memo_scalar_key(b),
            _chiral_eval_memo_scalar_key(c),
            int(q_order),
            _chiral_eval_memo_scalar_key(chiral_norm),
            _chiral_eval_memo_scalar_key(z_shift),
            z_of_u,
            _chiral_eval_memo_scalar_key(u_symmetric),
            str(coord_key),
            mp.mp.prec,
            kac_limit,
        )
    except (TypeError, ValueError):
        key = None
    if key is not None:
        hit = _CHIRAL_EVAL_MEMO.get(key)
        if hit is not None:
            return hit
    from .singularity import near_singular_compute

    def build():
        return _chiral_series_evaluator_uncached(
            h_externals, n_terms, b=b, c=c, q_order=q_order,
            chiral_norm=chiral_norm, z_shift=z_shift,
            z_of_u=z_of_u, u_symmetric=u_symmetric, coord_key=coord_key,
            merged_c0=merged_c0, kac_limit=kac_limit,
        )

    if merged_c0 is not None:
        # exactly at the singular c: no near-singular guard (nothing cancels between probe poles) and no probes
        evaluate = build()
        if key is not None:
            _CHIRAL_EVAL_MEMO[key] = evaluate
        return evaluate

    probe_h = mp.mpf("1.41421356237309504880")     # a generic weight (never a Kac weight)
    try:
        # near (not on) a singular locus the colliding Kac poles cancel pairwise: the evaluator is built AND
        # evaluated at the verified guard precision (G_zzb_table_many at 7/10 + 1e-16, dps 30, lost 14.5 of 30)
        evaluate, extra = near_singular_compute(
            c if c is not None else _c_input_of_b(b), f"series_evaluator:{coord_key}:{n_terms}:{kac_limit}", build,
            h_externals=h_externals, q_order=q_order,
            fingerprint=lambda ev: None if ev is None else ev(probe_h), return_guard=True,
            cache_context=(z_of_u, u_symmetric, chiral_norm, z_shift, b),
        )
        if extra and evaluate is not None:
            evaluate = _at_guard_precision(evaluate, mp.mp.dps + extra)
    except ZeroDivisionError:
        # an inert (zero-residue) singular term can still divide 0/0: same clear error
        reject_singular_c(c, b=b, q_order=q_order, where="chiral series evaluator")
        raise
    if key is not None and evaluate is not None:
        if len(_CHIRAL_EVAL_MEMO) >= _CHIRAL_EVAL_MEMO_MAXSIZE:
            try:
                _CHIRAL_EVAL_MEMO.pop(next(iter(_CHIRAL_EVAL_MEMO)))
            except (StopIteration, KeyError):
                pass
        _CHIRAL_EVAL_MEMO[key] = evaluate
    return evaluate


def _at_guard_precision(evaluate, work_dps):
    """Wrap an evaluator (and its attached helpers) so every call runs at ``work_dps``."""
    from functools import wraps

    def wrap(function):
        @wraps(function)
        def guarded(*args, **kwargs):
            with mp.workdps(max(work_dps, mp.mp.dps)):
                return function(*args, **kwargs)
        return guarded

    guarded_evaluate = wrap(evaluate)
    for name in ("pole_laurent", "pole_data", "kac_locations", "kac_poles", "kac_labels", "merged_pole_order", "many"):
        if hasattr(evaluate, name):
            setattr(guarded_evaluate, name, wrap(getattr(evaluate, name)))
    guarded_evaluate.real_analytic = getattr(evaluate, "real_analytic", False)
    return guarded_evaluate


def _chiral_series_evaluator_uncached(
    h_externals: Sequence,
    n_terms: int,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
    z_of_u=None,
    u_symmetric=None,
    coord_key="z",
    merged_c0=None,
    kac_limit=None,
):
    """Coordinate-general core of the fast chiral block evaluator.

    Returns a callable ``h -> [F-Taylor coeffs around u = u_symmetric]`` (flint
    series). The block factorizes as

        F(h, z(u)) = (16 q)^h * N^{-h} * BASE'(z(u)) * sum_k H_k(h) q(z(u))^k,

    where ``BASE'(z) = (16 q)^{-sigma} z^{sigma+z_shift} (1-z)^{sigma-h2-h3}
    theta3(q)^{(c-1)/2 - 4 sum h_i}`` and the nome series ``q(z)`` are
    **Delta-independent** -- they depend only on the geometry, externals, and
    central charge. We compute those once (as coordinate-Taylor series in
    ``t = u - u_symmetric``, composing with the chiral map ``z(u)``) and reuse
    them for every ``h``. Per ``h`` only the cheap Zamolodchikov
    ``H``-recursion plus a handful of truncated power-series operations remain
    -- no elliptic-nome / theta transcendentals, and no ``taylor_coefficients`` numerical
    differentiation. The kinematic ``z_shift`` exponent stays a power of ``z``
    (so it composes through ``z(u)`` unchanged), matching the slow per-``h``
    Taylor builders for every coordinate.

    At a singular c, an ``h`` ON a Kac weight gets the block ``kac_limit`` selects
    (:func:`~virasoro_blocks.recursion.kac_limit_mode`; the irreducible module's by default).
    """
    from .recursion import kac_limit_mode

    kac_limit = kac_limit_mode(kac_limit)
    n_terms = _integer_control('n_terms', n_terms, 1)
    q_order = _integer_control('q_order', q_order)
    _check_externals(h_externals)
    b = _resolve_b(b, c)
    c_val = mp.convert(_c_input_of_b(b) if c is None else c)
    N = _resolve_chiral_norm(chiral_norm)
    z_shift_val = mp.mpf(0) if z_shift is None else z_shift

    sigma = (c_val - 1) / 24
    exp_z = sigma + z_shift_val
    exp_omz = sigma - h_externals[1] - h_externals[2]
    exp_theta = (c_val - 1) / 2 - 4 * sum(h_externals)

    # Delta-independent coordinate-Taylor coefficients (cached across sample
    # Deltas and per-target interpolation groups -- the only transcendental
    # work).
    q_coeffs, base_coeffs = _chiral_geometry_series_coeffs(
        h_externals, n_terms, b, z_shift_val, sigma, exp_z, exp_omz, exp_theta,
        z_of_u=z_of_u, u_symmetric=u_symmetric, coord_key=coord_key,
    )

    from flint import acb, acb_mat, acb_series, ctx as flint_ctx

    prec_bits = _flint_prec_bits()
    flint_ctx.prec = prec_bits
    flint_ctx.cap = n_terms

    # Fetch the cached arb context for fast per-h H evaluation (same cache as
    # H_q_series; this call is free on subsequent evaluate() calls).
    b_typed = _maybe_real_scalar(b) if isinstance(b, mp.mpc) else b
    h_ext_typed = _kac_input_values(h_externals)
    if merged_c0 is not None:
        # at a recognized singular c: the exact merged-pole H (recursion.build_merged_H_context); the pole-data
        # helpers below see an unknown mode and return None (the singular-c scorers read the Laurent table)
        from .recursion import build_merged_H_context
        _pair_tbl, _H_tbl_ctx = None, None
        _arb_ctx = build_merged_H_context(q_order, merged_c0, h_externals)
        _is_strip = False
    else:
        _pair_tbl, _H_tbl_ctx, _arb_ctx = _build_H_context(q_order, b_typed, h_ext_typed)
        _is_strip = _pair_tbl.is_strip_symmetric
    # H construction sets its own ball precision. Geometry may now carry
    # more digits than a reused H table, so restore its arithmetic context.
    flint_ctx.prec = prec_bits
    _arb_ctx_mode = _arb_ctx.get("mode") if _arb_ctx is not None else None
    _merged_real = _arb_ctx_mode == "merged" and _arb_ctx["real"]

    # Below a structural singularity the bare context is valid away from
    # Kac weights, but a fixed-h limit still needs the exact-c Laurent table.
    # bare_singular_locus without a cutoff also respects bare probe contexts.
    fixed_kac_c = None
    fixed_kac_locations = frozenset()
    if _arb_ctx_mode != "merged":
        from .singularity import exact_b_squared, exact_locus_c

        c_input = _c_input_of_b(b) if c is None else c
        if bare_singular_locus(c_input) is not None:
            fixed_kac_c = exact_locus_c(c_input, q_order=q_order)
            beta = exact_b_squared(fixed_kac_c, q_order=q_order)
            if beta is not None:
                fixed_kac_locations = _rational_kac_weights(beta, q_order)


    # When the central charge and externals are real (real_b / strip_twin /
    # imag_b modes), every factor of the block series -- the geometry coeffs
    # q(z(u)) and BASE'(z(u)), the H-recursion output, and h itself -- is real,
    # so the whole series multiply runs in arb instead of acb (half the C-level
    # work, no always-zero imaginary channel). The acb path stays for genuinely
    # complex inputs (generic_complex mode) and the Kac-pole mpmath fallback.
    def _real_part_or_none_local(x):
        if isinstance(x, mp.mpf):
            return x
        if isinstance(x, mp.mpc):
            return mp.mpf(x.real) if x.imag == 0 else None
        return None

    _c_real = _real_part_or_none_local(c_val)
    _use_arb = (
        _arb_ctx is not None
        and (_arb_ctx_mode != "merged" or _merged_real)
        and _c_real is not None
        and all(_real_part_or_none_local(hi) is not None for hi in h_externals)
        and all(
            isinstance(v, mp.mpf) or (isinstance(v, mp.mpc) and v.imag == 0)
            for v in q_coeffs
        )
        and all(
            isinstance(v, mp.mpf) or (isinstance(v, mp.mpc) and v.imag == 0)
            for v in base_coeffs
        )
        # the arb path takes log(16 q): at real z < 0 the nome is negative (q(z/(z-1)) = -q(z)) while the base
        # series is real, and the log gave NaN -> a block of exactly 0
        and mp.re(q_coeffs[0]) > 0
    )
    # The series is real at real h -- so by reflection its data at conj(h) is the conjugate of that at h -- when the
    # Kac data are real or strip-symmetric and c, the externals and the geometry are real up to roundoff (b on the
    # unit circle leaves c_from_b(b) an ulp-sized imaginary part at some c, which only _use_arb's exact test sees).
    _noise = mp.mpf(10) ** (5 - mp.mp.dps)

    def _nearly_real(v):
        return not isinstance(v, mp.mpc) or abs(v.imag) <= _noise * max(1, abs(v))

    _real_analytic = (
        (_arb_ctx_mode in ("strip_real", "real_nonstrip") or _merged_real)
        and _nearly_real(c_val)
        and all(_nearly_real(v) for v in (*h_externals, *q_coeffs, *base_coeffs))
        and mp.re(q_coeffs[0]) > 0
    )

    if _use_arb:
        from flint import arb, arb_mat, arb_series

        def _re(v):
            return v if isinstance(v, mp.mpf) else mp.mpf(v.real)

        q_series = arb_series([_mpf_to_arb(_re(v)) for v in q_coeffs])
        q16_series = q_series * arb(16)
        base_series = arb_series([_mpf_to_arb(_re(v)) for v in base_coeffs])

        q_pow = [arb_series([arb(1)])]
        for _k in range(1, q_order + 1):
            q_pow.append(q_pow[-1] * q_series)
        Q_rows = []
        for k in range(q_order + 1):
            row = list(q_pow[k].coeffs())[:n_terms]
            row += [arb(0)] * (n_terms - len(row))
            Q_rows.append(row)
        Q_mat = arb_mat(Q_rows)
    else:
        q_series = acb_series([_mpf_to_acb(v) for v in q_coeffs])
        q16_series = q_series * acb(16)
        base_series = acb_series([_mpf_to_acb(v) for v in base_coeffs])

        # Precompute q^k series for k = 0..q_order (h-independent).
        q_pow = [acb_series([acb(1)])]
        for _k in range(1, q_order + 1):
            q_pow.append(q_pow[-1] * q_series)

        # Q_mat[k, i] = coefficient i of q(u)^k  (shape: (q_order+1) x n_terms).
        # In evaluate(h): H_series = sum_k H_k * q^k  ≡  H_vec @ Q_mat.
        Q_rows = []
        for k in range(q_order + 1):
            row = list(q_pow[k].coeffs())[:n_terms]
            row += [acb(0)] * (n_terms - len(row))
            Q_rows.append(row)
        Q_mat = acb_mat(Q_rows)   # (q_order+1) x n_terms

    # The block prefactor is (16 q / N)^h = exp(h * (log(16 q) - log N)); the log
    # of the series is h-independent, so precompute it once at construction and
    # reuse across every evaluate / evaluate_many call (flint's series ``** h``
    # otherwise recomputes the log on each call -- roughly half the per-h power
    # cost). The Delta-independent -log N is folded into the constant term here,
    # so the single per-h series exp yields the whole prefactor directly: this
    # removes the separate per-h N^{-h} scalar, which profiled as ~half the
    # F-series assembly cost (an mpmath ``N ** (-h)`` transcendental + flint
    # round trip on the real path, the mpc_pow hotspot on the complex path). N is
    # a positive real chiral norm, so log N is a real scalar. Native field: arb
    # when _use_arb, else acb.
    _log_N = mp.log(N)
    if _use_arb:
        log_q16N_series = q16_series.log() - arb_series([_mpf_to_arb(mp.mpf(_log_N))])
    else:
        log_q16N_series = q16_series.log() - acb_series([_mpf_to_acb(_log_N)])

    _complex_series_views = None

    def _complex_series_data():
        nonlocal _complex_series_views
        if _complex_series_views is None:
            if _use_arb:
                _q16_acb = acb_series([acb(c) for c in q16_series.coeffs()])
                _complex_series_views = (
                    _q16_acb,
                    acb_series([acb(c) for c in base_series.coeffs()]),
                    acb_mat(
                        [[acb(Q_mat[r, cc]) for cc in range(Q_mat.ncols())]
                         for r in range(Q_mat.nrows())]
                    ),
                    _q16_acb.log() - acb_series([_mpf_to_acb(_log_N)]),
                )
            else:
                _complex_series_views = (
                    q16_series, base_series, Q_mat, log_q16N_series,
                )
        return _complex_series_views

    _pole_cache = {}

    def _report_conditioning(h_label):
        # digits lost to h - d_rs at the nearest pole with a nonzero residue (conditioning.report warns or raises);
        # the merged singular-c modes decide their own poles
        if _arb_ctx_mode == "merged" or (fixed_kac_locations and h_label in fixed_kac_locations):
            return
        from .conditioning import pole_loss, report

        cached = _pole_cache.get(mp.mp.prec)
        if cached is None:
            import numpy as np

            poles = evaluate_kac_poles() or []
            cached = _pole_cache[mp.mp.prec] = (poles, np.array([complex(p) for p in poles], dtype=complex))
        poles, approx = cached
        report(pole_loss(h_label, approx, lambda i: poles[i]), what="chiral series evaluator", source=h_label)

    def evaluate(h, return_arb=False, return_acb=False):
        h = h_label = _coerce_delta_scalar(h)
        _report_conditioning(h_label)
        if not isinstance(h, (mp.mpf, mp.mpc)):
            h = mp.convert(h)
        flint_ctx.prec = _flint_prec_bits()
        flint_ctx.cap = n_terms

        # Type h for the arb-native H-recursion path.
        h_typed = (
            _maybe_real_scalar(h) if isinstance(h, (mp.mpf, mp.mpc))
            else _maybe_real_scalar(mp.mpc(h))
        )

        h_is_real = not isinstance(h, mp.mpc) or h.imag == 0

        # Fast arb-native H evaluation: stays in flint, avoids arb->mpf->acb.
        # ``H_arb`` is the real fast-path (arb) result; ``H_acb`` is the native
        # acb result for a complex h on real Kac data (the "complex pole" path).
        H_arb = None
        H_acb = None
        if fixed_kac_locations and h_label in fixed_kac_locations:
            from .regularization import H_q_series_regularized

            H_coeffs = H_q_series_regularized(h_label, h_externals, c=fixed_kac_c, q_order=q_order,
                                              kac_limit=kac_limit)
            flint_ctx.prec = _flint_prec_bits()
            if _use_arb and all(mp.im(hk) == 0 for hk in H_coeffs):
                H_arb = [_mpf_to_arb(mp.re(hk)) for hk in H_coeffs]
            else:
                H_acb = [_mpf_to_acb(hk) for hk in H_coeffs]
        elif _arb_ctx_mode == "merged":
            from .recursion import eval_merged_H
            if h_is_real and _merged_real:
                H_arb = eval_merged_H(h_label, _arb_ctx, q_order, kac_limit=kac_limit)
            else:
                H_acb = eval_merged_H(h_label, _arb_ctx, q_order, complex_out=True, kac_limit=kac_limit)
            if H_arb is None and H_acb is None:
                from .requests import BlockPoleError

                raise BlockPoleError(
                    f"the block has a genuine pole at this exchanged weight: h = {mp.nstr(h, 12)} is a merged Kac "
                    f"pole of order {evaluate.merged_pole_order(h)} at the singular c for these externals (the "
                    "channel is genuinely singular: a null state does not decouple)",
                    family="virasoro", exchanged=h, pole=h)
        elif _arb_ctx is not None:
            if _arb_ctx_mode == "complex_nonstrip":
                # Complex Kac data (generic-complex c): acb matmul for any h.
                H_acb = _eval_H_at_acb_nonstrip_complex(
                    h_typed, _arb_ctx, q_order, return_acb=True
                )
            elif h_is_real:
                if _is_strip:
                    H_arb = _eval_H_at_acb_strip_real(
                        h_typed, _arb_ctx, q_order, return_arb=True
                    )
                else:
                    H_arb = _eval_H_at_arb_real(
                        h_typed, _arb_ctx, q_order, return_arb=True
                    )
            else:
                # Complex h, real Kac data: stay in flint (acb matmul) rather
                # than dropping to the mpmath H_q_series Python loop.
                if _is_strip:
                    H_acb = _eval_H_at_acb_strip_complex(
                        h_typed, _arb_ctx, q_order, return_acb=True
                    )
                else:
                    H_acb = _eval_H_at_acb_nonstrip_complex(
                        h_typed, _arb_ctx, q_order, return_acb=True
                    )

        # All-real fast path: pure arb series arithmetic.
        flint_ctx.prec = _flint_prec_bits()
        if _use_arb and H_arb is not None and h_is_real:
            H_vec = arb_mat([list(H_arb)])
            H_row = H_vec * Q_mat
            H_series = arb_series([H_row[0, i] for i in range(n_terms)])
            h_mpf = mp.mpf(h) if not isinstance(h, mp.mpc) else mp.mpf(h.real)
            h_arb = _mpf_to_arb(h_mpf)
            F_series = (h_arb * log_q16N_series).exp() * base_series * H_series
            coeffs = list(F_series.coeffs())
            if return_arb:
                # Hand back native arb coeffs so downstream G-table assembly /
                # parity folding can stay in flint (no arb->mpf->arb round trip).
                return [
                    coeffs[i] if i < len(coeffs) else arb(0)
                    for i in range(n_terms)
                ]
            if return_acb:
                return [
                    acb(coeffs[i]) if i < len(coeffs) else acb(0)
                    for i in range(n_terms)
                ]
            return [
                _arb_to_mpf(coeffs[i]) if i < len(coeffs) else mp.mpf(0)
                for i in range(n_terms)
            ]

        if return_arb and not (_is_strip and h_is_real):
            # Native-arb output: either the all-real fast path (handled above) or
            # a strip-real block (1 < c < 25, real at real h -- computed via the
            # acb assembly below and returned as its arb real part). Any other
            # case (complex h / complex c) cannot produce arb; signal fallback.
            return None

        if H_arb is not None:
            # acb(arb_val) is a fast exact promotion (no string conversion).
            H_vec = acb_mat([[acb(hk) for hk in H_arb]])
        elif H_acb is not None:
            # Native acb H from the complex flint evaluators (no mpmath loop).
            H_vec = acb_mat([list(H_acb)])
        else:
            # Fallback: full mpmath path (Kac-pole hit or complex-b context with
            # no real-data flint evaluator).
            H_coeffs = H_q_series(h, h_externals, b=b, q_order=q_order)
            flint_ctx.prec = _flint_prec_bits()
            H_vec = acb_mat([[_mpf_to_acb(Hk) for Hk in H_coeffs]])

        _q16, _base, _Qm, _log_q16N = _complex_series_data()

        # Replace Horner loop with a single matrix-vector multiply.
        H_row = H_vec * _Qm               # (1, n_terms)
        H_series = acb_series([H_row[0, i] for i in range(n_terms)])

        # Use the full (possibly complex) h in the (16 q / N)^h prefactor;
        # dropping Im(h) here corrupts the block at complex Delta / complex c.
        # ``_log_q16N`` already folds in -log N, so one series exp gives the whole
        # prefactor (no separate N^{-h} scalar).
        h_full = h if isinstance(h, mp.mpc) else mp.mpf(h)
        h_acb = _mpf_to_acb(h_full)
        F_series = (h_acb * _log_q16N).exp() * _base * H_series
        coeffs = list(F_series.coeffs())
        if return_arb:
            # Reached only for a strip-real block (guarded above): real at real
            # h, so the acb series carries only roundoff in the imaginary part.
            return [
                coeffs[i].real if i < len(coeffs) else arb(0)
                for i in range(n_terms)
            ]
        if return_acb:
            # Native acb coeffs for the fused folded builder (no acb->mpc->acb
            # round trip); covers complex Delta on real Kac data, complex c, and
            # the Kac-pole mpmath-H fallback alike (F_series is acb in all).
            return [
                coeffs[i] if i < len(coeffs) else acb(0)
                for i in range(n_terms)
            ]
        return [
            _acb_to_scalar(coeffs[i]) if i < len(coeffs) else mp.mpf(0)
            for i in range(n_terms)
        ]

    _folded_map_cache = {}

    def _folded_taylor_map():
        """``(G, complex_q, active)`` for :func:`evaluate_many`: the h-independent matrix ``[Q[0] | Q[1:]^T M[1:]]``
        restricted to the pairs with a nonzero residue (``active``, indices into the pair/entry list), as real
        ``arb`` (for a complex Taylor map the real-part rows are stacked over the imaginary-part rows), built once per
        working precision."""
        from flint import arb_mat
        prec = flint_ctx.prec
        cached = _folded_map_cache.get(prec)
        if cached is not None:
            return cached
        n_k = Q_mat.nrows()
        if _use_arb:
            q_parts = [Q_mat]
        else:
            q_parts = [arb_mat([[Q_mat[k, i].real for i in range(n_terms)] for k in range(n_k)]),
                       arb_mat([[Q_mat[k, i].imag for i in range(n_terms)] for k in range(n_k)])]
        if _arb_ctx_mode == "real_nonstrip":
            active = [idx for idx, k_zero in enumerate(_arb_ctx["K_zero"]) if not k_zero]
            m_blocks = [_arb_ctx["M"]]
        elif _arb_ctx_mode == "merged":
            active = list(range(_arb_ctx["n_cols"]))
            m_blocks = [_arb_ctx["M"]]
        else:
            active = [idx for idx, entry in enumerate(_arb_ctx["entries"]) if not entry[2]]
            m_blocks = [_arb_ctx["M_re"], -_arb_ctx["M_im"]]
        m_tails = [arb_mat([[row[idx] for idx in active] for row in m.tolist()[1:]]) for m in m_blocks]
        g_rows = []
        for q_part in q_parts:
            q_t = q_part.transpose().tolist()                  # n_terms rows of q_order + 1
            q_tail = arb_mat([row[1:] for row in q_t])
            products = [(q_tail * m_tail).tolist() for m_tail in m_tails]
            for i in range(n_terms):
                g_rows.append([q_t[i][0]] + [x for p in products for x in p[i]])
        cached = (arb_mat(g_rows), not _use_arb, active)
        _folded_map_cache[prec] = cached
        return cached

    def evaluate_many(h_list):
        """Batched F-Taylor evaluation for every h at once.

        The Delta-independent H-recursion is the same compiled matrix product
        for every ``h`` (``H_k(h_j) = (M @ W)[k, j]`` with the per-(pole, h)
        weights ``W``), so the whole batch's H-table and the ``H @ Q`` step
        collapse into two ``arb_mat`` multiplies instead of one per ``h``. Only
        the ``(16 q)^h`` prefactor stays per-``h``.

        Returns a list (one per ``h``) of length-``n_terms`` real ``arb``
        coefficient lists. Returns ``None`` to signal the caller to fall back
        to per-``h`` :func:`evaluate` for complex weights or geometry,
        unsupported Kac data, or a merged Kac-pole hit.
        """
        h_list = [_coerce_delta_scalar(h) for h in h_list]
        for h_label in h_list:
            _report_conditioning(h_label)
        if fixed_kac_locations and any(h in fixed_kac_locations for h in h_list):
            return None                     # fixed-h Laurent limits use evaluate
        if _arb_ctx is None or not (_arb_ctx_mode in ("real_nonstrip", "strip_real") or _merged_real):
            return None
        if _arb_ctx_mode == "merged":
            from .recursion import merged_kac_hit

            # on a Kac location the block is the one kac_limit selects (eval_merged_H), not the merged columns
            if any(merged_kac_hit(h, _arb_ctx) for h in h_list):
                return None
        # the batch returns real coefficients: only right when the block is real at real h (a complex geometry --
        # a complex center, real z > 1, real z < 0 in y or q -- keeps an O(1) imaginary part)
        if not (_use_arb or _real_analytic):
            return None
        if any(isinstance(h, mp.mpc) and h.imag != 0 for h in h_list):
            return None

        h_mpfs = [mp.mpf(h.real) if isinstance(h, mp.mpc) else mp.mpf(h) for h in h_list]
        n_h = len(h_mpfs)
        if n_h == 0:
            return []

        from flint import arb, arb_mat
        flint_ctx.prec = _flint_prec_bits()
        flint_ctx.cap = n_terms
        zero = arb(0)
        one_arb = arb(1)

        # --- batched H-recursion folded into the Taylor map: one product per batch ---
        # H_series(h) = sum_k H_k(h) q^k = Q[0] + sum_{k>=1} Q[k] (M W)[k] with the per-(pole, h) weights
        # W = K/(h - d) (H_0 = 1, so M's row 0 is unused); G = [Q[0] | Q[1:]^T M[1:]] is h-independent, so each
        # batch is the single product G [1; W] with n_terms rows instead of the q_order + 1 rows of M W followed by
        # a second product with Q.
        G, complex_q, active = _folded_taylor_map()
        rows = [[one_arb] * n_h]
        h_acbs = None
        if _arb_ctx_mode == "real_nonstrip":
            K = _arb_ctx["K"]; d = _arb_ctx["d"]
            h_arbs = [_mpf_to_arb(h) for h in h_mpfs]
            rows += [[K[idx] / (hh - d[idx]) for hh in h_arbs] for idx in active]
        elif _arb_ctx_mode == "merged":
            # merged poles: the column weight is 1 / (h - d0)^order
            d0 = _arb_ctx["d0"]; order = _arb_ctx["power"]
            h_arbs = [_mpf_to_arb(h) for h in h_mpfs]
            inv = {}
            for idx in active:
                if d0[idx] is not None and idx not in inv:
                    inv[idx] = [1 / (hh - d0[idx]) for hh in h_arbs]
            rows += [[u ** order[idx] for u in inv[idx]] for idx in active]
        else:  # strip_real: real H from acb pole weights, as Re(M) Re(W) - Im(M) Im(W)
            entries = _arb_ctx["entries"]
            h_acbs = [_mpf_to_acb(h) for h in h_mpfs]
            re_rows, im_rows = [], []
            for idx in active:
                K_e, d_e = entries[idx][3], entries[idx][4]
                weights = [K_e / (hh_acb - d_e) for hh_acb in h_acbs]
                re_rows.append([w.real for w in weights])
                im_rows.append([w.imag for w in weights])
            rows += re_rows + im_rows
        # n_h rows of n_terms (or, for a complex Taylor map, 2 n_terms: real parts then imaginary parts)
        H_cols = (G * arb_mat(rows)).transpose().tolist()
        # an h on a Kac weight (a non-finite weight) poisons its column: 0 * nan is nan in arb
        if not all(entry.is_finite() for col in H_cols for entry in col):
            return None
        if complex_q:
            H_cols = [[acb(col[i], col[n_terms + i]) for i in range(n_terms)] for col in H_cols]

        # (16 q / N)^{h} = exp(h * log_q16N); ``log_q16N_series`` is precomputed once
        # at construction (see above) so every per-h power reuses it.
        out = []
        for j, h_mpf in enumerate(h_mpfs):
            if _use_arb:
                H_series = arb_series(H_cols[j])
                h_e = _mpf_to_arb(h_mpf)
                F_series = (h_e * log_q16N_series).exp() * base_series * H_series
                coeffs = list(F_series.coeffs())
                out.append([
                    coeffs[i] if i < len(coeffs) else arb(0)
                    for i in range(n_terms)
                ])
            else:
                H_series = acb_series(H_cols[j])
                h_e = h_acbs[j] if h_acbs is not None else _mpf_to_acb(h_mpf)
                F_series = (h_e * log_q16N_series).exp() * base_series * H_series
                coeffs = list(F_series.coeffs())
                out.append([
                    coeffs[i].real if i < len(coeffs) else arb(0)
                    for i in range(n_terms)
                ])
        return out

    def evaluate_pole_laurent(
        d_pole,
        *,
        principal_order,
        regular_order,
        matched_locations=None,
        allow_regular_point=False,
        matched_indices=None,
    ):
        """Analytic chiral Laurent data at/near a Kac pole ``h = d_pole``.

        The chiral block coefficient ``F_m(h)`` is the explicit rational
        ``P(h) * (H(h) @ Q)_m`` with ``H_k(h) = sum_idx M[k,idx] K[idx]/(h-d[idx])``
        and ``P(h) = (16 q)^h N^{-h} BASE'`` entire in ``h``. At a Kac weight the
        pole is simple (coincident Kac weights merely sum their residues), so the
        whole principal part is captured by two coordinate-Taylor series:

          F_m(h) = res_m / (h - d_pole) + reg_m + O(h - d_pole),

        with  res = P(d) * (H_res @ Q),  H_res[k] = sum_{idx in S} M[k,idx] K[idx]
        (S = recursion indices at this location), and
          reg = P(d) * (H_reg @ Q) + P'(d) * (H_res @ Q),
          H_reg[k] = sum_{idx not in S} M[k,idx] K[idx]/(d_pole - d[idx]),
          P'(d) = P(d) * (log(16 q) - log N).

        Returns ``(principal, regular)`` where ``principal[j-1]`` is the
        coefficient vector of ``(h-d_pole)^(-j)`` for ``j=1..principal_order`` and
        ``regular[q]`` is the coefficient vector of ``(h-d_pole)^q`` for
        ``q=0..regular_order`` (all length-``n_terms`` acb lists), or ``None`` for
        an unsupported context mode / no matching Kac index.

        Supported for the strip-real (1 < c < 25) and real-nonstrip (c > 25)
        modes: in both the Kac weights are real (real double poles at ell=0 /
        real simple poles at ell!=0) and the principal-part assembly is
        identical. The strip mode stores acb pole data (complex b on the unit
        circle); the real-nonstrip mode stores arb data (everything real),
        promoted to acb here so the assembly path is shared."""
        if _arb_ctx is None:
            return None
        if _arb_ctx_mode == "strip_real":
            K = _arb_ctx["K_full"]; d = _arb_ctx["d_full"]
            K_zero = _arb_ctx["K_zero_full"]
            n_pairs = _arb_ctx["n_pairs_full"]
            M_acb = _arb_ctx["M_full"]
            _to_mpc = _acb_to_mpc
            _to_acb_scalar = lambda v: v  # already acb
        elif _arb_ctx_mode == "real_nonstrip":
            # Real Kac data (c > 25): K / d are arb, M is an arb_mat -- promote
            # to acb so the (complex-Delta-capable) assembly below is shared with
            # the strip path. The promotion is exact (acb(arb)).
            K = _arb_ctx["K"]; d = _arb_ctx["d"]
            K_zero = _arb_ctx["K_zero"]
            n_pairs = _arb_ctx["n_pairs"]
            M_arb = _arb_ctx["M"]
            M_acb = None  # promoted once and cached below
            _to_mpc = lambda v: mp.mpc(_arb_to_mpf(v))
            _to_acb_scalar = acb  # promote arb -> acb
        elif _arb_ctx_mode == "complex_nonstrip":
            # Genuinely complex Kac data (e.g. the singular-c regularization
            # probe c = 1 + eps, where b ~ i carries a spurious ~1e-31 real
            # part). K / d are already acb and M is an acb_mat, so the shared
            # complex-Delta-capable assembly below applies directly.
            K = _arb_ctx["K_acb"]; d = _arb_ctx["d_acb"]
            K_zero = _arb_ctx["K_zero"]
            n_pairs = _arb_ctx["n_pairs"]
            M_acb = _arb_ctx["M_acb"]
            _to_mpc = _acb_to_mpc
            _to_acb_scalar = lambda v: v  # already acb
        elif _arb_ctx_mode == "merged":
            # AT the singular c: each column is a merged pole 1/(h - d0)^order (build_merged_H_context), so the
            # principal part at d0 is its columns there and the regular part the Taylor expansion of the others;
            # the column "residue" is 1 and the shared assembly below applies with these weights.
            n_pairs = _arb_ctx["n_cols"]
            d = [acb(x) for x in _arb_ctx["d0"]]
            K = [acb(1)] * n_pairs
            K_zero = [False] * n_pairs
            M_acb = _arb_ctx["M"] if not _arb_ctx["real"] else None
            M_arb = _arb_ctx["M"]
            _to_mpc = _acb_to_mpc
            _to_acb_scalar = lambda v: v
        else:
            return None
        principal_order = int(principal_order)
        regular_order = int(regular_order)
        if principal_order < 1 or regular_order < 0:
            return None

        # Per-evaluator cache (keyed on the immutable Kac context): the coarse
        # Python-complex Kac weights for the float prefilter, the acb-promoted
        # per-pair K / d scalars, and the acb H-table. All are fixed for this
        # evaluator, so rebuilding them on every call -- re-promoting
        # ~q_order*n_pairs arb scalars (real_nonstrip) and reconverting every Kac
        # weight to mpmath for the cluster match -- is the dominant wasted cost
        # in the profile.
        _plc = getattr(evaluate, "_pole_laurent_cache", None)
        if _plc is None or _plc["ctx"] is not _arb_ctx:
            d_cplx = [None] * n_pairs
            K_acb = [None] * n_pairs
            d_acb = [None] * n_pairs
            for idx in range(n_pairs):
                if K_zero[idx]:
                    continue
                de = d[idx]
                K_acb[idx] = _to_acb_scalar(K[idx])
                d_acb[idx] = _to_acb_scalar(de)
                if _arb_ctx_mode == "real_nonstrip":
                    d_cplx[idx] = complex(float(de), 0.0)
                else:
                    d_cplx[idx] = complex(float(de.real.mid()), float(de.imag.mid()))
            if M_acb is None:  # real_nonstrip: promote the H-table once
                M_acb = acb_mat(
                    [[acb(M_arb[r, cc]) for cc in range(M_arb.ncols())]
                     for r in range(M_arb.nrows())]
                )
            _plc = {"ctx": _arb_ctx, "d_cplx": d_cplx, "K_acb": K_acb,
                    "d_acb": d_acb, "M_acb": M_acb}
            evaluate._pole_laurent_cache = _plc
        d_cplx = _plc["d_cplx"]
        K_acb = _plc["K_acb"]
        d_acb = _plc["d_acb"]
        M_acb = _plc["M_acb"]

        M = M_acb
        flint_ctx.prec = _flint_prec_bits()
        flint_ctx.cap = n_terms
        dp = _mpf_to_acb(d_pole)
        dp_c = _acb_to_mpc(dp)
        dp_cplx = complex(float(dp.real), float(dp.imag))
        # Snap tolerance: loose enough to absorb cross-build rounding between the
        # caller's pole list and this evaluator's Kac weights (~1e-30), but far
        # below the Kac-weight spacing (~1e-2), so the matched cluster is
        # unambiguous and all genuinely-coincident weights land together in S.
        match_tol = mp.mpf(10) ** (-(mp.mp.dps // 3))
        # Coarse float gap for the cluster prefilter: >> match_tol (~1e-20) yet
        # << the Kac-weight spacing (~1e-2). A float miss is a guaranteed
        # non-match (skip the costly mpmath confirm); every genuine match still
        # clears it and is then confirmed exactly with match_tol, so the result
        # is identical to the all-mpmath test.
        coarse_tol = 1e-6

        matched_targets = None
        matched_cplx = None
        if matched_locations is not None:
            matched_targets = [mp.mpc(v) for v in matched_locations]
            if len(matched_targets) == 0:
                return None
            matched_cplx = [complex(t) for t in matched_targets]

        # Match detection first, in floats (with the exact mpmath confirm only
        # on the few coarse hits). A regular point -- the common case when the
        # analytic contribution scorer probes BOTH chiral halves of every
        # Delta-pole -- is rejected here for one float subtraction per pair,
        # before any acb weight-column work is spent on it.
        match_idx = set()
        for idx in (() if matched_indices is not None else range(n_pairs)):
            if K_zero[idx]:
                continue
            dc = d_cplx[idx]
            # Coarse float reject first; only the (few) coarse hits pay the exact
            # mpmath confirm -- the float gap is unambiguous (see coarse_tol).
            if matched_targets is None:
                if (
                    abs(dc - dp_cplx) <= coarse_tol
                    and abs(_to_mpc(d[idx]) - dp_c) <= match_tol
                ):
                    match_idx.add(idx)
            else:
                if any(abs(dc - tc) <= coarse_tol for tc in matched_cplx):
                    d_mpc = _to_mpc(d[idx])
                    if any(abs(d_mpc - t) <= match_tol for t in matched_targets):
                        match_idx.add(idx)
        if matched_indices is not None:
            match_idx = {int(i) for i in matched_indices if not K_zero[int(i)]}
        n_match = len(match_idx)
        if n_match == 0 and not allow_regular_point:
            # No Kac weight at d_pole. By default this is "not a pole" -> None.
            # With allow_regular_point, we continue instead and treat d_pole as
            # an ordinary analytic point: every weight then goes through the
            # regular (Taylor) branch below, so the principal columns are
            # identically zero and the regular columns hold the full Taylor
            # expansion of F at d_pole. This lets callers obtain higher-order
            # Taylor coefficients of the chiral factor at a regular point
            # (needed when one chiral half of a merged block pole is
            # non-singular but the other is a higher-order pole).
            return None

        # Split the pole weights into singular (the matched Kac cluster S, whose
        # residue is K) and regular (every other pole, K/(d_pole - d)) parts, then
        # push each through the SAME compiled H-table matmul the recursion uses
        # (M @ w). This keeps the per-pole work to two acb_mat products rather
        # than an O(n_pairs * q_order) Python loop.
        zero = acb(0)
        n_pr = principal_order
        n_rg = regular_order + 1
        n_cols = n_pr + n_rg

        # Build the per-pole weight matrix column-major (n_cols rows of length
        # n_pairs) so it can feed the collapsed matmul below directly as
        # ``acb_mat(cols)`` -- a (n_cols x n_pairs) matrix -- with no transpose.
        one_acb = acb(1)
        cols = [[zero] * n_pairs for _ in range(n_cols)]
        merged_power = _arb_ctx["power"] if _arb_ctx_mode == "merged" else None
        for idx in range(n_pairs):
            if K_zero[idx]:
                continue
            K_e = K_acb[idx]
            d_e = d_acb[idx]
            if merged_power is not None:
                a = merged_power[idx]
                if idx in match_idx:
                    if a > n_pr:
                        raise ValueError(f"pole_laurent: a merged pole of order {a} at {mp.nstr(d_pole, 8)} needs "
                                         f"principal_order >= {a} (got {n_pr})")
                    cols[a - 1][idx] = one_acb
                else:
                    # (h - d)^-a = sum_q binom(-a, q) (d_pole - d)^(-a-q) (h - d_pole)^q
                    inv = one_acb / (dp - d_e)
                    term = inv ** a
                    for q in range(n_rg):
                        cols[n_pr + q][idx] = term
                        term = term * inv * (-(a + q)) / (q + 1)
                continue
            if idx in match_idx:
                cols[0][idx] = K_e
                dpow = d_e - dp
                for j in range(1, n_pr):
                    cols[j][idx] = K_e * dpow
                    dpow = dpow * (d_e - dp)
            else:
                # term_q = K_e * (-1)^q * inv^(q+1), inv = 1/(d_pole - d_e);
                # each higher order is the previous term times -inv.
                inv = one_acb / (dp - d_e)
                term = K_e * inv
                cols[n_pr][idx] = term
                for q in range(1, n_rg):
                    term = -term * inv
                    cols[n_pr + q][idx] = term
        _q16, _base, _Qm, _log_q16N = _complex_series_data()
        # Collapse the two per-call matmuls (M @ Wcol, then H_rows @ Q) into a
        # single product against a precomputed, per-evaluator matrix
        #   A_cache = (M with row 0 zeroed)^T @ Q          (n_pairs x n_terms).
        # The k=0 "identity injection" row (row0 = e_{n_pr}, which overrides the
        # M@Wcol value at k=0) only feeds Q[0,:] into output row n_pr, so it is
        # added back as the cached vector Qm0 = Q[0,:].  Both A_cache and Qm0
        # depend solely on the fixed H-table M and series matrix Q, so they are
        # built once per evaluator (rebuilt only if Q's precision changes).
        if _plc.get("A_cache") is None or _plc.get("Qm_id") != id(_Qm):
            Mt = M.transpose()                     # (n_pairs x q1)
            for _r in range(n_pairs):
                Mt[_r, 0] = zero                   # zero row 0 of M (col 0 of M^T)
            _plc["A_cache"] = Mt * _Qm             # (n_pairs x n_terms)
            _plc["Qm0"] = [_Qm[0, _i] for _i in range(n_terms)]
            _plc["Qm_id"] = id(_Qm)
        A_cache = _plc["A_cache"]
        Qm0 = _plc["Qm0"]
        WcolT = acb_mat(cols)                      # (n_cols x n_pairs)
        HQ = WcolT * A_cache                       # (n_cols x n_terms)
        for _i in range(n_terms):                  # add the identity-injection row
            HQ[n_pr, _i] = HQ[n_pr, _i] + Qm0[_i]
        H_pr = [
            acb_series([HQ[j, i] for i in range(n_terms)]) for j in range(n_pr)
        ]
        H_rg = [
            acb_series([HQ[n_pr + q, i] for i in range(n_terms)])
            for q in range(n_rg)
        ]

        # P = (16 q / N)^dp * BASE' and dlogP = d/dh log P = log(16 q) - log N,
        # both from the N-folded ``_log_q16N`` (no separate N^{-dp} factor).
        P = (dp * _log_q16N).exp() * _base
        return compose_prefactor_laurent(P, _log_q16N, H_pr, H_rg, n_terms)

    def evaluate_pole_data(d_pole):
        """Compatibility wrapper: simple-pole residue+regular value.

        Equivalent to ``evaluate.pole_laurent(d_pole, principal_order=1,
        regular_order=0)`` and returns ``(res, reg)``.
        """
        out = evaluate_pole_laurent(
            d_pole, principal_order=1, regular_order=0
        )
        if out is None:
            return None
        principal, regular = out
        return principal[0], regular[0]

    def evaluate_kac_locations():
        """Distinct real parts of active Kac weights, as mpmath numbers.

        Available in strip and real-nonstrip modes; return ``None`` for other
        contexts. In strip mode, conjugate poles share one real part. Use
        ``evaluate_kac_poles`` for full complex locations and multiplicities.
        """
        if _arb_ctx is None:
            return None
        if _arb_ctx_mode == "strip_real":
            d = _arb_ctx["d_full"]; K_zero = _arb_ctx["K_zero_full"]
            n_pairs = _arb_ctx["n_pairs_full"]
            _to_re = lambda v: mp.mpf(_acb_to_mpc(v).real)
        elif _arb_ctx_mode == "real_nonstrip":
            d = _arb_ctx["d"]; K_zero = _arb_ctx["K_zero"]
            n_pairs = _arb_ctx["n_pairs"]
            _to_re = _arb_to_mpf
        else:
            return None
        locs = []
        locs_f = []
        tol = mp.mpf(10) ** (-(mp.mp.dps // 2))
        # Coarse float gap for the dedup prefilter: >> tol yet << Kac-location
        # spacing (~1e-2). A float miss is a guaranteed new location (skip the
        # O(n) mpmath scan); coarse hits are confirmed exactly with tol, so the
        # deduplicated set is identical to the all-mpmath result.
        coarse_tol = 1e-9
        for idx in range(n_pairs):
            if K_zero[idx]:
                continue
            vr = _to_re(d[idx])
            vf = float(vr)
            if any(abs(vf - uf) <= coarse_tol for uf in locs_f):
                if any(abs(vr - u) <= tol for u in locs):
                    continue
            locs.append(vr)
            locs_f.append(vf)
        return sorted(locs)

    def evaluate_kac_poles():
        """Full Kac pole list (with multiplicities), as mpmath numbers: complex where the Kac weights are.

        Unlike :func:`evaluate_kac_locations`, this returns one value per
        non-vanishing recursion residue index, preserving multiplicities. Used
        by singular-c regularization that must track coincident clusters across
        probe ``c`` values. For 1 < c < 25 (strip mode), weights with r != s
        form complex-conjugate pairs; retain their imaginary parts to track
        each pole separately.
        """
        if _arb_ctx is None:
            return None
        if _arb_ctx_mode == "strip_real":
            d = _arb_ctx["d_full"]; K_zero = _arb_ctx["K_zero_full"]
            n_pairs = _arb_ctx["n_pairs_full"]
            _snap = mp.mpf(10) ** (5 - mp.mp.dps)

            def _to_out(v):
                # r = s weights are real: their imaginary part is rounding
                z = _acb_to_mpc(v)
                return mp.mpf(z.real) if abs(z.imag) <= _snap * max(1, abs(z)) else z
        elif _arb_ctx_mode == "real_nonstrip":
            d = _arb_ctx["d"]; K_zero = _arb_ctx["K_zero"]
            n_pairs = _arb_ctx["n_pairs"]
            _to_out = _arb_to_mpf
        elif _arb_ctx_mode == "complex_nonstrip":
            # Complex Kac data (singular-c regularization probe, e.g. c = 1):
            # return the genuine complex weights so the merged-Laurent scorer
            # can match clusters across probe c values.
            d = _arb_ctx["d_acb"]; K_zero = _arb_ctx["K_zero"]
            n_pairs = _arb_ctx["n_pairs"]
            _to_out = _acb_to_mpc
        else:
            return None
        poles = []
        for idx in range(n_pairs):
            if K_zero[idx]:
                continue
            poles.append(_to_out(d[idx]))
        return poles

    def evaluate_kac_labels():
        """``[(index, r, s)]`` of the non-vanishing recursion residues, in :func:`evaluate_kac_poles` order: the
        exact labels that identify each probe Kac weight with its limit at a singular c."""
        if _arb_ctx is None or _arb_ctx_mode not in ("strip_real", "real_nonstrip", "complex_nonstrip"):
            return None
        K_zero = _arb_ctx["K_zero_full"] if _arb_ctx_mode == "strip_real" else _arb_ctx["K_zero"]
        return [(idx, p["r"], p["s"]) for idx, p in enumerate(_pair_tbl.pairs) if not K_zero[idx]]

    def evaluate_merged_pole_order(d_pole):
        """At a singular c (merged mode): the order of the merged pole at ``d_pole`` (0 at a regular point)."""
        if _arb_ctx_mode != "merged":
            return None
        tol = mp.mpf(10) ** (-(mp.mp.dps // 3))
        dp = mp.mpc(d_pole)
        return max((a for x, a in zip(_arb_ctx["d0"], _arb_ctx["power"])
                    if abs(_acb_to_mpc(acb(x)) - dp) <= tol * max(1, abs(dp))), default=0)

    evaluate.merged_pole_order = evaluate_merged_pole_order
    evaluate.kac_labels = evaluate_kac_labels
    # the certified H fill's guard beyond the working precision (bits): how deeply the Kac data cancel here
    evaluate.certification_guard_bits = None if _arb_ctx is None else _arb_ctx.get("guard_bits")
    evaluate.pole_laurent = evaluate_pole_laurent
    evaluate.pole_data = evaluate_pole_data
    evaluate.kac_locations = evaluate_kac_locations
    evaluate.kac_poles = evaluate_kac_poles
    evaluate.many = evaluate_many
    evaluate.real_analytic = bool(_real_analytic)    # see its definition
    return evaluate


def compose_prefactor_laurent(P, dlogP, H_pr, H_rg, n_terms):
    """The Laurent data in the weight of ``F = P(h) H(h)`` at a pole ``h0``, from those of ``H`` (flint series in the
    coordinate; the caller has set ``flint_ctx.cap``).

    ``P(h) = P(h0) exp((h - h0) dlogP)`` (the ``(16 q / N)^h`` prefactor, ``P`` its value at ``h0`` times the
    h-independent base); ``H_pr[j]`` is the coefficient series of ``(h - h0)^-(j+1)``, ``H_rg[q]`` of
    ``(h - h0)^q``.  Returns ``(principal, regular)`` coefficient lists of length ``n_terms`` in the same layout.
    """
    from flint import acb, acb_series

    n_pr, n_rg = len(H_pr), len(H_rg)
    n_p_rows = n_pr + n_rg
    P_pows = [acb_series([acb(1)])]
    for _ in range(n_p_rows - 1):
        P_pows.append(P_pows[-1] * dlogP)
    P_rows = [
        P * P_pows[r] * (acb(1) / acb(mp.factorial(r)))
        for r in range(n_p_rows)
    ]

    F_pr = []
    for j in range(n_pr):
        acc = acb_series([acb(0)])
        for r in range(0, n_pr - j):
            acc = acc + P_rows[r] * H_pr[j + r]
        F_pr.append(acc)

    F_rg = []
    for q in range(n_rg):
        acc = acb_series([acb(0)])
        for r in range(0, q + 1):
            acc = acc + P_rows[r] * H_rg[q - r]
        for j in range(1, n_pr + 1):
            acc = acc + P_rows[q + j] * H_pr[j - 1]
        F_rg.append(acc)

    principal = []
    for ser in F_pr:
        coeffs = list(ser.coeffs())
        principal.append([
            coeffs[i] if i < len(coeffs) else acb(0)
            for i in range(n_terms)
        ])
    regular = []
    for ser in F_rg:
        coeffs = list(ser.coeffs())
        regular.append([
            coeffs[i] if i < len(coeffs) else acb(0)
            for i in range(n_terms)
        ])
    return principal, regular


def chiral_z_series_evaluator(
    h_externals: Sequence,
    n_terms: int,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
):
    """Return a callable ``h -> [F-Taylor coeffs around z = 1/2]`` (flint series).

    Analytic-series port of :func:`chiral_z_taylor`; see
    :func:`_chiral_series_evaluator` for the factorization. The returned
    coefficients match :func:`chiral_z_taylor` to working precision; see
    ``tests/core/test_derivs.py``.
    """
    return _chiral_series_evaluator(
        h_externals, n_terms, b=b, c=c, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift,
        z_of_u=None, u_symmetric=None, coord_key="z",
    )


def chiral_y_series_evaluator(
    h_externals: Sequence,
    n_terms: int,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
):
    """Return a callable ``h -> [F-Taylor coeffs around y = 0]`` (flint series).

    Analytic-series port of :func:`chiral_y_taylor` using the blocks_3d radial
    map ``z(y) = (1 + y)^2 / (2 (1 + y^2))``. Mirrors
    :func:`chiral_z_series_evaluator` for the y coordinate so the large-spin
    tower and any per-``h`` sweep get the same cached, flint-accelerated path
    in ``yyb`` that they already had in ``zzb``. The returned coefficients match
    :func:`chiral_y_taylor` to working precision.
    """
    return _chiral_series_evaluator(
        h_externals, n_terms, b=b, c=c, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift,
        z_of_u=z_of_y, u_symmetric=y_symmetric(), coord_key="y",
    )


def chiral_q_series_evaluator(
    h_externals: Sequence,
    n_terms: int,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
):
    """Return a callable ``h -> [F-Taylor coeffs around q = exp(-pi)]``.

    Mirrors :func:`chiral_z_series_evaluator` and
    :func:`chiral_y_series_evaluator` for the nome coordinate so per-``h``
    sweeps can reuse the cached geometry and flint-native H-series assembly.
    """
    return _chiral_series_evaluator(
        h_externals, n_terms, b=b, c=c, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift,
        z_of_u=z_of_q, u_symmetric=q_symmetric(), coord_key="q",
    )


def chiral_z_taylor_fast(
    h,
    h_externals: Sequence,
    n_terms: int,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
):
    """Single-``h`` convenience wrapper over :func:`chiral_z_series_evaluator`."""
    evaluate = chiral_z_series_evaluator(
        h_externals, n_terms, b=b, c=c, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift,
    )
    return evaluate(h)


def chiral_y_taylor_fast(
    h,
    h_externals: Sequence,
    n_terms: int,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
):
    """Single-``h`` convenience wrapper over :func:`chiral_y_series_evaluator`."""
    evaluate = chiral_y_series_evaluator(
        h_externals, n_terms, b=b, c=c, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift,
    )
    return evaluate(h)


def chiral_q_taylor_fast(
    h,
    h_externals: Sequence,
    n_terms: int,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
):
    """Single-``h`` convenience wrapper over :func:`chiral_q_series_evaluator`."""
    evaluate = chiral_q_series_evaluator(
        h_externals, n_terms, b=b, c=c, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift,
    )
    return evaluate(h)


def chiral_y_taylor(
    h,
    h_externals: Sequence,
    n_terms: int,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
):
    """Taylor coefficients of F(h, z(y); {h_i}; c) around y = 0."""
    return _chiral_taylor(
        h,
        h_externals,
        n_terms,
        b=b, c=c,
        q_order=q_order,
        z_of_u=z_of_y,
        u_symmetric=y_symmetric(),
        chiral_norm=chiral_norm,
        z_shift=z_shift,
    )


def chiral_q_taylor(
    h,
    h_externals: Sequence,
    n_terms: int,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
):
    """Taylor coefficients of F(h, z(q); {h_i}; c) around q = exp(-pi)."""
    return _chiral_taylor(
        h,
        h_externals,
        n_terms,
        b=b, c=c,
        q_order=q_order,
        z_of_u=z_of_q,
        u_symmetric=q_symmetric(),
        chiral_norm=chiral_norm,
        z_shift=z_shift,
    )


# Coordinates that evaluate_block can recenter: the non-folded 2d coordinates
# and the chiral-only coordinates. The folded / radial coordinates are excluded
# -- their parity fold ((u+ubar)/2 - u_sym, ((u-ubar)/2)^2) is defined relative
# to the symmetric point and assumes crossing z <-> 1-z acts as u <-> -u, which
# only holds at the symmetric center. Evaluate in the non-folded parent and fold
# yourself if you need a folded quantity.
_EVALUATE_BLOCK_KINDS = ("non_folded_2d", "chiral")


def evaluate_block(
    z,
    zbar,
    h,
    hbar,
    h_externals: Sequence,
    *,
    b=None,
    c=None,
    q_order: int,
    coord: str = "qqb",
    m_max: int = 0,
    n_max: int = 0,
    chiral_norm=None,
    z_shift=None,
):
    """Evaluate the (ordered) 2d Virasoro block at an arbitrary cross ratio.

    Returns the chiral-factorized block ``F(h, z) * F(hbar, zbar)`` -- the
    ordered product of the two chiral halves, with no parity projection -- at a
    point ``(z, zbar)`` that need not be the crossing-symmetric ``z = zbar =
    1/2``. ``z`` and ``zbar`` are independent (so Lorentzian / complex kinematics
    with ``z != zbar`` are fine).

    With ``m_max == n_max == 0`` (the default) the return value is the block
    *value* (an ``mp.mpf`` or ``mp.mpc``). With larger ``m_max`` / ``n_max`` the
    return is the ``(m_max+1) x (n_max+1)`` grid of derivatives *at that point*,
    taken in the chiral *coordinate* ``u`` (z for ``zzb``, y for ``yyb``, the
    nome q for ``qqb``) -- the same convention as the symmetric-point tables:
    ``grid[m][n] = d_u^m d_ubar^n F / m! n!`` (Taylor coefficients in
    ``(u - u(z), ubar - u(zbar))``), with ``grid[0][0]`` the value. To get
    z-derivatives, use ``coord="zzb"`` (where ``u = z``).

    This recenters the same chiral series machinery the symmetric-point tables
    use (``u_symmetric`` is just set to ``u(z)`` / ``u(zbar)``); it is not a new
    block computation. To form parity-projected combinations
    ``F(h,hbar) +/- F(hbar,h)`` etc., call this twice and combine.

    ``coord`` selects the chiral expansion variable. Use ``"qqb"`` (the nome,
    the default) for points far from ``z = 1/2``: the chiral series converges
    for ``|q| < 1``, i.e. essentially the whole physical z-plane, whereas the
    ``"zzb"`` / ``"yyb"`` series degrade as ``z -> 0, 1``. Folded / radial
    coordinates are rejected (see ``_EVALUATE_BLOCK_KINDS``).

    Real negative ``z`` lies on the block's branch cut (``z^h``, and
    ``K(1 - z)`` in the nome). The coordinates need not agree there on which
    side of the cut they land: ``yyb`` (through ``sqrt(z (1 - z))``) can
    return the other side's value than ``zzb`` / ``qqb`` (the complex-conjugate
    value, e.g. at ``z < -1``). Give ``z`` a small imaginary part to pick a side.

    Bare recursion: raises ValueError at a recognized recursion-singular c
    (`singularity.recognized_singular_locus`); use the regularized
    `F_block` / `B_block` there.
    """
    m_max, n_max = _check_table_orders(m_max, n_max)
    q_order = _integer_control('q_order', q_order)
    spec = COORDINATES.get(coord)
    if spec is None or spec.kind not in _EVALUATE_BLOCK_KINDS:
        raise ValueError(
            f"evaluate_block does not support coord={coord!r}; choose a "
            f"non-folded 2d coordinate (zzb/yyb/qqb) or a chiral one (z/y/q)"
        )
    b = _resolve_b(b, c)
    coord_key = spec.chiral_var  # "z" / "y" / "q"
    z_of_u = spec.z_of_u
    u_of_z = spec.u_of_z

    def _half(point, exch_h, n_terms):
        evaluate = _chiral_series_evaluator(
            h_externals, n_terms, b=b, q_order=q_order,
            chiral_norm=chiral_norm, z_shift=z_shift,
            z_of_u=z_of_u, u_symmetric=u_of_z(point), coord_key=coord_key,
        )
        return evaluate(exch_h)

    diagonal = zbar == z and hbar == h
    # on the diagonal one half serves both factors, so it must reach n_max too
    c_h = _half(z, h, (max(m_max, n_max) if diagonal else m_max) + 1)
    if diagonal:
        c_hbar = c_h
    else:
        c_hbar = _half(zbar, hbar, n_max + 1)

    grid = _assemble_block_grid(c_h, c_hbar, m_max, n_max, sigma=0)
    if m_max == 0 and n_max == 0:
        return grid[0][0]
    return grid


def _global_pole_index(h):
    """The exact p in 2h=-p, or None, without a proximity tolerance."""
    if isinstance(h, str):
        from .numeric_input import parse_number
        h = parse_number(h)
    if isinstance(h, (complex, mp.mpc)):
        if h.imag != 0 or h.real > 0:
            return None
    elif h > 0:
        return None
    from .numeric_input import literal_components

    real, imag = literal_components(h)
    twice = 2*real
    return int(-twice) if imag == 0 and twice.denominator == 1 else None


def _global_parameter_values(h, differences):
    """Form affine parameters from exact external differences at working precision."""
    h = mp.convert(h) if isinstance(h, complex) else h
    shifts = [real if imag == 0 else mp.mpc(mp.convert(real), mp.convert(imag))
              for real, imag in differences]
    return (h if shifts[0] == 0 else h+shifts[0],
            h if shifts[1] == 0 else h+shifts[1], 2*h)


def _global_raise_genuine_pole(h, h_externals, differences):
    p = _global_pole_index(h)
    if p is None:
        return None
    removable = _global_removable_parameters(h, h_externals, differences=differences, pole_index=p)
    if removable is not None:
        return removable
    from .requests import BlockPoleError

    raise BlockPoleError(family='global', basis='global_chiral_F', exchanged=h,
                         pole=ExactRational(-p, 2))


def _global_conditioning(h, h_externals, differences, *, what):
    """Report the digits lost to ``h + p/2`` at the nearest genuine (non-removable) global pole ``2h = -p``."""
    from .conditioning import pole_loss, report
    from .numeric_input import parse_number

    try:
        x = complex(mp.convert(parse_number(h)))
    except (OverflowError, ValueError, TypeError):
        return
    p = int(round(-2 * x.real))
    if p < 0 or abs(x.imag) > 1:
        return
    if _global_removable_parameters(h, h_externals, differences=differences, pole_index=p) is not None:
        return
    report(pole_loss(h, [ExactRational(-p, 2)]), what=what, source=h)


def _global_chiral_block(
    h,
    z,
    h_externals: Sequence,
    *,
    chiral_norm=None,
    z_shift=None,
):
    from .numeric_input import parse_number

    _check_externals(h_externals)
    h = parse_number(h)
    h_externals = tuple(h_externals)
    differences = _global_external_components(h_externals)
    removable = _global_raise_genuine_pole(h, h_externals, differences)
    _global_conditioning(h, h_externals, differences, what="global chiral block")
    return _global_chiral_block_at_work(
        h, z, h_externals, chiral_norm=chiral_norm, z_shift=z_shift,
        differences=differences, removable=removable,
    )


def _global_chiral_block_at_work(h, z, h_externals, *, chiral_norm, z_shift, differences, removable):
    from .conditioning import endpoint_loss, report
    from .numeric_input import parse_number

    z_label = parse_number(z)
    from .conditioning import _carries_extra_bits

    if removable is None and not _carries_extra_bits(z_label):
        # the generic 2F1 at a z rounded to the working precision loses log10(1/|1-z|) digits near z = 1 (an mpf z
        # stored at more bits keeps them, and the removable limits stay accurate there)
        report(endpoint_loss(z_label), what="global chiral block", source=z_label)
    z = mp.convert(z_label)
    h = mp.convert(h) if isinstance(h, complex) else h
    N = _resolve_chiral_norm(parse_number(chiral_norm))
    z_shift_val = parse_number(0 if z_shift is None else z_shift)
    if removable is not None:
        p, j, other = removable
        if p == 0 and other == (0, 0):
            if z_shift_val == 0:
                return mp.mpf(1)
            return mp.power(N, -h) * z ** (h + z_shift_val)
        polynomial, tail, b_tail = _global_limit_terms(p, j, other)
        hyp = mp.polyval(list(reversed(polynomial)), z)
        if tail != 0:
            hyp += tail * z ** (p + 1) * mp.hyp2f1(p + 1 - j, b_tail, p + 2, z)
        return mp.power(N, -h) * z ** (h + z_shift_val) * hyp
    return (
        mp.power(N, -h)
        * z ** (h if z_shift_val == 0 else h + z_shift_val)
        * mp.hyp2f1(*_global_parameter_values(h, differences), z)
    )


def _global_removable_parameters(h, h_externals, *, differences=None, pole_index=None):
    """Exact removable ``2h=-p`` data: terminating index and other numerator.

    The limit is along ``a=h+d12, b=h+d34, c=2h``. Substituting the
    non-positive integers directly into 2F1 instead takes a terminating
    convention that drops a nonzero tail when only one numerator vanishes.
    """
    # Positive weights cannot be poles. Keep their hot path free of exact
    # component conversion (the native 2F1 already handles them).
    p = _global_pole_index(h) if pole_index is None else pole_index
    if p is None:
        return None
    if differences is None:
        differences = _global_external_components(h_externals)
    if p == 0 and differences == ((0, 0), (0, 0)):
        return 0, 0, (0, 0)
    real = ExactRational(-p, 2)
    numerators = [(real + re, im) for re, im in differences]
    for index, (a, ai) in enumerate(numerators):
        if ai == 0 and a.denominator == 1 and -p <= a <= 0:
            return p, int(-a), numerators[1 - index]
    return None


def _global_external_components(h_externals):
    from .numeric_input import literal_components

    parts = tuple(literal_components(value) for value in h_externals)
    return _global_external_differences(parts)


@lru_cache(maxsize=512)
def _global_external_differences(parts):
    return tuple((parts[a][0] - parts[b][0], parts[a][1] - parts[b][1]) for a, b in ((1, 0), (2, 3)))


def _global_tail_vanishes(p, other):
    return other[1] == 0 and other[0].denominator == 1 and -p <= other[0] <= 0


def _global_limit_terms(p, j, other):
    """Polynomial and shifted 2F1 tail of the removable h-limit.

    For n>p, ``(-j+eps)_n/(-p+2eps)_n`` tends to
    ``(-1)**(j-p) j! (n-j-1)! / (2 p! (n-p-1)!)``.
    """
    b = mp.convert(other[0]) if other[1] == 0 else mp.mpc(mp.convert(other[0]), mp.convert(other[1]))
    polynomial = [mp.mpf(1)]
    for n in range(j):
        polynomial.append(polynomial[-1] * (-j + n) * (b + n) / ((-p + n) * (n + 1)))
    tail = mp.mpf(0) if _global_tail_vanishes(p, other) else (
        (-1) ** (p - j) * mp.factorial(j) * mp.factorial(p - j) * mp.rf(b, p + 1)
        / (2 * mp.factorial(p) * mp.factorial(p + 1)))
    return polynomial, tail, b + p + 1


def _global_hyp2f1_series(h, h_externals, z_series, *, differences=None):
    """2F1 in the coordinate, with the analytic h-limit at removable poles."""
    from flint import acb, acb_series

    A = _mpf_to_acb
    if differences is None:
        differences = _global_external_components(h_externals)
    _global_raise_genuine_pole(h, h_externals, differences)
    removable = _global_removable_parameters(h, h_externals, differences=differences)
    if removable is None:
        a, b, c = _global_parameter_values(h, differences)
        return _hyp2f1_series(A(a), A(b), A(c), z_series)
    p, j, other = removable
    if p == 0 and other == (0, 0):
        return acb_series([acb(1)])
    polynomial, tail, b_tail = _global_limit_terms(p, j, other)
    hyp = acb_series([A(polynomial[-1])])
    for coefficient in reversed(polynomial[:-1]):
        hyp = hyp * z_series + A(coefficient)
    if tail != 0:
        hyp += A(tail) * z_series ** (p + 1) * _hyp2f1_series(
            acb(p + 1 - j), A(b_tail), acb(p + 2), z_series)
    return hyp


_GLOBAL_HYP_ACCURACY_BITS = ContextVar('global_hyp_accuracy_bits', default=None)


def _hyp2f1_ode_seeds(a, b, c, z0, *, regularized, flags):
    """Resolve scalar seeds before building a coordinate Taylor series.

    Large complex parameters can exhaust the scalar routine's truncation or
    cancellation guard as well. Keep its rigorous balls and increase only
    the seed precision; the coordinate geometry and source labels stay fixed.
    """
    from flint import ctx as flint_ctx
    from mpmath.libmp import prec_to_dps
    from .poles import next_certification_prec
    from .recursion import _relative_deficit

    original = flint_ctx.prec
    # Affine-input guards can be much larger than the requested accuracy.
    # Certifying to that entire guard would demand resolving rounding balls
    # beyond the precision at which those inputs were supplied.
    accuracy = _GLOBAL_HYP_ACCURACY_BITS.get() or mp.mp.prec
    target = accuracy + 10
    guard_base = max(original, target)
    work, previous = max(original, target + 32), None
    try:
        while True:
            flint_ctx.prec = work
            t0 = z0.hypgeom_2f1(a, b, c, regularized=regularized, **flags)
            shifted = z0.hypgeom_2f1(a + 1, b + 1, c + 1, regularized=regularized, **flags)
            t1 = a * b * shifted if regularized else a * b / c * shifted
            deficit = _relative_deficit([t0, t1], target)
            if deficit == 0:
                return t0, t1
            loss = deficit if deficit < target else 2 * guard_base
            step = next_certification_prec(work, loss, previous, guard_base)
            if step is None:
                raise ValueError(
                    f"hypergeometric Taylor seeds cannot be resolved to {prec_to_dps(accuracy)} digits "
                    f"with {work-original} guard bits")
            work, previous = step
    finally:
        flint_ctx.prec = original


def _hyp2f1_series(a, b, c, z_series, *, regularized=False):
    """``2F1(a, b; c; z(u))`` as an acb power series in ``u``, for acb parameters ``a, b, c``.

    Near the origin (``|z(u0)| <= 3/4``) first try flint's hypergeometric series.  Beyond that the series in ``z``
    converges slowly and then not at all: flint's rigorous radius grows but its midpoint was read as the value (30
    digits became 6 at ``z = 0.99``; ``|z| >= 1`` gave ``[+/- inf]``).  There the Taylor series about ``z0 = z(u0)`` is
    built from the hypergeometric ODE, seeded by the analytically continued ``2F1`` and its derivative (flint's
    ``acb.hypgeom_2f1``, principal branch), and composed with ``z(u) - z0``.
    The same ODE fallback handles a non-finite direct series at large parameters,
    where the automatic truncation can fail even well inside the unit disk.
    """
    from flint import acb, acb_series, arb, ctx as flint_ctx

    z0 = z_series.coeffs()[0] if z_series.coeffs() else acb(0)
    terminating = any(x.imag == 0 and x.real.is_integer() and x.real <= 0 for x in (a, b))
    if terminating or abs(z0).upper() <= 0.75:      # (a terminating series is a polynomial: exact at any z)
        result = acb_series.hypgeom([acb_series([a]), acb_series([b])], [acb_series([c])], z_series,
                                   regularized=regularized)
        if all(coefficient.is_finite() for coefficient in result.coeffs()) or z0.is_zero():
            # The ODE has a regular singular point at zero. Its defining
            # series already handles that point, including parameter poles.
            return result
    n = flint_ctx.cap
    # Exact (midpoint) parameters, with the integer-difference flags flint's connection formulas need: identical
    # externals put c - a - b = 0 exactly in exact arithmetic but only to rounding here, and flint then took the
    # generic formula through Gamma(c - a - b) and returned [+/- 15.4] (read as 0).  A difference within rounding of
    # an integer is made exactly that integer (a shift far below the working precision), so the limit formula
    # applies exactly (flagged but off by 1e-31 it kept 6 digits).
    a, b, c, z0 = a.mid(), b.mid(), c.mid(), z0.mid()
    tol = arb(10) ** (-(mp.mp.dps - 10))

    def integer_offset(x):
        """The integer ``x`` is within rounding of, or None."""
        value = float(x.real.mid())
        if abs(value) > 2 ** 40 or not (x.imag.mid().is_zero() or abs(x.imag) < tol):
            return None
        k = round(value)
        return k if abs(x - k) < tol else None

    k_ab = integer_offset(a - b)
    if k_ab is not None:
        b = a - k_ab
    k_abc = integer_offset(c - a - b)
    k_ac = integer_offset(a - c)
    if k_abc is not None:
        c = a + b + k_abc
    elif k_ac is not None:
        c = a - k_ac
    elif integer_offset(b - c) is not None:
        c = b - integer_offset(b - c)
    flags = dict(ab=k_ab is not None, ac=integer_offset(a - c) is not None, bc=integer_offset(b - c) is not None,
                 abc=integer_offset(c - a - b) is not None)
    t0, t1 = _hyp2f1_ode_seeds(a, b, c, z0, regularized=regularized, flags=flags)
    # z(1-z) y'' + [c - (a+b+1) z] y' - ab y = 0 about z0, for y = sum t_k w^k:
    #   P0 (k+2)(k+1) t_{k+2} = -[(P1 k (k+1) + q0 (k+1)) t_{k+1} + (-k(k-1) - (a+b+1) k - ab) t_k]
    P0, P1, q0 = z0 * (1 - z0), 1 - 2 * z0, c - (a + b + 1) * z0
    t = [t0, t1]
    for k in range(n - 2):
        t.append(-((P1 * k * (k + 1) + q0 * (k + 1)) * t[k + 1]
                   + (-k * (k - 1) - (a + b + 1) * k - a * b) * t[k]) / (P0 * (k + 2) * (k + 1)))
    shift = z_series.coeffs()[1:]
    if n <= 1 or not shift:
        return acb_series(t[:1])      # (flint refuses to compose with the zero series)
    return acb_series(t[:n])(acb_series([acb(0), *shift]))


def _global_chiral_taylor_flint(
    h, h_externals, n_terms, *, z_of_u, u_symmetric, chiral_norm, z_shift,
):
    from .numeric_input import parse_number

    h = parse_number(h)
    h_externals = tuple(parse_number(value) for value in h_externals)
    differences = _global_external_components(h_externals)
    _global_raise_genuine_pole(h, h_externals, differences)
    _global_conditioning(h, h_externals, differences, what="global chiral Taylor")
    token = _GLOBAL_HYP_ACCURACY_BITS.set(mp.mp.prec)
    try:
        return _global_chiral_taylor_flint_at_work(
            h, h_externals, n_terms, z_of_u=z_of_u, u_symmetric=u_symmetric,
            chiral_norm=chiral_norm, z_shift=z_shift,
        )
    finally:
        _GLOBAL_HYP_ACCURACY_BITS.reset(token)


def _global_chiral_taylor_flint_at_work(
    h, h_externals, n_terms, *, z_of_u, u_symmetric, chiral_norm, z_shift,
):
    """Compute the global chiral-block Taylor coefficients via flint
    `acb_series`, returning a list of `n_terms` mpmath coefficients.

    Builds the whole block

        N^{-h} z(u)^{h + z_shift} 2F1(h+h2-h1, h+h3-h4; 2h; z(u))

    as a single compiled-C power series in the coordinate `u`, instead of the
    O(10x dps) numerical differentiation `taylor_coefficients` needs.
    """
    from flint import arb, acb, acb_series, ctx as flint_ctx

    from .numeric_input import parse_number

    h = mp.convert(h) if isinstance(h, complex) else h
    N = _resolve_chiral_norm(parse_number(chiral_norm))
    z_shift_val = parse_number(0 if z_shift is None else z_shift)

    old_cap = flint_ctx.cap
    flint_ctx.cap = n_terms
    try:
        flint_ctx.prec = int(mp.mp.dps * 3.33) + 20
        ndig = mp.mp.dps + 5

        A = _mpf_to_acb       # exact (a string round trip kept only 5 guard digits)

        u0 = A(u_symmetric)
        if z_of_u is z_of_q:
            # Nome coordinate: z(q) = 16 q (theta2_reduced)^4 / theta3^4, the
            # same theta-series map used by the Virasoro geometry builder
            # (_flint_geometry_series_coeffs), expanded in q around q_*.
            q_series = acb_series([u0, acb(1)])
            from .theta_series import theta_series
            theta2_reduced, theta3_z, _ = theta_series(q_series, full=True)
            z_series = (
                acb_series([acb(16)]) * q_series
                * (theta2_reduced ** 4) / (theta3_z ** 4)
            )
        elif z_of_u is z_of_y:
            # blocks_3d radial map: z(y) = (1 + y)^2 / (2 * (1 + y^2))
            y_series = acb_series([u0, acb(1)])
            one_plus_y = acb_series([acb(1) + u0, acb(1)])
            one_series = acb_series([acb(1)])
            z_series = (one_plus_y * one_plus_y) / (
                acb_series([acb(2)]) * (one_series + y_series * y_series)
            )
        else:
            # z-coordinate (identity): z = u
            z_series = acb_series([u0, acb(1)])

        exp_z = acb_series([A(h + z_shift_val)])

        z_pow = (exp_z * z_series.log()).exp()           # z^{h + z_shift}
        hyp = _global_hyp2f1_series(h, h_externals, z_series)
        prefactor = acb_series([A(mp.power(N, -h))])      # N^{-h}
        block = prefactor * z_pow * hyp

        coeffs = block.coeffs()
        out = []
        for k in range(n_terms):
            if k >= len(coeffs):
                out.append(mp.mpf(0))
                continue
            ck = coeffs[k]
            re = mp.mpf(ck.real.str(ndig, radius=False))
            im = mp.mpf(ck.imag.str(ndig, radius=False))
            out.append(mp.mpc(re, im) if im != 0 else re)
        return out
    finally:
        flint_ctx.cap = old_cap


def _global_chiral_series_evaluator(
    h_externals, n_terms, *, z_of_u, u_symmetric, chiral_norm, z_shift,
):
    """Batched flint-native evaluator ``h -> [global chiral Taylor coeffs]``.

    The global-block analogue of :func:`chiral_z_series_evaluator`: the
    h-independent coordinate geometry (the ``z(u)`` series and its log) is built
    once, and each weight only adds the ``z^{h + z_shift}``, ``2F1``, and
    ``N^{-h}`` factors. ``evaluate(h)`` returns mpmath coefficients (matching
    :func:`global_chiral_z_taylor`); ``evaluate(h, return_acb=True)`` returns the
    raw flint ``acb`` coefficients; ``evaluate.many(h_list)`` returns a list of
    real ``arb`` coefficient arrays when *every* weight (and the externals) is
    real -- the interpolation's arb-native fast path -- or ``None`` to signal the
    caller to fall back (complex data)."""
    from flint import arb, acb, acb_series, ctx as flint_ctx

    from .numeric_input import parse_number

    n_terms = _integer_control('n_terms', n_terms, 1)
    _check_externals(h_externals)
    h_externals = tuple(parse_number(value) for value in h_externals)
    norm_source = parse_number(chiral_norm)
    _resolve_chiral_norm(norm_source)  # Validate at construction, retain its source.
    z_shift_val = parse_number(0 if z_shift is None else z_shift)
    ext_real = not any(isinstance(x, (mp.mpc, complex)) and x.imag != 0 for x in h_externals)
    differences = _global_external_components(h_externals)

    @lru_cache(maxsize=4)
    def _build_geometry(ndig):
        A = _mpf_to_acb       # exact (a string round trip kept only 5 guard digits)

        u0 = A(u_symmetric)
        if z_of_u is z_of_q:
            q_series = acb_series([u0, acb(1)])
            from .theta_series import theta_series
            theta2_reduced, theta3_z, _ = theta_series(q_series, full=True)
            z_series = (
                acb_series([acb(16)]) * q_series
                * (theta2_reduced ** 4) / (theta3_z ** 4)
            )
        elif z_of_u is z_of_y:
            y_series = acb_series([u0, acb(1)])
            one_plus_y = acb_series([acb(1) + u0, acb(1)])
            one_series = acb_series([acb(1)])
            z_series = (one_plus_y * one_plus_y) / (
                acb_series([acb(2)]) * (one_series + y_series * y_series)
            )
        else:
            z_series = acb_series([u0, acb(1)])
        return A, z_series, z_series.log()

    def _coeffs_acb(h, A, z_series, log_z, n):
        h = mp.convert(h) if isinstance(h, complex) else h
        N = _resolve_chiral_norm(norm_source)
        exp_z = acb_series([A(h + z_shift_val)])
        z_pow = (exp_z * log_z).exp()
        hyp = _global_hyp2f1_series(h, h_externals, z_series, differences=differences)
        prefactor = acb_series([A(mp.power(N, -h))])
        block = (prefactor * z_pow * hyp).coeffs()
        return [block[k] if k < len(block) else acb(0) for k in range(n)]

    def evaluate(h, return_arb=False, return_acb=False):
        _global_conditioning(h, h_externals, differences, what="global chiral series evaluator")
        old_cap = flint_ctx.cap
        flint_ctx.cap = n_terms
        try:
            flint_ctx.prec = _flint_prec_bits()
            ndig = mp.mp.dps + 5
            A, z_series, log_z = _build_geometry(ndig)
            coeffs = _coeffs_acb(h, A, z_series, log_z, n_terms)
            if return_acb:
                return coeffs
            h_real = not (isinstance(h, (mp.mpc, complex)) and h.imag != 0)
            if return_arb:
                if not (ext_real and h_real) or any(c.imag != 0 for c in coeffs):
                    return None
                return [c.real for c in coeffs]
            out = []
            for ck in coeffs:
                re = _arb_to_mpf_checked(ck.real, ndig)
                im = _arb_to_mpf_checked(ck.imag, ndig)
                out.append(mp.mpc(re, im) if im != 0 else re)
            return out
        finally:
            flint_ctx.cap = old_cap

    def evaluate_many(h_list):
        h_list = list(h_list)
        for h in h_list:
            _global_conditioning(h, h_externals, differences, what="global chiral series evaluator")
        if not ext_real or any(
            isinstance(h, (mp.mpc, complex)) and h.imag != 0 for h in h_list
        ):
            return None
        old_cap = flint_ctx.cap
        flint_ctx.cap = n_terms
        try:
            flint_ctx.prec = _flint_prec_bits()
            ndig = mp.mp.dps + 5
            A, z_series, log_z = _build_geometry(ndig)
            rows = []
            for h in h_list:
                coeffs = _coeffs_acb(h, A, z_series, log_z, n_terms)
                # Real parameters can still give complex coefficients at an
                # off-real center or across a hypergeometric branch cut.
                if any(c.imag != 0 for c in coeffs):
                    return None
                rows.append([c.real for c in coeffs])
            return rows
        finally:
            flint_ctx.cap = old_cap

    def _global_reg_2f1(a_par, b_par, c_par, z_series, n):
        """Regularized ``2F1(a, b; c; z(u))`` as an acb series in the coordinate;
        entire in the parameters, so it stays finite at the ``c = -p`` poles."""
        return _hyp2f1_series(a_par, b_par, c_par, z_series, regularized=True)

    def evaluate_pole_data(d_pole):
        """Analytic chiral Laurent data at a global-block h-pole ``h = d_pole``.

        The global chiral block ``F(h, z) = N^{-h} z^{h+z_shift} 2F1(h+d12,
        h+d34; 2h; z)`` has simple poles in ``h`` at ``h = -p/2`` (``2h = -p``,
        ``p = 0, 1, 2, ...``) coming from the ``Gamma(2h)`` of the hypergeometric.
        Writing ``2F1 = Gamma(2h) * 2F1reg`` with ``2F1reg`` entire, the Laurent
        of ``g = 2F1`` about ``loc = -p/2`` (``eps = h - loc``, ``2h = -p + 2eps``)
        is

          res_g = (-1)^p/p! * R(loc)/2,
          reg_g = (-1)^p/p! * (R'(loc)/2 + psi(p+1) R(loc)),

        with ``R(h) = 2F1reg(h+d12, h+d34, 2h, z)`` (DLMF 15.2). ``R(loc)`` is the
        regularized 2F1 at ``c = -p`` (exact, no blow-up); ``R'(loc)`` (the
        ``d/dh`` parameter derivative) is a central difference on the
        well-conditioned regularized function, taken at elevated working
        precision. ``F = P * g`` with ``P = N^{-h} z^{h+z_shift}`` entire, so
        ``res_F = P(loc) res_g``, ``reg_F = P(loc) reg_g + P'(loc) res_g``,
        ``P'(loc) = P(loc)(log z - log N)`` -- the same assembly as the Virasoro
        primitive. Returns ``(res, reg)`` acb lists, or ``None`` when ``d_pole``
        is not a genuine (non-removable) ``-p/2`` pole."""
        from flint import acb_series as _acb_series
        p = _global_pole_index(d_pole)
        if p is None:
            return None
        loc = ExactRational(-p, 2)
        a_star, b_star, _ = _global_parameter_values(loc, differences)
        N = _resolve_chiral_norm(norm_source)
        if _global_removable_parameters(loc, h_externals, differences=differences) is not None:
            return None

        prec_bits = _flint_prec_bits()
        flint_ctx.prec = prec_bits
        flint_ctx.cap = n_terms
        ndig = mp.mp.dps + 5
        A, z_series, log_z = _build_geometry(ndig)

        R_loc = _global_reg_2f1(
            A(a_star), A(b_star), A(-mp.mpf(p)), z_series, n_terms
        )

        # R'(loc) = d/dh 2F1reg(h+d12, h+d34, 2h, z): central difference on the
        # entire, well-conditioned regularized function, under elevated working
        # precision so the perturbation survives the mpmath round-off.
        base_dps = mp.mp.dps
        with mp.workdps(2 * base_dps + 20):
            flint_ctx.prec = _flint_prec_bits()
            A2, z2, _log2 = _build_geometry(mp.mp.dps + 5)
            delta = mp.mpf(10) ** (-(base_dps + 10))
            def _Rh(h):
                a, b, c = _global_parameter_values(h, differences)
                return _hyp2f1_series(A2(a), A2(b), A2(c), z2, regularized=True)
            # The parameter finite difference consumes these guard digits;
            # its seeds need the elevated accuracy, not the outer target.
            token = _GLOBAL_HYP_ACCURACY_BITS.set(mp.mp.prec)
            try:
                R_hi = _Rh(loc + delta)
                R_lo = _Rh(loc - delta)
            finally:
                _GLOBAL_HYP_ACCURACY_BITS.reset(token)
            inv = A2(mp.mpf(1) / (2 * delta))
            Rp_hi = [(R_hi.coeffs() + [0] * n_terms)[i] for i in range(n_terms)]
            Rp_lo = [(R_lo.coeffs() + [0] * n_terms)[i] for i in range(n_terms)]
            Rp_mpc = [
                _acb_to_mpc((Rp_hi[i] - Rp_lo[i]) * inv) for i in range(n_terms)
            ]
        flint_ctx.prec = prec_bits
        flint_ctx.cap = n_terms
        Rp = _acb_series([_mpf_to_acb(v) for v in Rp_mpc])

        sgn = mp.mpf((-1) ** p)
        fac = _mpf_to_acb(sgn / mp.factorial(p))
        half = _mpf_to_acb(mp.mpf(1) / 2)
        psi = _mpf_to_acb(mp.digamma(p + 1))
        res_g = fac * half * R_loc
        reg_g = fac * (half * Rp + psi * R_loc)

        # P(loc) = N^{-loc} z^{loc + z_shift};  P'(loc) = P(loc)(log z - log N).
        P = _mpf_to_acb(mp.power(N, -loc)) * (
            _mpf_to_acb(loc + z_shift_val) * log_z
        ).exp()
        dlogP = _mpf_to_acb(-mp.log(N)) + log_z
        F_res = P * res_g
        F_reg = P * reg_g + (P * dlogP) * res_g
        rc = F_res.coeffs()
        gc = F_reg.coeffs()
        from flint import acb as _acb
        return (
            [rc[i] if i < len(rc) else _acb(0) for i in range(n_terms)],
            [gc[i] if i < len(gc) else _acb(0) for i in range(n_terms)],
        )

    def evaluate_kac_locations():
        """Genuine (non-removable) chiral h-pole locations ``-p/2`` of the global
        block, as a representative finite list -- non-``None`` to mark the mode
        as analytic-supported (the analytic factor helper uses it only as the
        support gate; the per-pole work is in :func:`evaluate_pole_data`)."""
        locs = []
        p_cap = 4 * n_terms + 50
        twice = [int(2 * re) if im == 0 and (2 * re).denominator == 1 else None for re, im in differences]
        for p in range(p_cap + 1):
            if not any(d is not None and (d - p) % 2 == 0 and -p <= d <= p for d in twice):
                locs.append(-mp.mpf(p) / 2)
        return locs

    def guarded(function, *, many=False):
        def call(*args, **kwargs):
            values = tuple(parse_number(h) for h in args[0]) if many else (parse_number(args[0]),)
            args = (values if many else values[0], *args[1:])
            old_cap = flint_ctx.cap
            token = _GLOBAL_HYP_ACCURACY_BITS.set(mp.mp.prec)
            try:
                return function(*args, **kwargs)
            finally:
                _GLOBAL_HYP_ACCURACY_BITS.reset(token)
                flint_ctx.cap = old_cap
        return call

    evaluate = guarded(evaluate)
    evaluate_many = guarded(evaluate_many, many=True)
    evaluate_pole_data = guarded(evaluate_pole_data)

    evaluate.pole_data = evaluate_pole_data
    evaluate.kac_locations = evaluate_kac_locations
    evaluate.many = evaluate_many
    return evaluate


def global_chiral_z_series_evaluator(
    h_externals, n_terms, *, b=None, c=None, q_order=None,
    chiral_norm=None, z_shift=None,
):
    """Batched flint-native global chiral evaluator in the z coordinate
    (``b`` / ``c`` / ``q_order`` accepted for API parity, ignored)."""
    return _global_chiral_series_evaluator(
        h_externals, n_terms, z_of_u=(lambda u: u), u_symmetric=z_symmetric(),
        chiral_norm=chiral_norm, z_shift=z_shift,
    )


def global_chiral_y_series_evaluator(
    h_externals, n_terms, *, b=None, c=None, q_order=None,
    chiral_norm=None, z_shift=None,
):
    """Batched flint-native global chiral evaluator in the y coordinate."""
    return _global_chiral_series_evaluator(
        h_externals, n_terms, z_of_u=z_of_y, u_symmetric=y_symmetric(),
        chiral_norm=chiral_norm, z_shift=z_shift,
    )


def global_chiral_q_series_evaluator(
    h_externals, n_terms, *, b=None, c=None, q_order=None,
    chiral_norm=None, z_shift=None,
):
    """Batched flint-native global chiral evaluator in the q (nome) coordinate."""
    return _global_chiral_series_evaluator(
        h_externals, n_terms, z_of_u=z_of_q, u_symmetric=q_symmetric(),
        chiral_norm=chiral_norm, z_shift=z_shift,
    )


def _global_chiral_taylor(
    h,
    h_externals: Sequence,
    n_terms: int,
    *,
    z_of_u: Callable,
    u_symmetric,
    chiral_norm=None,
    z_shift=None,
):
    n_terms = _integer_control('n_terms', n_terms, 1)
    _check_externals(h_externals)
    return _global_chiral_taylor_flint(
        h, h_externals, n_terms,
        z_of_u=z_of_u, u_symmetric=u_symmetric,
        chiral_norm=chiral_norm, z_shift=z_shift,
    )


def global_chiral_z_taylor(
    h,
    h_externals: Sequence,
    n_terms: int,
    *,
    b=None,
    c=None,
    q_order: int | None = None,
    chiral_norm=None,
    z_shift=None,
):
    """Taylor coefficients of the 2d global chiral block around z = 1/2.

    The block is independent of `b`, `c`, and `q_order`; those keyword
    arguments are accepted for API compatibility with the Virasoro builders.
    """
    return _global_chiral_taylor(
        h,
        h_externals,
        n_terms,
        z_of_u=lambda u: u,
        u_symmetric=z_symmetric(),
        chiral_norm=chiral_norm,
        z_shift=z_shift,
    )


def global_chiral_y_taylor(
    h,
    h_externals: Sequence,
    n_terms: int,
    *,
    b=None,
    c=None,
    q_order: int | None = None,
    chiral_norm=None,
    z_shift=None,
):
    """Taylor coefficients of the 2d global chiral block around y = y_*.

    The block is independent of `b`, `c`, and `q_order`; those keyword
    arguments are accepted for API compatibility with the Virasoro builders.
    """
    return _global_chiral_taylor(
        h,
        h_externals,
        n_terms,
        z_of_u=z_of_y,
        u_symmetric=y_symmetric(),
        chiral_norm=chiral_norm,
        z_shift=z_shift,
    )


def global_chiral_q_taylor(
    h,
    h_externals: Sequence,
    n_terms: int,
    *,
    b=None,
    c=None,
    q_order: int | None = None,
    chiral_norm=None,
    z_shift=None,
):
    """Taylor coefficients of the 2d global chiral block around q = q_*.

    The block is independent of `b`, `c`, and `q_order`; those keyword
    arguments are accepted for API compatibility with the Virasoro builders.
    """
    return _global_chiral_taylor(
        h,
        h_externals,
        n_terms,
        z_of_u=z_of_q,
        u_symmetric=q_symmetric(),
        chiral_norm=chiral_norm,
        z_shift=z_shift,
    )


def _build_B_table(
    chiral_func: Callable,
    h,
    hbar,
    h_externals: Sequence,
    hbar_externals: Sequence,
    m_max: int,
    n_max: int,
    *,
    b,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
    zbar_shift=None,
):
    c_h = chiral_func(
        h,
        h_externals,
        m_max + 1,
        **({"b": b} if c is None else {"c": c, "b": b}),
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
    )
    c_hbar = chiral_func(
        hbar,
        hbar_externals,
        n_max + 1,
        **({"b": b} if c is None else {"c": c, "b": b}),
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=zbar_shift,
    )
    return _assemble_block_grid(c_h, c_hbar, m_max, n_max, sigma=0)


def _build_G_table(
    chiral_func: Callable,
    Delta,
    ell: int,
    h_externals: Sequence,
    m_max: int,
    n_max: int,
    *,
    b,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
):
    Delta = _coerce_delta_scalar(Delta)
    # Keep the supplied dimension exact while splitting it into the two
    # chiral weights; each chiral evaluator chooses its numerical precision.
    h = _exchange_weight(Delta, ell)
    hbar = _exchange_weight(Delta, -ell)
    n_top = max(m_max, n_max) + 1
    c_h = chiral_func(
        h,
        h_externals,
        n_top,
        **({"b": b} if c is None else {"c": c, "b": b}),
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
    )
    if ell == 0:
        # Reversed product equals the forward one; pass c_h for both halves so
        # the assembler returns (1 + sigma) * c_h ⊗ c_h, which is B(h, h) with sigma = 0 (see _conventions).
        return _assemble_block_grid(c_h, c_h, m_max, n_max, sigma=mp.mpf(scalar_G_swap_coefficient(0)))
    c_hbar = chiral_func(
        hbar,
        h_externals,
        n_top,
        **({"b": b} if c is None else {"c": c, "b": b}),
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
    )
    sign = mp.mpf(scalar_G_swap_coefficient(ell))
    return _assemble_block_grid(c_h, c_hbar, m_max, n_max, sigma=sign)


def _coerce_delta_scalar(Delta):
    """Keep real exchange labels exact while deriving chiral weights."""
    if isinstance(Delta, ExactRational):
        return Delta
    if isinstance(Delta, complex) and Delta.imag == 0:
        return literal_number(mp.mpf(Delta.real))
    if isinstance(Delta, mp.mpc) and Delta.imag == 0:
        return literal_number(Delta.real)
    value = literal_number(Delta)
    return mp.convert(value) if isinstance(value, complex) else value


def _build_G_table_many_from_evaluator(
    evaluate: Callable,
    deltas: Sequence,
    ell: int,
    m_max: int,
    n_max: int,
):
    sign = mp.mpf(scalar_G_swap_coefficient(ell))
    tables = []
    for Delta in deltas:
        Delta = _coerce_delta_scalar(Delta)
        h = _exchange_weight(Delta, ell)
        c_h = evaluate(h)
        if ell == 0:
            tables.append(_assemble_block_grid(c_h, c_h, m_max, n_max, sigma=sign))
        else:
            c_hbar = evaluate(_exchange_weight(Delta, -ell))
            tables.append(
                _assemble_block_grid(c_h, c_hbar, m_max, n_max, sigma=sign)
            )
    return tables


def assemble_block_batch(c_h_all, c_hbar_all, m_max, n_max, *, sigma,
                         fold_parity=None, entries=None,
                         c_swap_m_all=None, c_swap_n_all=None):
    """Batched 2d block-grid assembly from per-sample chiral coefficient arrays.

    The single numeric core for every B / G / parity-projected block table and
    for the global, large-spin, and fixed-spin interpolation samplers.
    ``c_h_all`` /
    ``c_hbar_all`` are lists -- one entry per sampled value -- of coefficient
    arrays in one field: real flint ``arb``, complex flint ``acb``, or mpmath
    ``mpf`` / ``mpc``.

        sigma None / 0 (ordered B):   grid[m][n] = c_h[m] c_hbar[n]
        sigma != 0    (G / parity):   grid[m][n] = c_h[m] c_hbar[n]
                                                 + sigma c_hbar[m] c_h[n]

    A complex ``sigma`` on real ``arb`` data promotes the block to ``acb``.

    Optional ``c_swap_m_all`` / ``c_swap_n_all`` (both or neither) switch the
    swapped term to the generalized off-diagonal assembly

        grid[m][n] = c_h[m] c_hbar[n] + sigma c_swap_m[m] c_swap_n[n]

    instead of the legacy in-place swap ``sigma c_hbar[m] c_h[n]``. This keeps
    the diagonal path bit-identical while supporting independent m-/n-axis
    chiral centers.
    ``fold_parity`` ``None`` -> non-folded; ``0`` / ``1`` -> fold to (w, s) with
    that r = sqrt(s) smoothing parity (the source arrays must reach the folded
    source order, and the flint field is required). ``entries`` (a list of
    output ``(m, n)``) restricts the build.

    Returns ``(tables, is_arb)``: one ``[m][n]`` grid per sample, plus whether
    the result stayed in real flint ``arb`` (vs ``acb`` / mpmath -- the
    interpolation uses it to skip an arb<->mpf round trip)."""
    n_samples = len(c_h_all)
    if (c_swap_m_all is None) != (c_swap_n_all is None):
        raise ValueError(
            "c_swap_m_all and c_swap_n_all must be provided together"
        )
    if c_swap_m_all is not None and len(c_swap_m_all) != n_samples:
        raise ValueError("c_swap_m_all must match c_h_all length")
    if c_swap_n_all is not None and len(c_swap_n_all) != n_samples:
        raise ValueError("c_swap_n_all must match c_h_all length")

    requested = entries if entries is not None else [
        (m, n) for m in range(m_max + 1) for n in range(n_max + 1)
    ]
    first = c_h_all[0][0] if n_samples and len(c_h_all[0]) else None
    if first is None and c_swap_m_all is not None and n_samples and len(c_swap_m_all[0]):
        first = c_swap_m_all[0][0]

    # --- mpmath input: the single-grid flint-matmul / elementwise path -------
    # mpmath callers (the per-Delta tables and the global / large-spin samplers)
    # fold separately via `_fold_parity`, so the mpmath branch is non-folded.
    if isinstance(first, (mp.mpf, mp.mpc)) or first is None:
        if fold_parity is not None:
            raise ValueError(
                "mpmath block assembly is non-folded; fold the source grid via "
                "_fold_parity"
            )
        sig = mp.mpf(0) if sigma is None else sigma
        # A real-valued complex sigma keeps the block real; demote it so the
        # real (arb) matmul fast path applies.
        if isinstance(sig, mp.mpc) and sig.imag == 0:
            sig = mp.mpf(sig.real)
        tables = []
        for j, (c_h, c_hbar) in enumerate(zip(c_h_all, c_hbar_all)):
            c_swap_m = c_swap_m_all[j] if c_swap_m_all is not None else None
            c_swap_n = c_swap_n_all[j] if c_swap_n_all is not None else None
            grid = None
            if entries is None and c_swap_m is None:
                grid = _assemble_block_grid_flint(c_h, c_hbar, m_max, n_max, sig)
            if grid is None:
                grid = _assemble_block_grid_elementwise(
                    c_h, c_hbar, m_max, n_max, sig, entries,
                    c_swap_m=c_swap_m, c_swap_n=c_swap_n,
                )
            tables.append(grid)
        return tables, False

    # --- flint input (real arb / complex acb): batched native path -----------
    from flint import arb as _arb, ctx as _flint_ctx

    # Set the flint working precision explicitly so the arithmetic is correct
    # regardless of caller ordering (the products/fold carry the precision).
    _flint_ctx.prec = _flint_prec_bits()
    field_is_arb = isinstance(first, _arb)
    sigma_complex = isinstance(sigma, mp.mpc) and sigma.imag != 0
    out_acb = (not field_is_arb) or sigma_complex

    if out_acb:
        from flint import acb
        sigma_s = _mpf_to_acb(sigma) if sigma not in (None, 0) else None
        zero = acb(0)
        to_out = _acb_to_mpc
        if field_is_arb:
            # Promote the arb halves to acb once (a complex sigma makes the
            # block complex even on real chiral data).
            CH = [[acb(x) for x in c_h_all[j]] for j in range(n_samples)]
            CHB = [[acb(x) for x in c_hbar_all[j]] for j in range(n_samples)]
        else:
            CH, CHB = c_h_all, c_hbar_all
        if c_swap_m_all is None:
            CHS_M = CHS_N = None
        elif field_is_arb:
            CHS_M = [[acb(x) for x in c_swap_m_all[j]] for j in range(n_samples)]
            CHS_N = [[acb(x) for x in c_swap_n_all[j]] for j in range(n_samples)]
        else:
            CHS_M, CHS_N = c_swap_m_all, c_swap_n_all
    else:
        from flint import arb
        sigma_s = (
            _mpf_to_arb(mp.mpf(mp.re(sigma))) if sigma not in (None, 0) else None
        )
        zero = arb(0)
        to_out = lambda x: x
        CH, CHB = c_h_all, c_hbar_all
        CHS_M, CHS_N = c_swap_m_all, c_swap_n_all

    if sigma_s is None:
        def entry_batch(m, n, _CH=CH, _CHB=CHB):
            return [_CH[j][m] * _CHB[j][n] for j in range(n_samples)]
    elif CHS_M is None:
        def entry_batch(m, n, _CH=CH, _CHB=CHB, _s=sigma_s):
            return [
                _CH[j][m] * _CHB[j][n] + _s * (_CHB[j][m] * _CH[j][n])
                for j in range(n_samples)
            ]
    else:
        def entry_batch(m, n, _CH=CH, _CHB=CHB, _CHSM=CHS_M, _CHSN=CHS_N, _s=sigma_s):
            return [
                _CH[j][m] * _CHB[j][n] + _s * (_CHSM[j][m] * _CHSN[j][n])
                for j in range(n_samples)
            ]

    if fold_parity is None:
        tables = [
            [[None] * (n_max + 1) for _ in range(m_max + 1)]
            for _ in range(n_samples)
        ]
        for (m, n) in requested:
            row = entry_batch(m, n)
            for j in range(n_samples):
                tables[j][m][n] = to_out(row[j])
        return tables, (not out_acb)

    odd = fold_parity % 2
    # The source degree only needs to reach max(a + 2b) over the requested
    # output entries (= Lambda for the m + 2n <= Lambda triangle), so the fold
    # plan shrinks with the truncation.
    M_total = max((m + 2 * n for (m, n) in requested), default=0) + odd
    plan, _ = _fold_plan(bool(odd), M_total + 1, M_total + 1, m_max, n_max)
    plan_key = (odd, M_total + 1, M_total + 1, m_max, n_max)
    tables = _fold_batch_to_grids(
        plan, plan_key, n_samples, m_max, n_max,
        entry_batch=entry_batch, requested=list(requested),
        complex_field=out_acb, to_out=to_out, zero=zero,
    )
    return tables, (not out_acb)


def _assemble_block_grid(
    c_h, c_hbar, m_max, n_max, *, sigma, entries=None, c_swap_m=None, c_swap_n=None
):
    """Single-sample, non-folded block grid -- a thin wrapper over
    :func:`assemble_block_batch` (batch of one). Used by the per-Delta tables
    and the global / large-spin samplers, which fold separately via
    :func:`_fold_parity`."""
    tables, _ = assemble_block_batch(
        [c_h], [c_hbar], m_max, n_max, sigma=sigma, entries=entries,
        c_swap_m_all=([c_swap_m] if c_swap_m is not None else None),
        c_swap_n_all=([c_swap_n] if c_swap_n is not None else None),
    )
    return tables[0]


def _assemble_block_grid_elementwise(
    c_h, c_hbar, m_max, n_max, sigma, entries, *, c_swap_m=None, c_swap_n=None
):
    if entries is None:
        entries = [(m, n) for m in range(m_max + 1) for n in range(n_max + 1)]
    grid = [[None] * (n_max + 1) for _ in range(m_max + 1)]
    if sigma == 0:
        for (m, n) in entries:
            grid[m][n] = c_h[m] * c_hbar[n]
    elif c_swap_m is None:
        for (m, n) in entries:
            grid[m][n] = c_h[m] * c_hbar[n] + sigma * (c_hbar[m] * c_h[n])
    else:
        for (m, n) in entries:
            grid[m][n] = c_h[m] * c_hbar[n] + sigma * (c_swap_m[m] * c_swap_n[n])
    return grid


def _assemble_block_grid_flint(c_h, c_hbar, m_max, n_max, sigma):
    """Full-grid flint matmul path for :func:`_assemble_block_grid`; returns
    None (deferring to the elementwise path) for entries that are not mpmath
    scalars."""
    def _real_tol(x) -> mp.mpf:
        # relative: a small coefficient (a large-weight block) keeps an imaginary part that is not negligible
        # against the coefficient itself
        return _validation_tolerance(guard=5) * abs(mp.mpc(x))

    def _canonicalize(value):
        # The common real-c / real-h path already produces mp.mpf coefficients
        # (the arb-native chiral evaluator), so skip the mpc wrap + tolerance
        # power that only matter for genuinely complex inputs.
        if isinstance(value, mp.mpf):
            return value
        z = mp.mpc(value)
        if abs(mp.im(z)) <= _real_tol(z):
            return mp.mpf(mp.re(z))
        return z

    same = c_h is c_hbar
    # Ordered B (sigma == 0) needs only c_h[0..m_max] and c_hbar[0..n_max] (its
    # halves may have those exact, unequal lengths). The reversed product
    # (sigma != 0) also reads c_hbar[0..m_max] and c_h[0..n_max], so both halves
    # must reach max(m_max, n_max); those callers build to that length.
    if sigma == 0:
        n_h, n_hbar = m_max + 1, n_max + 1
    else:
        n_h = n_hbar = max(m_max, n_max) + 1
    # one shared half serves both factors: canonicalize it to the longer of the two lengths
    c_h_used = [_canonicalize(c_h[index]) for index in range(max(n_h, n_hbar) if same else n_h)]
    c_hbar_used = (
        c_h_used if same
        else [_canonicalize(c_hbar[index]) for index in range(n_hbar)]
    )
    sample = c_h_used[0] if c_h_used else None
    # The flint field must cover every entry: real (arb) only if the
    # coefficients AND sigma are all real; a complex coefficient or a complex
    # sigma (parity-projected with a complex phase) forces acb.
    sigma_complex = isinstance(sigma, mp.mpc) and sigma.imag != 0
    if isinstance(sample, mp.mpf):
        if sigma_complex or any(isinstance(v, mp.mpc) for v in c_h_used) or any(
            isinstance(v, mp.mpc) for v in c_hbar_used
        ):
            sample = mp.mpc(0)

    if isinstance(sample, mp.mpf):
        from flint import arb, arb_mat, ctx as flint_ctx
        to_field, mat_cls, from_field = _mpf_to_arb, arb_mat, _arb_to_mpf
        scalar = arb
    elif isinstance(sample, mp.mpc):
        from flint import acb, acb_mat, ctx as flint_ctx
        to_field, mat_cls, from_field = _mpf_to_acb, acb_mat, _acb_to_mpc
        scalar = acb
    else:
        return None
    # Set the flint working precision explicitly so the matmul is correct
    # regardless of caller ordering (the matmul carries the precision, not the
    # mpf<->arb conversions).
    flint_ctx.prec = _flint_prec_bits()

    col_h = mat_cls([[to_field(c_h_used[m])] for m in range(m_max + 1)])
    row_hbar = mat_cls([[to_field(c_hbar_used[n]) for n in range(n_max + 1)]])
    mat = col_h * row_hbar
    if sigma == 0:
        return [[from_field(mat[m, n]) for n in range(n_max + 1)]
                for m in range(m_max + 1)]
    if same:
        # Reversed product equals the forward one: grid = (1 + sigma) * mat.
        scale = scalar(1) + scalar(sigma)
        return [[from_field(mat[m, n] * scale) for n in range(n_max + 1)]
                for m in range(m_max + 1)]
    col_hbar = mat_cls([[to_field(c_hbar_used[m])] for m in range(m_max + 1)])
    row_h = mat_cls([[to_field(c_h_used[n]) for n in range(n_max + 1)]])
    mat2 = col_hbar * row_h
    sigma_scale = scalar(sigma)
    return [
        [from_field(mat[m, n] + sigma_scale * mat2[m, n]) for n in range(n_max + 1)]
        for m in range(m_max + 1)
    ]


# zzb (non-folded, (z, zbar))


def B_zzb_table(
    h,
    hbar,
    h_externals: Sequence,
    hbar_externals: Sequence,
    m_max: int,
    n_max: int | None = None,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
    zbar_shift=None,
):
    """2D Taylor coefficients of B(h, hbar; z, zbar) at z = zbar = 1/2."""
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    return _build_B_table(
        chiral_z_taylor,
        h,
        hbar,
        h_externals,
        hbar_externals,
        m_max,
        n_max,
        b=b, c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
        zbar_shift=zbar_shift,
    )


def G_zzb_table(
    Delta,
    ell: int,
    h_externals: Sequence,
    m_max: int,
    n_max: int | None = None,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
):
    """2D Taylor coefficients of G_{Delta, ell}(z, zbar) at z = zbar = 1/2.

    Uses the same normalization convention as `F_block`: default `z_shift=0`
    gives small-z behavior `z^h / N^h`, while physical crossing conventions
    can be reached by passing `z_shift = -h1 - h2`.
    """
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    _check_externals(h_externals)
    ell = _check_ell(ell)
    return _build_G_table(
        chiral_z_taylor,
        Delta,
        ell,
        h_externals,
        m_max,
        n_max,
        b=b, c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
    )


def G_zzb_table_many(
    deltas: Sequence,
    ell: int,
    h_externals: Sequence,
    m_max: int,
    n_max: int | None = None,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
):
    """Batched :func:`G_zzb_table` sharing one chiral series evaluator.

    The Delta-independent geometry (nome series, prefactor base series) is
    computed once and reused across every Delta, so the per-Delta cost is the
    ``H``-recursion plus a few truncated power-series products -- no repeated
    elliptic-nome / theta evaluations and no ``taylor_coefficients`` numerical
    differentiation.  Returns one ``[m][n]`` table per Delta, matching
    ``[G_zzb_table(d, ...) for d in deltas]`` to working precision.
    """
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    _check_externals(h_externals)
    ell = _check_ell(ell)
    n_top = max(m_max, n_max) + 1
    evaluate = chiral_z_series_evaluator(
        h_externals, n_top, b=b, c=c, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift,
    )
    return _build_G_table_many_from_evaluator(evaluate, deltas, ell, m_max, n_max)


def global_G_zzb_table(
    Delta,
    ell: int,
    h_externals: Sequence,
    m_max: int,
    n_max: int | None = None,
    *,
    chiral_norm=None,
    z_shift=None,
):
    """2D Taylor coefficients of the global 2d scalar block at z = zbar = 1/2.

    Uses the same normalization convention as `F_block` / `G_zzb_table`:
    default `z_shift=0` gives small-z behavior `z^h / N^h`, while physical
    crossing conventions can be reached by passing `z_shift = -h1 - h2`.
    """
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    _check_externals(h_externals)
    ell = _check_ell(ell)

    def _global_chiral(h, ext, n_terms, *, b, q_order, chiral_norm=None, z_shift=None):
        return global_chiral_z_taylor(
            h,
            ext,
            n_terms,
            chiral_norm=chiral_norm,
            z_shift=z_shift,
        )

    return _build_G_table(
        _global_chiral,
        Delta,
        ell,
        h_externals,
        m_max,
        n_max,
        b=None,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
        q_order=0,
    )


def global_G_xt_table(
    Delta,
    ell: int,
    h_externals: Sequence,
    m_max: int,
    n_max: int | None = None,
    *,
    chiral_norm=None,
    z_shift=None,
):
    """Parity-folded (x, t) Taylor coefficients of the global 2d scalar block.

    The global analogue of :func:`G_xt_table`: same fold of the z-coordinate
    block (``x = (z + zbar)/2 - 1/2``, ``t = ((z - zbar)/2)^2`` around
    ``z = zbar = 1/2``), but built from the global SL(2) chiral half rather than
    the Virasoro recursion. Scalar externals only; the block is z <-> zbar even
    at every ``ell``, exactly as in :func:`G_xt_table`.
    """
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    _check_externals(h_externals)
    ell = _check_ell(ell)

    def _global_chiral(h, ext, n_terms, *, b, q_order, chiral_norm=None, z_shift=None):
        return global_chiral_z_taylor(
            h, ext, n_terms, chiral_norm=chiral_norm, z_shift=z_shift,
        )

    return _G_folded_table(
        _global_chiral,
        Delta,
        ell,
        h_externals,
        m_max,
        n_max,
        b=None,
        q_order=0,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
    )


def global_G_ws_table(
    Delta,
    ell: int,
    h_externals: Sequence,
    m_max: int,
    n_max: int | None = None,
    *,
    chiral_norm=None,
    z_shift=None,
):
    """Parity-folded (w, s) Taylor coefficients of the global 2d scalar block.

    The global analogue of :func:`G_ws_table`: same fold of the y-coordinate
    block (``w = (y + ybar)/2``, ``s = ((y - ybar)/2)^2`` around ``y = ybar = 0``),
    built from the global SL(2) chiral half. Scalar externals only.
    """
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    _check_externals(h_externals)
    ell = _check_ell(ell)

    def _global_chiral(h, ext, n_terms, *, b, q_order, chiral_norm=None, z_shift=None):
        return global_chiral_y_taylor(
            h, ext, n_terms, chiral_norm=chiral_norm, z_shift=z_shift,
        )

    return _G_folded_table(
        _global_chiral,
        Delta,
        ell,
        h_externals,
        m_max,
        n_max,
        b=None,
        q_order=0,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
    )


# yyb (non-folded, y = (2z - 1)/(1 + 2 sqrt(z(1-z))))


def B_yyb_table(
    h,
    hbar,
    h_externals: Sequence,
    hbar_externals: Sequence,
    m_max: int,
    n_max: int | None = None,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
    zbar_shift=None,
):
    """2D Taylor coefficients of B(h, hbar; y, ybar) at y = ybar = 0."""
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    return _build_B_table(
        chiral_y_taylor,
        h,
        hbar,
        h_externals,
        hbar_externals,
        m_max,
        n_max,
        b=b, c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
        zbar_shift=zbar_shift,
    )


def G_yyb_table(
    Delta,
    ell: int,
    h_externals: Sequence,
    m_max: int,
    n_max: int | None = None,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
):
    """2D Taylor coefficients of G_{Delta, ell}(y, ybar) at y = ybar = 0."""
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    _check_externals(h_externals)
    ell = _check_ell(ell)
    return _build_G_table(
        chiral_y_taylor,
        Delta,
        ell,
        h_externals,
        m_max,
        n_max,
        b=b, c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
    )


def G_yyb_table_many(
    deltas: Sequence,
    ell: int,
    h_externals: Sequence,
    m_max: int,
    n_max: int | None = None,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
):
    """Batched :func:`G_yyb_table` sharing one chiral series evaluator."""
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    _check_externals(h_externals)
    ell = _check_ell(ell)
    n_top = max(m_max, n_max) + 1
    evaluate = chiral_y_series_evaluator(
        h_externals, n_top, b=b, c=c, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift,
    )
    return _build_G_table_many_from_evaluator(evaluate, deltas, ell, m_max, n_max)


def global_G_yyb_table(
    Delta,
    ell: int,
    h_externals: Sequence,
    m_max: int,
    n_max: int | None = None,
    *,
    chiral_norm=None,
    z_shift=None,
):
    """2D Taylor coefficients of the global 2d scalar block at y = ybar = y_*."""
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    _check_externals(h_externals)
    ell = _check_ell(ell)

    def _global_chiral(h, ext, n_terms, *, b, q_order, chiral_norm=None, z_shift=None):
        return global_chiral_y_taylor(
            h,
            ext,
            n_terms,
            chiral_norm=chiral_norm,
            z_shift=z_shift,
        )

    return _build_G_table(
        _global_chiral,
        Delta,
        ell,
        h_externals,
        m_max,
        n_max,
        b=None,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
        q_order=0,
    )


# qqb (non-folded, (q, qbar) with q = exp(-pi K(1-z)/K(z)))


def B_qqb_table(
    h,
    hbar,
    h_externals: Sequence,
    hbar_externals: Sequence,
    m_max: int,
    n_max: int | None = None,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
    zbar_shift=None,
):
    """2D Taylor coefficients of B(h, hbar; q, qbar) at q = qbar = exp(-pi)."""
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    return _build_B_table(
        chiral_q_taylor,
        h,
        hbar,
        h_externals,
        hbar_externals,
        m_max,
        n_max,
        b=b, c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
        zbar_shift=zbar_shift,
    )


def G_qqb_table(
    Delta,
    ell: int,
    h_externals: Sequence,
    m_max: int,
    n_max: int | None = None,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
):
    """2D Taylor coefficients of G_{Delta, ell}(q, qbar) at q = qbar = exp(-pi)."""
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    _check_externals(h_externals)
    ell = _check_ell(ell)
    return _build_G_table(
        chiral_q_taylor,
        Delta,
        ell,
        h_externals,
        m_max,
        n_max,
        b=b, c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
    )


def G_qqb_table_many(
    deltas: Sequence,
    ell: int,
    h_externals: Sequence,
    m_max: int,
    n_max: int | None = None,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
):
    """Batched :func:`G_qqb_table` sharing one chiral series evaluator."""
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    _check_externals(h_externals)
    ell = _check_ell(ell)
    n_top = max(m_max, n_max) + 1
    evaluate = _chiral_series_evaluator(
        h_externals, n_top, b=b, c=c, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift,
        z_of_u=z_of_q, u_symmetric=q_symmetric(), coord_key="q",
    )
    return _build_G_table_many_from_evaluator(evaluate, deltas, ell, m_max, n_max)


def global_G_qqb_table(
    Delta,
    ell: int,
    h_externals: Sequence,
    m_max: int,
    n_max: int | None = None,
    *,
    chiral_norm=None,
    z_shift=None,
):
    """2D Taylor coefficients of the global 2d scalar block at q = qbar = q_*."""
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    _check_externals(h_externals)
    ell = _check_ell(ell)

    def _global_chiral(h, ext, n_terms, *, b, q_order, chiral_norm=None, z_shift=None):
        return global_chiral_q_taylor(
            h,
            ext,
            n_terms,
            chiral_norm=chiral_norm,
            z_shift=z_shift,
        )

    return _build_G_table(
        _global_chiral,
        Delta,
        ell,
        h_externals,
        m_max,
        n_max,
        b=None,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
        q_order=0,
    )


# parity-folded coordinates: substitute u - u_* = w + r, ubar - u_* = w - r,
# with s = r^2. A z <-> zbar-even block (the scalar G at every spin, a '+'
# parity projection with phase +1) is even in r and folds to a (w, s) series.
# An odd block is odd in r and factors as r * Gtilde(w, s); we return the
# Gtilde coefficients (r-stripped).


# --- shared fold geometry (single source for the (w, s) fold conventions) ---


def fold_parity_marker(*, ell=None, parity=None, phase=1):
    """The (w, s) fold's r = sqrt(s) smoothing parity (0 even, 1 odd).

    A scalar ``G`` is z <-> zbar even at every spin ``ell``, so its marker is 0
    (:func:`~._conventions.scalar_G_fold_parity`).  A parity-projected block
    ``B_+- = B +- phase P[B]`` has z <-> zbar parity ``sigma = (+-1) * phase``
    (P[B] is B with z <-> zbar), so it is odd when sigma is -1: the projection
    sign alone decides only for a +1 total phase (a '+' projection with phase -1
    is odd -- folding it as even wrote an all-zero table).  ``phase`` must be
    real (+-1) for folded coordinates.  Exactly one of ``ell`` / ``parity`` is
    given.
    """
    if parity is not None:
        if parity not in ("+", "-"):
            raise ValueError("parity must be '+' or '-'")
        phase = mp.mpc(phase)
        tol = _validation_tolerance(guard=5)
        if (not mp.isfinite(phase) or abs(phase.imag) > tol
                or min(abs(phase.real - 1), abs(phase.real + 1)) > tol):
            raise ValueError("parity-folded coordinates require a real (+/-1) total parity phase")
        sigma = (1 if parity == "+" else -1) * phase.real
        return 1 if sigma < 0 else 0
    return scalar_G_fold_parity(ell)


def folded_source_order(m_max, n_max, odd_factor):
    """Non-folded Taylor order needed in *each* index to build the folded
    ``(m_max, n_max)`` (w, s) table: the fold reads ``(u - u_*)^a (ubar - u_*)^b``
    up to ``a + 2b <= m_max + 2 n_max`` (one more for the odd r-stripped case)."""
    return m_max + 2 * n_max + (odd_factor % 2)


def folded_entry_power(alpha, beta, odd_factor):
    """Leading interpolation-variable power of a folded ``(alpha, beta)`` entry:
    it draws from non-folded ``(m, n)`` with ``m + n = alpha + 2 beta`` (+1 if
    odd), so the leading power is ``alpha + 2 beta`` (+1 if odd)."""
    return alpha + 2 * beta + (odd_factor % 2)


def _fold_parity(uubar_table, parity_marker: int, m_max: int, n_max: int, *, zero=None):
    """Convert a (u-u_*, ubar-u_*) Taylor table into a (w, s) Taylor table.

    For odd z <-> zbar parity (a parity-projected B_+/- with total z <-> zbar
    parity -1; never a scalar G, which is even at every spin) the block G
    vanishes at s = 0 with leading factor r = sqrt(s), so a naive Taylor
    expansion in (w, s) does not exist -- d_s G at s = 0 is divergent for any
    positive derivative order. We strip one factor of r:
    return the (w, s) coefficients of `Gtilde = G / r`, which IS smooth at
    s = 0. This is the same convention as blocks_3d's `p_-` smoothing factor
    (`doc/conventions.tex`): blocks_3d writes the odd-parity folded output as
    `p_- * G` with `p_- = 2/(z - zbar) = 1/r`, so blocks_3d's `p_- * G`
    equals our `Gtilde`. Even parity needs no smoothing; we return G directly,
    matching blocks_3d's `p_+ = 1`.

    Inputs:
        uubar_table: list-of-lists indexed [m][n], the (u-u_*)^m (ubar-u_*)^n
            Taylor coefficients of G at u = ubar = u_*. Must extend to at
            least order M_total = m_max + 2 n_max + parity_marker in each index.
        parity_marker: the z <-> zbar parity of the block (:func:`fold_parity_marker`);
            0 -> r-even (no smoothing), 1 -> r-odd (strip one r factor,
            applying the p_- smoothing).
        m_max, n_max: desired output dimensions; the returned table has shape
            (m_max + 1) x (n_max + 1).
        zero: additive identity for the table entries (default mpmath 0). Pass
            `Polynomial.zero()` to fold a table of symbolic (polynomial-in-Delta)
            entries -- the fold is a Delta-independent linear recombination, so
            it acts on numeric and symbolic entries identically.

    Returns table[alpha][beta] of size (m_max + 1) x (n_max + 1), where
        for even parity: G(u, ubar) = sum_{a,b} table[a][b] w^a s^b
        for odd parity:  G(u, ubar) = r * sum_{a,b} table[a][b] w^a s^b
                         (equivalently: table = (w, s) Taylor of `Gtilde = G/r`).
    """
    sample = uubar_table[0][0] if uubar_table and uubar_table[0] else None
    if parity_marker % 2 == 0:
        odd_factor = False
    else:
        odd_factor = True

    # max (m + n) we ever touch in uubar_table
    max_mn = m_max + 2 * n_max + (1 if odd_factor else 0)

    rows = len(uubar_table)
    cols = len(uubar_table[0]) if rows else 0

    row_limit = min(rows, max_mn + 1)
    col_limit = min(cols, max_mn + 1)

    plan, target_size = _fold_plan(odd_factor, row_limit, col_limit, m_max, n_max)
    width = n_max + 1

    # The real (arb) branches need EVERY folded entry real, not just the first:
    # a grid can mix mpf and mpc entries (e.g. NS / Ramond scalar_G, where some
    # entries carry a complex phase), and an mpc fed to the arb path fails.
    real_entries = isinstance(sample, mp.mpf) or (
        isinstance(sample, Polynomial) and sample.field is mp.mpf
    )
    if real_entries:
        for m, n, _contribs in plan:
            coef = uubar_table[m][n]
            if isinstance(coef, (mp.mpc, complex)) or (
                isinstance(coef, Polynomial) and coef.field is not mp.mpf
            ):
                real_entries = False
                break
    if zero is None:
        # Match the field of the actual table entries so the fold stays in
        # mp.mpf when uubar_table is mpf (the typical real-c, real-h case).
        zero = mp.mpf(0) if real_entries and isinstance(sample, mp.mpf) else mp.mpc(0)

    if isinstance(sample, Polynomial):
        if real_entries:
            from flint import arb, arb_poly, ctx as flint_ctx
            prec_bits = _flint_prec_bits()
            flint_ctx.prec = prec_bits
            accum = [arb_poly([arb(0)]) for _ in range(target_size)]
            for m, n, contribs in plan:
                coef = uubar_table[m][n]
                if coef == 0:
                    continue
                coef_f = coef._get_arb()
                for target_idx, weight in contribs:
                    accum[target_idx] += coef_f * arb(weight)
            # Lazy mpf: keep the arb_poly as the cache so a downstream
            # _get_arb() (the Vandermonde sampler) reuses it instead of
            # rebuilding mpf->arb; defer the arb->mpf materialization.
            return [
                [Polynomial._from_arb_trusted(accum[alpha * width + beta], prec_bits)
                 for beta in range(width)]
                for alpha in range(m_max + 1)
            ]

        if not real_entries and all(
            isinstance(uubar_table[m][n], Polynomial) or uubar_table[m][n] == 0
            for m, n, _contribs in plan
        ):
            from flint import acb, acb_poly, arb, ctx as flint_ctx
            prec_bits = _flint_prec_bits()
            flint_ctx.prec = prec_bits
            accum = [acb_poly([acb(arb(0))]) for _ in range(target_size)]
            for m, n, contribs in plan:
                coef = uubar_table[m][n]
                if coef == 0:
                    continue
                coef_f = coef._get_acb()
                for target_idx, weight in contribs:
                    accum[target_idx] += coef_f * acb(weight)
            # Lazy mpc: keep the acb_poly cache for a downstream _get_acb()
            # (see the real branch above).
            return [
                [Polynomial._from_acb_trusted(accum[alpha * width + beta], prec_bits)
                 for beta in range(width)]
                for alpha in range(m_max + 1)
            ]

    if real_entries and isinstance(sample, mp.mpf):
        from flint import arb
        accum = [arb(0) for _ in range(target_size)]
        for m, n, contribs in plan:
            coef = uubar_table[m][n]
            if coef == 0:
                continue
            coef_f = _mpf_to_arb(coef)
            for target_idx, weight in contribs:
                accum[target_idx] += coef_f * weight
        return [
            [_arb_to_mpf(accum[alpha * width + beta]) for beta in range(width)]
            for alpha in range(m_max + 1)
        ]

    if isinstance(sample, (mp.mpf, mp.mpc)) and not real_entries:
        from flint import acb
        accum = [acb(0) for _ in range(target_size)]
        for m, n, contribs in plan:
            coef = uubar_table[m][n]
            if coef == 0:
                continue
            coef_f = _mpf_to_acb(coef)
            for target_idx, weight in contribs:
                accum[target_idx] += coef_f * weight
        return [
            [_acb_to_mpc(accum[alpha * width + beta]) for beta in range(width)]
            for alpha in range(m_max + 1)
        ]

    flat = [zero for _ in range(target_size)]
    for m, n, contribs in plan:
        coef = uubar_table[m][n]
        if coef == 0:
            continue
        for target_idx, weight in contribs:
            flat[target_idx] += coef * weight
    return [
        [flat[alpha * width + beta] for beta in range(width)]
        for alpha in range(m_max + 1)
    ]


def _fold_plan(odd_factor: bool, row_limit: int, col_limit: int, m_max: int, n_max: int):
    key = (1 if odd_factor else 0, row_limit, col_limit, m_max, n_max)
    cached = _FOLD_PLAN_CACHE.get(key)
    if cached is not None:
        return cached

    max_mn = m_max + 2 * n_max + (1 if odd_factor else 0)
    binom_limit = max(row_limit, col_limit)
    binom_rows = []
    if binom_limit > 0:
        binom_rows.append([1])
        for degree in range(1, binom_limit):
            prev = binom_rows[-1]
            row = [1] * (degree + 1)
            for idx in range(1, degree):
                row[idx] = prev[idx - 1] + prev[idx]
            binom_rows.append(row)
    signed_binom_rows = [
        [(-coeff if (idx % 2) else coeff) for idx, coeff in enumerate(row)]
        for row in binom_rows[:col_limit]
    ]

    width = n_max + 1
    plan = []
    for m in range(row_limit):
        cm_row = binom_rows[m]
        for n in range(min(col_limit, max_mn + 1 - m)):
            cn_row = signed_binom_rows[n]
            contribs: dict[int, int] = {}
            # Expand (w + r)^m (w - r)^n = sum_{i, j}
            #   (-1)^j C(m, i) C(n, j) w^{m+n-i-j} r^{i+j}
            for i in range(m + 1):
                cm_i = cm_row[i]
                for j in range(n + 1):
                    alpha = m + n - i - j
                    if alpha > m_max:
                        continue
                    k = i + j
                    if odd_factor:
                        if k % 2 == 0:
                            continue
                        beta = (k - 1) // 2
                    else:
                        if k % 2 != 0:
                            continue
                        beta = k // 2
                    if beta > n_max:
                        continue
                    target_idx = alpha * width + beta
                    contribs[target_idx] = contribs.get(target_idx, 0) + cm_i * cn_row[j]
            if contribs:
                plan.append((m, n, list(contribs.items())))

    _FOLD_PLAN_CACHE[key] = (plan, (m_max + 1) * width)
    return _FOLD_PLAN_CACHE[key]


# The parity fold is a fixed (Delta-independent) sparse linear map from the
# source outer-product entries to the folded (w, s) grid. Two ways to apply it:
#   * sparse loop  -- one flint scalar mul-add per (source, target, weight) plan
#     contribution, Python-overhead bound at small/medium Lambda;
#   * dense matmul -- folded = F @ ENTRY with F the cached weight matrix, ~82x
#     more flops at Lambda 27 but at flint's compiled matmul speed (no per-op
#     Python tax).
#
# The triangular truncation and per-entry restrictions make the fold sparse.
# Constructing the dense weight matrix costs more than a single sparse fold;
# it pays off only when many calls share the same cache key. Keep the sparse
# path as the default (threshold 0).
_FOLD_MATMUL_MAX_CONTRIBS = 0

_FOLD_MATRIX_CACHE: dict = {}
_FOLD_MATRIX_CACHE_MAX = 32


def _fold_use_matmul(contribs: int) -> bool:
    return 0 < contribs <= _FOLD_MATMUL_MAX_CONTRIBS


def _fold_weight_matrix(cache_key, rplan, target_order, complex_field):
    """Cached fold-weight matrix ``F`` (len(target_order) x len(rplan)) so that
    ``folded = F @ ENTRY`` with ``ENTRY[src][delta]`` the source entry value.

    ``rplan`` is the (possibly pruned) plan; column ``i`` is ``rplan[i]``'s
    source. ``target_order`` is the list of target indices forming F's rows.
    Delta-independent, so it is cached on ``cache_key``.
    """
    full_key = (cache_key, complex_field)
    cached = _FOLD_MATRIX_CACHE.get(full_key)
    if cached is not None:
        return cached
    from flint import arb, acb, arb_mat, acb_mat

    tgt_row = {ti: i for i, ti in enumerate(target_order)}
    n_src = len(rplan)
    rows = [[0] * n_src for _ in range(len(target_order))]
    for si, (_m, _n, contribs) in enumerate(rplan):
        for ti, w in contribs:
            rows[tgt_row[ti]][si] += w
    scalar = acb if complex_field else arb
    mat_cls = acb_mat if complex_field else arb_mat
    F = mat_cls([[scalar(x) for x in row] for row in rows]) if n_src else mat_cls(
        len(target_order), 0
    )
    _FOLD_MATRIX_CACHE[full_key] = F
    if len(_FOLD_MATRIX_CACHE) > _FOLD_MATRIX_CACHE_MAX:
        _FOLD_MATRIX_CACHE.pop(next(iter(_FOLD_MATRIX_CACHE)))
    return F


def _fold_batch_to_grids(
    plan, plan_key, n_deltas, m_max, n_max, *,
    entry_batch, requested=None, complex_field, to_out, zero,
):
    """Fold a batch of ``n_deltas`` Deltas to sparse ``[m][n]`` grids.

    ``entry_batch(m, n)`` returns the length-``n_deltas`` sequence of source
    entry values ``e(m, n)`` (the per-Delta outer-product / projected term).
    ``requested`` is the list of output ``(alpha, beta)`` to fill (``None`` =
    full grid); unrequested entries stay ``None``. ``to_out`` converts a folded
    flint scalar to the output field; ``zero`` is the flint additive identity.

    Picks the dense flint matmul (``folded = F @ ENTRY``) when the pruned plan's
    contribution count is below the crossover, else the per-Delta sparse loop.
    Both produce identical values; only the contraction strategy differs.
    """
    width = n_max + 1
    if requested is None:
        requested = [
            (a, b) for a in range(m_max + 1) for b in range(n_max + 1)
        ]
    req_idx = {a * width + b for (a, b) in requested}

    # Prune the plan to contributions landing on a requested target.
    if len(req_idx) == (m_max + 1) * width:
        rplan = plan
        pruned_key = (plan_key, "full")
    else:
        rplan = []
        for (m, n, contribs) in plan:
            kept = [(ti, w) for (ti, w) in contribs if ti in req_idx]
            if kept:
                rplan.append((m, n, kept))
        pruned_key = (plan_key, tuple(sorted(req_idx)))

    contribs_total = sum(len(c) for _, _, c in rplan)

    if _fold_use_matmul(contribs_total):
        from flint import arb_mat, acb_mat

        mat_cls = acb_mat if complex_field else arb_mat
        target_order = sorted(req_idx)
        tgt_row = {ti: i for i, ti in enumerate(target_order)}
        F = _fold_weight_matrix(pruned_key, rplan, target_order, complex_field)
        entry_rows = [list(entry_batch(m, n)) for (m, n, _c) in rplan]
        if entry_rows:
            ENTRY = mat_cls(entry_rows)
            folded = F * ENTRY  # (n_targets x n_deltas)
        else:
            folded = None
        grids = []
        for j in range(n_deltas):
            grid = [[None] * (n_max + 1) for _ in range(m_max + 1)]
            for (a, b) in requested:
                ti = a * width + b
                if folded is not None and ti in tgt_row:
                    grid[a][b] = to_out(folded[tgt_row[ti], j])
                else:
                    grid[a][b] = to_out(zero)
            grids.append(grid)
        return grids

    # Sparse per-Delta contraction.
    entry_rows = {(m, n): list(entry_batch(m, n)) for (m, n, _c) in rplan}
    grids = []
    for j in range(n_deltas):
        accum: dict = {}
        for (m, n, contribs) in rplan:
            e = entry_rows[(m, n)][j]
            for ti, w in contribs:
                accum[ti] = (accum[ti] + e * w) if ti in accum else e * w
        grid = [[None] * (n_max + 1) for _ in range(m_max + 1)]
        for (a, b) in requested:
            ti = a * width + b
            grid[a][b] = to_out(accum[ti]) if ti in accum else to_out(zero)
        grids.append(grid)
    return grids


def _G_folded_table(
    chiral_func: Callable,
    Delta,
    ell: int,
    h_externals: Sequence,
    m_max: int,
    n_max: int,
    *,
    b,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
):
    """Common builder for ws / xt / qqb_ws tables."""
    odd_factor = fold_parity_marker(ell=ell)
    M_total = folded_source_order(m_max, n_max, odd_factor)
    uubar = _build_G_table(
        chiral_func, Delta, ell, h_externals, M_total, M_total, b=b, c=c,
        q_order=q_order, chiral_norm=chiral_norm, z_shift=z_shift,
    )
    return _fold_parity(uubar, odd_factor, m_max, n_max)


def _with_rounding_radius(values, work: int):
    """Flint balls widened to at least ``|value| 2^-work``: an evaluator's coefficients carry the rounding of its
    working-precision inputs (the geometry series, b), which a canceling linear map downstream amplifies."""
    from flint import arb, acb

    eps = arb(2) ** (-work)
    out = []
    for value in values:
        if isinstance(value, acb):
            r = arb(0, value.abs_upper() * eps)
            out.append(acb(value.real + r, value.imag + r))
        else:
            out.append(value + arb(0, value.abs_upper() * eps))
    return out


def _grid_deficit(grid, work: int) -> int:
    """Bits by which the worst radius of a grid of flint balls exceeds ``2^-work`` times the grid's largest
    magnitude (the scale-normalized accuracy the coordinate tables are held to)."""
    from flint import arb, acb

    scale, worst = arb(0), arb(0)
    for row in grid:
        for value in row:
            if not value.is_finite():
                return 2 * work
            size = value.abs_upper()            # exact numbers: the comparisons below are decided
            if size > scale:
                scale = size
            if isinstance(value, acb):
                radius = value.real.rad()
                if value.imag.rad() > radius:
                    radius = value.imag.rad()
            else:
                radius = value.rad()
            if radius > worst:
                worst = radius
    if worst.is_zero():
        return 0
    if scale.is_zero():
        return 2 * work
    return max(int(mp.ceil(mp.log(_arb_to_mpf(worst) / _arb_to_mpf(scale), 2))) + work, 0)


def _fold_guard_bits(Delta, source_order: int) -> int:
    """First guard (bits) of a folded build at this Delta.

    The parity fold's binomial recombination cancels against the chiral coefficients' growth: measured (balls,
    c = 8, dps 60) at 5 / 35 / 45 bits for source order 21 and 14 / 77 / 113 / 106 for 63 at Delta = 2.25 / 50 /
    120 / 300 -- about 0.3 M log2(1 + Delta/4), plus a margin, in 16-bit steps (a batch is built at the largest of its
    Deltas' guards).
    """
    size = abs(mp.mpc(Delta))
    guess = 0.3 * int(source_order) * float(mp.log(1 + size / 4, 2)) + 24
    return 16 * int(mp.ceil(guess / 16))


def _build_folded_many_flint(
    evaluate: Callable,
    deltas: Sequence,
    ell: int,
    m_max: int,
    n_max: int,
    *,
    certify_bits: int | None = None,
):
    """Fused flint-native folded (w, s) builder, or ``None`` to fall back.

    Mirrors ``_build_G_table_many_from_evaluator`` followed by
    :func:`_fold_parity`, but keeps the chiral coefficients, the 2D
    outer-product assembly, and the parity fold entirely in flint --
    converting to mpmath only once per output entry. This removes the two
    field round trips (one in the G-table outer product, one in the fold) that
    otherwise dominate the per-Delta cost.

    Real Kac data + real Delta runs in ``arb`` (output ``mp.mpf``); complex
    Delta, complex c, or a Kac-pole mpmath-H fallback runs in ``acb`` (output
    ``mp.mpc``) -- the evaluator hands back native arb / acb coefficients via
    ``return_arb`` / ``return_acb``. Each sampled Delta picks its field
    independently. Returns ``None`` when the evaluator declines a weight in
    both fields.

    With ``certify_bits`` the result is ``(grids, deficits)``: the chiral coefficients enter as balls of at least
    their rounding at ``certify_bits`` (:func:`_with_rounding_radius`) and ``deficits[i]`` is how many bits the i-th
    folded grid falls short of ``certify_bits`` (scale-normalized, :func:`_grid_deficit`); the caller repeats the
    build with more guard bits where it falls short (:func:`_G_folded_table_many`).
    """
    from flint import arb, arb_mat, acb, acb_mat

    def _keep(value):
        return value

    def _prepared(coefficients):
        if certify_bits is None or coefficients is None:
            return coefficients
        return _with_rounding_radius(coefficients, certify_bits)

    def _finish(grids):
        if certify_bits is None:
            return grids
        deficits = [_grid_deficit(grid, certify_bits) for grid in grids]
        out = [[[+(_arb_to_mpf(value) if isinstance(value, arb) else _acb_to_mpc(value)) for value in row]
                for row in grid] for grid in grids]
        return out, deficits

    odd_factor = fold_parity_marker(ell=ell)
    M_total = folded_source_order(m_max, n_max, odd_factor)
    sign = scalar_G_swap_coefficient(ell)
    # the spin-0 block is (1 + sign) c_h ⊗ c_h (see _conventions)
    doubled = ell == 0 and sign != 0
    plan, target_size = _fold_plan(
        bool(odd_factor), M_total + 1, M_total + 1, m_max, n_max
    )
    width = n_max + 1

    def _fold_one(c_h, c_hbar, scalar, mat_cls, to_mp):
        sign_s = scalar(sign)
        two_s = scalar(2)
        col_h = mat_cls([[c_h[m]] for m in range(M_total + 1)])
        row_hbar = mat_cls([[c_hbar[n] for n in range(M_total + 1)]])
        mat = col_h * row_hbar
        if ell == 0 and doubled:
            def _entry(m, n, _mat=mat):
                return _mat[m, n] * two_s
        elif ell == 0:
            def _entry(m, n, _mat=mat):
                return _mat[m, n]
        else:
            col_hbar = mat_cls([[c_hbar[m]] for m in range(M_total + 1)])
            row_h = mat_cls([[c_h[n] for n in range(M_total + 1)]])
            mat2 = col_hbar * row_h

            def _entry(m, n, _mat=mat, _mat2=mat2):
                return _mat[m, n] + sign_s * _mat2[m, n]

        accum = [scalar(0) for _ in range(target_size)]
        for m, n, contribs in plan:
            e = _entry(m, n)
            for target_idx, weight in contribs:
                accum[target_idx] += e * weight
        convert = _keep if certify_bits is not None else to_mp
        return [
            [convert(accum[alpha * width + beta]) for beta in range(width)]
            for alpha in range(m_max + 1)
        ]

    deltas = [_coerce_delta_scalar(Delta) for Delta in deltas]

    # Batched real-arb fast path: one evaluator.many pipeline computes the
    # chiral coefficients for every sample Delta at once (the h-recursion is the
    # same compiled matmul for all h), avoiding the per-Delta Python evaluator
    # calls below. many(...) returns None for complex c, any complex Delta, or a
    # Kac-pole hit, in which case the whole batch falls back to the per-Delta
    # loop (which picks arb/acb independently per sample).
    plan_key = (odd_factor, M_total + 1, M_total + 1, m_max, n_max)
    many = getattr(evaluate, "many", None)
    if many is not None:
        c_h_many = many([_exchange_weight(Delta, ell) for Delta in deltas])
        if c_h_many is not None:
            c_h_many = [_prepared(row) for row in c_h_many]
            if ell == 0:
                c_hbar_many = c_h_many
            else:
                c_hbar_many = many([_exchange_weight(Delta, -ell) for Delta in deltas])
                if c_hbar_many is not None:
                    c_hbar_many = [_prepared(row) for row in c_hbar_many]
            if c_hbar_many is not None:
                n_deltas = len(c_h_many)
                if ell == 0 and doubled:
                    def entry_batch(m, n, _ch=c_h_many):
                        return [_ch[j][m] * _ch[j][n] * 2 for j in range(n_deltas)]
                elif ell == 0:
                    def entry_batch(m, n, _ch=c_h_many):
                        return [_ch[j][m] * _ch[j][n] for j in range(n_deltas)]
                else:
                    def entry_batch(m, n, _ch=c_h_many, _chb=c_hbar_many):
                        return [
                            _ch[j][m] * _chb[j][n] + sign * (_chb[j][m] * _ch[j][n])
                            for j in range(n_deltas)
                        ]
                return _finish(_fold_batch_to_grids(
                    plan, plan_key, n_deltas, m_max, n_max,
                    entry_batch=entry_batch, requested=None,
                    complex_field=False, to_out=_keep if certify_bits is not None else _arb_to_mpf, zero=arb(0),
                ))

    out = []
    for Delta in deltas:
        h = _exchange_weight(Delta, ell)
        c_h = evaluate(h, return_arb=True)
        if c_h is not None:
            scalar, mat_cls, to_mp = arb, arb_mat, _arb_to_mpf
        else:
            # Complex Delta / complex c / Kac-pole fallback: native acb.
            c_h = evaluate(h, return_acb=True)
            if c_h is None:
                return None
            scalar, mat_cls, to_mp = acb, acb_mat, _acb_to_mpc
        c_h = _prepared(c_h)

        if ell == 0:
            c_hbar = c_h
        else:
            hbar = _exchange_weight(Delta, -ell)
            c_hbar = (
                evaluate(hbar, return_arb=True)
                if scalar is arb
                else evaluate(hbar, return_acb=True)
            )
            if c_hbar is None:
                return None
            c_hbar = _prepared(c_hbar)

        out.append(_fold_one(c_h, c_hbar, scalar, mat_cls, to_mp))
    return _finish(out)


def _certified_folded_many(evaluator_factory, deltas, ell, h_externals, m_max, n_max, M_total, *, b, c=None, q_order,
                           chiral_norm, z_shift):
    """The fused folded grids, each certified to the working precision (scale-normalized), or None to fall back.

    The parity fold's binomial recombination cancels against the chiral coefficients' large-Delta growth (it cost
    21 of 60 digits at Delta = 300, xt, c = 8, Lambda 21, while zzb / yyb kept 58), so the batch is built with the
    largest first guard of its Deltas (:func:`_fold_guard_bits`; one evaluator, one H context at that precision)
    and every Delta whose grid still falls short is rebuilt with its own measured deficit plus 32 bits, until each
    is certified.
    """
    from .recursion import _pin_mode

    resolved_b = _resolve_b(b, c)
    deltas = [_coerce_delta_scalar(value) for value in deltas]
    if not deltas:
        return []
    # the arithmetic mode is classified once, at the working precision: the guarded passes rebuild the recursion from
    # the same b, which at more bits would read a strip-real b as generic complex
    b_typed = _maybe_real_scalar(resolved_b) if isinstance(resolved_b, mp.mpc) else resolved_b
    h_ext_typed = _kac_input_values(h_externals)
    with _pin_mode(b_typed, h_ext_typed):
        return _certified_folded_many_pinned(evaluator_factory, deltas, ell, h_externals, m_max, n_max, M_total, b=b, c=c,
                                             q_order=q_order, chiral_norm=chiral_norm, z_shift=z_shift)


def _certified_folded_many_pinned(evaluator_factory, deltas, ell, h_externals, m_max, n_max, M_total, *, b, c=None, q_order,
                                  chiral_norm, z_shift):
    from . import poles as _poles

    work = mp.mp.prec
    guard = max(_fold_guard_bits(Delta, M_total) for Delta in deltas)
    shortfall = [None] * len(deltas)
    results = [None] * len(deltas)
    pending = list(range(len(deltas)))
    guards = {i: guard for i in pending}
    while pending:
        next_pending = []
        for guard in sorted({guards[i] for i in pending}):
            members = [i for i in pending if guards[i] == guard]
            with mp.workprec(work + guard):
                evaluate = evaluator_factory(
                    h_externals, M_total + 1, b=b, c=c, q_order=q_order, chiral_norm=chiral_norm, z_shift=z_shift,
                )
                fused = _build_folded_many_flint(
                    evaluate, [deltas[i] for i in members], ell, m_max, n_max, certify_bits=work + guard,
                )
            if fused is None:
                return None
            grids, deficits = fused
            for i, grid, deficit in zip(members, grids, deficits):
                short = deficit - guard
                if short <= 0:
                    results[i] = [[+value for value in row] for row in grid]
                    continue
                step = _poles.next_certification_prec(work + guard, short, shortfall[i], work)
                if step is None:
                    raise ValueError(
                        f"folded coordinate table at Delta = {mp.nstr(deltas[i], 12)}: the parity fold cannot be "
                        f"certified to {mp.mp.dps} digits ({short} bits short with {guard} guard bits)"
                    )
                shortfall[i] = step[1]
                guards[i] = 32 * int(mp.ceil(mp.mpf(step[0] - work) / 32))
                next_pending.append(i)
        pending = next_pending
    return results


def _G_folded_table_many(
    base_many: Callable,
    deltas: Sequence,
    ell: int,
    h_externals: Sequence,
    m_max: int,
    n_max: int,
    *,
    b,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
    evaluator_factory: Callable | None = None,
):
    odd_factor = fold_parity_marker(ell=ell)
    M_total = folded_source_order(m_max, n_max, odd_factor)

    # Fast path: assemble + fold entirely in flint (arb for real Delta, acb for
    # complex) when the coordinate exposes a series evaluator. Falls through
    # only if the evaluator declines a weight in both fields.
    if evaluator_factory is not None:
        fused = _certified_folded_many(
            evaluator_factory, deltas, ell, h_externals, m_max, n_max, M_total,
            b=b, c=c, q_order=q_order, chiral_norm=chiral_norm, z_shift=z_shift,
        )
        if fused is not None:
            return fused

    uubar_tables = base_many(
        deltas,
        ell,
        h_externals,
        M_total,
        M_total,
        b=b, c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
    )
    return [_fold_parity(table, odd_factor, m_max, n_max) for table in uubar_tables]


# ws (parity-folded over y): w = (y + ybar)/2, s = ((y - ybar)/2)^2


def G_ws_table(
    Delta,
    ell: int,
    h_externals: Sequence,
    m_max: int,
    n_max: int | None = None,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
):
    """Parity-folded (w, s) Taylor coefficients of G_{Delta, ell}.

    w = (y + ybar)/2, s = ((y - ybar)/2)^2, expanded around w = s = 0.

    G(z, zbar) = sum_{a, b} table[a][b] w^a s^b near w = s = 0 at every ell:
    the scalar G is z <-> zbar even, so no r = (y - ybar)/2 = sqrt(s) factor
    is stripped (that is the odd parity_projected_B '-' block).

    Scalar externals only (the caller is responsible). For spinning externals
    use the non-folded `zzb` table.
    """
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    _check_externals(h_externals)
    ell = _check_ell(ell)
    return _G_folded_table(
        chiral_y_taylor,
        Delta,
        ell,
        h_externals,
        m_max,
        n_max,
        b=b, c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
    )


def G_ws_table_many(
    deltas: Sequence,
    ell: int,
    h_externals: Sequence,
    m_max: int,
    n_max: int | None = None,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
):
    """Batched :func:`G_ws_table` via the shared yyb batched builder."""
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    _check_externals(h_externals)
    ell = _check_ell(ell)
    return _G_folded_table_many(
        G_yyb_table_many,
        deltas,
        ell,
        h_externals,
        m_max,
        n_max,
        b=b, c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
        evaluator_factory=chiral_y_series_evaluator,
    )


# xt (parity-folded over z): x = (z + zbar)/2 - 1/2, t = ((z - zbar)/2)^2


def G_xt_table(
    Delta,
    ell: int,
    h_externals: Sequence,
    m_max: int,
    n_max: int | None = None,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
):
    """Parity-folded (x, t) Taylor coefficients of G_{Delta, ell} in the
    z coordinate.

    Expansion point: z = zbar = 1/2, so x = t = 0. table[a][b] is the Taylor
    coefficient of x^a t^b in G̃(x, t).
    """
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    _check_externals(h_externals)
    ell = _check_ell(ell)
    return _G_folded_table(
        chiral_z_taylor,
        Delta,
        ell,
        h_externals,
        m_max,
        n_max,
        b=b, c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
    )


def G_xt_table_many(
    deltas: Sequence,
    ell: int,
    h_externals: Sequence,
    m_max: int,
    n_max: int | None = None,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
):
    """Batched :func:`G_xt_table` via the shared zzb batched builder."""
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    _check_externals(h_externals)
    ell = _check_ell(ell)
    return _G_folded_table_many(
        G_zzb_table_many,
        deltas,
        ell,
        h_externals,
        m_max,
        n_max,
        b=b, c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
        evaluator_factory=chiral_z_series_evaluator,
    )


# qqb_ws (parity-folded over q): w_q = (q + qbar)/2 - q_*,
# s_q = ((q - qbar)/2)^2


def G_qqb_ws_table(
    Delta,
    ell: int,
    h_externals: Sequence,
    m_max: int,
    n_max: int | None = None,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
):
    """Parity-folded (w_q, s_q) Taylor coefficients of G_{Delta, ell} in the
    q-nome coord. Expansion around w_q = s_q = 0 (i.e. q = qbar = exp(-pi))."""
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    _check_externals(h_externals)
    ell = _check_ell(ell)
    return _G_folded_table(
        chiral_q_taylor,
        Delta,
        ell,
        h_externals,
        m_max,
        n_max,
        b=b, c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
    )


def G_qqb_ws_table_many(
    deltas: Sequence,
    ell: int,
    h_externals: Sequence,
    m_max: int,
    n_max: int | None = None,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
):
    """Batched :func:`G_qqb_ws_table` via the shared qqb batched builder."""
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    _check_externals(h_externals)
    ell = _check_ell(ell)
    return _G_folded_table_many(
        G_qqb_table_many,
        deltas,
        ell,
        h_externals,
        m_max,
        n_max,
        b=b, c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
        evaluator_factory=chiral_q_series_evaluator,
    )


# Radial-only slices: the 2D (w, s) table at s = 0 (i.e. z = zbar).
#
# Calling the 2D parent with n_max = 0 already minimizes the chiral Taylor
# order (M_total = m_max + odd_factor instead of 3*m_max + odd_factor for
# n_max = m_max), so the radial wrappers are thin: extract column 0.


def _radial_slice(table_2d):
    return [row[0] for row in table_2d]


def G_ws_radial_table(
    Delta,
    ell: int,
    h_externals: Sequence,
    m_max: int,
    *,
    b=None,
    c=None,
    q_order: int,
):
    """Radial slice of G_ws_table at s = 0 (y = ybar). Length m_max + 1."""
    return _radial_slice(
        G_ws_table(Delta, ell, h_externals, m_max, 0, b=b, c=c, q_order=q_order)
    )


def G_xt_radial_table(
    Delta,
    ell: int,
    h_externals: Sequence,
    m_max: int,
    *,
    b=None,
    c=None,
    q_order: int,
):
    """Radial slice of G_xt_table at t = 0 (z = zbar). Length m_max + 1."""
    return _radial_slice(
        G_xt_table(Delta, ell, h_externals, m_max, 0, b=b, c=c, q_order=q_order)
    )


def G_qqb_ws_radial_table(
    Delta,
    ell: int,
    h_externals: Sequence,
    m_max: int,
    *,
    b=None,
    c=None,
    q_order: int,
):
    """Radial slice of G_qqb_ws_table at s_q = 0 (q = qbar). Length m_max + 1."""
    return _radial_slice(
        G_qqb_ws_table(Delta, ell, h_externals, m_max, 0, b=b, c=c, q_order=q_order)
    )


# ===========================================================================
# Numeric parity-projected B tables (Virasoro + global) and the global ordered
# / parity-projected spinning blocks. These mirror the symbolic-in-Delta
# `polyderivs` builders of the same name, evaluated numerically at a single
# Delta -- the per-Delta numeric analog used as controls and as independent
# cross-checks of the interpolated / symbolic paths. The parity-projected
# block is the ordered-B combination
#
#     B_parity = B(h, hbar) + sign * phase * B(hbar, h),
#
# with `sign = +1/-1` from the projection and `phase` the external parity
# reversal phase. Folded coordinates strip the same r = sqrt(s) smoothing
# factor as the scalar tables (parity marker from the projection sign, not the
# spin), via `_fold_parity`.
# ===========================================================================


def _build_parity_projected_B_numeric_table(
    chiral_func: Callable,
    Delta,
    ell,
    h_externals: Sequence,
    hbar_externals: Sequence,
    m_max: int,
    n_max: int,
    *,
    parity: str,
    parity_external=None,
    exchanged_parity: str = "1",
    b,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
    zbar_shift=None,
):
    """Non-folded numeric parity-projected B table from two chiral expansions.

    ``B_parity[m][n] = c_h[m] c_hbar[n] + (sign * phase) c_hbar[m] c_h[n]``
    with ``h = (Delta + ell)/2``, ``hbar = (Delta - ell)/2``. Mirrors
    :func:`polyderivs._build_parity_projected_B_poly_table` numerically.
    """
    if parity not in {"+", "-"}:
        raise ValueError("parity must be '+' or '-'")
    Delta = _coerce_delta_scalar(Delta)
    ell = _check_ell_signed(ell)
    h = _exchange_weight(Delta, ell)
    hbar = _exchange_weight(Delta, -ell)
    n_top = max(m_max, n_max) + 1
    c_h = chiral_func(
        h, h_externals, n_top, b=b, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift,
    )
    c_hbar = chiral_func(
        hbar, hbar_externals, n_top, b=b, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=zbar_shift,
    )
    phase = parity_reversal_phase(parity_external, exchanged_parity=exchanged_parity)
    sign = mp.mpf(1) if parity == "+" else mp.mpf(-1)
    swapped = sign * phase
    return _assemble_block_grid(c_h, c_hbar, m_max, n_max, sigma=swapped)


def _parity_projected_B_folded_numeric(
    chiral_func: Callable,
    Delta,
    ell,
    h_externals: Sequence,
    hbar_externals: Sequence,
    m_max: int,
    n_max: int,
    *,
    parity: str,
    parity_external=None,
    exchanged_parity: str = "1",
    b,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
    zbar_shift=None,
):
    """Folded (w, s) numeric parity-projected B table. The smoothing parity is
    the projection sign (``-`` strips one r = sqrt(s)), so the non-folded grid
    is built to the expanded order ``m_max + 2 n_max + (parity == '-')`` and
    folded via :func:`_fold_parity`. The total parity phase must be real."""
    if parity not in {"+", "-"}:
        raise ValueError("parity must be '+' or '-'")
    phase = parity_reversal_phase(parity_external, exchanged_parity=exchanged_parity)
    if abs(mp.im(phase)) > _validation_tolerance(guard=5):
        raise ValueError(
            "parity-folded coordinates require a real (+/-1) total parity "
            "phase; got a non-real phase, which has no definite z<->zbar parity "
            "-- use a non-folded coordinate"
        )
    parity_marker = fold_parity_marker(parity=parity, phase=phase)
    M_total = folded_source_order(m_max, n_max, parity_marker)
    full = _build_parity_projected_B_numeric_table(
        chiral_func, Delta, ell, h_externals, hbar_externals, M_total, M_total,
        parity=parity, parity_external=parity_external,
        exchanged_parity=exchanged_parity, b=b, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift, zbar_shift=zbar_shift,
    )
    return _fold_parity(full, parity_marker, m_max, n_max)


def _parity_projected_B_public(
    chiral_func, Delta, ell, h_externals, hbar_externals, m_max, n_max,
    *, parity, parity_external, exchanged_parity, b, c, q_order,
    chiral_norm, z_shift, zbar_shift, folded,
):
    """Shared front end for the public parity-projected B builders: validates,
    resolves b, and dispatches to the non-folded or folded numeric core."""
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    _check_externals(h_externals)
    _check_externals(hbar_externals)
    # The global chiral builders ignore `b`/`c` (they take both as None); only
    # resolve for the Virasoro path, where exactly one of b/c is supplied.
    if not (b is None and c is None):
        b = _resolve_b(b, c)
    builder = (
        _parity_projected_B_folded_numeric if folded
        else _build_parity_projected_B_numeric_table
    )
    return builder(
        chiral_func, Delta, ell, h_externals, hbar_externals, m_max, n_max,
        parity=parity, parity_external=parity_external,
        exchanged_parity=exchanged_parity, b=b, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift, zbar_shift=zbar_shift,
    )


# --- Virasoro parity-projected B: non-folded zzb / yyb / qqb ---------------


def parity_projected_B_zzb_table(
    Delta, ell, h_externals, hbar_externals, m_max, n_max=None,
    *, parity, parity_external=None, exchanged_parity="1", b=None, c=None,
    q_order: int, chiral_norm=None, z_shift=None, zbar_shift=None,
):
    """Numeric parity-projected B table in (z, zbar) at z = zbar = 1/2."""
    return _parity_projected_B_public(
        chiral_z_taylor, Delta, ell, h_externals, hbar_externals, m_max, n_max,
        parity=parity, parity_external=parity_external,
        exchanged_parity=exchanged_parity, b=b, c=c, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift, zbar_shift=zbar_shift,
        folded=False,
    )


def parity_projected_B_yyb_table(
    Delta, ell, h_externals, hbar_externals, m_max, n_max=None,
    *, parity, parity_external=None, exchanged_parity="1", b=None, c=None,
    q_order: int, chiral_norm=None, z_shift=None, zbar_shift=None,
):
    """Numeric parity-projected B table in (y, ybar) at y = ybar = 0."""
    return _parity_projected_B_public(
        chiral_y_taylor, Delta, ell, h_externals, hbar_externals, m_max, n_max,
        parity=parity, parity_external=parity_external,
        exchanged_parity=exchanged_parity, b=b, c=c, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift, zbar_shift=zbar_shift,
        folded=False,
    )


def parity_projected_B_qqb_table(
    Delta, ell, h_externals, hbar_externals, m_max, n_max=None,
    *, parity, parity_external=None, exchanged_parity="1", b=None, c=None,
    q_order: int, chiral_norm=None, z_shift=None, zbar_shift=None,
):
    """Numeric parity-projected B table in (q, qbar) at q = qbar = exp(-pi)."""
    return _parity_projected_B_public(
        chiral_q_taylor, Delta, ell, h_externals, hbar_externals, m_max, n_max,
        parity=parity, parity_external=parity_external,
        exchanged_parity=exchanged_parity, b=b, c=c, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift, zbar_shift=zbar_shift,
        folded=False,
    )


# --- Virasoro parity-projected B: folded xt / ws / qqb_ws ------------------


def parity_projected_B_xt_table(
    Delta, ell, h_externals, hbar_externals, m_max, n_max=None,
    *, parity, parity_external=None, exchanged_parity="1", b=None, c=None,
    q_order: int, chiral_norm=None, z_shift=None, zbar_shift=None,
):
    """Folded (x, t) numeric parity-projected B table (folds the z block)."""
    return _parity_projected_B_public(
        chiral_z_taylor, Delta, ell, h_externals, hbar_externals, m_max, n_max,
        parity=parity, parity_external=parity_external,
        exchanged_parity=exchanged_parity, b=b, c=c, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift, zbar_shift=zbar_shift,
        folded=True,
    )


def parity_projected_B_ws_table(
    Delta, ell, h_externals, hbar_externals, m_max, n_max=None,
    *, parity, parity_external=None, exchanged_parity="1", b=None, c=None,
    q_order: int, chiral_norm=None, z_shift=None, zbar_shift=None,
):
    """Folded (w, s) numeric parity-projected B table (folds the y block)."""
    return _parity_projected_B_public(
        chiral_y_taylor, Delta, ell, h_externals, hbar_externals, m_max, n_max,
        parity=parity, parity_external=parity_external,
        exchanged_parity=exchanged_parity, b=b, c=c, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift, zbar_shift=zbar_shift,
        folded=True,
    )


def parity_projected_B_qqb_ws_table(
    Delta, ell, h_externals, hbar_externals, m_max, n_max=None,
    *, parity, parity_external=None, exchanged_parity="1", b=None, c=None,
    q_order: int, chiral_norm=None, z_shift=None, zbar_shift=None,
):
    """Folded (w_q, s_q) numeric parity-projected B table (folds the q block)."""
    return _parity_projected_B_public(
        chiral_q_taylor, Delta, ell, h_externals, hbar_externals, m_max, n_max,
        parity=parity, parity_external=parity_external,
        exchanged_parity=exchanged_parity, b=b, c=c, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift, zbar_shift=zbar_shift,
        folded=True,
    )


# --- Virasoro parity-projected B: radial slices (s = 0 column) -------------


def parity_projected_B_xt_radial_table(
    Delta, ell, h_externals, hbar_externals, m_max,
    *, parity, parity_external=None, exchanged_parity="1", b=None, c=None,
    q_order: int, chiral_norm=None, z_shift=None, zbar_shift=None,
):
    """Radial slice (t = 0, z = zbar) of :func:`parity_projected_B_xt_table`."""
    return _radial_slice(parity_projected_B_xt_table(
        Delta, ell, h_externals, hbar_externals, m_max, 0, parity=parity,
        parity_external=parity_external, exchanged_parity=exchanged_parity,
        b=b, c=c, q_order=q_order, chiral_norm=chiral_norm, z_shift=z_shift,
        zbar_shift=zbar_shift,
    ))


def parity_projected_B_ws_radial_table(
    Delta, ell, h_externals, hbar_externals, m_max,
    *, parity, parity_external=None, exchanged_parity="1", b=None, c=None,
    q_order: int, chiral_norm=None, z_shift=None, zbar_shift=None,
):
    """Radial slice (s = 0, y = ybar) of :func:`parity_projected_B_ws_table`."""
    return _radial_slice(parity_projected_B_ws_table(
        Delta, ell, h_externals, hbar_externals, m_max, 0, parity=parity,
        parity_external=parity_external, exchanged_parity=exchanged_parity,
        b=b, c=c, q_order=q_order, chiral_norm=chiral_norm, z_shift=z_shift,
        zbar_shift=zbar_shift,
    ))


def parity_projected_B_qqb_ws_radial_table(
    Delta, ell, h_externals, hbar_externals, m_max,
    *, parity, parity_external=None, exchanged_parity="1", b=None, c=None,
    q_order: int, chiral_norm=None, z_shift=None, zbar_shift=None,
):
    """Radial slice (s_q = 0, q = qbar) of
    :func:`parity_projected_B_qqb_ws_table`."""
    return _radial_slice(parity_projected_B_qqb_ws_table(
        Delta, ell, h_externals, hbar_externals, m_max, 0, parity=parity,
        parity_external=parity_external, exchanged_parity=exchanged_parity,
        b=b, c=c, q_order=q_order, chiral_norm=chiral_norm, z_shift=z_shift,
        zbar_shift=zbar_shift,
    ))


# --- global ordered B: zzb / yyb / qqb ------------------------------------
#
# The global SL(2) chiral builders accept (and ignore) `b`/`q_order`, so the
# Virasoro `_build_B_table` assembler is reused directly with the global chiral
# function.


def global_B_zzb_table(
    h, hbar, h_externals, hbar_externals, m_max, n_max=None,
    *, chiral_norm=None, z_shift=None, zbar_shift=None,
):
    """Numeric global ordered B table in (z, zbar) at z = zbar = 1/2."""
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    return _build_B_table(
        global_chiral_z_taylor, h, hbar, h_externals, hbar_externals,
        m_max, n_max, b=None, q_order=0, chiral_norm=chiral_norm,
        z_shift=z_shift, zbar_shift=zbar_shift,
    )


def global_B_yyb_table(
    h, hbar, h_externals, hbar_externals, m_max, n_max=None,
    *, chiral_norm=None, z_shift=None, zbar_shift=None,
):
    """Numeric global ordered B table in (y, ybar) at y = ybar = 0."""
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    return _build_B_table(
        global_chiral_y_taylor, h, hbar, h_externals, hbar_externals,
        m_max, n_max, b=None, q_order=0, chiral_norm=chiral_norm,
        z_shift=z_shift, zbar_shift=zbar_shift,
    )


def global_B_qqb_table(
    h, hbar, h_externals, hbar_externals, m_max, n_max=None,
    *, chiral_norm=None, z_shift=None, zbar_shift=None,
):
    """Numeric global ordered B table in (q, qbar) at q = qbar = exp(-pi)."""
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    return _build_B_table(
        global_chiral_q_taylor, h, hbar, h_externals, hbar_externals,
        m_max, n_max, b=None, q_order=0, chiral_norm=chiral_norm,
        z_shift=z_shift, zbar_shift=zbar_shift,
    )


# --- global parity-projected B: non-folded, folded, radial ----------------


def _global_parity_projected_B(
    chiral_func, Delta, ell, h_externals, hbar_externals, m_max, n_max,
    *, parity, parity_external, exchanged_parity, chiral_norm, z_shift,
    zbar_shift, folded,
):
    return _parity_projected_B_public(
        chiral_func, Delta, ell, h_externals, hbar_externals, m_max, n_max,
        parity=parity, parity_external=parity_external,
        exchanged_parity=exchanged_parity, b=None, c=None, q_order=0,
        chiral_norm=chiral_norm, z_shift=z_shift, zbar_shift=zbar_shift,
        folded=folded,
    )


def global_parity_projected_B_zzb_table(
    Delta, ell, h_externals, hbar_externals, m_max, n_max=None,
    *, parity, parity_external=None, exchanged_parity="1",
    chiral_norm=None, z_shift=None, zbar_shift=None,
):
    """Numeric global parity-projected B table in (z, zbar)."""
    return _global_parity_projected_B(
        global_chiral_z_taylor, Delta, ell, h_externals, hbar_externals,
        m_max, n_max, parity=parity, parity_external=parity_external,
        exchanged_parity=exchanged_parity, chiral_norm=chiral_norm,
        z_shift=z_shift, zbar_shift=zbar_shift, folded=False,
    )


def global_parity_projected_B_yyb_table(
    Delta, ell, h_externals, hbar_externals, m_max, n_max=None,
    *, parity, parity_external=None, exchanged_parity="1",
    chiral_norm=None, z_shift=None, zbar_shift=None,
):
    """Numeric global parity-projected B table in (y, ybar)."""
    return _global_parity_projected_B(
        global_chiral_y_taylor, Delta, ell, h_externals, hbar_externals,
        m_max, n_max, parity=parity, parity_external=parity_external,
        exchanged_parity=exchanged_parity, chiral_norm=chiral_norm,
        z_shift=z_shift, zbar_shift=zbar_shift, folded=False,
    )


def global_parity_projected_B_qqb_table(
    Delta, ell, h_externals, hbar_externals, m_max, n_max=None,
    *, parity, parity_external=None, exchanged_parity="1",
    chiral_norm=None, z_shift=None, zbar_shift=None,
):
    """Numeric global parity-projected B table in (q, qbar)."""
    return _global_parity_projected_B(
        global_chiral_q_taylor, Delta, ell, h_externals, hbar_externals,
        m_max, n_max, parity=parity, parity_external=parity_external,
        exchanged_parity=exchanged_parity, chiral_norm=chiral_norm,
        z_shift=z_shift, zbar_shift=zbar_shift, folded=False,
    )


def global_parity_projected_B_xt_table(
    Delta, ell, h_externals, hbar_externals, m_max, n_max=None,
    *, parity, parity_external=None, exchanged_parity="1",
    chiral_norm=None, z_shift=None, zbar_shift=None,
):
    """Folded (x, t) numeric global parity-projected B table."""
    return _global_parity_projected_B(
        global_chiral_z_taylor, Delta, ell, h_externals, hbar_externals,
        m_max, n_max, parity=parity, parity_external=parity_external,
        exchanged_parity=exchanged_parity, chiral_norm=chiral_norm,
        z_shift=z_shift, zbar_shift=zbar_shift, folded=True,
    )


def global_parity_projected_B_ws_table(
    Delta, ell, h_externals, hbar_externals, m_max, n_max=None,
    *, parity, parity_external=None, exchanged_parity="1",
    chiral_norm=None, z_shift=None, zbar_shift=None,
):
    """Folded (w, s) numeric global parity-projected B table."""
    return _global_parity_projected_B(
        global_chiral_y_taylor, Delta, ell, h_externals, hbar_externals,
        m_max, n_max, parity=parity, parity_external=parity_external,
        exchanged_parity=exchanged_parity, chiral_norm=chiral_norm,
        z_shift=z_shift, zbar_shift=zbar_shift, folded=True,
    )


def global_parity_projected_B_qqb_ws_table(
    Delta, ell, h_externals, hbar_externals, m_max, n_max=None,
    *, parity, parity_external=None, exchanged_parity="1",
    chiral_norm=None, z_shift=None, zbar_shift=None,
):
    """Folded (w_q, s_q) numeric global parity-projected B table."""
    return _global_parity_projected_B(
        global_chiral_q_taylor, Delta, ell, h_externals, hbar_externals,
        m_max, n_max, parity=parity, parity_external=parity_external,
        exchanged_parity=exchanged_parity, chiral_norm=chiral_norm,
        z_shift=z_shift, zbar_shift=zbar_shift, folded=True,
    )


def global_parity_projected_B_xt_radial_table(
    Delta, ell, h_externals, hbar_externals, m_max,
    *, parity, parity_external=None, exchanged_parity="1",
    chiral_norm=None, z_shift=None, zbar_shift=None,
):
    """Radial slice of :func:`global_parity_projected_B_xt_table`."""
    return _radial_slice(global_parity_projected_B_xt_table(
        Delta, ell, h_externals, hbar_externals, m_max, 0, parity=parity,
        parity_external=parity_external, exchanged_parity=exchanged_parity,
        chiral_norm=chiral_norm, z_shift=z_shift, zbar_shift=zbar_shift,
    ))


def global_parity_projected_B_ws_radial_table(
    Delta, ell, h_externals, hbar_externals, m_max,
    *, parity, parity_external=None, exchanged_parity="1",
    chiral_norm=None, z_shift=None, zbar_shift=None,
):
    """Radial slice of :func:`global_parity_projected_B_ws_table`."""
    return _radial_slice(global_parity_projected_B_ws_table(
        Delta, ell, h_externals, hbar_externals, m_max, 0, parity=parity,
        parity_external=parity_external, exchanged_parity=exchanged_parity,
        chiral_norm=chiral_norm, z_shift=z_shift, zbar_shift=zbar_shift,
    ))


def global_parity_projected_B_qqb_ws_radial_table(
    Delta, ell, h_externals, hbar_externals, m_max,
    *, parity, parity_external=None, exchanged_parity="1",
    chiral_norm=None, z_shift=None, zbar_shift=None,
):
    """Radial slice of :func:`global_parity_projected_B_qqb_ws_table`."""
    return _radial_slice(global_parity_projected_B_qqb_ws_table(
        Delta, ell, h_externals, hbar_externals, m_max, 0, parity=parity,
        parity_external=parity_external, exchanged_parity=exchanged_parity,
        chiral_norm=chiral_norm, z_shift=z_shift, zbar_shift=zbar_shift,
    ))


# --- global scalar G: folded qqb_ws + radial completions ------------------


def global_G_qqb_ws_table(
    Delta, ell, h_externals, m_max, n_max=None,
    *, chiral_norm=None, z_shift=None,
):
    """Parity-folded (w_q, s_q) global scalar block (folds the global q block).

    The global analogue of :func:`G_qqb_ws_table`; scalar externals only."""
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    _check_externals(h_externals)
    ell = _check_ell(ell)

    def _global_chiral(h, ext, n_terms, *, b, q_order, chiral_norm=None, z_shift=None):
        return global_chiral_q_taylor(
            h, ext, n_terms, chiral_norm=chiral_norm, z_shift=z_shift,
        )

    return _G_folded_table(
        _global_chiral, Delta, ell, h_externals, m_max, n_max,
        b=None, q_order=0, chiral_norm=chiral_norm, z_shift=z_shift,
    )


def global_G_xt_radial_table(
    Delta, ell, h_externals, m_max, *, chiral_norm=None, z_shift=None,
):
    """Radial slice (t = 0) of :func:`global_G_xt_table`."""
    return _radial_slice(global_G_xt_table(
        Delta, ell, h_externals, m_max, 0, chiral_norm=chiral_norm,
        z_shift=z_shift,
    ))


def global_G_ws_radial_table(
    Delta, ell, h_externals, m_max, *, chiral_norm=None, z_shift=None,
):
    """Radial slice (s = 0) of :func:`global_G_ws_table`."""
    return _radial_slice(global_G_ws_table(
        Delta, ell, h_externals, m_max, 0, chiral_norm=chiral_norm,
        z_shift=z_shift,
    ))


def global_G_qqb_ws_radial_table(
    Delta, ell, h_externals, m_max, *, chiral_norm=None, z_shift=None,
):
    """Radial slice (s_q = 0) of :func:`global_G_qqb_ws_table`."""
    return _radial_slice(global_G_qqb_ws_table(
        Delta, ell, h_externals, m_max, 0, chiral_norm=chiral_norm,
        z_shift=z_shift,
    ))
