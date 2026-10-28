"""Virasoro parameterization and Kac data.

Conventions follow arXiv:1703.09805 section 2:
    c = 1 + 6 (b + 1/b)^2
    d_{r,s}(c) = (b + 1/b)^2 / 4 - (r b + s/b)^2 / 4
    h_i = (b + 1/b)^2 / 4 - lambda_i^2 / 4   (i.e. lambda_i^2 = (b+1/b)^2 - 4 h_i)

We pick the branch with Re(b) > 0 in `b_from_c`. The block depends only on
b^2, so the choice of sign cancels in `d_rs`, `A_rs`, `fusion_polynomial`.
"""

from __future__ import annotations

import mpmath as mp

from .tseries import LaurentSeries


def _coerce_b(b):
    """Pass a LaurentSeries or already-mpmath scalar through untouched; coerce
    other scalars to mp.mpc.

    The Kac functions below are written as generic arithmetic in `b`, so they
    serve three paths with this single coercion guard:
    - Laurent-in-t regularization via LaurentSeries.
    - The ordinary complex numeric path via mp.mpc.
    - The real-arithmetic fast path: when the caller passes `mp.mpf`, the
      downstream Kac quantities (d_rs, A_rs, fusion_polynomial) all stay mpf,
      cutting mpmath cost ~3x at high precision. The caller is responsible for
      only using this path when b is genuinely real.
    """
    if isinstance(b, LaurentSeries):
        return b
    if isinstance(b, (mp.mpf, mp.mpc)):
        return b
    from flint import acb

    if isinstance(b, acb):
        return b
    return mp.mpc(mp.convert(b))


def _sqrt(x):
    """Square root that dispatches on LaurentSeries vs scalar."""
    if isinstance(x, LaurentSeries):
        return x.sqrt()
    return mp.sqrt(x)


def _lead(x):
    """Constant term of a LaurentSeries, or the scalar itself."""
    if isinstance(x, LaurentSeries):
        return x.coeff(0)
    return x


def c_from_b(b):
    """c = 1 + 6 (b + 1/b)^2."""
    b = mp.mpc(mp.convert(b))
    return 1 + 6 * (b + 1 / b) ** 2


# The exact c each b returned by `b_from_c` was formed from.  A b rounded at some precision encodes c only to that
# precision, while the recursion's H table can be far more sensitive to c than that (heavy externals, high q_order,
# near a singular c: H_q_series at c = 8, h_i = 1, q_order 100, dps 30 kept -5 digits in its worst coefficient with
# a table certified for the rounded b, 34 with b a ball of the exact c), so the recursion recovers c here and forms
# its Kac data from a ball of b around the exact c (`c_origin_of_b`).
def _b_origin_key(b):
    if isinstance(b, mp.mpf):
        return (b._mpf_, mp.mpf(0)._mpf_)
    return b._mpc_


def _remember_b_origin(b, source):
    b._vb_c_origin = (_b_origin_key(b), source)


class _BSourceState:
    """Keep source labels in mpmath's otherwise numerical-only pickle state."""

    def __getstate__(self):
        return super().__getstate__(), c_origin_of_b(self)

    def __setstate__(self, state):
        numeric, source = state
        super().__setstate__(numeric)
        if source is not None:
            _remember_b_origin(self, source)


class _SourceRealB(_BSourceState, mp.mpf):
    pass


class _SourceB(_BSourceState, mp.mpc):
    @property
    def real(self):
        if self._mpc_[1] != mp.mpf(0)._mpf_:
            return super().real
        source = c_origin_of_b(self)
        if source is None:
            return super().real
        # Preserve stored bits: calling mpf(real) at a restored lower
        # precision must not round the parameter before it is transferred.
        real = object.__new__(_SourceRealB)
        real._mpf_ = self._mpc_[0]
        _remember_b_origin(real, source)
        return real


def c_origin_of_b(b):
    """The exact central charge a ``b`` returned by :func:`b_from_c` was formed from, or None (a b given directly,
    which is then exact as given). The source belongs to the returned object,
    rather than to equal numerical values. Source-carrying values retain their
    label through copying, pickling, and worker transfer."""
    if not isinstance(b, (mp.mpf, mp.mpc)):
        return None
    owned = getattr(b, "_vb_c_origin", None)
    if owned is not None and owned[0] == _b_origin_key(b):
        return owned[1]
    return None


def _c_input_of_b(b):
    """Resolve an input charge without round-tripping a known source label."""
    origin = c_origin_of_b(b)
    if origin is not None:
        return origin
    return c_from_b(b)


def b_from_c(c):
    """Inverse of c = 1 + 6 (b + 1/b)^2.

    Returns the b with Re(b) > 0 (and Im(b) >= 0 if real-part is zero).
    Note that (b, 1/b, -b, -1/b) all give the same c, and the block depends
    only on b^2, so the convention only matters for cosmetic reasons.

    Real literals are parsed exactly. The result is remembered with its
    source label (:func:`c_origin_of_b`).
    """
    from .numeric_input import parse_number

    source = parse_number(c)
    numeric = _b_from_c(mp.convert(source))
    b = object.__new__(_SourceB)
    b._mpc_ = numeric._mpc_
    # Object-owned provenance cannot attach this label to an unrelated
    # equal-valued parameter.
    _remember_b_origin(b, source)
    return b


def _b_from_c(c):
    c = mp.mpc(c)
    # u = (b + 1/b)^2 = (c-1)/6
    u = (c - 1) / 6
    # b^2 satisfies x + 1/x = u - 2 (since (b+1/b)^2 = b^2 + 2 + 1/b^2)
    # so b^2 + 1/b^2 = u - 2, i.e. b^2 is root of x^2 - (u-2) x + 1 = 0
    w = u - 2
    disc = mp.sqrt(w * w - 4)
    plus, minus = w + disc, w - disc
    # Pick |b^2| <= 1 so that |b| <= 1 (a common minimal-strip convention).
    small, large = (plus, minus) if abs(plus / 2) <= abs(minus / 2) else (minus, plus)
    # the roots multiply to 1 (plus * minus = 4): where the small one cancels (|w| >> 2, i.e. large |c|: b at
    # c = 1e20 lost every digit) take it from the large one instead
    b_sq = 2 / large if abs(large) > 2 * abs(small) else small / 2
    b = mp.sqrt(b_sq)
    if mp.re(b) < 0:
        b = -b
    return b


def _laurent_charge_input(c0):
    from .numeric_input import parse_number

    if isinstance(c0, bool):
        raise ValueError('c0 must be a finite number, not a boolean')
    try:
        source = parse_number(c0)
        if not mp.isfinite(mp.convert(source)):
            raise ValueError('nonfinite charge')
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise ValueError('c0 must be a finite number') from exc
    return source


def b_series(c0):
    """The Liouville parameter b as a Laurent series in t, where ``c = c0 + sigma t^p``
    (``p = laurent_t_power(c0)``, ``sigma = laurent_t_sq_sign(c0)``).

    At the recursion-singular loci c0 = 1 and c0 = 25 the map c -> b has a
    square-root branch point, so b is analytic in the signed square-root variable t but not in c; expanding in
    t is what makes the regularization exact. At every other locus b is analytic in c, and t = c - c0 itself
    (the series in sqrt(c - c0) would be even: half the window for the same depth).  From
    (b + 1/b)^2 = (c - 1)/6, the parameter b^2 solves
    x^2 - (u - 2) x + 1 = 0 with u = (c - 1)/6, and b is its square root on the
    branch that matches `b_from_c(c0)` at t = 0.
    """
    c0 = _laurent_charge_input(c0)
    b0 = b_from_c(c0)
    b = _b_sq_series(c0, laurent_t_sq_sign(c0)).sqrt(valuation=0)
    if abs(b.coeff(0) - b0) > abs(b.coeff(0) + b0):
        b = -b
    return b


def laurent_t_power(c0) -> int:
    """The power ``p`` of the Laurent recursion's local variable, ``c = c0 + sigma t^p``: 2 at the branch points
    ``c0 = 1`` and ``c0 = 25`` of ``c -> b`` (``b`` is analytic in ``t = sqrt(c - c0)`` there, not in ``c``), else 1
    (``b`` is analytic in ``c``, every series in ``sqrt(c - c0)`` is even, and ``t = c - c0`` halves its window)."""
    from .numeric_input import literal_components

    real, imaginary = literal_components(_laurent_charge_input(c0))
    return 2 if imaginary == 0 and real in (1, 25) else 1


def laurent_t_sq_sign(c0) -> int:
    """The sign ``sigma`` of the Laurent recursion's local variable, ``c = c0 + sigma t^p``
    (``p = laurent_t_power(c0)``).

    ``-1`` at the branch point ``c0 = 1``, so the recursion runs on the ``c < 1`` side where ``b = i beta`` with
    ``beta`` real and every series has real coefficients (on the ``c > 1`` side they alternate real/imaginary);
    ``+1`` elsewhere. Only valuations and the ``t^0`` coefficient are physical, and neither depends on the side.
    """
    from .numeric_input import literal_components

    real, imaginary = literal_components(_laurent_charge_input(c0))
    return -1 if imaginary == 0 and real == 1 else 1


def imaginary_b_locus(c0) -> bool:
    """True at a real ``c0 <= 1``: ``b = i beta`` with ``beta`` real (see :func:`beta_series`)."""
    try:
        from .numeric_input import literal_components

        real, imaginary = literal_components(_laurent_charge_input(c0))
    except (TypeError, ValueError, ArithmeticError):
        return False
    return imaginary == 0 and real <= 1


def beta_series(c0):
    """``beta = -i b`` as a real Laurent series in t, at a locus with :func:`imaginary_b_locus`.

    The local variable is ``c = c0 + laurent_t_sq_sign(c0) t^p`` (``p = laurent_t_power(c0)``).
    With it every recursion quantity is real: the
    Kac weights and fusion polynomials depend on ``b^2 = -beta^2``, and ``m b + n / b = i (m beta - n / beta)``.
    """
    c0 = _laurent_charge_input(c0)
    if not imaginary_b_locus(c0):
        raise ValueError('beta_series requires a real c0 <= 1')
    beta = (-_b_sq_series(c0, laurent_t_sq_sign(c0))).sqrt(valuation=0)
    if mp.re(beta.coeff(0)) < 0:
        beta = -beta
    return beta


def _b_sq_series(c0, t_sq_sign):
    """``b^2`` as a Laurent series in t with ``c = c0 + t_sq_sign t^p`` (``p = laurent_t_power(c0)``), on the branch
    of ``b_from_c(c0)^2``."""
    power = laurent_t_power(c0)
    u = LaurentSeries.constant((c0 - 1) / 6) + LaurentSeries.monomial(power, mp.mpf(t_sq_sign) / 6)
    w = u - 2
    # The discriminant has a constant term away from the two exact branch
    # points, even when that term is tiny beside its higher coefficients.
    disc = (w * w - 4).sqrt(valuation=2 if power == 2 else 0)
    b0 = b_from_c(c0)
    target_b_sq = b0 * b0
    plus, minus = (w + disc) / 2, (w - disc) / 2
    # The roots multiply to one. Construct the smaller root by inversion to
    # avoid catastrophic subtraction at large |c0|, retaining the branch
    # order used at c0 = 1 and c0 = 25.
    roots = ((plus, plus.reciprocal(valuation=0)) if abs(plus.coeff(0)) >= abs(minus.coeff(0))
             else (minus.reciprocal(valuation=0), minus))
    return min(roots, key=lambda x: abs(x.coeff(0) - target_b_sq))


def lambda_from_h(h, b):
    """lambda such that h = (b + 1/b)^2 / 4 - lambda^2 / 4.

    Returns the branch with Re(lambda) >= 0.
    """
    b = _coerce_b(b)
    Q = b + 1 / b
    lam_sq = Q * Q - 4 * h
    lam = _sqrt(lam_sq)
    if mp.re(_lead(lam)) < 0:
        lam = -lam
    return lam


def _kac_indices(r, s):
    """Normalize integral labels without adding conversion work to recursion's int path."""
    if type(r) is int and type(s) is int:
        return r, s
    from .requests import _integer_field

    if type(r) is not int:
        r = _integer_field("r", r)
    if type(s) is not int:
        s = _integer_field("s", s)
    return r, s


def d_rs(r: int, s: int, b):
    """Kac weight d_{r,s}(c) = (b + 1/b)^2 / 4 - (r b + s/b)^2 / 4.

    Defined for any integers r, s. The block has poles at h = d_{r,s} for r,s >= 1.
    """
    r, s = _kac_indices(r, s)
    # rounded to the working precision first: a b carrying extra digits (the resolvers form it at 2x dps for
    # large-c sensitivity) would round Q = b + 1/b and r b + s/b differently, and d_{1,1} = 0 came out 5e-233
    b = _coerce_b(b)
    if isinstance(b, (mp.mpf, mp.mpc)):
        b = +b          # (a LaurentSeries b -- the regularizer's exact analysis -- is left as is)
    Q = b + 1 / b
    return (Q * Q - (r * b + s / b) ** 2) / 4


def A_rs(r: int, s: int, b, *, b_sq_pq=None):
    """A^c_{r,s} = (1/2) prod_{(m,n) != (0,0), (r,s)} 1 / (m b + n / b),
    where m runs 1-r..r and n runs 1-s..s.

    This is eq (2.12) of arXiv:1703.09805.

    ``b_sq_pq`` is an optional ``(p, q)`` flagging that ``b`` sits at the
    minimal-model locus ``b^2 = -p/q`` (only meaningful on the Laurent-in-t
    path, where ``b`` is a :class:`LaurentSeries`). A factor ``m b + n/b``
    vanishes there iff ``b^2 = -n/m = -p/q``, i.e. ``n*q == m*p`` -- an exact
    integer condition. Because ``b^2`` is irrational and only representable to
    ~10^-dps, the factor's constant term leaks to ~10^-dps roundoff rather than
    the exact zero it must be (for non-dyadic ``q`` the structural cancellation
    ``m*b^2 + n`` is not exact in binary float). Left alone, that turns the
    clean ``t``-pole of ``A_rs`` into a ~10^dps finite term and corrupts the
    Laurent cancellation. Snapping the constant to exact zero on the exact
    integer condition restores the genuine pole order without any tolerance
    heuristic.
    """
    r, s = _kac_indices(r, s)
    if r < 1 or s < 1:
        raise ValueError("A_rs requires r, s >= 1")
    b = _coerce_b(b)
    inv_b = 1 / b
    from flint import acb

    half = acb(1) / 2 if isinstance(b, acb) else mp.mpf(1) / 2
    acc = half
    for m in range(1 - r, r + 1):
        for n in range(1 - s, s + 1):
            if (m, n) == (0, 0) or (m, n) == (r, s):
                continue
            factor = m * b + n * inv_b
            if b_sq_pq is not None and isinstance(factor, LaurentSeries):
                p, q = b_sq_pq
                if n * q == m * p:
                    factor = factor - LaurentSeries.constant(factor.coeff(0))
            acc /= factor
    return acc


def fusion_polynomial(r: int, s: int, b, h_left, h_right):
    """Fusion polynomial P^c_{r,s}[h_left, h_right] (eq 2.14 of arXiv:1703.09805).

        P = prod_{p, q} (1/4) (lambda_left + lambda_right + p b + q/b)
                              (lambda_left - lambda_right + p b + q/b)

    where p runs from 1-r to r-1 in steps of 2 (i.e. p has the same parity as r-1),
    and q runs from 1-s to s-1 in steps of 2.

    Internally we use the manifestly-real form via the (p, q) <-> (-p, -q)
    pairing identity
        [(lam_L + s)^2 - lam_R^2] * [(lam_L - s)^2 - lam_R^2]
          = (lam_L^2 + s^2 - lam_R^2)^2 - 4 lam_L^2 s^2,
    so the result stays in mp.mpf whenever lam_L^2, lam_R^2, and shift^2
    are real -- including the case lam^2 < 0 (heavy external) and the case
    of pure imaginary b. The legacy sqrt-of-lam^2 form is avoided so we
    never silently promote real inputs to mp.mpc.
    """
    r, s = _kac_indices(r, s)
    if r < 1 or s < 1:
        raise ValueError("fusion_polynomial requires r, s >= 1")
    b = _coerce_b(b)
    b_sq = b * b
    Q_sq = b_sq + 2 + 1 / b_sq  # (b + 1/b)^2 = b^2 + 2 + 1/b^2
    lam_L_sq = Q_sq - 4 * h_left
    lam_R_sq = Q_sq - 4 * h_right
    return fusion_polynomial_via_squares(r, s, b_sq, lam_L_sq, lam_R_sq)


def fusion_polynomial_via_squares(r: int, s: int, b_sq, lam_L_sq, lam_R_sq):
    """Manifestly-real form of `fusion_polynomial` taking squared inputs.

    Equivalent to `fusion_polynomial(r, s, b, d_left, d_right)` with
        b_sq = b * b
        lam_L_sq = (b + 1/b)^2 - 4 d_left
        lam_R_sq = (b + 1/b)^2 - 4 d_right.

    Pairs each (p, q) with (-p, -q) inside the product so that the linear-
    in-lambda cross-terms cancel; the surviving expression is a polynomial
    in lam_L^2, lam_R^2, and shift^2 = (p b + q/b)^2 = p^2 b^2 + 2 p q + q^2/b^2.

    All three inputs may be mp.mpf or mp.mpc (or LaurentSeries via the
    regularization path); when they are all mp.mpf the result stays in
    mp.mpf throughout. This is the entry point that lets the real_b
    (c > 25) and imag_b (c < 1) modes keep K_{r,s} in real arithmetic
    even when individual lam^2 values are negative.
    """
    r, s = _kac_indices(r, s)
    if r < 1 or s < 1:
        raise ValueError("fusion_polynomial_via_squares requires r, s >= 1")
    inv_b_sq = 1 / b_sq
    from flint import acb

    if isinstance(b_sq, LaurentSeries) or isinstance(lam_L_sq, LaurentSeries) or isinstance(lam_R_sq, LaurentSeries):
        acc = LaurentSeries.constant(1)
    elif any(isinstance(value, acb) for value in (b_sq, lam_L_sq, lam_R_sq)):
        acc = acb(1)
    elif isinstance(b_sq, mp.mpc) or isinstance(lam_L_sq, mp.mpc) or isinstance(lam_R_sq, mp.mpc):
        acc = mp.mpc(1)
    else:
        acc = mp.mpf(1)

    quarter = acb(1) / 4 if isinstance(acc, acb) else mp.mpf(1) / 4
    sixteenth = acb(1) / 16 if isinstance(acc, acb) else mp.mpf(1) / 16

    # Iterate canonical (p, q) representatives; the (-p, -q) partner is
    # absorbed into each paired factor.
    p_values = list(range(1 - r, r, 2))
    q_values = list(range(1 - s, s, 2))
    seen = set()
    for p in p_values:
        for q in q_values:
            if (p, q) in seen:
                continue
            if p == 0 and q == 0:
                # Unpaired self-term (only when both r and s are odd):
                #   factor at (0, 0) = (lam_L + lam_R)(lam_L - lam_R)/4
                #                    = (lam_L^2 - lam_R^2)/4.
                acc = acc * ((lam_L_sq - lam_R_sq) * quarter)
                seen.add((0, 0))
            else:
                shift_sq = (p * p) * b_sq + 2 * p * q + (q * q) * inv_b_sq
                u = lam_L_sq + shift_sq - lam_R_sq
                v_sq = 4 * lam_L_sq * shift_sq
                acc = acc * ((u * u - v_sq) * sixteenth)
                seen.add((p, q))
                seen.add((-p, -q))
    return acc


def _real_part_or_none(x):
    """Return mp.mpf real part if x is mathematically real, else None.

    LaurentSeries inputs return None. mp.mpc inputs return real part if
    the imaginary part is exactly zero, or within a `(dps - 5)`-digit
    tolerance of zero, capped at 1e-5 (covering finite-precision artifacts from a real
    quantity that happened to be computed through mp.mpc).
    """
    if isinstance(x, LaurentSeries):
        return None
    if isinstance(x, mp.mpf):
        return x
    if isinstance(x, mp.mpc):
        if x.imag == 0:
            return mp.mpf(x.real)
        from .precision import _validation_tolerance

        tol = _validation_tolerance(guard=5)
        if abs(x.imag) <= tol * (abs(x.real) + 1):
            return mp.mpf(x.real)
        return None
    try:
        return mp.mpf(x)
    except (TypeError, ValueError):
        return None


def arithmetic_mode(c, h_externals) -> str:
    """Classify the recursion arithmetic based on the central charge and externals.

    Returns one of:

    - ``"real_b"`` -- c real with c > 25 and all externals real. b lies on
      the real axis; the Kac data K_{r,s}, d_{r,s}, v_{r,s} stay in
      ``mp.mpf`` end to end, including when lambda^2 < 0 for some external
      (heavy externals), thanks to the manifestly-real fusion polynomial.

    - ``"strip_twin"`` -- c real with 1 < c < 25 and all externals real. b
      lies on the unit circle (off the real axis); K's come in complex-
      conjugate twins under (r, s) <-> (s, r), and the H_q_series_rational
      assembly recombines them into real-coefficient numerators.

    - ``"imag_b"`` -- c real with c < 1 and all externals real. b is purely
      imaginary; using b^2 = -y^2 (mp.mpf, negative) and a specialized
      A_rs formula, K_{r,s}, d_{r,s}, v_{r,s} stay in ``mp.mpf`` end to end.

    - ``"generic_complex"`` -- anything else (complex c, complex externals,
      LaurentSeries-valued b, or the singular endpoints c in {1, 25}). The
      legacy mp.mpc recursion handles this fallback; the c in {1, 25}
      points are normally routed through the epsilon-extrapolation or
      Laurent regularization paths.
    """
    if isinstance(c, LaurentSeries):
        return "generic_complex"
    c_real = _real_part_or_none(c)
    if c_real is None:
        return "generic_complex"
    for hi in h_externals:
        if isinstance(hi, LaurentSeries):
            return "generic_complex"
        if _real_part_or_none(hi) is None:
            return "generic_complex"
    one = mp.mpf(1)
    twenty_five = mp.mpf(25)
    if c_real > twenty_five:
        return "real_b"
    if c_real < one:
        return "imag_b"
    if one < c_real < twenty_five:
        return "strip_twin"
    # c_real == 1 or c_real == 25: singular locus, route through the
    # generic path so the regularization wrappers can wrap the call.
    return "generic_complex"


def A_rs_imag_b(r: int, s: int, y):
    """A^c_{r,s} when b = i*y is purely imaginary (c < 1, externals real).

    Substituting b = i*y into A_rs gives a product of pure-imaginary factors
    1/(i*(m y - n/y)). There are 4 r s - 2 such factors, and (-i)^(4 r s - 2)
    = -1 for every r, s >= 1 with r*s >= 1 (since 4 r s mod 4 = 0 forces the
    exponent to land at 2 mod 4). The net result is

        A_rs(b = i y) = - (1/2) * prod_{(m,n)} 1 / (m y - n / y),

    where (m, n) runs over the same lattice as `A_rs` (m in [1-r, r],
    n in [1-s, s], excluding (0, 0) and (r, s)). All arithmetic stays in
    mp.mpf for real y, avoiding the mp.mpc detour of the generic path.
    """
    r, s = _kac_indices(r, s)
    if r < 1 or s < 1:
        raise ValueError("A_rs_imag_b requires r, s >= 1")
    inv_y = 1 / y
    half = mp.mpf(1) / 2
    acc = -half
    for m in range(1 - r, r + 1):
        for n in range(1 - s, s + 1):
            if (m, n) == (0, 0) or (m, n) == (r, s):
                continue
            acc /= m * y - n * inv_y
    return acc
