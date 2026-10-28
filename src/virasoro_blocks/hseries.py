"""Power series in (u - u_*) whose coefficients are polynomials in h.

The symbolic derivative tables use the u-Taylor expansion of the chiral
block F(h, z(u)) around the crossing-symmetric point u = u_*, retaining exact
dependence on the exchanged weight h. From `chiral.py`'s decomposition

    F(h, z) = E(z)**h * A(z) * N_z(h) / D(h),

the u-Taylor coefficient of F at order m is

    c_m(h) = E(z_*)**h * P_m(h) / D(h),

where P_m(h) is a polynomial in h: it collects the u^m term of the product
exp(h * s(u)) * A(z(u)) * N_{z(u)}(h), with s(u) = log E(z(u)) - log E(z_*).

`HSeries` is exactly that product object -- a truncated u-series whose
coefficients are `Polynomial`-in-h. The only non-trivial builder is
`exp_h_times`, which raises e to `h` times a numeric u-series with no constant
term; its u^m coefficient is a degree-m polynomial in h.
"""

from __future__ import annotations

from fractions import Fraction

import mpmath as mp

from .polynomial import Polynomial, _flint_prec_bits
from .numeric_input import _integer_control


def _detect_series_field(values) -> type:
    """mp.mpc if any value is complex with nonzero imag, else mp.mpf."""
    for v in values:
        if isinstance(v, (mp.mpc, complex)) and v.imag != 0:
            return mp.mpc
    return mp.mpf


class HSeries:
    """Truncated power series sum_m coeffs[m] * (u - u_*)^m.

    Each `coeffs[m]` is a `Polynomial` in the exchanged weight h.
    """

    __slots__ = ("coeffs",)

    def __init__(self, coeffs):
        self.coeffs = list(coeffs)
        if not self.coeffs:
            raise ValueError("HSeries needs at least one coefficient")

    @property
    def order(self) -> int:
        """Number of stored u-powers (degrees 0 .. order-1)."""
        return len(self.coeffs)

    @classmethod
    def from_numeric(cls, numeric_coeffs) -> "HSeries":
        """Promote a numeric u-series to constant (degree-0) h-polynomials.

        The Polynomial constructor field-detects on its coefficient list, so
        real `numeric_coeffs` produces mp.mpf-coefficient polynomials.
        """
        return cls([Polynomial([c]) for c in numeric_coeffs])

    def truncate(self, order: int) -> "HSeries":
        """Return the series cut (or zero-padded) to exactly `order` terms."""
        order = _integer_control('order', order, 1)
        if order <= len(self.coeffs):
            return HSeries(self.coeffs[:order])
        pad = [Polynomial.zero(field=mp.mpf)] * (order - len(self.coeffs))
        return HSeries(self.coeffs + pad)

    def __mul__(self, other: "HSeries") -> "HSeries":
        """Cauchy product; the result keeps all degrees of the full product.

        Runs in flint (:func:`_hseries_mul_flint`): arb polynomials for real
        coefficients, acb when either side is complex.
        """
        return HSeries(_hseries_mul_flint(self.coeffs, other.coeffs))


def exp_h_times(s_numeric, order: int) -> HSeries:
    """Build exp(h * s(u)) as an `HSeries` of length `order`.

    `s_numeric` is a numeric u-series [s_0, s_1, ...]; the constant term s_0 is
    ignored (the caller's s has no constant term). With G(u) = exp(h * s(u)),
    differentiating gives G' = h * s'(u) * G, i.e.

        m * G_m = h * sum_{j=1}^{m} j * s_j * G_{m-j},   G_0 = 1.

    Each G_m is therefore a polynomial in h of degree m.

    The polynomial field follows `s_numeric`: when every s_j is mp.mpf the
    Polynomial coefficients stay in mp.mpf, propagating the real-arithmetic
    path down to the chiral symbolic builder.
    """
    order = _integer_control('order', order, 1)
    # Field detection and construction both traverse the input; materialize
    # iterators once so detection cannot consume part or all of the series.
    s_numeric = list(s_numeric)
    field = _detect_series_field(s_numeric)
    if field is mp.mpf:
        s_numeric = [c.real if isinstance(c, (mp.mpc, complex)) else c for c in s_numeric]
    s = [
        c if isinstance(c, (mp.mpf, mp.mpc))
        else field(mp.convert(c) if isinstance(c, Fraction) else c)
        for c in s_numeric
    ]
    # The polynomial whose value is the variable h itself, in the inferred field.
    h_var = Polynomial([field(0), field(1)], field=field)
    G = [Polynomial.zero(field=field)] * order
    G[0] = Polynomial.one(field=field)
    for m in range(1, order):
        acc = Polynomial.zero(field=field)
        for j in range(1, min(m, len(s) - 1) + 1):
            acc = acc + G[m - j] * (field(j) * s[j])
        G[m] = (h_var * acc) * (field(1) / m)
    return HSeries(G)


# Above this many series terms on *both* operands, the Cauchy product is faster
# as a single Kronecker-packed multiply than as the O(n^2) loop of per-term flint
# polynomial multiplies (see _hseries_kronecker_mul). Benchmarks at dps=160:
# ~1.2x at n=40, ~2.0x at n=60, ~2.8x at n=80. Below the threshold the packing
# overhead is not worth it, so the plain loop is used.
_KRONECKER_MIN_TERMS = 40


def _hseries_kronecker_mul(a_flint, b_flint, poly_cls, zero):
    """Cauchy product of two flint-poly *series* via Kronecker substitution.

    ``a_flint``/``b_flint`` are the u-series coefficient lists (each entry a
    flint ``arb_poly``/``acb_poly`` in the shared variable h). The product
    ``C[t] = sum_{i+j=t} A[i] * B[j]`` is a convolution in u whose terms are
    polynomial multiplies in h. Packing ``(u-power m, h-power i)`` into a single
    exponent ``x^{m*Dh + i}`` -- with ``Dh`` larger than the maximum h-degree of
    any product term so adjacent u-blocks cannot collide -- turns the whole
    convolution into *one* big flint multiply (FFT), then we unpack the blocks.

    Returns ``len(a) + len(b) - 1`` flint polys (the full, untruncated product),
    matching the plain-loop path.
    """
    a_coeffs = [list(p.coeffs()) for p in a_flint]
    b_coeffs = [list(p.coeffs()) for p in b_flint]
    maxa = max((len(c) for c in a_coeffs), default=1)
    maxb = max((len(c) for c in b_coeffs), default=1)
    # h-degree of any product term is <= (maxa-1) + (maxb-1); Dh strictly larger
    # guarantees block t and block t+1 never overlap after packing.
    Dh = maxa + maxb - 1
    if Dh < 1:
        Dh = 1
    na, nb = len(a_coeffs), len(b_coeffs)

    pack_a = [zero] * (na * Dh)
    for m, coeffs in enumerate(a_coeffs):
        base = m * Dh
        for i, v in enumerate(coeffs):
            pack_a[base + i] = v
    pack_b = [zero] * (nb * Dh)
    for m, coeffs in enumerate(b_coeffs):
        base = m * Dh
        for i, v in enumerate(coeffs):
            pack_b[base + i] = v

    packed = poly_cls(pack_a) * poly_cls(pack_b)
    prod_coeffs = list(packed.coeffs())
    n_out = na + nb - 1
    out = []
    for t in range(n_out):
        seg = prod_coeffs[t * Dh:(t + 1) * Dh]
        out.append(poly_cls(seg) if seg else poly_cls([zero]))
    return out


def _hseries_mul_flint(a, b):
    """Multiply two HSeries coefficient lists in flint space."""
    # Use the Kronecker single-multiply path only when both series are long
    # and both actually carry polynomial (degree > 0) coefficients; for the
    # scalar fast-path shapes the plain loop is already cheap. ``.degree`` is
    # read off the flint cache here, so this never materializes a lazy poly.
    use_kronecker = (
        len(a) >= _KRONECKER_MIN_TERMS
        and len(b) >= _KRONECKER_MIN_TERMS
        and any(p.degree > 0 for p in a)
        and any(p.degree > 0 for p in b)
    )

    if any(p.field is mp.mpc for p in a) or any(p.field is mp.mpc for p in b):
        from flint import acb, acb_poly, arb, ctx as flint_ctx

        prec_bits = _flint_prec_bits()
        flint_ctx.prec = prec_bits
        if use_kronecker:
            zero = acb(arb(0))
            a_f = [acb_poly([zero]) if p.is_zero else p._get_acb() for p in a]
            b_f = [acb_poly([zero]) if p.is_zero else p._get_acb() for p in b]
            accum = _hseries_kronecker_mul(a_f, b_f, acb_poly, zero)
        else:
            accum = [acb_poly([acb(arb(0))]) for _ in range(len(a) + len(b) - 1)]
            for i, ai in enumerate(a):
                if ai.is_zero:
                    continue
                ai_f = ai._get_acb()
                for j, bj in enumerate(b):
                    if bj.is_zero:
                        continue
                    accum[i + j] += ai_f * bj._get_acb()
        # Lazy: carry the acb_poly cache so a chained multiply
        # (exp_hs * a_hseries * n_useries) reuses it via _get_acb instead
        # of rebuilding from a materialized mpc list. is_zero above keeps
        # the skip check from materializing the lazy intermediates.
        return [
            Polynomial._from_acb_trusted(poly, prec_bits)
            for poly in accum
        ]

    from flint import arb, arb_poly, ctx as flint_ctx

    prec_bits = _flint_prec_bits()
    flint_ctx.prec = prec_bits
    if use_kronecker:
        zero = arb(0)
        a_f = [arb_poly([zero]) if p.is_zero else p._get_arb() for p in a]
        b_f = [arb_poly([zero]) if p.is_zero else p._get_arb() for p in b]
        accum = _hseries_kronecker_mul(a_f, b_f, arb_poly, zero)
    else:
        accum = [arb_poly([arb(0)]) for _ in range(len(a) + len(b) - 1)]
        for i, ai in enumerate(a):
            if ai.is_zero:
                continue
            ai_f = ai._get_arb()
            for j, bj in enumerate(b):
                if bj.is_zero:
                    continue
                accum[i + j] += ai_f * bj._get_arb()
    # Lazy: keep the arb_poly cache (see the complex branch above).
    return [
        Polynomial._from_arb_trusted(poly, prec_bits)
        for poly in accum
    ]
