"""Polynomial-in-h representation of H(c, h, d_i, q).

The Zamolodchikov h-recursion (eq 2.20 of arXiv:1703.09805) gives

    H_k(h) = delta_{k,0} + sum_{(r,s): r s <= k}  K_{r,s} / (h - d_{r,s})  *  H_{k - rs}(v_{r,s})

with H_{k-rs}(v_{r,s}) a numerical value (the argument v_{r,s} = d_{r,s} + r s
is a number, not a function of h). Multiplying through by the common denominator

    D(h) = prod_{(r,s): r s <= q_order} (h - d_{r,s})

turns each H_k(h) into a polynomial N_k(h) of degree at most #pairs, divided
by D(h). The numerator polynomials are independent of h (their *coefficients*
are functions of c and d_i but the variable in which they are polynomials is h).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mpmath as mp

from .params import b_from_c
from .numeric_input import _integer_control
from .polynomial import (
    Polynomial,
    build_linear_product_flint,
    build_prefix_products_flint,
    build_product_flint,
    _polynomial_input, _flint_prec_bits,
    _mpf_to_arb, _mpf_to_acb, _arb_to_mpf, _acb_to_mpc,
)
from .recursion import _build_rs_data, _kac_input_values, _pair_interaction_weights


_COFACTOR_ROW_CACHE: dict[tuple, tuple[tuple, ...]] = {}
_COFACTOR_ROW_CACHE_MAX = 16
_HFILL_WMAT_CACHE: dict[tuple, object] = {}
_HFILL_WMAT_CACHE_MAX = 16


def _scalar_cache_key(x):
    """Return a hashable key for mpf/mpc values used in cache tags."""
    if isinstance(x, mp.mpc):
        return ("mpc", x.real._mpf_, x.imag._mpf_)
    if isinstance(x, mp.mpf):
        return ("mpf", x._mpf_)
    y = mp.mpf(x)
    return ("mpf", y._mpf_)


@dataclass
class HRational:
    """H(c, h, d_i, q) as a rational function in h.

    H(h, q) = sum_k N_k(h) q^k / D(h),  D(h) = prod_p (h - d_{r_p, s_p}).

    Attributes:
        numerators: list of Polynomial, one per power of q.
        denominator_roots: list of mpmath numbers, the d_{r,s} values.
        denominator: the polynomial D(h) = prod (h - d_{r,s}).
        pairs: the (r, s, rs) tuples that contribute, in r*s order.
    """

    numerators: list[Polynomial]
    denominator_roots: list
    denominator: Polynomial
    pairs: list
    root_provenance: list | None = None

    @property
    def q_order(self) -> int:
        return len(self.numerators) - 1

    def evaluate(self, h, q) -> mp.mpc:
        """Evaluate H(h, q) by summing the polynomial expansion."""
        from flint import ctx as flint_ctx

        h, q = _polynomial_input(h), _polynomial_input(q)
        if isinstance(h, mp.mpc) and h.imag == 0:
            h = h.real
        if isinstance(q, mp.mpc) and q.imag == 0:
            q = q.real
        field = (mp.mpc if isinstance(h, mp.mpc) or isinstance(q, mp.mpc)
                 or any(p.field is mp.mpc for p in (self.denominator, *self.numerators)) else mp.mpf)
        try:
            result = self._evaluate_ball(h, q, field, _flint_prec_bits())
            value = _arb_to_mpf(result) if field is mp.mpf else _acb_to_mpc(result)
            return +value
        finally:
            flint_ctx.prec = _flint_prec_bits()

    def _evaluate_ball(self, h, q, field, bits):
        """Certify the whole two-variable sum, including cross-level cancellation."""
        from flint import arb, acb, ctx as flint_ctx

        flint_ctx.prec = bits
        convert = _mpf_to_arb if field is mp.mpf else _mpf_to_acb
        h_ball, q_ball = convert(h), convert(q)
        get_poly = Polynomial._get_arb if field is mp.mpf else Polynomial._get_acb
        denominator = get_poly(self.denominator, bits)(h_ball)
        total = arb(0) if field is mp.mpf else acb(0)
        for poly in reversed(self.numerators):
            total = total*q_ball+get_poly(poly, bits)(h_ball)
        return total/denominator


def _maybe_real(x):
    """If x is an mpc with exact zero imaginary part, return its stored real part;
    else return x unchanged. The intent is to opt the scalar in to the
    real-arithmetic fast path when it is genuinely real (caller has
    propagated `mp.mpc` for backward compat but mathematically the value
    is real).
    """
    if isinstance(x, mp.mpc) and x.imag == 0:
        return x.real
    return x


def _pair_label(pair) -> tuple[int, int]:
    return (pair["r"], pair["s"])


def _surviving_outer_pair_indices(pairs) -> list[int]:
    """Indices whose common-denominator factor is structurally active.

    With non-colliding Kac roots, the common factor `(h - d_{r,s})` is needed
    iff the corresponding outer residue can appear in some `H_k`. Since
    `H_0(v_{r,s}) = 1`, that happens exactly when `K_{r,s} != 0`.

    This is a *necessary* gate only: it removes factors whose residue is
    structurally zero, but it misses factors that survive `K != 0` yet still
    divide every numerator because two surviving Kac roots collide (the
    denominator picks up a repeated factor the numerators lack). Those extra
    removable factors are stripped by `_strip_common_chiral_factors` below.
    """
    return [index for index, pair in enumerate(pairs) if pair["K"] != 0]


class _ChiralStripNotDivisible(Exception):
    """Internal signal: a common-factor removal failed the exact divisibility
    the multiplicity argument guarantees, indicating a degenerate c. The strip
    aborts and returns the table unchanged."""


def _common_factor_coincidence_tol() -> "mp.mpf":
    """Tolerance for deciding that two Kac roots *coincide*.

    Distinct Kac weights ``d_{r,s}`` are O(1)-separated, so a gap below half the
    working precision is unambiguously a genuine algebraic coincidence (a
    collision ``(r+r') b^2 + (s+s') = 0``), not two distinct-but-nearby roots.
    Unlike a magnitude test on a numerator value -- whose "zero" floor sits only
    ~`dps-15` deep (limited by the coefficient build precision) and so cannot be
    separated from the `dps-8` validation tolerance -- this is a *well-
    conditioned* comparison of two O(1) quantities, effectively exact.
    """
    return mp.mpf(10) ** (-(max(8, mp.mp.dps // 2)))


def _quotient_by_real_quadratic(poly: "Polynomial", two_re, abs2) -> "Polynomial":
    """Divide a real polynomial by the monic quadratic ``h^2 - two_re*h + abs2``.

    `two_re = 2 Re(d)` and `abs2 = |d|^2` reconstruct the real quadratic whose
    roots are the conjugate pair ``d, conj(d)``. Synthetic division keeps mpf
    coefficients throughout (no complex intermediate, so the real structure of
    a strip-symmetric denominator is preserved). The quotient is exact whenever
    the quadratic divides `poly` (guaranteed by the multiplicity argument in
    `_strip_common_chiral_factors`); any remainder is dropped.
    """
    c = poly.coeffs
    n = poly.degree
    if n < 2:
        raise ValueError("cannot divide a degree < 2 polynomial by a quadratic")
    p = -two_re  # quadratic is h^2 + p*h + q
    q = abs2
    b = [mp.mpf(0)] * (n - 1)  # quotient has degree n-2 -> n-1 coefficients
    b[n - 2] = c[n]
    if n - 3 >= 0:
        b[n - 3] = c[n - 1] - p * b[n - 2]
    for j in range(n - 4, -1, -1):
        b[j] = c[j + 2] - p * b[j + 1] - q * b[j + 2]
    return Polynomial(b, field=mp.mpf)


def _strip_common_chiral_factors(numerators, denominator, roots, provenance):
    """Remove ``(h - d)`` factors common to the denominator and *every* numerator.

    The ``K != 0`` gate keeps one denominator factor per surviving Kac pair, but
    when ``m`` surviving pairs *collide* (``d_{r,s} == d_{r',s'}``, which happens
    exactly at ``(r+r') b^2 + (s+s') = 0`` -- only for ``c < 1``, where
    ``b^2 < 0``) the denominator carries ``(h - d^*)^m`` while the numerators
    carry only ``(h - d^*)^{m-1}``. That excess factor is a removable chiral
    pole: it cancels in ``H(h, q)`` and merely inflates the polynomial degree.

    The removal is *analytic*, with no tolerance on the numerators. Each
    numerator is

        ``N_k(h) = sum_p coef_{k,p} * prod_{p' != p} (h - d_{p'})``,

    and for a collision cluster of size ``m`` at ``d^*`` *every* cofactor
    ``prod_{p' != p} (h - d_{p'})`` contains ``(h - d^*)^{m-1}`` (a cluster
    member ``p`` omits only its own factor, leaving ``m-1`` cluster factors;
    a non-member omits none, leaving all ``m``). Hence ``N_k`` is *exactly*
    divisible by ``(h - d^*)^{m-1}`` by construction -- we divide out precisely
    ``m-1`` factors per cluster, determined by the multiplicity alone.

    The only numerical decision is detecting the collision itself, i.e. whether
    two roots coincide. That is a well-conditioned comparison of O(1)-separated
    Kac weights (see `_common_factor_coincidence_tol`), not a magnitude test
    against an ill-defined numerator-zero floor. For a real (strip-symmetric or
    ``c < 1``) table a complex collision is removed together with its conjugate
    via a real quadratic quotient, keeping coefficients real; real roots use a
    linear quotient; a genuinely complex table removes the complex factor
    directly.
    """
    from .singularity import _UNSTRIPPED_COLLISIONS

    if _UNSTRIPPED_COLLISIONS.get():
        # a pole-order oracle wants every colliding root (see singularity.unstripped_collisions)
        return list(numerators), denominator, list(roots), list(provenance) if provenance is not None else None
    real_table = denominator.field is mp.mpf
    coincide_tol = _common_factor_coincidence_tol()

    n = len(roots)
    out_roots = list(roots)
    out_provs = list(provenance) if provenance is not None else None
    if n == 0:
        return list(numerators), denominator, out_roots, out_provs

    rc = [mp.mpc(r) for r in roots]

    def _coincide(a, b):
        scale = max(mp.mpf(1), abs(a), abs(b))
        return abs(a - b) <= coincide_tol * scale

    # Group root indices into clusters of coincident values (conjugates fall
    # into distinct clusters because their imaginary parts differ).
    #
    # This scan is O(n^2) in the number of denominator roots; at the inflated
    # symbolic working precision the per-comparison cost dominates, and for a
    # generic table (every root simple) it runs the full n^2 just to confirm
    # there are no collisions. Keep the exact algorithm and the exact rep
    # selection (rep = first root of each cluster), but make each comparison
    # cheap: precompute re/im/|.| once (n square roots, not n^2) and test the
    # squared distance |a-b|^2 <= (tol*scale)^2 -- algebraically identical to
    # the abs comparison, with no per-iteration sqrt or mpc object churn.
    res = [mp.re(z) for z in rc]
    ims = [mp.im(z) for z in rc]
    absv = [abs(z) for z in rc]
    one = mp.mpf(1)
    tol2 = coincide_tol * coincide_tol
    reps_idx: list[int] = []
    members: list[list[int]] = []
    for i in range(n):
        ri, ii, ai = res[i], ims[i], absv[i]
        placed = False
        for cid, ridx in enumerate(reps_idx):
            scale = ai if ai > absv[ridx] else absv[ridx]
            if scale < one:
                scale = one
            dre = ri - res[ridx]
            dim = ii - ims[ridx]
            if dre * dre + dim * dim <= tol2 * scale * scale:
                members[cid].append(i)
                placed = True
                break
        if not placed:
            reps_idx.append(i)
            members.append([i])
    reps: list = [rc[idx] for idx in reps_idx]

    if all(len(m) == 1 for m in members):
        return list(numerators), denominator, out_roots, out_provs

    drop: set[int] = set()
    lin_ops: list = []  # linear factors (h - root_val) to divide out
    quad_ops: list = []  # (two_re, abs2) real quadratics to divide out
    processed: set[int] = set()

    for cid, idxs in enumerate(members):
        if cid in processed:
            continue
        size = len(idxs)
        rep = reps[cid]
        is_complex_root = abs(mp.im(rep)) > coincide_tol * max(mp.mpf(1), abs(rep))

        if real_table and is_complex_root:
            # Remove the conjugate pair together to keep the table real.
            conj = mp.conj(rep)
            partner = None
            for cid2, rep2 in enumerate(reps):
                if cid2 != cid and _coincide(rep2, conj):
                    partner = cid2
                    break
            if partner is None:
                processed.add(cid)
                continue
            remove = min(size, len(members[partner])) - 1
            if remove > 0:
                two_re = 2 * mp.re(rep)
                abs2 = mp.re(rep) ** 2 + mp.im(rep) ** 2
                for _ in range(remove):
                    quad_ops.append((two_re, abs2))
                for j in idxs[:remove]:
                    drop.add(j)
                for j in members[partner][:remove]:
                    drop.add(j)
            processed.add(cid)
            processed.add(partner)
        else:
            remove = size - 1
            if remove > 0:
                root_val = mp.re(rep) if real_table else rep
                for _ in range(remove):
                    lin_ops.append(root_val)
                for j in idxs[:remove]:
                    drop.add(j)
            processed.add(cid)

    def _is_zero_poly(poly):
        return all(coef == 0 for coef in poly.coeffs)

    def _safe_divide(poly, op):
        """Divide `poly` by a linear `(h - root)` (op is the root value) or a
        real quadratic (op is a ``(two_re, abs2)`` tuple). A zero polynomial
        divides trivially. A nonzero polynomial of insufficient degree signals
        that the divisibility assumption failed at this (degenerate) c -- the
        caller aborts the strip rather than silently corrupting the table.
        """
        is_quad = isinstance(op, tuple)
        min_deg = 2 if is_quad else 1
        if poly.degree < min_deg:
            if _is_zero_poly(poly):
                return poly
            raise _ChiralStripNotDivisible()
        if is_quad:
            return _quotient_by_real_quadratic(poly, op[0], op[1])
        return poly.quotient_by_linear(op)

    ops = list(lin_ops) + list(quad_ops)
    try:
        nums = list(numerators)
        D = denominator
        for op in ops:
            nums = [_safe_divide(num, op) for num in nums]
            D = _safe_divide(D, op)
    except _ChiralStripNotDivisible:
        # Degenerate c (e.g. a v_{r,s} = d_{r',s'} resonance) breaks the exact
        # divisibility the multiplicity argument relies on. Leave the table
        # unstripped so the existing downstream degenerate-c handling applies.
        return list(numerators), denominator, out_roots, out_provs

    new_roots = [out_roots[i] for i in range(n) if i not in drop]
    new_provs = (
        [out_provs[i] for i in range(n) if i not in drop]
        if out_provs is not None
        else None
    )
    return nums, D, new_roots, new_provs


def _matmul_coeff_rows(c_rows, m_rows, *, complex_field, take_real, return_flint=False):
    """Compute the matrix product ``C @ M`` of coefficient rows in flint.

    `c_rows` is a list of ``n_k`` rows each of length ``n_p`` (the per-q
    residue coefficients); `m_rows` is a list of ``n_p`` rows each of length
    ``n_cols`` (the cofactor-polynomial coefficients, optionally scaled). The
    product row ``k`` holds the coefficients of the assembled numerator
    ``N_k``: ``out[k][i] = sum_p c_rows[k][p] * m_rows[p][i]``.

    `complex_field=True` runs the multiply in ``acb`` (used whenever any input
    is genuinely complex); otherwise it runs in real ``arb``. `take_real=True`
    returns ``Re(.)`` of each product entry for the strip-symmetric assembly.
    With ``return_flint=True`` those entries stay as native flint scalars;
    otherwise they are converted to mpmath values. Returns the coefficient rows,
    or ``None`` when either operand is empty (no matrix to multiply).
    """
    if not c_rows or not m_rows:
        return None
    n_cols = len(m_rows[0])
    if n_cols == 0:
        return [[] for _ in c_rows]
    from flint import arb, arb_mat, acb, acb_mat, ctx as flint_ctx
    from .polynomial import _flint_prec_bits, _mpf_to_acb, _arb_to_mpf

    flint_ctx.prec = _flint_prec_bits()
    if complex_field:
        c_is_acb = bool(c_rows and c_rows[0]) and isinstance(c_rows[0][0], acb)
        m_is_acb = bool(m_rows and m_rows[0]) and isinstance(m_rows[0][0], acb)
        cm = acb_mat([list(row) for row in c_rows]) if c_is_acb else acb_mat(
            [[_mpf_to_acb(x) for x in row] for row in c_rows]
        )
        mm = acb_mat([list(row) for row in m_rows]) if m_is_acb else acb_mat(
            [[_mpf_to_acb(x) for x in row] for row in m_rows]
        )
        res = cm * mm
        if return_flint:
            if take_real:
                return [
                    [res[i, j].real for j in range(n_cols)]
                    for i in range(len(c_rows))
                ]
            return [
                [res[i, j] for j in range(n_cols)]
                for i in range(len(c_rows))
            ]
        if take_real:
            return [
                [_arb_to_mpf(res[i, j].real) for j in range(n_cols)]
                for i in range(len(c_rows))
            ]
        return [
            [mp.mpc(mp.mpf(res[i, j].real), mp.mpf(res[i, j].imag))
             for j in range(n_cols)]
            for i in range(len(c_rows))
        ]
    c_is_arb = bool(c_rows and c_rows[0]) and isinstance(c_rows[0][0], arb)
    m_is_arb = bool(m_rows and m_rows[0]) and isinstance(m_rows[0][0], arb)
    cm = arb_mat([list(row) for row in c_rows]) if c_is_arb else arb_mat(
        [[arb(x) for x in row] for row in c_rows]
    )
    mm = arb_mat([list(row) for row in m_rows]) if m_is_arb else arb_mat(
        [[arb(x) for x in row] for row in m_rows]
    )
    res = cm * mm
    if return_flint:
        return [
            [res[i, j] for j in range(n_cols)]
            for i in range(len(c_rows))
        ]
    return [
        [mp.mpf(res[i, j]) for j in range(n_cols)]
        for i in range(len(c_rows))
    ]


def _fill_h_table_flint(
    *,
    pairs,
    canonical_indices,
    is_canonical,
    twin_of,
    interaction_weights,
    q_order,
    field,
    return_flint=False,
):
    """Vectorized step-1 H-table recursion via flint matrix-vector products.

    The Zamolodchikov recursion for a fixed power ``k`` is

        H_k(v_outer) = sum_inner W[outer][inner] * H_{k-rs_inner}(v_inner),

    where ``W[outer][inner] = K_inner / (v_outer - d_inner)`` is *independent of
    ``k``*. Stacking the canonical outer pairs turns each ``k``-step into a
    single matrix-vector product ``acc = W @ v_k`` with the column
    ``v_k[inner] = lookup_h(inner, k - rs_inner)`` (the non-canonical entries
    are conjugates of their canonical twin). The ``k``-loop stays sequential
    (``v_k`` reads columns at smaller ``k``), but each step is now one compiled
    flint matvec instead of ``O(#canonical * #pairs)`` mpmath scalar mul/add/eq.

    Returns the completed H-table, or ``None`` to signal the caller to use the
    scalar fallback -- when any interaction weight is ``None`` (a
    ``v_outer == d_inner`` coincidence the matrix form cannot represent; arises
    e.g. for c < 1). With ``return_flint=True``, returns the flint-native rows
    directly.
    """
    from flint import arb, acb, arb_mat, acb_mat, ctx as flint_ctx
    from .polynomial import _flint_prec_bits, _mpf_to_acb, _mpf_to_arb

    n = len(pairs)
    rs = [p["rs"] for p in pairs]
    complex_field = field is mp.mpc

    # Dense weight matrix W (#canonical x #pairs); bail on any None weight.
    weight_rows = []
    for idx in canonical_indices:
        row = interaction_weights[idx]
        if any(w is None for w in row):
            return None
        weight_rows.append(row)

    flint_ctx.prec = _flint_prec_bits()
    # pairs/canonical_indices come from cached recursion metadata and are
    # immutable in this path, so an identity key avoids rebuilding a deep
    # scalar tuple on every call.
    wmat_key = (
        "hfill_wmat",
        "acb" if complex_field else "arb",
        flint_ctx.prec,
        id(pairs),
        id(canonical_indices),
    )
    Wmat = _HFILL_WMAT_CACHE.get(wmat_key)
    if Wmat is None:
        if complex_field:
            Wmat = acb_mat([[_mpf_to_acb(w) for w in row] for row in weight_rows])
        else:
            Wmat = arb_mat([[_mpf_to_arb(w) for w in row] for row in weight_rows])
        _HFILL_WMAT_CACHE[wmat_key] = Wmat
        if len(_HFILL_WMAT_CACHE) > _HFILL_WMAT_CACHE_MAX:
            _HFILL_WMAT_CACHE.pop(next(iter(_HFILL_WMAT_CACHE)))

    if complex_field:
        one, zero = acb(1), acb(0)
    else:
        one, zero = arb(1), arb(0)

    # Per-pair flint H-rows for canonical pairs; non-canonical stay None.
    Hf: list = [None] * n
    max_k = [0] * n
    for idx in canonical_indices:
        mk = q_order - pairs[idx]["rs"]
        max_k[idx] = mk
        rowf = [zero] * (mk + 1)
        rowf[0] = one
        Hf[idx] = rowf

    for k in range(1, q_order + 1):
        col = []
        for j in range(n):
            if rs[j] > k:
                col.append([zero])
                continue
            kk = k - rs[j]
            if is_canonical[j]:
                col.append([Hf[j][kk]])
            else:
                col.append([Hf[twin_of[j]][kk].conjugate()])
        vmat = acb_mat(col) if complex_field else arb_mat(col)
        acc = Wmat * vmat
        for i, idx in enumerate(canonical_indices):
            if k <= max_k[idx]:
                Hf[idx][k] = acc[i, 0]

    if return_flint:
        return Hf

    H_table: list = [None] * n
    if complex_field:
        for idx in canonical_indices:
            H_table[idx] = [
                mp.mpc(mp.mpf(e.real), mp.mpf(e.imag)) for e in Hf[idx]
            ]
    else:
        for idx in canonical_indices:
            H_table[idx] = [mp.mpf(e) for e in Hf[idx]]
    return H_table


def chiral_denominator_roots(h_ext, *, b=None, c=None, q_order: int):
    """The genuine chiral Kac-pole locations ``d_{r,s}`` (``r s <= q_order``) --
    identical to ``H_q_series_rational(...).denominator_roots`` -- WITHOUT building
    the numerator H-table.

    At generic (non-singular) c these are exactly the surviving Kac weights: the
    common-factor strip in :func:`H_q_series_rational` fires only on Kac-weight
    *collisions* (coincident ``d_{r,s}``), which occur solely at recognition-
    singular c. Callers MUST restrict this fast path to non-singular c -- the
    singular path needs the full numerator-carrying table to strip removable
    poles. Costs O(n_pairs) (Kac-data enumeration) vs the full build's
    O(numerator H-table fill): ~33x faster at q_order=60 and, crucially, far
    better q_order scaling (the numerator fill is the q_order blow-up in the
    large-spin tower's pole-pool build). Verified to match the full
    ``denominator_roots`` to ~1e-125 at generic c.
    """
    q_order = _integer_control('q_order', q_order)
    if (b is None) == (c is None):
        raise ValueError("specify exactly one of b, c")
    if b is None:
        b = b_from_c(c)
    if len(h_ext) != 4:
        raise ValueError("h_ext must be a length-4 sequence (h1, h2, h3, h4)")
    # Mirror H_q_series_rational's Kac-data typing exactly so the pair table (and
    # hence the surviving roots) is bit-identical to the full path.
    b = _maybe_real(b)
    h_ext_typed = _kac_input_values(h_ext)
    pair_table = _build_rs_data(q_order, b, h_ext_typed)
    surviving_indices = _surviving_outer_pair_indices(pair_table.pairs)
    return [pair_table.pairs[idx]["d"] for idx in surviving_indices]


def H_q_series_rational(
    h_ext: Sequence,
    *,
    b=None,
    c=None,
    q_order: int,
) -> HRational:
    """Compute H(c, h, h_i, q) as a rational function in h.

    The denominator is fixed: D(h) = prod_{(r,s): r s <= q_order} (h - d_{r,s}).
    Each q^k coefficient is N_k(h) / D(h) with N_k a polynomial in h.

    Storage type dispatch: when b is real (Im b == 0) and all h_i are real,
    the Kac data and the resulting polynomials are stored as mpf, which gives
    ~3x speedup at high working precision over the mpc default. Mixed cases
    (e.g. real b but lam_sq < 0 making fusion_polynomial mpc) auto-promote
    via the field-aware Polynomial type. The default complex path is used
    whenever any input is genuinely complex.
    """
    q_order = _integer_control('q_order', q_order)
    if (b is None) == (c is None):
        raise ValueError("specify exactly one of b, c")
    if b is None:
        b = b_from_c(c)
    if len(h_ext) != 4:
        raise ValueError("h_ext must be a length-4 sequence (h1, h2, h3, h4)")

    # Promote-down to mp.mpf where the scalar is mathematically real, so the
    # Kac functions (which propagate the field) produce mpf K, d, v.
    b = _maybe_real(b)
    h_ext_typed = _kac_input_values(h_ext)

    pair_table = _build_rs_data(q_order, b, h_ext_typed)
    pairs = pair_table.pairs
    is_canonical = pair_table.is_canonical
    twin_of = pair_table.twin_of
    canonical_indices = pair_table.canonical_indices
    interaction_weights = _pair_interaction_weights(pair_table)
    surviving_indices = _surviving_outer_pair_indices(pairs)
    surviving_mask = [False] * len(pairs)
    for index in surviving_indices:
        surviving_mask[index] = True

    # Decide field for H_table accumulators / polynomial coefficients. If
    # every K and every d in `pairs` is mpf, run real arithmetic; otherwise
    # mpc. This is decided per-call, after fusion_polynomial has had a chance
    # to promote to mpc (e.g. when lambda^2 < 0 for some external d_i).
    all_real = all(
        isinstance(p["K"], mp.mpf) and isinstance(p["d"], mp.mpf)
        for p in pairs
    )
    field = mp.mpf if all_real else mp.mpc
    strip = pair_table.is_strip_symmetric

    # 1) Build H_k(v_{r',s'}) numerically. In strip-symmetric mode we only fill
    # rows for canonical pairs (r <= s); non-canonical (s, r) rows are read off
    # as conjugates of the canonical twin, which preserves the b <-> 1/b
    # symmetry by construction (no roundoff-driven drift between twin rows).
    H_table: list = [None] * len(pairs)
    H_table_flint = None
    for idx in canonical_indices:
        max_k_needed = q_order - pairs[idx]["rs"]
        row = [field(0)] * (max_k_needed + 1)
        row[0] = field(1)
        H_table[idx] = row

    def lookup_h(pair_idx: int, k: int):
        if H_table_flint is not None:
            if is_canonical[pair_idx]:
                return H_table_flint[pair_idx][k]
            return H_table_flint[twin_of[pair_idx]][k].conjugate()
        if is_canonical[pair_idx]:
            return H_table[pair_idx][k]
        return mp.conj(H_table[twin_of[pair_idx]][k])

    filled = _fill_h_table_flint(
        pairs=pairs,
        canonical_indices=canonical_indices,
        is_canonical=is_canonical,
        twin_of=twin_of,
        interaction_weights=interaction_weights,
        q_order=q_order,
        field=field,
        return_flint=(strip or field is mp.mpc),
    )
    if filled is not None:
        if strip or field is mp.mpc:
            H_table_flint = filled
        else:
            H_table = filled
    else:
        for k in range(1, q_order + 1):
            for idx_outer in canonical_indices:
                p_outer = pairs[idx_outer]
                if k > q_order - p_outer["rs"]:
                    continue
                weights = interaction_weights[idx_outer]
                acc = field(0)
                for idx_inner, p_inner in enumerate(pairs):
                    if p_inner["rs"] > k:
                        break
                    weight = weights[idx_inner]
                    inner = lookup_h(idx_inner, k - p_inner["rs"])
                    if inner == 0:
                        continue
                    if weight is None:
                        acc += p_inner["K"] / (p_outer["v"] - p_inner["d"]) * inner
                        continue
                    if weight == 0:
                        continue
                    acc += weight * inner
                H_table[idx_outer][k] = acc

    # 2) Build D(h) and the partial products D(h) / (h - d_{r,s}) for each pair.
    #
    # In strip-symmetric mode the d_{r,s} come in conjugate pairs (r<s ↔ s<r)
    # plus real diagonals (r==s).  We build D(h) with mpf coefficients by
    # multiplying conjugate-pair quadratics (h² - 2Re(d)h + |d|²) and real
    # linear factors (h - d_{r,r}).  The cofactors (complex for off-diagonal
    # roots) are still needed for the twin-paired assembly below, but only for
    # canonical indices.
    if strip:
        surviving_canonical_indices = []
        for idx in canonical_indices:
            twin_idx = twin_of[idx]
            if surviving_mask[idx] != surviving_mask[twin_idx]:
                raise ValueError(
                    "strip-twin recursion expected twin pairs to share residue support"
                )
            if surviving_mask[idx]:
                surviving_canonical_indices.append(idx)

        # Build canonical factors: linear (h - d) for diagonal pairs, quadratic
        # (h² - 2Re(d)h + |d|²) for conjugate pairs.
        canonical_factors = []
        for idx in surviving_canonical_indices:
            p = pairs[idx]
            d_val = p["d"]
            if p["r"] == p["s"]:
                # Real root.
                factor = Polynomial.x_minus(mp.re(d_val), field=mp.mpf)
            else:
                # Conjugate pair: (h - d)(h - d*) = h^2 - 2Re(d)h + |d|^2
                re_d = mp.re(d_val)
                abs2_d = mp.re(d_val) ** 2 + mp.im(d_val) ** 2
                factor = Polynomial([abs2_d, -2 * re_d, mp.mpf(1)], field=mp.mpf)
            canonical_factors.append(factor)

        # Build D(h) as product of all canonical factors. Use flint-native
        # multiplication to reduce conversion overhead.
        D_poly = build_product_flint(canonical_factors, mp.mpf)
        if D_poly is None:
            # Flint not available, use Python fallback
            D_poly = Polynomial.one(field=mp.mpf)
            for factor in canonical_factors:
                D_poly = D_poly * factor

        # Cofactors for canonical pairs only. Build them from the same
        # conjugate-pair factorization used for D(h) rather than by synthetic
        # division of the real D(h) by (h - d): at finite precision D(h) is
        # only approximately divisible by the complex linear factor, and the
        # silent dropped remainder in quotient_by_linear would leak into the
        # strip path as a real-assembly error.
        # Use flint-native prefix products where possible.
        prefix = build_prefix_products_flint(canonical_factors, mp.mpf)
        if prefix is None:
            # Flint not available, use Python fallback
            prefix = [Polynomial.one(field=mp.mpf)]
            for factor in canonical_factors:
                prefix.append(prefix[-1] * factor)

        # Build suffix products in reverse order
        reversed_factors = list(reversed(canonical_factors))
        suffix_reversed = build_prefix_products_flint(reversed_factors, mp.mpf)
        if suffix_reversed is None:
            # Flint not available, use Python fallback
            suffix = [None] * (len(canonical_factors) + 1)
            suffix[-1] = Polynomial.one(field=mp.mpf)
            for pos in range(len(canonical_factors) - 1, -1, -1):
                suffix[pos] = canonical_factors[pos] * suffix[pos + 1]
        else:
            # Convert reversed suffix to forward-indexed suffix
            suffix = list(reversed(suffix_reversed))

        cofactors_canonical = {}
        for pos, idx in enumerate(surviving_canonical_indices):
            p = pairs[idx]
            others = prefix[pos] * suffix[pos + 1]
            if p["r"] == p["s"]:
                cofactors_canonical[idx] = others
            else:
                # D / (h - d) = (h - d*) * product(other canonical factors).
                cofactors_canonical[idx] = others * Polynomial.x_minus(
                    mp.conj(p["d"]), field=mp.mpc
                )
    else:
        # Non-strip mode: build D(h) and cofactors via flint prefix/suffix products.
        # cofactor[i] = prefix[i] * suffix[i+1] avoids n mpmath synthetic divisions.
        roots = [pairs[idx]["d"] for idx in surviving_indices]
        linear_factors = [Polynomial.x_minus(d, field=field) for d in roots]
        prefix = build_prefix_products_flint(linear_factors, field)
        if prefix is not None:
            reversed_factors = list(reversed(linear_factors))
            suffix_reversed = build_prefix_products_flint(reversed_factors, field)
            if suffix_reversed is not None:
                suffix = list(reversed(suffix_reversed))
                D_poly = prefix[-1]
                cofactors = [prefix[i] * suffix[i + 1] for i in range(len(linear_factors))]
            else:
                D_poly = prefix[-1]
                cofactors = [D_poly.quotient_by_linear(d) for d in roots]
        else:
            # Flint not available, use Python fallback
            D_poly = Polynomial.one(field=field)
            for f in linear_factors:
                D_poly = D_poly * f
            cofactors = [D_poly.quotient_by_linear(d) for d in roots]

    # 3) Build N_k(h) for k = 0..q_order.
    #
    # In strip mode, pair (r,s) and (s,r) contributions into a single real
    # polynomial term: for r < s,
    #   K * H * cofactor + conj(K * H * cofactor) = 2 Re(K * H * cofactor)
    # giving exactly-real numerator coefficients with no roundoff-driven Im.
    #
    # Each N_k is a fixed linear combination of the (k-independent) cofactor
    # polynomials: N_k = sum_p coef[k][p] * cofactor_p. Stacking over k turns
    # the whole assembly into one matrix product C @ M, where C[k][p] is the
    # per-q residue coefficient and M[p][i] the cofactor coefficients. The
    # flint matrix multiply replaces the O(q * pairs * degree) scalar-mpf/mpc
    # Python loop that dominated H_q_series_rational; the per-k loop below
    # serves only the empty-product cases (q_order == 0 or no active poles).
    out_field = mp.mpf if strip else field

    if strip:
        cof_polys = [cofactors_canonical[idx] for idx in surviving_canonical_indices]
        active_indices = surviving_canonical_indices
        mults = [
            mp.mpf(1) if pairs[idx]["r"] == pairs[idx]["s"] else mp.mpf(2)
            for idx in surviving_canonical_indices
        ]
        complex_field = field is mp.mpc or any(
            pairs[idx]["r"] != pairs[idx]["s"] for idx in surviving_canonical_indices
        )
        take_real = True
    else:
        cof_polys = list(cofactors)
        active_indices = surviving_indices
        mults = [mp.mpf(1)] * len(surviving_indices)
        complex_field = field is mp.mpc
        take_real = False

    pair_k_flint = None
    if H_table_flint is not None and complex_field:
        from .polynomial import _mpf_to_acb

        pair_k_flint = [_mpf_to_acb(p["K"]) for p in pairs]

    numerators = _assemble_numerators(
        q_order=q_order,
        D_poly=D_poly,
        pairs=pairs,
        active_indices=active_indices,
        cof_polys=cof_polys,
        mults=mults,
        lookup_h=lookup_h,
        out_field=out_field,
        complex_field=complex_field,
        take_real=take_real,
        pair_k_flint=pair_k_flint,
    )
    if numerators is None:
        numerators = _assemble_numerators_loop(
            q_order=q_order,
            D_poly=D_poly,
            pairs=pairs,
            strip=strip,
            field=field,
            surviving_canonical_indices=(
                surviving_canonical_indices if strip else None
            ),
            cofactors_canonical=cofactors_canonical if strip else None,
            cofactors=None if strip else cofactors,
            surviving_indices=None if strip else surviving_indices,
            lookup_h=lookup_h,
        )

    # Strip removable common factors that the K != 0 gate misses: when m
    # surviving Kac roots collide (degenerate / minimal-model c) the denominator
    # carries (h - d*)^m while the numerators carry only (h - d*)^{m-1}. Those
    # excess factors cancel in H(h, q) and inflate the polynomial degree, so
    # divide them out here. The amount removed is fixed by the collision
    # multiplicity (an analytic, tolerance-free divisibility), and the strip is
    # a no-op for generic c, where every surviving root is simple.
    surv_roots = [pairs[idx]["d"] for idx in surviving_indices]
    surv_provs = [(_pair_label(pairs[idx]),) for idx in surviving_indices]
    numerators, D_poly, surv_roots, surv_provs = _strip_common_chiral_factors(
        numerators, D_poly, surv_roots, surv_provs
    )

    return HRational(
        numerators=numerators,
        denominator_roots=surv_roots,
        denominator=D_poly,
        pairs=pairs,
        root_provenance=surv_provs,
    )
def _assemble_numerators(
    *,
    q_order,
    D_poly,
    pairs,
    active_indices,
    cof_polys,
    mults,
    lookup_h,
    out_field,
    complex_field,
    take_real,
    pair_k_flint,
):
    """Assemble N_0..N_q via a single flint matrix product.

    Returns the numerator Polynomials, or ``None`` when there is no product to
    form (``q_order == 0`` or no active poles; the caller then uses the scalar loop).
    """
    return_flint_rows = pair_k_flint is not None
    n_cols = max((len(p.coeffs) for p in cof_polys), default=0)
    # M[p][i] = mult_p * cofactor_p[i] (padded to n_cols).
    m_rows = []
    if return_flint_rows and complex_field:
        from flint import acb
        from .polynomial import _flint_prec_bits

        prec_bits = _flint_prec_bits()
        cache_key = (
            "acb_m_rows",
            prec_bits,
            n_cols,
            tuple(_scalar_cache_key(mult) for mult in mults),
            tuple(_scalar_cache_key(pairs[idx]["d"]) for idx in active_indices),
        )
        cached_rows = _COFACTOR_ROW_CACHE.get(cache_key)
        if cached_rows is not None:
            m_rows = [list(row) for row in cached_rows]
        else:
            for mult, poly in zip(mults, cof_polys):
                coeffs_f = list(poly._get_acb().coeffs())
                if mult == 1:
                    row = list(coeffs_f)
                else:
                    mult_f = acb(mult)
                    row = [mult_f * c for c in coeffs_f]
                if len(row) < n_cols:
                    row = row + [acb(0)] * (n_cols - len(row))
                m_rows.append(row)

            _COFACTOR_ROW_CACHE[cache_key] = tuple(tuple(row) for row in m_rows)
            if len(_COFACTOR_ROW_CACHE) > _COFACTOR_ROW_CACHE_MAX:
                _COFACTOR_ROW_CACHE.pop(next(iter(_COFACTOR_ROW_CACHE)))
    else:
        for mult, poly in zip(mults, cof_polys):
            coeffs = poly.coeffs
            if mult == 1:
                row = list(coeffs)
            else:
                row = [mult * c for c in coeffs]
            if len(row) < n_cols:
                zero = mp.mpf(0) if not complex_field else mp.mpc(0)
                row = row + [zero] * (n_cols - len(row))
            m_rows.append(row)

    # C[k-1][p] = K_p * H_p(k - rs_p) for rs_p <= k else 0, rows for k=1..q.
    zero = mp.mpf(0)
    c_rows = []
    for k in range(1, q_order + 1):
        crow = []
        for col, idx in enumerate(active_indices):
            p = pairs[idx]
            if p["rs"] <= k:
                coef = lookup_h(idx, k - p["rs"])
                if pair_k_flint is None:
                    crow.append(p["K"] * coef)
                else:
                    crow.append(pair_k_flint[idx] * coef)
            else:
                crow.append(zero)
        c_rows.append(crow)

    product = _matmul_coeff_rows(
        c_rows,
        m_rows,
        complex_field=complex_field,
        take_real=take_real,
        return_flint=return_flint_rows,
    )
    if product is None:
        return None

    numerators = [D_poly]  # N_0 = D(h) since H_0 = 1.
    if return_flint_rows and complex_field:
        from flint import arb_poly, acb_poly
        from .polynomial import _flint_prec_bits

        prec_bits = _flint_prec_bits()
        # Lazy: keep the flint poly as the cache so the per-spin compose_linear
        # (which calls _get_arb/_get_acb) reuses it instead of rebuilding from
        # the mpf/mpc list. Materialization is deferred to first .coeffs access.
        for row in product:
            if out_field is mp.mpf:
                numerators.append(
                    Polynomial._from_arb_trusted(arb_poly(list(row)), prec_bits)
                )
            else:
                numerators.append(
                    Polynomial._from_acb_trusted(acb_poly(list(row)), prec_bits)
                )
    else:
        for row in product:
            numerators.append(Polynomial(row, field=out_field))
    return numerators


def _assemble_numerators_loop(
    *,
    q_order,
    D_poly,
    pairs,
    strip,
    field,
    surviving_canonical_indices,
    cofactors_canonical,
    cofactors,
    surviving_indices,
    lookup_h,
):
    """Scalar numerator assembly, for the empty-product cases the matmul skips."""
    numerators: list[Polynomial] = []
    for k in range(q_order + 1):
        if k == 0:
            N_k = D_poly  # H_0(h) = 1, so N_0 = D(h).
            numerators.append(N_k)
            continue

        if strip:
            # Accumulate directly into coefficient list to avoid creating
            # thousands of intermediate Polynomial objects.
            n_coeffs = D_poly.degree + 1
            nk_coeffs = [mp.mpf(0)] * n_coeffs
            for idx in surviving_canonical_indices:
                p = pairs[idx]
                if p["rs"] > k:
                    continue
                coef = p["K"] * lookup_h(idx, k - p["rs"])
                if coef == 0:
                    continue
                cof_coeffs = cofactors_canonical[idx].coeffs
                if p["r"] == p["s"]:
                    # Diagonal: accumulate Re(cofactor_coeff * coef)
                    for i, c in enumerate(cof_coeffs):
                        nk_coeffs[i] += mp.re(c * coef)
                else:
                    # Off-diagonal canonical: accumulate 2*Re(cofactor_coeff * coef)
                    two_coef = 2 * coef
                    for i, c in enumerate(cof_coeffs):
                        nk_coeffs[i] += mp.re(c * two_coef)
            numerators.append(Polynomial(nk_coeffs, field=mp.mpf))
        else:
            # Accumulate directly into coefficient list.
            n_coeffs = D_poly.degree + 1
            nk_coeffs = [field(0)] * n_coeffs
            for cofactor, idx in zip(cofactors, surviving_indices):
                p = pairs[idx]
                if p["rs"] > k:
                    break
                coef = p["K"] * lookup_h(idx, k - p["rs"])
                if coef == 0:
                    continue
                for i, c in enumerate(cofactor.coeffs):
                    nk_coeffs[i] += c * coef
            numerators.append(Polynomial(nk_coeffs, field=field))

    return numerators
