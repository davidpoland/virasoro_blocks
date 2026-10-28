"""N=1 Neveu-Schwarz superconformal blocks via the elliptic recursion.

The formulas follow

  [HJS-a] Hadasz, Jaskolski, Suchanek, "Recursion representation of the
          Neveu-Schwarz superconformal block", hep-th/0611266, JHEP 03 (2007) 032
  [HJS-b] Hadasz, Jaskolski, Suchanek, "Elliptic recurrence representation of
          the N=1 Neveu-Schwarz blocks", arXiv:0711.1619, Nucl.Phys. B798 (2008)

with equation numbers recorded beside the implementation. The residues are
checked against the closed forms of [HJS-b] eq (6.2); see
``tests/core/test_recursion_ns.py``.

Conventions: the beta/alpha parametrization
-------------------------------------------
The two papers use different Liouville parameters, related by ``beta = i b``
(so [HJS-b]'s ``c = 3/2 + 3(b + 1/b)^2`` and [HJS-a]'s
``chat = 1 - 2(beta - 1/beta)^2`` agree). This module uses [HJS-a]'s ``beta``
throughout, because its fusion polynomials come with explicit index conditions
and no phase ambiguity. **Do not mix in the b convention** -- that is precisely
the kind of trap that has cost this project time before.

    chat = 2 c / 3                                    (reduced central charge)
    chat = 1 - 2 (beta - 1/beta)^2
    beta = (sqrt(1 - chat) + sqrt(9 - chat)) / (2 sqrt 2)          [HJS-a] (3.6)
    Delta = (chat - 1)/16 + alpha^2 / 8                           [HJS-a] (3.7)

The exchanged/external weights enter only through ``alpha``, and only through
``alpha_i^2`` -- the fusion polynomials are invariant under ``alpha_i -> -alpha_i``
because their index sets are closed under ``(p, q) -> (-p, -q)``. So the square
root's branch is irrelevant and no sign convention has to be fixed.

The pole set
------------
Poles sit at the NS degenerate weights, with ``r + s`` EVEN and ``r s >= 1``
(the pair (1, 1), Delta_11 = 0, is included; a pole is dropped only when
its residue vanishes in the selected sector):

    Delta_rs = (chat - 1)/16 + alpha_rs^2 / 8,  alpha_rs = r beta - s / beta
                                                                  [HJS-a] (3.5)

The null vector is at level ``r s / 2``, so the intermediate weight is
``v_rs = Delta_rs + r s / 2``. Because ``r + s`` is even, ``r`` and ``s`` are
either both even (then ``r s`` is a multiple of 4) or both odd (then ``r s`` is
odd, including ``rs = 1``). The odd block can therefore start at ``q^(1/2)``;
the ``(1,1)`` pole is required, for example, by the closed form
``H^(1/2) = theta_2(q^2)/Delta`` below. There is no Kac pole at level 1.

Two coupled sectors
-------------------
[HJS-b] eq (6.5) is not one recursion but a coupled pair, and this is the one
genuinely new structural feature. Writing ``H^0`` for the even block ``H^1`` and
``H^1`` for the odd block ``H^(1/2)`` of the papers:

    H^t = g^t + sum_{rs even} (16q)^(rs/2) K_rs/(Delta - Delta_rs) H^t(v_rs)
              + sum_{rs odd}  (16q)^(rs/2) K_rs/(Delta - Delta_rs) H^(1-t)(v_rs)

i.e. an odd-level pole FLIPS the sector and an even-level pole preserves it.
Equivalently ``sector_out = sector_in XOR (rs mod 2)``, which is what
:meth:`NSFamily.couples` implements.

The residue numerator ``K_rs`` DOES depend on the sector, through which of the
two fusion polynomials it uses -- see :meth:`NSFamily.build_pole`. That is the
one place the papers' compressed notation is ambiguous, and it is settled here
against the closed forms rather than by reading.

The ``(1, 1)`` pole is included, at level 1/2, with ``Delta_11 = 0`` identically
in ``c``. Excluding it (as the ``1 < rs`` in [HJS-a] (2.3) invites) deletes the
whole leading term of the odd sector.

The seeds are the ``Delta -> infinity`` regular parts, [HJS-b] eq (6.4):

    g^1 = theta_3(q^2),      g^(1/2) = 0

for unstarred external weights (the case of physical interest: a correlator of
four NS super-primaries). ``S_rs(Delta_3) = 1`` there too.

Expansion variable
------------------
Levels are half-integers, so the elliptic block is a series in ``q^(1/2)``, and
this module expands in ``t = q^(1/2)`` with an INTEGER level index ``rs``. That
is why the generic kernel in :mod:`virasoro_blocks.poles` needed no notion of
half-steps: ``level_unit = 1/2`` is carried by the family and consumed only when
the coefficient list is evaluated at a numeric ``q``.

``ns_order`` counts powers of ``q^(1/2)`` and reaches ``q^(ns_order/2)``;
``q_order`` counts powers of ``q``. Entry points require exactly one of these
arguments to make the truncation unit explicit. See :func:`_resolve_ns_order`.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from typing import Sequence

import mpmath as mp

from . import poles as _poles
from .recursion import _maybe_real_scalar, _resolve_chiral_norm, _theta3, q_from_z
from .numeric_input import parse_number
from .super_kac import _super_cutoff, exact_affine_number, exact_affine_ball, guard_super_inputs, ns_fusion_zero

#: The even (``H^1``) and odd (``H^(1/2)``) NS block sectors of [HJS-b].
SECTOR_EVEN = 0
SECTOR_ODD = 1


def ns_order_for_q_order(q_order: int) -> int:
    """Half-step order reaching ``q^q_order``. Always use this to convert.

    ``ns_order = 2 * q_order``. Kept as a named function rather than an inline
    factor of two so that call sites read as a unit conversion instead of a
    magic number.
    """
    return 2 * _super_cutoff("q_order", q_order)


def _resolve_ns_order(ns_order, q_order) -> int:
    """Exactly one of ``ns_order`` (half-steps) or ``q_order`` (powers of q).

    The NS series runs in ``q^(1/2)``, so the two differ by a factor of two and a
    caller who passes a Virasoro-tuned budget under the wrong name silently gets
    HALF the intended reach. Rather than warn about that, the API refuses to
    accept an unlabelled order at all: every call site has to state its unit.
    This is the cheap structural fix for the failure mode recorded in the
    ``q_order`` starvation note, where a silent reach loss corrupted blocks
    without any visible symptom.
    """
    if (ns_order is None) == (q_order is None):
        raise ValueError(
            "specify exactly one of ns_order (powers of q^(1/2)) or q_order "
            "(powers of q); they differ by a factor of two"
        )
    return _super_cutoff("ns_order", ns_order) if ns_order is not None else ns_order_for_q_order(q_order)


def chat_from_c(c):
    """``chat = 2c/3``, the reduced central charge of [HJS-a]."""
    return 2 * c / 3


def beta_from_c(c):
    """``beta`` from ``c``, via [HJS-a] (3.6).

    ``beta = (sqrt(1 - chat) + sqrt(9 - chat)) / (2 sqrt 2)``, equivalent to
    solving ``chat = 1 - 2(beta - 1/beta)^2`` on the branch with
    ``|beta| >= 1``. For the unitary N=1 minimal models (``chat < 1``) both
    square roots are real and ``beta`` is real.
    """
    first = exact_affine_number(0, c, delta_scale=0, charge_scale=Fraction(-2, 3), constant=1)
    second = exact_affine_number(0, c, delta_scale=0, charge_scale=Fraction(-2, 3), constant=9)
    return (mp.sqrt(first) + mp.sqrt(second)) / (2 * mp.sqrt(2))


def _beta_ball_from_c(c):
    from flint import acb

    first = exact_affine_ball(0, c, delta_scale=0, charge_scale=Fraction(-2, 3), constant=1)
    second = exact_affine_ball(0, c, delta_scale=0, charge_scale=Fraction(-2, 3), constant=9)
    return (first.sqrt()+second.sqrt())/(2*acb(2).sqrt())


def alpha_sq_from_delta(delta, c):
    """``alpha^2 = 8 Delta - (chat - 1)/2``, inverting [HJS-a] (3.7)."""
    return exact_affine_number(delta, c, delta_scale=8, charge_scale=Fraction(-1, 3), constant=Fraction(1, 2))


def delta_from_alpha(alpha, c):
    """``Delta = (chat - 1)/16 + alpha^2/8``, [HJS-a] (3.7)."""
    return (chat_from_c(c) - 1) / 16 + alpha**2 / 8


def ns_degenerate_weight(r: int, s: int, c):
    """``Delta_rs``, the NS degenerate weight of [HJS-a] (3.5).

    Computed as ``(chat-1)/16 + (r beta - s/beta)^2/8`` rather than from the
    expanded form, so it is manifestly the ``alpha_rs = r beta - s/beta``
    specialization of (3.7) and shares its rounding.
    """
    if r == s == 1:
        return mp.mpf(0)  # structural zero, independent of parametrization roundoff
    beta = beta_from_c(c)
    return delta_from_alpha(r * beta - s / beta, c)


def ns_kac_level(r: int, s: int) -> Fraction:
    """The null-vector level ``r s / 2`` as an exact Fraction."""
    return Fraction(r * s, 2)


def enumerate_ns_poles(ns_order: int):
    """``(r, s, rs)`` with ``r + s`` even, ``rs >= 1``, ``rs <= ns_order``.

    Ascending in ``rs``, which the kernel's triangular fill relies on. Note the
    level index ``rs`` counts powers of ``q^(1/2)``, so ``ns_order`` is a
    half-step budget.

    ``(1, 1)`` IS included, and this is easy to get wrong: [HJS-a] (2.3)/(6.1)
    restrict their pole sum to ``1 < rs``, but that is the expansion of a single
    *coefficient* of the x-series. The elliptic recursion [HJS-b] (6.5) sums over
    ``m, n > 0`` with ``m, n`` in ``2N+1``, which includes ``(1, 1)``. It has to:
    ``Delta_11 = 0`` identically in ``c``, and the closed form
    ``H^(1/2) = theta_2(q^2)/Delta`` of [HJS-b] (6.2) is nothing but that pole.
    Dropping it silently deletes the entire leading ``q^(1/2)`` term of the odd
    sector -- which is how it was caught here.
    """
    ns_order = _super_cutoff("ns_order", ns_order)
    for n in range(1, ns_order + 1):
        for r in range(1, n + 1):
            if n % r:
                continue
            s = n // r
            if (r + s) % 2 == 0:
                yield (r, s, n)


class NSDegenerateCError(ValueError):
    """The NS recursion is singular at this ``c``: a Kac pole has collided.

    ``A_rs`` divides by ``p beta - q/beta``, which vanishes whenever
    ``beta^2 = q/p`` is rational -- i.e. at every N=1 minimal model, including
    the ones of physical interest (``c = 7/10`` has ``beta^2 = 5/3``,
    ``c = 81/70`` has ``beta^2 = 7/5``). The bare NS recursion has no
    regularizer (the ``superconformal_blocks`` entry points with
    ``regularize='auto'`` instead use the Laurent recursion at the exact c), so rather than
    return ``inf``/``nan`` or a cryptic ``ZeroDivisionError`` from deep inside a
    product, this is raised with the offending indices.

    The supported workarounds are the Laurent recursion at the exact c (the
    ``superconformal_blocks`` entry points with ``regularize='auto'``) and an
    evaluation at ``c* + eps`` for a small ``eps``: the block is continuous in
    ``c`` away from the collision, and the physical truncation at a minimal
    model comes from the model's fusion rules, not from the block.
    """


def _A_rs(r: int, s: int, beta, *, _ball=False):
    """``A_rs(c)``, [HJS-a] (6.16).

    ``(1/2)(-1)^(rs-1) prod (1/sqrt2 (p beta - q/beta))^-1`` over
    ``p in [1-r, r]``, ``q in [1-s, s]`` with ``p + q`` even and
    ``(p, q) not in {(0,0), (r,s)}``.

    Note the ``(-1)^(rs-1)`` sign, which is present in [HJS-a] (6.16) but not
    in [HJS-b] (2.9) -- the two differ by the ``beta = i b`` change of variable,
    and mixing them flips the sign of every odd-level residue.
    """
    if _ball:
        from flint import acb

        root2, acc = acb(2).sqrt(), acb(1)/2
    else:
        root2, acc = mp.sqrt(2), mp.mpf(1)/2
    if (r * s - 1) % 2:
        acc = -acc
    for p in range(1 - r, r + 1):
        for q in range(1 - s, s + 1):
            if (p + q) % 2:
                continue
            if (p, q) == (0, 0) or (p, q) == (r, s):
                continue
            denom = (p * beta - q / beta) / root2
            if denom == 0:
                raise NSDegenerateCError(
                    f"A_rs({r}, {s}) is singular: p*beta - q/beta vanishes at "
                    f"(p, q) = ({p}, {q}), i.e. beta^2 = {q}/{p}. Evaluate at "
                    "c + eps instead (see NSDegenerateCError)."
                )
            acc /= denom
    return acc


def _fusion_polynomial(r: int, s: int, beta, alpha_a, alpha_b, *, starred: bool, _ball=False,
                       _smallest=False):
    """``P_rs[Delta_b; Delta_a]``, [HJS-a] (6.13) unstarred / (6.14) starred.

    ``prod (alpha_b - alpha_a + p beta - q/beta)/(2 sqrt2)
          * (alpha_b + alpha_a + p beta - q/beta)/(2 sqrt2)``
    over ``p in [1-r, r-1]``, ``q in [1-s, s-1]`` stepping by 2, subject to

        unstarred:  p + q - (r + s) in 4Z + 2       (6.13)
        starred:    p + q - (r + s) in 4Z           (6.14)

    The two index sets partition the ``(p, q)`` grid, so each polynomial carries
    ``rs/2`` factor pairs. Both are invariant under ``alpha -> -alpha`` on either
    argument (the sets are closed under ``(p,q) -> (-p,-q)``, using ``r+s`` even),
    which is why the ``alpha`` branch choice never matters.
    """
    target = 0 if starred else 2
    if _ball:
        from flint import acb

        denom, acc = 2*acb(2).sqrt(), acb(1)
    else:
        denom, acc = 2*mp.sqrt(2), mp.mpf(1)
    smallest = mp.inf
    for p in range(1 - r, r, 2):
        for q in range(1 - s, s, 2):
            if (p + q - (r + s)) % 4 != target:
                continue
            shift = p * beta - q / beta
            minus, plus = alpha_b - alpha_a + shift, alpha_b + alpha_a + shift
            if _smallest:
                # A factor that vanishes exactly cancels its terms to roundoff.
                scale = abs(alpha_a) + abs(alpha_b) + abs(shift)
                if scale:
                    smallest = min(smallest, abs(minus) / scale, abs(plus) / scale)
            acc *= (minus / denom) * (plus / denom)
    return (acc, smallest) if _smallest else acc


class NSFamily:
    """The NS pole family: two sectors, half-integer levels, theta_3 seed.

    Source parameters stay fixed; derived momenta and Kac data are cached by precision.
    """

    name = "ns"
    level_unit = Fraction(1, 2)
    n_sectors = 2

    @guard_super_inputs("ns")
    def __init__(self, c, h_ext: Sequence):
        if len(h_ext) != 4:
            raise ValueError("h_ext must be a length-4 sequence (D1, D2, D3, D4)")
        self.c = parse_number(c)
        self.h_ext = tuple(parse_number(h) for h in h_ext)
        self.beta = beta_from_c(self.c)
        # alpha enters only squared, and every polynomial is even in each
        # alpha_i, so any branch of the square root gives the same block.
        self.alpha = tuple(mp.sqrt(alpha_sq_from_delta(d, self.c)) for d in self.h_ext)
        self._parameter_prec = mp.mp.prec
        # (d, K) per (r, s, starred): each label's residue is shared by both sectors' rows
        self._residues = {}

    def _fusion(self, r, s, a, b, *, starred):
        aa, ab = self.alpha[a], self.alpha[b]
        value, smallest = _fusion_polynomial(r, s, self.beta, aa, ab, starred=starred, _smallest=True)
        # Only a factor canceled to roundoff can be an exact zero; prove it from the labels.
        if smallest <= mp.mpf(10)**(-max(8, mp.mp.dps//2)) and ns_fusion_zero(
                r, s, self.c, self.h_ext[a], self.h_ext[b], starred=starred, beta_squared=self.beta**2):
            return mp.mpf(0)
        return value

    def enumerate_poles(self, order: int):
        return list(enumerate_ns_poles(order))

    @guard_super_inputs("ns")
    def build_pole(self, label, sector: int) -> dict:
        """One pole entry, as it contributes to the block of ``sector``.

        The fusion-polynomial variant is ``starred = (rs mod 2) XOR sector``.
        This is the composition of two statements in [HJS-a]: (6.17)/(6.18) give
        the even block the unstarred polynomials and the odd block the starred
        ones, while (6.15) resolves each symbol by ``rs`` parity. Since an entry
        of sector ``t`` is consumed by the equation for sector
        ``t XOR (rs mod 2)``, the rule above is equivalent to the human-readable
        form: *the block being expanded picks the variant* -- unstarred for even,
        starred for odd, with no residual ``rs`` dependence.

        This was the one point the papers' compressed ``H^{1,1/2}`` notation left
        genuinely ambiguous, so it was settled against the closed forms of
        [HJS-b] (6.2) rather than by reading: of the four candidate rules only
        this one reproduces BOTH ``H^1 = theta_3(q^2)`` and
        ``H^(1/2) = theta_2(q^2)/Delta``. The three rejected rules fail loudly
        (relative errors 1 to 1e7), not marginally. See
        ``test_ns_starred_rule_is_pinned_by_closed_forms``.
        """
        r, s, rs = label
        if self._parameter_prec != mp.mp.prec:
            self.beta = beta_from_c(self.c)
            self.alpha = tuple(mp.sqrt(alpha_sq_from_delta(d, self.c)) for d in self.h_ext)
            self._parameter_prec = mp.mp.prec
        starred = bool((rs % 2) ^ sector)
        key = (r, s, starred, mp.mp.prec)
        if key not in self._residues:
            beta = self.beta
            K = (
                _A_rs(r, s, beta)
                * self._fusion(r, s, 3, 2, starred=starred)
                * self._fusion(r, s, 0, 1, starred=starred)
            )
            # Absorb (16 q)^(rs/2) = 16^(rs/2) t^rs = 4^rs t^rs into K, mirroring
            # how the Virasoro path absorbs 16^{rs}.
            self._residues[key] = (ns_degenerate_weight(r, s, self.c), K * mp.mpf(4) ** rs)
        d, K = self._residues[key]
        return dict(
            r=r,
            s=s,
            rs=rs,
            K=K,
            d=d,
            v=d + mp.mpf(rs) / 2,
            sector=sector,
        )

    def couples(self, outer: dict, inner: dict) -> bool:
        """Sector flips on an odd-level pole, [HJS-b] (6.5)."""
        return (outer.get("sector", 0) ^ (inner["rs"] % 2)) == inner.get("sector", 0)

    def seed(self, sector: int, order: int, field) -> list:
        """``g^1 = theta_3(q^2)``, ``g^(1/2) = 0`` -- [HJS-b] (6.4), unstarred."""
        out = [field(0)] * (order + 1)
        if sector != SECTOR_EVEN:
            return out
        # theta_3(q^2) = 1 + 2 sum_{n>=1} q^{2 n^2}; in t = q^(1/2) the powers
        # are t^{4 n^2}.
        out[0] = field(1)
        n = 1
        while 4 * n * n <= order:
            out[4 * n * n] = field(2)
            n += 1
        return out


def _field_for(*values):
    """``mp.mpf`` when every value is an mpf / int or an mpc with zero imaginary part; anything else (a Python
    float included) selects ``mp.mpc``."""
    for v in values:
        if isinstance(v, mp.mpc) and v.imag != 0:
            return mp.mpc
        if not isinstance(v, (mp.mpf, mp.mpc, int)):
            return mp.mpc
    return mp.mpf


def _ns_pole_balls(family, poles):
    """Kac enclosures from the exact source labels, in the table's row order."""
    from flint import acb

    beta = _beta_ball_from_c(family.c)
    beta_squared = beta*beta
    alpha = [exact_affine_ball(h, family.c, delta_scale=8, charge_scale=Fraction(-1, 3),
                               constant=Fraction(1, 2)).sqrt() for h in family.h_ext]
    cache, out = {}, []
    for pole in poles:
        r, s, rs = pole['r'], pole['s'], pole['rs']
        starred = bool((rs % 2)^pole['sector'])
        key = (r, s, starred)
        if key not in cache:
            def fusion(a, b):
                value = _fusion_polynomial(r, s, beta, alpha[a], alpha[b], starred=starred, _ball=True)
                if value.contains(0) and ns_fusion_zero(r, s, family.c, family.h_ext[a], family.h_ext[b],
                                                       starred=starred, beta_squared=family.beta**2):
                    return acb(0)
                return value

            left, right = fusion(3, 2), fusion(0, 1)
            residue = acb(0) if left.is_zero() or right.is_zero() else (
                _A_rs(r, s, beta, _ball=True)*left*right*4**rs)
            degenerate = ((r*r-1)*beta_squared+(s*s-1)/beta_squared-2*(rs-1))/8
            cache[key] = residue, degenerate
        residue, degenerate = cache[key]
        out.append(dict(pole, K=residue, d=degenerate))
    return out


def build_ns_level_table(c, h_ext, ns_order: int) -> _poles.LevelTable:
    """Kac data plus the filled inner-block table -- independent of ``Delta``.

    Split out (and cached by :func:`ns_level_table`) for the same reason
    as the Virasoro ``_build_H_context``: it depends only on ``(c, externals)``,
    so one build serves every exchanged weight, which is what makes sampling a
    grid of Deltas affordable.
    """
    ns_order = _super_cutoff("ns_order", ns_order)

    def poles_at(family):
        labels = family.enumerate_poles(ns_order)
        # One entry per (sector, pole): the pole locations d and intermediate
        # weights v are sector-independent, but the residue numerator K is not (the
        # starred/unstarred fusion polynomial depends on which block it feeds), so
        # the two sectors cannot share entries.
        doubled = [
            family.build_pole(label, sector)
            for sector in (SECTOR_EVEN, SECTOR_ODD)
            for label in labels
        ]
        # The kernel's fill requires ascending level order.
        doubled.sort(key=lambda p: (p["rs"], p["sector"]))
        return doubled

    def rebuild(prec_bits):
        return _ns_pole_balls(family, doubled)

    family = NSFamily(c, h_ext)
    doubled = poles_at(family)

    field = _field_for(c, *h_ext, family.beta, *family.alpha)
    seeds = {
        sector: family.seed(sector, ns_order, field)
        for sector in (SECTOR_EVEN, SECTOR_ODD)
    }
    from .recursion import _input_bits

    built = _poles.fill_level_table_flint(doubled, ns_order, seeds, field, couples=family.couples, rebuild=rebuild,
                                          level_unit=Fraction(1, 2), collision_check=_poles._kac_collision_checker(
                                              family.beta, c=family.c, superconformal=True),
                                          input_bits=_input_bits((c, *h_ext)))
    if built[0] == "collision":
        outer, inner = doubled[built[1][0]], doubled[built[1][1]]
        raise NSDegenerateCError(
            "NS Kac poles collide at this c: intermediate weight "
            f"v_{outer['r'],outer['s']} equals Delta_{inner['r'],inner['s']}. "
            "Evaluate at c + eps instead (see NSDegenerateCError)."
        )
    return _poles.LevelTable(poles=doubled, table=built[0], order=ns_order, family=family, flint_table=built[1],
                             flint_kd=built[2])


_NS_TABLE_CACHE: dict = {}
_NS_TABLE_CACHE_MAXSIZE = 32


def _exact_cache_part(x):
    """Return an exact, hashable scalar key.

    Exact mantissa/exponent tuples distinguish small regularization offsets
    that a rounded decimal representation would merge.
    """
    x = parse_number(x)
    if isinstance(x, Fraction):
        return ("rational", x.numerator, x.denominator)
    if isinstance(x, mp.mpf):
        return x._mpf_
    if isinstance(x, mp.mpc):
        return (x.real._mpf_, x.imag._mpf_)
    if isinstance(x, int):
        return ("integer", x)
    return mp.mpc(x).real._mpf_, mp.mpc(x).imag._mpf_


def _ns_table_cache_key(c, h_ext, ns_order: int):
    """Exact cache key, or ``None`` for un-hashable inputs."""
    try:
        parts = tuple(_exact_cache_part(x) for x in (c, *h_ext))
    except (TypeError, ValueError):
        return None
    return (parts, int(ns_order), mp.mp.dps)


def ns_level_table(c, h_ext, ns_order: int) -> _poles.LevelTable:
    """Cached :func:`build_ns_level_table`."""
    ns_order = _super_cutoff("ns_order", ns_order)
    key = _ns_table_cache_key(c, h_ext, ns_order)
    if key is not None and key in _NS_TABLE_CACHE:
        return _NS_TABLE_CACHE[key]
    tbl = build_ns_level_table(c, h_ext, ns_order)
    if key is not None:
        if len(_NS_TABLE_CACHE) >= _NS_TABLE_CACHE_MAXSIZE:
            _NS_TABLE_CACHE.clear()
        _NS_TABLE_CACHE[key] = tbl
    return tbl


@guard_super_inputs("ns")
def H_q_series_ns(
    delta,
    h_ext: Sequence,
    *,
    c,
    ns_order: int | None = None,
    q_order: int | None = None,
    sector: int = SECTOR_EVEN,
    as_acb: bool = False,
    certify: bool = True,
) -> list:
    """``[H_0, ..., H_ns_order]`` with ``H^sector(q) = sum_k H_k q^(k/2)``.

    Evaluated in flint (:func:`virasoro_blocks.poles.level_series_acb`), certified to the working precision
    (:func:`virasoro_blocks.poles.certified_level_series`); ``as_acb`` returns the flint values themselves (for the
    series samplers), else mpmath numbers of the inputs' field.  ``certify=False`` (with ``as_acb``) returns the balls
    at the working precision as they come, for a caller that certifies what it forms from them.

    NOTE THE HALF-POWERS: the coefficient list is indexed by powers of
    ``q^(1/2)``, so ``ns_order`` reaches only ``q^(ns_order/2)``. Use
    :func:`ns_order_for_q_order` to convert from a Virasoro-style budget.

    The ``16^(rs/2)`` factors of ``(16q)^(rs/2)`` are absorbed into the returned
    coefficients, so the caller multiplies by plain ``q^(1/2)``.

    ``sector`` picks the even block ``H^1`` (default, the one appearing in a
    correlator of four NS super-primaries) or the odd block ``H^(1/2)``.
    """
    ns_order = _resolve_ns_order(ns_order, q_order)
    if sector not in (SECTOR_EVEN, SECTOR_ODD):
        raise ValueError(f"sector must be {SECTOR_EVEN} or {SECTOR_ODD}")
    from .super_laurent import exact_singular_c, merged_H, merged_state

    c0 = exact_singular_c(c)
    if c0 is not None:
        # at an N=1 singular c the Kac poles collide: the exact Laurent-in-t recursion (the c -> c0 limit)
        values = merged_H(merged_state("ns", h_ext, c0=c0, order=ns_order, sector=sector), delta)
        if as_acb:
            from .polynomial import _mpf_to_acb

            return [_mpf_to_acb(value) for value in values]
        return values
    tbl = ns_level_table(c, h_ext, ns_order)
    field = _field_for(delta, c, *h_ext)
    # an mpc delta with zero imaginary part selects the real field; mp.mpf(mpc) would raise
    delta_typed = field(mp.re(delta)) if field is mp.mpf else mp.mpc(delta)
    seed = tbl.family.seed(sector, ns_order, field)
    if certify:
        from .recursion import _input_bits

        flint_values = _poles.certified_level_series(lambda: ns_level_table(c, h_ext, ns_order), delta_typed,
                                                     ns_order, seed, couples=tbl.family.couples, sector=sector,
                                                     what="the NS H series", input_bits=_input_bits((c, *h_ext, delta)))
    else:
        flint_values = _poles.level_series_acb(tbl, delta_typed, ns_order, seed, couples=tbl.family.couples,
                                               sector=sector)
    if flint_values is not None:
        if as_acb:
            return flint_values
        from .derivs import _acb_to_scalar
        from .polynomial import _arb_to_mpf

        if field is mp.mpf:
            return [_arb_to_mpf(value.real) for value in flint_values]
        return [mp.mpc(_acb_to_scalar(value)) for value in flint_values]
    values = _poles.eval_level_series(
        delta_typed,
        tbl.poles,
        tbl.table,
        ns_order,
        seed,
        field,
        couples=tbl.family.couples,
        sector=sector,
    )
    if as_acb:
        from .polynomial import _mpf_to_acb

        return [_mpf_to_acb(value) for value in values]
    return values


def _horner_half_series(coeffs, q):
    """``sum_k coeffs[k] q^(k/2)``, i.e. Horner in ``t = sqrt(q)``."""
    t = mp.sqrt(q)
    acc = mp.mpf(0) if (isinstance(t, mp.mpf) and
                        all(isinstance(x, mp.mpf) for x in coeffs)) else mp.mpc(0)
    for k in range(len(coeffs) - 1, -1, -1):
        acc = acc * t + coeffs[k]
    return acc


def certified_half_value(coefficients_at, z):
    """``sum_k H_k q^(k/2)`` certified to the working precision: ``coefficients_at()`` gives the ``H_k`` as flint balls
    at the current precision (as they come: one ladder certifies the coefficients' loss and the sum's together), the
    sum is formed in balls (``sqrt(q)`` with its rounding radius), and the whole evaluation repeats at the working
    precision plus the measured deficit while it falls short -- the sum cancels on its own at heavy externals and z
    near 1, as the Virasoro value (``recursion._certified_block_value``)."""
    from flint import acb, arb, ctx as flint_ctx

    from .polynomial import _acb_to_mpc, _arb_to_mpf, _flint_prec_bits, _mpf_to_arb
    from .recursion import _relative_deficit

    work = mp.mp.prec
    extra, previous = 0, None
    while True:
        with mp.workprec(work + extra):
            H = coefficients_at()
            t = mp.sqrt(q_from_z(z))
            flint_ctx.prec = _flint_prec_bits()
            real = isinstance(t, mp.mpf) and all(v.imag.is_zero() for v in H)
            rad = _mpf_to_arb(abs(t) * mp.eps)
            if real:
                t_ball, H, total = arb(_mpf_to_arb(t), rad), [v.real for v in H], arb(0)
            else:
                t_c = mp.mpc(t)
                t_ball, total = acb(arb(_mpf_to_arb(t_c.real), rad), arb(_mpf_to_arb(t_c.imag), rad)), acb(0)
            for coefficient in reversed(H):
                total = total * t_ball + coefficient
            deficit = _relative_deficit([total], work)
            if deficit == 0:
                value = _arb_to_mpf(total) if real else _acb_to_mpc(total)
                break
        step = _poles.next_certification_prec(work + extra, deficit, previous, work)
        if step is None:
            raise _poles.LevelTablePrecisionError(
                f"the half-power sum cannot be certified to {mp.mp.dps} digits at z = {mp.nstr(z, 12)}: it stays "
                f"{deficit} bits short with {extra} guard bits")
        extra, previous = step[0] - work, step[1]
    flint_ctx.prec = _flint_prec_bits()
    return +value


def block_prefactor_ns(
    delta, z, h_ext: Sequence, *, c, chiral_norm=None, z_shift=None
):
    """NS block prefactor, [HJS-b] eq (6.1).

    ``F = (16q)^(Delta - cs/24) z^(cs/24 - D1 - D2) (1-z)^(cs/24 - D2 - D3)
          theta_3(q)^(cs/2 - 4 sum D) H``   with ``cs = c - 3/2``.

    This is the Virasoro ``block_prefactor`` with the central-charge
    combination ``c - 1`` replaced by ``c - 3/2`` in each exponent. The
    repo's ``chiral_norm`` (N) and ``z_shift`` conventions are kept identical to
    the Virasoro case so that downstream code (crossing, moments) needs no
    NS-specific handling: ``F ~ z^(Delta + z_shift) / N^Delta`` as ``z -> 0``.
    """
    N = _resolve_chiral_norm(chiral_norm)
    c = _maybe_real_scalar(c) if isinstance(c, (mp.mpf, mp.mpc)) else mp.mpc(c)
    z = (
        _maybe_real_scalar(z)
        if isinstance(z, (mp.mpf, mp.mpc))
        else _maybe_real_scalar(mp.mpc(z))
    )
    delta = (
        _maybe_real_scalar(delta)
        if isinstance(delta, (mp.mpf, mp.mpc))
        else _maybe_real_scalar(mp.mpc(delta))
    )
    d1, d2, d3, d4 = (
        _maybe_real_scalar(di) if isinstance(di, (mp.mpf, mp.mpc))
        else _maybe_real_scalar(mp.mpc(di))
        for di in h_ext
    )
    sigma = (c - mp.mpf(3) / 2) / 24
    q = q_from_z(z)
    z_exp = sigma + (mp.mpf(0) if z_shift is None else _maybe_real_scalar(z_shift))
    q16 = 16 * q
    return (
        q16 ** (-sigma)
        * (q16 / N) ** delta
        * z**z_exp
        * (1 - z) ** (sigma - d2 - d3)
        * _theta3(q) ** ((c - mp.mpf(3) / 2) / 2 - 4 * (d1 + d2 + d3 + d4))
    )


@guard_super_inputs("ns")
def F_block_ns(
    delta,
    z,
    h_ext: Sequence,
    *,
    c,
    ns_order: int | None = None,
    q_order: int | None = None,
    chiral_norm=None,
    z_shift=None,
    sector: int = SECTOR_EVEN,
):
    """Sphere 4-point NS superconformal block ``F^sector(Delta, z; D_i; c)``.

    The even sector (default) is the block that appears in a four-point function
    of NS super-primary fields. Truncated at ``q^(ns_order/2)``.

    Recognized exact singular central charges use the Laurent recursion at
    that charge through :func:`H_q_series_ns`. A genuine exchanged-weight pole
    raises :class:`virasoro_blocks.super_laurent.SuperLaurentPoleError`.
    :func:`virasoro_blocks.superconformal_blocks.ns_chiral_block` supplies the
    shared policy and exposes such failures as ``SuperconformalPoleError``.
    """
    # exact inputs (Fraction, int, str) are converted at the working precision; mpc stays complex
    from .singularity import snap_float_c

    from .numeric_input import parse_number

    delta = parse_number(delta)
    h_ext = [parse_number(value) for value in h_ext]
    z, c = mp.convert(z), snap_float_c(c, family="ns")
    ns_order = _resolve_ns_order(ns_order, q_order)
    total = certified_half_value(
        lambda: H_q_series_ns(delta, h_ext, c=c, ns_order=ns_order, sector=sector, as_acb=True, certify=False), z)
    pref = block_prefactor_ns(
        mp.convert(delta), z, [mp.convert(v) for v in h_ext], c=mp.convert(c), chiral_norm=chiral_norm, z_shift=z_shift
    )
    return pref * total


@lru_cache(maxsize=64)
def _theta_half_powers(kind: int, order: int, dps: int) -> tuple:
    """Coefficients of ``theta_kind(q^2)`` in ``t = q^(1/2)``, for the gates."""
    with mp.workdps(dps):
        out = [mp.mpf(0)] * (order + 1)
        if kind == 3:
            out[0] = mp.mpf(1)
            n = 1
            while 4 * n * n <= order:
                out[4 * n * n] = mp.mpf(2)
                n += 1
        elif kind == 2:
            # theta_2(q^2) = 2 sum_{n>=0} q^{2(n+1/2)^2}; t-powers (2n+1)^2.
            n = 0
            while (2 * n + 1) ** 2 <= order:
                out[(2 * n + 1) ** 2] = mp.mpf(2)
                n += 1
        else:
            raise ValueError("kind must be 2 or 3")
        return tuple(out)


def theta_half_series(kind: int, order: int) -> list:
    """``theta_2(q^2)`` or ``theta_3(q^2)`` as a ``q^(1/2)`` coefficient list."""
    order = _super_cutoff("order", order)
    return list(_theta_half_powers(kind, order, mp.mp.dps))
