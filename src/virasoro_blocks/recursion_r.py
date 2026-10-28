"""N=1 Ramond-external blocks factorized on NS intermediate states.

This is the Ramond-sector counterpart of :mod:`recursion_ns`.  The recurrence
is the one in Hadasz--Jaskolski--Suchanek, arXiv:0810.1203, eq. (6.5): four
Ramond external fields factorize on NS states, and an odd Kac pole flips both
the even/odd block and the sign branch of the Ramond three-point form.

The public entry point defaults to the diagonal ``++`` branch.  This is the
branch for a correlator of identical bosonic Ramond components; its ``(1,1)``
odd residue vanishes, as it must for the identity NS module.  The other three
sign branches are also available explicitly because odd poles feed them while
filling the complete Ramond block family.

The paper uses ``b`` with ``c = 3/2 + 3 (b + b**-1)**2`` and Ramond momenta
``gamma`` defined by ``Delta = c/24 - gamma**2``.  The rest of this repository
uses ``beta`` with ``b = i beta``.  All residue formulae below are evaluated in
the paper's ``b`` convention, so no phase is silently mixed into the fusion
polynomials or ``A_rs``.
"""

from __future__ import annotations

from fractions import Fraction
from .numeric_input import parse_number
from .super_kac import _super_cutoff, exact_affine_number, exact_affine_ball, guard_super_inputs, ramond_fusion_zero
from typing import Sequence

import mpmath as mp

from . import poles as _poles
from .recursion import _maybe_real_scalar
from .recursion_ns import (
    SECTOR_EVEN,
    SECTOR_ODD,
    _resolve_ns_order,
    beta_from_c,
    _beta_ball_from_c,
    block_prefactor_ns,
    enumerate_ns_poles,
)


SIGN_PLUS = 0
SIGN_MINUS = 1
BRANCH_PP = 0  # (+ beta_3, + beta_2)
BRANCH_PM = 1  # (+ beta_3, - beta_2)
BRANCH_MP = 2  # (- beta_3, + beta_2)
BRANCH_MM = 3  # (- beta_3, - beta_2)
_N_STATES = 8


def _state(sector: int, sign: int, branch: int | None = None) -> int:
    if sector not in (SECTOR_EVEN, SECTOR_ODD):
        raise ValueError("sector must be 0 (even) or 1 (odd)")
    if branch is None:
        # Backwards-compatible diagonal branches: sign=0 -> ++, sign=1 -> --.
        if sign not in (SIGN_PLUS, SIGN_MINUS):
            raise ValueError("sign must be 0 (++) or 1 (--)")
        branch = BRANCH_PP if sign == SIGN_PLUS else BRANCH_MM
    if branch not in range(4):
        raise ValueError("branch must be one of 0 (++), 1 (+-), 2 (-+), 3 (--)")
    if sign not in (SIGN_PLUS, SIGN_MINUS):
        raise ValueError("sign must be 0 (++) or 1 (--)")
    return 4 * sector + branch


def _state_parts(state: int) -> tuple[int, int]:
    return state // 4, state % 4


def ramond_parameter_from_delta(delta, c):
    """Return ``gamma`` with ``Delta = c/24 - gamma**2``."""
    return mp.sqrt(exact_affine_number(delta, c, delta_scale=-1, charge_scale=Fraction(1, 24)))


def _b_from_c(c):
    """HJS ``b`` in terms of the repository's NS ``beta``."""
    return 1j * beta_from_c(c)


def _r_degenerate_weight(r: int, s: int, c):
    """NS Kac weight at the pole of a Ramond-external block."""
    if r == s == 1:
        return mp.mpf(0)
    b = _b_from_c(c)
    return (
        -mp.mpf(r * s - 1) / 4
        + mp.mpf(1 - r * r) * b * b / 8
        + mp.mpf(1 - s * s) / (8 * b * b)
    )


def _A_rs(r: int, s: int, b, *, _ball=False):
    """HJS eq. (5.12), with the source convention ``b`` kept explicit."""
    if _ball:
        from flint import acb

        acc, root2 = acb(1)/2, acb(2).sqrt()
    else:
        acc, root2 = mp.mpc(1)/2, mp.sqrt(2)
    for p in range(1 - r, r + 1):
        for q in range(1 - s, s + 1):
            if (p + q) % 2:
                continue
            if (p, q) in ((0, 0), (r, s)):
                continue
            acc /= (p * b + q / b) / root2
    return acc


def _fusion_polynomial(r: int, s: int, b, beta_lower, beta_upper, *, sign: int, _ball=False,
                       _smallest=False):
    """HJS eq. (5.10), ``P_c^{rs}[^{+/- beta_upper}_{beta_lower}]``."""
    upper = beta_upper if sign == SIGN_PLUS else -beta_upper
    if _ball:
        from flint import acb

        acc, two_root2 = acb(1), 2*acb(2).sqrt()
    else:
        acc, two_root2 = mp.mpc(1), 2*mp.sqrt(2)
    smallest = mp.inf
    for target, signed_upper in ((2, -upper), (0, upper)):
        for p in range(1 - r, r, 2):
            for q in range(1 - s, s, 2):
                if (p + q - (r + s)) % 4 != target:
                    continue
                shift = (p * b + q / b) / two_root2
                factor = beta_lower + signed_upper + shift
                if _smallest:
                    # A factor that vanishes exactly cancels its terms to roundoff.
                    scale = abs(beta_lower) + abs(upper) + abs(shift)
                    if scale:
                        smallest = min(smallest, abs(factor) / scale)
                acc *= factor
    return (acc, smallest) if _smallest else acc


class RDegenerateCError(ValueError):
    """The Ramond recursion is singular at an N=1 minimal-model ``c``."""


class RFamily:
    """Eight internal states: even/odd times the four HJS sign branches."""

    name = "r"
    level_unit = Fraction(1, 2)
    n_sectors = _N_STATES

    @guard_super_inputs("ramond")
    def __init__(self, c, h_ext: Sequence):
        if len(h_ext) != 4:
            raise ValueError("h_ext must be a length-4 sequence (D1, D2, D3, D4)")
        self.c = parse_number(c)
        self.h_ext = tuple(parse_number(h) for h in h_ext)
        self.b = _b_from_c(self.c)
        self.gamma = tuple(ramond_parameter_from_delta(d, self.c) for d in self.h_ext)
        self._parameter_prec = mp.mp.prec
        # (d, K) per (r, s, outer sign bits): the states' rows share them
        self._residues = {}

    def _fusion(self, r, s, lower, upper, *, sign):
        gl, gu = self.gamma[lower], self.gamma[upper]
        value, smallest = _fusion_polynomial(r, s, self.b, gl, gu, sign=sign, _smallest=True)
        # Only a factor canceled to roundoff can be an exact zero; prove it from the labels.
        if smallest <= mp.mpf(10)**(-max(8, mp.mp.dps//2)) and ramond_fusion_zero(
                r, s, self.c, self.h_ext[lower], self.h_ext[upper], sign=sign,
                b=self.b, gamma_lower=gl, gamma_upper=gu):
            return mp.mpc(0)
        return value

    def enumerate_poles(self, order: int):
        return list(enumerate_ns_poles(order))

    @guard_super_inputs("ramond")
    def build_pole(self, label, state: int) -> dict:
        r, s, rs = label
        if self._parameter_prec != mp.mp.prec:
            self.b = _b_from_c(self.c)
            self.gamma = tuple(ramond_parameter_from_delta(d, self.c) for d in self.h_ext)
            self._parameter_prec = mp.mp.prec
        # The generic pole kernel stores the state of the INNER block in a pole
        # row.  The residue, however, belongs to the OUTER block, so an odd
        # pole flips the supplied state before selecting the Ramond sign.
        inner_sector, inner_branch = _state_parts(state)
        outer_state = state if not (rs % 2) else state ^ 7
        outer_sector, outer_branch = _state_parts(outer_state)
        outer_sign3 = (outer_branch >> 1) & 1
        outer_sign2 = outer_branch & 1
        key = (r, s, outer_sign3, outer_sign2, mp.mp.prec)
        if key not in self._residues:
            b = self.b
            residue = (
                _A_rs(r, s, b)
                * self._fusion(r, s, 3, 2, sign=outer_sign3)
                * self._fusion(r, s, 0, 1, sign=outer_sign2)
                * mp.mpf(4) ** rs
            )
            # HJS eq. (6.5): odd poles have a minus sign and feed the opposite
            # sign branch as well as the opposite even/odd block.
            if rs % 2:
                residue = -residue
            self._residues[key] = (_r_degenerate_weight(r, s, self.c), residue)
        d, residue = self._residues[key]
        return dict(
            r=r, s=s, rs=rs, K=residue, d=d, v=d + mp.mpf(rs) / 2,
            # ``sector`` is the kernel's state key (eight states, not merely
            # the even/odd parity); ``parity`` and ``branch`` are diagnostic
            # labels.  This lets the generic pole kernel seed and route all
            # branches without a special case.
            sector=state, parity=inner_sector, branch=inner_branch,
            outer_parity=outer_sector, outer_branch=outer_branch,
        )

    def couples(self, outer: dict, inner: dict) -> bool:
        # For a pole of level rs, an odd level flips both state bits.  The
        # top evaluator is represented by the same integer in ``sector`` as
        # the table rows, so this one rule handles both forms.
        outer_state = outer.get("sector", 0)
        desired = outer_state if not (inner["rs"] % 2) else outer_state ^ 7
        return inner.get("sector", 0) == desired

    def seed(self, state: int, order: int, field) -> list:
        sector, _branch = _state_parts(state)
        out = [field(0)] * (order + 1)
        if sector == SECTOR_EVEN:
            out[0] = field(1)
        return out


def _field_for(*values):
    # b is imaginary in the minimal-model convention, so use a complex field
    # throughout.  Public evaluators collapse exact-zero imaginary parts.
    return mp.mpc


def _r_pole_balls(family, poles):
    """Kac enclosures in all eight states, retaining the HJS sign branches."""
    from flint import acb

    b = acb(0, 1)*_beta_ball_from_c(family.c)
    squared = b*b
    gamma = [exact_affine_ball(h, family.c, delta_scale=-1, charge_scale=Fraction(1, 24)).sqrt()
             for h in family.h_ext]
    cache, out = {}, []
    for pole in poles:
        r, s, rs = pole['r'], pole['s'], pole['rs']
        outer_state = pole['sector'] if not rs % 2 else pole['sector']^7
        branch = _state_parts(outer_state)[1]
        sign3, sign2 = (branch >> 1)&1, branch&1
        key = (r, s, sign3, sign2)
        if key not in cache:
            def fusion(lower, upper, sign):
                value = _fusion_polynomial(r, s, b, gamma[lower], gamma[upper], sign=sign, _ball=True)
                if value.contains(0) and ramond_fusion_zero(
                        r, s, family.c, family.h_ext[lower], family.h_ext[upper], sign=sign,
                        b=family.b, gamma_lower=family.gamma[lower], gamma_upper=family.gamma[upper]):
                    return acb(0)
                return value

            left, right = fusion(3, 2, sign3), fusion(0, 1, sign2)
            residue = acb(0) if left.is_zero() or right.is_zero() else (
                _A_rs(r, s, b, _ball=True)*left*right*4**rs)
            if rs % 2:
                residue = -residue
            degenerate = ((1-r*r)*squared+(1-s*s)/squared-2*(rs-1))/8
            cache[key] = residue, degenerate
        residue, degenerate = cache[key]
        out.append(dict(pole, K=residue, d=degenerate))
    return out


def build_r_level_table(c, h_ext, r_order: int) -> _poles.LevelTable:
    r_order = _super_cutoff("r_order", r_order)

    def poles_at(family):
        labels = family.enumerate_poles(r_order)
        built_poles = [
            family.build_pole(label, state)
            for state in range(_N_STATES)
            for label in labels
        ]
        built_poles.sort(key=lambda p: (p["rs"], p["sector"], p["branch"]))
        return built_poles

    def rebuild(prec_bits):
        return _r_pole_balls(family, poles)

    family = RFamily(c, h_ext)
    poles = poles_at(family)
    field = _field_for(c, *h_ext, family.b, *family.gamma)
    seeds = {
        state: family.seed(state, r_order, field)
        for state in range(_N_STATES)
    }
    from .recursion import _input_bits

    built = _poles.fill_level_table_flint(poles, r_order, seeds, field, couples=family.couples, rebuild=rebuild,
                                          level_unit=Fraction(1, 2), collision_check=_poles._kac_collision_checker(
                                              -1j*family.b, c=family.c, superconformal=True),
                                          input_bits=_input_bits((c, *h_ext)))
    if built[0] == "collision":
        outer, inner = poles[built[1][0]], poles[built[1][1]]
        raise RDegenerateCError(
            "Ramond recursion Kac poles collide at this c: "
            f"v_({outer['r']},{outer['s']}) equals "
            f"Delta_({inner['r']},{inner['s']}); evaluate at c + eps"
        )
    return _poles.LevelTable(poles=poles, table=built[0], order=r_order, family=family, flint_table=built[1],
                             flint_kd=built[2])


_R_TABLE_CACHE = {}


def _cache_part(x):
    if isinstance(x, mp.mpf):
        return ("mpf", x._mpf_)
    if isinstance(x, mp.mpc):
        return ("mpc", x._mpc_)
    return (type(x).__name__, str(x))


def r_level_table(c, h_ext, r_order: int):
    r_order = _super_cutoff("r_order", r_order)
    key = (tuple(_cache_part(x) for x in (c, *h_ext)), int(r_order), mp.mp.dps)
    cached = _R_TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    table = build_r_level_table(c, h_ext, r_order)
    if len(_R_TABLE_CACHE) > 32:
        _R_TABLE_CACHE.clear()
    _R_TABLE_CACHE[key] = table
    return table


@guard_super_inputs("ramond")
def H_q_series_r(
    delta,
    h_ext: Sequence,
    *,
    c,
    r_order: int | None = None,
    q_order: int | None = None,
    sector: int = SECTOR_EVEN,
    sign: int = SIGN_PLUS,
    branch: int | None = None,
    as_acb: bool = False,
    certify: bool = True,
) -> list:
    """Return the Ramond-external elliptic ``H`` series in ``q^(1/2)``.

    ``sector=0, sign=0`` is the physical identical-Ramond ``++`` branch.  The
    other states are available for tests and for future mixed-component blocks.
    Evaluated in flint (:func:`virasoro_blocks.poles.level_series_acb`), certified to the working precision
    (:func:`virasoro_blocks.poles.certified_level_series`); ``as_acb`` returns the flint values themselves, and
    ``certify=False`` (with ``as_acb``) the balls at the working precision as they come.
    """
    order = _resolve_ns_order(r_order, q_order)
    state = _state(sector, sign, branch)
    from .super_laurent import exact_singular_c, merged_H, merged_state

    c0 = exact_singular_c(c)
    if c0 is not None:
        # at an N=1 singular c the Kac poles collide: the exact Laurent-in-t recursion (the c -> c0 limit)
        values = merged_H(merged_state("ramond", h_ext, c0=c0, order=order, sector=state // 4, branch=state % 4),
                          delta)
        if as_acb:
            from .polynomial import _mpf_to_acb

            return [_mpf_to_acb(mp.mpc(value)) for value in values]
        return [_maybe_real_scalar(mp.mpc(v)) for v in values]
    table = r_level_table(c, h_ext, order)
    value = mp.mpc(delta)
    seed = table.family.seed(state, order, mp.mpc)
    if certify:
        from .recursion import _input_bits

        flint_values = _poles.certified_level_series(lambda: r_level_table(c, h_ext, order), value, order, seed,
                                                     couples=table.family.couples, sector=state,
                                                     what="the Ramond H series",
                                                     input_bits=_input_bits((c, *h_ext, delta)))
    else:
        flint_values = _poles.level_series_acb(table, value, order, seed, couples=table.family.couples, sector=state)
    if flint_values is not None:
        if as_acb:
            return flint_values
        from .derivs import _acb_to_scalar

        return [_acb_to_scalar(v) for v in flint_values]
    out = _poles.eval_level_series(
        value,
        table.poles,
        table.table,
        order,
        seed,
        mp.mpc,
        couples=table.family.couples,
        sector=state,
    )
    if as_acb:
        from .polynomial import _mpf_to_acb

        return [_mpf_to_acb(v) for v in out]
    return [_maybe_real_scalar(v) for v in out]


def block_prefactor_r(
    delta, z, h_ext: Sequence, *, c, chiral_norm=None, z_shift=None
):
    """Ramond-external elliptic prefactor.

    The Ramond recurrence changes the elliptic ``H`` series and its branch
    data, but not the free-field prefactor multiplying that series.  In the
    Hadasz--Jaskolski--Suchanek normalization it is therefore the same
    ``c - 3/2`` prefactor used by the NS block.  Keeping this as a named
    wrapper is useful to callers: the sector is visible in the public API and
    the normalization cannot accidentally drift between NS and Ramond code.
    """
    return block_prefactor_ns(
        mp.convert(delta), z, [mp.convert(v) for v in h_ext], c=mp.convert(c), chiral_norm=chiral_norm, z_shift=z_shift
    )


@guard_super_inputs("ramond")
def F_block_r(
    delta,
    z,
    h_ext: Sequence,
    *,
    c,
    r_order: int | None = None,
    q_order: int | None = None,
    chiral_norm=None,
    z_shift=None,
    sector: int = SECTOR_EVEN,
    sign: int = SIGN_PLUS,
    branch: int | None = None,
):
    """Ramond-external chiral block factorized on an NS intermediate module.

    ``branch`` selects one of the four HJS sign branches.  The default
    ``sector=even, branch=++`` is the diagonal block for identical bosonic
    Ramond components.  As for :func:`F_block_ns`, the returned series is
    truncated at ``q**(r_order/2)`` (or at ``q_order`` when that named unit is
    used).
    """
    # exact inputs (Fraction, int, str) are converted at the working precision; mpc stays complex
    from .singularity import snap_float_c

    from .numeric_input import parse_number

    delta = parse_number(delta)
    h_ext = [parse_number(value) for value in h_ext]
    z, c = mp.convert(z), snap_float_c(c, family="ns")
    order = _resolve_ns_order(r_order, q_order)
    from .recursion_ns import certified_half_value

    total = certified_half_value(lambda: H_q_series_r(
        delta, h_ext, c=c, r_order=order, sector=sector, sign=sign, branch=branch, as_acb=True, certify=False), z)
    pref = block_prefactor_r(
        mp.convert(delta), z, [mp.convert(v) for v in h_ext], c=mp.convert(c), chiral_norm=chiral_norm, z_shift=z_shift
    )
    return pref * total


__all__ = [
    "RDegenerateCError",
    "RFamily",
    "SIGN_PLUS",
    "SIGN_MINUS",
    "BRANCH_PP",
    "BRANCH_PM",
    "BRANCH_MP",
    "BRANCH_MM",
    "H_q_series_r",
    "block_prefactor_r",
    "F_block_r",
    "ramond_parameter_from_delta",
    "r_level_table",
]
