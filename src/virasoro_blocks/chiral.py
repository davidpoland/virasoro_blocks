"""Symbolic-h representation of the chiral block F(h, z; h_i; c) at fixed z.

The library's default block equals the standard eq. (2.19) block times z^(h1+h2),
divided by N^h, with N = `chiral_norm` (default 16). The z-power is controlled
by `z_shift` (default 0):

    F(h, z) = N^{-h} (16 q)^{h - sigma} z^{sigma + z_shift} (1 - z)^{sigma - h2 - h3}
              theta_3(q)^{(c-1)/2 - 4 (h1+h2+h3+h4)} H(c, h, h_i, q),

    sigma = (c - 1) / 24,   q = q(z).

With z_shift = 0 (default), the block behaves as z^h / N^h at small z.
With z_shift = -h1-h2, it behaves as z^{h-h1-h2} / N^h.

Regrouping h-dependence,

    N^{-h} (16 q)^{h - sigma}
        = (16 q / N)^h * (16 q)^{-sigma},

so at fixed z

    F(h, z) = E(z)^h * A(z) * N_z(h) / D(h),

where
    E(z)        = 16 q(z) / N                                       (h-exponential base)
    A(z)        = (16 q)^{-sigma} z^{sigma + z_shift} (1-z)^{sigma - h2 - h3} theta_3(q)^{...}
                                                                    (h-independent scalar)
    N_z(h)      = sum_k N_k(h) q^k                                  (polynomial in h)
    D(h)        = prod_{(r,s): rs <= q_order} (h - d_{r,s})         (polynomial in h)

D is z-independent; N_z is the q-series collapse at the chosen z. N_k(h), D(h)
come from HRational (polyblock).

This representation makes the pole structure explicit (zeros of D) and the
"positive in h" piece (E(z)^h) factorable from the rational part. It is the
chiral-side building block for the polynomial-in-Delta form used in the
scalar_blocks output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mpmath as mp

from .params import b_from_c, c_from_b, c_origin_of_b, _c_input_of_b
from .polyblock import H_q_series_rational
from .polynomial import Polynomial
from .coordinates import _coerce_keep_real
from .numeric_input import _integer_control, literal_number, parse_number
from .recursion import _resolve_chiral_norm, q_from_z


@dataclass
class FChiralH:
    """F(h, z; h_i; c) at a fixed z, viewed as a function of symbolic h.

        F(h) = exp_base ** h * scalar_prefactor * numerator(h) / denominator(h)

    `denominator` and `denominator_roots` are z-independent; `numerator` and
    `scalar_prefactor` and `exp_base` collapse the q-dependence at this z.
    """

    exp_base: mp.mpc
    scalar_prefactor: mp.mpc
    numerator: Polynomial
    denominator: Polynomial
    denominator_roots: list
    root_provenance: list | None = None
    #: Stored construction precision, including near-singular guards.
    working_dps: int | None = None

    def evaluate(self, h):
        h = literal_number(h)
        from .conditioning import pole_loss, report

        approx = getattr(self, "_pole_approx", None)
        if approx is None:
            import numpy as np

            approx = np.array([complex(r) for r in self.denominator_roots], dtype=complex)
            object.__setattr__(self, "_pole_approx", approx)
        report(pole_loss(h, approx, lambda i: self.denominator_roots[i]), what="F_block_chiral_h", source=h)
        with mp.workdps(max(mp.mp.dps, self.working_dps or 0)):
            value = self._evaluate(h)
        return +value

    def _evaluate(self, h):
        # Keep real h in mp.mpf so the whole evaluation stays in real
        # arithmetic when exp_base, scalar_prefactor, and the polynomial
        # coefficients are all real.
        h = _coerce_keep_real(mp.convert(parse_number(h)))
        N_h = self.numerator.evaluate(h)
        D_h = self.denominator.evaluate(h)
        return self.exp_base**h * self.scalar_prefactor * N_h / D_h


def _theta3(q):
    return mp.jtheta(3, 0, q)


def F_block_chiral_h(
    z,
    h_ext: Sequence,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
) -> FChiralH:
    """Build the symbolic-h representation of F(h, z; h_i; c) at this z.

    Pass exactly one of `b` or `c`. `chiral_norm` selects N (default 16).
    `z_shift` controls the z-power: the h-independent amplitude includes
    z^{sigma + z_shift}. Default z_shift=0 gives z^h behavior.

    Bare recursion: raises ValueError at a recognized recursion-singular c
    (`singularity.recognized_singular_locus`); use `F_block` (auto-regularized)
    or `regularization.F_block_regularized` there.
    """
    q_order = _integer_control('q_order', q_order)
    if (b is None) == (c is None):
        raise ValueError("specify exactly one of b, c")
    if len(h_ext) != 4:
        raise ValueError("h_ext must be a length-4 sequence (h1, h2, h3, h4)")
    if c is not None:
        from .singularity import snap_float_c

        source_c = snap_float_c(c)    # a float typed on a singular locus is that locus
    else:
        source_c = _c_input_of_b(b)
    source_ext = tuple(parse_number(value) for value in h_ext)
    source_z = parse_number(z)
    from .conditioning import endpoint_loss, report

    report(endpoint_loss(source_z), what="F_block_chiral_h")
    source_norm = None if chiral_norm is None else parse_number(chiral_norm)
    source_shift = None if z_shift is None else parse_number(z_shift)
    from .singularity import near_singular_compute, reject_singular_c

    reject_singular_c(source_c, b=b, q_order=q_order, h_externals=source_ext,
                      where="F_block_chiral_h")
    def build():
        # Rebuild geometry and derived b inside every precision retry. Kac
        # construction must see the source externals before fusion checks.
        b_work = b_from_c(source_c) if b is None or c_origin_of_b(b) is not None else b
        H_rat = H_q_series_rational(source_ext, b=b_work, q_order=q_order)
        z_work = _coerce_keep_real(mp.convert(source_z))
        h1, h2, h3, h4 = (_coerce_keep_real(mp.convert(value)) for value in source_ext)
        charge = c_from_b(b_work) if b is not None and c_origin_of_b(b) is None else source_c
        c_eff = _coerce_keep_real(mp.convert(charge))
        sigma = (c_eff - 1) / 24
        q = q_from_z(z_work)
        sixteen_q = 16 * q
        N = _resolve_chiral_norm(source_norm)
        z_exp = sigma + (_coerce_keep_real(mp.convert(source_shift)) if source_shift is not None else 0)
        scalar_prefactor = (
            sixteen_q ** (-sigma)
            * z_work ** z_exp
            * (1 - z_work) ** (sigma - h2 - h3)
            * _theta3(q) ** ((c_eff - 1) / 2 - 4 * (h1 + h2 + h3 + h4))
        )
        # Collapse sum_k N_k(h) * q^k by Horner, retaining the narrowest field.
        poly_field = H_rat.numerators[0].field if H_rat.numerators else mp.mpf
        numerator = Polynomial.zero(field=poly_field)
        for k in range(len(H_rat.numerators) - 1, -1, -1):
            numerator = numerator * q + H_rat.numerators[k]
        return FChiralH(
            exp_base=sixteen_q / N,
            scalar_prefactor=scalar_prefactor,
            numerator=numerator,
            denominator=H_rat.denominator,
            denominator_roots=list(H_rat.denominator_roots),
            root_provenance=list(H_rat.root_provenance or ()),
            working_dps=mp.mp.dps,
        )

    probe_h = mp.mpf("1.41421356237309504880")
    chiral, extra = near_singular_compute(
        source_c, "F_block_chiral_h", build, h_externals=source_ext, q_order=q_order,
        fingerprint=lambda obj: obj._evaluate(probe_h), return_guard=True,
        cache_context=(source_z, source_norm, source_shift, b),
    )
    chiral.working_dps = max(chiral.working_dps, mp.mp.dps + extra)
    return chiral
