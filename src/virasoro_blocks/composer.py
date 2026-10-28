"""Compose 2d blocks from chiral and antichiral halves.

The fundamental object is the ordered-(h, hbar) elementary block

    B(h, hbar; z, zbar; {h_i}, {hbar_i}; c)
        = F(h, z; {h_i}; c) * F(hbar, zbar; {hbar_i}; c)

where F is the holomorphic Virasoro block from `recursion.F_block`. Both halves
share the same central charge c (equivalently b), because b is set by the theory.
For scalar externals (h_i = hbar_i) we package the parity-even blocks

    G_{Delta, ell}(z, zbar) = [B(h, hbar; z, zbar) + B(hbar, h; z, zbar)] / (1 + delta_{ell, 0})

with h = (Delta + ell)/2, hbar = (Delta - ell)/2, for every integer ell. For scalar
externals B(h, hbar) and B(hbar, h) are related by z <-> zbar, so this is proportional
to B(h, hbar; z, zbar) + B(h, hbar; zbar, z): G_{Delta, ell} is z <-> zbar even at every
spin, as the four-point function of scalars in a parity-invariant theory is (the
Dolan-Osborn block in 2d). The z <-> zbar-odd combination B(h, hbar) - B(hbar, h)
belongs to parity-odd correlators; it is `parity_projected_B_block` with parity '-'.
At ell = 0 G is the product block F(h, z) F(h, zbar) itself (the identity G_{0,0} =
F_0 F_0 included).

Each chiral half carries the library normalization N^{-h} (`chiral_norm`, default 16; see
`recursion.F_block`), so B and G carry N^{-Delta}. The four-point function is G_{0,0} + sum p G_{Delta, ell} with
p = N^Delta lambda^2 in every channel: lambda^2 is the standard-normalization squared OPE coefficient of the scalar
at ell = 0, and of each member of the parity pair (h, hbar), (hbar, h), which G combines with equal weight,
otherwise.
"""

from __future__ import annotations

from typing import Sequence

import mpmath as mp

from ._conventions import scalar_G_spin0_weight
from .parity import parity_reversal_phase, parity_projection_is_zero
from .conditioning import conditioned_sum
from .params import b_from_c, c_from_b
from .recursion import F_block
from .numeric_input import _integer_spin, _exchange_weight, literal_number


def _resolve_b(b, c):
    if (b is None) == (c is None):
        raise ValueError("specify exactly one of b, c")
    return b if b is not None else b_from_c(c)


def _resolve_bc(b, c):
    """Resolve to (b, c) preserving the user's input. Pass at least one;
    the other is derived. If both are supplied (the chain `G_block → B_block
    → F_block` threads both for precision), they are returned verbatim --
    the chain itself guarantees consistency. Keeping the user's c exact
    means `recognized_singular_locus(c)` -- the auto-route check at the
    deepest F_block layer -- sees the user's c without the
    `c_from_b(b_from_c(c))` round-trip roundoff (~`10^{-(dps-few)}`) that
    would otherwise force the dual-family detection to lean on the
    precision-scaled tolerance in `positive_rational_b2_pq`."""
    if b is None and c is None:
        raise ValueError("specify at least one of b, c")
    if c is not None:
        from .singularity import snap_float_c

        c = snap_float_c(c)       # a float 0.7 means 7/10
    if b is None:
        with mp.workdps(2 * mp.mp.dps):    # extra digits: at large c the block is very sensitive to b
            b = b_from_c(c)
    elif c is None:
        from .params import c_origin_of_b

        origin = c_origin_of_b(b)
        if origin is None:
            c = c_from_b(b)
        else:
            c = origin
            with mp.workdps(2 * mp.mp.dps):
                b = b_from_c(c)
    return b, c


def _check_externals(h_externals: Sequence, hbar_externals: Sequence):
    if len(h_externals) != 4 or len(hbar_externals) != 4:
        raise ValueError(
            "h_externals and hbar_externals must each be length-4 sequences"
        )


def B_block(
    h,
    hbar,
    z,
    zbar,
    h_externals: Sequence,
    hbar_externals: Sequence,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    regularize: str = "auto",
    kac_limit: str | None = None,
    z_shift=None,
    zbar_shift=None,
):
    """Ordered-(h, hbar) elementary block B(h, hbar; z, zbar).

    Returns F(h, z; {h_i}; c) * F(hbar, zbar; {hbar_i}; c).

    The chiral half is computed at exchanged weight h, internal coordinate z,
    and external chiral weights h_externals; analogously for the antichiral
    half. The two halves share b (equivalently c). `chiral_norm` selects N in
    the rescaling F = N^{-h} (16 q)^h * ... (default 16). See `F_block`
    for the `regularize` and `kac_limit` parameters -- they are threaded through to
    both chiral halves so the singular-c guard fires once and dispatches both.
    `z_shift` / `zbar_shift` are the prefactor exponent shifts of the chiral /
    antichiral half (see `F_block`); in the parity-reversed term the two swap
    along with the externals they compensate.
    """
    b, c = _resolve_bc(b, c)
    _check_externals(h_externals, hbar_externals)

    F_holo = F_block(
        h, z, h_externals, b=b, c=c, q_order=q_order, chiral_norm=chiral_norm,
        regularize=regularize, kac_limit=kac_limit, z_shift=z_shift,
    )
    F_anti = F_block(
        hbar, zbar, hbar_externals, b=b, c=c, q_order=q_order, chiral_norm=chiral_norm,
        regularize=regularize, kac_limit=kac_limit, z_shift=zbar_shift,
    )
    return F_holo * F_anti


def parity_reversed_B_block(
    h,
    hbar,
    z,
    zbar,
    h_externals: Sequence,
    hbar_externals: Sequence,
    *,
    parity_external=None,
    exchanged_parity: str = "1",
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    regularize: str = "auto",
    kac_limit: str | None = None,
    z_shift=None,
    zbar_shift=None,
):
    """Parity-reversed partner of the ordered block B(h, hbar; z, zbar).

    The convention is:

        P[B(h, hbar; z, zbar; {h_i}, {hbar_i})]
            = phase * B(hbar, h; z, zbar; {hbar_i}, {h_i})

    where ``phase`` is the product of the external parity phases and an
    optional exchanged-operator parity phase. With no explicit parity metadata,
    this reduces to the chirality-swapped ordered block used already in the
    scalar G construction.
    """
    phase = parity_reversal_phase(
        parity_external, exchanged_parity=exchanged_parity
    )
    return phase * B_block(
        hbar,
        h,
        z,
        zbar,
        hbar_externals,
        h_externals,
        b=b,
        c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
        regularize=regularize, kac_limit=kac_limit,
        z_shift=zbar_shift,        # the shifts follow the externals they compensate
        zbar_shift=z_shift,
    )


def parity_projected_B_block(
    h,
    hbar,
    z,
    zbar,
    h_externals: Sequence,
    hbar_externals: Sequence,
    *,
    parity: str,
    parity_external=None,
    exchanged_parity: str = "1",
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    regularize: str = "auto",
    kac_limit: str | None = None,
    z_shift=None,
    zbar_shift=None,
):
    """Parity-projected ordered block for generic external spin.

    Defines the derived parity-eigenstate basis

        B_{parity}(h, hbar) = B(h, hbar) +/- P[B(h, hbar)]

    where the sign is selected by ``parity`` in {"+", "-"} and ``P`` is the
    shared parity-reversal convention implemented by ``parity_reversed_B_block``.
    The ordered block remains the primitive basis element; this is the derived
    parity-definite combination built on top of it.
    """
    if parity not in {"+", "-"}:
        raise ValueError("parity must be '+' or '-'")
    first = B_block(
        h,
        hbar,
        z,
        zbar,
        h_externals,
        hbar_externals,
        b=b,
        c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
        regularize=regularize, kac_limit=kac_limit,
        z_shift=z_shift,
        zbar_shift=zbar_shift,
    )
    second = parity_reversed_B_block(
        h,
        hbar,
        z,
        zbar,
        h_externals,
        hbar_externals,
        parity_external=parity_external,
        exchanged_parity=exchanged_parity,
        b=b,
        c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
        regularize=regularize, kac_limit=kac_limit,
        z_shift=z_shift,
        zbar_shift=zbar_shift,
    )
    sign = mp.mpf(1) if parity == "+" else mp.mpf(-1)
    phase = parity_reversal_phase(parity_external, exchanged_parity=exchanged_parity)
    return conditioned_sum(first, sign * second, exact_zero=parity_projection_is_zero(
        sign * phase, h, hbar, z, zbar, h_externals, hbar_externals, z_shift, zbar_shift))


def G_block(
    Delta,
    ell: int,
    z,
    zbar,
    h_externals: Sequence,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    regularize: str = "auto",
    kac_limit: str | None = None,
    z_shift=None,
    zbar_shift=None,
):
    """Parity-symmetrized (Delta, ell) block for scalar externals.

    G_{Delta, ell}(z, zbar) = [B(h, hbar; z, zbar) + B(hbar, h; z, zbar)] / (1 + delta_{ell, 0}),
    with h = (Delta + ell)/2, hbar = (Delta - ell)/2: z <-> zbar even at every spin (the z <-> zbar-odd
    combination is `parity_projected_B_block` with parity '-'). At ell = 0 this is the product block
    F(h, z) F(h, zbar), and each chiral half carries N^{-h} (`chiral_norm`; see the module docstring).

    Assumes scalar externals: the four external chiral weights `h_externals` are
    used for both the chiral and antichiral halves. The caller is responsible
    for ensuring this is the right physical setting; for spinning externals use
    `B_block` directly with distinct h/hbar external arrays.

    `ell` must be an integer. Negative `ell` is accepted: G_{Delta,-ell} equals
    G_{Delta,ell} (the sum is symmetric in h <-> hbar), so it is a redundant
    labeling of the same block, not an error.

    `regularize` controls dispatch at recursion-singular `c` (c=1, c=25, all
    minimal models including the Ising c=1/2). Default `"auto"` routes to
    `G_block_regularized` whenever `recognized_singular_locus(c)` matches,
    avoiding the ~10^88 silent-garbage output of the bare H-recursion at
    those loci. See `F_block` for the full semantics of the three modes.
    """
    b, c = _resolve_bc(b, c)
    if len(h_externals) != 4:
        raise ValueError("h_externals must be a length-4 sequence")
    ell = _integer_spin(ell)

    Delta = literal_number(Delta)
    if isinstance(Delta, complex):
        Delta = mp.mpc(Delta)
    h = _exchange_weight(Delta, ell)
    hbar = _exchange_weight(Delta, -ell)
    value = parity_projected_B_block(
        h,
        hbar,
        z,
        zbar,
        h_externals,
        h_externals,
        parity="+",
        b=b,
        c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
        regularize=regularize, kac_limit=kac_limit,
        z_shift=z_shift,
        zbar_shift=zbar_shift,
    )
    # the parity projection is B + P[B] = 2 B(h, h) at ell = 0: the spin-0 block is half of it under the default
    # "single" normalization (see _conventions)
    return value / 2 if ell == 0 and scalar_G_spin0_weight() == 1 else value
