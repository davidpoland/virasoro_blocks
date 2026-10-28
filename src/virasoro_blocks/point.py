"""Point-constraint block tables: a block frozen at a single value of its
positivity variable.

A point constraint imposes ``y . F(point) >= 0`` at one isolated dimension
(Delta) or chiral weight (h) -- e.g. an allowed island point or a known light
operator below the gap. The artifact is the full ``(m, n)`` derivative grid of
the block evaluated at that single point: **degree-0 numerators in the positivity
variable, full in the Taylor indices**, multiplied by a positive prefactor.

The grid is computed with the robust numeric block evaluator (the O(1) recursion
samplers, not the ill-conditioned symbolic ``P(Delta)/D(Delta)`` table, and not
an off-support evaluation of an order-reduced interpolation), then frozen as a
degree-0 :class:`~virasoro_blocks.polyderivs.InterpolatedDeltaTable`. The
per-basis evaluator dispatch is shared with the ordinary block path via
:func:`~virasoro_blocks.polyderivs.build_numeric_sampler` (Virasoro bases) and
:func:`~virasoro_blocks.global_blocks.build_global_sampler` (global bases), so a
point block covers exactly the same ``(basis, coordinate)`` matrix as a full one.
"""
from __future__ import annotations

import mpmath as mp

from .truncation import total_parity_phase
from .derivs import _acb_to_scalar
from .global_blocks import build_global_sampler, global_block_exp_base
from .interpolation import DampedRational
from .polynomial import Polynomial, _arb_to_mpf
from .polyderivs import InterpolatedDeltaTable, build_numeric_sampler
from .regularization import _RegularizedNumericSampler
from .serialization import serialize_deriv_tables
from .singularity import recognized_singular_locus
from .truncation import is_off_diagonal, kept_pairs

_GLOBAL_BASES = {
    "global_scalar_G",
    "global_ordered_B",
    "global_parity_projected_B",
    "global_chiral_F",
}
_CHIRAL_BASES = {"chiral_F", "global_chiral_F"}
_VIRASORO_BASES = {"scalar_G", "ordered_B", "parity_projected_B", "chiral_F"}


def _to_mp(value):
    """Coerce a sampler grid entry (mpmath, python, or flint arb/acb) to mp."""
    if isinstance(value, (mp.mpf, mp.mpc)):
        return value
    if isinstance(value, complex):
        return mp.mpc(value)
    if isinstance(value, (int, float)):
        return mp.mpf(value)
    # flint: acb exposes .imag, arb does not.
    if hasattr(value, "imag"):
        return _acb_to_scalar(value)
    return _arb_to_mpf(value)


def _assert_finite_grid(grid_by_pair, point_value):
    """Raise if any frozen block value is non-finite (a pole/overflow the
    pole-root check did not catch, e.g. on the non-regularized path)."""
    for (m, n), v in grid_by_pair.items():
        if not mp.isfinite(v):
            raise ValueError(
                f"point block is non-finite at (m, n) = ({m}, {n}) for "
                f"point_value {mp.nstr(mp.mpc(point_value), 12)}: the block "
                "likely has a genuine pole there (a degenerate dimension), so "
                "the point constraint is ill-defined."
            )


def _check_point_pole(*, basis, ell, point_value, h_externals, hbar_externals,
                      c, q_order):
    """Fail closed when the frozen point sits on a genuine block pole.

    Same checks as pointwise ``evaluate`` (``evaluation._check_virasoro_pole``
    / ``_check_global_pole``): a Kac pole at finite recursion order gives a
    huge finite grid, not inf/NaN, so :func:`_assert_finite_grid` alone lets it
    through. Scalar G also contains the weight-swapped term. Parity reversal
    instead swaps both the weights and their external lists, so its partner
    has the same weight/external-list pairings as the ordered term.
    """
    from .evaluation import _check_global_pole, _check_virasoro_pole, _mp

    chiral = basis in _CHIRAL_BASES
    if chiral:
        h = hbar = point_value
    else:
        h = (point_value + ell) / 2
        hbar = (point_value - ell) / 2
    h_ext = list(h_externals)
    hbar_ext = h_ext if hbar_externals is None else list(hbar_externals)
    checks = [(h, h_ext, "holomorphic")]
    if not chiral:
        checks.append((hbar, hbar_ext, "antiholomorphic"))
        if basis in {"scalar_G", "global_scalar_G"}:
            checks += [(hbar, h_ext, "holomorphic"), (h, hbar_ext, "antiholomorphic")]
    for weight, ext, side in checks:
        if basis in _GLOBAL_BASES:
            _check_global_pole(weight, ext, basis=basis, side=None if chiral else side)
        elif c is not None:
            _check_virasoro_pole(
                weight, ext, c=c,
                order=q_order or 1, basis=basis, side=None if chiral else side,
            )


def _point_grid(
    coordinate, *, basis, ell, point_value, h_externals, hbar_externals,
    lam, c, q_order, chiral_norm, z_shift, zbar_shift, center, center_bar,
    parity, parity_external, exchanged_parity, regularize="auto",
):
    """Robust numeric ``[m][n]`` block grid at ``point_value`` for one coordinate,
    over the kept-pairs triangle. Returns ``(grid_by_pair, m_max, n_max)``."""
    chiral = basis in _CHIRAL_BASES
    kept = kept_pairs(
        coordinate, lam, basis=basis, ell=ell, parity_projection=parity,
        parity_phase=total_parity_phase(parity_external, exchanged_parity),
        off_diagonal=is_off_diagonal(center, center_bar),
    )
    m_max = max((m for m, _n in kept), default=0)
    n_max = max((n for _m, n in kept), default=0)

    # Global blocks are exact 2F1, c-independent (no recursion singularity).
    if basis in _GLOBAL_BASES:
        sampler = build_global_sampler(
            coordinate, basis=basis, ell=ell, h_externals=h_externals,
            hbar_externals=hbar_externals, m_max=m_max, n_max=n_max,
            chiral_norm=chiral_norm, z_shift=z_shift, zbar_shift=zbar_shift,
            parity=parity, parity_external=parity_external,
            exchanged_parity=exchanged_parity, center=center, center_bar=center_bar,
        )
        sample_bucket = getattr(sampler, "sample_bucket", None)
        if sample_bucket is not None:
            tables, _is_arb = sample_bucket([point_value], list(kept))
            grid = tables[0]
        else:
            # the chiral global sampler is a plain h -> [[F^(m)]] callable
            grid = sampler(point_value)
        grid_by_pair = {(m, n): _to_mp(grid[m][n]) for (m, n) in kept}
        _assert_finite_grid(grid_by_pair, point_value)
        return grid_by_pair, m_max, n_max

    # Virasoro bases: one robust numeric-recursion sampler per central charge.
    spinning = basis in ("ordered_B", "parity_projected_B")
    projected = basis == "parity_projected_B"

    sampler_kwargs = dict(
        chiral=chiral, spinning=spinning, projected=projected,
        h_externals=h_externals, hbar_externals=hbar_externals, ell=ell,
        m_max=m_max, n_max=n_max, b=None, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift, zbar_shift=zbar_shift,
        center=center, center_bar=center_bar, parity=parity,
        parity_external=parity_external, exchanged_parity=exchanged_parity,
    )

    # Recursion-singular c (c = 1, c = 25, minimal models incl. Ising c = 1/2): the block AT the singular c, from the
    # merged-pole form of the one Laurent recursion table (_RegularizedNumericSampler); on a Kac weight where poles
    # collide the c -> c0 limit at fixed Delta (the irreducible block), and a BlockPoleError on one that keeps its pole.
    # ``regularize``: "auto" regularizes at a recognized singular c, "on" always, "off" never
    if regularize not in ("auto", "on", "off", True, False):
        raise ValueError("regularize must be 'auto', 'on' or 'off'")
    locus = None if c is None else recognized_singular_locus(c)
    use_regularizer = regularize in ("on", True) or (regularize == "auto" and locus is not None)
    if use_regularizer and c is not None:
        sampler = _RegularizedNumericSampler(coordinate=coordinate, c=c, **sampler_kwargs)
    else:
        sampler = build_numeric_sampler(coordinate, c=c, **sampler_kwargs)
    # The chiral sampler is parametrized by the weight h; the others by Delta.
    tables, _is_arb = sampler([point_value], entries=list(kept))
    grid = tables[0]

    grid_by_pair = {(m, n): _to_mp(grid[m][n]) for (m, n) in kept}
    _assert_finite_grid(grid_by_pair, point_value)
    return grid_by_pair, m_max, n_max


def point_block_table(
    coordinates, *, basis, ell, point_value, h_externals, hbar_externals=None,
    lam, c=None, q_order=None, chiral_norm=None, z_shift=None, zbar_shift=None,
    center=None, center_bar=None, parity=None, parity_external=None,
    exchanged_parity="1", regularize="auto",
):
    """Build a degree-0 :class:`SerializedDerivTable` freezing the block at
    ``point_value`` (a Delta for 2d bases, a chiral weight h for chiral bases).

    ``coordinates`` is the list of block coordinates to include (each gets its
    own frozen grid). Every ``(m, n)`` numerator is the degree-0 polynomial
    ``[block value]``, and the shift and single sample point are ``point_value``.
    The prefactor is ``base**(Delta - point_value)``, positive on the real
    positivity domain, so it does not affect the imposed inequality. Evaluating
    at ``point_value`` recovers the block's derivative grid; evaluation elsewhere
    only rescales that frozen grid and does not compute the block at a new
    dimension. For a chiral basis, replace Delta by h. The decay base also lets
    the single-node Bohr-Sommerfeld sample-point generator handle the channel.
    Scalar-G bases require matching holomorphic and antiholomorphic external
    weights and a common shift; independent inputs require an ordered or
    parity-projected basis. Matching compares the supplied exact labels.
    Chiral bases require ell=0 and have one weight list and one shift. Redundant
    antiholomorphic inputs must match their holomorphic counterparts.
    """
    from .evaluation import _label
    from .numeric_input import (
        literal_number, literal_components, _signed_half_integer, _integer_spin,
    )
    from .requests import _integer_field

    if basis not in _GLOBAL_BASES | _VIRASORO_BASES:
        raise ValueError(f"unsupported point-table basis {basis!r}")
    lam = _integer_field("lam", lam)
    if lam < 0:
        raise ValueError("lam must be >= 0")
    ell = _integer_spin(ell) if basis in {"scalar_G", "global_scalar_G"} else _signed_half_integer(ell)
    # Reuse the same weights for pole checks and every requested coordinate;
    # a one-shot iterable must not be consumed by the precheck alone.
    h_externals = [literal_number(_label(value)) for value in h_externals]
    hbar_externals = h_externals if hbar_externals is None else [literal_number(_label(value)) for value in hbar_externals]
    if len(h_externals) != 4 or len(hbar_externals) != 4:
        raise ValueError("h_externals and hbar_externals must each have length 4")
    point_value = literal_number(_label(point_value))
    c = None if c is None else literal_number(_label(c))
    z_shift = None if z_shift is None else literal_number(_label(z_shift))
    zbar_shift = None if zbar_shift is None else literal_number(_label(zbar_shift))
    if basis in _CHIRAL_BASES:
        if ell != 0:
            raise ValueError("chiral point tables require ell=0")
        if any(literal_components(h) != literal_components(hbar)
               for h, hbar in zip(h_externals, hbar_externals)):
            raise ValueError("chiral_F has one external-weight list: hbar_externals must match h_externals")
        if zbar_shift is not None and literal_components(zbar_shift) != literal_components(z_shift or 0):
            raise ValueError("chiral_F carries one shift, z_shift: zbar_shift must match it")
    elif basis in {"scalar_G", "global_scalar_G"}:
        if any(literal_components(h) != literal_components(hbar)
               for h, hbar in zip(h_externals, hbar_externals)):
            raise ValueError("scalar_G requires scalar external weights: hbar_externals must match h_externals")
        if zbar_shift is not None and literal_components(zbar_shift) != literal_components(z_shift or 0):
            raise ValueError("scalar_G carries one shift: zbar_shift must equal z_shift; "
                             "independent shifts require ordered_B or parity_projected_B")
    base = global_block_exp_base(chiral_norm, center=center, center_bar=center_bar)
    _check_point_pole(
        basis=basis, ell=ell, point_value=point_value, h_externals=h_externals,
        hbar_externals=hbar_externals, c=c, q_order=q_order,
    )
    tables_by_coordinate = {}
    for coordinate in coordinates:
        grid_by_pair, m_max, n_max = _point_grid(
            coordinate, basis=basis, ell=ell, point_value=point_value,
            h_externals=h_externals, hbar_externals=hbar_externals, lam=lam,
            c=c, q_order=q_order, chiral_norm=chiral_norm, z_shift=z_shift,
            zbar_shift=zbar_shift, center=center, center_bar=center_bar,
            parity=parity, parity_external=parity_external,
            exchanged_parity=exchanged_parity, regularize=regularize,
        )
        # Always lam + 1 rows, as the ordinary table path writes: an odd-parity
        # folded grid keeps m <= lam - 1 only, and the writers infer lambda
        # (and hence the kept pairs) from the grid shape.
        m_max = max(m_max, int(lam))
        numerators = [
            [
                Polynomial([grid_by_pair.get((m, n), mp.mpf(0))])
                for n in range(n_max + 1)
            ]
            for m in range(m_max + 1)
        ]
        nodes = [
            [(point_value,) for _n in range(n_max + 1)]
            for _m in range(m_max + 1)
        ]
        tables_by_coordinate[coordinate] = InterpolatedDeltaTable(
            prefactor=DampedRational(constant=1, poles=(), base=base),
            shift=point_value, sample_points=(point_value,),
            numerators=numerators, numerator_sample_points=nodes,
        )
    return serialize_deriv_tables(tables_by_coordinate)
