"""Regularization of the recursion at singular c.

The Zamolodchikov h-recursion (recursion.py) has spurious `1/(c - c0)` blowups
at c = 1, c = 25, the minimal models and the rational-b^2 loci above 25: the
residue factor `A^c_{r,s}` has b-poles, and the recursion's Kac weights `d_{r,s}`
collide.  The physical block is finite there, and is read exactly off one Laurent
expansion of the recursion in t (`params.laurent_t_power`).

At a Kac weight where poles collide the block is discontinuous in (h, c): the limit
in h at c = c0 and the c -> c0 limit at fixed h differ by a multiple of the
submodule block.  On a Kac weight every value path returns the block of the
irreducible module L(h) (`recursion.eval_merged_H`, `recursion.H_q_series_kac_irreducible`):
the block whose OPE coefficients a compact CFT at c0 has (the free-boson vertex
correlator peels into its exact Fock couplings with it); the merged form,
continuous in h, everywhere else -- which is also the function of Delta the
tables carry (the Fock-module block on a Kac weight: the free-boson correlator's
h = 4 coefficient is 25% off with it).  ``kac_limit="c_limit"`` selects instead
the c -> c0 limit at fixed h, which equals the irreducible block except where
Kac poles of one level merge (every Kac weight at c = 1 and 25): there it keeps
finite admixtures of the null submodule (at c = 1, h = m^2/4, of L((m+4)^2/4)
and L((m+8)^2/4), from level 2(m + 2) on).

`regularized_numeric_grids` (epsilon-extrapolation in c at fixed Delta) and the
probe-pair scorer stay as the references the merged forms are tested against.

At a singular c the block values (`_RegularizedNumericSampler`) and the pole scores are read off ONE Laurent-in-t
H-table (t = sqrt(c - c0), the block is its t^0 part), as merged poles: H_k(h) = sum over merged locations d0 and orders
a of M[k, (d0, a)] / (h - d0)^a (`recursion.build_merged_H_context`).  The generic chiral evaluator built at c0 under
`singularity.merged_singular_evaluators` takes its H from that form (its geometry is analytic at c0), so the sampler is
the ordinary numeric sampler at c0 and the scorer (`scorer_method="merged"`, the default) the probe pair's row assembly
run once on those evaluators.  Both replaced extrapolations in c (five probe builds per sampled grid; the full t-series
row assembly) and agree with them to the working precision for every basis and coordinate.  `scorer_method="laurent"`
(the t-series assembly) and `"pair"` (a symmetric probe pair c0 +/- delta) remain as references.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Sequence

import mpmath as mp

from .conditioning import quiet as _quiet_conditioning
from .numeric_input import _integer_control
from .precision import taylor_coefficients
from .singularity import bare_recursion_probes

from .composer import B_block, G_block
from .coordinates import q_symmetric, y_symmetric, z_of_q, z_of_y, z_symmetric
from .params import b_from_c, c_from_b, _c_input_of_b, d_rs, imaginary_b_locus, laurent_t_sq_sign
from .polyderivs import (
    DeltaDerivTable,
    _PrincipalPartScoring,
    build_numeric_sampler,
)
from .polynomial import (
    Polynomial,
    _acb_to_mpc,
    _arb_to_mpf,
    _mpf_to_acb,
    _mpf_to_arb,
    real_factor_coeffs_from_conjugate_roots,
)
from .recursion import (
    H_q_series,
    H_q_series_laurent,
    H_q_series_laurent_eval,
    H_q_series_laurent_precompute,
    _scalar_cache_key,
    block_prefactor,
    q_from_z,
)
from . import tseries as _ts
from .tseries import (
    LaurentSeries,
    StackedURows,
    numeric_linear_map,
    stacked_weighted_sum,
    u_convolve,
    u_numeric_convolve_sum,
)


def _fixed_h_block_evaluator_regularized(h, h_ext: Sequence, *, b=None, c=None, q_order: int, chiral_norm=None,
                                         z_shift=None, working_precision: bool = False):
    """Return a callable ``z -> F_block_regularized(h, z, ...)``: the H coefficients (:func:`H_q_series_regularized`)
    are formed once, and each ``z`` costs one Horner sum (the identity vector walks ``taylor_coefficients`` in z at
    the fixed ``h = 0``).  ``working_precision`` reads them at the working precision from the one merged context a
    singular-c table build shares (:func:`_merged_H_coefficients`; its digits are the table's to verify)."""
    q_order = _integer_control('q_order', q_order)
    c_val = _resolve_c(b, c)
    if working_precision:
        from .singularity import exact_locus_c

        c0 = exact_locus_c(c_val, q_order=q_order)
        H_coeffs = (H_q_series(h, h_ext, c=c_val, q_order=q_order) if c0 is None
                    else _merged_H_coefficients(h, h_ext, c0=c0, q_order=q_order))
    else:
        H_coeffs = H_q_series_regularized(h, h_ext, c=c_val, q_order=q_order)
    coeffs_real = all(isinstance(coeff, mp.mpf) for coeff in H_coeffs)

    def evaluate(z):
        q = q_from_z(z)
        H_val = mp.mpf(0) if coeffs_real else mp.mpc(0)
        for k in range(len(H_coeffs) - 1, -1, -1):
            H_val = H_val * q + H_coeffs[k]
        return block_prefactor(h, z, h_ext, c=c_val, chiral_norm=chiral_norm, z_shift=z_shift) * H_val

    return evaluate


def _lagrange_weights_at_zero(epsilons: Sequence) -> list:
    """Precompute Lagrange basis weights for extrapolation to x=0.

    weight[k] = prod_{j!=k} (-eps[j]) / (eps[k] - eps[j])

    These depend only on the epsilon sequence, not on the values, so they
    can be reused across all n_flat entries in the regularization loop.
    """
    n = len(epsilons)
    eps = [mp.mpc(e) for e in epsilons]
    weights = []
    for k in range(n):
        w = mp.mpc(1)
        for j in range(n):
            if j == k:
                continue
            w *= (-eps[j]) / (eps[k] - eps[j])
        weights.append(w)
    return weights


def _extrapolate_batch(weights: list, flat_columns) -> list:
    """Extrapolate many value-vectors using precomputed Lagrange weights.

    `flat_columns[j]` is the flattened scalar vector from probe j.
    Returns a list of extrapolated scalars, one per entry in the flat vector.
    """
    n_points = len(weights)
    n_flat = len(flat_columns[0])
    result = [mp.mpc(0)] * n_flat
    for k in range(n_points):
        wk = weights[k]
        col = flat_columns[k]
        for i in range(n_flat):
            result[i] += col[i] * wk
    return result


def _flint_batch_with_reduced(full_weights: list, reduced_weights: list, flat_columns):
    """Extrapolate full and reduced-point estimates as one flint matmul each.

    ``full_weights`` are for all probes; ``reduced_weights`` are for probes
    1..n-1 (the largest-epsilon probe dropped) as used by `_converged`.
    Returns ``(full, reduced)`` as mpmath lists.
    """
    from flint import acb_mat, ctx as flint_ctx

    n_points = len(full_weights)
    n_flat = len(flat_columns[0])
    if n_points == 0 or n_flat == 0:
        return [mp.mpc(0)] * n_flat, [mp.mpc(0)] * n_flat

    # Keep Flint precision consistent with existing acb usage in this codebase.
    flint_ctx.prec = int(mp.mp.dps * 3.33) + 20

    cols = [[_mpf_to_acb(v) for v in flat_columns[k]] for k in range(n_points)]
    V = acb_mat(cols)  # n_points x n_flat
    full_row = acb_mat([[ _mpf_to_acb(w) for w in full_weights ]])
    reduced_row = acb_mat([[ _mpf_to_acb(w) for w in reduced_weights ]])
    full_mat = full_row * V
    if n_points >= 2:
        reduced_mat = reduced_row * acb_mat(cols[1:])
    else:
        reduced_mat = acb_mat([[ _mpf_to_acb(0) for _ in range(n_flat) ]])

    full = [_acb_to_mpc(full_mat[0, i]) for i in range(n_flat)]
    reduced = [_acb_to_mpc(reduced_mat[0, i]) for i in range(n_flat)]
    return full, reduced


def _regularization_epsilons(n_points: int, base_dps: int, scale=1):
    """Geometric sequence epsilon_0, epsilon_0/2, ... of length n_points.

    epsilon_0 = scale * 10**-(base_dps // 2). With this choice the extrapolation error
    ~ (epsilon_0 / rho)**n_points sits at 10**-(n_points * base_dps / 2), well below
    the 10**-base_dps target for n_points >= 3, where rho is the radius in eps of the
    probe values: O(1), unless an exchanged weight sits near a Kac weight, where a
    ``scale < 1`` keeps epsilon_0 << rho.
    The sequence depends only on `base_dps` (and the scale), not on the working
    precision, so it is stable across retries.
    """
    n_points = _integer_control('n_points', n_points, 2)
    epsilon0 = mp.mpf(10) ** (-(base_dps // 2)) * scale
    return [epsilon0 / mp.mpf(2) ** k for k in range(n_points)]


def _resolve_c(b, c):
    """Resolve to c. Pass at least one of b, c; if both are given (the
    composer chain threads both for precision), prefer the user-provided
    c verbatim."""
    if b is None and c is None:
        raise ValueError("specify at least one of b, c")
    if c is None:
        return _c_input_of_b(b)
    from .singularity import snap_float_c
    from .numeric_input import _explicit_nonreal_input

    source = snap_float_c(c)
    return source if _explicit_nonreal_input(source) else mp.mpc(mp.convert(source))


def _agrees(value, other, digits, *, floor=0) -> bool:
    """``|value - other| <= 10^-digits max(|value|, floor)`` (a nan on either side fails)."""
    diff = abs(value - other)
    return diff <= mp.mpf(10) ** (-digits) * max(abs(value), floor) if mp.isfinite(diff) else False


def _extrapolation_accepted(value, reduced, previous, base_dps: int, *, floor=0) -> bool:
    """The full-vs-reduced gate of :func:`_converged` on precomputed estimates.

    The difference is the reduced estimate's truncation error, ``O((epsilon_0 / rho)^(n_points - 1))``, which
    :func:`_regularization_epsilons` puts at about ``10^-base_dps`` and which overstates the full estimate's error by
    ``rho / epsilon_0``, plus probe noise, which the full estimate carries at about the same size.  So an estimate
    whose reduced twin agrees to ``base_dps - 2`` digits is accepted; one that agrees only to ``base_dps - 10`` (a
    ``10^-(base_dps - 10)`` gate alone returned up to 10 digits of noise: 21 of 30 for an Ising h = 20.25 block at
    q_order = 100) is accepted when it also agrees to ``base_dps - 2`` digits with ``previous``, the estimate at half
    the working precision: then the difference is truncation (it does not move with the digits carried), not noise.
    """
    if _agrees(value, reduced, base_dps - 2, floor=floor):
        return True
    return (previous is not None and _agrees(value, reduced, base_dps - 10, floor=floor)
            and _agrees(value, previous, base_dps - 2, floor=floor))





def _probe_resolution_dps(c_val, n_points: int, base_dps: int, eps_base: int, *, squared: bool = True,
                          scale_digits: float = 0) -> int:
    """Digits needed to evaluate at the smallest probe offset of :func:`_regularization_epsilons` (``c + t**2`` when
    ``squared``, else ``c + eps``; offsets shrunk by ``10**-scale_digits``) keeping ``base_dps`` digits of the offset
    itself, plus a 10-digit guard."""
    offset_digits = (eps_base // 2 + (n_points - 1) * math.log10(2) + scale_digits) * (2 if squared else 1)
    magnitude = abs(c_val)
    native = float(magnitude)
    scale = (math.log10(max(native, 1.0)) if math.isfinite(native)
             else float(mp.log10(max(magnitude, 1))))
    return int(math.ceil(base_dps + offset_digits + scale)) + 10


def _resolve_work_dps(base_dps, work_dps, max_work_dps, *, c=None, n_points=None, scale=1):
    """``(work_dps, max_work_dps)`` of an adaptive epsilon loop.  With ``c`` and ``n_points`` (linear ``c + eps``
    probes, shrunk by ``scale``) the starting precision is raised to :func:`_probe_resolution_dps`: below it the probes
    round onto the singular c and agree with each other, so the convergence gate passes on garbage.  Shrunk probes
    (an exchanged weight near a Kac weight) also raise the default ceiling to twice that floor."""
    scale_digits = 0.0 if scale == 1 else max(0.0, -float(mp.log10(scale)))
    default_ceiling = max_work_dps is None
    if work_dps is None:
        work_dps = 3 * base_dps
    work_dps = _integer_control('work_dps', work_dps, base_dps)
    if max_work_dps is None:
        max_work_dps = 16 * base_dps
    max_work_dps = _integer_control('max_work_dps', max_work_dps, work_dps)
    if c is not None and n_points is not None:
        n_points = _integer_control('n_points', n_points, 2)
        floor = _probe_resolution_dps(c, n_points, base_dps, base_dps, squared=False, scale_digits=scale_digits)
        if default_ceiling:
            max_work_dps = max(max_work_dps, 2 * floor)
        if floor > max_work_dps:
            raise ValueError(f"insufficient work_dps: max_work_dps = {max_work_dps} cannot resolve the epsilon probes "
                             f"(need >= {floor})")
        work_dps = max(work_dps, floor)
    return work_dps, max_work_dps


def _grid_entry_to_mpc(value):
    """Coerce a numeric-sampler grid entry (mpmath or flint arb/acb) to mp."""
    if isinstance(value, (mp.mpf, mp.mpc)):
        return value
    if isinstance(value, complex):
        return mp.mpc(value)
    if isinstance(value, (int, float)):
        return mp.mpf(value)
    from .derivs import _acb_to_scalar
    from .polynomial import _arb_to_mpf
    # flint: acb exposes .imag, arb does not.
    return _acb_to_scalar(value) if hasattr(value, "imag") else _arb_to_mpf(value)


def regularized_numeric_grids(
    build_sampler,
    deltas,
    *,
    c,
    entries=None,
    n_points: int = 5,
    work_dps: int | None = None,
    max_work_dps: int | None = None,
):
    """Regularized block ``[m][n]`` grids at a recursion-singular ``c``, via
    epsilon-extrapolation of the *numeric-recursion* grids.

    The value-space analogue of :func:`regularize_delta_table`: it never forms
    the ill-conditioned symbolic ``P(Delta)/D(Delta)`` table and runs no pole
    strip, so it is immune to the large-Kac-pole deflation hazard. ``build_sampler
    (c_val)`` returns a numeric sampler (see
    :func:`~virasoro_blocks.polyderivs.build_numeric_sampler`) at central charge
    ``c_val``; this evaluates it at the probe charges ``c + eps_k`` (the block
    value is analytic in ``c`` at a fixed regular ``Delta`` near a minimal model,
    so the same linear extrapolation the regularized H-series uses applies) and
    Lagrange-extrapolates each grid entry to ``eps = 0``.

    Removable poles need no special handling -- the grid value is finite at every
    probe and extrapolates smoothly. A genuine pole (the block truly diverges at
    this ``(c, Delta)``) makes the probe grids blow up, so the consistency check
    fails and a ``ValueError`` is raised rather than emitting garbage.

    Every ``delta`` is extrapolated at fixed ``Delta`` (the c -> c0 limit at fixed weight).  It is the reference
    the merged sampler is tested against; probes near a Kac weight need offsets below the distance to it.

    Returns a list of ``[m][n]`` mpmath grids, one per entry of ``deltas``.
    ``entries`` (a list of ``(m, n)`` pairs) restricts which entries the sampler
    builds and which are extrapolated; the rest are left zero.
    """
    n_points = _integer_control('n_points', n_points, 2)
    c_val = _resolve_c(None, c)
    base_dps = mp.mp.dps
    deltas = list(deltas)
    kwargs = dict(c_val=c_val, base_dps=base_dps, entries=entries, n_points=n_points,
                  work_dps=work_dps, max_work_dps=max_work_dps)
    return _regularized_numeric_grids_at(build_sampler, [(d, 0) for d in deltas], scale=1, **kwargs)


def _regularized_numeric_grids_at(build_sampler, points, *, c_val, base_dps, entries, n_points, work_dps,
                                  max_work_dps, scale):
    """The epsilon-extrapolated grids of :func:`regularized_numeric_grids` at the sampler parameters
    ``delta + shift`` of ``points`` (formed at the working precision), probing at offsets shrunk by ``scale``."""
    work_dps, max_work_dps = _resolve_work_dps(base_dps, work_dps, max_work_dps, c=c_val, n_points=n_points,
                                               scale=scale)
    entry_pairs = list(entries) if entries is not None else None
    previous = {}           # delta index -> the full estimates of the last attempt (see _extrapolation_accepted)
    out = [None] * len(points)
    pending = list(range(len(points)))      # a retry recomputes only the grids that failed the gate

    while True:
        with mp.workdps(work_dps), bare_recursion_probes():
            deltas = [mp.convert(d) + shift if shift else d for d, shift in (points[i] for i in pending)]
            epsilons = _regularization_epsilons(n_points, base_dps, scale)
            full_weights = _lagrange_weights_at_zero(epsilons)
            reduced_weights = (
                _lagrange_weights_at_zero(epsilons[1:]) if len(epsilons) >= 3 else None
            )
            probe_grids = []  # probe_grids[k][j] = [m][n] grid for deltas[j]
            for eps in epsilons:
                sampler = build_sampler(c_val + eps)
                tables, _is_arb = sampler(deltas, entries=entries)
                probe_grids.append(tables)

            failed = []
            full_pairs = None
            for j, i in enumerate(pending):
                grid0 = probe_grids[0][j]
                m_dim = len(grid0)
                n_dim = len(grid0[0]) if grid0 else 0
                if entry_pairs is not None:
                    pairs = entry_pairs
                else:
                    if full_pairs is None:
                        full_pairs = [(m, n) for m in range(m_dim) for n in range(n_dim)]
                    pairs = full_pairs
                res = [[mp.mpf(0)] * n_dim for _ in range(m_dim)]
                flat_cols = []
                for k in range(n_points):
                    gridk = probe_grids[k][j]
                    flat_cols.append([
                        _grid_entry_to_mpc(gridk[m][n]) for (m, n) in pairs
                    ])

                if reduced_weights is not None:
                    coeffs, reduced = _flint_batch_with_reduced(
                        full_weights, reduced_weights, flat_cols
                    )
                else:
                    coeffs = _extrapolate_batch(full_weights, flat_cols)
                if not all(mp.isfinite(coeff) for coeff in coeffs):
                    raise ValueError("regularized_numeric_grids: a probe grid is not finite (nan or inf)")
                if reduced_weights is not None:
                    # every entry is held to the scale of its whole grid (an absolute floor of 1 passed noise in
                    # every grid below 1: a large-Delta block)
                    floor = max((abs(coeff) for coeff in coeffs), default=0)
                    last = previous.get(i)
                    previous[i] = coeffs
                    if not all(_extrapolation_accepted(coeff, reduced[idx], None if last is None else last[idx],
                                                       base_dps, floor=floor)
                               for idx, coeff in enumerate(coeffs)):
                        failed.append(i)
                        continue

                for (m, n), coeff in zip(pairs, coeffs):
                    res[m][n] = +coeff
                out[i] = res
        if not failed:
            return out
        if work_dps >= max_work_dps:
            raise ValueError(
                "regularized_numeric_grids: epsilon-extrapolation did not "
                f"converge even at work_dps = {work_dps}. The block likely has a "
                "genuine pole at this (c, Delta) -- a degenerate Kac dimension "
                "whose pole is not removable for these external dimensions -- so "
                "the value is ill-defined; otherwise max_work_dps is too low."
            )
        pending = failed
        work_dps = min(2 * work_dps, max_work_dps)


class _MergedScorerUnavailable(Exception):
    """Internal signal: the merged-Laurent contribution scorer cannot build its
    chiral Kac data at a probe c (unsupported evaluator), so the
    caller defers to the contour scorer."""


def _delta_frame_half(principal, regular, n_terms, jac=None):
    """Map an mpmath chiral Laurent ``(principal, regular)`` from the chiral
    weight frame into the ``Delta`` frame.

    ``jac`` is the chiral-to-Delta scale ``(h - h0) / (Delta - loc)``: for the
    two-sided ``G`` / ``B`` block ``Delta = 2 h -/+ ell`` so ``h - h0 =
    (Delta - loc)/2`` (``jac = 1/2``, the default); for a *chiral* ``F`` table
    ``Delta`` is ``h`` itself, so ``jac = 1`` (the frame is the identity). With
    ``1/(h-h0)^p = jac^{-p}/(Delta-loc)^p`` and ``(h-h0)^q = jac^q
    (Delta-loc)^q``, returns ``(alpha, beta)`` with ``alpha[p-1]`` the
    coefficient of ``1/(Delta-loc)^p`` and ``beta[q]`` of ``(Delta-loc)^q``
    (length-``n_terms`` rows). The Delta-frame scale tables are sized from the
    actual Laurent depth (a chiral half can collide to a higher order than the
    block multiplicity; the scorer has no order gate and scores it as it is)."""
    if jac is None:
        jac = mp.mpf(1) / 2
    inv = mp.mpf(1) / jac
    inv_pows = [inv ** p for p in range(len(principal) + 1)]
    jac_pows = [jac ** q for q in range(max(1, len(regular)))]
    alpha = [
        [inv_pows[p + 1] * principal[p][i] for i in range(n_terms)]
        for p in range(len(principal))
    ]
    beta = [
        [jac_pows[q] * regular[q][i] for i in range(n_terms)]
        for q in range(len(regular))
    ]
    return alpha, beta


def _assemble_cluster_rc(half_h, half_hbar, groups, ell, Mmax, entry_pairs,
                         sigma, n_terms, *, half_h_n=None, half_hbar_n=None, zero=None):
    """Assemble per-cluster flattened block RC rows from t=0 chiral half data.

    Chiral-confluence-first reconstruction of the merged order-``M`` block
    Laurent at each Delta-cluster ``loc``. Each block pole at ``Delta = loc``
    factorizes as ``F_h(h) * F_hbar(hbar)`` with the holomorphic/antiholomorphic
    weights ``h = (loc + ell)/2`` and ``hbar = (loc - ell)/2``. ``half_h(hval)``
    / ``half_hbar(hval)`` return the Delta-frame chiral Laurent ``(alpha, beta)``
    of the h-side / hbar-side half respectively (the two sides may carry
    independent external weights for spinning ``B`` / parity-projected ``B``;
    for scalar ``G`` / identical externals the two callables are the same), each
    already evaluated at the singular ``c`` (its confluent principal part is
    finite of the genuine collision order, including order-3 collisions at
    ``c = 28``); we convolve the two halves' Laurent series to obtain the block
    coefficients ``C[1..Mmax]`` of ``1/(Delta-loc)^b`` directly. One-sided poles
    (the other half regular) are handled through that half's Taylor
    coefficients.

    The block entry ``(m, n)`` is the ordered product ``F_h[m] * F_hbar[n]``
    plus ``sigma`` times the swapped product ``F_hbar[m] * F_h[n]``; ``sigma``
    is ``+1`` for scalar ``G`` at every ``ell != 0`` (``0`` for the public spin-0 block; see
    :func:`~virasoro_blocks._conventions.scalar_G_swap_coefficient`), ``0`` for ordered ``B`` (no swap term),
    and the parity-projection ``swapped_scale`` for parity-projected ``B``.

    For *off-diagonal* blocks (distinct holomorphic / antiholomorphic recenter
    points) the ``m``-axis and ``n``-axis carry different chiral evaluators:
    ``half_h`` / ``half_hbar`` are the ``m``-axis (recenter ``center``) h-/hbar
    halves, while ``half_h_n`` / ``half_hbar_n`` are the ``n``-axis (recenter
    ``center_bar``) halves. They default to the ``m``-axis callables, recovering
    the diagonal block where both axes share one recenter point.
    """
    if half_h_n is None:
        half_h_n = half_h
    if half_hbar_n is None:
        half_hbar_n = half_hbar
    if zero is None:
        zero = mp.mpc(0)
    if isinstance(zero, mp.mpc):
        sig = mp.mpc(sigma)
    else:
        sig = _mpf_to_acb(mp.mpc(sigma))
    swap = sigma != 0
    def conv(X, XB, Y, YB, i, j, b):
        """Coefficient of ``1/(Delta-loc)^b`` in ``F_X[i] F_Y[j]``: principal x principal (orders adding to ``b``)
        plus principal x regular on either side (order ``p`` against Taylor order ``p - b``)."""
        total = zero
        nX, nY, nXB, nYB = len(X), len(Y), len(XB), len(YB)
        for p in range(max(1, b - nY), min(b - 1, nX) + 1):
            total = total + X[p - 1][i] * Y[b - p - 1][j]
        for p in range(b, min(Mmax, nX, b + nYB - 1) + 1):
            total = total + X[p - 1][i] * YB[p - b][j]
        for p in range(b, min(Mmax, nY, b + nXB - 1) + 1):
            total = total + Y[p - 1][j] * XB[p - b][i]
        return total

    def cluster_frame(loc):
        """The four half lookups for one cluster location."""
        h0 = (loc + ell) / 2
        hbar0 = (loc - ell) / 2
        diag_pt = ell == 0          # h0 - hbar0 = ell exactly
        Apm, Bpm = half_h(h0)
        if half_h is half_hbar and diag_pt:
            Amm, Bmm = Apm, Bpm
        else:
            Amm, Bmm = half_hbar(hbar0)
        if half_h_n is half_h:
            Anp, Bnp = Apm, Bpm
        else:
            Anp, Bnp = half_h_n(h0)
        if half_hbar_n is half_hbar:
            Anm, Bnm = Amm, Bmm
        elif half_h_n is half_hbar_n and diag_pt:
            Anm, Bnm = Anp, Bnp
        else:
            Anm, Bnm = half_hbar_n(hbar0)
        return Apm, Bpm, Amm, Bmm, Anp, Bnp, Anm, Bnm

    one_plus_sig = 1 + sig
    rows = []
    for loc, idx in groups:
        Apm, Bpm, Amm, Bmm, Anp, Bnp, Anm, Bnm = cluster_frame(mp.mpc(loc))
        # the swapped product reads the same half data as the main one (ell = 0 on a shared evaluator): it is the
        # main term again, so the entry is (1 + sigma) times it
        self_swap = swap and Amm is Apm and Bmm is Bpm and Anp is Anm and Bnp is Bnm
        # ... and then symmetric in (m, n) as well, when both axes share the half (Anm is Apm): a mirrored entry
        # reuses its partner
        mirror = {} if self_swap and Anm is Apm and Bnm is Bpm else None
        row = []
        for (m, n) in entry_pairs:
            key = (min(m, n), max(m, n))
            if mirror is not None and key in mirror:
                row.extend(mirror[key])
                continue
            block = []
            for b in range(1, Mmax + 1):
                # main F_h_m[m] * F_hbar_n[n]; swapped F_hbar_m[m] * F_h_n[n]
                acc = conv(Apm, Bpm, Anm, Bnm, m, n, b)
                if self_swap:
                    acc = acc * one_plus_sig
                elif swap:
                    acc = acc + sig * conv(Amm, Bmm, Anp, Bnp, m, n, b)
                block.append(acc)
            if mirror is not None:
                mirror[key] = block
            row.extend(block)
        rows.append(row)
    return rows


def _kac_limits(ev, b0):
    """``[(index, weight)]``: each non-vanishing recursion residue of a probe evaluator with the exact limit of its
    Kac weight at the singular ``b0`` (``d_{r,s}(b0)`` from the pair's labels).  Clusters are read off these limits,
    never off distances between probe weights (whose collision spread depends on the probe offset)."""
    labels = getattr(ev, "kac_labels", None)
    labels = None if labels is None else labels()
    if labels is None or getattr(ev, "pole_laurent", None) is None:
        raise _MergedScorerUnavailable
    return [(idx, mp.mpc(d_rs(r, s, b0))) for idx, r, s in labels]


def _cluster_members(limits, hv, match_digits):
    """Indices whose limit weight is ``hv`` to ``match_digits`` digits (relative to ``max(1, |hv|)``).

    Distinct Kac limits of a rational ``b0**2`` are algebraic numbers a finite distance apart, and coincident ones
    agree to the working precision, so half the working precision separates them with room on both sides.  The float
    test only rejects pairs whose distance exceeds float rounding by far; every candidate is confirmed exactly."""
    hv = mp.mpc(hv)
    scale = max(mp.mpf(1), abs(hv))
    tol = scale * mp.mpf(10) ** (-int(match_digits))
    hv_f = complex(hv)
    reject = 2 * float(tol) + 1e-12 * float(scale)  # float rounding of O(scale) limits is ~1e-16 scale
    limits = limits if isinstance(limits, (list, tuple)) else tuple(limits)
    if math.isfinite(reject) and math.isfinite(hv_f.real) and math.isfinite(hv_f.imag):
        try:
            return [idx for idx, lim in limits if abs(complex(lim) - hv_f) <= reject and abs(lim - hv) <= tol]
        except OverflowError:
            pass
    # Coarse rejection is optional: neither an unavailable float bound nor
    # an overflowing complex norm may discard an exact Kac match.
    return [idx for idx, lim in limits if abs(lim - hv) <= tol]


def _make_half_lookup(ev, Mmax, n_terms, jac=None, *, b0, match_digits):
    """Cached Delta-frame chiral half data of a probe evaluator, as flint ``acb`` rows.

    At a cluster location ``hv`` the half's confluent Laurent is read off the probe evaluator
    (``pole_laurent`` summing the residues of the Kac indices whose limit at ``b0`` is ``hv``,
    :func:`_cluster_members`, into a principal part of the collision order) and mapped into the Delta frame.  The
    rows stay balls, so their radii carry the cancellation of the colliding residues.  ``jac`` is the chiral-to-Delta
    scale (``1/2`` for two-sided ``G`` / ``B``, ``1`` for a chiral ``F`` table).  Raises
    :class:`_MergedScorerUnavailable` if the evaluator has no Kac data at this probe ``c``.
    """
    from virasoro_blocks.polyderivs import _mpc_cache_key

    merged_order = getattr(ev, "merged_pole_order", None)
    merged = merged_order is not None and merged_order(0) is not None
    # an evaluator AT the singular c (merged poles): the principal part at hv is exactly its merged pole there
    limits = None if merged else _kac_limits(ev, b0)
    pole_laurent = ev.pole_laurent
    jac_a = _mpf_to_acb(mp.mpf(1) / 2 if jac is None else mp.mpf(jac))
    inv_a = 1 / jac_a
    regular_order = max(0, Mmax - 1)
    cache = {}

    def half_lookup(hv):
        key = _mpc_cache_key(hv)
        hit = cache.get(key)
        if hit is not None:
            return hit
        if merged:
            order = merged_order(hv)
            pl = pole_laurent(hv, principal_order=max(1, order), regular_order=regular_order,
                              allow_regular_point=True)
            if pl is not None and order == 0:
                pl = ([], pl[1])        # a regular point: no principal part (as the probe lookup's members=[])
        else:
            members = _cluster_members(limits, hv, match_digits)
            if members:
                pl = pole_laurent(hv, principal_order=len(members), regular_order=regular_order,
                                  matched_indices=members)
            else:
                pl = pole_laurent(hv, principal_order=1, regular_order=regular_order, allow_regular_point=True)
        if pl is None:
            raise _MergedScorerUnavailable
        principal, regular = pl
        alpha = [[inv_a ** (p + 1) * principal[p][i] for i in range(n_terms)] for p in range(len(principal))]
        beta = [[jac_a ** q * regular[q][i] for i in range(n_terms)] for q in range(len(regular))]
        out = (alpha, beta)
        cache[key] = out
        return out

    return half_lookup


def _probe_halves(ev_h, ev_hbar, ev_h_n, ev_hbar_n, Mmax, n_terms, lookup):
    """The four half lookups of a two-sided block at one probe, sharing one lookup per distinct evaluator."""
    made = {}

    def half(ev):
        if ev is None:
            return None
        if id(ev) not in made:
            made[id(ev)] = lookup(ev)
        return made[id(ev)]

    return half(ev_h), half(ev_hbar), half(ev_h_n), half(ev_hbar_n)


def _optionb_persp_rows_at_probe(ev_h, ev_hbar, *, groups, ell, Mmax,
                                 entry_pairs, sigma, n_terms, b0, match_digits,
                                 ev_h_n=None, ev_hbar_n=None):
    """Per-cluster flattened block RC rows for ONE probe ``c``, as flint ``acb`` balls.

    ``ev_h`` / ``ev_hbar`` are the chiral evaluators of the holomorphic /
    antiholomorphic halves (the same object for scalar ``G`` / identical
    externals; distinct evaluators carry independent externals for spinning
    ``B`` / parity-projected ``B``). The singular-c scorer averages these rows over
    the symmetric probe pair (:func:`_certified_probe_pair`); the *assembled block
    rows* are the object analytic in ``c`` (the chiral confluent coefficients alone
    are not, at uneven collisions).

    For *off-diagonal* blocks the ``n``-axis carries its own recenter point:
    ``ev_h_n`` / ``ev_hbar_n`` are the holomorphic / antiholomorphic evaluators
    recentered at ``center_bar`` (they alias the ``m``-axis evaluators on the
    diagonal, so no extra build there).
    """
    from flint import acb

    half_h, half_hbar, half_h_n, half_hbar_n = _probe_halves(
        ev_h, ev_hbar, ev_h_n, ev_hbar_n, Mmax, n_terms,
        lambda ev: _make_half_lookup(ev, Mmax, n_terms, b0=b0, match_digits=match_digits),
    )
    return _assemble_cluster_rc(
        half_h, half_hbar, groups, ell, Mmax, entry_pairs, sigma, n_terms,
        half_h_n=half_h_n, half_hbar_n=half_hbar_n, zero=acb(0),
    )


def _u_from_numeric(coeffs, n_terms):
    out = [LaurentSeries.zero() for _ in range(n_terms)]
    for i, c in enumerate(coeffs[:n_terms]):
        out[i] = LaurentSeries.constant(c)
    return out


def _u_mul(a, b, n_terms):
    # Native bivariate convolution: one packed flint multiply instead of the
    # O(n_terms**2) per-coefficient LaurentSeries loop.
    return u_convolve(a[:n_terms], b[:n_terms])


def _u_numeric_mul(a, b, n_terms):
    out = [mp.mpc(0)] * n_terms
    for i in range(n_terms):
        acc = mp.mpc(0)
        for j in range(i + 1):
            acc += a[j] * b[i - j]
        out[i] = acc
    return out


def _u_numeric_div(a, b, n_terms):
    """``a / b`` of numeric u-series (``b[0] != 0``)."""
    out = [mp.mpc(0)] * n_terms
    inv0 = 1 / mp.mpc(b[0])
    for n in range(n_terms):
        acc = mp.mpc(a[n]) if n < len(a) else mp.mpc(0)
        for k in range(1, min(n, len(b) - 1) + 1):
            acc -= b[k] * out[n - k]
        out[n] = acc * inv0
    return out


def _u_numeric_log(a, n_terms):
    """``log`` of a numeric u-series (``a[0] != 0``): ``n l_n a_0 = n a_n - sum_{k<n} k l_k a_{n-k}``."""
    a = [mp.mpc(v) for v in a] + [mp.mpc(0)] * max(0, n_terms - len(a))
    out = [mp.mpc(0)] * n_terms
    out[0] = mp.log(a[0])
    for n in range(1, n_terms):
        acc = n * a[n]
        for k in range(1, n):
            acc -= k * out[k] * a[n - k]
        out[n] = acc / (n * a[0])
    return out


class _UseriesCap:
    """Run flint series arithmetic truncated at ``n_terms`` u-powers, restoring the global series cap after (the
    t-series layer rests on its own cap, :data:`tseries._CAP`)."""

    def __init__(self, n_terms):
        self.n_terms = n_terms

    def __enter__(self):
        from flint import ctx as flint_ctx

        self.saved = flint_ctx.cap
        flint_ctx.prec = int(mp.mp.dps * 3.33) + 20
        flint_ctx.cap = self.n_terms
        return self

    def __exit__(self, *exc):
        from flint import ctx as flint_ctx

        flint_ctx.cap = self.saved
        return False


def _flint_useries(values, n_terms):
    """A numeric u-series as a flint ``acb_series`` (call inside :class:`_UseriesCap`)."""
    from flint import acb_series

    return acb_series([_mpf_to_acb(mp.mpc(v)) for v in values[:n_terms]])


def _series_coeffs(ser, n_terms):
    from flint import acb

    c = list(ser.coeffs())
    return c + [acb(0)] * (n_terms - len(c))


def _numeric_series_powers(values, power, n_terms):
    """``[values**k for k in 0..power]`` of a numeric u-series, as mpmath lists (flint series products)."""
    with _UseriesCap(n_terms):
        base = _flint_useries(values, n_terms)
        acc = _flint_useries([1], n_terms)
        out = []
        for _k in range(power + 1):
            out.append([_acb_to_mpc(v) for v in _series_coeffs(acc, n_terms)])
            acc = acc * base
    return out


def _prefactor_orders(log_q16N, dp0, n_orders, n_terms, cache):
    """The numeric u-series ``E_r = exp(dp L) L^r / r!`` (``L = log(16 q / N)``, ``r < n_orders``) as flint
    coefficient lists: the prefactor's orders around ``dp`` (flint series, ``L^r`` cached per evaluator)."""
    from flint import acb

    with _UseriesCap(n_terms):
        L = cache.get("L_series")
        if L is None or cache.get("L_cap") != n_terms:
            L = _flint_useries(log_q16N, n_terms)
            cache["L_series"], cache["L_cap"] = L, n_terms
            cache["L_pows"] = [_flint_useries([1], n_terms)]
        L_pows = cache["L_pows"]
        while len(L_pows) < n_orders:
            L_pows.append(L_pows[-1] * L)
        E_exp = (L * _mpf_to_acb(mp.mpc(dp0))).exp()
        out = []
        fact = acb(1)
        for r in range(n_orders):
            if r:
                fact = fact * r
            out.append(_series_coeffs(E_exp * L_pows[r] / fact, n_terms))
    return out


def _series_geometry(coord, h_ext, n_terms, *, c0, q_order, chiral_norm, z_shift,
                     u_center=None):
    import virasoro_blocks.derivs as _derivs_local
    from virasoro_blocks.recursion import _resolve_chiral_norm

    # Chiral F coordinates (z / y / q) expand the same single chiral half as
    # the diagonal two-sided coordinates (zzb / yyb / qqb); only the downstream
    # assembly (single half vs convolution, jac 1 vs 1/2) differs.
    if coord in ("zzb", "z"):
        z_of_u = lambda u: u
        u0 = z_symmetric()
        coord_key = "z"
    elif coord in ("yyb", "y"):
        z_of_u = z_of_y
        u0 = y_symmetric()
        coord_key = "y"
    elif coord in ("qqb", "q"):
        z_of_u = z_of_q
        u0 = q_symmetric()
        coord_key = "q"
    else:
        raise _MergedScorerUnavailable

    # Off-symmetric / off-diagonal blocks recenter the chiral expansion at
    # ``u_center`` (the same coordinate map ``z(u)``, a different expansion
    # point); ``u_center`` is already in this coordinate's ``u`` variable
    # (:func:`polyderivs._resolve_center_u` returns ``spec.u_of_z(center)``).
    if u_center is not None:
        u0 = u_center

    b0 = b_from_c(c0)
    c_val = mp.mpf(c0)
    z_shift_val = mp.mpf(0) if z_shift is None else z_shift
    sigma0 = (c_val - 1) / 24
    exp_z0 = sigma0 + z_shift_val
    exp_omz0 = sigma0 - h_ext[1] - h_ext[2]
    exp_theta0 = (c_val - 1) / 2 - 4 * sum(h_ext)
    q_coeffs, base0_coeffs = _derivs_local._chiral_geometry_series_coeffs(
        h_ext, n_terms, b0, z_shift_val, sigma0, exp_z0, exp_omz0, exp_theta0,
        z_of_u=z_of_u, u_symmetric=u0, coord_key=coord_key,
    )
    N = _resolve_chiral_norm(chiral_norm)

    # log(16 q / N) and the prefactor's c-derivative sigma_log = -log(16 q) + log z + log(1 - z) + 12 log theta3 as
    # u-series, from the (certified flint) geometry series: log of the q series, and log of the prefactor ratio at
    # c0 + 24 over c0, which shifts the exponents (sigma, z, 1 - z, theta3) by (1, 1, 1, 12)
    # (uncached: the geometry cache keys on c through b, not on the exponents, so a shifted call would read the
    # unshifted prefactor back)
    shifted = _derivs_local._flint_geometry_series_coeffs(
        n_terms, sigma0 + 1, exp_z0 + 1, exp_omz0 + 1, exp_theta0 + 12, z_of_u=z_of_u, u_symmetric=u0,
    )
    if shifted is None:
        raise _MergedScorerUnavailable
    _q_shift, base_shift_coeffs = shifted
    q_series = [mp.mpc(v) for v in q_coeffs[:n_terms]]
    log_q16N = _u_numeric_log(q_series, n_terms)
    log_q16N[0] += mp.log(16) - mp.log(N)
    sigma_log = _u_numeric_log(
        _u_numeric_div([mp.mpc(v) for v in base_shift_coeffs[:n_terms]],
                       [mp.mpc(v) for v in base0_coeffs[:n_terms]], n_terms),
        n_terms,
    )

    factor = [LaurentSeries.zero() for _ in range(n_terms)]
    factor[0] = LaurentSeries.constant(1)
    sigma_power = [mp.mpc(0)] * n_terms
    sigma_power[0] = mp.mpc(1)
    fact = mp.mpf(1)
    # (c - c0)/24 = sigma t^p/24, in the local variable of the H-table (H_q_series_laurent_precompute)
    from .params import laurent_t_power

    t_power = laurent_t_power(c0)
    t_sq_over_24 = mp.mpf(laurent_t_sq_sign(c0)) / 24
    for r in range(1, max(0, _ts.T_HI // t_power) + 1):
        sigma_power = _u_numeric_mul(sigma_power, sigma_log, n_terms)
        fact *= r
        scale = t_sq_over_24 ** r / fact
        for i in range(n_terms):
            if sigma_power[i] != 0:
                factor[i] = factor[i] + LaurentSeries.monomial(t_power * r, sigma_power[i] * scale)
    base_series = _u_mul(_u_from_numeric(base0_coeffs, n_terms), factor, n_terms)
    # The same prefactor as numeric u-series at each power of t: BASE' = sum_r t^(p r) B_r with
    # B_r = BASE'_0 * (sigma/24)^r sigma_log^r / r! (a product with it is a sum of numeric u-convolutions)
    base0 = [mp.mpc(v) for v in base0_coeffs[:n_terms]] + [mp.mpc(0)] * max(0, n_terms - len(base0_coeffs))
    base_terms = [(0, base0)]
    power = [mp.mpc(0)] * n_terms
    power[0] = mp.mpc(1)
    fact = mp.mpf(1)
    for r in range(1, max(0, _ts.T_HI // t_power) + 1):
        power = _u_numeric_mul(power, sigma_log, n_terms)
        fact *= r
        scale = t_sq_over_24 ** r / fact
        base_terms.append((t_power * r, _u_numeric_mul(base0, [v * scale for v in power], n_terms)))

    q_pows = _numeric_series_powers(q_coeffs[:n_terms], q_order, n_terms)
    return q_pows, base_series, log_q16N, base_terms


# Adaptive Laurent-window state for the t-series scorer. The needed window
# half-width (the deepest negative t-power that appears) is cheap to predict
# from input valuations and constant across a build, so it is measured once per
# (c0, q_order, dps, ell) and cached; subsequent builds at that key run at the
# tight window. ``_MEASURE_WINDOW`` gates the (cheap) per-call valuation scan so
# cache hits pay nothing for it.
_WINDOW_CACHE: dict = {}
_DEEPEST_NEG_SEEN = 0
_MEASURE_WINDOW = False
# Set by _build_series_adaptive's measure-only window-sizing pass (with
# _MEASURE_WINDOW): pole_laurent runs only the cheap pole-weight build +
# valuation scan and returns zero u-rows, skipping the expensive H/q/P
# contraction, so the window is measured exactly without a full block build.
_MEASURE_ONLY = False
_SERIES_HALF_LOOKUP_CACHE: dict = {}
_SERIES_HALF_LOOKUP_CACHE_MAXSIZE = 512


def _seed_window_half(q_order: int, c0=None) -> int:
    """Generous a-priori seed for the *measuring* Laurent-window build.

    The realized window half-width (deepest negative t-power = val(weight) +
    val(H_table)) grows linearly in q_order; a fit at c = 28 gives
    needed ~ 0.29*q + 3. This is only the seed for the first (measuring) build,
    which then records the *exact* tight width for every later build -- so the
    seed need only be a safe upper bound, not exact.  It is the window every
    chiral H-series build at this q_order uses (:func:`laurent_window_half`,
    0.4 q + 6, above the measured depth with a margin), so the measuring pass,
    the tight build (a narrowed view) and the identity vector share ONE H-table
    build; a measurement that touches the window edge widens and re-measures
    (cheaply), and the surviving-pole backstop guards the final build.  A fixed
    constant cannot do this: 12 is wasteful at small q yet silently truncates
    at q >~ 32 (deepest exceeds 12), corrupting the cached width.
    """
    from .recursion import laurent_window_half

    return max(8, laurent_window_half(q_order, c0))


def _series_chiral_evaluator(h_ext, n_terms, *, c0, q_order, coord,
                             chiral_norm=None, z_shift=None, u_center=None):
    # (the reference t-series assembly reads the H-table against its own t-dependent geometry: the whole window)
    from .recursion import _narrowed_state

    # (this assembly runs on its own window: a table built wider -- cached by another caller -- is read narrowed)
    state = _narrowed_state(H_q_series_laurent_precompute(
        h_ext, c0=c0, q_order=q_order, work_dps=mp.mp.dps, trim=False,
    ))
    series_cache_key = (
        mp.mp.prec, mp.mp.dps, _ts.window_half(), _scalar_cache_key(c0),
        int(q_order), int(n_terms),
        coord, _scalar_cache_key(chiral_norm), _scalar_cache_key(z_shift),
        _scalar_cache_key(u_center), tuple(_scalar_cache_key(x) for x in h_ext),
    )
    pairs = state["pairs"]
    H_table = state["H_table"]
    # Lazily-cached deepest (most negative) valuation of each H_table[idx] row,
    # used only during window measurement (cache miss). The deepest intermediate
    # negative power in the whole contraction is val(pole_weight) + val(H_table),
    # so this plus the per-pole weight valuation predicts the needed window.
    _vh_cache: dict = {}

    def _vh_for(idx):
        v = _vh_cache.get(idx, False)
        if v is False:
            ht = H_table[idx]
            vh = None
            for _m in range(len(ht)):
                vv = ht[_m].valuation()
                if vv is not None and (vh is None or vv < vh):
                    vh = vv
            _vh_cache[idx] = vh
            v = vh
        return v
    q_pows, base_series, log_q16N, base_terms = _series_geometry(
        coord, h_ext, n_terms, c0=c0, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift, u_center=u_center,
    )
    # Hoist the q-power churn out of the per-pole H_u contraction: q_pows[k][i] is
    # read for every pole, so do the mpmath "!= 0" test and the mpc->acb conversion
    # ONCE here (q_order*n_terms times) instead of inside pole_laurent (x n_poles x
    # n_rows). q_pows_nz is the nonzero mask; q_pows_acb the pre-converted acb
    # scalars (None where zero).
    q_pows_nz = [[v != 0 for v in row] for row in q_pows]
    _acb_of = _ts._acb_from_mpc
    q_pows_acb = [
        [(_acb_of(v) if nz else None) for v, nz in zip(row, nzrow)]
        for row, nzrow in zip(q_pows, q_pows_nz)
    ]
    active_pairs = [
        (idx, p, p["K"], p["d"], mp.mpc(p["d"].coeff(0)))
        for idx, p in enumerate(pairs)
        if not p["K"].is_zero()
    ]
    match_tol = mp.mpf(10) ** (-(mp.mp.dps // 3))

    # Per-evaluator scaffolding shared across every pole_laurent call (the dp-
    # independent pieces): dlogP = log(16q/N) lifted to a u-row, and its powers
    # P_pows = [1, dlogP, dlogP^2, ...] (extended on demand). Only ``P`` and
    # ``P_rows`` (which carry the (16q/N)^dp prefactor) stay per-pole.
    _pl_cache: dict = {}

    # Pole-INDEPENDENT contraction of each Kac pair's H-table row against the
    # q-powers, shared across every pole_laurent call. The per-pole block needs
    #   H_u[col][i] = sum_k (sum_{idx in col} w_idx H_table[idx][k-rs_idx]) q_pows[k][i]
    # and since the weight w_idx does not depend on k it factors out of the k-sum:
    #   H_u[col][i] = sum_{idx in col} w_idx * Q[idx][i],
    #   Q[idx][i]   = sum_k H_table[idx][k-rs_idx] q_pows[k][i].
    # Q depends only on (idx, i) -- not on the pole -- so the whole k-contraction
    # (the dominant per-pole cost, ~q_order deep) is computed ONCE here and reused
    # across all ~q^2 poles, leaving the per-pole work proportional to n_terms
    # (~m_max) instead of q_order. Built lazily (skipped by the measure-only pass).
    _Q_cache: dict = {}

    def _Q_for(idx):
        q = _Q_cache.get(idx)
        if q is not None:
            return q
        rs_idx = pairs[idx]["rs"]
        ht = H_table[idx]
        # Q[idx][i] = sum_m q_pows[m + rs][i] H_table[idx][m]: a numeric linear map on the row's
        # t-series, one flint matmul (numeric_linear_map); zero rows of the H-table are dropped.
        live = [m for m in range(len(ht)) if not ht[m].is_zero()]
        if not live:
            out = [(LaurentSeries.zero(), True) for _ in range(n_terms)]
        else:
            weights = [[q_pows_acb[m + rs_idx][i] if q_pows_nz[m + rs_idx][i] else 0 for m in live]
                       for i in range(n_terms)]
            mapped = numeric_linear_map(weights, [ht[m] for m in live])
            # Cache the structural-zero flag alongside the series: the per-pole H_u
            # loop tests it for every (pole, col, idx), and the is_zero scan
            # (_valuation_index over all coefficients) is otherwise the top hotspot.
            out = [(acc, acc.is_zero()) for acc in mapped]
        _Q_cache[idx] = out
        return out

    _active_pos = {idx: k for k, (idx, *_rest) in enumerate(active_pairs)}
    _stacked_cache: dict = {}

    def _stacked_Q():
        key = (mp.mp.prec, _ts.window_half(), _ts._real_path())
        hit = _stacked_cache.get(key)
        if hit is None:
            hit = StackedURows([[qi for qi, _qz in _Q_for(idx)] for idx, *_rest in active_pairs], n_terms)
            _stacked_cache.clear()
            _stacked_cache[key] = hit
        return hit

    def kac_poles_series():
        return [d_e for _idx, _p, _K_e, d_e, _d0 in active_pairs]

    def _measure_pole(dp, dp0, n_pr, n_rg, targets, allow_regular_point):
        """The window-measuring pass at one location: the deepest val(weight) + val(H row), from valuations alone
        (a weight's valuation is the sum of its factors': val K + j val(d - dp) on the cluster, val K - (q + 1)
        val(dp - d) off it), with no series product; zero rows out (the measure pass discards the grid)."""
        global _DEEPEST_NEG_SEEN
        n_match = 0
        for idx, _p, K_e, d_e, d0 in active_pairs:
            if targets is None:
                is_match = abs(d0 - dp0) <= match_tol
            else:
                is_match = any(abs(d0 - t) <= match_tol for t in targets)
            vK = K_e.valuation()
            vh = _vh_for(idx)
            if vK is None or vh is None:
                n_match += is_match
                continue
            if is_match:
                n_match += 1
                vd = (d_e - dp).valuation()
                vals = [vK] + ([vK + j * vd for j in range(1, n_pr)] if vd is not None else [])
            else:
                vd = (dp - d_e).valuation()
                vals = [vK - (q + 1) * vd for q in range(n_rg)] if vd is not None else []
            for v in vals:
                if v + vh < _DEEPEST_NEG_SEEN:
                    _DEEPEST_NEG_SEEN = v + vh
        if n_match == 0 and not allow_regular_point:
            return None
        zero = [LaurentSeries.zero() for _ in range(n_terms)]
        return [list(zero) for _ in range(n_pr)], [list(zero) for _ in range(n_rg)]

    def pole_laurent(d_pole, *, principal_order, regular_order,
                     matched_locations=None, allow_regular_point=False):
        dp = LaurentSeries.constant(d_pole)
        dp0 = mp.mpc(d_pole)
        n_pr = int(principal_order)
        n_rg = int(regular_order) + 1
        if n_pr < 1 or n_rg < 1:
            return None
        targets = None if matched_locations is None else [mp.mpc(v) for v in matched_locations]
        if _MEASURE_ONLY:
            return _measure_pole(dp, dp0, n_pr, n_rg, targets, allow_regular_point)
        cols = [[] for _ in range(n_pr + n_rg)]
        n_match = 0
        for idx, _p, K_e, d_e, d0 in active_pairs:
            if targets is None:
                is_match = abs(d0 - dp0) <= match_tol
            else:
                is_match = any(abs(d0 - t) <= match_tol for t in targets)
            if is_match:
                n_match += 1
                cols[0].append((idx, K_e))
                dpow = d_e - dp
                for j in range(1, n_pr):
                    cols[j].append((idx, K_e * dpow))
                    dpow = dpow * (d_e - dp)
            else:
                inv = (dp - d_e).reciprocal()
                term = K_e * inv
                cols[n_pr].append((idx, term))
                for q in range(1, n_rg):
                    term = -term * inv
                    cols[n_pr + q].append((idx, term))
        if n_match == 0 and not allow_regular_point:
            return None


        # Assemble H_u directly via the shared pole-independent Q (see _Q_for):
        #   H_u[col][i] = sum_{idx in col} w_idx * Q[idx][i]
        # plus, for the first regular column only, the H_rows[n_pr][0] = 1 leading
        # term, which contributes 1 * q_pows[0][i] = q_pows[0][i] to H_u[n_pr][i].
        # every u-component of a column at once: one matmul of the column's weights against the pairs' Q rows
        # stacked once per evaluator (_stacked_Q)
        stacked = _stacked_Q()
        H_u = []
        for ci, col in enumerate(cols):
            weights = [None] * len(active_pairs)
            for idx, w in col:
                weights[_active_pos[idx]] = w
            row = stacked_weighted_sum(weights, stacked)
            if ci == n_pr:
                for i in range(n_terms):
                    q0 = q_pows[0][i]
                    if q0 != 0:
                        row[i] = row[i] + LaurentSeries.constant(q0)
            H_u.append(row)

        # F = P(h) H(h) near dp with P(h) = BASE'(u; t) exp(h log(16 q / N)(u)).  Around dp the exponential is
        # exp(dp L) sum_r L^r (h - dp)^r / r! with L = log(16 q / N) t-INDEPENDENT (pure geometry), so the order-r
        # rows are P_r = BASE' * E_r with the numeric u-series E_r = exp(dp L) L^r / r!.  Every output row is then
        # BASE' * (sum_r E_r * H_u[...]), and BASE' itself is a short sum of numeric u-series times local powers of
        # t (_series_geometry's base_terms), so both products are numeric-by-series u-convolutions: one flint
        # matmul each (u_numeric_convolve_sum), no bivariate (u, t) convolution at all.
        n_p_rows = n_pr + n_rg
        E = _prefactor_orders(log_q16N, dp0, n_p_rows, n_terms, _pl_cache)

        def times_base(G):
            # BASE' * G = sum_r t^(p r) B_r * G, numeric u-convolutions with t-shifts (one matmul)
            return u_numeric_convolve_sum([(B, G, shift) for shift, B in base_terms], n_terms)

        F_pr = []
        for j in range(n_pr):
            G = u_numeric_convolve_sum([(E[r], H_u[j + r]) for r in range(0, n_pr - j)], n_terms)
            F_pr.append(times_base(G))

        F_rg = []
        for q in range(n_rg):
            terms = [(E[r], H_u[n_pr + q - r]) for r in range(0, q + 1)]
            terms += [(E[q + j], H_u[j - 1]) for j in range(1, n_pr + 1)]
            G = u_numeric_convolve_sum(terms, n_terms)
            F_rg.append(times_base(G))
        return F_pr, F_rg

    def evaluate(_h):
        raise _MergedScorerUnavailable

    evaluate._series_cache_key = series_cache_key
    evaluate.pole_laurent = pole_laurent
    evaluate.kac_poles_series = kac_poles_series
    return evaluate


def _series_build_real(c0, scalars) -> bool:
    """Whether the whole t-series scorer build is real (the "decide once" flag).

    Returns True iff the arb backing is safe for every LaurentSeries in the
    build: the Kac weights ``d_rs`` and fusion polynomials must carry real
    t-coefficients -- ``b(c0)`` real (``c0 >= 25``) or purely imaginary
    (``c0 <= 1``, where the H-table carries ``beta = -i b``; see
    :func:`params.beta_series`) -- AND every scalar feeding the
    geometry/recursion (externals, coordinate shifts, swap phase, chiral norm,
    recenter points) must be real. Any complex -- or any value that cannot be
    coerced to a scalar -- returns False, so the build falls back to the
    per-handle realness scan (correct, acb batches).
    """
    if not _ts._USE_ARB_LAURENT:
        return False

    def _real(x) -> bool:
        if x is None:
            return True
        try:
            return mp.mpc(x).imag == 0
        except (TypeError, ValueError):
            return False

    if not _real(c0):
        return False
    try:
        if b_from_c(mp.mpf(c0)).imag != 0 and not imaginary_b_locus(c0):
            return False
    except (TypeError, ValueError):
        return False
    return all(_real(x) for x in scalars)


def _build_series_adaptive(build_series, *, c0, q_order, ell, real=False):
    """Run the t-series scorer's ``build_series`` at an adaptively-sized window.

    The needed window half-width (the deepest negative t-power that appears in
    the contraction) is constant across a build. The *exact* value cannot be
    read off cheaply up front -- it is ``val(weight) + val(H_table)`` minimized
    over Kac pairs and columns (the regular-pole inverse-weight tower
    ``inv**(q+1)`` for ``inv = 1/(dp - d_e)`` plus the chiral A_rs/H depths) --
    but it grows linearly in q_order (~0.29*q + 3 at c = 28). On a cache miss the
    depth is found by a CHEAP measure-only pass (:data:`_MEASURE_ONLY`): it builds
    the pole weights and scans val(weight)+val(H_table) but skips the width^2
    ``t_conv_weighted_sum`` contraction that dominates a full build, so it can run
    at a generous :func:`_seed_window_half` window without paying for a wide full
    build. The exact depth is cached and the *real* build then runs tight -- much
    cheaper at high q_order than the former "seed the full measuring build wide"
    approach (e.g. q=60 at c=28: ~125s vs ~181s), and safe because the generous
    seed keeps the measurement unclamped (a floor contact triggers a cheap
    re-measure, and the surviving-pole backstop guards the final build).
    """
    global _MEASURE_WINDOW, _DEEPEST_NEG_SEEN, _MEASURE_ONLY
    # Key on (c0, q_order, dps) but NOT ell: the scorer runs ~once per spin, so an
    # ell in the key would make every spin a fresh miss (the cache would only help
    # across re-builds). The needed window is ell- AND m_max-independent in practice
    # (the deepest power is val(weight)+val(H_table), which the regular-pole inverse
    # tower does not deepen -- far poles have valuation 0 -- so it depends on
    # q_order alone; verified at c=28), so spin 0 measures it and later spins hit
    # the cache. If some locus ever needs a deeper window the surviving-pole
    # backstop widens and the cached width grows to the max seen -- always safe.
    key = (str(c0), int(q_order), int(mp.mp.dps))
    saved = _ts.window_half()
    # "Decide once" real-mode (mirrors derivs._use_arb): when the caller has
    # asserted the whole build is real (b(c0) real or purely imaginary, and all
    # externals/scalars real),
    # back every LaurentSeries with arb -- from_mpc then skips its per-handle
    # realness scan and the batch contractions run in arb (real arithmetic, no
    # always-zero imaginary channel). Restored in the finally.
    saved_real = _ts.set_real_mode(real)
    cached = _WINDOW_CACHE.get(key)
    try:
        if cached is None:
            # Window sizing via a CHEAP measure-only pass: it builds the pole
            # weights and scans val(weight)+val(H_table) but skips the width^2
            # t_conv_weighted_sum contraction (the build's dominant cost), so the
            # exact depth is found without paying for a wide *full* build. The
            # generous seed keeps the measurement unclamped; a floor contact
            # (deepest reaches the window edge) means even the seed was too small,
            # so widen and re-measure -- a cheap retry, not a full rebuild. This is
            # measured once per (c0, q_order, dps); later builds run tight.
            measure_half = _seed_window_half(q_order, c0)
            for _ in range(8):
                _ts.set_window(measure_half)
                _MEASURE_WINDOW = True
                _MEASURE_ONLY = True
                _DEEPEST_NEG_SEEN = 0
                try:
                    build_series()
                finally:
                    _MEASURE_WINDOW = False
                    _MEASURE_ONLY = False
                if abs(_DEEPEST_NEG_SEEN) < measure_half:
                    break  # no floor contact -> the measurement is exact
                measure_half += max(4, measure_half // 2)
            # Storing the deepest power t^-D requires half >= D. Products
            # contributing to t^0 also need positive guard powers beyond D;
            # omitted positive powers need not trigger the surviving-pole check.
            cached = max(4, abs(_DEEPEST_NEG_SEEN) + max(4, abs(_DEEPEST_NEG_SEEN) // 4))
            _WINDOW_CACHE[key] = cached
        # Real build at the exact tight window. The surviving-pole backstop is a
        # final safety net (e.g. should the measure pass ever under-report without
        # floor contact): widen, recache, retry the full build.
        half = cached
        attempts = 0
        while True:
            _ts.set_window(half)
            try:
                grid = build_series()
            except ValueError as exc:
                if "keeps a t" not in str(exc) or attempts >= 5:
                    raise
                attempts += 1
                half += 4
                _WINDOW_CACHE[key] = half
                continue
            return grid
    finally:
        _MEASURE_WINDOW = False
        _MEASURE_ONLY = False
        _ts.set_window(saved)
        _ts.set_real_mode(saved_real)


def _partner_reg_orders(ev, groups, ell):
    """Map each chiral half location (cache key) to the regular Taylor order the two-sided assembly reads there.

    :func:`_assemble_cluster_rc` pairs the principal rows of one half with the regular rows of its partner (the
    half at ``(loc -/+ ell)/2`` of the same Delta-cluster) up to index ``(partner principal order) - 1`` and no
    further, so that is the order each half needs.  The principal order of a half is the number of Kac weights
    colliding there (exact limits, ``kac_poles_series`` t^0 parts), or 1 at a regular point.  Taking the maximum
    over the clusters a location serves covers a half whose collision order exceeds the Delta-cluster size (the
    c = 28, q_order 80 case that once motivated padding every half to the global block depth)."""
    from virasoro_blocks.polyderivs import _mpc_cache_key

    kac = getattr(ev, "kac_poles_series", None)
    kp = None if kac is None else kac()
    if kp is None:
        raise _MergedScorerUnavailable
    kp0 = [mp.mpc(v.coeff(0)) for v in kp]
    order = {}

    def principal_order(x):
        key = _mpc_cache_key(x)
        if key not in order:
            order[key] = max(1, len(_collide_cluster(kp0, x)))
        return order[key]

    out = {}
    for loc, _idx in groups:
        locc = mp.mpc(loc)
        h0, hbar0 = (locc + ell) / 2, (locc - ell) / 2
        for here, partner in ((h0, hbar0), (hbar0, h0)):
            key = _mpc_cache_key(here)
            need = principal_order(partner) - 1
            if out.get(key, -1) < need:
                out[key] = need
    return out


def _make_half_lookup_series(ev, Mmax, n_terms, jac=None, reg_order_for=None, default_reg=None):
    from virasoro_blocks.polyderivs import _mpc_cache_key

    pole_laurent = getattr(ev, "pole_laurent", None)
    kac_poles = getattr(ev, "kac_poles_series", None)
    if pole_laurent is None or kac_poles is None:
        raise _MergedScorerUnavailable
    kp = kac_poles()
    if kp is None:
        raise _MergedScorerUnavailable
    kp0 = [mp.mpc(v.coeff(0)) for v in kp]
    cache = {}
    state_key = getattr(ev, "_series_cache_key", None)

    def half_lookup(hv):
        key = _mpc_cache_key(hv)
        hit = cache.get(key)
        if hit is not None:
            return hit
        # the regular Taylor order the assembly reads here (_partner_reg_orders); without a map, the global block
        # depth
        fallback = (Mmax - 1) if default_reg is None else default_reg
        reg = max(0, fallback if reg_order_for is None else reg_order_for.get(key, fallback))
        series_cache_key = None
        if state_key is not None:
            series_cache_key = (state_key, jac, Mmax, n_terms, reg, key)
            cached_series = _SERIES_HALF_LOOKUP_CACHE.get(series_cache_key)
            if cached_series is not None:
                cache[key] = cached_series
                return cached_series
        near = _collide_cluster(kp0, hv)
        k = len(near)
        if k > 0:
            pl = pole_laurent(
                hv, principal_order=k, regular_order=reg,
                matched_locations=near,
            )
        else:
            pl = pole_laurent(
                hv, principal_order=1, regular_order=reg,
                allow_regular_point=True,
            )
        if pl is None:
            raise _MergedScorerUnavailable
        principal, regular = pl
        out = _delta_frame_half(principal, regular, n_terms, jac=jac)
        cache[key] = out
        # never from the measure-only pass: its rows are zero placeholders (no contraction), and a real build at
        # the measuring window's width would read them back (c = 28, q_order 8: 3e-2 off)
        if series_cache_key is not None and not _MEASURE_ONLY:
            _SERIES_HALF_LOOKUP_CACHE[series_cache_key] = out
            if len(_SERIES_HALF_LOOKUP_CACHE) > _SERIES_HALF_LOOKUP_CACHE_MAXSIZE:
                _SERIES_HALF_LOOKUP_CACHE.pop(next(iter(_SERIES_HALF_LOOKUP_CACHE)))
        return out

    return half_lookup


def _optionb_persp_rows_series(ev_h, ev_hbar, *, groups, ell, Mmax,
                               entry_pairs, sigma, n_terms,
                               ev_h_n=None, ev_hbar_n=None):
    # the m- and n-axis halves share the Kac structure, so one map serves every evaluator
    reg_for = _partner_reg_orders(ev_h, groups, ell)
    half_h = _make_half_lookup_series(ev_h, Mmax, n_terms, reg_order_for=reg_for)
    half_hbar = half_h if ev_hbar is ev_h else _make_half_lookup_series(
        ev_hbar, Mmax, n_terms, reg_order_for=reg_for
    )
    # Off-diagonal blocks carry a second (n-axis) expansion recentered at
    # ``center_bar``; alias the m-axis halves when the evaluator is shared.
    half_h_n = None
    half_hbar_n = None
    if ev_h_n is not None:
        if ev_h_n is ev_h:
            half_h_n = half_h
        elif ev_h_n is ev_hbar:
            half_h_n = half_hbar
        else:
            half_h_n = _make_half_lookup_series(ev_h_n, Mmax, n_terms, reg_order_for=reg_for)
        if ev_hbar_n is ev_h:
            half_hbar_n = half_h
        elif ev_hbar_n is ev_hbar:
            half_hbar_n = half_hbar
        elif ev_hbar_n is ev_h_n:
            half_hbar_n = half_h_n
        else:
            half_hbar_n = _make_half_lookup_series(ev_hbar_n, Mmax, n_terms, reg_order_for=reg_for)
    return _assemble_cluster_rc(
        half_h, half_hbar, groups, ell, Mmax, entry_pairs, sigma, n_terms,
        half_h_n=half_h_n, half_hbar_n=half_hbar_n,
    )


def _series_rows_constant_part(rows, *, c0):
    out = []
    for gi, row in enumerate(rows):
        flat = []
        for ri, value in enumerate(row):
            if isinstance(value, LaurentSeries):
                valuation = value.valuation()
                if valuation is not None and valuation < 0:
                    raise ValueError(
                        f"merged t-series scorer: row {gi}, coefficient {ri} "
                        f"keeps a t^{valuation} pole at c0 = {c0}"
                    )
                flat.append(mp.mpc(value.coeff(0)))
            else:
                flat.append(mp.mpc(value))
        out.append(flat)
    return out


def _optionb_persp_rows_series_folded(ev_h, ev_hbar, *, groups, ell, Mmax,
                                      folded_entries, sigma, n_terms,
                                      fold_marker, m_max, n_max, c0):
    """Series analogue of :func:`_optionb_persp_rows_at_probe_folded`.

    Builds the non-folded merged Laurent rows over the fold *source* band as
    exact ``t``-series (via :func:`_assemble_cluster_rc`), reduces them to their
    ``t**0`` constant parts -- running the surviving-pole guard once on the
    non-folded source rows -- then applies the same Delta-independent parity
    fold the value assembler uses. The fold is a linear, ``t``-independent
    recombination, so folding the constant parts equals the constant part of the
    folded series; returning scalars keeps the flattened
    ``RC[ei * Mmax + (b - 1)]`` layout identical to the extrapolation path so the
    caller's scoring loop is unchanged.
    """
    from virasoro_blocks.derivs import _fold_parity, _fold_plan

    odd = fold_marker % 2
    M_total = m_max + 2 * n_max + odd
    src_dim = M_total + 1
    plan, _ = _fold_plan(bool(odd), src_dim, src_dim, m_max, n_max)
    src_entries = sorted({(m, n) for (m, n, _c) in plan})

    reg_for = _partner_reg_orders(ev_h, groups, ell)
    half_h = _make_half_lookup_series(ev_h, Mmax, n_terms, reg_order_for=reg_for)
    half_hbar = half_h if ev_hbar is ev_h else _make_half_lookup_series(
        ev_hbar, Mmax, n_terms, reg_order_for=reg_for
    )
    src_rows_series = _assemble_cluster_rc(
        half_h, half_hbar, groups, ell, Mmax, src_entries, sigma, n_terms
    )
    # Reduce to t^0 constants before the fold mixes the entries (also runs the
    # surviving-pole guard on the non-folded source rows).
    src_rows = _series_rows_constant_part(src_rows_series, c0=c0)
    src_index = {e: i for i, e in enumerate(src_entries)}

    folded_rows = []
    for row in src_rows:
        # Per Laurent order ``b`` rebuild the non-folded source grid and fold.
        folded_by_b = []  # folded_by_b[b-1][alpha][beta]
        for b in range(1, Mmax + 1):
            grid = [[mp.mpc(0)] * src_dim for _ in range(src_dim)]
            for (m, n), ei in src_index.items():
                if m < src_dim and n < src_dim:
                    grid[m][n] = row[ei * Mmax + (b - 1)]
            folded_by_b.append(
                _fold_parity(grid, fold_marker, m_max, n_max, zero=mp.mpc(0))
            )
        flat = []
        for (a, bb) in folded_entries:
            for b in range(1, Mmax + 1):
                flat.append(folded_by_b[b - 1][a][bb])
        folded_rows.append(flat)
    return folded_rows


def _optionb_persp_rows_at_probe_folded(ev_h, ev_hbar, *, groups, ell, Mmax,
                                        folded_entries, sigma, n_terms,
                                        fold_marker, m_max, n_max, b0, match_digits):
    """Per-cluster flattened *folded* block RC rows for ONE probe ``c``, as flint ``acb`` balls.

    The folded ``(w, s)`` entry ``(alpha, beta)`` is a fixed, Delta-independent
    linear recombination (:func:`derivs._fold_parity`) of a band of non-folded
    chiral ``(m, n)`` products. Because the fold is linear and Delta-independent
    it commutes with taking the principal part, so we reconstruct the full
    non-folded merged Laurent ``RC[m][n][b]`` (via :func:`_assemble_cluster_rc`
    over the source band), then fold each ``1/(Delta-loc)^b`` slice with the
    same map the value assembler uses, and return the folded rows in the same
    flattened ``RC[ei * Mmax + (b - 1)]`` layout the diagonal path produces (so
    the caller's scoring loop is unchanged). ``fold_marker`` is the
    ``r = sqrt(s)`` smoothing parity (0 for scalar ``G`` at every spin, the
    z <-> zbar parity sign x phase for projected ``B``)."""
    from flint import acb

    from virasoro_blocks.derivs import _fold_parity, _fold_plan

    odd = fold_marker % 2
    M_total = m_max + 2 * n_max + odd
    src_dim = M_total + 1
    plan, _ = _fold_plan(bool(odd), src_dim, src_dim, m_max, n_max)
    src_entries = sorted({(m, n) for (m, n, _c) in plan})

    half_h, half_hbar, _, _ = _probe_halves(
        ev_h, ev_hbar, None, None, Mmax, n_terms,
        lambda ev: _make_half_lookup(ev, Mmax, n_terms, b0=b0, match_digits=match_digits),
    )
    zero = acb(0)
    src_rows = _assemble_cluster_rc(
        half_h, half_hbar, groups, ell, Mmax, src_entries, sigma, n_terms, zero=zero,
    )
    src_index = {e: i for i, e in enumerate(src_entries)}

    folded_rows = []
    for row in src_rows:
        # Per Laurent order ``b`` rebuild the non-folded source grid and fold.
        folded_by_b = []  # folded_by_b[b-1][alpha][beta]
        for b in range(1, Mmax + 1):
            grid = [[zero] * src_dim for _ in range(src_dim)]
            for (m, n), ei in src_index.items():
                if m < src_dim and n < src_dim:
                    grid[m][n] = row[ei * Mmax + (b - 1)]
            folded_by_b.append(_fold_parity(grid, fold_marker, m_max, n_max, zero=zero))
        flat = []
        for (a, bb) in folded_entries:
            for b in range(1, Mmax + 1):
                flat.append(folded_by_b[b - 1][a][bb])
        folded_rows.append(flat)
    return folded_rows


def _optionb_chiral_rows_at_probe(ev_h, *, groups, Mmax, entry_pairs, n_terms, b0, match_digits):
    """Per-cluster flattened *chiral* ``F`` block RC rows for ONE probe ``c``, as flint ``acb`` balls.

    A chiral ``F(h)`` table is a *single* chiral half (no antiholomorphic
    factor, no convolution): the degenerate ``[m][0]`` block entry ``(m, 0)`` is
    just the chiral coefficient ``F[m]``, and its merged Laurent at a chiral Kac
    pole ``loc = d`` is that half's Delta-frame principal part directly (here
    ``Delta = h`` so ``loc`` is the Kac weight itself and the half is read at
    ``loc``, not ``(loc +/- ell)/2``; the jacobian is the identity). ``RC[m][b]``
    is therefore the order-``b`` principal coefficient of the ``m``-th chiral
    derivative -- no product, no swap, no regular convolution.
    """
    from flint import acb

    half = _make_half_lookup(ev_h, Mmax, n_terms, jac=1, b0=b0, match_digits=match_digits)
    zero = acb(0)
    rows = []
    for loc, idx in groups:
        Ap, _Bp = half(mp.mpc(loc))
        row = []
        for (m, _n) in entry_pairs:
            row.extend(Ap[b - 1][m] if b - 1 < len(Ap) else zero for b in range(1, Mmax + 1))
        rows.append(row)
    return rows


def _optionb_chiral_rows_series(ev_h, *, groups, Mmax, entry_pairs, n_terms):
    """Series analogue of :func:`_optionb_chiral_rows_at_probe`.

    Reads the exact t-series merged Laurent of a single chiral ``F`` half
    directly (``jac = 1``: ``Delta = h`` for a chiral table, so the half is
    looked up at ``loc`` itself) -- no antiholomorphic factor, no convolution,
    no swap term. Returns the same flattened ``RC[ei * Mmax + (b - 1)]`` rows
    the extrapolation chiral assembler produces, but with ``LaurentSeries``
    coefficients; the caller (:func:`_series_rows_constant_part`) reads off the
    ``t^0`` constant part.
    """
    # a chiral table reads only the principal rows: no regular part
    half = _make_half_lookup_series(
        ev_h, Mmax, n_terms, jac=mp.mpf(1), reg_order_for={}, default_reg=0,
    )
    n_entry = len(entry_pairs)

    def _coef(rows, k, i):
        if k < 0 or k >= len(rows):
            return LaurentSeries.zero()
        return rows[k][i]

    rows = []
    for loc, idx in groups:
        loc = mp.mpc(loc)
        Ap, _Bp = half(loc)
        RC = [[LaurentSeries.zero()] * (Mmax + 1) for _ in range(n_entry)]
        for ei, (m, _n) in enumerate(entry_pairs):
            for b in range(1, Mmax + 1):
                RC[ei][b] = _coef(Ap, b - 1, m)
        row = []
        for ei in range(n_entry):
            row.extend(RC[ei][1:Mmax + 1])
        rows.append(row)
    return rows


class _RegularizedNumericSampler(_PrincipalPartScoring):
    """Picklable numeric-recursion block sampler at a recursion-singular ``c``.

    Drop-in for the non-regularized numeric samplers' ``(deltas, entries) ->
    (tables, is_arb)`` interface (so it can be attached as a
    ``DeltaDerivTable.numeric_sampler`` and consumed by
    ``_evaluate_table_flint_many``): the generic :func:`build_numeric_sampler` built AT the singular ``c`` under
    :func:`~virasoro_blocks.singularity.merged_singular_evaluators`, whose chiral H coefficients are the exact
    merged-pole form of the Laurent-in-t recursion (it replaced epsilon-extrapolation in c, which needed five probe
    builds per grid; the two agree to the working precision for every basis and coordinate).  Holds only plain
    construction data (the ``build_numeric_sampler`` kwargs minus ``c``, plus the target ``c``), so it pickles
    cleanly into interpolation worker processes; the sampler is rebuilt there on first call.
    """

    # No single-c chiral evaluators, so numeric residue pole-ranking cannot run
    # on this sampler; `numeric_residue_scoring_available` consults this flag and
    # falls back to symbolic-numerator residue scoring.
    supports_residue_scoring = False

    def __init__(self, *, c, n_points: int = 5, **sampler_kwargs):
        for name in ('q_order', 'm_max', 'n_max'):
            if name in sampler_kwargs and sampler_kwargs[name] is not None:
                sampler_kwargs[name] = _integer_control(name, sampler_kwargs[name])
        self.c = c
        self.n_points = _integer_control('n_points', n_points, 2)
        # The singular-c principal parts: ``"merged"`` (the default) assembles the rows once from the evaluators AT c0,
        # their exact merged poles read off the one Laurent table; ``"laurent"`` builds the same t^0 rows through the
        # t-series assembly (~5x slower at Lambda 19), ``"pair"`` from a probe pair c0 +/- delta (references).
        self.scorer_method = sampler_kwargs.pop("scorer_method", "merged")
        if self.scorer_method not in ("pair", "laurent", "merged"):
            raise ValueError("scorer_method must be 'merged', 'laurent' or 'pair'")
        self.sampler_kwargs = dict(sampler_kwargs)
        # Last analytic scorer outcome for diagnostics. Populated by
        # _analytic_contribution_scores with status/reason/details.
        self._last_analytic_diag = None
        # Small per-instance cache keyed by (mp.dps, deltas). Stores the most
        # recent grids and entry set so repeated scorer calls can reuse values
        # exactly when points repeat.
        self._cache_key = None
        self._cache_entries = None
        self._cache_entry_set = None
        self._cache_grids = None

    def __getstate__(self):
        # Drop the per-instance grid cache before pickling into interpolation
        # worker processes (mirrors the non-regularized samplers' __getstate__).
        # The cache holds the most recent mpmath grids, which are heavy and
        # purely a re-call optimization, so workers rebuild it on demand.
        state = self.__dict__.copy()
        state.pop("_merged", None)
        state["_cache_key"] = None
        state["_cache_entries"] = None
        state["_cache_entry_set"] = None
        state["_cache_grids"] = None
        state["_last_analytic_diag"] = None
        return state

    def last_analytic_diagnostic(self):
        """Return the most recent analytic scorer diagnostic record."""
        if self._last_analytic_diag is None:
            return None
        return dict(self._last_analytic_diag)

    @staticmethod
    def _delta_cache_key(deltas):
        key = []
        for delta in deltas:
            z = mp.mpc(delta)
            key.append((+z.real, +z.imag))
        return tuple(key)

    @staticmethod
    def _entries_key(entries):
        if entries is None:
            return None
        return tuple(sorted((int(m), int(n)) for (m, n) in entries))

    @staticmethod
    def _extract_entries(grids, entries):
        if entries is None:
            return grids
        out = []
        for grid in grids:
            m_dim = len(grid)
            n_dim = len(grid[0]) if grid else 0
            res = [[mp.mpf(0)] * n_dim for _ in range(m_dim)]
            for (m, n) in entries:
                res[m][n] = grid[m][n]
            out.append(res)
        return out

    @staticmethod
    def _merge_entries(base_grids, extra_grids, extra_entries):
        out = []
        for base, extra in zip(base_grids, extra_grids):
            merged = [list(row) for row in base]
            for (m, n) in extra_entries:
                merged[m][n] = extra[m][n]
            out.append(merged)
        return out

    def _remember(self, key, entries_key, grids):
        self._cache_key = key
        self._cache_entries = entries_key
        self._cache_entry_set = None if entries_key is None else set(entries_key)
        self._cache_grids = grids

    # Coordinates whose non-folded chiral series evaluator exposes Kac
    # ``pole_data`` (the merged-Laurent scorer needs it).
    _MERGED_COORD_EVALUATOR = {
        "zzb": "chiral_z_series_evaluator",
        "yyb": "chiral_y_series_evaluator",
        "qqb": "chiral_q_series_evaluator",
    }
    # Folded (w, s) coordinates expand the *parent* non-folded chiral half
    # (ws -> y, xt -> z, qqb_ws -> q); the parity fold is then a Delta-
    # independent linear recombination applied on top.
    _FOLDED_MERGED_EVALUATOR = {
        "ws": "chiral_y_series_evaluator",
        "xt": "chiral_z_series_evaluator",
        "qqb_ws": "chiral_q_series_evaluator",
    }
    # Parent single-chiral coordinate each folded (w, s) coordinate expands in
    # (used by the t-series geometry, which is keyed by the chiral coordinate).
    _FOLDED_PARENT_COORD = {
        "ws": "y",
        "xt": "z",
        "qqb_ws": "q",
    }
    # Chiral F tables are sampled in the single chiral coordinate z / y / q (a
    # lone holomorphic half, no antiholomorphic factor).
    _CHIRAL_MERGED_EVALUATOR = {
        "z": "chiral_z_series_evaluator",
        "y": "chiral_y_series_evaluator",
        "q": "chiral_q_series_evaluator",
    }
    def _analytic_contribution_scores(self, roots, delta_eval, entries, exp_base):
        """The contribution scores from the principal parts to :func:`ranking_digits`: they only rank poles."""
        from virasoro_blocks.polyderivs import _scores_from_principal_parts

        parts = self._analytic_principal_parts(roots, entries, exp_base, digits=ranking_digits())
        return None if parts is None else _scores_from_principal_parts(parts, len(roots), delta_eval)

    def _analytic_principal_parts(self, roots, entries, exp_base, *, digits=None):
        """Exact principal parts at the poles from MERGED chiral Kac
        Laurent data, or ``None`` to defer to the contour scorer.

        Replaces the numerically-sensitive contour/DFT scorer at
        recursion-singular loci for the diagonal scalar-``G`` block.
        Each Delta-pole is a chiral Kac pole of one (or, at ``ell = 0``, both)
        chiral half ``(Delta +/- ell)/2``; at ``c = 25`` the colliding Kac poles
        are distinct at generic ``b**2``, so a probe ``c = 25 + t**2`` splits
        every cluster into SIMPLE poles. The merged order-``M`` block Laurent
        coefficients are reconstructed from the split poles
        (``RC_a = sum_p sum_{b<=a} C_{p,b} binom(a-1, b-1) (delta_p - loc)^{a-b}``,
        with the per-pole rational coefficients ``C_{p,1}``, ``C_{p,2}`` built
        from the chiral residue/regular streams), then the ``(16 q)^Delta``
        prefactor folds in and the score is ``max_e |sum_a g_{-a} / D**a|``.
        By default (``scorer_method="merged"``), the coefficients are read
        from the merged chiral evaluators at c0. The reference methods use
        the explicit Laurent-in-t assembly (``"laurent"``,
        :func:`_build_series_adaptive`) or a certified symmetric probe pair
        (``"pair"``, :func:`_certified_probe_pair`) to ``digits`` digits
        (default: the working precision; contribution scores request
        :func:`ranking_digits`).

        Handles every block basis and coordinate at a recognized singular ``c``:
        scalar ``G``, ordered ``B``, parity-projected ``B`` (diagonal and
        off-diagonal recenters), and chiral ``F``, on the non-folded
        (zzb/yyb/qqb), folded (ws/xt/qqb_ws) and chiral (z/y/q) coordinates.
        Two-sided ``G`` / ``B`` convolve the holomorphic and antiholomorphic
        chiral halves; the off-diagonal block carries an independent ``n``-axis
        recenter, the folded block applies the Delta-independent parity fold on
        top, and the chiral ``F`` table is the single holomorphic half with no
        convolution. Defers (returns ``None``) at a non-singular ``c`` or an
        unsupported configuration."""
        kw = self.sampler_kwargs
        self._last_analytic_diag = {
            "status": "started",
            "reason": None,
            "c": str(self.c),
            "ell": kw.get("ell"),
            "details": {},
        }

        def _set_diag(status, reason, **details):
            self._last_analytic_diag = {
                "status": status,
                "reason": reason,
                "c": str(self.c),
                "ell": kw.get("ell"),
                "details": details,
            }

        coord = kw.get("coordinate")
        is_chiral = bool(kw.get("chiral"))
        ell_raw = kw.get("ell")
        if is_chiral:
            # A chiral F table has no spin; ell is irrelevant (set 0). The
            # The pole locations are chiral weights ``h`` directly.
            ell = 0
        elif ell_raw is None:
            _set_diag("ineligible", "missing_ell")
            return None
        else:
            ell = int(ell_raw)
        # Off-diagonal (z != zbar) and folded coordinates are handled by
        # dedicated paths; chiral F is a single holomorphic half (no hbar,
        # no convolution). Everything else -- scalar G, ordered B, parity-
        # projected B with independent h/hbar externals on a shared (diagonal)
        # coordinate map -- routes through the two-sided convolution below.
        folded = (not is_chiral) and coord in self._FOLDED_MERGED_EVALUATOR
        if is_chiral:
            if (
                coord not in self._CHIRAL_MERGED_EVALUATOR
                or kw.get("center_bar") is not None
            ):
                _set_diag("ineligible", "unsupported_sampler_configuration",
                          coordinate=coord)
                return None
        elif (
            (coord not in self._MERGED_COORD_EVALUATOR and not folded)
            or kw.get("z_shift") != kw.get("zbar_shift")
        ):
            _set_diag("ineligible", "unsupported_sampler_configuration", coordinate=coord)
            return None
        if folded and (kw.get("center") is not None
                       or kw.get("center_bar") is not None):
            # Folded fold geometry is defined relative to the symmetric center;
            # off-diagonal recenters are non-folded only.
            _set_diag("ineligible", "folded_offdiagonal_unsupported", coordinate=coord)
            return None

        import virasoro_blocks.derivs as _derivs
        from virasoro_blocks.polyderivs import _group_roots_by_location
        from virasoro_blocks.precision import tolerance
        from virasoro_blocks.singularity import recognized_singular_locus

        # Swap-term coefficient: the block entry (m, n) is the ordered product
        # F_h[m] F_hbar[n] plus sigma times the swapped product F_hbar[m] F_h[n].
        #   * scalar G:            sigma = +1, 0 for the public spin-0 block
        #                          (scalar_G_swap_coefficient; as the sampler's values)
        #   * ordered B:           sigma = 0      (no swap term)
        #   * parity-projected B:  sigma = sign(parity) * parity_reversal_phase
        # A chiral F table has no swap term (single half).
        if is_chiral:
            sigma = mp.mpf(0)
            fold_marker = 0
        elif kw.get("projected"):
            from virasoro_blocks.parity import parity_reversal_phase
            parity = kw.get("parity")
            if parity not in ("+", "-"):
                _set_diag("ineligible", "missing_parity")
                return None
            sign = mp.mpf(1) if parity == "+" else mp.mpf(-1)
            phase = parity_reversal_phase(
                kw.get("parity_external"),
                exchanged_parity=kw.get("exchanged_parity") or "1",
            )
            sigma = sign * phase
            # Folded fold parity is sigma's sign (sign x phase), not ell.
            fold_marker = 1 if mp.re(mp.mpc(sigma)) < 0 else 0
        elif kw.get("spinning"):
            sigma = mp.mpf(0)
            fold_marker = ell % 2
        else:
            from virasoro_blocks._conventions import scalar_G_fold_parity, scalar_G_swap_coefficient

            sigma = mp.mpf(scalar_G_swap_coefficient(ell))
            fold_marker = scalar_G_fold_parity(ell)

        # Folded projected B requires a real (+/-1) parity phase: the fold map is
        # a real linear recombination, so a genuinely complex swap phase cannot
        # be folded. Defer to the contour scorer in that (unusual) case.
        if folded and abs(mp.im(sigma)) > mp.mpf("1e-30"):
            _set_diag("ineligible", "folded_complex_parity_phase")
            return None

        h_ext = list(kw.get("h_externals"))
        hbar_ext = list(kw.get("hbar_externals") or kw.get("h_externals"))
        same_side = (
            hbar_ext == h_ext
            and kw.get("zbar_shift") == kw.get("z_shift")
        )
        m_max = int(kw.get("m_max"))
        n_max_raw = kw.get("n_max")
        n_max = 0 if n_max_raw is None else int(n_max_raw)
        if is_chiral:
            n_terms = m_max + 1
        elif folded:
            from virasoro_blocks.derivs import folded_source_order
            n_terms = folded_source_order(m_max, n_max, fold_marker % 2) + 1
        else:
            n_terms = max(m_max, n_max) + 1
        q_order = kw.get("q_order")
        chiral_norm = kw.get("chiral_norm")
        z_shift = kw.get("z_shift")
        zbar_shift = kw.get("zbar_shift")
        eval_name = (
            self._CHIRAL_MERGED_EVALUATOR[coord] if is_chiral
            else self._FOLDED_MERGED_EVALUATOR[coord] if folded
            else self._MERGED_COORD_EVALUATOR[coord]
        )
        center = kw.get("center")
        center_bar = kw.get("center_bar")
        # Off-diagonal / recentered blocks: resolve the m-axis (``center``) and
        # n-axis (``center_bar``) chiral expansion points. ``offdiag`` is True
        # only when the n-axis recenter differs from the m-axis one. A bare
        # ``center`` (with ``center_bar`` None or equal) is a *diagonal* block at
        # an off-symmetric point -- still a single expansion, but recentered.
        from virasoro_blocks.polyderivs import (
            _resolve_center_u, _resolve_center_bar_u, COORDINATES,
        )
        recentered = center is not None or center_bar is not None
        if recentered:
            _z_of_u, u_center = _resolve_center_u(coord, center)
            u_center_bar = _resolve_center_bar_u(coord, center_bar)
            spec = COORDINATES[coord]
            z_of_u = spec.z_of_u
            coord_key = spec.chiral_var
            offdiag = u_center_bar is not None and u_center_bar != u_center
        else:
            u_center = u_center_bar = None
            offdiag = False

        # Only at a recognized recursion-singular c is each Delta-cluster a
        # genuine chiral Kac collision that this factorization can reconstruct.
        # At a generic c the production path never builds this sampler (it uses
        # the ordinary recursion), so decline cleanly -- the merged scorer is
        # not "deferring a case it should handle", it simply does not apply.
        if recognized_singular_locus(mp.mpf(self.c)) is None:
            _set_diag("ineligible", "non_singular_c")
            return None

        roots_mpc = [mp.mpc(r) for r in roots]
        groups = _group_roots_by_location(roots_mpc, tolerance(guard=10))
        Mmax = max((len(idx) for _, idx in groups), default=0)
        entry_pairs = [(int(m), int(n)) for (m, n) in entries]
        n_entry = len(entry_pairs)

        def make_evaluators(c_probe):
            """The chiral evaluators of the probe ``c``: the m-axis halves ``h`` / ``hbar`` and, off-diagonal, the
            n-axis halves recentered at ``center_bar`` (identical externals share one evaluator)."""
            import virasoro_blocks.derivs as _derivs_local

            cp = mp.mpf(c_probe)

            def evaluator(ext, zshift, u_ctr):
                if recentered:
                    ev = _derivs_local._chiral_series_evaluator(
                        ext, n_terms, b=None, c=cp, q_order=q_order, chiral_norm=chiral_norm, z_shift=zshift,
                        z_of_u=z_of_u, u_symmetric=u_ctr, coord_key=coord_key,
                    )
                else:
                    ev = getattr(_derivs, eval_name)(
                        ext, n_terms, b=None, c=cp, q_order=q_order, chiral_norm=chiral_norm, z_shift=zshift,
                    )
                if ev is None:
                    raise _MergedScorerUnavailable
                return ev

            ev_h = evaluator(h_ext, z_shift, u_center)
            if is_chiral:
                return {"h": ev_h}
            ev_hbar = ev_h if same_side else evaluator(hbar_ext, zbar_shift, u_center)
            evs = {"h": ev_h, "hbar": ev_hbar}
            if offdiag:
                evs["h_n"] = evaluator(h_ext, z_shift, u_center_bar)
                evs["hbar_n"] = evs["h_n"] if same_side else evaluator(hbar_ext, zbar_shift, u_center_bar)
            return evs

        def rows_from(evs):
            """Per-cluster flattened RC rows (flint balls) at one probe."""
            b0 = b_from_c(mp.mpf(self.c))
            if is_chiral:
                return _optionb_chiral_rows_at_probe(
                    evs["h"], groups=groups, Mmax=Mmax, entry_pairs=entry_pairs, n_terms=n_terms,
                    b0=b0, match_digits=match_digits,
                )
            if folded:
                return _optionb_persp_rows_at_probe_folded(
                    evs["h"], evs["hbar"], groups=groups, ell=ell, Mmax=Mmax,
                    folded_entries=entry_pairs, sigma=sigma, n_terms=n_terms,
                    fold_marker=fold_marker, m_max=m_max, n_max=n_max, b0=b0, match_digits=match_digits,
                )
            return _optionb_persp_rows_at_probe(
                evs["h"], evs["hbar"], groups=groups, ell=ell, Mmax=Mmax,
                entry_pairs=entry_pairs, sigma=sigma, n_terms=n_terms, b0=b0, match_digits=match_digits,
                ev_h_n=evs.get("h_n"), ev_hbar_n=evs.get("hbar_n"),
            )

        def build_series():
            series_coord = (
                self._FOLDED_PARENT_COORD[coord] if folded else coord
            )
            ev_h = _series_chiral_evaluator(
                h_ext, n_terms, c0=mp.mpf(self.c), q_order=q_order,
                coord=series_coord, chiral_norm=chiral_norm, z_shift=z_shift,
                u_center=u_center,
            )
            if is_chiral:
                # Single holomorphic half (no convolution, no swap term).
                rows = _optionb_chiral_rows_series(
                    ev_h, groups=groups, Mmax=Mmax, entry_pairs=entry_pairs,
                    n_terms=n_terms,
                )
                return _series_rows_constant_part(rows, c0=mp.mpf(self.c))
            ev_hbar = ev_h if same_side else _series_chiral_evaluator(
                hbar_ext, n_terms, c0=mp.mpf(self.c), q_order=q_order,
                coord=series_coord, chiral_norm=chiral_norm, z_shift=zbar_shift,
                u_center=u_center,
            )
            if folded:
                # Folded rows are returned already reduced to t^0 constants (the
                # parity fold is applied on the source constant parts).
                return _optionb_persp_rows_series_folded(
                    ev_h, ev_hbar, groups=groups, ell=ell, Mmax=Mmax,
                    folded_entries=entry_pairs, sigma=sigma, n_terms=n_terms,
                    fold_marker=fold_marker, m_max=m_max, n_max=n_max,
                    c0=mp.mpf(self.c),
                )
            if offdiag:
                # n-axis halves recentered at ``center_bar``.
                ev_h_n = _series_chiral_evaluator(
                    h_ext, n_terms, c0=mp.mpf(self.c), q_order=q_order,
                    coord=series_coord, chiral_norm=chiral_norm, z_shift=z_shift,
                    u_center=u_center_bar,
                )
                ev_hbar_n = ev_h_n if same_side else _series_chiral_evaluator(
                    hbar_ext, n_terms, c0=mp.mpf(self.c), q_order=q_order,
                    coord=series_coord, chiral_norm=chiral_norm,
                    z_shift=zbar_shift, u_center=u_center_bar,
                )
            else:
                ev_h_n = ev_hbar_n = None
            rows = _optionb_persp_rows_series(
                ev_h, ev_hbar, groups=groups, ell=ell, Mmax=Mmax,
                entry_pairs=entry_pairs, sigma=sigma, n_terms=n_terms,
                ev_h_n=ev_h_n, ev_hbar_n=ev_hbar_n,
            )
            return _series_rows_constant_part(rows, c0=mp.mpf(self.c))

        # Past the eligibility gate we are committed: at a recognized singular c with a supported configuration the
        # reconstruction must succeed, so any failure FAILS LOUD (``_MergedScorerUnavailable``: the chiral evaluator
        # has no Kac data here; :class:`ProbePairPrecisionError`: the pair did not certify its digits) rather than
        # routing to the contour scorer. The merged evaluator is the default;
        # the explicit t-series build and probe pair remain test references.
        use_series_scorer = self.scorer_method == "laurent"
        # "Decide once" whether the whole t-series build is real, so the arb
        # backing fires for every LaurentSeries (from_mpc skips its per-handle
        # scan, batch contractions run in arb). Real iff the Kac data is real --
        # b(c0) real (c0 >= 25) or purely imaginary (c0 <= 1, carried as the real
        # beta = -i b) -- AND every external/shift/swap scalar feeding the
        # geometry and recursion is real.
        # Conservative: any complex (or unknown) input keeps it off, falling back
        # to the per-handle realness scan (still correct, just acb batches).
        series_real = _series_build_real(
            self.c,
            (*h_ext, *hbar_ext, sigma, z_shift, zbar_shift, chiral_norm,
             u_center, u_center_bar),
        )
        target_digits = mp.mp.dps if digits is None else int(digits)
        # cluster limits agree to the working precision; distinct ones are a finite distance apart
        match_digits = mp.mp.dps // 2
        try:
            if self.scorer_method == "merged":
                # the pair's row assembly, once, on the evaluators AT c0 (merged poles from the one Laurent table)
                from .singularity import merged_singular_evaluators

                with merged_singular_evaluators():
                    rows = rows_from(make_evaluators(mp.mpf(self.c)))
                # a ball containing 0 is a structural zero at the working precision (a parity-odd block's vanishing
                # entries): exactly 0, not its rounding -- split off at c <= 1 next to an order-12 pole, the rounding
                # of a zero entry (1e-70 of its location) reached the table as 1e-11
                grid = [[mp.mpc(0) if (v.real.contains(0) and v.imag.contains(0)) else +_acb_to_mpc(v) for v in row]
                        for row in rows]
            elif use_series_scorer:
                grid = _build_series_adaptive(
                    build_series, c0=mp.mpf(self.c), q_order=q_order, ell=ell,
                    real=series_real,
                )
            else:
                grid = _certified_probe_pair(
                    make_evaluators, rows_from, c0=mp.mpf(self.c), digits=target_digits,
                    label=f"singular-c principal parts (c = {self.c}, coordinate {coord}, ell {ell})",
                )
        except _MergedScorerUnavailable as exc:
            _set_diag("failed", "chiral_laurent_unavailable")
            raise RuntimeError(
                f"analytic merged-Laurent scorer: the chiral evaluator could "
                f"not supply Kac/pole data at singular c = {self.c} "
                f"(coordinate {coord}, ell {ell}); this singular locus needs "
                f"support, not a silent contour fallback."
            ) from exc

        # No block-order / collapse gate. The singular-c eligibility gate above
        # already rejects generic-c, strip-regime non-Kac block poles. At a recognized singular c
        # the reconstruction is a genuine chiral Kac factorization and is scored
        # directly. A cluster with an order *mismatch* against the skeleton
        # multiplicity (degenerate externals, e.g. the Ising sigma h = 1/16) or
        # one that is identically zero (a kept pole with no residue for this
        # entry set) is scored faithfully -- both reproduce the contour to
        # ranking grade (verified). Any genuine inability to reconstruct surfaces
        # as a loud failure above (``_MergedScorerUnavailable`` /
        # non-convergence), never as a silent contour fallback.
        base = mp.mpc(exp_base)
        Lc = mp.log(base)
        parts = []
        for gi, (loc, idx) in enumerate(groups):
            loc = mp.mpc(loc)
            M = len(idx)
            ebloc = base ** (-loc)
            row = grid[gi]
            rows = []
            for ei in range(n_entry):
                RC = row[ei * Mmax: ei * Mmax + Mmax]
                g_row = []
                for a in range(1, M + 1):
                    g = mp.mpc(0)
                    fact = mp.mpf(1)
                    for k in range(0, M - a + 1):
                        g += ((-Lc) ** k / fact) * RC[a + k - 1]
                        fact *= (k + 1)
                    g_row.append(ebloc * g)
                rows.append(tuple(g_row))
            parts.append((loc, idx, rows))
        _set_diag("analytic-ok", "series" if use_series_scorer else "pair")
        return parts

    def _merged_sampler(self):
        """The numeric sampler AT the singular c (merged-pole H, exact), built once per working precision."""
        from .singularity import merged_singular_evaluators

        cached = getattr(self, "_merged", None)
        if cached is None or cached[0] != mp.mp.prec:
            with merged_singular_evaluators():
                cached = (mp.mp.prec, build_numeric_sampler(c=self.c, **self.sampler_kwargs))
            self._merged = cached
        return cached[1]

    @_quiet_conditioning
    def __call__(self, deltas, entries=None):
        """Block ``[m][n]`` grids at ``deltas``: the generic numeric sampler evaluated at the singular c itself, its H
        coefficients the exact merged-pole form of the one Laurent-in-t table (no probes, no extrapolation in c)."""
        from .singularity import merged_singular_evaluators

        with merged_singular_evaluators():
            return self._merged_sampler()(list(deltas), entries=entries)


def F_block_regularized(h, z, h_ext: Sequence, *, b=None, c=None, q_order: int, chiral_norm=None, z_shift=None,
                        kac_limit: str | None = None):
    """F_block at a possibly recursion-singular c (c = 1, c = 25, the minimal models, the rational-b^2 loci above 25),
    exactly: the block prefactor times the merged-pole H series (:func:`_merged_H_balls`) -- the block at the
    singular ``c``: on a Kac weight where poles collide the block of the irreducible module,
    elsewhere the merged form, and a :class:`~virasoro_blocks.requests.BlockPoleError` on a weight that keeps its pole.  The sum
    ``H(q)`` is formed in balls (``q`` with its rounding radius), so its radius bounds its error, and it is verified
    to the working precision by that bound (:func:`_verified_merged`).  At a generic ``c`` it is ``F_block``.  Pass at
    least one of ``b``, ``c``.  ``kac_limit`` (:func:`~virasoro_blocks.recursion.kac_limit_mode`): ``"irreducible"``
    (the default) the block of the irreducible module on a Kac weight
    (:func:`~virasoro_blocks.recursion.H_q_series_kac_irreducible`), ``"c_limit"`` the c -> c0 limit at fixed ``h``
    there (they differ where poles of one level merge, every Kac weight at c = 1 and 25)."""
    from .recursion import kac_limit_mode
    from .singularity import exact_locus_c, snap_float_c

    kac_limit = kac_limit_mode(kac_limit)
    q_order = _integer_control('q_order', q_order)
    _resolve_c(b, c)                                        # (neither b nor c: ValueError)
    c = _c_input_of_b(b) if c is None else snap_float_c(c)
    c0 = exact_locus_c(c, q_order=q_order)
    if c0 is None:
        # not on a locus to roundoff (possibly near one): the bare recursion, whose near-singular guard measures the
        # collision loss; the tolerant locus recognizers would refuse an mpf c near a locus here
        from .numeric_input import parse_number
        from .recursion import _bare_F_block

        c = _resolve_c(b, c)
        return _bare_F_block(parse_number(h), parse_number(z), tuple(parse_number(v) for v in h_ext),
                             b=b_from_c(c), c=c, q_order=q_order, chiral_norm=chiral_norm, z_shift=z_shift)
    from .numeric_input import parse_number

    h, z = parse_number(h), parse_number(z)
    h_ext = tuple(parse_number(value) for value in h_ext)
    return _F_block_regularized_at_locus(
        h, z, h_ext, c0=c0, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift, kac_limit=kac_limit,
    )


def _F_block_regularized_at_locus(h, z, h_ext, *, c0, q_order, chiral_norm, z_shift, kac_limit=None):
    """Evaluate at an already classified charge and resolved physical point."""
    h_ext = list(h_ext)

    def F_of(balls):
        from flint import acb, arb

        from .recursion import _exact_to_mp

        zz = _exact_to_mp(z)
        q = q_from_z(zz)
        # the prefactor at the precision of this build (as the bare path's certified value), with its rounding
        prefactor = block_prefactor(_exact_to_mp(h), zz, [_exact_to_mp(x) for x in h_ext], c=_exact_to_mp(c0),
                                    chiral_norm=chiral_norm, z_shift=z_shift)
        real = all(isinstance(v, arb) for v in balls) and isinstance(q, mp.mpf) and isinstance(prefactor, mp.mpf)
        if real:
            ball = lambda x: arb(_mpf_to_arb(x), _mpf_to_arb(abs(x) * mp.eps))
            total = arb(0)
        else:
            def ball(x):
                x = mp.mpc(x)
                rad = _mpf_to_arb(abs(x) * mp.eps)
                return acb(arb(_mpf_to_arb(x.real), rad), arb(_mpf_to_arb(x.imag), rad))
            balls, total = [acb(v) for v in balls], acb(0)
        q_ball = ball(q)
        for coefficient in reversed(balls):
            total = total * q_ball + coefficient
        return [ball(prefactor) * total]

    return _verified_merged(h, h_ext, c0=c0, q_order=q_order, reduce=F_of, kac_limit=kac_limit)[0]


def B_block_regularized(h, hbar, z, zbar, h_externals: Sequence, hbar_externals: Sequence, *, b=None, c=None,
                        q_order: int, chiral_norm=None, kac_limit: str | None = None):
    """Ordered (h, hbar) block B = F(h, z) F(hbar, zbar) at a possibly recursion-singular c, from the exact chiral
    factors (:func:`F_block_regularized`; ``kac_limit`` as there)."""
    return B_block(h, hbar, z, zbar, h_externals, hbar_externals, b=b, c=c, q_order=q_order,
                   chiral_norm=chiral_norm, regularize="on", kac_limit=kac_limit)


def G_block_regularized(Delta, ell: int, z, zbar, h_externals: Sequence, *, b=None, c=None, q_order: int,
                        chiral_norm=None, kac_limit: str | None = None):
    """Parity-symmetrized (Delta, ell) scalar-external block G at a possibly recursion-singular c, from the exact
    chiral factors (:func:`F_block_regularized`; ``kac_limit`` as there)."""
    return G_block(Delta, ell, z, zbar, h_externals, b=b, c=c, q_order=q_order, chiral_norm=chiral_norm,
                   regularize="on", kac_limit=kac_limit)


def _merged_H_balls(h, h_ext: Sequence, *, c0, q_order: int, kac_limit: str | None = None) -> list:
    """[H_0, ..., H_{q_order}] at the recognized singular ``c0`` as flint balls at the WORKING precision, from the one
    merged-pole context (:func:`~virasoro_blocks.recursion.build_merged_H_context`) every singular-c step of a block
    reads; each radius bounds its error (exact rationals enter as balls containing them).
    :class:`~virasoro_blocks.requests.BlockPoleError` on a weight that keeps its pole: on a merged pole location, or
    on a Kac location whose c -> c0 limit keeps a negative power of t, read from valuations at this precision, where a
    hidden structural zero reads the same way: :class:`_UncertainPole` asks for one build at doubled precision."""
    from flint import acb

    from .numeric_input import parse_number
    from .recursion import (_exact_to_mp, build_merged_H_context, c_limit_shortfall, eval_merged_H, kac_limit_mode,
                            merged_kac_hit)
    from .requests import BlockPoleError

    kac_limit = kac_limit_mode(kac_limit)
    h = parse_number(h)
    weight = h if isinstance(h, Fraction) else mp.convert(h)
    ctx = build_merged_H_context(q_order, c0, list(h_ext))

    def pole_message():
        hh = _mpf_to_acb(mp.mpc(_exact_to_mp(h)))
        orders = [a for d, a in zip(ctx["d0"], ctx["power"]) if (hh - acb(d)).contains(0)]
        return (
            f"the block has a genuine pole at this exchanged weight: h = {mp.nstr(_exact_to_mp(h), 12)} is a Kac pole"
            + (f" of order {max(orders)}" if orders else "")
            + f" at c = {mp.nstr(_exact_to_mp(c0), 12)} for these externals (the channel is genuinely singular: a "
            "null state resonates with the exchanged weight and does not decouple; degenerate externals must be "
            "passed exactly)")

    try:
        H = eval_merged_H(weight, ctx, q_order, kac_limit=kac_limit)
    except BlockPoleError as exc:
        if kac_limit != "irreducible":
            raise
        # the irreducible block decided exactly (from the fusion factors) that L(h) does not appear
        raise BlockPoleError(f"{pole_message()}: {exc}", family=exc.family, exchanged=exc.exchanged,
                             pole=exc.pole) from None
    if H is None:
        if kac_limit == "irreducible" and merged_kac_hit(weight, ctx):
            # the Kac-curve table keeps a negative power at this precision: rounding of an exact cancellation, or a
            # null vector of L(h) these externals do not decouple -- one build at doubled precision decides
            raise _UncertainPole(
                f"{pole_message()}: no irreducible block, a null vector of L(h) does not decouple for these "
                "externals", mp.mp.dps)
        message = pole_message()
        # on a Kac location the c -> c0 limit keeps a negative power of t, read from the table's valuations at this
        # precision: decided here unless the table lost more digits than the valuation tolerance keeps, else by a
        # build at the measured shortfall; off one, h sits on a merged pole location
        if merged_kac_hit(weight, ctx):
            # The surviving power was read from valuations at this precision: a structural zero that rounding hid
            # reads the same way.  The t^0 balls do not bound those coefficients, so decide by one build at doubled
            # precision (or at the t^0 shortfall when that is larger).
            raise _UncertainPole(message, max(mp.mp.dps, c_limit_shortfall(weight, ctx, q_order)))
        raise BlockPoleError(message)
    return H


class _UncertainPole(ValueError):
    """A pole read from the valuations of a table short of the working precision by ``short`` digits (measured from
    its balls): a structural zero and rounding are not told apart there."""

    def __init__(self, message, short: int):
        super().__init__(message)
        self.short = int(short)


def _ball_mid(v):
    return _acb_to_mpc(v) if type(v).__name__ == "acb" else _arb_to_mpf(v)


def _merged_H_coefficients(h, h_ext: Sequence, *, c0, q_order: int) -> list:
    """:func:`_merged_H_balls` read as mpmath numbers at the working precision (their digits are the caller's to
    verify).  A pole read from a table short of the precision is decided by one build at the measured shortfall;
    :class:`~virasoro_blocks.requests.BlockPoleError` on a pole."""
    from .requests import BlockPoleError

    try:
        balls = _merged_H_balls(h, h_ext, c0=c0, q_order=q_order)
    except _UncertainPole as exc:
        with mp.workdps(mp.mp.dps + exc.short):
            try:
                balls = _merged_H_balls(h, h_ext, c0=c0, q_order=q_order)
            except _UncertainPole as again:
                raise BlockPoleError(str(again)) from None
        return [+_ball_mid(v) for v in balls]
    return [_ball_mid(v) for v in balls]


#: per (c0, externals, q_order, precision): the digits a merged-pole value lost (its balls' measured amplification)
_MERGED_LOSS: dict = {}


def _verified_merged(h, h_ext: Sequence, *, c0, q_order: int, reduce=None, kac_limit: str | None = None) -> list:
    """``reduce(balls)`` (default the H coefficients themselves) of :func:`_merged_H_balls`, verified to the working
    precision ``P`` by the balls' own radii, and read at ``P``.

    The table runs in Arb with every exact input a ball containing it, so a radius bounds the error (checked against
    builds at 4x the precision: the error stayed under 0.3 of the radius).  The radius is the rounding unit times an
    amplification set by the table, not by the precision, so one build measures the loss: the first runs at ``P``,
    and if it falls short (``radius > 2^-prec max(1, |value|)``, the working precision's unit, as the bare value is
    certified) the rebuild runs at ``P`` plus the measured loss.  The
    loss is cached per table, so later calls build once.  An escalation whose bound does not shrink raises.  A pole
    read from a table short of the precision (:class:`_UncertainPole`) is decided by one rebuild at the measured
    shortfall; one that persists is a :class:`~virasoro_blocks.requests.BlockPoleError`."""
    from .numeric_input import parse_number

    h = parse_number(h)
    h_ext = [parse_number(value) for value in h_ext]
    from .requests import BlockPoleError
    from .singularity import _value_key

    from .recursion import kac_limit_mode

    kac_limit = kac_limit_mode(kac_limit)
    base = mp.mp.dps
    target = mp.eps         # certified to the working precision's bits, as the bare value (relative, unit floor)
    exact = lambda x: ("frac", x.numerator, x.denominator) if isinstance(x, Fraction) else _value_key(x)
    key = (str(c0), tuple(exact(x) for x in h_ext), int(q_order), mp.mp.prec, kac_limit)
    guard = _MERGED_LOSS.get(key, 0)
    previous, retried = None, False
    while True:
        with mp.workdps(base + guard):
            try:
                balls = _merged_H_balls(h, h_ext, c0=c0, q_order=q_order, kac_limit=kac_limit)
            except _UncertainPole as exc:
                if retried:
                    raise BlockPoleError(str(exc)) from None
                retried, guard = True, guard + exc.short
                continue
            out = reduce(balls) if reduce is not None else balls
            worst = max((_ball_radius(v) / max(1, abs(_ball_mid(v))) for v in out), default=mp.mpf(0))
        if worst <= target:
            break
        # every digit lost: the loss is at least the precision carried, so carry that much more
        short = base + guard if not mp.isfinite(worst) else int(mp.ceil(mp.log10(worst / target)))
        if previous is not None and mp.isfinite(worst) and worst >= previous:
            raise ArithmeticError(f"the block at c = {c0}: the merged-pole error bound does not shrink with the "
                                  "precision")
        previous = worst
        guard += max(1, short)
    _MERGED_LOSS[key] = max(guard, _MERGED_LOSS.get(key, 0))
    return [+_ball_mid(v) for v in out]


def _ball_radius(v) -> mp.mpf:
    if type(v).__name__ == "acb":
        return mp.mpf(v.real.rad()) + mp.mpf(v.imag.rad())
    return mp.mpf(v.rad())


def H_q_series_regularized(h, h_ext: Sequence, *, b=None, c=None, q_order: int, kac_limit: str | None = None) -> list:
    """[H_0, ..., H_{q_order}] at a possibly recursion-singular c, exactly, to the caller's precision.

    At a recognized singular ``c`` the coefficients are the merged-pole form of the one Laurent-in-t recursion table
    (:func:`~virasoro_blocks.recursion.build_merged_H_context`): ``H_k(h) = sum M[k, (d0, a)] / (h - d0)^a``.  It is
    the block as a function of ``h`` at the singular ``c``, which needs no special care near a Kac weight.  ON a Kac
    weight where poles collide it is the block of the irreducible module
    (:func:`~virasoro_blocks.recursion.H_q_series_kac_irreducible`), which differs from the limit in ``h`` there
    (``recursion.eval_merged_H``).  A Kac weight that keeps a pole raises :class:`~virasoro_blocks.requests.BlockPoleError` naming its order.  The recursion's
    cancellations cost digits; the table runs in balls, so each coefficient is verified to the caller's precision by
    its own error bound (:func:`_verified_merged`).  At any other ``c`` it is
    :func:`~virasoro_blocks.recursion.H_q_series`.

    ``kac_limit="c_limit"``: on a Kac weight, the c -> c0 limit at fixed ``h`` (the fixed-h t^0 of the same table)
    instead; it differs from the irreducible block where poles of one level merge (c = 1, 25).  ``None`` (default):
    :func:`~virasoro_blocks.recursion.kac_limit_mode`.
    """
    from .recursion import kac_limit_mode
    from .singularity import exact_locus_c, snap_float_c

    kac_limit = kac_limit_mode(kac_limit)
    q_order = _integer_control('q_order', q_order)
    _resolve_c(b, c)                                        # (neither b nor c: ValueError)
    c = _c_input_of_b(b) if c is None else snap_float_c(c)  # (a float 0.7 means 7/10)
    c0 = exact_locus_c(c, q_order=q_order)
    if c0 is None:
        return H_q_series(h, h_ext, c=_resolve_c(b, c), q_order=q_order)
    # exact inputs (a Fraction weight, the exact locus c0) enter the table as balls containing them: a degenerate
    # external rounded to the caller's digits is not degenerate (TIM 3/5 at 15 digits: a 1e-16 pole)
    return _verified_merged(h, list(h_ext), c0=c0, q_order=q_order, kac_limit=kac_limit)


def _denominator_from_roots_linear(roots, leading_coeff):
    denominator = Polynomial.one()
    for root in roots:
        denominator = denominator * Polynomial.x_minus(root)
    return denominator.scale(leading_coeff)


def _denominator_from_roots(roots, leading_coeff):
    """Prefer conjugate-pair real reconstruction, fallback to linear product."""
    real_coeffs = real_factor_coeffs_from_conjugate_roots(
        roots, leading_coeff=leading_coeff
    )
    if real_coeffs is not None:
        return Polynomial(real_coeffs, field=mp.mpf)
    return _denominator_from_roots_linear(roots, leading_coeff)


# Memoization for the removable-label set. Keyed on
# (precision, q_order, c_target, externals): independent of coordinate and
# spin, so one entry serves a whole run at a given singular c.
_REMOVABLE_LABELS_CACHE: dict = {}

# Side-analysis cache: same keying as removable labels, but stores both
# K-vanishing removable labels and surviving-pair collision clusters so the
# expensive Laurent rs-table is built only once per (c, q_order, externals).
_SIDE_ANALYSIS_CACHE: dict = {}


def _regularized_chiral_taylor(
    h,
    h_externals: Sequence,
    n_terms: int,
    *,
    c,
    q_order: int,
    z_of_u,
    u_symmetric,
    chiral_norm=None,
    z_shift=None,
    precomputed_state=None,
):
    """Chiral u-Taylor coefficients at a singular ``c``, using the same verified H coefficients as
    :func:`H_q_series_regularized` (on a Kac weight, the block :func:`~virasoro_blocks.recursion.kac_limit_mode`
    selects). The H coefficients do not depend
    on z, so they are formed once and reused across the taylor_coefficients steps, which only vary z.

    ``precomputed_state`` (a dict from :func:`H_q_series_laurent_precompute`) skips the H-table build.

    The coefficients are a function of their arguments (and of the precision, the active ``kac_limit`` and the
    Laurent window), so they are memoised: a cell grid at ``(Delta, ell)`` needs ``h = (Delta + ell)/2`` and ``hbar``,
    and a cut of cells repeats each chiral weight dozens of times (4371 calls for ~100 weights on one c = 1
    dressed-twist space, each rebuilding its verified H series).
    """
    n_terms = _integer_control('n_terms', n_terms, 1)
    q_order = _integer_control('q_order', q_order)
    key = None
    if precomputed_state is None:
        from .recursion import kac_limit_mode
        from .singularity import _value_key

        def k_(x):
            if isinstance(x, Fraction):
                return ("frac", x.numerator, x.denominator)
            return _value_key(mp.convert(x)) if x is not None else None

        try:
            key = (mp.mp.prec, kac_limit_mode(), _ts.window_half(), k_(h), tuple(k_(x) for x in h_externals),
                   int(n_terms), k_(c), int(q_order), z_of_u, k_(u_symmetric), k_(chiral_norm), k_(z_shift))
        except (TypeError, ValueError):
            key = None
        hit = _CHIRAL_TAYLOR_RESULTS.get(key) if key is not None else None
        if hit is not None:
            return list(hit)
    out = _regularized_chiral_taylor_uncached(h, h_externals, n_terms, c=c, q_order=q_order, z_of_u=z_of_u,
                                              u_symmetric=u_symmetric, chiral_norm=chiral_norm, z_shift=z_shift,
                                              precomputed_state=precomputed_state)
    if key is not None:
        if len(_CHIRAL_TAYLOR_RESULTS) >= _CHIRAL_TAYLOR_RESULTS_MAX:
            _CHIRAL_TAYLOR_RESULTS.pop(next(iter(_CHIRAL_TAYLOR_RESULTS)))
        _CHIRAL_TAYLOR_RESULTS[key] = tuple(out)
    return out


#: memoised :func:`_regularized_chiral_taylor` results (FIFO-bounded)
_CHIRAL_TAYLOR_RESULTS: dict = {}
_CHIRAL_TAYLOR_RESULTS_MAX = 8192


def _regularized_chiral_taylor_uncached(h, h_externals, n_terms, *, c, q_order, z_of_u, u_symmetric, chiral_norm,
                                        z_shift, precomputed_state):
    if precomputed_state is not None:
        from .recursion import H_q_series_kac_irreducible, kac_limit_mode, kac_weight_label

        h0 = kac_weight_label(h, c, q_order) if kac_limit_mode() == "irreducible" else None
        H_coeffs = (H_q_series_laurent_eval(h, precomputed_state) if h0 is None
                    else H_q_series_kac_irreducible(h0, h_externals, c0=c, q_order=q_order))
    else:
        H_coeffs = H_q_series_regularized(h, h_externals, c=c, q_order=q_order)
    from .coordinates import _IDENTITY

    if z_of_u is _IDENTITY or z_of_u is z_of_y or z_of_u is z_of_q:     # the chiral variables of the series route
        return _chiral_taylor_from_H(h, H_coeffs, h_externals, n_terms, c=c, z_of_u=z_of_u, u_symmetric=u_symmetric,
                                     chiral_norm=chiral_norm, z_shift=z_shift)

    def evaluate_block(z):
        q = q_from_z(z)
        H_val = mp.mpc(0)
        for k in range(len(H_coeffs) - 1, -1, -1):
            H_val = H_val * q + H_coeffs[k]
        return block_prefactor(h, z, h_externals, c=c, chiral_norm=chiral_norm, z_shift=z_shift) * H_val

    return taylor_coefficients(lambda u: evaluate_block(z_of_u(u)), u_symmetric, n_terms - 1)


#: per (precision, coordinate, point, terms, c, externals, norm, shift): the h-independent u-series of the block
#: (log E, A and the q-power rows), :func:`_chiral_useries`
_CHIRAL_USERIES_CACHE: dict = {}
_CHIRAL_USERIES_CACHE_MAX = 64
#: per chiral context: the bits a :func:`_chiral_taylor_from_H` series lost (measured from its balls)
_CHIRAL_TAYLOR_LOSS: dict = {}


def _chiral_useries(h_externals, n_terms, *, c, z_of_u, u_symmetric, chiral_norm, z_shift, length, prec):
    """``(log E, A, W)`` as flint ``acb_series`` / ``acb_mat`` at ``prec`` bits: ``F(h, z(u)) = exp(h log E) A H(q)``
    with ``E = 16 q / N``, ``A = (16 q)^-sigma z^(sigma + z_shift) (1 - z)^(sigma - h2 - h3) theta_3^theta_exp`` (the
    split of :func:`~virasoro_blocks.recursion.block_prefactor`) and ``W[m][k]`` the ``(u - u_*)^m`` coefficient of
    ``q(z(u))^k``, ``k < length``.  The u-series come from :func:`~virasoro_blocks.polyderivs._flint_taylor_series`
    (the symbolic tables' flint elliptic-K route)."""
    from flint import acb_mat, acb_series, ctx as flint_ctx

    from .numeric_input import parse_number
    from .polyderivs import _flint_taylor_series
    from .polynomial import _mpf_to_acb
    from .recursion import _resolve_chiral_norm
    from .singularity import _value_key

    def k_(x):
        x = parse_number(x)
        return ("frac", x.numerator, x.denominator) if isinstance(x, Fraction) else _value_key(mp.convert(x))

    key = (prec, mp.mp.prec, z_of_u, k_(u_symmetric), int(n_terms), k_(c), tuple(k_(x) for x in h_externals),
           None if chiral_norm is None else k_(chiral_norm), None if z_shift is None else k_(z_shift))
    hit = _CHIRAL_USERIES_CACHE.get(key)
    if hit is not None and hit[2].ncols() >= length:
        return hit
    N = _resolve_chiral_norm(chiral_norm)
    hs = [mp.convert(parse_number(x)) for x in h_externals]
    hs = [x.real if isinstance(x, mp.mpc) and x.imag == 0 else x for x in hs]
    c_eff = mp.convert(parse_number(c))
    sigma = (c_eff - 1) / 24
    theta_exp = (c_eff - 1) / 2 - 4 * sum(hs)
    shift = mp.mpf(0) if z_shift is None else mp.convert(parse_number(z_shift))
    with mp.workdps(int(prec / 3.32) + 1):
        q_s, log_e, a_s = _flint_taylor_series(z_of_u, u_symmetric, n_terms, sigma, theta_exp, shift, hs[1], hs[2], N)
    old_cap = flint_ctx.cap
    flint_ctx.prec, flint_ctx.cap = prec, n_terms
    try:
        q = acb_series([_mpf_to_acb(v) for v in q_s])
        rows, cur = [], acb_series([1])
        for k in range(length):
            if k:
                cur = cur * q
            co = cur.coeffs()[:n_terms]
            rows.append(co + [0] * (n_terms - len(co)))
        W = acb_mat([[rows[k][m] for k in range(length)] for m in range(n_terms)])
        out = (acb_series([_mpf_to_acb(v) for v in log_e]), acb_series([_mpf_to_acb(v) for v in a_s]), W)
    finally:
        flint_ctx.cap = old_cap
    if len(_CHIRAL_USERIES_CACHE) >= _CHIRAL_USERIES_CACHE_MAX:
        _CHIRAL_USERIES_CACHE.pop(next(iter(_CHIRAL_USERIES_CACHE)))
    _CHIRAL_USERIES_CACHE[key] = out
    return out


def _chiral_taylor_from_H(h, H_coeffs, h_externals, n_terms, *, c, z_of_u, u_symmetric, chiral_norm, z_shift):
    """``F(h, z(u))`` Taylor coefficients in ``u - u_*`` (``i = 0 .. n_terms - 1``) from the block's H coefficients,
    by flint power-series arithmetic: ``exp(h log E(u)) A(u) sum_k H_k q(u)^k`` (:func:`_chiral_useries`).

    It replaces the numerical differentiation of ``block_prefactor * H(q(z(u)))`` (``mpmath.diffs``: ~100 block
    evaluations at raised precision per call, the bulk of the c = 1 dressed-twist moment weights).  The result is read
    to the working precision and certified by its balls: a shortfall (the ``h log E`` amplification at large ``h``,
    measured from the radii) is made up by one rebuild at the measured loss.  An imaginary part negligible relative to
    the value is dropped, as :func:`~virasoro_blocks.precision.taylor_coefficients` does."""
    from flint import acb, acb_mat, acb_series, ctx as flint_ctx

    from .numeric_input import parse_number
    from .polynomial import _acb_to_mpc, _mpf_to_acb
    from .precision import _validation_tolerance

    hv = mp.convert(parse_number(h))
    coeffs = list(H_coeffs)
    target = mp.eps
    # build at the working precision plus the loss measured on earlier calls of this context; a shortfall measured
    # from the balls is made up by a rebuild at that many more bits (and remembered)
    loss_key = (int(mp.mp.prec), z_of_u, str(u_symmetric), int(n_terms), str(c), tuple(str(x) for x in h_externals),
                str(chiral_norm), str(z_shift))
    prec = int(mp.mp.prec) + _CHIRAL_TAYLOR_LOSS.get(loss_key, 0)
    old = (flint_ctx.prec, flint_ctx.cap)
    previous = None
    try:
        while True:
            log_e, a_s, W = _chiral_useries(h_externals, n_terms, c=c, z_of_u=z_of_u, u_symmetric=u_symmetric,
                                            chiral_norm=chiral_norm, z_shift=z_shift, length=len(coeffs), prec=prec)
            flint_ctx.prec, flint_ctx.cap = prec, n_terms
            Wk = W if W.ncols() == len(coeffs) else acb_mat([[W[m, k] for k in range(len(coeffs))]
                                                              for m in range(n_terms)])
            Hq = Wk * acb_mat([[_mpf_to_acb(v)] for v in coeffs])
            F = (log_e * _mpf_to_acb(hv)).exp() * a_s * acb_series([Hq[m, 0] for m in range(n_terms)])
            vals = F.coeffs()[:n_terms]
            vals = vals + [acb(0)] * (n_terms - len(vals))
            # each coefficient to the working precision of its own size (a structural zero: of the series scale)
            mags = [mp.mpf(abs(v).mid()) for v in vals]
            floor = max(mags, default=mp.mpf(0)) * target or mp.mpf(2) ** -prec
            worst = max((mp.mpf(v.real.rad() + v.imag.rad()) / max(m, floor) for v, m in zip(vals, mags)),
                        default=mp.mpf(0))
            if worst <= target:
                break
            if previous is not None and worst >= previous:
                raise ArithmeticError("chiral Taylor series: the error bound does not shrink with the precision")
            previous = worst
            prec += int(mp.ceil(mp.log(worst / target, 2))) + 1      # the measured shortfall
    finally:
        flint_ctx.prec, flint_ctx.cap = old
    _CHIRAL_TAYLOR_LOSS[loss_key] = max(_CHIRAL_TAYLOR_LOSS.get(loss_key, 0), prec - int(mp.mp.prec))
    tol = min(100 * mp.eps, _validation_tolerance(guard=5))
    out = []
    for v in vals:
        z = _acb_to_mpc(v)
        out.append(+z.real if abs(z.imag) <= tol * abs(z) else +z)
    return out


def ranking_digits() -> int:
    """Digits the singular-c pole scores are certified to: a quarter of the working precision.

    The scores only order the poles (the greedy's comparisons); two whose scores agree to 10**-(dps/4) are a tie, and
    the probe pair's cancellation loss grows faster than linearly in its target, so certifying more costs precision
    for nothing.  Exact principal parts (split off from a table) take the full working precision instead.
    """
    return max(1, -(-mp.mp.dps // 4))


class ProbePairPrecisionError(ValueError):
    """The symmetric probe pair of a singular-c quantity did not certify its target digits.

    Raised, never silently worked around: the pair's precision is sized from the probe's own measured cancellation
    (:func:`_certified_probe_pair`), so a failure means that sizing rule does not hold here, which must be looked at.
    """


def _probe_pair_offset_digits(digits: int) -> int:
    """``e`` of the probe offset ``10**-e`` for a target of ``digits`` digits.

    The pair average is exact to ``O(offset**2)``; its truncation check (below) models the leftover as
    ``(half-difference / value)**2``, which is pessimistic by the square of the relative ``c``-derivative of the rows
    (``1e1 .. 1e4`` measured: the Kac weights move at rates up to ``r s``).  An eighth of the target digits beyond
    ``digits / 2`` absorbs that without a second pass.
    """
    digits = int(digits)
    return -(-digits // 2) + -(-digits // 8)


# the sizing of the last :func:`_certified_probe_pair` (diagnostics)
_LAST_PROBE_PAIR: dict = {}


def _certified_probe_pair(make_evaluators, rows_from, *, c0, digits, label):
    """Rows of a quantity analytic in ``c`` at the singular ``c0``, from ONE symmetric probe pair ``c0 +/- delta``.

    ``make_evaluators(c)`` builds the chiral evaluators of the probe ``c`` (at the current precision) and returns
    them as a dict; ``rows_from(evaluators)`` computes the rows (lists of flint ``acb`` balls) from them.  The
    average of the two probes is exact to ``O(delta**2)`` (the rows are analytic in ``c``, not just in
    ``t = sqrt(c - c0)``); the half-difference measures the ``O(delta)`` term.

    Precision is chosen once, before the probes are evaluated.  The colliding Kac residues at the probe are
    ``O(delta**-k)`` and cancel in the rows; the probe's certified H-recursion fill measures exactly that
    cancellation depth (the guard it needed beyond the working precision), so one cheap construction of the probe
    evaluators sizes the working precision of the pair.  Every row of every cluster is then certified from its own
    balls: rounding to ``digits`` digits of the row's own scale (no cluster is negligible against another -- the
    greedy weighs scores by pole separations), and the truncation model to the same; a cluster whose midpoints are
    exactly zero at both probes is a structural zero.  A failure raises :class:`ProbePairPrecisionError` -- also
    where the quantity is not analytic in ``c`` at ``c0`` (an exchanged weight on a Kac weight of ``c0``), which a
    pair cannot tell from a cluster vanishing there, so neither is ever returned as a number.

    Returns the rows as mpc midpoints at the caller's precision.
    """
    from flint import ctx as flint_ctx

    from .polynomial import _arb_to_mpf

    base_dps = mp.mp.dps
    tau = int(digits)
    e = _probe_pair_offset_digits(tau)
    c0 = mp.mpf(c0) if not isinstance(c0, mp.mpc) else mp.mpf(c0.real)
    c_digits = max(0, math.ceil(math.log10(max(1.0, abs(float(c0))))))
    # the two probes must be symmetric about c0 to tau digits of the O(delta) term they cancel
    pair_dps = tau + -(-tau // 8) + c_digits + 2

    def evaluators_at(c_probe):
        evs = make_evaluators(c_probe)
        return evs, [ev for ev in {id(ev): ev for ev in evs.values() if ev is not None}.values()]

    with mp.workdps(pair_dps), bare_recursion_probes():
        _, sizing = evaluators_at(c0 + mp.mpf(10) ** (-e))
        guards = [getattr(ev, "certification_guard_bits", None) for ev in sizing]
    if not guards or any(g is None for g in guards):
        raise _MergedScorerUnavailable
    guard_dps = math.ceil(max(guards) / math.log2(10))
    work_dps = pair_dps + guard_dps
    _LAST_PROBE_PAIR.clear()
    _LAST_PROBE_PAIR.update(label=label, digits=tau, offset_digits=e, pair_dps=pair_dps, guard_dps=guard_dps,
                            work_dps=work_dps)

    with mp.workdps(work_dps), bare_recursion_probes():
        delta = mp.mpf(10) ** (-e)
        plus = rows_from(evaluators_at(c0 + delta)[0])
        minus = rows_from(evaluators_at(c0 - delta)[0])
        flint_ctx.prec = int(work_dps * math.log2(10)) + 20
        out = []
        for gi, (rp, rm) in enumerate(zip(plus, minus)):
            avg = [(x + y) / 2 for x, y in zip(rp, rm)]
            scale = max((abs(_acb_to_mpc(v)) for v in avg), default=mp.mpf(0))
            rad = max((_arb_to_mpf(v.rad()) for v in avg), default=mp.mpf(0))
            half = max((abs(_acb_to_mpc((x - y) / 2)) for x, y in zip(rp, rm)), default=mp.mpf(0))
            if scale == 0 and half == 0:
                # exactly zero midpoints at both probes: a structural cancellation (x - x, e.g. the antisymmetric
                # parity projection of identical externals), zero to within the rounding of its own terms
                out.append([mp.mpc(0)] * len(avg))
                continue
            tol = mp.mpf(10) ** (-tau)
            if rad > tol * scale:
                raise ProbePairPrecisionError(
                    f"{label}: cluster {gi} holds {float(-mp.log10(rad / scale)):.1f} of {tau} digits at "
                    f"{work_dps} working digits (probe offset 1e-{e}, H-fill guard {guard_dps} digits)")
            if scale == 0 or (half / scale) ** 2 > tol:
                # includes a quantity that is not analytic in c at c0 (the probes then differ at O(1) or worse:
                # e.g. an exchanged weight ON a Kac weight of c0, where the block itself is singular)
                digits_held = 0.0 if scale == 0 else 2 * float(-mp.log10(half / scale))
                raise ProbePairPrecisionError(
                    f"{label}: cluster {gi} truncation {digits_held:.1f} of {tau} digits (probe offset 1e-{e}; "
                    f"half-difference {mp.nstr(half, 3)} against the average {mp.nstr(scale, 3)}: not analytic in c "
                    "here if the half-difference is not O(offset))")
            out.append([_acb_to_mpc(v) for v in avg])
            worst_digits = min(_LAST_PROBE_PAIR.get('certified_digits', math.inf), float(-mp.log10(rad / scale)) if rad else math.inf)
            _LAST_PROBE_PAIR['certified_digits'] = worst_digits
    return [[+v for v in row] for row in out]


def _collide_cluster(poles, hval):
    """Probe Kac weights that the singular-c degeneracy collapses onto ``hval``.

    At a recursion-singular ``c`` the chiral weights that coincide in the
    ``t -> 0`` limit sit a tiny ``~|c_probe - c|^(1/order)`` away from their
    common limit ``hval`` (here ~1e-13..1e-30), while every genuinely-distinct
    Kac weight stays ``O(0.1..1)`` away -- a physical scale separation of many
    orders of magnitude that does not depend on the probe spacing. So we detect
    the colliding cluster from the data itself rather than a tuned absolute
    window: sort the probe weights by distance to ``hval`` and cut at the
    largest *multiplicative* jump in distance, accepting the near prefix as the
    cluster only when that jump is decisive (``> 1e6``) and the prefix is
    genuinely close (``< 1e-4``) -- both boundaries sit with >9 orders of
    margin inside the gap. Returns the colliding weights (empty when ``hval`` is
    a regular point of the chiral factor).
    """
    # Hoist the constant ``hval`` conversion out of the comparison genexpr and
    # avoid re-wrapping weights that are already mpc (the common case -- callers
    # pass mpc Kac-pole lists); this removes O(n) redundant mpc constructions
    # per call without touching the (precision-critical) distance arithmetic.
    hval_c = hval if isinstance(hval, mp.mpc) else mp.mpc(hval)
    poles_mpc = [w if isinstance(w, mp.mpc) else mp.mpc(w) for w in poles]
    items = sorted(
        ((abs(w - hval_c), w) for w in poles_mpc),
        key=lambda it: it[0],
    )
    if not items:
        return []
    dists = [d for d, _ in items]
    cluster = sum(1 for d in dists if d == 0)  # exact hits are always in
    # The cut is a *coarse* magnitude heuristic: the collision cluster sits
    # ~1e-13..1e-30 from hval while every genuine Kac weight is O(0.1..1), so the
    # decisive multiplicative jump (> 1e6) and the cluster threshold (< 1e-4)
    # both clear by >9 orders of margin. That needs nowhere near working
    # precision, so do the ratio scan in double -- one float() per distance
    # instead of n-1 full mpmath divisions at dps=200 (the dominant mpf_div
    # hotspot of the singular-c scoring build). The exact-hit test stays on the
    # mpf value.
    # (in log10: the cluster distances scale with the regularizer's probe offset, and at working precisions above
    # ~300 digits they underflow a float -- 1e-400 became 0.0 and the ratio divided by zero)
    import math

    ldists = [None if d == 0 else float(mp.log10(d)) for d in dists]
    best_gap = 0.0
    cut = -1
    for j in range(len(dists) - 1):
        if ldists[j] is None or ldists[j + 1] is None:
            continue
        gap = ldists[j + 1] - ldists[j]
        if gap > best_gap:
            best_gap = gap
            cut = j
    if cut >= 0 and best_gap > 6 and ldists[cut] < -4 and not math.isnan(best_gap):
        cluster = max(cluster, cut + 1)
    return [w for _, w in items[:cluster]]


def _skeleton_quotient_by_root_indices(
    table: DeltaDerivTable, root_indices
) -> DeltaDerivTable:
    """Skeleton analogue of `_quotient_table_by_root_indices`.

    A removable Delta-pole is a common factor ``(Delta - root)`` of the
    denominator *and* of every numerator, so dividing it out drops the
    denominator degree by one and every numerator degree by one. With no
    symbolic numerators to divide, the skeleton just removes the root and
    decrements every numerator degree by the number of removed roots. The
    numeric sampler returns the (value-preserving) block value, so it stays
    valid on the quotient."""
    drop = set(root_indices)
    if not drop:
        return table
    keep_roots = [
        root for index, root in enumerate(table.denominator_roots)
        if index not in drop
    ]
    n_removed = len(drop)
    leading_coeff = table.denominator.coeffs[-1]
    denominator = _denominator_from_roots(keep_roots, leading_coeff)
    numerator_degrees = [
        [max(deg - n_removed, 0) for deg in row]
        for row in table.numerator_degrees
    ]
    provenance = (
        None if table.root_provenance is None
        else [
            prov for index, prov in enumerate(table.root_provenance)
            if index not in drop
        ]
    )
    return DeltaDerivTable(
        exp_base=table.exp_base,
        denominator=denominator,
        denominator_roots=keep_roots,
        numerators=None,
        numerator_degrees=numerator_degrees,
        spin=table.spin,
        root_provenance=provenance,
        numeric_sampler=table.numeric_sampler,
    )


_CHIRAL_POLE_ORDER_CACHE: dict = {}


def merged_delta_table(build_table, *, c) -> DeltaDerivTable:
    """A full symbolic table (numerators included) at a recursion-singular ``c``, built AT ``c``: under
    :func:`~virasoro_blocks.singularity.merged_singular_evaluators` the chiral H is the exact rational function of
    the merged poles (:func:`~virasoro_blocks.recursion.merged_H_rational`), so the table is exact at c0 -- no
    probes, no extrapolation in t, no pole strip.  It replaced :func:`regularize_delta_table`."""
    from .singularity import merged_singular_evaluators

    with merged_singular_evaluators():
        table = build_table(c)
    table.requires_pole_strip = False
    return table


def merged_delta_skeleton(build_skeleton, *, c) -> DeltaDerivTable:
    """The skeleton (denominator roots + numerator degrees) of a table at a recursion-singular ``c``, built AT ``c``.

    ``build_skeleton(c_value)`` is the ordinary skeleton builder; under
    :func:`~virasoro_blocks.singularity.merged_singular_evaluators` its chiral denominators are the exact merged pole
    structure of the Laurent table (:func:`~virasoro_blocks.recursion.merged_skeleton_rational`): each merged Kac pole
    at its order after every cancellation.  So no probes, no root extrapolation, no removable-pole or collision strip,
    and no measured order correction at c < 1 -- it replaced :func:`regularize_delta_skeleton` (+
    :func:`measure_and_correct_skeletons`), whose skeletons it reproduces root for root, multiplicities included."""
    from .singularity import merged_singular_evaluators

    with merged_singular_evaluators():
        table = build_skeleton(c)
    table.requires_pole_strip = False
    return table
