"""Large-spin asymptotic blocks: fixed twist, variable spin L.

For the 2d numerical bootstrap, positivity must be imposed on the *continuous*
large-spin tower, not just a finite list of spins. At fixed twist ``tau``
(``Delta = L + tau``; ``h = L + tau/2``, ``hbar = tau/2``) the 2d block
factorizes into chiral halves:

    G_{L,tau}(z, zbar) = k_h(z) k_hbar(zbar) + rev * k_hbar(z) k_h(zbar),

with ``k`` the chiral block (Virasoro or global SL(2)). The reversed-term
coefficient ``rev`` selects the basis: ``ordered_B`` keeps only the first term
(``rev = None``); ``scalar_G`` is the symmetric scalar-external case
(``rev = +1``, z <-> zbar even, so one tower covers every integer
``L >= Lmax``); ``parity_projected_B`` carries ``rev = sign * phase`` from
the external parity eigenstate (`virasoro_blocks.parity`), and its ``'-'``
projection is the z <-> zbar-odd tower. The holomorphic
``{h_i}`` / antiholomorphic ``{hbar_i}`` externals may differ (spinning /
fermionic operators), and ``ordered_B`` / ``parity_projected_B`` admit
half-integer exchanged spin ``L``.

Spin convention: ``L = h - hbar`` is the *signed* 2d spin (the U(1) rotation
charge), and this is the ``L -> +infinity`` family -- the tower grows the
holomorphic weight at fixed antiholomorphic twist ``tau = 2 hbar``; the
negative-spin family is its parity image. Only the ``h = L + tau/2`` half
carries ``L``, so each entry is ``base^L * poly(L)`` with
per-entry power ``max(m, n)`` (``m`` for ``ordered_B``) and the same decay base as the Delta block
(``16 q(1/2)/N`` Virasoro, ``4 rho(1/2)/N`` global). We interpolate it in ``L``
(``shift = Lmax``) with the shared optimal-node machinery and return an
``InterpolatedDeltaTable`` whose positivity variable is the spin ``L``.

Performance: the chiral block is evaluated at many ``h = L + tau/2`` (the
interpolation nodes). The Delta-independent geometry (nome, theta3, BASE'(z))
depends only on ``(externals, c, q_order)`` -- not on ``tau`` or ``h`` -- so we
build one :func:`~virasoro_blocks.derivs.chiral_z_series_evaluator` and reuse it
for every node, and the ``hbar = tau/2`` half is computed once. (Calling
``chiral_z_taylor`` per node instead rebuilds that geometry every time and is
~50x slower.)

Pole structure / accuracy. ``mode='global'`` uses the finite q-order SL(2)
pole universe: a chiral pole ``2h=-p`` is included when it first appears by
elliptic level ``p+1 <= q_order``. The block is still evaluated with exact
2F1 values; q-order only fixes the candidate universe. ``mode='virasoro'``
uses the block's actual Virasoro Kac poles
``d_{r,s}(c) - twist/2`` (``_virasoro_chiral_pole_pool``) -- the SL(2) ladder is
their ``c -> inf`` limit -- recovering several extra digits over the SL(2)
approximation in the tail. The Kac poles are complex (conjugate pairs) for
``1 < c < 25`` and real elsewhere. At a recognized singular c, the pool
comes from the numerator-free merged chiral skeleton. It includes each
surviving Kac location at its order after Laurent cancellation, both for
minimal models and for the positive-rational b-squared family above c = 25.
The genuine poles that
land above ``Lmax`` (the c = 1 Kac tower) are real with even multiplicity, so the
interpolation force-keeps them in the returned prefactor (see
``interpolate_delta_tables`` / ``_interpolate_global_table``
``force_keep_real_positive``) and the prefactor stays sign-definite on the
physical domain. At singular c, contribution scores use merged chiral
principal parts. A finite-radius contour is available for configurations that
the analytic scorer explicitly defers; missing merged pole data raises.
"""
from __future__ import annotations

import mpmath as mp

from .conditioning import quiet as _quiet_conditioning

from virasoro_blocks.coordinates import COORDINATES, q_of_z, q_symmetric
from virasoro_blocks.derivs import (
    _chiral_series_evaluator,
    _global_chiral_series_evaluator,
    _global_chiral_taylor,
    global_chiral_y_taylor,
    global_chiral_z_taylor,
    _assemble_block_grid,
    fold_parity_marker,
    folded_entry_power,
    folded_source_order,
)
from virasoro_blocks.global_blocks import (
    DEFAULT_GLOBAL_Q_ORDER, _externals_diffs,
    _global_chiral_q_order_pole_pool, _interpolate_global_table,
    _make_residue_scorer, global_block_exp_base,
    _validate_q_order, _validate_derivative_order,
)
from virasoro_blocks.interpolation import pole_selection_score_entries, _validate_npoles
from virasoro_blocks.parity import normalize_parity_phase, parity_reversal_phase
from virasoro_blocks.precision import tolerance
from virasoro_blocks.recursion import _resolve_chiral_norm
from virasoro_blocks.singularity import recognized_singular_locus

class _MergedScorerUnavailable(Exception):
    """Internal signal: the merged singular-c residue scorer cannot build its
    chiral pole data (unsupported evaluator mode), so the caller
    defers to the contour scorer."""


LARGE_SPIN_MODES = ("virasoro", "global")
# Non-folded crossing-odd bases plus their parity folds: `xt` folds the `zzb`
# (z, z̄) block, `ws` folds the `yyb` (y, ȳ) block, `qqb_ws` folds the `qqb`
# (q, q̄ nome) block. The fold is an L-independent linear recombination of the
# non-folded outer-product grid (`derivs._fold_parity`), so it composes with the
# spin-L interpolation.
LARGE_SPIN_COORDINATES = (
    "zzb", "yyb", "xt", "ws", "qqb", "qqb_ws",
    "xt_radial", "ws_radial", "qqb_ws_radial",
)
# Folded tables and their radial slices share the same parity map.
_LARGE_SPIN_FOLDED_COORDINATES = tuple(
    coord for coord in LARGE_SPIN_COORDINATES
    if COORDINATES[coord].kind in ("folded_2d", "radial")
)


def _tower_exchange_weight(L, twist):
    """Form h = L + twist/2 before rounding a spin label or its components."""
    from .numeric_input import literal_components

    real, imaginary = literal_components(L)
    real += twist / 2
    if imaginary == 0:
        return real
    return mp.mpc(mp.convert(real), mp.convert(imaginary))
# Underlying non-folded 2d coordinate for each tower coordinate (xt -> zzb),
# from the canonical coordinate registry.
_LARGE_SPIN_CHIRAL_COORD = {
    coord: COORDINATES[coord].base_2d for coord in LARGE_SPIN_COORDINATES
}


def _large_spin_half_base(mode, center, chiral_norm):
    """Per-half per-L decay base at physical center z0."""
    if mode == "virasoro":
        cn = _resolve_chiral_norm(chiral_norm)
        z0 = mp.mpf(1) / 2 if center is None else center
        return mp.mpf(16) * q_of_z(z0) / cn
    if mode == "global":
        return global_block_exp_base(chiral_norm, center=center)
    raise ValueError(f"mode must be one of {LARGE_SPIN_MODES}; got {mode!r}")


def _real_tower_label(value):
    """Keep a real tower input exact until its numerical evaluation."""
    from .numeric_input import literal_number

    label = literal_number(value)
    numeric = mp.mpf(mp.convert(label))
    if isinstance(label, bool) or not mp.isfinite(numeric):
        raise ValueError("tower inputs must be finite real numbers")
    return label


def _virasoro_chiral_pole_pool(*, h_externals, c, q_order, twist):
    """Genuine-pole pool for the **Virasoro** tower: the chiral block's Kac poles.

    The fixed-twist tower interpolates the chiral block ``F(h, z)`` in the spin
    ``L`` with ``h = L + twist/2``, so the block's poles in ``h`` at the Kac
    dimensions ``d_{r,s}(c)`` (``rs <= q_order``, the recursion's truncated set)
    sit at ``L = d_{r,s} - twist/2`` in the interpolation variable -- the
    ``- twist/2`` shift is essential. The Kac dimensions come straight from a
    numerator-free skeleton chiral table (no symbolic numerators built); they
    are complex (in conjugate pairs) for ``1 < c < 25`` and real elsewhere.

    Returns the complete Kac pole set (all roots up to ``q_order``), shifted by
    ``-twist/2``. No proximity pre-filter is applied: the contribution scorer in
    ``_interpolate_global_table`` ranks and caps the full set to ``npoles``,
    ensuring that real (diagonal) Kac singletons are never silently excluded by
    a fixed-count window. The SL(2) tower likewise uses its full q-order-defined
    candidate pool, from ``_global_chiral_q_order_pole_pool``, before selection.
    """
    ext = [_real_tower_label(h) for h in h_externals]
    roots = _virasoro_chiral_roots(tuple(ext), c, q_order)
    twist = mp.mpf(mp.convert(twist))
    return [mp.mpc(r) - twist / 2 for r in roots]


# The chiral Kac-pole set is twist-independent (twist enters only as the final
# ``- twist/2`` shift in :func:`_virasoro_chiral_pole_pool`), but each twist
# channel of the tower rebuilds it -- and the build runs a symbolic chiral
# useries (~seconds at high Lambda). Memoize it across twists, keyed on the
# inputs it actually depends on (externals, c, q_order) plus the working
# precision so a dps change recomputes rather than returning a stale low-precision
# root set.
_VIRASORO_CHIRAL_ROOTS_CACHE = {}


def _virasoro_chiral_roots(ext, c, q_order):
    """Twist-independent genuine Kac-pole locations of the chiral tower block.

    The Kac denominator (hence the pole set) is independent of ``m_max`` and of
    the chiral coordinate; a degree-1 skeleton chiral build is the cheap source.
    At a recognition-singular c the raw recursion is itself singular (and the Kac
    dimensions collide), so the genuine pole set is the *regularized* chiral root
    set -- recovered from the numerator-free regularized skeleton, exactly as the
    fixed-spin path does. Memoized across twists (see the module cache)."""
    from virasoro_blocks.numeric_input import parse_number
    from virasoro_blocks.derivs import _chiral_eval_memo_scalar_key

    c = parse_number(c)
    ext = tuple(_real_tower_label(h) for h in ext)
    key = (
        tuple(_chiral_eval_memo_scalar_key(h) for h in ext),
        _chiral_eval_memo_scalar_key(c),
        int(q_order), mp.mp.prec,
    )
    cached = _VIRASORO_CHIRAL_ROOTS_CACHE.get(key)
    if cached is not None:
        return cached

    ext_list = list(ext)
    if recognized_singular_locus(c, q_order=q_order) is not None:
        # at a recognition-singular c the genuine chiral pole set is the exact merged pole structure of the Laurent
        # table (each merged Kac pole at its order, after every cancellation): no probes, no strips
        from virasoro_blocks.recursion import merged_skeleton_rational

        roots = merged_skeleton_rational(ext_list, c0=c, q_order=q_order).denominator_roots
    else:
        # Generic c: the genuine pole set is exactly the surviving Kac weights --
        # at non-singular c the Kac dimensions are distinct, so the numerator
        # common-factor strip is a no-op and the roots equal the bare Kac data.
        # Read the roots from the Kac pair table in O(n_pairs), without
        # building the numerator H-table.
        from virasoro_blocks.polyblock import chiral_denominator_roots
        roots = chiral_denominator_roots(ext_list, c=c, q_order=q_order)
    roots = tuple(roots)
    _VIRASORO_CHIRAL_ROOTS_CACHE[key] = roots
    return roots


def large_spin_base(mode, chiral_norm, *, center=None, center_bar=None, basis="scalar_G"):
    """Per-unit-L decay base of the spin tower.

    Only the holomorphic half carries L, so off the diagonal the two terms of
    ``k_{L+tau/2}(z) k_{tau/2}(zb) + sigma k_{tau/2}(z) k_{L+tau/2}(zb)`` decay like
    ``e_z^L`` and ``e_zb^L``: the tower decays like the larger base (the swap term
    dominates when ``e_zb > e_z``), and like ``e_z`` for ordered_B (no swap term).
    The geometric mean used before left a numerator growing like ``(max/gm)^L``:
    the table lost digits linearly in L (0.5 digits at L = 42 for 0.4 / 0.45).
    """
    if center_bar is None:
        center_bar = center
    ez = _large_spin_half_base(mode, center, chiral_norm)
    ezb = _large_spin_half_base(mode, center_bar, chiral_norm)
    if ez == ezb or basis in ("ordered_B", "global_ordered_B"):
        return ez
    return ez if abs(ez) >= abs(ezb) else ezb


class _LargeSpinTwistSampler:
    """Picklable sampler ``L -> fixed-twist spin-L block table [m][n]``.

    The Virasoro chiral evaluator and the ``hbar = tau/2`` coefficients are built
    lazily on first call and cached on the instance (so the geometry is built
    once per process, not per node). The cached Virasoro evaluator is a local
    closure (from :func:`chiral_z_series_evaluator`) that cannot be pickled, so
    ``__getstate__`` drops the lazily-built caches; each worker rebuilds them on
    first call after unpickling. This matters because the residue /
    contribution pole selectors evaluate the sampler in the *parent* process
    (priming ``_evaluator``) before the per-node-count buckets are dispatched to
    a ``ProcessPoolExecutor`` -- without the state hook that primed closure
    would make the sampler unpicklable and the parallel build would crash."""

    def __init__(self, mode, coordinate, h_ext, chiral_norm, z_shift, m_max, n_max, twist,
                 c=None, q_order=None, exchanged_parity="1",
                 hbar_externals=None, basis="scalar_G",
                 parity_projection=None, parity_external=None,
                 center=None, center_bar=None):
        self.mode = mode
        self.coordinate = coordinate
        self.h_ext = [_real_tower_label(h) for h in h_ext]
        # Antiholomorphic externals for the h̄ = twist/2 half. Scalar externals
        # (h_i == h̄_i) share one expansion; spinning/fermionic externals carry
        # an independent set. Only the holomorphic half carries L, so the
        # interpolation's pole pool keys off the holomorphic differences.
        self.hbar_ext = (
            list(self.h_ext) if hbar_externals is None
            else [_real_tower_label(h) for h in hbar_externals]
        )
        self._same_externals = self.hbar_ext == self.h_ext
        self.basis = basis
        self.parity = parity_projection
        self.chiral_norm = chiral_norm
        self.z_shift = z_shift
        self.center = center
        self.center_bar = center_bar
        if COORDINATES[coordinate].kind == "radial":
            n_max = 0
        self.m_max = m_max
        self.n_max = n_max
        self.twist = _real_tower_label(twist)
        from virasoro_blocks.numeric_input import parse_number

        self.c = parse_number(c)
        self.q_order = q_order
        self.exchanged_parity = normalize_parity_phase(exchanged_parity)
        # Coefficient of the parity-reversed block in
        # G = B(h,h̄) + rev_coeff * B(h̄,h). `None` means "no reversed partner"
        # (ordered_B). scalar_G is the symmetric special case (rev_coeff = +1 at
        # every spin); parity_projected_B carries sign*phase from the external
        # parity eigenstate. See `_grid_from`.
        self._rev_coeff = self._resolve_rev_coeff(
            basis, self.exchanged_parity, parity_projection, parity_external
        )
        # Parity fold: `xt`/`ws` outputs are a linear recombination of a larger
        # non-folded (m, n) grid. We build the block in the underlying chiral
        # coordinate (`zzb`/`yyb`) to the expanded order `m_max + 2 n_max + odd`
        # in each index, then strip to the folded (m_max, n_max) shape. The
        # fold's smoothing parity `odd` is the block's z <-> z̄ parity: even for
        # scalar_G, sign x phase for parity_projected_B (ordered_B has no folded
        # form).
        self.folded = coordinate in _LARGE_SPIN_FOLDED_COORDINATES
        self.chiral_coordinate = _LARGE_SPIN_CHIRAL_COORD[coordinate]
        if self.folded and (center is not None or center_bar is not None):
            raise ValueError(
                "off-diagonal centers are supported only on non-folded "
                "coordinates (zzb/yyb/qqb)"
            )
        self._centers_prec = None
        self._ensure_center_precision()
        self._fold_odd = self._resolve_fold_odd(
            self.folded, basis, self.exchanged_parity, parity_projection, parity_external
        )
        if self.folded:
            self._src_m = folded_source_order(m_max, n_max, self._fold_odd)
            self._src_n = self._src_m
            self.n_terms = self._src_m + 1
        else:
            self._src_m = m_max
            self._src_n = n_max
            self.n_terms = max(m_max, n_max) + 1
        self._evals = None       # (side, axis, c_val, prec) -> chiral evaluator
        self._hbar_coeffs = None  # (precision, L-independent h̄ = twist/2 coefficients)
        # At a recursion-singular c (minimal models, c = 1) the bare chiral
        # evaluator divides by zero; route those through the same
        # merged singular evaluator the Delta tables use. Global SL(2) blocks are
        # not c-singular, so only the Virasoro path needs it.
        self._singular = (
            mode == "virasoro"
            and c is not None
            and recognized_singular_locus(self.c, q_order=q_order) is not None
        )

    def _ensure_center_precision(self):
        """Recompute derived coordinate centers when the working precision changes."""
        if self._centers_prec == mp.mp.prec:
            return
        spec = COORDINATES[self.chiral_coordinate]
        self._u_center = (
            spec.u_symmetric() if self.center is None else spec.u_of_z(self.center)
        )
        self._u_center_bar = (
            self._u_center if self.center_bar is None else spec.u_of_z(self.center_bar)
        )
        self._offdiag = self._u_center_bar != self._u_center
        self._centers_prec = mp.mp.prec

    @staticmethod
    def _resolve_rev_coeff(basis, exchanged_parity, parity_projection, parity_external):
        if basis == "ordered_B":
            return None  # raw single ordering, no parity partner
        if basis == "parity_projected_B":
            if parity_projection not in {"+", "-"}:
                raise ValueError(
                    "the parity_projected_B tower requires parity_projection "
                    "'+' or '-'"
                )
            sign = mp.mpf(1) if parity_projection == "+" else mp.mpf(-1)
            return sign * parity_reversal_phase(
                parity_external, exchanged_parity=exchanged_parity
            )
        # scalar_G: symmetric, rev_coeff = +1 -- the z <-> zbar-even block at
        # every integer spin, so the one tower is analytic in L and covers them all.
        refuse_scalar_G_tower_exchanged_parity(basis, exchanged_parity)
        return mp.mpf(1)

    @staticmethod
    def _resolve_fold_odd(folded, basis, exchanged_parity, parity_projection, parity_external=None):
        """Smoothing parity for the (w, s) fold: 1 strips one r = sqrt(s) factor
        (odd block), 0 keeps the block as is. Mirrors the fixed-spin folded
        builders -- even for scalar_G, sign x phase for parity_projected_B."""
        if not folded:
            return 0
        if basis == "ordered_B":
            raise ValueError(
                "ordered_B has no parity fold; folded coordinates "
                "('xt', 'ws') require basis 'scalar_G' or 'parity_projected_B'"
            )
        if basis == "parity_projected_B":
            return fold_parity_marker(parity=parity_projection, phase=parity_reversal_phase(
                parity_external, exchanged_parity=exchanged_parity or "1"))
        # scalar_G: z <-> z̄ even at every spin
        return 0

    def __getstate__(self):
        # Drop the lazily-built caches: the Virasoro chiral evaluators are local
        # closures that cannot be pickled (and the h̄ coeffs are cheap to
        # rebuild). Each worker process re-creates them on first call.
        state = self.__dict__.copy()
        state["_evals"] = None
        state["_hbar_coeffs"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _externals_for(self, side):
        return self.h_ext if side == "h" else self.hbar_ext

    def _evaluator_for_center(self, externals, c_val, *, u_center):
        """Batched flint chiral series evaluator (``.many`` + ``return_acb``) for
        one chiral half, mode-aware: the Virasoro Kac-pole series for
        ``mode='virasoro'``, the SL(2) global series for ``mode='global'``. Both
        share the interface consumed by :meth:`_sample_bucket_at_c`."""
        spec = COORDINATES[self.chiral_coordinate]
        if self.mode == "virasoro":
            return _chiral_series_evaluator(
                externals,
                self.n_terms,
                b=None,
                c=c_val,
                q_order=self.q_order,
                chiral_norm=self.chiral_norm,
                z_shift=self.z_shift,
                z_of_u=spec.z_of_u,
                u_symmetric=u_center,
                coord_key=spec.chiral_var,
            )
        return _global_chiral_series_evaluator(
            externals,
            self.n_terms,
            z_of_u=spec.z_of_u,
            u_symmetric=u_center,
            chiral_norm=self.chiral_norm,
            z_shift=self.z_shift,
        )

    def _cached_evaluator(self, side, c_val):
        """Virasoro chiral evaluator for one chiral half at central charge
        ``c_val``, cached by (side, c_val, precision). Scalar externals share
        the holomorphic-side evaluator. Singular charges use the merged
        recursion at that charge, so the expensive geometry is built once
        per side and precision, rather than per spin node."""
        eff_side = "h" if self._same_externals else side
        if self._evals is None:
            self._evals = {}
        key = (eff_side, "m", c_val, mp.mp.prec)
        ev = self._evals.get(key)
        if ev is None:
            ev = self._evaluator_for_center(
                self._externals_for(eff_side), c_val, u_center=self._u_center,
            )
            self._evals[key] = ev
        return ev

    def _cached_evaluator_n(self, side, c_val):
        eff_side = "h" if self._same_externals else side
        if self._evals is None:
            self._evals = {}
        key = (eff_side, "n", c_val, mp.mp.prec)
        ev = self._evals.get(key)
        if ev is None:
            ev = self._evaluator_for_center(
                self._externals_for(eff_side), c_val, u_center=self._u_center_bar,
            )
            self._evals[key] = ev
        return ev

    def _analytic_residue_scores(self, roots, delta_eval, entries, base):
        """Closed-form principal-part contribution scores for the tower, or
        ``None`` to defer to the eps-probe / contour scorer.

        Only the holomorphic half carries the spin (``h = L + twist/2``), so the
        tower's L-poles are simple and built directly from the chiral residue /
        regular series (``evaluate.pole_data``) instead of an eps-probe near the
        Kac pole; the antiholomorphic half is the fixed ``hbar = twist/2`` value.
        Shares :func:`~virasoro_blocks.polyderivs._analytic_pp_scores` with the
        fixed-spin scorers, with ``deriv_factor = 1`` (``h = L + twist/2`` carries
        the variable with unit slope, vs the fixed-spin ``(Δ±ℓ)/2``) and the fixed
        ``hbar`` half evaluated once. ``sigma`` is the parity partner coefficient
        (``self._rev_coeff``: ``None`` for ordered_B). Off-diagonal
        (``u_center != u_center_bar``) the m-index uses the m-center evaluators
        and the n-index the n-center evaluators (four chiral-factor streams; the
        L-carrying holomorphic halves are the m-plus / n-plus streams). Folded
        coordinates (xt / ws) build the non-folded principal-part grids and fold
        them (diagonal, real parity phase only). Both the Virasoro tower (Kac
        ``pole_data``) and the global SL(2) tower (hypergeometric ``pole_data``,
        simple ``h = -p/2`` poles, pool placed at the true L-poles ``-p/2 -
        twist/2``) are covered. At a recognition-singular c the Virasoro
        tower routes to :meth:`_singular_merged_residue_scores` (merged
        higher-order chiral Laurent at the Kac collision), which itself covers
        the diagonal, off-diagonal, and folded coordinates. Also defers for a
        folded complex parity phase."""
        self._ensure_center_precision()
        if self.mode not in ("virasoro", "global"):
            return None
        if self._singular:
            return self._singular_merged_residue_scores(
                roots, delta_eval, entries, base
            )
        from virasoro_blocks.polyderivs import (
            _analytic_chiral_factor, _analytic_pp_scores,
            _analytic_pp_scores_axis, _analytic_pp_scores_folded,
        )

        # The global SL(2) tower is c-independent (``self.c`` may be ``None``);
        # the cached evaluator ignores ``c_val`` in that mode.
        c_val = self.c
        tw = self.twist

        if self.folded:
            # Only the holomorphic half carries L; the source grid is the same
            # diagonal a[m]c[n] + rev_coeff c[m]a[n] as the non-folded tower,
            # folded by the L-independent linear map. Real parity phase only.
            if isinstance(self._rev_coeff, mp.mpc) and self._rev_coeff.imag != 0:
                return None
            eval_h = self._cached_evaluator("h", c_val)
            eval_hbar = self._cached_evaluator("hbar", c_val)
            fac_h = _analytic_chiral_factor(eval_h, self.n_terms, deriv_factor=1)
            fac_hbar = (
                fac_h if eval_hbar is eval_h
                else _analytic_chiral_factor(eval_hbar, self.n_terms, deriv_factor=1)
            )
            if fac_h is None or fac_hbar is None:
                return None
            return _analytic_pp_scores_folded(
                roots, delta_eval, entries, base,
                factor_m_plus=lambda loc: fac_h(loc + tw / 2),
                factor_m_minus=lambda loc: fac_hbar(tw / 2),
                factor_n_plus=lambda loc: fac_h(loc + tw / 2),
                factor_n_minus=lambda loc: fac_hbar(tw / 2),
                sigma=self._rev_coeff, fold_marker=self._fold_odd,
                m_max=self.m_max, n_max=self.n_max,
            )

        if self._offdiag:
            eval_h_m = self._cached_evaluator("h", c_val)
            eval_h_n = self._cached_evaluator_n("h", c_val)
            eval_hbar_m = self._cached_evaluator("hbar", c_val)
            eval_hbar_n = self._cached_evaluator_n("hbar", c_val)
            fac_h_m = _analytic_chiral_factor(eval_h_m, self.n_terms, deriv_factor=1)
            fac_h_n = _analytic_chiral_factor(eval_h_n, self.n_terms, deriv_factor=1)
            fac_hbar_m = (
                fac_h_m if eval_hbar_m is eval_h_m
                else _analytic_chiral_factor(eval_hbar_m, self.n_terms, deriv_factor=1)
            )
            fac_hbar_n = (
                fac_h_n if eval_hbar_n is eval_h_n
                else _analytic_chiral_factor(eval_hbar_n, self.n_terms, deriv_factor=1)
            )
            if None in (fac_h_m, fac_h_n, fac_hbar_m, fac_hbar_n):
                return None
            return _analytic_pp_scores_axis(
                roots, delta_eval, entries, base,
                factor_m_plus=lambda loc: fac_h_m(loc + tw / 2),
                factor_m_minus=lambda loc: fac_hbar_m(tw / 2),
                factor_n_plus=lambda loc: fac_h_n(loc + tw / 2),
                factor_n_minus=lambda loc: fac_hbar_n(tw / 2),
                sigma=self._rev_coeff,
            )

        eval_h = self._cached_evaluator("h", c_val)
        eval_hbar = self._cached_evaluator("hbar", c_val)
        fac_h = _analytic_chiral_factor(eval_h, self.n_terms, deriv_factor=1)
        fac_hbar = (
            fac_h if eval_hbar is eval_h
            else _analytic_chiral_factor(eval_hbar, self.n_terms, deriv_factor=1)
        )
        if fac_h is None or fac_hbar is None:
            return None
        return _analytic_pp_scores(
            roots, delta_eval, entries, base,
            factor_h=lambda loc: fac_h(loc + tw / 2),
            factor_hbar=lambda loc: fac_hbar(tw / 2),
            sigma=self._rev_coeff,
        )

    def _singular_merged_residue_scores(self, roots, delta_eval, entries, base):
        """Merged principal-part scores for a Virasoro tower at singular c.

        Return ``None`` when the contour scorer is needed.

        At a singular c the chiral Kac poles collide, so the simple-pole model in
        :meth:`_analytic_residue_scores` (residue ``A ~ 1/eps``) diverges. The
        physical block is finite because the divergent residues *cancel*: the
        cluster's MERGED order-``M`` Laurent coefficients are finite. For ``M``
        Kac poles colliding at ``x* = loc + twist/2`` with residues ``r_k`` and
        offsets ``d_k = x_k - x*``, the merged holomorphic principal coefficients
        (per derivative order ``i``) are ``P_i[j] = sum_k r_k d_k**(j-1)``,
        ``j = 1..M`` -- each finite as ``c -> c*``. Only the holomorphic half
        carries the pole (``hbar = twist/2`` is regular), so the block pole stays
        order ``M`` (not ``2M``): the entry ``(mi, ni)`` principal coefficients are
        ``C_j = Bhbar[ni] P_mi[j] + sigma Bhbar[mi] P_ni[j]``. The whole object is
        read directly from the merged chiral evaluators at ``c0``. Then the
        ``(16q)^Δ`` prefactor folds in (``g_{-j} = ebloc * sum_{k>=0}
        (-ln base)**k/k! * C_{j+k}``) and the score is ``max_e |sum_j g_{-j}/D**j|``
        with ``D = delta_eval - loc``.

        Covers the diagonal, off-diagonal (m-center != n-center), and folded
        (xt / ws) Virasoro tower. Off-diagonal uses the m-center holomorphic
        principal for the m-index and the n-center holomorphic principal for the
        n-index (the antiholomorphic halves stay regular at hbar = twist/2);
        folded builds the merged principal coefficients over the non-folded
        source band and folds each Laurent slice with the SAME L-independent
        parity map the value assembler uses (the fold commutes with the merged
        L-Laurent extraction). Defers (``None``) for a folded complex parity
        phase; unavailable merged pole data raises."""
        self._ensure_center_precision()
        if self.mode != "virasoro":
            return None
        from virasoro_blocks.derivs import _fold_parity
        from virasoro_blocks.polyderivs import (
            _fold_plan,
            _group_roots_by_location,
        )
        # Convert Acb midpoints through dyadic tuples. Recombining the Laurent
        # coefficients inside mp.workdps rounds them at the working precision.
        from virasoro_blocks.polynomial import _mpf_to_acb
        from virasoro_blocks.polynomial import _acb_to_mpc
        from virasoro_blocks.regularization import _MergedScorerUnavailable as _RegMergedUnavailable

        c_val0 = self.c
        tw = self.twist
        sigma = self._rev_coeff
        sigma_c = None if sigma in (None, 0) else mp.mpc(sigma)
        # The (w, s) fold is a real linear recombination, so a genuinely complex
        # parity phase cannot be folded -- defer to the contour there.
        if (self.folded and sigma_c is not None
                and abs(mp.im(sigma_c)) > mp.mpf("1e-30")):
            return None
        roots_mpc = [mp.mpc(r) for r in roots]
        groups = _group_roots_by_location(roots_mpc, tolerance(guard=10))
        n_terms = self.n_terms
        entry_pairs = [(int(mi), int(ni)) for (mi, ni) in entries]
        Mmax = max((len(idx) for _, idx in groups), default=0)

        # Folded coordinates: the non-folded source band whose (m, n) products
        # the fold map reads (a band of the square, not the full square),
        # computed once -- it is c- and L-independent.
        if self.folded:
            odd = self._fold_odd % 2
            M_total = self.m_max + 2 * self.n_max + odd
            src_dim = M_total + 1
            fold_plan, _ = _fold_plan(
                bool(odd), src_dim, src_dim, self.m_max, self.n_max
            )
            src_entries = sorted({(m, n) for (m, n, _c) in fold_plan})
        else:
            src_dim = None
            src_entries = None


        def make_evaluators(c_probe):
            """The probe's m-axis halves (holomorphic, carrying the pole; antiholomorphic, regular at
            ``hbar = twist/2``) and, off-diagonal, the n-axis halves recentered at the n-center."""
            cprobe = c_probe
            ev_h_m = self._cached_evaluator("h", cprobe)
            ev_hbar_m = ev_h_m if self._same_externals else self._cached_evaluator("hbar", cprobe)
            if self._offdiag:
                ev_h_n = self._cached_evaluator_n("h", cprobe)
                ev_hbar_n = ev_h_n if self._same_externals else self._cached_evaluator_n("hbar", cprobe)
            else:
                ev_h_n, ev_hbar_n = ev_h_m, ev_hbar_m
            return {"h_m": ev_h_m, "hbar_m": ev_hbar_m, "h_n": ev_h_n, "hbar_n": ev_hbar_n}

        def merged_principal(ev, xstar, M):
            """Merged holomorphic principal coefficients ``P[i][j-1]`` (chiral coefficient ``i``, Laurent order
            ``j = 1..M``) of one half at the collision ``xstar``, as flint balls: the Kac indices whose exact limit
            is ``xstar`` summed into one principal part (the score variable is ``L`` with ``x = L + twist/2``, so
            the pole sits at ``L = loc`` with no Jacobian factor)."""
            from flint import acb

            order = ev.merged_pole_order(xstar)
            if order is None:
                raise _MergedScorerUnavailable
            if order == 0:
                return [[acb(0)] * M for _ in range(n_terms)]       # (a removable pole: no principal part)
            pl = ev.pole_laurent(xstar, principal_order=order, regular_order=0, allow_regular_point=True)
            if pl is None:
                raise _MergedScorerUnavailable
            principal, _regular = pl
            return [[principal[j - 1][i] if j <= order else acb(0) for j in range(1, M + 1)] for i in range(n_terms)]

        def rows_from(evs):
            from flint import acb

            zero = acb(0)
            sig = None if sigma_c is None else _mpf_to_acb(sigma_c)
            ev_h_m, ev_hbar_m, ev_h_n, ev_hbar_n = evs["h_m"], evs["hbar_m"], evs["h_n"], evs["hbar_n"]
            bh_m = ev_hbar_m(mp.mpc(tw / 2), return_acb=True)
            Bhbar_m = [bh_m[i] for i in range(n_terms)]
            if ev_hbar_n is ev_hbar_m:
                Bhbar_n = Bhbar_m
            else:
                bh_n = ev_hbar_n(mp.mpc(tw / 2), return_acb=True)
                Bhbar_n = [bh_n[i] for i in range(n_terms)]

            rows = []
            for loc, idx in groups:
                M = len(idx)
                xstar = mp.mpc(loc) + tw / 2
                P_m = merged_principal(ev_h_m, xstar, M)
                P_n = P_m if not self._offdiag else merged_principal(ev_h_n, xstar, M)
                # Entry (mi, ni) principal coefficients:
                #   C_j = Bhbar_n[ni] P_m[mi][j] + sigma Bhbar_m[mi] P_n[ni][j]
                # (the main term puts the pole on the m-index, the swap term on
                # the n-index; the regular antiholomorphic half rides the other).
                def coefficient(mm, nn, j):
                    val = Bhbar_n[nn] * P_m[mm][j - 1]
                    if sig is not None:
                        val = val + sig * Bhbar_m[mm] * P_n[nn][j - 1]
                    return val

                if self.folded:
                    # Build one full source grid per Laurent order j, fold each,
                    # then read off the folded entries.
                    folded_slices = []
                    for j in range(1, M + 1):
                        grid = [[zero] * src_dim for _ in range(src_dim)]
                        for (mm, nn) in src_entries:
                            grid[mm][nn] = coefficient(mm, nn, j)
                        folded_slices.append(
                            _fold_parity(grid, self._fold_odd, self.m_max, self.n_max, zero=zero)
                        )
                    row = []
                    for (a, b) in entry_pairs:
                        row.extend(folded_slices[j - 1][a][b] if j <= M else zero for j in range(1, Mmax + 1))
                    rows.append(row)
                else:
                    row = []
                    for (mi, ni) in entry_pairs:
                        row.extend(coefficient(mi, ni, j) if j <= M else zero for j in range(1, Mmax + 1))
                    rows.append(row)
            return rows

        try:
            # the row assembly, once, on the evaluators AT c0 (merged poles from the one Laurent table); a ball
            # containing 0 is a structural zero (exactly 0, not its rounding), as in the fixed-spin scorer
            from virasoro_blocks.singularity import merged_singular_evaluators

            with merged_singular_evaluators():
                rows = rows_from(make_evaluators(c_val0))
            grid = [[mp.mpc(0) if (v.real.contains(0) and v.imag.contains(0)) else +_acb_to_mpc(v) for v in row]
                    for row in rows]
        except (_MergedScorerUnavailable, _RegMergedUnavailable) as exc:
            raise RuntimeError(
                f"large-spin singular-c scorer: the chiral evaluator could not supply Kac data at c = {self.c}; "
                "this singular locus needs support, not a silent contour fallback."
            ) from exc

        Lc = mp.log(mp.mpc(base))
        ebase = mp.mpc(base)
        deval = mp.mpc(delta_eval)
        tol_d = mp.mpf("1e-12")
        n_entry = len(entry_pairs)
        scores = [mp.mpf(0)] * len(roots_mpc)
        for gi, (loc, idx) in enumerate(groups):
            M = len(idx)
            loc_c = mp.mpc(loc)
            D = deval - loc_c
            if abs(D) <= tol_d:
                for i in idx:
                    scores[i] = mp.inf
                continue
            ebloc = ebase ** (-loc_c)
            row = grid[gi]
            best = mp.mpf(0)
            for ei in range(n_entry):
                C = row[ei * Mmax: ei * Mmax + Mmax]  # C[j-1], j = 1..Mmax
                val = mp.mpc(0)
                for j in range(1, M + 1):
                    g = mp.mpc(0)
                    fact = mp.mpf(1)
                    for k in range(0, M - j + 1):
                        g += ((-Lc) ** k / fact) * C[j + k - 1]
                        fact *= (k + 1)
                    val += ebloc * g / D ** j
                mag = abs(val)
                if mag > best:
                    best = mag
            for i in idx:
                scores[i] = best
        return scores

    def _analytic_leading_residue_scores(self, roots, entries, base):
        """Closed-form leading-Laurent-coefficient residue scores for the
        large-spin tower, or ``None`` to defer to the eps-probe scorer.

        Uses the same chiral residue/regular series data as
        :meth:`_analytic_residue_scores`, but returns leading Laurent
        coefficients per pole location (simple: |g_-1|, repeated: |g_-2|).
        """
        self._ensure_center_precision()
        if self.mode not in ("virasoro", "global") or self._singular:
            return None
        from virasoro_blocks.polyderivs import (
            _analytic_chiral_factor,
            _analytic_residue_leading_scores,
            _analytic_residue_leading_scores_axis,
            _analytic_residue_leading_scores_folded,
        )

        c_val = self.c
        tw = self.twist

        if self.folded:
            if isinstance(self._rev_coeff, mp.mpc) and self._rev_coeff.imag != 0:
                return None
            eval_h = self._cached_evaluator("h", c_val)
            eval_hbar = self._cached_evaluator("hbar", c_val)
            fac_h = _analytic_chiral_factor(eval_h, self.n_terms, deriv_factor=1)
            fac_hbar = (
                fac_h if eval_hbar is eval_h
                else _analytic_chiral_factor(eval_hbar, self.n_terms, deriv_factor=1)
            )
            if fac_h is None or fac_hbar is None:
                return None
            return _analytic_residue_leading_scores_folded(
                roots, entries, base,
                factor_m_plus=lambda loc: fac_h(loc + tw / 2),
                factor_m_minus=lambda loc: fac_hbar(tw / 2),
                factor_n_plus=lambda loc: fac_h(loc + tw / 2),
                factor_n_minus=lambda loc: fac_hbar(tw / 2),
                sigma=self._rev_coeff, fold_marker=self._fold_odd,
                m_max=self.m_max, n_max=self.n_max,
            )

        if self._offdiag:
            eval_h_m = self._cached_evaluator("h", c_val)
            eval_h_n = self._cached_evaluator_n("h", c_val)
            eval_hbar_m = self._cached_evaluator("hbar", c_val)
            eval_hbar_n = self._cached_evaluator_n("hbar", c_val)
            fac_h_m = _analytic_chiral_factor(eval_h_m, self.n_terms, deriv_factor=1)
            fac_h_n = _analytic_chiral_factor(eval_h_n, self.n_terms, deriv_factor=1)
            fac_hbar_m = (
                fac_h_m if eval_hbar_m is eval_h_m
                else _analytic_chiral_factor(eval_hbar_m, self.n_terms, deriv_factor=1)
            )
            fac_hbar_n = (
                fac_h_n if eval_hbar_n is eval_h_n
                else _analytic_chiral_factor(eval_hbar_n, self.n_terms, deriv_factor=1)
            )
            if None in (fac_h_m, fac_h_n, fac_hbar_m, fac_hbar_n):
                return None
            return _analytic_residue_leading_scores_axis(
                roots, entries, base,
                factor_m_plus=lambda loc: fac_h_m(loc + tw / 2),
                factor_m_minus=lambda loc: fac_hbar_m(tw / 2),
                factor_n_plus=lambda loc: fac_h_n(loc + tw / 2),
                factor_n_minus=lambda loc: fac_hbar_n(tw / 2),
                sigma=self._rev_coeff,
            )

        eval_h = self._cached_evaluator("h", c_val)
        eval_hbar = self._cached_evaluator("hbar", c_val)
        fac_h = _analytic_chiral_factor(eval_h, self.n_terms, deriv_factor=1)
        fac_hbar = (
            fac_h if eval_hbar is eval_h
            else _analytic_chiral_factor(eval_hbar, self.n_terms, deriv_factor=1)
        )
        if fac_h is None or fac_hbar is None:
            return None
        return _analytic_residue_leading_scores(
            roots, entries, base,
            factor_h=lambda loc: fac_h(loc + tw / 2),
            factor_hbar=lambda loc: fac_hbar(tw / 2),
            sigma=self._rev_coeff,
        )

    def _chiral(self, side, weight, c_val=None):
        externals = self._externals_for(side)
        if self.mode == "virasoro":
            return self._cached_evaluator(
                side, self.c if c_val is None else c_val
            )(weight)
        spec = COORDINATES[self.chiral_coordinate]
        if self._u_center == spec.u_symmetric():
            # zzb/yyb have convenience symmetric-point builders; every other base
            # (qqb, ...) uses the general centered builder at the symmetric point.
            if self.chiral_coordinate == "zzb":
                return global_chiral_z_taylor(
                    weight, externals, self.n_terms, chiral_norm=self.chiral_norm,
                    z_shift=self.z_shift,
                )
            if self.chiral_coordinate == "yyb":
                return global_chiral_y_taylor(
                    weight, externals, self.n_terms, chiral_norm=self.chiral_norm,
                    z_shift=self.z_shift,
                )
        return _global_chiral_taylor(
            weight,
            externals,
            self.n_terms,
            z_of_u=spec.z_of_u,
            u_symmetric=self._u_center,
            chiral_norm=self.chiral_norm,
            z_shift=self.z_shift,
        )

    def _chiral_n(self, side, weight, c_val=None):
        externals = self._externals_for(side)
        if self.mode == "virasoro":
            return self._cached_evaluator_n(
                side, self.c if c_val is None else c_val
            )(weight)
        spec = COORDINATES[self.chiral_coordinate]
        if self._u_center_bar == spec.u_symmetric():
            # zzb/yyb have convenience symmetric-point builders; every other base
            # (qqb, ...) uses the general centered builder at the symmetric point.
            if self.chiral_coordinate == "zzb":
                return global_chiral_z_taylor(
                    weight, externals, self.n_terms, chiral_norm=self.chiral_norm,
                    z_shift=self.z_shift,
                )
            if self.chiral_coordinate == "yyb":
                return global_chiral_y_taylor(
                    weight, externals, self.n_terms, chiral_norm=self.chiral_norm,
                    z_shift=self.z_shift,
                )
        return _global_chiral_taylor(
            weight,
            externals,
            self.n_terms,
            z_of_u=spec.z_of_u,
            u_symmetric=self._u_center_bar,
            chiral_norm=self.chiral_norm,
            z_shift=self.z_shift,
        )

    def _grid_from(self, a, c, b=None, d=None):
        """Assemble the non-folded ``(z, z̄)`` block grid from the two chiral
        halves ``a = k_h(weight = L + twist/2)`` and ``c = k_h̄(weight =
        twist/2)``, to the source dimensions ``self._src_m`` x ``self._src_n``
        (the folded coordinates expand these beyond ``m_max`` x ``n_max``).

        ordered_B keeps the single ordering ``B(h,h̄) = a[m] c[n]``;
        scalar_G / parity_projected_B add the parity-reversed partner
        ``B(h̄,h) = c[m] a[n]`` with coefficient ``self._rev_coeff`` (+1 for
        scalar_G, sign*phase for parity_projected_B). ``rev_coeff is
        None`` (ordered_B, no partner) maps to the shared assembler's
        ``sigma = 0``."""
        sigma = 0 if self._rev_coeff is None else self._rev_coeff
        return _assemble_block_grid(
            a,
            c,
            self._src_m,
            self._src_n,
            sigma=sigma,
            c_swap_m=b,
            c_swap_n=d,
        )

    def _assemble(self, a, c, b=None, d=None):
        """Final spin-L block grid (``self.m_max`` x ``self.n_max``): the
        non-folded outer-product grid, parity-folded to the ``(w, s)`` shape
        when the coordinate is ``xt``/``ws``. The fold is the same
        L-independent linear recombination as the fixed-spin tables."""
        grid = self._grid_from(a, c, b=b, d=d)
        if not self.folded:
            return grid
        from virasoro_blocks.derivs import _fold_parity
        # `_fold_parity` selects its flint field (arb vs acb) from entry [0][0],
        # so the source grid must be uniformly typed. The Virasoro chiral series
        # evaluator returns a real mpf for the (real) leading coefficient but
        # mpc for the rest; promote any such mixed grid to mpc so the fold's
        # acb path sees a complex [0][0]. All-real grids (global real-c case)
        # stay mpf and keep the faster arb path.
        if isinstance(grid[0][0], mp.mpf) and any(
            isinstance(v, mp.mpc) for row in grid for v in row
        ):
            grid = [[v if isinstance(v, mp.mpc) else mp.mpc(v) for v in row]
                    for row in grid]
        return _fold_parity(grid, self._fold_odd, self.m_max, self.n_max)

    def _sample_bucket_at_c(self, values, entries, c_val):
        """Batched, flint-native block grids for a spin-sample bucket.

        Mode-agnostic: the chiral coefficients come from the Virasoro or global
        flint series evaluator (per ``self.mode``) via its batched real-``arb``
        ``.many`` path, falling back per-Delta to native ``acb`` (complex Delta /
        complex c / Kac-pole hit) -- never to mpmath. The outer-product assembly
        and, for the folded coordinates, the parity fold run entirely in flint
        (:func:`assemble_block_batch`, with ``fold_parity`` for ``xt``/``ws``).
        Real-valued buckets stay arb-native end-to-end (``is_arb=True``);
        complex/acb buckets are converted to mpmath for downstream consumers
        that expect ``is_arb=False`` to imply mpmath values."""
        from virasoro_blocks.derivs import assemble_block_batch
        from virasoro_blocks.regularization import _grid_entry_to_mpc

        def _coerce_tables(tables):
            return [
                [
                    [
                        mp.mpf(0) if value is None else _grid_entry_to_mpc(value)
                        for value in row
                    ]
                    for row in table
                ]
                for table in tables
            ]

        def _acb_batch(ev, hs):
            return [ev(h, return_acb=True) for h in hs]

        h_values = [_tower_exchange_weight(L, self.twist) for L in values]
        sigma = 0 if self._rev_coeff is None else self._rev_coeff
        n = len(values)

        if self._offdiag:
            eval_h_m = self._cached_evaluator("h", c_val)
            eval_h_n = self._cached_evaluator_n("h", c_val)
            eval_hbar_m = self._cached_evaluator("hbar", c_val)
            eval_hbar_n = self._cached_evaluator_n("hbar", c_val)
            many_h_m = getattr(eval_h_m, "many", None)
            many_h_n = getattr(eval_h_n, "many", None)
            many_hbar_m = getattr(eval_hbar_m, "many", None)
            many_hbar_n = getattr(eval_hbar_n, "many", None)
            c_h_m_all = c_h_n_all = c_hbar_m_one = c_hbar_n_one = None
            if (
                many_h_m is not None and many_h_n is not None
                and many_hbar_m is not None and many_hbar_n is not None
            ):
                c_h_m_all = many_h_m(h_values)
                c_h_n_all = many_h_n(h_values)
                c_hbar_m = many_hbar_m([self.twist / 2])
                c_hbar_n = many_hbar_n([self.twist / 2])
                if None not in (c_h_m_all, c_h_n_all, c_hbar_m, c_hbar_n):
                    c_hbar_m_one, c_hbar_n_one = c_hbar_m[0], c_hbar_n[0]
                else:
                    c_h_m_all = None
            if c_h_m_all is None:
                # complex Delta / Kac-pole hit: per-Delta acb, still in flint.
                c_h_m_all = _acb_batch(eval_h_m, h_values)
                c_h_n_all = _acb_batch(eval_h_n, h_values)
                c_hbar_m_one = eval_hbar_m(self.twist / 2, return_acb=True)
                c_hbar_n_one = eval_hbar_n(self.twist / 2, return_acb=True)
            tables, is_arb = assemble_block_batch(
                c_h_m_all,
                [c_hbar_n_one] * n,
                self.m_max,
                self.n_max,
                sigma=sigma,
                entries=entries,
                c_swap_m_all=(None if self._rev_coeff is None else [c_hbar_m_one] * n),
                c_swap_n_all=(None if self._rev_coeff is None else c_h_n_all),
            )
            if is_arb:
                return tables, True
            return _coerce_tables(tables), False

        eval_h = self._cached_evaluator("h", c_val)
        eval_hbar = self._cached_evaluator("hbar", c_val)
        fold_kw = {"fold_parity": self._fold_odd} if self.folded else {}
        many_h = getattr(eval_h, "many", None)
        many_hbar = getattr(eval_hbar, "many", None)
        c_h_all = c_hbar_one = None
        if many_h is not None and many_hbar is not None:
            c_h_all = many_h(h_values)
            c_hbar = many_hbar([self.twist / 2])
            if c_h_all is not None and c_hbar is not None:
                c_hbar_one = c_hbar[0]
            else:
                c_h_all = None
        if c_h_all is None:
            # complex Delta / Kac-pole hit: per-Delta acb, still in flint
            # (folded coordinates fold via assemble_block_batch below, never
            # the mpmath `_assemble` -> `_fold_parity` round trip).
            c_h_all = _acb_batch(eval_h, h_values)
            c_hbar_one = eval_hbar(self.twist / 2, return_acb=True)
        tables, is_arb = assemble_block_batch(
            c_h_all,
            [c_hbar_one] * n,
            self.m_max,
            self.n_max,
            sigma=sigma,
            entries=entries,
            **fold_kw,
        )
        if is_arb:
            return tables, True
        return _coerce_tables(tables), False

    def __call__(self, L):
        self._ensure_center_precision()
        if self._singular:
            # the tower AT the singular c: its chiral evaluators read the merged poles of the one Laurent table
            from virasoro_blocks.singularity import merged_singular_evaluators

            with merged_singular_evaluators():
                return self._grid_at(L)
        return self._grid_at(L)

    def _grid_at(self, L):
        h = _tower_exchange_weight(L, self.twist)
        a = self._chiral("h", h)
        if self._offdiag:
            c = self._chiral_n("hbar", self.twist / 2)
            if self._rev_coeff is None:
                return self._assemble(a, c)
            b = self._chiral("hbar", self.twist / 2)
            d = self._chiral_n("h", h)
            return self._assemble(a, c, b=b, d=d)
        if self._hbar_coeffs is None or self._hbar_coeffs[0] != mp.mp.prec:
            self._hbar_coeffs = (mp.mp.prec, self._chiral("hbar", self.twist / 2))
        c = self._hbar_coeffs[1]
        return self._assemble(a, c)

    def sample_bucket(self, values, entries):
        """``(tables, is_arb)`` block grids at the spin samples ``values``,
        restricted to a node-count bucket's ``entries``.

        The interpolation groups (m, n) into node-count buckets and consumes
        only one bucket's entries from each sampled grid; building the full
        ``m_max x n_max`` outer product at every node (as ``__call__`` does) is
        the dominant large-spin cost. This builds only the requested entries
        from the shared chiral halves. Folded coordinates cannot prune per entry
        -- each folded ``(w, s)`` entry mixes many non-folded source entries --
        so they build the full folded grid (folded in flint by
        ``assemble_block_batch``).

        The tower shares the unified `(tables, is_arb)` contract with the
        global / fixed-spin samplers. On the common real path this returns
        flint ``arb`` grids (``is_arb=True``) so interpolation can stay
        flint-native end-to-end; complex/acb and singular-c paths return
        mpmath grids (``is_arb=False``). At singular c, the bucket reads the
        same merged Laurent context as individual evaluations."""
        values = list(values)
        self._ensure_center_precision()
        if self._singular:
            from virasoro_blocks.singularity import merged_singular_evaluators

            with merged_singular_evaluators():
                return self._sample_bucket_at_c(values, entries, self.c)
        # Both modes: the whole bucket -- outer-product assembly and, for the
        # folded coordinates, the parity fold -- runs natively in flint (arb for
        # real Delta, acb otherwise) via `_sample_bucket_at_c`. Real buckets
        # remain arb-native (`is_arb=True`); only complex/acb buckets are
        # converted to mpmath. This avoids the per-L mpmath `_assemble` ->
        # `_grid_from` -> `_assemble_block_grid` + `_fold_parity` round trip
        # that dominated the folded-coordinate build.
        # Global mode carries no central charge (``self.c is None``); the global
        # series evaluator ignores ``c_val`` (it is only a cache key).
        c_val = self.c
        return self._sample_bucket_at_c(values, entries, c_val)


def refuse_scalar_G_tower_exchanged_parity(basis, exchanged_parity, *, flag="exchanged_parity"):
    """Refuse an exchanged parity other than +1 for a ``scalar_G`` / ``global_scalar_G`` tower.

    The scalar ``G`` is ``B(h, h̄) + B(h̄, h)``, z <-> z̄ even at every integer spin, so there is one tower and no
    odd one.  The z <-> z̄-odd ``B(h, h̄) - B(h̄, h)`` belongs to parity-odd correlators and is the ``parity_projected_B`` tower with projection
    ``'-'`` (``eta = -1``).
    """
    if basis not in ("scalar_G", "global_scalar_G"):
        return
    if normalize_parity_phase(exchanged_parity or "1") == "1":
        return
    cli = flag.startswith("--")
    setting = f"{flag} {exchanged_parity}" if cli else f"{flag} = {exchanged_parity!r}"
    # the CLI builds parity_projected_B towers for spinning externals only; the Python tower takes any externals
    where = ("; the CLI builds it for spinning externals only, so for scalar externals call "
             "large_spin_twist_table(..., basis='parity_projected_B', parity_projection='-') from Python" if cli
             else ": basis='parity_projected_B', parity_projection='-'")
    raise ValueError(
        f"{setting} does not apply to the {basis} tower: the scalar G = B(h, hbar) + B(hbar, h) is z <-> zbar "
        "even at every spin, so one tower covers every integer L (there is no odd scalar_G tower). The "
        "z <-> zbar-odd structure B(h, hbar) - B(hbar, h) is the parity_projected_B tower with projection '-' "
        f"(eta = -1){where}"
    )


def refuse_spin_zero_scalar_G_tower(basis, Lmax, exchanged_parity="1", *, flag="Lmax"):
    """Refuse a ``scalar_G`` tower whose floor is spin 0.

    The tower is analytic in L, so its L = 0 member is the doubled block ``2 B(h, h)``, while the spin-0 scalar
    ``G`` is ``B(h, h)``: a table holding the former under a ``scalar_G`` label would double the scalar channel
    for any consumer that adds it to an identity vector.  Spin 0 belongs in the fixed-spin channels.  There is
    one scalar ``G`` tower (z <-> z̄ even, covering every integer ``L >= Lmax``), and an ``exchanged_parity``
    other than +1 is refused for it (:func:`refuse_scalar_G_tower_exchanged_parity`); the ``ordered_B`` /
    ``parity_projected_B`` towers carry no spin-0 division.
    """
    from ._conventions import SCALAR_G_SPIN0_DOUBLE, scalar_G_spin0_convention

    if basis not in ("scalar_G", "global_scalar_G"):
        return
    refuse_scalar_G_tower_exchanged_parity(
        basis, exchanged_parity, flag="--parity-exchanged" if flag.startswith("--") else "exchanged_parity")
    if scalar_G_spin0_convention() == SCALAR_G_SPIN0_DOUBLE or _real_tower_label(Lmax) != 0:
        return
    setting = (lambda value: f"{flag} {value}") if flag.startswith("--") else (lambda value: f"{flag} = {value}")
    raise ValueError(
        f"{setting(0)} starts the scalar_G tower at spin 0: the tower is analytic in L, so its L = 0 member is the "
        "doubled block 2 B(h, h), not the spin-0 G = B(h, h). Build spin 0 as a fixed-spin table and start the "
        f"tower above it (for example {setting(2)})"
    )


def global_tower_q_order(q_order, npoles, h_externals):
    """The smallest q-order >= ``q_order`` whose global SL(2) pole pool offers ``npoles`` genuine poles.

    A global tower's ``q_order`` only sizes its candidate pool (the 2F1 values are exact), so a chosen pole count
    enlarges the pool as the global fixed-spin tables do.  With identical externals only the odd chiral poles
    ``h = -p/2`` are genuine, so the pool holds ``q_order // 2`` poles."""
    q_order = _validate_q_order(q_order)
    d12, d34 = _externals_diffs(h_externals)
    limit = q_order + 4 * int(npoles) + 8
    while len(_global_chiral_q_order_pole_pool(q_order, delta12=d12, delta34=d34)) < npoles:
        q_order += 1
        if q_order > limit:
            raise ValueError(f"no global q-order up to {limit} offers {npoles} genuine tower poles")
    return q_order


@_quiet_conditioning
def large_spin_twist_table(
    twist, *, mode, coordinate="zzb", h_externals, m_max, n_max, Lmax, npoles,
    c=None, q_order=None, chiral_norm=None, z_shift=None,
    pole_selection=None, num_threads=1, shared_sample_points=False,
    exchanged_parity="1", hbar_externals=None, basis="scalar_G",
    parity_projection=None, parity_external=None, kept_entries=None,
    center=None, center_bar=None, kept_poles=None,
):
    """Interpolate the fixed-``twist`` spin-L 2d block in ``L`` over
    ``[Lmax, inf)``; returns an ``InterpolatedDeltaTable`` in the spin variable.

    ``mode='virasoro'`` needs ``c`` and ``q_order``; ``mode='global'`` uses
    ``q_order`` to define its finite SL(2) candidate pole pool (but evaluates
    exact 2F1 values). With no explicit ``kept_poles``, ``npoles=None`` retains every pole in this finite
    q-order pool. Only the ``h = L + twist/2`` half carries ``L`` so the non-folded
    per-entry L-power is ``max(m, n)`` for a parity sum and ``m`` for
    ``ordered_B``, whose antiholomorphic half is independent of L.
    Supported coordinates are the non-folded
    crossing-odd bases `zzb`/`yyb`/`qqb` and their parity folds `xt`/`ws`/`qqb_ws`
    (`xt` folds `zzb`, `ws` folds `yyb`, `qqb_ws` folds `qqb`). The fold is an L-independent linear recombination
    of the non-folded grid, so a folded ``(w, s)`` entry ``(alpha, beta)`` draws
    from non-folded ``(m, n)`` with ``m + n = alpha + 2 beta (+1 if odd)`` and
    thus has L-power ``alpha + 2 beta (+ odd)``. Folded coordinates require
    basis `scalar_G` or `parity_projected_B` (ordered_B has no fold).
    Their `*_radial` slices retain only column n = 0, returning an
    ``(m_max + 1) x 1`` table regardless of the supplied ``n_max``.

    Bases (mirroring the fixed-spin builders), with the antiholomorphic half
    built from ``hbar_externals`` (default: scalar, ``h̄ = h``):
      * ``scalar_G`` -- symmetric ``B(h,h̄) + B(h̄,h)``, z <-> z̄ even at every
        spin, so one tower covers every integer ``L >= Lmax``.
        ``exchanged_parity`` must be +1 (the default); -1 is refused (there is
        no odd scalar_G tower; the z <-> z̄-odd ``B(h,h̄) - B(h̄,h)`` is the
        ``parity_projected_B`` tower with ``parity_projection='-'``).
      * ``ordered_B`` -- the single ordering ``B(h,h̄)`` (no parity partner).
      * ``parity_projected_B`` -- ``B(h,h̄) + sign*phase B(h̄,h)`` with
        ``sign`` from ``parity_projection`` (+/-) and ``phase`` from
        ``parity_external`` / ``exchanged_parity``.
    """
    if mode not in LARGE_SPIN_MODES:
        raise ValueError(f"mode must be one of {LARGE_SPIN_MODES}; got {mode!r}")
    if coordinate not in LARGE_SPIN_COORDINATES:
        raise ValueError(
            f"coordinate must be one of {LARGE_SPIN_COORDINATES}; got {coordinate!r}"
        )
    if coordinate in _LARGE_SPIN_FOLDED_COORDINATES and (
        center is not None or center_bar is not None
    ):
        raise ValueError(
            "off-diagonal centers are supported only on non-folded coordinates "
            "(zzb/yyb/qqb)"
        )
    if mode == "virasoro" and (c is None or q_order is None):
        raise ValueError("virasoro mode requires c and q_order")
    if mode == "global":
        q_order = DEFAULT_GLOBAL_Q_ORDER if q_order is None else _validate_q_order(q_order)
    else:
        q_order = _validate_derivative_order("q_order", q_order)
    m_max = _validate_derivative_order("m_max", m_max)
    n_max = _validate_derivative_order("n_max", n_max)
    if COORDINATES[coordinate].kind == "radial":
        n_max = 0
    if kept_poles is None and npoles is not None:
        npoles = _validate_npoles(npoles)
    if pole_selection is None:
        pole_selection = "contribution_optimal"
    h_externals = tuple(h_externals)
    h_ext = [_real_tower_label(h) for h in h_externals]
    hbar_ext = None if hbar_externals is None else [_real_tower_label(h) for h in hbar_externals]
    twist, shift = _real_tower_label(twist), _real_tower_label(Lmax)
    refuse_spin_zero_scalar_G_tower(basis, shift, exchanged_parity)
    z_shift = mp.mpf(0) if z_shift is None else z_shift
    base = large_spin_base(
        mode, chiral_norm, center=center, center_bar=center_bar, basis=basis,
    )
    sampler = _LargeSpinTwistSampler(
        mode, coordinate, h_ext, chiral_norm, z_shift, m_max, n_max, twist,
        c=c, q_order=q_order, exchanged_parity=exchanged_parity,
        hbar_externals=hbar_ext, basis=basis,
        parity_projection=parity_projection, parity_external=parity_external,
        center=center, center_bar=center_bar)
    # Only the holomorphic half carries L, so the L-pole pool keys off the
    # holomorphic external structure.
    #
    # Virasoro mode uses the block's actual Kac poles d_{r,s}(c) - twist/2 (vs
    # the global SL(2) ladder, which is their c -> inf limit), recovering several
    # tail digits. At a recognition-singular c (c = 1, 25, minimal models, the
    # c > 25 rational-b^2 duals) the genuine pole set is the *regularized* chiral
    # root set (collisions resolved), recovered from the regularized skeleton --
    # exactly as the fixed-spin path does.
    if mode == "virasoro":
        pool = _virasoro_chiral_pole_pool(
            h_externals=h_ext, c=c, q_order=q_order, twist=twist,
        )
        # The merged skeleton has already removed canceled poles and surplus
        # multiplicities. Keep its exact orders: finite-distance probes can
        # miss a small leading principal coefficient and lower a genuine pole.
    else:
        # The tower interpolates the block in L with h = L + twist/2, so the
        # chiral h-poles at -p/2 sit at L = -p/2 - twist/2 in the interpolation
        # variable -- the -twist/2 shift is essential (mirrors the Virasoro pool;
        # without it the prefactor poles land on regular points of the block).
        # q-order defines the finite chiral SL(2) universe just as it does for
        # the fixed-spin global builders.
        d12, d34 = _externals_diffs(h_externals)
        twist_mp = mp.mpf(mp.convert(twist))
        pool = [
            p - twist_mp / 2
            for p in _global_chiral_q_order_pole_pool(
                q_order, delta12=d12, delta34=d34,
            )
        ]
    candidate_count = len(pool)
    if kept_poles is not None:
        # another coordinate's table already chose the poles (L units): one prefactor for every coordinate of the
        # tower, as for the fixed-spin tables (a per-coordinate selection kept different poles and could not be
        # serialized together)
        pool = [mp.mpc(p) if isinstance(p, mp.mpc) else mp.mpf(p) for p in kept_poles]
        npoles, pole_selection = len(pool), "nearest"
    elif npoles is None:
        npoles = len(pool)
    if pole_selection == "contribution_optimal" and npoles > len(pool):
        if mode == "virasoro":
            # every genuine pole kept, as a fixed-spin --npoles at or above its pool: at c = 1 with h_ext = 1/16 the
            # block has none at all (H = 1), and raising q_order cannot add any
            npoles = len(pool)
        else:
            raise ValueError(
                f"{mode} q_order={q_order} provides only {len(pool)} genuine "
                "candidate poles; reduce npoles or raise q_order"
            )

    if sampler.folded:
        fold_odd = sampler._fold_odd

        def entry_power(alpha, beta):
            return folded_entry_power(alpha, beta, fold_odd)
    else:
        def entry_power(m, n):
            return m if basis == "ordered_B" else max(m, n)

    table = _interpolate_global_table(
        sampler=sampler, m_max=m_max, n_max=n_max, base=base, shift=shift,
        pool=pool, residue_scorer=_make_residue_scorer(
            sampler, base, m_max, n_max,
            score_entries=pole_selection_score_entries(pole_selection),
        ),
        entry_power=entry_power, npoles=npoles, pole_selection=pole_selection,
        num_threads=num_threads, shared_sample_points=shared_sample_points,
        kept_entries=kept_entries,
        # At singular c, use the analytic merged principal parts first. If
        # that scorer explicitly defers, use a finite-radius contour rather
        # than near-pole probes, whose fixed-weight limit can differ from the
        # continuation in exchanged weight that the table represents.
        contribution_via_contour=(mode == "virasoro" and sampler._singular),
        force_keep_real_positive=(mode == "virasoro"),
        # Both tower pools are q-order-bounded and npoles-independent, so
        # contribution_optimal can construct one nested greedy order.
        fixed_candidate_pool=True,
    )
    # the block the tower interpolates, for checking the table between its spin nodes (tower_verified_digits)
    table.tower_sampler = sampler
    # the poles a larger npoles could keep (a Virasoro pool at or below npoles keeps every pole): a chosen count
    # stops growing there
    table.remainder_pool = candidate_count
    return table


def tower_verified_digits(serialized, tables, kept_entries_by_coord, target):
    """How well a serialized large-spin tower matches the block it interpolates (a
    :class:`~virasoro_blocks.verified_build.Verification`, over the spin range between its nodes), or None
    (nothing to check).

    The tower is an order-reduced interpolation in the spin ``L`` (an explicit or a chosen ``npoles``), checked the way
    the fixed-spin ``--npoles`` tables are (:func:`virasoro_blocks.table_backends._verified_digits`): the stored
    table, evaluated at twice the working precision, against the tower's own direct sampler (the fixed-twist block
    at that ``L``) at points between its interpolation nodes.  Past the last node an order-reduced table
    extrapolates, so it is not checked there.  ``tables`` maps each coordinate to the table
    :func:`large_spin_twist_table` returned (which carries its sampler).
    """
    from virasoro_blocks.verified_build import check_points_x, point_agreement, table_values, verification_from_points

    xs = getattr(serialized, "sample_points_x", None)
    if not xs:
        return None
    points = check_points_x(xs, beyond_last=False)
    per_point = [None] * len(points)
    for coord, table in tables.items():
        sampler = getattr(table, "tower_sampler", None)
        kept = list(kept_entries_by_coord.get(coord) or ())
        if sampler is None or not kept:
            continue
        spins = [mp.mpf(serialized.shift) + x for x in points]
        refs = [sampler(L) for L in spins]
        gots = table_values(serialized, coord, spins, kept, dps=2 * mp.mp.dps)
        for i, (got, ref) in enumerate(zip(gots, refs)):
            measured = point_agreement(got, ref, kept, int(target))
            if measured is None:
                continue
            per_point[i] = measured if per_point[i] is None else (min(per_point[i][0], measured[0]),
                                                                  min(per_point[i][1], measured[1]))
    last = max(mp.re(mp.mpc(x)) for x in xs)
    # (the span, recorded as verified_spin_range, is the nodes' spin interval [Lmin, Lmin + last node])
    return verification_from_points(per_point, points, n_inner=len(points), last_node=last, shift=serialized.shift,
                                    variable="L - Lmin", far_x=last)
