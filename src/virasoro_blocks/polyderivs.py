"""Symbolic-in-Delta derivative tables for the scalar block G.

`derivs.py` produces the (u, ubar)-derivative tables of G at a *numeric* Delta:
each table entry is a number obtained from the Taylor coefficients of `F_block` (flint series, or
`precision.taylor_coefficients`). This
module produces the same tables kept *exact* in Delta -- each entry is a
polynomial numerator over a universal Delta-rational prefactor.

The carry rests on `chiral.py`'s decomposition F(h, z) = E(z)**h A(z) N_z(h)/D(h)
(see `hseries.py`). The u-Taylor coefficient of the chiral block is

    c_m(h) = E(z_*)**h * P_m(h) / D(h),

with P_m a polynomial in h and D the (z-independent) Kac denominator. The
scalar block G_{Delta, ell} = [B(h, hbar) + B(hbar, h)] / (1 + delta_{ell, 0}) with
h = (Delta + ell)/2, hbar = (Delta - ell)/2 (so G_{Delta, 0} = B(h, h); see `_conventions`), and substituting
these linear maps into the P_m and D polynomials gives, per (m, n),

    table[m][n](Delta) = E(z_*)**Delta * numerator_{m,n}(Delta) / denominator(Delta),

where `denominator(Delta) = D((Delta+ell)/2) * D((Delta-ell)/2)` and the
prefactor base E(z_*) are shared by every (m, n). The denominator roots sit at
Delta = 2 d_{r,s} +/- ell.

The parity-folded coordinates (ws, xt, qqb_ws) are a Delta-independent linear
recombination of the non-folded numerators, so `derivs._fold_parity` folds the
numerator polynomials directly.

Order reduction (`--npoles`) reuses the same `block_deriv(Delta) =
prefactor(Delta) * numerator(Delta)` framing: `interpolate_delta_table` takes
an exact `DeltaDerivTable`, keeps only a selected subset of Delta-poles in a
universal damped-rational prefactor, and replaces each (m, n) numerator with a
*lower-degree* polynomial obtained by sampling the block at the optimal
interpolation nodes (arXiv:2509.14307). Each (m, n) entry is fit on its own
degree-matched node set.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from fractions import Fraction
from .numeric_input import _exchange_weight, _evaluation_number, _integer_control, _signed_half_integer as _check_ell_signed
from dataclasses import dataclass, field as _dataclass_field, replace as _dataclass_replace
from typing import NamedTuple

import mpmath as mp

from .conditioning import quiet as _quiet_conditioning
from functools import lru_cache
from .parallel import usable_workers, validate_worker_count

from .coordinates import (
    COORDINATES,
    q_symmetric,
    y_symmetric,
    z_of_q,
    z_of_y,
    z_symmetric,
)
import multiprocessing

from .derivs import (
    _check_ell, _check_externals, _check_table_orders, _fold_parity, _fold_plan,
    fold_parity_marker, folded_source_order,
)
from .hseries import HSeries, exp_h_times
from .interpolation import (
    _validate_npoles,
    _UNREMOVED_KAC_ZERO_MARKER,
    DampedRational,
    DeltaInterpolant,
    get_sample_points,
    solve_support_endpoint,
    lagrange_interpolation_polynomial,
    lagrange_polynomial_from_basis,
    precompute_lagrange_basis,
    _shifted_sampling_points,
    pole_selection_is_contribution,
    pole_selection_is_full_contribution,
    pole_selection_score_entries,
    select_kept_poles,
)
from .params import b_from_c, c_from_b, c_origin_of_b
from .parity import parity_reversal_phase
from ._conventions import scalar_G_swap_coefficient
from ._conventions import pool_initializer_kwargs
from .polyblock import H_q_series_rational
from .polynomial import (
    Polynomial,
    _arb_to_mpf,
    _flint_prec_bits,
    _mpf_to_arb,
    _mpf_to_acb,
    real_factor_coeffs_from_conjugate_roots,
)

# Sentinel for build_table_from_cache's lambda_cut: distinguishes "caller left
# it unset, apply the default corner-skip" from an explicit None ("build the
# full square"). See build_table_from_cache.
_AUTO_LAMBDA_CUT = object()
from .precision import tolerance, _validation_tolerance
from .recursion import (
    _build_H_context,
    _chiral_residue_series,
    _eval_H_at,
    _eval_H_at_many,
    _resolve_chiral_norm,
    q_from_z,
)


def _acb_to_mpc(val, ndigits=None):
    """Convert an Acb midpoint to mpmath through exact dyadic tuples.

    ``ndigits`` is accepted for compatibility; the conversion preserves the
    stored midpoint without decimal rounding.
    """
    return mp.make_mpc((val.real._mpf_, val.imag._mpf_))


def _resolve_bc(b, c):
    if (b is None) == (c is None):
        raise ValueError("specify exactly one of b, c")
    if b is None:
        from .singularity import snap_float_c

        c = snap_float_c(c)       # a float 0.7 means 7/10
        b = b_from_c(c)
    else:
        from .params import c_origin_of_b

        origin = c_origin_of_b(b)
        if origin is None:
            from .params import _c_input_of_b

            c = _c_input_of_b(b)
        else:
            c = origin
            b = b_from_c(c)
    return b, c


def _symbolic_guard_digits(q_order: int, c=None, h_externals=None) -> int:
    """Extra working digits for the symbolic table builders.

    Two losses, and the larger one is covered:

    * small c: the common-denominator h-rational representation loses digits to cancellation between
      nearby Kac poles, roughly linearly in q_order.  The measured loss (G_zzb_poly_table, 2026-09-27) is
      15/34/53/>60 digits at c = 0.8 and q_order 10/20/30/40, 10/25/39/50 at c = 1.2, 7/18/27/34 at
      c = 1.7, 2/6/7/8 at c = 3.1, none from c ~ 5.  The slope ``2.5 exp(-(c - 1/2)/1.2)`` digits per
      unit of q_order, plus 10, bounds every point.  Without it, dps-60 tables at c = 0.8, q_order 30
      were good to 2e-7 only (their coefficients), and to nothing at q_order 40.
    * large q_order at large c: 13 digits per step above 40 stabilizes the q60/q80 c = 10000 control
      surfaces against the direct numeric recursion.
    """
    large_q = 0 if q_order <= 40 else 13 * (q_order - 40)
    small_c = 0
    if c is not None and q_order > 0:
        cr = max(float(mp.re(mp.mpc(mp.convert(c)))), 0.5)
        slope = 2.5 * float(mp.exp(-(cr - 0.5) / 1.2))
        if slope * q_order > 1:
            small_c = int(mp.ceil(slope * q_order)) + 10
    # (near, not on, a singular locus the verified guard of singularity.near_singular_compute is added on top by
    # chiral_symbolic_useries)
    return max(large_q, small_c)


def _symbolic_workdps_for_q_order(q_order: int, *, use_guard: bool = True, c=None, h_externals=None) -> int:
    guard = _symbolic_guard_digits(q_order, c, h_externals) if use_guard else 0
    return mp.mp.dps + guard


def _symbolic_work_context(q_order: int, *, use_guard: bool = True, c=None, h_externals=None):
    target = _symbolic_workdps_for_q_order(q_order, use_guard=use_guard, c=c, h_externals=h_externals)
    if target == mp.mp.dps:
        return nullcontext()
    return mp.workdps(target)


def _reuse_symbolic_work_context(*objects):
    target = mp.mp.dps
    for obj in objects:
        if obj is None:
            continue
        target = max(target, getattr(obj, "working_dps", mp.mp.dps) or mp.mp.dps)
    if target == mp.mp.dps:
        return nullcontext()
    return mp.workdps(target)


@dataclass
class ChiralSymbolic:
    """u-Taylor expansion of F(h, z(u)) around u = u_*, exact in h.

        chiral_coeff(m, h) = exp_base ** h * numerators[m](h) / denominator(h)

    `denominator` and `denominator_roots` are z-independent (the Kac data);
    `numerators` and `exp_base` collapse the q-series at the chosen u_*.
    """

    exp_base: mp.mpc
    denominator: Polynomial
    denominator_roots: list
    numerators: list  # Polynomial in h, one per u-power (None in skeleton mode)
    root_provenance: list | None = None
    working_dps: int | None = None
    # Skeleton mode: when `numerators` is None, this holds just the per-u-power
    # numerator degrees (degree of numerators[m] == denominator.degree + m).
    numerator_degrees: list | None = None
    # Kac truncation order used to build the denominator/numerators. Carried so
    # the downstream block-match collision oracle (which rebuilds the chiral
    # block via H_q_series_laurent) can recover the same q_order from the table's
    # stamped oracle_metadata; without it the oracle reads None and crashes.
    q_order: int | None = None

    def chiral_coeff(self, m: int, h):
        from .numeric_input import literal_number

        # Keep h in its input field: real h with a real exp_base / mpf
        # numerators evaluates entirely in mp.mpf. A table built with
        # symbolic guard digits is evaluated at its own working precision.
        h = literal_number(h)
        with mp.workdps(max(mp.mp.dps, self.working_dps or 0)):
            h = mp.convert(h)
            if isinstance(h, mp.mpc) and h.imag == 0:
                h = h.real
            value = (self.exp_base**h * self.numerators[m].evaluate(h)
                     / self.denominator.evaluate(h))
        return +value


@dataclass
class DeltaDerivTable:
    """G derivative table held exact in Delta.

        table[m][n](Delta) = exp_base ** Delta
                             * numerators[m][n](Delta) / denominator(Delta)

    `exp_base` and `denominator` are shared by every (m, n). The denominator
    roots are the block's Delta-poles, Delta = 2 d_{r,s} +/- ell.

    `spin` is the exchanged spin ell for the 2d bases (scalar G, ordered B,
    parity-projected B) and `None` for the chiral basis (where the table is
    a function of `h` rather than `Delta` and there is no exchanged spin).
    It determines the default polynomial-variable origin used by
    `interpolate_delta_table` and `serialize_deriv_tables` when no explicit
    `shift` is supplied: `abs(spin)` for the 2d bases (the unitarity bound),
    `0` for chiral.
    """

    exp_base: mp.mpc
    denominator: Polynomial
    denominator_roots: list
    numerators: list  # list of list of Polynomial in Delta
    spin: object = None  # ell for 2d bases, None for chiral_F
    root_provenance: list | None = None
    # Optional well-conditioned numeric block evaluator (a picklable
    # `_NumericBlockSampler`).  When set, interpolation sample-value evaluation
    # uses the O(1) numeric recursion instead of evaluating the ~1e500-coeff
    # symbolic P(Delta)/D(Delta) (which needs very high dps to survive
    # cancellation).  The numeric block equals this table's value, so the
    # produced interpolant is identical -- only the conditioning differs.
    numeric_sampler: object = None
    # Provenance flag: True when this table may carry assembled-level removable
    # poles that the build-time chiral gate did NOT remove. This is the case
    # only for ε-extrapolated tables from `regularize_delta_table` at a
    # recursion-singular c (the Kac-degenerate loci where distinct chiral roots
    # collide onto integer Delta with vanishing residue). At generic c the
    # build-time K != 0 gate already produces a minimal table, so the
    # precision-fragile numeric removable-pole strip is skipped (avoiding its
    # high-q / low-dps false positives). The two strip call sites -- serialize
    # and the interpolation prefactor builder -- consult this flag and only run
    # `_strip_removable_poles_from_exact_tables` when it is set.
    requires_pole_strip: bool = False
    # Skeleton mode: when `numerators` is None this grid carries only the
    # per-(m, n) numerator polynomial *degrees* (an int grid of the same
    # shape). Built when `numeric_sampler` makes the symbolic numerator
    # coefficients unnecessary -- the interpolation reads degrees for
    # node placement and gets sample values from the numeric sampler -- so the
    # ~1e500-coeff numerator assembly is skipped. Only valid alongside a
    # numeric sampler and numeric pole ranking; the symbolic numerator-using
    # paths (symbolic residue ranking, `evaluate`, serialization of the exact
    # table) require `numerators` and reject a skeleton table.
    numerator_degrees: list | None = None
    # Optional block-construction metadata for value-oracle checks.
    oracle_metadata: dict | None = None
    # Serialization must keep the context that assembled these coefficients,
    # including guards whose surrounding workdps has already exited.
    working_dps: int | None = _dataclass_field(default_factory=lambda: mp.mp.dps)
    # Structural zeros are separate from degree zero, which can be a genuine
    # constant numerator. Skeleton builders record their omitted corners and
    # parity zeros without constructing coefficients.
    zero_entries: frozenset[tuple[int, int]] = frozenset()

    @property
    def _shape_grid(self) -> list:
        """Numerators if present, else the degree grid -- whichever defines the
        (m, n) table shape (a skeleton table has numerators=None)."""
        return self.numerators if self.numerators is not None else self.numerator_degrees

    @property
    def m_max(self) -> int:
        return len(self._shape_grid) - 1

    @property
    def n_max(self) -> int:
        return len(self._shape_grid[0]) - 1

    def numerator_degree(self, m: int, n: int) -> int:
        """Degree of the (m, n) numerator polynomial, from the skeleton degree
        grid when present (no coefficients needed) else the polynomial."""
        if self.numerator_degrees is not None:
            return self.numerator_degrees[m][n]
        return self.numerators[m][n].degree

    def max_numerator_degree(self) -> int:
        """Worst-case numerator degree across the (m, n) table."""
        if self.numerator_degrees is not None:
            return max(d for row in self.numerator_degrees for d in row)
        return max(num.degree for row in self.numerators for num in row)

    def evaluate(self, Delta):
        """Collapse to the numeric (u, ubar)-derivative table at this Delta.

        Real Delta is preserved as mp.mpf so the standard real-c, real-h
        evaluation stays in real arithmetic end to end.  Dispatches to the
        flint arb/acb path for real-coefficient tables.
        """
        return self._evaluate_at_work(_evaluation_number(Delta))

    def _evaluate_at_work(self, Delta):
        # Flint-native path for real-coefficient tables
        if self._is_real_table():
            if isinstance(Delta, mp.mpf):
                return self._evaluate_arb(Delta)
            return self._evaluate_acb(Delta)
        pref = self.exp_base**Delta
        D = self.denominator.evaluate(Delta)
        return [
            [pref * num.evaluate(Delta) / D for num in row]
            for row in self.numerators
        ]

    def _is_real_table(self) -> bool:
        """Check if all polynomial data is real (mp.mpf field).

        Treats mp.mpc with imag==0 as real (common after regularization).
        """
        exp_real = isinstance(self.exp_base, mp.mpf) or (
            isinstance(self.exp_base, mp.mpc) and self.exp_base.imag == 0
        )
        return (
            exp_real
            and self.denominator.field is mp.mpf
            and all(
                num.field is mp.mpf
                for row in self.numerators
                for num in row
            )
        )

    def _build_arb_cache(self):
        """Build flint arb_poly representations for all polynomials.

        Returns (arb_denom, arb_nums, exp_base_arb).
        Uses _get_arb() so polynomials with a cached arb_poly (e.g. from
        _build_G_numerators_arb) avoid the mpf->arb round-trip.
        """
        from flint import arb, ctx as flint_ctx
        prec_bits = int(mp.mp.dps * 3.33) + 20
        flint_ctx.prec = prec_bits
        arb_denom = self.denominator._get_arb()
        arb_nums = [
            [num._get_arb() for num in row]
            for row in self.numerators
        ]
        # exp_base may be mpc(real, 0) after regularization
        eb = self.exp_base
        if isinstance(eb, mp.mpc):
            eb = mp.mpf(eb.real)
        exp_base_arb = arb(eb)
        return arb_denom, arb_nums, exp_base_arb

    def _evaluate_arb(self, Delta):
        """Evaluate at a real Delta using flint arb arithmetic.

        Builds arb_poly for all polynomials (not cached — safe for
        multiprocessing). Returns list-of-lists of mp.mpf values.
        """
        from flint import arb, ctx as flint_ctx
        prec_bits = int(mp.mp.dps * 3.33) + 20
        flint_ctx.prec = prec_bits
        ndigits = int(mp.mp.dps) + 5
        arb_denom, arb_nums, exp_base_arb = self._build_arb_cache()
        d_arb = arb(Delta)
        pref_arb = exp_base_arb ** d_arb
        D_arb = arb_denom(d_arb)
        pref_over_D = pref_arb / D_arb
        return [
            [mp.mpf((arb_nums[m][n](d_arb) * pref_over_D).str(ndigits, radius=False))
             for n in range(len(row))]
            for m, row in enumerate(arb_nums)
        ]

    def _evaluate_acb(self, Delta):
        """Evaluate at a complex Delta using flint acb arithmetic.

        Uses real arb_poly coefficients promoted to acb for the evaluation.
        Returns list-of-lists of mp.mpc values.
        """
        from flint import arb, acb, acb_poly, ctx as flint_ctx
        prec_bits = int(mp.mp.dps * 3.33) + 20
        flint_ctx.prec = prec_bits
        ndigits = int(mp.mp.dps) + 5
        # Build acb_poly from cached arb_poly when available (avoids mpf round-trip).
        acb_denom = acb_poly(self.denominator._get_arb())
        acb_nums = [
            [acb_poly(num._get_arb()) for num in row]
            for row in self.numerators
        ]
        eb = self.exp_base
        if isinstance(eb, mp.mpc):
            eb = mp.mpf(eb.real)
        exp_base_acb = acb(arb(eb))
        d_acb = acb(arb(mp.re(Delta)), arb(mp.im(Delta)))
        pref_acb = exp_base_acb ** d_acb
        D_acb = acb_denom(d_acb)
        pref_over_D = pref_acb / D_acb
        return [
            [_acb_to_mpc((acb_nums[m][n](d_acb) * pref_over_D), ndigits)
             for n in range(len(row))]
            for m, row in enumerate(acb_nums)
        ]

    def evaluate_many(self, deltas):
        """Evaluate at multiple real Delta values, building arb_poly once.

        Falls back to per-Delta evaluate() for complex tables or complex
        Deltas.
        """
        deltas = list(deltas)
        if not deltas:
            return []
        values = [_evaluation_number(d) for d in deltas]
        if self._is_real_table() and all(isinstance(d, mp.mpf) for d in values):
            return self._evaluate_many_arb(values)
        return [self._evaluate_at_work(d) for d in values]

    def _evaluate_many_arb(self, deltas):
        """Batch-evaluate at multiple real Deltas using flint.

        Builds arb_poly representations once and reuses them for all Deltas.
        Returns list of list-of-lists of mp.mpf values.
        """
        from flint import arb, ctx as flint_ctx
        prec_bits = int(mp.mp.dps * 3.33) + 20
        flint_ctx.prec = prec_bits
        ndigits = int(mp.mp.dps) + 5
        arb_denom, arb_nums, exp_base_arb = self._build_arb_cache()
        results = []
        for delta in deltas:
            d_arb = arb(delta)
            pref_arb = exp_base_arb ** d_arb
            D_arb = arb_denom(d_arb)
            pref_over_D = pref_arb / D_arb
            results.append([
                [mp.mpf((arb_nums[m][n](d_arb) * pref_over_D).str(ndigits, radius=False))
                 for n in range(len(row))]
                for m, row in enumerate(arb_nums)
            ])
        return results


def _table_value_scale(table) -> mp.mpf:
    return max(abs(mp.mpc(value)) for row in table for value in row)


def _max_table_value_error(table_a, table_b) -> mp.mpf:
    worst = mp.mpf(0)
    for row_a, row_b in zip(table_a, table_b):
        for value_a, value_b in zip(row_a, row_b):
            worst = max(worst, abs(mp.mpc(value_a) - mp.mpc(value_b)))
    return worst


def _removable_pole_probe_offsets() -> tuple:
    return (
        mp.mpc(mp.mpf("0.0"), mp.mpf("0.5")),
        mp.mpc(mp.mpf("0.375"), mp.mpf("0.875")),
        mp.mpc(mp.mpf("-0.625"), mp.mpf("1.125")),
    )


def _removable_pole_validation_tol() -> mp.mpf:
    return mp.mpf(10) ** (-(mp.mp.dps - 8))


def _flint_validation_gray_zone_factor() -> mp.mpf:
    """Tiny relative slack for strip validation near the tolerance edge.

    Both the Flint quotient path and the exact polynomial-quotient path can
    miss truly removable factors by a hair at very high dps: the validation
    probes compare two nearly identical large cancellations and the residual can
    sit slightly above the nominal 10^(-(dps-8)) threshold even though the
    factor is removable and the direct-build oracle agrees.

    A small fixed multiplicative gray band keeps the validator robust to those
    threshold-edge false negatives while remaining astronomically tighter than
    any genuinely bad strip we have observed.
    """
    return mp.mpf("32")


def _any_numerator_nonzero_at(tables_by_coordinate: dict, root) -> bool:
    """Return True as soon as any numerator polynomial evaluates to a clearly
    nonzero value at `root`.  Used in the stripping loop after the pre-scan
    has identified `root` as a candidate.

    The natural scale for |p(root)| is max|coefficient|: a genuinely nonzero
    evaluation must be at least tol * that scale above floating-point noise,
    while a genuinely zero evaluation sits at O(eps * scale * degree).
    """
    tol = _removable_pole_validation_tol()
    root_c = mp.mpc(root)
    for table in tables_by_coordinate.values():
        for row in table.numerators:
            for num in row:
                val_abs = abs(mp.mpc(num.evaluate(root_c)))
                scale = max((abs(mp.mpc(c)) for c in num.coeffs), default=mp.mpf(1))
                if val_abs > tol * max(scale, mp.mpf(1)):
                    return True
    return False


def _removable_pole_guard_dps(degree: int, root) -> int:
    """Working precision needed to resolve a degree-`degree` numerator at `root`.

    Evaluating a degree-d polynomial at ``|root| = R`` loses about
    ``d * log10(R)`` digits to cancellation, so we add that many guard digits
    (plus a fixed margin) on top of the base working precision. Sizing the
    guard *per root* -- rather than from the global maximum root magnitude --
    keeps the pre-scan cheap at the small-magnitude poles near the physical
    region (where the cancellation is mild) while still fully resolving the
    far-out poles.
    """
    mag = abs(mp.mpc(root))
    guard = int(degree * float(mp.log10(max(mp.mpf(2), mag)))) + 20
    return mp.mp.dps + guard


def _all_numerators_zero_at(tables_by_coordinate, root, tol, guard_dps) -> bool:
    """True iff every numerator entry reads as zero at `root`.

    A Kac pole is removable only if EVERY numerator entry (across all
    coordinate tables) vanishes there, so this returns ``False`` as soon as one
    entry is clearly nonzero -- ``|num(root)| > tol * max(|coeff|, 1)`` -- and
    ``True`` only when every entry is at or below that threshold.

    This is the same zero/nonzero criterion as :func:`_any_numerator_nonzero_at`
    (it is its negation), but `tol` is passed in explicitly so it stays fixed at
    the *base* working precision while the evaluation runs at `guard_dps`.
    Resolving the high-degree cancellation at guard precision -- instead of at
    working precision, where roundoff makes a generic nonzero numerator read as
    ambiguous -- lets the generic-c pre-scan rule a flagged pole out without
    paying the expensive quotient+probe validation, while keeping the decision
    boundary identical to the loop's.
    """
    root_c = mp.mpc(root)
    with mp.workdps(guard_dps):
        for table in tables_by_coordinate.values():
            for row in table.numerators:
                for num in row:
                    scale = max(
                        (abs(mp.mpc(c)) for c in num.coeffs), default=mp.mpf(1)
                    )
                    val_abs = abs(mp.mpc(num.evaluate(root_c)))
                    if val_abs > tol * max(scale, mp.mpf(1)):
                        return False
    return True


def _removable_pole_root_tol() -> mp.mpf:
    return mp.mpf(10) ** (-(mp.mp.dps - 10))


def _shared_exact_roots(tables_by_coordinate: dict) -> list:
    coordinates = tuple(tables_by_coordinate)
    first_roots = list(tables_by_coordinate[coordinates[0]].denominator_roots)
    tol = _removable_pole_root_tol()
    for coord in coordinates[1:]:
        roots = tables_by_coordinate[coord].denominator_roots
        if len(roots) != len(first_roots):
            raise ValueError("coordinate tables disagree on the pole count")
        for root_a, root_b in zip(roots, first_roots):
            if abs(mp.mpc(root_a) - mp.mpc(root_b)) > tol:
                raise ValueError(
                    "coordinate tables disagree on the denominator roots"
                )
    return first_roots


def _quotient_table_by_root_index(
    table: DeltaDerivTable,
    root_index: int,
) -> DeltaDerivTable:
    root = table.denominator_roots[root_index]
    return DeltaDerivTable(
        exp_base=table.exp_base,
        denominator=table.denominator.quotient_by_linear(root),
        denominator_roots=(
            list(table.denominator_roots[:root_index])
            + list(table.denominator_roots[root_index + 1 :])
        ),
        numerators=[
            [num.quotient_by_linear(root) for num in row]
            for row in table.numerators
        ],
        spin=table.spin,
        root_provenance=(
            None
            if table.root_provenance is None
            else list(table.root_provenance[:root_index])
            + list(table.root_provenance[root_index + 1 :])
        ),
    )


def _maybe_real_polynomial(poly: Polynomial) -> Polynomial:
    if poly.field is mp.mpf:
        return poly
    tol = _removable_pole_root_tol()
    if all(
        abs(mp.im(mp.mpc(coeff))) <= tol * max(abs(mp.re(mp.mpc(coeff))), mp.mpf(1))
        for coeff in poly.coeffs
    ):
        return Polynomial(
            [mp.mpf(mp.re(mp.mpc(coeff))) for coeff in poly.coeffs],
            field=mp.mpf,
        )
    return poly


def _is_zero_polynomial(poly: Polynomial) -> bool:
    return all(coeff == 0 for coeff in poly.coeffs)


def _tables_have_real_coefficients(tables_by_coordinate: dict) -> bool:
    return all(
        table.denominator.field is mp.mpf
        and all(num.field is mp.mpf for row in table.numerators for num in row)
        for table in tables_by_coordinate.values()
    )


def _quotient_by_real_quadratic(poly: Polynomial, root, partner) -> Polynomial:
    re_root = mp.mpf(mp.re(mp.mpc(root)))
    im_root = abs(mp.im(mp.mpc(root)))
    field = poly.field
    divisor_linear = field(-2 * re_root)
    divisor_constant = field(re_root * re_root + im_root * im_root)
    descending = [field(coeff) for coeff in reversed(poly.coeffs)]
    degree = poly.degree
    if degree < 2:
        if _is_zero_polynomial(poly):
            return Polynomial.zero(field=field)
        raise ValueError("cannot divide degree < 2 polynomial by quadratic")
    quotient_desc = [field(0)] * (degree - 1)
    for i in range(degree - 1):
        lead = descending[i]
        quotient_desc[i] = lead
        descending[i + 1] -= lead * divisor_linear
        descending[i + 2] -= lead * divisor_constant
    quotient = Polynomial(reversed(quotient_desc), field=field)
    return _maybe_real_polynomial(quotient)


def _conjugate_partner_index(roots, root_index: int) -> int | None:
    root = mp.mpc(roots[root_index])
    tol = _removable_pole_root_tol()
    if mp.im(root) == 0:
        return None
    target = mp.conj(root)
    for index, other in enumerate(roots):
        if index == root_index:
            continue
        other_val = mp.mpc(other)
        if mp.im(root) * mp.im(other_val) < 0 and abs(other_val - target) < tol:
            return index
    return None


def _quotient_poly_by_root_indices(poly: Polynomial, roots_to_remove) -> Polynomial:
    if len(roots_to_remove) == 2:
        root, partner = roots_to_remove
        if abs(mp.mpc(partner) - mp.conj(mp.mpc(root))) < _removable_pole_root_tol():
            return _quotient_by_real_quadratic(poly, root, partner)
    quotient = poly
    for root in roots_to_remove:
        if quotient.degree < 1:
            if _is_zero_polynomial(quotient):
                return Polynomial.zero(field=quotient.field)
            raise ValueError("cannot divide a nonzero constant by (x - root)")
        quotient = quotient.quotient_by_linear(root)
    return _maybe_real_polynomial(quotient)


def _quotient_table_by_root_indices(
    table: DeltaDerivTable,
    root_indices,
) -> DeltaDerivTable:
    root_indices = tuple(sorted(root_indices))
    roots_to_remove = [table.denominator_roots[index] for index in root_indices]
    keep_roots = [
        root
        for index, root in enumerate(table.denominator_roots)
        if index not in set(root_indices)
    ]
    denominator = _quotient_poly_by_root_indices(table.denominator, roots_to_remove)
    numerators = [
        [_quotient_poly_by_root_indices(num, roots_to_remove) for num in row]
        for row in table.numerators
    ]
    return DeltaDerivTable(
        exp_base=table.exp_base,
        denominator=denominator,
        denominator_roots=keep_roots,
        numerators=numerators,
        spin=table.spin,
        root_provenance=(
            None
            if table.root_provenance is None
            else [
                provenance
                for index, provenance in enumerate(table.root_provenance)
                if index not in set(root_indices)
            ]
        ),
        # Stripping removable poles is value-preserving, so the numeric block
        # sampler (which returns the block value) stays valid on the quotient.
        numeric_sampler=table.numeric_sampler,
        oracle_metadata=table.oracle_metadata,
    )


def _root_indices_from_probe_roots(roots, probe_roots) -> tuple[int, ...]:
    """Resolve strip probe roots back to denominator indices.

    The strip validators work in terms of root values so the same probe-root
    sequence can be reused as the table shrinks. For fallback validation we
    need the corresponding quotient-table indices on the current table.
    """
    tol = _removable_pole_root_tol()
    used: set[int] = set()
    indices: list[int] = []
    for probe_root in probe_roots:
        match = next(
            (
                index for index, root in enumerate(roots)
                if index not in used and abs(mp.mpc(root) - mp.mpc(probe_root)) < tol
            ),
            None,
        )
        if match is None:
            raise ValueError(f"could not resolve strip root {probe_root} in denominator")
        used.add(match)
        indices.append(match)
    return tuple(indices)


def _candidate_tables_after_root_division_by_value(
    original_tables: dict,
    probe_roots,
) -> dict:
    """Build exact quotient tables removing the specified probe-root values."""
    first_table = next(iter(original_tables.values()))
    root_indices = _root_indices_from_probe_roots(
        first_table.denominator_roots, probe_roots
    )
    return {
        coord: _quotient_table_by_root_indices(table, root_indices)
        for coord, table in original_tables.items()
    }


def _flint_division_cache(table: DeltaDerivTable, divisor, divisor_degree: int):
    """Build acb_poly quotients dividing every table polynomial by `divisor`.

    `divisor` is a monic acb_poly strip factor: a linear ``(x - root)`` for a
    real removable pole or the real quadratic ``(x - root)(x - conj root)`` for
    a conjugate pair.  Staying in flint space through validation keeps rejected
    candidates from paying a Polynomial round-trip and then rebuilding fresh
    flint caches on the first probe evaluation.
    """
    from flint import arb, acb, acb_poly, ctx as flint_ctx

    from .polynomial import _mpf_to_acb

    prec_bits = int(mp.mp.dps * 3.33) + 20
    flint_ctx.prec = prec_bits
    zero_poly = acb_poly([acb(arb(0))])

    def quotient(poly: Polynomial):
        if poly.degree < divisor_degree:
            if _is_zero_polynomial(poly):
                return zero_poly
            raise ValueError(
                "cannot divide a lower-degree nonzero polynomial by the strip factor"
            )
        return poly._get_acb() // divisor

    return {
        "exp_base": _mpf_to_acb(table.exp_base),
        "denominator": quotient(table.denominator),
        "numerators": [
            [quotient(num) for num in row]
            for row in table.numerators
        ],
    }


def _evaluate_flint_division_cache(table_cache, Delta):
    """Evaluate a flint-space quotient cache at one complex probe Delta."""
    from flint import ctx as flint_ctx

    from .polynomial import _mpf_to_acb

    prec_bits = int(mp.mp.dps * 3.33) + 20
    flint_ctx.prec = prec_bits
    ndigits = int(mp.mp.dps) + 5
    Delta_acb = _mpf_to_acb(Delta)
    pref_over_D = table_cache["exp_base"] ** Delta_acb / table_cache["denominator"](
        Delta_acb
    )
    return [
        [_acb_to_mpc(num(Delta_acb) * pref_over_D, ndigits) for num in row]
        for row in table_cache["numerators"]
    ]


def _tables_match_after_flint_division(
    original_tables: dict,
    probe_roots,
    divisor,
    divisor_degree: int,
) -> bool:
    """Validate a strip candidate entirely in flint space.

    `probe_roots` anchors the validation probes -- one root for a linear factor,
    both members of a conjugate pair for the real quadratic factor -- while
    `divisor` is the corresponding monic acb_poly strip factor.  The before
    values come from the original tables exactly as the pure-mpmath validator
    sees them; only the divided ``after`` values move into flint space.
    """
    tol = _removable_pole_validation_tol()
    quotient_caches = {
        coord: _flint_division_cache(table, divisor, divisor_degree)
        for coord, table in original_tables.items()
    }
    gray_factor = _flint_validation_gray_zone_factor()
    for probe_root in probe_roots:
        for offset in _removable_pole_probe_offsets():
            before_by_coord = None
            after_by_coord = None
            for scale in (1, 2, 4, 8):
                Delta = mp.mpc(probe_root) + scale * offset
                try:
                    before_by_coord = {
                        coord: table.evaluate(Delta)
                        for coord, table in original_tables.items()
                    }
                    after_by_coord = {
                        coord: _evaluate_flint_division_cache(
                            quotient_caches[coord], Delta
                        )
                        for coord in original_tables
                    }
                    break
                except ZeroDivisionError:
                    continue
            if before_by_coord is None or after_by_coord is None:
                raise ValueError(
                    f"could not find a non-pole validation probe near root {probe_root}"
                )

            for coord in original_tables:
                before = before_by_coord[coord]
                after = after_by_coord[coord]
                scale = max(_table_value_scale(before), _table_value_scale(after))
                err = _max_table_value_error(before, after)
                if scale == 0:
                    if err > gray_factor * tol:
                        return False
                elif err > gray_factor * tol * scale:
                    return False
    return True


def _flint_linear_divisor(root):
    """Monic acb_poly ``(x - root)`` at the current working precision."""
    from flint import arb, acb, acb_poly, ctx as flint_ctx

    from .polynomial import _mpf_to_acb

    flint_ctx.prec = int(mp.mp.dps * 3.33) + 20
    return acb_poly([-_mpf_to_acb(root), acb(arb(1))])


def _flint_conjugate_pair_divisor(root):
    """Monic real quadratic ``(x - root)(x - conj root)`` as an acb_poly.

    Built as ``x^2 - 2 Re(root) x + |root|^2`` to match
    `_quotient_by_real_quadratic`, so dividing real-coefficient tables by it
    keeps the strip real-coefficient.
    """
    from flint import arb, acb, acb_poly, ctx as flint_ctx

    from .polynomial import _mpf_to_acb

    flint_ctx.prec = int(mp.mp.dps * 3.33) + 20
    re_root = mp.mpf(mp.re(mp.mpc(root)))
    im_root = abs(mp.im(mp.mpc(root)))
    constant = re_root * re_root + im_root * im_root
    linear = -2 * re_root
    return acb_poly([_mpf_to_acb(constant), _mpf_to_acb(linear), acb(arb(1))])


def _tables_match_after_single_root_division_flint(
    original_tables: dict,
    probe_root,
) -> bool:
    """Validate a real linear-factor strip candidate entirely in flint space."""
    return _tables_match_after_flint_division(
        original_tables, (probe_root,), _flint_linear_divisor(probe_root), 1
    )


def _tables_match_after_conjugate_pair_division_flint(
    original_tables: dict,
    root,
    partner,
) -> bool:
    """Validate a conjugate-pair real-quadratic strip candidate in flint space."""
    return _tables_match_after_flint_division(
        original_tables,
        (root, partner),
        _flint_conjugate_pair_divisor(root),
        2,
    )


def _tables_match_after_root_division(
    original_tables: dict,
    candidate_tables: dict,
    probe_roots,
) -> bool:
    tol = _removable_pole_validation_tol()
    gray_factor = _flint_validation_gray_zone_factor()
    for probe_root in probe_roots:
        for offset in _removable_pole_probe_offsets():
            before_by_coord = None
            after_by_coord = None
            for scale in (1, 2, 4, 8):
                Delta = mp.mpc(probe_root) + scale * offset
                try:
                    before_by_coord = {
                        coord: table.evaluate(Delta)
                        for coord, table in original_tables.items()
                    }
                    after_by_coord = {
                        coord: table.evaluate(Delta)
                        for coord, table in candidate_tables.items()
                    }
                    break
                except ZeroDivisionError:
                    continue
            if before_by_coord is None or after_by_coord is None:
                raise ValueError(
                    f"could not find a non-pole validation probe near root {probe_root}"
                )

            for coord in original_tables:
                before = before_by_coord[coord]
                after = after_by_coord[coord]
                scale = max(_table_value_scale(before), _table_value_scale(after))
                err = _max_table_value_error(before, after)
                if scale == 0:
                    if err > gray_factor * tol:
                        return False
                elif err > gray_factor * tol * scale:
                    return False
    return True


def _strip_removable_poles_from_exact_tables(tables_by_coordinate: dict) -> dict:
    """Strip denominator factors common to every exact-table numerator.

    The exact symbolic builders carry a fixed common Kac denominator even when
    some factors cancel out of the final block. We test each candidate linear
    factor by dividing it out of every numerator and validating the reduced
    table at a few nearby complex probes. This keeps the criterion internal and
    tied to the current working precision rather than to any user-facing
    tolerance parameter.

    For generic c (≥ 1 and not a minimal model) no poles are removable.  A
    pre-scan of numerators[0][0] at every root detects this in O(n_roots) and
    returns immediately, so the full quotient + probe machinery is only reached
    when at least one root looks removable (c < 1 degenerate-external case).
    """
    if not tables_by_coordinate:
        raise ValueError("tables_by_coordinate must not be empty")

    _shared_exact_roots(tables_by_coordinate)

    # Pre-scan: a root is removable only if ALL numerators vanish there. The
    # (0,0) entry is the cheapest single filter, but it is the undifferentiated
    # block VALUE, which genuinely vanishes at some Kac poles even for generic c
    # (where no factor is removable), so a (0,0)-only scan cannot rule those
    # poles out on its own. The scan runs in two stages: flag the poles
    # where (0,0) reads ~0, then
    # confirm each flagged pole against ALL numerator entries -- a single
    # clearly-nonzero entry eliminates the candidate. For generic c every
    # flagged pole is ruled out and we return immediately.
    # Scale by max|coefficient| so the threshold tracks the polynomial's natural
    # magnitude regardless of how large or small the coefficients happen to be.
    tol = _removable_pole_validation_tol()
    first_table = next(iter(tables_by_coordinate.values()))
    first_num = first_table.numerators[0][0]
    first_num_scale = max(
        (abs(mp.mpc(c)) for c in first_num.coeffs), default=mp.mpf(1)
    )
    first_num_threshold = tol * max(first_num_scale, mp.mpf(1))
    # Evaluate the pre-scan with enough guard precision to resolve the
    # high-degree cancellation. A degree-d polynomial evaluated at |x| up to R
    # loses ~ d*log10(R) digits, so at the user's working precision a generic
    # (genuinely nonzero) numerator can read as ~0 at a Kac pole -- falsely
    # flagging it removable and triggering the expensive quotient+probe loop
    # for nothing (e.g. lam=11, c=10, q_order=44: 42 spurious candidates at
    # dps=80, 0 at dps>=200). The coefficients themselves are accurate; only
    # the *evaluation* needs guarding, and only here in the O(n_roots) pre-scan,
    # so the generic-c case returns immediately. The guard is sized per root
    # (from each root's own magnitude) so the small-magnitude poles near the
    # physical region -- where the (0,0) block value usually vanishes for
    # generic c -- stay cheap to resolve.
    roots = first_table.denominator_roots
    degree = first_num.degree
    flagged_roots = []
    for root in roots:
        with mp.workdps(_removable_pole_guard_dps(degree, root)):
            if abs(mp.mpc(first_num.evaluate(mp.mpc(root)))) <= first_num_threshold:
                flagged_roots.append(root)
    if not flagged_roots:
        return dict(tables_by_coordinate)
    # Confirm each flagged pole against ALL numerator entries using the same
    # working-precision zero/nonzero criterion the loop uses, but evaluated at
    # per-root guard precision. `tol` is the base working tolerance, so genuine
    # extrapolation-zero poles (recursion-singular / regularized c, zero only to
    # base precision) are still classified as candidates and stripped in the
    # loop, while a generic-c pole with some clearly-nonzero entry is ruled out
    # here without paying the quotient+probe validation.
    if all(
        not _all_numerators_zero_at(
            tables_by_coordinate, root, tol, _removable_pole_guard_dps(degree, root)
        )
        for root in flagged_roots
    ):
        return dict(tables_by_coordinate)

    current = dict(tables_by_coordinate)
    root_index = 0
    while root_index < len(next(iter(current.values())).denominator_roots):
        roots = next(iter(current.values())).denominator_roots
        probe_root = roots[root_index]
        partner_index = _conjugate_partner_index(roots, root_index)
        if _any_numerator_nonzero_at(current, probe_root):
            root_index += 1
            continue

        if partner_index is not None and root_index < partner_index:
            partner_root = roots[partner_index]
            # Validate the real-quadratic strip in flint space first; build the
            # Polynomial quotient only once the candidate is accepted, so
            # rejected pairs never pay the round-trip.
            if _tables_match_after_conjugate_pair_division_flint(
                current, probe_root, partner_root
            ):
                current = {
                    coord: _quotient_table_by_root_indices(
                        table, (root_index, partner_index)
                    )
                    for coord, table in current.items()
                }
                continue

        if partner_index is not None and _tables_have_real_coefficients(current):
            root_index += 1
            continue

        if abs(mp.im(mp.mpc(probe_root))) <= _removable_pole_root_tol():
            if _tables_match_after_single_root_division_flint(current, probe_root):
                current = {
                    coord: _quotient_table_by_root_indices(table, (root_index,))
                    for coord, table in current.items()
                }
                continue
            root_index += 1
            continue

        candidate = {
            coord: _quotient_table_by_root_indices(table, (root_index,))
            for coord, table in current.items()
        }
        if _tables_match_after_root_division(current, candidate, (probe_root,)):
            current = candidate
            continue

        root_index += 1
    return current


def _strip_removable_poles_from_exact_table(
    table: DeltaDerivTable,
) -> DeltaDerivTable:
    return _strip_removable_poles_from_exact_tables({"_": table})["_"]


def _provenance_chiral_labels(prov) -> tuple:
    """Extract chiral (r, s) labels from a root_provenance entry.

    Chiral-level entry shape: ``((r, s),)`` (or with a "chiral" side tag).
    G-level entry shape: ``(side, ((r, s),))`` with side in {"left", "right"}.
    """
    if isinstance(prov, tuple) and len(prov) == 2 and isinstance(prov[0], str):
        return prov[1]
    return prov


def strip_chiral_labels_from_table(
    table: DeltaDerivTable,
    removable_chiral_labels,
) -> DeltaDerivTable:
    """Algebraically strip Delta roots whose chiral Kac label is removable.

    `removable_chiral_labels` is either a set of ``(r, s)`` tuples (applied
    uniformly to all sides) or a dict mapping side tag -> set of ``(r, s)``,
    where the side tag matches the first element of the provenance entry
    (``"left"`` / ``"right"`` for G/B tables, ``"chiral"`` for chiral-F
    tables). Per-side dispatch is essential for spinning blocks where the
    h-side and hbar-side carry independent external weights and so have
    independent K-vanishing sets.

    Uses `table.root_provenance` to map each Delta root back to the chiral
    (r, s) pair it descended from, then divides numerators and denominator
    by ``(Delta - root)`` via `_quotient_table_by_root_indices`. No numerator
    probe -- the decision of which roots to strip comes from the caller's
    algebraic K-vanishing analysis.
    """
    if isinstance(removable_chiral_labels, dict):
        per_side = {
            side: set(labels) for side, labels in removable_chiral_labels.items()
        }
        if not any(per_side.values()):
            return table
        uniform = None
    else:
        per_side = None
        uniform = set(removable_chiral_labels)
        if not uniform:
            return table

    if not table.root_provenance:
        return table

    indices = []
    for idx, prov in enumerate(table.root_provenance):
        if isinstance(prov, tuple) and len(prov) == 2 and isinstance(prov[0], str):
            side, labels = prov[0], prov[1]
        else:
            side, labels = None, prov
        side_set = uniform if uniform is not None else per_side.get(side, set())
        for label in labels:
            if label in side_set:
                indices.append(idx)
                break
    if not indices:
        return table
    return _quotient_table_by_root_indices(table, indices)


def _tag_root_provenance(root_provenance, side: str):
    if root_provenance is None:
        return None
    return [(side, tuple(provenance)) for provenance in root_provenance]


def _flint_taylor_series(z_of_u, u_symmetric, n_terms, sigma, theta_exp, z_shift_val, h2, h3, N):
    """Compute q, log(E), and A Taylor series using flint acb_series.

    Returns (q_series, log_e_series, a_series) as lists of mp.mpf coefficients.

    python-flint truncates every power-series operation at the GLOBAL
    ``ctx.cap`` (default 10), NOT at any per-object construction kwarg. We must
    raise ``ctx.cap`` to ``n_terms`` for the duration of the computation,
    otherwise coefficients of order >= ctx.cap are silently dropped -- which
    corrupted all symbolic block tables at Lambda >= 11. The cap is restored
    afterwards so the global flint state is left unchanged.
    """
    from flint import ctx as flint_ctx

    old_cap = flint_ctx.cap
    flint_ctx.cap = n_terms
    try:
        return _flint_taylor_series_impl(
            z_of_u, u_symmetric, n_terms, sigma, theta_exp, z_shift_val, h2, h3, N)
    finally:
        flint_ctx.cap = old_cap


def _flint_taylor_series_impl(z_of_u, u_symmetric, n_terms, sigma, theta_exp, z_shift_val, h2, h3, N):
    """Inner body of `_flint_taylor_series`; run with ``ctx.cap == n_terms``.

    Uses flint's compiled power-series arithmetic (elliptic_k, exp, log on
    acb_series) which avoids the O(10×dps) precision boost that mp.taylor's
    Richardson extrapolation requires. Typically 100-2000× faster than mp.taylor.
    """
    from flint import arb, acb, arb_series, acb_series, ctx as flint_ctx

    prec_bits = int(mp.mp.dps * 3.33) + 20
    flint_ctx.prec = prec_bits
    ndigits = mp.mp.dps + 5

    _to_acb = _mpf_to_acb     # exact, and accepts Python complex / Fraction (the string round trip did not)

    u0 = _to_acb(u_symmetric)
    sigma_f = _to_acb(sigma)
    theta_exp_f = _to_acb(theta_exp)
    z_shift_f = _to_acb(z_shift_val)
    h2_f = _to_acb(h2)
    h3_f = _to_acb(h3)
    N_f = _to_acb(N)

    # Detect coordinate family and build z_series accordingly.
    # q-coordinate: z_of_u is z_of_q, meaning q_of_u = identity.
    # y-coordinate: z_of_u is z_of_y, rational: z = (1+y)^2 / (2*(1+y^2)).
    # z-coordinate: z_of_u is identity, z = u directly.
    is_q_coord = (z_of_u is z_of_q)
    is_y_coord = (z_of_u is z_of_y)

    q_series_known = None
    if is_q_coord:
        # q-coordinate: the nome IS the coordinate, so q(u) = u exactly and
        #   q_series = u0 + (u - u0).
        # The prefactor still needs z(q). Although theta2(0,q) alone carries a
        # q^{1/4}, the ratio z(q) = (theta2/theta3)^4 is a clean *integer* power
        # series in q (the q^{1/4} becomes q^1 under the 4th power), so flint can
        # build it directly:
        #   theta2(0,q) = 2 q^{1/4} (sum_{n>=0} q^{n(n+1)}),  exponents 0,2,6,12,...
        #   theta3(0,q) = 1 + 2 sum_{n>=1} q^{n^2},          exponents 1,4,9,16,...
        #   z(q) = 16 q (sum_{n>=0} q^{n(n+1)})^4 / (1 + 2 sum_{n>=1} q^{n^2})^4.
        # Bound the omitted Taylor coefficients at the actual nome, including
        # the derivative order; a symmetric-point term count is insufficient.
        q_series_known = acb_series([u0, 1])
        from .theta_series import theta_series
        theta2_reduced, theta3_z, _ = theta_series(q_series_known, full=True)
        z_series = (acb_series([acb(16)]) * q_series_known
                    * (theta2_reduced ** 4) / (theta3_z ** 4))
    elif is_y_coord:
        # y-coordinate: z(y) = (1+y)^2 / (2*(1+y^2))
        y_series = acb_series([u0, 1])
        one = acb(1)
        one_plus_y = acb_series([one + u0, 1])
        one_plus_y_sq = one_plus_y * one_plus_y
        y_sq = y_series * y_series
        denom = acb_series([one + u0 * u0, 2 * u0, one])
        z_series = one_plus_y_sq / (acb_series([acb(2)]) * denom)
    else:
        # z-coordinate (identity): z = u
        z_series = acb_series([u0, 1])

    # one_minus_z series
    z0_coeffs = z_series.coeffs()
    omz0 = acb(1) - z0_coeffs[0]
    omz_rest = [-c for c in z0_coeffs[1:]]
    one_minus_z = acb_series([omz0] + omz_rest)

    # q(z) = exp(-pi * K(1-z) / K(z)); for the q-coordinate the nome is the
    # coordinate itself, so reuse the exact q_series and skip the elliptic_k.
    if q_series_known is not None:
        q_series = q_series_known
    else:
        K_series = z_series.elliptic_k()
        Kp_series = one_minus_z.elliptic_k()
        pi_s = acb_series([acb(arb.pi())])
        q_series = (-pi_s * Kp_series / K_series).exp()

    # log(E(z)) = log(16*q/N)
    sixteen_s = acb_series([acb(16)])
    sixteen_over_N = acb_series([acb(16) / N_f])
    E_series = sixteen_over_N * q_series
    log_E_series = E_series.log()

    # A(z) = (16q)^{-sigma} * z^{sigma+z_shift} * (1-z)^{sigma-h2-h3} * theta3^{theta_exp}
    sixteen_q = sixteen_s * q_series
    factor1 = ((-sigma_f) * sixteen_q.log()).exp()
    exp_z = sigma_f + z_shift_f
    factor2 = (exp_z * z_series.log()).exp()
    exp_omz = sigma_f - h2_f - h3_f
    factor3 = (exp_omz * one_minus_z.log()).exp()

    # theta3(0, q) = 1 + 2*sum_{n>=1} q^{n^2}
    from .theta_series import theta_series
    theta3_s = theta_series(q_series)[1]

    factor4 = (theta_exp_f * theta3_s.log()).exp()
    A_series = factor1 * factor2 * factor3 * factor4

    # Convert flint coefficients back to mpmath.
    # Use arb(2)**(-prec_bits+10) as the imaginary-part threshold — avoids
    # the float underflow that occurs when mp.mp.dps > ~308 (symbolic context).
    from flint import arb as _arb
    _im_tol = _arb(2) ** (-(prec_bits - 10))

    def _to_mpmath(coeffs_list):
        result = []
        for c_val in coeffs_list:
            re_str = c_val.real.str(ndigits, radius=False)
            im_val = c_val.imag
            if abs(im_val) <= _im_tol:
                result.append(mp.mpf(re_str))
            else:
                im_str = im_val.str(ndigits, radius=False)
                result.append(mp.mpc(mp.mpf(re_str), mp.mpf(im_str)))
        return result

    q_coeffs = _to_mpmath(q_series.coeffs()[:n_terms])
    log_e_coeffs = _to_mpmath(log_E_series.coeffs()[:n_terms])
    a_coeffs = _to_mpmath(A_series.coeffs()[:n_terms])

    # Pad with zeros if series is shorter than n_terms
    while len(q_coeffs) < n_terms:
        q_coeffs.append(mp.mpf(0))
    while len(log_e_coeffs) < n_terms:
        log_e_coeffs.append(mp.mpf(0))
    while len(a_coeffs) < n_terms:
        a_coeffs.append(mp.mpf(0))

    return q_coeffs, log_e_coeffs, a_coeffs


def _assemble_n_coeffs_flint(q_series, numerators, q_order, n_terms, field):
    """Vectorized ``N(z(u))`` numerator assembly via flint.

    Computes ``n_coeffs[m] = sum_{k=0}^{q_order} (coeff m of q(z(u))^k) *
    numerators[k]`` for ``m = 0 .. n_terms-1``. The two scalar Python loops this
    replaces are (A) the Cauchy-product build of ``q(z(u))^k`` and (B) the
    weighted sum of the degree-~``q_order`` numerator polynomials. (A) becomes a
    sequence of truncated flint series multiplies; (B) becomes a single flint
    matrix product ``W @ M`` with ``W[m][k] = coeff m of q^k`` and ``M[k][i] =
    coeff i of numerators[k]`` -- the same pattern used in ``_assemble_numerators``
    and the derivs ``Q_mat``. Returns ``n_terms`` Polynomials.
    """
    from flint import (
        arb, acb, arb_mat, acb_mat, arb_series, acb_series,
        ctx as flint_ctx,
    )
    from .polynomial import (
        _flint_prec_bits, _mpf_to_arb, _mpf_to_acb, _arb_to_mpf, _acb_to_mpc,
    )

    complex_field = (
        field is mp.mpc
        or any(n.field is mp.mpc for n in numerators)
        or any(isinstance(v, mp.mpc) and v.imag != 0 for v in q_series)
    )
    flint_ctx.prec = _flint_prec_bits()
    flint_ctx.cap = n_terms

    deg = max((len(n.coeffs) for n in numerators), default=1)

    if complex_field:
        to_f, zero_f, mat_cls = _mpf_to_acb, acb(0), acb_mat
        qs = acb_series([_mpf_to_acb(v) for v in q_series[:n_terms]])
        one_series = acb_series([acb(1)])
    else:
        to_f, zero_f, mat_cls = _mpf_to_arb, arb(0), arb_mat
        qs = arb_series([_mpf_to_arb(v) for v in q_series[:n_terms]])
        one_series = arb_series([arb(1)])

    # (A) q^k coefficient rows, each padded to n_terms.
    qk_rows = []
    cur = one_series
    for k in range(q_order + 1):
        if k > 0:
            cur = cur * qs
        coeffs = list(cur.coeffs())[:n_terms]
        if len(coeffs) < n_terms:
            coeffs = coeffs + [zero_f] * (n_terms - len(coeffs))
        qk_rows.append(coeffs)

    # (B) W @ M.  W[m][k] = coeff m of q^k (transpose of qk_rows); M[k][i] =
    # coeff i of numerators[k], zero-padded to the common degree.
    W = mat_cls([[qk_rows[k][m] for k in range(q_order + 1)]
                 for m in range(n_terms)])
    M_rows = []
    for k in range(q_order + 1):
        row = [to_f(x) for x in numerators[k].coeffs]
        if len(row) < deg:
            row = row + [zero_f] * (deg - len(row))
        M_rows.append(row)
    P = W * mat_cls(M_rows)   # (n_terms x deg)

    # The output field follows the actual arithmetic: a complex q-series or
    # complex numerators promote the result to mpc, exactly as field promotion
    # through Polynomial * / + would.
    out_field = mp.mpc if complex_field else mp.mpf
    out = []
    for m in range(n_terms):
        if complex_field:
            row = [_acb_to_mpc(P[m, i]) for i in range(deg)]
        else:
            row = [_arb_to_mpf(P[m, i]) for i in range(deg)]
        out.append(Polynomial(row, field=out_field))
    return out


def chiral_symbolic_useries(h_externals, n_terms: int, *, b, c, skeleton: bool = False, **kwargs) -> ChiralSymbolic:
    """:func:`_chiral_symbolic_useries_impl` (see there), near (not on) a singular locus at a verified guard
    precision (:func:`singularity.near_singular_compute`); a skeleton carries no values and is built directly."""
    from .numeric_input import parse_number

    n_terms = _integer_control('n_terms', n_terms, 1)
    kwargs['q_order'] = _integer_control('q_order', kwargs['q_order'])
    h_externals = tuple(parse_number(value) for value in h_externals)
    if skeleton:
        return _pin_identity_root(_chiral_symbolic_useries_impl(h_externals, n_terms, b=b, c=c, skeleton=True, **kwargs))
    from .singularity import near_singular_compute

    probe_h = mp.mpf("1.41421356237309504880")     # a generic weight (never a Kac weight)
    entry_prec = mp.mp.prec

    def build():
        # under a raised (guard) precision b is re-derived from c there: at large c the block is so sensitive to b
        # that its rounding at the caller's precision alone cost ~9 digits (c = 1e5); unguarded, b is exactly as given
        independent_b = b is not None and c_origin_of_b(b) is None
        charge = c_from_b(b) if independent_b else c
        b_build = b_from_c(charge) if (not independent_b and charge is not None and mp.mp.prec > entry_prec) else b
        return _chiral_symbolic_useries_impl(h_externals, n_terms, b=b_build, c=charge, **kwargs)

    return _pin_identity_root(near_singular_compute(
        c, f"chiral_symbolic:{n_terms}",
        build,
        h_externals=h_externals, q_order=kwargs["q_order"],
        fingerprint=lambda cs: [cs.chiral_coeff(m, probe_h) for m in range(n_terms)],
        cache_context=(b, kwargs['z_of_u'], kwargs['u_symmetric'], kwargs.get('chiral_norm'),
                       kwargs.get('z_shift'), kwargs.get('symbolic_guard', True)),
    ))


def _pin_identity_root(cs):
    """The identity Kac weight d_{1,1} = (Q^2 - (b + 1/b)^2)/4 vanishes for every c, but its assembled value is
    roundoff (-2e-217 at dps 171 on the imaginary-b path), which the interpolation rightly refuses as a near-zero
    pole; the exported root is pinned to exactly 0.  (Inside the recursion the roundoff value stays: a direct build
    at a singular c, the collision oracle, relies on never dividing by an exact zero.)"""
    provenance = cs.root_provenance or ()
    tol = mp.mpf(10) ** (-(mp.mp.dps // 2))
    for index, prov in enumerate(provenance):
        labels = prov[1] if isinstance(prov, tuple) and len(prov) == 2 and isinstance(prov[0], str) else prov
        if tuple(labels or ()) == ((1, 1),) and 0 < abs(cs.denominator_roots[index]) < tol:
            cs.denominator_roots[index] = cs.denominator_roots[index] * 0
    return cs


def _chiral_symbolic_useries_impl(
    h_externals,
    n_terms: int,
    *,
    b,
    c,
    z_of_u,
    u_symmetric,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
    symbolic_guard: bool = True,
    skeleton: bool = False,
) -> ChiralSymbolic:
    """Build the symbolic u-Taylor expansion of F(h, z(u)) around u_symmetric.

    `b`, `c` must both be supplied already resolved (this is an internal
    helper; the public table builders do the b/c bookkeeping). `chiral_norm`
    selects N (default 16). `z_shift` controls the z-exponent in the block
    prefactor (default 0, giving z^h behavior; pass -h1-h2 for crossing).

    With ``skeleton=True`` the numerator coefficients are not built: only the
    Kac denominator / roots / provenance and the per-u-power numerator
    *degrees* (``denominator.degree + m``) are returned, with
    ``numerators=None``. This is the cheap build for the numeric-sampler
    interpolation path, which needs degrees for node placement but takes sample
    values (and numeric pole ranking) from the recursion -- skipping the
    expensive q-series / H-series numerator assembly.
    """
    if isinstance(b, (mp.mpf, mp.mpc)):
        # at the working precision: the resolvers form b with 2x digits (large-c sensitivity), which a guarded build
        # (raised working precision) keeps; an unguarded one -- e.g. the direct build at a singular c, the collision
        # oracle, which relies on roundoff never producing an exact zero -- sees the b it always saw
        from .params import c_origin_of_b, _remember_b_origin

        origin = c_origin_of_b(b)
        b = +b
        if origin is not None:
            _remember_b_origin(b, origin)
    n_terms = _integer_control('n_terms', n_terms, 1)
    _check_externals(h_externals)
    q_order = _integer_control('q_order', q_order)
    # Symbolic builders require merged regularization at recognized singular c.
    # A minimal-model skeleton may supply Kac denominators and degrees for the
    # regularizer's collision strips; its direct values are not valid blocks.
    # Use regularize_delta_table for values, or singularity.bare_recursion_probes()
    # when only the bare pole multiplicities are needed.
    from .singularity import merged_singular_active, reject_singular_c

    if not merged_singular_active():
        # (under merged_singular_evaluators a table is built AT the singular c from the exact merged poles)
        reject_singular_c(
            c, b=b, q_order=q_order, h_externals=h_externals, where="symbolic block table",
            allow_minimal_models=skeleton,
        )
    working_dps = _symbolic_workdps_for_q_order(
        q_order,
        use_guard=symbolic_guard,
        c=c,
        h_externals=h_externals,
    )
    with _symbolic_work_context(q_order, use_guard=symbolic_guard, c=c, h_externals=h_externals):
        N = _resolve_chiral_norm(chiral_norm)

        # Keep real externals in mp.mpf so downstream Taylor / fusion / H_rat
        # stay in real arithmetic when the input data is mathematically real.
        def _coerce_ext(di):
            from .numeric_input import parse_number

            value = mp.convert(parse_number(di))
            return value.real if isinstance(value, mp.mpc) and value.imag == 0 else value

        h1, h2, h3, h4 = (_coerce_ext(di) for di in h_externals)
        # c likewise: demote mpc(real, 0) to mpf so sigma and theta_exp stay real.
        # (a Python float or Fraction is converted: in float arithmetic sigma would keep only 16 digits)
        c_eff = mp.convert(c)
        if isinstance(c_eff, mp.mpc) and c_eff.imag == 0:
            c_eff = mp.mpf(c_eff.real)
        sigma = (c_eff - 1) / 24
        theta_exp = (c_eff - 1) / 2 - 4 * (h1 + h2 + h3 + h4)

        def E_of_u(u):
            z = z_of_u(u)
            return 16 * q_from_z(z) / N

        # Resolve z_shift: default 0
        _z_shift_val = mp.mpf(0) if z_shift is None else _coerce_ext(z_shift)

        exp_base = E_of_u(u_symmetric)

        if skeleton:
            # Degree-only build: skip the q-series / H-series numerator assembly
            # entirely. Only the Kac denominator (cheap) and the structural
            # numerator degrees (denominator.degree + m) are needed downstream.
            from .singularity import merged_singular_active, recognized_singular_locus

            from .singularity import exact_locus_c

            c_here = (exact_locus_c(c if c is not None else c_from_b(b), q_order=q_order) if merged_singular_active()
                      else None)
            if c_here is not None:
                # AT the singular c: the exact merged pole structure of the one Laurent table
                from .recursion import merged_skeleton_rational
                H_rat = merged_skeleton_rational(h_externals, c0=c_here, q_order=q_order)
            else:
                H_rat = H_q_series_rational(h_externals, b=b, q_order=q_order)
            D = H_rat.denominator.degree
            return ChiralSymbolic(
                exp_base=exp_base,
                denominator=H_rat.denominator,
                denominator_roots=list(H_rat.denominator_roots),
                numerators=None,
                numerator_degrees=[D + m for m in range(n_terms)],
                root_provenance=list(H_rat.root_provenance or ()),
                working_dps=working_dps,
                q_order=q_order,
            )

        # s(u) = log E(z(u)) - log E(z_*); the constant term is zeroed explicitly.
        # flint acb_series path (100-2000x faster than mp.taylor).
        q_series, log_e_series, a_series = _flint_taylor_series(
            z_of_u,
            u_symmetric,
            n_terms,
            sigma,
            theta_exp,
            _z_shift_val,
            h2,
            h3,
            N,
        )

        # Pick the accumulator field from the Taylor series: real inputs at a
        # real symmetric point yield mp.mpf coefficients; any genuine complex
        # value upgrades to mp.mpc.
        def _series_field(series):
            for v in series:
                if isinstance(v, mp.mpc) and v.imag != 0:
                    return mp.mpc
            return mp.mpf

        field = _series_field(log_e_series) if _series_field(log_e_series) is mp.mpc else (
            mp.mpc
            if (_series_field(a_series) is mp.mpc or _series_field(q_series) is mp.mpc)
            else mp.mpf
        )
        s_series = [field(0)] + [
            v if isinstance(v, (mp.mpf, mp.mpc)) else field(v)
            for v in log_e_series[1:]
        ]

        from .singularity import exact_locus_c, merged_singular_active

        c_here = (exact_locus_c(c if c is not None else c_from_b(b), q_order=q_order) if merged_singular_active()
                  else None)
        if c_here is not None:
            # AT the singular c: the exact rational H of the merged poles (numerators included)
            from .recursion import merged_H_rational
            H_rat = merged_H_rational(h_externals, c0=c_here, q_order=q_order)
        else:
            H_rat = H_q_series_rational(h_externals, b=b, q_order=q_order)

        # N_{z(u)}(h) = sum_k N_k(h) * q(z(u))^k as a u-series with h-poly coeffs.
        n_coeffs = _assemble_n_coeffs_flint(
            q_series, H_rat.numerators, q_order, n_terms, field
        )

        exp_hs = exp_h_times(s_series, n_terms)
        a_hseries = HSeries.from_numeric(a_series)
        n_useries = HSeries(n_coeffs)
        numerator_useries = (exp_hs * a_hseries * n_useries).truncate(n_terms)

        return ChiralSymbolic(
            exp_base=exp_base,
            denominator=H_rat.denominator,
            denominator_roots=list(H_rat.denominator_roots),
            numerators=numerator_useries.coeffs,
            root_provenance=list(H_rat.root_provenance or ()),
            working_dps=working_dps,
            q_order=q_order,
        )


def _build_G_numerators_arb(p_h, p_hbar, m_max, n_max, swap, lambda_cut=None):
    """Assemble the G-table numerators N[m][n] = p_h[m] p_hbar[n] + swap p_hbar[m]
    p_h[n] entirely in flint arb (``swap`` = :func:`~._conventions.scalar_G_swap_coefficient`:
    +1, or 0 for the public spin-0 block, where p_h = p_hbar).

    The scalar Polynomial path materializes each of the two products back to
    mpf (~deg arb->mpf conversions apiece) and then adds in mpf; doing the two
    products *and* the combination in arb_poly costs one materialization per
    numerator instead of two and keeps the add at the flint working precision.
    The result Polynomials carry their arb_poly cache so the downstream
    Vandermonde evaluation (`_evaluate_table_flint_many`) reuses it rather than
    re-converting mpf->arb. Returns ``None`` for non-real tables (the caller
    keeps the mpmath Polynomial path).
    """
    n_top = max(m_max, n_max) + 1
    if any(p_h[i].field is not mp.mpf or p_hbar[i].field is not mp.mpf
           for i in range(n_top)):
        return None
    from flint import ctx as flint_ctx
    from .polynomial import _flint_prec_bits

    prec = _flint_prec_bits()
    flint_ctx.prec = prec
    ph = [p._get_arb() for p in p_h]
    phb = [p._get_arb() for p in p_hbar]
    out = [[None] * (n_max + 1) for _ in range(m_max + 1)]
    # Corner-skip: under the m + n <= Lambda truncation the (m, n) entries with
    # m + n > lambda_cut are discarded by the interpolation step, so skip the
    # (expensive) product and leave a zero placeholder. Symmetric in (m, n), so
    # the mirror slot out[n][m] is skipped consistently.
    zero = Polynomial.zero(field=mp.mpf)
    for m in range(m_max + 1):
        for n in range(n_max + 1):
            if out[m][n] is not None:
                continue
            if lambda_cut is not None and m + n > lambda_cut:
                out[m][n] = zero
                if m != n and n <= m_max and m <= n_max:
                    out[n][m] = zero
                continue
            if swap == 0:
                res = ph[m] * phb[n]
            else:
                cross = phb[m] * ph[n]
                res = ph[m] * phb[n] + (cross if swap > 0 else -cross)
            # Lazy mpf: the downstream Vandermonde sampler reads only _get_arb()
            # and degree, so defer the arb->mpf coefficient conversion (the churn).
            out[m][n] = Polynomial._from_arb_trusted(res, prec)

            if m != n and n <= m_max and m <= n_max:
                mirrored = res if swap >= 0 else -res
                out[n][m] = Polynomial._from_arb_trusted(mirrored, prec)
    return out


def _build_G_folded_numerators_flint(
    p_h,
    p_hbar,
    m_max: int,
    n_max: int,
    *,
    ell: int,
):
    """Assemble folded scalar-G numerators directly in flint space.

    This computes the non-folded G-combination

        N[m][n] = p_h[m] p_hbar[n] + sigma p_hbar[m] p_h[n],

    sigma = :func:`~._conventions.scalar_G_swap_coefficient` (+1 at every ``ell != 0``, 0 for
    the public spin-0 block),

    and applies the parity fold plan on the fly, avoiding construction of the
    full ``(M_total+1) x (M_total+1)`` non-folded table.
    """
    from flint import acb, acb_poly, arb, arb_poly, ctx as flint_ctx
    from .polynomial import _flint_prec_bits

    rows = len(p_h)
    cols = len(p_hbar)
    odd_factor = bool(fold_parity_marker(ell=ell))
    plan, target_size = _fold_plan(odd_factor, rows, cols, m_max, n_max)
    width = n_max + 1
    swap = scalar_G_swap_coefficient(ell)

    complex_out = any(p.field is mp.mpc for p in p_h) or any(
        p.field is mp.mpc for p in p_hbar
    )

    prec = _flint_prec_bits()
    flint_ctx.prec = prec

    if complex_out:
        ph = [p._get_acb() for p in p_h]
        phb = [p._get_acb() for p in p_hbar]
        accum = [acb_poly([acb(arb(0))]) for _ in range(target_size)]
        for m, n, contribs in plan:
            if swap == 0:
                res = ph[m] * phb[n]
            else:
                cross = phb[m] * ph[n]
                res = ph[m] * phb[n] + (cross if swap > 0 else -cross)
            for target_idx, weight in contribs:
                accum[target_idx] += res * acb(weight)
        # Lazy mpc: defer the acb->mpc conversion (the downstream Vandermonde
        # sampler reads _get_acb() + degree only). See _build_G_numerators_arb.
        return [
            [
                Polynomial._from_acb_trusted(accum[alpha * width + beta], prec)
                for beta in range(width)
            ]
            for alpha in range(m_max + 1)
        ]

    ph = [p._get_arb() for p in p_h]
    phb = [p._get_arb() for p in p_hbar]
    accum = [arb_poly([arb(0)]) for _ in range(target_size)]
    for m, n, contribs in plan:
        if swap == 0:
            res = ph[m] * phb[n]
        else:
            cross = phb[m] * ph[n]
            res = ph[m] * phb[n] + (cross if swap > 0 else -cross)
        for target_idx, weight in contribs:
            accum[target_idx] += res * arb(weight)
    # Lazy mpf: defer the arb->mpf conversion (see _build_G_numerators_arb).
    return [
        [
            Polynomial._from_arb_trusted(accum[alpha * width + beta], prec)
            for beta in range(width)
        ]
        for alpha in range(m_max + 1)
    ]


def _build_G_poly_table(
    cs: ChiralSymbolic, ell: int, m_max: int, n_max: int,
    lambda_cut: int | None = None,
    cs_n: ChiralSymbolic | None = None,
) -> DeltaDerivTable:
    """Non-folded (zzb-style) symbolic G table from one (or two) chiral expansions.

    G_{Delta, ell} = [B(h, hbar) + B(hbar, h)] / (1 + delta_{ell,0}), with
    h = (Delta+ell)/2, hbar = (Delta-ell)/2. Substituting those linear maps into
    P_m(h) and D(h) turns each into a polynomial in Delta. The ell = 0 entry is
    c_h[m] * c_h[n].

    Polynomial.__mul__ and compose_linear dispatch to flint arb_poly arithmetic
    automatically when coefficients are real, so no special-casing is needed here.

    ``cs`` carries the *m-axis* (z) chiral expansion. ``cs_n`` carries the
    *n-axis* (zbar) expansion for an **off-diagonal** ``z != zbar`` build; when it
    is ``None`` (or the same object as ``cs``) the table is diagonal and the
    m <-> n mirror-symmetric fast path is used (bit-identical to the single-cs
    build -- no extra work on the diagonal). The Kac denominator / roots are
    center-independent, so they come from ``cs`` in both cases.

    Skeleton chiral input (``cs.numerators is None``) yields a degree-only G
    table: the (m, n) numerator degree is ``deg(P_m) + deg(P_n)`` (the parity
    sum does not change the degree), so it is read off the chiral
    degree grid without building any coefficients. Degrees are center-
    independent, so the skeleton ignores ``cs_n``.
    """
    ell = _check_ell(ell)
    with _reuse_symbolic_work_context(cs, cs_n):
        half = mp.mpf(1) / 2
        ell_half = mp.mpf(ell) / 2
        n_top = max(m_max, n_max) + 1

        # Kac denominator / roots / provenance: built in both modes (cheap).
        d_h = cs.denominator.compose_linear(half, ell_half)
        d_hbar = cs.denominator.compose_linear(half, -ell_half)
        denominator = d_h * d_hbar
        roots = [2 * d - ell for d in cs.denominator_roots] + [
            2 * d + ell for d in cs.denominator_roots
        ]
        root_provenance = (
            (_tag_root_provenance(cs.root_provenance, "left") or [])
            + (_tag_root_provenance(cs.root_provenance, "right") or [])
        )

        # Single Delta-base: the geometric mean of the two halves' bases off the
        # diagonal (E_z^h E_zbar^hbar = GM^Delta * R^(ell/2); see the off-diagonal
        # branch), or just cs.exp_base on the diagonal.
        off_diagonal = cs_n is not None and cs_n is not cs
        exp_base_out = (
            mp.sqrt(cs.exp_base * cs_n.exp_base) if off_diagonal else cs.exp_base
        )

        if cs.numerators is None:
            # G[m][n] = p_h[m] p_hbar[n] + sign * p_hbar[m] p_h[n], with
            # deg(p_h[k]) = deg(p_hbar[k]) = cd[k] and equal leading
            # coefficients (compose_linear's leading coeff is shift-independent).
            # sign is +1 at every ell != 0 (0 for the public spin-0 block), so the
            # leading terms add -> degree cd[m]+cd[n].
            cd = cs.numerator_degrees
            # Corner-skip mirrors the coefficient path: entries with
            # m + n > lambda_cut are discarded at interpolation, so report
            # degree 0 (matching the zero placeholder) -- otherwise the corner's
            # full degree would over-size the shared sample-point set relative to
            # the coefficient build.
            def _cut(m, n):
                return lambda_cut is not None and m + n > lambda_cut
            if off_diagonal:
                # off the diagonal the two halves expand around different centers: full degree for every entry
                cd_n = cs_n.numerator_degrees
                numerator_degrees = [
                    [0 if _cut(m, n) else cd[m] + cd_n[n] for n in range(n_max + 1)]
                    for m in range(m_max + 1)
                ]
            else:
                numerator_degrees = [
                    [0 if _cut(m, n) else cd[m] + cd[n] for n in range(n_max + 1)]
                    for m in range(m_max + 1)
                ]
            return DeltaDerivTable(
                exp_base=exp_base_out,
                denominator=denominator,
                denominator_roots=roots,
                numerators=None,
                numerator_degrees=numerator_degrees,
                zero_entries=frozenset((m, n) for m in range(m_max + 1) for n in range(n_max + 1)
                    if _cut(m, n)),
                spin=ell,
                root_provenance=root_provenance,
            )

        swap = scalar_G_swap_coefficient(ell)
        sign = mp.mpf(swap)

        if cs_n is None or cs_n is cs:
            # Diagonal z = zbar: one expansion, m <-> n mirror-symmetric. P_m
            # composed onto the h-side (Delta+ell)/2 and hbar-side (Delta-ell)/2.
            p_h = [cs.numerators[m].compose_linear(half, ell_half) for m in range(n_top)]
            p_hbar = [cs.numerators[m].compose_linear(half, -ell_half) for m in range(n_top)]
            numerators = _build_G_numerators_arb(
                p_h, p_hbar, m_max, n_max, swap, lambda_cut
            )
            if numerators is None:
                zero = Polynomial.zero(field=mp.mpf)
                numerators = [
                    [
                        zero if (lambda_cut is not None and m + n > lambda_cut)
                        else p_h[m] * p_hbar[n] if swap == 0
                        else p_h[m] * p_hbar[n] + (p_hbar[m] * p_h[n]) * sign
                        for n in range(n_max + 1)
                    ]
                    for m in range(m_max + 1)
                ]
        else:
            # Off-diagonal z != zbar: the m-axis derivatives expand around the
            # z-center (cs), the n-axis around the zbar-center (cs_n). Each chiral
            # half carries its own geometric base E = 16 q / N, so the ordered
            # block B(h,hbar)[m][n] = E_z^h P_m^z(h) * E_zbar^hbar P_n^zbar(hbar)
            # picks up E_z^h E_zbar^hbar = GM^Delta * R^(ell/2) with the geometric
            # mean GM = sqrt(E_z E_zbar) (the single Delta-base) and the scalar
            # R^(ell/2) = (E_z / E_zbar)^(ell/2); the swapped ordering carries
            # R^(-ell/2). So
            #   G[m][n] = GM^Delta/(D D) * [ R^(ell/2) a_m[m] b_n[n]
            #                                + sign R^(-ell/2) b_m[m] a_n[n] ],
            # a = h-side, b = hbar-side. No m <-> n mirror (the axes differ).
            a_m = [cs.numerators[m].compose_linear(half, ell_half) for m in range(m_max + 1)]
            b_m = [cs.numerators[m].compose_linear(half, -ell_half) for m in range(m_max + 1)]
            a_n = [cs_n.numerators[n].compose_linear(half, ell_half) for n in range(n_max + 1)]
            b_n = [cs_n.numerators[n].compose_linear(half, -ell_half) for n in range(n_max + 1)]
            r_plus = (cs.exp_base / cs_n.exp_base) ** ell_half
            r_minus = 1 / r_plus
            r_minus_signed = sign * r_minus
            zero = Polynomial.zero(field=(a_m[0].field if a_m else mp.mpf))
            numerators = [
                [
                    zero if (lambda_cut is not None and m + n > lambda_cut)
                    else (a_m[m] * b_n[n]) * r_plus if swap == 0
                    else (a_m[m] * b_n[n]) * r_plus + (b_m[m] * a_n[n]) * r_minus_signed
                    for n in range(n_max + 1)
                ]
                for m in range(m_max + 1)
            ]

        return DeltaDerivTable(
            exp_base=exp_base_out,
            denominator=denominator,
            denominator_roots=roots,
            numerators=numerators,
            spin=ell,
            root_provenance=root_provenance,
        )


def _G_folded_poly_table(
    cs: ChiralSymbolic, ell: int, m_max: int, n_max: int
) -> DeltaDerivTable:
    """Parity-folded (w, s)-style symbolic G table.

    The fold is a Delta-independent linear recombination of the non-folded
    numerator polynomials, so the universal prefactor and denominator carry
    through unchanged.

    Skeleton chiral input (``cs.numerators is None``) folds the *degree* grid
    instead: each folded entry's degree is the max degree of the non-folded
    entries that the fold plan recombines into it (the recombination is over
    distinct (m+n) monomial blocks, so the top-degree contribution does not
    cancel). Verified against the full-build degrees in the test-suite.
    """
    ell = _check_ell(ell)
    with _reuse_symbolic_work_context(cs):
        odd_factor = fold_parity_marker(ell=ell)
        M_total = folded_source_order(m_max, n_max, odd_factor)

        half = mp.mpf(1) / 2
        ell_half = mp.mpf(ell) / 2
        d_h = cs.denominator.compose_linear(half, ell_half)
        d_hbar = cs.denominator.compose_linear(half, -ell_half)
        denominator = d_h * d_hbar
        roots = [2 * d - ell for d in cs.denominator_roots] + [
            2 * d + ell for d in cs.denominator_roots
        ]
        root_provenance = (
            (_tag_root_provenance(cs.root_provenance, "left") or [])
            + (_tag_root_provenance(cs.root_provenance, "right") or [])
        )

        if cs.numerators is None:
            plan, target_size = _fold_plan(
                bool(odd_factor), M_total + 1, M_total + 1, m_max, n_max
            )
            width = n_max + 1
            flat = [0] * target_size
            cd = cs.numerator_degrees
            active_targets = set()
            for m, n, contribs in plan:
                # the scalar G is z <-> zbar even (sign +1, or 0 at spin 0): the leading terms add
                deg = cd[m] + cd[n]
                for target_idx, _weight in contribs:
                    if _weight:
                        active_targets.add(target_idx)
                    if deg > flat[target_idx]:
                        flat[target_idx] = deg
            folded_degrees = [
                [flat[alpha * width + beta] for beta in range(width)]
                for alpha in range(m_max + 1)
            ]
            return DeltaDerivTable(
                exp_base=cs.exp_base,
                denominator=denominator,
                denominator_roots=roots,
                numerators=None,
                numerator_degrees=folded_degrees,
                zero_entries=frozenset((alpha, beta) for alpha in range(m_max + 1) for beta in range(width)
                    if alpha * width + beta not in active_targets),
                spin=ell,
                root_provenance=root_provenance,
            )

        p_h = [
            cs.numerators[m].compose_linear(half, ell_half)
            for m in range(M_total + 1)
        ]
        p_hbar = [
            cs.numerators[n].compose_linear(half, -ell_half)
            for n in range(M_total + 1)
        ]
        folded = _build_G_folded_numerators_flint(
            p_h, p_hbar, m_max, n_max, ell=ell
        )

        return DeltaDerivTable(
            exp_base=cs.exp_base,
            denominator=denominator,
            denominator_roots=roots,
            numerators=folded,
            spin=ell,
            root_provenance=root_provenance,
        )


def _build_B_poly_table(
    cs_h: ChiralSymbolic,
    cs_hbar: ChiralSymbolic,
    ell: int,
    m_max: int,
    n_max: int,
    lambda_cut: int | None = None,
    cs_h_n: ChiralSymbolic | None = None,
    cs_hbar_n: ChiralSymbolic | None = None,
) -> DeltaDerivTable:
    """Non-folded symbolic ordered B table from two chiral expansions.

    B(h, hbar; z, zbar) = F(h, z; {h_i}) * F(hbar, zbar; {hbar_i}) with
    h = (Delta+ell)/2, hbar = (Delta-ell)/2. The two chiral halves carry
    independent external weights -- `cs_h` is built from the holomorphic
    externals {h_i}, `cs_hbar` from the antiholomorphic {hbar_i} -- and there
    is *no* parity symmetrization: for spinning externals B(h, hbar) and
    B(hbar, h) are distinct physical blocks.

    The Kac weights d_{r,s} depend only on c, so cs_h and cs_hbar share the
    same denominator roots; the Delta-pole structure is identical to the
    scalar G table and the universal prefactor carries through unchanged.
    """
    ell = _check_ell_signed(ell)
    with _reuse_symbolic_work_context(cs_h, cs_hbar, cs_h_n, cs_hbar_n):
        half = mp.mpf(1) / 2
        ell_half = mp.mpf(ell) / 2
        d_h = cs_h.denominator.compose_linear(half, ell_half)
        d_hbar = cs_hbar.denominator.compose_linear(half, -ell_half)
        denominator = d_h * d_hbar
        roots = [2 * d - ell for d in cs_h.denominator_roots] + [
            2 * d + ell for d in cs_hbar.denominator_roots
        ]
        root_provenance = (
            (_tag_root_provenance(cs_h.root_provenance, "left") or [])
            + (_tag_root_provenance(cs_hbar.root_provenance, "right") or [])
        )
        off_diagonal = (
            cs_h_n is not None
            and cs_hbar_n is not None
            and cs_h_n is not cs_h
        )
        exp_base_out = (
            mp.sqrt(cs_h.exp_base * cs_h_n.exp_base) if off_diagonal else cs_h.exp_base
        )

        if cs_h.numerators is None:
            # Skeleton (degree-only) build: the ordered numerator B[m][n] =
            # p_h[m] * p_hbar[n] is a single product (no swap term, so no leading-
            # coefficient cancellation), hence its degree is exactly the sum of
            # the per-side chiral degrees -- read off the two degree grids without
            # building any coefficients.
            cdh = cs_h.numerator_degrees
            cdhb = cs_hbar.numerator_degrees
            # Corner-skip (see _build_G_poly_table): non-kept entries report
            # degree 0 so the skeleton's sample-point sizing matches the
            # coefficient build's zero-placeholder corner.
            numerator_degrees = [
                [
                    0 if (lambda_cut is not None and m + n > lambda_cut)
                    else cdh[m] + cdhb[n]
                    for n in range(n_max + 1)
                ]
                for m in range(m_max + 1)
            ]
            return DeltaDerivTable(
                exp_base=exp_base_out,
                denominator=denominator,
                denominator_roots=roots,
                numerators=None,
                numerator_degrees=numerator_degrees,
                zero_entries=frozenset((m, n) for m in range(m_max + 1) for n in range(n_max + 1)
                    if lambda_cut is not None and m + n > lambda_cut),
                spin=ell,
                root_provenance=root_provenance,
            )

        if not off_diagonal:
            p_h = [
                cs_h.numerators[m].compose_linear(half, ell_half)
                for m in range(m_max + 1)
            ]
            p_hbar = [
                cs_hbar.numerators[n].compose_linear(half, -ell_half)
                for n in range(n_max + 1)
            ]
            # Corner-skip: entries with m + n > lambda_cut are dropped by the
            # interpolation step (the m + n <= Lambda kept triangle), so leave a
            # zero placeholder rather than forming the product.
            zero = Polynomial.zero(field=mp.mpf)
            numerators = [
                [
                    zero if (lambda_cut is not None and m + n > lambda_cut)
                    else p_h[m] * p_hbar[n]
                    for n in range(n_max + 1)
                ]
                for m in range(m_max + 1)
            ]
        else:
            # Off-diagonal z != zbar: m-axis derivatives come from the z-center
            # expansions (cs_h / cs_hbar), n-axis derivatives from the
            # zbar-center expansions (cs_h_n / cs_hbar_n). The single Delta-base
            # is the geometric mean GM = sqrt(E_z E_zbar), with an R^(ell/2)
            # scalar on the ordered block numerators.
            p_h_m = [
                cs_h.numerators[m].compose_linear(half, ell_half)
                for m in range(m_max + 1)
            ]
            p_hbar_n = [
                cs_hbar_n.numerators[n].compose_linear(half, -ell_half)
                for n in range(n_max + 1)
            ]
            r_plus = (cs_h.exp_base / cs_h_n.exp_base) ** ell_half
            zero = Polynomial.zero(field=(p_h_m[0].field if p_h_m else mp.mpf))
            numerators = [
                [
                    zero if (lambda_cut is not None and m + n > lambda_cut)
                    else (p_h_m[m] * p_hbar_n[n]) * r_plus
                    for n in range(n_max + 1)
                ]
                for m in range(m_max + 1)
            ]

        return DeltaDerivTable(
            exp_base=exp_base_out,
            denominator=denominator,
            denominator_roots=roots,
            numerators=numerators,
            spin=ell,
            root_provenance=root_provenance,
        )


def _build_projected_B_numerators_flint(
    p_h,
    p_hbar,
    m_max: int,
    n_max: int,
    *,
    swapped_scale,
    lambda_cut: int | None = None,
    p_swap_m=None,
    p_swap_n=None,
    ordered_scale=1,
    swapped_term_scale=1,
    max_degrees=None,
):
    """Assemble parity-projected B numerators directly in flint space.

    Computes

        N[m][n] = p_h[m] * p_hbar[n] + swapped_scale * p_hbar[m] * p_h[n]

    with a single materialization per output polynomial.
    """
    from flint import acb, arb, ctx as flint_ctx
    from .polynomial import _flint_prec_bits, _truncate_flint_poly

    p_sm = p_hbar if p_swap_m is None else p_swap_m
    p_sn = p_h if p_swap_n is None else p_swap_n
    ordered_scale_eff = ordered_scale
    swapped_scale_eff = swapped_scale * swapped_term_scale
    complex_out = (
        any(p_h[i].field is mp.mpc for i in range(m_max + 1))
        or any(p_hbar[n].field is mp.mpc for n in range(n_max + 1))
        or any(p_sm[i].field is mp.mpc for i in range(m_max + 1))
        or any(p_sn[n].field is mp.mpc for n in range(n_max + 1))
        or (isinstance(swapped_scale_eff, mp.mpc) and swapped_scale_eff.imag != 0)
        or (isinstance(ordered_scale_eff, mp.mpc) and ordered_scale_eff.imag != 0)
    )
    prec = _flint_prec_bits()
    flint_ctx.prec = prec

    if complex_out:
        scale_o = acb(ordered_scale_eff)
        scale_s = acb(swapped_scale_eff)
        ph = [p._get_acb() for p in p_h]
        phb = [p._get_acb() for p in p_hbar]
        psm = [p._get_acb() for p in p_sm]
        psn = [p._get_acb() for p in p_sn]
        zero = Polynomial.zero(field=mp.mpc)
        out = []
        for m in range(m_max + 1):
            row = []
            for n in range(n_max + 1):
                if lambda_cut is not None and m + n > lambda_cut:
                    row.append(zero)
                    continue
                res = scale_o * (ph[m] * phb[n]) + scale_s * (psm[m] * psn[n])
                if max_degrees is not None:
                    res = _truncate_flint_poly(res, max_degrees[m][n] + 1)
                # Lazy mpc: defer the acb->mpc conversion (the downstream
                # Vandermonde sampler reads _get_acb() + degree only).
                row.append(Polynomial._from_acb_trusted(res, prec))
            out.append(row)
        return out

    scale_o = arb(
        mp.mpf(ordered_scale_eff.real)
        if isinstance(ordered_scale_eff, mp.mpc)
        else ordered_scale_eff
    )
    scale_s = arb(
        mp.mpf(swapped_scale_eff.real)
        if isinstance(swapped_scale_eff, mp.mpc)
        else swapped_scale_eff
    )
    ph = [p._get_arb() for p in p_h]
    phb = [p._get_arb() for p in p_hbar]
    psm = [p._get_arb() for p in p_sm]
    psn = [p._get_arb() for p in p_sn]
    zero = Polynomial.zero(field=mp.mpf)
    out = []
    for m in range(m_max + 1):
        row = []
        for n in range(n_max + 1):
            if lambda_cut is not None and m + n > lambda_cut:
                row.append(zero)
                continue
            res = scale_o * (ph[m] * phb[n]) + scale_s * (psm[m] * psn[n])
            if max_degrees is not None:
                res = _truncate_flint_poly(res, max_degrees[m][n] + 1)
            # Lazy mpf: defer the arb->mpf conversion (see above).
            row.append(Polynomial._from_arb_trusted(res, prec))
        out.append(row)
    return out


def _build_projected_B_folded_numerators_flint(
    p_h,
    p_hbar,
    m_max: int,
    n_max: int,
    *,
    parity_marker: int,
    swapped_scale,
):
    """Assemble folded parity-projected B numerators directly in flint space."""
    from flint import acb, acb_poly, arb, arb_poly, ctx as flint_ctx
    from .polynomial import _flint_prec_bits

    rows = len(p_h)
    cols = len(p_hbar)
    odd_factor = (parity_marker % 2) != 0
    plan, target_size = _fold_plan(odd_factor, rows, cols, m_max, n_max)
    width = n_max + 1

    complex_out = (
        any(p.field is mp.mpc for p in p_h)
        or any(p.field is mp.mpc for p in p_hbar)
        or (isinstance(swapped_scale, mp.mpc) and swapped_scale.imag != 0)
    )
    prec = _flint_prec_bits()
    flint_ctx.prec = prec

    if complex_out:
        scale = acb(swapped_scale)
        ph = [p._get_acb() for p in p_h]
        phb = [p._get_acb() for p in p_hbar]
        accum = [acb_poly([acb(arb(0))]) for _ in range(target_size)]
        for m, n, contribs in plan:
            res = ph[m] * phb[n] + scale * (phb[m] * ph[n])
            for target_idx, weight in contribs:
                accum[target_idx] += res * acb(weight)
        # Lazy mpc: defer the acb->mpc conversion (the downstream Vandermonde
        # sampler reads _get_acb() + degree only). See _build_G_folded_numerators_flint.
        return [
            [Polynomial._from_acb_trusted(accum[alpha * width + beta], prec)
             for beta in range(width)]
            for alpha in range(m_max + 1)
        ]

    scale = arb(
        mp.mpf(swapped_scale.real) if isinstance(swapped_scale, mp.mpc) else swapped_scale
    )
    ph = [p._get_arb() for p in p_h]
    phb = [p._get_arb() for p in p_hbar]
    accum = [arb_poly([arb(0)]) for _ in range(target_size)]
    for m, n, contribs in plan:
        res = ph[m] * phb[n] + scale * (phb[m] * ph[n])
        for target_idx, weight in contribs:
            accum[target_idx] += res * arb(weight)
    # Lazy mpf: defer the arb->mpf conversion (see _build_G_folded_numerators_flint).
    return [
        [Polynomial._from_arb_trusted(accum[alpha * width + beta], prec)
         for beta in range(width)]
        for alpha in range(m_max + 1)
    ]


def _build_parity_projected_B_poly_table(
    cs_h: ChiralSymbolic,
    cs_hbar: ChiralSymbolic,
    ell,
    m_max: int,
    n_max: int,
    *,
    parity: str,
    parity_external=None,
    exchanged_parity: str = "1",
    lambda_cut: int | None = None,
    cs_h_n: ChiralSymbolic | None = None,
    cs_hbar_n: ChiralSymbolic | None = None,
) -> DeltaDerivTable:
    """Non-folded symbolic parity-projected B table from two chiral expansions.

    The universal prefactor and Delta-denominator are identical to the ordered
    B table. Only the numerators are recombined:

        B_parity = B_ordered +/- phase * B_reversed

    with the parity-reversed partner using swapped chiral data.
    """
    if parity not in {"+", "-"}:
        raise ValueError("parity must be '+' or '-'")

    ell = _check_ell_signed(ell)
    with _reuse_symbolic_work_context(cs_h, cs_hbar, cs_h_n, cs_hbar_n):
        half = mp.mpf(1) / 2
        ell_half = mp.mpf(ell) / 2
        off_diagonal = (
            cs_h_n is not None
            and cs_hbar_n is not None
            and cs_h_n is not cs_h
        )
        exp_base_out = (
            mp.sqrt(cs_h.exp_base * cs_h_n.exp_base) if off_diagonal else cs_h.exp_base
        )
        phase = parity_reversal_phase(
            parity_external, exchanged_parity=exchanged_parity
        )
        sign = mp.mpf(1) if parity == "+" else mp.mpf(-1)
        swapped_scale = sign * phase

        if cs_h.numerators is None:
            # Degree-only (skeleton) build: skip the numerator coefficients.
            # B_proj[m][n] = p_h[m] p_hbar[n] + swapped_scale * p_hbar[m] p_h[n]
            # (diagonal), or (a_m[m] b_n[n]) r_plus + swapped_scale (b_m[m] a_n[n])
            # r_minus (off-diagonal). The chiral halves come from different
            # externals, but their leading-in-h coefficients are PROPORTIONAL
            # with an order-independent ratio (the large-h leading term is the
            # external-independent geometry times a single m-independent overall
            # prefactor normalization per half / per center). So the leading
            # combination of entry (m, n) is L_hbar[m] L_hbar[n] times a single
            # m,n-independent scalar:
            #   diagonal:     (1 + swapped_scale)                 -> cancels iff
            #                 swapped_scale == -1;
            #   off-diagonal: (r_plus k_z + swapped_scale r_minus k_zbar), where
            #                 k_z, k_zbar are the per-center external-ratio
            #                 normalizations -- generically nonzero (the geometric
            #                 weights detune the swap-sign cancellation).
            # Hence:
            #   * diagonal, swapped_scale == -1: leading terms cancel (degree
            #     drops by 1) and the m == n entry vanishes identically (the
            #     z <-> zbar-odd pattern), keyed on the build-time swap sign (an
            #     exact root of unity from the parity inputs).
            #   * diagonal, swapped_scale != -1, and all off-diagonal: no
            #     cancellation, degree cd_h[m] + cd_hbar[n].
            #
            # The full symbolic build enforces the same structural upper bound:
            # roundoff in a canceled leading coefficient must not add a node
            # to the interpolant or change its large-Delta power.
            cdh = cs_h.numerator_degrees
            cdhb = cs_hbar.numerator_degrees
            # Only the diagonal projected build can cancel for the parity swap
            # signs (== -1); off-diagonal's geometric weights detune it, so it
            # keeps the full cd_h[m] + cd_hbar[n] degree (no m == n vanishing).
            cancels = (not off_diagonal) and swapped_scale == -1

            def _cut(m, n):
                return lambda_cut is not None and m + n > lambda_cut

            if cancels:
                numerator_degrees = [
                    [
                        0 if (m == n or _cut(m, n)) else cdh[m] + cdhb[n] - 1
                        for n in range(n_max + 1)
                    ]
                    for m in range(m_max + 1)
                ]
            else:
                numerator_degrees = [
                    [0 if _cut(m, n) else cdh[m] + cdhb[n] for n in range(n_max + 1)]
                    for m in range(m_max + 1)
                ]
            d_h = cs_h.denominator.compose_linear(half, ell_half)
            d_hbar = cs_hbar.denominator.compose_linear(half, -ell_half)
            return DeltaDerivTable(
                exp_base=exp_base_out,
                denominator=d_h * d_hbar,
                denominator_roots=(
                    [2 * d - ell for d in cs_h.denominator_roots]
                    + [2 * d + ell for d in cs_hbar.denominator_roots]
                ),
                numerators=None,
                numerator_degrees=numerator_degrees,
                zero_entries=frozenset((m, n) for m in range(m_max + 1) for n in range(n_max + 1)
                    if _cut(m, n) or (cancels and m == n)),
                spin=ell,
                root_provenance=(
                    (_tag_root_provenance(cs_h.root_provenance, "left") or [])
                    + (_tag_root_provenance(cs_hbar.root_provenance, "right") or [])
                ),
            )

        if not off_diagonal:
            p_h = [
                cs_h.numerators[m].compose_linear(half, ell_half)
                for m in range(m_max + 1)
            ]
            p_hbar = [
                cs_hbar.numerators[n].compose_linear(half, -ell_half)
                for n in range(n_max + 1)
            ]
            numerators = _build_projected_B_numerators_flint(
                p_h, p_hbar, m_max, n_max, swapped_scale=swapped_scale,
                lambda_cut=lambda_cut,
                max_degrees=(
                    [[cs_h.denominator.degree + cs_hbar.denominator.degree + m + n - 1
                      for n in range(n_max + 1)] for m in range(m_max + 1)]
                    if swapped_scale == -1 else None
                ),
            )
        else:
            a_m = [
                cs_h.numerators[m].compose_linear(half, ell_half)
                for m in range(m_max + 1)
            ]
            b_n = [
                cs_hbar_n.numerators[n].compose_linear(half, -ell_half)
                for n in range(n_max + 1)
            ]
            b_m = [
                cs_hbar.numerators[m].compose_linear(half, -ell_half)
                for m in range(m_max + 1)
            ]
            a_n = [
                cs_h_n.numerators[n].compose_linear(half, ell_half)
                for n in range(n_max + 1)
            ]
            r_plus = (cs_h.exp_base / cs_h_n.exp_base) ** ell_half
            r_minus = 1 / r_plus
            numerators = _build_projected_B_numerators_flint(
                a_m,
                b_n,
                m_max,
                n_max,
                swapped_scale=swapped_scale,
                lambda_cut=lambda_cut,
                p_swap_m=b_m,
                p_swap_n=a_n,
                ordered_scale=r_plus,
                swapped_term_scale=r_minus,
            )

        d_h = cs_h.denominator.compose_linear(half, ell_half)
        d_hbar = cs_hbar.denominator.compose_linear(half, -ell_half)
        denominator = d_h * d_hbar
        roots = [2 * d - ell for d in cs_h.denominator_roots] + [
            2 * d + ell for d in cs_hbar.denominator_roots
        ]
        return DeltaDerivTable(
            exp_base=exp_base_out,
            denominator=denominator,
            denominator_roots=roots,
            numerators=numerators,
            spin=ell,
            root_provenance=(
                (_tag_root_provenance(cs_h.root_provenance, "left") or [])
                + (_tag_root_provenance(cs_hbar.root_provenance, "right") or [])
            ),
        )


def _parity_projected_B_folded_table(
    cs_h: ChiralSymbolic,
    cs_hbar: ChiralSymbolic,
    ell,
    m_max: int,
    n_max: int,
    *,
    parity: str,
    parity_external=None,
    exchanged_parity: str = "1",
) -> DeltaDerivTable:
    """Folded parity-projected B table from two pre-built chiral expansions.

    The level-C counterpart of `_parity_projected_B_folded` -- takes already-
    computed ChiralSymbolic expansions (built at a Taylor order at least
    `m_max + 2 * n_max + (1 if parity == '-' else 0) + 1`) and consumes them
    without rebuilding. Used by the chiral-cache fast path in the CLI.
    """
    ell = _check_ell_signed(ell)
    with _reuse_symbolic_work_context(cs_h, cs_hbar):
        parity_marker = _parity_fold_marker(parity, parity_external, exchanged_parity)
        phase = parity_reversal_phase(
            parity_external, exchanged_parity=exchanged_parity
        )
        if abs(mp.im(phase)) > _validation_tolerance(guard=5):
            raise ValueError(
                "parity-folded coordinates require a real (+/-1) total parity "
                "phase; got a non-real phase, which has no definite z<->zbar "
                "parity and cannot be parity-folded -- use a non-folded coordinate"
            )
        odd_factor = parity_marker % 2
        M_total = folded_source_order(m_max, n_max, parity_marker)
        half = mp.mpf(1) / 2
        ell_half = mp.mpf(ell) / 2
        sign = mp.mpf(1) if parity == "+" else mp.mpf(-1)
        swapped_scale = sign * phase

        if cs_h.numerators is None:
            # Degree-only (skeleton) fold. Each folded entry's degree is the max
            # degree of the non-folded source entries the fold recombines into it
            # (over distinct (m+n) monomial blocks, so the top degree does not
            # cancel across the fold). The source entry (m, n) is the non-folded
            # projected numerator p_h[m] p_hbar[n] + swapped_scale p_hbar[m] p_h[n]:
            # its leading coefficients are proportional across orders, so it has
            # degree cd[m] + cd[n], dropping by one (and vanishing on m == n) only
            # when swapped_scale == -1 (the z <-> zbar-odd pattern).
            # (At swapped_scale == -1 the full build's distinct-polynomial swap
            # leaves arb roundoff at the canceled order, so its folded degree can
            # read higher; the structural degree here is the exact one and the
            # evaluated blocks still agree -- cf. the non-folded projected case.)
            plan, target_size = _fold_plan(
                bool(odd_factor), M_total + 1, M_total + 1, m_max, n_max
            )
            width = n_max + 1
            flat = [0] * target_size
            # The halves can have different Kac denominators (spinning
            # externals: pairwise-equal hbar externals remove poles the h
            # externals keep), so each term pairs one degree from each half.
            cdh = cs_h.numerator_degrees
            cdhb = cs_hbar.numerator_degrees
            cancels = abs(mp.mpc(swapped_scale) + 1) < mp.mpf("1e-40")
            active_targets = set()
            for m, n, contribs in plan:
                full = max(cdh[m] + cdhb[n], cdhb[m] + cdh[n])
                if cancels:
                    deg = 0 if m == n else full - 1
                else:
                    deg = full
                for target_idx, _weight in contribs:
                    if (not cancels or m != n) and _weight:
                        active_targets.add(target_idx)
                    if deg > flat[target_idx]:
                        flat[target_idx] = deg
            folded_degrees = [
                [flat[alpha * width + beta] for beta in range(width)]
                for alpha in range(m_max + 1)
            ]
            d_h = cs_h.denominator.compose_linear(half, ell_half)
            d_hbar = cs_hbar.denominator.compose_linear(half, -ell_half)
            return DeltaDerivTable(
                exp_base=cs_h.exp_base,
                denominator=d_h * d_hbar,
                denominator_roots=(
                    [2 * d - ell for d in cs_h.denominator_roots]
                    + [2 * d + ell for d in cs_hbar.denominator_roots]
                ),
                numerators=None,
                numerator_degrees=folded_degrees,
                zero_entries=frozenset((alpha, beta) for alpha in range(m_max + 1) for beta in range(width)
                    if alpha * width + beta not in active_targets),
                spin=ell,
                root_provenance=(
                    (_tag_root_provenance(cs_h.root_provenance, "left") or [])
                    + (_tag_root_provenance(cs_hbar.root_provenance, "right") or [])
                ),
            )

        p_h = [
            cs_h.numerators[m].compose_linear(half, ell_half)
            for m in range(M_total + 1)
        ]
        p_hbar = [
            cs_hbar.numerators[n].compose_linear(half, -ell_half)
            for n in range(M_total + 1)
        ]
        folded = _build_projected_B_folded_numerators_flint(
            p_h,
            p_hbar,
            m_max,
            n_max,
            parity_marker=parity_marker,
            swapped_scale=swapped_scale,
        )
        d_h = cs_h.denominator.compose_linear(half, ell_half)
        d_hbar = cs_hbar.denominator.compose_linear(half, -ell_half)
        denominator = d_h * d_hbar
        roots = [2 * d - ell for d in cs_h.denominator_roots] + [
            2 * d + ell for d in cs_hbar.denominator_roots
        ]
        root_provenance = (
            (_tag_root_provenance(cs_h.root_provenance, "left") or [])
            + (_tag_root_provenance(cs_hbar.root_provenance, "right") or [])
        )
        return DeltaDerivTable(
            exp_base=cs_h.exp_base,
            denominator=denominator,
            denominator_roots=roots,
            numerators=folded,
            spin=ell,
            root_provenance=root_provenance,
        )


def _chiral_for(z_of_u, u_symmetric, ell, h_externals, m_max, n_max, b, c, q_order,
                chiral_norm=None, z_shift=None, skeleton=False):
    n_terms = max(m_max, n_max) + 1
    return chiral_symbolic_useries(
        h_externals,
        n_terms,
        b=b,
        c=c,
        z_of_u=z_of_u,
        u_symmetric=u_symmetric,
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
        skeleton=skeleton,
    )


def _parity_fold_marker(parity: str, parity_external=None, exchanged_parity="1") -> int:
    """Return an even/odd marker compatible with derivs._fold_parity (the z <-> zbar parity sign x phase)."""
    return fold_parity_marker(parity=parity, phase=parity_reversal_phase(
        parity_external, exchanged_parity=exchanged_parity or "1"))


# --- chiral-only symbolic tables: F(h, z) at fixed h, one Taylor index ---
#
# A single chiral block carries one Taylor index `m` and one symbolic weight
# `h`, with Kac poles at `h = d_{r,s}`. We reuse `DeltaDerivTable` with shape
# (m_max + 1) x 1 -- the same trick the radial slices use -- so the
# serialization / interpolation / writer pipeline consumes a chiral table
# unchanged. The "Delta" variable is conceptually `h` here; we keep the field
# name to share the downstream code.


def _build_F_chiral_poly_table(cs: ChiralSymbolic, m_max: int) -> DeltaDerivTable:
    """Chiral-block derivative table from one chiral expansion.

    F(h, z(u)) Taylor-expanded around u = u_* is
        c_m(h) = exp_base ** h * numerators[m](h) / denominator(h)
    with `denominator_roots` at `h = d_{r,s}`. The polynomial in h is wrapped as
    a degenerate (m_max + 1) x 1 table so the downstream pipeline (which
    expects two-axis numerators) sees a familiar shape -- exactly the same
    arrangement used for the parity-folded radial slices.
    """
    with _reuse_symbolic_work_context(cs):
        if cs.numerators is None:
            return DeltaDerivTable(
                exp_base=cs.exp_base,
                denominator=cs.denominator,
                denominator_roots=list(cs.denominator_roots),
                numerators=None,
                numerator_degrees=[[cs.numerator_degrees[m]] for m in range(m_max + 1)],
                root_provenance=_tag_root_provenance(cs.root_provenance, "chiral"),
            )
        numerators = [[cs.numerators[m]] for m in range(m_max + 1)]
        return DeltaDerivTable(
            exp_base=cs.exp_base,
            denominator=cs.denominator,
            denominator_roots=list(cs.denominator_roots),
            numerators=numerators,
            root_provenance=_tag_root_provenance(cs.root_provenance, "chiral"),
        )


def _F_chiral_poly_table(
    z_of_u,
    u_symmetric,
    h_externals,
    m_max,
    *,
    b,
    c,
    q_order,
    chiral_norm=None,
    z_shift=None,
    center=None,
    coordinate: str = "z",
    skeleton=False,
) -> DeltaDerivTable:
    m_max, _ = _check_table_orders(m_max, 0)
    q_order = _integer_control('q_order', q_order)
    b, c = _resolve_bc(b, c)
    _check_externals(h_externals)
    cs = chiral_symbolic_useries(
        h_externals,
        m_max + 1,
        b=b,
        c=c,
        z_of_u=z_of_u,
        u_symmetric=u_symmetric,
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
        skeleton=skeleton,
    )
    table = _build_F_chiral_poly_table(cs, m_max)
    table.oracle_metadata = {
        "basis": "chiral_F",
        "coordinate": coordinate,
        "h_externals": list(h_externals),
        "q_order": q_order,
        "chiral_norm": chiral_norm,
        "z_shift": z_shift,
        "center": center,
    }
    return table


def F_z_poly_table(
    h_externals, m_max, *, b=None, c=None, q_order: int, chiral_norm=None,
    z_shift=None, zbar_shift=None, center=None, skeleton=False,
) -> DeltaDerivTable:
    """Symbolic-in-h chiral F(h, z) Taylor table around z = 1/2.

    ``center`` (default ``None`` = the symmetric point 1/2) recenters the single
    chiral expansion at ``z = center``; a chiral half has no zbar, so there is no
    off-diagonal notion. See :func:`_resolve_center_u`.
    """
    z_of_u, u_sym = _resolve_center_u("z", center)
    return _F_chiral_poly_table(
        z_of_u, u_sym, h_externals, m_max,
        b=b, c=c, q_order=q_order, chiral_norm=chiral_norm, z_shift=z_shift,
        center=center, coordinate="z", skeleton=skeleton,
    )


def F_y_poly_table(
    h_externals, m_max, *, b=None, c=None, q_order: int, chiral_norm=None,
    z_shift=None, zbar_shift=None, center=None, skeleton=False,
) -> DeltaDerivTable:
    """Symbolic-in-h chiral F(h, z(y)) Taylor table around y = 0.

    ``center`` recenters the chiral expansion at ``z = center`` (default 1/2).
    """
    z_of_u, u_sym = _resolve_center_u("y", center)
    return _F_chiral_poly_table(
        z_of_u, u_sym, h_externals, m_max,
        b=b, c=c, q_order=q_order, chiral_norm=chiral_norm, z_shift=z_shift,
        center=center, coordinate="y", skeleton=skeleton,
    )


def F_q_poly_table(
    h_externals, m_max, *, b=None, c=None, q_order: int, chiral_norm=None,
    z_shift=None, zbar_shift=None, center=None, skeleton=False,
) -> DeltaDerivTable:
    """Symbolic-in-h chiral F(h, z(q)) Taylor table around q = exp(-pi).

    ``center`` recenters the chiral expansion at ``z = center`` (default 1/2).
    """
    z_of_u, u_sym = _resolve_center_u("q", center)
    return _F_chiral_poly_table(
        z_of_u, u_sym, h_externals, m_max,
        b=b, c=c, q_order=q_order, chiral_norm=chiral_norm, z_shift=z_shift,
        center=center, coordinate="q", skeleton=skeleton,
    )


# --- non-folded symbolic tables: zzb, yyb, qqb ---


def _resolve_center_u(coord: str, center):
    """``(z_of_u, u_symmetric)`` for a non-folded or chiral coordinate, recentered
    at ``z = center``.

    ``center=None`` keeps the crossing-symmetric expansion point (bit-for-bit
    the legacy behavior). A real ``center != 1/2`` recenters the chiral
    expansion at ``u(center)`` for an off-symmetric point -- for a 2d coordinate
    that is the diagonal ``z = zbar = center`` (off-diagonal needs a separate
    ``center_bar``); for a chiral coordinate (z/y/q, a single chiral half F(h,z))
    there is no zbar, so ``center`` simply moves the single expansion point. See
    :func:`virasoro_blocks.derivs.evaluate_block`. The folded coordinates are
    excluded (their fold geometry is defined relative to the symmetric center).
    """
    spec = COORDINATES[coord]
    if spec.kind not in ("non_folded_2d", "chiral"):
        raise ValueError(
            f"off-symmetric center is only supported for non-folded 2d and "
            f"chiral coordinates; got coord={coord!r}"
        )
    if center is None:
        return spec.z_of_u, spec.u_symmetric()
    from .numeric_input import literal_number

    source = literal_number(center)
    return spec.z_of_u, spec.u_of_z(source)


def _resolve_lambda_cut(lambda_cut, m_max, n_max):
    """Resolve the order-Lambda derivative truncation for a non-folded table.

    Mirrors :func:`build_table_from_cache`: the ``_AUTO_LAMBDA_CUT`` sentinel
    (the default) resolves to the kept-triangle cut ``max(m_max, n_max)`` so a
    non-folded build keeps exactly the ``m + n <= Lambda`` entries -- the
    meaning of Lambda -- rather than the full ``(m_max, n_max)`` square. Pass
    ``lambda_cut=None`` to force the full square; an explicit int sets the cut.
    """
    if lambda_cut is _AUTO_LAMBDA_CUT:
        return max(m_max, n_max)
    return None if lambda_cut is None else _integer_control('lambda_cut', lambda_cut)


def _G_poly_table(
    z_of_u,
    u_symmetric,
    ell,
    h_externals,
    m_max,
    n_max,
    *,
    coordinate=None,
    center=None,
    center_bar=None,
    b,
    c,
    q_order,
    chiral_norm=None,
    z_shift=None,
    lambda_cut=_AUTO_LAMBDA_CUT,
    u_symmetric_bar=None,
    skeleton=False,
) -> DeltaDerivTable:
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    q_order = _integer_control('q_order', q_order)
    b, c = _resolve_bc(b, c)
    _check_externals(h_externals)
    ell = _check_ell(ell)
    cs = _chiral_for(
        z_of_u, u_symmetric, ell, h_externals, m_max, n_max, b, c, q_order,
        chiral_norm=chiral_norm, z_shift=z_shift, skeleton=skeleton,
    )
    # Off-diagonal z != zbar: a second expansion around the zbar-center for the
    # n-axis. None / equal center keeps the single diagonal expansion (the
    # zbar-center==z-center case reuses cs, so there is no double build and the
    # diagonal stays the mirror-symmetric fast path).
    cs_n = None
    if u_symmetric_bar is not None and u_symmetric_bar != u_symmetric:
        cs_n = _chiral_for(
            z_of_u, u_symmetric_bar, ell, h_externals, m_max, n_max, b, c, q_order,
            chiral_norm=chiral_norm, z_shift=z_shift, skeleton=skeleton,
        )
    table = _build_G_poly_table(
        cs, ell, m_max, n_max, _resolve_lambda_cut(lambda_cut, m_max, n_max),
        cs_n=cs_n,
    )
    table.oracle_metadata = {
        "basis": "scalar_G",
        "coordinate": coordinate,
        "h_externals": list(h_externals),
        "ell": ell,
        "q_order": q_order,
        "chiral_norm": chiral_norm,
        "z_shift": z_shift,
        "center": center,
        "center_bar": center_bar,
        "folded": False,
    }
    return table


def _resolve_center_bar_u(coord, center_bar):
    """``u_symmetric_bar`` for the n-axis (zbar) expansion, or ``None`` for the
    diagonal. ``center_bar=None`` keeps the diagonal (n-axis shares the m-axis
    center); a value gives the off-diagonal ``z != zbar`` zbar-center ``u(zbar)``.
    """
    if center_bar is None:
        return None
    return _resolve_center_u(coord, center_bar)[1]


def _check_scalar_G_shift(z_shift, zbar_shift):
    """Require the same shift on both chiral halves of scalar G."""
    from .numeric_input import literal_components

    if zbar_shift is not None and (z_shift is None or literal_components(zbar_shift) != literal_components(z_shift)):
        raise ValueError("scalar G takes one z_shift for both halves; zbar_shift must equal it (use ordered B)")


def G_zzb_poly_table(
    ell, h_externals, m_max, n_max=None, *, b=None, c=None, q_order: int,
    chiral_norm=None, z_shift=None, zbar_shift=None, center=None, center_bar=None,
    lambda_cut=_AUTO_LAMBDA_CUT, skeleton=False,
) -> DeltaDerivTable:
    """Symbolic-in-Delta G table in (z, zbar) around z = zbar = 1/2.

    ``center`` (default ``None`` = the symmetric point 1/2) recenters the
    expansion at a diagonal off-symmetric point ``z = zbar = center``; see
    :func:`_resolve_center_u`. ``center_bar`` (default ``None`` = the diagonal
    ``zbar = center``) sets an independent ``zbar`` center for an off-diagonal
    ``z != zbar`` table -- the m-axis expands around ``center``, the n-axis
    around ``center_bar``. ``lambda_cut`` (default = the order-Lambda triangle
    ``max(m_max, n_max)``) keeps only ``m + n <= Lambda``; pass ``None`` for the
    full square. See :func:`_resolve_lambda_cut`.
    """
    _check_scalar_G_shift(z_shift, zbar_shift)
    z_of_u, u_sym = _resolve_center_u("zzb", center)
    return _G_poly_table(
        z_of_u, u_sym, ell, h_externals, m_max, n_max,
        coordinate="zzb", center=center, center_bar=center_bar,
        b=b, c=c, q_order=q_order, chiral_norm=chiral_norm, z_shift=z_shift,
        lambda_cut=lambda_cut, u_symmetric_bar=_resolve_center_bar_u("zzb", center_bar),
        skeleton=skeleton,
    )


def G_yyb_poly_table(
    ell, h_externals, m_max, n_max=None, *, b=None, c=None, q_order: int,
    chiral_norm=None, z_shift=None, zbar_shift=None, center=None, center_bar=None,
    lambda_cut=_AUTO_LAMBDA_CUT, skeleton=False,
) -> DeltaDerivTable:
    """Symbolic-in-Delta G table in (y, ybar) around y = ybar = 0.

    ``center`` recenters at a diagonal off-symmetric point ``z = zbar = center``;
    ``center_bar`` sets an independent ``zbar`` center for an off-diagonal table
    (default ``None`` = diagonal). ``lambda_cut`` keeps the order-Lambda triangle
    (default) or the full square (``None``).
    """
    _check_scalar_G_shift(z_shift, zbar_shift)
    z_of_u, u_sym = _resolve_center_u("yyb", center)
    return _G_poly_table(
        z_of_u, u_sym, ell, h_externals, m_max, n_max,
        coordinate="yyb", center=center, center_bar=center_bar,
        b=b, c=c, q_order=q_order, chiral_norm=chiral_norm, z_shift=z_shift,
        lambda_cut=lambda_cut, u_symmetric_bar=_resolve_center_bar_u("yyb", center_bar),
        skeleton=skeleton,
    )


def G_qqb_poly_table(
    ell, h_externals, m_max, n_max=None, *, b=None, c=None, q_order: int,
    chiral_norm=None, z_shift=None, zbar_shift=None, center=None, center_bar=None,
    lambda_cut=_AUTO_LAMBDA_CUT, skeleton=False,
) -> DeltaDerivTable:
    """Symbolic-in-Delta G table in (q, qbar) around q = qbar = exp(-pi).

    ``center`` recenters at a diagonal off-symmetric point ``z = zbar = center``;
    ``center_bar`` sets an independent ``zbar`` center for an off-diagonal table
    (default ``None`` = diagonal). ``lambda_cut`` keeps the order-Lambda triangle
    (default) or the full square (``None``).
    """
    _check_scalar_G_shift(z_shift, zbar_shift)
    z_of_u, u_sym = _resolve_center_u("qqb", center)
    return _G_poly_table(
        z_of_u, u_sym, ell, h_externals, m_max, n_max,
        coordinate="qqb", center=center, center_bar=center_bar,
        b=b, c=c, q_order=q_order, chiral_norm=chiral_norm, z_shift=z_shift,
        lambda_cut=lambda_cut, u_symmetric_bar=_resolve_center_bar_u("qqb", center_bar),
        skeleton=skeleton,
    )


# --- non-folded symbolic ordered B tables (scalar or spinning externals) ---


def _B_poly_table(
    z_of_u,
    u_symmetric,
    ell,
    h_externals,
    hbar_externals,
    m_max,
    n_max,
    *,
    coordinate=None,
    center=None,
    center_bar=None,
    b,
    c,
    q_order,
    chiral_norm=None,
    z_shift=None,
    zbar_shift=None,
    lambda_cut=_AUTO_LAMBDA_CUT,
    u_symmetric_bar=None,
    skeleton=False,
) -> DeltaDerivTable:
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    q_order = _integer_control('q_order', q_order)
    b, c = _resolve_bc(b, c)
    _check_externals(h_externals)
    _check_externals(hbar_externals)
    ell = _check_ell_signed(ell)
    cs_h = _chiral_for(
        z_of_u, u_symmetric, ell, h_externals, m_max, n_max, b, c, q_order,
        chiral_norm=chiral_norm, z_shift=z_shift, skeleton=skeleton,
    )
    cs_hbar = _chiral_for(
        z_of_u, u_symmetric, ell, hbar_externals, m_max, n_max, b, c, q_order,
        chiral_norm=chiral_norm, z_shift=zbar_shift, skeleton=skeleton,
    )
    cs_h_n = cs_hbar_n = None
    if u_symmetric_bar is not None and u_symmetric_bar != u_symmetric:
        cs_h_n = _chiral_for(
            z_of_u, u_symmetric_bar, ell, h_externals, m_max, n_max, b, c, q_order,
            chiral_norm=chiral_norm, z_shift=z_shift, skeleton=skeleton,
        )
        cs_hbar_n = _chiral_for(
            z_of_u, u_symmetric_bar, ell, hbar_externals, m_max, n_max, b, c, q_order,
            chiral_norm=chiral_norm, z_shift=zbar_shift, skeleton=skeleton,
        )
    table = _build_B_poly_table(
        cs_h, cs_hbar, ell, m_max, n_max,
        _resolve_lambda_cut(lambda_cut, m_max, n_max),
        cs_h_n=cs_h_n,
        cs_hbar_n=cs_hbar_n,
    )
    table.oracle_metadata = {
        "basis": "ordered_B",
        "coordinate": coordinate,
        "h_externals": list(h_externals),
        "hbar_externals": list(hbar_externals),
        "ell": ell,
        "q_order": q_order,
        "chiral_norm": chiral_norm,
        "z_shift": z_shift,
        "zbar_shift": zbar_shift,
        "center": center,
        "center_bar": center_bar,
        "folded": False,
    }
    return table


def _parity_projected_B_poly_table(
    z_of_u,
    u_symmetric,
    ell,
    h_externals,
    hbar_externals,
    m_max,
    n_max,
    *,
    coordinate=None,
    center=None,
    center_bar=None,
    parity: str,
    parity_external=None,
    exchanged_parity: str = "1",
    b,
    c,
    q_order,
    chiral_norm=None,
    z_shift=None,
    zbar_shift=None,
    lambda_cut=_AUTO_LAMBDA_CUT,
    u_symmetric_bar=None,
    skeleton=False,
) -> DeltaDerivTable:
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    q_order = _integer_control('q_order', q_order)
    b, c = _resolve_bc(b, c)
    _check_externals(h_externals)
    _check_externals(hbar_externals)
    ell = _check_ell_signed(ell)
    cs_h = _chiral_for(
        z_of_u, u_symmetric, ell, h_externals, m_max, n_max, b, c, q_order,
        chiral_norm=chiral_norm, z_shift=z_shift, skeleton=skeleton,
    )
    cs_hbar = _chiral_for(
        z_of_u, u_symmetric, ell, hbar_externals, m_max, n_max, b, c, q_order,
        chiral_norm=chiral_norm, z_shift=zbar_shift, skeleton=skeleton,
    )
    cs_h_n = cs_hbar_n = None
    if u_symmetric_bar is not None and u_symmetric_bar != u_symmetric:
        cs_h_n = _chiral_for(
            z_of_u, u_symmetric_bar, ell, h_externals, m_max, n_max, b, c, q_order,
            chiral_norm=chiral_norm, z_shift=z_shift, skeleton=skeleton,
        )
        cs_hbar_n = _chiral_for(
            z_of_u, u_symmetric_bar, ell, hbar_externals, m_max, n_max, b, c, q_order,
            chiral_norm=chiral_norm, z_shift=zbar_shift, skeleton=skeleton,
        )
    table = _build_parity_projected_B_poly_table(
        cs_h,
        cs_hbar,
        ell,
        m_max,
        n_max,
        parity=parity,
        parity_external=parity_external,
        exchanged_parity=exchanged_parity,
        lambda_cut=_resolve_lambda_cut(lambda_cut, m_max, n_max),
        cs_h_n=cs_h_n,
        cs_hbar_n=cs_hbar_n,
    )
    table.oracle_metadata = {
        "basis": "parity_projected_B",
        "coordinate": coordinate,
        "h_externals": list(h_externals),
        "hbar_externals": list(hbar_externals),
        "ell": ell,
        "q_order": q_order,
        "chiral_norm": chiral_norm,
        "z_shift": z_shift,
        "zbar_shift": zbar_shift,
        "center": center,
        "center_bar": center_bar,
        "parity": parity,
        "parity_external": parity_external,
        "exchanged_parity": exchanged_parity,
        "folded": False,
    }
    return table


def B_zzb_poly_table(
    ell, h_externals, hbar_externals, m_max, n_max=None,
    *, b=None, c=None, q_order: int, chiral_norm=None,
    z_shift=None, zbar_shift=None, center=None, center_bar=None,
    lambda_cut=_AUTO_LAMBDA_CUT, skeleton=False,
) -> DeltaDerivTable:
    """Symbolic-in-Delta ordered B table in (z, zbar) around z = zbar = 1/2.

    The ordered block for scalar or spinning externals; the exchanged spin
    ell = h - hbar may be either sign. Equal external weight lists retain
    this single ordered product. `G_zzb_poly_table` instead combines the two
    spin orientations into scalar G.

    ``center`` recenters at a diagonal off-symmetric point ``z = zbar =
    center`` (default ``None`` = 1/2). ``center_bar`` (default ``None`` =
    diagonal) sets an independent zbar-center for off-diagonal ``z != zbar``
    tables. ``lambda_cut`` keeps the order-Lambda triangle (default) or the
    full square (``None``).
    """
    z_of_u, u_sym = _resolve_center_u("zzb", center)
    return _B_poly_table(
        z_of_u, u_sym, ell, h_externals, hbar_externals,
        m_max, n_max,
        coordinate="zzb", center=center, center_bar=center_bar,
        b=b, c=c, q_order=q_order, chiral_norm=chiral_norm,
        z_shift=z_shift, zbar_shift=zbar_shift, lambda_cut=lambda_cut,
        u_symmetric_bar=_resolve_center_bar_u("zzb", center_bar),
        skeleton=skeleton,
    )


def B_yyb_poly_table(
    ell, h_externals, hbar_externals, m_max, n_max=None,
    *, b=None, c=None, q_order: int, chiral_norm=None,
    z_shift=None, zbar_shift=None, center=None, center_bar=None,
    lambda_cut=_AUTO_LAMBDA_CUT, skeleton=False,
) -> DeltaDerivTable:
    """Symbolic-in-Delta ordered B table in (y, ybar) around y = ybar = 0.

    ``center`` recenters at a diagonal off-symmetric point (default 1/2).
    ``center_bar`` sets an independent zbar-center for off-diagonal tables
    (default ``None`` = diagonal). ``lambda_cut`` keeps the order-Lambda
    triangle (default) or full square.
    """
    z_of_u, u_sym = _resolve_center_u("yyb", center)
    return _B_poly_table(
        z_of_u, u_sym, ell, h_externals, hbar_externals,
        m_max, n_max,
        coordinate="yyb", center=center, center_bar=center_bar,
        b=b, c=c, q_order=q_order, chiral_norm=chiral_norm,
        z_shift=z_shift, zbar_shift=zbar_shift, lambda_cut=lambda_cut,
        u_symmetric_bar=_resolve_center_bar_u("yyb", center_bar),
        skeleton=skeleton,
    )


def B_qqb_poly_table(
    ell, h_externals, hbar_externals, m_max, n_max=None,
    *, b=None, c=None, q_order: int, chiral_norm=None,
    z_shift=None, zbar_shift=None, center=None, center_bar=None,
    lambda_cut=_AUTO_LAMBDA_CUT, skeleton=False,
) -> DeltaDerivTable:
    """Symbolic-in-Delta ordered B table in (q, qbar) around q = qbar = exp(-pi).

    ``center`` recenters at a diagonal off-symmetric point (default 1/2).
    ``center_bar`` sets an independent zbar-center for off-diagonal tables
    (default ``None`` = diagonal). ``lambda_cut`` keeps the order-Lambda
    triangle (default) or full square.
    """
    z_of_u, u_sym = _resolve_center_u("qqb", center)
    return _B_poly_table(
        z_of_u, u_sym, ell, h_externals, hbar_externals,
        m_max, n_max,
        coordinate="qqb", center=center, center_bar=center_bar,
        b=b, c=c, q_order=q_order, chiral_norm=chiral_norm,
        z_shift=z_shift, zbar_shift=zbar_shift, lambda_cut=lambda_cut,
        u_symmetric_bar=_resolve_center_bar_u("qqb", center_bar),
        skeleton=skeleton,
    )


def parity_projected_B_zzb_poly_table(
    ell,
    h_externals,
    hbar_externals,
    m_max,
    n_max=None,
    *,
    parity: str,
    parity_external=None,
    exchanged_parity: str = "1",
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
    zbar_shift=None,
    center=None,
    center_bar=None,
    lambda_cut=_AUTO_LAMBDA_CUT,
    skeleton: bool = False,
) -> DeltaDerivTable:
    """Symbolic-in-Delta parity-projected B table in (z, zbar).

    ``center`` recenters at a diagonal off-symmetric point (default 1/2).
    ``center_bar`` sets an independent zbar-center for off-diagonal tables
    (default ``None`` = diagonal). ``lambda_cut`` keeps the order-Lambda
    triangle (default) or full square.
    """
    z_of_u, u_sym = _resolve_center_u("zzb", center)
    return _parity_projected_B_poly_table(
        z_of_u,
        u_sym,
        ell,
        h_externals,
        hbar_externals,
        m_max,
        n_max,
        coordinate="zzb",
        center=center,
        center_bar=center_bar,
        parity=parity,
        parity_external=parity_external,
        exchanged_parity=exchanged_parity,
        b=b,
        c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
        zbar_shift=zbar_shift,
        lambda_cut=lambda_cut,
        u_symmetric_bar=_resolve_center_bar_u("zzb", center_bar),
        skeleton=skeleton,
    )


def parity_projected_B_yyb_poly_table(
    ell,
    h_externals,
    hbar_externals,
    m_max,
    n_max=None,
    *,
    parity: str,
    parity_external=None,
    exchanged_parity: str = "1",
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
    zbar_shift=None,
    center=None,
    center_bar=None,
    lambda_cut=_AUTO_LAMBDA_CUT,
    skeleton: bool = False,
) -> DeltaDerivTable:
    """Symbolic-in-Delta parity-projected B table in (y, ybar).

    ``center`` recenters at a diagonal off-symmetric point (default 1/2).
    ``center_bar`` sets an independent zbar-center for off-diagonal tables
    (default ``None`` = diagonal). ``lambda_cut`` keeps the order-Lambda
    triangle (default) or full square.
    """
    z_of_u, u_sym = _resolve_center_u("yyb", center)
    return _parity_projected_B_poly_table(
        z_of_u,
        u_sym,
        ell,
        h_externals,
        hbar_externals,
        m_max,
        n_max,
        coordinate="yyb",
        center=center,
        center_bar=center_bar,
        parity=parity,
        parity_external=parity_external,
        exchanged_parity=exchanged_parity,
        b=b,
        c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
        zbar_shift=zbar_shift,
        lambda_cut=lambda_cut,
        u_symmetric_bar=_resolve_center_bar_u("yyb", center_bar),
        skeleton=skeleton,
    )


def parity_projected_B_qqb_poly_table(
    ell,
    h_externals,
    hbar_externals,
    m_max,
    n_max=None,
    *,
    parity: str,
    parity_external=None,
    exchanged_parity: str = "1",
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
    zbar_shift=None,
    center=None,
    center_bar=None,
    lambda_cut=_AUTO_LAMBDA_CUT,
    skeleton: bool = False,
) -> DeltaDerivTable:
    """Symbolic-in-Delta parity-projected B table in (q, qbar).

    ``center`` recenters at a diagonal off-symmetric point (default 1/2).
    ``center_bar`` sets an independent zbar-center for off-diagonal tables
    (default ``None`` = diagonal). ``lambda_cut`` keeps the order-Lambda
    triangle (default) or full square.
    """
    z_of_u, u_sym = _resolve_center_u("qqb", center)
    return _parity_projected_B_poly_table(
        z_of_u,
        u_sym,
        ell,
        h_externals,
        hbar_externals,
        m_max,
        n_max,
        coordinate="qqb",
        center=center,
        center_bar=center_bar,
        parity=parity,
        parity_external=parity_external,
        exchanged_parity=exchanged_parity,
        b=b,
        c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
        zbar_shift=zbar_shift,
        lambda_cut=lambda_cut,
        u_symmetric_bar=_resolve_center_bar_u("qqb", center_bar),
        skeleton=skeleton,
    )


# --- parity-folded symbolic tables: ws, xt, qqb_ws ---


def _G_folded(
    z_of_u,
    u_symmetric,
    ell,
    h_externals,
    m_max,
    n_max,
    *,
    coordinate=None,
    b,
    c,
    q_order,
    chiral_norm=None,
    z_shift=None,
    skeleton=False,
) -> DeltaDerivTable:
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    q_order = _integer_control('q_order', q_order)
    b, c = _resolve_bc(b, c)
    _check_externals(h_externals)
    ell = _check_ell(ell)
    odd_factor = fold_parity_marker(ell=ell)
    M_total = folded_source_order(m_max, n_max, odd_factor)
    cs = chiral_symbolic_useries(
        h_externals,
        M_total + 1,
        b=b,
        c=c,
        z_of_u=z_of_u,
        u_symmetric=u_symmetric,
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
        skeleton=skeleton,
    )
    table = _G_folded_poly_table(cs, ell, m_max, n_max)
    table.oracle_metadata = {
        "basis": "scalar_G",
        "coordinate": coordinate,
        "h_externals": list(h_externals),
        "ell": ell,
        "q_order": q_order,
        "chiral_norm": chiral_norm,
        "z_shift": z_shift,
        "center": None,
        "center_bar": None,
        "folded": True,
    }
    return table


def G_ws_poly_table(
    ell, h_externals, m_max, n_max=None, *, b=None, c=None, q_order: int,
    chiral_norm=None, z_shift=None, zbar_shift=None, skeleton=False,
) -> DeltaDerivTable:
    """Symbolic-in-Delta parity-folded (w, s) G table over y."""
    _check_scalar_G_shift(z_shift, zbar_shift)
    return _G_folded(
        z_of_y, y_symmetric(), ell, h_externals, m_max, n_max,
        coordinate="ws",
        b=b, c=c, q_order=q_order, chiral_norm=chiral_norm, z_shift=z_shift,
        skeleton=skeleton,
    )


def G_xt_poly_table(
    ell, h_externals, m_max, n_max=None, *, b=None, c=None, q_order: int,
    chiral_norm=None, z_shift=None, zbar_shift=None, skeleton=False,
) -> DeltaDerivTable:
    """Symbolic-in-Delta parity-folded (x, t) G table over z."""
    _check_scalar_G_shift(z_shift, zbar_shift)
    return _G_folded(
        lambda u: u, z_symmetric(), ell, h_externals, m_max, n_max,
        coordinate="xt",
        b=b, c=c, q_order=q_order, chiral_norm=chiral_norm, z_shift=z_shift,
        skeleton=skeleton,
    )


def G_qqb_ws_poly_table(
    ell, h_externals, m_max, n_max=None, *, b=None, c=None, q_order: int,
    chiral_norm=None, z_shift=None, zbar_shift=None, skeleton=False,
) -> DeltaDerivTable:
    """Symbolic-in-Delta parity-folded (w_q, s_q) G table over the nome q."""
    _check_scalar_G_shift(z_shift, zbar_shift)
    return _G_folded(
        z_of_q, q_symmetric(), ell, h_externals, m_max, n_max,
        coordinate="qqb_ws",
        b=b, c=c, q_order=q_order, chiral_norm=chiral_norm, z_shift=z_shift,
        skeleton=skeleton,
    )


def _parity_projected_B_folded(
    z_of_u,
    u_symmetric,
    ell,
    h_externals,
    hbar_externals,
    m_max,
    n_max,
    *,
    coordinate=None,
    parity: str,
    parity_external=None,
    exchanged_parity: str = "1",
    b,
    c,
    q_order,
    chiral_norm=None,
    z_shift=None,
    zbar_shift=None,
    skeleton=False,
) -> DeltaDerivTable:
    if n_max is None:
        n_max = m_max
    m_max, n_max = _check_table_orders(m_max, n_max)
    q_order = _integer_control('q_order', q_order)
    parity_marker = _parity_fold_marker(parity, parity_external, exchanged_parity)
    odd_factor = parity_marker % 2
    M_total = folded_source_order(m_max, n_max, parity_marker)
    b, c = _resolve_bc(b, c)
    _check_externals(h_externals)
    _check_externals(hbar_externals)
    ell = _check_ell_signed(ell)
    cs_h = _chiral_for(
        z_of_u,
        u_symmetric,
        ell,
        h_externals,
        M_total,
        M_total,
        b=b,
        c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
        skeleton=skeleton,
    )
    cs_hbar = _chiral_for(
        z_of_u,
        u_symmetric,
        ell,
        hbar_externals,
        M_total,
        M_total,
        b=b,
        c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=zbar_shift,
        skeleton=skeleton,
    )
    table = _parity_projected_B_folded_table(
        cs_h,
        cs_hbar,
        ell,
        m_max,
        n_max,
        parity=parity,
        parity_external=parity_external,
        exchanged_parity=exchanged_parity,
    )
    table.oracle_metadata = {
        "basis": "parity_projected_B",
        "coordinate": coordinate,
        "h_externals": list(h_externals),
        "hbar_externals": list(hbar_externals),
        "ell": ell,
        "q_order": q_order,
        "chiral_norm": chiral_norm,
        "z_shift": z_shift,
        "zbar_shift": zbar_shift,
        "center": None,
        "center_bar": None,
        "parity": parity,
        "parity_external": parity_external,
        "exchanged_parity": exchanged_parity,
        "folded": True,
    }
    return table


def parity_projected_B_ws_poly_table(
    ell,
    h_externals,
    hbar_externals,
    m_max,
    n_max=None,
    *,
    parity: str,
    parity_external=None,
    exchanged_parity: str = "1",
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
    zbar_shift=None,
    skeleton=False,
) -> DeltaDerivTable:
    """Symbolic-in-Delta parity-folded (w, s) table for a projected spinning block over y."""
    return _parity_projected_B_folded(
        z_of_y,
        y_symmetric(),
        ell,
        h_externals,
        hbar_externals,
        m_max,
        n_max,
        coordinate="ws",
        parity=parity,
        parity_external=parity_external,
        exchanged_parity=exchanged_parity,
        b=b,
        c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
        zbar_shift=zbar_shift,
        skeleton=skeleton,
    )


def parity_projected_B_xt_poly_table(
    ell,
    h_externals,
    hbar_externals,
    m_max,
    n_max=None,
    *,
    parity: str,
    parity_external=None,
    exchanged_parity: str = "1",
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
    zbar_shift=None,
    skeleton=False,
) -> DeltaDerivTable:
    """Symbolic-in-Delta parity-folded (x, t) table for a projected spinning block over z."""
    return _parity_projected_B_folded(
        lambda u: u,
        z_symmetric(),
        ell,
        h_externals,
        hbar_externals,
        m_max,
        n_max,
        coordinate="xt",
        parity=parity,
        parity_external=parity_external,
        exchanged_parity=exchanged_parity,
        b=b,
        c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
        zbar_shift=zbar_shift,
        skeleton=skeleton,
    )


def parity_projected_B_qqb_ws_poly_table(
    ell,
    h_externals,
    hbar_externals,
    m_max,
    n_max=None,
    *,
    parity: str,
    parity_external=None,
    exchanged_parity: str = "1",
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
    zbar_shift=None,
    skeleton=False,
) -> DeltaDerivTable:
    """Symbolic-in-Delta parity-folded (w_q, s_q) table for a projected spinning block."""
    return _parity_projected_B_folded(
        z_of_q,
        q_symmetric(),
        ell,
        h_externals,
        hbar_externals,
        m_max,
        n_max,
        coordinate="qqb_ws",
        parity=parity,
        parity_external=parity_external,
        exchanged_parity=exchanged_parity,
        b=b,
        c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
        zbar_shift=zbar_shift,
        skeleton=skeleton,
    )


# --- radial-only slices: ws_radial, xt_radial, qqb_ws_radial ---
#
# The radial table is the n = 0 column of its parity-folded parent (the
# squared variable fixed at 0, i.e. the z = zbar diagonal). It is returned as
# a DeltaDerivTable whose numerators have shape (m_max + 1) x 1, so the
# serialization / interpolation / writer pipeline consumes it unchanged.
# Building the folded parent with n_max = 0 already minimizes the chiral
# Taylor order, so these wrappers are thin.


def G_ws_radial_poly_table(
    ell, h_externals, m_max, *, b=None, c=None, q_order: int, chiral_norm=None,
    z_shift=None, zbar_shift=None, skeleton=False,
) -> DeltaDerivTable:
    """Symbolic-in-Delta radial slice of the (w, s) G table at s = 0 (y = ybar)."""
    return G_ws_poly_table(
        ell, h_externals, m_max, 0, b=b, c=c, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift, zbar_shift=zbar_shift,
        skeleton=skeleton,
    )


def G_xt_radial_poly_table(
    ell, h_externals, m_max, *, b=None, c=None, q_order: int, chiral_norm=None,
    z_shift=None, zbar_shift=None, skeleton=False,
) -> DeltaDerivTable:
    """Symbolic-in-Delta radial slice of the (x, t) G table at t = 0 (z = zbar)."""
    return G_xt_poly_table(
        ell, h_externals, m_max, 0, b=b, c=c, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift, zbar_shift=zbar_shift,
        skeleton=skeleton,
    )


def G_qqb_ws_radial_poly_table(
    ell, h_externals, m_max, *, b=None, c=None, q_order: int, chiral_norm=None,
    z_shift=None, zbar_shift=None, skeleton=False,
) -> DeltaDerivTable:
    """Symbolic-in-Delta radial slice of the (w_q, s_q) G table at s_q = 0."""
    return G_qqb_ws_poly_table(
        ell, h_externals, m_max, 0, b=b, c=c, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift, zbar_shift=zbar_shift,
        skeleton=skeleton,
    )


def parity_projected_B_ws_radial_poly_table(
    ell, h_externals, hbar_externals, m_max, *,
    parity: str, parity_external=None, exchanged_parity: str = "1",
    b=None, c=None, q_order: int, chiral_norm=None,
    z_shift=None, zbar_shift=None, skeleton=False,
) -> DeltaDerivTable:
    """Symbolic-in-Delta radial slice of the projected (w, s) B table at s = 0 (y = ybar)."""
    return parity_projected_B_ws_poly_table(
        ell, h_externals, hbar_externals, m_max, 0,
        parity=parity, parity_external=parity_external,
        exchanged_parity=exchanged_parity, b=b, c=c, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift, zbar_shift=zbar_shift,
        skeleton=skeleton,
    )


def parity_projected_B_xt_radial_poly_table(
    ell, h_externals, hbar_externals, m_max, *,
    parity: str, parity_external=None, exchanged_parity: str = "1",
    b=None, c=None, q_order: int, chiral_norm=None,
    z_shift=None, zbar_shift=None, skeleton=False,
) -> DeltaDerivTable:
    """Symbolic-in-Delta radial slice of the projected (x, t) B table at t = 0 (z = zbar)."""
    return parity_projected_B_xt_poly_table(
        ell, h_externals, hbar_externals, m_max, 0,
        parity=parity, parity_external=parity_external,
        exchanged_parity=exchanged_parity, b=b, c=c, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift, zbar_shift=zbar_shift,
        skeleton=skeleton,
    )


def parity_projected_B_qqb_ws_radial_poly_table(
    ell, h_externals, hbar_externals, m_max, *,
    parity: str, parity_external=None, exchanged_parity: str = "1",
    b=None, c=None, q_order: int, chiral_norm=None,
    z_shift=None, zbar_shift=None, skeleton=False,
) -> DeltaDerivTable:
    """Symbolic-in-Delta radial slice of the projected (w_q, s_q) B table at s_q = 0."""
    return parity_projected_B_qqb_ws_poly_table(
        ell, h_externals, hbar_externals, m_max, 0,
        parity=parity, parity_external=parity_external,
        exchanged_parity=exchanged_parity, b=b, c=c, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift, zbar_shift=zbar_shift,
        skeleton=skeleton,
    )


# --- Chiral-expansion cache for amortizing the q-recursion across spins ---
#
# A ChiralSymbolic depends on (h_externals, b, c, q_order, chiral_norm,
# z_of_u, u_symmetric, n_terms) but NOT on the exchanged ell. A multi-spin
# sweep that recomputes it per (coord, ell) pair pays the q-recursion cost
# linearly in the spin count; precomputing once amortizes that cost. The
# `build_chiral_cache` helper builds the per-coord expansions at the worst-
# case Taylor order for that coordinate (folded coords need 3*lambda + 2;
# non-folded need lambda + 1; radial slices lambda + 2) -- a single cache
# then feeds every spin in the loop.


# (u-coord, z_of_u, u_symmetric_factory) -- coordinates that share a chiral
# variable also share the cached expansion when built at the highest Taylor
# order any of them requires. Derived from the canonical coordinate registry.
_COORD_CHIRAL_FAMILY = {
    coord: (spec.chiral_var, spec.z_of_u, spec.u_symmetric)
    for coord, spec in COORDINATES.items()
}


def _required_n_terms(coord: str, lambda_: int, *, truncated: bool = False) -> int:
    """Worst-case n_terms (Taylor count) for any spin at this coordinate.

    ``truncated`` reflects the m + 2n <= Lambda derivative truncation: when the
    consumer caps n_max at the kept maximum (Lambda // 2 for folded) the folded
    worst case is M_total = lambda + 2*(lambda // 2) + 1 <= 2*lambda + 1, not the
    full square's 3*lambda. Default False keeps the safe full-square size for
    callers that may still build the (lambda, lambda) folded table.
    """
    if coord in ("zzb", "yyb", "qqb", "z", "y", "q"):
        return lambda_ + 1
    if coord in ("ws", "xt", "qqb_ws"):
        # M_total = m_max + 2*n_max + odd_factor; worst case m=n=lambda, odd=1
        # for the full square, or m=lambda / n=lambda//2 when truncated.
        return (2 * lambda_ + 2) if truncated else (3 * lambda_ + 2)
    if coord in ("ws_radial", "xt_radial", "qqb_ws_radial"):
        # M_total = m_max + odd_factor; worst case m=lambda, odd=1.
        return lambda_ + 2
    raise ValueError(f"unknown coord {coord!r}")


def build_chiral_cache(
    coordinates,
    h_externals,
    hbar_externals=None,
    *,
    lambda_: int,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
    z_shift=None,
    zbar_shift=None,
    symbolic_guard: bool = True,
    skeleton: bool = False,
    truncated: bool = False,
) -> dict:
    """Pre-compute the per-coordinate ChiralSymbolic expansions used by every
    spin in a multi-spin sweep.

    ``truncated=True`` sizes folded coordinates for the m + 2n <= Lambda kept
    triangle (n_max capped at Lambda // 2), giving ~2*lambda Taylor terms
    instead of the full square's ~3*lambda -- valid only when every table built
    from this cache caps n_max likewise (the CLI interpolation path does).

    With ``skeleton=True`` the cached expansions carry only the Kac
    denominator / roots / provenance and the numerator degrees (no numerator
    coefficients) -- the cheap build for the numeric-sampler interpolation
    path. See :func:`chiral_symbolic_useries`.

    Returns ``dict[coord, (cs_h, cs_hbar)]`` where ``cs_hbar`` is ``None``
    when ``hbar_externals`` is ``None`` (the scalar G and chiral F paths use
    only ``cs_h``; the ordered B and parity-projected B paths use both).

    Per-coord ``n_terms`` is the worst case across all spins -- folded
    coordinates need ``3*lambda + 2`` Taylor terms (odd parity, m=n=lambda)
    while non-folded need ``lambda + 1`` -- so the same cache covers every
    spin and parity, scalar and folded coordinates, indiscriminately.

    Coordinates sharing the same chiral u-coordinate (e.g. ``zzb`` and ``ws``,
    both on z) are computed at the *maximum* required n_terms of the group;
    one ChiralSymbolic then feeds all of them.
    """
    coordinates = tuple(coordinates)
    lambda_ = _integer_control('lambda_', lambda_)
    q_order = _integer_control('q_order', q_order)
    b, c = _resolve_bc(b, c)
    _check_externals(h_externals)
    if hbar_externals is not None:
        _check_externals(hbar_externals)

    for coord in coordinates:
        if coord not in _COORD_CHIRAL_FAMILY:
            raise ValueError(f"unknown coord {coord!r}")

    # Group coords by chiral family and find the max n_terms for each group.
    # Each family builds its ChiralSymbolic once at the max; coords in the
    # same family share the same object.
    family_to_n_terms: dict = {}
    family_to_zu = {}
    for coord in coordinates:
        family, z_of_u, u_sym_fn = _COORD_CHIRAL_FAMILY[coord]
        n = _required_n_terms(coord, lambda_, truncated=truncated)
        family_to_n_terms[family] = max(family_to_n_terms.get(family, 0), n)
        family_to_zu[family] = (z_of_u, u_sym_fn)

    family_cs: dict = {}
    for family, n_terms in family_to_n_terms.items():
        z_of_u, u_sym_fn = family_to_zu[family]
        u_sym = u_sym_fn()
        cs_h = chiral_symbolic_useries(
            h_externals, n_terms,
            b=b, c=c, z_of_u=z_of_u, u_symmetric=u_sym,
            q_order=q_order, chiral_norm=chiral_norm, z_shift=z_shift,
            symbolic_guard=symbolic_guard, skeleton=skeleton,
        )
        if hbar_externals is None:
            cs_hbar = None
        else:
            cs_hbar = chiral_symbolic_useries(
                hbar_externals, n_terms,
                b=b, c=c, z_of_u=z_of_u, u_symmetric=u_sym,
                q_order=q_order, chiral_norm=chiral_norm, z_shift=zbar_shift,
                symbolic_guard=symbolic_guard, skeleton=skeleton,
            )
        family_cs[family] = (cs_h, cs_hbar)

    return {
        coord: family_cs[_COORD_CHIRAL_FAMILY[coord][0]]
        for coord in coordinates
    }


def build_table_from_cache(
    coord: str,
    cs_h: ChiralSymbolic,
    cs_hbar,
    ell,
    *,
    m_max: int,
    n_max: int,
    basis: str,
    parity_projection: str | None = None,
    parity_external=None,
    exchanged_parity: str = "1",
    lambda_cut: int | None | object = _AUTO_LAMBDA_CUT,
) -> DeltaDerivTable:
    """Build a `DeltaDerivTable` for one (coord, ell) using pre-computed
    chiral expansions.

    ``lambda_cut`` (non-folded coordinates only) skips building the numerator
    products for entries with ``m + n > lambda_cut`` -- the corner the
    ``m + n <= Lambda`` truncation discards at interpolation -- leaving zero
    placeholders. The table dimensions stay ``(m_max, n_max)`` so both chiral
    halves remain available to the parity-projected swap term.

    Corner-skip is the **default**: when ``lambda_cut`` is left unset it
    resolves to ``max(m_max, n_max)`` (the non-folded build is always a square
    at the derivative order Lambda), so every caller -- including
    ``virasoro_bootstrap`` -- gets the better-conditioned truncated build
    without having to opt in. Pass ``lambda_cut=None`` explicitly to force the
    full (m_max, n_max) square; the only in-tree consumer that needs it is the
    folded parity-projected B fallback, which folds the full square and calls
    the level-C builder directly (never through this dispatcher).

    Dispatches to the appropriate level-C builder based on (coord, basis).
    ``cs_hbar`` is consumed only by the spinning paths (``ordered_B``,
    ``parity_projected_B``); the scalar G and chiral F paths ignore it.
    """
    m_max, n_max = _check_table_orders(m_max, n_max)
    lambda_cut = _resolve_lambda_cut(lambda_cut, m_max, n_max)
    if basis == "chiral_F":
        # chiral_F has one Taylor index; the table is (m_max + 1) x 1.
        # ell is ignored.
        table = _build_F_chiral_poly_table(cs_h, m_max)
        table.oracle_metadata = {
            "basis": "chiral_F",
            "coordinate": coord,
            "h_externals": list(getattr(cs_h, "h_externals", []) or []),
            "q_order": getattr(cs_h, "q_order", None),
            "chiral_norm": getattr(cs_h, "chiral_norm", None),
            "z_shift": getattr(cs_h, "z_shift", None),
            "center": None,
        }
        return table

    # The default corner-skip uses a non-folded square at the
    # derivative order Lambda = max(m_max, n_max), so that is the kept-triangle
    # cutoff. Only relevant to the non-folded branch below; the folded / radial
    # builders carry their own m + 2n <= Lambda truncation and ignore it.
    if coord in ("zzb", "yyb", "qqb"):
        if basis == "scalar_G":
            table = _build_G_poly_table(cs_h, ell, m_max, n_max, lambda_cut)
            table.oracle_metadata = {
                "basis": "scalar_G",
                "coordinate": coord,
                "h_externals": list(getattr(cs_h, "h_externals", []) or []),
                "ell": ell,
                "q_order": getattr(cs_h, "q_order", None),
                "chiral_norm": getattr(cs_h, "chiral_norm", None),
                "z_shift": getattr(cs_h, "z_shift", None),
                "center": None,
                "center_bar": None,
                "folded": False,
            }
            return table
        if basis == "ordered_B":
            table = _build_B_poly_table(
                cs_h, cs_hbar, ell, m_max, n_max, lambda_cut
            )
            table.oracle_metadata = {
                "basis": "ordered_B",
                "coordinate": coord,
                "h_externals": list(getattr(cs_h, "h_externals", []) or []),
                "hbar_externals": list(getattr(cs_hbar, "h_externals", []) or []),
                "ell": ell,
                "q_order": getattr(cs_h, "q_order", None),
                "chiral_norm": getattr(cs_h, "chiral_norm", None),
                "z_shift": getattr(cs_h, "z_shift", None),
                "zbar_shift": getattr(cs_hbar, "z_shift", None),
                "center": None,
                "center_bar": None,
                "folded": False,
            }
            return table
        if basis == "parity_projected_B":
            table = _build_parity_projected_B_poly_table(
                cs_h, cs_hbar, ell, m_max, n_max,
                parity=parity_projection,
                parity_external=parity_external,
                exchanged_parity=exchanged_parity,
                lambda_cut=lambda_cut,
            )
            table.oracle_metadata = {
                "basis": "parity_projected_B",
                "coordinate": coord,
                "h_externals": list(getattr(cs_h, "h_externals", []) or []),
                "hbar_externals": list(getattr(cs_hbar, "h_externals", []) or []),
                "ell": ell,
                "q_order": getattr(cs_h, "q_order", None),
                "chiral_norm": getattr(cs_h, "chiral_norm", None),
                "z_shift": getattr(cs_h, "z_shift", None),
                "zbar_shift": getattr(cs_hbar, "z_shift", None),
                "center": None,
                "center_bar": None,
                "parity": parity_projection,
                "parity_external": parity_external,
                "exchanged_parity": exchanged_parity,
                "folded": False,
            }
            return table

    if coord in ("ws", "xt", "qqb_ws"):
        if basis == "scalar_G":
            table = _G_folded_poly_table(cs_h, ell, m_max, n_max)
            table.oracle_metadata = {
                "basis": "scalar_G",
                "coordinate": coord,
                "h_externals": list(getattr(cs_h, "h_externals", []) or []),
                "ell": ell,
                "q_order": getattr(cs_h, "q_order", None),
                "chiral_norm": getattr(cs_h, "chiral_norm", None),
                "z_shift": getattr(cs_h, "z_shift", None),
                "center": None,
                "center_bar": None,
                "folded": True,
            }
            return table
        if basis == "parity_projected_B":
            table = _parity_projected_B_folded_table(
                cs_h, cs_hbar, ell, m_max, n_max,
                parity=parity_projection,
                parity_external=parity_external,
                exchanged_parity=exchanged_parity,
            )
            table.oracle_metadata = {
                "basis": "parity_projected_B",
                "coordinate": coord,
                "h_externals": list(getattr(cs_h, "h_externals", []) or []),
                "hbar_externals": list(getattr(cs_hbar, "h_externals", []) or []),
                "ell": ell,
                "q_order": getattr(cs_h, "q_order", None),
                "chiral_norm": getattr(cs_h, "chiral_norm", None),
                "z_shift": getattr(cs_h, "z_shift", None),
                "zbar_shift": getattr(cs_hbar, "z_shift", None),
                "center": None,
                "center_bar": None,
                "parity": parity_projection,
                "parity_external": parity_external,
                "exchanged_parity": exchanged_parity,
                "folded": True,
            }
            return table
        # ordered_B is rejected by the truncation policy; defensive raise.
        raise ValueError(
            f"basis {basis!r} not supported on folded coordinate {coord!r}"
        )

    if coord in ("ws_radial", "xt_radial", "qqb_ws_radial"):
        # Radial = folded with n_max = 0.
        if basis == "scalar_G":
            table = _G_folded_poly_table(cs_h, ell, m_max, 0)
            table.oracle_metadata = {
                "basis": "scalar_G",
                "coordinate": coord,
                "h_externals": list(getattr(cs_h, "h_externals", []) or []),
                "ell": ell,
                "q_order": getattr(cs_h, "q_order", None),
                "chiral_norm": getattr(cs_h, "chiral_norm", None),
                "z_shift": getattr(cs_h, "z_shift", None),
                "center": None,
                "center_bar": None,
                "folded": True,
            }
            return table
        if basis == "parity_projected_B":
            table = _parity_projected_B_folded_table(
                cs_h, cs_hbar, ell, m_max, 0,
                parity=parity_projection,
                parity_external=parity_external,
                exchanged_parity=exchanged_parity,
            )
            table.oracle_metadata = {
                "basis": "parity_projected_B",
                "coordinate": coord,
                "h_externals": list(getattr(cs_h, "h_externals", []) or []),
                "hbar_externals": list(getattr(cs_hbar, "h_externals", []) or []),
                "ell": ell,
                "q_order": getattr(cs_h, "q_order", None),
                "chiral_norm": getattr(cs_h, "chiral_norm", None),
                "z_shift": getattr(cs_h, "z_shift", None),
                "zbar_shift": getattr(cs_hbar, "z_shift", None),
                "center": None,
                "center_bar": None,
                "parity": parity_projection,
                "parity_external": parity_external,
                "exchanged_parity": exchanged_parity,
                "folded": True,
            }
            return table
        raise ValueError(
            f"basis {basis!r} not supported on radial coordinate {coord!r}"
        )

    raise ValueError(f"unknown coord {coord!r}")


# --- The interpolation method (order reduction) ---


@dataclass
class InterpolatedDeltaTable:
    """G derivative table in the order-reduced interpolation representation.

        table[m][n](Delta) = prefactor(Delta - shift)
                             * numerators[m][n](Delta - shift)

    `prefactor` is the universal damped rational keeping the selected
    Delta-poles and every forced (x > 0) pole. `numerators[m][n]` is a polynomial in x = Delta - shift
    interpolating the exact table at *its own* set of optimal sample nodes:
    each (m, n) entry's polynomial has degree
    ``exact.numerators[m][n].degree - dropped_pole_count``, which is the
    natural degree of the rational `N · kept_poles / D_selectable` at Δ → ∞
    and preserves the large-Δ scaling of the underlying block. Lower-degree
    entries use a per-target Bohr-Sommerfeld run (cached) rather than a
    prefix of the highest-degree entry's nodes, so each entry gets nodes
    spread optimally across its support.

    `sample_points` is the *highest-degree* entry's node set (the one shared
    by every (m, n) with maximal exact polynomial degree). Per-entry node
    sets are available via `numerator_sample_points[m][n]`; tests that
    check exactness at sample nodes should index by entry.
    """

    prefactor: DampedRational
    shift: object
    sample_points: tuple
    numerators: list  # reduced Polynomial in x = Delta - shift
    min_pole_distance: object | None = None
    numerator_sample_points: list | None = None  # per-(m, n) node tuples
    # per-(m, n) block samples at those nodes, ``(is_arb, raw, reduced)`` with arb samples stored as their exact dyadic
    # midpoints (arb does not pickle): the clean-sample writer reuses them instead of re-sampling the block
    numerator_sample_values: list | None = None
    # Optional family/provenance metadata. Ordinary Virasoro interpolation
    # leaves this unset; the NS/Ramond adapter stamps its shared policy here so
    # serialization can preserve the block-family contract.
    metadata: dict | None = None
    # Poles on x > 0 (c < 1) split off as exact principal parts: the table is ``remainder_prefactor(x) N_rem(x) +
    # exp_base^Delta sum_a g_a (Delta - loc)^-a``, recombined into ``prefactor * numerators``.  The fit and its node
    # records are the remainder's; ``principal_parts = (exp_base, [(loc, {(m, n): (g_1, ..., g_M)})])``.
    principal_parts: tuple | None = None
    remainder_prefactor: DampedRational | None = None
    working_dps: int | None = _dataclass_field(default_factory=lambda: mp.mp.dps)

    @property
    def m_max(self) -> int:
        return len(self.numerators) - 1

    @property
    def n_max(self) -> int:
        return len(self.numerators[0]) - 1

    def interpolant(self, m: int, n: int) -> DeltaInterpolant:
        """Return the standalone `DeltaInterpolant` for one (m, n) entry."""
        return DeltaInterpolant(
            prefactor=self.prefactor,
            shift=self.shift,
            sample_points=self.sample_points,
            reduced_polynomial=self.numerators[m][n],
            min_pole_distance=self.min_pole_distance,
        )

    def _prefactor_value(self, x):
        if self.min_pole_distance is None:
            return self.prefactor.evaluate(x)
        return self.prefactor.evaluate_regulated(x, self.min_pole_distance)

    def _fit_prefactor_value(self, x):
        """The prefactor the numerator fit divided by: the remainder's when principal parts were split off."""
        if self.remainder_prefactor is None:
            return self._prefactor_value(x)
        if self.min_pole_distance is None:
            return self.remainder_prefactor.evaluate(x)
        return self.remainder_prefactor.evaluate_regulated(x, self.min_pole_distance)

    def evaluate(self, Delta):
        """Reconstruct the numeric (u, ubar)-derivative table at this Delta.

        Real Delta + a real shift keeps x in mp.mpf; if the prefactor is also
        all-real (every pole real, real_b / imag_b modes), the whole
        evaluation stays in mp.mpf.
        """
        x = _evaluation_number(Delta) - _evaluation_number(self.shift)
        if self.min_pole_distance is None:
            pref = self.prefactor._evaluate_at_work(x)
        else:
            pref = self.prefactor._evaluate_regulated_at_work(x, self.min_pole_distance)
        return [[pref * num.evaluate(x) for num in row] for row in self.numerators]


def _residue_denom_factors_acb(roots, leading_coeff, tol, prec_bits):
    """Per-pole denominator factors for residue scoring, computed in flint.

    For pole ``p_j`` returns ``leading_coeff * prod_{k != j, |p_j-p_k| > tol}
    (p_j - p_k)`` as a flint ``acb`` -- i.e. ``D'(p_j)`` for a simple pole, and
    the reduced product (with the coincident copies removed) for the ell = 0
    double poles. These factors depend only on the Kac denominator roots and
    leading coefficient, which are shared by every coordinate descended from
    the same block, so `_aggregate_residue_scores` computes them once and
    reuses them across all coordinate tables.

    Near-duplicate poles (``|p_j - p_k| <= tol``) are excluded from the product,
    reproducing the legacy mpmath handling of ell = 0 double poles and the
    epsilon-extrapolation coincidences at recursion-singular c.

    Returns ``(poles_acb, factors_acb)`` aligned to ``roots``. The whole
    O(n_poles^2) product runs in flint ``acb`` arithmetic, replacing the
    per-pole mpmath ``mpc`` product / ``abs`` loops that dominated residue
    scoring.
    """
    from flint import arb, acb, ctx as flint_ctx
    from .polynomial import _mpf_to_acb

    flint_ctx.prec = prec_bits
    poles_acb = [_mpf_to_acb(r) for r in roots]
    lead_acb = _mpf_to_acb(leading_coeff)
    tol2 = arb(tol) * arb(tol)

    factors = []
    n = len(poles_acb)
    for j in range(n):
        pj = poles_acb[j]
        product = acb(arb(1))
        for k in range(n):
            if k == j:
                continue
            diff = pj - poles_acb[k]
            mag2 = diff.real * diff.real + diff.imag * diff.imag
            if mag2 > tol2:
                product = product * diff
        factors.append(lead_acb * product)
    return poles_acb, factors


def _value_score_entry(exact: "DeltaDerivTable") -> tuple[int, int]:
    """The single (m, n) entry the full-contribution ``"value"`` scoring uses.

    Normally the undifferentiated ``(0, 0)`` block value. But that entry vanishes
    identically when the block is z<->z̄ ANTIsymmetric on an *unfolded* coordinate
    -- parity='-' ``projected_B`` on zzb/yyb/qqb (never ``scalar_G``, which is
    z<->z̄ even at every spin), whose diagonal value is antisymmetric (the folded coordinates strip the vanishing
    sqrt(s) factor via parity smoothing, so they keep a non-zero ``(0, 0)``).
    Scoring on a vanishing entry gives a degenerate all-zero ranking, so fall
    back to the FIRST kept ``(m, n)`` in ``(m + n, m)`` order whose entry is
    non-zero -- the lowest-order non-vanishing derivative.

    The choice is structural (a block entry is non-zero iff its numerator is, and
    the denominator is generically non-zero), so the symbolic path (via the
    numerators) and the numeric/skeleton path (via a sampler probe) resolve to
    the SAME entry, keeping numeric == symbolic and skeleton == full.
    """
    meta = getattr(exact, "oracle_metadata", None) or {}
    sampler = getattr(exact, "numeric_sampler", None)
    basis = meta.get("basis")
    folded = bool(meta.get("folded", False))
    center = meta.get("center")
    center_bar = meta.get("center_bar")
    parity = meta.get("parity")
    ell = meta.get("ell", getattr(exact, "spin", 0))
    exchanged_parity = meta.get("exchanged_parity")

    # The hot interpolation path often builds tables via build_table_from_cache,
    # which may reach here before any wrapper stamps oracle_metadata. When the
    # numeric sampler is attached, recover the same structural classification
    # directly from the sampler.
    if basis is None and sampler is not None:
        if hasattr(sampler, "projected"):
            basis = "parity_projected_B" if sampler.projected else "ordered_B"
            folded = getattr(sampler, "folded", folded)
            center = getattr(sampler, "center", center)
            center_bar = getattr(sampler, "center_bar", center_bar)
            parity = getattr(sampler, "parity", parity)
            ell = getattr(sampler, "ell", ell)
            exchanged_parity = getattr(sampler, "exchanged_parity", exchanged_parity)
        elif hasattr(sampler, "basis"):
            basis = getattr(sampler, "basis", basis)
            folded = getattr(sampler, "folded", folded)
            center = getattr(sampler, "center", center)
            center_bar = getattr(sampler, "center_bar", center_bar)
            parity = getattr(sampler, "parity", parity)
            ell = getattr(sampler, "ell", ell)
            exchanged_parity = getattr(sampler, "exchanged_parity", exchanged_parity)
        elif hasattr(sampler, "ell") and hasattr(sampler, "coordinate"):
            basis = "scalar_G"
            coord = getattr(sampler, "coordinate", None)
            folded = folded or coord in {"ws", "xt", "qqb_ws"}
            center = getattr(sampler, "center", center)
            center_bar = getattr(sampler, "center_bar", center_bar)
            ell = getattr(sampler, "ell", ell)
        elif hasattr(sampler, "coordinate"):
            basis = "chiral_F"

    structural = _value_score_entry_from_structure(
        basis=basis,
        folded=folded,
        center=center,
        center_bar=center_bar,
        parity=parity,
        ell=ell,
        exchanged_parity=exchanged_parity,
        n_max=exact.n_max,
    )
    if structural is not None:
        return structural

    m_max, n_max = exact.m_max, exact.n_max
    bound = max(m_max, n_max)
    order = sorted(
        ((m, n) for m in range(m_max + 1) for n in range(n_max + 1)
         if m + n <= bound),
        key=lambda e: (e[0] + e[1], e[0]),
    )
    nums = exact.numerators
    if nums is not None:
        for m, n in order:
            num = nums[m][n]
            if not ((not num.coeffs) or all(mp.mpc(c) == 0 for c in num.coeffs)):
                return (m, n)
        return (0, 0)
    sampler = getattr(exact, "numeric_sampler", None)
    if sampler is None:
        return (0, 0)
    # Skeleton (degree-only) path: probe the block at two generic off-pole Δ and
    # take the first entry whose value is non-negligible (structural zeros are
    # ~1e-40 vs O(1) for genuine entries).
    probes = [mp.mpf("3.7"), mp.mpf("6.1")]
    grids, is_arb = sampler(probes, entries=order)
    if is_arb:
        from .polynomial import _acb_to_mpc
        mag = lambda v: abs(mp.mpc(_acb_to_mpc(v)))
    else:
        mag = lambda v: abs(mp.mpc(v))
    mags = {
        (m, n): max(mag(grids[k][m][n]) for k in range(len(probes)))
        for m, n in order
    }
    scale = max(mags.values()) if mags else mp.mpf(0)
    if scale == 0:
        return (0, 0)
    for m, n in order:
        if mags[(m, n)] > mp.mpf("1e-18") * scale:
            return (m, n)
    return (0, 0)


def _canonical_value_basis(basis):
    if basis is None:
        return None
    if basis.startswith("global_"):
        return basis[len("global_"):]
    return basis


def _is_negative_one_phase(value):
    if value is None:
        return False
    z = mp.mpc(value)
    tol = _validation_tolerance(guard=5)
    return abs(z.imag) <= tol and abs(z.real + 1) <= tol


def _value_score_entry_from_structure(
    *, basis, folded, center, center_bar, parity, ell, exchanged_parity, n_max
):
    basis = _canonical_value_basis(basis)
    if basis is None:
        return None
    off_diagonal = center_bar is not None and center_bar != center
    if not folded and not off_diagonal and n_max >= 1:
        # (scalar_G is z<->z̄ even at every spin, so its (0, 0) entry never vanishes)
        if basis == "parity_projected_B" and parity == "-":
            return (0, 1)
    if basis in {"scalar_G", "ordered_B", "parity_projected_B", "chiral_F"}:
        return (0, 0)
    return None


def _value_score_entry_from_sampler(sampler, *, n_max):
    return _value_score_entry_from_structure(
        basis=getattr(sampler, "basis", None),
        folded=bool(getattr(sampler, "folded", False)),
        center=getattr(sampler, "center", None),
        center_bar=getattr(sampler, "center_bar", None),
        parity=getattr(sampler, "parity", None),
        ell=getattr(sampler, "ell", 0),
        exchanged_parity=getattr(sampler, "exchanged_parity", None),
        n_max=n_max,
    )


def _residue_scored_numerators(exact: "DeltaDerivTable", score_entries: str = "kept"):
    """Yield the numerator entries that residue scoring should rank over.

    ``score_entries`` selects which (m, n) entries contribute to the per-pole
    residue max:

    - ``"kept"`` (default): the ``m + n <= max(m_max, n_max)`` triangle. The
      exact table is built as a full square ``(m_max + 1) x (n_max + 1)`` grid,
      but downstream truncation (`truncation.kept_pairs`) only ever keeps the
      ``m + n <= Lambda`` triangle: every supported coordinate/basis bounds the
      total derivative weight by ``m + n <= Lambda`` (the parity-folded
      ``m + 2n <= Lambda`` and the ``n = 0`` chiral/radial slices are subsets
      of it), and the square table is sized with ``m_max = n_max = Lambda``. The
      ``m + n > max(m_max, n_max)`` corner entries are therefore discarded by
      every basis, so a pole's strength in them can never matter to the kept
      functional. Restricting the residue max to the triangle removes that
      always-discarded work (a ~1.8x cut of the dominant scoring loop) without
      changing which poles are selected; the bound is a safe superset of the
      true kept set for every basis, so the aggregated ranking is unchanged.

    - ``"value"``: only the ``(0, 0)`` entry -- the undifferentiated block
      value. Ranking poles by their residue in the block value itself (rather
      than the max over all derivatives) keeps the poles that matter most to
      the physical block, instead of being dominated by the highest-derivative
      residues that over-weight deep poles. Empirically this gives ~100-1000x
      smaller block reconstruction error at fixed ``npoles`` for the
      identical-scalar fixed-spin channels. ``pole_selection="residue"`` and
      the direct ``"contribution"`` selector use this value entry; the
      production ``"contribution_optimal"`` selector instead uses the retained
      derivative triangle in its optimal-node remainder estimate.
    """
    if score_entries == "value":
        # The undifferentiated (0, 0) block value, with the first-nonzero
        # fallback when (0, 0) vanishes identically (parity '-' projected_B on
        # an unfolded coordinate).
        m0, n0 = _value_score_entry(exact)
        yield exact.numerators[m0][n0]
        return
    if score_entries != "kept":
        raise ValueError(
            f"score_entries must be 'kept' or 'value'; got {score_entries!r}"
        )
    m_max = len(exact.numerators) - 1
    n_max = len(exact.numerators[0]) - 1 if exact.numerators else 0
    bound = max(m_max, n_max)
    for m, row in enumerate(exact.numerators):
        for n, num in enumerate(row):
            if m + n <= bound:
                yield num


def _residue_score_indices_from_dims(
    m_max: int, n_max: int, score_entries: str, value_entry: tuple = (0, 0)
):
    if score_entries == "value":
        # Normally (0, 0); the caller passes the first-nonzero fallback entry
        # when (0, 0) vanishes identically (odd-parity unfolded block), so the
        # residue ranking stays consistent with the contribution scorer.
        return [value_entry]
    if score_entries != "kept":
        raise ValueError(
            f"score_entries must be 'kept' or 'value'; got {score_entries!r}"
        )
    bound = max(m_max, n_max)
    return [
        (m, n)
        for m in range(m_max + 1)
        for n in range(n_max + 1)
        if m + n <= bound
    ]


def _series_mul_truncated_numeric(a, b, n_terms: int):
    if n_terms <= 0:
        return []
    zero = mp.mpf(0)
    for val in a:
        if isinstance(val, mp.mpc) and val.imag != 0:
            zero = mp.mpc(0)
            break
    if zero == mp.mpf(0):
        for val in b:
            if isinstance(val, mp.mpc) and val.imag != 0:
                zero = mp.mpc(0)
                break
    out = [zero] * n_terms
    for i in range(min(len(a), n_terms)):
        ai = a[i]
        if ai == 0:
            continue
        for j in range(min(len(b), n_terms - i)):
            bj = b[j]
            if bj != 0:
                out[i + j] += ai * bj
    return out


def _q_powers_truncated_numeric(q_series, q_order: int, n_terms: int):
    if n_terms <= 0:
        return []
    zero = mp.mpf(0)
    for val in q_series:
        if isinstance(val, mp.mpc) and val.imag != 0:
            zero = mp.mpc(0)
            break
    one = mp.mpf(1) if zero == mp.mpf(0) else mp.mpc(1)
    q_powers = [[zero] * n_terms for _ in range(q_order + 1)]
    q_powers[0][0] = one
    for k in range(1, q_order + 1):
        q_powers[k] = _series_mul_truncated_numeric(q_powers[k - 1], q_series, n_terms)
    return q_powers


def _build_numeric_flint_cache(q_powers, a_series, s_series, n_terms: int):
    from flint import acb_mat, acb_series, ctx as flint_ctx

    prec_bits = int(mp.mp.dps * 3.33) + 20
    ndigits = mp.mp.dps + 10
    flint_ctx.prec = prec_bits
    flint_ctx.cap = n_terms

    q_order = len(q_powers) - 1
    W = acb_mat([
        [_mpf_to_acb(q_powers[k][m]) for k in range(q_order + 1)]
        for m in range(n_terms)
    ])
    a_acb = acb_series([_mpf_to_acb(v) for v in a_series[:n_terms]])
    s_coeffs = [_mpf_to_acb(v) for v in s_series[:n_terms]]
    return {
        "prec_bits": prec_bits,
        "ndigits": ndigits,
        "n_terms": n_terms,
        "W": W,
        "a_series": a_acb,
        "s_coeffs": s_coeffs,
    }


def _u_series_from_q_coeffs_numeric_flint(
    q_coeffs,
    h_eval,
    *,
    flint_cache,
    as_mpmath: bool = True,
):
    from flint import acb, acb_mat, acb_series, ctx as flint_ctx

    n_terms = flint_cache["n_terms"]
    q_order = len(q_coeffs) - 1
    if q_order + 1 != flint_cache["W"].ncols():
        return None

    flint_ctx.prec = flint_cache["prec_bits"]
    flint_ctx.cap = n_terms

    q_vec = acb_mat([[_mpf_to_acb(q_coeffs[k])] for k in range(q_order + 1)])
    n_vec = flint_cache["W"] * q_vec
    n_series = acb_series([n_vec[m, 0] for m in range(n_terms)])

    h_acb = _mpf_to_acb(h_eval)
    hs_coeffs = [acb(0)] + [
        h_acb * flint_cache["s_coeffs"][m]
        for m in range(1, n_terms)
    ]
    exp_series = acb_series(hs_coeffs).exp()
    out_series = exp_series * flint_cache["a_series"] * n_series

    coeffs = list(out_series.coeffs())[:n_terms]
    if len(coeffs) < n_terms:
        coeffs += [acb(0)] * (n_terms - len(coeffs))
    if not as_mpmath:
        return coeffs
    return [_acb_to_mpc(c, flint_cache["ndigits"]) for c in coeffs]


def _u_series_leading_from_q_coeffs_flint(q_coeffs, flint_cache):
    """Leading u-Taylor coefficient (m=0) of the residue/regular series.

    Equals ``_u_series_from_q_coeffs_numeric_flint(...)[0]`` but skips the full
    (n_terms x q_order) matmul, the exp-series, and the two series products:
    the 0th coefficient of ``exp(h*s) * a * n`` is ``exp(0) * a[0] * n[0]`` with
    ``n[0] = sum_k W[0, k] * q_coeffs[k]`` (and ``exp(h*s)[0] = 1`` because the
    log-E constant term is zeroed). Used by the ``"value"``/``"contribution"``
    pole ranking, which only ever consults the (0, 0) entry. Returns an ``acb``
    (no h-dependence at order 0), or ``None`` if shapes disagree.
    """
    from flint import acb, ctx as flint_ctx
    W = flint_cache["W"]
    q_order = len(q_coeffs) - 1
    if q_order + 1 != W.ncols():
        return None
    flint_ctx.prec = flint_cache["prec_bits"]
    a_coeffs = flint_cache["a_series"].coeffs()
    a0 = a_coeffs[0] if a_coeffs else acb(0)
    acc = acb(0)
    for k in range(q_order + 1):
        ck = q_coeffs[k]
        if ck == 0:
            continue
        acc += W[0, k] * _mpf_to_acb(ck)
    return a0 * acc


def _u_series_from_q_coeffs_numeric(
    q_coeffs,
    h_eval,
    *,
    a_series,
    q_powers,
    exp_series,
    flint_cache=None,
    n_terms: int,
):
    if flint_cache is not None:
        flint_out = _u_series_from_q_coeffs_numeric_flint(
            q_coeffs,
            h_eval,
            flint_cache=flint_cache,
        )
        if flint_out is not None:
            return flint_out

    q_order = len(q_coeffs) - 1
    zero = mp.mpf(0)
    for val in q_coeffs:
        if isinstance(val, mp.mpc) and val.imag != 0:
            zero = mp.mpc(0)
            break
    if zero == mp.mpf(0):
        for val in a_series:
            if isinstance(val, mp.mpc) and val.imag != 0:
                zero = mp.mpc(0)
                break
    if zero == mp.mpf(0):
        for coeff in exp_series.coeffs[:n_terms]:
            if coeff.field is mp.mpc:
                zero = mp.mpc(0)
                break

    n_coeffs = [zero] * n_terms
    for k in range(q_order + 1):
        qk = q_coeffs[k]
        if qk == 0:
            continue
        wk = q_powers[k]
        for m in range(n_terms):
            wkm = wk[m]
            if wkm != 0:
                n_coeffs[m] += wkm * qk

    exp_vals = [exp_series.coeffs[m].evaluate(h_eval) for m in range(n_terms)]
    return _series_mul_truncated_numeric(
        _series_mul_truncated_numeric(exp_vals, a_series, n_terms),
        n_coeffs,
        n_terms,
    )


def _coordinate_series_from_sampler(sampler, n_terms: int, *, externals=None,
                                    z_shift=None):
    coord = sampler.coordinate
    if coord not in _COORD_CHIRAL_FAMILY:
        return None
    _family, z_of_u, u_sym_fn = _COORD_CHIRAL_FAMILY[coord]
    u_symmetric = u_sym_fn()

    b, c = _resolve_bc(sampler.b, sampler.c)
    N = _resolve_chiral_norm(sampler.chiral_norm)
    # `externals` / `z_shift` override the sampler's holomorphic side so the
    # caller can build the antiholomorphic chiral series (hbar_externals,
    # zbar_shift) for the two-sided residue scoring of a spinning ordered /
    # parity-projected block. Default to the sampler's h-side.
    if externals is None:
        externals = sampler.h_externals
    if z_shift is None:
        z_shift = sampler.z_shift
    h1, h2, h3, h4 = externals
    z_shift = mp.mpf(0) if z_shift is None else z_shift
    sigma = (c - 1) / 24
    theta_exp = (c - 1) / 2 - 4 * (h1 + h2 + h3 + h4)

    q_series, log_e_series, a_series = _flint_taylor_series(
        z_of_u,
        u_symmetric,
        n_terms,
        sigma,
        theta_exp,
        z_shift,
        h2,
        h3,
        N,
    )

    while len(q_series) < n_terms:
        q_series.append(mp.mpf(0))
    while len(log_e_series) < n_terms:
        log_e_series.append(mp.mpf(0))
    while len(a_series) < n_terms:
        a_series.append(mp.mpf(0))

    return {
        "b": b,
        "q_order": int(sampler.q_order),
        "h_externals": tuple(externals),
        "q_series": list(q_series[:n_terms]),
        "a_series": list(a_series[:n_terms]),
        "s_series": [mp.mpf(0)] + list(log_e_series[1:n_terms]),
    }


def _numeric_value_scores_fast(exact, pair_table, arb_ctx, flint_cache, ell, swap_sign=1):
    """Partial-fraction fast path for the (0, 0)-value pole ranking.

    The leading (m=0) u-Taylor coefficient of every residue/regular block is a
    linear functional of the chiral H-table, so the whole "value" ranking
    reduces to one precomputed vector and an O(n_pairs) sum per pole:

        P    = W[0] . M                       (one matrix-vector product)
        c    = a0 * P * K                      (elementwise)
        rho0_j  = c_j
        reg0(h) = a0 + sum_idx c_idx / (h - d_idx)

    where ``M[k][idx] = H_{k-rs}(v_idx)`` is the prebuilt H-table matrix from the
    recursion context, ``a0`` the constant u-Taylor term of the A-prefactor, and
    ``W[0][k] = q(z(u_sym))^k`` the geometry weights. This replaces the per-pole
    residue u-series and every regular-block H-series evaluation (which the
    general path computes in full only to read coefficient 0). Returns the
    scores, or ``None`` to defer to the general path (non-strip context or
    shape mismatch).
    """
    if arb_ctx is None:
        return None
    from flint import acb, acb_mat, ctx as flint_ctx

    flint_ctx.prec = flint_cache["prec_bits"]
    mode = arb_ctx.get("mode")
    if mode == "strip_real":
        # Kac data already acb (b on the unit circle).
        M = arb_ctx["M_full"]
        K = arb_ctx["K_full"]
        d = arb_ctx["d_full"]
        n_pairs = arb_ctx["n_pairs_full"]
    elif mode == "real_nonstrip":
        # Real Kac data (c > 25): promote the arb context to acb so the same
        # partial-fraction arithmetic applies (a0 / W from the cache are acb).
        # The numeric residue K*H(v) is well-conditioned; the symbolic score
        # (degree ~q_order numerator with ~1e74 coefficients evaluated at the
        # pole) is not, so numeric ranking is the *correct* choice here, not
        # just the fast one -- verified against the direct residue eps*G(p+eps).
        M = acb_mat(arb_ctx["M"])
        K = [acb(k) for k in arb_ctx["K"]]
        d = [acb(x) for x in arb_ctx["d"]]
        n_pairs = arb_ctx["n_pairs"]
    else:
        return None
    ndig = flint_cache["ndigits"]
    W = flint_cache["W"]
    qcols = W.ncols()
    if M.nrows() != qcols or len(K) != n_pairs or len(d) != n_pairs:
        return None
    a_coeffs = flint_cache["a_series"].coeffs()
    a0 = a_coeffs[0] if a_coeffs else acb(0)

    # P = W[0] . M  (1 x n_pairs); c_idx = a0 * P_idx * K_idx.
    W0 = acb_mat([[W[0, k] for k in range(qcols)]])
    P = W0 * M
    cvec = [a0 * P[0, idx] * K[idx] for idx in range(n_pairs)]

    lbl2idx = {(p["r"], p["s"]): i for i, p in enumerate(pair_table.pairs)}
    sign_int = swap_sign
    # the spin-0 double pole of (1 + sigma) c_h c_h: 4 rho0^2 per unit weight (one table, one scale: ranking only)
    spin0_factor = 4 * (1 + scalar_G_swap_coefficient(0))

    def reg0(h):
        h_acb = _mpf_to_acb(h)
        s = a0
        for idx in range(n_pairs):
            s = s + cvec[idx] / (h_acb - d[idx])
        return s

    scores = []
    for prov in exact.root_provenance:
        if not (isinstance(prov, tuple) and len(prov) == 2 and isinstance(prov[0], str)):
            return None
        side, labels = prov
        if side not in ("left", "right"):
            return None
        labels = tuple(labels) if isinstance(labels, (tuple, list)) else ()
        if not labels:
            return None
        idx = lbl2idx.get(tuple(labels[0]))
        if idx is None:
            return None
        rho0 = cvec[idx]
        if ell == 0:
            val = abs(spin0_factor * rho0 * rho0)
        else:
            d_val = mp.mpc(pair_table.pairs[idx]["d"])
            h = d_val - ell if side == "left" else d_val + ell
            r0 = reg0(h)
            val = abs(2 * (rho0 * r0 + sign_int * r0 * rho0))
        scores.append(mp.mpf(val.str(ndig, radius=False)))

    if len(scores) != len(exact.denominator_roots):
        return None
    return scores


def _root_provenance_key(prov):
    """Canonical hashable identity of a Delta-pole by its chiral Kac origin.

    Accepts one ``DeltaDerivTable.root_provenance`` entry -- a ``(side, labels)``
    pair with ``side`` a string (``"left"`` / ``"right"`` for G/B, ``"chiral"``
    for chiral-F) and ``labels`` a non-empty container of Kac ``(r, s)`` pairs --
    and returns the hashable key ``(side, tuple(labels))``. Returns ``None`` for
    a malformed entry so callers can decline cleanly (numeric scorers then fall
    back to symbolic).

    Pole identity must be tracked by this key across epsilon probes, never by
    root order or numeric value: distinct Kac pairs can collide to one root
    value, and the roots can reorder between probe central charges.
    """
    if not (isinstance(prov, tuple) and len(prov) == 2 and isinstance(prov[0], str)):
        return None
    side, labels = prov
    labels = tuple(labels) if isinstance(labels, (tuple, list)) else ()
    if not labels:
        return None
    return (side, labels)


def _ordered_scores_from_provenance(exact, score_map):
    """Restore the per-root score list (in ``denominator_roots`` order) from a
    provenance-keyed ``{key: score}`` map.

    Returns ``None`` if the map is ``None`` or does not cover every root (the
    legacy ``len(scores) != len(denominator_roots) -> None`` contract), so the
    caller defers to symbolic scoring. Distinct roots that share a provenance
    key (e.g. an ``ell = 0`` coincident double pole) resolve to the same score,
    matching the per-root recomputation the list-based scorers did.
    """
    if score_map is None:
        return None
    ordered = []
    for prov in exact.root_provenance:
        key = _root_provenance_key(prov)
        if key is None or key not in score_map:
            return None
        ordered.append(score_map[key])
    if len(ordered) != len(exact.denominator_roots):
        return None
    return ordered


def numeric_residue_scoring_available(exact: DeltaDerivTable) -> bool:
    """Whether numeric (sampler-based) residue pole ranking can run on `exact`.

    These are exactly the preconditions of :func:`_numeric_residue_scores`
    (a numeric sampler, a 2d spin, no removable-pole strip pending, matching
    root provenance, python-flint, and -- for spinning blocks -- non-degenerate
    poles). Exposed so the skeleton (degree-only) build path can confirm the
    symbolic numerators will not be needed for residue ranking before skipping
    them. Cheap: no scores are computed (only an O(n_poles^2) degeneracy scan).
    """
    sampler = getattr(exact, "numeric_sampler", None)
    if sampler is None or exact.spin is None:
        return False
    # The regularized (epsilon-extrapolation) sampler has no single-c chiral
    # evaluators, so it cannot supply the residue/regular series this scorer
    # needs; it declares itself unavailable and the caller falls back to the
    # symbolic-numerator residue ranking. (Epsilon-extrapolating the scores does
    # NOT work at a singular c >= 1 with Kac collisions: the individual residues
    # of near-coincident probe poles diverge like 1/sqrt(eps) -- only the
    # combined multiple-pole Laurent residue is finite. So the residue
    # selectors at singular c keep the symbolic numerators; see the
    # skeleton-eligibility gates, which fall back to the full table for them.
    # The contribution selectors use principal-part scores instead.)
    if not getattr(sampler, "supports_residue_scoring", True):
        return False
    if getattr(exact, "requires_pole_strip", False):
        return False
    # Spinning ordered / parity-projected B carry independent h-side and
    # hbar-side externals, so their poles need the two-sided scorer
    # (_numeric_residue_scores_two_sided) -- a single h-side chiral series would
    # misrank the hbar-side poles. That path requires ell != 0: at ell = 0 the
    # h-side and hbar-side roots coincide into a double pole whose two-sided
    # residue is not yet handled numerically, so fall back to symbolic scoring.
    if not getattr(sampler, "_same_half", True) and int(exact.spin) == 0:
        return False
    if not exact.root_provenance or len(exact.root_provenance) != len(exact.denominator_roots):
        return False

    if int(exact.spin) != 0:
        # Degenerate-pole guard: numeric residue scoring is undefined if any two
        # poles coincide. Squared-distance scan on precomputed re/im components.
        tol = tolerance(guard=10)
        rc = [mp.mpc(r) for r in exact.denominator_roots]
        re = [c.real for c in rc]
        im = [c.imag for c in rc]
        tol2 = tol * tol
        n_roots = len(rc)
        for j in range(n_roots):
            rj, ij = re[j], im[j]
            for k in range(j + 1, n_roots):
                dre = rj - re[k]
                dim = ij - im[k]
                if dre * dre + dim * dim <= tol2:
                    return False
    return True


def _residue_side_machinery(sampler, n_top, *, externals, z_shift):
    """Per-chiral-half residue/regular machinery for two-sided pole scoring.

    Returns ``(rho_by_label, d_by_label, regular_fn)`` where ``rho_by_label``
    maps a Kac label ``(r, s)`` to its residue u-Taylor series (mpmath, indexed
    by the Taylor order), ``d_by_label`` to the pole weight ``d_{r,s}``, and
    ``regular_fn(h_eval)`` returns the regular chiral block u-series at internal
    weight ``h_eval``. ``externals`` / ``z_shift`` select the chiral half (the
    h-side or the hbar-side of a spinning block). Returns ``None`` if the chiral
    series or residues are unavailable.
    """
    series_data = _coordinate_series_from_sampler(
        sampler, n_top, externals=externals, z_shift=z_shift
    )
    if series_data is None:
        return None
    pair_table, H_table, arb_ctx = _build_H_context(
        series_data["q_order"], series_data["b"], series_data["h_externals"]
    )
    q_powers = _q_powers_truncated_numeric(
        series_data["q_series"], series_data["q_order"], n_top
    )
    flint_cache = _build_numeric_flint_cache(
        q_powers, series_data["a_series"], series_data["s_series"], n_top
    )
    residue_data = _chiral_residue_series(pair_table, H_table, series_data["q_order"])
    if not residue_data:
        return None
    exp_series = exp_h_times(series_data["s_series"], n_top)
    use_flint_series = True  # cleared if a q-series shape disagrees with the cache

    def _u_series_for(q_series_coeffs, h_eval):
        nonlocal use_flint_series
        if use_flint_series:
            val = _u_series_from_q_coeffs_numeric_flint(
                q_series_coeffs, h_eval, flint_cache=flint_cache, as_mpmath=True
            )
            if val is not None:
                return val
            use_flint_series = False
        return _u_series_from_q_coeffs_numeric(
            q_series_coeffs, h_eval, a_series=series_data["a_series"],
            q_powers=q_powers, exp_series=exp_series, flint_cache=flint_cache,
            n_terms=n_top,
        )

    rho_by_label = {}
    d_by_label = {}
    for label, payload in residue_data.items():
        d_by_label[label] = payload["d"]
        rho_by_label[label] = _u_series_for(payload["series"], payload["d"])

    regular_cache = {}

    def regular_fn(h_eval):
        key = (
            mp.mpc(h_eval)._mpc_ if isinstance(h_eval, mp.mpc)
            else ("mpf", mp.mpf(h_eval)._mpf_)
        )
        cached = regular_cache.get(key)
        if cached is not None:
            return cached
        h_series = _eval_H_at(
            h_eval, pair_table.pairs, H_table, series_data["q_order"],
            strip_symmetric=pair_table.is_strip_symmetric,
            canonical_indices=pair_table.canonical_indices,
            is_canonical=pair_table.is_canonical, twin_of=pair_table.twin_of,
            arb_ctx=arb_ctx,
        )
        if h_series is None:
            return None
        val = _u_series_for(h_series, h_eval)
        regular_cache[key] = val
        return val

    return rho_by_label, d_by_label, regular_fn


def _numeric_residue_scores_two_sided(exact, sampler, score_entries):
    """Two-sided residue pole scoring for spinning ordered / parity-projected B.

    The block factorizes as ``c_h[m] * c_hbar[n]`` (plus the swapped term for the
    projected basis), with the h-side and hbar-side carrying independent
    externals. A pole on the h-side (provenance ``left``, Delta = 2d - ell)
    takes its residue from the h-side chiral block and multiplies the *regular*
    hbar-side block at ``hbar = d - ell``; an hbar-side pole (``right``,
    Delta = 2d + ell) is the mirror image. Returns per-root scores aligned with
    ``exact.root_provenance``, or ``None`` to defer to symbolic scoring.

    Requires ``ell != 0`` (the availability gate enforces this; at ell = 0 the
    two sides collapse to a coincident double pole).
    """
    ell = int(exact.spin)
    if ell == 0:
        return None
    folded_coord = sampler.coordinate in ("ws", "xt", "qqb_ws")
    n_top = (
        exact.m_max + 2 * exact.n_max + (1 if (ell % 2) else 0) + 1
        if folded_coord
        else max(exact.m_max, exact.n_max) + 1
    )

    h_side = _residue_side_machinery(
        sampler, n_top, externals=sampler.h_externals, z_shift=sampler.z_shift
    )
    hbar_side = _residue_side_machinery(
        sampler, n_top, externals=sampler.hbar_externals, z_shift=sampler.zbar_shift
    )
    if h_side is None or hbar_side is None:
        return None
    rho_h, d_h, reg_h = h_side
    rho_hbar, d_hbar, reg_hbar = hbar_side

    swap = getattr(sampler, "swapped_scale", None)
    fold_marker = (
        sampler.parity_marker
        if (folded_coord and getattr(sampler, "projected", False))
        else ell
    )
    score_indices = _residue_score_indices_from_dims(
        exact.m_max, exact.n_max, score_entries, value_entry=_value_score_entry(exact)
    )

    # The single-sided scorer (and the symbolic scorer it matches) carry an
    # overall factor 2 on the residue table; replicate it so the two-sided
    # scores are on the same scale (the constant is uniform, so it does not
    # change the ranking, but keeping the scale identical avoids any surprise
    # for absolute-magnitude consumers).
    two = mp.mpf(2)

    def _score_nonfolded(nonfolded):
        if not folded_coord:
            return two * max(abs(nonfolded[m][n]) for m, n in score_indices)
        folded = _fold_parity(
            nonfolded, fold_marker, exact.m_max, exact.n_max, zero=mp.mpf(0)
        )
        return two * max(abs(folded[m][n]) for m, n in score_indices)

    score_map: dict = {}
    for prov in exact.root_provenance:
        key = _root_provenance_key(prov)
        if key is None:
            return None
        side, labels = key
        if side not in ("left", "right"):
            return None
        if key in score_map:
            continue
        label = tuple(labels[0])

        if side == "left":
            rho = rho_h.get(label)
            d_val = d_h.get(label)
            if rho is None or d_val is None:
                return None
            reg = reg_hbar(d_val - ell)
            if reg is None:
                return None
            # B residue at an h-side pole: rho_h (x) reg_hbar, plus the swapped
            # term for the projected basis.
            if swap is None:
                nonfolded = [[rho[m] * reg[n] for n in range(n_top)] for m in range(n_top)]
            else:
                nonfolded = [
                    [rho[m] * reg[n] + swap * (reg[m] * rho[n]) for n in range(n_top)]
                    for m in range(n_top)
                ]
        else:
            rho = rho_hbar.get(label)
            d_val = d_hbar.get(label)
            if rho is None or d_val is None:
                return None
            reg = reg_h(d_val + ell)
            if reg is None:
                return None
            if swap is None:
                nonfolded = [[reg[m] * rho[n] for n in range(n_top)] for m in range(n_top)]
            else:
                nonfolded = [
                    [reg[m] * rho[n] + swap * (rho[m] * reg[n]) for n in range(n_top)]
                    for m in range(n_top)
                ]
        score_map[key] = _score_nonfolded(nonfolded)

    return _ordered_scores_from_provenance(exact, score_map)


def _numeric_residue_scores(exact: DeltaDerivTable, *, score_entries: str = "kept"):
    if not numeric_residue_scoring_available(exact):
        return None
    sampler = exact.numeric_sampler

    # Spinning ordered / parity-projected B: independent h-side and hbar-side
    # externals need the two-sided scorer.
    if not getattr(sampler, "_same_half", True):
        return _numeric_residue_scores_two_sided(exact, sampler, score_entries)

    folded_coord = sampler.coordinate in ("ws", "xt", "qqb_ws")
    ell = int(exact.spin)
    # "value"/"contribution" scoring ranks on the (0, 0) block value, or the
    # first-nonzero entry when (0, 0) vanishes identically (parity '-'
    # projected_B on an unfolded coordinate). The (0, 0) case keeps
    # the fast leading-u-coefficient path; a higher fallback entry takes the
    # general path (it needs the (0, n0) u-Taylor coefficients).
    value_entry = _value_score_entry(exact) if score_entries == "value" else (0, 0)
    value_is_origin = value_entry == (0, 0)
    # The reversed-product coefficient: +1 for the scalar G (its spin-0 case is handled apart), the real
    # sign x phase of a parity projection with equal halves.  A same-half ordered B has none, and a complex
    # phase no real sign: both defer to the symbolic scorer.
    if isinstance(sampler, _NumericOrderedBlockSampler):
        swap = mp.mpc(sampler.swapped_scale) if sampler.projected else None
        if swap is None or abs(swap.imag) > _validation_tolerance(guard=5):
            return None
        swap_sign = 1 if swap.real > 0 else -1
        fold_marker = sampler.parity_marker if folded_coord else 0
    else:
        swap_sign = 1 if scalar_G_swap_coefficient(ell) >= 0 else -1
        fold_marker = fold_parity_marker(ell=ell)
    n_top = (
        folded_source_order(exact.m_max, exact.n_max, fold_marker) + 1
        if folded_coord
        else max(exact.m_max, exact.n_max) + 1
    )

    series_data = _coordinate_series_from_sampler(sampler, n_top)
    if series_data is None:
        return None

    pair_table, H_table, arb_ctx = _build_H_context(
        series_data["q_order"],
        series_data["b"],
        series_data["h_externals"],
    )
    q_powers = _q_powers_truncated_numeric(
        series_data["q_series"],
        series_data["q_order"],
        n_top,
    )
    flint_cache = _build_numeric_flint_cache(
        q_powers,
        series_data["a_series"],
        series_data["s_series"],
        n_top,
    )

    # Fast "value"/"contribution" ranking: both the residue leading term rho0_j
    # and the regular leading term reg0(h) collapse to one precomputed vector
    #   c_idx = a0 * (W[0] . M)[idx] * K_idx,   rho0_j = c_j,
    #   reg0(h) = a0 + sum_idx c_idx / (h - d_idx).
    # This skips the per-pole residue u-series and *all* regular-block H-series
    # evaluations (the dominant cost of the general path). Bit-identical to it.
    if score_entries == "value" and value_is_origin:
        fast = _numeric_value_scores_fast(
            exact, pair_table, arb_ctx, flint_cache, ell, swap_sign
        )
        if fast is not None:
            return fast

    residue_data = _chiral_residue_series(pair_table, H_table, series_data["q_order"])
    if not residue_data:
        return None
    exp_series = exp_h_times(series_data["s_series"], n_top)

    # The (0, 0)-value ranking consults only the (0, 0) entry, so it needs only
    # the leading (m=0) u-Taylor coefficient of each residue/regular series --
    # a length-q_order dot product, avoiding the full (n_terms x q_order) matmul.
    # A higher first-nonzero fallback entry needs the (0, n0) coefficients, so it
    # takes the full-series path.
    leading_only = (score_entries == "value") and value_is_origin
    use_flint_series = True  # cleared if a q-series shape disagrees with the cache

    def _u_series_for(q_series_coeffs, h_eval):
        nonlocal use_flint_series
        if use_flint_series:
            if leading_only:
                t0 = _u_series_leading_from_q_coeffs_flint(q_series_coeffs, flint_cache)
                if t0 is not None:
                    return [t0]  # 1-element so series[0] indexing is unchanged
                use_flint_series = False
            else:
                val = _u_series_from_q_coeffs_numeric_flint(
                    q_series_coeffs, h_eval, flint_cache=flint_cache, as_mpmath=False
                )
                if val is not None:
                    return val
                use_flint_series = False
        return _u_series_from_q_coeffs_numeric(
            q_series_coeffs,
            h_eval,
            a_series=series_data["a_series"],
            q_powers=q_powers,
            exp_series=exp_series,
            flint_cache=flint_cache,
            n_terms=n_top,
        )

    rho_by_label = {}
    d_by_label = {}
    for label, payload in residue_data.items():
        d_val = payload["d"]
        rho_by_label[label] = _u_series_for(payload["series"], d_val)
        d_by_label[label] = d_val

    regular_cache = {}

    pending_h = {}
    if ell != 0:
        for prov in exact.root_provenance:
            prov_key = _root_provenance_key(prov)
            if prov_key is None:
                return None
            side, labels = prov_key
            if side not in ("left", "right"):
                return None
            label = tuple(labels[0])
            d_val = d_by_label.get(label)
            if d_val is None:
                return None
            h_eval = d_val - ell if side == "left" else d_val + ell
            key = (
                mp.mpc(h_eval)._mpc_
                if isinstance(h_eval, mp.mpc)
                else ("mpf", mp.mpf(h_eval)._mpf_)
            )
            pending_h.setdefault(key, h_eval)

    if pending_h:
        h_values = list(pending_h.values())
        h_series_list = _eval_H_at_many(
            h_values,
            pair_table.pairs,
            H_table,
            series_data["q_order"],
            strip_symmetric=pair_table.is_strip_symmetric,
            canonical_indices=pair_table.canonical_indices,
            is_canonical=pair_table.is_canonical,
            twin_of=pair_table.twin_of,
            arb_ctx=arb_ctx,
        )
        for h_eval, h_series in zip(h_values, h_series_list):
            key = (
                mp.mpc(h_eval)._mpc_
                if isinstance(h_eval, mp.mpc)
                else ("mpf", mp.mpf(h_eval)._mpf_)
            )
            if h_series is None:
                continue
            regular_cache[key] = _u_series_for(h_series, h_eval)

    def _regular_series(h_eval):
        key = mp.mpc(h_eval)._mpc_ if isinstance(h_eval, mp.mpc) else ("mpf", mp.mpf(h_eval)._mpf_)
        cached = regular_cache.get(key)
        if cached is not None:
            return cached
        h_series = _eval_H_at(
            h_eval,
            pair_table.pairs,
            H_table,
            series_data["q_order"],
            strip_symmetric=pair_table.is_strip_symmetric,
            canonical_indices=pair_table.canonical_indices,
            is_canonical=pair_table.is_canonical,
            twin_of=pair_table.twin_of,
            arb_ctx=arb_ctx,
        )
        val = _u_series_for(h_series, h_eval)
        regular_cache[key] = val
        return val

    score_indices = _residue_score_indices_from_dims(
        exact.m_max, exact.n_max, score_entries, value_entry=value_entry
    )
    sign = mp.mpf(swap_sign)
    # The score_value_only shortcut reads index 0 directly (the (0, 0) entry);
    # it only applies when the value entry IS (0, 0). A higher fallback entry
    # goes through the general score_indices = [(0, n0)] path below.
    score_value_only = (score_entries == "value") and value_is_origin
    sign_int = 1 if sign > 0 else -1
    # the spin-0 double pole of (1 + sigma) c_h c_h (see _numeric_value_scores_fast)
    spin0_factor = 4 * (1 + scalar_G_swap_coefficient(0))

    def _series_to_mpmath(values):
        if not use_flint_series:
            return values
        return [_acb_to_mpc(v, flint_cache["ndigits"]) for v in values]

    def _best_nonfolded_from_flint(rho, reg=None):
        best = None
        for m, n in score_indices:
            if reg is None:
                val = abs(spin0_factor * rho[m] * rho[n])
            else:
                val = abs(2 * (rho[m] * reg[n] + sign_int * reg[m] * rho[n]))
            if best is None or val > best:
                best = val
        if best is None:
            return mp.mpf(0)
        return mp.mpf(best.str(flint_cache["ndigits"], radius=False))

    def _score_from_nonfolded(nonfolded):
        if not folded_coord:
            return max(abs(nonfolded[m][n]) for m, n in score_indices)
        folded = _fold_parity(nonfolded, fold_marker, exact.m_max, exact.n_max, zero=mp.mpf(0))
        return max(abs(folded[m][n]) for m, n in score_indices)

    scores = []
    for prov in exact.root_provenance:
        prov_key = _root_provenance_key(prov)
        if prov_key is None:
            return None
        side, labels = prov_key
        if side not in ("left", "right"):
            return None
        label = tuple(labels[0])
        if label not in rho_by_label:
            return None

        rho = rho_by_label[label]
        d_val = d_by_label[label]
        if ell == 0:
            if use_flint_series and not folded_coord:
                if score_value_only:
                    val = abs(spin0_factor * rho[0] * rho[0])
                    scores.append(mp.mpf(val.str(flint_cache["ndigits"], radius=False)))
                else:
                    scores.append(_best_nonfolded_from_flint(rho))
                continue
            rho_mp = _series_to_mpmath(rho)
            if score_value_only:
                scores.append(abs(mp.mpf(spin0_factor) * rho_mp[0] * rho_mp[0]))
                continue
            nonfolded = [[mp.mpf(spin0_factor) * rho_mp[m] * rho_mp[n] for n in range(n_top)] for m in range(n_top)]
            scores.append(_score_from_nonfolded(nonfolded))
            continue

        if side == "left":
            reg = _regular_series(d_val - ell)
            if use_flint_series and not folded_coord:
                if score_value_only:
                    val = abs(2 * (rho[0] * reg[0] + sign_int * reg[0] * rho[0]))
                    scores.append(mp.mpf(val.str(flint_cache["ndigits"], radius=False)))
                else:
                    scores.append(_best_nonfolded_from_flint(rho, reg))
                continue
            rho_mp = _series_to_mpmath(rho)
            reg_mp = _series_to_mpmath(reg)
            if score_value_only:
                scores.append(abs(mp.mpf(2) * (rho_mp[0] * reg_mp[0] + sign * reg_mp[0] * rho_mp[0])))
                continue
            nonfolded = [
                [mp.mpf(2) * (rho_mp[m] * reg_mp[n] + sign * reg_mp[m] * rho_mp[n]) for n in range(n_top)]
                for m in range(n_top)
            ]
        else:
            reg = _regular_series(d_val + ell)
            if use_flint_series and not folded_coord:
                if score_value_only:
                    val = abs(2 * (reg[0] * rho[0] + sign_int * rho[0] * reg[0]))
                    scores.append(mp.mpf(val.str(flint_cache["ndigits"], radius=False)))
                else:
                    scores.append(_best_nonfolded_from_flint(rho, reg))
                continue
            rho_mp = _series_to_mpmath(rho)
            reg_mp = _series_to_mpmath(reg)
            if score_value_only:
                scores.append(abs(mp.mpf(2) * (reg_mp[0] * rho_mp[0] + sign * rho_mp[0] * reg_mp[0])))
                continue
            nonfolded = [
                [mp.mpf(2) * (reg_mp[m] * rho_mp[n] + sign * rho_mp[m] * reg_mp[n]) for n in range(n_top)]
                for m in range(n_top)
            ]
        scores.append(_score_from_nonfolded(nonfolded))

    if len(scores) != len(exact.denominator_roots):
        return None
    return scores


def _residue_scores(
    exact: DeltaDerivTable,
    *,
    prepared=None,
    score_entries: str = "kept",
    pole_ranking: str = "symbolic",
) -> list:
    """Per-Delta-pole strength score for residue-magnitude pole selection.

    For each pole p the score is the leading Laurent-coefficient magnitude of
    `numerator(Delta) / denominator(Delta)` -- i.e. the residue of the block
    derivative with the universal `exp_base**Delta` factor stripped (that
    factor cancels in `block / prefactor`, so it must not bias the ranking).
    Duplicates of p are excluded from the product, so the score is well-defined
    for the ell = 0 double poles too; it is then the |1/(Delta-p)^2| coefficient.
    The score is aggregated as the max over the (m, n) derivative entries that
    survive downstream truncation (the ``m + n <= max(m_max, n_max)`` triangle;
    see :func:`_residue_scored_numerators`).

    Duplicate detection uses a working-precision tolerance rather than exact
    equality. Two cases of "effective duplicate" both need handling: the
    bit-exact (Delta + ell, Delta - ell) collisions at ell = 0, and the
    near-equal pairs that come out of epsilon-extrapolation at recursion-
    singular c (where Kac weights coincide in the limit and the extrapolated
    values agree to base precision but not bit-exactly). The tolerance is
    intentionally well below the physical Kac-pole spacing for nondegenerate
    c, so genuinely distinct poles are never confused with each other.
    """
    if pole_ranking == "numeric":
        numeric_scores = _numeric_residue_scores(exact, score_entries=score_entries)
        if numeric_scores is not None:
            return numeric_scores
        pole_ranking = "symbolic"

    if pole_ranking != "symbolic":
        raise ValueError(
            f"pole_ranking must be 'numeric' or 'symbolic'; got {pole_ranking!r}"
        )

    if exact.numerators is None:
        # Skeleton (degree-only) table: the symbolic residue scorer needs the
        # numerator coefficients, which were deliberately not built. The caller
        # should only reach here when numeric residue scoring is unavailable
        # (e.g. degenerate poles, missing provenance); build a full table.
        raise ValueError(
            "symbolic residue ranking requires a full numerator table, but a "
            "degree-only skeleton table was supplied (numeric pole ranking "
            "declined). Rebuild the exact table without skeleton mode."
        )

    roots = [mp.mpc(r) for r in exact.denominator_roots]
    leading_coeff = exact.denominator.coeffs[-1]
    tol = tolerance(guard=10)

    # Keep the per-pole denominator product, the duplicate-pole detection, and
    # the numerator evaluations all in flint acb/arb space. `prepared` lets
    # the multi-table aggregator share the
    # (coordinate-independent) denominator factors across tables.
    from flint import arb, ctx as flint_ctx

    prec_bits = int(mp.mp.dps * 3.33) + 20
    flint_ctx.prec = prec_bits
    ndigits = int(mp.mp.dps) + 5

    if prepared is None:
        poles_acb, factors = _residue_denom_factors_acb(
            roots, leading_coeff, tol, prec_bits
        )
    else:
        poles_acb, factors = prepared

    num_polys = [
        num._get_acb() for num in _residue_scored_numerators(exact, score_entries)
    ]
    scores = []
    for x_acb, denom_acb in zip(poles_acb, factors):
        best = arb(0)
        for num_poly in num_polys:
            score = abs(num_poly(x_acb) / denom_acb)
            if score > best:
                best = score
        scores.append(mp.mpf(best.str(ndigits, radius=False)))
    return scores


def _group_roots_by_location(roots, tol):
    """Group an mpmath root list into distinct pole locations with multiplicity.

    Returns a list of ``(location, [indices])`` in first-appearance order; every
    index of a coincident cluster (within ``tol``) shares one group. Used by the
    full-principal-part contribution scorer to read each pole's order ``m``."""
    tagged = []
    for idx, root in enumerate(roots):
        root = mp.mpc(root)
        tagged.append((float(root.real), float(root.imag), idx, root))
    if not tagged:
        return []
    tagged.sort(key=lambda item: item[:2])

    # A float distance above the float rounding of the exact one (plus tol) rules a pair out without the
    # multiprecision test, so only near-coincident pairs pay it; the clusters are those of the exact test alone.
    tol_float = float(tol)
    clusters = []
    anchor_re, anchor_im, anchor_idx, anchor_root = tagged[0]
    anchor_c = complex(anchor_re, anchor_im)
    current = [(anchor_idx, anchor_root)]
    for re_f, im_f, idx, root in tagged[1:]:
        root_c = complex(re_f, im_f)
        if (abs(root_c - anchor_c) <= 1e-12 * (abs(root_c) + abs(anchor_c)) + 2 * tol_float
                and abs(root - anchor_root) <= tol):
            current.append((idx, root))
            continue
        clusters.append(current)
        anchor_idx, anchor_root, anchor_c = idx, root, root_c
        current = [(idx, root)]
    clusters.append(current)

    groups = []
    for cluster in sorted(clusters, key=lambda pts: min(idx for idx, _ in pts)):
        idxs = sorted(idx for idx, _ in cluster)
        groups.append((roots[idxs[0]], idxs))
    return groups


def _nearest_group_gaps(locations):
    """Distance from each group location to its nearest *other* group location.

    Sets the contour radius in the principal-part scorer (r = radius_fraction *
    gap), so it must be the true 2d nearest neighbor over the complex plane: if
    the gap is overestimated, the contour swallows a neighboring pole and the
    DFT extraction is corrupted. A real-axis-sorted adjacency sweep is only valid
    in 1d and silently overestimates here (e.g. a near-real pole whose closest
    neighbor sits off the real axis), so we take the exact all-pairs minimum.
    Pole counts are small (tens), so the O(n^2) cost is negligible."""
    pts = [mp.mpc(loc) for loc in locations]
    n = len(pts)
    best = [None] * n
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(pts[i] - pts[j])
            if best[i] is None or d < best[i]:
                best[i] = d
            if best[j] is None or d < best[j]:
                best[j] = d
    return {i: (best[i] if best[i] is not None else mp.mpf(1)) for i in range(n)}


def _contribution_scores(exact: DeltaDerivTable, *, delta_eval, score_entries="kept"):
    """Full principal-part contribution at ``delta_eval`` per Delta-pole.

    The ``contribution`` score: for a pole ``p`` of multiplicity
    ``m`` the score is ``|sum_{j<m} a_j d^j| / |d|^m`` with ``d = delta_eval - p``
    and ``a_0..a_{m-1}`` the first ``m`` Taylor coefficients of
    ``numerator_e(Delta) / cofactor(Delta)`` at ``p`` (``cofactor = leading *
    prod_{q != p}(Delta - q)``), maxed over the scored entries ``e``. This equals
    the magnitude of the pole's whole principal part evaluated at ``delta_eval``
    -- every Laurent coefficient, not just the leading one. For a simple pole it
    reduces to ``|R| / |d|``, bit-identical to the legacy ``contribution_leading``
    score.

    Returns one score per entry of ``exact.denominator_roots`` (every copy of a
    coincident pole gets the same score). Requires the symbolic numerators.
    """
    if exact.numerators is None:
        raise ValueError(
            "contribution scoring requires a full numerator table; a "
            "degree-only skeleton was supplied (skeletons are scored by the numeric principal-part path)."
        )
    if score_entries == "value":
        # Single low-order entry, with the first-nonzero fallback when the
        # undifferentiated (0,0) vanishes (odd-parity unfolded block).
        mi, ni = _value_score_entry(exact)
        scored_nums = [exact.numerators[mi][ni]]
    else:
        scored_nums = list(_residue_scored_numerators(exact, score_entries))
    # An identically-zero numerator entry contributes nothing to the score, and
    # dividing the resulting zero ``acb_series`` by the cofactor crashes flint
    # (0.8.0 raises a spurious AttributeError on a zero numerator). This occurs
    # e.g. for a parity-odd projected_B basis, whose undifferentiated (0, 0)
    # entry vanishes identically -- so skip such entries entirely.
    num_is_zero = [
        (not num.coeffs) or all(mp.mpc(c) == 0 for c in num.coeffs)
        for num in scored_nums
    ]
    from flint import acb
    from .polynomial import _mpf_to_acb

    ndigits = int(mp.mp.dps) + 5
    roots = [mp.mpc(r) for r in exact.denominator_roots]
    deval = mp.mpc(delta_eval)
    scores = [mp.mpf(0)] * len(roots)
    for location, idxs, rows in _symbolic_principal_parts(exact, roots, scored_nums, num_is_zero):
        m = len(idxs)
        d = deval - location
        dist = abs(d)
        # A pole at delta_eval has a divergent principal part. Assign infinite
        # score so it is retained ahead of every finite contribution.
        if not (dist > 1e-12):
            for idx in idxs:
                scores[idx] = mp.inf
            continue
        d_acb = _mpf_to_acb(d)
        best = mp.mpf(0)
        for row in rows:
            if row is None:
                continue  # zero numerator -> zero contribution
            # sum_j a_j d^j over the head a_0 .. a_{m-1} (a_j = g_{-(m-j)})
            val = acb(0)
            dpow = acb(1)
            for j in range(m):
                val = val + row[m - 1 - j] * dpow
                dpow = dpow * d_acb
            s_arb = abs(val) / dist ** m
            score = mp.mpf(s_arb.str(ndigits, radius=False))
            if score > best:
                best = score
        for idx in idxs:
            scores[idx] = best
    return scores


def _symbolic_principal_parts(exact, roots, nums, num_is_zero=None):
    """Exact principal parts of ``nums[e] / denominator`` at each location group of ``roots``, from a symbolic table.

    For a pole ``p`` of multiplicity ``m`` the head ``a_0 .. a_{m-1}`` of ``numerator(p + t) / cofactor(p + t)``
    (``cofactor = leading * prod_{q not at p}(Delta - q)`` over the table's roots) gives ``g_{-a} = a_{m-a}``: the
    block is ``exp_base^Delta (sum_a g_{-a} (Delta - p)^-a + regular)``.  Returns ``[(p, root_indices, rows)]`` with
    ``rows[e] = (g_{-1}, ..., g_{-m})`` as flint ``acb`` (``None`` for an identically-zero numerator, which flint
    cannot divide).
    """
    from flint import acb, acb_series, ctx as flint_ctx

    from .polynomial import _mpf_to_acb

    flint_ctx.prec = int(mp.mp.dps * 3.33) + 20
    tol = tolerance(guard=10)
    table_roots = [mp.mpc(r) for r in exact.denominator_roots]
    leading_acb = _mpf_to_acb(exact.denominator.coeffs[-1])
    num_polys = [[_mpf_to_acb(c) for c in num.coeffs] for num in nums]
    if num_is_zero is None:
        num_is_zero = [(not num.coeffs) or all(mp.mpc(c) == 0 for c in num.coeffs) for num in nums]
    parts = []
    old_cap = flint_ctx.cap
    try:
        for location, idxs in _group_roots_by_location([mp.mpc(r) for r in roots], tol):
            m = len(idxs)
            flint_ctx.cap = m
            p_acb = _mpf_to_acb(location)
            pt = acb_series([p_acb, acb(1)])  # p + t
            # cofactor(p + t) = leading * prod_{q not in the group} ((p - q) + t), over every root of the table
            cof = acb_series([leading_acb])
            for r in table_roots:
                if abs(r - location) <= tol * max(1, abs(location)):
                    continue
                cof = cof * acb_series([p_acb - _mpf_to_acb(r), acb(1)])
            rows = []
            for coeffs, is_zero in zip(num_polys, num_is_zero):
                if is_zero:
                    rows.append(None)
                    continue
                acc = acb_series([acb(0)])  # numerator(p + t) by Horner in t
                for c in reversed(coeffs):
                    acc = acc * pt + acb_series([c])
                head = (acc / cof).coeffs()
                head = [head[j] if j < len(head) else acb(0) for j in range(m)]
                rows.append(tuple(head[m - a] for a in range(1, m + 1)))
            parts.append((location, idxs, rows))
    finally:
        flint_ctx.cap = old_cap
    return parts


class _ExactTableValues:
    """A symbolic table's own values at given Deltas, in the numeric-sampler call form."""

    def __init__(self, exact):
        self.exact = exact

    @_quiet_conditioning
    def __call__(self, deltas, entries=None):
        return _evaluate_table_flint_many(self.exact, list(deltas), entries=entries)


# Coarse->fine pre-filter knobs (see principal_part_contribution_scores).
# The coarse pass ranks all poles to pick candidates; the fine pass scores only
# those over every entry. The coarse pass keeps the *full* n_extra -- shrinking
# it instead catastrophically misranks far-complex poles, where stripping
# exp_base^Δ forces a deep cancellation that needs the full DFT to resolve (a
# small-M coarse pass overestimates them by ~20 orders of magnitude). The saving
# comes purely from scoring a fixed-size entry subset (sampler and recombination
# cost both scale with the entry count). The window is sized well above the kept
# count so the selected set is unchanged (validated by the cross-(c,Lambda,dps)
# selection stress test).
_PREFILTER_CAND_MULT = 4     # keep top 4 * n_select finite candidates
_PREFILTER_MIN_CAND = 32     # ...but never fewer than this
# If the coarse candidate window is already this fraction of all groups, the
# coarse+fine split has little room to save work; do one full pass directly.
_PREFILTER_FULLPASS_RATIO = mp.mpf("0.80")

# Cross-call contour geometry cache for contribution scoring. Keyed by
# precision + group geometry + (delta_eval, exp_base, radius_fraction, n_extra).
_CONTRIB_GEOM_CACHE_MAXSIZE = 4096
_CONTRIB_GEOM_CACHE = {}


def _mpf_cache_key(x):
    return mp.mpf(x)._mpf_


def _mpc_cache_key(z):
    zz = mp.mpc(z)
    return (mp.mpf(zz.real)._mpf_, mp.mpf(zz.imag)._mpf_)


def _contour_params(dps=None):
    """Precision-aware contour radius / DFT oversampling for the scorer.

    The principal-part DFT (see :func:`principal_part_contribution_scores`) has a
    relative ranking accuracy governed by two competing terms, both verified
    numerically against a synthetic rational function:

        relerr  ~  radius_fraction^(m + n_extra)        (aliasing)
                +  radius_fraction^(-(m-1)) * 10^(-dps)  (roundoff)

    where ``m`` is the pole order. Aliasing shrinks with a smaller radius or more
    DFT points; roundoff (from sampling the block where it blows up as
    ``1/r^m``) *grows* as the radius shrinks, scaled by working precision -- so
    ``radius_fraction`` is a two-sided optimum and ``n_extra`` is one-sided.

    Use ``radius_fraction = 0.03`` and choose ``n_extra`` to target relative
    accuracy ``10^(-max(12, dps//2))`` for a simple pole, subject to
    ``6 <= n_extra <= 12``. The cap limits the cost of ranking: each group
    needs ``m + n_extra`` block samples. At the cap, the simple-pole aliasing
    estimate is ``0.03^13``, about ``1.6e-20``.
    """
    if dps is None:
        dps = mp.mp.dps
    radius_fraction = mp.mpf("0.03")
    target_digits = max(12, dps // 2)
    # rf^(1 + n_extra) <= 10^-target_digits  ->  1 + n_extra >= target/|log10 rf|
    per_point = -mp.log10(radius_fraction)  # ~1.523 digits of aliasing per order
    n_extra = int(mp.ceil(target_digits / per_point)) - 1
    n_extra = max(6, min(12, n_extra))
    return radius_fraction, n_extra


def _numeric_contribution_scores(
    exact: DeltaDerivTable, *, delta_eval, score_entries="kept",
    n_extra: int = None, radius_fraction=None, n_select=None,
):
    """Full principal-part contribution from a numeric block sampler.

    The numeric analogue of :func:`_contribution_scores`, computed from block
    *values* instead of the symbolic numerators -- so it serves the
    ``--sample-via numeric`` skeleton tables, the global / large-spin samplers,
    and the fallback for the regularized singular-c sampler (whose exact
    merged-Laurent principal parts are preferred), where the collapsed
    multiple pole's principal part is finite even though the individual probe
    residues diverge.

    For a pole ``p`` of order ``m``, ``h_p(Δ) = (Δ-p)^m · block(Δ)`` is analytic
    at ``p`` and its first ``m`` Taylor coefficients are the principal-part
    coefficients. They are extracted by a DFT over ``M = m + n_extra`` points on
    a circle of radius ``radius_fraction · (nearest other pole)`` around ``p``;
    the contribution at ``Δ_eval`` is ``|Σ_j a_j d^j| / |d|^m``. Other poles and
    the regular part enter ``h_p`` only at Taylor order ``≥ m``, so they do not
    contaminate the head except through DFT aliasing, which is
    ``(radius_fraction)^M``-suppressed. Returns ``None`` when no sampler is
    attached (caller falls back to the symbolic scorer)."""
    sampler = getattr(exact, "numeric_sampler", None)
    if sampler is None:
        return None
    # Use exactly the entry set the symbolic scorer maxes over (the truncation
    # triangle for "kept", a single low-order entry for "value") so the two
    # paths agree -- including the first-nonzero fallback when (0,0) vanishes.
    if score_entries == "value":
        entries = [_value_score_entry(exact)]
    else:
        entries = _residue_score_indices_from_dims(
            exact.m_max, exact.n_max, score_entries
        )
    roots = [mp.mpc(r) for r in exact.denominator_roots]

    # Generic-c analytic fast path: build each pole's principal part in closed
    # form from the chiral residue/regular series (exact, ~one evaluation per
    # pole) instead of the contour/DFT. Skipped for tables still flagged
    # `requires_pole_strip`, where coincident-pole residues are finite only in
    # the limit (the regularized singular-c samplers instead supply exact
    # merged-Laurent principal parts, `_analytic_principal_parts`). The scorer itself returns None
    # for any other unsupported shape (folded/off-diagonal coordinate,
    # unsupported context mode), so we always have the contour as the fallback.
    analytic = getattr(sampler, "_analytic_contribution_scores", None)
    if analytic is not None and not getattr(exact, "requires_pole_strip", False):
        scores = analytic(roots, delta_eval, entries, exact.exp_base)
        if scores is not None:
            return scores

    return principal_part_contribution_scores(
        roots,
        lambda pts, ent: sampler(pts, entries=ent),
        delta_eval=delta_eval,
        entries=entries,
        exp_base=exact.exp_base,
        n_extra=n_extra,
        radius_fraction=radius_fraction,
        n_select=n_select,
    )


def _coarse_entry_subset(entries, cap=12):
    """A fixed-size, spread-out subset of scored entries for the coarse filter.

    The contribution score maxes over the kept-triangle entries, which is
    ``O(Lambda^2)`` -- the factor that makes the full scorer ``O(Lambda^4)``.
    The coarse pre-filter only has to *rank* poles well enough to choose
    candidates, so it scores over a constant-size, geometrically spread subset
    (the corners and a diagonal sweep of the (m, n) grid) that captures the
    dynamic range across entries. Keeping it fixed-size is what turns the
    coarse pass into ``O(Lambda^2)``. The fine pass still uses all entries."""
    if len(entries) <= cap:
        return list(entries)
    ms = [e[0] for e in entries]; ns = [e[1] for e in entries]
    m_hi, n_hi = max(ms), max(ns)
    want = set()
    # corners / axis extremes of the kept region
    for target in ((0, 0), (m_hi, 0), (0, n_hi), (m_hi, n_hi)):
        want.add(min(entries, key=lambda e, t=target: (e[0]-t[0])**2 + (e[1]-t[1])**2))
    # a diagonal sweep to fill in the interior
    remaining = cap - len(want)
    if remaining > 0:
        diag = sorted(entries, key=lambda e: (e[0] + e[1]))
        step = max(1, len(diag) // (remaining + 1))
        for e in diag[::step]:
            if len(want) >= cap:
                break
            want.add(e)
    return [e for e in entries if e in want]


def principal_part_contribution_scores(
    roots, eval_block, *, delta_eval, entries, exp_base,
    n_extra: int = None, radius_fraction=None, n_select=None,
):
    """Full-principal-part contribution scores from a block evaluator.

    The shared contour/DFT core behind the numeric ``contribution``
    scoring, used by the 2d sampler path, the global / large-spin samplers, and
    the regularized singular-c sampler. ``eval_block(points, entries)`` must
    return ``(grids, is_arb)`` -- the block ``[m][n]`` tables at the given
    (possibly complex) points, restricted to ``entries``. ``roots`` is the full
    pole multiset (coincident copies repeated); ``exp_base`` is the universal
    prefactor base (stripped before extraction). Returns one score per root.

    For a pole ``p`` of order ``m``, ``h_p(Δ) = (Δ-p)^m · g(Δ)`` with
    ``g = block / exp_base^Δ`` is analytic at ``p``; its first ``m`` Taylor
    coefficients (the principal-part coefficients) are read off by a DFT over
    ``M = m + n_extra`` points on a circle of radius
    ``radius_fraction · (nearest other pole)``. The contribution at
    ``delta_eval`` is ``|Σ_j a_j d^j| / |d|^m``.

    ``n_select`` (the number of poles that will ultimately be kept) enables a
    coarse->fine pre-filter: every pole is first scored cheaply (small ``n_extra``
    over a fixed-size entry subset), then only the top candidates are scored at
    full fidelity. This drops the dominant cost from ``O(Lambda^4)`` toward
    ``O(Lambda^2)`` -- essential for high Lambda -- with the candidate window
    sized generously so the selected set is provably unchanged. ``n_select=None``
    scores every pole at full fidelity (the exact single-pass path)."""
    if n_extra is None or radius_fraction is None:
        rf_default, ne_default = _contour_params()
        if radius_fraction is None:
            radius_fraction = rf_default
        if n_extra is None:
            n_extra = ne_default
    roots = [mp.mpc(r) for r in roots]
    if not roots:
        return []
    tol = tolerance(guard=10)
    groups = _group_roots_by_location(roots, tol)
    locations = [loc for loc, _ in groups]
    nearest_gap = _nearest_group_gaps(locations)
    deval = mp.mpc(delta_eval)
    log_base = mp.log(mp.mpc(exp_base))
    tol_d = mp.mpf("1e-12")
    exp_base_key = _mpc_cache_key(exp_base)
    geom_cache = {}
    omega_cache = {}
    entry_index_cache = {}

    from flint import acb, acb_mat, ctx as flint_ctx

    def _score_group_indices(group_idxs, nev, score_entries):
        """Score the given groups via the contour DFT; return {gidx: score}.

        Builds one batched ``eval_block`` call over all the groups' contour
        points, strips ``exp_base^pt``, and recombines each group's per-entry
        principal part ``PP_e = Σ_k g_{k,e} u_k`` (the DFT, Taylor evaluation,
        ``1/d^m`` and prefactor all folded into the per-point weight ``u_k``).
        ``flint`` recombination is a ``(1×M)·(M×n_e)`` acb matvec; the mpmath
        loop is the mpc-sampler fallback."""
        entries_key = tuple((int(m), int(n)) for (m, n) in score_entries)
        idx_vectors = entry_index_cache.get(entries_key)
        if idx_vectors is None:
            em_idx = [mn[0] for mn in entries_key]
            en_idx = [mn[1] for mn in entries_key]
            idx_vectors = (entries_key, em_idx, en_idx)
            entry_index_cache[entries_key] = idx_vectors
        entry_pairs, em_idx, en_idx = idx_vectors
        n_entries = len(entry_pairs)

        plan = []  # (gidx, idxs, start_or_None, weights_or_None)
        all_pts = []
        for gidx in group_idxs:
            loc, idxs = groups[gidx]
            m = len(idxs)
            gap = nearest_gap.get(gidx, mp.mpf(1))
            cache_key = (
                mp.mp.prec,
                int(nev),
                _mpf_cache_key(radius_fraction),
                _mpc_cache_key(deval),
                exp_base_key,
                _mpc_cache_key(loc),
                int(m),
                _mpf_cache_key(gap),
            )
            cached_geom = geom_cache.get(cache_key)
            if cached_geom is None:
                cached_geom = _CONTRIB_GEOM_CACHE.get(cache_key)
            if cached_geom is None:
                d = deval - loc
                if abs(d) <= tol_d:
                    # This group's contribution diverges at delta_eval, so its
                    # score is +inf and no contour samples are needed.
                    cached_geom = ((), None)
                else:
                    r = radius_fraction * gap
                    if r <= 0:
                        r = mp.mpf(10) ** (-(mp.mp.dps // 4))
                    M = m + nev
                    omegas = omega_cache.get(M)
                    if omegas is None:
                        omegas = [mp.exp(2j * mp.pi * k / M) for k in range(M)]
                        omega_cache[M] = omegas
                    pts = [loc + r * w for w in omegas]
                    dr = d / r
                    scale = (r ** m) / (M * d ** m)
                    u = []
                    for k, wk in enumerate(omegas):
                        x = dr / wk
                        # Closed-form geometric sum with a near-1 guard.
                        if abs(x - 1) <= mp.mpf(10) ** (-(mp.mp.dps // 3)):
                            gsum = mp.mpf(m)
                        else:
                            gsum = (1 - x ** m) / (1 - x)
                        u.append(mp.exp(-pts[k] * log_base) * scale * (wk ** m) * gsum)
                    cached_geom = (tuple(pts), tuple(u))
                # Evict the single oldest entry (insertion-ordered dict) rather
                # than clearing the whole cache, so a bounded cache keeps its
                # warm geometry and each mutation stays a localized pop/set.
                if len(_CONTRIB_GEOM_CACHE) >= _CONTRIB_GEOM_CACHE_MAXSIZE:
                    try:
                        _CONTRIB_GEOM_CACHE.pop(next(iter(_CONTRIB_GEOM_CACHE)))
                    except (StopIteration, KeyError):
                        pass
                _CONTRIB_GEOM_CACHE[cache_key] = cached_geom
                geom_cache[cache_key] = cached_geom
            else:
                geom_cache[cache_key] = cached_geom
            pts, u = cached_geom
            if u is None:
                start = None
            else:
                start = len(all_pts)
                all_pts.extend(pts)
            plan.append((gidx, idxs, start, u))

        out = {}
        for gidx, _idxs, _start, u in plan:
            if u is None:
                out[gidx] = mp.inf
        if not all_pts:
            return out

        grids, is_arb = eval_block(all_pts, score_entries)

        sample_val = None
        if all_pts and score_entries:
            em0, en0 = score_entries[0]
            sample_val = grids[0][em0][en0]
        sample_mod = type(sample_val).__module__ if sample_val is not None else ""
        sample_is_flint = sample_mod.startswith("flint.types.")
        use_flint = is_arb or sample_is_flint
        if use_flint:
            flint_ctx.prec = int(mp.mp.dps * 3.33) + 20
            ndig = mp.mp.dps + 5

            def _to_acb(v):
                mod = type(v).__module__
                if mod.startswith("flint.types."):
                    return v if hasattr(v, "imag") else acb(v)
                return _mpf_to_acb(v)

            for gidx, idxs, start, u in plan:
                if u is None:
                    continue
                M = len(u)
                if n_entries == 1:
                    m0, n0 = entry_pairs[0]
                    acc = acb(0)
                    for k in range(M):
                        acc += acb(u[k].real, u[k].imag) * _to_acb(grids[start + k][m0][n0])
                    out[gidx] = abs(_acb_to_mpc(acc, ndig))
                    continue
                urow = acb_mat([[acb(c.real, c.imag) for c in u]])      # 1 × M
                Vg = acb_mat([
                    [_to_acb(grids[start + k][m][n]) for m, n in entry_pairs]
                    for k in range(M)
                ])
                pp = urow * Vg                                          # 1 × n_e
                out[gidx] = max(
                    (abs(_acb_to_mpc(pp[0, i], ndig)) for i in range(len(entry_pairs))),
                    default=mp.mpf(0),
                )
            return out

        if is_arb:
            from .polynomial import _acb_to_mpc as _acb_pt
            def _raw(k, e):
                return mp.mpc(_acb_pt(grids[k][e[0]][e[1]]))
        else:
            def _raw(k, e):
                return mp.mpc(grids[k][e[0]][e[1]])
        for gidx, idxs, start, u in plan:
            if u is None:
                continue
            M = len(u)
            if n_entries == 1:
                m0, n0 = entry_pairs[0]
                out[gidx] = abs(sum(_raw(start + k, (m0, n0)) * u[k] for k in range(M)))
                continue
            best = mp.mpf(0)
            for m, n in zip(em_idx, en_idx):
                s = abs(sum(_raw(start + k, (m, n)) * u[k] for k in range(M)))
                if s > best:
                    best = s
            out[gidx] = best
        return out

    n_groups = len(groups)
    scores = [mp.mpf(0)] * len(roots)

    # Single-pass full scoring: every group, full n_extra, every entry. Used when
    # no n_select is given or there are too few groups for the pre-filter to pay.
    if n_select is None or n_select >= n_groups - 1 or n_groups <= 16:
        full = _score_group_indices(range(n_groups), n_extra, entries)
        for gidx, (loc, idxs) in enumerate(groups):
            for idx in idxs:
                scores[idx] = full[gidx]
        return scores

    # If the prefilter candidate window cannot exclude any group, skip the
    # coarse pass entirely and run the exact single-pass full scorer.
    prefilter_window = max(_PREFILTER_MIN_CAND, _PREFILTER_CAND_MULT * n_select)
    if prefilter_window >= n_groups or (
        n_groups > 0
        and mp.mpf(prefilter_window) / mp.mpf(n_groups) >= _PREFILTER_FULLPASS_RATIO
    ):
        full = _score_group_indices(range(n_groups), n_extra, entries)
        for gidx, (loc, idxs) in enumerate(groups):
            for idx in idxs:
                scores[idx] = full[gidx]
        return scores

    # Coarse -> fine pre-filter. Coarse: rank every group at full n_extra but
    # over a fixed-size entry subset (cheap). Fine: all entries on the must-keeps
    # plus the top candidate window, sized well above n_select so the kept set
    # (which the caller sweeps over caps 0..n_select) is unchanged.
    coarse = _score_group_indices(
        range(n_groups), n_extra, _coarse_entry_subset(entries)
    )
    musts = [g for g in range(n_groups) if coarse[g] == mp.inf]
    finite = sorted(
        (g for g in range(n_groups) if coarse[g] != mp.inf),
        key=lambda g: coarse[g], reverse=True,
    )
    n_cand = min(len(finite), max(_PREFILTER_MIN_CAND, _PREFILTER_CAND_MULT * n_select))
    cand = set(musts) | set(finite[:n_cand])
    fine = _score_group_indices(sorted(cand), n_extra, entries)

    for gidx, (loc, idxs) in enumerate(groups):
        s = fine[gidx] if gidx in cand else coarse[gidx]
        for idx in idxs:
            scores[idx] = s
    return scores


# Accuracy target defining the Bernstein window for `contribution_bernstein`:
# X = log(eps)/log|base| is where the prefactor has decayed to eps.
_BERNSTEIN_EPS = mp.mpf("1e-20")   # legacy fallback only
# Bernstein window as a multiple of the support endpoint b. Measured optimum
# is kappa = 1 -- i.e. the window IS the support of the interpolation density,
# with no free factor. kappa scan (worst derivative-entry abs error):
#   c=5/4 de=1.5   npoles=32:  1.0 -> 1e-19.19   2.0 -> 1e-16.01
#   c=6/5 de=2.0   npoles=32:  1.0 -> 1e-17.98   2.0 -> 1e-15.60
#   c=8/7 de=4/35  npoles=48:  1.0 -> 1e-23.38   1.5 -> 1e-26.00 (only case
#                              where kappa>1 wins)
# kappa=1 is best or tied in 8 of 9 (c, Delta_ext, npoles) combinations.
_BERNSTEIN_WINDOW_FACTOR = mp.mpf("1")

# Which measure supplies the nodes used to price a dropped pole in
# `contribution_equilibrium`. Must NOT depend on the selection being computed
# (bootstrapping from `nearest` makes the scorer reproduce `nearest`).
#   "bare" : base^x with no poles -- the damped-exponential geometry alone.
#   "full" : ALL selectable poles -- the geometry of the EXACT block, which is
#            the function actually being approximated.
#   Measured at c=8/7, l=0, Lambda=11 (max abs block error):
#       npoles      12        16        24        32
#       bare     1e-14.15  1e-15.22  1e-16.63  1e-24.18
#       full     1e- 9.61  1e-10.61  1e-16.28  1e-17.96
#   "full" is WORSE and its kept-pole reach collapses to 0.48-4.41, i.e. it
#   degenerates toward `nearest`: with 354 poles but only npoles+1 nodes, the
#   equilibrium density is dominated by the dense pole cluster near the origin,
#   the support contracts, and every distant pole then divides by a huge |w(p)|
#   and looks free to drop. The `nearest` bootstrap fails the same way.
#   The only non-arbitrary node set -- that of the final kept-pole prefactor --
#   is precisely the circular one, which is the structural limit of this
#   single-pole exact-cost framework.
_EQUILIBRIUM_NODE_MEASURE = "bare"


def _EQUILIBRIUM_BOOTSTRAP(shifted, npoles, first):
    if _EQUILIBRIUM_NODE_MEASURE == "full":
        return shifted
    return DampedRational(constant=mp.mpc(1), poles=(), base=first.exp_base)


# ---------------------------------------------------------------------------
# flint kernels for pricing dropped poles on the finite BS support (the
# contribution_potential scores and the contribution_optimal greedy order).
# The mpmath inputs are copied once, exactly (dyadic), into arb/acb; all the
# nodal and separation logarithms and the greedy's max-plus sweeps stay in
# flint, and only the returned scores are converted back.  Both work with the
# products themselves rather than their logs (arb has no exponent overflow): a
# greedy update is a multiply, where the log form needed ~1e6 arb logs per
# table.  None stands for the log form's infinities (arb has no signed
# infinity): a nodal product or prefactor whose point sits on a node or kept
# pole, or an exactly vanishing separation.  The greedy prices in flint only
# the candidates a float64 log-domain screen (_DropCostScreen) cannot rule
# out -- a few per step instead of every candidate at every probe.
# ---------------------------------------------------------------------------

_POS_INF = "+inf"
_NEG_INF = "-inf"


def _flint_point(z):
    """An mpmath scalar as an exact flint arb when real, else acb (no string round-trip)."""
    z = mp.mpc(z)
    if z.imag == 0:
        return _mpf_to_arb(z.real)
    return _mpf_to_acb(z)


def _flint_nodal(z, nodes, tiny):
    """prod(|z - node|), or None (+inf in the log form) when ``z`` lies within ``tiny`` of a node."""
    from flint import acb, arb

    if isinstance(z, acb) and abs(z.imag) > tiny:
        # no node can be within tiny: |z - node| >= |Im z|
        product = acb(1)
        for node in nodes:
            product *= z - node
        return abs(product)
    product = arb(1)
    for node in nodes:
        distance = abs(z - node)
        if distance <= tiny:
            return None
        product *= distance
    return product


def _flint_probe_geometry(reference_nodes, base):
    """(tiny, nodes, probes, log|base|) in flint at the current flint precision."""
    tiny = _mpf_to_arb(mp.mpf(10) ** (-mp.mp.dps + 10))
    nodes = [_mpf_to_arb(mp.mpf(x)) for x in reference_nodes]
    # The interpolation remainder is priced on the actual positive support
    # represented by the reference measure. Probe between nodes, not at them,
    # because W vanishes at every node. Include the threshold endpoint x=0 in
    # the interval definition, while using midpoints for the maximization.
    hi = _mpf_to_arb(max(mp.mpf(x) for x in reference_nodes))
    n_probe = max(64, 4 * len(nodes))
    probes = [hi * (2 * j + 1) / (2 * n_probe) for j in range(n_probe)]
    return tiny, nodes, probes, abs(_flint_point(base)).log()


def _finite_bs_potential_scores(
    scores, roots, reference_nodes, kept_poles, *, base, deval=mp.mpc(0),
):
    """Price dropped poles with the finite-domain BS interpolation remainder.

    ``reference_nodes`` are the Bohr--Sommerfeld nodes of a fixed reference
    measure (the full selectable-pole measure in the production caller).  For
    a provisional kept set ``K`` and a simple dropped pole ``p``, the reduced
    function has a residue enhanced by ``prod(q in K, p-q)`` and the exact
    nodal interpolation remainder is

        prefactor_K(x) * W(x) / ((x-p) W(p)).

    The principal-part contribution score supplies the pole strength.  For a
    repeated pole we use its order to convert the value-at-``deval`` score to
    a leading-coefficient scale; the full finite-domain factor is still used.
    This is a conservative, measure-aware score without a Bernstein ellipse or
    an effective asymptotic exponent.  Computed in flint (see above).
    """
    from flint import ctx as flint_ctx

    roots = [mp.mpc(p) for p in roots]
    if not reference_nodes:
        return list(scores)

    tol = tolerance(guard=10)
    orders = [1] * len(roots)
    for _, idxs in _group_roots_by_location(roots, tol):
        for idx in idxs:
            orders[idx] = len(idxs)

    old_prec = flint_ctx.prec
    flint_ctx.prec = _flint_prec_bits()
    try:
        tol_f = _mpf_to_arb(mp.mpf(tol))
        tiny, nodes, probes, log_base = _flint_probe_geometry(reference_nodes, base)
        points = [_flint_point(p) for p in roots]
        kept = [_flint_point(q) for q in kept_poles]
        deval_f = _flint_point(deval)
        w_probe = [_flint_nodal(x, nodes, tiny) for x in probes]

        # The finite remainder contains the kept-pole prefactor
        #
        #     w_K(x) = base**x / prod(q in K, |x-q|).
        #
        # This is common to every dropped candidate, so it is computed once per
        # probe.  If p is itself kept, the factor |x-p| in the interpolation
        # remainder cancels the one occurrence removed from w_K, so its
        # probe-dependent part is also just w_K(x).
        pref_base = []
        for x in probes:
            pref = (x * log_base).exp()
            for q in kept:
                separation = abs(x - q)
                if separation <= tiny:
                    pref = None
                    break
                pref /= separation
            pref_base.append(pref)

        result = []
        for score, point, order in zip(scores, points, orders):
            if score == mp.inf:
                result.append(mp.inf)
                continue
            score = mp.mpf(score)
            if score <= 0:
                result.append(mp.mpf(0))
                continue

            # If p is currently kept, price the candidate in the neighboring
            # prefactor with that occurrence removed. If p is dropped, price it
            # in the current provisional prefactor. This makes the score usable
            # for a fixed-point pole-set update rather than only for the current
            # drop set.
            kept_for_p = list(kept)
            is_kept = False
            for index, q in enumerate(kept_for_p):
                if abs(q - point) <= tol_f:
                    kept_for_p.pop(index)
                    is_kept = True
                    break

            distance_to_eval = abs(deval_f - point)
            if distance_to_eval <= tiny:
                result.append(mp.inf)
                continue
            amp = _mpf_to_arb(score) * distance_to_eval ** order
            for q in kept_for_p:
                separation = abs(point - q)
                if separation <= tol_f:
                    # A repeated pole is a single local group, not a collection
                    # of independent simple poles.  The exact lower-order
                    # coefficient after canceling part of that group involves
                    # derivatives of the principal part, not a literal zero
                    # product.  The group-level selector therefore omits
                    # coincident factors here; otherwise every currently kept
                    # repeated pole would be assigned an artificial +inf cost.
                    continue
                amp *= separation

            w_at_p = _flint_nodal(point, nodes, tiny)
            if w_at_p is None:
                result.append(mp.inf)
                continue

            best, infinite = None, False
            for x, w_x, pref in zip(probes, w_probe, pref_base):
                if pref is None:
                    continue
                if w_x is None:
                    infinite = True
                    break
                if is_kept:
                    # w_K(x) / |x-p| with the p factor removed from w_K.
                    value = pref * w_x / w_at_p
                else:
                    separation = abs(x - point)
                    if separation <= tiny:
                        continue
                    value = pref * w_x / (separation * w_at_p)
                if best is None or value > best:
                    best = value
            if infinite:
                result.append(mp.inf)
            else:
                result.append(mp.mpf(0) if best is None else _arb_to_mpf(amp * best))
        return result
    finally:
        flint_ctx.prec = old_prec


# Twice the float64 unit roundoff; the float error bounds below are all multiples of it, with room to spare.
_F64_EPS = 2.0 ** -52


def _float_log_distances(a, b, floor):
    """``log|a_i - b_j|`` in float64 for complex arrays ``a`` (n,) and ``b`` (m,), an error bound, and the pairs the
    floats cannot price.

    ``a`` and ``b`` are float64 roundings of exact flint values.  The computed distance is within a relative ``rel``
    of the exact one (input rounding, subtraction and modulus) and ``|log(1 + rel)| <= 2 rel`` for ``rel <= 1/2``;
    the log itself adds a few ulps.  A pair is *close* -- left to flint -- when ``rel`` exceeds 1e-3 or the distance
    might not exceed ``floor`` (the flint code's own tiny/tol cut), so no cut the flint engine makes is ever read
    off a float.  Close entries carry log 0 and error 0 and must not be used.
    """
    import numpy as np

    d = np.abs(a[:, None] - b[None, :])
    scale = np.abs(a)[:, None] + np.abs(b)[None, :]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        rel = 4 * _F64_EPS * scale / d + 4 * _F64_EPS
        log_d = np.log(d)
        err = 2 * rel + 2 * _F64_EPS * np.abs(log_d)
        close = ~((rel <= 1e-3) & (d * (1 - rel) > 2 * floor))
    log_d[close] = 0.0
    err[close] = 0.0
    return log_d, err, close


class _DropCostScreen:
    """float64 brackets ``[lo, hi]`` of the log drop costs in the contribution_optimal greedy.

    The greedy's cost of a dropped candidate ``p`` is ``amp_p / W(p) * max_j pref(x_j) W(x_j) / |x_j - p|`` over the
    live probes, an exact flint product per (candidate, probe) at every step.  Its winner is nearly always clear by
    many orders of magnitude, so each candidate's log cost is bracketed here in floats with rigorous rounding bounds
    (sums of logs, updated incrementally as poles are kept), and the greedy prices in flint only the candidates
    whose ``hi`` reaches the best ``lo``.  A pruned candidate is certainly below another's exact cost (the flint
    midpoints agree with the exact values far below the float bounds), so the ranking, ties included, is the flint
    ranking of every candidate.  What the floats cannot decide is marked for flint: ``exact[i]`` for a candidate
    within the flint cuts of a node, probe, kept pole or ``deval``, or with a non-positive / infinite score, and
    ``uncertain[j]`` for a probe whose nodal product or prefactor is not safely finite (a live one disables the
    screen for the step).
    """

    def __init__(self, points, probes, nodes, kept, scores, orders, deval, log_base, tiny, tol):
        # every argument is a sequence of (or a) values complex() converts: mpmath or flint numbers
        import math

        import numpy as np

        self.tol = float(tol)
        tiny = float(tiny)
        points = np.array([complex(p) for p in points])
        probes = np.array([complex(x) for x in probes])
        nodes = np.array([complex(x) for x in nodes])
        kept = np.array([complex(q) for q in kept])
        self.points = points
        self.log_sep, self.err_sep, self.close_sep = _float_log_distances(points, probes, tiny)
        exact = self.close_sep.any(axis=1)

        def row_sums(log_d, err, close):
            # a sum of k logs: the terms' own errors plus k ulps of the magnitudes summed
            k = log_d.shape[1]
            return (log_d.sum(axis=1), err.sum(axis=1) + (k + 1) * _F64_EPS * np.abs(log_d).sum(axis=1),
                    close.any(axis=1))

        log_w_p, err_w_p, close_w_p = row_sums(*_float_log_distances(points, nodes, tiny))
        log_w_x, err_w_x, self.uncertain = row_sums(*_float_log_distances(probes, nodes, tiny))

        # A_i = log amp_i - log W(p_i): log score + order log|deval - p| + sum over kept of log|p - q| - log W(p)
        log_eval, err_eval, close_eval = _float_log_distances(points, np.array([complex(deval)]), tiny)
        log_score, err_score = np.zeros(len(points)), np.zeros(len(points))
        for i, score in enumerate(scores):
            if not mp.isfinite(score) or score <= 0:
                exact[i] = True
            else:
                # log(man * 2^exp) from the mpf's own dyadic parts (no multiprecision log)
                _sign, man, exp, _bc = score._mpf_
                log_man, log_scale = math.log(man), exp * math.log(2)
                log_score[i] = log_man + log_scale
                err_score[i] = 4 * _F64_EPS * (abs(log_man) + abs(log_scale))
        order = np.array(orders, dtype=float)
        log_kept, err_kept, close_kept = row_sums(*_float_log_distances(points, kept, self.tol))
        self.A = log_score + order * log_eval[:, 0] + log_kept - log_w_p
        self.err_A = (err_score + order * err_eval[:, 0] + err_kept + err_w_p
                      + 8 * _F64_EPS * (np.abs(log_score) + order * np.abs(log_eval[:, 0]) + np.abs(log_kept)
                                        + np.abs(log_w_p)))
        self.exact = exact | close_w_p | close_eval[:, 0] | close_kept

        # B_j = log pref(x_j) + log W(x_j): x log|base| - sum over kept of log|x - q| + log W(x)
        log_base = float(log_base)
        log_pref, err_pref, close_pref = row_sums(*_float_log_distances(probes, kept, tiny))
        linear = probes.real * log_base
        self.B = linear - log_pref + log_w_x
        self.err_B = (err_pref + err_w_x
                      + 8 * _F64_EPS * (np.abs(linear) + np.abs(log_pref) + np.abs(log_w_x)))
        self.uncertain = self.uncertain | close_pref

    def keep(self, index):
        """Account for keeping candidate ``index``: every prefactor loses ``|x - p|``, every amp gains ``|p - q|``."""
        import numpy as np

        log_sep, err_sep = self.log_sep[index], self.err_sep[index]
        self.B = self.B - log_sep
        self.err_B = self.err_B + err_sep + 4 * _F64_EPS * (np.abs(self.B) + np.abs(log_sep))
        # a close probe separation made ``index`` exact, so it is only ever kept by a flint decision; those probes
        # may then have a vanishing |x - q| (an infinite prefactor), which the floats leave to flint
        self.uncertain = self.uncertain | self.close_sep[index]
        log_root, err_root, close_root = _float_log_distances(self.points, self.points[index:index + 1], self.tol)
        self.A = self.A + log_root[:, 0]
        self.err_A = self.err_A + err_root[:, 0] + 4 * _F64_EPS * (np.abs(self.A) + np.abs(log_root[:, 0]))
        self.exact = self.exact | close_root[:, 0]

    def certain_losers(self, groups, live):
        """The positions in ``groups`` (lists of candidate indices) whose cost is certainly below another's.

        A group's log cost -- the max over its members, each a max over the ``live`` probes -- is bracketed by the
        members' brackets; a group whose upper bound is below the best lower bound (less a margin far above every
        float error) loses.  Groups with an ``exact`` member are never bracketed, and with no live probe, or a live
        ``uncertain`` one, nothing is.
        """
        import numpy as np

        screened = [k for k, group in enumerate(groups) if not self.exact[group].any()]
        if len(screened) < 2 or not live.any() or (self.uncertain & live).any():
            return set()
        idx = np.array([i for k in screened for i in groups[k]])
        starts = np.cumsum([0] + [len(groups[k]) for k in screened[:-1]])
        A, err_A = self.A[idx], self.err_A[idx]
        B, err_B = self.B[live], self.err_B[live]
        log_sep = self.log_sep[idx][:, live]
        value = A[:, None] + B[None, :] - log_sep
        err = (err_A[:, None] + err_B[None, :] + self.err_sep[idx][:, live]
               + 4 * _F64_EPS * (np.abs(A)[:, None] + np.abs(B)[None, :] + np.abs(log_sep)))
        lo = np.maximum.reduceat((value - err).max(axis=1), starts)
        hi = np.maximum.reduceat((value + err).max(axis=1), starts)
        floor = lo.max() - 1e-9 * (1 + abs(lo.max()))
        return {k for k, top in zip(screened, hi.tolist()) if top < floor}


def _contribution_optimal_order(
    scores, roots, reference_nodes, forced_poles, *, base,
    deval=mp.mpc(0), target_slots=None, exact_slots=False,
):
    """Build a nested greedy pole order from finite-measure drop costs.

    ``contribution_potential`` independently optimizes each requested pole
    count.  That is useful for finding a good individual fit, but the
    resulting sets need not be nested.  This variant starts with the forced
    poles and repeatedly adds the remaining pole (or conjugate pair) with the
    largest finite-support drop cost.  The reference nodes are supplied by
    the caller and are fixed for the whole sequence, so the first ``N``
    greedy choices do not depend on the final requested ``npoles``.

    The returned list is a ranking, not necessarily the final kept set.
    Complex-conjugate members are adjacent.  The caller takes the longest
    *object prefix* fitting under the pole cap; it never skips an unfit pair to
    take a later real pole, which guarantees nesting even at odd caps (an odd
    cap may simply leave one slot unused).
    """
    from operator import mul, truediv

    from flint import ctx as flint_ctx

    roots = [mp.mpc(p) for p in roots]
    scores = [mp.mpf(s) for s in scores]
    ranked = []
    if not roots:
        return ranked

    tol = tolerance(guard=10)

    # Treat a conjugate pair as one indivisible greedy object, while repeated
    # real poles remain separate objects: adding the second real occurrence
    # is precisely how a simple pole is promoted to a double pole.
    # The partner is the nearest unused root within tol (the first on ties).  An exact conjugate is at distance 0,
    # so it is looked up by value; only a partner off by roundoff needs the linear scan (it was 57% of the greedy
    # as an O(n^2) mpmath scan per epoch).
    unused = set(range(len(roots)))
    by_value = {}
    for index, root in enumerate(roots):
        by_value.setdefault(root, []).append(index)
    groups = []
    while unused:
        i = min(unused)
        unused.remove(i)
        group = [i]
        if abs(mp.im(roots[i])) > tol:
            conjugate = mp.conj(roots[i])
            exact = [j for j in by_value.get(conjugate, ()) if j in unused]
            if exact:
                match = exact[0]
            else:
                match = min(
                    (j for j in unused if abs(roots[j] - conjugate) <= tol),
                    default=None,
                    key=lambda j: abs(roots[j] - conjugate),
                )
            if match is not None:
                unused.remove(match)
                group.append(match)
        groups.append(group)

    # The fallback is only used if node construction fails.  It is still
    # nested, and retains the same conjugate-pair grouping convention.
    if not reference_nodes:
        groups.sort(key=lambda g: -max(scores[i] for i in g))
        for group in groups:
            ranked.extend(roots[i] for i in group)
        return ranked

    orders = [1] * len(roots)
    for _, idxs in _group_roots_by_location(roots, tol):
        for idx in idxs:
            orders[idx] = len(idxs)

    old_prec = flint_ctx.prec
    flint_ctx.prec = _flint_prec_bits()
    try:
        # Incremental engine for the same dropped-pole cost evaluated by
        # _finite_bs_potential_scores.  The node set is fixed throughout this
        # greedy sequence, so the nodal products and pole/probe distances are
        # computed at most once.  Adding q to K then requires only
        #
        #   pref_K(x) /= |x-q|,    amp_p *= |p-q|,
        #
        # rather than rebuilding every product for every candidate at every
        # greedy step.  Only the candidates the float screen (_DropCostScreen)
        # cannot rule out are priced in flint, so their per-candidate data --
        # W(p), the probe distances and kernel row, amp_p -- are built on first
        # use; the values, and so the ranking, are those of pricing every one.
        import numpy as np

        tol_f = _mpf_to_arb(mp.mpf(tol))
        tiny, nodes, probes, log_base = _flint_probe_geometry(reference_nodes, base)
        kept = [_flint_point(q) for q in forced_poles]
        deval_f = _flint_point(deval)
        count = len(roots)
        w_probe = [_flint_nodal(x, nodes, tiny) for x in probes]
        screen = _DropCostScreen(roots, probes, nodes, kept, scores, orders, deval_f, log_base, tiny, tol_f)

        _unset = object()
        points = [None] * count     # the flint candidates, converted on first use

        def _point(index):
            if points[index] is None:
                points[index] = _flint_point(roots[index])
            return points[index]

        w_at_p = [_unset] * count
        probe_sep = [None] * count
        kernel, kernel_infinite = [None] * count, [None] * count
        amp = [None] * count
        chosen_order = []   # the kept poles after ``kept``, in order: amp_p multiplies their separations in turn

        def _probe_sep(index):
            # None: an exactly vanishing separation (its probe's prefactor becomes infinite once p is kept)
            if probe_sep[index] is None:
                p = _point(index)
                probe_sep[index] = [None if d.is_zero() else d for d in (abs(x - p) for x in probes)]
            return probe_sep[index]

        def _w_at_p(index):
            if w_at_p[index] is _unset:
                w_at_p[index] = _flint_nodal(_point(index), nodes, tiny)
            return w_at_p[index]

        def _kernel(index):
            """kernel[i][j] = W(x_j) / (|x_j - p_i| W(p_i)); None is +inf (W(p_i) is not None here)."""
            if kernel[index] is None:
                wp = _w_at_p(index)
                row = [
                    None if wx is None or sep is None else wx / (sep * wp)
                    for wx, sep in zip(w_probe, _probe_sep(index))
                ]
                kernel[index] = row
                kernel_infinite[index] = any(v is None for v in row)
            return kernel[index]

        def _amp(index):
            if amp[index] is None:
                score, point = scores[index], _point(index)
                if score == mp.inf:
                    value = _POS_INF
                elif score <= 0:
                    value = _NEG_INF
                else:
                    distance = abs(deval_f - point)
                    value = _POS_INF if distance <= tiny else _mpf_to_arb(score) * distance ** orders[index]
                if value is not _POS_INF and value is not _NEG_INF:
                    for q in kept:
                        separation = abs(point - q)
                        if separation > tol_f:
                            value *= separation
                amp[index] = (value, 0)
            value, applied = amp[index]
            if value is not _POS_INF and value is not _NEG_INF:
                for chosen_index in chosen_order[applied:]:
                    low, high = sorted((index, chosen_index))
                    separation = abs(_point(low) - _point(high))
                    if separation > tol_f:
                        value *= separation
            amp[index] = (value, len(chosen_order))
            return value

        def _best(index):
            """max over live probes of pref * kernel (None: +inf; _NEG_INF: no live probe)."""
            row = _kernel(index)
            if live_all and not kernel_infinite[index]:
                return max(map(mul, pref, row))
            best = _NEG_INF
            for pj, kval in zip(pref, row):
                if pj is None:
                    continue
                if kval is None:
                    return None
                value = pj * kval
                if best is _NEG_INF or value > best:
                    best = value
            return best

        # a cost is (class, value): +inf (1), finite (0, the exact ball midpoint,
        # so ties compare as the mpmath floats did), zero -- the log form's -inf -- (-1)
        def _cost(group):
            candidate_costs = []
            for index in group:
                a = _amp(index)
                if a is _POS_INF or _w_at_p(index) is None:
                    candidate_costs.append((1, zero))
                    continue
                best = _best(index)
                if best is None and a is not _NEG_INF:
                    candidate_costs.append((1, zero))
                elif a is _NEG_INF or best is _NEG_INF:
                    candidate_costs.append((-1, zero))
                else:
                    candidate_costs.append((0, (a * best).mid()))
            return max(candidate_costs)

        # None marks a probe whose prefactor is infinite (it sits on a kept
        # pole); such probes are skipped, as the mpmath form skipped inf.
        pref = [(x * log_base).exp() for x in probes]
        for q in kept:
            for j, x in enumerate(probes):
                if pref[j] is not None:
                    separation = abs(x - q)
                    pref[j] = None if separation <= tiny else pref[j] / separation

        live_all = all(pj is not None for pj in pref)
        remaining = list(groups)
        selected_groups = []
        target = None if target_slots is None else int(target_slots)
        zero = _mpf_to_arb(mp.mpf(0))
        while remaining:
            eligible = list(range(len(remaining)))
            if exact_slots and target is not None:
                realized = sum(len(previous) for previous in selected_groups)
                slots_left = target - realized
                fitting = [
                    pos for pos, candidate in enumerate(remaining)
                    if len(candidate) <= slots_left
                ]
                if fitting:
                    eligible = fitting

            # The groups the float screen proves below another are dropped; the
            # rest are priced exactly, and the first maximal one wins, as in a
            # max over every eligible group.
            live = np.array([pj is not None for pj in pref])
            losers = screen.certain_losers([remaining[pos] for pos in eligible], live)
            priced = [pos for k, pos in enumerate(eligible) if k not in losers]
            group_costs = {pos: _cost(remaining[pos]) for pos in priced}
            best_pos = max(priced, key=group_costs.__getitem__)
            group = remaining.pop(best_pos)
            selected_groups.append(group)
            ranked.extend(roots[i] for i in group)

            # Insert the whole real pole / conjugate-pair object: the prefactor
            # and the screen now, every dropped candidate's amp on its next use.
            for chosen_index in group:
                sep_row = _probe_sep(chosen_index)
                if live_all and all(d is not None for d in sep_row):
                    pref = list(map(truediv, pref, sep_row))
                else:
                    # a vanishing |x - q| makes that probe's prefactor infinite
                    pref = [
                        None if pj is None or d is None else pj / d
                        for pj, d in zip(pref, sep_row)
                    ]
                    live_all = all(pj is not None for pj in pref)
                chosen_order.append(chosen_index)
                screen.keep(chosen_index)

            if target is not None:
                realized = sum(len(previous) for previous in selected_groups)
                if realized >= target:
                    break
        return ranked
    finally:
        flint_ctx.prec = old_prec


# The pure-mpmath form of the greedy kernel above: the original the flint form was checked against (identical
# rankings on recorded inputs, 2026-09-27).  No runtime path calls it; it is kept as the independent reference
# for the flint greedy's float screen (tests/core/test_contribution_full_scoring.py).


def _contribution_optimal_order_mpmath(
    scores, roots, reference_nodes, forced_poles, *, base,
    deval=mp.mpc(0), target_slots=None, exact_slots=False,
):
    """Build a nested greedy pole order from finite-measure drop costs.

    ``contribution_potential`` independently optimizes each requested pole
    count.  That is useful for finding a good individual fit, but the
    resulting sets need not be nested.  This variant starts with the forced
    poles and repeatedly adds the remaining pole (or conjugate pair) with the
    largest finite-support drop cost.  The reference nodes are supplied by
    the caller and are fixed for the whole sequence, so the first ``N``
    greedy choices do not depend on the final requested ``npoles``.

    The returned list is a ranking, not necessarily the final kept set.
    Complex-conjugate members are adjacent.  The caller takes the longest
    *object prefix* fitting under the pole cap; it never skips an unfit pair to
    take a later real pole, which guarantees nesting even at odd caps (an odd
    cap may simply leave one slot unused).
    """
    roots = [mp.mpc(p) for p in roots]
    scores = [mp.mpf(s) for s in scores]
    kept = [mp.mpc(p) for p in forced_poles]
    ranked = []
    if not roots:
        return ranked

    tol = tolerance(guard=10)

    # Treat a conjugate pair as one indivisible greedy object, while repeated
    # real poles remain separate objects: adding the second real occurrence
    # is precisely how a simple pole is promoted to a double pole.
    unused = set(range(len(roots)))
    groups = []
    while unused:
        i = min(unused)
        unused.remove(i)
        group = [i]
        if abs(mp.im(roots[i])) > tol:
            conjugate = mp.conj(roots[i])
            match = min(
                (j for j in unused if abs(roots[j] - conjugate) <= tol),
                default=None,
                key=lambda j: abs(roots[j] - conjugate),
            )
            if match is not None:
                unused.remove(match)
                group.append(match)
        groups.append(group)

    # The fallback is only used if node construction fails.  It is still
    # nested, and retains the same conjugate-pair grouping convention.
    if not reference_nodes:
        groups.sort(key=lambda g: -max(scores[i] for i in g))
        for group in groups:
            ranked.extend(roots[i] for i in group)
        return ranked

    # Incremental log-domain engine for the same dropped-pole cost evaluated
    # by _finite_bs_potential_scores. The node set is fixed throughout this
    # greedy sequence, so all nodal potentials and pole/probe distances can be
    # cached once. Adding q to K then requires only
    #
    #   log pref_K(x) -= log|x-q|,
    #   log amp_p     += log|p-q|,
    #
    # rather than rebuilding every high-precision product for every candidate
    # at every greedy step.
    nodes = [mp.mpf(x) for x in reference_nodes]
    tiny = mp.mpf(10) ** (-mp.mp.dps + 10)
    hi = max(nodes)
    n_probe = max(64, 4 * len(nodes))
    probes = [
        hi * mp.mpf(2 * j + 1) / (2 * n_probe)
        for j in range(n_probe)
    ]
    log_base = mp.log(abs(mp.mpc(base)))

    def _log_nodal(z):
        value = mp.mpf(0)
        for node in nodes:
            distance = abs(mp.mpc(z) - node)
            if distance <= tiny:
                return mp.inf
            value += mp.log(distance)
        return value

    log_w_probe = [_log_nodal(x) for x in probes]
    log_w_at_p = [_log_nodal(p) for p in roots]
    log_probe_sep = [
        [mp.log(abs(mp.mpc(x) - p)) for x in probes]
        for p in roots
    ]
    log_root_sep = []
    for p in roots:
        row = []
        for q in roots:
            separation = abs(p - q)
            row.append(
                None if separation <= tol else mp.log(separation)
            )
        log_root_sep.append(row)

    orders = [1] * len(roots)
    for _, idxs in _group_roots_by_location(roots, tol):
        for idx in idxs:
            orders[idx] = len(idxs)

    log_amp = []
    deval = mp.mpc(deval)
    for score, pole, order in zip(scores, roots, orders):
        if score == mp.inf:
            value = mp.inf
        elif score <= 0:
            value = mp.mpf("-inf")
        else:
            distance = abs(deval - pole)
            value = (
                mp.inf if distance <= tiny
                else mp.log(score) + order * mp.log(distance)
            )
        for q in kept:
            separation = abs(pole - q)
            if separation > tol and value not in (mp.inf, mp.mpf("-inf")):
                value += mp.log(separation)
        log_amp.append(value)

    log_pref = [x * log_base for x in probes]
    for q in kept:
        for j, x in enumerate(probes):
            separation = abs(mp.mpc(x) - q)
            log_pref[j] = (
                mp.inf if separation <= tiny
                else log_pref[j] - mp.log(separation)
            )

    kernel = [
        [
            lwx - lsep - lwp
            for lwx, lsep in zip(log_w_probe, sep_row)
        ]
        for lwp, sep_row in zip(log_w_at_p, log_probe_sep)
    ]

    remaining = list(groups)
    selected_groups = []
    target = None if target_slots is None else int(target_slots)
    while remaining:
        group_costs = []
        for group in remaining:
            candidate_costs = []
            for index in group:
                if log_amp[index] == mp.inf or log_w_at_p[index] == mp.inf:
                    candidate_costs.append(mp.inf)
                    continue
                best = max(
                    (
                        lp + kval
                        for lp, kval in zip(log_pref, kernel[index])
                        if lp != mp.inf
                    ),
                    default=mp.mpf("-inf"),
                )
                candidate_costs.append(log_amp[index] + best)
            group_costs.append(max(candidate_costs))

        eligible = list(range(len(remaining)))
        if exact_slots and target is not None:
            realized = sum(len(previous) for previous in selected_groups)
            slots_left = target - realized
            fitting = [
                pos for pos, candidate in enumerate(remaining)
                if len(candidate) <= slots_left
            ]
            if fitting:
                eligible = fitting
        best_pos = max(eligible, key=lambda pos: group_costs[pos])
        group = remaining.pop(best_pos)
        selected_groups.append(group)
        ranked.extend(roots[i] for i in group)

        # Insert the whole real pole / conjugate-pair object and update the
        # costs of every still-dropped candidate.
        remaining_indices = [i for candidate in remaining for i in candidate]
        for chosen_index in group:
            for j, separation_log in enumerate(log_probe_sep[chosen_index]):
                log_pref[j] -= separation_log
            for index in remaining_indices:
                separation_log = log_root_sep[index][chosen_index]
                if (
                    separation_log is not None
                    and log_amp[index] not in (mp.inf, mp.mpf("-inf"))
                ):
                    log_amp[index] += separation_log

        if target is not None:
            realized = sum(len(previous) for previous in selected_groups)
            if realized >= target:
                break
    return ranked


def _contribution_optimal_adaptive_order(
    scores, roots, forced_poles, shifted, *, node_excess, base,
    deval=mp.mpc(0), target_slots=None, epoch_size=8, exact_fill=True,
):
    """Build a nested order using a scale-matched measure in short epochs.

    The original nested selector used the node distribution at the *full*
    selectable-pole degree for every greedy decision.  At small retained
    degree this prices poles on a support much wider than the interpolant
    actually uses.  Here the order is grown in fixed, target-independent
    epochs.  At the start of each epoch we construct the optimal nodes for the
    currently kept prefactor at the degree of that epoch's upper endpoint,
    then reuse the fast fixed-node greedy engine within the epoch.

    Fixed epoch boundaries are essential: requesting a longer order extends,
    rather than changes, every previously constructed prefix.  ``epoch_size``
    controls only how stale the common measure can become, not the eventual
    interpolation nodes (which are always rebuilt from the final prefactor).
    """
    target = len(roots) if target_slots is None else max(0, int(target_slots))
    width = max(1, int(epoch_size))
    remaining_roots = [mp.mpc(p) for p in roots]
    remaining_scores = [mp.mpf(s) for s in scores]
    forced = [mp.mpc(p) for p in forced_poles]
    selected = []
    tol = tolerance(guard=10)

    while remaining_roots and len(selected) < target:
        epoch_upper = (
            (len(selected) // width + 1) * width
        )
        # Finish the fixed epoch even when the caller requests a shorter
        # prefix. Otherwise exact-fill choices at a partial epoch can replace
        # a conjugate pair with a real pole and make the ranking depend on
        # the requested cap (and on whether a longer ranking was cached).
        # The caller applies the requested conjugate-object prefix afterward.
        epoch_slots = epoch_upper - len(selected)

        current = DampedRational(
            constant=shifted.constant,
            poles=tuple(selected),
            base=shifted.base,
        )
        try:
            n_nodes = max(2, epoch_upper + int(node_excess))
            reference_nodes = [
                mp.mpf(t) for t in get_sample_points(current, n_nodes)
            ]
        except Exception:
            reference_nodes = []

        chunk = _contribution_optimal_order(
            remaining_scores,
            remaining_roots,
            reference_nodes,
            forced + selected,
            base=base,
            deval=deval,
            target_slots=epoch_slots,
            exact_slots=exact_fill,
        )
        if not chunk:
            break

        # Remove exactly the occurrences emitted by the chunk.  Repeated real
        # poles are distinct selectable factors, so matching consumes one
        # occurrence at a time.
        # The emitted pole is a copy of a remaining root, so the first exact
        # equal is the nearest match; the tol scan is kept for anything else.
        for pole in chunk:
            pole = mp.mpc(pole)
            match = next((i for i, other in enumerate(remaining_roots) if other == pole), None)
            if match is None:
                match = min(
                    (
                        i for i, other in enumerate(remaining_roots)
                        if abs(other - pole) <= tol
                    ),
                    default=None,
                    key=lambda i: abs(remaining_roots[i] - pole),
                )
            if match is None:
                raise ValueError(
                    "adaptive contribution_optimal emitted an unknown pole"
                )
            selected.append(remaining_roots.pop(match))
            remaining_scores.pop(match)
    return selected


def _take_contribution_optimal_prefix(shifted, ranked, cap):
    """Take the longest conjugate-object prefix fitting within ``cap``.

    Generic ranked selectors skip a two-pole conjugate pair when only one slot
    remains and may then take a later real pole.  That fill policy is sensible
    for a one-off selection but destroys nesting between adjacent caps.  The
    optimal selector instead stops at the first object that does not fit.
    """
    cap = int(cap)
    if cap < 0:
        raise ValueError("pole cap must be >= 0")
    ranked = [mp.mpc(p) for p in ranked]
    tol = tolerance(guard=10)
    chosen = []
    index = 0
    while index < len(ranked):
        pole = ranked[index]
        if abs(mp.im(pole)) <= tol:
            group = [pole]
            index += 1
        else:
            if (
                index + 1 >= len(ranked)
                or abs(ranked[index + 1] - mp.conj(pole)) > tol
            ):
                raise ValueError(
                    "contribution_optimal ranking contains an unpaired "
                    "complex pole"
                )
            group = [pole, ranked[index + 1]]
            index += 2
        if len(chosen) + len(group) > cap:
            break
        chosen.extend(group)
    return DampedRational(
        constant=shifted.constant,
        poles=tuple(chosen),
        base=shifted.base,
    )


def _aggregate_contribution_scores(
    exacts, *, delta_eval, score_entries="kept", pole_ranking="numeric",
    n_select=None,
):
    """Per-pole full-principal-part contribution aggregated across tables.

    The contribution analogue of :func:`_aggregate_residue_scores`: each table
    contributes its own contribution scores (max over its scored entries), and
    the multi-table score for each pole is the max across tables, so coordinates
    sharing the universal denominator share one selection.

    ``pole_ranking="numeric"`` extracts the principal part from the attached
    numeric sampler (:func:`_numeric_contribution_scores`); it falls back to the
    symbolic numerators (:func:`_contribution_scores`) when no sampler is present
    -- which raises if the table is a degree-only skeleton, exactly as the
    residue path does."""
    if not exacts:
        raise ValueError("at least one exact table is required")
    n_poles = len(exacts[0].denominator_roots)
    per_table = []
    numeric_cache = {}
    symbolic_cache = {}

    def _scalar_key(x):
        z = mp.mpc(x)
        return (mp.mpf(z.real)._mpf_, mp.mpf(z.imag)._mpf_)

    delta_key = _scalar_key(delta_eval)

    for exact in exacts:
        scores = None
        if pole_ranking == "numeric":
            sampler = getattr(exact, "numeric_sampler", None)
            num_key = (
                id(sampler),
                delta_key,
                str(score_entries),
                int(n_select) if n_select is not None else None,
                mp.mp.prec,
            )
            if sampler is not None and num_key in numeric_cache:
                scores = numeric_cache[num_key]
            else:
                scores = _numeric_contribution_scores(
                    exact, delta_eval=delta_eval, score_entries=score_entries,
                    n_select=n_select,
                )
                if sampler is not None and scores is not None:
                    numeric_cache[num_key] = scores
        if scores is None:
            sym_key = (id(exact), delta_key, str(score_entries), mp.mp.prec)
            cached_sym = symbolic_cache.get(sym_key)
            if cached_sym is None:
                cached_sym = _contribution_scores(
                    exact, delta_eval=delta_eval, score_entries=score_entries
                )
                symbolic_cache[sym_key] = cached_sym
            scores = cached_sym
        per_table.append(scores)
    return [max(scores[j] for scores in per_table) for j in range(n_poles)]


def _default_node_search_dps() -> int:
    """Default working precision for the optimal-node search.

    The interpolation polynomial passes through the *exact* block values at
    whichever nodes are returned, so node precision affects only which nodes
    are chosen -- not the accuracy of the polynomial at those nodes. The
    optimal-node interpolation error decays exponentially in n_sample_points,
    far below any plausible node-precision contribution, so the node search
    can run at a fraction of the user's working precision with no visible
    effect on the block output.

    Default: ``max(15, mp.mp.dps // 2)``. At ``dps <= 30`` (typical) this
    caps at 15 -- IEEE-double precision, plenty for 20-100 nodes. At higher
    dps the cap grows to half so that very-high-precision runs (dps > 60)
    still resolve very-fine nodes when needed.
    """
    return max(15, mp.mp.dps // 2)


def _kept_max_numerator_degree(exact, kept) -> int:
    """The largest numerator degree among the kept ``(m, n)`` entries (all of them when ``kept`` is None)."""
    if kept is None:
        return exact.max_numerator_degree()
    if exact.numerator_degrees is not None:
        degree = lambda m, n: exact.numerator_degrees[m][n]
    else:
        degree = lambda m, n: exact.numerators[m][n].degree
    return max((degree(m, n) for m, n in kept if m < len(exact.numerator_degrees or exact.numerators)), default=0)


def _default_interpolation_shift(exact: DeltaDerivTable):
    """Bootstrap-natural interpolation shift convention.

    Returns the unitarity bound `|spin|` for the 2d bases and `0` for the
    chiral basis, so the polynomial variable starts at the SDPB positivity
    bound. This helper never falls back to a conditioning-driven shift: a
    real shifted pole at x > 0 is split off as an exact principal part
    downstream (or rejected with `force_keep_real_positive=False`), never
    absorbed by changing variables.
    """
    if exact.spin is None:
        return mp.mpf(0)
    return mp.mpf(abs(mp.re(mp.mpc(exact.spin))))


def _aggregate_residue_scores(
    exacts,
    *,
    score_entries: str = "kept",
    pole_ranking: str = "numeric",
) -> list:
    """Per-pole residue score aggregated across one or more exact tables.

    Each table contributes its own `_residue_scores` (max over its (m, n)
    entries); the multi-table score for each pole is the max over tables.
    Aggregating this way lets several coordinates derived from the same
    universal denominator share a single residue-based pole selection (and
    therefore a single interpolation prefactor and a single set of sample
    points), which is what multi-coordinate serialization requires.

    ``score_entries`` (forwarded to :func:`_residue_scores`) selects which
    (m, n) numerator entries feed each table's per-pole residue max:
    ``"kept"`` (the truncation triangle) or ``"value"`` (only the
    undifferentiated (0, 0) block entry).
    """
    if not exacts:
        raise ValueError("at least one exact table is required")
    first = exacts[0]
    n_poles = len(first.denominator_roots)
    # The per-pole denominator factors depend only on the shared Kac
    # denominator roots shared by every coordinate table. Compute the products
    # once in flint, adjusting their normalization for each table below,
    # rather than rebuilding the O(n_poles^2) product per table.
    prepared = None
    if pole_ranking == "symbolic":
        prec_bits = int(mp.mp.dps * 3.33) + 20
        prepared = _residue_denom_factors_acb(
            [mp.mpc(r) for r in first.denominator_roots],
            first.denominator.coeffs[-1],
            tolerance(guard=10),
            prec_bits,
        )
    per_table = []
    for exact in exacts:
        table_prepared = prepared
        if prepared is not None and exact.denominator.coeffs[-1] != first.denominator.coeffs[-1]:
            # Shared roots do not require the same polynomial normalization.
            # Reuse the products, but restore this denominator's leading factor.
            ratio = (_mpf_to_acb(exact.denominator.coeffs[-1])
                     / _mpf_to_acb(first.denominator.coeffs[-1]))
            table_prepared = (prepared[0], [factor * ratio for factor in prepared[1]])
        per_table.append(_residue_scores(
            exact, prepared=table_prepared, score_entries=score_entries, pole_ranking=pole_ranking,
        ))
    return [
        max(scores[j] for scores in per_table)
        for j in range(n_poles)
    ]


def _partition_force_keep_indices(roots, shift) -> tuple[list[int], list[int]]:
    """Split denominator-root indices into (forced, selectable).

    A root is "forced" iff it is real and lies strictly above the shift on the
    real axis (i.e. Re(root - shift) > 0 with Im(root) == 0). These are
    exactly the poles that `interpolation.get_sample_points` rejects, because
    a real positive shifted pole would lie inside the matrix-model eigenvalue
    support [0, b] and break Bohr-Sommerfeld node placement. Selectable roots
    are the complement: real with Re <= shift, or any complex root (BS admits
    complex conjugate pairs regardless of Re).

    Returns index lists into `roots` so the caller can re-derive any aligned
    data (residue scores, provenance, etc.).
    """
    real_shift = mp.mpf(mp.re(shift))
    forced, selectable = [], []
    for idx, root in enumerate(roots):
        root_c = mp.mpc(root)
        if mp.im(root_c) == 0 and mp.re(root_c) > real_shift:
            forced.append(idx)
        else:
            selectable.append(idx)
    return forced, selectable


@lru_cache(maxsize=64)
def _canonical_real_block_roots(root_keys, dps):
    from .interpolation import _canonicalize_prefactor_poles

    tol = mp.mpf(10) ** (-(dps // 2))
    return _canonicalize_prefactor_poles((root for _type, root in root_keys),
                                       real_tol=tol, relative=True)


def _realify_conjugate_denominator(table: "DeltaDerivTable") -> "DeltaDerivTable":
    """Rebuild a real-block denominator as a real-coefficient polynomial.

    For 1 < c < 25 the Liouville b is complex, so the Kac Delta-poles are
    complex -- but for real c and real external weights they come in conjugate
    pairs, so the denominator polynomial is mathematically real. Built as a
    product of complex ``(Delta - root)`` factors, though, it carries complex
    coefficients whose imaginary parts only cancel *approximately* when
    evaluated at real Delta, leaving a ~10^-(0.7 dps) imaginary residue in the
    block value that propagates into the interpolated numerators (the source of
    the off-locus-regularize complex contamination).

    Rebuilding the denominator directly from its conjugate-symmetric roots as
    real quadratics ``(Delta - r)(Delta - conj r) = Delta^2 - 2 Re(r) Delta +
    |r|^2`` and real linears, multiplied in real arithmetic, makes the block
    value *exactly* real (no residue, hence no downstream chopping) and lets the
    Lagrange fit run in real rather than complex arithmetic. Returns the table
    unchanged when the denominator is already real or when the roots are not
    conjugate-symmetric (a genuine complex-c block, which must stay complex).
    """
    denom = table.denominator
    if not table.denominator_roots:
        return table
    # Include source types: float decimal labels must not alias an equal
    # binary-valued Fraction in a cache key. Precision sets the roundoff band.
    canonical_roots = list(_canonical_real_block_roots(
        tuple((type(root), root) for root in table.denominator_roots), mp.mp.dps,
    ))
    if denom.field is mp.mpf:
        if all(a == b for a, b in zip(canonical_roots, table.denominator_roots)):
            return table
        return _dataclass_replace(table, denominator_roots=canonical_roots)
    real_coeffs = real_factor_coeffs_from_conjugate_roots(
        canonical_roots, leading_coeff=denom.coeffs[-1]
    )
    if real_coeffs is None:
        return table  # not conjugate-symmetric -> genuine complex, leave it
    # The numerators are left as-is: with a real denominator the block value is
    # exactly real already, so the interpolant is real with no chopping. (The
    # numerators evaluate to real values regardless of their field; normalizing
    # their field too gives no measurable speedup here, so it is not worth the
    # extra projection.)
    return _dataclass_replace(
        table,
        denominator=Polynomial(real_coeffs, field=mp.mpf),
        denominator_roots=canonical_roots,
    )


def _interpolation_entries(exact, coord_kept):
    """The (m, n) entries an interpolation fits: the kept set, or every genuine entry of the table.

    A lambda_cut (order-Lambda triangle) build leaves the discarded m + n > Lambda corner as degree-0 zero
    placeholders, and the diagonal of a z <-> zbar-odd (parity '-') block is a genuine zero -- both reconstruct to
    zero, so skipping them keeps the numeric sampler from being asked for the corner without changing the result.
    Degree zero alone cannot identify a zero: low-order pole-free tables have genuine constant numerators. A
    skeleton has no coefficients to prove a zero, so its builders record structural zeros separately.
    """
    if coord_kept is None:
        return [(m, n) for m in range(exact.m_max + 1) for n in range(exact.n_max + 1)
                if (m, n) not in exact.zero_entries and (exact.numerators is None
                    or exact.numerator_degree(m, n) > 0 or exact.numerators[m][n].coeffs[0] != 0)]
    return [(m, n) for (m, n) in coord_kept if m <= exact.m_max and n <= exact.n_max]


def _align_interpolation_tables(exacts, *, validation_dps=None):
    """Validate shared geometry and align poles before index-based operations."""
    from .interpolation import _as_complex_input, _pole_geometry_scales, _same_geometry_pole

    first = next(iter(exacts.values()))
    roots = first.denominator_roots
    base = mp.mpc(first.exp_base)
    if not mp.isfinite(base):
        raise ValueError("exponential base must be finite")
    tol = mp.mpf(10) ** (-max(5, (mp.mp.dps if validation_dps is None else validation_dps) - 10))
    aligned = dict(exacts)
    for coord, other in list(exacts.items())[1:]:
        other_base = mp.mpc(other.exp_base)
        if not (mp.isfinite(base) and mp.isfinite(other_base)) or abs(other_base - base) > tol:
            raise ValueError("exacts must share the exponential base across coordinates")
        if other.spin != first.spin:
            raise ValueError("exacts must share spin across coordinates")
        if len(other.denominator_roots) != len(roots):
            raise ValueError("exacts must share denominator_roots across coordinates")
        other_roots = other.denominator_roots
        if all(left == right for left, right in zip(roots, other_roots)):
            continue
        remaining = list(range(len(roots)))
        permutation = []
        left_scales = right_scales = None
        for root in roots:
            index = next((i for i in remaining if root == other_roots[i]), None)
            if index is None:
                if left_scales is None:
                    left_scales = _pole_geometry_scales(roots)
                    right_scales = _pole_geometry_scales(other_roots)
                scale = left_scales[_as_complex_input(root)]
                candidates = [i for i in remaining if _same_geometry_pole(
                    root, other_roots[i], tol,
                    scale=min(scale, right_scales[_as_complex_input(other_roots[i])]),
                )]
                index = min(candidates, key=lambda i: abs(root - other_roots[i])) if candidates else None
            if index is None:
                raise ValueError("exacts must share denominator roots across coordinates")
            permutation.append(index)
            remaining.remove(index)
        if permutation == list(range(len(roots))):
            continue
        provenance = other.root_provenance
        aligned[coord] = _dataclass_replace(
            other, denominator_roots=[other_roots[i] for i in permutation],
            root_provenance=None if provenance is None else [provenance[i] for i in permutation],
        )
    return aligned


@_quiet_conditioning
def interpolate_delta_tables(
    exacts: dict,
    npoles: int | None,
    *,
    pole_selection: str = "contribution_optimal",
    n_sample_points: int | None = None,
    shift=None,
    min_pole_distance=None,
    sample_point_kwargs: dict | None = None,
    node_search_dps: int | None = None,
    delta_eval=None,
    pre_stripped: bool = False,
    force_keep_real_positive: bool = True,
    num_threads: int = 1,
    shared_sample_points: bool = False,
    node_subset_mode: str = "off",
    pole_ranking: str = "numeric",
    kept_entries_by_coord: dict | None = None,
) -> dict:
    """Interpolate several exact tables sharing a single prefactor.

    All tables in `exacts` (keyed by coordinate) must share the same
    `exp_base`, `denominator_roots`, and `spin`: the universal damped-rational
    prefactor depends only on c / ell / externals / q_order, not on the
    coordinate. Residue-magnitude scores are aggregated by taking the max
    across tables, so the selected pole subset, the matrix-model sample
    points, and the polynomial origin are *common to every coordinate*. The
    only per-coordinate data is each `(m, n)` interpolating polynomial.

    Returns a dict keyed by the same coordinates, with each value an
    `InterpolatedDeltaTable` carrying the shared prefactor and shift.
    Downstream `serialize_deriv_tables` consumes the dict directly.

    `pole_selection` controls *which* poles are kept and defaults to
    `"contribution_optimal"`. It starts from the full-principal-part
    contribution at `delta_eval`, then uses the optimal-node interpolation
    measure to construct a nested greedy pole order. `"contribution"` is the
    simpler direct ranking by full-principal-part magnitude; `"nearest"` and
    `"residue"` remain useful comparison controls.

    `"contribution_optimal"` constructs one nested greedy pole order using the
    finite-support BS interpolation remainder on a scale-adaptive reference
    measure.  The optimal measure is refreshed at fixed eight-pole boundaries
    from the prefix retained so far, avoiding the small-degree bias of pricing
    every pole against the full-pole support. Increasing `npoles` therefore
    extends the previous kept set (subject to conjugate pairs, for which an odd
    cap may leave one slot unused). This reference is used only to rank poles:
    every resulting interpolant still computes its own optimal BS nodes from
    its selected prefactor and actual polynomial degree.

    `n_sample_points` defaults to `max_num_deg - dropped_pole_count + 1` with
    `max_num_deg` taken across every coordinate's numerator entries. The
    polynomial degree it produces is linear in `npoles`: keeping all poles
    reproduces the exact rational function (degree `max_num_deg`), and each
    dropped pole reduces the polynomial degree by one. The polynomial
    coefficients then decay exponentially with `n_sample_points` under the
    matrix-model node placement (arXiv:2509.14307), so the tails of high-
    degree polynomials are not noise -- they are the optimal-interpolation
    coefficients (the SDP writer itself uses the block values at the nodes). More
    aggressive order reduction below this default is available through the
    explicit `n_sample_points` argument.

    `force_keep_real_positive=True` (default) lifts the legacy "no real
    positive shifted pole" restriction so c <= 1 blocks are interpolable.
    Real poles with Re(p) > shift are split off the selectable pool and
    kept in the final prefactor unconditionally. Their principal parts are
    split off exactly (`_split_forced_principal_parts`), only the regular
    remainder is fitted, and `_recombine_principal_parts` puts them back; when
    no principal parts are available, the fallback multiplies the samples by
    `D_F(Δ) = ∏(Δ - p_F)` and fits that instead. BS only sees the selectable
    (BS-admissible) prefactor, so the real-positive-pole guard is satisfied.
    `npoles` counts *selectable* kept poles only; forced poles are always
    kept on top. For tables without any
    real positive shifted pole (every c > 1 case under the bootstrap
    shift), the partition is empty and the path is bit-for-bit identical
    to the legacy code. Pass `force_keep_real_positive=False` to restore
    the legacy raise-on-real-positive contract.
    """
    num_threads = validate_worker_count(num_threads)
    if n_sample_points is not None:
        n_sample_points = _integer_control('n_sample_points', n_sample_points, 1)
    if node_search_dps is not None:
        node_search_dps = _integer_control('node_search_dps', node_search_dps, 1)
    keep_all = npoles is None
    if npoles is not None:
        npoles = _validate_npoles(npoles)
    if not exacts:
        raise ValueError("exacts must be non-empty")
    # Keep the caller-owned table as the cache anchor.  The real-denominator
    # normalization below may create a fresh dataclass on every call, whereas
    # npoles sweeps repeatedly pass the same original exact table.
    _selection_cache_owner = next(iter(exacts.values()))
    # Real-c blocks with complex (conjugate-paired) Delta-poles carry a complex
    # denominator polynomial that is only approximately real when evaluated;
    # rebuild it real so the whole fit runs in real arithmetic and the
    # interpolant is exactly real (no imaginary residue to chop downstream).
    exacts = {
        coord: _realify_conjugate_denominator(table)
        for coord, table in exacts.items()
    }
    exacts = _align_interpolation_tables(exacts)
    coords = list(exacts.keys())
    first = exacts[coords[0]]

    # Strip removable poles (e.g. identity at Δ=0 for ℓ=0) before building
    # the interpolation prefactor. The exact Kac denominator carries factors
    # that cancel from every numerator; including them as interpolation-pole
    # candidates corrupts all selection rules. Serialization already does this
    # step; interpolation must too.
    # pre_stripped=True skips the stripping step when the caller has already
    # done it once (e.g. sweeping many npoles over the same exact table).
    # The strip runs only for tables flagged `requires_pole_strip` -- the
    # ε-extrapolated singular-locus tables from `regularize_delta_table`.
    # Generic-c tables are already minimal (build-time K != 0 gate), so the
    # precision-fragile numeric strip is skipped (no high-q false positives).
    if not pre_stripped and any(
        getattr(t, "requires_pole_strip", False) for t in exacts.values()
    ):
        exacts = _strip_removable_poles_from_exact_tables(exacts)
    first = exacts[coords[0]]

    if shift is None:
        shift = _default_interpolation_shift(first)

    # Partition the denominator roots into BS-incompatible "forced" poles --
    # real, lying above the shift -- and the BS-admissible "selectable" rest.
    # The forced set is always kept in the final prefactor (so the c <= 1
    # Kac poles above the unitarity bound survive); only the selectable set
    # feeds the Bohr-Sommerfeld node placement and the pole-selection rules.
    # When `force_keep_real_positive=False` we skip the partition so the
    # node construction rejects any real positive shifted pole.
    all_roots = list(first.denominator_roots)
    if force_keep_real_positive:
        forced_idx, selectable_idx = _partition_force_keep_indices(all_roots, shift)
    else:
        forced_idx, selectable_idx = [], list(range(len(all_roots)))
    forced_roots = [all_roots[i] for i in forced_idx]
    selectable_roots = [all_roots[i] for i in selectable_idx]

    if forced_roots:
        # Split the forced poles off as exact principal parts: the pole selection below still ranks the selectable
        # poles on the block itself (their principal parts are the remainder's), and the fit order-reduces the
        # regular remainder (``fit_exacts``), whose poles are all selectable.  Fitting the forced poles into the
        # numerator instead (the fallback, when no principal parts are available) makes the fitted function vary by
        # the forced factor prod(x - p) -- many orders of magnitude across the nodes, which follow only the kept
        # poles -- and the c < 1 order-reduced tables lost every digit at any precision.
        entries_by_coord = {
            coord: _interpolation_entries(table, None if kept_entries_by_coord is None else kept_entries_by_coord.get(coord))
            for coord, table in exacts.items()
        }
        split = _split_forced_principal_parts(exacts, forced_idx, entries_by_coord)
    else:
        split = None
    # `npoles=None` means "keep every selectable pole" -- no order reduction.
    # The interpolant then has degree max_num_deg and reproduces the exact
    # rational function (bit-equal to the symbolic table up to node-eval
    # precision), but carries sample points and uses the well-conditioned
    # numeric sampler. This is the unified "full" method: full == interpolation
    # with nothing dropped.
    if npoles is None:
        npoles = len(selectable_roots)
    if int(npoles) >= len(selectable_roots):
        # Keeping every pole requires no ranking. The position-based selector
        # avoids the contour evaluations used by contribution scorers.
        pole_selection = "rightmost"

    full = DampedRational(
        constant=mp.mpc(1),
        poles=tuple(selectable_roots),
        base=first.exp_base,
    )
    shifted = full.shift_x(shift)
    scores = None
    greedy_order = None
    if pole_selection in (
        "residue", "residue_all", "contribution_leading",
        "contribution_leading_all",
    ):
        # "residue"/"contribution_leading" rank poles by their residue in the
        # (0, 0) block value alone; the "*_all" variants use the max over the
        # kept derivative triangle. The "contribution_leading" variants then
        # divide by the distance to the evaluation point; the "residue" variants
        # keep the raw residue. All then keep the highest-scoring poles. (The
        # full-principal-part "contribution"/"contribution_all" selectors are
        # handled in the branch below.)
        score_entries = pole_selection_score_entries(pole_selection)
        full_residue_scores = _aggregate_residue_scores(
            list(exacts.values()),
            score_entries=score_entries,
            pole_ranking=pole_ranking,
        )
        residue_scores = [full_residue_scores[i] for i in selectable_idx]
        if not pole_selection_is_contribution(pole_selection):
            scores = residue_scores
        else:
            # Evaluation point in the shifted variable x = Δ − shift.
            # Default: x_eval = 0, i.e. the unitarity threshold Δ = shift.
            if delta_eval is not None:
                x_eval = mp.mpc(delta_eval) - mp.mpc(shift)
            else:
                x_eval = mp.mpc(0)
            scores = []
            for s, p in zip(residue_scores, shifted.poles):
                dist = abs(mp.mpc(p) - x_eval)
                # Guard against evaluation exactly at a pole (shouldn't occur
                # after removable-pole stripping, but be safe).
                scores.append(s / float(dist) if dist > 1e-12 else s)
    elif pole_selection_is_full_contribution(pole_selection):
        # Contribution: rank each pole by the magnitude of its full
        # principal part evaluated at delta_eval -- every Laurent coefficient,
        # not just the leading one over distance^1. The distance is folded into
        # the principal-part evaluation, so there is no separate division here.
        score_entries = pole_selection_score_entries(pole_selection)
        deval = mp.mpc(delta_eval) if delta_eval is not None else mp.mpc(shift)
        full_contribution_scores = _aggregate_contribution_scores(
            list(exacts.values()), delta_eval=deval, score_entries=score_entries,
            pole_ranking=pole_ranking,
            n_select=None if pole_selection == "contribution_optimal" else npoles,
        )
        scores = [full_contribution_scores[i] for i in selectable_idx]
        if pole_selection == "contribution_relative":
            # Rank poles by their contribution to the max RELATIVE block error
            # over the sample range -- the quantity the SDP's positivity
            # constraints actually see.
            #
            # The SDP constrains the polynomial P = G / prefactor, so an
            # absolute block error dG becomes a polynomial error dG/prefactor:
            # at Delta=12 the prefactor is ~1e-32, so a 1e-27 absolute error is
            # a 1e+5.6 polynomial error. What stays bounded is the RELATIVE
            # error, dG/G = dP/P, and that is what must be small where the
            # contracted functional nearly cancels (i.e. at an active
            # constraint).
            #
            # Dropping pole p perturbs the reduced function by
            #     dP(x) = s_p * w(x) / ((x - p) w(p))
            # so its contribution to the relative block error is
            #     dG/G = prefactor(x) * dP(x) / G(x) .
            # Score = max over the sample range of that. prefactor(x) is taken
            # as base^x (the kept-pole product is a slowly varying polynomial
            # and, being selection-dependent, would reintroduce circularity).
            #
            # This differs from `contribution_density`, which maximizes the
            # ABSOLUTE error prefactor(x)*dP(x) and so under-weights the outer
            # sample points where G is exponentially small -- exactly where the
            # measured relative error degrades to ~1e-6 and where the
            # transplant test located the constraint violations.
            try:
                _mm_r = int(first.m_max)
                _n_r = max(2, int(npoles) + _mm_r + 1)
                _ts_r = [mp.mpf(t) for t in get_sample_points(shifted, _n_r)]
            except Exception:
                _ts_r = []
            if _ts_r:
                _lb_r = mp.log(abs(mp.mpc(first.exp_base)))
                _tiny_r = mp.mpf(10) ** (-mp.mp.dps + 10)
                _lo_r, _hi_r = min(_ts_r), max(_ts_r)
                _np_r = 2 * len(_ts_r)
                _probe_r = [
                    _lo_r + (_hi_r - _lo_r) * (mp.mpf(2 * i + 1) / (2 * _np_r))
                    for i in range(_np_r)
                ]

                def _logw_r(z):
                    acc = mp.mpf(0)
                    zz = mp.mpc(z)
                    for t in _ts_r:
                        d = abs(zz - t)
                        acc += mp.log(d if d > _tiny_r else _tiny_r)
                    return acc

                # log|G| at each probe, from the exact table (numeric sampler).
                _lg = []
                for x in _probe_r:
                    try:
                        gv = abs(mp.mpc(first.evaluate(x + mp.mpf(shift))[0][0]))
                    except Exception:
                        gv = mp.mpf(0)
                    _lg.append(mp.log(gv) if gv > 0 else mp.mpf("-inf"))
                _lwx = [(_logw_r(x), x) for x in _probe_r]
                _wr = []
                for _s, _root in zip(scores, selectable_roots):
                    if _s == mp.inf:
                        _wr.append(_s)
                        continue
                    _p = mp.mpc(_root)
                    _lp = _logw_r(_p)
                    _best = None
                    for (_lx, _x), _lgx in zip(_lwx, _lg):
                        if _lgx == mp.mpf("-inf"):
                            continue
                        _sep = abs(mp.mpc(_x) - _p)
                        if _sep < _tiny_r:
                            continue
                        # log[ base^x * |w(x)/((x-p) w(p))| / |G(x)| ]
                        _v = _lx - _lp - mp.log(_sep) + _x * _lb_r - _lgx
                        if _best is None or _v > _best:
                            _best = _v
                    _wr.append(_s if _best is None else mp.mpf(_s) * mp.e ** _best)
                scores = _wr
        if pole_selection == "contribution_potential":
            # Price each candidate with the finite-domain Bernstein--Szego
            # remainder rather than replacing the nodal polynomial by its
            # large-pole asymptotic power law.  The reference nodes are built
            # from the full selectable-pole measure, so this remains a
            # non-circular selector; a short fixed-point update then inserts
            # the denominator of the provisional kept prefactor.
            #
            # For a simple dropped pole p, the exact nodal interpolation
            # remainder is
            #
            #   w_K(x) W(x) / ((x-p) W(p)),
            #
            # where w_K is the final kept prefactor and W is the actual BS
            # nodal polynomial. `_finite_bs_potential_scores` evaluates the
            # maximum of this quantity on the finite positive support. This
            # is the same finite measure that generates the interpolation
            # nodes, including the soft edge from the exponential field and
            # the attraction of the full pole set.
            try:
                _mnd_p = max(e.max_numerator_degree() for e in exacts.values())
                _drop_p = len(all_roots) - (int(npoles) + len(forced_roots))
                # the fit's node count: with the forced poles split off it fits the remainder, len(forced) lower
                _n_p = max(2, _mnd_p - max(0, _drop_p) + 1 - (len(forced_roots) if split is not None else 0))
                _ts_p = [mp.mpf(t) for t in get_sample_points(shifted, _n_p)]
            except Exception:
                _ts_p = []

            if _ts_p:
                _base_p = first.exp_base
                _x_eval_p = (
                    mp.mpc(delta_eval) - mp.mpc(shift)
                    if delta_eval is not None else mp.mpc(0)
                )
                # split off as principal parts, the forced poles are not in the fit's prefactor (as for
                # contribution_optimal)
                _forced_shifted_p = [] if split is not None else [
                    mp.mpc(p) - mp.mpc(shift) for p in forced_roots
                ]
                _selectable_shifted_p = [
                    mp.mpc(p) - mp.mpc(shift) for p in selectable_roots
                ]

                def _same_pole_set(left, right):
                    if len(left) != len(right):
                        return False
                    remaining = list(right)
                    for pole in left:
                        for index, other in enumerate(remaining):
                            if abs(mp.mpc(pole) - mp.mpc(other)) <= tolerance(guard=10):
                                remaining.pop(index)
                                break
                        else:
                            return False
                    return not remaining

                # Start from the ordinary full-contribution ranking. Two
                # updates are enough in practice. If the kept-denominator
                # map cycles, choose the candidate with the smaller worst
                # dropped-pole cost rather than depending on iteration parity.
                _provisional = select_kept_poles(
                    shifted, int(npoles), "contribution", scores=scores
                )
                _scored = scores
                _history = []
                for _ in range(2):
                    _kept_for_cost = list(_provisional.poles) + _forced_shifted_p
                    _scored = _finite_bs_potential_scores(
                        scores, _selectable_shifted_p, _ts_p, _kept_for_cost,
                        base=_base_p, deval=_x_eval_p,
                    )
                    _next = select_kept_poles(
                        shifted, int(npoles), "contribution", scores=_scored,
                    )
                    _history.append((_next, _scored))
                    if _same_pole_set(_provisional.poles, _next.poles):
                        _provisional = _next
                        break
                    _provisional = _next

                if len(_history) > 1 and not _same_pole_set(
                    _history[-2][0].poles, _history[-1][0].poles
                ):
                    def _dropped_cost(item):
                        candidate, cost_map = item
                        remaining = list(candidate.poles)
                        dropped = []
                        for pole, cost in zip(_selectable_shifted_p, cost_map):
                            for index, kept_pole in enumerate(remaining):
                                if abs(pole - kept_pole) <= tolerance(guard=10):
                                    remaining.pop(index)
                                    break
                            else:
                                dropped.append(cost)
                        return max(dropped, default=mp.mpf(0))

                    _best_candidate, scores = min(
                        _history, key=_dropped_cost
                    )
                else:
                    # `_scored` is already the cost map that selected the
                    # stable candidate; re-evaluating it would only duplicate
                    # the expensive finite-support scan.
                    scores = _scored
        if pole_selection == "contribution_optimal":
            # Scale-adaptive nested variant of the finite-measure potential
            # selector.  The node-count excess is m_max + 1, independent of
            # the numerator degree.
            _node_excess_o = int(first.m_max) + 1
            _epoch_size_o = 8
            _x_eval_o = (
                mp.mpc(delta_eval) - mp.mpc(shift)
                if delta_eval is not None else mp.mpc(0)
            )
            _cache_key_o = (
                int(mp.mp.dps), str(mp.mpc(shift)), str(_x_eval_o),
                score_entries, pole_ranking, len(selectable_roots),
                _node_excess_o, _epoch_size_o, "adaptive-v3-full-epochs", split is not None,
            )
            _cache_o = getattr(
                _selection_cache_owner, "_contribution_optimal_cache", {}
            )
            _cached_o = _cache_o.get(_cache_key_o)
            if (
                _cached_o is not None
                and int(_cached_o["slots"]) >= int(npoles)
            ):
                greedy_order = list(_cached_o["order"])
            else:
                # with the forced poles split off as principal parts the fit's prefactor has only the selectable
                # ones, so the ranking's reference measure does too
                _forced_shifted_o = [] if split is not None else [
                    mp.mpc(p) - mp.mpc(shift) for p in forced_roots
                ]
                _selectable_shifted_o = [
                    mp.mpc(p) - mp.mpc(shift) for p in selectable_roots
                ]
                greedy_order = _contribution_optimal_adaptive_order(
                    scores,
                    _selectable_shifted_o,
                    _forced_shifted_o,
                    shifted,
                    node_excess=_node_excess_o,
                    base=first.exp_base,
                    deval=_x_eval_o,
                    target_slots=int(npoles),
                    epoch_size=_epoch_size_o,
                    exact_fill=True,
                )
                _cache_o[_cache_key_o] = {
                    "slots": len(greedy_order),
                    "order": tuple(greedy_order),
                }
                setattr(
                    _selection_cache_owner,
                    "_contribution_optimal_cache",
                    _cache_o,
                )
        if pole_selection == "contribution_density":
            # Weight each pole's principal-part score by the EXACT cost of
            # dropping it, priced against the optimal interpolation density
            # that the Bohr-Sommerfeld solver already computes for THIS
            # measure.
            #
            # For nodes {t_i} with nodal polynomial w(z) = prod(z - t_i),
            # interpolating a simple pole at p has error exactly
            #     w(x) / [ (x - p) w(p) ]
            # so, multiplying by the prefactor to convert reduced-function
            # error into BLOCK error (which is what the SDP contracts against),
            # the cost of dropping p is
            #     s_p * max_x |base^x w(x) / ((x - p) w(p))| ,  x in [0, b].
            #
            # The nodes come from `shifted` -- the FULL selectable-pole
            # prefactor -- so the density is the poles-aware equilibrium
            # measure whose support endpoint b solves the standard damped-
            # rational support equation (eq. 3.45 of 2509.14307; dropping the
            # pole terms recovers the bare Mhaskar-Rakhmanov-Saff number
            # a_n = 4n/lambda, lambda = -log|base|). This measure does not
            # depend on the selection being computed, so there is no
            # circularity, and b grows with the node count as it should --
            # unlike a fixed hand-chosen window.
            #
            # CRITICALLY the node count must be the one the fit actually uses,
            # n_sample_points = max_num_deg - dropped_pole_count + 1, not
            # npoles: pricing against a shorter node set under-resolves the
            # comparison and mis-ranks every pole.
            try:
                _mnd = max(e.max_numerator_degree() for e in exacts.values())
                _drop = len(all_roots) - (int(npoles) + len(forced_roots))
                # the fit's node count: with the forced poles split off it fits the remainder, len(forced) lower
                _n_act = max(2, _mnd - max(0, _drop) + 1 - (len(forced_roots) if split is not None else 0))
                _ts = [mp.mpf(t) for t in get_sample_points(shifted, _n_act)]
            except Exception:
                _ts = []
            if _ts:
                _lb = mp.log(abs(mp.mpc(first.exp_base)))
                _tiny = mp.mpf(10) ** (-mp.mp.dps + 10)

                def _logw(z):
                    acc = mp.mpf(0)
                    zz = mp.mpc(z)
                    for t in _ts:
                        d = abs(zz - t)
                        acc += mp.log(d if d > _tiny else _tiny)
                    return acc

                _lo, _hi = min(_ts), max(_ts)
                # Probe between the nodes: w vanishes AT them, so node points
                # would report zero cost for every pole.
                _n_probe = 4 * len(_ts)
                _probe = [
                    _lo + (_hi - _lo) * (mp.mpf(2 * i + 1) / (2 * _n_probe))
                    for i in range(_n_probe)
                ]
                _lw_probe = [(_logw(x), x) for x in _probe]
                _weighted = []
                for _s, _root in zip(scores, selectable_roots):
                    if _s == mp.inf:
                        _weighted.append(_s)
                        continue
                    _p = mp.mpc(_root)          # already in shifted coords
                    _lp = _logw(_p)
                    _best = None
                    for _lwx, _x in _lw_probe:
                        _sep = abs(mp.mpc(_x) - _p)
                        if _sep < _tiny:
                            _best = None
                            break
                        _v = _lwx - _lp - mp.log(_sep) + _x * _lb
                        if _best is None or _v > _best:
                            _best = _v
                    if _best is None:
                        _weighted.append(mp.inf)     # pole sits on the nodes
                    else:
                        _weighted.append(mp.mpf(_s) * mp.e ** _best)
                scores = _weighted
        if pole_selection == "contribution_equilibrium":
            # EXACT cost of dropping a pole, from the interpolation error
            # identity. For nodes {x_i} with nodal polynomial w(z)=prod(z-x_i),
            # interpolating f(z)=1/(z-p) has error exactly
            #
            #     f(x) - P(x) = w(x) / [ (x - p) * w(p) ]
            #
            # so a pole with principal-part weight s_p contributes, at x,
            #     s_p * |prefactor(x)| * |w(x)| / (|x - p| * |w(p)|)
            # to the error in the BLOCK (the prefactor converts reduced-function
            # error into block error, which is what the SDP sees). Score each
            # pole by the max of that over the physical window.
            #
            # This needs no Bernstein ellipse, no window-edge parameter X and
            # no accuracy target eps: the geometry enters through the actual
            # BS nodes, which are the optimal nodes for THIS damped-rational
            # measure (the ellipse belongs to the unweighted Chebyshev problem
            # and is the wrong level-curve family here).
            #
            # The nodes depend on the kept poles, which is circular. Using the
            # `nearest` selection as bootstrap made the scorer REPRODUCE
            # nearest (its nodes sit in [0,2], so every far pole divides by a
            # huge |w(p)| and looks free). Use instead the BARE measure
            # base^x with NO poles: the interpolation geometry of the damped
            # exponential alone, which is independent of any selection and so
            # cannot bias the ranking toward the scheme that generated it.
            boot = _EQUILIBRIUM_BOOTSTRAP(shifted, npoles, first)
            try:
                node_xs = [mp.mpf(x) for x in get_sample_points(boot, npoles + 1)]
            except Exception:
                node_xs = []
            if node_xs:
                lo = min(node_xs)
                hi = max(node_xs)
                n_probe = 64
                probe = [
                    lo + (hi - lo) * mp.mpf(i) / (n_probe - 1)
                    for i in range(n_probe)
                ]
                base_c = mp.mpc(first.exp_base)

                def _log_w(z):
                    tot = mp.mpf(0)
                    for xi in node_xs:
                        d = abs(mp.mpc(z) - xi)
                        tot += mp.log(d) if d > 0 else mp.mpf("-inf")
                    return tot

                logw_probe = [(_log_w(x), x) for x in probe]
                weighted = []
                for s_, root in zip(scores, selectable_roots):
                    if s_ == mp.inf:
                        weighted.append(s_)
                        continue
                    xp = mp.mpc(root) - mp.mpc(shift)
                    lwp = _log_w(xp)
                    best = mp.mpf("-inf")
                    for lwx, x in logw_probe:
                        sep = abs(mp.mpc(x) - xp)
                        if sep == 0:
                            best = mp.inf
                            break
                        val = (lwx - lwp - mp.log(sep)
                               + mp.re(mp.log(abs(base_c ** x))))
                        if val > best:
                            best = val
                    weighted.append(
                        s_ if best == mp.inf else s_ * mp.e ** best
                    )
                scores = weighted
        if pole_selection == "contribution_bernstein":
            # Weight each pole's principal-part score by the approximation-
            # theoretic cost of DROPPING it: the Bernstein ellipse parameter
            # rho of the pole relative to the window over which the block must
            # be accurate. The best degree-D polynomial approximation to
            # 1/(x - p) on that window converges like rho^-D, so the error a
            # dropped pole actually contributes is score * rho^-D.
            #
            # `contribution` ranks by score alone and so marches outward into
            # the near-c=1 complex Kac strip, keeping poles whose prefactor
            # weight is ~1e-55 and manufacturing numerically flat SDP
            # directions. This factor makes distant poles never worth a slot,
            # so the kept-pole reach saturates instead of growing with npoles.
            #
            # Window: [0, X] with X = _BERNSTEIN_WINDOW_FACTOR times the
            # density support endpoint (below); the fallback is
            # X = log(eps)/log|base|, i.e. where the
            # prefactor has decayed to the accuracy target -- beyond it block
            # error is multiplied by < eps and cannot matter. Measured against
            # directly-computed drop costs, this window is over-conservative
            # for near poles and over-optimistic for far ones; it is the best
            # simple choice tested for the compactified interpolation path.
            # Window = BERNSTEIN_WINDOW_FACTOR * b, where b is the support
            # endpoint of the interpolation density for THIS measure -- the
            # poles-aware Mhaskar-Rakhmanov-Saff number, solved from the
            # damped-rational support equation. This replaces the earlier
            # fixed eps-derived window, which had no derivation, did not scale
            # with the node count, and did not respond to the pole structure.
            #
            # Measured optimum (worst DERIVATIVE-entry abs error, scanning X):
            #   c=5/4 de=1.5   npoles=32: X=11 -> 1e-19.2  (b=9.87)
            #   c=6/5 de=2.0   npoles=32: X=11 -> 1e-18.0
            #   c=8/7 de=4/35  npoles=48: X=28 -> 1e-25.9  (b=14.27)
            # The optimum tracks b (5.4/9.9/14.3 at npoles 16/32/48) and NOT
            # the pole-free MRS number 4n/lambda (=56 at npoles=32), which
            # ignores the poles that pull the support in ~6x. Large X
            # degenerates toward all-entry scoring with no geometric weight
            # (X=80 -> pole reach 40-60, accuracy collapses to 1e-7..1e-10).
            # kappa = 1 measured optimal: the window IS the density support.
            base_abs = abs(mp.mpc(first.exp_base))
            # The score depends only on the pole set (`shifted`, universal
            # denominator), `npoles` and m_max, not on max_numerator_degree.
            try:
                # N reconstructs the block's node count from the
                # derivative-triangle size.
                _n_w = max(2, int(npoles) + int(first.m_max) + 1)
                _b = solve_support_endpoint(shifted, _n_w)
                X = mp.mpf(_BERNSTEIN_WINDOW_FACTOR) * mp.mpf(_b)
            except Exception:
                # Fall back to the legacy eps window if the support solve fails.
                X = mp.log(_BERNSTEIN_EPS) / mp.log(base_abs)
            # D = npoles, NOT the true fit degree max_num_deg - dropped (kept
            # from when the removed method-(B) moment build had to select the
            # same poles with a different numerator degree). Measured cost of this
            # compromise: <= 0.25 decades, since rho already dominates the
            # ordering and rescaling its exponent barely reorders poles.
            D = mp.mpf(max(1, int(npoles)))
            weighted = []
            for s_, root in zip(scores, selectable_roots):
                if s_ == mp.inf:
                    weighted.append(s_)
                    continue
                x_p = mp.mpc(root) - mp.mpc(shift)
                w_ = (2 * x_p - X) / X
                r_ = mp.sqrt(w_ * w_ - 1)
                rho = max(abs(w_ + r_), abs(w_ - r_))
                if rho <= 1:
                    weighted.append(mp.inf)      # on/inside the window: must keep
                else:
                    weighted.append(s_ * rho ** (-D))
            scores = weighted
        if pole_selection == "contribution_physical":
            # Physically-weighted contribution: multiply the principal-part
            # score by the damped-rational prefactor's decay at the pole,
            # |base^(Δ_p − shift)|. The plain `contribution` score de-weights a
            # distant pole only algebraically (~1/|Δ_eval − Δ_p|), but the
            # block's actual sensitivity to that pole on the physical window is
            # suppressed EXPONENTIALLY by the prefactor base^(Δ−shift). At
            # near-c=1 the Kac residues swing ~80 orders non-monotonically (the
            # complex m≠n strip), so the algebraic de-weighting is far too weak
            # and `contribution` drags in high-residue far poles that blow up
            # the interpolation support width and the SDP conditioning. Using
            # |base^(Δ_p−shift)| (not base^Re) keeps this correct when the base
            # is complex (complex c / external): the pole's imaginary part then
            # contributes e^{−Im·arg(base)}, which base^Re would wrongly drop.
            base = mp.mpc(first.exp_base)
            shift_c = mp.mpc(shift)
            weighted = []
            for s, root in zip(scores, selectable_roots):
                if s == mp.inf:
                    weighted.append(s)
                    continue
                weighted.append(s * abs(base ** (mp.mpc(root) - shift_c)))
            scores = weighted

    def _select_kept(cap: int) -> DampedRational:
        # Shared dispatch with the global-block builder (the `scores` here are
        # the residue / contribution-weighted scores prepared above).
        if pole_selection == "contribution_optimal":
            return _take_contribution_optimal_prefix(
                shifted, greedy_order or (), cap
            )
        return select_kept_poles(shifted, cap, pole_selection, scores=scores)

    fit_exacts = exacts if split is None else {coord: remainder for coord, (remainder, _parts) in split.items()}
    # over the KEPT entries only: a folded table's grid also holds (m, n) outside m + 2n <= Lambda_+-, of higher
    # degree, which padded every xt/ws block to degree 41 instead of 27 at Lambda 7 (3-4x the SDPB time)
    max_num_deg = max(
        _kept_max_numerator_degree(
            exact, None if kept_entries_by_coord is None else kept_entries_by_coord.get(coord))
        for coord, exact in fit_exacts.items()
    )
    # `denom_full_degree` is the *total* surviving denominator including the
    # forced poles (unless they were split off, below) -- it sets the
    # polynomial-degree budget (a forced pole reduces the polynomial degree by
    # one exactly as if a selector had picked it).
    denom_full_degree = len(all_roots)
    forced_pole_count = len(forced_roots)
    if split is not None:
        # the remainder's degree budget: its denominator is the selectable roots alone
        denom_full_degree -= forced_pole_count
        forced_pole_count = 0
        if n_sample_points is not None:
            n_sample_points = max(1, n_sample_points - len(forced_roots))

    sample_kwargs = dict(sample_point_kwargs or {})
    if "work_dps" not in sample_kwargs:
        sample_kwargs["work_dps"] = (
            node_search_dps if node_search_dps is not None
            else _default_node_search_dps()
        )

    last_sample_error = None
    kept = None
    sample_x = None
    chosen_n_sample_points = None
    dropped_pole_count = None
    # Retrying with fewer poles is permitted for an integer cap, but a keep-all request must fail if its
    # full prefactor has no nodes: reducing it would silently change the requested rational function.
    kept_caps = (npoles,) if keep_all else range(npoles, -1, -1)
    for kept_cap in kept_caps:
        candidate = _select_kept(kept_cap)
        # `kept_combined_degree` mirrors the legacy formula: every kept pole
        # (selectable or forced) contributes one polynomial degree.
        kept_combined_degree = candidate.pole_degree + forced_pole_count
        candidate_dropped_pole_count = denom_full_degree - kept_combined_degree
        candidate_n_sample_points = n_sample_points
        if candidate_n_sample_points is None:
            candidate_n_sample_points = max(
                1, max_num_deg - candidate_dropped_pole_count + 1
            )
        try:
            candidate_sample_x = get_sample_points(
                candidate, candidate_n_sample_points, **sample_kwargs
            )
        except ValueError as exc:
            # `Re(pole) <= 0` (a real pole inside the support) and an unremoved
            # near-zero Kac pole (an unregularized table) are both malformed-
            # prefactor errors that keeping fewer poles cannot fix -- propagate
            # them instead of silently working around by dropping poles.
            if "Re(pole) <= 0" in str(exc) or _UNREMOVED_KAC_ZERO_MARKER in str(exc):
                raise
            last_sample_error = exc
            continue
        kept = candidate
        sample_x = candidate_sample_x
        chosen_n_sample_points = candidate_n_sample_points
        dropped_pole_count = candidate_dropped_pole_count
        break

    if (
        kept is None
        or sample_x is None
        or chosen_n_sample_points is None
        or dropped_pole_count is None
    ):
        if last_sample_error is not None:
            raise last_sample_error
        raise ValueError("failed to construct an admissible interpolation prefactor")
    if chosen_n_sample_points < 1:
        raise ValueError("n_sample_points must be >= 1")
    real_shift = mp.mpf(mp.re(shift))
    sample_delta = _shifted_sampling_points(sample_x, shift)

    if min_pole_distance is None:
        # arb prefactor values for the arb pipeline; the mpmath values only when there are none (a complex
        # constant) or the table values come back as mpmath
        pref_values_arb_cached = kept.evaluate_arb_batch(sample_x)
        pref_values = None if pref_values_arb_cached is not None else [kept.evaluate(x) for x in sample_x]
    else:
        pref_values = [
            kept.evaluate_regulated(x, min_pole_distance) for x in sample_x
        ]
        pref_values_arb_cached = None

    # Fallback when the forced poles were not split off as principal parts:
    # pre-multiply the exact table by the forced denominator at each sample
    # point: E_div(Δ) = exact(Δ) · ∏(Δ − p_F). The Lagrange fit then runs on
    # E_div / kept_select, and the combined prefactor (kept_select · 1/D_F)
    # reintroduces the forced poles in the final reconstruction. BS never
    # sees them so its real-positive-pole guard is satisfied.
    if forced_roots and split is None:
        forced_factor_at_sample = [
            _product_over_roots(d, forced_roots) for d in sample_delta
        ]
    else:
        forced_factor_at_sample = None

    # Combined prefactor: selectable-kept poles plus *all* forced poles
    # (shifted to the same x = Δ − shift variable), whether they were split
    # off as principal parts or pre-multiplied above. Real positive poles are
    # accepted by DampedRational; only BS rejects them, and we never feed
    # this combined object to BS.
    if forced_roots:
        kept_combined = DampedRational(
            constant=kept.constant,
            poles=tuple(kept.poles) + tuple(p - real_shift for p in forced_roots),
            base=kept.base,
        )
    else:
        kept_combined = kept

    # Per-entry polynomial degrees. For an (m, n) entry whose exact numerator
    # polynomial has degree d_{m,n}, the rational function being interpolated
    # has polynomial part of degree d_{m,n} - dropped_pole_count at Δ → ∞.
    # Using exactly that degree per entry preserves the large-Δ scaling of
    # the underlying block, which is what SDPB's matrix-model positivity
    # proof relies on. A uniform-degree fit (max_num_deg - dropped across
    # every entry) over-fits the rational tail in low-degree entries with
    # high-order coefficients that diverge wildly past the BS sample range
    # -- empirically by 12 orders of magnitude at Δ = 200 for c=1/2 Ising
    # ell=0 (0, 0).
    #
    # To match each entry's natural degree, we re-run BS node generation
    # for each distinct target degree across the table (cached). At
    # lam=7 / q_order=8 there are ~14 distinct counts; BS is sub-second
    # per run at dps=80, so the overhead is small relative to the exact-
    # table evaluation cost.
    forced_factor_full = forced_factor_at_sample

    # Build the per-target job list per coordinate. Each distinct sample-
    # point count `n_use` defines one job. Jobs are independent (no shared
    # cache state across counts), so they're a natural fit for
    # multiprocessing pool when `num_threads > 1`.
    out = {}
    for coord, exact in fit_exacts.items():
        # Reuse flint evaluation structures (numerator coefficient matrix,
        # denominator polynomial, exp base) across all sample-count jobs for
        # this exact table.
        flint_table_cache = {}
        basis_cache = {}
        # The blocks_3d --lambda truncation keeps only the m + n <= Lambda
        # (non-folded) / m + 2n <= Lambda (folded) derivative triangle; the rest
        # of the square is discarded by every writer. Compute only the kept set
        # so the folded chiral series shrinks (its length tracks the max source
        # degree of the requested entries) and the corner entries are skipped.
        coord_kept = (
            kept_entries_by_coord.get(coord)
            if kept_entries_by_coord is not None
            else None
        )
        # the block's own entries (a remainder entry of degree 0 is a constant, not an empty placeholder)
        coord_entries = entries_by_coord[coord] if split is not None else _interpolation_entries(exact, coord_kept)
        entries_by_target: dict[int, list] = {}
        for (m, n) in coord_entries:
            entry_degree = exact.numerator_degree(m, n)
            target_degree = max(0, entry_degree - dropped_pole_count)
            # `shared_sample_points` collapses every entry onto the single
            # max-count node set (the `chosen_n_sample_points` group) instead
            # of giving each entry its own degree-matched Bohr-Sommerfeld
            # node set. It trades the per-entry optimal conditioning for a
            # single shared block-evaluation pass; see the per-entry vs
            # shared accuracy comparison in tests/diagnostics.
            if shared_sample_points:
                n_use = chosen_n_sample_points
            else:
                n_use = target_degree + 1
            if n_use <= 0:
                raise ValueError("target sample count must be >= 1")
            entries_by_target.setdefault(n_use, []).append((m, n))

        # Pre-compute the max-count job's outputs synchronously: the max-
        # count BS run has already happened (it's `sample_delta`, etc.),
        # so we can reuse those rather than re-running BS in a worker.
        max_count_entries = entries_by_target.pop(chosen_n_sample_points, [])
        polynomial_map: dict = {}
        if max_count_entries:
            sample_tuple_max = tuple(sample_delta)
            tables_max, is_arb_max = _evaluate_table_flint_many(
                exact,
                sample_delta,
                entries=max_count_entries,
                flint_table_cache=flint_table_cache,
            )
            basis_key_max = (sample_tuple_max, shift)
            basis_max, denoms_max, arb_basis_max, arb_denoms_max = basis_cache.get(
                basis_key_max,
                (None, None, None, None),
            )
            if basis_max is None:
                basis_max, denoms_max, arb_basis_max, arb_denoms_max = precompute_lagrange_basis(
                    sample_x,
                )
                basis_cache[basis_key_max] = (
                    basis_max,
                    denoms_max,
                    arb_basis_max,
                    arb_denoms_max,
                )
            # If arb values, do division in arb space and pass arb directly
            if is_arb_max:
                try:
                    from flint import arb as arb_type, ctx as flint_ctx
                    prec_bits = int(mp.mp.dps * 3.33) + 20
                    flint_ctx.prec = prec_bits
                    if pref_values_arb_cached is not None:
                        pref_values_arb = pref_values_arb_cached
                    else:
                        pref_values_arb = [arb_type(v) for v in pref_values]
                    if forced_factor_full is not None:
                        forced_factor_arb = [arb_type(v) for v in forced_factor_full]
                    else:
                        forced_factor_arb = None
                    use_arb_max = True
                except TypeError:
                    # arb() rejects mpc prefactor / forced-factor values (a
                    # complex constant or conjugate forced roots): mpmath division.
                    use_arb_max = False
            else:
                use_arb_max = False
            if not use_arb_max and pref_values is None:
                pref_values = [kept.evaluate(x) for x in sample_x]

            for (m, n) in max_count_entries:
                if use_arb_max:
                    if forced_factor_arb is None:
                        reduced = [
                            tables_max[i][m][n] / pref_values_arb[i]
                            for i in range(len(sample_delta))
                        ]
                    else:
                        reduced = [
                            tables_max[i][m][n] * forced_factor_arb[i]
                            / pref_values_arb[i]
                            for i in range(len(sample_delta))
                        ]
                    # Pass arb values directly — no mpf round-trip
                else:
                    if forced_factor_full is None:
                        reduced = [
                            tables_max[i][m][n] / pref_values[i]
                            for i in range(len(sample_delta))
                        ]
                    else:
                        reduced = [
                            tables_max[i][m][n] * forced_factor_full[i]
                            / pref_values[i]
                            for i in range(len(sample_delta))
                        ]
                poly = lagrange_polynomial_from_basis(
                    basis_max, denoms_max, reduced,
                    _arb_basis=arb_basis_max, _arb_denoms=arb_denoms_max,
                )
                polynomial_map[(m, n)] = (poly, sample_tuple_max, _raw_entry_samples(tables_max, is_arb_max, m, n, reduced))

        # Remaining per-target jobs go through the worker pool (or run
        # serially when num_threads == 1).
        if entries_by_target:
            worker_count = max(1, min(usable_workers(num_threads), len(entries_by_target)))
            local_table_cache = flint_table_cache if worker_count == 1 else None
            local_basis_cache = basis_cache if worker_count == 1 else None
            jobs = [
                (
                    mp.mp.dps, n_use, kept, forced_roots if split is None else [], shift, real_shift,
                    sample_kwargs, min_pole_distance, exact, entries,
                    local_table_cache,
                    local_basis_cache,
                    node_subset_mode,
                    tuple(mp.mpf(x) for x in sample_x),
                )
                for n_use, entries in entries_by_target.items()
            ]
            if worker_count == 1:
                target_results = [
                    _interpolate_per_target_task(job) for job in jobs
                ]
            else:
                context = multiprocessing.get_context()
                with context.Pool(processes=worker_count, **pool_initializer_kwargs()) as pool:
                    target_results = list(
                        pool.map(_interpolate_per_target_task, jobs)
                    )
            for sub in target_results:
                polynomial_map.update(sub)

        # Reassemble per-(m, n) grids. Entries outside the kept triangle were
        # never interpolated (every writer discards them under the --lambda
        # truncation); fill them with a zero placeholder so the grid keeps its
        # rectangular shape for any consumer that indexes it positionally.
        placeholder = (Polynomial.zero(field=mp.mpf), tuple(sample_delta), None)
        numerators = []
        per_entry_samples = []
        per_entry_values = []
        for m in range(exact.m_max + 1):
            row = []
            sample_row = []
            value_row = []
            for n in range(exact.n_max + 1):
                poly, sample_tuple, values = polynomial_map.get((m, n), placeholder)
                row.append(poly)
                sample_row.append(sample_tuple)
                value_row.append(values)
            numerators.append(row)
            per_entry_samples.append(sample_row)
            per_entry_values.append(value_row)

        out[coord] = InterpolatedDeltaTable(
            prefactor=kept_combined if split is None else kept,
            shift=shift,
            sample_points=tuple(sample_delta),  # max-count set; per-entry
                                                # sets in numerator_sample_points
            numerators=numerators,
            min_pole_distance=min_pole_distance,
            numerator_sample_points=per_entry_samples,
            numerator_sample_values=per_entry_values,
        )
        if split is not None:
            out[coord] = _recombine_principal_parts(out[coord], split[coord][1], forced_roots)
    return out


def _conjugate_nested(data):
    """The elementwise complex conjugate of nested lists/tuples of flint acb values (same nesting, plain types)."""
    if isinstance(data, list):
        return [_conjugate_nested(item) for item in data]
    if isinstance(data, tuple):
        return tuple(_conjugate_nested(item) for item in data)
    return data.conjugate()


class _ReflectedFactor(tuple):
    """Chiral-factor data at ``conj(x)`` taken as the exact conjugate of ``source``, the cached data at ``x``."""

    def __new__(cls, data, source):
        obj = super().__new__(cls, data)
        obj.source = source
        return obj


def _analytic_chiral_factor(
    evaluator,
    n_terms,
    *,
    deriv_factor,
    principal_order=1,
    regular_order=0,
):
    """Build a memoized chiral-factor callable ``f(x) -> (A, B)`` for the
    analytic principal-part scorer, or ``None`` if ``evaluator`` cannot supply
    closed-form pole data (unsupported context mode).

        For a chiral argument ``x`` that lands on a Kac weight the factor returns its
        Laurent/Taylor data around ``x`` in the score variable:

            F = sum_{j>=1} A[j-1] / (var-loc)^j + sum_{q>=0} B[q] (var-loc)^q,

        where ``A`` has ``principal_order`` rows and ``B`` has
        ``regular_order + 1`` rows (each row length ``n_terms``). ``A`` includes the
        Jacobian conversion factor ``deriv_factor^j`` for ``x(var)``. For a regular
        ``x`` it returns zero principal rows and ``B[0]=value``.

        Backward compatibility: with ``principal_order=1, regular_order=0`` callers
        get the original simple-pole data layout ``([A0], [B0])``."""
    pole_data = getattr(evaluator, "pole_data", None)
    kac_locations = getattr(evaluator, "kac_locations", None)
    # pole_data returns None both for "unsupported mode" and "x is not a Kac
    # pole"; kac_locations returns None exactly for the unsupported modes
    # (anything but strip-real -- real_nonstrip at c>25, generic-complex c), so
    # gate on it up front lest an unsupported mode score every factor as regular.
    kac_locs = kac_locations() if kac_locations is not None else None
    if pole_data is None or kac_locs is None:
        return None
    from flint import acb

    principal_order = int(principal_order)
    regular_order = int(regular_order)
    if principal_order < 1 or regular_order < 0:
        return None

    df = _mpf_to_acb(mp.mpf(deriv_factor))
    df_pows = [acb(1)]
    for _ in range(principal_order):
        df_pows.append(df_pows[-1] * df)
    df_pows_reg = [acb(1)]
    for _ in range(regular_order):
        df_pows_reg.append(df_pows_reg[-1] * df)
    # Persist the per-argument Laurent data on the evaluator rather than in a
    # closure-local dict: the factor callable is rebuilt on every scorer call,
    # but the evaluator (now shared across the spin family) is not, and the
    # expensive singular-side arguments -- the Kac weights h_{r,s} -- are the
    # same for every spin. Keyed by everything that shapes the output
    # (n_terms slice, Jacobian factor, Laurent orders); the working precision
    # at call time keys the inner per-argument dict, since ``pole_laurent``
    # computes at the precision current when it is called.
    store = getattr(evaluator, "_analytic_factor_cache", None)
    if store is None:
        store = {}
        try:
            evaluator._analytic_factor_cache = store
        except AttributeError:
            store = {}
    shape = (int(n_terms), _mpf_cache_key(mp.mpf(deriv_factor)), principal_order, regular_order)
    cache = store.setdefault(shape, {})
    compat_simple = (principal_order == 1 and regular_order == 0)
    # Poles off the real axis come in conjugate pairs, and so do their chiral arguments; for a real-analytic
    # evaluator the data at conj(x) is the conjugate of the data at x (the Jacobian factor is real), so each pair
    # costs one evaluation.  A pair need not be conjugate to the last digit, so the partner is found through a float
    # index of the evaluated arguments and accepted within roundoff (the data then agree to that roundoff).
    reflect = bool(getattr(evaluator, "real_analytic", False))
    evaluated = store.setdefault(("evaluated arguments",) + shape, {})   # (prec, complex(x)) -> (key, x)

    def factor(x):
        x = mp.mpc(x)
        key = (mp.mp.prec, _mpc_cache_key(x))
        hit = cache.get(key)
        if hit is not None:
            return hit
        if reflect and x.imag != 0:
            twin = mp.conj(x)
            mirror = cache.get((mp.mp.prec, _mpc_cache_key(twin)))
            if mirror is None:
                near = evaluated.get((mp.mp.prec, complex(twin)))
                if near is not None and abs(near[1] - twin) <= mp.mpf(10) ** (5 - mp.mp.dps) * max(1, abs(twin)):
                    mirror = cache.get(near[0])
            if mirror is not None:
                # the reflection of a reflection is its source itself
                out = mirror.source if isinstance(mirror, _ReflectedFactor) else _ReflectedFactor(
                    _conjugate_nested(mirror), mirror)
                cache[key] = out
                return out
        pole_laurent = getattr(evaluator, "pole_laurent", None)
        if pole_laurent is not None and regular_order > 0:
            # a product with a higher-order pole of the other factor reads this one's Taylor rows past its value, so a
            # regular argument needs them too (the value-only branch below would leave them zero)
            pd = pole_laurent(
                x,
                principal_order=principal_order,
                regular_order=regular_order,
                allow_regular_point=True,
            )
        elif pole_laurent is not None:
            pd = pole_laurent(
                x,
                principal_order=principal_order,
                regular_order=regular_order,
            )
        else:
            pd = pole_data(x)
            if pd is not None:
                res, reg = pd
                pd = ([res], [reg])

        if pd is None:
            v = evaluator(mp.mpc(x), return_acb=True)
            out = (
                [[acb(0)] * n_terms for _ in range(principal_order)],
                [[v[i] if q == 0 else acb(0) for i in range(n_terms)]
                 for q in range(regular_order + 1)],
            )
        else:
            principal, regular = pd
            if len(principal) < principal_order or len(regular) < regular_order + 1:
                return None
            A = []
            for j in range(principal_order):
                A.append([
                    df_pows[j + 1] * principal[j][i]
                    for i in range(n_terms)
                ])
            # (x - loc)^q = deriv_factor^-q (var - loc)^q: the regular rows past the value take the inverse Jacobian
            B = [
                [regular[q][i] if q == 0 else regular[q][i] / df_pows_reg[q] for i in range(n_terms)]
                for q in range(regular_order + 1)
            ]
            out = (A, B)
        if compat_simple:
            out = (out[0][0], out[1][0])
        cache[key] = out
        evaluated[(mp.mp.prec, complex(x))] = (key, x)
        return out

    from bisect import bisect_left

    kac_f = sorted(float(v) for v in kac_locs)

    def prefetch(xs):
        """Warm the cache for regular (non-Kac) real arguments in one batched
        ``evaluate_many`` call. The factor otherwise resolves each regular
        argument with a full single-point chiral evaluation; the scorer knows
        every pole location up front, so the whole regular side is one batched
        arb matmul instead. Purely a cache warm: complex arguments, near-Kac
        arguments (coarse float gap, same scale as ``pole_laurent``'s) and any
        batch the ``many`` path declines are left to the exact per-point path.
        """
        many = getattr(evaluator, "many", None)
        if many is None:
            return
        prec = mp.mp.prec
        todo_keys = []
        todo_args = []
        seen = set()
        for x in xs:
            xc = mp.mpc(x)
            if xc.imag != 0:
                continue
            key = (prec, _mpc_cache_key(xc))
            if key in cache or key in seen:
                continue
            xf = float(xc.real)
            j = bisect_left(kac_f, xf)
            if (j < len(kac_f) and kac_f[j] - xf <= 1e-6) or (
                j > 0 and xf - kac_f[j - 1] <= 1e-6
            ):
                continue
            seen.add(key)
            todo_keys.append(key)
            todo_args.append(mp.mpf(xc.real))
        if not todo_keys:
            return
        rows = many(todo_args)
        if rows is None:
            return
        for key, row in zip(todo_keys, rows):
            B0 = [
                acb(row[i]) if i < len(row) else acb(0)
                for i in range(n_terms)
            ]
            out = (
                [[acb(0)] * n_terms for _ in range(principal_order)],
                [
                    B0 if q == 0 else [acb(0)] * n_terms
                    for q in range(regular_order + 1)
                ],
            )
            if compat_simple:
                out = (out[0][0], out[1][0])
            cache[key] = out

    factor.prefetch = prefetch
    return factor


def _prefetch_chiral_factor(fac, roots, ell, halves=(1, -1)):
    """Warm an analytic-scorer chiral factor's cache for every pole group.

    Collects the ``(loc + half * ell) / 2`` arguments over all distinct pole
    locations and hands them to ``fac.prefetch`` so the regular (non-Kac) side
    is evaluated in one batched call rather than point by point. ``halves``
    names which chiral streams this factor serves (both on the diagonal, one
    each for split h / hbar factors). No-op when the factor has no prefetch."""
    prefetch = getattr(fac, "prefetch", None)
    if prefetch is None:
        return
    locs = [
        loc for loc, _ in _group_roots_by_location(
            [mp.mpc(r) for r in roots], tolerance(guard=10)
        )
    ]
    args = []
    for loc in locs:
        for half in halves:
            args.append((loc + half * ell) / 2)
    prefetch(args)


class _OrderTwoPrincipalRows(Sequence):
    """The ``(g_{-1}, g_{-2})`` rows of one pole location of order <= 2, kept as the chiral factors they come from.

    With ``e = exp_base^-loc`` and ``L = ln exp_base``, ``g_{-2} = e C2`` and ``g_{-1} = e (C1 - L C2)``, where ``C1``
    and ``C2`` are the bilinear chiral products of :func:`_analytic_principal_parts_axis`.  The rows themselves are
    formed on first access (the forced-pole split and the residue scores read them).  The contribution score needs
    only ``max_e |g_{-1} / D + g_{-2} / D^2| = |e / D| max_e |C1 + (1/D - L) C2|``, and that matrix is ``U V`` with
    ``U = [A^{m+}, B^{m+}, s A^{m-}, s B^{m-}]`` and ``V = [B^{n-} + k A^{n-}; A^{n-}; B^{n+} + k A^{n+}; A^{n+}]``
    (``k = 1/D - L``): one flint product of inner dimension 4 (2 without the swap term) instead of per-entry sums.

    For a folded coordinate (``fold``, the plan of :func:`_analytic_principal_parts_folded`) the entries are the
    folded ``(a, b)``, fixed integer combinations of the source ``(m, n)``: the same product is folded before the max,
    and the rows are formed as that function's mpc rows.
    """

    def __init__(self, entry_pairs, factors, sigma, log_base, scale, *, roots, mirror_of=None, fold=None):
        self._pairs = entry_pairs
        self._factors = factors
        self._sigma = sigma
        self._log_base = log_base
        self.scale = scale
        self.roots = roots
        # the rows of the conjugate location when these are their exact conjugates (see the axis function)
        self.mirror_of = mirror_of
        self._fold = fold
        self._rows = None

    def _materialized(self):
        if self._rows is None:
            Amp, Bmp, Amm, Bmm, Anp, Bnp, Anm, Bnm = self._factors
            sigma, log_base, scale = self._sigma, self._log_base, self.scale
            products = []
            for (m, n) in (self._pairs if self._fold is None else self._fold.src_entries):
                C2 = Amp[m] * Anm[n]
                C1 = Amp[m] * Bnm[n] + Anm[n] * Bmp[m]
                if sigma is not None:
                    C2 = C2 + sigma * (Amm[m] * Anp[n])
                    C1 = C1 + sigma * (Amm[m] * Bnp[n] + Anp[n] * Bmm[m])
                products.append((C2, C1))
            if self._fold is None:
                self._rows = [(scale * (C1 - log_base * C2), scale * C2) for C2, C1 in products]
            else:
                from .polynomial import _acb_to_mpc as _exact_mpc

                fold = self._fold
                c2_map = {entry: C2 for entry, (C2, _C1) in zip(fold.src_entries, products)}
                c1_map = {entry: C1 for entry, (_C2, C1) in zip(fold.src_entries, products)}
                C2f, C1f = _fold_pp_grids_acb(fold.plan, fold.target_size, fold.width, c2_map, c1_map,
                                              fold.m_max, fold.n_max)
                e, L = _exact_mpc(scale), _exact_mpc(log_base)
                self._rows = [(e * (C1f[a][b] - L * C2f[a][b]), e * C2f[a][b]) for (a, b) in self._pairs]
        return self._rows

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, index):
        return self._materialized()[index]

    def peak(self, inv):
        """``max_e |g_{-1} inv + g_{-2} inv^2|`` (flint arb) for ``inv = 1 / D``."""
        from flint import acb_mat

        Amp, Bmp, Amm, Bmm, Anp, Bnp, Anm, Bnm = self._factors
        k = inv - self._log_base
        columns = [Amp, Bmp]
        rows = [[b + k * a for a, b in zip(Anm, Bnm)], Anm]
        if self._sigma is not None:
            columns += [[self._sigma * a for a in Amm], [self._sigma * b for b in Bmm]]
            rows += [[b + k * a for a, b in zip(Anp, Bnp)], Anp]
        values = (acb_mat([list(u) for u in zip(*columns)]) * acb_mat(rows)).tolist()
        if self._fold is None:
            return abs(self.scale * inv) * max(abs(values[m][n]) for (m, n) in self._pairs)
        folded = [0] * self._fold.target_size
        for (m, n, contribs) in self._fold.plan:
            value = values[m][n]
            for target, weight in contribs:
                folded[target] = folded[target] + value * weight
        width = self._fold.width
        return abs(self.scale * inv) * max(abs(folded[a * width + b]) for (a, b) in self._pairs)


class _FoldPlan(NamedTuple):
    """A parity fold of the source ``(m, n)`` entries (:func:`derivs._fold_plan`) and the entries it reads."""
    plan: list
    target_size: int
    width: int
    m_max: int
    n_max: int
    src_entries: list


def _analytic_principal_parts_axis(*args, **kwargs):
    """Principal-part extraction evaluates near poles on purpose: no conditioning report inside."""
    from .conditioning import handled

    with handled():
        return _analytic_principal_parts_axis_impl(*args, **kwargs)


def _analytic_principal_parts_axis_impl(roots, entries, exp_base, *,
                                   factor_m_plus, factor_m_minus,
                                   factor_n_plus, factor_n_minus, sigma):
    """Exact principal parts of the prefactor-stripped block at each pole location, or ``None``.

    The block factorizes as ``B[m][n] = c_h_m[m] c_hbar_n[n] + σ c_hbar_m[m] c_h_n[n]``: the m-index factors come
    from the m-center evaluator at the chiral arguments ``(Δ+ℓ)/2`` (``factor_m_plus``) and ``(Δ-ℓ)/2``
    (``factor_m_minus``), the n-index factors from the n-center evaluator (the diagonal block uses one evaluator
    for both).  Near a pole ``loc`` each chiral factor is ``A_k/(var-loc) + B_k`` (or its order-M Laurent rows), so
    the block's Laurent coefficients are

        C2[m][n] = A^{m+}_m A^{n-}_n + σ A^{m-}_m A^{n+}_n
        C1[m][n] = A^{m+}_m B^{n-}_n + A^{n-}_n B^{m+}_m + σ (A^{m-}_m B^{n+}_n + A^{n+}_n B^{m-}_m)

    and, stripping ``exp_base^Δ``, ``g_{-1} = exp_base^{-loc} (C1 - ln(exp_base) C2)``, ``g_{-2} = exp_base^{-loc}
    C2`` (the general order-M path convolves the Laurent rows).  So ``B(Δ) = exp_base^Δ (Σ_a g_{-a} (Δ-loc)^{-a}
    + regular)``.  Returns ``[(loc, root_indices, rows)]`` with ``rows[e] = (g_{-1}, g_{-2}, ...)`` for
    ``entries[e]`` (flint ``acb`` on the order <= 2 path, mpc otherwise); ``None`` when a factor callable
    declines.  ``sigma`` ``None``/``0`` drops the swap term (ordered B).  The order <= 2 rows are an
    :class:`_OrderTwoPrincipalRows`, formed only when read: the contribution score works from the chiral factors
    directly, and a location whose factors are the reflections of an earlier one's is marked as its mirror.
    """
    entry_pairs = [(int(m), int(n)) for (m, n) in entries]
    sigma_c = None if sigma in (None, 0) else _mpf_to_acb(sigma)
    roots_mpc = [mp.mpc(r) for r in roots]
    groups = _group_roots_by_location(roots_mpc, tolerance(guard=10))
    exp_base_c = mp.mpc(exp_base)
    ndig = mp.mp.dps + 5
    Lc = _mpf_to_acb(mp.log(exp_base_c))
    # With a real positive base and a real swap coefficient, a location whose chiral factors are all the exact
    # conjugates of another's (_ReflectedFactor) has the conjugate principal part, so the same contribution score.
    reflecting = exp_base_c.imag == 0 and exp_base_c.real > 0 and (sigma_c is None or sigma_c.imag.is_zero())
    rows_by_factors = {}
    parts = []
    for loc, idxs in groups:
        fmp = factor_m_plus(loc)
        fmm = factor_m_minus(loc)
        fnp = factor_n_plus(loc)
        fnm = factor_n_minus(loc)
        if fmp is None or fmm is None or fnp is None or fnm is None:
            return None
        factor_ids = (id(fmp), id(fmm), id(fnp), id(fnm))
        sources = tuple(getattr(f, "source", None) for f in (fmp, fmm, fnp, fnm))
        mirror_of = rows_by_factors.get(tuple(map(id, sources))) if (
            reflecting and all(src is not None for src in sources)) else None
        Amp, Bmp = fmp
        Amm, Bmm = fmm
        Anp, Bnp = fnp
        Anm, Bnm = fnm
        simple_shape = all(
            series and not isinstance(series[0], (list, tuple))
            for series in (Amp, Amm, Anp, Anm, Bmp, Bmm, Bnp, Bnm)
        )
        M = len(idxs)
        if simple_shape and M <= 2:
            if not entry_pairs:
                parts.append((loc, idxs, []))
                continue
            if mirror_of is not None and len(mirror_of.roots) == M:
                ebloc = mirror_of.scale.conjugate()     # exp_base^-conj(loc) for a real positive base
            else:
                ebloc, mirror_of = _mpf_to_acb(exp_base_c ** (-loc)), None
            rows = _OrderTwoPrincipalRows(
                entry_pairs, (Amp, Bmp, Amm, Bmm, Anp, Bnp, Anm, Bnm), sigma_c, Lc, ebloc,
                roots=idxs, mirror_of=mirror_of,
            )
            rows_by_factors[factor_ids] = rows
            parts.append((loc, idxs, rows))
            continue

        Amp, Amm, Anp, Anm, Bmp, Bmm, Bnp, Bnm = (
            [series] if series and not isinstance(series[0], (list, tuple)) else series
            for series in (Amp, Amm, Anp, Anm, Bmp, Bmm, Bnp, Bnm)
        )
        fact_inv = [mp.mpf(1)]
        for k in range(1, M + 1):
            fact_inv.append(fact_inv[-1] / mp.mpf(k))
        minusL = -mp.log(mp.mpc(exp_base))
        pow_minusL = [mp.mpc(1)]
        for _ in range(1, M):
            pow_minusL.append(pow_minusL[-1] * minusL)
        ebloc = exp_base_c ** (-loc)

        def _A(series, p, idx):
            if p < 1 or p > len(series):
                return mp.mpc(0)
            return mp.mpc(_acb_to_mpc(series[p - 1][idx], ndig))

        def _B(series, q, idx):
            if q < 0 or q >= len(series):
                return mp.mpc(0)
            return mp.mpc(_acb_to_mpc(series[q][idx], ndig))

        sig = None if sigma_c is None else mp.mpc(_acb_to_mpc(sigma_c, ndig))
        # the highest principal order a factor carries: x^-j collects A_p B_{p-j} for every p >= j up to it
        top = max(len(series) for series in (Amp, Amm, Anp, Anm))
        rows = []
        for (m, n) in entry_pairs:
            C = [mp.mpc(0)] * (M + 1)  # 1-based principal coefficients
            for j in range(1, M + 1):
                acc = mp.mpc(0)
                # principal-principal, then principal-regular + regular-principal
                for p in range(1, j):
                    acc += _A(Amp, p, m) * _A(Anm, j - p, n)
                for p in range(j, max(j, top) + 1):
                    acc += _A(Amp, p, m) * _B(Bnm, p - j, n)
                    acc += _A(Anm, p, n) * _B(Bmp, p - j, m)
                if sig is not None:
                    swap = mp.mpc(0)
                    for p in range(1, j):
                        swap += _A(Amm, p, m) * _A(Anp, j - p, n)
                    for p in range(j, max(j, top) + 1):
                        swap += _A(Amm, p, m) * _B(Bnp, p - j, n)
                        swap += _A(Anp, p, n) * _B(Bmm, p - j, m)
                    acc += sig * swap
                C[j] = acc
            rows.append(tuple(
                ebloc * mp.fsum(pow_minusL[k] * fact_inv[k] * C[a + k] for k in range(0, M - a + 1))
                for a in range(1, M + 1)
            ))
        parts.append((loc, idxs, rows))
    return parts


def _scores_from_principal_parts(parts, n_roots, delta_eval):
    """``max_e |Σ_a g_{-a} / D^a|`` per root, ``D = delta_eval - loc`` (the analytic ``contribution`` score); a pole
    at ``delta_eval`` itself scores infinite.  ``parts`` as returned by :func:`_analytic_principal_parts_axis`: its
    order <= 2 rows score in flint from their chiral factors (:meth:`_OrderTwoPrincipalRows.peak`), the mpc rows of the
    higher-order / folded / merged-Laurent scorers in mpmath."""
    deval = mp.mpc(delta_eval)
    tol_d = mp.mpf("1e-12")
    scores = [mp.mpf(0)] * n_roots
    peaks = {}  # id(rows) -> score, for the mirrors (conjugate locations) at a real evaluation point
    for loc, idxs, rows in parts:
        D = deval - loc
        if abs(D) <= tol_d:
            best = mp.inf
        elif isinstance(rows, _OrderTwoPrincipalRows):
            partner = rows.mirror_of
            if partner is not None and deval.imag == 0 and id(partner) in peaks:
                best = peaks[id(partner)]
            else:
                best = _arb_to_mpf(rows.peak(1 / _mpf_to_acb(D)))
            peaks[id(rows)] = best
        else:
            best = max((abs(mp.fsum(g / D ** (a + 1) for a, g in enumerate(row))) for row in rows),
                       default=mp.mpf(0))
        for i in idxs:
            scores[i] = best
    return scores


def _leading_scores_from_principal_parts(parts, n_roots):
    """The leading Laurent-coefficient magnitude per root (``|g_{-1}|`` simple, ``|g_{-2}|`` double pole), maxed
    over the entries; ``None`` for a higher-order pole (not expected for the factorized global/tower blocks)."""
    scores = [mp.mpf(0)] * n_roots
    for _loc, idxs, rows in parts:
        mult = len(idxs)
        if mult > 2:
            return None
        best = max((abs(mp.mpc(row[mult - 1]) if isinstance(row[mult - 1], (mp.mpf, mp.mpc))
                        else _acb_to_mpc(row[mult - 1], mp.mp.dps + 5)) for row in rows), default=mp.mpf(0))
        for i in idxs:
            scores[i] = best
    return scores


def _analytic_pp_scores_axis(roots, delta_eval, entries, exp_base, **factors):
    """Analytic ``contribution`` scores from :func:`_analytic_principal_parts_axis`, or ``None``."""
    parts = _analytic_principal_parts_axis(roots, entries, exp_base, **factors)
    return None if parts is None else _scores_from_principal_parts(parts, len(roots), delta_eval)


def _analytic_residue_leading_scores_axis(roots, entries, exp_base, **factors):
    """Leading-Laurent residue scores from :func:`_analytic_principal_parts_axis`, or ``None``."""
    parts = _analytic_principal_parts_axis(roots, entries, exp_base, **factors)
    return None if parts is None else _leading_scores_from_principal_parts(parts, len(roots))


def _diagonal_factors(factor_h, factor_hbar):
    """On the diagonal the m- and n-index share one chiral evaluator: the ``+ℓ/2`` stream serves both plus axes."""
    return dict(factor_m_plus=factor_h, factor_m_minus=factor_hbar, factor_n_plus=factor_h, factor_n_minus=factor_hbar)


def _analytic_principal_parts(*args, **kwargs):
    """Principal-part extraction evaluates near poles on purpose: no conditioning report inside."""
    from .conditioning import handled

    with handled():
        return _analytic_principal_parts_impl(*args, **kwargs)


def _analytic_principal_parts_impl(roots, entries, exp_base, *, factor_h, factor_hbar, sigma):
    """Diagonal :func:`_analytic_principal_parts_axis` (scalar-G, ordered / parity-projected B, the tower)."""
    return _analytic_principal_parts_axis(roots, entries, exp_base, sigma=sigma,
                                          **_diagonal_factors(factor_h, factor_hbar))


def _analytic_residue_leading_scores(roots, entries, exp_base, *, factor_h, factor_hbar, sigma):
    """Diagonal leading-residue scorer."""
    return _analytic_residue_leading_scores_axis(roots, entries, exp_base, sigma=sigma,
                                                 **_diagonal_factors(factor_h, factor_hbar))


def _analytic_pp_scores(roots, delta_eval, entries, exp_base, *, factor_h, factor_hbar, sigma):
    """Diagonal analytic ``contribution`` scorer."""
    return _analytic_pp_scores_axis(roots, delta_eval, entries, exp_base, sigma=sigma,
                                    **_diagonal_factors(factor_h, factor_hbar))


def _fold_pp_grids_acb(plan, target_size, width, c2_map, c1_map, m_max, n_max):
    """Fold per-``(m, n)`` acb principal-part coefficients directly in flint.

    The parity fold is a fixed, Delta-independent integer-weighted linear map
    (:func:`derivs._fold_plan`). The C2/C1 products remain in acb and are
    accumulated in place to avoid repeated conversions. Only the
    ``(m_max + 1) x (n_max + 1)`` folded output is converted to mpc.

    ``c2_map`` / ``c1_map`` hold the acb values keyed by the source ``(m, n)``
    the plan reads. Returns ``(C2f, C1f)`` as mpc list-of-lists, matching
    :func:`_fold_parity`'s output shape."""
    from flint import acb
    from .polynomial import _acb_to_mpc as _acb_fast
    accum2 = [acb(0) for _ in range(target_size)]
    accum1 = [acb(0) for _ in range(target_size)]
    for (m, n, contribs) in plan:
        c2 = c2_map.get((m, n))
        if c2 is None:
            continue
        c1 = c1_map[(m, n)]
        for target_idx, weight in contribs:
            accum2[target_idx] += c2 * weight
            accum1[target_idx] += c1 * weight
    C2f = [[_acb_fast(accum2[a * width + b]) for b in range(width)]
           for a in range(m_max + 1)]
    C1f = [[_acb_fast(accum1[a * width + b]) for b in range(width)]
           for a in range(m_max + 1)]
    return C2f, C1f


def _analytic_principal_parts_folded(roots, entries, exp_base, *,
                                     factor_m_plus, factor_m_minus,
                                     factor_n_plus, factor_n_minus,
                                     sigma, fold_marker, m_max, n_max):
    """Principal parts (``g_{-1}``, ``g_{-2}``) for folded (ws / xt / qqb_ws) coordinates, or ``None``.

    A folded ``(w, s)`` entry is a fixed, Δ-independent linear recombination (``derivs._fold_parity``) of a band of
    non-folded ``(m, n)`` entries, so it commutes with taking the principal part: ``PP(fold(B)) = fold(PP(B))``.
    The non-folded ``C2`` / ``C1`` grids (the algebra of :func:`_analytic_principal_parts_axis`) are folded with the
    SAME map the value assembler uses.  ``fold_marker`` is the ``r = sqrt(s)`` smoothing parity (only its parity
    matters); folded projected B forbids a complex parity phase, so ``sigma`` is real.  Same return shape as
    :func:`_analytic_principal_parts_axis` (mpc rows).
    """
    folded_entries = [(int(a), int(b)) for (a, b) in entries]
    sigma_c = None if sigma in (None, 0) else _mpf_to_acb(sigma)
    roots_mpc = [mp.mpc(r) for r in roots]
    groups = _group_roots_by_location(roots_mpc, tolerance(guard=10))
    exp_base_c = mp.mpc(exp_base)
    Lc = _mpf_to_acb(mp.log(exp_base_c))

    odd = fold_marker % 2
    M_total = m_max + 2 * n_max + odd
    src_dim = M_total + 1
    # The non-folded source entries the fold map actually reads (a band of the (m, n) square) -- only those products.
    plan, target_size = _fold_plan(bool(odd), src_dim, src_dim, m_max, n_max)
    width = n_max + 1
    src_entries = sorted({(m, n) for (m, n, _c) in plan})
    fold = _FoldPlan(plan, target_size, width, m_max, n_max, src_entries)

    # the folded (g_{-1}, g_{-2}) of each location as an _OrderTwoPrincipalRows over the fold; conjugate locations
    # are mirrored exactly as in _analytic_principal_parts_axis
    reflecting = exp_base_c.imag == 0 and exp_base_c.real > 0 and (sigma_c is None or sigma_c.imag.is_zero())
    rows_by_factors = {}
    parts = []
    for loc, idxs in groups:
        fmp = factor_m_plus(loc)
        fmm = factor_m_minus(loc)
        fnp = factor_n_plus(loc)
        fnm = factor_n_minus(loc)
        if fmp is None or fmm is None or fnp is None or fnm is None:
            return None
        if not folded_entries:
            parts.append((loc, idxs, []))
            continue
        factor_ids = (id(fmp), id(fmm), id(fnp), id(fnm))
        sources = tuple(getattr(f, "source", None) for f in (fmp, fmm, fnp, fnm))
        mirror_of = rows_by_factors.get(tuple(map(id, sources))) if (
            reflecting and all(src is not None for src in sources)) else None
        if mirror_of is not None and len(mirror_of.roots) == len(idxs):
            ebloc = mirror_of.scale.conjugate()     # exp_base^-conj(loc) for a real positive base
        else:
            ebloc, mirror_of = _mpf_to_acb(exp_base_c ** (-loc)), None
        (Amp, Bmp), (Amm, Bmm), (Anp, Bnp), (Anm, Bnm) = fmp, fmm, fnp, fnm
        rows = _OrderTwoPrincipalRows(
            folded_entries, (Amp, Bmp, Amm, Bmm, Anp, Bnp, Anm, Bnm), sigma_c, Lc, ebloc,
            roots=idxs, mirror_of=mirror_of, fold=fold,
        )
        rows_by_factors[factor_ids] = rows
        parts.append((loc, idxs, rows))
    return parts


def _analytic_residue_leading_scores_folded(roots, entries, exp_base, **folded):
    """Folded leading-Laurent residue scores from :func:`_analytic_principal_parts_folded`, or ``None``."""
    parts = _analytic_principal_parts_folded(roots, entries, exp_base, **folded)
    return None if parts is None else _leading_scores_from_principal_parts(parts, len(roots))


def _analytic_pp_scores_folded(roots, delta_eval, entries, exp_base, **folded):
    """Folded analytic ``contribution`` scores from :func:`_analytic_principal_parts_folded`, or ``None``."""
    parts = _analytic_principal_parts_folded(roots, entries, exp_base, **folded)
    return None if parts is None else _scores_from_principal_parts(parts, len(roots), delta_eval)


class _PrincipalPartScoring:
    """A sampler whose ``_analytic_principal_parts(roots, entries, exp_base)`` gives the block's exact principal
    parts at its poles: the analytic ``contribution`` scores are derived from them."""

    def _analytic_contribution_scores(self, roots, delta_eval, entries, exp_base):
        parts = self._analytic_principal_parts(roots, entries, exp_base)
        return None if parts is None else _scores_from_principal_parts(parts, len(roots), delta_eval)


def _principal_part_values_many(principal_parts, deltas, entries, *, as_flint=False):
    """``[{(m, n): exp_base^Delta sum_a g_a (Delta - loc)^-a} for each Delta]`` -- the partial fractions, by products,
    in flint (arb when everything is real, else acb).  ``as_flint`` keeps the flint values (else mpmath)."""
    from flint import acb, arb, ctx as flint_ctx

    from .polynomial import _acb_to_mpc, _arb_to_mpf, _mpf_to_acb, _mpf_to_arb

    exp_base, groups = principal_parts
    flint_ctx.prec = int(mp.mp.dps * 3.33) + 20

    def is_real(value):
        return not isinstance(value, mp.mpc) or value.imag == 0

    real = is_real(exp_base) and all(is_real(d) for d in deltas) and all(
        is_real(loc) and all(is_real(g) for entry in entries for g in rows.get(entry, ())) for loc, rows in groups)
    if real:
        to = lambda v: _mpf_to_arb(mp.mpf(mp.re(v)))  # noqa: E731
        back = _arb_to_mpf
    else:
        to = lambda v: _mpf_to_acb(mp.mpc(v))  # noqa: E731
        back = _acb_to_mpc
    log_base = to(exp_base).log()
    flint_groups = [(to(loc), [[to(g) for g in rows.get(entry, ())] for entry in entries]) for loc, rows in groups]
    zero = arb(0) if real else acb(0)
    out = []
    for delta in deltas:
        d = to(delta)
        scale = (d * log_base).exp()
        totals = [zero] * len(entries)
        for loc, rows in flint_groups:
            inv = 1 / (d - loc)
            powers = [inv]
            for _ in range(1, max((len(row) for row in rows), default=1)):
                powers.append(powers[-1] * inv)
            for e, row in enumerate(rows):
                for a, g in enumerate(row):
                    totals[e] = totals[e] + g * powers[a]
        values = [scale * total for total in totals]
        out.append(dict(zip(entries, values if as_flint else [back(v) for v in values])))
    return out


class _RemainderSampler:
    """A numeric block sampler minus the block's principal parts at given poles (the c < 1 poles on x > 0).

    What the order reduction fits once those poles are split off: the remainder is regular there, so its fit sees
    only the x <= 0 poles, as at c >= 1.  Every other attribute (the analytic scorers at the remaining poles, whose
    principal parts are the block's own) is the inner sampler's.
    """

    def __init__(self, inner, principal_parts):
        self.inner = inner
        self.principal_parts = principal_parts

    def __getattr__(self, name):
        if name.startswith("__") or "inner" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.inner, name)

    @_quiet_conditioning
    def __call__(self, deltas, entries=None):
        deltas = list(deltas)
        entries = tuple(entries) if entries is not None else None
        tables, is_arb = self.inner(deltas, entries=entries)
        return self._minus_principal_parts(tables, is_arb, deltas, entries)

    @_quiet_conditioning
    def sample_bucket(self, values, entries):
        """The global core's bucket interface (:func:`global_blocks._global_bucket_task`), for an inner sampler of
        that convention (``inner(value)`` one grid, ``inner.sample_bucket(values, entries)`` ``(grids, is_arb)`` or
        None): its grids minus the principal parts."""
        values = list(values)
        entries = tuple(entries) if entries is not None else None
        bucket = getattr(self.inner, "sample_bucket", None)
        result = bucket(values, entries) if bucket is not None else None
        tables, is_arb = result if result is not None else ([self.inner(value) for value in values], False)
        return self._minus_principal_parts(tables, is_arb, values, entries)

    def _minus_principal_parts(self, tables, is_arb, deltas, entries):
        if not tables:
            return tables, is_arb
        # The inner sampler may return cached grids, including the same grid
        # for repeated points. Subtract into independent copies, once per point.
        tables = [[list(row) for row in table] for table in tables]
        wanted = list(entries) if entries is not None else [
            (m, n) for m in range(len(tables[0])) for n in range(len(tables[0][0]))]
        for table, values in zip(tables, _principal_part_values_many(self.principal_parts, deltas, wanted,
                                                                     as_flint=is_arb)):
            for (m, n), value in values.items():
                table[m][n] = table[m][n] - (value.real if is_arb and hasattr(value, "imag") else value)
        return tables, is_arb


def _split_forced_principal_parts(exacts, forced_idx, entries_by_coord):
    """``{coord: (remainder_table, principal_parts)}`` with the forced poles' exact principal parts split off, or
    ``None`` when a sampler cannot supply them (the caller then fits the forced poles into the numerator).

    The remainder view is the skeleton with the forced roots divided out (every numerator degree lowered by their
    number: ``regularization._skeleton_quotient_by_root_indices``) and a sampler returning the block minus the
    principal parts.
    """
    from dataclasses import replace

    from .regularization import _skeleton_quotient_by_root_indices

    out = {}
    for coord, exact in exacts.items():
        sampler = getattr(exact, "numeric_sampler", None)
        principal = getattr(sampler, "_analytic_principal_parts", None)
        entries = entries_by_coord[coord]
        roots = [exact.denominator_roots[i] for i in forced_idx]
        parts = None if principal is None else principal(roots, entries, exact.exp_base)
        if parts is None and exact.numerators is not None:
            # a symbolic table: its own partial fractions, and its own values for the remainder
            parts = _symbolic_principal_parts(exact, roots, [exact.numerators[m][n] for (m, n) in entries])
            if sampler is None:
                sampler = _ExactTableValues(exact)
        if parts is None:
            return None
        principal_parts = _principal_parts_record(exact.exp_base, entries, parts)
        skeleton = exact if exact.numerators is None else replace(
            exact, numerators=None,
            numerator_degrees=[[exact.numerator_degree(m, n) for n in range(exact.n_max + 1)]
                               for m in range(exact.m_max + 1)],
        )
        remainder = replace(
            _skeleton_quotient_by_root_indices(skeleton, forced_idx),
            numeric_sampler=_RemainderSampler(sampler, principal_parts), oracle_metadata=None,
        )
        out[coord] = (remainder, principal_parts)
    return out


def _principal_parts_record(exp_base, entries, parts):
    """The picklable ``(exp_base, [(loc, {(m, n): (g_1, ..., g_M)})])`` form of analytic principal parts
    (:func:`_analytic_principal_parts_axis`'s ``[(loc, root_indices, rows)]``, rows aligned with ``entries``), as
    :func:`_principal_part_values_many`, :class:`_RemainderSampler` and :func:`_recombine_principal_parts` read it."""
    return (
        mp.mpc(exp_base),
        [(mp.mpc(loc), {entry: tuple(_as_mp_number(g) for g in row)
                        for entry, row in zip(entries, rows) if row is not None})
         for loc, _idxs, rows in parts],
    )


def _as_mp_number(value):
    """mpmath value of an acb / mpc / mpf principal-part coefficient (real when its imaginary part vanishes)."""
    if isinstance(value, mp.mpf):
        return value
    if not isinstance(value, mp.mpc):
        from .polynomial import _acb_to_mpc as exact_midpoint

        value = exact_midpoint(value)
    # A tiny residue can become an order-one term near its pole. Its size
    # alone cannot identify numerical noise; keep every nonzero component.
    return value.real if value.imag == 0 else value


def _recombine_principal_parts(table, principal_parts, forced_roots):
    """The remainder's interpolated table with the split-off principal parts put back, in the ordinary
    ``prefactor * numerators`` form (the forced poles in the prefactor, ``N = N_rem F + (base^shift / C) Q S`` with
    ``F`` the forced and ``Q`` the kept denominator in x and ``S = sum g_a F / (x - loc)^a``; flint polynomials)."""
    from dataclasses import replace

    from flint import acb, acb_poly, arb, arb_poly, ctx as flint_ctx

    from .polynomial import _mpf_to_acb, _mpf_to_arb

    exp_base, groups = principal_parts
    kept = table.prefactor
    real_shift = mp.mpf(mp.re(table.shift))
    prec_bits = int(mp.mp.dps * 3.33) + 20
    flint_ctx.prec = prec_bits

    def realish(value):
        value = mp.mpc(value)
        return mp.mpf(value.real) if value.imag == 0 else value

    forced_x = [realish(mp.mpc(p) - real_shift) for p in forced_roots]
    scale = realish(mp.mpc(exp_base) ** real_shift / mp.mpc(kept.constant))
    values = [scale, *forced_x, *kept.poles] + [g for _loc, rows in groups for row in rows.values() for g in row]
    real = all(not isinstance(v, mp.mpc) or v.imag == 0 for v in values) and all(
        poly.field is mp.mpf for row in table.numerators for poly in row)
    num, poly_t = (lambda v: _mpf_to_arb(mp.mpf(mp.re(v))), arb_poly) if real else (lambda v: _mpf_to_acb(mp.mpc(v)), acb_poly)
    to_poly = (lambda poly: poly._get_arb()) if real else (lambda poly: poly._get_acb())
    from_poly = ((lambda poly: Polynomial._from_arb_trusted(poly, prec_bits)) if real
                 else (lambda poly: Polynomial._from_acb_trusted(poly, prec_bits)))

    forced_f = [num(p) for p in forced_x]
    F = poly_t.from_roots(forced_f)
    Q = poly_t.from_roots([num(p) for p in kept.poles])
    scale_f = num(scale)
    # F / (x - loc)^a for every group and order, as the product over the remaining roots (a division by (x - loc)
    # is not a stable exact division at this degree: it cost ~80 of 200 digits)
    group_tol = tolerance(guard=10)
    quotients = []
    for loc, rows in groups:
        loc_x = realish(mp.mpc(loc) - real_shift)
        at_loc = [i for i, p in enumerate(forced_x) if abs(mp.mpc(p) - mp.mpc(loc_x)) <= group_tol * max(1, abs(mp.mpc(loc_x)))]
        steps = []
        for a in range(1, max((len(row) for row in rows.values()), default=0) + 1):
            drop = set(at_loc[:a])
            steps.append(poly_t.from_roots([r for i, r in enumerate(forced_f) if i not in drop]))
        quotients.append((rows, steps))
    numerators = []
    for m, row in enumerate(table.numerators):
        out_row = []
        for n, n_rem in enumerate(row):
            S = None
            for rows, steps in quotients:
                for a, g in enumerate(rows.get((m, n), ())):
                    term = steps[a] * num(g)
                    S = term if S is None else S + term
            total = to_poly(n_rem) * F
            if S is not None:
                total = total + Q * S * scale_f
            out_row.append(from_poly(total))
        numerators.append(out_row)
    records = table.numerator_sample_values
    if records is not None:
        # the fit's node records are the remainder's: put the principal parts back into the raw block values
        entries = [(m, n) for m, row in enumerate(records) for n, record in enumerate(row) if record is not None]
        nodes = sorted({node for (m, n) in entries for node in table.numerator_sample_points[m][n]})
        pp_at = dict(zip(nodes, _principal_part_values_many(principal_parts, nodes, entries)))
        records = [[None if record is None else (record[0], tuple(
            value + pp_at[node][(m, n)] for value, node in zip(record[1], table.numerator_sample_points[m][n])),
            record[2]) for n, record in enumerate(row)] for m, row in enumerate(records)]
    return replace(
        table,
        prefactor=DampedRational(constant=kept.constant, poles=tuple(kept.poles) + tuple(forced_x), base=kept.base),
        numerators=numerators,
        numerator_sample_values=records,
        principal_parts=principal_parts,
        remainder_prefactor=kept,
    )


class _PrecisionAwareNumericSampler(_PrincipalPartScoring):
    """Invalidate derived numerical state when a sampler changes precision."""

    _precision_caches = ()

    def _ensure_precision(self):
        if getattr(self, "_cache_precision", None) == mp.mp.prec:
            return
        for name in self._precision_caches:
            setattr(self, name, None)
        spec = COORDINATES[self.coordinate]
        if spec.kind in ("non_folded_2d", "chiral"):
            _, self._u_center = _resolve_center_u(self.coordinate, self.center)
            center_bar = getattr(self, "center_bar", None)
            self._u_center_bar = (
                self._u_center if center_bar is None
                else _resolve_center_u(self.coordinate, center_bar)[1]
            )
            self._offdiag = self._u_center_bar != self._u_center
            self._recentered = self._u_center != spec.u_symmetric()
        self._cache_precision = mp.mp.prec


class _NumericBlockSampler(_PrecisionAwareNumericSampler):
    """Picklable numeric-recursion evaluator of a block table at sample Delta.

    Returns the same ``[m][n]`` block tables as ``DeltaDerivTable.evaluate``, but
    via the O(1) numeric recursion (``G_zzb_table`` / ``G_yyb_table`` /
    ``G_qqb_table``) rather than the ill-conditioned symbolic ``P(Delta)/D(Delta)``
    whose coefficients reach ~1e500 (so its evaluation needs very high dps to
    survive cancellation).  The numeric block equals the symbolic table's value,
    so interpolating against these sample values yields an identical interpolant
    -- only the conditioning of the sample-value step differs.

    Stored on a table as ``DeltaDerivTable.numeric_sampler`` and consumed by
    :func:`_evaluate_table_flint_many`.  Defined at module level and holding only
    plain data so it pickles cleanly into interpolation worker processes.
    """

    _COORD_FN = {
        "zzb": "G_zzb_table",
        "yyb": "G_yyb_table",
        "qqb": "G_qqb_table",
        "ws": "G_ws_table",
        "xt": "G_xt_table",
        "qqb_ws": "G_qqb_ws_table",
    }
    _COORD_MANY_FN = {
        "zzb": "G_zzb_table_many",
        "yyb": "G_yyb_table_many",
        "qqb": "G_qqb_table_many",
        "ws": "G_ws_table_many",
        "xt": "G_xt_table_many",
        "qqb_ws": "G_qqb_ws_table_many",
    }
    # Non-folded coordinates expose a single chiral series evaluator, so the
    # entry-restricted batched arb path (used per interpolation target) builds
    # only the requested (m, n) outer products from the shared chiral
    # coefficients. The folded coordinates recombine entries, so they stay on
    # the full-table path.
    _COORD_CHIRAL_EVALUATOR = {
        "zzb": "chiral_z_series_evaluator",
        "yyb": "chiral_y_series_evaluator",
        "qqb": "chiral_q_series_evaluator",
    }
    # Folded coordinates: the parity fold recombines a band of chiral
    # coefficients (up to M_total = m_max + 2*n_max), so the chiral series can
    # *not* be shortened to max(m, n) + 1 the way the non-folded outer product
    # can. Entry restriction here only prunes the fold accumulation and the
    # per-entry products to the requested output (m, n); the full-length chiral
    # `.many` evaluation is the irreducible floor. Scalar externals only (the
    # folded coordinates are scalar-only by construction; for scalars the
    # holomorphic and antiholomorphic external weights coincide, so a single
    # evaluator serves both chiral factors for identical *and* nonidentical
    # externals alike).
    _COORD_FOLDED_EVALUATOR = {
        "ws": "chiral_y_series_evaluator",
        "xt": "chiral_z_series_evaluator",
        "qqb_ws": "chiral_q_series_evaluator",
    }

    _precision_caches = ('_cache', '_restricted_cache', '_evaluator', '_evaluator_n', '_folded_evaluator')

    def __init__(self, coordinate, h_externals, ell, m_max, n_max,
                 *, b, c, q_order, chiral_norm, z_shift,
                 center=None, center_bar=None):
        m_max, n_max = _check_table_orders(m_max, n_max)
        q_order = _integer_control('q_order', q_order)
        if coordinate not in self._COORD_FN:
            raise ValueError(f"no numeric block sampler for coordinate {coordinate!r}")
        self.coordinate = coordinate
        from .numeric_input import parse_number
        self.h_externals = [parse_number(value) for value in h_externals]
        self.ell = ell
        self.m_max = m_max
        self.n_max = n_max
        self.b = b
        self.c = c
        self.q_order = q_order
        self.chiral_norm = chiral_norm
        self.z_shift = z_shift
        self.center = center
        self.center_bar = center_bar

        if coordinate in self._COORD_CHIRAL_EVALUATOR:
            _z_of_u, u_center = _resolve_center_u(coordinate, center)
            if center_bar is None:
                u_center_bar = u_center
            else:
                _z_of_u_bar, u_center_bar = _resolve_center_u(coordinate, center_bar)
            self._u_center = u_center
            self._u_center_bar = u_center_bar
            self._offdiag = (u_center_bar != u_center)
            # A diagonal off-symmetric center (z = zbar = center != symmetric point) keeps the
            # single-evaluator diagonal path, but that evaluator must be the recentered one: the
            # plain factories expand around the symmetric point.
            self._recentered = u_center != COORDINATES[coordinate].u_symmetric()
        else:
            if center is not None or center_bar is not None:
                raise ValueError(
                    "off-diagonal centers are supported only on non-folded "
                    "coordinates (zzb/yyb/qqb)"
                )
            self._u_center = None
            self._u_center_bar = None
            self._offdiag = False
            self._recentered = False

        self._cache = None
        self._restricted_cache = None
        self._evaluator = None
        self._evaluator_n = None
        self._folded_evaluator = None
        self._folded_evaluator_nterms = 0

    @staticmethod
    def _cache_key(delta):
        if isinstance(delta, Fraction):
            return ("rational", delta.numerator, delta.denominator)
        if isinstance(delta, mp.mpc):
            return ("mpf", mp.mpf(delta.real)._mpf_) if delta.imag == 0 else ("mpc", delta._mpc_)
        return ("mpf", mp.mpf(delta)._mpf_)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = None
        state["_restricted_cache"] = None
        state["_evaluator"] = None
        state["_evaluator_n"] = None
        state["_folded_evaluator"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._evaluator_n = None
        self._folded_evaluator = None
        self._folded_evaluator_nterms = 0

    @staticmethod
    def _entries_cache_key(entries):
        if entries is None:
            return None
        return tuple(sorted((int(m), int(n)) for (m, n) in entries))

    def _centered_nonfolded_evaluator(self, _derivs, n_terms, u_center):
        """Build one non-folded chiral evaluator recentered at ``u_center``."""
        spec = COORDINATES[self.coordinate]
        return _derivs._chiral_series_evaluator(
            self.h_externals,
            n_terms,
            b=self.b,
            c=self.c,
            q_order=self.q_order,
            chiral_norm=self.chiral_norm,
            z_shift=self.z_shift,
            z_of_u=spec.z_of_u,
            u_symmetric=u_center,
            coord_key=spec.chiral_var,
        )

    def _diagonal_evaluator(self, _derivs, n_terms):
        """The single chiral evaluator of the diagonal path: the symmetric-point
        factory, or the one recentered at ``u_center`` for an off-symmetric
        diagonal center."""
        if getattr(self, "_recentered", False):
            return self._centered_nonfolded_evaluator(_derivs, n_terms, self._u_center)
        fac_name = self._COORD_CHIRAL_EVALUATOR[self.coordinate]
        return getattr(_derivs, fac_name)(
            self.h_externals, n_terms, b=self.b, c=self.c,
            q_order=self.q_order, chiral_norm=self.chiral_norm,
            z_shift=self.z_shift,
        )

    def _per_delta_coeffs(self, evaluator, batch, half):
        """Per-Delta acb chiral coefficients (the complex c / Delta / Kac-pole
        path for the batched-arb bail), so the entry-restricted assembly stays
        available without ever materialising the full square."""
        ell = self.ell
        return [evaluator(_exchange_weight(d, half * ell), return_acb=True) for d in batch]

    def _analytic_principal_parts(self, roots, entries, exp_base):
        """Exact principal parts at the poles (generic c), or ``None`` to defer to the contour scorer.

        Scalar-G specialization of :func:`_analytic_pp_scores`: on the diagonal
        one chiral evaluator serves both factors (``c_h[k]=c[k]((Δ+ℓ)/2)``,
        ``c_hbar[k]=c[k]((Δ-ℓ)/2)``) and ``σ`` = :func:`~._conventions.scalar_G_swap_coefficient`
        (``(-1)**ℓ``; 0 for the public spin-0 block), as in :meth:`__call__`. Off-diagonal
        (``u_center != u_center_bar``) the m-index uses the m-center evaluator and
        the n-index the n-center evaluator (four chiral-factor streams). Folded
        coordinates (ws / xt / qqb_ws) build the non-folded principal-part grids
        and fold them (diagonal only). Returns one score per root or ``None`` to
        defer to the contour scorer (an unsupported context mode)."""
        self._ensure_precision()
        from . import derivs as _derivs

        ell = self.ell
        sigma = mp.mpf(scalar_G_swap_coefficient(ell))

        if self.coordinate in self._COORD_FOLDED_EVALUATOR:
            fold_marker = fold_parity_marker(ell=ell)
            odd = fold_marker % 2
            M_total = self.m_max + 2 * self.n_max + odd
            n_terms = M_total + 1
            if (
                self._folded_evaluator is None
                or self._folded_evaluator_nterms < n_terms
            ):
                fac_name = self._COORD_FOLDED_EVALUATOR[self.coordinate]
                self._folded_evaluator = getattr(_derivs, fac_name)(
                    self.h_externals, n_terms, b=self.b, c=self.c,
                    q_order=self.q_order, chiral_norm=self.chiral_norm,
                    z_shift=self.z_shift,
                )
                self._folded_evaluator_nterms = n_terms
            fac = _analytic_chiral_factor(self._folded_evaluator, n_terms, deriv_factor=2)
            if fac is None:
                return None
            _prefetch_chiral_factor(fac, roots, ell)
            fac_p = lambda loc: fac((loc + ell) / 2)
            fac_m = lambda loc: fac((loc - ell) / 2)
            return _analytic_principal_parts_folded(
                roots, entries, exp_base,
                factor_m_plus=fac_p, factor_m_minus=fac_m,
                factor_n_plus=fac_p, factor_n_minus=fac_m,
                sigma=sigma, fold_marker=fold_marker,
                m_max=self.m_max, n_max=self.n_max,
            )

        if self.coordinate not in self._COORD_CHIRAL_EVALUATOR:
            return None

        n_terms = max(self.m_max, self.n_max) + 1

        if self._offdiag:
            if self._evaluator is None:
                self._evaluator = self._centered_nonfolded_evaluator(
                    _derivs, n_terms, self._u_center
                )
                self._evaluator_n = self._centered_nonfolded_evaluator(
                    _derivs, n_terms, self._u_center_bar
                )
            fac_m = _analytic_chiral_factor(self._evaluator, n_terms, deriv_factor=2)
            fac_n = _analytic_chiral_factor(self._evaluator_n, n_terms, deriv_factor=2)
            if fac_m is None or fac_n is None:
                return None
            _prefetch_chiral_factor(fac_m, roots, ell)
            _prefetch_chiral_factor(fac_n, roots, ell)
            return _analytic_principal_parts_axis(
                roots, entries, exp_base,
                factor_m_plus=lambda loc: fac_m((loc + ell) / 2),
                factor_m_minus=lambda loc: fac_m((loc - ell) / 2),
                factor_n_plus=lambda loc: fac_n((loc + ell) / 2),
                factor_n_minus=lambda loc: fac_n((loc - ell) / 2),
                sigma=sigma,
            )

        if self._evaluator is None:
            self._evaluator = self._diagonal_evaluator(_derivs, n_terms)
        fac = _analytic_chiral_factor(self._evaluator, n_terms, deriv_factor=2)
        if fac is None:
            return None
        _prefetch_chiral_factor(fac, roots, ell)
        return _analytic_principal_parts(
            roots, entries, exp_base,
            factor_h=lambda loc: fac((loc + ell) / 2),
            factor_hbar=lambda loc: fac((loc - ell) / 2),
            sigma=sigma,
        )

    def _call_restricted(self, _derivs, batch, entries):
        """Entry-restricted block values for a non-folded coordinate, or ``None``
        only when the coordinate is folded (handled by the folded variant).

        Builds only the requested ``(m, n)`` entries from one chiral evaluation
        (shared ``c_h``). The all-real batch keeps everything in flint ``arb``
        (``(tables, True)``) so the interpolation's Lagrange step skips the
        arb->mpf->arb round trip; the complex c / complex Delta / Kac-pole case
        falls back to per-Delta ``acb`` coefficients -- still entry-restricted,
        never the full-square ``_COORD_FN`` path."""
        fac_name = self._COORD_CHIRAL_EVALUATOR.get(self.coordinate)
        if fac_name is None:
            return None
        ell = self.ell
        from virasoro_blocks.derivs import assemble_block_batch

        sign = mp.mpf(scalar_G_swap_coefficient(ell))
        n_terms = max(self.m_max, self.n_max) + 1

        if self._offdiag:
            if self._evaluator is None:
                self._evaluator = self._centered_nonfolded_evaluator(
                    _derivs, n_terms, self._u_center
                )
                self._evaluator_n = self._centered_nonfolded_evaluator(
                    _derivs, n_terms, self._u_center_bar
                )
            many_m = getattr(self._evaluator, "many", None)
            many_n = getattr(self._evaluator_n, "many", None)
            c_h_m_all = (
                many_m([_exchange_weight(d, ell) for d in batch]) if many_m is not None else None
            )
            c_hbar_n_all = (
                many_n([_exchange_weight(d, -ell) for d in batch]) if many_n is not None else None
            )
            if c_h_m_all is not None and c_hbar_n_all is not None:
                c_hbar_m_all = (
                    c_h_m_all if ell == 0
                    else many_m([_exchange_weight(d, -ell) for d in batch])
                )
                c_h_n_all = (
                    c_hbar_n_all if ell == 0
                    else many_n([_exchange_weight(d, ell) for d in batch])
                )
            else:
                c_hbar_m_all = c_h_n_all = None

            if (
                c_h_m_all is None
                or c_hbar_n_all is None
                or c_hbar_m_all is None
                or c_h_n_all is None
            ):
                # Batched-arb bail: per-Delta acb, still only requested entries.
                c_h_m_all = self._per_delta_coeffs(self._evaluator, batch, +1)
                c_hbar_n_all = self._per_delta_coeffs(self._evaluator_n, batch, -1)
                c_hbar_m_all = (
                    c_h_m_all if ell == 0
                    else self._per_delta_coeffs(self._evaluator, batch, -1)
                )
                c_h_n_all = (
                    c_hbar_n_all if ell == 0
                    else self._per_delta_coeffs(self._evaluator_n, batch, +1)
                )

            return assemble_block_batch(
                c_h_m_all,
                c_hbar_n_all,
                self.m_max,
                self.n_max,
                sigma=sign,
                entries=entries,
                c_swap_m_all=c_hbar_m_all,
                c_swap_n_all=c_h_n_all,
            )

        if self._evaluator is None:
            self._evaluator = self._diagonal_evaluator(_derivs, n_terms)
        many = getattr(self._evaluator, "many", None)
        c_h_all = many([_exchange_weight(d, ell) for d in batch]) if many is not None else None
        if c_h_all is not None:
            c_hbar_all = c_h_all if ell == 0 else many([_exchange_weight(d, -ell) for d in batch])
        else:
            c_hbar_all = None
        if c_h_all is None or c_hbar_all is None:
            # Batched-arb bail: per-Delta acb, still only the requested entries.
            c_h_all = self._per_delta_coeffs(self._evaluator, batch, +1)
            c_hbar_all = (
                c_h_all if ell == 0
                else self._per_delta_coeffs(self._evaluator, batch, -1)
            )
        return assemble_block_batch(
            c_h_all, c_hbar_all, self.m_max, self.n_max, sigma=sign,
            entries=entries,
        )

    def _call_restricted_folded(self, _derivs, batch, entries):
        """Entry-restricted, batched, arb-native folded block values, or ``None``.

        The folded analogue of :meth:`_call_restricted` for the ws / xt /
        qqb_ws coordinates. Computes the full-length chiral coefficients once
        per Delta (the irreducible cost -- the fold needs the whole
        ``M_total``-length series), then folds *only* the requested output
        ``(m, n)`` entries: the fold plan is pruned to the requested targets,
        and each contributing ``(m', n')`` product is formed on demand instead
        of materialising the full ``M_total x M_total`` outer product. Returns
        ``(tables, True)`` with sparse arb ``[m][n]`` grids, or ``None`` to fall
        back to the full-table path (complex c / complex Delta / Kac-pole hit).
        """
        fac_name = self._COORD_FOLDED_EVALUATOR.get(self.coordinate)
        if fac_name is None:
            return None

        ell = self.ell
        odd_factor = fold_parity_marker(ell=ell)
        # The chiral series only needs to reach the highest source degree any
        # requested folded entry consumes: producing w^a s^b needs source
        # y-degree a + 2b. Under the m + 2n <= Lambda truncation the requested
        # entries form that triangle, so M_total = max(a + 2b) = Lambda -- far
        # shorter than m_max + 2 n_max for a full square.
        M_total = max((m + 2 * n for (m, n) in entries), default=0) + odd_factor

        n_terms = M_total + 1
        if self._folded_evaluator is None or self._folded_evaluator_nterms < n_terms:
            self._folded_evaluator = getattr(_derivs, fac_name)(
                self.h_externals, n_terms, b=self.b, c=self.c,
                q_order=self.q_order, chiral_norm=self.chiral_norm,
                z_shift=self.z_shift,
            )
            self._folded_evaluator_nterms = n_terms
        many = getattr(self._folded_evaluator, "many", None)
        c_h_all = many([_exchange_weight(d, ell) for d in batch]) if many is not None else None
        if c_h_all is not None:
            c_hbar_all = c_h_all if ell == 0 else many([_exchange_weight(d, -ell) for d in batch])
        else:
            c_hbar_all = None
        if c_h_all is None or c_hbar_all is None:
            # Batched-arb bail: per-Delta acb, still only the requested entries.
            c_h_all = self._per_delta_coeffs(self._folded_evaluator, batch, +1)
            c_hbar_all = (
                c_h_all if ell == 0
                else self._per_delta_coeffs(self._folded_evaluator, batch, -1)
            )

        from virasoro_blocks.derivs import assemble_block_batch

        sign = mp.mpf(scalar_G_swap_coefficient(ell))
        return assemble_block_batch(
            c_h_all, c_hbar_all, self.m_max, self.n_max, sigma=sign,
            fold_parity=odd_factor, entries=entries,
        )

    @_quiet_conditioning
    def __call__(self, deltas, entries=None):
        self._ensure_precision()
        from . import derivs as _derivs

        entries = tuple(entries) if entries is not None else None
        batch = [_derivs._coerce_delta_scalar(delta) for delta in deltas]
        if entries is not None:
            entries_key = self._entries_cache_key(entries)
            if self._restricted_cache is None:
                self._restricted_cache = {}
            rcache = self._restricted_cache
            tables = [None] * len(batch)
            flags = []
            missing = []
            missing_keys = []
            missing_index_groups = []
            pending_positions = {}

            for i, delta in enumerate(batch):
                dkey = self._cache_key(delta)
                key = (dkey, entries_key)
                cached = rcache.get(key)
                if cached is None:
                    if dkey in pending_positions:
                        pending_positions[dkey].append(i)
                    else:
                        pending_positions[dkey] = [i]
                        missing.append(delta)
                        missing_keys.append(dkey)
                        missing_index_groups.append(pending_positions[dkey])
                else:
                    table, is_arb_cached = cached
                    tables[i] = table
                    flags.append(bool(is_arb_cached))

            if not missing:
                return tables, (all(flags) if flags else False)

            restricted = self._call_restricted(_derivs, missing, entries)
            if restricted is None:
                restricted = self._call_restricted_folded(_derivs, missing, entries)
            if restricted is not None:
                fetched, is_arb = restricted
                # fetched rows align with `missing`; fill only misses and cache by
                # (delta, entries) so repeated scorer calls can reuse exactly.
                for indices, dkey, table in zip(missing_index_groups, missing_keys, fetched):
                    rcache[(dkey, entries_key)] = (table, is_arb)
                    for i in indices:
                        tables[i] = table
                flags.append(bool(is_arb))
                return tables, (all(flags) if flags else bool(is_arb))
        if self._cache is None:
            self._cache = {}
        cache = self._cache
        tables = [None] * len(batch)
        flags = []
        missing = []
        missing_keys = []
        missing_index_groups = []
        pending_positions = {}
        for i, delta in enumerate(batch):
            key = self._cache_key(delta)
            cached = cache.get(key)
            if cached is None:
                if key in pending_positions:
                    pending_positions[key].append(i)
                else:
                    pending_positions[key] = [i]
                    missing.append(delta)
                    missing_keys.append(key)
                    missing_index_groups.append(pending_positions[key])
            else:
                if isinstance(cached, tuple) and len(cached) == 2 and isinstance(cached[1], bool):
                    table, is_arb_cached = cached
                else:
                    table, is_arb_cached = cached, False
                tables[i] = table
                flags.append(bool(is_arb_cached))

        if not missing:
            return tables, (all(flags) if flags else False)

        if self._offdiag:
            fetched, is_arb = self._call_restricted(_derivs, missing, entries=None)
            for indices, key, table in zip(missing_index_groups, missing_keys, fetched):
                cache[key] = (table, bool(is_arb))
                for i in indices:
                    tables[i] = table
            flags.append(bool(is_arb))
            return tables, (all(flags) if flags else bool(is_arb))

        # Full-table structural fast path: reuse the batched flint assembler used
        # by the entry-restricted interpolation path so real samples stay
        # arb-native (is_arb=True) instead of crossing the flint->mpmath
        # boundary in the *_table_many helpers.
        full_entries = [
            (m, n) for m in range(self.m_max + 1) for n in range(self.n_max + 1)
        ]
        restricted = self._call_restricted(_derivs, missing, full_entries)
        if restricted is None:
            restricted = self._call_restricted_folded(_derivs, missing, full_entries)
        if restricted is not None:
            fetched, is_arb = restricted
            for indices, key, table in zip(missing_index_groups, missing_keys, fetched):
                cache[key] = (table, bool(is_arb))
                for i in indices:
                    tables[i] = table
            flags.append(bool(is_arb))
            return tables, (all(flags) if flags else bool(is_arb))

        fetched = getattr(_derivs, self._COORD_MANY_FN[self.coordinate])(
            missing, self.ell, self.h_externals,
            self.m_max, self.n_max,
            b=self.b, c=self.c, q_order=self.q_order,
            chiral_norm=self.chiral_norm, z_shift=self.z_shift,
        )
        for indices, key, table in zip(missing_index_groups, missing_keys, fetched):
            cache[key] = (table, False)
            for i in indices:
                tables[i] = table
        flags.append(False)
        return tables, (all(flags) if flags else False)


class _NumericOrderedBlockSampler(_PrecisionAwareNumericSampler):
    """Numeric-recursion sampler for ordered_B / projected_B, including scalar externals.

    Returns the same ``[m][n]`` tables as the symbolic ordered or parity-
    projected B ``DeltaDerivTable.evaluate``, but via the O(1) chiral recursion.
    Unlike the scalar-G :class:`_NumericBlockSampler` this carries *two* chiral
    evaluators -- the holomorphic factor uses ``(h_externals, z_shift)``, the
    antiholomorphic uses ``(hbar_externals, zbar_shift)``. A single evaluator
    is reused when the two external weight lists and shifts coincide::

        ordered_B[m][n]   = c_h[m] * c_hbar[n]
        projected_B[m][n] = c_h[m] * c_hbar[n] + swap * c_hbar[m] * c_h[n]

    with ``swap = (+/-1) * parity_phase`` and ``parity_phase`` in {1, -1, i, -i}
    (so ``swap`` can be imaginary -> a complex block at real Delta). For the
    folded coordinates (ws/xt/qqb_ws, projected only) the parity fold is applied
    on top, with the *parity* sign -- not ell -- setting the fold marker (the
    folded path requires a real +/-1 phase, matching the symbolic guard).

    Like :meth:`_NumericBlockSampler._call_restricted`, ``entries`` restricts
    the build to the requested output ``(m, n)`` and keeps the values native in
    flint (``arb`` when everything is real, returning ``is_arb=True``); a complex
    swap, complex c, complex Delta, or complex externals route through ``acb``
    and return mpmath ``mpc`` values (``is_arb=False``).
    """

    _NONFOLDED_EVALUATOR = {
        "zzb": "chiral_z_series_evaluator",
        "yyb": "chiral_y_series_evaluator",
        "qqb": "chiral_q_series_evaluator",
    }
    _FOLDED_EVALUATOR = {
        "ws": "chiral_y_series_evaluator",
        "xt": "chiral_z_series_evaluator",
        "qqb_ws": "chiral_q_series_evaluator",
    }

    _precision_caches = ('_eval_h', '_eval_hbar', '_eval_h_n', '_eval_hbar_n')

    def __init__(self, coordinate, h_externals, hbar_externals, ell, m_max, n_max,
                 *, b, c, q_order, chiral_norm, z_shift, zbar_shift,
                 projected, parity=None, parity_external=None, exchanged_parity="1",
                 center=None, center_bar=None):
        m_max, n_max = _check_table_orders(m_max, n_max)
        q_order = _integer_control('q_order', q_order)
        folded = coordinate in self._FOLDED_EVALUATOR
        if not folded and coordinate not in self._NONFOLDED_EVALUATOR:
            raise ValueError(
                f"no ordered/projected block sampler for coordinate {coordinate!r}"
            )
        if folded and not projected:
            raise ValueError(
                "folded coordinates require the parity-projected basis"
            )
        self.coordinate = coordinate
        self.folded = folded
        from .numeric_input import parse_number
        self.h_externals = [parse_number(value) for value in h_externals]
        self.hbar_externals = [parse_number(value) for value in hbar_externals]
        self.ell = ell
        self.m_max = m_max
        self.n_max = n_max
        self.b = b
        self.c = c
        self.q_order = q_order
        self.chiral_norm = chiral_norm
        self.z_shift = z_shift
        self.zbar_shift = zbar_shift
        self.center = center
        self.center_bar = center_bar
        if not folded:
            _z_of_u, u_center = _resolve_center_u(coordinate, center)
            if center_bar is None:
                u_center_bar = u_center
            else:
                _z_of_u_bar, u_center_bar = _resolve_center_u(coordinate, center_bar)
            self._u_center = u_center
            self._u_center_bar = u_center_bar
            self._offdiag = (u_center_bar != u_center)
            # diagonal off-symmetric center: recentered diagonal evaluators (see _NumericBlockSampler)
            self._recentered = u_center != COORDINATES[coordinate].u_symmetric()
        else:
            if center is not None or center_bar is not None:
                raise ValueError(
                    "off-diagonal centers are supported only on non-folded "
                    "coordinates (zzb/yyb/qqb)"
                )
            self._u_center = None
            self._u_center_bar = None
            self._offdiag = False
            self._recentered = False
        self.projected = projected
        self.parity = parity
        if projected:
            from .parity import parity_reversal_phase

            phase = parity_reversal_phase(
                parity_external, exchanged_parity=exchanged_parity
            )
            sign = mp.mpf(1) if parity == "+" else mp.mpf(-1)
            self.swapped_scale = sign * phase
            # the z <-> zbar parity is sign x phase (fold_parity_marker), not the projection sign alone
            self.parity_marker = fold_parity_marker(parity=parity, phase=phase) if folded else None
            if folded and abs(mp.im(self.swapped_scale)) > _validation_tolerance(guard=5):
                raise ValueError(
                    "folded parity-projected coordinates require a real (+/-1) "
                    "parity phase"
                )
        else:
            self.swapped_scale = None
            self.parity_marker = None
        # Reuse a single chiral evaluator when both halves are identical
        # (scalar externals with equal shifts) -- saves one chiral
        # pass and reproduces the scalar-G arithmetic.
        self._same_half = (
            list(self.h_externals) == list(self.hbar_externals)
            and self.z_shift == self.zbar_shift
        )
        self._eval_h = None
        self._eval_nterms = 0
        self._eval_hbar = None
        self._eval_h_n = None
        self._eval_hbar_n = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_eval_h"] = None
        state["_eval_hbar"] = None
        state["_eval_h_n"] = None
        state["_eval_hbar_n"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._eval_h = None
        self._eval_nterms = 0
        self._eval_hbar = None
        self._eval_h_n = None
        self._eval_hbar_n = None

    def _centered_nonfolded_evaluator(
        self, _derivs, h_externals, *, z_shift, n_terms, u_center
    ):
        """Build one non-folded chiral evaluator recentered at ``u_center``."""
        spec = COORDINATES[self.coordinate]
        return _derivs._chiral_series_evaluator(
            h_externals,
            n_terms,
            b=self.b,
            c=self.c,
            q_order=self.q_order,
            chiral_norm=self.chiral_norm,
            z_shift=z_shift,
            z_of_u=spec.z_of_u,
            u_symmetric=u_center,
            coord_key=spec.chiral_var,
        )

    def _ensure_offdiag_evaluators(self, _derivs, n_terms):
        """Build/cache the four off-diagonal chiral evaluators.

        Off-diagonal block ``grid[m][n] = c_h_m[m] c_hbar_n[n] + σ c_hbar_m[m]
        c_h_n[n]``: the m-index uses the m-center evaluators (``_eval_h`` /
        ``_eval_hbar``) and the n-index the n-center evaluators (``_eval_h_n`` /
        ``_eval_hbar_n``); ``h``/``hbar`` distinguish the holomorphic /
        antiholomorphic external weights and z-shift. Shared by ``__call__`` and
        the analytic contribution scorer."""
        if self._eval_h is not None and self._eval_nterms >= n_terms:
            return
        self._eval_h = self._centered_nonfolded_evaluator(
            _derivs, self.h_externals, z_shift=self.z_shift,
            n_terms=n_terms, u_center=self._u_center,
        )
        if self._same_half:
            self._eval_hbar = self._eval_h
        else:
            self._eval_hbar = self._centered_nonfolded_evaluator(
                _derivs, self.hbar_externals, z_shift=self.zbar_shift,
                n_terms=n_terms, u_center=self._u_center,
            )
        self._eval_h_n = self._centered_nonfolded_evaluator(
            _derivs, self.h_externals, z_shift=self.z_shift,
            n_terms=n_terms, u_center=self._u_center_bar,
        )
        if self._same_half:
            self._eval_hbar_n = self._eval_h_n
        else:
            self._eval_hbar_n = self._centered_nonfolded_evaluator(
                _derivs, self.hbar_externals, z_shift=self.zbar_shift,
                n_terms=n_terms, u_center=self._u_center_bar,
            )
        self._eval_nterms = n_terms

    def _n_terms(self, entries=None):
        if self.folded:
            odd = self.parity_marker % 2
            # The chiral series only needs the highest source degree a + 2b
            # any requested folded entry consumes (= Lambda under m + 2n <=
            # Lambda); fall back to the full square when no entry set is given.
            if entries:
                top = max(m + 2 * n for (m, n) in entries)
            else:
                top = self.m_max + 2 * self.n_max
            return top + odd + 1
        return max(self.m_max, self.n_max) + 1

    def _evaluators(self, _derivs, n_terms):
        if getattr(self, "_recentered", False) and not self._offdiag:
            # diagonal off-symmetric center: the same two halves, recentered at u_center
            if self._eval_h is None or self._eval_nterms < n_terms:
                self._eval_h = self._centered_nonfolded_evaluator(
                    _derivs, self.h_externals, z_shift=self.z_shift,
                    n_terms=n_terms, u_center=self._u_center,
                )
                self._eval_hbar = self._eval_h if self._same_half else self._centered_nonfolded_evaluator(
                    _derivs, self.hbar_externals, z_shift=self.zbar_shift,
                    n_terms=n_terms, u_center=self._u_center,
                )
                self._eval_nterms = n_terms
            return self._eval_h, self._eval_hbar
        if self._eval_h is None or self._eval_nterms < n_terms:
            fac_name = (
                self._FOLDED_EVALUATOR if self.folded else self._NONFOLDED_EVALUATOR
            )[self.coordinate]
            self._eval_h = getattr(_derivs, fac_name)(
                self.h_externals, n_terms, b=self.b, c=self.c,
                q_order=self.q_order, chiral_norm=self.chiral_norm,
                z_shift=self.z_shift,
            )
            if self._same_half:
                self._eval_hbar = self._eval_h
            else:
                self._eval_hbar = getattr(_derivs, fac_name)(
                    self.hbar_externals, n_terms, b=self.b, c=self.c,
                    q_order=self.q_order, chiral_norm=self.chiral_norm,
                    z_shift=self.zbar_shift,
                )
            self._eval_nterms = n_terms
        return self._eval_h, self._eval_hbar

    def _analytic_principal_parts(self, roots, entries, exp_base):
        """Exact principal parts at the poles (generic c), or ``None`` to defer to the contour scorer.

        Ordered / parity-projected B specialization of :func:`_analytic_pp_scores`
        with two chiral evaluators -- holomorphic ``(h_externals, z_shift)`` and
        antiholomorphic ``(hbar_externals, zbar_shift)`` -- and ``σ`` the parity
        swap ``self.swapped_scale`` (``None`` for ordered B; a possibly complex
        phase for parity-projected B). Folded coordinates build the non-folded
        principal-part grids and fold them (diagonal, real parity phase only).
        Off-diagonal (``u_center != u_center_bar``) the m-index uses the m-center
        evaluators and the n-index the n-center evaluators (four chiral-factor
        streams). Defers to the contour scorer (``None``) for an unsupported
        context mode, or a folded complex parity phase."""
        self._ensure_precision()
        from . import derivs as _derivs

        ell = self.ell

        if self.folded:
            # Folded projected B forbids a complex parity phase (constructor
            # guard); guard again here lest a complex sigma reach the real-only
            # folded scorer, and defer to the contour if one ever does.
            if isinstance(self.swapped_scale, mp.mpc) and self.swapped_scale.imag != 0:
                return None
            n_terms = self._n_terms()  # full source order m_max + 2 n_max + odd + 1
            eval_h, eval_hbar = self._evaluators(_derivs, n_terms)
            fac_h = _analytic_chiral_factor(eval_h, n_terms, deriv_factor=2)
            fac_hbar = (
                fac_h if eval_hbar is eval_h
                else _analytic_chiral_factor(eval_hbar, n_terms, deriv_factor=2)
            )
            if fac_h is None or fac_hbar is None:
                return None
            if fac_hbar is fac_h:
                _prefetch_chiral_factor(fac_h, roots, ell)
            else:
                _prefetch_chiral_factor(fac_h, roots, ell, halves=(1,))
                _prefetch_chiral_factor(fac_hbar, roots, ell, halves=(-1,))
            return _analytic_principal_parts_folded(
                roots, entries, exp_base,
                factor_m_plus=lambda loc: fac_h((loc + ell) / 2),
                factor_m_minus=lambda loc: fac_hbar((loc - ell) / 2),
                factor_n_plus=lambda loc: fac_h((loc + ell) / 2),
                factor_n_minus=lambda loc: fac_hbar((loc - ell) / 2),
                sigma=self.swapped_scale, fold_marker=self.parity_marker,
                m_max=self.m_max, n_max=self.n_max,
            )

        if self._offdiag:
            n_terms = self._n_terms(entries)
            self._ensure_offdiag_evaluators(_derivs, n_terms)
            fac_h_m = _analytic_chiral_factor(self._eval_h, n_terms, deriv_factor=2)
            fac_hbar_m = (
                fac_h_m if self._eval_hbar is self._eval_h
                else _analytic_chiral_factor(self._eval_hbar, n_terms, deriv_factor=2)
            )
            fac_h_n = _analytic_chiral_factor(self._eval_h_n, n_terms, deriv_factor=2)
            fac_hbar_n = (
                fac_h_n if self._eval_hbar_n is self._eval_h_n
                else _analytic_chiral_factor(self._eval_hbar_n, n_terms, deriv_factor=2)
            )
            if None in (fac_h_m, fac_hbar_m, fac_h_n, fac_hbar_n):
                return None
            for fac_i, halves_i in (
                (fac_h_m, (1,)), (fac_hbar_m, (-1,)),
                (fac_h_n, (1,)), (fac_hbar_n, (-1,)),
            ):
                _prefetch_chiral_factor(fac_i, roots, ell, halves=halves_i)
            return _analytic_principal_parts_axis(
                roots, entries, exp_base,
                factor_m_plus=lambda loc: fac_h_m((loc + ell) / 2),
                factor_m_minus=lambda loc: fac_hbar_m((loc - ell) / 2),
                factor_n_plus=lambda loc: fac_h_n((loc + ell) / 2),
                factor_n_minus=lambda loc: fac_hbar_n((loc - ell) / 2),
                sigma=self.swapped_scale,
            )

        n_terms = self._n_terms()
        eval_h, eval_hbar = self._evaluators(_derivs, n_terms)
        fac_h = _analytic_chiral_factor(eval_h, n_terms, deriv_factor=2)
        fac_hbar = (
            fac_h if eval_hbar is eval_h
            else _analytic_chiral_factor(eval_hbar, n_terms, deriv_factor=2)
        )
        if fac_h is None or fac_hbar is None:
            return None
        if fac_hbar is fac_h:
            _prefetch_chiral_factor(fac_h, roots, ell)
        else:
            _prefetch_chiral_factor(fac_h, roots, ell, halves=(1,))
            _prefetch_chiral_factor(fac_hbar, roots, ell, halves=(-1,))
        return _analytic_principal_parts(
            roots, entries, exp_base,
            factor_h=lambda loc: fac_h((loc + ell) / 2),
            factor_hbar=lambda loc: fac_hbar((loc - ell) / 2),
            sigma=self.swapped_scale,
        )

    def _assemble(
        self,
        batch,
        entries,
        c_h_all,
        c_hbar_all,
        *,
        field_is_arb,
        c_swap_m_all=None,
        c_swap_n_all=None,
    ):
        # The arb/acb-native batched assembly + fused fold is shared with the
        # scalar-G sampler and (in its mpmath form) the per-Delta tables and the
        # global / large-spin samplers; `assemble_block_batch` detects the field
        # from the coefficient arrays, so `field_is_arb` is no longer needed.
        from virasoro_blocks.derivs import assemble_block_batch

        return assemble_block_batch(
            c_h_all, c_hbar_all, self.m_max, self.n_max,
            sigma=self.swapped_scale,
            fold_parity=(self.parity_marker if self.folded else None),
            entries=entries,
            c_swap_m_all=c_swap_m_all,
            c_swap_n_all=c_swap_n_all,
        )

    def _complex_coeffs(self, eval_h, eval_hbar, batch):
        from virasoro_blocks.derivs import _coerce_delta_scalar

        ell = self.ell
        c_h_all, c_hbar_all = [], []
        for d in batch:
            d = _coerce_delta_scalar(d)
            c_h = eval_h(_exchange_weight(d, ell), return_acb=True)
            if c_h is None:
                return None
            if self._same_half and ell == 0:
                c_hbar = c_h
            else:
                c_hbar = eval_hbar(_exchange_weight(d, -ell), return_acb=True)
                if c_hbar is None:
                    return None
            c_h_all.append(c_h)
            c_hbar_all.append(c_hbar)
        return c_h_all, c_hbar_all

    @_quiet_conditioning
    def __call__(self, deltas, entries=None):
        self._ensure_precision()
        from . import derivs as _derivs
        from virasoro_blocks.derivs import _coerce_delta_scalar

        entries = tuple(entries) if entries is not None else None
        batch = [_coerce_delta_scalar(d) for d in deltas]
        if self._offdiag:
            ell = self.ell
            n_terms = self._n_terms(entries)
            self._ensure_offdiag_evaluators(_derivs, n_terms)

            many_h_m = getattr(self._eval_h, "many", None)
            many_hbar_n = getattr(self._eval_hbar_n, "many", None)
            c_h_all = many_h_m([_exchange_weight(d, ell) for d in batch]) if many_h_m else None
            c_hbar_all = (
                many_hbar_n([_exchange_weight(d, -ell) for d in batch]) if many_hbar_n else None
            )
            c_swap_m_all = c_swap_n_all = None
            if c_h_all is not None and c_hbar_all is not None and self.projected:
                many_hbar_m = getattr(self._eval_hbar, "many", None)
                many_h_n = getattr(self._eval_h_n, "many", None)
                c_swap_m_all = (
                    c_h_all if (self._same_half and ell == 0)
                    else (many_hbar_m([_exchange_weight(d, -ell) for d in batch]) if many_hbar_m else None)
                )
                c_swap_n_all = (
                    c_hbar_all if (self._same_half and ell == 0)
                    else (many_h_n([_exchange_weight(d, ell) for d in batch]) if many_h_n else None)
                )

            if (
                c_h_all is None
                or c_hbar_all is None
                or (self.projected and (c_swap_m_all is None or c_swap_n_all is None))
            ):
                c_h_all, c_hbar_all, c_swap_m_all, c_swap_n_all = [], [], [], []
                for d in batch:
                    c_h = self._eval_h(_exchange_weight(d, ell), return_acb=True)
                    c_hbar = self._eval_hbar_n(_exchange_weight(d, -ell), return_acb=True)
                    if c_h is None or c_hbar is None:
                        raise RuntimeError(
                            "ordered/projected numeric sampler requires python-flint"
                        )
                    c_h_all.append(c_h)
                    c_hbar_all.append(c_hbar)
                    if self.projected:
                        if self._same_half and ell == 0:
                            c_swap_m_all.append(c_h)
                            c_swap_n_all.append(c_hbar)
                        else:
                            c_sm = self._eval_hbar(_exchange_weight(d, -ell), return_acb=True)
                            c_sn = self._eval_h_n(_exchange_weight(d, ell), return_acb=True)
                            if c_sm is None or c_sn is None:
                                raise RuntimeError(
                                    "ordered/projected numeric sampler requires python-flint"
                                )
                            c_swap_m_all.append(c_sm)
                            c_swap_n_all.append(c_sn)

            if not self.projected:
                c_swap_m_all = c_swap_n_all = None

            return self._assemble(
                batch,
                entries,
                c_h_all,
                c_hbar_all,
                field_is_arb=True,
                c_swap_m_all=c_swap_m_all,
                c_swap_n_all=c_swap_n_all,
            )

        eval_h, eval_hbar = self._evaluators(_derivs, self._n_terms(entries))
        ell = self.ell

        # All-real fast path: batched arb chiral coefficients for both halves.
        many_h = getattr(eval_h, "many", None)
        c_h_all = many_h([_exchange_weight(d, ell) for d in batch]) if many_h else None
        if c_h_all is not None:
            if self._same_half and ell == 0:
                c_hbar_all = c_h_all
            else:
                many_hbar = getattr(eval_hbar, "many", None)
                c_hbar_all = (
                    many_hbar([_exchange_weight(d, -ell) for d in batch]) if many_hbar else None
                )
            if c_hbar_all is not None:
                return self._assemble(
                    batch, entries, c_h_all, c_hbar_all, field_is_arb=True
                )

        # Complex fallback (complex c / Delta / externals): per-Delta acb.
        coeffs = self._complex_coeffs(eval_h, eval_hbar, batch)
        if coeffs is None:
            raise RuntimeError(
                "ordered/projected numeric sampler requires python-flint"
            )
        return self._assemble(
            batch, entries, coeffs[0], coeffs[1], field_is_arb=False
        )


class _NumericChiralSampler(_PrecisionAwareNumericSampler):
    """Picklable numeric-recursion evaluator of a chiral F table at sample h.

    Returns the same degenerate ``[m][0]`` tables as ``DeltaDerivTable.evaluate``
    on the chiral path, but via the direct fixed-h Taylor builders in
    ``virasoro_blocks.derivs`` rather than symbolic polynomial evaluation.
    """

    _COORD_EVALUATOR_FN = {
        "z": "chiral_z_series_evaluator",
        "y": "chiral_y_series_evaluator",
        "q": "chiral_q_series_evaluator",
    }

    _precision_caches = ('_cache', '_evaluate')

    def __init__(self, coordinate, h_externals, m_max, *, b, c, q_order,
                 chiral_norm, z_shift, center=None):
        m_max = _integer_control('m_max', m_max)
        q_order = _integer_control('q_order', q_order)
        if coordinate not in self._COORD_EVALUATOR_FN:
            raise ValueError(f"no numeric chiral sampler for coordinate {coordinate!r}")
        self.coordinate = coordinate
        from .numeric_input import parse_number
        self.h_externals = [parse_number(value) for value in h_externals]
        self.m_max = m_max
        self.b = b
        self.c = c
        self.q_order = q_order
        self.chiral_norm = chiral_norm
        self.z_shift = z_shift
        self.center = center
        spec = COORDINATES[coordinate]
        # The chiral half F(h, z) has a single expansion point; `center` (a
        # z-value) moves it off the symmetric z = 1/2 (no zbar, so no off-diagonal).
        self._u_center = spec.u_symmetric() if center is None else spec.u_of_z(center)
        self._cache = None
        self._evaluate = None

    @staticmethod
    def _cache_key(h):
        if isinstance(h, Fraction):
            return ("rational", h.numerator, h.denominator)
        if isinstance(h, mp.mpc):
            return ("mpf", mp.mpf(h.real)._mpf_) if h.imag == 0 else ("mpc", h._mpc_)
        return ("mpf", mp.mpf(h)._mpf_)

    def _evaluator(self, _derivs):
        """The chiral series evaluator (built once): the public one at the symmetric point, else recentered."""
        if self._evaluate is None:
            if self.center is None:
                # Symmetric: the public evaluator (bit-identical to production).
                self._evaluate = getattr(
                    _derivs, self._COORD_EVALUATOR_FN[self.coordinate]
                )(
                    self.h_externals,
                    self.m_max + 1,
                    b=self.b,
                    c=self.c,
                    q_order=self.q_order,
                    chiral_norm=self.chiral_norm,
                    z_shift=self.z_shift,
                )
            else:
                # Off-center: recenter the single chiral expansion at u(center).
                spec = COORDINATES[self.coordinate]
                self._evaluate = _derivs._chiral_series_evaluator(
                    self.h_externals,
                    self.m_max + 1,
                    b=self.b,
                    c=self.c,
                    q_order=self.q_order,
                    chiral_norm=self.chiral_norm,
                    z_shift=self.z_shift,
                    z_of_u=spec.z_of_u,
                    u_symmetric=self._u_center,
                    coord_key=spec.chiral_var,
                )
        return self._evaluate

    def _analytic_principal_parts(self, roots, entries, exp_base):
        """Exact principal parts at the Kac weights (generic c), or ``None`` to defer to the contour.

        The chiral table ``F[m][0](h)`` is the scalar-G scorer's axis form with the chiral factor on the m-axis
        (score variable h itself, so no Jacobian) and a unit n-factor (``A = 0``, ``B = 1``): its principal part at a
        Kac weight is then ``A_m / (h - loc)``.  The contour/DFT scorer it replaces took ~2000 complex-h
        evaluations per table.
        """
        self._ensure_precision()
        from flint import acb

        from . import derivs as _derivs

        fac = _analytic_chiral_factor(self._evaluator(_derivs), self.m_max + 1, deriv_factor=1)
        if fac is None:
            return None
        unit = ([acb(0)], [acb(1)])
        return _analytic_principal_parts_axis(
            roots, entries, exp_base,
            factor_m_plus=fac,
            factor_m_minus=lambda loc: unit,
            factor_n_plus=lambda loc: unit,
            factor_n_minus=lambda loc: unit,
            sigma=None,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = None
        state["_evaluate"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @_quiet_conditioning
    def __call__(self, hs, entries=None):
        self._ensure_precision()
        # `entries` (the per-target restriction used by the block sampler) does
        # not apply to the degenerate chiral table; accepted for a uniform
        # sampler call signature and ignored.
        from . import derivs as _derivs

        batch = [_derivs._coerce_delta_scalar(h) for h in hs]
        if self._cache is None:
            self._cache = {}
        cache = self._cache
        tables = [None] * len(batch)
        flags = []
        missing = []
        missing_keys = []
        missing_index_groups = []
        pending_positions = {}
        for i, h in enumerate(batch):
            key = self._cache_key(h)
            cached = cache.get(key)
            if cached is None:
                if key in pending_positions:
                    pending_positions[key].append(i)
                else:
                    pending_positions[key] = [i]
                    missing.append(h)
                    missing_keys.append(key)
                    missing_index_groups.append(pending_positions[key])
            else:
                table, is_arb = cached
                tables[i] = table
                flags.append(is_arb)

        if not missing:
            return tables, all(flags) if flags else False

        self._evaluator(_derivs)

        # Batched real-h fast path: evaluator.many returns native arb
        # coefficients in one flint pipeline, avoiding per-h Python calls and
        # keeping interpolation sample values in flint space.
        many = getattr(self._evaluate, "many", None)
        if many is not None:
            coeffs_many = many(missing)
            if coeffs_many is not None:
                fetched = [
                    [[coeffs[m]] for m in range(self.m_max + 1)]
                    for coeffs in coeffs_many
                ]
                for indices, key, table in zip(missing_index_groups, missing_keys, fetched):
                    cache[key] = (table, True)
                    for i in indices:
                        tables[i] = table
                return tables, all(flags)

        fetched = []
        for h in missing:
            coeffs = self._evaluate(h)
            fetched.append([[coeffs[m]] for m in range(self.m_max + 1)])

        for indices, key, table in zip(missing_index_groups, missing_keys, fetched):
            cache[key] = (table, False)
            for i in indices:
                tables[i] = table
        return tables, False


def build_numeric_sampler(
    coordinate, *, chiral, spinning, projected,
    h_externals, hbar_externals, ell, m_max, n_max,
    b=None, c, q_order, chiral_norm,
    z_shift=None, zbar_shift=None, center=None, center_bar=None,
    parity=None, parity_external=None, exchanged_parity="1",
):
    """Construct the robust numeric-recursion block sampler for one basis.

    Dispatches on the legacy ``(chiral, spinning, projected)`` flags; ``spinning``
    selects ordered B even when its external weight lists coincide. The samplers are
    :class:`_NumericChiralSampler` (chiral F, one Taylor axis, sampled in h),
    :class:`_NumericOrderedBlockSampler` (ordered / parity-projected B, including
    their scalar-external limits),
    or :class:`_NumericBlockSampler` (scalar G). All three return the same
    ``[m][n]`` block tables as :meth:`DeltaDerivTable.evaluate` but via the O(1)
    recursion rather than the ill-conditioned symbolic ``P(Delta)/D(Delta)``.

    Shared by the block CLI's interpolation node sampling and the point-constraint
    builder so the two stay in lock-step across every basis.
    """
    m_max = _integer_control('m_max', m_max)
    if n_max is not None or not (chiral or coordinate.endswith('_radial')):
        n_max = _integer_control('n_max', n_max)
    q_order = _integer_control('q_order', q_order)
    # A radial slice is the n = 0 column (squared variable fixed at 0) of its
    # parity-folded parent, so it is sampled by the parent's folded evaluator
    # restricted to n_max = 0 -- exactly the (m_max + 1) x 1 radial table shape.
    if coordinate.endswith("_radial"):
        coordinate = coordinate[: -len("_radial")]
        n_max = 0
    if chiral:
        return _NumericChiralSampler(
            coordinate, h_externals, m_max,
            b=b, c=c, q_order=q_order, chiral_norm=chiral_norm,
            z_shift=z_shift, center=center,
        )
    if spinning or projected:
        return _NumericOrderedBlockSampler(
            coordinate, h_externals, hbar_externals, ell, m_max, n_max,
            b=b, c=c, q_order=q_order, chiral_norm=chiral_norm,
            z_shift=z_shift, zbar_shift=zbar_shift, projected=projected,
            parity=parity, parity_external=parity_external,
            exchanged_parity=exchanged_parity, center=center, center_bar=center_bar,
        )
    return _NumericBlockSampler(
        coordinate, h_externals, ell, m_max, n_max,
        b=b, c=c, q_order=q_order, chiral_norm=chiral_norm,
        z_shift=z_shift, center=center, center_bar=center_bar,
    )


def _evaluate_table_flint_many(
    exact: DeltaDerivTable,
    deltas,
    *,
    entries=None,
    flint_table_cache: dict | None = None,
):
    """Evaluate a DeltaDerivTable at multiple real Delta values using flint.

    Converts all polynomials to arb_poly once (amortised over all sample
    points), then evaluates them at every point with one Vandermonde arb_mat product.
    Returns arb values directly (no mpf conversion) when possible, for use
    in the flint-native Lagrange sum. Falls back to mpmath path for complex-
    coefficient tables.

    When the table carries a ``numeric_sampler``, the well-conditioned numeric
    recursion provides the sample values instead -- avoiding the ~1e500-coeff
    symbolic evaluation that needs very high dps to survive cancellation.
    """
    sampler = getattr(exact, "numeric_sampler", None)
    if sampler is not None:
        return sampler(deltas, entries=entries)
    is_real = (
        (isinstance(exact.exp_base, mp.mpf) or
         (isinstance(exact.exp_base, mp.mpc) and exact.exp_base.imag == 0))
        and exact.denominator.field is mp.mpf
        and all(
            num.field is mp.mpf
            for row in exact.numerators
            for num in row
        )
    )
    if not is_real:
        return [exact.evaluate(d) for d in deltas], False
    from flint import arb, arb_mat, arb_poly, ctx as flint_ctx

    prec_bits = int(mp.mp.dps * 3.33) + 20
    flint_ctx.prec = prec_bits

    cache_tag = (id(exact.numerators), id(exact.denominator), prec_bits)
    if flint_table_cache is not None and flint_table_cache.get("tag") == cache_tag:
        exp_base_arb = flint_table_cache["exp_base_arb"]
        arb_denom = flint_table_cache["arb_denom"]
        row_slices = flint_table_cache["row_slices"]
        n_cols = flint_table_cache["n_cols"]
        Nmat = flint_table_cache["Nmat"]
    else:
        # exp_base may be mpc(real, 0) after regularization; extract real part
        eb = exact.exp_base
        if isinstance(eb, mp.mpc):
            eb = mp.mpf(eb.real)
        exp_base_arb = _mpf_to_arb(eb)

        # The Delta-independent denominator stays a polynomial: it is evaluated once
        # per sample point (cheap) and is not part of the per-(m, n) assembly.
        arb_denom = arb_poly([_mpf_to_arb(c) for c in exact.denominator.coeffs])

        # Every (m, n) numerator is a polynomial in the *same* Delta, so evaluating
        # the whole table at all sample points is one Vandermonde matrix product:
        #   P = N @ V,  N[e] = coeffs of numerator_e (zero-padded to the max degree),
        #   V[i][j] = Delta_j ** i,  P[e][j] = numerator_e(Delta_j).
        # One compiled arb_mat multiply evaluates all entries and nodes, with
        # Arb tracking the rounding error. row_slices preserves the ragged
        # layout of numerators.
        flat_nums = []
        row_slices = []
        start = 0
        for row in exact.numerators:
            flat_nums.extend(row)
            row_slices.append((start, start + len(row)))
            start += len(row)

        n_cols = max((p.degree + 1 for p in flat_nums), default=1)
        zero = arb(0)
        N_rows = []
        for p in flat_nums:
            # Reuse the cached arb_poly; _get_arb converts coefficients only
            # when no cached representation is available.
            row = list(p._get_arb().coeffs())[:n_cols]
            if len(row) < n_cols:
                row = row + [zero] * (n_cols - len(row))
            N_rows.append(row)
        Nmat = arb_mat(N_rows)
        if flint_table_cache is not None:
            flint_table_cache.clear()
            flint_table_cache.update({
                "tag": cache_tag,
                "exp_base_arb": exp_base_arb,
                "arb_denom": arb_denom,
                "row_slices": row_slices,
                "n_cols": n_cols,
                "Nmat": Nmat,
            })

    da_list = [_mpf_to_arb(mp.mpf(d)) for d in deltas]
    nd = len(da_list)
    V_rows = [[arb(1)] * nd]
    for _ in range(1, n_cols):
        prev = V_rows[-1]
        V_rows.append([prev[j] * da_list[j] for j in range(nd)])
    Vmat = arb_mat(V_rows)

    P = Nmat * Vmat  # (#entries) x (#deltas), P[e, j] = numerator_e(Delta_j)

    results = []
    for j, d_arb in enumerate(da_list):
        pref_arb = exp_base_arb ** d_arb
        D_arb = arb_denom(d_arb)
        pref_over_D = pref_arb / D_arb
        results.append([
            [P[e, j] * pref_over_D for e in range(s, end)]
            for (s, end) in row_slices
        ])
    return results, True  # True = values are arb


def _entry_sample_points(kept, n_use, sample_kwargs, node_subset_mode, shared_sx):
    """The ``n_use`` fit nodes for one entry-degree group.

    ``node_subset_mode`` controls where they come from:

    ``"off"`` (default, historical)
        Each group runs its own Bohr-Sommerfeld search. Optimal for that group's
        measure, but the nodes are NOT a subset of the shared set handed to SDPB,
        so a low-degree entry carries interpolation error at the points the SDP
        actually evaluates it (measured ~1e-12 at npoles=16, growing with
        derivative order; the top-order entry is exact there by construction).

    ``"nearest"``
        Run the same BS search, then snap each node to the nearest point of the
        shared set. Keeps the optimal spacing as closely as the shared grid
        allows; can collapse two BS nodes onto one shared node, in which case the
        next-nearest unused node is taken so the count is preserved.

    ``"first"``
        Simply take the first ``n_use`` points of the shared set. Cheapest, and a
        deliberately naive control for ``"nearest"`` -- the shared set is ordered
        from the threshold outward, so this concentrates every low-degree entry
        near the threshold and abandons the outer region.

    Both subset modes leave each entry's DEGREE untouched, which is the property
    that ``shared_sample_points`` destroys: padding every entry onto the
    max-degree node set raises the low-order degrees and wrecks the large-Delta
    tail power (ratio 5.6e14 off at npoles=16 by Delta=400), even though it makes
    every entry exact at the shared nodes.
    """
    if node_subset_mode == "off" or not shared_sx:
        return get_sample_points(kept, n_use, **sample_kwargs)

    pool = sorted(mp.mpf(x) for x in shared_sx)
    if n_use >= len(pool):
        return list(pool)
    if node_subset_mode == "first":
        return pool[:n_use]
    if node_subset_mode == "nearest":
        want = get_sample_points(kept, n_use, **sample_kwargs)
        used: set[int] = set()
        out = []
        for w in want:
            order = sorted(range(len(pool)), key=lambda i: abs(pool[i] - mp.mpf(w)))
            for i in order:
                if i not in used:
                    used.add(i)
                    out.append(pool[i])
                    break
        return sorted(out)
    raise ValueError(f"unknown node_subset_mode {node_subset_mode!r}")


def _interpolate_per_target_task(args):
    """Per-target sample-point + exact-evaluate + Lagrange-fit job.

    Run as a worker function in a multiprocessing pool. Each call handles
    one distinct sample-point count `n_use` across all `(m, n)` entries that
    share that target. Workers receive the prefactor and exact table via
    pickle; the worker reconstructs the mpmath state, performs the BS run,
    evaluates the exact table at the sample points, and fits Lagrange
    polynomials for each entry. Returns the (m, n) -> (Polynomial,
    sample_delta_tuple, samples) mapping for the caller to assemble into the final
    InterpolatedDeltaTable.

    Defined at module level so the function pickles cleanly across processes.
    """
    (
        dps, n_use, kept, forced_roots, shift, real_shift,
        sample_kwargs, min_pole_distance, exact, entries,
        flint_table_cache, basis_cache, node_subset_mode, shared_sx,
    ) = args
    mp.mp.dps = dps
    local_sx = _entry_sample_points(
        kept, n_use, sample_kwargs, node_subset_mode, shared_sx
    )
    local_sd = _shifted_sampling_points(local_sx, shift)
    # arb prefactor values straight from the arb evaluator (a real constant); mpmath values otherwise
    local_pv_arb = kept.evaluate_arb_batch(local_sx) if min_pole_distance is None else None
    if local_pv_arb is not None:
        local_pv = None
    elif min_pole_distance is None:
        local_pv = [kept.evaluate(x) for x in local_sx]
    else:
        local_pv = [
            kept.evaluate_regulated(x, min_pole_distance) for x in local_sx
        ]
    if forced_roots:
        local_ff = [_product_over_roots(d, forced_roots) for d in local_sd]
    else:
        local_ff = None
    local_ex, is_arb = _evaluate_table_flint_many(
        exact,
        local_sd,
        entries=entries,
        flint_table_cache=flint_table_cache,
    )
    sample_tuple = tuple(local_sd)
    # Precompute Lagrange basis polynomials once -- all entries with this
    # target share the same sample nodes, so the basis is identical. Each
    # entry's polynomial is then a sum of N scaled-basis-polynomial adds,
    # O(N²) per entry rather than O(N³) for a from-scratch Lagrange call.
    basis_key = (sample_tuple, shift)
    if basis_cache is not None and basis_key in basis_cache:
        basis, denominators, arb_basis, arb_denoms = basis_cache[basis_key]
    else:
        basis, denominators, arb_basis, arb_denoms = precompute_lagrange_basis(
            local_sx,
        )
        if basis_cache is not None:
            basis_cache[basis_key] = (basis, denominators, arb_basis, arb_denoms)

    # If values are arb and prefactor values can be converted to arb,
    # do division in arb space to feed the flint Lagrange path directly.
    if is_arb:
        try:
            from flint import arb as arb_type, ctx as flint_ctx
            prec_bits = int(dps * 3.33) + 20
            flint_ctx.prec = prec_bits
            if local_pv_arb is None:
                local_pv_arb = [arb_type(v) for v in local_pv]
            if local_ff is not None:
                local_ff_arb = [arb_type(v) for v in local_ff]
            else:
                local_ff_arb = None
            use_arb_values = True
        except TypeError:
            # arb() rejects mpc prefactor / forced-factor values (a complex
            # constant or conjugate forced roots): mpmath division.
            use_arb_values = False
    else:
        use_arb_values = False
    if not use_arb_values and local_pv is None:
        local_pv = [kept.evaluate(x) for x in local_sx]

    results: dict = {}
    for (m, n) in entries:
        if use_arb_values:
            if local_ff_arb is None:
                reduced = [
                    local_ex[i][m][n] / local_pv_arb[i] for i in range(n_use)
                ]
            else:
                reduced = [
                    local_ex[i][m][n] * local_ff_arb[i] / local_pv_arb[i]
                    for i in range(n_use)
                ]
            # Pass arb values directly — no mpf round-trip
        else:
            if local_ff is None:
                reduced = [
                    local_ex[i][m][n] / local_pv[i] for i in range(n_use)
                ]
            else:
                reduced = [
                    local_ex[i][m][n] * local_ff[i] / local_pv[i]
                    for i in range(n_use)
                ]
        poly = lagrange_polynomial_from_basis(
            basis, denominators, reduced,
            _arb_basis=arb_basis, _arb_denoms=arb_denoms,
        )
        results[(m, n)] = (poly, sample_tuple, _raw_entry_samples(local_ex, is_arb, m, n, reduced))
    return results


def _raw_entry_samples(tables, is_arb, m, n, reduced=None):
    """One entry's samples at its fit nodes: ``(is_arb, raw block values, fitted numerator values)``.

    The numerator values are what the Lagrange fit interpolated (the raw values over the prefactor); a consumer
    evaluating the fitted numerator elsewhere uses them instead of re-dividing.  Arb values are stored as their exact
    dyadic midpoints (the records cross process boundaries).
    """
    as_mpf = _arb_to_mpf if is_arb else (lambda value: value)
    raw = tuple(as_mpf(table[m][n]) for table in tables)
    return is_arb, raw, None if reduced is None else tuple(
        _arb_to_mpf(value) if not isinstance(value, (mp.mpf, mp.mpc)) else value for value in reduced
    )


def _product_over_roots(delta, roots):
    """Compute ∏(delta − root) staying in mp.mpf when delta and every root
    are real.  The real product is taken in flint arb."""
    all_real = (
        not isinstance(delta, mp.mpc) or delta.imag == 0
    ) and all(
        not isinstance(r, mp.mpc) or r.imag == 0 for r in roots
    )
    if all_real:
        from flint import arb, ctx as flint_ctx

        d = mp.mpf(mp.re(delta) if isinstance(delta, mp.mpc) else delta)
        prec_bits = int(mp.mp.dps * 3.33) + 20
        flint_ctx.prec = prec_bits
        d_arb = arb(d)
        acc = arb(1)
        for r in roots:
            r_real = mp.re(r) if isinstance(r, mp.mpc) else r
            acc = acc * (d_arb - arb(r_real))
        ndigits = int(mp.mp.dps) + 5
        return mp.mpf(acc.str(ndigits, radius=False))
    d = mp.mpc(delta)
    acc = mp.mpc(1)
    for r in roots:
        acc = acc * (d - mp.mpc(r))
    return acc


def interpolate_delta_table(
    exact: DeltaDerivTable,
    npoles: int | None,
    *,
    pole_selection: str = "contribution_optimal",
    n_sample_points: int | None = None,
    shift=None,
    min_pole_distance=None,
    sample_point_kwargs: dict | None = None,
    node_search_dps: int | None = None,
    delta_eval=None,
    pre_stripped: bool = False,
    force_keep_real_positive: bool = True,
    num_threads: int = 1,
    shared_sample_points: bool = False,
    node_subset_mode: str = "off",
    pole_ranking: str = "numeric",
) -> InterpolatedDeltaTable:
    """Order-reduce a single exact `DeltaDerivTable` via optimal interpolation.

    Thin single-coordinate wrapper around `interpolate_delta_tables`; see that
    function for the algorithm, the `pole_selection` choices
    (`"contribution_optimal"` is the production default), the
    `n_sample_points` heuristic, and the
    `force_keep_real_positive` flag that controls the c <= 1 real-positive
    pole handling. The `"residue"` selector aggregates scores across
    coordinates when called via `interpolate_delta_tables`, so
    multi-coordinate residue is the right call -- this single-table entry
    point is for the bootstrap pipeline where one coordinate (`zzb`) drives
    the interpolation.

    `pre_stripped=True` skips the internal removable-pole stripping step; use
    this when sweeping many npoles over a single pre-stripped table to avoid
    repeating the O(N_poles × N_entries × poly_degree) stripping work each call.
    """
    result = interpolate_delta_tables(
        {"_": exact},
        npoles,
        pole_selection=pole_selection,
        n_sample_points=n_sample_points,
        shift=shift,
        min_pole_distance=min_pole_distance,
        sample_point_kwargs=sample_point_kwargs,
        node_search_dps=node_search_dps,
        delta_eval=delta_eval,
        pre_stripped=pre_stripped,
        force_keep_real_positive=force_keep_real_positive,
        num_threads=num_threads,
        shared_sample_points=shared_sample_points,
        node_subset_mode=node_subset_mode,
        pole_ranking=pole_ranking,
    )
    return result["_"]
