"""Tests for the exact Laurent-in-t regularization path.

The Laurent path carries every recursion quantity as a Laurent series in
t = sqrt(c - c0) and reads off the t^0 coefficient; the regularized wrappers
read the same table as merged poles in h (the block AT c0). These tests do two
things:

  1. Sanity-check the LaurentSeries arithmetic and the b(t) construction.
  2. Cross-check the merged-pole blocks against the fixed-h t^0 coefficients
     and the bare recursion just off c0 (and, where available, against exact
     closed forms).
"""

from __future__ import annotations

from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_blocks.params import b_from_c, b_series
from virasoro_blocks.recursion import H_q_series, H_q_series_laurent
from virasoro_blocks.regularization import (
    B_block_regularized,
    F_block_regularized,
    G_block_regularized,
    H_q_series_regularized,
)
from virasoro_blocks.tseries import LaurentSeries, T_LO


# --- LaurentSeries sanity ---


def test_laurent_series_reciprocal_geometric_series():
    """1/(1+t) = 1 - t + t^2 - t^3 + ..."""
    mp.mp.dps = 50
    one = LaurentSeries.constant(1)
    t = LaurentSeries.monomial(1)
    inv = one / (one + t)
    for k in range(6):
        expected = mp.mpc(1 if k % 2 == 0 else -1)
        assert abs(inv.coeff(k) - expected) < mp.mpf("1e-40")


def test_laurent_series_reciprocal_picks_up_negative_powers():
    """1/(t + t^2) starts at t^-1: the structural-zero detector finds valuation 1."""
    mp.mp.dps = 50
    t = LaurentSeries.monomial(1)
    one = LaurentSeries.constant(1)
    inv = one / (t + t * t)
    assert inv.valuation() == -1
    # Series: t^-1 - 1 + t - t^2 + ...
    for k, expected in [(-1, 1), (0, -1), (1, 1), (2, -1)]:
        assert abs(inv.coeff(k) - mp.mpc(expected)) < mp.mpf("1e-40")


def test_laurent_series_sqrt_pulls_out_even_valuation():
    """sqrt(4 t^2 + 4 t^3) = 2t + t^2 - t^3/4 + ..."""
    mp.mp.dps = 50
    s = LaurentSeries.monomial(2, 4) + LaurentSeries.monomial(3, 4)
    root = s.sqrt()
    assert root.valuation() == 1
    assert abs(root.coeff(1) - mp.mpc(2)) < mp.mpf("1e-40")
    assert abs(root.coeff(2) - mp.mpc(1)) < mp.mpf("1e-40")
    assert abs(root.coeff(3) - mp.mpc("-0.25")) < mp.mpf("1e-40")


def test_laurent_series_tolerance_treats_tiny_leading_as_zero():
    """A leading coefficient ~10^-(dps/2) below scale is structurally zero."""
    mp.mp.dps = 50
    s = LaurentSeries.zero()
    # Place a tiny value at t^0 and an O(1) value at t^1.
    s.coeffs[0 - T_LO] = mp.mpf(10) ** (-45)
    s.coeffs[1 - T_LO] = mp.mpf(2)
    assert s.valuation() == 1


def test_laurent_coefficient_edits_invalidate_native_and_structural_caches():
    import copy
    import pickle
    import virasoro_blocks.tseries as ts
    from flint import acb, arb

    with mp.workdps(50):
        for source in [LaurentSeries.zero(), LaurentSeries.constant(1),
                       LaurentSeries.constant(1) + LaurentSeries.monomial(1)]:
            source.is_zero()
            original_row = ts._coeff_row(source, ts._flint_prec_bits(), acb(0), False)
            view = source.coeffs
            # Retain the view across another cache fill: invalidation merely
            # on property access would still leave this edit stale.
            _ = source * 2
            view[-T_LO] = mp.mpc(3)
            assert not source.is_zero()
            assert (source * 2).coeff(0) == 6
            assert ts._coeff_row(source, ts._flint_prec_bits(), acb(0), False) is not original_row
            view[-T_LO:-T_LO+1] = [mp.mpc(4)]
            assert (source * 2).coeff(0) == 8
            cloned = copy.deepcopy(source)
            assert (cloned * 2).coeff(0) == 8
            cloned.coeffs[-T_LO] = mp.mpc(5)
            assert source.coeff(0) == 4 and (cloned * 2).coeff(0) == 10
            # The cached native handle must not prevent a value round trip.
            ts._coeff_row(source, ts._flint_prec_bits(), acb(0), False)
            restored = pickle.loads(pickle.dumps(source))
            assert (restored * 2).coeff(0) == 8
            complex_row = ts._coeff_row(source, ts._flint_prec_bits(), acb(0), False)
            real_row = ts._coeff_row(source, ts._flint_prec_bits(), arb(0), True)
            assert isinstance(complex_row[0][-T_LO], acb)
            assert isinstance(real_row[0][-T_LO], arb)
        plain = LaurentSeries.constant(1)
        plain.coeffs[-T_LO] = mp.mpc(3)
        restored = pickle.loads(pickle.dumps(plain))
        restored.coeffs[-T_LO] = mp.mpc(4)
        assert plain.coeff(0) == 3 and (restored * 2).coeff(0) == 8


def test_laurent_coefficient_views_preserve_width_and_do_not_alias_replacements():
    with mp.workdps(50):
        source = LaurentSeries.constant(3)
        view = source.coeffs
        expected = list(view)
        for mutation in [lambda: view.append(1), lambda: view.clear(), lambda: view.pop(),
                         lambda: view.__setitem__(slice(None), []), lambda: view.__iadd__([1]),
                         lambda: view.__imul__(2)]:
            with pytest.raises(ValueError, match='window length'):
                mutation()
            assert view == expected and (source * 2).coeff(0) == 6
        source.coeffs = expected
        expected[-T_LO] = mp.mpc(9)
        view[-T_LO] = mp.mpc(8)
        assert source.coeff(0) == 3 and (source * 2).coeff(0) == 6
        with pytest.raises(ValueError, match='length'):
            source.coeffs = []
        assert source.coeff(0) == 3


def test_numeric_laurent_prefactor_uses_the_same_local_variable_as_the_recursion():
    import virasoro_blocks.tseries as ts
    from virasoro_blocks.params import laurent_t_power, laurent_t_sq_sign
    from virasoro_blocks.recursion import block_prefactor
    from virasoro_blocks.regularization import _series_geometry

    with mp.workdps(50):
        ext = [mp.mpf(1)/16]*4
        for c in [mp.mpf('.5'), 1, 25, 28]:
            _, _, _, pieces = _series_geometry('z', ext, 1, c0=c, q_order=4,
                                               chiral_norm=16, z_shift=0)
            p, sign = laurent_t_power(c), laurent_t_sq_sign(c)
            # A negative-power probe selects each central-charge Taylor term,
            # including the highest term the recursion window can consume.
            for order in [1, ts.T_HI//p]:
                probe = [LaurentSeries.monomial(-p*order)]
                actual = ts.u_numeric_convolve_sum([(values, probe, shift) for shift, values in pieces], 1)[0].coeff(0)
                expected = sign**order * mp.diff(
                    lambda charge: block_prefactor(0, mp.mpf('.5'), ext, c=charge, chiral_norm=16),
                    c, order) / mp.factorial(order)
                assert abs(actual/expected - 1) < mp.mpf('1e-40')


def test_laurent_batch_helpers_validate_shapes_even_for_zero_inputs():
    import virasoro_blocks.tseries as ts

    for value in [LaurentSeries.zero(), LaurentSeries.constant(2)]:
        for ws, hs in [([], [value]), ([value], []), ([value, value], [1])]:
            with pytest.raises(ValueError, match='equal lengths'):
                ts.t_conv_weighted_sum(ws, hs)
        for weights in [[[1, 2]], [[1], []]]:
            with pytest.raises(ValueError, match='one entry per series'):
                ts.numeric_linear_map(weights, [value])
        with pytest.raises(ValueError, match='length n_terms'):
            ts.StackedURows([[value]], 2)
        stacked = ts.StackedURows([[value], None], 1)
        for weights in [[], [None], [value, value, value]]:
            with pytest.raises(ValueError, match='one entry per stacked'):
                ts.stacked_weighted_sum(weights, stacked)


def test_laurent_batch_helpers_preserve_scalar_and_series_contracts():
    import virasoro_blocks.tseries as ts

    constant = ts.t_conv_weighted_sum([2, 3], [4, 5])
    assert isinstance(constant, LaurentSeries)
    assert constant.coeff(0) == 23 and constant.coeff(1) == 0
    t = LaurentSeries.monomial(1)
    inverse_t = LaurentSeries.monomial(-1)
    mapped = ts.numeric_linear_map([[2, 3], [-1, 4]], [t, inverse_t])
    assert [(r.coeff(1), r.coeff(-1)) for r in mapped] == [(2, 3), (-1, 4)]
    stacked = ts.StackedURows([[t, inverse_t], None, [inverse_t, t]], 2)
    combined = ts.stacked_weighted_sum([inverse_t, 9, t], stacked)
    assert combined[0].coeff(0) == 2
    assert combined[1].coeff(-2) == 1 and combined[1].coeff(2) == 1
    empty_columns = ts.numeric_linear_map([[], []], [])
    assert len(empty_columns) == 2 and all(row.is_zero() for row in empty_columns)


# --- b_series consistency at recognized singular loci ---


@pytest.mark.parametrize(
    "c0",
    [mp.mpf(1), mp.mpf(25), mp.mpf(1) / 2, -mp.mpf(22) / 5, mp.mpf(28)],
)
def test_b_series_constant_term_matches_b_from_c(c0):
    """b(t)|_{t=0} agrees with the scalar b_from_c(c0)."""
    mp.mp.dps = 80
    b = b_series(c0)
    assert abs(b.coeff(0) - b_from_c(c0)) < mp.mpf("1e-60")


def test_laurent_parameter_helpers_retain_exact_branch_labels():
    from decimal import Decimal
    from virasoro_blocks.params import beta_series, imaginary_b_locus, laurent_t_power, laurent_t_sq_sign

    with mp.workdps(100):
        native = 1 + mp.mpf('1e-70')
    with mp.workdps(32):
        for center in (1, 25):
            for sign in (-1, 1):
                value = Fraction(center) + sign*Fraction(1, 10**70)
                for label in [value, str(value)]:
                    assert laurent_t_power(label) == 1
                    assert laurent_t_sq_sign(label) == 1
                    assert imaginary_b_locus(label) == (value < 1)
        assert laurent_t_power(native) == laurent_t_sq_sign(native) == 1
        assert not imaginary_b_locus(native)
        for label in ['.7', .7, Decimal('.7'), Fraction(7, 10)]:
            b, beta = b_series(label), beta_series(label)
            for k in range(4):
                assert abs(b.coeff(k) - 1j*beta.coeff(k)) < mp.mpf('1e-30')
        b, beta = b_series(1), beta_series(1)
        assert abs(b.coeff(1) + 1j/mp.sqrt(24)) < mp.mpf('1e-30')
        assert abs(beta.coeff(1) + 1/mp.sqrt(24)) < mp.mpf('1e-30')
        for invalid in (True, mp.nan, mp.inf, -mp.inf):
            assert not imaginary_b_locus(invalid)
            with pytest.raises(ValueError):
                b_series(invalid)
        for invalid in (2, mp.mpc('.5', '.1')):
            with pytest.raises(ValueError, match='real c0 <= 1'):
                beta_series(invalid)
        assert mp.mp.dps == 32


# --- H_q_series_laurent: at generic c it must equal H_q_series ---


def test_H_q_series_laurent_matches_plain_at_generic_c():
    """At a generic non-singular c, the Laurent t^0 coefficient is just the
    plain numeric H_k. The two paths agree to working precision."""
    mp.mp.dps = 50
    h = mp.mpf("1.3")
    d = [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")]
    plain = H_q_series(h, d, c=mp.mpf(5), q_order=8)
    laurent = H_q_series_laurent(h, d, c0=mp.mpf(5), q_order=8)
    worst = max(abs(a - b) for a, b in zip(plain, laurent))
    assert worst < mp.mpf("1e-40")


def test_laurent_recursion_accepts_equivalent_exact_input_labels():
    from decimal import Decimal
    from virasoro_blocks.recursion import H_q_series_laurent_precompute

    with mp.workdps(40):
        reference = H_q_series(Fraction(1, 3), [Fraction(1, 16)]*4,
                               c=Fraction(1, 2), q_order=2)
        for c in [Fraction(1, 2), Decimal('.5'), '.5', .5]:
            values = H_q_series_laurent('1/3', ['1/16']*4, c0=c, q_order=2)
            assert max(abs(a-b) for a, b in zip(values, reference)) < mp.mpf('1e-37')
            state = H_q_series_laurent_precompute(['1/16']*4, c0=c, q_order=2)
            assert isinstance(state['c0'], Fraction)
            assert state['h_ext'] == (Fraction(1, 16),)*4
        assert mp.mp.dps == 40


def test_laurent_precompute_preserves_adjacent_bit_precisions():
    from virasoro_blocks.recursion import H_q_series_laurent_precompute

    states = []
    for precision in [199, 200]:
        with mp.workprec(precision):
            states.append(H_q_series_laurent_precompute(['1/16']*4, c0='.5', q_order=1))
            assert states[-1]['work_prec'] >= precision and mp.mp.prec == precision
    assert states[0] is not states[1]


@pytest.mark.slow  # 5 s at q_order 40: the only guard of this cache key, not a fast-tier check
def test_H_q_series_laurent_widens_window_and_avoids_stale_cache():
    """At c=28, q=40 needs a deeper t-window than the module default.

    A previous cache key omitted the Laurent window size, so a failed narrow
    build could poison a later wider build with stale fixed-width series.
    """
    import virasoro_blocks.recursion as R
    import virasoro_blocks.tseries as ts

    mp.mp.dps = 80
    h = mp.mpf("1.3")
    h_ext = [mp.mpf("0.075")] * 4
    c = mp.mpf(28)
    q_order = 40
    saved = ts.window_half()
    try:
        R._H_LAURENT_PRECOMP_CACHE.clear()
        ts.set_window(12)
        got = H_q_series_laurent(
            h, h_ext, c0=c, q_order=q_order, work_dps=80,
        )
        assert abs(got[-1] - mp.mpc("66835660851.8867803")) < mp.mpf("1e-3")

        ts.set_window(16)
        got_wide = H_q_series_laurent(
            h, h_ext, c0=c, q_order=q_order, work_dps=80,
        )
        assert abs(got_wide[-1] - got[-1]) < mp.mpf("1e-3")
    finally:
        ts.set_window(saved)
        R._H_LAURENT_PRECOMP_CACHE.clear()


# --- Cross-check the merged-pole blocks at recognized singular loci ---


_H_CROSS_CHECK_LOCI = [
    pytest.param(mp.mpf(1), id="c1"),
    pytest.param(mp.mpf(25), id="c25"),
    pytest.param(mp.mpf(1) / 2, id="ising"),
    pytest.param(Fraction(-22, 5), id="lee_yang"),   # exact: -mp.mpf(22)/5 at import time is a double
    pytest.param(mp.mpf(28), id="c28"),
]


@pytest.mark.parametrize("c", _H_CROSS_CHECK_LOCI)
def test_H_q_series_regularized_matches_the_fixed_h_laurent_coefficients(c):
    """At every recognized exact singular locus and a weight off the Kac weights, the merged-pole H series equals
    the t^0 coefficients of the Laurent recursion at that h."""
    mp.mp.dps = 40
    h = mp.mpf("1.3")
    d = [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")]
    merged = H_q_series_regularized(h, d, c=c, q_order=6)
    laurent = H_q_series_laurent(h, d, c0=mp.convert(c), q_order=6)
    worst = max(abs(a - b) for a, b in zip(merged, laurent))
    assert worst < mp.mpf("1e-35")


def _just_off_c1(fn, *args):
    """The bare block at c = 1 + 1e-60 at 140 digits (analytic in c at a fixed weight: O(1e-60) from c = 1)."""
    with mp.workdps(140):
        return fn(*args, c=1 + mp.mpf(10) ** -60, q_order=6)


def test_F_block_regularized_matches_the_bare_block_just_off_c1():
    from virasoro_blocks.recursion import F_block

    mp.mp.dps = 40
    args = (mp.mpf("1.0"), mp.mpf("0.4"), [mp.mpf("0.1")] * 4)
    assert abs(F_block_regularized(*args, c=mp.mpf(1), q_order=6) - _just_off_c1(F_block, *args)) < mp.mpf("1e-38")


def test_B_block_regularized_matches_the_bare_block_just_off_c1():
    """Generic (unequal) externals."""
    from virasoro_blocks.composer import B_block

    mp.mp.dps = 40
    h_ext = [mp.mpf("0.20"), mp.mpf("0.30"), mp.mpf("0.25"), mp.mpf("0.35")]
    hbar_ext = [mp.mpf("0.18"), mp.mpf("0.28"), mp.mpf("0.27"), mp.mpf("0.33")]
    args = (mp.mpf("1.4"), mp.mpf("1.1"), mp.mpf("0.4"), mp.mpf("0.35"), h_ext, hbar_ext)
    ref = _just_off_c1(B_block, *args)
    assert abs(B_block_regularized(*args, c=mp.mpf(1), q_order=6) - ref) < mp.mpf("1e-38") * abs(ref)


@pytest.mark.parametrize("ell", [0, 1, 2])
def test_G_block_regularized_matches_the_bare_block_just_off_c1(ell):
    from virasoro_blocks.composer import G_block

    mp.mp.dps = 40
    args = (mp.mpf("3.1"), ell, mp.mpf("0.4"), mp.mpf("0.35"), [mp.mpf("0.15")] * 4)
    ref = _just_off_c1(G_block, *args)
    assert abs(G_block_regularized(*args, c=mp.mpf(1), q_order=6) - ref) < mp.mpf("1e-38") * abs(ref)


# --- Genuine singularity: the Laurent path raises with a structural reason ---


def test_laurent_rounding_residue_is_not_a_pole():
    """At c = 1, h = 0, h_ext = 49/36 the high-order coefficients cancel exactly; at 15 digits the rounding residue
    read as "H_32 keeps a t^-2 pole".  A negative power is now confirmed by one build at doubled precision before it
    is called a pole."""
    with mp.workdps(40):
        reference = H_q_series_laurent(0, [Fraction(49, 36)] * 4, c0=1, q_order=34)
    with mp.workdps(15):
        values = H_q_series_laurent(0, [Fraction(49, 36)] * 4, c0=1, q_order=34)
        assert all(abs(a - b) <= mp.mpf("1e-13") * max(1, abs(b)) for a, b in zip(values, reference))


def test_merged_context_probe_confirms_a_rounding_pole():
    """The merged-pole context probes the table at a generic weight for a structural pole; at 30 digits c0 = 1,
    h_ext = 121/64, q_order 46 read rounding residue as "H_46 keeps a t^-2 pole".  The probe now confirms it."""
    with mp.workdps(60):
        reference = H_q_series_regularized(0, [Fraction(121, 64)] * 4, c=1, q_order=46)
    with mp.workdps(30):
        values = H_q_series_regularized(0, [Fraction(121, 64)] * 4, c=1, q_order=46)
        assert all(abs(a - b) <= mp.mpf("1e-28") * max(1, abs(b)) for a, b in zip(values, reference))


def test_laurent_genuine_pole_survives_the_confirmation_build():
    with mp.workdps(15), pytest.raises(ValueError, match="genuine"):
        H_q_series_laurent(0, [Fraction(1, 5)] * 4, c0=Fraction(1, 2), q_order=12)


def test_laurent_detects_genuine_singularity_at_ising():
    """A channel with an h = 0 identity for generic externals at the Ising
    c = 1/2 is genuinely singular: h = 0 is a merged pole whose residue
    survives."""
    mp.mp.dps = 50
    with pytest.raises(ValueError, match="genuine"):
        F_block_regularized(
            mp.mpf(0),
            mp.mpf("0.2"),
            [mp.mpf("0.2")] * 4,
            c=mp.mpf(1) / 2,
            q_order=12,
            chiral_norm=4,
        )


# --- Gold standard: Laurent reproduces the exact Ising closed forms ---


def test_laurent_matches_ising_identity_block_closed_form():
    """F_block_regularized reproduces the exact identity-channel
    chiral block for the 2d Ising model (see tests/core/test_ising_blocks.py)."""
    mp.mp.dps = 50
    sigma = mp.mpf(1) / 16
    externals = [sigma] * 4
    z = mp.mpf("0.2")
    lib = F_block_regularized(
        mp.mpf(0), z, externals,
        c=mp.mpf(1) / 2, q_order=16,
    )
    # Convention (z_shift=0): F = z^h * N^{-h} * F_standard.
    # For h=0: F = z^0 * 1 * F_identity_closed(z) = F_identity_closed(z).
    root = mp.sqrt(1 - z)
    closed = (1 - z) ** (mp.mpf(-1) / 8) * mp.sqrt((1 + root) / 2)
    assert abs(mp.mpc(lib) - closed) < mp.mpf("1e-30")


def test_u_convolve_matches_naive_with_sparse_rows():
    """``u_convolve`` (the packed Kronecker flint multiply) is bit-identical to
    the naive per-u-power convolution, including when u-rows carry
    structurally-zero blocks (the skip-zero fast path) and negative-power
    t-blocks (the window), and when the inputs are freshly mpc-backed (the
    cap-clobber guard)."""
    from virasoro_blocks.tseries import u_convolve, T_HI

    mp.mp.dps = 50

    def ls(*pairs):
        s = LaurentSeries.zero()
        for power, coeff in pairs:
            s = s + LaurentSeries.monomial(power, mp.mpc(coeff))
        return s

    a_rows = [
        ls((0, "1.5"), (1, "-0.5"), (-1, "0.25")),
        LaurentSeries.zero(),                       # zero u^1 block (skip path)
        ls((2, "0.7"), (-2, "0.1")),
    ]
    b_rows = [
        ls((0, "2.0"), (-1, "0.3")),
        ls((1, "0.4"), (0, "-1.1")),
        LaurentSeries.zero(),                       # zero u^2 block (skip path)
    ]
    got = u_convolve(a_rows, b_rows)
    assert got is not None

    n = len(a_rows)
    for k in range(n):
        naive = LaurentSeries.zero()
        for j in range(k + 1):
            naive = naive + a_rows[j] * b_rows[k - j]
        for p in range(T_LO, T_HI + 1):
            assert abs(got[k].coeff(p) - naive.coeff(p)) < mp.mpf("1e-40")


def test_u_convolve_zero_operand_short_circuits_to_zero():
    """A structurally-zero operand makes the product zero without a multiply."""
    from virasoro_blocks.tseries import u_convolve

    mp.mp.dps = 50
    a = [LaurentSeries.zero(), LaurentSeries.zero()]
    b = [LaurentSeries.constant(1), LaurentSeries.monomial(1)]
    out = u_convolve(a, b)
    assert out is not None
    assert len(out) == 2
    assert all(s.is_zero() for s in out)
