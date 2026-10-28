"""Theta tails and coordinate jets are resolved at their actual centers."""
from math import comb

import mpmath as mp
import pytest
from flint import acb_series, ctx

from virasoro_blocks.coordinates import z_of_q
from virasoro_blocks.derivs import _global_chiral_series_evaluator, _global_chiral_taylor_flint
from virasoro_blocks.global_blocks import global_evaluate_block
from virasoro_blocks.polynomial import _mpf_to_acb
from virasoro_blocks.polyderivs import _flint_taylor_series
from virasoro_blocks.theta_series import theta_series


def _complement(q):
    # S-transform computes 1-lambda(q) without subtracting from 1.
    dual = mp.exp(mp.pi**2/mp.log(q))
    return (mp.jtheta(2, 0, dual)/mp.jtheta(3, 0, dual))**4


@pytest.mark.parametrize('q', ['.043', '.5', '.7', '.99', '-.7', '.7+.01j'])
@pytest.mark.parametrize('count', [1, 4, 20])
def test_theta_balls_contain_independent_high_precision_coefficients(q, count):
    old_cap, old_prec = ctx.cap, ctx.prec
    try:
        ctx.cap, ctx.prec = count, 180
        with mp.workdps(50):
            center = mp.mpmathify(q)
            rows = theta_series(acb_series([_mpf_to_acb(center), 1]), full=True)
        with mp.workdps(220):
            # Sum far beyond the production tail cutoff. These closed Taylor
            # coefficients check the independently bounded production tail.
            # Reuse q**exponent across derivative orders and across theta3/4.
            # Repeating hundreds of large complex powers for every coefficient
            # made the reference calculation dominate this module's runtime.
            powers = {
                a: [(n, n*(n+a), center**(n*(n+a))) for n in range(1, 350)]
                for a in (0, 1)
            }
            inverse_powers = [center**(-k) for k in range(count)]
            for i, (a, row) in enumerate(zip((1, 0, 0), rows)):
                values = row.coeffs()
                for k in range(count):
                    reference = mp.mpf(k == 0)
                    for n, exponent, power in powers[a]:
                        if exponent >= k:
                            factor = 1 if a else 2*(-1)**n if i == 2 else 2
                            reference += factor*comb(exponent, k)*power*inverse_powers[k]
                    assert values[k].contains(_mpf_to_acb(reference))
    finally:
        ctx.cap, ctx.prec = old_cap, old_prec


def test_theta_at_zero_has_exact_integer_taylor_coefficients():
    old_cap, old_prec = ctx.cap, ctx.prec
    try:
        ctx.cap, ctx.prec = 20, 100
        theta2, theta3, theta4 = theta_series(acb_series([0, 1]), full=True)
        assert theta2.coeffs()[12] == 1
        assert theta3.coeffs()[16] == 2
        assert theta4.coeffs()[9] == -2
        assert all(x.imag == 0 for row in (theta2, theta3, theta4) for x in row.coeffs())
    finally:
        ctx.cap, ctx.prec = old_cap, old_prec


def test_theta_high_derivative_is_not_cut_off_by_a_fixed_theta_count():
    old_cap, old_prec = ctx.cap, ctx.prec
    try:
        ctx.cap, ctx.prec = 100, 180
        with mp.workdps(50):
            center = mp.mpf('.043')
            row = theta_series(acb_series([_mpf_to_acb(center), 1]))[1]
        with mp.workdps(220):
            k = 99
            reference = sum(2*comb(n*n, k)*center**(n*n-k) for n in range(10, 35))
            assert reference != 0
            assert row.coeffs()[k].contains(_mpf_to_acb(reference))
    finally:
        ctx.cap, ctx.prec = old_cap, old_prec


def test_theta_tail_bounds_survive_a_nonlinear_nome_composition():
    old_cap, old_prec = ctx.cap, ctx.prec
    try:
        ctx.cap, ctx.prec = 5, 200
        with mp.workdps(50):
            labels = tuple(mp.mpf(s) for s in ('.7', '.4', '.1'))
            row = theta_series(acb_series([_mpf_to_acb(v) for v in labels]))[1].coeffs()
        with mp.workdps(150):
            a, b, c = labels
            expected = mp.taylor(lambda u: mp.jtheta(3, 0, a+b*u+c*u*u), 0, 4)
            assert all(got.contains(_mpf_to_acb(wanted)) for got, wanted in zip(row, expected))
    finally:
        ctx.cap, ctx.prec = old_cap, old_prec


@pytest.mark.parametrize('q', ['.5', '.7', '-.7', '.7+.01j'])
@pytest.mark.parametrize('method', ['direct', 'closure', 'batch', 'grid'])
def test_global_nome_jets_match_an_analytic_block(q, method):
    with mp.workdps(100):
        center = mp.mpmathify(q)
        f = lambda x: -mp.log(_complement(x))/16
        expected = mp.taylor(f, center, 2)
    before = mp.mp.dps
    with mp.workdps(40):
        if method == 'direct':
            actual = _global_chiral_taylor_flint(
                1, (0,)*4, 3, z_of_u=z_of_q, u_symmetric=q,
                chiral_norm=None, z_shift=None,
            )
        elif method in ('closure', 'batch'):
            evaluate = _global_chiral_series_evaluator(
                (0,)*4, 3, z_of_u=z_of_q, u_symmetric=q,
                chiral_norm=None, z_shift=None,
            )
            if method == 'batch':
                rows = evaluate.many([1, 1])
                if mp.im(center) != 0:
                    assert rows is None  # The documented complex fallback.
                    actual = evaluate(1)
                else:
                    assert rows is not None
                    actual = [mp.mpf(x.mid().str(60, radius=False)) for x in rows[0]]
            else:
                actual = evaluate(1)
        else:
            z = z_of_q(q)
            grid = global_evaluate_block(z, z, 1, 1, (0,)*4, coord='qqb', m_max=2)
            actual = [row[0]/expected[0] for row in grid]
        with mp.workdps(80):
            for got, wanted in zip(actual, expected):
                # 1 - z(q) is ~1e-11 at q = .7: the jets keep the digits it leaves.
                assert abs(got-wanted) < mp.mpf('1e-28')*max(1, abs(wanted))
        assert mp.mp.dps == 40
    assert mp.mp.dps == before


