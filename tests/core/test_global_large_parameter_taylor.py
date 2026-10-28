"""Large hypergeometric parameters must not invalidate convergent global jets."""
from fractions import Fraction as F
import pickle

import mpmath as mp
import pytest
from flint import acb, acb_series, ctx

from virasoro_blocks import derivs as d, global_blocks as gb
from virasoro_blocks.coordinates import COORDINATES
from virasoro_blocks.polynomial import _arb_to_mpf, _acb_to_mpc

EXT = (F(1, 10),)*4


def _hyp_coefficient(a, b, c, z, k, regularized=False):
    denominator = mp.gamma(c+k) if regularized else mp.rf(c, k)
    return (mp.rf(a, k)*mp.rf(b, k)/denominator/mp.factorial(k)
            * mp.hyp2f1(a+k, b+k, c+k, z))


def _half_coefficients(h, z, ext=EXT, count=2):
    h = mp.convert(h)
    ext = tuple(mp.convert(value) for value in ext)
    a, b, c = h+ext[1]-ext[0], h+ext[2]-ext[3], 2*h
    return [sum(mp.binomial(h, j)*z**(h-j)/16**h
                * _hyp_coefficient(a, b, c, z, k-j) for j in range(k+1))
            for k in range(count)]


def _jacobian(coordinate, z):
    if coordinate == 'zzb':
        return mp.mpf(1)
    if coordinate == 'yyb':
        y = (2*z-1)/(1+2*mp.sqrt(z*(1-z)))
        return (1-y*y)/(1+y*y)**2
    q = mp.exp(-mp.pi*mp.ellipk(1-z)/mp.ellipk(z))
    return mp.diff(lambda u: (mp.jtheta(2, 0, u)/mp.jtheta(3, 0, u))**4, q)


@pytest.mark.parametrize('regularized', [False, True])
@pytest.mark.parametrize('count', [1, 4])
def test_large_parameter_hypergeometric_and_regularized_jets(regularized, count):
    with mp.workdps(200):
        z = mp.mpf('.3')
        expected = [_hyp_coefficient(1000, 1000, 2000, z, k, regularized) for k in range(count)]
    old_prec, old_cap = ctx.prec, ctx.cap
    try:
        with mp.workdps(60):
            ctx.prec, ctx.cap = 220, count
            result = d._hyp2f1_series(acb(1000), acb(1000), acb(2000),
                                     acb_series([acb('0.3'), 1]), regularized=regularized)
            for actual, reference in zip(result.coeffs(), expected):
                assert actual.is_finite()
                assert abs(_acb_to_mpc(actual)-reference) < abs(reference)*mp.mpf('1e-57')
            assert len(result.coeffs()) == count
    finally:
        ctx.prec, ctx.cap = old_prec, old_cap


def test_ode_fallback_composes_with_nonlinear_coordinate_series():
    with mp.workdps(200):
        z = mp.mpf('.3')
        t = [_hyp_coefficient(1000, 1000, 2000, z, k) for k in range(4)]
        # z(u) = .3 + u + .2 u², composed independently through cubic order.
        expected = [t[0], t[1], t[2]+t[1]/5, t[3]+2*t[2]/5]
    old_prec, old_cap = ctx.prec, ctx.cap
    try:
        with mp.workdps(60):
            ctx.prec, ctx.cap = 220, 4
            result = d._hyp2f1_series(acb(1000), acb(1000), acb(2000),
                                     acb_series([acb('.3'), 1, acb('.2')]))
            for actual, reference in zip(result.coeffs(), expected):
                assert abs(_acb_to_mpc(actual)-reference) < abs(reference)*mp.mpf('1e-57')
    finally:
        ctx.prec, ctx.cap = old_prec, old_cap


@pytest.mark.parametrize('coordinate', ['z', 'y', 'q'])
def test_public_chiral_taylor_builders_keep_large_exchanges_finite(coordinate):
    with mp.workdps(200):
        expected = _half_coefficients(1000, mp.mpf('.5'))
        expected[1] *= _jacobian({'z': 'zzb', 'y': 'yyb', 'q': 'qqb'}[coordinate], mp.mpf('.5'))
    with mp.workdps(60):
        old_cap = ctx.cap
        values = getattr(d, f'global_chiral_{coordinate}_taylor')(1000, EXT, 2)
        for actual, reference in zip(values, expected):
            assert mp.isfinite(actual)
            assert abs(actual-reference) < abs(reference)*mp.mpf('1e-55')
        assert ctx.cap == old_cap and mp.mp.dps == 60


@pytest.mark.parametrize('coordinate', ['zzb', 'yyb', 'qqb'])
@pytest.mark.parametrize('basis', ['global_ordered_B', 'global_scalar_G'])
def test_off_diagonal_large_exchange_sampling_and_batches_match_independent_products(coordinate, basis):
    with mp.workdps(200):
        left, right = mp.mpf('.3'), mp.mpf('.7')
        hp, hm = F(2001, 2), F(1999, 2)
        lp, lm = _half_coefficients(hp, left), _half_coefficients(hm, left)
        rp, rm = _half_coefficients(hp, right), _half_coefficients(hm, right)
        jl, jr = _jacobian(coordinate, left), _jacobian(coordinate, right)
        # the odd-spin scalar G is z <-> zbar even: B(h, hbar) + B(hbar, h)
        expected = [[(lp[m]*rm[n]+(lm[m]*rp[n] if basis == 'global_scalar_G' else 0))
                     *jl**m*jr**n for n in range(2)] for m in range(2)]
    with mp.workdps(60):
        sampler = gb.build_global_sampler(coordinate, basis=basis, ell=1,
            h_externals=EXT, hbar_externals=EXT, m_max=1, n_max=1,
            center=F(3, 10), center_bar=F(7, 10))
        sampler = pickle.loads(pickle.dumps(sampler))
        scalar = sampler(2000)
        rows, is_arb = sampler.sample_bucket(iter([2000]), iter([(0, 0), (1, 1)]))
        assert is_arb
        for m, n in [(0, 0), (1, 1)]:
            for actual in (scalar[m][n], _arb_to_mpf(rows[0][m][n])):
                assert mp.isfinite(actual)
                assert abs(actual-expected[m][n]) < abs(expected[m][n])*mp.mpf('1e-55')


@pytest.mark.parametrize('coordinate', ['zzb', 'yyb', 'qqb'])
def test_public_two_dimensional_tables_preserve_odd_spin_symmetry_at_large_exchange(coordinate):
    with mp.workdps(200):
        z = mp.mpf('.5')
        hp = _half_coefficients(F(2001, 2), z)
        hm = _half_coefficients(F(1999, 2), z)
        jacobian = _jacobian(coordinate, z)
        expected = (hp[1]*hm[0]-hm[1]*hp[0])*jacobian
        symmetric = [[(hp[m]*hm[n]+hm[m]*hp[n])*jacobian**(m+n) for n in range(2)] for m in range(2)]
    with mp.workdps(60):
        # the odd-spin scalar G is z <-> zbar even
        scalar = getattr(d, f'global_G_{coordinate}_table')(2000, 1, EXT, 1, 1)
        for m in range(2):
            for n in range(2):
                assert abs(scalar[m][n]-symmetric[m][n]) < abs(symmetric[m][n])*mp.mpf('1e-55')
        # the z <-> zbar-odd structure is the parity '-' projection
        table = getattr(d, f'global_parity_projected_B_{coordinate}_table')(2000, 1, EXT, EXT, 1, 1, parity='-')
        assert table[0][0] == table[1][1] == 0
        # Tables retain guard digits; unary negation would round one entry
        # to the lower ambient precision before comparing its raw value.
        assert mp.fadd(table[1][0], table[0][1], exact=True) == 0
        assert mp.isfinite(table[1][0]) and expected != 0
        assert abs(table[1][0]-expected) < abs(expected)*mp.mpf('1e-55')


@pytest.mark.parametrize('center', [F(3, 10), F(1, 2), F(95, 100)])
def test_large_complex_parameter_series_uses_the_scalar_sampling_fallback(center):
    ext = (F(1, 10), complex(.2, .1), F(3, 10), F(2, 5))
    with mp.workdps(200):
        reference = _half_coefficients(1000, mp.convert(center), ext)
    with mp.workdps(60):
        sampler = gb.build_global_sampler('z', basis='global_chiral_F', ell=0,
            h_externals=ext, hbar_externals=ext, m_max=1, n_max=0, center=center)
        actual = sampler(1000)
        for k in range(2):
            assert mp.isfinite(actual[k][0])
            assert abs(actual[k][0]-reference[k]) < abs(reference[k])*mp.mpf('1e-55')


def test_large_parameter_regularized_jet_at_a_denominator_parameter_pole():
    def regularized(a, b, p, z):
        return (mp.rf(a, p+1)*mp.rf(b, p+1)*z**(p+1)/mp.factorial(p+1)
                * mp.hyp2f1(a+p+1, b+p+1, p+2, z))
    with mp.workdps(200):
        a, b, z = mp.mpf(1000), mp.mpc('1001', '.1'), mp.mpf('.3')
        expected = [regularized(a, b, 2, z), a*b*regularized(a+1, b+1, 1, z)]
    old_prec, old_cap = ctx.prec, ctx.cap
    try:
        with mp.workdps(60):
            ctx.prec, ctx.cap = 220, 2
            result = d._hyp2f1_series(acb(1000), acb(1001, '.1'), acb(-2),
                                     acb_series([acb('.3'), 1]), regularized=True)
            for actual, reference in zip(result.coeffs(), expected):
                assert actual.is_finite()
                assert abs(_acb_to_mpc(actual)-reference) < abs(reference)*mp.mpf('1e-55')
            assert ctx.prec == 220 and ctx.cap == 2
    finally:
        ctx.prec, ctx.cap = old_prec, old_cap


def test_zero_center_and_terminating_polynomials_keep_the_direct_series_rules():
    old_prec, old_cap = ctx.prec, ctx.cap
    try:
        with mp.workdps(60):
            ctx.prec, ctx.cap = 220, 4
            # Taylor at the origin has explicit Pochhammer coefficients;
            # no division by the ODE's zero leading coefficient is allowed.
            values = d._hyp2f1_series(acb(1000), acb(1000), acb(2000), acb_series([0, 1]))
            for k, actual in enumerate(values.coeffs()):
                expected = mp.rf(1000, k)**2/mp.rf(2000, k)/mp.factorial(k)
                assert abs(_acb_to_mpc(actual)-expected) < abs(expected)*mp.mpf('1e-57')
            regularized = d._hyp2f1_series(acb(1000), acb(1000), acb(-2),
                                          acb_series([0, 1]), regularized=True)
            assert all(coefficient.is_zero() for coefficient in regularized.coeffs()[:3])
            assert regularized.coeffs()[3].is_finite()
            # A terminating polynomial is valid beyond the unit disk.
            values = d._hyp2f1_series(acb(-2), acb(3), acb(4), acb_series([2, 1]))
            expected = [mp.mpf(2)/5, mp.mpf(9)/10, mp.mpf(3)/5]
            for actual, reference in zip(values.coeffs(), expected):
                assert abs(_acb_to_mpc(actual)-reference) < mp.mpf('1e-57')
    finally:
        ctx.prec, ctx.cap = old_prec, old_cap


def test_accuracy_context_and_flint_precision_are_restored_after_a_pole_error():
    from virasoro_blocks.requests import BlockPoleError

    ext = (F(1, 10), F(1, 5), F(3, 10), F(2, 5))
    token = d._GLOBAL_HYP_ACCURACY_BITS.set(71)
    old_prec, old_cap = ctx.prec, ctx.cap
    try:
        with mp.workdps(40):
            ev = d._global_chiral_series_evaluator(ext, 2, z_of_u=lambda u: u,
                u_symmetric=F(95, 100), chiral_norm=None, z_shift=None)
            with pytest.raises(BlockPoleError):
                ev(0)
            assert d._GLOBAL_HYP_ACCURACY_BITS.get() == 71
            assert ctx.cap == old_cap
            assert mp.mp.dps == 40
    finally:
        d._GLOBAL_HYP_ACCURACY_BITS.reset(token)
        ctx.prec, ctx.cap = old_prec, old_cap


def test_seed_ladder_can_extend_an_already_large_input_guard():
    old_prec, old_cap = ctx.prec, ctx.cap
    try:
        with mp.workdps(60):
            # The seed guard limit must start from the existing precision,
            # rather than clipping a larger input guard to the caller's dps.
            ctx.prec, ctx.cap = 5000, 1
            result = d._hyp2f1_series(acb('4000.1', '.1').mid(), acb('3999.9').mid(),
                                     acb(8000), acb_series([acb('.95').mid(), 1]))
            assert result.coeffs()[0].is_finite()
            assert result.coeffs()[0].rel_accuracy_bits() >= mp.mp.prec
            assert ctx.prec == 5000 and ctx.cap == 1
    finally:
        ctx.prec, ctx.cap = old_prec, old_cap


def test_unresolvable_seed_reports_requested_accuracy_and_restores_precision():
    old_prec, old_cap = ctx.prec, ctx.cap
    with mp.workdps(40):
        token = d._GLOBAL_HYP_ACCURACY_BITS.set(mp.mp.prec)
    try:
        with mp.workdps(125):
            ctx.prec, ctx.cap = 600, 2
            with pytest.raises(ValueError, match='40 digits'):
                # Unregularized 2F1 has a genuine c=0 parameter pole.
                d._hyp2f1_series(acb(1000), acb(1000), acb(0), acb_series([acb('.95'), 1]))
            assert ctx.prec == 600 and ctx.cap == 2
            assert mp.mp.dps == 125
    finally:
        d._GLOBAL_HYP_ACCURACY_BITS.reset(token)
        ctx.prec, ctx.cap = old_prec, old_cap


