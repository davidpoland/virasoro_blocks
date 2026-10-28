"""Discrete block orders are validated before caches or numerical construction."""

from fractions import Fraction as F

import mpmath as mp
import numpy as np
import pytest

from virasoro_blocks import global_blocks as gb
from virasoro_blocks.requests import NumericalPolicy, RequestValidationError


@pytest.mark.parametrize('order', [True, np.bool_(True), 1.5, F(3, 2), '3', mp.nan, mp.inf, 0, -1])
def test_q_order_validator_rejects_invalid_types_and_bounds(order):
    with pytest.raises(ValueError, match='q_order'):
        gb._validate_q_order(order)


# The full type matrix belongs to the validator. Callers cover truncation,
# bool-as-int, and the positive lower bound before numerical work.
@pytest.mark.parametrize('order', [True, 1.5, 0])
@pytest.mark.parametrize('chiral', [False, True])
def test_global_invalid_order_rejected_before_numerical_work(monkeypatch, order, chiral):
    def unexpected(*args, **kwargs):
        pytest.fail('invalid q_order reached numerical processing')
    monkeypatch.setattr(gb, 'global_block_exp_base', unexpected)
    with pytest.raises(ValueError, match='q_order'):
        if chiral:
            gb.global_chiral_F_interpolated_table('z', [.1]*4, 0, q_order=order, npoles=8)
        else:
            gb.global_scalar_G_interpolated_table('zzb', 0, [.1]*4, 0, q_order=order, npoles=8)


@pytest.mark.parametrize('order', [True, 1.5, 0])
@pytest.mark.parametrize('chiral', [False, True])
def test_pole_pool_rejects_invalid_order(order, chiral):
    with pytest.raises(ValueError, match='q_order'):
        if chiral:
            gb._global_chiral_q_order_pole_pool(order, delta12=.1, delta34=.2)
        else:
            gb._global_q_order_pole_pool(0, order, dh12=.1, dh34=.2, dhb12=.1, dhb34=.2)


@pytest.mark.parametrize('order', [3, 3.0, F(3), mp.mpf(3), np.int64(3)])
def test_integral_order_preserves_candidate_pool(order):
    assert gb._validate_q_order(order) == 3
    assert type(gb._validate_q_order(order)) is int
    assert gb._global_chiral_q_order_pole_pool(order, delta12=.1, delta34=.2) == [0, -.5, -1]


@pytest.mark.parametrize('name', ['order', 'lambda_order', 'npoles'])
def test_policy_rejects_numpy_boolean_integers(name):
    with pytest.raises(RequestValidationError, match=name):
        NumericalPolicy(**{name: np.bool_(True)})


@pytest.mark.parametrize('field', ['q_order', 'n_terms'])
def test_sphere_evaluator_rejects_invalid_controls_even_with_a_warm_cache(field):
    from virasoro_blocks.derivs import chiral_z_series_evaluator

    with mp.workdps(100):
        stored = mp.mpf(2)+mp.mpf('1e-70')
    with mp.workdps(30):
        reference = chiral_z_series_evaluator(['.1']*4, 2, c=32, q_order=2)
        for bad in (2.5, True, stored):
            options = dict(n_terms=2, q_order=2)
            options[field] = bad
            with pytest.raises(ValueError, match=field):
                chiral_z_series_evaluator(['.1']*4, c=32, **options)
        assert chiral_z_series_evaluator(['.1']*4, 2, c=32, q_order=2) is reference


def test_sphere_cutoff_rejection_precedes_parameter_and_laurent_work(monkeypatch):
    from virasoro_blocks import recursion, polyblock, regularization

    def unexpected(*args, **kwargs):
        pytest.fail('invalid cutoff reached numerical construction')
    monkeypatch.setattr(recursion, 'b_from_c', unexpected)
    monkeypatch.setattr(polyblock, 'b_from_c', unexpected)
    monkeypatch.setattr(regularization, '_resolve_c', unexpected)
    external = ['.1']*4
    paths = (
        lambda bad: recursion.H_q_series('.7', external, c=32, q_order=bad),
        lambda bad: recursion.F_block('.7', '.5', external, c=1, q_order=bad),
        lambda bad: polyblock.H_q_series_rational(external, c=32, q_order=bad),
        lambda bad: polyblock.chiral_denominator_roots(external, c=32, q_order=bad),
        lambda bad: recursion.H_q_series_laurent_precompute(external, c0=1, q_order=bad),
        lambda bad: regularization.H_q_series_regularized('.7', external, c=1, q_order=bad),
        lambda bad: regularization.F_block_regularized('.7', '.5', external, c=1, q_order=bad),
    )
    for path in paths:
        for bad in (True, 2.5):
            with pytest.raises(ValueError, match='q_order'):
                path(bad)


def test_sampler_discrete_controls_are_checked_before_coordinate_conversion():
    from virasoro_blocks.polyderivs import build_numeric_sampler
    from virasoro_blocks.regularization import _RegularizedNumericSampler

    for basis in ('scalar', 'chiral', 'ordered', 'projected'):
        for field in ('q_order', 'm_max', 'n_max'):
            options = dict(q_order=2, m_max=1, n_max=1)
            options[field] = 1.5
            with pytest.raises(ValueError, match=field):
                build_numeric_sampler(
                    'z' if basis == 'chiral' else 'zzb',
                    chiral=basis == 'chiral', spinning=basis == 'ordered',
                    projected=basis == 'projected', h_externals=['.1']*4,
                    hbar_externals=['.1']*4, ell=0, c=32, chiral_norm=None,
                    center='invalid coordinate', **options,
                )
            with pytest.raises(ValueError, match=field):
                _RegularizedNumericSampler(c=1, **options)


def test_cached_symbolic_table_rejects_fractional_derivative_controls():
    from virasoro_blocks.polyderivs import build_chiral_cache, build_table_from_cache

    with mp.workdps(30):
        chiral, _ = build_chiral_cache(['zzb'], ['.1']*4, lambda_=2, c=32, q_order=2)['zzb']
        for field in ('m_max', 'n_max', 'lambda_cut'):
            options = dict(m_max=2, n_max=2, lambda_cut=2)
            options[field] = 2.5
            with pytest.raises(ValueError, match=field):
                build_table_from_cache('zzb', chiral, None, 0, basis='scalar_G', **options)


def test_integral_sphere_and_derivative_controls_keep_the_same_values():
    from decimal import Decimal
    from virasoro_blocks.recursion import H_q_series
    from virasoro_blocks.derivs import chiral_z_series_evaluator, G_zzb_table, global_G_zzb_table
    from virasoro_blocks.polyderivs import G_zzb_poly_table, build_numeric_sampler
    from virasoro_blocks.polynomial import _arb_to_mpf

    with mp.workdps(30):
        external = ['.1']*4
        reference = H_q_series('.7', external, c=32, q_order=2)
        evaluate = chiral_z_series_evaluator(external, 2, c=32, q_order=2)
        for order in (2.0, F(2), Decimal(2), mp.mpf(2), np.int64(2)):
            assert H_q_series('.7', external, c=32, q_order=order) == reference
            assert chiral_z_series_evaluator(external, order, c=32, q_order=order) is evaluate
        sphere_grid = G_zzb_table('2.6', 0, external, 1.0, 1.0, c=32, q_order=2.0)
        assert sphere_grid == G_zzb_table('2.6', 0, external, 1, 1, c=32, q_order=2)
        actual = global_G_zzb_table('2.6', 0, external, 1.0, 1.0)
        assert actual == global_G_zzb_table('2.6', 0, external, 1, 1)
        table = G_zzb_poly_table(0, external, 1.0, 1.0, c=32, q_order=2.0)
        assert table.m_max == 1 and type(table.m_max) is int
        assert table.n_max == 1 and type(table.n_max) is int
        sampler = build_numeric_sampler(
            'zzb', chiral=False, spinning=False, projected=False,
            h_externals=external, hbar_externals=None, ell=0, m_max=1.0, n_max=1.0,
            c=32, q_order=2.0, chiral_norm=None,
        )
        assert type(sampler.q_order) is int
        assert type(sampler.m_max) is int and type(sampler.n_max) is int
        sampled, is_arb = sampler(['2.6'])
        assert len(sampled) == 1 and len(sampled[0]) == len(sphere_grid)
        for row, reference_row in zip(sampled[0], sphere_grid):
            assert len(row) == len(reference_row)
            for value, reference_value in zip(row, reference_row):
                value = _arb_to_mpf(value) if is_arb else value
                assert mp.almosteq(value, reference_value, rel_eps=mp.mpf('1e-27'), abs_eps=mp.mpf('1e-27'))


def test_regularization_precision_and_window_controls_are_discrete():
    from virasoro_blocks.recursion import H_q_series_laurent_precompute
    from virasoro_blocks.regularization import regularized_numeric_grids

    def unexpected(*args, **kwargs):
        pytest.fail('invalid control reached a probe sampler')

    with mp.workdps(30):
        for field, bad in (('work_dps', 100.5), ('half', 2.5), ('half', -1)):
            with pytest.raises(ValueError, match=field):
                H_q_series_laurent_precompute(['.0625']*4, c0=1, q_order=0, **{field: bad})
        for field, bad in (('work_dps', 100.5), ('max_work_dps', 200.5), ('n_points', 3.5)):
            with pytest.raises(ValueError, match=field):
                regularized_numeric_grids(unexpected, ['2.6'], c=1, **{field: bad})


def test_irreducible_torus_order_is_not_truncated():
    from virasoro_blocks.torus_onept import irreducible_onept_F_series

    for bad in (1.5, True, -1):
        with pytest.raises(ValueError, match='n_max'):
            irreducible_onept_F_series(F(1, 4), 1, 0, bad, exact=True)
    assert irreducible_onept_F_series(F(1, 4), 1, 0, 1.0, exact=True) == \
        irreducible_onept_F_series(F(1, 4), 1, 0, 1, exact=True)


def test_laurent_layout_controls_cannot_silently_change_the_window():
    from virasoro_blocks import tseries

    original = tseries.window_half()
    try:
        for bad in (1.5, True):
            with pytest.raises(ValueError, match='half-width'):
                tseries.set_window(bad)
            assert tseries.window_half() == original
            with pytest.raises(ValueError, match='n_terms'):
                tseries.StackedURows([[tseries.LaurentSeries.constant(1)]], bad)
    finally:
        tseries.set_window(original)
