"""Global derivative orders retain integer meaning at direct API boundaries."""

from fractions import Fraction as F

import mpmath as mp
import numpy as np
import pytest

from virasoro_blocks import global_blocks as gb


@pytest.mark.parametrize('value', [1.5, -1])
@pytest.mark.parametrize('name,chiral', [('m_max', False), ('n_max', False),
    ('selection_m_max', False), ('selection_n_max', False),
    ('m_max', True), ('selection_m_max', True)])
def test_table_rejects_invalid_derivative_order_early(monkeypatch, name, chiral, value):
    def unexpected(*args, **kwargs):
        pytest.fail('invalid derivative order reached numerical work')
    monkeypatch.setattr(gb, 'global_block_exp_base', unexpected)
    kwargs = {name: value}
    kwargs.setdefault('m_max', 0)
    kwargs['selection_coordinate'] = 'z' if chiral else 'zzb'
    kwargs['npoles'] = 8
    with pytest.raises(ValueError, match=name):
        if chiral:
            gb.global_chiral_F_interpolated_table('z', [.1]*4, **kwargs)
        else:
            gb.global_scalar_G_interpolated_table('zzb', 0, [.1]*4, **kwargs)


@pytest.mark.parametrize('name', ['m_max', 'n_max'])
@pytest.mark.parametrize('value', [1.5, -1])
def test_direct_evaluation_rejects_invalid_derivative_order(monkeypatch, name, value):
    def unexpected(*args, **kwargs):
        pytest.fail('invalid derivative order reached evaluation')
    monkeypatch.setattr(gb, '_global_chiral_series_evaluator', unexpected)
    with pytest.raises(ValueError, match=name):
        gb.global_evaluate_block(.3, .4, 1, 2, [.1]*4, **{name: value})


@pytest.mark.parametrize('value', [1, 1.0, F(1), mp.mpf(1), np.int64(1)])
def test_integral_orders_produce_requested_grid(value):
    grid = gb.global_evaluate_block(.3, .4, 1, 2, [.1]*4, m_max=value, n_max=value)
    assert len(grid) == 2 and all(len(row) == 2 for row in grid)
    assert mp.almosteq(grid[0][0], gb.global_evaluate_block(.3, .4, 1, 2, [.1]*4))


@pytest.mark.parametrize('chiral', [False, True])
@pytest.mark.parametrize('value', [1.5, -1])
def test_sampler_rejects_invalid_order(chiral, value):
    with pytest.raises(ValueError, match='m_max'):
        gb.build_global_sampler('z' if chiral else 'zzb',
            basis='global_chiral_F' if chiral else 'global_scalar_G',
            ell=0, h_externals=[.1]*4, hbar_externals=[.1]*4,
            m_max=value, n_max=0)


@pytest.mark.parametrize('chiral', [False, True])
def test_sampler_normalizes_integral_orders(chiral):
    sampler = gb.build_global_sampler('z' if chiral else 'zzb',
        basis='global_chiral_F' if chiral else 'global_scalar_G',
        ell=0, h_externals=[.1]*4, hbar_externals=[.1]*4,
        m_max=F(1), n_max=0)
    grid = sampler(2)
    assert len(grid) == 2
    assert all(len(row) == 1 for row in grid)
