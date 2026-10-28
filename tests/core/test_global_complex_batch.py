"""Real batch optimizations must never discard a complex block component."""

import mpmath as mp
import pytest
from virasoro_blocks.derivs import _global_chiral_series_evaluator
from virasoro_blocks.polynomial import _arb_to_mpf


def evaluator(externals=None, center=.5):
    return _global_chiral_series_evaluator(externals or [.1]*4, 3,
        z_of_u=lambda u: u, u_symmetric=center, chiral_norm=1, z_shift=0)


@pytest.mark.parametrize('case', ['weight', 'external', 'center', 'branch_cut'])
def test_complex_result_declines_real_projection(case):
    with mp.workdps(40):
        h = .8 + .3j if case == 'weight' else mp.mpf('.8')
        ext = [.1, .2 + .3j, .1, .1] if case == 'external' else None
        center = .5 + .2j if case == 'center' else 2 if case == 'branch_cut' else .5
        ev = evaluator(ext, center)
        result = ev(h)
        assert any(abs(mp.im(v)) > mp.mpf('1e-10') for v in result)
        assert ev(h, return_arb=True) is None
        assert ev.many([h]) is None


def test_real_batch_materializes_iterator_once():
    ev = evaluator()
    points = [mp.mpf('.8'), mp.mpf('1.2')]
    actual = ev.many(iter(points))
    assert len(actual) == len(points)
    for h, row in zip(points, actual):
        assert all(mp.almosteq(_arb_to_mpf(a), b) for a, b in zip(row, ev(h)))


def test_zero_imaginary_native_complex_keeps_real_fast_path():
    ev = evaluator([complex(.1, 0)]*4)
    assert ev.many([complex(.8, 0)]) is not None
    assert ev(complex(.8, 0), return_arb=True) is not None


def test_offreal_global_sampler_batch_preserves_complex_grid():
    from virasoro_blocks.global_blocks import build_global_sampler
    with mp.workdps(40):
        sampler = build_global_sampler('zzb', basis='global_ordered_B', ell=0,
            h_externals=[.1]*4, hbar_externals=[.1]*4, m_max=1, n_max=1,
            center=mp.mpc('.4', '.1'), center_bar=mp.mpf('.3'))
        reference = sampler(mp.mpf('1.6'))
        actual, is_arb = sampler.sample_bucket([mp.mpf('1.6')], [(0, 0), (1, 1)])
        assert not is_arb
        assert abs(mp.im(reference[0][0])) > mp.mpf('1e-10')
        for m, n in [(0, 0), (1, 1)]:
            assert mp.almosteq(actual[0][m][n], reference[m][n])
