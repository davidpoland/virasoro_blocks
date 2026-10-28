"""Global kernels retain source offsets before forming affine parameters."""
from fractions import Fraction as F
import pickle

import mpmath as mp
import pytest
from flint import ctx

from virasoro_blocks import global_blocks as gb
from virasoro_blocks.derivs import (
    _global_chiral_block, _global_chiral_series_evaluator,
    global_chiral_z_taylor,
)
from virasoro_blocks.polynomial import _arb_to_mpf
from virasoro_blocks.requests import BlockPoleError


EXT = (F(3, 10), F(4, 10), F(5, 10), F(6, 10))


def _reference(h, z, ext=EXT):
    h, z = mp.convert(h), mp.convert(z)
    ext = [mp.convert(x) for x in ext]
    return 16**(-h) * z**h * mp.hyp2f1(
        h + ext[1]-ext[0], h + ext[2]-ext[3], 2*h, z)


def _reference_value_and_derivative(h, z, ext=EXT):
    # Independent scalar 2F1 identities avoid mp.taylor's high-precision
    # finite differences near poles. Keep the same oracle precision and
    # accuracy checks; no production jet/ODE implementation is involved.
    h, z = mp.convert(h), mp.convert(z)
    ext = [mp.convert(x) for x in ext]
    a, b = h+ext[1]-ext[0], h+ext[2]-ext[3]
    hyper = mp.hyp2f1(a, b, 2*h, z)
    derivative = a*b/(2*h)*mp.hyp2f1(a+1, b+1, 2*h+1, z)
    prefactor = 16**(-h)*z**h
    return [prefactor*hyper, prefactor*(h*hyper/z+derivative)]


def _evaluator(ext=EXT, count=2):
    return _global_chiral_series_evaluator(
        ext, count, z_of_u=lambda u: u, u_symmetric=F(1, 2),
        chiral_norm=None, z_shift=None)


@pytest.mark.parametrize('h', [0, F(-1, 2), -1, -100])
@pytest.mark.parametrize('path', ['point', 'taylor', 'closure', 'batch', 'sampler'])
def test_genuine_poles_raise_in_every_numeric_global_path(h, path):
    with mp.workdps(40), pytest.raises(BlockPoleError):
        if path == 'point':
            _global_chiral_block(h, F(1, 2), EXT)
        elif path == 'taylor':
            global_chiral_z_taylor(h, EXT, 2)
        elif path == 'closure':
            _evaluator()(h)
        elif path == 'batch':
            _evaluator().many([F(7, 10), h])
        else:
            gb.build_global_sampler('z', basis='global_chiral_F', ell=0,
                h_externals=EXT, hbar_externals=EXT, m_max=1, n_max=0)(h)


@pytest.mark.parametrize('path', ['point', 'taylor', 'closure'])
def test_subprecision_external_offsets_do_not_turn_a_genuine_pole_into_a_limit(path):
    eps = F(1, 10**70)
    ext = (F(3, 10), F(3, 10)+eps, F(1, 2)+eps, F(1, 2))
    with mp.workdps(40), pytest.raises(BlockPoleError):
        if path == 'point':
            _global_chiral_block(0, F(1, 2), ext)
        elif path == 'taylor':
            global_chiral_z_taylor(0, ext, 2)
        else:
            _evaluator(ext)(0)


@pytest.mark.parametrize('direction', ['real', 'imag'])
def test_laurent_data_requires_the_actual_pole_not_a_nearby_label(direction):
    with mp.workdps(160):
        shifted = F(-1)+F(1, 10**70) if direction == 'real' else mp.mpc(-1, '1e-70')
    with mp.workdps(40):
        assert _evaluator().pole_data(shifted) is None


def test_native_float_inputs_have_decimal_intent_in_point_and_taylor_paths():
    with mp.workdps(40):
        point = _global_chiral_block(.7, .3, [.1]*4)
        reference = _global_chiral_block(F(7, 10), F(3, 10), [F(1, 10)]*4)
        assert point == reference
        assert global_chiral_z_taylor(.7, [.1]*4, 2) == global_chiral_z_taylor(F(7, 10), [F(1, 10)]*4, 2)
        shifted = _global_chiral_block(.7000000000000001, .3, [.1]*4)
        assert abs(shifted-point) > mp.mpf('1e-18')


@pytest.mark.parametrize('chiral', [False, True])
def test_pickled_samplers_keep_exact_external_labels_and_removable_limits(chiral):
    ext = (.3, 1.3, 1.7, .7)
    with mp.workdps(15):
        sampler = gb.build_global_sampler('z' if chiral else 'zzb',
            basis='global_chiral_F' if chiral else 'global_ordered_B', ell=0,
            h_externals=ext, hbar_externals=ext, m_max=0, n_max=0)
        sampler = pickle.loads(pickle.dumps(sampler))
    with mp.workdps(40):
        assert sampler.h_ext == [F(3, 10), F(13, 10), F(17, 10), F(7, 10)]
        assert sampler(-1 if chiral else -2)[0][0] == (32 if chiral else 1024)


@pytest.mark.parametrize('chiral', [False, True])
def test_table_and_selection_samplers_retain_exact_source_externals(monkeypatch, chiral):
    ext = (F(3, 10), F(13, 10)+F(1, 10**70), F(17, 10), F(7, 10))
    monkeypatch.setattr(gb, '_interpolate_global_table', lambda **kw: kw)
    with mp.workdps(40):
        if chiral:
            table = gb.global_chiral_F_interpolated_table('z', ext, 0, npoles=0,
                selection_coordinate='y')
        else:
            table = gb.global_scalar_G_interpolated_table('zzb', 0, ext, 0, npoles=0,
                selection_coordinate='yyb')
        for key in ('sampler', 'selection_sampler'):
            assert table[key].h_ext == list(ext)
            assert table[key].h_ext[1]-table[key].h_ext[0] == 1+F(1, 10**70)


def test_diagonal_reuse_compares_source_labels_instead_of_rounded_values():
    with mp.workdps(15):
        binary_external = mp.mpf('.1')
    assert not gb._global_label_equal(.1, binary_external)
    with mp.workdps(100):
        expected = _reference(F(7, 10), F(3, 10), [F(1, 10)]*4) * _reference(
            F(7, 10), F(3, 10), [binary_external, F(1, 10), F(1, 10), F(1, 10)])
    with mp.workdps(40):
        actual = gb.global_evaluate_block(F(3, 10), F(3, 10), F(7, 10), F(7, 10), [.1]*4,
            hbar_externals=[binary_external, .1, .1, .1], coord='zzb', m_max=0, n_max=0)
        assert abs(actual-expected) < abs(expected)*mp.mpf('1e-37')


def test_complex_externals_work_in_sampler_and_real_table_builder():
    with mp.workdps(100):
        ext = (mp.mpc('.3', '.01'), F(4, 10), F(5, 10), F(6, 10))
        expected = _reference(F(7, 10), F(1, 2), ext)
    with mp.workdps(40):
        sampler = gb.build_global_sampler('z', basis='global_chiral_F', ell=0,
            h_externals=ext, hbar_externals=ext, m_max=0, n_max=0)
        assert abs(sampler(F(7, 10))[0][0]-expected) < abs(expected)*mp.mpf('1e-37')
        table = gb.global_chiral_F_interpolated_table('z', ext, 0, npoles=2, q_order=6)
        assert table is not None


@pytest.mark.parametrize('count', [0, -1])
def test_empty_global_series_requests_raise_clearly(count):
    with pytest.raises(ValueError, match='n_terms must be >= 1'):
        _evaluator(count=count)
