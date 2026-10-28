"""Distinct centers retain their difference through coordinate maps and assembly."""
from fractions import Fraction as F
from functools import lru_cache
import pickle

import mpmath as mp
import pytest

from virasoro_blocks import global_blocks as gb, polyderivs as pd
from virasoro_blocks.numeric_input import literal_components, literal_number
from virasoro_blocks.polynomial import _arb_to_mpf
from virasoro_blocks.recursion import F_block
from virasoro_blocks.regularization import H_q_series_regularized

EXT = (F(1, 10),)*4


def _mp_label(value):
    real, imag = literal_components(value)
    return mp.convert(real) if imag == 0 else mp.mpc(mp.convert(real), mp.convert(imag))


def _global_half(h, z, derivative=0):
    value = z**h*mp.hyp2f1(h, h, 2*h, z)/16**h
    if not derivative:
        return value
    return (h*z**(h-1)*mp.hyp2f1(h, h, 2*h, z)
            + z**h*h/2*mp.hyp2f1(h+1, h+1, 2*h+1, z))/16**h


def _coordinate_jacobian(coordinate, z):
    if coordinate == 'zzb':
        return mp.mpf(1)
    if coordinate == 'yyb':
        y = (2*z-1)/(1+2*mp.sqrt(z*(1-z)))
        return (1-y*y)/(1+y*y)**2
    q = mp.exp(-mp.pi*mp.ellipk(1-z)/mp.ellipk(z))
    return mp.diff(lambda q: (mp.jtheta(2, 0, q)/mp.jtheta(3, 0, q))**4, q)


def _global_reference(left, right, coordinate, delta=3, derivative=0):
    with mp.workdps(450):
        z, zb, delta = _mp_label(left), _mp_label(right), _mp_label(delta)
        h, hb = (delta+1)/2, (delta-1)/2
        value = (_global_half(h, z, derivative)*_global_half(hb, zb, derivative)
                 - _global_half(hb, z, derivative)*_global_half(h, zb, derivative))
        if derivative:
            value *= _coordinate_jacobian(coordinate, z)*_coordinate_jacobian(coordinate, zb)
        return value


@lru_cache(maxsize=None)
def _virasoro_reference(exponent, c):
    with mp.workdps(exponent+150):
        h_data = {h: H_q_series_regularized(h, EXT, c=c, q_order=4) for h in (1, 2)}
        def half(h, z):
            q = mp.exp(-mp.pi*mp.ellipk(1-z)/mp.ellipk(z))
            sigma = (mp.convert(c)-1)/24
            prefactor = (16*q)**(-sigma)*q**h*z**sigma*(1-z)**(sigma-mp.mpf('.2'))
            prefactor *= mp.jtheta(3, 0, q)**((mp.convert(c)-1)/2-mp.mpf('1.6'))
            return prefactor*mp.polyval(list(reversed(h_data[h])), q)
        left, right = mp.mpf('.5'), mp.mpf('.5')+mp.mpf(10)**(-exponent)
        return half(2, left)*half(1, right)-half(1, left)*half(2, right)


def _global_sampler(coordinate, left, right, *, projected=False, m_max=1):
    return gb.build_global_sampler(coordinate,
        basis='global_parity_projected_B' if projected else 'global_scalar_G',
        parity='-' if projected else None, ell=1, h_externals=EXT, hbar_externals=EXT,
        m_max=m_max, n_max=m_max, center=left, center_bar=right)


def test_diagonal_antiparity_degree_cancels_even_for_unequal_externals(monkeypatch):
    monkeypatch.setattr(gb, '_interpolate_global_table', lambda **kw: kw)
    with mp.workdps(40):
        data = gb.global_parity_projected_B_interpolated_table('zzb', 1, EXT,
            [F(1, 10), F(1, 5), F(3, 10), F(2, 5)], 1, parity='-', npoles=0)
        assert data['entry_power'](1, 0) == 0


@pytest.mark.parametrize('off_diagonal', [False, True])
def test_large_dimension_hypergeometric_reference_confirms_the_leading_power(off_diagonal):
    # Independent scalar hypergeometric values (the production Taylor engine
    # is not used here). A constant ratio after removing base**Delta verifies
    # power zero both off diagonal at (0,0) and antiparity's diagonal (1,0).
    with mp.workdps(100):
        left, right = (mp.mpf('.3'), mp.mpf('.7')) if off_diagonal else (mp.mpf('.5'),)*2
        ext = [mp.mpf('.1')]*4
        bar = ext if off_diagonal else [mp.mpf('.1'), mp.mpf('.2'), mp.mpf('.3'), mp.mpf('.4')]
        def half(h, z, ext, derivative):
            a, b, c = h+ext[1]-ext[0], h+ext[2]-ext[3], 2*h
            phi = mp.hyp2f1(a, b, c, z)
            if not derivative:
                return z**h*phi/16**h
            return (h*z**(h-1)*phi+z**h*a*b/c*mp.hyp2f1(a+1, b+1, c+1, z))/16**h
        def base(z):
            root = mp.sqrt(1-z)
            return (1-root)/(1+root)/4
        decay = mp.sqrt(base(left)*base(right))
        rows = []
        for delta in (1000, 2000):
            h, hb = mp.mpf(delta+1)/2, mp.mpf(delta-1)/2
            m = 0 if off_diagonal else 1
            value = half(h, left, ext, m)*half(hb, right, bar, 0)-half(hb, left, bar, m)*half(h, right, ext, 0)
            rows.append(value/decay**delta)
        assert abs(rows[1]/rows[0]-1) < mp.mpf('.001')


def test_global_decay_base_accepts_exact_center_and_normalization_labels():
    with mp.workdps(40):
        reference = gb.global_block_exp_base(F(7, 10), center=F(3, 10))
        assert gb.global_block_exp_base(.7, center='.3') == reference
        assert gb.global_block_exp_base('7/10', center='3/10') == reference


@pytest.mark.parametrize('coordinate', ['z', 'y', 'q'])
def test_symbolic_chiral_fields_accept_fraction_and_native_complex_inputs(coordinate):
    ext = (complex(.1, .01), .15, .2, .25)
    with mp.workdps(40):
        table = getattr(pd, 'F_'+coordinate+'_poly_table')(ext, 0, c=32,
            q_order=4, center='.3', z_shift=F(1, 5))
        reference = F_block(F(7, 10), F(3, 10), ext, c=32, q_order=4, z_shift=F(1, 5))
        actual = table.evaluate(F(7, 10))[0][0]
        assert abs(actual-reference) < abs(reference)*mp.mpf('1e-36')


@pytest.mark.parametrize('coordinate', ['zzb', 'yyb', 'qqb'])
def test_symbolic_external_floats_follow_decimal_intent_in_both_prefactor_and_recursion(coordinate):
    with mp.workdps(40):
        builder = getattr(pd, 'G_'+coordinate+'_poly_table')
        actual = builder(0, [.1, .15, .2, .25], 0, c=32, q_order=4, center='.3').evaluate(F(7, 5))[0][0]
        expected = builder(0, [F(1, 10), F(15, 100), F(1, 5), F(1, 4)], 0,
            c=32, q_order=4, center=F(3, 10)).evaluate(F(7, 5))[0][0]
        assert abs(actual-expected) < abs(expected)*mp.mpf('1e-37')
