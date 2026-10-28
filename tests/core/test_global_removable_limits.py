"""The removable global block is the analytic limit in h, not a terminating convention."""
from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_blocks.api import BlockRequest, NumericalPolicy, evaluate_block_request
from virasoro_blocks.coordinates import q_symmetric, z_of_q, z_of_y
from virasoro_blocks.derivs import (
    _global_chiral_block,
    _global_chiral_series_evaluator,
    global_chiral_q_taylor,
    global_chiral_y_taylor,
    global_chiral_z_taylor,
)


def _nearby_reference(h, z, ext, *, norm=16, shift=0):
    """Independent nonsingular 2F1 evaluation, well beyond the tested accuracy."""
    h = mp.convert(h) + mp.mpf('1e-90')
    ext = [mp.convert(value) for value in ext]
    return norm**(-h) * z**(h + shift) * mp.hyp2f1(h + ext[1] - ext[0], h + ext[2] - ext[3], 2*h, z)


@pytest.mark.parametrize('p,j', [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 1), (4, 3)])
@pytest.mark.parametrize('side', ['left', 'right', 'both'])
def test_removable_half_integer_matches_the_limit_in_the_exchange_weight(p, j, side):
    h = Fraction(-p, 2)
    delta = Fraction(p, 2) - j
    other = delta if side == 'both' else Fraction(3, 10)
    ext = (Fraction(1, 4), Fraction(1, 4) + delta, Fraction(1, 2) + other, Fraction(1, 2))
    if side == 'right':
        ext = (ext[3], ext[2], ext[1], ext[0])
    with mp.workdps(180):
        expected = _nearby_reference(h, mp.mpf('.3'), ext, norm=4, shift=mp.mpf('.2'))
    with mp.workdps(40):
        actual = _global_chiral_block(h, mp.mpf('.3'), ext, chiral_norm=4, z_shift='.2')
        assert abs(actual - expected) < mp.mpf('1e-38') * max(1, abs(expected))


@pytest.mark.parametrize('z', ['.3', '.95', '.6+.2j'])
@pytest.mark.parametrize('complex_other', [False, True])
def test_single_removable_numerator_retains_the_hypergeometric_tail(z, complex_other):
    with mp.workdps(100):
        point = mp.mpc('.6', '.2') if z == '.6+.2j' else mp.mpf(z)
        b = mp.mpc('.3', '.125') if complex_other else mp.mpf('.3')
        ext = (mp.mpf('.25'), mp.mpf('.25'), b + mp.mpf('.5'), mp.mpf('.5'))
        # At h=0, (h)_n/(2h)_n -> 1/2 for every n>=1.
        expected = (1 + (1 - point)**(-b)) / 2
    with mp.workdps(40):
        actual = _global_chiral_block(0, point, ext)
        ev = _global_chiral_series_evaluator(ext, 1, z_of_u=lambda u: u, u_symmetric=point,
                                             chiral_norm=None, z_shift=None)
        assert abs(actual - expected) < mp.mpf('1e-38') * abs(expected)
        assert abs(ev(0)[0] - expected) < mp.mpf('1e-38') * abs(expected)


@pytest.mark.parametrize('coordinate', ['z', 'y', 'q'])
def test_removable_taylor_coefficients_and_batched_evaluations_include_the_tail(coordinate):
    ext = (Fraction(3, 10), Fraction(3, 10), Fraction(1, 2), Fraction(1, 5))
    builder = {'z': global_chiral_z_taylor, 'y': global_chiral_y_taylor, 'q': global_chiral_q_taylor}[coordinate]
    with mp.workdps(100):
        z_of_u = {'z': lambda u: u, 'y': z_of_y, 'q': z_of_q}[coordinate]
        center = {'z': mp.mpf('.5'), 'y': mp.mpf(0), 'q': q_symmetric()}[coordinate]
        expected = mp.taylor(lambda u: (1 + (1 - z_of_u(u))**(-mp.mpf('.3'))) / 2, center, 3)
    with mp.workdps(40):
        coefficients = builder(0, ext, 4)
        ev = _global_chiral_series_evaluator(ext, 4, z_of_u=z_of_u, u_symmetric=center,
                                             chiral_norm=None, z_shift=None)
        scalar = ev(0)
        batched = ev.many([0, Fraction(4, 5)])[0]
        for actual_row in (coefficients, scalar, [mp.mpf(c.str(50, radius=False)) for c in batched]):
            for actual, reference in zip(actual_row, expected):
                assert abs(actual - reference) < mp.mpf('1e-37') * max(1, abs(reference))


@pytest.mark.parametrize('basis', ['chiral_F', 'ordered_B', 'scalar_G', 'parity_projected_B'])
def test_public_global_bases_use_the_removable_limit(basis):
    chiral = basis == 'chiral_F'
    actual = evaluate_block_request(BlockRequest(
        operation='evaluate', family='global', basis=basis, coordinate='z' if chiral else 'zzb',
        h_ext=('.3', '.3', '.5', '.2'), hbar_ext=None if chiral else ('.3', '.3', '.5', '.2'),
        exchange_h=0, exchange_hbar=None if chiral else 0,
        parity='+' if basis == 'parity_projected_B' else None,
        point={'z': '.3'} if chiral else {'z': '.3', 'zbar': '.4'},
        policy=NumericalPolicy(precision=40)), q_gate='off').value
    with mp.workdps(100):
        def half(z):
            return (1 + (1 - z)**(-mp.mpf('.3'))) / 2
        expected = half(mp.mpf('.3'))
        if not chiral:
            expected *= half(mp.mpf('.4'))
        if basis == 'parity_projected_B':
            expected *= 2      # B + P[B]; the spin-0 scalar_G is the product block itself
        assert abs(actual - expected) < mp.mpf('1e-38') * abs(expected)


def test_analytic_pole_support_agrees_with_exact_external_classification():
    eps = Fraction(1, 10**70)
    ext = (Fraction(3, 10), Fraction(3, 10) + eps, Fraction(1, 2) + eps, Fraction(1, 2))
    with mp.workdps(40):
        ev = _global_chiral_series_evaluator(ext, 1, z_of_u=lambda u: u,
                                             u_symmetric=mp.mpf('.3'), chiral_norm=None, z_shift=None)
        assert ev.kac_locations()[:3] == [0, mp.mpf('-.5'), -1]
        data = ev.pole_data(0)
        assert data is not None
        residue = mp.mpf(data[0][0].real.str(50, radius=False))
    with mp.workdps(400):
        h, z, e = mp.mpf('1e-200'), mp.mpf('.3'), mp.convert(eps)
        expected = h * 16**(-h) * z**h * mp.hyp2f1(h + e, h + e, 2*h, z)
        assert abs(residue - expected) < mp.mpf('1e-37') * abs(expected)
