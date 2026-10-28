"""Point accuracy includes cancellation in the composed block, including near a symmetry zero."""
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
import warnings

import mpmath as mp
import pytest

from virasoro_blocks.api import BlockRequest, NumericalPolicy, evaluate_block_request
from virasoro_blocks.conditioning import InsufficientPrecisionError, PrecisionLossWarning
from virasoro_blocks.verified_build import TargetDigitsError


def _request(offset=40, *, family='global', target=None, cap=None):
    return BlockRequest(
        operation='evaluate', family=family, basis='parity_projected_B', coordinate='zzb',
        central_charge=None if family == 'global' else 10,
        h_ext=('0.1',) * 4, hbar_ext=('0.1',) * 4, exchange_delta='1.4', exchange_spin=1,
        parity='-', point={'z': '0.5', 'zbar': '0.5' + '0' * (offset - 2) + '1'},
        policy=NumericalPolicy(precision=60, order=6, target_digits=target, max_precision=cap),
    )


def _global_reference(offset):
    # Independent of the block engine and its projection/composition code.
    with mp.workdps(180):
        z, zbar = mp.mpf('.5'), mp.mpf('.5') + mp.mpf(10) ** -offset
        h, hb = mp.mpf('1.2'), mp.mpf('.2')
        def k(weight, point):
            return (point / 16) ** weight * mp.hyp2f1(weight, weight, 2 * weight, point)
        return k(h, z) * k(hb, zbar) - k(hb, z) * k(h, zbar)


@pytest.mark.parametrize('offset', [40, 80])
@pytest.mark.parametrize('kind', ['text', 'fraction', 'decimal', 'mpf'])
def test_global_projection_reaches_target_from_original_inputs(offset, kind):
    request = _request(offset, target=40)
    label = request.point['zbar']
    if kind == 'fraction':
        label = Fraction(label)
    elif kind == 'decimal':
        label = Decimal('0.5' + '0' * (offset - 2) + '1')
    elif kind == 'mpf':
        with mp.workdps(180):
            label = mp.mpf('.5') + mp.mpf(10) ** -offset
    request = replace(request, point={'z': '0.5', 'zbar': label})
    actual = evaluate_block_request(request, q_gate='strict')
    with mp.workdps(180):
        expected = _global_reference(offset)
        assert abs(actual.value / expected - 1) < mp.mpf('1e-40')
    assert actual.metadata['precision_actual'] >= offset + 40
    assert actual.metadata['estimated_digits'] >= 40
    assert mp.mp.dps == 15


def test_projection_without_target_reports_the_digits_lost():
    with pytest.warns(PrecisionLossWarning, match='block composition'):
        actual = evaluate_block_request(_request(), q_gate='strict')
    assert 19 < actual.metadata['estimated_digits'] < 22
    with mp.workdps(180):
        error = abs(actual.value / _global_reference(40) - 1)
        assert error < mp.mpf(10) ** -actual.metadata['estimated_digits']
    with pytest.raises(InsufficientPrecisionError, match='block composition'):
        evaluate_block_request(_request(80), q_gate='strict')


def test_projection_stops_at_precision_cap():
    with pytest.raises(TargetDigitsError, match='max_precision 90'):
        evaluate_block_request(_request(80, target=40, cap=90), q_gate='strict')
    assert mp.mp.dps == 15


@pytest.mark.parametrize('family', ['virasoro', 'ns', 'ramond'])
def test_recursive_projection_reaches_target_for_the_truncated_block(family):
    request = _request(80, family=family, target=30)
    expected = evaluate_block_request(replace(request, policy=NumericalPolicy(precision=180, order=6)),
                                      q_gate='off')
    actual = evaluate_block_request(request, q_gate='off')
    with mp.workdps(180):
        assert expected.value != 0
        assert abs(actual.value / expected.value - 1) < mp.mpf('1e-30')
    assert actual.metadata['precision_actual'] >= 110
    # Increasing arithmetic precision does not remove the q-order truncation error.
    assert actual.metadata['estimated_digits'] <= actual.metadata['q_truncation_digits'] < 30


@pytest.mark.parametrize('family', ['global', 'virasoro', 'ns', 'ramond'])
@pytest.mark.parametrize('zero', ['diagonal', 'identical_halves'])
@pytest.mark.parametrize('parity,phase', [('-', '1'), ('+', '-1')])
def test_exact_symmetry_zeros_do_not_escalate(family, zero, parity, phase):
    request = replace(_request(family=family, target=40), parity=parity, parity_exchanged=phase)
    if zero == 'diagonal':
        # Parity reverses the weights and shifts together, even for spinning externals.
        request = replace(request, point={'z': '1/2', 'zbar': '0.5'},
                          hbar_ext=('0.2',) * 4, z_shift='0.2', zbar_shift='0.3')
    else:
        request = replace(request, point={'z': '.3', 'zbar': '.4'}, exchange_spin=0)
    with warnings.catch_warnings():
        warnings.simplefilter('error', PrecisionLossWarning)
        actual = evaluate_block_request(request, q_gate='off')
    assert actual.value == 0
    assert actual.metadata.get('precision_actual', 60) == 60


def test_imaginary_phase_does_not_turn_the_diagonal_into_a_zero():
    request = replace(_request(target=40), point={'z': '.5', 'zbar': '.5'}, parity_exchanged='i')
    actual = evaluate_block_request(request, q_gate='strict')
    assert actual.value.real > 0
    with mp.workdps(80):
        assert abs(actual.value.imag + actual.value.real) < mp.mpf('1e-60')
    assert actual.metadata['estimated_digits'] == 60


@pytest.mark.parametrize('ext,anti', [
    (('.1',) * 4, ('.2',) * 4),
    (('.1', '.2', '.5', '.3'), ('.3', '.5', '.2', '.1')),
])
def test_global_equivalent_chiral_parameters_are_an_exact_zero(ext, anti):
    request = replace(_request(target=40), point={'z': '.3', 'zbar': '.4'}, exchange_spin=0,
                      h_ext=ext, hbar_ext=anti)
    actual = evaluate_block_request(request, q_gate='strict')
    assert actual.value == 0
    assert actual.metadata.get('precision_actual', 60) == 60
    # Unequal shifts break the equality even when the 2F1 parameters coincide.
    shifted = evaluate_block_request(replace(request, z_shift='.1', zbar_shift='.2'), q_gate='strict')
    assert shifted.value != 0


@pytest.mark.parametrize('coordinate,points', [
    ('zzb', {'z': Fraction(1, 2), 'zbar': Fraction(1, 2) + Fraction(1, 10**400)}),
    ('qqb', {'q': Fraction(3, 100), 'qbar': Fraction(3, 100) + Fraction(1, 10**400)}),
])
def test_unresolved_coordinates_are_not_a_symmetry_zero(coordinate, points):
    request = replace(_request(target=20, cap=80), coordinate=coordinate, point=points)
    with pytest.raises(TargetDigitsError, match='max_precision 80'):
        evaluate_block_request(request, q_gate='strict')


def test_low_level_composer_reports_cancellation():
    from virasoro_blocks.composer import parity_projected_B_block

    with mp.workdps(60), pytest.warns(PrecisionLossWarning, match='block composition'):
        parity_projected_B_block('1.2', '.2', Fraction(1, 2), Fraction(1, 2) + Fraction(1, 10**40),
                                 ['.1'] * 4, ['.1'] * 4, parity='-', c=10, q_order=6)
