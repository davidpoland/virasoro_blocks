"""Coordinate conversion retains offsets and the requested evaluation accuracy."""
from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_blocks.api import BlockRequest, NumericalPolicy, RequestValidationError, evaluate_block_request
from virasoro_blocks.derivs import _global_chiral_block
from virasoro_blocks.numeric_input import literal_number


def _request(point, coordinate='y', *, precision=40):
    return BlockRequest(
        operation='evaluate', family='global', basis='chiral_F', coordinate=coordinate,
        h_ext=(0,)*4, exchange_h=1, point=point,
        policy=NumericalPolicy(precision=precision),
    )


@pytest.mark.parametrize('coordinate', ['z', 'y'])
def test_real_backend_keeps_endpoint_accuracy(coordinate):
    exponent = 30
    x = Fraction(1)-Fraction(1, 10**exponent)
    with mp.workdps(60):
        expected = (exponent*mp.log(10))/16 if coordinate == 'z' else (
            2*exponent*mp.log(10)+mp.log(2*(1+mp.mpf(literal_number(x))**2)))/16
    actual = evaluate_block_request(
        _request({coordinate: x}, coordinate, precision=20), q_gate='off').value
    with mp.workdps(40):
        assert abs(actual-expected) < mp.mpf('1e-20')*abs(expected)


@pytest.mark.parametrize('coordinate,value', [('y', 1j), ('y', -1j), ('q', 1), ('q', -1)])
def test_coordinate_conversion_failures_name_the_input(coordinate, value):
    with pytest.raises(RequestValidationError, match=f'cannot convert {coordinate} .*finite cross ratio'):
        evaluate_block_request(_request({coordinate: value}, coordinate), q_gate='off')


def test_full_coordinate_pole_error_names_the_antiholomorphic_input():
    request = BlockRequest(
        operation='evaluate', family='global', basis='ordered_B', coordinate='yyb',
        h_ext=(0,)*4, hbar_ext=(0,)*4, exchange_h=1, exchange_hbar=1,
        point={'y': 0, 'ybar': 1j}, policy=NumericalPolicy(precision=20),
    )
    with pytest.raises(RequestValidationError, match='cannot convert ybar .*coordinate map has a pole'):
        evaluate_block_request(request, q_gate='off')


def test_coordinate_pole_cli_error_is_not_blank(capsys):
    from virasoro_blocks.__main__ import main

    with pytest.raises(SystemExit) as exc:
        main(['evaluate', '--family', 'global', '--basis', 'chiral_F',
              '--h-ext', '0', '0', '0', '0', '--exchange-h', '1',
              '--coordinate', 'y', '--point', 'y=1j', '--precision', '20'])
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert 'cannot convert y' in error and 'coordinate map has a pole' in error
