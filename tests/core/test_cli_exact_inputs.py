"""CLI validation must preserve the exact labels supplied by the user."""

from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_blocks.__main__ import _parse_spin_token, _parse_spins, _json_spin_value, main
from virasoro_blocks.cli import evaluate, point_table


@pytest.mark.parametrize('module', [evaluate, point_table])
@pytest.mark.parametrize('token', ['.7', '1/1e-2', '.14/.2', '1/-2', '1e400/2'])
def test_cli_real_literals_preserve_spelling(module, token):
    assert module._number(token) == token


@pytest.mark.parametrize('module', [evaluate, point_table])
@pytest.mark.parametrize('token', ['abc/1', '1/abc', '1/0', 'nan', 'inf', '1/inf', '1/2/3', '1+2j'])
def test_cli_invalid_real_literals_are_argument_errors(module, token):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        module._number(token)


@pytest.mark.parametrize('operation', ['evaluate', 'point-table', 'table'])
@pytest.mark.parametrize('token', ['0.500000000000000000000001', '1/0', 'nan', '1/abc'])
def test_cli_rejects_invalid_spin_cleanly(operation, token, capsys):
    if operation == 'evaluate':
        args = ['--point', 'z=.3', '--exchange-spin', token]
    elif operation == 'point-table':
        args = ['--lambda', '0', '--exchange-spin', token]
    else:
        args = ['--lambda', '0', '--spin-ext', token, '0', '0', '0']
    with pytest.raises(SystemExit) as error:
        main([operation, *args])
    assert error.value.code == 2
    assert 'error:' in capsys.readouterr().err


@pytest.mark.parametrize('module', [evaluate, point_table])
def test_cli_accepts_negative_fraction_with_decimal_denominator(module):
    required = ['--point', 'z=.3'] if module is evaluate else ['--lambda', '0']
    args = module.build_parser().parse_args([*required, '--z-shift', '-.1/2e-1'])
    assert args.z_shift == '-.1/2e-1'


def test_large_spin_range_terminates_and_retains_adjacent_labels():
    # Preserve unit increments even when the start exceeds the working precision.
    start = 2**100 + 1
    with mp.workdps(15):
        assert _parse_spin_token(f'{start}:{start+2}') == [start, start+1, start+2]
        spins = _parse_spins([str(start+1), str(start), str(start)])
        assert spins == [start, start+1]
        assert _json_spin_value(spins[0]) == start


@pytest.mark.parametrize('token', ['.500000000000000000000001', '1.000000000000000000000001:2', '0:.500000000000000000000001:2'])
def test_table_spin_ranges_reject_explicit_offsets(token):
    with mp.workdps(15), pytest.raises(ValueError, match='integer or half-integer'):
        _parse_spin_token(token)


def test_table_spin_range_retains_exact_half_integer_endpoints():
    with mp.workdps(15):
        assert _parse_spin_token('1/2:5e-1:1.5') == [Fraction(1, 2), 1, Fraction(3, 2)]


def test_evaluate_rejects_nonfinite_complex_point(capsys):
    for token in ('nan+1j', '(.7+1j', '.7+1j)', '1'*20000+'xj'):
        with pytest.raises(SystemExit) as error:
            main(['evaluate', '--point', 'z='+token])
        assert error.value.code == 2
        assert 'invalid complex value' in capsys.readouterr().err


def test_table_rejects_explicit_external_spin_offset(capsys):
    with pytest.raises(SystemExit) as error:
        main(['table', '--family', 'global', '--basis', 'ordered_B', '--lambda', '0',
              '--h-ext', '.50000000000000000000000000000000000000000000000000000000000000001', '0', '0', '0',
              '--hbar-ext', '0', '0', '0', '0'])
    assert error.value.code == 2
    assert 'not an integer or half-integer' in capsys.readouterr().err


@pytest.mark.parametrize('operation', ['evaluate', 'point-table', 'table'])
@pytest.mark.parametrize('token', ['garbagej', 'nan+1j', 'abc/1', '1/abc', '1/0', 'inf'])
def test_bad_central_charge_is_a_clean_cli_error(operation, token, capsys):
    required = ['--point', 'z=.3'] if operation == 'evaluate' else ['--lambda', '0']
    try:
        status = main([operation, '--central-charge', token, *required])
    except SystemExit as error:
        status = error.code
    assert status == 2
    message = capsys.readouterr().err
    assert 'error:' in message
    assert 'Traceback (most recent call last)' not in message
    assert token in message or 'denominator' in message


@pytest.mark.parametrize('operation', ['table', 'point-table'])
def test_cli_preserves_exact_external_labels_in_written_tables(operation, tmp_path):
    import json

    extra = ['--spin-ranges', '0'] if operation == 'table' else ['--exchange-delta', '2']
    main([operation, '--family', 'global', '--basis', 'scalar_G', '--lambda', '0',
          '--h-ext', *(['1/7']*4), '--hbar-ext', *(['1/7']*4),
          '--coordinates', 'zzb', '--precision', '25', '-o', str(tmp_path), *extra])
    [path] = list(tmp_path.glob('*.json'))
    doc = json.loads(path.read_text())
    assert doc['h_external'] == ['1/7']*4
    assert doc['delta_external'] == ['2/7']*4


def test_torus_cli_preserves_exact_central_charge_label(tmp_path):
    import json

    main(['table', '--surface', 'torus', '--family', 'torus-character',
          '--central-charge', '8/7', '--lambda', '0', '--spin-ranges', '0',
          '--precision', '25', '-o', str(tmp_path)])
    [path] = list(tmp_path.glob('*.json'))
    assert json.loads(path.read_text())['central_charge'] == '8/7'


def test_large_spin_cli_keeps_distinct_twists_and_exact_generated_labels(tmp_path):
    import argparse
    import json
    from types import SimpleNamespace
    from virasoro_blocks.__main__ import _resolve_large_spin_twists

    epsilon = Fraction(1, 10**70)
    args = SimpleNamespace(large_spin_double_twists=2,
        large_spin_twists=['.7', '7/10', Fraction(7, 10)+epsilon, .7, '.7'])
    expected = [Fraction(7, 10), Fraction(27, 10), Fraction(7, 10)+epsilon]
    for dps in (3, 25, 80):
        with mp.workdps(dps):
            actual = _resolve_large_spin_twists(argparse.ArgumentParser(), args, [Fraction(7, 40)]*4)
            assert actual == expected
            assert all(isinstance(value, Fraction) for value in actual)
    # The CLI writes each distinct slice, preserving its exact twist metadata.
    shifted = '.700000000000000000000001'
    main(['table', '--family', 'global', '--basis', 'scalar_G', '--lambda', '0', '--npoles', '2',
          '--large-spin-lmin', '10', '--large-spin-double-twists', '1',
          '--large-spin-twists', '.7', shifted, '7/10',
          '--delta-ext', *(['.35']*4), '--coordinates', 'zzb', '--precision', '25', '-o', str(tmp_path)])
    files = sorted(tmp_path.glob('*.json'))
    assert [path.name for path in files] == ['derivs_large_spin_0.json', 'derivs_large_spin_1.json']
    assert [json.loads(path.read_text())['twist'] for path in files] == ['0.7', '0'+shifted]
