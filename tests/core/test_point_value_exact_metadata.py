"""Saved point-value input labels reproduce the evaluated physical block."""
from fractions import Fraction
import json

import mpmath as mp
import pytest

from virasoro_blocks.api import BlockRequest, NumericalPolicy, evaluate_block_request
from virasoro_blocks.numeric_input import literal_number


def _decode(value):
    if isinstance(value, dict):
        with mp.workdps(400):
            return mp.mpc(value['real'], value['imag'])
    return value


def _replay(doc):
    fields = dict(operation='evaluate', surface=doc['surface'], family=doc['family'],
                  basis=doc['basis'], coordinate=doc['coordinate'],
                  point={key: _decode(value) for key, value in doc['point'].items()},
                  policy=NumericalPolicy(precision=doc['precision'], order=doc['order']))
    if 'central_charge' in doc:
        fields['central_charge'] = _decode(doc['central_charge'])
    for key, values in doc.get('external', {}).items():
        fields[{'h': 'h_ext', 'hbar': 'hbar_ext', 'delta': 'delta_ext', 'spin': 'spin_ext'}[key]] = tuple(
            _decode(value) for value in values)
    exchange = doc['exchange']
    if doc['surface'] == 'torus' or 'hbar' in exchange:
        fields['exchange_delta'] = _decode(exchange['delta'])
        fields['exchange_spin'] = _decode(exchange['spin'])
    else:
        fields['exchange_h'] = _decode(exchange['h'])
    for key in ('chiral_norm', 'z_shift', 'zbar_shift', 'torus_onepoint_ext',
                'parity', 'parity_external', 'parity_exchanged', 'sector', 'branch'):
        if key in doc:
            fields[key] = _decode(doc[key])
    return evaluate_block_request(BlockRequest(**fields), q_gate='off')


def _round_trip(request):
    original = evaluate_block_request(request, q_gate='off')
    with mp.workdps(15):
        doc = json.loads(json.dumps(original.as_dict()))
        replay = _replay(doc)
    with mp.workdps(50):
        assert abs(replay.value-original.value) < mp.mpf('1e-29')*max(1, abs(original.value))
    return doc


def test_minimal_model_value_replays_without_detuning_its_exact_inputs():
    request = BlockRequest(
        operation='evaluate', basis='chiral_F', coordinate='z', central_charge=Fraction(6, 7),
        h_ext=(Fraction(1, 56),)*4, exchange_h=Fraction(1, 21), point={'z': '.3'},
        policy=NumericalPolicy(precision=30, order=12),
    )
    doc = _round_trip(request)
    assert doc['central_charge'] == '6/7'
    assert doc['external']['h'] == ['1/56']*4
    assert doc['regularization_policy'] == 'merged'


def test_nonterminating_point_and_prefactor_labels_replay_exactly():
    request = BlockRequest(
        operation='evaluate', family='global', basis='chiral_F', coordinate='z',
        h_ext=(Fraction(1, 3),)*4, exchange_h=Fraction(4, 7), point={'z': Fraction(2, 7)},
        chiral_norm=Fraction(13, 7), z_shift=Fraction(1, 11),
        policy=NumericalPolicy(precision=40),
    )
    doc = _round_trip(request)
    assert doc['point']['z'] == '2/7'
    assert doc['chiral_norm'] == '13/7' and doc['z_shift'] == '1/11'


@pytest.mark.parametrize('family', ['torus-character', 'torus-onepoint'])
def test_torus_input_labels_replay(family):
    request = BlockRequest(
        operation='evaluate', surface='torus', family=family, coordinate='vvb',
        central_charge=Fraction(2), exchange_delta=Fraction(5, 7), exchange_spin=0,
        torus_onepoint_ext=Fraction(1, 3) if family == 'torus-onepoint' else None,
        point={'v': Fraction(1, 7), 'vbar': Fraction(1, 7)},
        policy=NumericalPolicy(precision=30, order=4),
    )
    doc = _round_trip(request)
    assert doc['point']['v'] == '1/7'
    if family == 'torus-onepoint':
        assert doc['torus_onepoint_ext'] == '1/3'


@pytest.mark.parametrize('phase', ['-1', 'i'])
def test_projected_block_metadata_keeps_its_reversal_phase(phase):
    request = BlockRequest(
        operation='evaluate', family='global', basis='parity_projected_B', coordinate='zzb',
        h_ext=('.1', '.2', '.3', '.4'), hbar_ext=('.2', '.1', '.4', '.3'),
        exchange_h='.8', exchange_hbar='.3', point={'z': '.3', 'zbar': '.4'},
        parity='+', parity_external=('1', '-1', '1', '1'), parity_exchanged=phase,
        policy=NumericalPolicy(precision=40),
    )
    doc = _round_trip(request)
    assert doc['parity_external'] == ['1', '-1', '1', '1']
    assert doc['parity_exchanged'] == phase


def test_complex_source_components_are_retained_as_input_metadata():
    with mp.workdps(120):
        external = mp.mpc('.3', '1e-70')
        point = mp.mpc('.4', '1e-70')
    request = BlockRequest(
        operation='evaluate', family='virasoro', basis='chiral_F', coordinate='z',
        central_charge=32, h_ext=(external,)*4, exchange_h='.8', point={'z': point},
        policy=NumericalPolicy(precision=40, order=4),
    )
    doc = _round_trip(request)
    for stored, original in [(doc['external']['h'][0], external), (doc['point']['z'], point)]:
        assert literal_number(stored['real']) == literal_number(original.real)
        assert literal_number(stored['imag']) == literal_number(original.imag)


@pytest.mark.parametrize('coordinate,point', [
    ('q', '.95'),
    ('y', '0.' + '9'*40),
    ('y', '-0.' + '9'*40),
])
def test_transformed_point_metadata_replays_near_an_endpoint(coordinate, point):
    request = BlockRequest(
        operation='evaluate', family='global', basis='chiral_F', coordinate=coordinate,
        h_ext=(0,)*4, exchange_h=1, point={coordinate: point},
        policy=NumericalPolicy(precision=40),
    )
    doc = _round_trip(request)
    assert literal_number(doc['point']['z']) not in (0, 1)


def test_transformed_complex_point_metadata_keeps_a_small_resolved_component():
    with mp.workdps(120):
        point = mp.mpc('.2', '1e-70')
    request = BlockRequest(
        operation='evaluate', family='global', basis='chiral_F', coordinate='y',
        h_ext=(0,)*4, exchange_h=1, point={'y': point},
        policy=NumericalPolicy(precision=40),
    )
    doc = _round_trip(request)
    assert isinstance(doc['point']['z'], dict)
    assert literal_number(doc['point']['z']['imag']) != 0
