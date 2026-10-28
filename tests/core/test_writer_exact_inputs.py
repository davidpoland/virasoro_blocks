"""Output metadata must not change the identity or parity of input labels."""
from fractions import Fraction
import json

import mpmath as mp
import numpy as np
import pytest

from virasoro_blocks.numeric_input import parse_number
from virasoro_blocks.polynomial import Polynomial
from virasoro_blocks.serialization import SerializedDerivTable, serialized_table_from_json
from virasoro_blocks.truncation import kept_pairs, parity_smoothing_label
from virasoro_blocks.writers import _decimal_label_str, _metadata_num_str, _num_str, _spin_json_value, write_json, write_mathematica, write_torus_json


@pytest.mark.parametrize("basis", [None, "ordered_B"])
def test_json_external_iterators_preserve_all_metadata(basis):
    h, hb = [".7"] * 4, [".2"] * 4
    kwargs = dict(c=8, q_order=0, basis=basis)
    expected = write_json(table(), h_ext=h, hbar_ext=hb, **kwargs)
    actual = write_json(table(), h_ext=iter(h), hbar_ext=iter(hb), **kwargs)
    assert json.loads(actual) == json.loads(expected)


def test_mathematica_filename_external_iterators_preserve_spin_labels():
    from virasoro_blocks.writers import mathematica_filename

    h, hb = [".7"] * 4, [".2"] * 4
    kwargs = dict(c=8, spin=0, q_order=4, basis="ordered_B")
    expected = mathematica_filename("zzb", h_ext=h, hbar_ext=hb, **kwargs)
    actual = mathematica_filename("zzb", h_ext=iter(h), hbar_ext=iter(hb), **kwargs)
    assert actual == expected
    assert "spin1-0.5" in actual


@pytest.mark.parametrize("kind", ["json", "mathematica", "filename"])
@pytest.mark.parametrize("field", ["h_ext", "hbar_ext"])
@pytest.mark.parametrize("values", [[0] * 3, [0] * 5, "0000"])
def test_writers_require_four_supplied_external_weights(kind, field, values):
    from virasoro_blocks.writers import mathematica_filename

    kwargs = dict(c=8, q_order=0, h_ext=[0] * 4, hbar_ext=[0] * 4)
    kwargs[field] = values
    with pytest.raises(ValueError, match=field + ".*four"):
        if kind == "filename":
            mathematica_filename("zzb", spin=0, **kwargs)
        elif kind == "mathematica":
            write_mathematica(table(), coordinate="zzb", **kwargs)
        else:
            write_json(table(), **kwargs)


def table(coord='zzb', size=1):
    return SerializedDerivTable(shift=0, base=1, pole_list_x=(), coordinates=(coord,),
                                numerators={coord: [[Polynomial([1]) for _ in range(size)] for _ in range(size)]})


@pytest.mark.parametrize('spin', ['.500000000000000000000000001', 0.5000000000000001, complex(2, 3), mp.mpc(2, '1e-30'),
                                  float('inf'), mp.mpf('nan')])
def test_writer_rejects_invalid_spin_instead_of_relabelling(spin):
    with mp.workdps(30), pytest.raises(ValueError):
        write_json(table(), c=8, q_order=0, h_ext=(0,)*4, hbar_ext=(0,)*4, spin=spin)


@pytest.mark.parametrize("spin", [True, False, np.bool_(True), np.bool_(False)])
@pytest.mark.parametrize("kind", ["json", "mathematica", "torus"])
def test_writers_reject_boolean_spin_labels(spin, kind):
    with pytest.raises(ValueError, match="spin"):
        if kind == "torus":
            write_torus_json(table("vvb"), c=8, spin=spin)
        elif kind == "mathematica":
            write_mathematica(table(), coordinate="zzb", ell=spin)
        else:
            write_json(table(), c=8, q_order=0, h_ext=(0,) * 4, hbar_ext=(0,) * 4, spin=spin)


@pytest.mark.parametrize('spin', [2**100+1, -(2**100+1), Fraction(3, 2), Fraction(-3, 2)])
def test_exact_spin_is_written_without_ambient_rounding(spin):
    from virasoro_blocks.writers import mathematica_filename

    with mp.workdps(15):
        doc=json.loads(write_json(table(), c=8, q_order=0, h_ext=(0,)*4, hbar_ext=(0,)*4, spin=spin))
        if Fraction(spin).denominator == 2:
            assert doc['basis'] == 'ordered_B' and doc['parity_smoothing'] is None
            mma=write_mathematica(table(), coordinate='zzb', ell=spin)
            assert 'basis -> "ordered_B"' in mma and 'paritySmoothing -> Null' in mma
            filename=mathematica_filename('zzb', c=8, q_order=0, h_ext=(0,)*4,
                                          hbar_ext=(0,)*4, spin=spin)
            assert 'basisOrderedB' in filename and '/' not in filename
            with pytest.raises(ValueError, match='integer'):
                write_json(table(), c=8, q_order=0, h_ext=(0,)*4, hbar_ext=(0,)*4,
                           spin=spin, basis='scalar_G')
    assert Fraction(doc['spin']) == spin


@pytest.mark.parametrize('sign', [1, -1])
def test_unrepresentable_half_integer_spin_fails_closed(sign):
    with pytest.raises(ValueError, match='too large'):
        _spin_json_value(sign * Fraction(2**53+1, 2))
    assert Fraction(_spin_json_value(sign * Fraction(2**53-1, 2))) == sign * Fraction(2**53-1, 2)


def test_external_spin_is_derived_before_rounding_weights():
    h=Fraction(10**60)+Fraction(1, 2)
    hb=Fraction(10**60)
    with mp.workdps(15):
        doc=json.loads(write_json(table(), c=8, q_order=0, h_ext=(h,)*4, hbar_ext=(hb,)*4, spin=0))
        mma=write_mathematica(table(), coordinate='zzb', ell=0, basis='ordered_B', c=8, q_order=0,
                              h_ext=(h,)*4, hbar_ext=(hb,)*4)
    assert doc['spin_external'] == [.5]*4
    assert doc['basis'] == 'ordered_B'
    assert 'spinExternal -> {0.5, 0.5, 0.5, 0.5}' in mma


@pytest.mark.parametrize("kind", ["json", "mathematica"])
@pytest.mark.parametrize("basis", [None, "ordered_B"])
@pytest.mark.parametrize("complex_weights", [False, True])
def test_binary_external_spin_does_not_round_an_unrepresentable_half_to_integer(kind, basis, complex_weights):
    with mp.workdps(100):
        h, hb = mp.mpf(2) ** 100 + mp.mpf(".5"), mp.mpf(0)
        if complex_weights:
            h, hb = mp.mpc(h, 2), mp.mpc(hb, 2)
    kwargs = dict(c=8, q_order=0, basis=basis, h_ext=[h] * 4, hbar_ext=[hb] * 4)
    with mp.workdps(15), pytest.raises(ValueError, match="too large"):
        if kind == "json":
            write_json(table(), **kwargs)
        else:
            write_mathematica(table(), coordinate="zzb", **kwargs)


def test_binary_external_spin_retains_existing_roundoff_allowance():
    with mp.workdps(15):
        h, hb = mp.mpf(".7"), mp.mpf(".2")
        doc = json.loads(write_json(table(), c=8, q_order=0, h_ext=[h] * 4, hbar_ext=[hb] * 4))
    assert doc["spin_external"] == [.5] * 4


def test_external_spin_offset_is_not_lost_during_weight_conversion():
    with mp.workdps(15), pytest.raises(ValueError, match='half-integer'):
        write_json(table(), c=8, q_order=0, h_ext=('.500000000000000000000000001',)*4, hbar_ext=(0,)*4, spin=0)


def test_complex_external_dimensions_are_not_discarded():
    doc=json.loads(write_json(table(), c=8, q_order=0, h_ext=(1+2j,)*4, hbar_ext=(2j,)*4, spin=0))
    assert doc['spin_external'] == [1]*4
    assert all(mp.mpmathify(x)==1+4j for x in doc['delta_external'])


def test_exact_written_tail_is_not_treated_as_guard_digit_noise():
    literal='.700000000000000000000000001'
    with mp.workdps(30):
        assert Fraction(_num_str(parse_number(literal),30)) == Fraction(literal)
        assert Fraction(_num_str(literal,30)) == Fraction(literal)
        doc=json.loads(write_json(table(), c=literal, q_order=0, h_ext=(literal,)*4, hbar_ext=(literal,)*4,
                                   spin=0, z_center=literal, z_shift=literal, chiral_norm=literal))
    for key in ['central_charge', 'z_center', 'z_shift', 'chiral_normalization_N']:
        assert Fraction(doc[key]) == Fraction(literal)
    assert all(Fraction(v)==Fraction(literal) for v in doc['h_external'])


def test_exact_fraction_formatting_honours_requested_digits_above_ambient_precision():
    with mp.workdps(15):
        text=_num_str(Fraction(1, 7),60)
    with mp.workdps(75):
        assert abs(mp.mpf(text)-mp.mpf(1)/7)<mp.mpf('1e-60')


@pytest.mark.parametrize('imaginary', ['1e-70', '-1e-70', '0'])
def test_complex_input_metadata_retains_both_supplied_components(imaginary):
    from virasoro_blocks.numeric_input import literal_components

    with mp.workdps(100):
        value = mp.mpc('1.00000000000000000000000000000000000000000000000000000000000000000000001', imaginary)
        real, imag = literal_components(value)
    with mp.workdps(15):
        doc = json.loads(write_json(table('z'), basis='chiral_F', c=value, q_order=0,
                                   h_ext=(value,)*4, z_center=value, z_shift=value))
        mma = write_mathematica(table('z'), coordinate='z', basis='chiral_F', c=value,
                                h_ext=(value,)*4, z_center=value, z_shift=value)
        expected_mma = _decimal_label_str(value, 15, mma=True)
        expected_shift = _decimal_label_str(value, 15)
    with mp.workdps(150):
        for text in (doc['central_charge'], doc['z_center'], *doc['h_external']):
            assert literal_components(parse_number(text)) == (real, imag)
    assert doc['z_shift'] == expected_shift
    assert f'centralCharge -> {expected_mma}' in mma
    assert f'zCenter -> {expected_mma}' in mma
    assert f'zShift -> {expected_mma}' in mma


@pytest.mark.parametrize('value', [Fraction(1, 7), Fraction(-11, 28), Fraction(7, 10),
                                  Fraction(1, 2**100), Fraction(1, 5**100),
                                  Fraction(10**60)+Fraction(1, 2), 0, '.700000000000000000000000001'])
def test_exact_metadata_survives_below_label_precision(value):
    expected = Fraction(value)
    with mp.workdps(15):
        text = _metadata_num_str(value, 15)
        mma = _metadata_num_str(value, 15, mma=True)
    assert Fraction(text) == expected
    assert Fraction(mma) == expected
    assert ('/' in text) == (expected.denominator % 7 == 0)


def test_rational_metadata_round_trips_through_json_and_mathematica():
    value = Fraction(1, 7)
    with mp.workdps(15):
        doc = json.loads(write_json(table(), c=value, q_order=0, h_ext=(value,)*4,
                                    hbar_ext=(value,)*4, spin=0, z_center=value,
                                    z_shift=value, chiral_norm=value))
        mma = write_mathematica(table(), coordinate='zzb', ell=0, c=value,
                                h_ext=(value,)*4, hbar_ext=(value,)*4,
                                z_center=value, z_shift=value, chiral_norm=value)
        restored = serialized_table_from_json(doc)
        assert restored.evaluate('zzb', 1)[0][0] == 1
    # Input identity labels stay exact; approximate metadata uses decimals.
    for key in ['central_charge', 'z_center']:
        assert doc[key] == '1/7'
    for key in ['z_shift', 'chiral_normalization_N']:
        assert doc[key] == '0.142857142857143'
    assert doc['h_external'] == ['1/7']*4
    assert doc['delta_external'] == ['2/7']*4
    assert 'centralCharge -> 0.142857142857143`15' in mma
    assert 'zCenter -> 0.142857142857143`15' in mma
    assert '1/7' not in mma


def test_binary_external_and_crossing_shift_labels_remain_consistent():
    from virasoro_blocks.numeric_input import literal_number

    with mp.workdps(40):
        weight = mp.mpf(1) / 3
        shift = -2 * weight
    exact_weight = literal_number(weight)
    with mp.workdps(15):
        doc = json.loads(write_json(table('z'), c=8, q_order=0, basis='chiral_F',
                                    h_ext=(weight,)*4, z_shift=shift, sig_figs=15))
        mma = write_mathematica(table('z'), coordinate='z', ell=0, basis='chiral_F', c=8,
                                h_ext=(weight,)*4, z_shift=shift, sig_figs=15)
        sphere = json.loads(write_json(table(), c=8, q_order=0, spin=0,
                                       h_ext=(weight,)*4, hbar_ext=(weight,)*4, sig_figs=15))
    assert all(Fraction(value) == exact_weight for value in doc['h_external'])
    assert doc['z_shift'] == '-0.666666666666667'
    assert 'hExternal -> {0.333333333333333`15, 0.333333333333333`15, 0.333333333333333`15, 0.333333333333333`15}' in mma
    assert 'zShift -> -0.666666666666667`15' in mma
    assert all(Fraction(value) == 2 * exact_weight for value in sphere['delta_external'])


def test_binary_input_metadata_survives_requested_output_rounding():
    from virasoro_blocks.numeric_input import literal_number

    with mp.workdps(40):
        value = mp.mpf(1) / 3
    with mp.workdps(15):
        doc = json.loads(write_json(table(), c=value, q_order=0, spin=0,
                                    h_ext=(value,)*4, hbar_ext=(value,)*4,
                                    z_center=value, zbar_center=value, z_shift=value,
                                    zbar_shift=value, chiral_norm=value, sig_figs=15))
        torus = json.loads(write_torus_json(table('vvb'), c=value, basis='torus_onept_G',
                                            onept_d_ext=value, q_order=0, sig_figs=15))
    expected = literal_number(value)
    for key in ('central_charge', 'z_center', 'zbar_center'):
        assert Fraction(doc[key]) == expected
    for key in ('z_shift', 'zbar_shift', 'chiral_normalization_N'):
        assert doc[key] == '0.333333333333333'
    assert Fraction(torus['central_charge']) == expected
    assert torus['onept_d_ext'] == '0.333333333333333'


def test_mathematica_rejects_invalid_spin_before_parity_selection():
    with pytest.raises(ValueError, match='real'):
        write_mathematica(table('xt'), coordinate='xt', ell=2+3j, c=8)


def test_torus_metadata_preserves_rational_labels():
    doc = json.loads(write_torus_json(table('vvb'), c=Fraction(8, 7),
                                      basis='torus_onept_G', onept_d_ext=Fraction(1, 7),
                                      q_order=0))
    assert doc['central_charge'] == '8/7'
    assert doc['onept_d_ext'] == '0.142857142857143'
    assert serialized_table_from_json(doc).evaluate('vvb', 1)[0][0] == 1


def test_computed_spin_roundoff_still_survives_precision_escalation():
    with mp.workdps(300):
        spin=mp.mpf(1)+mp.mpf('1e-240')
    with mp.workdps(316):
        assert _spin_json_value(spin, allow_roundoff=True)==1
        with pytest.raises(ValueError, match='spin'):
            _spin_json_value(spin)


def test_stored_mpmath_spin_is_not_rounded_at_writer_precision():
    with mp.workdps(100):
        spin = mp.mpf(2**100+1)
        half_spin = mp.mpf(2**100)+mp.mpf('.5')
        shifted = mp.mpf('.5') + mp.mpf('1e-80')
        complex_shifted = mp.mpc(shifted, 0)
    with mp.workdps(15):
        assert _spin_json_value(spin) == 2**100+1
        # (the scalar G is z <-> zbar even at every spin, so the label no longer depends on the spin's parity)
        assert parity_smoothing_label('scalar_G', ell=spin) == 'p_plus'
        with pytest.raises(ValueError, match='too large'):
            _spin_json_value(half_spin)
        for value in (shifted, complex_shifted, complex(0.5000000000000001, 0)):
            with pytest.raises(ValueError, match='spin'):
                write_json(table(), c=8, q_order=0, h_ext=(0,) * 4,
                           hbar_ext=(0,) * 4, spin=value, basis='ordered_B')
            with pytest.raises(ValueError, match='spin'):
                write_mathematica(table(), coordinate='zzb', ell=value, basis='ordered_B')
            with pytest.raises(ValueError, match='spin'):
                write_torus_json(table('vvb'), c=8, spin=value)


@pytest.mark.parametrize('center, center_bar', [(None, '.4'), ('.5', '.500000000000000000000000000000001')])
def test_distinct_centers_keep_off_diagonal_derivatives(center, center_bar):
    with mp.workdps(15):
        doc = json.loads(write_json(table(size=2), c=8, q_order=0,
                                    h_ext=(0,)*4, hbar_ext=(0,)*4, spin=0,
                                    z_center=center, zbar_center=center_bar))
        assert len(doc['derivs'][0][0]) == 2
        restored = serialized_table_from_json(doc)
        assert restored.evaluate('zzb', 1)[0][1] == 1
        mma = write_mathematica(table(size=2), coordinate='zzb', ell=0, c=8,
                                z_center=center, zbar_center=center_bar)
        assert 'zzbDeriv[0,1] ->' in mma


def test_equivalent_rational_centers_still_use_symmetric_layout():
    with mp.workdps(15):
        doc = json.loads(write_json(table(size=2), c=8, q_order=0,
                                    h_ext=(0,)*4, hbar_ext=(0,)*4, spin=0,
                                    z_center='.7', zbar_center='7/10'))
    assert len(doc['derivs'][0][0]) == 1


@pytest.mark.parametrize('family, c', [('global', None), ('virasoro', '8')])
def test_default_center_matches_explicit_center_in_point_tables(family, c):
    from virasoro_blocks.api import BlockRequest, NumericalPolicy, build_table_request

    with mp.workdps(40):
        tables = []
        for center in (None, '.5'):
            request = BlockRequest(operation='point-table', family=family, basis='scalar_G',
                                   central_charge=c, h_ext=('.13',)*4, hbar_ext=('.13',)*4,
                                   exchange_delta=2, exchange_spin=0, coordinates=('zzb',),
                                   center=center, center_bar='.4',
                                   policy=NumericalPolicy(precision=35, order=4, lambda_order=1))
            tables.append(build_table_request(request).evaluate('zzb', 2))
        assert abs(tables[1][0][1]) > mp.mpf('1e-5')
        assert abs(tables[0][0][1] - tables[1][0][1]) < mp.mpf('1e-25')


def test_large_odd_spin_keeps_its_derivative_layout_when_round_tripped():
    spin=2**100+1
    with mp.workdps(15):
        assert parity_smoothing_label('scalar_G', ell=spin)=='p_plus'
        assert kept_pairs('xt', 2, ell=spin)==kept_pairs('xt', 2, ell=1)==kept_pairs('xt', 2, ell=0)
        doc=json.loads(write_json(table('xt',3), c=8, q_order=0, h_ext=(0,)*4, hbar_ext=(0,)*4, spin=spin))
        restored=serialized_table_from_json(doc)
    assert doc['spin']==spin and doc['scalar_G_odd']=='even'
    assert doc['parity_smoothing']=='p_plus'
    assert restored.numerators['xt'][1][0].coeffs==[1]


@pytest.mark.parametrize('kind', ['sphere', 'torus', 'mathematica'])
def test_writer_retains_stored_precision_after_api_context_exits(kind):
    import re

    with mp.workdps(100):
        coord = 'vvb' if kind == 'torus' else 'zzb'
        source = table(coord)
        source.numerators[coord][0][0] = Polynomial([mp.mpf('1e50')+mp.pi, -mp.mpf('1e50')])
        source.working_dps = 100
    with mp.workdps(15):
        if kind == 'mathematica':
            text = write_mathematica(source, coordinate=coord, c=8)
        elif kind == 'torus':
            text = write_torus_json(source, c=8)
        else:
            text = write_json(source, c=8, h_ext=(0,)*4, hbar_ext=(0,)*4, q_order=0)
        assert mp.mp.dps == 15
    with mp.workdps(120):
        if kind == 'mathematica':
            assert 'precisionActual -> 100' in text
            coefficients = re.search(r'zzbDeriv\[0,0\] -> \{([^}]+)\}', text).group(1).split(', ')
            result = sum(mp.mpf(re.sub(r'`\d+', '', s).replace('*^', 'e')) for s in coefficients)
        else:
            doc = json.loads(text)
            assert doc['precision_actual'] == 100
            result = serialized_table_from_json(doc).evaluate(coord, 1)[0][0]
        assert abs(result-mp.pi) < mp.mpf('1e-48')
