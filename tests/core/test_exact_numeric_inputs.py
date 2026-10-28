"""Exact literals retain their identity through parsing and block dispatch."""
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
import json

import mpmath as mp
import pytest

from virasoro_blocks.api import BlockRequest, NumericalPolicy, evaluate_block_request
from virasoro_blocks.numeric_input import parse_number, literal_number
from virasoro_blocks.requests import RequestValidationError


@pytest.mark.parametrize('text', ['.7', '0.7', '+.7', '7e-1', '.70E+0', '7/10', ' .7 ', ' 7 / 10 ', '.14/.2'])
def test_real_literal_is_exact_at_any_precision(text):
    with mp.workdps(15):
        value = parse_number(text)
    assert isinstance(value, Fraction)
    assert value == Fraction(7, 10)
    with mp.workdps(100):
        assert mp.convert(value) == mp.mpf(7) / 10
        assert mp.mpf(value) == mp.mpf(7) / 10
        assert (value + parse_number('.1')) / 2 == Fraction(2, 5)


@pytest.mark.parametrize('value', ['', '.', 'nan', 'inf', '-inf', '1/0', '1/2/3', '7/abc', 'nan+1j', 'garbagej', Decimal('NaN'), float('nan'), float('inf'),
                                   mp.mpf('nan'), complex(float('inf'), 0)])
def test_bad_literals_raise_request_validation_errors(value):
    with pytest.raises(RequestValidationError):
        BlockRequest(operation='evaluate', central_charge=value)


def test_decimal_and_fraction_never_pass_through_a_float():
    value = Decimal('.70000000000000000000000000000000000000000000000001')
    with mp.workdps(15):
        req = BlockRequest(operation='evaluate', central_charge=value)
    assert req.central_charge == Fraction(value)
    assert req.central_charge != Fraction(7, 10)
    import numpy as np
    import pickle
    import warnings
    from virasoro_blocks.numeric_input import ExactRational
    from virasoro_blocks.evaluation import _resolve_external_h

    largest = 2**63-1
    native = np.int64(largest)
    source = Fraction(native)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        for supplied, expected in (
                (native, Fraction(largest)),
                (source, Fraction(largest)),
                (ExactRational(native), Fraction(largest)),
                (ExactRational(source), Fraction(largest)),
                (ExactRational(native, np.int64(-2)), Fraction(largest, -2)),
                (ExactRational(np.int64(1), np.int64(-2**63)), Fraction(1, -2**63)),
                (ExactRational(np.int64(-2**63), np.int64(-1)), Fraction(2**63))):
            parsed = parse_number(supplied)
            assert type(parsed.numerator) is type(parsed.denominator) is int
            assert parsed == expected
            assert pickle.loads(pickle.dumps(parsed)) == expected
        exact = parse_number(source)
        for one in (np.int64(1), Fraction(np.int64(1))):
            assert exact+one == one+exact == Fraction(largest+1)
            assert exact-one == Fraction(largest-1)
            assert one-exact == Fraction(1-largest)
        for two in (np.int64(2), Fraction(np.int64(2))):
            assert exact*two == two*exact == Fraction(2*largest)
            assert exact/two == Fraction(largest, 2)
            assert two/exact == Fraction(2, largest)
            assert exact**two == Fraction(largest**2)
            assert two**parse_number(2) == Fraction(4)
        request = BlockRequest(operation='evaluate', family='global',
            delta_ext=(source,)*4, spin_ext=(1,)*4)
        h, hbar = _resolve_external_h(request, chiral=False)
        assert h == (Fraction(largest+1, 2),)*4
        assert hbar == (Fraction(largest-1, 2),)*4


def test_native_numpy_real_labels_preserve_ratios_through_arithmetic():
    import numpy as np
    from virasoro_blocks.numeric_input import _label_scale, _label_sum

    # Leave booleans recognizable to the downstream input validators.
    assert parse_number(True) is True
    assert parse_number(False) is False
    with mp.workdps(5):
        for dtype in (np.float16, np.float32, np.longdouble):
            value = dtype('.7')
            expected = Fraction(*value.as_integer_ratio())
            assert isinstance(parse_number(value), Fraction)
            assert parse_number(value) == expected
            assert _label_scale(value, Fraction(1, 2)) == expected / 2
            shift = Fraction(1, 10**70)
            assert _label_sum(value, -expected + shift) == shift
        # This scalar follows Python float's established decimal convention.
        assert parse_number(np.float64('.7')) == Fraction(7, 10)
        assert mp.mp.dps == 5


def test_nonzero_complex_charge_literals_keep_their_intent_through_transfer():
    import copy
    import pickle
    from virasoro_blocks.params import b_from_c, c_origin_of_b
    from virasoro_blocks.singularity import exact_locus_c, positive_rational_b2_pq, recognized_singular_locus

    with mp.workdps(30):
        for center in ('.5', '1', '25', '28'):
            charge = parse_number(center + '+1e-70j')
            for value in (charge, copy.copy(charge), copy.deepcopy(charge), pickle.loads(pickle.dumps(charge))):
                assert value._mpc_ == charge._mpc_
                assert recognized_singular_locus(value) is None
                assert exact_locus_c(value) is None
            parameter = pickle.loads(pickle.dumps(b_from_c(charge)))
            assert recognized_singular_locus(c_origin_of_b(parameter)) is None
            # Lower-level native numerical inputs retain their historical
            # roundoff recognition; constructing one explicitly drops intent.
            assert recognized_singular_locus(mp.mpc(center, '1e-70')) is not None
        assert positive_rational_b2_pq('28+1e-70j') is None
        assert positive_rational_b2_pq('28+1e-70j', tolerance='.01') == (1, 2)


@pytest.mark.parametrize('family,center', [('virasoro', '.5'), ('ns', '1.5'), ('ramond', '1.5')])
def test_complex_charge_request_validation_preserves_supplied_bits(family, center):
    from virasoro_blocks.requests import validate_request
    from virasoro_blocks.superconformal_blocks import is_superconformal_singular_c, resolve_super_c

    with mp.workdps(100):
        native = mp.mpc(center, '1e-70')
    with mp.workdps(15):
        for supplied in (center + '+1e-70j', native):
            request = BlockRequest(operation='evaluate', family=family, central_charge=supplied,
                policy=NumericalPolicy(precision=30, order=2, c_regularization='off'))
            validate_request(request)
            if supplied is native:
                assert request.central_charge._mpc_ == native._mpc_
            if family != 'virasoro':
                assert not is_superconformal_singular_c(request.central_charge)
                assert not resolve_super_c(request.central_charge)[2]
            with pytest.raises(RequestValidationError, match='singular'):
                validate_request(replace(request, central_charge=center))


def test_zero_imaginary_native_charge_uses_the_same_label_as_its_real_component():
    from virasoro_blocks.requests import validate_request

    with mp.workdps(100):
        charge = mp.mpc('.70000000000000000000000000000000000000000000000000000000000000000000001', 0)
        expected = literal_number(charge.real)
    with mp.workdps(15):
        request = BlockRequest(operation='evaluate', central_charge=charge,
            policy=NumericalPolicy(precision=30, order=2, c_regularization='off'))
        assert isinstance(request.central_charge, Fraction)
        assert request.central_charge == expected
        validate_request(request)
        with pytest.raises(RequestValidationError, match='singular'):
            validate_request(replace(request, central_charge='.7+0j'))


@pytest.mark.parametrize('family', ['ns', 'ramond'])
def test_complex_super_charge_does_not_take_a_real_offset(family):
    from virasoro_blocks.superconformal_blocks import ns_chiral_block, ramond_chiral_block

    external = ('.1', '.2', '.3', '.4')
    with mp.workdps(220):
        reference = (ns_chiral_block if family == 'ns' else ramond_chiral_block)(
            '.3', '.3', external, c=mp.mpc('1.5', '1e-70'), q_order=2, regularize='off')
    with mp.workdps(15):
        for policy in ('auto', 'off'):
            request = BlockRequest(operation='evaluate', family=family, basis=family+'_chiral_F',
                coordinate='z', central_charge='1.5+1e-70j', h_ext=external, exchange_h='.3', point={'z': '.3'},
                policy=NumericalPolicy(precision=30, order=2, c_regularization=policy))
            result = evaluate_block_request(request, q_gate='off')
            assert result.metadata['regularization_policy'] == 'direct'
            with mp.workdps(35):
                assert abs(result.value/reference - 1) < mp.mpf('1e-28')


def test_float_uses_its_decimal_spelling_and_mpf_keeps_its_supplied_value():
    assert literal_number(0.7) == Fraction(7, 10)
    assert literal_number(0.7000000000000001) == Fraction("0.7000000000000001")
    assert literal_number(0.7000000000000001) != Fraction(7, 10)
    with mp.workdps(30):
        value = mp.mpf('.7')
        req = BlockRequest(operation='evaluate', central_charge=value)
    assert req.central_charge != Fraction(7, 10)
    with mp.workdps(100):
        assert mp.convert(req.central_charge) == value


@pytest.mark.parametrize('family', ['virasoro', 'ns'])
def test_exact_locus_detection_cannot_erase_a_decimal_offset(family):
    from virasoro_blocks.singularity import exact_locus_c, recognized_singular_locus
    from virasoro_blocks.super_laurent import exact_singular_c
    from virasoro_blocks.superconformal_blocks import is_superconformal_singular_c
    detect = exact_locus_c if family == 'virasoro' else exact_singular_c
    dispatch = recognized_singular_locus if family == 'virasoro' else is_superconformal_singular_c
    with mp.workdps(15):
        c = parse_number('.7')
        shifted = c + Fraction(1, 10**100)
        assert detect(c) is not None
        assert dispatch(c)
        assert detect(shifted) is None
        assert not dispatch(shifted)


def request(family='virasoro', **kwargs):
    return BlockRequest(operation='evaluate', family=family, central_charge='1',
                        coordinate='z', exchange_h='.25', h_ext=('.0625',)*4,
                        point={'z':'.3'}, policy=NumericalPolicy(precision=35, order=8), **kwargs)


@pytest.mark.parametrize('offset_digits', [25, 80])
def test_explicit_near_kac_weight_is_the_continuous_block(offset_digits):
    from virasoro_blocks.recursion import block_prefactor
    on = request()
    near = replace(on, exchange_h=Fraction(1, 4) + Fraction(1, 10**offset_digits))
    got = evaluate_block_request(near, q_gate='off')
    exact = evaluate_block_request(on, q_gate='off')
    with mp.workdps(130):
        pref = block_prefactor(mp.convert(near.exchange_h), mp.mpf('.3'), [mp.mpf(1)/16]*4, c=1)
        assert abs(got.value / pref - 1) < mp.mpf('1e-32')
        assert abs(exact.value / pref - 1) > mp.mpf('1e-6')
    json.dumps(got.as_dict())


@pytest.mark.parametrize('family', ['virasoro', 'ns', 'ramond'])
def test_decimal_api_value_matches_fraction_input(family):
    r = replace(request(family), central_charge='.7', exchange_h='.23', h_ext=('.1',)*4)
    decimal = evaluate_block_request(r, q_gate='off')
    rational = evaluate_block_request(replace(r, central_charge=Fraction(7, 10), exchange_h=Fraction(23, 100),
                                             h_ext=(Fraction(1, 10),)*4), q_gate='off')
    assert decimal.value == rational.value
    json.dumps(decimal.as_dict())


@pytest.mark.parametrize("offset_digits", [25, 80])
def test_point_table_preserves_a_shift_that_old_kac_tolerance_swallowed(offset_digits):
    from virasoro_blocks.api import build_table_request
    r = replace(request(), operation='point-table', point={}, coordinates=('z',), basis='chiral_F',
                exchange_h=Fraction(1, 4)+Fraction(1, 10**offset_digits),
                policy=NumericalPolicy(precision=35, order=8, lambda_order=0))
    table = build_table_request(r)
    # Compare its value at the frozen point with the direct continuous block.
    with mp.workdps(50):
        value = table.evaluate("z", mp.convert(r.exchange_h))[0][0]
    ref = evaluate_block_request(replace(r, operation='evaluate', coordinates=(), point={'z':'.5'}), q_gate='off')
    with mp.workdps(50):
        assert abs(value / ref.value - 1) < mp.mpf('1e-30')


def test_exact_numeric_arithmetic_does_not_demote_to_python_double():
    x = parse_number('.7')
    assert x + 0.1 == Fraction(4, 5)
    assert 0.1 + x == Fraction(4, 5)
    assert x * 0.1 == Fraction(7, 100)
    assert x / 0.1 == 7
    assert x ** -2 == Fraction(100, 49)
    with mp.workdps(80):
        assert x ** parse_number('.5') == mp.sqrt(mp.mpf(7) / 10)
        assert 2 ** x == mp.power(2, mp.mpf(7) / 10)


def test_torus_character_distinguishes_exact_and_nearby_weights():
    from virasoro_blocks.torus_onept import irreducible_character_factor
    with mp.workdps(30):
        q = mp.mpf('.01')
        _, exact = irreducible_character_factor(parse_number('1'), parse_number('.25'), q)
        factor, near = irreducible_character_factor(parse_number('1'), parse_number('.25') + Fraction(1,10**80), q)
        assert exact and not near
        assert factor == 1


def test_float_and_leading_dot_cli_parsers_agree():
    from virasoro_blocks.__main__ import _input_mpf
    from virasoro_blocks.cli.point_table import _mp_number
    from virasoro_blocks.singularity import _SNAP_WARNED, snap_float_c, recognized_singular_locus
    _SNAP_WARNED.clear()
    with mp.workdps(15):
        assert _input_mpf('.7') == _mp_number('.7') == snap_float_c(0.7) == Fraction(7,10)
        # (the next double up is within float rounding of 7/10 too: it is the locus, with a notice)
        with pytest.warns(UserWarning):
            assert recognized_singular_locus(snap_float_c(0.7000000000000001)) is not None


@pytest.mark.parametrize('offset_digits', [25, 80])
def test_full_scalar_block_keeps_the_derived_near_kac_weights(offset_digits):
    from virasoro_blocks.recursion import block_prefactor
    h = parse_number('.25') + Fraction(1, 10**offset_digits)
    r = replace(request(), coordinate='zzb', basis='scalar_G', exchange_h=None,
                exchange_delta=2*h, exchange_spin=0, h_ext=None, delta_ext=('.125',)*4,
                point={'z':'.3', 'zbar':'.4'})
    got = evaluate_block_request(r, q_gate='off')
    with mp.workdps(130):
        ext = [mp.mpf(1)/16]*4
        expected = block_prefactor(mp.convert(h), mp.mpf('.3'), ext, c=1) * block_prefactor(
            mp.convert(h), mp.mpf('.4'), ext, c=1)
        assert abs(got.value / expected - 1) < mp.mpf('1e-32')


def test_half_integer_spin_and_large_integer_labels_remain_exact():
    from virasoro_blocks.evaluation import _resolve_exchange
    r = BlockRequest(operation='evaluate', exchange_delta='1.000000000000000000000000000001',
                     exchange_spin='.5')
    h, hb = _resolve_exchange(r, chiral=False)
    assert isinstance(h, Fraction) and isinstance(hb, Fraction)
    assert h + hb == r.exchange_delta
    assert h - hb == Fraction(1, 2)
    r = replace(r, exchange_delta=2**100+1, exchange_spin=0)
    h, hb = _resolve_exchange(r, chiral=False)
    assert h == hb == Fraction(2**100+1, 2)


def test_bare_policy_does_not_reject_an_explicitly_shifted_c_at_low_ambient_precision():
    from virasoro_blocks.requests import validate_request
    r = replace(request(), central_charge=parse_number('.7')+Fraction(1,10**80),
                policy=NumericalPolicy(precision=35, order=8, c_regularization='off'))
    with mp.workdps(15):
        validate_request(r)
        with pytest.raises(RequestValidationError, match='singular'):
            validate_request(replace(r, central_charge='.7'))


def test_exact_external_asymmetry_and_noninteger_spin_are_not_erased():
    from virasoro_blocks.evaluation import _canonical_basis, _scalar_G_spin
    shifted = parse_number('.125') + Fraction(1,10**80)
    r = BlockRequest(operation='evaluate', h_ext=('.125',)*4, hbar_ext=(shifted,)*4)
    with mp.workdps(15):
        assert _canonical_basis(r) == 'ordered_B'
        with pytest.raises(RequestValidationError, match='integer spin'):
            _scalar_G_spin(r, parse_number('1')+Fraction(1,10**80), parse_number('0'))


def test_explicit_near_global_pole_is_evaluated_at_the_requested_weight():
    from virasoro_blocks.derivs import _global_chiral_block
    r = replace(request('global'), central_charge=None, exchange_h=Fraction(1,10**30),
                h_ext=('.1','.2','.3','.4'), policy=NumericalPolicy(precision=35, order=None))
    got = evaluate_block_request(r, q_gate='off')
    with mp.workdps(100):
        ref = _global_chiral_block(mp.convert(r.exchange_h), mp.mpf('.3'), [mp.convert(h) for h in r.h_ext])
        assert abs(got.value/ref-1) < mp.mpf('1e-32')


def test_explicit_super_offset_starts_from_the_parsed_float():
    from virasoro_blocks.superconformal_blocks import resolve_super_c
    with mp.workdps(30):
        original, shifted, active = resolve_super_c(8.3, regularize='on', eps_exp=50)
        assert original == Fraction(83, 10) and active
    with mp.workdps(100):
        assert abs(shifted - (mp.mpf('8.3') + mp.mpf('1e-50'))) < mp.mpf('1e-88')


def test_explicit_spin_offset_is_not_rounded_to_a_half_integer():
    from virasoro_blocks.evaluation import _resolve_exchange
    from virasoro_blocks.api import build_table_request
    r = BlockRequest(operation='evaluate', exchange_delta='2',
                     exchange_spin=Fraction(1,2)+Fraction(1,10**80))
    with mp.workdps(15):
        with pytest.raises(RequestValidationError, match='half-integer'):
            _resolve_exchange(r, chiral=False)
        with pytest.raises(RequestValidationError, match='integer exchanged spin'):
            build_table_request(replace(r, operation='point-table', basis='scalar_G',
                                        exchange_spin=Fraction(1)+Fraction(1,10**80)))


@pytest.mark.parametrize('c', ['-2', '.5', '.7', '1', '25', '26', '33', '50'])
def test_exact_virasoro_parameter_matches_the_numerical_branch(c):
    from virasoro_blocks.numeric_input import exact_b_squared
    from virasoro_blocks.params import b_from_c
    with mp.workdps(80):
        x = exact_b_squared(parse_number(c))
        assert x is not None and abs(x) <= 1
        assert abs(mp.convert(x) - b_from_c(c)**2) < mp.mpf('1e-75')


@pytest.mark.parametrize('c', ['.7', '81/70', '1.5', '13.5', '15'])
def test_exact_super_parameter_matches_the_numerical_branch(c):
    from virasoro_blocks.numeric_input import exact_b_squared
    from virasoro_blocks.recursion_ns import beta_from_c
    with mp.workdps(80):
        value = parse_number(c)
        x = exact_b_squared(value, superconformal=True)
        assert x is not None and abs(x) >= 1
        assert abs(mp.convert(x) - beta_from_c(mp.convert(value))**2) < mp.mpf('1e-75')


@pytest.mark.parametrize("number_type", ["rational", "mpf", "mpc"])
def test_derivative_evaluator_cache_keeps_distinct_exact_inputs(monkeypatch, number_type):
    import virasoro_blocks.derivs as derivs
    from virasoro_blocks.singularity import merged_singular_evaluators

    monkeypatch.setattr(derivs, '_CHIRAL_EVAL_MEMO', {})
    builds = []

    def build(externals, *args, **kwargs):
        builds.append(tuple(externals))
        return lambda h: tuple(externals)

    monkeypatch.setattr(derivs, '_chiral_series_evaluator_uncached', build)
    external = parse_number('.0625')
    shifted = external + Fraction(1, 10**80)
    with mp.workdps(100):
        if number_type == "mpf":
            external, shifted = mp.mpf(external), mp.mpf(shifted)
        elif number_type == "mpc":
            external, shifted = mp.mpc(external, '.125'), mp.mpc(shifted, '.125')
    with mp.workdps(15), merged_singular_evaluators():
        on = derivs._chiral_series_evaluator((external,) * 4, 1, c=1, q_order=4)
        near = derivs._chiral_series_evaluator((shifted,) * 4, 1, c=1, q_order=4)
        again = derivs._chiral_series_evaluator((external,) * 4, 1, c=1, q_order=4)
    assert on(0) == (external,) * 4
    assert near(0) == (shifted,) * 4
    assert again is on
    assert len(builds) == 2


@pytest.mark.parametrize("number_type", ["rational", "mpf", "mpc"])
def test_geometry_cache_keeps_distinct_stored_inputs(monkeypatch, number_type):
    from collections import OrderedDict
    import virasoro_blocks.derivs as derivs

    monkeypatch.setattr(derivs, "_GEOMETRY_CACHE", OrderedDict())
    builds = []
    def build(n_terms, sigma, exp_z, *args, **kwargs):
        builds.append(exp_z)
        return [mp.mpf(1)], [exp_z]
    monkeypatch.setattr(derivs, "_flint_geometry_series_coeffs", build)
    first = parse_number(".125")
    second = first + Fraction(1, 10**80)
    with mp.workdps(100):
        if number_type == "mpf":
            first, second = mp.mpf(first), mp.mpf(second)
        elif number_type == "mpc":
            first, second = mp.mpc(first, ".125"), mp.mpc(second, ".125")
    def geometry(value):
        return derivs._chiral_geometry_series_coeffs((value,) * 4, 1, 1, 0, 0, value, 0, 0)
    with mp.workdps(15):
        assert geometry(first)[1] == [first]
        assert geometry(second)[1] == [second]
        assert geometry(first)[1] == [first]
    assert len(builds) == 2
