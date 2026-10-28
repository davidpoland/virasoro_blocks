"""Modular builds validate numerical controls before constructing tables."""
from dataclasses import replace
from fractions import Fraction
import numpy as np
import pytest
from virasoro_bootstrap import modular_crossing as modular


def test_modular_specs_capture_single_pass_inputs_for_direct_and_prepared_reuse():
    params = dict(c=4, delta_gap=1, lam=1, dps=30)
    expected_spec = modular.ModularGapSpec(**params, spin_list=(0, 1),
        isolated_scalars=(('.7', 2),), allowed_isolated_scalars=('.9',))
    expected = modular.build_modular_gap_problem(expected_spec)
    for prepare in (False, True):
        spec = modular.ModularGapSpec(**params, spin_list=iter((1, 0)),
            isolated_scalars=iter((('.7', 2),)), allowed_isolated_scalars=iter(('.9',)))
        prepared = modular.prepare_modular_gap_build(spec) if prepare else None
        for _ in range(2):
            actual = modular.build_modular_gap_problem(spec, prepared=prepared)
            assert actual.pmp_doc == expected.pmp_doc
            assert actual.spec.spin_list == (0, 1)
            assert actual.spec.isolated_scalars == ((Fraction(7, 10), 2),)
            assert actual.spec.allowed_isolated_scalars == (Fraction(9, 10),)
            # spins 0 and 1, the spin-1 current (its edge is the unitarity bound), and the allowed scalar
            assert len(actual.spin_channels) == 4


# The integer type matrix is tested at _integer_field in test_request_contract_audit.
# Both public routes must still reject fractions and their own lower bounds.
@pytest.mark.parametrize('builder, field, value', [
    (modular.prepare_modular_gap_build, 'lam', 2.5), (modular.build_modular_gap_problem, 'n_sample_points', 0),
])
def test_invalid_controls_fail_before_numerical_work(monkeypatch, builder, field, value):
    spec = replace(modular.ModularGapSpec(c=4, delta_gap=1, lam=1, dps=30,
                                         spin_list=(0,), n_sample_points=2), **{field: value})
    def unexpected(*args, **kwargs):
        pytest.fail('invalid control reached numerical table construction')
    monkeypatch.setattr(modular, 'torus_vacuum_table', unexpected)
    with pytest.raises(ValueError, match=field):
        builder(spec)


@pytest.mark.parametrize('constructor', [np.int64])
def test_integral_controls_match_integer_build_and_reuse(constructor):
    spec = modular.ModularGapSpec(c=4, delta_gap=1, lam=1, dps=30,
                                 spin_list=(0,), n_sample_points=2)
    prepared = modular.prepare_modular_gap_build(spec)
    expected = modular.build_modular_gap_problem(spec, prepared=prepared)
    equivalent = replace(spec, lam=constructor(1), dps=constructor(30),
                         n_sample_points=constructor(2))
    actual = modular.build_modular_gap_problem(equivalent, prepared=prepared)
    fresh = modular.build_modular_gap_problem(equivalent)
    assert actual.pmp_doc == fresh.pmp_doc == expected.pmp_doc


@pytest.mark.parametrize('value', [1.5])
def test_standalone_S_odd_pairs_validates_order(value):
    with pytest.raises(ValueError, match='lam'):
        modular.S_odd_pairs(value)


@pytest.mark.parametrize('field', ['dps'])
def test_prepared_reuse_does_not_bypass_control_validation(field):
    spec = modular.ModularGapSpec(c=4, delta_gap=1, lam=1, dps=30, spin_list=(0,))
    prepared = modular.prepare_modular_gap_build(spec)
    with pytest.raises(ValueError, match=field):
        modular.build_modular_gap_problem(replace(spec, **{field: 1.5}), prepared=prepared)


def test_exact_modular_domains_are_validated_before_table_construction(monkeypatch):
    tiny = Fraction(1, 10**70)
    spec = modular.ModularGapSpec(c=4, delta_gap=0, lam=1, dps=30, spin_list=(0, 1))
    def unexpected(*args, **kwargs):
        pytest.fail('invalid domain reached numerical table construction')
    monkeypatch.setattr(modular, 'torus_vacuum_table', unexpected)
    for changed, message in [
        (replace(spec, delta_min_by_spin={1: 1-tiny}), 'unitarity'),
        (replace(spec, c=1-tiny), 'c > 1'),
        (replace(spec, isolated_scalars=((0, 1),)), 'positive'),
        (replace(spec, allowed_isolated_scalars=(0,)), 'positive'),
    ]:
        with pytest.raises(ValueError, match=message):
            modular.build_modular_gap_problem(changed)


def test_modular_continuum_and_isolated_channels_keep_exact_labels():
    from virasoro_blocks.numeric_input import literal_number

    tiny = Fraction(1, 10**70)
    gap, override, point = Fraction(7, 10)+tiny, 1+tiny, Fraction(1, 3)+tiny
    spec = modular.ModularGapSpec(c=4, delta_gap=gap, lam=1, dps=30,
        spin_list=(0, 1), delta_min_by_spin={1: override}, allowed_isolated_scalars=(point,))
    problem = modular.build_modular_gap_problem(spec)
    for channel, label in zip(problem.spin_channels, (gap, override, point)):
        assert literal_number(channel.delta_min) == literal_number(channel.block.shift) == label
    delta, degeneracy = modular._parse_isolated_scalar((point, 1+tiny))
    assert literal_number(delta) == point
    assert literal_number(degeneracy) == 1+tiny


def test_character_backend_validates_controls_before_building(monkeypatch):
    from virasoro_blocks import modular as characters
    from virasoro_blocks.table_adapters import _torus_spin
    from virasoro_blocks.numeric_input import ExactRational

    def unexpected(*args, **kwargs):
        pytest.fail('invalid control reached character polynomial construction')
    monkeypatch.setattr(characters, '_chiral_coeff_polys', unexpected)
    # The full integer-type matrix lives at _integer_field. These public
    # callers must enforce their own integer and lower-bound contracts.
    for field, value in [('spin', 1.5), ('lam', -1)]:
        params = dict(c=4, spin=0, lam=1)
        params[field] = value
        with pytest.raises(ValueError, match=field):
            characters.torus_character_table(**params)
    with pytest.raises(ValueError, match='lam'):
        characters.torus_vacuum_table(4, 1.5)
    with pytest.raises(ValueError, match='lam'):
        characters.tau2_power_table(1.5, '.5')
    with pytest.raises(ValueError, match='integer'):
        _torus_spin(ExactRational(1)+ExactRational(1, 10**70))
