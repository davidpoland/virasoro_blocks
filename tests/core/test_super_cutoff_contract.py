"""Super-block cutoffs are discrete controls, never truncating casts."""

from decimal import Decimal
from fractions import Fraction

import mpmath as mp
import numpy as np
import pytest

from virasoro_blocks import recursion_ns as ns, recursion_r as ramond
from virasoro_blocks import superconformal_blocks as sb
from virasoro_blocks.numeric_input import parse_number
from virasoro_blocks.super_kac import _super_cutoff


BAD = [True, np.bool_(True), 2.5, Fraction(5, 2), Decimal('2.5'), '2',
       mp.nan, mp.inf, -1]
EXT = ['.1', '.2', '.3', '.4']


def _laurent_factory(kind, *, family='ns', order=2, **kwargs):
    from virasoro_blocks import super_laurent as laurent

    ext = kwargs.pop('h_ext', tuple(map(parse_number, EXT)))
    if kind == 'table':
        return laurent.super_laurent_state(family, Fraction(3, 2), ext, order, **kwargs)
    if kind == 'merged':
        return laurent.merged_state(family, ext, c0=Fraction(3, 2), order=order, **kwargs)
    return laurent.singular_H(family, Fraction(13, 10), ext, c0=Fraction(3, 2), order=order, **kwargs)


@pytest.mark.parametrize('kind', ['table', 'merged', 'values'])
def test_direct_laurent_cutoffs_validate_before_cache_lookup(kind, monkeypatch):
    from virasoro_blocks import super_laurent as laurent

    with mp.workdps(30):
        for order in (1, 2):
            _laurent_factory(kind, order=order)
        def unexpected(*args, **kwargs):
            pytest.fail('invalid cutoff reached Kac work')
        monkeypatch.setattr(laurent, 'exact_beta_squared', unexpected)
        for bad in (2.5, True):
            with pytest.raises(ValueError, match='order'):
                _laurent_factory(kind, order=bad)


@pytest.mark.parametrize('kind', ['table', 'merged', 'values'])
def test_direct_laurent_unknown_family_does_not_select_ramond(kind, monkeypatch):
    from virasoro_blocks import super_laurent as laurent

    def unexpected(*args, **kwargs):
        pytest.fail('invalid family reached Kac work')
    monkeypatch.setattr(laurent, 'exact_beta_squared', unexpected)
    with pytest.raises(ValueError, match='family'):
        _laurent_factory(kind, family='typo')


def test_direct_laurent_window_controls_cannot_hide_behind_a_cache():
    with mp.workdps(30):
        _laurent_factory('table')
        for bad in (1.5, True, -1):
            with pytest.raises(ValueError, match='half'):
                _laurent_factory('table', half=bad)


@pytest.mark.parametrize('kind', ['merged', 'values'])
def test_direct_laurent_sector_and_branch_controls_are_exact(kind):
    with mp.workdps(30):
        _laurent_factory(kind)
        for name, bad in (('sector', .5), ('branch', .5), ('sector', True), ('branch', True),
                          ('sector', 2), ('branch', 4)):
            with pytest.raises(ValueError, match=name):
                _laurent_factory(kind, **{name: bad})


def test_direct_laurent_aliases_and_integral_orders_keep_the_same_blocks():
    with mp.workdps(30):
        for alias, canonical in (('NS', 'ns'), ('R', 'ramond'), ('r', 'ramond')):
            expected = _laurent_factory('values', family=canonical, order=4)
            assert _laurent_factory('values', family=alias, order=4.0) == expected


def test_direct_laurent_builders_require_exactly_four_external_weights(monkeypatch):
    from virasoro_blocks import super_laurent as laurent

    def unexpected(*args, **kwargs):
        pytest.fail('invalid external list reached Kac work')
    monkeypatch.setattr(laurent, 'exact_beta_squared', unexpected)
    for kind in ('table', 'merged', 'values'):
        for length in (3, 5):
            with pytest.raises(ValueError, match='length-4'):
                _laurent_factory(kind, h_ext=[Fraction(1, 9)] * length)


# The shared validator gets the full type matrix; callers below check that
# validation runs before caches, pole work, or geometry construction.
@pytest.mark.parametrize('bad', BAD)
def test_super_cutoff_validator_rejects_invalid_types_and_bounds(bad):
    with pytest.raises(ValueError, match='order'):
        _super_cutoff('order', bad)


@pytest.mark.parametrize('bad', [2.5, -1])
@pytest.mark.parametrize('family', ['ns', 'ramond'])
@pytest.mark.parametrize('units', ['q_order', 'half_steps'])
def test_recursion_rejects_invalid_cutoffs_before_pole_work(monkeypatch, bad, family, units):
    def unexpected(*args, **kwargs):
        pytest.fail('invalid cutoff reached pole work')
    monkeypatch.setattr(sb, '_chiral_poles', unexpected)
    fn = ns.H_q_series_ns if family == 'ns' else ramond.H_q_series_r
    name = 'q_order' if units == 'q_order' else ('ns_order' if family == 'ns' else 'r_order')
    with pytest.raises(ValueError, match=name):
        fn('.6', EXT, c=30, **{name: bad})


@pytest.mark.parametrize('bad', [2.5, -1])
@pytest.mark.parametrize('factory', [ns.build_ns_level_table, ns.ns_level_table,
                                     ramond.build_r_level_table, ramond.r_level_table])
def test_level_table_cutoffs_cannot_hide_behind_a_cache(monkeypatch, bad, factory):
    def unexpected(*args, **kwargs):
        pytest.fail('invalid cutoff reached table construction')
    monkeypatch.setattr(ns._poles, 'fill_level_table_flint', unexpected)
    with pytest.raises(ValueError, match='order'):
        factory(30, EXT, bad)


@pytest.mark.parametrize('bad', [2.5, -1])
def test_half_step_conversion_and_enumeration_validate_exactly(bad):
    with pytest.raises(ValueError, match='q_order'):
        ns.ns_order_for_q_order(bad)
    with pytest.raises(ValueError, match='ns_order'):
        list(ns.enumerate_ns_poles(bad))
    with pytest.raises(ValueError, match='order'):
        ns.theta_half_series(3, bad)


@pytest.mark.parametrize('bad', [2.5, -1])
@pytest.mark.parametrize('field', ['q_order', 'm_max', 'n_max'])
@pytest.mark.parametrize('family', ['ns', 'ramond'])
def test_numeric_full_table_rejects_invalid_controls_before_pole_checks(monkeypatch, bad, field, family):
    def unexpected(*args, **kwargs):
        pytest.fail('invalid cutoff reached a pole check')
    monkeypatch.setattr(sb, '_chiral_poles', unexpected)
    options = dict(q_order=2, m_max=1, n_max=1)
    options[field] = bad
    with pytest.raises(ValueError, match=field):
        sb.numeric_full_table(family, 'ordered_B', '2.6', 0, 'zzb', EXT, c=30, **options)


@pytest.mark.parametrize('bad', [2.5])
@pytest.mark.parametrize('chiral', [False, True])
def test_interpolation_sampler_does_not_truncate_controls(bad, chiral):
    options = dict(c=30, q_order=bad, m_max=1, chiral_norm=None, z_shift=None,
                   sector=0, branch='++', center=None)
    with pytest.raises(ValueError, match='q_order'):
        if chiral:
            sb.SuperChiralSampler('ns', 'z', EXT, **options)
        else:
            sb.SuperBlockSampler2d('ns', 'zzb', 0, EXT, EXT, n_max=1,
                                  sigma=1, fold_parity=None, zbar_shift=None,
                                  center_bar=None, **options)


@pytest.mark.parametrize('bad', [2.5])
@pytest.mark.parametrize('chiral', [False, True])
def test_interpolation_builder_validates_before_constructing_geometry(monkeypatch, bad, chiral):
    def unexpected(*args, **kwargs):
        pytest.fail('invalid cutoff reached nome geometry')
    monkeypatch.setattr(sb, '_half_exp_base', unexpected)
    with pytest.raises(ValueError, match='m_max'):
        if chiral:
            sb.super_chiral_F_interpolated_table('ns', 'z', EXT, c=30, m_max=bad, q_order=2)
        else:
            sb.super_2d_interpolated_table('ns', 'ordered_B', 'zzb', 0, EXT,
                                         c=30, m_max=bad, n_max=1, q_order=2)


def test_folded_sampler_uses_normalized_integral_derivative_cutoffs():
    sampler = sb.SuperBlockSampler2d('ns', 'xt', 0, EXT, EXT, c=30, q_order=2.0,
                                    m_max=1.0, n_max=1.0, sigma=1, fold_parity=0,
                                    chiral_norm=None, z_shift=None, zbar_shift=None,
                                    sector=0, branch='++', center=None, center_bar=None)
    assert type(sampler.src_m) is int and type(sampler.src_n) is int
    assert len(sampler('2.6')) == 2


@pytest.mark.parametrize('order', [0, 2, 2.0, Fraction(2), Decimal('2'), mp.mpf(2), np.int64(2)])
def test_super_cutoff_normalizes_all_supported_integral_types(order):
    result = _super_cutoff('order', order)
    assert result == int(order) and type(result) is int


@pytest.mark.parametrize('order', [0, 2.0])
@pytest.mark.parametrize('family', ['ns', 'ramond'])
def test_integral_cutoffs_keep_the_same_series(order, family):
    fn = ns.H_q_series_ns if family == 'ns' else ramond.H_q_series_r
    with mp.workdps(30):
        assert fn('.6', EXT, c=30, q_order=order) == fn('.6', EXT, c=30, q_order=int(order))
    assert ns.ns_order_for_q_order(order) == 2*int(order)
    assert list(ns.enumerate_ns_poles(order)) == list(ns.enumerate_ns_poles(int(order)))


@pytest.mark.parametrize('family', ['ns', 'ramond'])
def test_a_stored_subprecision_cutoff_offset_is_not_an_integer(family):
    with mp.workdps(100):
        bad = mp.mpf(2)+mp.mpf('1e-70')
    fn = ns.H_q_series_ns if family == 'ns' else ramond.H_q_series_r
    with mp.workdps(15), pytest.raises(ValueError, match='q_order'):
        fn('.6', EXT, c=30, q_order=bad)


def test_ns_table_cache_keeps_distinct_exact_charges_below_working_precision():
    first_c = parse_number(Fraction(30)+Fraction(1, 10**70))
    second_c = first_c+Fraction(1, 10**70)
    with mp.workdps(15):
        first = ns.ns_level_table(first_c, EXT, 2)
        second = ns.ns_level_table(second_c, EXT, 2)
        assert first is not second
        assert first.family.c == first_c and second.family.c == second_c
        assert ns.ns_level_table(first_c, EXT, 2) is first
        assert ns.ns_level_table(second_c, EXT, 2) is second
