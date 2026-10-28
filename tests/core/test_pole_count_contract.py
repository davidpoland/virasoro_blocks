"""Pole-count requests are integer caps, never silently rounded or boolean."""

from fractions import Fraction as F

import mpmath as mp
import numpy as np
import pytest

from virasoro_blocks.interpolation import DampedRational, _validate_npoles
from virasoro_blocks import global_blocks as gb, polyderivs as pd


BAD = [True, np.bool_(True), 1.5, F(3, 2), mp.nan, mp.inf, -1]


@pytest.mark.parametrize("count", BAD)
def test_pole_cap_validator_rejects_invalid_types_and_bounds(count):
    with pytest.raises(ValueError, match="npoles"):
        _validate_npoles(count)


# All consumers use the validator above. Retain boundary checks for each
# selector and builder without repeating every numeric representation.
@pytest.mark.parametrize("count", [True, 1.5, -1])
@pytest.mark.parametrize("method", ["take_rightmost_poles", "take_nearest_poles", "take_real_first_poles"])
def test_selectors_reject_invalid_caps(count, method):
    prefactor = DampedRational(1, (-1, -2, -3), mp.exp(-mp.pi))
    with pytest.raises(ValueError, match="npoles"):
        getattr(prefactor, method)(count)


@pytest.mark.parametrize("count", [True, 1.5, -1])
def test_virasoro_refuses_invalid_cap_before_touching_tables(monkeypatch, count):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid npoles reached numerical table processing")
    monkeypatch.setattr(pd, "_realify_conjugate_denominator", unexpected)
    with pytest.raises(ValueError, match="npoles"):
        pd.interpolate_delta_tables({"zzb": object()}, count)


@pytest.mark.parametrize("count", [True, 1.5, -1])
@pytest.mark.parametrize("chiral", [False, True])
def test_global_refuses_invalid_cap_before_numerical_work(monkeypatch, count, chiral):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid npoles reached numerical block processing")
    monkeypatch.setattr(gb, "global_block_exp_base", unexpected)
    with pytest.raises(ValueError, match="npoles"):
        if chiral:
            gb.global_chiral_F_interpolated_table("z", [.1] * 4, 0, npoles=count)
        else:
            gb.global_scalar_G_interpolated_table("zzb", 0, [.1] * 4, 0, npoles=count)


@pytest.mark.parametrize("count", [2, 2.0, F(2), mp.mpf(2), np.int64(2)])
def test_integral_numeric_caps_keep_requested_count(count):
    assert _validate_npoles(count) == 2
    assert type(_validate_npoles(count)) is int
    prefactor = DampedRational(1, (-1, -2, -3), mp.exp(-mp.pi))
    assert prefactor.take_rightmost_poles(count).poles == (mp.mpc(-1), mp.mpc(-2))


@pytest.mark.parametrize("count", [True, 1.5, -1])
def test_global_pole_list_rejects_invalid_count(count):
    with pytest.raises(ValueError, match="npoles"):
        gb.global_sl2_delta_poles(0, count)


def test_zero_cap_keeps_no_poles():
    prefactor = DampedRational(1, (-1, -2), mp.exp(-mp.pi))
    assert prefactor.take_rightmost_poles(0).poles == ()
    assert gb.global_sl2_delta_poles(0, 0) == []


@pytest.mark.parametrize('npoles', [None, 2])
def test_keep_all_does_not_drop_poles_after_node_search_failure(monkeypatch, npoles):
    from virasoro_blocks.polynomial import Polynomial

    exact = pd.DeltaDerivTable(spin=0, exp_base=mp.mpf('.5'),
        denominator=Polynomial([2, 3, 1]), denominator_roots=[-1, -2],
        numerators=[[Polynomial([1])]])
    attempted = []
    def nodes(prefactor, count, **kwargs):
        attempted.append(prefactor.pole_degree)
        if prefactor.pole_degree == 2:
            raise ValueError('node search did not converge')
        return [mp.mpf(i + 1) for i in range(count)]
    monkeypatch.setattr(pd, 'get_sample_points', nodes)
    if npoles is None:
        with pytest.raises(ValueError, match='node search did not converge'):
            pd.interpolate_delta_table(exact, npoles, pole_selection='rightmost', shift=0)
        assert attempted == [2]
    else:
        # An explicit integer is a cap; its existing recovery can retain fewer poles.
        reduced = pd.interpolate_delta_table(exact, npoles, pole_selection='rightmost', shift=0)
        assert reduced.prefactor.pole_degree == 1
        assert attempted == [2, 1]
