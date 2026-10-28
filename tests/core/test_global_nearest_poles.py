"""Nearest genuine poles agree with exhaustive finite-window enumeration."""
import mpmath as mp
import pytest
from virasoro_blocks import global_blocks as gb


@pytest.mark.parametrize('shift', [-100, -5.25, 0, 20])
@pytest.mark.parametrize('diffs', [(0, 0), (.2, .3), (0, .5), (4, 3.5)])
@pytest.mark.parametrize('ell', [0, 2, -3])
def test_2d_nearest_poles_match_enumeration(shift, diffs, ell):
    candidates = []
    for p in range(300):
        if not gb._chiral_pole_is_removable(p, *diffs):
            candidates.extend([mp.mpf(ell-p), mp.mpf(-ell-p)])
    expected = sorted(candidates, key=lambda x: (abs(x-shift), -x))[:7]
    assert gb._genuine_2d_poles(ell, 7, dh12=diffs[0], dh34=diffs[1],
        dhb12=diffs[0], dhb34=diffs[1], shift=shift) == expected


@pytest.mark.parametrize('shift', [-100, -5.25, 0, 20])
@pytest.mark.parametrize('diffs', [(0, 0), (.2, .3), (0, .5), (4, 3.5)])
def test_chiral_nearest_poles_match_enumeration(shift, diffs):
    candidates = [-mp.mpf(p)/2 for p in range(300) if not gb._chiral_pole_is_removable(p, *diffs)]
    expected = sorted(candidates, key=lambda x: (abs(x-shift), -x))[:7]
    assert gb._genuine_chiral_poles(7, delta12=diffs[0], delta34=diffs[1], shift=shift) == expected


def test_far_shift_preserves_coincident_poles():
    assert gb.global_sl2_delta_poles(0, 2, shift=-1000000) == [-999999, -999999]


def test_asymmetric_halves_keep_distinct_removability_rules():
    ell, shift = mp.mpf('1.5'), mp.mpf('-40.25')
    candidates = []
    for p in range(120):
        if not gb._chiral_pole_is_removable(p, 0, 0):
            candidates.append(ell-p)
        if not gb._chiral_pole_is_removable(p, .2, .3):
            candidates.append(-ell-p)
    expected = sorted(candidates, key=lambda x: (abs(x-shift), -x))[:9]
    assert gb._genuine_2d_poles(ell, 9, dh12=.2, dh34=.3,
        dhb12=0, dhb34=0, shift=shift) == expected


@pytest.mark.parametrize('shift', [mp.nan, mp.inf, -mp.inf])
def test_nearest_poles_reject_nonfinite_centers(shift):
    with pytest.raises(ValueError, match='finite'):
        gb.global_sl2_delta_poles(0, 2, shift=shift)
