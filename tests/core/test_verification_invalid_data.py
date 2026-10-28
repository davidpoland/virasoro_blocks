"""Verification must not silently omit zero references or invalid arithmetic."""
import os
import subprocess
import sys
from pathlib import Path

import mpmath as mp
import pytest
from flint import acb

from virasoro_blocks.verified_build import (
    PrecisionRequest, Verification, agreement_digits, point_agreement, run_verified_build,
)


@pytest.mark.parametrize("value, digits", [("1", 0), ("1e-25", 25)])
def test_zero_reference_uses_absolute_error(value, digits):
    measured = point_agreement([[mp.mpf(value)]], [[0]], [(0, 0)], 40)
    assert measured == pytest.approx((digits, digits))
    assert point_agreement([[0]], [[0]], [(0, 0)], 40) is None


def test_bad_zero_reference_point_is_not_skipped():
    assert agreement_digits([[[1]], [[1]]], [[[1]], [[0]]], [(0, 0)], 40) == 0
    assert point_agreement([], [], [], 40) is None


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
@pytest.mark.parametrize("side", ["got", "ref"])
@pytest.mark.parametrize("index", [0, 1])
def test_nonfinite_flint_entries_cannot_pass(bad, side, index):
    rows = {"got": [[acb(1), acb(1)]], "ref": [[acb(1), acb(1)]]}
    rows[side][0][index] = acb(bad)
    with pytest.raises(ValueError, match="non-finite"):
        point_agreement(rows["got"], rows["ref"], [(0, 0), (0, 1)], 40)


@pytest.mark.parametrize("target", [None, 30])
def test_nan_verification_cannot_satisfy_a_build(target):
    with pytest.raises(ValueError, match="NaN"):
        run_verified_build(lambda work: object(), lambda built, work: Verification(float("nan")),
                           request=PrecisionRequest.make(20, target), what="test table",
                           order_reduced=False, report=False)


def test_far_check_points_reach_the_large_delta_regime_off_the_poles(tmp_path):
    from virasoro_blocks.table_backends import _far_check_points_x

    nodes = [mp.mpf(k) / 3 for k in range(1, 31)]            # last node 10
    poles = [mp.mpc(-3, 150), mp.mpc(-3, -150), mp.mpf("25")]  # |pole| up to 150; 25 = 1.25 x 10 x 2 exactly
    points = _far_check_points_x(nodes, poles)
    assert points[0] > 10 and points[-1] >= 16 * abs(poles[0])
    assert all(b / a <= 2.1 for a, b in zip(points, points[1:]))
    assert all(abs(x - 25) > mp.mpf("0.02") for x in points)
    # Bound this check in a child process: ordinary floats made the large/zero
    # cases loop forever, and underflow placed a checkpoint exactly on a pole.
    import virasoro_blocks.verified_build as verification

    code = '''
import mpmath as mp
from virasoro_blocks.verified_build import far_check_points_x
for nodes, poles in ((["1e400"], ["-1e400"]), ([0], [-1]),
                     (["1e-500"], ["1.25e-500"]), (["1e-322"], ["1.25e-322"]),
                     (["1e308"], [mp.mpc("1.7e308", "1.7e308")]),
                     (["1e308"], [mp.mpc(0, "1.5e308")])):
    nodes, poles = [mp.mpf(x) for x in nodes], [mp.mpc(x) for x in poles]
    points = far_check_points_x(nodes, poles)
    assert points and all(mp.isfinite(x) and x > 0 for x in points)
    assert points[-1] >= 16 * max(abs(x) for x in poles)
    assert all(abs(x - pole) >= x / 1000 for x in points for pole in poles)
for bad in (mp.inf, -mp.inf, mp.nan, mp.mpc(1, mp.inf)):
    for nodes, poles in (([bad], []), ([1], [bad])):
        try:
            far_check_points_x(nodes, poles)
        except ValueError:
            pass
        else:
            raise AssertionError("nonfinite checkpoint geometry was accepted")
with mp.workprec(3):
    try:
        far_check_points_x([1], ["1.25"])
    except ValueError:
        pass
    else:
        raise AssertionError("unresolvable pole displacement was accepted")
'''
    checked = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                             timeout=5, cwd=tmp_path,
                             env={**os.environ, 'PYTHONPATH': str(Path(verification.__file__).resolve().parents[1])})
    assert checked.returncode == 0, checked.stderr
