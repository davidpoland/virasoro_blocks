"""Interpolation diagnostics cannot report success after losing invalid samples."""

import mpmath as mp
import pytest

from virasoro_blocks.interpolation import DeltaInterpolant, DampedRational, max_abs_errors_by_npoles
from virasoro_blocks.polynomial import Polynomial


@pytest.fixture
def interpolant():
    return DeltaInterpolant(DampedRational(1, (), mp.exp(-mp.pi)), 0, (0,), Polynomial([1]))


def test_relative_error_evaluates_reference_once(interpolant):
    values = iter([mp.mpf(2), mp.mpf(4)])
    calls = []
    def evaluator(delta):
        calls.append(delta)
        return next(values)
    assert interpolant.relative_error(evaluator, 0) == mp.mpf(".5")
    assert calls == [0]


@pytest.mark.parametrize("bad", [mp.nan, mp.inf, -mp.inf, -1])
def test_relative_error_rejects_invalid_floor(interpolant, bad):
    with pytest.raises(ValueError, match="floor"):
        interpolant.relative_error(lambda _: mp.mpf(2), 0, floor=bad)


@pytest.mark.parametrize("bad", [mp.nan, mp.inf, mp.mpc(1, mp.inf)])
def test_relative_error_reports_nonfinite_reference_as_infinite(interpolant, bad):
    assert interpolant.relative_error(lambda _: bad, 0) == mp.inf


@pytest.mark.parametrize("bad", [mp.nan, mp.mpc(1, mp.nan)])
@pytest.mark.parametrize("index", [0, 1])
def test_maximum_error_does_not_hide_nan(bad, index):
    curve = [mp.mpf(0)]
    curve.insert(index, bad)
    assert max_abs_errors_by_npoles({2: curve})[2] == mp.inf


def test_zero_floor_remains_supported_for_nonzero_reference(interpolant):
    assert interpolant.relative_error(lambda _: mp.mpf(2), 0, floor=0) == mp.mpf(".5")


@pytest.mark.parametrize("bad", [mp.nan, mp.inf])
def test_plot_refuses_nonfinite_error_summary(monkeypatch, bad):
    plt = pytest.importorskip("matplotlib.pyplot")
    from virasoro_blocks.interpolation import plot_max_abs_error_by_npoles
    def unexpected():
        pytest.fail("invalid error summaries must fail before rendering")
    monkeypatch.setattr(plt, "subplots", unexpected)
    with pytest.raises(ValueError, match="finite"):
        plot_max_abs_error_by_npoles({2: [mp.mpf(1), bad]})


@pytest.mark.parametrize("bad", [mp.nan, mp.inf, mp.mpc(1, mp.nan)])
def test_relative_error_reports_nonfinite_approximation_as_infinite(monkeypatch, interpolant, bad):
    monkeypatch.setattr(DeltaInterpolant, "evaluate", lambda self, delta: bad)
    assert interpolant.relative_error(lambda _: mp.mpf(2), 0) == mp.inf


def test_finite_complex_errors_keep_their_magnitude():
    assert max_abs_errors_by_npoles({2: [mp.mpc(3, 4), -2]}) == {2: mp.mpf(5)}
