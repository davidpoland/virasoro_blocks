"""Merged rational adapters retain exact external labels for the shared Laurent state."""
from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_blocks import recursion
from virasoro_blocks.numeric_input import ExactRational, literal_number


class StopAtContext(BaseException):
    pass


@pytest.mark.parametrize('builder', [recursion.merged_skeleton_rational, recursion.merged_H_rational])
@pytest.mark.parametrize('label', [ExactRational(9,10), ExactRational(9,10)+ExactRational(1,10**70)])
def test_exact_externals_reach_merged_context(monkeypatch, builder, label):
    captured = []
    def capture(order, c, externals, **kwargs):
        captured.append(externals)
        raise StopAtContext
    monkeypatch.setattr(recursion, 'build_merged_H_context', capture)
    with mp.workdps(30), pytest.raises(StopAtContext):
        builder([label]*4,c0=26,q_order=4)
    assert [literal_number(h) for h in captured[0]] == [Fraction(label)]*4


def test_rounded_degenerate_externals_stay_degenerate():
    # the Z3 parafermion (c = 4/5, sigma = 1/15) handed over as mpf values: h = 3 = d_{4,1} is removable for the
    # degenerate externals, as for the exact labels, not a "genuine Kac pole" of their rounding
    with mp.workdps(30):
        exact = recursion.F_block(3, Fraction(3, 10), [Fraction(1, 15)]*4, c=Fraction(4, 5), q_order=30)
        rounded = recursion.F_block(3, mp.mpf('0.3'), [mp.mpf(1)/15]*4, c=mp.mpf(4)/5, q_order=30)
        assert abs(rounded - exact) < mp.mpf('1e-28')*abs(exact)


