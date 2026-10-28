"""Tower slice policies retain exact labels and compare exact gap boundaries."""
from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_blocks.numeric_input import ExactRational
from virasoro_bootstrap.large_spin import double_twist_set, twists_above_gap


@pytest.mark.parametrize('h_ext', [0.7])
def test_double_twists_preserve_decimal_intent(h_ext):
    with mp.workdps(15):
        values = double_twist_set(h_ext, 2)
    assert values == [Fraction(14,5), Fraction(24,5)]


def test_gap_comparison_does_not_round_away_a_slice():
    gap = ExactRational(3,5)
    above = gap + ExactRational(1,10**70)
    below = gap - ExactRational(1,10**70)
    with mp.workdps(15):
        assert twists_above_gap([below,gap,above], gap) == [gap,above]


def test_gap_label_preserves_decimal_intent():
    with mp.workdps(15):
        assert twists_above_gap(['.7','.8'], '.7') == [Fraction(7,10), '.8']


def test_real_complex_inputs_follow_bootstrap_real_projection():
    import numpy as np

    assert double_twist_set(complex(.5,0),2) == [2,4]
    assert double_twist_set(complex(.5,0),np.int64(2)) == [2,4]
    for count in (np.bool_(True), np.bool_(False)):
        with pytest.raises(ValueError, match='n_count'):
            double_twist_set('.5', count)
    assert twists_above_gap([complex(.5,0),complex(.75,0)],complex(.5,0)) == [Fraction(1,2),complex(.75,0)]
    with pytest.raises(ValueError,match='real'):
        twists_above_gap([complex(.75,.1)],'.5')


def test_tower_worker_preserves_labels_and_crossing_shift(monkeypatch):
    from virasoro_bootstrap import large_spin

    class ReachedSampler(BaseException):
        pass
    h = ExactRational(7,10)+ExactRational(1,10**70)
    def sampler(mode,coordinate,externals,norm,z_shift,*args,**kwargs):
        assert externals == [h]*4
        assert z_shift == -2*h
        raise ReachedSampler
    monkeypatch.setattr(large_spin,'_LargeSpinTwistSampler',sampler)
    job = large_spin.large_spin_channel_jobs(mode='virasoro',c=2,h_ext=h,lam=1,
        q_order=8,npoles=2,Lmax=2,twists=[ExactRational(1,3)])[0]
    with pytest.raises(ReachedSampler):
        large_spin._large_spin_channel_task(job)


def test_tower_builder_captures_single_pass_slice_inputs(monkeypatch):
    from virasoro_bootstrap import large_spin

    calls = []
    def build(job):
        calls.append(job)
        return job[1]
    monkeypatch.setattr(large_spin, '_large_spin_channel_task', build)
    for twists in ((), (Fraction(1, 3),), (Fraction(1, 3), '.7')):
        calls.clear()
        result = large_spin.build_large_spin_channels(
            mode='virasoro', c=2, h_ext='.1', lam=1, q_order=8, npoles=2, Lmax=2,
            twists=iter(twists), num_threads=1)
        assert result == twists and len(calls) == len(twists)
        assert all(job[-1] == 1 for job in calls)
