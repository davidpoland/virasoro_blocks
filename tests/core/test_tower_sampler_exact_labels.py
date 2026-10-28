"""Tower evaluators must retain labels across construction and precision changes."""
from fractions import Fraction
from types import SimpleNamespace

import mpmath as mp
import pytest

from virasoro_blocks import large_spin
from virasoro_blocks.numeric_input import literal_number


@pytest.mark.parametrize('mode', ['global','virasoro'])
def test_sampler_constructed_at_low_precision_matches_fresh_high_precision(mode):
    def make():
        return large_spin._LargeSpinTwistSampler(mode,'zzb',['.7','.1','.3','.5'],
            16,0,1,1,'.7',c=32,q_order=4)
    with mp.workdps(15):
        reused = make()
    with mp.workdps(50):
        expected = make()(3)
        actual = reused(3)
        assert max(abs(actual[m][n]-expected[m][n]) / max(1,abs(expected[m][n]))
                   for m in range(2) for n in range(2)) < mp.mpf('1e-45')


def test_nearby_external_labels_do_not_share_the_two_chiral_halves():
    different = Fraction(7,10)+Fraction(1,10**70)
    with mp.workdps(15):
        sampler = large_spin._LargeSpinTwistSampler('virasoro','zzb',[Fraction(7,10)]*4,
            16,0,1,1,'.7',c=32,q_order=4,hbar_externals=[different]*4)
    assert sampler._same_externals is False
    assert all(literal_number(v) == different for v in sampler.hbar_ext)


def test_singular_root_cache_distinguishes_exact_external_labels(monkeypatch):
    from virasoro_blocks import recursion
    calls = []
    def skeleton(externals,**kwargs):
        calls.append(tuple(literal_number(v) for v in externals))
        return SimpleNamespace(denominator_roots=(mp.mpf(-1),))
    monkeypatch.setattr(recursion,'merged_skeleton_rational',skeleton)
    monkeypatch.setattr(large_spin,'_VIRASORO_CHIRAL_ROOTS_CACHE',{})
    h = Fraction(7,10)
    different = h+Fraction(1,10**70)
    with mp.workdps(15):
        large_spin._virasoro_chiral_roots((h,)*4,1,8)
        large_spin._virasoro_chiral_roots((different,)*4,1,8)
        large_spin._virasoro_chiral_roots((h,)*4,1,8)
    assert calls == [(h,)*4,(different,)*4]


def test_table_facade_preserves_external_labels_before_sampler(monkeypatch):
    class ReachedSampler(BaseException):
        pass
    def sampler(mode,coordinate,externals,*args,**kwargs):
        assert all(literal_number(v) == Fraction(7,10) for v in externals)
        raise ReachedSampler
    monkeypatch.setattr(large_spin,'_LargeSpinTwistSampler',sampler)
    with mp.workdps(15),pytest.raises(ReachedSampler):
        large_spin.large_spin_twist_table('.7',mode='global',coordinate='zzb',
            h_externals=[0.7]*4,m_max=1,n_max=1,Lmax=2,npoles=0,q_order=4)
