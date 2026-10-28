"""External labels determine genuine poles even below working precision."""
from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_blocks.api import BlockPoleError, BlockRequest, NumericalPolicy, evaluate_block_request
from virasoro_blocks.params import b_from_c
from virasoro_blocks.recursion import F_block
from virasoro_blocks import global_blocks as gb


@pytest.mark.parametrize('source', ['fraction', 'mpf', 'mpc', 'mixed'])
@pytest.mark.parametrize('entry', ['direct_c', 'direct_b', 'request'])
def test_small_external_differences_do_not_remove_the_vacuum_pole(source, entry):
    shift = Fraction(1, 10**70)
    ext = (Fraction(3, 10), Fraction(3, 10)+shift, Fraction(1, 2)+shift, Fraction(1, 2))
    if source != 'fraction':
        with mp.workdps(160):
            ext = tuple(mp.convert(value) for value in ext)
            if source == 'mpc':
                ext = tuple(mp.mpc(value) for value in ext)
            elif source == 'mixed':
                # A Fraction and an mpc can compare equal after rounding at
                # low precision. Their exact source components differ here.
                ext = (Fraction(3, 10), mp.mpc(ext[1]), mp.mpc(ext[2]), Fraction(1, 2))
    with mp.workdps(40), pytest.raises(BlockPoleError):
        if entry == 'request':
            evaluate_block_request(BlockRequest(
                operation='evaluate', basis='chiral_F', coordinate='z', central_charge=32,
                h_ext=ext, exchange_h=0, point={'z': '.3'},
                policy=NumericalPolicy(precision=40, order=4)), q_gate='off')
        else:
            kwargs = {'c': 32} if entry == 'direct_c' else {'b': b_from_c(32)}
            F_block(0, mp.mpf('.3'), ext, q_order=4, **kwargs)


def test_external_shift_removes_nonzero_kac_fusion_cancellation():
    with mp.workdps(160):
        degenerate = mp.mpc('-.5', '.75')
        shifted = degenerate + mp.mpf('1e-70')
        ext = (degenerate, shifted, shifted, degenerate)
    with mp.workdps(40), pytest.raises(BlockPoleError):
        F_block(mp.mpc(-1, 2), mp.mpf('.3'), ext, c=13, q_order=4)


def test_exactly_paired_vacuum_keeps_its_existing_working_precision(monkeypatch):
    import virasoro_blocks.recursion as recursion

    # Probe the dispatch precision, not the intentional precision increase
    # inside the removable-vacuum symmetric limit.
    calls = []
    monkeypatch.setattr(recursion, '_bare_F_block', lambda *a, **kw: calls.append(mp.mp.dps) or mp.mpf(1))
    with mp.workdps(160):
        external = mp.mpf('.3')
    with mp.workdps(40):
        assert recursion.F_block(0, mp.mpf('.3'), (external,)*4, c=32, q_order=4) == 1
    assert calls == [40]


@pytest.mark.parametrize('center', [Fraction(0), Fraction(1, 2), Fraction(1)])
@pytest.mark.parametrize('direction', ['real', 'imag'])
def test_global_pole_removability_requires_an_exact_half_integer(center, direction):
    with mp.workdps(160):
        offset = center+Fraction(1, 10**70) if direction == 'real' else mp.mpc(mp.convert(center), '1e-70')
    with mp.workdps(40):
        # Neither external difference can kill any pole once shifted.
        assert gb._global_chiral_q_order_pole_pool(4, delta12=offset, delta34=offset) == [0, -.5, -1, -1.5]
        assert gb._genuine_chiral_poles(2, delta12=offset, delta34=offset, shift=-1) == [-1, -.5]


def test_global_external_differences_are_taken_before_numeric_rounding():
    ext = (Fraction(3, 10), Fraction(13, 10), Fraction(17, 10), Fraction(7, 10))
    with mp.workdps(15):
        assert gb._externals_diffs(ext) == (1, 1)
        assert gb.global_sl2_delta_poles(0, 2, h_externals=ext, shift=-2) == [-1, -1]
        shifted = (ext[0], ext[1]+Fraction(1, 10**70), ext[2]+Fraction(1, 10**70), ext[3])
        assert gb.global_sl2_delta_poles(0, 2, h_externals=shifted, shift=-2) == [-2, -2]


def test_native_mpf_external_differences_are_half_integers_within_roundoff():
    # mpf('.6') - mpf('.1') is not exactly 1/2 in binary; at the working precision it is.
    exact = [Fraction(1, 10), Fraction(6, 10)]*2
    with mp.workdps(30):
        native = [mp.mpf('.1'), mp.mpf('.6')]*2
        assert gb.global_sl2_delta_poles(0, 6, h_externals=native) == \
            gb.global_sl2_delta_poles(0, 6, h_externals=exact) == [0, 0, -2, -2, -4, -4]


@pytest.mark.parametrize('chiral', [False, True])
def test_global_table_budget_uses_poles_of_exact_external_labels(monkeypatch, chiral):
    shift = Fraction(1, 10**70)
    ext = (Fraction(3, 10), Fraction(3, 10)+shift, Fraction(1, 2)+shift, Fraction(1, 2))
    monkeypatch.setattr(gb, '_interpolate_global_table', lambda **kw: kw)
    with mp.workdps(40):
        if chiral:
            table = gb.global_chiral_F_interpolated_table('z', ext, 0, q_order=4,
                npoles=3, pole_selection='contribution_optimal')
            assert sorted(table['pool']) == [-1.5, -1, -.5, 0]
        else:
            table = gb.global_scalar_G_interpolated_table('zzb', 0, ext, 0, q_order=4,
                npoles=5, pole_selection='contribution_optimal')
            assert sorted(table['pool']) == [-3, -3, -2, -2, -1, -1, 0, 0]
