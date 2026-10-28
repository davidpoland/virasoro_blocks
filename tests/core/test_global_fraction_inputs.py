"""Ordinary rational inputs agree with their working-precision equivalents."""

from fractions import Fraction as F

import mpmath as mp
import pytest

from virasoro_blocks import global_blocks as gb
from virasoro_blocks.polynomial import _arb_to_mpf
from virasoro_blocks.numeric_input import literal_number


@pytest.mark.parametrize('basis', ['global_scalar_G', 'global_ordered_B', 'global_chiral_F'])
@pytest.mark.parametrize('fraction_part', ['external', 'dimension'])
def test_global_sampler_rational_inputs(basis, fraction_part):
    chiral = basis == 'global_chiral_F'
    with mp.workdps(40):
        external = F(1, 8) if fraction_part == 'external' else mp.mpf(1)/8
        sampler = gb.build_global_sampler('z' if chiral else 'zzb', basis=basis,
            ell=0, h_externals=[external]*4, hbar_externals=[external]*4, m_max=1, n_max=0 if chiral else 1)
        dimension = F(3, 2) if fraction_part == 'dimension' else mp.mpf('1.5')
        actual = sampler(dimension)
        expected = sampler(mp.mpf('1.5'))
        assert actual == expected
        if not chiral:
            batch, is_arb = sampler.sample_bucket([dimension], [(0, 0), (1, 0)])
            for m, n in [(0, 0), (1, 0)]:
                v = _arb_to_mpf(batch[0][m][n]) if is_arb else batch[0][m][n]
                assert mp.almosteq(v, expected[m][n])


@pytest.mark.parametrize('chiral', [False, True])
@pytest.mark.parametrize('fraction_part', ['external', 'shift'])
def test_global_table_rational_inputs(monkeypatch, chiral, fraction_part):
    external = F(1, 8) if fraction_part == 'external' else mp.mpf('.125')
    shift = F(1, 3) + F(1, 10**70) if fraction_part == 'shift' else mp.mpf('.5')
    monkeypatch.setattr(gb, '_interpolate_global_table', lambda **kw: kw)
    if chiral:
        result = gb.global_chiral_F_interpolated_table('z', [external]*4, 0, shift=shift, npoles=0)
    else:
        result = gb.global_scalar_G_interpolated_table('zzb', 0, [external]*4, 0, shift=shift, npoles=0)
    assert literal_number(result['shift']) == literal_number(shift)
    assert result['sampler'].h_ext == [mp.mpf('.125')]*4


def test_global_pole_list_accepts_rational_external_weights():
    assert gb.global_sl2_delta_poles(0, 2, h_externals=[F(1, 8)]*4) == gb.global_sl2_delta_poles(0, 2)


def test_global_chiral_interpolation_and_serialization_keep_the_exact_origin():
    from virasoro_blocks.serialization import serialize_deriv_tables
    shift = F(1, 3) + F(1, 10**70)
    with mp.workdps(30):
        table = gb.global_chiral_F_interpolated_table('z', [F(1, 8)]*4, 0,
                                                     shift=shift, npoles=0)
        serialized = serialize_deriv_tables({'z': table})
    assert literal_number(table.shift) == literal_number(serialized.shift) == shift
