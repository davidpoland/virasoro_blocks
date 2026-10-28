"""Parity reversal carries each external-weight list with its exchanged weight."""
import mpmath as mp
import pytest

from virasoro_blocks.point import point_block_table
from virasoro_blocks.api import BlockRequest, NumericalPolicy, evaluate_block_request
from virasoro_blocks.requests import BlockPoleError


@pytest.mark.parametrize('family', ['global', 'virasoro'])
@pytest.mark.parametrize('parity', ['+', '-'])
def test_point_parity_projection_preserves_removable_half(family, parity):
    with mp.workdps(40):
        ext = [mp.mpf(v) for v in ('.1', '.2', '.3', '.4')]
        anti = [mp.mpf('.1')] * 4
        basis = ('global_' if family == 'global' else '') + 'parity_projected_B'
        table = point_block_table(['zzb'], basis=basis, ell=2, point_value=2,
            h_externals=ext, hbar_externals=anti, lam=0, c=32, q_order=8,
            parity=parity, center=mp.mpf('.3'), center_bar=mp.mpf('.4'))
        request = BlockRequest(operation='evaluate', family=family, basis=basis,
            coordinate='zzb', central_charge=32, h_ext=ext, hbar_ext=anti,
            exchange_delta=2, exchange_spin=2, parity=parity,
            point={'z': '.3', 'zbar': '.4'},
            policy=NumericalPolicy(precision=40, order=8))
        expected = evaluate_block_request(request, q_gate='off').value
        actual = table.evaluate('zzb', mp.mpf(2))[0][0]
        assert mp.isfinite(actual)
        assert abs(actual - expected) < mp.mpf('1e-30')


@pytest.mark.parametrize('family', ['global', 'virasoro'])
def test_point_parity_projection_still_rejects_genuine_half_pole(family):
    basis = ('global_' if family == 'global' else '') + 'parity_projected_B'
    with mp.workdps(40), pytest.raises(BlockPoleError):
        point_block_table(['zzb'], basis=basis, ell=2, point_value=2,
            h_externals=[mp.mpf('.1')] * 4,
            hbar_externals=[mp.mpf(v) for v in ('.1', '.2', '.3', '.4')],
            lam=0, c=32, q_order=8, parity='+')


@pytest.mark.parametrize('family', ['virasoro', 'ns', 'ramond'])
@pytest.mark.parametrize('basis', ['chiral_F', 'ordered_B', 'scalar_G', 'parity_projected_B'])
def test_tail_report_only_probes_factors_present_in_block(monkeypatch, family, basis):
    from virasoro_blocks import qgate
    from virasoro_blocks.evaluation import _q_tail
    seen = set()
    def report(**kwargs):
        seen.add((kwargs['h'], kwargs['z'], tuple(kwargs['h_externals'])))
        return {'status': 'converged', 'digits': 30}
    monkeypatch.setattr(qgate, 'point_q_tail_report', report)
    monkeypatch.setattr(qgate, 'super_point_q_tail_report', report)
    ext = (1, 2, 3, 4)
    anti = ext if basis == 'scalar_G' else (5, 6, 7, 8)
    _q_tail(family, 2, 0, '.3', None if basis == 'chiral_F' else '.4',
            ext, anti, c=32, order=8, sector='even', branch=None, basis=basis)
    expected = {(2, '.3', ext)}
    if basis != 'chiral_F':
        expected.add((0, '.4', anti))
    if basis in {'scalar_G', 'parity_projected_B'}:
        expected.update({(2, '.4', ext), (0, '.3', anti)})
    assert seen == expected
