"""PMP sampling preserves real positivity-domain nodes rather than projecting them."""
from types import SimpleNamespace
import mpmath as mp
import pytest
from virasoro_bootstrap.pmp_writer import _build_sampling_data


@pytest.mark.parametrize('points', [(1, 1), (-1, 2), (1+1j,)])
def test_pmp_rejects_invalid_sampling_nodes(points):
    block = SimpleNamespace(sample_points_x=points, base=mp.mpf('.5'))
    with pytest.raises(ValueError, match='sample'):
        _build_sampling_data(block, (), coordinate='zzb', F_pairs=[(0, 0)], sig_figs=30)


def test_pmp_preserves_real_sampling_nodes():
    block = SimpleNamespace(sample_points_x=(mp.mpc(0), mp.mpf(2)), base=mp.mpf('.5'))
    data = _build_sampling_data(block, (-1,), coordinate='zzb', F_pairs=[(0, 0)], sig_figs=30)
    assert list(map(mp.mpf, data['samplePoints'])) == [0, 2]
    assert mp.almosteq(mp.mpf(data['sampleScalings'][1]), mp.mpf(1)/12)


def test_no_sample_nodes_retains_implicit_sampling():
    block = SimpleNamespace(sample_points_x=None)
    assert _build_sampling_data(block, (), coordinate='zzb', F_pairs=[], sig_figs=30) is None


@pytest.mark.parametrize('bad', [mp.mpc(1, mp.nan)])
def test_pmp_numeric_output_rejects_nonfinite_values(bad):
    from virasoro_bootstrap.pmp_writer import _pmp_real_str
    with pytest.raises(ValueError, match='finite'):
        _pmp_real_str(bad, 30, where='objective')


@pytest.mark.parametrize('scaling', [0])
def test_pmp_rejects_invalid_sampling_scalings(monkeypatch, scaling):
    from virasoro_bootstrap import pmp_writer
    monkeypatch.setattr(pmp_writer, '_compute_sample_scalings', lambda *a, **kw: [scaling])
    block = SimpleNamespace(sample_points_x=(1,), base=mp.mpf('.5'))
    with pytest.raises(ValueError, match='scalings'):
        _build_sampling_data(block, (), coordinate='zzb', F_pairs=[(0, 0)], sig_figs=30)
