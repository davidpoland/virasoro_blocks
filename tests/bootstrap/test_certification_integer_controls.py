"""Requested certificate coverage cannot be silently truncated or skipped."""

from fractions import Fraction
from types import SimpleNamespace

import mpmath as mp
import numpy as np
import pytest

from virasoro_bootstrap import certification


def channel():
    return SimpleNamespace(ell=0, delta_min=0,
        prefactored=SimpleNamespace(evaluate=lambda coordinate, delta: [[1]]))


@pytest.mark.parametrize('count', [True, 2.5])
def test_scan_worker_counts_are_not_silently_truncated(count):
    with pytest.raises(ValueError, match='num_threads'):
        certification.certify_dual_functional([1], [channel()], [(0, 0)],
                                               num_threads=count, n_test_points=2)


@pytest.mark.parametrize('field, count', [
    ('extra_spins', True), ('moment_cert_q_order', 2.5), ('num_threads', 2.5),
])
def test_wrapper_rejects_invalid_controls_before_loading_dual(monkeypatch, tmp_path, field, count):
    def unexpected(*args):
        pytest.fail('invalid controls reached the dual loader')
    monkeypatch.setattr(certification, 'load_y_vector', unexpected)
    problem = SimpleNamespace(spec=SimpleNamespace(dps=30, num_threads=1))
    with pytest.raises(ValueError, match=field):
        certification.certify_dual_from_sdpb_output(problem, tmp_path,
            q_gate='off', **{field: count})


@pytest.mark.parametrize('field', ['interstitial_twists'])
def test_negative_coverage_is_rejected_before_loading_dual(monkeypatch, tmp_path, field):
    monkeypatch.setattr(certification, 'load_y_vector', lambda *args: pytest.fail('reached loader'))
    problem = SimpleNamespace(spec=SimpleNamespace(dps=30))
    with pytest.raises(ValueError, match=field):
        certification.certify_dual_from_sdpb_output(problem, tmp_path,
            q_gate='off', **{field: -1})


@pytest.mark.parametrize('count', [-3, 1.0])
def test_documented_serial_worker_counts_still_certify(count):
    ok, details = certification.certify_dual_functional([1], [channel()], [(0, 0)],
        num_threads=count, n_test_points=2)
    assert ok and details[0]['n_test_points'] == 2


def test_subprecision_worker_offset_is_not_an_integer():
    with mp.workdps(150):
        count = mp.mpf(2) + mp.mpf('1e-100')
    with mp.workdps(15), pytest.raises(ValueError, match='num_threads'):
        certification.certify_dual_functional([1], [channel()], [(0, 0)], num_threads=count)


@pytest.mark.parametrize('count', [True])
def test_inherited_worker_count_is_validated_before_loading(monkeypatch, tmp_path, count):
    monkeypatch.setattr(certification, 'load_y_vector', lambda *args: pytest.fail('reached loader'))
    problem = SimpleNamespace(spec=SimpleNamespace(dps=30, num_threads=count))
    with pytest.raises(ValueError, match='num_threads'):
        certification.certify_dual_from_sdpb_output(problem, tmp_path, q_gate='off')


@pytest.mark.parametrize('field', ['extra_twists'])
def test_stored_subprecision_coverage_offsets_are_rejected(monkeypatch, tmp_path, field):
    monkeypatch.setattr(certification, 'load_y_vector', lambda *args: pytest.fail('reached loader'))
    problem = SimpleNamespace(spec=SimpleNamespace(dps=30))
    with mp.workdps(150):
        count = mp.mpf(2) + mp.mpf('1e-100')
    with mp.workdps(15), pytest.raises(ValueError, match=field):
        certification.certify_dual_from_sdpb_output(problem, tmp_path,
            q_gate='off', **{field: count})


@pytest.mark.parametrize('field, count', [('extra_spins', 0), ('log_near_points', 2.0)])
def test_valid_coverage_controls_reach_dual_loading(monkeypatch, tmp_path, field, count):
    class ReachedDualLoad(Exception):
        pass
    def reached(*args):
        raise ReachedDualLoad
    monkeypatch.setattr(certification, 'load_y_vector', reached)
    problem = SimpleNamespace(spec=SimpleNamespace(dps=30))
    with pytest.raises(ReachedDualLoad):
        certification.certify_dual_from_sdpb_output(problem, tmp_path,
            q_gate='off', **{field: count})


@pytest.mark.parametrize('override,inherited,expected', [
    (2.0, 1, 2), (None, 2.0, 2),
])
def test_wrapper_normalizes_workers_and_coverage_at_builder_boundaries(
        monkeypatch, tmp_path, override, inherited, expected):
    (tmp_path / 'y.txt').write_text('1 1\n1\n')
    problem = SimpleNamespace(spec=SimpleNamespace(dps=30, num_threads=inherited),
        F_pairs=[(0, 0)], spin_channels=['base'],
        pmp_build=SimpleNamespace(doc={'objective': ['0'], 'normalization': ['1']}))
    seen = []
    def extra_spins(problem, count, **kwargs):
        assert count == 2 and type(count) is int
        return ['spin']
    def extra_twists(problem, count):
        assert count == 2 and type(count) is int
        return ['twist']
    def certify(y, channels, pairs, **kwargs):
        assert y == [1] and channels == ['base', 'spin', 'twist']
        assert kwargs['num_threads'] == expected and type(kwargs['num_threads']) is int
        seen.append(kwargs['num_threads'])
        return True, [{'passed': True}]
    monkeypatch.setattr(certification, '_build_extra_spin_channels', extra_spins)
    monkeypatch.setattr(certification, '_build_extra_twist_channels', extra_twists)
    monkeypatch.setattr(certification, 'certify_dual_functional', certify)
    ok, details = certification.certify_dual_from_sdpb_output(problem, tmp_path,
        q_gate='off', exact_blocks=False, extra_spins=2.0, extra_twists=Fraction(2),
        num_threads=override)
    assert ok and seen == [expected]


def test_invalid_worker_count_does_not_consume_channel_iterator():
    def channels():
        pytest.fail('invalid control consumed channels')
        yield channel()
    with pytest.raises(ValueError, match='num_threads'):
        certification.certify_dual_functional([1], channels(), [(0, 0)], num_threads=1.5)


def test_root_controls_validate_before_combining_polynomials(monkeypatch):
    from virasoro_bootstrap import positivity

    entry = {'polynomials': [[[[1]]]]}
    with mp.workdps(150):
        offset = mp.mpf(2) + mp.mpf('1e-100')
    invalid = [True, 2.5, 0, offset]
    with monkeypatch.context() as guard:
        guard.setattr(positivity, '_combined_exact',
                      lambda *args: pytest.fail('invalid root controls reached polynomial construction'))
        for value in invalid:
            with pytest.raises(ValueError, match='dps'):
                positivity.block_positivity(entry, [1], dps=value)
            with pytest.raises(ValueError, match='dps'):
                positivity.positive_real_roots([1], dps=value)
            with pytest.raises(ValueError, match='maxsteps'):
                positivity.positive_real_roots([1], maxsteps=value)
    old_dps = mp.mp.dps
    for valid in (30.0, np.int64(30)):
        assert positivity.block_positivity(entry, [1], dps=valid)['passed'] is True
        assert positivity.positive_real_roots([1, -1], dps=valid) == [1]
        assert positivity.positive_real_roots([1, -1], maxsteps=valid) == [1]
    assert positivity.positive_real_roots([1], dps=None, maxsteps=None) == []
    assert mp.mp.dps == old_dps


def test_certification_switches_validate_before_loading_or_probing(monkeypatch, tmp_path):
    import virasoro_blocks.qgate as qgate

    class ReachedDualLoad(Exception):
        pass

    def load(*args):
        raise ReachedDualLoad

    def unexpected_probe(**kwargs):
        pytest.fail('invalid certificate options reached a numerical probe')

    monkeypatch.setattr(certification, 'load_y_vector', load)
    monkeypatch.setattr(qgate, 'gate_q_convergence', unexpected_probe)
    with_charge = SimpleNamespace(spec=SimpleNamespace(dps=30, c=4, h_ext='.1', q_order=4, lam=1))
    without_charge = SimpleNamespace(spec=SimpleNamespace(dps=30))
    switches = ('warn', 'adaptive_moment', 'physical_moment_blocks', 'exact_blocks', 'force_exact_included')
    for field in switches:
        for value in ('true', 0, None):
            with pytest.raises(ValueError, match=field + '.*True or False'):
                certification.certify_dual_from_sdpb_output(with_charge, tmp_path, **{field: value})
        for value in (True, False):
            with pytest.raises(ReachedDualLoad):
                certification.certify_dual_from_sdpb_output(without_charge, tmp_path,
                    q_gate='off', **{field: value})
    for problem in (with_charge, without_charge):
        for mode in ('strcit', True):
            with pytest.raises(ValueError, match='q_gate.*off.*warn.*strict'):
                certification.certify_dual_from_sdpb_output(problem, tmp_path, q_gate=mode)
    for mode in ('off', 'warn', 'strict'):
        with pytest.raises(ReachedDualLoad):
            certification.certify_dual_from_sdpb_output(without_charge, tmp_path, q_gate=mode)
