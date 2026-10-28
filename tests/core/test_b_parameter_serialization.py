"""Parameter transfer preserves both numerical bits and the originating theory."""

import copy
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor

import mpmath as mp
import pytest

from virasoro_blocks.params import b_from_c, c_origin_of_b
from virasoro_blocks.recursion import F_block, H_q_series, _maybe_real_scalar


EXT = ('.1', '.2', '.3', '.4')


def check_values(actual, expected, digits=55):
    if not isinstance(actual, list):
        actual, expected = [actual], [expected]
    with mp.workdps(100):
        for value, ref in zip(actual, expected):
            assert abs(value-ref) < mp.mpf(10)**-digits*max(1, abs(ref))


def transferred_values(parameters):
    # A spawned worker receives the source label with the parameter object.
    with mp.workdps(60):
        return [(c_origin_of_b(b), F_block('.7', '.3', EXT, b=b, q_order=4),
                 H_q_series('.7', EXT, b=b, q_order=4)) for b in parameters]


@pytest.mark.parametrize('charge', ['.3', 8, 30, complex(30, 1)])
@pytest.mark.parametrize('protocol', range(pickle.HIGHEST_PROTOCOL+1))
def test_pickle_retains_source_and_bits_at_a_lower_restoration_precision(charge, protocol):
    with mp.workdps(180):
        original = b_from_c(charge)
    source, raw = c_origin_of_b(original), original._mpc_
    payload = pickle.dumps(original, protocol=protocol)
    with mp.workdps(15):
        restored = pickle.loads(payload)
        assert isinstance(restored, mp.mpc) and restored._mpc_ == raw
        assert c_origin_of_b(restored) == source
        assert mp.mp.dps == 15
    # Exercise repeated transfers, rather than only the first restoration.
    twice = pickle.loads(pickle.dumps(restored, protocol=protocol))
    assert c_origin_of_b(twice) == source and twice._mpc_ == raw


@pytest.mark.parametrize('transfer', [copy.copy, copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))])
@pytest.mark.parametrize('real_view', [False, True])
def test_copy_and_pickle_preserve_theory_across_precision_changes(transfer, real_view):
    with mp.workdps(20):
        b = b_from_c(30 if real_view else '.3')
        if real_view:
            b = b.real
    source = c_origin_of_b(b)
    with mp.workdps(15):
        restored = transfer(b)
        assert c_origin_of_b(restored) == source
    with mp.workdps(60):
        check_values(F_block('.7', '.3', EXT, b=restored, q_order=4),
                     F_block('.7', '.3', EXT, c=source, q_order=4))
        check_values(H_q_series('.7', EXT, b=restored, q_order=4),
                     H_q_series('.7', EXT, c=source, q_order=4))


def test_real_arithmetic_view_keeps_object_owned_provenance():
    with mp.workdps(20):
        original = b_from_c(30)
    real = _maybe_real_scalar(original)
    assert isinstance(real, mp.mpf) and c_origin_of_b(real) == 30
    with mp.workdps(60):
        check_values(H_q_series('.7', EXT, b=original, q_order=4),
                     H_q_series('.7', EXT, c=30, q_order=4))


def test_a_nonreal_parameters_real_component_is_not_a_parameter_for_an_unrelated_charge():
    with mp.workdps(40):
        unrelated = b_from_c('38.5')
        b = b_from_c(complex(13, -9))
        component = b.real
    assert unrelated.real == component == mp.mpf('.5')
    assert b.imag != 0 and c_origin_of_b(component) is None


def test_spawned_worker_uses_exact_sources_for_blocks_and_recursion():
    with mp.workdps(20):
        parameters = [b_from_c('.3'), b_from_c(8), b_from_c(30), b_from_c(30).real]
    with mp.workdps(60):
        references = [(c_origin_of_b(b), F_block('.7', '.3', EXT, c=c_origin_of_b(b), q_order=4),
                       H_q_series('.7', EXT, c=c_origin_of_b(b), q_order=4)) for b in parameters]
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
        results = pool.submit(transferred_values, parameters).result(timeout=30)
    for (source, value, series), (ref_source, ref_value, ref_series) in zip(results, references):
        assert source == ref_source
        check_values(value, ref_value)
        check_values(series, ref_series)


def test_direct_numeric_parameters_still_have_no_source_after_pickle():
    with mp.workdps(40):
        original = mp.mpc('.37', '.12')
    restored = pickle.loads(pickle.dumps(original))
    assert type(restored) is mp.mpc and c_origin_of_b(restored) is None
    assert restored._mpc_ == original._mpc_
