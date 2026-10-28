"""Standalone reconstruction follows the task's input and default policies."""
from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_bootstrap import identical_scalar
from virasoro_bootstrap.hpc import recertify, tasks


@pytest.fixture
def captured_spec(monkeypatch):
    monkeypatch.setattr(identical_scalar,'build_identical_scalar_gap_problem',lambda spec: spec)


def test_standalone_preserves_exact_external_label_at_ambient_precision(captured_spec):
    with mp.workdps(15):
        spec = recertify.build_scalar_gap_problem(dict(c='2',h_ext=.7,lam=7,dps=40),None)
    assert spec.h_ext == Fraction(7,10)


@pytest.mark.parametrize('name,value', [('q_order',8.5),('spins',[0,2.5]),('tower','false')])
def test_standalone_refuses_silent_control_changes(captured_spec,name,value):
    params = dict(c='2',lam=7)
    params[name] = value
    with pytest.raises(ValueError):
        recertify.build_scalar_gap_problem(params,None)


def test_standalone_preserves_zero_profile_settings(captured_spec,monkeypatch):
    preset = dict(tasks.LAMBDA_PRESETS[7],large_spin_npoles=0,large_spin_double_twists=0)
    monkeypatch.setitem(tasks.LAMBDA_PRESETS,7,preset)
    spec = recertify.build_scalar_gap_problem(dict(c='2',lam=7,twist_gap='match_gap'),None)
    assert spec.large_spin_npoles == 0
    assert spec.large_spin_twists == (spec.delta_gap,)
