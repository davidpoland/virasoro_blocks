"""Explicit scan points remain exact and have validated sequence shapes."""
from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_bootstrap import identical_scalar, modular_crossing
from virasoro_bootstrap.hpc import tasks
from virasoro_bootstrap.hpc.journal import Journal
from virasoro_blocks.numeric_input import literal_number


CASES = [('scalar_ope_max_scan','delta_stars',False),
         ('isolated_relevant_scalar_delta_phi_intervals','delta_phi_grid',True),
         ('isolated_relevant_scalar_degeneracy_intervals','degeneracy_grid',False)]


class ReachedEngine(BaseException):
    pass


@pytest.fixture
def context(tmp_path,monkeypatch):
    calls = []
    def stop(*args,**kwargs):
        calls.append(args)
        raise ReachedEngine
    monkeypatch.setattr(identical_scalar,'prepare_identical_scalar_gap_build',stop)
    monkeypatch.setattr(modular_crossing,'prepare_modular_gap_build',stop)
    params = dict(c='2',lam=7,dps=40,delta_phi='.3',tower=False)
    return tasks.TaskContext(params,tmp_path,Journal(tmp_path/'progress.json')),calls


@pytest.mark.parametrize('task,field,value', [(CASES[0][0],CASES[0][1],'12'),
                                              (CASES[1][0],CASES[1][1],[-1]),
                                              (CASES[2][0],CASES[2][1],['nan'])])
def test_malformed_points_are_refused_before_engine(context,task,field,value):
    ctx,calls = context
    ctx.params[field] = value
    with pytest.raises(ValueError,match=field):
        tasks.get_task(task)(ctx)
    assert calls == []


@pytest.mark.parametrize('task,field,positive', CASES)
def test_explicit_decimal_points_reach_probe_without_rounding(context,monkeypatch,task,field,positive):
    ctx,calls = context
    value = Fraction(7,10)+Fraction(1,10**70)
    ctx.params[field] = [str(value)]
    if task == 'scalar_ope_max_scan':
        def stop(spec,*args,**kwargs):
            assert literal_number(spec.delta_gap) == value
            raise ReachedEngine
        monkeypatch.setattr(identical_scalar,'prepare_identical_scalar_gap_build',stop)
    else:
        monkeypatch.setattr(modular_crossing,'prepare_modular_gap_build',lambda spec: object())
        def stop(spec,*args,**kwargs):
            actual = spec.allowed_isolated_scalars[0] if positive else spec.isolated_scalars[0][1]
            assert literal_number(actual) == value
            raise ReachedEngine
        monkeypatch.setattr(modular_crossing,'build_modular_gap_problem',stop)
    with pytest.raises(ReachedEngine):
        tasks.get_task(task)(ctx)


def test_exact_probe_keys_do_not_merge_offsets_below_working_precision():
    first = Fraction(7,10)
    second = first+Fraction(1,10**70)
    with mp.workdps(15):
        labels = (tasks._label(first),tasks._label(second),tasks._label(Fraction(1,3)))
        assert labels[0] != labels[1]
        assert labels[0] == tasks._label('.7') == tasks._label(0.7)
    with mp.workdps(90):
        assert labels == (tasks._label(first),tasks._label(second),tasks._label(Fraction(1,3)))


def test_exact_binary_rational_keeps_existing_numeric_probe_key():
    with mp.workdps(60):
        value = mp.mpf('.7')
        exact = literal_number(value)
        key = tasks._label(value)
    with mp.workdps(15):
        assert tasks._label(exact) == key


@pytest.mark.parametrize('task,field', [(name,field) for name,field,_ in CASES if name != 'scalar_ope_max_scan'])
def test_interval_grids_still_require_strict_order(context,task,field):
    ctx,calls = context
    ctx.params[field] = ['.7','.7']
    with pytest.raises(ValueError,match='strictly increasing'):
        tasks.get_task(task)(ctx)
    assert calls == []


def test_exact_interval_endpoints_do_not_bypass_precision_limit():
    with mp.workdps(15),pytest.raises(ValueError,match='increase dps'):
        tasks._bisection_midpoint(Fraction(7,10),Fraction(7,10)+Fraction(1,10**70))


def test_exact_interval_midpoint_is_computed_at_task_precision():
    with mp.workdps(40):
        assert tasks._bisection_midpoint(Fraction(7,10),Fraction(4,5)) == mp.mpf('.75')


def test_bisection_twist_edges_and_self_consistent_weight_match_recertification(context,monkeypatch):
    from virasoro_bootstrap.hpc.recertify import build_scalar_gap_problem

    ctx,calls = context
    gap = Fraction(1,10**70)
    ctx.params.update(gap_lo=str(gap),all_spins_twist_gap=True,
                      self_consistent_h_ext=True,tower=True,twist_gap='match_gap',spins=[0,2])
    def stop(spec,*args,**kwargs):
        calls.append(spec)
        raise ReachedEngine
    monkeypatch.setattr(identical_scalar,'build_identical_scalar_gap_problem',stop)
    with pytest.raises(ReachedEngine):
        tasks.scalar_gap_bisection(ctx)
    solved = calls[0]
    assert literal_number(solved.delta_gap) == gap
    assert literal_number(solved.h_ext) == gap/2
    assert literal_number(solved.delta_min_by_spin[2]) == 2+gap
    assert literal_number(solved.large_spin_twists[0]) == gap
    monkeypatch.setattr(identical_scalar,'build_identical_scalar_gap_problem',lambda spec: spec)
    assert build_scalar_gap_problem(ctx.params,solved.delta_gap) == solved


@pytest.mark.parametrize('modular', [False,True])
def test_bisection_preserves_supplied_brackets_and_resumes_completed_probes(context,monkeypatch,modular):
    from types import SimpleNamespace

    ctx,_ = context
    lo,hi,cap = Fraction(7,20),Fraction(3,5),Fraction(1)+Fraction(1,10**70)
    ctx.params.update(spins=[0,2],gap_lo=str(lo),gap_hi=str(hi),max_hi=str(cap),tol='1/20')
    module = modular_crossing if modular else identical_scalar
    prefix = 'modular' if modular else 'identical_scalar'
    monkeypatch.setattr(module,f'prepare_{prefix}_gap_build',lambda spec: object())
    monkeypatch.setattr(module,f'build_{prefix}_gap_problem',lambda spec,**kwargs: spec)
    monkeypatch.setattr(tasks,'sdp_structure_signature',lambda problem: 'same')
    probes = []
    def run(spec,*args,**kwargs):
        gap = literal_number(spec.delta_min_by_spin[0] if modular else spec.delta_gap)
        probes.append(gap)
        return SimpleNamespace(sdpb_status='primal_feasible' if gap < Fraction(3,4) else 'dual_feasible',
                               sdpb_log_path=ctx.output_dir/'absent.log',checkpoint_dir=None,sdpb_summary=None)
    monkeypatch.setattr(module,f'run_{prefix}_gap_problem',run)
    task = tasks.modular_scalar_gap_bisection if modular else tasks.scalar_gap_bisection
    result = task(ctx)
    assert probes[:3] == [lo,hi,Fraction(9,10) if modular else cap]
    lower = Fraction(result['scalar_gap_lo' if modular else 'delta_gap_lo'])
    upper = Fraction(result['scalar_gap_hi' if modular else 'delta_gap_hi'])
    assert lower < Fraction(3,4) <= upper
    assert upper-lower <= Fraction(1,20)
    before = list(probes)
    repeated = task(ctx)
    assert {key:value for key,value in repeated.items() if key != 'logs'} == {
        key:value for key,value in result.items() if key != 'logs'}
    assert all(len(line) < 160 for line in result.get('logs', []))
    assert probes == before


def test_exact_fractional_probe_keys_fit_filename_limits():
    with mp.workdps(15):
        labels = [tasks._label(Fraction(1)+Fraction(n,10**900)) for n in (1,2)]
    assert labels[0] != labels[1]
    assert all(len('degeneracy_'+label) < 255 for label in labels)


@pytest.mark.parametrize('task,field', [(name,field) for name,field,_ in CASES if name != 'scalar_ope_max_scan'])
def test_interval_grid_is_required_during_submission(task,field):
    from virasoro_bootstrap.hpc.params import validate_sweep, task_signature

    assert task_signature(task).params[field].required is True
    errors = validate_sweep({'task':task,'params':{'c':'2','delta_phi':'.3'}})
    assert any(field in error and 'needs' in error for error in errors)
