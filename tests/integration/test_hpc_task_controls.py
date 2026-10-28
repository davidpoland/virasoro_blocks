"""Cluster tasks validate integer, spin-list and boolean controls before engine dispatch."""
import pytest

from virasoro_bootstrap import identical_scalar, modular_crossing
from virasoro_bootstrap.hpc import tasks
from virasoro_bootstrap.hpc.journal import Journal


class ReachedEngine(BaseException):
    pass


@pytest.fixture
def task_context(tmp_path, monkeypatch):
    calls = []
    def stop(spec, *args, **kwargs):
        calls.append(spec)
        raise ReachedEngine
    for module, names in (
        (identical_scalar, ('prepare_identical_scalar_gap_build','build_identical_scalar_gap_problem',
                            'build_identical_scalar_ope_max_problem')),
        (modular_crossing, ('prepare_modular_gap_build','build_modular_gap_problem')),
    ):
        for name in names:
            monkeypatch.setattr(module, name, stop)
    params = dict(c='2',lam=7,dps=40,delta_ext='.7',delta_phi='.7',delta='.7',
        ope_squared='.1',delta_phi_grid=['.4','.8'],degeneracy_grid=['1','2'],
        source_output_dir=str(tmp_path/'source'),moment_cert_q_order=80)
    return tasks.TaskContext(params,tmp_path,Journal(tmp_path/'progress.json')),calls


@pytest.mark.parametrize('value', [True, 7.5, float('nan')])
def test_task_integer_wrapper_rejects_invalid_inputs(value):
    with pytest.raises(ValueError, match='lam'):
        tasks._task_integer('lam', value)


# Every task routes its integer controls through _task_integer; a few
# task/control pairs from the different task families cover the wiring.
@pytest.mark.parametrize('task,name', [
    ('scalar_gap_bisection', 'large_spin_npoles'),
    ('modular_scalar_gap_bisection', 'modular_spin_max'),
    ('isolated_relevant_scalar_grid_point', 'four_point_dps'),
    ('moment_recertify', 'cert_extra_spins'),
])
def test_fractional_controls_refused_before_engine(task_context, task, name):
    context,calls = task_context
    context.params[name] = 7.5
    with pytest.raises(ValueError, match=name):
        tasks.get_task(task)(context)
    assert calls == []


@pytest.mark.parametrize('task,name', [('scalar_gap_bisection', 'spins'),
                                       ('isolated_relevant_scalar_grid_point', 'modular_spins')])
def test_spin_lists_refuse_truncation_before_engine(task_context, task, name):
    context,calls = task_context
    context.params[name] = [0,2.5]
    with pytest.raises(ValueError):
        tasks.get_task(task)(context)
    assert calls == []


@pytest.mark.parametrize('validator,value', [(tasks._task_spins, [0,False]),
                                             (tasks._task_modular_spins, '024')])
def test_spin_list_wrappers_reject_booleans_and_scalar_strings(validator, value):
    with pytest.raises(ValueError):
        validator(value)


def test_ope_bounds_validate_directions_before_preparing(task_context):
    context,calls = task_context
    context.params['directions'] = ['lowre']
    with pytest.raises(ValueError, match='directions'):
        tasks.isolated_scalar_four_point_ope_bounds(context)
    assert calls == []


def test_integer_strings_and_integral_values_still_reach_engine(task_context):
    context,calls = task_context
    context.params.update(lam='7',dps='40',num_threads=1.0)
    with pytest.raises(ReachedEngine):
        tasks.scalar_only_gap_probe(context)
    assert (calls[0].lam,calls[0].dps,calls[0].num_threads) == (7,40,1)


@pytest.mark.parametrize('task', ['scalar_ope_max_scan'])
def test_zero_tower_pole_count_in_profile_is_preserved(task_context, monkeypatch, task):
    context,calls = task_context
    preset = dict(tasks.LAMBDA_PRESETS[7],large_spin_npoles=0)
    monkeypatch.setitem(tasks.LAMBDA_PRESETS,7,preset)
    with pytest.raises(ReachedEngine):
        tasks.get_task(task)(context)
    assert calls[0].large_spin_npoles == 0


@pytest.mark.parametrize('task', ['scalar_ope_max_scan','isolated_scalar_four_point_fixed_ope'])
def test_odd_exchange_spins_refused_before_preparation(task_context, task):
    context,calls = task_context
    context.params['spin_star'] = 3
    with pytest.raises(ValueError,match='spin_star'):
        tasks.get_task(task)(context)
    assert calls == []


@pytest.mark.parametrize('task', ['scalar_gap_bisection'])
def test_zero_tower_count_in_profile_preserves_gap_only_tower(task_context, monkeypatch, task):
    context,calls = task_context
    preset = dict(tasks.LAMBDA_PRESETS[7],large_spin_double_twists=0)
    monkeypatch.setitem(tasks.LAMBDA_PRESETS,7,preset)
    context.params.update(twist_gap='match_gap',all_spins_twist_gap=True)
    with pytest.raises(ReachedEngine):
        tasks.get_task(task)(context)
    assert len(calls[0].large_spin_twists) == 1
    assert calls[0].large_spin_twists[0] == calls[0].delta_gap


def test_modular_task_retains_its_lower_precision_range(task_context):
    context,calls = task_context
    context.params['dps'] = 1
    with pytest.raises(ReachedEngine):
        tasks.modular_scalar_gap_bisection(context)
    assert calls[0].dps == 1


def test_sphere_task_rejects_precision_below_backend_minimum(task_context):
    context,calls = task_context
    context.params['dps'] = 9
    with pytest.raises(ValueError,match='dps'):
        tasks.scalar_only_gap_probe(context)
    assert calls == []


@pytest.mark.parametrize('value', ['false', 2, None])
def test_task_boolean_validator_rejects_invalid_types(value):
    with pytest.raises(ValueError, match='flag'):
        tasks._task_boolean('flag', value)


@pytest.mark.parametrize('task,name', [('scalar_gap_bisection', 'tower'),
                                       ('moment_nu1_over_nu0', 'spot_check'),
                                       ('isolated_relevant_scalar_grid_point', 'run_four_point')])
def test_invalid_boolean_refused_before_engine(task_context, task, name):
    context,calls = task_context
    context.params[name] = 'false'
    with pytest.raises(ValueError,match=name):
        tasks.get_task(task)(context)
    assert calls == []


@pytest.mark.parametrize('value', [True, 0])
def test_supported_boolean_inputs_reach_engine(task_context,value):
    context,calls = task_context
    context.params['tower'] = value
    with pytest.raises(ReachedEngine):
        tasks.scalar_gap_bisection(context)
    assert bool(calls[0].large_spin_twists) is bool(value)
