"""Task entry points must preserve fixed dimension labels before engine dispatch."""
from fractions import Fraction

import pytest

from virasoro_blocks.numeric_input import ExactRational, literal_number
from virasoro_bootstrap import identical_scalar, modular_crossing
from virasoro_bootstrap.hpc import tasks
from virasoro_bootstrap.hpc.journal import Journal


class StopAtBuild(BaseException):
    pass


SHIFTED = str(ExactRational(7,10)+ExactRational(1,10**70))


@pytest.mark.parametrize('task,param,half,modular,label', [
    ('scalar_gap_bisection','h_ext',False,False,SHIFTED),
    ('isolated_scalar_four_point_ope_bounds','delta_ext',True,False,0.7),
    ('scalar_only_gap_probe','delta',True,False,SHIFTED),
    ('isolated_relevant_scalar_grid_point','delta_phi',False,True,SHIFTED),
])
def test_fixed_label_reaches_builder_exactly(tmp_path, monkeypatch, task, param, half, modular, label):
    captured = []
    def capture(spec, *args, **kwargs):
        captured.append(spec)
        raise StopAtBuild
    monkeypatch.setattr(identical_scalar, 'prepare_identical_scalar_gap_build', capture)
    monkeypatch.setattr(identical_scalar, 'build_identical_scalar_gap_problem', capture)
    monkeypatch.setattr(modular_crossing, 'prepare_modular_gap_build', lambda spec: object())
    monkeypatch.setattr(modular_crossing, 'build_modular_gap_problem', capture)
    params = dict(c='2',lam=7,dps=15,degeneracy_grid=['1','2'],ope_squared='.2')
    params[param] = label
    context = tasks.TaskContext(params, tmp_path, Journal(tmp_path/'progress.json'))
    with pytest.raises(StopAtBuild):
        tasks.get_task(task)(context)
    expected = Fraction(str(label)) / (2 if half else 1)
    actual = captured[0].isolated_scalars[0][0] if modular else captured[0].h_ext
    assert literal_number(actual) == expected
