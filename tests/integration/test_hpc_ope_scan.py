"""OPE-scan task bookkeeping, without invoking the numerical engine or SDPB."""

from types import SimpleNamespace
from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_bootstrap import identical_scalar
from virasoro_bootstrap.hpc.journal import Journal
from virasoro_bootstrap.hpc.tasks import (
    TaskContext, _label, isolated_scalar_four_point_ope_bounds, result_error, scalar_ope_max_scan,
)


@pytest.fixture
def scan(tmp_path, monkeypatch):
    state = {"objective": {"dual": mp.mpf(-2)}, "runs": [],
             "real_build": identical_scalar.build_identical_scalar_ope_max_problem,
             "real_prepare": identical_scalar.prepare_identical_scalar_gap_build}
    monkeypatch.setattr(identical_scalar, "prepare_identical_scalar_gap_build", lambda spec: object())
    monkeypatch.setattr(identical_scalar, "build_identical_scalar_ope_max_problem",
                        lambda spec, **kw: SimpleNamespace(delta=kw["delta_star"]))

    def run(problem, output_dir, **kwargs):
        state["runs"].append(problem.delta)
        return SimpleNamespace(sdpb_status="optimal" if problem.delta < 1 else "max_complementarity",
                               sdpb_out_dir=output_dir, sdpb_log_path=output_dir / "sdpb.log")

    monkeypatch.setattr(identical_scalar, "run_identical_scalar_gap_problem", run)
    monkeypatch.setattr(identical_scalar, "read_sdpb_objective", lambda path: state["objective"])
    context = TaskContext(
        params={"c": "2", "lam": 7, "spins": [0], "tower": False, "delta_stars": [".5", "1"]},
        output_dir=tmp_path, journal=Journal(tmp_path / "progress.json"),
    )
    return context, state


@pytest.mark.parametrize("objective", [{"primal": 1}, {"dual": mp.nan}])
def test_optimal_scan_requires_a_finite_objective(scan, objective):
    context, state = scan
    state["objective"] = objective
    with pytest.raises(ValueError, match="finite dual objective"):
        scalar_ope_max_scan(context)
    assert _label(mp.mpf(".5")) not in context.journal.data.get("points", {})


@pytest.mark.parametrize("bound", [None, "nan"])
def test_scan_resume_retries_optimal_entries_without_a_valid_bound(scan, bound):
    context, state = scan
    entry = {"status": "optimal"}
    if bound is not None:
        entry["ope_squared_bound"] = bound
    context.journal.update(points={_label(mp.mpf(".5")): entry,
                                   _label(mp.mpf(1)): {"status": "max_complementarity"}})
    result = scalar_ope_max_scan(context)
    assert state["runs"] == [mp.mpf(".5")]
    assert mp.mpf(result["gap_bound_lo"]) == mp.mpf(".5")
    assert mp.mpf(result["gap_bound_hi"]) == 1
    assert mp.mpf(result["points"][_label(mp.mpf(".5"))]["ope_squared_bound"]) == 2


def test_scan_resume_keeps_completed_valid_points(scan):
    context, state = scan
    result = scalar_ope_max_scan(context)
    assert len(state["runs"]) == 2
    repeated = scalar_ope_max_scan(context)
    assert len(state["runs"]) == 2
    assert repeated["points"] == result["points"]
    points = dict(context.journal.data['points'])
    points[_label(mp.mpf('.5'))] = None
    context.journal.update(points=points)
    retried = scalar_ope_max_scan(context)
    assert state['runs'] == [mp.mpf('.5'), mp.mpf(1), mp.mpf('.5')]
    assert retried['points'] == result['points']


def test_scan_does_not_report_an_inverted_gap_bracket(scan):
    context, state = scan
    points = {_label(mp.mpf(".5")): {"status": "max_complementarity"},
              _label(mp.mpf(1)): {"status": "optimal", "ope_squared_bound": "2"}}
    context.journal.update(points=points)
    result = scalar_ope_max_scan(context)
    assert result["gap_bound_lo"] is None
    assert result["gap_bound_hi"] is None
    assert result["reason"] == "inconsistent_scan"
    assert result_error(result) is None  # Inconclusive numerical outcome, not an execution failure.
    assert result["points"] == points
    assert state["runs"] == []


def test_scan_bracket_does_not_depend_on_requested_point_order(scan):
    context, _ = scan
    context.params["delta_stars"] = ["1", ".5"]
    result = scalar_ope_max_scan(context)
    assert mp.mpf(result["gap_bound_lo"]) == mp.mpf(".5")
    assert mp.mpf(result["gap_bound_hi"]) == 1


def test_scan_keeps_nearby_exact_points_and_serializes_their_labels(scan):
    context,state = scan
    first = Fraction(7,10)
    second = first+Fraction(1,10**70)
    context.params.update(dps=15,delta_stars=[str(first),str(second)])
    result = scalar_ope_max_scan(context)
    assert state['runs'] == [first,second]
    assert len(result['points']) == 2
    assert {Fraction(entry['delta_star']) for entry in result['points'].values()} == {first,second}
    assert all(len('dstar_'+key) < 255 for key in result['points'])
    assert all(len(line) < 160 for line in result['logs'])
    scalar_ope_max_scan(context)
    assert state['runs'] == [first,second]


def test_twist_gap_normalizes_spinning_operator_at_its_own_continuum_edge(scan,monkeypatch):
    from virasoro_blocks.numeric_input import literal_number
    context,state = scan
    tiny = Fraction(1, 10**70)
    context.params.update(spin_star=2,spins=[0,2],all_spins_twist_gap=True,
                          dps=30,delta_stars=[str(tiny), '.35'])
    def build(spec,**kwargs):
        assert literal_number(kwargs['delta_star']) == literal_number(spec.delta_min_by_spin[2])
        assert kwargs['delta_star'] >= kwargs['spin_star']
        return SimpleNamespace(delta=spec.delta_gap)
    monkeypatch.setattr(identical_scalar,'build_identical_scalar_ope_max_problem',build)
    result = scalar_ope_max_scan(context)
    assert {Fraction(entry['operator_delta']) for entry in result['points'].values()} == {2+tiny,Fraction(47,20)}


@pytest.mark.parametrize('params', [dict(spin_star=2),dict(spin_star=2,spins=[0,2])])
def test_invalid_spinning_scan_refused_before_preparation(scan,monkeypatch,params):
    context,_ = scan
    context.params.update(params)
    def prepare(spec):
        pytest.fail('invalid distinguished operator reached block preparation')
    monkeypatch.setattr(identical_scalar,'prepare_identical_scalar_gap_build',prepare)
    with pytest.raises(ValueError,match='spin_star|delta_stars'):
        scalar_ope_max_scan(context)


@pytest.mark.parametrize('old_dimension', [None,'.5'])
def test_spinning_twist_scan_retries_journal_entries_for_wrong_operator(scan,monkeypatch,old_dimension):
    context,state = scan
    context.params.update(spin_star=2,spins=[0,2],all_spins_twist_gap=True)
    monkeypatch.setattr(identical_scalar,'build_identical_scalar_ope_max_problem',
                        lambda spec,**kwargs: SimpleNamespace(delta=spec.delta_gap))
    previous = {'status':'optimal','ope_squared_bound':'2'}
    if old_dimension is not None:
        previous['operator_delta'] = old_dimension
    context.journal.update(points={_label(mp.mpf('.5')):previous})
    result = scalar_ope_max_scan(context)
    assert state['runs'] == [Fraction(1,2),Fraction(1)]
    assert result['points'][_label(mp.mpf('.5'))]['operator_delta'] == '2.5'
    scalar_ope_max_scan(context)
    assert state['runs'] == [Fraction(1,2),Fraction(1)]


def test_spinning_twist_scan_constructs_a_valid_real_backend_problem(scan,monkeypatch):
    import warnings
    from virasoro_bootstrap.hpc import tasks

    context,state = scan
    monkeypatch.setitem(tasks.LAMBDA_PRESETS,1,tasks.LAMBDA_PRESETS[7])
    context.params.update(lam=1,dps=40,q_order=8,npoles=4,num_threads=1,
                          spin_star=2,spins=[0,2],all_spins_twist_gap=True,delta_stars=['.5'])
    monkeypatch.setattr(identical_scalar,'build_identical_scalar_ope_max_problem',state['real_build'])
    monkeypatch.setattr(identical_scalar,'prepare_identical_scalar_gap_build',state['real_prepare'])
    def run(problem,output_dir,**kwargs):
        assert problem.normalization_operator['spin_star'] == 2
        assert problem.normalization_operator['delta_star'] == mp.mpf('2.5')
        assert next(ch for ch in problem.spin_channels if ch.ell == 2).delta_min == mp.mpf('2.5')
        return SimpleNamespace(sdpb_status='optimal',sdpb_out_dir=output_dir,
                               sdpb_log_path=output_dir/'sdpb.log')
    monkeypatch.setattr(identical_scalar,'run_identical_scalar_gap_problem',run)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        result = scalar_ope_max_scan(context)
    assert result['points'][_label(mp.mpf('.5'))]['operator_delta'] == '2.5'


@pytest.fixture
def bounds(scan, monkeypatch):
    context, state = scan
    context.params.update(delta_ext=".5", four_point_spins=[0], four_point_large_spin=False)
    monkeypatch.setattr(identical_scalar, "build_identical_scalar_ope_bound_problem",
                        lambda spec, **kw: SimpleNamespace(delta=kw["delta_star"]))
    return context, state


@pytest.mark.parametrize("direction, objective", [("lower", None), ("upper", {"dual": mp.nan})])
def test_four_point_optimal_bound_requires_a_finite_objective(bounds, direction, objective):
    context, state = bounds
    context.params["directions"] = [direction]
    state["objective"] = objective
    with pytest.raises(ValueError, match="finite dual objective"):
        isolated_scalar_four_point_ope_bounds(context)
    assert direction not in context.journal.data.get("solved", {})


@pytest.mark.parametrize("direction", ["upper"])
def test_four_point_resume_retries_incomplete_optimal_bound(bounds, direction):
    context, state = bounds
    context.params["directions"] = [direction]
    context.journal.update(solved={direction: {"status": "optimal", "ope_squared_bound": None}})
    result = isolated_scalar_four_point_ope_bounds(context)
    assert len(state["runs"]) == 1
    assert mp.isfinite(mp.mpf(result[direction]["ope_squared_bound"]))
    assert isolated_scalar_four_point_ope_bounds(context) == result
    assert len(state["runs"]) == 1
