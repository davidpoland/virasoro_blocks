"""A new solver attempt cannot certify a dual or objective left by an earlier attempt."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from virasoro_bootstrap.certification import load_y_vector
from virasoro_bootstrap.hpc.recertify import incomplete_probe_reason
from virasoro_bootstrap.hpc.runner import SdpbRunner
from virasoro_bootstrap.hpc.storage import (
    SDP_SOURCE_NAME, record_sdp_source, recorded_sdp_source, recorded_sdp_state,
)
from virasoro_bootstrap.identical_scalar import read_sdpb_objective


RESULTS = ('y.txt', 'out.txt', 'x_0.txt', 'x_12.txt', 'X_matrix_0.txt', 'Y_matrix_24.txt')


def _document():
    return {'objective': ['0', '1'], 'normalization': ['1', '0'],
            'PositiveMatrixWithPrefactorArray': [{
                'polynomials': [[[['1'], ['2']]]], 'samplePoints': ['0'], 'sampleScalings': ['1']}]}


def _old_results(output):
    output.mkdir(parents=True, exist_ok=True)
    for name in RESULTS:
        (output / name).write_text('old result')
    (output / 'y.txt').write_text('1 1\n1\n')
    (output / 'out.txt').write_text('dualObjective = 1;\n')


@pytest.fixture(params=['sphere', 'sphere_nested', 'modular'])
def attempt(request, tmp_path, monkeypatch):
    from virasoro_bootstrap import identical_scalar, modular_crossing

    kind = request.param
    module = modular_crossing if kind == 'modular' else identical_scalar
    root = tmp_path / 'solve'
    output = root / ('outputs/custom' if kind == 'sphere_nested' else 'sdpb_out')
    source = root / ('inputs/custom' if kind == 'sphere_nested' else 'sdp')
    checkpoint = Path(str(source) + '.ck')
    doc = _document()
    problem = SimpleNamespace(spec=SimpleNamespace(dps=40), pmp_doc=doc,
                              pmp_build=SimpleNamespace(doc=doc), sample_values_by_block=None)
    runner = SdpbRunner(mode='native', bin_dir='/unused')
    monkeypatch.setattr(module.shutil, 'which', lambda name: str(name))

    def write_problem(problem, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / 'pmp.json').write_text(json.dumps(problem.pmp_build.doc))
        return path / 'manifest.json', path / 'pmp.json'

    if kind != 'modular':
        monkeypatch.setattr(module, 'write_identical_scalar_gap_problem', write_problem)

    def run(**kwargs):
        options = dict(runner=runner, sdpb_options={'--writeSolution': 'x'})
        options.update(kwargs)
        if kind == 'modular':
            return module.run_modular_gap_problem(problem, root, **options)
        return module.run_identical_scalar_gap_problem(
            problem, root, sdp_dir_name=str(source.relative_to(root)),
            sdpb_out_dir_name=str(output.relative_to(root)), **options)

    return SimpleNamespace(module=module, root=root, output=output, source=source,
                           checkpoint=checkpoint, doc=doc, kind=kind, run=run)


@pytest.mark.parametrize('outcome', ['complete_without_y', 'complete_with_y', 'failed', 'interrupted'])
def test_attempt_removes_old_results_and_preserves_hotstart(attempt, monkeypatch, outcome):
    root, output = attempt.root, attempt.output
    _old_results(output)
    (output / 'notes.txt').write_text('keep notes')
    (output / 'iterations.json').write_text('keep diagnostics')
    seed = attempt.checkpoint
    seed.mkdir(parents=True)
    (seed / 'checkpoint.0').write_text('saved iterate')

    def solve(command, log_path, *, check=True, cwd=None):
        assert not any((output / name).exists() for name in RESULTS)
        assert recorded_sdp_state(output) == 'running'
        assert (root / 'initial.ck/checkpoint.0').read_text() == 'saved iterate'
        assert seed.is_dir() and not list(seed.iterdir())
        log_path.write_text('-----found primal-dual optimal solution-----\n')
        if outcome == 'interrupted':
            raise RuntimeError('solver interrupted')
        if outcome == 'complete_with_y':
            (output / 'y.txt').write_text('1 1\n3\n')
            (output / 'out.txt').write_text('dualObjective = 3;\n')
        return 1 if outcome == 'failed' else 0

    monkeypatch.setattr(attempt.module, '_run_command', solve)
    if outcome == 'interrupted':
        with pytest.raises(RuntimeError, match='solver interrupted'):
            attempt.run(initial_checkpoint_dir=seed)
    else:
        result = attempt.run(initial_checkpoint_dir=seed)
        assert result.checkpoint_dir is None
    assert (output / 'notes.txt').read_text() == 'keep notes'
    assert (output / 'iterations.json').read_text() == 'keep diagnostics'
    if outcome.startswith('complete'):
        assert recorded_sdp_state(output) == 'completed'
        assert recorded_sdp_source(output).resolve() == attempt.source.resolve()
    else:
        assert recorded_sdp_state(output) == ('failed' if outcome == 'failed' else 'running')
        with pytest.raises(ValueError, match='no completed attempt'):
            recorded_sdp_source(output)
    if outcome == 'complete_with_y':
        assert load_y_vector(output, full=True) == [1, 3]
        assert read_sdpb_objective(output)['dual'] == 3
    else:
        with pytest.raises(FileNotFoundError):
            load_y_vector(output, full=True)
        assert read_sdpb_objective(output) is None
        if output.name == 'sdpb_out':
            assert incomplete_probe_reason(root) is not None


def test_preparation_only_invalidates_results_and_keeps_checkpoint(attempt, monkeypatch):
    _old_results(attempt.output)
    attempt.checkpoint.mkdir(parents=True)
    (attempt.checkpoint / 'checkpoint.0').write_text('saved iterate')
    monkeypatch.setattr(attempt.module, '_run_command', lambda *a, **kw: pytest.fail('solver launched'))
    result = attempt.run(run_sdpb=False)
    assert result.sdpb_out_dir is None and result.checkpoint_dir is None
    assert not any((attempt.output / name).exists() for name in RESULTS)
    assert recorded_sdp_state(attempt.output) == 'prepared'
    with pytest.raises(ValueError, match='no completed attempt'):
        recorded_sdp_source(attempt.output)
    assert (attempt.checkpoint / 'checkpoint.0').read_text() == 'saved iterate'


@pytest.mark.parametrize('failure', ['checkpoint', 'executable'])
def test_preflight_failure_preserves_completed_inputs_and_results(attempt, monkeypatch, failure):
    attempt.run(run_sdpb=False)
    _old_results(attempt.output)
    record_sdp_source(attempt.source, attempt.output, state='completed')
    before = {p.relative_to(attempt.root): p.read_bytes()
              for p in attempt.root.rglob('*') if p.is_file()}
    attempt.doc['objective'] = ['0', '3']
    options = {}
    if failure == 'checkpoint':
        options['initial_checkpoint_dir'] = attempt.root / 'missing.ck'
    else:
        monkeypatch.setattr(attempt.module.shutil, 'which', lambda name: None)
    monkeypatch.setattr(attempt.module, '_run_command', lambda *a, **kw: pytest.fail('solver launched'))
    with pytest.raises(FileNotFoundError):
        attempt.run(**options)
    assert before == {p.relative_to(attempt.root): p.read_bytes()
                      for p in attempt.root.rglob('*') if p.is_file()}
    assert recorded_sdp_state(attempt.output) == 'completed'
    assert read_sdpb_objective(attempt.output)['dual'] == 1


@pytest.mark.parametrize('failure', ['pmp', 'sdp', 'pmp2sdp'])
def test_preparation_failure_cannot_reuse_an_old_solution(attempt, monkeypatch, failure):
    if failure == 'pmp2sdp' and attempt.kind == 'modular':
        pytest.skip('modular runner uses the direct SDP writer')
    attempt.run(run_sdpb=False)
    _old_results(attempt.output)
    record_sdp_source(attempt.source, attempt.output, state='completed')
    attempt.doc['objective'] = ['0', '3']

    def assert_invalidated():
        assert recorded_sdp_state(attempt.output) == 'preparing'
        assert not any((attempt.output / name).exists() for name in RESULTS)

    if failure == 'pmp2sdp':
        def convert(command, *args, **kwargs):
            assert 'pmp2sdp' in Path(command[0]).name
            assert_invalidated()
            raise RuntimeError('preparation failed')
        monkeypatch.setattr(attempt.module, '_run_command', convert)
        options = {'backend': 'pmp2sdp'}
    else:
        name = ('write_sdp_directory' if failure == 'sdp' else
                'write_pmp_json' if attempt.kind == 'modular' else 'write_identical_scalar_gap_problem')
        writer = getattr(attempt.module, name)

        def fail(*args, **kwargs):
            assert_invalidated()
            writer(*args, **kwargs)
            raise RuntimeError('preparation failed')
        monkeypatch.setattr(attempt.module, name, fail)
        monkeypatch.setattr(attempt.module, '_run_command', lambda *a, **kw: pytest.fail('solver launched'))
        options = {}
    with pytest.raises(RuntimeError, match='preparation failed'):
        attempt.run(**options)
    assert json.loads((attempt.root / 'pmp.json').read_text())['objective'] == ['0', '3']
    assert recorded_sdp_state(attempt.output) == 'preparing'
    with pytest.raises(ValueError, match='no completed attempt'):
        recorded_sdp_source(attempt.output)
    with pytest.raises(FileNotFoundError):
        load_y_vector(attempt.output, full=True)
    assert read_sdpb_objective(attempt.output) is None
    if attempt.output.name == 'sdpb_out':
        assert incomplete_probe_reason(attempt.root) is not None


@pytest.mark.parametrize('hotstart', [False, True])
def test_checkpoint_follows_input_directory_and_is_only_loaded_explicitly(attempt, monkeypatch, hotstart):
    checkpoint = attempt.checkpoint
    checkpoint.mkdir(parents=True)
    (checkpoint / 'checkpoint.0').write_text('previous iterate')

    def solve(command, log_path, *, check=True, cwd=None):
        # SDPB appends .ck to the -s argument, including a nested input path.
        target = cwd / (command[command.index('-s') + 1] + '.ck')
        assert target == checkpoint
        if hotstart:
            assert '--initialCheckpointDir' in command
            assert (cwd / 'initial.ck/checkpoint.0').read_text() == 'previous iterate'
            assert target.is_dir() and not list(target.iterdir())
        else:
            assert '--initialCheckpointDir' not in command
            assert not target.exists()
            target.mkdir()
        (target / 'checkpoint.0').write_text('new iterate')
        log_path.write_text('-----found dual feasible solution-----\n')
        return 0

    monkeypatch.setattr(attempt.module, '_run_command', solve)
    result = attempt.run(initial_checkpoint_dir=checkpoint if hotstart else None)
    assert result.checkpoint_dir == checkpoint
    assert (checkpoint / 'checkpoint.0').read_text() == 'new iterate'


def test_cleanup_failure_prevents_launch_and_leaves_attempt_incomplete(attempt, monkeypatch):
    _old_results(attempt.output)
    unlink = Path.unlink

    def fail(path, *args, **kwargs):
        if path == attempt.output / 'y.txt':
            raise OSError('cannot remove old dual')
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', fail)
    monkeypatch.setattr(attempt.module, '_run_command', lambda *a, **kw: pytest.fail('solver launched'))
    with pytest.raises(OSError, match='cannot remove old dual'):
        attempt.run()
    assert recorded_sdp_state(attempt.output) == 'preparing'
    with pytest.raises(ValueError, match='no completed attempt'):
        recorded_sdp_source(attempt.output)


def _source(tmp_path):
    source = tmp_path / 'sdp'
    source.mkdir()
    (source / 'control.json').write_text('{}')
    (source / 'normalization.json').write_text('{"normalization":["1","0"]}')
    return source


def test_result_symlinks_are_unlinked_without_touching_targets(tmp_path):
    source, output = _source(tmp_path), tmp_path / 'out'
    output.mkdir()
    target = tmp_path / 'keep.txt'
    target.write_text('keep')
    (output / 'y.txt').symlink_to(target)
    (output / 'out.txt').symlink_to(tmp_path / 'missing')
    record_sdp_source(source, output, state='running')
    assert not (output / 'y.txt').is_symlink() and not (output / 'out.txt').is_symlink()
    assert target.read_text() == 'keep'


def test_unexpected_result_directory_is_preserved_and_attempt_stays_incomplete(tmp_path):
    source, output = _source(tmp_path), tmp_path / 'out'
    unexpected = output / 'y.txt'
    unexpected.mkdir(parents=True)
    (unexpected / 'keep.txt').write_text('keep')
    with pytest.raises(OSError):
        record_sdp_source(source, output, state='running')
    assert (unexpected / 'keep.txt').read_text() == 'keep'
    assert recorded_sdp_state(output) == 'running'


def test_record_symlink_is_refused_before_results_are_removed(tmp_path):
    source, output = _source(tmp_path), tmp_path / 'out'
    _old_results(output)
    target = tmp_path / 'record.json'
    target.write_text('{}')
    (output / SDP_SOURCE_NAME).symlink_to(target)
    with pytest.raises(ValueError, match='symlinked SDP source record'):
        record_sdp_source(source, output, state='running')
    assert (output / 'y.txt').read_text() == '1 1\n1\n'
    assert target.read_text() == '{}'
