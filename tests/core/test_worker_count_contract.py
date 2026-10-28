"""Process-pool controls are validated independently of numerical work."""

from fractions import Fraction
import warnings

import mpmath as mp
import numpy as np
import pytest

from virasoro_blocks import global_blocks, parallel, polyderivs
from virasoro_blocks.requests import RequestValidationError
from virasoro_blocks.super_table_backends import SuperBackendOptions


BAD = [True, np.bool_(False), 0, -1, 2.5, Fraction(5, 2), '2', None,
       float('nan'), float('inf'), complex(2, 0)]


@pytest.mark.parametrize('value', BAD)
def test_worker_count_validator_rejects_invalid_types_and_bounds(value):
    with pytest.raises(RequestValidationError, match='num_threads'):
        parallel.validate_worker_count(value)


# Type coverage belongs to the shared validator above. Each caller still has
# regressions for truncation, bool-as-int, and its positive lower bound.
@pytest.mark.parametrize('value', [2.5, True, 0])
@pytest.mark.parametrize('path', ['global', 'virasoro', 'super'])
def test_invalid_worker_counts_fail_before_interpolation(value, path):
    with pytest.raises(RequestValidationError, match='num_threads'):
        if path == 'global':
            # None numerical collaborators cannot be touched before validation.
            global_blocks._interpolate_global_table(
                sampler=None, m_max=1, n_max=0, base=None, shift=None, pool=None,
                residue_scorer=None, entry_power=None, npoles=0,
                pole_selection='rightmost', num_threads=value)
        elif path == 'virasoro':
            polyderivs.interpolate_delta_tables({}, 0, num_threads=value)
        else:
            SuperBackendOptions(num_threads=value)


@pytest.mark.parametrize('value', [2, 2.0, Fraction(2), mp.mpf(2), np.int64(2), np.float64(2)])
def test_integral_worker_counts_are_normalized(monkeypatch, value):
    monkeypatch.setattr(parallel, '_main_script_unguarded', lambda: False)
    normalized = parallel.usable_workers(value)
    assert normalized == 2 and type(normalized) is int
    options = SuperBackendOptions(num_threads=value)
    assert options.num_threads == 2 and type(options.num_threads) is int


def test_a_stored_subprecision_worker_offset_is_not_integral():
    with mp.workdps(150):
        value = mp.mpf(2) + mp.mpf('1e-100')
    with mp.workdps(15), pytest.raises(RequestValidationError, match='num_threads'):
        parallel.usable_workers(value)


def test_global_integral_float_count_matches_serial_interpolation(monkeypatch):
    # Real pools exercise the accepted control all the way into multiprocessing.
    monkeypatch.setattr(parallel, '_main_script_unguarded', lambda: False)
    with mp.workdps(30):
        kwargs = dict(q_order=2, npoles=0, pole_selection='rightmost', shift='.5')
        serial = global_blocks.global_scalar_G_interpolated_table(
            'zzb', 0, ('.1',)*4, 1, 0, num_threads=1, **kwargs)
        workers = global_blocks.global_scalar_G_interpolated_table(
            'zzb', 0, ('.1',)*4, 1, 0, num_threads=2.0, **kwargs)
        assert [[p.coeffs for p in row] for row in serial.numerators] == [
            [p.coeffs for p in row] for row in workers.numerators]


@pytest.mark.parametrize('method', ['spawn', 'forkserver'])
def test_unavailable_main_script_still_falls_back_once(monkeypatch, method):
    monkeypatch.setattr(parallel.multiprocessing, 'get_start_method', lambda **kw: method)
    monkeypatch.setattr(parallel, '_main_script_unguarded', lambda: True)
    monkeypatch.setattr(parallel, '_WARNED', False)
    with pytest.warns(RuntimeWarning, match='running serially'):
        assert parallel.usable_workers(2.0) == 1
    # The existing one-time warning policy survives strict input validation.
    with warnings.catch_warnings(record=True) as caught:
        assert parallel.usable_workers(3) == 1
    assert not caught


def test_bootstrapping_workers_cannot_start_a_nested_pool(monkeypatch):
    monkeypatch.setattr(parallel.multiprocessing.current_process(), '_inheriting', True, raising=False)
    assert parallel.usable_workers(2) == 1


def test_daemon_workers_use_serial_execution_before_start_method_checks(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(parallel.multiprocessing, 'current_process', lambda: SimpleNamespace(daemon=True))
    monkeypatch.setattr(parallel.multiprocessing, 'get_start_method',
                        lambda **kw: pytest.fail('daemon workers cannot create children under any start method'))
    assert parallel.usable_workers(4) == 1


@pytest.mark.parametrize('value', [0, -1])
def test_serial_fallback_preserves_nonpositive_certification_controls(value):
    # The certification API documents <= 1 as serial and clamps the result.
    assert max(1, parallel.usable_workers(value)) == 1


def test_unguarded_scripts_run_serially(tmp_path, monkeypatch):
    """An unguarded script runs serially when multiprocessing uses spawn."""
    import sys
    import types
    import warnings

    from virasoro_blocks import parallel

    monkeypatch.setattr(parallel.multiprocessing, "get_start_method", lambda allow_none=False: "spawn")
    monkeypatch.setattr(parallel, "_WARNED", False)
    for text, expected in (
        ("run()\n", 1),
        ('# if __name__ == "__main__":\nrun()\n', 1),
        ('"""Use if __name__ == "__main__" around the build."""\nrun()\n', 1),
        ('def example():\n    if __name__ == "__main__":\n        run()\n', 1),
        ('class Example:\n    if __name__ == "__main__":\n        run()\n', 1),
        ('if __name__ == "__main__" or enabled:\n    run()\n', 1),
        ('if __name__ == "__main__":\n    run()\n', 4),
        ('if ("__main__" == __name__):\n    run()\n', 4),
        ('if __name__ == "__main__" and enabled:\n    run()\n', 4),
        ('if enabled:\n    if __name__ == "__main__":\n        run()\n', 4),
        ('\ufeffif __name__ == "__main__":\n    run()\n', 4),
        ('if broken syntax:\n    run()\n', 1),
    ):
        script = tmp_path / "script.py"
        script.write_text(text)
        monkeypatch.setitem(sys.modules, "__main__", types.SimpleNamespace(__file__=str(script)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert parallel.usable_workers(4) == expected
    script.write_bytes(b'# coding: latin-1\ncaf\xe9 = 1\nif __name__ == "__main__":\n    run()\n')
    assert parallel.usable_workers(4) == 4
    monkeypatch.setitem(sys.modules, "__main__", types.SimpleNamespace())      # interactive / notebook
    assert parallel.usable_workers(4) == 4


def test_unavailable_or_unguarded_scripts_fall_back_before_spawn_reloads_them(tmp_path):
    import subprocess
    import sys

    script = '''
# Put if __name__ == "__main__" around the build.
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
from virasoro_blocks.parallel import usable_workers
if len(sys.argv) > 1:
    with Path(sys.argv[1]).open('a') as handle:
        handle.write('top-level execution\\n')
multiprocessing.set_start_method("spawn", force=True)
workers = usable_workers(2)
if workers == 1:
    print(pow(2, 3))
else:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        print(pool.submit(pow, 2, 3).result())
'''
    source = tmp_path / 'fake_guard.py'
    source.write_text(script)
    marker = tmp_path / 'executions.txt'
    for command, stdin in (([sys.executable, '-'], script),
                           ([sys.executable, str(source), str(marker)], None)):
        result = subprocess.run(command, input=stdin, text=True,
                                capture_output=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "8"
        assert "running serially" in result.stderr
    assert marker.read_text().splitlines() == ['top-level execution']
