"""Selected sample regeneration validates its own outputs and rejects typos before writing."""
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys

import pytest


@pytest.fixture
def driver(tmp_path):
    if shutil.which('bash') is None:
        pytest.skip('sample regeneration requires bash')
    root = Path(__file__).resolve().parents[2]
    (tmp_path / 'tools').mkdir()
    script = tmp_path / 'tools/regen_block_samples.sh'
    shutil.copyfile(root / 'tools/regen_block_samples.sh', script)
    python = tmp_path / '.venv/bin/python'
    python.parent.mkdir(parents=True)
    # the fake interpreter is a /bin/sh wrapper around a Python body: a #! line cannot name an interpreter
    # whose path contains spaces (a virtualenv under a directory with spaces)
    body = python.with_name(python.name + '_body.py')
    body.write_text(f'''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if args[:2] == ['-m', 'virasoro_blocks']:
    with Path('calls').open('a') as stream:
        stream.write(json.dumps(args) + '\\n')
    if not os.environ.get('SKIP_SAMPLE_OUTPUT'):
        output = Path(args[args.index('-o') + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / 'sample.json').write_text(json.dumps({{'args': 'virasoro_blocks table --family stub'}}))
elif args[:1] == ['-']:
    os.execv(sys.executable, [sys.executable, *args])
else:
    raise SystemExit('unexpected interpreter invocation')
''')
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(body))} "$@"\n')
    python.chmod(0o755)
    def run(*targets, skip_output=False):
        return subprocess.run(['bash', str(script), *targets], text=True, capture_output=True,
                              env={**os.environ, 'SKIP_SAMPLE_OUTPUT': '1' if skip_output else ''})
    return tmp_path, run


def existing_sample(root, invocation):
    directory = root / 'examples/block_samples/unselected'
    directory.mkdir(parents=True)
    (directory / 'old.json').write_text(json.dumps({'args': invocation}))


def test_selected_regeneration_does_not_validate_unrelated_stale_files(driver):
    root, run = driver
    existing_sample(root, 'old --method invocation')
    result = run('scalar_G')
    assert result.returncode == 0, result.stderr
    assert 'validated 1 canonical JSON artifacts' in result.stdout


def test_missing_selected_output_cannot_be_hidden_by_other_samples(driver):
    root, run = driver
    existing_sample(root, 'virasoro_blocks table --family stub')
    result = run('scalar_G', skip_output=True)
    assert result.returncode != 0
    assert 'scalar_G' in result.stderr


def test_unknown_target_fails_before_any_sample_generation(driver):
    root, run = driver
    result = run('scalar_G', 'typo')
    assert result.returncode == 2
    assert 'unknown basis' in result.stderr
    assert not (root / 'calls').exists()


def test_default_run_generates_and_validates_all_dispatched_targets(driver):
    from virasoro_blocks.__main__ import _build_parser, _CLI_BASES
    from virasoro_blocks.cli.point_table import build_parser as build_point_parser

    root, run = driver
    result = run()
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in (root / 'calls').read_text().splitlines()]
    outputs = {args[args.index('-o') + 1] for args in calls}
    assert len(outputs) > 1
    assert all((root / output / 'sample.json').is_file() for output in outputs)
    assert f'validated {len(outputs)} canonical JSON artifacts' in result.stdout
    repository = Path(__file__).resolve().parents[2]
    for args in calls:
        assert args[:2] == ['-m', 'virasoro_blocks'] and args[2] in ('table', 'point-table')
        output = repository / args[args.index('-o') + 1]
        if args[2] == 'table':
            parsed = _build_parser().parse_args(args[3:])
            assert parsed.basis in _CLI_BASES
            committed = sorted(output.rglob('*.json'))
        else:
            # several point tables share one directory, each under its own output template
            build_point_parser().parse_args(args[3:])
            committed = [output / args[args.index('--output-template') + 1]]
        assert committed and all(path.is_file() for path in committed)
        for path in committed:
            invocation = json.loads(path.read_text())['args']
            assert shlex.split(invocation) == ['virasoro_blocks', *args[2:]], path
