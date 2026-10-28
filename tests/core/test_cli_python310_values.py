"""Option-looking scientific values have the same CLI meaning across Python versions."""
import pytest
from virasoro_blocks.__main__ import _build_parser
from virasoro_blocks.cli.evaluate import build_parser as evaluate_parser
from virasoro_blocks.cli.point_table import build_parser as point_parser
from virasoro_blocks.cli import join_option_values, RAMOND_BRANCHES


@pytest.mark.parametrize('factory,prefix', [(_build_parser, ['--lambda', '0']),
                                           (evaluate_parser, ['--point', 'z=.3']), (point_parser, ['--lambda', '0'])])
@pytest.mark.parametrize('phase', ['-i', '-I'])
def test_negative_imaginary_parity_value(factory, prefix, phase):
    args = factory().parse_args(prefix + ['--parity-ext', '1', '-1', 'i', phase])
    assert args.parity_ext == ['1', '-1', 'i', '-i']


@pytest.mark.parametrize('factory,prefix', [(_build_parser, ['--lambda', '0']), (evaluate_parser, ['--point', 'z=.3'])])
@pytest.mark.parametrize('tokens,expected', [(['--ramond-branch', '--'], '--'),
    (['--ramond-branch=--'], '--'), (['--ramond-branch', '-+'], '-+')])
def test_ramond_sign_branch_remains_a_string(factory, prefix, tokens, expected):
    args = factory().parse_args(join_option_values(prefix + tokens, '--ramond-branch', RAMOND_BRANCHES))
    assert args.ramond_branch == expected
