"""Embedding the CLI with a one-shot argument iterable preserves the invocation."""

import json

import pytest

from virasoro_blocks.__main__ import main


@pytest.mark.parametrize("prefix", [[], ["table"]])
def test_table_iterator_matches_list(tmp_path, prefix):
    argv = prefix + [
        "--family", "global", "--basis", "scalar_G",
        "--spin-ranges", "0", "--lambda", "0", "--npoles", "2",
        "--delta-ext", ".7", ".7", ".7", ".7",
        "--precision", "30", "--format", "json", "-o", str(tmp_path),
    ]
    assert main(argv) in (None, 0)
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    expected = json.loads(files[0].read_text())
    files[0].unlink()

    assert main(iter(argv)) in (None, 0)
    assert json.loads(files[0].read_text()) == expected
