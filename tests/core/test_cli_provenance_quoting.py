"""Recorded commands retain exact argument boundaries and a usable entry point."""

import json
import shlex
import sys

import pytest

from virasoro_blocks.__main__ import main


@pytest.mark.parametrize("prefix", [
    ["table", "--family", "global", "--basis", "scalar_G", "--npoles", "2"],
    ["table", "--family", "ns", "--central-charge", "2", "--order", "2", "--npoles", "2", "--q-gate", "off"],
    ["table", "--surface", "torus", "--central-charge", "2"],
    ["table", "--family", "global", "--basis", "scalar_G", "--large-spin-lmin", "2",
     "--large-spin-twists", "1", "--npoles", "2"],
    ["point-table", "--family", "global", "--basis", "scalar_G", "--exchange-delta", "1"],
])
def test_artifact_commands_quote_output_paths(prefix, tmp_path):
    output = tmp_path / "a path's $label"
    argv = prefix + ["--lambda", "0", "--precision", "25", "-o", str(output)]
    assert main(argv) in (None, 0)
    files = list(output.glob("*.json"))
    assert files
    for path in files:
        assert shlex.split(json.loads(path.read_text())["args"]) == ["virasoro_blocks", *argv]


def test_module_invocation_records_canonical_console_command(tmp_path, monkeypatch):
    argv = ["table", "--family", "global", "--basis", "scalar_G", "--lambda", "0",
            "--npoles", "2", "--precision", "25", "-o", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", ["/installed/virasoro_blocks/__main__.py", *argv])
    assert main() in (None, 0)
    doc = json.loads((tmp_path / "derivs_0.json").read_text())
    assert shlex.split(doc["args"]) == ["virasoro_blocks", *argv]
