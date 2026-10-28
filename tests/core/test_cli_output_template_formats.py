"""JSON filename collision checks apply only when JSON files are emitted."""

import pytest

from virasoro_blocks.__main__ import main


@pytest.mark.parametrize("family", ["global", "ns"])
@pytest.mark.parametrize("output_format", ["m", "json", "both"])
def test_multi_spin_constant_json_template(tmp_path, family, output_format, capsys):
    argv = [
        "table", "--family", family, "--basis", "scalar_G", "--central-charge", "10",
        "--spin-ranges", "0", "2", "--lambda", "0", "--order", "2",
        "--npoles", "2", "--delta-ext", ".2", ".2", ".2", ".2",
        "--precision", "30", "--q-gate", "off",
        "--format", output_format, "--output-template", "fixed.json",
        "-o", str(tmp_path),
    ]
    if output_format != "m":
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert exc.value.code == 2
        assert "overwrite the same file" in capsys.readouterr().err
        assert not list(tmp_path.iterdir())
        return

    assert main(argv) in (None, 0)
    files = sorted(tmp_path.glob("*.m"))
    assert len(files) == 2
    assert all("zzbDeriv[0,0]" in path.read_text() for path in files)
    assert not list(tmp_path.glob("*.json"))
