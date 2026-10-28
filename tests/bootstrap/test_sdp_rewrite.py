"""Rewriting an SDP directory must not retain obsolete solver metadata."""
import json
import pytest
from virasoro_bootstrap.sdp_writer import write_sdp_directory


def document(count):
    return {'objective': ['0'], 'normalization': ['1'],
        'PositiveMatrixWithPrefactorArray': [
            {'polynomials': [[[['1']]]], 'samplePoints': ['0'], 'sampleScalings': ['1']}
            for _ in range(count)]}


def test_rewrite_removes_obsolete_generated_files_only(tmp_path):
    output = tmp_path / 'sdp'
    write_sdp_directory(document(2), output, eval_dps=30, pmp_info={'label': 'old problem'})
    note = output / 'notes.txt'
    note.write_text('retain this note')
    write_sdp_directory(document(1), output, eval_dps=30)
    assert not (output / 'pmp_info.json').exists()
    assert not (output / 'block_data_1.json').exists()
    assert not (output / 'block_info_1.json').exists()
    assert (output / 'block_data_0.json').exists()
    assert note.read_text() == 'retain this note'
    assert json.loads((output / 'control.json').read_text())['num_blocks'] == 1


def test_invalid_rewrite_preserves_previous_directory(tmp_path):
    import pytest
    output = tmp_path / 'sdp'
    write_sdp_directory(document(2), output, eval_dps=30, pmp_info={'label': 'old problem'})
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    invalid = document(1)
    invalid['normalization'] = ['0']
    with pytest.raises(ValueError):
        write_sdp_directory(invalid, output, eval_dps=30)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


@pytest.mark.parametrize('existing', [False, True])
def test_numerical_build_failure_does_not_publish_partial_sdp(tmp_path, monkeypatch, existing):
    from virasoro_bootstrap import sdp_writer
    output = tmp_path / 'sdp'
    if existing:
        write_sdp_directory(document(1), output, eval_dps=30, pmp_info={'label': 'original'})
    before = {p.name: p.read_bytes() for p in output.iterdir()} if existing else None
    original = sdp_writer._block_data
    calls = []
    def failing(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ArithmeticError('failed basis construction')
        return original(*args, **kwargs)
    monkeypatch.setattr(sdp_writer, '_block_data', failing)
    replacement = document(2)
    replacement['objective'] = ['5']
    with pytest.raises(ArithmeticError, match='basis construction'):
        write_sdp_directory(replacement, output, eval_dps=30)
    if existing:
        assert {p.name: p.read_bytes() for p in output.iterdir()} == before
    else:
        assert not output.exists()


def test_metadata_serialization_failure_preserves_old_sdp(tmp_path):
    output = tmp_path / 'sdp'
    write_sdp_directory(document(1), output, eval_dps=30)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(TypeError):
        write_sdp_directory(document(2), output, eval_dps=30, pmp_info={'invalid': object()})
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before
    assert not list(tmp_path.glob('.sdp-build-*'))
