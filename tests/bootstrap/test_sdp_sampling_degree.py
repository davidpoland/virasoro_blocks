"""Direct SDP sampling must retain the requested polynomial constraints."""
import pytest
from virasoro_bootstrap.sdp_writer import write_sdp_directory


def document(coefficients):
    return {'objective': ['0'], 'normalization': ['1'],
            'PositiveMatrixWithPrefactorArray': [{'polynomials': [[[coefficients]]],
                'samplePoints': ['0', '1'], 'sampleScalings': ['1', '1']}]}


@pytest.mark.parametrize('clean', [False, True])
def test_sdp_rejects_under_sampled_polynomial_before_writing(tmp_path, clean):
    output = tmp_path / 'sdp'
    values = {0: [['1'], ['2']]} if clean else None
    with pytest.raises(ValueError, match='degree|sample'):
        write_sdp_directory(document(['1', '0', '0', '1']), output,
                            eval_dps=40, sample_values_by_block=values)
    assert not output.exists()


def test_sdp_accepts_zero_padding_beyond_actual_degree(tmp_path):
    output = write_sdp_directory(document(['1', '1', '0', '0']), tmp_path / 'sdp', eval_dps=40)
    assert (output / 'control.json').exists()


@pytest.mark.parametrize('key', ['0'])
def test_sdp_rejects_unknown_clean_sample_block(tmp_path, key):
    output = tmp_path / 'sdp'
    with pytest.raises(ValueError, match='block'):
        write_sdp_directory(document(['1', '1']), output, eval_dps=40,
                            sample_values_by_block={key: [['1'], ['2']]})
    assert not output.exists()
