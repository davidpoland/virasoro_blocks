"""SDP precision settings are positive integral digit counts."""
import json
import numpy as np
import pytest
from virasoro_bootstrap.sdp_writer import write_sdp_directory


def document():
    return {'objective': ['0'], 'normalization': ['1'],
        'PositiveMatrixWithPrefactorArray': [{'polynomials': [[[['1']]]],
            'samplePoints': ['0'], 'sampleScalings': ['1']}]}


# Check each writer argument's boundary; shared integer validators already
# exercise the complete numeric type matrix.
@pytest.mark.parametrize('name, value', [('eval_dps', True), ('output_digits', 0)])
def test_invalid_precision_fails_before_output(tmp_path, name, value):
    output = tmp_path / 'sdp'
    kwargs = dict(eval_dps=40, output_digits=30)
    kwargs[name] = value
    with pytest.raises(ValueError, match=name):
        write_sdp_directory(document(), output, **kwargs)
    assert not output.exists()


@pytest.mark.parametrize('value', [np.int64(30)])
def test_integral_numeric_precision_is_accepted(tmp_path, value):
    output = write_sdp_directory(document(), tmp_path / 'sdp', eval_dps=value, output_digits=value)
    assert json.loads((output / 'control.json').read_text())['num_blocks'] == 1
