"""User-supplied PMP sampling must represent the full polynomial vector."""
from copy import deepcopy
import pytest
from virasoro_blocks.polynomial import Polynomial
from virasoro_blocks.serialization import SerializedDerivTable
from virasoro_bootstrap.pmp_writer import build_pmp_doc


def build(sampling, degree=1):
    block = SerializedDerivTable(0, .5, (), ('zzb',),
        {'zzb': [[Polynomial([1] + [0]*(degree-1) + [1])]]})
    return build_pmp_doc([block], [(0, 0)], objective=[0], sampling_data_per_block=[sampling])


@pytest.mark.parametrize('damage', ['missing_points', 'short_scalings', 'duplicate', 'negative_scaling'])
def test_supplied_sampling_rejects_invalid_payload(damage):
    sampling = {'samplePoints': ['0', '1'], 'sampleScalings': ['1', '.5']}
    if damage.startswith('missing_'):
        sampling.pop('samplePoints' if damage == 'missing_points' else 'sampleScalings')
    elif damage == 'empty':
        sampling = {'samplePoints': [], 'sampleScalings': []}
    elif damage == 'short_scalings':
        sampling['sampleScalings'].pop()
    else:
        key, value = {
            'duplicate': ('samplePoints', '0'), 'negative_point': ('samplePoints', '-1'),
            'complex_point': ('samplePoints', '(1+1j)'), 'nan_point': ('samplePoints', 'nan'),
            'negative_scaling': ('sampleScalings', '-1'), 'infinite_scaling': ('sampleScalings', 'inf'),
        }[damage]
        sampling[key][1] = value
    with pytest.raises(ValueError, match='sampl'):
        build(sampling)


def test_insufficient_nodes_reject_polynomial_under_sampling():
    with pytest.raises(ValueError, match='degree|sample'):
        build({'samplePoints': ['0', '1'], 'sampleScalings': ['1', '.5']}, degree=3)


def test_extra_nodes_pad_polynomials_without_mutating_sampling():
    sampling = {'samplePoints': ['0', '1', '2'], 'sampleScalings': ['1', '.5', '.25']}
    original = deepcopy(sampling)
    doc = build(sampling)
    block = doc['PositiveMatrixWithPrefactorArray'][0]
    assert len(block['polynomials'][0][0][0]) == 3
    assert sampling == original


@pytest.mark.parametrize('key, basis', [('bilinearBasis_0', [['1', 'nan']]), ('bilinearBasis_1', [['1'], ['1']])])
def test_supplied_bilinear_basis_has_valid_shape_and_numbers(key, basis):
    sampling = {'samplePoints': ['0', '1'], 'sampleScalings': ['1', '.5'], key: basis}
    with pytest.raises(ValueError, match='bilinear'):
        build(sampling)


def test_supplied_bilinear_basis_is_preserved():
    sampling = {'samplePoints': ['0', '1'], 'sampleScalings': ['1', '.5'],
                'bilinearBasis_0': [['1']], 'bilinearBasis_1': [['1', '0']]}
    block = build(sampling)['PositiveMatrixWithPrefactorArray'][0]
    assert block['bilinearBasis_0'] == sampling['bilinearBasis_0']
    assert block['bilinearBasis_1'] == sampling['bilinearBasis_1']
