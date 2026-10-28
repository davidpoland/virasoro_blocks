"""PMP builders must not reinterpret boolean derivative indices as integers."""
from fractions import Fraction
import numpy as np
import pytest
from virasoro_blocks.polynomial import Polynomial
from virasoro_blocks.serialization import SerializedDerivTable
from virasoro_bootstrap.pmp_writer import build_pmp_doc, build_prefactored_pmp_doc


def block():
    return SerializedDerivTable(0, .5, (), ('zzb',),
        {'zzb': [[Polynomial([2]), Polynomial([3])], [Polynomial([5]), Polynomial([7])]]})


@pytest.mark.parametrize('builder, pair', [
    (build_pmp_doc, (True, 0)), (build_prefactored_pmp_doc, (.5, 0)), (build_pmp_doc, (0,)),
])
def test_bad_derivative_indices_are_rejected(builder, pair):
    with pytest.raises(ValueError, match='F-pair'):
        builder([block()], [pair], objective=[0], normalization=[1])


@pytest.mark.parametrize('builder', [build_pmp_doc, build_prefactored_pmp_doc])
def test_integral_indices_select_the_requested_polynomials(builder):
    result = builder([block()], [(Fraction(1), 0.0), (0, np.int64(1))],
                     objective=[0, 0], normalization=[1, 0])
    doc = result if isinstance(result, dict) else result.doc
    values = doc['PositiveMatrixWithPrefactorArray'][0]['polynomials'][0][0]
    assert [[float(c) for c in poly] for poly in values] == [[5.0], [3.0]]
