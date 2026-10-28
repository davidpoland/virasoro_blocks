"""Polynomial truncation works on the declared python-flint minimum too."""
import pytest
from flint import arb_poly, acb_poly
from virasoro_blocks.polynomial import _truncate_flint_poly


@pytest.mark.parametrize('polynomial_type', [arb_poly, acb_poly])
@pytest.mark.parametrize('length', [0, 1, 2, 5])
def test_flint_truncation_keeps_requested_coefficients(polynomial_type, length):
    original = polynomial_type([1, 2, 3])
    truncated = _truncate_flint_poly(original, length)
    assert truncated.coeffs() == [1, 2, 3][:length]
    assert original.coeffs() == [1, 2, 3]
