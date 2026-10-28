"""Rational labels agree across direct and precomputed interpolation."""

from fractions import Fraction as F

import mpmath as mp
import pytest

from virasoro_blocks.interpolation import (
    lagrange_interpolation_polynomial, lagrange_polynomial_from_basis,
    precompute_lagrange_basis,
)


@pytest.mark.parametrize("rational_part", ["nodes", "values", "shift"])
@pytest.mark.parametrize("precomputed", [False, True])
def test_fraction_interpolation_inputs(rational_part, precomputed):
    with mp.workdps(60):
        nodes = [F(1, 3), F(5, 3)]
        values = [F(2, 7), F(8, 7)]
        shift = F(1, 9)
        if rational_part != "nodes":
            nodes = list(map(mp.convert, nodes))
        if rational_part != "values":
            values = list(map(mp.convert, values))
        if rational_part != "shift":
            shift = mp.convert(shift)
        if precomputed:
            basis, denoms, arb_basis, arb_denoms = precompute_lagrange_basis(nodes, shift=shift)
            result = lagrange_polynomial_from_basis(basis, denoms, values,
                _arb_basis=arb_basis, _arb_denoms=arb_denoms)
        else:
            result = lagrange_interpolation_polynomial(nodes, values, shift=shift)
        assert abs(result.evaluate(1 - mp.convert(shift)) - mp.mpf(5)/7) < mp.mpf('1e-58')


@pytest.mark.parametrize("precomputed", [False, True])
def test_fraction_values_with_complex_nodes(precomputed):
    with mp.workdps(60):
        nodes = [1j, 2 + 1j]
        values = [F(2, 7), F(8, 7)]
        if precomputed:
            basis, denoms, arb_basis, arb_denoms = precompute_lagrange_basis(nodes)
            result = lagrange_polynomial_from_basis(basis, denoms, values,
                _arb_basis=arb_basis, _arb_denoms=arb_denoms)
        else:
            result = lagrange_interpolation_polynomial(nodes, values)
        assert abs(result.evaluate(1 + 1j) - mp.mpf(5)/7) < mp.mpf('1e-58')


