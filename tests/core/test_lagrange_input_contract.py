"""Direct, precomputed and matrix interpolation agree for complex input."""

import mpmath as mp
import pytest

from virasoro_blocks.interpolation import (
    lagrange_matrix, lagrange_polynomial_from_basis, precompute_lagrange_basis,
)


@pytest.mark.parametrize("nodes,point", [([0, 1], .5 + 1j), ([0, 1j], 1 + .5j),
                                        ([mp.mpc(0), 1j], mp.mpc(1, .5))])
def test_matrix_preserves_native_complex_parts(nodes, point):
    rows = lagrange_matrix(nodes, [point])
    assert isinstance(rows, list)
    expected = [(mp.mpc(nodes[1]) - point) / (mp.mpc(nodes[1]) - nodes[0]),
                (point - mp.mpc(nodes[0])) / (mp.mpc(nodes[1]) - nodes[0])]
    assert rows[0] == expected


def test_matrix_materializes_iterators_once():
    from virasoro_blocks.polynomial import _arb_to_mpf
    matrix = lagrange_matrix(iter([0, 1]), iter([.25, 1]))
    assert [[_arb_to_mpf(matrix[i, j]) for j in range(2)] for i in range(2)] == [[.75, .25], [0, 1]]


@pytest.mark.parametrize("nodes", [[0, 0], [1j, 1j]])
def test_matrix_rejects_duplicate_nodes(nodes):
    with pytest.raises(ValueError, match="distinct"):
        lagrange_matrix(nodes, [.5])


@pytest.mark.parametrize("ball", [False, True])
@pytest.mark.parametrize("nodes", [[0, 1], [0, 1j]])
def test_precomputed_basis_accepts_complex_values(ball, nodes):
    from flint import acb
    with mp.workdps(40):
        basis, denoms, arb_basis, arb_denoms = precompute_lagrange_basis(nodes)
        values = [1 + 2j, 3 - 1j]
        if ball:
            values = [acb(v.real, v.imag) for v in values]
        polynomial = lagrange_polynomial_from_basis(basis, denoms, values,
            _arb_basis=arb_basis, _arb_denoms=arb_denoms)
        assert mp.almosteq(polynomial.evaluate(mp.mpc(nodes[1]) / 2), mp.mpc(2, '.5'))


def test_matrix_empty_evaluation_preserves_node_columns():
    from flint import arb_mat
    matrix = lagrange_matrix([0, 1], [])
    assert (matrix.nrows(), matrix.ncols()) == (0, 2)
    result = matrix * arb_mat([[2], [3]])
    assert (result.nrows(), result.ncols()) == (0, 1)


def test_precomputed_basis_reuses_its_matrix_without_retaining_discarded_fits():
    import gc
    import weakref
    from flint import ctx
    from virasoro_blocks.interpolation import _lagrange_basis_matrix

    with mp.workdps(40):
        basis, denoms, arb_basis, arb_denoms = precompute_lagrange_basis([0, 1, 2])
        reference = weakref.ref(arb_basis)
        matrix = _lagrange_basis_matrix(arb_basis, 2, ctx.prec)
        assert _lagrange_basis_matrix(arb_basis, 2, ctx.prec) is matrix
        for values in ([0, 1, 4], [1, 2, 5]):
            poly = lagrange_polynomial_from_basis(basis, denoms, values,
                _arb_basis=arb_basis, _arb_denoms=arb_denoms)
            assert mp.almosteq(poly.evaluate(mp.mpf('.5')), values[0] + mp.mpf('.25'))
        # Keeping the result and even the matrix must not keep the basis alive.
        del arb_basis
        gc.collect()
        assert reference() is None
