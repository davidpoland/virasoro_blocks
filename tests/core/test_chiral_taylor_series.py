"""Compare singular-c chiral Taylor series with numerical differentiation.

The flint series for ``exp(h log E(u)) A(u) H(q(u))`` must agree with
numerical derivatives at c=1, on and off Kac weights, in all three chiral
coordinates.
"""
from fractions import Fraction as F

import mpmath as mp
import pytest

from virasoro_blocks import regularization as R
from virasoro_blocks.coordinates import COORDINATES
from virasoro_blocks.precision import taylor_coefficients
from virasoro_blocks.recursion import block_prefactor, q_from_z


def _numeric(h, H, h_ext, n_terms, z_of_u, u0, z_shift):
    def block(u):
        z = z_of_u(u)
        q = q_from_z(z)
        value = mp.mpc(0)
        for coefficient in reversed(H):
            value = value * q + coefficient
        return block_prefactor(h, z, h_ext, c=F(1), chiral_norm=16, z_shift=z_shift) * value

    return taylor_coefficients(block, u0, n_terms - 1)


@pytest.mark.parametrize("coord", ["z", "y", "q"])
@pytest.mark.parametrize("h", [F(5, 2), F(25, 4), F(9)])          # generic, and on the c = 1 Kac locus n^2/4
def test_series_chiral_taylor_equals_numerical_derivatives(coord, h):
    with mp.workdps(30):
        h_ext = [F(25, 16)] * 4
        z_shift = -2 * F(25, 16)
        spec = COORDINATES[coord]
        H = R.H_q_series_regularized(h, h_ext, c=F(1), q_order=16)
        new = R._regularized_chiral_taylor(h, h_ext, 9, c=F(1), q_order=16, z_of_u=spec.z_of_u,
                                           u_symmetric=spec.u_symmetric(), chiral_norm=16, z_shift=z_shift)
        old = _numeric(h, H, h_ext, 9, spec.z_of_u, spec.u_symmetric(), z_shift)
        assert len(new) == len(old) == 9
        for a, b in zip(new, old):
            assert abs(a - b) <= mp.mpf(10) ** -27 * abs(b)
