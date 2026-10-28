"""Test the Ising crossing equation using Virasoro blocks.

The 2d Ising model <σσσσ> correlator has a known OPE:
- Virasoro identity (h=0): C²=1
- Virasoro epsilon (h=1/2): C²_σσε = 1/4

The crossing equation for the library blocks (z_shift = zbar_shift = 0) is:
    G(z, z̄) = ((z z̄) / ((1-z)(1-z̄)))^{Δ_ext} × G(1-z, 1-z̄)

On the diagonal z = z̄ (real section), this reduces to:
    G(z) = (z/(1-z))^{2 Δ_ext} × G(1-z)

For the Ising model, Δ_ext = 2 h_σ = 1/8, so the exponent is 1/4.

The asymmetric prefactor arises because the block includes a (1-z)^{σ-d₂-d₃}
factor in its definition, making it not manifestly crossing-symmetric.

This test verifies the Virasoro blocks (with the default z_shift=0 convention
giving F ~ z^h/N^h) satisfy the Ising crossing equation.
"""

from __future__ import annotations

import mpmath as mp
import pytest

from virasoro_blocks.regularization import F_block_regularized


C_ISING = mp.mpf("0.5")
H_SIGMA = mp.mpf("0.0625")  # h_ext = 1/16
H_EPSILON = mp.mpf("0.5")  # h_ε = 1/2
C2_EPSILON = mp.mpf("0.25")  # C²_σσε = 1/4


def _ising_correlator(z, q_order=20):
    """Compute g(z) = Σ C²_eff F_lib(h, z)² for the Ising <σσσσ> OPE.

    In the library convention F_lib = F_standard / N^h, so the physical
    correlator uses C²_eff = C²_physical × N^Δ. For spin-0: Δ = 2h.
    - Identity: C²_eff = 1 × N^0 = 1
    - Epsilon: C²_eff = (1/4) × N^1 = (1/4) × 4 = 1
    """
    mp.mp.dps = 50
    ext = [H_SIGMA] * 4

    # Identity Virasoro block (h=0): use regularization since c=1/2.
    # The "C^2_eff = 1" relation below holds in the N=4 (self-dual-point base
    # 4q) normalization; pin chiral_norm=4 so the test is independent of the
    # library default (now N=16). The h=0 identity block is N-independent.
    F_id = F_block_regularized(
        mp.mpf(0), z, ext, c=C_ISING, q_order=q_order, chiral_norm=4,
    )
    # Epsilon Virasoro block (h=1/2)
    F_eps = F_block_regularized(
        H_EPSILON, z, ext, c=C_ISING, q_order=q_order, chiral_norm=4,
    )

    # C²_eff: identity=1, epsilon=1 in the N=4 library normalization
    return F_id ** 2 + F_eps ** 2


@pytest.mark.parametrize("z_val", [
    mp.mpf("0.15"), mp.mpf("0.3"), mp.mpf("0.4"),
])
def test_ising_crossing_equation(z_val):
    """G(z) = (z/(1-z))^{2 Δ_ext} × G(1-z) for the Ising <σσσσ> on the diagonal.

    On the real section z = z̄, the full crossing equation
        G(z,z̄) = ((z z̄)/((1-z)(1-z̄)))^{Δ_ext} G(1-z,1-z̄)
    reduces to this form.  For Ising: Δ_ext = 2 h_σ = 1/8, exponent = 1/4.
    """
    mp.mp.dps = 50
    g_z = _ising_correlator(z_val)
    g_1mz = _ising_correlator(1 - z_val)
    delta_ext = 2 * H_SIGMA  # Δ_ext = 1/8
    crossing_prefactor = (z_val / (1 - z_val)) ** (2 * delta_ext)
    # Tolerance accounts for q-expansion truncation at q_order=20.
    tol = mp.mpf("1e-15")
    lhs = g_z
    rhs = crossing_prefactor * g_1mz
    assert abs(mp.mpc(lhs) - mp.mpc(rhs)) < tol, (
        f"Crossing violation at z={z_val}: G(z)={lhs}, "
        f"prefactor*G(1-z)={rhs}, diff={abs(lhs - rhs)}"
    )
