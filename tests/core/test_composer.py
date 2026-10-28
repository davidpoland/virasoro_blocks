"""Validate the chiral x antichiral composer and the (Delta, ell) symmetrization."""

from __future__ import annotations

import mpmath as mp
import pytest

from virasoro_blocks.composer import (
    B_block,
    G_block,
    parity_projected_B_block,
    parity_reversed_B_block,
)
from virasoro_blocks.parity import parity_reversal_phase
from virasoro_blocks.params import b_from_c
from virasoro_blocks.recursion import F_block


def _close(x, y, tol):
    return abs(mp.mpc(x) - mp.mpc(y)) < tol


def test_B_factorizes_as_product_of_F():
    """B(h, hbar; z, zbar) is exactly F_holo * F_anti."""
    mp.mp.dps = 50
    c = mp.mpf("1.7")
    b = b_from_c(c)
    h_ext = [mp.mpf("0.20"), mp.mpf("0.30"), mp.mpf("0.25"), mp.mpf("0.35")]
    hbar_ext = [mp.mpf("0.18"), mp.mpf("0.28"), mp.mpf("0.27"), mp.mpf("0.33")]

    h = mp.mpf("1.4")
    hbar = mp.mpf("1.1")
    z = mp.mpf("0.4")
    zbar = mp.mpf("0.35")

    B = B_block(h, hbar, z, zbar, h_ext, hbar_ext, b=b, q_order=10)
    F_holo = F_block(h, z, h_ext, b=b, q_order=10)
    F_anti = F_block(hbar, zbar, hbar_ext, b=b, q_order=10)
    assert _close(B, F_holo * F_anti, tol=mp.mpf("1e-40"))


def test_parity_reversed_B_swaps_chiral_data_with_unit_phase():
    mp.mp.dps = 50
    c = mp.mpf("1.7")
    b = b_from_c(c)
    h_ext = [mp.mpf("0.20"), mp.mpf("0.30"), mp.mpf("0.25"), mp.mpf("0.35")]
    hbar_ext = [mp.mpf("0.18"), mp.mpf("0.28"), mp.mpf("0.27"), mp.mpf("0.33")]
    h = mp.mpf("1.4")
    hbar = mp.mpf("1.1")
    z = mp.mpf("0.4")
    zbar = mp.mpf("0.35")

    reversed_block = parity_reversed_B_block(
        h, hbar, z, zbar, h_ext, hbar_ext, b=b, q_order=10
    )
    expected = B_block(hbar, h, z, zbar, hbar_ext, h_ext, b=b, q_order=10)
    assert _close(reversed_block, expected, tol=mp.mpf("1e-40"))


def test_parity_reversal_phase_multiplies_external_and_exchanged_phases():
    phase = parity_reversal_phase(["1", "-1", "i", "-i"], exchanged_parity="-1")
    assert phase == mp.mpc(1)


def test_parity_reversed_B_applies_phase_convention():
    mp.mp.dps = 50
    c = mp.mpf("1.7")
    b = b_from_c(c)
    h_ext = [mp.mpf("0.20"), mp.mpf("0.30"), mp.mpf("0.25"), mp.mpf("0.35")]
    hbar_ext = [mp.mpf("0.18"), mp.mpf("0.28"), mp.mpf("0.27"), mp.mpf("0.33")]
    h = mp.mpf("1.4")
    hbar = mp.mpf("1.1")
    z = mp.mpf("0.4")
    zbar = mp.mpf("0.35")

    reversed_block = parity_reversed_B_block(
        h,
        hbar,
        z,
        zbar,
        h_ext,
        hbar_ext,
        parity_external=["1", "-1", "i", "-i"],
        exchanged_parity="-1",
        b=b,
        q_order=10,
    )
    expected = B_block(hbar, h, z, zbar, hbar_ext, h_ext, b=b, q_order=10)
    assert _close(reversed_block, expected, tol=mp.mpf("1e-40"))


def test_parity_projected_B_plus_minus_match_definition():
    mp.mp.dps = 50
    c = mp.mpf("1.7")
    b = b_from_c(c)
    h_ext = [mp.mpf("0.20"), mp.mpf("0.30"), mp.mpf("0.25"), mp.mpf("0.35")]
    hbar_ext = [mp.mpf("0.18"), mp.mpf("0.28"), mp.mpf("0.27"), mp.mpf("0.33")]
    h = mp.mpf("1.4")
    hbar = mp.mpf("1.1")
    z = mp.mpf("0.4")
    zbar = mp.mpf("0.35")

    ordered = B_block(h, hbar, z, zbar, h_ext, hbar_ext, b=b, q_order=10)
    reversed_block = parity_reversed_B_block(
        h, hbar, z, zbar, h_ext, hbar_ext, b=b, q_order=10
    )
    plus = parity_projected_B_block(
        h, hbar, z, zbar, h_ext, hbar_ext, parity="+", b=b, q_order=10
    )
    minus = parity_projected_B_block(
        h, hbar, z, zbar, h_ext, hbar_ext, parity="-", b=b, q_order=10
    )
    assert _close(plus, ordered + reversed_block, tol=mp.mpf("1e-40"))
    assert _close(minus, ordered - reversed_block, tol=mp.mpf("1e-40"))


def test_parity_projected_B_requires_valid_sign():
    with pytest.raises(ValueError):
        parity_projected_B_block(
            mp.mpf("1.0"),
            mp.mpf("1.0"),
            mp.mpf("0.4"),
            mp.mpf("0.4"),
            [mp.mpf("0.1")] * 4,
            [mp.mpf("0.1")] * 4,
            parity="even",
            c=mp.mpf("1.5"),
            q_order=4,
        )


def test_parity_projected_B_scalar_limit_reproduces_G():
    mp.mp.dps = 50
    b = b_from_c(mp.mpf("1.4"))
    h_ext = [mp.mpf("0.1")] * 4
    Delta = mp.mpf("3.7")
    z = mp.mpf("0.4")
    zbar = mp.mpf("0.45")

    for ell in (0, 1, 2, 3):
        h = (Delta + ell) / 2
        hbar = (Delta - ell) / 2
        # the scalar G is the '+' projection at every spin, odd spins included
        projected = parity_projected_B_block(
            h, hbar, z, zbar, h_ext, h_ext, parity="+", b=b, q_order=10
        )
        G = G_block(Delta, ell, z, zbar, h_ext, b=b, q_order=10)
        # B + P[B] = 2 B(h, h) at ell = 0, where G is B(h, h) itself
        assert _close(projected, (2 if ell == 0 else 1) * G, tol=mp.mpf("1e-40"))


def test_parity_projected_B_minus_is_the_zzbar_odd_structure():
    """The z <-> zbar-odd B(h, hbar) - B(hbar, h) is parity_projected_B with eta = -1 (not a scalar G)."""
    mp.mp.dps = 50
    b = b_from_c(mp.mpf("1.4"))
    h_ext = [mp.mpf("0.1")] * 4
    Delta, z, zbar = mp.mpf("3.7"), mp.mpf("0.4"), mp.mpf("0.45")
    for ell in (1, 2, 3):
        h, hbar = (Delta + ell) / 2, (Delta - ell) / 2
        B1 = B_block(h, hbar, z, zbar, h_ext, h_ext, b=b, q_order=10)
        B2 = B_block(hbar, h, z, zbar, h_ext, h_ext, b=b, q_order=10)
        odd = parity_projected_B_block(h, hbar, z, zbar, h_ext, h_ext, parity="-", b=b, q_order=10)
        assert _close(odd, B1 - B2, tol=mp.mpf("1e-40"))
        # eta = -1 from the exchanged parity with the '+' projection is the same structure
        eta = parity_projected_B_block(h, hbar, z, zbar, h_ext, h_ext, parity="+", exchanged_parity="-1",
                                       b=b, q_order=10)
        assert _close(eta, B1 - B2, tol=mp.mpf("1e-40"))
        swapped = parity_projected_B_block(h, hbar, zbar, z, h_ext, h_ext, parity="-", b=b, q_order=10)
        assert _close(swapped, -odd, tol=mp.mpf("1e-35"))


def test_G_ell_zero_equals_BB():
    """For ell = 0 the two ordered blocks coincide; G = F(h,z) F(h,zbar)."""
    mp.mp.dps = 50
    b = b_from_c(mp.mpf("1.7"))
    h_ext = [mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.25"), mp.mpf("0.35")]
    Delta = mp.mpf("2.8")
    z = mp.mpf("0.45")
    zbar = mp.mpf("0.4")

    G = G_block(Delta, 0, z, zbar, h_ext, b=b, q_order=10)
    h = Delta / 2
    F_z = F_block(h, z, h_ext, b=b, q_order=10)
    F_zbar = F_block(h, zbar, h_ext, b=b, q_order=10)
    assert _close(G, F_z * F_zbar, tol=mp.mpf("1e-40"))


def test_G_z_swap_parity():
    """G_{Delta, ell}(zbar, z) = G_{Delta, ell}(z, zbar) for scalar externals, at every spin."""
    mp.mp.dps = 50
    b = b_from_c(mp.mpf("1.7"))
    h_ext = [mp.mpf("0.15")] * 4  # scalar externals
    Delta = mp.mpf("3.1")
    z = mp.mpf("0.45")
    zbar = mp.mpf("0.4")

    for ell in (0, 1, 2, 3, 4):
        G_zzb = G_block(Delta, ell, z, zbar, h_ext, b=b, q_order=10)
        G_zbz = G_block(Delta, ell, zbar, z, h_ext, b=b, q_order=10)
        assert _close(G_zbz, G_zzb, tol=mp.mpf("1e-35"))


def test_G_diagonal_limit_zbar_to_z():
    """For every ell != 0, G_{Delta, ell}(z, z) is twice the chiral product on the diagonal: the scalar G is
    z <-> zbar even, so an odd spin does not vanish there."""
    mp.mp.dps = 50
    b = b_from_c(mp.mpf("1.7"))
    h_ext = [mp.mpf("0.12")] * 4
    Delta = mp.mpf("2.6")
    z = mp.mpf("0.4")

    for ell in (1, 2):
        G = G_block(Delta, ell, z, z, h_ext, b=b, q_order=10)
        h = (Delta + ell) / 2
        hbar = (Delta - ell) / 2
        expected = 2 * F_block(h, z, h_ext, b=b, q_order=10) * F_block(
            hbar, z, h_ext, b=b, q_order=10
        )
        assert _close(G, expected, tol=mp.mpf("1e-35"))


def test_G_uses_chiral_inputs_consistently():
    """Internally G splits Delta into (h, hbar) via ell. Verify against the explicit
    B-sum formula."""
    mp.mp.dps = 50
    b = b_from_c(mp.mpf("1.4"))
    h_ext = [mp.mpf("0.1")] * 4  # scalar externals (h_ext used for both halves)
    Delta = mp.mpf("3.7")
    ell = 3
    z = mp.mpf("0.4")
    zbar = mp.mpf("0.45")

    G = G_block(Delta, ell, z, zbar, h_ext, b=b, q_order=10)

    h = (Delta + ell) / 2
    hbar = (Delta - ell) / 2
    B1 = B_block(h, hbar, z, zbar, h_ext, h_ext, b=b, q_order=10)
    B2 = B_block(hbar, h, z, zbar, h_ext, h_ext, b=b, q_order=10)
    # odd ell: B(h, hbar) + B(hbar, h), the parity-even block (not the odd B - B)
    assert _close(G, B1 + B2, tol=mp.mpf("1e-40"))


def test_B_block_rejects_wrong_length_externals():
    with pytest.raises(ValueError):
        B_block(
            mp.mpf("1.0"),
            mp.mpf("1.0"),
            mp.mpf("0.4"),
            mp.mpf("0.4"),
            [mp.mpf("0.1")] * 3,
            [mp.mpf("0.1")] * 4,
            c=mp.mpf("1.5"),
            q_order=4,
        )


def test_G_block_requires_integer_ell():
    with pytest.raises(ValueError):
        G_block(
            mp.mpf("2.0"),
            mp.mpf("1.5"),
            mp.mpf("0.4"),
            mp.mpf("0.4"),
            [mp.mpf("0.1")] * 4,
            c=mp.mpf("1.5"),
            q_order=4,
        )


def test_G_block_requires_b_xor_c():
    with pytest.raises(ValueError):
        G_block(
            mp.mpf("2.0"),
            0,
            mp.mpf("0.4"),
            mp.mpf("0.4"),
            [mp.mpf("0.1")] * 4,
            q_order=4,
        )


@pytest.mark.parametrize("ell", [1, 2, 3])
def test_G_block_negative_ell_is_redundant_with_positive(ell):
    """G_{Delta, -ell} = G_{Delta, +ell} for integer ell.

    ell -> -ell swaps h <-> hbar, under which the symmetric sum is invariant,
    so the negative-ell labeling is redundant rather than wrong. We accept it
    (no error) and document the symmetry here.
    """
    mp.mp.dps = 50
    c = mp.mpf("1.7")
    b = b_from_c(c)
    Delta = mp.mpf("5.4")
    z = mp.mpf("0.4")
    zbar = mp.mpf("0.35")
    h_ext = [mp.mpf("0.20"), mp.mpf("0.25"), mp.mpf("0.22"), mp.mpf("0.27")]
    pos = G_block(Delta, ell, z, zbar, h_ext, b=b, q_order=6)
    neg = G_block(Delta, -ell, z, zbar, h_ext, b=b, q_order=6)
    assert _close(neg, pos, tol=mp.mpf("1e-35"))
