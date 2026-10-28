"""Validate the c = 1/2 (2d Ising) blocks against exact closed forms.

The 2d Ising model is the m = 3 minimal model, c = 1/2. Its spin field
sigma = phi_{1,2} has h = 1/16 and is degenerate at level 2, so the four-spin
chiral conformal blocks satisfy a hypergeometric ODE and are *algebraic*
(BPZ 1984). The sigma x sigma OPE exchanges the identity 1 (h = 0) and the
energy operator epsilon = phi_{1,3} (h = 1/2):

    F_identity(z) = (1 - z)^(-1/8) * sqrt((1 + sqrt(1-z)) / 2)
    F_epsilon(z)  = 2 z^(-1/2) (1 - z)^(-1/8) * sqrt((1 - sqrt(1-z)) / 2)

both normalized to start at z^0 -> 1 (the convention `F_block` uses: as
c -> infinity it tends to 2F1, which starts at 1). The library further divides
the standard normalization by N^h with N = DEFAULT_CHIRAL_NORM (= 16), so

    F_block(h, z; {1/16}; c=1/2) = N^(-h) * F_{closed}(h, z).

c = 1/2 is a recursion singularity (a minimal model), and the exchanged
operators sit exactly on Kac poles, so the blocks are computed through the
`*_regularized` wrappers. This is a strong end-to-end check: the regularized
Zamolodchikov recursion must reproduce an independently known exact answer.
"""

from __future__ import annotations

import mpmath as mp
import pytest

from virasoro_blocks.recursion import DEFAULT_CHIRAL_NORM, F_block
from virasoro_blocks.regularization import (
    B_block_regularized,
    F_block_regularized,
    G_block_regularized,
)


_N = DEFAULT_CHIRAL_NORM

C_ISING = mp.mpf(1) / 2
_EIGHTH = mp.mpf(1) / 8
_HALF = mp.mpf(1) / 2
_MIXED_ISING_ORDERINGS = [
    "sigma sigma epsilon epsilon",
    "sigma epsilon epsilon sigma",
    "sigma epsilon sigma epsilon",
]


def _h_sigma():
    return mp.mpf("0.0625")


def _h_epsilon():
    return mp.mpf("0.5")


def _sigma_ext():
    sigma = _h_sigma()
    return [sigma] * 4


def _library_sigma_factor():
    return _N ** (-_h_sigma())


def _mixed_case_data(ordering):
    sigma = _h_sigma()
    epsilon = _h_epsilon()
    if ordering == "sigma sigma epsilon epsilon":
        return mp.mpf(0), [sigma, sigma, epsilon, epsilon]
    if ordering == "sigma epsilon epsilon sigma":
        return sigma, [sigma, epsilon, epsilon, sigma]
    if ordering == "sigma epsilon sigma epsilon":
        return sigma, [sigma, epsilon, sigma, epsilon]
    raise ValueError(f"unknown mixed-correlator ordering: {ordering}")


def _F_identity_closed(z):
    """Ising sigmasigmasigmasigma identity-channel chiral block (z^0-normalized)."""
    root = mp.sqrt(1 - z)
    return (1 - z) ** (-_EIGHTH) * mp.sqrt((1 + root) / 2)


def _F_epsilon_closed(z):
    """Ising sigmasigmasigmasigma epsilon-channel chiral block (z^0-normalized)."""
    root = mp.sqrt(1 - z)
    return 2 * z ** (-_HALF) * (1 - z) ** (-_EIGHTH) * mp.sqrt((1 - root) / 2)


def _F_library_convention(h, z):
    """The closed-form chiral block in the library's convention.

    Convention: F = z^h × N^{-h} × F_standard.
    For sigma-sigma-sigma-sigma: the standard z^0-normalized forms are used.
    To get the physical kinematic z^{h-h1-h2} factor, pass z_shift=-h1-h2.
    """
    closed = _F_identity_closed(z) if h == 0 else _F_epsilon_closed(z)
    return z ** h * closed / _N ** h


def _F_mixed_library_convention(ordering, z):
    """Exact mixed-correlator Ising chiral blocks in the library convention.

    Convention: F = z^h × N^{-h} × F_standard.
    Section 4 of Maloney-Maxfield-Ng, JHEP 06 (2017) 117 gives the mixed
    Ising blocks in a symmetric four-point-function normalization. Converting to
    the repo's chiral normalization (z_shift=0 default gives z^h/N^h).
    """
    sigma = _h_sigma()
    epsilon = _h_epsilon()
    if ordering == "sigma sigma epsilon epsilon":
        # h_exchanged = 0, d = [sigma, sigma, epsilon, epsilon]
        # z^0 (identity exchange)
        # Standard form (z^0-normalized): (1 - z/2) / sqrt(1-z)
        return (1 - z / 2) / mp.sqrt(1 - z)
    if ordering == "sigma epsilon epsilon sigma":
        # h_exchanged = sigma, d = [sigma, epsilon, epsilon, sigma]
        # z^sigma / N^sigma
        return z ** sigma * _library_sigma_factor() * (1 + z) / (1 - z)
    if ordering == "sigma epsilon sigma epsilon":
        # h_exchanged = sigma, d = [sigma, epsilon, sigma, epsilon]
        # z^sigma / N^sigma
        return z ** sigma * _library_sigma_factor() * (1 - 2 * z) / mp.sqrt(1 - z)
    raise ValueError(f"unknown mixed-correlator ordering: {ordering}")


def _B_mixed_library_convention(ordering, z):
    value = _F_mixed_library_convention(ordering, z)
    return value * value


def test_ising_chiral_identity_block_matches_closed_form():
    """F_block_regularized reproduces the exact identity-channel chiral block."""
    mp.mp.dps = 50
    for z in [mp.mpf("0.1"), mp.mpf("0.2"), mp.mpf("0.3")]:
        lib = F_block_regularized(mp.mpf(0), z, _sigma_ext(), c=C_ISING, q_order=20)
        closed = _F_library_convention(mp.mpf(0), z)
        assert abs(mp.mpc(lib) - closed) < mp.mpf("1e-30")


def test_ising_chiral_epsilon_block_matches_closed_form():
    """F_block_regularized reproduces the exact epsilon-channel chiral block."""
    mp.mp.dps = 50
    for z in [mp.mpf("0.1"), mp.mpf("0.2"), mp.mpf("0.3")]:
        lib = F_block_regularized(_HALF, z, _sigma_ext(), c=C_ISING, q_order=20)
        closed = _F_library_convention(_HALF, z)
        assert abs(mp.mpc(lib) - closed) < mp.mpf("1e-30")


@pytest.mark.parametrize("ordering", _MIXED_ISING_ORDERINGS)
def test_ising_mixed_chiral_blocks_match_closed_forms(ordering):
    """The three mixed Ising orderings reproduce the exact section-4 blocks."""
    mp.mp.dps = 50
    tol = mp.mpf("1e-16")
    h, externals = _mixed_case_data(ordering)
    for z in [mp.mpf("0.1"), mp.mpf("0.2"), mp.mpf("0.3")]:
        lib = F_block_regularized(h, z, externals, c=C_ISING, q_order=20)
        closed = _F_mixed_library_convention(ordering, z)
        assert abs(mp.mpc(lib) - closed) < tol


@pytest.mark.parametrize("ordering", ["sigma epsilon sigma epsilon"])
def test_ising_mixed_ordered_blocks_match_closed_forms_at_real_cross_ratio(ordering):
    """At real z = zbar the ordered 2d block equals the square of the exact chiral block.  One ordering: the chiral
    blocks of all three are checked against their closed forms above."""
    mp.mp.dps = 50
    tol = mp.mpf("1e-16")
    h, externals = _mixed_case_data(ordering)
    for z in [mp.mpf("0.2"), mp.mpf("0.3")]:
        lib = B_block_regularized(
            h,
            h,
            z,
            z,
            externals,
            externals,
            c=C_ISING,
            q_order=20,
        )
        closed = _B_mixed_library_convention(ordering, z)
        assert abs(mp.mpc(lib) - closed) < tol


@pytest.mark.parametrize(
    "ordering,order",
    [
        ("sigma sigma sigma sigma", 2),
        ("sigma epsilon epsilon sigma", 2),
    ],
)
def test_ising_self_dual_chiral_derivatives_match_closed_form(ordering, order):
    """The exact second derivative at z = 1/2 matches the regularized recursion."""
    mp.mp.dps = 60
    z0 = mp.mpf("0.5")
    tol = mp.mpf("1e-22")

    if ordering == "sigma sigma sigma sigma":
        h = mp.mpf(0)
        externals = _sigma_ext()
        closed = lambda z: _F_library_convention(h, z)
    else:
        h, externals = _mixed_case_data(ordering)
        closed = lambda z: _F_mixed_library_convention(ordering, z)

    lib = lambda z: F_block_regularized(
        h,
        z,
        externals,
        c=C_ISING,
        q_order=20,
    )

    assert abs(mp.diff(lib, z0, order) - mp.diff(closed, z0, order)) < tol


@pytest.mark.parametrize("Delta,h", [(mp.mpf(0), mp.mpf(0)), (mp.mpf(1), _HALF)])
def test_ising_2d_block_matches_closed_form(Delta, h):
    """The 2d block G_{Delta, 0} reproduces the closed form. For scalar
    externals and ell = 0 the library normalization is
    G_{Delta,0}(z, zbar) = F(h, z) F(h, zbar) with h = Delta/2."""
    mp.mp.dps = 50
    for z, zbar in [(mp.mpf("0.2"), mp.mpf("0.2")),
                    (mp.mpf("0.3"), mp.mpf("0.15")),
                    (mp.mpf("0.25"), mp.mpf("0.1"))]:
        lib = G_block_regularized(Delta, 0, z, zbar, _sigma_ext(), c=C_ISING, q_order=20)
        closed = _F_library_convention(h, z) * _F_library_convention(h, zbar)
        assert abs(mp.mpc(lib) - closed) < mp.mpf("1e-30")


def test_identity_block_with_equal_externals_computes():
    """Regression: the identity exchange (h = 0) sits on the Kac pole d_{1,1}=0,
    where the (1,1) recursion term has K = 0. The float recursion must skip
    that identically-zero term rather than divide 0/0. Checked at a generic,
    non-singular c so it isolates the recursion fix from c-regularization."""
    mp.mp.dps = 40
    z = mp.mpf("0.2")
    value = F_block(mp.mpf(0), z, [mp.mpf("0.2")] * 4, c=mp.mpf("10"), q_order=8)
    assert mp.isfinite(abs(mp.mpc(value)))
