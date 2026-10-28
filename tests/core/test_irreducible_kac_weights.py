"""On a Kac weight of a singular c the block of the irreducible module, for the superconformal and torus families.

NS / Ramond: the c -> c0 limit at fixed weight is the physical block (as for Virasoro), anchored on the free boson +
fermion at c = 3/2; the value paths returned "on the Kac pole" for every Kac weight, the identity included.  Torus:
neither limit of the recursion is it (the near-null states keep a finite diagonal matrix element, and at d = 0 the
recursion is the Verma character), so the trace over L(h) is built exactly (`irreducible_onept_F_series`), anchored on
Ising <epsilon> on sigma (prod(1 - q^n)), c = 1 zero modes and the minimal-model characters."""

from fractions import Fraction

import mpmath as mp
import pytest


def _eta_power(q_order, power):
    out = [mp.mpf(1)] + [mp.mpf(0)] * q_order
    for _ in range(power):
        for k in range(1, q_order + 1):
            new = list(out)
            for i in range(k, q_order + 1):
                new[i] -= out[i - k]
            out = new
    return out


def test_n1_free_field_coefficient_fixes_the_physical_block():
    """<V_a V_-a V_a V_-a> = z^-a^2 (1 - z)^-a^2 at c = 3/2 (free boson + Majorana; bottom components of the vertex
    superfields, h_ext = a^2/2).  To level 2: the identity (even, h = 0), psi (odd, h = 1/2: its G_{-1/2} is the U(1)
    current, coupling a^2) and the even superprimary P2 = a_{-1}^2 - 2 psi_{-3/2} psi_{-1/2} (norm 6, overlap a^2), so
    its coefficient is a^4 / 6.  All three are Kac weights of c = 3/2; with the c -> c0 limit the peel is exact, with
    the limit in the weight the identity block is off at level 2 (0.03375 against 0.03)."""
    from virasoro_blocks.superconformal_blocks import ns_chiral_block

    with mp.workdps(30):
        a2 = mp.mpf(3) / 10
        he = a2 / 2
        ext = [he] * 4
        N, r = 12, mp.mpf("0.02")

        def coeffs(h, sector, lead):
            out = [mp.mpc(0)] * 3
            for j in range(N):
                z = r * mp.expjpi(2 * mp.mpf(j) / N)
                u = ns_chiral_block(h, z, ext, c=mp.mpf(3) / 2, q_order=8, chiral_norm=1, z_shift=-2 * he,
                                    sector=sector) / z ** (lead - 2 * he)
                for k in range(3):
                    out[k] += u / z ** k / N
            return [mp.re(x) for x in out]

        g = mp.taylor(lambda z: (1 - z) ** (-a2), 0, 2)
        identity, psi = coeffs(mp.mpf(0), 0, 0), coeffs(mp.mpf(1) / 2, 1, 1)
        a_psi = (g[1] - identity[1]) / psi[0]
        assert abs(a_psi * psi[0] - a2) < mp.mpf(10) ** -18
        assert abs((g[2] - identity[2] - a_psi * psi[1]) / (a2 ** 2 / 6) - 1) < mp.mpf(10) ** -18


@pytest.mark.parametrize("family", ["ns", "ramond"])
def test_super_value_on_a_kac_weight_is_the_c_limit(family):
    """The value on a Kac weight of c = 3/2 is the bare block just off it (here at 300 digits, c0 + 1e-80)."""
    from virasoro_blocks import superconformal_blocks as SB
    from virasoro_blocks.recursion_ns import F_block_ns
    from virasoro_blocks.recursion_r import F_block_r

    with mp.workdps(30):
        if family == "ns":
            ext, cases = [mp.mpf(3) / 20] * 4, [(mp.mpf(0), 0), (mp.mpf(2), 0), (mp.mpf(1) / 2, 1)]
            block, bare = SB.ns_chiral_block, F_block_ns
        else:
            ext, cases = [mp.mpf(9) / 16] * 4, [(mp.mpf(0), 0), (mp.mpf(2), 1), (mp.mpf(9) / 2, 0)]
            block, bare = SB.ramond_chiral_block, F_block_r
        for h, sector in cases:
            got = block(h, mp.mpf("0.3"), ext, c=mp.mpf(3) / 2, q_order=10, sector=sector)
            with mp.workdps(300):
                ref = bare(h, mp.mpf("0.3"), ext, c=mp.mpf(3) / 2 + mp.mpf(10) ** -80, q_order=10, sector=sector)
            assert abs(got / ref - 1) < mp.mpf(10) ** -28


@pytest.mark.parametrize("q_order", [5, 10])
def test_super_removable_kac_location_is_the_c_limit(q_order):
    """h = 0 at c = 3/2 is a Kac location whose poles all cancel (no merged column): the value there is still the
    c -> c0 limit (H_4 = 72/25), not the merged form's limit in h (the Fock block, 96/25)."""
    from virasoro_blocks.recursion_ns import H_q_series_ns

    with mp.workdps(30):
        H = H_q_series_ns(mp.mpf(0), [mp.mpf(3) / 20] * 4, c=mp.mpf(3) / 2, q_order=q_order, sector=0)
        assert abs(H[4] - mp.mpf(72) / 25) < mp.mpf(10) ** -25


def test_super_pole_at_every_c_is_a_genuine_pole():
    """The odd NS block at Delta = 0 sits on d_{1,1} = 0 at every c (H_1 ~ 2 / Delta): at the singular c = 81/70 it
    is refused as a genuine pole (it divided by an identically zero series)."""
    from virasoro_blocks.recursion_ns import H_q_series_ns
    from virasoro_blocks.super_laurent import SuperLaurentPoleError

    with mp.workdps(30):
        with pytest.raises(SuperLaurentPoleError, match="every c"):
            H_q_series_ns(mp.mpf(0), [mp.mpf(8) / 7] * 4, c=mp.mpf(81) / 70, ns_order=8, sector=1)


def test_super_genuine_pole_on_a_kac_weight_raises():
    """NS h = 1/2, even sector, at c = 3/2: H_4 grows like 0.24 / (c - 3/2), a genuine pole."""
    from virasoro_blocks.superconformal_blocks import SuperconformalPoleError, ns_chiral_block

    with mp.workdps(30), pytest.raises(SuperconformalPoleError, match="genuine pole"):
        ns_chiral_block(mp.mpf(1) / 2, mp.mpf("0.3"), [mp.mpf(3) / 20] * 4, c=mp.mpf(3) / 2, q_order=10)


def test_ising_epsilon_on_sigma_is_eta():
    """Tr_{L(1/16)} epsilon q^{L_0 - c/24} = q^{1/24} prod(1 - q^n): reduced f = prod(1 - q^n)^2, exactly; the
    recursion's limit (the Verma trace) is 1, -2, 0, 1.94, ..., off from q^2 where sigma's null sits."""
    from virasoro_blocks.torus_onept import _f_irr_from_F_irr, irreducible_onept_F_series

    with mp.workdps(30):
        F = irreducible_onept_F_series(Fraction(1, 16), Fraction(1, 2), Fraction(1, 2), 24, exact=True)
        f = _f_irr_from_F_irr(F)
        assert [mp.nint(x) for x in f] == _eta_power(24, 2) and all(x == mp.nint(x) for x in f)


@pytest.mark.parametrize("h, level", [(Fraction(1, 4), 2), (Fraction(1), 3), (Fraction(9, 4), 4)])
def test_c1_zero_mode_trace_is_the_irreducible_character(h, level):
    """c = 1, an external of weight 1 (its zero mode commutes with Virasoro): the trace over L(h) is the irreducible
    character (q^h - q^{h + level}) / eta, i.e. f = 1 - q^level; the recursion gives the Verma character, f = 1."""
    from virasoro_blocks.torus_onept import _f_irr_from_F_irr, irreducible_onept_F_series

    F = irreducible_onept_F_series(h, 1, 1, 14, exact=True)
    f = _f_irr_from_F_irr(F)
    assert [mp.nint(x) for x in f] == [1] + [-1 if k == level else 0 for k in range(1, 15)]


@pytest.mark.parametrize("c, h, d", [(Fraction(1, 2), Fraction(1, 16), Fraction(1, 16)),
                                     (1, Fraction(1, 4), 4), (1, 1, "0.3")])
def test_torus_external_that_does_not_decouple_is_refused(c, h, d):
    """sigma x sigma has no sigma; at c = 1, j = 1/2 x k = 2 has no j = 1/2 (<chi|O|h> = d(d - 1) = 12 for the
    level-2 null), and a generic weight never decouples: the irreducible block does not exist."""
    from virasoro_blocks.torus_onept import TorusOnePtDecouplingError, irreducible_onept_F_series

    with pytest.raises(TorusOnePtDecouplingError, match="does not decouple"):
        irreducible_onept_F_series(h, c, d, 8)


def test_torus_onepoint_cli_value_is_the_exact_ising_block():
    """`evaluate --family torus-onepoint` at Ising, d = 1/2, Delta = 1/8 refused the Kac weight; it is
    tau_2 |q^{1/24} prod(1 - q^n)|^2 ... in the reduced convention tau_2 |q^{1/16 - (c-1)/24} prod(1 - q^n)^2|^2."""
    from virasoro_blocks.api import BlockRequest, NumericalPolicy, evaluate_block_request
    from virasoro_blocks.modular import torus_disk_point

    with mp.workdps(30):
        value = evaluate_block_request(BlockRequest(
            operation="evaluate", surface="torus", family="torus-onepoint", coordinate="vvb",
            central_charge=mp.mpf(1) / 2, torus_onepoint_ext=mp.mpf(1) / 2, exchange_delta=mp.mpf(1) / 8,
            exchange_spin=0, point={"v": mp.mpf("0.1"), "vbar": mp.mpf("0.1")},
            policy=NumericalPolicy(precision=30, order=14)))
        _t, _tb, tau2, q, qbar = torus_disk_point(mp.mpf("0.1"), mp.mpf("0.1"))
        shift = mp.mpf(1) / 16 + mp.mpf(1) / 48
        exact = tau2 * q ** shift * qbar ** shift * mp.qp(q) ** 2 * mp.qp(qbar) ** 2
        assert value.metadata["regularization_policy"] == "irreducible"
        assert abs(value.value / exact - 1) < mp.mpf(10) ** -25


def test_characters_on_kac_weights_are_irreducible():
    """Ising characters from theta functions, tricritical Ising from Rocha-Caridi, c = 1 (q^h - q^{h + level}) / eta,
    and the vacuum (1 - q) at a generic c: the reduced character q^{h - (c-1)/24} g(q)."""
    from virasoro_blocks.modular import torus_disk_point
    from virasoro_blocks.torus_onept import irreducible_character_factor

    with mp.workdps(30):
        tau, _tb, _t2, q, _qb = torus_disk_point(mp.mpf("0.1"), mp.mpf("0.1"))
        eta = q ** (mp.mpf(1) / 24) * mp.qp(q)
        th2, th3, th4 = (mp.jtheta(k, 0, mp.exp(mp.pi * 1j * tau)) for k in (2, 3, 4))
        half = mp.mpf(1) / 2
        ising = {mp.mpf(0): (mp.sqrt(th3 / eta) + mp.sqrt(th4 / eta)) / 2,
                 half: (mp.sqrt(th3 / eta) - mp.sqrt(th4 / eta)) / 2, mp.mpf(1) / 16: mp.sqrt(th2 / (2 * eta))}
        for h, chi in ising.items():
            g, degenerate = irreducible_character_factor(half, h, q)
            assert degenerate and abs(q ** (h + mp.mpf(1) / 48) * g / (eta * chi) - 1) < mp.mpf(10) ** -27

        def rocha_caridi(p, pp, r, s, K=20):
            return sum(q ** (mp.mpf((2 * p * pp * k + pp * r - p * s) ** 2 - (pp - p) ** 2) / (4 * p * pp))
                       - q ** (mp.mpf((2 * p * pp * k + pp * r + p * s) ** 2 - (pp - p) ** 2) / (4 * p * pp))
                       for k in range(-K, K + 1))

        c = mp.mpf(7) / 10
        for r, s, h in ((1, 2, mp.mpf(1) / 10), (1, 3, mp.mpf(3) / 5), (2, 2, mp.mpf(3) / 80)):
            g, _ = irreducible_character_factor(c, h, q)
            assert abs(q ** (h - (c - 1) / 24) * g / (q ** ((1 - c) / 24) * rocha_caridi(4, 5, r, s)) - 1) \
                < mp.mpf(10) ** -27
        for h, level in ((mp.mpf(1) / 4, 2), (mp.mpf(1), 3)):
            g, _ = irreducible_character_factor(1, h, q)
            assert abs(g - (1 - q ** level)) < mp.mpf(10) ** -28
        g, degenerate = irreducible_character_factor(mp.mpf(3), mp.mpf(0), q)
        assert degenerate and abs(g - (1 - q)) < mp.mpf(10) ** -28
        assert irreducible_character_factor(mp.mpf(3), mp.mpf("0.3"), q) == (1, False)
