"""Identity-block consistency checks.

The identity block is the contribution of the vacuum primary (h = h_bar = 0 = d_{1,1})
to the OPE. For it to appear, the external operators must form identical pairs
(d1 = d2 and/or d3 = d4). The Zamolodchikov recursion encodes this through the
fusion polynomial:

    P^c_{r,s}[d, d] = 0    iff (r, s) both odd

This kills the (1,1) Kac residue 1/(h - d_{1,1}) = 1/h that would otherwise
make F(h, z, d_i, c) singular at h = 0. The tests below verify:

  1. P^c_{r,s}[d, d] vanishes exactly for (r,s) both odd.
  2. The recursion coefficient K_{r,s} = 16^{rs} A^c_{r,s} P^c_{r,s}[d1,d2] P^c_{r,s}[d4,d3]
     vanishes for (r,s) both odd whenever d1 = d2 or d3 = d4.
  3. With d1 = d2 (or d3 = d4), the polynomial numerators N_k(h) of the rational
     representation of H all vanish at h = 0 -- so D(h)/N(h) has a removable
     singularity there, i.e. the identity block is well-defined as a limit
     h -> 0 even though the raw recursion has 1/(h - d_{1,1}) = 1/h.
"""

from __future__ import annotations

import mpmath as mp
import pytest

from virasoro_blocks.params import b_from_c, fusion_polynomial
from virasoro_blocks.polyblock import H_q_series_rational
from virasoro_blocks.recursion import _build_rs_data


def _close(x, y, tol):
    return abs(mp.mpc(x) - mp.mpc(y)) < tol


@pytest.mark.parametrize(
    "r, s",
    [(1, 1), (1, 3), (3, 1), (3, 3), (1, 5), (5, 1), (3, 5), (5, 3)],
)
def test_P_vanishes_when_d_equal_and_rs_both_odd(r, s):
    """P^c_{r,s}[d, d] = 0 for (r,s) both odd (any c, any d)."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    for d in (mp.mpf("0"), mp.mpf("0.3"), mp.mpf("-1.5")):
        P = fusion_polynomial(r, s, b, d, d)
        assert _close(P, mp.mpc(0), tol=mp.mpf("1e-30")), (
            f"P^c_{{{r},{s}}}[{d}, {d}] should vanish but is {P}"
        )


@pytest.mark.parametrize(
    "r, s",
    [(1, 2), (2, 1), (2, 2), (1, 4), (4, 1), (2, 3), (3, 2)],
)
def test_P_nonzero_when_d_equal_and_rs_not_both_odd(r, s):
    """P^c_{r,s}[d, d] is generically nonzero for (r,s) not both odd."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    d = mp.mpf("0.3")
    P = fusion_polynomial(r, s, b, d, d)
    assert abs(P) > mp.mpf("1e-20"), (
        f"P^c_{{{r},{s}}}[{d}, {d}] should be nonzero but is {P}"
    )


def test_K11_vanishes_when_d1_eq_d2():
    """K_{1,1} = 16 . A^c_{1,1} . P^c_{1,1}[d1,d2] . P^c_{1,1}[d4,d3] = 0 when d1=d2."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    d = (mp.mpf("0.4"), mp.mpf("0.4"), mp.mpf("0.2"), mp.mpf("0.3"))
    pairs = _build_rs_data(1, b, d)
    # Only one pair at q_order=1: (1,1).
    assert pairs[0]["r"] == 1 and pairs[0]["s"] == 1
    assert _close(pairs[0]["K"], mp.mpc(0), tol=mp.mpf("1e-35"))


def test_K11_vanishes_when_d3_eq_d4():
    """K_{1,1} = 0 when d3 = d4 (mirror of the above)."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    d = (mp.mpf("0.4"), mp.mpf("0.3"), mp.mpf("0.2"), mp.mpf("0.2"))
    pairs = _build_rs_data(1, b, d)
    assert pairs[0]["r"] == 1 and pairs[0]["s"] == 1
    assert _close(pairs[0]["K"], mp.mpc(0), tol=mp.mpf("1e-35"))


def test_K_for_odd_rs_vanishes_when_di_zero():
    """For d_i = 0 every (r,s) both odd has K_{r,s} = 0 (since both P factors vanish)."""
    mp.mp.dps = 40
    b = b_from_c(mp.mpf("1.7"))
    pairs = _build_rs_data(9, b, (mp.mpf(0),) * 4)
    for p in pairs:
        if p["r"] % 2 == 1 and p["s"] % 2 == 1:
            assert _close(p["K"], mp.mpc(0), tol=mp.mpf("1e-35")), (
                f"K_{{{p['r']},{p['s']}}} should vanish at d_i=0 but is {p['K']}"
            )
        else:
            # Sanity: non-both-odd (r,s) typically have nonzero K.
            assert abs(p["K"]) > mp.mpf("1e-20")


def _relative_constant_term(N_k):
    """|N_k(0)| / max|coefficients|; should be ~ working precision if N_k(0)=0 exactly."""
    val = abs(N_k.evaluate(mp.mpf(0)))
    scale = max((abs(c) for c in N_k.coeffs), default=mp.mpf(1))
    if scale == 0:
        return val
    return val / scale


def _surviving_pair_labels(rational):
    return {provenance[0] for provenance in (rational.root_provenance or [])}


@pytest.mark.parametrize("q_order", [4, 6, 8])
def test_both_odd_roots_are_stripped_when_d1_eq_d2(q_order):
    """When d1 = d2, every both-odd `(r,s)` residue vanishes and its h-pole drops out."""
    mp.mp.dps = 60
    b = b_from_c(mp.mpf("1.7"))
    d = (mp.mpf("0.4"), mp.mpf("0.4"), mp.mpf("0.2"), mp.mpf("0.3"))
    rational = H_q_series_rational(d, b=b, q_order=q_order)
    surviving = _surviving_pair_labels(rational)
    tol = mp.mpf("1e-40")

    assert all(abs(mp.mpc(root)) > tol for root in rational.denominator_roots)
    assert all(not (r % 2 == 1 and s % 2 == 1) for r, s in surviving)


@pytest.mark.parametrize("q_order", [4, 6, 8])
def test_both_odd_roots_are_stripped_when_di_all_zero(q_order):
    """At d_i = 0 every both-odd `(r,s)` residue vanishes, so those h-poles are absent."""
    mp.mp.dps = 60
    b = b_from_c(mp.mpf("1.7"))
    rational = H_q_series_rational((mp.mpf(0),) * 4, b=b, q_order=q_order)
    surviving = _surviving_pair_labels(rational)
    tol = mp.mpf("1e-40")

    assert all(abs(mp.mpc(root)) > tol for root in rational.denominator_roots)
    assert all(not (r % 2 == 1 and s % 2 == 1) for r, s in surviving)


def test_identity_block_H_has_finite_limit_at_h_zero():
    """The rational H(h, q) has a removable singularity at h = 0 when d1=d2 and d3=d4.

    Evaluate at a sequence of h -> 0; H should converge (not blow up). We use
    d1 = d2 = d3 = d4 = 0.3 -- a generic pair configuration -- so the test
    isn't conflated with the special d_i = 0 case.
    """
    mp.mp.dps = 60
    b = b_from_c(mp.mpf("1.7"))
    d = (mp.mpf("0.3"),) * 4
    rational = H_q_series_rational(d, b=b, q_order=10)
    z = mp.mpf("0.3")
    from virasoro_blocks.recursion import q_from_z
    q = q_from_z(z)

    values = [rational.evaluate(mp.mpf(eps), q) for eps in
              (mp.mpf("1e-3"), mp.mpf("1e-6"), mp.mpf("1e-9"), mp.mpf("1e-12"))]
    # Differences between successive evaluations should shrink (the limit exists).
    diffs = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
    assert diffs[-1] < diffs[0], (
        f"H(h, q) does not appear to have a finite limit at h=0; values={values}"
    )
    # All values should be of order unity, not divergent.
    for v in values:
        assert abs(v) < mp.mpf("10"), f"H value at h~0 is too large: {v}"


def test_identity_block_source_strips_dead_h_zero_pole():
    mp.mp.dps = 60
    b = b_from_c(mp.mpf("1.7"))
    d = (mp.mpf("0.3"),) * 4
    rational = H_q_series_rational(d, b=b, q_order=8)
    tol = mp.mpf("1e-40")

    assert all(abs(mp.mpc(root)) > tol for root in rational.denominator_roots)
    assert rational.root_provenance is not None
    assert all(len(provenance) == 1 for provenance in rational.root_provenance)
