"""Sanity checks on the Virasoro parameter helpers."""

from __future__ import annotations

from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_blocks.params import (
    A_rs,
    A_rs_imag_b,
    b_from_c,
    c_from_b,
    d_rs,
    fusion_polynomial,
    fusion_polynomial_via_squares,
    lambda_from_h,
)


def _close(x, y, tol=mp.mpf("1e-30")):
    return abs(mp.mpc(x) - mp.mpc(y)) < tol


@pytest.mark.parametrize(
    "c_value",
    [mp.mpf("1/2"), mp.mpf("7/10"), mp.mpf(1), mp.mpf("25/26"), mp.mpf(30), mp.mpc(1, 5)],
)
def test_b_c_roundtrip(c_value):
    mp.mp.dps = 50
    b = b_from_c(c_value)
    c_back = c_from_b(b)
    assert _close(c_back, c_value, tol=mp.mpf("1e-40"))


def test_d_rs_at_unitary_ising():
    # 2d Ising: c = 1/2. Primaries epsilon (h=1/2) and sigma (h=1/16).
    # In the Cho-Collier-Yin convention d_{r,s} = (b+1/b)^2/4 - (r b + s/b)^2/4,
    # so d_{1,2} = 1/2 (epsilon) and d_{2,1} = 1/16 (sigma).
    mp.mp.dps = 50
    c = mp.mpf(1) / 2
    b = b_from_c(c)
    assert _close(d_rs(1, 2, b), mp.mpf(1) / 2, tol=mp.mpf("1e-40"))
    assert _close(d_rs(2, 1, b), mp.mpf(1) / 16, tol=mp.mpf("1e-40"))


def test_lambda_inverts_d():
    mp.mp.dps = 50
    c = mp.mpf("3.14")
    b = b_from_c(c)
    h = mp.mpf("0.123")
    lam = lambda_from_h(h, b)
    Q = b + 1 / b
    # h = Q^2/4 - lam^2/4
    h_back = (Q * Q - lam * lam) / 4
    assert _close(h_back, h, tol=mp.mpf("1e-40"))


def test_fusion_polynomial_r1_s1():
    # For (r,s)=(1,1) there is a single (p,q)=(0,0) factor:
    # P^c_{1,1}[d1,d2] = (lambda1+lambda2)/2 * (lambda1-lambda2)/2 = (lambda1^2 - lambda2^2)/4 = d2 - d1.
    mp.mp.dps = 30
    b = b_from_c(mp.mpf("0.5"))
    d1, d2 = mp.mpf("0.3"), mp.mpf("0.4")
    P = fusion_polynomial(1, 1, b, d1, d2)
    assert _close(P, d2 - d1, tol=mp.mpf("1e-25"))


def test_A_rs_simple_case_r1_s1():
    # A^c_{1,1} = 1/2: only (m,n) in {(0,1),(1,0)} survive in the product,
    # giving 1/(1/b) * 1/b = 1, times the prefactor 1/2.
    mp.mp.dps = 30
    b = b_from_c(mp.mpf("0.7"))
    assert _close(A_rs(1, 1, b), mp.mpf(1) / 2, tol=mp.mpf("1e-25"))


def test_A_rs_finite_and_real_for_real_c():
    # For real c in unitary range, b^2 is real and A_rs should be real.
    mp.mp.dps = 30
    c = mp.mpf("1.7")
    b = b_from_c(c)
    for r, s in [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (1, 3)]:
        a = A_rs(r, s, b)
        assert mp.isfinite(mp.re(a))
        # b can be complex with |Im b| small; A_rs should still be finite.


def _kac_calls():
    b = mp.mpf("1.7")
    return (
        lambda r, s: d_rs(r, s, b),
        lambda r, s: A_rs(r, s, b),
        lambda r, s: fusion_polynomial(r, s, b, mp.mpf(".3"), mp.mpf(".4")),
        lambda r, s: fusion_polynomial_via_squares(r, s, b*b, mp.mpf(".3"), mp.mpf(".4")),
        lambda r, s: A_rs_imag_b(r, s, b),
    )


def test_kac_indices_reject_nonintegral_labels_before_numerical_work():
    import numpy as np

    with mp.workdps(150):
        offset = mp.mpf(1) + mp.mpf("1e-100")
    with mp.workdps(30):
        for call in _kac_calls():
            for value in (True, np.bool_(False), "1", None, float("inf"),
                          float("nan"), 1.5, Fraction(3, 2), offset):
                with pytest.raises(ValueError, match="r must be an integer"):
                    call(value, 1)
                with pytest.raises(ValueError, match="s must be an integer"):
                    call(1, value)


def test_kac_indices_accept_integral_numbers_and_preserve_label_bounds():
    import numpy as np

    with mp.workdps(30):
        for call in _kac_calls():
            expected = call(2, 1)
            for two, one in ((2.0, 1.0), (Fraction(2), Fraction(1)),
                             (mp.mpf(2), mp.mpf(1)), (np.int64(2), np.float64(1))):
                assert call(two, one) == expected
        b = mp.mpf("1.7")
        # d_rs permits any signed integer; product formulas need positive labels.
        assert d_rs(Fraction(-2), 0.0, b) == d_rs(-2, 0, b)
        for call in _kac_calls()[1:]:
            for value in (0, -1):
                with pytest.raises(ValueError, match="r, s >= 1"):
                    call(value, 1)
                with pytest.raises(ValueError, match="r, s >= 1"):
                    call(1, value)


# ---------------------------------------------------------------------------
# Package data is the source of truth for the selector defaults.
# ---------------------------------------------------------------------------


def test_selector_defaults_come_from_the_config_not_code():
    """All selector defaults follow default_params.json across entry points."""
    import json

    from virasoro_bootstrap import defaults

    raw = json.loads(defaults.DEFAULT_PARAMS_PATH.read_text())["defaults"]
    assert defaults.DEFAULT_POLE_SELECTION == raw["pole_selection"]
    assert defaults.DEFAULT_LARGE_SPIN_POLE_SELECTION == raw["large_spin_pole_selection"]
    assert defaults.DEFAULT_GLOBAL_POLE_SELECTION == raw["global_pole_selection"]


def test_configured_selector_defaults_are_valid_names():
    from virasoro_blocks.interpolation import POLE_SELECTIONS
    from virasoro_bootstrap import defaults

    for name in (
        defaults.DEFAULT_POLE_SELECTION,
        defaults.DEFAULT_LARGE_SPIN_POLE_SELECTION,
        defaults.DEFAULT_GLOBAL_POLE_SELECTION,
    ):
        assert name in POLE_SELECTIONS, name


def test_global_fixed_spin_default_is_contribution_optimal():
    """Fixed-spin global blocks use the q-order-defined SL(2) pole universe."""
    from virasoro_bootstrap import defaults

    assert defaults.DEFAULT_GLOBAL_POLE_SELECTION == "contribution_optimal"


def test_lambda_profiles_inherit_the_shared_selector_defaults():
    """A lambdaN profile need not restate the selectors; the merge supplies them."""
    import json

    from virasoro_bootstrap import defaults

    raw = json.loads(defaults.DEFAULT_PARAMS_PATH.read_text())
    for lam in (7, 11, 19):
        assert "pole_selection" not in raw[f"lambda{lam}"], (
            f"lambda{lam} restates pole_selection; it should inherit from 'defaults'"
        )
        merged = defaults.profile(lam)
        assert merged["pole_selection"] == defaults.DEFAULT_POLE_SELECTION
        assert (
            merged["large_spin_pole_selection"]
            == defaults.DEFAULT_LARGE_SPIN_POLE_SELECTION
        )
