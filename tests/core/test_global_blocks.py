"""Tests for the 2d global (SL(2)xSL(2)) scalar block basis.

The global block has no exact finite rational DeltaDerivTable (infinitely many
SL(2) Delta-poles), so `global_scalar_G_interpolated_table` builds the
order-reduced `InterpolatedDeltaTable` directly by optimal-node interpolation
of the numeric block. These tests pin the decay base, check the interpolation
is bootstrap-grade against the numeric block, and confirm the result flows
through serialization and the JSON / Mathematica writers exactly like the
Virasoro `scalar_G` basis.
"""

from __future__ import annotations

import json

import mpmath as mp
import pytest

from virasoro_blocks.derivs import (
    global_G_qqb_table,
    global_G_qqb_ws_table,
    global_G_ws_table,
    global_G_xt_table,
    global_G_yyb_table,
    global_G_zzb_table,
    global_chiral_z_taylor,
    global_parity_projected_B_ws_table,
    global_parity_projected_B_xt_table,
)
from virasoro_blocks.global_blocks import (
    GLOBAL_COORDINATES,
    _global_q_order_pole_pool,
    global_block_asymptotic_power,
    global_chiral_F_interpolated_table,
    global_block_exp_base,
    global_evaluate_block,
    global_ordered_B_interpolated_table,
    global_parity_projected_B_interpolated_table,
    global_scalar_G_interpolated_table,
    global_sl2_delta_poles,
)
from virasoro_blocks.parity import parity_reversal_phase
from virasoro_blocks.serialization import serialize_deriv_tables
from virasoro_blocks.truncation import kept_pairs
from virasoro_blocks.writers import write_json, write_mathematica
import virasoro_blocks.__main__ as cli


EXT = [mp.mpf("0.0625")] * 4  # Delta_ext = 1/8
CN = mp.mpf(4)
SPIN_H_EXT = [mp.mpf("0.075"), mp.mpf("1.125"), mp.mpf("0.175"), mp.mpf("1.225")]
SPIN_HBAR_EXT = [mp.mpf("0.075"), mp.mpf("0.125"), mp.mpf("0.175"), mp.mpf("0.225")]
CHIRAL_EXT = [mp.mpf("0.05"), mp.mpf("0.10"), mp.mpf("0.15"), mp.mpf("0.20")]


def _manual_global_ordered_zzb(delta, ell, h_ext, hbar_ext, m_max, n_max, *,
                               z_shift, zbar_shift):
    h = (delta + ell) / 2
    hbar = (delta - ell) / 2
    n_top = max(m_max, n_max) + 1
    c_h = global_chiral_z_taylor(
        h, h_ext, n_top, chiral_norm=CN, z_shift=z_shift,
    )
    c_hb = global_chiral_z_taylor(
        hbar, hbar_ext, n_top, chiral_norm=CN, z_shift=zbar_shift,
    )
    return [[c_h[m] * c_hb[n] for n in range(n_max + 1)]
            for m in range(m_max + 1)]


def _manual_global_parity_zzb(delta, ell, h_ext, hbar_ext, m_max, n_max, *,
                              parity, parity_external, exchanged_parity,
                              z_shift, zbar_shift):
    sigma = (
        (mp.mpf(1) if parity == "+" else mp.mpf(-1))
        * parity_reversal_phase(
            parity_external, exchanged_parity=exchanged_parity,
        )
    )
    h = (delta + ell) / 2
    hbar = (delta - ell) / 2
    n_top = max(m_max, n_max) + 1
    c_h = global_chiral_z_taylor(
        h, h_ext, n_top, chiral_norm=CN, z_shift=z_shift,
    )
    c_hb = global_chiral_z_taylor(
        hbar, hbar_ext, n_top, chiral_norm=CN, z_shift=zbar_shift,
    )
    return [[c_h[m] * c_hb[n] + sigma * c_hb[m] * c_h[n]
             for n in range(n_max + 1)] for m in range(m_max + 1)]


def _worst_rel_2d(tab, exact_fn, deltas, *, m_max, n_max):
    worst = mp.mpf(0)
    for d in deltas:
        approx = tab.evaluate(mp.mpf(d))
        exact = exact_fn(mp.mpf(d))
        for m in range(m_max + 1):
            for n in range(n_max + 1):
                denom = abs(exact[m][n]) + mp.mpf(10) ** (-200)
                worst = max(worst, abs(approx[m][n] - exact[m][n]) / denom)
    return worst


def _worst_rel_chiral(tab, exact_fn, hs, *, m_max):
    worst = mp.mpf(0)
    for h in hs:
        approx = tab.evaluate(mp.mpf(h))
        exact = exact_fn(mp.mpf(h))
        for m in range(m_max + 1):
            denom = abs(exact[m]) + mp.mpf(10) ** (-200)
            worst = max(worst, abs(approx[m][0] - exact[m]) / denom)
    return worst


def test_exp_base_is_four_rho_over_n():
    mp.mp.dps = 50
    # rho(1/2) = 3 - 2 sqrt(2); exp_base = 4 rho / N = (3 - 2 sqrt(2)) for N=4.
    expected = 4 * (3 - 2 * mp.sqrt(2)) / CN
    assert abs(global_block_exp_base(CN) - expected) < mp.mpf(10) ** (-40)
    assert abs(global_block_exp_base(CN) - (3 - 2 * mp.sqrt(2))) < mp.mpf(10) ** (-40)


def test_genuine_poles_identical_externals_are_odd_offset():
    # Identical / pairwise-equal externals (delta12 = delta34 = 0): genuine
    # poles only at odd p (Delta = ell - p, -ell - p), even p removable.
    # ell = 2 (shift = 2): nearest 4 are 1, -1, -3 (double).
    poles = global_sl2_delta_poles(2, 4)
    assert sorted(float(p) for p in poles) == [-3, -3, -1, 1]
    # ell = 0 (shift = 0): both branches coincide at -p (p odd) -> double poles.
    poles0 = global_sl2_delta_poles(0, 4)
    assert sorted(float(p) for p in poles0) == [-3, -3, -1, -1]


def test_genuine_pole_locations_depend_on_externals():
    # The genuine set depends on the external differences delta12 = h2 - h1,
    # delta34 = h3 - h4. Generic (non-half-integer) externals have NO removable
    # poles -- every p is genuine (even and odd), unlike the identical case.
    generic = [mp.mpf("0.0625"), mp.mpf("0.13"), mp.mpf("0.21"), mp.mpf("0.07")]
    poles = global_sl2_delta_poles(2, 6, h_externals=generic)
    locs = sorted(set(float(p) for p in poles))
    # Includes even-offset Delta (e.g. 2, 0, -2) that the identical case drops.
    assert 2.0 in locs and 0.0 in locs
    identical_locs = set(
        float(p) for p in global_sl2_delta_poles(2, 6))
    assert 0.0 not in identical_locs  # even offset removable when delta = 0


def test_q_order_defines_a_fixed_global_pole_universe():
    """A chiral 2h=-p pole first enters the q series at grade p+1."""
    pool = _global_q_order_pole_pool(
        2, 4, dh12=0, dh34=0, dhb12=0, dhb34=0,
    )
    # p=0,2 are removable for identical externals; q_order=4 therefore
    # includes only p=1,3 from each chiral half.
    assert sorted(float(pole) for pole in pool) == [-5.0, -3.0, -1.0, 1.0]


def test_contribution_optimal_global_prefix_uses_one_q_order_pool():
    mp.mp.dps = 40
    common = dict(
        coordinate="zzb", ell=2, h_externals=EXT, m_max=2, n_max=2,
        chiral_norm=CN, z_shift=-2 * EXT[0], q_order=12,
        shift=mp.mpf("2.5"), pole_selection="contribution_optimal",
    )
    short = global_scalar_G_interpolated_table(npoles=4, **common)
    long = global_scalar_G_interpolated_table(npoles=6, **common)
    assert list(short.prefactor.poles) == list(long.prefactor.poles[:4])
    low_order = {key: value for key, value in common.items() if key != "q_order"}
    with pytest.raises(ValueError, match="raise q_order"):
        global_scalar_G_interpolated_table(npoles=5, q_order=3, **low_order)


def test_analytic_genuine_poles_match_numeric_detector():
    """The analytic removable rule reproduces a direct numeric residue/order
    detector across external regimes (identical, pairwise, generic)."""
    mp.mp.dps = 60
    base = global_block_exp_base(CN)
    ell = 2

    def detected_orders(h_ext):
        e1 = mp.mpf(10) ** (-(mp.mp.dps // 3))
        e2 = e1 / 100
        orders = {}
        for p in range(0, 6):
            for d in (mp.mpf(ell - p), mp.mpf(-ell - p)):
                def s(e):
                    t = global_G_zzb_table(d + e, ell, h_ext, 2, 2,
                                           chiral_norm=CN, z_shift=0)
                    return max(abs(t[m][n]) for m in range(3)
                               for n in range(3)) / abs(base ** (d + e))
                s1 = s(e1)
                if s1 == 0:
                    continue
                k = int(mp.nint(mp.log(s(e2) / s1) / mp.log(e1 / e2)))
                if k >= 1:
                    orders[float(mp.re(d))] = max(orders.get(float(mp.re(d)), 0), k)
        return orders

    from collections import Counter
    configs = [
        [mp.mpf("0.0625")] * 4,                                   # identical
        [mp.mpf("0.0625"), mp.mpf("0.0625"), mp.mpf("0.1"), mp.mpf("0.1")],  # pairwise
        [mp.mpf("0.0625"), mp.mpf("0.13"), mp.mpf("0.21"), mp.mpf("0.07")],  # generic
    ]
    for h_ext in configs:
        det = detected_orders(h_ext)
        ana = Counter(
            float(mp.re(p))
            for p in global_sl2_delta_poles(ell, 30, h_externals=h_ext))
        # Every detector-genuine location (well inside the generated window)
        # appears in the analytic multiset with the same multiplicity.
        for loc, order in det.items():
            if loc < -4:  # only compare the shallow, fully-generated region
                continue
            assert ana.get(loc, 0) == order, (h_ext, loc, order, ana.get(loc, 0))


def test_num_threads_is_bit_identical():
    """Per-bucket parallelism (num_threads > 1) gives bit-identical output to
    the serial path -- the buckets are independent and the node search is
    deterministic."""
    mp.mp.dps = 60
    z_shift = -2 * EXT[0]
    serial = global_scalar_G_interpolated_table(
        "zzb", 2, EXT, 6, 6, chiral_norm=CN, z_shift=z_shift, npoles=10,
        shift=mp.mpf("2.5"), num_threads=1)
    parallel = global_scalar_G_interpolated_table(
        "zzb", 2, EXT, 6, 6, chiral_norm=CN, z_shift=z_shift, npoles=10,
        shift=mp.mpf("2.5"), num_threads=4)
    for rs, rp in zip(serial.numerators, parallel.numerators):
        for ps, pp in zip(rs, rp):
            assert [mp.mpf(c) for c in ps.coeffs] == [mp.mpf(c) for c in pp.coeffs]


@pytest.mark.parametrize("coord, builder", [
    ("zzb", global_G_zzb_table),
    ("yyb", global_G_yyb_table),
    ("qqb", global_G_qqb_table),
])
@pytest.mark.parametrize("ell, shift", [(0, mp.mpf("0.5")), (3, mp.mpf("3.5"))])
def test_interpolation_matches_numeric_block(coord, builder, ell, shift):
    """The interpolant reproduces the numeric block to bootstrap precision over
    the physical window, for even and odd spin and every global coordinate."""
    mp.mp.dps = 60
    z_shift = -2 * EXT[0]
    tab = global_scalar_G_interpolated_table(
        coord, ell, EXT, 2, 2, chiral_norm=CN, z_shift=z_shift,
        npoles=12, shift=shift,
    )
    worst = mp.mpf(0)
    for dval in ("0.5", "2.6", "11.0", "25.0"):
        d = mp.mpf(dval)
        if d < shift:
            continue
        approx = tab.evaluate(d)
        exact = builder(d, ell, EXT, 2, 2, chiral_norm=CN, z_shift=z_shift)
        for m in range(3):
            for n in range(3):
                denom = abs(exact[m][n]) + mp.mpf(10) ** (-200)
                worst = max(worst, abs(approx[m][n] - exact[m][n]) / denom)
    # Odd spin entries carry one lower degree (a = m+n-1) so are the weakest;
    # npoles=12 reaches ~1e-13 there, well below this bound.
    assert worst < mp.mpf(10) ** (-11), f"{coord} ell={ell}: worst rel {worst}"


def test_asymptotic_power_even_and_odd_spin():
    # The scalar G is z <-> zbar even at every spin: a = m + n, odd spin included (no parity cancellation).
    assert global_block_asymptotic_power(2, 3, 3) == 6
    assert global_block_asymptotic_power(0, 2, 1) == 3
    assert global_block_asymptotic_power(1, 3, 2) == 5
    assert global_block_asymptotic_power(3, 1, 0) == 1
    assert global_block_asymptotic_power(1, 0, 0) == 0


def test_large_delta_tail_is_bounded():
    """Per-entry degree = a + npoles keeps the large-Delta tail power correct,
    so the relative error stays bounded far past the node range rather than
    diverging (the historical 'global mode unbounded' failure)."""
    mp.mp.dps = 80
    z_shift = -2 * EXT[0]
    ell, shift = 2, mp.mpf("2.5")
    tab = global_scalar_G_interpolated_table(
        "zzb", ell, EXT, 2, 2, chiral_norm=CN, z_shift=z_shift, npoles=10,
        shift=shift,
    )
    worst = mp.mpf(0)
    for dval in ("40", "120"):
        d = mp.mpf(dval)
        approx = tab.evaluate(d)
        exact = global_G_zzb_table(d, ell, EXT, 2, 2, chiral_norm=CN,
                                   z_shift=z_shift)
        for m in range(3):
            for n in range(3):
                if exact[m][n] == 0:
                    continue
                worst = max(worst, abs(approx[m][n] - exact[m][n])
                            / abs(exact[m][n]))
    assert worst < mp.mpf(10) ** (-6), f"tail not bounded: worst rel {worst}"


def test_global_ordered_b_matches_numeric_block_and_tail_is_bounded():
    mp.mp.dps = 80
    ell = 2
    z_shift = -(SPIN_H_EXT[0] + SPIN_H_EXT[1])
    zbar_shift = -(SPIN_HBAR_EXT[0] + SPIN_HBAR_EXT[1])
    # Per-entry node counts (the default): the bounded-tail property comes from
    # the degree-tied-to-npoles design. shared_sample_points=True would trade it
    # away for in-window accuracy; pinned False here to assert the default tail.
    tab = global_ordered_B_interpolated_table(
        "zzb", ell, SPIN_H_EXT, SPIN_HBAR_EXT, 2, 2,
        chiral_norm=CN, z_shift=z_shift, zbar_shift=zbar_shift,
        npoles=10, shift=mp.mpf("2.5"), shared_sample_points=False,
    )

    def exact(delta):
        return _manual_global_ordered_zzb(
            delta, ell, SPIN_H_EXT, SPIN_HBAR_EXT, 2, 2,
            z_shift=z_shift, zbar_shift=zbar_shift,
        )

    window = _worst_rel_2d(
        tab, exact, ("2.6", "9.0", "20.0"), m_max=2, n_max=2,
    )
    tail = _worst_rel_2d(
        tab, exact, ("40", "120"), m_max=2, n_max=2,
    )
    assert window < mp.mpf(10) ** (-9)
    assert tail < mp.mpf(10) ** (-6)


def test_global_parity_projected_b_matches_numeric_block_and_tail_is_bounded():
    mp.mp.dps = 80
    ell = 2
    z_shift = -(SPIN_H_EXT[0] + SPIN_H_EXT[1])
    zbar_shift = -(SPIN_HBAR_EXT[0] + SPIN_HBAR_EXT[1])
    # Per-entry node counts (the default): the bounded tail is the
    # degree-tied-to-npoles property, which shared_sample_points=True trades away.
    tab = global_parity_projected_B_interpolated_table(
        "zzb", ell, SPIN_H_EXT, SPIN_HBAR_EXT, 2, 2,
        parity="+", parity_external=["1"] * 4, exchanged_parity="1",
        chiral_norm=CN, z_shift=z_shift, zbar_shift=zbar_shift,
        npoles=10, shift=mp.mpf("2.5"), shared_sample_points=False,
    )

    def exact(delta):
        return _manual_global_parity_zzb(
            delta, ell, SPIN_H_EXT, SPIN_HBAR_EXT, 2, 2,
            parity="+", parity_external=["1"] * 4,
            exchanged_parity="1", z_shift=z_shift, zbar_shift=zbar_shift,
        )

    window = _worst_rel_2d(
        tab, exact, ("2.6", "9.0", "20.0"), m_max=2, n_max=2,
    )
    tail = _worst_rel_2d(
        tab, exact, ("40", "120"), m_max=2, n_max=2,
    )
    assert window < mp.mpf(10) ** (-9)
    assert tail < mp.mpf(10) ** (-6)


def test_global_chiral_f_matches_numeric_block_and_tail_is_bounded():
    mp.mp.dps = 80
    z_shift = -(CHIRAL_EXT[0] + CHIRAL_EXT[1])
    tab = global_chiral_F_interpolated_table(
        "z", CHIRAL_EXT, 2, chiral_norm=CN, z_shift=z_shift,
        npoles=10,
    )

    def exact(h):
        return global_chiral_z_taylor(
            h, CHIRAL_EXT, 3, chiral_norm=CN, z_shift=z_shift,
        )

    window = _worst_rel_chiral(tab, exact, ("0.3", "2.0", "6.0"), m_max=2)
    tail = _worst_rel_chiral(tab, exact, ("20", "80"), m_max=2)
    assert window < mp.mpf(10) ** (-12)
    assert tail < mp.mpf(10) ** (-6)


def test_npoles_is_the_precision_knob():
    """Raising npoles lifts every entry's degree in lockstep, improving the
    in-window fit while preserving the tail -- the single sound precision knob."""
    mp.mp.dps = 70
    z_shift = -2 * EXT[0]
    ell, shift = 2, mp.mpf("2.5")

    def worst_window(npoles):
        tab = global_scalar_G_interpolated_table(
            "zzb", ell, EXT, 2, 2, chiral_norm=CN, z_shift=z_shift,
            npoles=npoles, shift=shift,
        )
        w = mp.mpf(0)
        for dval in ("2.6", "9.0"):
            d = mp.mpf(dval)
            approx = tab.evaluate(d)
            exact = global_G_zzb_table(d, ell, EXT, 2, 2, chiral_norm=CN,
                                       z_shift=z_shift)
            for m in range(3):
                for n in range(3):
                    if exact[m][n] == 0:
                        continue
                    w = max(w, abs(approx[m][n] - exact[m][n])
                            / abs(exact[m][n]))
        return w

    assert worst_window(12) < worst_window(6) * mp.mpf(10) ** (-3)


def _delta_poles(tab, shift):
    return sorted(round(float(mp.re(p)) + float(shift), 6) for p in tab.prefactor.poles)


def test_pole_selection_is_threaded():
    """Every selector keeps the same count and reproduces the block with a
    bounded tail. With removables stripped, the genuine-pole residues decrease
    monotonically with depth, so residue / contribution / rightmost coincide
    with nearest; real_first (breadth-first over locations) is the selector
    that genuinely differs, proving the choice is consumed (not ignored)."""
    mp.mp.dps = 60
    z_shift = -2 * EXT[0]
    ell, shift, npoles = 2, mp.mpf("2.5"), 6

    def make(sel):
        return global_scalar_G_interpolated_table(
            "zzb", ell, EXT, 2, 2, chiral_norm=CN, z_shift=z_shift,
            npoles=npoles, pole_selection=sel, shift=shift,
        )

    tabs = {sel: make(sel) for sel in
            ("nearest", "rightmost", "residue", "contribution", "real_first")}
    # Same total pole count for every selector (so degree / tail are identical).
    for tab in tabs.values():
        assert len(tab.prefactor.poles) == npoles
    near = _delta_poles(tabs["nearest"], shift)
    # nearest == the standalone nearest-pole helper.
    assert near == sorted(
        round(float(p), 6) for p in global_sl2_delta_poles(ell, npoles, shift=shift))
    # residue / rightmost coincide with nearest (monotonic genuine residues).
    assert _delta_poles(tabs["residue"], shift) == near
    assert _delta_poles(tabs["rightmost"], shift) == near
    # real_first spreads across locations -> a genuinely different set, proving
    # the selector is threaded through (not silently ignored).
    assert _delta_poles(tabs["real_first"], shift) != near
    # Every selector still reproduces the block (window) with a bounded tail.
    for tab in tabs.values():
        for dval in ("2.6", "40"):
            d = mp.mpf(dval)
            approx = tab.evaluate(d)
            exact = global_G_zzb_table(d, ell, EXT, 2, 2, chiral_norm=CN,
                                       z_shift=z_shift)
            for m in range(3):
                for n in range(3):
                    if exact[m][n] == 0:
                        continue
                    rel = abs(approx[m][n] - exact[m][n]) / abs(exact[m][n])
                    assert rel < mp.mpf(10) ** (-4)


def test_pole_selection_rejects_unknown():
    mp.mp.dps = 30
    with pytest.raises(ValueError, match="pole_selection"):
        global_scalar_G_interpolated_table(
            "zzb", 0, EXT, 1, 1, npoles=8, pole_selection="bogus")


def test_cli_global_basis_pole_selection_changes_blocks(tmp_path):
    """--pole-selection is consumed on the global path (no longer silently
    ignored): real_first selects a different pole set from nearest."""
    for sel in ("nearest", "real_first"):
        cli.main([
            "table", "--family", "global",
            "--central-charge", "10", "--spin-ranges", "2", "--lambda", "3",
            "--delta-ext", "0.15", "0.15", "0.15", "0.15",
            "--basis", "global_scalar_G", "--coordinates", "zzb",
            "--npoles", "6", "--precision", "50",
            "--pole-selection", sel, "--format", "json",
            "-o", str(tmp_path / sel),
        ])
    a = json.loads((tmp_path / "nearest" / "derivs_2.json").read_text())
    b = json.loads((tmp_path / "real_first" / "derivs_2.json").read_text())
    assert a["pole_list_x"] != b["pole_list_x"]


def test_cli_global_defaults_to_contribution_optimal_with_its_q_pool(tmp_path):
    common = [
        "table", "--family", "global", "--spin-ranges", "2", "--lambda", "2",
        "--delta-ext", "0.15", "0.15", "0.15", "0.15",
        "--basis", "global_scalar_G", "--coordinates", "zzb", "--order", "12",
        "--npoles", "4", "--precision", "40", "--format", "json",
    ]
    cli.main(common + ["-o", str(tmp_path / "default")])
    cli.main(common + ["--pole-selection", "contribution_optimal", "-o", str(tmp_path / "explicit")])
    default = json.loads((tmp_path / "default" / "derivs_2.json").read_text())
    explicit = json.loads((tmp_path / "explicit" / "derivs_2.json").read_text())
    assert default["pole_list_x"] == explicit["pole_list_x"]
    assert default["order"] == 12


@pytest.mark.parametrize("coord,ell,numeric", [
    ("xt", 2, global_G_xt_table),
    ("ws", 3, global_G_ws_table),
    ("qqb_ws", 2, global_G_qqb_ws_table),
])
def test_folded_scalar_G_interpolated_matches_numeric(coord, ell, numeric):
    """The folded scalar_G interpolated table reproduces the per-Delta numeric
    folded global block (an independent path) inside the window."""
    mp.mp.dps = 60
    z_shift = -2 * EXT[0]
    mn = 2
    tab = global_scalar_G_interpolated_table(
        coord, ell, EXT, mn, mn, chiral_norm=CN, z_shift=z_shift, npoles=10,
    )
    worst = mp.mpf(0)
    for delta in (mp.mpf(ell) + mp.mpf("0.4"), mp.mpf(ell) + 3, mp.mpf(ell) + 7):
        got = tab.evaluate(delta)
        ref = numeric(delta, ell, EXT, mn, mn, chiral_norm=CN, z_shift=z_shift)
        scale = max(abs(ref[m][n]) for m in range(mn + 1) for n in range(mn + 1))
        for m in range(mn + 1):
            for n in range(mn + 1):
                worst = max(worst, abs(got[m][n] - ref[m][n]) / scale)
    assert worst < mp.mpf(10) ** -15


@pytest.mark.parametrize("coord,ell,parity,numeric", [
    ("xt", 2, "+", global_parity_projected_B_xt_table),
    ("ws", 3, "-", global_parity_projected_B_ws_table),
])
def test_folded_parity_projected_interpolated_matches_numeric(
    coord, ell, parity, numeric
):
    """The folded parity-projected interpolated table reproduces the per-Delta
    numeric folded global parity-projected block."""
    mp.mp.dps = 60
    z_shift = -2 * EXT[0]
    mn = 2
    pe = ["1", "1", "1", "1"]
    tab = global_parity_projected_B_interpolated_table(
        coord, ell, EXT, EXT, mn, mn, parity=parity, parity_external=pe,
        chiral_norm=CN, z_shift=z_shift, zbar_shift=z_shift, npoles=10,
    )
    worst = mp.mpf(0)
    for delta in (mp.mpf(ell) + mp.mpf("0.4"), mp.mpf(ell) + 3, mp.mpf(ell) + 7):
        got = tab.evaluate(delta)
        ref = numeric(
            delta, ell, EXT, EXT, mn, mn, parity=parity, parity_external=pe,
            chiral_norm=CN, z_shift=z_shift, zbar_shift=z_shift,
        )
        scale = max(abs(ref[m][n]) for m in range(mn + 1) for n in range(mn + 1))
        for m in range(mn + 1):
            for n in range(mn + 1):
                worst = max(worst, abs(got[m][n] - ref[m][n]) / scale)
    assert worst < mp.mpf(10) ** -15


def test_global_ordered_B_interpolated_rejects_folded():
    """Ordered B has no parity fold."""
    mp.mp.dps = 30
    with pytest.raises(ValueError, match="no parity fold"):
        global_ordered_B_interpolated_table(
            "xt", 2, SPIN_H_EXT, SPIN_HBAR_EXT, 2, 2, npoles=8,
        )


def test_global_offdiagonal_scalar_interpolated_matches_pointwise_oracle():
    mp.mp.dps = 80
    z_shift = -2 * EXT[0]
    ell = 2
    z0, zbar0 = mp.mpf("0.37"), mp.mpf("0.55")
    tab = global_scalar_G_interpolated_table(
        "qqb", ell, EXT, 2, 2,
        chiral_norm=CN, z_shift=z_shift, npoles=10, shift=mp.mpf("2.5"),
        center=z0, center_bar=zbar0,
    )
    sign = mp.mpf(1) if ell % 2 == 0 else mp.mpf(-1)
    for delta in (mp.mpf("2.8"), mp.mpf("8.3")):
        got = tab.evaluate(delta)
        h, hbar = (delta + ell) / 2, (delta - ell) / 2
        b1 = global_evaluate_block(
            z0, zbar0, h, hbar, EXT,
            coord="qqb", m_max=2, n_max=2, chiral_norm=CN,
            z_shift=z_shift, zbar_shift=z_shift,
        )
        b2 = global_evaluate_block(
            z0, zbar0, hbar, h, EXT,
            coord="qqb", m_max=2, n_max=2, chiral_norm=CN,
            z_shift=z_shift, zbar_shift=z_shift,
        )
        ref = [[b1[m][n] + sign * b2[m][n] for n in range(3)] for m in range(3)]
        scale = max(abs(ref[m][n]) for m in range(3) for n in range(3))
        err = max(abs(got[m][n] - ref[m][n]) for m in range(3) for n in range(3))
        assert err / scale < mp.mpf(10) ** -9


def test_global_offdiagonal_ordered_spinning_interpolated_matches_pointwise_oracle():
    mp.mp.dps = 80
    ell = 2
    z0, zbar0 = mp.mpf("0.37"), mp.mpf("0.55")
    z_shift = -(SPIN_H_EXT[0] + SPIN_H_EXT[1])
    zbar_shift = -(SPIN_HBAR_EXT[0] + SPIN_HBAR_EXT[1])
    tab = global_ordered_B_interpolated_table(
        "qqb", ell, SPIN_H_EXT, SPIN_HBAR_EXT, 2, 2,
        chiral_norm=CN, z_shift=z_shift, zbar_shift=zbar_shift,
        npoles=10, shift=mp.mpf("2.5"), center=z0, center_bar=zbar0,
    )
    for delta in (mp.mpf("2.8"), mp.mpf("8.3")):
        got = tab.evaluate(delta)
        h, hbar = (delta + ell) / 2, (delta - ell) / 2
        ref = global_evaluate_block(
            z0, zbar0, h, hbar, SPIN_H_EXT, hbar_externals=SPIN_HBAR_EXT,
            coord="qqb", m_max=2, n_max=2, chiral_norm=CN,
            z_shift=z_shift, zbar_shift=zbar_shift,
        )
        scale = max(abs(ref[m][n]) for m in range(3) for n in range(3))
        err = max(abs(got[m][n] - ref[m][n]) for m in range(3) for n in range(3))
        assert err / scale < mp.mpf(10) ** -9


def test_global_offdiagonal_projected_spinning_interpolated_matches_pointwise_oracle():
    mp.mp.dps = 80
    ell = 2
    z0, zbar0 = mp.mpf("0.37"), mp.mpf("0.55")
    z_shift = -(SPIN_H_EXT[0] + SPIN_H_EXT[1])
    zbar_shift = -(SPIN_HBAR_EXT[0] + SPIN_HBAR_EXT[1])
    parity = "+"
    phase = parity_reversal_phase(["1", "1", "1", "1"], exchanged_parity="1")
    sigma = (mp.mpf(1) if parity == "+" else mp.mpf(-1)) * phase
    tab = global_parity_projected_B_interpolated_table(
        "qqb", ell, SPIN_H_EXT, SPIN_HBAR_EXT, 2, 2,
        parity=parity, parity_external=["1"] * 4, exchanged_parity="1",
        chiral_norm=CN, z_shift=z_shift, zbar_shift=zbar_shift,
        npoles=10, shift=mp.mpf("2.5"), center=z0, center_bar=zbar0,
    )
    for delta in (mp.mpf("2.8"), mp.mpf("8.3")):
        got = tab.evaluate(delta)
        h, hbar = (delta + ell) / 2, (delta - ell) / 2
        b1 = global_evaluate_block(
            z0, zbar0, h, hbar, SPIN_H_EXT, hbar_externals=SPIN_HBAR_EXT,
            coord="qqb", m_max=2, n_max=2, chiral_norm=CN,
            z_shift=z_shift, zbar_shift=zbar_shift,
        )
        b2 = global_evaluate_block(
            z0, zbar0, hbar, h, SPIN_HBAR_EXT, hbar_externals=SPIN_H_EXT,
            coord="qqb", m_max=2, n_max=2, chiral_norm=CN,
            z_shift=zbar_shift, zbar_shift=z_shift,
        )
        ref = [[b1[m][n] + sigma * b2[m][n] for n in range(3)] for m in range(3)]
        scale = max(abs(ref[m][n]) for m in range(3) for n in range(3))
        err = max(abs(got[m][n] - ref[m][n]) for m in range(3) for n in range(3))
        assert err / scale < mp.mpf(10) ** -9


def test_unknown_coordinate_rejected():
    mp.mp.dps = 30
    with pytest.raises(ValueError, match="global block coordinate"):
        global_scalar_G_interpolated_table("nope", 0, EXT, 1, 1, npoles=8)


def test_serialized_prefactor_and_reconstruction():
    """Serialization carries the global base + SL(2) poles, and the serialized
    numerators reconstruct the block (the canonical evaluation path)."""
    mp.mp.dps = 60
    z_shift = -2 * EXT[0]
    shift = mp.mpf("2.5")
    tab = global_scalar_G_interpolated_table(
        "zzb", 2, EXT, 2, 2, chiral_norm=CN, z_shift=z_shift, npoles=10,
        shift=shift,
    )
    ser = serialize_deriv_tables({"zzb": tab})
    assert ser.coordinates == ("zzb",)
    assert abs(mp.mpf(ser.base) - (3 - 2 * mp.sqrt(2))) < mp.mpf(10) ** (-40)
    assert abs(mp.mpf(ser.shift) - shift) < mp.mpf(10) ** (-40)
    assert len(ser.pole_list_x) == 10  # npoles counts total kept poles

    base = mp.mpf(ser.base)
    poles_x = [mp.mpf(p) for p in ser.pole_list_x]
    d = mp.mpf("5.0")
    x = d - mp.mpf(ser.shift)
    exact = global_G_zzb_table(d, 2, EXT, 2, 2, chiral_norm=CN, z_shift=z_shift)
    for m in range(3):
        for n in range(3):
            num = ser.numerators["zzb"][m][n]
            val = base ** x
            for p in poles_x:
                val /= (x - p)
            val *= num.evaluate(x)
            denom = abs(exact[m][n]) + mp.mpf(10) ** (-200)
            assert abs(val - exact[m][n]) / denom < mp.mpf(10) ** (-13)


def _reconstruct_from_json(doc, coord_index=0):
    """Map the jagged JSON `derivs` back to a dict[(m, n)] -> coeff list using
    the same kept-pair policy the writer emits with."""
    base = mp.mpf(doc["exp_base"])
    shift = mp.mpf(doc["variable_minus_x"])
    poles_x = [mp.mpf(p) for p in doc["pole_list_x"]]
    lam = doc["lambda"]
    ell = doc["spin"]
    pairs = kept_pairs("zzb", lam, basis="global_scalar_G", ell=ell)
    rows_by_m: dict[int, list[int]] = {}
    for (m, n) in pairs:
        rows_by_m.setdefault(m, []).append(n)
    derivs = doc["derivs"][coord_index]

    def block(m_target, n_target, d):
        mi = sorted(rows_by_m).index(m_target)
        ni = sorted(rows_by_m[m_target]).index(n_target)
        coeffs = [mp.mpf(c) for c in derivs[mi][ni]]
        x = d - shift
        val = base ** x
        for p in poles_x:
            val /= (x - p)
        val *= sum(coeffs[k] * x ** k for k in range(len(coeffs)))
        return val

    return block


def test_json_writer_roundtrip_reconstructs_block():
    """A full JSON write -> read -> reconstruct round-trip reproduces the
    numeric block."""
    mp.mp.dps = 80
    z_shift = -2 * EXT[0]
    shift = mp.mpf("2.5")
    tab = global_scalar_G_interpolated_table(
        "zzb", 2, EXT, 2, 2, chiral_norm=CN, z_shift=z_shift, npoles=8,
        shift=shift,
    )
    ser = serialize_deriv_tables({"zzb": tab})
    doc = json.loads(write_json(
        ser, c=mp.mpf(10), h_ext=EXT, hbar_ext=EXT, spin=2, q_order=0,
        npoles=8, basis="global_scalar_G",
    ))
    assert doc["basis"] == "global_scalar_G"
    assert doc["label"] == "delta_l"
    assert doc["central_charge"] is None
    assert doc["order"] == 0
    block = _reconstruct_from_json(doc)
    for dval in ("2.5", "9.0"):
        d = mp.mpf(dval)
        exact = global_G_zzb_table(d, 2, EXT, 2, 2, chiral_norm=CN,
                                   z_shift=z_shift)
        for (m, n) in [(0, 0), (1, 1), (2, 0)]:
            denom = abs(exact[m][n]) + mp.mpf(10) ** (-200)
            assert abs(block(m, n, d) - exact[m][n]) / denom < mp.mpf(10) ** (-12)


def test_mathematica_writer_accepts_global_basis():
    mp.mp.dps = 60
    tab = global_scalar_G_interpolated_table(
        "zzb", 2, EXT, 2, 2, chiral_norm=CN, npoles=6, shift=mp.mpf("2.5"),
    )
    ser = serialize_deriv_tables({"zzb": tab})
    text = write_mathematica(ser, coordinate="zzb", ell=2,
                             basis="global_scalar_G")
    assert "basis -> \"global_scalar_G\"" in text


def test_global_coordinates_set():
    assert GLOBAL_COORDINATES == frozenset({"zzb", "yyb", "qqb"})


def test_global_folded_blocks_match_block_value_and_global_control():
    """Folded global blocks (xt/ws) agree with zzb at the symmetric point and
    flow through the global control."""
    import mpmath as mp
    from virasoro_blocks.derivs import (
        global_G_xt_table,
        global_G_ws_table,
        global_G_zzb_table,
    )
    from virasoro_bootstrap.controls.global_control import (
        build_global_scalar_spin_channel,
    )

    ext = [mp.mpf("0.1")] * 4
    d, ell, zs = mp.mpf("3.0"), 2, mp.mpf("-0.2")
    # The (0, 0) Taylor entry is the block value at the common symmetric point
    # z = zbar = 1/2, so it is coordinate-independent across zzb / xt / ws.
    base = global_G_zzb_table(d, ell, ext, 3, 3, chiral_norm=4, z_shift=zs)[0][0]
    for folded in (global_G_xt_table, global_G_ws_table):
        tab = folded(d, ell, ext, 3, 3, chiral_norm=4, z_shift=zs)
        assert abs(tab[0][0] - base) < mp.mpf("1e-40")

    # The global control accepts the folded coordinates end to end.
    for coord in ("xt", "ws"):
        channel = build_global_scalar_spin_channel(
            coordinate=coord, ell=0, delta_min=mp.mpf("0.5"),
            h_ext=mp.mpf("0.0625"), lam=4,
        )
        assert channel.ell == 0
        assert channel.serialized.numerators[coord]


def test_cli_global_basis_end_to_end(tmp_path):
    """The canonical global table operation writes JSON and Mathematica; it
    needs no --central-charge (the global block is c-independent: the record
    carries central_charge None) and records the default pole q-order 30.
    A given --central-charge is covered by the notes-ignored-c test below."""
    cli.main([
        "table", "--family", "global",
        "--spin-ranges", "0:2:2",
        "--lambda", "3",
        "--delta-ext", "0.125", "0.125", "0.125", "0.125",
        "--basis", "global_scalar_G",
        "--coordinates", "zzb",
        "--chiral-norm", "4",
        "--npoles", "6",
        "--precision", "50",
        "--format", "both",
        "-o", str(tmp_path),
    ])
    assert (tmp_path / "derivs_0.json").exists()
    assert (tmp_path / "derivs_2.json").exists()
    ms = list(tmp_path.glob("*basisGlobalScalarG*.m"))
    assert len(ms) == 2
    doc = json.loads((tmp_path / "derivs_2.json").read_text())
    assert doc["basis"] == "global_scalar_G"
    assert doc["central_charge"] is None
    assert doc["order"] == 30
    assert all("DerivTable-c" not in path.name for path in ms)
    assert all("-order30" in path.name for path in ms)


def test_cli_global_basis_notes_ignored_c_and_uses_order_for_its_pole_pool(tmp_path, capsys):
    cli.main([
        "table", "--family", "global",
        "--central-charge", "10",
        "--order", "99",
        "--spin-ranges", "0",
        "--lambda", "3",
        "--delta-ext", "0.125", "0.125", "0.125", "0.125",
        "--basis", "global_scalar_G",
        "--coordinates", "zzb",
        "--npoles", "6",
        "--precision", "40",
        "--format", "json",
        "-o", str(tmp_path),
    ])
    captured = capsys.readouterr()
    assert "independent of --central-charge" in captured.err
    assert "ignore --order" not in captured.err
    doc = json.loads((tmp_path / "derivs_0.json").read_text())
    assert doc["order"] == 99


def test_cli_new_global_bases_end_to_end(tmp_path):
    ordered_dir = tmp_path / "ordered"
    cli.main([
        "table", "--family", "global",
        "--central-charge", "10",
        "--spin-ranges", "0:2:2",
        "--lambda", "3",
        "--delta-ext", "0.15", "1.25", "0.35", "1.45",
        "--spin-ext", "0", "1", "0", "1",
        "--basis", "global_ordered_B",
        "--coordinates", "zzb",
        "--u-shift", "-0.2",
        "--z-shift", "-1.2",
        "--npoles", "6",
        "--precision", "50",
        "--format", "json",
        "-o", str(ordered_dir),
    ])
    ordered_doc = json.loads((ordered_dir / "derivs_2.json").read_text())
    assert ordered_doc["basis"] == "global_ordered_B"
    assert ordered_doc["label"] == "h_hbar"
    assert len(ordered_doc["derivs"][0][0]) == 4

    parity_dir = tmp_path / "parity"
    cli.main([
        "table", "--family", "global",
        "--central-charge", "10",
        "--spin-ranges", "0:2:2",
        "--lambda", "3",
        "--delta-ext", "0.15", "1.25", "0.35", "1.45",
        "--spin-ext", "0", "1", "0", "1",
        "--parity-ext", "1", "1", "1", "1",
        "--parity-project", "+",
        "--parity-exchanged", "1",
        "--basis", "global_parity_projected_B",
        "--coordinates", "zzb",
        "--u-shift", "-0.2",
        "--z-shift", "-1.2",
        "--npoles", "6",
        "--precision", "50",
        "--format", "json",
        "-o", str(parity_dir),
    ])
    parity_doc = json.loads((parity_dir / "derivs_2.json").read_text())
    assert parity_doc["basis"] == "global_parity_projected_B"
    assert parity_doc["label"] == "parity_pm"
    assert parity_doc["parity_projection"] == "+"

    chiral_dir = tmp_path / "chiral"
    cli.main([
        "table", "--family", "global",
        "--central-charge", "10",
        "--lambda", "3",
        "--basis", "global_chiral_F",
        "--h-ext", "0.05", "0.10", "0.15", "0.20",
        "--coordinates", "z",
        "--z-shift", "-0.15",
        "--npoles", "6",
        "--precision", "50",
        "--format", "json",
        "-o", str(chiral_dir),
    ])
    chiral_doc = json.loads((chiral_dir / "derivs_chiral.json").read_text())
    assert chiral_doc["basis"] == "global_chiral_F"
    assert chiral_doc["label"] == "chiral_h"
    assert chiral_doc["spin"] is None
    assert len(chiral_doc["derivs"][0]) == 4


def test_cli_global_basis_rejects_spinning_externals(tmp_path):
    with pytest.raises(SystemExit):
        cli.main([
            "table", "--family", "global",
            "--central-charge", "10",
            "--spin-ranges", "0",
            "--lambda", "3",
            "--delta-ext", "0.125", "0.125", "0.125", "0.125",
            "--spin-ext", "0", "2", "0", "2",
            "--basis", "global_scalar_G",
            "--coordinates", "zzb",
            "-o", str(tmp_path),
        ])


def test_cli_global_scalar_G_accepts_folded_coordinate(tmp_path):
    """global_scalar_G now emits the parity-folded coordinates (the global
    interpolation core folds via the same _fold_parity recombination)."""
    cli.main([
        "table", "--family", "global",
        "--central-charge", "10",
        "--spin-ranges", "2",
        "--lambda", "2",
        "--delta-ext", "0.125", "0.125", "0.125", "0.125",
        "--basis", "global_scalar_G",
        "--coordinates", "xt", "ws", "qqb_ws", "xt_radial",
        # a folding smoke test: an order-reduced table (without --npoles a global table grows its poles to deliver
        # --precision, a 50 s build)
        "--npoles", "8",
        "--format", "json",
        "-o", str(tmp_path),
    ])
    assert (tmp_path / "derivs_2.json").exists()


def test_cli_global_ordered_B_rejects_folded_coordinate(tmp_path):
    """Ordered B has no parity fold, so folded coordinates stay rejected."""
    with pytest.raises(SystemExit):
        cli.main([
            "table", "--family", "global",
            "--central-charge", "10",
            "--spin-ranges", "1",
            "--lambda", "3",
            "--h-ext", "0.6", "0.6", "0.6", "0.6",
            "--hbar-ext", "0.1", "0.1", "0.1", "0.1",
            "--basis", "global_ordered_B",
            "--coordinates", "xt",
            "-o", str(tmp_path),
        ])


def test_cli_global_basis_rejects_full_method_and_regularize_on(tmp_path):
    with pytest.raises(SystemExit):
        cli.main([
            "table", "--family", "global",
            "--spin-ranges", "0",
            "--lambda", "3",
            "--delta-ext", "0.125", "0.125", "0.125", "0.125",
            "--basis", "global_scalar_G",
            "--coordinates", "zzb",
            "--npoles", "all",
            "-o", str(tmp_path / "full"),
        ])
    with pytest.raises(SystemExit):
        cli.main([
            "table", "--family", "global",
            "--spin-ranges", "0",
            "--lambda", "3",
            "--delta-ext", "0.125", "0.125", "0.125", "0.125",
            "--basis", "global_scalar_G",
            "--coordinates", "zzb",
            "--regularize", "on",
            "-o", str(tmp_path / "regularize_on"),
        ])


def test_global_shared_sample_points_pads_to_max_degree():
    """`shared_sample_points` fits every global (m,n) entry on the single
    max-degree node set (one node search, all entries uniform degree) instead of
    the per-entry degree buckets; the reconstructed block is unchanged at the
    nodes, only the off-node high-order coefficients differ."""
    mp.mp.dps = 60
    h_ext = [mp.mpf("0.15")] * 4

    per = global_scalar_G_interpolated_table(
        "zzb", 0, h_ext, 4, 4, npoles=6, shared_sample_points=False)
    shared = global_scalar_G_interpolated_table(
        "zzb", 0, h_ext, 4, 4, npoles=6, shared_sample_points=True)

    per_degs = {p.degree for row in per.numerators for p in row}
    shared_degs = {p.degree for row in shared.numerators for p in row}
    assert len(per_degs) > 1                      # per-entry: many degrees
    assert len(shared_degs) == 1                  # shared: one (max) degree
    assert max(per_degs) == next(iter(shared_degs))

    # Every shared entry uses one and the same node set.
    shared_sets = {s for row in shared.numerator_sample_points for s in row}
    assert len(shared_sets) == 1

    # Both fit the same numeric global block, so they agree in the node range
    # up to their (sub-percent) interpolation-error difference -- not bit-equal.
    for Delta in (mp.mpf("0.7"), mp.mpf("2.0"), mp.mpf("5.0")):
        a = per.evaluate(Delta)
        b = shared.evaluate(Delta)
        worst = max(
            abs(mp.mpc(x) - mp.mpc(y))
            for ra, rb in zip(a, b) for x, y in zip(ra, rb)
        )
        scale = max(abs(mp.mpc(x)) for ra in a for x in ra)
        assert worst / scale < mp.mpf("1e-6"), Delta


@pytest.mark.parametrize("coord,evfac,taylor", [
    ("z", "global_chiral_z_series_evaluator", "global_chiral_z_taylor"),
    ("y", "global_chiral_y_series_evaluator", "global_chiral_y_taylor"),
    ("q", "global_chiral_q_series_evaluator", "global_chiral_q_taylor"),
])
def test_global_chiral_series_evaluator_matches_taylor(coord, evfac, taylor):
    """The batched arb-native global chiral evaluator reproduces the mpmath
    `global_chiral_*_taylor` (real `.many`, complex `return_acb`, default
    mpmath)."""
    import virasoro_blocks.derivs as d
    from virasoro_blocks.polynomial import _arb_to_mpf, _acb_to_mpc

    mp.mp.dps = 50
    z_shift = -2 * EXT[0]
    n = 9
    ev = getattr(d, evfac)(EXT, n, chiral_norm=CN, z_shift=z_shift)
    ref = getattr(d, taylor)

    hs = [mp.mpf("0.8"), mp.mpf("1.6"), mp.mpf("3.2")]
    many = ev.many(hs)
    for h, arr in zip(hs, many):
        exact = ref(h, EXT, n, chiral_norm=CN, z_shift=z_shift)
        for k in range(n):
            assert abs(_arb_to_mpf(arr[k]) - exact[k]) < mp.mpf("1e-30")

    # complex weight -> .many declines (None), evaluate(return_acb=True) works.
    assert ev.many([mp.mpc("0.8", "0.3")]) is None
    acbc = ev(mp.mpc("0.8", "0.3"), return_acb=True)
    refc = ref(mp.mpc("0.8", "0.3"), EXT, n, chiral_norm=CN, z_shift=z_shift)
    for k in range(n):
        assert abs(_acb_to_mpc(acbc[k]) - refc[k]) < mp.mpf("1e-30")
