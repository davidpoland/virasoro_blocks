"""The recursion's level tables and the folded coordinates are certified from their own balls (2026-09-29).

References use exact rational formulas or the same source labels and stored binary inputs at higher
working precision; the cases expose lost digits or incorrect ball enclosures before their respective fixes.
"""

from __future__ import annotations

from fractions import Fraction

import mpmath as mp
import pytest

pytestmark = pytest.mark.blocks

C_NEAR_ONE = mp.mpf(1) + mp.mpf(2) ** -40            # 1 + 2^-40, exact
DIFF_EXT = ["0.125", "0.375", "1.25", "0.8125"]


def _digits(value, reference):
    if value == reference:
        return mp.inf
    return -mp.log10(abs(value - reference) / abs(reference))


def _worst_digits(values, references):
    return min((_digits(v, r) for v, r in zip(values, references) if r != 0), default=mp.inf)


def _table_digits(table, reference):
    scale = max(abs(x) for row in reference for x in row)
    error = max(abs(x - y) for row, ref_row in zip(table, reference) for x, y in zip(row, ref_row))
    return mp.inf if error == 0 else -mp.log10(error / scale)


def test_heavy_external_block_keeps_its_digits():
    """F_block at h_i = 5, h = 50, z = 119/128, c = 2: the H table cancels ~1 digit per q-order (12 of 30 digits
    at q_order 40, none at 100 before)."""
    from virasoro_blocks.recursion import F_block

    def value(dps):
        with mp.workdps(dps):
            return F_block(mp.mpf(50), mp.mpf("0.9296875"), [mp.mpf(5)] * 4, c=mp.mpf(2), q_order=40)

    low, reference = value(30), value(90)
    with mp.workdps(100):
        assert _digits(low, reference) > 28


def test_H_q_series_is_certified_at_generic_c():
    """H_q_series at c = 8, h_i = 1, q_order 100, dps 30: the worst coefficient had -6.7 correct digits (the table
    and the rounding of b = b(c) at the working precision)."""
    from virasoro_blocks.recursion import H_q_series

    def series(dps):
        with mp.workdps(dps):
            return H_q_series(mp.mpf("2.25"), [mp.mpf(1)] * 4, c=mp.mpf(8), q_order=100)

    low, reference = series(30), series(60)
    with mp.workdps(70):
        assert _worst_digits(low[1:], reference[1:]) > 28


def test_ns_ramond_and_torus_level_tables_are_certified():
    """The generic pole kernel (poles.fill_level_table_flint): NS at Delta_i = 5 had no correct digit in its worst
    entry at dps 30, the torus one-point table at c = 8, d = 5, q_order 60 -8.5, Ramond at Delta_i = 3 15.8."""
    from virasoro_blocks.recursion_ns import build_ns_level_table
    from virasoro_blocks.recursion_r import build_r_level_table
    from virasoro_blocks.torus_onept import build_torus_onept_kernel

    builds = (
        lambda: build_ns_level_table(mp.mpf("1.296875"), [mp.mpf(5)] * 4, 40).table,
        lambda: build_r_level_table(mp.mpf("1.296875"), [mp.mpf(3)] * 4, 30).table,
        lambda: build_torus_onept_kernel(40, d_ext=mp.mpf(5), c=mp.mpf(8)).f_table,
    )
    for build in builds:
        with mp.workdps(30):
            low = build()
        with mp.workdps(90):
            reference = build()
        with mp.workdps(100):
            worst = min(_worst_digits(a, b) for a, b in zip(low, reference))
        assert worst > 28


@pytest.mark.parametrize('family', ['torus', 'ns', 'ramond', 'sphere'])
def test_kac_residue_balls_enclose_exact_shifts_from_fusion_zeros(family):
    from flint import acb, arb, ctx
    from virasoro_blocks.params import b_from_c
    from virasoro_blocks.recursion_ns import build_ns_level_table
    from virasoro_blocks.recursion_r import build_r_level_table
    from virasoro_blocks.torus_onept import build_torus_onept_kernel

    if family == 'sphere':
        from virasoro_blocks.recursion import _build_rs_data

        t = Fraction(49, 100)
        external = (3*t/4+2+1/t)/4
        with mp.workdps(60):
            absent = _build_rs_data(2, Fraction(7, 10), [external]*4)
            table = _build_rs_data(2, Fraction(7, 10), [external+Fraction(1, 10**70)]*4)
        index = next(i for i, p in enumerate(table.pairs) if (p['r'], p['s']) == (2, 1))
        assert next(iter(absent.kac_balls.values()))[0][index].is_zero()
        actual = next(iter(table.kac_balls.values()))[0][index]
        # P_21=t*shift and the sphere recursion absorbs 16^2.
        expected = 64*t*t*Fraction(1, 10**140)/(t*t-1)
        previous = ctx.prec
        try:
            ctx.prec = 1000
            assert actual.overlaps(acb(arb(expected.numerator)/expected.denominator))
        finally:
            ctx.prec = previous
        return

    shift = Fraction(1, 10**70)
    with mp.workdps(60):
        if family == 'torus':
            t = Fraction(49, 100)
            d = 2-2*t+shift
            kernel = build_torus_onept_kernel(2, b=b_from_c(13+6*(t+1/t)), d_ext=d)
            level, pair, sector = kernel.level, (2, 1), 0
            # A_21 = 1/[4(t²-1)], P_lower = d(d+1+2t),
            # P_upper = (d-1)(d-2+2t).
            expected = d*(d+1+2*t)*(d-1)*(d-2+2*t)/(4*(t*t-1))
        elif family == 'ns':
            level = build_ns_level_table(Fraction(27, 20), [Fraction(3, 32)+shift]*4, 3)
            pair, sector = (1, 3), 1
            # beta²=5/4, A_13=-25/18, P_unstarred=-8 shift/5;
            # the recursion absorbs 4^3.
            expected = -Fraction(2048, 9)*shift**2
        else:
            level = build_r_level_table(Fraction(27, 20), [Fraction(117, 640)+shift]*4, 4)
            pair, sector = (2, 2), 0
            # HJS b²=-5/4, A_22=400/81, P_++=-shift/40;
            # the recursion absorbs 4^4.
            expected = Fraction(64, 81)*shift**2
    index = next(i for i, p in enumerate(level.poles)
                 if (p['r'], p['s']) == pair and p.get('sector', 0) == sector)
    previous = ctx.prec
    try:
        ctx.prec = 1000
        reference = acb(arb(expected.numerator)/arb(expected.denominator))
        assert level.flint_kd[0][index].overlaps(reference)
    finally:
        ctx.prec = previous


def test_rational_ball_conversion_preserves_the_source_at_lower_mpmath_precision():
    from flint import acb, arb, ctx
    from virasoro_blocks.polynomial import _mpf_to_acb

    previous = ctx.prec
    try:
        ctx.prec = 200
        with mp.workdps(15):
            actual = _mpf_to_acb(Fraction(1, 10))
        ctx.prec = 1000
        assert actual.overlaps(acb(arb(1)/10))
    finally:
        ctx.prec = previous


@pytest.mark.parametrize('parameter', ['real', 'strip', 'imaginary', 'independent'])
def test_sphere_kac_balls_enclose_exact_source_labels(parameter):
    from flint import acb, arb, ctx
    from virasoro_blocks.params import b_from_c
    from virasoro_blocks.recursion import _build_rs_data
    from virasoro_blocks.numeric_input import literal_components

    external = tuple(Fraction(i, 10) for i in range(1, 5))
    with mp.workdps(100):
        imaginary_b = mp.mpc(0, mp.mpf('.73123'))
    with mp.workdps(30):
        charges = {'real': Fraction(327, 10), 'strip': Fraction(83, 10), 'imaginary': Fraction(7, 17)}
        parameters = (Fraction(7, 10), imaginary_b) if parameter == 'independent' else (b_from_c(charges[parameter]),)
        for b in parameters:
            table = _build_rs_data(32, b, external)
            residues, weights = next(iter(table.kac_balls.values()))
            previous = ctx.prec
            try:
                ctx.prec = 1000
                # K_11 = 8(h2-h1)(h3-h4), independently of b.
                assert residues[0].overlaps(acb(arb(-2)/25))
                if parameter == 'independent':
                    index = next(i for i, p in enumerate(table.pairs) if (p['r'], p['s']) == (2, 1))
                    br, bi = literal_components(b)
                    expected = -(3*(br*br-bi*bi)+2)/4
                    assert weights[index].overlaps(acb(arb(expected.numerator)/expected.denominator))
            finally:
                ctx.prec = previous
        if parameter == 'imaginary':
            # At c=0, b²=-2/3 and d_21=0 exactly. Its algebraic
            # enclosure must be pinned before a relative-accuracy check.
            table = _build_rs_data(2, b_from_c(0), external)
            index = next(i for i, p in enumerate(table.pairs) if (p['r'], p['s']) == (2, 1))
            assert next(iter(table.kac_balls.values()))[1][index].is_zero()


def test_sphere_public_paths_keep_rational_labels_in_their_kac_bounds(monkeypatch):
    from collections import OrderedDict

    from flint import acb, arb, ctx
    from virasoro_blocks import recursion
    from virasoro_blocks.derivs import chiral_z_series_evaluator
    from virasoro_blocks.polyblock import H_q_series_rational, chiral_denominator_roots

    external = tuple(Fraction(i, 10) for i in range(1, 5))
    original = recursion._kac_balls
    calls = []

    def check(*args, **kwargs):
        residues, weights = original(*args, **kwargs)
        previous = ctx.prec
        try:
            ctx.prec = 1000
            assert residues[0].overlaps(acb(arb(-2)/25))
            calls.append(1)
        finally:
            ctx.prec = previous
        return residues, weights

    monkeypatch.setattr(recursion, '_kac_balls', check)
    monkeypatch.setattr(recursion, '_PAIR_TABLE_CACHE', OrderedDict())
    monkeypatch.setattr(recursion, '_H_CONTEXT_CACHE', OrderedDict())
    with mp.workdps(30):
        # Different charges exercise each entry point without a preceding
        # construction supplying a shared cached context.
        paths = (
            lambda: recursion.H_q_series('2.25', external, c='32.7', q_order=32),
            lambda: recursion.F_block('2.25', '.5', external, c='32.8', q_order=32),
            lambda: chiral_z_series_evaluator(external, 2, c='32.9', q_order=32)('2.25'),
            lambda: H_q_series_rational(external, c='33.1', q_order=32),
            lambda: chiral_denominator_roots(external, c='33.2', q_order=32),
        )
        for path in paths:
            before = len(calls)
            path()
            assert len(calls) > before


def test_sphere_coefficients_resolve_a_finite_shift_from_an_exact_kac_weight():
    from virasoro_blocks.recursion import H_q_series

    pole = -Fraction(23, 80)  # d_22 at c=33/10
    shifted = pole+Fraction(1, 10**70)
    external = tuple(Fraction(i, 10) for i in range(1, 5))
    with mp.workdps(220):
        stored = mp.convert(shifted)
    labels = (shifted, f'{shifted.numerator}/{shifted.denominator}', stored)
    with mp.workdps(240):
        references = [H_q_series(h, external, c='3.3', q_order=4) for h in labels]
    with mp.workdps(30):
        for label, reference in zip(labels, references):
            actual = H_q_series(label, external, c='3.3', q_order=4)
            assert _worst_digits(actual, reference) > 28
        with pytest.raises(ZeroDivisionError, match='Kac pole'):
            H_q_series(pole, external, c='3.3', q_order=4)
        assert mp.mp.dps == 30

    # Moving c also moves its pole: d_22 = -(c-1)/8. A
    # rounded midpoint equal to -1/4 must not erase that finite separation.
    shifted_c = Fraction(3)+Fraction(1, 10**70)
    labels = (Fraction(-1, 4), '-.25', -.25, mp.mpf('-.25'))
    with mp.workdps(160):
        reference = H_q_series(Fraction(-1, 4), external, c=shifted_c, q_order=4)
    with mp.workdps(30):
        for label in labels:
            actual = H_q_series(label, external, c=shifted_c, q_order=4)
            assert _worst_digits(actual, reference) > 28
        with pytest.raises(ZeroDivisionError, match='Kac pole'):
            H_q_series(Fraction(-1, 4), external, c=3, q_order=4)


def test_sphere_near_singular_charge_keeps_its_source_during_precision_escalation():
    from virasoro_blocks.params import b_from_c
    from virasoro_blocks.recursion import H_q_series, _build_rs_data

    external = ('.2', '.3', '.4', '.5')
    for center in (Fraction(1), Fraction(25), Fraction(7, 10)):
        charge = center+Fraction(1, 10**70)
        with mp.workdps(200):
            reference = H_q_series('.71', external, c=charge, q_order=25)
        with mp.workdps(30):
            actual = H_q_series('.71', external, c=charge, q_order=25)
            assert _worst_digits(actual, reference) > 28
            assert _build_rs_data(25, b_from_c(charge), external).kac_from_c


def test_torus_coefficients_rebuild_a_kernel_when_evaluation_precision_increases():
    from virasoro_blocks.torus_onept import build_torus_onept_kernel, torus_onept_f_coeffs

    with mp.workdps(30):
        kernel = build_torus_onept_kernel(6, c='32.7', d_ext='.7')
    with mp.workdps(120):
        reference = torus_onept_f_coeffs(build_torus_onept_kernel(6, c='32.7', d_ext='.7'), '.71')
    with mp.workdps(100):
        actual = torus_onept_f_coeffs(kernel, '.71')
        assert _worst_digits(actual[1:], reference[1:]) > 97
        assert torus_onept_f_coeffs(kernel, '.71') == actual


def test_near_singular_c_tables_derive_b_from_c():
    """c = 1 + 2^-40: the derivative tables formed b at 2 dps while the near-singular guard ran at dps + ~126
    (34 of 60 digits); an evaluator handed b = b_from_c(c) at the working precision had none (-25)."""
    from virasoro_blocks.derivs import G_zzb_table_many, chiral_z_series_evaluator
    from virasoro_blocks.params import b_from_c

    ext = [mp.mpf(e) for e in DIFF_EXT]

    def table(dps):
        with mp.workdps(dps):
            return G_zzb_table_many([mp.mpf("2.25")], 0, ext, 5, c=C_NEAR_ONE, q_order=40)[0]

    low, reference = table(60), table(140)
    with mp.workdps(150):
        assert _table_digits(low, reference) > 56

    with mp.workdps(60):
        via_b = chiral_z_series_evaluator(ext, 8, b=b_from_c(C_NEAR_ONE), q_order=24)(mp.mpf("1.125"))
    with mp.workdps(140):
        exact = chiral_z_series_evaluator(ext, 8, c=C_NEAR_ONE, q_order=24)(mp.mpf("1.125"))
    with mp.workdps(150):
        scale = max(abs(x) for x in exact)
        assert min(-mp.log10(abs(x - y) / scale) for x, y in zip(via_b, exact) if x != y) > 56


def test_F_block_near_singular_guard_does_not_depend_on_evaluation_order():
    """The verified guard was cached without h: at c = 1 + 2^-40, h = 1.25 + 2i kept 51.6 of 60 digits after
    h = 2^-27 had set the guard, 61.9 fresh."""
    import virasoro_blocks.recursion as R
    import virasoro_blocks.singularity as S

    ext = [mp.mpf(e) for e in DIFF_EXT]
    z = mp.mpf("0.9296875")
    h = mp.mpc("1.25", "2")

    def fresh(first=None):
        S._NEAR_GUARD_CACHE.clear()
        R._H_CONTEXT_CACHE.clear()
        R._PAIR_TABLE_CACHE.clear()
        with mp.workdps(60):
            if first is not None:
                R.F_block(first, z, ext, c=C_NEAR_ONE, q_order=40)
            return R.F_block(h, z, ext, c=C_NEAR_ONE, q_order=40)

    alone = fresh()
    after = fresh(mp.mpf(2) ** -27)
    with mp.workdps(60):
        reference = R.F_block(h, z, ext, c=C_NEAR_ONE, q_order=40)
    with mp.workdps(120):
        assert _digits(after, alone) > 58
        assert _digits(alone, reference) > 58


def test_c_below_one_kac_data_are_exact_at_d11():
    """For c < 1 d_{1,1} came out -3.9e-32 (difference of squares) instead of 0, and an exchanged h near 0 lost
    log10(1/h) digits against it (h = 2^-40: 19 of 30)."""
    from virasoro_blocks.params import b_from_c
    from virasoro_blocks.recursion import F_block, _build_rs_data, _maybe_real_scalar

    c = mp.mpf("0.712890625")
    ext = [mp.mpf(e) for e in DIFF_EXT]
    with mp.workdps(30):
        table = _build_rs_data(10, _maybe_real_scalar(b_from_c(c)), tuple(ext))
        assert table.mode == "imag_b"
        assert table.pairs[0]["d"] == 0          # (r, s) = (1, 1)

    def value(dps):
        with mp.workdps(dps):
            return F_block(mp.mpf(2) ** -40, mp.mpf("0.5"), ext, c=c, q_order=10)

    low, reference = value(30), value(80)
    with mp.workdps(90):
        assert _digits(low, reference) > 28


@pytest.mark.parametrize("builder", ["G_xt_table_many", "G_ws_table_many"])
def test_folded_coordinates_keep_their_digits_at_large_delta(builder):
    """The parity fold cancels against the chiral growth: xt / ws at Delta = 300 (c = 8, h_ext = 1/16, Lambda 21,
    dps 60) kept 38.8 / 39.0 digits on the Lambda triangle, while zzb / yyb kept 58."""
    import virasoro_blocks.derivs as D

    build = getattr(D, builder)
    ext = [mp.mpf("0.0625")] * 4

    def table(dps):
        with mp.workdps(dps):
            return build([mp.mpf(300)], 0, ext, 21, c=mp.mpf(8), q_order=40)[0]

    low, reference = table(60), table(130)
    triangle = [(m, n) for m in range(22) for n in range(22) if m + n <= 21]
    with mp.workdps(140):
        scale = max(abs(reference[m][n]) for m, n in triangle)
        error = max(abs(low[m][n] - reference[m][n]) for m, n in triangle)
        assert -mp.log10(error / scale) > 56


def test_escalated_h_series_keeps_a_real_series_real():
    """Precision escalation preserves real H coefficients at c=3/2."""
    import mpmath as mp

    from virasoro_blocks.recursion import H_q_series

    with mp.workdps(30):
        H = H_q_series(mp.mpf(37) / 18, [mp.mpf(1) / 16] * 4, c=mp.mpf(3) / 2, q_order=50)
    assert all(isinstance(value, mp.mpf) for value in H)


def test_regularizer_probe_next_to_c_one_is_certified():
    """A regularizer's probe at c = 1 + 1e-15 (under bare_recursion_probes) kept the first, uncertified fill: its
    midpoints were noise (the chiral coefficients came back as 0.385 / 0.375 / 2.0 against 0.503 / 0.463 / 2.453), and
    the c = 1 crossing check of the twist/vertex mixtures failed (crossing 0.46)."""
    from virasoro_blocks.derivs import chiral_z_series_evaluator
    from virasoro_blocks.singularity import bare_recursion_probes

    ext = [mp.mpf(9) / 16] * 4

    def coefficients(dps):
        with mp.workdps(dps), bare_recursion_probes():
            c = 1 + mp.mpf(10) ** -15
            evaluate = chiral_z_series_evaluator(ext, 4, c=c, q_order=50, z_shift=-mp.mpf(9) / 8)
            return evaluate(mp.mpf("0.5625")), evaluate.many([mp.mpf("0.5625")])[0]

    (low, low_many), (reference, _many) = coefficients(45), coefficients(160)
    with mp.workdps(170):
        scale = max(abs(x) for x in reference)
        for values in (low, [_arb_mid(x) for x in low_many]):
            assert max(abs(mp.re(x) - mp.re(y)) for x, y in zip(values, reference)) < mp.mpf(10) ** -40 * scale


def _arb_mid(value):
    from virasoro_blocks.polynomial import _arb_to_mpf

    return _arb_to_mpf(value.mid() if not hasattr(value, "imag") else value.real.mid())


# (c, h, h_ext, the coefficient that vanishes exactly): Vir(4) blocks of Ising^8 sigma x 1 (Delta 1/8, 3/8) and of
# the U(1)^4 vertex points (Delta 0.3, 0.5), and one at c = 5 found by a scan
EXACT_ZERO_CASES = [
    (4, 0, Fraction(1, 16), 2),
    (4, 0, Fraction(3, 16), 2),
    (4, 2, Fraction(3, 20), 2),
    (4, 2, Fraction(1, 4), 2),
    (5, Fraction(1, 2), Fraction(1, 4), 2),
]


@pytest.mark.parametrize("c, h, h_ext, k_zero", EXACT_ZERO_CASES)
def test_H_q_series_certifies_an_exactly_vanishing_coefficient(c, h, h_ext, k_zero):
    """A coefficient the exact inputs make vanish (H_2 = 0 here) is a ball around 0 with no relative bit, so the
    per-coefficient relative test escalated to its cap and raised RecursionPrecisionError (2026-10-08; the heavy-
    external script averaged blocks at c -+ 1e-12 instead).  It is certified against the series' scale and comes
    back as exactly 0; the series agrees with a 2x-precision build and with the c-limit at fixed h."""
    from virasoro_blocks.recursion import F_block, H_q_series

    ext = [h_ext] * 4
    with mp.workdps(30):
        series = H_q_series(h, ext, c=c, q_order=24)
        delta = Fraction(1, 10 ** 12)
        lo = H_q_series(h, ext, c=Fraction(c) - delta, q_order=24)
        hi = H_q_series(h, ext, c=Fraction(c) + delta, q_order=24)
        block = F_block(h, mp.mpf("0.4"), ext, c=c, q_order=24)
    with mp.workdps(60):
        reference = H_q_series(h, ext, c=c, q_order=24)
    assert series[k_zero] == 0
    assert any(value != 0 for value in series[k_zero + 1:])
    with mp.workdps(70):
        assert max(abs(x - y) for x, y in zip(series, reference)) < mp.mpf(10) ** -29
        # the blocks are analytic in c at fixed h: the two-sided average is the c-limit to O(delta^2)
        assert max(abs(x - (a + b) / 2) for x, a, b in zip(series, lo, hi)) < mp.mpf(10) ** -20
        assert mp.isfinite(block) and block != 0


def test_input_bits_measure_the_exact_inputs():
    from virasoro_blocks.recursion import _input_bits

    assert _input_bits((Fraction(1, 16), 4, 0)) == 1 + 1      # 1/16: one mantissa bit; 4: one (dyadic)
    assert _input_bits((Fraction(3, 20),)) == 2 + 5
    with mp.workdps(250):
        binary = mp.mpf("0.1")
    assert 820 < _input_bits((binary,)) <= 840                 # its odd mantissa (250 digits ~ 834 bits)


def test_power_series_deficit_holds_coefficients_to_the_running_scale():
    """The series rule: relative bits for a coefficient as large as everything before it, the scale
    ``max_{j <= k} |a_j|`` (less the inputs' own resolution) for a smaller one (a ball around 0 included), never a
    later, larger coefficient."""
    from flint import arb, ctx as flint_ctx

    from virasoro_blocks.recursion import _power_series_deficit, _zero_centred

    saved = flint_ctx.prec
    flint_ctx.prec = 200
    try:
        work = 100
        tiny = arb(0, arb(2) ** -110)
        assert _power_series_deficit([arb(1), tiny, arb(3)], work) == (0, None)
        # a tiny coefficient is still certified, by the scale: 2^-80 against 1 is 20 bits short
        short, where = _power_series_deficit([arb(1), arb("1e-40", arb(2) ** -80), arb(3)], work)
        assert where == 1 and 19 <= short <= 21
        # a later, larger coefficient does not relax it
        short_big, _ = _power_series_deficit([arb(1), arb(0, arb(2) ** -80), arb(2) ** 60], work)
        assert 19 <= short_big <= 21
        # an ordinary coefficient keeps the relative test
        short_rel, where_rel = _power_series_deficit([arb(1), arb(5, arb(2) ** -95)], work)
        assert where_rel == 1 and short_rel > 0
        # no scale at all: nothing certified
        assert _power_series_deficit([arb(0, 1)], work)[0] == 2 * work
        # inputs carrying more bits resolve a coefficient they make small: the scale target moves down by them
        short_in, where_in = _power_series_deficit([arb(1), tiny, arb(3)], work, 30)
        assert where_in == 1 and 19 <= short_in <= 21
        centred = _zero_centred([arb(1), arb(arb(2) ** -120, arb(2) ** -110)])
        assert centred[0] == arb(1) and centred[1].mid() == 0 and centred[1].contains(arb(2) ** -120)
    finally:
        flint_ctx.prec = saved
