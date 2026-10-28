"""Cross-check public entry points, metadata, validation, and block accuracy."""

import mpmath as mp
import pytest

from virasoro_blocks.recursion import F_block


@pytest.mark.parametrize("c0, eps, dps", [("25", "1e-12", 60), ("1", "1e-12", 30), ("-2", "1e-8", 30)])
def test_near_singular_c_is_verified_at_any_precision(c0, eps, dps):
    """Near (not on) a locus the bare recursion lost up to 43 of 60 digits (25 + 1e-12) with no guard: the
    verification now triggers below a fixed Kac-pole spacing, whatever the working precision."""
    h = [mp.mpf("0.3")] * 4
    with mp.workdps(dps):
        c = mp.mpf(c0) + mp.mpf(eps)
        value = F_block(mp.mpf("0.8"), mp.mpf("0.4"), h, c=c, q_order=30)
    with mp.workdps(300):
        ref = F_block(mp.mpf("0.8"), mp.mpf("0.4"), h, c=mp.mpf(c0) + mp.mpf(eps), q_order=30)
        assert abs(mp.mpc(value) - ref) <= mp.mpf(10) ** -(dps - 3) * abs(ref)


def test_non_finite_values_are_refused_not_zeroed():
    """mpmath's nan/inf became an exact 0 in the flint conversion (a nan exchanged Delta gave the Delta = 0 table)."""
    from virasoro_blocks.polynomial import Polynomial

    with pytest.raises(ValueError, match="non-finite"):
        Polynomial([mp.nan, 1, 2]).evaluate(2)
    with pytest.raises(ValueError, match="non-finite"):
        Polynomial([1, 1]).evaluate(mp.inf)


@pytest.mark.parametrize("c", ["1e10", "1e20", "1e40", "-1e20"])
def test_b_from_c_is_stable_at_large_c(c):
    """b^2 = (w - sqrt(w^2 - 4))/2 canceled at large |c| (every digit of b lost at c = 1e20, dps 30)."""
    from virasoro_blocks.params import b_from_c, c_from_b

    with mp.workdps(30):
        b = b_from_c(mp.mpf(c))
        assert abs(c_from_b(b) - mp.mpf(c)) <= mp.mpf(10) ** -28 * abs(mp.mpf(c))


def test_cli_reports_user_errors_in_one_line(capsys):
    from virasoro_blocks.__main__ import main

    assert main(["table", "--central-charge", "abc", "--lambda", "2"]) == 2
    assert "'abc' is not a real number" in capsys.readouterr().err
    assert main(["table", "--central-charge", "nan", "--lambda", "2"]) == 2
    assert "not a finite number" in capsys.readouterr().err
    assert main(["tabel", "--lambda", "2"]) == 2
    assert "did you mean 'table'" in capsys.readouterr().err
    assert main(["--version"]) == 0
    assert "virasoro_blocks" in capsys.readouterr().out
    assert main([]) == 0
    assert "operation" in capsys.readouterr().out


def test_near_locus_hint_ignores_astronomical_minimal_models():
    """Near c = 1 every c is close to some M(p, q) with p ~ 1e13; the hint named one.  It now names c = 1."""
    from virasoro_blocks.singularity import nearby_singular_locus

    with mp.workdps(90):
        label, _distance = nearby_singular_locus(mp.mpf("0.9999999999999999999"))
    assert label == "c = 1"


@pytest.mark.parametrize("ell", [1, 2])
def test_off_diagonal_scalar_G_table_is_complete_and_right(tmp_path, ell):
    """Off-diagonal scalar tables retain both orientations and correct odd-spin degrees."""
    import json

    from virasoro_blocks.__main__ import main
    from virasoro_blocks.api import BlockRequest, NumericalPolicy, build_table_request
    from virasoro_blocks.serialization import serialized_table_from_json

    main(["table", "--central-charge", "8", "--delta-ext", "0.26", "0.26", "0.26", "0.26", "--lambda", "3",
          "--order", "16", "--spin-ranges", str(ell), "--npoles", "16", "--precision", "40",
          "--z-center", "0.4", "--zbar-center", "0.45", "-o", str(tmp_path)])
    with mp.workdps(40):
        table = serialized_table_from_json(json.load(open(tmp_path / f"derivs_{ell}.json")))
        delta = mp.mpf(table.shift) + mp.mpf("0.77")
        got = table.evaluate("zzb", delta)
        ref = build_table_request(BlockRequest(
            operation="point-table", family="virasoro", basis="scalar_G", central_charge="8",
            delta_ext=("0.26",) * 4, coordinates=("zzb",), exchange_delta=delta, exchange_spin=ell,
            center=mp.mpf("0.4"), center_bar=mp.mpf("0.45"),
            policy=NumericalPolicy(precision=60, order=16, lambda_order=3))).evaluate("zzb", delta)
        for m in range(4):
            for n in range(4 - m):
                assert abs(mp.mpc(got[m][n]) - mp.mpc(ref[m][n])) <= mp.mpf(10) ** -12 * max(1, abs(mp.mpc(ref[m][n])))


@pytest.mark.parametrize("family", [["--family", "global", "--basis", "global_parity_projected_B"],
                                    ["--central-charge", "5", "--order", "16", "--basis", "parity_projected_B"]])
def test_folded_parity_projected_table_uses_sign_times_phase(tmp_path, family):
    """B_+- = B +- eta P[B] has z <-> zbar parity sigma = (+-1) eta, so '+' with eta = -1 is B - P[B], the same block
    as '-' with eta = +1.  The fold used the projection sign alone: the '+' / eta = -1 folded table was all zeros."""
    import json

    from virasoro_blocks.__main__ import main
    from virasoro_blocks.serialization import serialized_table_from_json

    common = [*family, "--spin-ext", "1", "0", "0", "1", "--delta-ext", "0.9", "0.7", "1.1", "1.45",
              "--coordinates", "xt", "--lambda", "3", "--precision", "30", "--npoles", "6", "--spin-ranges", "1"]
    main(["table", *common, "--parity-project", "+", "--parity-ext", "1", "1", "1", "-1", "-o", str(tmp_path / "a")])
    main(["table", *common, "--parity-project", "-", "--parity-ext", "1", "1", "1", "1", "-o", str(tmp_path / "b")])
    with mp.workdps(40):
        a = serialized_table_from_json(json.load(open(tmp_path / "a" / "derivs_1.json")))
        b = serialized_table_from_json(json.load(open(tmp_path / "b" / "derivs_1.json")))
        delta = mp.mpf(a.shift) + mp.mpf("1.3")
        A, B = a.evaluate("xt", delta), b.evaluate("xt", delta)
        assert max(abs(mp.mpc(v)) for row in A for v in row) > mp.mpf("1e-3")
        assert all(abs(mp.mpc(x) - mp.mpc(y)) <= mp.mpf(10) ** -25 for ra, rb in zip(A, B) for x, y in zip(ra, rb))


def test_large_spin_tower_with_several_coordinates(tmp_path):
    """Each coordinate chose its own tower poles, so the tables could not be serialized together (ValueError); the
    first coordinate's poles are now shared."""
    import json

    from virasoro_blocks.__main__ import main
    from virasoro_blocks.serialization import serialized_table_from_json

    main(["table", "--central-charge", "8", "--delta-ext", "0.26", "0.26", "0.26", "0.26", "--lambda", "3",
          "--order", "24", "--large-spin-lmin", "12", "--large-spin-twists", "1.7", "--coordinates", "zzb", "ws",
          "--npoles", "8", "--precision", "30", "-o", str(tmp_path)])
    table = serialized_table_from_json(json.load(open(tmp_path / "derivs_large_spin_0.json")))
    assert set(table.coordinates) == {"zzb", "ws"}


@pytest.mark.parametrize("family, fn, c", [("ns", "ns_chiral_block", "1.50000001"),
                                           ("ramond", "ramond_chiral_block", "1.50000001")])
def test_super_bare_recursion_is_guarded_near_singular_c(family, fn, c):
    """Just outside the regularization window the bare NS/Ramond recursion had no guard (1.5 + 1e-8 at 30 digits
    gave -6165.68 for 0.0626)."""
    from virasoro_blocks import superconformal_blocks as sb

    h = [mp.mpf(x) for x in ("0.2", "0.25", "0.15", "0.3")]
    with mp.workdps(30):
        value = getattr(sb, fn)(mp.mpf("0.8"), mp.mpf("0.4"), h, c=mp.mpf(c), q_order=8)
    with mp.workdps(80):
        ref = getattr(sb, fn)(mp.mpf("0.8"), mp.mpf("0.4"), h, c=mp.mpf(c), q_order=8)
        assert abs(mp.mpc(value) - mp.mpc(ref)) <= mp.mpf(10) ** -27 * abs(mp.mpc(ref))


def test_global_half_integer_exchanged_spin_builds(tmp_path):
    """The shift defaulted to int(ell): at ell = 3/2 the pole at Delta = 3/2 landed at x = +1/2 and the build failed."""
    from virasoro_blocks.__main__ import main

    assert main(["table", "--family", "global", "--basis", "global_ordered_B", "--lambda", "2", "--delta-ext", "0.9",
                 "0.7", "1.1", "1.45", "--spin-ext", "0.5", "0", "0", "0.5", "--spin-ranges=3/2", "--precision", "30",
                 "-o", str(tmp_path)]) in (0, None)
    assert (tmp_path / "derivs_1.5.json").exists()


def _evaluate(**kw):
    from virasoro_blocks.api import BlockRequest, NumericalPolicy, evaluate_block_request

    base = dict(operation="evaluate", central_charge="8", delta_ext=("1/10",) * 4, exchange_delta="2",
                exchange_spin=2, point={"z": "0.4", "zbar": "0.3"}, policy=NumericalPolicy(precision=30, order=16))
    base.update(kw)
    return evaluate_block_request(BlockRequest(**base)).value


@pytest.mark.parametrize("family", ["virasoro", "global", "ns"])
def test_evaluate_scalar_G_shift_is_symmetric(family):
    """A lone z_shift shifted only B's z (a value no table has); scalar_G carries one shift, (z zbar)^S."""
    from virasoro_blocks.requests import RequestValidationError

    kw = {"family": family} | ({"central_charge": None} if family == "global" else {})
    with mp.workdps(30):
        plain = mp.mpc(_evaluate(**kw))
        shifted = mp.mpc(_evaluate(z_shift="-1/4", **kw))
        assert abs(shifted - plain * mp.mpf("0.12") ** mp.mpf("-0.25")) <= mp.mpf(10) ** -25 * abs(plain)
    with pytest.raises(RequestValidationError, match="one shift"):
        _evaluate(z_shift="-1/4", zbar_shift="0", **kw)


@pytest.mark.parametrize("family", ["virasoro", "global", "ramond"])
def test_evaluate_on_the_real_cut_is_the_euclidean_limit(family):
    """At z = zbar = x > 1 both halves took the same principal branch: a complex value where the block is real."""
    kw = {"family": family} | ({"central_charge": None} if family == "global" else {})
    with mp.workdps(30):
        on_cut = mp.mpc(_evaluate(point={"z": "1.4", "zbar": "1.4"}, **kw))
        eps = mp.mpf("1e-40")
        limit = mp.mpc(_evaluate(point={"z": mp.mpc("1.4", eps), "zbar": mp.mpc("1.4", -eps)}, **kw))
        assert abs(on_cut.imag) <= mp.mpf(10) ** -25 * abs(on_cut)
        assert abs(on_cut - limit) <= mp.mpf(10) ** -25 * abs(limit)


@pytest.mark.parametrize("z", ["0.99", "0.99999", "-5", "0.5+0.9j", "1.5-0.2j"])
def test_global_block_is_accurate_near_and_beyond_z_equals_one(z):
    """flint's 2F1 series was read by its midpoint: 30 digits became 6 at z = 0.99 and |z| >= 1 raised."""
    from virasoro_blocks.global_blocks import global_evaluate_block

    with mp.workdps(30):
        h = [mp.mpf("0.1"), mp.mpf("0.3"), mp.mpf("0.2"), mp.mpf("0.1")]
        H, zz = mp.mpf("1.3"), mp.mpmathify(z)
        k = zz ** H * mp.hyp2f1(H + h[1] - h[0], H + h[2] - h[3], 2 * H, zz) / mp.mpf(16) ** H
        got = global_evaluate_block(zz, zz, H, H, h, coord="zzb")
        assert abs(got - k * k) <= mp.mpf(10) ** -27 * abs(k * k)


@pytest.mark.parametrize("extra, match", [
    ({"parity_exchanged": "-1"}, "parity_exchanged"),
    ({"sector": "odd"}, "sector"),
    ({"branch": "+-"}, "branch"),
    ({"torus_onepoint_ext": "0.3"}, "torus_onepoint_ext"),
    ({"point": {"z": "0.4", "zbar": "0.3", "y": "0.2"}}, "point name"),
])
def test_evaluate_refuses_inputs_it_would_ignore(extra, match):
    from virasoro_blocks.requests import RequestValidationError

    with pytest.raises(RequestValidationError, match=match):
        _evaluate(**extra)


def test_request_strings_are_numbers_at_the_request_precision():
    """String centers reached the backends as str (TypeError); every numeric field now takes '0.4' or '7/10'."""
    from virasoro_blocks.api import BlockRequest, NumericalPolicy

    req = BlockRequest(operation="point-table", center="0.4", central_charge="7/10",
                       policy=NumericalPolicy(precision=50))
    from fractions import Fraction

    assert req.center == Fraction(2, 5)
    assert req.central_charge == Fraction(7, 10)
    with mp.workdps(50):
        assert abs(req.central_charge - mp.mpf(7) / 10) <= mp.mpf(10) ** -49


def test_cli_flags_are_not_abbreviated(capsys):
    """argparse took '--delta 1.3 --spin 0' as a prefix of --delta-ext and consumed the next four tokens."""
    from virasoro_blocks.cli.evaluate import main as evaluate_main

    with pytest.raises(SystemExit):
        evaluate_main(["--central-charge", "8", "--delta-ext", "0.1", "0.1", "0.1", "0.1", "--exchange-delta", "2",
                       "--point", "z=0.4", "--point", "zbar=0.3", "--prec", "20"])
    assert "unrecognized arguments: --prec" in capsys.readouterr().err


def test_evaluate_measures_the_q_truncation():
    """c = 1e5 at order 30 returned 2e-20 for a block of ~2e-3 at every precision, with no warning: the order needed
    grows with c. The measured accuracy is now recorded, and strict mode refuses."""
    from virasoro_blocks.api import BlockRequest, NumericalPolicy, evaluate_block_request
    from virasoro_blocks.qgate import QConvergenceError

    def request(c):
        return BlockRequest(operation="evaluate", central_charge=c, delta_ext=("0.2",) * 4, exchange_delta="2",
                            point={"z": "0.4", "zbar": "0.4"}, policy=NumericalPolicy(precision=30, order=30))

    assert evaluate_block_request(request("8")).metadata["q_truncation_digits"] >= 29
    with pytest.raises(QConvergenceError, match="NOT entered"):
        evaluate_block_request(request("100000"), q_gate="strict")
    assert evaluate_block_request(request("100000"), q_gate="off").metadata["q_truncation_digits"] == 0


@pytest.mark.parametrize("extra, match", [
    ({"large_spin_twists": (1.5,)}, "needs large_spin_lmin"),
    ({"positive_pole_selection": "rightmots"}, "positive_pole_selection"),
    ({"delta_min_by_spin": {0: 1.2}}, "conflicts with delta_gap"),
])
def test_gap_spec_refuses_fields_it_would_ignore(extra, match):
    """Each of these was accepted and had no effect (or silently replaced delta_gap)."""
    from virasoro_bootstrap.identical_scalar import IdenticalScalarGapSpec, prepare_identical_scalar_gap_build

    spec = IdenticalScalarGapSpec(c=8, h_ext=mp.mpf(1) / 16, delta_gap=1, lam=3, npoles=4, q_order=10, dps=40,
                                  num_threads=1, **extra)
    with pytest.raises(ValueError, match=match):
        prepare_identical_scalar_gap_build(spec)


def test_q_gate_probes_the_table_center():
    """The gate always probed z0 = 1/2; an off-symmetric table converges more slowly in q."""
    from virasoro_blocks.qgate import q_tail_report

    with mp.workdps(40):
        kw = dict(c=mp.mpf(8), h_externals=[mp.mpf("0.1")] * 4, q_order=20, n_terms=5)
        sym = q_tail_report(**kw)
        off = q_tail_report(**kw, center=mp.mpf("0.8"))
        both = q_tail_report(**kw, center=[mp.mpf("0.4"), mp.mpf("0.8")])
    assert off["digits"] < sym["digits"] - 3
    assert both["digits"] == off["digits"]



@pytest.mark.parametrize("extra, message", [
    (["--spin-ranges", "4:2"], "empty spin range"),
    (["--spin-ranges", "4-2"], "invalid spin '4-2'"),
    (["--spin-ranges", "0", "--z-center", "1.5"], "--z-center must be a cross ratio in (0, 1)"),
    (["--large-spin-lmin", "-3", "--large-spin-twists", "1", "--npoles", "4"], "--large-spin-lmin must be >= 0"),
    (["--spin-ranges", "0", "-o", "/dev/null/sub"], "virasoro_blocks: error:"),
])
def test_table_cli_input_errors_are_one_line(capsys, extra, message):
    from virasoro_blocks.__main__ import main

    import contextlib

    with pytest.raises(SystemExit) if "error:" not in message else contextlib.nullcontext():
        assert main(["table", "--central-charge", "8", "--delta-ext", "0.2", "0.2", "0.2", "0.2", "--lambda", "2",
                     "--order", "12", "--precision", "30", *extra]) == 2
    err = capsys.readouterr().err
    assert message in err and "Traceback" not in err


def test_negative_fractions_parse_as_values(tmp_path):
    """'--u-shift -1/5' failed with 'expected one argument' (argparse's negative-number test is -2 / -0.5 only)."""
    import json

    from virasoro_blocks.__main__ import main

    main(["table", "--central-charge", "8", "--delta-ext", "0.2", "0.2", "0.2", "0.2", "--lambda", "2", "--order",
          "12", "--precision", "30", "--spin-ranges", "0", "--npoles", "4", "--u-shift", "-1/5", "-o", str(tmp_path)])
    assert mp.almosteq(mp.mpf(json.load(open(tmp_path / "derivs_0.json"))["z_shift"]), mp.mpf("-0.2"))


def test_F_block_takes_fractions():
    """Exact Fraction externals raised TypeError in the pole-root builder (mpc(Fraction))."""
    from fractions import Fraction

    with mp.workdps(30):
        exact = F_block(Fraction(0), mp.mpf("0.4"), [Fraction(1, 10)] * 4, c=8, q_order=10)
        assert abs(exact - F_block(0, mp.mpf("0.4"), [mp.mpf(1) / 10] * 4, c=8, q_order=10)) < mp.mpf(10) ** -28


@pytest.mark.parametrize("extra, message", [
    (["--family", "ns", "--central-charge", "1.3", "--sample-via", "symbolic"], "--sample-via is not implemented"),
    (["--family", "ns", "--central-charge", "1.3", "--q-gate", "strict"], "q-order gate [ns tables]"),
    (["--central-charge", "8", "--torus-onept-ext", "0.3"], "--surface torus only"),
    (["--central-charge", "8", "--large-spin-twists", "1"], "needs --large-spin-lmin"),
])
def test_sphere_table_refuses_flags_it_would_ignore(capsys, extra, message):
    """(--q-gate on NS/Ramond is honored now: strict refuses an order-12 table short of 20 digits.)"""
    from virasoro_blocks.__main__ import main

    with pytest.raises(SystemExit):
        main(["table", *extra, "--delta-ext", "0.2", "0.2", "0.2", "0.2", "--lambda", "2", "--order", "12",
              "--precision", "30", "--spin-ranges", "0", "--npoles", "4"])
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("operation", ["table", "point-table"])
def test_api_scalar_G_refuses_half_integer_spin(operation):
    """The CLI and evaluate refused it; build_table_request built a table with (-1)^(3/2)."""
    from virasoro_blocks.api import BlockRequest, NumericalPolicy, build_table_request
    from virasoro_blocks.requests import RequestValidationError

    kw = dict(operation=operation, central_charge="8", delta_ext=("0.2",) * 4, coordinates=("zzb",),
              policy=NumericalPolicy(precision=30, order=12, lambda_order=2, npoles=4))
    with pytest.raises(RequestValidationError, match="integer exchanged spin"):
        if operation == "table":
            build_table_request(BlockRequest(**kw), spin=mp.mpf(3) / 2)
        else:
            build_table_request(BlockRequest(exchange_delta="2", exchange_spin=mp.mpf(3) / 2, **kw))


def test_block_reader_refuses_malformed_files(tmp_path):
    """Missing keys were KeyErrors, a truncated grid an IndexError, and node values that no longer matched their
    nodes were accepted."""
    import copy
    import json

    from virasoro_blocks.__main__ import main
    from virasoro_blocks.serialization import serialized_table_from_json

    main(["table", "--central-charge", "8", "--delta-ext", "0.2", "0.2", "0.2", "0.2", "--lambda", "3", "--order",
          "12", "--spin-ranges", "0", "--npoles", "6", "--precision", "30", "-o", str(tmp_path)])
    doc = json.load(open(tmp_path / "derivs_0.json"))
    edits = {
        "missing 'derivs'": lambda d: d.pop("derivs"),
        "rows of lengths": lambda d: d["derivs"][0].pop(),
        "node values for": lambda d: d["sample_points_x"].pop(),
        "coordinate blocks": lambda d: d.__setitem__("coordinates", ["zzb", "yyb"]),
    }
    for message, edit in edits.items():
        broken = copy.deepcopy(doc)
        edit(broken)
        with pytest.raises(ValueError, match=message):
            serialized_table_from_json(broken)
    serialized_table_from_json(doc)


def test_evaluate_cli_takes_complex_points(capsys):
    """'--point z=0.4+0.3j' was 'invalid numeric value'; a conjugate pair is the Euclidean section (a real value)."""
    from virasoro_blocks.cli.evaluate import main as evaluate_main

    common = ["--central-charge", "8", "--delta-ext", "0.2", "0.2", "0.2", "0.2", "--exchange-delta", "2",
              "--exchange-spin", "2", "--format", "text", "--precision", "25"]
    assert evaluate_main([*common, "--point", "z=0.4+0.3j", "--point", "zbar=0.4-0.3j"]) == 0
    assert "j" not in capsys.readouterr().out
    assert evaluate_main([*common, "--point", "z=0.4-0.3j", "--point", "zbar=0.4-0.3j"]) == 0
    assert " - " in capsys.readouterr().out


@pytest.mark.parametrize("family", ["ns", "ramond"])
def test_super_evaluate_measures_the_q_truncation(family):
    """NS at c = 3e4, order 20 gave 0.0034 for 0.0319 with no warning (the order needed grows with c)."""
    from virasoro_blocks.api import BlockRequest, NumericalPolicy, evaluate_block_request
    from virasoro_blocks.qgate import QConvergenceError

    def request(c):
        return BlockRequest(operation="evaluate", family=family, central_charge=c, delta_ext=("0.2",) * 4,
                            exchange_delta="2", point={"z": "0.4", "zbar": "0.3"},
                            policy=NumericalPolicy(precision=30, order=12))

    assert evaluate_block_request(request("8")).metadata["q_truncation_digits"] >= 15
    with pytest.raises(QConvergenceError):
        evaluate_block_request(request("30000"), q_gate="strict")


@pytest.mark.parametrize("ext", [("0.125",) * 4, ("0.125", "0.625", "0.125", "0.625")])
@pytest.mark.parametrize("z", ["0.9", "-5+1e-20j", "1.4+0.001j"])
def test_global_block_beyond_three_quarters_with_integer_parameter_differences(ext, z):
    """Identical externals (c - a - b = 0) or a - b = 1: flint's continuation took the generic connection formula
    through Gamma(0) and returned [+/- 15.4], read as 0 (or kept 6 digits when merely flagged)."""
    from virasoro_blocks.global_blocks import global_evaluate_block

    with mp.workdps(30):
        h = [mp.mpf(x) for x in ext]
        zz, H = mp.mpmathify(z), mp.mpf("0.65")
        k = zz ** H * mp.hyp2f1(H + h[1] - h[0], H + h[2] - h[3], 2 * H, zz) / mp.mpf(16) ** H
        got = global_evaluate_block(zz, zz, H, H, h, coord="zzb")
        assert abs(got - k * k) <= mp.mpf(10) ** -26 * abs(k * k)


@pytest.mark.parametrize("family", ["virasoro", "global"])
def test_evaluate_on_the_negative_real_axis_is_the_euclidean_limit(family):
    """z = zbar = -0.3 at c = 4.5 gave -0.0056 - 0.0077i for 0.0095641 (only z > 1 was taken to z +- i0)."""
    kw = {"family": family, "central_charge": None if family == "global" else "4.5", "delta_ext": ("0.25",) * 4,
          "exchange_delta": "1.3", "exchange_spin": 0}
    with mp.workdps(30):
        on_axis = mp.mpc(_evaluate(point={"z": "-0.3", "zbar": "-0.3"}, **kw))
        eps = mp.mpf("1e-40")
        limit = mp.mpc(_evaluate(point={"z": mp.mpc("-0.3", eps), "zbar": mp.mpc("-0.3", -eps)}, **kw))
        assert abs(on_axis.imag) <= mp.mpf(10) ** -25 * abs(on_axis)
        assert abs(on_axis - limit) <= mp.mpf(10) ** -25 * abs(limit)


def test_torus_refuses_non_integer_spins_and_points_off_the_disk():
    """exchange_spin 1.5 was truncated to the spin-1 value (labeled 1.5); v = 1.5 (Im tau < 0) returned -1e572."""
    from virasoro_blocks.api import BlockRequest, NumericalPolicy, build_table_request, evaluate_block_request
    from virasoro_blocks.requests import RequestValidationError

    kw = dict(surface="torus", family="torus-character", coordinate="vvb", central_charge="8", exchange_delta="1.2",
              policy=NumericalPolicy(precision=30, order=10))
    with pytest.raises(RequestValidationError, match="non-negative integers"):
        evaluate_block_request(BlockRequest(operation="evaluate", exchange_spin=mp.mpf("1.5"),
                                            point={"v": "0.3", "vbar": "0.3"}, **kw))
    with pytest.raises(RequestValidationError, match=r"\|v\| < 1"):
        evaluate_block_request(BlockRequest(operation="evaluate", point={"v": "1.5", "vbar": "1.5"}, **kw))
    table_kw = dict(kw, coordinates=("vvb",), policy=NumericalPolicy(precision=30, order=10, lambda_order=2))
    with pytest.raises(RequestValidationError, match="non-negative integers"):
        build_table_request(BlockRequest(operation="table", **{k: v for k, v in table_kw.items() if k != "exchange_delta"}),
                            spin=mp.mpf("1.5"))


@pytest.mark.parametrize("field, value", [("lam", 3.5), ("q_order", "12"), ("c", "abc"), ("chiral_norm", 4.5),
                                          ("shared_sample_points", "no")])
def test_gap_spec_type_errors_are_clear(field, value):
    from virasoro_bootstrap.identical_scalar import IdenticalScalarGapSpec, prepare_identical_scalar_gap_build

    kw = dict(c=8, h_ext=mp.mpf(1) / 16, delta_gap=1, lam=3, npoles=4, q_order=10, dps=40, num_threads=1)
    kw[field] = value
    with pytest.raises(ValueError, match=field):
        prepare_identical_scalar_gap_build(IdenticalScalarGapSpec(**kw))


def test_api_table_defaults_to_the_cli_sampler(tmp_path):
    """build_table_request sampled nodes symbolically while the CLI's auto chose numeric: a parity '-' table got
    another node set (13 vs 14 nodes) and values off at 1e-6."""
    import json

    from virasoro_blocks.__main__ import main
    from virasoro_blocks.api import BlockRequest, NumericalPolicy, build_table_request
    from virasoro_blocks.serialization import serialized_table_from_json

    common = dict(h=("0.7", "0.35", "0.55", "0.975"), hb=("0.2", "0.35", "0.55", "0.475"))
    main(["table", "--central-charge", "8", "--basis", "parity_projected_B", "--parity-project", "-", "--parity-ext",
          "1", "1", "1", "1", "--h-ext", *common["h"], "--hbar-ext", *common["hb"], "--lambda", "3", "--order", "16",
          "--npoles", "6", "--spin-ranges", "1", "--precision", "40", "-o", str(tmp_path)])
    cli = serialized_table_from_json(json.load(open(tmp_path / "derivs_1.json")))
    with mp.workdps(40):
        api = build_table_request(BlockRequest(
            operation="table", central_charge="8", basis="parity_projected_B", parity="-",
            parity_external=("1",) * 4, h_ext=common["h"], hbar_ext=common["hb"], coordinates=("zzb",),
            policy=NumericalPolicy(precision=40, order=16, lambda_order=3, npoles=6)), spin=1)
        assert len(api.sample_points_x) == len(cli.sample_points_x)
        delta = mp.mpf(api.shift) + mp.mpf("0.7")
        a, b = api.evaluate("zzb", delta), cli.evaluate("zzb", delta)
        assert all(abs(mp.mpc(a[m][n]) - mp.mpc(b[m][n])) < mp.mpf(10) ** -35 for m in range(len(b))
                   for n in range(len(b[m])))


@pytest.mark.parametrize("extra, match", [
    ({"sector": "odd"}, "sector"),
    ({"exchange_delta": "2"}, "exchange_delta"),
    ({"torus_onepoint_ext": "0.3"}, "torus_onepoint_ext"),
    ({"zbar_shift": "-0.2"}, "zbar_shift"),
])
def test_api_table_refuses_fields_it_would_ignore(extra, match):
    from virasoro_blocks.api import BlockRequest, NumericalPolicy, build_table_request
    from virasoro_blocks.requests import RequestValidationError

    kw = dict(operation="table", central_charge="8", delta_ext=("0.2",) * 4, coordinates=("zzb",),
              policy=NumericalPolicy(precision=30, order=10, lambda_order=2, npoles=4))
    with pytest.raises(RequestValidationError, match=match):
        build_table_request(BlockRequest(**kw, **extra), spin=0)


def test_api_table_refuses_unknown_options():
    from virasoro_blocks.api import BlockRequest, NumericalPolicy, build_table_request

    request = BlockRequest(operation="table", central_charge="8", delta_ext=("0.2",) * 4, coordinates=("zzb",),
                           policy=NumericalPolicy(precision=30, order=10, lambda_order=2, npoles=4))
    with pytest.raises(TypeError, match="typo_option"):
        build_table_request(request, spin=0, typo_option=1)


@pytest.mark.parametrize("extra, match", [
    ({"gap": "1"}, "gap"), ({"center": "0.4"}, "center"), ({"basis": "scalar_G", "parity": "+"}, "parity"),
    ({"h_ext": ("0.1",) * 4, "hbar_ext": ("0.1",) * 4}, "delta_ext"),
])
def test_evaluate_refuses_table_only_and_shadowed_fields(extra, match):
    from virasoro_blocks.requests import RequestValidationError

    with pytest.raises(RequestValidationError, match=match):
        _evaluate(**extra)


def test_q_gate_no_false_alarm_at_large_c_and_high_order():
    """c = 1000 at orders 60-240 was 'unconverged' (the gate's series ran unguarded; the value was right)."""
    from virasoro_blocks.api import BlockRequest, NumericalPolicy, evaluate_block_request

    request = BlockRequest(operation="evaluate", central_charge="1000", delta_ext=("3",) * 4, exchange_delta="0.5",
                           point={"z": "0.5", "zbar": "0.5"}, policy=NumericalPolicy(precision=40, order=60))
    assert evaluate_block_request(request, q_gate="strict").metadata["q_truncation_digits"] > 30


def test_tables_record_the_q_truncation(tmp_path):
    """verified_digits compares against the same --order sampler; a Ramond order-10 table recorded 53 verified
    digits for a 12-15 digit table, with no warning.  Tables now also record the q-order gate's estimate."""
    import json

    from virasoro_blocks.__main__ import main

    main(["table", "--family", "ramond", "--basis", "ramond_scalar_G", "--central-charge", "1.2", "--delta-ext",
          "0.2", "0.2", "0.2", "0.2", "--lambda", "3", "--order", "10", "--precision", "40", "--target-digits", "40",
          "-o", str(tmp_path / "r")])
    doc = json.load(open(tmp_path / "r" / "derivs_0.json"))
    assert float(doc["q_truncation_digits"]) < 16 < float(doc["verified_digits"])
    main(["table", "--central-charge", "8", "--delta-ext", "0.2", "0.2", "0.2", "0.2", "--lambda", "3", "--order",
          "16", "--npoles", "6", "--spin-ranges", "0", "--precision", "40", "-o", str(tmp_path / "v")])
    assert "q_truncation_digits" in json.load(open(tmp_path / "v" / "derivs_0.json"))


def test_point_table_and_evaluate_take_the_spins_tables_take(tmp_path):
    """All three CLI commands accept signed scalar spin and half-integer ordered spin."""
    from virasoro_blocks.cli.evaluate import main as evaluate_main
    from virasoro_blocks.cli.point_table import main as point_table_main

    spinning = ["--central-charge", "8", "--spin-ext", "0.5", "0", "0", "0.5", "--delta-ext", "0.9", "0.7", "1.1",
                "1.45", "--exchange-delta", "3", "--exchange-spin", "5/2"]
    assert evaluate_main([*spinning, "--point", "z=0.4", "--point", "zbar=0.3", "--precision", "25",
                          "--format", "text"]) == 0
    assert point_table_main([*spinning, "--lambda", "2", "--order", "12", "--precision", "30",
                             "-o", str(tmp_path)]) in (0, None)
    assert point_table_main(["--central-charge", "8", "--delta-ext", "0.2", "0.2", "0.2", "0.2", "--exchange-delta",
                             "3", "--exchange-spin", "-2", "--lambda", "2", "--order", "12", "--precision", "30",
                             "-o", str(tmp_path)]) in (0, None)
    assert (tmp_path / "derivs_point_2.5.json").exists() and (tmp_path / "derivs_point_-2.json").exists()


def test_large_c_order_reduced_tail_is_reported_and_certified_far_enough():
    """At c = 1000 the order-reduced interpolant left the true block near x ~ 80 (nodes reach x ~ 8): the bootstrap
    reported gap 3 allowed where the true blocks exclude it, and the certificate only looked up to 30 above an edge.
    The build's verification (between the nodes and past them to the large-Delta regime) warns."""
    from virasoro_bootstrap.certification import _certification_delta_range
    from virasoro_bootstrap.identical_scalar import IdenticalScalarGapSpec, prepare_identical_scalar_gap_build

    with mp.workdps(60):
        spec = IdenticalScalarGapSpec(c=mp.mpf(1000), h_ext=mp.mpf(1) / 16, delta_gap=3, lam=5, spin_list=(0, 2),
                                      q_order=40, dps=60, npoles=10, num_threads=1)
        with pytest.warns(RuntimeWarning, match="does not reproduce the block") as record:
            prepare_identical_scalar_gap_build(spec)
    assert "Raise npoles" in str(record[0].message)
    assert _certification_delta_range(30.0, spec) > 250
    assert _certification_delta_range(30.0, IdenticalScalarGapSpec(c=8, h_ext=0.1, delta_gap=1, lam=3)) == 30.0


def test_sweep_validation_catches_enumerated_typos():
    """vb-submit --dry-run passed backend='sdpp' or twist_gap='matchgap'; the task failed at run time."""
    from virasoro_bootstrap.hpc.params import validate_sweep

    config = {"task": "scalar_gap_bisection", "params": {"c": "8", "h_ext": "0.0625", "lam": 7, "backend": "sdpp",
                                                         "twist_gap": "matchgap"}, "sweep": {"c": ["8"]}}
    errors = " ".join(validate_sweep(config))
    assert "backend = 'sdpp'" in errors and "did you mean 'match_gap'" in errors
    config["params"].update(backend="sdp", twist_gap="1.5")
    assert not [e for e in validate_sweep(config) if "backend" in e or "twist_gap" in e]



@pytest.mark.parametrize("field", ["constant", "base", "pole"])
@pytest.mark.parametrize("value", [mp.inf, -mp.inf, mp.nan, mp.mpc(1, mp.nan), mp.mpc(1, mp.inf)])
def test_damped_rational_rejects_nonfinite_inputs(field, value):
    from virasoro_blocks.interpolation import DampedRational

    kwargs = {"constant": 1, "base": ".2", "poles": (-1,)}
    if field == "pole":
        kwargs["poles"] = (value,)
    else:
        kwargs[field] = value
    with pytest.raises(ValueError, match="finite"):
        DampedRational(**kwargs)


@pytest.mark.parametrize("operation", ["evaluate", "evaluate_regulated", "shift_x"])
def test_prefactor_operations_do_not_drop_nan_imaginary_input(operation):
    from virasoro_blocks.interpolation import DampedRational

    prefactor = DampedRational(1, (-1,), ".2")
    args = (mp.mpc(1, mp.nan),)
    if operation == "evaluate_regulated":
        args += (mp.mpf(".01"),)
    with pytest.raises(ValueError, match="real"):
        getattr(prefactor, operation)(*args)


def test_interpolant_signed_difference_rejects_nan_imaginary_reference():
    from virasoro_blocks.interpolation import DampedRational, DeltaInterpolant
    from virasoro_blocks.polynomial import Polynomial

    interpolant = DeltaInterpolant(DampedRational(1, (), ".2"), 0, (), Polynomial([1]))
    with pytest.raises(ValueError, match="real-valued"):
        interpolant.signed_difference(lambda delta: mp.mpc(1, mp.nan), 1)
