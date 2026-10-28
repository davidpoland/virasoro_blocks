"""Edge cases of the public request, API, and CLI contracts.

Builds use lambda <= 4 and q-order <= 8 to keep this module in the fast tier.
"""

from __future__ import annotations

import json
from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_blocks.__main__ import main as table_main
from virasoro_blocks.api import (
    BlockPoleError,
    BlockRequest,
    BlockValue,
    NumericalPolicy,
    RequestValidationError,
    build_table_request,
    evaluate_block_request,
)
from virasoro_blocks.cli.point_table import main as point_table_main
from virasoro_blocks.point import point_block_table
from virasoro_blocks.superconformal_blocks import SuperBlockSampler2d
from virasoro_blocks.writers import _num_str


SCALAR = ("0.1",) * 4
SPINNING_H = ("0.6", "0.1", "0.1", "0.1")
SPINNING_HB = ("0.1", "0.1", "0.1", "0.1")


def _cli_error(capsys, runner, argv):
    with pytest.raises(SystemExit) as exc:
        runner(argv)
    assert exc.value.code == 2
    return capsys.readouterr().err


# ---------------------------------------------------------------------------
# 1. NS/Ramond 2d sampler: an unset zbar-center follows the z-center
# ---------------------------------------------------------------------------


def test_super_sampler_default_zbar_center_is_diagonal():
    with mp.workdps(30):
        kw = dict(
            c=mp.mpf("2.3"), q_order=6, m_max=2, n_max=2, sigma=mp.mpf(1),
            fold_parity=0, chiral_norm=None, z_shift=None, zbar_shift=None,
            sector=0, branch="++",
        )
        ext = [mp.mpf("0.1")] * 4
        center = mp.mpf("0.3")
        default = SuperBlockSampler2d(
            "ns", "zzb", 0, ext, ext, center=center, center_bar=None, **kw
        )(mp.mpf("1.7"))
        explicit = SuperBlockSampler2d(
            "ns", "zzb", 0, ext, ext, center=center, center_bar=center, **kw
        )(mp.mpf("1.7"))
        for m in range(3):
            for n in range(3):
                assert default[m][n] == explicit[m][n]
        # scalar G on the diagonal is z <-> zbar symmetric
        assert mp.almosteq(default[1][0], default[0][1], 1e-25)


# ---------------------------------------------------------------------------
# 2. point-table at a genuine Kac pole fails closed
# ---------------------------------------------------------------------------


def test_point_table_kac_pole_raises_block_pole_error():
    with mp.workdps(30):
        with pytest.raises(BlockPoleError):
            point_block_table(
                ["z"], basis="chiral_F", ell=0, point_value=mp.mpf("-3.625"),
                h_externals=[mp.mpf(x) for x in ("0.1", "0.2", "0.3", "0.4")],
                lam=2, c=mp.mpf(30), q_order=4,
            )


def test_point_table_cli_kac_pole_is_parser_error(capsys, tmp_path):
    err = _cli_error(capsys, point_table_main, [
        "--central-charge", "30", "--h-ext", ".1", ".2", ".3", ".4",
        "--coordinates", "z", "--exchange-h", "-3.625", "--order", "4",
        "--lambda", "2", "--precision", "30", "-o", str(tmp_path),
    ])
    assert "pole" in err


def test_point_table_global_sl2_pole_raises():
    with mp.workdps(30):
        with pytest.raises(BlockPoleError):
            point_block_table(
                ["z"], basis="global_chiral_F", ell=0, point_value=mp.mpf(-2),
                h_externals=[mp.mpf(x) for x in ("0.1", "0.2", "0.3", "0.4")],
                lam=2,
            )


def test_point_table_global_chiral_builds():
    """Build a global chiral point table without a sample_bucket interface."""
    with mp.workdps(30):
        h_ext = [mp.mpf(x) for x in ("0.1", "0.2", "0.3", "0.4")]
        table = point_block_table(
            ["z"], basis="global_chiral_F", ell=0, point_value=mp.mpf("1.3"),
            h_externals=h_ext, lam=2,
        )
        direct = evaluate_block_request(BlockRequest(
            operation="evaluate", family="global", basis="chiral_F",
            coordinate="z", h_ext=tuple(h_ext), exchange_h=mp.mpf("1.3"),
            point={"z": mp.mpf(1) / 2}, policy=NumericalPolicy(order=6, precision=30),
        ))
        frozen = table.evaluate("z", mp.mpf("1.3"))[0][0]
        assert mp.almosteq(frozen, direct.value, 1e-25)


# ---------------------------------------------------------------------------
# 3. odd-spin folded point tables keep lambda and every kept pair
# ---------------------------------------------------------------------------


def test_point_table_odd_spin_folded_keeps_lambda(tmp_path):
    rc = point_table_main([
        "--central-charge", "2.3", "--h-ext", *SCALAR, "--hbar-ext", *SCALAR,
        "--coordinates", "xt", "--exchange-delta", "2.3", "--exchange-spin", "1",
        "--order", "6", "--lambda", "4", "--precision", "30",
        "-o", str(tmp_path),
    ])
    assert rc == 0
    doc = json.loads((tmp_path / "derivs_point_1.json").read_text())
    assert doc["lambda"] == 4
    # the odd-spin scalar G is z <-> zbar even: kept pairs for xt, Lambda_+ = 4 (m + 2n <= 4)
    rows = doc["derivs"][0]
    assert [len(row) for row in rows] == [3, 2, 2, 1, 1]
    assert doc["parity_smoothing"] == "p_plus" and doc["scalar_G_odd"] == "even"


# ---------------------------------------------------------------------------
# 4/5. scalar_G with spinning externals is rejected everywhere
# ---------------------------------------------------------------------------


def test_table_cli_rejects_scalar_G_with_spinning_externals(capsys, tmp_path):
    err = _cli_error(capsys, table_main, [
        "--central-charge", "2.3", "--h-ext", *SPINNING_H,
        "--hbar-ext", *SPINNING_HB, "--basis", "scalar_G", "--lambda", "2",
        "--order", "4", "--precision", "30", "-o", str(tmp_path),
    ])
    assert "scalar externals" in err


@pytest.mark.parametrize("family, c", [
    ("virasoro", "2.3"), ("global", None), ("ns", "2.3"), ("ramond", "2.3"),
])
def test_table_adapters_reject_scalar_G_with_spinning_externals(family, c):
    request = BlockRequest(
        operation="table", family=family, basis="scalar_G",
        coordinates=("zzb",), central_charge=c, h_ext=SPINNING_H,
        hbar_ext=SPINNING_HB,
        policy=NumericalPolicy(order=4, lambda_order=2, npoles=2, precision=30),
    )
    with pytest.raises(RequestValidationError, match="scalar external"):
        build_table_request(request)


@pytest.mark.parametrize("family", ["ns", "ramond"])
def test_super_evaluate_scalar_G_rejects_spinning_externals(family):
    request = BlockRequest(
        operation="evaluate", family=family, basis="scalar_G",
        coordinate="zzb", central_charge="2.3", h_ext=SPINNING_H,
        hbar_ext=SPINNING_HB, exchange_delta="1.3",
        point={"z": "0.3", "zbar": "0.4"},
        policy=NumericalPolicy(order=6, precision=30),
    )
    with pytest.raises(RequestValidationError, match="scalar external"):
        evaluate_block_request(request)


# ---------------------------------------------------------------------------
# 6. BlockValue number formatting is precision-safe and relative
# ---------------------------------------------------------------------------


def test_block_value_keeps_digits_beyond_ambient_precision():
    with mp.workdps(60):
        value = mp.mpf(1) / 3
    assert mp.mp.dps < 60
    text = BlockValue(value=value, metadata={"precision": 60}).as_dict()["value"]
    assert text.startswith("0." + "3" * 55)


def test_block_value_keeps_small_genuine_imaginary_part():
    with mp.workdps(40):
        value = mp.mpc("1e-20", "1e-20")
        out = BlockValue._number(value, 40)
        assert isinstance(out, dict) and mp.mpf(out["imag"]) > 0
        # rounding noise relative to the value is still dropped
        assert isinstance(BlockValue._number(mp.mpc(2, "1e-60"), 40), str)


# ---------------------------------------------------------------------------
# 7. Mathematica number literals
# ---------------------------------------------------------------------------


def test_mathematica_number_literals():
    with mp.workdps(30):
        assert _num_str(mp.mpf("1.04e-27"), 10, mma=True) == "1.04`10*^-27"
        assert _num_str(mp.mpf("-3e40"), 10, mma=True) == "-3.0`10*^40"
        assert _num_str(mp.nan, 10, mma=True) == "Indeterminate"
        assert _num_str(mp.inf, 10, mma=True) == "Infinity"
        assert _num_str(-mp.inf, 10, mma=True) == "-Infinity"
        assert _num_str(mp.mpf("2.5"), 10, mma=True) == "2.5`10"
        assert _num_str(mp.mpf(0), 10, mma=True) == "0"
        # JSON keeps the mpmath spelling
        assert _num_str(mp.mpf("1.04e-27"), 10) == "1.04e-27"


# ---------------------------------------------------------------------------
# 8. decimal exchange labels satisfy h - hbar == spin exactly
# ---------------------------------------------------------------------------


def test_decimal_exchange_spin_matches_h_minus_hbar_exactly():
    request = BlockRequest(
        operation="evaluate", family="global", basis="global_ordered_B",
        coordinate="zzb", h_ext=SPINNING_H, hbar_ext=SPINNING_HB,
        exchange_h=2.7, exchange_hbar=0.7, exchange_spin=2,
        point={"z": "0.3", "zbar": "0.4"},
        policy=NumericalPolicy(order=6, precision=30),
    )
    evaluate_block_request(request)  # Exact decimal spin difference: 2.7 - 0.7 = 2.


# ---------------------------------------------------------------------------
# 9. build_table_request robustness
# ---------------------------------------------------------------------------


_SMALL = dict(order=6, lambda_order=2, npoles=4, precision=30)


@pytest.mark.parametrize("family, c", [
    # auto basis is "<family>_<short>" for every non-virasoro family (ns covers it; a ramond table is built by
    # test_cli_evaluate.py::test_super_table_adapter_builds_without_cli_callback)
    ("virasoro", "2.3"), ("global", None), ("ns", "2.3"),
])
def test_build_table_request_auto_basis_default_coordinate(family, c):
    table = build_table_request(BlockRequest(
        operation="table", family=family, central_charge=c,
        h_ext=SCALAR, hbar_ext=SCALAR, policy=NumericalPolicy(**_SMALL),
    ))
    assert table.coordinates == ["zzb"] or tuple(table.coordinates) == ("zzb",)


@pytest.mark.parametrize("family, c", [
    # the delta_ext -> h_ext normalization is family-independent (_normalize_sphere_table_request)
    ("virasoro", "2.3"), ("global", None),
])
def test_build_table_request_delta_ext_matches_h_ext(family, c):
    common = dict(
        operation="table", family=family, basis="scalar_G",
        coordinates=("zzb",), central_charge=c,
        policy=NumericalPolicy(**_SMALL),
    )
    by_delta = build_table_request(BlockRequest(delta_ext=("0.2",) * 4, **common))
    by_h = build_table_request(BlockRequest(h_ext=SCALAR, hbar_ext=SCALAR, **common))
    with mp.workdps(30):
        a = by_delta.evaluate("zzb", mp.mpf("1.7"))
        b = by_h.evaluate("zzb", mp.mpf("1.7"))
        assert mp.almosteq(a[1][0], b[1][0], 1e-20)


def test_build_table_request_npoles_all_virasoro_is_exact_path():
    policy = NumericalPolicy(order=6, lambda_order=2, npoles="all", precision=30)
    table = build_table_request(BlockRequest(
        operation="table", central_charge="2.3", coordinates=("zzb",),
        h_ext=SCALAR, hbar_ext=SCALAR, policy=policy,
    ))
    assert table.sample_points_x is None  # the exact (method="full") table


@pytest.mark.parametrize("family, c", [("global", None), ("ns", "2.3")])
def test_build_table_request_npoles_all_rejected_where_unsupported(family, c):
    policy = NumericalPolicy(order=6, lambda_order=2, npoles="all", precision=30)
    with pytest.raises(RequestValidationError, match="npoles='all'"):
        build_table_request(BlockRequest(
            operation="table", family=family, basis="scalar_G",
            central_charge=c, coordinates=("zzb",), h_ext=SCALAR,
            hbar_ext=SCALAR, policy=policy,
        ))


def test_build_table_request_needs_lambda_order():
    with pytest.raises(RequestValidationError, match="lambda_order"):
        build_table_request(BlockRequest(
            operation="table", central_charge="2.3", coordinates=("zzb",),
            h_ext=SCALAR, hbar_ext=SCALAR,
            policy=NumericalPolicy(order=6, npoles=4, precision=30),
        ))


@pytest.mark.parametrize("kwargs", [
    dict(order=2.7), dict(order=-0.5), dict(npoles=2.5), dict(npoles="most"),
    dict(lambda_order=1.5), dict(precision=30.5), dict(order=True),
])
def test_numerical_policy_rejects_non_integers(kwargs):
    with pytest.raises(RequestValidationError):
        NumericalPolicy(**kwargs)


def test_numerical_policy_normalizes_all_spelling():
    assert NumericalPolicy(npoles="ALL").npoles == "all"
    assert NumericalPolicy(order=6.0).order == 6


def test_table_request_precision_minimum():
    with pytest.raises(RequestValidationError, match="precision"):
        build_table_request(BlockRequest(
            operation="table", central_charge="2.3", coordinates=("zzb",),
            h_ext=SCALAR, hbar_ext=SCALAR,
            policy=NumericalPolicy(order=6, lambda_order=2, npoles=4, precision=12),
        ))


# ---------------------------------------------------------------------------
# 10. table CLI: parser errors instead of tracebacks; output paths
# ---------------------------------------------------------------------------


_TABLE = [
    "--central-charge", "2.3", "--delta-ext", "0.2", "0.2", "0.2", "0.2",
    "--precision", "30",
]


@pytest.mark.parametrize("extra, message", [
    (["--lambda", "-1"], "--lambda must be >= 0"),
    (["--lambda", "2", "--precision", "10"], "--precision >= 20"),
    # (an integer N: at z0 = 0.7 the base 16 q(z0) / N is 1.195 for N = 1)
    (["--lambda", "2", "--order", "4", "--chiral-norm", "1", "--z-center", "0.7"], "damping base"),
    (["--lambda", "2", "--order", "4", "--chiral-norm", "1", "--z-center", "0.7", "--family", "ns"],
     "damping base"),
    (["--lambda", "2", "--order", "4", "--chiral-norm", "0.5"], "--chiral-norm must be a positive integer"),
    (["--lambda", "2", "--order", "0", "--basis", "global_scalar_G"],
     "--order must be >= 1"),
    (["--lambda", "2", "--order", "4", "--spin-ranges", "0", "2",
      "--output-template", "fixed.json"], "{spin}"),
])
def test_table_cli_parser_errors(capsys, tmp_path, extra, message):
    err = _cli_error(capsys, table_main, [*_TABLE, *extra, "-o", str(tmp_path)])
    assert message in err


def test_table_cli_order_zero_integer_npoles_builds(tmp_path):
    table_main([*_TABLE, "--lambda", "2", "--order", "0", "--npoles", "4",
                "-o", str(tmp_path)])
    assert (tmp_path / "derivs_0.json").exists()


def test_table_cli_output_template_creates_subdirectory(tmp_path):
    table_main([*_TABLE, "--lambda", "2", "--order", "4",
                "--output-template", "sub/x_{spin}.json", "-o", str(tmp_path)])
    assert (tmp_path / "sub" / "x_0.json").exists()


def test_ns_chiral_table_honours_output_template(tmp_path):
    table_main([
        "--family", "ns", "--central-charge", "2.3", "--coordinates", "z",
        "--h-ext", *SCALAR, "--lambda", "2", "--order", "4", "--npoles", "2",
        "--precision", "30", "--output-template", "ns_{spin}.json",
        "-o", str(tmp_path),
    ])
    assert (tmp_path / "ns_chiral.json").exists()


def test_ns_auto_basis_scalar_spelled_with_hbar_ext(tmp_path):
    table_main([
        "--family", "ns", "--central-charge", "2.3", "--h-ext", *SCALAR,
        "--hbar-ext", *SCALAR, "--lambda", "2", "--order", "4", "--npoles", "2",
        "--precision", "30", "-o", str(tmp_path),
    ])
    doc = json.loads((tmp_path / "derivs_0.json").read_text())
    assert doc["basis"] == "ns_scalar_G"


# ---------------------------------------------------------------------------
# 11. Fraction inputs
# ---------------------------------------------------------------------------


def test_fraction_inputs_are_accepted_exactly():
    request = BlockRequest(
        operation="table", central_charge=Fraction(7, 10), coordinates=("zzb",),
        basis="scalar_G", h_ext=(Fraction(1, 10),) * 4,
        hbar_ext=(Fraction(1, 10),) * 4,
        policy=NumericalPolicy(order=6, lambda_order=2, npoles=4, precision=40),
    )
    with mp.workdps(40):
        # Stored inputs carry guard digits for verification and escalation;
        # rounding to the working precision must recover the requested value.
        assert +request.central_charge == mp.mpf(7) / 10
    build_table_request(request)
    value = evaluate_block_request(BlockRequest(
        operation="evaluate", central_charge=Fraction(23, 10), coordinate="z",
        basis="chiral_F", h_ext=(Fraction(1, 10),) * 4,
        exchange_h=Fraction(3, 5), point={"z": Fraction(1, 3)},
        policy=NumericalPolicy(order=6, precision=30),
    ))
    assert mp.isfinite(value.value)
