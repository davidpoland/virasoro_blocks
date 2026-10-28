"""Explicit ordered tables retain individual spin orientations for scalar externals."""

from dataclasses import replace
import json
import subprocess
import sys

import mpmath as mp
import pytest

from virasoro_blocks.api import (
    BlockRequest, NumericalPolicy, RequestValidationError, build_table_request,
)
from virasoro_blocks.serialization import serialized_table_from_json


EXTERNALS = (".1", ".2", ".3", ".4")
PAIRS = ((0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0))


def _request(family="virasoro", **kwargs):
    fields = dict(
        operation="table", family=family, basis="ordered_B", coordinate="zzb",
        h_ext=EXTERNALS, central_charge=None if family == "global" else 30,
        policy=NumericalPolicy(precision=45, lambda_order=2, order=2),
    )
    fields.update(kwargs)
    return BlockRequest(**fields)


def _assert_grid(actual, expected, digits=18):
    for m, n in PAIRS:
        assert abs(actual[m][n] / expected[m][n] - 1) < mp.mpf(10) ** -digits, (m, n)
    # Both derivatives must survive serialization, and spin one must not turn
    # the point value into scalar_G's zero or its derivatives into a parity pair.
    assert actual[0][0] != 0
    assert abs(actual[0][1] - actual[1][0]) > abs(actual[0][1]) / 10


@pytest.mark.parametrize("family", ["virasoro", "global", "ns", "ramond"])
def test_public_ordered_scalar_table_matches_direct_derivatives(family):
    from virasoro_blocks.derivs import B_zzb_table, global_B_zzb_table
    from virasoro_blocks.superconformal_blocks import numeric_full_table

    table = build_table_request(_request(family), spin=1)
    with mp.workdps(60):
        delta = mp.mpf("3.7")
        ext = list(map(mp.mpf, EXTERNALS))
        h, hbar = (delta + 1) / 2, (delta - 1) / 2
        if family == "virasoro":
            expected = B_zzb_table(h, hbar, ext, ext, 2, 2, c=30, q_order=2)
        elif family == "global":
            expected = global_B_zzb_table(h, hbar, ext, ext, 2, 2)
        else:
            expected = numeric_full_table(
                family, "ordered_B", delta, 1, "zzb", ext,
                c=30, q_order=2, m_max=2, n_max=2,
            )
        _assert_grid(table.evaluate("zzb", delta), expected)


@pytest.mark.parametrize("c,center,sample_via,npoles", [
    (30, None, "symbolic", None),          # cached chiral series, symbolic samples
    (30, ".3", "numeric", None),        # fresh series, ordered numeric samples
    (30, ".3", "symbolic", None),       # fresh ordered symbolic table
    (30, None, "symbolic", "all"),       # exact rational table, numeric verification
    (1, None, "numeric", None),           # merged-pole regularized sampler
])
def test_virasoro_ordered_scalar_build_paths(c, center, sample_via, npoles):
    from virasoro_blocks.coordinates import COORDINATES
    from virasoro_blocks.recursion import F_block
    from virasoro_blocks.table_backends import VirasoroBackendOptions

    request = _request(central_charge=c, center=center, center_bar=".45" if center else None)
    request = replace(request, coordinates=("zzb", "yyb", "qqb"),
                      z_shift=".125" if center else None, zbar_shift="-.25" if center else None,
                      policy=replace(request.policy, npoles=npoles))
    table = build_table_request(
        request, spin=1, backend_options=VirasoroBackendOptions(sample_via=sample_via),
    )
    with mp.workdps(60):
        delta = mp.mpf("3.7")
        ext = list(map(mp.mpf, EXTERNALS))
        z0, zb0 = mp.mpf(center or ".5"), mp.mpf(".45" if center else ".5")
        for coordinate in request.coordinates:
            spec = COORDINATES[coordinate]
            # Differentiate the chiral point functions independently of the
            # polynomial-table and numeric-sampler assembly being exercised.
            left = mp.taylor(lambda u: F_block((delta + 1) / 2, spec.z_of_u(u), ext,
                                               c=c, q_order=2, z_shift=request.z_shift), spec.u_of_z(z0), 2)
            right = mp.taylor(lambda u: F_block((delta - 1) / 2, spec.z_of_u(u), ext,
                                                c=c, q_order=2, z_shift=request.zbar_shift), spec.u_of_z(zb0), 2)
            expected = [[a * b for b in right] for a in left]
            _assert_grid(table.evaluate(coordinate, delta), expected)


@pytest.mark.parametrize("spin", [0, -1, .5])
def test_ordered_scalar_table_uses_signed_spin_and_single_product_normalization(spin):
    from virasoro_blocks.derivs import B_zzb_table

    table = build_table_request(_request(), spin=spin)
    with mp.workdps(60):
        delta = mp.mpf("3.7")
        h, hbar = (delta + spin) / 2, (delta - spin) / 2
        ext = list(map(mp.mpf, EXTERNALS))
        expected = B_zzb_table(h, hbar, ext, ext, 2, 2, c=30, q_order=2)
        actual = table.evaluate("zzb", delta)
        for m, n in PAIRS:
            assert abs(actual[m][n] / expected[m][n] - 1) < mp.mpf("1e-30")


@pytest.mark.parametrize("family,basis,sector,branch", [
    ("virasoro", "ordered_B", "even", None),
    ("global", "global_ordered_B", "even", None),
    ("ns", "ns_ordered_B", "even", None),
    ("ns", "ordered_B", "odd", None),
    ("ramond", "ramond_ordered_B", "even", "++"),
    ("ramond", "ordered_B", "odd", "+-"),
    ("ramond", "ramond_ordered_B", "even", "-+"),
    ("ramond", "ordered_B", "odd", "--"),
])
def test_table_cli_writes_ordered_scalar_basis_and_both_derivative_orientations(
    family, basis, sector, branch, tmp_path,
):
    from virasoro_blocks import derivs
    from virasoro_blocks.superconformal_blocks import numeric_full_table

    args = ["table", "--family", family, "--basis", basis,
            "--delta-ext", ".2", ".4", ".6", ".8", "--spin-ranges", "1",
            "--coordinates", "zzb", "yyb", "qqb", "--lambda", "2", "--order", "2",
            "--precision", "45", "--format", "both", "--q-gate", "off", "-o", str(tmp_path)]
    if family != "global":
        args += ["--central-charge", "30"]
    if family in {"ns", "ramond"}:
        args += ["--super-sector", sector]
    if branch is not None:
        args += ["--ramond-branch", branch]
    result = subprocess.run([sys.executable, "-m", "virasoro_blocks", *args],
                            cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    document = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert document["basis"] == ("ordered_B" if family == "virasoro" else f"{family}_ordered_B")
    mathematica = list(tmp_path.glob("*.m"))
    assert len(mathematica) == 3
    text = "\n".join(path.read_text() for path in mathematica)
    table = serialized_table_from_json(document)
    with mp.workdps(60):
        delta = mp.mpf("3.7")
        ext = list(map(mp.mpf, EXTERNALS))
        for coordinate in ("zzb", "yyb", "qqb"):
            assert f"{coordinate}Deriv[0,1]" in text
            assert f"{coordinate}Deriv[1,0]" in text
            if family in {"ns", "ramond"}:
                expected = numeric_full_table(
                    family, "ordered_B", delta, 1, coordinate, ext, c=30,
                    q_order=2, m_max=2, n_max=2, sector=sector, branch=branch or "++",
                )
            else:
                builder = getattr(derivs, f"{'global_' if family == 'global' else ''}B_{coordinate}_table")
                kwargs = {} if family == "global" else {"c": 30, "q_order": 2}
                expected = builder((delta+1)/2, (delta-1)/2, ext, ext, 2, 2, **kwargs)
            _assert_grid(table.evaluate(coordinate, delta), expected)


@pytest.mark.parametrize("family", ["virasoro", "global"])
def test_table_cli_keeps_single_product_at_zero_and_half_integer_spin(family, tmp_path):
    from virasoro_blocks.derivs import B_zzb_table, global_B_zzb_table

    args = ["table", "--family", family, "--basis", "ordered_B",
            "--h-ext", *EXTERNALS, "--hbar-ext", *EXTERNALS,
            "--spin-ranges", "0", "1/2", "--coordinates", "zzb", "--lambda", "1",
            "--order", "2", "--precision", "45", "--q-gate", "off", "-o", str(tmp_path)]
    if family == "virasoro":
        args += ["--central-charge", "30"]
    result = subprocess.run([sys.executable, "-m", "virasoro_blocks", *args],
                            cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    documents = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    assert {doc["spin"] for doc in documents} == {0, .5}
    with mp.workdps(60):
        delta, ext = mp.mpf("3.7"), list(map(mp.mpf, EXTERNALS))
        for doc in documents:
            spin = mp.mpf(doc["spin"])
            builder = B_zzb_table if family == "virasoro" else global_B_zzb_table
            kwargs = {"c": 30, "q_order": 2} if family == "virasoro" else {}
            expected = builder((delta+spin)/2, (delta-spin)/2, ext, ext, 1, 1, **kwargs)
            got = serialized_table_from_json(doc).evaluate("zzb", delta)
            for m, n in ((0, 0), (0, 1), (1, 0)):
                assert abs(got[m][n]/expected[m][n]-1) < mp.mpf("1e-25")


@pytest.mark.parametrize("family", ["virasoro", "global"])
def test_point_table_cli_accepts_ordered_scalar_externals(family, tmp_path):
    from virasoro_blocks.cli.point_table import main

    args = ["--family", family, "--basis", "ordered_B", "--delta-ext", ".2", ".4", ".6", ".8",
            "--exchange-delta", "3.7", "--exchange-spin", "1", "--coordinates", "zzb",
            "--lambda", "2", "--order", "2", "--precision", "30", "--q-gate", "off",
            "-o", str(tmp_path)]
    if family == "virasoro":
        args += ["--central-charge", "30"]
    main(args)
    document = json.loads(next(tmp_path.glob("*.json")).read_text())
    table = serialized_table_from_json(document)
    values = table.evaluate("zzb", mp.mpf("3.7"))
    assert values[0][0] != 0 and values[0][1] != 0


@pytest.mark.parametrize("family", ["virasoro", "global", "ns", "ramond"])
def test_ordered_scalar_table_rejects_parity_fold_before_build(family):
    with pytest.raises(RequestValidationError, match="non-folded"):
        build_table_request(_request(family, coordinate="xt"), spin=1)


@pytest.mark.parametrize("family", ["virasoro", "global"])
def test_large_spin_cli_accepts_ordered_scalar_externals(family, tmp_path):
    from virasoro_blocks.__main__ import main

    args = ["table", "--family", family, "--basis", "ordered_B",
            "--delta-ext", ".2", ".4", ".6", ".8", "--coordinates", "zzb",
            "--large-spin-lmin", "2", "--large-spin-twists", ".7",
            "--lambda", "1", "--order", "6", "--npoles", "4",
            "--precision", "30", "--q-gate", "off", "-o", str(tmp_path)]
    if family == "virasoro":
        args += ["--central-charge", "30"]
    main(args)
    document = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert document["basis"] == ("ordered_B" if family == "virasoro" else "global_ordered_B")
    with mp.workdps(30):
        values = serialized_table_from_json(document).evaluate("zzb", mp.mpf("3.7"))
        assert values[0][0] != 0 and values[0][1] != 0
        assert abs(values[0][1] - values[1][0]) > abs(values[0][1]) / 10


@pytest.mark.parametrize("family", ["virasoro", "global", "ns", "ramond"])
def test_table_cli_rejects_folded_ordered_scalar_basis(family, tmp_path, capsys):
    from virasoro_blocks.__main__ import main

    args = ["table", "--family", family, "--basis", "ordered_B",
            "--coordinates", "xt", "--lambda", "1", "--order", "2",
            "--precision", "30", "--q-gate", "off", "-o", str(tmp_path)]
    if family != "global":
        args += ["--central-charge", "30"]
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    message = capsys.readouterr().err.split("error:")[-1]
    assert "coordinates" in message or "fold" in message
    assert not list(tmp_path.iterdir())
