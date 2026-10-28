"""The scalar G is z <-> zbar even at every spin: G = [B(h, hbar) + B(hbar, h)] / (1 + delta_{ell, 0}).

The z <-> zbar-odd B(h, hbar) - B(hbar, h) is the parity_projected_B basis with eta = -1.  Written scalar-G tables
and towers record ``scalar_G_odd = "even"``, and the readers refuse odd-spin scalar-G data without it.
"""

from __future__ import annotations

import json

import mpmath as mp
import pytest

from virasoro_blocks.composer import B_block, G_block, parity_projected_B_block
from virasoro_blocks.serialization import serialized_table_from_json

H_EXT = [mp.mpf("0.1"), mp.mpf("0.13"), mp.mpf("0.11"), mp.mpf("0.07")]


def _close(a, b, tol):
    return abs(mp.mpc(a) - mp.mpc(b)) <= tol * (1 + abs(mp.mpc(b)))


@pytest.mark.parametrize("ell", [1, 3])
def test_odd_spin_scalar_G_is_the_symmetric_sum(ell):
    mp.mp.dps = 40
    delta, z, zbar = mp.mpf("4.2"), mp.mpf("0.31"), mp.mpf("0.47")
    h, hbar = (delta + ell) / 2, (delta - ell) / 2
    forward = B_block(h, hbar, z, zbar, H_EXT, H_EXT, c=7, q_order=10)
    reverse = B_block(hbar, h, z, zbar, H_EXT, H_EXT, c=7, q_order=10)
    G = G_block(delta, ell, z, zbar, H_EXT, c=7, q_order=10)
    assert _close(G, forward + reverse, mp.mpf("1e-35"))
    assert _close(G_block(delta, ell, zbar, z, H_EXT, c=7, q_order=10), G, mp.mpf("1e-30"))
    # the z <-> zbar-odd structure is the eta = -1 projection, antisymmetric under z <-> zbar
    odd = parity_projected_B_block(h, hbar, z, zbar, H_EXT, H_EXT, parity="-", c=7, q_order=10)
    assert _close(odd, forward - reverse, mp.mpf("1e-35"))
    swapped = parity_projected_B_block(h, hbar, zbar, z, H_EXT, H_EXT, parity="-", c=7, q_order=10)
    assert _close(swapped, -odd, mp.mpf("1e-30"))


def test_odd_spin_tables_are_symmetric_in_every_family():
    from virasoro_blocks.derivs import G_zzb_table, global_G_zzb_table, global_parity_projected_B_zzb_table
    from virasoro_blocks.superconformal_blocks import super_scalar_G_block

    mp.mp.dps = 40
    delta, ell = mp.mpf("3.6"), 1
    tables = [G_zzb_table(delta, ell, H_EXT, 2, 2, c=7, q_order=10), global_G_zzb_table(delta, ell, H_EXT, 2, 2)]
    for table in tables:
        for m in range(3):
            assert abs(table[m][m]) > mp.mpf("1e-10")       # no z <-> zbar-odd diagonal zero
            for n in range(3):
                assert _close(table[m][n], table[n][m], mp.mpf("1e-30"))
    odd = global_parity_projected_B_zzb_table(delta, ell, H_EXT, H_EXT, 2, 2, parity="-")
    assert all(abs(odd[m][m]) < mp.mpf("1e-30") for m in range(3))
    z, zbar = mp.mpf("0.3"), mp.mpf("0.42")
    ns = super_scalar_G_block("ns", delta, 3, z, zbar, [mp.mpf("0.2")] * 4, c=mp.mpf("1.3"), q_order=6)
    ns_swapped = super_scalar_G_block("ns", delta, 3, zbar, z, [mp.mpf("0.2")] * 4, c=mp.mpf("1.3"), q_order=6)
    assert _close(ns, ns_swapped, mp.mpf("1e-25"))


def _doc(spin, **extra):
    return {"coordinates": ["zzb"], "lambda": 0, "basis": "scalar_G", "spin": spin,
            "exp_base": "1", "variable_minus_x": "0", "pole_list_x": [], "derivs": [[[["1"]]]], **extra}


@pytest.mark.parametrize("basis", ["scalar_G", "global_scalar_G", "ns_scalar_G"])
def test_old_convention_odd_files_are_refused_and_even_ones_load(basis):
    with pytest.raises(ValueError, match="scalar_G_odd"):
        serialized_table_from_json(_doc(1, basis=basis))
    # an odd tower of the old convention (exchanged parity -1), whatever its spin label
    with pytest.raises(ValueError, match="scalar_G_odd"):
        serialized_table_from_json(_doc(4, basis=basis, variable="L", parity_exchanged="-1"))
    assert serialized_table_from_json(_doc(1, basis=basis, scalar_G_odd="even")).evaluate("zzb", 1)[0][0] == 1
    # nothing changed for even spins: an even-spin file without the record still loads
    assert serialized_table_from_json(_doc(2, basis=basis)).evaluate("zzb", 1)[0][0] == 1
    assert serialized_table_from_json(_doc(4, basis=basis, variable="L")).evaluate("zzb", 1)[0][0] == 1


def test_bootstrap_block_check_refuses_an_old_odd_tower(tmp_path):
    from virasoro_blocks._conventions import check_scalar_G_odd_record

    check_scalar_G_odd_record(_doc(8, variable="L"))
    with pytest.raises(ValueError, match=r"block file .*tower\.json holds odd-spin scalar_G data"):
        check_scalar_G_odd_record(_doc(9, variable="L", parity_exchanged="-1"), f"block file {tmp_path / 'tower.json'}")


def _cli_error(capsys, args):
    from virasoro_blocks.__main__ import main

    with pytest.raises(SystemExit):
        main(args)
    return capsys.readouterr().err


@pytest.mark.parametrize("family", ["virasoro", "global"])
def test_cli_refuses_an_odd_scalar_G_tower(capsys, tmp_path, family):
    options = ["--central-charge", "32"] if family == "virasoro" else []
    err = _cli_error(capsys, [
        "table", "--family", family, *options, "--delta-ext", *["0.125"] * 4, "--large-spin-lmin", "3",
        "--large-spin-twists", "0.25", "--npoles", "4", "--lambda", "1", "--order", "8", "--precision", "30",
        "--parity-exchanged", "-1", "--coordinates", "zzb", "--format", "json", "-o", str(tmp_path)])
    assert "no odd scalar_G tower" in err
    assert "parity_projected_B tower with projection '-'" in err and "spinning externals only" in err
    assert not list(tmp_path.iterdir())


def test_cli_odd_spin_tower_and_fixed_spin_tables_record_the_convention(capsys, tmp_path):
    from virasoro_blocks.__main__ import main

    common = ["--family", "global", "--delta-ext", *["0.125"] * 4, "--npoles", "4", "--lambda", "1", "--order", "8",
              "--precision", "30", "--coordinates", "zzb", "xt", "--format", "json"]
    main(["table", *common, "--large-spin-lmin", "3", "--large-spin-twists", "0.25", "-o", str(tmp_path / "tower")])
    tower = json.loads((tmp_path / "tower" / "derivs_large_spin_0.json").read_text())
    # one tower covers every integer L >= 3, so it is labeled by spin 3 and records no exchanged parity
    assert tower["spin"] == 3 and tower["parity_exchanged"] is None
    assert tower["scalar_G_odd"] == "even" and tower["parity_smoothing"] == "p_plus"
    serialized_table_from_json(tower)
    err = _cli_error(capsys, ["table", *common, "--basis", "scalar_G", "--spin-ranges", "1", "--parity-exchanged", "-1",
                              "-o", str(tmp_path / "fixed")])
    assert "does not apply to scalar_G" in err and "parity_projected_B with parity '-'" in err


def test_api_refuses_an_exchanged_parity_for_scalar_G():
    from virasoro_blocks.api import build_table_request
    from virasoro_blocks.requests import BlockRequest, NumericalPolicy, RequestValidationError

    request = BlockRequest(operation="table", surface="sphere", family="global", basis="global_scalar_G",
                           coordinate="zzb", h_ext=("0.1",) * 4, hbar_ext=("0.1",) * 4, parity_exchanged="-1",
                           policy=NumericalPolicy(precision=30, order=8, lambda_order=1, npoles=4))
    with pytest.raises(RequestValidationError, match="parity_projected_B with parity '-'"):
        build_table_request(request, spin=1)


def test_evaluate_output_records_the_convention():
    from virasoro_blocks.api import evaluate_block_request
    from virasoro_blocks.requests import BlockRequest, NumericalPolicy

    request = BlockRequest(operation="evaluate", surface="sphere", family="global", basis="global_scalar_G",
                           coordinate="zzb", h_ext=("0.1",) * 4, hbar_ext=("0.1",) * 4,
                           point={"z": ".3", "zbar": ".42"},
                           exchange_delta="3.6", exchange_spin=1, policy=NumericalPolicy(precision=30))
    value = evaluate_block_request(request)
    assert value.as_dict()["scalar_G_odd"] == "even"
    with mp.workdps(30):
        h, hbar = mp.mpf("2.3"), mp.mpf("1.3")
        from virasoro_blocks.global_blocks import global_evaluate_block

        z, zbar = mp.mpf(".3"), mp.mpf(".42")
        expected = (global_evaluate_block(z, zbar, h, hbar, [mp.mpf("0.1")] * 4)
                    + global_evaluate_block(z, zbar, hbar, h, [mp.mpf("0.1")] * 4))
        assert _close(value.value, expected, mp.mpf("1e-25"))
