"""Tests for the operation-oriented CLI and point-value schema."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import mpmath as mp
import pytest

from virasoro_blocks.__main__ import main
from virasoro_blocks.api import build_table_request
from virasoro_blocks.evaluation import evaluate_block_request
from virasoro_blocks.recursion import F_block
from virasoro_blocks.composer import G_block
from virasoro_blocks.requests import BlockPoleError, BlockRequest, NumericalPolicy
from virasoro_blocks.table_adapters import (
    table_adapter_for,
    table_adapters,
)
from virasoro_blocks.superconformal_blocks import (
    ns_chiral_block,
    ramond_chiral_block,
)
from virasoro_blocks.modular import torus_character_table
from virasoro_blocks.serialization import serialized_table_from_json
from virasoro_blocks.torus_onept import (
    build_torus_onept_kernel,
    torus_onept_table_values,
)


def test_table_adapter_registry_covers_all_capability_families():
    families = {adapter.family for adapter in table_adapters()}
    assert families == {
        "virasoro", "global", "ns", "ramond",
        "torus-character", "torus-onepoint",
    }
    assert table_adapter_for("torus-character").capabilities.family == "torus-character"


def test_root_help_is_a_concise_operation_index(capsys):
    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "evaluate" in output
    assert "point-table" in output
    assert "--basis" not in output


def test_table_help_names_the_operation_and_hides_legacy_point_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["table", "--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "usage: virasoro_blocks table" in output
    assert "--point-constraint" not in output


def test_operation_help_uses_info_for_basis_discovery(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["evaluate", "--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--basis BASIS" in output
    assert "--basis {" not in output


def test_module_cli_uses_importable_parallel_workers(tmp_path):
    """The documented module invocation must execute a real spin pool."""
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable, "-m", "virasoro_blocks", "table",
            "--family", "virasoro", "--central-charge", "10",
            "--spin-ranges", "0", "2", "--lambda", "0",
            "--delta-ext", "0.15", "0.15", "0.15", "0.15",
            "--order", "4", "--npoles", "1", "--precision", "25",
            "--q-gate", "off", "--num-threads", "2",
            "-o", str(tmp_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "derivs_0.json").exists()
    assert (tmp_path / "derivs_2.json").exists()
    assert "falling back to serial" not in result.stdout.lower()
    assert "falling back to serial" not in result.stderr.lower()


def test_installed_console_script_dispatches_public_info_operation():
    """The packaged entry point must expose the same operation interface."""
    executable = shutil.which("virasoro_blocks")
    if executable is None:
        pytest.skip("installed virasoro_blocks console script is unavailable")
    result = subprocess.run(
        [executable, "info", "--family", "ramond", "--format", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    assert document["family"] == "ramond"
    assert document["operations"]


def test_torus_character_adapter_matches_backend_table():
    adapter = table_adapter_for("torus-character")
    request = adapter.request(
        central_charge=mp.mpf("2.5"), lambda_order=1, precision=35,
    )
    # the adapter builds at the request's precision, whatever the ambient one
    with mp.workdps(15):
        actual = adapter.build_table(request, spin=0)
    with mp.workdps(35):
        expected = torus_character_table(mp.mpf("2.5"), 0, 1)
    assert actual.numerators == expected.numerators
    assert actual.pole_list_x == expected.pole_list_x


def test_evaluate_chiral_json_matches_python_api():
    with tempfile.NamedTemporaryFile(suffix=".json") as fh:
        main([
            "evaluate", "--family", "virasoro", "--basis", "chiral_F",
            "--central-charge", "2", "--h-ext", "1/8", "1/8", "1/8", "1/8",
            "--coordinate", "z", "--exchange-h", "7/10", "--point", "z=3/10",
            "--order", "12", "--precision", "40", "--format", "json",
            "-o", fh.name,
        ])
        doc = json.loads(fh.read().decode())
    expected = F_block(
        mp.mpf("0.7"), mp.mpf("0.3"), [mp.mpf(1) / 8] * 4,
        c=mp.mpf(2), q_order=12,
    )
    assert abs(mp.mpf(doc["value"]) - expected) < mp.mpf("1e-35")
    assert doc["schema"] == "block-value/v2"
    assert doc["pole_policy"] == "raise"
    assert doc["point"] == {"z": "0.3"}


def test_evaluate_global_scalar_uses_no_q_order():
    with tempfile.NamedTemporaryFile(suffix=".json") as fh:
        main([
            "evaluate", "--family", "global", "--basis", "global_scalar_G",
            "--coordinate", "zzb", "--delta-ext", "1/5", "1/5", "1/5", "1/5",
            "--exchange-delta", "7/5", "--exchange-spin", "0",
            "--point", "z=3/10", "--point", "zbar=2/5",
            "--precision", "40", "-o", fh.name,
        ])
        doc = json.loads(fh.read().decode())
    assert doc["family"] == "global"
    assert doc["basis"] == "global_scalar_G"
    assert doc["order"] is None
    assert isinstance(doc["value"], str)


def test_evaluate_rejects_generic_virasoro_exchanged_pole():
    with pytest.raises(SystemExit):
        main([
            "evaluate", "--family", "virasoro", "--basis", "chiral_F",
            "--central-charge", "2", "--h-ext", "1/8", "1/8", "1/8", "1/8",
            "--coordinate", "z", "--exchange-h=-1/8", "--point", "z=3/10",
            "--order", "8", "--precision", "35", "--format", "text",
        ])


def test_table_subcommand_records_canonical_provenance():
    with tempfile.TemporaryDirectory() as directory:
        main([
            "table", "--central-charge", "10", "--spin-ranges", "0",
            "--lambda", "1", "--delta-ext", "0.15", "0.15", "0.15", "0.15",
            "--order", "6", "--npoles", "2", "--precision", "30",
            "--q-gate", "off", "-o", directory,
        ])
        assert os.path.exists(f"{directory}/derivs_0.json")
        with open(f"{directory}/derivs_0.json") as fh:
            document = json.load(fh)
    assert " table " in f" {document['args']} "


def test_family_selector_is_accepted_by_table_cli():
    with tempfile.TemporaryDirectory() as directory:
        main([
            "--family", "virasoro", "--central-charge", "10", "--spin-ranges", "0",
            "--lambda", "0", "--delta-ext", "0.15", "0.15", "0.15", "0.15",
            "--order", "4", "--npoles", "1", "--precision", "25",
            "--q-gate", "off", "-o", directory,
        ])


def test_table_family_global_maps_short_basis_to_global_backend():
    with tempfile.TemporaryDirectory() as directory:
        main([
            "table", "--family", "global", "--basis", "scalar_G",
            "--spin-ranges", "0", "--lambda", "0",
            "--delta-ext", "0.15", "0.15", "0.15", "0.15",
            "--npoles", "1", "--precision", "25", "-o", directory,
        ])
        with open(f"{directory}/derivs_0.json") as fh:
            doc = json.load(fh)
    assert doc["basis"] == "global_scalar_G"


def test_virasoro_table_adapter_preserves_center_value():
    with tempfile.TemporaryDirectory() as directory:
        main([
            "table", "--family", "virasoro", "--central-charge", "10",
            "--spin-ranges", "0", "--lambda", "0",
            "--delta-ext", "0.15", "0.15", "0.15", "0.15",
            "--order", "8", "--npoles", "all", "--precision", "50",
            "--q-gate", "off", "-o", directory,
        ])
        with open(f"{directory}/derivs_0.json") as fh:
            table = serialized_table_from_json(json.load(fh))
    actual = table.evaluate("zzb", mp.mpf("1.7"))[0][0]
    expected = G_block(
        mp.mpf("1.7"), 0, mp.mpf("0.5"), mp.mpf("0.5"),
        [mp.mpf("0.075")] * 4, c=mp.mpf("10"), q_order=8,
    )
    assert abs(actual - expected) < mp.mpf("1e-40")


def test_multispin_table_workers_are_importable_under_spawn():
    with tempfile.TemporaryDirectory() as directory:
        main([
            "table", "--family", "global", "--basis", "scalar_G",
            "--spin-ranges", "0", "2", "--lambda", "0",
            "--delta-ext", "0.15", "0.15", "0.15", "0.15",
            "--npoles", "1", "--precision", "25", "--num-threads", "2",
            "--q-gate", "off", "-o", directory,
        ])
        assert os.path.exists(f"{directory}/derivs_0.json")
        assert os.path.exists(f"{directory}/derivs_2.json")


def test_table_subcommand_ns_family_writes_json():
    with tempfile.TemporaryDirectory() as directory:
        main([
            "table", "--family", "ns", "--basis", "ns_scalar_G",
            "--central-charge", "1.3", "--spin-ranges", "0", "--lambda", "0",
            "--delta-ext", "0.25", "0.25", "0.25", "0.25",
            "--coordinates", "zzb", "--order", "3", "--npoles", "1",
            "--precision", "25", "--q-gate", "off", "-o", directory,
        ])
        with open(f"{directory}/derivs_0.json") as fh:
            doc = json.load(fh)
    assert doc["basis"] == "ns_scalar_G"
    assert doc["block_family"] == "ns"


@pytest.mark.parametrize(
    "family, basis, branch",
    [
        ("ns", "ns_chiral_F", "++"),
        ("ramond", "ramond_chiral_F", "+-"),
    ],
)
def test_super_table_adapter_builds_without_cli_callback(family, basis, branch):
    """The Python table adapter owns NS/R construction as well as the CLI."""
    request = BlockRequest(
        operation="table",
        surface="sphere",
        family=family,
        basis=basis,
        coordinate="z",
        coordinates=("z",),
        h_ext=(".125",) * 4,
        central_charge="1.3",
        branch=branch,
        policy=NumericalPolicy(
            precision=25,
            order=2,
            lambda_order=0,
            npoles=1,
            pole_selection="contribution",
        ),
    )
    table = build_table_request(request, spin=0)
    assert table.coordinates == ("z",)
    assert table.metadata["block_family"] == family
    if family == "ramond":
        assert table.metadata["ramond_branch"] == branch


def test_table_subcommand_torus_family_aliases_write_json():
    with tempfile.TemporaryDirectory() as directory:
        main([
            "table", "--surface", "torus", "--family", "torus-character",
            "--central-charge", "2.5", "--spin-ranges", "0", "--lambda", "0",
            "--precision", "25", "-o", directory,
        ])
        with open(f"{directory}/derivs_0.json") as fh:
            doc = json.load(fh)
    assert doc["basis"] == "torus_G"


def test_table_subcommand_torus_onepoint_family_alias_writes_json():
    with tempfile.TemporaryDirectory() as directory:
        main([
            "table", "--surface", "torus", "--family", "torus-onepoint",
            "--central-charge", "2.5", "--torus-onept-ext", "0.2",
            "--torus-scalar-shift", "0.1", "--spin-ranges", "0",
            "--lambda", "0", "--order", "3", "--npoles", "1",
            "--precision", "25", "-o", directory,
        ])
        with open(f"{directory}/derivs_0.json") as fh:
            doc = json.load(fh)
    assert doc["basis"] == "torus_onept_G"


def test_point_table_subcommand_writes_frozen_table():
    with tempfile.TemporaryDirectory() as directory:
        main([
            "point-table", "--family", "global", "--basis", "scalar_G",
            "--delta-ext", "1/5", "1/5", "1/5", "1/5",
            "--coordinates", "zzb", "--lambda", "1",
            "--exchange-delta", "7/5", "--exchange-spin", "0",
            "--precision", "30", "--format", "both", "-o", directory,
        ])
        with open(f"{directory}/derivs_point_0.json") as fh:
            doc = json.load(fh)
        mathematica = next(Path(directory).glob("*.m"))
        mathematica_text = mathematica.read_text()
    assert doc["variable"] == "point"
    assert doc["basis"] == "global_scalar_G"
    assert doc["point_value"] == "1.4"
    assert doc["args"].startswith("virasoro_blocks point-table ")
    assert "--point-constraint" not in doc["args"]
    assert 'args -> "virasoro_blocks point-table ' in mathematica_text


def test_point_table_subcommand_translates_virasoro_constraints():
    with tempfile.TemporaryDirectory() as directory:
        main([
            "point-table", "--family", "virasoro", "--basis", "scalar_G",
            "--central-charge", "2", "--delta-ext", "1/8", "1/8", "1/8", "1/8",
            "--coordinates", "zzb", "--lambda", "0", "--order", "4",
            "--exchange-delta", "7/5", "--exchange-spin", "0",
            "--precision", "30", "-o", directory,
        ])
        with open(f"{directory}/derivs_point_0.json") as fh:
            doc = json.load(fh)
    assert doc["variable"] == "point"
    assert doc["basis"] == "scalar_G"
    assert doc["args"].startswith("virasoro_blocks point-table ")
    assert "--point-constraint" not in doc["args"]


def test_ns_direct_evaluator_matches_chiral_backend():
    mp.mp.dps = 35
    request = BlockRequest(
        operation="evaluate", family="ns", basis="ns_chiral_F", coordinate="z",
        central_charge=mp.mpf("1.3"), h_ext=(mp.mpf("0.1"),) * 4,
        exchange_h=mp.mpf("0.7"), point={"z": mp.mpf("0.3")},
        policy=NumericalPolicy(precision=35, order=5),
    )
    actual = evaluate_block_request(request)
    expected = ns_chiral_block(
        mp.mpf("0.7"), mp.mpf("0.3"), [mp.mpf("0.1")] * 4,
        c=mp.mpf("1.3"), q_order=5,
    )
    assert abs(actual.value - expected) < mp.mpf("1e-30")
    assert actual.as_dict()["sector"] == "even"


def test_ramond_direct_evaluator_matches_chiral_backend():
    mp.mp.dps = 35
    request = BlockRequest(
        operation="evaluate", family="ramond", basis="ramond_chiral_F",
        coordinate="z", central_charge=mp.mpf("1.3"),
        h_ext=(mp.mpf("0.1"),) * 4, exchange_h=mp.mpf("0.8"),
        point={"z": mp.mpf("0.3")}, branch="+-",
        policy=NumericalPolicy(precision=35, order=5),
    )
    actual = evaluate_block_request(request)
    expected = ramond_chiral_block(
        mp.mpf("0.8"), mp.mpf("0.3"), [mp.mpf("0.1")] * 4,
        c=mp.mpf("1.3"), q_order=5, branch="+-",
    )
    assert abs(actual.value - expected) < mp.mpf("1e-30")
    assert actual.as_dict()["branch"] == "+-"


def test_torus_character_direct_value_agrees_with_center_table():
    mp.mp.dps = 35
    c = mp.mpf("2.5")
    delta = mp.mpf("1.7")
    request = BlockRequest(
        operation="evaluate", surface="torus", family="torus-character",
        coordinate="vvb", central_charge=c, exchange_delta=delta,
        exchange_spin=0, point={"v": 0, "vbar": 0},
        policy=NumericalPolicy(precision=35, order=None),
    )
    actual = evaluate_block_request(request).value
    table = torus_character_table(c, 0, 0)
    expected = table.evaluate("vvb", delta)[0][0]
    assert abs(actual - expected) < mp.mpf("1e-30")


def test_torus_onepoint_direct_value_agrees_with_center_table():
    mp.mp.dps = 35
    c = mp.mpf("2.5")
    delta_ext = mp.mpf("0.2")
    delta = mp.mpf("1.7")
    kernel = build_torus_onept_kernel(5, d_ext=delta_ext, c=c)
    request = BlockRequest(
        operation="evaluate", surface="torus", family="torus-onepoint",
        coordinate="vvb", central_charge=c, torus_onepoint_ext=delta_ext,
        exchange_delta=delta, exchange_spin=0, point={"v": 0, "vbar": 0},
        policy=NumericalPolicy(precision=35, order=5),
    )
    actual = evaluate_block_request(request).value
    expected = torus_onept_table_values(kernel, 0, 0, delta)[0][0]
    assert abs(actual - expected) < mp.mpf("1e-30")


def test_torus_onepoint_pole_uses_common_public_error():
    mp.mp.dps = 35
    c = mp.mpf("2.5")
    delta_ext = mp.mpf("0.2")
    kernel = build_torus_onept_kernel(5, d_ext=delta_ext, c=c)
    pole = kernel.poles[0]
    request = BlockRequest(
        operation="evaluate", surface="torus", family="torus-onepoint",
        coordinate="vvb", central_charge=c, torus_onepoint_ext=delta_ext,
        exchange_delta=2 * pole, exchange_spin=0,
        point={"v": 0, "vbar": 0},
        policy=NumericalPolicy(precision=35, order=5),
    )
    with pytest.raises(BlockPoleError):
        evaluate_block_request(request)


def test_table_request_builds_at_its_own_precision():
    """build_table_request runs at policy.precision whatever the ambient dps (only the CLI set mp.dps)."""
    from virasoro_blocks.api import BlockRequest as Request, NumericalPolicy as Policy, build_table_request

    def value(ambient):
        with mp.workdps(ambient):
            request = Request(operation="table", family="virasoro", basis="scalar_G", coordinate="zzb",
                              coordinates=("zzb",), h_ext=(mp.mpf(1) / 8,) * 4, hbar_ext=(mp.mpf(1) / 8,) * 4,
                              central_charge=mp.mpf("0.73"),
                              policy=Policy(precision=40, order=10, lambda_order=2, npoles=4))
            table = build_table_request(request, spin=2)
        with mp.workdps(40):
            return table.evaluate("zzb", mp.mpf("2.3"))[0][0]

    assert abs(value(15) / value(40) - 1) < 1e-25


def test_virasoro_evaluate_honours_the_z_shifts_and_integer_spin():
    """The full Virasoro bases dropped --z-shift / --zbar-shift, and scalar_G rounded a non-integer h - hbar."""
    from virasoro_blocks.api import BlockRequest as Request, NumericalPolicy as Policy
    from virasoro_blocks.requests import RequestValidationError

    def request(**kw):
        base = dict(operation="evaluate", family="virasoro", basis="scalar_G", coordinate="zzb",
                    h_ext=(mp.mpf("0.1"),) * 4, hbar_ext=(mp.mpf("0.1"),) * 4, central_charge=mp.mpf("0.7"),
                    exchange_h=mp.mpf("0.65"), exchange_hbar=mp.mpf("0.65"),
                    point={"z": mp.mpf("0.3"), "zbar": mp.mpf("0.2")}, policy=Policy(precision=30, order=12))
        base.update(kw)
        return Request(**base)

    with mp.workdps(30):
        plain = evaluate_block_request(request()).value
        shifted = evaluate_block_request(request(z_shift=mp.mpf("-0.5"), zbar_shift=mp.mpf("-0.5"))).value
        assert abs(shifted / plain - 1 / mp.sqrt(mp.mpf("0.06"))) < 1e-20
        with pytest.raises(RequestValidationError):
            evaluate_block_request(request(exchange_h=mp.mpf("1.3"), exchange_hbar=mp.mpf("0.5")))
