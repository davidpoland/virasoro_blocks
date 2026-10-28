"""Block files and manifests record the spin-0 scalar_G normalization, and the bootstrap refuses a file without it."""

from __future__ import annotations

import dataclasses
import json
import shutil

import mpmath as mp
import pytest

from virasoro_blocks._conventions import SCALAR_G_SPIN0_SINGLE
from virasoro_bootstrap.identical_scalar import (
    IdenticalScalarGapSpec,
    build_identical_scalar_gap_problem,
    prepare_identical_scalar_gap_build_from_blocks,
)
from virasoro_bootstrap.manifest import (
    MANIFEST_SCHEMA_VERSION,
    build_identical_scalar_problem_from_manifest,
    write_identical_scalar_manifest,
)

pytestmark = pytest.mark.bootstrap


def _spec(**changes):
    mp.mp.dps = 60
    spec = IdenticalScalarGapSpec(
        c=mp.mpf(4), h_ext=mp.mpf("0.0625"), delta_gap=mp.mpf("0.5"), lam=3, spin_list=(0,),
        q_order=10, dps=60, npoles=4, num_threads=1,
    )
    return dataclasses.replace(spec, **changes)


@pytest.fixture(scope="module")
def spin0_file(tmp_path_factory):
    """The spec's spin-0 table, written by the CLI."""
    from virasoro_blocks.__main__ import main as blocks_main

    out = tmp_path_factory.mktemp("spin0")
    saved = mp.mp.dps
    try:
        blocks_main(["table", "--family", "virasoro", "--basis", "scalar_G", "--central-charge", "4",
                     "--delta-ext", *["0.125"] * 4, "--coordinates", "zzb", "--lambda", "3", "--order", "10",
                     "--npoles", "4", "--precision", "60", "--format", "json", "--spin-ranges", "0",
                     "--gap", "0.5", "--u-shift", "-1/8", "-o", str(out)])
    finally:
        mp.mp.dps = saved
    (path,) = out.glob("derivs_*.json")
    return path


def _variant(source, tmp_path, tag):
    doc = json.loads(source.read_text())
    if tag is None:
        doc.pop("scalar_G_spin0")
    else:
        doc["scalar_G_spin0"] = tag
    directory = tmp_path / "blocks"
    directory.mkdir()
    (directory / source.name).write_text(json.dumps(doc))
    return directory


def test_cli_spin0_file_is_tagged_and_accepted(spin0_file):
    assert json.loads(spin0_file.read_text())["scalar_G_spin0"] == SCALAR_G_SPIN0_SINGLE
    prepare_identical_scalar_gap_build_from_blocks(_spec(), [spin0_file])


@pytest.mark.parametrize("tag, message", [
    (None, "records no scalar_G_spin0"),
])
def test_untagged_or_mismatched_spin0_file_is_refused(spin0_file, tmp_path, tag, message):
    blocks = _variant(spin0_file, tmp_path, tag)
    with pytest.raises(ValueError, match=message) as info:
        prepare_identical_scalar_gap_build_from_blocks(_spec(), blocks)
    assert "regenerate" in str(info.value)




@pytest.fixture(scope="module")
def manifest_path(tmp_path_factory):
    with mp.workdps(60):
        problem = build_identical_scalar_gap_problem(_spec())
    return write_identical_scalar_manifest(problem, tmp_path_factory.mktemp("manifest"))


def test_manifest_records_the_convention(manifest_path):
    doc = json.loads(manifest_path.read_text())
    assert doc["schema_version"] == MANIFEST_SCHEMA_VERSION == "1.3"
    assert doc["scalar_G_spin0"] == SCALAR_G_SPIN0_SINGLE
    build_identical_scalar_problem_from_manifest(manifest_path)


