"""IdenticalScalarGapSpec.dps is the working precision; target_digits opts in to verified digits (2026-09-29).

A keep-all (npoles="all") build at the default dps 80 wrote pmp.json polynomials off by 1e23 without a word, and an
integer npoles near the pole count (100 of ~150 at dps 80) by 7e-11.  Every fixed-spin block is now verified
against its numeric sampler (between its nodes and past them to the large-Delta regime), wrong SDP data raise a
RuntimeWarning with the dps that fixes them, and target_digits rebuilds at a raised working precision.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import replace

import mpmath as mp
import pytest

from virasoro_blocks.verified_build import TargetDigitsError
from virasoro_bootstrap import IdenticalScalarGapSpec, build_identical_scalar_gap_problem
from virasoro_bootstrap.identical_scalar import write_identical_scalar_gap_problem
from virasoro_bootstrap.manifest import build_identical_scalar_problem_from_manifest


def _spec(**kw):
    base = dict(c=8, h_ext=mp.mpf(1) / 16, delta_gap=1, lam=3, spin_list=(0,), q_order=20, dps=40, npoles="all",
                num_threads=1)
    base.update(kw)
    return IdenticalScalarGapSpec(**base)


def test_keep_all_build_warns_with_the_dps_that_fixes_it():
    with pytest.warns(RuntimeWarning, match=r"Raise dps to about \d+") as record:
        problem = build_identical_scalar_gap_problem(_spec())
    assert "target_digits=40" in str(record[0].message)
    block = problem.spin_channels[0].serialized
    assert problem.spec.dps == 40 and block.working_dps == 40 and block.verified_digits < 20


def test_integer_npoles_near_all_is_verified_too():
    with pytest.warns(RuntimeWarning, match="the spin-0 block \\(npoles 100\\)"):
        build_identical_scalar_gap_problem(_spec(lam=5, q_order=30, dps=80, npoles=100))


def test_target_digits_escalates_and_round_trips_through_the_manifest(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)          # (the escalated build is not short)
        problem = build_identical_scalar_gap_problem(_spec(target_digits=30))
    block = problem.spin_channels[0].serialized
    assert problem.spec.dps > 40 and problem.spec.target_digits == 30 and problem.requested_dps == 40
    assert block.working_dps == problem.spec.dps and block.verified_digits >= 30
    manifest_path, _pmp = write_identical_scalar_gap_problem(problem, tmp_path / "p")
    doc = json.load(open(manifest_path))
    assert doc["spec"]["target_digits"] == 30 and doc["provenance"]["precision"]["dps_requested"] == 40
    assert doc["provenance"]["precision"]["working_dps"] == problem.spec.dps
    reloaded = build_identical_scalar_problem_from_manifest(manifest_path)
    assert reloaded.spec.target_digits == 30 and reloaded.spec.dps == problem.spec.dps

    # an edited target is honored by the files' verified digits (a lower one), or refused (a higher one)
    doc["spec"]["target_digits"] = 20
    json.dump(doc, open(manifest_path, "w"))
    assert build_identical_scalar_problem_from_manifest(manifest_path).spec.target_digits == 20
    doc["spec"]["target_digits"] = 90
    json.dump(doc, open(manifest_path, "w"))
    with pytest.raises(ValueError, match="short of spec.target_digits 90"):
        build_identical_scalar_problem_from_manifest(manifest_path)


def test_target_digits_refuses_what_precision_cannot_fix():
    with pytest.raises(TargetDigitsError, match="the order reduction .* raise npoles"):
        build_identical_scalar_gap_problem(_spec(npoles=4, target_digits=30))
    with pytest.raises(ValueError, match="max_dps caps the escalation"):
        build_identical_scalar_gap_problem(_spec(max_dps=90))
    with pytest.raises(TargetDigitsError, match="above max_dps=45"):
        build_identical_scalar_gap_problem(_spec(target_digits=30, max_dps=45))


def test_ope_and_moment_builders_assemble_at_the_working_precision():
    """The OPE and moment builders assemble the PMP around the gap build: for a target_digits spec at the raised
    working precision, never at the requested dps (which would round the escalated blocks back)."""
    from virasoro_bootstrap import build_identical_scalar_ope_max_problem
    from virasoro_bootstrap.moments import MomentWeight, build_identical_scalar_moment_upper_problem

    spec = _spec(target_digits=30)
    ope = build_identical_scalar_ope_max_problem(spec)
    assert ope.spec.dps > 40 and ope.spin_channels[0].serialized.verified_digits >= 30
    moment = build_identical_scalar_moment_upper_problem(spec, MomentWeight.monomial(Delta=1))
    assert moment.spec.dps == ope.spec.dps


def test_prepared_build_serves_specs_of_the_same_target():
    from virasoro_bootstrap.identical_scalar import prepare_identical_scalar_gap_build

    spec = _spec(target_digits=30)
    prepared = prepare_identical_scalar_gap_build(spec)
    assert prepared.working_dps > 40
    moved = build_identical_scalar_gap_problem(replace(spec, delta_gap=mp.mpf("1.5")), prepared=prepared)
    assert moved.spec.dps == prepared.working_dps
