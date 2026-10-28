"""Tests for the file-backed identical-scalar bootstrap manifest."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import mpmath as mp
import pytest

from virasoro_bootstrap.identical_scalar import (
    IdenticalScalarGapSpec,
    build_identical_scalar_gap_problem,
    build_identical_scalar_ope_max_problem,
)
from virasoro_bootstrap.manifest import (
    BlockRef,
    BootstrapManifest,
    CrossingSpec,
    IdentityVector,
    PmpVectors,
    block_content_hash,
    build_identical_scalar_problem_from_manifest,
    load_identical_scalar_manifest,
    write_identical_scalar_manifest,
)


def _small_spec(**overrides) -> IdenticalScalarGapSpec:
    # Set the working precision before building the mpf-valued fields so the
    # spec carries full-precision numbers (not double-precision literals).
    mp.mp.dps = overrides.get("dps", 60)
    base = dict(
        c=mp.mpf("10"),
        h_ext=mp.mpf("0.15"),
        delta_gap=mp.mpf("1.0"),
        lam=3,
        spin_list=(0, 2),
        q_order=8,
        dps=60,
        npoles=4,
        n_sample_points=9,
        num_threads=1,
    )
    base.update(overrides)
    # Re-coerce the numeric fields through string form at the now-active dps so
    # values passed by the caller (possibly a string or a literal built at a
    # lower ambient dps) are stored at full precision.
    for field in ("c", "h_ext", "delta_gap"):
        base[field] = mp.mpf(str(base[field]))
    return IdenticalScalarGapSpec(**base)


def _max_rel_diff(a, b):
    """Recursively compare two PMP JSON structures, returning the max relative
    difference over every numeric leaf (and asserting structural equality)."""
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (sorted(a.keys()), sorted(b.keys()))
        return max((_max_rel_diff(a[k], b[k]) for k in a), default=mp.mpf(0))
    if isinstance(a, list):
        assert len(a) == len(b), (len(a), len(b))
        return max((_max_rel_diff(x, y) for x, y in zip(a, b)), default=mp.mpf(0))
    try:
        za, zb = mp.mpf(a), mp.mpf(b)
    except (TypeError, ValueError):
        assert a == b, (a, b)
        return mp.mpf(0)
    scale = max(abs(za), abs(zb), mp.mpf(1))
    return abs(za - zb) / scale


def test_manifest_round_trip_reproduces_pmp(tmp_path):
    spec = _small_spec()
    mp.mp.dps = spec.dps
    problem = build_identical_scalar_gap_problem(spec)

    manifest_path = write_identical_scalar_manifest(problem, tmp_path)
    assert manifest_path == tmp_path / "manifest.json"
    assert manifest_path.exists()

    rebuilt = build_identical_scalar_problem_from_manifest(manifest_path)

    tol = mp.mpf(10) ** (-(spec.dps - 6))
    max_diff = _max_rel_diff(problem.pmp_build.doc, rebuilt.pmp_build.doc)
    assert max_diff <= tol, f"max relative PMP diff {mp.nstr(max_diff, 5)} > {tol}"
    # the SDP data at the nodes are the files' numeric values, not the file polynomials re-evaluated
    assert rebuilt.sample_values_by_block is not None
    assert rebuilt.sample_values_by_block.keys() == problem.sample_values_by_block.keys()
    for index, rows in problem.sample_values_by_block.items():
        for row, row_back in zip(rows, rebuilt.sample_values_by_block[index]):
            assert all(abs(mp.mpc(x) - mp.mpc(y)) <= tol * max(1, abs(mp.mpc(x))) for x, y in zip(row, row_back))


def test_manifest_ope_max_round_trip_reproduces_swapped_pmp(tmp_path):
    """An OPE-max problem round-trips through the manifest: the schema-1.1 `pmp`
    block records objective = V_id and normalization = V_* (the swapped roles),
    plus the normalization operator's (spin_star, delta_star), so the
    reconstructed PMP reproduces the extremal problem -- not the gap convention."""
    spec = _small_spec()
    mp.mp.dps = spec.dps
    problem = build_identical_scalar_ope_max_problem(spec, delta_star=mp.mpf("1.3"), spin_star=0)
    assert problem.problem_kind == "ope_max"

    manifest_path = write_identical_scalar_manifest(problem, tmp_path)
    rebuilt = build_identical_scalar_problem_from_manifest(manifest_path)

    # The swapped roles survive the round trip.
    assert rebuilt.problem_kind == "ope_max"
    assert rebuilt.normalization_operator is not None
    assert int(rebuilt.normalization_operator["spin_star"]) == 0
    assert abs(mp.mpf(rebuilt.normalization_operator["delta_star"]) - mp.mpf("1.3")) < mp.mpf(10) ** -30
    # objective is the identity vector; normalization is the distinguished V_*.
    tol = mp.mpf(10) ** (-(spec.dps - 6))
    obj_diff = max(abs(mp.mpf(a) - mp.mpf(b))
                   for a, b in zip(rebuilt.objective_vector, problem.identity_vector))
    assert obj_diff <= tol
    # The full reconstructed PMP matches (so SDPB would solve the same extremal SDP).
    max_diff = _max_rel_diff(problem.pmp_build.doc, rebuilt.pmp_build.doc)
    assert max_diff <= tol, f"max relative PMP diff {mp.nstr(max_diff, 5)} > {tol}"


def test_manifest_full_method_round_trip_reproduces_pmp(tmp_path):
    spec = _small_spec(npoles="all", n_sample_points=None)
    mp.mp.dps = spec.dps
    problem = build_identical_scalar_gap_problem(spec)

    manifest_path = write_identical_scalar_manifest(problem, tmp_path)
    rebuilt = build_identical_scalar_problem_from_manifest(manifest_path)

    tol = mp.mpf(10) ** (-(spec.dps - 6))
    max_diff = _max_rel_diff(problem.pmp_build.doc, rebuilt.pmp_build.doc)
    assert max_diff <= tol, f"max relative PMP diff {mp.nstr(max_diff, 5)} > {tol}"


def test_manifest_large_spin_tower_round_trip_reproduces_pmp(tmp_path):
    """A spec carrying a double-twist large-spin tower round-trips through the
    manifest: tower channels are persisted as ``variable="L"`` files keyed by
    twist index (no fixed-spin filename collision) and the reload reproduces the
    full PMP, including the appended tower channels."""
    from virasoro_bootstrap.large_spin import double_twist_set

    h_ext = mp.mpf("0.0625")
    spec = _small_spec(
        h_ext=h_ext,
        large_spin_lmin=4,
        large_spin_twists=tuple(double_twist_set(h_ext, 3)),
    )
    mp.mp.dps = spec.dps
    problem = build_identical_scalar_gap_problem(spec)
    # Two fixed-spin channels (0, 2) plus three tower channels.
    assert len(problem.spin_channels) == 5

    manifest_path = write_identical_scalar_manifest(problem, tmp_path)
    manifest = load_identical_scalar_manifest(manifest_path)

    fixed_refs = [r for r in manifest.blocks if r.kind == "fixed_spin"]
    tower_refs = [r for r in manifest.blocks if r.kind == "large_spin"]
    assert {r.ell for r in fixed_refs} == {0, 2}
    assert len(tower_refs) == 3
    # Tower refs are keyed by distinct files and carry twist/Lmax metadata.
    assert len({r.path for r in tower_refs}) == 3
    for ref in tower_refs:
        assert ref.variable == "L"
        assert ref.twist is not None
        assert ref.large_spin_lmin is not None
        block_doc = json.loads((tmp_path / ref.path).read_text())
        assert block_doc["variable"] == "L"
    recorded_twists = sorted(mp.mpf(r.twist) for r in tower_refs)
    expected_twists = sorted(double_twist_set(h_ext, 3))
    for got, want in zip(recorded_twists, expected_twists):
        assert abs(got - want) < mp.mpf("1e-40")

    rebuilt = build_identical_scalar_problem_from_manifest(manifest_path)
    assert len(rebuilt.spin_channels) == 5
    assert rebuilt.spec.large_spin_lmin is not None
    assert len(rebuilt.spec.large_spin_twists) == 3

    tol = mp.mpf(10) ** (-(spec.dps - 6))
    max_diff = _max_rel_diff(problem.pmp_build.doc, rebuilt.pmp_build.doc)
    assert max_diff <= tol, f"max relative PMP diff {mp.nstr(max_diff, 5)} > {tol}"


def test_manifest_point_constraint_round_trip_reproduces_pmp(tmp_path):
    """A spec carrying point constraints round-trips through the manifest: each
    point channel is persisted as a ``variable="point"`` file keyed by point
    index (no fixed-spin filename collision) and the reload reproduces the full
    PMP, including the appended degree-0 point channels."""
    spec = _small_spec(
        point_constraints=((0, mp.mpf("0.7")), (2, mp.mpf("3.0"))),
    )
    mp.mp.dps = spec.dps
    problem = build_identical_scalar_gap_problem(spec)
    # Two fixed-spin channels (0, 2) plus two point channels.
    assert len(problem.spin_channels) == 4
    assert len(problem.point_channels) == 2

    manifest_path = write_identical_scalar_manifest(problem, tmp_path)
    manifest = load_identical_scalar_manifest(manifest_path)

    fixed_refs = [r for r in manifest.blocks if r.kind == "fixed_spin"]
    point_refs = [r for r in manifest.blocks if r.kind == "point"]
    assert {r.ell for r in fixed_refs} == {0, 2}
    assert len(point_refs) == 2
    # Point refs are keyed by distinct files and carry the frozen point_value.
    assert len({r.path for r in point_refs}) == 2
    for ref in point_refs:
        assert ref.variable == "point"
        assert ref.point_value is not None
        block_doc = json.loads((tmp_path / ref.path).read_text())
        assert block_doc["variable"] == "point"
        # A point block is degree 0 in its positivity variable.
        assert len(block_doc["derivs"][0][0][0]) == 1
    # (the input mpf was built at the ambient dps, so compare at double tol)
    recorded = sorted((r.ell, mp.mpf(r.point_value)) for r in point_refs)
    assert recorded[0][0] == 0 and abs(recorded[0][1] - mp.mpf("0.7")) < mp.mpf("1e-12")
    assert recorded[1][0] == 2 and abs(recorded[1][1] - mp.mpf("3.0")) < mp.mpf("1e-12")

    rebuilt = build_identical_scalar_problem_from_manifest(manifest_path)
    assert len(rebuilt.spin_channels) == 4
    assert len(rebuilt.point_channels) == 2
    assert len(rebuilt.spec.point_constraints) == 2

    tol = mp.mpf(10) ** (-(spec.dps - 6))
    max_diff = _max_rel_diff(problem.pmp_build.doc, rebuilt.pmp_build.doc)
    assert max_diff <= tol, f"max relative PMP diff {mp.nstr(max_diff, 5)} > {tol}"


def test_manifest_records_delta_min_on_refs_not_in_block_file(tmp_path):
    spec = _small_spec(delta_gap="1.3")
    mp.mp.dps = spec.dps
    problem = build_identical_scalar_gap_problem(spec)
    manifest_path = write_identical_scalar_manifest(problem, tmp_path)

    manifest = load_identical_scalar_manifest(manifest_path)
    refs = {ref.ell: ref for ref in manifest.blocks}
    assert set(refs) == {0, 2}
    # Scalar gap edge is the requested delta_gap; spin-2 edge is the unitarity
    # bound 2. Both live on the BlockRef, never inside the block file.
    assert abs(mp.mpf(refs[0].delta_min) - mp.mpf("1.3")) < mp.mpf("1e-40")
    assert abs(mp.mpf(refs[2].delta_min) - mp.mpf("2.0")) < mp.mpf("1e-40")
    for ref in manifest.blocks:
        block_doc = json.loads((tmp_path / ref.path).read_text())
        assert "delta_min" not in block_doc
        assert block_doc["variable"] == "Delta"


def test_manifest_same_block_file_reused_across_gap_values(tmp_path):
    """The spin-2 block file is gap-independent: writing manifests at two
    different scalar gaps produces an identical spin-2 block payload (same
    content hash)."""
    spec_a = _small_spec(delta_gap="1.0")
    spec_b = _small_spec(delta_gap="1.5")
    mp.mp.dps = spec_a.dps

    problem_a = build_identical_scalar_gap_problem(spec_a)
    problem_b = build_identical_scalar_gap_problem(spec_b)

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    manifest_a = load_identical_scalar_manifest(
        write_identical_scalar_manifest(problem_a, dir_a)
    )
    manifest_b = load_identical_scalar_manifest(
        write_identical_scalar_manifest(problem_b, dir_b)
    )

    hashes_a = {ref.ell: ref.content_hash for ref in manifest_a.blocks}
    hashes_b = {ref.ell: ref.content_hash for ref in manifest_b.blocks}
    # Spin-2 channel is unaffected by the scalar gap -> identical block payload.
    assert hashes_a[2] == hashes_b[2]
    # The scalar channel's lower edge moved, so its block table (anchored at the
    # new shift) differs.
    assert hashes_a[0] != hashes_b[0]


@pytest.fixture(scope="module")
def written_manifest(tmp_path_factory):
    """One small gap problem's manifest directory, written once per module; the
    corruption tests below each mutate a private copy of it."""
    spec = _small_spec()
    mp.mp.dps = spec.dps
    problem = build_identical_scalar_gap_problem(spec)
    return write_identical_scalar_manifest(problem, tmp_path_factory.mktemp("manifest"))


def _copy_manifest(manifest_path, tmp_path):
    dest = tmp_path / "manifest_copy"
    shutil.copytree(manifest_path.parent, dest)
    return dest / manifest_path.name


@pytest.mark.parametrize("change", ["base", "remove_samples"])
def test_manifest_content_hash_mismatch_is_detected(written_manifest, tmp_path, change):
    manifest_path = _copy_manifest(written_manifest, tmp_path)

    # Corrupt one referenced block file's payload.
    manifest_doc = json.loads(manifest_path.read_text())
    victim_rel = manifest_doc["blocks"][0]["path"]
    victim = manifest_path.parent / victim_rel
    block_doc = json.loads(victim.read_text())
    if change == "base":
        block_doc["exp_base"] = "0.123456789"
    elif change == "sample_value":
        block_doc["sample_block_values"]["zzb"][0][0][0] = "-123456789"
    elif change == "remove_samples":
        del block_doc["sample_block_values"]
    else:
        del block_doc["sample_block_values_index_names"]
    victim.write_text(json.dumps(block_doc))

    with pytest.raises(ValueError, match="content_hash mismatch"):
        load_identical_scalar_manifest(manifest_path)


def test_manifest_accepts_legacy_hashes_that_omit_sample_values(written_manifest, tmp_path):
    # manifests written before 2026-10-03 hashed the table without the stored sample values
    manifest_path = _copy_manifest(written_manifest, tmp_path)
    document = json.loads(manifest_path.read_text())
    old_fields = ("exp_base", "variable_minus_x", "variable", "pole_list_x",
                  "coordinates", "derivs", "sample_points_x")
    for ref in document["blocks"]:
        block = json.loads((manifest_path.parent / ref["path"]).read_text())
        assert block["sample_block_values"]
        payload = {key: block[key] for key in old_fields if key in block}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        ref["content_hash"] = "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    manifest_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="content_hash mismatch"):   # a current (1.2) manifest
        load_identical_scalar_manifest(manifest_path)
    document["schema_version"] = "1.1"
    manifest_path.write_text(json.dumps(document))
    load_identical_scalar_manifest(manifest_path)
    block = json.loads((manifest_path.parent / document["blocks"][0]["path"]).read_text())
    block["exp_base"] = "0.123456789"   # a payload field both hashes cover
    (manifest_path.parent / document["blocks"][0]["path"]).write_text(json.dumps(block))
    with pytest.raises(ValueError, match="content_hash mismatch"):
        load_identical_scalar_manifest(manifest_path)


def test_manifest_reconstruction_rechecks_consumed_blocks(tmp_path, monkeypatch):
    import virasoro_bootstrap.manifest as manifests
    from virasoro_bootstrap.controls.chiral_control import build_chiral_control_problem
    from virasoro_bootstrap.moments import MomentWeight, build_identical_scalar_moment_lower_problem

    spec = IdenticalScalarGapSpec(c=4, h_ext='.1', delta_gap=2, lam=1, spin_list=(0,),
        q_order=2, dps=30, npoles='all', num_threads=1,
        large_spin_lmin=2, large_spin_twists=('.3',), point_constraints=((0, 1),))
    gap = build_identical_scalar_gap_problem(spec)
    chiral = build_chiral_control_problem(mode='global', c=4, h_ext='.1', h_gap=1,
                                         lam=1, q_order=2, dps=30)
    weight = MomentWeight.monomial(Delta=1)
    problems = (gap, build_identical_scalar_moment_lower_problem(chiral.spec, weight, base_problem=chiral))
    original_loader = manifests.load_identical_scalar_manifest
    for index, problem in enumerate(problems):
        path = write_identical_scalar_manifest(problem, tmp_path / str(index))
        original_manifest = path.read_text()
        refs = json.loads(original_manifest)['blocks']
        block_files = {ref['path']: (path.parent / ref['path']).read_text() for ref in refs}
        expected = build_identical_scalar_problem_from_manifest(path)
        with mp.workdps(30):
            assert _max_rel_diff(expected.pmp_build.doc, problem.pmp_build.doc) < mp.mpf('1e-24')
        # Both a replaced payload and physical labels omitted from its hash
        # must be checked on the document actually deserialized. Updating the
        # on-disk manifest too cannot replace the already verified snapshot.
        changes = [(ref, 'exp_base', '0.123456789') for ref in refs]
        first = refs[0]
        changes.append((first, 'spin', 2))
        for ref, field, value in changes:
            def replace_after_check(manifest_path):
                manifest = original_loader(manifest_path)
                victim = path.parent / ref['path']
                block = json.loads(victim.read_text())
                block[field] = value
                victim.write_text(json.dumps(block))
                replacement = json.loads(original_manifest)
                for stored in replacement['blocks']:
                    if stored['path'] == ref['path']:
                        stored['content_hash'] = block_content_hash(block)
                path.write_text(json.dumps(replacement))
                return manifest

            with monkeypatch.context() as patch:
                patch.setattr(manifests, 'load_identical_scalar_manifest', replace_after_check)
                with pytest.raises(ValueError, match='content_hash mismatch|spin'):
                    build_identical_scalar_problem_from_manifest(path)
            path.write_text(original_manifest)
            for relative, content in block_files.items():
                (path.parent / relative).write_text(content)


def test_manifest_regularization_matches_frozen_blocks_and_identity(tmp_path):
    from virasoro_bootstrap.moments import MomentWeight, build_identical_scalar_moment_lower_problem

    for c, equivalent, changed in ((25, 'on', 'off'), (4, 'off', 'on')):
        spec = IdenticalScalarGapSpec(c=c, h_ext='.125', delta_gap=3, lam=1, spin_list=(0,),
            q_order=4, dps=30, npoles=2, num_threads=1)
        gap = build_identical_scalar_gap_problem(spec)
        moment = build_identical_scalar_moment_lower_problem(spec, MomentWeight.monomial(Delta=1), base_problem=gap)
        for problem in (gap, moment):
            path = write_identical_scalar_manifest(problem, tmp_path / str(c) / problem.problem_kind)
            document = json.loads(path.read_text())
            document['spec']['regularize'] = equivalent
            path.write_text(json.dumps(document))
            restored = build_identical_scalar_problem_from_manifest(path)
            assert restored.used_regularize == problem.used_regularize
            with mp.workdps(30):
                assert _max_rel_diff(restored.pmp_build.doc, problem.pmp_build.doc) < mp.mpf('1e-24')
            document['spec']['regularize'] = changed
            path.write_text(json.dumps(document))
            with pytest.raises(ValueError, match='spec.regularize.*recorded'):
                build_identical_scalar_problem_from_manifest(path)
            document['spec']['regularize'] = equivalent
            for malformed in ('true',):
                document['provenance']['used_regularize'] = malformed
                path.write_text(json.dumps(document))
                with pytest.raises(ValueError, match='used_regularize.*boolean'):
                    build_identical_scalar_problem_from_manifest(path)


def test_manifest_rejects_unknown_kind(written_manifest, tmp_path):
    manifest_path = _copy_manifest(written_manifest, tmp_path)

    doc = json.loads(manifest_path.read_text())
    doc["manifest_kind"] = "something_else"
    manifest_path.write_text(json.dumps(doc))

    with pytest.raises(ValueError, match="unexpected manifest_kind"):
        load_identical_scalar_manifest(manifest_path)


@pytest.mark.parametrize("verified", ["nan"])
def test_manifest_nan_verification_cannot_satisfy_target(written_manifest, tmp_path, verified):
    path = _copy_manifest(written_manifest, tmp_path)
    doc = json.loads(path.read_text())
    doc["spec"]["target_digits"] = 1
    block_path = path.parent / doc["blocks"][0]["path"]
    block = json.loads(block_path.read_text())
    block["verified_digits"] = verified
    block_path.write_text(json.dumps(block))
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="verified_digits|target_digits"):
        build_identical_scalar_problem_from_manifest(path)


def test_legacy_manifest_reads_rounded_labels_as_their_rationals(written_manifest, tmp_path):
    # writers before schema 1.2 printed c, h_ext and the prefactor as mpf values at dps + 5 digits
    from fractions import Fraction
    from virasoro_bootstrap.manifest import _spec_from_json

    path = _copy_manifest(written_manifest, tmp_path)
    doc = json.loads(path.read_text())
    dps = doc["spec"]["dps"]
    with mp.workdps(dps):
        rounded = mp.nstr(mp.mpf(3) / 20, dps + 5)
    assert Fraction(rounded) != Fraction(3, 20)
    doc["schema_version"] = "1.1"
    doc["spec"]["h_ext"] = doc["spec"]["prefactor_exponent"] = doc["crossing"]["prefactor_exponent"] = rounded
    path.write_text(json.dumps(doc))
    assert build_identical_scalar_problem_from_manifest(path).spec.h_ext == Fraction(3, 20)
    # an old c < 1 manifest keeps its singular minimal-model locus (c = 6/7, M(6,7))
    with mp.workdps(160):
        old_c = dict(doc["spec"], c=mp.nstr(mp.mpf(6) / 7, 165), dps=160)
    assert _spec_from_json(old_c, legacy=True).c == Fraction(6, 7)
    assert _spec_from_json(old_c).c != Fraction(6, 7)


def test_legacy_global_control_manifest_accepts_its_scalar_G_block_label(tmp_path):
    from virasoro_bootstrap.controls.global_control import build_global_identical_scalar_control_problem

    spec = IdenticalScalarGapSpec(c=4, h_ext='.1', delta_gap=2, lam=1, spin_list=(0,),
                                  q_order=2, dps=30, npoles='all', num_threads=1)
    path = write_identical_scalar_manifest(build_global_identical_scalar_control_problem(spec), tmp_path)
    doc = json.loads(path.read_text())
    for ref in doc["blocks"]:   # older writers labeled every block scalar_G
        block_path = path.parent / ref["path"]
        block_path.write_text(json.dumps(dict(json.loads(block_path.read_text()), basis="scalar_G")))
    with pytest.raises(ValueError, match="basis"):
        build_identical_scalar_problem_from_manifest(path)
    path.write_text(json.dumps(dict(doc, schema_version="1.1")))
    assert build_identical_scalar_problem_from_manifest(path).resolved_method == "global_control"


def test_manifest_rejects_tiny_but_deliberate_c_edit(written_manifest, tmp_path):
    path = _copy_manifest(written_manifest, tmp_path)
    doc = json.loads(path.read_text())
    doc["spec"]["c"] = "10.000000000000000000001"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="central_charge"):
        build_identical_scalar_problem_from_manifest(path)


@pytest.mark.parametrize("change", ["fractional_spin", "duplicate_spin", "swapped_spins"])
def test_manifest_rejects_inconsistent_block_references(written_manifest, tmp_path, change):
    path = _copy_manifest(written_manifest, tmp_path)
    doc = json.loads(path.read_text())
    refs = doc["blocks"]
    if change == "fractional_spin":
        refs[0]["ell"] = 0.5
    elif change == "unknown_kind":
        refs[0]["kind"] = "fixed_spinn"
    elif change == "wrong_variable":
        refs[0]["variable"] = "L"
    elif change == "duplicate_spin":
        refs.append(dict(refs[0]))
    else:
        refs[0]["ell"], refs[1]["ell"] = refs[1]["ell"], refs[0]["ell"]
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="(ell|spin|kind|variable)"):
        load_identical_scalar_manifest(path)


def test_manifest_cannot_silently_drop_every_tower(tmp_path):
    spec = _small_spec(large_spin_lmin=4, large_spin_twists=("0.25",))
    problem = build_identical_scalar_gap_problem(spec)
    path = write_identical_scalar_manifest(problem, tmp_path)
    doc = json.loads(path.read_text())
    doc["blocks"] = [ref for ref in doc["blocks"] if ref["kind"] != "large_spin"]
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="tower block files"):
        build_identical_scalar_problem_from_manifest(path)


@pytest.mark.parametrize("field, value", [("coordinate", "yyb"), ("crossing_pairs", [[3, 0], [1, 0]])])
def test_manifest_rejects_ignored_crossing_edits(written_manifest, tmp_path, field, value):
    path = _copy_manifest(written_manifest, tmp_path)
    doc = json.loads(path.read_text())
    doc["crossing"][field] = value
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="crossing"):
        build_identical_scalar_problem_from_manifest(path)


def test_manifest_block_reference_rejects_numpy_boolean_integer_labels():
    import numpy as np

    doc = dict(path='blocks/derivs_2.json', ell=2, variable='Delta', delta_min='2',
               content_hash='sha256:unused', npoles=2)
    for field in ('ell', 'npoles'):
        for value in (True, np.bool_(False)):
            with pytest.raises(ValueError, match=field + '.*integer'):
                BlockRef.from_json({**doc, field: value})
        for value in ('2', np.int64(2)):
            assert getattr(BlockRef.from_json({**doc, field: value}), field) == 2


def test_manifest_refuses_fractional_point_spin(written_manifest, tmp_path):
    path = _copy_manifest(written_manifest, tmp_path)
    doc = json.loads(path.read_text())
    doc["spec"]["point_constraints"] = [[0.5, "0.7"]]
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="point constraint spin.*integer"):
        build_identical_scalar_problem_from_manifest(path)


@pytest.mark.parametrize("change", ["renamed_files", "removed_reference"])
def test_moment_reload_uses_only_manifest_references(tmp_path, change):
    from virasoro_bootstrap.moments import MomentTerm, MomentWeight, build_identical_scalar_moment_lower_problem

    weight = MomentWeight((MomentTerm("0.123456789012345678901234567890123456789", {"Delta": 1}),))
    problem = build_identical_scalar_moment_lower_problem(_small_spec(), weight)
    path = write_identical_scalar_manifest(problem, tmp_path)
    doc = json.loads(path.read_text())
    if change == "renamed_files":
        expected = build_identical_scalar_problem_from_manifest(path)
        for index, ref in enumerate(doc["blocks"]):
            new_path = f"blocks/channel_{index}.json"
            (tmp_path / ref["path"]).rename(tmp_path / new_path)
            ref["path"] = new_path
        path.write_text(json.dumps(doc))
        with mp.workdps(15):
            reloaded = build_identical_scalar_problem_from_manifest(path)
        assert reloaded.pmp_build.doc == expected.pmp_build.doc
    else:
        doc["blocks"].pop(0)
        path.write_text(json.dumps(doc))
        with pytest.raises(ValueError, match="block file.*spin"):
            build_identical_scalar_problem_from_manifest(path)


def test_manifest_rejects_unsupported_major_version(written_manifest, tmp_path):
    manifest_path = _copy_manifest(written_manifest, tmp_path)

    doc = json.loads(manifest_path.read_text())
    doc["schema_version"] = "99.0"
    manifest_path.write_text(json.dumps(doc))

    with pytest.raises(ValueError, match="unsupported manifest schema_version"):
        load_identical_scalar_manifest(manifest_path)


def test_manifest_rejects_absolute_block_path(written_manifest, tmp_path):
    manifest_path = _copy_manifest(written_manifest, tmp_path)

    doc = json.loads(manifest_path.read_text())
    doc["blocks"][0]["path"] = "/etc/passwd"
    manifest_path.write_text(json.dumps(doc))

    with pytest.raises(ValueError, match="must be relative"):
        load_identical_scalar_manifest(manifest_path)


def test_block_content_hash_excludes_cosmetic_metadata():
    doc = {
        "exp_base": "0.5",
        "variable_minus_x": "0.0",
        "variable": "Delta",
        "pole_list_x": [],
        "coordinates": ["zzb"],
        "derivs": [[[["1.0"]]]],
    }
    base = block_content_hash(doc)
    with_meta = dict(doc, args="--foo", output_template="x", precision_actual=80)
    assert block_content_hash(with_meta) == base
    changed = dict(doc, exp_base="0.6")
    assert block_content_hash(changed) != base


def test_manifest_dataclass_json_round_trip():
    manifest = BootstrapManifest(
        schema_version="1.0",
        manifest_kind="identical_scalar_gap",
        precision_dps=60,
        spec={"c": "10.0"},
        crossing=CrossingSpec(
            sector_kind="identical_scalar_F_minus",
            coordinate="zzb",
            crossing_pairs=((1, 0), (2, 1)),
            prefactor_exponent="0.15",
        ),
        identity=IdentityVector(crossing_vector=("1.0", "2.0")),
        blocks=(
            BlockRef(
                path="blocks/derivs_0.json",
                ell=0,
                variable="Delta",
                delta_min="1.0",
                content_hash="sha256:abc",
            ),
        ),
        provenance={"resolved_method": "interpolation"},
        pmp=PmpVectors(
            problem_kind="ope_max",
            objective=("1.0", "2.0"),
            normalization=("0.3", "0.4"),
            normalization_operator={"spin_star": 0, "delta_star": "1.25"},
        ),
    )
    reloaded = BootstrapManifest.from_json(json.loads(json.dumps(manifest.to_json())))
    assert reloaded == manifest


def test_manifest_schema_1_0_backward_compat_loads_as_gap():
    """A schema-1.0 manifest (identity.functional + stage, no pmp) still loads:
    functional maps to crossing_vector, stage is ignored, pmp is None."""
    doc = {
        "schema_version": "1.0",
        "manifest_kind": "identical_scalar_gap",
        "precision_dps": 60,
        "spec": {"c": "10.0"},
        "crossing": {
            "sector_kind": "identical_scalar_F_minus",
            "coordinate": "zzb",
            "crossing_pairs": [[1, 0], [2, 1]],
            "prefactor_exponent": "0.15",
        },
        "identity": {"stage": "crossing_functional", "functional": ["1.0", "2.0"]},
        "blocks": [
            {
                "path": "blocks/derivs_0.json",
                "ell": 0,
                "variable": "Delta",
                "delta_min": "1.0",
                "content_hash": "sha256:abc",
            }
        ],
        "provenance": {"resolved_method": "interpolation"},
    }
    m = BootstrapManifest.from_json(doc)
    assert m.identity.crossing_vector == ("1.0", "2.0")
    assert m.pmp is None
