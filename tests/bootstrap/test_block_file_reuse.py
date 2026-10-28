"""Reusing block JSON files (a manifest's ``blocks/`` or CLI output) for modified problems."""

from __future__ import annotations

import dataclasses
import json

import mpmath as mp
import pytest

from virasoro_bootstrap.identical_scalar import (
    IdenticalScalarGapSpec,
    build_identical_scalar_gap_problem,
    build_identical_scalar_ope_max_problem,
    prepare_identical_scalar_gap_build,
    prepare_identical_scalar_gap_build_from_blocks,
    write_identical_scalar_gap_problem,
)
from virasoro_bootstrap.manifest import build_identical_scalar_problem_from_manifest
from virasoro_bootstrap.moments import MomentWeight, build_identical_scalar_normalized_moment_lower_problem

pytestmark = pytest.mark.bootstrap


def _spec(gap="0.5", **changes):
    spec = IdenticalScalarGapSpec(
        c=mp.mpf(4), h_ext=mp.mpf("0.0625"), delta_gap=mp.mpf(gap), lam=3, spin_list=(0, 2, 4),
        q_order=10, dps=60, npoles=4, num_threads=1,
    )
    return dataclasses.replace(spec, **changes)


@pytest.fixture(scope="module")
def blocks_dir(tmp_path_factory):
    with mp.workdps(60):
        out = tmp_path_factory.mktemp("reuse") / "at_0.5"
        write_identical_scalar_gap_problem(build_identical_scalar_gap_problem(_spec()), out)
    return out / "blocks"


def _cli_blocks(out, *extra, gap="0.5", scalar_only=False):
    """The same blocks as ``_spec()`` written by the CLI (``--u-shift -2 h_ext``; ``--gap`` on spin 0 only).

    Runs ``virasoro_blocks.__main__.main`` in process; the CLI sets ``mp.mp.dps`` from ``--precision``, so the
    caller's precision is restored afterwards.  ``scalar_only`` writes the spin-0 file alone."""
    from virasoro_blocks.__main__ import main as blocks_main

    base = ["--family", "virasoro", "--basis", "scalar_G", "--central-charge", "4", "--delta-ext", *["0.125"] * 4,
            "--coordinates", "zzb", "--lambda", "3", "--order", "10", "--npoles", "4", "--precision", "60",
            "--format", "json", "-o", str(out), *extra]
    runs = [["--spin-ranges", "0", "--gap", gap]]
    if not scalar_only:
        runs.append(["--spin-ranges", "2", "4"])
    for spins in runs:
        saved = mp.mp.dps
        try:
            blocks_main(["table", *base, *spins])
        finally:
            mp.mp.dps = saved
    return out


def _numbers(doc):
    if isinstance(doc, dict):
        return [v for key in sorted(doc) for v in _numbers(doc[key])]
    if isinstance(doc, list):
        return [v for item in doc for v in _numbers(item)]
    return [doc]


def _same_pmp(a, b):
    """Same PMP up to last-place noise: every number agrees to half the working digits."""
    xs, ys = _numbers(a), _numbers(b)
    assert len(xs) == len(ys)
    tol = mp.mpf(10) ** -(mp.mp.dps // 2)
    for x, y in zip(xs, ys):
        if isinstance(x, str) and isinstance(y, str) and x != y:
            x, y = mp.mpf(x), mp.mpf(y)
            assert abs(x - y) <= tol * max(1, abs(x))
        else:
            assert x == y


def test_same_spec_from_files_reproduces_the_manifest_pmp(blocks_dir):
    with mp.workdps(60):
        spec = _spec()
        manifest = build_identical_scalar_problem_from_manifest(blocks_dir.parent / "manifest.json")
        reused = build_identical_scalar_gap_problem(spec, prepared=prepare_identical_scalar_gap_build_from_blocks(spec, blocks_dir))
        assert reused.pmp_build.doc == manifest.pmp_build.doc


@pytest.mark.parametrize('source', ['memory', 'manifest'])
def test_deliberate_subprecision_gap_move_rebuilds_its_channel(blocks_dir, tmp_path, source):
    from fractions import Fraction
    from virasoro_blocks.numeric_input import literal_number
    import json
    import os
    spec = _spec(spin_list=(0,))
    gap = Fraction(1, 2) + Fraction(1, 10**70)
    changed = dataclasses.replace(spec, delta_gap=gap)
    if source == 'manifest':
        # Keep the original files and references; only the requested domain moves.
        document = json.loads((blocks_dir.parent / 'manifest.json').read_text())
        document['spec']['delta_gap'] = str(gap)
        for ref in document['blocks']:
            ref['path'] = os.path.relpath(blocks_dir.parent / ref['path'], tmp_path)
        path = tmp_path / 'manifest.json'
        path.write_text(json.dumps(document))
        result = build_identical_scalar_problem_from_manifest(path)
    else:
        prepared = (prepare_identical_scalar_gap_build(spec) if source == 'memory' else
                    prepare_identical_scalar_gap_build_from_blocks(spec, blocks_dir))
        original = prepared.prepared_spin_channels[0]
        result = build_identical_scalar_gap_problem(changed, prepared=prepared)
        assert result.spin_channels[0].serialized is not original.serialized
        if source == 'memory':
            ope = build_identical_scalar_ope_max_problem(changed, prepared=prepared)
            assert literal_number(ope.normalization_operator['delta_star']) == gap
    channel = result.spin_channels[0]
    assert literal_number(result.spec.delta_gap) == gap
    assert literal_number(channel.delta_min) == literal_number(channel.serialized.shift) == gap


@pytest.mark.parametrize("gap", ["1.75", "0.25"])
def test_other_gap_and_subset_of_spins_from_files(blocks_dir, gap):
    # a spin whose edge moved (above or below the files' anchor 0.5) is built fresh at the new edge, sampled
    # numerically at its own nodes: the problem is the in-memory one; the other spins come from the files
    with mp.workdps(60):
        spec = _spec(gap, spin_list=(0, 2))
        prepared = prepare_identical_scalar_gap_build_from_blocks(spec, blocks_dir)
        reused = build_identical_scalar_gap_problem(spec, prepared=prepared)
        assert [ch.ell for ch in reused.spin_channels] == [0, 2]
        assert reused.spin_channels[0].delta_min == mp.mpf(gap)
        assert set(prepared.exact_tables) == {0}          # only the moved spin needed the engine
        assert len(reused.sample_values_by_block) == len(reused.spin_channels)
        _same_pmp(reused.pmp_build.doc, build_identical_scalar_gap_problem(spec).pmp_build.doc)


def test_other_objectives_from_files(blocks_dir):
    with mp.workdps(60):
        spec = _spec("0.8")
        prepared = prepare_identical_scalar_gap_build_from_blocks(spec, blocks_dir)
        ope = build_identical_scalar_ope_max_problem(spec, delta_star=mp.mpf("0.8"), prepared=prepared)
        assert ope.spin_channels[0].delta_min == mp.mpf("0.8")
        moment = build_identical_scalar_normalized_moment_lower_problem(
            spec, MomentWeight.monomial(Delta=2), MomentWeight.one(), identity_mode="exclude", prepared=prepared,
        )
        values = [v for rows in moment.sample_values_by_block.values() for row in rows for v in row]
        assert values and all(mp.isfinite(mp.mpf(v)) for v in values)


def test_files_that_do_not_describe_the_problem_are_refused(blocks_dir):
    with mp.workdps(60):
        with pytest.raises(ValueError, match="central_charge"):
            prepare_identical_scalar_gap_build_from_blocks(_spec(c=mp.mpf(5)), blocks_dir)
        with pytest.raises(ValueError, match="lambda"):
            prepare_identical_scalar_gap_build_from_blocks(_spec(lam=5), blocks_dir)
        with pytest.raises(ValueError, match="pole_selection"):
            prepare_identical_scalar_gap_build_from_blocks(_spec(pole_selection="residue"), blocks_dir)
        with pytest.raises(ValueError, match="npoles 4 != spec.npoles 6"):
            prepare_identical_scalar_gap_build_from_blocks(_spec(npoles=6), blocks_dir)
        with pytest.raises(ValueError, match="chiral_normalization_N"):
            prepare_identical_scalar_gap_build_from_blocks(_spec(chiral_norm=4), blocks_dir)
        with pytest.raises(ValueError, match=r"spins \[6\]"):
            prepare_identical_scalar_gap_build_from_blocks(_spec(spin_list=(0, 2, 4, 6)), blocks_dir)


# Offsets of 1e-20: far above the roundoff tolerance 10^-(dps/2) of the 60-digit spec.
@pytest.mark.parametrize("key,value", [
    ("z_center", "0.5" + "0" * 19 + "1"),
    ("delta_external", [".125"] * 3),
    ("h_external", ["0.0625" + "0" * 19 + "1"] * 4),
    ("z_shift", "-0.125" + "0" * 19 + "1"),
])
def test_file_reuse_rejects_conflicting_conventions(blocks_dir, tmp_path, key, value):
    import json

    document = json.loads((blocks_dir / "derivs_0.json").read_text())
    document[key] = value
    if key in {"z_center", "zbar_center"}:
        # An off-center table uses the full derivative triangle. Expand the
        # symmetric source so layout validation does not mask the label check.
        for field in ("derivs", "sample_block_values"):
            coordinate_key = 0 if field == "derivs" else "zzb"
            rows = document[field][coordinate_key]
            document[field][coordinate_key] = [
                [rows[max(m, n)][min(m, n)] for n in range(4 - m)]
                for m in range(4)
            ]
    target = tmp_path / "derivs_0.json"
    target.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=key):
        prepare_identical_scalar_gap_build_from_blocks(_spec(spin_list=(0,)), [target])


def test_file_reuse_accepts_equivalent_convention_labels(blocks_dir, tmp_path):
    import json

    document = json.loads((blocks_dir / "derivs_0.json").read_text())
    document.update(z_center="1/2", zbar_center="5e-1", chiral_normalization_N="32/2")
    # Labels rounded to the metadata digits (a non-terminating rational's decimal) are the same label.
    document.update(h_external=["1/16"] * 4, hbar_external=["0.06250"] * 4,
                    delta_external=["0.125" + "0" * 60 + "1"] * 4, z_shift="-1/8", zbar_shift="-0.1250")
    target = tmp_path / "derivs_0.json"
    target.write_text(json.dumps(document))
    spec = _spec(spin_list=(0,))
    prepared = prepare_identical_scalar_gap_build_from_blocks(spec, [target])
    original = prepare_identical_scalar_gap_build_from_blocks(spec, [blocks_dir / "derivs_0.json"])
    assert prepared.identity_vector == original.identity_vector
    assert prepared.prepared_spin_channels[0].serialized.numerators == original.prepared_spin_channels[0].serialized.numerators


def test_cli_blocks_serve_the_problem(blocks_dir, tmp_path):
    with mp.workdps(60):
        spec = _spec()
        cli = _cli_blocks(tmp_path / "cli", "--u-shift", "-0.125")
        with pytest.raises(ValueError, match="pole_selection"):
            prepare_identical_scalar_gap_build_from_blocks(_spec(pole_selection="nearest"), cli)
        manifest = build_identical_scalar_problem_from_manifest(blocks_dir.parent / "manifest.json")
        reused = build_identical_scalar_gap_problem(spec, prepared=prepare_identical_scalar_gap_build_from_blocks(spec, cli))
        _same_pmp(reused.pmp_build.doc, manifest.pmp_build.doc)
        # the CLI files carry the block values at their nodes: every block's SDP data are numeric samples
        assert len(reused.sample_values_by_block) == len(reused.spin_channels)
        # without the crossing shift the CLI files describe another function, and the refusal says how to fix it
        # (each file is checked on load, so the spin-0 file alone triggers it)
        unshifted = _cli_blocks(tmp_path / "unshifted", scalar_only=True)
        with pytest.raises(ValueError, match="--u-shift -1/8"):
            prepare_identical_scalar_gap_build_from_blocks(spec, unshifted)


def test_edited_manifest_gap_rebuilds_the_moved_spin(blocks_dir, tmp_path):
    import json
    import shutil

    with mp.workdps(60):
        edited = tmp_path / "edited"
        shutil.copytree(blocks_dir.parent, edited)
        manifest = json.loads((edited / "manifest.json").read_text())
        for gap in ("1.75", "0.25"):
            manifest["spec"]["delta_gap"] = gap
            (edited / "manifest.json").write_text(json.dumps(manifest))
            rebuilt = build_identical_scalar_problem_from_manifest(edited / "manifest.json")
            assert rebuilt.spin_channels[0].serialized.shift == mp.mpf(gap)
            _same_pmp(rebuilt.pmp_build.doc, build_identical_scalar_gap_problem(_spec(gap)).pmp_build.doc)


def test_files_at_lower_precision_serve_the_same_edge(tmp_path):
    # The CLI's exact .35 label is independent of its table precision.
    from virasoro_blocks.numeric_input import literal_number
    from fractions import Fraction
    _cli_blocks(tmp_path, "--u-shift", "-0.125", gap="0.35")
    with mp.workdps(90):
        spec = _spec(delta_gap="0.35", dps=90)
        prepared = prepare_identical_scalar_gap_build_from_blocks(spec, tmp_path)
        problem = build_identical_scalar_gap_problem(spec, prepared=prepared)
        assert literal_number(problem.spin_channels[0].delta_min) == Fraction(7, 20)
        assert not prepared.exact_tables      # the same edge: served by the file, nothing rebuilt
        binary_gap = mp.mpf('.35')
        changed = dataclasses.replace(spec, delta_gap=binary_gap)
        moved = build_identical_scalar_gap_problem(changed, prepared=prepared)
        assert set(prepared.exact_tables) == {0}
        assert literal_number(moved.spin_channels[0].serialized.shift) == literal_number(binary_gap)


def test_legacy_manifest_blocks_are_vouched_for_by_their_manifest(blocks_dir, tmp_path):
    # manifest block files written before 2026-09-28 record z_shift 0 although they were built at -2 h_ext
    import json
    import shutil

    with mp.workdps(60):
        legacy = tmp_path / "legacy"
        shutil.copytree(blocks_dir.parent, legacy)
        for path in (legacy / "blocks").glob("derivs_*.json"):
            doc = json.loads(path.read_text())
            doc["z_shift"] = doc["zbar_shift"] = "0.0"
            path.write_text(json.dumps(doc))
        spec = _spec()
        reused = build_identical_scalar_gap_problem(spec, prepared=prepare_identical_scalar_gap_build_from_blocks(spec, legacy))
        _same_pmp(reused.pmp_build.doc, build_identical_scalar_problem_from_manifest(blocks_dir.parent / "manifest.json").pmp_build.doc)
        # the same files away from their manifest carry no such guarantee
        loose = tmp_path / "loose"
        shutil.copytree(legacy / "blocks", loose)
        with pytest.raises(ValueError, match="z_shift"):
            prepare_identical_scalar_gap_build_from_blocks(spec, loose)


def test_prefactor_exponent_other_than_h_ext_is_refused():
    # it used to be recorded in the manifest and silently ignored (the blocks always carry z_shift = -2 h_ext)
    from fractions import Fraction
    from virasoro_bootstrap.identical_scalar import _validate_gap_spec_common

    with pytest.raises(ValueError, match="prefactor_exponent"):
        _validate_gap_spec_common(_spec(prefactor_exponent=Fraction(1,16)+Fraction(1,10**70)))
    with mp.workdps(60):
        with pytest.raises(ValueError, match="prefactor_exponent"):
            build_identical_scalar_gap_problem(_spec(prefactor_exponent=mp.mpf("0.2")))
        build_identical_scalar_gap_problem(_spec(prefactor_exponent=mp.mpf("0.0625")))


def test_exact_cli_file_serves_keep_all_request(tmp_path):
    from virasoro_blocks.__main__ import main as blocks_main
    from virasoro_blocks.derivs import G_zzb_table

    blocks_main(['table', '--basis', 'scalar_G', '--central-charge', '4',
        '--h-ext', '.1', '.1', '.1', '.1', '--hbar-ext', '.1', '.1', '.1', '.1',
        '--coordinates', 'zzb', '--format', 'json',
        '--lambda', '1', '--order', '2', '--npoles', 'all', '--u-shift', '-.2',
        '--spin-ranges', '0', '--gap', '1', '--precision', '30', '--q-gate', 'off',
        '-o', str(tmp_path)])
    with mp.workdps(30):
        for npoles in ('all', None):
            spec = _spec(delta_gap='1', h_ext='.1', lam=1, q_order=2, npoles=npoles,
                         spin_list=(0,), dps=30)
            prepared = prepare_identical_scalar_gap_build_from_blocks(spec, tmp_path)
            assert not prepared.exact_tables
            table = prepared.prepared_spin_channels[0].serialized
            actual = table.evaluate('zzb', '1.5')[1][0]
            expected = G_zzb_table('1.5', 0, ['.1'] * 4, 1, 1, c=4, q_order=2,
                                    z_shift='-.2')[1][0]
            assert abs(actual / expected - 1) < mp.mpf('1e-25')


@pytest.mark.parametrize('entry', ['files'])
def test_reduced_file_cannot_satisfy_keep_all_request(blocks_dir, tmp_path, entry):
    import json
    import shutil

    with mp.workdps(60), pytest.raises(ValueError, match='keep-all.*requires'):
        if entry == 'files':
            prepare_identical_scalar_gap_build_from_blocks(_spec(npoles='all'), blocks_dir)
        else:
            copy = tmp_path / 'edited'
            shutil.copytree(blocks_dir.parent, copy)
            path = copy / 'manifest.json'
            document = json.loads(path.read_text())
            document['spec']['npoles'] = 'all'
            path.write_text(json.dumps(document))
            build_identical_scalar_problem_from_manifest(path)


def test_keep_all_tower_inherits_pole_request_and_round_trips(tmp_path):
    import json
    from virasoro_bootstrap.manifest import write_identical_scalar_manifest

    spec = _spec(c=4, h_ext='.1', delta_gap=1, lam=1, q_order=2, npoles='all',
                 spin_list=(0,), dps=30, large_spin_lmin=2, large_spin_twists=('.3',))
    with mp.workdps(30):
        original = build_identical_scalar_gap_problem(spec)
        path = write_identical_scalar_manifest(original, tmp_path)
        restored = build_identical_scalar_problem_from_manifest(path)
        assert len(original.large_spin_channels[0].serialized.pole_list_x) == 2
        _same_pmp(restored.pmp_build.doc, original.pmp_build.doc)
        # A subsequent request for all tower poles cannot reuse a capped tower file.
        document = json.loads(path.read_text())
        tower = next(ref for ref in document['blocks'] if ref['kind'] == 'large_spin')
        block_path = path.parent / tower['path']
        block = json.loads(block_path.read_text())
        block['pole_list_x'].pop()
        from virasoro_bootstrap.manifest import block_content_hash
        tower['content_hash'] = block_content_hash(block)
        path.write_text(json.dumps(document))
        block_path.write_text(json.dumps(block))
        with pytest.raises(ValueError, match='keep-all.*requires 2'):
            build_identical_scalar_problem_from_manifest(path)


def test_unrequested_tower_file_is_unused_before_physical_checks(tmp_path, monkeypatch):
    import json
    from virasoro_blocks.polynomial import Polynomial
    from virasoro_blocks.serialization import SerializedDerivTable
    from virasoro_blocks.writers import write_json
    import virasoro_bootstrap.identical_scalar as scalar

    table = SerializedDerivTable(shift=0, base=mp.mpf('.5'), pole_list_x=(),
        coordinates=('zzb',), sample_points_x=(1, 2),
        numerators={'zzb': [[Polynomial([1])], [Polynomial([2])]]})
    path = tmp_path / 'derivs_0.json'
    path.write_text(write_json(table, c=4, h_ext=['.1'] * 4, hbar_ext=['.1'] * 4,
        spin=0, q_order=2, npoles=0, z_shift='-.2', zbar_shift='-.2'))
    spec = _spec(delta_gap=0, h_ext='.1', lam=1, q_order=2, npoles=0, spin_list=(0,), dps=30)
    monkeypatch.setattr(scalar, '_numeric_identity_vector', lambda *args, **kwargs: (1,))
    expected = prepare_identical_scalar_gap_build_from_blocks(spec, tmp_path)
    tower = json.loads(path.read_text())
    tower.update(variable='L', central_charge='7', twist='.3', large_spin_lmin='2', npoles=17)
    (tmp_path / 'derivs_large_spin_0.json').write_text(json.dumps(tower))
    actual = prepare_identical_scalar_gap_build_from_blocks(spec, tmp_path)
    assert actual.prepared_spin_channels == expected.prepared_spin_channels
    assert actual.large_spin_channels == ()


@pytest.mark.parametrize('changed,complex_label', [('spec', False), ('crossing', True)])
def test_manifest_refuses_subprecision_prefactor_edits(blocks_dir, tmp_path, changed, complex_label):
    from fractions import Fraction
    import json
    import shutil

    shutil.copytree(blocks_dir.parent, tmp_path/'edited')
    path = tmp_path/'edited'/'manifest.json'
    document = json.loads(path.read_text())
    shifted = str(Fraction(1,16)+Fraction(1,10**70))
    if complex_label:
        shifted = '.0625'+'0'*65+'1+0j'
    for key in ('spec', 'crossing'):
        if changed in (key, 'both'):
            document[key]['prefactor_exponent'] = shifted
    path.write_text(json.dumps(document))
    with mp.workdps(15), pytest.raises(ValueError, match='prefactor_exponent'):
        build_identical_scalar_problem_from_manifest(path)


def _moment_spec():
    from virasoro_bootstrap.large_spin import double_twist_set

    return _spec("0.1", large_spin_lmin=6, large_spin_twists=tuple(double_twist_set(mp.mpf("0.0625"), 2)), q_order=12)


@pytest.fixture(scope="module")
def moment_prepared():
    """The moment problems' shared block build (with a large-spin tower), prepared once per module."""
    with mp.workdps(60):
        return prepare_identical_scalar_gap_build(_moment_spec())


# moment_upper, normalized_moment_upper and normalized_ratio_lower_probe take the same reload branch as
# moment_lower, normalized_moment_lower and normalized_ratio_probe, so only one of each pair is run.
@pytest.mark.parametrize("kind", [
    "moment_lower", "normalized_moment_lower", "normalized_ratio_probe", "ratio_basis", "no_tower",
])
def test_moment_manifest_reloads_the_problem(kind, tmp_path, moment_prepared):
    # the moment insertion is rebuilt from the recorded weights on the files' positivity blocks
    from virasoro_bootstrap import moments as M

    spec = _moment_spec()
    W, D1, one = M.MomentWeight.monomial(Delta=2), M.MomentWeight.monomial(Delta=1), M.MomentWeight.one()
    builders = {
        "moment_lower": lambda p: M.build_identical_scalar_moment_lower_problem(spec, W, prepared=p),
        "normalized_moment_lower": lambda p: M.build_identical_scalar_normalized_moment_lower_problem(
            spec, W, D1, alpha=mp.mpf("0.3"), identity_mode="exclude", prepared=p),
        "normalized_ratio_probe": lambda p: M.build_identical_scalar_normalized_ratio_probe_problem(
            spec, W, one, mp.mpf("1.7"), prepared=p),
        "ratio_basis": lambda p: M.build_identical_scalar_normalized_moment_lower_problem(
            spec, W, one, block_basis="ratio", prepared=p),
        "no_tower": lambda p: M.build_identical_scalar_normalized_moment_lower_problem(
            spec, W, one, include_tower=False, prepared=p),
    }
    with mp.workdps(60):
        original = builders[kind](moment_prepared)
        write_identical_scalar_gap_problem(original, tmp_path)
        reloaded = build_identical_scalar_problem_from_manifest(tmp_path / "manifest.json")
        assert reloaded.problem_kind == original.problem_kind
        assert reloaded.objective_vector == original.objective_vector
        assert len(reloaded.sample_values_by_block) == len(original.sample_values_by_block) == len(original.spin_channels)
        _same_pmp(reloaded.pmp_build.doc, original.pmp_build.doc)


def test_problem_and_manifest_do_not_depend_on_ambient_precision(tmp_path):
    """Spec values were parsed at the caller's ambient precision: a build at mp.dps = 15 recorded
    h_ext = 0.14999999999999999445 and wrote 15-digit block files."""
    import json

    from virasoro_bootstrap.identical_scalar import write_identical_scalar_gap_problem

    docs, files = [], []
    for ambient in (15, 60, 100):
        with mp.workdps(ambient):
            spec = IdenticalScalarGapSpec(c="2.5", h_ext="0.15", delta_gap="0.4", lam=3, spin_list=(0, 2),
                                          q_order=12, dps=30, npoles=6, num_threads=1)
            problem = build_identical_scalar_gap_problem(spec)
            manifest_path, pmp_path = write_identical_scalar_gap_problem(problem, tmp_path / str(ambient))
            docs.append(json.dumps(problem.pmp_build.doc))
            # (a higher ambient precision wrote 100-digit strings whose last 70 digits were noise)
            files.append((open(pmp_path).read(), open(manifest_path).read()))
    assert docs[0] == docs[1] == docs[2]
    assert files[0] == files[1] == files[2]
    manifest = json.load(open(tmp_path / "15" / "manifest.json"))
    assert manifest["spec"]["h_ext"] == "0.15"


def test_ope_max_distinguished_operator_uses_its_exact_block():
    """ope_max read V* off the channel's order-reduced table, an extrapolation below the channel edge (V* 2e-3 off at
    Delta* = 0.1 under a 1.5 gap; the OPE^2 bound moved 0.7%).  It now uses the exact point block, as ope_bound does."""
    from virasoro_bootstrap.identical_scalar import (
        build_identical_scalar_ope_bound_problem, build_identical_scalar_ope_max_problem,
    )

    with mp.workdps(40):
        spec = IdenticalScalarGapSpec(c="2", h_ext="0.1", delta_gap="1.5", lam=3, spin_list=(0, 2), q_order=12,
                                      dps=40, npoles=6, num_threads=1)
        top = build_identical_scalar_ope_max_problem(spec, spin_star=0, delta_star=mp.mpf("0.1"))
        upper = build_identical_scalar_ope_bound_problem(spec, spin_star=0, delta_star=mp.mpf("0.1"), direction="upper")
        assert top.normalization_vector == upper.normalization_vector


def test_files_about_another_centre_or_duplicated_are_refused(tmp_path):
    """A --z-center 0.4 table entered the z = 1/2 crossing equation unchecked; a second file for a spin silently
    replaced the first."""
    import json
    import shutil

    _cli_blocks(tmp_path / "a", "--u-shift", "-0.125")
    with mp.workdps(60):
        spec = _spec()
        moved = tmp_path / "moved"
        shutil.copytree(tmp_path / "a", moved)
        for path in moved.glob("derivs_*.json"):
            doc = json.loads(path.read_text())
            doc["z_center"] = "0.4"
            path.write_text(json.dumps(doc))
        with pytest.raises(ValueError, match="z_center"):
            prepare_identical_scalar_gap_build_from_blocks(spec, moved)
        files = sorted((tmp_path / "a").glob("derivs_*.json"))
        dup = tmp_path / "dup"
        dup.mkdir()
        shutil.copy(files[0], dup / files[0].name)
        with pytest.raises(ValueError, match="two block files"):
            prepare_identical_scalar_gap_build_from_blocks(spec, [*files, dup / files[0].name])


@pytest.mark.parametrize("key, value, message", [
    ("c", "5", "central_charge"),
    ("q_order", 14, "order"),
    ("npoles", 6, "npoles"),
    ("lam", 5, "lambda"),
])
def test_manifest_with_an_edited_spec_is_refused(blocks_dir, tmp_path, key, value, message):
    """An edited manifest spec rebuilt a bit-identical PMP from the old files and labeled it with the new spec."""
    import json
    import shutil

    from virasoro_bootstrap.manifest import build_identical_scalar_problem_from_manifest

    edited = tmp_path / "edited"
    shutil.copytree(blocks_dir.parent, edited)
    doc = json.loads((edited / "manifest.json").read_text())
    doc["spec"][key] = value
    (edited / "manifest.json").write_text(json.dumps(doc))
    with mp.workdps(60), pytest.raises(ValueError, match=message):
        build_identical_scalar_problem_from_manifest(edited / "manifest.json")


@pytest.mark.parametrize('spin', [0.5, '2.00000000000000000000000000001'])
def test_reused_scalar_file_cannot_relabel_an_invalid_spin(spin):
    from virasoro_bootstrap.identical_scalar import _check_block_doc

    doc = {'basis': 'scalar_G', 'coordinates': ['zzb'], 'spin': spin,
           'central_charge': '4', 'delta_external': ['0.125']*4,
           'z_shift': '-0.125', 'zbar_shift': '-0.125', 'sample_points_x': ['1'],
           'lambda': 3, 'order': 10, 'npoles': 4}
    with pytest.raises(ValueError, match='spin'):
        _check_block_doc(doc, 'test.json', _spec(), c=4, h_ext=mp.mpf('0.0625'), coordinate='zzb')


def test_lower_precision_cli_file_with_a_rounded_shift_serves_a_higher_precision_spec(tmp_path):
    """A 30-digit table rounds z_shift = -2/15 to 30 digits; a 64-digit spec compared it at 32 digits, refused it,
    and told the user to pass the --u-shift they had passed."""
    from fractions import Fraction

    from virasoro_blocks.__main__ import main as blocks_main

    saved = mp.mp.dps
    try:
        blocks_main(["table", "--family", "virasoro", "--basis", "scalar_G", "--central-charge", "4",
                     "--delta-ext", *["2/15"] * 4, "--coordinates", "zzb", "--lambda", "3", "--order", "10",
                     "--npoles", "4", "--precision", "30", "--u-shift", "-2/15", "--format", "json",
                     "-o", str(tmp_path), "--spin-ranges", "0", "--gap", "0.5"])
    finally:
        mp.mp.dps = saved
    spec = _spec(h_ext=Fraction(1, 15), spin_list=(0,), dps=64)
    with mp.workdps(64):
        prepared = prepare_identical_scalar_gap_build_from_blocks(spec, tmp_path)
        assert [channel.ell for channel in prepared.prepared_spin_channels] == [0]
        # the file records a direct (unregularized) table: a regularized request is a different block
        with pytest.raises(ValueError, match="regularization_policy 'direct'.*--regularize on"):
            prepare_identical_scalar_gap_build_from_blocks(dataclasses.replace(spec, regularize="on"), tmp_path)


def test_prepared_build_serves_the_manifest_round_trip_spec(blocks_dir):
    """A manifest records chiral_norm None as 16; the prepared build keyed on the raw None refused the reloaded
    spec, without saying which field differed."""
    with mp.workdps(60):
        prepared = prepare_identical_scalar_gap_build_from_blocks(_spec(), blocks_dir)
        reloaded = build_identical_scalar_problem_from_manifest(blocks_dir.parent / "manifest.json").spec
        assert reloaded.chiral_norm == 16
        build_identical_scalar_gap_problem(reloaded, prepared=prepared)
        with pytest.raises(ValueError, match=r"q_order \(prepared 10, requested 12\)"):
            build_identical_scalar_gap_problem(_spec(q_order=12), prepared=prepared)


def test_manifest_with_an_edited_dps_is_refused(blocks_dir, tmp_path):
    """The block files keep their written precision: an edited dps would write and solve the SDP at digits the
    data do not have."""
    import shutil

    edited = tmp_path / "edited"
    shutil.copytree(blocks_dir.parent, edited)
    doc = json.loads((edited / "manifest.json").read_text())
    doc["spec"]["dps"] = 90
    (edited / "manifest.json").write_text(json.dumps(doc))
    with mp.workdps(60), pytest.raises(ValueError, match="spec.dps was edited .*written at 60 digits"):
        build_identical_scalar_problem_from_manifest(edited / "manifest.json")
