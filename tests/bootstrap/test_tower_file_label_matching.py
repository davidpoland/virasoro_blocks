"""Exact tower inputs must not reuse a table for a different nearby label."""
from dataclasses import replace
from fractions import Fraction
import json
import warnings

import mpmath as mp
import pytest

from virasoro_blocks.numeric_input import literal_number
from virasoro_bootstrap.identical_scalar import (
    IdenticalScalarGapSpec, build_identical_scalar_gap_problem,
    prepare_identical_scalar_gap_build_from_blocks,
)
from virasoro_bootstrap.manifest import (
    write_identical_scalar_manifest, build_identical_scalar_problem_from_manifest,
    block_content_hash,
)


@pytest.fixture
def tower_files(tmp_path):
    spec = IdenticalScalarGapSpec(c='2',h_ext='.3',delta_gap=0,lam=1,
        spin_list=(0,),q_order=8,npoles=4,dps=40,num_threads=1,
        large_spin_lmin=2,large_spin_twists=(Fraction(1,3),),large_spin_npoles=2)
    with mp.workdps(spec.dps), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        problem = build_identical_scalar_gap_problem(spec)
        path = write_identical_scalar_manifest(problem,tmp_path)
    return problem.spec,path


def test_manifest_rejects_subprecision_twist_edit(tower_files):
    spec,path = tower_files
    document = json.loads(path.read_text())
    document['spec']['large_spin_twists'] = [str(Fraction(1,3)+Fraction(1,10**70))]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError,match='twist'):
        build_identical_scalar_problem_from_manifest(path)


def test_file_preparation_rejects_a_twist_change(tower_files):
    spec,path = tower_files
    different = replace(spec,large_spin_twists=(Fraction(1,3)+Fraction(1,10**20),))
    with pytest.raises(ValueError,match='twist'):
        prepare_identical_scalar_gap_build_from_blocks(different,path.parent)


def test_equal_exact_labels_load_from_files(tower_files):
    spec,path = tower_files
    equivalent = replace(spec,large_spin_twists=('1/3',))
    prepared = prepare_identical_scalar_gap_build_from_blocks(equivalent,path.parent)
    assert literal_number(prepared.large_spin_channels[0].twist) == Fraction(1,3)


@pytest.mark.parametrize('entry', ['manifest'])
def test_subprecision_tower_floor_change_refused(tower_files,entry):
    spec,path = tower_files
    document = json.loads(path.read_text())
    ref = next(ref for ref in document['blocks'] if ref['kind'] == 'large_spin')
    block_path = path.parent/ref['path']
    block = json.loads(block_path.read_text())
    block['large_spin_lmin'] = str(Fraction(2)+Fraction(1,10**70))
    block_path.write_text(json.dumps(block))
    ref['content_hash'] = block_content_hash(block)
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError,match='large_spin_lmin'):
        if entry == 'manifest':
            build_identical_scalar_problem_from_manifest(path)
        else:
            prepare_identical_scalar_gap_build_from_blocks(spec,path.parent)


def test_nearby_distinct_exact_twists_do_not_collapse_to_duplicate_files(tower_files):
    spec,path = tower_files
    document = json.loads(path.read_text())
    ref = next(ref for ref in document['blocks'] if ref['kind'] == 'large_spin')
    block = json.loads((path.parent/ref['path']).read_text())
    different = Fraction(1,3)+Fraction(1,10**70)
    block['twist'] = str(different)
    # Only label matching is exercised: these deliberately equal table values
    # would also occur when the labels round together at the working precision.
    (path.parent/'blocks'/'derivs_large_spin_extra.json').write_text(json.dumps(block))
    requested = replace(spec,large_spin_twists=(Fraction(1,3),different))
    prepared = prepare_identical_scalar_gap_build_from_blocks(requested,path.parent)
    assert tuple(literal_number(c.twist) for c in prepared.large_spin_channels) == (Fraction(1,3),different)


def test_subprecision_fractional_floor_refused_before_preparation(tower_files):
    spec,path = tower_files
    different = replace(spec,large_spin_lmin=Fraction(2)+Fraction(1,10**70))
    with pytest.raises(ValueError,match='large_spin_lmin'):
        prepare_identical_scalar_gap_build_from_blocks(different,path.parent)


@pytest.mark.parametrize('field, entry', [('lambda', 'manifest'), ('npoles', 'files')])
def test_fractional_integer_metadata_cannot_be_truncated(tower_files,field,entry):
    spec,path = tower_files
    document = json.loads(path.read_text())
    ref = next(ref for ref in document['blocks'] if ref['kind'] == 'fixed_spin')
    block_path = path.parent/ref['path']
    block = json.loads(block_path.read_text())
    block[field] = float(block[field])+0.5
    block_path.write_text(json.dumps(block))
    # Integer accuracy controls are metadata, so the table hash is unchanged.
    assert ref['content_hash'] == block_content_hash(block)
    with pytest.raises(ValueError,match=field):
        if entry == 'manifest':
            build_identical_scalar_problem_from_manifest(path)
        else:
            prepare_identical_scalar_gap_build_from_blocks(spec,path.parent)


@pytest.mark.parametrize('field', ['twist', 'ell'])
def test_tower_reference_cannot_disagree_with_its_block(tower_files,field):
    spec,path = tower_files
    document = json.loads(path.read_text())
    ref = next(ref for ref in document['blocks'] if ref['kind'] == 'large_spin')
    ref[field] = 3 if field == 'ell' else str(Fraction(ref[field])+Fraction(1,10**70))
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError,match='tower'):
        build_identical_scalar_problem_from_manifest(path)
