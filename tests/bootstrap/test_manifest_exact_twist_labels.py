"""Manifest serialization preserves the tower labels retained by preparation."""
from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_blocks.numeric_input import ExactRational, literal_number
from virasoro_bootstrap.identical_scalar import IdenticalScalarGapSpec
from virasoro_bootstrap.manifest import _spec_from_json, _spec_to_json


def test_gap_and_spin_edges_round_trip_as_exact_domain_labels():
    gap, edge = Fraction(1, 3), Fraction(2) + Fraction(1, 10**70)
    external = Fraction(1,3)+Fraction(1,10**70)
    spec = IdenticalScalarGapSpec(c=32, h_ext=external, prefactor_exponent=external, delta_gap=gap, lam=1,
        spin_list=(0, 2), dps=20, delta_min_by_spin={2: edge})
    with mp.workdps(15):
        document = _spec_to_json(spec)
        restored = _spec_from_json(document)
    assert Fraction(document['delta_gap']) == literal_number(restored.delta_gap) == gap
    assert Fraction(document['delta_min_by_spin']['2']) == literal_number(restored.delta_min_by_spin[2]) == edge
    assert Fraction(document['prefactor_exponent']) == literal_number(restored.prefactor_exponent) == external


@pytest.mark.parametrize('floor', [str(Fraction(2) + Fraction(1, 10**70))])
def test_manifest_tower_floor_is_validated_before_numeric_rounding(floor):
    spec = IdenticalScalarGapSpec(c=4, h_ext='.1', delta_gap=0, lam=1,
        spin_list=(0,), dps=20, large_spin_lmin=2, large_spin_twists=('.3',))
    document = _spec_to_json(spec)
    document['large_spin_lmin'] = floor
    with mp.workdps(15), pytest.raises(ValueError, match='large_spin_lmin'):
        _spec_from_json(document)
    for exact_floor in (2, '2/1'):
        document['large_spin_lmin'] = exact_floor
        with mp.workdps(15):
            assert _spec_from_json(document).large_spin_lmin == 2


@pytest.mark.parametrize('twist', [ExactRational(7,10)+ExactRational(1,10**70)])
def test_twist_labels_round_trip_without_working_precision_rounding(twist):
    spec = IdenticalScalarGapSpec(c='2',h_ext='.3',delta_gap=0,lam=1,
        spin_list=(0,),q_order=2,npoles=0,dps=20,num_threads=1,
        large_spin_lmin=2,large_spin_twists=(twist,))
    with mp.workdps(15):
        document = _spec_to_json(spec)
        restored = _spec_from_json(document)
    assert Fraction(document['large_spin_twists'][0]) == twist
    assert literal_number(restored.large_spin_twists[0]) == twist


def test_nonterminating_twist_survives_a_real_manifest_reload(tmp_path):
    import json
    import warnings
    from virasoro_bootstrap.identical_scalar import build_identical_scalar_gap_problem
    from virasoro_bootstrap.manifest import (
        write_identical_scalar_manifest, build_identical_scalar_problem_from_manifest)
    spec = IdenticalScalarGapSpec(c='2',h_ext=Fraction(1,3),prefactor_exponent='1/3',delta_gap=0,lam=1,
        spin_list=(0,),q_order=8,npoles=4,dps=40,num_threads=1,
        large_spin_lmin=2,large_spin_twists=(ExactRational(1,3),),large_spin_npoles=2)
    with mp.workdps(40), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        problem = build_identical_scalar_gap_problem(spec)
        path = write_identical_scalar_manifest(problem,tmp_path)
        restored = build_identical_scalar_problem_from_manifest(path)
    document = json.loads(path.read_text())
    assert document['spec']['prefactor_exponent'] == document['crossing']['prefactor_exponent'] == '1/3'
    assert literal_number(restored.spec.h_ext) == literal_number(restored.prefactor_exponent) == Fraction(1,3)
    assert restored.spec.large_spin_twists == (Fraction(1,3),)
    assert literal_number(restored.large_spin_channels[0].twist) == Fraction(1,3)

    # Older valid manifests recorded this derived field numerically. Accept
    # its exactly reproducible old spelling, without accepting nearby edits.
    from virasoro_bootstrap.identical_scalar import _stringify_number
    with mp.workdps(spec.dps):
        old_label = _stringify_number(mp.mpf('1/3'))
    for key in ('spec', 'crossing'):
        document[key]['prefactor_exponent'] = old_label
    path.write_text(json.dumps(document))
    with mp.workdps(15):
        legacy = build_identical_scalar_problem_from_manifest(path)
    assert literal_number(legacy.prefactor_exponent) == Fraction(1,3)
    assert legacy.pmp_build.doc == restored.pmp_build.doc
