"""Isolated dimensions must remain exact through validation and file reuse."""
from fractions import Fraction
import json
import warnings

import mpmath as mp
import pytest

from virasoro_blocks.numeric_input import literal_number
from virasoro_bootstrap import identical_scalar as scalar, moments, point_constraint
from virasoro_bootstrap.manifest import (
    write_identical_scalar_manifest, build_identical_scalar_problem_from_manifest,
)
from types import SimpleNamespace
from virasoro_blocks._conventions import scalar_G_spin0_convention


@pytest.mark.parametrize('spin', [2])
def test_subprecision_dimension_below_unitarity_is_rejected(spin):
    with mp.workdps(40), pytest.raises(ValueError, match='unitarity'):
        scalar._validate_identical_scalar_operator(
            spin, Fraction(spin)-Fraction(1,10**70), name='point')


@pytest.mark.parametrize('value', [Fraction(7,10)+Fraction(1,10**70)])
def test_point_adapter_passes_exact_label_to_engine(monkeypatch,value):
    seen = []
    def build(*args,**kwargs):
        seen.append(kwargs['point_value'])
        return object()
    monkeypatch.setattr(point_constraint,'point_block_table',build)
    with mp.workdps(40):
        channel = point_constraint.build_point_channel(0,value,h_ext='.3',c='2',lam=1)
    assert literal_number(seen[0]) == literal_number(value)
    assert literal_number(channel.delta_min) == literal_number(value)


def test_known_operator_normalization_keeps_dimension_label_and_float_coefficient():
    point = Fraction(7,10)+Fraction(1,10**70)
    seen = []
    def evaluate(coordinate,value):
        seen.append(value)
        return [[0],[1]]
    reference = SimpleNamespace(prefactored=SimpleNamespace(evaluate=evaluate))
    with mp.workdps(30):
        result = point_constraint.normalization_with_fixed_operators(
            [0],[(1,0)],[(reference,point,0.7)])
        assert literal_number(seen[0]) == point
        assert result == (mp.mpf('.7'),)


def spec(**kwargs):
    return scalar.IdenticalScalarGapSpec(c='2',h_ext='.3',delta_gap=1,
        lam=1,spin_list=(0,),q_order=8,npoles=4,dps=40,num_threads=1,**kwargs)


@pytest.fixture
def point_files(tmp_path):
    values = (Fraction(1,3),Fraction(7,10)+Fraction(1,10**70))
    with mp.workdps(40),warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        problem = scalar.build_identical_scalar_gap_problem(spec(point_constraints=tuple((0,v) for v in values)))
        path = write_identical_scalar_manifest(problem,tmp_path)
    return problem,path,values


def test_exact_point_labels_survive_build_and_manifest(point_files):
    problem,path,values = point_files
    assert tuple(literal_number(ch.delta_min) for ch in problem.point_channels) == values
    document = json.loads(path.read_text())
    assert tuple(Fraction(delta) for _,delta in document['spec']['point_constraints']) == values
    with mp.workdps(15):
        rebuilt = build_identical_scalar_problem_from_manifest(path)
    assert tuple(literal_number(delta) for _,delta in rebuilt.spec.point_constraints) == values
    assert tuple(literal_number(ch.delta_min) for ch in rebuilt.point_channels) == values
    def compare(left,right):
        if isinstance(left,dict):
            assert left.keys() == right.keys()
            for key in left:
                compare(left[key],right[key])
        elif isinstance(left,list):
            assert len(left) == len(right)
            for a,b in zip(left,right):
                compare(a,b)
        else:
            a,b = mp.mpf(left),mp.mpf(right)
            assert abs(a-b) <= mp.mpf("1e-34")*max(1,abs(a),abs(b))
    with mp.workdps(40):
        compare(rebuilt.pmp_build.doc,problem.pmp_build.doc)


@pytest.mark.parametrize('edit_snapshot', [False,True])
def test_subprecision_manifest_point_edit_cannot_reuse_frozen_block(point_files,edit_snapshot):
    _,path,values = point_files
    document = json.loads(path.read_text())
    document['spec']['point_constraints'][0][1] = str(values[0]+Fraction(1,10**70))
    if edit_snapshot:
        document['provenance']['block_settings']['point_constraints'] = document['spec']['point_constraints']
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError,match='point_constraints|point constraints'):
        build_identical_scalar_problem_from_manifest(path)


@pytest.mark.parametrize('kind', ['max', 'fixed'])
def test_ope_operator_metadata_keeps_exact_labels(tmp_path,kind):
    delta = Fraction(1,3)+Fraction(1,10**70)
    coefficient = Fraction(2,3)+Fraction(1,10**70)
    with mp.workdps(40),warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        if kind == 'max':
            problem = scalar.build_identical_scalar_ope_max_problem(spec(),delta_star=delta)
        elif kind == 'fixed':
            problem = scalar.build_identical_scalar_fixed_ope_problem(spec(),delta_star=delta,ope_squared=coefficient)
        else:
            problem = scalar.build_identical_scalar_ope_bound_problem(spec(),delta_star=delta,direction=kind)
        path = write_identical_scalar_manifest(problem,tmp_path)
    assert literal_number(problem.normalization_operator['delta_star']) == delta
    with mp.workdps(15):
        rebuilt = build_identical_scalar_problem_from_manifest(path)
    assert literal_number(rebuilt.normalization_operator['delta_star']) == delta
    if kind == 'fixed':
        assert literal_number(problem.normalization_operator['ope_squared']) == coefficient
        assert literal_number(rebuilt.normalization_operator['ope_squared']) == coefficient


def test_moment_reuse_rejects_distinct_exact_point_constraints():
    original = spec(point_constraints=((0,Fraction(1,3)),))
    different = spec(point_constraints=((0,Fraction(1,3)+Fraction(1,10**70)),))
    with mp.workdps(15),pytest.raises(ValueError,match='point_constraints'):
        moments._resolve_base_problem(different,None,SimpleNamespace(spec=original, scalar_G_spin0=scalar_G_spin0_convention()))
