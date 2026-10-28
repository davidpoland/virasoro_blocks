"""Fixed-operator additions cannot silently target the wrong functional entries."""
from types import SimpleNamespace
from fractions import Fraction
import numpy as np
import mpmath as mp
import pytest
from virasoro_bootstrap.point_constraint import normalization_with_fixed_operators


def channel():
    return SimpleNamespace(prefactored=SimpleNamespace(
        evaluate=lambda coordinate, delta: [[2, 3], [5, 7]]))


@pytest.mark.parametrize('identity,pairs', [([10], [(0, 0), (1, 0)])])
def test_fixed_operator_vector_length_must_match_functional(identity, pairs):
    with pytest.raises(ValueError, match='length|dimension'):
        normalization_with_fixed_operators(identity, pairs, [(channel(), 1, 2)])


@pytest.mark.parametrize('pair', [(True, 0), (2, 0)])
def test_fixed_operator_indices_are_validated(pair):
    with pytest.raises(ValueError, match='F-pair'):
        normalization_with_fixed_operators([10], [pair], [(channel(), 1, 2)])


def test_fixed_operator_uses_requested_entries_and_keeps_inputs_unchanged():
    identity = [10, 20]
    pairs = [(Fraction(1), 0.0), (0, np.int64(1))]
    result = normalization_with_fixed_operators(identity, pairs, [(channel(), 1, 2)])
    assert result == (20, 26)
    assert identity == [10, 20]


@pytest.mark.parametrize('pair', [(0,)])
def test_malformed_fixed_operator_pair_is_rejected(pair):
    with pytest.raises(ValueError, match='F-pair'):
        normalization_with_fixed_operators([10], [pair], [(channel(), 1, 2)])


def test_fixed_operator_sums_preserve_coefficients_and_stored_vector_offsets():
    from virasoro_blocks.numeric_input import literal_number

    tiny = Fraction(1, 10**70)
    ref = SimpleNamespace(prefactored=SimpleNamespace(evaluate=lambda *_: [[0], [1]]))
    with mp.workdps(150):
        stored_coefficient = mp.mpf(1) + mp.mpf('1e-70')
        stored_identity = -mp.mpf(1) + mp.mpf('1e-70')
    with mp.workdps(30):
        for coeff in (1+tiny, '1.'+'0'*69+'1', stored_coefficient):
            result = normalization_with_fixed_operators([-1], [(1, 0)], [(ref, 1, coeff)])
            expected = literal_number(coeff)-1
            assert abs(result[0]/mp.convert(expected)-1) < mp.mpf('1e-28')
        result = normalization_with_fixed_operators([stored_identity], [(1, 0)], [(ref, 1, 1)])
        assert abs(result[0]/mp.convert(literal_number(stored_identity)+1)-1) < mp.mpf('1e-28')
        # All coefficients are positive. Accumulating each contribution at
        # ambient precision would erase the result in either ordering.
        operators = [(ref, 1, 10**100-1), (ref, 1, 1+tiny)]
        for ordered in (operators, operators[::-1]):
            result = normalization_with_fixed_operators([-10**100], [(1, 0)], ordered)
            assert abs(result[0]/mp.convert(tiny)-1) < mp.mpf('1e-28')


def test_fixed_ope_builder_retains_exact_coefficient_in_pmp_normalization(monkeypatch):
    import virasoro_bootstrap.identical_scalar as scalar

    tiny = Fraction(1, 10**70)
    spec = scalar.IdenticalScalarGapSpec(c=4, h_ext='.1', delta_gap=2, lam=1, spin_list=(0,), dps=30)
    gap = scalar.IdenticalScalarGapProblem(
        spec=spec, resolved_method='interpolation', used_regularize=False,
        prefactor_exponent=0, F_pairs=((1, 0),), identity_vector=(-1,),
        spin_channels=(), pmp_build=None,
    )
    monkeypatch.setattr(scalar, 'build_identical_scalar_gap_problem', lambda *_a, **_kw: gap)
    monkeypatch.setattr(scalar, '_exact_operator_vector', lambda *_a: (1,))
    monkeypatch.setattr(scalar, 'build_prefactored_pmp_doc',
                        lambda *_a, **kw: SimpleNamespace(normalization=kw['normalization']))
    with mp.workdps(15):
        problem = scalar.build_identical_scalar_fixed_ope_problem(spec, delta_star=1, ope_squared=1+tiny)
    with mp.workdps(30):
        assert abs(problem.normalization_vector[0]/mp.convert(tiny)-1) < mp.mpf('1e-28')
        assert problem.pmp_build.normalization == list(problem.normalization_vector)
    assert problem.normalization_operator['ope_squared'] == 1+tiny


def test_fixed_operator_labels_validate_before_table_evaluation():
    ref = SimpleNamespace(prefactored=SimpleNamespace(
        evaluate=lambda *args: pytest.fail('invalid coefficient reached the block table')))
    for coefficient in (True, 1j):
        with pytest.raises(ValueError, match='coefficient'):
            normalization_with_fixed_operators([10], [(1, 0)], [(ref, 1, coefficient)])


def test_fixed_operator_problem_modifications_keep_the_crossing_coordinate():
    import virasoro_bootstrap.identical_scalar as scalar
    from virasoro_blocks.polynomial import Polynomial
    from virasoro_blocks.serialization import SerializedDerivTable
    from virasoro_bootstrap.point_constraint import input_fixed_operators

    table = SerializedDerivTable(
        shift=0, base=Fraction(1, 2), pole_list_x=(), coordinates=('zzb', 'yyb'),
        numerators={name: [[Polynomial([0])], [Polynomial([value])]]
                    for name, value in (('zzb', 1), ('yyb', 100))},
    )
    ref = scalar.IdenticalScalarSpinChannel(ell=0, delta_min=0, serialized=table, prefactored=table)
    spec = scalar.IdenticalScalarGapSpec(c=4, h_ext='.1', delta_gap=2, lam=1, spin_list=(0,), dps=30)
    problem = scalar.IdenticalScalarGapProblem(
        spec=spec, resolved_method='interpolation', used_regularize=False,
        prefactor_exponent=0, F_pairs=((1, 0),), identity_vector=(10,),
        spin_channels=(ref,), pmp_build=None,
    )
    for coordinate in (None, 'zzb'):
        modified = input_fixed_operators(problem, [(ref, 0, 2)], coordinate=coordinate)
        assert modified.spec.coordinate == 'zzb' and modified.identity_vector == (12,)
    with pytest.raises(ValueError, match='crossing coordinate'):
        input_fixed_operators(problem, [(ref, 0, 2)], coordinate='yyb')
    # The standalone vector helper has no problem coordinate to preserve.
    assert normalization_with_fixed_operators([10], [(1, 0)], [(ref, 0, 2)], coordinate='yyb') == (210,)


def test_appended_point_channels_preserve_physical_labels_and_frozen_dimensions():
    from dataclasses import replace
    import virasoro_bootstrap.identical_scalar as scalar
    from virasoro_blocks.polynomial import Polynomial
    from virasoro_blocks.serialization import SerializedDerivTable
    from virasoro_bootstrap.point_constraint import append_positivity_channels

    table = SerializedDerivTable(
        shift=Fraction(7, 10), base=Fraction(1, 2), pole_list_x=(), coordinates=('zzb',),
        numerators={'zzb': [[Polynomial([0])], [Polynomial([1])]]},
    )
    point = scalar.IdenticalScalarSpinChannel(ell=0, delta_min=.7, serialized=table, prefactored=table)
    spec = scalar.IdenticalScalarGapSpec(c=4, h_ext='.1', delta_gap=2, lam=1, spin_list=(), dps=30)
    problem = scalar.IdenticalScalarGapProblem(
        spec=spec, resolved_method='interpolation', used_regularize=False,
        prefactor_exponent=0, F_pairs=((1, 0),), identity_vector=(1,),
        spin_channels=(point,), point_channels=(point,), pmp_build=None,
    )
    assert append_positivity_channels(problem, [point]).point_channels == (point, point)
    bad_points = [replace(point, ell=spin) for spin in (True, 1)]
    bad_points.extend((
        replace(point, delta_min=Fraction(7, 10)+Fraction(1, 10**100)),
        replace(point, serialized=replace(table, shift=Fraction(1, 3))),
        replace(point, prefactored=replace(table, shift=Fraction(1, 3))),
        replace(point, serialized=replace(table, numerators={'zzb': [[Polynomial([0, 1])]]})),
    ))
    with mp.workdps(15):
        for bad in bad_points:
            with pytest.raises(ValueError, match='spin|unitarity|frozen|point-constraint'):
                append_positivity_channels(problem, [bad])
    assert problem.point_channels == (point,)


def test_fixed_operator_references_preserve_physical_dimensions_and_variable():
    from dataclasses import replace
    import virasoro_bootstrap.identical_scalar as scalar
    from virasoro_blocks.polynomial import Polynomial
    from virasoro_blocks.serialization import SerializedDerivTable
    from virasoro_bootstrap.point_constraint import input_fixed_operators

    table = SerializedDerivTable(
        shift=1, base=Fraction(1, 2), pole_list_x=(), coordinates=('zzb',),
        numerators={'zzb': [[Polynomial([0])], [Polynomial([8])]]},
    )
    point = scalar.IdenticalScalarSpinChannel(ell=0, delta_min=1, serialized=table, prefactored=table)
    spec = scalar.IdenticalScalarGapSpec(c=4, h_ext='.1', delta_gap=2, lam=1, spin_list=(), dps=30)
    problem = scalar.IdenticalScalarGapProblem(
        spec=spec, resolved_method='interpolation', used_regularize=False,
        prefactor_exponent=0, F_pairs=((1, 0),), identity_vector=(10,),
        spin_channels=(point,), point_channels=(point,), pmp_build=None,
    )
    # A one-shot input is checked and then used exactly once.
    assert input_fixed_operators(problem, iter(((point, '1.0', 1),))).identity_vector == (18,)
    with mp.workdps(15):
        for delta in (2, Fraction(1)+Fraction(1, 10**100)):
            with pytest.raises(ValueError, match='frozen dimension'):
                input_fixed_operators(problem, [(point, delta, 1)])
        for spin, delta in ((2, Fraction(2)-Fraction(1, 10**100)), (1, 2), (True, 2)):
            with pytest.raises(ValueError, match='unitarity|spin'):
                input_fixed_operators(problem, [(replace(point, ell=spin), delta, 1)])
    with pytest.raises(ValueError, match='fixed-spin.*tower'):
        input_fixed_operators(problem, [(replace(point, twist=1), 3, 1)])
    # Continuous references can be evaluated below their continuum edge.
    continuum = replace(point, ell=2, delta_min=4)
    assert input_fixed_operators(problem, [(continuum, 2, 1)]).identity_vector == (14,)
    assert problem.identity_vector == (10,)


def test_helper_built_points_use_and_preserve_the_problem_crossing_context(tmp_path):
    from dataclasses import replace
    import virasoro_bootstrap.identical_scalar as scalar
    from virasoro_blocks.point import point_block_table
    from virasoro_bootstrap.point_constraint import build_point_channel, append_positivity_channels, input_fixed_operators
    from virasoro_bootstrap.manifest import write_identical_scalar_manifest, build_identical_scalar_problem_from_manifest

    spec = scalar.IdenticalScalarGapSpec(c=4, h_ext='.1', delta_gap=2, lam=1,
        spin_list=(0,), q_order=2, dps=30, npoles='all', num_threads=1)
    problem = scalar.build_identical_scalar_gap_problem(spec)
    with mp.workdps(30):
        point = build_point_channel(0, 1, h_ext='.1', c=4, q_order=2, lam=1)
        oracle = point_block_table(['zzb'], basis='scalar_G', ell=0, point_value=1,
            h_externals=['.1']*4, c=4, q_order=2, lam=1,
            z_shift=Fraction(-1, 5), zbar_shift=Fraction(-1, 5))
        assert point.prefactored.evaluate('zzb', 1) == oracle.evaluate('zzb', 1)
        explicit = build_point_channel(0, 1, h_ext='.1', c=4, q_order=2, lam=1,
                                      z_shift='-.2', zbar_shift='-.2')
        assert explicit.point_context == point.point_context
        aliases = build_point_channel(0, 1, h_ext=.1, c='4', q_order=Fraction(2), lam=1.0)
        assert aliases.point_context == point.point_context
        extended = append_positivity_channels(problem, [point])
        assert extended.point_channels == (point,)
        rebuilt = build_identical_scalar_problem_from_manifest(write_identical_scalar_manifest(extended, tmp_path))
        assert rebuilt.point_channels[0].prefactored.evaluate('zzb', 1) == point.prefactored.evaluate('zzb', 1)
        imported_point = rebuilt.point_channels[0]
        foreign_problem = replace(problem, spec=replace(problem.spec, h_ext='.2'))
        with pytest.raises(ValueError, match='h_ext'):
            append_positivity_channels(foreign_problem, [imported_point])
        with pytest.raises(ValueError, match='frozen dimension'):
            input_fixed_operators(problem, [(imported_point, 2, 1)])
        input_fixed_operators(extended, [(explicit, 1, 1)])
        with pytest.raises(ValueError, match='frozen dimension'):
            input_fixed_operators(extended, [(explicit, 2, 1)])
        with pytest.raises(ValueError, match='frozen dimension'):
            normalization_with_fixed_operators(problem.identity_vector, problem.F_pairs, [(explicit, 2, 1)])
        foreign_weight = build_point_channel(0, 1, h_ext='.2', c=4, q_order=2, lam=1)
        with pytest.raises(ValueError, match='h_ext'):
            append_positivity_channels(problem, [foreign_weight])
        for field, value in [('c', 5), ('basis', 'global_scalar_G'), ('lam', 0), ('zbar_shift', 0)]:
            foreign = replace(point, point_context={**point.point_context, field: value})
            for modify in (lambda: append_positivity_channels(problem, [foreign]),
                           lambda: input_fixed_operators(problem, [(foreign, 1, 1)])):
                with pytest.raises(ValueError, match=field):
                    modify()


def test_chiral_point_helpers_match_continuum_context_and_survive_manifest_reload(tmp_path):
    from virasoro_blocks.point import point_block_table
    from virasoro_bootstrap.controls.chiral_control import build_chiral_control_problem
    from virasoro_bootstrap.point_constraint import build_point_channel, append_positivity_channels, input_fixed_operators
    from virasoro_bootstrap.manifest import write_identical_scalar_manifest, build_identical_scalar_problem_from_manifest

    for mode in ('virasoro',):
        basis = 'global_chiral_F' if mode == 'global' else 'chiral_F'
        problem = build_chiral_control_problem(mode=mode, c=2, h_ext='.125', h_gap='.3',
            lam=1, q_order=4, dps=30, n_sample_points=2)
        with mp.workdps(30):
            point = build_point_channel(0, '.2', basis=basis, h_ext='.125', c=2, q_order=4, lam=1)
            oracle = point_block_table(['z'], basis=basis, ell=0, point_value='.2',
                h_externals=['.125']*4, c=2, q_order=4, lam=1, z_shift='-.25', zbar_shift='-.25')
            expected = oracle.evaluate('z', '.2')[1][0]
            assert point.prefactored.coordinates == ('zzb',)
            assert point.point_context['coordinate'] == 'zzb'
            assert abs(point.prefactored.evaluate('zzb', '.2')[1][0]-expected) < mp.mpf('1e-25')
            extended = append_positivity_channels(problem, [point])
            fixed = input_fixed_operators(problem, [(point, '.2', '.1')])
            assert abs(fixed.identity_vector[0]-problem.identity_vector[0]-expected/10) < mp.mpf('1e-25')
            for original, name in ((extended, 'appended'), (fixed, 'fixed')):
                rebuilt = build_identical_scalar_problem_from_manifest(
                    write_identical_scalar_manifest(original, tmp_path/mode/name))
                assert abs(rebuilt.identity_vector[0]-original.identity_vector[0]) < mp.mpf('1e-25')
                if name == 'appended':
                    assert len(rebuilt.point_channels) == 1
                    imported = rebuilt.point_channels[0]
                    assert abs(imported.prefactored.evaluate('zzb', '.2')[1][0]-expected) < mp.mpf('1e-25')
                    input_fixed_operators(rebuilt, [(imported, '.2', '.1')])
            with pytest.raises(ValueError, match='chiral.*ell=0'):
                build_point_channel(2, 3, basis=basis, h_ext='.125', c=2, q_order=4, lam=1)
            with pytest.raises(ValueError, match='zbar_shift'):
                build_point_channel(0, '.2', basis=basis, h_ext='.125', c=2,
                    q_order=4, lam=1, zbar_shift='.7')


def test_continuum_fixed_operator_context_is_preserved_across_reuse(tmp_path):
    from dataclasses import replace
    import virasoro_bootstrap.identical_scalar as scalar
    from virasoro_bootstrap.point_constraint import input_fixed_operators
    from virasoro_bootstrap.manifest import write_identical_scalar_manifest, build_identical_scalar_problem_from_manifest

    spec = scalar.IdenticalScalarGapSpec(c=4, h_ext='.1', delta_gap=2, lam=1,
        spin_list=(0,), q_order=2, dps=30, npoles='all', num_threads=1)
    prepared = scalar.prepare_identical_scalar_gap_build(spec)
    problem = scalar.build_identical_scalar_gap_problem(spec, prepared=prepared)
    moved = scalar.build_identical_scalar_gap_problem(replace(spec, delta_gap=3), prepared=prepared)
    path = write_identical_scalar_manifest(problem, tmp_path)
    reloaded = build_identical_scalar_problem_from_manifest(path)
    from_files = scalar.build_identical_scalar_gap_problem(spec,
        prepared=scalar.prepare_identical_scalar_gap_build_from_blocks(spec, path.parent / 'blocks'))
    larger = scalar.build_identical_scalar_gap_problem(replace(spec, lam=3))
    for source in (problem, moved, reloaded, from_files, larger):
        channel = source.spin_channels[0]
        assert channel.crossing_context['h_ext'] == Fraction(1, 10)
        modified = input_fixed_operators(problem, [(channel, 3, 1)])
        with mp.workdps(30):
            expected = normalization_with_fixed_operators(problem.identity_vector, problem.F_pairs, [(channel, 3, 1)])
            assert modified.identity_vector == modified.normalization_vector == expected
    for field, value in (('c', 5), ('chiral_norm', 8)):
        foreign = scalar.build_identical_scalar_gap_problem(replace(spec, **{field: value})).spin_channels[0]
        with pytest.raises(ValueError, match=field):
            input_fixed_operators(problem, [(foreign, 3, 1)])


def test_fixed_operator_context_checks_cover_control_families():
    from virasoro_bootstrap.controls.global_control import (
        build_global_scalar_spin_channel, build_numeric_virasoro_scalar_spin_channel,
    )
    from virasoro_bootstrap.controls.chiral_control import build_chiral_channel, build_chiral_symbolic_channel
    from virasoro_bootstrap.identical_scalar import IdenticalScalarGapSpec, IdenticalScalarGapProblem
    from virasoro_bootstrap.point_constraint import input_fixed_operators

    spec = IdenticalScalarGapSpec(c=4, h_ext='.1', delta_gap=2, lam=1, spin_list=(0,), q_order=2, dps=30)
    with mp.workdps(30):
        for method, build in (
            ('global_control', lambda he: build_global_scalar_spin_channel(
                ell=0, delta_min=2, h_ext=he, lam=1, n_sample_points=1)),
            ('numeric_virasoro_control', lambda he: build_numeric_virasoro_scalar_spin_channel(
                c=4, q_order=2, ell=0, delta_min=2, h_ext=he, lam=1, n_sample_points=2)),
            ('chiral_global_control', lambda he: build_chiral_channel(
                mode='global', h_gap=2, h_ext=he, lam=1, n_sample_points=2)),
            ('chiral_virasoro_control', lambda he: build_chiral_channel(
                mode='virasoro', c=4, q_order=2, h_gap=2, h_ext=he, lam=1, n_sample_points=2)),
            ('chiral_virasoro_symbolic_control', lambda he: build_chiral_symbolic_channel(
                c=4, q_order=2, h_gap=2, h_ext=he, lam=1, npoles=1)),
        ):
            own, foreign = build('.1'), build('.2')
            decimal_alias = build(.1)
            assert decimal_alias.prefactored.evaluate('zzb', 3) == own.prefactored.evaluate('zzb', 3)
            problem = IdenticalScalarGapProblem(spec=spec, resolved_method=method,
                used_regularize=False, prefactor_exponent=Fraction(1, 10), F_pairs=((1, 0),),
                identity_vector=(1,), spin_channels=(own,), pmp_build=None)
            input_fixed_operators(problem, [(own, 3, 1)])
            with pytest.raises(ValueError, match='h_ext'):
                input_fixed_operators(problem, [(foreign, 3, 1)])


@pytest.mark.parametrize('chiral', [False, True])
def test_manifest_point_files_validate_context_and_frozen_shape(tmp_path, chiral):
    import json
    from copy import deepcopy
    import virasoro_bootstrap.identical_scalar as scalar
    from virasoro_blocks.polynomial import Polynomial
    from virasoro_blocks.serialization import SerializedDerivTable
    from virasoro_bootstrap.manifest import (
        write_identical_scalar_manifest, build_identical_scalar_problem_from_manifest, block_content_hash,
    )

    def channel(shift, point=False):
        table = SerializedDerivTable(
            shift=shift, base=Fraction(1, 2), pole_list_x=(), coordinates=('zzb',),
            sample_points_x=None if point else (1, 2),
            numerators={'zzb': [[Polynomial([1])], [Polynomial([2])]]},
        )
        return scalar.IdenticalScalarSpinChannel(ell=0, delta_min=shift, serialized=table, prefactored=table)

    fixed, point = channel(Fraction(1)), channel(Fraction(7, 10), True)
    spec = scalar.IdenticalScalarGapSpec(c=4, h_ext='.1', delta_gap=1, lam=1,
        spin_list=(0,), q_order=2, npoles=1, dps=30, num_threads=1)
    problem = scalar.IdenticalScalarGapProblem(
        spec=spec, resolved_method='chiral_virasoro_symbolic_control' if chiral else 'interpolation',
        used_regularize=False, prefactor_exponent=Fraction(1, 10), F_pairs=((1, 0),),
        identity_vector=(1,), spin_channels=(fixed, point), point_channels=(point,), pmp_build=None,
        objective_vector=(0,), normalization_vector=(1,),
    )
    path = write_identical_scalar_manifest(problem, tmp_path)
    rebuilt = build_identical_scalar_problem_from_manifest(path)
    assert rebuilt.point_channels[0].prefactored.evaluate('zzb', Fraction(7, 10))[1][0] == 2
    from virasoro_bootstrap.point_constraint import append_positivity_channels
    assert len(append_positivity_channels(rebuilt, rebuilt.point_channels).point_channels) == 2
    original_manifest = json.loads(path.read_text())
    index = next(i for i, ref in enumerate(original_manifest['blocks']) if ref['kind'] == 'point')
    point_path = path.parent / original_manifest['blocks'][index]['path']
    original_block = json.loads(point_path.read_text())
    edits = [('central_charge', '5'), ('basis', 'ordered_B'), ('hbar_external', ['.2'] * 4), ('spin', 2),
             ('derivs', [[['1'], ['2', '1']]] if chiral else [[[["1"]], [["2", "1"]]]])]
    for field, value in edits:
        document, block = deepcopy(original_manifest), deepcopy(original_block)
        block[field] = value
        if field == 'spin':
            if chiral:
                continue  # Chiral manifest loading already rejects any spin label.
            document['blocks'][index]['ell'] = value
        document['blocks'][index]['content_hash'] = block_content_hash(block)
        path.write_text(json.dumps(document))
        point_path.write_text(json.dumps(block))
        with pytest.raises(ValueError, match='spec|point|unitarity'):
            build_identical_scalar_problem_from_manifest(path)
