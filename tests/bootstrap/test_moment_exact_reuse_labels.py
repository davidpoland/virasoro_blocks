"""Moment reuse and certification rebuilds respect exact backend labels."""
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import mpmath as mp
import pytest
from flint import ctx

from virasoro_blocks.numeric_input import ExactRational, literal_components, literal_number
from virasoro_bootstrap import moments
from virasoro_bootstrap.identical_scalar import IdenticalScalarGapSpec
from virasoro_blocks._conventions import scalar_G_spin0_convention


EPS = Fraction(1, 10**70)


def assert_relative(actual, expected):
    assert abs(actual / mp.convert(literal_number(expected)) - 1) < mp.mpf('1e-27')


@pytest.mark.parametrize('form', ['text', 'stored_mpc'])
def test_weight_arithmetic_and_json_preserve_source_coefficients(form):
    value = Fraction(1) + EPS
    if form == 'text':
        value = str(value)
    elif form in ('stored_mpf', 'stored_mpc'):
        with mp.workdps(150):
            value = mp.convert(literal_number(value))
            if form == 'stored_mpc':
                value = mp.mpc(value,0)
    expected = literal_components(value)[0]
    weight = moments.MomentWeight((moments.MomentTerm(value, {}),))
    with mp.workdps(15):
        scaled = weight.scale(Fraction(1, 3))
        assert literal_number(scaled.terms[0].coefficient) == expected / 3
        assert literal_number((-weight).terms[0].coefficient) == -expected
        restored = moments.MomentWeight.from_json(scaled.to_json())
        assert literal_number(restored.terms[0].coefficient) == expected / 3


def test_float_coefficient_json_uses_its_exact_decimal_label():
    with mp.workdps(60):
        data = moments.MomentTerm(0.7).to_json()
    assert literal_number(data['coefficient']) == Fraction(7, 10)


def test_moment_coefficient_cannot_bypass_real_input_validation_with_boolean():
    with pytest.raises(ValueError, match='coefficient.*real'):
        moments.MomentTerm(True).to_json()


def test_moment_discrete_controls_reject_numpy_booleans():
    import numpy as np

    for value in (np.bool_(True),):
        with pytest.raises(ValueError, match='powers.*non-negative integers'):
            moments.MomentTerm(1, {'Delta': value}).to_json()
        with pytest.raises(ValueError, match='spin_filter.*integer'):
            moments.MomentWeight((moments.MomentTerm(1),), spin_filter=value)


def test_fixed_spin_polynomial_keeps_small_antiholomorphic_weight():
    channel = SimpleNamespace(ell=2, prefactored=SimpleNamespace(shift=Fraction(2) + EPS))
    with mp.workdps(30):
        before = ctx.prec
        poly = moments.moment_weight_polynomial(moments.MomentWeight.monomial(hbar=2), channel)
        assert ctx.prec == before
        for actual, expected in zip(poly.coeffs, (EPS**2 / 4, EPS / 2, Fraction(1, 4))):
            assert_relative(actual, expected)
        assert len(poly.coeffs) == 3


@pytest.mark.parametrize('builder', [moments.moment_weight_polynomial, moments.moment_weight_polynomial_in_delta])
def test_tower_polynomial_combines_monomials_before_rounding(builder):
    # (Delta-L)^2 = tau^2 on a tower slice, even if tau is below
    # the precision of the separately represented Delta and L constants.
    channel = SimpleNamespace(ell=4, twist=EPS, prefactored=SimpleNamespace(shift=4))
    weight = moments.MomentWeight((moments.MomentTerm(1, {'Delta': 2}),
        moments.MomentTerm(-2, {'Delta': 1, 'L': 1}), moments.MomentTerm(1, {'L': 2})))
    with mp.workdps(30):
        poly = builder(weight, channel)
        assert len(poly.coeffs) == 1
        assert_relative(poly.coeffs[0], EPS**2)


def test_small_constant_cancellations_survive_polynomial_identity_and_probe_threshold():
    weight = moments.MomentWeight((moments.MomentTerm(1 + EPS), moments.MomentTerm(-1)))
    channel = SimpleNamespace(ell=0, prefactored=SimpleNamespace(shift=0))
    with mp.workdps(30):
        assert_relative(moments.moment_weight_polynomial(weight, channel).coeffs[0], EPS)
        assert_relative(moments._identity_weight_value(weight), EPS)
        assert_relative(moments.normalized_ratio_probe_threshold(1 + EPS, alpha=3, beta=3), 3 * EPS)


def test_effective_constants_add_identity_before_rounding_input_offsets():
    problem = SimpleNamespace(pmp_metadata={'self_dual_identity_measure': '1'})
    with mp.workdps(30):
        alpha, beta, _, _ = moments._effective_ratio_constants(problem,
            moments.MomentWeight.one(), moments.MomentWeight.one(),
            alpha=-1 + EPS, beta=-1 + EPS, identity_mode='include')
        assert_relative(alpha, EPS)
        assert_relative(beta, EPS)
        shifted = moments.MomentWeight((moments.MomentTerm(1 + EPS),))
        alpha, beta, _, _ = moments._effective_ratio_constants(problem, shifted, shifted,
            alpha=-1, beta=-1, identity_mode='include')
        assert_relative(alpha, EPS)
        assert_relative(beta, EPS)


@pytest.mark.parametrize('from_blocks', [True])
def test_rebuilders_deliver_exact_ratio_constants_to_builders(monkeypatch, from_blocks):
    alpha, beta = Fraction(1, 3), Fraction(1) + EPS
    meta = dict(kind='normalized', direction='upper', alpha=str(alpha), beta=str(beta),
        numerator=moments.MomentWeight.one().to_json(), denominator=moments.MomentWeight.one().to_json())
    captured = []
    def builder(new_spec, *args, **kwargs):
        captured.append(kwargs)
    monkeypatch.setattr(moments, '_NORMALIZED_MOMENT_BUILDERS', {'upper': builder})
    monkeypatch.setattr(moments, '_cached_base_problem', lambda value: SimpleNamespace(spec=value))
    source = spec()
    with mp.workdps(15):
        if from_blocks:
            moments.rebuild_moment_problem_from_blocks(source, meta, 'normalized_moment_upper', [],
                base_problem=SimpleNamespace(spec=source, scalar_G_spin0=scalar_G_spin0_convention()))
        else:
            moments.rebuild_normalized_moment_problem(SimpleNamespace(spec=source, pmp_metadata={'moment': meta}))
    assert literal_number(captured[0]['alpha']) == alpha
    assert literal_number(captured[0]['beta']) == beta


@pytest.mark.parametrize('direction', ['lower'])
def test_ratio_probe_pmp_and_samples_keep_small_candidate_shift(direction):
    from virasoro_bootstrap.controls.global_control import build_global_identical_scalar_control_problem

    source = spec(delta_gap='.1', large_spin_lmin=None, large_spin_twists=())
    with mp.workdps(150):
        alpha = mp.mpf(1) + mp.mpf('1e-70')
    with mp.workdps(30):
        base = build_global_identical_scalar_control_problem(source, global_identity=True)
        builder = (moments.build_identical_scalar_normalized_ratio_probe_problem if direction == 'upper'
                   else moments.build_identical_scalar_normalized_ratio_lower_probe_problem)
        result = builder(source, moments.MomentWeight.one(), moments.MomentWeight.one(),
            1 + EPS, alpha=alpha, beta=Fraction(1, 3), identity_mode='exclude',
            base_problem=base, block_basis='ratio')
        expected = EPS if direction == 'upper' else -EPS
        component = result.pmp_build.doc['PositiveMatrixWithPrefactorArray'][0]['polynomials'][0][0][-1]
        assert all(mp.mpf(value) == 0 for value in component[1:])
        assert_relative(mp.mpf(component[0]), expected)
        for row in result.sample_values_by_block[0]:
            assert_relative(mp.mpf(row[-1]), expected)
        meta = result.pmp_metadata['moment']
        assert literal_number(meta['candidate']) == 1 + EPS
        assert literal_number(meta['alpha']) == literal_number(alpha)
        assert literal_number(meta['beta']) == Fraction(1, 3)
    with mp.workdps(15):
        restored = moments.rebuild_moment_problem_from_blocks(source, meta, result.problem_kind, [], base_problem=base)
    assert restored.pmp_build.doc == result.pmp_build.doc


def test_normalized_upper_preserves_offsets_through_sign_change_and_identity_shift():
    from virasoro_bootstrap.controls.global_control import build_global_identical_scalar_control_problem

    source = spec(delta_gap='.1', large_spin_lmin=None, large_spin_twists=())
    with mp.workdps(30):
        base = build_global_identical_scalar_control_problem(source, global_identity=True)
        numerator = moments.MomentWeight((moments.MomentTerm(1 + EPS),))
        vacuum = literal_number(base.pmp_metadata['self_dual_identity_measure'])
        result = moments.build_identical_scalar_normalized_moment_upper_problem(
            source, numerator, moments.MomentWeight.one(), alpha=-vacuum + EPS, beta=-vacuum,
            base_problem=base, block_basis='ratio')
        scalar = result.pmp_build.doc['PositiveMatrixWithPrefactorArray'][-1]['polynomials'][0][0]
        assert_relative(mp.mpf(scalar[-2][0]), -vacuum * EPS)
        assert_relative(mp.mpf(scalar[-1][0]), EPS)
        meta = result.pmp_metadata['moment']
        assert literal_number(meta['numerator']['terms'][0]['coefficient']) == 1 + EPS
        assert literal_number(meta['alpha']) == -vacuum + EPS
        restored = moments.rebuild_moment_problem_from_blocks(source, meta, result.problem_kind, [], base_problem=base)
        assert restored.pmp_build.doc == result.pmp_build.doc


def _stored_moment_problem():
    from virasoro_blocks.polynomial import Polynomial
    from virasoro_blocks.serialization import SerializedDerivTable
    from virasoro_bootstrap.identical_scalar import IdenticalScalarGapProblem, IdenticalScalarSpinChannel
    from virasoro_bootstrap.pmp_writer import PrefactoredPmpBuild

    source = spec(delta_gap=0, large_spin_lmin=None, large_spin_twists=())
    with mp.workdps(120):
        slope = 1+mp.mpf('1e-70')
        table = SerializedDerivTable(0, mp.mpf('.5'), (), ('zzb',),
            {'zzb': [[Polynomial.one()], [Polynomial([-1, slope])]]},
            sample_points_x=(0, 1), working_dps=120)
    channel = IdenticalScalarSpinChannel(0, 0, table, table)
    problem = IdenticalScalarGapProblem(source, 'synthetic', False, 2*source.h_ext,
        ((1, 0),), (0,), (channel,), PrefactoredPmpBuild({}, (), ()))
    return problem, table, slope


def test_ratio_pmp_pads_exact_low_degree_components_to_the_sample_count(tmp_path):
    from virasoro_blocks.polynomial import Polynomial
    from virasoro_bootstrap.sdp_writer import write_sdp_directory

    problem, table, _ = _stored_moment_problem()
    extras = (Polynomial.zero(), Polynomial([3]), Polynomial([0, 1]))
    with mp.workdps(30):
        for primary in (Polynomial.zero(), Polynomial.one(), Polynomial([1, 1])):
            channel = replace(problem.spin_channels[0], prefactored=replace(table,
                numerators={'zzb': [[Polynomial.one()], [primary]]}, sample_points_x=(0, '.5', 1)))
            build, values = moments._sampled_moment_pmp_doc(problem, channels=[channel],
                extra_components_by_channel=lambda c: extras, objective=[0]*4,
                normalization=[1, 0, 0, 0], sig_figs=None, block_basis='ratio')
            block = build.doc['PositiveMatrixWithPrefactorArray'][0]
            components = block['polynomials'][0][0]
            assert len(block['samplePoints']) == 3
            assert all(len(coeffs) == 3 for coeffs in components)
            for a, (coeffs, poly) in enumerate(zip(components, (primary, *extras))):
                for k, x in enumerate((0, mp.mpf('.5'), 1)):
                    actual = sum(mp.mpf(coef) * x**i for i, coef in enumerate(coeffs))
                    assert actual == poly.evaluate(x)
                    assert actual == values[0][k][a]
            # Both routes consume the same padded polynomial and clean values.
            write_sdp_directory(build.doc, tmp_path/'polynomial', eval_dps=80)
            write_sdp_directory(build.doc, tmp_path/'clean', eval_dps=80, sample_values_by_block=values)
            assert (tmp_path/'polynomial'/'block_data_0.json').read_text() == (
                tmp_path/'clean'/'block_data_0.json').read_text()


def test_moment_polynomial_exports_validate_complex_values_and_restore_formatting():
    from virasoro_blocks.polynomial import Polynomial

    problem, _, _ = _stored_moment_problem()
    with mp.workdps(30):
        complex_poly = Polynomial([mp.mpc(1, '.1')], field=mp.mpc)
        with pytest.raises(ValueError, match='PMP entries must be real'):
            moments._pmp_poly_coeffs(complex_poly, sig_figs=20)
        for basis in ('ratio', 'nonratio'):
            with pytest.raises(ValueError, match='PMP entries must be real'):
                moments._sampled_moment_pmp_doc(problem, channels=problem.spin_channels,
                    extra_components_by_channel=lambda channel: (complex_poly,), objective=[0, 0],
                    normalization=[1, 0], sig_figs=None, block_basis=basis)
            assert mp.mp.dps == 30


def test_primary_moment_samples_reject_material_complex_and_nonfinite_parts(monkeypatch):
    from virasoro_blocks.polynomial import Polynomial

    problem, table, _ = _stored_moment_problem()
    with mp.workdps(30):
        for path in ('polynomial', 'stored', 'direct'):
            for cell in (1,):
                for bad in (mp.mpc(1, '.1'), mp.mpc(1, mp.inf)):
                    if path == 'polynomial' and not mp.isfinite(bad):
                        continue   # polynomial evaluation already refuses nonfinite coefficients
                    rows = [[Polynomial.one()], [Polynomial.one()]]
                    raw = [[mp.mpf(1)], [mp.mpf(1)]]
                    if path == 'polynomial':
                        rows[cell][0] = Polynomial([bad], field=mp.mpc)
                    else:
                        raw[cell][0] = bad
                    sampled = replace(table, numerators={'zzb': rows},
                        sample_block_values={'zzb': [raw, raw]} if path == 'stored' else None)
                    channel = replace(problem.spin_channels[0], prefactored=sampled)
                    monkeypatch.setattr(moments, '_direct_control_table_at',
                        (lambda *args: raw) if path == 'direct' else (lambda *args: None))
                    for basis in ('ratio',):
                        with pytest.raises(ValueError, match='PMP entries must be (real|finite)'):
                            moments._sampled_moment_pmp_doc(problem, channels=[channel],
                                extra_components_by_channel=lambda channel: (Polynomial.one(),),
                                objective=[0, 0], normalization=[1, 0], sig_figs=None, block_basis=basis)
                        assert mp.mp.dps == 30


def test_primary_moment_validation_accepts_roundoff_and_retains_sample_bits(monkeypatch):
    from virasoro_blocks.polynomial import Polynomial

    problem, table, slope = _stored_moment_problem()
    with mp.workdps(120):
        noise = mp.mpf('1e-140')
        rows = {'zzb': [[Polynomial([mp.mpc(1, noise)], field=mp.mpc)],
                        [Polynomial([mp.mpc(-1, noise), slope], field=mp.mpc)]]}
        def grid_at(x):
            scale = mp.mpf('.5')**x
            return [[mp.mpc(scale, noise*scale)],
                    [mp.mpc((-1+slope*x)*scale, noise*scale)]]
        raw = [grid_at(0), grid_at(1)]
    with mp.workdps(30):
        for path in ('polynomial', 'stored', 'direct'):
            sampled = replace(table, numerators=rows,
                sample_block_values={'zzb': raw} if path == 'stored' else None)
            channel = replace(problem.spin_channels[0], prefactored=sampled)
            monkeypatch.setattr(moments, '_direct_control_table_at',
                (lambda problem, channel, x: grid_at(x)) if path == 'direct' else (lambda *args: None))
            for basis in ('ratio', 'nonratio'):
                build, values = moments._sampled_moment_pmp_doc(problem, channels=[channel],
                    extra_components_by_channel=lambda channel: (Polynomial.one(),),
                    objective=[0, 0], normalization=[1, 0], sig_figs=None, block_basis=basis)
                assert mp.mp.dps == 30
                with mp.workdps(200):
                    block = build.doc['PositiveMatrixWithPrefactorArray'][0]
                    coefs = block['polynomials'][0][0][0]
                    assert abs(sum(mp.mpf(s) for s in coefs)/(slope-1)-1) < mp.mpf('1e-27')
                    if path != 'polynomial' or basis == 'ratio':
                        scaling = mp.mpf(block['sampleScalings'][1])
                        expected = slope-1 if basis == 'ratio' else (slope-1)/2/scaling
                        assert abs(values[0][1][0]/expected-1) < mp.mpf('1e-27')
    # The shared formatter also accepts real Python complex inputs without
    # rounding their binary components to a low ambient precision first.
    value = complex(.12345678901234567, 0)
    with mp.workdps(3):
        text = moments._pmp_real_str(value, 20, where='sample')
        assert mp.mp.dps == 3
    with mp.workdps(60):
        expected = mp.convert(literal_components(value)[0])
        assert abs(mp.mpf(text)-expected) < mp.mpf('1e-20')


@pytest.mark.parametrize('direction', ['upper'])
def test_ratio_probes_apply_distinct_moment_filters_per_channel(direction):
    from virasoro_bootstrap.controls.global_control import build_global_identical_scalar_control_problem

    source = spec(delta_gap='.1', spin_list=(0, 2), large_spin_lmin=None, large_spin_twists=())
    with mp.workdps(30):
        base = build_global_identical_scalar_control_problem(source, global_identity=True)
        builder = (moments.build_identical_scalar_normalized_ratio_probe_problem if direction == 'upper'
                   else moments.build_identical_scalar_normalized_ratio_lower_probe_problem)
        numerator = moments.MomentWeight((moments.MomentTerm(1),), spin_filter=0)
        denominator = moments.MomentWeight((moments.MomentTerm(1),), spin_filter=2)
        result = builder(source, numerator, denominator, 3, identity_mode='exclude',
            base_problem=base, block_basis='ratio')
        sign = -1 if direction == 'upper' else 1
        components = result.pmp_build.doc['PositiveMatrixWithPrefactorArray']
        assert [mp.mpf(block['polynomials'][0][0][-1][0]) for block in components] == [sign, -3 * sign]
        assert result.pmp_metadata['moment']['numerator']['spin_filter'] == 0
        assert result.pmp_metadata['moment']['denominator']['spin_filter'] == 2


def spec(**updates):
    values = dict(c='2',h_ext='.3',delta_gap=0,lam=1,spin_list=(0,),
        q_order=2,npoles=0,dps=30,num_threads=1,
        large_spin_lmin=2,large_spin_twists=(ExactRational(1,3),))
    values.update(updates)
    return IdenticalScalarGapSpec(**values)


@pytest.mark.parametrize('field', ['h_ext', 'delta_min_by_spin'])
def test_subprecision_label_changes_refuse_base_problem_reuse(field):
    original = spec()
    offset = ExactRational(1,10**70)
    value = {'h_ext': original.h_ext+offset,
             'large_spin_twists': (original.large_spin_twists[0]+offset,),
             'delta_gap': offset, 'delta_min_by_spin': {0: offset}}[field]
    if field == 'delta_min_by_spin':
        original = spec(spin_list=(0, 2))
        value = {2: 2 + offset}
    changed = replace(original, **{field:value})
    with mp.workdps(15), pytest.raises(ValueError, match=field):
        moments._resolve_base_problem(changed,None,SimpleNamespace(spec=original, scalar_G_spin0=scalar_G_spin0_convention()))


def test_equivalent_decimal_and_fraction_labels_can_reuse_base_problem():
    original = spec(h_ext=0.7,large_spin_twists=('.7',))
    requested = spec(h_ext=Fraction(7,10),prefactor_exponent='.7',
                     large_spin_twists=(Fraction(7,10),), delta_min_by_spin={0: 0})
    base = SimpleNamespace(spec=original, scalar_G_spin0=scalar_G_spin0_convention())
    assert moments._resolve_base_problem(requested,None,base) is base


@pytest.mark.parametrize('mode', ['numeric_virasoro_control'])
def test_control_resampling_keeps_edge_offsets_and_backend_labels(monkeypatch, mode):
    from virasoro_bootstrap.controls import global_control as control
    label = Fraction(7, 10) + EPS
    p = SimpleNamespace(spec=spec(h_ext=label, c=2 + EPS), resolved_method=mode)
    channel = SimpleNamespace(delta_min=2 + EPS, ell=2)
    calls = []
    def builder(delta, ell, ext, *args, **kwargs):
        calls.append((delta, ext, kwargs))
        return [[1]]
    dispatch = control._GLOBAL_CONTROL_BUILDERS if mode == 'global_control' else control._NUMERIC_CONTROL_BUILDERS
    monkeypatch.setitem(dispatch, p.spec.coordinate, builder)
    with mp.workdps(15):
        assert moments._direct_control_table_at(p, channel, EPS) == [[1]]
    delta, ext, kwargs = calls[0]
    assert literal_number(delta) == 2 + 2*EPS
    assert all(literal_number(value) == label for value in ext)
    assert literal_number(kwargs['z_shift']) == -2*label
    if mode != 'global_control':
        assert literal_number(kwargs['c']) == 2 + EPS


@pytest.fixture
def rebuild(monkeypatch):
    captured = []
    def builder(new_spec, *args, **kwargs):
        captured.append(new_spec)
        return SimpleNamespace(spec=new_spec)
    monkeypatch.setattr(moments,'_NORMALIZED_MOMENT_BUILDERS',{'upper':builder})
    monkeypatch.setattr(moments,'_cached_base_problem',lambda value: SimpleNamespace(spec=value))
    problem = SimpleNamespace(spec=spec(),pmp_metadata={'moment':dict(kind='normalized',direction='upper',
        numerator=moments.MomentWeight.one().to_json(),denominator=moments.MomentWeight.one().to_json())})
    return problem, captured


@pytest.mark.parametrize('label', [ExactRational(7,10)+ExactRational(1,10**70)])
def test_explicit_rebuild_twists_keep_exact_labels(rebuild, label):
    problem,captured = rebuild
    moments.rebuild_normalized_moment_problem(problem,twists=[label])
    assert literal_number(captured[0].large_spin_twists[0]) == Fraction(str(label))


@pytest.mark.parametrize('overrides', [dict(n_twists=1.5), dict(spin_list=[0,False])])
def test_rebuild_controls_do_not_silently_truncate(rebuild, overrides):
    problem,captured = rebuild
    with pytest.raises(ValueError):
        moments.rebuild_normalized_moment_problem(problem,**overrides)
    assert captured == []
