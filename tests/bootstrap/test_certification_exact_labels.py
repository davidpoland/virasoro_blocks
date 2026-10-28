"""Certification retains the theories and tower slices it was asked to test."""

from decimal import Decimal
from fractions import Fraction
import json
from types import SimpleNamespace

import mpmath as mp
import pytest

from virasoro_blocks.numeric_input import literal_number
from virasoro_blocks.polyderivs import _NumericBlockSampler
from virasoro_bootstrap import certification as C
from virasoro_bootstrap.identical_scalar import IdenticalScalarGapSpec, IdenticalScalarSpinChannel


def problem(h=Fraction(1, 10)+Fraction(1, 10**60), twists=('.6', '1.6')):
    spec = IdenticalScalarGapSpec(c=32, h_ext=h, delta_gap='1.1', lam=1,
        spin_list=(0,), q_order=4, dps=30, npoles=2, large_spin_lmin=4,
        large_spin_twists=twists, num_threads=1)
    ch = IdenticalScalarSpinChannel(ell=0, delta_min=1, serialized=None, prefactored=None)
    return SimpleNamespace(spec=spec, spin_channels=[ch], resolved_method='interpolation')


def test_exact_channels_keep_external_labels_and_precision_on_reuse():
    with mp.workdps(30):
        p = problem()
        channels = C._build_exact_block_channels(p)
    h = p.spec.h_ext
    assert literal_number(channels[0].prefactored._sampler.h_externals[0]) == h
    for ch in channels[1:]:
        sampler = ch.prefactored._sampler
        assert literal_number(sampler.h_ext[0]) == h
        assert literal_number(sampler.z_shift) == -2*h
    with mp.workdps(60):
        actual = channels[0].prefactored.evaluate('zzb', 3)[0][0]
        ref = _NumericBlockSampler('zzb', [h]*4, 0, 1, 1, b=None, c=32,
            q_order=4, chiral_norm=p.spec.chiral_norm, z_shift=-2*h)([3])[0][0][0][0]
        assert abs(actual-ref) < mp.mpf('1e-55')*abs(ref)


def test_extra_twist_channels_accept_fractions_and_keep_derived_labels():
    lo = Fraction(3, 5)+Fraction(1, 10**60)
    hi = Fraction(8, 5)+Fraction(1, 10**60)
    with mp.workdps(30):
        p = problem(twists=(lo, hi))
        channels = C._build_extra_twist_channels(p, 1)
    assert len(channels) == 2
    for ch, want in zip(channels, [(lo+hi)/2, hi+(hi-lo)/2]):
        sampler = ch.prefactored._sampler
        assert literal_number(sampler.twist) == want
        assert literal_number(sampler.h_ext[0]) == p.spec.h_ext
        assert literal_number(sampler.z_shift) == -2*p.spec.h_ext
        assert literal_number(ch.twist) == want


def test_extra_spin_builder_keeps_a_subprecision_common_twist_floor(monkeypatch):
    from virasoro_bootstrap.controls import global_control as control
    p = problem(twists=())
    floor = Fraction(1, 10**70)
    p.resolved_method = 'global_control'
    p.spin_channels = [SimpleNamespace(ell=ell, delta_min=ell+floor) for ell in (2, 4)]
    calls = []
    monkeypatch.setattr(control, 'build_global_scalar_spin_channel', lambda **kw: calls.append(kw))
    with mp.workdps(15):
        C._build_extra_spin_channels(p, 1)
    assert len(calls) == 1
    assert literal_number(calls[0]['delta_min']) == 6 + floor


@pytest.mark.parametrize('wrapper, form', [
    (C._ExactFixedSpinPrefactored, 'mpf'), (C._ExactTowerPrefactored, 'string'),
])
def test_true_block_wrappers_preserve_exact_query_offsets(wrapper, form):
    value = Fraction(2)+Fraction(1, 10**100)
    if form == 'mpf':
        with mp.workdps(150):
            value = mp.convert(literal_number(value))
    elif form == 'string':
        value = str(value)
    expected = literal_number(value)
    def sample(query):
        if wrapper is C._ExactFixedSpinPrefactored:
            assert literal_number(query[0]) == expected
            return [[[1]]], True
        assert literal_number(query) == expected
        return [[1]]
    wrapped = wrapper(sample, {(0, 0)}, value) if wrapper is C._ExactFixedSpinPrefactored else wrapper(sample, value)
    assert literal_number(wrapped.shift) == expected
    with mp.workdps(15):
        assert wrapped.evaluate('zzb', value) == [[1]]


@pytest.mark.parametrize('wrapper, value', [
    (C._ExactFixedSpinPrefactored, mp.nan), (C._ExactTowerPrefactored, complex(2, 1)),
])
def test_true_block_wrappers_reject_invalid_queries_before_sampling(wrapper, value):
    def unexpected(query):
        pytest.fail('invalid query reached the sampler')
    wrapped = wrapper(unexpected, {(0, 0)}, 0) if wrapper is C._ExactFixedSpinPrefactored else wrapper(unexpected, 0)
    with pytest.raises(ValueError):
        wrapped.evaluate('zzb', value)


@pytest.mark.parametrize('form', ['string', 'mpf'])
def test_subprecision_slices_and_interstitial_lines_are_distinct(form):
    lo, hi = Fraction(3, 5), Fraction(3, 5)+Fraction(1, 10**60)
    if form == 'string':
        inputs = ('.6', '0.6'+'0'*58+'1')
    elif form == 'decimal':
        inputs = (Decimal('.6'), Decimal('0.6'+'0'*58+'1'))
    elif form == 'mpf':
        with mp.workdps(150):
            inputs = (mp.mpf('.6'), mp.mpf('.6')+mp.mpf('1e-60'))
    else:
        inputs = (lo, hi)
    lo, hi = map(literal_number, inputs)
    spec = SimpleNamespace(large_spin_twists=inputs)
    with mp.workdps(15):
        placed = C._interstitial_twist_set(spec, 1)
    assert list(map(literal_number, placed)) == [lo, (lo+hi)/2, hi, hi+(hi-lo)/2, hi+(hi-lo)]


def test_single_base_slice_uses_exact_default_gap():
    base = Fraction(1, 3)
    with mp.workdps(15):
        placed = C._interstitial_twist_set(SimpleNamespace(large_spin_twists=(base,)), 2)
    assert list(map(literal_number, placed)) == [base, base+Fraction(2, 3), base+Fraction(4, 3), base+2]


@pytest.mark.parametrize('source', [Fraction(11, 10)+Fraction(1, 10**60)])
def test_gap_certificate_details_preserve_exact_slice_labels(source):
    ch = SimpleNamespace(ell=4, twist=source, delta_min=4,
        prefactored=SimpleNamespace(evaluate=lambda coordinate, point: [[1]]))
    with mp.workdps(15):
        ok, details = C.certify_dual_functional([1], [ch], [(0, 0)], n_test_points=2)
        detail = json.loads(json.dumps(details))[0]
    assert ok and literal_number(detail['twist_label']) == source
    assert detail['twist'] == float(source)


def test_large_finite_twist_metadata_remains_json_serializable():
    source = Fraction(10**500)
    ch = SimpleNamespace(ell=4, twist=source, delta_min=4,
        prefactored=SimpleNamespace(evaluate=lambda coordinate, point: [[1]]))
    with mp.workdps(15):
        ok, details = C.certify_dual_functional([1], [ch], [(0, 0)], n_test_points=2)
        detail = json.loads(json.dumps(details, allow_nan=False))[0]
    assert ok and literal_number(detail['twist_label']) == source
    assert literal_number(detail['twist']) == source


def test_large_finite_scan_coordinate_remains_json_serializable():
    source = Fraction(10**500)
    ch = SimpleNamespace(ell=0, delta_min=source,
        prefactored=SimpleNamespace(evaluate=lambda coordinate, point: [[-1]]))
    ok, details = C.certify_dual_functional([1], [ch], [(0, 0)], n_test_points=1)
    detail = json.loads(json.dumps(details, allow_nan=False))[0]
    assert not ok
    assert literal_number(detail['min_delta']) == literal_number(detail['min_delta_label']) == source


@pytest.mark.parametrize('kind', ['pmp', 'physical'])
def test_extra_tower_coverage_preserves_gap_selected_slices_and_checks_above_the_top(monkeypatch, kind):
    import virasoro_bootstrap.moments as M
    from virasoro_bootstrap.large_spin import double_twist_set

    eps = Fraction(1, 10**70)
    original = (Fraction(102, 5) + eps, Fraction(112, 5) + eps)
    spec = SimpleNamespace(spin_list=(0,), h_ext='.1', large_spin_lmin=2, large_spin_twists=original)
    built = []
    def rebuild(problem, *, spin_list, n_twists=None, twists=None):
        labels = tuple(twists) if twists is not None else (
            tuple(double_twist_set(spec.h_ext, n_twists)) if n_twists is not None else original)
        built.append(tuple(map(literal_number, labels)))
        channels = [SimpleNamespace(ell=2, twist=t) for t in labels]
        values = ['-1' if literal_number(t) > original[-1] else '1' for t in labels]
        blocks = [{'polynomials': [[[[value]]]], 'samplePoints': ['0'],
                   'DampedRational': {'constant': '1', 'base': '.5', 'poles': []}} for value in values]
        return SimpleNamespace(spec=spec, spin_channels=channels,
            pmp_build=SimpleNamespace(doc={'PositiveMatrixWithPrefactorArray': blocks}),
            sample_values_by_block={i: [[value]] for i, value in enumerate(values)})
    monkeypatch.setattr(M, 'rebuild_normalized_moment_problem', rebuild)
    if kind == 'physical':
        monkeypatch.setattr(C, '_build_exact_block_channels', lambda p: p.spin_channels)
        monkeypatch.setattr(C, '_normalized_moment_cert_weights', lambda p: [])
        monkeypatch.setattr(C, '_normalized_moment_physical_channel_detail',
            lambda y, p, ch, **kw: {'passed': literal_number(ch.twist) <= original[-1]})
    kwargs = dict(extra_spins=0, extra_twists=2)
    if kind != 'sample':
        kwargs.update(n_test_points=2, delta_range=1)
    if kind == 'physical':
        kwargs.update(cert_q_order=None)
    with mp.workdps(15):
        ok, details = getattr(C, '_certify_normalized_moment_extra_' + kind + '_blocks')(
            [1], SimpleNamespace(spec=spec), **kwargs)
    expected = (Fraction(122, 5), Fraction(132, 5))
    assert not ok
    assert built == [original + expected]
    assert tuple(literal_number(d['twist_label']) for d in details) == expected


@pytest.mark.parametrize('kind', ['interstitial'])
def test_repeated_top_slice_does_not_eliminate_above_top_coverage(kind):
    # the spec refuses a repeated twist; certification must still cope with one in a spec-like object
    import dataclasses

    p = problem(twists=('.6', '1.6'))
    fields = {f.name: getattr(p.spec, f.name) for f in dataclasses.fields(p.spec)}
    p.spec = SimpleNamespace(**dict(fields, large_spin_twists=('.6', '1.6', '1.6')))
    with mp.workdps(15):
        if kind == 'interstitial':
            actual = C._interstitial_twist_set(p.spec, 1)
            expected = [Fraction(3, 5), Fraction(11, 10), Fraction(8, 5), Fraction(21, 10), Fraction(13, 5)]
        else:
            actual = [ch.twist for ch in C._build_extra_twist_channels(p, 1)]
            expected = [Fraction(11, 10), Fraction(21, 10)]
    assert list(map(literal_number, actual)) == expected


@pytest.mark.parametrize('base,expected', [
    (('2.4', '.4'), [Fraction(2, 5), Fraction(12, 5), Fraction(22, 5), Fraction(32, 5)]),
    ((10**80,), [10**80, Fraction(10**80) + Fraction(2, 5), Fraction(10**80) + Fraction(12, 5)]),
])
def test_extra_twist_placement_matches_canonical_sequence_without_building_its_prefix(base, expected):
    spec = SimpleNamespace(h_ext=0.1, large_spin_lmin=2, large_spin_twists=base)
    with mp.workdps(15):
        actual = C._normalized_moment_extra_twist_set(spec, 2, 0)
    assert list(map(literal_number, actual)) == expected


@pytest.mark.parametrize('side', [-1, 1])
def test_extra_twist_placement_uses_exact_side_of_a_canonical_slice(side):
    with mp.workdps(150):
        top = mp.convert(literal_number(Fraction(112, 5) + side * Fraction(1, 10**70)))
    spec = SimpleNamespace(h_ext='.1', large_spin_lmin=2, large_spin_twists=(top,))
    with mp.workdps(15):
        actual = C._normalized_moment_extra_twist_set(spec, 1, 0)
    expected_next = Fraction(112 if side < 0 else 122, 5)
    assert list(map(literal_number, actual)) == [literal_number(top), expected_next]


@pytest.mark.parametrize('kind, skip_kind', [('pmp', 'original'), ('physical', 'checked')])
def test_extra_slice_is_not_skipped_due_to_roundoff(monkeypatch, kind, skip_kind):
    import virasoro_bootstrap.moments as M
    original = '.6'
    new = '.6'+'0'*59+'1' if skip_kind == 'original' else '1.6'+'0'*59+'2'
    skip = () if skip_kind == 'original' else ('1.6'+'0'*59+'1',)
    spec = SimpleNamespace(spin_list=(0,), h_ext='.1', large_spin_lmin=4, large_spin_twists=(original,))
    channels = [SimpleNamespace(ell=4, twist=original), SimpleNamespace(ell=4, twist=new)]
    def entry(coefficient):
        return {'polynomials': [[[[coefficient]]]], 'samplePoints': ['0'],
            'DampedRational': {'constant': '1', 'base': '.5', 'poles': []}}
    ext = SimpleNamespace(spec=spec, spin_channels=channels,
        pmp_build=SimpleNamespace(doc={'PositiveMatrixWithPrefactorArray': [entry('1'), entry('-1')]}),
        sample_values_by_block={0: [['1']], 1: [['-1']]})
    monkeypatch.setattr(M, 'rebuild_normalized_moment_problem', lambda *args, **kwargs: ext)
    if kind == 'physical':
        monkeypatch.setattr(C, '_build_exact_block_channels', lambda p: channels)
        monkeypatch.setattr(C, '_normalized_moment_cert_weights', lambda p: [])
        monkeypatch.setattr(C, '_normalized_moment_physical_channel_detail',
            lambda y, p, ch, **kw: {'ell': ch.ell, 'twist': float(mp.mpf(ch.twist)), 'passed': False})
    kwargs = dict(extra_spins=0, extra_twists=1, skip_twists=skip,
        interstitial_twists=1 if skip_kind == 'checked' else 0)
    if kind != 'sample':
        kwargs.update(n_test_points=2, delta_range=1)
    if kind == 'physical':
        kwargs.update(cert_q_order=None)
    fn = getattr(C, '_certify_normalized_moment_extra_'+kind+'_blocks')
    with mp.workdps(15):
        ok, details = fn([1], SimpleNamespace(spec=spec), **kwargs)
    assert not ok and len(details) == 1
    assert literal_number(details[0]['twist_label']) == literal_number(new)


@pytest.mark.parametrize('kind', ['sample'])
def test_adaptive_stages_keep_exact_checked_slice_labels(monkeypatch, kind):
    a, b = Fraction(11, 10)+Fraction(1, 10**60), Fraction(11, 10)+Fraction(2, 10**60)
    seen = []
    def extras(*args, **kwargs):
        stage = kwargs['adaptive_stage']
        seen.append(stage)
        if stage == 'initial':
            label = a
        else:
            expected = {a} if stage == 'spin_wide' else {a, b}
            assert set(map(literal_number, kwargs['skip_twists'])) == expected
            label = b
        return True, [{'ell': 4, 'twist': 1.1, 'twist_label': str(label), 'passed': True}]
    monkeypatch.setattr(C, '_certify_normalized_moment_extra_'+kind+'_blocks', extras)
    fn = getattr(C, '_certify_normalized_moment_adaptive_'+kind+'_blocks')
    kwargs = dict(extra_spins=0, extra_twists=1, max_extra_spins=1, max_extra_twists=2)
    if kind != 'sample':
        kwargs.update(n_test_points=2, delta_range=1)
    if kind == 'physical':
        kwargs.update(cert_q_order=None)
    with mp.workdps(15):
        assert fn([1], None, **kwargs)[0]
    assert seen == ['initial', 'spin_wide', 'twist_wide']


@pytest.mark.parametrize('stored', [True])
def test_scan_checks_the_exact_threshold_and_requested_endpoint(stored):
    start, end = Fraction(1)+Fraction(1, 10**70), Fraction(30)+Fraction(1, 10**60)
    if stored:
        with mp.workdps(150):
            start, end = mp.convert(literal_number(start)), mp.convert(literal_number(end))
    lo, width = literal_number(start), literal_number(end)
    eps = width-30
    seen = []
    def evaluate(coordinate, point):
        point = literal_number(point)
        seen.append(point)
        return [[lo+width-point-eps/2]]
    ch = SimpleNamespace(ell=0, delta_min=start, prefactored=SimpleNamespace(evaluate=evaluate))
    with mp.workdps(15):
        ok, details = C.certify_dual_functional([1], [ch], [(0, 0)], n_test_points=2, delta_range=end)
    assert seen == [lo, lo+width]
    assert not ok and details[0]['n_negative'] == 1
    assert literal_number(details[0]['min_delta_label']) == lo+width


def test_small_physical_grid_includes_both_requested_edges():
    end = Fraction(1, 3)+Fraction(1, 10**60)
    with mp.workdps(15):
        points = C._channel_test_offsets(2, end)
    assert list(map(literal_number, points)) == [0, end]


@pytest.mark.parametrize('end', [0, Fraction(1, 3)])
def test_log_grid_stays_inside_zero_tiny_and_nonterminating_ranges(end):
    with mp.workdps(15):
        points = C._channel_test_offsets(4, end, log_near_points=5)
    labels = list(map(literal_number, points))
    assert labels[0] == 0 and labels[-1] == end
    assert all(0 <= point <= end for point in labels)


def test_pmp_grid_retains_close_samples_and_the_fallback_endpoint():
    lo, hi = Fraction(1), Fraction(1)+Fraction(1, 10**60)
    with mp.workdps(15):
        points = C._pmp_test_points({'samplePoints': [str(lo), str(hi)]}, n_test_points=2, delta_range=hi)
        fallback = C._pmp_test_points({}, n_test_points=2, delta_range=hi)
    assert list(map(literal_number, points)) == [lo, (lo+hi)/2, hi]
    assert list(map(literal_number, fallback)) == [0, hi]


def test_pmp_grid_preserves_the_sign_after_exact_polynomial_contraction():
    eps = Fraction(1, 10**60)
    end = literal_number(Fraction(30)+eps)
    entry = {'polynomials': [[[[str(Fraction(30)+eps/2)], ['0', '-1']]]]}
    with mp.workdps(15):
        detail = C._pmp_channel_detail([1, 1], entry, block_index=0,
            n_test_points=2, delta_range=end)
    assert not detail['passed'] and detail['n_negative'] == 1
    assert literal_number(detail['min_x_label']) == end


def test_normalized_scalar_constraint_keeps_a_tiny_negative_constant():
    eps = Fraction(1, 10**60)
    entry = {'polynomials': [[[[str(Fraction(7, 10)+eps)], ['.7']]]], 'samplePoints': ['0']}
    p = SimpleNamespace(pmp_build=SimpleNamespace(doc={'PositiveMatrixWithPrefactorArray': [entry]}))
    with mp.workdps(15):
        detail = C._certify_normalized_moment_scalar_block([-1, 1], p, block_index=0)
    assert not detail['passed'] and detail['n_negative'] == 1


def test_normalized_scalar_constraint_refuses_a_nonconstant_block():
    entry = {'polynomials': [[[['1', '-1']]]], 'samplePoints': ['0']}
    p = SimpleNamespace(pmp_build=SimpleNamespace(doc={'PositiveMatrixWithPrefactorArray': [entry]}))
    detail = C._certify_normalized_moment_scalar_block([1], p, block_index=0)
    assert not detail['passed']
    assert detail['reason'] == 'nonconstant_normalized_scalar_block'


def test_physical_batch_checks_a_negative_exact_endpoint():
    from virasoro_bootstrap.moments import MomentTerm, MomentWeight
    lo, width = Fraction(1)+Fraction(1, 10**70), Fraction(30)+Fraction(1, 10**60)
    weight = MomentWeight((MomentTerm(lo+width-Fraction(1, 2*10**60), {}), MomentTerm(-1, {'Delta': 1})))
    seen = []
    def bucket(values, entries):
        assert entries == [(0, 0)]
        seen.extend(map(literal_number, values))
        return [[[1]] for value in values], False
    ch = SimpleNamespace(ell=0, delta_min=lo,
        prefactored=SimpleNamespace(shift=lo, _sampler=SimpleNamespace(sample_bucket=bucket)))
    p = SimpleNamespace(spec=SimpleNamespace(coordinate='zzb'), F_pairs=[(0, 0)],
        pmp_metadata={'moment': {'block_basis': 'ratio'}})
    with mp.workdps(15):
        detail = C._normalized_moment_physical_channel_detail([0, 1, 0], p, ch,
            block_index=0, n_test_points=2, delta_range=width,
            weights=[weight, MomentWeight.one()], extra=False)
    assert seen == [lo, lo+width]
    assert not detail['passed'] and detail['n_negative'] == 1
    assert literal_number(detail['min_x_label']) == width


@pytest.mark.parametrize('tower, basis', [(False, 'ratio'), (True, 'nonratio')])
def test_physical_moment_weights_keep_exact_quantum_number_cancellations(tower, basis):
    from virasoro_bootstrap.moments import MomentTerm, MomentWeight
    eps = Fraction(1, 10**70)
    if tower:
        ch = SimpleNamespace(ell=4, twist=eps, delta_min=4, prefactored=SimpleNamespace(shift=4))
        value = literal_number(Fraction(4)+eps)
        weight = MomentWeight((MomentTerm(1, {'Delta': 1}), MomentTerm(-1, {'L': 1})))
        expected = eps
    else:
        ch = SimpleNamespace(ell=2, delta_min=0, prefactored=SimpleNamespace(shift=0))
        value = literal_number(Fraction(2)+eps)
        weight = MomentWeight.monomial(hbar=2)
        expected = (eps/2)**2
    p = SimpleNamespace(F_pairs=[(0, 0)], pmp_metadata={'moment': {'block_basis': basis}})
    with mp.workdps(30):
        actual = C._normalized_moment_physical_row_from_table([0, 1], p, ch, value, [[2]], weights=[weight])
        reference = mp.convert(literal_number(expected*(1 if basis == 'ratio' else 2)))
        assert abs(actual/reference-1) < mp.mpf('1e-27')


def test_physical_moment_contraction_preserves_weight_and_measure_cancellation():
    from flint import ctx
    from virasoro_blocks.polynomial import _mpf_to_arb
    from virasoro_bootstrap.moments import MomentTerm, MomentWeight

    with mp.workdps(180), ctx.workdps(180):
        height = mp.mpf(2)**250
        smaller = height-1
        arb_table = [[_mpf_to_arb(height)], [_mpf_to_arb(smaller)]]
    channel = SimpleNamespace(ell=0, delta_min=0)
    eps = Fraction(1, 2**250)
    weight = MomentWeight((MomentTerm(1+eps, {}),))
    with mp.workdps(40):
        for basis in ('nonratio', 'ratio'):
            p = SimpleNamespace(spec=SimpleNamespace(coordinate='zzb'),
                F_pairs=[(1, 0)], pmp_metadata={'moment': {'block_basis': basis}})
            got = C._normalized_moment_physical_row_from_table(
                [1, -1], p, channel, 0, [[height], [height]], weights=[weight])
            expected = mp.mpf(-1 if basis == 'nonratio' else -literal_number(eps))
            assert abs(got/expected-1) < mp.mpf('1e-37')
            # The ratio division and nonratio measure product must also happen
            # after contraction; native Arb midpoints keep their supplied bits.
            for table in ([[height], [smaller]], arb_table):
                got = C._normalized_moment_physical_row_from_table(
                    [1, -1], p, channel, 0, table, weights=[MomentWeight.one()])
                assert abs(got/expected-1) < mp.mpf('1e-37')
            channel.prefactored = SimpleNamespace(evaluate=lambda coordinate, value: [[height], [height]])
            detail = C._normalized_moment_physical_channel_detail([1, -1], p, channel,
                block_index=0, n_test_points=2, delta_range=1, weights=[weight], extra=False)
            assert not detail['passed'] and detail['n_negative'] == 2
            precisions = []
            def bucket(values, entries):
                precisions.append((mp.mp.dps, len(values)))
                return [[[+height], [+smaller]] for _ in values], False
            channel.prefactored = SimpleNamespace(_sampler=SimpleNamespace(sample_bucket=bucket))
            detail = C._normalized_moment_physical_channel_detail([1, -1], p, channel,
                block_index=0, n_test_points=3, delta_range=1,
                weights=[MomentWeight.one()], extra=False)
            assert not detail['passed'] and detail['n_negative'] == 3
            assert precisions[0] == (40, 3)
            assert len(precisions) == 2 and precisions[1][0] > 100 and precisions[1][1] == 3
        assert mp.mp.dps == 40


def test_physical_moment_certification_rejects_incomplete_batches():
    p = SimpleNamespace(spec=SimpleNamespace(coordinate='zzb'), F_pairs=[(0, 0)], pmp_metadata={})
    for tables in ([], [[[1]]], [[[1]], [[1]], [[1]]]):
        sampler = SimpleNamespace(sample_bucket=lambda values, entries: (tables, False))
        channel = SimpleNamespace(ell=0, delta_min=0, prefactored=SimpleNamespace(_sampler=sampler))
        with pytest.raises(ValueError, match='tables for 2 points'):
            C._normalized_moment_physical_channel_detail([1], p, channel,
                block_index=0, n_test_points=2, delta_range=1, weights=[], extra=False)


def test_physical_weight_evaluation_keeps_spin_filter_rules():
    from virasoro_bootstrap.moments import MomentWeight
    p = SimpleNamespace(F_pairs=[], pmp_metadata={})
    fixed = SimpleNamespace(ell=2, delta_min=0, prefactored=SimpleNamespace(shift=0))
    tower = SimpleNamespace(ell=4, twist='.6', delta_min=4, prefactored=SimpleNamespace(shift=4))
    filtered = MomentWeight(MomentWeight.one().terms, spin_filter=0)
    assert C._normalized_moment_physical_row_from_table([1], p, fixed, 3, [[1]], weights=[filtered]) == 0
    assert C._normalized_moment_physical_row_from_table([1], p, tower, 5, [[1]], weights=[filtered]) == 0
    forbidden = MomentWeight(MomentWeight.one().terms, spin_filter=4)
    with pytest.raises(ValueError, match='inside the large-spin tower'):
        C._normalized_moment_physical_row_from_table([1], p, tower, 5, [[1]], weights=[forbidden])


def test_derived_certification_range_preserves_the_charge_label():
    c = Fraction(32)+Fraction(1, 10**70)
    with mp.workdps(15):
        result = C._certification_delta_range(1, SimpleNamespace(c=c, dps=30))
    assert literal_number(result) == (c-1)/4+10
