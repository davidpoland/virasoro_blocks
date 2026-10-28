"""Moment task overrides must reach the shared solve/recertify specification intact."""
import pytest

from virasoro_bootstrap.hpc.tasks import _moment_nu1_over_nu0_spec

PROFILE = dict(q_order=8, npoles=4, spin_list=[0,2], large_spin_double_twists=0,
               large_spin_lmin=4, large_spin_npoles=2)


def build(**overrides):
    params = dict(c='10', delta_ext='.3', large_spin_twists=['1'])
    params.update(overrides)
    return _moment_nu1_over_nu0_spec(params, PROFILE, lam=3, dps=40, num_threads=1)


def test_explicit_tower_spin_threshold_overrides_profile():
    from virasoro_bootstrap.identical_scalar import _resolved_large_spin_tower
    spec = build(large_spin_lmin=2)
    assert spec.large_spin_lmin == 2
    assert _resolved_large_spin_tower(spec, spec.spin_list)[0] == 2
    assert build().large_spin_lmin == 4


def test_default_tower_floor_follows_the_fixed_spins():
    # The requested spin range determines the default, independently of an older profile's floor.
    assert build(spins=[0, 2, 4, 6]).large_spin_lmin == 8
    assert build(spins=[0, 2, 4, 6], large_spin_lmin=6).large_spin_lmin == 6


@pytest.mark.parametrize('name,value', [('q_order',4.5), ('spins',[0,2.5]), ('npoles',True)])
def test_integer_controls_do_not_silently_truncate(name, value):
    with pytest.raises(ValueError):
        build(**{name:value})


def test_integral_string_controls_remain_supported():
    spec = build(q_order='8', npoles='4', spins=['0','2'], large_spin_lmin='8', large_spin_npoles='2')
    assert (spec.q_order,spec.npoles,spec.spin_list,spec.large_spin_lmin,spec.large_spin_npoles) == (8,4,(0,2),8,2)


@pytest.mark.parametrize('value', [0.7, '7/10'])
def test_external_decimal_labels_are_exact(value):
    from fractions import Fraction
    import mpmath as mp
    with mp.workdps(15):
        spec = build(delta_ext=value)
    assert spec.h_ext == Fraction(7,20)


def test_explicit_subprecision_shifts_survive_spec_construction():
    from fractions import Fraction
    import mpmath as mp
    from virasoro_blocks.numeric_input import ExactRational
    offset = ExactRational(1,10**70)
    with mp.workdps(15):
        spec = build(delta_ext=ExactRational(7,10)+offset,
                     delta_gap=ExactRational(1,5)+offset,
                     large_spin_twists=[ExactRational(3,5)+offset],
                     all_spins_twist_gap=True)
    assert spec.h_ext == Fraction(7,20)+offset/2
    assert spec.delta_gap == Fraction(1,5)+offset
    assert spec.large_spin_twists == (Fraction(3,5)+offset,)
    assert spec.delta_min_by_spin[2] == Fraction(11,5)+offset


@pytest.mark.parametrize('field', ['delta_ext','large_spin_twists'])
def test_nonfinite_labels_rejected_before_spec_construction(field):
    values = {'delta_ext':'nan','delta_gap':'inf','large_spin_twists':['nan']}
    with pytest.raises(ValueError):
        build(**{field:values[field]})


@pytest.mark.parametrize('field,value', [('spins','024'),('large_spin_low_twist_step','inf')])
def test_malformed_tower_inputs_are_rejected(field, value):
    with pytest.raises(ValueError):
        build(**{field:value})


@pytest.mark.parametrize('task', ['moment_recertify'])
def test_task_signature_preserves_validated_control_types(task):
    from virasoro_bootstrap.hpc.params import task_signature
    signature = task_signature(task)
    assert signature is not None
    for name in ('lam','dps','q_order','npoles','num_threads','cert_n_test_points',
                 'cert_extra_spins','cert_log_near_points','large_spin_lmin','moment_cert_q_order'):
        assert signature.params[name].kind == 'int', name
    assert signature.params['spins'].kind == 'list[int]'
    assert signature.params['delta_ext'].kind == 'real'
    assert signature.params['physical_moment_blocks'].kind == 'bool'
    assert signature.params['moment_cert_q_order'].required


@pytest.mark.parametrize('field', ['q_order','large_spin_npoles'])
def test_catalogue_presets_reject_fractional_controls(field):
    from virasoro_bootstrap.hpc.tasks import _preset_from_profile
    profile = dict(PROFILE, dps=40)
    profile[field] = 2.5
    with pytest.raises(ValueError, match=field):
        _preset_from_profile(profile)


@pytest.mark.parametrize('value', ['1/3', '.3000000000000000000000000000000000000000000000000000000000000000000001'])
def test_certification_range_preserves_exact_real_inputs(value):
    from fractions import Fraction
    from virasoro_bootstrap.hpc.tasks import _moment_certification_controls
    controls = _moment_certification_controls({'cert_delta_range':value}, {}, q_order=None)
    assert controls['delta_range'] == Fraction(value)


@pytest.mark.parametrize('value', [True, 'nan'])
def test_invalid_certification_ranges_are_rejected(value):
    from virasoro_bootstrap.hpc.tasks import _moment_certification_controls
    with pytest.raises(ValueError):
        _moment_certification_controls({'cert_delta_range':value}, {}, q_order=None)


@pytest.mark.parametrize('source', ['profile','input'])
def test_optional_tower_pole_count_keeps_backend_default(source):
    profile = dict(PROFILE)
    params = dict(c='10',delta_ext='.3',large_spin_twists=['1'])
    if source == 'profile':
        profile.pop('large_spin_npoles')
    else:
        params['large_spin_npoles'] = None
    spec = _moment_nu1_over_nu0_spec(params,profile,lam=3,dps=40,num_threads=1)
    from virasoro_bootstrap.identical_scalar import _resolved_large_spin_npoles
    assert spec.large_spin_npoles is None
    assert _resolved_large_spin_npoles(spec) == spec.npoles
