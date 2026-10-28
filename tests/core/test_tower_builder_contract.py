"""Tower builders preserve requested orders and basis-specific tail powers."""
from types import SimpleNamespace
from fractions import Fraction as F
import mpmath as mp
import pytest
from virasoro_blocks import large_spin as ls


def build(mode, **kwargs):
    opts = dict(mode=mode, h_externals=[mp.mpf('.1')]*4, m_max=0, n_max=0,
                Lmax=2, npoles=1, q_order=4, c=32)
    opts.update(kwargs)
    return ls.large_spin_twist_table(mp.mpf('.5'), **opts)


@pytest.mark.parametrize('mode', ['global', 'virasoro'])
@pytest.mark.parametrize('name', ['q_order', 'npoles', 'm_max', 'n_max'])
@pytest.mark.parametrize('value', [True, F(3, 2), -1, mp.inf])
def test_invalid_tower_controls_fail_before_numerics(monkeypatch, mode, name, value):
    def unexpected(*args, **kwargs):
        pytest.fail('invalid control reached numerical work')
    monkeypatch.setattr(ls, 'large_spin_base', unexpected)
    with pytest.raises(ValueError, match=name):
        build(mode, **{name: value})


@pytest.mark.parametrize('mode', ['global', 'virasoro'])
@pytest.mark.parametrize('basis', ['ordered_B', 'scalar_G'])
def test_tower_tail_power_respects_basis(monkeypatch, mode, basis):
    captured = {}
    def interpolate(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()
    monkeypatch.setattr(ls, '_interpolate_global_table', interpolate)
    build(mode, basis=basis, m_max=1, n_max=3)
    power = captured['entry_power']
    assert power(0, 3) == (0 if basis == 'ordered_B' else 3)
    assert power(1, 3) == (1 if basis == 'ordered_B' else 3)
    assert power(1, 0) == 1


@pytest.mark.parametrize('mode', ['global', 'virasoro'])
def test_ordered_interpolation_preserves_antiholomorphic_factor(mode):
    with mp.workdps(40):
        table = build(mode, basis='ordered_B', m_max=0, n_max=2, pole_selection='nearest')
        direct = table.tower_sampler(3)
        ratio = direct[0][2] / direct[0][0]
        for x in [0, 5, 100]:
            p0 = table.numerators[0][0].evaluate(x)
            p2 = table.numerators[0][2].evaluate(x)
            assert abs(p2 / p0 - ratio) < mp.mpf('1e-32')


@pytest.mark.parametrize('mode,count', [('global', 1), ('virasoro', 2)])
def test_tower_none_keeps_the_complete_order_two_pool(mode, count):
    with mp.workdps(40):
        table = build(mode, q_order=2, npoles=None)
        assert table.prefactor.pole_degree == count
        for spin in table.sample_points:
            expected = table.tower_sampler(spin)[0][0]
            assert abs(table.evaluate(spin)[0][0] / expected - 1) < mp.mpf('1e-33')


@pytest.mark.parametrize('mode', ['global', 'virasoro'])
@pytest.mark.parametrize('zero', [0, '0.0', '0/1', F(0)])
def test_even_scalar_G_tower_refuses_a_spin_zero_floor_before_numerics(monkeypatch, mode, zero):
    """The tower is analytic in L, so its L = 0 member is the doubled 2 B(h, h), not the spin-0 G = B(h, h)."""
    def unexpected(*args, **kwargs):
        pytest.fail('a spin-0 scalar_G tower reached numerical work')
    monkeypatch.setattr(ls, 'large_spin_base', unexpected)
    with pytest.raises(ValueError, match='Lmax = 0 starts the scalar_G tower at spin 0'):
        build(mode, Lmax=zero)


def test_spin_zero_floor_is_allowed_where_no_doubled_member_exists():
    # the ordered_B / parity_projected_B towers carry no spin-0 division
    ls.refuse_spin_zero_scalar_G_tower('ordered_B', 0)
    ls.refuse_spin_zero_scalar_G_tower('parity_projected_B', 0, '-1')
    ls.refuse_spin_zero_scalar_G_tower('parity_projected_B', 0)
    ls.refuse_spin_zero_scalar_G_tower('scalar_G', 2)


@pytest.mark.parametrize('basis', ['scalar_G', 'global_scalar_G'])
def test_there_is_no_odd_scalar_G_tower(basis):
    # the scalar G is z <-> zbar even at every spin: an exchanged parity of -1 is refused, at any floor, pointing to
    # the parity_projected_B tower (the CLI spelling names the Python route, as the CLI needs spinning externals)
    for floor in (0, 3):
        with pytest.raises(ValueError, match="parity_projected_B tower with projection '-'"):
            ls.refuse_spin_zero_scalar_G_tower(basis, floor, '-1')
    with pytest.raises(ValueError, match='--parity-exchanged -1 .*spinning externals only'):
        ls.refuse_spin_zero_scalar_G_tower(basis, 3, '-1', flag='--large-spin-lmin')
    ls.refuse_scalar_G_tower_exchanged_parity(basis, '1')




def test_tower_builder_runs_under_quiet_conditioning():
    # the builder samples near poles on purpose; the refusal helper inserted above it must not take the decorator
    assert getattr(ls.large_spin_twist_table, '__wrapped__', None) is not None
    assert getattr(ls.refuse_spin_zero_scalar_G_tower, '__wrapped__', None) is None
