"""Direct samplers retain exact exchange labels before Kac routing and caching."""
from fractions import Fraction

import mpmath as mp
import pytest

from virasoro_blocks.numeric_input import literal_number
from virasoro_blocks.polyderivs import build_numeric_sampler
from virasoro_blocks.singularity import merged_singular_evaluators
from virasoro_blocks.requests import BlockPoleError
from virasoro_blocks import derivs


MODES = [('scalar','zzb'),('scalar','ws'),('ordered','zzb'),
         ('projected','xt'),('chiral','z')]


@pytest.mark.parametrize('coordinate', ['z', 'y', 'q'])
def test_low_order_singular_charge_taylor_uses_fixed_kac_limits(coordinate):
    from virasoro_blocks.recursion import F_block

    direct = getattr(derivs, f'chiral_{coordinate}_taylor')
    fast = getattr(derivs, f'chiral_{coordinate}_taylor_fast')
    factory = getattr(derivs, f'chiral_{coordinate}_series_evaluator')
    with mp.workdps(30):
        for weight, order in [(Fraction(1, 2), 2), (0, 6)]:
            # Outer Kac poles occur before the first structural recursion
            # singularity at Ising. Exact sigma externals remove them.
            kwargs = dict(c=Fraction(1, 2), q_order=order, z_shift=-Fraction(1, 8))
            external = [Fraction(1, 16)] * 4
            expected = direct(weight, external, 3, **kwargs)
            assert abs(expected[0] - F_block(weight, '.5', external, **kwargs)) < mp.mpf('1e-28')
            evaluate = factory(external, 3, **kwargs)
            for actual in (fast(weight, external, 3, **kwargs), evaluate(weight)):
                assert max(abs(a-b) for a,b in zip(actual, expected)) < mp.mpf('1e-27')
            assert evaluate.many([weight]) is None    # caller uses fixed-h evaluation

            generic = [Fraction(1, 10)] * 4
            for call in (lambda: direct(weight, generic, 3, **kwargs),
                         lambda: fast(weight, generic, 3, **kwargs),
                         lambda: factory(generic, 3, **kwargs)(weight)):
                with pytest.raises(BlockPoleError, match='genuine pole'):
                    call()


@pytest.mark.parametrize('route', ['shared', 'per_entry', 'global'])
def test_interpolation_samplers_receive_exact_origin_plus_node(monkeypatch, route):
    from virasoro_blocks import polyderivs as pd, global_blocks as gb
    from virasoro_blocks.interpolation import DampedRational
    from virasoro_blocks.polynomial import Polynomial

    # A large origin also makes rounded physical nodes coincide. The fit must
    # build its basis in x rather than subtracting the origin after rounding.
    edge = Fraction(10**40 if route == 'per_entry' else 1) + Fraction(1, 10**70)
    seen = []
    def nodes(prefactor, count, **kwargs):
        return [mp.mpf(k) for k in range(count)]
    def sample(points, entries=None):
        seen.extend(points)
        return [[[mp.mpf(1), mp.mpf(1)]] for _ in points], False
    with mp.workdps(30):
        if route == 'global':
            monkeypatch.setattr(gb, 'get_sample_points', nodes)
            def one(point):
                seen.append(point)
                return [[mp.mpf(1)]]
            gb._global_bucket_task((30, DampedRational(1, (), mp.mpf('.5')),
                [], edge, 15, one, 2, [(0, 0)]))
        else:
            monkeypatch.setattr(pd, 'get_sample_points', nodes)
            exact = pd.DeltaDerivTable(exp_base=mp.mpf('.5'), denominator=Polynomial.one(),
                denominator_roots=[], numerators=[[Polynomial([1, 1]), Polynomial.one()]],
                numeric_sampler=sample)
            pd.interpolate_delta_tables({'z': exact}, 0, shift=edge, n_sample_points=2,
                shared_sample_points=route == 'shared',
                kept_entries_by_coord={'z': [(0, 0), (0, 1)]})
    expected = [edge, edge + 1] + ([edge] if route == 'per_entry' else [])
    assert [literal_number(point) for point in seen] == expected


def sampler(mode,coordinate,c):
    return build_numeric_sampler(coordinate,chiral=mode=='chiral',
        spinning=mode in {'ordered','projected'},projected=mode=='projected',
        h_externals=[literal_number('.3')]*4,hbar_externals=[literal_number('.3')]*4,
        ell=0,m_max=1,n_max=0,b=None,c=c,q_order=4,chiral_norm=None,
        parity="+" if mode=="projected" else None)


def read(value):
    # Native Arb/Acb values expose their midpoint through the same conversion
    # used by tables; mpmath fallback grids already have numeric entries.
    from virasoro_blocks.point import _to_mp
    return _to_mp(value)


@pytest.mark.parametrize('label', [0.7,'.7',Fraction(7,10)])
def test_direct_sampler_accepts_exact_literals_consistently(label):
    with mp.workdps(40):
        sample = sampler('scalar','zzb',literal_number('2'))
        actual,_ = sample([label],entries=[(0,0)])
        expected,_ = sample([literal_number('.7')],entries=[(0,0)])
        assert read(actual[0][0][0]) == read(expected[0][0][0])


@pytest.mark.parametrize('entries', [None,[(0,0)]])
def test_distinct_far_from_pole_labels_do_not_share_cached_grids(entries):
    first = Fraction(7,10)
    second = first+Fraction(1,10**70)
    with mp.workdps(40):
        reused = sampler('scalar','zzb','2')
        grids,_ = reused([first,second],entries=entries)
        # Their values can round together; their exact cache identities cannot.
        assert grids[0] is not grids[1]
        again,_ = reused([first,second],entries=entries)
        assert all(a is b for a,b in zip(grids,again))


@pytest.mark.parametrize('entries', [None,[(0,0)]])
def test_generic_c_offsets_from_genuine_poles_match_high_precision(entries):
    def make():
        return build_numeric_sampler('zzb',chiral=False,spinning=False,projected=False,
            h_externals=['.3','.4','.5','.6'],hbar_externals=None,ell=0,
            m_max=1,n_max=0,c='32',q_order=4,chiral_norm=None)
    points = [Fraction(n,10**70) for n in (1,2)]
    with mp.workdps(130):
        grids,_ = make()(points,entries=entries)
        expected = [read(grid[0][0]) for grid in grids]
    with mp.workdps(40):
        grids,_ = make()(points,entries=entries)
        values = [read(grid[0][0]) for grid in grids]
        assert values[0] != values[1]
        for value,target in zip(values,expected):
            assert abs(value-target) < mp.mpf('1e-30')*abs(target)


@pytest.mark.parametrize('mode,coordinate', MODES)
@pytest.mark.parametrize('pole_index', [0,1])
@pytest.mark.parametrize('entries', [None,[(0,0)]])
def test_complex_kac_offsets_survive_sampling_and_integer_spin_split(mode,coordinate,pole_index,entries):
    with mp.workdps(130):
        evaluate = derivs.chiral_z_series_evaluator(['.3']*4,2,c=13+1j,q_order=4)
        poles = [p for p in evaluate.kac_poles() if mp.im(p) != 0]
        pole = poles[pole_index]  # These two poles have the same real part.
        ell = 0 if mode == 'chiral' else 2
        points = [((pole if mode == 'chiral' else 2*pole-ell)
                   + mp.mpf('1e-70')*n) for n in (1,2)]
    def make():
        return build_numeric_sampler(coordinate,chiral=mode=='chiral',
            spinning=mode in {'ordered','projected'},projected=mode=='projected',
            h_externals=['.3']*4,hbar_externals=['.3']*4,ell=ell,
            m_max=1,n_max=0,b=None,c=13+1j,q_order=4,chiral_norm=None,
            parity='+' if mode=='projected' else None)
    with mp.workdps(130):
        grids,_ = make()(points,entries=entries)
        expected = [read(grid[0][0]) for grid in grids]
    with mp.workdps(40):
        reused = make()
        grids,_ = reused(points,entries=entries)
        values = [read(grid[0][0]) for grid in grids]
        assert values[0] != values[1]
        for value,target in zip(values,expected):
            assert abs(value-target) < mp.mpf('1e-30')*abs(target)
        repeated,_ = reused(points,entries=entries)
        assert [read(grid[0][0]) for grid in repeated] == values


@pytest.mark.parametrize('coordinate', ['zzb','yyb','qqb','ws','xt','qqb_ws'])
def test_single_and_batched_complex_derivative_tables_keep_the_same_labels(coordinate):
    single = getattr(derivs, f'G_{coordinate}_table')
    batch = getattr(derivs, f'G_{coordinate}_table_many')
    with mp.workdps(130):
        evaluate = derivs.chiral_z_series_evaluator(['.3']*4,2,c=13+1j,q_order=4)
        pole = next(p for p in evaluate.kac_poles() if mp.im(p))
        points = [2*pole-2+mp.mpf('1e-70')*n for n in (1,2)]
        expected = batch(points,2,['.3']*4,1,0,c=13+1j,q_order=4)
    with mp.workdps(40):
        actual = batch(points,2,['.3']*4,1,0,c=13+1j,q_order=4)
        individual = [single(p,2,['.3']*4,1,0,c=13+1j,q_order=4) for p in points]
        for grids in (actual,individual):
            assert grids[0][0][0] != grids[1][0][0]
            for grid, target in zip(grids,expected):
                for m in range(2):
                    assert abs(grid[m][0]-target[m][0]) < mp.mpf('1e-30')*abs(target[m][0])


@pytest.mark.parametrize('coordinate', ['z','y','q'])
def test_single_chiral_and_ordered_tables_keep_complex_labels(coordinate):
    chiral = getattr(derivs, f'chiral_{coordinate}_taylor')
    ordered = getattr(derivs, f'B_{coordinate*2}b_table')
    with mp.workdps(130):
        evaluate = getattr(derivs, f'chiral_{coordinate}_series_evaluator')(['.3']*4,2,c=13+1j,q_order=4)
        pole = next(p for p in evaluate.kac_poles() if mp.im(p))
        h = pole+mp.mpf('1e-70')
        expected = evaluate(h)
    with mp.workdps(40):
        actual = chiral(h,['.3']*4,2,c=13+1j,q_order=4)
        grid = ordered(h,h,['.3']*4,['.3']*4,1,0,c=13+1j,q_order=4)
        for m in range(2):
            assert abs(actual[m]-expected[m]) < mp.mpf('1e-30')*abs(expected[m])
            target = expected[m]*expected[0]
            assert abs(grid[m][0]-target) < mp.mpf('1e-30')*abs(target)


@pytest.mark.parametrize('weight', [Fraction(0),Fraction(1,4),Fraction(1)])
def test_exact_kac_single_table_reuses_the_shared_laurent_build(monkeypatch,weight):
    from virasoro_blocks import recursion, regularization
    calls = []
    original = recursion._laurent_levels
    monkeypatch.setattr(recursion,'_laurent_levels',lambda *a,**kw: calls.append((mp.mp.dps,kw.get('half'))) or original(*a,**kw))
    recursion._H_LAURENT_PRECOMP_CACHE.clear()
    if recursion._MERGED_CONTEXT_CACHE is not None:
        recursion._MERGED_CONTEXT_CACHE.clear()
    derivs._CHIRAL_EVAL_MEMO.clear()
    with mp.workdps(40),merged_singular_evaluators():
        # A preceding detuned call needs label-resolution digits, which must
        # not become a precision surcharge on every later exact Kac weight.
        regularization._MERGED_LOSS.clear()
        regularization.H_q_series_regularized(
            literal_number('1/4')+literal_number('1e-25'),
            [literal_number('1/16')]*4,c=1,q_order=8,
        )
        calls.clear()
        recursion._H_LAURENT_PRECOMP_CACHE.clear()
        recursion._MERGED_CONTEXT_CACHE.clear()
        evaluate = derivs.chiral_z_series_evaluator(['1/16']*4,2,c=1,q_order=8)
        expected = evaluate(weight)
        actual = derivs.chiral_z_taylor(weight,['1/16']*4,2,c=1,q_order=8)
        for value,target in zip(actual,expected):
            assert abs(value-target) < mp.mpf('1e-30')*max(1,abs(target))
    assert len(calls) == 1, calls
