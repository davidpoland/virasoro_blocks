"""First-class 2d *global* (SL(2)xSL(2)) scalar block basis.

The global block G_{Delta, ell} = [k_h(z) k_hbar(zbar) + (z<->zbar)] / (1 + delta_{ell, 0}),
with k_h(z) = z^h 2F1(h, h; 2h; z) (so G_{Delta, 0} = k_h(z) k_h(zbar); see `_conventions`), has no exact
finite rational representation in Delta: its SL(2) Delta-poles (Delta = ell - p and -ell - p,
p >= 0, minus the removable ones -- for pairwise-equal externals every even p
is removable, so the first poles are at Delta = +-ell - 1; ell = 0 gives
double poles) are infinite in number. So unlike the
Virasoro bases there is no exact `DeltaDerivTable`. Instead we build the
order-reduced `InterpolatedDeltaTable` directly, by optimal-node (Bohr-
Sommerfeld) Lagrange interpolation of the *numeric* block tables in
`virasoro_blocks.derivs`.

The decay base removes the block's exponential dependence on Delta before
interpolation:

    exp_base = 4 * rho(z*) / N ,   rho(z) = (1 - sqrt(1-z)) / (1 + sqrt(1-z)),

the radial coordinate at the crossing-symmetric point z* = 1/2 (so
rho(1/2) = 3 - 2 sqrt(2)); for chiral_norm N = 4 this is (3 - 2 sqrt(2)).
The block decays as exp_base**Delta because Delta = h + hbar and each chiral
half k_h decays as (4 rho / N)**h.

Each (m, n) entry's interpolating polynomial is given the degree
`global_block_asymptotic_power(ell, m, n) + npoles`, the unique degree whose
reduced-rational tail power (deg - npoles) matches the block's true
large-Delta power a = m + n for scalar blocks (which are z <-> zbar even at
every spin). The z <-> zbar-odd combination of a parity-projected B cancels
it to m + n - 1, only at coincident centers, including when the two
external-weight vectors differ. Tying the node count to npoles makes it the single precision knob:
raising it lifts every degree in
lockstep, improving the in-window fit while keeping the correct tail (npoles
6 -> 10 -> 16 gives window error ~1e-9 -> 1e-16 -> 1e-25, tail bounded
throughout). Decoupling the degree from npoles (a larger uniform node count)
would make the tail power deg - npoles > a and the representation diverge past
the node range. The 2F1 itself is evaluated exactly.
``q_order`` therefore does not truncate its evaluation: it fixes the finite
SL(2) pole universe. A chiral pole ``2h = -p`` first occurs at elliptic level
``p + 1``, so a q-order ``Q`` pool retains the genuine left/right poles with
``p + 1 <= Q``. This gives the global and Virasoro paths the same separation:
q-order fixes the candidate universe and ``npoles`` takes a selected prefix.

Output flows through the same `serialize_deriv_tables` ->
`write_json` / `write_mathematica` pipeline as the Virasoro interpolated
tables.
"""

from __future__ import annotations

import bisect
import multiprocessing
from typing import Sequence

import mpmath as mp

from .conditioning import quiet as _quiet_conditioning
from .parallel import usable_workers, validate_worker_count

from .coordinates import COORDINATES

from .derivs import (
    global_G_qqb_table,
    global_G_yyb_table,
    global_G_zzb_table,
    _global_chiral_series_evaluator,
    global_chiral_q_taylor,
    global_chiral_y_taylor,
    global_chiral_z_taylor,
    global_chiral_q_series_evaluator,
    global_chiral_y_series_evaluator,
    global_chiral_z_series_evaluator,
    _assemble_block_grid,
    _fold_parity,
    _resolve_chiral_norm,
    assemble_block_batch,
    fold_parity_marker,
    folded_entry_power,
    folded_source_order,
)
from .interpolation import (
    _validate_npoles,
    DampedRational,
    POLE_SELECTIONS,
    get_sample_points,
    lagrange_polynomial_from_basis,
    precompute_lagrange_basis,
    pole_selection_is_contribution,
    pole_selection_score_entries,
    select_kept_poles,
)
from .parity import parity_reversal_phase
from ._conventions import scalar_G_swap_coefficient
from ._conventions import pool_initializer_kwargs
from .polyderivs import InterpolatedDeltaTable, _default_node_search_dps, _raw_entry_samples
from .polynomial import Polynomial, _arb_to_mpf
from .precision import tolerance, _validation_tolerance
from .numeric_input import literal_components, literal_number, _exchange_weight, _signed_half_integer, _integer_spin


def _global_label_equal(a, b):
    return literal_components(a) == literal_components(b)


def _global_exchange_half(delta, spin):
    delta, spin = literal_number(delta), literal_number(spin)
    if isinstance(delta, complex):
        delta = mp.mpc(delta)
    return _exchange_weight(delta, spin)


def _global_u_center(spec, center, n_terms):
    if center is None:
        return spec.u_symmetric()
    return spec.u_of_z(literal_number(center))

# Numeric 2d global scalar-block builders, keyed by coordinate. Each has the
# signature fn(Delta, ell, h_externals, m_max, n_max, *, chiral_norm, z_shift).
_GLOBAL_BLOCK_BUILDERS = {
    "zzb": global_G_zzb_table,
    "yyb": global_G_yyb_table,
    "qqb": global_G_qqb_table,
}

# Chiral-half Taylor expansions, keyed by the 2d coordinate (zzb/yyb/qqb) and
# by the chiral coordinate (z/y/q). Both name the same chiral block; the 2d
# bases compose two of them, chiral_F emits one.
_CHIRAL_TAYLOR_2D = {
    "zzb": global_chiral_z_taylor,
    "yyb": global_chiral_y_taylor,
    "qqb": global_chiral_q_taylor,
}
# Parity-folded 2d coordinates and the non-folded 2d coordinate each folds
# (`xt` folds the `zzb` block, etc.), derived from the canonical coordinate
# registry. The fold is a Delta-independent linear recombination of the
# non-folded outer-product grid (`derivs._fold_parity`), so it composes with the
# optimal-node Delta interpolation just as it does with the large-spin tower.
_FOLDED_2D_TO_CHIRAL = {
    coord: spec.base_2d
    for coord, spec in COORDINATES.items()
    if spec.kind == "folded_2d"
}
GLOBAL_FOLDED_2D_COORDINATES = frozenset(_FOLDED_2D_TO_CHIRAL)

# Batched arb-native chiral evaluators (the global analogue of the Virasoro
# chiral_*_series_evaluator), keyed by the non-folded 2d coordinate.
_CHIRAL_SERIES_EVALUATOR_2D = {
    "zzb": global_chiral_z_series_evaluator,
    "yyb": global_chiral_y_series_evaluator,
    "qqb": global_chiral_q_series_evaluator,
}

_CHIRAL_TAYLOR_CHIRAL = {
    "z": global_chiral_z_taylor,
    "y": global_chiral_y_taylor,
    "q": global_chiral_q_taylor,
}
GLOBAL_CHIRAL_COORDINATES = frozenset(_CHIRAL_TAYLOR_CHIRAL)

GLOBAL_COORDINATES = frozenset(_GLOBAL_BLOCK_BUILDERS)

# Direct helpers need a stable finite candidate universe even without the CLI.
DEFAULT_GLOBAL_Q_ORDER = 30


def _validate_q_order(value):
    from .requests import _integer_field

    order = _integer_field("q_order", value)
    if order < 1:
        raise ValueError("q_order must be >= 1")
    return order


def _validate_derivative_order(name, value):
    from .requests import _integer_field

    order = _integer_field(name, value)
    if order < 0:
        raise ValueError(f"{name} must be >= 0")
    return order

# Coordinates that global_evaluate_block can recenter: non-folded 2d and
# chiral-only coordinates. Folded/radial coordinates are excluded for the same
# reason as the Virasoro pointwise evaluator.
_GLOBAL_EVALUATE_BLOCK_KINDS = ("non_folded_2d", "chiral")


def _global_half_exp_base_at_center(center, chiral_norm=None):
    """Single-half global decay factor 4 rho(z0) / N at physical center z0."""
    from .numeric_input import parse_number

    source = literal_number(.5 if center is None else center)
    # rho(z) = (1 - sqrt(1-z)) / (1 + sqrt(1-z)); this is the correct global
    # large-delta decay coordinate (not the crossing-odd y coordinate).
    N = _resolve_chiral_norm(parse_number(chiral_norm))
    z0 = mp.convert(source)
    sqrt1mz = mp.sqrt(1-z0)
    rho = (1-sqrt1mz)/(1+sqrt1mz)
    return 4*rho/N


def global_block_exp_base(chiral_norm=None, *, center=None, center_bar=None):
    """Decay base of the 2d global block at the chosen centers.

    With one center (default symmetric z=1/2), this is 4 rho(z0)/N.
    Off diagonal (center_bar != center) uses the geometric mean
    sqrt((4 rho(z0)/N) * (4 rho(zbar0)/N)).
    """
    if center_bar is None:
        center_bar = center
    ez = _global_half_exp_base_at_center(center, chiral_norm)
    ezb = _global_half_exp_base_at_center(center_bar, chiral_norm)
    if ez == ezb:
        return ez
    return mp.sqrt(ez * ezb)


def global_evaluate_block(
    z,
    zbar,
    h,
    hbar,
    h_externals: Sequence,
    *,
    hbar_externals: Sequence | None = None,
    coord: str = "qqb",
    m_max: int = 0,
    n_max: int = 0,
    chiral_norm=None,
    z_shift=None,
    zbar_shift=None,
):
    """Evaluate the ordered 2d global block at an arbitrary (z, zbar).

    Returns the ordered product k_h(z) * k_hbar(zbar), where
    k_h(z) = z^h 2F1(h+h2-h1, h+h3-h4; 2h; z) / N^h, at independent points
    ``z`` and ``zbar``. With ``m_max == n_max == 0`` the return value is a
    scalar block value. Otherwise this returns the Taylor grid in the selected
    chiral coordinate ``u`` (z/y/q):
    ``grid[m][n] = d_u^m d_ubar^n B / m! n!`` around ``(u(z), u(zbar))``.

    ``h_externals`` controls the holomorphic half and ``hbar_externals`` the
    antiholomorphic half (defaults to ``h_externals``).
    """
    m_max = _validate_derivative_order("m_max", m_max)
    n_max = _validate_derivative_order("n_max", n_max)
    spec = COORDINATES.get(coord)
    if spec is None or spec.kind not in _GLOBAL_EVALUATE_BLOCK_KINDS:
        raise ValueError(
            f"global_evaluate_block does not support coord={coord!r}; choose a "
            f"non-folded 2d coordinate (zzb/yyb/qqb) or a chiral one (z/y/q)"
        )

    if len(h_externals) != 4:
        raise ValueError("h_externals must have length 4")
    if hbar_externals is None:
        hbar_externals = h_externals
    if len(hbar_externals) != 4:
        raise ValueError("hbar_externals must have length 4")

    z_of_u = spec.z_of_u
    u_of_z = spec.u_of_z

    def _half(point, exch_h, ext, n_terms, shift):
        if n_terms == 1:
            # No derivatives: evaluate at the supplied physical point. The
            # coordinate Taylor backend needlessly reconstructs z from q/y,
            # which can lose endpoint offsets through conversion or truncation.
            from .derivs import _global_chiral_block

            return [_global_chiral_block(
                exch_h, point, ext, chiral_norm=chiral_norm, z_shift=shift,
            )]
        center = u_of_z(point)
        evaluate = _global_chiral_series_evaluator(
            ext,
            n_terms,
            z_of_u=z_of_u,
            u_symmetric=center,
            chiral_norm=chiral_norm,
            z_shift=shift,
        )
        return evaluate(exch_h)

    diagonal = (
        _global_label_equal(zbar, z)
        and _global_label_equal(hbar, h)
        and all(_global_label_equal(a, b) for a, b in zip(hbar_externals, h_externals))
        and _global_label_equal(zbar_shift, z_shift)
    )
    # on the diagonal one half serves both factors, so it must reach n_max too
    c_h = _half(z, h, h_externals, (max(m_max, n_max) if diagonal else m_max) + 1, z_shift)
    if diagonal:
        c_hbar = c_h
    else:
        c_hbar = _half(zbar, hbar, hbar_externals, n_max + 1, zbar_shift)

    grid = _assemble_block_grid(c_h, c_hbar, m_max, n_max, sigma=0)
    if m_max == 0 and n_max == 0:
        return grid[0][0]
    return grid


def _half_integer_difference(value):
    """Return exact 2*value when it is a real half-integer, else None."""
    from .numeric_input import literal_components

    real, imag = literal_components(value)
    if imag != 0:
        return None
    twice = 2 * real
    return int(twice) if twice.denominator == 1 else None


def _chiral_pole_is_removable(p: int, delta12, delta34) -> bool:
    """Whether the chiral pole at 2h = -p has a VANISHING residue (removable).

    k_h(z) = z^h 2F1(h + delta12, h + delta34; 2h; z) has Delta-pole candidates
    from (2h)_k at 2h = -p (p = 0, 1, 2, ...), but the residue vanishes when the
    numerator (h + delta12)_k (h + delta34)_k kills every term of the residue
    sum (k >= p + 1). At the pole h = -p/2 that happens iff a = delta12 - p/2 or
    b = delta34 - p/2 is a NON-POSITIVE INTEGER in [-p, 0]. This is an exact
    rule on the external differences (no residue threshold): for identical /
    pairwise-equal externals (delta = 0) it removes even p (leaving the odd-p
    poles); for generic externals nothing is removable (every p is a pole); for
    special half-integer delta it can remove more. Verified against a numeric
    residue/order detector across all four regimes.
    """
    for delta in (delta12, delta34):
        twice = _half_integer_difference(delta)
        if twice is not None and (twice - p) % 2 == 0 and -p <= twice <= p:
            return True
    return False


def _nearest_pole_ladder(offset, spacing, count, delta12, delta34, shift):
    """Nearest poles of offset-spacing*p, without scanning from p=0.

    Each parity of p is an arithmetic progression. An external difference
    d with integral 2*d removes that parity starting at p=abs(2*d); two
    external differences can therefore leave a finite or empty pole set.
    """
    limits = [None, None]
    for delta in (delta12, delta34):
        twice = _half_integer_difference(delta)
        if twice is not None:
            parity = twice % 2
            cutoff = abs(twice)
            limits[parity] = cutoff if limits[parity] is None else min(limits[parity], cutoff)
    poles = []
    for parity, cutoff in enumerate(limits):
        # p=parity+2*k, with k>=0 and (when bounded) p<cutoff.
        last = None if cutoff is None else (cutoff - 1 - parity) // 2
        if last is not None and last < 0:
            continue
        target = (offset - shift - spacing * parity) / (2 * spacing)
        left = int(mp.floor(target))
        if last is not None:
            left = min(left, last)
        left = max(-1, left)
        right = left + 1
        for _ in range(count):
            lower = offset - spacing * (parity + 2 * left) if left >= 0 else None
            upper = offset - spacing * (parity + 2 * right) if last is None or right <= last else None
            if lower is None and upper is None:
                break
            if lower is not None and (upper is None or abs(lower - shift) <= abs(upper - shift)):
                poles.append(lower)
                left -= 1
            else:
                poles.append(upper)
                right += 1
    return poles


def _genuine_2d_poles(
    ell, count, *, dh12, dh34, dhb12, dhb34, shift,
) -> list:
    """Genuine 2d-block Delta-poles nearest `shift`, with multiplicity.

    The block is k_h(z) k_hbar(zbar) (+ a parity partner with the same poles),
    so candidates sit at Delta = -ell - p (the h half, 2h = -p, removable per
    the holomorphic differences dh12 = h2-h1, dh34 = h3-h4) and Delta = ell - p
    (the hbar half, 2 hbar = -p, removable per the antiholomorphic differences
    dhb12 = hbar2-hbar1, dhb34 = hbar3-hbar4). For scalar / pairwise-equal
    externals the two branches share the same rule. Removability is the exact
    integer check `_chiral_pole_is_removable` -- no residue tolerance.
    """
    count = _validate_npoles(count)
    shift = mp.mpf(mp.convert(shift))
    ell = mp.mpf(mp.convert(ell))
    if not mp.isfinite(shift) or not mp.isfinite(ell):
        raise ValueError("shift and ell must be finite")
    poles = _nearest_pole_ladder(ell, 1, count, dhb12, dhb34, shift)
    poles += _nearest_pole_ladder(-ell, 1, count, dh12, dh34, shift)
    poles.sort(key=lambda q: (abs(q - shift), -q))
    return poles[:count]


def _genuine_chiral_poles(count, *, delta12, delta34, shift=0) -> list:
    """Genuine chiral-block h-poles nearest `shift` (default 0), with
    multiplicity. k_h has candidates at h = -p/2 (2h = -p); removable per the
    external differences via `_chiral_pole_is_removable`."""
    count = _validate_npoles(count)
    shift = mp.mpf(mp.convert(shift))
    if not mp.isfinite(shift):
        raise ValueError("shift must be finite")
    poles = _nearest_pole_ladder(mp.mpf(0), mp.mpf(1)/2, count, delta12, delta34, shift)
    poles.sort(key=lambda q: (abs(q - shift), -q))
    return poles[:count]


def _global_q_order_pole_pool(
    ell, q_order, *, dh12, dh34, dhb12, dhb34,
) -> list:
    """Genuine 2d SL(2) poles first visible through elliptic q-order ``Q``.

    In a chiral factor, the candidate ``2h=-p`` first enters at q-grade
    ``p+1`` through the ``(2h)_{p+1}`` denominator. A 2d q-order contributes
    the union of the two chiral ladders for ``0 <= p < Q``. Coincident
    left/right locations stay duplicated because they are genuine higher-order
    poles of the full block.
    """
    q_order = _validate_q_order(q_order)
    poles = []
    for p in range(q_order):
        if not _chiral_pole_is_removable(p, dhb12, dhb34):
            poles.append(mp.mpf(ell - p))
        if not _chiral_pole_is_removable(p, dh12, dh34):
            poles.append(mp.mpf(-ell - p))
    return poles


def _global_chiral_q_order_pole_pool(q_order, *, delta12, delta34) -> list:
    """Genuine chiral SL(2) poles first visible through q-order ``Q``."""
    q_order = _validate_q_order(q_order)
    return [
        -mp.mpf(p) / 2 for p in range(q_order)
        if not _chiral_pole_is_removable(p, delta12, delta34)
    ]


def _externals_diffs(h_externals):
    """The differences ``h2 - h1`` and ``h3 - h4`` that decide which chiral poles are removable.

    Exact labels decide exactly. A numerical (mpmath) input is a value at its
    precision: a difference involving one that is a half-integer within
    roundoff at the working precision is that half-integer (mpf('0.6') -
    mpf('0.1') is 1/2), as the removable-pole rule needs.
    """
    from fractions import Fraction
    from .numeric_input import ExactRational, literal_components, parse_number
    from .precision import tolerance

    h_externals = list(h_externals)
    parts = [literal_components(value) for value in h_externals]
    numeric = [not isinstance(parse_number(value), (int, Fraction)) for value in h_externals]

    def difference(left, right):
        real, imag = parts[left][0] - parts[right][0], parts[left][1] - parts[right][1]
        if numeric[left] or numeric[right]:
            twice = round(2 * real)
            scale = max(1, *(abs(component) for component in (*parts[left], *parts[right])))
            if abs(mp.mpc(mp.convert(2 * real - twice), mp.convert(2 * imag))) <= tolerance(guard=5) * scale:
                return ExactRational(twice, 2)
        # Integer eligibility depends on the exact real difference. When the
        # imaginary difference is nonzero no chiral pole can be removed, so
        # its numerical conversion need only retain that nonzero component.
        return real if imag == 0 else mp.mpc(mp.convert(real), mp.convert(imag))

    return difference(1, 0), difference(2, 3)


def global_sl2_delta_poles(ell: int, npoles: int, *, h_externals=None, shift=None) -> list:
    """The `npoles` genuine 2d scalar-block Delta-poles nearest the unitarity
    region. `h_externals` fixes the external differences the genuine set depends
    on; default (None) assumes identical / pairwise-equal externals (the odd-p
    poles). For scalar externals both chiral halves share the same differences.
    """
    npoles = _validate_npoles(npoles)
    if shift is None:
        shift = abs(mp.mpf(ell))    # the unitarity bound |ell|, half-integer for fermionic exchange
    if h_externals is None:
        d12 = d34 = mp.mpf(0)
    else:
        d12, d34 = _externals_diffs(h_externals)
    return _genuine_2d_poles(
        ell, npoles, dh12=d12, dh34=d34, dhb12=d12, dhb34=d34, shift=shift,
    )


def global_block_asymptotic_power(ell: int, m: int, n: int) -> int:
    """Leading large-Delta power a of the 2d scalar block:
    block[m][n] ~ base**Delta * Delta**a.

    Each z-derivative of k_h(z) ~ base**h brings a factor h ~ Delta/2, so a
    chiral product k_h^(m) k_hbar^(n) grows as Delta**(m+n). The scalar block
    is the symmetric combination at every spin ``ell``, so the leading term
    never cancels. (The ordered and parity-projected B bases compute their own
    per-entry power; the antisymmetric '-' projection drops it to m+n-1. This
    helper is the scalar-G case.)
    """
    return m + n


def _pole_location_multiplicities(pool):
    """Return each pole's multiplicity, grouping full complex locations.

    Locations within tolerance share a multiplicity. Repeated poles require
    principal-part scoring; conjugate simple poles remain separate even when
    they have the same real part.
    """
    tol = tolerance(guard=5)
    tol2 = tol * tol
    # representatives [re, im, count] kept sorted by real part, so a location is looked up by bisection over the
    # real parts within tol and a squared-distance test (no quadratic scan, no complex abs)
    reps: list = []
    rep_re: list = []

    def find(re, im):
        for i in range(bisect.bisect_left(rep_re, re - tol), len(reps)):
            r_re, r_im, _count = reps[i]
            if r_re > re + tol:
                break
            if (r_re - re) ** 2 + (r_im - im) ** 2 <= tol2:
                return i
        return None

    for p in pool:
        z = mp.mpc(p)
        i = find(z.real, z.imag)
        if i is None:
            i = bisect.bisect_right(rep_re, z.real)
            rep_re.insert(i, z.real)
            reps.insert(i, [z.real, z.imag, 1])
        else:
            reps[i][2] += 1

    def order_of(p):
        z = mp.mpc(p)
        i = find(z.real, z.imag)
        return 1 if i is None else reps[i][2]

    return order_of, max((count for _, _, count in reps), default=1)


def _make_residue_scorer(sampler, base, m_max, n_max, score_entries="kept"):
    """Return a callable `pool -> scores` giving the residue magnitude |R| of
    each genuine pole for the residue / contribution / residue_all selectors.
    The multiplicity is read off `pool` (the analytic genuine multiset), so the
    score is a high-precision limit, never a thresholded keep/drop decision.
    `sampler(value)` returns the numeric block table; the pole variable is Delta
    for the 2d bases and h for chiral_F.

    ``score_entries`` selects which (m, n) entries the residue magnitude maxes
    over: ``"kept"`` (every entry, used by the residue_all/contribution_all
    variants) or
    ``"value"`` (only the (0, 0) undifferentiated block entry, for ``residue``
    and ``contribution``).
    """
    if score_entries == "value":
        from .polyderivs import _value_score_entry_from_sampler

        value_entry = _value_score_entry_from_sampler(sampler, n_max=n_max)
        mn_pairs = [value_entry if value_entry is not None else (0, 0)]
    elif score_entries == "kept":
        mn_pairs = [(m, n) for m in range(m_max + 1) for n in range(n_max + 1)]
    else:
        raise ValueError(
            f"score_entries must be 'kept' or 'value'; got {score_entries!r}"
        )

    sample_bucket = getattr(sampler, "sample_bucket", None)

    @_quiet_conditioning
    def score(pool):
        order_of, _ = _pole_location_multiplicities(pool)
        eps = mp.mpf(10) ** (-(mp.mp.dps // 2))
        out = []
        for p in pool:
            order = order_of(p)
            # Only `mn_pairs` enter the magnitude, so build just those entries
            # (the value selectors need only (0, 0)) instead of the full grid.
            result = (
                sample_bucket([p + eps], mn_pairs)
                if sample_bucket is not None else None
            )
            if result is None:
                tbl = sampler(p + eps)
                is_arb = False
            else:
                grids, is_arb = result
                tbl = grids[0]
            # The score needs an mpmath magnitude; the arb grids carry real
            # values (the probe p + eps is real), so take the arb midpoint.
            if is_arb:
                mag = max(abs(_arb_to_mpf(tbl[m][n])) for (m, n) in mn_pairs)
            else:
                mag = max(abs(tbl[m][n]) for (m, n) in mn_pairs)
            out.append(mag * eps ** order / abs(base ** (p + eps)))
        return out

    return score


class _GlobalBlockSampler2d:
    """Picklable numeric 2d global-block sampler at one Delta.

    Holds plain data (a coordinate string, exact input labels, a real/complex sigma)
    so it ships cleanly to multiprocessing workers; the lazily-built arb
    chiral evaluators are dropped in ``__getstate__`` and rebuilt per worker.

    ``sample_bucket(values, entries)`` returns ``(tables, is_arb)`` -- the
    batched, entry-restricted block grids plus whether they are native flint
    ``arb`` (the interpolation's arb fast path) or mpmath (real-c global blocks
    are real, so the arb path is the norm; the mpmath fallback covers complex
    externals / sigma)."""

    def __init__(self, coordinate, ell, h_ext, hbar_ext, sigma,
                 chiral_norm, z_shift, zbar_shift, m_max, n_max,
                 *, folded=False, fold_parity=0, center=None, center_bar=None,
                 basis="scalar_G", parity=None):
        self.coordinate = coordinate
        self.ell = _signed_half_integer(ell)
        self.basis = basis
        self.parity = parity
        self.h_ext = [literal_number(value) for value in h_ext]
        self.hbar_ext = [literal_number(value) for value in hbar_ext]
        self.sigma = sigma
        self.chiral_norm = chiral_norm
        self.z_shift = z_shift
        self.zbar_shift = zbar_shift
        self.m_max = m_max
        self.n_max = n_max
        # Parity fold: `m_max`/`n_max` are the folded (w, s) output dimensions.
        # The non-folded outer-product grid is built in the underlying chiral
        # coordinate to the expanded order `m_max + 2 n_max + odd` in each index
        # and stripped to the folded shape via `_fold_parity`.
        self.folded = folded
        self.fold_parity = fold_parity
        self.center = center
        self.center_bar = center_bar
        spec = None
        if folded:
            if center is not None or center_bar is not None:
                raise ValueError(
                    "off-diagonal centers are supported only on non-folded "
                    "coordinates (zzb/yyb/qqb)"
                )
            self.chiral_coordinate = _FOLDED_2D_TO_CHIRAL[coordinate]
            spec = COORDINATES[self.chiral_coordinate]
            self._src_m = folded_source_order(m_max, n_max, fold_parity)
            self._src_n = self._src_m
            self._u_center = spec.u_symmetric()
            self._u_center_bar = self._u_center
            self._offdiag = False
        else:
            self.chiral_coordinate = coordinate
            self._src_m = m_max
            self._src_n = n_max
            spec = COORDINATES[self.chiral_coordinate]
            self._u_center = (
                _global_u_center(spec, center, max(self._src_m, self._src_n)+1)
            )
            if center_bar is None:
                self._u_center_bar = self._u_center
            else:
                self._u_center_bar = _global_u_center(spec, center_bar, max(self._src_m, self._src_n)+1)
            self._offdiag = (self._u_center_bar != self._u_center)
        # A single chiral evaluator serves both halves when they coincide
        # (identical externals + equal shifts, e.g. the scalar block).
        self._same_half = (
            all(_global_label_equal(a, b) for a, b in zip(self.h_ext, self.hbar_ext))
            and _global_label_equal(self.z_shift, self.zbar_shift)
        )
        self._ev_h = None
        self._ev_hbar = None
        self._ev_h_n = None
        self._ev_hbar_n = None
        self._ev_nterms = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_ev_h"] = None
        state["_ev_hbar"] = None
        state["_ev_h_n"] = None
        state["_ev_hbar_n"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._ev_h = None
        self._ev_hbar = None
        self._ev_h_n = None
        self._ev_hbar_n = None
        self._ev_nterms = 0

    def _centered_evaluator(self, h_ext, n_terms, *, z_shift, u_center):
        spec = COORDINATES[self.chiral_coordinate]
        return _global_chiral_series_evaluator(
            h_ext,
            n_terms,
            z_of_u=spec.z_of_u,
            u_symmetric=u_center,
            chiral_norm=self.chiral_norm,
            z_shift=z_shift,
        )

    def _evaluators(self, n_terms):
        """Cache the chiral evaluators at the current working precision."""
        if getattr(self, "_cache_precision", None) != mp.mp.prec:
            spec = COORDINATES[self.chiral_coordinate]
            self._u_center = (
                _global_u_center(spec, self.center, max(self._src_m, self._src_n)+1)
            )
            self._u_center_bar = (
                self._u_center if self.center_bar is None else _global_u_center(spec, self.center_bar, max(self._src_m, self._src_n)+1)
            )
            self._offdiag = self._u_center_bar != self._u_center
            self._ev_h = self._ev_hbar = self._ev_h_n = self._ev_hbar_n = None
            self._ev_nterms = 0
            self._cache_precision = mp.mp.prec
        if self._ev_h is None or self._ev_nterms < n_terms:
            self._ev_h = self._centered_evaluator(
                self.h_ext,
                n_terms,
                z_shift=self.z_shift,
                u_center=self._u_center,
            )
            self._ev_hbar = self._ev_h if self._same_half else self._centered_evaluator(
                self.hbar_ext,
                n_terms,
                z_shift=self.zbar_shift,
                u_center=self._u_center,
            )
            if self._offdiag:
                self._ev_h_n = self._centered_evaluator(
                    self.h_ext,
                    n_terms,
                    z_shift=self.z_shift,
                    u_center=self._u_center_bar,
                )
                self._ev_hbar_n = self._ev_h_n if self._same_half else self._centered_evaluator(
                    self.hbar_ext,
                    n_terms,
                    z_shift=self.zbar_shift,
                    u_center=self._u_center_bar,
                )
            else:
                self._ev_h_n = self._ev_h
                self._ev_hbar_n = self._ev_hbar
            self._ev_nterms = n_terms
        return self._ev_h, self._ev_hbar, self._ev_h_n, self._ev_hbar_n

    def _halves(self, delta):
        n_top = max(self._src_m, self._src_n) + 1
        ell = self.ell
        ev_h, ev_hbar, ev_h_n, ev_hbar_n = self._evaluators(n_top)
        c_h_m = ev_h(_global_exchange_half(delta, ell))
        c_hbar_n = ev_hbar_n(_global_exchange_half(delta, -ell))
        if self.sigma == 0 or not self._offdiag:
            return c_h_m, c_hbar_n, None, None
        c_hbar_m = (
            c_h_m if (self._same_half and ell == 0)
            else ev_hbar(_global_exchange_half(delta, -ell))
        )
        c_h_n = (
            c_hbar_n if (self._same_half and ell == 0)
            else ev_h_n(_global_exchange_half(delta, ell))
        )
        return c_h_m, c_hbar_n, c_hbar_m, c_h_n

    def _assemble(self, c_h, c_hb, c_swap_m=None, c_swap_n=None, entries=None):
        """Final (folded if needed) mpmath block grid (the fallback path). The
        non-folded path honors `entries`; the folded path builds the full grid
        and the interpolation reads only the entries it needs."""
        if not self.folded:
            return _assemble_block_grid(
                c_h, c_hb, self._src_m, self._src_n, sigma=self.sigma,
                c_swap_m=c_swap_m, c_swap_n=c_swap_n,
                entries=entries,
            )
        if c_swap_m is not None or c_swap_n is not None:
            raise ValueError("folded global assembly does not support off-diagonal swap vectors")
        full = _assemble_block_grid(
            c_h, c_hb, self._src_m, self._src_n, sigma=self.sigma
        )
        return _fold_parity(full, self.fold_parity, self.m_max, self.n_max)

    def _analytic_residue_scores(self, roots, delta_eval, entries, base):
        """Closed-form principal-part contribution scores for the 2d global
        block, or ``None`` to defer to the contour scorer.

        The global chiral half has simple h-poles at ``h = -p/2`` with closed-form
        residue / regular series (``evaluate.pole_data``); the two halves coincide
        into a genuine double pole at ``ell = 0`` (and at ``Delta`` where both
        ``-ell-p`` and ``ell-p`` are non-removable). Reuses the shared
        principal-part cores: diagonal, off-diagonal (four chiral-factor streams),
        and folded (xt/ws, real parity phase only). ``sigma = self.sigma`` (0 for
        ordered_B drops the swap term). Named ``_analytic_residue_scores`` to
        match the hook :func:`_interpolate_global_table` already consults."""
        from virasoro_blocks.polyderivs import (
            _analytic_chiral_factor, _analytic_pp_scores,
            _analytic_pp_scores_axis, _analytic_pp_scores_folded,
        )
        ell = self.ell
        sigma = self.sigma
        sigma_arg = None if sigma == 0 else sigma

        if self.folded:
            # Folded blocks need a definite z<->zbar parity, i.e. a real (+/-1)
            # parity phase; defer a complex phase to the contour (it cannot be
            # folded at all -- the table builder guards the same way).
            if isinstance(sigma, mp.mpc) and sigma.imag != 0:
                return None
            n_terms = max(self._src_m, self._src_n) + 1
            ev_h, ev_hbar, _evh_n, _evhb_n = self._evaluators(n_terms)
            fac_h = _analytic_chiral_factor(ev_h, n_terms, deriv_factor=2)
            fac_hbar = (
                fac_h if ev_hbar is ev_h
                else _analytic_chiral_factor(ev_hbar, n_terms, deriv_factor=2)
            )
            if fac_h is None or fac_hbar is None:
                return None
            return _analytic_pp_scores_folded(
                roots, delta_eval, entries, base,
                factor_m_plus=lambda loc: fac_h((loc + ell) / 2),
                factor_m_minus=lambda loc: fac_hbar((loc - ell) / 2),
                factor_n_plus=lambda loc: fac_h((loc + ell) / 2),
                factor_n_minus=lambda loc: fac_hbar((loc - ell) / 2),
                sigma=sigma_arg, fold_marker=self.fold_parity,
                m_max=self.m_max, n_max=self.n_max,
            )

        n_terms = max(self._src_m, self._src_n) + 1
        ev_h, ev_hbar, ev_h_n, ev_hbar_n = self._evaluators(n_terms)
        fac_h = _analytic_chiral_factor(ev_h, n_terms, deriv_factor=2)
        fac_hbar = (
            fac_h if ev_hbar is ev_h
            else _analytic_chiral_factor(ev_hbar, n_terms, deriv_factor=2)
        )
        if fac_h is None or fac_hbar is None:
            return None

        if self._offdiag:
            fac_h_n = (
                fac_h if ev_h_n is ev_h
                else _analytic_chiral_factor(ev_h_n, n_terms, deriv_factor=2)
            )
            fac_hbar_n = (
                fac_hbar if ev_hbar_n is ev_hbar
                else _analytic_chiral_factor(ev_hbar_n, n_terms, deriv_factor=2)
            )
            if fac_h_n is None or fac_hbar_n is None:
                return None
            return _analytic_pp_scores_axis(
                roots, delta_eval, entries, base,
                factor_m_plus=lambda loc: fac_h((loc + ell) / 2),
                factor_m_minus=lambda loc: fac_hbar((loc - ell) / 2),
                factor_n_plus=lambda loc: fac_h_n((loc + ell) / 2),
                factor_n_minus=lambda loc: fac_hbar_n((loc - ell) / 2),
                sigma=sigma_arg,
            )

        return _analytic_pp_scores(
            roots, delta_eval, entries, base,
            factor_h=lambda loc: fac_h((loc + ell) / 2),
            factor_hbar=lambda loc: fac_hbar((loc - ell) / 2),
            sigma=sigma_arg,
        )

    def _analytic_leading_residue_scores(self, roots, entries, base):
        """Closed-form leading-Laurent-coefficient residue scores for the
        2d global block, or ``None`` to defer to the eps-probe scorer.

        Uses the same chiral residue/regular series data as
        :meth:`_analytic_residue_scores`, but returns the leading Laurent
        coefficient magnitude per pole (|g_-1| for simple poles, |g_-2| for
        genuine double poles) rather than the full principal part at an
        evaluation point.
        """
        from virasoro_blocks.polyderivs import (
            _analytic_chiral_factor,
            _analytic_residue_leading_scores,
            _analytic_residue_leading_scores_axis,
            _analytic_residue_leading_scores_folded,
        )
        ell = self.ell
        sigma = self.sigma
        sigma_arg = None if sigma == 0 else sigma

        if self.folded:
            if isinstance(sigma, mp.mpc) and sigma.imag != 0:
                return None
            n_terms = max(self._src_m, self._src_n) + 1
            ev_h, ev_hbar, _evh_n, _evhb_n = self._evaluators(n_terms)
            fac_h = _analytic_chiral_factor(ev_h, n_terms, deriv_factor=2)
            fac_hbar = (
                fac_h if ev_hbar is ev_h
                else _analytic_chiral_factor(ev_hbar, n_terms, deriv_factor=2)
            )
            if fac_h is None or fac_hbar is None:
                return None
            return _analytic_residue_leading_scores_folded(
                roots, entries, base,
                factor_m_plus=lambda loc: fac_h((loc + ell) / 2),
                factor_m_minus=lambda loc: fac_hbar((loc - ell) / 2),
                factor_n_plus=lambda loc: fac_h((loc + ell) / 2),
                factor_n_minus=lambda loc: fac_hbar((loc - ell) / 2),
                sigma=sigma_arg, fold_marker=self.fold_parity,
                m_max=self.m_max, n_max=self.n_max,
            )

        n_terms = max(self._src_m, self._src_n) + 1
        ev_h, ev_hbar, ev_h_n, ev_hbar_n = self._evaluators(n_terms)
        fac_h = _analytic_chiral_factor(ev_h, n_terms, deriv_factor=2)
        fac_hbar = (
            fac_h if ev_hbar is ev_h
            else _analytic_chiral_factor(ev_hbar, n_terms, deriv_factor=2)
        )
        if fac_h is None or fac_hbar is None:
            return None

        if self._offdiag:
            fac_h_n = (
                fac_h if ev_h_n is ev_h
                else _analytic_chiral_factor(ev_h_n, n_terms, deriv_factor=2)
            )
            fac_hbar_n = (
                fac_hbar if ev_hbar_n is ev_hbar
                else _analytic_chiral_factor(ev_hbar_n, n_terms, deriv_factor=2)
            )
            if fac_h_n is None or fac_hbar_n is None:
                return None
            return _analytic_residue_leading_scores_axis(
                roots, entries, base,
                factor_m_plus=lambda loc: fac_h((loc + ell) / 2),
                factor_m_minus=lambda loc: fac_hbar((loc - ell) / 2),
                factor_n_plus=lambda loc: fac_h_n((loc + ell) / 2),
                factor_n_minus=lambda loc: fac_hbar_n((loc - ell) / 2),
                sigma=sigma_arg,
            )

        return _analytic_residue_leading_scores(
            roots, entries, base,
            factor_h=lambda loc: fac_h((loc + ell) / 2),
            factor_hbar=lambda loc: fac_hbar((loc - ell) / 2),
            sigma=sigma_arg,
        )

    def __call__(self, delta):
        return self._assemble(*self._halves(delta))

    def _sample_arb(self, values, entries):
        """Batched arb-native block grids, or ``None`` to fall back to mpmath
        (complex externals / sigma, or complex sample points).

        The arb batch path assumes real sample points (it builds ``arb`` from
        each value). Complex points -- needed by the contour/DFT contribution
        scorer -- decline here and route to the per-value ``_assemble(_halves(d))``
        path, which evaluates the chiral halves in complex arithmetic."""
        if any(isinstance(v, (mp.mpc, complex)) and v.imag != 0 for v in values):
            return None
        values = [literal_number(v.real if isinstance(v, (mp.mpc, complex)) else v)
                  for v in values]
        ell = self.ell
        ev_h, ev_hbar, ev_h_n, ev_hbar_n = self._evaluators(max(self._src_m, self._src_n) + 1)
        c_h_all = ev_h.many([_global_exchange_half(v, ell) for v in values])
        if c_h_all is None:
            return None
        c_hbar_all = ev_hbar_n.many([_global_exchange_half(v, -ell) for v in values])
        if c_hbar_all is None:
            return None
        c_swap_m_all = c_swap_n_all = None
        if self.sigma != 0 and self._offdiag:
            c_swap_m_all = (
                c_h_all if (self._same_half and ell == 0)
                else ev_hbar.many([_global_exchange_half(v, -ell) for v in values])
            )
            c_swap_n_all = (
                c_hbar_all if (self._same_half and ell == 0)
                else ev_h_n.many([_global_exchange_half(v, ell) for v in values])
            )
            if c_swap_m_all is None or c_swap_n_all is None:
                return None
        return assemble_block_batch(
            c_h_all, c_hbar_all, self.m_max, self.n_max, sigma=self.sigma,
            fold_parity=(self.fold_parity if self.folded else None),
            entries=entries,
            c_swap_m_all=c_swap_m_all,
            c_swap_n_all=c_swap_n_all,
        )

    def sample_bucket(self, values, entries):
        """``(tables, is_arb)`` block grids at `values`, restricted to a node-
        count bucket's `entries`. Tries the batched arb-native path; falls back
        to per-value mpmath assembly (returning ``is_arb=False``)."""
        values = list(values)
        entries = tuple(entries) if entries is not None else None
        arb = self._sample_arb(values, entries)
        if arb is not None:
            return arb
        tables = [self._assemble(*self._halves(d), entries=entries) for d in values]
        return tables, False


class _GlobalBlockSamplerChiral:
    """Picklable numeric chiral-block sampler at one h (n_max = 0).

    ``center`` (a z-value, default ``None`` = the symmetric point) recenters the
    single global chiral expansion at ``z = center``; a chiral half has no zbar,
    so there is no off-diagonal notion.
    """

    def __init__(self, coordinate, h_ext, chiral_norm, z_shift, m_max, *, center=None):
        self.coordinate = coordinate
        self.h_ext = [literal_number(value) for value in h_ext]
        self.chiral_norm = chiral_norm
        self.z_shift = z_shift
        self.m_max = m_max
        self.center = center
        spec = COORDINATES[coordinate]
        self._u_center = _global_u_center(spec, center, self.m_max+1)
        self._evaluate = None  # lazily built centered evaluator (off-center only)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_evaluate"] = None  # closure: rebuilt per worker
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._evaluate = None

    def __call__(self, h):
        if getattr(self, "_cache_precision", None) != mp.mp.prec:
            spec = COORDINATES[self.coordinate]
            self._u_center = (
                _global_u_center(spec, self.center, self.m_max+1)
            )
            self._evaluate = None
            self._cache_precision = mp.mp.prec
        if self.center is None:
            # Symmetric: the per-h Taylor builder (bit-identical to production).
            chiral = _CHIRAL_TAYLOR_CHIRAL[self.coordinate]
            c = chiral(h, self.h_ext, self.m_max + 1, chiral_norm=self.chiral_norm,
                       z_shift=self.z_shift)
        else:
            if self._evaluate is None:
                spec = COORDINATES[self.coordinate]
                self._evaluate = _global_chiral_series_evaluator(
                    self.h_ext, self.m_max + 1,
                    z_of_u=spec.z_of_u, u_symmetric=self._u_center,
                    chiral_norm=self.chiral_norm, z_shift=self.z_shift,
                )
            c = self._evaluate(h)
        return [[c[m]] for m in range(self.m_max + 1)]


def _product_over_roots(delta, roots):
    """Compute ∏(delta − root) staying in mp.mpf when all inputs are real."""
    all_real = (
        not isinstance(delta, mp.mpc) or delta.imag == 0
    ) and all(
        not isinstance(r, mp.mpc) or r.imag == 0 for r in roots
    )
    if all_real:
        d = mp.mpf(mp.re(delta) if isinstance(delta, mp.mpc) else delta)
        acc = mp.mpf(1)
        for r in roots:
            r_real = mp.mpf(mp.re(r) if isinstance(r, mp.mpc) else r)
            acc = acc * (d - r_real)
        return acc
    d = mp.mpc(delta)
    acc = mp.mpc(1)
    for r in roots:
        acc = acc * (d - mp.mpc(r))
    return acc


def _partition_force_keep_indices(roots, shift) -> tuple[list[int], list[int]]:
    """Split roots into forced real-positive-shifted poles and the selectable rest."""
    real_shift = mp.mpf(mp.re(shift))
    forced, selectable = [], []
    for idx, root in enumerate(roots):
        root_c = mp.mpc(root)
        if mp.im(root_c) == 0 and mp.re(root_c) > real_shift:
            forced.append(idx)
        else:
            selectable.append(idx)
    return forced, selectable


def _origin_snapped_for_nodes(prefactor):
    """The prefactor the nodes are placed for: real poles in the node code's near-origin band put at 0.

    Nodes are only where the interpolant is sampled (it is exact at whichever nodes are used), so this changes no
    value.  An offset minimal-model c (the superconformal c + eps path) has genuine real poles O(eps) from the
    origin; the Bohr--Sommerfeld density counts them as a delta at 0 but its integrated form does not, so the
    node code refuses them.  The table itself keeps them at their true location.
    """
    band = mp.mpf("1e-10")      # get_sample_points' small_pole_threshold
    if not any(mp.im(p) == 0 and 0 < abs(p) < band for p in prefactor.poles):
        return prefactor
    poles = tuple(mp.mpf(0) if (mp.im(p) == 0 and abs(p) < band) else p for p in prefactor.poles)
    return DampedRational(constant=prefactor.constant, poles=poles, base=prefactor.base)


def _global_bucket_task(job):
    """Build the interpolating polynomials for one node-count bucket.

    All (m, n) entries in a bucket share the same node count, so the optimal
    nodes, block samples, and Lagrange basis are computed once and reused
    across the entries. Self-contained and picklable so it runs in a worker
    process. Returns {(m, n): (Polynomial, sample_tuple)}.
    """
    (dps, prefactor, forced_roots, shift, work_dps, sampler, count, entries) = job
    mp.mp.dps = dps
    sample_x = get_sample_points(_origin_snapped_for_nodes(prefactor), count, work_dps=work_dps)
    from .interpolation import _shifted_sampling_points
    sample_val = _shifted_sampling_points(sample_x, shift)
    # Entry-restricted sampling: `sample_bucket` builds only this bucket's
    # (m, n) entries at each node and returns `(tables, is_arb)` -- arb-native
    # grids on the common real path, else mpmath.
    sample_bucket = getattr(sampler, "sample_bucket", None)
    result = sample_bucket(sample_val, entries) if sample_bucket is not None else None
    if result is None:
        # No bucket sampler (or it declined, e.g. singular-c large spin): use
        # the per-node mpmath path.
        tables = [sampler(v) for v in sample_val]
        is_arb = False
    else:
        tables, is_arb = result
    basis, denoms, arb_basis, arb_denoms = precompute_lagrange_basis(
        sample_x,
    )
    n_pts = len(sample_val)
    # The block values divide by the prefactor at the sample points; do that
    # (and the Lagrange sum) in arb when the grids are arb, avoiding the
    # arb<->mpf round trip.
    pref_arb = (
        prefactor.evaluate_arb_batch([mp.mpf(x) for x in sample_x])
        if is_arb else None
    )
    def _as_mpmath(grids):
        # the mpmath route divides by mpmath prefactor values and records mpmath samples
        return [[[_arb_to_mpf(v) if hasattr(v, "mid") else v for v in row] for row in grid] for grid in grids]

    if pref_arb is None:
        if is_arb:
            tables = _as_mpmath(tables)
        is_arb = False
        pref_values = [prefactor.evaluate(mp.mpf(x)) for x in sample_x]
    if forced_roots:
        forced_factor = [_product_over_roots(v, forced_roots) for v in sample_val]
    else:
        forced_factor = None
    if is_arb and forced_factor is not None:
        from flint import arb as arb_type
        try:
            forced_factor_arb = [arb_type(v) for v in forced_factor]
        except TypeError:  # a complex forced-root product (mp.mpc) has no arb form
            is_arb = False
            tables = _as_mpmath(tables)
            pref_values = [prefactor.evaluate(mp.mpf(x)) for x in sample_x]
            forced_factor_arb = None
    else:
        forced_factor_arb = None
    sample_tuple = tuple(sample_val)
    out = {}
    for (m, n) in entries:
        if is_arb:
            if forced_factor_arb is None:
                reduced = [tables[i][m][n] / pref_arb[i] for i in range(n_pts)]
            else:
                reduced = [
                    tables[i][m][n] * forced_factor_arb[i] / pref_arb[i]
                    for i in range(n_pts)
                ]
        else:
            if forced_factor is None:
                reduced = [tables[i][m][n] / pref_values[i] for i in range(n_pts)]
            else:
                reduced = [
                    tables[i][m][n] * forced_factor[i] / pref_values[i]
                    for i in range(n_pts)
                ]
        poly = lagrange_polynomial_from_basis(
            basis, denoms, reduced,
            _arb_basis=arb_basis, _arb_denoms=arb_denoms,
        )
        # the fit's node samples (raw and over the prefactor), so consumers reuse them rather than re-evaluate
        out[(m, n)] = (poly, sample_tuple, _raw_entry_samples(tables, is_arb, m, n, reduced))
    return out


def _contribution_optimal_prefactor(
    full, npoles, scores, *, selectable_roots, forced_roots, shift, base, m_max,
    selection_base=None,
):
    """``contribution_optimal`` kept-pole set for the shared (tower) core.

    ``select_kept_poles`` cannot serve this selector: the others rank poles by a
    per-pole score and take the top ``npoles``, whereas this one builds ONE
    nested greedy order against a scale-adaptive reference measure and takes an
    ``npoles``-length prefix of it. That construction lives in
    :mod:`virasoro_blocks.polyderivs` for the fixed-spin path; it is reused here
    verbatim rather than reimplemented, so the two paths cannot drift.

    Valid only when the candidate pool is npoles-INDEPENDENT, which the caller
    asserts via ``fixed_candidate_pool``. The Virasoro large-spin tower and
    the fixed-spin global path qualify: their finite Kac / SL(2) pools are
    bounded by ``q_order``. The global large-spin ladder still does not.

    ``deval`` is taken at the shifted origin, i.e. the positivity threshold
    itself -- for the tower that is ``L = Lmax``, the left edge of the region
    positivity is enforced on, which is the analogue of the fixed-spin path's
    ``delta_eval`` default at the unitarity threshold.
    """
    from .polyderivs import (
        _contribution_optimal_adaptive_order,
        _take_contribution_optimal_prefix,
    )

    ranking_full = full
    if selection_base is not None and selection_base != base:
        ranking_full = DampedRational(
            constant=mp.mpc(1), poles=full.poles, base=selection_base,
        )
    greedy_order = _contribution_optimal_adaptive_order(
        scores,
        [mp.mpc(p) - mp.mpc(shift) for p in selectable_roots],
        [mp.mpc(p) - mp.mpc(shift) for p in forced_roots],
        ranking_full,
        node_excess=int(m_max) + 1,
        base=selection_base if selection_base is not None else base,
        deval=mp.mpc(0),
        target_slots=int(npoles),
        epoch_size=8,
        exact_fill=True,
    )
    return _take_contribution_optimal_prefix(full, greedy_order, int(npoles))


@_quiet_conditioning
def _interpolate_global_table(
    *, sampler, m_max, n_max, base, shift, pool, residue_scorer,
    entry_power, npoles, pole_selection, num_threads=1,
    shared_sample_points=False, kept_entries=None,
    contribution_via_contour=False, force_keep_real_positive=False,
    fixed_candidate_pool=False, selection_sampler=None,
    selection_base=None, selection_m_max=None, selection_n_max=None,
    block_pool=None,
) -> InterpolatedDeltaTable:
    """Generic optimal-node interpolation of a numeric global block.

    Shared core for every global basis. `sampler(value)` returns the numeric
    block table (indexed [m][n]; chiral_F uses n_max = 0); `pool` is the
    genuine-pole multiset in the interpolation variable; `entry_power(m, n)`
    is the block's leading large-variable power a_mn, so each entry's
    polynomial degree is a_mn + npoles (the unique degree preserving the tail
    power). `pole_selection` keeps `npoles` poles via the shared
    `select_kept_poles`; `residue_scorer` is only invoked for
    residue / contribution.

    `fixed_candidate_pool` asserts that `pool` does not depend on `npoles`,
    which is the precondition `contribution_optimal` needs (it builds ONE nested
    greedy order and takes an `npoles`-length prefix, so a pool that grows with
    `npoles` would have it ranking against a different set at every count). The
    fixed-spin SL(2) global pool is the finite set first appearing through
    `q_order`, as is the Virasoro large-spin tower's Kac pool. The global
    large-spin ladder remains npoles-dependent and does not qualify.

    `block_pool`, when the candidate `pool` reaches past the sampled block's own
    truncation (the NS/Ramond pools run two levels deeper), is the multiset of
    poles the block itself has. A forced pole outside it has a structurally zero
    principal part, and is left out of the table (its prefactor would only raise
    the written degree and the monomial form's cancellation).

    `num_threads > 1` dispatches the per-node-count buckets to a multiprocessing
    pool (the buckets are independent: each runs one optimal-node search,
    samples the block at those nodes, and fits its entries). The output is
    identical to the serial path; `sampler` must be picklable (the
    `_GlobalBlockSampler*` classes are).
    """
    num_threads = validate_worker_count(num_threads)
    if pole_selection not in POLE_SELECTIONS:
        raise ValueError(
            f"pole_selection must be one of {POLE_SELECTIONS}; "
            f"got {pole_selection!r}"
        )
    if pole_selection == "contribution_optimal" and not fixed_candidate_pool:
        raise ValueError(
            "contribution_optimal requires a candidate pole set that does NOT "
            "depend on npoles: it builds one nested greedy order and takes a "
            "prefix of it, so a pool that grows with npoles would rank against "
            "a different set at every npoles. Pass fixed_candidate_pool=True "
            "only when the caller's pool is npoles-independent."
        )
    all_roots = list(pool)
    score_sampler = sampler if selection_sampler is None else selection_sampler
    score_base = base if selection_base is None else selection_base
    score_m_max = m_max if selection_m_max is None else int(selection_m_max)
    score_n_max = n_max if selection_n_max is None else int(selection_n_max)
    if selection_sampler is not None:
        residue_scorer = _make_residue_scorer(
            score_sampler, score_base, score_m_max, score_n_max,
            score_entries=pole_selection_score_entries(pole_selection),
        )
    if force_keep_real_positive:
        forced_idx, selectable_idx = _partition_force_keep_indices(all_roots, shift)
    else:
        forced_idx, selectable_idx = [], list(range(len(all_roots)))
    forced_roots = [all_roots[i] for i in forced_idx]
    if block_pool is not None:
        forced_roots = _poles_of_block(forced_roots, block_pool)
    selectable_roots = [all_roots[i] for i in selectable_idx]
    # blocks_3d --lambda truncation: only the kept derivative triangle is ever
    # emitted, so interpolate only those entries (the discarded corner of the
    # square never reaches any writer).
    if kept_entries is None:
        coord_entries = [
            (m, n) for m in range(m_max + 1) for n in range(n_max + 1)
        ]
    else:
        coord_entries = [
            (m, n) for (m, n) in kept_entries if m <= m_max and n <= n_max
        ]
    # The forced poles split off as exact principal parts (the NS/Ramond poles on x > 0 at c < 3/2), as the
    # Virasoro c <= 1 tables do (polyderivs.interpolate_delta_tables): the pole selection still ranks the selectable
    # poles on the block itself, while the fit order-reduces the regular remainder, whose poles are all selectable,
    # and _recombine_principal_parts puts the parts back.  Fitting the forced poles into the numerator instead (the
    # fallback, when the sampler has no principal parts) makes the fitted function vary by prod(x - p) across nodes
    # that follow only the kept poles: such a table lost every digit at any precision.
    split = _split_global_principal_parts(sampler, forced_roots, coord_entries, base) if forced_roots else None
    fit_forced_roots = [] if split is not None else forced_roots
    if npoles is not None and int(npoles) >= len(selectable_roots):
        # every pole is kept: nothing to rank (the contribution scorers were ~75% of an NS table build that keeps
        # its whole pool); the position-based selector keeps them all
        pole_selection = "rightmost"
    shifted_all = DampedRational(
        constant=mp.mpc(1), poles=tuple(all_roots), base=base
    ).shift_x(shift)
    full = DampedRational(
        constant=mp.mpc(1), poles=tuple(selectable_roots), base=base
    ).shift_x(shift)
    scores = None
    if pole_selection in (
        "contribution", "contribution_all",
        # Geometry-weighted variants: they use the SAME principal-part scores,
        # then rescale them below (without them in this list the shared core
        # would hand select_kept_poles scores=None and raise).
        "contribution_physical", "contribution_bernstein",
        "contribution_equilibrium", "contribution_density",
        "contribution_potential",
        # contribution_optimal ranks against the same principal-part scores; it
        # differs only in how it TURNS them into a kept set (a nested greedy
        # order rather than a top-npoles cut), so it needs them computed here too.
        "contribution_optimal",
    ):
        # Full-principal-part contribution scores for direct selectors and the
        # FULL principal part at the evaluation point (the unitarity threshold =
        # shift). For a SIMPLE pole the full principal part is just the leading
        # residue, so this is identical to contribution_leading -- and we take
        # the cheap real-axis residue+distance path (no complex contour needed).
        # This is the common case for the global tower (every twist gives a
        # distinct, simple pole in L -- including each member of a complex
        # conjugate pair, which is a distinct location) and lets the real-axis /
        # complex-point residue+distance path serve the full contribution. Only
        # genuine repeated poles (same location, e.g. global ℓ=0 double poles)
        # need the contour/DFT scorer, which evaluates the block at COMPLEX points
        # -- supported by the global 2d sampler and the Virasoro tower sampler.
        #
        # `contribution_via_contour` forces the contour/DFT scorer regardless of
        # multiplicity: the Virasoro tower (whose chiral block is regularized at a
        # singular c) sets it so the scorer probes the block at a *finite* radius
        # from each pole -- exactly the fixed-spin contribution path -- instead of
        # the eps-probe residue scorer, which evaluates the block ~1e-25 from a
        # genuine Kac pole and breaks the epsilon-extrapolation there.
        from .polyderivs import (
            _residue_score_indices_from_dims,
            _value_score_entry_from_sampler,
        )
        value_entry = _value_score_entry_from_sampler(
            score_sampler, n_max=score_n_max,
        )
        entries = _residue_score_indices_from_dims(
            score_m_max,
            score_n_max,
            pole_selection_score_entries(pole_selection),
            value_entry=value_entry if value_entry is not None else (0, 0),
        )

        _, max_mult = _pole_location_multiplicities(all_roots)

        # Analytic fast path (Virasoro tower AND the global 2d / SL(2) tower):
        # build each pole's full principal part in closed form from the chiral
        # residue/regular series (Kac data for Virasoro, the hypergeometric
        # Gamma(2h) residue + regularized 2F1 for global) --
        # exact and ~one evaluation per pole, replacing both the eps-probe (which
        # probes ~1e-25 from a Kac pole) and the contour/DFT. At a singular c the
        # Virasoro tower uses the merged higher-order chiral Laurent variant
        # (still analytic, covering the diagonal, off-diagonal, and folded
        # coordinates), so the analytic path is ALWAYS attempted -- even when
        # `contribution_via_contour` is set. It declines (None) for the global
        # SL(2) tower (no Kac pole_data), a folded complex parity phase, the
        # c = 1 divergence where even the block sampler diverges, and only THEN does `contribution_via_contour` force the contour over
        # the eps-probe below.
        all_scores = None
        analytic = getattr(score_sampler, "_analytic_residue_scores", None)
        if analytic is not None:
            all_scores = analytic(list(all_roots), shift, entries, score_base)

        if all_scores is not None:
            scores = [all_scores[i] for i in selectable_idx]
        elif max_mult <= 1 and not contribution_via_contour:
            # SIMPLE poles (the common tower case): full principal part = leading
            # residue, so the cheap real-axis eps-probe residue+distance serves.
            raw = residue_scorer(all_roots)
            all_scores = [
                s / float(abs(mp.mpc(q))) if abs(mp.mpc(q)) > 1e-12 else s
                for s, q in zip(raw, shifted_all.poles)
            ]
            scores = [all_scores[i] for i in selectable_idx]
        else:
            # Repeated poles or forced contour (singular-c regularized tower):
            # the contour/DFT scorer, evaluating the block at COMPLEX points.
            from .polyderivs import principal_part_contribution_scores

            sample_bucket = getattr(score_sampler, "sample_bucket", None)

            def _eval_block(pts, ent):
                # sample_bucket may exist yet decline (return None) -- e.g. the
                # Virasoro tower's singular-c path is per-node only -- so fall
                # back to the per-value sampler in that case.
                if sample_bucket is not None:
                    res = sample_bucket(list(pts), ent)
                    if res is not None:
                        return res
                return [score_sampler(p) for p in pts], False

            all_scores = principal_part_contribution_scores(
                list(all_roots), _eval_block, delta_eval=shift, entries=entries,
                exp_base=score_base, n_select=npoles,
            )
            scores = [all_scores[i] for i in selectable_idx]
    if scores is not None and pole_selection == "contribution_potential":
        # Use the same finite-support BS remainder as the fixed-spin path.
        # `full` is already shifted, so its poles and the provisional kept set
        # are in the x = L - shift variable used by the node search.
        from .polyderivs import _finite_bs_potential_scores

        try:
            _a = max(
                int(entry_power(m, n))
                for m in range(m_max + 1)
                for n in range(n_max + 1)
            )
        except Exception:
            _a = max(int(m_max), int(n_max))
        _n_tot = max(2, int(npoles) + _a + 1)
        try:
            _ts = [mp.mpf(t) for t in get_sample_points(full, _n_tot)]
        except Exception:
            _ts = []
        if _ts:
            cap = min(int(npoles), len(full.poles))
            forced_shifted = [mp.mpc(p) - mp.mpc(shift) for p in fit_forced_roots]

            def same_pole_set(left, right):
                if len(left) != len(right):
                    return False
                remaining = list(right)
                tol = tolerance(guard=10)
                for pole in left:
                    for index, other in enumerate(remaining):
                        if abs(mp.mpc(pole) - mp.mpc(other)) <= tol:
                            remaining.pop(index)
                            break
                    else:
                        return False
                return not remaining

            provisional = select_kept_poles(
                full, cap, "contribution", scores=scores
            )
            scored = scores
            history = []
            for _ in range(2):
                kept_for_cost = list(provisional.poles) + forced_shifted
                scored = _finite_bs_potential_scores(
                    scores, list(full.poles), _ts, kept_for_cost,
                    base=base, deval=mp.mpc(0),
                )
                next_kept = select_kept_poles(
                    full, cap, "contribution", scores=scored
                )
                history.append((next_kept, scored))
                if same_pole_set(provisional.poles, next_kept.poles):
                    provisional = next_kept
                    break
                provisional = next_kept
            if len(history) > 1 and not same_pole_set(
                history[-2][0].poles, history[-1][0].poles
            ):
                def dropped_cost(item):
                    candidate, cost_map = item
                    remaining = list(candidate.poles)
                    dropped = []
                    for pole, cost in zip(full.poles, cost_map):
                        for index, kept_pole in enumerate(remaining):
                            if abs(pole - kept_pole) <= tolerance(guard=10):
                                remaining.pop(index)
                                break
                        else:
                            dropped.append(cost)
                    return max(dropped, default=mp.mpf(0))

                _, scores = min(history, key=dropped_cost)
            else:
                scores = scored

    elif pole_selection in (
        "residue", "residue_all", "contribution_leading",
        "contribution_leading_all",
    ):
        from .polyderivs import (
            _residue_score_indices_from_dims,
            _value_score_entry_from_sampler,
        )

        all_scores = None
        analytic_leading = getattr(sampler, "_analytic_leading_residue_scores", None)
        if analytic_leading is not None:
            value_entry = _value_score_entry_from_sampler(sampler, n_max=n_max)
            entries = _residue_score_indices_from_dims(
                m_max,
                n_max,
                pole_selection_score_entries(pole_selection),
                value_entry=value_entry if value_entry is not None else (0, 0),
            )
            all_scores = analytic_leading(list(all_roots), entries, base)
        if all_scores is None:
            all_scores = residue_scorer(all_roots)
        if pole_selection_is_contribution(pole_selection):
            all_scores = [
                s / float(abs(mp.mpc(q))) if abs(mp.mpc(q)) > 1e-12 else s
                for s, q in zip(all_scores, shifted_all.poles)
            ]
        scores = [all_scores[i] for i in selectable_idx]
    # `npoles` may exceed the selectable genuine-pole count for special externals;
    # forced real-positive poles are always kept on top.
    npoles = min(npoles, len(full.poles))
    if pole_selection == "contribution_optimal":
        # (split off as principal parts, the forced poles are not in the fit's prefactor, so not in the ranking's
        # reference measure either)
        prefactor = _contribution_optimal_prefactor(
            full, npoles, scores,
            selectable_roots=selectable_roots, forced_roots=fit_forced_roots,
            shift=shift, base=base, m_max=m_max,
            selection_base=selection_base,
        )
    else:
        prefactor = select_kept_poles(full, npoles, pole_selection, scores=scores)
    real_shift = mp.mpf(mp.re(shift))
    if fit_forced_roots:
        combined_prefactor = DampedRational(
            constant=prefactor.constant,
            poles=tuple(prefactor.poles) + tuple(mp.mpc(p) - real_shift for p in fit_forced_roots),
            base=prefactor.base,
        )
    else:
        combined_prefactor = prefactor
    # the fitted function's poles: the split-off forced poles are not among them
    effective_npoles = prefactor.pole_degree + len(fit_forced_roots)

    # Group (m, n) by node count (= polynomial degree + 1). The node search runs
    # at a fraction of the working precision (node positions only pick which
    # near-optimal nodes are used; the polynomial is exact at them, so block
    # accuracy is set by the pole count, not node precision), matching the
    # Virasoro path.
    work_dps = _default_node_search_dps()
    buckets: dict[int, list] = {}
    for (m, n) in coord_entries:
        count = max(0, entry_power(m, n)) + effective_npoles + 1
        buckets.setdefault(count, []).append((m, n))

    if shared_sample_points and buckets:
        # Pad every (m, n) onto the single max-degree node set (computed once)
        # instead of its own degree-matched set -- the global-block analog of
        # interpolate_delta_table's shared_sample_points. More accurate inside
        # the node range, Runge-prone far beyond it (see the Virasoro path).
        max_count = max(buckets)
        buckets = {
            max_count: [mn for entries in buckets.values() for mn in entries]
        }

    fit_sampler = sampler if split is None else split[1]
    jobs = [
        (mp.mp.dps, prefactor, fit_forced_roots, shift, work_dps, fit_sampler, count, entries)
        for count, entries in buckets.items()
    ]
    poly_map: dict = {}
    worker_count = usable_workers(num_threads)
    if worker_count > 1 and len(jobs) > 1:
        # Use multiprocessing.Pool rather than concurrent.futures' process
        # executor: the latter performs an eager POSIX named-semaphore limit
        # check, which can be inaccessible in a restricted macOS session even
        # though ordinary multiprocessing is available.
        context = multiprocessing.get_context()
        with context.Pool(
            processes=min(worker_count, len(jobs)), **pool_initializer_kwargs()
        ) as pool_exec:
            for sub in pool_exec.map(_global_bucket_task, jobs):
                poly_map.update(sub)
    else:
        for job in jobs:
            poly_map.update(_global_bucket_task(job))

    # Corner entries outside the kept triangle were never interpolated; fill a
    # zero placeholder so the grid stays rectangular for positional consumers
    # (every writer reads only the kept pairs).
    max_sample_points = poly_map[buckets[max(buckets)][0]][1] if buckets else ()
    placeholder = (Polynomial.zero(field=mp.mpf), max_sample_points, None)
    numerators = []
    numerator_sample_points = []
    numerator_sample_values = []
    for m in range(m_max + 1):
        row = []
        sample_row = []
        value_row = []
        for n in range(n_max + 1):
            poly, sample_tuple, values = poly_map.get((m, n), placeholder)
            row.append(poly)
            sample_row.append(sample_tuple)
            value_row.append(values)
        numerators.append(row)
        numerator_sample_points.append(sample_row)
        numerator_sample_values.append(value_row)

    table = InterpolatedDeltaTable(
        prefactor=combined_prefactor,
        shift=shift,
        sample_points=max_sample_points,
        numerators=numerators,
        numerator_sample_points=numerator_sample_points,
        numerator_sample_values=numerator_sample_values,
    )
    if split is not None:
        from .polyderivs import _recombine_principal_parts

        table = _recombine_principal_parts(table, split[0], forced_roots)
    return table


def _poles_of_block(roots, block_pool):
    """The sub-multiset of ``roots`` that ``block_pool`` (the poles of the sampled block, each repeated by its order)
    contains: the other roots are poles of a deeper truncation only (matched at the working precision; both lists
    come from the same pole data)."""
    remaining = [mp.mpc(p) for p in block_pool]
    tol = tolerance(guard=10)
    kept = []
    for root in roots:
        z = mp.mpc(root)
        for index, other in enumerate(remaining):
            if abs(other - z) <= tol * max(1, abs(z)):
                remaining.pop(index)
                kept.append(root)
                break
    return kept


def _split_global_principal_parts(sampler, forced_roots, entries, base):
    """``(principal_parts, remainder_sampler)`` with the forced poles' exact principal parts split off the block
    (the sampler's ``_analytic_principal_parts``, in :func:`polyderivs._principal_parts_record` form), or None when
    the sampler cannot supply them (the forced poles are then fit into the numerator)."""
    from .polyderivs import _RemainderSampler, _principal_parts_record

    principal = getattr(sampler, "_analytic_principal_parts", None)
    parts = None if principal is None else principal(list(forced_roots), list(entries), base)
    if parts is None:
        return None
    principal_parts = _principal_parts_record(base, list(entries), parts)
    return principal_parts, _RemainderSampler(sampler, principal_parts)


# --- 2d bases: scalar_G, ordered_B, parity_projected_B -----------------------
#
# All three are k_h(z) k_hbar(zbar) + sigma * k_hbar(z) k_h(zbar) for a single
# real/complex coefficient sigma: 0 (ordered B, no parity partner),
# +1 (scalar G; 0 for its public spin-0 block), or sign*phase (parity-projected B). They share the
# pole structure, base, and interpolation core; only sigma, the per-half
# externals, and the leading-power cancellation differ.


def _global_2d_interpolated_table(
    coordinate, ell, h_externals, hbar_externals, m_max, n_max, *,
    sigma, chiral_norm, z_shift, zbar_shift, q_order, npoles, pole_selection, shift,
    num_threads=1, shared_sample_points=False, kept_entries=None,
    fold_parity=None, center=None, center_bar=None, basis="scalar_G",
    parity=None, selection_coordinate=None, selection_m_max=None,
    selection_n_max=None,
) -> InterpolatedDeltaTable:
    ell = _signed_half_integer(ell)
    folded = coordinate in _FOLDED_2D_TO_CHIRAL
    if not folded and coordinate not in _CHIRAL_TAYLOR_2D:
        raise ValueError(
            f"global block coordinate must be one of "
            f"{sorted(set(_CHIRAL_TAYLOR_2D) | set(_FOLDED_2D_TO_CHIRAL))}; "
            f"got {coordinate!r}"
        )
    if folded and fold_parity is None:
        raise ValueError(
            f"folded global coordinate {coordinate!r} requires a fold_parity "
            "(0 for an even / '+' block, 1 for an odd / '-' block)"
        )
    if folded and (center is not None or center_bar is not None):
        raise ValueError(
            "off-diagonal centers are supported only on non-folded "
            "coordinates (zzb/yyb/qqb)"
        )
    if n_max is None:
        n_max = m_max
    m_max = _validate_derivative_order("m_max", m_max)
    n_max = _validate_derivative_order("n_max", n_max)
    if selection_m_max is not None:
        selection_m_max = _validate_derivative_order("selection_m_max", selection_m_max)
    if selection_n_max is not None:
        selection_n_max = _validate_derivative_order("selection_n_max", selection_n_max)
    npoles = _validate_npoles(npoles)
    q_order = DEFAULT_GLOBAL_Q_ORDER if q_order is None else _validate_q_order(q_order)
    base = global_block_exp_base(
        chiral_norm,
        center=center,
        center_bar=center_bar,
    )
    from .serialization import _real_serialization_number
    shift = _real_serialization_number(abs(ell) if shift is None else shift, "shift")
    from .numeric_input import literal_number

    h_externals, hbar_externals = tuple(h_externals), tuple(hbar_externals)
    h_labels = [literal_number(h) for h in h_externals]
    hbar_labels = [literal_number(h) for h in hbar_externals]
    h_ext = h_labels
    hbar_ext = hbar_labels

    sampler = _GlobalBlockSampler2d(
        coordinate, ell, h_ext, hbar_ext, sigma, chiral_norm, z_shift,
        zbar_shift, m_max, n_max,
        folded=folded, fold_parity=(0 if fold_parity is None else fold_parity),
        center=center, center_bar=center_bar, basis=basis, parity=parity,
    )
    selection_sampler = None
    selection_base = None
    if selection_coordinate is not None:
        if selection_coordinate not in _CHIRAL_TAYLOR_2D:
            raise ValueError(
                "global selection_coordinate must be a non-folded 2d coordinate"
            )
        selection_m_max = m_max if selection_m_max is None else selection_m_max
        selection_n_max = n_max if selection_n_max is None else selection_n_max
        selection_sampler = _GlobalBlockSampler2d(
            selection_coordinate, ell, h_ext, hbar_ext, sigma, chiral_norm,
            z_shift, zbar_shift, selection_m_max, selection_n_max,
            folded=False, fold_parity=0, center=None, center_bar=None,
            basis=basis, parity=parity,
        )
        selection_base = global_block_exp_base(chiral_norm)

    # The fold is a Delta-independent linear recombination, so the genuine
    # Delta-pole multiset of the underlying non-folded block is unchanged.
    dh12, dh34 = _externals_diffs(h_externals)
    dhb12, dhb34 = _externals_diffs(hbar_externals)
    pool = _global_q_order_pole_pool(
        ell, q_order, dh12=dh12, dh34=dh34, dhb12=dhb12, dhb34=dhb34,
    )
    if pole_selection == "contribution_optimal" and npoles > len(pool):
        raise ValueError(
            f"global q_order={q_order} provides only {len(pool)} genuine "
            f"candidate poles; reduce npoles or raise q_order"
        )
    # At a common point the leading amplitudes commute, independently of
    # external weights. Distinct points retain the spin-dependent ratio of
    # their decay bases, so their leading antiparity term does not cancel.
    diagonal = center_bar is None or _global_label_equal(.5 if center is None else center, center_bar)
    cancels = diagonal and sigma == -1
    drop = 1 if cancels else 0

    if folded:
        # A folded (w, s) entry's leading Delta-power is the folded entry power
        # (alpha + 2 beta + odd) minus the symmetric-antiparity cancellation,
        # the same recombination logic as the large-spin tower's L-power.
        def entry_power(alpha, beta):
            return folded_entry_power(alpha, beta, fold_parity) - drop
    else:
        def entry_power(m, n):
            return m + n - drop

    return _interpolate_global_table(
        sampler=sampler, m_max=m_max, n_max=n_max, base=base, shift=shift,
        pool=pool, residue_scorer=_make_residue_scorer(
            sampler, base, m_max, n_max,
            score_entries=pole_selection_score_entries(pole_selection),
        ),
        entry_power=entry_power, npoles=npoles, pole_selection=pole_selection,
        num_threads=num_threads, shared_sample_points=shared_sample_points,
        kept_entries=kept_entries, fixed_candidate_pool=True,
        selection_sampler=selection_sampler, selection_base=selection_base,
        selection_m_max=selection_m_max, selection_n_max=selection_n_max,
    )


def global_scalar_G_interpolated_table(
    coordinate, ell, h_externals, m_max, n_max=None, *,
    chiral_norm=None, z_shift=None, q_order=DEFAULT_GLOBAL_Q_ORDER,
    npoles, pole_selection="contribution_optimal",
    shift=None, num_threads=1, shared_sample_points=False, kept_entries=None,
    center=None, center_bar=None, selection_coordinate=None,
    selection_m_max=None, selection_n_max=None,
) -> InterpolatedDeltaTable:
    """Order-reduced 2d global *scalar* block G = k_h k_hbar + sigma (swap).

    Scalar externals (h = hbar), so sigma = +1 at every ell != 0 (0 at ell = 0,
    where the block is k_h k_h; see :func:`~._conventions.scalar_G_swap_coefficient`) and
    both chiral halves share the external differences. ``q_order`` fixes the finite candidate
    SL(2) pole pool; it does not truncate the exact 2F1 evaluation. See
    `_global_2d_interpolated_table` and the module docstring for the base,
    pole, and degree conventions.
    `num_threads > 1` parallelizes the per-degree node buckets.
    """
    ell = _integer_spin(ell)
    sigma = mp.mpf(scalar_G_swap_coefficient(ell))
    # Scalar block: z <-> z̄ even at every spin, so the fold keeps it as is.
    fold_parity = fold_parity_marker(ell=ell)
    return _global_2d_interpolated_table(
        coordinate, ell, h_externals, h_externals, m_max, n_max, sigma=sigma,
        chiral_norm=chiral_norm, z_shift=z_shift, zbar_shift=z_shift,
        q_order=q_order, npoles=npoles, pole_selection=pole_selection, shift=shift,
        num_threads=num_threads, shared_sample_points=shared_sample_points,
        kept_entries=kept_entries, fold_parity=fold_parity,
        basis="global_scalar_G",
        center=center, center_bar=center_bar,
        selection_coordinate=selection_coordinate,
        selection_m_max=selection_m_max, selection_n_max=selection_n_max,
    )


def global_ordered_B_interpolated_table(
    coordinate, ell, h_externals, hbar_externals, m_max, n_max=None, *,
    chiral_norm=None, z_shift=None, zbar_shift=None,
    q_order=DEFAULT_GLOBAL_Q_ORDER, npoles,
    pole_selection="contribution_optimal", shift=None, num_threads=1,
    shared_sample_points=False, kept_entries=None,
    center=None, center_bar=None, selection_coordinate=None,
    selection_m_max=None, selection_n_max=None,
) -> InterpolatedDeltaTable:
    """Order-reduced 2d global *ordered* block B = k_h(z) k_hbar(zbar).

    Spinning externals: the holomorphic half uses `h_externals`, the
    antiholomorphic half uses `hbar_externals`, with no parity partner
    (sigma = 0). Each chiral half's poles are stripped by its own external
    differences. `num_threads > 1` parallelizes the per-degree node buckets.

    Ordered B has no z <-> z̄ parity partner, hence no parity fold: the folded
    coordinates `xt`/`ws`/`qqb_ws` are rejected (use `global_scalar_G` or
    `global_parity_projected_B`).
    """
    if coordinate in _FOLDED_2D_TO_CHIRAL:
        raise ValueError(
            "global_ordered_B has no parity fold; folded coordinates "
            f"({sorted(_FOLDED_2D_TO_CHIRAL)}) require global_scalar_G or "
            "global_parity_projected_B"
        )
    return _global_2d_interpolated_table(
        coordinate, ell, h_externals, hbar_externals, m_max, n_max,
        sigma=mp.mpf(0), chiral_norm=chiral_norm, z_shift=z_shift,
        zbar_shift=zbar_shift, q_order=q_order, npoles=npoles,
        pole_selection=pole_selection,
        shift=shift, num_threads=num_threads,
        shared_sample_points=shared_sample_points, kept_entries=kept_entries,
        basis="global_ordered_B",
        center=center, center_bar=center_bar,
        selection_coordinate=selection_coordinate,
        selection_m_max=selection_m_max, selection_n_max=selection_n_max,
    )


def global_parity_projected_B_interpolated_table(
    coordinate, ell, h_externals, hbar_externals, m_max, n_max=None, *,
    parity, parity_external=None, exchanged_parity="1", chiral_norm=None,
    z_shift=None, zbar_shift=None, q_order=DEFAULT_GLOBAL_Q_ORDER,
    npoles, pole_selection="contribution_optimal",
    shift=None, num_threads=1, shared_sample_points=False, kept_entries=None,
    center=None, center_bar=None, selection_coordinate=None,
    selection_m_max=None, selection_n_max=None,
) -> InterpolatedDeltaTable:
    """Order-reduced 2d global *parity-projected* block
    B_+/- = B_ordered +/- phase * B_reversed, with the same phase convention as
    the Virasoro `parity_projected_B` (`parity_reversal_phase`).
    `num_threads > 1` parallelizes the per-degree node buckets."""
    if parity not in {"+", "-"}:
        raise ValueError("parity must be '+' or '-'")
    phase = parity_reversal_phase(parity_external, exchanged_parity=exchanged_parity)
    sigma = (mp.mpf(1) if parity == "+" else mp.mpf(-1)) * phase
    # A real (+/-1) sigma keeps the block real; demote mpc(real, 0) to mpf.
    if isinstance(sigma, mp.mpc) and sigma.imag == 0:
        sigma = mp.mpf(sigma.real)
    folded = coordinate in _FOLDED_2D_TO_CHIRAL
    if folded and abs(mp.im(phase)) > _validation_tolerance(guard=5):
        raise ValueError(
            "parity-folded coordinates require a real (+/-1) total parity "
            "phase; got a non-real phase, which has no definite z<->zbar parity "
            "-- use a non-folded coordinate"
        )
    # Folded smoothing parity is sigma's sign (sigma = -1 strips one r = sqrt(s)).
    fold_parity = fold_parity_marker(parity=parity, phase=phase) if folded else None
    return _global_2d_interpolated_table(
        coordinate, ell, h_externals, hbar_externals, m_max, n_max, sigma=sigma,
        chiral_norm=chiral_norm, z_shift=z_shift, zbar_shift=zbar_shift,
        q_order=q_order, npoles=npoles, pole_selection=pole_selection, shift=shift,
        num_threads=num_threads, shared_sample_points=shared_sample_points,
        kept_entries=kept_entries, fold_parity=fold_parity,
        basis="global_parity_projected_B", parity=parity,
        center=center, center_bar=center_bar,
        selection_coordinate=selection_coordinate,
        selection_m_max=selection_m_max, selection_n_max=selection_n_max,
    )


# --- chiral basis: F(h, z) ---------------------------------------------------


def global_chiral_F_interpolated_table(
    coordinate, h_externals, m_max, *,
    chiral_norm=None, z_shift=None, q_order=DEFAULT_GLOBAL_Q_ORDER,
    npoles, pole_selection="contribution_optimal",
    shift=None, num_threads=1, shared_sample_points=False, kept_entries=None,
    center=None, selection_coordinate=None, selection_m_max=None,
) -> InterpolatedDeltaTable:
    """Order-reduced chiral half F(h, z) = k_h(z) = z^h 2F1(...).

    A single Taylor index (n_max = 0) and no exchanged spin; the interpolation
    variable is the chiral weight h, the poles are at h = -p/2 (genuine per the
    external differences), the base is the same 4 rho/N, and F^(m)(h) ~ base^h
    h^m so the per-entry power is a_m = m. shift defaults to 0 (chiral
    convention). `num_threads > 1` parallelizes the per-degree node buckets.

    ``center`` (default ``None`` = the symmetric point 1/2) recenters the single
    chiral expansion at ``z = center`` -- the global-block analog of the Virasoro
    chiral off-center table. The h = -p/2 poles are center-independent, so the
    interpolation machinery is unchanged; only the base (4 rho(z0)/N) and the
    sampled values move.
    """
    if coordinate not in _CHIRAL_TAYLOR_CHIRAL:
        raise ValueError(
            f"global chiral coordinate must be one of "
            f"{sorted(_CHIRAL_TAYLOR_CHIRAL)}; got {coordinate!r}"
        )
    m_max = _validate_derivative_order("m_max", m_max)
    if selection_m_max is not None:
        selection_m_max = _validate_derivative_order("selection_m_max", selection_m_max)
    npoles = _validate_npoles(npoles)
    q_order = DEFAULT_GLOBAL_Q_ORDER if q_order is None else _validate_q_order(q_order)
    base = global_block_exp_base(chiral_norm, center=center)
    from .serialization import _real_serialization_number
    shift = _real_serialization_number(0 if shift is None else shift, "shift")
    from .numeric_input import literal_number

    h_externals = tuple(h_externals)
    h_labels = [literal_number(h) for h in h_externals]
    h_ext = h_labels

    sampler = _GlobalBlockSamplerChiral(coordinate, h_ext, chiral_norm, z_shift,
                                        m_max, center=center)
    selection_sampler = None
    selection_base = None
    if selection_coordinate is not None:
        if selection_coordinate not in _CHIRAL_TAYLOR_CHIRAL:
            raise ValueError("global selection_coordinate must be a chiral coordinate")
        selection_m_max = m_max if selection_m_max is None else selection_m_max
        selection_sampler = _GlobalBlockSamplerChiral(
            selection_coordinate, h_ext, chiral_norm, z_shift,
            selection_m_max, center=None,
        )
        selection_base = global_block_exp_base(chiral_norm)

    d12, d34 = _externals_diffs(h_externals)
    pool = _global_chiral_q_order_pole_pool(
        q_order, delta12=d12, delta34=d34,
    )
    if pole_selection == "contribution_optimal" and npoles > len(pool):
        raise ValueError(
            f"global q_order={q_order} provides only {len(pool)} genuine "
            f"candidate poles; reduce npoles or raise q_order"
        )

    def entry_power(m, n):
        return m

    return _interpolate_global_table(
        sampler=sampler, m_max=m_max, n_max=0, base=base, shift=shift,
        pool=pool, residue_scorer=_make_residue_scorer(
            sampler, base, m_max, 0,
            score_entries=pole_selection_score_entries(pole_selection),
        ),
        entry_power=entry_power, npoles=npoles, pole_selection=pole_selection,
        num_threads=num_threads, shared_sample_points=shared_sample_points,
        kept_entries=kept_entries, fixed_candidate_pool=True,
        selection_sampler=selection_sampler, selection_base=selection_base,
        selection_m_max=selection_m_max, selection_n_max=0,
    )


def build_global_sampler(
    coordinate, *, basis, ell, h_externals, hbar_externals, m_max, n_max,
    chiral_norm=None, z_shift=None, zbar_shift=None,
    parity=None, parity_external=None, exchanged_parity="1",
    center=None, center_bar=None,
):
    """Construct the numeric global-block node sampler for one global basis.

    Resolves the per-basis ``sigma`` (parity-projection coefficient) and fold
    parity exactly as the ``global_*_interpolated_table`` builders do --
    ``sigma = +1`` for ``global_scalar_G`` (0 for its public spin-0 block,
    :func:`~._conventions.scalar_G_swap_coefficient`), ``0`` for ``global_ordered_B``,
    and ``(+/-1) * parity_reversal_phase`` for ``global_parity_projected_B`` --
    then returns a :class:`_GlobalBlockSampler2d` (or
    :class:`_GlobalBlockSamplerChiral` for ``global_chiral_F``). The returned
    sampler is callable at a single Delta (or h) and returns the exact 2F1
    block grid. The 2d sampler also provides ``sample_bucket(values, entries)``
    for batched, entry-restricted evaluation.
    """
    from .numeric_input import literal_number

    m_max = _validate_derivative_order("m_max", m_max)
    n_max = _validate_derivative_order("n_max", n_max)
    if basis == "global_chiral_F":
        return _GlobalBlockSamplerChiral(
            coordinate, [literal_number(h) for h in h_externals], chiral_norm, z_shift,
            m_max, center=center,
        )
    if COORDINATES[coordinate].kind == "radial":
        # A radial slice uses the n = 0 row of its folded parent, including
        # that coordinate's derivative normalization.
        coordinate, n_max = coordinate[: -len("_radial")], 0
    folded = coordinate in _FOLDED_2D_TO_CHIRAL
    h_ext = [literal_number(h) for h in h_externals]
    if basis == "global_scalar_G":
        ell = _integer_spin(ell)
        sigma = mp.mpf(scalar_G_swap_coefficient(ell))
        hbar_ext = h_ext
        zbar_shift = z_shift
        fold_parity = fold_parity_marker(ell=ell) if folded else 0
    elif basis == "global_ordered_B":
        if folded:
            raise ValueError(
                "global_ordered_B has no parity fold; folded coordinates require "
                "global_scalar_G or global_parity_projected_B"
            )
        sigma = mp.mpf(0)
        hbar_ext = [literal_number(h) for h in hbar_externals]
        fold_parity = 0
    elif basis == "global_parity_projected_B":
        if parity not in {"+", "-"}:
            raise ValueError("parity must be '+' or '-'")
        phase = parity_reversal_phase(parity_external, exchanged_parity=exchanged_parity)
        sigma = (mp.mpf(1) if parity == "+" else mp.mpf(-1)) * phase
        if isinstance(sigma, mp.mpc) and sigma.imag == 0:
            sigma = mp.mpf(sigma.real)
        hbar_ext = [literal_number(h) for h in hbar_externals]
        fold_parity = fold_parity_marker(parity=parity, phase=phase) if folded else 0
    else:
        raise ValueError(f"unknown global basis {basis!r}")
    return _GlobalBlockSampler2d(
        coordinate, ell, h_ext, hbar_ext, sigma, chiral_norm, z_shift,
        zbar_shift, m_max, n_max,
        folded=folded, fold_parity=fold_parity,
        center=center, center_bar=center_bar, basis=basis, parity=parity,
    )
