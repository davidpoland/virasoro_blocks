"""Generic level-indexed pole recursion: the kernel shared by Virasoro and NS.

Both the Virasoro Zamolodchikov q-recursion and the N=1 Neveu-Schwarz elliptic
recursion have the same shape. Write the elliptic block as a series in a
variable ``t`` (``t = q`` for Virasoro, ``t = q^(1/2)`` for NS), and both read

    H^sigma(x, t) = g^sigma(t)
                  + sum_j  K_j / (x - d_j) * t^(L_j) * H^(sigma_j)(v_j, t)

where the sum runs over a finite set of *poles* j, each carrying

    L_j   integer level index    (the t-power it contributes)
    d_j   pole location in x     (the exchanged weight)
    v_j   intermediate weight    (where the inner block is evaluated)
    K_j   residue numerator      (any constant powers absorbed)

Nothing in the recursion itself depends on where those four numbers came from,
on how the poles are labeled, or on what ``t`` means physically -- so this
module implements the four non-trivial generic operations once:

* :func:`fixed_pole_weights` -- ``K_j / (x - d_j)`` for one evaluation point
* :func:`interaction_weights` -- the ``K_j / (v_i - d_j)`` matrix
* :func:`fill_level_table` -- the bottom-up fill of ``H(v_i, t)``
* :func:`eval_level_series` -- one evaluation at an arbitrary ``x``

:mod:`virasoro_blocks.recursion` (Virasoro) and
:mod:`virasoro_blocks.recursion_ns` (NS) both delegate to these. The Virasoro
module additionally carries flint-vectorized and strip-symmetric fast paths
which are *its own* performance specializations, not part of this contract.

Two generalizations over the Virasoro case are needed by NS and are no-ops for
Virasoro:

**Sectors.** The NS recursion couples two block types: a pole of odd level
sends the even block to the odd one and vice versa. So a pole entry may carry a
``"sector"`` and the family may restrict which (outer, inner) pairs interact.
With one sector and no restriction this reduces to the Virasoro case exactly.

**Series seeds.** Virasoro seeds the recursion with ``H = 1 + ...``, i.e. the
constant 1. NS seeds it with ``g^1(t) = theta_3(q^2)`` and ``g^(1/2)(t) = 0``,
which are *series* in ``t``. So the seed is a per-sector coefficient list;
``[1, 0, 0, ...]`` recovers Virasoro.

Note on the pole-dict key ``"rs"``: it is the integer level index ``L_j``, kept
under that name because it is literally ``r*s`` for *both* families -- Virasoro
poles sit at level ``r*s`` in ``q``, NS poles at level ``r*s/2`` in ``q``, i.e.
``r*s`` in ``q^(1/2)``. The level *unit* is what differs, and that lives on the
family (:attr:`PoleFamily.level_unit`), never in the recursion.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from fractions import Fraction
from typing import Protocol, Sequence

import mpmath as mp


class PoleFamily(Protocol):
    """What a block family must supply for the kernel below to run.

    ``level_unit`` is the q-power carried by one unit of a pole's level index:
    ``Fraction(1)`` for Virasoro, ``Fraction(1, 2)`` for NS. It is consumed only
    when a coefficient list is turned back into a function of ``q`` (and by the
    order gate) -- never inside the recursion, which is purely index-based.
    """

    name: str
    level_unit: Fraction
    n_sectors: int

    def enumerate_poles(self, order: int) -> Sequence[tuple]:
        """Pole labels with level index <= ``order``, in ascending level order."""

    def build_pole(self, label, b, h_ext) -> dict:
        """``dict(rs=, d=, v=, K=, sector=, ...)`` for one label."""

    def seed(self, sector: int, order: int, field) -> list:
        """Coefficients of ``g^sector`` in ``t``, length ``order + 1``."""

    def couples(self, outer: dict, inner: dict) -> bool:
        """Whether ``inner`` contributes to ``outer``'s recursion."""


@dataclass
class LevelTable:
    """Poles plus the filled inner-block table for one (family, c, externals).

    ``table[i][k]`` is the ``t^k`` coefficient of ``H^(sector_i)(v_i, t)``,
    filled for ``k <= order - rs_i``. It is independent of the exchanged weight,
    so one table serves every evaluation point -- which is what makes sampling
    an interpolation grid cheap.
    """

    poles: list
    table: list
    order: int
    family: PoleFamily
    weights: dict | None = None
    # flint evaluation matrices (level_series_acb), keyed by all top-level inputs
    flint_cache: dict = _field(default_factory=dict, repr=False, compare=False)
    # the table as acb, when fill_level_table_flint built it (level_series_acb reads it without converting)
    flint_table: list | None = _field(default=None, repr=False, compare=False)
    # the poles' (K, d) as the certified fill's balls (their radii kept in the evaluation)
    flint_kd: tuple | None = _field(default=None, repr=False, compare=False)

    @property
    def level_unit(self) -> Fraction:
        return self.family.level_unit

    def __len__(self) -> int:
        return len(self.poles)


def fixed_pole_weights(eval_arg, poles, zero):
    """``K_j / (eval_arg - d_j)`` for every pole, or ``None`` on a collision.

    The recursion revisits the same denominator against many different inner
    values, so the quotient is formed once per evaluation point. ``None`` marks
    ``eval_arg == d_j`` -- the caller decides what a pole collision means
    (Virasoro's regularization path substitutes a limit there); a zero ``K_j``
    short-circuits to exact ``zero`` so that structurally-absent poles cost
    nothing and cannot introduce rounding.
    """
    weights = [zero] * len(poles)
    for idx, pole in enumerate(poles):
        if pole["K"] == 0:
            continue
        if eval_arg == pole["d"]:
            weights[idx] = None
            continue
        weights[idx] = pole["K"] / (eval_arg - pole["d"])
    return weights


def interaction_weights(poles, outer_indices, field, *, couples=None):
    """``{outer: [K_j / (v_outer - d_j) per inner j]}``.

    ``couples(outer_pole, inner_pole) -> bool`` suppresses forbidden (outer,
    inner) combinations to exact zero; the fill drops zeros entirely, so a
    sector restriction costs nothing at fill time. ``None`` entries are the
    ``v_outer == d_inner`` collision sentinel, preserved for the caller.
    """
    zero = field(0)
    out = {}
    for idx in outer_indices:
        outer = poles[idx]
        row = fixed_pole_weights(outer["v"], poles, zero)
        if couples is not None:
            for j, inner in enumerate(poles):
                if not couples(outer, inner):
                    row[j] = zero
        out[idx] = row
    return out


def _active_inner(poles, weights, outer_indices):
    """Per outer, the inner poles that can contribute, in ascending level.

    Dropping zero-weight entries once beats testing ``weight == 0`` inside the
    (k, outer, inner) triple loop, and leaves both the sum and its accumulation
    order unchanged -- so the result is bit-identical. ``None`` weights (the
    collision sentinel) are kept, since ``None != 0``.
    """
    return {
        idx: [
            (j, poles[j]["rs"], weights[idx][j])
            for j in range(len(poles))
            if weights[idx][j] != 0
        ]
        for idx in outer_indices
    }


def fill_level_table(poles, weights, order, seeds, *, outer_indices=None, lookup=None):
    """Bottom-up fill of ``table[i][k]``, the inner blocks at each ``v_i``.

    Filled in increasing ``k`` so that every inner value at ``k - rs_j`` is
    already known: the recursion is strictly triangular in the level index
    because every pole has ``rs_j >= 1``.

    ``seeds`` maps sector -> coefficient list for ``g^sector``; pass
    ``{0: [one, zero, zero, ...]}`` for a plain ``H = 1 + ...`` recursion.

    ``lookup(table, j, k)`` reads one inner value, defaulting to
    ``table[j][k]``. It exists so a caller that stores only a canonical subset
    of rows can reconstruct the rest on read -- Virasoro's strip-symmetric mode
    keeps only ``r <= s`` and conjugates. ``outer_indices`` then restricts which
    rows are filled at all.
    """
    if outer_indices is None:
        outer_indices = range(len(poles))
    outer_indices = list(outer_indices)
    if lookup is None:
        def lookup(table, j, k):
            return table[j][k]

    table: list = [None] * len(poles)
    for idx in outer_indices:
        seed = seeds[poles[idx].get("sector", 0)]
        table[idx] = [seed[k] for k in range(order - poles[idx]["rs"] + 1)]

    active = _active_inner(poles, weights, outer_indices)

    for k in range(1, order + 1):
        for idx in outer_indices:
            outer = poles[idx]
            if k > order - outer["rs"]:
                continue
            acc = table[idx][k]
            for j, rs_inner, weight in active[idx]:
                if rs_inner > k:
                    break
                inner = lookup(table, j, k - rs_inner)
                if inner == 0:
                    continue
                if weight is None:
                    p_inner = poles[j]
                    acc += p_inner["K"] / (outer["v"] - p_inner["d"]) * inner
                    continue
                acc += weight * inner
            table[idx][k] = acc
    return table


def eval_level_series(
    eval_arg, poles, table, order, seed, field, *, couples=None, sector=0, lookup=None
):
    """``[H^sector_0(x), ..., H^sector_order(x)]`` at ``x = eval_arg``.

    The per-``k`` sum is over poles whose level fits under ``k``; the inner
    values come from the shared ``table``, so one evaluation costs
    O(order * npoles) multiplications and no recursion re-fill. ``lookup`` has
    the same meaning as in :func:`fill_level_table`.
    """
    if lookup is None:
        def lookup(table, j, k):
            return table[j][k]

    top = fixed_pole_weights(eval_arg, poles, field(0))
    probe = {"sector": sector, "v": eval_arg}

    out = [seed[k] for k in range(order + 1)]
    active = [
        (j, poles[j]["rs"], top[j])
        for j in range(len(poles))
        if top[j] != 0 and (couples is None or couples(probe, poles[j]))
    ]
    for k in range(1, order + 1):
        acc = out[k]
        for j, rs_inner, weight in active:
            if rs_inner > k:
                break
            inner = lookup(table, j, k - rs_inner)
            if inner == 0:
                continue
            if weight is None:
                acc += poles[j]["K"] / (eval_arg - poles[j]["d"]) * inner
                continue
            acc += weight * inner
        out[k] = acc
    return out


def _evaluation_rows(level_table, eval_arg, order, couples, sector):
    """The active top-level rows, including a custom rule's dependence on the query."""
    probe = {"sector": sector, "v": eval_arg}
    kd = level_table.flint_kd
    return tuple(
        j for j, pole in enumerate(level_table.poles)
        if (not kd[0][j].is_zero() if kd is not None else pole["K"] != 0)
        and pole["rs"] <= order and (couples is None or couples(probe, pole))
    )


def _evaluation_data(level_table, eval_arg, order, seed, couples, sector, active=None):
    """``(d, K, d_balls, M, seed)`` of :func:`level_series_acb` (the active rows' exact locations, their ``K`` and
    ``d`` balls, the ``(order + 1) x rows`` matrix ``M[k, j] = table[j][k - rs_j]`` and the seed as acb), built at the
    current flint precision."""
    from flint import acb, acb_mat

    from .polynomial import _mpf_to_acb

    kd = level_table.flint_kd
    if active is None:
        active = _evaluation_rows(level_table, eval_arg, order, couples, sector)
    rows = [[acb(0)] * len(active) for _ in range(order + 1)]
    source = level_table.flint_table
    for col, j in enumerate(active):
        rs = level_table.poles[j]["rs"]
        inner = source[j] if source is not None else [_mpf_to_acb(v) for v in level_table.table[j]]
        for k in range(rs, order + 1):
            if k - rs < len(inner):
                rows[k][col] = inner[k - rs]
    return (
        [level_table.poles[j]["d"] for j in active],
        [kd[0][j] if kd is not None else _mpf_to_acb(level_table.poles[j]["K"]) for j in active],
        [kd[1][j] if kd is not None else _mpf_to_acb(level_table.poles[j]["d"]) for j in active],
        acb_mat(rows) if active else None,
        [_mpf_to_acb(value) for value in seed[: order + 1]],
    )


def level_series_acb(level_table, eval_arg, order, seed, *, couples=None, sector=0):
    """:func:`eval_level_series` in flint: the ``order + 1`` coefficients as acb, or None.

    The series is linear in the top-level pole weights,

        H_k(x) = seed_k + sum_j M[k, j] K_j / (x - d_j),     M[k, j] = table[j][k - rs_j],

    and the inner rows do not depend on ``x``. The cache also distinguishes the seed and routing rule. The
    family's routing is fixed for a constructed level table; custom predicates are evaluated at each query and
    cache by their selected rows, so a query-dependent rule agrees with :func:`eval_level_series`.
    Each evaluation is one weight vector and one matrix-vector product (the Virasoro evaluator's scheme), instead
    of the O(order * npoles) mpmath loop. Returns None when ``x`` sits exactly on a pole (the mpmath form handles
    that collision sentinel).
    """
    from flint import acb, acb_mat, ctx as flint_ctx

    from .polynomial import _flint_prec_bits, _mpf_to_acb

    prec = _flint_prec_bits()
    active = None
    routing_key = couples
    if couples is not None and couples != getattr(level_table.family, "couples", None):
        active = _evaluation_rows(level_table, eval_arg, order, couples, sector)
        routing_key = active
    # Raw native tuples avoid repeated mpmath equality/conversion in this hot
    # path and retain every supplied seed bit, independently of ambient dps.
    seed_key = tuple(value._mpf_ if isinstance(value, mp.mpf) else value._mpc_
                     if isinstance(value, mp.mpc) else value for value in seed[: order + 1])
    key = (prec, int(order), sector, routing_key, seed_key)
    try:
        cached = level_table.flint_cache.get(key)
    except TypeError:
        # An unhashable seed or family predicate remains a valid input.
        # Evaluate it without caching rather than dropping its arguments.
        key, cached = None, None
    old_prec = flint_ctx.prec
    flint_ctx.prec = prec
    try:
        if cached is None:
            cached = _evaluation_data(level_table, eval_arg, order, seed, couples, sector, active)
            if key is not None:
                level_table.flint_cache[key] = cached
        d_exact, K, d, M, seed_acb = cached
        if any(eval_arg == value for value in d_exact):
            return None
        if M is None:
            return list(seed_acb)
        x = _mpf_to_acb(eval_arg)
        weights = [k / (x - dj) for k, dj in zip(K, d)]
        if not all(w.is_finite() for w in weights):
            return None         # x within a pole's ball: the mpmath form's collision handling
        column = M * acb_mat([[w] for w in weights])
        return [seed_acb[k] + column[k, 0] for k in range(order + 1)]
    finally:
        flint_ctx.prec = old_prec


def certified_level_series(table_at, eval_arg, order, seed, *, couples=None, sector=0, what="the series",
                           input_bits=0):
    """:func:`level_series_acb` certified to the working precision, or None (``eval_arg`` on a pole).

    The table is certified (:func:`fill_level_table_flint`), but the top-level sum ``sum_j K_j / (x - d_j) T_j``
    still cancels next to a singular c (NS at c = 3/2 + 1e-20, 30 digits: no correct digit; Ramond at 3/2 + 1e-12:
    15), so its own balls are read and the evaluation repeated at the working precision plus the measured deficit,
    as the Virasoro coefficients (``recursion._H_coefficient_balls``): each coefficient against the series' scale
    (``recursion._power_series_deficit``, ``input_bits`` the resolution of the exact inputs,
    ``recursion._input_bits``), so one that cancels to zero is certified, and comes back centered on 0
    (``recursion._zero_centred``).  ``table_at()`` returns the level table at the current precision."""
    from .recursion import _power_series_deficit, _zero_centred

    values = _certified_ladder(
        lambda: level_series_acb(table_at(), eval_arg, order, seed, couples=couples, sector=sector),
        lambda values, work: _power_series_deficit(values, work, input_bits)[0], order, what)
    return None if values is None else _zero_centred(values)


def _laurent_deficit(rows, work: int) -> int:
    """Bits by which Laurent rows (:func:`laurent_rows_acb`'s ``(principal, regular)``) fall short of ``work``: each row
    is a coefficient vector that later enters linear combinations, so every entry's radius is measured against the
    row's largest entry (an entry that cancels to zero is certified by a radius small against its row, which a
    relative measure cannot see)."""
    import math

    worst = 0
    for part in rows:
        for row in part:
            if not row:
                continue
            top = max(v.abs_upper() for v in row)
            rad = max(max(v.real.rad(), v.imag.rad()) for v in row)
            if top.is_zero() or rad.is_zero():
                continue
            if not (rad.is_finite() and top.is_finite()):
                return 2 * work
            accuracy = math.floor(float((top / rad).log()) / math.log(2))
            worst = max(worst, work - accuracy)
    return worst


def _certified_ladder(compute, deficit_of, order, what):
    """``compute()`` (flint balls, or None) repeated at the working precision plus the deficit its own balls measure
    (``deficit_of(values, work)``) until every value is certified to the working precision."""
    work = mp.mp.prec
    extra, previous = 0, None
    while True:
        with mp.workprec(work + extra):
            values = compute()
        if values is None:
            return None
        deficit = deficit_of(values, work)
        if deficit == 0:
            return values
        step = next_certification_prec(work + extra, deficit, previous, work)
        if step is None:
            raise LevelTablePrecisionError(
                f"{what} cannot be certified to {mp.mp.dps} digits: a coefficient stays {deficit} bits short with "
                f"{extra} guard bits (order {order})")
        extra, previous = step[0] - work, step[1]


def laurent_rows_acb(M, K, d, powers, at_loc, loc, seed, principal_order, regular_order):
    """Laurent data at ``x = loc`` of ``H_k(x) = seed_k + sum_j M[k, j] K_j / (x - d_j)^(a_j)`` (flint).

    ``at_loc`` are the columns located at ``loc`` (each contributes ``M[:, j] K_j`` to the coefficient of
    ``(x - loc)^-a_j``); every other column is expanded about ``loc``,
    ``(x - d)^-a = sum_q binom(-a, q) (loc - d)^(-a-q) (x - loc)^q``.  Returns ``(principal, regular)``:
    ``principal[j][k]`` the coefficient of ``(x - loc)^-(j+1)`` in ``H_k`` (``j < principal_order``) and
    ``regular[q][k]`` that of ``(x - loc)^q`` (``q <= regular_order``).  ``powers`` None means every column is a
    simple pole.  One matrix product, as :func:`level_series_acb`.
    """
    from flint import acb, acb_mat

    n_pr, n_rg = int(principal_order), int(regular_order) + 1
    n_rows = len(seed)
    ncols = len(K)
    zero = acb(0)
    cols = [[zero] * ncols for _ in range(n_pr + n_rg)]
    at_loc = set(at_loc)
    for j in range(ncols):
        a = 1 if powers is None else int(powers[j])
        if j in at_loc:
            if a > n_pr:
                raise ValueError(f"a pole of order {a} at {mp.nstr(loc, 8)} needs principal_order >= {a}")
            cols[a - 1][j] = K[j]
            continue
        inv = 1 / (loc - d[j])
        term = K[j] * inv ** a
        for q in range(n_rg):
            cols[n_pr + q][j] = term
            term = term * inv * (-(a + q)) / (q + 1)
    if M is None or ncols == 0:
        HQ = None
    else:
        HQ = M * acb_mat(cols).transpose()          # (n_rows x (n_pr + n_rg))
    principal = [[zero if HQ is None else HQ[k, j] for k in range(n_rows)] for j in range(n_pr)]
    regular = [[(zero if HQ is None else HQ[k, n_pr + q]) + (seed[k] if q == 0 else zero) for k in range(n_rows)]
               for q in range(n_rg)]
    return principal, regular


def location_tolerance(loc):
    """How close a pole must be to ``loc`` to be the one at ``loc``: ``10^-(dps/3)`` (relative past 1), far below the
    Kac spacing and loose enough for a location that went through ``2 d - l`` and back.  Taken at the precision
    ``loc`` was given at (a certification ladder raises the precision, not the input's accuracy)."""
    return mp.mpf(10) ** (-(mp.mp.dps // 3)) * max(1, abs(mp.mpc(loc)))


def level_series_laurent_acb(level_table, loc, order, seed, *, couples=None, sector=0, principal_order=1,
                             regular_order=0, tol=None, allow_regular_point=False):
    """:func:`laurent_rows_acb` of the level series (the rows :func:`level_series_acb` evaluates) at ``x = loc``, or
    None when no active row has its pole there (to ``tol``, default :func:`location_tolerance`) -- unless
    ``allow_regular_point``, which returns the Taylor rows there (zero principal rows)."""
    from flint import ctx as flint_ctx

    from .polynomial import _flint_prec_bits, _mpf_to_acb

    old_prec = flint_ctx.prec
    flint_ctx.prec = _flint_prec_bits()
    try:
        d_exact, K, d, M, seed_acb = _evaluation_data(level_table, loc, order, seed, couples, sector)
        loc_c = mp.mpc(loc)
        tol = location_tolerance(loc_c) if tol is None else tol
        at_loc = [j for j, dj in enumerate(d_exact) if abs(mp.mpc(dj) - loc_c) <= tol]
        if not at_loc and not allow_regular_point:
            return None
        return laurent_rows_acb(M, K, d, None, at_loc, _mpf_to_acb(loc), seed_acb, principal_order, regular_order)
    finally:
        flint_ctx.prec = old_prec


def certified_level_laurent(table_at, loc, order, seed, *, couples=None, sector=0, principal_order=1,
                            regular_order=0, allow_regular_point=False, what="the Laurent data"):
    """:func:`level_series_laurent_acb` certified to the working precision (the regular part's top-level sum cancels
    as :func:`level_series_acb`'s does), or None (``loc`` is not a pole of the series)."""
    tol = location_tolerance(loc)
    return _certified_ladder(
        lambda: level_series_laurent_acb(table_at(), loc, order, seed, couples=couples, sector=sector,
                                         principal_order=principal_order, regular_order=regular_order, tol=tol,
                                         allow_regular_point=allow_regular_point),
        _laurent_deficit, order, what)


def next_certification_prec(prec: int, deficit: int, previous, work: int):
    """The next precision of a certification ladder (``prec`` bits fell ``deficit`` short of ``work``), as
    ``(prec, previous)`` for the next call, or ``None`` to give up.

    While a quantity keeps some certified bits, the next pass adds the measured deficit (plus 32), and a deficit that
    no longer shrinks means more bits do not help.  A ball with no certified bit at all (``deficit >= work``: next to
    a singular c the probes' colliding poles make the entries 1/eps large and cancel to O(1)) says nothing about how
    many are missing, and more bits can even widen it (its midpoint is noise), so the precision doubles instead -- up
    to ``8 work + 4096`` bits.
    """
    if deficit >= work:
        nxt, prev = 2 * prec + 32, None
    else:
        if previous is not None and deficit >= previous:
            return None
        nxt, prev = prec + deficit + 32, deficit
    if nxt > 8 * work + 4096:
        return None
    return nxt, prev


class LevelTablePrecisionError(ValueError):
    """A level table could not be certified to the working precision: more guard bits did not shrink the loss."""


def _kac_relation_checker(parameter, *, c=None, superconformal=False):
    """Prove a*t+b/t+constant=0 from t's source, with t=b² or beta²."""
    from .numeric_input import literal_components, _zero_on_quadratic_root

    if c is not None:
        cr, ci = literal_components(c)
        total = ((Fraction(5, 2)-cr/3, -ci/3) if superconformal else ((cr-13)/6, ci/6))
        hint = parameter**2
    else:
        real, imaginary = literal_components(parameter)
        tr, ti = real*real-imaginary*imaginary, 2*real*imaginary
        norm = tr*tr+ti*ti
        inverse = tr/norm, -ti/norm

    def relation(a, b, constant):
        if c is not None:
            return _zero_on_quadratic_root((a-b, 0), (b*total[0]+constant[0], b*total[1]+constant[1]),
                                            total, hint)
        return a*tr+b*inverse[0]+constant[0] == 0 and a*ti+b*inverse[1]+constant[1] == 0

    return relation


def _kac_collision_checker(parameter, *, c=None, superconformal=False):
    """Prove shifted Kac collisions using source labels, rather than rounded poles."""
    relation = _kac_relation_checker(parameter, c=c, superconformal=superconformal)
    sign = 1 if superconformal else -1

    def collision(outer, inner):
        return relation(sign*(outer['r']**2-inner['r']**2), sign*(outer['s']**2-inner['s']**2),
                        (2*(outer['rs']+inner['rs']), 0))

    return collision


def _kac_weight_checker(parameter, *, c=None, superconformal=False):
    from .numeric_input import literal_components

    relation = _kac_relation_checker(parameter, c=c, superconformal=superconformal)
    sign, unit = (1, 8) if superconformal else (-1, 4)

    def matches(weight, r, s):
        hr, hi = literal_components(weight)
        return relation(sign*(r*r-1), sign*(s*s-1), (-2*(r*s-1)-unit*hr, -unit*hi))

    return matches


def fill_level_table_flint(poles, order, seeds, field, *, couples=None, rebuild=None, level_unit=1,
                           collision_check=None, input_bits=0):
    """:func:`interaction_weights` + :func:`fill_level_table` in flint, certified to the working precision.

    Returns ``(table, flint_table, flint_kd)`` (the mpmath table of ``field``, the same values as acb balls with
    their radii, and the ``(K, d)`` balls of every pole), or ``("collision", (i, j))`` when ``v_i == d_j`` exactly for
    a coupled pair (the caller names the degenerate c). Each level is one matrix-vector product
    ``T[:, k] = seed_k + W T_k``, ``W[i, j] = K_j / (v_i - d_j)``. Rebuilt data use ``collision_check`` to prove
    equality from source labels; a ball containing zero alone asks for more precision.

    The recursion cancels (about a digit per level at heavy externals or near a singular c), so the fill runs in
    ball arithmetic with guard bits, widened by the measured deficit until every entry is certified to the working
    precision (``mp.mp.prec`` bits against each row's scale, ``recursion._power_series_deficit``; ``input_bits`` the
    resolution of the exact inputs, ``recursion._input_bits``), as the Virasoro H table
    (``recursion._certified_h_table``).
    ``rebuild(prec_bits)`` returns the same pole list with ``K`` and ``d`` computed as acb enclosures from the
    source inputs at ``prec_bits``. Their radii propagate through the fill; a rounding allowance added after
    mpmath evaluation cannot bound cancellation in those products. ``v`` is ``d`` plus exact ``level_unit * rs``.
    Without ``rebuild`` the given numerical pole data are treated as exact supplied values.
    """
    from flint import acb, acb_mat, arb, ctx as flint_ctx
    from .derivs import _acb_to_scalar
    from .polynomial import _arb_to_mpf, _flint_prec_bits, _mpf_to_acb
    from .recursion import _power_series_deficit

    old_prec = flint_ctx.prec
    flint_ctx.prec = _flint_prec_bits()
    try:
        n = len(poles)
        live = [j for j, p in enumerate(poles) if p["K"] != 0]
        # the exact collision test on the working-precision data
        d_exact = [_mpf_to_acb(p["d"]) for p in poles]
        for i, outer in enumerate(poles) if rebuild is None else ():
            v = _mpf_to_acb(outer["v"])
            for j in live:
                if couples is not None and not couples(outer, poles[j]):
                    continue
                if (v - d_exact[j]).is_zero():
                    return "collision", (i, j)
    finally:
        flint_ctx.prec = old_prec

    rs = [p["rs"] for p in poles]
    unit = Fraction(level_unit)
    shifts = [Fraction(r) * unit for r in rs]
    work = mp.mp.prec
    prec = _flint_prec_bits() + 2 * int(order) + 16
    previous = None
    try:
        while True:
            flint_ctx.prec = prec
            if rebuild is not None:
                source = rebuild(prec)
                if not all(isinstance(p[name], acb) for p in source for name in ('K', 'd')):
                    raise TypeError('rebuild must return Kac data as acb enclosures')
            else:
                source = poles
            K = [p['K'] if isinstance(p['K'], acb) else _mpf_to_acb(p['K']) for p in source]
            d = [p['d'] if isinstance(p['d'], acb) else _mpf_to_acb(p['d']) for p in source]
            if not all(value.is_finite() for value in (*K, *d)):
                step = next_certification_prec(prec, work, None, work)
                if step is None:
                    raise LevelTablePrecisionError(f"Kac data do not resolve at {prec} bits (order {order})")
                prec = step[0]
                continue
            live = [j for j, value in enumerate(K) if not value.is_zero()]
            zero = acb(0)
            rows = []
            singular = False
            for i, outer in enumerate(source):
                v = d[i] + arb(shifts[i].numerator) / shifts[i].denominator
                row = [zero] * n
                for j in live:
                    if couples is not None and not couples(outer, source[j]):
                        continue
                    gap = v - d[j]
                    if gap.contains(0):
                        if collision_check is not None and collision_check(outer, source[j]):
                            return 'collision', (i, j)
                        singular = True
                        break
                    row[j] = K[j] / gap
                rows.append(row)
            if singular:
                step = next_certification_prec(prec, work, None, work)
                if step is None:
                    raise LevelTablePrecisionError(
                        f"level table: a Kac-pole near-collision does not resolve at {prec} bits (order {order})"
                    )
                prec = step[0]
                continue
            W = acb_mat(rows)
            seed_acb = {key: [_mpf_to_acb(value) for value in values] for key, values in seeds.items()}
            table = [list(seed_acb[p.get("sector", 0)][: order - p["rs"] + 1]) for p in source]
            for k in range(1, order + 1):
                column = acb_mat([[table[j][k - rs[j]]] if rs[j] <= k else [zero] for j in range(n)])
                product = W * column
                for i in range(n):
                    if k <= order - rs[i]:
                        table[i][k] = table[i][k] + product[i, 0]
            # each row is a level series, held to its own scale (an entry that cancels to zero has no relative bit)
            deficit = max((_power_series_deficit(row, work, input_bits)[0] for row in table), default=0)
            if deficit == 0:
                break
            step = next_certification_prec(prec, deficit, previous, work)
            if step is None:
                raise LevelTablePrecisionError(
                    f"level table cannot be certified to {mp.mp.dps} digits: an entry stays {deficit} bits short at "
                    f"{prec} bits (order {order})"
                )
            prec, previous = step
        flint_ctx.prec = _flint_prec_bits()
        flint_table = [[+value for value in row] for row in table]
        flint_kd = ([+k for k in K], [+x for x in d])
        if field is mp.mpf:
            mp_table = [[+_arb_to_mpf(value.real) for value in row] for row in table]
        else:
            mp_table = [[+mp.mpc(_acb_to_scalar(value)) for value in row] for row in table]
        return mp_table, flint_table, flint_kd
    finally:
        flint_ctx.prec = old_prec
