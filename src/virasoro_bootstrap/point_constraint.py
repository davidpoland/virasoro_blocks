"""Point constraints: positivity (or a fixed value) at an isolated (Delta, ell).

A point constraint is the **degree-0** case of a constraint block. Imposing
``y . F(Delta_point) >= 0`` is a single linear inequality on the dual functional;
in the PMP it is a channel whose numerator is the constant vector
``F(Delta_point)`` (degree 0), so SDPB enforcing ``prefactor(x) * [const] >= 0``
over ``x >= 0`` reduces -- the prefactor being positive -- to exactly that
inequality. This is how one imposes positivity at an isolated dimension below
the gap (e.g. a known operator, or an allowed-island point) alongside the
continuous ``Delta > Delta_gap`` positivity.

Requires the degree-0 guard in `pmp_writer._compute_bilinear_bases` (a degree-0
SOS has an empty `bilinearBasis_1`).
"""
from __future__ import annotations

from dataclasses import replace
from functools import wraps

import mpmath as mp

from virasoro_blocks.point import point_block_table

from .identical_scalar import (
    IdenticalScalarSpinChannel, _as_real_label, _known_vector_sum,
    _validate_identical_scalar_operator,
)
from .pmp_writer import build_prefactored_pmp_doc, _normalize_F_pairs, _validate_F_pair


def _at_problem_precision(function):
    """Rebuild a problem at its recorded precision, including the PMP arithmetic."""
    @wraps(function)
    def wrapped(problem, *args, **kwargs):
        with mp.workdps(int(problem.spec.dps)):
            return function(problem, *args, **kwargs)
    return wrapped


def build_point_channel(ell, delta_point, *, basis="scalar_G", coordinate="zzb",
                        h_ext, c=None, q_order=None, lam, chiral_norm=None,
                        z_shift=None, zbar_shift=None):
    """A degree-0 positivity channel imposing ``y . F(delta_point, ell) >= 0``
    for identical-scalar externals.

    The block is computed *directly at* ``delta_point`` with the robust numeric
    evaluator (:func:`virasoro_blocks.point.point_block_table`) rather than by
    evaluating an order-reduced interpolation off its accurate support -- point
    constraints typically sit below the gap, exactly where the gap-domain
    interpolation is least trustworthy. ``basis`` selects the block family
    (``"scalar_G"`` for the Virasoro path, ``"global_scalar_G"`` for the global
    control); the frozen constant block matches the fixed-spin channels'
    convention, so it feeds the PMP directly as the channel's ``prefactored``
    block. The crossing shifts default to ``-2*h_ext`` on both axes. For
    ``chiral_F`` or ``global_chiral_F``, the default ``coordinate="zzb"``
    evaluates the chiral ``z`` derivatives and stores them under the bootstrap
    coordinate name, as the chiral continuum builders do; ``ell`` must be zero.
    """
    ell, dp = _validate_identical_scalar_operator(ell, delta_point, name="point constraint")
    chiral = basis in {"chiral_F", "global_chiral_F"}
    if chiral and ell != 0:
        raise ValueError("chiral point constraints require ell=0")
    point_coordinate = "z" if chiral and coordinate == "zzb" else coordinate
    h_ext = _as_real_label(h_ext, name="point external weight")
    c = None if c is None else _as_real_label(c, name="point central charge")
    z_shift = -2*h_ext if z_shift is None else _as_real_label(z_shift, name="point z_shift")
    zbar_shift = -2*h_ext if zbar_shift is None else _as_real_label(zbar_shift, name="point zbar_shift")
    from virasoro_blocks.recursion import _resolve_chiral_norm
    from virasoro_blocks.requests import _integer_field

    lam = _integer_field("lam", lam)
    if q_order is not None:
        q_order = _integer_field("q_order", q_order)

    serialized = point_block_table(
        [point_coordinate], basis=basis, ell=ell, point_value=dp,
        h_externals=[h_ext] * 4, hbar_externals=[h_ext] * 4, lam=lam, c=c,
        q_order=q_order, chiral_norm=chiral_norm, z_shift=z_shift,
        zbar_shift=zbar_shift,
    )
    if point_coordinate != coordinate:
        from .chiral_from_blocks import _relabel_chiral_as_zzb

        serialized = _relabel_chiral_as_zzb(serialized)
    return IdenticalScalarSpinChannel(
        ell=ell, delta_min=dp, serialized=serialized,
        prefactored=serialized, point_context={
            "basis": basis, "coordinate": coordinate, "c": c, "h_ext": h_ext,
            "q_order": q_order, "lam": lam, "chiral_norm": _resolve_chiral_norm(chiral_norm),
            "z_shift": z_shift, "zbar_shift": zbar_shift,
        })


def _crossing_context_from_inputs(*, basis, coordinate, h_ext, lam, chiral_norm=None,
                                  c=None, q_order=None, z_shift=None, zbar_shift=None):
    from virasoro_blocks.recursion import _resolve_chiral_norm

    h_ext = _as_real_label(h_ext, name="h_ext")
    context = {
        "basis": basis, "coordinate": coordinate, "h_ext": h_ext, "lam": lam,
        "chiral_norm": _resolve_chiral_norm(chiral_norm),
        "z_shift": -2*h_ext if z_shift is None else _as_real_label(z_shift, name="z_shift"),
        "zbar_shift": -2*h_ext if zbar_shift is None else _as_real_label(zbar_shift, name="zbar_shift"),
    }
    if not basis.startswith("global_"):
        context.update(c=_as_real_label(c, name="c"), q_order=q_order)
    return context


def _crossing_context_for_spec(spec, resolved_method):
    from .manifest import _CHIRAL_METHODS

    chiral = resolved_method in _CHIRAL_METHODS
    global_basis = resolved_method in {"global_control", "chiral_global_control"}
    basis = "chiral_F" if chiral else "scalar_G"
    if global_basis:
        basis = "global_" + basis
    return _crossing_context_from_inputs(
        basis=basis, coordinate=spec.coordinate, h_ext=spec.h_ext, lam=spec.lam,
        chiral_norm=spec.chiral_norm, c=spec.c, q_order=spec.q_order)


def _check_channel_context(problem, channel, *, for_fixed_operator=False):
    context = channel.point_context if channel.point_context is not None else channel.crossing_context
    if context is None:
        return
    expected = _crossing_context_for_spec(problem.spec, problem.resolved_method)
    different = [name for name, value in expected.items() if context.get(name) != value
                 and not (for_fixed_operator and name == "lam" and context.get(name, -1) >= value)]
    if different:
        raise ValueError("reference channel does not match the problem's crossing context: " + ", ".join(different))


def point_constraint_channel(reference_channel, delta_point, *, basis="scalar_G",
                             coordinate="zzb", h_ext, c=None, q_order=None, lam,
                             chiral_norm=None, z_shift=None, zbar_shift=None):
    """A degree-0 positivity channel at ``delta_point`` in ``reference_channel``'s
    spin sector. Thin wrapper over :func:`build_point_channel` (kept for callers
    that already hold a reference channel for its spin); the block value is
    recomputed numerically at the point, not read off the reference block."""
    return build_point_channel(
        reference_channel.ell, delta_point, basis=basis, coordinate=coordinate,
        h_ext=h_ext, c=c, q_order=q_order, lam=lam, chiral_norm=chiral_norm,
        z_shift=z_shift, zbar_shift=zbar_shift)


def normalization_with_fixed_operators(identity_vector, F_pairs, fixed_operators,
                                        *, coordinate="zzb"):
    """Fold operators with *specified* OPE coefficients into the normalization
    vector. A known operator (Delta*, ell*, c*) sits on the known side of
    crossing alongside the identity, so it does not add a constraint row: the
    single normalization equality just acts on
    ``N = F_id + sum_i c_i F(Delta_i, ell_i)``.

    `fixed_operators` is a list of ``(reference_channel, delta_point,
    coefficient)``; F is read from the reference channel's block (same spin
    sector / convention) at the point. The coefficient is the operator's
    ``p = N^Delta lambda^2`` against the identity's 1, in every spin channel
    (see :mod:`virasoro_blocks.composer`).
    """
    N = list(identity_vector)
    F_pairs = list(F_pairs)
    if len(N) != len(F_pairs):
        raise ValueError("identity vector and F-pairs must have the same length")
    normalized_pairs = _normalize_F_pairs(F_pairs)
    terms = [(1, N)]
    for ref_channel, delta_point, coeff in fixed_operators:
        delta_point = _as_real_label(delta_point, name="fixed operator Delta")
        c = _as_real_label(coeff, name="fixed operator coefficient")
        if getattr(ref_channel, "twist", None) is not None:
            raise ValueError("fixed operators require a fixed-spin reference channel, not a large-spin tower")
        if hasattr(ref_channel, "ell"):
            _validate_identical_scalar_operator(ref_channel.ell, delta_point, name="fixed operator")
        if getattr(ref_channel, "point_context", None) is not None:
            if any(_as_real_label(table.shift, name="point block shift") != delta_point
                   for table in (ref_channel.serialized, ref_channel.prefactored)):
                raise ValueError("fixed operator Delta must match the reference point channel's frozen dimension")
        table = ref_channel.prefactored.evaluate(coordinate, delta_point)
        for m, n in normalized_pairs:
            _validate_F_pair(table, m, n, where="fixed operator")
        terms.append((c, [table[m][n] for m, n in normalized_pairs]))
    # the block tables are mpc with a rounding-level imaginary part; the
    # normalization is a real vector (as the identity vector is)
    return _known_vector_sum(terms)


@_at_problem_precision
def input_fixed_operators(problem, fixed_operators, *, coordinate=None):
    """Return a copy of `problem` that *inputs* operators with specified OPE
    coefficients by grouping them with the identity and rebuilding the PMP
    -- still one equality constraint, on a more
    complicated object. (Contrast `point_constraint_channel`, which imposes
    *positivity* at a point, i.e. an operator is merely allowed there.)

    ``coordinate`` defaults to the problem's own crossing coordinate.  The
    returned problem records the folded vector ``N`` as its identity. For gap
    and fixed-OPE feasibility it is the normalization; for OPE upper/lower/max
    bounds it is the objective, with the distinguished-operator normalization
    retained. Moment problems require their dedicated builders, since their
    additional moment components also depend on the known operators.

    References must use a fixed-spin dimension variable. A reference already
    recorded as a point channel can only supply its frozen dimension.
    Library-built references retain their crossing inputs, which must match
    this problem; their gap edge and numerical precision may differ. A
    reference with more derivatives can supply the requested subset.
    """
    feasibility = problem.problem_kind in {"gap", "fixed_ope_feasibility"}
    ope_bound = problem.problem_kind in {"ope_upper", "ope_lower", "ope_max"}
    if not (feasibility or ope_bound):
        raise ValueError(f"input_fixed_operators does not support problem kind {problem.problem_kind!r}; "
                         "use the corresponding problem builder")
    coordinate = problem.spec.coordinate if coordinate is None else coordinate
    if coordinate != problem.spec.coordinate:
        raise ValueError("coordinate must match the problem's crossing coordinate; rebuild the problem to change it")
    fixed_operators = tuple(fixed_operators)
    for channel, delta_point, _coefficient in fixed_operators:
        if channel.twist is not None:
            raise ValueError("fixed operators require a fixed-spin reference channel, not a large-spin tower")
        _ell, dimension = _validate_identical_scalar_operator(
            channel.ell, delta_point, name="fixed operator")
        _check_channel_context(problem, channel, for_fixed_operator=True)
        if channel.point_context is not None or any(channel is point for point in problem.point_channels):
            if any(_as_real_label(table.shift, name="point block shift") != dimension
                   for table in (channel.serialized, channel.prefactored)):
                raise ValueError("fixed operator Delta must match the reference point channel's frozen dimension")
    N = normalization_with_fixed_operators(
        problem.identity_vector, problem.F_pairs, fixed_operators,
        coordinate=coordinate)
    objective = [mp.mpf(0)] * len(list(problem.F_pairs)) if feasibility else list(N)
    normalization = N if feasibility else problem.normalization_vector
    pmp_build = build_prefactored_pmp_doc(
        [ch.prefactored for ch in problem.spin_channels], list(problem.F_pairs),
        coordinate=coordinate,
        objective=objective, normalization=normalization,
        positive_pole_npoles=problem.spec.positive_pole_npoles,
        positive_pole_selection=problem.spec.positive_pole_selection,
        explicit_sample_points=problem.spec.explicit_sample_points,
    )
    return replace(
        problem, identity_vector=N, pmp_build=pmp_build,
        objective_vector=tuple(objective), normalization_vector=tuple(normalization),
    )


def _is_point_channel(channel) -> bool:
    """A degree-0 channel frozen at one dimension (what :func:`build_point_channel` returns)."""
    if getattr(channel, "twist", None) is not None:
        return False
    return all(
        table is not None and not tuple(table.pole_list_x)
        and all(poly.degree == 0 for rows in table.numerators.values() for row in rows for poly in row)
        for table in (channel.serialized, channel.prefactored)
    )


@_at_problem_precision
def append_positivity_channels(problem, channels):
    """Return a copy of ``problem`` with extra point-constraint channels appended
    and the PMP rebuilt.

    The problem keeps its channels ordered fixed-spin, large-spin tower, point,
    and the manifest writer and exact certification classify them by that
    layout (``point_channels`` mirrors the tail), so an appended channel must be
    a degree-0 point channel and is recorded in ``point_channels``.  Extra
    fixed spins or tower twists belong in the spec (``spin_list`` /
    ``large_spin_twists``), and are refused here rather than misclassified.
    The problem's own objective and normalization are kept, so this works for
    OPE problems as well as gap feasibility."""
    if channels is None:
        return problem
    channels = tuple(channels)
    if not channels:
        return problem
    bad = [ch for ch in channels if not _is_point_channel(ch)]
    if bad:
        raise ValueError(
            "append_positivity_channels appends point-constraint channels only; "
            f"got {len(bad)} non-point channel(s) (ell={[ch.ell for ch in bad]}). "
            "Add fixed spins or tower twists through the spec instead."
        )
    for channel in channels:
        _check_channel_context(problem, channel)
        _ell, dimension = _validate_identical_scalar_operator(channel.ell, channel.delta_min,
                                                             name="appended point constraint")
        if any(_as_real_label(table.shift, name="point block shift") != dimension
               for table in (channel.serialized, channel.prefactored)):
            raise ValueError("point channel Delta must match both frozen block shifts")
    all_channels = tuple(problem.spin_channels) + channels
    F_pairs = list(problem.F_pairs)
    objective = list(problem.objective_vector) or [mp.mpf(0)] * len(F_pairs)
    normalization = list(problem.normalization_vector) or list(problem.identity_vector)
    pmp_build = build_prefactored_pmp_doc(
        [ch.prefactored for ch in all_channels], F_pairs,
        coordinate=problem.spec.coordinate,
        objective=objective, normalization=normalization,
        positive_pole_npoles=problem.spec.positive_pole_npoles,
        positive_pole_selection=problem.spec.positive_pole_selection,
        explicit_sample_points=problem.spec.explicit_sample_points,
    )
    return replace(
        problem, spin_channels=all_channels, pmp_build=pmp_build,
        point_channels=tuple(problem.point_channels) + channels,
    )
