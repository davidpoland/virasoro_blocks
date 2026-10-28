"""Numerical table backend for the NS and Ramond sphere families.

The superconformal recurrences live in :mod:`superconformal_blocks`; this
module translates requests for the canonical table adapter. The CLI and
Python API share this table construction path.
"""

from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp

from .truncation import is_off_diagonal, kept_pairs, total_parity_phase
from .serialization import serialize_deriv_tables
from .table_backends import positivity_anchor
from .superconformal_blocks import (
    FAMILY_NS,
    FAMILY_RAMOND,
    SuperBlockPolicy,
    resolve_super_c,
    super_2d_interpolated_table,
    super_chiral_F_interpolated_table,
)


def super_table_offset_eps_exp(work: int) -> int:
    """Choose the c-offset exponent for a table built at precision ``work``.

    A shallow offset limits agreement with the c -> c0 limit; an excessively
    small offset makes the split Kac poles difficult to fit. The choice
    ``max(8, work // 2 + 5)`` balances these errors. It applies to the optional
    offset path; recognized singular charges use the merged Laurent recursion.
    Direct evaluation uses :func:`superconformal_blocks.super_offset_eps_exp`.
    """
    return max(8, int(work) // 2 + 5)


@dataclass(frozen=True)
class SuperBackendOptions:
    """Execution controls not represented by the canonical request."""

    eps_exp: int | None = None      # None: follows each build's working precision (super_table_offset_eps_exp)
    num_threads: int = 1
    shared_sample_points: bool = False

    def __post_init__(self):
        from .parallel import validate_worker_count

        object.__setattr__(self, "num_threads", validate_worker_count(self.num_threads))
        object.__setattr__(self, "eps_exp", SuperBlockPolicy(eps_exp=self.eps_exp).eps_exp)


def build_super_table(request, *, spin: int = 0, options=None):
    """Build one fixed-spin NS/Ramond table from a ``BlockRequest``.

    The returned object is the same ``SerializedDerivTable`` consumed by the
    JSON and Mathematica writers.  Central-charge regularization and the
    exchanged-pole policy remain owned by the superconformal backend, while
    the request supplies the family, sector, basis, coordinates, and numerical
    truncation parameters.
    """
    if request.family not in {FAMILY_NS, FAMILY_RAMOND}:
        raise ValueError(
            f"super table backend cannot handle family {request.family!r}"
        )
    options = SuperBackendOptions() if options is None else options
    if not isinstance(options, SuperBackendOptions):
        raise TypeError("options must be a SuperBackendOptions instance")
    if request.central_charge is None:
        raise ValueError("superconformal tables require central_charge")
    if request.policy.lambda_order is None:
        raise ValueError("superconformal tables require lambda_order")
    if request.policy.order is None:
        raise ValueError("superconformal tables require order")
    if request.policy.npoles == "all":
        raise ValueError(
            "superconformal tables use interpolation; npoles='all' is unsupported"
        )
    family = request.family
    basis = request.basis
    coordinates = tuple(request.coordinates or (request.coordinate,))
    # Preserve exact labels through singularity/fusion decisions and rebuilds.
    h_ext = list(request.h_ext or ())
    hbar_ext = (
        None
        if request.hbar_ext is None
        else list(request.hbar_ext)
    )
    if not h_ext:
        raise ValueError("superconformal tables require h_ext")
    is_chiral = basis.endswith("chiral_F")
    if not is_chiral and hbar_ext is None:
        hbar_ext = list(h_ext)

    q_order = int(request.policy.order)
    lam = int(request.policy.lambda_order)
    explicit_npoles = request.policy.npoles
    pole_selection = request.policy.pole_selection or "contribution_optimal"
    regularize = request.policy.c_regularization
    c_input, _c_eval, regularized = resolve_super_c(
        request.central_charge,
        regularize=regularize,
        eps_exp=int(options.eps_exp if options.eps_exp is not None
                    else super_table_offset_eps_exp(request.policy.precision or mp.mp.dps)),
    )
    sector = request.sector
    branch = request.branch
    parity_exchanged = request.parity_exchanged or "1"

    def build_once(npoles, options):
        return _build_super_tables(
            request, spin, options, family, basis, coordinates, h_ext, hbar_ext, is_chiral, q_order, lam,
            npoles, pole_selection, regularize, c_input, sector, branch, parity_exchanged,
        )

    from .super_laurent import exact_singular_c

    # at an exactly singular c the tables and their references run the exact Laurent recursion at c itself: there
    # is no offset to deepen or measure
    offset = regularized and exact_singular_c(c_input) is None
    return _verified_super_build(
        build_once, request, spin, options, family, basis, h_ext, hbar_ext, is_chiral, q_order, lam,
        explicit_npoles, regularize, c_input, sector, branch, parity_exchanged, regularized=offset,
    )


def build_super_point_table(request):
    """A frozen NS/Ramond point table from a resolved ``point-table`` request, at the current precision.

    The NS/Ramond counterpart of :func:`virasoro_blocks.point.point_block_table`: per coordinate the direct numeric
    super Taylor grid at the request's exchanged value (``exchange_h`` for a chiral basis, ``exchange_delta`` and
    ``exchange_spin`` otherwise) -- the sampler the tables are verified against -- frozen as degree-0 numerators with
    ``shift`` the exchanged value and the Virasoro point tables' decay base.  At an N=1 singular c the grid is the
    c -> c0 limit at fixed weight from the exact Laurent recursion (there is no irreducible alternative, as for
    ``evaluate``); an offset c (``c_regularization="on"`` elsewhere) follows the working precision as the tables'
    does.  A genuine exchanged pole raises the common ``BlockPoleError``.  The table carries the family metadata of
    the NS/Ramond tables, so the writers record it unchanged."""
    from .global_blocks import global_block_exp_base
    from .interpolation import DampedRational
    from .polynomial import Polynomial
    from .polyderivs import InterpolatedDeltaTable
    from .super_laurent import exact_singular_c
    from .superconformal_blocks import SuperBlockPolicy, _stamp_super_table

    family, basis = request.family, request.basis
    if family not in {FAMILY_NS, FAMILY_RAMOND}:
        raise ValueError(f"super point-table backend cannot handle family {family!r}")
    if request.central_charge is None:
        raise ValueError("superconformal point tables require central_charge")
    if request.policy.order is None:
        raise ValueError("superconformal point tables require order")
    is_chiral = basis.endswith("chiral_F")
    h_ext = list(request.h_ext or ())
    hbar_ext = None if is_chiral else list(request.hbar_ext if request.hbar_ext is not None else h_ext)
    point_value = request.exchange_h if is_chiral else request.exchange_delta
    if point_value is None:
        raise ValueError("superconformal point tables need exchange_h (chiral) or exchange_delta (2d)")
    from .numeric_input import _integer_spin

    spin = 0 if is_chiral else _integer_spin(request.exchange_spin or 0)
    q_order = int(request.policy.order)
    lam = int(request.policy.lambda_order)
    regularize = request.policy.c_regularization
    eps_exp = super_table_offset_eps_exp(mp.mp.dps)
    c_input, c_eval, regularized = resolve_super_c(request.central_charge, regularize=regularize, eps_exp=eps_exp)
    exact = regularized and exact_singular_c(c_input) is not None
    parity_exchanged = request.parity_exchanged or "1"
    tables = {}
    for coordinate in tuple(request.coordinates or (request.coordinate,)):
        kept = _super_kept_pairs(coordinate, request, spin, basis, is_chiral, lam)
        grid = _super_reference(point_value, coordinate, request, spin, family, basis, h_ext, hbar_ext, is_chiral,
                                q_order, lam, regularize, c_input, request.sector, request.branch, parity_exchanged,
                                eps_exp)
        m_max = max(lam, max((m for m, _n in kept), default=0))
        n_max = max((n for _m, n in kept), default=0)
        keep = set(kept)
        numerators = [[Polynomial([grid[m][n] if (m, n) in keep else mp.mpf(0)]) for n in range(n_max + 1)]
                      for m in range(m_max + 1)]
        nodes = [[(point_value,) for _n in range(n_max + 1)] for _m in range(m_max + 1)]
        base = global_block_exp_base(request.chiral_norm, center=request.center,
                                     center_bar=None if is_chiral else request.center_bar)
        tables[coordinate] = InterpolatedDeltaTable(
            prefactor=DampedRational(constant=1, poles=(), base=base), shift=point_value,
            sample_points=(point_value,), numerators=numerators, numerator_sample_points=nodes,
        )
    serialized = serialize_deriv_tables(tables)
    return _stamp_super_table(serialized, policy=SuperBlockPolicy(regularize=regularize, eps_exp=eps_exp),
                              family=family, c_input=c_input, c_eval=c_eval, regularized=regularized,
                              sector=request.sector, branch=request.branch, exact=exact)


def _verified_super_build(build_once, request, spin, options, family, basis, h_ext, hbar_ext, is_chiral, q_order,
                          lam, explicit_npoles, regularize, c_input, sector, branch, parity_exchanged, *,
                          regularized=False):
    """An NS/Ramond table built at the working precision ``--precision``, verified against the direct numeric super
    tables and reported (:func:`~virasoro_blocks.verified_build.run_verified_build`).

    With every pole kept (the default), the table
    represents the q_order-truncated block exactly; either way it is checked between and past its nodes against
    :func:`superconformal_blocks.numeric_full_table` / ``numeric_chiral_table``.  With ``--target-digits`` the
    working precision grows by the measured deficit.  At a regularized (offset) singular c the offset follows each
    build's working precision (:func:`super_table_offset_eps_exp`, unless ``options.eps_exp`` fixes it), so a
    precision escalation deepens it too; how far it leaves the c -> c0 limit is measured when the table falls short
    (:func:`_super_offset_digits`) and enters the precision the shortfall message names.
    """
    from dataclasses import replace as _replace

    from .verified_build import (
        diagnose_monomial, precision_request_of, record_verification, run_verified_build, verification_margin,
    )

    precision = precision_request_of(request.policy)
    digits = precision.digits
    order_reduced = explicit_npoles is not None
    npoles = 10 ** 6 if explicit_npoles is None else int(explicit_npoles)     # (clamped to the pole pool)
    fixed_eps = options.eps_exp is not None
    built_options = {}

    def options_at(work):
        return options if fixed_eps else _replace(options, eps_exp=super_table_offset_eps_exp(work))

    def build(work):
        with mp.workdps(work):
            serialized = build_once(npoles, options_at(work))
        built_options[id(serialized)] = options_at(work)
        return serialized

    def verify(serialized, work):
        with mp.workdps(work):
            return _super_verified_digits(
                serialized, request, spin, family, basis, h_ext, hbar_ext, is_chiral, q_order, lam, regularize,
                c_input, sector, branch, parity_exchanged, built_options[id(serialized)], digits,
            )

    def diagnose(serialized, verification, work):
        with mp.workdps(work):
            found = diagnose_monomial(serialized, verification, digits=digits, work=work, order_reduced=order_reduced)
            if not regularized or found.cause == "npoles":
                return found
            eps = int(built_options[id(serialized)].eps_exp)
            offset = _super_offset_digits(
                serialized, verification, request, spin, family, basis, h_ext, hbar_ext, is_chiral, q_order, lam,
                regularize, c_input, sector, branch, parity_exchanged, eps, digits,
            )
        if offset is None:
            return found
        margin = verification_margin(digits)
        # the offset's own loss: the limit is left by ~10^-(eps - amplification); the offset needs
        # eps >= digits + margin + amplification, i.e. the working precision that sets it there
        amplification = max(0.0, eps - offset)
        need_offset = 2 * (int(mp.ceil(digits + margin + amplification)) - 5)
        offset_bound = offset < verification.digits + 1
        if fixed_eps:
            if offset_bound:
                return _replace(found, cause="offset", lost=amplification, offset_eps=eps, offset_digits=offset)
            return found
        needed = max(found.needed_precision or 0, need_offset)
        if offset_bound:
            return _replace(found, cause="precision", mechanism="offset", needed_precision=needed, offset_eps=eps,
                            offset_digits=offset)
        return _replace(found, needed_precision=needed)

    prefix = "" if is_chiral else f"spin {spin}: "
    label = f"{family} {prefix}the " + (f"order-reduced table (--npoles {explicit_npoles})" if order_reduced
                                        else "table")
    serialized, verification, work = run_verified_build(
        build, verify, request=precision, what=label, order_reduced=order_reduced, diagnose=diagnose,
    )
    record_verification(serialized, verification, work, precision)
    serialized.regularization_eps_exp = int(built_options[id(serialized)].eps_exp) if regularized else None
    return serialized


def _super_kept_pairs(coordinate, request, spin, basis, is_chiral, lam):
    """Use the writer's derivative selection for construction and verification."""
    return kept_pairs(coordinate, lam, basis=basis, ell=0 if is_chiral else int(spin),
                      parity_projection=request.parity,
                      parity_phase=total_parity_phase(request.parity_external, request.parity_exchanged),
                      off_diagonal=is_off_diagonal(request.center, request.center_bar))


def _super_reference(delta, coordinate, request, spin, family, basis, h_ext, hbar_ext, is_chiral, q_order, lam,
                     regularize, c_input, sector, branch, parity_exchanged, eps_exp):
    """The direct numeric super table at ``delta``, the recurrence run at ``c + 10^-eps_exp``."""
    from .superconformal_blocks import numeric_chiral_table, numeric_full_table

    radial = coordinate.endswith("_radial")
    common = dict(c=c_input, q_order=q_order, m_max=lam, chiral_norm=request.chiral_norm,
                  z_shift=request.z_shift, sector=sector, branch=branch, regularize=regularize,
                  eps_exp=int(eps_exp), center=request.center)
    if is_chiral:
        return numeric_chiral_table(family, delta, coordinate, h_ext, **common)
    return numeric_full_table(
        family, basis, delta, int(spin), coordinate, h_ext, hbar_ext=hbar_ext,
        n_max=0 if radial else lam, zbar_shift=request.zbar_shift, parity=request.parity,
        parity_external=request.parity_external, exchanged_parity=parity_exchanged,
        center_bar=request.center_bar, **common,
    )


def _super_offset_digits(serialized, verification, request, spin, family, basis, h_ext, hbar_ext, is_chiral, q_order,
                         lam, regularize, c_input, sector, branch, parity_exchanged, eps_exp, target):
    """Digits to which the block at the table's offset ``c + 10^-eps_exp`` matches the c -> c0 limit (the block at
    ``10^-2 eps_exp``, the verification's reference), worst over the check points: what the offset alone leaves."""
    from .verified_build import point_agreement

    worst = None
    for x in verification.points:
        delta = mp.mpf(serialized.shift) + x
        for coordinate in serialized.coordinates:
            kept = _super_kept_pairs(coordinate, request, spin, basis, is_chiral, lam)
            if not kept:
                continue
            args = (delta, coordinate, request, spin, family, basis, h_ext, hbar_ext, is_chiral, q_order, lam,
                    regularize, c_input, sector, branch, parity_exchanged)
            measured = point_agreement(_super_reference(*args, eps_exp), _super_reference(*args, 2 * eps_exp),
                                       kept, target)
            if measured is not None and (worst is None or measured[0] < worst):
                worst = measured[0]
    return worst


def _super_verified_digits(serialized, request, spin, family, basis, h_ext, hbar_ext, is_chiral, q_order, lam,
                           regularize, c_input, sector, branch, parity_exchanged, options, target):
    """How well an NS/Ramond table matches the direct numeric super tables (a
    :class:`~virasoro_blocks.verified_build.Verification`), between its nodes and past them out to the
    large-Delta regime, as the Virasoro tables are checked; None when it has no nodes."""
    from .verified_build import (
        check_points_x, far_check_points_x, point_agreement, table_values, verification_from_points,
    )

    xs = getattr(serialized, "sample_points_x", None)
    if not xs:
        return None
    inner = check_points_x(xs, beyond_last=False)
    far = far_check_points_x(xs, list(serialized.pole_list_x))
    points = inner + far
    per_point = []
    for x in points:
        delta = mp.mpf(serialized.shift) + x
        worst = None
        for coordinate in serialized.coordinates:
            kept = _super_kept_pairs(coordinate, request, spin, basis, is_chiral, lam)
            if not kept:
                continue
            # at an offset singular c the table represents the c -> c0 limit (split pairs back on the axis), and
            # the limit is what it is checked against: the offset block itself strays from it by O(eps/(x-a)^4)
            # near a pole on the support (c = 3/2), so the reference is taken at eps^2
            ref = _super_reference(delta, coordinate, request, spin, family, basis, h_ext, hbar_ext, is_chiral,
                                   q_order, lam, regularize, c_input, sector, branch, parity_exchanged,
                                   2 * int(options.eps_exp))
            (got,) = table_values(serialized, coordinate, [delta], kept, dps=2 * mp.mp.dps)
            measured = point_agreement(got, ref, kept, target)
            if measured is not None:
                worst = measured if worst is None else (min(worst[0], measured[0]), min(worst[1], measured[1]))
        per_point.append(worst)
    last = max(mp.re(mp.mpc(x)) for x in xs)
    return verification_from_points(per_point, points, n_inner=len(inner), last_node=last, shift=serialized.shift,
                                    far_x=far[-1])


def _build_super_tables(request, spin, options, family, basis, coordinates, h_ext, hbar_ext, is_chiral, q_order, lam,
                        npoles, pole_selection, regularize, c_input, sector, branch, parity_exchanged):
    tables = {}
    for coordinate in coordinates:
        kept = _super_kept_pairs(coordinate, request, spin, basis, is_chiral, lam)
        if is_chiral:
            anchor = positivity_anchor(request.gap, 0, True)
            table = super_chiral_F_interpolated_table(
                family,
                coordinate,
                h_ext,
                c=c_input,
                m_max=lam,
                q_order=q_order,
                shift=anchor,
                chiral_norm=request.chiral_norm,
                z_shift=request.z_shift,
                sector=sector,
                branch=branch,
                npoles=npoles,
                pole_selection=pole_selection,
                center=request.center,
                regularize=regularize,
                eps_exp=int(options.eps_exp),
                num_threads=options.num_threads,
                shared_sample_points=options.shared_sample_points,
                kept_entries=kept,
            )
        else:
            anchor = positivity_anchor(request.gap, int(spin), False)
            table = super_2d_interpolated_table(
                family,
                basis,
                coordinate,
                int(spin),
                h_ext,
                hbar_ext=hbar_ext,
                c=c_input,
                m_max=lam,
                n_max=lam,
                q_order=q_order,
                shift=anchor,
                chiral_norm=request.chiral_norm,
                z_shift=request.z_shift,
                zbar_shift=request.zbar_shift,
                sector=sector,
                branch=branch,
                parity=request.parity,
                parity_external=request.parity_external,
                exchanged_parity=parity_exchanged,
                npoles=npoles,
                pole_selection=pole_selection,
                center=request.center,
                center_bar=request.center_bar,
                regularize=regularize,
                eps_exp=int(options.eps_exp),
                num_threads=options.num_threads,
                shared_sample_points=options.shared_sample_points,
                kept_entries=kept,
            )
        tables[coordinate] = table
    serialized = serialize_deriv_tables(tables)
    # the values at the sample nodes, from the fit's own records, written with the table
    from .clean_samples import attach_recorded_sample_block_values

    attach_recorded_sample_block_values(serialized, tables)
    return serialized


__all__ = ["SuperBackendOptions", "build_super_point_table", "build_super_table", "super_table_offset_eps_exp"]
