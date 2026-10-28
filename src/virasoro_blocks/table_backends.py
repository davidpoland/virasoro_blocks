"""Numerical backends for sphere derivative-table construction.

The command-line module owns parsing, validation, and artifact writing.  This
module owns the Virasoro fixed-spin derivative/interpolation build so the
family adapter and multiprocessing workers can call the same backend without
importing ``virasoro_blocks.__main__``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import mpmath as mp

from .truncation import total_parity_phase
from .params import b_from_c
from .polyderivs import (
    B_qqb_poly_table,
    B_yyb_poly_table,
    B_zzb_poly_table,
    F_q_poly_table,
    F_y_poly_table,
    F_z_poly_table,
    G_qqb_poly_table,
    G_qqb_ws_poly_table,
    G_qqb_ws_radial_poly_table,
    G_ws_poly_table,
    G_ws_radial_poly_table,
    G_xt_poly_table,
    G_xt_radial_poly_table,
    G_yyb_poly_table,
    G_zzb_poly_table,
    build_chiral_cache,
    build_numeric_sampler,
    build_table_from_cache,
    interpolate_delta_tables,
    numeric_residue_scoring_available,
    parity_projected_B_qqb_poly_table,
    parity_projected_B_qqb_ws_poly_table,
    parity_projected_B_qqb_ws_radial_poly_table,
    parity_projected_B_ws_poly_table,
    parity_projected_B_ws_radial_poly_table,
    parity_projected_B_xt_poly_table,
    parity_projected_B_xt_radial_poly_table,
    parity_projected_B_yyb_poly_table,
    parity_projected_B_zzb_poly_table,
)
from .clean_samples import attach_clean_sample_block_values
from .serialization import serialize_deriv_tables
from .verified_build import (
    agreement_digits,
    check_points_x,
    diagnose_monomial,
    far_check_points_x,
    point_agreement,
    precision_limit_digits,
    precision_request_of,
    record_verification,
    run_verified_build,
    table_values,
    verification_from_points,
    verification_margin,
)


_POLY_TABLE_BUILDERS = {
    "zzb": G_zzb_poly_table,
    "yyb": G_yyb_poly_table,
    "qqb": G_qqb_poly_table,
    "ws": G_ws_poly_table,
    "xt": G_xt_poly_table,
    "qqb_ws": G_qqb_ws_poly_table,
}
_B_POLY_TABLE_BUILDERS = {
    "zzb": B_zzb_poly_table,
    "yyb": B_yyb_poly_table,
    "qqb": B_qqb_poly_table,
}
_PROJECTED_B_POLY_TABLE_BUILDERS = {
    "zzb": parity_projected_B_zzb_poly_table,
    "yyb": parity_projected_B_yyb_poly_table,
    "qqb": parity_projected_B_qqb_poly_table,
    "ws": parity_projected_B_ws_poly_table,
    "xt": parity_projected_B_xt_poly_table,
    "qqb_ws": parity_projected_B_qqb_ws_poly_table,
}
_RADIAL_POLY_TABLE_BUILDERS = {
    "ws_radial": G_ws_radial_poly_table,
    "xt_radial": G_xt_radial_poly_table,
    "qqb_ws_radial": G_qqb_ws_radial_poly_table,
}
_PROJECTED_RADIAL_POLY_TABLE_BUILDERS = {
    "ws_radial": parity_projected_B_ws_radial_poly_table,
    "xt_radial": parity_projected_B_xt_radial_poly_table,
    "qqb_ws_radial": parity_projected_B_qqb_ws_radial_poly_table,
}
_RADIAL_COORDINATES = set(_RADIAL_POLY_TABLE_BUILDERS)
_FOLDED_COORDINATES = {"ws", "xt", "qqb_ws"}
_CHIRAL_POLY_TABLE_BUILDERS = {
    "z": F_z_poly_table,
    "y": F_y_poly_table,
    "q": F_q_poly_table,
}
_CHIRAL_COORDINATES = set(_CHIRAL_POLY_TABLE_BUILDERS)
_SKELETON_SAFE_COORDINATES = {
    "zzb", "yyb", "qqb", "ws", "xt", "qqb_ws",
    "ws_radial", "xt_radial", "qqb_ws_radial", "z", "y", "q",
}


@dataclass(frozen=True)
class VirasoroBackendOptions:
    """Non-kinematic controls needed by the fixed-spin backend.

    ``method`` is ``interpolation`` or ``full``; ``sample_via`` is ``symbolic``
    or ``numeric``. Frontends resolve automatic choices before constructing
    these options. ``regularize=None`` inherits the canonical request's policy.
    """

    method: str = "interpolation"
    sample_via: str = "symbolic"
    shared_sample_points: bool = False
    regularize: bool | None = None
    chiral_cache: Any = None

    def __post_init__(self):
        if self.method not in ("interpolation", "full"):
            raise ValueError("method must be 'interpolation' or 'full'")
        if self.sample_via not in ("symbolic", "numeric"):
            raise ValueError("sample_via must be 'symbolic' or 'numeric'")

    def regularize_for(self, request) -> bool:
        if self.regularize is not None:
            return self.regularize
        from .singularity import recognized_singular_locus

        return request.policy.c_regularization == "on" or (
            request.policy.c_regularization == "auto"
            and request.central_charge is not None
            and recognized_singular_locus(request.central_charge) is not None
        )


# coordinates the numeric (arb-native) node sampler serves for the 2d bases; chiral_F takes the chiral ones
NUMERIC_SAMPLE_VIA_COORDINATES = frozenset({
    "zzb", "yyb", "qqb", "ws", "xt", "qqb_ws",
    "ws_radial", "xt_radial", "qqb_ws_radial",
})


def default_sample_via(request, *, method: str) -> str:
    """What ``--sample-via auto`` resolves to for ``request``: ``numeric`` wherever the numeric sampler applies.

    The API used ``symbolic`` always, so ``build_table_request`` built different tables from the CLI's (parity '-'
    tables got other node sets and values).
    """
    chiral = request.basis in ("chiral_F", "virasoro_chiral_F")
    coordinates = tuple(request.coordinates or (request.coordinate,))
    allowed = {"z", "y", "q"} if chiral else NUMERIC_SAMPLE_VIA_COORDINATES
    if (method != "interpolation" or not request.policy.order
            or request.basis not in ("scalar_G", "chiral_F", "ordered_B", "parity_projected_B")
            or any(coordinate not in allowed for coordinate in coordinates)):
        return "symbolic"
    return "numeric"


def _selector_needs_symbolic_residue_scores(pole_selection: str) -> bool:
    return pole_selection in {
        "residue", "residue_all",
        "contribution_leading", "contribution_leading_all",
    }


def _numeric_skeleton_eligible(args, basis, *, regularize: bool) -> bool:
    """Whether a degree-only skeleton is safe for this interpolation run."""
    return numeric_skeleton_eligible(
        method=getattr(args, "method", None), sample_via=getattr(args, "sample_via", None), basis=basis,
        coordinates=args.coordinates, pole_selection=getattr(args, "pole_selection", None), regularize=regularize,
    )


def numeric_skeleton_eligible(*, method, sample_via, basis, coordinates, pole_selection, regularize: bool) -> bool:
    """Whether a degree-only skeleton (Kac poles and numerator degrees, no symbolic numerators) is safe.

    It is whenever the samples come from the numeric recursion and the pole selector ranks from numeric values:
    the symbolic numerators are then never read, and building them (the ~1e500-coefficient chiral u-series) was
    the largest serial step of a bootstrap build at large lambda.
    """
    if regularize:
        return False
    if method != "interpolation":
        return False
    if sample_via != "numeric":
        return False
    if basis not in ("scalar_G", "chiral_F", "ordered_B", "parity_projected_B"):
        return False
    if any(coord not in _SKELETON_SAFE_COORDINATES for coord in coordinates):
        return False
    if basis == "chiral_F" and _selector_needs_symbolic_residue_scores(pole_selection):
        return False
    return True


def _skeleton_regularize_eligible(args, basis, c, *, regularize: bool) -> bool:
    """Whether the regularized degree-only path is safe."""
    if not regularize:
        return False
    if getattr(args, "method", None) != "interpolation":
        return False
    if getattr(args, "sample_via", None) != "numeric":
        return False
    if basis not in ("scalar_G", "ordered_B", "parity_projected_B", "chiral_F"):
        return False
    if any(coord not in _SKELETON_SAFE_COORDINATES for coord in args.coordinates):
        return False
    if (
        (args.z_center is not None or args.zbar_center is not None)
        and basis not in ("scalar_G", "ordered_B", "parity_projected_B")
    ):
        return False
    if _selector_needs_symbolic_residue_scores(args.pole_selection):
        return False
    return True


def _backend_args(request, options: VirasoroBackendOptions):
    policy = request.policy
    return SimpleNamespace(
        lam=int(policy.lambda_order),
        order=None if policy.order is None else int(policy.order),
        coordinates=list(request.coordinates),
        parity_project=request.parity,
        parity_ext=(
            None if request.parity_external is None
            else list(request.parity_external)
        ),
        parity_exchanged=request.parity_exchanged,
        chiral_norm=request.chiral_norm,
        z_shift=request.z_shift,
        zbar_shift=request.zbar_shift,
        z_center=request.center,
        zbar_center=request.center_bar,
        method=options.method,
        sample_via=options.sample_via,
        shared_sample_points=options.shared_sample_points,
        npoles=policy.npoles,
        pole_selection=policy.pole_selection or "contribution_optimal",
        gap=request.gap,
        precision=policy.precision,
        target_digits=policy.target_digits,
        max_precision=policy.max_precision,
    )



def positivity_anchor(gap, ell, chiral):
    """Polynomial anchor for the emitted table: ``None`` = the basis default.

    The block's own ``shift`` IS the positivity domain -- the polynomial variable
    is ``x = Delta - shift`` and SDPB imposes positivity on ``x >= 0``, so the
    constraint is ``Delta >= shift``. (``build_prefactored_pmp_doc`` never sees a
    separate ``delta_min``; that field is bookkeeping.) So a gap is imposed by
    moving this anchor, which is what the in-process bootstrap does when it calls
    ``interpolate_delta_table(..., shift=delta_min)`` per probe.

    ``--gap G`` raises the anchor G ABOVE the basis's unitarity bound: ``|ell| + G``
    for the 2d bases, ``G`` for chiral (whose bound is 0). Measuring from
    unitarity rather than absolutely is what makes one flag mean the same physics
    across spins -- a uniform twist gap -- and keeps ``--gap 0`` identical to the
    current default.

    Not to be confused with ``--z-shift``/``--zbar-shift``/``--u-shift``, which
    move the KINEMATIC z-exponent in the block prefactor (the crossing stripping,
    typically ``-2 h_ext``) and have nothing to do with the positivity domain.
    """
    if gap is None:
        return None
    from .numeric_input import literal_components, literal_number

    unitarity = 0 if chiral else abs(literal_components(ell)[0])
    return unitarity + literal_number(gap)

# (shared with the other table paths; the historical names stay importable from here)
_check_points_x = check_points_x
_far_check_points_x = far_check_points_x
_verification_margin = verification_margin
_agreement_digits = agreement_digits
_precision_limit_digits = precision_limit_digits


def _exact_poles_x(exact_tables, shift):
    """The exact tables' Delta-poles in x = Delta - shift (the full pool, also for an order-reduced table)."""
    exact = next(iter(exact_tables.values()), None)       # (the Kac data are shared by every coordinate)
    return [mp.mpc(root) - mp.mpf(shift) for root in (getattr(exact, "denominator_roots", None) or ())]


def _verified_digits(serialized, exact_tables, kept_entries_by_coord, target: int):
    """How well the serialized table matches its own exact sampler (a :class:`~virasoro_blocks.verified_build.
    Verification`), or None (nothing to check against).

    Checked between the nodes and past them out to the large-Delta regime (:func:`far_check_points_x`), for a
    keep-all and an order-reduced table alike: SDPB uses the table at every x >= 0.  (An exact ``--npoles all``
    table has no nodes: it is checked at 0.31, 2.3 and the far points.)  The table is evaluated from its stored
    coefficients at twice the working precision (a consumer's working precision); the reference is the numeric
    sampler (or the exact symbolic table, for ``--sample-via symbolic``) at the build's precision.  Per entry,
    entries more than ``target`` digits below the largest are compared at that absolute level; the largest-entry
    measure compares every entry with the largest.
    """
    xs = getattr(serialized, "sample_points_x", None) or ()
    inner = check_points_x(xs, beyond_last=False)
    far = far_check_points_x(xs, _exact_poles_x(exact_tables, serialized.shift) + list(serialized.pole_list_x))
    points = inner + far
    per_point = [None] * len(points)
    for coord, exact in exact_tables.items():
        kept = list(kept_entries_by_coord.get(coord) or ())
        if not kept:
            continue
        deltas = [mp.mpf(serialized.shift) + x for x in points]
        sampler = getattr(exact, "numeric_sampler", None)
        if sampler is not None:
            refs, _is_arb = sampler(deltas, entries=kept)
        elif xs and getattr(exact, "numerators", None) is not None:
            refs = [exact.evaluate(d) for d in deltas]
        else:
            continue
        gots = table_values(serialized, coord, deltas, kept, dps=2 * mp.mp.dps)
        for i, (got, ref) in enumerate(zip(gots, refs)):
            measured = point_agreement(got, ref, kept, target)
            if measured is None:
                continue
            if per_point[i] is None:
                per_point[i] = measured
            else:
                per_point[i] = (min(per_point[i][0], measured[0]), min(per_point[i][1], measured[1]))
    last = max((mp.re(mp.mpc(x)) for x in xs), default=None)
    return verification_from_points(per_point, points, n_inner=len(inner), last_node=last, shift=serialized.shift,
                                    far_x=far[-1])


def _process_spin(
    ell, args, h_ext, hbar_ext, b, c, spinning, projected, chiral, basis,
    *, regularize: bool, chiral_cache=None,
) -> object:
    """Build one fixed-spin Virasoro serialized derivative table at ``--precision``, verified and reported.

    The table is built from the baseline working precision ``--precision`` (the chiral cache the CLI built there is reused),
    checked against its own exact sampler between the nodes and past them out to its large-Delta regime
    (:func:`_verified_digits`; past the last node the monomial form extrapolates its node values, and loses the most
    there), and its ``verified_digits`` recorded.  Monomial-form tables lose digits to coefficient cancellation
    (about 0.6-0.75 per unit of degree), so a keep-all table built at P delivers fewer than P: the shortfall is
    reported in one line with its measured cause and the precision that fixes it
    (:func:`~virasoro_blocks.verified_build.shortfall_message`).  With ``--target-digits`` the build escalates
    by the measured deficit instead (:func:`~virasoro_blocks.verified_build.run_verified_build`), and a table that
    cannot reach the target raises. The build precision is recorded as ``working_dps`` and the table is written at
    it.
    """
    request = precision_request_of(args)
    spin = int(ell) if ell is not None and ell == int(ell) else ell       # (a float 0.0 read "spin 0.0")
    order_reduced = getattr(args, "npoles", None) is not None
    cache_dps = mp.mp.dps          # the precision the caller's chiral cache was built at

    def build(work):
        with mp.workdps(work):
            b_work = None if (regularize or b is None) else b_from_c(c)
            return _process_spin_once(
                ell, args, h_ext, hbar_ext, b_work, c, spinning, projected, chiral, basis,
                regularize=regularize, chiral_cache=chiral_cache if work == cache_dps else None,
            )

    def verify(built, work):
        serialized, exact_tables, kept = built
        with mp.workdps(work):
            return _verified_digits(serialized, exact_tables, kept, request.digits)

    def diagnose(built, verification, work):
        with mp.workdps(work):
            return diagnose_monomial(built[0], verification, digits=request.digits, work=work,
                                     order_reduced=order_reduced)

    prefix = "" if chiral else f"spin {spin}: "
    if order_reduced:
        label = f"{prefix}the order-reduced table (--npoles {args.npoles})"
    elif args.method == "full":
        label = f"{prefix}the exact table (--npoles all)"
    else:
        label = f"{prefix}the table"
    (serialized, _exact, _kept), verification, work = run_verified_build(
        build, verify, request=request, what=label, order_reduced=order_reduced, diagnose=diagnose,
    )
    record_verification(serialized, verification, work, request)
    return serialized


def _process_spin_once(
    ell, args, h_ext, hbar_ext, b, c, spinning, projected, chiral, basis,
    *, regularize: bool, chiral_cache=None,
) -> object:
    """One build of a fixed-spin Virasoro serialized derivative table at the current precision."""
    lam = args.lam
    q_order = args.order
    parity_exchanged = (
        args.parity_exchanged if args.parity_exchanged is not None else "1"
    )

    def _make_table_from_cache(coord):
        cs_h, cs_hbar = chiral_cache[coord]
        radial = coord in _RADIAL_COORDINATES
        chiral_coord = coord in _CHIRAL_COORDINATES
        lambda_cut = None
        if chiral_coord:
            m_max, n_max = lam, 0
        elif radial:
            m_max, n_max = lam, 0
        elif coord in _FOLDED_COORDINATES:
            from .truncation import kept_pairs as _kp
            kept = _kp(
                coord, lam, basis=basis, ell=ell,
                parity_projection=args.parity_project, parity_phase=total_parity_phase(args.parity_ext, args.parity_exchanged),
            )
            m_max = lam
            n_max = max((n for _m, n in kept), default=0)
        else:
            m_max, n_max = lam, lam
            lambda_cut = lam
        return build_table_from_cache(
            coord, cs_h, cs_hbar, ell,
            m_max=m_max, n_max=n_max,
            basis=basis,
            parity_projection=args.parity_project,
            parity_external=args.parity_ext,
            exchanged_parity=parity_exchanged,
            lambda_cut=lambda_cut,
        )

    def _make_table_fresh(coord, *, b_value=None, c_value=None, skeleton=False):
        skel_kw = {"skeleton": True} if skeleton else {}
        common = dict(
            b=b_value, c=c_value, q_order=q_order,
            chiral_norm=args.chiral_norm,
            z_shift=args.z_shift, zbar_shift=args.zbar_shift,
        )
        center_kw = {} if args.z_center is None else {"center": args.z_center}
        center_2d_kw = dict(center_kw)
        if args.zbar_center is not None:
            center_2d_kw["center_bar"] = args.zbar_center
        if chiral:
            return _CHIRAL_POLY_TABLE_BUILDERS[coord](
                h_ext, lam, **common, **center_kw, **skel_kw,
            )
        radial = coord in _RADIAL_COORDINATES
        dims = (lam,) if radial else (lam, lam)
        nonfolded_center = {} if radial else center_2d_kw
        if basis == "ordered_B" or projected:
            if projected:
                builders = (
                    _PROJECTED_RADIAL_POLY_TABLE_BUILDERS if radial
                    else _PROJECTED_B_POLY_TABLE_BUILDERS
                )
                return builders[coord](
                    ell, h_ext, hbar_ext, *dims,
                    parity=args.parity_project,
                    parity_external=args.parity_ext,
                    exchanged_parity=parity_exchanged,
                    **common, **nonfolded_center, **skel_kw,
                )
            return _B_POLY_TABLE_BUILDERS[coord](
                ell, h_ext, hbar_ext, *dims,
                **common, **nonfolded_center, **skel_kw,
            )
        builders = _RADIAL_POLY_TABLE_BUILDERS if radial else _POLY_TABLE_BUILDERS
        return builders[coord](
            ell, h_ext, *dims, **common, **nonfolded_center, **skel_kw,
        )

    def _make_table(coord, *, b_value=None, c_value=None, skeleton=False):
        off_center = args.z_center is not None or args.zbar_center is not None
        if chiral_cache is not None and c_value is None and not off_center:
            return _make_table_from_cache(coord)
        return _make_table_fresh(
            coord, b_value=b_value, c_value=c_value, skeleton=skeleton
        )

    skeleton_regularize = _skeleton_regularize_eligible(
        args, basis, c, regularize=regularize
    )
    numeric_skeleton = _numeric_skeleton_eligible(
        args, basis, regularize=regularize
    )
    exact_tables = {}
    for coord in args.coordinates:
        if skeleton_regularize:
            from .regularization import merged_delta_skeleton
            exact = merged_delta_skeleton(
                lambda c_val, _coord=coord: _make_table(_coord, c_value=c_val, skeleton=True), c=c,
            )
        elif regularize:
            from .regularization import merged_delta_table
            exact = merged_delta_table(lambda c_val, _coord=coord: _make_table(_coord, c_value=c_val), c=c)
        else:
            exact = _make_table(coord, b_value=b, skeleton=numeric_skeleton)
        exact_tables[coord] = exact
    # the numeric sampler supplies the interpolation's node values, and is the reference the table is verified
    # against (for an exact --npoles all table too)
    if (args.method == "interpolation" and args.sample_via == "numeric") or args.method == "full":
        for coord, exact in exact_tables.items():
            sampler_kwargs = dict(
                coordinate=coord,
                chiral=chiral,
                # This legacy sampler flag selects ordered B, including its
                # scalar-external limit; the requested basis owns that choice.
                spinning=basis == "ordered_B",
                projected=projected,
                h_externals=h_ext,
                hbar_externals=hbar_ext,
                ell=ell,
                m_max=exact.m_max,
                n_max=exact.n_max,
                b=None,
                q_order=q_order,
                chiral_norm=args.chiral_norm,
                z_shift=args.z_shift,
                zbar_shift=args.zbar_shift,
                center=args.z_center,
                center_bar=args.zbar_center,
                parity=args.parity_project,
                parity_external=args.parity_ext,
                exchanged_parity=parity_exchanged,
            )
            if regularize:
                from .regularization import _RegularizedNumericSampler
                exact.numeric_sampler = _RegularizedNumericSampler(
                    c=c, **sampler_kwargs
                )
            else:
                exact.numeric_sampler = build_numeric_sampler(
                    c=c, **sampler_kwargs
                )

        if _selector_needs_symbolic_residue_scores(args.pole_selection):
            for coord, exact in list(exact_tables.items()):
                if exact.numerators is None and not numeric_residue_scoring_available(
                    exact
                ):
                    if regularize:
                        from .regularization import merged_delta_table
                        full = merged_delta_table(
                            lambda c_val, _coord=coord: _make_table(_coord, c_value=c_val), c=c)
                    else:
                        full = _make_table_fresh(coord, b_value=b)
                    full.numeric_sampler = exact.numeric_sampler
                    exact_tables[coord] = full

    if args.method == "interpolation":
        first_exact = next(iter(exact_tables.values()))
        kept = (
            args.npoles if args.npoles is not None
            else len(first_exact.denominator_roots)
        )
        kept_entries_by_coord = {
            coord: _kept_pairs_for(coord, args, basis, ell)
            for coord in exact_tables
        }
        tables = interpolate_delta_tables(
            exact_tables, npoles=kept, pole_selection=args.pole_selection,
            shared_sample_points=args.shared_sample_points,
            kept_entries_by_coord=kept_entries_by_coord,
            shift=positivity_anchor(getattr(args, "gap", None), ell, chiral),
        )
        # The interpolated tables already contain the anchor; serialization
        # rejects a second one.
        serialized = serialize_deriv_tables(tables)
        # Store the node values so the SDP writer can use them directly,
        # avoiding cancellation in the monomial numerators.
        for coord, table in tables.items():
            attach_clean_sample_block_values(serialized, table, exact_tables[coord], coord)
    else:
        tables = exact_tables
        # The exact path carries no anchor of its own, so it takes one here.
        serialized = serialize_deriv_tables(
            tables, shift=positivity_anchor(getattr(args, "gap", None), ell, chiral)
        )
    if args.method == "interpolation" and basis in {"scalar_G", "chiral_F"}:
        serialized.sample_via = args.sample_via
    return serialized, exact_tables, kept_entries_by_coord if args.method == "interpolation" else {
        coord: _kept_pairs_for(coord, args, basis, ell) for coord in exact_tables
    }


def _kept_pairs_for(coord, args, basis, ell):
    from .truncation import is_off_diagonal, kept_pairs as _kept_pairs

    return _kept_pairs(coord, args.lam, basis=basis, ell=ell, parity_projection=args.parity_project, parity_phase=total_parity_phase(args.parity_ext, args.parity_exchanged),
                       off_diagonal=is_off_diagonal(getattr(args, "z_center", None),
                                                    getattr(args, "zbar_center", None)))


def build_cli_chiral_cache(args, h_ext, hbar_ext, b, *, regularize: bool, basis):
    """Build the fixed-c chiral cache used by the CLI and worker pool."""
    if regularize:
        return None
    return build_chiral_cache(
        args.coordinates,
        h_ext,
        hbar_externals=hbar_ext,
        lambda_=args.lam,
        b=b,
        q_order=args.order,
        chiral_norm=args.chiral_norm,
        z_shift=args.z_shift,
        zbar_shift=args.zbar_shift,
        skeleton=_numeric_skeleton_eligible(args, basis, regularize=regularize),
        truncated=True,
    )


def build_virasoro_table(
    request,
    *,
    spin,
    options: VirasoroBackendOptions,
):
    """Build one Virasoro table from a canonical request."""
    args = _backend_args(request, options)
    h_ext = list(request.h_ext or ())
    hbar_ext = None if request.hbar_ext is None else list(request.hbar_ext)
    c = request.central_charge
    if c is None:
        raise ValueError("Virasoro tables require central_charge")
    spinning = hbar_ext is not None and any(
        h != hb for h, hb in zip(h_ext, hbar_ext)
    )
    projected = request.parity is not None
    chiral = request.basis == "chiral_F"
    regularize = options.regularize_for(request)
    b = None if regularize else b_from_c(c)
    return _process_spin(
        spin, args, h_ext, hbar_ext, b, c, spinning, projected, chiral,
        request.basis,
        regularize=regularize,
        chiral_cache=options.chiral_cache,
    )


__all__ = [
    "VirasoroBackendOptions",
    "positivity_anchor",
    "build_cli_chiral_cache",
    "build_virasoro_table",
]
