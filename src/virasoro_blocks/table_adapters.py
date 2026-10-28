"""Family adapters for derivative-table operations.

The numerical table builders predate the operation-oriented API and therefore
have deliberately different call signatures.  This module is the narrow
boundary between the canonical :class:`~virasoro_blocks.requests.BlockRequest`
and those builders.  Global, Virasoro, and NS/R sphere tables, the torus
tables, and the frozen point tables of every family are called through
package-owned backends.

Keeping the registry here, rather than in ``__main__``, is important for two
reasons:

* Python callers and future CLI frontends can discover the same table
  capabilities; and
* choosing a family is no longer itself a numerical implementation detail.

The adapters intentionally do not import CLI modules.  Backends can import
this registry safely, and the CLI only supplies parsing and artifact writing.
"""

from __future__ import annotations

import mpmath as mp

from dataclasses import replace
from types import MappingProxyType
from typing import Any, Mapping

from .truncation import total_parity_phase
from .requests import (
    MIN_TABLE_PRECISION,
    BlockCapabilities,
    BlockRequest,
    NumericalPolicy,
    RequestValidationError,
    capabilities_for,
    validate_request,
)
from .params import b_from_c


class TableAdapter:
    """Common contract for one block family's table operation.

    Registered production adapters provide concrete package backends.  The
    base class only supplies validation and a clear failure for an adapter
    that has not implemented its family-specific operation.
    """

    family: str

    @property
    def capabilities(self) -> BlockCapabilities:
        return capabilities_for(self.family)[0]

    def validate(self, request: BlockRequest) -> BlockCapabilities:
        """Validate that ``request`` belongs to this table adapter."""
        if request.family != self.family:
            raise RequestValidationError(
                f"table adapter {self.family!r} cannot handle family "
                f"{request.family!r}"
            )
        if request.operation not in {"table", "point-table"}:
            raise RequestValidationError(
                f"table adapter {self.family!r} does not support operation "
                f"{request.operation!r}"
            )
        capability = validate_request(request)
        if (
            request.surface == "sphere"
            and int(request.policy.precision) < MIN_TABLE_PRECISION
        ):
            raise RequestValidationError(
                f"sphere table builds need precision >= {MIN_TABLE_PRECISION}"
            )
        return capability

    def build_table(
        self,
        request: BlockRequest,
        **_: Any,
    ) -> Any:
        """Build one table through the registered family backend."""
        self.validate(request)
        raise NotImplementedError(
            f"the {self.family} table backend has not been extracted"
        )

    def build_verified_point_table(self, request: BlockRequest) -> Any:
        """The ``point-table`` operation: :meth:`build_point_table` at the request's working precision, verified
        against the same table at more digits and reported (escalated for ``target_digits``) like every table."""
        request = _normalize_sphere_table_request(request)
        chiral = _short_table_basis(request.basis) == "chiral_F" or (
            request.basis == "auto" and all(co in _CHIRAL_TABLE_COORDINATES for co in request.coordinates))
        value = request.exchange_h if chiral else request.exchange_delta
        spin = "" if chiral else f" spin {request.exchange_spin or 0}"
        what = f"the point table at {'h' if chiral else 'Delta'} = {mp.nstr(mp.mpf(mp.re(mp.mpc(value))), 8)}{spin}"

        def build(work):
            with mp.workdps(work):
                return self.build_point_table(request)

        return _self_verified_build(build, request, what=what, point=True)

    def build_point_table(self, request: BlockRequest) -> Any:
        """Build a frozen Virasoro/global point table from a canonical request (at the current precision,
        unverified: :meth:`build_verified_point_table` is the operation).  The NS/Ramond and torus adapters
        override it."""
        self.validate(request)
        if self.family not in {"virasoro", "global"}:
            raise NotImplementedError(
                f"the {self.family} family has no point-table backend"
            )
        from .point import point_block_table

        request = _resolve_sphere_table_basis(request)
        basis = request.basis
        chiral = basis in {"chiral_F", "global_chiral_F"}
        if request.h_ext is None or len(request.h_ext) != 4:
            raise RequestValidationError(
                "point-table requests need four external holomorphic weights"
            )
        if not chiral and (request.hbar_ext is None or len(request.hbar_ext) != 4):
            raise RequestValidationError(
                "2d point-table requests need four antiholomorphic weights"
            )
        point_value = request.exchange_h if chiral else request.exchange_delta
        if point_value is None:
            raise RequestValidationError(
                "point-table requests need exchange_h for chiral bases or "
                "exchange_delta for 2d bases"
            )
        if request.policy.lambda_order is None:
            raise RequestValidationError(
                "point-table requests need a lambda_order"
            )
        return point_block_table(
            request.coordinates,
            basis=basis,
            ell=0 if chiral else (request.exchange_spin or 0),
            point_value=point_value,
            h_externals=request.h_ext,
            hbar_externals=None if chiral else request.hbar_ext,
            lam=int(request.policy.lambda_order),
            c=request.central_charge,
            q_order=request.policy.order,
            chiral_norm=request.chiral_norm,
            z_shift=request.z_shift,
            zbar_shift=None if chiral else request.zbar_shift,
            center=request.center,
            center_bar=None if chiral else request.center_bar,
            parity=None if chiral else request.parity,
            parity_external=None if chiral else request.parity_external,
            exchanged_parity=request.parity_exchanged or "1",
            regularize=request.policy.c_regularization,
        )


_CHIRAL_TABLE_COORDINATES = {"z", "y", "q"}
_NON_FOLDED_TABLE_COORDINATES = {"zzb", "yyb", "qqb"}
_SHORT_BASES = ("scalar_G", "ordered_B", "parity_projected_B", "chiral_F")


def _short_table_basis(basis: str) -> str:
    for prefix in ("virasoro_", "global_", "ns_", "ramond_"):
        if basis.startswith(prefix):
            return basis[len(prefix):]
    return basis


def _normalize_sphere_table_request(request: BlockRequest) -> BlockRequest:
    """Fill the defaults a sphere table/point-table backend needs.

    * ``coordinates=()`` falls back to ``(request.coordinate,)`` (the backends
      iterate ``coordinates`` and failed with StopIteration / an empty table);
    * a ``delta_ext`` (+ ``spin_ext``) request gets ``h = (Delta + s)/2``,
      ``hbar = (Delta - s)/2`` exactly as the table CLI derives them;
    * ``lambda_order`` is required.
    """
    if request.policy.lambda_order is None:
        raise RequestValidationError(
            f"{request.operation} requests need a lambda_order"
        )
    if not request.coordinates:
        request = replace(request, coordinates=(request.coordinate,))
    if request.h_ext is None and request.delta_ext is not None:
        from .evaluation import _label
        from fractions import Fraction
        from .numeric_input import _label_scale, _label_sum

        delta = tuple(_label(value) for value in request.delta_ext)
        spin = tuple(_label(value) for value in (request.spin_ext or (0, 0, 0, 0)))
        if len(delta) != 4 or len(spin) != 4:
            raise RequestValidationError(
                "delta_ext and spin_ext must contain four values"
            )
        chiral = _short_table_basis(request.basis) == "chiral_F" or (
            request.basis == "auto"
            and all(co in _CHIRAL_TABLE_COORDINATES for co in request.coordinates)
        )
        if chiral and any(value != 0 for value in spin):
            raise RequestValidationError("chiral blocks require scalar external data")
        from .verified_build import precision_request_of

        with mp.workdps(precision_request_of(request.policy).input_dps):
            request = replace(
                request,
                h_ext=tuple(_label_scale(_label_sum(d, s), Fraction(1, 2)) for d, s in zip(delta, spin)),
                hbar_ext=(
                    None if chiral
                    else tuple(_label_scale(_label_sum(d, _label_scale(s, -1)), Fraction(1, 2))
                               for d, s in zip(delta, spin))
                ),
            )
    return request


def _resolve_sphere_table_basis(request: BlockRequest) -> BlockRequest:
    """Resolve ``basis="auto"`` and enforce the basis / externals rules of the table CLI.

    Virasoro keeps the short basis names its backend knows; global, NS and
    Ramond get their family prefix. Scalar externals default to ``scalar_G``;
    an explicit ``ordered_B`` keeps the individual spin orientation even when
    the two external weight lists coincide.
    """
    request = _normalize_sphere_table_request(request)
    spinning = _spinning_externals(request)
    basis = request.basis
    if basis == "auto":
        coords = tuple(request.coordinates)
        if all(co in _CHIRAL_TABLE_COORDINATES for co in coords):
            basis = "chiral_F"
        elif request.parity is not None:
            basis = "parity_projected_B"
        else:
            basis = "ordered_B" if spinning else "scalar_G"
    short = _short_table_basis(basis)
    if short == "parity_projected_B":
        if request.parity not in {"+", "-"}:
            raise RequestValidationError(
                "parity_projected_B requires parity '+' or '-'"
            )
    elif request.parity is not None:
        raise RequestValidationError(
            f"parity does not apply to {basis}; use parity_projected_B"
        )
    basis = short if request.family == "virasoro" else f"{request.family}_{short}"
    if basis != request.basis:
        request = replace(request, basis=basis)
    if short == "ordered_B":
        if any(co not in _NON_FOLDED_TABLE_COORDINATES for co in request.coordinates):
            raise RequestValidationError(
                f"{basis} requires non-folded coordinates zzb, yyb, or qqb; "
                "use scalar_G or parity_projected_B for a parity fold"
            )
        if request.hbar_ext is None:
            request = replace(request, hbar_ext=request.h_ext)
    _reject_spinning_scalar_G(basis, spinning)
    return request


def _reject_spinning_scalar_G(basis: str, spinning: bool) -> None:
    """scalar_G is built from the holomorphic externals alone; with h_i != hbar_i
    the backends would silently drop hbar_ext (or build the ordered B block)."""
    if basis.endswith("scalar_G") and spinning:
        raise RequestValidationError(
            f"{basis} requires scalar external weights (h_i == hbar_i); use "
            "ordered_B or parity_projected_B for spinning externals"
        )


def _spinning_externals(request: BlockRequest) -> bool:
    if request.h_ext is None or request.hbar_ext is None:
        return False
    return any(h != hb for h, hb in zip(request.h_ext, request.hbar_ext))


class VirasoroTableAdapter(TableAdapter):
    family = "virasoro"

    def build_table(
        self,
        request: BlockRequest,
        *,
        spin: Any = 0,
        backend_options=None,
        **_: Any,
    ) -> Any:
        """Build one fixed-spin table through the package backend."""
        self.validate(request)
        request = _resolve_sphere_table_basis(request)
        from .table_backends import (
            VirasoroBackendOptions,
            _backend_args as _backend_args_for_cache,
            build_cli_chiral_cache,
            build_virasoro_table,
        )

        # npoles="all" keeps every pole: the exact high-degree table, exactly
        # as the CLI maps ``--npoles all`` to the exact (method="full") path.
        keep_all_poles = request.policy.npoles == "all"
        if keep_all_poles:
            request = replace(request, policy=replace(request.policy, npoles=None))
        if backend_options is None:
            from .table_backends import default_sample_via

            backend_options = VirasoroBackendOptions(
                sample_via=default_sample_via(request, method="full" if keep_all_poles else "interpolation"),
            )
        if backend_options.regularize is None:
            backend_options = replace(
                backend_options, regularize=backend_options.regularize_for(request),
            )
        if keep_all_poles and backend_options.method != "full":
            backend_options = replace(backend_options, method="full")
        if (
            backend_options.chiral_cache is None
            and not backend_options.regularize
            and request.central_charge is not None
        ):
            backend_args = _backend_args_for_cache(request, backend_options)
            cache = build_cli_chiral_cache(
                backend_args,
                list(request.h_ext or ()),
                None if request.hbar_ext is None else list(request.hbar_ext),
                b_from_c(request.central_charge),
                regularize=False,
                basis=request.basis,
            )
            backend_options = replace(backend_options, chiral_cache=cache)
        return build_virasoro_table(
            request, spin=spin, options=backend_options
        )


class GlobalTableAdapter(TableAdapter):
    family = "global"

    def build_table(
        self,
        request: BlockRequest,
        *,
        spin: Any = 0,
        num_threads: int = 1,
        shared_sample_points: bool = False,
        **_: Any,
    ) -> Any:
        """Build a global table directly from a canonical request.

        Global blocks do not use the Virasoro q-recursion.  This is therefore
        a small, direct adapter around the existing 2F1 interpolation
        backends, including their folded/radial-coordinate handling.
        """
        self.validate(request)
        request = _resolve_sphere_table_basis(request)
        if request.policy.npoles == "all":
            raise RequestValidationError(
                "global tables only expose integer-npoles interpolation; "
                "npoles='all' is not supported"
            )
        from .global_blocks import (
            global_chiral_F_interpolated_table,
            global_ordered_B_interpolated_table,
            global_parity_projected_B_interpolated_table,
            global_scalar_G_interpolated_table,
        )
        from .serialization import serialize_deriv_tables
        from .table_backends import positivity_anchor
        from .truncation import is_off_diagonal, kept_pairs

        basis = request.basis
        lam = int(request.policy.lambda_order)
        explicit_npoles = request.policy.npoles
        h_ext = list(request.h_ext or ())
        hbar_ext = None if request.hbar_ext is None else list(request.hbar_ext)
        parity_exchanged = request.parity_exchanged or "1"

        def build_coordinate(coord, npoles, q_order):
            # npoles None keeps the whole order-q_order pool (nothing to rank)
            selection = request.policy.pole_selection or "contribution_optimal"
            if npoles is None:
                npoles, selection = 10 ** 6, "rightmost"
            kept = kept_pairs(
                coord, lam, basis=basis, ell=spin,
                parity_projection=request.parity, parity_phase=total_parity_phase(request.parity_external, request.parity_exchanged),
                off_diagonal=is_off_diagonal(request.center, request.center_bar),
            )
            if coord.endswith("_radial"):
                build_coord = coord[: -len("_radial")]
                n_max = 0
            else:
                build_coord = coord
                n_max = lam
            anchor = positivity_anchor(
                request.gap, spin, basis == "global_chiral_F"
            )
            common = dict(
                shift=anchor,
                chiral_norm=request.chiral_norm,
                z_shift=request.z_shift,
                q_order=q_order,
                npoles=npoles,
                pole_selection=selection,
                num_threads=num_threads,
                shared_sample_points=shared_sample_points,
                kept_entries=kept,
                center=request.center,
                selection_coordinate="zzb",
                selection_m_max=lam,
                selection_n_max=lam,
            )
            if basis == "global_scalar_G":
                return global_scalar_G_interpolated_table(
                    build_coord, spin, h_ext, lam, n_max,
                    center_bar=request.center_bar, **common,
                )
            if basis == "global_ordered_B":
                return global_ordered_B_interpolated_table(
                    build_coord, spin, h_ext, hbar_ext, lam, n_max,
                    zbar_shift=request.zbar_shift,
                    center_bar=request.center_bar, **common,
                )
            if basis == "global_parity_projected_B":
                return global_parity_projected_B_interpolated_table(
                    build_coord, spin, h_ext, hbar_ext, lam, n_max,
                    parity=request.parity,
                    parity_external=request.parity_external,
                    exchanged_parity=parity_exchanged,
                    zbar_shift=request.zbar_shift,
                    center_bar=request.center_bar, **common,
                )
            if basis == "global_chiral_F":
                # Chiral tables have no antiholomorphic center or shift and
                # retain the requested coordinate rather than folding it.
                return global_chiral_F_interpolated_table(
                    coord, h_ext, lam,
                    shift=anchor,
                    chiral_norm=request.chiral_norm,
                    z_shift=request.z_shift,
                    q_order=q_order,
                    npoles=npoles,
                    pole_selection=selection,
                    num_threads=num_threads,
                    shared_sample_points=shared_sample_points,
                    kept_entries=kept,
                    center=request.center,
                    selection_coordinate="z",
                    selection_m_max=lam,
                )
            raise RequestValidationError(f"unsupported global basis {basis!r}")

        return _verified_global_build(self, request, build_coordinate, explicit_npoles, lam, spin)


def chosen_pole_start(precision, lam):
    """The first pole count of a chosen-count build (global fixed-spin tables and large-spin towers without
    ``--npoles``): ~1.6 digits per pole (64 poles: 105 digits, 76: 124), and a degree ~ poles + 2 lambda that loses
    ~0.75 digits per unit, so with a target the precision follows and without one it is the count the working
    precision carries.  :func:`pole_count_knobs` grows it from there."""
    first = int(mp.ceil(mp.mpf(precision.digits) / mp.mpf("1.6"))) + 4
    if precision.target_digits is None:
        carried = int((precision.precision - mp.mpf(3) / 2 * lam - 10) / mp.mpf("2.35"))
        first = max(4, min(first, carried))
    return first


def pole_count_knobs(state, precision, cap_poles):
    """The chosen pole count's :class:`~virasoro_blocks.verified_build.Knob` s (global and torus one-point tables and
    large-spin towers without ``--npoles``).  While the kept poles limit the table, ``state["npoles"]`` grows by the
    measured deficit (~1.6 digits per pole), as long as that helps (an attempt closing less than a quarter of the gap
    means more poles do not).  A step that overshoots -- no better than that, or limited by the precision its larger
    degree costs -- is tried once halfway back to the count before it (a Virasoro tower at 100 digits: 30 poles give
    43 digits, 72 are precision-limited, and about 50 give 56).  The precision back-off is offered only without
    ``--target-digits``, which raises the precision instead; the build keeps the best attempt."""
    from .verified_build import Knob

    goal = precision.goal

    def kept(count):
        # a build that reports its pool (``state["pool"]``, the poles a larger count could add) keeps at most that
        pool = state.get("pool")
        return count if pool is None else min(count, pool)

    def halve_back():
        previous = state["previous"]
        if previous is None or state.get("halved") or state["npoles"] - previous[1] <= 4:
            return False
        state["halved"] = True
        halfway = (state["npoles"] + previous[1]) // 2
        if kept(halfway) == kept(state["npoles"]):
            return False        # (the same table again)
        state["npoles"] = halfway
        return True

    def more_poles(verification, work):
        previous = state["previous"]
        if previous is not None and verification.digits - previous[0] < (goal - previous[0]) / 4:
            return work if halve_back() else None
        if state["npoles"] >= cap_poles or kept(state["npoles"] + 1) == kept(state["npoles"]):
            return None
        state["previous"] = (verification.digits, state["npoles"])
        state["npoles"] = min(cap_poles, state["npoles"] + int(mp.ceil((goal - verification.digits) / mp.mpf("1.6"))) + 4)
        return work

    def fewer_poles(verification, work):
        return work if halve_back() else None

    if precision.target_digits is not None:
        return (Knob("npoles", more_poles),)
    return Knob("npoles", more_poles), Knob("precision", fewer_poles)


def _verified_global_build(adapter, request, build_coordinate, explicit_npoles, lam, spin):
    """Build a global table at the working precision ``--precision``, verified against the exact 2F1 point tables.

    A global table is an order-reduced approximation of the exact 2F1 block: its accuracy is set by the kept poles
    (about 1.6-2 digits per pole: 8 poles 13.5 digits, 16 poles 28, 32 poles 65, 76 poles 124; 2026-09-27), and the
    pole pool by ``--order`` (``order`` poles at most).  With ``--npoles`` given the table is built once and its
    verified digits recorded.  Without it the pole count is chosen: the first guess is the count the working
    precision can carry (each pole adds ~1.6 digits, each unit of degree costs ~0.75 of the precision), and it grows
    by the measured deficit while the diagnosis says the kept poles limit the table and growing them helps
    (:class:`~virasoro_blocks.verified_build.Knob`).  The working precision is raised only for ``--target-digits``
    (:func:`~virasoro_blocks.verified_build.run_verified_build`).
    """
    from dataclasses import replace as _replace

    from .clean_samples import attach_recorded_sample_block_values
    from .serialization import serialize_deriv_tables
    from .verified_build import (
        CLI_KNOBS, diagnose_monomial, precision_request_of, record_verification, run_verified_build,
    )

    precision = precision_request_of(request.policy)
    precision = _replace(precision, knobs=_replace(CLI_KNOBS, keep_all="omit it to choose the pole count"))
    digits = precision.digits
    base_order = int(request.policy.order)
    cap_poles = 8 * digits
    first = int(explicit_npoles) if explicit_npoles is not None else chosen_pole_start(precision, lam)
    state = {"npoles": first, "previous": None}

    def build(work):
        # Without --npoles or --pole-selection the order is the knob and the whole pool is kept (the pool at order
        # q offers about q genuine poles; order 77: 76): ranking a pool of ~1.1 npoles to keep npoles of them was
        # 83% of the build for no accuracy.  An explicit selection ranks a pool with a margin, as it always did.
        npoles = state["npoles"]
        if explicit_npoles is not None:
            q_order, build_npoles = base_order, npoles
        elif request.policy.pole_selection is None:
            q_order, build_npoles = max(base_order, npoles + 1), None
        else:
            q_order, build_npoles = max(base_order, npoles + max(4, npoles // 10)), npoles
        with mp.workdps(work):
            tables = {coord: build_coordinate(coord, build_npoles, q_order) for coord in request.coordinates}
            serialized = serialize_deriv_tables(tables)
            # the values at the sample nodes, from the fit's own records, written with the table
            attach_recorded_sample_block_values(serialized, tables)
        return serialized, q_order

    def verify(built, work):
        serialized, q_order = built
        with mp.workdps(work):
            point_request = _replace(request, policy=_replace(request.policy, order=q_order))
            return _global_verified_digits(adapter, point_request, serialized, lam, spin, digits)

    def diagnose(built, verification, work):
        with mp.workdps(work):
            return diagnose_monomial(built[0], verification, digits=digits, work=work, order_reduced=True)

    shown_spin = int(spin) if spin is not None and mp.mpf(spin) == int(mp.mpf(spin)) else spin
    label = (("the global table" if request.basis == "global_chiral_F" else f"spin {shown_spin}: the global table")
             + ("" if explicit_npoles is None else f" (--npoles {explicit_npoles})"))
    knobs = () if explicit_npoles is not None else pole_count_knobs(state, precision, cap_poles)
    (serialized, _q_order), verification, work = run_verified_build(
        build, verify, request=precision, what=label, order_reduced=True, diagnose=diagnose, knobs=knobs,
    )
    record_verification(serialized, verification, work, precision)
    serialized.q_order_used = _q_order
    return serialized


def _global_verified_digits(adapter, request, serialized, lam, spin, target):
    """How well a global table matches the exact point tables between its nodes (a
    :class:`~virasoro_blocks.verified_build.Verification`), or None (nothing to check).  Past its nodes an
    order-reduced global table degrades where the block is exponentially damped, so it is not checked there."""
    from dataclasses import replace as _replace

    from .table_backends import _check_points_x
    from .truncation import is_off_diagonal, kept_pairs
    from .verified_build import point_agreement, table_values, verification_from_points

    xs = getattr(serialized, "sample_points_x", None)
    if not xs:
        return None
    chiral = request.basis == "global_chiral_F"
    points = _check_points_x(xs, beyond_last=False)
    per_point = []
    for x in points:
        delta = mp.mpf(serialized.shift) + x
        point_request = (_replace(request, exchange_h=delta) if chiral
                         else _replace(request, exchange_delta=delta, exchange_spin=spin))
        point = adapter.build_point_table(point_request)
        worst = None
        for coord in serialized.coordinates:
            kept = kept_pairs(coord, lam, basis=request.basis, ell=spin, parity_projection=request.parity, parity_phase=total_parity_phase(request.parity_external, request.parity_exchanged),
                              off_diagonal=is_off_diagonal(request.center, request.center_bar))
            (got,) = table_values(serialized, coord, [delta], kept, dps=2 * mp.mp.dps)
            measured = point_agreement(got, point.evaluate(coord, delta), kept, target)
            if measured is not None:
                worst = measured if worst is None else (min(worst[0], measured[0]), min(worst[1], measured[1]))
        per_point.append(worst)
    last = max(mp.re(mp.mpc(x)) for x in xs)
    return verification_from_points(per_point, points, n_inner=len(points), last_node=last, shift=serialized.shift)


class SuperTableAdapter(TableAdapter):
    """Adapter shared by the NS and Ramond table paths."""

    def __init__(self, family: str):
        if family not in {"ns", "ramond"}:
            raise ValueError(f"not a superconformal table family: {family!r}")
        self.family = family

    def build_table(
        self,
        request: BlockRequest,
        *,
        spin: int = 0,
        backend_options=None,
        **_: Any,
    ) -> Any:
        """Build an NS/Ramond table through the package backend."""
        self.validate(request)
        request = _resolve_sphere_table_basis(request)
        if request.policy.npoles == "all":
            raise RequestValidationError(
                "superconformal tables use interpolation; npoles='all' is unsupported"
            )
        from .super_table_backends import build_super_table

        return build_super_table(
            request, spin=spin, options=backend_options
        )

    def build_point_table(self, request: BlockRequest) -> Any:
        """Build a frozen NS/Ramond point table from a canonical request (at the current precision, unverified:
        :meth:`build_verified_point_table` is the operation)."""
        self.validate(request)
        request = _resolve_sphere_table_basis(request)
        chiral = request.basis.endswith("chiral_F")
        if request.h_ext is None or len(request.h_ext) != 4:
            raise RequestValidationError("point-table requests need four external holomorphic weights")
        if (request.exchange_h if chiral else request.exchange_delta) is None:
            raise RequestValidationError(
                "point-table requests need exchange_h for chiral bases or exchange_delta for 2d bases"
            )
        if not chiral and request.exchange_spin is not None and request.exchange_spin != int(request.exchange_spin):
            raise RequestValidationError("superconformal full blocks require an integer exchanged spin")
        from .super_table_backends import build_super_point_table

        return build_super_point_table(request)


def _torus_spin(spin) -> int:
    """A torus sector's spin: a non-negative integer, validated before conversion."""
    from .requests import _integer_field

    try:
        value = _integer_field("spin", spin)
    except RequestValidationError:
        raise RequestValidationError(f"torus spins are non-negative integers, got {spin}") from None
    if value < 0:
        raise RequestValidationError(f"torus spins are non-negative integers, got {spin}")
    return value


class TorusCharacterTableAdapter(TableAdapter):
    """Adapter for exact reduced-character derivative tables."""

    family = "torus-character"

    @staticmethod
    def request(
        *,
        central_charge: Any,
        lambda_order: int,
        precision: int,
        basis: str = "torus_G",
        coordinate: str = "vvb",
        target_digits: int | None = None,
        max_precision: int | None = None,
    ) -> BlockRequest:
        request = BlockRequest(
            operation="table",
            surface="torus",
            family="torus-character",
            basis=basis,
            coordinate=coordinate,
            coordinates=(coordinate,),
            central_charge=central_charge,
            policy=NumericalPolicy(
                precision=precision,
                order=None,
                lambda_order=lambda_order,
                npoles=None,
                target_digits=target_digits,
                max_precision=max_precision,
            ),
        )
        # Validate at construction time so a future caller cannot accidentally
        # bypass the family/surface/coordinate contract.
        validate_request(request)
        return request

    def build_table(
        self,
        request: BlockRequest,
        *,
        spin: int,
        shift: Any | None = None,
        **_: Any,
    ) -> Any:
        self.validate(request)
        spin = _torus_spin(spin)
        from .modular import torus_character_table

        # at the request's precision whoever calls (build_table_request does too; a direct adapter call did not)
        def build(work):
            with mp.workdps(work):
                return torus_character_table(
                    request.central_charge,
                    int(spin),
                    int(request.policy.lambda_order),
                    shift=shift,
                )

        # exact (no fit): built and written at the working precision, and verified against the same table at more
        # digits (what it misses is rounding), recorded like every other table's
        return _self_verified_build(build, request, what=f"spin {spin}: the torus character table")

    def build_point_table(self, request: BlockRequest) -> Any:
        """The character point table of the module at (exchange_delta, exchange_spin), at the current precision
        (:func:`~virasoro_blocks.modular.torus_character_point_table`: the irreducible character on a Kac weight)."""
        self.validate(request)
        delta, spin, lam = _torus_point_inputs(request)
        from .modular import torus_character_point_table

        return torus_character_point_table(request.central_charge, delta, spin, lam)

    def build_verified_point_table(self, request: BlockRequest) -> Any:
        """The ``point-table`` operation: exact (the infinite null series of a minimal-model character cut below
        the working precision), built at the working precision and verified against the same table at more digits,
        which also measures that cut."""
        self.validate(request)
        delta, spin, _lam = _torus_point_inputs(request)

        def build(work):
            with mp.workdps(work):
                return self.build_point_table(request)

        what = f"spin {spin}: the torus character point table at Delta = {mp.nstr(mp.mpf(delta), 8)}"
        return _self_verified_build(build, request, what=what, point=True)


def _torus_point_inputs(request: BlockRequest):
    """``(Delta, spin, lambda)`` of a torus point-table request (the spin an integer >= 0, the j >= 0 sector)."""
    if request.operation != "point-table":
        raise RequestValidationError("torus point tables need operation='point-table'")
    if request.central_charge is None:
        raise RequestValidationError("torus point tables need central_charge")
    if request.exchange_delta is None:
        raise RequestValidationError("torus point tables need exchange_delta (and exchange_spin, default 0)")
    if request.policy.lambda_order is None:
        raise RequestValidationError("point-table requests need a lambda_order")
    spin = _torus_spin(0 if request.exchange_spin is None else request.exchange_spin)
    return request.exchange_delta, spin, int(request.policy.lambda_order)


class TorusOnepointTableAdapter(TableAdapter):
    """Adapter for interpolated torus one-point tables."""

    family = "torus-onepoint"

    @staticmethod
    def request(
        *,
        central_charge: Any,
        torus_onepoint_ext: Any,
        lambda_order: int,
        q_order: int,
        npoles: int | None,
        precision: int,
        torus_scalar_shift: Any | None = None,
        gap: Any | None = None,
        basis: str = "torus_onept_G",
        coordinate: str = "vvb",
        target_digits: int | None = None,
        max_precision: int | None = None,
    ) -> BlockRequest:
        request = BlockRequest(
            operation="table",
            surface="torus",
            family="torus-onepoint",
            basis=basis,
            coordinate=coordinate,
            coordinates=(coordinate,),
            central_charge=central_charge,
            torus_onepoint_ext=torus_onepoint_ext,
            torus_scalar_shift=torus_scalar_shift,
            gap=gap,
            policy=NumericalPolicy(
                precision=precision,
                order=q_order,
                lambda_order=lambda_order,
                npoles=npoles,
                target_digits=target_digits,
                max_precision=max_precision,
            ),
        )
        validate_request(request)     # (refuses gap together with torus_scalar_shift)
        return request

    def build_kernel(self, request: BlockRequest) -> Any:
        """The one-point recursion state at the request's precision and q-order (the Laurent kernel at a singular
        c); share it across spins."""
        self.validate(request)
        from .torus_onept import build_torus_onept_state

        with mp.workdps(int(request.policy.precision)):
            return build_torus_onept_state(
                int(request.policy.order),
                d_ext=request.torus_onepoint_ext,
                c=request.central_charge,
            )

    def build_reference(self, request: BlockRequest, kernel: Any) -> Any:
        """The deeper, higher-precision reference each table is verified against (share it across spins)."""
        self.validate(request)
        from .torus_onept import build_torus_onept_reference, _check_kernel_inputs

        _check_kernel_inputs(kernel, request.central_charge, request.torus_onepoint_ext, request.policy.order)
        with mp.workdps(int(request.policy.precision)):
            return build_torus_onept_reference(kernel)

    def build_table(
        self,
        request: BlockRequest,
        *,
        spin: int,
        kernel: Any | None = None,
        reference: Any | None = None,
        **_: Any,
    ) -> Any:
        """One spin's order-reduced one-point table, built at the request's working precision (as the sphere
        ``--npoles`` tables are) and verified against a deeper, higher-precision reference: ``verified_digits``
        and its fit / precision / q-truncation parts are recorded on the table, and a shortfall is reported on
        stderr with its cause.  With ``target_digits`` the working precision grows by the measured deficit
        (:func:`~virasoro_blocks.verified_build.run_verified_build`); ``kernel`` / ``reference`` serve the first
        build (they are rebuilt at a raised precision).

        Without ``npoles`` the pole count is chosen, as for the global tables: the first build keeps few remainder
        poles, since a one-point table often needs almost none (2026-10-27, lambda 5-11, precision 100-200: 0-4
        poles were within 3 digits of the best at c = 0.7, 1, 2, 10, 26, 30, 100 with d = 1, where 16 or more lose
        digits to the fit's degree), and the count grows by the measured deficit while the kept poles limit the
        table (heavy or light externals: 1-4 digits at 4 poles, 18 at 64 for c = 10, d = 6).  The chosen count is
        recorded as ``table.npoles_chosen``."""
        self.validate(request)
        spin = _torus_spin(spin)
        from dataclasses import replace as _replace

        from .table_backends import positivity_anchor
        from .torus_onept import (
            build_torus_onept_reference, build_torus_onept_state, torus_onept_delta_table, torus_onept_diagnose,
            torus_onept_verify_table, torus_onept_reference_order, torus_onept_reference_dps,
            _check_kernel_inputs, _kernel_precision,
        )
        from .verified_build import CLI_KNOBS, precision_request_of, record_verification, run_verified_build

        precision = precision_request_of(request.policy)
        precision = _replace(precision, knobs=_replace(CLI_KNOBS, keep_all="raise --order for a larger pole pool"))
        start = int(precision.precision)
        lam = int(request.policy.lambda_order)
        explicit_npoles = request.policy.npoles
        if explicit_npoles == "all":
            raise RequestValidationError("torus one-point tables are order-reduced; npoles='all' is unsupported")
        poles = {"npoles": int(explicit_npoles) if explicit_npoles is not None else 4, "previous": None}
        reports = {}
        if kernel is not None:
            _check_kernel_inputs(kernel, request.central_charge, request.torus_onepoint_ext, request.policy.order)
        if reference is not None:
            _check_kernel_inputs(reference, request.central_charge, request.torus_onepoint_ext,
                                 torus_onept_reference_order(request.policy.order), what="supplied reference")

        def build(work):
            with mp.workdps(work):
                reuse_state = kernel is not None and work == start and _kernel_precision(kernel) >= work
                state = kernel if reuse_state else build_torus_onept_state(
                    int(request.policy.order), d_ext=request.torus_onepoint_ext, c=request.central_charge)
                table = torus_onept_delta_table(
                    request.central_charge,
                    int(spin),
                    lam,
                    d_ext=request.torus_onepoint_ext,
                    q_order=int(request.policy.order),
                    npoles=poles["npoles"],
                    # --gap G anchors every spin at its unitarity bound + G (the sphere meaning: a uniform twist
                    # gap); --torus-scalar-shift X is the spin-0-only spelling. The request refuses both at once.
                    # Without either, every spin sits at its unitarity bound |j|, where the hbar = 0 Kac pole
                    # (double at spin 0) is kept in the prefactor at x = 0, as on the sphere.
                    shift=(
                        positivity_anchor(request.gap, spin, False)
                        if request.gap is not None
                        else (request.torus_scalar_shift if int(spin) == 0 else None)
                    ),
                    kernel=state,
                    # the verification below measures the precision; a starved kernel reports its digits
                    guard_imag=False,
                )
                table.npoles_chosen = None if explicit_npoles is not None else poles["npoles"]
                poles["pool"] = table.remainder_pool
                reuse_ref = (reference is not None and work == start
                             and _kernel_precision(reference) >= torus_onept_reference_dps(work))
                ref = reference if reuse_ref else build_torus_onept_reference(state)
            return table, state, ref

        def verify(built, work):
            table, state, ref = built
            with mp.workdps(work):
                report = torus_onept_verify_table(table, state, ref, int(spin), lam, target=precision.digits,
                                                  q_order=request.policy.order)
            reports[id(table)] = report
            return report["verification"]

        def diagnose(built, verification, work):
            with mp.workdps(work):
                return torus_onept_diagnose(built[0], reports[id(built[0])], verification, digits=precision.digits,
                                            work=work)

        what = f"spin {spin}: the torus one-point table" + ("" if explicit_npoles is None else f" (--npoles {explicit_npoles})")
        knobs = () if explicit_npoles is not None else pole_count_knobs(poles, precision, 8 * precision.digits)
        (table, _state, _ref), verification, work = run_verified_build(
            build, verify, request=precision, what=what, order_reduced=True, diagnose=diagnose, knobs=knobs,
        )
        record_verification(table, verification, work, precision)
        return table

    def build_verified_point_table(self, request: BlockRequest) -> Any:
        """The ``point-table`` operation: the one-point block at (exchange_delta, exchange_spin), f truncated at the
        request's order, built at its working precision and verified against a deeper, higher-precision reference
        (:func:`~virasoro_blocks.torus_onept.torus_onept_point_table`; the irreducible trace on a Kac weight of a
        singular c, as ``evaluate`` uses)."""
        self.validate(request)
        delta, spin, lam = _torus_point_inputs(request)
        if request.torus_onepoint_ext is None:
            raise RequestValidationError("torus one-point point tables need torus_onepoint_ext")
        if request.policy.order is None or int(request.policy.order) < 1:
            raise RequestValidationError("torus one-point point tables need order >= 1 (order 0 is the bare character)")
        from .torus_onept import torus_onept_point_table

        return torus_onept_point_table(
            request.central_charge, request.torus_onepoint_ext, delta, spin, lam,
            q_order=int(request.policy.order), precision=int(request.policy.precision),
            target_digits=request.policy.target_digits, max_precision=request.policy.max_precision,
        )

    def build_point_table(self, request: BlockRequest) -> Any:
        """:meth:`build_verified_point_table` (the one-point point table is always verified against its reference,
        whose deeper order also measures the truncation)."""
        return self.build_verified_point_table(request)


def _self_verified_build(build, request, *, what, point=False):
    """An exact table (no fit) built at the request's working precision and verified against the same
    construction at more digits (:func:`~virasoro_blocks.torus_onept.torus_onept_reference_dps`): what it misses
    is rounding.  Checked at 0.31, 2.3 and doubling points out to ~37 in x = variable - shift (the tables have no
    nodes), or at the table's own point (a point table).  Escalates for ``target_digits`` as every table does."""
    from dataclasses import replace as _replace

    from .torus_onept import torus_onept_reference_dps
    from .verified_build import (
        check_points_x, far_check_points_x, point_agreement, precision_request_of, record_verification,
        run_verified_build, table_values, verification_from_points, verification_margin,
    )

    precision = precision_request_of(request.policy)

    def verify(table, work):
        reference = build(torus_onept_reference_dps(work))
        inner = [mp.mpf(0)] if point else check_points_x(())
        points = inner if point else inner + far_check_points_x((), ())
        per_point = []
        for x in points:
            delta = mp.mpf(table.shift) + x
            worst = None
            for coord in table.coordinates:
                rows = table.numerators[coord]
                kept = [(m, n) for m, row in enumerate(rows) for n, poly in enumerate(row)
                        if any(coeff != 0 for coeff in poly.coeffs)]
                if not kept:
                    continue
                (got,) = table_values(table, coord, [delta], kept, dps=2 * work)
                (ref,) = table_values(reference, coord, [delta], kept, dps=torus_onept_reference_dps(work))
                measured = point_agreement(got, ref, kept, precision.digits)
                if measured is not None:
                    worst = measured if worst is None else (min(worst[0], measured[0]), min(worst[1], measured[1]))
            ceiling = float(torus_onept_reference_dps(work))
            per_point.append(None if worst is None else (min(worst[0], ceiling), min(worst[1], ceiling)))
        found = verification_from_points(per_point, points, n_inner=len(inner), shift=table.shift)
        if found is not None and point:
            found = _replace(found, where_text="at its point", span=None)
        return found

    def diagnose(table, verification, work):
        lost = float(int(work) - verification.digits)
        return _replace(verification, cause="precision", mechanism="exact", lost=lost,
                        needed_precision=int(mp.ceil(precision.digits + lost + verification_margin(precision.digits))))

    table, verification, work = run_verified_build(build, verify, request=precision, what=what, order_reduced=False,
                                                   diagnose=diagnose)
    record_verification(table, verification, work, precision)
    return table


_TABLE_ADAPTERS: Mapping[str, TableAdapter] = MappingProxyType({
    "virasoro": VirasoroTableAdapter(),
    "global": GlobalTableAdapter(),
    "ns": SuperTableAdapter("ns"),
    "ramond": SuperTableAdapter("ramond"),
    "torus-character": TorusCharacterTableAdapter(),
    "torus-onepoint": TorusOnepointTableAdapter(),
})


def table_adapter_for(family: str) -> TableAdapter:
    """Return the registered table adapter for ``family``."""
    try:
        return _TABLE_ADAPTERS[family]
    except KeyError as exc:
        available = ", ".join(_TABLE_ADAPTERS)
        raise RequestValidationError(
            f"unknown table family {family!r}; choose from {available}"
        ) from exc


def table_adapters() -> tuple[TableAdapter, ...]:
    """Return the registered adapters in stable CLI/help order."""
    return tuple(_TABLE_ADAPTERS.values())


__all__ = [
    "GlobalTableAdapter",
    "SuperTableAdapter",
    "TableAdapter",
    "TorusCharacterTableAdapter",
    "TorusOnepointTableAdapter",
    "VirasoroTableAdapter",
    "table_adapter_for",
    "table_adapters",
]
