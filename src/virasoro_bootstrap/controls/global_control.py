"""Global 2d scalar-block controls on the identical-scalar bootstrap surface.

These helpers reuse the identical-scalar normalization / PMP assembly from
`virasoro_bootstrap.identical_scalar`, but replace the positivity blocks with
numerically interpolated 2d global scalar blocks in the same normalization
convention. This is intended as a debugging control: at very large `c`, the
Virasoro positivity blocks should approach this global path.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from virasoro_blocks._conventions import pool_initializer_kwargs

import mpmath as mp
from virasoro_blocks.parallel import usable_workers
from virasoro_blocks.numeric_input import literal_number, _integer_control

from virasoro_blocks.coordinates import q_symmetric, z_symmetric
from virasoro_blocks.derivs import (
    G_xt_table,
    G_ws_table,
    G_yyb_table,
    G_zzb_table,
    global_G_ws_table,
    global_G_xt_table,
    global_G_yyb_table,
    global_G_zzb_table,
)
from virasoro_blocks.global_blocks import (
    global_block_exp_base,
    global_scalar_G_interpolated_table,
)
from virasoro_blocks.interpolation import (
    DampedRational,
    get_sample_points,
    lagrange_interpolation_polynomial,
)
from virasoro_blocks.polyderivs import InterpolatedDeltaTable
from virasoro_blocks.recursion import _resolve_chiral_norm
from virasoro_blocks.serialization import serialize_deriv_tables

from ..crossing import F_minus_pairs
from ..identical_scalar import (
    IdenticalScalarGapProblem,
    IdenticalScalarGapSpec,
    IdenticalScalarSpinChannel,
    _as_real_label,
    _canonical_non_scalar_delta_min_overrides,
    _canonicalize_real_output,
    _numeric_identity_vector,
    _normalize_npoles,
    _resolved_delta_min_by_ell,
    _resolved_regularize,
    _validate_gap_spec_common,
)
from ..pmp_writer import build_prefactored_pmp_doc


_GLOBAL_CONTROL_BUILDERS = {
    "zzb": global_G_zzb_table,
    "yyb": global_G_yyb_table,
    "xt": global_G_xt_table,
    "ws": global_G_ws_table,
}
_NUMERIC_CONTROL_BUILDERS = {
    "zzb": G_zzb_table,
    "yyb": G_yyb_table,
    "xt": G_xt_table,
    "ws": G_ws_table,
}


def _default_global_n_sample_points(lam: int) -> int:
    """Conservative degree budget for the global-control interpolants."""
    return max(9, 4 * lam + 1)


def _default_node_search_dps() -> int:
    """Match the interpolation layer's default node-search precision policy."""
    return max(15, mp.mp.dps // 2)


def _global_prefactor_in_x(*, shift, chiral_norm) -> DampedRational:
    # The global control must use the same large-Delta decay base as the
    # verified global block basis: 4*rho(1/2)/N, not z(1/2)/N.
    base = global_block_exp_base(chiral_norm)
    return DampedRational(constant=1, poles=(), base=base).shift_x(shift)


def _numeric_virasoro_prefactor_in_x(*, shift, chiral_norm) -> DampedRational:
    base = mp.mpf(16) * mp.mpf(q_symmetric()) / mp.mpf(chiral_norm)
    return DampedRational(constant=1, poles=(), base=base).shift_x(shift)


def _global_vacuum_table(coordinate, h_ext, lam, chiral_norm):
    """Global 2d vacuum (Delta = ell = 0) derivative grid at the self-dual point.

    This is the c-independent global identity, the global-block counterpart of
    the Virasoro vacuum used by :func:`_numeric_identity_vector`. The global
    block is regular at the vacuum, so it is evaluated directly at ``Delta = 0``
    (no eps offset or removable-pole stripping). Parity folding for the
    ``xt``/``ws`` coordinates is handled inside the builder, matching the
    Virasoro path.
    """
    builder = _GLOBAL_CONTROL_BUILDERS[coordinate]
    resolved_chiral_norm = _resolve_chiral_norm(chiral_norm)
    h_ext = _as_real_label(h_ext, name="h_ext")
    return builder(
        mp.mpf(0),
        0,
        [h_ext] * 4,
        lam,
        lam,
        chiral_norm=resolved_chiral_norm,
        z_shift=-2 * h_ext,
    )


def global_identity_vector(coordinate, h_ext, lam, F_pairs, chiral_norm):
    """c-independent global identity crossing vector on the F^- pairs."""
    table = _global_vacuum_table(coordinate, h_ext, lam, chiral_norm)
    return tuple(_canonicalize_real_output(table[m][n]) for (m, n) in F_pairs)


def global_identity_measure_value(coordinate, h_ext, lam, chiral_norm):
    """Self-dual global vacuum block value I[1] = G_vac(1/2, 1/2) (global)."""
    table = _global_vacuum_table(coordinate, h_ext, lam, chiral_norm)
    return _canonicalize_real_output(table[0][0])


def _resolved_sample_x(*, base, n_sample_points, node_search_dps, sample_x):
    if n_sample_points is not None:
        n_sample_points = _integer_control("n_sample_points", n_sample_points, 1)
    if node_search_dps is not None:
        node_search_dps = _integer_control("node_search_dps", node_search_dps, 1)
    if sample_x is not None:
        sample_x = tuple(_as_real_label(x, name="sample_x coordinate") for x in sample_x)
        if not sample_x or any(x < 0 for x in sample_x):
            raise ValueError("sample_x must contain nonnegative coordinates")
        if len(set(sample_x)) != len(sample_x):
            raise ValueError("sample_x coordinates must be distinct")
        if n_sample_points is not None and n_sample_points != len(sample_x):
            raise ValueError("n_sample_points must match the number of sample_x coordinates")
        return sample_x
    sample_work_dps = (
        node_search_dps if node_search_dps is not None else _default_node_search_dps()
    )
    return tuple(
        get_sample_points(
            DampedRational(constant=1, poles=(), base=base),
            n_sample_points,
            work_dps=sample_work_dps,
        )
    )


def _sampled_spin_channel(
    *,
    coordinate: str,
    ell: int,
    delta_min,
    lam: int,
    prefactor: DampedRational,
    sample_x: tuple,
    numeric_tables,
    chiral_norm,
    crossing_context=None,
) -> IdenticalScalarSpinChannel:
    sample_delta = tuple(delta_min + literal_number(x) for x in sample_x)
    pref_values = [prefactor.evaluate(x) for x in sample_x]

    numerators = []
    numerator_sample_points = []
    for m in range(lam + 1):
        row = []
        sample_row = []
        for n in range(lam + 1):
            reduced_values = [
                numeric_tables[idx][m][n] / pref_values[idx]
                for idx in range(len(sample_delta))
            ]
            row.append(
                lagrange_interpolation_polynomial(
                    sample_x,
                    reduced_values,
                )
            )
            sample_row.append(sample_delta)
        numerators.append(row)
        numerator_sample_points.append(sample_row)

    interpolated = InterpolatedDeltaTable(
        prefactor=prefactor,
        shift=delta_min,
        sample_points=sample_delta,
        numerators=numerators,
        numerator_sample_points=numerator_sample_points,
    )
    serialized = serialize_deriv_tables({coordinate: interpolated})
    # Store the direct block values at the writer nodes.  Moment bounds in the
    # F/G_self basis can then feed ratio sample values straight to the direct SDP
    # writer instead of re-evaluating a monomial interpolant for the ratio.
    serialized.sample_block_values = {coordinate: numeric_tables}
    # Keep blocks in the (4 rho / N) bootstrap-natural normalization rather than
    # reapplying N^Delta: the physical N^{-Delta} block normalization is the whole
    # point of choosing N (it places the damped-rational base at the VMFT-physical
    # decay so nodes and measure agree). N^Delta is a per-operator positive rescale
    # that leaves the gap/feasibility verdict unchanged and only shifts OPE-coeff
    # units, so it is absorbed into the OPE coefficients. This matches the 2d
    # production path (manifest/identical_scalar) and the certification path, which
    # already use prefactored = serialized.
    prefactored = serialized
    return IdenticalScalarSpinChannel(
        ell=ell,
        delta_min=delta_min,
        serialized=serialized,
        prefactored=prefactored,
        crossing_context=crossing_context,
    )


def build_global_scalar_spin_channel(
    *,
    coordinate: str = "zzb",
    ell: int,
    delta_min,
    h_ext,
    lam: int,
    chiral_norm=None,
    n_sample_points: int | None = None,
) -> IdenticalScalarSpinChannel:
    """Build one identical-scalar spin channel from global 2d blocks.

    This matches the Virasoro bootstrap path's derivative basis and physical
    normalization conventions (selected coordinate `zzb`/`yyb`/`xt`/`ws`, `z_shift = -2 h_ext`, no `N^Delta`
    rescaling). It is built from `global_scalar_G_interpolated_table` (the SL(2) poles kept, selected by
    `contribution_optimal`), whose own nodes the SDP data are sampled at; ``n_sample_points`` sets its kept-pole
    count (default 8).
    """
    resolved_chiral_norm = _resolve_chiral_norm(chiral_norm)
    from ..point_constraint import _crossing_context_from_inputs
    lam = _integer_control("lam", lam)
    ell = _integer_control("ell", ell)
    delta_min = _as_real_label(delta_min, name="delta_min")
    h_ext = _as_real_label(h_ext, name="h_ext")
    h_externals = [h_ext] * 4
    kept_entries = tuple(
        (m, n) for m in range(lam + 1) for n in range(lam + 1)
    )
    # Use the first-class global_scalar_G interpolator, which keeps the analytic
    # SL(2) Delta poles and assigns per-entry degrees from the large-Delta power.
    # The older control path used a pole-free exponential interpolant; that is
    # not comparable to the Virasoro interpolation surface at large c.
    if n_sample_points is None:
        npoles = 8
    else:
        npoles = _integer_control("n_sample_points", n_sample_points)
    table = global_scalar_G_interpolated_table(
        coordinate,
        ell,
        h_externals,
        lam,
        lam,
        chiral_norm=resolved_chiral_norm,
        z_shift=-2 * h_ext,
        q_order=30,
        npoles=npoles,
        pole_selection="contribution_optimal",
        shift=delta_min,
        num_threads=1,
        shared_sample_points=True,
        kept_entries=kept_entries,
    )
    serialized = serialize_deriv_tables({coordinate: table})
    sample_x = tuple(
        literal_number(delta) - delta_min
        for delta in table.sample_points
    )
    builder = _GLOBAL_CONTROL_BUILDERS[coordinate]
    serialized.sample_block_values = {
        coordinate: [
            builder(
                delta_min + literal_number(x),
                ell,
                h_externals,
                lam,
                lam,
                chiral_norm=resolved_chiral_norm,
                z_shift=-2 * h_ext,
            )
            for x in sample_x
        ]
    }
    return IdenticalScalarSpinChannel(
        ell=ell,
        delta_min=delta_min,
        serialized=serialized,
        prefactored=serialized,
        crossing_context=_crossing_context_from_inputs(
            basis="global_scalar_G", coordinate=coordinate, h_ext=h_ext, lam=lam,
            chiral_norm=resolved_chiral_norm),
    )


def build_numeric_virasoro_scalar_spin_channel(
    *,
    coordinate: str = "zzb",
    c,
    q_order: int,
    ell: int,
    delta_min,
    h_ext,
    lam: int,
    chiral_norm=None,
    n_sample_points: int | None = None,
    node_search_dps: int | None = None,
    sample_x: tuple | None = None,
) -> IdenticalScalarSpinChannel:
    """Build one identical-scalar spin channel from direct numeric Virasoro tables.

    This bypasses the symbolic-in-Delta Virasoro table path and interpolates
    direct numeric table evaluations on the physical bootstrap surface.
    It uses a pole-free exponential prefactor (unlike the global control, which
    keeps its SL(2) poles) rather than the symbolic interpolation path's kept
    Kac-pole prefactor, so matching
    `n_sample_points` alone does not make this control share the same rational
    positivity surface as the symbolic builder.
    """
    lam = _integer_control("lam", lam)
    ell = _integer_control("ell", ell)
    q_order = _integer_control("q_order", q_order)
    if n_sample_points is None and sample_x is None:
        n_sample_points = _default_global_n_sample_points(lam)

    resolved_chiral_norm = _resolve_chiral_norm(chiral_norm)
    from ..point_constraint import _crossing_context_from_inputs
    c = _as_real_label(c, name="c")
    delta_min = _as_real_label(delta_min, name="delta_min")
    h_ext = _as_real_label(h_ext, name="h_ext")
    h_externals = [h_ext] * 4
    try:
        builder = _NUMERIC_CONTROL_BUILDERS[coordinate]
    except KeyError as exc:
        raise ValueError(
            "numeric Virasoro identical-scalar control coordinate must be one of "
            "{'zzb', 'yyb', 'xt', 'ws'}"
        ) from exc
    prefactor = _numeric_virasoro_prefactor_in_x(
        shift=delta_min,
        chiral_norm=resolved_chiral_norm,
    )
    sample_x = _resolved_sample_x(
        base=prefactor.base,
        n_sample_points=n_sample_points,
        node_search_dps=node_search_dps,
        sample_x=sample_x,
    )
    numeric_tables = [
        builder(
            delta_min + literal_number(x),
            ell,
            h_externals,
            lam,
            lam,
            c=c,
            q_order=q_order,
            chiral_norm=resolved_chiral_norm,
            z_shift=-2 * h_ext,
        )
        for x in sample_x
    ]
    return _sampled_spin_channel(
        coordinate=coordinate,
        ell=ell,
        delta_min=delta_min,
        lam=lam,
        prefactor=prefactor,
        sample_x=sample_x,
        numeric_tables=numeric_tables,
        chiral_norm=resolved_chiral_norm,
        crossing_context=_crossing_context_from_inputs(
            basis="scalar_G", coordinate=coordinate, h_ext=h_ext, lam=lam,
            chiral_norm=resolved_chiral_norm, c=c, q_order=q_order),
    )


def _build_global_spin_channel_task(args):
    (
        dps,
        coordinate,
        ell,
        delta_min,
        h_ext,
        lam,
        chiral_norm,
        n_sample_points,
    ) = args
    mp.mp.dps = dps
    return build_global_scalar_spin_channel(
        coordinate=coordinate,
        ell=ell,
        delta_min=delta_min,
        h_ext=h_ext,
        lam=lam,
        chiral_norm=chiral_norm,
        n_sample_points=n_sample_points,
    )


def _build_numeric_virasoro_spin_channel_task(args):
    (
        dps,
        coordinate,
        c,
        q_order,
        ell,
        delta_min,
        h_ext,
        lam,
        chiral_norm,
        n_sample_points,
        node_search_dps,
        sample_x,
    ) = args
    mp.mp.dps = dps
    return build_numeric_virasoro_scalar_spin_channel(
        coordinate=coordinate,
        c=c,
        q_order=q_order,
        ell=ell,
        delta_min=delta_min,
        h_ext=h_ext,
        lam=lam,
        chiral_norm=chiral_norm,
        n_sample_points=n_sample_points,
        node_search_dps=node_search_dps,
        sample_x=sample_x,
    )


def _reject_unsupported_control_options(spec, where: str) -> None:
    """The control problems build channels, the identity and the prefactor only: spec options they cannot honor
    fail closed instead of being dropped (a point constraint or a large-spin tail silently ignored changes the
    bound being computed)."""
    unsupported = []
    if tuple(spec.point_constraints or ()):
        unsupported.append("point_constraints")
    if spec.large_spin_lmin is not None or tuple(spec.large_spin_twists or ()):
        unsupported.append("large_spin_lmin / large_spin_twists")
    if spec.shared_sample_points:
        unsupported.append("shared_sample_points")
    if unsupported:
        raise ValueError(f"{where} does not support {', '.join(unsupported)}; build the Virasoro problem instead")


def build_global_identical_scalar_control_problem(
    spec: IdenticalScalarGapSpec,
    *,
    n_sample_points: int | None = None,
    node_search_dps: int | None = None,
    global_identity: bool = False,
) -> IdenticalScalarGapProblem:
    """Build a global-block control problem on the identical-scalar PMP surface.

    The positivity blocks are interpolated from the 2d global scalar blocks.

    By default the normalization vector is reused from the Virasoro identity
    channel at the same `(c, h_ext)` as `spec`; that keeps the control on the
    exact same bootstrap surface and lets very-large-`c` Virasoro runs be
    compared against the global control directly.

    Set ``global_identity=True`` to instead use the c-independent global vacuum
    block as the identity/normalization vector. This makes the crossing equation
    fully global (vacuum and operator blocks from the same family) -- the
    faithful setup for reproducing the global-block moment bounds of
    arXiv:2603.18140.
    """
    if not isinstance(global_identity, bool):
        raise ValueError("global_identity must be True or False")
    spin_list, coordinate, c, h_ext, prefactor_exponent = _validate_gap_spec_common(spec)
    delta_min_by_ell = _resolved_delta_min_by_ell(spec, spin_list)
    _reject_unsupported_control_options(spec, "build_global_identical_scalar_control_problem")
    if node_search_dps is not None:
        raise ValueError(
            "build_global_identical_scalar_control_problem places its nodes with the global interpolator; "
            "node_search_dps does not apply"
        )
    if n_sample_points is None:
        n_sample_points = spec.n_sample_points
    if n_sample_points is not None:
        n_sample_points = _integer_control("n_sample_points", n_sample_points)

    with mp.workdps(spec.dps):
        F_pairs = tuple(F_minus_pairs(spec.lam, coordinate=coordinate))
        used_regularize = _resolved_regularize(c, spec.regularize)
        resolved_chiral_norm = _resolve_chiral_norm(spec.chiral_norm)
        if n_sample_points is None:
            global_npoles = _normalize_npoles(spec.npoles)
            if global_npoles is None:
                global_npoles = _default_global_n_sample_points(spec.lam)
        else:
            global_npoles = n_sample_points
        if global_identity:
            identity_vector = global_identity_vector(
                coordinate,
                h_ext,
                spec.lam,
                F_pairs,
                spec.chiral_norm,
            )
        else:
            identity_vector = _numeric_identity_vector(
                c,
                h_ext,
                spec.lam,
                spec.q_order,
                F_pairs,
                coordinate=coordinate,
                chiral_norm=spec.chiral_norm,
            )
        jobs = [
            (
                spec.dps,
                coordinate,
                ell,
                delta_min_by_ell[ell],
                h_ext,
                spec.lam,
                spec.chiral_norm,
                global_npoles,
            )
            for ell in spin_list
        ]
        if usable_workers(spec.num_threads) == 1 or len(jobs) <= 1:
            spin_channels = tuple(
                _build_global_spin_channel_task(job) for job in jobs
            )
        else:
            with ProcessPoolExecutor(
                max_workers=min(spec.num_threads, len(jobs)), **pool_initializer_kwargs()
            ) as pool:
                spin_channels = tuple(pool.map(_build_global_spin_channel_task, jobs))
        objective = [mp.mpf(0)] * len(F_pairs)
        pmp_build = build_prefactored_pmp_doc(
            [channel.prefactored for channel in spin_channels],
            list(F_pairs),
            coordinate=coordinate,
            objective=objective,
            normalization=identity_vector,
            positive_pole_npoles=spec.positive_pole_npoles,
            positive_pole_selection=spec.positive_pole_selection,
            explicit_sample_points=spec.explicit_sample_points,
        )

        canonical_spec = IdenticalScalarGapSpec(
            c=c,
            h_ext=spec.h_ext,
            delta_gap=delta_min_by_ell[0],
            delta_min_by_spin=_canonical_non_scalar_delta_min_overrides(delta_min_by_ell),
            lam=spec.lam,
            coordinate=coordinate,
            spin_list=spin_list,
            q_order=spec.q_order,
            dps=spec.dps,
            npoles=global_npoles,
            pole_selection=spec.pole_selection,
            positive_pole_npoles=spec.positive_pole_npoles,
            positive_pole_selection=spec.positive_pole_selection,
            regularize=spec.regularize,
            prefactor_exponent=prefactor_exponent,
            chiral_norm=spec.chiral_norm,
            explicit_sample_points=spec.explicit_sample_points,
            num_threads=spec.num_threads,
        )
        pmp_metadata = None
        if global_identity:
            # Carry the global self-dual vacuum measure I[1] so the moment path
            # uses the global G_vac(1/2,1/2) for identity-including moments rather
            # than recomputing the Virasoro vacuum from spec.c.
            pmp_metadata = {
                "self_dual_identity_measure": mp.nstr(
                    global_identity_measure_value(
                        coordinate, h_ext, spec.lam, spec.chiral_norm
                    ),
                    mp.mp.dps + 5,
                )
            }
        return IdenticalScalarGapProblem(
            spec=canonical_spec,
            resolved_method="global_control",
            used_regularize=used_regularize,
            prefactor_exponent=prefactor_exponent,
            F_pairs=F_pairs,
            identity_vector=identity_vector,
            spin_channels=spin_channels,
            pmp_build=pmp_build,
            pmp_metadata=pmp_metadata,
            objective_vector=tuple(objective),
            normalization_vector=tuple(identity_vector),
        )


def build_numeric_virasoro_identical_scalar_control_problem(
    spec: IdenticalScalarGapSpec,
    *,
    n_sample_points: int | None = None,
    node_search_dps: int | None = None,
) -> IdenticalScalarGapProblem:
    """Build a numeric-Virasoro control problem on the identical-scalar PMP surface.

    The positivity blocks are interpolated from direct numeric `G_zzb_table`
    evaluations rather than the symbolic-in-Delta Virasoro tables.
    """
    spin_list, coordinate, c, h_ext, prefactor_exponent = _validate_gap_spec_common(spec)
    delta_min_by_ell = _resolved_delta_min_by_ell(spec, spin_list)
    _reject_unsupported_control_options(spec, "build_numeric_virasoro_identical_scalar_control_problem")
    if n_sample_points is None:
        n_sample_points = spec.n_sample_points

    with mp.workdps(spec.dps):
        F_pairs = tuple(F_minus_pairs(spec.lam, coordinate=coordinate))
        used_regularize = _resolved_regularize(c, spec.regularize)
        resolved_chiral_norm = _resolve_chiral_norm(spec.chiral_norm)
        if n_sample_points is None:
            n_sample_points = _default_global_n_sample_points(spec.lam)
        shared_sample_x = _resolved_sample_x(
            base=mp.mpf(16) * mp.mpf(q_symmetric()) / mp.mpf(resolved_chiral_norm),
            n_sample_points=n_sample_points,
            node_search_dps=node_search_dps,
            sample_x=None,
        )
        identity_vector = _numeric_identity_vector(
            c,
            h_ext,
            spec.lam,
            spec.q_order,
            F_pairs,
            coordinate=coordinate,
            chiral_norm=spec.chiral_norm,
        )
        jobs = [
            (
                spec.dps,
                coordinate,
                c,
                spec.q_order,
                ell,
                delta_min_by_ell[ell],
                h_ext,
                spec.lam,
                spec.chiral_norm,
                n_sample_points,
                node_search_dps,
                shared_sample_x,
            )
            for ell in spin_list
        ]
        if usable_workers(spec.num_threads) == 1 or len(jobs) <= 1:
            spin_channels = tuple(
                _build_numeric_virasoro_spin_channel_task(job) for job in jobs
            )
        else:
            with ProcessPoolExecutor(
                max_workers=min(spec.num_threads, len(jobs)), **pool_initializer_kwargs()
            ) as pool:
                spin_channels = tuple(
                    pool.map(_build_numeric_virasoro_spin_channel_task, jobs)
                )
        objective = [mp.mpf(0)] * len(F_pairs)
        pmp_build = build_prefactored_pmp_doc(
            [channel.prefactored for channel in spin_channels],
            list(F_pairs),
            coordinate=coordinate,
            objective=objective,
            normalization=identity_vector,
            positive_pole_npoles=spec.positive_pole_npoles,
            positive_pole_selection=spec.positive_pole_selection,
            explicit_sample_points=spec.explicit_sample_points,
        )

        canonical_spec = IdenticalScalarGapSpec(
            c=c,
            h_ext=spec.h_ext,
            delta_gap=delta_min_by_ell[0],
            delta_min_by_spin=_canonical_non_scalar_delta_min_overrides(delta_min_by_ell),
            lam=spec.lam,
            coordinate=coordinate,
            spin_list=spin_list,
            q_order=spec.q_order,
            dps=spec.dps,
            npoles=spec.npoles,
            pole_selection=spec.pole_selection,
            positive_pole_npoles=spec.positive_pole_npoles,
            positive_pole_selection=spec.positive_pole_selection,
            regularize=spec.regularize,
            prefactor_exponent=prefactor_exponent,
            chiral_norm=spec.chiral_norm,
            explicit_sample_points=spec.explicit_sample_points,
            num_threads=spec.num_threads,
            n_sample_points=n_sample_points,
        )
        return IdenticalScalarGapProblem(
            spec=canonical_spec,
            resolved_method="numeric_virasoro_control",
            used_regularize=used_regularize,
            prefactor_exponent=prefactor_exponent,
            F_pairs=F_pairs,
            identity_vector=identity_vector,
            spin_channels=spin_channels,
            pmp_build=pmp_build,
            objective_vector=tuple(objective),
            normalization_vector=tuple(identity_vector),
        )
