"""Chiral (1d) global-vs-Virasoro bootstrap controls.

A cheap 1d cross-check for the large-c question: a single chiral block
``k_h(z)`` plays the role of an operator of dimension ``h``, the crossing
functional is the antisymmetric (odd-order) set of z-derivatives at z = 1/2,
and the identity is the block at ``h = 0``. At large ``c`` the Virasoro chiral
block approaches the global SL(2) block, so the two bound problems should agree.

This reuses the 2d PMP / interpolation / certification machinery by emitting a
degenerate ``(m, 0)`` derivative table (``n_max = 0``): F-pairs are the odd
``(m, 0)`` and every block carries a single z-derivative tower.
"""

from __future__ import annotations

import mpmath as mp

from virasoro_blocks.derivs import chiral_z_taylor, global_chiral_z_taylor
from virasoro_blocks.interpolation import (
    DampedRational,
    get_sample_points,
    lagrange_interpolation_polynomial,
)
from virasoro_blocks.numeric_input import literal_number, _integer_control
from virasoro_blocks.polyderivs import (
    F_z_poly_table,
    InterpolatedDeltaTable,
    interpolate_delta_table,
)
from virasoro_blocks.recursion import _resolve_chiral_norm
from virasoro_blocks.regularization import merged_delta_table
from virasoro_blocks.serialization import serialize_deriv_tables
from virasoro_blocks.singularity import recognized_singular_locus

from ..identical_scalar import _central_charge_input, _exact_identity_vector, _as_real_label, _validate_delta_gap, _normalize_npoles
from ..defaults import DEFAULT_POLE_SELECTION
from .global_control import (
    _default_global_n_sample_points,
    _global_prefactor_in_x,
    _numeric_virasoro_prefactor_in_x,
    _resolved_sample_x,
)
from ..identical_scalar import (
    IdenticalScalarGapProblem,
    IdenticalScalarGapSpec,
    IdenticalScalarSpinChannel,
)
from ..pmp_writer import build_prefactored_pmp_doc


def chiral_F_minus_pairs(lam: int) -> list:
    """1d F^- index set: odd-order z-derivatives, packed as ``(m, 0)``."""
    lam = _integer_control("lam", lam, 1)
    return [(m, 0) for m in range(1, lam + 1, 2)]


def _chiral_block_coeffs(h, *, mode, h_ext, lam, c, q_order, chiral_norm):
    """z-Taylor coefficients of the chiral block at z = 1/2 (length lam+1)."""
    h_externals = [h_ext] * 4
    if mode == "global":
        return global_chiral_z_taylor(
            h, h_externals, lam + 1,
            chiral_norm=chiral_norm, z_shift=-2 * h_ext,
        )
    return chiral_z_taylor(
        h, h_externals, lam + 1,
        c=c, q_order=q_order, chiral_norm=chiral_norm, z_shift=-2 * h_ext,
    )


def build_chiral_channel(
    *,
    mode: str,
    h_gap,
    h_ext,
    lam: int,
    c=None,
    q_order: int = 0,
    chiral_norm=None,
    n_sample_points: int | None = None,
    node_search_dps: int | None = None,
    sample_x: tuple | None = None,
) -> IdenticalScalarSpinChannel:
    """One chiral channel: operators with dimension h >= h_gap."""
    if mode not in {"global", "virasoro"}:
        raise ValueError("mode must be 'global' or 'virasoro'")
    lam = _integer_control("lam", lam)
    q_order = _integer_control("q_order", q_order)
    if n_sample_points is None and sample_x is None:
        n_sample_points = _default_global_n_sample_points(lam)
    resolved_chiral_norm = _resolve_chiral_norm(chiral_norm)
    from ..point_constraint import _crossing_context_from_inputs
    h_gap = _validate_delta_gap(h_gap)
    h_ext = _as_real_label(h_ext, name="h_ext")
    if mode == "global":
        prefactor = _global_prefactor_in_x(shift=h_gap, chiral_norm=resolved_chiral_norm)
    else:
        prefactor = _numeric_virasoro_prefactor_in_x(
            shift=h_gap, chiral_norm=resolved_chiral_norm)
    sample_x = _resolved_sample_x(
        base=prefactor.base, n_sample_points=n_sample_points,
        node_search_dps=node_search_dps, sample_x=sample_x,
    )
    sample_h = tuple(h_gap + literal_number(x) for x in sample_x)
    coeffs = [
        _chiral_block_coeffs(
            h, mode=mode, h_ext=h_ext, lam=lam,
            c=c, q_order=q_order, chiral_norm=resolved_chiral_norm,
        )
        for h in sample_h
    ]
    pref_values = [prefactor.evaluate(x) for x in sample_x]

    numerators = []
    numerator_sample_points = []
    for m in range(lam + 1):
        reduced = [coeffs[i][m] / pref_values[i] for i in range(len(sample_h))]
        poly = lagrange_interpolation_polynomial(sample_x, reduced)
        numerators.append([poly])
        numerator_sample_points.append([sample_h])

    interpolated = InterpolatedDeltaTable(
        prefactor=prefactor,
        shift=h_gap,
        sample_points=sample_h,
        numerators=numerators,
        numerator_sample_points=numerator_sample_points,
    )
    serialized = serialize_deriv_tables({"zzb": interpolated})
    # Keep blocks in the (16q/N) bootstrap-natural normalization (see the note in
    # global_control); N^Delta is absorbed into the OPE coefficients, matching the
    # 2d production and certification paths.
    prefactored = serialized
    return IdenticalScalarSpinChannel(
        ell=0, delta_min=h_gap, serialized=serialized, prefactored=prefactored,
        crossing_context=_crossing_context_from_inputs(
            basis="global_chiral_F" if mode == "global" else "chiral_F", coordinate="zzb",
            h_ext=h_ext, lam=lam, chiral_norm=resolved_chiral_norm, c=c, q_order=q_order),
    )


def _resolved_chiral_regularize(c, regularize: str) -> bool:
    if regularize == "off":
        return False
    if regularize == "on":
        return True
    if regularize == "auto":
        return recognized_singular_locus(_central_charge_input(c)) is not None
    raise ValueError("regularize must be 'auto', 'on', or 'off'")


def _chiral_symbolic_exact_table(*, h_ext, lam, c, q_order, chiral_norm,
                                 regularize: str = "auto"):
    """Symbolic-in-h chiral z-derivative table (degenerate (m,0)), with Kac poles.

    At a recursion-singular c (c = 1, minimal models such as Ising c = 1/2, ...)
    the bare table build divides by zero, so the table is built AT the singular c
    from the exact merged poles (`merged_delta_table`). `regularize="auto"`
    triggers only on a recognized singular locus; generic c builds directly.
    """
    h_ext = _as_real_label(h_ext, name="h_ext")
    h_externals = [h_ext] * 4
    z_shift = -2 * h_ext
    c = _central_charge_input(c)
    resolved_norm = _resolve_chiral_norm(chiral_norm)

    def _build(c_value):
        return F_z_poly_table(
            h_externals, lam, c=c_value, q_order=q_order,
            chiral_norm=resolved_norm, z_shift=z_shift,
        )

    if _resolved_chiral_regularize(c, regularize):
        return merged_delta_table(_build, c=c)
    return _build(c)


def build_chiral_symbolic_channel(
    *,
    h_gap,
    h_ext,
    lam: int,
    c,
    q_order: int,
    chiral_norm=None,
    npoles: int,
    n_sample_points: int | None = None,
    pole_selection: str = DEFAULT_POLE_SELECTION,
    regularize: str = "auto",
    shared_sample_points: bool = False,
) -> IdenticalScalarSpinChannel:
    """Chiral channel via the symbolic pole-carrying path (F_z_poly_table ->
    interpolate_delta_table), i.e. the 1d analog of the 2d production path."""
    lam = _integer_control("lam", lam)
    q_order = _integer_control("q_order", q_order)
    _normalize_npoles(npoles)
    if n_sample_points is not None:
        n_sample_points = _integer_control("n_sample_points", n_sample_points, 1)
    if not isinstance(shared_sample_points, bool):
        raise ValueError("shared_sample_points must be True or False")
    resolved_chiral_norm = _resolve_chiral_norm(chiral_norm)
    from ..point_constraint import _crossing_context_from_inputs
    h_gap = _validate_delta_gap(h_gap)
    exact = _chiral_symbolic_exact_table(
        h_ext=h_ext, lam=lam, c=c, q_order=q_order,
        chiral_norm=resolved_chiral_norm, regularize=regularize)
    reduced = interpolate_delta_table(
        exact, npoles, shift=h_gap,
        n_sample_points=n_sample_points, pole_selection=pole_selection,
        shared_sample_points=shared_sample_points,
    )
    serialized = serialize_deriv_tables({"zzb": reduced})
    # Bootstrap-natural normalization; N^Delta absorbed into OPE coefficients
    # (see build_chiral_interp_channel / global_control for the rationale).
    prefactored = serialized
    return IdenticalScalarSpinChannel(
        ell=0, delta_min=h_gap, serialized=serialized, prefactored=prefactored,
        crossing_context=_crossing_context_from_inputs(
            basis="chiral_F", coordinate="zzb", h_ext=h_ext, lam=lam,
            chiral_norm=resolved_chiral_norm, c=c, q_order=q_order),
    )


def _chiral_identity_vector(*, mode, h_ext, lam, c, q_order, chiral_norm, F_pairs,
                            regularize: str = "auto"):
    """Odd-order z-derivatives of the identity (h=0) chiral block at z=1/2."""
    if mode == "virasoro_symbolic":
        # Same vacuum block as the numeric path, but via the exact symbolic
        # table with the removable vacuum pole stripped before evaluating at 0.
        exact = _chiral_symbolic_exact_table(
            h_ext=h_ext, lam=lam, c=c, q_order=q_order, chiral_norm=chiral_norm,
            regularize=regularize)
        return _exact_identity_vector(exact, F_pairs)
    coeffs = _chiral_block_coeffs(
        mp.mpf(0), mode=mode, h_ext=_as_real_label(h_ext, name="h_ext"), lam=lam,
        c=c, q_order=q_order, chiral_norm=_resolve_chiral_norm(chiral_norm),
    )
    return tuple(coeffs[m] for (m, n) in F_pairs)


def build_chiral_control_problem(
    *,
    mode: str,
    c,
    h_ext,
    h_gap,
    lam: int,
    q_order: int = 12,
    dps: int = 80,
    chiral_norm=None,
    n_sample_points: int | None = None,
    node_search_dps: int | None = None,
    npoles: int = 6,
    pole_selection: str = DEFAULT_POLE_SELECTION,
    explicit_sample_points: bool = True,
    regularize: str = "auto",
    shared_sample_points: bool = False,
) -> IdenticalScalarGapProblem:
    """Assemble a 1d chiral gap problem.

    mode='global'            : 2d global SL(2) block, pole-free Lagrange.
    mode='virasoro'          : direct numeric Virasoro block, pole-free Lagrange.
    mode='virasoro_symbolic' : symbolic-in-h Virasoro block with Kac poles, fed
                               through interpolate_delta_table (the npoles path).

    `regularize` ('auto'/'on'/'off') controls the singular-c merged-pole evaluation
    for the 'virasoro_symbolic' table (e.g. Ising c = 1/2); it is a no-op for the
    pole-free 'global'/'virasoro' modes, whose blocks build directly.

    `shared_sample_points` defaults to False (per-entry): every (m, 0) entry
    keeps its own degree-matched node set, tying its degree to npoles so the
    large-Delta tail power stays correct. Setting it True pads every entry onto
    the single max-degree node set (more accurate in-window, Runge-prone in the
    far tail). It only affects 'virasoro_symbolic' (the per-entry interpolation
    path); the 'global' and 'virasoro' pole-free modes already share one node set.
    """
    if mode not in {"global", "virasoro", "virasoro_symbolic"}:
        raise ValueError("mode must be 'global', 'virasoro', or 'virasoro_symbolic'")
    lam = _integer_control("lam", lam, 1)
    q_order = _integer_control("q_order", q_order, 1)
    dps = _integer_control("dps", dps, 10)
    _normalize_npoles(npoles)
    if n_sample_points is not None:
        n_sample_points = _integer_control("n_sample_points", n_sample_points, 1)
    if node_search_dps is not None:
        node_search_dps = _integer_control("node_search_dps", node_search_dps, 1)
        if mode == "virasoro_symbolic":
            raise ValueError("node_search_dps does not apply to the symbolic chiral interpolator")
    for name, value in (("explicit_sample_points", explicit_sample_points),
                        ("shared_sample_points", shared_sample_points)):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be True or False")
    if regularize not in {"auto", "on", "off"}:
        raise ValueError("regularize must be 'auto', 'on', or 'off'")
    with mp.workdps(dps):
        c = _central_charge_input(c)
        h_ext_input = literal_number(h_ext)
        h_ext = _as_real_label(h_ext_input, name="h_ext")
        h_gap = _validate_delta_gap(h_gap)
        F_pairs = tuple(chiral_F_minus_pairs(lam))
        if n_sample_points is None and mode != "virasoro_symbolic":
            n_sample_points = _default_global_n_sample_points(lam)
        if mode == "virasoro_symbolic":
            channel = build_chiral_symbolic_channel(
                h_gap=h_gap, h_ext=h_ext, lam=lam, c=c, q_order=q_order,
                chiral_norm=chiral_norm, npoles=npoles,
                n_sample_points=n_sample_points, pole_selection=pole_selection,
                regularize=regularize, shared_sample_points=shared_sample_points,
            )
        else:
            channel = build_chiral_channel(
                mode=mode, h_gap=h_gap, h_ext=h_ext, lam=lam,
                c=c, q_order=q_order, chiral_norm=chiral_norm,
                n_sample_points=n_sample_points, node_search_dps=node_search_dps,
            )
        identity_with_measure = _chiral_identity_vector(
            mode=mode, h_ext=h_ext, lam=lam, c=c, q_order=q_order,
            chiral_norm=chiral_norm, F_pairs=((0, 0),) + F_pairs, regularize=regularize,
        )
        vacuum_measure, identity_vector = identity_with_measure[0], tuple(identity_with_measure[1:])
        objective = [mp.mpf(0)] * len(F_pairs)
        # The virasoro_symbolic table at c < 1 (e.g. Ising c = 1/2) carries many
        # real positive Kac poles above the gap; lowering them re-interpolates the
        # cleared block so its Bohr-Sommerfeld nodes / sampleScalings are emitted
        # (without which pmp2sdp sees the raw high-degree coefficients and SDPB
        # diverges). The pole-free modes have no such poles, so leave it off.
        positive_pole_npoles = npoles if mode == "virasoro_symbolic" else None
        pmp_build = build_prefactored_pmp_doc(
            [channel.prefactored], list(F_pairs),
            objective=objective, normalization=identity_vector,
            explicit_sample_points=explicit_sample_points,
            positive_pole_npoles=positive_pole_npoles,
        )
        spec = IdenticalScalarGapSpec(
            c=c, h_ext=h_ext_input, delta_gap=h_gap, lam=lam, spin_list=(0,),
            q_order=q_order, dps=dps, n_sample_points=n_sample_points,
            npoles=npoles, pole_selection=pole_selection,
            regularize=regularize,
            chiral_norm=chiral_norm, explicit_sample_points=explicit_sample_points,
            num_threads=1,
        )
        return IdenticalScalarGapProblem(
            spec=spec,
            resolved_method=f"chiral_{mode}_control",
            used_regularize=(mode != "global" and _resolved_chiral_regularize(
                c, regularize if mode == "virasoro_symbolic" else "auto")),
            prefactor_exponent=h_ext,
            F_pairs=F_pairs,
            identity_vector=identity_vector,
            spin_channels=(channel,),
            pmp_build=pmp_build,
            objective_vector=tuple(objective),
            normalization_vector=tuple(identity_vector),
            pmp_metadata={"self_dual_identity_measure": mp.nstr(vacuum_measure, mp.mp.dps + 5)},
        )
