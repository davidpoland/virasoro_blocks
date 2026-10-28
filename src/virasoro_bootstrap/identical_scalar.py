"""Minimal identical-scalar bootstrap gap problems on top of the bootstrap layer.

This module assembles the smallest production-facing single-correlator problem
we can currently support:

- scalar external operators with identical chiral weights `h_ext`,
- the odd crossing sector `F^-` on the selected crossing-odd coordinate
    (`zzb` by default, optionally `yyb`),
- one positivity block per exchanged even spin,
- bootstrap-side lower edges `Delta_min(ell)`, defaulting to `Delta_gap` in the
    scalar channel and the unitarity bound `Delta = ell` for `ell >= 2`, with
    optional per-spin overrides,
- npoles-controlled interpolation at every c, plus bootstrap-side
    positive-pole lowering where a block has real poles at x > 0,
- prepared-build reuse of the exact tables, together with automatic reuse of
    any serialized/prefactored spin channel whose requested lower edge is
    unchanged.

The resulting object can be written as a pmp2sdp JSON document and, when the
SDPB Docker image is available, converted and solved via Docker.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from virasoro_blocks._conventions import pool_initializer_kwargs, scalar_G_spin0_convention
from dataclasses import dataclass, field as _dataclass_field, fields as _dataclass_fields, replace as _dataclass_replace
import json
import math
import numbers
import os
from pathlib import Path
import shutil
import subprocess
from typing import Mapping, Sequence
import warnings

import mpmath as mp
from mpmath.libmp import from_rational, round_nearest
from virasoro_blocks.parallel import usable_workers
from virasoro_blocks.precision import taylor_coefficients
from virasoro_blocks.requests import _integer_field

from virasoro_blocks.coordinates import y_symmetric, z_of_y, z_symmetric
from virasoro_blocks.derivs import (
    G_ws_table_many,
    G_xt_table_many,
    G_yyb_table_many,
    G_zzb_table_many,
    _fold_parity,
)
from virasoro_blocks.polyderivs import (
    G_ws_poly_table,
    G_xt_poly_table,
    G_yyb_poly_table,
    G_zzb_poly_table,
    _NumericBlockSampler,
    build_chiral_cache,
    build_table_from_cache,
    interpolate_delta_table,
    interpolate_delta_tables,
)
from virasoro_blocks.polyderivs import (
    _strip_removable_poles_from_exact_table as strip_removable_poles_from_exact_table,
)
from virasoro_blocks.clean_samples import (
    _canonicalize_real_output,
    _real_tol,
    attach_clean_sample_block_values as _attach_clean_sample_block_values,
)
from virasoro_blocks.regularization import (
    _fixed_h_block_evaluator_regularized,
)
from virasoro_blocks.serialization import SerializedDerivTable, serialize_deriv_tables
from virasoro_blocks.singularity import recognized_singular_locus
from virasoro_blocks.truncation import kept_pairs

from .crossing import F_minus_pairs
from .hpc.runner import DOCKER_SDPB_IMAGE as _DOCKER_SDPB_IMAGE
from .hpc.runner import SdpbRunner, resolve_sdpb_runner, stage_initial_checkpoint
from .pmp_writer import (
    PrefactoredPmpBuild, build_prefactored_pmp_doc, write_pmp_json,
    _sampling_point_labels, _saved_sample_tables_at_nodes,
)
from .sdp_writer import write_sdp_directory
from .defaults import (
    DEFAULT_LARGE_SPIN_POLE_SELECTION,
    DEFAULT_POLE_SELECTION,
)


_IDENTICAL_SCALAR_CROSSING_COORDINATES = frozenset({"zzb", "yyb", "xt", "ws"})
_EXACT_SCALAR_TABLE_BUILDERS = {
    "zzb": G_zzb_poly_table,
    "yyb": G_yyb_poly_table,
    "xt": G_xt_poly_table,
    "ws": G_ws_poly_table,
}
# numeric tables at one Delta through the flint chiral series evaluator (the per-h mpmath differentiation route
# took ~7 s for the Lambda = 19 identity grid)
_NUMERIC_SCALAR_TABLE_BUILDERS = {
    "zzb": G_zzb_table_many,
    "yyb": G_yyb_table_many,
    "xt": G_xt_table_many,
    "ws": G_ws_table_many,
}


def _numeric_scalar_grid(coordinate, delta, h_ext, lam, *, c, q_order, chiral_norm):
    """The numeric ``[m][n]`` scalar block grid at one ``Delta`` for identical externals ``h_ext``."""
    h_ext = _as_real_label(h_ext, name="h_ext")
    (table,) = _NUMERIC_SCALAR_TABLE_BUILDERS[coordinate](
        [delta], 0, [h_ext] * 4, lam, lam,
        c=c, q_order=q_order, chiral_norm=chiral_norm, z_shift=-2 * h_ext,
    )
    return table


def _convert_number(value):
    """``mp.convert(value)``, with a ``Fraction`` rounded to nearest.

    ``mp.mpf`` / ``mp.mpc`` refuse a ``Fraction`` (the exact-input convention), and ``mp.convert`` rounds one toward
    zero (``from_rational``'s default), an ulp below the string ``"1/3"``.
    """
    if isinstance(value, numbers.Rational) and not isinstance(value, int):
        return mp.mpf(from_rational(int(value.numerator), int(value.denominator), mp.mp.prec, round_nearest))
    return mp.convert(value)


def _as_real_mpf(value, *, name: str) -> mp.mpf:
    """Convert a validated real input at the current working precision."""
    return mp.mpf(_convert_number(_as_real_label(value, name=name)))


def _as_real_label(value, *, name: str):
    """Validate a real input without replacing its exact label by a rounded mpf."""
    from virasoro_blocks.numeric_input import real_input_label

    return real_input_label(value, name=name)


def _known_vector_sum(terms):
    """Combine exact real coefficients and stored vector components, rounding once.

    This preserves small specified offsets and cancellations between known
    operators. It does not improve the accuracy of the computed block vectors.
    Terms are ``(validated_coefficient, vector)`` pairs of a common length;
    rounding-level imaginary block data are handled after the whole sum.
    """
    from virasoro_blocks.numeric_input import literal_components

    total = None
    for coefficient, vector in terms:
        parts = [literal_components(value) for value in vector]
        if total is None:
            total = [[0, 0] for _ in parts]
        elif len(parts) != len(total):
            raise ValueError("known operator vectors must have the same length")
        for target, (real, imaginary) in zip(total, parts):
            target[0] += coefficient * real
            target[1] += coefficient * imaginary
    return tuple(
        _convert_number(real) if imaginary == 0 else _canonicalize_real_output(
            mp.mpc(_convert_number(real), _convert_number(imaginary)))
        for real, imaginary in (total or ())
    )


def _stringify_number(value) -> str:
    z = mp.mpc(_convert_number(value))     # a Fraction has no mpc()
    sig_figs = mp.mp.dps + 5
    if abs(mp.im(z)) <= _real_tol(z):
        return mp.nstr(mp.mpf(mp.re(z)), sig_figs)
    real = mp.nstr(mp.re(z), sig_figs)
    imag = mp.nstr(abs(mp.im(z)), sig_figs)
    sign = "+" if mp.im(z) >= 0 else "-"
    return f"({real}{sign}{imag}j)"


def _central_charge_input(value):
    """Validate c without rounding an exact label before singular-locus dispatch."""
    return _as_real_label(value, name="c")


def _validate_even_spins(spin_list: Sequence[int]) -> tuple[int, ...]:
    try:
        spin_list = tuple(spin_list)
    except TypeError:
        raise ValueError("spin_list must be a non-empty sequence") from None
    if not spin_list:
        raise ValueError("spin_list must be non-empty")
    cleaned = []
    for spin in spin_list:
        try:
            spin = _integer_field("spin", spin)
        except ValueError:
            raise ValueError("identical-scalar spin_list entries must be integers") from None
        if spin < 0:
            raise ValueError("identical-scalar spin_list entries must be >= 0")
        if spin % 2 != 0:
            raise ValueError(
                "identical-scalar spin_list entries must be even spins"
            )
        cleaned.append(spin)
    cleaned = tuple(sorted(set(cleaned)))
    if 0 not in cleaned:
        raise ValueError("spin_list must include the scalar channel ell = 0")
    return cleaned


def _default_delta_min(ell: int) -> int:
    return ell


def _resolved_regularize(c, regularize: str) -> bool:
    if regularize == "auto":
        return recognized_singular_locus(c) is not None
    if regularize == "on":
        return True
    if regularize == "off":
        return False
    raise ValueError("regularize must be 'auto', 'on', or 'off'")


def _normalize_npoles(npoles) -> int | None:
    """Map the public ``npoles`` value to the internal kept-pole count.

    ``"all"`` and ``None`` retain every pole and map to internal ``None``.
    A nonnegative integer gives the pole count for order-reduced interpolation.
    """
    if npoles is None or npoles == "all":
        return None
    if isinstance(npoles, bool) or not isinstance(npoles, int) or npoles < 0:
        raise ValueError("npoles must be a non-negative int or 'all'")
    return npoles


def _resolved_method(npoles) -> str:
    """Derive the internal block method from ``npoles`` alone.

    ``npoles="all"`` (keep every pole) -> ``"full"``; any int kept-pole count ->
    ``"interpolation"``. Both route through the same interpolation node
    machinery (the sdp writer needs sample points); "full" is just the
    npoles=None / keep-everything limit (no order reduction).
    """
    return "full" if _normalize_npoles(npoles) is None else "interpolation"


def _resolved_positive_pole_npoles(
    c,
    resolved_method: str,
    npoles: int | None,
    positive_pole_npoles: int | None,
) -> int | None:
    if positive_pole_npoles is not None:
        return positive_pole_npoles
    if _as_real_mpf(c, name="c") < 1 and resolved_method == "full":
        return npoles
    return None


def _resolved_large_spin_npoles(spec) -> int | None:
    """Tower kept-pole count: `large_spin_npoles` if set, else the fixed-spin
    `npoles` (legacy fallback). The tower's inverted geometry wants this low."""
    if spec.large_spin_npoles is not None:
        return spec.large_spin_npoles
    return _normalize_npoles(spec.npoles)


@dataclass(frozen=True)
class IdenticalScalarGapSpec:
    """Everything that defines one identical-scalar crossing problem.

    The four-point function <phi phi phi phi> of identical scalars with chiral
    weights ``h = hbar = h_ext`` (so ``Delta_phi = 2 h_ext``) is expanded in
    Virasoro blocks of even spin, and positivity is imposed on every exchanged
    primary above a per-spin lower edge.  One spec feeds every problem builder
    (gap, OPE, moments); the builders differ only in their objective and
    normalization.  Numbers may be given as ``int``, ``str``, ``Fraction`` or
    ``mpf``. Decimal literals and Python floats denote their written decimal
    values; existing ``mpf`` inputs retain their supplied binary values.

    Physics
        ``c``
            Central charge.  Recursion-singular values (``c = 1``, ``25``,
            minimal models) are handled by ``regularize``.
        ``h_ext``
            External chiral weight; ``Delta_phi = 2 h_ext``.
        ``delta_gap``
            Lower edge of the spin-0 continuum (``>= 0``): positivity is
            imposed for scalars with ``Delta >= delta_gap``.
        ``delta_min_by_spin``
            Optional ``{ell: Delta_min}`` overrides for the non-zero spins in
            ``spin_list`` (default: the unitarity bound ``Delta = ell``).  The
            spin-0 edge is always ``delta_gap``.
        ``spin_list``
            The even spins with a fixed-spin positivity channel.  Default
            ``(0, 2)``; production runs use far more (for example ``0..10``
            plus the large-spin tower, or ``0..24``).
        ``point_constraints``
            ``((ell, Delta), ...)``: extra positivity at isolated operators
            (for example a known light operator below the gap).

    Truncation and precision
        ``lam``
            Derivative order Lambda: the crossing-odd sector keeps the
            ``(m, n)`` derivatives with odd ``m + n <= lam``, ``n <= m``
            (``zzb``), so the functional has ``len(F_minus_pairs(lam))``
            components.
        ``q_order``
            Truncation order of the elliptic ``q`` recursion (default 12, which
            is only enough for small ``lam`` and ``c``).  Too low an order
            leaves a large truncation error; the certifier's q-order gate warns.
            ``q_order ~ lam + 20`` or more is typical at moderate ``c``.
        ``dps``
            Decimal working precision of the block build (default 80): every
            block is built, and the SDP data written, at ``dps`` digits unless
            ``target_digits`` raises it. Each fixed-spin block is verified against its
            numeric sampler (between its nodes and past them to the large-Delta
            regime); a block that loses most of its digits to the monomial
            form's rounding, or does not reproduce the
            block at all, raises a ``RuntimeWarning`` naming the ``dps`` that
            fixes it.  The runner converts it to bits for SDPB's
            ``--precision`` unless ``sdpb_options`` sets one.
        ``target_digits``, ``max_dps``
            Opt in to delivering at least ``target_digits`` verified digits
            per fixed-spin block: the build starts at ``dps`` and is rebuilt at
            a working precision raised by the measured deficit (at most
            ``max_dps``, default ``max(4 target_digits, 2 dps, target_digits + 200)``), and stops
            with an error naming the reason (the cap, a stall, or the order
            reduction of an integer ``npoles``).  The problem's spec then
            records the working precision as ``dps`` (the precision its data
            are written and solved at) and the manifest's provenance records
            the requested one.  ``max_dps`` needs ``target_digits``.
        ``npoles``
            Kept poles of each order-reduced block (default 8).  An int is the
            production path; ``"all"`` keeps every Kac pole, the exact but
            badly conditioned high-degree block.
        ``coordinate``
            Crossing coordinate: ``"zzb"`` (default), ``"yyb"``, or the
            parity-folded ``"xt"`` / ``"ws"``.
        ``chiral_norm``
            Block normalization ``N`` (blocks ~ ``(16 q / N)^h``), a positive
            integer; ``None`` is the library's ``N = 16``.  It does not change a feasibility verdict
            but rescales a reported OPE coefficient.
        ``regularize``
            ``"auto"`` (regularize exactly at recognized singular ``c``),
            ``"on"`` or ``"off"``.

    Large-spin tower
        ``large_spin_lmin``
            When set with ``large_spin_twists``, adds one channel per twist
            that imposes positivity for all spins ``L >= large_spin_lmin`` at
            that twist, covering the spins above ``spin_list``.  A
            non-negative integer no larger than ``max(spin_list) + 2`` (the
            usual choice), so that no spin is left unconstrained; each needs
            the other.
        ``large_spin_twists``
            The twist slices, distinct, typically
            ``virasoro_bootstrap.large_spin.double_twist_set(h_ext, 5)``.
        ``large_spin_npoles``, ``large_spin_pole_selection``
            Kept poles and pole selection of the tower blocks (default: the
            fixed-spin ``npoles``; a low count, 8-12, conditions best).

    Interpolation details (defaults are the tested production choice)
        ``pole_selection``
            Which ``npoles`` poles the fixed-spin blocks keep
            (default ``"contribution_optimal"``; see the CLI reference).
        ``n_sample_points``
            Override the number of interpolation nodes (default: degree + 1).
            A larger count adds nodes.  A smaller one order-reduces the block
            only with ``shared_sample_points=True``; with per-entry node sets
            it would not take effect, and is refused.
        ``sample_via``
            ``"numeric"`` (default) evaluates node values with the numeric
            recursion; ``"symbolic"`` from the exact rational table.
        ``explicit_sample_points``
            Write the interpolation nodes and scalings into the PMP (default
            True), so the SDP is built at the nodes the blocks were fit on.
        ``shared_sample_points``
            Fit every derivative entry on one max-degree node set (default
            False; True can spoil the large-Delta tail).
        ``positive_pole_npoles``, ``positive_pole_selection``
            Remainder interpolation after genuine positive-real poles are
            cleared (a ``c <= 1`` feature).
        ``prefactor_exponent``
            Fixed by crossing at ``h_ext``; leave unset.

    Execution
        ``num_threads``
            Worker processes for the block build and certification (default
            ``min(4, cpu_count)``).  Scripts that use more than one must run
            their work under ``if __name__ == "__main__":``.
    """

    c: object
    h_ext: object
    delta_gap: object
    lam: int
    delta_min_by_spin: dict[int, object] | None = None
    spin_list: tuple[int, ...] = (0, 2)
    q_order: int = 12
    dps: int = 80
    # opt-in: at least this many verified digits per fixed-spin block (escalating the working precision from dps,
    # capped at max_dps); None builds once at dps and reports a shortfall
    target_digits: int | None = None
    max_dps: int | None = None
    # Single block-fidelity knob (replaces the former `method` + `npoles` pair):
    # a non-negative int is the interpolation kept-pole count; the string "all"
    # keeps every pole (the former method="full" -- the exact, no-order-reduction
    # block, which is the conditioning-fragile high-degree extreme). Both route
    # through the same interpolation node machinery.
    npoles: int | str = 8
    n_sample_points: int | None = None
    # Pole selection for the fixed-spin Delta channels (the interpolated block's
    # kept prefactor poles). Defaults to DEFAULT_POLE_SELECTION
    # ("contribution_optimal" since 2026-07-30). Other options:
    # "contribution" (the (0,0)-value residue divided by the distance to the
    # evaluation point -- the previous default; geometry-adaptive, reduces to a
    # residue-style ranking when the poles already decay across the window and
    # to a nearest-style ranking in the inverted tower), "contribution_bernstein"
    # (the previous production choice), "residue" (ranks poles by their residue
    # in the undifferentiated (0,0) block value), "nearest" (smallest |x|,
    # closest to the positivity interval), "rightmost" (largest Re, widest
    # support), "residue_all" (max residue over every kept (m, n) entry; the
    # former "residue"), "contribution_all" (the all-derivative residue divided
    # by the distance; the former "contribution"), plus "real_first" /
    # "real_deep".
    pole_selection: str = DEFAULT_POLE_SELECTION
    positive_pole_npoles: int | None = None
    # Pole selection for the genuine positive-real-pole lowering remainder
    # (`crossing.interpolate_serialized_block`), a SEPARATE knob from
    # `pole_selection`. Options: "rightmost", "nearest", "residue" (ranks poles
    # by their residue in the undifferentiated (0,0) block value) and
    # "residue_all" (max residue over every kept (m, n) entry). Default
    # "rightmost".
    positive_pole_selection: str = "rightmost"
    regularize: str = "auto"
    coordinate: str = "zzb"
    # Fixed by crossing at h_ext (the blocks carry z_shift = -2 h_ext); kept so manifests record it.  Any other
    # value is refused.
    prefactor_exponent: object | None = None
    # Block normalization N: blocks decay as (16q/N)^h, so N sets the
    # damped-rational base (= 16q/N) that drives the Bohr-Sommerfeld sample-point
    # density. None (default) is the library's N=16 -> base q~0.043, which
    # matches the physical large-weight decay of OPE^2*block (~q^Delta and ~q^ell,
    # from Virasoro Mean Field Theory, arXiv:1811.05710) and concentrates the
    # nodes where the contributions actually live -> better conditioning; N=4 gives base 4q~0.173. N is
    # free for gap feasibility (the verdict is
    # invariant); for OPE-max it rescales the reported OPE^2 value.
    chiral_norm: int | None = None
    # How the interpolation node values are evaluated: "numeric" (default) uses
    # the well-conditioned O(1) block recursion; "symbolic" evaluates the exact
    # P(Delta)/D(Delta) (~1e500 coeffs, needs high dps). Both yield the same
    # interpolant -- only conditioning differs. Orthogonal to `npoles` (which
    # poles are kept) and to the SDP backend (pmp2sdp vs sdp).
    sample_via: str = "numeric"
    explicit_sample_points: bool = True
    # Default False (per-entry): every (m, n) entry of an interpolated block keeps
    # its own degree-matched Bohr-Sommerfeld node set, tying its polynomial degree
    # to npoles so the large-Delta tail power stays correct. The block pmp2sdp
    # receives still reproduces the true block at its (max-degree) sample points to
    # bootstrap precision -- the per-entry node mismatch is harmless. Set True to
    # pad every entry onto the single max-degree node set: more accurate strictly
    # in-window, but the far tail is no longer power-matched and can blow up
    # (Runge), which can spuriously inflate certified bounds that probe large
    # Delta (observed ~9x inflation of a 2d twist-gap bound at lambda=7).
    shared_sample_points: bool = False
    # Large-spin asymptotic positivity tower. When `large_spin_lmin` is set and
    # `large_spin_twists` is non-empty, one continuous-in-L positivity channel
    # per twist (covering the tower L >= large_spin_lmin) is appended to the PMP.
    # These blocks are gap-independent, so they are built once in `prepare` and
    # reused across a whole binary search. The double-twist twist set
    # (`virasoro_bootstrap.large_spin.double_twist_set`) is the physically
    # motivated choice for `large_spin_twists`.
    large_spin_lmin: object | None = None
    large_spin_twists: tuple = ()
    # Pole selection for the large-spin tower blocks. This is a SEPARATE knob
    # from `pole_selection` (which governs the fixed-spin Delta channels)
    # because the tower's geometry is inverted: the positivity domain is the
    # spin tail L >= large_spin_lmin while *every* candidate pole sits at L < 0,
    # so the block's decay across the fit region is shaped by the poles nearest
    # the region, not the largest-residue ones. `contribution_optimal` is the
    # default; the tower remains a separate knob because its inverted geometry
    # has distinct conditioning.
    large_spin_pole_selection: str = DEFAULT_LARGE_SPIN_POLE_SELECTION
    # Kept-pole count for the large-spin tower blocks, SEPARATE from the
    # fixed-spin `npoles`. The tower's geometry is inverted (positivity domain
    # L >= large_spin_lmin, all poles at L < 0), so its block is already
    # converged in value at a handful of poles; keeping *more* poles adds no
    # accuracy in the L >= Lmax region but makes the order-reduced polynomial's
    # coefficients explode (empirically ~1e25 at npoles=8 -> ~1e50 at npoles=28
    # for Lambda=11), which wrecks the SDP conditioning and produces spurious
    # bounds. So the tower wants a LOW, independently-set npoles (~8-12). When
    # `None`, falls back to the fixed-spin `npoles` (legacy behavior).
    large_spin_npoles: int | None = None
    # Point-constraint channels: each ``(ell, delta_point)`` imposes positivity
    # ``y . F(delta_point, ell) >= 0`` at one isolated dimension (e.g. an allowed
    # island point or a known light operator below the gap). The block is frozen
    # at ``delta_point`` via the robust numeric evaluator (a degree-0
    # ``variable="point"`` channel), independent of the gap, so -- like the
    # large-spin tower -- it is built once in `prepare` and reused across a
    # binary search.
    point_constraints: tuple = ()
    # `num_threads` controls parallelism across spin channels for both the
    # exact-table assembly (`build_table_from_cache` per spin) and the
    # interpolation/serialization step. Matches `virasoro_blocks.__main__`'s
    # `--num-threads` flag in spirit -- both parallelize across spins via a
    # `ProcessPoolExecutor`, but emit to different downstream artifacts
    # (block-table writers vs PMP/SDPB). The shared chiral cache is built once
    # before fanning out, so the workers don't redo it. Empirically a 4-worker
    # build at lambda=7, q_order=12, dps=80 cuts `prepare_seconds` roughly 2.5x
    # vs the serial path, with diminishing returns past 4. Default scales to
    # `min(4, os.cpu_count())` so the typical CLI / driver invocation gets the
    # parallel path automatically; pass `num_threads=1` to force serial.
    num_threads: int = max(1, min(4, os.cpu_count() or 1))

    def __post_init__(self) -> None:
        # Preserve input labels exactly; an mpf denotes its supplied binary value. A float c within float rounding
        # of a singular locus is that locus (singularity.snap_float_c: 0.7 is 7/10, 6/7 the minimal model M(6,7)),
        # so the spec's own method decisions and every builder it calls see the same exact c.
        from virasoro_blocks.numeric_input import literal_number
        from virasoro_blocks.singularity import snap_float_c

        object.__setattr__(self, "c", literal_number(snap_float_c(self.c)))
        object.__setattr__(self, "h_ext", literal_number(self.h_ext))
        # List-valued sequence fields (a JSON round trip, a hand-written call)
        # made the spec unhashable, and the moment layer keys caches on it
        # (``moments._cached_base_problem``).  Canonicalize them to tuples so
        # equal specs hash and compare equal.
        for name in ("spin_list", "large_spin_twists"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, tuple):
                object.__setattr__(self, name, tuple(value))
        points = self.point_constraints
        if points is not None and (
            not isinstance(points, tuple)
            or any(not isinstance(point, tuple) for point in points)
        ):
            object.__setattr__(
                self, "point_constraints", tuple(tuple(point) for point in points)
            )
        # The named choices are read only deep in the build (a tower selector only once the tower is sampled), so
        # a typo there surfaced after minutes of block work, or not at all.
        _check_spec_choices(self)

    def __hash__(self) -> int:
        """Hash the spec with per-spin overrides in canonical order.

        Convert the override mapping to a sorted tuple so equal mappings have
        equal hashes, consistent with the dataclass equality operation.
        """
        overrides = self.delta_min_by_spin
        canonical = (
            None
            if overrides is None
            else tuple(sorted(
                ((int(ell), delta) for ell, delta in overrides.items()),
                key=lambda item: item[0],
            ))
        )
        return hash(
            (canonical,)
            + tuple(
                getattr(self, f.name)
                for f in _dataclass_fields(self)
                if f.name != "delta_min_by_spin"
            )
        )


@dataclass(frozen=True)
class IdenticalScalarSpinChannel:
    ell: int
    delta_min: object
    serialized: SerializedDerivTable
    prefactored: SerializedDerivTable
    # Large-spin tower slices set ``twist`` (= tau); the positivity variable is
    # then the spin L (x = L - delta_min) at fixed twist, with Delta = tau + L.
    # Fixed-spin channels leave it ``None`` (variable is Delta, L = ell fixed).
    twist: object | None = None
    # In-memory inputs of a helper-built frozen point, so a consumer cannot
    # silently use its vector for another dimension or crossing convention.
    # Manifest readers retain the context after validating file metadata.
    point_context: dict | None = None
    # Continuum channels carry the same physical crossing inputs, independent
    # of their positivity edge, so fixed-operator additions can validate reuse.
    crossing_context: dict | None = None


@dataclass(frozen=True)
class IdenticalScalarGapProblem:
    spec: IdenticalScalarGapSpec
    resolved_method: str
    used_regularize: bool
    prefactor_exponent: object
    F_pairs: tuple[tuple[int, int], ...]
    identity_vector: tuple
    spin_channels: tuple[IdenticalScalarSpinChannel, ...]
    pmp_build: PrefactoredPmpBuild
    # The fixed-spin gap channels are ``spin_channels[:len(spec.spin_list)]``;
    # the gap-independent large-spin tower and point-constraint channels follow.
    # ``large_spin_channels`` / ``point_channels`` mirror those tails so the
    # manifest writer can classify by kind instead of by position.
    large_spin_channels: tuple[IdenticalScalarSpinChannel, ...] = ()
    point_channels: tuple[IdenticalScalarSpinChannel, ...] = ()
    # v2 clean F-vector sample values {block_index: P[k][a]} from the numeric
    # recursion, consumed by the sdp backend (write_sdp_directory). Blocks absent
    # from the dict fall back to the writer's v1 monomial evaluation.
    sample_values_by_block: dict | None = None
    # The SDP's two scalar vectors and which problem they encode, recorded
    # explicitly so the manifest is self-describing (see manifest.PmpVectors):
    #   gap     -> objective = 0,    normalization = V_id
    #   ope_max -> objective = V_id, normalization = V_* at (spin_star, delta_star)
    problem_kind: str = "gap"
    objective_vector: tuple = ()
    normalization_vector: tuple = ()
    # Distinguished operator labels, with direction/coefficient where applicable.
    normalization_operator: dict | None = None
    pmp_metadata: dict | None = None
    # Keep the already-built exact fixed-spin tables available to diagnostics
    # and physical-block certification.  Moment builders receive a prepared
    # build, so throwing these away forces the certifier to rebuild the same
    # high-precision recursion tables a second time (particularly expensive in
    # the c -> 1 audit).  This is an in-memory provenance/cache handle; it is
    # not serialized into the PMP or task output.
    exact_tables_by_ell: dict[int, object] | None = None
    # the dps the spec asked for when a target_digits build raised it (spec.dps is the working precision); recorded
    # in the manifest's provenance
    requested_dps: int | None = None
    # the scalar-G spin-0 normalization its blocks and identity vector were built in (virasoro_blocks._conventions),
    # captured when the problem is built; the manifest writer tags the manifest and block files with it
    scalar_G_spin0: str = _dataclass_field(default_factory=scalar_G_spin0_convention)


@dataclass(frozen=True)
class PreparedIdenticalScalarGapBuild:
    reuse_key: tuple
    resolved_method: str
    used_regularize: bool
    resolved_positive_pole_npoles: int | None
    prefactor_exponent: object
    F_pairs: tuple[tuple[int, int], ...]
    identity_vector: tuple
    exact_tables: dict[int, object]
    prepared_spin_channels: tuple[IdenticalScalarSpinChannel, ...]
    large_spin_channels: tuple[IdenticalScalarSpinChannel, ...] = ()
    point_channels: tuple[IdenticalScalarSpinChannel, ...] = ()
    # the working precision the blocks were built (and must be written) at: spec.dps, or above it for a
    # target_digits build; None for a build that predates the field (spec.dps)
    working_dps: int | None = None
    # the scalar-G spin-0 normalization the blocks and identity vector were built in, captured at build time: a
    # problem built from this under another convention would mix spin-0 normalizations, so it is refused
    scalar_G_spin0: str = _dataclass_field(default_factory=scalar_G_spin0_convention)


@dataclass(frozen=True)
class DockerBootstrapRun:
    output_dir: Path
    manifest_path: Path
    pmp_path: Path
    sdp_dir: Path
    sdpb_out_dir: Path | None
    precision_bits: int
    pmp2sdp_command: tuple[str, ...]
    sdpb_command: tuple[str, ...] | None
    pmp2sdp_log_path: Path | None = None
    sdpb_log_path: Path | None = None
    pmp2sdp_summary: str | None = None
    sdpb_status: str | None = None
    sdpb_summary: str | None = None
    # The SDPB checkpoint directory written by this solve (None if SDPB wrote no
    # final checkpoint, e.g. --noFinalCheckpoint). Feed it as the next solve's
    # ``initial_checkpoint_dir`` to warm-start (hotstart) a nearby problem -- in
    # a gap binary search only delta_gap changes, so the SDP block structure is
    # preserved and the prior iterate is a valid, close starting point.
    checkpoint_dir: Path | None = None


def _chiral_memo_scalar_key(value):
    """Hashable (precision-implicit) key for a probe central charge."""
    if isinstance(value, mp.mpc):
        return ("mpf", mp.mpf(value.real)._mpf_) if value.imag == 0 else ("mpc", value._mpc_)
    return ("mpf", mp.mpf(value)._mpf_)


def _regularized_scalar_sampler(c, h_ext, coordinate, ell, table, q_order, chiral_norm):
    """The value-space regularized numeric sampler of one identical-scalar table at a singular c."""
    from virasoro_blocks.regularization import _RegularizedNumericSampler

    h_externals = [h_ext] * 4
    z_shift = -2 * h_ext
    return _RegularizedNumericSampler(
        c=c, coordinate=coordinate, chiral=False, spinning=False, projected=False, h_externals=h_externals,
        hbar_externals=h_externals, ell=ell, m_max=table.m_max, n_max=table.n_max, b=None, q_order=q_order,
        chiral_norm=chiral_norm, z_shift=z_shift, zbar_shift=z_shift,
    )


def _build_exact_scalar_table(
    c,
    h_ext,
    coordinate: str,
    ell: int,
    lam: int,
    q_order: int,
    *,
    used_regularize: bool,
    chiral_norm=None,
    chiral_memo=None,
    skeleton: bool = False,
    keep_all: bool = False,
):
    h_externals = [h_ext] * 4
    z_shift_val = -2 * h_ext
    builder = _EXACT_SCALAR_TABLE_BUILDERS[coordinate]

    def _build(c_value):
        # The chiral expansion is spin-independent, so the merged-pole build
        # shares it across spin channels via `chiral_memo`. The fold
        # (build_table_from_cache) is the only per-spin step and is cheap; the
        # result is bit-identical to the all-in-one `builder`. The memo is keyed
        # on precision, c, and collision context to avoid sharing incompatible
        # chiral expansions.
        if chiral_memo is not None:
            from virasoro_blocks.singularity import _UNSTRIPPED_COLLISIONS

            key = (mp.mp.dps, _chiral_memo_scalar_key(c_value), _UNSTRIPPED_COLLISIONS.get())
            cs_pair = chiral_memo.get(key)
            if cs_pair is None:
                chiral_cache = build_chiral_cache(
                    (coordinate,),
                    h_externals,
                    lambda_=lam,
                    c=c_value,
                    q_order=q_order,
                    chiral_norm=chiral_norm,
                    z_shift=z_shift_val,
                    skeleton=skeleton,
                )
                cs_pair = chiral_cache[coordinate]
                chiral_memo[key] = cs_pair
            cs_h, cs_hbar = cs_pair
            # build_table_from_cache infers the skeleton (degree-only) build from
            # the chiral input carrying numerators=None.
            return build_table_from_cache(
                coordinate, cs_h, cs_hbar, ell,
                m_max=lam, n_max=lam, basis="scalar_G",
            )
        return builder(
            ell,
            h_externals,
            lam,
            lam,
            c=c_value,
            q_order=q_order,
            chiral_norm=chiral_norm,
            z_shift=z_shift_val,
            skeleton=skeleton,
        )

    if used_regularize:
        if skeleton:
            # Build the exact merged pole structure at c, with every pole at
            # its order after cancellation. Node values come from the merged
            # numeric sampler, so no symbolic numerators are needed.
            from virasoro_blocks.regularization import merged_delta_skeleton

            table = merged_delta_skeleton(_build, c=c)
            table.numeric_sampler = _regularized_scalar_sampler(c, h_ext, coordinate, ell, table, q_order, chiral_norm)
            return table
        # the full table AT the singular c: the exact merged poles, numerators included
        from virasoro_blocks.regularization import merged_delta_table

        return merged_delta_table(_build, c=c)
    return _build(c)


def _build_exact_scalar_table_from_cache_task(args):
    dps, coordinate, cs_h, cs_hbar, ell, lam = args
    mp.mp.dps = dps
    return ell, build_table_from_cache(
        coordinate,
        cs_h,
        cs_hbar,
        ell,
        m_max=lam,
        n_max=lam,
        basis="scalar_G",
    )


def _interpolation_kept_entries(coordinate: str, m_max: int, n_max: int, ell: int) -> list[tuple[int, int]]:
    """Entries the writer keeps for one (coordinate, spin), via the shared
    ``truncation.kept_pairs`` policy.

    The identical-scalar channels are scalar G, which has definite z<->zbar
    parity, so the non-folded coordinates fold to the ``n <= min(m, Lambda - m)``
    lower triangle -- the ``n > m`` half is determined by symmetry and is never
    read by the crossing functional (``F_minus_pairs`` lives in ``n <= m``), so
    interpolating it is wasted work. Delegating to ``kept_pairs`` keeps this in
    lockstep with the writer and, crucially, stays correct for any future basis
    without a z<->zbar symmetry (e.g. ordered B), where ``kept_pairs`` returns
    the full triangle with both ``(m, n)`` and ``(n, m)``.
    """
    return kept_pairs(coordinate, max(m_max, n_max), basis="scalar_G", ell=ell)


def _build_exact_scalar_tables(
    c,
    h_ext,
    coordinate: str,
    spin_list: Sequence[int],
    lam: int,
    q_order: int,
    *,
    used_regularize: bool,
    resolved_method: str,
    chiral_norm=None,
    num_threads: int = 1,
    sample_via: str = "numeric",
    pole_selection: str = DEFAULT_POLE_SELECTION,
) -> dict[int, object]:
    if sample_via not in ("numeric", "symbolic"):
        raise ValueError(f"sample_via must be 'numeric' or 'symbolic', got {sample_via!r}")
    if used_regularize:
        # Singular c under the numeric sampler: build the regularized pole
        # structure from a degree-only skeleton (no large symbolic
        # numerators) whenever the pole selector can work from numeric values.
        # The merged skeleton includes minimal-model cancellations
        # analytically as well, so c < 1 can use the same fast path; only the
        # legacy residue-based selectors still need full symbolic numerators.
        # sample_via="symbolic" also needs the numerators.
        #
        # Legacy residue-based selection (residue / contribution_leading) needs
        # per-pole LEADING residue magnitudes from the symbolic numerators.
        # Those keep the full merged symbolic table. The contribution
        # selectors rank by the FULL principal part,
        # finite for the collapsed pole and extracted numerically from block
        # values, so they use the skeleton; geometric selectors need no residues.
        legacy_residue_selection = pole_selection in (
            "residue", "residue_all", "contribution_leading",
            "contribution_leading_all",
        )
        skeleton = (
            sample_via == "numeric"
            and not legacy_residue_selection
        )
        # Share the spin-independent chiral expansions across the spin channels:
        # the Laurent recursion's chiral cache is independent of spin. One
        # memo across spins avoids rebuilding the same chiral expansion.
        chiral_memo: dict = {}
        tables = {
            ell: _build_exact_scalar_table(
                c,
                h_ext,
                coordinate,
                ell,
                lam,
                q_order,
                used_regularize=True,
                chiral_norm=chiral_norm,
                chiral_memo=chiral_memo,
                skeleton=skeleton,
                keep_all=resolved_method == "full",
            )
            for ell in spin_list
        }
        # Attach the value-space regularized numeric sampler so the interpolation
        # fits the block's exact merged-pole recursion at c at the sample
        # points, avoiding evaluation of the high-degree P(Delta)/D(Delta). The
        # regularized table's poles/degrees are used; its symbolic numerators are
        # not read for sample values. (sample_via="symbolic" keeps the direct
        # symbolic evaluation -- the cross-check path.)
        if sample_via == "numeric":
            for ell, table in tables.items():
                if table.numeric_sampler is None:
                    table.numeric_sampler = _regularized_scalar_sampler(
                        c, h_ext, coordinate, ell, table, q_order, chiral_norm)
        return tables

    h_externals = [h_ext] * 4
    z_shift = -2 * h_ext
    from virasoro_blocks.table_backends import numeric_skeleton_eligible

    # the numeric sampler supplies every value, so (as on the CLI) only the pole structure is built
    skeleton = numeric_skeleton_eligible(
        method=resolved_method, sample_via=sample_via, basis="scalar_G", coordinates=(coordinate,),
        pole_selection=pole_selection, regularize=False,
    )
    chiral_cache = build_chiral_cache(
        (coordinate,),
        h_externals,
        lambda_=lam,
        c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
        symbolic_guard=(resolved_method == "full"),
        skeleton=skeleton,
    )
    cs_h, cs_hbar = chiral_cache[coordinate]
    dps = mp.mp.dps
    jobs = [(dps, coordinate, cs_h, cs_hbar, ell, lam) for ell in spin_list]
    if usable_workers(num_threads) == 1 or len(jobs) <= 1:
        tables = {
            ell: table
            for ell, table in (
                _build_exact_scalar_table_from_cache_task(job) for job in jobs
            )
        }
    else:
        with ProcessPoolExecutor(max_workers=min(num_threads, len(jobs)), **pool_initializer_kwargs()) as pool:
            tables = dict(pool.map(_build_exact_scalar_table_from_cache_task, jobs))
    # Attach the well-conditioned numeric block sampler: interpolation then fits
    # to the O(1) recursion at the sample points instead of evaluating the
    # ~1e500-coeff symbolic P(Delta)/D(Delta) (which silently needs very high
    # dps to survive cancellation). The numeric block equals the symbolic value,
    # so the interpolant is identical -- only conditioning improves.
    # `sample_via="symbolic"` skips the sampler so the interpolation falls back
    # to evaluating the symbolic table directly (the cross-check path).
    if sample_via == "numeric":
        for ell, table in tables.items():
            table.numeric_sampler = _NumericBlockSampler(
                coordinate, h_externals, ell, table.m_max, table.n_max,
                b=None, c=c, q_order=q_order, chiral_norm=chiral_norm,
                z_shift=z_shift,
            )
    return tables


def _serialize_spin_channel(
    exact,
    *,
    coordinate: str,
    ell: int,
    delta_min,
    resolved_method: str,
    npoles: int | None,
    n_sample_points: int | None,
    pole_selection: str,
    sample_via: str,
    shared_sample_points: bool = False,
    precision_request=None,
):
    # Unified path: both "interpolation" and "full" route through the
    # interpolation node machinery so every channel carries sample points (the
    # sdp writer needs them, and they enable clean recursion sampling). "full"
    # is simply "keep every pole" (npoles=None) -- no order reduction, so the
    # interpolant reproduces the exact rational function while still being built
    # via the well-conditioned numeric sampler and carrying sample points.
    if resolved_method in ("interpolation", "full"):
        kept_entries = _interpolation_kept_entries(
            coordinate,
            exact.m_max,
            exact.n_max,
            ell,
        )
        reduced = interpolate_delta_tables(
            {coordinate: exact},
            npoles if resolved_method == "interpolation" else None,
            shift=delta_min,
            n_sample_points=n_sample_points,
            pole_selection=pole_selection,
            shared_sample_points=shared_sample_points,
            kept_entries_by_coord={coordinate: kept_entries},
        )[coordinate]
        serialized = serialize_deriv_tables({coordinate: reduced})
        if n_sample_points is not None and not shared_sample_points:
            _check_n_sample_points(serialized, coordinate, kept_entries, n_sample_points, ell)
        sample_source = reduced
    else:
        serialized = serialize_deriv_tables({coordinate: exact}, shift=delta_min)
        sample_source = exact
    serialized.sample_via = sample_via
    _attach_clean_sample_block_values(serialized, sample_source, exact, coordinate)
    if precision_request is not None and resolved_method in ("interpolation", "full"):
        # (after the node values are attached: the diagnosis compares the fit with them)
        _verify_spin_channel(serialized, exact, coordinate, kept_entries, precision_request,
                             order_reduced=resolved_method == "interpolation")
    return serialized


def _check_n_sample_points(serialized, coordinate, kept_entries, n_sample_points, ell) -> None:
    """Reject a node count below the requirement of a per-entry fit.

    With ``shared_sample_points=False``, each derivative entry uses degree-matched
    nodes. The highest degree d requires d + 1 nodes, which the PMP also needs.
    A larger requested count adds nodes in either sampling mode. With
    ``shared_sample_points=True``, every entry is fit on the requested node set.
    """
    rows = serialized.numerators[coordinate]
    degree = max((rows[m][n].degree for m, n in kept_entries), default=0)
    if n_sample_points < degree + 1:
        raise ValueError(
            f"n_sample_points = {n_sample_points} is below the {degree + 1} nodes the spin-{ell} block needs (its "
            f"polynomial degree {degree} + 1): with per-entry node sets (shared_sample_points=False) the top-degree "
            f"entries and the PMP keep {degree + 1} nodes, so the request would not take effect. Leave it unset, "
            f"use n_sample_points >= {degree + 1}, or set shared_sample_points=True to fit every entry on "
            f"{n_sample_points} nodes (a lower-degree block)"
        )


def _has_positive_axis_poles(exact, delta_min) -> bool:
    """Whether the block has poles on the positivity axis Delta > delta_min (c <= 1)."""
    tol = mp.mpf(10) ** (-(mp.mp.dps // 2))
    return any(abs(mp.im(mp.mpc(root))) <= tol * max(1, abs(mp.mpc(root))) and mp.re(mp.mpc(root)) > delta_min
               for root in (getattr(exact, "denominator_roots", None) or ()))


def _warn_if_c_le_1_degree(c, channels, exact_tables, delta_min_by_ell) -> None:
    """Warn when retained positive-axis Kac poles produce an unsupported SDP.

    For c <= 1, these poles can make the lowered problem too ill-conditioned
    for SDPB. Report the affected spins and node count before solving.
    """
    import warnings

    if mp.re(mp.mpc(c)) > 1:
        return
    on_axis = [ell for ell, table in (exact_tables or {}).items()
               if _has_positive_axis_poles(table, delta_min_by_ell.get(ell, 0))]
    if not on_axis:
        return
    nodes = max((len(getattr(channel.serialized, "sample_points_x", None) or ()) for channel in channels), default=0)
    warnings.warn(
        f"c = {mp.nstr(mp.re(mp.mpc(c)), 8)} <= 1: the blocks of spins {sorted(on_axis)} have Kac poles on the "
        f"positivity axis, kept and lowered into the SDP (up to {nodes} nodes per block). SDPB does not converge on "
        "these problems yet (Ising at Lambda 5: data ~1e253, zero step lengths, max_complementarity); the c <= 1 "
        "bootstrap is not supported in this release.", RuntimeWarning, stacklevel=3)


def _verify_spin_channel(serialized, exact, coordinate, kept_entries, request, *, order_reduced) -> None:
    """Verify a fixed-spin block between and beyond its interpolation nodes.

    Use the CLI table verification routine and record ``verified_digits``,
    ``verified_digits_largest_entry``, and ``working_dps`` on ``serialized``.
    Attach the :class:`~virasoro_blocks.verified_build.Verification` as
    ``serialized.verification`` for :func:`_warn_block_shortfalls`. If the
    requested accuracy is not met, diagnose monomial-rounding losses.
    """
    from virasoro_blocks.table_backends import _verified_digits
    from virasoro_blocks.verified_build import diagnose_monomial, record_verification

    verification = _verified_digits(serialized, {coordinate: exact}, {coordinate: list(kept_entries)}, request.digits)
    if verification is not None and request.short(verification.digits):
        verification = diagnose_monomial(serialized, verification, digits=request.digits, work=mp.mp.dps,
                                         order_reduced=order_reduced, kept_by_coord={coordinate: list(kept_entries)})
    record_verification(serialized, verification, mp.mp.dps, request)
    serialized.verification = verification


def _channel_shortfalls(channels, request):
    """``[(ell, verification)]`` of the fixed-spin channels short of ``request``'s digits."""
    out = []
    for channel in channels:
        verification = getattr(channel.serialized, "verification", None)
        if verification is not None and request.short(verification.digits):
            out.append((channel.ell, verification))
    return out


def _warn_block_shortfalls(channels, request, *, npoles, work, tower_channels=(), tower_npoles=None) -> None:
    """Warn about inaccurate fixed-spin blocks and large-spin tower slices.

    Report precision-limited blocks with less than half the requested digits,
    blocks with fewer than three digits relative to their largest entry, and
    blocks with any entry below three digits relative to itself. Every entry
    is a separate crossing component. Pole reduction with at least three
    digits per entry is treated as the requested approximation.

    The CLI table check uses only the largest-entry reproduction rule
    (:func:`virasoro_blocks.verified_build.reproduces`).
    """
    from virasoro_blocks.verified_build import reproduces, shortfall_message

    def report(group, what, request=request):
        reported = []
        for label, verification in group:
            severe = (not reproduces(verification) or verification.digits < 3
                      or (verification.cause == "precision" and verification.digits < request.digits / 2))
            if severe:
                reported.append((label, verification))
        if not reported:
            return
        label, worst = max(reported, key=lambda item: item[1].needed_precision or 0) if all(
            v.cause == "precision" for _label, v in reported) else min(reported, key=lambda item: item[1].digits)
        message = shortfall_message(what(label, [lab for lab, _v in reported if lab != label]), worst, request, work)
        warnings.warn(message.split(": ", 1)[1] if message.startswith(("warning: ", "note: ")) else message,
                      RuntimeWarning, stacklevel=4)

    fixed = "keep-all" if npoles == "all" else f"npoles {npoles}"
    report(_channel_shortfalls(channels, request),
           lambda ell, others: f"the spin-{ell} block ({fixed})" + (f" (spins {sorted(others)} too)" if others else ""))
    towers = [(mp.nstr(ch.twist, 6), v) for ch, (_ell, v) in
              ((ch, item) for ch in tower_channels for item in _channel_shortfalls([ch], request))]
    lmin = min((int(ch.ell) for ch in tower_channels), default=None)
    from dataclasses import replace

    tower_request = replace(request, knobs=replace(request.knobs, npoles="large_spin_npoles",
                                                   keep_all="leave large_spin_npoles unset to use npoles"))
    report(towers, lambda tw, others: (f"the large-spin tower slice at twist {tw} (L >= {lmin}, large_spin_npoles "
                                       f"{tower_npoles})" + (f" (twists {others} too)" if others else "")),
           request=tower_request)


def _binding_verification(channels, request):
    """The Verification the escalation of a ``target_digits`` build follows (the whole build is rebuilt at one
    working precision): a short channel whose cause more precision cannot fix, else the precision-limited one that
    needs the most, with the fewest digits of any channel (the stall test compares those)."""
    from dataclasses import replace

    measured = [getattr(ch.serialized, "verification", None) for ch in channels]
    measured = [v for v in measured if v is not None]
    if not measured:
        return None
    short = [(ell, v) for ell, v in _channel_shortfalls(channels, request)]
    fewest = min(v.digits for v in measured)
    if not short:
        return replace(min(measured, key=lambda v: v.digits), digits=fewest)
    unfixable = [(ell, v) for ell, v in short if v.cause != "precision"]
    if unfixable:
        ell, v = min(unfixable, key=lambda item: item[1].digits)
    else:
        ell, v = max(short, key=lambda item: item[1].needed_precision or 0)
    return replace(v, digits=fewest, detail=f"(the spin-{ell} block)")


def _build_spin_channel_task(args):
    (dps, coordinate, ell, exact, delta_min, resolved_method, npoles, n_sample_points,
     pole_selection, sample_via, chiral_norm, shared_sample_points, precision_request) = args
    mp.mp.dps = dps
    serialized = _serialize_spin_channel(
        exact,
        coordinate=coordinate,
        ell=ell,
        delta_min=delta_min,
        resolved_method=resolved_method,
        npoles=npoles,
        n_sample_points=n_sample_points,
        pole_selection=pole_selection,
        sample_via=sample_via,
        shared_sample_points=shared_sample_points,
        precision_request=precision_request,
    )
    # The serialized block is already library-normalized (self-dual base
    # 16 q(1/2)/N); feed it straight to the PMP. An N^Delta rebase would only
    # revert the conditioning benefit and cannot move the bound, since N^Delta > 0
    # preserves per-operator positivity.
    prefactored = serialized
    return ell, delta_min, serialized, prefactored


def _clean_sample_values_by_block(spin_channels, pmp_doc, F_pairs, *, coordinate,
                                  lowerings=None, sampler_for=None, sampling_blocks=None):
    """Build the v2 ``{block_index: P[k][a]}`` clean F-vector sample values.

    For each PMP block, ``P[k][a] = block_value(Delta_k)[m_a][n_a] /
    sampleScaling[k]``: the writer multiplies back by ``sampleScaling`` to
    recover the exact block value, but the value comes straight from the numeric
    recursion (``sample_block_values``) rather than evaluating the large-monomial
    numerator polynomial. Since ``sampleScaling * P = block_value`` is invariant,
    this is correct through complex-pole folding and is independent of how the
    prefactor/numerator split is bookkept.

    A block whose PMP sample points are not its own nodes (the positive-real-pole
    lowering re-derived a denser node set for the cleared block) is sampled
    numerically at the PMP's nodes: ``sampler_for(index)`` gives the channel's
    numeric block sampler, and the cleared block is the block times the lowering's
    clearing factor (``lowerings[index]``, in product form). A block with neither
    stored values at its nodes nor a sampler is skipped (-> the writer's v1
    monomial path for that block).

    ``sampling_blocks`` carries the writer's prepared source grids; without
    that provenance the coordinate labels in the PMP document are authoritative.
    Arithmetic runs at the source tables' working precision (or the caller's,
    if higher). Real-output validation uses the caller's precision and its
    roundoff allowance.
    """
    from flint import arb, ctx
    from mpmath.libmp import dps_to_prec
    from virasoro_blocks.numeric_input import _label_sum
    from virasoro_blocks.polynomial import _arb_to_mpf
    from virasoro_blocks.serialization import _parse_serialized_number
    from .crossing import clearing_factor_value
    from .pmp_writer import _pmp_real_value

    blocks = pmp_doc["PositiveMatrixWithPrefactorArray"]
    entries = list(dict.fromkeys(F_pairs))
    validation_prec = mp.mp.prec
    out: dict[int, list] = {}
    for i, channel in enumerate(spin_channels):
        if i >= len(blocks):
            break
        block_doc = blocks[i]
        if "sampleScalings" not in block_doc:
            continue
        source_nodes = (None if sampling_blocks is None else sampling_blocks[i].sample_points_x)
        doc_sp = _sampling_point_labels(
            block_doc["samplePoints"] if source_nodes is None else source_nodes,
            where="clean block sampling",
        )
        lowering = None if lowerings is None or i >= len(lowerings) else lowerings[i]
        groups = () if lowering is None else tuple(lowering.selected_groups)
        tables = _saved_sample_tables_at_nodes(channel.prefactored, coordinate, doc_sp)
        if tables is None and sampler_for is None:
            continue
        source_tables = (channel.prefactored,) if sampling_blocks is None else (channel.prefactored, sampling_blocks[i])
        source_dps = max(int(getattr(block, "working_dps", None) or 0) for block in source_tables)
        work_prec = max(validation_prec, dps_to_prec(source_dps))
        with mp.workprec(work_prec):
            try:
                sc = [_parse_serialized_number(s) for s in block_doc["sampleScalings"]]
            except ValueError as exc:
                raise ValueError(f"clean block sampling: invalid scalings: {exc}") from exc
            if (len(sc) != len(doc_sp) or any(not mp.isfinite(s) or mp.im(s) != 0 or mp.re(s) <= 0 for s in sc)):
                raise ValueError("clean block sampling: scalings must be finite, positive and match the sample points")
            if tables is None:
                shift = _as_real_label(channel.prefactored.shift, name="block shift")
                points = [_label_sum(shift, _as_real_label(x, name="sample point")) for x in doc_sp]
                before = ctx.prec
                try:
                    sampler = sampler_for(i)
                    if sampler is None:
                        continue
                    tables, _is_arb = sampler(points, entries=entries)
                    # A cached batch can mix arb and mpmath rows, in
                    # which case its aggregate is_arb flag is false.
                    values = [[_arb_to_mpf(tables[k][m][n]) if isinstance(tables[k][m][n], arb) else tables[k][m][n]
                               for m, n in F_pairs] for k in range(len(doc_sp))]
                finally:
                    ctx.prec = before
            else:
                values = [[tables[k][m][n] for m, n in F_pairs] for k in range(len(doc_sp))]
            # Native real components pass through unchanged. Validate complex
            # residuals at the original precision before projecting them.
            with mp.workprec(validation_prec):
                values = [[_pmp_real_value(value, where="clean block sampling") for value in row]
                          for row in values]
            factors = [clearing_factor_value(groups, x) for x in doc_sp] if groups else None
            out[i] = [[(value * factors[k] if factors is not None else value) / sc[k]
                       for value in row] for k, row in enumerate(values)]
    return out


def _exact_operator_vector(gap_problem, spin_star, delta_star):
    """The crossing vector V* of one operator ``(spin_star, delta_star)``, from its exact point block."""
    from .point_constraint import build_point_channel

    spec = gap_problem.spec
    point_channel = build_point_channel(
        int(spin_star), delta_star, coordinate=spec.coordinate, h_ext=spec.h_ext, c=spec.c,
        q_order=spec.q_order, lam=spec.lam, chiral_norm=spec.chiral_norm,
        z_shift=-2 * spec.h_ext, zbar_shift=-2 * spec.h_ext,
    )
    # the point channel's table is shifted to Delta*: evaluate it there (at 0 it carries base^(-Delta*))
    return _extract_vector(point_channel.prefactored, gap_problem.F_pairs, delta_star, coordinate=spec.coordinate)


def _extract_vector(block: SerializedDerivTable, F_pairs, Delta, *, coordinate: str):
    table = block.evaluate(coordinate, Delta)
    return tuple(
        _canonicalize_real_output(table[m][n])
        for (m, n) in F_pairs
    )


def _exact_identity_vector(exact_identity, F_pairs):
    """Shift-independent identity normalization vector at full precision.

    The physical identity contribution is finite at `Delta = 0`, but the raw
    exact symbolic block still carries the removable vacuum pole there. We
    strip the removable factors first (the same canonicalization the
    serializer runs) and then evaluate the regular table at `Delta = 0`
    exactly. With the physical kinematic factor z^{h-d1-d2} already inside
    the block, no additional cross-ratio prefactor is needed: N^0 = 1 at
    the identity (Delta = 0).
    """
    stripped = strip_removable_poles_from_exact_table(exact_identity)
    table = stripped.evaluate(mp.mpf(0))
    return tuple(
        _canonicalize_real_output(table[m][n])
        for (m, n) in F_pairs
    )


def _regularized_identity_grid(
    c,
    h_ext,
    lam,
    q_order,
    *,
    coordinate: str,
    chiral_norm,
):
    """Identity derivative grid at a singular c from the regularized chiral path.

    Build the vacuum chiral Taylor coefficients directly from the exact
    Laurent-in-t regularized chiral block at ``h = 0`` and assemble the
    ``ell = 0`` scalar block as ``G = F(z) F(zbar)``. This stays entirely on
    the well-conditioned chiral path and avoids both the unstable eps-sampled
    numeric regularization and the large-cancellation symbolic-in-Delta table.
    """
    h_ext = _as_real_label(h_ext, name="h_ext")
    h_externals = [h_ext] * 4
    z_shift = -2 * h_ext

    if coordinate in {"zzb", "xt"}:
        z_of_u = lambda u: u
        u_symmetric = z_symmetric()
    elif coordinate in {"yyb", "ws"}:
        z_of_u = z_of_y
        u_symmetric = y_symmetric()
    else:
        raise ValueError(
            "singular-c identity vector supports coordinates {'zzb', 'yyb', 'xt', 'ws'}"
        )

    evaluate_block = _fixed_h_block_evaluator_regularized(
        mp.mpf(0),
        h_externals,
        c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
        z_shift=z_shift,
        working_precision=True,     # the blocks' one Laurent table, at the build's working precision
    )
    coeffs = taylor_coefficients(
        lambda u: evaluate_block(z_of_u(u)),
        u_symmetric,
        lam,
    )
    from virasoro_blocks._conventions import scalar_G_spin0_weight

    if scalar_G_spin0_weight() == 2:
        uubar_table = [
            [2 * coeffs[m] * coeffs[n] for n in range(lam + 1)]
            for m in range(lam + 1)
        ]
    else:
        uubar_table = [
            [coeffs[m] * coeffs[n] for n in range(lam + 1)]
            for m in range(lam + 1)
        ]
    if coordinate in {"xt", "ws"}:
        return _fold_parity(uubar_table, 0, lam, lam)
    return uubar_table


def _numeric_identity_vector(
    c, h_ext, lam, q_order, F_pairs, *, coordinate: str, chiral_norm, eps=None
):
    """Well-conditioned exact vacuum normalization vector.

    The symbolic-in-Delta table stores each block as ``P(Delta)/D(Delta)`` whose
    coefficients reach ~1e500+ (the q-recursion Kac-pole products), so evaluating
    it at the vacuum needs dps well above the physically required precision to
    resolve cancellation at high ``lam`` or large ``c``. The numeric recursion block stays
    O(1) and dps-stable and cancels the removable h=0 pole for identical
    externals, so it evaluates the vacuum at ``Delta = 0`` directly.

    Away from singular central charges, pass ``eps`` explicitly to sample a
    nearby nonzero dimension instead. At recognized singular
    central charges, :func:`_regularized_identity_grid` instead evaluates the
    Laurent-regularized chiral vacuum directly at h = 0; ``eps`` is not used.
    """
    if eps is None:
        eps = 0
    c = _central_charge_input(c)

    def _build_grid(c_val):
        delta = _as_real_label(eps, name="eps")
        return _numeric_scalar_grid(coordinate, delta, h_ext, lam, c=c_val, q_order=q_order, chiral_norm=chiral_norm)

    if recognized_singular_locus(c) is not None:
        table = _regularized_identity_grid(
            c,
            h_ext,
            lam,
            q_order,
            coordinate=coordinate,
            chiral_norm=chiral_norm,
        )
    else:
        table = _build_grid(c)
    return tuple(
        _canonicalize_real_output(table[m][n])
        for (m, n) in F_pairs
    )


def self_dual_identity_measure_value(
    c, h_ext, lam, q_order, *, coordinate: str, chiral_norm=None, eps=None
):
    """Return the vacuum block contribution to the self-dual moment measure.

    This is the `(0, 0)` entry of the same vacuum derivative grid used for the
    crossing identity vector. It is generally not equal to one in the Virasoro
    normalization used by the block tables. At generic central charge an
    explicit ``eps`` samples that dimension instead of the exact vacuum.
    """
    if eps is None:
        eps = 0
    c = _central_charge_input(c)
    if recognized_singular_locus(c) is not None:
        table = _regularized_identity_grid(
            c,
            h_ext,
            lam,
            q_order,
            coordinate=coordinate,
            chiral_norm=chiral_norm,
        )
    else:
        table = _numeric_scalar_grid(coordinate, _as_real_label(eps, name="eps"), h_ext, lam, c=c, q_order=q_order,
                                     chiral_norm=chiral_norm)
    return _canonicalize_real_output(table[0][0])


def _gap_precision_request(spec):
    """The spec's :class:`~virasoro_blocks.verified_build.PrecisionRequest` (``dps``, ``target_digits``,
    ``max_dps``), in the spec's own field names."""
    from virasoro_blocks.verified_build import BOOTSTRAP_KNOBS, PrecisionRequest

    return PrecisionRequest.make(spec.dps, spec.target_digits, spec.max_dps, knobs=BOOTSTRAP_KNOBS)


def _precision_key(spec):
    """What the precision part of a prepared build's reuse key compares: the working precision, or -- with a
    target, where the working precision is the build's to choose -- the target and cap."""
    if spec.target_digits is None:
        return ("dps", int(spec.dps))
    return ("target", int(spec.target_digits), None if spec.max_dps is None else int(spec.max_dps))


def _validate_gap_spec_common(
    spec: IdenticalScalarGapSpec,
) -> tuple[tuple[int, ...], str, object, object, object]:
    # parsed at the spec's precision, never the caller's ambient one (a build at mp.dps = 15 recorded
    # h_ext = 0.14999999999999999445 in its manifest and wrote 15-digit block files)
    with mp.workdps(int(spec.dps)):
        return _validate_gap_spec_common_at_dps(spec)


def _check_spec_field_types(spec):
    """Type errors in a spec surfaced as raw TypeErrors deep in the build (lam=3.5, q_order='12'), and a truthy
    string ('no') passed as True."""
    import numbers

    for name in ("lam", "q_order", "dps", "num_threads", "n_sample_points", "positive_pole_npoles",
                 "large_spin_npoles", "target_digits", "max_dps"):
        value = getattr(spec, name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, numbers.Integral)):
            raise ValueError(f"{name} must be an integer, got {value!r}")
    for name in ("shared_sample_points", "explicit_sample_points"):
        if not isinstance(getattr(spec, name), bool):
            raise ValueError(f"{name} must be True or False, got {getattr(spec, name)!r}")
    if spec.chiral_norm is not None:
        norm = _as_real_label(spec.chiral_norm, name="chiral_norm")
        if norm <= 0 or norm != int(norm):
            # The manifest stores the chiral normalization as an integer.
            raise ValueError(
                f"chiral_norm must be a positive integer N (blocks ~ (16 q / N)^h; None means 16), got "
                f"{spec.chiral_norm!r}: the bootstrap records and checks N as an integer, so a block table written "
                "with a non-integer --chiral-norm cannot be used here"
            )


_REGULARIZE_CHOICES = ("auto", "on", "off")
_SAMPLE_VIA_CHOICES = ("numeric", "symbolic")


def _check_spec_choices(spec) -> None:
    """Refuse a named choice outside its set, a negative tower pole count, and a repeated tower twist."""
    from virasoro_blocks.interpolation import POLE_SELECTIONS

    for name in ("pole_selection", "large_spin_pole_selection"):
        value = getattr(spec, name)
        if value not in POLE_SELECTIONS:
            raise ValueError(f"{name} must be one of {', '.join(POLE_SELECTIONS)}; got {value!r}")
    if spec.regularize not in _REGULARIZE_CHOICES:
        raise ValueError(f"regularize must be 'auto', 'on', or 'off'; got {spec.regularize!r}")
    if spec.sample_via not in _SAMPLE_VIA_CHOICES:
        raise ValueError(f"sample_via must be 'numeric' or 'symbolic'; got {spec.sample_via!r}")
    npoles = spec.large_spin_npoles
    if npoles is not None and (isinstance(npoles, bool) or not isinstance(npoles, numbers.Integral) or npoles < 0):
        raise ValueError(f"large_spin_npoles must be a non-negative integer or None; got {npoles!r}")
    from virasoro_blocks.numeric_input import literal_number

    seen = {}
    for twist in spec.large_spin_twists or ():
        try:
            label = literal_number(twist)
        except (TypeError, ValueError, ArithmeticError):
            continue    # an invalid label is reported by the full validation, with its name
        if label in seen:
            raise ValueError(
                f"large_spin_twists lists the twist {twist!r} twice (as {seen[label]!r} and {twist!r}): each twist "
                "is one tower slice, so a repeat duplicates its positivity block"
            )
        seen[label] = twist


def _validate_gap_spec_common_at_dps(spec):
    _check_spec_field_types(spec)
    _check_spec_choices(spec)
    if spec.lam < 1:
        raise ValueError("lam must be >= 1 for the odd crossing sector")
    if spec.q_order < 1:
        raise ValueError("q_order must be >= 1")
    if spec.dps < 10:
        raise ValueError("dps must be >= 10")
    _gap_precision_request(spec)      # validates target_digits / max_dps against dps
    if spec.num_threads < 1:
        raise ValueError("num_threads must be >= 1")
    _normalize_npoles(spec.npoles)  # validates: non-negative int or "all"
    if spec.n_sample_points is not None and spec.n_sample_points < 1:
        raise ValueError("n_sample_points must be >= 1")
    if spec.positive_pole_npoles is not None and spec.positive_pole_npoles < 0:
        raise ValueError("positive_pole_npoles must be >= 0")
    if spec.coordinate not in _IDENTICAL_SCALAR_CROSSING_COORDINATES:
        raise ValueError(
            "identical-scalar coordinate must be one of "
            "{'zzb', 'yyb', 'xt', 'ws'}"
        )

    if spec.positive_pole_selection not in ("rightmost", "nearest", "residue", "residue_all"):
        # (read only where a positive-real-pole lowering happens, c <= 1, so a typo went unnoticed elsewhere)
        raise ValueError(
            f"positive_pole_selection must be 'rightmost', 'nearest', 'residue' or 'residue_all', "
            f"got {spec.positive_pole_selection!r}"
        )
    spin_list = _validate_even_spins(spec.spin_list)
    _resolved_point_constraints(spec)
    _resolved_large_spin_tower(spec, spin_list)
    coordinate = spec.coordinate
    c = _central_charge_input(spec.c)
    h_ext = _as_real_label(spec.h_ext, name="h_ext")
    prefactor_exponent = _as_real_label(
        h_ext if spec.prefactor_exponent is None else spec.prefactor_exponent,
        name="prefactor_exponent",
    )
    # Identical-scalar crossing fixes the prefactor: every block uses
    # z_shift = zbar_shift = -2 h_ext.
    if prefactor_exponent != h_ext:
        from virasoro_blocks.writers import _metadata_num_str

        raise ValueError(
            f"prefactor_exponent = {_metadata_num_str(prefactor_exponent, spec.dps + 5)} != h_ext = "
            f"{_metadata_num_str(h_ext, spec.dps + 5)}: the "
            "identical-scalar crossing equation fixes the prefactor at h_ext (blocks carry z_shift = -2 h_ext); "
            "leave prefactor_exponent unset"
        )
    return spin_list, coordinate, c, h_ext, prefactor_exponent


def _validate_identical_scalar_operator(ell, delta, *, name: str) -> tuple[int, object]:
    """Check one isolated operator ``(ell, Delta)`` exchanged between identical scalars.

    Require a nonnegative even integer spin and ``Delta >= ell``, the
    unitarity bound. Return the validated ``(ell, Delta)``.
    """
    try:
        ell = _integer_field(f"{name} spin", ell)
    except ValueError:
        raise ValueError(f"{name} spin must be an integer, got {ell!r}") from None
    if ell < 0 or ell % 2:
        raise ValueError(
            f"{name} spin must be a non-negative even integer for identical "
            f"scalars, got {ell}"
        )
    delta = _as_real_label(delta, name=f"{name} Delta")
    if delta < ell:
        raise ValueError(
            f"{name} Delta = {mp.nstr(_convert_number(delta), 8)} is below the unitarity bound "
            f"{ell} for spin {ell}"
        )
    return ell, delta


def _resolved_point_constraints(spec) -> tuple[tuple[int, object], ...]:
    """``spec.point_constraints`` as validated ``(ell, Delta)`` pairs (call at the spec's precision).

    ``Delta`` retains its exact input label until numerical evaluation.
    """
    resolved = []
    for point in spec.point_constraints or ():
        try:
            ell, delta = point
        except (TypeError, ValueError):
            raise ValueError(f"point_constraints entries must be (ell, Delta) pairs, got {point!r}") from None
        resolved.append(_validate_identical_scalar_operator(ell, delta, name="point constraint"))
    return tuple(resolved)


def _resolved_large_spin_tower(spec, spin_list) -> tuple[int | None, tuple[object, ...]]:
    """The large-spin tower floor and twists of ``spec``, validated (call at the spec's precision).

    Returns ``(None, ())`` without a tower, else ``(large_spin_lmin, twists)``.  The floor is the first spin the
    tower covers (every slice imposes positivity for all ``L >= large_spin_lmin``), so it must be a non-negative
    integer no larger than ``max(spin_list) + 2`` to cover the spins above the
    fixed-spin range. A lower floor overlaps that range, which is valid, except
    for a floor of 0: the tower is analytic in L, so its L = 0 member is the
    doubled block ``2 B(h, h)``, not the spin-0 ``G = B(h, h)``, and spin 0
    belongs in ``spin_list``.  The floor and twist list must be supplied together.
    """
    twists = []
    for twist in spec.large_spin_twists or ():
        value = _as_real_label(twist, name="large_spin_twists entry")
        if value < 0:
            raise ValueError(f"large_spin_twists entries must be >= 0 (unitarity), got {twist}")
        twists.append(value)
    lmin = spec.large_spin_lmin
    if lmin is None:
        if twists:
            raise ValueError(
                "large_spin_twists needs large_spin_lmin (the tower covers spins L >= large_spin_lmin); without it "
                "the twists were ignored and no tower was built"
            )
        return None, ()
    if not twists:
        raise ValueError(
            f"large_spin_lmin = {lmin!r} needs large_spin_twists (one tower slice per twist, for example "
            "virasoro_bootstrap.large_spin.double_twist_set(h_ext, 5)); without them no tower was built"
        )
    try:
        floor = _integer_field("large_spin_lmin", _as_real_label(lmin, name="large_spin_lmin"))
    except (TypeError, ValueError, ArithmeticError):
        floor = -1
    if floor < 0:
        raise ValueError(
            f"large_spin_lmin must be a non-negative integer spin (the tower covers L >= large_spin_lmin), "
            f"got {lmin!r}"
        )
    if floor == 0 and scalar_G_spin0_convention() != "double":
        raise ValueError(
            f"large_spin_lmin = {lmin!r} starts the tower at spin 0: the tower is analytic in L, so its L = 0 member "
            "is the doubled block 2 B(h, h), not the spin-0 G = B(h, h); list spin 0 in spin_list and start the "
            "tower at large_spin_lmin >= 1 (the usual choice is max(spin_list) + 2)"
        )
    first_uncovered = max(spin_list) + 2
    if floor > first_uncovered:
        gap_spins = list(range(first_uncovered, min(floor, first_uncovered + 12), 2))
        raise ValueError(
            f"large_spin_lmin = {floor} leaves the spins {gap_spins}{' ...' if floor > first_uncovered + 12 else ''} "
            f"unconstrained: spin_list ends at {max(spin_list)}, so the tower must start at L <= {first_uncovered} "
            "(the usual choice is large_spin_lmin = max(spin_list) + 2), or spin_list must extend up to it"
        )
    return floor, tuple(twists)


def _validate_delta_gap(delta_gap):
    delta_gap = _as_real_label(delta_gap, name="delta_gap")
    if delta_gap < 0:
        raise ValueError("delta_gap must be >= 0")
    return delta_gap


def _validate_delta_min_for_spin(delta_min, ell: int):
    delta_min = _as_real_label(delta_min, name=f"delta_min(ell={ell})")
    unitary = _default_delta_min(ell)
    if ell == 0:
        if delta_min < 0:
            raise ValueError("scalar lower edge must be >= 0")
        return delta_min
    if delta_min < unitary:
        raise ValueError(
            f"lower edge for ell = {ell} must be >= the unitarity bound {unitary}"
        )
    return delta_min


def _resolved_delta_min_by_ell(
    spec: IdenticalScalarGapSpec,
    spin_list: tuple[int, ...],
) -> dict[int, object]:
    with mp.workdps(int(spec.dps)):
        return _resolved_delta_min_by_ell_at_dps(spec, spin_list)


def _resolved_delta_min_by_ell_at_dps(spec, spin_list):
    resolved = {ell: _default_delta_min(ell) for ell in spin_list}
    resolved[0] = _validate_delta_gap(spec.delta_gap)

    overrides = spec.delta_min_by_spin or {}
    for raw_ell, raw_delta_min in overrides.items():
        ell = _integer_field("delta_min_by_spin key", raw_ell)
        if ell not in resolved:
            raise ValueError(
                f"delta_min_by_spin contains ell = {ell}, which is not in spin_list"
            )
        value = _validate_delta_min_for_spin(raw_delta_min, ell)
        if ell == 0 and value != resolved[0]:
            raise ValueError(
                f"delta_min_by_spin[0] = {mp.nstr(value, 12)} conflicts with delta_gap = "
                f"{mp.nstr(resolved[0], 12)}: the scalar gap is delta_gap (the override silently replaced it)"
            )
        resolved[ell] = value
    return resolved


def _canonical_non_scalar_delta_min_overrides(
    delta_min_by_ell: dict[int, object],
) -> dict[int, object] | None:
    overrides = {
        ell: delta_min
        for ell, delta_min in delta_min_by_ell.items()
        if ell != 0 and delta_min != _default_delta_min(ell)
    }
    return overrides or None


# What each entry of a prepared build's reuse key is, in order (named in the error when a build is refused).
_REUSE_KEY_FIELDS = (
    "coordinate", "c", "h_ext", "prefactor_exponent", "lam", "spin_list", "q_order",
    "dps (or target_digits and max_dps)", "npoles", "n_sample_points", "pole_selection", "positive_pole_npoles",
    "positive_pole_selection", "regularize", "chiral_norm", "shared_sample_points", "large_spin_lmin",
    "large_spin_twists", "large_spin_pole_selection", "large_spin_npoles", "point_constraints", "sample_via",
    "scalar_G_spin0",
)


def _prepared_reuse_key(
    spec: IdenticalScalarGapSpec,
    *,
    spin_list: tuple[int, ...],
    c,
    h_ext,
    prefactor_exponent,
) -> tuple:
    from virasoro_blocks.recursion import _resolve_chiral_norm

    lmin, twists = _resolved_large_spin_tower(spec, spin_list)
    key = (
        spec.coordinate,
        c,
        h_ext,
        prefactor_exponent,
        spec.lam,
        spin_list,
        spec.q_order,
        _precision_key(spec),
        _normalize_npoles(spec.npoles),
        spec.n_sample_points,
        spec.pole_selection,
        spec.positive_pole_npoles,
        spec.positive_pole_selection,
        spec.regularize,
        # the normalization the blocks use: None is the library's 16, and a manifest round trip records it as 16
        int(_resolve_chiral_norm(spec.chiral_norm)),
        spec.shared_sample_points,
        lmin,
        twists,
        spec.large_spin_pole_selection,
        _resolved_large_spin_npoles(spec),
        # a prepared build without these would silently drop the point channels / sample the wrong way
        repr(_resolved_point_constraints(spec)),
        spec.sample_via,
        # the spin-0 normalization of the blocks and identity vector (the active one when the key is made)
        scalar_G_spin0_convention(),
    )
    return key


def _reuse_key_mismatch(prepared_key: tuple, requested_key: tuple) -> str:
    """The fields in which a prepared build's key differs from the requested spec's, with both values."""
    if len(prepared_key) != len(requested_key):
        return "the prepared build's key has another layout (it was made by an older version)"
    parts = [
        f"{name} (prepared {built!r}, requested {wanted!r})"
        for name, built, wanted in zip(_REUSE_KEY_FIELDS, prepared_key, requested_key)
        if built != wanted
    ]
    return "; ".join(parts)


def _build_spin_channel_results(
    exact_tables: dict[int, object],
    spin_list: Sequence[int],
    *,
    delta_min_by_ell: dict[int, object],
    dps: int,
    coordinate: str,
    resolved_method: str,
    npoles: int | None,
    n_sample_points: int | None,
    pole_selection: str,
    sample_via: str,
    chiral_norm,
    num_threads: int,
    shared_sample_points: bool = False,
    precision_request=None,
):
    build_jobs = _spin_channel_jobs(
        exact_tables, spin_list, delta_min_by_ell=delta_min_by_ell, dps=dps, coordinate=coordinate,
        resolved_method=resolved_method, npoles=npoles, n_sample_points=n_sample_points,
        pole_selection=pole_selection, sample_via=sample_via, chiral_norm=chiral_norm,
        shared_sample_points=shared_sample_points, precision_request=precision_request,
    )
    if not build_jobs:
        return []
    if usable_workers(num_threads) == 1 or len(build_jobs) == 1:
        return [_build_spin_channel_task(job) for job in build_jobs]
    with ProcessPoolExecutor(max_workers=min(num_threads, len(build_jobs)), **pool_initializer_kwargs()) as pool:
        return list(pool.map(_build_spin_channel_task, build_jobs))


def _spin_channel_jobs(
    exact_tables, spin_list, *, delta_min_by_ell, dps, coordinate, resolved_method, npoles, n_sample_points,
    pole_selection, sample_via, chiral_norm, shared_sample_points=False, precision_request=None,
):
    """The picklable per-spin jobs of :func:`_build_spin_channel_results`."""
    return [
        (
            dps,
            coordinate,
            ell,
            exact_tables[ell],
            delta_min_by_ell[ell],
            resolved_method,
            npoles,
            n_sample_points,
            pole_selection,
            sample_via,
            chiral_norm,
            shared_sample_points,
            precision_request,
        )
        for ell in spin_list
    ]


def _point_channels(spec, *, coordinate, c, h_ext):
    """Point-constraint channels (gap-independent): each block frozen at delta_point by the numeric evaluator."""
    points = _resolved_point_constraints(spec)
    if not points:
        return ()
    from .point_constraint import build_point_channel

    return tuple(
        build_point_channel(
            ell, delta_point, coordinate=coordinate, h_ext=h_ext,
            c=c, q_order=spec.q_order, lam=spec.lam,
            chiral_norm=spec.chiral_norm,
            z_shift=-2 * h_ext,
            zbar_shift=-2 * h_ext,
        )
        for ell, delta_point in points
    )


def _manifest_block_hashes(paths):
    """``{resolved path: content_hash}`` from the manifest next to each block directory (``<dir>/../manifest.json``)."""
    hashes = {}
    for manifest in {path.resolve().parent.parent / "manifest.json" for path in paths}:
        if manifest.is_file():
            with open(manifest) as f:
                doc = json.load(f)
            for ref in doc.get("blocks") or ():
                if "path" in ref and "content_hash" in ref:
                    hashes[(manifest.parent / ref["path"]).resolve()] = ref["content_hash"]
    return hashes


def _is_manifest_block(doc, path, manifest_hashes):
    """Whether a manifest vouches for ``doc``: its recorded content hash matches, including a pre-1.2 hash."""
    from .manifest import block_content_hash_matches

    expected = manifest_hashes.get(Path(path).resolve())
    return expected is not None and block_content_hash_matches(doc, expected, legacy=True)


def _file_label_dps(doc, dps):
    """The digits a block file's decimal labels can be compared at: ``dps``, or fewer when the file was written
    at a lower precision.

    A writer rounds a non-terminating label (``z_shift = -2/15``) to at most its ``precision_input`` digits, so a
    30-digit file serves a 62-digit spec only if its labels are compared at 30 digits.
    """
    from virasoro_blocks.serialization import _file_integer

    digits = int(dps)
    for key in ("precision_input", "precision_actual"):
        if doc.get(key) is not None:
            digits = min(digits, _file_integer(doc, key, 1))
            break
    return digits


def _same_file_label(value, expected, dps):
    """Whether a block file's label ``value`` denotes ``expected``, up to roundoff at ``dps`` digits.

    Files store labels as decimals (a non-terminating rational such as 1/3 is rounded to the metadata digits) or
    as ``p/q``; either form matches the spec's exact label within ``10^-(dps/2)`` relative.  Pass the file's own
    precision (:func:`_file_label_dps`), not the spec's, when the file may have been written at fewer digits.
    """
    from virasoro_blocks.numeric_input import literal_number

    if value is None or isinstance(value, bool):
        return False
    try:
        value, expected = literal_number(value), literal_number(expected)
        if value == expected:
            return True
        with mp.workdps(dps):
            a, b = mp.convert(value), mp.convert(expected)
            return bool(abs(a - b) <= mp.mpf(10) ** (-(dps // 2)) * max(1, abs(b)))
    except (TypeError, ValueError, ArithmeticError):
        return False


def _check_keep_all_poles(doc, path, spec, *, c, h_ext, coordinate, expected_basis):
    """A reduced file cannot satisfy keep-all merely because its interpolation layout is valid.

    A fixed-spin file declares keep-all itself: its writer records no ``pole_selection`` and ``npoles`` equal to
    the poles it kept. A tower file (whose writer always records a selection) is counted against the chiral roots.
    """
    from virasoro_blocks.serialization import _file_integer

    kept = len(doc.get("pole_list_x") or ())
    if doc.get("variable") == "L" or expected_basis == "chiral_F":
        for key in ("precision_actual", "precision_input"):
            if doc.get(key) is not None:
                _file_integer(doc, key, 1)
        dps = int(doc.get("precision_actual") or doc.get("precision_input") or spec.dps)
        from virasoro_blocks.large_spin import _virasoro_chiral_roots

        with mp.workdps(dps):
            count = len(_virasoro_chiral_roots((h_ext,) * 4, c, spec.q_order))
        if kept != count:
            raise ValueError(f"block file {path} keeps {kept} poles, but the spec's keep-all request requires "
                             f"{count}; rebuild the table with every pole retained")
        return
    recorded = doc.get("npoles")
    selection = doc.get("pole_selection")
    if selection is not None or not _same_file_label(recorded, kept, int(spec.dps)):
        raise ValueError(f"block file {path} keeps {kept} poles of an order-reduced table (npoles {recorded}, "
                         f"pole_selection {selection}), but the spec's keep-all request requires every pole; "
                         "rebuild the table with every pole retained")


def _check_scalar_G_spin0(doc, path, expected_basis):
    """Refuse a spin-0 scalar-``G`` file whose normalization is unrecorded or not this process's.

    A file without ``scalar_G_spin0``, or with another normalization, combined with this process's identity
    vector would mis-scale the scalar channel (gap bounds are immune; OPE, fixed-OPE, and moment bounds are
    not).
    """
    from fractions import Fraction
    from virasoro_blocks._conventions import SCALAR_G_SPIN0_KEY, scalar_G_spin0_convention
    from virasoro_blocks.numeric_input import parse_number

    if not str(expected_basis).endswith("scalar_G") or doc.get("variable", "Delta") not in ("Delta", "point"):
        return
    try:
        spin_zero = parse_number(doc.get("spin")) == Fraction(0)
    except (TypeError, ValueError, ArithmeticError):
        return      # an invalid spin is reported with the other metadata problems
    if not spin_zero:
        return
    recorded, active = doc.get(SCALAR_G_SPIN0_KEY), scalar_G_spin0_convention()
    if recorded == active:
        return
    what = ("records no scalar_G_spin0 normalization" if recorded is None
            else f"was built with scalar_G_spin0 = {recorded!r}, but this process uses {active!r}")
    raise ValueError(f"spin-0 block file {path} {what}; regenerate the table with the current virasoro_blocks "
                     "(an old spin-0 table and a new identity vector would double the scalar channel)")


def _check_block_doc(doc, path, spec, *, c, h_ext, coordinate, manifest_block=False, expected_basis="scalar_G",
                     verify_keep_all=True):
    """Refuse a block file whose own metadata disagrees with ``spec`` (it describes a different block).

    ``manifest_block``: a manifest vouches for the file (its content hash matches), so its crossing shift is the
    manifest's ``-2 h_ext`` whatever the file records (manifest files written before 2026-09-28 record 0).
    """
    problems = []
    if doc.get("variable", "Delta") == "Delta" and expected_basis not in {"chiral_F", "global_chiral_F"}:
        from fractions import Fraction
        from virasoro_blocks.numeric_input import parse_number

        try:
            spin = parse_number(doc.get("spin"))
            valid_spin = isinstance(spin, Fraction) and spin.denominator == 1 and spin >= 0 and spin % 2 == 0
        except (TypeError, ValueError, ArithmeticError):
            valid_spin = False
        if not valid_spin:
            problems.append(f"spin {doc.get('spin')!r} (the identical-scalar problem needs a non-negative even integer)")
    if doc.get("basis") != expected_basis:
        problems.append(f"basis {doc.get('basis')!r} (the identical-scalar problem uses {expected_basis})")
    _check_scalar_G_spin0(doc, path, expected_basis)
    if str(expected_basis).endswith("scalar_G"):
        # an odd tower or odd-spin table must record that it is the z <-> zbar-even block
        from virasoro_blocks._conventions import check_scalar_G_odd_record

        check_scalar_G_odd_record(doc, f"block file {path}")
    if coordinate not in (doc.get("coordinates") or ()):
        problems.append(f"coordinates {doc.get('coordinates')} lack {coordinate!r}")
    from virasoro_blocks.numeric_input import literal_number

    # compared at the file's precision: a 30-digit file's rounded labels serve a higher-precision spec
    label_dps = _file_label_dps(doc, spec.dps)

    def same_label(value, expected):
        return _same_file_label(value, expected, label_dps)

    same_c = same_label(doc.get("central_charge"), c)
    if not expected_basis.startswith("global_") and not same_c:
        problems.append(f"central_charge {doc.get('central_charge')} != {mp.nstr(c, 12)}")
    # a regularized (merged-pole) table and a direct one are different blocks at a singular c; files that predate
    # the record are not checked
    policy = doc.get("regularization_policy")
    if not expected_basis.startswith("global_") and policy in ("merged", "direct"):
        wanted = "merged" if _resolved_regularize(c, spec.regularize) else "direct"
        if policy != wanted:
            problems.append(f"regularization_policy {policy!r}, but spec.regularize={spec.regularize!r} needs "
                            f"{wanted!r} blocks (CLI: --regularize {'on' if wanted == 'merged' else 'auto'})")
    h_label = literal_number(spec.h_ext)
    chiral = expected_basis in {"chiral_F", "global_chiral_F"}
    for key in ("h_external", "hbar_external"):
        if doc.get(key) is not None or (chiral and key == "h_external"):
            if len(doc.get(key) or ()) != 4 or any(not same_label(h, h_label) for h in doc[key]):
                problems.append(f"{key} {doc.get(key)} != four identical weights {h_label}")
    if not chiral and (len(doc.get("delta_external") or ()) != 4 or any(
        not same_label(d, 2 * h_label) for d in doc["delta_external"]
    )):
        problems.append(f"delta_external {doc.get('delta_external')} != 2 h_ext = {2 * h_label}")
    for key in () if manifest_block else ("z_shift", "zbar_shift"):
        if not same_label(doc.get(key, 0), -2 * h_label):
            problems.append(
                f"{key} {doc.get(key, 0)} != -2 h_ext = {-2 * h_label} at the file's {label_dps} digits (the table "
                f"was built for another crossing shift; the CLI writes this one with --u-shift {-2 * h_label})")
    # the crossing equation is expanded about z = zbar = 1/2: a table about another point is a different block set
    for key in ("z_center", "zbar_center"):
        if doc.get(key) is not None and not same_label(doc[key], mp.mpf(1) / 2):
            problems.append(f"{key} {doc[key]} (the identical-scalar crossing equation is expanded about 1/2)")
    if doc.get("lambda") is not None and not same_label(doc["lambda"], spec.lam):
        problems.append(f"lambda {doc.get('lambda')} != spec.lam {spec.lam}")
    if doc.get("order") is not None and not same_label(doc["order"], spec.q_order):
        problems.append(f"order {doc.get('order')} != spec.q_order {spec.q_order}")
    from virasoro_blocks.recursion import _resolve_chiral_norm

    if doc.get("chiral_normalization_N") is not None and not same_label(
        doc["chiral_normalization_N"], _resolve_chiral_norm(spec.chiral_norm)
    ):
        problems.append(f"chiral_normalization_N {doc['chiral_normalization_N']} != spec.chiral_norm "
                        f"{_resolve_chiral_norm(spec.chiral_norm)}")
    # the kept poles: their count and the ranking that chose them (files that predate the record skip the ranking)
    tower = doc.get("variable") == "L"
    npoles = _resolved_large_spin_npoles(spec) if tower else _normalize_npoles(spec.npoles)
    selection = spec.large_spin_pole_selection if tower else spec.pole_selection
    npoles_name = "large_spin_npoles" if tower else "npoles"
    if doc.get("variable", "Delta") in ("Delta", "L"):
        # Integer caps require interpolation nodes. Keep-all requests can use either an exact CLI table
        # or a bootstrap table interpolated with every pole retained.
        interpolated = bool(doc.get("sample_points_x"))
        if npoles is not None and not interpolated:
            problems.append(f"an exact (npoles 'all') table, but spec.{npoles_name} is {npoles}")
        if (spec.n_sample_points is not None and interpolated and not tower
                and len(doc["sample_points_x"]) != spec.n_sample_points):
            problems.append(f"{len(doc['sample_points_x'])} sample points != spec.n_sample_points {spec.n_sample_points}")
    if isinstance(npoles, int) and doc.get("variable", "Delta") != "point":
        if doc.get("npoles") is not None and not same_label(doc["npoles"], npoles):
            problems.append(f"npoles {doc['npoles']} != spec.{npoles_name} {npoles}")
        if doc.get("pole_selection") is not None and doc["pole_selection"] != selection:
            problems.append(f"pole_selection {doc['pole_selection']!r} != spec's {selection!r}")
    if problems:
        raise ValueError(f"block file {path} does not match the spec: " + "; ".join(problems))
    if (verify_keep_all and npoles is None and expected_basis in {"scalar_G", "chiral_F"}
            and doc.get("variable", "Delta") in {"Delta", "L"}):
        _check_keep_all_poles(doc, path, spec, c=c, h_ext=h_ext,
                              coordinate=coordinate, expected_basis=expected_basis)


def _tower_block_twist(doc, path, *, lmin):
    """Validate a tower's exact labels before selecting its stored table."""
    from virasoro_blocks.numeric_input import literal_number

    try:
        twist = _as_real_label(doc.get("twist"), name="tower twist")
        floor = _as_real_label(doc.get("large_spin_lmin"), name="large_spin_lmin")
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise ValueError(f"tower block {path} has invalid twist/large_spin_lmin labels: {exc}") from exc
    if twist < 0:
        raise ValueError(f"tower block {path} has a negative twist {twist}")
    if lmin is None or floor != literal_number(lmin):
        raise ValueError(f"tower block {path} covers L >= {doc.get('large_spin_lmin')}, but "
                         f"spec.large_spin_lmin is {lmin}")
    return twist


def _file_backed_exact_tables(spec, ells, *, c, h_ext, coordinate, used_regularize, resolved_method):
    """The exact (skeleton) tables of ``ells`` for a build whose other tables came from files."""
    return _build_exact_scalar_tables(
        c, h_ext, coordinate, tuple(ells), spec.lam, spec.q_order,
        used_regularize=used_regularize, resolved_method=resolved_method,
        chiral_norm=spec.chiral_norm, num_threads=spec.num_threads, sample_via=spec.sample_via,
        pole_selection=spec.pole_selection,
    )


def _fixed_spin_sampler_for(spec, spin_list, exact_tables, *, c, h_ext, coordinate, used_regularize, resolved_method):
    """``sampler_for(index)`` of :func:`_clean_sample_values_by_block`: the numeric sampler of fixed-spin channel
    ``index`` (its exact table built from the block engine, and kept, when the build holds none)."""
    def sampler_for(index):
        if index >= len(spin_list):
            return None     # a tower or point channel
        ell = spin_list[index]
        if ell not in exact_tables:
            exact_tables.update(_file_backed_exact_tables(
                spec, (ell,), c=c, h_ext=h_ext, coordinate=coordinate,
                used_regularize=used_regularize, resolved_method=resolved_method,
            ))
        return getattr(exact_tables[ell], "numeric_sampler", None)
    return sampler_for


def _spin_channels_at_edges(spec, exact_tables, ells, delta_min_by_ell, *, coordinate, resolved_method):
    """Fixed-spin channels of ``ells`` interpolated from ``exact_tables`` at their edges ``delta_min_by_ell``, at the
    working precision ``spec.dps``, verified (a shortfall is warned about; an edge rebuild does not escalate)."""
    from virasoro_blocks.recursion import _resolve_chiral_norm
    from .point_constraint import _crossing_context_for_spec

    request = _gap_precision_request(spec)
    results = _build_spin_channel_results(
        exact_tables, tuple(ells), delta_min_by_ell=delta_min_by_ell, dps=spec.dps, coordinate=coordinate,
        resolved_method=resolved_method, npoles=_normalize_npoles(spec.npoles),
        n_sample_points=spec.n_sample_points, pole_selection=spec.pole_selection, sample_via=spec.sample_via,
        chiral_norm=_resolve_chiral_norm(spec.chiral_norm), num_threads=spec.num_threads,
        shared_sample_points=spec.shared_sample_points, precision_request=request,
    )
    channels = {
        ell: IdenticalScalarSpinChannel(ell=ell, delta_min=delta_min, serialized=serialized, prefactored=prefactored,
                                       crossing_context=_crossing_context_for_spec(spec, resolved_method))
        for ell, delta_min, serialized, prefactored in results
    }
    _warn_block_shortfalls(list(channels.values()), request, npoles=spec.npoles, work=int(spec.dps))
    return channels


def prepare_identical_scalar_gap_build_from_blocks(
    spec: IdenticalScalarGapSpec, blocks, *, _block_reader=None,
) -> PreparedIdenticalScalarGapBuild:
    """A prepared build whose tables come from block JSON files (CLI output, or an earlier manifest's ``blocks/``).

    ``blocks`` is a directory (of ``derivs_*.json`` files, or a manifest directory holding ``blocks/``) or an
    iterable of paths.  Every file is checked against ``spec`` by
    its own metadata (basis, coordinate, central charge, externals, ``z_shift = -2 h_ext``, lambda, order, chiral
    normalization N, regularization, the kept-pole count and selector; for the large-spin tower its twist and
    ``large_spin_lmin``) instead of by a content hash, a label the file rounded being compared at the file's own
    ``precision_input``, so a block set can be reused
    for any problem it describes: another gap, another objective (``build_identical_scalar_ope_max_problem``, the
    moment builders), a subset of its spins.  Pass the result as ``prepared=`` to any problem builder.  A spin whose
    requested ``delta_min`` differs from its file's anchor (a higher or a lower gap) is built fresh at the new edge
    by the block engine, as an in-memory ``prepared`` build re-interpolates it: its SDP data are then numeric block
    values at its own new nodes, never the file polynomial re-evaluated elsewhere. Every other spin and the tower
    come from the files; point constraints are built fresh from the spec. A file keeps its own precision
    (60-digit files serve a 90-digit spec with 60-digit values) and its own node sets
    (``shared_sample_points`` applies to rebuilt spins only). Two
    files for one spin or one twist are refused; tower files are unused when the spec has no tower.
    """
    from virasoro_blocks.serialization import serialized_table_from_json
    from virasoro_blocks.numeric_input import parse_number
    from .point_constraint import _crossing_context_for_spec

    spin_list, coordinate, c, h_ext, prefactor_exponent = _validate_gap_spec_common(spec)
    if isinstance(blocks, (str, Path)):
        blocks = Path(blocks)
        if (blocks / "manifest.json").is_file() and (blocks / "blocks").is_dir():
            blocks = blocks / "blocks"
        paths = sorted(blocks.glob("derivs_*.json"))
    else:
        paths = [Path(p) for p in blocks]
    # Manifest reconstruction supplies a reader pinned to its original refs;
    # ordinary file reuse instead checks the files' own metadata.
    manifest_hashes = _manifest_block_hashes(paths) if _block_reader is None else {}
    with mp.workdps(spec.dps):
        lmin, twists = _resolved_large_spin_tower(spec, spin_list)
        fixed, tower = {}, {}
        for path in paths:
            if _block_reader is None:
                with open(path) as f:
                    doc = json.load(f)
            else:
                doc = _block_reader(path)
            variable = doc.get("variable", "Delta")
            if variable not in ("Delta", "L"):
                continue
            if variable == "L" and not twists:
                continue    # No tower was requested, so these files do not contribute to this build.
            _check_block_doc(doc, path, spec, c=c, h_ext=h_ext, coordinate=coordinate,
                             manifest_block=_block_reader is not None or _is_manifest_block(doc, path, manifest_hashes))
            if variable == "Delta" and int(parse_number(doc["spin"])) in spin_list:
                _check_block_target(doc, path, spec)
            table = serialized_table_from_json(doc)
            if variable == "Delta":
                ell = int(parse_number(doc["spin"]))
                if ell in fixed:
                    raise ValueError(f"two block files for spin {ell}: {fixed[ell][1]} and {path}")
                fixed[ell] = (table, path)
            else:
                twist = _tower_block_twist(doc, path, lmin=lmin)
                if twist in tower:
                    raise ValueError(f"two tower block files for twist {mp.nstr(twist, 12)}: {tower[twist][1]} and {path}")
                tower[twist] = (table, path, _file_label_dps(doc, spec.dps))
        fixed = {ell: table for ell, (table, _path) in fixed.items()}
        missing = [ell for ell in spin_list if ell not in fixed]
        if missing:
            raise ValueError(f"no block file for spins {missing} among {len(paths)} files")
        tower_channels = []
        for twist in twists:
            match = next((table for key, (table, _path, label_dps) in tower.items()
                          if _same_file_label(key, twist, label_dps)), None)
            if match is None:
                raise ValueError(f"no large-spin block file for twist {mp.nstr(twist, 12)}")
            tower_channels.append(IdenticalScalarSpinChannel(
                ell=lmin, delta_min=mp.mpf(lmin), serialized=match, prefactored=match, twist=twist))
        resolved_method = _resolved_method(spec.npoles)
        F_pairs = tuple(F_minus_pairs(spec.lam, coordinate=coordinate))
        from virasoro_blocks.recursion import _resolve_chiral_norm
        identity_vector = _numeric_identity_vector(
            c, h_ext, spec.lam, spec.q_order, F_pairs, coordinate=coordinate,
            chiral_norm=_resolve_chiral_norm(spec.chiral_norm),
        )
        return PreparedIdenticalScalarGapBuild(
            reuse_key=_prepared_reuse_key(spec, spin_list=spin_list, c=c, h_ext=h_ext,
                                          prefactor_exponent=prefactor_exponent),
            resolved_method=resolved_method,
            used_regularize=_resolved_regularize(c, spec.regularize),
            resolved_positive_pole_npoles=_resolved_positive_pole_npoles(
                c, resolved_method, _normalize_npoles(spec.npoles), spec.positive_pole_npoles),
            prefactor_exponent=prefactor_exponent,
            F_pairs=F_pairs,
            identity_vector=identity_vector,
            exact_tables={},
            prepared_spin_channels=tuple(
                IdenticalScalarSpinChannel(ell=ell, delta_min=_as_real_label(fixed[ell].shift, name="block shift"), serialized=fixed[ell],
                                           prefactored=fixed[ell], crossing_context=_crossing_context_for_spec(spec, resolved_method))
                for ell in sorted(fixed) if ell in spin_list
            ),
            large_spin_channels=tuple(tower_channels),
            point_channels=_point_channels(spec, coordinate=coordinate, c=c, h_ext=h_ext),
            working_dps=int(spec.dps),
        )


def _check_block_target(doc, path, spec) -> None:
    """Check a loaded fixed-spin block against ``spec.target_digits``.

    When a target is set, the file must record sufficient ``verified_digits``.
    No accuracy requirement is imposed when the target is unset.
    """
    if spec.target_digits is None:
        return
    request = _gap_precision_request(spec)
    verified = doc.get("verified_digits")
    try:
        digits = None if verified is None else float(verified)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"block file {path} has invalid verified_digits {verified!r}") from exc
    if digits is None or math.isnan(digits) or request.short(digits):
        shown = "records no verified_digits" if verified is None else f"verifies to {verified} digits"
        raise ValueError(
            f"block file {path} {shown}, short of spec.target_digits {spec.target_digits}: the file cannot honor "
            "the target; rebuild the problem from the spec (its build escalates the precision until every block "
            "delivers it)")


def prepare_identical_scalar_gap_build(
    spec: IdenticalScalarGapSpec,
) -> PreparedIdenticalScalarGapBuild:
    """Precompute the gap-independent identical-scalar block data.

    The exact tables depend only on the CFT data and truncation settings, not
    on the bootstrap lower edges. We therefore build those exact tables once,
    together with one prepared set of serialized/prefactored channels, and let
    callers reuse any channel whose requested lower edge is unchanged while
    rebuilding only the sectors whose lower edge moved.

    Everything is built at the working precision ``spec.dps`` and every fixed-spin block verified
    (:func:`_verify_spin_channel`); blocks whose SDP data are wrong raise one ``RuntimeWarning`` naming the ``dps``
    that fixes them.  With ``spec.target_digits`` the whole build is repeated at a working precision raised by the
    measured deficit until every fixed-spin block delivers the target
    (:func:`virasoro_blocks.verified_build.run_verified_build`), or it raises naming why it cannot; the result's
    ``working_dps`` is the precision its data are written at.
    """
    from virasoro_blocks.verified_build import run_verified_build

    request = _gap_precision_request(spec)
    _validate_gap_spec_common(spec)
    if request.target_digits is None:
        prepared = _prepare_identical_scalar_gap_build_at(spec, int(spec.dps), request)
        _warn_block_shortfalls(prepared.prepared_spin_channels, request, npoles=spec.npoles, work=int(spec.dps),
                               tower_channels=prepared.large_spin_channels,
                               tower_npoles=_resolved_large_spin_npoles(spec))
        return prepared

    def verify(prepared, work):
        return _binding_verification(prepared.prepared_spin_channels, request)

    prepared, _verification, _work = run_verified_build(
        lambda work: _prepare_identical_scalar_gap_build_at(spec, work, request), verify, request=request,
        what="the identical-scalar blocks", order_reduced=_normalize_npoles(spec.npoles) is not None, report=False,
    )
    return prepared


def _prepare_identical_scalar_gap_build_at(spec, work: int, request) -> PreparedIdenticalScalarGapBuild:
    """:func:`prepare_identical_scalar_gap_build` at working precision ``work`` (the reuse key is ``spec``'s)."""
    key_spec = spec
    spec = _dataclass_replace(spec, dps=int(work))
    spin_list, coordinate, c, h_ext, prefactor_exponent = _validate_gap_spec_common(spec)
    delta_min_by_ell = _resolved_delta_min_by_ell(spec, spin_list)

    with mp.workdps(spec.dps):
        used_regularize = _resolved_regularize(c, spec.regularize)
        resolved_method = _resolved_method(spec.npoles)
        resolved_positive_pole_npoles = _resolved_positive_pole_npoles(
            c,
            resolved_method,
            _normalize_npoles(spec.npoles),
            spec.positive_pole_npoles,
        )
        F_pairs = tuple(F_minus_pairs(spec.lam, coordinate=coordinate))
        from virasoro_blocks.recursion import _resolve_chiral_norm
        resolved_chiral_norm = _resolve_chiral_norm(spec.chiral_norm)
        # q-order convergence gate (warn-only): estimate the q-truncation
        # accuracy of the blocks this build is about to produce. Low q_order
        # corrupts blocks silently and no amount of dps can fix it, so
        # surface the measured digits before the (expensive) build. Cached
        # per (c, externals, q_order, precision); unavailable probes (no
        # flint) skip silently.
        from virasoro_blocks.qgate import gate_q_convergence
        gate_q_convergence(
            c=c, h_externals=[h_ext] * 4, q_order=spec.q_order,
            n_terms=int(spec.lam) + 2, coordinate=coordinate,
            chiral_norm=spec.chiral_norm, mode="warn",
            label="identical-scalar build",
        )
        spin_channel_kwargs = dict(
            delta_min_by_ell=delta_min_by_ell,
            dps=spec.dps,
            coordinate=coordinate,
            resolved_method=resolved_method,
            npoles=_normalize_npoles(spec.npoles),
            n_sample_points=spec.n_sample_points,
            pole_selection=spec.pole_selection,
            sample_via=spec.sample_via,
            chiral_norm=resolved_chiral_norm,
            shared_sample_points=spec.shared_sample_points,
            precision_request=request,
        )
        # Large-spin asymptotic positivity tower (gap-independent -> built once
        # here, reused across the whole binary search).
        tower_jobs = []
        lmin, twists = _resolved_large_spin_tower(spec, spin_list)
        if lmin is not None:
            from .large_spin import build_large_spin_channels, large_spin_channel_jobs, _large_spin_channel_task
            tower_kwargs = dict(
                mode="virasoro", c=c, h_ext=h_ext, lam=spec.lam,
                q_order=spec.q_order, npoles=_resolved_large_spin_npoles(spec),
                coordinate=coordinate,
                Lmax=lmin, twists=twists,
                chiral_norm=spec.chiral_norm,
                pole_selection=spec.large_spin_pole_selection,
                shared_sample_points=spec.shared_sample_points,
                sample_via=spec.sample_via,
            )
            tower_jobs = large_spin_channel_jobs(**tower_kwargs)

        def _exact_tables():
            return _build_exact_scalar_tables(
                c,
                h_ext,
                coordinate,
                spin_list,
                spec.lam,
                spec.q_order,
                used_regularize=used_regularize,
                resolved_method=resolved_method,
                chiral_norm=spec.chiral_norm,
                num_threads=spec.num_threads,
                sample_via=spec.sample_via,
                pole_selection=spec.pole_selection,
            )

        def _identity_vector():
            return _numeric_identity_vector(
                c,
                h_ext,
                spec.lam,
                spec.q_order,
                F_pairs,
                coordinate=coordinate,
                chiral_norm=resolved_chiral_norm,
            )

        if usable_workers(spec.num_threads) > 1 and len(spin_list) + len(tower_jobs) > 1:
            # One pool for every independent table: the tower's twist slices are queued first (they need nothing
            # from the parent), the parent builds the exact tables meanwhile, then the spin channels are queued and
            # the identity vector is computed while the workers run -- no pool drains before the next one fills.
            with ProcessPoolExecutor(max_workers=spec.num_threads, **pool_initializer_kwargs()) as pool:
                tower_futures = [pool.submit(_large_spin_channel_task, job) for job in tower_jobs]
                exact_tables = _exact_tables()
                spin_futures = [
                    pool.submit(_build_spin_channel_task, job)
                    for job in _spin_channel_jobs(exact_tables, spin_list, **spin_channel_kwargs)
                ]
                identity_vector = _identity_vector()
                prepared_results = [future.result() for future in spin_futures]
                large_spin_channels = tuple(future.result() for future in tower_futures)
        else:
            exact_tables = _exact_tables()
            prepared_results = _build_spin_channel_results(
                exact_tables, spin_list, num_threads=spec.num_threads, **spin_channel_kwargs,
            )
            identity_vector = _identity_vector()
            large_spin_channels = (
                build_large_spin_channels(**tower_kwargs, num_threads=spec.num_threads) if tower_jobs else ()
            )
        prepared_channels = []
        from .point_constraint import _crossing_context_for_spec

        crossing_context = _crossing_context_for_spec(spec, resolved_method)
        for ell, delta_min, serialized, prefactored in prepared_results:
            prepared_channels.append(
                IdenticalScalarSpinChannel(
                    ell=ell,
                    delta_min=delta_min,
                    serialized=serialized,
                    prefactored=prefactored,
                    crossing_context=crossing_context,
                )
            )

        point_channels = _point_channels(spec, coordinate=coordinate, c=c, h_ext=h_ext)
        _warn_if_c_le_1_degree(c, prepared_channels, exact_tables, delta_min_by_ell)

        return PreparedIdenticalScalarGapBuild(
            reuse_key=_prepared_reuse_key(
                key_spec,
                spin_list=spin_list,
                c=c,
                h_ext=h_ext,
                prefactor_exponent=prefactor_exponent,
            ),
            resolved_method=resolved_method,
            used_regularize=used_regularize,
            resolved_positive_pole_npoles=resolved_positive_pole_npoles,
            prefactor_exponent=prefactor_exponent,
            F_pairs=F_pairs,
            identity_vector=identity_vector,
            exact_tables=exact_tables,
            prepared_spin_channels=tuple(sorted(prepared_channels, key=lambda channel: channel.ell)),
            large_spin_channels=large_spin_channels,
            point_channels=point_channels,
            working_dps=int(work),
        )


def build_identical_scalar_gap_problem(
    spec: IdenticalScalarGapSpec,
    *,
    prepared: PreparedIdenticalScalarGapBuild | None = None,
) -> IdenticalScalarGapProblem:
    """Build a minimal identical-scalar gap feasibility problem.

    The produced PMP uses the odd crossing sector on the selected crossing-odd
    coordinate, objective zero, and the identity block's odd crossing vector as
    the normalization condition.
    The positivity blocks are generated for each even spin in `spec.spin_list`.
    """
    spin_list, coordinate, c, h_ext, prefactor_exponent = _validate_gap_spec_common(spec)
    prepared_key = _prepared_reuse_key(
        spec,
        spin_list=spin_list,
        c=c,
        h_ext=h_ext,
        prefactor_exponent=prefactor_exponent,
    )
    if prepared is None:
        prepared = prepare_identical_scalar_gap_build(spec)
    elif getattr(prepared, "scalar_G_spin0", None) != scalar_G_spin0_convention():
        raise ValueError(
            f"prepared identical-scalar gap build was built with scalar_G_spin0 = "
            f"{getattr(prepared, 'scalar_G_spin0', None)!r}, but this process uses {scalar_G_spin0_convention()!r}: "
            "its spin-0 blocks and identity vector would mix two normalizations of the scalar channel; prepare "
            "the build again under the active convention"
        )
    elif prepared.reuse_key != prepared_key:
        raise ValueError(
            "prepared identical-scalar gap build is incompatible with the requested spec: it differs in "
            + _reuse_key_mismatch(prepared.reuse_key, prepared_key)
        )
    requested_dps = int(spec.dps)
    work = int(prepared.working_dps or spec.dps)
    if work < int(spec.dps):
        raise ValueError(f"the prepared build is at working precision {work}, below spec.dps {spec.dps}")
    # everything downstream (the edge rebuilds, the PMP, the manifest, the SDPB precision) at the working precision
    spec = _dataclass_replace(spec, dps=work)
    spin_list, coordinate, c, h_ext, prefactor_exponent = _validate_gap_spec_common(spec)
    delta_min_by_ell = _resolved_delta_min_by_ell(spec, spin_list)

    with mp.workdps(spec.dps):
        lmin, twists = _resolved_large_spin_tower(spec, spin_list)

        # Reuse only the same input label. A deliberate small edge change
        # changes the positivity domain even when its numerical values round alike.
        spin_channels_by_ell = {
            channel.ell: channel
            for channel in prepared.prepared_spin_channels
            if channel.ell in delta_min_by_ell
            and _as_real_label(channel.delta_min, name="prepared lower edge") == delta_min_by_ell[channel.ell]
        }
        missing_ells = tuple(
            ell for ell in spin_list if ell not in spin_channels_by_ell
        )
        absent = tuple(ell for ell in missing_ells if ell not in prepared.exact_tables)
        if absent:
            # a file-backed build (prepare_identical_scalar_gap_build_from_blocks) holds no exact tables: a spin whose
            # edge moved is built fresh from the block engine, as in memory -- its values sampled numerically at
            # the new nodes, never its file polynomial re-evaluated there -- and its exact table kept for later edges
            prepared.exact_tables.update(_file_backed_exact_tables(
                spec, absent, c=c, h_ext=h_ext, coordinate=coordinate,
                used_regularize=prepared.used_regularize, resolved_method=prepared.resolved_method,
            ))
        if missing_ells:
            spin_channels_by_ell.update(_spin_channels_at_edges(
                spec, prepared.exact_tables, missing_ells, delta_min_by_ell,
                coordinate=coordinate, resolved_method=prepared.resolved_method,
            ))

        identity_vector = prepared.identity_vector
        objective = [mp.mpf(0)] * len(prepared.F_pairs)
        # Fixed-spin channels, then the gap-independent large-spin tower and
        # point-constraint channels (all built once in `prepare`). They feed the
        # PMP and certification in this order; the manifest writer classifies
        # them by kind.
        spin_channels = (
            tuple(spin_channels_by_ell[ell] for ell in spin_list)
            + tuple(prepared.large_spin_channels)
            + tuple(prepared.point_channels)
        )
        prefactored_blocks = [channel.prefactored for channel in spin_channels]
        pmp_build = build_prefactored_pmp_doc(
            prefactored_blocks,
            prepared.F_pairs,
            coordinate=coordinate,
            objective=objective,
            normalization=identity_vector,
            positive_pole_npoles=prepared.resolved_positive_pole_npoles,
            positive_pole_selection=spec.positive_pole_selection,
            explicit_sample_points=spec.explicit_sample_points,
        )
        sample_values_by_block = _clean_sample_values_by_block(
            spin_channels, pmp_build.doc, prepared.F_pairs, coordinate=coordinate,
            lowerings=pmp_build.positive_pole_lowerings,
            sampling_blocks=pmp_build.positivity_blocks,
            sampler_for=_fixed_spin_sampler_for(
                spec, spin_list, prepared.exact_tables, c=c, h_ext=h_ext, coordinate=coordinate,
                used_regularize=prepared.used_regularize, resolved_method=prepared.resolved_method,
            ),
        )

        return IdenticalScalarGapProblem(
            spec=IdenticalScalarGapSpec(
                c=c,
                h_ext=spec.h_ext,
                delta_gap=delta_min_by_ell[0],
                coordinate=coordinate,
                delta_min_by_spin=_canonical_non_scalar_delta_min_overrides(delta_min_by_ell),
                lam=spec.lam,
                spin_list=spin_list,
                q_order=spec.q_order,
                dps=spec.dps,
                target_digits=spec.target_digits,
                max_dps=spec.max_dps,
                npoles=spec.npoles,
                n_sample_points=spec.n_sample_points,
                pole_selection=spec.pole_selection,
                sample_via=spec.sample_via,
                positive_pole_npoles=prepared.resolved_positive_pole_npoles,
                positive_pole_selection=spec.positive_pole_selection,
                regularize=spec.regularize,
                prefactor_exponent=prefactor_exponent,
                chiral_norm=spec.chiral_norm,
                explicit_sample_points=spec.explicit_sample_points,
                num_threads=spec.num_threads,
                shared_sample_points=spec.shared_sample_points,
                # Keep domain and operator labels exact for backend dispatch and reuse.
                large_spin_lmin=lmin,
                large_spin_twists=twists,
                large_spin_pole_selection=spec.large_spin_pole_selection,
                large_spin_npoles=spec.large_spin_npoles,
                point_constraints=_resolved_point_constraints(spec),
            ),
            resolved_method=prepared.resolved_method,
            used_regularize=prepared.used_regularize,
            prefactor_exponent=prepared.prefactor_exponent,
            F_pairs=prepared.F_pairs,
            identity_vector=identity_vector,
            spin_channels=spin_channels,
            pmp_build=pmp_build,
            large_spin_channels=tuple(prepared.large_spin_channels),
            point_channels=tuple(prepared.point_channels),
            sample_values_by_block=sample_values_by_block,
            problem_kind="gap",
            objective_vector=tuple(objective),
            normalization_vector=tuple(identity_vector),
            normalization_operator=None,
            exact_tables_by_ell=prepared.exact_tables,
            requested_dps=requested_dps,
            scalar_G_spin0=prepared.scalar_G_spin0,
        )


def build_identical_scalar_ope_max_problem(
    spec: IdenticalScalarGapSpec,
    *,
    delta_star: object | None = None,
    spin_star: int = 0,
    prepared: PreparedIdenticalScalarGapBuild | None = None,
) -> IdenticalScalarGapProblem:
    """Build an OPE-coefficient *maximization* problem (extremal, not feasibility).

    This reuses the identical-scalar gap build verbatim -- the same prefactored
    spin-channel blocks, identity vector, F_pairs, prefactor poles and sample
    points -- and differs from :func:`build_identical_scalar_gap_problem` only in
    the PMP's two scalar vectors:

      * **normalization** = the distinguished operator's crossing vector
        ``V_*`` (the block of spin ``spin_star`` evaluated at ``delta_star``),
        which fixes the functional value on that operator to 1, and
      * **objective** = the identity crossing vector ``V_id``.

    SDPB is then run *to optimality* (no ``--findPrimalFeasible`` /
    ``--findDualFeasible``); the optimal objective value gives a rigorous bound
    on the squared OPE coefficient of the distinguished operator. Convert it via
    :func:`ope_squared_bound_from_objective` (the sign/normalization mapping is
    validated against gap behavior).

    ``delta_star`` defaults to the continuum bottom of the ``spin_star`` channel
    (i.e. the gap), the standard "OPE coefficient of the gap-saturating
    operator". The distinguished operator sits at the bottom of its spin's
    continuum, so ``alpha . V_* = 1`` is consistent with the continuum positivity
    ``alpha . V_O >= 0`` it also satisfies.
    """
    from .moments import at_working_precision

    spec, _kw = at_working_precision(spec, {"prepared": prepared})
    prepared = _kw["prepared"]
    with mp.workdps(spec.dps):
        spin_star, delta_star_val = _validate_identical_scalar_operator(
            spin_star,
            spin_star if delta_star is None else delta_star,
            name="distinguished operator",
        )
    gap_problem = build_identical_scalar_gap_problem(spec, prepared=prepared)

    # Only a fixed-spin Delta channel can carry the distinguished operator: a
    # large-spin tower channel is parametrized by spin at fixed twist and a point
    # channel is frozen at its own Delta, so matching their ``ell`` would build
    # V_* from the wrong block.
    n_fixed = (
        len(gap_problem.spin_channels)
        - len(gap_problem.large_spin_channels)
        - len(gap_problem.point_channels)
    )
    fixed_channels = gap_problem.spin_channels[:n_fixed]
    channel = next(
        (ch for ch in fixed_channels if ch.ell == spin_star), None
    )
    if channel is None:
        raise ValueError(
            f"spin_star={spin_star} is not among the problem's fixed spins "
            f"{tuple(ch.ell for ch in fixed_channels)}"
        )

    coordinate = gap_problem.spec.coordinate
    F_pairs = gap_problem.F_pairs
    with mp.workdps(spec.dps):
        if delta_star is None:
            delta_star_val = channel.delta_min
        # the exact point block: the channel's table is an order-reduced interpolant, and below its edge an
        # extrapolation (V* 2e-3 off at Delta* = 0.1 under a 1.5 gap, the OPE^2 bound 0.7%)
        v_star = _exact_operator_vector(gap_problem, spin_star, delta_star_val)
        objective = list(gap_problem.identity_vector)
        prefactored_blocks = [ch.prefactored for ch in gap_problem.spin_channels]
        pmp_build = build_prefactored_pmp_doc(
            prefactored_blocks,
            F_pairs,
            coordinate=coordinate,
            objective=objective,
            normalization=list(v_star),
            positive_pole_npoles=gap_problem.spec.positive_pole_npoles,
            positive_pole_selection=spec.positive_pole_selection,
            explicit_sample_points=spec.explicit_sample_points,
        )

    return _dataclass_replace(
        gap_problem,
        pmp_build=pmp_build,
        problem_kind="ope_max",
        objective_vector=tuple(objective),
        normalization_vector=tuple(v_star),
        normalization_operator={"spin_star": int(spin_star), "delta_star": delta_star_val},
    )


def build_identical_scalar_ope_bound_problem(
    spec: IdenticalScalarGapSpec,
    *,
    delta_star: object,
    spin_star: int = 0,
    direction: str = "upper",
    prepared: PreparedIdenticalScalarGapBuild | None = None,
) -> IdenticalScalarGapProblem:
    """Build an upper/lower bound on one separated OPE-squared coefficient.

    The distinguished operator is evaluated as a frozen point block but is not
    included in the positivity cone. This is the right setup for an isolated
    exchanged operator below the continuum gap: all *other* operators are
    constrained positive, while the normalization fixes the functional value on
    the separated operator to ``+1`` (upper bound) or ``-1`` (lower bound).
    """
    from .moments import at_working_precision

    spec, _kw = at_working_precision(spec, {"prepared": prepared})
    prepared = _kw["prepared"]
    if direction not in {"upper", "lower"}:
        raise ValueError("direction must be 'upper' or 'lower'")

    with mp.workdps(spec.dps):
        spin_star, delta_star_val = _validate_identical_scalar_operator(
            spin_star, delta_star, name="distinguished operator"
        )
    gap_problem = build_identical_scalar_gap_problem(spec, prepared=prepared)
    coordinate = gap_problem.spec.coordinate
    F_pairs = gap_problem.F_pairs
    with mp.workdps(spec.dps):
        v_star = _exact_operator_vector(gap_problem, spin_star, delta_star_val)
        normalization = (
            tuple(v_star)
            if direction == "upper"
            else tuple(-mp.mpf(v) for v in v_star)
        )
        objective = list(gap_problem.identity_vector)
        prefactored_blocks = [ch.prefactored for ch in gap_problem.spin_channels]
        pmp_build = build_prefactored_pmp_doc(
            prefactored_blocks,
            F_pairs,
            coordinate=coordinate,
            objective=objective,
            normalization=list(normalization),
            positive_pole_npoles=gap_problem.spec.positive_pole_npoles,
            positive_pole_selection=spec.positive_pole_selection,
            explicit_sample_points=spec.explicit_sample_points,
        )

    return _dataclass_replace(
        gap_problem,
        pmp_build=pmp_build,
        problem_kind=f"ope_{direction}",
        objective_vector=tuple(objective),
        normalization_vector=tuple(normalization),
        normalization_operator={
            "spin_star": int(spin_star),
            "delta_star": delta_star_val,
            "direction": direction,
        },
    )


def build_identical_scalar_fixed_ope_problem(
    spec: IdenticalScalarGapSpec,
    *,
    delta_star: object,
    ope_squared: object,
    spin_star: int = 0,
    prepared: PreparedIdenticalScalarGapBuild | None = None,
) -> IdenticalScalarGapProblem:
    """Build a feasibility problem with one isolated OPE coefficient fixed.

    ``ope_squared`` is the operator's ``p = N^Delta lambda^2`` against the identity's 1, in any spin channel
    (see :func:`ope_squared_bound_from_objective`).
    """
    from .moments import at_working_precision

    spec, _kw = at_working_precision(spec, {"prepared": prepared})
    prepared = _kw["prepared"]
    with mp.workdps(spec.dps):
        spin_star, delta_star_val = _validate_identical_scalar_operator(
            spin_star, delta_star, name="distinguished operator"
        )
        coeff_label = _as_real_label(ope_squared, name="ope_squared")
    gap_problem = build_identical_scalar_gap_problem(spec, prepared=prepared)
    coordinate = gap_problem.spec.coordinate
    F_pairs = gap_problem.F_pairs
    with mp.workdps(spec.dps):
        v_star = _exact_operator_vector(gap_problem, spin_star, delta_star_val)
        known_vector = _known_vector_sum(((1, gap_problem.identity_vector), (coeff_label, v_star)))
        objective = [mp.mpf(0)] * len(F_pairs)
        prefactored_blocks = [ch.prefactored for ch in gap_problem.spin_channels]
        pmp_build = build_prefactored_pmp_doc(
            prefactored_blocks,
            F_pairs,
            coordinate=coordinate,
            objective=objective,
            normalization=list(known_vector),
            positive_pole_npoles=gap_problem.spec.positive_pole_npoles,
            positive_pole_selection=spec.positive_pole_selection,
            explicit_sample_points=spec.explicit_sample_points,
        )

    return _dataclass_replace(
        gap_problem,
        identity_vector=known_vector,
        pmp_build=pmp_build,
        problem_kind="fixed_ope_feasibility",
        objective_vector=tuple(objective),
        normalization_vector=tuple(known_vector),
        normalization_operator={
            "spin_star": int(spin_star),
            "delta_star": delta_star_val,
            "ope_squared": coeff_label,
        },
    )


def read_sdpb_objective(sdpb_out_dir) -> dict[str, mp.mpf] | None:
    """Read ``primalObjective`` / ``dualObjective`` from an SDPB ``out.txt``.

    Returns ``{"primal": ..., "dual": ...}`` (mpf), or ``None`` if the file or
    fields are absent. The values are returned AS SDPB REPORTS THEM, which
    already include ``sdp/objectives.json``'s constant offset -- do not add it
    again.

    That offset is not physical. The PMP imposes a normalization
    ``alpha . nm = 1``, which SDPB's format cannot express as an affine
    constraint on the dual vector, so :mod:`sdp_writer` eliminates the pivot
    component ``piv = argmax|nm|``::

        alpha . obj = obj[piv]/nm[piv]  +  sum_{a != piv} b[a] alpha[a]
                    = constant         +  b . alpha_free

    i.e. ``constant`` is the objective evaluated at the base point of the
    affine slice, a bookkeeping artifact of WHICH component was eliminated.

    Adding it double-counts it (verified by solving one ``sdp/`` directory with and without the constant: the
    dual objective moves by exactly the constant).  For OPE-max problems the pivot ``argmax|V_*|`` can change with
    delta_star, so a double-counted constant jumps between adjacent points and can even make an OPE^2 bound
    negative.
    """
    sdpb_out_dir = Path(sdpb_out_dir)
    out_txt = sdpb_out_dir / "out.txt"
    if not out_txt.exists():
        return None
    values: dict[str, mp.mpf] = {}
    for line in out_txt.read_text(errors="replace").splitlines():
        lhs, separator, rhs = line.partition("=")
        if not separator:
            continue
        for field, key in (("primalObjective", "primal"), ("dualObjective", "dual")):
            if lhs.strip() == field:
                token = rhs.strip().rstrip(";").strip()
                if token:
                    # at the digits SDPB printed, not the caller's ambient precision (15 digits truncated the bound)
                    with mp.workdps(max(mp.mp.dps, sum(ch.isdigit() for ch in token.split("e")[0]) + 5)):
                        values[key] = mp.mpf(token)
    return values or None


def ope_squared_bound_from_objective(objective_value) -> mp.mpf:
    """Map an SDPB OPE-max optimal objective to the squared-OPE upper bound.

    With objective = identity vector ``V_id`` and normalization = ``V_*``, the
    crossing sum rule gives ``lambda_*^2 <= -alpha . V_id`` at the optimum, so
    the bound is the negated optimal objective. (Validated empirically.)
    Here ``lambda_*^2`` is the coefficient ``p = N^Delta lambda^2`` of the
    operator's block against the identity's 1, the same in every spin channel
    (the scalar G_{Delta,0} is B(h, h); see :mod:`virasoro_blocks.composer`).
    """
    value = objective_value if isinstance(objective_value, mp.mpf) else mp.mpf(_convert_number(objective_value))
    return mp.fneg(value, exact=True)


def ope_squared_lower_bound_from_objective(objective_value) -> mp.mpf:
    """Map the lower-OPE SDPB objective to a squared-OPE lower bound.

    The lower builder fixes ``alpha(V_*) = -1``. Then crossing and positivity
    imply ``lambda_*^2 >= alpha(V_id)``, and SDPB maximizes that objective.
    """
    return objective_value if isinstance(objective_value, mp.mpf) else mp.mpf(_convert_number(objective_value))


def write_identical_scalar_gap_problem(
    problem: IdenticalScalarGapProblem,
    output_dir,
) -> tuple[Path, Path]:
    """(see :func:`_write_identical_scalar_gap_problem`), written at the problem's precision whatever the ambient one."""
    with mp.workdps(int(problem.spec.dps)):
        return _write_identical_scalar_gap_problem(problem, output_dir)


def _write_identical_scalar_gap_problem(
    problem: IdenticalScalarGapProblem,
    output_dir,
) -> tuple[Path, Path]:
    """Write the bound inputs to disk as the reusable manifest plus ``pmp.json``.

    Emits the single canonical problem format -- ``manifest.json`` plus a
    ``blocks/`` directory of per-spin block files
    (:func:`virasoro_bootstrap.manifest.write_identical_scalar_manifest`) -- and
    the SDPB-ready ``pmp.json`` derived from it. The manifest is built directly
    from the in-memory ``problem``; it does not require the blocks to have been
    written as files beforehand (it writes them). Returns
    ``(manifest_path, pmp_path)``.
    """
    # Imported lazily to avoid a circular import: ``manifest`` imports from this
    # module (the spec/problem dataclasses and helpers).
    from .manifest import write_identical_scalar_manifest

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = write_identical_scalar_manifest(problem, output_dir)
    pmp_path = output_dir / "pmp.json"
    write_pmp_json(problem.pmp_build.doc, pmp_path)
    return manifest_path, pmp_path


def _dps_to_bits(dps: int) -> int:
    return max(448, math.ceil(dps * math.log2(10)) + 32)


def _first_matching_line(text: str, needles: Sequence[str]) -> str | None:
    for line in text.splitlines():
        for needle in needles:
            if needle in line:
                return line.strip()
    return None


# SDPB's terminate reasons, as it prints them ("-----<reason>-----"), and the status each one is reported as.
SDPB_TERMINATE_STATUSES = {
    "found primal-dual optimal solution": "optimal",
    "found primal feasible solution": "primal_feasible",
    "primal feasible jump detected": "primal_feasible",
    "found dual feasible solution": "dual_feasible",
    "dual feasible jump detected": "dual_feasible",
    "maxComplementarity exceeded": "max_complementarity",
    "maxIterations exceeded": "max_iterations",
    "maxRuntime exceeded": "max_runtime",
    "primal step too small": "primal_step_too_small",
    "dual step too small": "dual_step_too_small",
}


def _sdpb_terminate_line(log_text: str) -> str | None:
    """The first ``-----<terminate reason>-----`` line of an SDPB log that names a known reason."""
    for line in log_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("-----") and stripped.endswith("-----"):
            if stripped.strip("-").strip() in SDPB_TERMINATE_STATUSES:
                return stripped
    return None


def classify_sdpb_log_text(log_text: str) -> str | None:
    """The status of an SDPB solve from its log: SDPB's terminate reason as one of
    :data:`SDPB_TERMINATE_STATUSES`' values, or None when the log names none (SDPB did not finish).

    ``primal_feasible`` and ``dual_feasible`` are verdicts of a feasibility (gap) problem, ``optimal`` of an
    optimization (OPE or moment) problem; ``max_complementarity``, ``max_iterations``, ``max_runtime``,
    ``primal_step_too_small`` and ``dual_step_too_small`` stopped the solve without one.
    """
    line = _sdpb_terminate_line(log_text)
    return None if line is None else SDPB_TERMINATE_STATUSES[line.strip("-").strip()]


def _sdpb_error_summary(log_text: str) -> str | None:
    """The cause of a failed SDPB run from its log: the last line reporting an error, else the last line.

    SDPB's parameter echo (``primalErrorThreshold = ...``) is not an error.  A runner failure need not say
    "error" (``Cannot connect to the Docker daemon ...``), so the log's last line stands in for one.
    """
    lines = [line.strip() for line in log_text.splitlines() if line.strip()]
    for line in reversed(lines):
        lowered = line.lower()
        if "error" in lowered and "errorthreshold" not in lowered:
            return line
    return lines[-1] if lines else None


def _run_command(
    command: Sequence[str],
    log_path: Path | None,
    *,
    check: bool = True,
    cwd: Path | None = None,
) -> int:
    """Run *command* (optionally with working directory *cwd*), returning the exit code.

    If *check* is True (the default) a non-zero exit raises CalledProcessError.
    When False, the exit code is returned so the caller can decide how to react.
    """
    if log_path is None:
        result = subprocess.run(command, check=check, cwd=cwd)
        return result.returncode
    with open(log_path, "w") as f:
        result = subprocess.run(
            command, check=check, cwd=cwd, stdout=f, stderr=subprocess.STDOUT
        )
    return result.returncode


def sdpb_option_args(
    options: Sequence[str] | Mapping[str, str | bool | None],
) -> list[str]:
    """Normalize SDPB options into a flat command-line arg list.

    A mapping ``{flag: value}`` is flattened so that ``True`` emits a bare flag,
    a string (or other value) emits ``flag value``, and ``False``/``None`` are
    omitted. A plain sequence is returned as a list of strings unchanged.
    """
    if isinstance(options, Mapping):
        args: list[str] = []
        for flag, value in options.items():
            if value is False or value is None:
                continue
            args.append(str(flag))
            if value is not True:
                args.append(str(value))
        return args
    if isinstance(options, (str, bytes)):
        raise ValueError("sdpb_options must be a mapping or an iterable of argument tokens")
    try:
        args = iter(options)
    except TypeError:
        raise ValueError("sdpb_options must be a mapping or an iterable of argument tokens") from None
    return [str(arg) for arg in args]


# Two `sdpb_options` keys are interpreted specially rather than forwarded to
# sdpb verbatim: `--precision` sets the bit precision for *both* pmp2sdp and
# sdpb, and `--mpiProcs` (not a native sdpb flag) wraps sdpb in `mpirun -n N`.
_SDPB_PRECISION_OPTION = "--precision"
_SDPB_MPI_PROCS_OPTION = "--mpiProcs"


def _split_sdpb_options(
    options: Sequence[str] | Mapping[str, str | bool | int | None] | None,
    default_precision_bits: int,
) -> tuple[int, int | None, list[str]]:
    """Split SDPB options into ``(precision_bits, mpi_procs, sdpb_flag_args)``.

    ``--precision`` and ``--mpiProcs`` are pulled out for special handling, in
    either the two-token (``--precision 640``) or the equals form
    (``--precision=640``, also as a mapping key ``{"--precision=640": True}``);
    everything else, equals forms included, is returned verbatim as flat sdpb
    command-line args.
    """
    args = sdpb_option_args(options) if options is not None else []
    precision_bits = default_precision_bits
    mpi_procs: int | None = None
    flag_args: list[str] = []
    special = {_SDPB_PRECISION_OPTION: "precision", _SDPB_MPI_PROCS_OPTION: "mpi"}
    i = 0
    while i < len(args):
        token = args[i]
        name, eq, inline_value = token.partition("=")
        if name in special:
            if not eq and i + 1 == len(args):
                raise ValueError(f"{name} needs a positive integer value")
            raw = inline_value if eq else args[i + 1]
            try:
                value = int(raw)
            except ValueError:
                raise ValueError(f"{name} needs an integer, got {raw!r}") from None
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {raw!r}")
            if special[name] == "precision":
                precision_bits = value
            else:
                mpi_procs = value
            i += 1 if eq else 2
            continue
        flag_args.append(token)
        i += 1
    return precision_bits, mpi_procs, flag_args


def _sdpb_outcome(returncode: int, log_path: Path | None) -> tuple[str | None, str | None]:
    """``(sdpb_status, sdpb_summary)`` of a finished SDPB run.

    A nonzero exit is ``solver_error``, summarized by the exit code and the log's last error line (a stopped
    Docker daemon, an MPI failure, a malformed input: the cause is only in the log).  Otherwise the status is the
    log's terminate reason (:func:`classify_sdpb_log_text`) and the summary its terminate line.  Without a log
    (streamed to the terminal) a clean exit has no status.
    """
    text = None
    if log_path is not None and log_path.exists():
        text = log_path.read_text(errors="replace")
    if returncode != 0:
        summary = f"SDPB exited with code {returncode}"
        cause = None if text is None else _sdpb_error_summary(text)
        if cause is not None:
            summary += f": {cause}"
        elif log_path is not None:
            summary += f" (empty log {log_path})"
        return "solver_error", summary
    if text is None:
        return None, None
    status = classify_sdpb_log_text(text)
    if status is None:
        return None, _sdpb_error_summary(text) or "SDPB exited without reporting a terminate reason"
    return status, _sdpb_terminate_line(text)


def run_identical_scalar_gap_problem(
    problem: IdenticalScalarGapProblem,
    output_dir,
    *,
    image: str = _DOCKER_SDPB_IMAGE,
    runner: SdpbRunner | None = None,
    run_sdpb: bool = True,
    backend: str = "sdp",
    eval_dps: int | None = None,
    sdp_dir_name: str = "sdp",
    sdpb_out_dir_name: str = "sdpb_out",
    sdpb_options: Sequence[str] | Mapping[str, str | bool | int | None] | None = None,
    pmp2sdp_log_name: str | None = "pmp2sdp.log",
    sdpb_log_name: str | None = "sdpb.log",
    initial_checkpoint_dir: Path | str | None = None,
) -> DockerBootstrapRun:
    """Write the gap problem and invoke the SDP build + sdpb.

    The solver output is written to ``output_dir/sdpb_log_name`` (``pmp2sdp_log_name`` for pmp2sdp), from which
    ``sdpb_status`` and ``sdpb_summary`` are read; pass ``None`` to stream it to the terminal instead (the status
    is then not classified).

    How the SDPB toolchain is executed is delegated to *runner* (an
    :class:`~virasoro_bootstrap.hpc.runner.SdpbRunner`); when omitted it is
    resolved from the ``VB_SDPB_MODE`` / ``VB_SDPB_IMAGE`` / ``VB_SDPB_BIN_DIR``
    environment (default: docker with the upstream image, the historical
    behavior). *image* is the docker-mode image tag and is kept for backward
    compatibility; a non-default value overrides the environment.

    ``backend`` selects how the SDPB ``sdp`` directory is produced:

      * ``"sdp"`` (default): write it directly from the in-memory PMP doc via
        :func:`virasoro_bootstrap.sdp_writer.write_sdp_directory`, building the
        per-sample-point data (from the numeric block values) and bilinear
        bases at ``eval_dps`` digits, by default the working precision
        ``problem.spec.dps``. This bypasses pmp2sdp's monomial round-trip, which
        loses precision for high-degree, many-pole blocks (the monomial
        coefficients span tens of orders of magnitude). No pmp2sdp Docker call
        is made.
      * ``"pmp2sdp"``: run the SDPB image's ``pmp2sdp`` on the written PMP JSON.

    Both feed the same ``sdpb`` invocation. The ``pmp2sdp_*`` return/log fields
    carry the sdp-writer summary when ``backend="sdp"``.

    Hotstart (SDPB warm-start): pass ``initial_checkpoint_dir`` (typically a
    prior run's :attr:`DockerBootstrapRun.checkpoint_dir`) to seed SDPB's
    initial iterate from that checkpoint via ``--initialCheckpointDir``. The
    checkpoint is copied under the mount as ``initial.ck`` so it is reachable in
    the container. This only helps -- and is only valid -- when the SDP block
    structure matches the checkpoint's (e.g. a gap binary search varying only
    delta_gap); SDPB errors on a dimension mismatch. To produce a checkpoint to
    chain from, omit ``--noFinalCheckpoint`` from ``sdpb_options``; the written
    ``<sdp_dir_name>.ck`` is then returned as ``checkpoint_dir``.

    Every solver knob is passed through the single ``sdpb_options`` mapping (or
    a flat arg sequence), keyed by its SDPB command-line flag, e.g.::

        sdpb_options={
            "--precision": 768,            # bit precision; also drives pmp2sdp
            "--findPrimalFeasible": True,
            "--findDualFeasible": True,
            "--writeSolution": "x,y",
            "--noFinalCheckpoint": True,
            "--mpiProcs": 4,               # run sdpb under `mpirun -n 4`
        }

    A ``True`` value emits a bare flag; ``False``/``None`` are omitted; any
    other value emits ``flag value``. ``--precision`` defaults to a bit budget
    derived from the spec's decimal ``dps``. It and ``--mpiProcs`` are
    interpreted specially; everything else is forwarded to sdpb verbatim.
    Before replacing saved inputs, previous solution and objective files in the output directory are removed,
    including when ``run_sdpb=False`` prepares inputs without solving.
    If ``--writeSolution`` omits ``y``, no dual solution remains available for certification. Hotstart checkpoints
    are staged separately before this cleanup.
    """
    if runner is None:
        # A non-default `image` is an explicit caller choice; the default lets
        # VB_SDPB_IMAGE (and VB_SDPB_MODE) steer execution.
        runner = resolve_sdpb_runner(
            image=image if image != _DOCKER_SDPB_IMAGE else None
        )

    if backend not in ("pmp2sdp", "sdp"):
        raise ValueError(f"backend must be 'pmp2sdp' or 'sdp', got {backend!r}")
    precision_bits, mpi_procs, sdpb_flag_args = _split_sdpb_options(
        sdpb_options, _dps_to_bits(problem.spec.dps)
    )
    if precision_bits < 1:
        raise ValueError("--precision must be >= 1")

    output_dir = Path(output_dir).resolve()
    sdp_dir = output_dir / sdp_dir_name
    sdpb_out_dir = output_dir / sdpb_out_dir_name if run_sdpb else None
    checkpoint_dir = output_dir / (str(sdp_dir_name) + ".ck")
    pmp2sdp_log_path = output_dir / pmp2sdp_log_name if pmp2sdp_log_name else None
    sdpb_log_path = output_dir / sdpb_log_name if run_sdpb and sdpb_log_name else None

    # The sdp writer consumes the PMP's explicit samplePoints/sampleScalings (it
    # writes the SDP at the sample points directly). Every npoles (including
    # "all") normally carries them; fall back to pmp2sdp (which derives nodes
    # from the poles) only if some block lacks explicit samplePoints, so
    # backend="sdp" stays a safe default.
    resolved_backend = backend
    _pmp_blocks = problem.pmp_build.doc["PositiveMatrixWithPrefactorArray"]
    _has_sample_points = all("samplePoints" in b for b in _pmp_blocks)
    if backend == "sdp" and not _has_sample_points:
        resolved_backend = "pmp2sdp"

    if resolved_backend == "sdp":
        from virasoro_blocks.requests import _integer_field

        # the working precision: the block values and every number derived from them carry spec.dps digits, so
        # writing more digits would only pad them (SDPB reads the data at its own --precision either way)
        resolved_eval_dps = (
            _integer_field("eval_dps", eval_dps)
            if eval_dps is not None
            else int(problem.spec.dps)
        )
        if resolved_eval_dps < 1:
            raise ValueError("eval_dps must be >= 1")

    # Conditioning advisory: when pmp2sdp runs WITHOUT explicit sample data, it
    # derives its nodes, sample scalings, AND bilinear basis from the
    # DampedRational -- from which complex conjugate pole pairs have been dropped
    # (positivity-equivalently, so the result is still correct). That
    # complex-pole-free measure is less well conditioned. (When explicit sample
    # points/scalings ARE supplied -- the default explicit_sample_points=True --
    # pmp2sdp consumes the complex-aware samplePoints/sampleScalings and builds
    # its basis from them, so there is no complex-pole penalty and we stay
    # silent.) backend="sdp" always folds the dropped pairs back into the
    # scalings/basis. Advisory only -- the verdict/objective is unchanged.
    if resolved_backend == "pmp2sdp" and not _has_sample_points:
        _all_channels = (
            problem.spin_channels
            + problem.large_spin_channels
            + problem.point_channels
        )
        if any(
            abs(mp.im(p)) > 0
            for ch in _all_channels
            for p in getattr(ch.prefactored, "pole_list_x", ()) or ()
        ):
            warnings.warn(
                "running pmp2sdp without explicit sample points while the block "
                "has complex poles: the conjugate pairs are dropped from the "
                "DampedRational (positivity-equivalent, so the result is correct) "
                "and pmp2sdp then builds its nodes, scalings, and bilinear basis "
                "from that complex-pole-free measure, which is less well "
                "conditioned. Use backend='sdp' (or explicit_sample_points=True) "
                "to fold the pairs back into the scalings/basis.",
                stacklevel=2,
            )

    # Resolve the executable and stage the seed before replacing saved inputs.
    # A missing tool or checkpoint leaves the previous solve intact.
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_ck_args = stage_initial_checkpoint(output_dir, initial_checkpoint_dir) if run_sdpb else []
    pmp2sdp_command: tuple[str, ...] = ()
    if resolved_backend == "pmp2sdp":
        pmp2sdp_argv, pmp2sdp_cwd = runner.command_argv(
            "pmp2sdp",
            ["-i", "pmp.json", "-o", sdp_dir_name, "-p", str(precision_bits)],
            output_dir,
        )
        pmp2sdp_command = tuple(pmp2sdp_argv)
    if run_sdpb:
        # --mpiProcs adds an MPI wrapper; otherwise SDPB runs on one rank.
        sdpb_command_list, sdpb_cwd = runner.command_argv(
            "sdpb",
            ["-s", sdp_dir_name, "-o", sdpb_out_dir_name,
             "--precision", str(precision_bits), *initial_ck_args, *sdpb_flag_args],
            output_dir,
            mpi_procs=mpi_procs,
        )
    from .hpc.storage import record_sdp_source

    # Invalidate old outputs before the first input write, even for preparation-only calls.
    # An exception during conversion leaves this attempt incomplete.
    record_sdp_source(sdp_dir, output_dir / sdpb_out_dir_name, state="preparing")
    manifest_path, pmp_path = write_identical_scalar_gap_problem(problem, output_dir)
    pmp2sdp_summary = None
    if resolved_backend == "sdp":
        # Write the SDPB `sdp` directory directly from the in-memory PMP doc
        # (bypass pmp2sdp's monomial round-trip): the per-sample-point data come
        # from the numeric block values and the bilinear bases are built at
        # eval_dps, which is cleaner than pmp2sdp evaluating the monomial
        # polynomials for high-degree, many-pole blocks.
        write_sdp_directory(
            problem.pmp_build.doc, sdp_dir, eval_dps=resolved_eval_dps,
            sample_values_by_block=problem.sample_values_by_block,
        )
        n_clean = 0 if problem.sample_values_by_block is None else len(problem.sample_values_by_block)
        pmp2sdp_summary = (
            f"sdp_writer (eval_dps={resolved_eval_dps}, "
            f"output_digits={resolved_eval_dps}, v2_clean_blocks={n_clean})"
        )
        if pmp2sdp_log_path is not None:
            pmp2sdp_log_path.write_text(pmp2sdp_summary + "\n")
    else:
        _run_command(pmp2sdp_command, pmp2sdp_log_path, cwd=pmp2sdp_cwd)

        if pmp2sdp_log_path is not None:
            pmp2sdp_summary = _first_matching_line(
                pmp2sdp_log_path.read_text(),
                ("Processed",),
            )

    sdpb_command = None
    sdpb_status = None
    sdpb_summary = None
    if run_sdpb:
        # Reused output directories can leave behind an incompatible SDPB
        # checkpoint tree from an earlier solve with a different
        # block count. Clear the *write* target before each run so SDPB starts
        # from the fresh PMP we just wrote (the warm-start initial checkpoint,
        # if any, lives in a separate `initial.ck`).
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        if initial_ck_args:
            # SDPB needs an existing write target when loading a separate seed.
            checkpoint_dir.mkdir()

        source_record = record_sdp_source(sdp_dir, sdpb_out_dir, state="running")
        sdpb_rc = _run_command(sdpb_command_list, sdpb_log_path, check=False, cwd=sdpb_cwd)
        # Keep the hashes captured before launch, including after a failed or
        # interrupted attempt. An exception leaves the record incomplete.
        source_record.update(state="completed" if sdpb_rc == 0 else "failed")
        sdpb_command = tuple(sdpb_command_list)
        sdpb_status, sdpb_summary = _sdpb_outcome(sdpb_rc, sdpb_log_path)
    else:
        record_sdp_source(sdp_dir, output_dir / sdpb_out_dir_name, state="prepared")

    return DockerBootstrapRun(
        output_dir=output_dir,
        manifest_path=manifest_path,
        pmp_path=pmp_path,
        sdp_dir=sdp_dir,
        sdpb_out_dir=sdpb_out_dir,
        precision_bits=precision_bits,
        pmp2sdp_command=tuple(pmp2sdp_command),
        sdpb_command=sdpb_command,
        pmp2sdp_log_path=pmp2sdp_log_path,
        sdpb_log_path=sdpb_log_path,
        pmp2sdp_summary=pmp2sdp_summary,
        sdpb_status=sdpb_status,
        sdpb_summary=sdpb_summary,
        # Expose the final checkpoint SDPB wrote (if any) so a caller can chain
        # it into the next solve's initial_checkpoint_dir for hotstarting.
        checkpoint_dir=(
            checkpoint_dir
            if run_sdpb and checkpoint_dir.is_dir() and any(checkpoint_dir.iterdir())
            else None
        ),
    )


# Backward-compatible name from when Docker was the only execution mode; the
# runner abstraction made the "docker" suffix a misnomer (VB_SDPB_MODE may
# select apptainer or native), but existing scripts import this name.
run_identical_scalar_gap_problem_in_docker = run_identical_scalar_gap_problem
