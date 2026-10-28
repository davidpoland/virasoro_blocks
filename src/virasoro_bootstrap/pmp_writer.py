"""Write Polynomial Matrix Programs in pmp2sdp's JSON schema.

pmp2sdp expects the structure documented at
https://github.com/davidsd/sdpb/blob/master/docs/json_schema/pmp_schema.json:

    {
        "objective": ["c0", ...],
        "normalization": ["c0", ...],     # optional
        "PositiveMatrixWithPrefactorArray": [
            {
                "DampedRational": {       # optional; pmp2sdp uses base^x / 1 otherwise
                    "base": "...",
                    "constant": "1",
                    "poles": ["...", ...]   # real numbers only
                },
                "polynomials": [[[[c0, c1, ...], [c0, c1, ...], ...]]]
            },
            ...
        ]
    }

All numbers are strings parsed at SDPB's precision (Boost_Float). The `pole`
list is strictly real -- complex conjugate pairs must be eliminated upstream
(see `drop_complex_pole_pairs`).

`samplePoints` and `sampleScalings` are emitted whenever
`explicit_sample_points` is set (the production default,
`IdenticalScalarGapSpec.explicit_sample_points`); the bilinear bases are left
to pmp2sdp or to `sdp_writer`, which build their own. Without explicit
sampling, pmp2sdp computes the nodes from the prefactor and polynomial degrees
(SDPB's `pmp-no-optional-fields.json` example).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import json
import os
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
from typing import Iterable

import mpmath as mp

from virasoro_blocks.interpolation import DampedRational, get_sample_points
from virasoro_blocks.numeric_input import (
    parse_number, real_input_label, _evaluation_number,
)
from virasoro_blocks.serialization import SerializedDerivTable
from virasoro_blocks.writers import _num_str
from .crossing import (
    PositivePoleLowering,
    _max_numerator_degree_all,
    _remove_selected_root_groups,
    _serialized_prefactor,
    drop_complex_pole_pairs,
    lower_positive_real_poles,
)


# ---------------------------------------------------------------------------
# Explicit sampling fields for pmp2sdp.
#
# When `samplePoints` and `sampleScalings` are provided in the PMP JSON,
# pmp2sdp uses them directly instead of computing its own Bohr-Sommerfeld
# nodes from the DampedRational poles.  This is essential when complex poles
# have been dropped from the DampedRational: pmp2sdp cannot see them and
# computes different (suboptimal) nodes.  By passing our interpolation nodes
# directly, pmp2sdp evaluates the polynomial exactly where it was fitted,
# eliminating the sample-point-mismatch artifact.
#
# The bilinear bases are not emitted (pmp2sdp and sdp_writer build their own).
# ---------------------------------------------------------------------------


def _compute_sample_scalings(
    sample_points_x: tuple,
    base,
    pole_list_x: tuple,
    *,
    min_pole_distance=None,
) -> list:
    """Evaluate the full damped-rational prefactor at each sample point.

    sampleScaling[i] = base^x_i / prod_j (x_i - pole_j)

    where the product runs over ALL poles (including complex ones that were
    dropped from the DampedRational for SDPB's real-pole constraint).
    """
    if min_pole_distance is None:
        min_pole_distance = mp.mpf(10) ** (-(mp.mp.dps - 10))

    prefactor = DampedRational(
        constant=mp.mpf(1),
        poles=tuple(pole_list_x),
        base=base,
    )
    scalings = []
    for x in sample_points_x:
        x_real = mp.mpf(mp.re(mp.mpc(x)))
        val = prefactor.evaluate_regulated(x_real, min_pole_distance)
        # Result must be real and positive for the SDP positivity interval.
        scalings.append(mp.re(mp.mpc(val)))
    return scalings


def _compute_bilinear_bases(
    sample_points_x: tuple,
    sample_scalings: list,
    numerator_degree: int,
    sig_figs: int,
) -> tuple[list, list]:
    """Compute bilinearBasis_0 and bilinearBasis_1 at the ambient mpmath precision.

    Following the pmp2sdp algorithm (see pmp2sdp_samplePoints_prototype_v2_cxx.wl):

        The effective degree for the bilinear decomposition is
        numSamplePoints - 1 (this is the degree pmp2sdp uses internally).

        delta1 = floor(effective_degree / 2)
        delta2 = floor((effective_degree - 1) / 2)

        Q1[n, i] = sqrt(scaling[i]) * x_i^n       for n=0..delta1
        Q2[n, i] = sqrt(scaling[i]) * x_i^(n+1/2) for n=0..delta2

        bilinearBasis_0 = an orthonormal basis of row-space(Q1)
        bilinearBasis_1 = an orthonormal basis of row-space(Q2)

    SDPB's bilinear-basis constraint is invariant under an orthogonal rotation
    of the basis, so any orthonormal basis of the row space is valid. The one
    returned is the degree-ordered one: row ``n`` is ``sqrt(w_i) p_n(x_i)``
    with ``p_n`` the degree-``n`` orthonormal polynomial (positive leading
    coefficient) of the discrete measure ``sum_i w_i delta(x - x_i)``, with
    ``w_i = |scaling_i|`` (basis 0) or ``|scaling_i| x_i`` (basis 1) -- exactly
    what Gram-Schmidt of the weighted-monomial rows of Q1/Q2 gives, but computed
    by the Stieltjes/Lanczos recurrence (:func:`_orthonormal_polynomial_rows`),
    which never forms the monomials whose rows span ~120 orders of magnitude.

    pmp2sdp requires bilinearBasis_0 to have exactly delta1+1 rows and
    bilinearBasis_1 to have exactly delta2+1 rows. Only a measure with fewer
    distinct, nonzero-weight support points than rows is zero-padded.
    ``numerator_degree`` and ``sig_figs`` are unused (kept for the call sites).
    These are weighted basis values for the direct SDP writer, not polynomial
    coefficient arrays for the PMP JSON ``bilinearBasis_0/1`` fields.
    """
    n_samples = len(sample_points_x)
    # pmp2sdp uses effective_degree = numSamplePoints - 1
    effective_degree = n_samples - 1
    delta1 = effective_degree // 2
    delta2 = (effective_degree - 1) // 2

    xs = [mp.mpf(mp.re(x)) for x in sample_points_x]
    weights = [abs(mp.mpf(mp.re(s))) for s in sample_scalings]
    if len(xs) != len(weights) or not all(mp.isfinite(v) for v in xs + weights):
        raise ValueError("bilinear basis: points and scalings must be finite arrays of equal length")
    # Degree-0 blocks (a single sample point, e.g. a point constraint
    # y . F(Delta_point) >= 0) have effective_degree = 0, so delta2 = -1: the
    # x*measure SOS term does not exist and bilinearBasis_1 is empty (a degree-0
    # SOS is just the constant square).
    if delta2 >= 0 and any(x < 0 for x in xs):
        raise ValueError(
            "bilinear basis: sample points must be >= 0 (the x*measure basis "
            "uses sqrt(x_i))"
        )
    from flint import arb, arb_mat, ctx as flint_ctx
    from virasoro_blocks.polynomial import _arb_to_mpf, _mpf_to_arb

    support_0 = len({x for x, w in zip(xs, weights) if w != 0})
    support_1 = len({x for x, w in zip(xs, weights) if w != 0 and x != 0})
    saved_prec = flint_ctx.prec
    flint_ctx.prec = mp.mp.prec + _BILINEAR_GUARD_BITS
    try:
        xs_f = [_mpf_to_arb(x) for x in xs]
        w0 = [_mpf_to_arb(w) for w in weights]
        rows_0 = _orthonormal_polynomial_rows(
            xs_f, [w.sqrt() for w in w0], delta1 + 1, support_0, arb, arb_mat,
        )
        if delta2 < 0:
            rows_1 = None
        else:
            rows_1 = _orthonormal_polynomial_rows(
                xs_f, [(w * x).sqrt() for w, x in zip(w0, xs_f)],
                delta2 + 1, support_1, arb, arb_mat,
            )
    finally:
        flint_ctx.prec = saved_prec

    def to_rows(Q, m):
        # round the guard-digit midpoints to the ambient precision
        return [[+_arb_to_mpf(Q[a, i]) for i in range(n_samples)] for a in range(m)]

    basis_0 = to_rows(rows_0, delta1 + 1)
    basis_1 = [] if rows_1 is None else to_rows(rows_1, delta2 + 1)
    return basis_0, basis_1


# Guard bits carried by the flint bilinear-basis recurrence above the ambient
# mpmath precision (the output is rounded back to it).
_BILINEAR_GUARD_BITS = 64


def _orthonormal_polynomial_rows(xs, sqrt_w, m, support_size, arb, arb_mat):
    """Rows ``Q[n, i] = sqrt(w_i) p_n(x_i)``, n < m, as an (m x N) arb_mat.

    ``p_n`` are the orthonormal polynomials of the discrete measure
    ``sum_i w_i delta(x - x_i)``, built by the Stieltjes/Lanczos recurrence on
    ``diag(x)`` started from ``sqrt(w)/|sqrt(w)|``: ``q_n`` is the normalized
    part of ``x * q_{n-1}`` orthogonal to ``q_0..q_{n-1}``, with full
    re-orthogonalisation (classical Gram-Schmidt against all previous rows,
    twice -- CGS2) so the rows stay orthonormal to working precision instead of
    relying on the three-term recurrence alone. ``span(q_0..q_n) =
    span(sqrt(w) x^k, k <= n)`` and each ``q_n`` keeps a positive leading
    coefficient, so the rows equal Gram-Schmidt of the weighted monomials.

    Rank equals the number of distinct support points with nonzero weight
    (``support_size``); the remaining rows are left zero (pmp2sdp's padding),
    as they are after a zero residual. One pass at the flint precision. Midpoints
    are kept (radii dropped) at each step: this is floating-point Gram-Schmidt
    at the working precision, not an enclosure.
    """
    n = len(xs)
    Q = arb_mat(m, n)
    if m == 0 or support_size == 0:
        return Q

    # A positive affine change of variable preserves every degree-ordered
    # polynomial row space and its leading-coefficient sign. Center and scale
    # the recurrence so a large common offset cannot make its residual look
    # rank deficient. Keep sqrt_w unchanged: the odd basis uses the original x.
    if xs:
        left, right = min(xs), max(xs)
        width = (right - left).mid()
        if not width.is_zero():
            xs = [((x - left) / width).mid() for x in xs]

    def norm(v):  # v: 1 x n
        return (v * v.transpose())[0, 0].sqrt()

    v = arb_mat(1, n, sqrt_w)
    nrm = norm(v).mid()
    if not nrm.is_finite() or nrm.is_zero():
        return Q  # zero measure: every row is padding
    q = [(v[0, i] / nrm).mid() for i in range(n)]
    for i in range(n):
        Q[0, i] = q[i]
    for a in range(1, min(m, support_size)):
        v = arb_mat(1, n, [xs[i] * q[i] for i in range(n)])
        QT = Q.transpose()
        for _ in range(2):  # CGS2: project out span(q_0..q_{a-1}) twice (rows >= a are zero)
            v = (v - (v * QT) * Q).mid()
        nrm = norm(v).mid()
        if not nrm.is_finite() or nrm.is_zero():
            break
        q = [(v[0, i] / nrm).mid() for i in range(n)]
        for i in range(n):
            Q[a, i] = q[i]
    return Q


@dataclass(frozen=True)
class PrefactoredPmpBuild:
    """Result of lowering prefactor-applied serialized blocks into a PMP doc."""

    doc: dict
    positivity_blocks: tuple[SerializedDerivTable, ...]
    positive_pole_lowerings: tuple[PositivePoleLowering | None, ...]


def _default_sig_figs() -> int:
    """Match the JSON writer's default: dps + 5 guard digits."""
    return mp.mp.dps + 5


def _resolve_sig_figs(value) -> int:
    from virasoro_blocks.requests import _integer_field

    if value is None:
        return _default_sig_figs()
    value = _integer_field("sig_figs", value)
    if value < 1:
        raise ValueError("sig_figs must be >= 1")
    return value


def _pmp_real_tol(x) -> mp.mpf:
    """Tolerance for quantities that are mathematically real in PMP output.

    Use the shared output-validation policy, including its low-precision cap.
    Formatting noise allowances must not turn materially complex input into
    real SDP data when the caller lowers precision.
    """
    from virasoro_blocks.clean_samples import _real_tol

    return _real_tol(x)


def _pmp_real_value(x, *, where: str):
    """Validate a real PMP datum before any projection or numerical formatting."""
    x = parse_number(x)
    if not mp.isfinite(x):
        raise ValueError(f"{where}: PMP entries must be finite; got {x}")
    if isinstance(x, (mp.mpf, Fraction)):
        return x
    if isinstance(x, complex):
        from virasoro_blocks.polynomial import _polynomial_input

        z = _polynomial_input(x)  # retain both supplied binary components
    else:
        z = x if isinstance(x, mp.mpc) else mp.mpc(x)
    if z.imag == 0:
        return z.real
    if abs(mp.im(z)) > _pmp_real_tol(z):
        raise ValueError(
            f"{where}: PMP entries must be real after bootstrap-side folding; "
            f"got {x}"
        )
    return z.real


def _pmp_real_str(x, sig_figs: int, *, where: str) -> str:
    return _num_str(_pmp_real_value(x, where=where), sig_figs, preserve_data=True)


def _sampling_real_label(value, *, where: str):
    """Retain a coordinate's exact identity, including native binary inputs."""
    value = parse_number(value)
    if isinstance(value, (mp.mpf, mp.mpc)):
        if not mp.isfinite(value) or mp.im(value) != 0:
            raise ValueError(f"{where}: sample inputs must be finite and real")
        return value.real if isinstance(value, mp.mpc) else value
    value = real_input_label(value, name=f"{where}: sample input")
    if isinstance(value, Fraction) and value.denominator & (value.denominator - 1) == 0:
        # Dyadic rationals and their stored mpf representations are the same
        # label. Pack them exactly, without rounding at the ambient precision
        # or inflating the arithmetic guard for already-binary coordinates.
        from mpmath.libmp import from_man_exp

        return mp.make_mpf(from_man_exp(value.numerator, 1 - value.denominator.bit_length()))
    return value


def _sampling_label_key(value):
    # Mixed mpf/Fraction equality can round the Fraction before comparing it.
    return ("binary", value._mpf_) if isinstance(value, mp.mpf) else ("rational", value)


def _sampling_point_labels(values, *, where: str):
    labels = tuple(_sampling_real_label(value, where=where) for value in values)
    if (not labels or any(value < 0 for value in labels)
            or len({_sampling_label_key(value) for value in labels}) != len(labels)):
        raise ValueError(f"{where}: sample points must be nonempty, distinct and nonnegative")
    return labels


def _saved_sample_tables_at_nodes(block, coordinate: str, xs) -> list | None:
    """Reuse a saved grid only at identical source coordinates."""
    sbv = getattr(block, "sample_block_values", None)
    nodes = getattr(block, "sample_points_x", None)
    if sbv is None or coordinate not in sbv or nodes is None:
        return None
    nodes = _sampling_point_labels(nodes, where="saved block sampling")
    if len(sbv[coordinate]) != len(nodes):
        raise ValueError("saved block sample values must match the sample point count")
    index = {_sampling_label_key(value): i for i, value in enumerate(nodes)}
    xs = _sampling_point_labels(xs, where="block sampling")
    keys = tuple(_sampling_label_key(value) for value in xs)
    if any(key not in index for key in keys):
        return None
    return [sbv[coordinate][index[key]] for key in keys]


def _validate_real_poles(pole_list_x, *, where: str) -> None:
    """SDPB stores poles as std::vector<Boost_Float>; they must also stay
    out of the open positivity interval x > 0.

    A real pole at x > 0 makes the damped-rational prefactor singular inside
    the SDP positivity domain [0, infinity), so the resulting PMP block is not
    positivity-equivalent to the original conformal block.
    """
    for p in pole_list_x:
        z = mp.mpc(parse_number(p))
        tol = _pmp_real_tol(z)
        if abs(mp.im(z)) > tol:
            raise ValueError(
                f"{where}: pole_list_x contains complex value {p}; call "
                "virasoro_bootstrap.drop_complex_pole_pairs first to fold "
                "conjugate pairs out of the denominator"
            )
        if mp.re(z) > 0:
            raise ValueError(
                f"{where}: pole_list_x contains real pole {p} with x > 0; "
                "the PMP damped-rational must stay nonsingular on the "
                "positivity interval [0, infinity)"
            )


def _normalize_F_pairs(pairs):
    """Materialize nonnegative integer derivative pairs without truncation."""
    from virasoro_blocks.requests import _integer_field

    result = []
    for pair in pairs:
        try:
            m, n = pair
        except (TypeError, ValueError):
            raise ValueError("each F-pair must contain two derivative indices") from None
        m = _integer_field("F-pair m", m)
        n = _integer_field("F-pair n", n)
        if m < 0 or n < 0:
            raise ValueError("F-pair indices must be nonnegative")
        result.append((m, n))
    return result


def _validate_F_pair(rows, m: int, n: int, *, where: str) -> None:
    if m < 0 or n < 0:
        raise ValueError(f"{where}: F-pair ({m}, {n}) has negative index")
    if m >= len(rows):
        raise ValueError(
            f"{where}: F-pair ({m}, {n}) exceeds numerator row count {len(rows)}"
        )
    if n >= len(rows[m]):
        raise ValueError(
            f"{where}: F-pair ({m}, {n}) exceeds numerator column count "
            f"{len(rows[m])} at m={m}"
        )


def _polynomial_to_str_list(poly, sig_figs: int) -> list:
    return [
        _pmp_real_str(c, sig_figs, where="polynomial coefficient")
        for c in poly.coeffs
    ]


def _block_to_pmp_entry(
    block: SerializedDerivTable,
    F_pairs: list,
    coordinate: str,
    sig_figs: int,
    *,
    where: str,
    sampling_data: dict | None = None,
    extra_polynomials: Iterable | None = None,
) -> dict:
    if coordinate not in block.numerators:
        raise KeyError(
            f"{where}: coordinate {coordinate!r} not in block.numerators "
            f"(available: {sorted(block.numerators)})"
        )
    _validate_real_poles(block.pole_list_x, where=where)

    rows = block.numerators[coordinate]

    # When explicit sample points are provided, pmp2sdp requires
    # len(samplePoints) == max_poly_degree + 1. Our interpolation may use
    # more sample points than the max degree in the F-vector polynomials
    # (because the prefactor convolution selects lower-degree entries). Pad
    # polynomials with zero high-order coefficients to match.
    target_degree = None
    if sampling_data is not None:
        try:
            xs = _sampling_point_labels(sampling_data["samplePoints"], where=where)
            scalings = [_sampling_real_label(s, where=where) for s in sampling_data["sampleScalings"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{where}: sampling requires valid real samplePoints and sampleScalings: {exc}") from exc
        if not xs or len(xs) != len(scalings):
            raise ValueError(f"{where}: sample points and scalings must have the same nonzero length")
        if any(s <= 0 for s in scalings):
            raise ValueError(f"{where}: sample points must be finite, distinct and nonnegative; "
                             "sample scalings must be positive and finite")
        n_sp = len(xs)
        target_degree = n_sp - 1  # degree = numSamplePoints - 1
        for key, n_rows in (("bilinearBasis_0", (n_sp + 1) // 2),
                            ("bilinearBasis_1", n_sp // 2)):
            if key not in sampling_data:
                continue
            try:
                matrix = sampling_data[key]
                # PMP fields contain polynomial coefficients, not the sampled
                # basis rows returned by _compute_bilinear_bases for direct SDP.
                valid = (len(matrix) == n_rows and all(len(row) > 0 for row in matrix)
                         and all(mp.isfinite(mp.mpf(x)) for row in matrix for x in row)
                         and all(mp.mpf(x) == 0 for row in matrix for x in row[n_rows:]))
            except (TypeError, ValueError):
                valid = False
            if not valid:
                raise ValueError(f"{where}: {key} must contain {n_rows} finite real bilinear "
                                 f"basis polynomials of degree at most {n_rows - 1}")

    poly_vector = []
    for (m, n) in F_pairs:
        _validate_F_pair(rows, m, n, where=where)
        poly_str = _polynomial_to_str_list(rows[m][n], sig_figs)
        if target_degree is not None and len(poly_str) - 1 < target_degree:
            # Pad with zeros to reach target_degree
            poly_str.extend(["0"] * (target_degree + 1 - len(poly_str)))
        poly_vector.append(poly_str)
    if extra_polynomials is not None:
        for poly in extra_polynomials:
            poly_str = _polynomial_to_str_list(poly, sig_figs)
            if target_degree is not None and len(poly_str) - 1 < target_degree:
                poly_str.extend(["0"] * (target_degree + 1 - len(poly_str)))
            poly_vector.append(poly_str)

    if target_degree is not None and any(len(poly) > target_degree + 1 for poly in poly_vector):
        raise ValueError(f"{where}: too few sample points for the polynomial degree")

    # 1x1 matrix wrapping the polynomial vector: [[poly_vector]] inside the
    # outer 4-deep array required by the schema.
    polynomials = [[poly_vector]]

    entry = {
        "DampedRational": {
            "base": _pmp_real_str(block.base, sig_figs, where=f"{where}.base"),
            "constant": "1",
            "poles": [
                _pmp_real_str(p, sig_figs, where=f"{where}.pole")
                for p in block.pole_list_x
            ],
        },
        "polynomials": polynomials,
    }

    # Include explicit sampling fields when provided.
    if sampling_data is not None:
        entry["samplePoints"] = sampling_data["samplePoints"]
        entry["sampleScalings"] = sampling_data["sampleScalings"]
        if "bilinearBasis_0" in sampling_data:
            entry["bilinearBasis_0"] = sampling_data["bilinearBasis_0"]
        if "bilinearBasis_1" in sampling_data:
            entry["bilinearBasis_1"] = sampling_data["bilinearBasis_1"]

    return entry


def build_pmp_doc(
    positivity_blocks: Iterable[SerializedDerivTable],
    F_pairs: list,
    *,
    coordinate: str = "zzb",
    objective: Iterable,
    normalization: Iterable | None = None,
    sig_figs: int | None = None,
    sampling_data_per_block: list[dict] | None = None,
    extra_polynomials_per_block: list[Iterable] | None = None,
) -> dict:
    """Build a pmp2sdp-compatible PMP document as a Python dict.

    Parameters
    ----------
    positivity_blocks:
        One `SerializedDerivTable` per positivity block (typically one per spin
        in the single-correlator case). Each must (a) have only real entries in
        `pole_list_x` -- complex conjugate pairs must already have been folded
        out via `drop_complex_pole_pairs`, and (b) carry the bootstrap-target
        cross-ratio prefactor already (the library's ``z_shift = -2 h_ext`` for
        identical scalars) so the numerators are the F-vector polynomials.
    F_pairs:
        List of `(m, n)` tuples selecting the polynomial vector entries from
        each block. Typically `F_minus_pairs(lambda)` for an identical-scalar
        bootstrap on `zzb`. The length determines the F-vector dimension and
        must match `objective` (and `normalization` if given).
    coordinate:
        Which coordinate in `block.numerators` to consume; default `"zzb"`.
    objective:
        SDP objective vector. Length must equal `len(F_pairs)`.
    normalization:
        SDP normalization vector. Length must equal `len(F_pairs)`. Optional;
        omit only for problems where pmp2sdp/sdpb can run without one
        (typically you do want to supply one to pin the functional scale).
    sig_figs:
        Number of significant figures for the string-serialized numbers.
        Default `mp.mp.dps + 5`.
    sampling_data_per_block:
        Optional list of dicts (one per block) with pre-computed sampling
        fields to include in the PMP output. Each dict requires
        ``samplePoints`` and ``sampleScalings`` and may also contain
        ``bilinearBasis_0`` and ``bilinearBasis_1`` as string-serialized
        polynomial coefficient arrays (not values at the sample nodes).
        Nodes must be distinct, finite and nonnegative, with positive finite
        scalings, and cover the maximum polynomial degree plus one.

    Returns
    -------
    dict
        Python dict ready for `json.dump`; all numbers are strings.
    """
    positivity_blocks = list(positivity_blocks)
    if not positivity_blocks:
        raise ValueError("positivity_blocks must be non-empty")

    F_pairs = _normalize_F_pairs(F_pairs)
    if not F_pairs:
        raise ValueError("F_pairs must be non-empty")

    objective = list(objective)
    vector_length = len(F_pairs)
    if extra_polynomials_per_block is not None:
        extra_polynomials_per_block = [
            tuple(extra) for extra in extra_polynomials_per_block
        ]
        if len(extra_polynomials_per_block) != len(positivity_blocks):
            raise ValueError(
                "extra_polynomials_per_block length "
                f"{len(extra_polynomials_per_block)} must equal number of "
                f"blocks {len(positivity_blocks)}"
            )
        extra_lengths = [len(tuple(extra)) for extra in extra_polynomials_per_block]
        if len(set(extra_lengths)) > 1:
            raise ValueError(
                "all blocks must carry the same number of extra polynomial "
                f"components, got {extra_lengths}"
            )
        vector_length += extra_lengths[0] if extra_lengths else 0
    if len(objective) != vector_length:
        raise ValueError(
            f"objective length {len(objective)} must equal polynomial vector "
            f"length {vector_length}"
        )
    if normalization is not None:
        normalization = list(normalization)
        if len(normalization) != vector_length:
            raise ValueError(
                f"normalization length {len(normalization)} must equal "
                f"polynomial vector length {vector_length}"
            )

    sig_figs = _resolve_sig_figs(sig_figs)

    if sampling_data_per_block is not None:
        if len(sampling_data_per_block) != len(positivity_blocks):
            raise ValueError(
                f"sampling_data_per_block length {len(sampling_data_per_block)} "
                f"must equal number of blocks {len(positivity_blocks)}"
            )

    entries = []
    for idx, block in enumerate(positivity_blocks):
        sd = sampling_data_per_block[idx] if sampling_data_per_block else None
        extra = (
            extra_polynomials_per_block[idx]
            if extra_polynomials_per_block is not None
            else None
        )
        entries.append(
            _block_to_pmp_entry(
                block, F_pairs, coordinate, sig_figs,
                where=f"positivity_blocks[{idx}]",
                sampling_data=sd,
                extra_polynomials=extra,
            )
        )

    doc: dict = {
        "objective": [
            _pmp_real_str(c, sig_figs, where="objective")
            for c in objective
        ],
        "PositiveMatrixWithPrefactorArray": entries,
    }
    if normalization is not None:
        doc["normalization"] = [
            _pmp_real_str(c, sig_figs, where="normalization")
            for c in normalization
        ]

    return doc


def _build_sampling_data(
    block: SerializedDerivTable,
    full_pole_list_x: tuple,
    *,
    coordinate: str,
    F_pairs: list,
    sig_figs: int,
) -> dict | None:
    """Compute sampling fields for a single positivity block.

    Returns a dict with ``samplePoints`` and ``sampleScalings``, or None if
    the block has no stored sample points. The solver builds bilinear bases.
    """
    if block.sample_points_x is None:
        return None

    sample_labels = _sampling_point_labels(block.sample_points_x, where="PMP sampling")
    sample_points = [_evaluation_number(parse_number(x)) for x in sample_labels]
    # Compute sampleScalings using the FULL prefactor (including complex poles
    # that were dropped from the DampedRational in the PMP output).
    scalings = _compute_sample_scalings(
        sample_points, block.base, full_pole_list_x,
    )
    if any(not mp.isfinite(s) or s <= 0 for s in scalings):
        raise ValueError("sample scalings must be positive and finite")

    # Emit only samplePoints + sampleScalings; the bilinear bases are NOT
    # precomputed here. Both consumers recompute them anyway -- pmp2sdp builds
    # its own (compiled, at solve precision) and the sdp writer builds its own
    # (`sdp_writer._block_data`, at eval_dps) -- and neither reads a basis from
    # the PMP doc. ``bilinearBasis_0``/``bilinearBasis_1`` are still
    # threaded through `_block_to_pmp_entry` if a caller injects them into the
    # sampling-data dict, but the default path does not.
    return {
        "samplePoints": [
            _pmp_real_str(x, sig_figs, where="samplePoint")
            for x in sample_labels
        ],
        "sampleScalings": [
            _pmp_real_str(s, sig_figs, where="sampleScaling")
            for s in scalings
        ],
    }


def build_prefactored_pmp_doc(
    prefactored_blocks: Iterable[SerializedDerivTable],
    F_pairs: list,
    *,
    coordinate: str = "zzb",
    objective: Iterable,
    normalization: Iterable | None = None,
    sig_figs: int | None = None,
    lower_positive_real: bool = True,
    positive_pole_npoles: int | None = None,
    positive_pole_selection: str = "rightmost",
    explicit_sample_points: bool = False,
    extra_polynomials_per_block: list[Iterable] | None = None,
) -> PrefactoredPmpBuild:
    """Build a PMP doc from prefactor-applied serialized blocks.

    This is the production bootstrap-side pipeline on top of the strict
    low-level writer:

    1. fold out any complex-conjugate pole pairs from each block's denominator,
    2. optionally lower genuine positive real poles into a positivity-equivalent
       cleared block, with optional remainder interpolation,
    3. pass the resulting PMP-admissible blocks to `build_pmp_doc`.

    Use `build_pmp_doc` directly if you want strict validation with no
    preprocessing. This wrapper is intended for prefactor-applied blocks that
    are meant to be consumed by SDPB.

    Parameters
    ----------
    explicit_sample_points : bool
        When True, include ``samplePoints`` and ``sampleScalings`` in the PMP
        output for each block (degree-matched nodes are attached to a block
        that lacks enough).  This forces pmp2sdp to use our
        interpolation nodes rather than computing its own from the
        (reduced) DampedRational poles, eliminating the sample-point-mismatch
        artifact when complex poles have been dropped.
    """
    requested_sig_figs = sig_figs
    sig_figs = _resolve_sig_figs(sig_figs)
    if positive_pole_npoles is not None and not lower_positive_real:
        raise ValueError(
            "positive_pole_npoles requires lower_positive_real=True"
        )

    F_pairs = _normalize_F_pairs(F_pairs)
    prefactored_blocks = list(prefactored_blocks)
    prepared_blocks = []
    lowerings = []
    # The pole list each block's sampleScalings multiply through (base^x divided
    # by these). Complex pairs dropped from the DampedRational are re-injected
    # here, but real positive poles that were *lowered* (cleared into the
    # numerator by an even-power multiply-through) must NOT appear -- they are no
    # longer in the prefactor, and a positive pole inside the range would flip
    # the scaling sign.
    full_pole_lists = []
    raw_extra_polynomials = (
        [tuple(extra) for extra in extra_polynomials_per_block]
        if extra_polynomials_per_block is not None
        else None
    )
    if raw_extra_polynomials is not None and len(raw_extra_polynomials) != len(prefactored_blocks):
        raise ValueError(
            "extra_polynomials_per_block length must match prefactored_blocks"
        )
    prepared_extra_polynomials = [] if raw_extra_polynomials is not None else None
    for idx, block in enumerate(prefactored_blocks):
        scaling_poles = block.pole_list_x  # before dropping complex pairs
        prepared_block = drop_complex_pole_pairs(block)
        prepared_extra = (
            tuple(raw_extra_polynomials[idx])
            if raw_extra_polynomials is not None
            else None
        )
        lowering = None
        if lower_positive_real:
            lowering = lower_positive_real_poles(
                prepared_block,
                npoles=positive_pole_npoles,
                pole_selection=positive_pole_selection,
            )
            if lowering.selected_groups and prepared_extra:
                raise NotImplementedError(
                    "moment extra polynomial components are not yet supported "
                    "together with positive-real-pole lowering"
                )
            prepared_block = (
                lowering.cleared_interpolated_block
                if lowering.cleared_interpolated_block is not None
                else lowering.cleared_block
            )
            if lowering.selected_groups:
                # Drop the lowered positive real poles from the scaling product.
                scaling_poles = tuple(
                    _remove_selected_root_groups(
                        block.pole_list_x, lowering.selected_groups
                    )
                )
        # Attach degree-matched Bohr-Sommerfeld nodes whenever explicit sampling
        # is requested and the block lacks enough nodes for the final polynomial
        # vector. Moment components can raise the degree above the original
        # interpolation degree, and pmp2sdp requires num_points = degree + 1.
        required_degree = _max_numerator_degree_all(prepared_block)
        if prepared_extra:
            required_degree = max(
                required_degree,
                max(poly.degree for poly in prepared_extra),
            )

        # Attach Bohr-Sommerfeld nodes to any block that still lacks them so the
        # explicit-sample-scaling path fires and the block can ride the direct
        # sdp writer (conditioning win: high-precision evaluation + optimal node
        # placement, vs pmp2sdp's monomial round-trip at solve precision and the
        # suboptimal nodes it derives from a DampedRational whose complex poles
        # were dropped). NB this is a conditioning/precision improvement, not a
        # correctness fix: dropping complex conjugate pole pairs is positivity-
        # equivalent (each (x-p)(x-p*) > 0 on the real axis), so pmp2sdp gets the
        # right SDP without them too. This covers a block without stored nodes
        # and the positive-pole lowering whose
        # clearing multiply-through raises the cleared block's degree above the
        # remainder interpolation's node count. Nodes follow the block's *own*
        # (reduced) prefactor at its *own* max degree (num_points = degree + 1,
        # as pmp2sdp requires) -- per-block degree-matched, not one global
        # max-degree set. ``get_sample_points`` follows the same optimal
        # interpolation density pmp2sdp uses.
        sample_points = prepared_block.sample_points_x
        if (
            explicit_sample_points
            and (
                sample_points is None
                or len(sample_points) < required_degree + 1
            )
        ):
            blk_pref = _serialized_prefactor(prepared_block)
            nodes = tuple(get_sample_points(blk_pref, required_degree + 1))
            prepared_block = replace(prepared_block, sample_points_x=nodes)
        prepared_blocks.append(prepared_block)
        lowerings.append(lowering)
        full_pole_lists.append(scaling_poles)
        if prepared_extra_polynomials is not None:
            prepared_extra_polynomials.append(prepared_extra)

    # Compute explicit sampling data when requested.
    sampling_data_list = None
    if explicit_sample_points:
        sampling_data_list = []
        for idx, (block, full_poles) in enumerate(zip(prepared_blocks, full_pole_lists)):
            sd = _build_sampling_data(
                block,
                full_poles,
                coordinate=coordinate,
                F_pairs=F_pairs,
                sig_figs=sig_figs,
            )
            sampling_data_list.append(sd)

    doc = build_pmp_doc(
        prepared_blocks,
        F_pairs,
        coordinate=coordinate,
        objective=objective,
        normalization=normalization,
        sig_figs=requested_sig_figs,
        sampling_data_per_block=sampling_data_list,
        extra_polynomials_per_block=prepared_extra_polynomials,
    )
    return PrefactoredPmpBuild(
        doc=doc,
        positivity_blocks=tuple(prepared_blocks),
        positive_pole_lowerings=tuple(lowerings),
    )


def write_pmp_json(doc: dict, output_path) -> None:
    """Dump a PMP document, replacing a file only after the write succeeds.

    All numeric values are already strings. File symlinks keep their targets,
    and existing file mode bits are retained. File descriptors, devices,
    and pipes remain streaming destinations.
    """
    if not isinstance(output_path, (str, bytes, os.PathLike)):
        with open(output_path, "w") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")
        return
    destination = os.fsdecode(output_path)
    previous_mode = None
    try:
        descriptor = os.open(destination, os.O_WRONLY)
    except FileNotFoundError:
        if not destination:
            raise
    else:
        # Check write permission without truncating the previous document.
        with os.fdopen(descriptor, "w") as existing:
            mode = os.fstat(existing.fileno()).st_mode
            if not stat.S_ISREG(mode):
                json.dump(doc, existing, indent=2)
                existing.write("\n")
                return
            previous_mode = stat.S_IMODE(mode)
    # Follow file aliases without normalizing away missing path components,
    # trailing separators, or '..'. The staging parent must exist as written.
    while os.path.islink(destination):
        destination = os.path.join(os.path.dirname(destination), os.readlink(destination))
    parent = os.path.dirname(destination) or os.curdir
    with TemporaryDirectory(prefix=".vb-pmp-", dir=parent) as directory:
        staged = Path(directory) / "pmp.json"
        with staged.open("w") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")
        if previous_mode is not None:
            staged.chmod(previous_mode)
        staged.replace(destination)
