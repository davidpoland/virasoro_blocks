"""Bootstrap-side transforms on the canonical SerializedDerivTable.

The block library expresses each derivative as

    block_deriv(Delta) = base^x * numerator(x) / prod_j (x - pole_j),
    x = Delta - shift.

This module owns the bootstrap-side transforms that preserve positivity on the
real Delta axis while adapting the canonical serialized block to SDPB's PMP
surface. Two important cases are:

- For c in (1, 25), poles come as complex-conjugate pairs in `pole_list_x`,
  which `pmp2sdp` cannot consume because it stores poles as
  `std::vector<Boost_Float>`. `drop_complex_pole_pairs` folds those conjugate
  pairs out of the denominator.
- For singular/minimal-model loci, a prefactor-applied serialized block can
  carry genuine positive real poles in the bootstrap-natural variable
  x = Delta - Delta_unitarity. `lower_positive_real_poles` splits those poles
  into an explicit tower plus an admissible remainder, and can optionally
  interpolate the remainder down to a smaller shared damped-rational prefactor.

In both cases the transformed blocks are positivity-equivalent on the real
axis, which is the only property the PMP layer needs.
"""

from __future__ import annotations

from virasoro_blocks.requests import _integer_field

from dataclasses import dataclass, replace
from functools import wraps

import mpmath as mp

from virasoro_blocks.interpolation import (
    DampedRational,
    get_sample_points,
    lagrange_interpolation_polynomial,
    _canonicalize_prefactor_poles,
    _conjugate_pole,
    _validation_tolerance,
)
from virasoro_blocks.polynomial import (
    Polynomial,
    real_factor_coeffs_from_conjugate_roots,
)
from virasoro_blocks.serialization import SerializedDerivTable
from virasoro_blocks.numeric_input import (
    parse_number, _integer_control, real_input_label,
    _evaluation_number,
)


@dataclass(frozen=True)
class PositivePoleLowering:
    """Bootstrap-side split of a prefactor-applied SerializedDerivTable.

    `selected_block + remainder_block` reconstructs the input block. The
    `cleared_block` stores a positivity-equivalent version of the original with
    the stripped positive poles cleared by an everywhere-nonnegative factor.
    If `npoles` was requested, the result also carries an interpolated version
    of the remainder and its corresponding cleared block.
    """

    selected_groups: tuple[tuple[mp.mpf, int], ...]
    selected_block: SerializedDerivTable | None
    remainder_block: SerializedDerivTable
    cleared_block: SerializedDerivTable
    interpolated_remainder_block: SerializedDerivTable | None = None
    cleared_interpolated_block: SerializedDerivTable | None = None
    sample_points_x: tuple = ()
    requested_npoles: int | None = None
    kept_npoles: int | None = None
    pole_selection: str | None = None


def _at_block_precision(function):
    """Transform stored tables at their working precision (or the caller's, if higher)."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        inputs = [value for value in (*args, *kwargs.values()) if isinstance(value, SerializedDerivTable)]
        work = max([mp.mp.dps] + [int(value.working_dps or 0) for value in inputs])

        def retain_precision(value):
            if isinstance(value, SerializedDerivTable):
                if all(value is not source for source in inputs):
                    value.working_dps = max(work, int(value.working_dps or 0))
            elif isinstance(value, PositivePoleLowering):
                for item in vars(value).values():
                    retain_precision(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    retain_precision(item)

        with mp.workdps(work):
            result = function(*args, **kwargs)
            retain_precision(result)
        return result
    return wrapped


def _imag_tol(x) -> mp.mpf:
    """Below this threshold an imaginary part is treated as numerical noise."""
    magnitude = abs(mp.mpc(x))
    return _validation_tolerance(guard=5) * max(magnitude, mp.mpf(1))


def _pair_tol(p) -> mp.mpf:
    """Roundoff allowance relative to a complex pole's peak width."""
    return _validation_tolerance(guard=5) * abs(mp.im(p))


def _crossing_poles(poles):
    """Apply the shared near-real policy without collapsing explicit pairs."""
    return _canonicalize_prefactor_poles(
        poles, real_tol=_validation_tolerance(guard=5),
        relative=True,
    )


def _crossing_conjugates(p, q, distance):
    # Canonicalized poles are finite. Require opposite half-planes and
    # agreement relative to their widths, independent of a large real part.
    return p.imag*q.imag < 0 and distance < min(_pair_tol(p), _pair_tol(q))


def _split_real_complex(poles):
    """Partition `poles` into (real_poles, complex_poles) by imaginary part."""
    real_poles: list = []
    complex_poles: list = []
    for z in _crossing_poles(poles):
        if z.imag == 0:
            real_poles.append(z.real)
        else:
            complex_poles.append(z)
    return real_poles, complex_poles


def _pop_conjugate(p: mp.mpc, pool: list) -> mp.mpc:
    """Remove and return the entry of `pool` closest to conj(p).

    Raises ValueError if no opposite-half-plane entry agrees with the
    conjugate, including relative agreement at narrow peaks.
    """
    target = _conjugate_pole(p)
    best_idx = None
    best_dist = None
    tolerance = _pair_tol(p)
    for i, q in enumerate(pool):
        if p.imag*q.imag >= 0:
            continue
        dist = abs(q - target)
        if dist >= tolerance or best_dist is not None and dist >= best_dist:
            continue
        if not _crossing_conjugates(p, q, dist):
            continue
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_idx = i
    if best_idx is None:
        raise ValueError(
            f"complex pole {p} has no conjugate partner in pole_list_x; "
            "block must be real on the real axis, so every complex pole "
            "should appear with its conjugate"
        )
    return pool.pop(best_idx)


@_at_block_precision
def drop_complex_pole_pairs(serialized: SerializedDerivTable) -> SerializedDerivTable:
    """Return a new table whose `pole_list_x` contains only real poles.

    Each complex pole in the input must appear together with its conjugate
    (with multiplicity); a `ValueError` is raised otherwise. The conjugate
    pairs are dropped from the denominator while the numerators are left
    unchanged. The returned table represents

        block_deriv(Delta) * prod_pairs (x - p_k)(x - p_k*)

    which equals `block_deriv(Delta) * prod_pairs Q_k(x)` with each
    `Q_k(x) > 0` on the real axis -- the same sign as the original on the
    real Delta axis, hence positivity-equivalent.

    The two pieces of metadata that downstream code needs to remember about
    this rescaling are
      - the list of complex poles that were dropped (which determines the
        Q_k factors), and
      - the fact that `block_deriv` no longer evaluates to the actual block.
    The Q_k factors are not stored explicitly: they are recoverable from the
    set difference between the input and output `pole_list_x`.
    """
    real_poles, complex_remaining = _split_real_complex(serialized.pole_list_x)

    # Validate that every complex pole has a conjugate partner; reject otherwise.
    while complex_remaining:
        p = complex_remaining.pop(0)
        _pop_conjugate(p, complex_remaining)

    return SerializedDerivTable(
        shift=serialized.shift,
        base=serialized.base,
        pole_list_x=tuple(real_poles),
        coordinates=serialized.coordinates,
        numerators=serialized.numerators,
        sample_points_x=serialized.sample_points_x,
    )
#
# For an identical-scalar 4-point function, the crossing-symmetric combination
# is built from v^{Delta_phi} G(u, v), with u = z*zbar and v = (1-z)(1-zbar).
# At the central point z = zbar = 1/2 we have v = 1/4 and v^{Delta_phi} expands
# as a product of two univariate Taylor series:
#
#   v^{Delta_phi} = (1/2 - a)^{Delta_phi} * (1/2 - b)^{Delta_phi},
#   a = z - 1/2, b = zbar - 1/2.
#
# The block derivative table stores Taylor coefficients (derivs[m][n] =
# (1/(m!n!)) d^{m+n} G / dz^m dzbar^n at z = zbar = 1/2; cf. virasoro_blocks/
# derivs.py:12), so the Leibniz rule for the product is a plain 2D convolution
# without binomial weights:
#
#   (v^{Delta_phi} G)_{m,n} = sum_{i<=m, j<=n} P_{i,j} G_{m-i, n-j}
#
# where P is the prefactor's Taylor coefficient table.
# ---------------------------------------------------------------------------


def _crossing_number(value, *, name):
    import numpy as np
    from numbers import Integral

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number, not a boolean")
    try:
        if isinstance(value, Integral):
            value = int(value)
        value = parse_number(value)
        if not mp.isfinite(value):
            raise ValueError("nonfinite number")
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    return value


def _crossing_real_number(value, *, name):
    value = _crossing_number(value, name=name)
    label = real_input_label(value, name=name)
    # Validation uses exact components; arithmetic retains native source bits
    # without treating every bit of an ordinary mpf as an input offset.
    if isinstance(value, mp.mpc):
        return value.real
    return value if isinstance(value, mp.mpf) else label


def _univariate_v_factor_taylor(Delta_phi, m_max: int) -> list:
    """Taylor coefficients of (1/2 - t)^{Delta_phi} at t = 0, up to t^{m_max}.

    Built recursively via the generalized binomial:
        f_0     = (1/2)^{Delta_phi},
        f_{k+1} = f_k * (-2) * (Delta_phi - k) / (k + 1).
    Returns an `m_max + 1` list of `mp.mpf` (or `mp.mpc` if Delta_phi is complex).
    """
    if m_max < 0:
        raise ValueError("m_max must be >= 0")
    Delta = mp.mpc(Delta_phi)
    field = mp.mpf if mp.im(Delta) == 0 else mp.mpc
    Delta = field(mp.re(Delta)) if field is mp.mpf else mp.mpc(Delta_phi)
    f = [field(0)] * (m_max + 1)
    f[0] = field(mp.mpf("0.5")) ** Delta
    for k in range(m_max):
        f[k + 1] = f[k] * field(-2) * (Delta - k) / (k + 1)
    return f


def v_prefactor_taylor(Delta_phi, m_max: int, n_max: int) -> list:
    """2D Taylor coefficients of v^{Delta_phi} at z = zbar = 1/2.

    `v = (1 - z)(1 - zbar)` factorizes as `(1/2 - a)(1/2 - b)` with
    `a = z - 1/2`, `b = zbar - 1/2`, so v^{Delta_phi} is the outer product of
    the two univariate expansions of `(1/2 - t)^{Delta_phi}` at t = 0.

    Returns an (m_max + 1) x (n_max + 1) list of lists; entry [m][n] is the
    coefficient of `a^m b^n`. Cutoffs are nonnegative integers.
    """
    m_max = _integer_control("m_max", m_max)
    n_max = _integer_control("n_max", n_max)
    Delta_phi = _crossing_number(Delta_phi, name="Delta_phi")
    f_m = _univariate_v_factor_taylor(Delta_phi, m_max)
    f_n = _univariate_v_factor_taylor(Delta_phi, n_max)
    return [[f_m[m] * f_n[n] for n in range(n_max + 1)] for m in range(m_max + 1)]


def _weighted_numerator_sum(terms, snapshots):
    """Accumulate weighted stored polynomials in balls at the working precision."""
    from flint import arb_poly, acb_poly, ctx as flint_ctx
    from virasoro_blocks.polynomial import _mpf_to_arb, _mpf_to_acb, _arb_to_mpf, _acb_to_mpc, _flint_prec_bits

    field = mp.mpc if any(poly.field is mp.mpc or isinstance(coef, (complex, mp.mpc))
                         for coef, poly in terms) else mp.mpf
    poly_type = acb_poly if field is mp.mpc else arb_poly
    to_ball = _mpf_to_acb if field is mp.mpc else _mpf_to_arb
    from_ball = _acb_to_mpc if field is mp.mpc else _arb_to_mpf
    saved_prec = flint_ctx.prec
    try:
        precision = _flint_prec_bits()
        flint_ctx.prec = precision
        total = poly_type([0])
        for coef, poly in terms:
            source = snapshots.get(id(poly))
            if source is None:
                source = snapshots[id(poly)] = Polynomial._from_coeffs_trusted(list(poly.coeffs), field=poly.field)
            values = source._get_acb(precision) if field is mp.mpc else source._get_arb(precision)
            total += values * to_ball(coef)
        return Polynomial._from_coeffs_trusted([+from_ball(c) for c in total.coeffs()], field=field)
    finally:
        flint_ctx.prec = saved_prec


@_at_block_precision
def apply_prefactor_to_table(
    serialized: SerializedDerivTable, prefactor_taylor: list
) -> SerializedDerivTable:
    """Multiply every coordinate's per-(m, n) numerator by a Taylor prefactor.

    `prefactor_taylor[i][j]` is the (a^i b^j) Taylor coefficient of a function
    of the local coordinates `a, b = z - 1/2, zbar - 1/2` (or the analogous
    pair for `yyb` / `qqb`). The Leibniz rule on Taylor coefficients gives

        new_numerators[coord][m][n](x)
          = sum_{i=0..m, j=0..n} prefactor_taylor[i][j] * numerators[coord][m-i][n-j](x)

    so the new numerator is again a polynomial in x. The damped-rational
    prefactor (`shift`, `base`, `pole_list_x`) is independent of the cross-ratio
    prefactor and is carried through unchanged.

    Weighted sums are accumulated in balls before rounding.

    The output table has the same (m, n) shape as the input. Entries of
    `prefactor_taylor` beyond that shape are ignored (they would only affect
    higher-(m, n) entries, which we do not produce); entries below it are zero
    by truncation. Both the prefactor and each coordinate's numerator grid
    must be rectangular; ragged grids are rejected rather than truncated.
    """
    if not serialized.numerators:
        raise ValueError("serialized has no coordinates")
    if not prefactor_taylor or not prefactor_taylor[0]:
        raise ValueError("prefactor_taylor must be a non-empty (m+1) x (n+1) table")

    pre_m_max = len(prefactor_taylor) - 1
    pre_n_max = len(prefactor_taylor[0]) - 1
    if any(len(row) != pre_n_max + 1 for row in prefactor_taylor):
        raise ValueError("prefactor_taylor must be rectangular")

    used_m = min(pre_m_max+1, max(len(rows) for rows in serialized.numerators.values()))
    used_n = min(pre_n_max+1, max(len(rows[0]) if rows else 0 for rows in serialized.numerators.values()))
    prefactor_taylor = [
        [_crossing_number(coef, name="prefactor_taylor coefficient")
         if any(i < len(rows) and j < len(rows[0]) for rows in serialized.numerators.values()) else 0
         for j, coef in enumerate(row[:used_n])]
        for i, row in enumerate(prefactor_taylor[:used_m])
    ]
    snapshots = {}
    new_numerators: dict = {}
    for coord, rows in serialized.numerators.items():
        m_size = len(rows)
        n_size = len(rows[0]) if rows else 0
        if any(len(row) != n_size for row in rows):
            raise ValueError(f"numerators[{coord!r}] must be rectangular")
        new_rows = []
        for m in range(m_size):
            new_row = []
            for n in range(n_size):
                # 2D convolution over i in [0, min(m, pre_m_max)],
                # j in [0, min(n, pre_n_max)].
                terms = []
                for i in range(min(m, pre_m_max) + 1):
                    for j in range(min(n, pre_n_max) + 1):
                        coef = prefactor_taylor[i][j]
                        if coef == 0:
                            continue
                        terms.append((coef, rows[m - i][n - j]))
                if terms:
                    accumulator = _weighted_numerator_sum(terms, snapshots)
                else:
                    # Every contributing prefactor entry was zero (e.g. degree
                    # mismatch). Fall back to a zero polynomial in the same
                    # field as the input numerator so downstream code does not
                    # need to handle a missing entry.
                    accumulator = Polynomial.zero(field=rows[m][n].field)
                new_row.append(accumulator)
            new_rows.append(new_row)
        new_numerators[coord] = new_rows

    return SerializedDerivTable(
        shift=serialized.shift,
        base=serialized.base,
        pole_list_x=serialized.pole_list_x,
        coordinates=serialized.coordinates,
        numerators=new_numerators,
        sample_points_x=serialized.sample_points_x,
        working_dps=serialized.working_dps,
    )


@_at_block_precision
def apply_norm_base_scaling(
    serialized: SerializedDerivTable,
    chiral_norm,
) -> SerializedDerivTable:
    """Apply the N^Δ physical normalization as a base and polynomial scaling.

    The physical crossing equation is Σ C² × N^Δ × G_new(Δ) = 0. In the
    PMP format, N^Δ modifies the DampedRational:

        effective_base = N × exp_base
        effective_poly(x) = N^shift × poly(x)

    where x = Δ - shift. This function returns a new SerializedDerivTable
    with the adjusted base and scaled polynomials. N must be positive, finite,
    and real. A nonzero base phase is retained; PMP exports require real data.
    """
    N = _crossing_real_number(chiral_norm, name="chiral_norm")
    if N <= 0:
        raise ValueError("chiral_norm must be positive")
    shift = _crossing_real_number(serialized.shift, name="shift")
    base = _crossing_number(serialized.base, name="base")
    work = mp.mp.dps
    N_numeric = mp.convert(N)
    new_base = N_numeric * mp.convert(base)
    if isinstance(new_base, mp.mpc) and new_base.imag == 0:
        new_base = new_base.real
    scale = N_numeric ** mp.convert(shift)
    new_numerators = {
        coord: [[poly * scale for poly in row] for row in rows]
        for coord, rows in serialized.numerators.items()
    }

    return SerializedDerivTable(
        shift=serialized.shift,
        base=new_base,
        pole_list_x=serialized.pole_list_x,
        coordinates=serialized.coordinates,
        numerators=new_numerators,
        sample_points_x=serialized.sample_points_x,
        working_dps=work,
    )


# ---------------------------------------------------------------------------
# Positive-real-pole lowering on the serialized/bootstrap surface.
# ---------------------------------------------------------------------------


def _cast_field(value, *, field):
    value = mp.mpc(parse_number(value))
    if field is mp.mpf:
        if abs(mp.im(value)) >= _imag_tol(value):
            raise ValueError(f"expected a real value, got {value}")
        return mp.mpf(mp.re(value))
    return value


def _poly_as_field(poly: Polynomial, *, field) -> Polynomial:
    if poly.field is field:
        return poly
    return Polynomial(
        [_cast_field(coeff, field=field) for coeff in poly.coeffs],
        field=field,
    )


def _serialized_data_field(*blocks: SerializedDerivTable):
    for block in blocks:
        for pole in _crossing_poles(block.pole_list_x):
            if pole.imag != 0:
                return mp.mpc
        for rows in block.numerators.values():
            for row in rows:
                for poly in row:
                    for coeff in poly.coeffs:
                        if abs(mp.im(mp.mpc(coeff))) >= _imag_tol(coeff):
                            return mp.mpc
    return mp.mpf


def _group_real_roots(roots) -> list[tuple[mp.mpf, int]]:
    grouped: list[list[object]] = []
    for root in sorted(mp.mpf(mp.re(mp.mpc(r))) for r in roots):
        if grouped and root == grouped[-1][0]:
            grouped[-1][1] += 1
        else:
            grouped.append([root, 1])
    return [(value, mult) for value, mult in grouped]


@_at_block_precision
def positive_real_root_groups(
    serialized: SerializedDerivTable,
) -> list[tuple[mp.mpf, int]]:
    """Group strictly positive real poles, combining only equal stored roots."""
    roots = []
    for z in _crossing_poles(serialized.pole_list_x):
        if z.imag == 0 and z.real > 0:
            roots.append(z.real)
    return _group_real_roots(roots)


@_at_block_precision
def select_positive_real_root_groups(
    serialized: SerializedDerivTable,
    count: int | None,
) -> list[tuple[mp.mpf, int]]:
    """Select up to `count` positive real pole groups from the serialized block."""
    groups = positive_real_root_groups(serialized)
    if count is None:
        return groups
    count = _integer_field("strip_count", count)
    if count < 0:
        raise ValueError("strip_count must be >= 0")
    if count > len(groups):
        raise ValueError(
            f"requested {count} positive real pole groups, but only found "
            f"{len(groups)}"
        )
    return groups[:count]


def _factor_from_roots_linear(roots, *, field) -> Polynomial:
    factor = Polynomial.one(field=field)
    for root in roots:
        factor = factor * Polynomial.x_minus(_cast_field(root, field=field), field=field)
    return factor


def _factor_from_roots(roots, *, field) -> Polynomial:
    # When the roots are conjugate-symmetric, multiplying real quadratics keeps
    # cancellation residue out of intermediate coefficients; otherwise fall back
    # to the complex-linear product.
    real_coeffs = real_factor_coeffs_from_conjugate_roots(roots)
    if real_coeffs is not None:
        return Polynomial(
            [_cast_field(coeff, field=field) for coeff in real_coeffs],
            field=field,
        )
    return _factor_from_roots_linear(roots, field=field)


def _positive_clearing_exponent(multiplicity: int) -> int:
    multiplicity = _integer_field("pole multiplicity", multiplicity)
    if multiplicity < 1:
        raise ValueError("multiplicity must be >= 1")
    return multiplicity if multiplicity % 2 == 0 else multiplicity + 1


def _positive_clearing_factor(selected_groups, *, field) -> Polynomial:
    roots = []
    for root, multiplicity in selected_groups:
        roots.extend([root] * _positive_clearing_exponent(multiplicity))
    return _factor_from_roots(roots, field=field)


def clearing_factor_value(selected_groups, x):
    """The cleared/original block ratio ``prod (x - p)^e`` at ``x``, in product form (see
    :func:`_positive_clearing_factor`)."""
    selected_groups = tuple(selected_groups)
    if not selected_groups:
        return mp.mpf(1)
    numbers = [_evaluation_number(parse_number(v)) for v in (x, *(root for root, _ in selected_groups))]
    value = mp.mpf(1)
    for root, (_, multiplicity) in zip(numbers[1:], selected_groups):
        value *= (numbers[0] - root) ** _positive_clearing_exponent(multiplicity)
    return value


def _odd_multiplicity_extra_factor(selected_groups, *, field) -> Polynomial:
    roots = []
    for root, multiplicity in selected_groups:
        if multiplicity % 2 == 1:
            roots.append(root)
    return _factor_from_roots(roots, field=field)


def _cleanup_remaining_roots(roots) -> list:
    """Symmetrize already classified roots after selection/removal.

    - Real roots are projected to mpf; classified complex roots stay complex.
    - Matched conjugate pairs are symmetrized to exact conjugates by averaging
      their real parts and imaginary magnitudes.
    """
    # Classification happened before removal. Removing a partner must not
    # make a previously explicit thin pair look like lone numerical noise.
    roots_c = [mp.mpc(root) for root in roots]
    used = [False] * len(roots_c)
    cleaned = []

    for i, root in enumerate(roots_c):
        if used[i]:
            continue

        if root.imag == 0:
            used[i] = True
            cleaned.append(mp.mpf(mp.re(root)))
            continue

        target = _conjugate_pole(root)
        partner = None
        best_dist = None
        tolerance = _pair_tol(root)
        for j in range(i + 1, len(roots_c)):
            if used[j] or root.imag*roots_c[j].imag >= 0:
                continue
            dist = abs(roots_c[j] - target)
            if dist >= tolerance or best_dist is not None and dist >= best_dist:
                continue
            if not _crossing_conjugates(root, roots_c[j], dist):
                continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
                partner = j

        if partner is None:
            used[i] = True
            cleaned.append(root)
            continue

        used[i] = True
        used[partner] = True
        partner_root = roots_c[partner]
        re = (mp.re(root) + mp.re(partner_root)) / 2
        im_abs = (abs(mp.im(root)) + abs(mp.im(partner_root))) / 2
        if mp.im(root) >= 0:
            cleaned.extend([mp.mpc(re, im_abs), mp.mpc(re, -im_abs)])
        else:
            cleaned.extend([mp.mpc(re, -im_abs), mp.mpc(re, im_abs)])

    return cleaned


def _remove_selected_root_groups(all_roots, selected_groups) -> list:
    remaining = list(_crossing_poles(tuple(parse_number(root) for root in all_roots)))
    # Match the same real projection used when selecting pole groups.
    remaining = [mp.re(root) if root.imag == 0 else root for root in remaining]
    for selected, multiplicity in selected_groups:
        selected = mp.mpc(mp.convert(parse_number(selected)))
        for _ in range(multiplicity):
            for index, root in enumerate(remaining):
                if abs(root - selected) < _imag_tol(selected):
                    remaining.pop(index)
                    break
            else:
                raise ValueError(
                    f"selected root {selected} with multiplicity {multiplicity} "
                    "not found in pole_list_x"
                )
    return _cleanup_remaining_roots(remaining)


def _local_series(poly: Polynomial, center, order: int) -> list:
    local = poly.compose_linear(1, center)
    coeffs = list(local.coeffs[:order])
    coeffs.extend([local.field(0)] * (order - len(coeffs)))
    return coeffs


def _series_multiply_truncated(lhs, rhs, order: int, *, field) -> list:
    out = [field(0)] * order
    for i in range(min(len(lhs), order)):
        if lhs[i] == 0:
            continue
        for j in range(min(len(rhs), order - i)):
            out[i + j] += lhs[i] * rhs[j]
    return out


def _series_inverse_truncated(series, order: int, *, field) -> list:
    if order < 1:
        raise ValueError("order must be >= 1")
    if series[0] == 0:
        raise ValueError("series does not have an invertible constant term")
    inverse = [field(0)] * order
    inverse[0] = field(1) / series[0]
    for n in range(1, order):
        total = field(0)
        for k in range(1, n + 1):
            total += series[k] * inverse[n - k]
        inverse[n] = -total / series[0]
    return inverse


def _local_series_to_global_poly(coeffs, center, *, field) -> Polynomial:
    return Polynomial(
        [_cast_field(coeff, field=field) for coeff in coeffs],
        field=field,
    ).compose_linear(1, -center)


def _quotient_by_power(poly: Polynomial, root, multiplicity: int) -> Polynomial:
    quotient = poly
    for _ in range(multiplicity):
        # Zero polynomial is divisible by any linear factor to any power.
        if quotient.degree == 0 and quotient.coeffs[0] == 0:
            return quotient
        quotient = quotient.quotient_by_linear(root)
    return quotient


def _split_serialized_block_by_simple_poles_fast(
    serialized: SerializedDerivTable,
    selected_roots,
) -> tuple[SerializedDerivTable, SerializedDerivTable]:
    if not selected_roots:
        raise ValueError("selected_roots must be non-empty")

    work_field = _serialized_data_field(serialized)
    selected_roots = [_cast_field(root, field=work_field) for root in selected_roots]
    full_den = _factor_from_roots(serialized.pole_list_x, field=work_field)
    selected_den = _factor_from_roots(selected_roots, field=work_field)
    remainder_poles = _remove_selected_root_groups(
        serialized.pole_list_x,
        [(root, 1) for root in selected_roots],
    )
    remainder_den = _factor_from_roots(remainder_poles, field=work_field)

    if not remainder_poles and all(
        num.degree < len(selected_roots)
        for rows in serialized.numerators.values() for row in rows for num in row
    ):
        # A proper fraction whose entire denominator was selected is already
        # its pole part. Reconstructing it by residues introduces cancellation
        # noise and can leave a nonzero constant that cannot be divided by S.
        return serialized, SerializedDerivTable(
            shift=serialized.shift, base=serialized.base, pole_list_x=(),
            coordinates=serialized.coordinates,
            numerators={coord: [[Polynomial.zero(field=work_field) for _ in row]
                                for row in rows]
                        for coord, rows in serialized.numerators.items()},
        )

    selected_basis = [selected_den.quotient_by_linear(root) for root in selected_roots]
    denom_quotients = [full_den.quotient_by_linear(root) for root in selected_roots]
    derivative_values = [
        denom_quot.evaluate(root)
        for root, denom_quot in zip(selected_roots, denom_quotients)
    ]
    for root, derivative_at_root in zip(selected_roots, derivative_values):
        if derivative_at_root == 0:
            raise ValueError(
                f"selected root {root} is not simple enough for the fast splitter"
            )

    selected_numerators: dict = {}
    remainder_numerators: dict = {}
    for coord, rows in serialized.numerators.items():
        selected_rows = []
        remainder_rows = []
        for row in rows:
            selected_row = []
            remainder_row = []
            for num in row:
                num = _poly_as_field(num, field=work_field)
                if not remainder_poles and num.degree < selected_den.degree:
                    selected_row.append(num)
                    remainder_row.append(Polynomial.zero(field=work_field))
                    continue
                selected_num = Polynomial.zero(field=work_field)
                for root, basis, derivative_at_root in zip(
                    selected_roots,
                    selected_basis,
                    derivative_values,
                ):
                    coefficient = num.evaluate(root) / derivative_at_root
                    selected_num = selected_num + basis.scale(coefficient)

                remainder_num = num - selected_num * remainder_den
                for root in selected_roots:
                    # A zero numerator (e.g. the vanishing n > m half of an
                    # identical-scalar spin >= 2 channel) stays zero; synthetic
                    # division refuses a constant, so pass it through.
                    if remainder_num.degree == 0 and remainder_num.coeffs[0] == 0:
                        break
                    remainder_num = remainder_num.quotient_by_linear(root)

                selected_row.append(selected_num)
                remainder_row.append(remainder_num)
            selected_rows.append(selected_row)
            remainder_rows.append(remainder_row)
        selected_numerators[coord] = selected_rows
        remainder_numerators[coord] = remainder_rows

    selected_block = SerializedDerivTable(
        shift=serialized.shift,
        base=serialized.base,
        pole_list_x=tuple(selected_roots),
        coordinates=serialized.coordinates,
        numerators=selected_numerators,
    )
    remainder_block = SerializedDerivTable(
        shift=serialized.shift,
        base=serialized.base,
        pole_list_x=tuple(remainder_poles),
        coordinates=serialized.coordinates,
        numerators=remainder_numerators,
    )
    return selected_block, remainder_block


@_at_block_precision
def split_serialized_block_by_pole_groups(
    serialized: SerializedDerivTable, selected_groups,
) -> tuple[SerializedDerivTable, SerializedDerivTable]:
    """Split a stored block into the selected pole part and its remainder."""
    selected_groups = _validated_pole_groups(selected_groups)
    # Freeze the existing imaginary-noise projection before adding an
    # arithmetic guard. A tighter tolerance inside that guard must not turn
    # an already selected real pole back into a distinct complex root.
    roots = tuple(root.real if root.imag == 0 else root
                  for root in _crossing_poles(serialized.pole_list_x))
    if roots != serialized.pole_list_x or any(not isinstance(root, (mp.mpf, mp.mpc))
                                             for root in serialized.pole_list_x):
        serialized = replace(serialized, pole_list_x=roots)
    return _split_serialized_block_by_pole_groups(serialized, selected_groups)


def _validated_pole_groups(selected_groups):
    result = []
    for root, multiplicity in selected_groups:
        try:
            integer = _integer_field("pole multiplicity", multiplicity)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("pole multiplicity must be a positive integer") from exc
        if integer < 1:
            raise ValueError("pole multiplicity must be a positive integer")
        result.append((parse_number(root), integer))
    return tuple(result)


def _split_serialized_block_by_pole_groups(
    serialized: SerializedDerivTable,
    selected_groups,
) -> tuple[SerializedDerivTable, SerializedDerivTable]:
    """Split `serialized` into the selected pole tower and its regular remainder."""
    if not selected_groups:
        raise ValueError("selected_groups must be non-empty")

    work_field = _serialized_data_field(serialized)
    selected_groups = [(_cast_field(root, field=work_field), multiplicity)
                       for root, multiplicity in selected_groups]
    if all(multiplicity == 1 for _, multiplicity in selected_groups):
        return _split_serialized_block_by_simple_poles_fast(
            serialized,
            [root for root, _ in selected_groups],
        )

    full_den = _factor_from_roots(serialized.pole_list_x, field=work_field)
    selected_roots = [
        root for root, multiplicity in selected_groups for _ in range(multiplicity)
    ]
    selected_den = _factor_from_roots(selected_roots, field=work_field)
    remainder_poles = _remove_selected_root_groups(
        serialized.pole_list_x,
        selected_groups,
    )
    remainder_den = _factor_from_roots(remainder_poles, field=work_field)

    group_data = []
    for root, multiplicity in selected_groups:
        basis_den = _quotient_by_power(selected_den, root, multiplicity)
        remainder_den_local = _local_series(remainder_den, root, multiplicity)
        basis_local = _local_series(basis_den, root, multiplicity)
        group_data.append(
            {
                "root": root,
                "multiplicity": multiplicity,
                "basis_den": basis_den,
                "remainder_den_inv": _series_inverse_truncated(
                    remainder_den_local,
                    multiplicity,
                    field=work_field,
                ),
                "basis_inv": _series_inverse_truncated(
                    basis_local,
                    multiplicity,
                    field=work_field,
                ),
            }
        )

    selected_numerators: dict = {}
    remainder_numerators: dict = {}
    for coord, rows in serialized.numerators.items():
        selected_rows = []
        remainder_rows = []
        for row in rows:
            selected_row = []
            remainder_row = []
            for num in row:
                num = _poly_as_field(num, field=work_field)
                if not remainder_poles and num.degree < selected_den.degree:
                    selected_row.append(num)
                    remainder_row.append(Polynomial.zero(field=work_field))
                    continue
                selected_num = Polynomial.zero(field=work_field)
                for group in group_data:
                    root = group["root"]
                    multiplicity = group["multiplicity"]
                    target_local = _series_multiply_truncated(
                        _local_series(num, root, multiplicity),
                        group["remainder_den_inv"],
                        multiplicity,
                        field=work_field,
                    )
                    basis_coeffs = _series_multiply_truncated(
                        target_local,
                        group["basis_inv"],
                        multiplicity,
                        field=work_field,
                    )
                    basis_poly = _local_series_to_global_poly(
                        basis_coeffs,
                        root,
                        field=work_field,
                    )
                    selected_num = selected_num + group["basis_den"] * basis_poly

                remainder_num = num - selected_num * remainder_den
                for root, multiplicity in selected_groups:
                    remainder_num = _quotient_by_power(
                        remainder_num,
                        root,
                        multiplicity,
                    )

                selected_row.append(selected_num)
                remainder_row.append(remainder_num)
            selected_rows.append(selected_row)
            remainder_rows.append(remainder_row)
        selected_numerators[coord] = selected_rows
        remainder_numerators[coord] = remainder_rows

    selected_block = SerializedDerivTable(
        shift=serialized.shift,
        base=serialized.base,
        pole_list_x=tuple(selected_roots),
        coordinates=serialized.coordinates,
        numerators=selected_numerators,
    )
    remainder_block = SerializedDerivTable(
        shift=serialized.shift,
        base=serialized.base,
        pole_list_x=tuple(remainder_poles),
        coordinates=serialized.coordinates,
        numerators=remainder_numerators,
    )
    return selected_block, remainder_block


def _max_numerator_degree(block: SerializedDerivTable, coordinate: str) -> int:
    rows = block.numerators[coordinate]
    return max(poly.degree for row in rows for poly in row)


def _max_numerator_degree_all(block: SerializedDerivTable) -> int:
    return max(_max_numerator_degree(block, coord) for coord in block.coordinates)


def _serialized_prefactor(block: SerializedDerivTable) -> DampedRational:
    return DampedRational(
        constant=mp.mpf(1),
        poles=tuple(block.pole_list_x),
        base=block.base,
    )


def _serialized_residue_scores(
    serialized: SerializedDerivTable, *, score_entries: str = "kept"
) -> list:
    """Serialized-table analogue of `polyderivs._residue_scores`.

    Duplicate detection uses the stored pole values. Distinct nearby poles
    retain their separation in the residue product; exact Laurent tables
    already supply the multiplicities at singular central charges.

    ``score_entries`` selects which numerator entries the residue magnitude
    maxes over: ``"kept"`` (every (m, n) entry, used by ``residue_all``)
    or ``"value"`` (only the (0, 0) undifferentiated block entry per
    coordinate, for ``residue``). Scoring poles by their residue in the block
    *value* keeps the poles that matter most to the physical block rather than
    being dominated by the highest-derivative residues that over-weight deep
    poles (the same finding as the Delta-channel ``residue`` selector).
    """
    if score_entries not in ("kept", "value"):
        raise ValueError(
            f"score_entries must be 'kept' or 'value'; got {score_entries!r}"
        )
    roots = [mp.mpc(parse_number(root)) for root in serialized.pole_list_x]
    scores = []
    for j, pole in enumerate(roots):
        product = mp.mpc(1)
        for k, other in enumerate(roots):
            if k != j and pole != other:
                product *= pole - other
        best = mp.mpf(0)
        for coord in serialized.coordinates:
            if score_entries == "value":
                num = serialized.numerators[coord][0][0]
                best = max(best, abs(num.evaluate(pole) / product))
            else:
                for row in serialized.numerators[coord]:
                    for num in row:
                        best = max(best, abs(num.evaluate(pole) / product))
        scores.append(best)
    return scores



@_at_block_precision
def interpolate_serialized_block(
    serialized: SerializedDerivTable,
    npoles: int,
    *,
    pole_selection: str = "rightmost",
    n_sample_points: int | None = None,
    sample_point_kwargs: dict | None = None,
) -> tuple[SerializedDerivTable, tuple]:
    """Interpolate a serialized block down to an admissible shared pole cap.

    The values at the new nodes come from the block's polynomial form: this shapes the polynomial written to the
    PMP (read by the pmp2sdp backend). The sdp backend does not use it -- its data at the nodes are the numeric
    block values there (``identical_scalar._clean_sample_values_by_block``).
    """
    npoles = _integer_control("npoles", npoles)
    if n_sample_points is not None:
        n_sample_points = _integer_control("n_sample_points", n_sample_points, minimum=1)

    full = _serialized_prefactor(serialized)
    if pole_selection == "residue":
        scores = _serialized_residue_scores(serialized, score_entries="value")
    elif pole_selection == "residue_all":
        scores = _serialized_residue_scores(serialized, score_entries="kept")
    else:
        scores = None

    def _select_kept(cap: int) -> DampedRational:
        if pole_selection == "rightmost":
            return full.take_rightmost_poles(cap)
        if pole_selection == "nearest":
            return full.take_nearest_poles(cap)
        if pole_selection in ("residue", "residue_all"):
            return full.take_strongest_poles(cap, scores)
        raise ValueError(
            "pole_selection must be 'rightmost', 'nearest', 'residue' or "
            "'residue_all'"
        )

    max_num_deg = _max_numerator_degree_all(serialized)
    denom_full_degree = len(full.poles)
    sample_point_kwargs = dict(sample_point_kwargs or {})

    last_sample_error = None
    kept = None
    sample_points_x = None
    for kept_cap in range(npoles, -1, -1):
        candidate = _select_kept(kept_cap)
        candidate_n_sample_points = n_sample_points
        if candidate_n_sample_points is None:
            dropped_pole_count = denom_full_degree - candidate.pole_degree
            candidate_n_sample_points = max(
                1,
                max_num_deg - dropped_pole_count + 1,
            )
        try:
            candidate_sample_points_x = tuple(
                get_sample_points(
                    candidate,
                    candidate_n_sample_points,
                    **sample_point_kwargs,
                )
            )
        except ValueError as exc:
            if "Re(pole) <= 0" in str(exc):
                raise
            last_sample_error = exc
            continue
        kept = candidate
        sample_points_x = candidate_sample_points_x
        break

    if kept is None or sample_points_x is None:
        if last_sample_error is not None:
            raise last_sample_error
        raise ValueError("failed to construct an admissible interpolation prefactor")

    numerators = {}
    kept_at_nodes = [kept.evaluate(x) for x in sample_points_x]
    for coord in serialized.coordinates:
        rows = serialized.numerators[coord]
        tables_at_nodes = [serialized.evaluate(coord, serialized.shift + x) for x in sample_points_x]
        interp_rows = []
        for m, row in enumerate(rows):
            interp_row = []
            for n, _poly in enumerate(row):
                values = [table[m][n] / kept_value for table, kept_value in zip(tables_at_nodes, kept_at_nodes)]
                interp_row.append(
                    lagrange_interpolation_polynomial(
                        sample_points_x,
                        values,
                        shift=0,
                    )
                )
            interp_rows.append(interp_row)
        numerators[coord] = interp_rows

    return (
        SerializedDerivTable(
            shift=serialized.shift,
            base=serialized.base,
            pole_list_x=tuple(kept.poles),
            coordinates=serialized.coordinates,
            numerators=numerators,
        ),
        sample_points_x,
    )


@_at_block_precision
def build_positive_cleared_block(
    selected_block: SerializedDerivTable,
    remainder_block: SerializedDerivTable,
    selected_groups,
) -> SerializedDerivTable:
    """Multiply a split block by its nonnegative positive-pole clearing factor."""
    selected_groups = _validated_pole_groups(selected_groups)
    return _build_positive_cleared_block(selected_block, remainder_block, selected_groups)


def _build_positive_cleared_block(
    selected_block: SerializedDerivTable,
    remainder_block: SerializedDerivTable,
    selected_groups,
) -> SerializedDerivTable:
    """Multiply the original block by an everywhere-nonnegative clearing factor."""
    if selected_block.shift != remainder_block.shift:
        raise ValueError("selected and remainder blocks disagree on shift")
    if selected_block.base != remainder_block.base:
        raise ValueError("selected and remainder blocks disagree on base")
    if selected_block.coordinates != remainder_block.coordinates:
        raise ValueError("selected and remainder blocks disagree on coordinates")
    for coord in selected_block.coordinates:
        selected_rows = selected_block.numerators[coord]
        remainder_rows = remainder_block.numerators[coord]
        if (len(selected_rows) != len(remainder_rows)
                or any(len(lhs) != len(rhs) for lhs, rhs in zip(selected_rows, remainder_rows))):
            raise ValueError(f"selected and remainder blocks disagree on numerator shape for {coord}")

    work_field = _serialized_data_field(selected_block, remainder_block)
    remainder_den = _factor_from_roots(remainder_block.pole_list_x, field=work_field)
    clearing_factor = _positive_clearing_factor(selected_groups, field=work_field)
    extra_factor = _odd_multiplicity_extra_factor(selected_groups, field=work_field)

    numerators = {}
    for coord in remainder_block.coordinates:
        cleared_rows = []
        selected_rows = selected_block.numerators[coord]
        remainder_rows = remainder_block.numerators[coord]
        for selected_row, remainder_row in zip(selected_rows, remainder_rows):
            cleared_row = []
            for selected_num, remainder_num in zip(selected_row, remainder_row):
                selected_num = _poly_as_field(selected_num, field=work_field)
                remainder_num = _poly_as_field(remainder_num, field=work_field)
                cleared_num = (
                    extra_factor * selected_num * remainder_den
                    + clearing_factor * remainder_num
                )
                cleared_row.append(cleared_num)
            cleared_rows.append(cleared_row)
        numerators[coord] = cleared_rows

    return SerializedDerivTable(
        shift=remainder_block.shift,
        base=remainder_block.base,
        pole_list_x=remainder_block.pole_list_x,
        coordinates=remainder_block.coordinates,
        numerators=numerators,
    )


@_at_block_precision
def lower_positive_real_poles(
    serialized: SerializedDerivTable,
    *,
    strip_count: int | None = None,
    npoles: int | None = None,
    pole_selection: str = "rightmost",
    n_sample_points: int | None = None,
    sample_point_kwargs: dict | None = None,
) -> PositivePoleLowering:
    """Lower genuine positive real poles into an explicit tower plus remainder.

    The input must already have any bootstrap-side cross-ratio prefactor folded
    into its numerator polynomials. The returned remainder is regular at the
    stripped positive real poles and is admissible for the PMP writer when all
    positive real pole groups are lowered.
    If `npoles` is provided, the helper also interpolates the remainder down to
    the largest admissible kept-pole cap at or below the requested value (not
    when there is no pole to lower and the block already keeps at most `npoles`).
    The cleared block is the original times ``prod (x - p)^e`` over the lowered
    poles (``e`` the multiplicity rounded up to even; :func:`clearing_factor_value`).
    """
    if npoles is not None:
        npoles = _integer_control("npoles", npoles)
    if n_sample_points is not None:
        n_sample_points = _integer_control("n_sample_points", n_sample_points, minimum=1)
    selected_groups = tuple(select_positive_real_root_groups(serialized, strip_count))

    if selected_groups:
        selected_block, remainder_block = split_serialized_block_by_pole_groups(
            serialized,
            selected_groups,
        )
        cleared_block = build_positive_cleared_block(
            selected_block,
            remainder_block,
            selected_groups,
        )
    else:
        selected_block = None
        remainder_block = serialized
        cleared_block = serialized

    interpolated_remainder_block = None
    cleared_interpolated_block = None
    kept_npoles = None
    sample_points_x = ()
    # without a pole to lower, a block that already keeps at most `npoles` poles is left as it is: re-fitting it
    # would only re-approximate it at new nodes (from its polynomial form) and drop its numeric node values
    if npoles is not None and (selected_groups or len(serialized.pole_list_x) > npoles):
        interpolated_remainder_block, sample_points_x = interpolate_serialized_block(
            remainder_block,
            npoles,
            pole_selection=pole_selection,
            n_sample_points=n_sample_points,
            sample_point_kwargs=sample_point_kwargs,
        )
        kept_npoles = len(interpolated_remainder_block.pole_list_x)
        cleared_interpolated_block = (
            build_positive_cleared_block(
                selected_block,
                interpolated_remainder_block,
                selected_groups,
            )
            if selected_block is not None
            else interpolated_remainder_block
        )

    return PositivePoleLowering(
        selected_groups=selected_groups,
        selected_block=selected_block,
        remainder_block=remainder_block,
        cleared_block=cleared_block,
        interpolated_remainder_block=interpolated_remainder_block,
        cleared_interpolated_block=cleared_interpolated_block,
        sample_points_x=sample_points_x,
        requested_npoles=npoles,
        kept_npoles=kept_npoles,
        pole_selection=pole_selection if npoles is not None else None,
    )


# ---------------------------------------------------------------------------
# Crossing index set.
#
# At the crossing fixed point z = zbar = 1/2, the crossing operator z -> 1-z
# acts on Taylor derivatives `(m, n)` as a sign (-1)^(m+n). For identical
# scalars on the `zzb` coordinate, the antisymmetric crossing combination
# F^-(z, zbar) = v^{Delta_phi} G(u, v) - u^{Delta_phi} G(v, u) is then twice
# `v^{Delta_phi} G` restricted to odd (m + n). F^- vanishes on the even
# entries; F^+ instead selects those even entries.
#
# On the parity-folded coordinates `xt` (folds z) and `ws` (folds y), the
# expansion variables are the crossing-even `w = (u + ubar)/2 - u_*` and the
# z<->zbar-even `s = ((u - ubar)/2)^2`. Crossing (u <-> ubar reflection of the
# *radial* variable) sends y -> -y / z -> 1-z, i.e. w -> -w and s -> s, so the
# folded monomial w^a s^b carries crossing sign (-1)^a. The F^- sector is
# therefore the odd-`a` entries, keeping the folded truncation a + 2 b <= Lambda.
# This odd-`a` folded index set has the same cardinality as the non-folded
# odd-(m + n) lower-triangle set (verified for Lambda = 4..7), as it must: the
# parity fold is an invertible change of basis on the z<->zbar-symmetric
# even-spin functional space.
# ---------------------------------------------------------------------------

_FOLDED_CROSSING_COORDINATES = frozenset({"xt", "ws"})


def F_minus_pairs(lam: int, *, coordinate: str = "zzb") -> list:
    """The (m, n) index set for F^- on crossing-odd identical-scalar coordinates.

    For the non-folded `zzb` / `yyb` derivatives, crossing acts as a linear
    sign `(-1)^(m + n)` at the symmetric point, so F^- keeps odd `(m + n)` over
    the triangular grid `n <= min(m, lam - m)`.

    For the parity-folded `xt` / `ws` coordinates, crossing acts as `(-1)^a` on
    the `(w, s)` monomial `w^a s^b`, so F^- keeps odd `a` over the folded
    truncation `a + 2 b <= lam`.
    """
    lam = _integer_field("lam", lam)
    if lam < 0:
        raise ValueError("lam must be >= 0")
    if coordinate in _FOLDED_CROSSING_COORDINATES:
        return [
            (a, b)
            for a in range(1, lam + 1, 2)
            for b in range((lam - a) // 2 + 1)
        ]
    if coordinate not in {"zzb", "yyb"}:
        raise ValueError(
            "F_minus_pairs supports identical-scalar coordinates "
            "{'zzb', 'yyb', 'xt', 'ws'}"
        )
    return [
        (m, n)
        for m in range(lam + 1)
        for n in range(min(m, lam - m) + 1)
        if (m + n) % 2 == 1
    ]
