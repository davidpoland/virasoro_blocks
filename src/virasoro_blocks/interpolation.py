"""Interpolation helpers: sample points, pole selection, Lagrange values.

This module ports the sample-point numerics from the Mathematica reference
`pmp2sdp_samplePoints_prototype_v2_cxx.wl` into a small Python library API.

Scope:
    - represent the positive prefactor as a damped rational function
    - solve for the support endpoint b of the interpolation density
    - evaluate the integrated density on [0, b]
    - generate Bohr-Sommerfeld sample points
    - build a polynomial interpolant of a prefactor-divided function of Delta
    - compare interpolation error over a Delta grid
    - the pole-selection rules (`POLE_SELECTIONS`, `select_kept_poles`)
    - Lagrange values at new points by products (`lagrange_matrix`)

The bilinear basis and the rest of the SDPB-input encoding are out of scope
for this library: they are computed downstream (by `virasoro_bootstrap`'s SDP
writer, or by `pmp2sdp`). This library is the block-table generator (the
scalar_blocks / blocks_3d analogue); it stops at the polynomial tables.
"""

from __future__ import annotations

import math as _math
from dataclasses import dataclass
from fractions import Fraction
from functools import wraps
from typing import Callable, Sequence

import mpmath as mp

from .chiral import F_block_chiral_h
from .composer import B_block, G_block
from .polynomial import Polynomial, _mpf_to_arb, _arb_to_mpf
from .precision import iteration_cap, _validation_tolerance
from .numeric_input import _evaluation_number, literal_number, parse_number, _exchange_weight, _signed_half_integer, _integer_control


# Stable marker prefixing the "near-zero real pole" error raised by
# `get_sample_points`. `interpolate_delta_tables` matches on it to distinguish a
# malformed (unregularized) prefactor -- which must propagate -- from a benign
# "too many poles" rejection that its candidate loop legitimately works around
# by keeping fewer poles.
_UNREMOVED_KAC_ZERO_MARKER = "unremoved near-zero Kac pole"


def _validate_bs_controls(function):
    """Validate the integer controls of a Bohr-Sommerfeld node routine."""
    from inspect import signature

    call_signature = signature(function)

    @wraps(function)
    def validated(*args, **kwargs):
        bound = call_signature.bind(*args, **kwargs)
        for name in ('num_sample_points', 'n_start', 'work_dps'):
            if name not in bound.arguments or (name == 'work_dps' and bound.arguments[name] is None):
                continue
            minimum = 0 if name == 'n_start' or (name == 'num_sample_points' and 'n_start' in call_signature.parameters) else 1
            bound.arguments[name] = _integer_control(name, bound.arguments[name], minimum)
        return function(*bound.args, **bound.kwargs)
    return validated


def _validate_npoles(value):
    """Normalize an integer pole cap without truncating or accepting booleans."""
    import numpy as np

    try:
        integer = int(value)
        valid = not isinstance(value, (bool, np.bool_)) and value == integer and integer >= 0
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError("npoles must be a non-negative integer")
    return integer


def _as_complex_input(value):
    """Numerical constructor conversion at the working precision."""
    value = parse_number(value)
    return mp.mpc(mp.convert(value) if isinstance(value, Fraction) else value)


def _pole_side(value):
    imaginary = mp.im(value)
    return 1 if imaginary > 0 else -1 if imaginary < 0 else 0


def _conjugate_pole(value):
    """Conjugate stored components without rounding to ambient precision."""
    from mpmath.libmp import mpf_neg

    real, imaginary = value._mpc_
    return mp.make_mpc((real, mpf_neg(imaginary)))


def _same_pole_location(left, right, tol):
    # Explicit near-real conjugate pairs must remain distinct locations,
    # including when their separation is smaller than the roundoff tolerance.
    return _pole_side(left) == _pole_side(right) and abs(left-right) < tol


def _pole_geometry_scales(poles):
    """Widths of finite peaks and separations of distinct real poles."""
    poles = tuple(_as_complex_input(pole) for pole in poles)
    if any(not mp.isfinite(pole) for pole in poles):
        raise ValueError("prefactor poles must be finite")
    real = sorted({pole.real for pole in poles if pole.imag == 0})
    real_scales = {}
    for i, value in enumerate(real):
        distances = [mp.mpf(1)]
        if i:
            distances.append(value - real[i-1])
        if i + 1 < len(real):
            distances.append(real[i+1] - value)
        real_scales[value] = min(distances)
    return {pole: min(1, abs(pole.imag)) if pole.imag else real_scales[pole.real]
            for pole in poles}


def _same_geometry_pole(left, right, tol, *, scale=1):
    """Compare shared poles without merging real poles or finite narrow peaks.

    Near-real conjugates need agreement relative to their width: an absolute
    roundoff allowance could otherwise accept twice the width, or a displaced
    peak, and change the shared rational function by order one.
    """
    left, right = _as_complex_input(left), _as_complex_input(right)
    if not (mp.isfinite(left) and mp.isfinite(right)):
        return False
    if left == right:
        return True
    if _pole_side(left) != _pole_side(right):
        return False
    if left.imag:
        scale = min(scale, abs(left.imag), abs(right.imag))
    tol *= min(1, scale)
    return abs(left-right) <= tol


def _as_real(value, *, name: str, tol=None):
    if tol is None:
        # Input validation: the caller passed something that should be exact
        # real (e.g. a support endpoint or sample x); allow only roundoff.
        tol = _validation_tolerance(guard=5)
    value = _as_complex_input(value)
    if not mp.isfinite(value):
        raise ValueError(f"{name} must be finite and real in the current interpolation implementation")
    imaginary = mp.im(value)
    if not mp.isfinite(imaginary) or abs(imaginary) > tol:
        raise ValueError(f"{name} must be real in the current interpolation implementation")
    return value.real


def _as_complex_pole(value, *, real_tol=None):
    """Return mp.mpc(value), collapsing the imaginary part to exactly 0 when
    it is small enough that the pole should be treated as real. This keeps
    `pole == 0` comparisons and Re-sorting tidy for poles that came out of a
    formula with a tiny imaginary residue."""
    if real_tol is None:
        real_tol = _validation_tolerance()
    value = _as_complex_input(value)
    if abs(mp.im(value)) < real_tol:
        return mp.make_mpc((value.real._mpf_, mp.mpf(0)._mpf_))
    return value


def _canonicalize_prefactor_poles(poles, *, real_tol=None, relative=False) -> tuple:
    """Canonicalize prefactor poles while preserving explicit conjugate pairs.

    Tiny-imaginary poles often arise from exact singular-locus regularization.
    If such a pole appears together with its explicit conjugate partner, keep
    the pair complex so interpolation treats it as a smooth positive quadratic
    factor on the real axis rather than as a real pole on the support.
    Lone near-real poles are still snapped to the real axis.
    """
    if real_tol is None:
        real_tol = _validation_tolerance()
    raw = tuple(_as_complex_input(pole) for pole in poles)
    if any(not mp.isfinite(pole) for pole in raw):
        raise ValueError("prefactor poles must be finite")
    keep_complex = [False] * len(raw)
    tolerances = [real_tol * max(1, abs(pole.real)) if relative else real_tol for pole in raw]
    for i, pole in enumerate(raw):
        if abs(mp.im(pole)) >= tolerances[i] or mp.im(pole) == 0:
            continue
        target = _conjugate_pole(pole)
        for j, other in enumerate(raw):
            if i == j:
                continue
            if mp.im(pole) * mp.im(other) >= 0:
                continue
            if abs(other - target) < tolerances[i]:
                keep_complex[i] = True
                keep_complex[j] = True
                break
    return tuple(
        raw[i] if keep_complex[i] else _as_complex_pole(raw[i], real_tol=tolerances[i])
        for i in range(len(raw))
    )


def _validate_conjugate_symmetry(
    poles: tuple, tol=None
) -> None:
    """Verify each pole with Im > 0 has its conjugate (Im < 0) in the list.

    Non-real denominator factors must occur in conjugate pairs so their
    products are real and positive on the real axis.
    """
    if tol is None:
        tol = _validation_tolerance()
    upper = [(i, p) for i, p in enumerate(poles) if mp.im(p) > 0]
    lower = [(i, p) for i, p in enumerate(poles) if mp.im(p) < 0]
    if len(upper) != len(lower):
        raise ValueError(
            "complex poles must come in conjugate pairs; got "
            f"{len(upper)} upper-half and {len(lower)} lower-half"
        )
    used = [False] * len(lower)
    for _, p in upper:
        target = _conjugate_pole(p)
        match = next(
            (k for k, (_, q) in enumerate(lower)
             if not used[k] and abs(q - target) < tol),
            None,
        )
        if match is None:
            raise ValueError(
                f"complex pole {p} has no conjugate partner in the pole list"
            )
        used[match] = True


@dataclass(frozen=True)
class DampedRational:
    """Prefactor of the form c * base**x / prod(x - pole_i).

    Poles may be real or come in complex-conjugate pairs. Conjugate symmetry
    of the pole list is enforced on construction. With a positive constant
    and real poles at or below zero, the prefactor is positive for x > 0.
    Node generation requires real poles <= 0 and a damping base in (0, 1);
    non-real conjugate pairs may have positive real parts.
    Stored mpmath data retain their precision; near-real poles still follow
    the canonicalization policy. Exact real query labels are converted only
    after resolving their distance from the poles.
    """

    constant: object
    poles: tuple
    base: object

    def __post_init__(self):
        constant = _as_complex_input(self.constant)
        if not mp.isfinite(constant):
            raise ValueError("prefactor constant must be finite")
        raw_base = _as_complex_input(self.base)
        if not mp.isfinite(raw_base):
            raise ValueError("prefactor base must be finite")
        poles = _canonicalize_prefactor_poles(self.poles)
        _validate_conjugate_symmetry(poles)
        base = _as_real(raw_base, name="base")
        if not (base > 0):
            raise ValueError("base must be > 0")
        object.__setattr__(self, "constant", constant)
        object.__setattr__(self, "poles", poles)
        object.__setattr__(self, "base", base)

        # Partition poles into real singletons and conjugate-pair quadratic
        # factors so `evaluate` can stay in mp.mpf even when individual poles
        # are off-axis (strip-mode interpolation). For real x and conjugate
        # pole p, (x - p)(x - p_conj) = (x - Re p)^2 + (Im p)^2 is real and
        # positive. The constant is "real-evaluable" when it has imag == 0;
        # then the whole prefactor stays in mp.mpf.
        constant_real = constant.real if constant.imag == 0 else None
        real_poles_mpf = []
        pair_data = []  # list of (re_mpf, im_squared_mpf)
        used = [False] * len(poles)
        tol = _validation_tolerance()
        for i, p in enumerate(poles):
            if used[i]:
                continue
            if p.imag == 0:
                real_poles_mpf.append(p.real)
                used[i] = True
                continue
            if p.imag <= 0:
                # `Im < 0` partners are picked up when their `Im > 0`
                # counterpart processes them; defer.
                continue
            # `Im > 0`: find the unused conjugate partner.
            target = _conjugate_pole(p)
            for j in range(len(poles)):
                if used[j] or j == i:
                    continue
                if poles[j].imag < 0 and abs(poles[j] - target) < tol:
                    used[i] = True
                    used[j] = True
                    pair_data.append(
                        (p.real, p.imag * p.imag)
                    )
                    break
        # `_validate_conjugate_symmetry` already ensures the partition is
        # complete; if anything is still unmarked we cannot run the mpf path.
        if not all(used):
            constant_real = None
            real_poles_mpf = []
            pair_data = []
        object.__setattr__(self, "_constant_real", constant_real)
        object.__setattr__(self, "_real_poles_mpf", tuple(real_poles_mpf))
        object.__setattr__(self, "_pair_data", tuple(pair_data))

    def with_base(self, base, *, constant=None) -> "DampedRational":
        """Return the same pole data with a different exponential base."""
        if constant is None:
            constant = self.constant
        return DampedRational(constant=constant, poles=self.poles, base=base)

    def shift_x(self, shift) -> "DampedRational":
        """Rewrite the same prefactor in the shifted variable x' = x - shift."""
        shift = _as_real(shift, name="shift")
        return DampedRational(
            constant=self.constant * self.base**shift,
            poles=tuple(pole - shift for pole in self.poles),
            base=self.base,
        )

    def evaluate_regulated(self, x, min_pole_distance):
        """Evaluate c * base**x / prod_i d_i, where d_i = (x - pole_i) with
        its magnitude raised to at least `min_pole_distance` (phase preserved).

        When the constant is real the evaluation stays in mp.mpf even with
        conjugate-pair poles: each pair contributes the real quadratic factor
        (x - Re p)^2 + (Im p)^2 = |x - p|^2. The per-linear-factor regulator
        translates to "if |x - p|^2 < min_pole_distance^2, clamp the pair
        contribution to min_pole_distance^2" -- equivalent to raising each
        of (x - p), (x - p_conj) to magnitude min_pole_distance independently
        (their product magnitude is then min_pole_distance^2). For real-x +
        off-axis-pole evaluation this branch is rarely hit (the distance is
        bounded below by |Im p|), but the regulator is preserved exactly for
        consistency with the unpaired case.
        """
        x = _as_real(x, name="x")
        min_pole_distance = _as_real(min_pole_distance, name="min_pole_distance")
        if min_pole_distance <= 0:
            raise ValueError("min_pole_distance must be > 0")
        if self._constant_real is not None:
            min_pole_distance_sq = min_pole_distance * min_pole_distance
            denominator = mp.mpf(1)
            for re_p, im_sq in self._pair_data:
                diff = x - re_p
                quad = diff * diff + im_sq
                if quad < min_pole_distance_sq:
                    quad = min_pole_distance_sq
                denominator *= quad
            for p in self._real_poles_mpf:
                diff = x - p
                mag = abs(diff)
                if mag < min_pole_distance:
                    diff = (
                        diff * (min_pole_distance / mag)
                        if mag > 0
                        else min_pole_distance
                    )
                denominator *= diff
            return self._constant_real * self.base**x / denominator
        denominator = mp.mpc(1)
        for pole in self.poles:
            diff = x - pole
            mag = abs(diff)
            if mag < min_pole_distance:
                diff = (
                    diff * (min_pole_distance / mag)
                    if mag > 0
                    else mp.mpc(min_pole_distance)
                )
            denominator *= diff
        return self.constant * self.base**x / denominator

    def evaluate(self, x):
        """Evaluate c * base**x / prod(x - pole_i).

        When the constant is real, conjugate-pair poles are folded into real
        quadratic factors so the whole evaluation runs in mp.mpf -- including
        the strip-mode interpolation prefactor where every pole is off-axis.
        Only a genuinely complex constant falls back to mp.mpc.
        """
        x = _as_real(x, name="x")
        if self._constant_real is not None:
            denominator = mp.mpf(1)
            for re_p, im_sq in self._pair_data:
                diff = x - re_p
                denominator *= diff * diff + im_sq
            for p in self._real_poles_mpf:
                denominator *= x - p
            return self._constant_real * self.base**x / denominator
        denominator = mp.mpc(1)
        for pole in self.poles:
            denominator *= x - pole
        return self.constant * self.base**x / denominator

    # Former names, still called by polyderivs.
    _evaluate_at_work = evaluate
    _evaluate_regulated_at_work = evaluate_regulated

    def evaluate_arb_batch(self, x_values):
        """Evaluate at multiple real x values using flint arb arithmetic.

        Returns a list of arb values, or ``None`` (the caller then uses the
        mpmath ``evaluate()``) when the poles are not conjugate-paired.
        """
        if self._constant_real is None:
            return None
        from flint import arb, ctx as flint_ctx

        prec_bits = int(mp.mp.dps * 3.33) + 20
        flint_ctx.prec = prec_bits

        const_f = arb(self._constant_real)
        base_f = arb(self.base)
        pair_re = [arb(re_p) for re_p, _ in self._pair_data]
        pair_im_sq = [arb(im_sq) for _, im_sq in self._pair_data]
        real_poles = [arb(p) for p in self._real_poles_mpf]

        results = []
        for x_val in x_values:
            x_f = arb(_as_real(x_val, name="x")) if not isinstance(x_val, arb) else x_val
            denom = arb(1)
            for k in range(len(pair_re)):
                diff = x_f - pair_re[k]
                denom *= diff * diff + pair_im_sq[k]
            for p_f in real_poles:
                denom *= x_f - p_f
            results.append(const_f * base_f ** x_f / denom)
        return results

    @property
    def pole_degree(self) -> int:
        return len(self.poles)

    @property
    def rightmost_real_pole(self):
        """Maximum Re of any pole. Used as the natural shift target so that
        all poles sit at Re <= 0 in the shifted variable."""
        if not self.poles:
            raise ValueError("damped rational has no poles")
        return max(mp.re(pole) for pole in self.poles)

    def filter_poles(self, predicate: Callable[[object], bool]) -> "DampedRational":
        return DampedRational(
            constant=self.constant,
            poles=tuple(pole for pole in self.poles if predicate(pole)),
            base=self.base,
        )

    @staticmethod
    def _conjugate_symmetric(poles, match_tol):
        """Return `poles` with each non-real pole's multiplicity raised to
        match its conjugate partner's, so the multiset is conjugate-symmetric.

        Real poles pass through unchanged. This handles repeated poles (e.g.
        the ell = 0 double poles, where every Kac pole appears twice) and a
        cut that lands between members of a pair or between duplicates."""
        distinct = []  # [representative value, count]
        for p in poles:
            for entry in distinct:
                if _same_pole_location(entry[0], p, match_tol):
                    entry[1] += 1
                    break
            else:
                distinct.append([p, 1])
        result = []
        done = [False] * len(distinct)
        for i, (value, count) in enumerate(distinct):
            if done[i]:
                continue
            done[i] = True
            if mp.im(value) == 0:
                result.extend([value] * count)
                continue
            partner = _conjugate_pole(value)
            partner_count = 0
            for j in range(len(distinct)):
                if not done[j] and _same_pole_location(distinct[j][0], partner, match_tol):
                    partner = distinct[j][0]
                    partner_count = distinct[j][1]
                    done[j] = True
                    break
            target = max(count, partner_count)
            result.extend([value] * target)
            result.extend([partner] * target)
        return result

    @staticmethod
    def _cluster_ids(points, match_tol):
        """Cluster complex points coincident within ``match_tol``; return one
        int id per point (same id == same location).

        Distinct pole locations are separated by Kac spacings (or are exact
        duplicates), so a float-coarse chaining pass finds the clusters; an
        exact-mpmath confirm runs only inside multi-member coarse clusters, so
        near-degenerate points (closer than the coarse gap but farther than
        ``match_tol``) are still split exactly as an all-mpmath scan would.
        Coordinates outside the finite float range use mpmath for the coarse
        ordering and scan as well.
        """
        n = len(points)
        if n == 0:
            return []
        sides = [_pole_side(point) for point in points]
        probes = [complex(point) for point in points]
        native_range = all(_math.isfinite(point.real) and _math.isfinite(point.imag) for point in probes)
        locations = probes if native_range else points
        order = sorted(
            range(n),
            key=lambda i: (locations[i].real, locations[i].imag),
        )
        coarse_tol = 1e-9
        coarse = []
        cur = [order[0]]
        anchor = locations[order[0]]
        for k in order[1:]:
            zk = locations[k]
            distance = (_math.hypot(zk.real-anchor.real, zk.imag-anchor.imag) if native_range
                        else abs(zk-anchor))
            if distance <= coarse_tol:
                cur.append(k)
            else:
                coarse.append(cur)
                cur = [k]
                anchor = zk
        coarse.append(cur)
        ids = [0] * n
        next_id = 0
        for cluster in coarse:
            if len(cluster) == 1:
                ids[cluster[0]] = next_id
                next_id += 1
                continue
            reps = []  # (representative value, id, half-plane) sub-clusters
            for k in cluster:
                for value, gid, side in reps:
                    if sides[k] == side and abs(points[k] - value) < match_tol:
                        ids[k] = gid
                        break
                else:
                    reps.append((points[k], next_id, sides[k]))
                    ids[k] = next_id
                    next_id += 1
        return ids

    def _take_ranked_poles(self, ranked_poles, npoles: int) -> "DampedRational":
        """Take as many ranked poles as fit within `npoles` after restoring
        conjugate symmetry.

        For complex poles we never split a conjugate pair, so the realized pole
        count can be smaller than the requested cap.

        Tracks the completed length incrementally via cluster ids precomputed
        for every pole and its conjugate, so the greedy fill is O(n) dict
        lookups instead of an O(n²) mpmath tolerance scan of the chosen list
        per candidate (that scan dominated interpolate_delta_tables at ~150
        poles x a per-spin kept-cap sweep).
        """
        npoles = _validate_npoles(npoles)
        chosen = []
        match_tol = _validation_tolerance()

        # Resolve every pole and its conjugate to a location id once; the
        # per-candidate pair counting below then needs no mpmath arithmetic.
        n = len(ranked_poles)
        pts = [_as_complex_input(p) for p in ranked_poles]
        ids = self._cluster_ids(pts + [_conjugate_pole(p) for p in pts], match_tol)
        pole_gid = ids[:n]
        conj_gid = ids[n:]
        counts = {}

        # Track the completed length incrementally.
        # For a real pole: completed_len grows by 1.
        # For a complex pole: if its conjugate is already in chosen (unpaired),
        # the pair is already represented and completed_len grows by 1.
        # If its conjugate is NOT in chosen, completed_len grows by 2.
        completed_len = 0

        for i, pole in enumerate(ranked_poles):
            if mp.im(pole) == 0:
                delta = 1
            else:
                # How many chosen copies sit at this pole's location and at
                # its conjugate's location (tolerance matching via the ids).
                pole_count = counts.get(pole_gid[i], 0)
                conj_count = counts.get(conj_gid[i], 0)
                # The completed set will have max(pole_count+1, conj_count)
                # copies of the pole and same of conjugate.
                # Current completed contribution from this pair:
                #   2 * max(pole_count, conj_count)
                # After adding this pole:
                #   2 * max(pole_count + 1, conj_count)
                old_pair_len = 2 * max(pole_count, conj_count)
                new_pair_len = 2 * max(pole_count + 1, conj_count)
                delta = new_pair_len - old_pair_len

            if completed_len + delta <= npoles:
                chosen.append(pole)
                counts[pole_gid[i]] = counts.get(pole_gid[i], 0) + 1
                completed_len += delta

        completed = self._conjugate_symmetric(chosen, match_tol)
        return DampedRational(
            constant=self.constant, poles=tuple(completed), base=self.base
        )

    def take_rightmost_poles(self, npoles: int) -> "DampedRational":
        """Take up to `npoles` poles with the largest Re part, without ever
        splitting a conjugate pair.

        This rule tends to maximize the optimal-interpolation support width by
        keeping the most inert complex poles. It remains useful as a comparison
        rule and can still be attractive when broad support width is the main
        objective, but it is no longer the production default for the fixed
        bootstrap window used in the identical-scalar diagnostics.
        """
        ranked = sorted(
            self.poles,
            key=lambda p: (-mp.re(p), abs(mp.im(p)), -mp.im(p)),
        )
        return self._take_ranked_poles(
            ranked, npoles
        )

    def take_nearest_poles(self, npoles: int) -> "DampedRational":
        """Take up to `npoles` poles closest to the real interpolation
        interval, i.e. the smallest |pole| in the shifted variable (every
        shifted pole has Re <= 0, so the nearest interval point is x = 0).

        This emphasizes the poles geometrically closest to the physical
        interval. It is a robust geometric comparison rule: it avoids the
        residue selector's even/odd instability in the strip regime while still
        tracking the poles closest to the unitarity threshold.
        """
        ranked = sorted(
            self.poles,
            key=lambda p: (abs(p), -mp.re(p), -mp.im(p)),
        )
        return self._take_ranked_poles(
            ranked, npoles
        )

    def take_strongest_poles(self, npoles: int, scores) -> "DampedRational":
        """Keep up to `npoles` poles with the largest `scores` (a sequence
        aligned with `self.poles`) without splitting a conjugate pair. Used for
        residue-magnitude pole selection, where the score is computed
        externally from data the prefactor itself does not carry.

        This remains a useful comparison rule and can win at hand-tuned pole
        counts, but it is no longer the production default because the strip
        regime can show an even/odd stability gap that `nearest` avoids.
        """
        if len(scores) != len(self.poles):
            raise ValueError("scores must be aligned with poles")
        order = sorted(range(len(self.poles)), key=lambda i: -scores[i])
        ranked = [self.poles[i] for i in order]
        return self._take_ranked_poles(ranked, npoles)

    def take_real_first_poles(self, npoles: int, *, double_first: bool = False) -> "DampedRational":
        """Take poles prioritising real poles over complex conjugate pairs.

        At c=2 the only real poles are the diagonal Kac weights
        Δ_{r,r} = (1−r²)/12, each appearing twice (ℓ=0 double poles).
        Real poles are ranked by |pole| (nearest first); complex conjugate
        pairs follow, also ranked by |pole|.

        `double_first=False`  — breadth-first among reals: one occurrence of
        every real pole location before any location is doubled.  At q=30 this
        gives npoles 1..4 = four distinct single real poles, then npoles 5..8
        = promoting each to a double pole.

        `double_first=True`   — depth-first among reals: both occurrences of
        the nearest real pole before moving to the next location.  npoles=1
        gives a simple pole at Δ_{2,2}, npoles=2 gives a double pole there,
        npoles=3 adds a single pole at Δ_{3,3}, etc.

        Comparing the two modes at the same npoles isolates whether coverage
        breadth (many distinct real locations) or pole-order fidelity (full
        double-pole treatment at each location) matters more for accuracy.
        """
        tol = _validation_tolerance()

        real_poles = [p for p in self.poles if mp.im(p) == 0]
        complex_poles = [p for p in self.poles if mp.im(p) != 0]

        # Group real poles by unique location, sorted by distance from origin.
        groups: list[list] = []
        used = [False] * len(real_poles)
        for i, p in enumerate(real_poles):
            if used[i]:
                continue
            used[i] = True
            group = [p]
            for j in range(i + 1, len(real_poles)):
                if not used[j] and abs(real_poles[j] - p) <= tol:
                    group.append(real_poles[j])
                    used[j] = True
            groups.append(group)
        groups.sort(key=lambda g: abs(g[0]))

        if double_first:
            # Depth-first: all occurrences of nearest group, then next group, …
            ranked_real = [p for g in groups for p in g]
        else:
            # Breadth-first: one occurrence from each group per "round".
            max_mult = max((len(g) for g in groups), default=0)
            ranked_real = []
            for occ_idx in range(max_mult):
                for g in groups:
                    if occ_idx < len(g):
                        ranked_real.append(g[occ_idx])

        ranked_complex = sorted(complex_poles, key=lambda p: (abs(p), -mp.re(p), -mp.im(p)))
        return self._take_ranked_poles(ranked_real + ranked_complex, npoles)


POLE_SELECTIONS = (
    "rightmost", "nearest", "residue", "residue_all", "contribution",
    "contribution_all", "contribution_leading", "contribution_leading_all",
    "contribution_physical", "contribution_bernstein",
    "contribution_equilibrium", "contribution_density",
    "contribution_potential", "contribution_optimal", "contribution_relative",
    "real_first", "real_deep",
)

# NOTE: the production DEFAULTS for these selectors are not declared here. They
# live in virasoro_bootstrap/default_params.json and are read by
# virasoro_bootstrap.defaults, so that changing a default means editing the
# config and nothing else. This tuple is only the set of VALID names, which is a
# block-engine concern.


def pole_selection_score_entries(pole_selection: str) -> str:
    """Which numerator entries the residue scorer maxes over for a selector.

    ``"value"`` (only the (0, 0) undifferentiated block entry per coordinate)
    for ``residue`` / ``contribution`` / ``contribution_leading``; ``"kept"``
    (every kept (m, n) entry) for the ``*_all`` variants. The value-only scoring
    keeps the poles that matter most to the physical block rather than being
    dominated by the highest-derivative residues that over-weight deep poles.
    """
    return (
        "value"
        if pole_selection in (
            "residue", "contribution", "contribution_leading",
            "contribution_physical",
        )
        else "kept"
    )


def pole_selection_is_contribution(pole_selection: str) -> bool:
    """Whether the selector post-divides the (leading-coefficient) residue score
    by the distance to the evaluation point -- the legacy ``contribution_leading``
    / ``contribution_leading_all`` weighting. The full ``contribution`` /
    ``contribution_all`` selectors do NOT use this: they compute the full
    principal-part contribution at the evaluation point directly (distance
    already folded in), so they return ``False``."""
    return pole_selection in ("contribution_leading", "contribution_leading_all")


def pole_selection_is_full_contribution(pole_selection: str) -> bool:
    """Whether the selector ranks poles by their *full principal-part*
    contribution at the evaluation point -- every Laurent coefficient of the
    pole, not just the leading one. This feeds ``contribution``,
    ``contribution_all``, and the measure-aware ``contribution_optimal``;
    the legacy leading-coefficient-only scorer is
    ``contribution_leading`` / ``contribution_leading_all``."""
    return pole_selection in (
        "contribution", "contribution_all", "contribution_physical",
        "contribution_bernstein", "contribution_equilibrium",
        "contribution_density", "contribution_potential", "contribution_optimal",
        "contribution_relative",
    )



def select_kept_poles(shifted, npoles, pole_selection, *, scores=None):
    """Keep `npoles` poles from `shifted` (a DampedRational whose poles are in
    the shifted variable x) by the named rule.

    This is the single pole-selection dispatch shared by the Virasoro
    order-reduction (`polyderivs.interpolate_delta_tables`) and the global
    block builder (`global_blocks.global_scalar_G_interpolated_table`), so the
    selectors behave identically and only need updating in one place. The two
    paths differ only in how they obtain the inputs: the Virasoro path strips
    removable poles and scores poles from its numeric sampler (principal parts;
    the exact symbolic numerator is the fallback),
    while the global path uses the analytic SL(2) genuine-pole rule and scores
    residues numerically -- but the keep-rule itself is common.

    `scores` (aligned with `shifted.poles`) is required for `residue`,
    `residue_all`, `contribution`, `contribution_all`, `contribution_leading`
    and `contribution_leading_all` and ignored otherwise; the caller pre-weights
    it (e.g. divides by the distance to the evaluation point for the
    `contribution_leading` selectors).
    """
    if pole_selection == "rightmost":
        return shifted.take_rightmost_poles(npoles)
    if pole_selection == "nearest":
        return shifted.take_nearest_poles(npoles)
    if pole_selection == "contribution_optimal":
        raise ValueError(
            "contribution_optimal requires a nested greedy order constructed "
            "from the fixed finite reference measure; use "
            "interpolate_delta_table(s) rather than select_kept_poles directly"
        )
    if pole_selection in (
        "residue", "residue_all", "contribution", "contribution_all",
        "contribution_leading", "contribution_leading_all",
        "contribution_physical", "contribution_bernstein",
        "contribution_equilibrium", "contribution_density",
        "contribution_potential", "contribution_relative",
    ):
        if scores is None:
            raise ValueError(
                f"pole_selection={pole_selection!r} requires residue scores"
            )
        return shifted.take_strongest_poles(npoles, scores)
    if pole_selection == "real_first":
        return shifted.take_real_first_poles(npoles, double_first=False)
    if pole_selection == "real_deep":
        return shifted.take_real_first_poles(npoles, double_first=True)
    raise ValueError(
        f"pole_selection must be one of {POLE_SELECTIONS}; got {pole_selection!r}"
    )


@dataclass(frozen=True)
class DeltaInterpolant:
    """Interpolant of a one-variable function of Delta.

    The stored polynomial is in x = Delta - shift and approximates the reduced
    function f(Delta) / prefactor(Delta). Multiplying back by the prefactor
    reconstructs the block-level quantity.
    """

    prefactor: DampedRational
    shift: object
    sample_points: tuple
    reduced_polynomial: Polynomial
    min_pole_distance: object | None = None

    def __post_init__(self):
        # The interpolant can be extracted after the table's build context
        # has ended. Re-converting here would erase stored shift digits.
        object.__setattr__(self, "shift", literal_number(self.shift))
        object.__setattr__(
            self,
            "sample_points",
            tuple(_as_real(point, name="sample_point") for point in self.sample_points),
        )
        if self.min_pole_distance is not None:
            min_pole_distance = _as_real(
                self.min_pole_distance,
                name="min_pole_distance",
            )
            if min_pole_distance <= 0:
                raise ValueError("min_pole_distance must be > 0")
            object.__setattr__(self, "min_pole_distance", min_pole_distance)

    def reduced_value(self, Delta):
        return self.reduced_polynomial.evaluate(_evaluation_number(Delta) - self.shift)

    def prefactor_value(self, Delta):
        return self._prefactor_at(_evaluation_number(Delta) - self.shift)

    def _prefactor_at(self, x):
        if self.min_pole_distance is None:
            return self.prefactor.evaluate(x)
        return self.prefactor.evaluate_regulated(x, self.min_pole_distance)

    def evaluate(self, Delta):
        x = _evaluation_number(Delta) - self.shift
        return self._prefactor_at(x) * self.reduced_polynomial.evaluate(x)

    def absolute_error(self, evaluator: Callable, Delta):
        return abs(self.evaluate(Delta) - evaluator(Delta))

    def signed_difference(self, evaluator: Callable, Delta, *, tol=None):
        """Return exact(Delta) - approximated(Delta) when it is real-valued."""
        if tol is None:
            tol = _validation_tolerance()
        difference = evaluator(Delta) - self.evaluate(Delta)
        imaginary = mp.im(difference)
        if not mp.isfinite(imaginary) or abs(imaginary) > tol:
            raise ValueError("signed_difference requires a real-valued difference")
        return mp.re(difference)

    def relative_error(self, evaluator: Callable, Delta, *, floor=mp.mpf("1e-30")):
        """Relative error from one reference evaluation; nonfinite data give infinity.

        The denominator floor must be finite and non-negative. With a zero
        floor, a zero reference still has undefined relative error.
        """
        # `floor` is a reporting-policy minimum denominator, not a precision
        # tolerance; it stays at a fixed physical magnitude.
        floor = _as_real(mp.convert(floor), name="floor")
        if not mp.isfinite(floor) or floor < 0:
            raise ValueError("floor must be finite and non-negative")
        reference = mp.convert(evaluator(Delta))
        approximation = self.evaluate(Delta)
        if not mp.isfinite(reference) or not mp.isfinite(approximation):
            return mp.inf
        return abs(approximation - reference) / max(abs(reference), floor)


def support_endpoint_equation(prefactor: DampedRational, support_endpoint, num_sample_points: int):
    """Bohr-Sommerfeld support equation for the endpoint b.

    This is the equation called `bEquation` in the Mathematica reference. For
    complex-conjugate-pair poles the conjugate sum lifts the imaginary part
    out automatically, so we just sum as complex and take the real part.

    Real poles must satisfy Re(pole) <= 0 (otherwise they would sit inside
    the eigenvalue support [0, b] and the matrix-model derivation breaks).
    Complex-conjugate-pair poles are accepted with Re(pole) > 0 as long as
    the pair has nonzero imaginary part: on the real axis the pair
    contributes the strictly-positive smooth factor
    `(x - Re p)^2 + (Im p)^2` (no singularity inside [0, b]), and the sqrt
    branch in the support equation is well-defined for the complex argument.
    Empirically this admits the strip-c (1 < c < 25) bootstrap setup, where
    `shift = Delta_unit` can leave surviving complex poles at `Re(x) > 0`
    with `|Im(x)| >> b`, while the resulting sample points still cover the
    physical positivity interval `[Delta_unit, ...)`.
    """
    if num_sample_points < 1:
        raise ValueError("num_sample_points must be >= 1")
    b = _as_real(support_endpoint, name="support_endpoint")
    if b <= 0:
        raise ValueError("support_endpoint must be > 0")
    _check_support_prefactor(prefactor)

    equation = _support_equation_arb(prefactor, num_sample_points, int(mp.mp.dps * 3.33) + 20)
    value, _slope = equation(_mpf_to_arb(b))
    return _arb_to_mpf(value.mid())


def _check_support_prefactor(prefactor):
    if not (0 < prefactor.base < 1):
        raise ValueError("support equation requires a damping base satisfying 0 < base < 1")
    if any(mp.im(pole) == 0 and mp.re(pole) > 0 for pole in prefactor.poles):
        raise ValueError(
            "support equation requires real poles to satisfy Re(pole) <= 0; "
            "a real pole at Re > 0 would lie inside the eigenvalue support "
            "[0, b] and is a genuine singularity for the matrix-model density"
        )


def _support_equation_arb(prefactor, num_sample_points, prec_bits):
    """The support equation g(b) and its slope g'(b) as a flint closure ``b -> (g, g')``.

    g(b) = sum_p Re(1 - s_p) - b log(base)/2 - N with s_p = sqrt(-p/(b - p)), and
    g'(b) = sum_p Re(s_p/(b - p))/2 - log(base)/2. The poles and log(base) are
    converted once (exactly, at `prec_bits`); the closure evaluates at whatever
    `flint.ctx.prec` is current, so a solver can run it on a precision ladder.
    The results are arb balls (their radii bound the rounding error).
    """
    from flint import arb, acb, ctx as flint_ctx

    flint_ctx.prec = prec_bits
    parts = [(p.real, p.imag) for p in prefactor.poles]
    real_poles = [_mpf_to_arb(re) for re, im in parts if im == 0]
    complex_poles = [acb(_mpf_to_arb(re), _mpf_to_arb(im)) for re, im in parts if im != 0]
    half_log_base = _mpf_to_arb(prefactor.base).log() / 2
    pole_count_minus_n = arb(len(prefactor.poles) - num_sample_points)

    def equation(b):
        root_sum = arb(0)
        slope_sum = arb(0)
        for p in real_poles:
            b_minus_p = b - p
            root = (-p / b_minus_p).sqrt()
            root_sum += root
            slope_sum += root / b_minus_p
        if complex_poles:
            b_c = acb(b)
            c_root_sum = acb(0)
            c_slope_sum = acb(0)
            for p in complex_poles:
                b_minus_p = b_c - p
                root = (-p / b_minus_p).sqrt()
                c_root_sum += root
                c_slope_sum += root / b_minus_p
            root_sum += c_root_sum.real
            slope_sum += c_slope_sum.real
        value = pole_count_minus_n - root_sum - b * half_log_base
        slope = slope_sum / 2 - half_log_base
        return value, slope

    return equation


def _solve_support_endpoint_arb(prefactor, num_sample_points, guess, target_dps):
    """Safeguarded Newton for the support endpoint in flint arb, on a precision ladder.

    Newton on g(b) with its analytic slope, kept inside a bracket [lo, hi] (g(lo)
    < 0 < g(hi); lo starts at 0, hi at +inf until an iterate has g > 0) and
    falling back to bisection or doubling when a step would leave it. The bracket
    only moves on a rigorous sign (g's arb ball excludes 0), so a noisy low-
    precision evaluation can never cut the root out. The working precision starts
    near 100 bits (where the search from the initial guess takes its ~5 steps)
    and roughly doubles each time an iterate is accurate to the current
    precision, ending at the precision the equation was always solved at,
    int(target_dps * 3.33) + 20 bits. Typically one evaluation per rung, a single
    one at full precision, where the secant search it replaces made ~14 (each
    also re-converting every pole through a decimal string): ~10x faster, and
    the root agrees with the secant's to far below the precision b is returned
    at. Returns the root as an mpf (its arb midpoint), or None (not converged)
    so the caller falls back to the mpmath search.
    """
    from flint import arb, ctx as flint_ctx
    final_prec = int(target_dps * 3.33) + 20
    equation = _support_equation_arb(prefactor, num_sample_points, final_prec)
    ladder = [final_prec]
    while ladder[-1] > 128:
        ladder.append(ladder[-1] // 2 + 24)
    ladder.reverse()

    level = 0
    x = _mpf_to_arb(guess)
    lo = arb(0)
    hi = None
    root = None
    for _ in range(iteration_cap()):
        prec = ladder[level]
        flint_ctx.prec = prec
        value, slope = equation(x)
        if not (value.is_finite() and slope.is_finite()):
            break
        if value > 0:
            hi = x
        elif value < 0:
            lo = x
        step = (value.mid() / slope.mid()).mid() if slope.mid() > 0 else None
        x_new = None if step is None else (x - step).mid()
        if x_new is None or not (x_new > lo and (hi is None or x_new < hi)):
            # Newton would leave the bracket: bisect it, or double while it is open.
            x_new = (x * 2).mid() if hi is None else ((lo + hi) / 2).mid()
        step = abs(x - x_new)
        x = x_new
        scale = abs(x)
        if level + 1 < len(ladder):
            # quadratic convergence: x is now good to ~prec bits; climb the ladder.
            # At the top, stop on a roundoff-level step, or when the step's square
            # (the error left after it) is already below the precision.
            if (step * step) < scale * scale * arb(2) ** (-(prec + 16)):
                level += 1
        elif (step < scale * arb(2) ** (-(prec - 24))
              or step * step < scale * scale * arb(2) ** (-(prec + 32))):
            root = x
            break
    flint_ctx.prec = final_prec
    if root is None or not (root > 0):
        return None
    return _arb_to_mpf(root)


@_validate_bs_controls
def solve_support_endpoint(
    prefactor: DampedRational,
    num_sample_points: int,
    *,
    initial_guess=None,
    work_dps: int | None = None,
):
    """Solve the support equation for the interpolation density endpoint b."""
    if num_sample_points < 1:
        raise ValueError("num_sample_points must be >= 1")
    if initial_guess is None:
        initial_guess = -(2 * num_sample_points) / mp.log(prefactor.base)
    guess = _as_real(initial_guess, name="initial_guess")
    if guess <= 0:
        raise ValueError("initial_guess must be > 0")

    target_dps = work_dps if work_dps is not None else mp.mp.dps
    with mp.workdps(target_dps):
        _check_support_prefactor(prefactor)
        # flint Newton first; the mpmath secant/bisection search below is the
        # fallback for a non-converging Newton.
        b = _solve_support_endpoint_arb(prefactor, num_sample_points, guess, target_dps)
        equation = lambda x: support_endpoint_equation(
            prefactor, x, num_sample_points
        )

        if b is None:
            try:
                b = mp.findroot(equation, guess)
            except ValueError:
                b = None

        if b is None:
            tol = _validation_tolerance()
            lower = mp.mpf(guess)
            f_lower = equation(lower)
            if f_lower == 0:
                b = lower
            else:
                if f_lower > 0:
                    upper = lower
                    for _ in range(64):
                        lower /= 2
                        f_lower = equation(lower)
                        if f_lower <= 0:
                            break
                    else:
                        raise ValueError("could not bracket support endpoint below initial guess")
                    f_upper = equation(upper)
                else:
                    upper = lower
                    f_upper = f_lower
                    for _ in range(64):
                        upper *= 2
                        f_upper = equation(upper)
                        if f_upper >= 0:
                            break
                    else:
                        raise ValueError("could not bracket support endpoint above initial guess")

                for _ in range(4 * target_dps):
                    midpoint = (lower + upper) / 2
                    f_mid = equation(midpoint)
                    if (
                        abs(f_mid) <= tol
                        or abs(upper - lower) <= tol * max(abs(midpoint), mp.mpf(1))
                    ):
                        b = midpoint
                        break
                    if f_mid > 0:
                        upper = midpoint
                    else:
                        lower = midpoint
                else:
                    b = (lower + upper) / 2
    b = _as_real(b, name="support_endpoint")
    if b <= 0:
        raise ValueError("support_endpoint solve returned a non-positive value")
    return b


def integrated_density(prefactor: DampedRational, support_endpoint, x):
    """Integrated interpolation density ω(x).

    The physical support is x in [0, b], but the root finder may probe nearby
    real values outside that interval. We therefore evaluate the same analytic
    expression without clipping or rejecting such probes.

    For complex-conjugate-pair poles the conjugate sum lifts the imaginary part
    out automatically, so we just sum as complex and take the real part.
    """
    b = _as_real(support_endpoint, name="support_endpoint")
    x = _as_real(x, name="x")
    if b <= 0:
        raise ValueError("support_endpoint must be > 0")

    total = mp.mpc(0)
    # Compute acos(1 - 2x/b) once — used by every pole's term_2 and by airy_term
    acos_common = mp.acos(1 - (2 * x) / b)
    for pole in prefactor.poles:
        if pole == 0:
            total += 1
            continue
        term_1 = mp.acos(1 - (2 * x * (b - pole)) / (b * (x - pole)))
        term_2 = mp.sqrt(-pole / (b - pole)) * acos_common
        total += (term_1 - term_2) / mp.pi
    airy_term = mp.sqrt((b - x) * x) + b * acos_common / 2
    return mp.re(total) - mp.log(prefactor.base) * airy_term / mp.pi


class _FlintIntegratedDensity:
    """The integrated density omega(x) and the density rho(x) = omega'(x) in flint arb.

    Same function as `integrated_density`, in a form with no complex
    transcendental. For a pole p (real p < 0, or one of a conjugate pair) put
    K_p = sqrt((b - p)/(-p)) (principal, so Re K_p > 0) and u = (b - x)/x. Then
    on 0 < x < b

        Re acos(1 - 2x(b - p)/(b(x - p))) = atan2(2 Re(K_p) sqrt(u), u - |K_p|^2),

    as both sides are continuous there (the acos argument is never real outside
    [-1, 1]; the atan2 point stays in the upper half plane), vanish as x -> 0,
    and equal 2 Re atan(K_p / sqrt(u)) near it. A conjugate pair contributes
    twice its representative's term, and acos(1 - 2x/b) = atan2(2 sqrt(u), u - 1).
    So a pole costs one real atan2 on two precomputed constants, where the acos
    form needed an acb acos of a complex argument (~3x the cost). Differentiating
    gives the density in a cancellation-free form, one division per pole:

        rho(x) = (b - x) / (pi s) * [sum_p w_p Re(sigma_p/(x - p)) - log(base)],

    s = sqrt(x (b - x)), sigma_p = sqrt(-p/(b - p)) = 1/K_p, w_p = 1 (real pole)
    or 2 (pair); a pair's term is taken in real arithmetic. Poles exactly at 0
    are the delta weights counted in `zero_count`.

    `omega` and `density` evaluate at the current `flint.ctx.prec` (the root
    finder runs them on a precision ladder); the constants are computed once,
    with guard bits beyond the working precision of construction. Both take an
    arb (for `density` a ball gives a certified enclosure over it) or an mpf,
    and return arb balls.
    """

    def __init__(self, prefactor: DampedRational, support_endpoint):
        from flint import acb, arb, ctx as flint_ctx

        self._arb = arb
        # 128 guard bits: the constants' rounding stays far below any precision
        # the root finder evaluates at, its raised top rung included
        flint_ctx.prec = int(mp.mp.dps * 3.33) + 20 + 128
        b = _mpf_to_arb(_as_real(support_endpoint, name="support_endpoint"))
        b_c = acb(b)
        self.b = b
        self.zero_count = arb(sum(1 for pole in prefactor.poles if pole == 0))
        self.pi = arb.pi()
        self.log_base = _mpf_to_arb(prefactor.base).log()
        self.log_base_over_pi = self.log_base / self.pi

        # omega terms (2 Re K_p, |K_p|^2) and density terms, for the real poles
        # (sigma_p, p) and for one representative per pair (Re sigma, Re p,
        # Im sigma Im p, (Im p)^2). The coefficient of acos(1 - 2x/b) sums
        # Re sigma over every pole, each member of a pair from its own value.
        self.real_terms, self.real_density = [], []
        self.pair_terms, self.pair_density = [], []
        self.sum_sigma = arb(0)
        parts = [(pole.real, pole.imag) for pole in prefactor.poles]
        paired = [False] * len(parts)
        tol = _validation_tolerance()
        for i, (re, im) in enumerate(parts):
            if paired[i] or (re == 0 and im == 0):
                continue
            paired[i] = True
            if im == 0:
                p = _mpf_to_arb(re)
                k_sq = (b - p) / -p
                sigma = (-p / (b - p)).sqrt()
                self.sum_sigma += sigma
                self.real_terms.append((2 * k_sq.sqrt(), k_sq))
                self.real_density.append((sigma, p))
                continue
            for j in range(i + 1, len(parts)):  # the partner: usually the next pole
                if not paired[j] and parts[j][1] * im < 0 and abs(parts[j][0] - re) < tol and abs(
                        parts[j][1] + im) < tol:
                    paired[j] = True
                    break
            else:
                raise ValueError(f"complex pole {prefactor.poles[i]} has no conjugate partner")
            p = acb(_mpf_to_arb(re), _mpf_to_arb(im))
            p_partner = acb(_mpf_to_arb(parts[j][0]), _mpf_to_arb(parts[j][1]))
            k = ((b_c - p) / -p).sqrt()
            sigma = (-p / (b_c - p)).sqrt()
            self.sum_sigma += sigma.real + (-p_partner / (b_c - p_partner)).sqrt().real
            self.pair_terms.append((2 * k.real, k.real * k.real + k.imag * k.imag))
            self.pair_density.append((sigma.real, p.real, sigma.imag * p.imag, p.imag * p.imag))

    def omega(self, x):
        arb = self._arb
        x = x if isinstance(x, arb) else _mpf_to_arb(x)
        u = (self.b - x) / x
        v = u.sqrt()
        atan2 = arb.atan2
        total = arb(0)
        for two_k, k_sq in self.real_terms:
            total += atan2(two_k * v, u - k_sq)
        if self.pair_terms:
            pairs = arb(0)
            for two_re_k, abs_k_sq in self.pair_terms:
                pairs += atan2(two_re_k * v, u - abs_k_sq)
            total += 2 * pairs
        theta = atan2(2 * v, u - 1)  # acos(1 - 2x/b)
        total -= self.sum_sigma * theta
        # the airy part: sqrt(x (b - x)) + (b/2) acos(1 - 2x/b)
        return total / self.pi + self.zero_count - self.log_base_over_pi * (x * v + self.b * theta / 2)

    __call__ = omega

    def density(self, x):
        arb = self._arb
        x = x if isinstance(x, arb) else _mpf_to_arb(x)
        total = arb(0)
        for sigma, p in self.real_density:
            total += sigma / (x - p)
        if self.pair_density:
            pairs = arb(0)
            for sigma_re, p_re, sigma_im_p_im, p_im_sq in self.pair_density:
                d = x - p_re
                pairs += (sigma_re * d - sigma_im_p_im) / (d * d + p_im_sq)
            total += 2 * pairs
        b_minus_x = self.b - x
        return b_minus_x * (total - self.log_base) / (self.pi * (x * b_minus_x).sqrt())

    def float_seeds(self, n_start, n_stop):
        """Double-precision nodes for targets n + 1/2 (n_start <= n < n_stop), with
        the density and |rho'/(2 rho)| there.

        The same formulas in numpy, vectorized over nodes x poles and solved by a
        bracketed Newton on all nodes at once: ~50-bit starts for the arb
        iteration, its first derivative, and the Newton constant that decides
        when it moves up the precision ladder. Not certified, and need not be:
        an entry that is not finite or not in its bracket only makes a worse
        start (the arb iteration brackets and certifies every node itself)."""
        import numpy as np

        def floats(values):
            return np.array([float(v) for v in values], dtype=float)

        two_k = floats([t[0] for t in self.real_terms + self.pair_terms])
        k_sq = floats([t[1] for t in self.real_terms + self.pair_terms])
        weight = np.array([1.0] * len(self.real_terms) + [2.0] * len(self.pair_terms))
        sigma_r, pole_r = (floats(t[i] for t in self.real_density) for i in (0, 1))
        sigma_re, p_re, c1, c2 = (floats(t[i] for t in self.pair_density) for i in range(4))
        b, zero, sum_sigma, log_base = (float(v) for v in (self.b, self.zero_count, self.sum_sigma, self.log_base))
        pi = np.pi

        def omega(x):
            u = (b - x) / x
            v = np.sqrt(u)
            terms = np.arctan2(two_k * v[:, None], u[:, None] - k_sq) @ weight
            theta = np.arctan2(2 * v, u - 1)
            return (terms - sum_sigma * theta) / pi + zero - log_base / pi * (x * v + b * theta / 2)

        def bracket_sums(x):
            """sum_p w_p Re(sigma_p/(x - p)) - log(base), and its x-derivative."""
            dr = x[:, None] - pole_r
            dc = x[:, None] - p_re
            q = dc * dc + c2
            num = sigma_re * dc - c1
            t = (sigma_r / dr).sum(axis=1) + 2 * (num / q).sum(axis=1) - log_base
            t_prime = -(sigma_r / (dr * dr)).sum(axis=1) + 2 * ((sigma_re * q - 2 * dc * num) / (q * q)).sum(axis=1)
            return t, t_prime

        def density(x):
            t, _ = bracket_sums(x)
            return (b - x) * t / (pi * np.sqrt(x * (b - x)))

        target = np.arange(n_start, n_stop, dtype=float) + 0.5
        with np.errstate(all="ignore"):
            lo = np.zeros_like(target)
            hi = np.full_like(target, b)
            top = float(omega(np.array([b * (1 - 1e-15)]))[0])
            # arcsine-law start between omega(0) = zero and omega(b)
            x = b * (1 - np.cos(pi * np.clip((target - zero) / (top - zero), 0, 1))) / 2
            x = np.where((x > lo) & (x < hi), x, (lo + hi) / 2)
            for _ in range(200):
                f = omega(x) - target
                lo = np.where(f < 0, x, lo)
                hi = np.where(f > 0, x, hi)
                x_new = x - f / density(x)
                x_new = np.where((x_new > lo) & (x_new < hi), x_new, (lo + hi) / 2)
                done = np.all(np.abs(x_new - x) <= 1e-15 * x)
                x = x_new
                if done:
                    break
            t, t_prime = bracket_sums(x)
            bx = b - x
            rho = bx * t / (pi * np.sqrt(x * bx))
            # d/dx log rho = -1/(b - x) - (b - 2x)/(2 x (b - x)) + t'/t
            curvature = np.abs(-1 / bx - (bx - x) / (2 * x * bx) + t_prime / t) / 2
        return x.tolist(), rho.tolist(), curvature.tolist()


def _make_flint_integrated_density(prefactor: DampedRational, support_endpoint):
    """omega(x) (and its density) in flint arb: a `_FlintIntegratedDensity`."""
    return _FlintIntegratedDensity(prefactor, support_endpoint)


@_validate_bs_controls
def find_bohr_sommerfeld_roots(
    integrated_density_fn: Callable,
    n_start: int,
    num_sample_points: int,
    *,
    upper_bound,
    initial_guess,
    work_dps: int | None = None,
):
    """Solve ω(x_n) = n + 1/2 sequentially for n = n_start, ..., N - 1.

    `integrated_density_fn` is the flint density built by
    `_make_flint_integrated_density`; the roots come from the bracketed Newton
    of `_find_bs_roots_arb`, run on a precision ladder and certifying every
    node at full precision.
    """
    if n_start < 0:
        raise ValueError("n_start must be >= 0")
    if num_sample_points < n_start:
        raise ValueError("num_sample_points must be >= n_start")

    upper_bound = _as_real(upper_bound, name="upper_bound")
    if upper_bound <= 0:
        raise ValueError("upper_bound must be > 0")
    _as_real(initial_guess, name="initial_guess")
    target_dps = work_dps if work_dps is not None else mp.mp.dps

    if not isinstance(integrated_density_fn, _FlintIntegratedDensity):
        raise TypeError(
            "integrated_density_fn must be the flint density from "
            "_make_flint_integrated_density (it supplies the density and "
            "double-precision seeds the root finder needs)"
        )
    roots = _find_bs_roots_arb(
        integrated_density_fn, n_start, num_sample_points,
        upper_bound, target_dps,
    )
    return [_arb_to_mpf(r) for r in roots]


# Bits of an evaluation at precision p that the ladder counts as rounding
# noise (the omega sum loses a few to its ~#poles terms and to the root's
# conditioning), and how far below full precision a node is certified.
_BS_NOISE_BITS = 16
_BS_CERTIFY_BITS = 12


def _bs_precision_ladder(final_prec):
    """Precisions for the node iteration, ending at `final_prec`.

    A Newton step from an iterate good to (p - 16) bits, on the rung below,
    gives 2(p - 16) - log2(Newton constant) bits, so rung p is followed by
    2(p - 12): each rung is about twice the one below. The bottom rung (~100
    to 128 bits) is one step above the ~50-bit double-precision starts. For
    686 bits (200 digits): 106, 189, 355, 686; for 353: 106, 188, 353."""
    ladder = [final_prec]
    while ladder[-1] > 128:
        ladder.append(ladder[-1] // 2 + 12)
    return ladder[::-1]


def _find_bs_roots_arb(density, n_start, num_sample_points, upper_bound, target_dps):
    """Bracketed Newton for the Bohr-Sommerfeld nodes, on a precision ladder.

    Each node solves f(x) = omega(x) - (n + 1/2) = 0 in [lo, hi], lo the
    previous node (where f = -1) or 0 and hi the support endpoint b. Every
    evaluation is an arb ball and moves the bracket only on a certified sign; a
    Newton step (derivative: the analytic density) is taken only when it lands
    strictly inside the bracket and at least halves the step (`rtsafe`),
    otherwise the step bisects. So the iteration cannot leave the bracket or
    stall, whatever its start.

    The starts are the double-precision nodes of `float_seeds` (all nodes at
    once, vectorized), with their density and Newton constant |rho'/(2 rho)|.
    The iteration climbs a precision ladder (`_bs_precision_ladder`): it stays
    on a rung until the error left by its last step (Newton constant x step^2)
    is below that rung's rounding noise, or the ball of f contains 0 (the node
    is resolved at this precision), then moves up. The derivative only has to
    carry the bits a step gains, so it is re-evaluated only when the one in hand
    has fewer, at about half the rung's precision (the double-precision one
    serves the two lowest rungs). Typically one omega per rung, and two
    densities per node.

    At full precision (int(target_dps * 3.33) + 20 bits, as before) each node is
    certified by an interval-Newton step (`_certify_bs_node`), whose density
    evaluation also supplies the derivative there: the node is the midpoint of an
    enclosure of the root no wider than 2^-(full - 12) relative. When rounding
    alone keeps the enclosure wider (a node very close to 0, next to a near-zero
    pole), the top rung is raised by the bits it lacks and the same iterate is
    evaluated again. A node that is not certified keeps iterating; one still
    not certified at the iteration cap raises, so no uncertified node is ever
    returned. Cost: one full-precision omega per node, against the ~6 of the
    full-precision safeguarded Newton this replaced (with omega's acb acos).
    """
    from flint import arb, ctx as flint_ctx

    final_prec = int(target_dps * 3.33) + 20
    ladder = _bs_precision_ladder(final_prec)
    top = len(ladder) - 1
    flint_ctx.prec = final_prec
    # room for pure bisection from [0, b] down to the certificate, and more
    max_iter = 2 * final_prec + 200
    half = arb(1) / 2
    zero = arb(0)
    upper_val = _mpf_to_arb(upper_bound)
    noise = [arb(2) ** -(p - _BS_NOISE_BITS) for p in ladder]
    certify_tol = arb(2) ** -(final_prec - _BS_CERTIFY_BITS)
    dens_prec = [min(p, max(p // 2 + 2 * _BS_NOISE_BITS, 128)) for p in ladder]
    # bits a step gains on each rung: the relative accuracy the derivative needs
    gain = [0] + [p - q + 8 for q, p in zip(ladder, ladder[1:])]
    seeds, seed_rhos, seed_curvatures = density.float_seeds(n_start, num_sample_points)

    roots = []
    for n, seed, seed_rho, seed_curvature in zip(
            range(n_start, num_sample_points), seeds, seed_rhos, seed_curvatures):
        flint_ctx.prec = final_prec
        target = arb(n) + half
        lo = roots[-1] if roots else zero
        hi = upper_val
        x = arb(seed) if _math.isfinite(seed) else None
        if x is None or not (lo < x < hi):
            x = ((lo + hi) / 2).mid()
            seed_rho = _math.nan
        # the derivative in use and its correct bits (the double-precision one
        # has ~50: the density formula has no cancellation)
        rho_m, rho_bits = (arb(seed_rho), 48) if _math.isfinite(seed_rho) and seed_rho > 0 else (None, 0)
        # 4 |rho'/(2 rho)|: the error after a Newton step is < curvature * step^2
        # (not certified: it only decides when to move up the ladder)
        curvature = arb(4 * seed_curvature) if _math.isfinite(seed_curvature) else arb(2) ** 64
        level = 0
        boost = 0  # extra bits on the top rung for a node whose omega loses them to rounding
        step_old = (hi - lo).mid()
        root = None
        for _ in range(max_iter):
            prec = ladder[level] + (boost if level == top else 0)
            flint_ctx.prec = prec
            f = density.omega(x) - target
            if f > 0:
                hi = x
            elif f < 0:
                lo = x
            scale = x  # tolerances are relative: 0 < lo < x (a node can sit very close to 0)
            finite = f.is_finite()
            if not finite and level < top:
                # A coarse iterate may round outside a very narrow support
                # interval. Refine it at the next rung before bisecting again.
                level += 1
                continue
            if level == top and finite:
                root, rho_top, extra = _certify_bs_node(density, x, f, rho_m, scale, certify_tol, dens_prec[top])
                if root is not None:
                    flint_ctx.prec = final_prec
                    root = (+root).mid()  # nodes carry the working precision
                    break
                rho_m = rho_top
                if extra:
                    # resolved as far as this precision allows: evaluate the
                    # same iterate again with the bits the rounding cost
                    boost += extra
                    continue
            elif rho_m is None or rho_bits < gain[level]:
                flint_ctx.prec = dens_prec[level]
                rho = density.density(x)
                flint_ctx.prec = prec
                rho_m, rho_bits = (rho.mid(), dens_prec[level] - _BS_NOISE_BITS) if rho.is_finite() else (None, 0)
            if finite and level < top and not (f > 0 or f < 0):
                level += 1  # the ball of f contains 0: resolved at this precision
                continue
            # rtsafe: a Newton step when it lands strictly inside the bracket
            # and at least halves the step, else bisection
            fm = f.mid()
            newton = finite and rho_m is not None and rho_m > 0
            if newton:
                x_new = (x - fm / rho_m).mid()
                newton = lo < x_new < hi and abs(2 * fm) <= abs(step_old * rho_m)
            if not newton:
                x_new = ((lo + hi) / 2).mid()
            step = abs(x_new - x)
            if level < top and (step.is_zero() or not (lo < x_new < hi)):
                # Bisection can exhaust this rung's representable points
                # before Newton is available near a very narrow peak.
                level += 1
                flint_ctx.prec = ladder[level]
                x = ((lo + hi) / 2).mid()
                step_old = (hi - lo).mid()
                continue
            x = x_new
            step_old = step
            if not newton:
                continue
            # Newton's error after the step is ~ |rho'/(2 rho)| step^2 (with a
            # safety factor 4); move up once it is below this rung's noise
            if level < top and min(step, curvature * step * step) <= noise[level] * scale:
                level += 1
        if root is None:
            raise RuntimeError(
                f"Bohr-Sommerfeld node {n} was not certified within {max_iter} iterations "
                f"(bracket [{lo.mid()}, {hi.mid()}])"
            )
        roots.append(root)

    flint_ctx.prec = final_prec
    return roots


def _certify_bs_node(density, x, f, rho_m, scale, certify_tol, dens_prec):
    """Interval-Newton certificate of a node.

    X = [x - r, x + r], r twice the Newton step plus the rounding noise of f
    (`rho_m`, the last derivative, sizes the step; any positive estimate
    works). If rho > 0 on X and N = x - f/rho(X) lies inside X, N encloses the
    unique root in X. Returns (node, rho(X) midpoint, extra bits):
    node = N's midpoint when N is within `certify_tol` (relative to `scale`),
    else None; extra bits > 0 when N is wider only because of the rounding of
    f (the root is resolved as far as this precision allows), giving the
    precision increase that resolves it.
    """
    from flint import arb, ctx as flint_ctx

    prec = flint_ctx.prec
    if rho_m is None or not rho_m > 0:
        flint_ctx.prec = dens_prec
        rho_m = density.density(x).mid()
        flint_ctx.prec = prec
        if not (rho_m.is_finite() and rho_m > 0):
            return None, None, 0
    tol = certify_tol * scale
    floor = f.rad() / rho_m
    ball = arb(x, (2 * abs(f.mid() / rho_m) + 4 * floor + tol).mid())
    flint_ctx.prec = dens_prec
    rho = density.density(ball)
    flint_ctx.prec = prec
    if not (rho.is_finite() and rho > 0):
        return None, None, 0
    enclosure = x - f / rho
    if ball.contains(enclosure):
        width = enclosure.rad()
        if width <= tol:
            return enclosure.mid(), rho.mid(), 0
        if width <= 4 * floor:
            return None, rho.mid(), int(float((width / tol).log() / arb(2).log())) + 8
    return None, rho.mid(), 0


def _default_boundary_offset() -> mp.mpf:
    """Tiny positive lift applied to the lowest sample point.

    With `shift = Delta_unitarity` (the bootstrap-natural convention), the
    matrix-model places sample points starting at x ~ 0, which corresponds
    physically to Delta = Delta_unitarity. The block typically has a pole
    there (identity at Delta = 0; conserved currents at Delta = ell), so we
    lift the lowest sample point off the boundary by a tiny epsilon to keep
    `exact.evaluate(Delta_unitarity + epsilon)` numerically safe. The default
    scales with the current precision: `10^{-dps/4}` keeps the lifted node far
    enough off singular-c threshold poles to stabilize the regularized numeric
    sampler, while still leaving ~`10^{3*dps/4}` digits between the lift and
    the polynomial-evaluation noise floor.
    """
    return mp.mpf(10) ** -(mp.mp.dps // 4)


@_validate_bs_controls
def get_sample_points(
    prefactor: DampedRational,
    num_sample_points: int,
    *,
    small_pole_threshold=mp.mpf("1e-10"),
    initial_guess=mp.mpf("0.1"),
    work_dps: int | None = None,
    boundary_offset=None,
    method: str = "bohr_sommerfeld",
):
    """Compute the near-optimal interpolation nodes for a damped rational prefactor.

    Parameters
    ----------
    method : str
        Algorithm for computing sample points:
        - ``"bohr_sommerfeld"`` (default): sequential root-finding for the
          Bohr-Sommerfeld quantization conditions on the matrix-model equilibrium
          density. Nodes lie in ``[0, b]`` where b is the support endpoint.
        - ``"jacobi"``: Gauss quadrature nodes from the Jacobi-matrix eigenvalue
          algorithm (pmp2sdp-style). Computes moments of the damped-rational
          measure on ``[0, b]``, extracts orthogonal polynomial recurrence
          coefficients via the Stieltjes algorithm, then finds all N sample points
          as eigenvalues of the tridiagonal Jacobi matrix. Typically 2-10× faster
          than the sequential root-finder, especially with complex-conjugate poles.

    The current implementation follows the Mathematica prototype's treatment of
    poles at 0: a single pole at 0 contributes an explicit sample point x = 0,
    while higher multiplicity at 0 is left unimplemented.

    `boundary_offset` (default `_default_boundary_offset()`) is the minimum
    positive value of x for any returned sample point: if the matrix-model
    lowest node is below this offset, it is lifted to the offset. This is
    relevant when the bootstrap shift convention places the polynomial-
    variable origin at a block-pole location (e.g. identity at the unitarity
    bound), so the matrix-model node accumulating near x = 0 would collide
    with `exact.evaluate(...)`'s pole singularity during interpolation. The
    bootstrap-convention default ensures the lowest sample point sits at a
    safe distance from that pole. Pass `boundary_offset = 0` to disable.
    """
    if method not in ("bohr_sommerfeld", "jacobi"):
        raise ValueError(f"method must be 'bohr_sommerfeld' or 'jacobi', got {method!r}")
    if method == "jacobi":
        return get_sample_points_jacobi(
            prefactor,
            num_sample_points,
            small_pole_threshold=small_pole_threshold,
            work_dps=work_dps,
            boundary_offset=boundary_offset,
        )
    if num_sample_points < 1:
        raise ValueError("num_sample_points must be >= 1")
    threshold = _as_real(small_pole_threshold, name="small_pole_threshold")
    if threshold < 0:
        raise ValueError("small_pole_threshold must be >= 0")
    if not (0 < prefactor.base < 1):
        raise ValueError("sample-point generation requires a damping base satisfying 0 < base < 1")
    if any(mp.im(pole) == 0 and mp.re(pole) > 0 for pole in prefactor.poles):
        raise ValueError(
            "sample-point generation requires real poles to satisfy Re(pole) <= 0; "
            "a real pole at Re > 0 would lie inside the eigenvalue support "
            "[0, b]. Complex-conjugate-pair poles are admitted regardless of "
            "Re(pole) since they contribute a strictly-positive smooth factor "
            "on the real axis (see `support_endpoint_equation` docstring)."
        )

    if boundary_offset is None:
        boundary_offset = _default_boundary_offset()
    boundary_offset = _as_real(boundary_offset, name="boundary_offset")
    if boundary_offset < 0:
        raise ValueError("boundary_offset must be >= 0")

    target_dps = work_dps if work_dps is not None else mp.mp.dps

    # A real pole inside the zero-classification band but *not exactly* at the
    # origin desynchronizes the two consumers of that band. `real_zero_poles`
    # below counts it as a delta at x = 0 (so omega(0+) = real_zero_poles and the
    # Bohr-Sommerfeld finder starts at n_start = real_zero_poles), but the
    # integrated-density builders treat only an exactly-zero pole as that delta --
    # a nonzero pole becomes a regular `acos` term whose argument sits at -1,
    # returning nan and stalling the root finder to its iteration cap on garbage
    # nodes. Such a pole is the roundoff image of an exact Kac zero (e.g. the
    # identity-collision pole d_{3,2} = d_{1,1} = 0 at a minimal model) that
    # `regularize_delta_table` strips algebraically; reaching here means that
    # step was skipped. Fail loudly rather than emit silent nan nodes.
    near_zero = [
        pole for pole in prefactor.poles
        if mp.im(pole) == 0 and 0 < abs(pole) < threshold
    ]
    # A genuine pole this close to the origin, far above the working-precision floor (c = 1 + 1e-12 puts one at
    # x = -2.5e-13; it reaches 0 at c = 1), is placed as the delta at 0 it is for the node density: node placement
    # only needs a nearby measure, and the prefactor itself (scalings, the table's verification) keeps the pole.
    noise = mp.mpf(10) ** (-(mp.mp.dps - 10))
    if near_zero and all(abs(pole) > noise for pole in near_zero):
        prefactor = DampedRational(
            constant=prefactor.constant, base=prefactor.base,
            poles=tuple(mp.mpf(0) if (mp.im(pole) == 0 and 0 < abs(pole) < threshold) else pole
                        for pole in prefactor.poles),
        )
        near_zero = []
    if near_zero:
        closest = min(abs(pole) for pole in near_zero)
        raise ValueError(
            f"{_UNREMOVED_KAC_ZERO_MARKER}: sample-point generation received a "
            f"real pole within {mp.nstr(threshold, 3)} of the origin but not "
            f"exactly at 0 (closest |pole| = {mp.nstr(closest, 3)}). This is an "
            "unremoved roundoff image of an exact Kac zero (e.g. the identity "
            "pole at a c < 1 minimal model): the Bohr-Sommerfeld density would "
            "count it as a delta at x = 0 while the integrated-density evaluator "
            "treats it as a regular pole, producing nan nodes. Build the exact "
            "table at the singular c through merged_delta_table (its merged poles "
            "carry no removable roundoff images) before interpolating."
        )

    # Real poles at x = 0 each contribute a delta-function of weight 1 to the
    # matrix-model density, so `integrated_density` starts at
    # omega(0+) = real_zero_poles. Hence the Bohr-Sommerfeld root finder must
    # start at n_start = real_zero_poles (skipping the indices already
    # accounted for by the delta).
    real_zero_poles = sum(
        1 for pole in prefactor.poles if mp.im(pole) == 0 and abs(pole) < threshold
    )

    # Two regimes for placing sample points near x = 0:
    # - boundary_offset > 0 (the bootstrap-natural default): we don't sample
    #   *at* the boundary even if the kept-prefactor has poles there
    #   (evaluating the damped-rational at its own pole is 1/0). Replace any
    #   multiplicity-k delta contribution with a single lifted sample at
    #   `boundary_offset`. We need (num_sample_points - 1) BS roots; with
    #   omega(0+) = k their indices run k..k + (N - 1) - 1 = k + N - 2 and
    #   the support equation must accommodate omega(b) = k + N - 1 (= N for
    #   k = 1, but N + k - 1 in general). Solve for b with effective node
    #   count N + (k - 1) so the topmost BS target lies inside [0, b].
    # - boundary_offset == 0: strict matrix-model placement. A single pole at
    #   0 places a sample point at x = 0; higher multiplicity is unsupported
    #   because distinct degenerate sample points cannot be expressed.
    if boundary_offset > 0 and real_zero_poles >= 1 and num_sample_points == 1:
        # Degenerate: 1 lifted boundary point covers everything, no BS roots
        # needed. The matrix-model density's deltas at x = 0 alone exhaust the
        # node count, so `solve_support_endpoint` would converge to b = 0+.
        # Short-circuit.
        return [mp.mpf(boundary_offset)]

    effective_n = num_sample_points
    if boundary_offset > 0 and real_zero_poles >= 1:
        effective_n = num_sample_points + (real_zero_poles - 1)
    support_endpoint = solve_support_endpoint(
        prefactor,
        effective_n,
        work_dps=2 * target_dps,
    )
    omega = _make_flint_integrated_density(prefactor, support_endpoint)

    if boundary_offset > 0 and real_zero_poles >= 1:
        # 1 lifted boundary point + (num_sample_points - 1) BS roots
        # = num_sample_points total.
        roots = find_bohr_sommerfeld_roots(
            omega,
            real_zero_poles,
            real_zero_poles + (num_sample_points - 1),
            upper_bound=support_endpoint,
            initial_guess=initial_guess,
            work_dps=target_dps,
        )
        return [mp.mpf(boundary_offset)] + roots
    if real_zero_poles == 0:
        roots = find_bohr_sommerfeld_roots(
            omega,
            0,
            num_sample_points,
            upper_bound=support_endpoint,
            initial_guess=initial_guess,
            work_dps=target_dps,
        )
        # boundary_offset > 0 (no zero poles): lift the lowest matrix-model
        # node if it drifted below the offset.
        if boundary_offset > 0 and roots:
            lowest = mp.re(_as_real(roots[0], name="sample_points[0]"))
            if lowest < boundary_offset:
                roots = [mp.mpf(boundary_offset)] + list(roots[1:])
        return list(roots)
    # boundary_offset == 0 and real_zero_poles >= 1
    if real_zero_poles > 1:
        raise NotImplementedError(
            "higher-order poles at 0 are not yet supported without a "
            "positive boundary_offset; pass boundary_offset > 0 to "
            "collapse the multiplicity into a single lifted sample point"
        )
    small_roots = [mp.mpf(0)]
    roots = find_bohr_sommerfeld_roots(
        omega,
        1,
        num_sample_points,
        upper_bound=support_endpoint,
        initial_guess=initial_guess,
        work_dps=target_dps,
    )
    return small_roots + roots


# ---------------------------------------------------------------------------
# Jacobi-matrix sample-point algorithm (pmp2sdp-style)
# ---------------------------------------------------------------------------


def _pole_free_moments(alpha, num_moments):
    """Compute moments m_k = ∫_0^∞ x^k e^{-αx} dx = k! / α^{k+1}.

    Returns a list [m_0, m_1, ..., m_{num_moments-1}].
    """
    moments = [None] * num_moments
    # m_0 = 1/α
    inv_alpha = mp.mpf(1) / alpha
    moments[0] = inv_alpha
    # m_k = k * m_{k-1} / α  (recurrence from k!/α^{k+1})
    for k in range(1, num_moments):
        moments[k] = k * moments[k - 1] * inv_alpha
    return moments


def _truncated_exponential_integral_term(alpha, pole, b):
    """Compute L_0(p) = ∫_0^b e^{-αx}/(x - p) dx.

    For Re(pole) ≤ 0 (real) or Im(pole) ≠ 0 (complex), the integrand is
    non-singular on [0, b]. Uses the identity:
        L_0 = S_0 - R_0 = ∫_0^∞ - ∫_b^∞ = e^{-αp}[E_1(-αp) - E_1(α(b-p))]

    where S_0 = G_1(p) is the full half-line integral and
    R_0 = ∫_b^∞ e^{-αx}/(x-p) dx = e^{-αp} · E_1(α(b-p)).
    """
    # S_0 = e^{-αp} · E_1(-αp)  [full half-line]
    # R_0 = e^{-αp} · E_1(α(b-p))  [tail beyond b]
    # L_0 = S_0 - R_0
    arg_full = -alpha * pole       # E_1 argument for [0,∞): Re ≥ 0 for Re(pole) ≤ 0
    arg_tail = alpha * (b - pole)  # E_1 argument for [b,∞): Re > 0 always
    return mp.exp(-alpha * pole) * (mp.e1(arg_full) - mp.e1(arg_tail))


def _exponential_integral_term(alpha, pole):
    """Compute G_1(p) = ∫_0^∞ e^{-αx}/(x - p) dx = e^{-αp} · E_1(-αp).

    Requires Re(pole) <= 0 and pole != 0 (the integrand is non-singular on
    [0, ∞) when Re(pole) ≤ 0 and pole ≠ 0).
    """
    z = -alpha * pole  # Re(z) >= 0 for Re(pole) <= 0
    return mp.exp(-alpha * pole) * mp.e1(z)


def _higher_exponential_integral(alpha, pole, max_order):
    """Compute G_j(p) = ∫_0^∞ e^{-αx}/(x - p)^j dx for j = 1, ..., max_order.

    Evaluate G_1 and G_2 directly, then use numerical differentiation and
    the identity G_{j+1}(p) = G_j'(p) / j for higher orders.
    """
    g_values = [None] * max_order
    g_values[0] = _exponential_integral_term(alpha, pole)
    if max_order == 1:
        return g_values
    # G_2(p) = -α·G_1(p) - 1/p
    g_values[1] = -alpha * g_values[0] - mp.mpf(1) / pole
    if max_order == 2:
        return g_values
    # Differentiate with respect to the pole: G_{j+1} = G_j' / j.
    for j in range(2, max_order):
        _j = j
        g_values[j] = mp.diff(
            lambda pp: _higher_exponential_integral(alpha, pp, j)[j - 1],
            pole,
        ) / j
    return g_values


def _partial_fraction_coefficients(poles):
    """Compute partial fraction coefficients for 1/∏(x - p_i).

    For distinct poles: A_i = ∏_{j≠i} 1/(p_i - p_j).
    For repeated poles: returns coefficients for the expansion
    1/∏(x-p_i) = Σ_groups Σ_{j=1}^{mult} B_{group,j} / (x - p_group)^j.

    Returns: list of (pole_value, [(coeff, order)]) where order is 1-indexed.
    """
    tol = _validation_tolerance()
    # Group poles by value
    groups = []  # [(value, multiplicity)]
    remaining = list(poles)
    while remaining:
        p = remaining.pop(0)
        mult = 1
        i = 0
        while i < len(remaining):
            if _same_pole_location(remaining[i], p, tol):
                mult += 1
                remaining.pop(i)
            else:
                i += 1
        groups.append((p, mult))

    # For partial fraction decomposition of 1/∏(x-p_i):
    # If all poles are simple (mult=1 for all), use the standard formula
    # A_i = ∏_{j≠i} 1/(p_i - p_j)
    all_simple = all(m == 1 for _, m in groups)
    if all_simple:
        result = []
        all_poles = [g[0] for g in groups]
        for i, p_i in enumerate(all_poles):
            coeff = mp.mpf(1)
            for j, p_j in enumerate(all_poles):
                if i != j:
                    coeff /= (p_i - p_j)
            result.append((p_i, [(coeff, 1)]))
        return result

    # General case with repeated poles: use residue computation
    # For pole p with multiplicity m, the coefficient of 1/(x-p)^j is:
    # B_j = (1/(m-j)!) * lim_{x->p} d^{m-j}/dx^{m-j} [(x-p)^m / ∏(x-p_i)]
    result = []
    for g_idx, (p_val, mult) in enumerate(groups):
        # Build the "remaining" polynomial: ∏(x-p_i) / (x-p_val)^mult
        # evaluated near x = p_val
        coeffs_for_pole = []
        for j in range(1, mult + 1):
            # Coefficient of 1/(x - p_val)^j in the partial fraction expansion
            # = (1/(mult-j)!) * [d^{mult-j}/dx^{mult-j} of g(x)] at x=p_val
            # where g(x) = (x-p_val)^mult / ∏_all (x-p_i)
            #            = 1 / ∏_{other groups} (x-p_k)^{m_k}
            order_of_deriv = mult - j
            if order_of_deriv == 0:
                # Just evaluate g(p_val) = 1/∏_{k≠g_idx} (p_val - p_k)^{m_k}
                val = mp.mpf(1)
                for k_idx, (p_k, m_k) in enumerate(groups):
                    if k_idx != g_idx:
                        val /= (p_val - p_k) ** m_k
                coeffs_for_pole.append((val, j))
            else:
                # Use mpmath numerical differentiation
                def g_func(x, _g_idx=g_idx, _groups=groups):
                    val = mp.mpc(1)
                    for k_idx, (p_k, m_k) in enumerate(_groups):
                        if k_idx != _g_idx:
                            val /= (x - p_k) ** m_k
                    return val

                deriv_val = mp.diff(g_func, p_val, n=order_of_deriv)
                coeff = deriv_val / mp.factorial(order_of_deriv)
                coeffs_for_pole.append((coeff, j))
        result.append((p_val, coeffs_for_pole))
    return result


def _single_pole_moment_contribution(alpha, pole, order, pole_free_moments, g_values, k):
    """Compute ∫_0^∞ x^k · e^{-αx} / (x - pole)^order dx.

    Uses the identity for order=1:
        ∫_0^∞ x^k e^{-αx}/(x-p) dx = Σ_{j=0}^{k-1} p^j · I_{k-1-j} + p^k · G_1(p)

    For order > 1:
        ∫_0^∞ x^k e^{-αx}/(x-p)^m dx uses the recurrence
        x^k/(x-p)^m = x^{k-1}/(x-p)^{m-1} + p·x^{k-1}/(x-p)^m
    """
    if order == 1:
        # ∫_0^∞ x^k e^{-αx}/(x-p) dx = Σ_{j=0}^{k-1} p^j · I_{k-1-j} + p^k · G_1(p)
        total = mp.mpc(0)
        p_power = mp.mpc(1)  # p^0
        for j in range(k):
            total += p_power * pole_free_moments[k - 1 - j]
            p_power *= pole
        total += p_power * g_values[0]  # p^k · G_1(p)
        return total
    else:
        # For higher order poles, use: x/(x-p)^m = 1/(x-p)^{m-1} + p/(x-p)^m
        # So ∫ x^k e^{-αx}/(x-p)^m dx = ∫ x^{k-1} e^{-αx}/(x-p)^{m-1} dx
        #                                + p · ∫ x^{k-1} e^{-αx}/(x-p)^m dx
        # Recurse on k until k=0:
        # ∫ e^{-αx}/(x-p)^m dx = G_m(p) = g_values[m-1]
        if k == 0:
            return g_values[order - 1]
        return _single_pole_moment_contribution(
            alpha, pole, order - 1, pole_free_moments, g_values, k - 1
        ) + pole * _single_pole_moment_contribution(
            alpha, pole, order, pole_free_moments, g_values, k - 1
        )


def _compute_damped_rational_moments(prefactor: DampedRational, num_moments: int, work_dps: int):
    """Compute moments m_k = ∫_0^∞ x^k · w(x) dx for k = 0, ..., num_moments-1.

    The weight is w(x) = base^x / ∏(x - p_i) on [0, ∞) (ignoring the constant,
    which cancels in the Gauss node computation).

    Poles at x = 0 must be excluded before calling this function (they make the
    integral divergent).

    Uses the recurrence S_k(p) = I_{k-1} + p · S_{k-1}(p) where I_m are the
    pole-free moments, giving O(1) per moment per pole (total O(N·M) instead
    of O(N²·M) for N moments and M poles).
    """
    alpha = -mp.log(prefactor.base)  # base = e^{-α}, α > 0

    # Separate non-zero poles
    nonzero_poles = [p for p in prefactor.poles if p != 0]
    if not nonzero_poles:
        # Pure exponential weight: m_k = k!/α^{k+1}
        return _pole_free_moments(alpha, num_moments)

    # Compute pole-free moments (needed for the partial-fraction expansion)
    pole_free_mom = _pole_free_moments(alpha, num_moments)

    # Get partial fraction decomposition
    pf_data = _partial_fraction_coefficients(tuple(nonzero_poles))

    # For simple poles (the common case), use the fast O(1)-per-moment recurrence:
    # S_k(p) = ∫_0^∞ x^k e^{-αx}/(x-p) dx satisfies S_0 = G_1(p),
    # S_k = I_{k-1} + p · S_{k-1} for k >= 1.
    # Total moment: m_k = Σ_i A_i · S_k(p_i)
    all_simple = all(len(coeff_list) == 1 and coeff_list[0][1] == 1 for _, coeff_list in pf_data)

    if all_simple:
        # Fast path: all poles are simple
        # Precompute G_1(p_i) for each pole
        pole_data = []  # (A_i, p_i, G_1(p_i))
        for pole_val, coeff_list in pf_data:
            A_i = coeff_list[0][0]
            g1 = _exponential_integral_term(alpha, pole_val)
            pole_data.append((A_i, pole_val, g1))

        # Compute moments using recurrence: S_k = I_{k-1} + p · S_{k-1}
        moments = [mp.mpc(0)] * num_moments
        # Initialize S_0 = G_1(p) for each pole
        s_values = [g1 for (_, _, g1) in pole_data]
        # m_0 = Σ A_i · S_0(p_i)
        for i, (A_i, _, _) in enumerate(pole_data):
            moments[0] += A_i * s_values[i]

        for k in range(1, num_moments):
            for i, (A_i, p_i, _) in enumerate(pole_data):
                s_values[i] = pole_free_mom[k - 1] + p_i * s_values[i]
                moments[k] += A_i * s_values[i]

        return moments

    # General case with repeated poles: fall back to the recursive method
    g_cache = {}
    for pole_val, coeff_list in pf_data:
        max_order = max(order for _, order in coeff_list)
        g_cache[pole_val] = _higher_exponential_integral(alpha, pole_val, max_order)

    moments = []
    for k in range(num_moments):
        m_k = mp.mpc(0)
        for pole_val, coeff_list in pf_data:
            g_vals = g_cache[pole_val]
            for coeff, order in coeff_list:
                contribution = _single_pole_moment_contribution(
                    alpha, pole_val, order, pole_free_mom, g_vals, k
                )
                m_k += coeff * contribution
        moments.append(m_k)
    return moments


def _jacobi_coefficients(prefactor: DampedRational, b, n: int, dps: int):
    """``(alphas, betas, digits)``: the Jacobi recurrence coefficients of the measure ``base^x / prod(x - p_i)``
    on ``[0, b]`` computed in Arb balls at ``dps`` digits (the truncated moments by
    the pole recurrences, then the Stieltjes recursion), their
    midpoints, and the fewest correct digits among them (measured from the ball radii)."""
    from flint import acb, arb, ctx

    prev = ctx.prec
    try:
        with mp.workdps(dps):
            ctx.prec = mp.mp.prec
            ulp = arb(2) ** (1 - mp.mp.prec)

            def ball(x):
                # an mpmath result correct to its last bit enters with that bit as its radius
                z = mp.mpc(x)
                v = acb(_mpf_to_arb(z.real), _mpf_to_arb(z.imag))
                r = arb(abs(v).mid()) * ulp
                return v + acb(arb(0, r), arb(0, r))

            alpha = -mp.log(prefactor.base)
            b = _as_real(b, name="support_endpoint")
            A, B = ball(alpha), ball(b)
            E = ball(mp.exp(-alpha * b))
            count = 2 * n
            F = [(1 - E) / A]
            power = acb(1)
            for k in range(1, count):
                power *= B
                F.append((k * F[k - 1] - power * E) / A)
            poles = [p for p in prefactor.poles if p != 0]
            if not poles:
                moments = F
            else:
                moments = [acb(0)] * count
                for pole_val, coeff_list in _partial_fraction_coefficients(tuple(poles)):
                    if max(order for _, order in coeff_list) > 2:
                        raise NotImplementedError("truncated moments for pole multiplicity > 2 not implemented")
                    P = ball(pole_val)
                    l1 = ball(_truncated_exponential_integral_term(alpha, pole_val, b))
                    l2 = -A * l1 - 1 / P - E / (B - P)
                    terms = [(ball(coeff), order) for coeff, order in coeff_list]
                    for k in range(count):
                        if k:
                            l1, l2 = F[k - 1] + P * l1, l1 + P * l2
                        for coeff, order in terms:
                            moments[k] += coeff * (l1 if order == 1 else l2)
            sigma_prev = [acb(0)] * count
            sigma = list(moments)
            alphas = [(sigma[1] / sigma[0]).real]
            betas = [sigma[0].real]
            for k in range(1, n):
                nxt = [acb(0)] * count
                for l in range(k, count - k):
                    nxt[l] = sigma[l + 1] - alphas[k - 1] * sigma[l] - betas[k - 1] * sigma_prev[l]
                sigma_prev, sigma = sigma, nxt
                alphas.append((sigma[k + 1] / sigma[k] - sigma_prev[k] / sigma_prev[k - 1]).real)
                betas.append((sigma[k] / sigma_prev[k - 1]).real)
            digits = float(dps)
            for x in alphas + betas:
                scale = max(abs(x.mid()), arb(b) / n)
                rad = x.rad()
                if not x.is_finite():
                    digits = -float("inf")
                elif rad > 0:
                    digits = min(digits, float(-(arb(rad) / scale).log() / arb(10).log()))
            return [_arb_to_mpf(x.mid()) for x in alphas], [_arb_to_mpf(x.mid()) for x in betas], digits
    finally:
        ctx.prec = prev


def _jacobi_matrix_eigenvalues(alphas, betas):
    """Compute eigenvalues of the symmetric tridiagonal Jacobi matrix.

    The Jacobi matrix has alphas on the diagonal and sqrt(betas[k]) on the
    sub/super-diagonal (for k >= 1). Its eigenvalues are the zeros of the
    N-th orthogonal polynomial = the Gauss quadrature nodes.

    Uses a hybrid approach: numpy for fast double-precision initial estimates,
    then Newton refinement at full mpmath precision. The characteristic
    polynomial p_N(λ) and its derivative are evaluated in O(N) via the
    three-term recurrence, so each Newton step costs O(N) and convergence
    is quadratic (doubling correct digits per step).
    """
    n = len(alphas)
    if n == 1:
        return [alphas[0]]

    import numpy as np

    # Step 1: Get initial estimates via numpy (double precision)
    diag = np.array([float(alphas[i]) for i in range(n)])
    off_diag = np.array([float(mp.sqrt(betas[i + 1])) for i in range(n - 1)])
    # numpy's eigvalsh_tridiagonal is O(N²) and very fast
    try:
        initial_estimates = np.linalg.eigvalsh_tridiagonal(diag, off_diag)
    except AttributeError:
        # Fallback: build full tridiagonal and use eigvalsh
        mat = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
        initial_estimates = np.linalg.eigvalsh(mat)

    # Step 2: Refine each eigenvalue with Newton's method at full precision.
    # The three-term recurrence evaluates the characteristic polynomial:
    #   p_0(λ) = 1
    #   p_1(λ) = λ - α_0
    #   p_k(λ) = (λ - α_{k-1}) · p_{k-1}(λ) - β_{k-1} · p_{k-2}(λ)
    # and its derivative simultaneously.
    tol = mp.eps * 16
    eigenvalues = []

    for est in initial_estimates:
        lam = mp.mpf(float(est))
        for _ in range(100):  # Newton iterations (typically converges in 3-5)
            # Evaluate p_N(lam) and p_N'(lam) via three-term recurrence
            p_prev, p_curr = mp.mpf(1), lam - alphas[0]
            dp_prev, dp_curr = mp.mpf(0), mp.mpf(1)
            for k in range(1, n):
                b_k = betas[k]
                a_k = alphas[k]
                p_next = (lam - a_k) * p_curr - b_k * p_prev
                dp_next = p_curr + (lam - a_k) * dp_curr - b_k * dp_prev
                p_prev, p_curr = p_curr, p_next
                dp_prev, dp_curr = dp_curr, dp_next

            if dp_curr == 0:
                break
            correction = p_curr / dp_curr
            lam -= correction
            if abs(correction) < tol * max(abs(lam), mp.mpf(1)):
                break

        eigenvalues.append(lam)

    return sorted(eigenvalues)


def get_sample_points_jacobi(
    prefactor: DampedRational,
    num_sample_points: int,
    *,
    small_pole_threshold=mp.mpf("1e-10"),
    work_dps: int | None = None,
    boundary_offset=None,
    guard_dps: int | None = None,
):
    """Compute sample points via Jacobi-matrix eigenvalues (pmp2sdp algorithm).

    This is an alternative to the sequential Bohr-Sommerfeld root-finding in
    `get_sample_points`. It computes all N sample points simultaneously as
    eigenvalues of the N×N tridiagonal Jacobi matrix whose entries are the
    recurrence coefficients of the orthogonal polynomials with respect to the
    damped-rational measure on the compact support [0, b].

    The algorithm:
      1. Solve for the support endpoint b (same as Bohr-Sommerfeld)
      2. Compute 2N moments of the measure w(x) = base^x / ∏(x - p_i) on [0, b]
         using the truncated-moment recurrence L_k = F_{k-1} + p·L_{k-1}
      3. Stieltjes recursion to extract (α_k, β_k) coefficients
      4. Eigenvalues of the Jacobi matrix = optimal interpolation nodes on [0, b]

    These Gauss nodes are the zeros of the N-th orthogonal polynomial for
    the original damped-rational measure restricted to [0, b]. They generally
    differ from the Bohr-Sommerfeld nodes, which quantize the associated
    matrix-model equilibrium density.

    Parameters
    ----------
    guard_dps : int, optional
        Extra decimal digits for the moment/Stieltjes computation, honored as
        given.  By default the loss of that ill-conditioned map is measured
        from Arb ball radii at the working precision and the map is run once
        more at the working precision plus the measured loss.
    """
    num_sample_points = _integer_control('num_sample_points', num_sample_points, 1)
    if work_dps is not None:
        work_dps = _integer_control('work_dps', work_dps, 1)
    if guard_dps is not None:
        guard_dps = _integer_control('guard_dps', guard_dps)
    threshold = _as_real(small_pole_threshold, name="small_pole_threshold")
    if threshold < 0:
        raise ValueError("small_pole_threshold must be >= 0")
    if not (0 < prefactor.base < 1):
        raise ValueError("sample-point generation requires a damping base satisfying 0 < base < 1")
    if any(mp.im(pole) == 0 and mp.re(pole) > 0 for pole in prefactor.poles):
        raise ValueError(
            "sample-point generation requires real poles to satisfy Re(pole) <= 0; "
            "a real pole at Re > 0 would lie inside the eigenvalue support."
        )

    if boundary_offset is None:
        boundary_offset = _default_boundary_offset()
    boundary_offset = _as_real(boundary_offset, name="boundary_offset")
    if boundary_offset < 0:
        raise ValueError("boundary_offset must be >= 0")

    target_dps = work_dps if work_dps is not None else mp.mp.dps

    # Handle zero poles: they make the integral divergent, so exclude them
    # and place a fixed sample point at boundary_offset (or 0).
    real_zero_poles = sum(
        1 for pole in prefactor.poles if mp.im(pole) == 0 and abs(pole) < threshold
    )

    # Degenerate case: all nodes are from zero-pole delta contributions
    if boundary_offset > 0 and real_zero_poles >= 1 and num_sample_points == 1:
        return [mp.mpf(boundary_offset)]

    # Build prefactor without zero poles for the Jacobi computation
    nonzero_prefactor = DampedRational(
        constant=prefactor.constant,
        poles=tuple(p for p in prefactor.poles if not (mp.im(p) == 0 and abs(p) < threshold)),
        base=prefactor.base,
    )

    # Number of Gauss nodes to compute (excluding the fixed boundary point)
    if boundary_offset > 0 and real_zero_poles >= 1:
        n_gauss = num_sample_points - 1
    elif boundary_offset == 0 and real_zero_poles >= 1:
        if real_zero_poles > 1:
            raise NotImplementedError(
                "higher-order poles at 0 are not yet supported without a "
                "positive boundary_offset; pass boundary_offset > 0 to "
                "collapse the multiplicity into a single lifted sample point"
            )
        n_gauss = num_sample_points - 1
    else:
        n_gauss = num_sample_points

    if n_gauss == 0:
        if boundary_offset > 0 and real_zero_poles >= 1:
            return [mp.mpf(boundary_offset)]
        return [mp.mpf(0)]

    # Step 1: Solve for the support endpoint b. The effective number of
    # sample points for the support equation accounts for zero-pole delta
    # contributions (same logic as the BS path).
    effective_n = num_sample_points
    if boundary_offset > 0 and real_zero_poles >= 1:
        # The zero-pole contributes a delta; the support equation sees the
        # total number of sample points including that delta contribution.
        effective_n = num_sample_points + (real_zero_poles - 1)
    with mp.workdps(target_dps):
        support_b = solve_support_endpoint(nonzero_prefactor, effective_n, work_dps=target_dps)

    # Step 2: the moments on [0, b] and the Stieltjes recursion lose O(N) digits (the moment -> recurrence-
    # coefficient map is exponentially ill-conditioned).  An explicit guard_dps is honored as given; otherwise
    # the map runs in Arb balls and the loss is read off the radii of (alpha_k, beta_k): at the working precision,
    # or, when the loss exceeds it (the balls do not stay finite), at doubled precision until they do; the
    # coefficients are then built once at the working precision plus the measured loss.
    if guard_dps is not None:
        alphas, betas, digits = _jacobi_coefficients(nonzero_prefactor, support_b, n_gauss, target_dps + guard_dps)
        if not digits > 0:
            raise ValueError(f"Jacobi sample points: guard_dps={guard_dps} leaves no correct digit in the moment "
                             "recursion; omit guard_dps to have the loss measured")
    else:
        measure_dps = target_dps
        while True:
            alphas, betas, digits = _jacobi_coefficients(nonzero_prefactor, support_b, n_gauss, measure_dps)
            if digits > 0:
                break
            if measure_dps > 64 * target_dps + 64 * n_gauss:
                raise ValueError("Jacobi sample points: the moment recursion kept no correct digit")
            measure_dps *= 2
        if digits < target_dps:
            lost = measure_dps - digits
            alphas, betas, _ = _jacobi_coefficients(
                nonzero_prefactor, support_b, n_gauss, target_dps + _math.ceil(lost))
    for k, b_k in enumerate(betas):
        if b_k <= 0:
            raise ValueError(
                f"Stieltjes recursion produced non-positive β_{k} = {float(b_k):.6e}; "
                "the moment sequence may be numerically degenerate.")

    # Eigenvalue step at target precision (well-conditioned for symmetric
    # tridiagonal matrices -- no extra guard needed here).
    with mp.workdps(target_dps):
        alphas_t = [mp.mpf(a) for a in alphas]
        betas_t = [mp.mpf(b) for b in betas]
        gauss_nodes = _jacobi_matrix_eigenvalues(alphas_t, betas_t)

    # Validate nodes are positive
    gauss_nodes = [mp.re(x) for x in gauss_nodes]
    for i, x in enumerate(gauss_nodes):
        if x <= 0:
            raise ValueError(
                f"Jacobi eigenvalue solver produced non-positive node x_{i} = {float(x):.6e}"
            )

    # Assemble final sample points
    if boundary_offset > 0 and real_zero_poles >= 1:
        result = [mp.mpf(boundary_offset)] + gauss_nodes
    elif boundary_offset == 0 and real_zero_poles >= 1:
        result = [mp.mpf(0)] + gauss_nodes
    else:
        result = gauss_nodes
        # Lift lowest node if below boundary_offset
        if boundary_offset > 0 and result:
            if result[0] < boundary_offset:
                result = [mp.mpf(boundary_offset)] + result[1:]

    return result


def ordered_block_delta_value(
    Delta,
    ell: int,
    z,
    zbar,
    h_externals: Sequence,
    hbar_externals: Sequence,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
):
    """Ordered block B written as a function of Delta at fixed signed ell.

    This is the interpolation-ready path for arbitrary external weights.
    The exchanged weights are h = (Delta + ell)/2 and hbar = (Delta - ell)/2.
    """
    ell = _signed_half_integer(ell)
    Delta = literal_number(Delta)
    if isinstance(Delta, complex):
        Delta = mp.mpc(Delta)
    h = _exchange_weight(Delta, ell)
    hbar = _exchange_weight(Delta, -ell)
    return B_block(
        h,
        hbar,
        z,
        zbar,
        h_externals,
        hbar_externals,
        b=b,
        c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
    )


def ordered_block_delta_prefactor(
    ell,
    z,
    zbar,
    h_externals: Sequence,
    hbar_externals: Sequence,
    *,
    shift=0,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
):
    """Build the damped-rational Delta prefactor for the ordered block B.

    For fixed signed `ell`, the ordered block is

        B(Delta) = prefactor(Delta) * P(Delta),

    where `P` is polynomial in Delta after substituting
    h = (Delta + ell) / 2 and hbar = (Delta - ell) / 2 into the symbolic-h
    chiral numerators.
    """
    ell = _signed_half_integer(ell)
    shift = _as_real(literal_number(shift), name="shift")

    left = F_block_chiral_h(z, h_externals, b=b, c=c, q_order=q_order, chiral_norm=chiral_norm)
    right = F_block_chiral_h(zbar, hbar_externals, b=b, c=c, q_order=q_order, chiral_norm=chiral_norm)

    base = mp.sqrt(left.exp_base * right.exp_base)
    degree_left = left.denominator.degree
    degree_right = right.denominator.degree
    constant = (
        (mp.mpf(2) ** (degree_left + degree_right))
        * left.scalar_prefactor
        * right.scalar_prefactor
        * left.exp_base ** (ell / 2)
        * right.exp_base ** (-ell / 2)
        * base**shift
    )
    poles = tuple(2 * root - ell - shift for root in left.denominator_roots) + tuple(
        2 * root + ell - shift for root in right.denominator_roots
    )
    return DampedRational(constant=constant, poles=poles, base=base)


def normalize_delta_prefactor(
    prefactor: DampedRational,
    normalization_base,
    *,
    shift=0,
):
    """Return the prefactor for f(Delta) / normalization_base**Delta.

    `prefactor` is assumed to be written in x = Delta - shift, so dividing by
    normalization_base**Delta rescales the x-exponential base to
    base / normalization_base and the constant by
    normalization_base**(-shift).
    """
    normalization_base = _as_real(normalization_base, name="normalization_base")
    shift = _as_real(shift, name="shift")
    if normalization_base <= 0:
        raise ValueError("normalization_base must be > 0")

    return DampedRational(
        constant=prefactor.constant / normalization_base**shift,
        poles=prefactor.poles,
        base=prefactor.base / normalization_base,
    )


def normalize_delta_evaluator(evaluator: Callable, normalization_base):
    """Return Delta -> evaluator(Delta) / normalization_base**Delta."""
    normalization_base = _as_real(normalization_base, name="normalization_base")
    if normalization_base <= 0:
        raise ValueError("normalization_base must be > 0")

    def normalized_evaluator(Delta):
        Delta = mp.mpc(Delta)
        return evaluator(Delta) / normalization_base**Delta

    return normalized_evaluator


def scalar_block_delta_value(
    Delta,
    ell: int,
    z,
    zbar,
    h_externals: Sequence,
    *,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
):
    """Scalar-external parity eigenstate G written as a function of Delta."""
    return G_block(
        Delta,
        ell,
        z,
        zbar,
        h_externals,
        b=b,
        c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
    )


def scalar_block_delta_prefactor(
    ell: int,
    z,
    zbar,
    h_externals: Sequence,
    *,
    shift=0,
    b=None,
    c=None,
    q_order: int,
    chiral_norm=None,
):
    """Build the damped-rational Delta prefactor for the scalar-external block G.

    For scalar externals, both ordered terms entering G share the same Delta
    prefactor, so the ordered-block construction applies directly.
    """
    return ordered_block_delta_prefactor(
        ell,
        z,
        zbar,
        h_externals,
        h_externals,
        shift=shift,
        b=b,
        c=c,
        q_order=q_order,
        chiral_norm=chiral_norm,
    )


def _preserve_flint_precision(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        from flint import ctx

        before = ctx.prec
        try:
            return function(*args, **kwargs)
        finally:
            ctx.prec = before
    return guarded


def _guard_lagrange_inputs(function):
    """Parse supplied labels; the fit runs at the caller's working precision."""
    from inspect import signature

    sig = signature(function)

    @wraps(function)
    def guarded(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        for name in ('sample_points', 'nodes', 'eval_points', 'values'):
            if name in bound.arguments:
                bound.arguments[name] = tuple(parse_number(v) for v in bound.arguments[name])
        if 'shift' in bound.arguments:
            bound.arguments['shift'] = parse_number(bound.arguments['shift'])
        return function(*bound.args, **bound.kwargs)

    return guarded


@_preserve_flint_precision
@_guard_lagrange_inputs
def lagrange_interpolation_polynomial(sample_points: Sequence, values: Sequence, *, shift=0):
    """Return the Lagrange interpolation polynomial in x = Delta - shift."""
    if len(sample_points) != len(values):
        raise ValueError("sample_points and values must have the same length")
    if len(sample_points) == 0:
        raise ValueError("at least one sample point is required")

    # Build the basis and sum in O(N^2), using flint arb for real data and
    # acb for complex data.
    basis, denominators, arb_basis, arb_denoms = precompute_lagrange_basis(
        sample_points, shift=shift,
    )
    return lagrange_polynomial_from_basis(
        basis, denominators, values,
        _arb_basis=arb_basis, _arb_denoms=arb_denoms,
    )


def _build_product_polynomial_flint(nodes, field):
    """Build P(x) = ∏(x - node_j) using flint-native operations.

    Delegates to `polynomial.build_linear_product_flint` for the actual
    computation.
    """
    from .polynomial import build_linear_product_flint
    return build_linear_product_flint(nodes, field)


def _shifted_sampling_points(sample_x, shift):
    """Keep the real origin and node labels exact until the sampler evaluates them."""
    from .numeric_input import literal_components

    origin = literal_components(shift)[0]
    return [origin + literal_number(x) for x in sample_x]


@_preserve_flint_precision
@_guard_lagrange_inputs
def precompute_lagrange_basis(sample_points: Sequence, *, shift=0):
    """Precompute Lagrange basis polynomials and denominators for fixed nodes.

    Returns `(basis, denominators, arb_basis, arb_denoms)` (the last two the
    flint copies, or None) such that the Lagrange interpolant
    through values `y_i` at `sample_points` is

        interpolant(x) = sum_i (y_i / denominators[i]) * basis[i](x).

    For N entries that share the same `sample_points`, computing the basis
    once and reusing it cuts the per-entry cost from O(N³) to O(N²) -- a
    ~7× speedup at N = 60 / dps = 80, where each Lagrange call is otherwise
    rebuilding the same N basis polynomials. Pair with
    `lagrange_polynomial_from_basis` for the per-entry sum.

    Implementation uses the product-polynomial approach for O(n²) complexity:
    1. Build P(x) = ∏(x - node_j) once: O(n²) via n-1 growing multiplications
    2. For each basis i, compute P(x)/(x - node_i) via synthetic division: O(n)
    Total: O(n²) instead of O(n³) for the naive per-basis construction.
    """
    if len(sample_points) == 0:
        raise ValueError("at least one sample point is required")
    for point in (*sample_points, shift):
        if isinstance(point, (mp.mpf, mp.mpc)) and not mp.isfinite(point):
            raise ValueError("numeric inputs must be finite")

    def _to_field(x):
        if isinstance(x, mp.mpc):
            return mp.mpf(x.real) if x.imag == 0 else x
        if isinstance(x, mp.mpf):
            return x
        if isinstance(x, complex):
            return mp.mpf(x.real) if x.imag == 0 else mp.mpc(x)
        return mp.mpf(mp.convert(x))

    shift = _to_field(shift)
    nodes = [_to_field(point) - shift for point in sample_points]
    # exact comparison: nodes around a near-double pole (a c+eps-split pair) differ by far less than a float ulp
    if len(set(nodes)) != len(nodes):
        raise ValueError("sample_points must be distinct after subtracting shift")

    field = mp.mpf if (isinstance(shift, mp.mpf)
                       and all(isinstance(n, mp.mpf) for n in nodes)) else mp.mpc
    one_const = field(1)

    # Build the product polynomial P(x) = ∏(x - node_j) once: O(n²), staying
    # in flint space for all the multiplications.
    product_poly = _build_product_polynomial_flint(nodes, field)

    # For each basis polynomial, divide P(x) by (x - node_i) via synthetic
    # division: O(n) per division → O(n²) total instead of O(n³).
    # The arb path does the divisions + denominator products for real nodes;
    # complex nodes use Polynomial.quotient_by_linear below.
    basis_result = _precompute_basis_flint(product_poly, nodes, field)
    if basis_result is not None:
        arb_basis, arb_denoms = basis_result
        # Keep the documented public pair usable without the optional FLINT
        # arguments. Lazy wrappers share the ball polynomials without eagerly
        # materializing their O(n²) coefficient arrays on the production path.
        from flint import ctx

        basis = _LagrangeBasis()
        basis.extend(Polynomial._from_arb_trusted(poly, ctx.prec) for poly in arb_basis)
        denominators = [_arb_to_mpf(value) for value in arb_denoms]
    else:
        basis = _LagrangeBasis()
        for i, node_i in enumerate(nodes):
            b_poly = product_poly.quotient_by_linear(node_i)
            basis.append(b_poly)
        arb_basis = None
        arb_denoms = None
        denominators = []
        for i, node_i in enumerate(nodes):
            denom = one_const
            for j, node_j in enumerate(nodes):
                if i != j:
                    denom = denom * (node_i - node_j)
            denominators.append(denom)
    return basis, denominators, arb_basis, arb_denoms


def _precompute_basis_flint(product_poly, nodes, field):
    """Compute all N basis polynomials via O(n) synthetic division in arb.

    Returns (list[arb_poly], list[arb]), or None for complex nodes.
    The arb_poly and arb denominator lists are used for the Lagrange sum.
    """
    if field is not mp.mpf:
        return None
    from flint import arb, arb_poly, ctx as flint_ctx

    prec_bits = int(mp.mp.dps * 3.33) + 20
    flint_ctx.prec = prec_bits

    # Convert product_poly coefficients to arb array once
    prod_coeffs = [arb(c) for c in product_poly.coeffs]
    n_coeffs = len(prod_coeffs)  # degree + 1 of product poly

    # Convert nodes to arb once
    nodes_arb = [arb(node) for node in nodes]

    arb_basis = _LagrangeBasis()
    arb_denoms = []

    for i, node_arb in enumerate(nodes_arb):
        # O(n) synthetic division: P(x) / (x - a) where a = node_arb
        # P has coefficients prod_coeffs[0..n_coeffs-1] (low to high)
        # Result Q has degree n_coeffs-2
        # q[n-2] = prod_coeffs[n-1]
        # q[k] = prod_coeffs[k+1] + a * q[k+1]  for k = n-3, ..., 0
        q_len = n_coeffs - 1
        q_coeffs = [None] * q_len
        q_coeffs[q_len - 1] = prod_coeffs[n_coeffs - 1]
        for k in range(q_len - 2, -1, -1):
            q_coeffs[k] = prod_coeffs[k + 1] + node_arb * q_coeffs[k + 1]

        q_arb = arb_poly(q_coeffs)
        arb_basis.append(q_arb)

        # Compute denominator: product of (node_i - node_j) for j != i
        denom_arb = arb(1)
        for j, nj_arb in enumerate(nodes_arb):
            if j != i:
                denom_arb *= node_arb - nj_arb
        arb_denoms.append(denom_arb)

    return arb_basis, arb_denoms


@_preserve_flint_precision
def lagrange_polynomial_from_basis(basis, denominators, values, *,
                                   _arb_basis=None, _arb_denoms=None):
    """Build the Lagrange interpolant from precomputed basis polynomials.

    `basis` and `denominators` come from `precompute_lagrange_basis`. `values`
    are the function values at the corresponding `sample_points`. The cost
    is one scalar division + one scalar-times-polynomial multiply + one
    polynomial add per node, i.e. O(N²) where the basis poly degree is N − 1
    -- vs O(N³) for a from-scratch `lagrange_interpolation_polynomial` call
    that rebuilds the basis.

    Optional `_arb_basis` and `_arb_denoms` supply pre-built arb_poly and arb
    arrays from precompute_lagrange_basis, avoiding re-conversion in the flint
    fast path. Values may be arb objects directly (skipping mpf round-trips).
    """
    values = tuple(values)
    if len(basis) != len(values) or len(basis) != len(denominators):
        raise ValueError("basis, denominators, values must agree in length")
    from flint import arb, acb
    from .polynomial import _acb_to_mpc

    # Ball grids keep their radii; scalar samples are checked and parsed.
    normalized = []
    for value in values:
        if isinstance(value, (arb, acb)):
            if not value.is_finite():
                raise ValueError("sample values must be finite")
            if not value.is_exact():
                normalized.append(value)
                continue
            value = _arb_to_mpf(value) if isinstance(value, arb) else _acb_to_mpc(value)
        if not isinstance(value, (mp.mpf, mp.mpc)):
            value = parse_number(value)
        if not mp.isfinite(mp.convert(value) if isinstance(value, Fraction) else value):
            raise ValueError("sample values must be finite")
        normalized.append(value)
    return _lagrange_polynomial_from_basis_at_work(basis, denominators, normalized, _arb_basis, _arb_denoms)


def _lagrange_polynomial_from_basis_at_work(basis, denominators, values, _arb_basis, _arb_denoms):
    # Keep Arb balls intact, but normalize Python complex and rational values before
    # choosing real versus complex accumulation.
    from flint import acb
    values = [mp.convert(v) if isinstance(v, (complex, Fraction)) else v for v in values]

    # --- Real nodes: accumulate in flint (arb_poly for real values, acb_poly
    # for complex ones), from the precomputed arb basis when there is one ---
    field = next((b.field for b in basis if b is not None), mp.mpf)
    if _arb_basis is not None or field is mp.mpf:
        all_real = all(
            not isinstance(v, acb) and (not isinstance(v, mp.mpc) or v.imag == 0)
            for v in values
        )
        flint_sum = _lagrange_sum_flint_real if all_real else _lagrange_sum_flint_complex
        return flint_sum(
            basis, denominators, values,
            _arb_basis=_arb_basis, _arb_denoms=_arb_denoms,
        )

    # --- Complex nodes (an mpc basis): Python-level accumulation ---
    interpolant = Polynomial.zero(field=field)
    for i in range(len(basis)):
        v = values[i]
        if isinstance(v, mp.mpc) and v.imag == 0:
            v = mp.mpf(v.real)
        interpolant = interpolant + basis[i] * (v / denominators[i])
    return interpolant


class _LagrangeBasis(list):
    """A precomputed basis owns its precision and any reusable Arb matrix."""

    __slots__ = ("_matrix_cache", "_construction_prec", "__weakref__")

    def __init__(self):
        super().__init__()
        self._matrix_cache = None
        self._construction_prec = mp.mp.prec


def _lagrange_basis_matrix(arb_basis, degree, prec_bits, *, cache=True):
    """Return the ``n x (degree+1)`` arb matrix of basis-polynomial
    coefficients, B[i, k] = coeff k of basis i.

    When ``cache`` is set, a precomputed basis owns one matrix at the selected
    precision and degree. Entries sharing that basis reuse the matrix without
    retaining discarded fits in a global cache. Ordinary lists and the
    non-cached path build a matrix without storing it."""
    from flint import arb, arb_mat

    cache = cache and isinstance(arb_basis, _LagrangeBasis)
    if cache:
        cached = arb_basis._matrix_cache
        if cached is not None and cached[:2] == (prec_bits, degree):
            return cached[2]

    rows = []
    for poly in arb_basis:
        cs = list(poly.coeffs())
        if len(cs) < degree + 1:
            cs = cs + [arb(0)] * (degree + 1 - len(cs))
        rows.append(cs[:degree + 1])
    B = arb_mat(rows)
    if cache:
        arb_basis._matrix_cache = (prec_bits, degree, B)
    return B


def _lagrange_sum_flint_real(basis, denominators, values, *,
                             _arb_basis=None, _arb_denoms=None):
    """Accumulate the Lagrange sum entirely in arb_poly space.

    When _arb_basis and _arb_denoms are provided (from precompute_lagrange_basis),
    skips re-conversion of basis polynomials and denominators.
    When values are already arb objects, skips mpf→arb conversion.

    Returns a Polynomial.
    """
    from flint import arb, arb_poly, arb_mat, ctx as flint_ctx

    prec_bits = int(mp.mp.dps * 3.33) + 20
    flint_ctx.prec = prec_bits

    n = len(basis)
    # Get degree from arb_basis if available, else from mpmath basis
    if _arb_basis is not None and n > 0:
        degree = _arb_basis[0].degree()
    elif n > 0 and basis[0] is not None:
        degree = basis[0].degree
    else:
        degree = 0

    # Use cached arb_poly basis or convert
    if _arb_basis is not None:
        arb_basis = _arb_basis
    else:
        arb_basis = [arb_poly([arb(c) for c in b.coeffs]) for b in basis]

    # Use cached arb denominators or convert
    if _arb_denoms is not None:
        arb_denom_vals = _arb_denoms
    else:
        arb_denom_vals = [arb(d) for d in denominators]

    # interpolant = sum_i (values[i] / denominators[i]) * basis[i]. The basis
    # coefficients are the same across every entry that shares this node set, so
    # the sum is a single matmul ``scalars @ B`` (B[i, k] = coeff k of basis i)
    # instead of n polynomial scalar-multiply-and-adds. B is built once per
    # distinct basis (the Lagrange call is invoked once per fit entry, hundreds
    # of times, all sharing one cached ``_arb_basis``).
    from .polynomial import Polynomial as Poly, _arb_to_mpf

    scalars = []
    for i in range(n):
        v = values[i]
        # Accept arb values directly; convert mpf/mpc otherwise
        if isinstance(v, arb):
            v_arb = v
        elif isinstance(v, mp.mpc):
            v_arb = arb(v.real)
        else:
            v_arb = arb(v)
        scalars.append(v_arb / arb_denom_vals[i])

    if n == 0:
        return Poly([_arb_to_mpf(arb(0))], field=mp.mpf)

    B = _lagrange_basis_matrix(
        arb_basis, degree, prec_bits, cache=(_arb_basis is not None)
    )
    res = arb_mat([scalars]) * B          # 1 x (degree + 1)
    # Convert back via the dyadic arb->mpf (exact midpoint, no decimal-string
    # round-trip): the `.str` conversion dominated the per-entry fit cost at
    # large m_max. Shifts only the sub-working-precision bits.
    coeffs = [_arb_to_mpf(res[0, k]) for k in range(degree + 1)]
    return Poly(coeffs, field=mp.mpf)


@_guard_lagrange_inputs
def lagrange_matrix(nodes, eval_points):
    """``L[k][i] = l_i(eval_points[k])``, the Lagrange basis of ``nodes`` at the points, formed by products.

    ``l_i(x) = l(x) w_i / (x - x_i)`` with ``l(x) = prod_j (x - x_j)`` and ``w_i = 1 / prod_{j != i} (x_i - x_j)``
    (the first barycentric form, backward stable): no basis polynomial is expanded into monomials, whose
    coefficients cancel on evaluation (~0.7 digits per unit of degree).  A point that coincides with a node gets
    that node's unit row.  Returns a flint ``arb_mat`` for real nodes and points, else a nested list of mpmath
    numbers.  Values at the points of the interpolant through ``values`` at the nodes are ``L @ values``.
    """
    nodes = [mp.convert(v) for v in nodes]
    eval_points = [mp.convert(v) for v in eval_points]
    n = len(nodes)
    if n == 0:
        raise ValueError("at least one node is required")
    if len(set(nodes)) != n:
        raise ValueError("nodes must be distinct")
    real = all(not isinstance(v, mp.mpc) or v.imag == 0 for v in nodes + eval_points)
    if real:
        from flint import arb, arb_mat, ctx as flint_ctx

        flint_ctx.prec = int(mp.mp.dps * 3.33) + 20
        xs = [arb(mp.mpf(mp.re(v))) for v in nodes]
        mids = [mp.mpf(mp.re(v)) for v in nodes]
        one, zero = arb(1), arb(0)
    else:
        xs = mids = [mp.mpc(v) if isinstance(v, mp.mpc) and v.imag != 0 else mp.mpf(mp.re(v)) for v in nodes]
        one, zero, arb_mat = mp.mpf(1), mp.mpf(0), None
    weights = []
    for i in range(n):
        denominator = one
        for j in range(n):
            if j != i:
                denominator *= xs[i] - xs[j]
        weights.append(one / denominator)
    rows = []
    for point in eval_points:
        p_mid = mp.mpf(mp.re(point)) if real else point
        hit = next((i for i in range(n) if mids[i] == p_mid), None)
        if hit is not None:
            rows.append([one if i == hit else zero for i in range(n)])
            continue
        x = arb(p_mid) if arb_mat is not None else p_mid
        diffs = [x - xi for xi in xs]
        ell = one
        for d in diffs:
            ell *= d
        rows.append([ell * weights[i] / diffs[i] for i in range(n)])
    if arb_mat is not None:
        return arb_mat(rows) if rows else arb_mat(0, n)
    return rows


def _lagrange_sum_flint_complex(basis, denominators, values, *,
                                _arb_basis=None, _arb_denoms=None):
    """Complex counterpart of `_lagrange_sum_flint_real`, accumulating in
    acb_poly space.

    The Lagrange basis polynomials and denominators are real (the sample nodes
    are real); only the function `values` are complex (e.g. c=2 chiral blocks).
    Without this path the complex case falls back to O(N^2) mpmath
    `Polynomial` multiply/add accumulation, which dominates the large-spin
    build at high Lambda. Returns a Polynomial.
    """
    from flint import arb, acb, acb_poly, ctx as flint_ctx
    from .polynomial import Polynomial as Poly, _acb_to_mpc, _mpf_to_acb

    prec_bits = int(mp.mp.dps * 3.33) + 20
    flint_ctx.prec = prec_bits

    n = len(basis)
    if _arb_basis is not None and n > 0:
        degree = _arb_basis[0].degree()
    elif n > 0 and basis[0] is not None:
        degree = basis[0].degree
    else:
        degree = 0

    if _arb_basis is not None:
        acb_basis = [acb_poly(ab) for ab in _arb_basis]
        denom_vals = _arb_denoms
    else:
        acb_basis = [
            acb_poly([_mpf_to_acb(c) for c in b.coeffs]) for b in basis
        ]
        denom_vals = [_mpf_to_acb(d) for d in denominators]

    result = acb_poly([acb(0)] * (degree + 1))
    for i in range(n):
        v = values[i]
        if isinstance(v, (acb, arb)):
            v_acb = acb(v)
        else:
            v_acb = _mpf_to_acb(v)
        scalar = v_acb / denom_vals[i]
        result = result + acb_basis[i] * scalar

    coeffs = [_acb_to_mpc(c) for c in result.coeffs()]
    return Poly(coeffs, field=mp.mpc)


def build_reduced_delta_polynomial(
    sample_points: Sequence,
    evaluator: Callable,
    prefactor: DampedRational,
    *,
    shift=0,
    min_pole_distance=None,
):
    """Interpolate the prefactor-divided function of Delta.

    Returns a `Polynomial` in x = Delta - shift whose values match

        evaluator(Delta) / prefactor(Delta)

    at the provided sample points. If `min_pole_distance` is passed, the
    regulated prefactor is used instead of the exact one.
    """
    if len(sample_points) == 0:
        raise ValueError("at least one sample point is required")

    reduced_values = []
    for point in sample_points:
        x = mp.mpc(point) - mp.mpc(shift)
        if min_pole_distance is None:
            pref_value = prefactor.evaluate(x)
        else:
            pref_value = prefactor.evaluate_regulated(x, min_pole_distance)
        reduced_values.append(evaluator(point) / pref_value)
    return lagrange_interpolation_polynomial(sample_points, reduced_values, shift=shift)


def build_delta_interpolant(
    sample_points: Sequence,
    evaluator: Callable,
    prefactor: DampedRational,
    *,
    shift=0,
    min_pole_distance=None,
):
    """Bundle the reduced polynomial and reconstruction data into one object."""
    reduced_polynomial = build_reduced_delta_polynomial(
        sample_points,
        evaluator,
        prefactor,
        shift=shift,
        min_pole_distance=min_pole_distance,
    )
    return DeltaInterpolant(
        prefactor=prefactor,
        shift=shift,
        sample_points=tuple(sample_points),
        reduced_polynomial=reduced_polynomial,
        min_pole_distance=min_pole_distance,
    )


def compute_interpolation_error_curve(
    Delta_values: Sequence,
    evaluator: Callable,
    interpolant: DeltaInterpolant,
    *,
    mode: str = "relative",
    floor=mp.mpf("1e-30"),
):
    """Evaluate interpolation error on a Delta grid."""
    if len(Delta_values) == 0:
        raise ValueError("Delta_values must not be empty")
    errors = []
    for Delta in Delta_values:
        if mode == "relative":
            error = interpolant.relative_error(evaluator, Delta, floor=floor)
        elif mode == "absolute":
            error = interpolant.absolute_error(evaluator, Delta)
        elif mode == "difference":
            error = interpolant.signed_difference(evaluator, Delta, tol=floor)
        else:
            raise ValueError("mode must be one of: relative, absolute, difference")
        errors.append(error)
    return errors


def build_interpolants_for_npoles(
    prefactor: DampedRational,
    npoles_values: Sequence[int],
    evaluator: Callable,
    *,
    shift=0,
    num_sample_points: int | None = None,
    sample_point_kwargs: dict | None = None,
    min_pole_distance=None,
):
    """Build one interpolant per pole cap, translating real nodes by a real shift."""
    if len(npoles_values) == 0:
        raise ValueError("npoles_values must not be empty")
    shift = _as_real(mp.convert(shift), name="shift")
    if not mp.isfinite(shift):
        raise ValueError("shift must be finite")
    sample_point_kwargs = {} if sample_point_kwargs is None else dict(sample_point_kwargs)

    interpolants = {}
    for npoles in npoles_values:
        npoles = _validate_npoles(npoles)
        reduced_prefactor = prefactor.take_rightmost_poles(npoles)
        sample_count = num_sample_points
        if sample_count is None:
            sample_count = reduced_prefactor.pole_degree + 1
        sample_points_x = get_sample_points(
            reduced_prefactor,
            sample_count,
            **sample_point_kwargs,
        )
        sample_points = [mp.mpf(point) + shift for point in sample_points_x]
        interpolants[npoles] = build_delta_interpolant(
            sample_points,
            evaluator,
            reduced_prefactor,
            shift=shift,
            min_pole_distance=min_pole_distance,
        )
    return interpolants


def interpolation_error_curves_for_npoles(
    Delta_values: Sequence,
    evaluator: Callable,
    prefactor: DampedRational,
    npoles_values: Sequence[int],
    *,
    shift=0,
    num_sample_points: int | None = None,
    sample_point_kwargs: dict | None = None,
    min_pole_distance=None,
    mode: str = "relative",
    floor=mp.mpf("1e-30"),
):
    """Return interpolation error curves keyed by requested `npoles`."""
    interpolants = build_interpolants_for_npoles(
        prefactor,
        npoles_values,
        evaluator,
        shift=shift,
        num_sample_points=num_sample_points,
        sample_point_kwargs=sample_point_kwargs,
        min_pole_distance=min_pole_distance,
    )
    curves = {}
    for npoles, interpolant in interpolants.items():
        curves[npoles] = compute_interpolation_error_curve(
            Delta_values,
            evaluator,
            interpolant,
            mode=mode,
            floor=floor,
        )
    return curves, interpolants


def plot_interpolation_error_curves(
    Delta_values: Sequence,
    error_curves: dict,
    *,
    interpolants: dict | None = None,
    title: str | None = None,
    output_path=None,
    ylabel: str | None = None,
    log_scale: bool = True,
):
    """Plot interpolation error curves.

    Requires `matplotlib` to be installed. The function returns `(fig, ax)`.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_interpolation_error_curves requires matplotlib; install with .[plot]"
        ) from exc

    fig, ax = plt.subplots()
    x_values = [float(_as_real(Delta, name="Delta")) for Delta in Delta_values]
    for npoles in sorted(error_curves):
        raw_y_values = [mp.mpc(value) for value in error_curves[npoles]]
        if log_scale:
            y_values = [float(abs(value)) for value in raw_y_values]
            line = ax.semilogy(x_values, y_values, label=f"npoles = {npoles}")[0]
        else:
            y_values = [float(_as_real(value, name="curve_value")) for value in raw_y_values]
            line = ax.plot(x_values, y_values, label=f"npoles = {npoles}")[0]
        if interpolants is not None and npoles in interpolants:
            node_x = [float(point) for point in interpolants[npoles].sample_points]
            ax.scatter(node_x, [0.0] * len(node_x), color=line.get_color(), s=22, zorder=3)
            for node in node_x:
                ax.axvline(node, color=line.get_color(), alpha=0.12, linewidth=0.8)
    ax.set_xlabel("Delta")
    ax.set_ylabel(("relative error" if log_scale else "exact - approximated") if ylabel is None else ylabel)
    if title is not None:
        ax.set_title(title)
    if not log_scale:
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    return fig, ax


def max_abs_errors_by_npoles(error_curves: dict):
    """Maximum absolute error per pole count; any nonfinite sample gives infinity."""
    summary = {}
    for npoles, curve in error_curves.items():
        if len(curve) == 0:
            raise ValueError("each error curve must contain at least one value")
        maximum = mp.mpf(0)
        for value in curve:
            value = mp.convert(value)
            if not mp.isfinite(value):
                maximum = mp.inf
                break
            maximum = max(maximum, abs(value))
        summary[npoles] = maximum
    return summary


def plot_max_abs_error_by_npoles(
    error_curves: dict,
    *,
    title: str | None = None,
    output_path=None,
    ylabel: str | None = None,
):
    """Plot log10(max_Delta |error|) as a function of kept pole count."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_max_abs_error_by_npoles requires matplotlib; install with .[plot]"
        ) from exc

    summary = max_abs_errors_by_npoles(error_curves)
    x_values = sorted(summary)
    y_values = []
    for npoles in x_values:
        error = summary[npoles]
        if not mp.isfinite(error) or error <= 0:
            raise ValueError("max abs error must be finite and > 0 to plot log10")
        y_values.append(float(mp.log10(error)))

    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, marker="o")
    ax.set_xlabel("n_poles")
    ax.set_ylabel(
        "log10(max |interpolation error|)"
        if ylabel is None
        else ylabel
    )
    if title is not None:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    return fig, ax
