"""Uniform pointwise evaluation for the public block API.

This module is intentionally an adapter layer.  The established recurrence and
global-block implementations remain the numerical authorities; this module
only normalizes requests, points, external/exchanged labels, and metadata.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import mpmath as mp

from ._conventions import (
    SCALAR_G_ODD_EVEN, SCALAR_G_ODD_KEY, scalar_G_spin0_convention, scalar_G_spin0_weight,
)
from .composer import B_block, G_block, parity_projected_B_block
from .coordinates import COORDINATES
from .derivs import _global_chiral_block
from .global_blocks import global_evaluate_block
from .params import b_from_c
from .polyblock import chiral_denominator_roots
from .recursion import F_block
from .requests import (
    BlockPoleError,
    BlockRequest,
    BlockValue,
    RequestValidationError,
    validate_request,
)
from .singularity import snap_float_c, recognized_singular_locus


def _mp(value):
    if isinstance(value, (mp.mpf, mp.mpc)):
        return value
    if isinstance(value, complex):
        return mp.mpc(value)
    if isinstance(value, (int, float, str)):
        return mp.mpf(value)
    # fractions.Fraction, decimal.Decimal, numpy scalars, ...: mpmath's own
    # conversion keeps an exact rational exact at the working precision.
    return mp.mpmathify(value)


def _label(value):
    from .numeric_input import parse_number
    from fractions import Fraction

    value = parse_number(value)
    return value if isinstance(value, (int, Fraction)) else _mp(value)


def _label_combination(terms):
    """``sum(sign * label)`` from exact label components, and the magnitude scale of its terms."""
    from .numeric_input import ExactRational, literal_components

    real, imaginary, scale = ExactRational(0), ExactRational(0), 1
    for sign, value in terms:
        a, b = literal_components(value)
        real, imaginary = real + sign * a, imaginary + sign * b
        scale = max(scale, abs(a), abs(b))
    return real, imaginary, scale


def _within_roundoff(real, imaginary, scale, sources):
    """Whether the combination ``real + i imaginary`` of ``sources`` vanishes.

    Exact rational labels decide exactly. An mpmath input is a numerical value
    at its precision, so a combination involving one vanishes within roundoff
    at the working precision (mpf('2.65') - mpf('0.65') is not exactly 2 in
    binary, but it is 2 at the working precision).
    """
    from fractions import Fraction
    from .precision import tolerance

    if all(isinstance(_label(value), (int, Fraction)) for value in sources):
        return real == 0 and imaginary == 0
    return abs(mp.mpc(mp.convert(real), mp.convert(imaginary))) <= tolerance(guard=5) * scale


def _scalar_G_spin(request, h, hbar) -> int:
    """Return the scalar_G spin ``h - hbar``, rejecting noninteger values."""
    real, imaginary, scale = _label_combination(((1, h), (-1, hbar)))
    ell = round(real)
    if not _within_roundoff(real - ell, imaginary, scale, (h, hbar)):
        raise RequestValidationError(
            f"scalar_G needs an integer spin h - hbar, got "
            f"{mp.nstr(mp.mpc(mp.convert(real), mp.convert(imaginary)) if imaginary else mp.convert(real), 12)}; "
            "use ordered_B"
        )
    ell = int(ell)
    if request.exchange_spin is not None and _exchange_spin_label(request) != ell:
        raise RequestValidationError("exchange_spin does not match h - hbar")
    return ell


def _scalar_G_shifts(request):
    """scalar_G is z <-> zbar symmetric: one shift S multiplies it by (z zbar)^S, as in its tables.

    ``z_shift`` applies to both sides. An explicit ``zbar_shift`` must match it
    (default zero), as required by the table CLI and Python API.
    """
    shift = request.z_shift
    if request.zbar_shift is not None and _label(request.zbar_shift) != _label(shift if shift is not None else 0):
        raise RequestValidationError(
            "scalar_G is z<->zbar symmetric and carries one shift; a zbar_shift that differs from z_shift cannot be "
            "applied (independent shifts need ordered_B or parity_projected_B)"
        )
    return shift, shift


def _same(value, target):
    """Compare supplied labels without a tolerance or mixed-type rounding."""
    from .numeric_input import literal_components

    return literal_components(value) == literal_components(target)



def _pole_same(value, target):
    # Pole recognition only accommodates rounding in the computed root.
    left, right = mp.convert(value), mp.convert(target)
    return abs(left - right) <= 8 * mp.eps * max(1, abs(left), abs(right))


def _canonical_basis(request: BlockRequest) -> str:
    family = request.family
    basis = request.basis
    if family == "torus-character":
        return "torus_G"
    if family == "torus-onepoint":
        return "torus_onept_G"
    if basis == "auto":
        if request.coordinate in {"z", "y", "q"}:
            short = "chiral_F"
        elif request.parity:
            short = "parity_projected_B"
        elif (
            request.hbar_ext is not None and request.h_ext is not None
            and any(not _same(a, b) for a, b in zip(request.h_ext, request.hbar_ext))
        ) or any(_mp(s) != 0 for s in (request.spin_ext or ())):
            # spinning externals; scalar ones spelled as h = hbar pick scalar_G, as the table CLIs do
            short = "ordered_B"
        else:
            short = "scalar_G"
        return short if family == "virasoro" else f"{family}_{short}"
    prefixes = ("virasoro_", "global_", "ns_", "ramond_")
    for prefix in prefixes:
        if basis.startswith(prefix):
            stripped = basis[len(prefix):]
            expected = "virasoro" if prefix == "virasoro_" else prefix[:-1]
            if expected != family and not (family == "virasoro" and expected == "virasoro"):
                raise RequestValidationError(
                    f"basis {basis!r} conflicts with family {family!r}"
                )
            return basis
    if family in {"ns", "ramond", "global"}:
        return f"{family}_{basis}"
    return basis


def _short_basis(basis: str) -> str:
    for prefix in ("global_", "ns_", "ramond_", "virasoro_"):
        if basis.startswith(prefix):
            return basis[len(prefix):]
    return basis


def _refuse_unused_inputs(request: BlockRequest, *, short: str, chiral: bool) -> None:
    """Reject request fields that do not apply to the selected block."""
    torus = request.family in {"torus-character", "torus-onepoint"}
    unused = []
    if torus:
        unused += [name for name in ("chiral_norm", "z_shift", "zbar_shift") if getattr(request, name) is not None]
    elif chiral and request.zbar_shift is not None:
        unused.append("zbar_shift")
    if short != "parity_projected_B":
        unused += [name for name in ("parity_external", "parity_exchanged") if getattr(request, name) is not None]
    if request.family not in {"ns", "ramond"} and request.sector != "even":
        unused.append("sector")
    if request.family != "ramond" and request.branch != "++":
        unused.append("branch")
    if request.family != "torus-onepoint":
        unused += [name for name in ("torus_onepoint_ext", "torus_scalar_shift") if getattr(request, name) is not None]
    # table-only fields (a point value has no positivity anchor, expansion center or coordinate list)
    unused += [name for name in ("gap", "center", "center_bar") if getattr(request, name) is not None]
    if request.coordinates:
        unused.append("coordinates (evaluate takes one coordinate)")
    if short != "parity_projected_B" and request.parity is not None:
        unused.append("parity")
    if chiral:
        unused += [name for name in ("exchange_hbar", "exchange_delta", "exchange_spin")
                   if getattr(request, name) is not None]
    if torus:
        unused += [name for name in ("h_ext", "hbar_ext", "delta_ext", "spin_ext", "exchange_h", "exchange_hbar",
                                     "parity") if getattr(request, name) is not None and name not in unused]
    elif request.h_ext is not None:
        # External chiral weights and dimension/spin labels are mutually exclusive.
        unused += [name for name in ("delta_ext", "spin_ext") if getattr(request, name) is not None]
    elif request.hbar_ext is not None:
        unused.append("hbar_ext (without h_ext)")
    if unused:
        raise RequestValidationError(
            f"{', '.join(unused)} do(es) not apply to {request.family} {short} evaluation"
        )
    if torus:
        allowed = {"v", "vbar"}
    else:
        spec = COORDINATES.get(request.coordinate)
        var = getattr(spec, "chiral_var", None) or request.coordinate
        allowed = {"z", request.coordinate, var} if chiral else {"z", "zbar", var, f"{var}bar"}
    names = set(dict(request.point))
    extra = sorted(names - allowed)
    if extra:
        raise RequestValidationError(
            f"point name(s) {extra} are not used by coordinate {request.coordinate!r} (use {sorted(allowed)})"
        )
    if not torus and len(names & ({"z"} if chiral else {"z", "zbar"})) and len(names - {"z", "zbar"}):
        raise RequestValidationError("give the point as z/zbar or in the coordinate's own variable, not both")


def _resolve_external_h(request: BlockRequest, *, chiral: bool):
    from fractions import Fraction
    from .numeric_input import _label_scale, _label_sum

    if request.h_ext is not None:
        h_ext = tuple(_label(x) for x in request.h_ext)
        if len(h_ext) != 4:
            raise RequestValidationError("h_ext must contain four weights")
        if chiral:
            if request.hbar_ext is not None:
                raise RequestValidationError("a chiral request cannot have hbar_ext")
            return h_ext, None
        if request.hbar_ext is None:
            raise RequestValidationError("full requests need hbar_ext with h_ext")
        hbar_ext = tuple(_label(x) for x in request.hbar_ext)
        if len(hbar_ext) != 4:
            raise RequestValidationError("hbar_ext must contain four weights")
        return h_ext, hbar_ext

    if request.delta_ext is None:
        raise RequestValidationError(
            "provide h_ext (and hbar_ext for full blocks), or delta_ext"
        )
    delta = tuple(_label(x) for x in request.delta_ext)
    if len(delta) != 4:
        raise RequestValidationError("delta_ext must contain four dimensions")
    spin = tuple(_label(x) for x in (request.spin_ext or (0, 0, 0, 0)))
    if len(spin) != 4:
        raise RequestValidationError("spin_ext must contain four spins")
    h_ext = tuple(_label_scale(_label_sum(d, s), Fraction(1, 2)) for d, s in zip(delta, spin))
    hbar_ext = tuple(_label_scale(_label_sum(d, _label_scale(s, -1)), Fraction(1, 2))
                     for d, s in zip(delta, spin))
    if chiral:
        if any(s != 0 for s in spin):
            raise RequestValidationError("chiral blocks require scalar external data")
        return h_ext, None
    return h_ext, hbar_ext


def _point_value(points: Mapping[str, Any], key: str):
    if key not in points:
        return None
    return _mp(points[key])


def _physical_points(request: BlockRequest):
    """Resolve keyed coordinate values to physical cross ratios z/zbar."""
    coord = request.coordinate
    if request.family in {"torus-character", "torus-onepoint"}:
        v = _point_value(request.point, "v")
        vbar = _point_value(request.point, "vbar")
        if v is None or vbar is None:
            raise RequestValidationError(
                "torus evaluation needs --point v=... --point vbar=..."
            )
        return v, vbar
    spec = COORDINATES.get(coord)
    if spec is None or spec.kind in {"folded_2d", "radial"}:
        raise RequestValidationError(
            f"direct evaluation does not support folded/radial coordinate {coord!r}"
        )
    points = dict(request.point)

    def convert_point(value, key):
        try:
            converted = spec.z_of_u(value)
        except (ZeroDivisionError, ValueError, OverflowError) as exc:
            reason = str(exc).strip() or "the coordinate map has a pole"
            raise RequestValidationError(
                f"cannot convert {key} = {mp.nstr(value, 12)} to a finite cross ratio: {reason}"
            ) from exc
        if not mp.isfinite(converted):
            raise RequestValidationError(f"{key} = {mp.nstr(value, 12)} does not map to a finite cross ratio")
        return converted

    if spec.kind == "chiral":
        raw = _point_value(points, "z")
        if raw is None:
            raw = _point_value(points, coord)
            if raw is None:
                raise RequestValidationError(
                    f"chiral evaluation needs --point z=... or --point {coord}=..."
                )
            raw = convert_point(raw, coord)
        return raw, None

    raw_z = _point_value(points, "z")
    raw_zbar = _point_value(points, "zbar")
    if raw_z is None:
        raw_z = _point_value(points, spec.chiral_var)
        if raw_z is not None:
            raw_z = convert_point(raw_z, spec.chiral_var)
    if raw_zbar is None:
        raw_zbar = _point_value(points, f"{spec.chiral_var}bar")
        if raw_zbar is not None:
            raw_zbar = convert_point(raw_zbar, f"{spec.chiral_var}bar")
    if raw_z is None or raw_zbar is None:
        raise RequestValidationError(
            "full evaluation needs --point z=... --point zbar=... "
            f"(or {spec.chiral_var}/{spec.chiral_var}bar values)"
        )
    return raw_z, raw_zbar


def _euclidean_side(x, side: int):
    """A real cross ratio on a cut, (-oo, 0) or (1, oo), as its limit from above (side +1) or below (side -1).

    The Euclidean section is zbar = z*, so a real point z = zbar = x outside [0, 1] is z = x + i0, zbar = x - i0;
    evaluating both halves on the principal branch at x itself gave a complex value where the block is real (x > 1,
    and x < 0: c = 4.5, x = -0.3 gave -0.0056 - 0.0077i for 0.0095641).  The offset is far below the working
    precision.
    """
    value = _mp(x)
    if mp.im(value) != 0 or 0 < mp.re(value) < 1:
        return value
    if mp.re(value) in (0, 1):
        raise RequestValidationError("the block is singular at z = 0 and z = 1 (or zbar = 0, 1)")
    return mp.mpc(value, side * mp.mpf(10) ** (-(mp.mp.dps + 10)) * abs(value))


def _exchange_spin_label(request):
    from .numeric_input import ExactRational, _signed_half_integer

    spin = 0 if request.exchange_spin is None else request.exchange_spin
    try:
        if isinstance(spin, (mp.mpf, mp.mpc)):
            # A numerical spin is a half-integer within roundoff.
            real, imaginary, scale = _label_combination(((2, spin),))
            twice = round(real)
            if not _within_roundoff(real - twice, imaginary, scale, (spin,)):
                raise ValueError("not a half-integer")
            spin = ExactRational(twice, 2)
        return _signed_half_integer(spin)
    except ValueError as exc:
        raise RequestValidationError("exchange_spin must be an integer or half-integer") from exc


def _resolve_exchange(request: BlockRequest, *, chiral: bool):
    if chiral:
        if request.exchange_h is None:
            raise RequestValidationError("chiral evaluation needs exchange_h")
        return _label(request.exchange_h), None
    if request.exchange_h is not None or request.exchange_hbar is not None:
        if request.exchange_h is None or request.exchange_hbar is None:
            raise RequestValidationError(
                "full h-parametrized evaluation needs exchange_h and exchange_hbar"
            )
        h = _label(request.exchange_h)
        hbar = _label(request.exchange_hbar)
        if request.exchange_delta is not None:
            delta = _label(request.exchange_delta)
            if not _within_roundoff(*_label_combination(((1, h), (1, hbar), (-1, delta))), (h, hbar, delta)):
                raise RequestValidationError(
                    "exchange_delta must equal exchange_h + exchange_hbar"
                )
        if request.exchange_spin is not None:
            ell = _exchange_spin_label(request)
            if not _within_roundoff(*_label_combination(((1, h), (-1, hbar), (-1, ell))), (h, hbar)):
                raise RequestValidationError(
                    "exchange_spin must be an integer or half-integer equal to "
                    "exchange_h - exchange_hbar"
                )
        return h, hbar
    if request.exchange_delta is None:
        raise RequestValidationError(
            "full evaluation needs exchange_delta or exchange_h/exchange_hbar"
        )
    ell = _exchange_spin_label(request)
    from .numeric_input import _exchange_weight

    delta = _label(request.exchange_delta)
    return _exchange_weight(delta, ell), _exchange_weight(delta, -ell)


def _check_virasoro_pole(h, h_ext, *, c, order, basis, side=None):
    """Fail closed at generic-c genuine Virasoro Kac poles.

    Singular-c calls are delegated to the existing c-regularization path,
    which has its own genuine-pole convergence check.  The fast denominator
    root path is valid only away from those collision loci.
    """
    if recognized_singular_locus(c) is not None:
        return
    roots = chiral_denominator_roots(h_ext, c=c, q_order=max(1, int(order)))
    for pole in roots:
        if _pole_same(h, pole):
            raise BlockPoleError(
                family="virasoro", basis=basis, exchanged=h, pole=pole, side=side
            )


def _check_global_pole(h, h_ext, *, basis, side=None, count=16):
    from .global_blocks import _externals_diffs, _genuine_chiral_poles

    d12, d34 = _externals_diffs(h_ext)
    # The pole ladder is unbounded to the left. Search around the requested
    # weight, not zero, so negative weights beyond the first levels receive
    # the same genuine-pole check. Equality below still uses the full value,
    # including any deliberate real offset or imaginary component.
    for pole in _genuine_chiral_poles(
        max(4, int(count)), delta12=d12, delta34=d34, shift=mp.re(_mp(h))
    ):
        if _pole_same(h, pole):
            raise BlockPoleError(
                family="global", basis=basis, exchanged=h, pole=pole, side=side
            )


def _input_metadata_parts(real, imaginary):
    """Serialize exact components of a supplied or derived input label."""
    from .writers import _input_metadata_num_str

    if imaginary == 0:
        return _input_metadata_num_str(real, mp.mp.dps)
    return {"real": _input_metadata_num_str(real, mp.mp.dps),
            "imag": _input_metadata_num_str(imaginary, mp.mp.dps)}


def _input_metadata_number(value):
    """Serialize supplied labels exactly, keeping the complex metadata schema."""
    from fractions import Fraction
    from .numeric_input import literal_components, literal_number

    value = literal_number(value)
    if isinstance(value, Fraction):
        return _input_metadata_parts(value, 0)
    if isinstance(value, (complex, mp.mpc)):
        return _input_metadata_parts(*literal_components(value))
    return BlockValue._number(_mp(value))


def _metadata(request, *, basis, point, exchange, regularization="direct"):
    policy = request.policy
    out = {
        "surface": request.surface,
        "family": request.family,
        "basis": basis,
        "coordinate": request.coordinate,
        "point": {
            key: (_input_metadata_number(request.point[key]) if key in request.point
                  else _input_metadata_number(value)) for key, value in point.items()
        },
        "precision": int(policy.precision),
        "order": (
            None if request.family == "global" or policy.order is None
            else int(policy.order)
        ),
        "regularization_policy": regularization,
        "pole_policy": policy.pole_policy,
    }
    if request.central_charge is not None:
        out["central_charge"] = _input_metadata_number(request.central_charge)
    if request.h_ext is not None:
        out["external"] = {
            "h": [_input_metadata_number(value) for value in request.h_ext],
        }
        if request.hbar_ext is not None:
            out["external"]["hbar"] = [
                _input_metadata_number(value) for value in request.hbar_ext
            ]
    elif request.delta_ext is not None:
        out["external"] = {
            "delta": [_input_metadata_number(value) for value in request.delta_ext],
            "spin": [
                _input_metadata_number(value)
                for value in (request.spin_ext or (0, 0, 0, 0))
            ],
        }
    if request.family in {"ns", "ramond"}:
        out["sector"] = request.sector
        out["branch"] = request.branch
    if str(basis).endswith("scalar_G"):
        # which spin-0 normalization produced the value, and that odd spins are z <-> zbar even (see _conventions)
        out["scalar_G_spin0"] = scalar_G_spin0_convention()
        out[SCALAR_G_ODD_KEY] = SCALAR_G_ODD_EVEN
    if request.family == "torus-onepoint":
        out["torus_onepoint_ext"] = _input_metadata_number(request.torus_onepoint_ext)
    if request.parity is not None:
        out["parity"] = request.parity
    if request.parity_external is not None:
        from .parity import normalize_external_parity

        out["parity_external"] = normalize_external_parity(request.parity_external)
    if request.parity_exchanged is not None:
        from .parity import normalize_parity_phase

        out["parity_exchanged"] = normalize_parity_phase(request.parity_exchanged)
    for key in ("chiral_norm", "z_shift", "zbar_shift"):
        if getattr(request, key) is not None:
            out[key] = _input_metadata_number(getattr(request, key))
    if exchange[1] is None:
        out["exchange"] = {"h": _input_metadata_number(exchange[0])}
    else:
        from .numeric_input import literal_components

        hr, hi = literal_components(exchange[0])
        hbr, hbi = literal_components(exchange[1])
        out["exchange"] = {
            "h": _input_metadata_number(exchange[0]),
            "hbar": _input_metadata_number(exchange[1]),
            "delta": _input_metadata_parts(hr + hbr, hi + hbi),
            "spin": (_input_metadata_number(request.exchange_spin)
                     if request.exchange_spin is not None
                     else _input_metadata_parts(hr - hbr, hi - hbi)),
        }
    return out


def _check_torus_onepoint_pole(kernel, h, *, basis, side=None):
    from .torus_onept import _kernel_poles

    for pole in _kernel_poles(kernel):
        if _pole_same(h, pole):
            raise BlockPoleError(
                family="torus-onepoint", basis=basis, exchanged=h,
                pole=pole, side=side,
            )


def _torus_metadata(request, *, basis, point, exchange, order, regularization="direct"):
    metadata = _metadata(
        request, basis=basis, point=point, exchange=exchange,
        regularization=regularization,
    )
    metadata["order"] = order
    return metadata


def _evaluate_block_request(
    request: BlockRequest
) -> BlockValue:
    """Evaluate one public point request through the family adapter layer."""
    validate_request(request)
    if request.operation != "evaluate":
        raise RequestValidationError("evaluate_block_request needs operation='evaluate'")
    if request.family not in {
        "virasoro", "global", "ns", "ramond", "torus-character", "torus-onepoint"
    }:
        raise NotImplementedError(
            f"direct evaluation adapter for family {request.family!r} is not wired yet"
        )
    if request.family in {"torus-character", "torus-onepoint"}:
        if request.surface != "torus":
            raise RequestValidationError("torus families require surface='torus'")
        if request.central_charge is None:
            raise RequestValidationError("central_charge is required for torus evaluation")
    elif request.surface != "sphere":
        raise RequestValidationError("sphere is required for sphere block evaluation")
    elif request.family != "global" and request.central_charge is None:
        raise RequestValidationError(
            f"central_charge is required for {request.family} evaluation"
        )

    basis = _canonical_basis(request)
    short = _short_basis(basis)
    chiral = short == "chiral_F"
    if request.family in {"torus-character", "torus-onepoint"}:
        if request.coordinate != "vvb":
            raise RequestValidationError(
                "torus direct evaluation requires coordinate='vvb'"
            )
    elif chiral and request.coordinate not in {"z", "y", "q"}:
        raise RequestValidationError(
            f"chiral basis {basis!r} requires coordinate z, y, or q"
        )
    elif not chiral and request.coordinate in {"z", "y", "q"}:
        raise RequestValidationError(
            f"full basis {basis!r} requires coordinate zzb, yyb, or qqb"
        )
    _refuse_unused_inputs(request, short=short, chiral=chiral)
    order = request.policy.order
    if order is None and request.family not in {"global", "torus-character"}:
        raise RequestValidationError("this direct evaluator needs an order")
    from .verified_build import precision_request_of

    with mp.workdps(max(mp.mp.dps, precision_request_of(request.policy).input_dps)):
        z, zbar = _physical_points(request)
    if request.family in {"torus-character", "torus-onepoint"}:
        c = snap_float_c(request.central_charge)
        if request.exchange_delta is None:
            raise RequestValidationError("torus evaluation needs exchange_delta")
        from .table_adapters import _torus_spin

        spin = _torus_spin(0 if request.exchange_spin is None else request.exchange_spin)
        if abs(z) >= 1 or abs(zbar) >= 1:
            # tau = i (1 + v)/(1 - v) leaves the upper half plane: |q| > 1 and the q-series diverges (-1e572 at 1.5)
            raise RequestValidationError(f"torus points need |v| < 1 and |vbar| < 1 (Im tau > 0), got v = "
                                         f"{mp.nstr(z, 8)}, vbar = {mp.nstr(zbar, 8)}")
        delta = _label(request.exchange_delta)
        from .verified_build import precision_request_of

        # One-point evaluation may escalate. Keep the derived labels as accurate
        # as the parsed inputs rather than freezing their initial rounding.
        with mp.workdps(precision_request_of(request.policy).input_dps):
            h = (delta + spin) / 2
            hbar = (delta - spin) / 2
        extra = {}
        if request.family == "torus-character":
            from .modular import torus_disk_point

            from .torus_onept import irreducible_character_factor

            _tau, _taubar, tau2, q, qbar = torus_disk_point(z, zbar)
            cshift = (c - 1) / 24
            # on a Kac weight the character is the irreducible module's (the Verma one counts its null states)
            g_h, deg_h = irreducible_character_factor(c, h, q)
            g_hb, deg_hb = irreducible_character_factor(c, hbar, qbar)
            value = mp.sqrt(tau2) * q ** (h - cshift) * qbar ** (hbar - cshift) * g_h * g_hb
            if spin != 0:
                g_hq, _ = irreducible_character_factor(c, hbar, q)
                g_hqb, _ = irreducible_character_factor(c, h, qbar)
                value += mp.sqrt(tau2) * q ** (hbar - cshift) * qbar ** (h - cshift) * g_hq * g_hqb
            irreducible_used = deg_h or deg_hb
        else:
            if request.torus_onepoint_ext is None:
                raise RequestValidationError(
                    "torus-onepoint evaluation needs torus_onepoint_ext"
                )
            if int(order) < 1:
                raise RequestValidationError(
                    "torus-onepoint evaluation needs order >= 1 (order 0 is the bare character, f = 1)"
                )
            d_ext = _label(request.torus_onepoint_ext)
            from .torus_onept import build_torus_onept_state, torus_onept_point_value

            # the Laurent kernel at a singular c (c = 25, 26, ...), where the plain kernel's Kac weights collide
            kernel = build_torus_onept_state(int(order), d_ext=d_ext, c=c)
            from .torus_onept import kac_hit_weight

            # on a Kac weight of a singular c the value is the irreducible-module trace (onept_point_f_coeffs);
            # elsewhere a Kac weight is a pole of the block
            if kac_hit_weight(kernel, h) is None:
                _check_torus_onepoint_pole(kernel, h, basis=basis, side="holomorphic")
            if kac_hit_weight(kernel, hbar) is None:
                _check_torus_onepoint_pole(kernel, hbar, basis=basis, side="antiholomorphic")
            # at the working precision, verified against a deeper, higher-precision reference (the recursion
            # cancels; target_digits escalates); the order tail is measured and reported
            value, extra = torus_onept_point_value(
                c, d_ext, h, hbar, spin, z, zbar, q_order=int(order),
                precision=int(request.policy.precision), target_digits=request.policy.target_digits,
                max_precision=request.policy.max_precision, kernel=kernel,
            )
        from .torus_onept import TorusOnePtLaurentState, kac_hit_weight

        if request.family == "torus-character":
            regularization_tag = "irreducible" if irreducible_used else "direct"
        elif kac_hit_weight(kernel, h) is not None or kac_hit_weight(kernel, hbar) is not None:
            regularization_tag = "irreducible"
        else:
            regularization_tag = "laurent" if isinstance(kernel, TorusOnePtLaurentState) else "direct"

        # The reported labels must reproduce the escalated value as well.
        with mp.workdps(extra.get("precision_actual", request.policy.precision)):
            metadata = _torus_metadata(
                request, basis=basis, point={"v": z, "vbar": zbar},
                exchange=(h, hbar), order=(None if request.family == "torus-character" else order),
                regularization=regularization_tag,
            )
        if extra:
            metadata["precision_actual"] = extra["precision_actual"]
            verified = extra.get("verified_digits")
            metadata["verified_digits"] = None if verified is None else round(verified, 1)
            truncation = extra["q_truncation_digits"]
            # digits the order truncation leaves at this point (its tail against a deeper reference)
            metadata["q_truncation_digits"] = None if truncation is None else round(truncation, 1)
        return BlockValue(value=value, metadata=metadata)
    h_ext, hbar_ext = _resolve_external_h(request, chiral=chiral)
    h, hbar = _resolve_exchange(request, chiral=chiral)
    if short == "scalar_G":
        if any(not _same(a, b) for a, b in zip(h_ext, hbar_ext)):
            prefix = "" if request.family == "virasoro" else f"{request.family} "
            raise RequestValidationError(f"{prefix}scalar_G requires scalar external weights")
        ell = _scalar_G_spin(request, h, hbar)
    reported = (z, zbar)
    z = _euclidean_side(z, +1)
    zbar = None if zbar is None else _euclidean_side(zbar, -1)
    if short == "parity_projected_B" and request.parity in {"+", "-"} and _same(z, zbar):
        from .conditioning import report
        from .parity import parity_reversal_phase, parity_projection_is_zero

        var = "z" if "z" in request.point else COORDINATES[request.coordinate].chiral_var
        left, right = request.point[var], request.point[f"{var}bar"]
        coefficient = (1 if request.parity == "+" else -1) * parity_reversal_phase(
            request.parity_external, exchanged_parity=request.parity_exchanged or "1")
        zbs = (request.z_shift if request.family in {"ns", "ramond"} and request.zbar_shift is None
               else request.zbar_shift)
        if coefficient == -1 and not parity_projection_is_zero(
                coefficient, h, hbar, left, right, h_ext, hbar_ext, request.z_shift, zbs,
                global_block=request.family == "global"):
            # The coordinate conversion rounded distinct labels onto the diagonal. No symmetry proves zero.
            report(float(precision_request_of(request.policy).input_dps), what="coordinate conversion of distinct points")
    if order is None and request.family != "global":
        raise RequestValidationError("direct recursive evaluation needs order")

    if request.family == "virasoro":
        c = snap_float_c(request.central_charge)
        if chiral:
            _check_virasoro_pole(h, h_ext, c=c, order=order, basis=basis)
            value = F_block(
                h, z, h_ext, c=c, q_order=order,
                chiral_norm=request.chiral_norm, z_shift=request.z_shift,
                regularize=request.policy.c_regularization,
            )
        else:
            _check_virasoro_pole(h, h_ext, c=c, order=order, basis=basis, side="holomorphic")
            _check_virasoro_pole(hbar, hbar_ext, c=c, order=order, basis=basis, side="antiholomorphic")
            if short == "ordered_B":
                value = B_block(
                    h, hbar, z, zbar, h_ext, hbar_ext, c=c, q_order=order,
                    chiral_norm=request.chiral_norm, regularize=request.policy.c_regularization,
                    z_shift=request.z_shift, zbar_shift=request.zbar_shift,
                )
            elif short == "scalar_G":
                zs, zbs = _scalar_G_shifts(request)
                value = G_block(
                    h + hbar, int(ell), z, zbar, h_ext,
                    c=c, q_order=order, chiral_norm=request.chiral_norm,
                    regularize=request.policy.c_regularization,
                    z_shift=zs, zbar_shift=zbs,
                )
            elif short == "parity_projected_B":
                if request.parity not in {"+", "-"}:
                    raise RequestValidationError("parity_projected_B needs parity '+' or '-'")
                value = parity_projected_B_block(
                    h, hbar, z, zbar, h_ext, hbar_ext, parity=request.parity,
                    parity_external=request.parity_external,
                    exchanged_parity=request.parity_exchanged or "1",
                    c=c, q_order=order, chiral_norm=request.chiral_norm,
                    regularize=request.policy.c_regularization,
                    z_shift=request.z_shift, zbar_shift=request.zbar_shift,
                )
            else:
                raise RequestValidationError(f"unsupported Virasoro basis {basis!r}")
        # (at a recognized singular c the value is the block AT c, from the merged poles of the Laurent recursion)
        from .singularity import exact_locus_c

        regularization = (
            "merged" if request.policy.c_regularization != "off" and exact_locus_c(c, q_order=order) is not None
            else "direct"
        )
    elif request.family == "global":
        from .conditioning import conditioned_sum

        if chiral:
            _check_global_pole(h, h_ext, basis=basis)
            value = _global_chiral_block(
                h, z, h_ext, chiral_norm=request.chiral_norm, z_shift=request.z_shift
            )
        else:
            _check_global_pole(h, h_ext, basis=basis, side="holomorphic")
            _check_global_pole(hbar, hbar_ext, basis=basis, side="antiholomorphic")
            zs, zbs = (_scalar_G_shifts(request) if short == "scalar_G"
                       else (request.z_shift, request.zbar_shift))
            ordered = global_evaluate_block(
                z, zbar, h, hbar, h_ext, hbar_externals=hbar_ext,
                coord=request.coordinate, chiral_norm=request.chiral_norm,
                z_shift=zs, zbar_shift=zbs,
            )
            if short == "ordered_B":
                value = ordered
            elif short == "scalar_G":
                swapped = global_evaluate_block(
                    z, zbar, hbar, h, h_ext, hbar_externals=h_ext,
                    coord=request.coordinate, chiral_norm=request.chiral_norm,
                    z_shift=zs, zbar_shift=zbs,
                )
                # z <-> zbar even at every spin (see _conventions)
                value = conditioned_sum(ordered, swapped)
                if int(ell) == 0 and scalar_G_spin0_weight() == 1:
                    # the public spin-0 block is the symmetrized product itself (see _conventions)
                    value = value / 2
            elif short == "parity_projected_B":
                if request.parity not in {"+", "-"}:
                    raise RequestValidationError("parity_projected_B needs parity '+' or '-'")
                swapped = global_evaluate_block(
                    z, zbar, hbar, h, hbar_ext, hbar_externals=h_ext,
                    coord=request.coordinate, chiral_norm=request.chiral_norm,
                    z_shift=request.zbar_shift, zbar_shift=request.z_shift,
                )
                from .parity import parity_reversal_phase, parity_projection_is_zero
                # as composer.parity_reversed_B_block: the exchanged parity applies without external parities too
                phase = parity_reversal_phase(
                    request.parity_external,
                    exchanged_parity=request.parity_exchanged or "1",
                )
                sign = 1 if request.parity == "+" else -1
                value = conditioned_sum(ordered, sign * phase * swapped, exact_zero=parity_projection_is_zero(
                    sign * phase, h, hbar, z, zbar, h_ext, hbar_ext, request.z_shift, request.zbar_shift,
                    global_block=True))
            else:
                raise RequestValidationError(f"unsupported global basis {basis!r}")
        regularization = "direct"
    else:
        c = snap_float_c(request.central_charge)
        from .superconformal_blocks import (
            FAMILY_NS, FAMILY_RAMOND, ns_chiral_block,
            is_superconformal_singular_c,
            super_ordered_B_block, super_parity_projected_B_block,
            super_scalar_G_block,
        )
        family = FAMILY_NS if request.family == "ns" else FAMILY_RAMOND
        if chiral:
            evaluator = ns_chiral_block if request.family == "ns" else None
            if evaluator is None:
                from .superconformal_blocks import ramond_chiral_block
                evaluator = ramond_chiral_block
            kwargs = dict(
                c=c, q_order=order, chiral_norm=request.chiral_norm,
                z_shift=request.z_shift, sector=request.sector,
                regularize=request.policy.c_regularization,
            )
            if request.family == "ramond":
                kwargs["branch"] = request.branch
            value = evaluator(h, z, h_ext, **kwargs)
        else:
            if short == "ordered_B":
                value = super_ordered_B_block(
                    family, h, hbar, z, zbar, h_ext, hbar_ext=hbar_ext,
                    c=c, q_order=order, chiral_norm=request.chiral_norm,
                    z_shift=request.z_shift, zbar_shift=request.zbar_shift,
                    sector=request.sector, branch=request.branch,
                    regularize=request.policy.c_regularization,
                )
            elif short == "scalar_G":
                zs, zbs = _scalar_G_shifts(request)
                value = super_scalar_G_block(
                    family, h + hbar, int(ell), z, zbar, h_ext, c=c,
                    q_order=order, chiral_norm=request.chiral_norm,
                    z_shift=zs, zbar_shift=zbs,
                    sector=request.sector, branch=request.branch,
                    regularize=request.policy.c_regularization,
                )
            elif short == "parity_projected_B":
                if request.parity not in {"+", "-"}:
                    raise RequestValidationError("parity_projected_B needs parity '+' or '-'")
                value = super_parity_projected_B_block(
                    family, h, hbar, z, zbar, h_ext, hbar_ext=hbar_ext,
                    parity=request.parity, parity_external=request.parity_external,
                    exchanged_parity=request.parity_exchanged or "1", c=c,
                    q_order=order, chiral_norm=request.chiral_norm,
                    z_shift=request.z_shift, zbar_shift=request.zbar_shift,
                    sector=request.sector, branch=request.branch,
                    regularize=request.policy.c_regularization,
                )
            else:
                raise RequestValidationError(f"unsupported super basis {basis!r}")
        from .super_laurent import exact_singular_c

        regularization = (
            # at the singular c itself: the exact Laurent-in-t recursion; "on" at another c offsets it
            ("laurent" if exact_singular_c(c) is not None else "c+eps")
            if request.policy.c_regularization == "on"
            or (
                request.policy.c_regularization == "auto"
                and is_superconformal_singular_c(c)
            )
            else "direct"
        )

    point = {"z": reported[0]} if chiral else {"z": reported[0], "zbar": reported[1]}
    metadata = _metadata(request, basis=basis, point=point, exchange=(h, hbar), regularization=regularization)
    if request.family == "virasoro" and regularization == "merged":
        from .recursion import kac_limit_mode

        metadata["kac_limit"] = kac_limit_mode()       # the block on a Kac weight of the singular c
    if request.family in ("virasoro", "ns", "ramond"):
        report = _q_tail(request.family, h, hbar, z, zbar, h_ext, hbar_ext, c=c, order=order,
                         sector=request.sector, branch=request.branch, basis=short)
        # (capped at the working precision; evaluate_block_request re-caps it at an escalated one)
        metadata["q_truncation_digits"] = (
            None if report is None or report["digits"] is None
            else round(min(float(report["digits"]), float(mp.mp.dps)), 1)
        )
        _Q_REPORT.set(report)
    return BlockValue(value=value, metadata=metadata)


import contextvars

_Q_REPORT = contextvars.ContextVar("virasoro_evaluate_q_report", default=None)


def _q_tail(family, h, hbar, z, zbar, h_ext, hbar_ext, *, c, order, sector, branch, basis):
    """The worst q-truncation report over the block's actual chiral factors."""
    from .qgate import point_q_tail_report, super_point_q_tail_report

    halves = {(h, z, tuple(h_ext))}
    if zbar is not None:
        halves.add((hbar, zbar, tuple(hbar_ext)))
        if basis in {"scalar_G", "parity_projected_B"}:
            # Parity reverses the points, keeping each weight paired with its
            # own external list. Scalar G has identical external lists.
            halves.update({(h, zbar, tuple(h_ext)), (hbar, z, tuple(hbar_ext))})
    worst = None
    for weight, x, ext in halves:
        if family == "virasoro":
            report = point_q_tail_report(h=weight, z=x, h_externals=ext, c=c, q_order=order)
        else:
            report = super_point_q_tail_report(family=family, h=weight, z=x, h_externals=ext, c=c, q_order=order,
                                               sector=sector, branch=branch if family == "ramond" else None)
        if report["status"] == "unavailable":
            continue
        rank = (report["status"] != "unconverged", report["digits"])
        if worst is None or rank < (worst["status"] != "unconverged", worst["digits"]):
            worst = report
    return worst


def evaluate_block_request(request: BlockRequest, *, q_gate: str = "warn") -> BlockValue:
    """Evaluate one point request at the precision declared by its policy.

    The numerical backends use mpmath's process-wide context.  Keeping the
    precision change in a work context makes the request self-contained while
    avoiding a surprising mutation of the caller's context.

    The value is the block truncated at the request's ``order``, computed at the working precision.  Virasoro, NS
    and Ramond values carry ``metadata["q_truncation_digits"]``, the measured accuracy of that truncation as an
    approximation to the untruncated block (raising the precision cannot fix a short order, and at large c the order
    needed grows with c).  Every value carries ``metadata["estimated_digits"]``, its accuracy as the block: the
    working precision less the measured conditioning loss (a nearby pole, z = 1, or cancellation between block
    components), and no more than
    ``q_truncation_digits`` (a torus one-point value: its ``verified_digits`` and ``q_truncation_digits``).
    ``q_gate``: ``"warn"`` (default) prints a warning when the q-truncation leaves fewer than 20 digits (fewer than
    ``target_digits`` when the policy sets it) or the series has not converged, ``"strict"`` raises
    :class:`~virasoro_blocks.qgate.QConvergenceError` there, and ``"off"`` does neither.
    """
    from .qgate import DEFAULT_MIN_DIGITS, act_on_q_report

    if q_gate not in ("off", "warn", "strict"):
        raise RequestValidationError(f"q_gate must be off, warn, or strict; got {q_gate!r}")
    from .conditioning import capture

    token = _Q_REPORT.set(None)
    try:
        precision = int(request.policy.precision)
        target = request.policy.target_digits
        if target is None or request.family == "torus-onepoint":
            # (the torus one-point value escalates against its deeper reference itself)
            with mp.workdps(precision), capture(silent=False) as state:
                result = _evaluate_block_request(request)
            work = precision
        else:
            cap = request.policy.max_precision
            cap = max(4 * int(target), 2 * precision, int(target) + 200) if cap is None else int(cap)
            result, work, state = _evaluate_to_target(request, precision, int(target), cap)
        report = _Q_REPORT.get()
        # the truncation must leave the digits asked for: 20 by default, target_digits when given
        act_on_q_report(report, mode=q_gate, label="evaluate",
                        min_digits=DEFAULT_MIN_DIGITS if target is None else float(target))
    finally:
        _Q_REPORT.reset(token)
    metadata = dict(result.metadata)
    # digits of the order-truncated block the working precision leaves after the measured conditioning loss
    # (cancellation near a pole or z = 1, including that in the final block composition)
    conditioned = max(0.0, work - state["loss"])
    if request.family == "torus-onepoint":
        # verified against a deeper, higher-precision reference; its order tail measured separately
        measured = [metadata.get(key) for key in ("verified_digits", "q_truncation_digits")]
        measured = [float(value) for value in measured if value is not None]
        estimated = min(measured) if measured else conditioned
    else:
        estimated = conditioned
        if report is not None and report.get("status") == "unconverged":
            estimated = 0.0             # (the q-series has not converged: no digit the truncation can vouch for)
        elif report is not None and report.get("digits") is not None:
            truncation = min(float(report["digits"]), float(work))
            metadata["q_truncation_digits"] = round(truncation, 1)
            estimated = min(estimated, truncation)
    # the accuracy of the value as the block (not merely as its --order truncation)
    metadata["estimated_digits"] = round(estimated, 1)
    if (state["loss"] > 0 or work != precision) and request.family != "torus-onepoint":
        metadata["precision_actual"] = work
    return BlockValue(value=result.value, metadata=metadata)


def _evaluate_to_target(request, precision, target, max_precision):
    """``target_digits`` for a point value: evaluate at ``P``, measure the conditioning loss ``L`` (a nearby pole,
    z = 1, or cancellation between block components), and evaluate again at ``target + L`` until ``P - L`` delivers
    the target. An input closer to a pole than
    a precision resolves reads as a loss of about that precision, so the measurement can grow; a shortfall that does
    not shrink, or ``max_precision``, stops with :class:`~virasoro_blocks.verified_build.TargetDigitsError`."""
    import math

    from .conditioning import InsufficientPrecisionError, capture
    from .verified_build import TargetDigitsError

    work, previous = max(precision, target), None
    while True:
        with mp.workdps(work), capture(silent=True) as state:
            try:
                result = _evaluate_block_request(request)
            except BlockPoleError:
                # on a pole to roundoff at this precision: the offset (if any) is below it, a loss of ~work digits;
                # a genuine pole stays one at every precision and stops the escalation below
                state["loss"], result = max(state["loss"], float(work)), None
            except (ArithmeticError, InsufficientPrecisionError):
                if state["loss"] < work - 1:
                    raise
                result = None           # no digit at this precision: the measured loss sets the next one
        need = max(precision, int(math.ceil(target + state["loss"])))
        if need <= work and result is not None:
            return result, work, state
        if need > max_precision:
            raise TargetDigitsError(
                f"target_digits {target} needs precision {need} here (~{state['loss']:.0f} digits lost to "
                f"cancellation), above max_precision {max_precision}; raise max_precision")
        if previous is not None and need <= previous:
            raise TargetDigitsError(
                f"target_digits {target}: the measured loss stopped shrinking at precision {work} "
                f"(~{state['loss']:.0f} digits); the evaluation cannot resolve the cancellation")
        previous, work = work, max(need, work + 1)
