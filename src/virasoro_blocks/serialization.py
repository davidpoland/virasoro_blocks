"""Common serialized representation of block derivative tables.

`DeltaDerivTable` (the `--npoles all` exact path) and
`InterpolatedDeltaTable` (an integer or omitted `--npoles`) are two algorithms that
produce the *same* object: per `(coordinate, m, n)` a block derivative

    block_deriv(Delta) = base ** x * numerator_x(x) / prod_j (x - pole_x_j),

written in the shifted variable x = Delta - shift. This module converts either
table type to a `SerializedDerivTable`, which the Mathematica `.m` and
JSON writers consume. The two methods differ only in the *degree* of
`numerator_x` and in how many poles sit in `pole_list_x`; the shape is identical.

The damped-rational prefactor is universal -- it depends on c, the spin ell and
the external weights, but not on the coordinate -- so every coordinate in one
file shares a single `shift`, `base` and `pole_list_x`; only the numerators
differ. Any scalar constant (e.g. base ** shift) is folded into `numerator_x`,
so the reconstruction formula above carries no separate constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from contextvars import ContextVar
from collections.abc import Mapping

import mpmath as mp
from mpmath.libmp import fzero as _fzero, from_float as _from_float

from .polynomial import Polynomial
from .polyderivs import (
    DeltaDerivTable,
    InterpolatedDeltaTable,
    _strip_removable_poles_from_exact_tables,
)
from .truncation import is_off_diagonal, kept_pairs, total_parity_phase
from .numeric_input import literal_components, literal_number, parse_number


_READ_NUMBER_TEXTS = ContextVar("serialized_read_number_texts", default=None)


def _source_number_key(value):
    """An exact stored value key, independent of precision and real promotion."""
    if isinstance(value, mp.mpf):
        return value._mpf_
    if isinstance(value, mp.mpc):
        components = value._mpc_
    elif isinstance(value, complex):
        components = _from_float(value.real), _from_float(value.imag)
    else:
        return None
    # Reuse the tuple the number already owns. Real-to-complex promotion
    # still has the same key, without an extra pair allocation per real datum.
    return components[0] if components[1] == _fzero else components


def _canonicalize_output_scalar(x):
    """Collapse exactly real data without discarding an exponential phase."""
    value = parse_number(x)
    if isinstance(value, mp.mpc):
        return value.real if value.imag == 0 else value
    return value if isinstance(value, mp.mpf) else mp.convert(value)


def _canonical_pole_sort_key(pole):
    pole = mp.mpc(pole)
    imag = mp.im(pole)
    sign_key = 0 if imag > 0 else 1 if imag < 0 else 2
    return (mp.re(pole), abs(imag), sign_key)


def _canonicalize_pole_list(poles) -> tuple:
    from .interpolation import _canonicalize_prefactor_poles

    cleaned = _canonicalize_prefactor_poles(poles,
        real_tol=mp.mpf(10) ** (-max(5, mp.mp.dps - 5)))
    cleaned = [pole.real if pole.imag == 0 else pole for pole in cleaned]
    return tuple(sorted(cleaned, key=_canonical_pole_sort_key))


@dataclass
class SerializedDerivTable:
    """Block derivative tables for one or more coordinates in the common
    shifted-variable form, ready for the JSON and Mathematica writers.

        block_deriv[coordinate][m][n](Delta)
            = base ** x * numerators[coordinate][m][n](x) / prod_j (x - pole)

    with x = Delta - shift.

    `block_deriv[coordinate][m][n](Delta)` is the Taylor coefficient

        (1 / (m! n!)) * d^{m+n} B / dz^m dzbar^n  at z = zbar = 1/2

    (and analogously for the other coordinates around their symmetric
    points), not the raw partial derivative. The `1/(m! n!)` is folded into
    the stored entry, matching the `blocks_3d` / `scalar_blocks` convention.
    Practical consequence: the Leibniz rule for a product `P(z, zbar) *
    B(z, zbar)` reduces to a plain 2D convolution of Taylor coefficients
    with no binomial weights, which is what `virasoro_bootstrap.crossing.
    apply_prefactor_to_table` relies on.

    For `chiral_F` the table has only the `m` axis and the Taylor-coefficient
    prefactor is `1/m!`.
    """

    shift: object
    base: object
    pole_list_x: tuple
    coordinates: tuple
    numerators: dict  # coordinate -> list[list[Polynomial]] in x
    sample_points_x: tuple | None = None  # interpolation sample points in x = Delta - shift
    # Values of the stored interpolant at the writer nodes: coordinate ->
    # [node k][m][n], at Delta = x_k + shift. Assembled from recursion samples
    # or each entry's Lagrange interpolant, with split principal parts put
    # back; never re-evaluated from monomial coefficients. The SDP writer
    # uses these values (divide by sampleScaling, select F-pairs). None for
    # tables with no sample nodes (--npoles all) or a fit that recorded none.
    sample_block_values: dict | None = None  # in memory: coordinate -> [node k][m][n]; the JSON nests [m][n][k]
    sample_via: str | None = None
    # Optional family/provenance metadata.  Ordinary Virasoro tables leave this
    # unset; the NS/Ramond adapters use it so JSON and Mathematica writers can
    # reproduce the same interpretation without a parallel table schema.
    metadata: dict | None = None
    # These are part of the table, so dataclasses.replace (used by coordinate
    # relabeling) must retain them along with the coefficients.
    working_dps: int | None = None
    verified_digits: float | None = None
    verified_digits_largest_entry: float | None = None
    target_digits: int | None = None
    verified_range: tuple | None = None
    q_truncation_digits: float | None = None
    # The scalar-G spin-0 normalization the table was built under (see
    # virasoro_blocks._conventions), recorded when it is serialized (or read from the file), so a writer tags the
    # table with the convention that produced it rather than whichever is active when it is written.  None: not
    # recorded (a table assembled by hand); the writer then uses the active convention.
    scalar_G_spin0: str | None = field(default=None, compare=False)
    # Owned read provenance: the text each loaded payload number was read from,
    # so an unchanged table is written again digit for digit. A changed
    # numerical value no longer matches its original text key.
    _source_number_texts: dict | None = field(default=None, repr=False, compare=False)

    def _prefactor(self, x):
        denominator = mp.mpc(1)
        for pole in self.pole_list_x:
            denominator *= x - pole
        return self.base**x / denominator

    def _is_real(self, coordinate: str) -> bool:
        """Whether the table is real on the real axis: real shift, base and numerator coefficients, and poles that
        are real or come in complex-conjugate pairs (to the working precision, as the canonical pole list stores
        them)."""
        if any(literal_components(value)[1] != 0 for value in (self.shift, self.base)):
            return False
        for row in self.numerators[coordinate]:
            for poly in row:
                if getattr(poly, "field", None) is mp.mpc and any(mp.im(coeff) != 0 for coeff in poly.coeffs):
                    return False
        poles = [mp.mpc(mp.convert(real), mp.convert(imaginary))
                 for real, imaginary in (literal_components(pole) for pole in self.pole_list_x)]
        unpaired = [pole for pole in poles if mp.im(pole) != 0]
        while unpaired:
            pole = unpaired.pop()
            tolerance = mp.mpf(10) ** (5 - mp.mp.dps) * max(1, abs(pole))
            partner = next((i for i, other in enumerate(unpaired) if abs(other - mp.conj(pole)) <= tolerance), None)
            if partner is None:
                return False
            unpaired.pop(partner)
        return True

    def evaluate(self, coordinate: str, Delta):
        """Reconstruct the numeric (m, n) derivative table at this Delta.

        The monomial-form numerators cancel on evaluation (~0.6-0.7 digits per unit of degree), which is why a table
        is built and written at a working precision above its requested digits (``working_dps``, the JSON's
        ``precision_actual``).  It is evaluated at that precision, whatever the caller's, and the entries rounded to
        the caller's: a 50-digit table evaluated at the default 15 digits was otherwise off by 2e-4.

        A real table (see :meth:`_is_real`) at a real ``Delta`` returns real ``mpf`` entries: the complex
        arithmetic of the conjugate-pair prefactor leaves only a rounding-level imaginary part, which is dropped.
        Otherwise the entries are ``mpc``.
        """
        Delta = literal_number(Delta)
        caller = mp.mp.dps
        work = max(caller, int(self.working_dps or 0))
        with mp.workdps(work):
            x = mp.mpc(Delta) - mp.mpc(self.shift)
            pref = self._prefactor(x)
            rows = [
                [pref * poly.evaluate(x) for poly in row]
                for row in self.numerators[coordinate]
            ]
            real = mp.im(x) == 0 and self._is_real(coordinate)
        if real:
            return [[+mp.re(value) for value in row] for row in rows]     # (rounded to the caller's precision)
        if work <= caller:
            return rows
        return [[+value for value in row] for row in rows]      # rounded to the caller's precision


def _real_serialization_number(value, name):
    try:
        real, imaginary = literal_components(value)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError(f"{name} must be finite and real") from exc
    if imaginary != 0:
        raise ValueError(f"{name} must be finite and real")
    return real


def _default_shift_for_spin(spin) -> mp.mpf:
    """Bootstrap-natural shift convention.

    Returns the unitarity bound `|spin|` for the 2d bases (scalar G, ordered B,
    parity-projected B) and `0` for the chiral basis (spin is None there).
    This makes the polynomial variable `x = Delta - shift` start at `x = 0`
    on the SDPB positivity interval `[Delta_unitarity, ...)`.
    """
    if spin is None:
        return mp.mpf(0)
    return abs(_real_serialization_number(spin, "spin"))


def _default_exact_shift(first) -> mp.mpf:
    """Bootstrap-natural exact-serialization shift convention.

    Serialization keeps the canonical polynomial variable anchored at the
    bootstrap unitarity bound `|spin|` (or `0` for chiral tables).
    """
    return _default_shift_for_spin(first.spin)


def _serialize_exact(tables_by_coordinate: dict, shift) -> SerializedDerivTable:
    from .polyderivs import _align_interpolation_tables

    # Removable-pole stripping is needed only for tables that may carry
    # assembled removable poles -- the ε-extrapolated singular-locus tables
    # flagged by `regularize_delta_table`. Generic-c tables are already
    # minimal (build-time K != 0 gate), so the precision-fragile numeric strip
    # is skipped to avoid its high-q / low-dps false positives.
    if any(
        getattr(t, "requires_pole_strip", False)
        for t in tables_by_coordinate.values()
    ):
        # Stripping uses root indices too, so align before removing common factors.
        tables_by_coordinate = _align_interpolation_tables(tables_by_coordinate)
        tables_by_coordinate = _strip_removable_poles_from_exact_tables(
            tables_by_coordinate
        )
    coordinates = tuple(tables_by_coordinate)
    first = tables_by_coordinate[coordinates[0]]
    base = first.exp_base
    if not mp.isfinite(base):
        raise ValueError("coordinate tables require a finite exponential base")
    roots = first.denominator_roots
    tol = mp.mpf(10) ** (-max(5, mp.mp.dps - 10))
    for coord in coordinates[1:]:
        table = tables_by_coordinate[coord]
        if table.spin != first.spin:
            raise ValueError("coordinate tables disagree on the spin")
        other_base = mp.mpc(table.exp_base)
        if not (mp.isfinite(base) and mp.isfinite(other_base)) or abs(other_base - mp.mpc(base)) > tol:
            raise ValueError(
                "coordinate tables disagree on the exponential base; they must "
                "come from the same (c, ell, externals, q_order)"
            )
        if len(table.denominator_roots) != len(roots):
            raise ValueError("coordinate tables disagree on the pole count")
        if not _same_pole_multiset(roots, table.denominator_roots, tol):
            raise ValueError("coordinate tables disagree on the poles")

    if shift is None:
        shift = _default_exact_shift(first)
    shift = _real_serialization_number(shift, "shift")
    # The canonical form's denominator is the bare prod (x - pole_x), but the
    # symbolic builders' `denominator` is non-monic (leading coefficient
    # (1/2)^(#poles) from the h = (Delta +- ell)/2 substitution). That constant
    # is folded into the numerator so prod (x - pole_x) can be used directly.
    pole_list_x = _canonicalize_pole_list(
        mp.mpc(root) - shift for root in roots
    )

    numerators = {}
    for coord in coordinates:
        table = tables_by_coordinate[coord]
        numerator_scale = mp.mpc(base) ** shift / table.denominator.coeffs[-1]
        numerators[coord] = [
            [
                poly.compose_linear(1, shift).scale(numerator_scale)
                for poly in row
            ]
            for row in table.numerators
        ]
    return SerializedDerivTable(
        shift=shift,
        base=_canonicalize_output_scalar(base),
        pole_list_x=pole_list_x,
        coordinates=coordinates,
        numerators=numerators,
    )


def _serialize_interpolated(tables_by_coordinate: dict) -> SerializedDerivTable:
    coordinates = tuple(tables_by_coordinate)
    first = tables_by_coordinate[coordinates[0]]
    shift = _real_serialization_number(first.shift, "shift")
    prefactor = first.prefactor
    tol = mp.mpf(10) ** (-max(5, mp.mp.dps - 10))
    for coord in coordinates[1:]:
        table = tables_by_coordinate[coord]
        if _real_serialization_number(table.shift, "shift") != shift:
            raise ValueError("coordinate tables disagree on the shift")
        # Compare the prefactor component-wise within tolerance rather than by
        # exact equality: the exponential base is recomputed through each
        # coordinate's chiral transform, so it agrees only up to roundoff. The
        # constant is base**shift and so follows from base + shift. (The exact
        # path, _serialize_exact, likewise uses a tolerance on the base.)
        other = table.prefactor
        same_prefactor = (
            abs(mp.mpf(other.base) - mp.mpf(prefactor.base)) <= tol
            and _same_pole_multiset(prefactor.poles, other.poles, tol)
        )
        if not same_prefactor:
            raise ValueError(
                "coordinate tables disagree on the interpolation prefactor"
            )

    numerators = {}
    for coord in coordinates:
        table = tables_by_coordinate[coord]
        numerators[coord] = [
            [poly.scale(table.prefactor.constant) for poly in row]
            for row in table.numerators
        ]
    sample_points_x = None
    if first.sample_points:
        sample_points_x = tuple(_real_serialization_number(point, "sample point") - shift
                                for point in first.sample_points)
    def table_metadata(table):
        metadata = getattr(table, "metadata", None)
        return metadata if metadata is not None else getattr(table, "super_metadata", None)

    metadata = table_metadata(first)
    for coord in coordinates[1:]:
        if table_metadata(tables_by_coordinate[coord]) != metadata:
            raise ValueError("coordinate tables disagree on super-block family metadata")
    if metadata is not None:
        metadata = dict(metadata)
    return SerializedDerivTable(
        shift=shift,
        base=_canonicalize_output_scalar(prefactor.base),
        pole_list_x=_canonicalize_pole_list(prefactor.poles),
        coordinates=coordinates,
        numerators=numerators,
        sample_points_x=sample_points_x,
        metadata=metadata,
    )


def serialize_deriv_tables(tables_by_coordinate: dict, *, shift=None):
    """Collapse a coordinate -> table mapping into a `SerializedDerivTable`.

    Every value must be the same table type: all `DeltaDerivTable` (the
    `--npoles all` path, every Delta-pole kept) or all
    `InterpolatedDeltaTable` (an integer or omitted `--npoles`; the poles are
    chosen by `pole_selection`). `shift` applies to the exact path only; the interpolated
    tables already carry their own shift.
    """
    if not tables_by_coordinate:
        raise ValueError("tables_by_coordinate must not be empty")
    types = {type(table) for table in tables_by_coordinate.values()}
    if len(types) != 1:
        raise ValueError("all coordinate tables must be the same type")
    table_type = types.pop()
    if table_type is DeltaDerivTable:
        if shift is None:
            shift = _default_exact_shift(next(iter(tables_by_coordinate.values())))
    elif table_type is InterpolatedDeltaTable:
        if shift is not None:
            raise ValueError(
                "shift is determined by the interpolated tables; do not pass it"
            )
        shift = next(iter(tables_by_coordinate.values())).shift
    else:
        raise TypeError(
            "tables must be DeltaDerivTable or InterpolatedDeltaTable instances"
        )
    shift = _real_serialization_number(shift, "shift")
    if table_type is DeltaDerivTable:
        result = _serialize_exact(tables_by_coordinate, shift)
    else:
        result = _serialize_interpolated(tables_by_coordinate)
    result.working_dps = mp.mp.dps
    from ._conventions import scalar_G_spin0_convention

    result.scalar_G_spin0 = scalar_G_spin0_convention()
    return result


# ---------------------------------------------------------------------------
# JSON -> SerializedDerivTable loader (inverse of writers.write_json)
# ---------------------------------------------------------------------------

_CHIRAL_BASES = {
    "chiral_F", "global_chiral_F", "ns_chiral_F", "ramond_chiral_F",
}


def _parse_serialized_number(text, *, payload=False):
    """Parse one numeric string emitted by ``writers._num_str``.

    Real values are plain decimals (returned as ``mp.mpf``); genuinely complex
    values use the ``"(re+imj)"`` / ``"(re-imj)"`` form (returned as
    ``mp.mpc``). Parsing goes through mpmath so the full serialized precision is
    preserved -- Python's built-in ``complex`` would truncate to double.
    The value is rounded to the working precision. With ``payload``, the
    text of a block-file payload number (a coefficient, base, pole, node value
    or range endpoint) is recorded, so an unchanged table is written again
    digit for digit.
    """
    t = str(text).strip()
    original = t
    if t.startswith("(") and t.endswith(")"):
        t = t[1:-1]

    component = mp.mpf
    if t.endswith("j"):
        body = t[:-1]
        split = None
        for i in range(1, len(body)):
            if body[i] in "+-" and body[i - 1] not in "eE":
                split = i
        if split is None:
            real, imaginary = mp.mpf(0), component(body)
        else:
            real, imaginary = component(body[:split]), component(body[split:])
        value = mp.make_mpc((real._mpf_, imaginary._mpf_))
    else:
        value = component(t)
    if not mp.isfinite(value):
        raise ValueError(f"block file: non-finite numeric value {text!r}")
    sources = _READ_NUMBER_TEXTS.get()
    if payload and sources is not None:
        sources.setdefault(_source_number_key(value), original)
    return value


def _polynomial_from_coeff_strings(coeff_strings) -> Polynomial:
    if not isinstance(coeff_strings, (list, tuple)):
        raise ValueError("block file: polynomial coefficients must be an array")
    return Polynomial([_parse_serialized_number(s, payload=True) for s in coeff_strings])


def _parse_serialized_real_label(text, name):
    try:
        return _real_serialization_number(parse_number(text), name)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError):
        # Keep the block-file parser's established error for malformed and
        # non-finite values, while accepting exact rational labels directly.
        _parse_serialized_number(text)
        raise


def _same_pole_multiset(left, right, tol) -> bool:
    """The same poles (with multiplicity) within ``tol``, in any order."""
    from .interpolation import _as_complex_input, _pole_geometry_scales, _same_geometry_pole

    if len(left) != len(right):
        return False
    remaining = [_as_complex_input(p) for p in right]
    left_scales = right_scales = None
    for pole in left:
        pole = _as_complex_input(pole)
        index = next((i for i, other in enumerate(remaining) if pole == other), None)
        if index is None:
            if left_scales is None:
                left_scales = _pole_geometry_scales(left)
                right_scales = _pole_geometry_scales(right)
            candidates = [i for i, other in enumerate(remaining) if _same_geometry_pole(
                pole, other, tol, scale=min(left_scales[pole], right_scales[other]),
            )]
            index = min(candidates, key=lambda i: abs(pole-remaining[i])) if candidates else None
        if index is None:
            return False
        remaining.pop(index)
    return True


def _row_lengths(pairs) -> list:
    """The jagged row lengths (entries per m, in increasing m) of a kept-pair set."""
    counts: dict = {}
    for m, _n in pairs:
        counts[m] = counts.get(m, 0) + 1
    return [counts[m] for m in sorted(counts)]


_REQUIRED_KEYS = ("coordinates", "lambda", "exp_base", "variable_minus_x", "pole_list_x", "derivs")


def _file_integer(doc, key, minimum):
    value = doc[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"block file: {key} must be an integer >= {minimum}")
    return value


def _file_array(value, name):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"block file: {name} must be an array")
    return value


def serialized_table_from_json(doc: dict) -> SerializedDerivTable:
    """Reconstruct a :class:`SerializedDerivTable` from a ``write_json`` document.

    This restores the numerical payload of :func:`virasoro_blocks.writers.write_json`. It reads
    ``exp_base``, ``variable_minus_x``, ``pole_list_x``, ``coordinates`` and the
    jagged ``derivs`` payload back into the dense
    ``coordinate -> list[list[Polynomial]]`` form, restoring every
    high-precision value and guard digits from its decimal string. Reported
    working precision follows ``precision_actual`` (or the caller's precision
    when it is absent); evaluation also borrows the stored coefficients'
    precision when needed (:meth:`SerializedDerivTable.evaluate`).

    The writer only emits the kept ``(m, n)`` pairs for each coordinate, so the
    reconstructed numerator grid retains the original lambda along its m axis,
    zero-padding entries outside the keep rule. This preserves the cutoff when
    an odd-parity folded table is written again. Block values
    at the kept pairs round-trip to the serialized precision; the keep rule is
    recomputed from ``lambda``/``basis``/``spin`` exactly as the writer applies
    it, so write and read agree by construction.

    When the document carries ``sample_points_x`` (the interpolation method
    persists them), they are restored onto
    :attr:`SerializedDerivTable.sample_points_x` so the explicit-sample-point
    PMP path survives the round trip.

    When present, ``sample_block_values`` and ``sample_via`` are likewise
    restored so block-file provenance preserves both the clean raw node values
    and how those node values were obtained.

    The ``variable`` field (``"Delta"`` for fixed-spin tables, ``"L"`` for
    large-spin tables) is metadata about how a consumer interprets the shifted
    variable; it does not change the table payload and is not stored on
    :class:`SerializedDerivTable`. Callers that need it should read
    ``doc.get("variable", "Delta")`` directly.

    This object is not a complete copy of the file's input metadata. To write
    it again, pass the original physical context to the writer: central charge,
    external weights, basis/parities, spin or twist, normalization, coordinate
    centers and kinematic shifts, and numerical controls. In particular,
    omitting off-diagonal centers makes the writer apply its default diagonal
    derivative keep rule and can discard entries. Keep the original document
    alongside the reconstructed table when a faithful rewrite is needed.
    """
    if not isinstance(doc, Mapping):
        raise ValueError("block file must be a JSON object")
    missing = [key for key in _REQUIRED_KEYS if key not in doc]
    if missing:
        raise ValueError(f"block file is missing {', '.join(repr(k) for k in missing)} (not a write_json table?)")
    for key in ("coordinates", "derivs", "pole_list_x", "sample_points_x",
                "verified_delta_range", "verified_spin_range"):
        if key not in ("coordinates", "derivs", "pole_list_x") and doc.get(key) is None:
            continue
        _file_array(doc[key], key)
    for key in ("basis", "parity_projection"):
        if doc.get(key) is not None and not isinstance(doc[key], str):
            raise ValueError(f"block file: {key} must be a string")
    if doc.get("parity_external") is not None:
        _file_array(doc["parity_external"], "parity_external")
    _file_integer(doc, "lambda", 0)
    for key in ("precision_actual", "precision_input", "target_digits"):
        if doc.get(key) is not None:
            _file_integer(doc, key, 1)
    written = doc.get("precision_actual") or doc.get("precision_input")
    work = max(mp.mp.dps, int(written)) if written is not None else mp.mp.dps
    sources = {}
    token = _READ_NUMBER_TEXTS.set(sources)
    try:
        with mp.workdps(work):
            table = _restore_serialized_payload(doc)
    finally:
        _READ_NUMBER_TEXTS.reset(token)
    table._source_number_texts = sources
    # Files without an explicit precision were parsed at the caller's chosen
    # precision. Retain that too, rather than losing it when the context exits.
    table.working_dps = int(written) if written is not None else work
    # what the build verified (verified_build.record_verification), kept so a re-emitted file records it again
    for key in ("verified_digits", "verified_digits_largest_entry"):
        if doc.get(key) is not None:
            setattr(table, key, float(doc[key]))
    if doc.get("target_digits") is not None:
        table.target_digits = int(doc["target_digits"])
    return table


def _restore_serialized_payload(doc):
    table = _serialized_table_from_json(doc)
    range_key = "verified_spin_range" if doc.get("variable") == "L" else "verified_delta_range"
    if doc.get(range_key) is not None:
        table.verified_range = tuple(_parse_serialized_number(x, payload=True) for x in doc[range_key])
    # Torus writers store these diagnostics in metadata; sphere writers
    # use attributes. Restore both representations for a faithful rewrite.
    diagnostics = {}
    for key in ("fit_digits", "precision_digits", "q_truncation_digits"):
        if doc.get(key) is not None:
            diagnostics[key] = float(doc[key])
    if doc.get("verified_delta_range") is not None:
        diagnostics["verified_delta_range"] = tuple(
            _parse_serialized_number(x, payload=True) for x in doc["verified_delta_range"]
        )
    if diagnostics:
        table.metadata = {**(table.metadata or {}), **diagnostics}
    if doc.get("q_truncation_digits") is not None:
        table.q_truncation_digits = float(doc["q_truncation_digits"])
    return table


def _serialized_table_from_json(doc: dict) -> SerializedDerivTable:
    coordinates = tuple(doc["coordinates"])
    if any(not isinstance(coord, str) for coord in coordinates):
        raise ValueError("block file: coordinates must contain string names")
    if not coordinates or len(set(coordinates)) != len(coordinates):
        raise ValueError("block file: coordinates must be non-empty and unique")
    basis = doc.get("basis")
    # Use the writer's sphere-basis contract before interpreting the layout.
    # Import lazily because writers imports SerializedDerivTable from here.
    from .writers import _spin_json_value, _validate_basis_metadata

    if basis in {"torus_G", "torus_onept_G"}:
        if doc.get("parity_projection") is not None:
            raise ValueError("block file: parity_projection is not valid for a torus basis")
    else:
        basis = _validate_basis_metadata(basis, doc.get("parity_projection"))
    if str(basis).endswith("scalar_G"):
        # odd-spin scalar_G data must record that it is the z <-> zbar-even block
        from ._conventions import check_scalar_G_odd_record

        check_scalar_G_odd_record(doc)
    is_chiral = basis in _CHIRAL_BASES
    axes = ["coordinate_index", "deriv"] if is_chiral else ["coordinate_index", "deriv_0", "deriv_1"]
    coefficient_axes = ["coordinate", "m", "n"] if basis in {"torus_G", "torus_onept_G"} else axes
    for key, expected in (("index_names", coefficient_axes + ["monomial_degree"]),
                          ("sample_block_values_index_names", axes + ["sample_point"])):
        if key in doc and doc[key] != expected:
            raise ValueError(f"block file: {key} must be {expected!r}")
    lam = doc["lambda"]
    spin = doc.get("spin")
    ell = 0 if spin is None else _spin_json_value(spin)
    parity_projection = doc.get("parity_projection")

    base = _parse_serialized_number(doc["exp_base"], payload=True)
    shift = _parse_serialized_real_label(doc["variable_minus_x"], "variable_minus_x")
    pole_list_x = tuple(_parse_serialized_number(p, payload=True) for p in doc["pole_list_x"])

    derivs = doc["derivs"]
    if len(derivs) != len(coordinates):
        raise ValueError(f"block file: derivs has {len(derivs)} coordinate blocks for coordinates {list(coordinates)}")
    numerators: dict = {}
    pairs_by_coord: dict = {}
    parity_phase = total_parity_phase(doc.get("parity_external"), doc.get("parity_exchanged"))
    for coord_index, coord in enumerate(coordinates):
        coord_derivs = _file_array(derivs[coord_index], f"derivs[{coord!r}]")
        pairs = kept_pairs(
            coord, lam, basis=basis, ell=ell, parity_projection=parity_projection, parity_phase=parity_phase,
            off_diagonal=is_off_diagonal(doc.get("z_center"), doc.get("zbar_center")),
        )
        row_lengths = (None if is_chiral else [
            len(_file_array(row, f"derivs[{coord!r}][{i}]")) for i, row in enumerate(coord_derivs)
        ])
        if not is_chiral and _row_lengths(pairs) != row_lengths:
            raise ValueError(
                f"block file: derivs[{coord!r}] has rows of lengths {row_lengths}, but "
                f"lambda {lam}, basis {basis!r}, spin {spin} keep {_row_lengths(pairs)}; "
                "regenerate the table (incomplete or obsolete derivative layout)"
            )
        elif is_chiral and len(coord_derivs) != len(pairs):
            raise ValueError(f"block file: derivs[{coord!r}] has {len(coord_derivs)} entries, but lambda {lam} keeps "
                             f"{len(pairs)}")
        pairs_by_coord[coord] = pairs
        if is_chiral:
            # Collapsed one-axis layout: derivs[coord][k] is the polynomial for
            # pairs[k] = (m, 0).
            m_max = max(lam, max((m for m, _n in pairs), default=0))
            grid = [[Polynomial.zero()] for _ in range(m_max + 1)]
            for k, (m, _n) in enumerate(pairs):
                grid[m][0] = _polynomial_from_coeff_strings(coord_derivs[k])
        else:
            # Jagged grouped-by-m layout: derivs[coord] is indexed by the sorted
            # unique m values; within each m the entries run over increasing n.
            rows_by_m: dict[int, list[int]] = {}
            for (m, n) in pairs:
                rows_by_m.setdefault(m, []).append(n)
            sorted_m = sorted(rows_by_m)
            m_max = max(lam, max((m for m, _n in pairs), default=0))
            n_max = max((n for _m, n in pairs), default=0)
            grid = [
                [Polynomial.zero() for _ in range(n_max + 1)]
                for _ in range(m_max + 1)
            ]
            for row_index, m in enumerate(sorted_m):
                row = coord_derivs[row_index]
                for col_index, n in enumerate(rows_by_m[m]):
                    grid[m][n] = _polynomial_from_coeff_strings(row[col_index])
        numerators[coord] = grid

    sample_points_x = None
    if doc.get("sample_points_x") is not None:
        sample_points_x = tuple(_parse_serialized_real_label(point, "sample point")
                                for point in doc["sample_points_x"])

    sample_block_values = None
    if doc.get("sample_block_values") is not None:
        raw_sbv = doc["sample_block_values"]
        if not isinstance(raw_sbv, Mapping):
            raise ValueError("block file: sample_block_values must be a JSON object")
        if set(raw_sbv) - set(coordinates):
            raise ValueError("block file: sample_block_values contains unknown coordinates")
        if "sample_block_values_index_names" in doc:
            # Jagged kept-pairs layout (sample axis innermost), the same shape as
            # `derivs`. Recompute the kept pairs and reconstruct a dense,
            # zero-padded ``[k][m][n]`` grid -- mirroring the `derivs`
            # reconstruction above so the in-memory shape (consumed by the sdp
            # writer via F_minus_pairs) is unchanged.
            sample_block_values = {}
            for coord in coordinates:
                if coord not in raw_sbv:
                    continue
                coord_sbv = _file_array(raw_sbv[coord], f"sample_block_values[{coord!r}]")
                pairs = pairs_by_coord[coord]      # the layout the derivs were read with
                if is_chiral:
                    shape_matches = len(coord_sbv) == len(pairs)
                    leaves = coord_sbv
                else:
                    shape_matches = [
                        len(_file_array(row, f"sample_block_values[{coord!r}][{i}]"))
                        for i, row in enumerate(coord_sbv)
                    ] == _row_lengths(pairs)
                    leaves = [leaf for row in coord_sbv for leaf in row]
                if not shape_matches:
                    raise ValueError(f"block file: sample_block_values[{coord!r}] does not match the derivative layout")
                if any(not isinstance(leaf, (list, tuple)) for leaf in leaves):
                    raise ValueError(f"block file: sample_block_values[{coord!r}] must contain arrays of node values")
                n_samples = (len(sample_points_x) if sample_points_x is not None
                             else len(leaves[0]) if leaves else 0)
                if any(len(leaf) != n_samples for leaf in leaves):
                    raise ValueError(f"block file: sample_block_values[{coord!r}] must have {n_samples} node values for "
                                     "every derivative entry")
                if is_chiral:
                    m_max_c = max(lam, max((m for m, _n in pairs), default=0))
                    tables = [
                        [[_parse_serialized_number("0")] for _ in range(m_max_c + 1)]
                        for _ in range(n_samples)
                    ]
                    for m_idx, (m, _n) in enumerate(pairs):
                        for k in range(n_samples):
                            tables[k][m][0] = _parse_serialized_number(coord_sbv[m_idx][k], payload=True)
                else:
                    rows_by_m: dict[int, list[int]] = {}
                    for (m, n) in pairs:
                        rows_by_m.setdefault(m, []).append(n)
                    sorted_m = sorted(rows_by_m)
                    m_max_c = max(lam, max((m for m, _n in pairs), default=0))
                    n_max_c = max((n for _m, n in pairs), default=0)
                    tables = [
                        [
                            [_parse_serialized_number("0") for _ in range(n_max_c + 1)]
                            for _ in range(m_max_c + 1)
                        ]
                        for _ in range(n_samples)
                    ]
                    for row_index, m in enumerate(sorted_m):
                        row = coord_sbv[row_index]
                        for col_index, n in enumerate(rows_by_m[m]):
                            leaf = row[col_index]
                            for k in range(n_samples):
                                tables[k][m][n] = _parse_serialized_number(leaf[k], payload=True)
                if sample_points_x is not None and n_samples != len(sample_points_x):
                    raise ValueError(f"block file: sample_block_values[{coord!r}] has {n_samples} node values for "
                                     f"{len(sample_points_x)} sample_points_x")
                sample_block_values[coord] = tables
        else:
            # Legacy dense ``[coord][k][m][n]`` layout (no index-names marker).
            for coord, tables in raw_sbv.items():
                if not isinstance(tables, (list, tuple)) or any(
                    not isinstance(table, (list, tuple)) or any(
                        not isinstance(row, (list, tuple)) for row in table
                    ) for table in tables
                ):
                    raise ValueError(f"block file: sample_block_values[{coord!r}] must contain arrays of node values")
            sample_block_values = {
                coord: [
                    [
                        [_parse_serialized_number(value, payload=True) for value in row]
                        for row in table
                    ]
                    for table in tables
                ]
                for coord, tables in raw_sbv.items()
            }
            for coord, tables in sample_block_values.items():
                if sample_points_x is not None and len(tables) != len(sample_points_x):
                    raise ValueError(f"block file: sample_block_values[{coord!r}] has {len(tables)} node values for "
                                     f"{len(sample_points_x)} sample_points_x")
                if any(m >= len(table) or n >= len(table[m])
                       for table in tables for m, n in pairs_by_coord[coord]):
                    raise ValueError(f"block file: sample_block_values[{coord!r}] is missing derivative entries")

    metadata = {}
    for key in (
        "block_family", "super_sector", "ramond_branch",
        "central_charge_evaluated", "regularization_policy",
        "regularization_eps_exp", "collision_policy", "kac_limit",
    ):
        if key in doc:
            metadata[key] = doc[key]
    return SerializedDerivTable(
        shift=shift,
        base=base,
        pole_list_x=pole_list_x,
        coordinates=coordinates,
        numerators=numerators,
        sample_points_x=sample_points_x,
        sample_block_values=sample_block_values,
        sample_via=doc.get("sample_via"),
        metadata=metadata or None,
        scalar_G_spin0=doc.get("scalar_G_spin0"),
    )
