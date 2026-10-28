"""Output writers: JSON (blocks_3d schema) and Mathematica .m (scalar_blocks style).

Both consume a SerializedDerivTable produced by serialization.serialize_deriv_tables.
The JSON schema is Virasoro-adapted from blocks_3d; the .m format mirrors scalar_blocks.

Reconstruction formula (shared by both formats):
    block_deriv[coord][m][n](Delta) = base^x * numerator_x(x) / prod_j (x - pole_j)
with x = Delta - shift.  Coefficients are stored in ascending degree order (coeffs[k]
is the coefficient of x^k), which is the Polynomial class convention.
"""

from __future__ import annotations

import json
from fractions import Fraction
from functools import wraps
from contextvars import ContextVar

import mpmath as mp

from .serialization import SerializedDerivTable, _source_number_key
from .numeric_input import literal_components, literal_number, parse_number
from .truncation import is_off_diagonal, kept_pairs, parity_smoothing_label, total_parity_phase
from ._conventions import SCALAR_G_ODD_EVEN, SCALAR_G_ODD_KEY, SCALAR_G_SPIN0_KEY, scalar_G_spin0_convention

# The bases whose spin-0 table depends on the scalar-G normalization (written with SCALAR_G_SPIN0_KEY), and whose
# odd-spin tables are z <-> zbar even (written with SCALAR_G_ODD_KEY).
_SCALAR_G_BASES = frozenset({"scalar_G", "global_scalar_G", "ns_scalar_G", "ramond_scalar_G"})


def _table_scalar_G_spin0(serialized, requested=None, *, spin_zero=True) -> str:
    """The convention a scalar-G table is tagged with: the one it was built under, not the active one.

    ``requested`` (a caller that knows the build's convention, such as a bootstrap problem) wins; on a spin-0
    table, the only one whose values depend on the convention, it must agree with the table's own record.  A
    table that records none falls back to ``requested``, else to the active convention.
    """
    from ._conventions import _checked

    recorded = getattr(serialized, "scalar_G_spin0", None)
    if requested is not None:
        requested = _checked(requested)
        if spin_zero and recorded is not None and recorded != requested:
            raise ValueError(f"the table was built with scalar_G_spin0 = {recorded!r}, but the caller tags it "
                             f"{requested!r}")
        return requested
    return _checked(recorded) if recorded is not None else scalar_G_spin0_convention()


_WRITE_NUMBER_TEXTS = ContextVar("serialized_write_number_texts", default=None)


def _at_table_precision(writer):
    """Write a table at its working precision (``working_dps``, else the caller's).

    A table loaded from a file and not changed since is written again with the
    texts it was read from.
    """
    @wraps(writer)
    def wrapped(serialized, *args, **kwargs):
        for name in ("precision_input", "precision_actual"):
            if kwargs.get(name) is not None:
                kwargs[name] = _positive_integer(name, kwargs[name])
        stored = getattr(serialized, "working_dps", None)
        work = max(mp.mp.dps, int(stored or 0))
        sources = serialized._source_number_texts if kwargs.get("sig_figs") is None else None
        if stored is not None and kwargs.get("precision_actual") is None:
            kwargs["precision_actual"] = int(stored) if sources is not None else work
        token = _WRITE_NUMBER_TEXTS.set(sources)
        try:
            with mp.workdps(work):
                return writer(serialized, *args, **kwargs)
        finally:
            _WRITE_NUMBER_TEXTS.reset(token)
    return wrapped


def _sig_figs() -> int:
    # Coefficients benefit from a few guard digits so round-tripping through
    # text loses less precision than the ambient working dps.
    return mp.mp.dps + 5


def _positive_integer(name, value) -> int:
    from .requests import _integer_field

    value = _integer_field(name, value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _resolve_sig_figs(value) -> int:
    return _sig_figs() if value is None else _positive_integer("sig_figs", value)


def _external_weights(values, name):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must contain exactly four weights")
    try:
        values = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain exactly four weights") from exc
    if len(values) != 4:
        raise ValueError(f"{name} must contain exactly four weights")
    return [parse_number(value) for value in values]


def _metadata_sig_figs(sig_figs: int) -> int:
    # Metadata should not expose guard-digit noise from values that are meant
    # to read as simple decimal parameters or pole locations.
    return min(sig_figs, mp.mp.dps)


_NOISE_TOL: dict = {}


def _writer_noise_abs_tol() -> mp.mpf:
    # formed once per precision: rebuilding 10**-(dps-5) was a third of the per-number formatting cost
    prec = mp.mp.prec
    tol = _NOISE_TOL.get(prec)
    if tol is None:
        tol = _NOISE_TOL[prec] = mp.mpf(10) ** (-max(1, mp.mp.dps - 5))
    return tol


def _writer_guard_tol(x) -> mp.mpf:
    return _writer_noise_abs_tol() * max(abs(x), mp.mpf(1))


def _trim_guard_digit_noise(x, text: str) -> str:
    if "." not in text or "e" in text or "E" in text:
        return text

    sign = ""
    body = text
    if body.startswith(("-", "+")):
        sign, body = body[0], body[1:]
    whole, frac = body.split(".", 1)

    max_tail = min(6, len(frac) - 1)
    for tail_len in range(1, max_tail + 1):
        prefix = frac[:-tail_len]
        zero_run = len(prefix) - len(prefix.rstrip("0"))
        if zero_run < 12:
            continue
        trimmed_frac = prefix.rstrip("0") or "0"
        candidate = f"{sign}{whole}.{trimmed_frac}"
        if abs(mp.mpf(candidate) - mp.mpf(x)) <= _writer_noise_abs_tol():
            return candidate
    return text


def _nstr_clean(x, sig_figs: int) -> str:
    text = mp.nstr(x, sig_figs)
    try:
        text = _trim_guard_digit_noise(x, text)
        # a negative zero: the mantissa has digits and all are 0 (a digit scan -- parsing the string back cost as
        # much as nstr; "-inf" has no digits and stays negative)
        if text.startswith("-"):
            mantissa = text[1:].lower().split("e")[0]
            if any(ch.isdigit() for ch in mantissa) and not mantissa.strip("0."):
                return text[1:]
    except (TypeError, ValueError):
        pass
    return text


def _mma_real(text: str, precision: int | None = None) -> str:
    """Rewrite one mpmath decimal string as a Mathematica number literal.

    Mathematica reads ``1.04e-27`` as ``1.04 e - 27`` (a product with the
    symbol ``e``); its exponent marker is ``*^``. Non-finite values become the
    Mathematica symbols ``Indeterminate`` / ``Infinity``.  With ``precision`` a
    decimal carries Mathematica's precision mark (``-2.375`40``): a plain short
    literal is MachinePrecision there, and one such number (a shift ``1.0``, a
    pole ``-2.375 + 1.3635...*I``) collapsed every evaluation to 16 digits.
    """
    lowered = text.strip().lower()
    if lowered.lstrip("+-") == "nan":
        return "Indeterminate"
    if lowered.lstrip("+") == "inf":
        return "Infinity"
    if lowered == "-inf":
        return "-Infinity"
    mantissa, _, exponent = lowered.partition("e")
    if precision is not None and "." in mantissa:
        if not mantissa.strip("+-0."):
            return "0"
        mantissa = f"{mantissa}`{int(precision)}"
    if exponent:
        return f"{mantissa}*^{int(exponent)}"
    return mantissa if precision is not None and "." in lowered else text


def _c_eval_sig_figs(meta_sig_figs, eps_exp):
    """Digits for ``central_charge_evaluated``: enough to show the ``c + 10^-eps_exp`` offset (at the metadata
    digits 1.5 + 1e-61 was written as 1.5, indistinguishable from the singular c it avoids)."""
    return meta_sig_figs if eps_exp is None else max(meta_sig_figs, int(eps_exp) + 5)


def _metadata_num_str(x, sig_figs: int, *, mma: bool = False) -> str:
    """Preserve exact real labels; format computed values at metadata precision.

    JSON uses terminating decimals or p/q. Mathematica uses exact rational
    expressions, since a decimal literal there denotes an approximate number.
    """
    value = parse_number(x)
    if not isinstance(value, Fraction):
        return _num_str(value, sig_figs, mma=mma)
    numerator, denominator = value.numerator, value.denominator
    if mma:
        return str(numerator) if denominator == 1 else f"{numerator}/{denominator}"
    twos = fives = 0
    rest = denominator
    while rest % 2 == 0:
        rest //= 2
        twos += 1
    while rest % 5 == 0:
        rest //= 5
        fives += 1
    if rest != 1:
        return f"{numerator}/{denominator}"
    places = max(twos, fives)
    digits = str(abs(numerator) * 2 ** (places - twos) * 5 ** (places - fives))
    sign = "-" if numerator < 0 else ""
    if not places:
        return f"{sign}{digits}.0"
    digits = digits.zfill(places + 1)
    return f"{sign}{digits[:-places]}.{digits[-places:]}"


def _decimal_label_str(x, sig_figs: int, *, mma: bool = False) -> str:
    """Format approximate metadata to ``sig_figs`` decimal digits.

    With ``mma=True``, use Mathematica number syntax. Non-terminating rationals
    are rounded; readers compare these labels up to roundoff and also accept
    rational ``p/q`` spellings.
    """
    return _num_str(x, sig_figs, mma=mma)


def _input_metadata_num_str(x, sig_figs: int, *, mma: bool = False) -> str:
    """Format an input label, retaining every stored real and complex component."""
    value = literal_number(x)
    if isinstance(value, (complex, mp.mpc)):
        from .numeric_input import literal_components

        real, imaginary = literal_components(value)
        real_text = _metadata_num_str(real, sig_figs, mma=mma)
        if imaginary == 0:
            return real_text
        imaginary_text = _metadata_num_str(abs(imaginary), sig_figs, mma=mma)
        sign = '-' if imaginary < 0 else '+'
        if mma:
            return f"({real_text} {sign} {imaginary_text}*I)"
        return f"({real_text}{sign}{imaginary_text}j)"
    return _metadata_num_str(value, sig_figs, mma=mma)


def _num_str(x, sig_figs: int, *, mma: bool = False, preserve_data: bool = False) -> str:
    """Format a numeric value at the requested number of significant figures.

    ``preserve_data`` rounds to ``sig_figs`` without trimming small real
    offsets or nonzero imaginary components from numerical table data.
    For real numbers (or numbers with negligible imaginary part) returns a plain
    decimal.  For genuinely complex numbers returns "(re+imj)" or Mathematica
    "(re + im*I)" according to the *mma* flag. Exact real inputs bypass the
    guard-noise cleanup used for computed values.
    """
    if preserve_data:
        source = _source_num_str(x, sig_figs, mma=mma)
        if source is not None:
            return source
    fmt = (lambda text: _mma_real(text, sig_figs)) if mma else (lambda text: text)
    number_text = mp.nstr if preserve_data else _nstr_clean
    x = parse_number(x)
    if isinstance(x, Fraction):
        # Written offsets are input data, not guard-digit noise. Convert at
        # the formatting precision and bypass cleanup of computed roundoff.
        with mp.workdps(max(mp.mp.dps, sig_figs + 10)):
            return fmt(mp.nstr(mp.convert(x), sig_figs))
    x = mp.convert(x)
    if isinstance(x, mp.mpf):
        return fmt(number_text(x, sig_figs))
    if not isinstance(x, mp.mpc):
        x = mp.mpc(x)
    re, im = mp.re(x), mp.im(x)
    # ``im == 0`` first: a non-finite real part makes the tolerance NaN/inf.
    if im == 0 or (not preserve_data and mp.isfinite(re) and abs(im) < _writer_guard_tol(re)):
        return fmt(number_text(re, sig_figs))
    re_str = fmt(number_text(re, sig_figs))
    im_abs_str = fmt(number_text(abs(im), sig_figs))
    sign = "-" if im < 0 else "+"
    if mma:
        return f"({re_str} {sign} {im_abs_str}*I)"
    return f"({re_str}{sign}{im_abs_str}j)"


def _source_num_str(value, sig_figs, *, mma=False):
    sources = _WRITE_NUMBER_TEXTS.get()
    if sources is None:
        return None
    key = _source_number_key(value)
    original = sources.get(key)
    if original is None:
        return None
    if not mma:
        return original

    body = original.strip('()')
    if body.endswith('j'):
        body = body[:-1]
        split = next((index for index in range(len(body)-1, 0, -1)
                      if body[index] in '+-' and body[index-1] not in 'eE'), None)
        real, imaginary = ('0', body) if split is None else (body[:split], body[split:])
    else:
        real, imaginary = body, None

    def component(text):
        text = text.lstrip('+')
        mantissa, separator, exponent = text.lower().partition('e')
        if separator and '.' not in mantissa:
            text = f"{mantissa}.0e{exponent}"
        return _mma_real(text, sig_figs)

    real = component(real)
    if len(key) == 4:
        return real
    sign = '-' if key[1][0] else '+'
    imaginary = component(imaginary.lstrip('+-'))
    return f"({real} {sign} {imaginary}*I)"


def _range_num_str(value, sig_figs):
    return _num_str(parse_number(value), sig_figs, preserve_data=True)


def _label_float(x, sig_figs: int = 6) -> str:
    """Compact decimal for use in filenames (no trailing zeros, uses g-format)."""
    return f"{float(mp.re(mp.mpc(x))):.{sig_figs}g}"


_VALID_BASES = {
    "scalar_G", "ordered_B", "parity_projected_B", "chiral_F",
    "global_scalar_G", "global_ordered_B", "global_parity_projected_B",
    "global_chiral_F",
    "ns_scalar_G", "ns_ordered_B", "ns_parity_projected_B", "ns_chiral_F",
    "ramond_scalar_G", "ramond_ordered_B", "ramond_parity_projected_B",
    "ramond_chiral_F",
}


def _basis_label(basis: str | None, parity_projection: str | None) -> str:
    """Compact ASCII basis tag for filenames."""
    basis = _validate_basis_metadata(basis, parity_projection)
    if basis == "scalar_G":
        return ""
    if basis == "global_scalar_G":
        return "-basisGlobalScalarG"
    if basis == "global_ordered_B":
        return "-basisGlobalOrderedB"
    if basis == "global_chiral_F":
        return "-basisGlobalChiralF"
    if basis == "ns_scalar_G":
        return "-basisNSSScalarG"
    if basis == "ns_ordered_B":
        return "-basisNSOrderedB"
    if basis == "ns_chiral_F":
        return "-basisNSChiralF"
    if basis == "ramond_scalar_G":
        return "-basisRamondScalarG"
    if basis == "ramond_ordered_B":
        return "-basisRamondOrderedB"
    if basis == "ramond_chiral_F":
        return "-basisRamondChiralF"
    if basis == "ordered_B":
        return "-basisOrderedB"
    if basis == "chiral_F":
        return "-basisChiralF"
    if basis in {"parity_projected_B", "ns_parity_projected_B", "ramond_parity_projected_B"}:
        if parity_projection == "+":
            return "-basisParityPlus" if basis == "parity_projected_B" else f"-basis{basis.replace('_', '').title()}Plus"
        if parity_projection == "-":
            return "-basisParityMinus" if basis == "parity_projected_B" else f"-basis{basis.replace('_', '').title()}Minus"
    if basis == "global_parity_projected_B":
        if parity_projection == "+":
            return "-basisGlobalParityPlus"
        if parity_projection == "-":
            return "-basisGlobalParityMinus"
    raise ValueError(f"basis must be one of {sorted(_VALID_BASES)}")


def _validate_basis_metadata(basis: str | None, parity_projection: str | None) -> str:
    if basis is None:
        basis = "scalar_G"
    if basis not in _VALID_BASES:
        raise ValueError(f"basis must be one of {sorted(_VALID_BASES)}")
    if basis in {
        "parity_projected_B", "global_parity_projected_B",
        "ns_parity_projected_B", "ramond_parity_projected_B",
    }:
        if parity_projection not in {"+", "-"}:
            raise ValueError(
                f"{basis} outputs require parity_projection +/-"
            )
    elif parity_projection is not None:
        raise ValueError(
            "parity_projection is only valid for a parity_projected_B basis"
        )
    return basis


def _infer_half_integer_basis(basis, spin):
    """A half-integer spin has no scalar-G even/odd parity convention."""
    if basis is None and _spin_json_value(spin) % 1:
        return "ordered_B"
    return basis


def _mma_string_or_null(value) -> str:
    if value is None:
        return "Null"
    return json.dumps(value)


def _mma_string_list(values) -> str:
    if values is None:
        return "{}"
    return "{" + ", ".join(json.dumps(v) for v in values) + "}"


def _kept_pairs_for_writer(
    coord, numerators_row_count, numerators_col_count, *, basis, ell, parity_projection, off_diagonal=False,
    parity_phase=1,
):
    """Resolve kept pairs, requiring every entry implied by the cutoff."""
    lam_axis = max(numerators_row_count - 1, numerators_col_count - 1)
    pairs = kept_pairs(
        coord,
        lam_axis,
        basis=basis,
        ell=ell,
        parity_projection=parity_projection,
        off_diagonal=off_diagonal,
        parity_phase=parity_phase,
    )
    if any(m >= numerators_row_count or n >= numerators_col_count for m, n in pairs):
        raise ValueError(f"coordinate {coord!r} is missing derivative entries required by cutoff {lam_axis}")
    return pairs


def _external_spin_value(h, hb):
    left, right = literal_number(h), literal_number(hb)
    if isinstance(left, (complex, mp.mpc)) or isinstance(right, (complex, mp.mpc)):
        if literal_number(left.imag) != literal_number(right.imag):
            raise ValueError("spin values must be finite and real")
        return _external_spin_value(left.real, right.real)
    if isinstance(left, Fraction) and isinstance(right, Fraction):
        return _spin_json_value(left - right, allow_roundoff=isinstance(h, mp.mpf) or isinstance(hb, mp.mpf))
    return _spin_json_value(h - hb)


def _spin_json_value(x, *, allow_roundoff=False):
    """Return an exact JSON spin label; tolerate only explicitly computed roundoff."""
    import numpy as np

    if isinstance(x, (bool, np.bool_)):
        raise ValueError("spin values must be integers or half-integers, not booleans")
    try:
        # Read native components before working-precision conversion can
        # round a supplied offset onto the integer/half-integer lattice.
        real, imaginary = literal_components(x)
        if imaginary != 0:
            raise ValueError("spin values must be finite and real")
        twice = 2 * Fraction(real)
        twice_int = round(twice)
        offset = abs(twice - twice_int)
        if offset and (not allow_roundoff or
                       offset > Fraction(1, 10 ** max(10, mp.mp.dps // 2))):
            raise ValueError("spin values must be integer or half-integer in current JSON writer")
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError("spin values must be finite real integers or half-integers") from exc
    if twice_int % 2 == 0:
        return twice_int // 2
    try:
        result = twice_int / 2
    except OverflowError as exc:
        raise ValueError("half-integer spin is too large for exact JSON numeric representation") from exc
    if Fraction(result) != Fraction(twice_int, 2):
        raise ValueError("half-integer spin is too large for exact JSON numeric representation")
    return result


# ---------------------------------------------------------------------------
# Mathematica .m writer
# ---------------------------------------------------------------------------

def mathematica_filename(
    coordinate: str,
    *,
    c,
    h_ext,
    hbar_ext,
    spin: int,
    q_order: int | None,
    m_max: int = None,
    n_max: int = None,
    npoles: int = None,
    basis: str | None = None,
    parity_projection: str | None = None,
    chiral_norm=None,
    pole_selection: str | None = None,
    shared_sample_points: bool = False,
    parity_external=None,
    parity_exchanged=None,
    z_shift=None,
    zbar_shift=None,
    z_center=None,
    zbar_center=None,
    gap=None,
) -> str:
    """Filename for the Mathematica .m output file.

    Pattern::

        {coord}DerivTable[-c{c}]
            -delta1-{d1}-delta2-{d2}-delta3-{d3}-delta4-{d4}
            [-spin1-{s1}-spin2-{s2}-spin3-{s3}-spin4-{s4}]
            -L{ell}-lambda{Lam}[-npoles{K}][-poles{Selection}][-sharedNodes]
            [-basisX][-parityExt{p1}{p2}{p3}{p4}][-parityExch{p}]-chiralNorm{M}
            [-zShift{S}][-zbShift{S}][-zCenter{z0}][-zbCenter{zb0}][-gap{G}][-order{O}].m

    Every input that changes the table appears, so two tables that differ only in
    one of them never share a name.  The bracketed tags appear only away from their
    defaults (``pole_selection`` other than ``contribution_optimal``, which is passed
    only for an order-reduced table; external parities other than ``1 1 1 1``; an
    exchanged parity other than ``1``; nonzero shifts; an expansion center other than
    ``1/2``, and a ``zbar`` center other than the ``z`` one; a nonzero gap), so a
    default table keeps its short name.  A two-sided basis (``ordered_B``,
    ``parity_projected_B``) records the ``zbar`` shift separately; the symmetric and
    chiral bases carry one shift.  The working precision, the sampler choice and
    ``--target-digits`` change only the digits of the same table and do not appear.

    The Virasoro filenames include ``c`` and ``order`` because those inputs
    change the q-recursive block. The explicit global_* filenames omit ``c``
    but retain a supplied ``order``, which controls the finite SL(2) pole pool
    used for interpolation.
    """
    from .recursion import DEFAULT_CHIRAL_NORM
    chiral_norm_value = (
        DEFAULT_CHIRAL_NORM if chiral_norm is None else parse_number(chiral_norm)
    )
    basis = _infer_half_integer_basis(basis, spin)
    basis_resolved = _validate_basis_metadata(basis, parity_projection)
    is_chiral = basis_resolved in {
        "chiral_F", "global_chiral_F", "ns_chiral_F", "ramond_chiral_F",
    }
    is_global = basis_resolved.startswith("global_")
    if not is_chiral:
        spin = _spin_json_value(spin)
        parity_smoothing_label(basis_resolved, ell=spin, parity_projection=parity_projection)
    h_ext = _external_weights(h_ext, "h_ext")
    # Chiral output is labeled by the holomorphic externals h_i alone (the
    # antiholomorphic half is not in the file). The 2d label still records the
    # combined Delta_i = h_i + hbar_i and the spins s_i = h_i - hbar_i.
    if is_chiral:
        ext_label = "".join(
            f"-h{i+1}-{_label_float(mp.re(mp.mpc(h)))}"
            for i, h in enumerate(h_ext)
        )
        spin_tag = ""
    else:
        hbar_ext = _external_weights(hbar_ext, "hbar_ext")
        delta_list = [mp.re(mp.mpc(h) + mp.mpc(hb)) for h, hb in zip(h_ext, hbar_ext)]
        spin_list = [mp.re(mp.mpc(h) - mp.mpc(hb)) for h, hb in zip(h_ext, hbar_ext)]
        ext_label = "".join(
            f"-delta{i+1}-{_label_float(d)}" for i, d in enumerate(delta_list)
        )
        if any(abs(s) > mp.mpf("1e-30") for s in spin_list):
            ext_label += "".join(
                f"-spin{i+1}-{_label_float(s)}" for i, s in enumerate(spin_list)
            )
        spin_tag = f"-L{spin}"
    lam = max(m_max or 0, n_max or 0)
    npoles_label = f"-npoles{npoles}" if npoles is not None else ""
    basis_label = _basis_label(basis, parity_projection)
    chiral_norm_label = f"-chiralNorm{_label_float(chiral_norm_value)}"
    if not is_global and c is None:
        raise ValueError("non-global filenames require c")
    if not is_global and q_order is None:
        raise ValueError("non-global filenames require q_order")
    c_label = "" if is_global else f"-c{_label_float(c)}"
    # For global fixed-spin tables, q_order records the finite SL(2) pole
    # universe even though the 2F1 values themselves are evaluated exactly.
    order_label = f"-order{q_order}" if q_order is not None else ""
    two_sided = basis_resolved.endswith(("ordered_B", "parity_projected_B"))
    settings_label = _mathematica_settings_label(
        pole_selection=pole_selection, shared_sample_points=shared_sample_points,
    )
    parity_label = _mathematica_parity_label(parity_external, parity_exchanged)
    kinematics_label = _mathematica_kinematics_label(
        z_shift=z_shift, zbar_shift=zbar_shift if two_sided and not is_chiral else None,
        z_center=z_center, zbar_center=None if is_chiral else zbar_center, gap=gap,
    )
    return (
        f"{coordinate}DerivTable"
        f"{c_label}"
        f"{ext_label}"
        f"{spin_tag}"
        f"-lambda{lam}"
        f"{npoles_label}"
        f"{settings_label}"
        f"{basis_label}"
        f"{parity_label}"
        f"{chiral_norm_label}"
        f"{kinematics_label}"
        f"{order_label}.m"
    )


def _label_value(x) -> str:
    """``_label_float`` of a CLI literal, an exact label, or an mpmath number."""
    return _label_float(mp.convert(parse_number(x)))


def _mathematica_settings_label(*, pole_selection, shared_sample_points) -> str:
    """The interpolation settings that change an order-reduced table: a non-default pole selection, shared nodes."""
    label = ""
    if pole_selection is not None and pole_selection != "contribution_optimal":
        label += "-poles" + "".join(part.title() for part in str(pole_selection).split("_"))
    if shared_sample_points:
        label += "-sharedNodes"
    return label


def _mathematica_parity_label(parity_external, parity_exchanged) -> str:
    """External parities other than all ``1`` and an exchanged parity other than ``1`` (signed phases: ``+1``, ``-i``)."""
    from .parity import normalize_parity_phase

    def signed(phase):
        phase = normalize_parity_phase(str(phase))
        return phase if phase.startswith("-") else f"+{phase}"

    label = ""
    if parity_external is not None:
        phases = [signed(phase) for phase in parity_external]
        if any(phase != "+1" for phase in phases):
            label += "-parityExt" + "".join(phases)
    if parity_exchanged is not None and signed(parity_exchanged) != "+1":
        label += f"-parityExch{signed(parity_exchanged)}"
    return label


def _mathematica_kinematics_label(*, z_shift, zbar_shift, z_center, zbar_center, gap) -> str:
    """Nonzero shifts, an off-symmetric expansion center, and a nonzero gap (each only away from its default)."""
    def nonzero(x):
        return x is not None and literal_components(x) != (0, 0)

    label = ""
    if nonzero(z_shift):
        label += f"-zShift{_label_value(z_shift)}"
    if nonzero(zbar_shift):
        label += f"-zbShift{_label_value(zbar_shift)}"
    half = Fraction(1, 2)
    if z_center is not None and literal_components(z_center) != (half, 0):
        label += f"-zCenter{_label_value(z_center)}"
    if zbar_center is not None and literal_components(zbar_center) != literal_components(
            half if z_center is None else z_center):
        label += f"-zbCenter{_label_value(zbar_center)}"
    if nonzero(gap):
        label += f"-gap{_label_value(gap)}"
    return label


def _kac_limit_record(kac_limit, regularization_policy, block_family, metadata):
    """The ``kac_limit`` a Virasoro table at a singular c was built with: given, recorded in its metadata, or the
    active default (:func:`~virasoro_blocks.recursion.kac_limit_mode`) for a ``merged`` table; None otherwise (a
    direct table, the N=1 families, whose Kac-weight blocks stay the c -> c0 limit)."""
    from .recursion import kac_limit_mode

    if kac_limit is None:
        kac_limit = metadata.get("kac_limit")
    if kac_limit is None and regularization_policy == "merged" and block_family in (None, "virasoro"):
        kac_limit = kac_limit_mode()
    return None if kac_limit is None else kac_limit_mode(kac_limit)


@_at_table_precision
def write_mathematica(
    serialized: SerializedDerivTable,
    *,
    coordinate: str,
    ell=0,
    basis: str | None = None,
    parity_projection: str | None = None,
    parity_external=None,
    parity_exchanged: str | None = None,
    chiral_norm=None,
    sig_figs: int = None,
    z_center=None,
    zbar_center=None,
    z_shift=None,
    zbar_shift=None,
    c=None,
    h_ext=None,
    hbar_ext=None,
    q_order=None,
    npoles=None,
    pole_selection: str | None = None,
    variable: str = "Delta",
    point_value=None,
    precision_input: int | None = None,
    precision_actual: int | None = None,
    output_template: str | None = None,
    args: str | None = None,
    block_family: str | None = None,
    sector: str | None = None,
    ramond_branch: str | None = None,
    central_charge_evaluated=None,
    regularization_policy: str | None = None,
    regularization_eps_exp: int | None = None,
    collision_policy: str | None = None,
    kac_limit: str | None = None,
    scalar_G_spin0: str | None = None,
) -> str:
    """Return the Mathematica association string for one coordinate.

    Format::

        {
          zzbDeriv[0,0] -> {a0, a1, ...},
          zzbDeriv[0,1] -> {b0, b1, ...},
          ...
          shiftedPoles -> {p1, p2, ...}
        }

    Polynomial coefficients are in ascending degree order (coeffs[k] = coefficient
    of x^k); ``shiftedPoles`` are the poles in x = Delta - shift. Basis/parity
    metadata is emitted as additional rules so Mathematica-side consumers can
    distinguish scalar G, ordered B, and parity-projected B outputs.
    """
    sig_figs = _resolve_sig_figs(sig_figs)
    meta_sig_figs = _metadata_sig_figs(sig_figs)
    # Keep numerical table data at working precision. The input-precision cap
    # applies to approximate metadata; exact real input labels bypass it.
    data_sig_figs = meta_sig_figs
    if precision_input is not None:
        meta_sig_figs = min(meta_sig_figs, int(precision_input))

    metadata = getattr(serialized, "metadata", None) or {}
    if block_family is None:
        block_family = metadata.get("block_family")
    if sector is None:
        sector = metadata.get("super_sector")
    if ramond_branch is None:
        ramond_branch = metadata.get("ramond_branch")
    if central_charge_evaluated is None:
        central_charge_evaluated = metadata.get("central_charge_evaluated")
    if regularization_policy is None:
        regularization_policy = metadata.get("regularization_policy")
    if regularization_eps_exp is None:
        regularization_eps_exp = metadata.get("regularization_eps_exp")
    if collision_policy is None:
        collision_policy = metadata.get("collision_policy")
    kac_limit = _kac_limit_record(kac_limit, regularization_policy, block_family, metadata)

    basis = _validate_basis_metadata(_infer_half_integer_basis(basis, ell), parity_projection)
    is_chiral = basis in {
        "chiral_F", "global_chiral_F", "ns_chiral_F", "ramond_chiral_F",
    }
    if not is_chiral:
        ell = _spin_json_value(ell)

    coord_name = coordinate + "Deriv"
    numerators = serialized.numerators[coordinate]

    pairs = _kept_pairs_for_writer(
        coordinate,
        len(numerators),
        len(numerators[0]) if numerators else 0,
        basis=basis,
        ell=ell,
        parity_projection=parity_projection,
        off_diagonal=is_off_diagonal(z_center, zbar_center),
        parity_phase=total_parity_phase(parity_external, parity_exchanged),
    )
    entries = []
    for m, n in pairs:
        poly = numerators[m][n]
        coeffs_str = "{" + ", ".join(_num_str(ck, sig_figs, mma=True, preserve_data=True) for ck in poly.coeffs) + "}"
        # Chiral blocks have one Taylor index; the degenerate n = 0 column is
        # collapsed so the emitted symbol is zDeriv[m] / yDeriv[m] / qDeriv[m].
        if is_chiral:
            entries.append(f"  {coord_name}[{m}] -> {coeffs_str}")
        else:
            entries.append(f"  {coord_name}[{m},{n}] -> {coeffs_str}")

    from .recursion import DEFAULT_CHIRAL_NORM
    chiral_norm_value = (
        DEFAULT_CHIRAL_NORM if chiral_norm is None else parse_number(chiral_norm)
    )
    poles_str = "{" + ", ".join(_num_str(p, data_sig_figs, mma=True, preserve_data=True) for p in serialized.pole_list_x) + "}"
    entries.append(f"  shiftedPoles -> {poles_str}")
    entries.append(f"  chiralNormalization -> {_decimal_label_str(chiral_norm_value, meta_sig_figs, mma=True)}")
    # Expansion center (z cross ratio) and kinematic prefactor shift, mirroring
    # the JSON z_center / z_shift fields: always emitted (defaults z=1/2, shift=0)
    # so the .m table self-documents the point its derivatives are taken around
    # and the z^{h + shift} prefactor baked into the coefficients. The zbar_*
    # rules are omitted for the chiral basis (no antiholomorphic half).
    _half = mp.mpf(1) / 2
    _z_center = _half if z_center is None else parse_number(z_center)
    _z_shift = mp.mpf(0) if z_shift is None else parse_number(z_shift)
    entries.append(f"  zCenter -> {_decimal_label_str(_z_center, meta_sig_figs, mma=True)}")
    if not is_chiral:
        _zbar_center = _z_center if zbar_center is None else parse_number(zbar_center)
        entries.append(f"  zbarCenter -> {_decimal_label_str(_zbar_center, meta_sig_figs, mma=True)}")
    entries.append(f"  zShift -> {_decimal_label_str(_z_shift, meta_sig_figs, mma=True)}")
    if not is_chiral:
        _zbar_shift = _z_shift if zbar_shift is None else parse_number(zbar_shift)
        entries.append(f"  zbarShift -> {_decimal_label_str(_zbar_shift, meta_sig_figs, mma=True)}")
    smoothing = parity_smoothing_label(
        basis, ell=ell, parity_projection=parity_projection,
        parity_phase=total_parity_phase(parity_external, parity_exchanged),
    )
    entries.append(f"  paritySmoothing -> {_mma_string_or_null(smoothing)}")
    entries.append(f"  basis -> {_mma_string_or_null(basis)}")
    if basis in _SCALAR_G_BASES:
        # as the JSON scalar_G_spin0 ("single": G_{Delta,0} = B(h, h))
        tag = _table_scalar_G_spin0(serialized, scalar_G_spin0, spin_zero=variable != "L" and ell == 0)
        entries.append(f"  scalarGSpin0 -> {_mma_string_or_null(tag)}")
        # as the JSON scalar_G_odd: odd spins are B(h, hbar) + B(hbar, h), z <-> zbar even
        entries.append(f"  scalarGOdd -> {_mma_string_or_null(SCALAR_G_ODD_EVEN)}")
    entries.append(f"  blockFamily -> {_mma_string_or_null(block_family)}")
    entries.append(f"  superSector -> {_mma_string_or_null(sector)}")
    entries.append(f"  ramondBranch -> {_mma_string_or_null(ramond_branch)}")
    entries.append(
        f"  centralChargeEvaluated -> "
        f"{_decimal_label_str(central_charge_evaluated, _c_eval_sig_figs(meta_sig_figs, regularization_eps_exp), mma=True) if central_charge_evaluated is not None else 'Null'}"
    )
    entries.append(
        f"  regularizationPolicy -> {_mma_string_or_null(regularization_policy)}"
    )
    entries.append(
        f"  regularizationEpsExp -> "
        f"{regularization_eps_exp if regularization_eps_exp is not None else 'Null'}"
    )
    entries.append(
        f"  collisionPolicy -> {_mma_string_or_null(collision_policy)}"
    )
    if kac_limit is not None:
        # the block on a Kac weight of the singular c (point values there; a Delta table carries the merged form)
        entries.append(f"  kacLimit -> {_mma_string_or_null(kac_limit)}")
    entries.append(f"  parityProjection -> {_mma_string_or_null(parity_projection)}")
    entries.append(
        f"  parityBasis -> {_mma_string_or_null('diagonal' if parity_external is not None else None)}"
    )
    entries.append(f"  parityExternal -> {_mma_string_list(parity_external)}")
    entries.append(f"  parityExchanged -> {_mma_string_or_null(parity_exchanged)}")

    # Self-contained block data, so the .m does not rely on the filename and
    # carries everything needed to reconstruct/interpret the table. expBase and
    # shift come from the serialized table and are required for evaluation:
    #   value(variable) = expBase^x / prod(x - shiftedPole) * deriv(x),
    #   x = variable - shift.
    # The rest mirror the JSON header (filename-encoded fields brought in-body).
    def _mma_num_list(xs):
        return "{" + ", ".join(_decimal_label_str(x, meta_sig_figs, mma=True) for x in xs) + "}"

    m_max = len(numerators) - 1
    n_max = (len(numerators[0]) - 1) if numerators else 0
    entries.append(f"  expBase -> {_num_str(serialized.base, data_sig_figs, mma=True, preserve_data=True)}")
    entries.append(f"  shift -> {_decimal_label_str(serialized.shift, data_sig_figs, mma=True)}")
    entries.append(f"  variable -> {_mma_string_or_null(variable)}")
    # A point-constraint block is degree 0 in its positivity variable; the single
    # frozen value (Delta, or the chiral weight h) sits in `pointValue` -- the
    # .m analog of the JSON `point_value` field. (`shift` carries the same value,
    # so the table evaluates to the frozen block at x = 0.)
    if variable == "point" and point_value is not None:
        entries.append(
            f"  pointValue -> {_decimal_label_str(point_value, meta_sig_figs, mma=True)}"
        )
    entries.append(f"  coordinate -> {_mma_string_or_null(coordinate)}")
    entries.append(
        f"  centralCharge -> {_decimal_label_str(c, meta_sig_figs, mma=True) if c is not None else 'Null'}"
    )
    if h_ext is not None:
        h4m = _external_weights(h_ext, "h_ext")
        entries.append(f"  hExternal -> {_mma_num_list(h4m)}")
        if not is_chiral and hbar_ext is not None:
            hb4m = _external_weights(hbar_ext, "hbar_ext")
            entries.append(f"  hbarExternal -> {_mma_num_list(hb4m)}")
            entries.append(
                f"  deltaExternal -> {_mma_num_list([literal_number(h) + literal_number(hb) for h, hb in zip(h4m, hb4m)])}"
            )
            spins = [_external_spin_value(h, hb) for h, hb in zip(h4m, hb4m)]
            entries.append(f"  spinExternal -> {{{', '.join(str(s) for s in spins)}}}")
        else:
            entries.append("  hbarExternal -> Null")
            entries.append("  deltaExternal -> Null")
            entries.append("  spinExternal -> Null")
    else:
        for k in ("hExternal", "hbarExternal", "deltaExternal", "spinExternal"):
            entries.append(f"  {k} -> Null")
    entries.append(f"  spin -> {'Null' if is_chiral else _spin_json_value(ell)}")
    entries.append(f"  lambda -> {max(m_max, n_max)}")
    entries.append(f"  order -> {q_order if q_order is not None else 'Null'}")
    entries.append(
        f"  npoles -> {npoles if npoles is not None else len(serialized.pole_list_x)}"
    )
    # as in the JSON header: the order a build actually used, the poles kept, the ranking, and the checked digits
    order_used = getattr(serialized, "q_order_used", None)
    if order_used is not None and order_used != q_order:
        entries.append(f"  orderUsed -> {order_used}")
    kept = sum(1 for p in serialized.pole_list_x if not (mp.im(mp.mpc(p)) == 0 and mp.re(mp.mpc(p)) > 0))
    if npoles is not None and serialized.pole_list_x and kept != npoles:
        entries.append(f"  npolesKept -> {kept}")
    entries.append(f"  poleSelection -> {_mma_string_or_null(pole_selection)}")
    for key, name in (("verified_digits", "verifiedDigits"), ("verified_digits_largest_entry", "verifiedDigitsLargestEntry")):
        value = getattr(serialized, key, None)
        entries.append(f"  {name} -> {'Null' if value is None else format(float(value), '.1f')}")

    # Keep interpolation nodes at coefficient precision (not metadata
    # precision), matching the JSON writer's exact-node intent.
    if serialized.sample_points_x is not None:
        sample_points_str = "{" + ", ".join(
            _num_str(point, sig_figs, mma=True, preserve_data=True) for point in serialized.sample_points_x
        ) + "}"
        entries.append(f"  samplePointsX -> {sample_points_str}")

    if serialized.sample_via is not None:
        entries.append(f"  sampleVia -> {_mma_string_or_null(serialized.sample_via)}")

    if precision_input is not None:
        entries.append(f"  precisionInput -> {precision_input}")
    if precision_actual is None and precision_input is not None:
        precision_actual = mp.mp.dps
    if precision_actual is not None:
        entries.append(f"  precisionActual -> {precision_actual}")
    if output_template is not None:
        entries.append(f"  outputTemplate -> {_mma_string_or_null(output_template)}")
    if args is not None:
        entries.append(f"  args -> {_mma_string_or_null(args)}")

    return "{\n" + ",\n".join(entries) + "\n}"


# ---------------------------------------------------------------------------
# JSON writer
# ---------------------------------------------------------------------------


def _write_verification_extras(doc, serialized) -> None:
    """``verified_digits_largest_entry`` (the worst error relative to the largest entry at each check point) and
    ``target_digits`` (the --target-digits the build escalated to deliver; absent when it was not asked for)."""
    largest = getattr(serialized, "verified_digits_largest_entry", None)
    if largest is not None:
        doc["verified_digits_largest_entry"] = f"{largest:.1f}"
    target = getattr(serialized, "target_digits", None)
    if target is not None:
        doc["target_digits"] = int(target)

@_at_table_precision
def write_json(
    serialized: SerializedDerivTable,
    *,
    c,
    h_ext,
    hbar_ext=None,
    spin: int = 0,
    q_order: int | None,
    npoles: int = None,
    basis: str | None = None,
    parity_projection: str | None = None,
    parity_external=None,
    parity_exchanged: str | None = None,
    chiral_norm=None,
    precision_input: int = None,
    precision_actual: int = None,
    spin_input=None,
    output_template: str = None,
    args: str = None,
    sig_figs: int = None,
    variable: str = "Delta",
    twist=None,
    large_spin_lmin=None,
    point_value=None,
    sample_via: str | None = None,
    z_center=None,
    zbar_center=None,
    z_shift=None,
    zbar_shift=None,
    block_family: str | None = None,
    sector: str | None = None,
    ramond_branch: str | None = None,
    central_charge_evaluated=None,
    regularization_policy: str | None = None,
    regularization_eps_exp: int | None = None,
    collision_policy: str | None = None,
    pole_selection: str | None = None,
    kac_limit: str | None = None,
    scalar_G_spin0: str | None = None,
) -> str:
    """Return a JSON string in the blocks_3d-compatible Virasoro schema.

    For 2d bases (``scalar_G``, ``ordered_B``, ``parity_projected_B``) the
    ``derivs`` field is a 4-D nested list indexed as
    ``derivs[coordinate_index][m][n][monomial_degree]``. For the chiral basis
    (``chiral_F``) the ``n`` axis is collapsed to a 3-D
    ``derivs[coordinate_index][m][monomial_degree]`` -- chiral blocks carry one
    Taylor index. ``hbar_ext`` and ``spin`` are not used in the chiral case and
    may be omitted.

    All high-precision numeric values (central charge when applicable,
    external weights, shift, base, poles, polynomial coefficients) are stored
    as strings; order/lambda are stored as JSON integers or null; spin data
    are stored as JSON ints or .5-valued floats. Parity metadata, when
    present, is carried separately as strings.

    The ``label`` field records the operator-labeling scheme. ``basis`` records
    the block family explicitly: scalar G, ordered B, parity-projected B, or
    chiral F.

    ``variable`` records which positivity variable the shifted polynomial
    variable ``x = <variable> - variable_minus_x`` runs over: ``"Delta"`` for an
    ordinary fixed-spin block table, ``"L"`` for a fixed-twist large-spin
    table (where ``variable_minus_x`` is the interpolation lower edge ``Lmax``),
    or ``"point"`` for a degree-0 point-constraint block frozen at a single value.
    For ``variable == "L"`` the large-spin labels ``twist`` and
    ``large_spin_lmin`` are recorded as strings; for ``variable == "point"`` the
    frozen value is recorded as ``point_value``; they are omitted otherwise.

    A scalar-``G`` table records its spin-0 normalization as ``scalar_G_spin0``:
    the convention the table was built under (``serialized.scalar_G_spin0``),
    or ``scalar_G_spin0`` when the caller passes it (it must agree with the
    table's record), else the active convention.  It also records
    ``scalar_G_odd = "even"``: its odd-spin blocks are the z <-> zbar-even
    ``B(h, hbar) + B(hbar, h)``.
    """
    if variable not in ("Delta", "L", "point"):
        raise ValueError(
            f"variable must be 'Delta', 'L', or 'point'; got {variable!r}"
        )
    sig_figs = _resolve_sig_figs(sig_figs)
    meta_sig_figs = _metadata_sig_figs(sig_figs)
    # Write table data at working precision and user parameters at input
    # precision, preserving computed digits without overstating input accuracy.
    data_sig_figs = meta_sig_figs
    if precision_input is not None:
        meta_sig_figs = min(meta_sig_figs, int(precision_input))

    metadata = getattr(serialized, "metadata", None) or {}
    if block_family is None:
        block_family = metadata.get("block_family")
    if sector is None:
        sector = metadata.get("super_sector")
    if ramond_branch is None:
        ramond_branch = metadata.get("ramond_branch")
    if central_charge_evaluated is None:
        central_charge_evaluated = metadata.get("central_charge_evaluated")
    if regularization_policy is None:
        regularization_policy = metadata.get("regularization_policy")
    if regularization_eps_exp is None:
        regularization_eps_exp = metadata.get("regularization_eps_exp")
    if collision_policy is None:
        collision_policy = metadata.get("collision_policy")
    kac_limit = _kac_limit_record(kac_limit, regularization_policy, block_family, metadata)

    h4 = _external_weights(h_ext, "h_ext")

    # Resolve basis early -- the chiral path takes a one-axis branch through
    # the derivs builder and skips the 2d-only metadata fields. For everything
    # else we keep the existing inference from the spin_external pattern.
    basis = _infer_half_integer_basis(basis, spin)
    if basis is None:
        if hbar_ext is None:
            basis = "scalar_G"
        else:
            hb_tmp = _external_weights(hbar_ext, "hbar_ext")
            hbar_ext = hb_tmp
            spin_tmp = [_external_spin_value(h, hb) for h, hb in zip(h4, hb_tmp)]
            basis = "ordered_B" if any(spin_tmp) else "scalar_G"
    basis = _validate_basis_metadata(basis, parity_projection)
    is_chiral = basis in {
        "chiral_F", "global_chiral_F", "ns_chiral_F", "ramond_chiral_F",
    }
    is_global = basis.startswith("global_")

    if not is_global and c is None:
        raise ValueError("non-global JSON output requires c")
    if not is_global and q_order is None:
        raise ValueError("non-global JSON output requires q_order")

    if not is_chiral:
        spin = _spin_json_value(spin)
    if is_chiral:
        hb4 = None
        delta4 = None
        spin_external = None
    else:
        if hbar_ext is None:
            raise ValueError("non-chiral bases require hbar_ext")
        hb4 = _external_weights(hbar_ext, "hbar_ext")
        delta4 = [literal_number(h) + literal_number(hb) for h, hb in zip(h4, hb4)]
        spin_external = [_external_spin_value(h, hb) for h, hb in zip(h4, hb4)]

    first_coord = serialized.coordinates[0]
    first_table = serialized.numerators[first_coord]
    m_max = len(first_table) - 1
    n_max = len(first_table[0]) - 1

    if any(max(len(serialized.numerators[coord]) - 1,
               len(serialized.numerators[coord][0]) - 1) != max(m_max, n_max)
           for coord in serialized.coordinates[1:]):
        raise ValueError("coordinate tables disagree on the derivative cutoff; JSON stores one shared lambda")

    derivs = []
    for coord in serialized.coordinates:
        table = serialized.numerators[coord]
        pairs = _kept_pairs_for_writer(
            coord,
            len(table),
            len(table[0]) if table else 0,
            basis=basis,
            ell=spin,
            parity_projection=parity_projection,
            off_diagonal=is_off_diagonal(z_center, zbar_center),
        parity_phase=total_parity_phase(parity_external, parity_exchanged),
        )
        if is_chiral:
            # Chiral: collapse the degenerate n = 0 column to a flat
            # derivs[coord][m][k] shape so the JSON matches the one-index
            # math.
            coord_derivs = [
                [_num_str(ck, sig_figs, preserve_data=True) for ck in table[m][0].coeffs]
                for (m, _n) in pairs
            ]
        else:
            # Group kept pairs by m to emit jagged rows:
            # derivs[coord][m_index][n_index][k]. m_index is 0..len(rows)-1
            # and indexes into the *jagged* shape, not the dense one.
            rows_by_m: dict[int, list] = {}
            for (m, n) in pairs:
                rows_by_m.setdefault(m, []).append(
                    [_num_str(ck, sig_figs, preserve_data=True) for ck in table[m][n].coeffs]
                )
            coord_derivs = [rows_by_m[m] for m in sorted(rows_by_m)]
        derivs.append(coord_derivs)

    if basis in (
        "scalar_G", "global_scalar_G", "ns_scalar_G", "ramond_scalar_G",
    ):
        label = "delta_l"
    elif basis in ("ordered_B", "global_ordered_B", "ns_ordered_B", "ramond_ordered_B"):
        label = "h_hbar"
    elif basis in (
        "parity_projected_B", "global_parity_projected_B",
        "ns_parity_projected_B", "ramond_parity_projected_B",
    ):
        label = "parity_pm"
    elif basis in ("chiral_F", "global_chiral_F", "ns_chiral_F", "ramond_chiral_F"):
        label = "chiral_h"
    else:
        raise ValueError(f"basis must be one of {sorted(_VALID_BASES)}")

    from .recursion import DEFAULT_CHIRAL_NORM
    chiral_norm_value = (
        DEFAULT_CHIRAL_NORM if chiral_norm is None else parse_number(chiral_norm)
    )
    doc = {
        "central_charge": None if is_global else _input_metadata_num_str(c, meta_sig_figs),
        "chiral_normalization_N": _decimal_label_str(chiral_norm_value, meta_sig_figs),
        "h_external": [_input_metadata_num_str(h, meta_sig_figs) for h in h4],
        "hbar_external": (
            None if hb4 is None else [_input_metadata_num_str(hb, meta_sig_figs) for hb in hb4]
        ),
        "delta_external": (
            None if delta4 is None else [_input_metadata_num_str(d, meta_sig_figs) for d in delta4]
        ),
        "spin_external": spin_external,
        "basis": basis,
        "parity_projection": parity_projection,
        "parity_basis": "diagonal" if parity_external is not None else None,
        "parity_external": list(parity_external) if parity_external is not None else None,
        "parity_exchanged": parity_exchanged,
        # blocks_3d's p_+/- smoothing factor applied before parity-folded
        # derivatives. "p_plus" means p_+ = 1 (no smoothing); "p_minus" means
        # p_- = 2/(z - zbar) for xt, 2/(y - ybar) for ws, 2/(q - qbar) for
        # qqb_ws. null for raw ordered_B (folded coords are rejected there)
        # and for chiral_F (chiral blocks live on a single coordinate).
        "parity_smoothing": parity_smoothing_label(
            basis, ell=spin, parity_projection=parity_projection,
            parity_phase=total_parity_phase(parity_external, parity_exchanged),
        ),
        # "delta_l": scalar_G, labeled by (Delta, ell); "h_hbar":
        # ordered_B, labeled by (h, hbar), also for scalar externals;
        # "parity_pm": parity-projected spinning basis, with the sign stored
        # separately in parity_projection; "chiral_h": chiral-only block,
        # labeled by the chiral weight h alone (no exchanged spin).
        "label": label,
        # `delta_12` and `delta_43` are intentionally omitted: they are pure
        # functions of `delta_external` (subtract two entries), and the
        # Virasoro block also depends on the sum d_2 + d_3 and on Sum d_i,
        # not just on the differences. Consumers that want the differences
        # can compute them from `delta_external` in one subtraction.
        "spin": None if is_chiral else _spin_json_value(spin),
        "variable_minus_x": _input_metadata_num_str(serialized.shift, data_sig_figs),
        "variable": variable,
    }
    if basis in _SCALAR_G_BASES:
        # the spin-0 normalization that produced the table ("single": G_{Delta,0} = B(h, h)); consumers refuse a
        # spin-0 table without it.  The convention the
        # table was built under (recorded on it when serialized), never merely the one active at write time
        doc[SCALAR_G_SPIN0_KEY] = _table_scalar_G_spin0(
            serialized, scalar_G_spin0, spin_zero=variable != "L" and doc["spin"] == 0)
        # odd spins are the z <-> zbar-even B(h, hbar) + B(hbar, h); readers refuse odd-spin scalar_G data
        # without this record
        doc[SCALAR_G_ODD_KEY] = SCALAR_G_ODD_EVEN
    if block_family is not None:
        doc["block_family"] = block_family
    if sector is not None:
        doc["super_sector"] = sector
    if ramond_branch is not None:
        doc["ramond_branch"] = ramond_branch
    if central_charge_evaluated is not None:
        doc["central_charge_evaluated"] = _decimal_label_str(
            central_charge_evaluated, _c_eval_sig_figs(meta_sig_figs, regularization_eps_exp)
        )
    if regularization_policy is not None:
        doc["regularization_policy"] = regularization_policy
    if regularization_eps_exp is not None:
        doc["regularization_eps_exp"] = int(regularization_eps_exp)
    if collision_policy is not None:
        doc["collision_policy"] = collision_policy
    if kac_limit is not None:
        # the block on a Kac weight of the singular c (point values there; a Delta table carries the merged form)
        doc["kac_limit"] = kac_limit

    # Expansion center: the cross ratio z0 (and zbar0 for 2d blocks) about which
    # the derivative table is Taylor-expanded. Always emitted and specified as a
    # *z* cross ratio (even for the y / q chiral coordinates, whose --center is a
    # z-value mapped internally); the default 1/2 is the crossing-symmetric
    # point. A value != 1/2 means every derivative coefficient is taken around
    # that point, not around 1/2. For a 2d block zbar_center gives the
    # antiholomorphic center (== z_center on the diagonal); chiral blocks have no
    # zbar, so only z_center is emitted.
    _half = mp.mpf(1) / 2
    _z_center = _half if z_center is None else parse_number(z_center)
    doc["z_center"] = _input_metadata_num_str(_z_center, meta_sig_figs)
    if not is_chiral:
        _zbar_center = _z_center if zbar_center is None else parse_number(zbar_center)
        doc["zbar_center"] = _input_metadata_num_str(_zbar_center, meta_sig_figs)

    # Kinematic exponent shift: the block prefactor is F ~ z^{h + z_shift} (and
    # zbar^{hbar + zbar_shift} for the antiholomorphic half), so the shift is
    # baked into the derivative coefficients and is NOT recoverable from the
    # other fields. Always emitted, default 0 (leading power z^h / N^h). For a 2d block
    # zbar_shift gives the antiholomorphic shift (== z_shift for the symmetric
    # scalar_G; independent for the two-sided ordered_B / parity_projected_B);
    # chiral blocks have no zbar, so only z_shift is emitted.
    _z_shift = mp.mpf(0) if z_shift is None else parse_number(z_shift)
    doc["z_shift"] = _decimal_label_str(_z_shift, meta_sig_figs)
    if not is_chiral:
        _zbar_shift = _z_shift if zbar_shift is None else parse_number(zbar_shift)
        doc["zbar_shift"] = _decimal_label_str(_zbar_shift, meta_sig_figs)

    # Large-spin identity metadata sits next to `variable`: for variable == "L"
    # the `twist` and `large_spin_lmin` define what the block is (parallel to
    # `spin`), so they belong in the header, not the provenance footer. Omitted
    # for ordinary fixed-spin (`variable == "Delta"`) tables.
    if variable == "L":
        if twist is not None:
            doc["twist"] = _input_metadata_num_str(twist, meta_sig_figs)
        if large_spin_lmin is not None:
            doc["large_spin_lmin"] = _input_metadata_num_str(large_spin_lmin, meta_sig_figs)
    # A point-constraint block is degree 0 in its positivity variable: the
    # single frozen value sits in `point_value` (a Delta for 2d bases, a chiral
    # weight h for chiral bases -- `basis` disambiguates). `variable_minus_x` (=
    # the shift) carries the same value, so the table evaluates to the frozen
    # block at x = 0.
    elif variable == "point":
        if point_value is not None:
            doc["point_value"] = _input_metadata_num_str(point_value, meta_sig_figs)

    doc["exp_base"] = _num_str(serialized.base, data_sig_figs, preserve_data=True)
    # Global fixed-spin tables use q_order for the SL(2) candidate pole pool;
    # it is not a numerical truncation of their exact 2F1 evaluation.
    doc["order"] = q_order
    # a build that enlarges its candidate pool (global tables without --npoles) records the order it used
    order_used = getattr(serialized, "q_order_used", None)
    if order_used is not None and order_used != q_order:
        doc["order_used"] = order_used
    doc["lambda"] = max(m_max, n_max)
    doc["npoles"] = (
        npoles if npoles is not None else len(serialized.pole_list_x)
    )
    # npoles is the requested label; a conjugate pair is never split, so an odd request can keep one pole fewer
    # (poles at real x > 0, split off exactly at c <= 1, are not part of the count)
    kept = sum(1 for p in serialized.pole_list_x if not (mp.im(mp.mpc(p)) == 0 and mp.re(mp.mpc(p)) > 0))
    if npoles is not None and serialized.pole_list_x and kept != npoles:
        doc["npoles_kept"] = kept
    if pole_selection is not None:
        # the ranking that chose the npoles kept poles (a reused file is checked against the requested one)
        doc["pole_selection"] = pole_selection
    doc["coordinates"] = list(serialized.coordinates)
    doc["pole_list_x"] = [
        _num_str(p, data_sig_figs, preserve_data=True) for p in serialized.pole_list_x
    ]

    # Interpolation sample points (in the shifted variable x = <variable> -
    # variable_minus_x). Present only for the interpolation method; absent for
    # `--npoles all` tables. They are reconstruction data parallel to
    # `pole_list_x` (both are needed to evaluate the interpolant), so they sit
    # in the header right after the poles. Stored at the full coefficient sig
    # figs (not the metadata sig figs) because a downstream pmp2sdp
    # `sampleScalings` path evaluates the damped-rational prefactor at exactly
    # these nodes and needs them lossless. The loader restores them onto
    # `SerializedDerivTable.sample_points_x`, so the explicit-sample-point PMP
    # path survives the round trip.
    if serialized.sample_points_x is not None:
        doc["sample_points_x"] = [
            _input_metadata_num_str(point, sig_figs) for point in serialized.sample_points_x
        ]

    if sample_via is None:
        sample_via = serialized.sample_via
    if sample_via is not None:
        doc["sample_via"] = sample_via

    if serialized.sample_block_values is not None:
        _write_sample_block_values(doc, serialized, basis=basis, spin=spin,
                                   parity_projection=parity_projection, is_chiral=is_chiral, sig_figs=sig_figs,
                                   off_diagonal=is_off_diagonal(z_center, zbar_center),
                                   parity_phase=total_parity_phase(parity_external, parity_exchanged))

    doc["index_names"] = (
        ["coordinate_index", "deriv", "monomial_degree"] if is_chiral
        else ["coordinate_index", "deriv_0", "deriv_1", "monomial_degree"]
    )
    doc["derivs"] = derivs

    if precision_input is not None:
        doc["precision_input"] = precision_input
    # `precision_actual` mirrors blocks_3d: the working precision in effect at
    # output time, which may exceed `precision_input` when an internal pass
    # bumped mp.mp.dps. Default to the current mp.mp.dps if the caller did not
    # supply a value, so the field is always present when `precision_input` is.
    if precision_actual is None and precision_input is not None:
        precision_actual = mp.mp.dps
    if precision_actual is not None:
        doc["precision_actual"] = precision_actual
    verified = getattr(serialized, "verified_digits", None)
    if verified is not None:
        # digits to which the stored table matches its own numeric sampler (worst entry, relative to itself) over
        # verified_delta_range: between the interpolation nodes and past them to the large-Delta regime
        doc["verified_digits"] = f"{verified:.1f}"
    _write_verification_extras(doc, serialized)
    verified_range = getattr(serialized, "verified_range", None)
    if verified is not None and verified_range is not None:
        # the interval of the table's variable the check covered (past its last node the table extrapolates)
        doc["verified_spin_range" if variable == "L" else "verified_delta_range"] = [
            _range_num_str(x, meta_sig_figs) for x in verified_range
        ]
    truncation = getattr(serialized, "q_truncation_digits", None)
    if truncation is not None:
        # the q-order gate's estimate of the --order truncation (verified_digits compares at the same order)
        doc["q_truncation_digits"] = f"{min(float(truncation), 999.0):.1f}"
    if spin_input is not None:
        doc["spin_input"] = list(spin_input)
    if output_template is not None:
        doc["output_template"] = output_template
    if args is not None:
        doc["args"] = args

    return json.dumps(doc, indent=2)


def _write_sample_block_values(doc, serialized, *, basis, spin, parity_projection, is_chiral, sig_figs,
                               off_diagonal=False, parity_phase=1) -> None:
    """Put the table's clean node values (``serialized.sample_block_values``) into ``doc``.

    Same jagged kept-pairs shape as `derivs`, with the sample-point axis innermost (the analogue of the
    monomial-degree axis): sample_block_values[coord][m_idx][n_idx][k] for 2d, collapsed to [coord][m_idx][k] for
    chiral. Only the coordinate's kept pairs are serialized, as for `derivs`. The n > m symmetry half is omitted
    only for scalar and parity-projected non-folded tables at equal centers, and for torus tables. Ordered and
    off-diagonal non-folded tables retain both orientations. The loader recomputes `kept_pairs` and zero-pads
    omitted entries, exactly as it does for `derivs`.
    """
    sbv_doc: dict = {}
    for coord, tables in serialized.sample_block_values.items():
        n_samples = len(tables)
        grid = serialized.numerators[coord]
        pairs = _kept_pairs_for_writer(
            coord,
            len(grid),
            max((len(row) for row in grid), default=0),
            basis=basis,
            ell=spin,
            parity_projection=parity_projection,
            off_diagonal=off_diagonal,
            parity_phase=parity_phase,
        )
        if is_chiral:
            coord_sbv = [
                [_num_str(tables[k][m][0], sig_figs, preserve_data=True) for k in range(n_samples)]
                for (m, _n) in pairs
            ]
        else:
            rows_by_m: dict[int, list] = {}
            for (m, n) in pairs:
                rows_by_m.setdefault(m, []).append(
                    [_num_str(tables[k][m][n], sig_figs, preserve_data=True) for k in range(n_samples)]
                )
            coord_sbv = [rows_by_m[m] for m in sorted(rows_by_m)]
        sbv_doc[coord] = coord_sbv
    doc["sample_block_values"] = sbv_doc
    doc["sample_block_values_index_names"] = (
        ["coordinate_index", "deriv", "sample_point"] if is_chiral
        else ["coordinate_index", "deriv_0", "deriv_1", "sample_point"]
    )


# ---------------------------------------------------------------------------
# Torus (genus-1) JSON writer
# ---------------------------------------------------------------------------


@_at_table_precision
def write_torus_json(
    serialized: SerializedDerivTable,
    *,
    c,
    spin: int = 0,
    basis: str = "torus_G",
    onept_d_ext=None,
    q_order: int = None,
    npoles: int = None,
    npoles_chosen: int = None,
    spin_pair_symmetrized: bool = True,
    precision_input: int = None,
    precision_actual: int = None,
    spin_input=None,
    output_template: str = None,
    args: str = None,
    sig_figs: int = None,
    regularization_policy: str | None = None,
    variable: str = "Delta",
    point_value=None,
    kac_limit: str | None = None,
) -> str:
    """Return a JSON string for a torus derivative table.

    Two torus bases share this schema (the torus member of the
    blocks_3d-compatible family written by :func:`write_json`, with the same
    reconstruction contract ``exp_base``/``variable_minus_x``/``pole_list_x``
    /jagged ``derivs`` and the same loader
    :func:`virasoro_blocks.serialization.serialized_table_from_json` via the
    ``vvb`` keep rule):

    * ``torus_G`` -- reduced-character (partition function) tables.
      Exact: empty pole list, no ``order``/``npoles`` knobs, numerator of
      entry ``(m, n)`` of degree exactly ``m + n``.
    * ``torus_onept_G`` -- reduced one-point tables from the CCY
      4.27 recursion. These carry genuine Kac ``pole_list_x``, a series
      truncation ``q_order`` (~2.7 digits/order at tau = i), a kept-pole
      count ``npoles``, and the external scalar's chiral weight
      ``onept_d_ext`` (Delta_Phi = 2 * onept_d_ext).
      Positive real poles are always retained, with their merged orders;
      ``npoles_requested`` caps only the remaining poles, while ``npoles``
      in the output counts every retained pole.  Without ``--npoles`` the
      cap is chosen from the measured digits and recorded as
      ``npoles_chosen`` (pass it as ``--npoles`` to rebuild the table).

    Sphere-specific metadata (external cross-ratio weights, centers/shifts,
    parity fields, ``chiral_normalization_N``) is intentionally absent;
    surface-identifying metadata (``surface``, ``nome_convention``,
    ``crossing_point_tau``) replaces it.

    ``variable="point"`` writes a point table (the block frozen at one Delta,
    degree-0 numerators, as :func:`write_json` writes the sphere ones): the
    value is recorded as ``point_value`` and the spin as ``spin``.  ``kac_limit``
    (``"irreducible"``) is recorded when a degenerate module's irreducible
    character or trace entered.
    """
    if variable not in ("Delta", "point"):
        raise ValueError(f"torus tables take variable 'Delta' or 'point', got {variable!r}")
    if basis not in ("torus_G", "torus_onept_G"):
        raise ValueError(
            f"basis must be 'torus_G' or 'torus_onept_G', got {basis!r}"
        )
    if basis == "torus_onept_G":
        if onept_d_ext is None:
            raise ValueError("torus_onept_G requires onept_d_ext")
        if q_order is None:
            raise ValueError("torus_onept_G requires q_order")
    else:
        if onept_d_ext is not None:
            raise ValueError("onept_d_ext only applies to torus_onept_G")
    sig_figs = _resolve_sig_figs(sig_figs)
    meta_sig_figs = _metadata_sig_figs(sig_figs)
    # Write table data at working precision and user parameters at input
    # precision, preserving computed digits without overstating input accuracy.
    data_sig_figs = meta_sig_figs
    if precision_input is not None:
        meta_sig_figs = min(meta_sig_figs, int(precision_input))
    first_coord = serialized.coordinates[0]
    first_table = serialized.numerators[first_coord]
    m_max = len(first_table) - 1
    n_max = max(len(row) for row in first_table) - 1

    derivs = []
    for coord in serialized.coordinates:
        table = serialized.numerators[coord]
        pairs = _kept_pairs_for_writer(
            coord,
            len(table),
            max(len(row) for row in table),
            basis=basis,
            ell=spin,
            parity_projection=None,
        )
        rows_by_m: dict[int, list] = {}
        for (m, n) in pairs:
            if n >= len(table[m]):
                continue
            rows_by_m.setdefault(m, []).append(
                [_num_str(ck, sig_figs, preserve_data=True) for ck in table[m][n].coeffs]
            )
        derivs.append([rows_by_m[m] for m in sorted(rows_by_m)])

    doc = {
        "surface": "torus",
        "central_charge": _input_metadata_num_str(c, meta_sig_figs),
        "basis": basis,
        "label": "delta_j",
        "spin": _spin_json_value(spin),
        # The (j, -j) pair sum (reality of Z pairs the degeneracies); the
        # bootstrap consumes this directly. False only for the raw
        # single-orientation table.
        "spin_pair_symmetrized": bool(spin_pair_symmetrized),
        "variable_minus_x": _input_metadata_num_str(serialized.shift, data_sig_figs),
        "variable": variable,
        # Modular conventions: q_tau = e^(2 pi i tau); the sphere writers'
        # elliptic nome is a different q (q_modular = q_nome^2 at equal tau).
        "nome_convention": "modular_2pi",
        "crossing_point_tau": "i",
        "exp_base": _num_str(serialized.base, data_sig_figs, preserve_data=True),
        "order": q_order,
        "lambda": max(m_max, n_max),
        # the poles the table KEEPS (a conjugate pair is never split, so the cut can keep fewer than asked;
        # a spin-0 double pole counts twice); the requested cap is npoles_requested
        "npoles": (
            len(serialized.pole_list_x) if basis == "torus_onept_G"
            else (npoles if npoles is not None else (len(serialized.pole_list_x) or None))
        ),
        "coordinates": list(serialized.coordinates),
        "pole_list_x": [
            _num_str(p, data_sig_figs, preserve_data=True) for p in serialized.pole_list_x
        ],
    }
    if regularization_policy is not None:
        doc["regularization_policy"] = regularization_policy
    if kac_limit is not None:
        doc["kac_limit"] = kac_limit
    if variable == "point" and point_value is not None:
        # the frozen Delta (variable_minus_x carries the same value as the table's shift)
        doc["point_value"] = _input_metadata_num_str(point_value, meta_sig_figs)
    if basis == "torus_onept_G" and npoles is not None:
        doc["npoles_requested"] = int(npoles)
    if basis == "torus_onept_G" and npoles_chosen is not None:
        doc["npoles_chosen"] = int(npoles_chosen)
    if onept_d_ext is not None:
        doc["onept_d_ext"] = _decimal_label_str(onept_d_ext, meta_sig_figs)
    if serialized.sample_points_x is not None:
        doc["sample_points_x"] = [
            _input_metadata_num_str(x, sig_figs) for x in serialized.sample_points_x
        ]
    if serialized.sample_via is not None:
        doc["sample_via"] = serialized.sample_via
    if serialized.sample_block_values is not None:
        _write_sample_block_values(doc, serialized, basis=basis, spin=spin, parity_projection=None,
                                   is_chiral=False, sig_figs=sig_figs)
    doc["index_names"] = ["coordinate", "m", "n", "monomial_degree"]
    doc["derivs"] = derivs
    if precision_input is not None:
        doc["precision_input"] = precision_input
    # as in write_json: present whenever precision_input is -- the precision the table was built at (its
    # working_dps), else the one in effect now
    if precision_actual is None and precision_input is not None:
        precision_actual = int(getattr(serialized, "working_dps", None) or mp.mp.dps)
    if precision_actual is not None:
        doc["precision_actual"] = precision_actual
    verified = getattr(serialized, "verified_digits", None)
    if verified is not None:
        # digits to which the stored table reproduces the block (for torus_onept_G: against a deeper,
        # higher-precision reference, on verified_delta_range)
        doc["verified_digits"] = f"{verified:.1f}"
    _write_verification_extras(doc, serialized)
    meta = serialized.metadata or {}
    for key in ("fit_digits", "precision_digits", "q_truncation_digits"):
        if meta.get(key) is not None:
            doc[key] = f"{meta[key]:.1f}"
    if meta.get("verified_delta_range") is not None:
        # the Delta range the check points span: between the nodes and far past the last one
        doc["verified_delta_range"] = [_range_num_str(x, meta_sig_figs) for x in meta["verified_delta_range"]]
    if spin_input is not None:
        doc["spin_input"] = list(spin_input)
    if output_template is not None:
        doc["output_template"] = output_template
    if args is not None:
        doc["args"] = args

    return json.dumps(doc, indent=2)
