"""Fast positivity certification of SDPB dual functionals.

After SDPB reports ``dual_feasible``, the dual solution vector ``y``
defines a linear functional that *should* be non-negative on every
spin-channel block for all Δ ≥ Δ_min.  When explicit sample points are
not provided and pmp2sdp evaluates the polynomial at its own (mismatched)
nodes, the functional may be spuriously feasible — it satisfies positivity
on the *polynomial approximation* at pmp2sdp's nodes but violates it on
the true blocks at our interpolation nodes.

This module evaluates the functional at a grid of Δ values and checks
positivity. On the gap path (and every problem whose ``y`` has the
F-pair length) the default ``exact_blocks=True`` re-derives the EXACT
blocks for the check — the numeric block recursion for the fixed-spin
Δ channels and the true Virasoro chiral q-series for the large-spin
tower — rather than the npoles interpolants the SDP solved against, so
block interpolation error is caught rather than hidden. The
normalized-moment direct-SDP path is different: its INCLUDED blocks are
checked against the SDP's own interpolant (sample scan plus root
isolation of the PMP polynomials) unless ``force_exact_included``, and
only its extra/adaptive coverage uses true blocks. The check is fast
(a few seconds for typical Lambda=11 problems with ~10 spins).
"""

from __future__ import annotations

import dataclasses
import math
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from virasoro_blocks._conventions import pool_initializer_kwargs
from fractions import Fraction
from numbers import Integral
from pathlib import Path

import mpmath as mp
from virasoro_blocks.parallel import usable_workers
from virasoro_blocks.requests import _integer_field

from virasoro_blocks.serialization import SerializedDerivTable
from virasoro_blocks.truncation import kept_pairs
from .defaults import DEFAULT_POLE_SELECTION


def _certification_real_label(value, *, name="twist"):
    # These real types need no numerical conversion to validate them.
    # literal_number also rejects nonfinite mpf values without rounding bits.
    if isinstance(value, (Fraction, mp.mpf, int)) and not isinstance(value, bool):
        from virasoro_blocks.numeric_input import literal_number

        return literal_number(value)
    from .identical_scalar import _as_real_label

    return _as_real_label(value, name=name)


def _channel_twist_metadata(channel):
    """Keep a display value and an exact identity for adaptive coverage."""
    twist = getattr(channel, "twist", None)
    if twist is None:
        return {"twist": None}
    return _certification_point_metadata("twist", twist)


def _certification_point_metadata(name, value):
    """A finite display coordinate and an exact label for reproducing a scan."""
    from virasoro_blocks.writers import _metadata_num_str

    label = _certification_real_label(value, name=name)
    text = _metadata_num_str(label, mp.mp.dps)
    try:
        display = float(label)
    except OverflowError:
        display = text
    else:
        if not math.isfinite(display):
            display = text
    return {name: display, name + "_label": text}


def _detail_twist_label(detail):
    return _certification_real_label(detail.get("twist_label", detail["twist"]))


def _finite_real(value) -> bool:
    """Allow only the existing output tolerance for imaginary roundoff."""
    if not mp.isfinite(value):
        return False
    imaginary = abs(mp.im(value))
    if imaginary == 0:
        return True
    from virasoro_blocks.clean_samples import _real_tol

    return imaginary <= _real_tol(value)


def _finite_real_nonnegative(value) -> bool:
    return _finite_real(value) and mp.re(value) >= 0


def _certification_display_value(value):
    """Use floats when their range retains the value, text otherwise."""
    try:
        display = float(value)
    except (OverflowError, TypeError, ValueError):
        return mp.nstr(value, 17)
    if not math.isfinite(display) or (display == 0 and value != 0):
        return mp.nstr(value, 17)
    return display


def _min_value_fields(value) -> dict:
    """``min_value`` (the JSON-safe display value) and ``min_value_text`` (always its decimal string)."""
    display = _certification_display_value(value)
    text = display if isinstance(display, str) else mp.nstr(value, 17)
    return {"min_value": display, "min_value_text": text}


def _detail_sort_value(detail) -> mp.mpf:
    """A detail's smallest functional value as an ``mpf``; a missing or non-finite one sorts first."""
    value = detail.get("min_value_text", detail.get("min_value"))
    try:
        value = mp.mpf(str(value)) if not isinstance(value, mp.mpf) else value
    except (TypeError, ValueError):
        return mp.ninf
    return value if mp.isfinite(value) else mp.ninf


def worst_certification_detail(details):
    """The detail of the channel where the functional is smallest, or None for no details.

    Failed channels come first, then the smallest ``min_value``; a missing,
    NaN, or infinite minimum counts as the worst value.  ``min_value`` itself
    is a float when the double range holds it and its decimal string otherwise
    (JSON cannot store ``1e-500`` or NaN as a number), so it is not a sort key
    on its own; ``min_value_text`` is always the string.
    """
    details = list(details)
    if not details:
        return None
    return min(details, key=lambda d: (bool(d.get("passed")), _detail_sort_value(d)))


def _record_negative_sample(witness, value, *, point=None, point_value=None, **location):
    """Retain a finite real witness separately from invalid sample values.

    ``n_negative`` counts all failed samples, including nonfinite
    and nonreal values. Its count alone does not establish negative sign.
    """
    if witness["first_negative_value"] is None and _finite_real(value) and mp.re(value) < 0:
        witness["first_negative_value"] = _certification_display_value(mp.re(value))
        witness.update(location)
        if point is not None:
            witness.update(_certification_point_metadata("first_negative_" + point, point_value))


class NonFiniteDualVectorError(ValueError):
    """``y.txt`` holds a NaN or infinite component: it certifies nothing."""


def _stored_real(value):
    """Read numerical data at its printed precision, without reboxing an mpf."""
    if isinstance(value, mp.mpf):
        return value
    token = str(value).strip()
    mantissa = token.lower().split('e', 1)[0]
    digits = sum(ch.isdigit() for ch in mantissa)
    with mp.workdps(max(mp.mp.dps, digits + 5)):
        return mp.mpf(token)


def load_y_vector(sdpb_out_dir, *, full: bool = False) -> list[mp.mpf]:
    """Load the dual solution y without truncating the digits SDPB printed.

    SDPB solves for the functional with one component eliminated through the
    normalization ``n . y = 1`` (the pivot, the component with the largest
    ``|n_i|``), so ``y.txt`` holds ``N - 1`` values for an ``N``-component
    functional.  By default they are returned as printed.  With ``full=True``
    the eliminated component is restored (:func:`reconstruct_full_y`), giving
    the ``N``-component functional that :func:`certify_dual_functional` takes::

        y = load_y_vector(out, full=True)
        ok, details = certify_dual_functional(y, problem.spin_channels, problem.F_pairs)

    Raises :class:`NonFiniteDualVectorError` when a component is NaN or
    infinite: every positivity test here is a ``value < 0`` comparison, which
    is False for NaN, so a non-finite functional would otherwise certify.
    """
    if not isinstance(full, bool):
        raise ValueError("full must be True or False")
    y_path = Path(sdpb_out_dir) / "y.txt"
    with open(y_path) as f:
        lines = f.readlines()
    try:
        header = lines[0].strip().split()
        if len(header) != 2:
            raise ValueError
        count, columns = map(int, header)
        if count < 0 or columns != 1:
            raise ValueError
    except (IndexError, ValueError):
        raise ValueError("y.txt: header must be '<nonnegative row count> 1'") from None
    y = [_stored_real(line.strip()) for line in lines[1:] if line.strip()]
    if len(y) != count:
        raise ValueError(f"y.txt: expected {count} values, got {len(y)}")
    bad = [i for i, value in enumerate(y) if not mp.isfinite(value)]
    if bad:
        raise NonFiniteDualVectorError(
            f"y.txt: {len(bad)} non-finite component(s), first at index {bad[0]}"
        )
    return reconstruct_full_y(y, sdpb_out_dir) if full else y


def _normalization_of_length(problem, sdpb_out_dir, length: int):
    """The functional normalization ``n`` (``n . y = 1``) of the given length, or None."""
    import json

    doc = getattr(getattr(problem, "pmp_build", None), "doc", None)
    norm = (doc or {}).get("normalization")
    if norm is not None and len(norm) == length:
        return [_stored_real(x) for x in norm]
    norm_path = _sdp_dir_for_output(sdpb_out_dir) / "normalization.json"
    if norm_path.exists():
        with open(norm_path) as f:
            norm = json.load(f).get("normalization")
        if norm is not None and len(norm) == length:
            return [_stored_real(x) for x in norm]
    return None


def _normalization_violation(y_full, problem, sdpb_out_dir) -> dict | None:
    """A failure detail when the full functional violates ``n . y = 1``, else None.

    pmp2sdp eliminates one component through this constraint, so a reconstructed
    SDPB vector satisfies it to rounding.  A full-length ``y`` that does not
    (the all-zero vector, which is trivially "non-negative" everywhere) is not
    a certificate of this problem.  Nothing is checked when no normalization of
    ``y``'s length is known.
    """
    norm = _normalization_of_length(problem, sdpb_out_dir, len(y_full))
    if norm is None:
        return None
    value = mp.fdot(norm, y_full)
    # This is a constraint on the result, not a backward-error test on the
    # summands. Scaling by their magnitudes would accept n.y = 0 (or negative)
    # whenever large terms cancel, which cannot certify the normalized problem.
    if mp.isfinite(value) and abs(value - 1) <= mp.mpf("1e-8"):
        return None
    return {
        "ell": None,
        "twist": None,
        "passed": False,
        "reason": "normalization_violated",
        "normalization_value": _certification_display_value(value),
        **_min_value_fields(mp.nan),
        "n_test_points": 0,
    }


def reconstruct_full_y(y_sdpb, sdpb_out_dir) -> list[mp.mpf]:
    """The full ``N``-component functional from SDPB's ``N - 1`` printed components.

    The eliminated (pivot) component is recovered from the normalization
    ``n . y = 1`` recorded in the ``normalization.json`` of the SDP directory
    that produced *sdpb_out_dir* (written by both the direct SDP writer and
    pmp2sdp).  A vector that is already ``N`` long is returned unchanged.
    Raises ``ValueError`` when the SDP directory records no normalization or
    the length is neither ``N - 1`` nor ``N``.
    """
    y_sdpb = [_stored_real(value) for value in y_sdpb]
    # restore the pivot at the precision y was printed at, not the (possibly 15-digit) ambient one
    bits = max([mp.mp.prec, *(value._mpf_[3] for value in y_sdpb)])
    with mp.workprec(bits + 20):
        return _reconstruct_full_y(y_sdpb, sdpb_out_dir, strict=True)


def _reconstruct_full_y(y_sdpb: list[mp.mpf], sdpb_out_dir, *, strict: bool = False) -> list[mp.mpf]:
    """Reconstruct the full N-component y vector from SDPB's (N-1)-component output.

    pmp2sdp eliminates the functional component with the largest absolute
    normalization coefficient using the constraint
    ``sum_i normalization[i] * y_full[i] = 1``.  The SDPB output y.txt
    contains the remaining N-1 components (in order, skipping the eliminated
    index).  The eliminated component is recovered from the normalization
    constraint.  Without ``strict`` a vector that cannot be reconstructed (no
    normalization, an unexpected length) is returned as it is.
    """
    import json

    sdp_dir = _sdp_dir_for_output(sdpb_out_dir)
    from .hpc.storage import RECORD_NAME, _read_cleanup_record

    if (sdp_dir / RECORD_NAME).exists():
        _read_cleanup_record(sdp_dir)
    norm_path = sdp_dir / "normalization.json"
    if not norm_path.exists():
        if strict:
            raise ValueError(
                f"cannot restore the eliminated component of y: {norm_path} does not exist "
                "(the SDP directory SDPB solved records the normalization there)"
            )
        return y_sdpb  # fallback: cannot reconstruct without normalization

    with open(norm_path) as f:
        norm_data = json.load(f)
    from .sdp_writer import _normalization_basis

    original_dps = None
    log_path = sdp_dir.parent / 'pmp2sdp.log'
    if log_path.exists():
        match = re.search(r'eval_dps=(\d+)', log_path.read_text(errors='replace'))
        original_dps = int(match[1]) if match else None
    norm, elim_idx = _normalization_basis(norm_data, eval_dps=original_dps)
    N = len(norm)

    if len(y_sdpb) == N:
        return y_sdpb  # already full dimension
    if len(y_sdpb) != N - 1:
        if strict:
            raise ValueError(
                f"y has {len(y_sdpb)} components, but the normalization in {norm_path} has {N}: "
                f"expected {N - 1} (SDPB's output) or {N}"
            )
        return y_sdpb  # unexpected dimension, return as-is

    # The N-1 values in y.txt correspond to the non-eliminated components of
    # the full positivity functional; the eliminated component is fixed by
    # N.alpha = 1.
    y_full = [mp.mpf(0)] * N
    free_indices = [i for i in range(N) if i != elim_idx]
    for fi, yi in zip(free_indices, y_sdpb):
        y_full[fi] = yi

    # Recover eliminated component from normalization constraint.
    residual = mp.fdot([(mp.mpf(1), mp.mpf(1)), *(
        (norm[i], mp.fneg(y_full[i], exact=True)) for i in free_indices
    )])
    y_full[elim_idx] = residual / norm[elim_idx]
    return y_full


_SDP_DIRECTORY_LOG_RE = re.compile(r'^SDP directory[ \t]*:[ \t]*(.+?)[ \t]*$', re.MULTILINE)
_OUT_DIRECTORY_LOG_RE = re.compile(r'^out directory[ \t]*:[ \t]*(.+?)[ \t]*$', re.MULTILINE)


def _logged_directory(token, parent):
    token = token.strip()
    if token.startswith('"'):
        import shlex

        try:
            parts = shlex.split(token)
        except ValueError:
            return None
        if len(parts) != 1:
            return None
        token = parts[0]
    if not token:
        return None
    path = Path(token)
    return path if path.is_absolute() else parent / path


def _is_sdp_dir(path: Path) -> bool:
    return path.is_dir() and (
        (path / "control.json").exists() or (path / "normalization.json").exists()
    )


def _sdp_dir_for_output(sdpb_out_dir) -> Path:
    """The SDP directory SDPB solved to produce *sdpb_out_dir*.

    The solve helpers take an ``sdp_dir_name`` (default ``"sdp"``), so the
    directory is not always called ``sdp``. In order: the output's source
    record, a matching SDPB log (``SDP directory`` and ``out directory``),
    a sibling ``sdp``,
    then the only sibling that looks like an SDP directory (holds
    ``control.json`` or ``normalization.json``). When nothing matches, the
    historical ``<parent>/sdp`` is returned, so callers see the same "missing"
    behavior as before.
    """
    out = Path(sdpb_out_dir)
    from .hpc.storage import recorded_sdp_source

    recorded = recorded_sdp_source(out)
    if recorded is not None:
        return recorded
    parent = out.parent
    try:
        logs = sorted(parent.glob("*.log"))
    except OSError:
        logs = []
    matched, legacy, foreign = {}, {}, set()
    for log in logs:
        try:
            text = log.read_text(errors="replace")
        except OSError:
            continue
        records = list(_SDP_DIRECTORY_LOG_RE.finditer(text))
        for i, match in enumerate(records):
            candidate = _logged_directory(match.group(1), parent)
            if candidate is None:
                continue
            end = records[i+1].start() if i+1 < len(records) else len(text)
            output_match = _OUT_DIRECTORY_LOG_RE.search(text, match.end(), end)
            key = candidate.resolve()
            if output_match is not None:
                recorded_out = _logged_directory(output_match.group(1), parent)
                if recorded_out is not None and recorded_out.resolve() == out.resolve():
                    matched[key] = candidate
                else:
                    foreign.add(key)
            elif _is_sdp_dir(candidate):
                legacy[key] = candidate
    choices = matched or {key: value for key, value in legacy.items() if key not in foreign}
    if len(choices) > 1:
        raise ValueError(f"ambiguous SDP directories for solver output: {out}")
    if choices:
        candidate = next(iter(choices.values()))
        if not _is_sdp_dir(candidate):
            raise ValueError(f"recorded SDP directory is unavailable: {candidate}")
        return candidate
    default = parent / "sdp"
    if _is_sdp_dir(default) and default.resolve() not in foreign:
        return default
    try:
        candidates = [
            d for d in sorted(parent.iterdir()) if d != out and _is_sdp_dir(d)
            and d.resolve() not in foreign
        ]
    except OSError:
        candidates = []
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1 or (foreign and _is_sdp_dir(default)):
        raise ValueError(f"ambiguous SDP directories for solver output: {out}")
    return default


def _has_direct_sdp_data(sdpb_out_dir) -> bool:
    sdp_dir = _sdp_dir_for_output(sdpb_out_dir)
    return (sdp_dir / "control.json").exists() and (sdp_dir / "normalization.json").exists()


def _channel_shift(channel):
    """The table's ``shift``, so a polynomial root in ``x`` can be reported as a
    physical ``Delta = x + shift``.  None when unavailable -- callers then omit
    the Delta rather than guessing one."""
    for attr in ("prefactored", "serialized"):
        table = getattr(channel, attr, None)
        shift = getattr(table, "shift", None)
        if shift is not None:
            return shift
    return None


def _direct_sdp_block_detail(
    y_sdpb: list[mp.mpf],
    sdp_dir: Path,
    block_index: int,
    *,
    channel=None,
    basis: str = "direct_sdp_samples",
    extra: bool = False,
) -> dict:
    import json

    data_path = sdp_dir / f"block_data_{block_index}.json"
    try:
        with data_path.open() as f:
            block_data = json.load(f)
    except FileNotFoundError:
        from .hpc.storage import load_compacted_sdp_block

        block_data = load_compacted_sdp_block(sdp_dir, block_index)
    if block_data is None:
        detail = {
            "block_index": block_index,
            "ell": getattr(channel, "ell", None),
            **_channel_twist_metadata(channel),
            "passed": False,
            "reason": "missing_direct_sdp_block_data",
            "n_test_points": 0,
            "basis": basis,
        }
        if extra:
            detail["extra"] = True
        return detail

    c_values = [_stored_real(v) for v in block_data.get("c", [])]
    b_rows = block_data.get("B", [])
    if not c_values or len(c_values) != len(b_rows):
        return {
            "block_index": block_index,
            "ell": getattr(channel, "ell", None),
            "passed": False,
            "reason": "direct_sdp_sample_count_mismatch",
            "n_test_points": 0,
            "basis": basis,
            **({"extra": True} if extra else {}),
        }
    min_val = mp.mpf("+inf")
    min_x = None
    n_negative = 0
    witness = {"first_negative_value": None}
    for k, (c, row) in enumerate(zip(c_values, b_rows)):
        b = [_stored_real(v) for v in row]
        if len(b) != len(y_sdpb):
            return {
                "block_index": block_index,
                "ell": getattr(channel, "ell", None),
                **_channel_twist_metadata(channel),
                "passed": False,
                "reason": "direct_sdp_dimension_mismatch",
                "row_length": len(b),
                "y_length": len(y_sdpb),
                "n_test_points": len(c_values),
                "basis": basis,
            }
        val = mp.fdot([(c, mp.mpf(1)), *(
            (bj, mp.fneg(yj, exact=True)) for bj, yj in zip(b, y_sdpb)
        )])
        real_val = mp.re(val)
        if real_val < min_val or mp.isnan(real_val):
            min_val = real_val
            min_x = k
        if not _finite_real_nonnegative(val):
            n_negative += 1
            _record_negative_sample(witness, val, first_negative_sample_index=k)

    detail = {
        "block_index": block_index,
        "ell": getattr(channel, "ell", None),
        **_channel_twist_metadata(channel),
        "passed": n_negative == 0,
        **_min_value_fields(min_val),
        "min_sample_index": min_x,
        "n_negative": n_negative,
        "n_test_points": len(c_values),
        "basis": basis,
        **witness,
    }
    if extra:
        detail["extra"] = True
    return detail


def _evaluate_functional_on_block(
    y: list[mp.mpf],
    block: SerializedDerivTable,
    F_pairs: list[tuple[int, int]],
    Delta,
    coordinate: str = "zzb",
    *,
    _work_cache: dict | None = None,
) -> mp.mpf:
    """Evaluate sum_i y_i * F_i(Delta) for one block at a single Delta."""
    caller = mp.mp.dps
    cache = {} if _work_cache is None else _work_cache
    work = max(caller, int(getattr(block, 'working_dps', None) or 0),
               cache.get('work_dps', 0))

    def evaluate():
        table = block.evaluate(coordinate, Delta)
        values = []
        for m, n in F_pairs:
            if not (0 <= m < len(table) and 0 <= n < len(table[m])):
                raise ValueError(f"certification block is missing functional entry ({m}, {n})")
            values.append(table[m][n])
        # fdot forms the products and their sum before rounding. fsum of
        # already-rounded products can erase an order-one negative residual.
        return mp.fdot(y, values), y_scale * max(abs(v) for v in values)

    with mp.workdps(work):
        # ||y||_1 max |F_i| bounds the term scale. The vector is fixed across
        # a channel's grid, so reuse its norm rather than multiplying every
        # absolute term again at every point.
        y_scale = cache.get('y_scale')
        if y_scale is None:
            y_scale = mp.fsum(abs(yi) for yi in y)
        result, scale = evaluate()
        if mp.isfinite(scale) and scale > 0 and abs(result) <= mp.eps*scale:
            # Fresh block entries can also have rounded onto one another.
            # Resolve their contraction at absolute caller precision. The
            # extra digits follow the measured term scale, not a fixed pad.
            needed = caller + max(0, int(mp.floor(mp.log10(scale))) + 1)
            if needed > work:
                with mp.workdps(needed):
                    result, _ = evaluate()
                work = needed
        cache.update(work_dps=work, y_scale=y_scale)
    return +result


def _pmp_vector_length(pmp_doc: dict) -> int | None:
    blocks = pmp_doc.get("PositiveMatrixWithPrefactorArray") or []
    for block in blocks:
        try:
            return len(block["polynomials"][0][0])
        except (KeyError, IndexError, TypeError):
            continue
    return None


def _pmp_test_points(block_doc: dict, *, n_test_points: int, delta_range: float):
    sample_points = _sample_points_for_block(block_doc)
    if sample_points:
        points = set(sample_points)
        ordered = sorted(sample_points)
        for left, right in zip(ordered, ordered[1:]):
            points.add((left + right) / 2)
        return sorted(points)

    n = max(int(n_test_points), 1)
    if n == 1:
        return [mp.mpf(0)]
    end = _certification_real_label(delta_range, name="delta_range")
    return [end * k / (n - 1) for k in range(n)]


def _pmp_block_min(
    y: list[mp.mpf],
    block_doc: dict,
    *,
    n_test_points: int = 100,
    delta_range: float = 30.0,
) -> tuple[mp.mpf, mp.mpf, int]:
    components = block_doc["polynomials"][0][0]
    test_xs = _pmp_test_points(
        block_doc,
        n_test_points=n_test_points,
        delta_range=delta_range,
    )
    min_val = mp.mpf("+inf")
    min_x = mp.mpf(0)
    n_negative = 0
    # Use the same exact combination as the whole-ray root certificate. A
    # rounded component-wise contraction can erase a negative endpoint or a
    # small negative constant before the grid checks its sign.
    from .positivity import _as_mpf, _combined_exact, _exact_real

    try:
        polynomial = _combined_exact(components, y)
    except ValueError:
        return mp.nan, test_xs[0], len(test_xs)
    for x in test_xs:
        val = _as_mpf(polynomial(_exact_real(x)))
        real_val = mp.re(val)
        if real_val < min_val or mp.isnan(real_val):
            min_val = real_val
            min_x = x
        if not _finite_real_nonnegative(val):
            n_negative += 1
    return min_val, min_x, n_negative


def _pmp_channel_detail(
    y: list[mp.mpf],
    block_doc: dict,
    *,
    block_index: int,
    channel=None,
    n_test_points: int = 100,
    delta_range: float = 30.0,
    basis: str = "pmp_polynomial",
    extra: bool = False,
) -> dict:
    test_xs = _pmp_test_points(
        block_doc,
        n_test_points=n_test_points,
        delta_range=delta_range,
    )
    min_val, min_x, n_negative = _pmp_block_min(
        y,
        block_doc,
        n_test_points=n_test_points,
        delta_range=delta_range,
    )
    detail = {
        "block_index": block_index,
        "ell": getattr(channel, "ell", None),
        **_channel_twist_metadata(channel),
        "passed": n_negative == 0,
        **_min_value_fields(min_val),
        **_certification_point_metadata("min_x", min_x),
        "n_negative": n_negative,
        "n_test_points": len(test_xs),
        "basis": basis,
    }
    if extra:
        detail["extra"] = True
    return _with_polynomial_positivity(detail, block_doc, y, channel)


def _sample_points_for_block(block_doc: dict):
    return [_certification_real_label(x, name="sample point") for x in block_doc.get("samplePoints", [])]


def _sample_value_channel_detail(
    y_full: list[mp.mpf],
    block_doc: dict,
    sample_values,
    *,
    block_index: int,
    channel=None,
    basis: str = "clean_sample_values",
    extra: bool = False,
) -> dict:
    xs = _sample_points_for_block(block_doc)
    if not sample_values or len(sample_values) != len(xs):
        return {
            "block_index": block_index,
            "ell": getattr(channel, "ell", None),
            "passed": False,
            "reason": "sample_value_count_mismatch",
            "n_test_points": 0,
            "basis": basis,
            **({"extra": True} if extra else {}),
        }
    min_val = mp.mpf("+inf")
    min_x = None
    min_sample_index = None
    n_negative = 0
    witness = {"first_negative_value": None}
    for k, row in enumerate(sample_values):
        if len(row) != len(y_full):
            detail = {
                "block_index": block_index,
                "ell": getattr(channel, "ell", None),
                **_channel_twist_metadata(channel),
                "passed": False,
                "reason": "sample_value_dimension_mismatch",
                "row_length": len(row),
                "y_length": len(y_full),
                "n_test_points": len(sample_values),
                "basis": basis,
            }
            if extra:
                detail["extra"] = True
            return detail
        val = mp.fdot(y_full, [_stored_real(v) for v in row])
        real_val = mp.re(val)
        if real_val < min_val or mp.isnan(real_val):
            min_val = real_val
            min_sample_index = k
            min_x = xs[k] if k < len(xs) else None
        if not _finite_real_nonnegative(val):
            n_negative += 1
            _record_negative_sample(witness, val, point="x", point_value=xs[k])

    detail = {
        "block_index": block_index,
        "ell": getattr(channel, "ell", None),
        **_channel_twist_metadata(channel),
        "passed": n_negative == 0,
        **_min_value_fields(min_val),
        "min_sample_index": min_sample_index,
        "n_negative": n_negative,
        "n_test_points": len(sample_values),
        "basis": basis,
        **witness,
    }
    if min_x is not None:
        detail.update(_certification_point_metadata("min_x", min_x))
    if extra:
        detail["extra"] = True
    return detail


def _linear_certification_offsets(n, end):
    """Exact linear grid, including both edges when at least two are requested."""
    if n < 4 or end <= 2:
        return [end * i / max(n - 1, 1) for i in range(n)]
    n_near, n_far = n // 2, n - n // 2
    near = _certification_real_label(2)
    return ([near * i / (n_near - 1) for i in range(n_near)]
            + [near + (end - near) * i / (n_far - 1) for i in range(n_far)])


def _channel_test_offsets(
    n_test_points: int, delta_range: float, log_near_points: int = 0
):
    """Offsets above a channel's threshold at which to test positivity.

    The linear grid places half its points in the first 2 units and half in
    the remaining range. Interpolation nodes cluster near the threshold, so
    this grid can miss a negative functional value between nearby nodes.

    ``log_near_points`` adds that many logarithmically-spaced offsets from
    ``1e-9`` up to ``min(2, delta_range)``, which resolves the node-dense
    region. Shorter ranges collapse these additions to their endpoint. Zero
    (the default) uses only the linear grid.
    """
    n = max(int(n_test_points), 1)
    if n == 1:
        return [mp.mpf(0)]
    end = _certification_real_label(delta_range, name="delta_range")
    offsets = _linear_certification_offsets(n, end)
    if log_near_points > 0 and end > 0:
        hi = min(_certification_real_label(2), end)
        lo = min(_certification_real_label("1e-9"), hi)
        steps = max(int(log_near_points), 2)
        for i in range(steps):
            if i == 0:
                offsets.append(lo)
            elif i == steps - 1:
                offsets.append(hi)
            else:
                frac = mp.mpf(i) / (steps - 1)
                point = _certification_real_label(lo * (hi / lo) ** frac)
                offsets.append(min(hi, max(lo, point)))
    return sorted(set(offsets))


def _normalized_moment_cert_weights(problem):
    from virasoro_bootstrap.moments import MomentWeight

    meta = (getattr(problem, "pmp_metadata", None) or {}).get("moment") or {}
    numerator = MomentWeight.from_json(meta["numerator"])
    denominator = MomentWeight.from_json(meta["denominator"])
    if meta.get("direction") == "upper":
        numerator = numerator.scale(-1)
    return numerator, denominator


def _normalized_moment_physical_row(
    y_full: list[mp.mpf],
    problem,
    channel,
    value,
    *,
    weights,
) -> mp.mpf:
    table = channel.prefactored.evaluate(
        getattr(problem.spec, "coordinate", "zzb"), value
    )
    return _normalized_moment_physical_row_from_table(
        y_full, problem, channel, value, table, weights=weights,
    )


def _normalized_moment_physical_row_from_table(
    y_full: list[mp.mpf],
    problem,
    channel,
    value,
    table,
    *,
    weights,
    _scale_out=None,
) -> mp.mpf:
    """Contract one already-evaluated physical block table.

    The batch path below uses this to evaluate all offsets of a fixed-spin or
    tower channel in one numeric-recursion call.  Keeping the contraction
    separate also makes it explicit that batching changes no normalization or
    moment convention.
    """
    # Evaluate physical quantum numbers directly, retaining exact labels and
    # the SDP's spin-filter rules. Reconstructing a rounded polynomial in
    # value - delta_min loses small offsets in hbar or Delta - L.
    from virasoro_bootstrap.moments import (
        _channel_weight_value,
        _normalize_moment_block_basis,
    )

    basis_name = (
        (problem.pmp_metadata or {}).get("moment", {}).get("block_basis", "nonratio")
    )
    basis_key = str(basis_name).lower()
    if "over_self_dual_measure" in basis_key:
        basis = "ratio"
    elif "plus_self_dual_measure" in basis_key:
        basis = "nonratio"
    else:
        basis = _normalize_moment_block_basis(basis_name)
    entries = [table[m][n] for m, n in problem.F_pairs]
    weight_values = [_channel_weight_value(w, channel, value) for w in weights]
    if not all(_finite_real(v) for v in [table[0][0], *entries, *weight_values]):
        return mp.nan
    measure = mp.re(table[0][0])
    row_length = len(entries) + len(weight_values)
    if row_length != len(y_full):
        raise ValueError(
            f"normalized moment row has length {row_length} but y has length {len(y_full)}"
        )
    if basis == "ratio" and measure <= 0:
        return mp.mpf("-inf")
    if not all(_finite_real(y) for y in y_full):
        return mp.nan
    from flint import fmpq
    from .positivity import _as_mpf, _exact_real

    # The weights are exact rational functions of supplied quantum numbers.
    # Converting them, multiplying by the measure, or dividing each crossing
    # entry first can erase a negative residual before even an exact dot sum.
    # Contract the stored block values and weights as rationals, then round
    # only the final physical value. Arb inputs retain their native midpoint.
    split = len(entries)
    crossing_terms = [_exact_real(y) * _exact_real(mp.re(v))
                      for y, v in zip(y_full[:split], entries)]
    moment_terms = [_exact_real(y) * _exact_real(v)
                    for y, v in zip(y_full[split:], weight_values)]
    crossing = sum(crossing_terms, fmpq(0))
    moment = sum(moment_terms, fmpq(0))
    exact_measure = _exact_real(measure)
    if _scale_out is not None:
        crossing_scale = sum(map(abs, crossing_terms), fmpq(0))
        moment_scale = sum(map(abs, moment_terms), fmpq(0))
        source_scale = crossing_scale + abs(exact_measure) * moment_scale
        physical_scale = source_scale / exact_measure if basis == 'ratio' else source_scale
        _scale_out['scale'] = _as_mpf(physical_scale)
        # Ratio normalization can hide how many absolute digits the underlying
        # block entries need: two enormous entries divided by the same measure
        # look like ordinary numbers even after their difference has rounded out.
        _scale_out['source_scale'] = _as_mpf(max(source_scale, physical_scale))
    return _as_mpf(crossing / exact_measure + moment if basis == "ratio"
                   else crossing + exact_measure * moment)


def _normalized_moment_physical_tables(problem, channel, values):
    """Evaluate a physical channel in a batch when the sampler supports it.

    Restrict the batch to the required entries and share the q-series setup.
    Custom channels without a batch interface use scalar evaluation.
    """
    coordinate = getattr(problem.spec, "coordinate", "zzb")
    prefactored = channel.prefactored
    sampler = getattr(prefactored, "_sampler", None)
    entries = set(getattr(problem, "F_pairs", ()) or ())
    entries.add((0, 0))
    entries = sorted(entries)
    if sampler is not None:
        try:
            if hasattr(sampler, "sample_bucket"):
                tables, _ = sampler.sample_bucket(list(values), entries)
            else:
                tables, _ = sampler(list(values), entries=entries)
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
            # A sampler may reject a restricted batch for a folded/custom basis;
            # the scalar path remains the correctness-preserving fallback.
            pass
        else:
            if len(tables) != len(values):
                raise ValueError(
                    f"physical block sampler returned {len(tables)} tables for {len(values)} points"
                )
            return tables
    return [prefactored.evaluate(coordinate, value) for value in values]


def _normalized_moment_physical_channel_detail(
    y_full: list[mp.mpf],
    problem,
    channel,
    *,
    block_index: int,
    n_test_points: int,
    delta_range: float,
    weights,
    extra: bool,
    adaptive_stage: str | None = None,
    log_near_points: int = 0,
) -> dict:
    offsets = _channel_test_offsets(n_test_points, delta_range, log_near_points)
    threshold = _certification_real_label(channel.delta_min, name="delta_min")
    values = [threshold + x for x in offsets]
    caller = mp.mp.dps

    def evaluate(work):
        results = []
        needed = work
        with mp.workdps(work):
            tables = _normalized_moment_physical_tables(problem, channel, values)
            for value, table in zip(values, tables):
                info = {}
                val = _normalized_moment_physical_row_from_table(
                    y_full, problem, channel, value, table, weights=weights, _scale_out=info,
                )
                results.append(val)
                scale = info.get('scale', mp.mpf(0))
                if mp.isfinite(scale) and scale > 0 and abs(val) <= mp.eps*scale:
                    source_scale = info['source_scale']
                    needed = max(needed, caller + max(0, int(mp.floor(mp.log10(source_scale))) + 1))
        return results, needed

    results, needed = evaluate(caller)
    if needed > caller:
        # Preserve batching: one unresolved contraction raises the precision
        # for the channel's entire grid, rather than rebuilding per point.
        results, _ = evaluate(needed)
    min_val = mp.mpf("+inf")
    min_x = mp.mpf(0)
    n_negative = 0
    witness = {"first_negative_value": None}
    for x, val in zip(offsets, results):
        real_val = mp.re(val)
        if real_val < min_val or mp.isnan(real_val):
            min_val = real_val
            min_x = x
        if not _finite_real_nonnegative(val):
            n_negative += 1
            _record_negative_sample(witness, val, point="x", point_value=x)
    detail = {
        "block_index": block_index,
        "ell": getattr(channel, "ell", None),
        **_channel_twist_metadata(channel),
        "passed": n_negative == 0,
        **_min_value_fields(min_val),
        **_certification_point_metadata("min_x", min_x),
        "n_negative": n_negative,
        "n_test_points": len(offsets),
        "basis": "physical_numeric_blocks",
        "extra": bool(extra),
        **witness,
    }
    if adaptive_stage is not None:
        detail["adaptive_stage"] = adaptive_stage
    return detail


def _certify_normalized_moment_scalar_block(
    y_full: list[mp.mpf],
    problem,
    *,
    block_index: int,
) -> dict:
    """Check the constant scalar inequality in a normalized-moment PMP.

    The operator channels and the normalized ratio constraint are separate PMP
    blocks. The physical-block recertifier replaces only the former; omitting
    this block would make a positive channel scan look complete while silently
    dropping the scalar inequality that fixes the ratio. It is a constant
    polynomial, so checking its one coefficient is exact (there is no
    continuous Delta or tower variable here).
    """
    pmp_build = getattr(problem, "pmp_build", None)
    doc = getattr(pmp_build, "doc", None) or {}
    blocks = list(doc.get("PositiveMatrixWithPrefactorArray") or [])
    if block_index < 0 or block_index >= len(blocks):
        return {
            "block_index": block_index,
            "ell": None,
            "twist": None,
            "passed": False,
            "reason": "missing_normalized_scalar_block",
            "n_test_points": 0,
            "basis": "exact_scalar_block",
            "certificate_scope": "exact_scalar_block",
        }
    block = blocks[block_index]
    try:
        components = block["polynomials"][0][0]
    except (KeyError, IndexError, TypeError):
        return {
            "block_index": block_index,
            "ell": None,
            "twist": None,
            "passed": False,
            "reason": "unsupported_normalized_scalar_block",
            "n_test_points": 0,
            "basis": "exact_scalar_block",
            "certificate_scope": "exact_scalar_block",
        }
    if len(components) != len(y_full):
        return {
            "block_index": block_index,
            "ell": None,
            "twist": None,
            "passed": False,
            "reason": "component_length_mismatch",
            "component_length": len(components),
            "y_length": len(y_full),
            "n_test_points": 0,
            "basis": "exact_scalar_block",
            "certificate_scope": "exact_scalar_block",
        }
    # This entry represents a scalar inequality, not an operator ray. Checking
    # a nonconstant replacement only at its stored nodes could falsely label
    # it an exact scalar certificate (for example 1-x at the node zero).
    from .positivity import _exact_real

    reason = None
    try:
        if any(_exact_real(c) != 0 for component in components for c in component[1:]):
            reason = "nonconstant_normalized_scalar_block"
    except (TypeError, ValueError, OverflowError):
        reason = "invalid_normalized_scalar_block"
    if reason is not None:
        return {
            "block_index": block_index, "ell": None, "twist": None,
            "passed": False, "reason": reason, "n_test_points": 0,
            "basis": "exact_scalar_block", "certificate_scope": "exact_scalar_block",
        }
    points = _pmp_test_points(block, n_test_points=1, delta_range=0)
    min_value, min_x, n_negative = _pmp_block_min(y_full, block, n_test_points=1, delta_range=0)
    return {
        "block_index": block_index,
        "ell": None,
        "twist": None,
        "passed": n_negative == 0,
        **_min_value_fields(min_value),
        **_certification_point_metadata("min_x", min_x),
        "n_negative": n_negative,
        "n_test_points": len(points),
        "basis": "exact_scalar_block",
        "certificate_scope": "exact_scalar_block",
    }


def _with_cert_q_order(problem, q_order: int | None):
    if q_order is None:
        return problem
    spec = dataclasses.replace(problem.spec, q_order=int(q_order))
    return dataclasses.replace(problem, spec=spec)


def _certify_pmp_dual_functional(
    y: list[mp.mpf],
    pmp_doc: dict,
    *,
    spin_channels=(),
    n_test_points: int = 100,
    delta_range: float = 30.0,
) -> tuple[bool, list[dict]]:
    """Certify the dual against the serialized scalar PMP polynomials.

    This checks the exact polynomial problem passed to pmp2sdp/SDPB, including
    moment extra components whose vector length exceeds ``len(F_pairs)``.  The
    contracted polynomial is checked on the whole ray by root isolation,
    after verifying that its damped-rational prefactor preserves its sign.
    The finite grid is retained for diagnostic minima.
    """

    blocks = list(pmp_doc.get("PositiveMatrixWithPrefactorArray") or [])
    channels = list(spin_channels or ())
    details = []
    all_positive = True
    for block_index, block_doc in enumerate(blocks):
        polys = block_doc.get("polynomials")
        if (
            not isinstance(polys, list)
            or len(polys) != 1
            or not isinstance(polys[0], list)
            or len(polys[0]) != 1
        ):
            details.append(
                {
                    "block_index": block_index,
                    "ell": None,
                    "passed": False,
                    "reason": "unsupported_non_scalar_pmp_block",
                    "n_test_points": 0,
                }
            )
            all_positive = False
            continue

        components = polys[0][0]
        if len(components) != len(y):
            details.append(
                {
                    "block_index": block_index,
                    "ell": None,
                    "passed": False,
                    "reason": "component_length_mismatch",
                    "component_length": len(components),
                    "y_length": len(y),
                    "n_test_points": 0,
                }
            )
            all_positive = False
            continue

        channel = channels[block_index] if block_index < len(channels) else None
        detail = _pmp_channel_detail(
            y,
            block_doc,
            block_index=block_index,
            channel=channel,
            n_test_points=n_test_points,
            delta_range=delta_range,
        )
        details.append(detail)
        all_positive = all_positive and detail["passed"]

    return all_positive, details


def _json_safe_positivity(detail: dict) -> dict:
    """Coerce a :func:`~.positivity.block_positivity` result to JSON-safe types.

    That function reports in mpmath precision -- ``leading``, ``roots``,
    ``first_negative_x``/``_value``/``_delta`` are all ``mpf`` -- which is right
    for the interactive diagnostic but not for a task result: the HPC worker
    ``json.dumps`` the payload straight to disk and dies with "Object of type
    mpf is not JSON serializable". That failure was latent for as long as the
    exact check was dying on its length guard and returning an all-int
    dimension_mismatch dict; fixing the off-by-one is what first put real mpf
    values on this path.

    Values outside float range, including nonzero values that underflow, are
    kept as strings instead. Nonfinite diagnostics use strings too, so the
    payload remains valid JSON even when a certificate fails on invalid data.
    """
    def coerce(value):
        if isinstance(value, (mp.mpf, mp.mpc, float)):
            return _certification_display_value(value)
        if isinstance(value, (tuple, list)):
            return [coerce(v) for v in value]
        return value

    return {k: coerce(v) for k, v in detail.items()}


def _with_polynomial_positivity(detail, entry, y_full, channel):
    """A node scan cannot stand in for an available whole-ray polynomial check."""
    from .positivity import block_positivity

    exact = (block_positivity(entry, y_full, shift=_channel_shift(channel)) if y_full is not None
             else {"passed": None, "reason": "no_full_y_vector"})
    detail["exact_positivity"] = _json_safe_positivity(exact)
    # The represented interpolation polynomial, rather than the physical block.
    detail["certificate_scope"] = "interpolated_pmp_polynomial"
    if exact.get("passed") is False:
        detail["passed"] = False
        detail["reason"] = "negative_between_sample_points"
        detail["first_negative_x"] = _certification_display_value(exact["first_negative_x"])
        detail["first_negative_value"] = _certification_display_value(exact["first_negative_value"])
        if "first_negative_delta" in exact:
            detail["first_negative_delta"] = _certification_display_value(exact["first_negative_delta"])
    elif exact.get("passed") is None:
        detail["passed"] = False
        detail["reason"] = "exact_positivity_undecided"
        detail["exact_positivity_undecided"] = exact.get("reason")
    return detail


def _certify_direct_sdp_included_blocks(
    y_sdpb: list[mp.mpf],
    problem,
    sdpb_out_dir,
    y_full: list[mp.mpf] | None = None,
) -> tuple[bool, list[dict]]:
    """Certify the included blocks of a direct-SDP solve.

    ``y_sdpb`` has N-1 components after pmp2sdp eliminates one component using
    ``sum_i normalization[i] y_i = 1``. The per-sample scan combines it with
    ``block_data_*.json`` vectors in this reduced basis.

    The exact root-isolation check requires the N-component ``y_full`` from
    :func:`_reconstruct_full_y`, matching ``entry["polynomials"][0][0]``.
    If ``y_full`` is absent, the exact check is undecided and certification
    fails: sample positivity cannot establish positivity between nodes.
    """
    sdp_dir = _sdp_dir_for_output(sdpb_out_dir)
    channels = list(getattr(problem, "spin_channels", ()) or ())
    entries = list(problem.pmp_build.doc.get("PositiveMatrixWithPrefactorArray") or [])
    details = []
    all_positive = True
    for block_index in range(len(entries)):
        channel = channels[block_index] if block_index < len(channels) else None
        detail = _direct_sdp_block_detail(
            y_sdpb,
            sdp_dir,
            block_index,
            channel=channel,
            basis="direct_sdp_samples",
        )
        # The sample-point scan above evaluates ONLY at the SDP's own nodes --
        # exactly the points the block polynomial was fitted through, where it is
        # positive by construction. It cannot see a dip between them, and one
        # such dip (~0.004 wide) hid an invalid functional for a full day. Decide
        # positivity properly, on the whole ray, by root isolation.
        _with_polynomial_positivity(detail, entries[block_index], y_full, channel)
        details.append(detail)
        all_positive = all_positive and detail["passed"]
    return all_positive, details


def _is_normalized_moment_problem(problem) -> bool:
    meta = (getattr(problem, "pmp_metadata", None) or {}).get("moment") or {}
    return meta.get("kind") == "normalized"


def _normalized_moment_extra_twist_set(spec, extra_twists, interstitial_twists):
    """Preserve solved slices while placing extra normalized-moment checks."""
    if getattr(spec, "large_spin_lmin", None) is None:
        return None
    if interstitial_twists > 0:
        return _interstitial_twist_set(spec, interstitial_twists, extra_above=extra_twists)
    if extra_twists <= 0:
        return None
    base = sorted(set(_certification_real_label(t) for t in spec.large_spin_twists))
    first = 4 * _certification_real_label(spec.h_ext, name="h_ext")
    # A count-based rebuild starts at n=0 and can replace custom/gap-selected
    # slices with lower twists. Compute the first n above the exact top, without
    # materializing the entire canonical prefix (the top can be very large).
    start = max(0, math.floor((base[-1] - first) / 2) + 1) if base else 0
    return base + [first + 2 * (start + i) for i in range(extra_twists)]


def _certify_normalized_moment_extra_pmp_blocks(
    y: list[mp.mpf],
    problem,
    *,
    extra_spins: int,
    extra_twists: int,
    n_test_points: int,
    delta_range: float,
    skip_ells=(),
    skip_twists=(),
    adaptive_stage: str | None = None,
    interstitial_twists: int = 0,
) -> tuple[bool, list[dict]]:
    if extra_spins <= 0 and extra_twists <= 0 and interstitial_twists <= 0:
        return True, []

    from virasoro_bootstrap.moments import rebuild_normalized_moment_problem

    spec = problem.spec
    spin_list = tuple(int(s) for s in spec.spin_list)
    max_spin = max(spin_list) if spin_list else -2
    ext_spins = spin_list + tuple(
        max_spin + 2 * (i + 1) for i in range(max(int(extra_spins), 0))
    )
    explicit_twists = _normalized_moment_extra_twist_set(spec, extra_twists, interstitial_twists)
    ext = rebuild_normalized_moment_problem(
        problem,
        spin_list=ext_spins,
        twists=explicit_twists,
    )
    blocks = list(
        ext.pmp_build.doc.get("PositiveMatrixWithPrefactorArray") or []
    )
    channels = list(getattr(ext, "spin_channels", ()) or ())
    original_twists = {_certification_real_label(t) for t in getattr(spec, "large_spin_twists", ())}
    skipped_ells = {int(ell) for ell in skip_ells}
    skipped_twists = {_certification_real_label(tw) for tw in skip_twists}

    details: list[dict] = []
    all_positive = True
    for block_index, block_doc in enumerate(blocks):
        if block_index >= len(channels):
            continue
        channel = channels[block_index]
        twist = getattr(channel, "twist", None)
        if twist is None:
            if int(channel.ell) <= max_spin:
                continue
            if int(channel.ell) in skipped_ells:
                continue
        else:
            if _certification_real_label(twist) in original_twists:
                continue
            if _certification_real_label(twist) in skipped_twists:
                continue

        polys = block_doc.get("polynomials")
        if (
            not isinstance(polys, list)
            or len(polys) != 1
            or not isinstance(polys[0], list)
            or len(polys[0]) != 1
        ):
            detail = {
                "block_index": block_index,
                "ell": getattr(channel, "ell", None),
                **_channel_twist_metadata(channel),
                "passed": False,
                "reason": "unsupported_non_scalar_pmp_block",
                "n_test_points": 0,
                "basis": "pmp_polynomial_extra",
                "extra": True,
            }
        elif len(polys[0][0]) != len(y):
            detail = {
                "block_index": block_index,
                "ell": getattr(channel, "ell", None),
                **_channel_twist_metadata(channel),
                "passed": False,
                "reason": "component_length_mismatch",
                "component_length": len(polys[0][0]),
                "y_length": len(y),
                "n_test_points": 0,
                "basis": "pmp_polynomial_extra",
                "extra": True,
            }
        else:
            detail = _pmp_channel_detail(
                y,
                block_doc,
                block_index=block_index,
                channel=channel,
                n_test_points=n_test_points,
                delta_range=delta_range,
                basis="pmp_polynomial_extra",
                extra=True,
            )
        detail.update(_channel_twist_metadata(channel))
        details.append(detail)
        if adaptive_stage is not None:
            detail["adaptive_stage"] = adaptive_stage
        all_positive = all_positive and detail["passed"]

    return all_positive, details


def _certify_normalized_moment_extra_sample_blocks(
    y_full: list[mp.mpf],
    problem,
    *,
    extra_spins: int,
    extra_twists: int,
    skip_ells=(),
    skip_twists=(),
    adaptive_stage: str | None = None,
    interstitial_twists: int = 0,
) -> tuple[bool, list[dict]]:
    if extra_spins <= 0 and extra_twists <= 0 and interstitial_twists <= 0:
        return True, []

    from virasoro_bootstrap.moments import rebuild_normalized_moment_problem

    spec = problem.spec
    spin_list = tuple(int(s) for s in spec.spin_list)
    max_spin = max(spin_list) if spin_list else -2
    ext_spins = spin_list + tuple(
        max_spin + 2 * (i + 1) for i in range(max(int(extra_spins), 0))
    )
    explicit_twists = _normalized_moment_extra_twist_set(spec, extra_twists, interstitial_twists)
    ext = rebuild_normalized_moment_problem(
        problem,
        spin_list=ext_spins,
        twists=explicit_twists,
    )
    blocks = list(ext.pmp_build.doc.get("PositiveMatrixWithPrefactorArray") or [])
    channels = list(getattr(ext, "spin_channels", ()) or ())
    sample_values_by_block = getattr(ext, "sample_values_by_block", None) or {}
    original_twists = {_certification_real_label(t) for t in getattr(spec, "large_spin_twists", ())}
    skipped_ells = {int(ell) for ell in skip_ells}
    skipped_twists = {_certification_real_label(tw) for tw in skip_twists}

    details: list[dict] = []
    all_positive = True
    for block_index, block_doc in enumerate(blocks):
        if block_index >= len(channels):
            continue
        channel = channels[block_index]
        twist = getattr(channel, "twist", None)
        if twist is None:
            if int(channel.ell) <= max_spin:
                continue
            if int(channel.ell) in skipped_ells:
                continue
        else:
            if _certification_real_label(twist) in original_twists:
                continue
            if _certification_real_label(twist) in skipped_twists:
                continue

        sample_values = sample_values_by_block.get(block_index)
        if sample_values is None:
            detail = {
                "block_index": block_index,
                "ell": getattr(channel, "ell", None),
                **_channel_twist_metadata(channel),
                "passed": False,
                "reason": "missing_clean_sample_values",
                "n_test_points": 0,
                "basis": "clean_sample_values_extra",
                "extra": True,
            }
        else:
            detail = _sample_value_channel_detail(
                y_full,
                block_doc,
                sample_values,
                block_index=block_index,
                channel=channel,
                basis="clean_sample_values_extra",
                extra=True,
            )
            _with_polynomial_positivity(detail, block_doc, y_full, channel)
        if adaptive_stage is not None:
            detail["adaptive_stage"] = adaptive_stage
        detail.update(_channel_twist_metadata(channel))
        details.append(detail)
        all_positive = all_positive and detail["passed"]

    return all_positive, details


def _certify_normalized_moment_adaptive_sample_blocks(
    y_full: list[mp.mpf],
    problem,
    *,
    extra_spins: int,
    extra_twists: int,
    max_extra_spins: int,
    max_extra_twists: int,
    interstitial_twists: int = 0,
) -> tuple[bool, list[dict]]:
    initial_spins = max(0, int(extra_spins))
    initial_twists = max(0, int(extra_twists))
    max_spins = max(initial_spins, int(max_extra_spins))
    max_twists = max(initial_twists, int(max_extra_twists))

    ok, details = _certify_normalized_moment_extra_sample_blocks(
        y_full,
        problem,
        extra_spins=initial_spins,
        extra_twists=initial_twists,
        adaptive_stage="initial",
        interstitial_twists=interstitial_twists,
    )
    if not ok:
        return False, details
    if max_spins == initial_spins and max_twists == initial_twists:
        return ok, details

    checked_ells = {
        int(d["ell"])
        for d in details
        if d.get("ell") is not None and d.get("twist") is None
    }
    checked_twists = {
        _detail_twist_label(d)
        for d in details
        if d.get("twist") is not None
    }
    if max_spins > initial_spins:
        spin_ok, spin_details = _certify_normalized_moment_extra_sample_blocks(
            y_full,
            problem,
            extra_spins=max_spins,
            extra_twists=initial_twists,
            skip_ells=checked_ells,
            skip_twists=checked_twists,
            adaptive_stage="spin_wide",
            interstitial_twists=interstitial_twists,
        )
        details.extend(spin_details)
        if not spin_ok:
            return False, details
        checked_ells.update(
            int(d["ell"])
            for d in spin_details
            if d.get("ell") is not None and d.get("twist") is None
        )
        checked_twists.update(
            _detail_twist_label(d)
            for d in spin_details
            if d.get("twist") is not None
        )

    if max_twists > initial_twists:
        twist_ok, twist_details = _certify_normalized_moment_extra_sample_blocks(
            y_full,
            problem,
            extra_spins=max_spins,
            extra_twists=max_twists,
            skip_ells=checked_ells,
            skip_twists=checked_twists,
            adaptive_stage="twist_wide",
            interstitial_twists=interstitial_twists,
        )
        details.extend(twist_details)
        if not twist_ok:
            return False, details

    return True, details


def _interstitial_twist_set(spec, interstitial: int, *, extra_above: int = 0):
    """Base twists plus ``interstitial`` lines in each gap, plus above-top lines.

    Same placement as :func:`_build_extra_twist_channels` -- gaps divided into
    ``interstitial + 1`` pieces, and one gap's worth above the top slice at the
    same density -- but returned as an explicit list so it can be pushed through
    ``rebuild_normalized_moment_problem`` on the normalized-moment path, whose
    only tower knob was a double-twist COUNT and therefore could never express a
    line between two slices.
    """
    base = sorted(set(_certification_real_label(t) for t in getattr(spec, "large_spin_twists", ())))
    if not base:
        return None
    out = list(base)
    for lo, hi in zip(base[:-1], base[1:]):
        step = (hi - lo) / (interstitial + 1)
        out.extend(lo + k * step for k in range(1, interstitial + 1))
    top_gap = base[-1] - base[-2] if len(base) >= 2 else _certification_real_label(2)
    n_above = max(int(extra_above), 1)
    step = top_gap / (interstitial + 1)
    out.extend(
        base[-1] + k * step
        for k in range(1, (interstitial + 1) * n_above + 1)
    )
    return sorted(set(out))


def _certify_normalized_moment_extra_physical_blocks(
    y_full: list[mp.mpf],
    problem,
    *,
    extra_spins: int,
    extra_twists: int,
    n_test_points: int,
    delta_range: float,
    cert_q_order: int | None,
    skip_ells=(),
    skip_twists=(),
    adaptive_stage: str | None = None,
    interstitial_twists: int = 0,
    log_near_points: int = 0,
) -> tuple[bool, list[dict]]:
    if extra_spins <= 0 and extra_twists <= 0 and interstitial_twists <= 0:
        return True, []

    from virasoro_bootstrap.moments import rebuild_normalized_moment_problem

    spec = problem.spec
    spin_list = tuple(int(s) for s in spec.spin_list)
    max_spin = max(spin_list) if spin_list else -2
    ext_spins = spin_list + tuple(
        max_spin + 2 * (i + 1) for i in range(max(int(extra_spins), 0))
    )
    explicit_twists = _normalized_moment_extra_twist_set(spec, extra_twists, interstitial_twists)
    ext = rebuild_normalized_moment_problem(
        problem,
        spin_list=ext_spins,
        twists=explicit_twists,
    )
    ext = _with_cert_q_order(ext, cert_q_order)
    exact_channels = _build_exact_block_channels(ext)
    original_twists = {_certification_real_label(t) for t in getattr(spec, "large_spin_twists", ())}
    skipped_ells = {int(ell) for ell in skip_ells}
    skipped_twists = {_certification_real_label(tw) for tw in skip_twists}
    weights = _normalized_moment_cert_weights(ext)

    details: list[dict] = []
    all_positive = True
    for block_index, channel in enumerate(exact_channels):
        twist = getattr(channel, "twist", None)
        if twist is None:
            if int(channel.ell) <= max_spin:
                continue
            if int(channel.ell) in skipped_ells:
                continue
        else:
            if _certification_real_label(twist) in original_twists:
                continue
            if _certification_real_label(twist) in skipped_twists:
                continue
        detail = _normalized_moment_physical_channel_detail(
            y_full,
            ext,
            channel,
            block_index=block_index,
            n_test_points=n_test_points,
            delta_range=delta_range,
            weights=weights,
            extra=True,
            adaptive_stage=adaptive_stage,
            log_near_points=log_near_points,
        )
        detail["certificate_scope"] = "finite_true_block_grid"
        detail.update(_channel_twist_metadata(channel))
        details.append(detail)
        all_positive = all_positive and detail["passed"]

    return all_positive, details


def _certify_normalized_moment_included_physical_blocks(
    y_full: list[mp.mpf],
    problem,
    *,
    n_test_points: int,
    delta_range: float,
    cert_q_order: int | None,
    log_near_points: int = 0,
) -> tuple[bool, list[dict]]:
    """Recheck the INCLUDED moment channels against the true blocks.

    A normalized-moment direct-SDP problem normally takes the cheaper direct
    branch in :func:`certify_dual_from_sdpb_output`, which root-isolates the
    serialized PMP polynomials that SDPB saw.  That is useful for the finite
    interpolant problem, but it does not test interpolation error in the
    underlying Virasoro blocks.  In particular, a tower slice can be positive
    at every interpolation node while the true q-series block dips between
    them.

    ``force_exact_included`` requests this independent check.  The block
    sampler is exact at each requested point (fixed-spin recursion, and the
    true large-spin q-series), but a tower is a continuous ray, so this
    routine deliberately reports ``basis=physical_numeric_blocks`` and a
    finite-grid scope instead of pretending that a sampled tower is a global
    root-isolation certificate.  Fixed-spin entries still use the exact
    finite rational block at every point.
    """
    exact_problem = _with_cert_q_order(problem, cert_q_order)
    exact_channels = _build_exact_block_channels(exact_problem)
    weights = _normalized_moment_cert_weights(exact_problem)
    details: list[dict] = []
    all_positive = True
    for block_index, channel in enumerate(exact_channels):
        detail = _normalized_moment_physical_channel_detail(
            y_full,
            exact_problem,
            channel,
            block_index=block_index,
            n_test_points=n_test_points,
            delta_range=delta_range,
            weights=weights,
            extra=False,
            adaptive_stage="included_exact",
            log_near_points=log_near_points,
        )
        detail["certificate_scope"] = "finite_true_block_grid"
        detail["q_order"] = int(exact_problem.spec.q_order)
        details.append(detail)
        all_positive = all_positive and detail["passed"]
    # The final PMP entry is the normalized ratio's scalar inequality, not an
    # operator block. It has no physical block sampler, but it is still part of
    # the dual feasibility certificate and must be checked and reported alongside
    # the true channel grid.
    pmp_doc = getattr(getattr(exact_problem, "pmp_build", None), "doc", None) or {}
    pmp_blocks = list(pmp_doc.get("PositiveMatrixWithPrefactorArray") or [])
    scalar_detail = _certify_normalized_moment_scalar_block(
        y_full, exact_problem, block_index=len(pmp_blocks) - 1,
    )
    details.append(scalar_detail)
    all_positive = all_positive and scalar_detail["passed"]
    return all_positive, details


def _certify_normalized_moment_adaptive_physical_blocks(
    y_full: list[mp.mpf],
    problem,
    *,
    extra_spins: int,
    extra_twists: int,
    max_extra_spins: int,
    max_extra_twists: int,
    n_test_points: int,
    delta_range: float,
    cert_q_order: int | None,
    interstitial_twists: int = 0,
    log_near_points: int = 0,
) -> tuple[bool, list[dict]]:
    initial_spins = max(0, int(extra_spins))
    initial_twists = max(0, int(extra_twists))
    max_spins = max(initial_spins, int(max_extra_spins))
    max_twists = max(initial_twists, int(max_extra_twists))

    ok, details = _certify_normalized_moment_extra_physical_blocks(
        y_full,
        problem,
        extra_spins=initial_spins,
        extra_twists=initial_twists,
        n_test_points=n_test_points,
        delta_range=delta_range,
        cert_q_order=cert_q_order,
        adaptive_stage="initial",
        interstitial_twists=interstitial_twists,
        log_near_points=log_near_points,
    )
    if not ok:
        return False, details
    if max_spins == initial_spins and max_twists == initial_twists:
        return ok, details

    checked_ells = {
        int(d["ell"])
        for d in details
        if d.get("ell") is not None and d.get("twist") is None
    }
    checked_twists = {
        _detail_twist_label(d)
        for d in details
        if d.get("twist") is not None
    }
    if max_spins > initial_spins:
        spin_ok, spin_details = _certify_normalized_moment_extra_physical_blocks(
            y_full,
            problem,
            extra_spins=max_spins,
            extra_twists=initial_twists,
            n_test_points=n_test_points,
            delta_range=delta_range,
            cert_q_order=cert_q_order,
            skip_ells=checked_ells,
            skip_twists=checked_twists,
            adaptive_stage="spin_wide",
            interstitial_twists=interstitial_twists,
            log_near_points=log_near_points,
        )
        details.extend(spin_details)
        if not spin_ok:
            return False, details
        checked_ells.update(
            int(d["ell"])
            for d in spin_details
            if d.get("ell") is not None and d.get("twist") is None
        )
        checked_twists.update(
            _detail_twist_label(d)
            for d in spin_details
            if d.get("twist") is not None
        )

    if max_twists > initial_twists:
        twist_ok, twist_details = _certify_normalized_moment_extra_physical_blocks(
            y_full,
            problem,
            extra_spins=max_spins,
            extra_twists=max_twists,
            n_test_points=n_test_points,
            delta_range=delta_range,
            cert_q_order=cert_q_order,
            skip_ells=checked_ells,
            skip_twists=checked_twists,
            adaptive_stage="twist_wide",
            interstitial_twists=interstitial_twists,
            log_near_points=log_near_points,
        )
        details.extend(twist_details)
        if not twist_ok:
            return False, details

    return True, details


def _certify_normalized_moment_adaptive_pmp_blocks(
    y: list[mp.mpf],
    problem,
    *,
    extra_spins: int,
    extra_twists: int,
    max_extra_spins: int,
    max_extra_twists: int,
    n_test_points: int,
    delta_range: float,
    interstitial_twists: int = 0,
) -> tuple[bool, list[dict]]:
    """Adaptive normalized-moment certification.

    First check the caller's requested extras. If that passes, widen fixed spins
    while keeping the original tower depth; this catches high-spin low-twist
    negativity without paying to build many new tower slices. Only after that
    succeeds do we optionally widen tower twists.
    """
    initial_spins = max(0, int(extra_spins))
    initial_twists = max(0, int(extra_twists))
    max_spins = max(initial_spins, int(max_extra_spins))
    max_twists = max(initial_twists, int(max_extra_twists))

    ok, details = _certify_normalized_moment_extra_pmp_blocks(
        y,
        problem,
        extra_spins=initial_spins,
        extra_twists=initial_twists,
        n_test_points=n_test_points,
        delta_range=delta_range,
        adaptive_stage="initial",
        interstitial_twists=interstitial_twists,
    )
    if not ok:
        return False, details
    if max_spins == initial_spins and max_twists == initial_twists:
        return ok, details

    checked_ells = {
        int(d["ell"])
        for d in details
        if d.get("ell") is not None and d.get("twist") is None
    }
    checked_twists = {
        _detail_twist_label(d)
        for d in details
        if d.get("twist") is not None
    }
    if max_spins > initial_spins:
        spin_ok, spin_details = _certify_normalized_moment_extra_pmp_blocks(
            y,
            problem,
            extra_spins=max_spins,
            extra_twists=initial_twists,
            n_test_points=n_test_points,
            delta_range=delta_range,
            skip_ells=checked_ells,
            skip_twists=checked_twists,
            adaptive_stage="spin_wide",
            interstitial_twists=interstitial_twists,
        )
        details.extend(spin_details)
        if not spin_ok:
            return False, details
        checked_ells.update(
            int(d["ell"])
            for d in spin_details
            if d.get("ell") is not None and d.get("twist") is None
        )
        checked_twists.update(
            _detail_twist_label(d)
            for d in spin_details
            if d.get("twist") is not None
        )

    if max_twists > initial_twists:
        twist_ok, twist_details = _certify_normalized_moment_extra_pmp_blocks(
            y,
            problem,
            extra_spins=max_spins,
            extra_twists=max_twists,
            n_test_points=n_test_points,
            delta_range=delta_range,
            skip_ells=checked_ells,
            skip_twists=checked_twists,
            adaptive_stage="twist_wide",
            interstitial_twists=interstitial_twists,
        )
        details.extend(twist_details)
        if not twist_ok:
            return False, details

    return True, details


_CHIRAL_METHODS = {
    "zbar_half_chiral_from_cli": "virasoro",
    "chiral_global_control": "global",
    "chiral_virasoro_control": "virasoro",
    "chiral_virasoro_symbolic_control": "virasoro",
}


class _ExactChiralPrefactored:
    """Evaluate a chiral channel in h, without replacing it by a 2d scalar block."""

    def __init__(self, spec, mode, shift):
        self.spec = spec
        self.mode = mode
        self.shift = shift

    def evaluate(self, coordinate, value):
        from .controls.chiral_control import _chiral_block_coeffs

        spec = self.spec
        coefficients = _chiral_block_coeffs(
            value, mode=self.mode, h_ext=spec.h_ext, lam=spec.lam,
            c=spec.c, q_order=spec.q_order, chiral_norm=spec.chiral_norm,
        )
        return [[coefficient] for coefficient in coefficients]


class _ExactTowerPrefactored:
    """Duck-types ``SerializedDerivTable.evaluate`` for the exact large-spin
    tower block, so certification can re-check the dual functional against the
    *true* Virasoro chiral block instead of the npoles interpolant the SDP saw.

    The sampler already returns the library-normalized block (self-dual base
    ``16 q(1/2)/N``), which is exactly what the production tower channel feeds
    the SDP, so no further base rescaling is applied here:
    ``prefactored(L)[m][n] = sampler(L)[m][n]``.
    """

    def __init__(self, sampler, shift):
        self._sampler = sampler
        self.shift = _certification_real_label(shift, name="shift")

    def evaluate(self, coordinate, value):  # noqa: ARG002 - coord fixed by sampler
        L = _certification_real_label(value, name="spin")
        table = self._sampler(L)
        return [
            [table[m][n] for n in range(len(table[m]))]
            for m in range(len(table))
        ]


class _ExactFixedSpinPrefactored:
    """Duck-types ``SerializedDerivTable.evaluate`` for exact fixed-spin blocks.

    Wraps the numeric block sampler (_NumericBlockSampler or
    _RegularizedNumericSampler) so certification evaluates the true block
    recursion at each test Delta instead of the Lagrange polynomial.  The
    polynomial has ~1e-57 relative error at sample points which, amplified by
    |y| ~ 1e79, produces wrong-signed functional values.

    Non-kept (m,n) entries are zeroed to match the production keep-rule.
    """

    def __init__(self, sampler, kept, shift):
        self._sampler = sampler
        self._kept = frozenset(kept)
        self.shift = _certification_real_label(shift, name="shift")

    def evaluate(self, coordinate, value):  # noqa: ARG002 - coordinate baked into sampler
        Delta = _certification_real_label(value, name="Delta")
        tables, _ = self._sampler([Delta])
        table = tables[0]
        return [
            [
                table[m][n] if (m, n) in self._kept else mp.mpf(0)
                for n in range(len(table[m]))
            ]
            for m in range(len(table))
        ]


def _build_exact_block_channels(problem):
    """Return spin channels whose ``.prefactored`` evaluates the EXACT block
    (no npoles interpolation), for use in certification.

    - Fixed-spin Delta channels: evaluated pointwise through the exact table's
      numeric block sampler (:class:`_ExactFixedSpinPrefactored`), not the
      Lagrange interpolant; ``serialized`` carries the exact serialized table
      when one exists (the channel's own table at a recognized singular c).
    - Large-spin tower channels: the Virasoro chiral block is an infinite
      q-series with no finite serialized form, so it is evaluated pointwise via
      the exact :class:`~virasoro_blocks.large_spin._LargeSpinTwistSampler`
      wrapped in :class:`_ExactTowerPrefactored`.

    - Point-constraint channels (``problem.point_channels``) are already frozen
      at their exact point, so they are passed through unchanged and appended
      LAST, after the fixed-spin and tower channels.

    Chiral file/control problems are evaluated directly as chiral blocks in h;
    they do not use the fixed-spin two-dimensional scalar reconstruction.

    Fixed-spin/tower order and ``delta_min`` are preserved. For the global /
    numeric control methods (different block type) the channels are returned
    unchanged.
    """
    channels = list(getattr(problem, "spin_channels", ()) or ())
    if not channels:
        return channels
    chiral_mode = _CHIRAL_METHODS.get(getattr(problem, "resolved_method", None))
    if chiral_mode is not None:
        point_ids = {id(ch) for ch in (getattr(problem, "point_channels", ()) or ())}
        return [
            ch if id(ch) in point_ids else dataclasses.replace(
                ch, prefactored=_ExactChiralPrefactored(problem.spec, chiral_mode, ch.delta_min)
            )
            for ch in channels
        ]
    if getattr(problem, "resolved_method", None) in (
        "global_control",
        "numeric_virasoro_control",
    ):
        return channels

    from virasoro_bootstrap.identical_scalar import (
        IdenticalScalarSpinChannel,
        _NumericBlockSampler,
        _resolved_regularize,
        _central_charge_input,
    )
    from virasoro_blocks.large_spin import _LargeSpinTwistSampler
    from virasoro_blocks.serialization import serialize_deriv_tables
    from .point_constraint import _crossing_context_for_spec

    spec = problem.spec
    coordinate = getattr(spec, "coordinate", "zzb")
    c = _central_charge_input(spec.c)
    h_ext = _certification_real_label(spec.h_ext, name="h_ext")
    used_regularize = _resolved_regularize(c, spec.regularize)

    twists = (
        tuple(spec.large_spin_twists)
        if getattr(spec, "large_spin_lmin", None) is not None
        else ()
    )
    n_fixed = len(tuple(getattr(spec, "spin_list", ())))
    # point-constraint channels are frozen at their point (already exact); they are certified as they are
    point_channels = list(getattr(problem, "point_channels", ()) or ())
    block_channels = [ch for ch in channels if all(ch is not p for p in point_channels)]
    fixed_channels = block_channels[:n_fixed]
    existing_tower_channels = block_channels[n_fixed:n_fixed + len(twists)]

    # Reuse the prepared problem's exact tables, whose construction is costly
    # near c=1. Problems without them use the direct numeric samplers below.
    exact_tables_dict = getattr(problem, "exact_tables_by_ell", None)
    direct_numeric_sampler = None
    def make_direct_numeric_sampler():
        # Build all fixed-spin exact tables at once to attach numeric_sampler.
        # The sampler evaluates the true block recursion at any Delta, bypassing
        # the Lagrange polynomial whose ~1e-57 relative error gets amplified by
        # |y|~1e79.
        # Use a direct numeric sampler for lightweight diagnostic problems.  A
        # full symbolic table is still used when the prepared problem carries
        # one, but it is unnecessary for a finite-grid true-block check.
        h_externals = [h_ext] * 4
        z_shift = -2 * h_ext
        if used_regularize:
            from virasoro_blocks.regularization import _RegularizedNumericSampler
            return lambda ell: _RegularizedNumericSampler(
                c=c, coordinate=coordinate, chiral=False, spinning=False,
                projected=False, h_externals=h_externals,
                hbar_externals=h_externals, ell=ell, m_max=spec.lam,
                n_max=spec.lam, b=None, q_order=spec.q_order,
                chiral_norm=spec.chiral_norm, z_shift=z_shift, zbar_shift=z_shift,
            )
        else:
            return lambda ell: _NumericBlockSampler(
                coordinate, h_externals, ell, spec.lam, spec.lam,
                b=None, c=c, q_order=spec.q_order,
                chiral_norm=spec.chiral_norm, z_shift=z_shift,
            )
    if not exact_tables_dict:
        direct_numeric_sampler = make_direct_numeric_sampler()

    out = []
    for ch in fixed_channels:
        exact_table = exact_tables_dict.get(ch.ell) if exact_tables_dict else None
        if exact_table is not None:
            # At a recognized singular c (c = 1, 25, 26) the retained exact
            # table is a skeleton with ``numerators=None``: only its numeric
            # sampler is evaluated here, so keep the channel's own serialized
            # table rather than serializing the empty skeleton.
            serialized = (
                ch.serialized
                if getattr(exact_table, "numerators", None) is None
                else serialize_deriv_tables(
                    {coordinate: exact_table}, shift=ch.delta_min
                )
            )
            sampler = exact_table.numeric_sampler
        else:
            serialized = ch.serialized
            if direct_numeric_sampler is None:
                direct_numeric_sampler = make_direct_numeric_sampler()
            sampler = direct_numeric_sampler(ch.ell)
        # Match production's writer keep-rule exactly (the z<->zbar fold to
        # n <= m for scalar G on the non-folded coords). kept_pairs is the
        # single source of truth and is basis-aware.
        kept = set(kept_pairs(coordinate, spec.lam, basis="scalar_G", ell=ch.ell))
        prefactored = _ExactFixedSpinPrefactored(
            sampler,
            kept,
            ch.delta_min,
        )
        out.append(IdenticalScalarSpinChannel(
            ell=ch.ell, delta_min=ch.delta_min,
            serialized=serialized, prefactored=prefactored,
            crossing_context=_crossing_context_for_spec(spec, problem.resolved_method),
        ))

    z_shift = -2 * h_ext
    # A diagnostic may deliberately omit the expensive tower interpolation from
    # the problem build.  In that case there is no serialized source channel,
    # but the physical tower is completely specified by ``spec``; retain a
    # None placeholder for ``serialized`` and use the true sampler below.
    for tower_index, twist in enumerate(twists):
        source = (
            existing_tower_channels[tower_index]
            if tower_index < len(existing_tower_channels) else None
        )
        delta_min = (
            source.delta_min
            if source is not None
            else mp.mpf(spec.large_spin_lmin)
        )
        sampler = _LargeSpinTwistSampler(
            "virasoro", coordinate, [h_ext] * 4, spec.chiral_norm, z_shift,
            spec.lam, spec.lam, twist, c=c, q_order=spec.q_order,
        )
        out.append(IdenticalScalarSpinChannel(
            ell=(source.ell if source is not None else int(spec.large_spin_lmin)),
            delta_min=delta_min,
            serialized=(source.serialized if source is not None else None),
            prefactored=_ExactTowerPrefactored(sampler, delta_min),
            twist=twist,
        ))
    out.extend(point_channels)
    return out


def _certify_one_channel(
    y: list[mp.mpf],
    channel,
    F_pairs: list[tuple[int, int]],
    coordinate: str,
    n_test_points: int,
    delta_range: float,
) -> dict:
    """Evaluate the dual functional on a single spin channel and return its
    per-channel detail dict.

    A fixed-spin channel is sampled in ``Delta`` from its lower edge.  A
    large-spin tower channel (``channel.twist`` set) is sampled in the spin
    ``L`` from its floor at fixed twist; its detail reports the spin as
    ``min_spin`` / ``first_negative_spin`` and the dimension
    ``Delta = twist + L`` as ``min_delta`` / ``first_negative_delta``, so
    ``min_delta`` is a scaling dimension for every channel.

    Pulled out of :func:`certify_dual_functional` so the work for one channel is
    a self-contained unit that can be dispatched to a worker process.
    """
    block = channel.prefactored
    delta_min = _certification_real_label(channel.delta_min, name="delta_min")
    tower = getattr(channel, "twist", None) is not None
    twist = _certification_real_label(channel.twist) if tower else None
    # Generate test points within the requested range, including the threshold.
    # NOTE: a block-optimal (Bohr-Sommerfeld) node density would track the
    # block's base^x decay better than this fixed grid, but it must be paired
    # with evaluating the *true* block (numeric recursion) here -- otherwise
    # the finer/extended sampling hits small interpolation artifacts of the
    # npoles `channel.prefactored` polynomial and reports spurious failures.
    end = _certification_real_label(delta_range, name="delta_range")
    offsets = _linear_certification_offsets(n_test_points, end)
    # the channel's own variable: Delta for a fixed spin, the spin L for a tower slice
    test_points = [delta_min + x for x in offsets]

    def location(prefix, point):
        if not tower:
            return _certification_point_metadata(prefix + "delta", point)
        return {
            **_certification_point_metadata(prefix + "spin", point),
            **_certification_point_metadata(prefix + "delta", twist + point),
        }

    min_val = mp.mpf("+inf")
    min_point = delta_min
    n_negative = 0
    witness = {"first_negative_value": None}

    work_cache = {}
    for point in test_points:
        val = _evaluate_functional_on_block(
            y, block, F_pairs, point, coordinate, _work_cache=work_cache,
        )
        real_val = mp.re(val)
        if real_val < min_val or mp.isnan(real_val):
            min_val = real_val
            min_point = point
        if not _finite_real_nonnegative(val):
            n_negative += 1
            if witness["first_negative_value"] is None:
                _record_negative_sample(witness, val)
                if witness["first_negative_value"] is not None:
                    witness.update(location("first_negative_", point))

    return {
        "ell": channel.ell,
        **_channel_twist_metadata(channel),
        "variable": "L" if tower else "Delta",
        "passed": n_negative == 0,
        **_min_value_fields(min_val),
        **location("min_", min_point),
        "n_negative": n_negative,
        "n_test_points": n_test_points,
        **witness,
    }


# Shared read-only state for parallel certification workers. Populated once per
# worker process by ``_certify_worker_init`` so the (potentially large) dual
# vector ``y`` is shipped to each worker a single time, rather than re-pickled
# for every channel task.
_CERT_WORKER_STATE: dict = {}


def _certify_worker_init(cert_dps, y, F_pairs, coordinate, n_test_points, delta_range):
    """ProcessPool initializer: set mpmath precision and cache shared payload."""
    mp.mp.dps = cert_dps
    _CERT_WORKER_STATE["y"] = y
    _CERT_WORKER_STATE["F_pairs"] = F_pairs
    _CERT_WORKER_STATE["coordinate"] = coordinate
    _CERT_WORKER_STATE["n_test_points"] = n_test_points
    _CERT_WORKER_STATE["delta_range"] = delta_range


def _certify_channel_task(indexed_channel):
    """ProcessPool task: certify one ``(index, channel)`` against shared state."""
    idx, channel = indexed_channel
    st = _CERT_WORKER_STATE
    detail = _certify_one_channel(
        st["y"], channel, st["F_pairs"], st["coordinate"],
        st["n_test_points"], st["delta_range"],
    )
    return idx, detail


def _refuse_modular(*, problem=None, spin_channels=()):
    """These certifiers check the sphere four-point problems; refuse a modular (torus) problem clearly."""
    from .modular_crossing import ModularGapProblem, ModularSpinChannel

    if isinstance(problem, ModularGapProblem) or any(isinstance(ch, ModularSpinChannel) for ch in spin_channels):
        raise TypeError("dual-functional certification covers the sphere four-point problems; a modular gap "
                        "problem has none yet (check its bound against a larger spin_list instead)")


def certify_dual_functional(
    y: list[mp.mpf],
    spin_channels,
    F_pairs: list[tuple[int, int]],
    *,
    coordinate: str = "zzb",
    n_test_points: int = 100,
    delta_range: float = 30.0,
    num_threads: int = 1,
) -> tuple[bool, list[dict]]:
    """Check positivity of the dual functional on all spin channels.

    Parameters
    ----------
    y : list of mpf
        The dual solution vector from SDPB.
    spin_channels : sequence
        Iterable of objects with ``.ell``, ``.delta_min``, and
        ``.prefactored`` (a ``SerializedDerivTable`` with the full
        poles, before ``drop_complex_pole_pairs``).
    F_pairs : list of (int, int)
        The F-vector index pairs.
    n_test_points : int
        Number of Δ values to test per spin channel.
    delta_range : float
        How far above Δ_min to test.
    num_threads : int
        Number of worker processes to fan the per-channel positivity checks
        across. ``<= 1`` (or a single channel) runs serially in-process. This
        reuses the same parallelism knob as the block build
        (``IdenticalScalarGapSpec.num_threads``); no separate certification
        thread count is introduced. Channels are independent, so results are
        reassembled in the original order and the pass/fail verdict is identical
        to the serial path.

    Returns
    -------
    (all_positive, details) where all_positive is True if every
    test point passed, and details is a list of per-channel dicts with
    ``ell``, ``twist`` (None for a fixed spin), ``variable`` (``"Delta"``, or
    ``"L"`` for a tower slice), ``passed``, ``min_value`` (a float when the
    double range holds it, else its decimal string), ``min_value_text``
    (always the string), ``min_delta`` (the scaling dimension of the minimum;
    a tower slice also reports its spin ``min_spin``), ``n_negative`` and
    ``n_test_points``.  :func:`worst_certification_detail` picks the channel
    where the functional is smallest.
    """
    delta_range = _validate_certification_grid(n_test_points, delta_range)
    num_threads = _integer_field("num_threads", num_threads)
    spin_channels = list(spin_channels)     # (after the controls are validated: an invalid one consumes nothing)
    _refuse_modular(spin_channels=spin_channels)
    if not F_pairs or len(y) != len(F_pairs):
        hint = ""
        if F_pairs and len(y) == len(F_pairs) - 1:
            hint = (
                ": y looks like SDPB's output with the normalization pivot eliminated; "
                "load it with load_y_vector(sdpb_out_dir, full=True)"
            )
        raise ValueError(
            f"dual vector length ({len(y)}) must match the non-empty F_pairs ({len(F_pairs)}){hint}"
        )
    y = [_stored_real(value) for value in y]
    channels = list(spin_channels)
    n_workers = max(1, min(usable_workers(num_threads), len(channels)))

    if n_workers <= 1 or len(channels) <= 1:
        details = [
            _certify_one_channel(
                y, channel, F_pairs, coordinate, n_test_points, delta_range
            )
            for channel in channels
        ]
        return all(d["passed"] for d in details), details

    cert_dps = mp.mp.dps
    try:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            **pool_initializer_kwargs(
                _certify_worker_init, (cert_dps, y, F_pairs, coordinate, n_test_points, delta_range)),
        ) as pool:
            indexed_results = list(
                pool.map(_certify_channel_task, list(enumerate(channels)))
            )
        indexed_results.sort(key=lambda item: item[0])
        details = [detail for _, detail in indexed_results]
    except Exception as exc:  # pragma: no cover - environment-dependent fallback
        # Pickling failures (unpicklable channel/block) or a broken pool must
        # never break a scan: fall back to the serial path, which is exact.
        warnings.warn(
            f"Parallel certification failed ({exc!r}); falling back to serial.",
            stacklevel=2,
        )
        details = [
            _certify_one_channel(
                y, channel, F_pairs, coordinate, n_test_points, delta_range
            )
            for channel in channels
        ]

    return all(d["passed"] for d in details), details



def _validate_certification_grid(n_test_points, delta_range):
    if isinstance(n_test_points, bool) or not isinstance(n_test_points, Integral) or n_test_points < 1:
        raise ValueError("n_test_points must be a positive integer")
    try:
        end = _certification_real_label(delta_range, name="delta_range")
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        raise ValueError("delta_range must be finite and non-negative")
    if end < 0:
        raise ValueError("delta_range must be finite and non-negative")
    return end


def _certification_count(name, value):
    value = _integer_field(name, value)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _certification_delta_range(delta_range, spec):
    """The Delta range above each channel edge the certificate tests: at least ``delta_range``, and at large c
    past the region where the blocks change shape (the Liouville threshold (c - 1)/12): at c = 1000 an
    order-reduced interpolant left the true block near x ~ 80, far beyond a fixed 30."""
    c = getattr(spec, "c", None)
    try:
        from virasoro_blocks.numeric_input import literal_components

        c = literal_components(c)[0] if c is not None else None
    except (TypeError, ValueError):
        c = None
    if c is None or c <= 1:
        return delta_range
    # Retain both the caller's endpoint and the source of the derived edge.
    return max(delta_range, (c - 1) / 4 + 10)


def certify_dual_from_sdpb_output(
    problem,
    sdpb_out_dir,
    *,
    n_test_points: int = 100,
    delta_range: float = 30.0,
    warn: bool = True,
    extra_spins: int = 0,
    extra_twists: int = 0,
    adaptive_moment: bool = False,
    adaptive_moment_extra_spins: int = 20,
    adaptive_moment_extra_twists: int = 2,
    physical_moment_blocks: bool = True,
    moment_cert_q_order: int | None = None,
    exact_blocks: bool = True,
    interstitial_twists: int = 0,
    force_exact_included: bool = False,
    log_near_points: int = 0,
    num_threads: int | None = None,
    q_gate: str = "warn",
) -> tuple[bool, list[dict]]:
    """Convenience: load y and certify the functional for a problem.

    Parameters
    ----------
    problem : IdenticalScalarGapProblem
        The problem that was solved.
    sdpb_out_dir : path-like
        SDPB output directory containing y.txt.
    warn : bool
        If True, emit a warning when certification fails.
    extra_spins : int
        Number of extra even spins beyond the problem's spin_list to check.
        They use ell plus the common twist floor of the fixed-spin channels as their lower edge when at least two
        spinning channels share it (capped at the lowest tower twist), and the unitarity bound otherwise.
    extra_twists : int
        Extra large-spin tower coverage. On the gap path: the number of
        intermediate twists between the discrete double-twist slices the SDP
        constrained -- ``extra_twists`` evenly-spaced values in each gap between
        consecutive base twists (plus one gap above the top slice), each
        evaluated on the exact Virasoro tower block for L >= Lmax. A failure
        there is a genuine robustness gap (the discrete-twist relaxation was too
        coarse), not an interpolation artifact. On the normalized-moment path it
        instead counts extra double-twist slices ABOVE the top one (it extends
        the double-twist count; use ``interstitial_twists`` for lines between
        slices), and a control-method (global / numeric control) moment problem
        refuses extra coverage (``NotImplementedError``). No effect for
        problems without a tower.
    adaptive_moment : bool
        For normalized-moment PMP problems, first check ``extra_spins`` /
        ``extra_twists``. If that passes, widen to
        ``adaptive_moment_extra_spins`` / ``adaptive_moment_extra_twists`` and
        check only the newly-added blocks. This catches high-spin low-twist
        negativity that can appear beyond the first few extra fixed spins while
        still failing fast on obviously under-covered functionals.
    adaptive_moment_extra_spins : int
        Widened fixed-spin count for ``adaptive_moment``. Interpreted as the
        number of even spins beyond ``max(spec.spin_list)``.
    adaptive_moment_extra_twists : int
        Widened double-twist slice count for ``adaptive_moment``.
    physical_moment_blocks : bool
        For normalized-moment direct-SDP problems, certify extra/adaptive
        coverage against freshly-evaluated numeric blocks on a grid rather than
        the solver's npoles interpolants or PMP polynomial coefficients. This
        keeps npoles as a solver compression parameter, not part of the
        mathematical certificate.
    moment_cert_q_order : int, optional
        q-series truncation order for ``physical_moment_blocks``. Defaults to
        the problem's ``spec.q_order``. Set this higher than the solve q_order
        to test q-order convergence of the certificate without rerunning SDPB.
    interstitial_twists : int
        Number of tower twist lines to check BETWEEN each pair of double-twist
        slices (and an equal density above the top slice). The SDP constrains
        positivity only on the discrete slices, while unitarity permits
        operators at any tau >= 0 at large spin, so a dip in between is a real
        gap in the relaxation. Zero (the default) disables checks between
        slices on the normalized-moment path.
    force_exact_included : bool
        Skip the direct-SDP shortcut for the INCLUDED blocks of a
        normalized-moment problem and check them on freshly evaluated true
        blocks instead. Requires ``physical_moment_blocks=True`` (the default);
        without it the included blocks fall back to the PMP-polynomial check.
        ``exact_blocks`` has no effect on this path. The shortcut otherwise
        checks the included blocks against the SDP's own ``block_data_*.json``
        -- the npoles interpolant the solve used -- so block interpolation
        error is invisible to it. Costs a block rebuild; off by default so
        existing verdicts are unchanged until the cost is measured.
    exact_blocks : bool
        If True (default), re-evaluate the dual functional against the EXACT
        blocks (the numeric block recursion for the fixed-spin Delta
        channels; the true Virasoro chiral q-series for the large-spin tower)
        rather than the npoles interpolants the SDP solved against. This catches
        interpolation error in the SDP blocks. Set False to certify against the
        same interpolated blocks pmp2sdp received. Ignored
        for the global / numeric control methods, whose channels are used as-is.
    num_threads : int, optional
        Worker-process count for the per-channel positivity checks. When None
        (default) it is taken from ``problem.spec.num_threads`` (the same knob
        that parallelizes the block build), so no separate certification thread
        count is introduced. ``<= 1`` runs serially.
    q_gate : str
        q-order convergence gate on the problem's blocks: ``"warn"`` (default)
        prints a warning when the estimated q-truncation accuracy of the
        underlying blocks is poor (a certificate against q-starved blocks is
        exactly the silent-invalid scenario certification exists to prevent);
        ``"strict"`` raises :class:`virasoro_blocks.qgate.QConvergenceError`
        instead; ``"off"`` skips the probe. The gate never alters the
        returned ``(ok, details)``.
    """
    _refuse_modular(problem=problem)
    for name, value in (("warn", warn), ("adaptive_moment", adaptive_moment),
                        ("physical_moment_blocks", physical_moment_blocks), ("exact_blocks", exact_blocks),
                        ("force_exact_included", force_exact_included)):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be True or False")
    if not isinstance(q_gate, str) or q_gate not in {"off", "warn", "strict"}:
        raise ValueError("q_gate must be 'off', 'warn', or 'strict'")
    delta_range = _validate_certification_grid(n_test_points, delta_range)
    extra_spins = _certification_count("extra_spins", extra_spins)
    extra_twists = _certification_count("extra_twists", extra_twists)
    adaptive_moment_extra_spins = _certification_count("adaptive_moment_extra_spins", adaptive_moment_extra_spins)
    adaptive_moment_extra_twists = _certification_count("adaptive_moment_extra_twists", adaptive_moment_extra_twists)
    interstitial_twists = _certification_count("interstitial_twists", interstitial_twists)
    log_near_points = _certification_count("log_near_points", log_near_points)
    if moment_cert_q_order is not None:
        moment_cert_q_order = _certification_count("moment_cert_q_order", moment_cert_q_order)
    spec = getattr(problem, "spec", None)
    cert_threads = num_threads if num_threads is not None else getattr(spec, "num_threads", 1)
    if cert_threads is None:
        cert_threads = 1
    cert_threads = _integer_field("num_threads", cert_threads)
    base_dps = max(mp.mp.dps, int(getattr(problem.spec, "dps", mp.mp.dps)))
    # q-order convergence gate on the blocks underlying this certificate.
    delta_range = _certification_delta_range(delta_range, spec)
    if q_gate != "off" and spec is not None and getattr(spec, "c", None) is not None:
        from virasoro_blocks.qgate import gate_q_convergence

        with mp.workdps(base_dps):
            gate_q_convergence(
                c=spec.c, h_externals=[_certification_real_label(spec.h_ext, name="h_ext")] * 4,
                q_order=int(spec.q_order), n_terms=int(spec.lam) + 2,
                coordinate=getattr(spec, "coordinate", "zzb"),
                chiral_norm=getattr(spec, "chiral_norm", None),
                mode=q_gate, label="certification",
            )
    # Load y at base dps to estimate the scale, then boost precision if needed.
    # With |y_max| ~ 10^p, evaluating Σ y × block requires ~p extra digits beyond
    # base_dps to resolve the functional after cancellation.
    with mp.workdps(base_dps):
        try:
            y_for_scale = load_y_vector(sdpb_out_dir)
        except NonFiniteDualVectorError as exc:
            if warn:
                warnings.warn(
                    f"Dual functional certification FAILED: {exc}.", stacklevel=2
                )
            return False, [{
                "ell": None,
                "twist": None,
                "passed": False,
                "reason": "non_finite_dual_vector",
                **_min_value_fields(mp.nan),
                "n_test_points": 0,
            }]
        y_max_scale = max((abs(yi) for yi in y_for_scale), default=mp.mpf(0))
        # mp.log10 rather than float(): a finite mpf far beyond the double range
        # (``1e500``) must not overflow the precision estimate
        y_log10 = (
            max(0, int(mp.floor(mp.log10(y_max_scale))) + 1) if y_max_scale > 0 else 0
        )
    cert_dps = base_dps + max(0, y_log10 - 20)
    with mp.workdps(cert_dps):
        # re-read y at the certification precision: converting the base-dps values only pads digits
        y_sdpb = list(load_y_vector(sdpb_out_dir)) if cert_dps > base_dps else list(y_for_scale)
        y = list(y_sdpb)
        del y_for_scale
        n_constraints = len(list(problem.F_pairs))
        # the full functional has the PMP vector length: n_constraints, plus the moment component of a
        # normalized-moment problem (whose SDPB output is then n_constraints long with the pivot eliminated)
        pmp_doc_full = getattr(getattr(problem, "pmp_build", None), "doc", None)
        full_len = (_pmp_vector_length(pmp_doc_full) if pmp_doc_full is not None else None) or n_constraints
        # the normalization n . y = 1 is what makes a non-negative functional a
        # certificate; an unnormalized one (e.g. all zeros) proves nothing
        violation = _normalization_violation(
            _reconstruct_full_y(list(y), sdpb_out_dir) if len(y) < full_len else y,
            problem,
            sdpb_out_dir,
        )
        if violation is not None:
            if warn:
                warnings.warn(
                    "Dual functional certification FAILED: the functional violates "
                    f"its normalization (n . y = {mp.nstr(_stored_real(violation['normalization_value']), 6)}, "
                    "expected 1).",
                    stacklevel=2,
                )
            return False, [violation]
        if len(y) != n_constraints or len(y) < full_len:
            # pmp2sdp eliminates one component via the normalization constraint.
            # Reconstruct the full vector.
            if len(y) < full_len:
                y = _reconstruct_full_y(y, sdpb_out_dir)
            if len(y) != n_constraints:
                pmp_build = getattr(problem, "pmp_build", None)
                pmp_doc = getattr(pmp_build, "doc", None)
                pmp_len = _pmp_vector_length(pmp_doc) if pmp_doc is not None else None
                if pmp_doc is not None and pmp_len == len(y):
                    if (
                        _is_normalized_moment_problem(problem)
                        and _has_direct_sdp_data(sdpb_out_dir)
                        and not force_exact_included
                    ):
                        all_positive, details = _certify_direct_sdp_included_blocks(
                            y_sdpb,
                            problem,
                            sdpb_out_dir,
                            # y (not y_sdpb) is the reconstructed full-length
                            # vector; this branch is only reached when
                            # pmp_len == len(y), i.e. exactly the length
                            # block_positivity needs.
                            y_full=y,
                        )
                        if adaptive_moment:
                            if physical_moment_blocks:
                                extra_positive, extra_details = (
                                    _certify_normalized_moment_adaptive_physical_blocks(
                                        y,
                                        problem,
                                        extra_spins=extra_spins,
                                        extra_twists=extra_twists,
                                        max_extra_spins=adaptive_moment_extra_spins,
                                        max_extra_twists=adaptive_moment_extra_twists,
                                        n_test_points=n_test_points,
                                        delta_range=delta_range,
                                        cert_q_order=moment_cert_q_order,
                                        interstitial_twists=interstitial_twists,
                                        log_near_points=log_near_points,
                                    )
                                )
                            else:
                                extra_positive, extra_details = (
                                    _certify_normalized_moment_adaptive_sample_blocks(
                                        y,
                                        problem,
                                        extra_spins=extra_spins,
                                        extra_twists=extra_twists,
                                        max_extra_spins=adaptive_moment_extra_spins,
                                        max_extra_twists=adaptive_moment_extra_twists,
                                        interstitial_twists=interstitial_twists,
                                    )
                                )
                        else:
                            if physical_moment_blocks:
                                extra_positive, extra_details = (
                                    _certify_normalized_moment_extra_physical_blocks(
                                        y,
                                        problem,
                                        extra_spins=extra_spins,
                                        extra_twists=extra_twists,
                                        n_test_points=n_test_points,
                                        delta_range=delta_range,
                                        cert_q_order=moment_cert_q_order,
                                        interstitial_twists=interstitial_twists,
                                        log_near_points=log_near_points,
                                    )
                                )
                            else:
                                extra_positive, extra_details = (
                                    _certify_normalized_moment_extra_sample_blocks(
                                        y,
                                        problem,
                                        extra_spins=extra_spins,
                                        extra_twists=extra_twists,
                                        interstitial_twists=interstitial_twists,
                                    )
                                )
                        details.extend(extra_details)
                        all_positive = all_positive and extra_positive
                    elif (
                        _is_normalized_moment_problem(problem)
                        and _has_direct_sdp_data(sdpb_out_dir)
                        and force_exact_included
                        and physical_moment_blocks
                    ):
                        # Evaluate the included fixed-spin and tower blocks
                        # directly, then extend coverage if requested.
                        exact_positive, exact_details = (
                            _certify_normalized_moment_included_physical_blocks(
                                y,
                                problem,
                                n_test_points=n_test_points,
                                delta_range=delta_range,
                                cert_q_order=moment_cert_q_order,
                                log_near_points=log_near_points,
                            )
                        )
                        details = exact_details
                        all_positive = exact_positive
                        if adaptive_moment:
                            extra_positive, extra_details = (
                                _certify_normalized_moment_adaptive_physical_blocks(
                                    y,
                                    problem,
                                    extra_spins=extra_spins,
                                    extra_twists=extra_twists,
                                    max_extra_spins=adaptive_moment_extra_spins,
                                    max_extra_twists=adaptive_moment_extra_twists,
                                    n_test_points=n_test_points,
                                    delta_range=delta_range,
                                    cert_q_order=moment_cert_q_order,
                                    interstitial_twists=interstitial_twists,
                                    log_near_points=log_near_points,
                                )
                            )
                        else:
                            extra_positive, extra_details = (
                                _certify_normalized_moment_extra_physical_blocks(
                                    y,
                                    problem,
                                    extra_spins=extra_spins,
                                    extra_twists=extra_twists,
                                    n_test_points=n_test_points,
                                    delta_range=delta_range,
                                    cert_q_order=moment_cert_q_order,
                                    interstitial_twists=interstitial_twists,
                                    log_near_points=log_near_points,
                                )
                            )
                        details.extend(extra_details)
                        all_positive = all_positive and extra_positive
                    else:
                        all_positive, details = _certify_pmp_dual_functional(
                            y,
                            pmp_doc,
                            spin_channels=getattr(problem, "spin_channels", ()),
                            n_test_points=n_test_points,
                            delta_range=delta_range,
                        )
                        if _is_normalized_moment_problem(problem):
                            if adaptive_moment:
                                extra_positive, extra_details = (
                                    _certify_normalized_moment_adaptive_pmp_blocks(
                                        y,
                                        problem,
                                        extra_spins=extra_spins,
                                        extra_twists=extra_twists,
                                        max_extra_spins=adaptive_moment_extra_spins,
                                        max_extra_twists=adaptive_moment_extra_twists,
                                        n_test_points=n_test_points,
                                        delta_range=delta_range,
                                        interstitial_twists=interstitial_twists,
                                    )
                                )
                            else:
                                extra_positive, extra_details = (
                                    _certify_normalized_moment_extra_pmp_blocks(
                                        y,
                                        problem,
                                        extra_spins=extra_spins,
                                        extra_twists=extra_twists,
                                        n_test_points=n_test_points,
                                        delta_range=delta_range,
                                        interstitial_twists=interstitial_twists,
                                    )
                                )
                            details.extend(extra_details)
                            all_positive = all_positive and extra_positive
                    if not all_positive and warn:
                        _warn_certification_failed(problem, details, "PMP positivity ")
                    return all_positive, details
                message = (
                    f"y vector length ({len(y)}) != number of F-pairs ({n_constraints}) "
                    "even after reconstruction"
                )
                if warn:
                    warnings.warn(f"{message}. Skipping certification.", stacklevel=2)
                return False, [{
                    "ell": None,
                    "twist": None,
                    "passed": False,
                    "reason": "dual_vector_length_mismatch",
                    "message": message,
                    **_min_value_fields(mp.nan),
                    "n_test_points": 0,
                }]

        # Re-evaluate the functional at the same working precision that built
        # the block tables. Otherwise diagnostics wrappers can silently fall
        # back to mpmath's default 15-digit context and report bogus failures.
        if exact_blocks:
            base_channels = _build_exact_block_channels(problem)
        else:
            base_channels = list(problem.spin_channels)

        extra_channels = []
        if extra_spins > 0:
            extra_channels = _build_extra_spin_channels(
                problem, extra_spins, exact_blocks=exact_blocks
            )

        extra_twist_channels = []
        if extra_twists > 0:
            extra_twist_channels = _build_extra_twist_channels(
                problem, extra_twists
            )

        all_channels = list(base_channels) + extra_channels + extra_twist_channels
        all_positive, details = certify_dual_functional(
            y,
            all_channels,
            list(problem.F_pairs),
            coordinate=getattr(problem.spec, "coordinate", "zzb"),
            n_test_points=n_test_points,
            delta_range=delta_range,
            num_threads=cert_threads,
        )
    if not all_positive and warn:
        _warn_certification_failed(problem, details)
    return all_positive, details


def _certified_result(problem) -> str:
    """What a passing certificate of *problem* establishes, for the failure warning."""
    kind = getattr(problem, "problem_kind", None) or "gap"
    if kind in {"gap", "fixed_ope_feasibility"}:
        return "the dual_feasible (exclusion) result"
    if kind.startswith("ope_"):
        return "the OPE-coefficient bound"
    if "moment" in kind or "ratio" in kind:
        return "the moment bound"
    return f"the {kind} result"


def _failed_location(detail) -> str:
    """Where a failed detail lies: a tower slice by its twist and spin floor, a spin, or an SDP block."""
    if detail.get("twist") is not None:
        twist = detail.get("twist_label", detail["twist"])
        floor = f", L >= {detail['ell']}" if detail.get("ell") is not None else ""
        return f"tower twist {twist}{floor}"
    if detail.get("ell") is not None:
        return f"spin {detail['ell']}"
    if detail.get("block_index") is not None:
        return f"block {detail['block_index']}"
    return str(detail.get("reason", "unlabeled check"))


def _warn_certification_failed(problem, details, what: str = "") -> None:
    locations = [_failed_location(d) for d in details if not d["passed"]]
    result = _certified_result(problem)
    kind = getattr(problem, "problem_kind", None) or "gap"
    warnings.warn(
        f"Dual functional {what}certification FAILED on {', '.join(locations)}. "
        f"{result[0].upper()}{result[1:]} of this {kind} problem is not certified on the checked channels.",
        stacklevel=3,
    )


def _build_extra_spin_channels(problem, extra_spins: int, *, exact_blocks: bool = False):
    """Build spin channels for spins not in the problem's spin_list.

    When ``exact_blocks`` is True the Virasoro extra-spin channels use the exact
    (non-interpolated) serialized table regardless of the problem's solve
    method, matching the exact-block certification path.


    This includes:
    1. Even spins in gaps between non-consecutive entries in spin_list
       (e.g., spins 22-98 when spin_list has [20, 100]).
    2. Even spins above max(spin_list) up to extra_spins beyond max.

    Each extra spin uses the problem's spinning-twist convention as its lower
    edge: if at least two included spinning channels share a uniform twist
    floor (delta_min - ell = const), extra spins use ell + that floor, capped
    at the lowest tower twist; otherwise the unitarity bound (delta_min = ell). Block type matches the problem (global
    control vs Virasoro), so the certificate is checked against the same surface.
    """
    from virasoro_bootstrap.identical_scalar import (
        IdenticalScalarSpinChannel,
        _build_exact_scalar_tables,
        _build_exact_scalar_table,
        _normalize_npoles,
        _resolved_regularize,
        _central_charge_input,
        _resolved_method,
        _serialize_spin_channel,
    )
    from virasoro_blocks.serialization import serialize_deriv_tables
    from .point_constraint import _crossing_context_for_spec

    # A chiral channel constrains the h ray, not a truncated list of 2d spins.
    if getattr(problem, "resolved_method", None) in _CHIRAL_METHODS:
        return []
    spec = problem.spec
    # fixed-spin channels only: a tower channel carries ell = int(Lmax) (odd Lmax made the "above" spins odd) and a
    # point channel is not a spin sector of its own
    point_ids = {id(ch) for ch in (getattr(problem, "point_channels", ()) or ())}
    existing_spins = sorted(set(
        ch.ell for ch in problem.spin_channels
        if getattr(ch, "twist", None) is None and id(ch) not in point_ids
    )) or sorted(set(ch.ell for ch in problem.spin_channels))
    max_existing = max(existing_spins)

    # Build the set of extra spins to check:
    # (a) Sample gaps between non-consecutive spins in spin_list.
    #     For each gap, pick up to `extra_spins` evenly-spaced even spins.
    gap_ells = []
    for i in range(len(existing_spins) - 1):
        lo = existing_spins[i]
        hi = existing_spins[i + 1]
        if hi - lo > 2:  # there's a gap
            all_in_gap = list(range(lo + 2, hi, 2))
            if len(all_in_gap) <= extra_spins:
                gap_ells.extend(all_in_gap)
            else:
                # Evenly sample extra_spins from the gap
                step = len(all_in_gap) / extra_spins
                gap_ells.extend(
                    all_in_gap[int(j * step)] for j in range(extra_spins)
                )

    # (b) Spins above max_existing
    above_ells = [max_existing + 2 * (i + 1) for i in range(extra_spins)]

    extra_ells = sorted(set(gap_ells + above_ells))

    channels = []
    c = _central_charge_input(spec.c)
    h_ext = _certification_real_label(spec.h_ext, name="h_ext")
    coordinate = getattr(spec, "coordinate", "zzb")
    used_regularize = _resolved_regularize(c, spec.regularize)
    resolved_method = _resolved_method(spec.npoles)
    is_global = problem.resolved_method == "global_control"
    is_numeric_virasoro = problem.resolved_method == "numeric_virasoro_control"

    # Infer the spinning-twist floor from the included spinning channels so
    # extra spins are tested on the same lower edge the SDP imposed (otherwise a
    # twist floor would make us check the excluded twist-0 region and report
    # spurious certificate failures).
    # (fixed-spin channels only: a large-spin tower channel carries ell = int(Lmax), delta_min = Lmax, i.e.
    # twist 0, and a point channel sits at its point -- either collapsed the floor to unitarity)
    # A floor is a convention only when at least two spinning channels share it: one per-spin override
    # (delta_min_by_spin = {2: 3}) says nothing about the other spins, whose edge is the unitarity bound, and
    # taking it as the floor left the twists below it unchecked.  A tower imposes positivity down to its lowest
    # twist, so the floor never exceeds that either.
    point_ids = {id(ch) for ch in (getattr(problem, "point_channels", ()) or ())}
    spinning = [
        ch for ch in problem.spin_channels
        if ch.ell > 0 and getattr(ch, "twist", None) is None and id(ch) not in point_ids
    ]
    spin_twists = {
        _certification_real_label(ch.delta_min, name="lower edge") - ch.ell
        for ch in spinning
    }
    if len(spin_twists) == 1 and len(spinning) >= 2:
        twist_floor = next(iter(spin_twists))
    else:
        twist_floor = 0
    if getattr(spec, "large_spin_lmin", None) is not None and spec.large_spin_twists:
        lowest_tower_twist = min(_certification_real_label(t) for t in spec.large_spin_twists)
        twist_floor = max(0, min(twist_floor, lowest_tower_twist))

    for ell in extra_ells:
        delta_min = ell + twist_floor
        if is_global:
            from virasoro_bootstrap.controls.global_control import (
                build_global_scalar_spin_channel,
            )
            channels.append(build_global_scalar_spin_channel(
                coordinate=coordinate,
                ell=ell, delta_min=delta_min, h_ext=h_ext, lam=spec.lam,
                chiral_norm=spec.chiral_norm,
                n_sample_points=spec.n_sample_points,
            ))
            continue
        if is_numeric_virasoro:
            from virasoro_bootstrap.controls.global_control import (
                build_numeric_virasoro_scalar_spin_channel,
            )
            channels.append(build_numeric_virasoro_scalar_spin_channel(
                coordinate=coordinate,
                c=c,
                q_order=spec.q_order,
                ell=ell,
                delta_min=delta_min,
                h_ext=h_ext,
                lam=spec.lam,
                chiral_norm=spec.chiral_norm,
                n_sample_points=spec.n_sample_points,
            ))
            continue
        sample_via = getattr(spec, "sample_via", "numeric")
        if not exact_blocks and resolved_method in ("interpolation", "full"):
            exact_table = _build_exact_scalar_tables(
                c,
                h_ext,
                coordinate,
                (ell,),
                spec.lam,
                spec.q_order,
                used_regularize=used_regularize,
                resolved_method=resolved_method,
                chiral_norm=spec.chiral_norm,
                num_threads=1,
                sample_via=sample_via,
                pole_selection=spec.pole_selection,
            )[ell]
            serialized = _serialize_spin_channel(
                exact_table,
                coordinate=coordinate,
                ell=ell,
                delta_min=delta_min,
                resolved_method=resolved_method,
                npoles=_normalize_npoles(spec.npoles),
                n_sample_points=spec.n_sample_points,
                pole_selection=spec.pole_selection,
                sample_via=sample_via,
            )
        else:
            exact_table = _build_exact_scalar_table(
                c, h_ext, coordinate, ell, spec.lam, spec.q_order,
                used_regularize=used_regularize,
                chiral_norm=spec.chiral_norm,
            )
            serialized = serialize_deriv_tables({coordinate: exact_table}, shift=delta_min)
        prefactored = serialized
        channels.append(IdenticalScalarSpinChannel(
            ell=ell,
            delta_min=delta_min,
            serialized=serialized,
            prefactored=prefactored,
            crossing_context=_crossing_context_for_spec(spec, problem.resolved_method),
        ))
    return channels


def _build_extra_twist_channels(problem, extra_twists: int):
    """Build extra large-spin tower channels at twists *between* the discrete
    double-twist slices the SDP constrained.

    The SDP only imposes tower positivity on the lines ``tau = tau_n`` (the
    double-twist set). An optimized dual functional can dip negative at
    intermediate twists, which is not an interpolation artifact: unitarity
    permits operators at any ``tau >= 0`` for large spin, so a negative value
    there means the discrete-twist relaxation was too coarse (the fix is to add
    those twists to the SDP constraint set, not just certification).

    Full-gap sweep: between each consecutive pair of base twists, place
    ``extra_twists`` evenly-spaced interior values (dividing the gap into
    ``extra_twists + 1`` pieces), plus one extra gap's worth above the top
    slice. Each channel evaluates the EXACT Virasoro tower block (no npoles
    interpolation) on ``L >= Lmax``, matching the production tower channels.
    Returns ``[]`` for the global / numeric control methods or when the problem
    has no tower.
    """
    if getattr(problem, "resolved_method", None) in _CHIRAL_METHODS:
        return []
    spec = problem.spec
    if extra_twists <= 0:
        return []
    if getattr(spec, "large_spin_lmin", None) is None:
        return []
    if problem.resolved_method in ("global_control", "numeric_virasoro_control"):
        return []
    base_twists = sorted(set(_certification_real_label(t) for t in spec.large_spin_twists))
    if not base_twists:
        return []

    sample_twists = []
    for lo, hi in zip(base_twists[:-1], base_twists[1:]):
        step = (hi - lo) / (extra_twists + 1)
        sample_twists.extend(lo + k * step for k in range(1, extra_twists + 1))
    # Extend one gap above the top slice at the same density.
    top_gap = (
        base_twists[-1] - base_twists[-2] if len(base_twists) >= 2 else _certification_real_label(2)
    )
    step = top_gap / (extra_twists + 1)
    sample_twists.extend(
        base_twists[-1] + k * step for k in range(1, extra_twists + 1)
    )

    from virasoro_bootstrap.identical_scalar import (
        IdenticalScalarSpinChannel, _central_charge_input,
    )
    from virasoro_blocks.large_spin import _LargeSpinTwistSampler

    coordinate = getattr(spec, "coordinate", "zzb")
    c = _central_charge_input(spec.c)
    h_ext = _certification_real_label(spec.h_ext, name="h_ext")
    z_shift = -2 * h_ext
    Lmax = mp.mpf(spec.large_spin_lmin)

    channels = []
    for tw in sample_twists:
        sampler = _LargeSpinTwistSampler(
            "virasoro", coordinate, [h_ext] * 4, spec.chiral_norm, z_shift,
            spec.lam, spec.lam, tw, c=c, q_order=spec.q_order,
        )
        channels.append(IdenticalScalarSpinChannel(
            ell=int(Lmax), delta_min=Lmax, serialized=None,
            prefactored=_ExactTowerPrefactored(sampler, Lmax),
            twist=tw,
        ))
    return channels
