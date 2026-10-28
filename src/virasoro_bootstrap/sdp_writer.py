"""Write an SDPB ``sdp`` directory (JSON format) directly from a PMP build,
bypassing pmp2sdp.

pmp2sdp converts our PMP (objective/normalization + per-block prefactor,
sample points, sample scalings, and *monomial* polynomial coefficients) into the
SDP it solves -- and in doing so it (a) evaluates the monomial polynomials at the
sample points at the *solve* precision and (b) builds the bilinear bases at the
solve precision. For high-degree, many-pole blocks the monomial coefficients span
tens of orders of magnitude, so both steps lose precision at the working
precision SDPB later solves at.

Writing the ``sdp`` ourselves lets us do both at high internal precision:

  * the per-sample-point constraint data ``c``/``B`` are linear combinations of
    the block values at the sample points, ``block_a(x_k) = sampleScaling[k] *
    P_a(x_k)``. When ``sample_values`` is supplied (the v2 path) these come
    straight from the numeric block recursion -- no monomial coefficients at any
    stage. Otherwise (v1) we evaluate the PMP's monomial polynomials at
    ``eval_dps`` (set it high), which is already cleaner than pmp2sdp at the
    solve precision but is bounded by the build dps.
  * the bilinear bases are orthonormal bases of pmp2sdp's weighted-monomial row
    spaces (``_compute_bilinear_bases``: flint Stieltjes/Lanczos with full
    re-orthogonalisation), rounded to ``eval_dps`` -- equivalent to pmp2sdp's
    up to a harmless orthogonal rotation (the SDP is invariant under it).
    Small nonzero sample weights do not reduce the polynomial row space.

Conventions (verified to ~1e-197 against ``pmp2sdp --outputFormat json``):
    pivot   = argmax|normalization|
    c[k]    = sampleScaling[k] * P_pivot(x_k) / N_pivot
    B[k][j] = sampleScaling[k] * ( N_j/N_pivot * P_pivot(x_k) - P_j(x_k) )   (non-pivot j)
    objectives.constant = obj_pivot / N_pivot
    objectives.b[j]     = obj_j - N_j/N_pivot * obj_pivot
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import mpmath as mp

from .pmp_writer import _compute_bilinear_bases


def _s(x, sig_figs: int):
    return mp.nstr(x, sig_figs, strip_zeros=False)


def _evalpoly(coeffs, x):
    r = mp.mpf(0)
    for c in reversed(coeffs):
        r = r * x + c
    return r


def _normalization_basis(data, *, eval_dps=None):
    """The normalization and eliminated index used by a saved SDP."""
    from virasoro_blocks.requests import _integer_field

    raw = data['normalization']
    precision = data.get('eval_dps', eval_dps)
    if precision is None:
        precision = max([mp.mp.dps, *(sum(ch.isdigit() for ch in str(v).lower().split('e', 1)[0])+5
                                     for v in raw)])
    else:
        precision = _integer_field('eval_dps', precision)
        if precision < 1:
            raise ValueError('normalization eval_dps must be positive')
    with mp.workdps(precision):
        norm = [mp.mpf(v) for v in raw]
    if not norm or not all(mp.isfinite(v) for v in norm) or not any(norm):
        raise ValueError('normalization must be finite and nonzero')
    if 'eliminated_index' in data:
        pivot = _integer_field('eliminated_index', data['eliminated_index'])
        if not 0 <= pivot < len(norm) or norm[pivot] == 0:
            raise ValueError('invalid eliminated normalization index')
    else:
        # abs(mpf) rounds at ambient precision; preserve the saved values
        # when comparing nearly tied positive or negative coefficients.
        pivot = max(range(len(norm)), key=lambda i: norm[i] if norm[i] >= 0 else mp.fneg(norm[i], exact=True))
    return norm, pivot


def _block_data(block, normalization, eval_dps, output_digits, sample_values=None):
    """Return ({bilinear_bases_even/odd, c, B}, num_points) for one PMP block.

    ``sample_values`` (v2): optional list ``P[k][a]`` of the F-vector values at
    the block's sample points, computed cleanly (numeric block recursion). When
    omitted (v1) the values are obtained by evaluating the block's monomial
    polynomials at ``eval_dps``.
    """
    with mp.workdps(eval_dps):
        xs = [mp.mpf(s) for s in block["samplePoints"]]
        sc = [mp.mpf(s) for s in block["sampleScalings"]]
        nm = [mp.mpf(s) for s in normalization]
        piv = max(range(len(nm)), key=lambda a: abs(nm[a]))
        nonpiv = [a for a in range(len(nm)) if a != piv]

        if sample_values is not None:
            P = [[mp.mpf(v) for v in row] for row in sample_values]
        else:
            comps = [[mp.mpf(c) for c in comp] for comp in block["polynomials"][0][0]]
            P = [[_evalpoly(comps[a], xk) for a in range(len(comps))] for xk in xs]

        c, B = [], []
        for k in range(len(xs)):
            Pp = P[k][piv]
            c.append(_s(sc[k] * Pp / nm[piv], output_digits))
            B.append([
                _s(sc[k] * (nm[a] / nm[piv] * Pp - P[k][a]), output_digits)
                for a in nonpiv
            ])

        n_samples = len(xs)
        be, bo = _compute_bilinear_bases(
            tuple(xs), list(sc), n_samples - 1, output_digits
        )
        be = [[_s(v, output_digits) for v in row] for row in be]
        bo = [[_s(v, output_digits) for v in row] for row in bo]
    return {"bilinear_bases_even": be, "bilinear_bases_odd": bo, "c": c, "B": B}, len(xs)


def write_sdp_directory(pmp_doc, output_dir, *, eval_dps=400,
                        output_digits=None, sample_values_by_block=None,
                        pmp_info=None):
    """Write an SDPB ``sdp`` directory from a scalar-constraint PMP document.

    ``pmp_doc``: the PMP dict (``PrefactoredPmpBuild.doc``) -- objective,
    normalization, and ``PositiveMatrixWithPrefactorArray``.
    ``output_digits``: decimal digits written to SDPB's JSON input. Defaults to
    ``eval_dps`` so we do not silently throw away precision after high-precision
    evaluation. ``sample_values_by_block``: optional ``{block_index: P[k][a]}``
    clean F-vector sample values (v2). ``pmp_info``: optional pre-built
    ``pmp_info.json`` content (kept for downstream functional/certification
    tooling; not needed to solve).
    """
    from virasoro_blocks.requests import _integer_field

    eval_dps = _integer_field("eval_dps", eval_dps)
    if eval_dps < 1:
        raise ValueError("eval_dps must be >= 1")
    if output_digits is None:
        output_digits = eval_dps
    output_digits = _integer_field("output_digits", output_digits)
    if output_digits < 1:
        raise ValueError("output_digits must be >= 1")

    blocks = pmp_doc["PositiveMatrixWithPrefactorArray"]
    nm = pmp_doc["normalization"]
    if sample_values_by_block is not None and any(
        key not in range(len(blocks)) for key in sample_values_by_block
    ):
        raise ValueError("sample_values_by_block contains an unknown block index")

    with mp.workdps(eval_dps):
        # Parse decimal strings at eval_dps to preserve their stored accuracy.
        obj = [mp.mpf(s) for s in pmp_doc["objective"]]
        nmf = [mp.mpf(s) for s in nm]
        if not nmf or len(obj) != len(nmf):
            raise ValueError("objective and normalization must have the same nonzero length")
        if not all(mp.isfinite(v) for v in obj + nmf) or not any(nmf):
            raise ValueError("objective and normalization must be finite, with nonzero normalization")
        # Validate the whole input before creating a partially usable SDP.
        # This writer supports scalar constraints only; taking [0][0] from
        # a larger matrix would silently discard the other constraints.
        for i, block in enumerate(blocks):
            matrix = block["polynomials"]
            if len(matrix) != 1 or len(matrix[0]) != 1:
                raise ValueError("direct SDP writer supports only scalar polynomial matrices")
            components = matrix[0][0]
            if len(components) != len(nmf):
                raise ValueError("polynomial and normalization dimensions disagree")
            if not all(mp.isfinite(mp.mpf(c)) for comp in components for c in comp):
                raise ValueError("polynomial coefficients must be finite")
            xs = [mp.mpf(s) for s in block["samplePoints"]]
            sc = [mp.mpf(s) for s in block["sampleScalings"]]
            if not xs or len(xs) != len(sc):
                raise ValueError("sample points and scalings must have the same nonzero length")
            if (not all(mp.isfinite(x) and x >= 0 for x in xs)
                    or len(set(xs)) != len(xs)
                    or not all(mp.isfinite(s) and s > 0 for s in sc)):
                raise ValueError("sample points must be finite, distinct and nonnegative; scalings positive and finite")
            if any(mp.mpf(c) != 0 for comp in components for c in comp[len(xs):]):
                raise ValueError("too few sample points for the polynomial degree")
            sv = None if sample_values_by_block is None else sample_values_by_block.get(i)
            if sv is not None:
                if len(sv) != len(xs) or any(len(row) != len(nmf) for row in sv):
                    raise ValueError("sample values must match the sample-point and normalization dimensions")
                if not all(mp.isfinite(mp.mpf(v)) for row in sv for v in row):
                    raise ValueError("sample values must be finite")
        piv = max(range(len(nmf)), key=lambda a: abs(nmf[a]))
        nonpiv = [a for a in range(len(nmf)) if a != piv]
        constant = _s(obj[piv] / nmf[piv], output_digits)
        b = [
            _s(obj[a] - nmf[a] / nmf[piv] * obj[piv], output_digits)
            for a in nonpiv
        ]
    out = Path(output_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Build on disk first: large SDPs need not fit in memory, and a failure
    # during numerical evaluation must not overwrite a usable previous SDP.
    with TemporaryDirectory(prefix=".sdp-build-", dir=out.parent) as temporary:
        stage = Path(temporary)
        (stage / "objectives.json").write_text(json.dumps({"constant": constant, "b": b}))
        (stage / "normalization.json").write_text(json.dumps({
            "normalization": nm, "eliminated_index": piv, "eval_dps": eval_dps,
        }))
        for i, block in enumerate(blocks):
            sv = None if sample_values_by_block is None else sample_values_by_block.get(i)
            bd, npts = _block_data(block, nm, eval_dps, output_digits, sample_values=sv)
            (stage / f"block_data_{i}.json").write_text(json.dumps(bd))
            (stage / f"block_info_{i}.json").write_text(json.dumps({"dim": 1, "num_points": npts}))
        if pmp_info is not None:
            (stage / "pmp_info.json").write_text(json.dumps(pmp_info))
        (stage / "control.json").write_text(json.dumps(
            {"num_blocks": len(blocks), "command": "virasoro_bootstrap.sdp_writer"}))

        out.mkdir(parents=True, exist_ok=True)
        for path in stage.iterdir():
            if path.name != "control.json":
                path.replace(out / path.name)
        if pmp_info is None:
            (out / "pmp_info.json").unlink(missing_ok=True)
        # Remove only obsolete filenames owned by this writer.
        for path in out.iterdir():
            match = re.fullmatch(r"block_(?:data|info)_(0|[1-9][0-9]*)\.json", path.name)
            if match and int(match[1]) >= len(blocks):
                path.unlink()
        (stage / "control.json").replace(out / "control.json")
        # A fresh solve supersedes a previous completed task's compact sample
        # bundle. Leave it intact on failed builds, just like the original SDP.
        from .hpc.storage import RECORD_NAME, SAMPLES_NAME

        (out / RECORD_NAME).unlink(missing_ok=True)
        (out / SAMPLES_NAME).unlink(missing_ok=True)
    return out
