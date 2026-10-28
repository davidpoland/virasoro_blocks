"""Command-line interface for virasoro_blocks.

Compute Virasoro sphere 4-point conformal block derivative tables and write them
as JSON (blocks_3d schema) or Mathematica .m (scalar_blocks style).

Examples::

    # Order-reduced interpolation (npoles=10), spin 0..10 even, lambda=12, c=10
    virasoro_blocks table --family virasoro --basis scalar_G \\
                   --central-charge 10 --spin-ranges 0:2:10 --lambda 12 \\
                   --delta-ext 0.15 0.15 0.15 0.15 \\
                   --order 30 --npoles 10 --precision 120 \
                   --format json -o out/

    # Exact (keep every pole), Mathematica output, specific spins
    virasoro_blocks table --family virasoro --basis scalar_G \\
                   --central-charge 10 --spin-ranges 0 2 4 --lambda 6 \\
                   --order 20 --precision 60 --format m --npoles all -o out/

    # Spinning externals: --spin-ext gives the four external operator spins;
    # any nonzero entry selects the ordered-block path (non-folded coords only)
    virasoro_blocks table --family virasoro --basis ordered_B \\
                   --central-charge 10 --spin-ranges 0 1 2 --lambda 8 \\
                   --delta-ext 0.5 0.5 0.5 0.5 --spin-ext 0 2 0 2 \\
                   --order 25 --precision 80 --coordinates zzb -o out/

    # Chiral-only block F(h, z): one Taylor index, no exchanged spin. Pair
    # --basis chiral_F with the chiral coordinates {z, y, q}; the chiral path
    # takes external chiral weights via --h-ext (not --delta-ext), and
    # --spin-ranges / --spin-ext / --parity-* are rejected.
    virasoro_blocks table --family virasoro --basis chiral_F \\
                   --central-charge 10 --lambda 12 \\
                   --h-ext 0.075 0.075 0.075 0.075 \\
                   --basis chiral_F --coordinates z y q \\
                   --order 30 --npoles 10 --precision 120 -o out/
"""

from __future__ import annotations

import argparse
import multiprocessing
from dataclasses import replace
import os
import shlex
import sys

import mpmath as mp

from .cli import (
    check_fixed_value_counts,
    KAC_LIMIT_HELP, RAMOND_BRANCHES, RamondBranchAction, accept_negative_numbers, chiral_norm as _chiral_norm,
    join_option_values,
)

from .truncation import total_parity_phase
from .parity import (
    normalize_external_parity,
    normalize_parity_phase,
    parity_reversal_phase,
)
from .params import b_from_c
from .recursion import set_kac_limit
from .singularity import nearby_singular_locus, recognized_singular_locus
from .precision import tolerance
from .requests import MIN_TABLE_PRECISION
from .serialization import serialize_deriv_tables
from .writers import mathematica_filename, write_json, write_mathematica
from .large_spin import (
    LARGE_SPIN_COORDINATES,
    global_tower_q_order,
    large_spin_twist_table,
    refuse_spin_zero_scalar_G_tower,
)
from .interpolation import POLE_SELECTIONS
from .table_backends import (
    _B_POLY_TABLE_BUILDERS,
    _CHIRAL_POLY_TABLE_BUILDERS,
    _POLY_TABLE_BUILDERS,
    _PROJECTED_B_POLY_TABLE_BUILDERS,
    _PROJECTED_RADIAL_POLY_TABLE_BUILDERS,
    _RADIAL_POLY_TABLE_BUILDERS,
    build_cli_chiral_cache as _build_cli_chiral_cache,
)

from .table_backends import NUMERIC_SAMPLE_VIA_COORDINATES as _NUMERIC_SAMPLE_VIA_COORDINATES

_RADIAL_COORDINATES = set(_RADIAL_POLY_TABLE_BUILDERS)
_CHIRAL_COORDINATES = set(_CHIRAL_POLY_TABLE_BUILDERS)

_GLOBAL_2D_BASES = {
    "global_scalar_G",
    "global_ordered_B",
    "global_parity_projected_B",
}
_GLOBAL_CHIRAL_BASES = {"global_chiral_F"}
_GLOBAL_BASES = _GLOBAL_2D_BASES | _GLOBAL_CHIRAL_BASES

_SUPER_2D_BASES = {
    "ns_scalar_G", "ns_ordered_B", "ns_parity_projected_B",
    "ramond_scalar_G", "ramond_ordered_B", "ramond_parity_projected_B",
}
_SUPER_CHIRAL_BASES = {"ns_chiral_F", "ramond_chiral_F"}
_SUPER_BASES = _SUPER_2D_BASES | _SUPER_CHIRAL_BASES

_ALL_COORDINATES = (
    list(_POLY_TABLE_BUILDERS)
    + list(_RADIAL_POLY_TABLE_BUILDERS)
    + list(_CHIRAL_POLY_TABLE_BUILDERS)
)
# Coordinates whose blocks have a definite z <-> zbar parity are admissible for
# the raw ordered-B path; the parity-folded ones (incl. radial slices) are not.
_NON_FOLDED_COORDINATES = set(_B_POLY_TABLE_BUILDERS)
# Parity-folded 2d coordinates (m + 2n <= Lambda truncation; squared second
# variable, so n_max only reaches Lambda // 2).
_FOLDED_COORDINATES = {"ws", "xt", "qqb_ws"}

_CLI_BASES = frozenset({
    "auto", "scalar_G", "ordered_B", "parity_projected_B", "chiral_F",
    *_GLOBAL_BASES, *_SUPER_BASES,
})


def _parse_half_integer(token: str, *, name: str):
    """Parse spin labels exactly, including decimal and rational text."""
    from .cli import half_integer
    from .numeric_input import parse_number

    try:
        return parse_number(half_integer(token))
    except argparse.ArgumentTypeError as exc:
        raise ValueError(f"invalid {name} {token!r}: {exc}") from exc


def _parse_positive_int(token: str, *, name: str) -> int:
    """Parse a strictly positive integer CLI flag."""
    try:
        value = int(token)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"{name} must be >= 1")
    return value


def _parse_npoles(token: str):
    """Parse --npoles: a non-negative integer kept-pole count, or 'all'.

    'all' keeps every pole (the former --method full); it is normalized to the
    internal ``method="full"`` / ``npoles=None`` path post-parse in ``main``.
    """
    if token == "all":
        return "all"
    try:
        value = int(token)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("npoles must be an integer or 'all'") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("npoles must be >= 0 or 'all'")
    return value


def _json_spin_value(value):
    """Return an int for integral spins, a float for half-integers."""
    from .writers import _spin_json_value

    return _spin_json_value(value)


def _parse_spin_token(token: str):
    """Parse one spin spec: integer/half-integer or start:step:stop."""
    parts = token.split(":")
    if len(parts) == 1:
        return [_parse_half_integer(parts[0], name="spin")]
    if len(parts) == 2:
        start = _parse_half_integer(parts[0], name="spin")
        stop = _parse_half_integer(parts[1], name="spin")
        step = 1
    elif len(parts) == 3:
        start = _parse_half_integer(parts[0], name="spin")
        step = _parse_half_integer(parts[1], name="spin step")
        stop = _parse_half_integer(parts[2], name="spin")
    else:
        raise ValueError(f"invalid spin spec: {token!r}  (expected int/half-int or start:step:stop)")
    if step <= 0:
        raise ValueError("spin step must be > 0")
    if start > stop:
        raise ValueError(f"empty spin range {token!r} (start > stop)")
    values = []
    current = start
    while current <= stop:
        values.append(current)
        current += step
    return values


def _parse_spins(tokens: list[str]):
    spins = []
    for t in tokens:
        spins.extend(_parse_spin_token(t))
    unique = {int(2 * s): s for s in spins}
    return [unique[key] for key in sorted(unique)]


def _print_root_help() -> None:
    """Show the small operation index instead of the legacy flag matrix."""
    parser = argparse.ArgumentParser(
        prog="virasoro_blocks",
        description=(
            "Conformal-block evaluation and derivative tables. Choose an "
            "operation first; each operation has focused help."
        ),
        epilog=(
            "Examples: virasoro_blocks evaluate --help; "
            "virasoro_blocks table --help; "
            "virasoro_blocks point-table --help; "
            "virasoro_blocks info.\n"
            "Table flags with no operation (virasoro_blocks --lambda 3 ...) run `table`."
        ),
    )
    parser.add_argument(
        "operation", nargs="?",
        choices=["evaluate", "table", "point-table", "info"],
        metavar="{evaluate,table,point-table,info}",
        help="Operation to run; use its --help for the available options.",
    )
    parser.print_help()


class _DefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show a default only when it says something ("(default: None)" on a dozen options did not), and not twice."""

    def _get_help_string(self, action):
        text = action.help or ""
        if action.default in (None, False, argparse.SUPPRESS) or "default" in text.lower():
            return text
        return super()._get_help_string(action)


def _build_parser(*, prog: str = "virasoro_blocks") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        allow_abbrev=False,   # a prefix like --delta silently meant --delta-ext
        description=(
            "Compute conformal blocks and derivative tables. The canonical "
            "operation commands are `evaluate`, `table`, `point-table`, and "
            "`info`; table flags with no operation run `table`."
        ),
        epilog=(
            "Examples: `virasoro_blocks evaluate --help`, "
            "`virasoro_blocks table --help`, `virasoro_blocks point-table "
            "--help`, and `virasoro_blocks info`."
        ),
        formatter_class=_DefaultsHelpFormatter,
    )
    advanced = p.add_argument_group("advanced numerical controls")
    large_spin = p.add_argument_group("large-spin tower")
    p.add_argument(
        "--central-charge", default=None, metavar="C",
        help="Virasoro central charge c. Required for the Virasoro bases; "
             "explicit global_* bases are independent of c (a given c is noted and unused).",
    )
    p.add_argument(
        "--surface", choices=["sphere", "torus"], default="sphere",
        help="Which surface's blocks to emit. 'sphere' (default) is the "
             "4-point block pipeline. 'torus' emits exact genus-1 reduced-"
             "character tables (basis torus_G, coordinate vvb, JSON only): "
             "--central-charge, --spin-ranges (integer spins), --lambda, "
             "--precision, -o and --output-template apply; the sphere-only "
             "knobs (--order, --npoles, external weights, parity, "
             "coordinates, shifts/centers, pole selection, sampling, regularization, "
             "threads, the large-spin tower) are rejected. --torus-onept-ext switches to "
             "one-point tables.",
    )
    p.add_argument(
        "--torus-onept-ext", default=None, metavar="D",
        help="With --surface torus: emit torus ONE-POINT tables "
             "(basis torus_onept_G) for a scalar external of chiral weight "
             "D (Delta_Phi = 2 D), instead of the partition-function "
             "character tables. Re-enables --order (q-expansion truncation; "
             "~2.7 digits/order), --npoles (kept Kac poles, "
             "nearest-to-positivity selection; chosen when omitted) and --q-gate. The spin-0 anchor "
             "(--torus-scalar-shift X or --gap G, default 0) must be >= 0; its double Kac pole at "
             "Delta = 0 stays in the prefactor. "
             "Built at the working precision --precision and verified against a deeper, higher-precision "
             "reference: the file records verified_digits and its fit/precision/q-truncation parts.",
    )
    p.add_argument(
        "--gap", default=None, metavar="G",
        help="Raise the emitted table's positivity anchor to G >= 0 above the "
             "basis unitarity bound: |ell| + G for the 2d bases, G for chiral (a negative G is refused on "
             "the sphere). "
             "The block's anchor is its positivity domain (the polynomial "
             "variable is x = Delta - shift and SDPB imposes x >= 0), so this "
             "is how a gap is expressed in a block file -- the same thing the "
             "in-process bootstrap does per probe. Default: the unitarity "
             "bound, i.e. no gap. Not to be confused with --z-shift, which moves the kinematic "
             "z-exponent (crossing stripping) and is unrelated. Torus one-point "
             "tables: every spin at |j| + G (exclusive with --torus-scalar-shift); "
             "refused for torus character tables and the large-spin tower.",
    )
    p.add_argument(
        "--torus-scalar-shift", default=None, metavar="X",
        help="Anchor (shift) for the spin-0 torus one-point table alone; must be "
             ">= 0 (typically the gap / positivity edge). The other "
             "spins anchor at the unitarity bound |j|. Refused with --gap (which anchors every "
             "spin at |j| + G, spin 0 at G), without spin 0 in --spin-ranges, and for "
             "character tables.",
    )
    p.add_argument(
        "--spin-ranges", nargs="+", default=["0"], metavar="SPIN",
        help="Spins to compute: integers/half-integers or ranges start:stop / start:step:stop (e.g. '0:2:10' or "
             "'1/2:1:5/2'); default 0",
    )
    p.add_argument(
        "--lambda", dest="lam", required=True, type=int, metavar="N",
        help="Derivative order Lambda: the entries d^m d^n with m + n <= Lambda (scalar_G at zbar0 = z0 keeps n <= m by symmetry; "
             "folded coordinates keep their parity-allowed subset)",
    )
    p.add_argument(
        "--delta-ext", nargs=4, default=None, metavar="D",
        help="External scaling dimensions d1 d2 d3 d4. Paired with optional "
             "--spin-ext for the 2d bases; mutually exclusive with "
             "--h-ext / --hbar-ext. Defaults to all zeros when neither "
             "parametrization is supplied (2d only).",
    )
    p.add_argument(
        "--spin-ext", nargs=4, type=lambda s: _parse_half_integer(s, name="external spin"),
        default=None, metavar="S",
        help="External operator spins s1 s2 s3 s4 for the (Delta, spin) "
             "parametrization (2d bases only). Defaults to all zeros (scalar "
             "externals); any nonzero value selects the spinning-external "
             "ordered-block path, which allows only non-folded coordinates "
             "(with --parity-project it selects parity_projected_B, which "
             "allows folded coordinates too). "
             "Mutually exclusive with --h-ext / --hbar-ext.",
    )
    p.add_argument(
        "--h-ext", nargs=4, default=None, metavar="H",
        help="External holomorphic weights h1 h2 h3 h4. On the 2d bases, "
             "must be paired with --hbar-ext (mutually exclusive with "
             "--delta-ext / --spin-ext). For --basis chiral_F, --h-ext is "
             "the only external-weight input and --hbar-ext is rejected "
             "(the chiral block has no antiholomorphic half).",
    )
    p.add_argument(
        "--hbar-ext", nargs=4, default=None, metavar="HB",
        help="External antiholomorphic weights hbar1 hbar2 hbar3 hbar4 (2d "
             "bases only). Paired with --h-ext as the chiral-halves "
             "parametrization; mutually exclusive with --delta-ext / "
             "--spin-ext.",
    )
    p.add_argument(
        "--parity-ext", nargs=4, type=normalize_parity_phase,
        default=None, metavar="P",
        help="Optional external parity eigenphases for the four operators in "
             "a diagonal parity basis; each P must be one of 1, -1, i, -i. "
             "With --parity-project they set the parity-reversal phase of the "
             "parity-projected block (and are recorded in the metadata)",
    )
    p.add_argument(
        "--parity-project", choices=["+", "-"], default=None,
        help="For spinning externals, compute the parity-projected B_+ or B_- basis "
             "instead of the raw ordered B basis. Requires --parity-ext.",
    )
    p.add_argument(
        "--basis",
        metavar="BASIS",
        default="auto",
        help="Block basis to emit (auto, scalar_G, ordered_B, "
             "parity_projected_B, or chiral_F; family-prefixed spellings are "
             "accepted). 'auto' (default) infers from --spin-ext and "
             "--parity-project: scalar_G when externals are scalar, ordered_B / "
             "parity_projected_B when they are spinning. 'chiral_F' emits the "
             "chiral half F(h, z) only (no antiholomorphic externals, no "
             "exchanged spin); it must be paired with the chiral coordinates "
             f"{sorted(_CHIRAL_COORDINATES)}. The 'global_*' bases emit the "
             "corresponding 2d global (SL(2)xSL(2)) blocks via optimal-node "
             "interpolation of the numeric 2F1 block (no q-recursion; --order "
             "sets the SL(2) pole pool). 'global_scalar_G' needs scalar "
             "externals (any 2d coordinate, folded ones included); "
             "'global_ordered_B' / 'global_parity_projected_B' are the "
             "spinning-external analogues on the non-folded coordinates "
             f"{sorted(_B_POLY_TABLE_BUILDERS)}; "
             f"'global_chiral_F' is the chiral-only global half on "
             f"{sorted(_CHIRAL_COORDINATES)}. No global basis is inferred by 'auto'.",
    )
    p.add_argument(
        "--family",
        dest="block_family",
        choices=[
            "auto", "virasoro", "global", "ns", "ramond",
            "torus-character", "torus-onepoint",
        ],
        default="auto",
        help="Select the conformal block family. 'auto' infers it from --basis "
             "(Virasoro by default); ns and ramond expose the N=1 superconformal "
             "backends, with the same chiral/full bases and coordinates. On "
             "the torus, use torus-character or torus-onepoint to make the "
             "operation explicit; for large-spin tables, this selects the "
             "Virasoro or exact-global tower.",
    )
    p.add_argument(
        "--super-sector", choices=["even", "odd"], default="even",
        help="N=1 superconformal sector for NS/Ramond blocks (--family ns / ramond only; refused for the other "
             "families).",
    )
    p.add_argument(
        "--ramond-branch", choices=list(RAMOND_BRANCHES), default="++", action=RamondBranchAction,
        help="Ramond three-point sign branch (--family ramond only; refused for the other families). All four "
             "spellings work as a separate value, e.g. `--ramond-branch --`, or as `--ramond-branch=--`.",
    )
    p.add_argument(
        "--parity-exchanged", type=normalize_parity_phase, default=None, metavar="P",
        help="Parity phase of the exchanged operator in the current diagonal-basis "
             "convention. Used with --parity-project (and the parity_projected_B "
             "tower). The scalar_G basis is z<->zbar even at every spin, so it takes "
             "no exchanged parity: its one large-spin tower covers every integer "
             "L >= --large-spin-lmin, and -1 is refused (the z<->zbar-odd structure "
             "is parity_projected_B with --parity-project -).",
    )
    p.add_argument(
        "--order", type=int, default=30, metavar="N",
        help="q-expansion truncation order for Virasoro bases; for global "
             "blocks and global large-spin towers, the q-order of the finite SL(2) pole pool. "
             "Global 2F1 values are not truncated (evaluated at the working precision).",
    )
    advanced.add_argument(
        "--chiral-norm", type=_chiral_norm, default=None, metavar="N",
        help="Chiral block normalization N, a positive integer: F = N^{-h} (16 q)^h * ..., i.e. "
             "N^{-h} times the standard block z^{h-h1-h2}(1 + O(z)) at --z-shift=-h1-h2 (a spin-0 "
             "scalar_G block is F(z) F(zbar), as is the identity, and every channel's coefficient is "
             "N^Delta lambda^2). "
             "Default (None) uses the library default N=16, base "
             "q(1/2)=e^{-pi}~0.043, matched to the physical OPE^2*block decay "
             "(VMFT, arXiv:1811.05710) for well-conditioned bootstrap node "
             "placement. Pass 4 for base 4 q(1/2)~0.173~rho(1/2).",
    )
    advanced.add_argument(
        "--z-shift", type=str, default=None, metavar="S",
        help="Shift in the holomorphic z-exponent of the block: "
             "F ~ z^{h + z_shift} / N^h at small z (N = --chiral-norm). Default 0: leading power z^h, "
             "the standard block z^{h-h1-h2}(1 + O(z)) times z^{h1+h2}, divided by N^h. "
             "Pass -h1-h2 (e.g. --z-shift=-0.25 for h_ext=0.125) "
             "for the standard power z^{h-h1-h2}.",
    )
    advanced.add_argument(
        "--zbar-shift", type=str, default=None, metavar="S",
        help="Shift in the antiholomorphic zbar-exponent. Default 0, independent "
             "of --z-shift (use --u-shift to shift both). Only the two-sided "
             "bases (ordered_B / parity_projected_B) carry independent z/zbar "
             "shifts; scalar_G is symmetric (single shift from --z-shift) and "
             "rejects a --zbar-shift that differs from --z-shift.",
    )
    advanced.add_argument(
        "--u-shift", type=str, default=None, metavar="S",
        help="Set both --z-shift and --zbar-shift to the same value. "
             "Convenience option for the common case z_shift = zbar_shift. "
             "Overridden if --z-shift or --zbar-shift are also specified.",
    )
    advanced.add_argument(
        "--z-center", type=str, default=None, metavar="Z0",
        help="Off-symmetric expansion center: a cross ratio z0 != 1/2 about "
             "which the block is Taylor-expanded (default 1/2, the symmetric "
             "point). For a 2d block this is the diagonal point z = zbar = z0; "
             "use --zbar-center for an independent zbar. Non-folded 2d and chiral "
             "coordinates only (folded coordinates are rejected).",
    )
    advanced.add_argument(
        "--zbar-center", type=str, default=None, metavar="ZB0",
        help="Independent antiholomorphic expansion center zbar0 for an "
             "off-diagonal 2d block (z = center != zbar = center-bar). Requires "
             "--z-center; not valid for chiral coordinates (a chiral half has no "
             "zbar). Default: the diagonal (zbar0 = center).",
    )
    p.add_argument(
        "--npoles", type=_parse_npoles, default=None, metavar="K",
        help="Block fidelity knob: an integer is the "
             "kept-pole count for the order-reduced interpolation (an approximation; "
             "its verified_digits are recorded). When omitted, "
             "Virasoro/NS/Ramond tables keep every pole (the --order-truncated "
             "block, exactly; its high degree loses digits to rounding at --precision, "
             "which --target-digits compensates) and global tables, torus one-point tables, and large-spin towers choose the pole count; "
             "the literal 'all' is supported only for fixed-spin/chiral Virasoro sphere "
             "tables, writing the exact high-degree polynomial without sample values.",
    )
    p.add_argument(
        "--precision", type=int, default=60, metavar="P",
        help="Working precision in decimal digits (at least 20 for sphere tables, 10 for torus tables): every "
             "table is built and written at P digits, never silently at more. Each table is verified against "
             "its own sampler, between its nodes and past them to the large-Delta regime, and records "
             "verified_digits (per entry), verified_digits_largest_entry and the range covered; a monomial-form "
             "table loses digits to rounding (~0.6-0.75 per unit of degree), and a table short of P prints "
             "one line naming the cause and the fix. With --target-digits, P is where the build starts.",
    )
    p.add_argument(
        "--target-digits", type=lambda s: _parse_positive_int(s, name="target-digits"), default=None, metavar="D",
        help="Deliver at least D verified digits: the build starts at --precision and raises the working "
             "precision by the measured deficit (recorded as precision_actual, the precision the file is written "
             "at) until the table delivers D, and stops with an error naming the reason when --max-precision is "
             "reached, when more precision stops helping (the kept poles or the pole structure are the limit), or "
             "when an order-reduced (--npoles) table is limited by its order reduction. Off by default.",
    )
    p.add_argument(
        "--max-precision", type=lambda s: _parse_positive_int(s, name="max-precision"), default=None, metavar="M",
        help="Cap on the working precision --target-digits may escalate to (default max(4 D, 2 P, D + 200)); refused "
             "without --target-digits, and below --precision or D.",
    )
    advanced.add_argument(
        "--num-threads",
        type=lambda s: _parse_positive_int(s, name="num-threads"),
        default=1,
        metavar="N",
        help="blocks_3d-style worker count for sweeps over multiple exchanged spins; 1 keeps the serial path",
    )
    p.add_argument(
        "--coordinates", nargs="+", default=["zzb"], choices=_ALL_COORDINATES,
        metavar="COORD",
        help=f"Coordinates to compute; one or more of: {', '.join(_ALL_COORDINATES)} (default zzb)",
    )
    p.add_argument(
        "--format", choices=["m", "json", "both"], default="json",
        help="Output format",
    )
    advanced.add_argument(
        "--pole-selection",
        choices=POLE_SELECTIONS, metavar="SELECTOR",
        default=None,
        help="Which poles the interpolation prefactor keeps (integer --npoles; refused without one, where every "
             "pole is kept -- except for the fixed-spin global bases, whose --npoles-free table ranks the pool it "
             "grows). "
             "For Virasoro and superconformal blocks the default is "
             "'contribution_optimal': it "
             "uses full-principal-part contributions in a measure-aware, "
             "optimal-node remainder estimate to build a nested greedy order. "
             "Fixed-spin global blocks use the same default with their "
             "q-order-defined SL(2) candidate pool. "
             "'contribution' ranks each pole directly by the magnitude of its "
             "full principal part at the evaluation point (every Laurent "
             "coefficient, distance folded in to the correct power); "
             "'contribution_all' does the same maxed over all kept derivatives. "
             "'contribution_leading' is the legacy variant that divides only the "
             "(0,0) leading residue by distance^1 (correct only for simple "
             "poles); 'contribution_leading_all' its all-derivative form. "
             "'residue' ranks poles by their residue in the undifferentiated "
             "(0,0) block value; 'residue_all' takes the max residue over all "
             "kept derivatives. 'nearest' keeps the poles with smallest "
             "|Delta_pole|. For the global bases the genuine-pole residues "
             "decrease with depth, so nearest / rightmost / residue / "
             "contribution coincide; real_first / real_deep differ in how they "
             "handle double-pole multiplicity.",
    )
    advanced.add_argument(
        "--sample-via",
        choices=["numeric", "symbolic"],
        default="auto",
        help="How the Virasoro fixed-spin interpolation path (integer or omitted "
             "--npoles) obtains the block values at the interpolation sample points "
             "(omitted: chosen automatically, as described at the end; 'auto' is not a "
             "value to pass): 'numeric' uses the well-conditioned direct recursion/Taylor "
             "builders -- robust at the working precision, since the symbolic "
             "table's P(Delta)/D(Delta) coefficients reach ~1e500 and lose "
             "accuracy to cancellation; 'symbolic' evaluates the exact symbolic "
             "table directly (needs much higher precision). Outside that path the "
             "flag cannot change anything, so it is refused there: --npoles all "
             "does no sampling at all, and the global bases, the NS/Ramond "
             "families and the large-spin tower always sample numerically. On the "
             "Virasoro interpolation path "
             "the automatic choice is 'numeric' (at a regularized c the "
             "regularized numeric sampler), except it is 'symbolic' for a "
             "coordinate the numeric sampler does not cover (passing 'numeric' "
             "explicitly there is an error, not a silent fallback).",
    )
    advanced.add_argument(
        "--shared-sample-points",
        dest="shared_sample_points",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Interpolated tables only (refused with --npoles all). "
             "Default OFF (per-entry): each (m, n) interpolated entry keeps its "
             "own degree-matched node set, tying its polynomial degree to npoles "
             "so the large-Delta tail power stays correct, for both the Virasoro "
             "and global interpolated bases. Pass --shared-sample-points to "
             "instead pad every entry onto the single max-degree node set: more "
             "accurate strictly inside the node range, but the far tail is no "
             "longer power-matched and can blow up (Runge), which can spuriously "
             "inflate certified bounds that probe large Delta.",
    )
    advanced.add_argument(
        "--q-gate", dest="q_gate", choices=["off", "warn", "strict"],
        default="warn",
        help="q-order convergence gate: probe the chiral block's q-expansion "
             "tail at this (c, externals, --order) and estimate the block "
             "accuracy in decimal digits (low q_order silently corrupts "
             "blocks, and raising --precision cannot fix it). 'warn' "
             "(default) prints a one-line warning with the measured digits "
             "and a suggested --order when the estimate falls below 20 "
             "digits or the series has not entered its geometric regime; "
             "'strict' errors out instead; 'off' skips the probe. Covers the "
             "Virasoro and NS/Ramond families; each table records the estimate as "
             "q_truncation_digits. Global bases have no q-series (nothing to gate: only 'off' is accepted there). "
             "Torus one-point tables gate their "
             "measured q_truncation_digits the same way; torus character tables "
             "(exact) refuse it.",
    )
    advanced.add_argument(
        "--regularize", nargs="?", const="on", default="auto",
        choices=["auto", "on", "off"],
        help="Regularization mode for recursion-singular c (c = 1, c = 25, "
             "minimal models): the block AT c, from the merged Kac poles of the "
             "Laurent recursion. 'auto' (default) enables it iff c is a "
             "recognized singular locus, with a printed notice. 'on' always "
             "enables it (at a c that is no recognized locus it is the direct "
             "recursion). "
             "'off' selects the direct recursion and is refused at a "
             "recognized singular c. Bare "
             "--regularize is equivalent to --regularize on.",
    )
    advanced.add_argument(
        "--kac-limit", choices=["irreducible", "c_limit"], default="irreducible",
        help=KAC_LIMIT_HELP,
    )
    p.add_argument(
        "-o", "--output", default=".", metavar="DIR",
        help="Output directory",
    )
    p.add_argument(
        "--output-template", default="derivs_{spin}.json", metavar="TEMPLATE",
        help="JSON filename template; {spin} is replaced by the spin value. "
             "The default matches blocks_3d's `derivs_{}.json` convention.",
    )
    # --- Large-spin tower emission -----------------------------------------
    # When --large-spin-lmin is set the CLI emits fixed-twist, variable-spin-L
    # blocks (`derivs_large_spin_{index}.json`, `variable="L"`) instead of the
    # fixed-spin path, matching the bootstrap manifest's large-spin block files.
    large_spin.add_argument(
        "--large-spin-lmin", default=None, metavar="LMIN",
        help="Emit fixed-twist large-spin tower blocks instead of fixed-spin "
             "blocks. LMIN is the lower spin edge: the spin-L interpolation is "
             "valid for L >= LMIN, and LMIN becomes the block's shift, recorded "
             "as the `large_spin_lmin` field (and `variable=\"L\"`). An even "
             "scalar_G tower needs LMIN > 0 (its L = 0 member would be the "
             "doubled 2 B(h, h); spin 0 is a fixed-spin table). Requires "
             "--large-spin-double-twists and/or --large-spin-twists to choose "
             "the twist slices. Coordinates "
             f"{sorted(LARGE_SPIN_COORDINATES)}; spinning externals select the "
             "ordered_B / parity_projected_B tower.",
    )
    large_spin.add_argument(
        "--large-spin-double-twists", type=int, default=None, metavar="K",
        help="Auto-build the leading K double-twist twists from the external "
             "dimensions: the s-channel families tau_n = Delta_1+Delta_2+2n and "
             "Delta_3+Delta_4+2n for n = 0..K-1 (they coincide as "
             "2*Delta_ext+2n for identical externals). Combined (de-duplicated) "
             "with any explicit --large-spin-twists values.",
    )
    large_spin.add_argument(
        "--large-spin-twists", nargs="+", default=None, metavar="TAU",
        help="Explicit isolated twist values to emit, in addition to any "
             "--large-spin-double-twists family. Each twist becomes one tower "
             "block file.",
    )
    large_spin.add_argument(
        "--large-spin-pole-selection",
        choices=POLE_SELECTIONS, metavar="SELECTOR",
        default=None,
        help="Pole-selection scheme for the large-spin tower interpolation, "
             "separate from --pole-selection (which governs fixed-spin Delta "
             "channels). For Virasoro towers the default is "
             "'contribution_optimal', using the fixed Kac pool and the "
             "optimal-node, measure-aware remainder estimate. Global towers use "
             "the same default with their q-order-defined SL(2) candidate pool. "
             "'contribution_leading' is the legacy leading-residue/distance "
             "variant (identical here, since the tower poles are simple).",
    )
    return accept_negative_numbers(p)


def _q_gate_centers(args):
    """The expansion points the tables use (None: the symmetric point), for the q-gate."""
    if args.z_center is None and args.zbar_center is None:
        return None
    z0 = args.z_center if args.z_center is not None else mp.mpf(1) / 2
    zb0 = args.zbar_center if args.zbar_center is not None else z0
    return [z0] if zb0 == z0 else [z0, zb0]


def _run_q_gate(parser, args, *, c, h_ext, hbar_ext, lam, label):
    """Run the q-order convergence gate once per distinct chiral-external set.

    Probes the chiral q-series truncation tail (see
    :mod:`virasoro_blocks.qgate`) and warns or errors per ``--q-gate``.
    No-ops when ``c`` / ``--order`` are unset (global bases) and skips
    silently when the probe is unavailable (an unsupported coordinate). A strict-mode
    failure exits through ``parser.error`` with the measured digits and a
    suggested ``--order``.
    """
    if args.q_gate == "off" or c is None or args.order is None:
        return
    from .qgate import QConvergenceError, gate_q_convergence

    ext_sets = [tuple(h_ext)]
    if hbar_ext is not None and tuple(hbar_ext) != tuple(h_ext):
        ext_sets.append(tuple(hbar_ext))
    try:
        for ext in ext_sets:
            report = gate_q_convergence(
                c=c, h_externals=list(ext), q_order=args.order,
                n_terms=int(lam) + 2, coordinate=args.coordinates[0],
                chiral_norm=args.chiral_norm, z_shift=args.z_shift,
                center=_q_gate_centers(args), mode=args.q_gate, label=label,
            )
            # recorded in each table as q_truncation_digits (verified_digits compares at the same --order)
            digits = None if report is None else report.get("digits")
            if digits is not None:
                previous = getattr(args, "_q_truncation_digits", None)
                args._q_truncation_digits = min(float(digits), float(args.precision),
                                                previous if previous is not None else float("inf"))
    except QConvergenceError as exc:
        parser.error(str(exc))


def _resolve_large_spin_externals(parser, args):
    """Resolve external weights for the large-spin tower.

    Accepts the ``--h-ext`` / ``--hbar-ext`` or the ``--delta-ext`` /
    ``--spin-ext`` parametrization; returns ``(h_ext, hbar_ext, delta_ext)`` as
    lists of exact real input labels. Spinning externals (``h_i != hbar_i``)
    select the ordered_B / parity_projected_B tower, with the same
    integer-or-half-integer external-spin requirement as the fixed-spin path.
    """
    using_chiral_halves = args.h_ext is not None or args.hbar_ext is not None
    using_delta_spin = args.delta_ext is not None or args.spin_ext is not None
    if using_chiral_halves and using_delta_spin:
        parser.error(
            "--h-ext / --hbar-ext and --delta-ext / --spin-ext are mutually "
            "exclusive parametrizations of the external operators; choose one"
        )
    if using_chiral_halves:
        if args.h_ext is None:
            parser.error("--hbar-ext requires --h-ext")
        h_ext = [_input_mpf(h) for h in args.h_ext]
        hbar_ext = (
            [_input_mpf(hb) for hb in args.hbar_ext]
            if args.hbar_ext is not None else list(h_ext)
        )
    else:
        delta_in = [_input_mpf(d) for d in (args.delta_ext or ["0", "0", "0", "0"])]
        spin_in = [_input_mpf(s) for s in (args.spin_ext or ["0"] * 4)]
        with mp.workdps(3 * mp.mp.dps + 50):     # (keep the guard digits through the halving)
            h_ext = [(d + s) / 2 for d, s in zip(delta_in, spin_in)]
            hbar_ext = [(d - s) / 2 for d, s in zip(delta_in, spin_in)]
    for i, (h, hb) in enumerate(zip(h_ext, hbar_ext)):
        twice_s = 2 * (h - hb)
        if twice_s != int(twice_s):
            parser.error(
                f"external operator {i + 1} has spin h - hbar = {mp.nstr(h - hb, 6)}; "
                "external spins must be integer or half-integer"
            )
    delta_ext = [h + hb for h, hb in zip(h_ext, hbar_ext)]
    return h_ext, hbar_ext, delta_ext


def _resolve_large_spin_twists(parser, args, hbar_ext):
    """Build the ordered, de-duplicated twist list for the tower.

    Combines the leading-``K`` double-twist families
    ``tau^{12}_n = 2(hbar_1 + hbar_2) + 2n`` and
    ``tau^{34}_n = 2(hbar_3 + hbar_4) + 2n`` with any explicit
    ``--large-spin-twists`` values. The tower fixes the antiholomorphic weight
    ``hbar = tau/2`` of the exchanged double-twist operator, whose leading-``n``
    value is ``hbar_i + hbar_j`` -- so the twist is set by the *antiholomorphic*
    external weights (this reduces to ``Delta_i + Delta_j`` for scalar
    externals). At least one source must be supplied.
    """
    from .numeric_input import real_input_label

    twists: list = []
    seen = set()

    def _push(value):
        label = real_input_label(value, name="large-spin twist")
        if label not in seen:
            seen.add(label)
            twists.append(label)

    if args.large_spin_double_twists is not None:
        k = args.large_spin_double_twists
        if k < 1:
            parser.error("--large-spin-double-twists must be >= 1")
        hbar_labels = [real_input_label(h, name="antiholomorphic external weight")
                       for h in hbar_ext]
        d12 = 2 * (hbar_labels[0] + hbar_labels[1])
        d34 = 2 * (hbar_labels[2] + hbar_labels[3])
        for n in range(k):
            _push(d12 + 2 * n)
            _push(d34 + 2 * n)
    if args.large_spin_twists is not None:
        for tok in args.large_spin_twists:
            _push(tok)
    if not twists:
        parser.error(
            "the large-spin tower needs at least one twist source: pass "
            "--large-spin-double-twists K and/or --large-spin-twists TAU ..."
        )
    return twists


def _emit_large_spin_tower(parser, args, raw_args, argv):
    """Build and write the fixed-twist large-spin (variable-``L``) tower.

    One ``derivs_large_spin_{index}.json`` file per twist slice, in the same
    ``variable="L"`` schema the bootstrap manifest writes, so CLI- and
    API-produced tower blocks are interchangeable.
    """
    # Resolve externals first: spinning externals (h_i != hbar_i) select the
    # ordered_B / parity_projected_B tower, mirroring the fixed-spin --basis auto.
    h_ext, hbar_ext, delta_ext = _resolve_large_spin_externals(parser, args)
    spinning = any(h != hb for h, hb in zip(h_ext, hbar_ext))
    projected = args.parity_project is not None

    if args.block_family == "global":
        mode = "global"
    elif args.block_family == "virasoro":
        mode = "virasoro"
    elif args.block_family == "auto":
        mode = "virasoro"
    else:
        parser.error(
            f"--family {args.block_family} is not supported for the large-spin tower"
        )

    if args.basis in ("chiral_F", "global_chiral_F"):
        parser.error(
            "the large-spin tower is a 2d block; --basis chiral_F is not applicable"
        )
    if args.basis in _GLOBAL_BASES:
        if mode != "global":
            parser.error(
                f"--basis {args.basis} requires --family global for the "
                "large-spin tower"
            )
        args.basis = args.basis[len("global_"):]
    if args.basis == "auto":
        basis = (
            "parity_projected_B" if projected
            else "ordered_B" if spinning
            else "scalar_G"
        )
    else:
        basis = args.basis
    # basis / externals / parity consistency
    if basis == "scalar_G":
        if spinning:
            parser.error(
                "--basis scalar_G requires scalar externals (h_i == hbar_i); got "
                "spinning externals -- use ordered_B / parity_projected_B"
            )
        if projected or args.parity_ext is not None:
            parser.error(
                "--parity-project / --parity-ext are not valid for scalar_G "
                "(it is already the parity sum of scalar externals)"
            )
    elif basis == "ordered_B":
        if any(co not in _NON_FOLDED_COORDINATES for co in args.coordinates):
            parser.error("ordered_B has no parity fold; use non-folded coordinates zzb, yyb, or qqb")
        if projected or args.parity_ext is not None or args.parity_exchanged is not None:
            parser.error(
                "--parity-project / --parity-ext / --parity-exchanged are "
                "parity-projection concepts; not valid for the raw ordered_B"
            )
    elif basis == "parity_projected_B":
        if not spinning:
            parser.error("--basis parity_projected_B requires spinning externals")
        if not projected:
            parser.error("--basis parity_projected_B requires --parity-project + or -")
    _z_eff = args.z_shift if args.z_shift is not None else mp.mpf(0)
    if args.zbar_shift is not None and args.zbar_shift != _z_eff:
        parser.error(
            "the large-spin tower currently uses a single symmetric shift; a "
            "--zbar-shift that differs from --z-shift is not applicable (use "
            "--z-shift / --u-shift)"
        )
    # The tower interpolates the block in the continuous spin L and has its own
    # pole selector / internal regularization, so the fixed-spin Delta-path
    # flags below are silently inert. Reject them when explicitly passed (they
    # all have non-None argparse defaults, so only raw_args distinguishes an
    # explicit value) rather than letting the user think they took effect.
    def _explicit(flag):
        return _flag_given(raw_args, flag)

    _tower_inapplicable = {
        "--pole-selection": "use --large-spin-pole-selection",
        "--regularize": "the tower regularizes internally at singular c",
        "--spin-ranges": "the tower's positivity variable is the spin L "
                         ">= --large-spin-lmin, not a fixed spin list",
    }
    bad_flags = [f for f in _tower_inapplicable if _explicit(f)]
    if bad_flags:
        parser.error(
            "the large-spin tower does not use "
            + ", ".join(bad_flags)
            + ": "
            + "; ".join(f"{f} ({_tower_inapplicable[f]})" for f in bad_flags)
        )
    # --parity-exchanged: the scalar_G tower is z<->zbar even at every spin, so
    # only +1 (the default) is accepted for it. For parity_projected_B it feeds
    # parity_reversal_phase and may be any of the Z4 phases (i / -i are the
    # fermionic-exchanged phases). ordered_B rejected it above.
    exchanged_parity = args.parity_exchanged or "1"
    try:
        refuse_spin_zero_scalar_G_tower(basis, _input_mpf(args.large_spin_lmin), exchanged_parity,
                                        flag="--large-spin-lmin")
    except ValueError as exc:
        parser.error(str(exc))
    bad_coords = [co for co in args.coordinates if co not in LARGE_SPIN_COORDINATES]
    if bad_coords:
        parser.error(
            f"the large-spin tower supports only coordinates "
            f"{sorted(LARGE_SPIN_COORDINATES)}; got {bad_coords}"
        )
    if args.format != "json":
        parser.error(
            "the large-spin tower is emitted as JSON only; --format must be "
            "'json' (the Mathematica writer has no variable='L' schema)"
        )
    if args.method == "full":
        parser.error(
            "the large-spin tower is an order-reduced interpolation in L: pass an integer --npoles, "
            "or omit it to choose the pole count (not --npoles all)"
        )
    if mode == "virasoro":
        if args.central_charge is None:
            parser.error(
                "--family virasoro large-spin tables require --central-charge"
            )
        c_build = _input_mpf(args.central_charge)
        order_build = args.order
        c_meta = c_build
        order_meta = args.order
        writer_basis = basis
        # q-order convergence gate for the Virasoro tower (the tower shares
        # the same H recursion truncated at the same --order; small-h fixed
        # probes are the conservative end, since large tower h converge
        # faster).
        _run_q_gate(
            parser, args, c=c_build, h_ext=h_ext, hbar_ext=hbar_ext,
            lam=args.lam, label="large-spin tower",
        )
    else:  # global
        if args.central_charge is not None:
            print(
                "note: --family global evaluates exact 2F1 blocks and "
                "is independent of --central-charge; recording null "
                "central_charge metadata",
                file=sys.stderr, flush=True,
            )
        c_build = None
        order_build = args.order
        c_meta = None
        order_meta = args.order
        writer_basis = "global_" + basis

    twists = _resolve_large_spin_twists(parser, args, hbar_ext)
    lmin = mp.mpf(args.large_spin_lmin)
    # the smallest integer spin >= lmin labels the files (the scalar_G tower is z<->zbar even and covers every
    # integer L >= lmin, so the label does not change its kept (m, n) entries)
    spin_meta = int(mp.ceil(lmin))

    os.makedirs(args.output, exist_ok=True)
    metadata_args = getattr(args, "_cli_invocation_args", raw_args)
    cli_args_str = shlex.join(["virasoro_blocks", *metadata_args])

    print(
        f"large-spin tower ({mode}): {len(twists)} twist(s), L >= "
        f"{mp.nstr(lmin, 6)}, coordinates {args.coordinates}",
        flush=True,
    )
    for index, tw in enumerate(twists):
        twist_display = mp.nstr(mp.convert(tw), 8)
        print(f"  twist {twist_display} -> derivs_large_spin_{index}.json ...",
              flush=True)
        from .truncation import is_off_diagonal, kept_pairs as _kept_pairs

        def _kept(coord):
            # blocks_3d --lambda truncation: non-folded coords keep the ell-independent
            # m + n <= Lambda triangle; folded coords depend on the fold parity (spin_meta).
            return _kept_pairs(
                coord, args.lam, basis=basis, ell=spin_meta if basis == "scalar_G" else 0,
                parity_projection=args.parity_project, parity_phase=total_parity_phase(args.parity_ext, args.parity_exchanged),
                off_diagonal=is_off_diagonal(args.z_center, args.zbar_center),
            )

        # without --npoles the pole count is chosen, as for the global fixed-spin tables: it starts at the count the
        # working precision carries and grows by the measured deficit while the kept poles limit the tower
        from .table_adapters import chosen_pole_start, pole_count_knobs

        poles = {"npoles": (int(args.npoles) if args.npoles is not None
                            else chosen_pole_start(args.precision_request, args.lam)), "previous": None}

        def _tower_order():
            # a global tower's order only sizes its pole pool, so a chosen count enlarges it (a Virasoro tower's
            # order is its q truncation and stays as given)
            if mode == "global" and args.npoles is None:
                return global_tower_q_order(order_build, poles["npoles"], h_ext)
            return order_build

        def _tower_table(coord, kept_poles=None):
            return large_spin_twist_table(
                tw, mode=mode, coordinate=coord, kept_poles=kept_poles,
                h_externals=h_ext, hbar_externals=hbar_ext, basis=basis,
                parity_projection=args.parity_project,
                parity_external=args.parity_ext,
                m_max=args.lam, n_max=0 if coord in _RADIAL_COORDINATES else args.lam,
                Lmax=lmin, npoles=poles["npoles"],
                c=c_build, q_order=_tower_order(), chiral_norm=args.chiral_norm,
                z_shift=args.z_shift,
                center=args.z_center, center_bar=args.zbar_center,
                pole_selection=args.large_spin_pole_selection,
                num_threads=args.num_threads,
                shared_sample_points=args.shared_sample_points,
                exchanged_parity=exchanged_parity,
                kept_entries=_kept(coord),
            )

        def _build_tower(work):
            with mp.workdps(work):
                # the first coordinate chooses the poles, the others reuse them (one prefactor for the whole tower)
                first = _tower_table(args.coordinates[0])
                if mode == "virasoro":
                    poles["pool"] = first.remainder_pool     # (a global pool grows with the count)
                chosen = [p + first.shift for p in first.prefactor.poles]
                tables = {args.coordinates[0]: first}
                for coord in args.coordinates[1:]:
                    tables[coord] = _tower_table(coord, kept_poles=chosen)
                serialized = serialize_deriv_tables(tables)
                # the tower's values at its sample nodes, from the fit's own records (as for the fixed-spin tables)
                from .clean_samples import attach_recorded_sample_block_values

                attach_recorded_sample_block_values(serialized, tables)
                serialized.q_order_used = _tower_order()
            return serialized, tables

        # the tower is an order-reduced interpolation in L: measure what it reproduces between its spin nodes
        # against the block it interpolates, as for a fixed-spin --npoles table
        from dataclasses import replace as _replace

        from .large_spin import tower_verified_digits
        from .verified_build import diagnose_monomial, record_verification, run_verified_build

        request = _replace(args.precision_request,
                           knobs=_replace(args.precision_request.knobs, keep_all="omit it to choose the pole count"))

        def _verify_tower(built, work):
            serialized, tables = built
            with mp.workdps(work):
                return tower_verified_digits(serialized, tables, {coord: _kept(coord) for coord in tables},
                                             request.digits)

        def _diagnose_tower(built, verification, work):
            with mp.workdps(work):
                return diagnose_monomial(built[0], verification, digits=request.digits, work=work, order_reduced=True)

        (serialized, _tables), verification, work = run_verified_build(
            _build_tower, _verify_tower, request=request, order_reduced=True, diagnose=_diagnose_tower,
            knobs=() if args.npoles is not None else pole_count_knobs(poles, request, 8 * request.digits),
            what=f"twist {twist_display}: the tower ("
                 + ("" if args.npoles is None else f"--npoles {args.npoles}; ")
                 + "checked between its spin nodes, past which it extrapolates)",
        )
        record_verification(serialized, verification, work, request)
        serialized.q_truncation_digits = getattr(args, "_q_truncation_digits", None)
        with mp.workdps(max(mp.mp.dps, work)):      # (written at the precision it was built at)
            text = write_json(
                serialized,
                c=c_meta, h_ext=h_ext, hbar_ext=hbar_ext,
                spin=spin_meta,
                q_order=order_meta,
                npoles=len(serialized.pole_list_x) if args.npoles is None else int(args.npoles),
                pole_selection=args.large_spin_pole_selection,
                basis=writer_basis,
                parity_projection=args.parity_project,
                parity_external=args.parity_ext,
                chiral_norm=args.chiral_norm,
                precision_input=args.precision,
                args=cli_args_str,
                variable="L",
                twist=tw,
                large_spin_lmin=lmin,
                parity_exchanged=exchanged_parity if exchanged_parity != "1" else None,
                z_center=args.z_center,
                zbar_center=args.zbar_center,
                z_shift=args.z_shift,
                zbar_shift=args.zbar_shift,
                regularization_policy=("merged" if c_meta is not None and recognized_singular_locus(c_meta)
                                       is not None else "direct"),
            )
        path = os.path.join(args.output, f"derivs_large_spin_{index}.json")
        with open(path, "w") as fh:
            fh.write(text)
        print(f"    wrote {path}")


def _cli_spinning_externals(args) -> bool:
    """Whether the raw external flags describe spinning operators (h_i != hbar_i)."""
    if args.h_ext is not None and args.hbar_ext is not None:
        return any(
            mp.mpf(h) != mp.mpf(hb) for h, hb in zip(args.h_ext, args.hbar_ext)
        )
    if args.spin_ext is not None:
        return any(mp.mpf(s) != 0 for s in args.spin_ext)
    return False


def _check_damping_base(parser, args, *, is_global):
    """Reject a --chiral-norm whose interpolation damping base is not in (0, 1).

    The interpolated tables carry the prefactor ``base**Delta`` with
    ``base = 16 q(z0) / N`` (Virasoro, NS, Ramond) or ``4 rho(z0) / N``
    (global). The sample-point generator requires ``0 < base < 1``.
    """
    if args.chiral_norm is None:
        return
    norm = mp.mpf(args.chiral_norm)
    if not norm > 0:
        parser.error("--chiral-norm must be positive")
    centers = [args.z_center]
    if args.zbar_center is not None:
        centers.append(args.zbar_center)
    from .recursion import q_from_z

    for center in centers:
        z0 = mp.mpf(1) / 2 if center is None else mp.mpf(center)
        if is_global:
            root = mp.sqrt(1 - z0)
            raw = 4 * (1 - root) / (1 + root)
        else:
            raw = 16 * mp.re(q_from_z(z0))
        if not raw / norm < 1:
            parser.error(
                f"--chiral-norm {mp.nstr(norm, 8)} gives the interpolation "
                f"damping base {mp.nstr(raw / norm, 8)} >= 1 at the expansion "
                f"center z0 = {mp.nstr(z0, 8)}; interpolated tables need "
                f"--chiral-norm > {mp.nstr(raw, 8)} (or --npoles all where supported)"
            )


def _check_output_template(parser, args, n_spins):
    """A template without ``{spin}`` would overwrite one file per spin."""
    if args.format in ("json", "both") and n_spins > 1 and "{spin}" not in args.output_template:
        parser.error(
            f"--output-template {args.output_template!r} has no {{spin}} "
            f"placeholder but {n_spins} spins were requested; every spin would "
            "overwrite the same file"
        )


def _input_mpf(value):
    """Parse real CLI literals exactly, independent of the working precision."""
    from .numeric_input import parse_number

    try:
        number = parse_number(value)
        converted = mp.convert(number)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        try:
            nonfinite = not mp.isfinite(mp.convert(value))
        except (TypeError, ValueError, AttributeError, ArithmeticError):
            nonfinite = False
        description = "not a finite number" if nonfinite else "not a real number"
        raise ValueError(f"{value!r} is {description}") from None
    if isinstance(converted, mp.mpc) or not mp.isfinite(converted):
        raise ValueError(f"{value!r} is not a finite real number")
    return number


def _output_path(args, fname):
    """Join ``fname`` onto ``-o`` and create any subdirectory it names."""
    path = os.path.join(args.output, fname)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def _run_super_q_gate(parser, args, *, family, c, h_ext, hbar_ext, q_order, sector, branch):
    """The NS/Ramond q-order gate (their tables are verified against a same-order sampler, blind to truncation);
    returns the digits recorded as ``q_truncation_digits`` (None when not measured)."""
    if args.q_gate == "off":
        return None
    from .qgate import QConvergenceError, act_on_q_report, super_q_tail_report

    worst = None
    for ext in {tuple(h_ext), tuple(hbar_ext or h_ext)}:
        report = super_q_tail_report(family=family, c=c, h_externals=list(ext), q_order=q_order,
                                     n_terms=int(args.lam) + 2, coordinate=args.coordinates[0], sector=sector,
                                     branch=branch, center=args.z_center, chiral_norm=args.chiral_norm,
                                     z_shift=args.z_shift)
        try:
            act_on_q_report(report, mode=args.q_gate, label=f"{family} tables")
        except QConvergenceError as exc:
            parser.error(str(exc))
        if report.get("digits") is not None and (worst is None or report["digits"] < worst):
            worst = report["digits"]
    return None if worst is None else min(float(worst), float(args.precision))


def _refuse_inapplicable_sphere_flags(parser, args, raw_args):
    """Reject sphere-table flags that do not apply to the selected operation."""
    if args.surface == "torus":
        return

    def given(flag):
        return any(token == flag or token.startswith(flag + "=") for token in raw_args)

    problems = []
    def listed(flags):
        return ", ".join(flags) + (" is" if len(flags) == 1 else " are")

    torus = [flag for flag in ("--torus-onept-ext", "--torus-scalar-shift") if given(flag)]
    if torus:
        problems.append(f"{listed(torus)} for --surface torus only")
    if args.large_spin_lmin is None:
        tower = [flag for flag in ("--large-spin-twists", "--large-spin-double-twists", "--large-spin-pole-selection")
                 if given(flag)]
        if tower:
            problems.append(f"{listed(tower)} for the large-spin tower, which needs --large-spin-lmin")
    super_family = args.block_family in {"ns", "ramond"} or args.basis in _SUPER_BASES
    ramond = args.block_family == "ramond" or args.basis.startswith("ramond_")
    is_global = args.block_family == "global" or args.basis in _GLOBAL_BASES
    tower = args.large_spin_lmin is not None
    family_name = "global" if is_global else ("NS" if super_family else "Virasoro")
    if super_family:
        unused = [flag for flag in ("--sample-via",) if given(flag)]
        if unused:
            problems.append(f"{listed(unused)} not implemented for the NS/Ramond families (their tables sample the "
                            "block numerically and are verified against it)")
    # Restrict sector and branch options to the families that use them.
    if not super_family:
        n1 = [flag for flag in ("--super-sector", "--ramond-branch") if given(flag)]
        if n1:
            problems.append(f"{listed(n1)} for the N=1 superconformal blocks (--family ns / ramond); the "
                            f"{family_name} blocks have no sector or sign branch")
    elif not ramond and given("--ramond-branch"):
        problems.append("--ramond-branch is for --family ramond; the NS blocks have no three-point sign branch")
    # --pole-selection ranks the poles an order-reduced (integer --npoles) table keeps; with every pole kept there
    # is nothing to rank. (A fixed-spin global table without --npoles does use it: it ranks the pool it grows.
    # The large-spin tower refuses the flag itself: it has --large-spin-pole-selection.)
    if given("--pole-selection") and args.npoles is None and not tower and not (is_global and args.method != "full"):
        problems.append(
            "--pole-selection chooses the poles an order-reduced table keeps and needs an integer --npoles; "
            + ("--npoles all" if args.method == "full" else "without --npoles the table")
            + " keeps every pole, so there is nothing to select")
    if args.method == "full" and not is_global and not super_family and not tower:
        # --npoles all writes the exact table directly: no interpolation nodes, no sampler
        nodes = [flag for flag in ("--sample-via", "--shared-sample-points", "--no-shared-sample-points")
                 if given(flag)]
        if nodes:
            problems.append(f"{listed(nodes)} not used with --npoles all (the exact table is written directly, "
                            "with no interpolation nodes to sample)")
    if is_global and given("--sample-via"):
        problems.append("--sample-via is not used by the global blocks (they interpolate the numeric 2F1 block "
                        "directly)")
    elif tower and given("--sample-via") and not super_family:
        problems.append("--sample-via is not used by the large-spin tower (it always samples the block "
                        "numerically)")
    if (super_family or is_global) and given("--kac-limit"):
        problems.append(f"--kac-limit chooses the Virasoro block on a Kac weight of a singular c; the "
                        f"{family_name} tables do not read it"
                        + (" (the N=1 blocks keep the c -> c0 limit at fixed weight)" if super_family else ""))
    if is_global and given("--q-gate") and args.q_gate != "off":
        problems.append(f"--q-gate {args.q_gate} has nothing to check for the global blocks (the exact 2F1 block "
                        "has no q-series truncation)")
    if problems:
        parser.error("; ".join(problems))


def _refuse_chiral_2d_flags(parser, args, raw_args, basis):
    """A chiral table has no exchanged spin and no zbar half: refuse the flags that set them (each was dropped).

    ``--spin-ranges 0`` (the default spelled out) is accepted; ``--u-shift`` shifts z alone here (its zbar copy is
    dropped), while an explicit ``--zbar-shift`` / ``--zbar-center`` is an error.
    """
    if _flag_given(raw_args, "--spin-ranges"):
        try:
            spins = _parse_spins(args.spin_ranges)
        except ValueError as exc:
            parser.error(str(exc))
        if any(s != 0 for s in spins):
            parser.error(f"--spin-ranges does not apply to --basis {basis}: the chiral block F(h, z) has no "
                         "exchanged spin (one table, tagged 'chiral')")
    bad = [flag for flag in ("--zbar-shift", "--zbar-center") if _flag_given(raw_args, flag)]
    if bad:
        parser.error(f"{', '.join(bad)} {'is' if len(bad) == 1 else 'are'} not valid for --basis {basis}: the "
                     "chiral block has no zbar half (use --z-shift / --z-center)")
    args.zbar_shift = None      # (the zbar copy of --u-shift)


def _validate_centers(parser, args, *, chiral):
    """Validate --z-center / --zbar-center against the requested coordinates.

    Off-symmetric / off-diagonal expansion centers are supported only for the
    non-folded 2d and chiral coordinates (the folded fold geometry is defined
    relative to the symmetric center). --zbar-center is a 2d-only notion (a
    chiral half has no zbar) and requires --z-center. Shared by the fixed-spin
    and large-spin CLI paths.
    """
    if args.zbar_center is not None and args.z_center is None:
        parser.error("--zbar-center requires --z-center")
    if args.z_center is None and args.zbar_center is None:
        return
    for flag, value in (("--z-center", args.z_center), ("--zbar-center", args.zbar_center)):
        if value is not None and not 0 < value < 1:
            parser.error(f"{flag} must be a cross ratio in (0, 1), got {mp.nstr(value, 12)}")
    bad = [
        cd for cd in args.coordinates
        if cd in _FOLDED_COORDINATES or cd in _RADIAL_COORDINATES
    ]
    if bad:
        parser.error(
            "--z-center/--zbar-center are not supported for folded/radial "
            f"coordinates {sorted(bad)}; use a non-folded 2d (zzb/yyb/qqb) "
            "or chiral (z/y/q) coordinate."
        )


def _super_main(parser, args, raw_args, argv):
    """CLI entry point for the selectable NS/Ramond block families."""
    from .superconformal_blocks import (
        FAMILY_NS,
        FAMILY_RAMOND,
        normalize_sector,
        resolve_super_c,
    )

    if args.surface != "sphere":
        parser.error("superconformal blocks are available only on the sphere")
    if args.large_spin_lmin is not None:
        parser.error("the large-spin tower does not yet support superconformal families")
    if args.npoles == "all" or getattr(args, "method", None) == "full":
        parser.error(
            "NS/Ramond CLI tables currently use the numeric/interpolated path; "
            "use an integer --npoles (or omit it), not --npoles all"
        )
    if args.central_charge is None:
        parser.error("--central-charge is required for NS/Ramond block families")
    c = _input_mpf(args.central_charge)
    family = args.block_family

    # Explicit family-prefixed bases are the durable on-disk spelling.  The
    # short chiral/full names remain convenient aliases when a family is set.
    if args.basis in _SUPER_BASES:
        basis = args.basis
        inferred = "ramond" if args.basis.startswith("ramond_") else "ns"
        if family not in {"auto", inferred}:
            parser.error(
                f"--basis {args.basis} conflicts with --family {family}"
            )
        family = inferred
    elif family in {"ns", "ramond"}:
        prefix = "ramond" if family == "ramond" else "ns"
        if args.basis == "auto":
            chiral_requested = all(co in _CHIRAL_COORDINATES for co in args.coordinates)
            if chiral_requested:
                basis = f"{prefix}_chiral_F"
            elif args.parity_project is not None:
                basis = f"{prefix}_parity_projected_B"
            elif _cli_spinning_externals(args):
                # the values decide, as on the Virasoro path: a scalar spelled
                # --h-ext h --hbar-ext h (or --spin-ext 0 0 0 0) is scalar_G
                basis = f"{prefix}_ordered_B"
            else:
                basis = f"{prefix}_scalar_G"
        elif args.basis in {"chiral_F", "scalar_G", "ordered_B", "parity_projected_B"}:
            basis = f"{prefix}_{args.basis}"
        else:
            parser.error(f"--basis {args.basis} is not a {family} basis")
    else:
        parser.error(
            "select --family ns or ramond, or use an explicit ns_* / "
            "ramond_* basis"
        )

    is_chiral = basis in _SUPER_CHIRAL_BASES
    if is_chiral:
        _refuse_chiral_2d_flags(parser, args, raw_args, basis)
    if is_chiral and args.zbar_center is not None:
        parser.error(
            "--zbar-center is not valid for a chiral basis; use --z-center alone"
        )
    if args.coordinates and (
        any(co in _CHIRAL_COORDINATES for co in args.coordinates)
        != is_chiral
    ):
        parser.error(
            "superconformal chiral bases require only z/y/q, while full bases "
            "require only 2d or folded/radial coordinates"
        )
    if is_chiral and args.hbar_ext is not None:
        parser.error("--hbar-ext is not valid with a superconformal chiral basis")
    if is_chiral and (args.delta_ext is not None or args.spin_ext is not None):
        parser.error("superconformal chiral bases take external weights through --h-ext")
    if not is_chiral and basis.endswith("ordered_B") and any(
        co in _FOLDED_COORDINATES or co in _RADIAL_COORDINATES
        for co in args.coordinates
    ):
        parser.error("ordered_B has no parity fold; use scalar_G or parity_projected_B")
    if basis.endswith("parity_projected_B") and args.parity_project is None:
        parser.error("parity-projected superconformal bases require --parity-project +/-")

    using_chiral_halves = args.h_ext is not None or args.hbar_ext is not None
    using_delta_spin = args.delta_ext is not None or args.spin_ext is not None
    if using_chiral_halves and using_delta_spin:
        parser.error("--h-ext/--hbar-ext and --delta-ext/--spin-ext are mutually exclusive")
    if args.h_ext is not None:
        h_ext = [_input_mpf(h) for h in args.h_ext]
        hbar_ext = (
            None if args.hbar_ext is None
            else [_input_mpf(h) for h in args.hbar_ext]
        )
    else:
        delta_ext = [_input_mpf(d) for d in (args.delta_ext or ["0"] * 4)]
        spin_ext = [_input_mpf(s) for s in (args.spin_ext or ["0"] * 4)]
        with mp.workdps(3 * mp.mp.dps + 50):
            h_ext = [(d + s) / 2 for d, s in zip(delta_ext, spin_ext)]
            hbar_ext = [(d - s) / 2 for d, s in zip(delta_ext, spin_ext)]
    if is_chiral:
        if args.h_ext is None:
            parser.error("superconformal chiral bases require --h-ext")
        hbar_ext = None
    elif hbar_ext is None:
        hbar_ext = list(h_ext)

    if not is_chiral:
        spinning = any(h != hb for h, hb in zip(h_ext, hbar_ext))
        if basis.endswith("scalar_G") and spinning:
            parser.error("scalar_G requires scalar external weights")

    # All sphere q-recursive families share the same public truncation knob.
    # The NS/R recurrence uses half-levels internally, but its user-facing
    # cutoff is still the ordinary q-order.
    q_order = args.order
    if q_order is None or q_order < 0:
        parser.error("superconformal blocks require a non-negative q-order")
    # Direct super-block evaluation is fail-closed at exchanged Kac poles.
    # Interpolation samplers use their private off-pole probe path; no unsafe
    # bypass is exposed by the CLI or the canonical table backend.
    # Omitting --npoles retains every pole of the q_order-truncated block.
    # The super table backend verifies the resulting representation.
    npoles = None if args.npoles is None else int(args.npoles)
    if npoles is not None and npoles < 0:
        parser.error("--npoles must be >= 0")
    sector = normalize_sector(args.super_sector)
    branch = args.ramond_branch
    parity_external = normalize_external_parity(args.parity_ext)
    parity_exchanged = args.parity_exchanged or "1"
    # the table's c-offset follows its working precision (half its digits; super_table_backends); the note names
    # the one at --precision, and each file records the one it was built with
    from .super_table_backends import super_table_offset_eps_exp

    super_eps_exp = super_table_offset_eps_exp(args.precision)
    c_input, c_eval, regularized = resolve_super_c(
        c, regularize=args.regularize, eps_exp=super_eps_exp
    )
    from .super_laurent import exact_singular_c

    if regularized and exact_singular_c(c_input) is not None:
        print(f"note: c = {mp.nstr(c_input, 12)} is an N=1 singular central charge: the recursion runs exactly there "
              "(Laurent in t, merged poles)", flush=True)
    elif regularized:
        print(
            f"note: superconformal recursion evaluated at c+eps = "
            f"{mp.nstr(c_eval, 12)} (input c={mp.nstr(c_input, 12)})",
            flush=True,
        )

    try:
        spins = [mp.mpf(0)] if is_chiral else _parse_spins(args.spin_ranges)
    except ValueError as exc:
        parser.error(str(exc))
    if not is_chiral and any(s != int(s) for s in spins):
        parser.error("superconformal full blocks currently require integer exchanged spins")
    _check_output_template(parser, args, len(spins))
    _check_damping_base(parser, args, is_global=False)

    # Route superconformal tables through the same canonical family registry as
    # Virasoro/global/torus tables.  The adapter owns the numerical backend;
    # this function is responsible only for CLI validation and artifact I/O.
    from .requests import BlockRequest, NumericalPolicy, RequestValidationError
    from .table_adapters import table_adapter_for
    from .super_table_backends import SuperBackendOptions
    from .api import build_table_request

    try:
        super_table_request = BlockRequest(
            operation="table",
            surface="sphere",
            family=family,
            basis=basis,
            coordinate=args.coordinates[0],
            coordinates=tuple(args.coordinates),
            h_ext=tuple(h_ext),
            hbar_ext=None if hbar_ext is None else tuple(hbar_ext),
            central_charge=c_input,
            sector=sector,
            branch=branch,
            parity=args.parity_project,
            parity_external=(
                None if parity_external is None else tuple(parity_external)
            ),
            parity_exchanged=parity_exchanged,
            chiral_norm=args.chiral_norm,
            z_shift=args.z_shift,
            zbar_shift=args.zbar_shift,
            center=args.z_center,
            center_bar=args.zbar_center,
            gap=args.gap,
            policy=NumericalPolicy(
                precision=args.precision,
                order=q_order,
                lambda_order=args.lam,
                npoles=npoles,
                pole_selection=args.pole_selection,
                c_regularization=args.regularize,
                target_digits=args.target_digits,
                max_precision=args.max_precision,
            ),
        )
        super_table_adapter = table_adapter_for(family)
        super_table_adapter.validate(super_table_request)
    except RequestValidationError as exc:
        parser.error(str(exc))

    os.makedirs(args.output, exist_ok=True)
    metadata_args = getattr(args, "_cli_invocation_args", raw_args)
    cli_args_str = shlex.join(["virasoro_blocks", *metadata_args])

    super_backend_options = SuperBackendOptions(
        eps_exp=None,
        num_threads=args.num_threads,
        shared_sample_points=args.shared_sample_points,
    )

    print(
        f"{family} superconformal blocks ({basis}): c={mp.nstr(c_input, 10)}, "
        f"sector={args.super_sector}, q-order={q_order}, coordinates={args.coordinates}",
        flush=True,
    )
    q_digits = _run_super_q_gate(parser, args, family=family, c=c_input, h_ext=h_ext, hbar_ext=hbar_ext,
                                 q_order=q_order, sector=sector, branch=branch)
    for ell_value in spins:
        ell = 0 if is_chiral else int(ell_value)
        serialized = build_table_request(
            super_table_request,
            spin=ell,
            backend_options=super_backend_options,
        )
        serialized.q_truncation_digits = q_digits
        basis_for_writer = basis
        # the chiral table is one file, tagged "chiral" as on the Virasoro path
        fname = args.output_template.replace(
            "{spin}", "chiral" if is_chiral else str(ell)
        )
        # written at the precision it was built and verified at; npoles records the requested kept-pole count, as
        # the Virasoro tables (and the bootstrap manifest) do -- at c < 3/2 the always-kept x > 0 poles come on top,
        # and the pole list gives the kept ones -- or, with every pole kept, their number
        write_dps = max(mp.mp.dps, getattr(serialized, "working_dps", None) or mp.mp.dps)
        kept_npoles = len(serialized.pole_list_x) if args.npoles is None else int(args.npoles)
        with mp.workdps(write_dps):
            _write_super_outputs(
                args, serialized, fname, ell, basis_for_writer, kept_npoles, c_input, c_eval, h_ext, hbar_ext,
                is_chiral, parity_external, parity_exchanged, q_order, cli_args_str, family, branch, regularized,
            )


def _write_super_outputs(args, serialized, fname, ell, basis_for_writer, npoles, c_input, c_eval, h_ext, hbar_ext,
                         is_chiral, parity_external, parity_exchanged, q_order, cli_args_str, family, branch,
                         regularized):
    from .superconformal_blocks import FAMILY_RAMOND, resolve_super_c
    from .super_table_backends import super_table_offset_eps_exp

    from .super_laurent import exact_singular_c

    eps_exp = getattr(serialized, "regularization_eps_exp", None)
    if eps_exp is not None:     # the verification may have deepened the offset for this spin
        _c, c_eval, _regularized = resolve_super_c(c_input, regularize=args.regularize, eps_exp=eps_exp)
    # at an exactly singular c: the exact Laurent recursion at c itself, its merged poles the skeleton
    exact = regularized and exact_singular_c(c_input) is not None
    policy_tag = "merged" if exact else ("c+eps" if regularized else "direct")
    if exact:
        c_eval = c_input
    if True:
        text = write_json(
            serialized, c=c_input, h_ext=h_ext,
            hbar_ext=None if is_chiral else hbar_ext,
            spin=ell, q_order=q_order, npoles=npoles, basis=basis_for_writer,
            pole_selection=args.pole_selection if args.npoles is not None else None,
            parity_projection=None if is_chiral else args.parity_project,
            parity_external=None if is_chiral else parity_external,
            parity_exchanged=None if is_chiral else parity_exchanged,
            chiral_norm=args.chiral_norm, precision_input=args.precision,
            args=cli_args_str, z_center=args.z_center,
            zbar_center=None if is_chiral else args.zbar_center,
            z_shift=args.z_shift,
            zbar_shift=None if is_chiral else args.zbar_shift,
            block_family=family, sector=args.super_sector,
            ramond_branch=branch if family == FAMILY_RAMOND else None,
            central_charge_evaluated=c_eval,
            regularization_policy=policy_tag,
            regularization_eps_exp=None if exact else (getattr(serialized, "regularization_eps_exp", None)
                                    or super_table_offset_eps_exp(serialized.working_dps or args.precision)),
            collision_policy="raise",
        )
        if args.format in ("json", "both"):
            path = _output_path(args, fname)
            with open(path, "w") as fh:
                fh.write(text)
                fh.write("\n")
            print(f"  wrote {path}", flush=True)
        if args.format in ("m", "both"):
            for coord in args.coordinates:
                mtext = write_mathematica(
                    serialized, coordinate=coord, ell=ell, basis=basis_for_writer,
                    parity_projection=None if is_chiral else args.parity_project,
                    parity_external=None if is_chiral else parity_external,
                    parity_exchanged=None if is_chiral else parity_exchanged,
                    chiral_norm=args.chiral_norm, z_center=args.z_center,
                    zbar_center=None if is_chiral else args.zbar_center,
                    z_shift=args.z_shift,
                    zbar_shift=None if is_chiral else args.zbar_shift,
                    c=c_input, h_ext=h_ext,
                    hbar_ext=None if is_chiral else hbar_ext,
                    q_order=q_order, npoles=npoles,
                    pole_selection=args.pole_selection if args.npoles is not None else None,
                    precision_input=args.precision, args=cli_args_str,
                    block_family=family, sector=args.super_sector,
                    ramond_branch=branch if family == FAMILY_RAMOND else None,
                    central_charge_evaluated=c_eval,
                    regularization_policy=policy_tag,
                    regularization_eps_exp=None if exact else (getattr(serialized, "regularization_eps_exp", None)
                                    or super_table_offset_eps_exp(serialized.working_dps or args.precision)),
                    collision_policy="raise",
                )
                # named after the JSON file (--output-template), so a custom template separates runs here too;
                # one file per spin even when the template has no {spin} (a Mathematica-only run allows that)
                stem = fname[:-len(".json")] if fname.endswith(".json") else fname
                if "{spin}" not in args.output_template:
                    stem = f"{stem}_{'chiral' if is_chiral else ell}"
                mpath = _output_path(args, f"{stem}_{coord}.m")
                with open(mpath, "w") as fh:
                    fh.write(mtext)
                print(f"  wrote {mpath}", flush=True)
def _flag_given(raw_args, flag) -> bool:
    """Whether ``flag`` was passed explicitly (``--flag V`` or ``--flag=V``): the only way to tell an explicit value
    from a non-None argparse default."""
    return any(tok == flag or tok.startswith(flag + "=") for tok in raw_args)


def _torus_q_gate(parser, args, table, spin) -> None:
    """``--q-gate`` for a torus one-point table: its ``--order`` truncation is measured against a deeper reference
    (``q_truncation_digits``, :func:`virasoro_blocks.torus_onept.torus_onept_verify_table`); below the sphere gate's
    threshold (20 digits, or --precision if lower) 'warn' prints a warning and 'strict' refuses the table."""
    from .qgate import DEFAULT_MIN_DIGITS

    digits = (table.metadata or {}).get("q_truncation_digits")
    threshold = min(DEFAULT_MIN_DIGITS, float(args.precision))
    if args.q_gate == "off" or digits is None or digits >= threshold:
        return
    message = (
        f"spin {spin}: --order {args.order} truncates the torus one-point table at ~{digits:.1f} digits "
        f"(below {threshold:.0f}); raise --order"
    )
    if args.q_gate == "strict":
        parser.error(f"q-gate (strict): {message}")
    print(f"warning: q-gate: {message}", file=sys.stderr, flush=True)


def _torus_main(args, parser, raw_args, order_was_provided, argv):
    """--surface torus: emit exact genus-1 reduced-character tables (JSON).

    One file per integer spin sector j >= 0, basis torus_G on the vvb
    coordinate, written by `writers.write_torus_json` (empty pole list,
    base e^{-2 pi}, symmetrized (j, -j) pair tables). Sphere-only flags are
    rejected loudly rather than silently ignored.
    """
    from .api import build_table_request
    from .table_adapters import table_adapter_for
    from .writers import write_torus_json

    if args.central_charge is None:
        parser.error("--surface torus requires --central-charge")
    if args.block_family not in {"auto", "torus-character", "torus-onepoint"}:
        parser.error(
            "--surface torus requires --family torus-character or "
            "--family torus-onepoint"
        )
    if args.block_family == "torus-character" and args.torus_onept_ext is not None:
        parser.error(
            "--family torus-character cannot be combined with "
            "--torus-onept-ext; choose --family torus-onepoint"
        )
    if args.block_family == "torus-onepoint" and args.torus_onept_ext is None:
        parser.error(
            "--family torus-onepoint requires --torus-onept-ext"
        )
    if args.format != "json":
        parser.error("--surface torus emits JSON only; use --format json")
    onept = args.torus_onept_ext is not None
    if onept and args.method == "full":
        parser.error(
            "--npoles all is not supported for torus one-point tables; "
            "pass an integer kept-pole count"
        )

    def given(flag):
        return _flag_given(raw_args, flag)

    # Flags with non-None defaults that no torus table reads: only raw_args tells an explicit value from the
    # default, and an explicit one would otherwise be silently ignored.
    inert = {
        "--pole-selection": "the one-point tables keep the Kac poles nearest the positivity half-line "
                            "(conjugate pairs together); there is no selector to choose",
        "--sample-via": "the one-point tables always sample the recursion numerically",
        "--shared-sample-points": "each table has one node set",
        "--no-shared-sample-points": "each table has one node set",
        "--regularize": "a recognized singular c is routed to the exact Laurent kernel automatically",
        "--num-threads": "torus tables are built serially",
        "--super-sector": "no N=1 superconformal torus family",
        "--ramond-branch": "no N=1 superconformal torus family",
        "--large-spin-double-twists": "no large-spin torus tower",
        "--large-spin-twists": "no large-spin torus tower",
        "--large-spin-pole-selection": "no large-spin torus tower",
    }
    if not onept:
        inert["--q-gate"] = "character tables are exact (no q-series truncation)"
        inert["--torus-scalar-shift"] = "it anchors torus ONE-POINT tables (--torus-onept-ext)"
    bad = [flag for flag in inert if given(flag)]
    if bad:
        parser.error(
            "--surface torus does not use "
            + "; ".join(f"{flag} ({inert[flag]})" for flag in bad)
        )
    rejected = [
        (args.delta_ext is not None, "--delta-ext"),
        (args.h_ext is not None, "--h-ext"),
        (args.hbar_ext is not None, "--hbar-ext"),
        (args.spin_ext is not None, "--spin-ext"),
        (args.parity_ext is not None, "--parity-ext"),
        (args.parity_project is not None, "--parity-project"),
        (args.parity_exchanged is not None, "--parity-exchanged"),
        # --basis auto lets the family choose its basis (torus_G / torus_onept_G); the
        # torus tables use vvb, so any explicit --coordinates (even the sphere default zzb) is refused.
        (given("--basis") and args.basis != "auto", "--basis"),
        (given("--coordinates"), "--coordinates"),
        (order_was_provided and not onept, "--order"),
        (given("--npoles") and not onept, "--npoles"),
        (args.chiral_norm is not None, "--chiral-norm"),
        (args.z_shift is not None, "--z-shift"),
        (args.zbar_shift is not None, "--zbar-shift"),
        (args.u_shift is not None, "--u-shift"),
        (args.z_center is not None, "--z-center"),
        (args.zbar_center is not None, "--zbar-center"),
        (args.large_spin_lmin is not None, "--large-spin-lmin"),
    ]
    offending = [flag for hit, flag in rejected if hit]
    if offending:
        parser.error(
            f"{', '.join(offending)} do(es) not apply to --surface torus; "
            "torus tables use vvb and carry no sphere kinematics"
        )

    spins = []
    for value in _parse_spins(args.spin_ranges):
        if value != int(value) or value < 0:
            parser.error(
                "--surface torus requires integer spins >= 0 "
                f"(got {mp.nstr(value, 6)})"
            )
        spins.append(int(value))

    _check_output_template(parser, args, len(spins))

    c = _input_mpf(args.central_charge)
    if not onept and not c > 1:
        # the tables are the Verma characters, the characters of the non-degenerate modules at any c; at c <= 1 the
        # Kac weights of the half-line are degenerate modules, whose irreducible characters are point tables
        print(
            f"note: at c = {mp.nstr(c, 8)} <= 1 a character table is the non-degenerate (Verma) character; a module "
            "on a Kac weight (the vacuum, a minimal-model primary, ...) has the irreducible character of "
            "`virasoro_blocks point-table --surface torus`",
            file=sys.stderr, flush=True,
        )

    os.makedirs(args.output, exist_ok=True)
    metadata_args = getattr(args, "_cli_invocation_args", raw_args)
    cli_args_str = shlex.join(["virasoro_blocks", *metadata_args])

    if onept:
        adapter = table_adapter_for("torus-onepoint")
        d_ext = _input_mpf(args.torus_onept_ext)
        q_order = args.order
        npoles = args.npoles     # (None: chosen per spin from the measured digits, as for the global tables)
        # The anchor (positivity edge): --gap G puts every spin at |j| + G, the sphere meaning of a uniform gap;
        # --torus-scalar-shift X puts spin 0 alone at X (the other spins at |j|).  Both name the spin-0 anchor, so
        # together they are refused. The anchor may be 0: the double Kac pole at Delta = 0 stays in the prefactor
        # (the nodes lift the first point off it); below 0 it is under the unitarity bound.
        scalar_shift = (
            _input_mpf(args.torus_scalar_shift)
            if args.torus_scalar_shift is not None
            else None
        )
        if scalar_shift is not None and args.gap is not None:
            parser.error(
                "--gap and --torus-scalar-shift both set the spin-0 anchor: --gap G anchors every spin at "
                "|j| + G (spin 0 at G); --torus-scalar-shift X anchors spin 0 alone at X. Pass one"
            )
        if scalar_shift is not None and 0 not in spins:
            parser.error(
                "--torus-scalar-shift anchors the spin-0 table only, and spin 0 is not in --spin-ranges "
                "(use --gap to raise every spin's anchor)"
            )
        gap = _input_mpf(args.gap) if args.gap is not None else None
        spin0_anchor = gap if gap is not None else scalar_shift
        if 0 in spins and spin0_anchor is not None and spin0_anchor < 0:
            parser.error(
                f"the spin-0 anchor {mp.nstr(spin0_anchor, 12)} is below the unitarity bound 0: pass "
                "--torus-scalar-shift X >= 0 or --gap G >= 0"
            )
        request = adapter.request(
            central_charge=c,
            torus_onepoint_ext=d_ext,
            torus_scalar_shift=scalar_shift,
            gap=gap,
            lambda_order=args.lam,
            q_order=q_order,
            npoles=npoles,
            precision=args.precision,
            target_digits=args.target_digits,
            max_precision=args.max_precision,
        )
        print(
            f"torus one-point tables: c={args.central_charge}, "
            f"d_ext={args.torus_onept_ext}, lambda={args.lam}, "
            f"spins {spins}, order={q_order}, npoles={'chosen' if npoles is None else npoles}, "
            f"dps={args.precision}",
            flush=True,
        )
        # one recursion state and one verification reference (deeper in q, more digits) for every spin
        kernel = adapter.build_kernel(request)
        reference = adapter.build_reference(request, kernel)
        for spin in spins:
            print(f"  spin {spin} ...", flush=True)
            table = build_table_request(
                request,
                spin=spin,
                kernel=kernel,
                reference=reference,
            )
            _torus_q_gate(parser, args, table, spin)
            with mp.workdps(max(mp.mp.dps, int(getattr(table, "working_dps", None) or 0))):   # (its working precision)
                text = write_torus_json(
                    table,
                    c=c,
                    spin=spin,
                    basis="torus_onept_G",
                    onept_d_ext=d_ext,
                    q_order=q_order,
                    npoles=None if npoles is None else int(npoles),
                    npoles_chosen=table.npoles_chosen,
                    # (a recognized singular c: the exact Laurent kernel, the c -> c0 limit at fixed weight)
                    regularization_policy=("laurent" if recognized_singular_locus(c) is not None else "direct"),
                    precision_input=args.precision,
                    spin_input=spins,
                    output_template=args.output_template,
                    args=cli_args_str,
                )
            fname = args.output_template.replace("{spin}", str(spin))
            path = _output_path(args, fname)
            with open(path, "w") as f:
                f.write(text)
                f.write("\n")
            print(f"    wrote {path}", flush=True)
        return 0

    print(
        f"torus reduced-character tables: c={args.central_charge}, "
        f"lambda={args.lam}, spins {spins}, dps={args.precision}",
        flush=True,
    )
    adapter = table_adapter_for("torus-character")
    request = adapter.request(
        central_charge=c,
        lambda_order=args.lam,
        precision=args.precision,
        target_digits=args.target_digits,
        max_precision=args.max_precision,
    )
    for spin in spins:
        print(f"  spin {spin} ...", flush=True)
        table = build_table_request(request, spin=spin)
        with mp.workdps(max(mp.mp.dps, int(getattr(table, "working_dps", None) or 0))):   # (its working precision)
            text = write_torus_json(
                table,
                c=c,
                spin=spin,
                precision_input=args.precision,
                spin_input=spins,
                output_template=args.output_template,
                args=cli_args_str,
            )
        fname = args.output_template.replace("{spin}", str(spin))
        path = _output_path(args, fname)
        with open(path, "w") as f:
            f.write(text)
            f.write("\n")
        print(f"    wrote {path}", flush=True)
    return 0


_OPERATIONS = ("table", "evaluate", "point-table", "info")


def main(argv=None, *, invocation_args=None):
    """The ``virasoro_blocks`` command.  A user-facing error (a bad value, a refused combination) is printed as one
    line and exits with status 2; set ``VB_TRACEBACK=1`` for the full traceback."""
    try:
        return _main(argv, invocation_args=invocation_args)
    except (ValueError, ArithmeticError, OSError) as exc:
        if os.environ.get("VB_TRACEBACK"):
            raise
        print(f"virasoro_blocks: error: {exc}", file=sys.stderr)
        print("(set VB_TRACEBACK=1 to see the full traceback)", file=sys.stderr)
        return 2


def _main(argv=None, *, invocation_args=None):
    raw_args = sys.argv[1:] if argv is None else list(argv)
    # Keep the user-facing operation in artifact provenance even though the
    # legacy table parser receives the arguments after ``table`` has been
    # stripped.  This lets regenerated samples advertise the canonical
    # interface instead of looking like old root-level invocations.
    if invocation_args is None:
        invocation_args = tuple(raw_args)
    else:
        invocation_args = tuple(invocation_args)

    # New operation-oriented commands are dispatched before the historical
    # table parser.  ``table`` is intentionally a transparent alias for the
    # old root invocation so existing scripts and JSON artifacts remain stable.
    if not raw_args:
        _print_root_help()
        return 0
    if raw_args in (["--version"], ["-V"]):
        from ._version import __version__
        print(f"virasoro_blocks {__version__}")
        return 0
    subcommand = raw_args[0] if raw_args and not raw_args[0].startswith("-") else None
    if subcommand is not None and subcommand not in _OPERATIONS:
        import difflib
        close = difflib.get_close_matches(subcommand, _OPERATIONS, n=1)
        hint = f"; did you mean {close[0]!r}?" if close else ""
        print(f"virasoro_blocks: error: unknown operation {subcommand!r} (choose from {', '.join(_OPERATIONS)}){hint}",
              file=sys.stderr)
        return 2
    if subcommand == "evaluate":
        from .cli.evaluate import main as evaluate_main
        return evaluate_main(raw_args[1:])
    if subcommand == "info":
        from .cli.info import main as info_main
        return info_main(raw_args[1:])
    if subcommand == "point-table":
        from .cli.point_table import main as point_table_main
        return point_table_main(raw_args[1:])
    if subcommand == "table":
        raw_args = raw_args[1:]

    if subcommand is None and raw_args in (["--help"], ["-h"]):
        _print_root_help()
        return 0

    parser = _build_parser(
        prog="virasoro_blocks table" if subcommand == "table" else "virasoro_blocks"
    )
    # "--ramond-branch --" / "--ramond-branch -+": argparse reads a value that looks like an option as one
    check_fixed_value_counts(parser, raw_args)
    args = parser.parse_args(join_option_values(raw_args, "--ramond-branch",
                                                RAMOND_BRANCHES))
    args._cli_invocation_args = invocation_args
    if args.surface == "torus" and _flag_given(raw_args, "--kac-limit"):
        parser.error("--kac-limit chooses the Virasoro sphere block on a Kac weight of a singular c; the torus "
                     "tables do not read it (their singular-c blocks are exact traces over irreducible modules)")
    set_kac_limit(args.kac_limit)
    if args.basis not in _CLI_BASES:
        parser.error(
            f"unknown --basis {args.basis!r}; use `virasoro_blocks info` "
            "to see the supported family-prefixed bases"
        )
    order_was_provided = _flag_given(raw_args, "--order")
    if args.lam < 0:
        parser.error("--lambda must be >= 0")
    if args.order is not None and args.order < 0:
        parser.error("--order must be >= 0")
    if args.precision < 10:
        parser.error("--precision must be >= 10")

    # --npoles all selects the exact symbolic form. An integer selects
    # order-reduced interpolation; omission keeps all poles through interpolation.
    # Both forms use --precision and record their measured accuracy.
    if args.npoles == "all":
        args.method = "full"
        args.npoles = None
    else:
        args.method = "interpolation"

    mp.mp.dps = args.precision
    # --precision is the working precision; --target-digits (opt-in) asks for verified digits, escalating from it
    # (verified_build.run_verified_build)
    from .verified_build import PrecisionRequest

    try:
        args.precision_request = PrecisionRequest.make(args.precision, args.target_digits, args.max_precision)
    except ValueError as exc:
        parser.error(str(exc))

    if (
        args.block_family in {"torus-character", "torus-onepoint"}
        and args.surface != "torus"
    ):
        parser.error(
            f"--family {args.block_family} requires --surface torus"
        )
    if args.block_family == "global":
        if args.large_spin_lmin is not None:
            # Large-spin tables use the same short basis names as the
            # fixed-spin path. The family flag selects the exact 2F1 tower;
            # normalize an explicitly family-prefixed spelling for the shared
            # tower dispatcher below.
            if args.basis in _GLOBAL_BASES:
                args.basis = args.basis[len("global_"):]
            elif args.basis not in {
                "auto", "scalar_G", "ordered_B", "parity_projected_B",
            }:
                parser.error(
                    f"--basis {args.basis} is not a global large-spin basis"
                )
        else:
            if args.basis == "auto":
                parser.error(
                    "--family global requires an explicit global_* basis (or a "
                    "short basis such as --basis scalar_G)"
                )
            if args.basis in {"scalar_G", "ordered_B", "parity_projected_B", "chiral_F"}:
                args.basis = f"global_{args.basis}"
            elif not args.basis.startswith("global_"):
                parser.error(f"--basis {args.basis} is not a global basis")
    elif args.block_family == "virasoro" and args.basis.startswith("global_"):
        parser.error("global_* bases require --family global")

    is_global = args.block_family == "global" or args.basis in _GLOBAL_BASES
    if args.gap is not None:
        # --gap moves the POSITIVITY ANCHOR (table_backends.positivity_anchor),
        # honored by the virasoro/global/ns/ramond sphere backends and torus
        # one-point. Any other family would emit the unitarity anchor while
        # reporting success -- a block you believe carries a gap and does not.
        # Reject rather than ignore.
        if args.surface == "torus" and args.torus_onept_ext is None:
            parser.error(
                "--gap applies to torus ONE-POINT tables (--torus-onept-ext); "
                "the character tables carry no Delta positivity anchor"
            )
        if args.block_family not in (None, "auto", "virasoro", "global", "ns", "ramond", "torus-onepoint"):
            parser.error(
                f"--gap is not wired to the {args.block_family!r} family; it "
                "would be silently ignored"
            )
        if args.large_spin_lmin is not None:
            parser.error(
                "--gap does not apply to the large-spin tower: its positivity "
                "variable is the spin L, not Delta, so a Delta anchor is "
                "meaningless there (the tower floor is --large-spin-lmin)"
            )
        if args.surface != "torus":
            # (the torus one-point path checks its spin-0 anchor itself)
            from .requests import sphere_gap_problem

            problem = sphere_gap_problem(args.gap)
            if problem is not None:
                parser.error(problem.replace("{name}", "--gap"))
    # a global table without --npoles keeps its whole pool unless a selection was asked for (table_adapters)
    args.pole_selection_given = args.pole_selection is not None
    if args.pole_selection is None:
        args.pole_selection = "contribution_optimal"
    if args.large_spin_pole_selection is None:
        args.large_spin_pole_selection = "contribution_optimal"
    # The torus surface is a self-contained exact-table branch: no recursion,
    # no interpolation, none of the sphere kinematics. Dispatch before any
    # sphere-specific argument processing.
    if args.surface == "torus":
        return _torus_main(args, parser, raw_args, order_was_provided, argv)
    if args.precision < MIN_TABLE_PRECISION:
        parser.error(
            f"sphere tables need --precision >= {MIN_TABLE_PRECISION} (the "
            "interpolation and pole-recognition tolerances leave no digits "
            "below that)"
        )

    if args.chiral_norm is not None:
        args.chiral_norm = _input_mpf(args.chiral_norm)
    if args.z_shift is not None:
        args.z_shift = _input_mpf(args.z_shift)
    if args.zbar_shift is not None:
        args.zbar_shift = _input_mpf(args.zbar_shift)
    # --u-shift sets both z_shift and zbar_shift unless individually overridden.
    if args.u_shift is not None:
        u_val = _input_mpf(args.u_shift)
        if args.z_shift is None:
            args.z_shift = u_val
        if args.zbar_shift is None:
            args.zbar_shift = u_val
    # Off-symmetric / off-diagonal expansion centers (a cross ratio z0; None =
    # the symmetric point 1/2). center_bar gives an independent zbar-center.
    if args.z_center is not None:
        args.z_center = _input_mpf(args.z_center)
    if args.zbar_center is not None:
        args.zbar_center = _input_mpf(args.zbar_center)

    _refuse_inapplicable_sphere_flags(parser, args, raw_args)
    if args.block_family in {"ns", "ramond"} or args.basis in _SUPER_BASES:
        return _super_main(parser, args, raw_args, argv)

    # Large-spin tower emission is a self-contained branch: it builds
    # fixed-twist, variable-L blocks and writes them in the variable="L"
    # schema, bypassing the fixed-spin Delta path entirely. The tower is a 2d
    # (k_h k_hbar + swap) construction, never chiral, and supports off-center /
    # off-diagonal expansion just like the fixed-spin path.
    if args.large_spin_lmin is not None:
        if _input_mpf(args.large_spin_lmin) < 0:
            parser.error(f"--large-spin-lmin must be >= 0 (the tower covers spins L >= Lmin), got {args.large_spin_lmin}")
        _validate_centers(parser, args, chiral=False)
        _emit_large_spin_tower(parser, args, raw_args, argv)
        return

    explicit_global_basis = args.basis in _GLOBAL_BASES
    if explicit_global_basis and args.order < 1:
        parser.error(
            "--order must be >= 1 for global_* bases (it sizes the SL(2) pole pool)"
        )
    if args.method == "interpolation":
        _check_damping_base(parser, args, is_global=explicit_global_basis)
    if explicit_global_basis:
        if args.method == "full":
            parser.error(
                "--npoles all (keep every pole) is not supported for explicit "
                "global_* bases; the global path only exposes integer-npoles "
                "interpolation"
            )
        if args.regularize == "on":
            parser.error(
                "--regularize on is not supported for explicit global_* bases; "
                "the global path does not use the Virasoro regularization wrapper"
            )
        if args.central_charge is not None:
            print(
                "note: explicit global_* bases are independent of --central-charge; "
                "recording null central_charge metadata",
                file=sys.stderr,
                flush=True,
            )
        c = None
        b = None
        regularize_effective = False
        singular_label = None
    else:
        if args.central_charge is None:
            parser.error("--central-charge is required unless --basis is an explicit global_* basis")
        c = _input_mpf(args.central_charge)
        b = b_from_c(c)

        # Resolve --regularize: auto enables iff c is a recognized singular locus;
        # 'on' / 'off' force the choice; 'off' is refused at a singular locus.
        singular_label = recognized_singular_locus(c)
        if args.regularize == "auto":
            regularize_effective = singular_label is not None
            if regularize_effective:
                print(
                    f"note: c is at a recognized Virasoro singular locus "
                    f"({singular_label}); enabling regularization",
                    flush=True,
                )
            else:
                # Not exactly singular -- but if c is suspiciously close to one,
                # hint that the answer is at the user's c, not at the locus. Common
                # causes: typos, IEEE-double residuals, optimizer outputs.
                nearby = nearby_singular_locus(c)
                if nearby is not None:
                    label, distance = nearby
                    print(
                        f"hint: c is {mp.nstr(distance, 3)} from a recognized "
                        f"Virasoro singular locus ({label}). The output is the "
                        f"block at your exact c, not at the locus -- the block "
                        f"varies non-linearly near these loci, so a small offset "
                        f"in c can translate to a much larger offset in the "
                        f"output. Pass the exact value to land on the locus.",
                        file=sys.stderr, flush=True,
                    )
        elif args.regularize == "on":
            regularize_effective = True
        elif args.regularize == "off":
            regularize_effective = False
            if singular_label is not None:
                parser.error(
                    f"c is a recognized Virasoro singular locus ({singular_label}), where the bare recursion is "
                    f"singular: --regularize off cannot build it. Use --regularize auto (the default)."
                )
        else:  # argparse choices guarantees we don't reach here
            regularize_effective = False
    # External operators come in two equivalent parametrizations:
    #   (Delta_i, s_i):  h_i = (Delta_i + s_i)/2,  hbar_i = (Delta_i - s_i)/2
    #   (h_i, hbar_i):   used directly
    # The chiral_F path skips the 2d composition and takes h_i directly via
    # --h-ext only (--hbar-ext is rejected). For the 2d bases either
    # parametrization is accepted, but the two are mutually exclusive; we
    # resolve below once the basis is known.
    using_chiral_halves = args.h_ext is not None or args.hbar_ext is not None
    using_delta_spin = args.delta_ext is not None or args.spin_ext is not None
    if using_chiral_halves and using_delta_spin:
        parser.error(
            "--h-ext / --hbar-ext and --delta-ext / --spin-ext are mutually "
            "exclusive parametrizations of the external operators; choose one"
        )

    if using_chiral_halves:
        if args.h_ext is None:
            parser.error("--hbar-ext requires --h-ext")
        h_ext = [_input_mpf(h) for h in args.h_ext]
        hbar_ext_provided = (
            [_input_mpf(hb) for hb in args.hbar_ext]
            if args.hbar_ext is not None else None
        )
    else:
        delta_ext = [_input_mpf(d) for d in (args.delta_ext or ["0", "0", "0", "0"])]
        spin_ext = [_input_mpf(s) for s in (args.spin_ext or ["0"] * 4)]
        with mp.workdps(3 * mp.mp.dps + 50):     # (keep the guard digits through the halving)
            h_ext = [(d + s) / 2 for d, s in zip(delta_ext, spin_ext)]
            hbar_ext_provided = [(d - s) / 2 for d, s in zip(delta_ext, spin_ext)]

    if using_chiral_halves and hbar_ext_provided is not None:
        # The (Delta, spin) path enforces half-integer spins through
        # _parse_half_integer on --spin-ext; mirror that here so the JSON
        # writer (which serializes spin as int or .5 float, per blocks_3d) sees
        # the same input. The spin of external operator i is s_i = h_i - hbar_i.
        for i, (h, hb) in enumerate(zip(h_ext, hbar_ext_provided)):
            diff = h - hb
            twice = 2 * diff
            if twice != int(twice):
                parser.error(
                    f"--h-ext[{i+1}] - --hbar-ext[{i+1}] = {mp.nstr(diff, 8)} "
                    "is not an integer or half-integer; external spins must "
                    "be integer or half-integer"
                )
    spinning = (
        hbar_ext_provided is not None
        and any(h != hb for h, hb in zip(h_ext, hbar_ext_provided))
    )
    parity_ext = normalize_external_parity(args.parity_ext)
    projected = args.parity_project is not None
    parity_exchanged = args.parity_exchanged if args.parity_exchanged is not None else "1"
    chiral_coords_requested = [co for co in args.coordinates if co in _CHIRAL_COORDINATES]
    twodim_coords_requested = [co for co in args.coordinates if co not in _CHIRAL_COORDINATES]
    if args.basis == "auto":
        if chiral_coords_requested and twodim_coords_requested:
            parser.error(
                f"chiral coordinates {chiral_coords_requested} and 2d "
                f"coordinates {twodim_coords_requested} are mutually exclusive; "
                "select one family per invocation"
            )
        if chiral_coords_requested:
            basis = "chiral_F"
        elif projected:
            basis = "parity_projected_B"
        elif spinning:
            basis = "ordered_B"
        else:
            basis = "scalar_G"
    else:
        basis = args.basis
    chiral = basis in {"chiral_F", "global_chiral_F"}
    is_global = basis in _GLOBAL_BASES
    is_global_2d = basis in _GLOBAL_2D_BASES
    is_global_chiral = basis in _GLOBAL_CHIRAL_BASES

    if chiral:
        _refuse_chiral_2d_flags(parser, args, raw_args, basis)
    _validate_centers(parser, args, chiral=chiral)

    # scalar_G is z<->zbar symmetric (one chiral expansion driven by --z-shift),
    # so a --zbar-shift that differs from --z-shift cannot be applied. Reject it
    # loudly rather than silently ignoring it.
    if basis in {"scalar_G", "global_scalar_G"}:
        _z_eff = args.z_shift if args.z_shift is not None else mp.mpf(0)
        if args.zbar_shift is not None and args.zbar_shift != _z_eff:
            parser.error(
                f"--basis {basis} is z<->zbar symmetric and built from a single "
                "chiral expansion, so it carries one shift (from --z-shift / "
                "--u-shift); a --zbar-shift that differs from --z-shift cannot "
                "be applied. Use --u-shift for symmetric conditioning; "
                "independent z/zbar shifts need ordered_B or parity_projected_B."
            )

    if is_global_2d:
        # global_scalar_G / global_parity_projected_B fold like their Virasoro
        # counterparts (the interpolation core applies the same Delta-independent
        # `_fold_parity` recombination); global_ordered_B has no parity partner,
        # so it keeps the non-folded coordinates only.
        if basis == "global_ordered_B":
            allowed_2d = _NON_FOLDED_COORDINATES
        else:
            allowed_2d = (
                _NON_FOLDED_COORDINATES | _FOLDED_COORDINATES | _RADIAL_COORDINATES
            )
        bad = [co for co in args.coordinates if co not in allowed_2d]
        if bad:
            parser.error(
                f"--basis {basis} supports the coordinates "
                f"{sorted(allowed_2d)}; got {bad}"
            )
        folded_req = [
            co for co in args.coordinates
            if co in _FOLDED_COORDINATES or co in _RADIAL_COORDINATES
        ]
        if folded_req and basis == "global_parity_projected_B" and projected:
            phase = parity_reversal_phase(
                args.parity_ext, exchanged_parity=(args.parity_exchanged or "1")
            )
            if abs(mp.im(phase)) > tolerance(guard=5):
                parser.error(
                    "parity-folded coordinates require a real (+/-1) total "
                    "parity phase; the given --parity-ext / --parity-exchanged "
                    f"multiply to a non-real phase, so the folded {folded_req} "
                    "have no definite z<->zbar parity -- use non-folded coordinates"
                )
    if basis == "global_scalar_G" and spinning:
        parser.error(
            "--basis global_scalar_G is a scalar-external basis; spinning "
            "externals (--spin-ext / unequal --h-ext, --hbar-ext) are not "
            "supported"
        )
    if basis == "global_ordered_B":
        if projected:
            parser.error(
                "--parity-project is not compatible with --basis global_ordered_B "
                "(use --basis global_parity_projected_B instead)"
            )
    if basis == "global_parity_projected_B":
        if not spinning:
            parser.error(
                "--basis global_parity_projected_B requires spinning externals "
                "(--spin-ext or unequal --h-ext / --hbar-ext)"
            )
        if not projected:
            parser.error(
                "--basis global_parity_projected_B requires --parity-project "
                "+ or -"
            )
    if is_global_2d and basis != "global_scalar_G" and args.parity_exchanged is not None and not projected:
        parser.error("--parity-exchanged is only valid together with --parity-project")

    if chiral:
        bad = [co for co in args.coordinates if co not in _CHIRAL_COORDINATES]
        if bad:
            parser.error(
                f"--basis {basis} requires chiral coordinates "
                f"{sorted(_CHIRAL_COORDINATES)}; got {bad}"
            )
        if using_delta_spin:
            parser.error(
                f"--basis {basis} takes external chiral weights via --h-ext, "
                "not --delta-ext / --spin-ext (the chiral block has no "
                "antiholomorphic half, so the (Delta, spin) split is "
                "meaningless)"
            )
        if hbar_ext_provided is not None:
            parser.error(
                f"--hbar-ext is not valid with --basis {basis} (the chiral "
                "block has only one chiral half)"
            )
        if args.h_ext is None:
            parser.error(f"--basis {basis} requires --h-ext")
        if projected:
            parser.error(f"--parity-project is not compatible with --basis {basis}")
        if args.parity_ext is not None or args.parity_exchanged is not None:
            parser.error(
                "--parity-ext / --parity-exchanged are not meaningful for "
                f"--basis {basis} (parity is a 2d concept)"
            )
        # chiral_F regularization: the chiral F(h, z) table is a DeltaDerivTable
        # (shape (m_max + 1) x 1, "Delta" = h), so the same regularize_delta_table
        # / regularize_delta_skeleton machinery serves it -- the removable /
        # collision strips key on provenance and already map the "chiral" side
        # tag to h_externals. So no chiral-specific gate is needed; chiral_F +
        # regularize flows through the standard Virasoro table backend.
        hbar_ext = None
    else:
        if chiral_coords_requested:
            parser.error(
                f"chiral coordinates {sorted(_CHIRAL_COORDINATES)} require "
                f"--basis chiral_F or --basis global_chiral_F; got coordinates {chiral_coords_requested} "
                f"with basis {basis!r}"
            )
        if using_chiral_halves and args.hbar_ext is None:
            parser.error(
                "on the 2d bases, --h-ext requires --hbar-ext (the chiral "
                "halves parametrization needs both; for chiral-only output, "
                "pass --basis chiral_F)"
            )
        hbar_ext = hbar_ext_provided

    if projected and not spinning:
        parser.error(
            "--parity-project is only valid on the spinning-external path; "
            "scalar externals already use the scalar parity eigenstate G"
        )
    # Resolve --sample-via. The default ("auto") picks 'numeric' wherever the
    # well-conditioned numeric sampler applies and 'symbolic' otherwise; an
    # *explicit* 'numeric' in an unsupported configuration is an error (rather
    # than a silent fallback), so the user gets told why it cannot be honored.
    allowed_sample_coords = (
        _CHIRAL_COORDINATES if chiral else _NUMERIC_SAMPLE_VIA_COORDINATES
    )
    bad_sample_coords = [co for co in args.coordinates if co not in allowed_sample_coords]
    numeric_unsupported_reason = None
    if args.method != "interpolation":
        numeric_unsupported_reason = (
            "--sample-via numeric requires an integer --npoles (not 'all')"
        )
    elif not is_global and args.order == 0:
        # q-order 0 has no Kac poles; the exact (symbolic) table is trivial,
        # while the numeric sampler's arb matrices have no rows.
        numeric_unsupported_reason = (
            "--sample-via numeric needs --order >= 1 (--order 0 has no Kac "
            "poles; the symbolic sampler handles it)"
        )
    elif is_global:
        numeric_unsupported_reason = (
            "--sample-via numeric is not applicable to explicit global_* "
            "bases; they already interpolate directly from the numeric 2F1 block"
        )
    elif basis not in {"scalar_G", "chiral_F", "ordered_B", "parity_projected_B"}:
        numeric_unsupported_reason = (
            "--sample-via numeric is currently supported only for the "
            "Virasoro scalar_G, chiral_F, ordered_B and parity_projected_B bases"
        )
    elif bad_sample_coords:
        numeric_unsupported_reason = (
            "--sample-via numeric currently supports only the coordinates "
            f"{sorted(allowed_sample_coords)}; got {bad_sample_coords}"
        )
    # Recursion-singular c is now supported under --sample-via numeric: the
    # regularized numeric sampler (epsilon-extrapolation of the numeric grid in
    # c) supplies node values, so the symbolic numerators are not used.
    if args.sample_via == "auto":
        args.sample_via = "symbolic" if numeric_unsupported_reason else "numeric"
    elif args.sample_via == "numeric" and numeric_unsupported_reason:
        parser.error(numeric_unsupported_reason)
    if projected and parity_ext is None:
        parser.error("--parity-project requires --parity-ext")
    if basis in ("scalar_G", "global_scalar_G") and args.parity_exchanged is not None:
        from .requests import scalar_G_exchanged_parity_message

        parser.error(scalar_G_exchanged_parity_message(args.parity_exchanged, "--parity-exchanged")
                     if args.parity_exchanged != "1" else
                     f"--parity-exchanged does not apply to --basis {basis} (z<->zbar even at every spin); "
                     "it is only valid together with --parity-project")
    if not projected and args.parity_exchanged is not None:
        parser.error("--parity-exchanged is only valid together with --parity-project")
    if basis == "parity_projected_B" and not projected:
        parser.error("--basis parity_projected_B needs --parity-project + or - (and --parity-ext)")
    if basis == "ordered_B" and any(co not in _NON_FOLDED_COORDINATES for co in args.coordinates):
        parser.error("ordered_B has no parity fold; use non-folded coordinates zzb, yyb, or qqb")
    if basis == "scalar_G" and spinning:
        parser.error(
            "--basis scalar_G requires scalar externals (h_i == hbar_i); use "
            "ordered_B or parity_projected_B for spinning externals"
        )

    if spinning:
        folded = [co for co in args.coordinates if co not in _NON_FOLDED_COORDINATES]
        if folded and not projected:
            parser.error(
                "spinning externals (--spin-ext) admit only the non-folded "
                f"coordinates {sorted(_NON_FOLDED_COORDINATES)} unless "
                "--parity-project is used; the parity-folded "
                f"{folded} need a parity-definite basis"
            )
        if folded and projected:
            phase = parity_reversal_phase(parity_ext, exchanged_parity=parity_exchanged)
            if abs(mp.im(phase)) > tolerance(guard=5):
                parser.error(
                    "parity-folded coordinates require a real (+/-1) total "
                    "parity phase; the given --parity-ext / --parity-exchanged "
                    f"multiply to a non-real phase, so the folded {folded} have "
                    "no definite z<->zbar parity -- use non-folded coordinates"
                )

    if chiral:
        # Chiral output has no exchanged spin: a single file per (c, externals).
        # `_parse_spins` is bypassed and we run one iteration with ell = None.
        spins = [None]
    else:
        try:
            spins = _parse_spins(args.spin_ranges)
        except ValueError as exc:
            parser.error(str(exc))
        _check_output_template(parser, args, len(spins))
        if basis in ("scalar_G", "global_scalar_G") and any(s != int(s) for s in spins):
            # G_{Delta, ell} = B(h, hbar) + B(hbar, h) is the block of an integer-spin exchange (as evaluate enforces)
            parser.error(f"--basis {basis} needs integer exchanged spins; use ordered_B or "
                         "parity_projected_B for half-integer spins")

    # The canonical table request is the hand-off point between the CLI and
    # the family adapter. Family/surface/basis/coordinate and numerical-policy
    # validation therefore happens through the same registry used by the torus
    # branch and the supported Python API.
    from .requests import BlockRequest, NumericalPolicy, RequestValidationError
    from .table_adapters import table_adapter_for
    from .api import build_table_request

    table_family = "global" if is_global else "virasoro"
    try:
        table_request = BlockRequest(
            operation="table",
            surface="sphere",
            family=table_family,
            basis=basis,
            coordinate=args.coordinates[0],
            coordinates=tuple(args.coordinates),
            h_ext=tuple(h_ext),
            hbar_ext=None if hbar_ext is None else tuple(hbar_ext),
            central_charge=c,
            sector="even",
            branch="++",
            parity=args.parity_project,
            parity_external=None if parity_ext is None else tuple(parity_ext),
            parity_exchanged=parity_exchanged,
            chiral_norm=args.chiral_norm,
            z_shift=args.z_shift,
            zbar_shift=args.zbar_shift,
            center=args.z_center,
            center_bar=args.zbar_center,
            gap=args.gap,
            policy=NumericalPolicy(
                precision=args.precision,
                order=args.order,
                lambda_order=args.lam,
                npoles=args.npoles,
                pole_selection=(args.pole_selection if args.pole_selection_given or not is_global else None),
                c_regularization=args.regularize,
                target_digits=args.target_digits,
                max_precision=args.max_precision,
            ),
        )
        table_adapter = table_adapter_for(table_family)
        table_adapter.validate(table_request)
    except RequestValidationError as exc:
        parser.error(str(exc))

    os.makedirs(args.output, exist_ok=True)
    # Record the full invocation in the JSON output, mirroring blocks_3d
    # (which writes `build/blocks_3d -o ...`).
    metadata_args = getattr(args, "_cli_invocation_args", raw_args)
    cli_args_str = shlex.join(["virasoro_blocks", *metadata_args])

    # Pre-compute the per-coordinate chiral expansions once and reuse them
    # across every spin. ChiralSymbolic does not depend on ell, so a single
    # cache amortizes the q-recursion across the whole sweep. Skip when
    # regularizing: those probes evaluate at c +/- t^2, so a cache built at
    # the user's c is invalid (the regularize path falls back to per-probe
    # chiral builds inside the Virasoro table backend).
    chiral_cache = (
        None if is_global else _build_cli_chiral_cache(
            args,
            h_ext,
            hbar_ext,
            b,
            regularize=regularize_effective,
            basis=basis,
        )
    )
    from .table_backends import VirasoroBackendOptions

    virasoro_backend_options = VirasoroBackendOptions(
        method=args.method,
        sample_via=args.sample_via,
        shared_sample_points=args.shared_sample_points,
        regularize=regularize_effective,
        chiral_cache=chiral_cache,
    )

    # --npoles at or above the Delta-pole count keeps every pole: the full table, not an order reduction. The count
    # is the same across spins (2 x the chiral Kac weights for the 2d bases, the chiral count for chiral_F), read off
    # the chiral cache (not built in the regularize path).
    if (
        args.method == "interpolation"
        and chiral_cache is not None
        and chiral_cache
        and args.npoles is not None
    ):
        cs_h, _ = chiral_cache[args.coordinates[0]]
        chiral_pole_count = len(cs_h.denominator_roots)
        actual_pole_count = chiral_pole_count if chiral else 2 * chiral_pole_count
        if args.npoles >= actual_pole_count:
            print(
                f"note: --npoles {args.npoles} keeps every Delta-pole at --order {args.order} "
                f"({actual_pole_count}): building the full table, as with --npoles omitted. To get a "
                "higher-degree interpolation polynomial, increase --order.",
                flush=True,
            )
            args.npoles = None
            table_request = replace(table_request, policy=replace(table_request.policy, npoles=None))

    def _emit_spin_result(ell, serialized):
        # coefficients are written at the precision the table was built and verified at
        with mp.workdps(max(mp.mp.dps, getattr(serialized, "working_dps", None) or mp.mp.dps)):
            return _emit_spin_result_at_precision(ell, serialized)

    def _emit_spin_result_at_precision(ell, serialized):
        if chiral:
            print("  chiral F(h, z) ...", flush=True)
        else:
            print(f"  spin {_json_spin_value(ell)} ...", flush=True)
        # the requested kept-pole count (as the bootstrap manifest records it); the pole list gives the kept one
        npoles_meta = len(serialized.pole_list_x) if args.npoles is None else int(args.npoles)

        first_coord = serialized.coordinates[0]
        m_max = len(serialized.numerators[first_coord]) - 1
        n_max = len(serialized.numerators[first_coord][0]) - 1

        spin_value = None if chiral else _json_spin_value(ell)
        # The {spin} token is replaced with the spin for 2d outputs and with
        # the literal "chiral" tag for the chiral_F basis (single file).
        spin_tag = "chiral" if chiral else str(spin_value)

        if args.format in ("json", "both"):
            fname = args.output_template.replace("{spin}", spin_tag)
            path = _output_path(args, fname)
            serialized.q_truncation_digits = getattr(args, "_q_truncation_digits", None)
            text = write_json(
                serialized,
                c=c, h_ext=h_ext,
                hbar_ext=None if chiral else hbar_ext,
                spin=0 if chiral else spin_value,
                q_order=args.order,
                npoles=npoles_meta,
                pole_selection=args.pole_selection if args.npoles is not None else None,
                basis=basis,
                parity_projection=None if chiral else args.parity_project,
                parity_external=None if chiral else parity_ext,
                parity_exchanged=(
                    None if chiral else (parity_exchanged if projected else None)
                ),
                chiral_norm=args.chiral_norm,
                precision_input=args.precision,
                output_template=args.output_template,
                args=cli_args_str,
                z_center=args.z_center,
                zbar_center=None if chiral else args.zbar_center,
                z_shift=args.z_shift,
                zbar_shift=None if chiral else args.zbar_shift,
                # (at a recognized singular c the table is the block AT c, from the merged poles of the Laurent
                # recursion)
                regularization_policy=("merged" if regularize_effective else "direct"),
            )
            with open(path, "w") as fh:
                fh.write(text)
            print(f"    wrote {path}")

        if args.format in ("m", "both"):
            for coord in args.coordinates:
                fname = mathematica_filename(
                    coord,
                    c=c, h_ext=h_ext, hbar_ext=hbar_ext,
                    spin=spin_value, q_order=args.order,
                    m_max=m_max, n_max=n_max,
                    npoles=npoles_meta,
                    basis=basis,
                    parity_projection=None if chiral else args.parity_project,
                    chiral_norm=args.chiral_norm,
                    # every other table-changing setting, tagged away from its default (no silent overwrites)
                    pole_selection=(args.pole_selection
                                    if args.npoles is not None or args.pole_selection_given else None),
                    shared_sample_points=args.shared_sample_points,
                    parity_external=None if chiral else parity_ext,
                    parity_exchanged=None if chiral else (parity_exchanged if projected else None),
                    z_shift=args.z_shift,
                    zbar_shift=None if chiral else args.zbar_shift,
                    z_center=args.z_center,
                    zbar_center=None if chiral else args.zbar_center,
                    gap=args.gap,
                )
                path = os.path.join(args.output, fname)
                text = write_mathematica(
                    serialized,
                    coordinate=coord,
                    ell=0 if chiral else spin_value,
                    basis=basis,
                    parity_projection=None if chiral else args.parity_project,
                    parity_external=None if chiral else parity_ext,
                    parity_exchanged=(
                        None if chiral else (parity_exchanged if projected else None)
                    ),
                    chiral_norm=args.chiral_norm,
                    z_center=args.z_center,
                    zbar_center=None if chiral else args.zbar_center,
                    z_shift=args.z_shift,
                    zbar_shift=None if chiral else args.zbar_shift,
                    c=c,
                    h_ext=h_ext,
                    hbar_ext=None if chiral else hbar_ext,
                    q_order=args.order,
                    npoles=npoles_meta,
                    pole_selection=args.pole_selection if args.npoles is not None else None,
                    precision_input=args.precision,
                    output_template=args.output_template,
                    args=cli_args_str,
                    regularization_policy=("merged" if regularize_effective else "direct"),
                )
                with open(path, "w") as fh:
                    fh.write(text)
                print(f"    wrote {path}")

    if not is_global:
        # q-order convergence gate (once per external set; cached across the
        # spin family). Global bases have no q-series and are skipped.
        _run_q_gate(
            parser, args, c=c, h_ext=h_ext,
            hbar_ext=None if chiral else hbar_ext, lam=args.lam,
            label=f"basis {basis}",
        )

    def _run_serial_spins(build_num_threads):
        for ell in spins:
            if table_family == "global":
                serialized = build_table_request(
                    table_request,
                    spin=ell,
                    num_threads=build_num_threads,
                    shared_sample_points=args.shared_sample_points,
                )
            else:
                serialized = build_table_request(
                    table_request,
                    spin=ell, backend_options=virasoro_backend_options,
                )
            _emit_spin_result(ell, serialized)
        return

    if args.num_threads == 1 or len(spins) == 1:
        # Serial over spins. With a single spin the global bases can still use
        # --num-threads for within-build (per-node-bucket) parallelism; the
        # multi-spin path below parallelizes across spins instead (and keeps
        # build_num_threads=1 to avoid nesting process pools).
        return _run_serial_spins(args.num_threads)

    # The worker entry points live in an importable package module so both the
    # installed console script and ``python -m virasoro_blocks`` can use the
    # platform's normal process start method (including spawn on macOS).
    from .cli.workers import init_spin_worker, process_spin_worker

    worker_state = {
        "args": args,
        "h_ext": h_ext,
        "hbar_ext": hbar_ext,
        "b": b,
        "regularize": regularize_effective,
        "table_request": table_request,
    }
    worker_count = min(args.num_threads, len(spins))
    # ``multiprocessing.Pool`` does not impose concurrent.futures' eager
    # POSIX named-semaphore limit check, while retaining the same process
    # parallelism and initializer contract.  This matters on macOS and in
    # restricted developer environments where ``os.sysconf`` is inaccessible
    # even though ordinary multiprocessing works.
    context = multiprocessing.get_context()
    from ._conventions import pool_initializer_kwargs

    with context.Pool(
        processes=worker_count,
        **pool_initializer_kwargs(init_spin_worker, (worker_state,)),
    ) as pool:
        for ell, serialized in pool.imap(process_spin_worker, spins):
            _emit_spin_result(ell, serialized)


if __name__ == "__main__":
    sys.exit(main())
