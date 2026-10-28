"""Implementation of ``virasoro_blocks point-table``.

This operation builds bootstrap point-constraint tables directly: a block
frozen at one exchanged value, for every family -- the Virasoro, global, NS and
Ramond sphere blocks, and on the torus the reduced characters (with the
irreducible character of a degenerate module) and the one-point blocks.  It
shares the numerical point backends and the standard artifact writers with
the ordinary table path.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys

import mpmath as mp

from . import (
    check_fixed_value_counts,
    KAC_LIMIT_HELP, RAMOND_BRANCHES, RamondBranchAction, accept_negative_numbers, chiral_norm as _chiral_norm,
    join_option_values, real_number as _number, half_integer as _half_integer,
)

from ..parity import normalize_external_parity, normalize_parity_phase
from ..requests import (
    MIN_TABLE_PRECISION, BlockPoleError, BlockRequest, NumericalPolicy,
)
from ..table_adapters import table_adapter_for
from ..api import build_table_request
from ..writers import write_json, write_mathematica, write_torus_json


_FAMILIES = ("virasoro", "global", "ns", "ramond", "torus-character", "torus-onepoint")
_TORUS_FAMILIES = ("torus-character", "torus-onepoint")
#: the families whose bases carry their name as a prefix (Virasoro keeps the short names)
_PREFIXED_FAMILIES = ("global", "ns", "ramond")
_SHORT_BASES = ("scalar_G", "ordered_B", "parity_projected_B", "chiral_F")
_BASES = (
    "auto", *_SHORT_BASES,
    *(f"{family}_{short}" for family in _PREFIXED_FAMILIES for short in _SHORT_BASES),
    "torus_G", "torus_onept_G",
)
_COORDINATES = (
    "zzb", "yyb", "qqb", "ws", "xt", "qqb_ws",
    "ws_radial", "xt_radial", "qqb_ws_radial", "z", "y", "q",
)
_CHIRAL_COORDINATES = {"z", "y", "q"}
_NON_FOLDED_COORDINATES = {"zzb", "yyb", "qqb"}
_FOLDED_COORDINATES = {
    "ws", "xt", "qqb_ws", "ws_radial", "xt_radial", "qqb_ws_radial",
}
_CHIRAL_BASES = {"chiral_F", *(f"{family}_chiral_F" for family in _PREFIXED_FAMILIES)}


def _point_regularization_policy(c, regularize) -> str:
    """``merged`` where the point grid is the block AT a singular c (merged poles), else ``direct``."""
    from virasoro_blocks.singularity import recognized_singular_locus

    if c is None or regularize == "off":
        return "direct"
    return "merged" if (regularize == "on" or recognized_singular_locus(c) is not None) else "direct"


def _mp_number(value) -> mp.mpf:
    from ..numeric_input import parse_number

    return parse_number(value)


def _center(token: str):
    if "=" not in token:
        raise argparse.ArgumentTypeError(
            "centers must have the form z=value or zbar=value"
        )
    key, value = token.split("=", 1)
    if key not in {"z", "zbar"}:
        raise argparse.ArgumentTypeError("center names must be z or zbar")
    return key, _number(value)


def _output_path(output_dir: str, fname: str) -> str:
    """Join ``fname`` onto ``-o`` and create any subdirectory it names."""
    path = os.path.join(output_dir, fname)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="virasoro_blocks point-table",
        allow_abbrev=False,   # a prefix like --delta silently meant --delta-ext
        description=(
            "Build a frozen derivative table for one exchanged dimension. "
            "This is the bootstrap point-constraint operation, not pointwise "
            "cross-ratio evaluation. Every family has one: the Virasoro, global, "
            "NS and Ramond sphere blocks, and on the torus the reduced character "
            "(the irreducible one on a Kac weight: the vacuum, a current, a "
            "minimal-model primary) and the one-point block."
        ),
        epilog="Example: virasoro_blocks point-table --surface torus --family torus-character "
               "--central-charge 1/2 --exchange-delta 1/8 --lambda 10 -o out/   (the Ising sigma character)",
    )
    p.add_argument("--surface", choices=["sphere", "torus"], default="sphere",
                   help="sphere four-point blocks (default), or torus for --family torus-character / "
                        "torus-onepoint (coordinate vvb, JSON only)")
    p.add_argument("--family", choices=_FAMILIES, default="virasoro",
                   help="block family (default virasoro); torus-character and torus-onepoint imply --surface torus")
    p.add_argument(
        "--basis", metavar="BASIS", default="auto",
        help="Block representation. Use `virasoro_blocks info` for the "
             "supported bases for each family.",
    )
    p.add_argument("--central-charge", type=_number, default=None, metavar="C",
                   help="central charge (every family but global); fractions like 7/10 are exact")
    p.add_argument(
        "--coordinates", nargs="+", default=["zzb"],
        choices=_COORDINATES, metavar="COORD",
        help=f"derivative coordinates (default zzb; any of {', '.join(_COORDINATES)})",
    )
    p.add_argument("--lambda", dest="lam", type=int, required=True, metavar="N",
                   help="derivative order: entries d^m d^n with m + n <= N")
    p.add_argument("--order", type=int, default=30, metavar="N",
                   help="q-expansion order of the recursion (default 30; the torus character tables are exact and "
                        "take none)")
    p.add_argument(
        "--precision", type=int, default=60, metavar="P",
        help=f"Working precision in decimal digits (at least {MIN_TABLE_PRECISION} on the sphere, 10 on the torus): the "
             "table is built and written at P digits, verified against the same table at more digits "
             "(verified_digits), and a shortfall is reported",
    )
    p.add_argument(
        "--target-digits", type=int, default=None, metavar="D",
        help="Deliver at least D verified digits, raising the working precision from --precision by the measured "
             "deficit (recorded as precision_actual); an error when --max-precision is reached. Off by default",
    )
    p.add_argument("--max-precision", type=int, default=None, metavar="M",
                   help="cap for --target-digits (default max(4 D, 2 P, D + 200)); refused without it")
    p.add_argument("--delta-ext", nargs=4, type=_number, default=None, metavar="D",
                   help="the four external dimensions (default 0)")
    p.add_argument("--spin-ext", nargs=4, type=_half_integer, default=None, metavar="S",
                   help="the four external spins (default 0)")
    p.add_argument("--h-ext", nargs=4, type=_number, default=None, metavar="H",
                   help="the four external holomorphic weights (exclusive with --delta-ext)")
    p.add_argument("--hbar-ext", nargs=4, type=_number, default=None, metavar="HB",
                   help="the four external antiholomorphic weights")
    p.add_argument("--exchange-h", type=_number, default=None, metavar="H", help="exchanged weight (chiral bases)")
    p.add_argument("--exchange-delta", type=_number, default=None, metavar="D",
                   help="exchanged dimension (2d bases and the torus families)")
    p.add_argument("--exchange-spin", type=_half_integer, default=None, metavar="L",
                   help="exchanged spin (default 0; half-integers like 5/2 and negative spins as in `table`; on the "
                        "torus an integer j >= 0, the module (h, hbar) = ((D + j)/2, (D - j)/2) paired with its "
                        "mirror)")
    p.add_argument("--torus-onept-ext", type=_number, default=None, metavar="D",
                   help="--family torus-onepoint: the external scalar's chiral weight (Delta_Phi = 2 D), as in `table`")
    p.add_argument("--super-sector", choices=["even", "odd"], default="even",
                   help="NS/Ramond exchanged component (default even; --family ns / ramond only)")
    p.add_argument("--ramond-branch", choices=list(RAMOND_BRANCHES), default="++", action=RamondBranchAction,
                   help="Ramond external branch (default ++; --family ramond only). All four work as a separate "
                        "value, e.g. `--ramond-branch --`, or as `--ramond-branch=--`")
    p.add_argument(
        "--parity-ext", nargs=4, type=normalize_parity_phase, default=None, metavar="P",
        help="external parity phases, e.g. 1 1 1 -1 (with --parity-project)",
    )
    p.add_argument("--parity-project", choices=["+", "-"], default=None,
                   help="parity projection sign; selects parity_projected_B under --basis auto")
    p.add_argument(
        "--parity-exchanged", type=normalize_parity_phase, default=None, metavar="P",
        help="exchanged-operator parity phase (default 1)",
    )
    p.add_argument("--chiral-norm", type=_chiral_norm, default=None, metavar="N",
                   help="chiral normalization N, a positive integer: the block is N^{-h} times the standard one "
                        "(F = N^{-h} (16 q)^h ...; default 16)")
    p.add_argument("--z-shift", type=_number, default=None, metavar="S", help="z-exponent shift, as in `table`")
    p.add_argument("--zbar-shift", type=_number, default=None, metavar="S", help="zbar-exponent shift, as in `table`")
    p.add_argument("--u-shift", type=_number, default=None, metavar="S",
                   help="set both shifts (either one given explicitly wins)")
    p.add_argument(
        "--center", action="append", type=_center, default=[], metavar="z=VALUE",
        help="expansion center, z=.. and optionally zbar=.. (default the symmetric point 1/2)",
    )
    p.add_argument(
        "--regularize", nargs="?", const="on", default="auto",
        choices=["auto", "on", "off"],
        help="c-regularization at singular central charges (default auto; bare flag = on; Virasoro, NS and "
             "Ramond)",
    )
    p.add_argument("--kac-limit", choices=["irreducible", "c_limit"], default="irreducible", help=KAC_LIMIT_HELP)
    p.add_argument(
        "--q-gate", dest="q_gate", choices=["off", "warn", "strict"], default="warn",
        help="q-order convergence gate, as in `table`: probe the q-expansion tail at the exchanged weight(s) "
             "(Virasoro; NS/Ramond at the table probes) and --order, record the estimate as q_truncation_digits; "
             "'warn' (default) warns below 20 digits, 'strict' errors, 'off' skips. A torus one-point table "
             "measures its truncation against a deeper reference and is gated the same way. The global blocks and "
             "the torus characters have no q-series: only 'off' is accepted there.",
    )
    p.add_argument("--format", choices=["m", "json", "both"], default="json",
                   help="output format: json (default), m (Mathematica), or both; the torus families write JSON only")
    p.add_argument("-o", "--output", default=".", metavar="DIR", help="output directory (default .)")
    p.add_argument("--output-template", default="derivs_point_{spin}.json",
                   help="output file name; {spin} is replaced (default derivs_point_{spin}.json)")
    return accept_negative_numbers(p)


def _family_of_basis(basis: str):
    """The family a prefixed basis names (``global_scalar_G`` -> ``global``), or None for a short name."""
    return next((family for family in _PREFIXED_FAMILIES if basis.startswith(f"{family}_")), None)


def _family_basis(family: str, basis: str, *, chiral: bool, projected: bool, spinning_external: bool) -> str:
    """Resolve ``--basis`` for ``family``: ``auto`` picks chiral_F for chiral coordinates, parity_projected_B with
    --parity-project, ordered_B for spinning externals and scalar_G otherwise; the global, NS and Ramond bases carry
    their family prefix, the Virasoro ones the short names."""
    named = _family_of_basis(basis)
    if named is not None and named != family:
        raise ValueError(f"{named}_* bases require --family {named}")
    if named is not None:
        return basis
    if basis == "auto":
        if chiral:
            basis = "chiral_F"
        elif projected:
            basis = "parity_projected_B"
        elif spinning_external:
            basis = "ordered_B"
        else:
            basis = "scalar_G"
    return basis if family == "virasoro" else f"{family}_{basis}"


def _resolve_external_weights(args, *, chiral: bool):
    using_chiral_halves = args.h_ext is not None or args.hbar_ext is not None
    using_delta_spin = args.delta_ext is not None or args.spin_ext is not None
    if using_chiral_halves and using_delta_spin:
        raise ValueError(
            "--h-ext/--hbar-ext and --delta-ext/--spin-ext are mutually exclusive"
        )

    if chiral:
        if args.h_ext is None or args.hbar_ext is not None or using_delta_spin:
            raise ValueError("chiral point-table needs --h-ext only")
        return [_mp_number(h) for h in args.h_ext], None

    if using_chiral_halves:
        if args.h_ext is None or args.hbar_ext is None:
            raise ValueError("full h-parametrized point-table needs --hbar-ext")
        h_ext = [_mp_number(h) for h in args.h_ext]
        hbar_ext = [_mp_number(h) for h in args.hbar_ext]
    else:
        delta_ext = [
            _mp_number(d) for d in (args.delta_ext or ["0", "0", "0", "0"])
        ]
        spin_ext = [
            _mp_number(s) for s in (args.spin_ext or ["0", "0", "0", "0"])
        ]
        h_ext = [(d + s) / 2 for d, s in zip(delta_ext, spin_ext)]
        hbar_ext = [(d - s) / 2 for d, s in zip(delta_ext, spin_ext)]
    return h_ext, hbar_ext


def _resolve_request(args):
    requested_chiral = any(co in _CHIRAL_COORDINATES for co in args.coordinates)
    requested_full = any(co not in _CHIRAL_COORDINATES for co in args.coordinates)
    if requested_chiral and requested_full:
        raise ValueError(
            "chiral coordinates and 2d coordinates are mutually exclusive"
        )

    spinning_external = False
    if args.spin_ext is not None:
        spinning_external = any(_mp_number(s) != 0 for s in args.spin_ext)
    if args.h_ext is not None and args.hbar_ext is not None:
        spinning_external = any(
            _mp_number(h) != _mp_number(hb)
            for h, hb in zip(args.h_ext, args.hbar_ext)
        )
    chiral_requested = (
        requested_chiral
        or args.basis in _CHIRAL_BASES
    )
    basis = _family_basis(
        args.family, args.basis, chiral=chiral_requested, projected=args.parity_project is not None,
        spinning_external=spinning_external,
    )

    chiral = basis in _CHIRAL_BASES
    if chiral != requested_chiral:
        raise ValueError(
            f"basis {basis!r} requires "
            f"{'chiral' if chiral else '2d'} coordinates"
        )
    if chiral and args.parity_project is not None:
        raise ValueError("--parity-project is not compatible with a chiral basis")
    if chiral and (args.parity_ext is not None or args.parity_exchanged is not None):
        raise ValueError("parity metadata is not valid for a chiral basis")

    projected = args.parity_project is not None
    if projected and not spinning_external:
        raise ValueError(
            "--parity-project is only valid for spinning external operators"
        )
    if projected and args.parity_ext is None:
        raise ValueError("--parity-project requires --parity-ext")
    if not projected and args.parity_exchanged is not None:
        raise ValueError(
            "--parity-exchanged is only valid together with --parity-project"
        )
    if basis.endswith("ordered_B") and any(co not in _NON_FOLDED_COORDINATES for co in args.coordinates):
        raise ValueError(f"--basis {basis} requires non-folded coordinates zzb, yyb, or qqb")
    if basis.endswith("parity_projected_B") and not projected:
        raise ValueError(f"--basis {basis} requires --parity-project +/-")
    if basis.endswith("scalar_G") and spinning_external:
        raise ValueError(f"--basis {basis} requires scalar externals")
    if spinning_external and not projected:
        folded = [co for co in args.coordinates if co not in _NON_FOLDED_COORDINATES]
        if folded:
            raise ValueError(
                "spinning externals admit only non-folded coordinates unless "
                f"--parity-project is used; got {folded}"
            )

    h_ext, hbar_ext = _resolve_external_weights(args, chiral=chiral)
    parity_ext = normalize_external_parity(args.parity_ext)
    parity_exchanged = args.parity_exchanged or "1"

    if chiral:
        if args.exchange_h is None:
            raise ValueError("chiral point-table needs --exchange-h")
        if args.exchange_delta is not None or args.exchange_spin is not None:
            raise ValueError("chiral point-table uses --exchange-h, not Delta/spin")
        ell = 0
        point_value = _mp_number(args.exchange_h)
    else:
        if args.exchange_h is not None:
            raise ValueError("2d point-table uses --exchange-delta, not --exchange-h")
        if args.exchange_delta is None:
            raise ValueError("full point-table needs --exchange-delta")
        ell = 0 if args.exchange_spin is None else _mp_number(args.exchange_spin)
        ell = int(ell) if ell == int(ell) else ell
        point_value = _mp_number(args.exchange_delta)

    centers = dict(args.center)
    if len(centers) != len(args.center):
        raise ValueError("duplicate --center names are not allowed")
    if "zbar" in centers and "z" not in centers:
        raise ValueError("zbar center requires a z center")
    center = None if "z" not in centers else _mp_number(centers["z"])
    center_bar = None if "zbar" not in centers else _mp_number(centers["zbar"])
    if chiral and center_bar is not None:
        raise ValueError("zbar center is not valid for a chiral basis")
    if any(co in _FOLDED_COORDINATES for co in args.coordinates) and (
        center is not None or center_bar is not None
    ):
        raise ValueError("centers are not supported for folded/radial coordinates")

    z_shift = None if args.z_shift is None else _mp_number(args.z_shift)
    zbar_shift = None if args.zbar_shift is None else _mp_number(args.zbar_shift)
    if args.u_shift is not None:
        u_shift = _mp_number(args.u_shift)
        if z_shift is None:
            z_shift = u_shift
        if zbar_shift is None:
            zbar_shift = u_shift
    if basis.endswith("scalar_G"):
        effective_z_shift = mp.mpf(0) if z_shift is None else z_shift
        if zbar_shift is not None and zbar_shift != effective_z_shift:
            raise ValueError(
                f"--basis {basis} is symmetric and cannot use a distinct "
                "--zbar-shift; use --u-shift for a common shift"
            )

    if args.family != "global":
        if args.central_charge is None:
            raise ValueError(f"--central-charge is required for {args.family} point tables")
        central_charge = _mp_number(args.central_charge)
        q_order = args.order
    else:
        central_charge = None
        q_order = None
    if args.family in ("ns", "ramond") and not chiral and ell != int(ell):
        raise ValueError("superconformal full blocks require an integer exchanged spin")

    request = BlockRequest(
        operation="point-table", surface=args.surface, family=args.family,
        basis=basis, coordinate=args.coordinates[0],
        coordinates=tuple(args.coordinates), h_ext=tuple(h_ext),
        hbar_ext=None if chiral else tuple(hbar_ext),
        central_charge=central_charge,
        exchange_h=point_value if chiral else None,
        exchange_delta=None if chiral else point_value,
        exchange_spin=None if chiral else ell,
        parity=args.parity_project,
        parity_external=(
            None if chiral or parity_ext is None else tuple(parity_ext)
        ),
        parity_exchanged=None if chiral else parity_exchanged,
        chiral_norm=(
            None if args.chiral_norm is None else _mp_number(args.chiral_norm)
        ),
        z_shift=z_shift, zbar_shift=None if chiral else zbar_shift,
        center=center, center_bar=None if chiral else center_bar,
        **({"sector": args.super_sector, "branch": args.ramond_branch} if args.family in ("ns", "ramond") else {}),
        policy=NumericalPolicy(
            precision=args.precision, order=q_order, lambda_order=args.lam,
            c_regularization=args.regularize, target_digits=args.target_digits, max_precision=args.max_precision,
        ),
    )
    adapter = table_adapter_for(args.family)
    adapter.validate(request)
    return {
        "basis": basis, "chiral": chiral, "projected": projected,
        "ell": ell, "point_value": point_value, "h_ext": h_ext,
        "hbar_ext": hbar_ext, "parity_ext": parity_ext,
        "parity_exchanged": parity_exchanged, "central_charge": central_charge,
        "q_order": q_order, "center": center, "center_bar": center_bar,
        "z_shift": z_shift, "zbar_shift": zbar_shift,
        "request": request,
    }


def _point_q_gate(parser, args, raw_argv, resolved):
    """The q-order gate for a Virasoro point table, at its own exchanged weight(s); returns the digits recorded as
    ``q_truncation_digits`` (None when not measured).  The table path records the same estimate."""
    given = any(tok == "--q-gate" or tok.startswith("--q-gate=") for tok in raw_argv)
    if args.family == "global":
        if given and args.q_gate != "off":
            parser.error(f"--q-gate {args.q_gate} has nothing to check for the global blocks (the exact 2F1 "
                         "block has no q-series truncation)")
        return None
    if args.q_gate == "off":
        return None
    if args.family in ("ns", "ramond"):
        return _super_point_q_gate(parser, args, resolved)
    from ..qgate import QConvergenceError, gate_q_convergence

    if resolved["chiral"]:
        probes = [resolved["point_value"]]
    else:
        delta, ell = resolved["point_value"], mp.mpf(resolved["ell"])
        probes = sorted({(delta + ell) / 2, (delta - ell) / 2})
    center, center_bar = resolved["center"], resolved["center_bar"]
    if center is None:
        centers = None
    elif center_bar is None or center_bar == center:
        centers = [center]
    else:
        centers = [center, center_bar]
    ext_sets = [tuple(resolved["h_ext"])]
    if resolved["hbar_ext"] is not None and tuple(resolved["hbar_ext"]) != ext_sets[0]:
        ext_sets.append(tuple(resolved["hbar_ext"]))
    chiral_norm = None if args.chiral_norm is None else _mp_number(args.chiral_norm)
    worst = None
    try:
        for ext in ext_sets:
            report = gate_q_convergence(
                c=resolved["central_charge"], h_externals=list(ext), q_order=args.order, n_terms=int(args.lam) + 2,
                coordinate=args.coordinates[0], chiral_norm=chiral_norm, z_shift=resolved["z_shift"],
                h_probes=probes, center=centers, mode=args.q_gate, label="point table",
            )
            digits = None if report is None else report.get("digits")
            if digits is not None and (worst is None or digits < worst):
                worst = digits
    except QConvergenceError as exc:
        parser.error(str(exc))
    return None if worst is None else min(float(worst), float(args.precision))


def _super_point_q_gate(parser, args, resolved):
    """The NS/Ramond q-order gate of the point table, the probe the NS/Ramond tables record
    (:func:`virasoro_blocks.qgate.super_q_tail_report` at its default weights)."""
    from ..qgate import QConvergenceError, act_on_q_report, super_q_tail_report

    request = resolved["request"]
    worst = None
    ext_sets = [tuple(resolved["h_ext"])]
    if resolved["hbar_ext"] is not None and tuple(resolved["hbar_ext"]) != ext_sets[0]:
        ext_sets.append(tuple(resolved["hbar_ext"]))
    for ext in ext_sets:
        report = super_q_tail_report(family=args.family, c=resolved["central_charge"], h_externals=list(ext),
                                     q_order=args.order, n_terms=int(args.lam) + 2, coordinate=args.coordinates[0],
                                     sector=request.sector, branch=request.branch, center=resolved["center"],
                                     chiral_norm=request.chiral_norm, z_shift=resolved["z_shift"])
        try:
            act_on_q_report(report, mode=args.q_gate, label=f"{args.family} point table")
        except QConvergenceError as exc:
            parser.error(str(exc))
        if report.get("digits") is not None and (worst is None or report["digits"] < worst):
            worst = report["digits"]
    return None if worst is None else min(float(worst), float(args.precision))


def _given(raw_argv, flag) -> bool:
    """Whether ``flag`` was passed explicitly (``--flag V`` or ``--flag=V``)."""
    return any(tok == flag or tok.startswith(flag + "=") for tok in raw_argv)


def _torus_main(parser, args, raw_argv) -> int:
    """``point-table --surface torus``: the reduced character (torus-character) or one-point block (torus-onepoint)
    frozen at (exchange_delta, exchange_spin), written as a ``vvb`` torus JSON table with ``variable: "point"``.

    A character on a Kac weight -- the vacuum and the currents at every c, every Kac weight of a recognized singular
    c -- is the irreducible module's (recorded as ``regularization_policy: "irreducible"`` and ``kac_limit``); a
    one-point block there is the irreducible-module trace, as ``evaluate`` gives it."""
    family = args.family
    if family not in _TORUS_FAMILIES:
        parser.error("--surface torus requires --family torus-character or --family torus-onepoint")
    onept = family == "torus-onepoint"
    if args.central_charge is None:
        parser.error("torus point tables require --central-charge")
    if args.exchange_delta is None:
        parser.error("torus point tables require --exchange-delta (and --exchange-spin, default 0)")
    if onept and args.torus_onept_ext is None:
        parser.error("--family torus-onepoint requires --torus-onept-ext")
    if not onept and args.torus_onept_ext is not None:
        parser.error("--torus-onept-ext is for --family torus-onepoint")
    if args.format != "json":
        parser.error("torus point tables are written as JSON only; use --format json")
    spin = 0 if args.exchange_spin is None else _mp_number(args.exchange_spin)
    if spin != int(spin) or spin < 0:
        parser.error(f"torus point tables take an integer --exchange-spin j >= 0 (the module (h, hbar) and its "
                     f"mirror), got {args.exchange_spin}")
    spin = int(spin)
    # (flags with a non-None default only count when given; the torus tables use vvb and carry no sphere kinematics)
    inert = {
        "--basis": args.basis != "auto",
        "--coordinates": _given(raw_argv, "--coordinates"),
        "--delta-ext": args.delta_ext is not None, "--spin-ext": args.spin_ext is not None,
        "--h-ext": args.h_ext is not None, "--hbar-ext": args.hbar_ext is not None,
        "--exchange-h": args.exchange_h is not None, "--parity-ext": args.parity_ext is not None,
        "--parity-project": args.parity_project is not None, "--parity-exchanged": args.parity_exchanged is not None,
        "--chiral-norm": args.chiral_norm is not None, "--z-shift": args.z_shift is not None,
        "--zbar-shift": args.zbar_shift is not None, "--u-shift": args.u_shift is not None,
        "--center": bool(args.center), "--regularize": _given(raw_argv, "--regularize"),
        "--kac-limit": _given(raw_argv, "--kac-limit"), "--super-sector": _given(raw_argv, "--super-sector"),
        "--ramond-branch": _given(raw_argv, "--ramond-branch"),
        "--order": not onept and _given(raw_argv, "--order"),
        "--q-gate": not onept and _given(raw_argv, "--q-gate") and args.q_gate != "off",
    }
    offending = [flag for flag, hit in inert.items() if hit]
    if offending:
        parser.error(
            f"{', '.join(offending)} do(es) not apply to a torus point table: it is the {family} block on vvb at "
            "(--exchange-delta, --exchange-spin); a Kac weight is resolved as the irreducible module automatically "
            "(a recognized singular c needs no --regularize), and the characters are exact (no --order)"
        )
    if onept and args.order < 1:
        parser.error("torus one-point point tables need --order >= 1 (order 0 is the bare character)")
    mp.mp.dps = args.precision
    try:
        request = BlockRequest(
            operation="point-table", surface="torus", family=family, coordinate="vvb", coordinates=("vvb",),
            central_charge=args.central_charge, exchange_delta=args.exchange_delta, exchange_spin=spin,
            torus_onepoint_ext=args.torus_onept_ext if onept else None,
            policy=NumericalPolicy(
                precision=args.precision, order=args.order if onept else None, lambda_order=args.lam,
                target_digits=args.target_digits, max_precision=args.max_precision,
            ),
        )
    except ValueError as exc:
        parser.error(str(exc))
    os.makedirs(args.output, exist_ok=True)
    print(f"torus point table ({family}): Delta = {args.exchange_delta}, spin {spin}, c = {args.central_charge}"
          + (f", d_ext = {args.torus_onept_ext}, order {args.order}" if onept else ""), flush=True)
    try:
        table = build_table_request(request)
    except BlockPoleError as exc:
        parser.error(str(exc))
    except ValueError as exc:
        # (a fusion rule forbidding a degenerate module's one-point block, an input the backend refuses)
        parser.error(str(exc))
    meta = table.metadata or {}
    truncation = meta.get("q_truncation_digits")
    if onept and args.q_gate != "off" and truncation is not None:
        from ..qgate import DEFAULT_MIN_DIGITS

        threshold = min(DEFAULT_MIN_DIGITS, float(args.precision))
        if truncation < threshold:
            message = (f"--order {args.order} truncates the torus one-point point table at ~{truncation:.1f} digits "
                       f"(below {threshold:.0f}); raise --order")
            if args.q_gate == "strict":
                parser.error(f"q-gate (strict): {message}")
            print(f"warning: q-gate: {message}", file=sys.stderr, flush=True)
    policy = meta.get("regularization_policy")
    cli_args = shlex.join(["virasoro_blocks", "point-table", *map(str, raw_argv)])
    with mp.workdps(max(mp.mp.dps, int(getattr(table, "working_dps", None) or 0))):   # (its working precision)
        text = write_torus_json(
            table, c=request.central_charge, spin=spin, basis="torus_onept_G" if onept else "torus_G",
            onept_d_ext=request.torus_onepoint_ext if onept else None, q_order=args.order if onept else None,
            regularization_policy=policy, variable="point", point_value=request.exchange_delta,
            kac_limit="irreducible" if policy == "irreducible" else None,
            precision_input=args.precision, args=cli_args,
        )
    path = _output_path(args.output, args.output_template.replace("{spin}", str(spin)))
    with open(path, "w") as fh:
        fh.write(text)
        fh.write("\n")
    print(f"    wrote {path}", flush=True)
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    # "--ramond-branch --" / "--ramond-branch -+": argparse reads a value that looks like an option as one
    check_fixed_value_counts(parser, raw_argv)
    args = parser.parse_args(join_option_values(raw_argv, "--ramond-branch", RAMOND_BRANCHES))
    if args.lam < 0:
        parser.error("--lambda must be >= 0")
    if args.order < 0:
        parser.error("--order must be >= 0")
    if args.precision < 10:
        parser.error("--precision must be >= 10")
    if args.basis not in _BASES:
        parser.error(
            f"unknown --basis {args.basis!r}; use `virasoro_blocks info` "
            "to see the supported family-prefixed bases"
        )
    torus_basis = {"torus_G": "torus-character", "torus_onept_G": "torus-onepoint"}
    if args.surface == "torus" and not _given(raw_argv, "--family"):
        # (as `table --surface torus`: the characters, or the one-point blocks with --torus-onept-ext)
        args.family = torus_basis.get(args.basis) or (
            "torus-onepoint" if args.torus_onept_ext is not None else "torus-character")
    if (args.family in _TORUS_FAMILIES) != (args.surface == "torus") and _given(raw_argv, "--surface"):
        parser.error(f"--family {args.family} and --surface {args.surface} do not go together: the torus families "
                     "are torus-character and torus-onepoint")
    if args.basis in torus_basis and torus_basis[args.basis] != args.family:
        parser.error(f"--basis {args.basis} belongs to --family {torus_basis[args.basis]}")
    if args.family in _TORUS_FAMILIES:
        from ..verified_build import PrecisionRequest

        try:
            PrecisionRequest.make(args.precision, args.target_digits, args.max_precision)
        except ValueError as exc:
            parser.error(str(exc))
        if args.basis in torus_basis:
            args.basis = "auto"      # (the family's own basis, spelled explicitly)
        return _torus_main(parser, args, raw_argv)
    if args.family == "virasoro" and args.order < 1:
        # the numeric point sampler has no q-terms to evaluate at order 0
        parser.error("--order must be >= 1 for Virasoro point tables")
    if args.precision < MIN_TABLE_PRECISION:
        parser.error(f"--precision must be >= {MIN_TABLE_PRECISION}")
    if args.torus_onept_ext is not None:
        parser.error("--torus-onept-ext is for --surface torus --family torus-onepoint")
    # the N=1 choices, refused off their families even when spelled with the default value
    if _given(raw_argv, "--super-sector") and args.family not in ("ns", "ramond"):
        parser.error(f"--super-sector is for the N=1 superconformal blocks (--family ns / ramond); the "
                     f"{args.family} blocks have no sector")
    if _given(raw_argv, "--ramond-branch") and args.family != "ramond":
        parser.error(f"--ramond-branch is for --family ramond; the {args.family} blocks have no three-point sign "
                     "branch")
    from ..verified_build import PrecisionRequest

    try:
        precision_request = PrecisionRequest.make(args.precision, args.target_digits, args.max_precision)
    except ValueError as exc:
        parser.error(str(exc))
    if args.chiral_norm is not None and not _mp_number(args.chiral_norm) > 0:
        parser.error("--chiral-norm must be positive")
    named = _family_of_basis(args.basis)
    if named is not None and named != args.family:
        parser.error(f"{named}_* bases require --family {named}")
    if args.family != "virasoro" and _given(raw_argv, "--kac-limit"):
        parser.error(f"--kac-limit chooses the Virasoro block on a Kac weight of a singular c; the {args.family} "
                     "point tables do not read it"
                     + (" (the N=1 blocks keep the c -> c0 limit at fixed weight)" if args.family in ("ns", "ramond")
                        else ""))
    if args.family == "global" and _given(raw_argv, "--regularize"):
        parser.error("--regularize does not apply to the global blocks (exact 2F1, no central charge)")
    from ..recursion import set_kac_limit

    set_kac_limit(args.kac_limit)
    if args.family == "global" and args.central_charge is not None:
        print(
            "note: explicit global point tables are independent of "
            "--central-charge; recording null central_charge metadata",
            file=sys.stderr,
        )
        args.central_charge = None
    # Parse and derive weights above the escalation cap; the build still starts
    # at the requested working precision.
    mp.mp.dps = args.precision
    try:
        with mp.workdps(precision_request.input_dps):
            resolved = _resolve_request(args)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    if args.family == "global":
        _point_q_gate(parser, args, raw_argv, resolved)       # (refuses an explicit warn/strict up front)

    os.makedirs(args.output, exist_ok=True)

    basis = resolved["basis"]
    chiral = resolved["chiral"]
    projected = resolved["projected"]
    ell = resolved["ell"]
    point_value = resolved["point_value"]
    h_ext = resolved["h_ext"]
    hbar_ext = resolved["hbar_ext"]
    parity_ext = resolved["parity_ext"]
    parity_exchanged = resolved["parity_exchanged"]
    c = resolved["central_charge"]
    q_order = resolved["q_order"]
    center = resolved["center"]
    center_bar = resolved["center_bar"]
    z_shift = resolved["z_shift"]
    zbar_shift = resolved["zbar_shift"]

    print(
        f"point table ({basis}): {('h' if chiral else 'Delta')} = "
        f"{mp.nstr(point_value, 8)} (spin {ell}), "
        f"coordinates {args.coordinates}", flush=True,
    )
    try:
        serialized = build_table_request(resolved["request"])
    except BlockPoleError as exc:
        # a genuine exchanged-module pole: the point constraint is undefined
        parser.error(str(exc))
    # the --order truncation at this weight (probed once the weight is known to be off the poles), recorded as the
    # table JSON records it
    serialized.q_truncation_digits = _point_q_gate(parser, args, raw_argv, resolved)
    npoles = len(serialized.pole_list_x)
    cli_args = shlex.join(["virasoro_blocks", "point-table", *map(str, raw_argv)])
    # written at the precision it was built at (--target-digits may have raised it)
    mp.mp.dps = max(mp.mp.dps, int(getattr(serialized, "working_dps", None) or 0))
    chiral_norm = (
        None if args.chiral_norm is None else _mp_number(args.chiral_norm)
    )
    if args.format in ("json", "both"):
        text = write_json(
            serialized, c=c, h_ext=h_ext,
            hbar_ext=None if chiral else hbar_ext, spin=0 if chiral else ell,
            q_order=q_order, npoles=npoles, basis=basis,
            parity_projection=None if chiral else args.parity_project,
            parity_external=None if chiral else parity_ext,
            parity_exchanged=(
                None if chiral else (parity_exchanged if projected else None)
            ), chiral_norm=chiral_norm, precision_input=args.precision,
            args=cli_args, variable="point", point_value=point_value,
            z_center=center, zbar_center=None if chiral else center_bar,
            z_shift=z_shift, zbar_shift=None if chiral else zbar_shift,
            # (the NS/Ramond point tables carry their family and regularization metadata, read by the writer)
            regularization_policy=(None if args.family in ("ns", "ramond")
                                   else _point_regularization_policy(c, args.regularize)),
        )
        spin_tag = "0" if chiral else (str(int(ell)) if ell == int(ell) else str(mp.convert(ell)))
        path = _output_path(
            args.output, args.output_template.replace("{spin}", spin_tag)
        )
        with open(path, "w") as fh:
            fh.write(text)
        print(f"    wrote {path}", flush=True)
    if args.format in ("m", "both"):
        for coordinate in args.coordinates:
            text = write_mathematica(
                serialized, coordinate=coordinate, ell=0 if chiral else ell,
                basis=basis,
                parity_projection=None if chiral else args.parity_project,
                parity_external=None if chiral else parity_ext,
                parity_exchanged=(
                    None if chiral else (parity_exchanged if projected else None)
                ), chiral_norm=chiral_norm,
                z_center=center, zbar_center=None if chiral else center_bar,
                z_shift=z_shift, zbar_shift=None if chiral else zbar_shift,
                c=c, h_ext=h_ext, hbar_ext=None if chiral else hbar_ext,
                q_order=q_order, npoles=npoles, variable="point",
                point_value=point_value, precision_input=args.precision,
                output_template=args.output_template, args=cli_args,
            )
            stem = args.output_template.replace("{spin}", "0" if chiral else (str(int(ell)) if ell == int(ell) else str(mp.convert(ell))))
            stem = stem[:-5] if stem.endswith(".json") else stem
            path = _output_path(args.output, f"{stem}_{coordinate}.m")
            with open(path, "w") as fh:
                fh.write(text)
            print(f"    wrote {path}", flush=True)
    return 0
