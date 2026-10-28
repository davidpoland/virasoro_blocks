"""Implementation of ``virasoro_blocks evaluate``."""

from __future__ import annotations

import argparse
import json
import sys

import mpmath as mp

from . import (
    check_fixed_value_counts,
    KAC_LIMIT_HELP, RAMOND_BRANCHES, RamondBranchAction, accept_negative_numbers, chiral_norm as _chiral_norm,
    join_option_values, real_number as _number, half_integer as _half_integer,
)

from ..evaluation import BlockPoleError, evaluate_block_request
from ..parity import normalize_external_parity, normalize_parity_phase
from ..requests import (
    BlockRequest,
    NumericalPolicy,
    RequestValidationError,
    capabilities_for,
)


_COORDINATES = ("zzb", "yyb", "qqb", "z", "y", "q", "vvb")
_BASES = (
    "auto", "scalar_G", "ordered_B", "parity_projected_B", "chiral_F",
    "global_scalar_G", "global_ordered_B", "global_parity_projected_B",
    "global_chiral_F",
    "ns_scalar_G", "ns_ordered_B", "ns_parity_projected_B", "ns_chiral_F",
    "ramond_scalar_G", "ramond_ordered_B", "ramond_parity_projected_B",
    "ramond_chiral_F", "torus_G", "torus_onept_G",
)


def _point(token: str):
    if "=" not in token:
        raise argparse.ArgumentTypeError("points must have the form name=value")
    key, value = token.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("point name cannot be empty")
    value = value.strip()
    if "j" in value.lower():
        # a complex cross ratio (the Euclidean section is zbar = conj(z)), e.g. z=0.4+0.3j zbar=0.4-0.3j
        try:
            from ..numeric_input import parse_number

            parsed = parse_number(value)
            if not mp.isfinite(parsed):
                raise ValueError("point must be finite")
        except (TypeError, ValueError, AttributeError) as exc:
            raise argparse.ArgumentTypeError(f"invalid complex value {value!r} (write it like 0.4+0.3j)") from exc
        return key, value.replace(" ", "")
    return key, _number(value)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="virasoro_blocks evaluate",
        description="Evaluate one conformal block (Virasoro, global, NS/Ramond, or torus) at an explicit point.",
        epilog="Example: virasoro_blocks evaluate --central-charge 8 --delta-ext 1/8 1/8 1/8 1/8 "
               "--exchange-delta 2 --exchange-spin 2 --point z=0.4 --point zbar=0.3",
        allow_abbrev=False,
    )
    p.add_argument("--surface", choices=["sphere", "torus"], default="sphere",
                   help="sphere (four-point blocks, default) or torus (torus families)")
    p.add_argument(
        "--family",
        choices=["virasoro", "global", "ns", "ramond", "torus-character", "torus-onepoint"],
        default="virasoro", help="block family (default virasoro)",
    )
    p.add_argument(
        "--basis", metavar="BASIS", default="auto",
        help="Block representation (default auto: chiral_F for a chiral coordinate, parity_projected_B with "
             "--parity-project, ordered_B for spinning externals, scalar_G otherwise). Use `virasoro_blocks info` "
             "for the supported bases for each family.",
    )
    p.add_argument("--central-charge", type=_number, default=None, metavar="C",
                   help="central charge (required except for --family global); fractions like 7/10 are exact")
    p.add_argument("--coordinate", choices=_COORDINATES, default=None,
                   help="coordinate the --point values are given in (default zzb, or vvb on --surface torus; "
                        "z/y/q are chiral)")
    p.add_argument("--point", action="append", type=_point, required=True, metavar="NAME=VALUE",
                   help="evaluation point, repeated: z=.. zbar=.. (or the coordinate's own variable, e.g. y=.. ybar=.. "
                        "for yyb); v=.. vbar=.. on the torus")
    p.add_argument("--delta-ext", nargs=4, type=_number, default=None, metavar="D",
                   help="the four external dimensions (with --spin-ext; exclusive with --h-ext)")
    p.add_argument("--spin-ext", nargs=4, type=_number, default=None, metavar="S",
                   help="the four external spins (default 0)")
    p.add_argument("--h-ext", nargs=4, type=_number, default=None, metavar="H",
                   help="the four external holomorphic weights (chiral bases, or with --hbar-ext)")
    p.add_argument("--hbar-ext", nargs=4, type=_number, default=None, metavar="HB",
                   help="the four external antiholomorphic weights")
    p.add_argument("--exchange-h", type=_number, default=None, metavar="H",
                   help="exchanged holomorphic weight (chiral bases; with --exchange-hbar for full ones)")
    p.add_argument("--exchange-hbar", type=_number, default=None, metavar="HB",
                   help="exchanged antiholomorphic weight")
    p.add_argument("--exchange-delta", type=_number, default=None, metavar="D",
                   help="exchanged dimension (with --exchange-spin)")
    p.add_argument("--exchange-spin", type=_half_integer, default=None, metavar="L",
                   help="exchanged spin (default 0; integers, or half-integers like 5/2 for spinning externals)")
    p.add_argument("--torus-onept-ext", type=_number, default=None, metavar="D",
                   help="external dimension of the torus one-point block (torus-onepoint only)")
    p.add_argument("--super-sector", choices=["even", "odd"], default="even",
                   help="NS/Ramond exchanged component (default even; --family ns / ramond only)")
    p.add_argument("--ramond-branch", choices=list(RAMOND_BRANCHES), default="++", action=RamondBranchAction,
                   help="Ramond external branch (default ++; --family ramond only). All four work as a separate "
                        "value, e.g. `--ramond-branch --`, or as `--ramond-branch=--`")
    p.add_argument("--parity-ext", nargs=4, type=normalize_parity_phase, default=None, metavar="P",
                   help="external parity phases, e.g. 1 1 1 -1 (with --parity-project)")
    p.add_argument("--parity-project", choices=["+", "-"], default=None,
                   help="parity projection sign; selects parity_projected_B under --basis auto")
    p.add_argument("--parity-exchanged", type=normalize_parity_phase, default=None, metavar="P",
                   help="exchanged-operator parity phase (default 1; parity_projected_B only)")
    p.add_argument("--chiral-norm", type=_chiral_norm, default=None, metavar="N",
                   help="chiral normalization N, a positive integer: F = N^{-h} (16 q)^h ..., N^{-h} times the "
                        "standard block at --z-shift=-h1-h2 (default 16)")
    p.add_argument("--z-shift", type=_number, default=None, metavar="S",
                   help="shift of the z exponent, F ~ z^{h + S} / N^h; for scalar_G it multiplies G by (z zbar)^S, "
                        "as in tables")
    p.add_argument("--zbar-shift", type=_number, default=None, metavar="S",
                   help="shift of the zbar exponent (ordered_B / parity_projected_B; default 0)")
    p.add_argument("--u-shift", type=_number, default=None, metavar="S",
                   help="set --z-shift and --zbar-shift together (either one given explicitly wins)")
    p.add_argument("--order", type=int, default=30, metavar="N",
                   help="q-expansion order of the recursion (default 30)")
    p.add_argument("--precision", type=int, default=60, metavar="P",
                   help="working precision in decimal digits (default 60). The value is the --order block computed "
                        "at P digits; the JSON records estimated_digits, its accuracy as the block (P less the "
                        "measured loss near a pole or z = 1, and no more than q_truncation_digits). A torus "
                        "one-point value is also checked against a deeper, higher-precision reference (the JSON "
                        "records verified_digits) and a shortfall is reported")
    p.add_argument("--target-digits", type=int, default=None, metavar="D",
                   help="deliver D digits of the --order block: the working precision is raised from --precision "
                        "by the measured loss (near a pole or z = 1; for a torus one-point value, against its "
                        "deeper reference), recorded as precision_actual; an error when --max-precision is reached. "
                        "The q-truncation is not a precision limit: --q-gate then checks it against D (raise "
                        "--order)")
    p.add_argument("--max-precision", type=int, default=None, metavar="M",
                   help="cap for --target-digits (default max(4 D, 2 P, D + 200)); refused without it")
    p.add_argument("--q-gate", dest="q_gate", choices=["off", "warn", "strict"], default="warn",
                   help="q-expansion truncation check (Virasoro, NS, Ramond): 'warn' (default) warns when the "
                        "measured accuracy at --order is below 20 digits (below D with --target-digits D) or the "
                        "series has not converged (large c needs a larger order), 'strict' errors, 'off' skips; the "
                        "JSON records q_truncation_digits either way. A torus one-point value warns when its "
                        "q-truncation is below --precision (below D with --target-digits D)")
    p.add_argument("--regularize", choices=["auto", "on", "off"], default="auto",
                   help="c-regularization at singular central charges (default auto)")
    p.add_argument("--kac-limit", choices=["irreducible", "c_limit"], default="irreducible", help=KAC_LIMIT_HELP)
    p.add_argument("--format", choices=["json", "text"], default="json",
                   help="json (value plus metadata, default) or text (the value alone)")
    p.add_argument("-o", "--output", default="-", metavar="FILE",
                   help="output file; '-' writes to stdout")
    return accept_negative_numbers(p)


def _write(text: str, output: str):
    if output == "-":
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        return
    with open(output, "w") as fh:
        fh.write(text)
        if not text.endswith("\n"):
            fh.write("\n")


def main(argv=None) -> int:
    parser = build_parser()
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    # "--ramond-branch --" / "--ramond-branch -+": argparse reads a value that looks like an option as one
    check_fixed_value_counts(parser, raw_argv)
    args = parser.parse_args(join_option_values(raw_argv, "--ramond-branch", RAMOND_BRANCHES))

    def _provided(flag: str) -> bool:
        return flag in raw_argv or any(token.startswith(flag + "=") for token in raw_argv)

    if args.order < 0:
        parser.error("--order must be >= 0")
    if args.precision < 10:
        parser.error("--precision must be >= 10")
    from ..verified_build import PrecisionRequest

    try:
        PrecisionRequest.make(args.precision, args.target_digits, args.max_precision)
    except ValueError as exc:
        parser.error(str(exc))
    if args.basis not in _BASES:
        parser.error(
            f"unknown --basis {args.basis!r}; use `virasoro_blocks info` "
            "to see the supported family-prefixed bases"
        )
    torus_family = args.family in {"torus-character", "torus-onepoint"}
    if args.coordinate is None:
        args.coordinate = "vvb" if torus_family or args.surface == "torus" else "zzb"
    capability = capabilities_for(args.family)[0]
    if args.surface not in capability.surfaces:
        parser.error(
            f"family {args.family!r} supports surfaces {capability.surfaces}, "
            f"not {args.surface!r}"
        )
    if args.coordinate not in capability.coordinates_for("evaluate"):
        parser.error(
            f"family {args.family!r} direct evaluation does not support "
            f"coordinate {args.coordinate!r}; choose one of "
            f"{capability.coordinates_for('evaluate')}"
        )
    # the N=1 choices, refused off their families even when spelled with the default value
    if _provided("--super-sector") and args.family not in {"ns", "ramond"}:
        parser.error(f"--super-sector is for the N=1 superconformal blocks (--family ns / ramond); the "
                     f"{args.family} blocks have no sector")
    if _provided("--ramond-branch") and args.family != "ramond":
        parser.error(f"--ramond-branch is for --family ramond; the {args.family} blocks have no three-point sign "
                     "branch")
    if _provided("--order") and not capability.supports_order:
        parser.error(f"family {args.family!r} direct evaluation has no --order")
    if _provided("--regularize") and args.regularize == "on" and args.family == "global":
        parser.error("global direct evaluation does not use c-regularization")
    if _provided("--kac-limit") and args.family != "virasoro":
        parser.error(f"--kac-limit chooses the Virasoro block on a Kac weight of a singular c; the {args.family} "
                     "evaluation does not read it"
                     + (" (the N=1 blocks keep the c -> c0 limit at fixed weight)" if args.family in {"ns", "ramond"}
                        else ""))
    from ..recursion import set_kac_limit

    set_kac_limit(args.kac_limit)
    if args.h_ext is not None and args.delta_ext is not None:
        parser.error("--h-ext/--hbar-ext and --delta-ext/--spin-ext are exclusive")
    if args.hbar_ext is not None and args.h_ext is None:
        parser.error("--hbar-ext requires --h-ext")
    if not torus_family and args.delta_ext is None and args.h_ext is None:
        parser.error("provide --h-ext or --delta-ext")
    if args.parity_project is not None and args.parity_ext is None:
        parser.error("--parity-project requires --parity-ext")
    explicit_chiral = args.basis in {
        "chiral_F", "global_chiral_F", "ns_chiral_F", "ramond_chiral_F"
    }
    chiral_requested = args.coordinate in {"z", "y", "q"}
    if explicit_chiral or (args.basis == "auto" and chiral_requested):
        if args.coordinate not in {"z", "y", "q"}:
            parser.error("chiral bases require --coordinate z, y, or q")
        if args.hbar_ext is not None or args.delta_ext is not None:
            parser.error("chiral bases require --h-ext only")
    elif args.coordinate in {"z", "y", "q"}:
        parser.error("full bases require --coordinate zzb, yyb, or qqb")
    chiral_eval = explicit_chiral or (args.basis == "auto" and chiral_requested)
    if torus_family:
        if args.surface != "torus" or args.coordinate != "vvb":
            parser.error("torus families require --surface torus --coordinate vvb")
        if args.central_charge is None:
            parser.error("torus evaluation requires --central-charge")
        if args.family == "torus-onepoint" and args.torus_onept_ext is None:
            parser.error("torus-onepoint evaluation requires --torus-onept-ext")
    elif args.surface != "sphere":
        parser.error("sphere families require --surface sphere")
    mp.mp.dps = args.precision
    # BlockRequest parses the original spellings above the escalation cap.
    point = dict(args.point)
    if len(point) != len(args.point):
        parser.error("duplicate --point names are not allowed")
    try:
        request = BlockRequest(
            operation="evaluate", surface=args.surface, family=args.family,
            basis=args.basis, coordinate=args.coordinate,
            h_ext=args.h_ext,
            hbar_ext=args.hbar_ext,
            delta_ext=args.delta_ext,
            spin_ext=args.spin_ext,
            central_charge=args.central_charge,
            exchange_h=args.exchange_h,
            exchange_hbar=args.exchange_hbar,
            exchange_delta=args.exchange_delta,
            exchange_spin=args.exchange_spin,
            torus_onepoint_ext=args.torus_onept_ext,
            point=point, parity=args.parity_project,
            parity_external=tuple(args.parity_ext) if args.parity_ext else None,
            parity_exchanged=args.parity_exchanged,
            sector=args.super_sector, branch=args.ramond_branch,
            chiral_norm=args.chiral_norm,
            z_shift=args.z_shift if args.z_shift is not None else args.u_shift,
            # a chiral block has no zbar half: --u-shift shifts z alone there (an explicit --zbar-shift is refused)
            zbar_shift=args.zbar_shift if args.zbar_shift is not None or chiral_eval else args.u_shift,
            policy=NumericalPolicy(
                precision=args.precision,
                order=(args.order if capability.supports_order else None),
                c_regularization=args.regularize,
                target_digits=args.target_digits,
                max_precision=args.max_precision,
            ),
        )
        result = evaluate_block_request(request, q_gate=args.q_gate)
    except (RequestValidationError, BlockPoleError, ValueError) as exc:
        parser.error(str(exc))

    truncation = result.metadata.get("q_truncation_digits")
    # (sphere values are gated by --q-gate inside evaluate_block_request)
    wanted = args.precision if args.target_digits is None else args.target_digits
    if torus_family and truncation is not None and truncation < wanted:
        # the value is the --order block, as asked; say how much of the digits asked for that truncation leaves here
        asked = f"{args.precision}" if args.target_digits is None else f"the {args.target_digits} target"
        print(
            f"warning: --order {args.order} truncates this torus one-point value at ~{truncation:.1f} of "
            f"{asked} digits (its q-series tail at this point); raise --order for more",
            file=sys.stderr,
        )

    if args.format == "json":
        text = json.dumps(result.as_dict(), indent=2, sort_keys=True)
    else:
        value = result.as_dict()["value"]
        if isinstance(value, str):
            text = value
        else:
            imag = value["imag"]
            text = f"{value['real']} - {imag[1:]}j" if imag.startswith("-") else f"{value['real']} + {imag}j"
    _write(text, args.output)
    return 0
