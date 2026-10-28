"""Command implementations for the public :mod:`virasoro_blocks` CLI."""

import argparse
import re
from fractions import Fraction

from ..numeric_input import parse_number

# Recognize negative fractions, scientific notation, and complex numbers as
# argument values, so options such as "--u-shift -1/5" work without '='.
_DECIMAL = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_NEGATIVE_NUMBER = re.compile(
    rf"^(?:-{_DECIMAL}(?:/[-+]?{_DECIMAL})?|-{_DECIMAL}(?:[+-]{_DECIMAL})?[jJ]|-i|-I)$"
)

# The four Ramond three-point sign branches (--ramond-branch).
RAMOND_BRANCHES = ("++", "+-", "-+", "--")

#: --kac-limit (recursion.kac_limit_mode), shared by table, point-table and evaluate
KAC_LIMIT_HELP = (
    "The Virasoro block ON a Kac weight of a recognized singular c (c = 1, c = 25, minimal models, rational "
    "b^2): 'irreducible' (default) the block of the irreducible module L(h), whose OPE coefficients a compact CFT "
    "at that c has; 'c_limit' the c -> c0 limit at fixed h, which equals it except where Kac poles of one level "
    "merge (every Kac weight at c = 1 and c = 25, from level 2(m+2) at h = m^2/4 for c = 1). Off the Kac weights, "
    "and in Delta tables (which carry the block as a function of Delta), the two agree. Recorded as kac_limit. "
    "Virasoro family only (the NS/Ramond blocks keep the c -> c0 limit)."
)


class RamondBranchAction(argparse.Action):
    """Keep the literal '--' branch with Python 3.10's argparse too."""

    def __call__(self, parser, namespace, values, option_string=None):
        # Python 3.10 removes '--' even from an explicit option value, yielding
        # [] instead of '--'. A required single-value option cannot otherwise
        # produce an empty list. Newer argparse versions retain the string.
        setattr(namespace, self.dest, "--" if values == [] else values)


def real_number(token: str) -> str:
    """Validate a finite real literal while retaining its exact spelling."""
    token = str(token).strip()
    try:
        value = parse_number(token)
    except ZeroDivisionError as exc:
        raise argparse.ArgumentTypeError("numeric denominator must be nonzero") from exc
    except (TypeError, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError(f"invalid numeric value {token!r}") from exc
    if not isinstance(value, Fraction):
        raise argparse.ArgumentTypeError(f"invalid real numeric value {token!r}")
    return token


def chiral_norm(token: str) -> str:
    """Validate ``--chiral-norm``: a positive integer, kept in its own spelling.

    The library's chiral normalization accepts any positive real ``N``, but the CLI writes tables for the bootstrap
    layer, whose specs and manifests reuse integer normalizations only; a non-integer ``N`` would write a table no
    problem can read back.
    """
    token = real_number(token)
    value = parse_number(token)
    if value.denominator != 1 or value <= 0:
        raise argparse.ArgumentTypeError(
            f"--chiral-norm must be a positive integer (default 16), got {token!r}; "
            "the bootstrap specs and block manifests reuse integer normalizations only"
        )
    return token


def half_integer(token: str) -> str:
    """Validate spin quantization before any floating-point conversion."""
    token = real_number(token)
    if parse_number(token).denominator not in (1, 2):
        raise argparse.ArgumentTypeError("value must be an integer or half-integer")
    return token


def accept_negative_numbers(parser):
    """Accept negative numeric literals and the ``-i`` parity phase as values."""
    parser._negative_number_matcher = _NEGATIVE_NUMBER
    return parser


def check_fixed_value_counts(parser, argv):
    """Refuse a fixed-count option (``--delta-ext D D D D``) given too few values before the next option.

    argparse (3.12 and later) fills such an option with the following option names, so ``--delta-ext 0.5 --lambda
    3`` read ``--lambda`` and ``3`` as dimensions and failed later with an unrelated "unrecognized arguments" or
    "required: --lambda" error.  The count is checked here, on the raw arguments, with the option named."""
    options = parser._option_string_actions
    for i, token in enumerate(argv):
        action = options.get(token)
        if action is None or not isinstance(action.nargs, int) or action.nargs < 2:
            continue
        values = argv[i + 1:i + 1 + action.nargs]
        given = next((k for k, value in enumerate(values) if value in options or value == "--"), len(values))
        if given < action.nargs:
            parser.error(f"{token} expects {action.nargs} values, got {given}")


def join_option_values(argv, flag, values):
    """Rewrite ``flag V`` as ``flag=V`` wherever ``V`` is one of ``values``.

    argparse cannot take a separate value that looks like an option: ``--ramond-branch --`` read ``--`` as the
    end-of-options marker and ``--ramond-branch -+`` read ``-+`` as an unknown option, so both failed with
    "expected one argument" while ``--ramond-branch=--`` worked.  The flag always takes exactly one value, so the
    token after it is that value; joining the two is unambiguous.
    """
    argv = list(argv)
    out = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == flag and i + 1 < len(argv) and argv[i + 1] in values:
            out.append(f"{flag}={argv[i + 1]}")
            i += 2
            continue
        out.append(token)
        i += 1
    return out
