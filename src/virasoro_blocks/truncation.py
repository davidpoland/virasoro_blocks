"""Per-coordinate (m, n) truncation policy matching blocks_3d's --lambda convention.

The `--lambda Lambda` flag bounds the total derivative weight, with the
constraint depending on the coordinate and the block basis. The rules
are pulled directly from blocks_3d's source (`congruence_transform.cxx`,
`zzb_to_xt.cxx`) and from its `doc/conventions.tex`.

Non-folded coordinates (`zzb`, `yyb`, `qqb`):
  - scalar G / parity-projected B at equal centers: m + n <= Lambda AND n <= m
    (definite z <-> zbar parity makes n > m entries redundant at that center).
  - at unequal centers, every basis keeps the full m + n <= Lambda triangle;
    swapping the derivative indices also swaps the expansion point.
  - raw ordered B (scalar or spinning externals, no parity projection):
    m + n <= Lambda with both (m, n) and (n, m) kept (no z <-> zbar
    symmetry to fold).

Parity-folded coordinates (`xt`, `ws`, `qqb_ws`):
  - m + 2n <= Lambda_+/- where Lambda_+ = Lambda (even z <-> zbar parity: the
    scalar G at every ell, a parity-projected B with sign x phase = +1) and
    Lambda_- = Lambda - 1 (odd parity: sign x phase = -1). Only the scalar G and
    parity-projected B paths can land here; raw ordered B is rejected by the
    CLI for folded coordinates already.

Radial slices (`xt_radial`, `ws_radial`, `qqb_ws_radial`):
  - only n = 0 entries, m in [0, Lambda], for either parity. The Lambda - 1
    cutoff applies only to the two-variable folded coordinates above.

Chiral coordinates (`z`, `y`, `q`):
  - only n = 0 entries, m in [0, Lambda]. The block is one-variable (the
    chiral half F(h, z)), so there is no second derivative index; the writer
    consumes the table as a degenerate (m_max + 1) x 1 layout, the same trick
    used for the radial slices.
"""

from __future__ import annotations

from .numeric_input import literal_number, _integer_control, _integer_spin


_NON_FOLDED = {"zzb", "yyb", "qqb"}
_FOLDED = {"xt", "ws", "qqb_ws"}
_RADIAL = {"xt_radial", "ws_radial", "qqb_ws_radial"}
_CHIRAL = {"z", "y", "q"}


def _ordinary_basis_name(basis: str | None) -> str | None:
    """Strip the explicit superconformal family prefix for keep rules."""
    if basis is None:
        return basis
    for prefix in ("ns_", "ramond_"):
        if basis.startswith(prefix):
            return basis[len(prefix):]
    return basis
# Torus (genus-1) modular coordinate: the disk variable v = (tau-i)/(tau+i)
# and its antichiral partner. Tables are (m, n)-symmetric (j = 0 directly;
# j > 0 after the (j, -j) pair sum), so the keep rule matches the symmetric
# non-folded one: m + n <= Lambda and n <= m.
_TORUS = {"vvb"}


def _basis_keeps_zzb_symmetry(basis: str | None) -> bool:
    """Return True if the (m, n) <-> (n, m) symmetry holds on this basis.

    Scalar G and parity-projected B both have definite z <-> zbar parity, so
    the n > m entries are determined by the n <= m entries (identically equal
    for even parity, which includes the scalar G at every spin, and up to a
    sign for odd parity). The raw ordered B has
    no such symmetry, including when its external operators are scalar.
    """
    basis = _ordinary_basis_name(basis)
    if basis is None or basis == "scalar_G" or basis == "global_scalar_G":
        return True
    if basis == "parity_projected_B" or basis == "global_parity_projected_B":
        return True
    if basis == "ordered_B" or basis == "global_ordered_B":
        return False
    if basis == "chiral_F" or basis == "global_chiral_F":
        # Chiral blocks have only one Taylor index, so the question doesn't
        # arise; report True so the non-folded branch falls through to the
        # one-axis case (chiral coordinates dispatch separately anyway).
        return True
    raise ValueError(f"unknown basis {basis!r}")


def _folded_lambda(lam: int, ell, parity_projection: str | None, basis: str | None, parity_phase=1) -> int:
    """Compute Lambda_+/- for the parity-folded coordinates.

    Even parity (Lambda_+ = Lambda) is selected by the scalar G (z <-> zbar even
    at every ell) or by a parity-projected block with z <-> zbar parity sigma =
    (projection sign) x ``parity_phase`` = +1; odd parity (Lambda_- = Lambda - 1)
    by sigma = -1.
    """
    basis = _ordinary_basis_name(basis)
    if basis == "parity_projected_B" or basis == "global_parity_projected_B":
        if parity_projection in ("+", "-"):
            import mpmath as mp

            sigma = (1 if parity_projection == "+" else -1) * mp.re(mp.mpc(parity_phase))
            return lam if sigma > 0 else lam - 1
        raise ValueError(
            "parity_projected_B with folded coordinates requires parity_projection in {'+', '-'}"
        )
    # scalar G: z <-> zbar even at every (integer) ell
    from ._conventions import scalar_G_fold_parity

    return lam if scalar_G_fold_parity(_integer_spin(ell)) == 0 else lam - 1


def is_off_diagonal(center, center_bar) -> bool:
    """True for an expansion point with ``zbar0 != z0`` (``center_bar`` None means ``zbar0 = z0``)."""
    if center_bar is None:
        return False
    # A missing holomorphic center means z0 = 1/2. Compare the supplied
    # labels before working-precision rounding can merge distinct centers.
    return literal_number(".5" if center is None else center) != literal_number(center_bar)


def kept_pairs(
    coord: str,
    lam: int,
    *,
    basis: str | None = None,
    ell=0,
    parity_projection: str | None = None,
    off_diagonal: bool = False,
    parity_phase=1,
):
    """Return the sorted list of (m, n) pairs to keep for one coordinate.

    The list is sorted lexicographically in (m, n) so the writer can emit one
    row of n entries per m index.  Off the diagonal (``off_diagonal``: an expansion
    point with ``zbar0 != z0``) no basis has the ``(m, n) <-> (n, m)`` symmetry, so
    the whole ``m + n <= Lambda`` triangle is kept.
    """
    lam = _integer_control('lambda', lam)
    ordinary_basis = _ordinary_basis_name(basis)
    if coord in _NON_FOLDED:
        if _basis_keeps_zzb_symmetry(basis) and not off_diagonal:
            return [
                (m, n)
                for m in range(lam + 1)
                for n in range(min(m, lam - m) + 1)
            ]
        # raw ordered B: m + n <= Lambda, both (m, n) and (n, m) kept.
        return [(m, n) for m in range(lam + 1) for n in range(lam - m + 1)]
    if coord in _FOLDED:
        if ordinary_basis == "ordered_B" or ordinary_basis == "global_ordered_B":
            raise ValueError(
                f"folded coordinate {coord!r} is not defined for raw ordered B; "
                "use parity_projected_B or scalar G"
            )
        lam_eff = _folded_lambda(lam, ell, parity_projection, ordinary_basis or "scalar_G", parity_phase)
        return [
            (m, n)
            for m in range(lam_eff + 1)
            for n in range((lam_eff - m) // 2 + 1 if lam_eff - m >= 0 else 0)
        ]
    if coord in _TORUS:
        if basis not in (None, "torus_G", "torus_onept_G"):
            raise ValueError(
                f"torus coordinate {coord!r} requires a torus basis "
                "('torus_G' or 'torus_onept_G')"
            )
        return [
            (m, n)
            for m in range(lam + 1)
            for n in range(min(m, lam - m) + 1)
        ]
    if coord in _RADIAL:
        return [(m, 0) for m in range(lam + 1)]
    if coord in _CHIRAL:
        if ordinary_basis not in (None, "chiral_F", "global_chiral_F"):
            raise ValueError(
                f"chiral coordinate {coord!r} requires basis='chiral_F'"
            )
        return [(m, 0) for m in range(lam + 1)]
    raise ValueError(f"unknown coordinate {coord!r}")


def total_parity_phase(parity_external=None, exchanged_parity=None):
    """The phase eta of ``B_+- = B +- eta P[B]`` (see :func:`parity.parity_reversal_phase`)."""
    from .parity import parity_reversal_phase

    return parity_reversal_phase(parity_external, exchanged_parity=exchanged_parity or "1")


def parity_smoothing_label(
    basis: str | None,
    *,
    ell=0,
    parity_projection: str | None = None,
    parity_phase=1,
) -> str | None:
    """Return blocks_3d's parity-smoothing label for this block configuration.

    For the parity-folded coordinates (`xt`, `ws`, `qqb_ws`) blocks_3d
    multiplies the block by a smoothing factor `p_+/-` before differentiating,
    so the result is smooth at `s = 0` (equivalently `t = 0`). The factor is
    `p_+ = 1` for even parity and `p_- = 2/(chiral - chiral_bar) = 1/r` for
    odd parity, where the chiral coordinate is z, y, or q depending on the
    folded coordinate family.

    We apply the same smoothing implicitly in `derivs._fold_parity` (the
    "r-stripped" branch when the z <-> zbar parity, sign x phase, is -1),
    so this helper returns the corresponding label so the output writers can
    record it as metadata. The label is:

      - "p_plus"   for even-parity outputs (scalar G at every ell;
                   parity_projected_B with sign x phase = +1).
      - "p_minus"  for odd-parity outputs (parity_projected_B with
                   sign x phase = -1).
      - None       for raw ordered_B (parity is undefined and folded
                   coordinates are rejected anyway).
    """
    basis = _ordinary_basis_name(basis)
    if basis is None or basis == "scalar_G" or basis == "global_scalar_G":
        from ._conventions import scalar_G_fold_parity

        return "p_plus" if scalar_G_fold_parity(_integer_spin(ell)) == 0 else "p_minus"
    if basis == "parity_projected_B" or basis == "global_parity_projected_B":
        if parity_projection in ("+", "-"):
            import mpmath as mp

            sigma = (1 if parity_projection == "+" else -1) * mp.re(mp.mpc(parity_phase))
            return "p_plus" if sigma > 0 else "p_minus"
        raise ValueError(
            "parity_projected_B requires parity_projection in {'+', '-'}"
        )
    if basis == "ordered_B" or basis == "global_ordered_B":
        return None
    if basis == "chiral_F" or basis == "global_chiral_F":
        # Chiral blocks live on a single complex coordinate; the p_+/- factor
        # is a 2d concept tied to z <-> zbar parity and has no chiral analogue.
        return None
    raise ValueError(f"unknown basis {basis!r}")


def required_m_max_n_max(
    coord: str,
    lam: int,
    *,
    basis: str | None = None,
    ell=0,
    parity_projection: str | None = None,
    off_diagonal: bool = False,
    parity_phase=1,
) -> tuple[int, int]:
    """Return the (m_max, n_max) bounds needed to cover every kept pair.

    Useful for sizing the internal DeltaDerivTable before truncation.
    """
    pairs = kept_pairs(
        coord, lam, basis=basis, ell=ell, parity_projection=parity_projection, off_diagonal=off_diagonal,
        parity_phase=parity_phase,
    )
    if not pairs:
        return (0, 0)
    return (max(m for m, _ in pairs), max(n for _, n in pairs))
