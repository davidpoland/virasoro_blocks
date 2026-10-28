"""Tests for the blocks_3d-style truncation policy in `virasoro_blocks.truncation`."""

from __future__ import annotations

import pytest
import mpmath as mp
from fractions import Fraction

from virasoro_blocks.truncation import (
    kept_pairs,
    parity_smoothing_label,
    required_m_max_n_max,
)


@pytest.mark.parametrize('family', ['global', 'ns', 'ramond'])
def test_table_cli_handles_an_empty_odd_folded_derivative_selection(family, tmp_path):
    """An odd-parity folded table at lambda 0 keeps no entry (Lambda_- = -1).  That is a parity '-' projection;
    an odd-spin scalar_G is z <-> zbar even and keeps (0, 0)."""
    import json
    from virasoro_blocks.__main__ import main

    common = ['--lambda', '0', '--order', '2', '--precision', '25', '--coordinates', 'xt',
              '--q-gate', 'off', '--format', 'json']
    if family != 'global':
        common += ['--central-charge', '30']
    odd = ['table', '--family', family, '--basis', 'parity_projected_B', '--spin-ranges', '1',
           '--delta-ext', '0.15', '1.25', '0.35', '1.45', '--spin-ext', '0', '1', '0', '1',
           '--parity-ext', '1', '1', '1', '1', '--parity-project', '-', *common, '-o', str(tmp_path / 'odd')]
    assert main(odd) in (None, 0)
    doc = json.loads((tmp_path / 'odd' / 'derivs_1.json').read_text())
    assert doc['derivs'] == [[]]
    assert 'verified_digits' not in doc
    even = ['table', '--family', family, '--basis', 'scalar_G', '--spin-ranges', '1', *common,
            '-o', str(tmp_path / 'even')]
    assert main(even) in (None, 0)
    doc = json.loads((tmp_path / 'even' / 'derivs_1.json').read_text())
    assert [len(rows) for rows in doc['derivs'][0]] == [1]
    assert doc['parity_smoothing'] == 'p_plus' and doc['scalar_G_odd'] == 'even'


def test_zzb_scalar_G_jagged_at_lambda_5():
    """blocks_3d's congruence_transform writes row m with length min(m, L-m)+1
    (n <= m AND m + n <= L). For L = 5 the row lengths are 1,2,3,3,2,1."""
    pairs = kept_pairs("zzb", 5, basis="scalar_G", ell=0)
    rows = {}
    for m, n in pairs:
        rows.setdefault(m, []).append(n)
    lengths = [len(rows[m]) for m in sorted(rows)]
    assert lengths == [1, 2, 3, 3, 2, 1]
    for m, ns in rows.items():
        assert all(n <= m for n in ns)
        assert all(m + n <= 5 for n in ns)


def test_yyb_qqb_same_policy_as_zzb():
    p_z = set(kept_pairs("zzb", 4, basis="scalar_G", ell=2))
    p_y = set(kept_pairs("yyb", 4, basis="scalar_G", ell=2))
    p_q = set(kept_pairs("qqb", 4, basis="scalar_G", ell=2))
    assert p_z == p_y == p_q


def test_zzb_parity_projected_uses_scalar_rule():
    p_g = set(kept_pairs("zzb", 4, basis="scalar_G", ell=2))
    p_b = set(kept_pairs("zzb", 4, basis="parity_projected_B", ell=2))
    assert p_g == p_b


def test_zzb_ordered_B_triangle():
    """Raw ordered B has no z <-> zbar symmetry, so we keep both (m, n) and
    (n, m), but the total-derivative cap m + n <= Lambda still applies."""
    pairs = kept_pairs("zzb", 3, basis="ordered_B", ell=0)
    expected = [
        (m, n) for m in range(4) for n in range(4) if m + n <= 3
    ]
    assert sorted(pairs) == sorted(expected)
    # Both halves of the triangle are present (no n <= m folding).
    assert (0, 3) in pairs and (3, 0) in pairs
    assert (1, 2) in pairs and (2, 1) in pairs


def test_xt_even_ell_uses_lambda_plus():
    """xt: m + 2n <= Lambda_+ = Lambda for even ell."""
    pairs = kept_pairs("xt", 4, basis="scalar_G", ell=0)
    assert all(m + 2 * n <= 4 for m, n in pairs)
    # Largest m: 4 (n=0). Largest n: 2 (m=0).
    assert (4, 0) in pairs
    assert (0, 2) in pairs
    assert (2, 1) in pairs
    assert (3, 1) not in pairs  # 3 + 2 = 5 > 4
    assert (1, 2) not in pairs  # 1 + 4 = 5 > 4


def test_xt_odd_ell_uses_lambda_plus():
    """xt: the scalar G is z <-> zbar even at every spin, so odd ell keeps m + 2n <= Lambda_+ = Lambda too."""
    assert kept_pairs("xt", 4, basis="scalar_G", ell=1) == kept_pairs("xt", 4, basis="scalar_G", ell=0)
    assert kept_pairs("xt", 4, basis="scalar_G", ell=3) == kept_pairs("xt", 4, basis="scalar_G", ell=2)


def test_xt_odd_parity_uses_lambda_minus():
    """xt: m + 2n <= Lambda_- = Lambda - 1 for a z <-> zbar-odd (parity '-') block."""
    pairs = kept_pairs("xt", 4, basis="parity_projected_B", ell=1, parity_projection="-")
    assert all(m + 2 * n <= 3 for m, n in pairs)
    assert (3, 0) in pairs
    assert (4, 0) not in pairs  # 4 > 3
    assert (1, 1) in pairs
    assert (0, 2) not in pairs  # 0 + 4 > 3


def test_ws_qqb_ws_share_xt_policy():
    p_xt = set(kept_pairs("xt", 5, basis="scalar_G", ell=2))
    p_ws = set(kept_pairs("ws", 5, basis="scalar_G", ell=2))
    p_qw = set(kept_pairs("qqb_ws", 5, basis="scalar_G", ell=2))
    assert p_xt == p_ws == p_qw


def test_folded_with_ordered_B_is_rejected():
    """blocks_3d's folded coords require parity-eigenstate input."""
    with pytest.raises(ValueError):
        kept_pairs("xt", 4, basis="ordered_B", ell=0)


def test_parity_projected_xt_requires_explicit_projection_sign():
    with pytest.raises(ValueError):
        kept_pairs("xt", 4, basis="parity_projected_B", ell=0, parity_projection=None)
    plus = set(kept_pairs("xt", 4, basis="parity_projected_B", ell=0, parity_projection="+"))
    minus = set(kept_pairs("xt", 4, basis="parity_projected_B", ell=0, parity_projection="-"))
    assert plus != minus
    assert all(m + 2 * n <= 4 for m, n in plus)
    assert all(m + 2 * n <= 3 for m, n in minus)


def test_radial_only_n_zero():
    for coord in ("xt_radial", "ws_radial", "qqb_ws_radial"):
        pairs = kept_pairs(coord, 4, basis="scalar_G", ell=0)
        assert pairs == [(m, 0) for m in range(5)]


def test_required_m_max_n_max_for_zzb():
    m_max, n_max = required_m_max_n_max("zzb", 5, basis="scalar_G", ell=0)
    assert m_max == 5
    # n_max sits at m = floor(L/2) = 2, so n_max = 2.
    assert n_max == 2


def test_unknown_coordinate_raises():
    with pytest.raises(ValueError):
        kept_pairs("nope", 3)


def test_truncation_cutoffs_follow_exact_integer_control_validation():
    for invalid in (True, False, -1, 2.5, float('inf'), float('nan'),
                    Fraction(2) + Fraction(1, 10**70)):
        with pytest.raises(ValueError):
            kept_pairs('z', invalid)
    for integral in (2.0, Fraction(2), mp.mpf(2)):
        assert kept_pairs('z', integral) == [(0, 0), (1, 0), (2, 0)]
        assert required_m_max_n_max('z', integral) == (2, 0)


def test_scalar_truncation_and_smoothing_do_not_round_fractional_spins():
    for invalid in (True, float('inf'), float('nan'), .5, 2.5,
                    Fraction(2) + Fraction(1, 10**70)):
        with pytest.raises(ValueError):
            kept_pairs('ws', 3, ell=invalid)
        with pytest.raises(ValueError):
            parity_smoothing_label('scalar_G', ell=invalid)
    for spin in (Fraction(-3), '-3', mp.mpf(-3), -3.0):
        assert kept_pairs('ws', 3, ell=spin) == kept_pairs('ws', 3, ell=3)
        assert parity_smoothing_label('scalar_G', ell=spin) == 'p_plus'


# ---------------------------------------------------------------------------
# parity_smoothing_label
# ---------------------------------------------------------------------------


def test_parity_smoothing_scalar_G_even_ell_is_p_plus():
    assert parity_smoothing_label("scalar_G", ell=0) == "p_plus"
    assert parity_smoothing_label("scalar_G", ell=2) == "p_plus"
    assert parity_smoothing_label("scalar_G", ell=4) == "p_plus"
    # default basis (None) = scalar_G.
    assert parity_smoothing_label(None, ell=2) == "p_plus"


def test_parity_smoothing_scalar_G_odd_ell_is_p_plus():
    # z <-> zbar even at every spin: no p_- smoothing for an odd spin
    assert parity_smoothing_label("scalar_G", ell=1) == "p_plus"
    assert parity_smoothing_label("global_scalar_G", ell=3) == "p_plus"


def test_parity_smoothing_parity_projected_B_follows_projection():
    assert parity_smoothing_label(
        "parity_projected_B", ell=0, parity_projection="+"
    ) == "p_plus"
    assert parity_smoothing_label(
        "parity_projected_B", ell=2, parity_projection="-"
    ) == "p_minus"
    # ell value is ignored for the projected basis; only the projection sign
    # determines the parity.
    assert parity_smoothing_label(
        "parity_projected_B", ell=1, parity_projection="+"
    ) == "p_plus"


def test_parity_smoothing_parity_projected_B_requires_projection_sign():
    with pytest.raises(ValueError):
        parity_smoothing_label(
            "parity_projected_B", ell=0, parity_projection=None
        )


def test_parity_smoothing_ordered_B_is_null():
    assert parity_smoothing_label("ordered_B", ell=0) is None
    assert parity_smoothing_label("ordered_B", ell=1) is None
