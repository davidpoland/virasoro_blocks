import json

import mpmath as mp
import pytest

from virasoro_blocks.__main__ import main
from virasoro_blocks.recursion_ns import F_block_ns
from virasoro_blocks.serialization import (
    serialize_deriv_tables,
    serialized_table_from_json,
)
from virasoro_blocks.superconformal_blocks import (
    SuperBlockPolicy,
    SuperconformalPoleError,
    _chiral_poles,
    FAMILY_NS,
    FAMILY_RAMOND,
    numeric_chiral_table,
    numeric_full_table,
    ns_chiral_block,
    ns_chiral_F_interpolated_table,
    ramond_chiral_F_interpolated_table,
    resolve_super_c,
    ns_B_block,
    super_full_block,
    super_2d_interpolated_table,
)
from virasoro_blocks.writers import write_json


def _close(a, b, digits=18):
    return abs(mp.mpc(a) - mp.mpc(b)) < mp.mpf(10) ** (-digits)


def test_super_fold_requires_real_parity_while_unfolded_complex_phase_is_valid():
    with mp.workdps(30):
        for family in (FAMILY_NS, FAMILY_RAMOND):
            common = dict(c=8, q_order=2, m_max=0, n_max=0)
            for basis, phase in (('parity_projected_B', 'i'), ('ordered_B', '1')):
                phases = dict(parity='+', exchanged_parity=phase)
                for coordinate in ('xt', 'ws', 'xt_radial'):
                    with pytest.raises(ValueError, match='real.*parity phase'):
                        numeric_full_table(family, basis, 2, 0, coordinate, ['.125']*4,
                                           **common, **phases)
                    with pytest.raises(ValueError, match='real.*parity phase'):
                        super_2d_interpolated_table(family, basis, coordinate, 0, ['.125']*4,
                                                   npoles=0, **common, **phases)
            table = numeric_full_table(family, 'parity_projected_B', 2, 0, 'zzb', ['.125']*4,
                                       parity='+', exchanged_parity='i', **common)
            assert table[0][0].real == table[0][0].imag != 0
            valid = numeric_full_table(family, 'parity_projected_B', 2, 0, 'xt', ['.125']*4,
                                       parity='+', exchanged_parity='i',
                                       parity_external=['i', '1', '1', '1'], **common)
            assert valid[0][0] == 0  # spin zero with equal halves and total phase -1


@pytest.mark.parametrize('family', ['ns', 'ramond'])
def test_canonical_super_tables_keep_both_derivative_axes_at_independent_centers(family):
    from virasoro_blocks.api import BlockRequest, NumericalPolicy, build_table_request

    for basis, parity in (('scalar_G', None), ('parity_projected_B', '-')):
        request = BlockRequest(operation='table', family=family, basis=basis, coordinate='zzb',
            h_ext=('.1', '.2', '.3', '.4'), hbar_ext=('.1', '.2', '.3', '.4'),
            central_charge=30, center='.3', center_bar='.45', parity=parity,
            policy=NumericalPolicy(precision=30, order=0, lambda_order=1))
        with mp.workdps(30):
            table = build_table_request(request, spin=1)
            actual = table.evaluate('zzb', '3.7')
            assert table.sample_block_values['zzb'][0][0][1] != 0
            encoded = write_json(table, c=30, h_ext=request.h_ext, hbar_ext=request.hbar_ext,
                spin=1, q_order=0, basis=f'{family}_{basis}', parity_projection=parity,
                z_center=request.center, zbar_center=request.center_bar)
            decoded = serialized_table_from_json(json.loads(encoded)).evaluate('zzb', '3.7')
        with mp.workdps(60):
            reference = numeric_full_table(family, basis, '3.7', 1, 'zzb', request.h_ext,
                hbar_ext=request.hbar_ext, c=30, q_order=0, m_max=1, n_max=1,
                center=request.center, center_bar=request.center_bar, parity=parity)
            for m, n in ((0, 0), (0, 1), (1, 0)):
                assert reference[m][n] != 0
                for values in (actual, decoded):
                    assert abs(values[m][n] - reference[m][n]) < mp.mpf('1e-24') * abs(reference[m][n])


def test_ns_numeric_adapter_matches_authoritative_block():
    mp.mp.dps = 40
    h_ext = [mp.mpf(".125")] * 4
    z = mp.mpf(".5")
    wrapped = numeric_chiral_table(
        FAMILY_NS, mp.mpf(".7"), "z", h_ext, c=mp.mpf("1.3"),
        q_order=3, m_max=0, regularize=False,
    )[0][0]
    direct = F_block_ns(
        mp.mpf(".7"), z, h_ext, c=mp.mpf("1.3"), q_order=3,
    )
    assert _close(wrapped, direct)


def test_numeric_ns_and_ramond_cover_all_coordinate_shapes():
    mp.mp.dps = 30
    h_ext = [mp.mpf(".125")] * 4
    for family in (FAMILY_NS, FAMILY_RAMOND):
        for coord in ("z", "y", "q"):
            table = numeric_chiral_table(
                family, mp.mpf(".7"), coord, h_ext, c=mp.mpf("1.3"),
                q_order=2, m_max=1, regularize=False,
            )
            assert len(table) == 2 and all(len(row) == 1 for row in table)
        for coord in ("zzb", "yyb", "qqb", "xt", "ws", "qqb_ws", "xt_radial"):
            table = numeric_full_table(
                family, "ns_scalar_G" if family == FAMILY_NS else "ramond_scalar_G",
                mp.mpf("1.4"), 0, coord, h_ext, c=mp.mpf("1.3"),
                q_order=2, m_max=1, n_max=1, regularize=False,
            )
            assert len(table) == 2
            assert len(table[0]) == (1 if coord.endswith("_radial") else 2)
            assert mp.isfinite(abs(table[0][0]))


def test_ramond_sign_branches_are_first_class_numeric_states():
    with mp.workdps(30):
        h_ext = [mp.mpf(".125")] * 4
        values = []
        for branch in ("++", "+-", "-+", "--"):
            table = numeric_chiral_table(
                FAMILY_RAMOND, mp.mpf(".7"), "z", h_ext, c=mp.mpf("1.3"),
                q_order=2, m_max=0, branch=branch, regularize=False,
            )
            values.append(table[0][0])
        assert all(mp.isfinite(abs(value)) for value in values)
        assert len({str(value) for value in values}) > 1


def test_full_ordered_and_parity_projected_forms_match_virasoro_basis_shapes():
    with mp.workdps(30):
        h_ext = [mp.mpf(".2"), mp.mpf(".25"), mp.mpf(".2"), mp.mpf(".25")]
        hbar_ext = [mp.mpf(".15"), mp.mpf(".30"), mp.mpf(".15"), mp.mpf(".30")]
        ordered = numeric_full_table(
            FAMILY_NS, "ns_ordered_B", mp.mpf("1.6"), 1, "zzb", h_ext,
            hbar_ext=hbar_ext, c=mp.mpf("1.3"), q_order=2, m_max=1,
            n_max=1, regularize=False,
        )
        projected = numeric_full_table(
            FAMILY_RAMOND, "ramond_parity_projected_B", mp.mpf("1.6"), 1,
            "ws", h_ext, hbar_ext=hbar_ext, c=mp.mpf("1.3"), q_order=2,
            m_max=1, n_max=1, parity="+", parity_external=["1"] * 4,
            regularize=False,
        )
        assert len(ordered) == 2 and len(ordered[0]) == 2
        assert len(projected) == 2 and len(projected[0]) == 2
        assert mp.isfinite(abs(ordered[0][0]))
        assert mp.isfinite(abs(projected[0][0]))


def test_interpolated_ns_and_ramond_tables_reproduce_node_values():
    mp.mp.dps = 35
    h_ext = [mp.mpf(".125")] * 4
    ns = ns_chiral_F_interpolated_table(
        "z", h_ext, c=mp.mpf("1.3"), m_max=1, q_order=2,
        npoles=2, regularize=False,
    )
    r = ramond_chiral_F_interpolated_table(
        "z", h_ext, c=mp.mpf("1.3"), m_max=1, q_order=2,
        npoles=2, regularize=False,
    )
    for table, family in ((ns, FAMILY_NS), (r, FAMILY_RAMOND)):
        x = table.numerator_sample_points[0][0][0]
        h = x + table.shift
        interpolated = table.evaluate(h)[0][0]
        direct = numeric_chiral_table(
            family, h, "z", h_ext, c=mp.mpf("1.3"), q_order=2,
            m_max=0, regularize=False,
        )[0][0]
        assert _close(interpolated, direct, digits=14)


def test_odd_pole_pool_contains_the_flipped_universal_h_zero_pole():
    """Odd blocks reach the (1,1) pole through the even inner row."""
    with mp.workdps(35):
        h_ext = [mp.mpf(".125")] * 4
        table = ns_chiral_F_interpolated_table(
            "z", h_ext, c=mp.mpf("1.3"), m_max=0, q_order=2,
            npoles=1, regularize=False, sector="odd",
        )
        assert any(abs(pole) < mp.mpf("1e-30") for pole in table.prefactor.poles)


def test_degenerate_c_auto_offset_is_recorded_and_finite():
    mp.mp.dps = 35
    c = mp.mpf(81) / 70
    c_input, c_eval, regularized = resolve_super_c(c)
    assert regularized
    assert c_eval > c_input
    h_ext = [mp.mpf(".125")] * 4
    table = ns_chiral_F_interpolated_table(
        "z", h_ext, c=c, m_max=0, q_order=2, npoles=1,
        regularize="auto",
    )
    assert mp.isfinite(abs(table.evaluate(mp.mpf(".7"))[0][0]))


@pytest.mark.parametrize("family", ["ns", "ramond"])
def test_order_reduced_tables_below_three_halves_split_the_positive_poles(tmp_path, family):
    """At c < 3/2 most NS/Ramond poles sit at x = Delta - shift > 0, where they are always kept.  They are split off
    as exact principal parts (as for the Virasoro c <= 1 tables), so --npoles counts the x <= 0 poles, the fit runs
    on nodes for those alone, and the table verifies (here to 15-25 digits: the monomial form of some 60 forced poles
    cancels the rest); fitting them into the numerator reproduced not even its own nodes (c = 1.2: -121 digits at
    precision 120)."""
    main(["table", "--family", family, "--central-charge", "0.9", "--delta-ext", "0.1", "0.1", "0.1", "0.1",
          "--spin-ranges", "0", "--lambda", "3", "--order", "12", "--npoles", "2", "--precision", "40",
          "-o", str(tmp_path)])
    doc = json.loads((tmp_path / "derivs_0.json").read_text())
    poles = [mp.mpf(p) for p in doc["pole_list_x"]]
    positive = sum(1 for p in poles if p > 0)
    assert positive > 20 and len(poles) - positive == 2
    # npoles records the requested count, as for Virasoro (the bootstrap checks it against its spec's npoles)
    assert doc["npoles"] == 2
    # the candidate pool runs two levels past the block's own recursion (half-level 2 * 12); a forced pole of those
    # deeper levels has no principal part in the block and is not written (each chiral pole d is a double pole 2d)
    with mp.workdps(40):
        def chiral_positive(level):
            pool = _chiral_poles(family, mp.mpf("0.9"), [mp.mpf("0.05")] * 4, q_order=level, sector=0, branch=0)
            return sum(1 for d in pool if mp.im(mp.mpc(d)) == 0 and mp.re(d) > 0)

        assert chiral_positive(28) > chiral_positive(24)
        assert positive == 2 * chiral_positive(24)
    # the fit's nodes follow the two kept poles and the entries' tail powers, not the forced ones
    assert len(doc["sample_points_x"]) <= 2 + 3 + 1
    assert float(doc["verified_digits"]) > 10
    # independently: the direct block between the nodes and well past them
    table = serialized_table_from_json(doc)
    h_ext = [mp.mpf("0.05")] * 4
    with mp.workdps(40):
        for delta in (mp.mpf("0.7"), mp.mpf("5.5"), mp.mpf(30)):
            got = table.evaluate("zzb", delta)
            ref = numeric_full_table(family, "scalar_G", delta, 0, "zzb", h_ext, c=mp.mpf("0.9"), q_order=12, m_max=3)
            for m in range(4):
                for n in range(min(m, 3 - m) + 1):
                    assert abs(got[m][n] - ref[m][n]) <= mp.mpf(10) ** -10 * abs(ref[m][n])


def test_offset_c_tables_split_the_positive_poles_too(tmp_path):
    """--regularize on at a c that is not singular evaluates at c + 10^-eps_exp: the x > 0 poles are split off there
    too (principal parts of the offset block, under the sampler's guard), not fit into the numerator (which needed a
    node per forced pole and reproduced not one digit: -47 at c = 1.2, lambda 4, order 20)."""
    main(["table", "--family", "ns", "--central-charge", "0.9", "--regularize", "on", "--delta-ext", "0.1", "0.1",
          "0.1", "0.1", "--spin-ranges", "0", "--lambda", "3", "--order", "12", "--npoles", "2", "--precision", "40",
          "-o", str(tmp_path)])
    doc = json.loads((tmp_path / "derivs_0.json").read_text())
    assert doc["regularization_policy"] == "c+eps"
    assert len(doc["sample_points_x"]) <= 2 + 3 + 1
    assert float(doc["verified_digits"]) > 10


def test_super_cli_writes_family_metadata(tmp_path):
    main([
        "--family", "ramond", "--central-charge", "1.3",
        "--lambda", "0", "--h-ext", ".125", ".125", ".125", ".125",
        "--coordinates", "z", "--order", "2", "--npoles", "1",
        "--precision", "25", "--format", "json", "-o", str(tmp_path),
    ])
    doc = json.loads((tmp_path / "derivs_chiral.json").read_text())
    assert doc["basis"] == "ramond_chiral_F"
    assert doc["block_family"] == "ramond"
    assert doc["ramond_branch"] == "++"
    assert doc["regularization_policy"] == "direct"
    assert doc["collision_policy"] == "raise"


def test_super_cli_writes_family_metadata_to_mathematica(tmp_path):
    main([
        "--family", "ns", "--central-charge", "1.3",
        "--lambda", "0", "--h-ext", ".125", ".125", ".125", ".125",
        "--coordinates", "z", "--order", "2", "--npoles", "1",
        "--precision", "25", "--format", "m", "-o", str(tmp_path),
    ])
    text = (tmp_path / "derivs_chiral_z.m").read_text()
    assert 'blockFamily -> "ns"' in text
    assert 'superSector -> "even"' in text
    assert 'regularizationPolicy -> "direct"' in text
    assert 'collisionPolicy -> "raise"' in text


@pytest.mark.parametrize(
    "flag, value",
    [("--super-order", "2"), ("--super-collision", "allow")],
)
def test_super_cli_has_no_family_specific_order_or_pole_bypass_flags(
    tmp_path, flag, value,
):
    """The public CLI shares --order and never exposes raw pole evaluation."""
    with pytest.raises(SystemExit):
        main([
            "--family", "ns", "--central-charge", "1.3",
            "--lambda", "0", "--h-ext", ".125", ".125", ".125", ".125",
            "--coordinates", "z", "--order", "2", "--npoles", "1",
            flag, value, "--precision", "25", "--format", "json",
            "-o", str(tmp_path),
        ])


def test_super_cli_accepts_explicit_full_family_basis(tmp_path):
    """The installed CLI must dispatch explicit ns_* full bases as well."""
    main([
        "--family", "ns", "--basis", "ns_scalar_G", "--central-charge", "1.3",
        "--lambda", "0", "--h-ext", ".125", ".125", ".125", ".125",
        "--hbar-ext", ".125", ".125", ".125", ".125",
        "--coordinates", "zzb", "--spin-ranges", "0", "--order", "2",
        "--npoles", "1", "--precision", "25", "--format", "json",
        "-o", str(tmp_path),
    ])
    doc = json.loads((tmp_path / "derivs_0.json").read_text())
    assert doc["basis"] == "ns_scalar_G"
    assert doc["block_family"] == "ns"
    restored = serialized_table_from_json(doc)
    with mp.workdps(30):
        assert mp.isfinite(abs(restored.evaluate("zzb", mp.mpf("1.4"))[0][0]))


def test_direct_full_block_matches_left_right_chiral_product():
    """The numeric full API uses the same ordered-product convention as Virasoro."""
    mp.mp.dps = 35
    h_ext = [mp.mpf(".125")] * 4
    h, hbar = mp.mpf(".7"), mp.mpf(".8")
    z, zbar = mp.mpf(".37"), mp.mpf(".61")
    left = ns_chiral_block(h, z, h_ext, c=mp.mpf("1.3"), q_order=3)
    right = ns_chiral_block(hbar, zbar, h_ext, c=mp.mpf("1.3"), q_order=3)
    got = ns_B_block(
        h, hbar, z, zbar, h_ext, c=mp.mpf("1.3"), q_order=3
    )
    assert _close(got, left * right, digits=28)
    dispatched = super_full_block(
        FAMILY_NS, "ns_ordered_B", h, hbar, z, zbar, h_ext,
        c=mp.mpf("1.3"), q_order=3,
    )
    assert _close(dispatched, got, digits=28)


def test_direct_api_fails_closed_at_an_exchanged_kac_pole():
    """A raw h=0 odd block must not leak a precision-dependent pole value."""
    with mp.workdps(40):
        h_ext = [mp.mpf(".125")] * 4
        with pytest.raises(SuperconformalPoleError, match="Kac pole"):
            ns_chiral_block(
                0, mp.mpf(".5"), h_ext, c=mp.mpf("1.3"), q_order=3,
                sector="odd",
            )
        with pytest.raises(TypeError):
            ns_chiral_block(
                0, mp.mpf(".5"), h_ext, c=mp.mpf("1.3"), q_order=3,
                sector="odd", collision="allow",
            )


def test_degenerate_c_policy_is_shared_by_numeric_and_interpolation():
    """The singular-c decision is shared by the direct and interpolated paths: both run the exact Laurent recursion
    at c itself (merged poles, no offset), and agree."""
    with mp.workdps(40):
        c = mp.mpf(81) / 70
        h_ext = [mp.mpf(3) / 70] * 4
        policy = SuperBlockPolicy(regularize="auto", eps_exp=24)
        direct = numeric_chiral_table(
            FAMILY_RAMOND, mp.mpf(".7"), "q", h_ext, c=c, q_order=2,
            m_max=1, policy=policy,
        )
        table = ramond_chiral_F_interpolated_table(
            "q", h_ext, c=c, m_max=1, q_order=2, npoles=1,
            policy=policy,
        )
        assert mp.isfinite(abs(direct[0][0]))
        value = table.evaluate(mp.mpf(".7"))[0][0]
        assert abs(value - direct[0][0]) <= mp.mpf(10) ** -20 * abs(direct[0][0])
        meta = getattr(table, "super_metadata")
        assert meta["regularization_policy"] == "merged" and meta["regularization_eps_exp"] is None


def test_super_metadata_round_trips_through_shared_serialization():
    with mp.workdps(35):
        h_ext = [mp.mpf(".125")] * 4
        table = ns_chiral_F_interpolated_table(
            "q", h_ext, c=mp.mpf("1.3"), m_max=1, q_order=2,
            npoles=1, policy=SuperBlockPolicy(),
        )
        serialized = serialize_deriv_tables({"q": table})
        doc = json.loads(write_json(
            serialized, c=mp.mpf("1.3"), h_ext=h_ext, basis="ns_chiral_F",
            q_order=2, npoles=1,
        ))
        assert doc["block_family"] == "ns"
        assert doc["collision_policy"] == "raise"
        restored = serialized_table_from_json(doc)
        assert restored.metadata["block_family"] == "ns"
        assert restored.metadata["collision_policy"] == "raise"


def test_ns_scalar_G_block_accepts_real_weights():
    """NS evaluation accepts complex-typed weights with zero imaginary part."""
    from virasoro_blocks.superconformal_blocks import ns_B_block, ns_G_block

    R = mp.mpf
    with mp.workdps(30):
        g = ns_G_block(R("1.37"), 0, R("0.3"), R("0.3"), [R("0.1")] * 4, c=R("0.9"), q_order=6)
        b = ns_B_block(mp.mpc("0.6", 0), R("0.7"), R("0.3"), R("0.3"), [R("0.1")] * 4, c=R("0.9"), q_order=6)
    assert abs(mp.im(g)) < 1e-25 and mp.isfinite(b)


def test_c_offset_evaluation_keeps_its_digits_at_moderate_precision():
    """At an N=1 minimal-model c the block is evaluated at c + 1e-30; the coincident Kac poles cancel as
    1/eps^k and ate every digit below dps ~100 (dps 60 returned -1.5e5).  The evaluation now runs with guard
    digits, and a weight exactly on a pole of the input c still raises."""
    R = mp.mpf
    args = (R("0.37"), R("0.3"), [R(1) / 10] * 4)
    # the offset follows the precision (eps = 10^-(dps + 10) at dps 100), so a dps-100 reference is good to ~1e-100
    with mp.workdps(100):
        ref = ns_chiral_block(*args, c=R(7) / 10, q_order=16)
    for dps in (30, 60):
        with mp.workdps(dps):
            value = ns_chiral_block(*args, c=R(7) / 10, q_order=16)
        assert abs(value / ref - 1) < mp.mpf(10) ** -(dps - 3)
    with mp.workdps(120):
        with pytest.raises(SuperconformalPoleError):
            ns_chiral_block(R(1) / 10, R("0.3"), [R(1) / 10] * 4, c=R(7) / 10, q_order=8)
