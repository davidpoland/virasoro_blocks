"""Clean block values at a table's interpolation nodes.

An order-reduced table is written, and handed to the SDP, as its values at its own sample nodes: the numeric
recursion's samples (or, for an entry fit on its own smaller node set, its Lagrange interpolant evaluated there),
never the monomial-form numerator evaluated at the node, whose coefficients cancel ~0.7 digits per unit of degree.
Shared by the CLI table path and the bootstrap builders.
"""

from __future__ import annotations

import mpmath as mp

from .interpolation import lagrange_matrix
from .polyderivs import InterpolatedDeltaTable, _principal_part_values_many
from .polynomial import _arb_to_mpf
from .precision import _validation_tolerance


def _real_tol(x) -> mp.mpf:
    eps = _validation_tolerance(guard=5)
    magnitude = abs(mp.mpc(x))
    return eps * max(magnitude, mp.mpf(1))


def _canonicalize_real_output(value):
    z = mp.mpc(value)
    if abs(mp.im(z)) <= _real_tol(z):
        return mp.mpf(mp.re(z))
    return z


def _canonicalize_sample_output(value):
    """Keep complex sample data; only an exactly zero imaginary part is real."""
    if isinstance(value, mp.mpf):
        return value
    if not isinstance(value, mp.mpc):
        value = mp.convert(value)
    return value.real if isinstance(value, mp.mpc) and value.imag == 0 else value


def _as_mp(value):
    """An mpmath number from an mpmath or (exact-midpoint) flint arb value."""
    return value if isinstance(value, (mp.mpf, mp.mpc)) else _arb_to_mpf(value)


def _interpolation_samples(sample_source, entries, count):
    """``(tables, is_arb)`` of the raw samples an interpolated table kept for ``entries``, or None."""
    records = getattr(sample_source, "numerator_sample_values", None)
    if records is None:
        return None
    picked = [records[m][n] for (m, n) in entries]
    if any(record is None or len(record[1]) != count for record in picked):
        return None
    is_arb = picked[0][0]
    if any(record[0] != is_arb for record in picked):
        return None
    convert = (lambda value: value)
    if is_arb:
        from virasoro_blocks.polynomial import _mpf_to_arb as convert
    tables = [
        [[None] * (sample_source.n_max + 1) for _ in range(sample_source.m_max + 1)]
        for _ in range(count)
    ]
    for (m, n), record in zip(entries, picked):
        for k, value in enumerate(record[1]):
            tables[k][m][n] = convert(value)
    return tables, is_arb


def _interpolation_numerator_rows(sample_source, entries, count):
    """Per entry, the arb numerator values the Lagrange fit interpolated at its nodes, or None when not recorded."""
    records = getattr(sample_source, "numerator_sample_values", None)
    if records is None:
        return None
    picked = [records[m][n] for (m, n) in entries]
    if any(record is None or len(record) < 3 or record[2] is None or len(record[2]) != count for record in picked):
        return None
    if any(isinstance(value, mp.mpc) and value.imag != 0 for record in picked for value in record[2]):
        return None
    from .polynomial import _mpf_to_arb

    return [[_mpf_to_arb(mp.mpf(mp.re(value))) for value in record[2]] for record in picked]


def attach_clean_sample_block_values(serialized, sample_source, exact, coordinate: str):
    """Sample the raw block cleanly at the serialized sample points.

    Stores ``serialized.sample_block_values[coordinate][k] = block_table(Delta_k)``
    (the ``[m][n]`` derivative table at ``Delta_k = x_k + shift``) from the
    numeric recursion -- the v2 source consumed by the sdp writer and written
    into block files. For per-entry interpolation the shared writer nodes are
    not the same as every entry's own fit nodes, so the clean values must be
    reconstructed for the reduced interpolant rather than re-sampling the exact
    block at the shared nodes. The values are the fit's recorded node samples when
    the source is an interpolated table, else the numeric sampler's; no-op when
    neither exists or there are no sample points (e.g. ``explicit_sample_points``
    suppressed).

    Only the kept (nonzero-numerator) ``(m, n)`` entries are sampled and stored;
    the rest of the grid stays zero. For scalar G the non-kept entries are the
    ``n > m`` half (fixed by the ``G[n][m] = G[m][n]`` symmetry and never
    read by ``F_minus_pairs``) plus the ``m + n > Lambda`` corner -- both unused
    by the crossing functional, so restricting the sampler to the kept entries
    avoids assembling those outer products without changing any consumed value.
    """
    sampler = getattr(exact, "numeric_sampler", None)
    # without a numeric sampler (symbolic mode) the interpolation's own recorded node samples still serve
    if serialized.sample_points_x is None or (sampler is None and not isinstance(sample_source, InterpolatedDeltaTable)):
        return
    shift = mp.mpf(mp.re(serialized.shift))
    sample_x = [mp.mpf(mp.re(x)) for x in serialized.sample_points_x]
    deltas = [x + shift for x in sample_x]
    if not isinstance(sample_source, InterpolatedDeltaTable):
        tables, _ = sampler(deltas)
    else:
        # the prefactor the numerators were fit against (the remainder's when poles on x > 0 were split off as
        # principal parts, which are added back at the writer nodes below)
        principal_parts = getattr(sample_source, "principal_parts", None)
        fit_prefactor = getattr(sample_source, "_fit_prefactor_value", sample_source._prefactor_value)
        writer_prefactors = [fit_prefactor(x) for x in sample_x]
        lagrange_entries = []
        # Real arb views of the writer prefactors for the arb-native fill below.
        # The clean-sample grid is (kept entries) x (n_samples) ~ O(grid * deg)
        # values; doing the prefactor multiply + canonicalization in mpmath there
        # was an O(Lambda^3) churn of mpf objects (the dominant super-quadratic
        # term at large Lambda). When every writer prefactor is real, keep the
        # whole fill in arb and store via the fast dyadic arb->mpf.
        writer_prefactors_arb = None
        from flint import arb as _arb_t, ctx as _flint_ctx_t
        if all(
            isinstance(p, mp.mpf) or (isinstance(p, mp.mpc) and p.imag == 0)
            for p in writer_prefactors
        ):
            _flint_ctx_t.prec = int(mp.mp.dps * 3.33) + 20
            writer_prefactors_arb = [
                _arb_t(p if isinstance(p, mp.mpf) else mp.mpf(p.real))
                for p in writer_prefactors
            ]
        node_table_cache: dict[tuple, tuple[list, bool]] = {}
        prefactor_cache: dict[tuple, list] = {}
        basis_cache: dict[tuple, tuple] = {}
        zero = mp.mpc(0)   # the non-kept placeholders share one (immutable) zero
        tables = [
            [[zero] * (sample_source.n_max + 1) for _ in range(sample_source.m_max + 1)]
            for _ in sample_x
        ]
        default_nodes = tuple(mp.mpf(mp.re(delta)) for delta in sample_source.sample_points)
        # Pre-compute writer delta tuple once for fast-path identity check below.
        writer_deltas = tuple(deltas)

        # Group the entries that carry a (nonzero) interpolant by their fit-node
        # set, and record the (m, n) targets per node set.  Zero-numerator entries
        # are the non-kept placeholders (the n > m half -- determined by the
        # G[n][m] = G[m][n] symmetry and never read by F_minus_pairs --
        # plus the m + n > Lambda corner); they are never sampled or stored, so
        # the numeric sampler only assembles the kept (m, n) outer products
        # instead of the full (m_max+1)(n_max+1) square.
        entries_by_nodes: dict[tuple, list] = {}
        for m, row in enumerate(sample_source.numerators):
            for n, poly_mn in enumerate(row):
                if not any(poly_mn.coeffs):
                    continue
                raw_nodes = (
                    sample_source.numerator_sample_points[m][n]
                    if sample_source.numerator_sample_points is not None
                    else default_nodes
                )
                node_deltas = tuple(mp.mpf(mp.re(delta)) for delta in raw_nodes)
                entries_by_nodes.setdefault(node_deltas, []).append((m, n))

        # `node_deltas` is the group key, so every entry in a group shares the
        # same fit nodes -- the fast/slow decision is per group, not per entry.
        for node_deltas, node_entries in entries_by_nodes.items():
            # The raw block at this node set (only the kept entries): the interpolation sampled exactly these
            # nodes for exactly these entries, so its samples are reused; the sampler runs only without them.
            if node_deltas not in node_table_cache:
                recorded = _interpolation_samples(sample_source, node_entries, len(node_deltas))
                if recorded is None and sampler is None:
                    return
                node_table_cache[node_deltas] = recorded or sampler(node_deltas, entries=node_entries)
            sampled_tables, sampled_is_arb = node_table_cache[node_deltas]

            # Fast path: fit nodes == writer nodes (max-count kept entries).
            # The Lagrange interpolant evaluated at one of its own fit nodes
            # trivially returns the fit value, and local_prefactors equals
            # writer_prefactors at the same x-coordinates so they cancel:
            #   tables[k][m][n] = writer_pf[k] * (sampled[k][m][n] / local_pf[k])
            #                    = sampled[k][m][n]
            # Skip the Lagrange fit and polynomial evaluation entirely.
            if node_deltas == writer_deltas:
                if sampled_is_arb:
                    # Real arb sample values: the dyadic arb->mpf is exact and
                    # skips the per-value mpc boxing + imag-check of
                    # _canonicalize_real_output (this fill is O(grid * n_samples)).
                    for (m, n) in node_entries:
                        for k in range(len(deltas)):
                            tables[k][m][n] = _arb_to_mpf(sampled_tables[k][m][n])
                else:
                    for (m, n) in node_entries:
                        for k in range(len(deltas)):
                            tables[k][m][n] = _canonicalize_sample_output(sampled_tables[k][m][n])
                continue

            # Lower-count entries are fit on their own, smaller node set: their value at a writer node is their
            # Lagrange interpolant there, formed by products from the values the fit interpolated (never a
            # monomial-form polynomial evaluated at the node).
            n_nodes = len(node_deltas)
            if node_deltas not in basis_cache:
                basis_cache[node_deltas] = lagrange_matrix([delta - shift for delta in node_deltas], sample_x)
            L = basis_cache[node_deltas]
            lagrange_entries.extend(node_entries)
            rows = _interpolation_numerator_rows(sample_source, node_entries, n_nodes)
            if rows is None:
                if node_deltas not in prefactor_cache:
                    prefactor_cache[node_deltas] = [fit_prefactor(d - shift) for d in node_deltas]
                local_prefactors = prefactor_cache[node_deltas]
                pp = (_principal_part_values_many(principal_parts, list(node_deltas), node_entries)
                      if principal_parts is not None else None)
                rows = [
                    [(_canonicalize_sample_output(_as_mp(sampled_tables[k][m][n]))
                      - (0 if pp is None else pp[k][(m, n)])) / local_prefactors[k]
                     for k in range(n_nodes)]
                    for (m, n) in node_entries
                ]
            if not isinstance(L, list) and all(not isinstance(v, mp.mpc) for row in rows for v in row):
                from flint import arb, arb_mat

                values = L * arb_mat([[v if isinstance(v, arb) else arb(v) for v in row] for row in rows]).transpose()
                for e, (m, n) in enumerate(node_entries):
                    for k in range(len(sample_x)):
                        if writer_prefactors_arb is not None:
                            tables[k][m][n] = _arb_to_mpf(values[k, e] * writer_prefactors_arb[k])
                        else:
                            tables[k][m][n] = _canonicalize_sample_output(writer_prefactors[k] * _arb_to_mpf(values[k, e]))
                continue
            L_mp = [[_as_mp(v) for v in row] for row in (L if isinstance(L, list) else L.tolist())]
            for e, (m, n) in enumerate(node_entries):
                row = [_as_mp(v) for v in rows[e]]
                for k in range(len(sample_x)):
                    tables[k][m][n] = _canonicalize_sample_output(writer_prefactors[k] * mp.fdot(L_mp[k], row))
        if principal_parts is not None and lagrange_entries:
            for k, values in enumerate(_principal_part_values_many(principal_parts, deltas, lagrange_entries)):
                for (m, n), value in values.items():
                    tables[k][m][n] = _canonicalize_sample_output(tables[k][m][n] + value)
    if serialized.sample_block_values is None:
        serialized.sample_block_values = {}
    serialized.sample_block_values[coordinate] = tables


def attach_recorded_sample_block_values(serialized, tables_by_coordinate) -> None:
    """:func:`attach_clean_sample_block_values` for tables whose fit recorded its node samples (no sampler needed).

    A table without such records is left as it is (the SDP writer then evaluates its polynomial form).
    """
    from types import SimpleNamespace

    no_sampler = SimpleNamespace(numeric_sampler=None)
    for coordinate, table in tables_by_coordinate.items():
        attach_clean_sample_block_values(serialized, table, no_sampler, coordinate)
