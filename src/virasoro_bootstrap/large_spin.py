"""Large-spin asymptotic positivity channels for the identical-scalar bootstrap.

The block construction lives in `virasoro_blocks.large_spin` (fast,
`chiral_z_series_evaluator`). This module is the bootstrap-side **policy** + PMP
adapter: the physically motivated double-twist twist set, and turning the
fixed-twist spin-L blocks into `IdenticalScalarSpinChannel`s. The channels are
gap-independent, so `prepare_identical_scalar_gap_build` builds them once and
reuses them across a whole binary search.
"""
from __future__ import annotations

import mpmath as mp
from virasoro_blocks.parallel import usable_workers
from types import SimpleNamespace

from virasoro_blocks.large_spin import _LargeSpinTwistSampler, large_spin_twist_table
from virasoro_blocks.serialization import serialize_deriv_tables

from .defaults import DEFAULT_LARGE_SPIN_POLE_SELECTION
from .identical_scalar import (
    IdenticalScalarSpinChannel,
    _as_real_label,
    _attach_clean_sample_block_values,
)


class _LargeSpinBatchSampler:
    """Adapt the scalar ``L -> table`` tower sampler to the fixed-spin batch API."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __call__(self, values, entries=None):
        # Route through the batched `sample_bucket` (flint-native) rather than the
        # per-L `__call__` (which takes the mpmath `_assemble` -> `_fold_parity`
        # round trip). The caller (`_attach_clean_sample_block_values`) passes the
        # kept (nonzero-numerator) entries it will actually reconstruct; restrict
        # the bucket assembly to those instead of the full (m, n) square. Falls
        # back to the full grid when no restriction is supplied.
        s = self.sampler
        if entries is None:
            entries = [
                (m, n) for m in range(s.m_max + 1) for n in range(s.n_max + 1)
            ]
        tables, is_arb = s.sample_bucket(list(values), entries)
        return tables, is_arb


def double_twist_set(h_ext, n_count):
    """The single-correlator double twists ``tau_n = 2*Delta_ext + 2n`` (``Delta_ext = 2 h_ext``),
    ``n = 0 .. n_count-1``.  Which twists a problem's tower constrains is the caller's choice: see
    :func:`twists_above_gap` for a twist gap."""
    import operator
    import numpy as np

    try:
        if isinstance(n_count, (bool, np.bool_)):
            raise TypeError
        n_count = operator.index(n_count)
    except TypeError:
        raise ValueError("n_count must be a non-negative integer") from None
    if n_count < 0:
        raise ValueError("n_count must be a non-negative integer")
    h_ext = _as_real_label(h_ext, name="h_ext")
    delta_ext = 2 * h_ext
    return [2 * delta_ext + 2 * n for n in range(n_count)]


def twists_above_gap(twists, twist_gap):
    """The tower slices of a problem with a twist gap: the gap itself (the lowest twist that exists), then the
    ``twists`` above it (those at or below the gap do not exist, so no positivity is asked there)."""
    tg = _as_real_label(twist_gap, name="twist_gap")
    retained = [tg]
    for twist in twists:
        label = _as_real_label(twist, name="twist")
        if label > tg:
            retained.append(twist)
    return retained


def build_large_spin_channels(
    *, mode, c, h_ext, lam, q_order, npoles, Lmax, twists,
    coordinate="zzb",
    chiral_norm=None, pole_selection=DEFAULT_LARGE_SPIN_POLE_SELECTION, shared_sample_points=False,
    num_threads=1, sample_via="numeric",
):
    """One positivity channel per twist slice, enforcing positivity on the
    continuous spin tower ``L >= Lmax`` at each twist. The positivity variable is
    the spin ``L``; ``delta_min = Lmax`` labels each channel."""
    if sample_via not in ("numeric", "symbolic"):
        raise ValueError(f"sample_via must be 'numeric' or 'symbolic', got {sample_via!r}")
    twists = tuple(twists)
    # The twist slices are independent: they run in parallel (one table per worker, its own bucket pool off),
    # as the fixed-spin channels do; serially they were the largest serial step of a bootstrap build.
    num_threads = usable_workers(num_threads)
    jobs = large_spin_channel_jobs(
        mode=mode, c=c, h_ext=h_ext, lam=lam, q_order=q_order, npoles=npoles, Lmax=Lmax, twists=twists,
        coordinate=coordinate, chiral_norm=chiral_norm, pole_selection=pole_selection,
        shared_sample_points=shared_sample_points, sample_via=sample_via,
        inner_threads=num_threads if len(twists) <= 1 else 1,
    )
    if num_threads <= 1 or len(jobs) <= 1:
        return tuple(_large_spin_channel_task(job) for job in jobs)
    from concurrent.futures import ProcessPoolExecutor
    from virasoro_blocks._conventions import pool_initializer_kwargs

    with ProcessPoolExecutor(max_workers=min(num_threads, len(jobs)), **pool_initializer_kwargs()) as pool:
        return tuple(pool.map(_large_spin_channel_task, jobs))


def large_spin_channel_jobs(
    *, mode, c, h_ext, lam, q_order, npoles, Lmax, twists, coordinate="zzb", chiral_norm=None,
    pole_selection=DEFAULT_LARGE_SPIN_POLE_SELECTION, shared_sample_points=False, sample_via="numeric",
    inner_threads=1,
):
    """The picklable per-twist jobs of :func:`build_large_spin_channels` (for a caller's own process pool)."""
    if sample_via not in ("numeric", "symbolic"):
        raise ValueError(f"sample_via must be 'numeric' or 'symbolic', got {sample_via!r}")
    return [
        (mp.mp.dps, tw, mode, c, h_ext, lam, q_order, npoles, Lmax, coordinate, chiral_norm, pole_selection,
         shared_sample_points, sample_via, inner_threads)
        for tw in twists
    ]


def _verify_tower_slice(serialized, table, coordinate, lam, Lmax, dps) -> None:
    """Verify one tower slice as the fixed-spin blocks are (``verified_digits`` recorded, ``serialized.verification``
    diagnosed for the shortfall report): against the fixed-twist block between its spin nodes.  Its SDP data are
    order-reduced like theirs (Lambda 19, c = 1.00001, large_spin_npoles 12: -2.4 digits per entry, unreported)."""
    from virasoro_blocks.large_spin import tower_verified_digits
    from virasoro_blocks.truncation import kept_pairs
    from virasoro_blocks.verified_build import (
        BOOTSTRAP_KNOBS, PrecisionRequest, diagnose_monomial, record_verification,
    )

    request = PrecisionRequest.make(int(dps), knobs=BOOTSTRAP_KNOBS)
    kept = kept_pairs(coordinate, lam, basis="scalar_G", ell=int(Lmax))
    verification = tower_verified_digits(serialized, {coordinate: table}, {coordinate: kept}, request.digits)
    if verification is not None and request.short(verification.digits):
        verification = diagnose_monomial(serialized, verification, digits=request.digits, work=int(dps),
                                         order_reduced=True, kept_by_coord={coordinate: kept})
    record_verification(serialized, verification, int(dps), request)
    serialized.verification = verification


def _large_spin_channel_task(job):
    """One twist slice of the large-spin tower (picklable, for the process pool)."""
    (dps, tw, mode, c, h_ext, lam, q_order, npoles, Lmax, coordinate, chiral_norm, pole_selection,
     shared_sample_points, sample_via, num_threads) = job
    mp.mp.dps = dps
    h_ext = _as_real_label(h_ext, name="h_ext")
    tw = _as_real_label(tw, name="twist")
    z_shift = -2 * h_ext
    sampler = _LargeSpinTwistSampler(
        mode, coordinate, [h_ext] * 4, chiral_norm, z_shift,
        lam, lam, tw, c=c, q_order=q_order,
    )
    table = large_spin_twist_table(
        tw, mode=mode, coordinate=coordinate,
        h_externals=[h_ext] * 4, m_max=lam, n_max=lam,
        Lmax=Lmax, npoles=npoles, c=c, q_order=q_order, chiral_norm=chiral_norm,
        z_shift=z_shift, pole_selection=pole_selection,
        shared_sample_points=shared_sample_points, num_threads=num_threads)
    serialized = serialize_deriv_tables({coordinate: table})
    serialized.sample_via = sample_via
    if sample_via == "numeric":
        _attach_clean_sample_block_values(
            serialized,
            table,
            SimpleNamespace(numeric_sampler=_LargeSpinBatchSampler(sampler)),
            coordinate,
        )
    _verify_tower_slice(serialized, table, coordinate, lam, Lmax, dps)
    # Feed the library-normalized block (self-dual base 16 q(1/2)/N) straight
    # to the PMP. Reverting via N^Delta would only undo the conditioning gain
    # and cannot change the bound (N^Delta > 0 preserves per-operator
    # positivity), so the SDP sees base 16 q(1/2)/N directly.
    prefactored = serialized
    return IdenticalScalarSpinChannel(
        ell=int(Lmax), delta_min=mp.mpf(Lmax), serialized=serialized,
        prefactored=prefactored, twist=tw)
