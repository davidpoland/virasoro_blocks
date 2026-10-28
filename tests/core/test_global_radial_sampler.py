"""Global radial slices are sampled in their own variable (the n = 0 row of the folded parent)."""

from __future__ import annotations

import glob
import json
import os
import tempfile

import mpmath as mp

from virasoro_blocks.__main__ import main




def _tables(args):
    with tempfile.TemporaryDirectory() as out:
        main(["table", *args, "--format", "json", "-o", out])
        return [json.load(open(path)) for path in sorted(glob.glob(os.path.join(out, "derivs_*.json")))]


def test_global_radial_slice_is_verified_in_its_own_variable():
    """A global xt_radial table is the n = 0 row of its folded xt parent; the point sampler built it as the
    non-folded zzb grid (d^m/dz^m, 2^m off), so a 28-digit table recorded -1.5 verified digits, and without
    --npoles the build escalated for nothing (npoles 30 -> 62, precision 80 -> 135)."""
    from virasoro_blocks.global_blocks import build_global_sampler

    h = [mp.mpf("0.1")] * 4
    common = dict(basis="global_scalar_G", ell=0, h_externals=h, hbar_externals=h, m_max=5)
    row = [(m, 0) for m in range(6)]
    (radial,), _ = build_global_sampler("xt_radial", n_max=0, **common).sample_bucket([mp.mpf("3.7")], row)
    (folded,), _ = build_global_sampler("xt", n_max=2, **common).sample_bucket([mp.mpf("3.7")], row)
    for m, _n in row:
        want = mp.mpc(folded[m][0])
        assert abs(mp.mpc(radial[m][0]) - want) <= mp.mpf(10) ** -40 * abs(want)       # (was off by 2^m)

    (doc,) = _tables(["--family", "global", "--basis", "global_scalar_G", "--lambda", "5", "--precision", "40",
                      "--delta-ext", "0.2", "0.2", "0.2", "0.2", "--spin-ranges", "0", "--coordinates", "xt_radial",
                      "--npoles", "16"])
    assert float(doc["verified_digits"]) > 25

