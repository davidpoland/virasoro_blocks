import mpmath as mp

import virasoro_blocks.regularization as R


def test_collide_cluster_detects_coincident_weights_by_ratio_gap():
    """The colliding probe weights (tiny distance to the limit) are returned;
    the genuinely-distinct O(1)-separated weights are excluded. Detection is by
    the multiplicative gap in sorted distance, not a tuned absolute window."""
    mp.mp.dps = 60
    hv = mp.mpf("-6.875")
    # Three weights collapsing onto hv as t -> 0, plus distinct neighbors.
    poles = [
        hv + mp.mpf("1e-25"),
        hv - mp.mpf("2e-25"),
        hv + mp.mpf("3e-25"),
        hv + mp.mpf("0.4"),
        hv - mp.mpf("1.1"),
    ]
    near = R._collide_cluster(poles, hv)
    assert len(near) == 3
    for w in near:
        assert abs(w - hv) < mp.mpf("1e-10")


def test_collide_cluster_returns_empty_at_a_regular_point():
    """With every weight O(0.1..1) away there is no decisive gap, so hv is a
    regular point and no weight is reported as colliding."""
    mp.mp.dps = 60
    hv = mp.mpf("2.25")
    poles = [hv + mp.mpf("0.3"), hv - mp.mpf("0.5"), hv + mp.mpf("1.2")]
    assert R._collide_cluster(poles, hv) == []


def test_collide_cluster_includes_exact_hits():
    """Weights sitting exactly on hv are always part of the cluster."""
    mp.mp.dps = 60
    hv = mp.mpf("-2")
    poles = [mp.mpf("-2"), mp.mpf("-2"), mp.mpf("-3.375"), mp.mpf("-9")]
    near = R._collide_cluster(poles, hv)
    assert len(near) == 2
    assert all(abs(w - hv) == 0 for w in near)


def test_collide_cluster_is_scale_free_across_probe_spacings():
    """The same cluster is found whether the colliding weights sit 1e-13 or
    1e-30 from the limit -- the ratio-gap criterion does not depend on the
    probe spacing (unlike a fixed absolute window)."""
    mp.mp.dps = 80
    hv = mp.mpf("0.5")
    for gap in (mp.mpf("1e-13"), mp.mpf("1e-22"), mp.mpf("1e-30")):
        poles = [hv + gap, hv - gap, hv + mp.mpf("0.7"), hv - mp.mpf("0.9")]
        near = R._collide_cluster(poles, hv)
        assert len(near) == 2, gap
