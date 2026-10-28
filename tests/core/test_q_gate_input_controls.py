"""Convergence diagnostics must honor the supplied controls and iterables."""
import pytest

from virasoro_blocks import qgate


def _report(route, **overrides):
    kwargs = dict(c=2, h_externals=[.1] * 4, q_order=0)
    if "point" in route:
        kwargs.update(h=.3, z=.5)
    else:
        kwargs.update(n_terms=1)
    if route.startswith("super"):
        kwargs.update(family="ns", sector="even")
    kwargs.update(overrides)
    return getattr(qgate, route)(**kwargs)


ROUTES = ["q_tail_report", "point_q_tail_report", "super_q_tail_report", "super_point_q_tail_report"]


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("field,bad", [("q_order", -1), ("q_order", 6.5), ("q_order", True),
                                      ("window", 0), ("window", -1), ("window", 2.5), ("window", True)])
def test_invalid_controls_rejected_even_for_short_orders(route, field, bad):
    with pytest.raises(ValueError, match=field):
        _report(route, **{field: bad})


@pytest.mark.parametrize("route", ["q_tail_report", "super_q_tail_report"])
@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_invalid_coefficient_counts_rejected(route, bad):
    with pytest.raises(ValueError, match="n_terms"):
        _report(route, n_terms=bad)


@pytest.mark.parametrize("route", ROUTES)
def test_integral_numeric_short_orders_remain_supported(route):
    assert _report(route, q_order=0.0, window=4.0)["status"] == "unconverged"


def test_external_weight_iterator_survives_cache_key_and_all_builds(monkeypatch):
    seen = []

    def factory(ext, n, **kwargs):
        seen.append(tuple(ext))
        return lambda h: [1] * n

    monkeypatch.setattr(qgate, "_evaluator_factory", lambda coordinate: factory)
    qgate._REPORT_CACHE.clear()
    try:
        report = _report("q_tail_report", q_order=12, h_externals=iter([.1] * 4))
        assert report["status"] == "ok"
        assert seen == [(.1,) * 4] * 3
    finally:
        qgate._REPORT_CACHE.clear()
