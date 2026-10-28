"""A vanishing truncated row cannot hide nonzero preceding tail windows."""
import mpmath as mp
import pytest

from virasoro_blocks.qgate import _rows_digits, _probes_report, act_on_q_report, QConvergenceError
from virasoro_blocks import qgate


@pytest.mark.parametrize('previous, earlier', [([1], [2]), ([0], [1]), ([1j], [0])])
def test_zero_final_row_with_nonzero_history_is_unconverged(previous, earlier):
    digits, ratio, bad = _rows_digits([mp.mpf(0)], previous, earlier, 1, 40)
    assert bad and digits == 0
    report = _probes_report([{'h': '1', 'digits': float(digits)}], digits, ratio, bad,
                            q_order=12, window=4, probe_c=2)
    with pytest.raises(QConvergenceError):
        act_on_q_report(report, mode='strict')


def test_identically_zero_row_remains_ignorable():
    assert _rows_digits([0], [0], [0], 1, 40) == (None, None, False)


def test_bad_zero_probe_is_not_hidden_by_other_converged_probes(monkeypatch):
    def factory(externals, n_terms, *, q_order, **kwargs):
        return lambda h: [mp.mpf((12-q_order)//4) if h < 1 else mp.mpf(1)]
    monkeypatch.setattr(qgate, '_evaluator_factory', lambda coordinate: factory)
    qgate._REPORT_CACHE.clear()
    try:
        report = qgate.q_tail_report(c=2, h_externals=['.1']*4, q_order=12, n_terms=1)
        assert report['status'] == 'unconverged'
        with pytest.raises(QConvergenceError):
            act_on_q_report(report, mode='strict')
    finally:
        qgate._REPORT_CACHE.clear()
