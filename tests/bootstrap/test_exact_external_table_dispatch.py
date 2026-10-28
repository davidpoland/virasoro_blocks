"""Bootstrap preparation retains external labels through the exact-table boundary."""
from fractions import Fraction

import pytest

from virasoro_blocks import qgate
from virasoro_blocks.numeric_input import ExactRational, literal_number
from virasoro_bootstrap import identical_scalar


class StopAtTables(BaseException):
    pass


@pytest.mark.parametrize('label', [ExactRational(7,10)+ExactRational(1,10**70)])
def test_external_label_reaches_exact_table_backend(tmp_path, monkeypatch, label):
    captured = []
    def capture(c, h_ext, *args, **kwargs):
        captured.append(h_ext)
        raise StopAtTables
    monkeypatch.setattr(qgate, 'gate_q_convergence', lambda **kwargs: None)
    monkeypatch.setattr(identical_scalar, '_build_exact_scalar_tables', capture)
    spec = identical_scalar.IdenticalScalarGapSpec(
        c='2', h_ext=label, delta_gap=0, lam=1, spin_list=(0,), q_order=2,
        npoles=0, dps=20, num_threads=1)
    with pytest.raises(StopAtTables):
        identical_scalar.prepare_identical_scalar_gap_build(spec)
    assert literal_number(captured[0]) == Fraction(str(label))


@pytest.mark.parametrize('label', [0.7])
def test_twist_label_reaches_tower_backend(monkeypatch, label):
    from virasoro_bootstrap import large_spin
    captured = []
    def capture(**kwargs):
        captured.append(kwargs)
        raise StopAtTables
    monkeypatch.setattr(qgate, 'gate_q_convergence', lambda **kwargs: None)
    monkeypatch.setattr(large_spin, 'large_spin_channel_jobs', capture)
    spec = identical_scalar.IdenticalScalarGapSpec(
        c='2', h_ext='.3', delta_gap=0, lam=1, spin_list=(0,), q_order=2,
        npoles=0, dps=20, num_threads=1,
        large_spin_lmin=2, large_spin_twists=(label,), large_spin_npoles=0)
    with pytest.raises(StopAtTables):
        identical_scalar.prepare_identical_scalar_gap_build(spec)
    assert literal_number(captured[0]['twists'][0]) == Fraction(str(label))
    assert literal_number(captured[0]['h_ext']) == Fraction(3,10)


def test_chiral_symbolic_external_and_crossing_labels_reach_the_backend_exactly(monkeypatch):
    import mpmath as mp
    from virasoro_bootstrap.controls import chiral_control

    captured = []
    def capture(externals, *_args, **kwargs):
        captured.append((externals, kwargs['z_shift']))
        return object()
    monkeypatch.setattr(chiral_control, 'F_z_poly_table', capture)
    with mp.workdps(30):
        for label in (.7, ExactRational(7, 10)+ExactRational(1, 10**70)):
            chiral_control._chiral_symbolic_exact_table(
                h_ext=label, lam=1, c=4, q_order=2, chiral_norm=None)
            externals, shift = captured[-1]
            assert all(literal_number(value) == literal_number(label) for value in externals)
            assert literal_number(shift) == -2*literal_number(label)


def test_chiral_numeric_identity_keeps_the_same_exact_external_as_its_channels(monkeypatch):
    import mpmath as mp
    from virasoro_bootstrap.controls import chiral_control

    captured = []
    def capture(_h, **kwargs):
        captured.append(kwargs['h_ext'])
        return (0, 1)
    monkeypatch.setattr(chiral_control, '_chiral_block_coeffs', capture)
    with mp.workdps(30):
        for mode in ('virasoro',):
            for label in (.7, ExactRational(7, 10)+ExactRational(1, 10**70)):
                chiral_control._chiral_identity_vector(mode=mode, h_ext=label, lam=1,
                    c=4, q_order=2, chiral_norm=4, F_pairs=((1, 0),))
                assert literal_number(captured[-1]) == literal_number(label)
