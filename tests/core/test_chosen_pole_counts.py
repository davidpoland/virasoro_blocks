"""Chosen pole counts: global and torus one-point tables and large-spin towers without --npoles."""

import json
from types import SimpleNamespace

import mpmath as mp
import pytest

from virasoro_blocks.__main__ import main as table_main
from virasoro_blocks.verified_build import PrecisionRequest


TOWER = ["table", "--delta-ext", "0.1", "0.1", "0.1", "0.1", "--large-spin-lmin", "10",
         "--large-spin-double-twists", "1", "--lambda", "1", "--precision", "30"]


def _tower(tmp_path, *extra):
    assert table_main([*TOWER, *extra, "-o", str(tmp_path)]) in (0, None)
    return json.loads((tmp_path / "derivs_large_spin_0.json").read_text())


def test_virasoro_tower_without_npoles_chooses_a_better_count(tmp_path):
    chosen = _tower(tmp_path / "chosen", "--central-charge", "8", "--order", "8")
    fixed = _tower(tmp_path / "fixed", "--central-charge", "8", "--order", "8", "--npoles", "2")
    assert chosen["npoles"] > 2 and float(chosen["verified_digits"]) > float(fixed["verified_digits"])


def test_global_tower_without_npoles_enlarges_its_pole_pool(tmp_path):
    from virasoro_blocks.global_blocks import _global_chiral_q_order_pole_pool

    doc = _tower(tmp_path, "--family", "global", "--order", "4")
    assert doc["order"] == 4 and doc["order_used"] > 4
    pool = _global_chiral_q_order_pole_pool(doc["order_used"], delta12=0, delta34=0)
    assert len(pool) >= doc["npoles"] > len(_global_chiral_q_order_pole_pool(4, delta12=0, delta34=0))


def test_tower_refuses_npoles_all(tmp_path, capsys):
    with pytest.raises(SystemExit):
        table_main([*TOWER, "--central-charge", "8", "--npoles", "all", "-o", str(tmp_path)])
    assert "not --npoles all" in capsys.readouterr().err


def test_global_tower_q_order_reaches_the_requested_pool():
    from virasoro_blocks.large_spin import global_tower_q_order

    # identical externals: only the odd chiral poles h = -p/2 are genuine, so n poles need order 2n
    assert global_tower_q_order(4, 10, [mp.mpf("0.05")] * 4) == 20
    assert global_tower_q_order(30, 10, [mp.mpf("0.05")] * 4) == 30


@pytest.mark.parametrize("target", [None, 40])
def test_pole_count_knobs_grow_then_halve_back_once(target):
    from virasoro_blocks.table_adapters import pole_count_knobs

    precision = PrecisionRequest.make(30, target)
    state = {"npoles": 30, "previous": None}
    knobs = {knob.cause: knob for knob in pole_count_knobs(state, precision, 1000)}
    assert set(knobs) == ({"npoles"} if target else {"npoles", "precision"})
    grow = knobs["npoles"].advance
    assert grow(SimpleNamespace(digits=10), 30) == 30 and state["npoles"] > 30
    grown = state["npoles"]
    # the grown count closed less than a quarter of the gap: one step halfway back, then stop
    assert grow(SimpleNamespace(digits=11), 30) == 30 and state["npoles"] == (grown + 30) // 2
    assert grow(SimpleNamespace(digits=11), 30) is None
    if target is None:
        state.update(npoles=grown, halved=False)
        assert knobs["precision"].advance(SimpleNamespace(digits=5), 30) == 30
        assert state["npoles"] == (grown + 30) // 2
        assert knobs["precision"].advance(SimpleNamespace(digits=5), 30) is None


@pytest.mark.parametrize("call", [
    lambda: __import__("virasoro_blocks.torus_onept", fromlist=["x"]).torus_onept_delta_table(
        4, 0, 0, d_ext=".7", q_order=1),
    lambda: __import__("virasoro_blocks.global_blocks", fromlist=["x"]).global_scalar_G_interpolated_table(
        "zzb", 0, [.1] * 4, 0),
    lambda: __import__("virasoro_blocks.global_blocks", fromlist=["x"]).global_chiral_F_interpolated_table(
        "z", [.1] * 4, 0),
])
def test_table_helpers_take_no_default_pole_count(call):
    with pytest.raises(TypeError, match="npoles"):
        call()


def test_pole_count_knobs_do_not_rebuild_a_table_the_pool_already_fixes():
    # at c <= 1 a torus one-point table can have only a few remainder poles: a larger count keeps the same table
    from virasoro_blocks.table_adapters import pole_count_knobs

    state = {"npoles": 4, "previous": None, "pool": 6}
    grow, back_off = (knob.advance for knob in pole_count_knobs(state, PrecisionRequest.make(60), 1000))
    assert grow(SimpleNamespace(digits=5), 60) == 60 and state["npoles"] > 6
    # the grown count already keeps the whole pool, so neither more poles nor halfway back can change the table
    assert grow(SimpleNamespace(digits=40), 60) is None
    assert back_off(SimpleNamespace(digits=40), 60) is None


def test_torus_table_reports_its_remainder_pool():
    from virasoro_blocks.torus_onept import build_torus_onept_state, torus_onept_delta_table

    with mp.workdps(30):
        kernel = build_torus_onept_state(6, d_ext=".7", c=4)
        small = torus_onept_delta_table(4, 0, 1, d_ext=".7", q_order=6, npoles=2, kernel=kernel)
        whole = torus_onept_delta_table(4, 0, 1, d_ext=".7", q_order=6, npoles=small.remainder_pool, kernel=kernel)
    assert small.remainder_pool == whole.remainder_pool >= len(whole.pole_list_x) > len(small.pole_list_x)
