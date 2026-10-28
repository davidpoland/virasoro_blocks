"""NS/Ramond tables at a singular c where a pole near the support amplifies the c + eps offset."""

import json

import mpmath as mp


def test_ns_three_halves_generic_externals_are_built_exactly(tmp_path, capsys):
    """c = 3/2 with generic externals: at c + eps the table was left 2 digits short at every precision ('not a
    precision problem'), then needed an offset that followed the working precision.  It is now built at c itself
    from the exact Laurent recursion (merged poles, no offset): the files record c itself and no offset, and
    --target-digits escalates to the advised precision and delivers (the shortfall at --precision 40 is the
    monomial form's rounding)."""
    import re

    from virasoro_blocks.__main__ import main

    args = ["table", "--family", "ns", "--central-charge", "1.5", "--delta-ext", "0.1", "0.3", "0.25", "0.15",
            "--lambda", "3", "--order", "12", "--spin-ranges", "0", "--precision", "40"]
    assert main([*args, "-o", str(tmp_path / "p")]) in (0, None)
    advised = int(re.search(r"Raise --precision to about (\d+)", capsys.readouterr().err).group(1))
    assert main([*args, "--target-digits", "40", "-o", str(tmp_path)]) in (0, None)
    working = int(re.search(r"built at working precision (\d+)", capsys.readouterr().err).group(1))
    doc = json.load(open(tmp_path / "derivs_0.json"))
    # The advice estimates precision for 40 digits. Automatic rebuilding also
    # requires its verification margin: 75 working digits can deliver 42.5,
    # enough for the advice but just short of the automatic 43-digit goal.
    from virasoro_blocks.verified_build import verification_margin

    assert float(doc["verified_digits"]) >= 40 + verification_margin(40)
    assert advised > 40 and working > 40
    # Serialized construction digits include input/evaluation guards; the
    # precision advised for rebuilding describes the outer working context.
    assert int(doc["precision_actual"]) >= working
    assert doc["regularization_policy"] == "merged" and doc.get("regularization_eps_exp") is None
    assert mp.mpf(doc["central_charge_evaluated"]) == mp.mpf("1.5")
