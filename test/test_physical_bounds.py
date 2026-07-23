"""
The sanity floor: no recommendation may claim a physically absurd drag.

This test exists because the ORIGINAL solver failed it. A sign error in the
spoiler stall limit let bolt-on parts push a Maruti Swift to Cd 0.166 — below
the Mercedes EQXX prototype, the most aerodynamic car ever built. A model that
makes a hatchback slipperier than a wind-tunnel-honed EV is broken, and no
amount of plausible-looking output redeems it.

So this is a hard, permanent canary: across every car and every modification
combination the optimiser can produce, the modified Cd must stay above the
most aerodynamic PRODUCTION car in the world.
"""

import pytest

from core.panel_solver import INDIAN_CARS, solve_car
from core.modifications import CD_PRODUCTION_RECORD, CD_PROTOTYPE_RECORD
from core.optimizer import run_grid_search

ALL = list(INDIAN_CARS)


@pytest.mark.parametrize("key", ALL)
def test_no_mod_set_beats_the_production_record(key):
    """
    The strong guarantee: EVERY feasible modification combination stays above
    the production world record (0.20). Not just the recommended tiers — the
    full grid, including the most aggressive stack of every part at once.
    """
    worst = min(s.Cd_final for s in run_grid_search(key))
    assert worst > CD_PRODUCTION_RECORD, (
        f"{key}: modifications reach Cd {worst:.3f}, at or below the most "
        f"aerodynamic production car ever ({CD_PRODUCTION_RECORD}). A bolt-on "
        f"kit cannot do that — the model is broken.")


@pytest.mark.parametrize("key", ALL)
def test_no_car_ever_approaches_the_prototype_record(key):
    """Belt and braces: nothing may even come near the EQXX prototype (0.17)."""
    worst = min(s.Cd_final for s in run_grid_search(key))
    assert worst > CD_PROTOTYPE_RECORD + 0.02, \
        f"{key}: Cd {worst:.3f} is within a whisker of the EQXX prototype"


@pytest.mark.parametrize("key", ALL)
def test_baseline_is_a_believable_car(key):
    """Every unmodified car sits in the real passenger-car band."""
    cd = solve_car(key)["Cd"]
    assert 0.24 < cd < 0.46, f"{key}: baseline Cd {cd:.3f} is not a real car"


def test_reductions_are_modest_for_the_legal_tier():
    """
    The parts an owner can actually fit without paperwork (wheel covers +
    underbody panel) must give a SMALL, believable reduction — a few percent,
    not a transformation. Overpromising on the headline recommendation is the
    most damaging kind of wrong.
    """
    for key in ["maruti_swift", "tata_nexon"]:
        base = solve_car(key)["Cd"]
        legal = run_grid_search(key, legal_only=True)
        best = min(s.Cd_final for s in legal)
        drop_pct = (base - best) / base * 100
        assert 2 < drop_pct < 15, \
            f"{key}: legal mods claim a {drop_pct:.0f}% drag cut — not believable"
