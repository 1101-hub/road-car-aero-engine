"""
Tests for cost and payback — the decision layer.

The payback range must be built from the honest corners: cheapest parts with
the optimistic saving, priciest parts with the pessimistic saving. A single
confident payback number would repeat the false-precision mistake this
project was built to correct.
"""

import pytest

from core.costs import (MOD_COST_INR, mod_set_cost, compute_payback,
                        CAR_LIFETIME_YR)
from core.optimizer import run_grid_search


def test_every_catalogue_mod_is_priced():
    for name in ["wheel_covers", "underbody_panel", "front_splitter",
                 "side_skirts", "rear_spoiler", "rear_diffuser"]:
        lo, hi, note = MOD_COST_INR[name]
        assert 0 < lo < hi, f"{name}: cost range must be ordered and positive"
        assert len(note) > 10, f"{name}: the money must say what it buys"


def test_set_cost_is_the_sum():
    lo, hi = mod_set_cost(["wheel_covers", "underbody_panel"])
    assert lo == MOD_COST_INR["wheel_covers"][0] + MOD_COST_INR["underbody_panel"][0]
    assert hi == MOD_COST_INR["wheel_covers"][1] + MOD_COST_INR["underbody_panel"][1]


def test_payback_corners_are_honest():
    """Best case = cheap parts / optimistic saving; worst = pricey parts /
    pessimistic saving. The corners must bracket the naive midpoint."""
    pb = compute_payback(["wheel_covers"], annual_saving_INR=1000.0,
                         annual_saving_unc_INR=300.0)
    lo, hi = MOD_COST_INR["wheel_covers"][:2]
    assert pb.payback_low_yr == pytest.approx(lo / 1300.0)
    assert pb.payback_high_yr == pytest.approx(hi / 700.0)
    naive = (lo + hi) / 2 / 1000.0
    assert pb.payback_low_yr < naive < pb.payback_high_yr


def test_worthless_mod_never_pays_back():
    pb = compute_payback(["rear_spoiler"], annual_saving_INR=0.0,
                         annual_saving_unc_INR=0.0)
    assert pb.payback_low_yr is None
    assert not pb.pays_back_within_life
    assert "never" in pb.payback_str()


def test_saving_swallowed_by_uncertainty_flags_the_risk():
    """If the pessimistic saving is zero or negative, the worst case is
    'never' and the flag must say so, even though the best case looks fine."""
    pb = compute_payback(["wheel_covers"], annual_saving_INR=500.0,
                         annual_saving_unc_INR=600.0)
    assert pb.payback_low_yr is not None
    assert pb.payback_high_yr is None
    assert not pb.pays_back_within_life
    assert "NOT" in pb.payback_str()


def test_optimizer_solutions_carry_payback():
    """Every recommendation the user sees must come priced."""
    sols = run_grid_search("maruti_swift", legal_only=True)
    assert sols
    for s in sols:
        assert s.payback is not None
        assert s.payback.cost_low_INR > 0
        assert s.payback.payback_low_yr is not None


def test_legal_tier_pays_back_within_car_life():
    """The headline recommendation (legal-only, best saving) must pay for
    itself within the car's remaining life even in the WORST corner —
    otherwise the tool is recommending a bad purchase."""
    sols = run_grid_search("maruti_swift", legal_only=True)
    best = max(sols, key=lambda s: s.delta_L_100km)
    assert best.payback.pays_back_within_life, \
        (f"{best.mod_names}: worst-case payback "
         f"{best.payback.payback_high_yr} yr exceeds {CAR_LIFETIME_YR} yr")