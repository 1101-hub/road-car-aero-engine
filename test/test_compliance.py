"""
Tests for Layer 0 — legal and practical compliance.

The point of this layer is that a tool telling an Indian owner to bolt a
splitter onto their car owes them the truth about whether that is legal and
whether it survives the first speed breaker.

The speed-breaker limits are DERIVED from geometry, so they are testable against
closed-form arithmetic. The statutory limits are an engineering reading of the
Motor Vehicles Act and are marked UNVERIFIED in the source — these tests check
that the tool SURFACES them, not that the law is what we think it is.
"""

import numpy as np
import pytest

from core.panel_solver import INDIAN_CARS, get_car_params
from core.compliance import (
    hump_radius_m, belly_clearance_required_mm, hump_ramp_angle_deg,
    approach_angle_deg, check_compliance, legally_unrestricted_mods,
    MOD_LEGALITY, Legality, VerificationStatus, MVA_SECTION_52,
    HUMP_CHORD_M, HUMP_HEIGHT_M, CLEARANCE_MARGIN_MM,
)
from core.optimizer import run_grid_search

CARS = list(INDIAN_CARS)


# ════════════════════════════════════════════════════════════════
# THE GEOMETRY IS ACTUALLY GEOMETRY
# ════════════════════════════════════════════════════════════════

def test_hump_radius_matches_closed_form():
    """R = c^2/(8h) + h/2 for a circular segment."""
    expected = HUMP_CHORD_M ** 2 / (8 * HUMP_HEIGHT_M) + HUMP_HEIGHT_M / 2
    assert hump_radius_m() == pytest.approx(expected)
    # 3.7m x 0.10m hump -> about a 17m radius arc
    assert 17.0 < hump_radius_m() < 17.3


def test_belly_clearance_is_the_sagitta():
    """Clearance needed = R - sqrt(R^2 - (wb/2)^2), plus the safety margin."""
    wb = 2.45
    R = hump_radius_m()
    sag_mm = (R - np.sqrt(R ** 2 - (wb / 2) ** 2)) * 1000
    assert belly_clearance_required_mm(wb) == pytest.approx(sag_mm + CLEARANCE_MARGIN_MM)


def test_longer_wheelbase_needs_more_clearance():
    """A longer car straddling the same hump has to lift its belly further.
    This is why long-wheelbase cars ground out on speed breakers."""
    short = belly_clearance_required_mm(2.40)
    long = belly_clearance_required_mm(2.80)
    assert long > short


def test_hump_ramp_angle_is_shallow_but_real():
    """A standard hump face is about 6 degrees. Any overhang with a shallower
    approach angle than that will strike it."""
    a = hump_ramp_angle_deg()
    assert 5.0 < a < 8.0


def test_approach_angle_falls_as_the_overhang_grows():
    """A splitter that reaches further forward at the same height has a worse
    approach angle. This is the whole reason splitters scrape."""
    assert approach_angle_deg(0.6, 0.12) > approach_angle_deg(0.9, 0.12)


@pytest.mark.parametrize("key", CARS)
def test_every_car_has_a_wheelbase(key):
    """The speed-breaker check is only meaningful with a real wheelbase."""
    assert INDIAN_CARS[key].get("wheelbase_m", 0) > 2.0


@pytest.mark.parametrize("key", CARS)
def test_stock_cars_clear_a_standard_speed_breaker(key):
    """
    Every car in the database, unmodified, should clear a standard hump.
    If a stock production car fails this, the hump model is wrong.
    """
    car = INDIAN_CARS[key]
    params, _ = get_car_params(key)
    clearance_mm = params["underbody_clearance_norm"] * params["height_m"] * 1000
    needed = belly_clearance_required_mm(car["wheelbase_m"])
    assert clearance_mm > needed, \
        (f"{key}: stock clearance {clearance_mm:.0f}mm < {needed:.0f}mm needed. "
         f"A production car that cannot cross a speed breaker means the model is wrong.")


# ════════════════════════════════════════════════════════════════
# THE STATUTORY POSITION IS SURFACED, NOT BURIED
# ════════════════════════════════════════════════════════════════

def test_legal_citations_are_marked_unverified():
    """
    The Motor Vehicles Act reading must NOT claim to be verified.

    An earlier version of this project asserted a 120mm ground-clearance minimum
    sourced to "CMVR Rule 95(1)" — a rule number that could not be checked
    against a primary source. Inventing a statute to justify a constraint is
    exactly what a physics-first tool must not do. If someone later marks this
    VERIFIED, they must have actually read the Act.
    """
    assert MVA_SECTION_52.status is VerificationStatus.NEEDS_VERIFICATION
    assert "s.52" in MVA_SECTION_52.source


def test_external_body_mods_require_approval():
    """Anything that changes the external body is flagged for RTO endorsement."""
    for name in ["front_splitter", "side_skirts", "rear_spoiler", "rear_diffuser"]:
        legality, _ = MOD_LEGALITY[name]
        assert legality is Legality.NEEDS_RTO_APPROVAL, \
            f"{name} alters the external body and must be flagged"


def test_only_concealed_mods_are_unrestricted():
    """
    Wheel covers and the underbody panel are the only two modifications that do
    not alter a particular recorded at registration. This is the finding, and
    the tool must not quietly widen it.
    """
    assert set(legally_unrestricted_mods()) == {"wheel_covers", "underbody_panel"}


@pytest.mark.parametrize("key", CARS)
def test_report_flags_approval_for_a_body_mod(key):
    car = INDIAN_CARS[key]
    params, _ = get_car_params(key)
    rep = check_compliance(car, params, ["rear_spoiler"])
    assert rep.needs_approval, f"{key}: a fabricated spoiler was not flagged"
    assert not rep.road_legal_as_is


@pytest.mark.parametrize("key", CARS)
def test_wheel_covers_are_always_fit_as_is(key):
    """The one modification an owner can fit today, with no paperwork."""
    car = INDIAN_CARS[key]
    params, _ = get_car_params(key)
    rep = check_compliance(car, params, ["wheel_covers"])
    assert rep.road_legal_as_is
    assert not rep.blockers and not rep.needs_approval


# ════════════════════════════════════════════════════════════════
# --legal-only ACTUALLY RESTRICTS THE OPTIMISER
# ════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("key", ["maruti_swift", "tata_nexon", "honda_city"])
def test_legal_only_never_recommends_a_body_mod(key):
    """
    The flag has to be load-bearing, not decorative. Nothing the optimiser
    returns under --legal-only may need RTO approval.
    """
    allowed = set(legally_unrestricted_mods())
    for sol in run_grid_search(key, legal_only=True):
        illegal = set(sol.mod_names) - allowed
        assert not illegal, f"{key}: --legal-only returned {illegal}"


@pytest.mark.parametrize("key", ["maruti_swift", "tata_nexon"])
def test_legal_only_still_finds_a_real_saving(key):
    """
    Restricting to the law must not leave the user with nothing. Wheel covers
    plus a floor panel are both legal AND, as it happens, the best return on
    effort in the catalogue — so the honest answer is still a useful one.
    """
    sols = run_grid_search(key, legal_only=True)
    assert sols, f"{key}: --legal-only found no viable modification at all"
    assert max(s.delta_L_100km for s in sols) > 0.1, \
        f"{key}: the legal options save nothing worth having"


@pytest.mark.parametrize("key", ["maruti_swift", "tata_nexon"])
def test_legal_only_is_a_subset_of_the_full_search(key):
    """Restricting the search cannot invent solutions the full search missed."""
    full = {tuple(sorted(s.mod_names)) for s in run_grid_search(key)}
    legal = {tuple(sorted(s.mod_names)) for s in run_grid_search(key, legal_only=True)}
    assert legal <= full
