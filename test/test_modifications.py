"""
Tests for Layer 2 — modification physics.

The central test here is test_delta_cd_within_published_range. Every
modification's DeltaCd is checked against the range that has actually been
measured in a wind tunnel on a real car. This is the guard that the project
did not have, and its absence is why the headline result was wrong:

    rear diffuser   model said DeltaCd = 0.1183, real cars measure 0.01-0.03
    side skirts     model said DeltaCd = 0.0001, real cars measure 0.005-0.015
    rear spoiler    model said DeltaCd < 0 always, so it was never once selected
    wheel covers    model said DeltaCd = 0.0240, more wheel drag than the car had

The diffuser error was worth 4-10x and it dominated the Pareto frontier for
every car, so the tool's recommendation was an artefact of a bug rather than a
physical result. A test this simple would have caught it immediately.
"""

import numpy as np
import pytest

from core.panel_solver import solve_car, INDIAN_CARS
from core.modifications import (
    make_budget, budget_Cd, apply_mod_set,
    mod_wheel_covers, mod_underbody_panel, mod_rear_diffuser,
    mod_rear_spoiler, mod_front_splitter, mod_side_skirts,
    DIFFUSER_MAX_ANGLE_DEG, SPOILER_STALL_ANGLE_DEG,
    DIFFUSER_MIN_CLEARANCE_MM, SPLITTER_MAX_DEPTH_MM,
)

CARS = list(INDIAN_CARS)

# Published wind-tunnel DeltaCd ranges for a passenger car, at sensible settings.
# Sources: Hucho "Aerodynamics of Road Vehicles" 4th ed.; Cogotti (1983);
# Senior & Zhang SAE 2000-01-0354; SAE 2003-01-0655.
PUBLISHED_RANGE = {
    "wheel_covers":    (0.004, 0.018),
    "underbody_panel": (0.008, 0.045),
    "rear_diffuser":   (0.005, 0.035),
    "front_splitter":  (0.002, 0.020),
    "side_skirts":     (0.002, 0.018),
    "rear_spoiler":    (-0.010, 0.020),   # a spoiler CAN make things worse
}

SENSIBLE_SETTINGS = [
    (mod_wheel_covers,    dict(n_wheels_covered=4)),
    (mod_underbody_panel, dict(coverage_fraction=1.0)),
    (mod_rear_diffuser,   dict(angle_deg=7.0, length_norm=0.14)),
    (mod_front_splitter,  dict(depth_mm=60.0)),
    (mod_side_skirts,     dict(height_mm=50.0)),
    (mod_rear_spoiler,    dict(chord_m=0.25, angle_deg=8.0)),
]


# ════════════════════════════════════════════════════════════════
# THE TEST THAT MATTERS
# ════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("car_key", CARS)
@pytest.mark.parametrize("fn,kwargs", SENSIBLE_SETTINGS)
def test_delta_cd_within_published_range(car_key, fn, kwargs):
    """
    At an aggressive-but-sensible setting, each modification must produce a
    DeltaCd inside the range measured on real cars. A model that claims a
    bolt-on part removes 38% of a car's drag is not optimistic, it is broken.
    """
    b = make_budget(solve_car(car_key))
    r = fn(b, **kwargs)
    if not r.feasible:
        pytest.skip(f"{r.mod_name} not feasible on {car_key}: {r.constraint_hit}")

    lo, hi = PUBLISHED_RANGE[r.mod_name]
    assert lo <= r.delta_Cd <= hi, (
        f"{car_key} / {r.mod_name}: DeltaCd={r.delta_Cd:.4f} is outside the "
        f"published range {lo}..{hi}")


# ════════════════════════════════════════════════════════════════
# NO MODIFICATION MAY OVERDRAW ITS BUDGET
# ════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("car_key", CARS)
def test_wheel_covers_cannot_exceed_wheel_drag(car_key):
    """
    The original model removed Cd 0.024 with wheel covers while the car's entire
    parasitic allowance was 0.022, of which only 0.015 was wheels. Fitting
    covers drove the implied wheel drag negative.
    """
    base = solve_car(car_key)
    b = make_budget(base)
    r = mod_wheel_covers(b, n_wheels_covered=4)
    assert r.delta_Cd < base["Cd_wheels"], "removed more drag than the wheels have"
    assert b["Cd_wheels"] > 0, "wheel drag went negative"


@pytest.mark.parametrize("car_key", CARS)
def test_underbody_mods_cannot_exceed_underbody_drag(car_key):
    """A full floor panel, skirts and a splitter together still cannot remove
    more than the car's underbody drag."""
    base = solve_car(car_key)
    b = make_budget(base)
    for fn, kw in [(mod_underbody_panel, dict(coverage_fraction=1.0)),
                   (mod_side_skirts, dict(height_mm=50.0)),
                   (mod_front_splitter, dict(depth_mm=60.0))]:
        fn(b, **kw)
    assert b["Cd_underbody"] >= 0, "underbody drag went negative"


@pytest.mark.parametrize("car_key", CARS)
def test_all_components_stay_positive_after_everything(car_key):
    """Fit every modification at its most aggressive legal setting. No component
    of the drag budget may go negative, and Cd must stay a plausible car."""
    base = solve_car(car_key)
    ms = apply_mod_set(base, SENSIBLE_SETTINGS)
    for name, val in ms.budget_final.items():
        if name.startswith("Cd_"):
            assert val >= 0, f"{car_key}: {name} = {val:.4f} went negative"
    assert ms.Cd_final > 0.15, \
        (f"{car_key}: Cd fell to {ms.Cd_final:.3f} — lower than the most "
         f"aerodynamic car ever built. The old model drove a Swift to 0.166.")


@pytest.mark.parametrize("car_key", CARS)
def test_total_reduction_is_believable(car_key):
    """
    Every modification fitted at its most aggressive LEGAL setting, all at once.

    This is an upper bound, not a recommendation. Real full aero packages on
    production cars achieve 10-20%, so a ceiling of 26% is the absurdity guard:
    it leaves room for a maximal package while still failing loudly if the
    physics starts manufacturing drag reductions.

    The old model drove a Swift to Cd 0.166 here — lower than the most
    aerodynamic car ever built, from bolt-on parts.
    """
    base = solve_car(car_key)
    ms = apply_mod_set(base, SENSIBLE_SETTINGS)
    pct = ms.delta_Cd_total / base["Cd"] * 100
    assert 0 < pct < 26, \
        f"{car_key}: total reduction {pct:.1f}% is not physically believable"


# ════════════════════════════════════════════════════════════════
# CONSTRAINTS ARE REAL
# ════════════════════════════════════════════════════════════════

def test_diffuser_stalls_past_the_limit():
    b = make_budget(solve_car("tata_nexon"))
    r = mod_rear_diffuser(b, angle_deg=DIFFUSER_MAX_ANGLE_DEG + 1.0)
    assert not r.feasible and r.delta_Cd == 0.0


def test_spoiler_stalls_past_the_limit():
    b = make_budget(solve_car("maruti_swift"))
    r = mod_rear_spoiler(b, angle_deg=SPOILER_STALL_ANGLE_DEG + 1.0)
    assert not r.feasible and r.delta_Cd == 0.0


def test_splitter_respects_structural_limit():
    b = make_budget(solve_car("maruti_swift"))
    r = mod_front_splitter(b, depth_mm=SPLITTER_MAX_DEPTH_MM + 10.0)
    assert not r.feasible and r.delta_Cd == 0.0


def test_diffuser_rejected_below_legal_clearance():
    """The Honda City has 119mm of clearance against the 120mm workable minimum,
    so the diffuser must be refused."""
    base = solve_car("honda_city")
    clearance_mm = (base["params"]["underbody_clearance_norm"]
                    * base["params"]["height_m"] * 1000)
    r = mod_rear_diffuser(make_budget(base), angle_deg=5.0)
    if clearance_mm < DIFFUSER_MIN_CLEARANCE_MM:
        assert not r.feasible
        assert "clearance" in (r.constraint_hit or "").lower()


def test_infeasible_mod_confers_no_synergy():
    """
    THE SYNERGY BUG. The old code granted an 8% "skirt + diffuser" bonus
    whenever a diffuser appeared in the list, without checking that it had been
    fitted. On a Honda City the diffuser is rejected for insufficient ground
    clearance — and the car still collected the synergy for it.

    An unfitted part must contribute exactly nothing.
    """
    base = solve_car("honda_city")

    skirts_only = apply_mod_set(base, [(mod_side_skirts, dict(height_mm=45.0))])
    with_dead_diffuser = apply_mod_set(base, [
        (mod_side_skirts, dict(height_mm=45.0)),
        (mod_rear_diffuser, dict(angle_deg=5.0, length_norm=0.12)),
    ])

    diffuser = next(r for r in with_dead_diffuser.modifications
                    if r.mod_name == "rear_diffuser")
    if diffuser.feasible:
        pytest.skip("diffuser is feasible on this car; bug not exercised")

    assert with_dead_diffuser.delta_Cd_total == pytest.approx(
        skirts_only.delta_Cd_total, rel=1e-9), \
        "an infeasible diffuser changed the result"


# ════════════════════════════════════════════════════════════════
# COMPOSITION
# ════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("car_key", CARS)
def test_shared_component_mods_do_not_double_count(car_key):
    """
    A floor panel and side skirts both draw on underbody drag. Their combined
    gain must be LESS than the sum of their individual gains, because whichever
    is fitted second finds less left to take.
    """
    base = solve_car(car_key)
    panel = apply_mod_set(base, [(mod_underbody_panel, dict(coverage_fraction=1.0))])
    skirts = apply_mod_set(base, [(mod_side_skirts, dict(height_mm=50.0))])
    both = apply_mod_set(base, [
        (mod_underbody_panel, dict(coverage_fraction=1.0)),
        (mod_side_skirts, dict(height_mm=50.0)),
    ])
    assert both.delta_Cd_total < panel.delta_Cd_total + skirts.delta_Cd_total, \
        "two mods on the same drag component added up as if independent"


def test_result_is_independent_of_listed_order():
    """apply_mod_set sorts into physical order, so the caller's ordering must
    not change the answer."""
    base = solve_car("tata_nexon")
    mods = [
        (mod_rear_diffuser, dict(angle_deg=5.0, length_norm=0.12)),
        (mod_wheel_covers, dict(n_wheels_covered=4)),
        (mod_underbody_panel, dict(coverage_fraction=0.7)),
    ]
    a = apply_mod_set(base, mods)
    b = apply_mod_set(base, list(reversed(mods)))
    assert a.Cd_final == pytest.approx(b.Cd_final, rel=1e-9)


@pytest.mark.parametrize("car_key", CARS)
def test_final_cd_equals_sum_of_components(car_key):
    """The budget must stay self-consistent after modification."""
    base = solve_car(car_key)
    ms = apply_mod_set(base, SENSIBLE_SETTINGS)
    parts = sum(v for k, v in ms.budget_final.items() if k.startswith("Cd_"))
    assert parts == pytest.approx(ms.Cd_final, rel=1e-9)


# ════════════════════════════════════════════════════════════════
# MONOTONICITY
# ════════════════════════════════════════════════════════════════

def test_more_coverage_never_saves_less():
    base = solve_car("maruti_swift")
    prev = -1.0
    for cov in [0.2, 0.4, 0.6, 0.8, 1.0]:
        d = mod_underbody_panel(make_budget(base), coverage_fraction=cov).delta_Cd
        assert d >= prev
        prev = d


def test_more_wheels_covered_never_saves_less():
    base = solve_car("maruti_swift")
    prev = -1.0
    for n in [1, 2, 3, 4]:
        d = mod_wheel_covers(make_budget(base), n_wheels_covered=n).delta_Cd
        assert d >= prev
        prev = d


def test_steeper_diffuser_recovers_more_up_to_the_stall():
    base = solve_car("tata_nexon")
    prev = -1.0
    for ang in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]:
        r = mod_rear_diffuser(make_budget(base), angle_deg=ang, length_norm=0.12)
        assert r.feasible and r.delta_Cd >= prev
        prev = r.delta_Cd
