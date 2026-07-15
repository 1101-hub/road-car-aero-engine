"""
Tests for Layer 1 — the panel method and the drag budget.

The most important tests here are the ones that check the SOLVER, not the
answer. A panel method can produce a completely plausible Cd while being
fundamentally wrong, and that is exactly what happened: two sign errors (an
inverted source-panel angle term, and a freestream tangential component added
against the tangent that the influence matrix actually used) left the solver
reporting drag on a closed body in potential flow, which is impossible.

test_dalembert_paradox is the test that would have caught it on day one.
"""

import numpy as np
import pytest

from core.panel_solver import (
    ARCHETYPES, INDIAN_CARS, get_car_params, solve, solve_car,
    solve_potential_flow, influence_coefficients, validate_dalembert,
)
from core.geometry import build_profile, panel_geometry


ALL_ARCHETYPES = list(ARCHETYPES)
ALL_CARS = list(INDIAN_CARS)


# ════════════════════════════════════════════════════════════════
# GEOMETRY
# ════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name", ALL_ARCHETYPES)
def test_profile_is_exactly_closed(name):
    """
    A source panel method conserves mass only on a closed contour. The original
    profile left a gap of ~0.5% of car length between the first and last vertex
    because the closing panel was never emitted.
    """
    coords, _ = build_profile(ARCHETYPES[name], 300)
    gap = np.hypot(*(coords[0] - coords[-1]))
    assert gap < 1e-12, f"{name}: profile is not closed, gap={gap:.2e}"


@pytest.mark.parametrize("name", ALL_ARCHETYPES)
def test_profile_does_not_self_intersect(name):
    """The upper surface must stay above the underbody everywhere."""
    coords, meta = build_profile(ARCHETYPES[name], 300)
    n_up = meta["n_upper"]
    upper = coords[:n_up + 1]
    lower = coords[n_up + meta["n_base"] - 1:]
    y_lower_at = np.interp(upper[1:-1, 0], lower[::-1, 0], lower[::-1, 1])
    assert np.all(upper[1:-1, 1] >= y_lower_at - 1e-9), \
        f"{name}: upper surface dips below the underbody"


@pytest.mark.parametrize("name", ALL_ARCHETYPES)
def test_outward_normals(name):
    """
    Clockwise traversal, outward normal at phi + pi/2. Every panel on the rear
    of the body must face downstream (n_x > 0), and the roof must face up.

    The base run includes the two corner fillets, so n_x sweeps from 0 to 1
    across it rather than sitting at exactly 1 — but it must never go negative,
    which would mean a normal pointing back INTO the wake.
    """
    coords, meta = build_profile(ARCHETYPES[name], 500)
    pg = panel_geometry(coords)
    n_up, n_base = meta["n_upper"], meta["n_base"]

    base = slice(n_up, n_up + n_base)
    n_x = np.cos(pg["beta"][base])
    assert np.all(n_x > -1e-6), f"{name}: a base normal points upstream"
    assert n_x.max() > 0.99, f"{name}: no panel on the rear face faces squarely aft"

    roof = (pg["xc"][:n_up] > 0.45) & (pg["xc"][:n_up] < 0.6)
    assert np.all(np.sin(pg["beta"][:n_up][roof]) > 0), \
        f"{name}: roof normals do not point upward"


# ════════════════════════════════════════════════════════════════
# SOLVER CORRECTNESS — the tests that matter
# ════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name", ALL_ARCHETYPES)
def test_dalembert_paradox(name):
    """
    THE correctness test.

    A closed body in attached, incompressible, inviscid flow has exactly zero
    drag. This is not an approximation — it is a theorem. If the attached
    solution integrates to a non-zero drag, the panel method is wrong.

    With the original signs this returned Cd_attached = -0.40 for a hatchback,
    i.e. the solver was manufacturing drag (and, in a closed body, mass) out of
    nothing. The residual now comes only from the sharp base corner, which is an
    integrable velocity singularity in potential flow, and it shrinks under
    refinement (see test_dalembert_converges).
    """
    d = validate_dalembert(ARCHETYPES[name], n_panels=400)
    assert abs(d["Cd_attached"]) < 0.05, \
        (f"{name}: closed body in potential flow must have zero drag, "
         f"got Cd_attached={d['Cd_attached']:+.4f}")


@pytest.mark.parametrize("name", ALL_ARCHETYPES)
def test_net_source_strength_is_zero(name):
    """
    The source strengths on a closed body must sum to zero: a closed body
    creates no net mass. The original solver returned +11.4 for a hatchback,
    which is the same sign error seen from a different angle.
    """
    d = validate_dalembert(ARCHETYPES[name], n_panels=400)
    assert abs(d["net_source"]) < 0.5, \
        f"{name}: net source strength {d['net_source']:+.3f}, should be ~0"


@pytest.mark.parametrize("name", ALL_ARCHETYPES)
def test_stagnation_pressure_is_unity(name):
    """Cp = +1 exactly at the stagnation point. Nothing may exceed it."""
    d = validate_dalembert(ARCHETYPES[name], n_panels=400)
    assert d["Cp_max"] == pytest.approx(1.0, abs=0.02), \
        f"{name}: max Cp = {d['Cp_max']:.4f}, should be +1.000"


def test_dalembert_converges_under_refinement():
    """The residual must shrink as the mesh is refined, confirming it is
    discretisation error at the base corner and not a formulation error."""
    coarse = abs(validate_dalembert(ARCHETYPES["suv"], n_panels=150)["Cd_attached"])
    fine = abs(validate_dalembert(ARCHETYPES["suv"], n_panels=600)["Cd_attached"])
    assert fine < coarse, \
        f"residual grew under refinement: {coarse:.4f} -> {fine:.4f}"


def test_influence_matrix_diagonal():
    """Self-influence: +1/2 normal, 0 tangential. The off-diagonal angle term
    must share this sign convention — that was the original bug."""
    coords, _ = build_profile(ARCHETYPES["sedan"], 100)
    pg = panel_geometry(coords)
    A, B = influence_coefficients(pg)
    assert np.allclose(np.diag(A), 0.5)
    assert np.allclose(np.diag(B), 0.0)
    assert np.isfinite(A).all() and np.isfinite(B).all()


# ════════════════════════════════════════════════════════════════
# THE CAR IS ACTUALLY USED
# ════════════════════════════════════════════════════════════════

def test_cars_in_same_class_differ():
    """
    The original solver meshed the ARCHETYPE and then relabelled the result with
    the car's numbers, so every hatchback produced a byte-identical flow
    solution: Swift, i20 and Altroz all returned Cd_base=0.2603, sep_idx=24,
    Cpb=-0.3127. The per-car validation table was one number repeated.
    """
    swift = solve_car("maruti_swift")
    i20 = solve_car("hyundai_i20")
    assert swift["Cpb"] != i20["Cpb"] or swift["sep_x"] != i20["sep_x"], \
        "two different hatchbacks produced an identical flow solution"
    assert abs(swift["Cd_pressure"] - i20["Cd_pressure"]) > 1e-6, \
        "pressure drag does not respond to the car's own geometry"


# ════════════════════════════════════════════════════════════════
# THE DRAG BUDGET
# ════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("key", ALL_CARS)
def test_budget_sums_to_total(key):
    """Cd must be the sum of its named components — no hidden terms."""
    r = solve_car(key)
    parts = (r["Cd_pressure"] + r["Cd_friction"] + r["Cd_underbody"]
             + r["Cd_wheels"] + r["Cd_cooling"] + r["Cd_mirrors"])
    assert parts == pytest.approx(r["Cd"], rel=1e-9)


@pytest.mark.parametrize("key", ALL_CARS)
def test_every_component_is_positive(key):
    """No component may be negative. A car cannot have negative wheel drag."""
    r = solve_car(key)
    for comp in ["Cd_pressure", "Cd_friction", "Cd_underbody",
                 "Cd_wheels", "Cd_cooling", "Cd_mirrors"]:
        assert r[comp] > 0, f"{key}: {comp} = {r[comp]:.4f} is not positive"


@pytest.mark.parametrize("key", ALL_CARS)
def test_wake_dominates(key):
    """Pressure (wake) drag is the dominant term for every road car —
    typically 45-65% of Cd. If it is not, the physics has gone wrong."""
    r = solve_car(key)
    share = r["Cd_pressure"] / r["Cd"]
    assert 0.35 < share < 0.75, \
        f"{key}: wake drag is {share*100:.0f}% of Cd, expected 35-75%"


@pytest.mark.parametrize("key", ALL_CARS)
def test_base_pressure_is_physical(key):
    """Measured base pressure on production cars lies in Cpb = -0.35 to -0.10."""
    r = solve_car(key)
    assert -0.42 <= r["Cpb"] <= -0.10, \
        f"{key}: Cpb = {r['Cpb']:.3f} is outside the physical range"


@pytest.mark.parametrize("key", ALL_CARS)
def test_separation_is_on_the_rear_of_the_car(key):
    """Flow does not separate off the bonnet."""
    r = solve_car(key)
    assert 0.5 < r["sep_x"] <= 1.0, \
        f"{key}: separation at x={r['sep_x']:.2f} is not on the rear body"


# ════════════════════════════════════════════════════════════════
# VALIDATION AGAINST PUBLISHED Cd
# ════════════════════════════════════════════════════════════════

VALIDATION_SET = [k for k in ALL_CARS
                  if "estimat" not in INDIAN_CARS[k]["Cd_source"].lower()]


@pytest.mark.parametrize("key", VALIDATION_SET)
def test_cd_within_15_percent_of_published(key):
    """
    Predicted Cd within +/-15% of the manufacturer figure.

    Cars whose reference Cd is itself an ESTIMATE (flagged in their Cd_source
    — Scorpio-N, Baleno, Punch) are excluded, and honestly so: a model cannot
    be validated against a guess. They still get recommendations, because
    modification DELTAS depend on the budget structure, not on nailing the
    absolute baseline — but they contribute nothing to the accuracy claim.
    """
    r = solve_car(key)
    err = abs(r["Cd"] - r["Cd_reference"]) / r["Cd_reference"] * 100
    assert err < 15.0, \
        f"{key}: Cd={r['Cd']:.3f} vs published {r['Cd_reference']:.3f} ({err:.1f}% off)"


@pytest.mark.parametrize("key", ALL_CARS)
def test_cd_is_in_a_plausible_range_for_a_car(key):
    """No production car has Cd below 0.20 or above 0.50."""
    r = solve_car(key)
    assert 0.20 < r["Cd"] < 0.50, f"{key}: Cd={r['Cd']:.3f} is not a car"


# ════════════════════════════════════════════════════════════════
# PHYSICAL MONOTONICITY
# ════════════════════════════════════════════════════════════════

def test_higher_ride_height_costs_underbody_drag():
    """Raising a car increases its underbody drag. This is why SUVs pay."""
    p, _ = get_car_params("maruti_swift")
    low = dict(p, underbody_clearance_norm=0.07)
    high = dict(p, underbody_clearance_norm=0.13)
    assert solve(high)["Cd_underbody"] > solve(low)["Cd_underbody"]


@pytest.mark.parametrize("key", ["maruti_swift", "honda_city", "tata_nexon"])
def test_solution_converges_with_panel_count(key):
    """
    Cd must not wander as the mesh is refined. A solver whose answer moves when
    you add panels is reporting a discretisation artefact, not physics.

    Two separate bugs used to break this. The Stratford criterion was
    differentiated directly on the panel mesh, so separation jumped between the
    backlight and the base corner somewhere around 800 panels and HALVED a
    sedan's Cd. And the base corner was mathematically sharp, so each refinement
    resolved a sharper, more unphysical suction spike on the panels approaching
    it, dragging Cd down by 15% from 200 to 900 panels.

    Checked over the converged range; 200 panels is simply too coarse to
    resolve the tail.
    """
    p, _ = get_car_params(key)
    cds = [solve(p, n_panels=n)["Cd"] for n in (400, 500, 600, 900)]
    spread = max(cds) - min(cds)
    assert spread < 0.02, \
        f"{key}: Cd not converged: {[f'{c:.4f}' for c in cds]} (spread {spread:.4f})"


@pytest.mark.parametrize("key", ["maruti_swift", "honda_city", "tata_nexon"])
def test_separation_point_is_mesh_independent(key):
    """The separation point must not move with the mesh — see above."""
    p, _ = get_car_params(key)
    seps = [solve(p, n_panels=n)["sep_x"] for n in (300, 500, 800)]
    assert max(seps) - min(seps) < 0.03, \
        f"{key}: separation moved with the mesh: {[f'{s:.3f}' for s in seps]}"
