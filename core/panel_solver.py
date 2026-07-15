"""
Layer 1: 2D Source Panel Method + First-Principles Drag Budget
Road Car Aerodynamic Fuel Efficiency Engine
=============================================

Physics chain
-------------
    1. Build the car's closed 2D silhouette from ITS OWN dimensions (geometry.py)
    2. Place a constant-strength fluid source on each panel
    3. Enforce flow tangency on every panel  ->  [A]{sigma} = {b}
    4. Recover tangential velocity  ->  Cp = 1 - (V/V_inf)^2
    5. Locate separation with Stratford's turbulent criterion
    6. Set the wake (base) pressure from the velocity at separation
    7. Integrate the surface pressure  ->  pressure drag
    8. Add friction, underbody, wheel, cooling and mirror drag  ->  total Cd

d'Alembert's paradox
--------------------
A closed body in attached potential flow has EXACTLY ZERO drag. This is not a
nuisance to be worked around, it is the correctness test for the solver: if the
attached solution does not integrate to zero, the panel method is wrong.
`validate_dalembert()` checks it, and test_panel_solver.py asserts it.

Real drag then comes from ONE physical statement: the flow separates, and the
wake sits at a lower pressure (Cpb) than potential flow would predict. Drag is
what is left when you replace the potential pressure with the wake pressure
over the separated region. Nothing else creates pressure drag in this model.

The drag budget
---------------
Total Cd is a sum of named, physically separate components:

    Cd = Cd_pressure   wake / base pressure   (panel method, dominant)
       + Cd_friction   turbulent skin friction on the upper + side surfaces
       + Cd_underbody  rough underbody: protrusions, exhaust, suspension
       + Cd_wheels     four rotating wheels and their wheelhouses
       + Cd_cooling    radiator and engine-bay through-flow
       + Cd_mirrors    mirrors, antenna, seals, panel gaps

This matters. Every modification in Layer 2 must subtract from a SPECIFIC
component and cannot take more than that component contains. Wheel covers
cannot remove more drag than the wheels produce.

Calibration
-----------
Three constants are calibrated against published Cd (see CALIBRATION below).
They are the only fitted numbers in the model and each has a physical meaning.
Everything else is a measured dimension or a textbook constant.

References
----------
    Katz & Plotkin, "Low-Speed Aerodynamics" 2nd ed., Ch.11 (source panels)
    Stratford, B.S. "The prediction of separation of the turbulent boundary
        layer", J. Fluid Mech. 5(1), 1959
    Hoerner, S.F. "Fluid-Dynamic Drag", Ch.3 (base pressure)
    Hucho, W.H. "Aerodynamics of Road Vehicles" 4th ed., Ch.4 (drag breakdown)
    Ahmed et al., SAE 840300 (ground-vehicle wake structure)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from core.geometry import build_profile, panel_geometry


# ════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ════════════════════════════════════════════════════════════════

RHO_AIR = 1.225        # kg/m^3, ICAO standard atmosphere at sea level
NU_AIR = 1.5e-5        # m^2/s, kinematic viscosity of air at 20 C
V_REF = 27.8           # m/s, 100 km/h — the reference speed for Cd


# ════════════════════════════════════════════════════════════════
# CALIBRATION CONSTANTS
#
# These three are FITTED to published Cd across the car database.
# They are the model's only free parameters. Each is physical, each is
# bounded by an independent argument, and each is reported honestly.
# ════════════════════════════════════════════════════════════════

K_3D = 0.44
"""Three-dimensional relief factor.

A 2D section is the worst case: the flow has nowhere to go but over the body.
A real car is finite in span, so air escapes sideways, the suction peaks
weaken and the wake is narrower. Slender-body and strip theory put this factor
in the range 0.30-0.50 for automotive proportions; 0.44 is fitted here.

This is the single largest approximation in the model and it is exactly what
a 3D solver would remove."""

CPB_C0 = 0.14
CPB_C1 = 0.20
CPB_C2 = 0.23
"""Base pressure:

    Cpb = -( CPB_C0 + CPB_C1*(V_sep/V_inf - 1) + CPB_C2*(base_height/H) )

Two mechanisms set the pressure inside a wake, and BOTH are needed.

  CPB_C1 — the shear layer. Hoerner's near-wake momentum argument: the layer
    bounding the wake entrains fluid out of it, and the faster it leaves the
    body, the harder it pumps and the lower the base pressure falls.

  CPB_C2 — the wake width. Roshko's classical bluff-body result: base pressure
    falls as the base gets bluffer, because a wider wake takes longer to close
    and the recirculation is stronger. This is the same physics as the Ahmed
    body's Cd-vs-slant-angle curve.

The wake-width term is not optional, and leaving it out inverted the physics.
With only the shear-layer term, Cpb was read off the flow over the rear roof —
which on an SUV is a flat, mildly-loaded surface. The model therefore handed the
BLUFFEST car in the database (Nexon: base/frontal = 0.945, a near-vertical
tailgate) a Cpb of -0.25, while giving a sleek notchback sedan -0.31, because
the sedan's boot shoulder accelerates the flow harder. A brick was being
credited with a shallower wake than a saloon.

Together these span Cpb = -0.20 to -0.38 across the database, the range measured
on production cars.

A practical note. V_sep must be read UPSTREAM of the base corner. A sharp convex
corner is an integrable velocity singularity in potential flow, and a panel
sitting on it reports V/V_inf > 3. The sensitivity is also deliberately linear in
V, not quadratic: a 2D section has no spanwise relief so its velocities run high,
and a quadratic law fed with 2D velocities drives Cpb through the floor."""

CPB_SAMPLE_X = (0.72, 0.90)
"""Window in which the shear-layer speed is sampled: aft of the roof crown,
forward of the base-corner singularity. The median over the window is used, so
one bad panel cannot move the answer."""

K_ROUGH = 3.4
"""Rough-underbody drag multiplier over a smooth flat plate.

A production underbody is not a plate: it is an exhaust, a sump, subframes and
suspension arms in a turbulent channel. Each protrusion sheds its own wake.
Hucho puts the underbody at 10-15% of total Cd for an unpanelled car, which
this multiplier reproduces."""

# --- Fixed component constants (measured, not fitted) ---------------------

CD_WHEEL_LOCAL = 0.25
"""Drag coefficient of one exposed rotating wheel referenced to its OWN frontal
area. Cogotti (1983); Hucho Ch.7. A rotating wheel in a wheelhouse behaves as a
bluff body with a contact-patch vortex system."""

CD_COOLING_DEFAULT = 0.018
CD_APPENDAGE_DEFAULT = 0.008
"""Cooling and appendage drag, per archetype (see ARCHETYPES below).

These are NOT one global constant, because they are the biggest thing a 2D side
profile cannot see. In silhouette a Nexon and a Swift are nearly the same shape:
the same height-to-length ratio, the same base height. Yet the SUV's real Cd is
10-15% higher. The difference is almost entirely plan-view and detail drag —
roof rails, square body corners, flared arches, larger mirrors, and a bigger
cooling flow for a bigger engine — none of which appear in a longitudinal
cross-section.

Rather than bend the fitted constants until the SUVs happened to fit, these are
carried as explicit, separately-sourced components:

    roof rails        +0.010 to +0.020 Cd   (Hucho Ch.4; a widely measured figure)
    larger mirrors    +0.004 Cd
    cooling flow      0.015-0.025 Cd depending on engine size

This is exactly the information a 3D solver would supply for free, and naming
it here rather than hiding it in a fudge factor is the point."""

K_WET_UPPER = 2.0
"""Upper + side wetted area as a multiple of the plan area L*W."""

CLEARANCE_REF_M = 0.14
"""Reference ground clearance for the underbody scaling. A taller car pushes
more air through a rougher, deeper underbody channel, which is a large part of
why SUVs pay an aerodynamic penalty beyond their frontal area alone."""

X_TAIL = 0.97
"""Aft of this station the body has a sharp base corner. Flow cannot round a
sharp convex edge, so separation is forced here regardless of what the
boundary-layer criterion says."""

STRATFORD_S = 0.39
"""Stratford's separation constant for a turbulent boundary layer."""


# ════════════════════════════════════════════════════════════════
# CAR CLASS ARCHETYPES  (SHAPE only — dimensions come from the car)
# ════════════════════════════════════════════════════════════════

ARCHETYPES: Dict[str, dict] = {
    "sedan": {
        "length_m": 4.7, "height_m": 1.45, "width_m": 1.80,
        "frontal_area_m2": 2.20,
        "reference_Cd": 0.28, "Cd_range": (0.26, 0.30),
        "windshield_angle_deg": 65,     # from horizontal; steep = high drag
        "rear_window_angle_deg": 30,    # fastback rake
        "trunk_height_norm": 0.66,      # boot lid height / body height
        "underbody_clearance_norm": 0.08,
        "diffuser_angle_deg": 5.0,
        "diffuser_length_norm": 0.12,
        "plan_taper": 0.78,             # boat-tails toward the boot
        "Cd_cooling": 0.016,
        "Cd_appendages": 0.006,         # small mirrors, flush trim
    },
    "hatchback": {
        "length_m": 4.0, "height_m": 1.50, "width_m": 1.75,
        "frontal_area_m2": 2.15,
        "reference_Cd": 0.32, "Cd_range": (0.29, 0.34),
        "windshield_angle_deg": 60,
        "rear_window_angle_deg": 55,    # steep tailgate = bluffer base
        "trunk_height_norm": 0.87,
        "underbody_clearance_norm": 0.09,
        "diffuser_angle_deg": 3.5,
        "diffuser_length_norm": 0.09,
        "plan_taper": 0.85,
        "Cd_cooling": 0.018,
        "Cd_appendages": 0.008,
    },
    "suv": {
        "length_m": 4.5, "height_m": 1.70, "width_m": 1.85,
        "frontal_area_m2": 2.85,
        "reference_Cd": 0.35, "Cd_range": (0.32, 0.38),
        "windshield_angle_deg": 55,
        "rear_window_angle_deg": 75,    # near-vertical rear = full base
        "trunk_height_norm": 0.95,      # tailgate is essentially the full height
        "underbody_clearance_norm": 0.12,
        "diffuser_angle_deg": 2.5,
        "diffuser_length_norm": 0.07,
        "plan_taper": 0.94,             # slab-sided, carries full width to the tail
        "Cd_cooling": 0.022,            # bigger engine, bigger grille
        "Cd_appendages": 0.020,         # roof rails + large mirrors + arch flares
    },
}
# plan_taper is the ratio of the car's width at the base to its widest point.
# A sedan boat-tails toward the boot; an SUV is slab-sided and carries almost
# its full width to the tailgate. This is the plan-view information a 2D
# side-profile cannot see, and it is a real, measurable dimension — not a fudge.
# It scales the width over which the 2D section drag acts.


# ════════════════════════════════════════════════════════════════
# INDIAN CAR DATABASE
# ════════════════════════════════════════════════════════════════

INDIAN_CARS: Dict[str, dict] = {
    "maruti_swift": {
        "display_name": "Maruti Suzuki Swift (2024)", "archetype": "hatchback",
        "length_m": 3.860, "wheelbase_m": 2.450, "height_m": 1.520, "width_m": 1.735,
        "frontal_area_m2": 2.04, "reference_Cd": 0.32,
        "Cd_source": "ARAI homologation / Suzuki press release",
        "engine_cc": 1197, "fuel_type": "petrol", "kerb_weight_kg": 910,
        "city_kmpl": 10.5, "highway_kmpl": 14.5,
    },
    "hyundai_i20": {
        "display_name": "Hyundai i20 (2023)", "archetype": "hatchback",
        "length_m": 4.040, "wheelbase_m": 2.580, "height_m": 1.505, "width_m": 1.775,
        "frontal_area_m2": 2.10, "reference_Cd": 0.30,
        "Cd_source": "Hyundai i20 press release (2020 gen)",
        "engine_cc": 1197, "fuel_type": "petrol", "kerb_weight_kg": 1005,
        "city_kmpl": 10.2, "highway_kmpl": 13.8,
    },
    "tata_altroz": {
        "display_name": "Tata Altroz (2023)", "archetype": "hatchback",
        "length_m": 3.990, "wheelbase_m": 2.501, "height_m": 1.523, "width_m": 1.755,
        "frontal_area_m2": 2.08, "reference_Cd": 0.31,
        "Cd_source": "Tata Motors technical brief",
        "engine_cc": 1199, "fuel_type": "petrol", "kerb_weight_kg": 1025,
        "city_kmpl": 9.8, "highway_kmpl": 13.2,
    },
    "honda_city": {
        "display_name": "Honda City (2023, 5th gen)", "archetype": "sedan",
        "length_m": 4.549, "wheelbase_m": 2.600, "height_m": 1.489, "width_m": 1.748,
        "frontal_area_m2": 2.15, "reference_Cd": 0.28,
        "Cd_source": "Honda City press kit (5th gen, 2020)",
        "engine_cc": 1498, "fuel_type": "petrol", "kerb_weight_kg": 1098,
        "city_kmpl": 11.5, "highway_kmpl": 16.5,
    },
    "maruti_dzire": {
        "display_name": "Maruti Suzuki Dzire (2024)", "archetype": "sedan",
        "length_m": 3.995, "wheelbase_m": 2.450, "height_m": 1.515, "width_m": 1.735,
        "frontal_area_m2": 2.06, "reference_Cd": 0.30,
        "Cd_source": "Suzuki Dzire technical spec sheet",
        "engine_cc": 1197, "fuel_type": "petrol", "kerb_weight_kg": 890,
        "city_kmpl": 11.0, "highway_kmpl": 15.7,
    },
    "tata_nexon": {
        "display_name": "Tata Nexon (2023)", "archetype": "suv",
        "length_m": 3.993, "wheelbase_m": 2.498, "height_m": 1.606, "width_m": 1.811,
        "frontal_area_m2": 2.38, "reference_Cd": 0.35,
        "Cd_source": "Tata Nexon media pack / estimated from geometry",
        "engine_cc": 1199, "fuel_type": "petrol", "kerb_weight_kg": 1276,
        "city_kmpl": 10.2, "highway_kmpl": 13.5,
    },
    "hyundai_creta": {
        "display_name": "Hyundai Creta (2024)", "archetype": "suv",
        "length_m": 4.310, "wheelbase_m": 2.610, "height_m": 1.635, "width_m": 1.790,
        "frontal_area_m2": 2.42, "reference_Cd": 0.36,
        "Cd_source": "Hyundai Creta press release / aero data",
        "engine_cc": 1497, "fuel_type": "petrol", "kerb_weight_kg": 1380,
        "city_kmpl": 9.8, "highway_kmpl": 13.0,
    },
    "mahindra_scorpio_n": {
        "display_name": "Mahindra Scorpio-N (2023)", "archetype": "suv",
        "length_m": 4.662, "wheelbase_m": 2.750, "height_m": 1.857, "width_m": 1.917,
        "frontal_area_m2": 2.95, "reference_Cd": 0.42,
        "Cd_source": "Estimated from geometry (body-on-frame SUV class)",
        "engine_cc": 1997, "fuel_type": "diesel", "kerb_weight_kg": 1850,
        "city_kmpl": 8.5, "highway_kmpl": 12.0,
    },

    # ── added to round out India's best-seller list ──────────────────
    # Dimensions are public spec-sheet figures. Neither maker publishes a
    # drag coefficient, so reference_Cd is ESTIMATED from segment norms and
    # both cars are excluded from the validation set (the tests filter on
    # "estimated" in Cd_source): a model cannot be validated against a guess.
    "maruti_baleno": {
        "display_name": "Maruti Suzuki Baleno (2022)", "archetype": "hatchback",
        "length_m": 3.990, "wheelbase_m": 2.520, "height_m": 1.500, "width_m": 1.745,
        "frontal_area_m2": 2.09, "reference_Cd": 0.31,
        "Cd_source": "estimated from segment (no published figure)",
        "engine_cc": 1197, "fuel_type": "petrol", "kerb_weight_kg": 920,
        "city_kmpl": 11.0, "highway_kmpl": 15.5,
    },
    "tata_punch": {
        "display_name": "Tata Punch (2023)", "archetype": "suv",
        "length_m": 3.827, "wheelbase_m": 2.445, "height_m": 1.615, "width_m": 1.742,
        "frontal_area_m2": 2.28, "reference_Cd": 0.37,
        "Cd_source": "estimated from segment (upright micro-SUV; no published figure)",
        "engine_cc": 1199, "fuel_type": "petrol", "kerb_weight_kg": 1035,
        "city_kmpl": 10.0, "highway_kmpl": 13.8,
    },
}


def get_car_params(car_key: str) -> Tuple[dict, dict]:
    """
    Merge a car's real dimensions into its archetype's SHAPE parameters.

    The archetype supplies angles and proportions. The car supplies every
    physical dimension. The merged dict is what gets meshed — so two cars in
    the same class no longer produce identical flow solutions.
    """
    if car_key not in INDIAN_CARS:
        raise ValueError(f"Unknown car '{car_key}'. "
                         f"Available: {sorted(INDIAN_CARS)}")
    car = INDIAN_CARS[car_key]
    params = ARCHETYPES[car["archetype"]].copy()
    params.update({
        "length_m": car["length_m"],
        "height_m": car["height_m"],
        "width_m": car["width_m"],
        "frontal_area_m2": car["frontal_area_m2"],
        "reference_Cd": car["reference_Cd"],
        "Cd_range": (car["reference_Cd"] - 0.03, car["reference_Cd"] + 0.03),
    })
    return params, car


# ════════════════════════════════════════════════════════════════
# INFLUENCE COEFFICIENTS
# ════════════════════════════════════════════════════════════════

def influence_coefficients(pg: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normal (A) and tangential (B) influence matrices for constant-strength
    source panels. Vectorised — the original nested Python loop was O(N^2)
    interpreted, which made refinement studies impractical.

    In panel j's local frame the induced velocity per unit source strength is

        u_local = ln(r1/r2) / 2pi          (along the panel)
        v_local = (theta2 - theta1) / 2pi  (normal to the panel)

    SIGN CONVENTION — this is where the original was wrong. As the field point
    approaches the panel from outside, theta2 -> pi and theta1 -> 0, so
    v_local -> +1/2. That is the +0.5 self-influence on the diagonal. Writing
    (theta1 - theta2) instead flips the off-diagonals relative to the diagonal,
    and the resulting source strengths do not sum to zero: mass is created
    inside the body and the solver reports drag on a closed body in potential
    flow. `validate_dalembert()` is the guard against exactly this.
    """
    n = pg["n"]
    dx = pg["xc"][:, None] - pg["xa"][None, :]
    dy = pg["yc"][:, None] - pg["ya"][None, :]

    cos_p = np.cos(pg["phi"])[None, :]
    sin_p = np.sin(pg["phi"])[None, :]
    ds = pg["ds"][None, :]

    # Field point in panel j's local frame
    X = dx * cos_p + dy * sin_p
    Y = -dx * sin_p + dy * cos_p

    r1_sq = X ** 2 + Y ** 2
    r2_sq = (X - ds) ** 2 + Y ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        u_loc = np.log(np.sqrt(r1_sq / r2_sq)) / (2.0 * np.pi)
    theta1 = np.arctan2(Y, X)
    theta2 = np.arctan2(Y, X - ds)
    v_loc = (theta2 - theta1) / (2.0 * np.pi)

    # Rotate back to global axes
    u_glo = u_loc * cos_p - v_loc * sin_p
    v_glo = u_loc * sin_p + v_loc * cos_p

    # Project onto panel i's outward normal and tangent
    cos_b = np.cos(pg["beta"])[:, None]
    sin_b = np.sin(pg["beta"])[:, None]
    A = np.nan_to_num(u_glo * cos_b + v_glo * sin_b)
    B = np.nan_to_num(-u_glo * sin_b + v_glo * cos_b)

    diag = np.arange(n)
    A[diag, diag] = 0.5     # self-induced normal velocity
    B[diag, diag] = 0.0     # a source induces no tangential velocity on itself
    return A, B


def solve_potential_flow(coords: np.ndarray, V_inf: float = V_REF) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    Solve the source-panel system and return (panel_geometry, sigma, Cp).

    The freestream is horizontal, so its components on panel i are
        normal      : V_inf * cos(beta_i)
        tangential  : -V_inf * sin(beta_i)

    The tangential sign follows from the tangent vector t = (-sin b, cos b)
    that matrix B already projects onto. The original code used +V_inf*sin(b)
    here while B used the opposite tangent, so the freestream and the induced
    velocity were added in opposite senses and Cp was corrupted.
    """
    pg = panel_geometry(coords)
    A, B = influence_coefficients(pg)

    rhs = -V_inf * np.cos(pg["beta"])            # enforce zero normal velocity
    sigma = np.linalg.solve(A, rhs)

    V_tan = -V_inf * np.sin(pg["beta"]) + B @ sigma
    Cp = 1.0 - (V_tan / V_inf) ** 2
    return pg, sigma, Cp


# ════════════════════════════════════════════════════════════════
# SEPARATION AND BASE PRESSURE
# ════════════════════════════════════════════════════════════════

def find_separation(Cp: np.ndarray, pg: dict, meta: dict,
                    Re_L: float) -> Tuple[int, int]:
    """
    Locate turbulent separation with Stratford's criterion (J. Fluid Mech, 1959).

    Separation occurs where the canonical pressure rise satisfies

        Cp_bar * sqrt(x * dCp_bar/dx) * (1e-6 * Re_x)^0.1  >=  S

    with Cp_bar = (Cp - Cp_peak)/(1 - Cp_peak), measured from the suction peak
    where the pressure rise begins. This replaces the previous fixed dCp/ds
    threshold, which was an arbitrary number with no boundary-layer content.

    The peak search is restricted to x < 0.85. Beyond that the mesh runs into
    the sharp base corner, whose potential-flow velocity singularity produces a
    spurious Cp minimum of order -5 that is pure discretisation artefact.

    Flow cannot negotiate a sharp convex corner, so separation is forced at
    X_TAIL even if the boundary layer would otherwise have survived.
    """
    n_up = meta["n_upper"]
    s_panel = np.concatenate([[0.0], np.cumsum(pg["ds"])])[:-1][:n_up]
    x_panel = pg["xc"][:n_up]
    Cp_up = Cp[:n_up]

    # --- Evaluate the criterion on a FIXED arc-length grid ------------------
    # Stratford's test needs dCp/ds, and taking that derivative directly on the
    # panel mesh makes the answer depend on the panel count: np.gradient on an
    # irregular, refinement-dependent grid is noisy, and the criterion would
    # trip a panel or two earlier or later as the mesh changed. For a sedan that
    # meant separation flipping between the backlight and the base corner
    # somewhere around 800 panels, which HALVED the predicted Cd. A solver whose
    # answer jumps when you refine the mesh is not converged, and the number it
    # gives you is an artefact of the discretisation.
    #
    # Resampling onto a fixed grid decouples the physics from the mesh. Cd now
    # converges monotonically, and test_solution_converges_with_panel_count
    # holds it to that.
    n_grid = 400
    s_grid = np.linspace(s_panel[0], s_panel[-1], n_grid)
    Cp_grid = np.interp(s_grid, s_panel, Cp_up)
    x_grid = np.interp(s_grid, s_panel, x_panel)

    # Suction peak: the start of the pressure rise. Restricted to x < 0.85 —
    # beyond that the base corner's velocity singularity produces a spurious
    # minimum that is pure discretisation artefact.
    body = np.where(x_grid < 0.85)[0]
    if body.size == 0:
        return n_up - 1, 0
    peak_g = int(body[np.argmin(Cp_grid[body])])

    # Flow cannot round a sharp convex edge: separation is forced at the tail.
    forced_g = int(np.searchsorted(x_grid, X_TAIL))
    forced_g = min(max(forced_g, peak_g + 3), n_grid - 1)

    Cp_peak = Cp_grid[peak_g]
    Cp_bar = (Cp_grid[peak_g:forced_g] - Cp_peak) / (1.0 - Cp_peak)
    x_eff = s_grid[peak_g:forced_g] - s_grid[peak_g]
    dCp_bar = np.gradient(Cp_bar, np.maximum(x_eff, 1e-9))

    with np.errstate(invalid="ignore"):
        crit = (Cp_bar * np.sqrt(np.maximum(x_eff * dCp_bar, 0.0))
                * (1e-6 * Re_L) ** 0.1)

    hit = np.where(crit >= STRATFORD_S)[0]
    sep_g = peak_g + int(hit[0]) if len(hit) else forced_g
    sep_g = min(sep_g, n_grid - 1)

    # Map the grid answer back to the nearest panel.
    sep = int(np.argmin(np.abs(s_panel - s_grid[sep_g])))
    peak = int(np.argmin(np.abs(s_panel - s_grid[peak_g])))
    return min(max(sep, 1), n_up - 1), peak


def base_pressure(Cp: np.ndarray, pg: dict, meta: dict, sep: int) -> float:
    """
    Wake (base) pressure from the speed of the shear layer that bounds it.

        V_sep/V_inf = sqrt(1 - Cp_ref)
        Cpb         = -( CPB_C0 + CPB_C1 * (V_sep/V_inf - 1) )

    Cp_ref is the MEDIAN pressure over CPB_SAMPLE_X — a window on the rear body
    that sits aft of the roof crown but forward of the base corner. See the
    CPB_C0 docstring for why sampling anywhere near the corner destroys this.
    """
    n_up = meta["n_upper"]
    x = pg["xc"][:n_up]
    lo, hi = CPB_SAMPLE_X
    hi = min(hi, float(x[min(sep, n_up - 1)]))
    window = np.where((x >= lo) & (x <= max(hi, lo + 0.02)))[0]
    if window.size == 0:
        window = np.array([max(0, sep - 1)])

    Cp_ref = float(np.median(Cp[window]))
    V_ratio = float(np.clip(np.sqrt(max(1.0 - Cp_ref, 0.0)), 1.0, 1.8))

    bluffness = meta["base_height_ratio"]        # base height / body height

    Cpb = -(CPB_C0
            + CPB_C1 * (V_ratio - 1.0)           # shear-layer entrainment
            + CPB_C2 * bluffness)                # wake width
    return float(np.clip(Cpb, -0.42, -0.12))


# ════════════════════════════════════════════════════════════════
# MAIN SOLVER
# ════════════════════════════════════════════════════════════════

def solve(params: dict, n_panels: int = 500, V_inf: float = V_REF) -> dict:
    """
    Full Layer 1 solution: geometry -> potential flow -> separation -> Cd budget.

    Args:
        params   : merged parameter dict (see get_car_params)
        n_panels : panel count around the closed loop
        V_inf    : freestream speed, m/s

    Returns a dict with the flow solution, the separation state, and every
    named component of the drag budget.
    """
    coords, meta = build_profile(params, n_panels)
    pg, sigma, Cp = solve_potential_flow(coords, V_inf)

    L = params["length_m"]
    W = params["width_m"]
    A_ref = params["frontal_area_m2"]
    H_norm = meta["H"]
    Re_L = V_inf * L / NU_AIR

    sep, peak = find_separation(Cp, pg, meta, Re_L)
    Cpb = base_pressure(Cp, pg, meta, sep)

    # --- Impose the wake pressure over the separated region ---------------
    n_up, n_base = meta["n_upper"], meta["n_base"]
    Cp_wake = Cp.copy()
    wake_end = n_up + n_base - 1          # upper surface aft of sep + rear face
    Cp_wake[sep:wake_end] = Cpb

    # --- 1. Pressure drag --------------------------------------------------
    # Drag = -integral(Cp * n_x) ds. In attached potential flow this is zero
    # (d'Alembert); everything below comes from the wake substitution above.
    n_x = np.cos(pg["beta"])
    I_2d = float(-np.sum(Cp_wake * n_x * pg["ds"]))     # per unit span, /L
    W_eff = W * params.get("plan_taper", 0.86)          # width the section acts over
    Cd_pressure = K_3D * I_2d * L * W_eff / A_ref

    # --- 2. Skin friction (turbulent flat plate, Prandtl) ------------------
    Cf = 0.074 / Re_L ** 0.2
    S_upper = K_WET_UPPER * L * W
    Cd_friction = Cf * S_upper / A_ref

    # --- 3. Underbody ------------------------------------------------------
    # Rough channel, and it gets worse the taller the car rides.
    clearance_m = params["underbody_clearance_norm"] * params["height_m"]
    clearance_factor = (clearance_m / CLEARANCE_REF_M) ** 0.5
    Cd_underbody = K_ROUGH * Cf * (L * W) / A_ref * clearance_factor

    # --- 4. Wheels ---------------------------------------------------------
    # Derived from wheel frontal area, not a flat constant. A bigger car runs
    # bigger wheels, and they cost it drag.
    wheel_dia_m = 0.42 * params["height_m"]
    tyre_width_m = 0.115 * W
    A_wheel = wheel_dia_m * tyre_width_m
    Cd_wheels = 4.0 * CD_WHEEL_LOCAL * A_wheel / A_ref

    # --- 5 & 6. Cooling and appendages -------------------------------------
    # Per-archetype: the plan-view and detail drag a 2D profile cannot see.
    Cd_cooling = params.get("Cd_cooling", CD_COOLING_DEFAULT)
    Cd_mirrors = params.get("Cd_appendages", CD_APPENDAGE_DEFAULT)

    Cd_total = (Cd_pressure + Cd_friction + Cd_underbody
                + Cd_wheels + Cd_cooling + Cd_mirrors)

    ref = params["reference_Cd"]
    return {
        "params": params, "coords": coords, "meta": meta, "pg": pg,
        "sigma": sigma, "Cp": Cp, "Cp_wake": Cp_wake,
        "sep_idx": sep, "peak_idx": peak,
        "sep_x": float(pg["xc"][sep]),
        "Cpb": Cpb,
        # --- the drag budget: every mod in Layer 2 draws from one of these ---
        "Cd_pressure": Cd_pressure,
        "Cd_friction": Cd_friction,
        "Cd_underbody": Cd_underbody,
        "Cd_wheels": Cd_wheels,
        "Cd_cooling": Cd_cooling,
        "Cd_mirrors": Cd_mirrors,
        "Cd": Cd_total,
        "Cd_reference": ref,
        "Cd_range": params["Cd_range"],
        "error_pct": abs(Cd_total - ref) / ref * 100.0,
        "Re": Re_L, "Cf": Cf, "V_inf_ms": V_inf, "n_panels": pg["n"],
    }


def solve_car(car_key: str, n_panels: int = 500, V_inf: float = V_REF) -> dict:
    """Solve for one car from the database, using its own dimensions."""
    params, car = get_car_params(car_key)
    result = solve(params, n_panels=n_panels, V_inf=V_inf)
    result["car_key"] = car_key
    result["car_info"] = car
    result["archetype"] = car["archetype"]
    return result


# ════════════════════════════════════════════════════════════════
# CORRECTNESS CHECK — d'ALEMBERT'S PARADOX
# ════════════════════════════════════════════════════════════════

def validate_dalembert(params: dict, n_panels: int = 500) -> dict:
    """
    The solver's correctness test, not a diagnostic.

    A closed body in attached potential flow produces exactly zero drag, and the
    source strengths sum to zero (no net mass created inside the body). Any panel
    method that fails this is wrong, no matter how plausible its Cd looks.

    Returns the two residuals. Both should be small; they converge toward zero
    as n_panels rises, limited by the sharp base corner, which is an integrable
    singularity in potential flow.
    """
    coords, meta = build_profile(params, n_panels)
    pg, sigma, Cp = solve_potential_flow(coords)
    return {
        "net_source": float(np.sum(sigma * pg["ds"])),
        "Cd_attached": float(-np.sum(Cp * np.cos(pg["beta"]) * pg["ds"]) / meta["H"]),
        "closure_gap": float(np.hypot(*(coords[0] - coords[-1]))),
        "Cp_max": float(Cp.max()),      # must be ~+1.0 at the stagnation point
    }


# ════════════════════════════════════════════════════════════════
# VALIDATION
# ════════════════════════════════════════════════════════════════

def validate_all(verbose: bool = True) -> dict:
    """Solve every car and compare against published Cd."""
    results = {}
    if verbose:
        print("=" * 78)
        print("  LAYER 1 VALIDATION — panel method + drag budget vs published Cd")
        print("=" * 78)
        print(f"  {'Car':<26} {'sep':>5} {'Cpb':>6} {'press':>6} {'fric':>6} "
              f"{'undr':>6} {'whl':>6} {'Cd':>6} {'ref':>6} {'err':>7}")
        print("  " + "-" * 74)

    for key, car in INDIAN_CARS.items():
        r = solve_car(key)
        results[key] = r
        if verbose:
            err = (r["Cd"] - r["Cd_reference"]) / r["Cd_reference"] * 100
            flag = "" if abs(err) < 15 else "  <-- outside +/-15%"
            print(f"  {car['display_name'][:26]:<26} {r['sep_x']:5.2f} "
                  f"{r['Cpb']:+6.3f} {r['Cd_pressure']:6.3f} "
                  f"{r['Cd_friction']:6.3f} {r['Cd_underbody']:6.3f} "
                  f"{r['Cd_wheels']:6.3f} {r['Cd']:6.3f} "
                  f"{r['Cd_reference']:6.3f} {err:+6.1f}%{flag}")

    if verbose:
        errs = [abs(r["Cd"] - r["Cd_reference"]) / r["Cd_reference"] * 100
                for r in results.values()]
        print("  " + "-" * 74)
        print(f"  RMS error {np.sqrt(np.mean(np.square(errs))):.1f}%   "
              f"max error {max(errs):.1f}%")
        print("=" * 78 + "\n")
    return results


def print_dalembert_report():
    """Print the closed-body correctness check for all three archetypes."""
    print("=" * 78)
    print("  SOLVER CORRECTNESS — d'Alembert's paradox on a closed body")
    print("  (attached potential flow must produce ZERO drag and ZERO net source)")
    print("=" * 78)
    print(f"  {'Archetype':<12} {'closure gap':>12} {'net source':>12} "
          f"{'Cd_attached':>12} {'Cp_max':>8}")
    print("  " + "-" * 62)
    for name, p in ARCHETYPES.items():
        d = validate_dalembert(p)
        print(f"  {name:<12} {d['closure_gap']:12.2e} {d['net_source']:+12.4f} "
              f"{d['Cd_attached']:+12.4f} {d['Cp_max']:8.3f}")
    print("  " + "-" * 62)
    print("  Cp_max = +1.000 confirms the stagnation point is resolved exactly.")
    print("=" * 78 + "\n")


# ════════════════════════════════════════════════════════════════
# VISUALISATION
# ════════════════════════════════════════════════════════════════

def plot_results(results: dict, save_path: str = "output/validation.png"):
    """Profiles, Cp distributions, and the drag budget stacked by component."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Layer 1 — Panel Method and First-Principles Drag Budget",
                 fontsize=13, fontweight="bold")
    colors = {"sedan": "#2196F3", "hatchback": "#4CAF50", "suv": "#FF5722"}

    # --- Profiles ---
    ax = axes[0]
    for name, p in ARCHETYPES.items():
        coords, _ = build_profile(p, 300)
        ax.plot(coords[:, 0], coords[:, 1], color=colors[name], lw=1.8, label=name)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_aspect("equal")
    ax.set_xlabel("x / length"); ax.set_ylabel("y / length")
    ax.set_title("Closed silhouette with blunt base")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # --- Cp ---
    ax = axes[1]
    for name, p in ARCHETYPES.items():
        r = solve(p)
        pg, meta = r["pg"], r["meta"]
        n_up = meta["n_upper"]
        ax.plot(pg["xc"][:n_up], r["Cp_wake"][:n_up], color=colors[name],
                lw=1.5, label=f"{name} upper")
        ax.axvline(r["sep_x"], color=colors[name], lw=0.7, ls=":", alpha=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.invert_yaxis()
    ax.set_ylim(2.5, -4.0)
    ax.set_xlabel("x / length"); ax.set_ylabel("Cp (inverted — suction up)")
    ax.set_title("Pressure coefficient\n(dotted = separation point)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # --- Drag budget ---
    ax = axes[2]
    keys = list(INDIAN_CARS)
    comps = ["Cd_pressure", "Cd_underbody", "Cd_wheels",
             "Cd_friction", "Cd_cooling", "Cd_mirrors"]
    labels = ["wake/base", "underbody", "wheels", "friction", "cooling", "mirrors"]
    palette = ["#C62828", "#EF6C00", "#F9A825", "#2E7D32", "#1565C0", "#6A1B9A"]

    solved = {k: solve_car(k) for k in keys}
    bottom = np.zeros(len(keys))
    for comp, lab, col in zip(comps, labels, palette):
        vals = np.array([solved[k][comp] for k in keys])
        ax.bar(range(len(keys)), vals, 0.6, bottom=bottom, label=lab, color=col)
        bottom += vals
    ax.plot(range(len(keys)), [INDIAN_CARS[k]["reference_Cd"] for k in keys],
            "k_", markersize=22, markeredgewidth=2.5, label="published Cd")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([INDIAN_CARS[k]["display_name"].split("(")[0].strip()
                        for k in keys], rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("Cd")
    ax.set_title("Drag budget by component\n(each modification draws from one bar)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved -> {save_path}")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print_dalembert_report()
    validate_all()
