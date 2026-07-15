"""
Layer 2: Aerodynamic Modification Physics
Road Car Aerodynamic Fuel Efficiency Engine
============================================

The rule this layer obeys
-------------------------
Every modification must subtract drag from a SPECIFIC, NAMED component of the
Layer 1 budget, and it can never take more than that component contains.

    wheel covers      ->  Cd_wheels
    underbody panel   ->  Cd_underbody
    side skirts       ->  Cd_underbody
    front splitter    ->  Cd_underbody  (+ a small stagnation term)
    rear diffuser     ->  Cd_pressure   (by raising the wake pressure Cpb)
    rear spoiler      ->  Cd_pressure   (by raising Cpb, minus its own drag)

This is the discipline the previous version lacked. It computed Layer 1 drag as
base + friction + parasitic, then had the modifications subtract from six other
buckets that did not exist in that budget. Wheel covers removed Cd 0.024 from a
parasitic allowance of 0.022, of which only 0.015 was wheels — fit the covers
and the car's implied wheel drag went negative. A rear diffuser returned
Cd 0.118, four to ten times the largest figure ever measured on a road car, and
because it dominated everything else it won the optimiser for every car in the
database. The headline results of the project were an artefact of that bug.

Now a modification physically cannot remove drag that the car does not have.

State threading
---------------
apply_mod_set() carries a live budget. Each modification sees the budget left
by the ones before it and writes back what it consumed. Two mods that both act
on the wake pressure therefore compound correctly instead of both claiming the
same improvement against the same untouched Cpb — which is what the old
sequential loop did, because it reset Cpb from the pristine baseline on every
iteration and only carried Cd forward.

Every DeltaCd below is cross-checked against published wind-tunnel ranges in
test/test_modifications.py. If a model drifts outside what has actually been
measured on a real car, the test suite fails.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

from core.panel_solver import (solve_car, INDIAN_CARS, K_ROUGH, K_3D,
                               CD_WHEEL_LOCAL)


# ════════════════════════════════════════════════════════════════
# HARD PHYSICAL AND LEGAL LIMITS
# ════════════════════════════════════════════════════════════════

SPOILER_STALL_ANGLE_DEG = 15.0
"""Above this the spoiler stalls: the flow separates off the spoiler itself and
it becomes a drag device. Katz, "Race Car Aerodynamics" (1995)."""

DIFFUSER_MAX_ANGLE_DEG = 7.0
"""Above this the flow separates inside the diffuser and pressure recovery
collapses to nothing. Senior & Zhang, SAE 2000-01-0354."""

DIFFUSER_MIN_CLEARANCE_MM = 120.0
"""Minimum ground clearance for a diffuser to be viable on Indian roads.

NOT a statute. An earlier version of this file sourced this number to
"CMVR Rule 95(1)" — a rule number that could not be verified against the primary
source. A physics tool that invents a statute to justify a constraint is doing
the exact thing this project claims not to do, so the citation is gone.

The constraint itself survives, because it can be DERIVED. A standard Indian
speed breaker is a 3.7 m x 0.10 m round-topped hump; a car straddling it must
lift its belly by the sagitta of that arc over half a wheelbase, which is about
45 mm, plus margin for suspension travel and for the many humps built to no
standard at all. See core/compliance.py, which computes this per car from its
wheelbase — geometry needs no citation to be true.

120 mm is the working minimum below which a diffuser has nowhere to go.
The STATUTORY position (Motor Vehicles Act s.52 — external body alterations
require RTO endorsement) is handled separately in compliance.py, because it
applies to the modification regardless of how much clearance the car has."""

SPLITTER_MAX_DEPTH_MM = 80.0
"""Structural limit for an unsupported splitter in road use."""

SKIRT_MIN_GROUND_GAP_MM = 50.0
"""Minimum skirt-to-ground gap: below this, road debris and speed breakers."""


# ════════════════════════════════════════════════════════════════
# MODIFICATION EFFECTIVENESS CONSTANTS
#   Each is a fraction of a real, computed drag component.
# ════════════════════════════════════════════════════════════════

WHEEL_COVER_EFFECTIVENESS = 0.17
"""A flush disc cover removes the spoke turbulence and the ventilation
(pumping) drag of an open wheel, but not the wheel's bluff-body form drag or
its contact-patch vortex. Cogotti (1983); consistent with the ~0.01 Cd gain
measured on aero wheel covers for production EVs."""

DIFFUSER_EFFICIENCY = 0.55
DIFFUSER_EFFICIENCY_MAX = 0.75
"""Fraction of the ideal Bernoulli pressure recovery a real diffuser achieves.
The rest is lost to boundary-layer growth on the diffuser ramp.

The MAX is a hard ceiling on the sum of the base value and the skirt/floor
bonuses. Without it, a diffuser fitted behind both skirts and a flat floor
climbed to eta = 0.81 — a better pressure recovery than a purpose-built
wind-tunnel diffuser, on a road car with 190 mm of ground clearance."""

DIFFUSER_ETA_SKIRT_BONUS = 0.10
"""Skirts stop high-pressure side air spilling into the diffuser and destroying
its spanwise pressure gradient, so the diffuser recovers more."""

DIFFUSER_ETA_PANEL_BONUS = 0.06
"""A smooth floor delivers a thinner boundary layer into the diffuser throat,
which delays separation on the ramp and raises recovery."""

SPOILER_TURNING_GAIN = 1.8
"""Rise in Cpb per unit of (effective spoiler height / base height). The spoiler
fixes the separation line and turns the shear layer inward, narrowing the wake.
Hucho Fig 5.45."""

SPOILER_PROFILE_CD = 0.35
"""Drag coefficient of the spoiler's own projected frontal area. It is not a
bluff plate in clean freestream — it sits in the decelerated flow at the roof
trailing edge, which is why a well-sized lip can be a net win."""

SPLITTER_UNDERBODY_GAIN = 0.25
"""Underbody drag removed per unit of (splitter depth / ground clearance): the
splitter deflects flow around the car instead of under it."""

SPLITTER_STAG_CD = 0.004
"""Front-face pressure benefit from lowering the stagnation point, at the
reference depth of 50 mm. Hucho Fig 5.21."""

SKIRT_SEAL_GAIN = 0.60
"""Underbody drag removed per unit of (skirt height / ground clearance), capped
below. Skirts seal the lateral pressure leak that drives the underbody edge
vortices."""
SKIRT_SEAL_MAX = 0.35
"""A skirt can never remove more than this share of underbody drag — it does
nothing about the exhaust, sump or suspension in the middle of the floor."""


# ════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════

@dataclass
class ModResult:
    """Result of applying one modification. delta_Cd > 0 means drag went DOWN."""
    mod_name: str
    params_used: dict
    delta_Cd: float
    delta_Cd_pct: float
    component: str                  # which budget line this drew from
    constraint_hit: Optional[str]
    explanation: str
    Cd_new: float
    feasible: bool


@dataclass
class ModSet:
    modifications: List[ModResult]
    delta_Cd_total: float
    Cd_final: float
    interaction_note: str
    budget_final: dict = field(default_factory=dict)


def _infeasible(name, params, reason, Cd0) -> ModResult:
    return ModResult(mod_name=name, params_used=params, delta_Cd=0.0,
                     delta_Cd_pct=0.0, component="-", constraint_hit=reason,
                     explanation=f"Not fitted: {reason}", Cd_new=Cd0,
                     feasible=False)


def make_budget(baseline: dict) -> dict:
    """
    Extract the mutable drag budget from a Layer 1 solution.

    This dict is what the modifications consume. Cd is always the sum of its
    components, so it cannot drift away from them.
    """
    return {
        "Cd_pressure":  baseline["Cd_pressure"],
        "Cd_friction":  baseline["Cd_friction"],
        "Cd_underbody": baseline["Cd_underbody"],
        "Cd_wheels":    baseline["Cd_wheels"],
        "Cd_cooling":   baseline["Cd_cooling"],
        "Cd_mirrors":   baseline["Cd_mirrors"],
        "Cpb":          baseline["Cpb"],
        # The pristine base suction, so wake-acting mods can apply
        # diminishing returns against what is actually left.
        "Cpb_baseline": baseline["Cpb"],
        "params":       baseline["params"],
        "meta":         baseline["meta"],
        "V_inf_ms":     baseline["V_inf_ms"],
        # Set by mods that change the diffuser's operating environment.
        "diffuser_eta_bonus": 0.0,
    }


def budget_Cd(b: dict) -> float:
    """Total Cd is the sum of the components. Always."""
    return (b["Cd_pressure"] + b["Cd_friction"] + b["Cd_underbody"]
            + b["Cd_wheels"] + b["Cd_cooling"] + b["Cd_mirrors"])


def _base_sensitivity(b: dict) -> float:
    """
    dCd_pressure / dCpb — how much total drag falls per unit rise in wake pressure.

    The rear face contributes -Cpb * base_height to the surface pressure
    integral, so differentiating the Layer 1 pressure-drag expression

        Cd_pressure = K_3D * (-integral Cp n_x ds) * L * W_eff / A_ref

    with respect to Cpb over the base panels gives the factor below. Any mod
    that raises Cpb converts through this number — the diffuser and the spoiler
    both use it, so they share one consistent route into the drag budget.
    """
    p, m = b["params"], b["meta"]
    W_eff = p["width_m"] * p.get("plan_taper", 0.86)
    base_h_norm = m["base_height"]            # already normalised by length
    return K_3D * base_h_norm * p["length_m"] * W_eff / p["frontal_area_m2"]


def _suction_remaining(b: dict) -> float:
    """
    Fraction of the ORIGINAL base suction still left to recover, in [0, 1].

    Diminishing returns, and they are physical. A diffuser and a spoiler both
    work by raising the wake pressure toward ambient. Once the diffuser has
    already recovered part of that suction, the spoiler arrives to find less
    left to take — it cannot recover pressure that is no longer missing.

    Without this, the two mods each computed their full ideal DeltaCpb against a
    constant sensitivity and simply added. Stacking every modification at its
    most aggressive legal setting then removed 31% of an SUV's drag, when a real
    full aero package on a production car achieves 10-20%.
    """
    cpb0 = b.get("Cpb_baseline")
    if not cpb0:
        return 1.0
    return float(np.clip(b["Cpb"] / cpb0, 0.0, 1.0))


# ════════════════════════════════════════════════════════════════
# A. WHEEL COVERS  ->  Cd_wheels
# ════════════════════════════════════════════════════════════════

def mod_wheel_covers(b: dict, n_wheels_covered: int = 4) -> ModResult:
    """
    Flush disc covers over the wheel faces.

    An open wheel loses energy three ways: bluff-body form drag, the vortex
    system at the rotating contact patch, and the centrifugal pumping of air
    out through the spokes. A cover kills the third and part of the first. It
    cannot touch the contact patch, which is why the ceiling is a fraction of
    wheel drag, not all of it.

    Draws from Cd_wheels, which Layer 1 computed from the car's actual wheel
    frontal area. A car with small wheels has less to gain, and the model now
    knows that.
    """
    Cd0 = budget_Cd(b)
    n = int(np.clip(n_wheels_covered, 0, 4))
    if n == 0:
        return _infeasible("wheel_covers", dict(n_wheels_covered=0),
                           "No wheels covered", Cd0)

    available = b["Cd_wheels"]
    delta = available * WHEEL_COVER_EFFECTIVENESS * (n / 4.0)

    b["Cd_wheels"] = available - delta
    Cd_new = budget_Cd(b)

    return ModResult(
        mod_name="wheel_covers",
        params_used=dict(n_wheels_covered=n),
        delta_Cd=delta, delta_Cd_pct=delta / Cd0 * 100,
        component="Cd_wheels",
        constraint_hit=None,
        explanation=(
            f"{n} flush wheel cover(s). The car's four wheels contribute "
            f"Cd={available:.4f}, computed from their frontal area "
            f"(Cd_local={CD_WHEEL_LOCAL} per wheel). Covers remove the spoke "
            f"turbulence and ventilation drag — "
            f"{WHEEL_COVER_EFFECTIVENESS*100:.0f}% of wheel drag — but not the "
            f"contact-patch vortex. DeltaCd={delta:.4f} "
            f"({delta/Cd0*100:.1f}% of baseline)."),
        Cd_new=Cd_new, feasible=True)


# ════════════════════════════════════════════════════════════════
# B. UNDERBODY PANEL  ->  Cd_underbody
# ════════════════════════════════════════════════════════════════

def mod_underbody_panel(b: dict, coverage_fraction: float = 0.70) -> ModResult:
    """
    Flat panels covering the exhaust, sump, subframes and suspension.

    Layer 1 modelled the underbody as a rough channel with K_ROUGH times the
    drag of a smooth flat plate. A panel does one thing: it turns the covered
    fraction back into a smooth flat plate. So the drag removed is exactly the
    roughness excess over the covered area,

        DeltaCd = Cd_underbody * coverage * (1 - 1/K_ROUGH)

    which is derived from the Layer 1 model rather than asserted. Full coverage
    cannot remove all underbody drag, because a smooth floor still has skin
    friction. The old model asserted underbody drag was "12% of total Cd" — a
    number with no connection to the car's actual floor.
    """
    Cd0 = budget_Cd(b)
    cov = float(np.clip(coverage_fraction, 0.0, 1.0))
    if cov <= 0:
        return _infeasible("underbody_panel", dict(coverage_fraction=cov),
                           "Zero coverage", Cd0)

    available = b["Cd_underbody"]
    roughness_excess = 1.0 - 1.0 / K_ROUGH        # share of underbody drag that
    delta = available * cov * roughness_excess    # is roughness, not friction

    b["Cd_underbody"] = available - delta
    Cd_new = budget_Cd(b)

    return ModResult(
        mod_name="underbody_panel",
        params_used=dict(coverage_fraction=cov),
        delta_Cd=delta, delta_Cd_pct=delta / Cd0 * 100,
        component="Cd_underbody",
        constraint_hit=None,
        explanation=(
            f"Flat panel over {cov*100:.0f}% of the floor. The rough underbody "
            f"carries Cd={available:.4f}, of which "
            f"{roughness_excess*100:.0f}% is roughness excess over a smooth "
            f"plate (K_ROUGH={K_ROUGH}); the remainder is irreducible skin "
            f"friction that a panel cannot remove. DeltaCd={delta:.4f} "
            f"({delta/Cd0*100:.1f}%). No moving parts, no angle to tune — the "
            f"highest return on effort of any modification here."),
        Cd_new=Cd_new, feasible=True)


# ════════════════════════════════════════════════════════════════
# C. REAR DIFFUSER  ->  Cd_pressure (via Cpb)
# ════════════════════════════════════════════════════════════════

def mod_rear_diffuser(b: dict, angle_deg: float = 5.0,
                      length_norm: float = 0.12) -> ModResult:
    """
    Diverging ramp at the rear of the underbody.

    Physics, in order:
      1. Continuity + Bernoulli in the diverging duct. Area ratio AR gives an
         ideal static-pressure recovery of  Cp_rec = 1 - 1/AR^2.
      2. Real recovery is a fraction of that, lost to boundary-layer growth
         on the ramp (DIFFUSER_EFFICIENCY).
      3. THE TERM THE OLD MODEL MISSED. That recovered pressure only reaches
         the part of the base that the underbody stream actually feeds — a
         band of height h_exit at the bottom of the rear face. The rest of the
         base is fed by the flow off the roof and the sides, which the diffuser
         never touches. So the improvement in base pressure is weighted by
         f_base = h_exit / base_height, which is typically 0.15-0.25.

         The old code applied the full duct recovery to the entire base and
         got DeltaCd = 0.118 — a 38% drag reduction from one bolt-on part,
         which is why it won every optimisation in the project.
      4. The resulting rise in Cpb converts to drag through _base_sensitivity().
    """
    Cd0 = budget_Cd(b)
    p, m = b["params"], b["meta"]
    params_used = dict(angle_deg=angle_deg, length_norm=length_norm)

    clearance_m = p["underbody_clearance_norm"] * p["height_m"]
    clearance_mm = clearance_m * 1000.0

    if angle_deg > DIFFUSER_MAX_ANGLE_DEG:
        return _infeasible("rear_diffuser", params_used,
                           f"Angle {angle_deg} deg exceeds the "
                           f"{DIFFUSER_MAX_ANGLE_DEG} deg separation limit — "
                           f"flow detaches inside the diffuser and recovery "
                           f"collapses", Cd0)
    if clearance_mm < DIFFUSER_MIN_CLEARANCE_MM:
        return _infeasible("rear_diffuser", params_used,
                           f"Ground clearance {clearance_mm:.0f}mm is below the "
                           f"workable minimum ({DIFFUSER_MIN_CLEARANCE_MM:.0f}mm) "
                           f"— a diffuser here would ground on speed breakers",
                           Cd0)

    # 1. Duct geometry
    L_diff = length_norm * p["length_m"]
    h_exit = clearance_m + L_diff * np.tan(np.radians(angle_deg))
    AR = h_exit / clearance_m

    # 2. Ideal recovery, degraded by boundary-layer losses
    Cp_rec_ideal = 1.0 - (1.0 / AR) ** 2
    eta = DIFFUSER_EFFICIENCY + b.get("diffuser_eta_bonus", 0.0)
    eta = float(np.clip(eta, 0.0, DIFFUSER_EFFICIENCY_MAX))
    Cp_rec = Cp_rec_ideal * eta

    # 3. Only the underbody-fed band of the base sees that recovery
    base_height_m = m["base_height"] * p["length_m"]
    f_base = float(np.clip(h_exit / base_height_m, 0.0, 1.0))

    delta_Cpb = Cp_rec * f_base * _suction_remaining(b)

    # 4. Convert to drag
    sens = _base_sensitivity(b)
    delta = delta_Cpb * sens

    # A diffuser cannot push the wake above ambient.
    Cpb_new = min(b["Cpb"] + delta_Cpb, -0.02)
    delta = min(delta, b["Cd_pressure"])

    b["Cpb"] = Cpb_new
    b["Cd_pressure"] -= delta
    Cd_new = budget_Cd(b)

    constraint = (f"Approaching the {DIFFUSER_MAX_ANGLE_DEG} deg separation limit"
                  if angle_deg > 5.5 else None)

    return ModResult(
        mod_name="rear_diffuser",
        params_used=params_used,
        delta_Cd=delta, delta_Cd_pct=delta / Cd0 * 100,
        component="Cd_pressure",
        constraint_hit=constraint,
        explanation=(
            f"Diffuser {angle_deg:.1f} deg over {L_diff*1000:.0f}mm. Area ratio "
            f"AR={AR:.2f} gives an ideal recovery of {Cp_rec_ideal:.3f}, times "
            f"{eta:.2f} efficiency = {Cp_rec:.3f}. That recovery only reaches "
            f"the {h_exit*1000:.0f}mm band of the {base_height_m*1000:.0f}mm "
            f"base fed by the underbody stream, so it is weighted by "
            f"f_base={f_base:.2f}. Wake pressure rises "
            f"Cpb={b['Cpb']-delta_Cpb:+.3f} -> {Cpb_new:+.3f}, giving "
            f"DeltaCd={delta:.4f} ({delta/Cd0*100:.1f}%)."),
        Cd_new=Cd_new, feasible=True)


# ════════════════════════════════════════════════════════════════
# D. REAR SPOILER  ->  Cd_pressure (via Cpb), minus its own drag
# ════════════════════════════════════════════════════════════════

def mod_rear_spoiler(b: dict, chord_m: float = 0.20, angle_deg: float = 7.0,
                     span_fraction: float = 0.85) -> ModResult:
    """
    Lip spoiler at the roof / tailgate trailing edge.

    Benefit: it pins the separation line at a known place and turns the shear
    layer inward, so the wake closes faster and the base pressure rises. The
    effective turning height is chord*sin(angle) and what matters is that height
    relative to the base it is trying to close.

    Cost: the spoiler puts its own projected area into the flow.

    The net can go either way, and that is the physically correct answer — a
    spoiler helps a bluff-backed hatchback and hurts an already-clean shape.
    The old model got this wrong in a different way: it charged the spoiler the
    INDUCED drag of a lifting wing (Cl = 2*pi*sin(alpha)), which made the net
    negative for every car and every setting, so no spoiler was ever selected
    anywhere in the project.
    """
    Cd0 = budget_Cd(b)
    p, m = b["params"], b["meta"]
    params_used = dict(chord_m=chord_m, angle_deg=angle_deg,
                       span_fraction=span_fraction)

    if angle_deg > SPOILER_STALL_ANGLE_DEG:
        return _infeasible("rear_spoiler", params_used,
                           f"Angle {angle_deg} deg is past the "
                           f"{SPOILER_STALL_ANGLE_DEG} deg stall limit — the "
                           f"spoiler separates and becomes a drag device", Cd0)
    if chord_m <= 0 or span_fraction <= 0:
        return _infeasible("rear_spoiler", params_used,
                           "Non-positive geometry", Cd0)

    base_height_m = m["base_height"] * p["length_m"]
    h_eff = chord_m * np.sin(np.radians(angle_deg))     # flow-turning height

    # Benefit: base pressure rises with turning height relative to base height
    delta_Cpb = (SPOILER_TURNING_GAIN * (h_eff / base_height_m) * span_fraction
                 * _suction_remaining(b))
    sens = _base_sensitivity(b)
    gain = delta_Cpb * sens

    # Cost: the spoiler's own projected frontal area
    A_proj = h_eff * p["width_m"] * span_fraction
    cost = SPOILER_PROFILE_CD * A_proj / p["frontal_area_m2"]

    delta = gain - cost

    Cpb_new = min(b["Cpb"] + delta_Cpb, -0.02)
    b["Cpb"] = Cpb_new
    # Net effect lands on the pressure budget; a negative delta raises it.
    b["Cd_pressure"] = max(b["Cd_pressure"] - delta, 0.0)
    Cd_new = budget_Cd(b)

    constraint = (f"Approaching the {SPOILER_STALL_ANGLE_DEG} deg stall limit"
                  if angle_deg > 12 else None)

    verdict = "net gain" if delta > 0 else "NET LOSS — this spoiler adds drag"
    return ModResult(
        mod_name="rear_spoiler",
        params_used=params_used,
        delta_Cd=delta, delta_Cd_pct=delta / Cd0 * 100,
        component="Cd_pressure",
        constraint_hit=constraint,
        explanation=(
            f"Lip spoiler, {chord_m*1000:.0f}mm chord at {angle_deg:.1f} deg. "
            f"Effective turning height {h_eff*1000:.0f}mm against a "
            f"{base_height_m*1000:.0f}mm base raises the wake pressure by "
            f"{delta_Cpb:.3f}, worth DeltaCd={gain:.4f}. Its own projected area "
            f"({A_proj*1e4:.0f} cm^2) costs {cost:.4f}. Net {delta:+.4f} "
            f"({delta/Cd0*100:+.1f}%) — {verdict}."),
        Cd_new=Cd_new, feasible=True)


# ════════════════════════════════════════════════════════════════
# E. FRONT SPLITTER  ->  Cd_underbody (+ stagnation term)
# ════════════════════════════════════════════════════════════════

def mod_front_splitter(b: dict, depth_mm: float = 50.0,
                       height_mm: float = 40.0) -> ModResult:
    """
    Horizontal blade projecting forward at the bottom of the front bumper.

    It lowers the front stagnation point and deflects air around the car rather
    than under it. Less mass flow through the underbody channel means less
    underbody drag — so the benefit is proportional to how much underbody drag
    the car has left, and it shrinks if a floor panel has already been fitted.
    A small extra term comes from the reduced pressure on the bumper's lower face.
    """
    Cd0 = budget_Cd(b)
    p = b["params"]
    params_used = dict(depth_mm=depth_mm, height_mm=height_mm)

    if depth_mm > SPLITTER_MAX_DEPTH_MM:
        return _infeasible("front_splitter", params_used,
                           f"Depth {depth_mm:.0f}mm exceeds the "
                           f"{SPLITTER_MAX_DEPTH_MM:.0f}mm structural limit for "
                           f"road use", Cd0)
    if depth_mm <= 0:
        return _infeasible("front_splitter", params_used, "Zero depth", Cd0)

    clearance_mm = p["underbody_clearance_norm"] * p["height_m"] * 1000.0

    # Flow diverted around rather than under, as a share of underbody drag
    effectiveness = float(np.clip(
        SPLITTER_UNDERBODY_GAIN * (depth_mm / clearance_mm), 0.0, 0.30))
    available = b["Cd_underbody"]
    delta_under = available * effectiveness

    # Stagnation-face benefit, scaled from the 50 mm reference depth
    delta_stag = SPLITTER_STAG_CD * (depth_mm / 50.0) ** 0.6

    delta = delta_under + delta_stag

    b["Cd_underbody"] = available - delta_under
    b["Cd_pressure"] = max(b["Cd_pressure"] - delta_stag, 0.0)
    Cd_new = budget_Cd(b)

    constraint = ("Approaching the structural limit for road use"
                  if depth_mm > 60 else None)

    return ModResult(
        mod_name="front_splitter",
        params_used=params_used,
        delta_Cd=delta, delta_Cd_pct=delta / Cd0 * 100,
        component="Cd_underbody",
        constraint_hit=constraint,
        explanation=(
            f"Splitter projecting {depth_mm:.0f}mm forward against a "
            f"{clearance_mm:.0f}mm ride height. It diverts "
            f"{effectiveness*100:.0f}% of the underbody flow around the car, "
            f"taking DeltaCd={delta_under:.4f} off the remaining underbody drag "
            f"of {available:.4f}. Lowering the stagnation point on the bumper "
            f"face adds {delta_stag:.4f}. Total DeltaCd={delta:.4f} "
            f"({delta/Cd0*100:.1f}%)."),
        Cd_new=Cd_new, feasible=True)


# ════════════════════════════════════════════════════════════════
# F. SIDE SKIRTS  ->  Cd_underbody
# ════════════════════════════════════════════════════════════════

def mod_side_skirts(b: dict, height_mm: float = 50.0,
                    coverage_fraction: float = 0.75) -> ModResult:
    """
    Skirts sealing the gap between the sill and the road.

    The car's flanks sit near stagnation pressure; the underbody is low
    pressure. That difference drives a continuous spanwise leak under the sills,
    which rolls up into a pair of edge vortices and thickens the underbody
    boundary layer. Skirts seal the leak.

    A skirt only fixes the EDGES of the floor. It does nothing about the
    exhaust and suspension down the middle, so its share of underbody drag is
    capped. The old model tried to compute this as lifting-line induced drag
    from vortex circulation and produced DeltaCd = 0.0001 — a hundred times too
    small, making side skirts a no-op that the optimiser correctly never chose.
    """
    Cd0 = budget_Cd(b)
    p = b["params"]
    params_used = dict(height_mm=height_mm, coverage_fraction=coverage_fraction)

    clearance_mm = p["underbody_clearance_norm"] * p["height_m"] * 1000.0
    ground_gap_mm = clearance_mm - height_mm

    if ground_gap_mm < SKIRT_MIN_GROUND_GAP_MM:
        return _infeasible("side_skirts", params_used,
                           f"A {height_mm:.0f}mm skirt leaves only "
                           f"{ground_gap_mm:.0f}mm to the road, below the "
                           f"{SKIRT_MIN_GROUND_GAP_MM:.0f}mm debris minimum",
                           Cd0)
    if height_mm <= 0:
        return _infeasible("side_skirts", params_used, "Zero height", Cd0)

    seal = float(np.clip(SKIRT_SEAL_GAIN * (height_mm / clearance_mm),
                         0.0, SKIRT_SEAL_MAX))
    available = b["Cd_underbody"]
    delta = available * seal * coverage_fraction

    b["Cd_underbody"] = available - delta
    # Skirts also protect the diffuser's spanwise pressure gradient.
    b["diffuser_eta_bonus"] = b.get("diffuser_eta_bonus", 0.0) + DIFFUSER_ETA_SKIRT_BONUS
    Cd_new = budget_Cd(b)

    return ModResult(
        mod_name="side_skirts",
        params_used=params_used,
        delta_Cd=delta, delta_Cd_pct=delta / Cd0 * 100,
        component="Cd_underbody",
        constraint_hit=None,
        explanation=(
            f"Skirts {height_mm:.0f}mm deep over {coverage_fraction*100:.0f}% of "
            f"the wheelbase, leaving {ground_gap_mm:.0f}mm to the road. They "
            f"seal the pressure leak between the flanks and the floor, killing "
            f"the underbody edge vortices: {seal*100:.0f}% of the remaining "
            f"underbody drag ({available:.4f}). DeltaCd={delta:.4f} "
            f"({delta/Cd0*100:.1f}%)."),
        Cd_new=Cd_new, feasible=True)


# ════════════════════════════════════════════════════════════════
# COMBINING MODIFICATIONS
# ════════════════════════════════════════════════════════════════

# Order matters physically: flow-conditioning mods (skirts, floor) must be
# applied before the diffuser, because they change the air the diffuser is
# given. Sorting here makes the result independent of the order the caller
# happened to list them in.
_APPLY_ORDER = ["wheel_covers", "underbody_panel", "side_skirts",
                "front_splitter", "rear_diffuser", "rear_spoiler"]

_MOD_NAME = {
    "mod_wheel_covers": "wheel_covers",
    "mod_underbody_panel": "underbody_panel",
    "mod_side_skirts": "side_skirts",
    "mod_front_splitter": "front_splitter",
    "mod_rear_diffuser": "rear_diffuser",
    "mod_rear_spoiler": "rear_spoiler",
}


def apply_mod_set(baseline: dict, mod_list: list) -> ModSet:
    """
    Apply several modifications to one car and return the combined result.

    The budget is threaded through every modification. Each one sees what the
    previous ones left behind, so:

      * two mods drawing on the same component cannot both spend it
      * two mods raising the wake pressure compound on the running Cpb rather
        than both claiming credit against the pristine baseline
      * a mod that is INFEASIBLE consumes nothing and confers nothing

    That last point was a real bug: the old code granted an 8% "skirt + diffuser
    synergy" bonus whenever a diffuser appeared in the list, without checking
    that the diffuser had actually been fitted. On a Honda City, where the
    diffuser is rejected for having 119 mm of ground clearance against a 120 mm
    legal minimum, the car still collected the synergy for it.

    Interactions are now physical rather than a bonus multiplier: skirts and a
    floor panel raise the DIFFUSER'S EFFICIENCY, and that improved efficiency
    then flows through the diffuser's own Bernoulli calculation.
    """
    b = make_budget(baseline)
    Cd_start = budget_Cd(b)

    ordered = sorted(
        mod_list,
        key=lambda fk: _APPLY_ORDER.index(_MOD_NAME.get(fk[0].__name__, "rear_spoiler")))

    results = []
    for fn, kwargs in ordered:
        results.append(fn(b, **(kwargs or {})))

    Cd_final = budget_Cd(b)
    total = Cd_start - Cd_final

    fitted = {r.mod_name for r in results if r.feasible}
    notes = []
    if "side_skirts" in fitted and "rear_diffuser" in fitted:
        notes.append(
            f"Skirts seal the floor, so the diffuser sees a clean spanwise "
            f"pressure field: its efficiency rises by "
            f"{DIFFUSER_ETA_SKIRT_BONUS:.2f} and that is already inside the "
            f"diffuser's recovery figure above.")
    if "underbody_panel" in fitted and "rear_diffuser" in fitted:
        notes.append(
            f"The floor panel delivers a thinner boundary layer into the "
            f"diffuser throat (+{DIFFUSER_ETA_PANEL_BONUS:.2f} efficiency).")
    if "underbody_panel" in fitted and "side_skirts" in fitted:
        notes.append(
            "Panel and skirts both draw on underbody drag, so their gains do "
            "not add: whichever is fitted second finds less left to take.")

    return ModSet(
        modifications=results,
        delta_Cd_total=total,
        Cd_final=Cd_final,
        interaction_note=(" ".join(notes) if notes
                          else "No significant interaction between these modifications."),
        budget_final={k: v for k, v in b.items()
                      if k.startswith("Cd_") or k == "Cpb"},
    )


# The panel-panel interaction above is handled by state threading, but the
# diffuser needs the floor bonus set BEFORE it runs. side_skirts sets its own;
# underbody_panel sets its bonus here so ordering stays declarative.
_orig_panel = mod_underbody_panel


def mod_underbody_panel(b: dict, coverage_fraction: float = 0.70) -> ModResult:  # noqa: F811
    r = _orig_panel(b, coverage_fraction=coverage_fraction)
    if r.feasible:
        b["diffuser_eta_bonus"] = (b.get("diffuser_eta_bonus", 0.0)
                                   + DIFFUSER_ETA_PANEL_BONUS * coverage_fraction)
    return r


mod_underbody_panel.__doc__ = _orig_panel.__doc__


# ════════════════════════════════════════════════════════════════
# PRINTING
# ════════════════════════════════════════════════════════════════

def print_mod_result(r: ModResult, width: int = 76):
    status = "FEASIBLE" if r.feasible else "NOT FITTED"
    print(f"\n  [{r.mod_name.upper().replace('_', ' ')}]  {status}")
    if r.feasible:
        print(f"    DeltaCd = {r.delta_Cd:+.4f}  ({r.delta_Cd_pct:+.2f}%)   "
              f"from {r.component}")
        print(f"    Cd after: {r.Cd_new:.4f}")
    if r.constraint_hit:
        print(f"    Constraint: {r.constraint_hit}")
    line = "    "
    for word in r.explanation.split():
        if len(line) + len(word) > width:
            print(line)
            line = "    " + word + " "
        else:
            line += word + " "
    if line.strip():
        print(line)


def print_mod_set(ms: ModSet, baseline_Cd: float):
    print("\n" + "=" * 72)
    print("  MODIFICATION SET")
    print("=" * 72)
    for r in ms.modifications:
        flag = "+" if r.feasible else "x"
        if r.feasible:
            print(f"  {flag} {r.mod_name:<18} DeltaCd={r.delta_Cd:+.4f} "
                  f"({r.delta_Cd_pct:+5.2f}%)  <- {r.component}")
        else:
            print(f"  {flag} {r.mod_name:<18} not fitted: {r.constraint_hit}")
    print("  " + "-" * 68)
    print(f"  Baseline Cd : {baseline_Cd:.4f}")
    print(f"  Final Cd    : {ms.Cd_final:.4f}")
    print(f"  Total       : DeltaCd = {ms.delta_Cd_total:+.4f}  "
          f"({ms.delta_Cd_total / baseline_Cd * 100:.1f}% of baseline)")
    print(f"\n  Interaction: {ms.interaction_note}")
    print("=" * 72)


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    for car_key in ["maruti_swift", "tata_nexon"]:
        base = solve_car(car_key)
        print(f"\n{'=' * 72}")
        print(f"  {INDIAN_CARS[car_key]['display_name']}   baseline Cd = {base['Cd']:.4f}")
        print(f"{'=' * 72}")
        for fn, kw in [
            (mod_wheel_covers, dict(n_wheels_covered=4)),
            (mod_underbody_panel, dict(coverage_fraction=0.70)),
            (mod_side_skirts, dict(height_mm=50.0)),
            (mod_front_splitter, dict(depth_mm=50.0)),
            (mod_rear_diffuser, dict(angle_deg=5.0, length_norm=0.12)),
            (mod_rear_spoiler, dict(chord_m=0.20, angle_deg=7.0)),
        ]:
            print_mod_result(fn(make_budget(base), **kw))

        ms = apply_mod_set(base, [
            (mod_underbody_panel, dict(coverage_fraction=0.70)),
            (mod_side_skirts, dict(height_mm=50.0)),
            (mod_rear_diffuser, dict(angle_deg=5.0, length_norm=0.12)),
            (mod_wheel_covers, dict(n_wheels_covered=4)),
        ])
        print_mod_set(ms, base["Cd"])
