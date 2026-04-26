"""
Layer 2: Aerodynamic Modification Physics Engine
Road Car Aerodynamic Fuel Efficiency Engine
==============================================

Each modification is encoded as a physics-derived delta on the car's
drag coefficient. The approach per modification:

    1. Express the geometry change in terms the panel method understands
       (change in angle, area, clearance, etc.)
    2. Compute how that geometry change affects the dominant drag source
       (base pressure, skin friction, or parasitic)
    3. Apply physical feasibility constraints — the optimizer cannot
       request a modification that would cause flow separation, structural
       failure, or violate road legality
    4. Return ΔCd, the physical explanation, and whether constraints were hit

Physics sources cited inline per modification.

Modification catalogue:
    A. Rear spoiler      — controls separation point, reduces wake width
    B. Front splitter    — shifts stagnation, reduces underbody flow
    C. Underbody panel   — smooths turbulent underbody, reduces skin friction
    D. Rear diffuser     — accelerates underbody air, reduces base pressure
    E. Side skirts       — seals underbody from high-pressure sides
    F. Wheel covers      — reduces rotating-wheel turbulence drag
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from core.panel_solver import solve_car, solve, ARCHETYPES, INDIAN_CARS


# ════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════

@dataclass
class ModResult:
    """
    Output of applying one modification to one car.

    All deltas are REDUCTIONS (positive = drag reduced = good).
    ΔCd_total is what gets passed to the WLTP layer.
    """
    mod_name:        str
    params_used:     dict          # geometric parameters of the modification
    delta_Cd:        float         # drag reduction (positive = improvement)
    delta_Cd_pct:    float         # as % of baseline Cd
    constraint_hit:  Optional[str] # which constraint limited the result, if any
    explanation:     str           # physics explanation of why this works
    Cd_new:          float         # absolute Cd after modification
    feasible:        bool          # False if modification violates hard limits


@dataclass
class ModSet:
    """A combination of modifications applied together."""
    modifications:   list          # list of ModResult
    delta_Cd_total:  float         # combined drag reduction
    Cd_final:        float         # final Cd after all modifications
    interaction_note: str          # any interaction effects between mods


# ════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS AND LIMITS
# ════════════════════════════════════════════════════════════════

# Hard physical limits — these are not arbitrary.
# Exceeding them makes drag WORSE, not better.

SPOILER_STALL_ANGLE_DEG   = 15.0   # flow separates off spoiler above this
                                    # Source: Katz, "Race Car Aerodynamics" (1995)

DIFFUSER_MAX_ANGLE_DEG    = 7.0    # underbody flow separates above this
                                    # Source: Senior & Zhang, SAE 2000-01-0354

DIFFUSER_MIN_CLEARANCE_MM = 120.0  # road legal ground clearance (India CMVR)
                                    # Source: CMVR Rule 95(1)

SPLITTER_MAX_DEPTH_MM     = 80.0   # structural limit for road use (empirical)
SKIRT_MIN_CLEARANCE_MM    = 50.0   # minimum skirt-to-ground gap (road debris)

RHO = 1.225   # kg/m³, standard atmosphere
NU  = 1.5e-5  # m²/s, kinematic viscosity of air at 20°C


# ════════════════════════════════════════════════════════════════
# A. REAR SPOILER
# ════════════════════════════════════════════════════════════════

def mod_rear_spoiler(baseline: dict,
                     chord_m: float = 0.25,
                     angle_deg: float = 8.0,
                     span_fraction: float = 0.85) -> ModResult:
    """
    Rear spoiler — reduces wake width by controlling separation point.

    Physics mechanism:
        A spoiler at the roof trailing edge acts as a raised Kutta condition.
        It forces the flow to leave the body cleanly at the spoiler tip
        rather than separating gradually up the rear window. This raises the
        effective separation point (moves it rearward), which:
          1. Reduces the width of the separated wake
          2. Increases base pressure (less negative Cpb)
          3. Reduces base drag

        The effect on Cpb is derived from the trailing edge Kutta condition:
        imposing clean separation raises Cpb approximately as:
            ΔCpb ≈ +0.08 × (chord/H_car) × cos(angle_rad)
        (Calibrated from Hucho 1998, Fig 5.45 — spoiler height vs Cpb)

        At angles above SPOILER_STALL_ANGLE_DEG, the spoiler itself
        generates separated flow and INCREASES drag. This is the hard limit.

    Args:
        baseline      : result dict from solve_car() or solve()
        chord_m       : spoiler chord length in metres (depth front-to-back)
        angle_deg     : spoiler angle relative to horizontal flow
        span_fraction : fraction of car width covered by spoiler (0–1)

    Returns:
        ModResult with ΔCd and explanation
    """
    params = baseline['params']
    Cd0    = baseline['Cd']
    Cpb0   = baseline['Cpb']
    H_car  = params['height_m']
    A_ref  = params['frontal_area_m2']

    # ── Feasibility check ─────────────────────────────────────────
    if angle_deg > SPOILER_STALL_ANGLE_DEG:
        return ModResult(
            mod_name="rear_spoiler",
            params_used=dict(chord_m=chord_m, angle_deg=angle_deg,
                             span_fraction=span_fraction),
            delta_Cd=0.0, delta_Cd_pct=0.0,
            constraint_hit=f"Angle {angle_deg}° exceeds stall limit "
                           f"({SPOILER_STALL_ANGLE_DEG}°) — flow "
                           f"separates off spoiler, drag increases",
            explanation="Modification infeasible at this angle.",
            Cd_new=Cd0, feasible=False
        )

    if chord_m <= 0 or span_fraction <= 0:
        return ModResult("rear_spoiler", {}, 0.0, 0.0,
                         "Non-positive geometry", "", Cd0, False)

    # ── ΔCpb from spoiler (Kutta condition model) ─────────────────
    angle_rad  = np.radians(angle_deg)
    h_ratio    = chord_m / H_car           # spoiler chord / car height

    # Improvement in base pressure (less negative = less drag)
    # Reference: Hucho "Aerodynamics of Road Vehicles" 4th ed., Fig 5.45
    delta_Cpb = 0.08 * h_ratio * np.cos(angle_rad) * span_fraction

    # New base pressure (less negative = less drag)
    Cpb_new   = Cpb0 + delta_Cpb          # e.g. -0.28 → -0.24

    # ── ΔCd_base ─────────────────────────────────────────────────
    rwa_rad      = np.radians(params['rear_window_angle_deg'])
    A_base_ratio = 0.30 + 0.65 * np.sin(rwa_rad)

    Cd_base_old  = abs(Cpb0)    * A_base_ratio
    Cd_base_new  = abs(Cpb_new) * A_base_ratio
    delta_Cd_base = Cd_base_old - Cd_base_new   # positive = improvement

    # ── Spoiler own drag (induced drag from lift on spoiler) ──────
    # Spoiler acts as a small wing. Its drag penalty:
    #   Cd_spoiler = Cl_spoiler² / (π × AR × e) × (A_spoiler / A_ref)
    # For a flat plate: Cl ≈ 2π sin(α), AR = span/chord
    A_spoiler_m2 = chord_m * params['width_m'] * span_fraction
    Cl_spoiler   = 2 * np.pi * np.sin(angle_rad)
    AR_spoiler   = (params['width_m'] * span_fraction) / chord_m
    e_oswald     = 0.7   # Oswald efficiency for a short-span spoiler
    Cd_spoiler_own = (Cl_spoiler**2 / (np.pi * AR_spoiler * e_oswald)
                      * A_spoiler_m2 / A_ref)

    delta_Cd = delta_Cd_base - Cd_spoiler_own
    Cd_new   = max(Cd0 - delta_Cd, 0.05)

    # ── Which constraint was binding? ─────────────────────────────
    constraint = None
    if angle_deg > 12:
        constraint = f"Approaching stall limit ({SPOILER_STALL_ANGLE_DEG}°)"

    explanation = (
        f"Spoiler (chord {chord_m*1000:.0f}mm, angle {angle_deg:.1f}°) "
        f"raises the effective separation point at the car's trailing edge. "
        f"This increases base pressure from Cpb={Cpb0:.3f} to "
        f"Cpb={Cpb_new:.3f} (less suction in wake), reducing base drag by "
        f"ΔCd_base={delta_Cd_base:.4f}. The spoiler's own induced drag "
        f"penalty is {Cd_spoiler_own:.4f}. "
        f"Net drag reduction: ΔCd={delta_Cd:.4f} "
        f"({delta_Cd/Cd0*100:.1f}% of baseline)."
    )

    return ModResult(
        mod_name="rear_spoiler",
        params_used=dict(chord_m=chord_m, angle_deg=angle_deg,
                         span_fraction=span_fraction),
        delta_Cd=delta_Cd,
        delta_Cd_pct=delta_Cd / Cd0 * 100,
        constraint_hit=constraint,
        explanation=explanation,
        Cd_new=Cd_new,
        feasible=True
    )


# ════════════════════════════════════════════════════════════════
# B. FRONT SPLITTER
# ════════════════════════════════════════════════════════════════

def mod_front_splitter(baseline: dict,
                       depth_mm: float = 50.0,
                       height_mm: float = 40.0) -> ModResult:
    """
    Front splitter — shifts stagnation point, reduces underbody mass flow.

    Physics mechanism:
        Without a splitter, the stagnation point sits at the car's nose,
        and significant airflow is directed under the car (high dynamic
        pressure under the body = increased underbody drag).

        A splitter is a horizontal plate extending forward from the front
        bumper at near-ground level. It forces the stagnation point lower
        and blocks high-pressure air from entering the underbody channel.

        The drag reduction comes from two effects:
          1. Reduced underbody mass flow → lower skin friction underneath
          2. Raised front stagnation point → less adverse pressure gradient
             on front bumper face

        Model: underbody velocity reduction estimated from continuity.
        Reduced underbody area (splitter blocks fraction of inlet height):
            ΔV_underbody / V_∞ ≈ - depth_mm / (underbody_gap_mm × 2)
        Reduced underbody Cf → ΔCd_friction (Prandtl formula, local Re)

    Args:
        baseline   : result dict from solve_car() or solve()
        depth_mm   : how far splitter extends forward of bumper (mm)
        height_mm  : vertical thickness of splitter plate (mm)

    Returns:
        ModResult with ΔCd and explanation
    """
    params = baseline['params']
    Cd0    = baseline['Cd']
    L_m    = params['length_m']
    A_ref  = params['frontal_area_m2']
    V_inf  = baseline['V_inf_ms']

    # ── Feasibility ───────────────────────────────────────────────
    if depth_mm > SPLITTER_MAX_DEPTH_MM:
        return ModResult(
            mod_name="front_splitter",
            params_used=dict(depth_mm=depth_mm, height_mm=height_mm),
            delta_Cd=0.0, delta_Cd_pct=0.0,
            constraint_hit=f"Depth {depth_mm}mm exceeds road-use structural "
                           f"limit ({SPLITTER_MAX_DEPTH_MM}mm)",
            explanation="Modification infeasible at this depth.",
            Cd_new=Cd0, feasible=False
        )

    # ── Underbody inlet height (ground clearance) ─────────────────
    clearance_m = params['underbody_clearance_norm'] * params['height_m']
    clearance_mm = clearance_m * 1000

    # Fraction of underbody inlet blocked by splitter
    block_fraction = min(height_mm / clearance_mm, 0.6)  # cap at 60% of gap

    # Velocity reduction in underbody channel (continuity: A1V1 = A2V2)
    # If inlet area reduced by block_fraction, underbody V increases
    # (venturi effect). But this is at the INLET — behind the splitter,
    # the channel is normal width, so total mass flow is reduced.
    # Net effect: less total air under car.
    underbody_flow_reduction = block_fraction * 0.5  # conservative 50% transfer

    # ── ΔCd from reduced underbody skin friction ──────────────────
    # Underbody wetted area ≈ L × width (flat underside assumption)
    S_underbody_m2 = L_m * params['width_m']
    Re_underbody   = V_inf * L_m / NU
    Cf_underbody   = 0.074 / (Re_underbody ** 0.2)

    Cd_underbody_0   = Cf_underbody * S_underbody_m2 / A_ref
    delta_Cd_friction = Cd_underbody_0 * underbody_flow_reduction * 0.4
    # × 0.4 because only part of friction drag is from underbody,
    # and reduction is partial (continuity, not complete blockage)

    # ── ΔCd from frontal stagnation pressure improvement ──────────
    # Raising stagnation reduces adverse dp/dx on bumper face.
    # Empirical: Hucho 1998, Fig 5.21 — splitter depth vs ΔCd.
    # Fitted: ΔCd_stag ≈ 0.004 × (depth_mm / 50)^0.6
    delta_Cd_stag = 0.004 * (depth_mm / 50.0) ** 0.6

    delta_Cd = delta_Cd_friction + delta_Cd_stag
    Cd_new   = max(Cd0 - delta_Cd, 0.05)

    constraint = None
    if depth_mm > 60:
        constraint = "Approaching structural limit for road use"

    explanation = (
        f"Front splitter (depth {depth_mm:.0f}mm) blocks {block_fraction*100:.0f}% "
        f"of underbody inlet height, reducing underbody mass flow. "
        f"Lower underbody flow → reduced skin friction under car: "
        f"ΔCd_friction={delta_Cd_friction:.4f}. "
        f"Stagnation point raised on bumper face: ΔCd_stag={delta_Cd_stag:.4f}. "
        f"Net drag reduction: ΔCd={delta_Cd:.4f} "
        f"({delta_Cd/Cd0*100:.1f}% of baseline)."
    )

    return ModResult(
        mod_name="front_splitter",
        params_used=dict(depth_mm=depth_mm, height_mm=height_mm),
        delta_Cd=delta_Cd,
        delta_Cd_pct=delta_Cd / Cd0 * 100,
        constraint_hit=constraint,
        explanation=explanation,
        Cd_new=Cd_new,
        feasible=True
    )


# ════════════════════════════════════════════════════════════════
# C. UNDERBODY PANEL
# ════════════════════════════════════════════════════════════════

def mod_underbody_panel(baseline: dict,
                        coverage_fraction: float = 0.70) -> ModResult:
    """
    Underbody panel — smooths turbulent underbody flow, reduces skin friction.

    Physics mechanism:
        A road car's underbody is highly irregular — engine sumps, exhaust
        pipes, suspension components, fuel tanks all protrude downward.
        Each protrusion creates a local separated flow region with high
        skin friction and form drag. Flat underbody panels cover these
        components, creating a smooth surface.

        The drag reduction is dominated by two effects:
          1. Elimination of form drag from protruding components
             (each protrusion approximated as a bluff body: Cd_protrusion ≈ 1.0
             × projected area × local dynamic pressure)
          2. Reduction in turbulent skin friction from rough surface to smooth

        Reference: Hucho (1998), Chapter 4 — underbody contribution to total
        drag is typically 10–15% of Cd for an unpaneled road car.
        SAE 2003-01-0655 (Gilliéron & Kourta) quantifies underbody drag
        on production vehicles: 0.025–0.045 Cd units savings with full panels.

    Args:
        baseline            : result dict from solve_car() or solve()
        coverage_fraction   : fraction of underbody area covered (0–1)
                              0.7 = typical partial panel (engine + gearbox)
                              1.0 = full flat underbody (rare on road cars)

    Returns:
        ModResult with ΔCd and explanation
    """
    params = baseline['params']
    Cd0    = baseline['Cd']

    coverage_fraction = np.clip(coverage_fraction, 0.0, 1.0)

    # ── Underbody drag contribution (pre-modification) ────────────
    # From SAE 2003-01-0655: underbody = 12% of total Cd for typical car.
    # This includes both skin friction and protrusion form drag.
    underbody_Cd_fraction = 0.12   # 12% of total Cd is from underbody
    Cd_underbody_baseline  = Cd0 * underbody_Cd_fraction

    # Panel removes the form drag component (protrusions).
    # Smooth panel retains only skin friction (about 35% of underbody drag).
    # So effective removal = 65% of underbody drag × coverage.
    protrusion_fraction    = 0.65
    delta_Cd = (Cd_underbody_baseline * protrusion_fraction
                * coverage_fraction)

    Cd_new = max(Cd0 - delta_Cd, 0.05)

    # ── Additional check: clearance ───────────────────────────────
    clearance_mm = (params['underbody_clearance_norm']
                    * params['height_m'] * 1000)
    constraint = None
    if clearance_mm < DIFFUSER_MIN_CLEARANCE_MM:
        constraint = (f"Ground clearance {clearance_mm:.0f}mm is already "
                      f"below recommended minimum for paneling "
                      f"({DIFFUSER_MIN_CLEARANCE_MM}mm)")

    explanation = (
        f"Underbody panel covering {coverage_fraction*100:.0f}% of floor area. "
        f"Smooth panel eliminates form drag from engine/exhaust protrusions "
        f"(≈{protrusion_fraction*100:.0f}% of underbody drag) and reduces "
        f"skin friction on covered section. "
        f"Underbody drag is ≈{underbody_Cd_fraction*100:.0f}% of total Cd "
        f"(SAE 2003-01-0655), giving ΔCd={delta_Cd:.4f} "
        f"({delta_Cd/Cd0*100:.1f}% of baseline). "
        f"This is the highest return-on-simplicity modification — no moving "
        f"parts, no angle tuning required."
    )

    return ModResult(
        mod_name="underbody_panel",
        params_used=dict(coverage_fraction=coverage_fraction),
        delta_Cd=delta_Cd,
        delta_Cd_pct=delta_Cd / Cd0 * 100,
        constraint_hit=constraint,
        explanation=explanation,
        Cd_new=Cd_new,
        feasible=True
    )


# ════════════════════════════════════════════════════════════════
# D. REAR DIFFUSER
# ════════════════════════════════════════════════════════════════

def mod_rear_diffuser(baseline: dict,
                      angle_deg: float = 5.0,
                      length_norm: float = 0.12) -> ModResult:
    """
    Rear diffuser — accelerates underbody air, reduces base pressure drag.

    Physics mechanism:
        A diffuser is a diverging duct at the rear of the underbody.
        By expanding the duct cross-section, it decelerates the underbody
        airflow and recovers static pressure (Bernoulli).

        The key physics: if underbody air exits at near-ambient pressure
        (rather than low pressure), the pressure difference between the
        underbody and the wake is reduced. This reduces the suction in the
        separated wake (raises Cpb), directly reducing base drag.

        Energy recovery model (from continuity + Bernoulli):
            Area ratio: AR = A_exit / A_inlet = 1 + 2 × length × tan(θ)
            Exit velocity: V_exit = V_inlet / AR   (continuity)
            Pressure recovery: ΔCp_recovery = 1 - (1/AR)²  (Bernoulli)

        Constraint: flow separates inside diffuser if angle > ~7°
        (Senior & Zhang, SAE 2000-01-0354 — the definitive reference).

    Args:
        baseline    : result dict from solve_car() or solve()
        angle_deg   : diffuser expansion half-angle (degrees)
        length_norm : diffuser length as fraction of car length

    Returns:
        ModResult with ΔCd and explanation
    """
    params = baseline['params']
    Cd0    = baseline['Cd']
    Cpb0   = baseline['Cpb']
    H_car  = params['height_m']

    # ── Feasibility ───────────────────────────────────────────────
    clearance_m  = params['underbody_clearance_norm'] * H_car
    clearance_mm = clearance_m * 1000

    if angle_deg > DIFFUSER_MAX_ANGLE_DEG:
        return ModResult(
            mod_name="rear_diffuser",
            params_used=dict(angle_deg=angle_deg, length_norm=length_norm),
            delta_Cd=0.0, delta_Cd_pct=0.0,
            constraint_hit=(f"Angle {angle_deg}° exceeds separation limit "
                            f"({DIFFUSER_MAX_ANGLE_DEG}°). Flow separates "
                            f"inside diffuser, eliminating pressure recovery."),
            explanation="Modification infeasible at this angle.",
            Cd_new=Cd0, feasible=False
        )

    if clearance_mm < DIFFUSER_MIN_CLEARANCE_MM:
        return ModResult(
            mod_name="rear_diffuser",
            params_used=dict(angle_deg=angle_deg, length_norm=length_norm),
            delta_Cd=0.0, delta_Cd_pct=0.0,
            constraint_hit=(f"Ground clearance {clearance_mm:.0f}mm below "
                            f"CMVR minimum ({DIFFUSER_MIN_CLEARANCE_MM}mm)"),
            explanation="Insufficient ground clearance for diffuser.",
            Cd_new=Cd0, feasible=False
        )

    # ── Area ratio and pressure recovery ─────────────────────────
    L_diffuser_m = length_norm * params['length_m']
    angle_rad    = np.radians(angle_deg)

    # Diffuser height rises from clearance_m by angle over length
    h_exit = clearance_m + L_diffuser_m * np.tan(angle_rad)
    AR     = h_exit / clearance_m     # area ratio (2D; width constant)

    # Pressure recovery coefficient (Bernoulli, ideal):
    # Cp_recovery = 1 - (1/AR)²
    Cp_recovery_ideal = 1.0 - (1.0 / AR) ** 2

    # Real diffusers recover about 60–70% of ideal (boundary layer losses)
    diffuser_efficiency = 0.65
    Cp_recovery = Cp_recovery_ideal * diffuser_efficiency

    # ── Effect on base pressure and Cd ───────────────────────────
    # Higher underbody exit pressure → less negative Cpb in wake
    # Empirical coupling factor (fraction of Cp_recovery that improves Cpb):
    # Reference: Zhang et al., SAE 2006-01-0337 — diffuser-wake interaction
    coupling = 0.40
    delta_Cpb = Cp_recovery * coupling

    Cpb_new  = Cpb0 + delta_Cpb
    rwa_rad  = np.radians(params['rear_window_angle_deg'])
    A_base_r = 0.30 + 0.65 * np.sin(rwa_rad)

    Cd_base_old = abs(Cpb0)    * A_base_r
    Cd_base_new = abs(Cpb_new) * A_base_r
    delta_Cd    = Cd_base_old - Cd_base_new
    Cd_new      = max(Cd0 - delta_Cd, 0.05)

    constraint = None
    if angle_deg > 5.5:
        constraint = f"Approaching separation limit ({DIFFUSER_MAX_ANGLE_DEG}°)"

    explanation = (
        f"Rear diffuser: angle {angle_deg:.1f}°, length "
        f"{L_diffuser_m*1000:.0f}mm. "
        f"Area ratio AR={AR:.2f} — underbody air decelerates from inlet "
        f"to exit, recovering Cp_recovery={Cp_recovery:.3f} (ideal × "
        f"{diffuser_efficiency} efficiency). "
        f"This raises base pressure from Cpb={Cpb0:.3f} to "
        f"Cpb={Cpb_new:.3f}, reducing base drag by ΔCd={delta_Cd:.4f} "
        f"({delta_Cd/Cd0*100:.1f}% of baseline). "
        f"Physics: Bernoulli pressure recovery in diverging duct "
        f"(Senior & Zhang, SAE 2000-01-0354)."
    )

    return ModResult(
        mod_name="rear_diffuser",
        params_used=dict(angle_deg=angle_deg, length_norm=length_norm),
        delta_Cd=delta_Cd,
        delta_Cd_pct=delta_Cd / Cd0 * 100,
        constraint_hit=constraint,
        explanation=explanation,
        Cd_new=Cd_new,
        feasible=True
    )


# ════════════════════════════════════════════════════════════════
# E. SIDE SKIRTS
# ════════════════════════════════════════════════════════════════

def mod_side_skirts(baseline: dict,
                    height_mm: float = 60.0,
                    coverage_fraction: float = 0.75) -> ModResult:
    """
    Side skirts — seal underbody from high-pressure side flow.

    Physics mechanism:
        Car sides are at high pressure (near stagnation). The underbody
        is at low pressure (high-speed flow). Without skirts, air
        constantly spills from the sides under the car, creating:
          1. Spanwise vortices along the underbody edges
          2. Additional turbulence mixing (viscous dissipation → drag)
          3. Disruption of any diffuser action at the rear

        Skirts seal this pressure differential. The drag reduction
        is modelled as eliminating the vortex-induced drag from the
        underbody edge vortices.

        Vortex drag model:
            Each underbody edge vortex has an effective circulation Γ.
            The side pressure differential drives Γ:
                Γ ≈ ΔCp_side × V_inf × h_skirt
            Induced drag from one vortex pair:
                Cd_vortex = Γ² / (π × V_inf² × b²) × (A_vortex / A_ref)
            where b = car width.

    Args:
        baseline            : result dict from solve_car() or solve()
        height_mm           : skirt depth below car sill (mm)
        coverage_fraction   : fraction of wheelbase length covered

    Returns:
        ModResult with ΔCd and explanation
    """
    params = baseline['params']
    Cd0    = baseline['Cd']
    V_inf  = baseline['V_inf_ms']

    height_m   = height_mm / 1000.0
    clearance_m = params['underbody_clearance_norm'] * params['height_m']

    # ── Feasibility ───────────────────────────────────────────────
    clearance_mm = clearance_m * 1000
    gap_mm = clearance_mm - height_mm
    if gap_mm < SKIRT_MIN_CLEARANCE_MM:
        return ModResult(
            mod_name="side_skirts",
            params_used=dict(height_mm=height_mm,
                             coverage_fraction=coverage_fraction),
            delta_Cd=0.0, delta_Cd_pct=0.0,
            constraint_hit=(f"Skirt height {height_mm}mm leaves only "
                            f"{gap_mm:.0f}mm ground clearance — below "
                            f"minimum {SKIRT_MIN_CLEARANCE_MM}mm for road "
                            f"debris safety"),
            explanation="Modification infeasible at this height.",
            Cd_new=Cd0, feasible=False
        )

    # ── Vortex drag calculation ────────────────────────────────────
    # Side pressure coefficient (stagnation region at B-pillar area)
    # Cp_side ≈ 0.3 for typical road car
    Cp_side_diff   = 0.30

    # Circulation of underbody edge vortex (per unit span):
    Gamma = Cp_side_diff * V_inf * height_m

    # Car width (span for vortex calculation)
    b_m = params['width_m']
    A_ref = params['frontal_area_m2']

    # Induced drag of two symmetric vortices at underbody edges
    # Simplified lifting-line analogy for ground vortex pair
    Cd_vortex_pair = (2 * Gamma**2 / (np.pi * V_inf**2 * b_m**2)
                      * (b_m * params['length_m'] / A_ref))

    # Skirts eliminate this vortex drag proportional to coverage
    delta_Cd = Cd_vortex_pair * coverage_fraction
    Cd_new   = max(Cd0 - delta_Cd, 0.05)

    explanation = (
        f"Side skirts ({height_mm:.0f}mm deep, {coverage_fraction*100:.0f}% "
        f"wheelbase coverage) seal the pressure differential between car "
        f"sides (Cp≈+0.30) and underbody (low pressure). "
        f"Without skirts, this drives underbody edge vortices with "
        f"circulation Γ={Gamma:.2f} m²/s. Skirts eliminate these vortices, "
        f"removing their induced drag: ΔCd={delta_Cd:.4f} "
        f"({delta_Cd/Cd0*100:.1f}% of baseline)."
    )

    return ModResult(
        mod_name="side_skirts",
        params_used=dict(height_mm=height_mm,
                         coverage_fraction=coverage_fraction),
        delta_Cd=delta_Cd,
        delta_Cd_pct=delta_Cd / Cd0 * 100,
        constraint_hit=None,
        explanation=explanation,
        Cd_new=Cd_new,
        feasible=True
    )


# ════════════════════════════════════════════════════════════════
# F. WHEEL COVERS
# ════════════════════════════════════════════════════════════════

def mod_wheel_covers(baseline: dict,
                     n_wheels_covered: int = 4) -> ModResult:
    """
    Wheel covers — reduce rotating-wheel turbulence drag.

    Physics mechanism:
        An exposed rotating wheel is a highly complex aerodynamic body.
        It contributes drag through:
          1. Bluff body form drag (wheel face into flow)
          2. Rotating contact patch vortices (wheel-ground junction)
          3. Spanwise flow ejected centrifugally from tire tread

        Wheel covers (flat disc covers over the wheel face) eliminate
        the form drag and much of the rotating surface turbulence.

        Each wheel contributes approximately Cd_wheel ≈ 0.007–0.012
        to the total vehicle Cd (Hucho 1998, Chapter 7; Cogotti 1983
        "Aerodynamic characteristics of car wheels").

        Covered vs open: reduction of ≈60% of wheel drag contribution.
        (Aerodynamically smooth wheels like Tesla Model 3 Aero demonstrate
        this: +0.025 Cd improvement from aero wheel covers, SAE 2019.)

    Args:
        baseline          : result dict from solve_car() or solve()
        n_wheels_covered  : number of wheels fitted with covers (1–4)

    Returns:
        ModResult with ΔCd and explanation
    """
    params = baseline['params']
    Cd0    = baseline['Cd']

    n_wheels_covered = int(np.clip(n_wheels_covered, 0, 4))

    # Wheel drag per wheel (Cogotti 1983, Hucho 1998 Table 7.2)
    Cd_per_wheel_open     = 0.009   # typical road car wheel, open
    Cd_per_wheel_covered  = 0.003   # smooth disc cover (60% reduction)
    delta_per_wheel       = Cd_per_wheel_open - Cd_per_wheel_covered

    # Total saving
    delta_Cd = delta_per_wheel * n_wheels_covered
    Cd_new   = max(Cd0 - delta_Cd, 0.05)

    explanation = (
        f"{n_wheels_covered} wheel cover(s) fitted. "
        f"Each open wheel contributes Cd≈{Cd_per_wheel_open:.3f} via "
        f"rotating bluff-body drag and contact-patch vortices "
        f"(Cogotti 1983). A flush disc cover reduces this to "
        f"Cd≈{Cd_per_wheel_covered:.3f} — eliminating spoke turbulence "
        f"and most rotating-face form drag. "
        f"Total saving: ΔCd={delta_Cd:.4f} "
        f"({delta_Cd/Cd0*100:.1f}% of baseline). "
        f"Note: rear wheel covers have diminishing returns on cars "
        f"with rear wheel arch fairings already fitted."
    )

    return ModResult(
        mod_name="wheel_covers",
        params_used=dict(n_wheels_covered=n_wheels_covered),
        delta_Cd=delta_Cd,
        delta_Cd_pct=delta_Cd / Cd0 * 100,
        constraint_hit=None,
        explanation=explanation,
        Cd_new=Cd_new,
        feasible=True
    )


# ════════════════════════════════════════════════════════════════
# COMBINED MODIFICATION SET
# ════════════════════════════════════════════════════════════════

def apply_mod_set(baseline: dict, mod_list: list) -> ModSet:
    """
    Apply a list of modifications sequentially and compute combined ΔCd.

    Modifications interact — applying a diffuser after side skirts is more
    effective because skirts prevent lateral contamination of the diffuser
    flow. The interaction model is conservative (additive with one
    interaction correction term) rather than claiming perfect independence.

    Args:
        baseline  : result dict from solve_car() or solve()
        mod_list  : list of (function, kwargs) tuples e.g.:
                    [(mod_underbody_panel, {'coverage_fraction': 0.7}),
                     (mod_rear_diffuser,   {'angle_deg': 5.0})]

    Returns:
        ModSet with combined ΔCd and per-modification breakdown
    """
    results    = []
    Cd_current = baseline['Cd']
    # Build a mutable baseline copy so each mod sees updated Cd
    current_baseline = dict(baseline)

    for fn, kwargs in mod_list:
        result = fn(current_baseline, **kwargs)
        results.append(result)
        if result.feasible:
            current_baseline = dict(baseline)
            current_baseline['Cd'] = result.Cd_new
            Cd_current = result.Cd_new

    total_delta = baseline['Cd'] - Cd_current

    # ── Interaction effect ────────────────────────────────────────
    has_skirts   = any(r.mod_name == "side_skirts"   for r in results)
    has_diffuser = any(r.mod_name == "rear_diffuser"  for r in results)
    has_panel    = any(r.mod_name == "underbody_panel" for r in results)

    interaction_note = ""
    if has_skirts and has_diffuser:
        # Skirts improve diffuser effectiveness by ~15%
        bonus = total_delta * 0.08
        Cd_current   -= bonus
        total_delta  += bonus
        interaction_note += (
            "Skirt + diffuser synergy: skirts seal underbody, improving "
            f"diffuser pressure recovery by ~8% (ΔCd bonus ≈ {bonus:.4f}). "
        )
    if has_panel and has_diffuser:
        bonus = total_delta * 0.05
        Cd_current  -= bonus
        total_delta += bonus
        interaction_note += (
            "Underbody panel + diffuser synergy: smooth panel reduces "
            f"boundary layer thickness entering diffuser (+5% recovery). "
        )

    if not interaction_note:
        interaction_note = "No significant interaction effects between these modifications."

    return ModSet(
        modifications=results,
        delta_Cd_total=total_delta,
        Cd_final=max(Cd_current, 0.05),
        interaction_note=interaction_note
    )


# ════════════════════════════════════════════════════════════════
# PRINT HELPERS
# ════════════════════════════════════════════════════════════════

def print_mod_result(r: ModResult):
    status = "✓ FEASIBLE" if r.feasible else "✗ INFEASIBLE"
    print(f"\n  [{r.mod_name.upper().replace('_',' ')}]  {status}")
    print(f"    ΔCd = {r.delta_Cd:+.4f}  ({r.delta_Cd_pct:+.1f}%)")
    print(f"    Cd after: {r.Cd_new:.4f}")
    if r.constraint_hit:
        print(f"    Constraint: {r.constraint_hit}")
    # Wrap explanation at 70 chars
    words  = r.explanation.split()
    line   = "    "
    for w in words:
        if len(line) + len(w) > 74:
            print(line); line = "    " + w + " "
        else:
            line += w + " "
    if line.strip():
        print(line)


def print_mod_set(ms: ModSet, baseline_Cd: float):
    print("\n" + "=" * 66)
    print("  MODIFICATION SET SUMMARY")
    print("=" * 66)
    for r in ms.modifications:
        flag = "✓" if r.feasible else "✗"
        print(f"  {flag} {r.mod_name:<22}  ΔCd={r.delta_Cd:+.4f}  "
              f"({r.delta_Cd_pct:+.1f}%)")
    print("  " + "-" * 62)
    print(f"  TOTAL drag reduction:   ΔCd = {ms.delta_Cd_total:+.4f}")
    print(f"  Baseline Cd:            {baseline_Cd:.4f}")
    print(f"  Final Cd:               {ms.Cd_final:.4f}")
    pct = ms.delta_Cd_total / baseline_Cd * 100
    print(f"  Total improvement:      {pct:.1f}%")
    print(f"\n  Interaction note: {ms.interaction_note}")
    print("=" * 66)


# ════════════════════════════════════════════════════════════════
# DEMO
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n  LAYER 2 — MODIFICATION PHYSICS ENGINE")
    print("  Testing on Maruti Swift and Tata Nexon\n")

    for car_key in ["maruti_swift", "tata_nexon"]:
        baseline = solve_car(car_key)
        car_name = INDIAN_CARS[car_key]["display_name"]

        print(f"\n{'═'*66}")
        print(f"  {car_name}")
        print(f"  Baseline Cd = {baseline['Cd']:.4f}")
        print(f"{'═'*66}")

        # Test each modification individually at sensible defaults
        mods_to_test = [
            (mod_underbody_panel,  dict(coverage_fraction=0.70)),
            (mod_rear_diffuser,    dict(angle_deg=5.0, length_norm=0.12)),
            (mod_rear_spoiler,     dict(chord_m=0.25, angle_deg=8.0)),
            (mod_front_splitter,   dict(depth_mm=50.0)),
            (mod_side_skirts,      dict(height_mm=55.0)),
            (mod_wheel_covers,     dict(n_wheels_covered=4)),
        ]

        for fn, kwargs in mods_to_test:
            r = fn(baseline, **kwargs)
            print_mod_result(r)

        # Full combined set
        print(f"\n  ── COMBINED MODIFICATION SET ──")
        ms = apply_mod_set(baseline, [
            (mod_underbody_panel, dict(coverage_fraction=0.70)),
            (mod_rear_diffuser,   dict(angle_deg=5.0, length_norm=0.12)),
            (mod_side_skirts,     dict(height_mm=55.0)),
            (mod_wheel_covers,    dict(n_wheels_covered=4)),
            (mod_rear_spoiler,    dict(chord_m=0.20, angle_deg=7.0)),
        ])
        print_mod_set(ms, baseline['Cd'])