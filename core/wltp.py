"""
Layer 3: WLTP Drive Cycle Integration
Road Car Aerodynamic Fuel Efficiency Engine
============================================

Converts a drag coefficient into real-world fuel consumption figures
by integrating the drag force over the WLTP (Worldwide Harmonised Light
Vehicle Test Procedure) drive cycle.

Physics chain:
    Cd + A_frontal
        → F_drag(t) = ½ρv(t)²CdA          [drag force at each timestep]
        → P_drag(t) = F_drag(t) × v(t)     [power required to overcome drag]
        → E_aero = ∫ P_drag dt             [total aerodynamic energy, joules]
        → E_fuel = E_aero / η_engine        [fuel energy needed]
        → V_fuel = E_fuel / E_density       [fuel volume, litres]
        → L/100km = V_fuel / distance × 100

Every constant in this chain is a physical constant or public standard.
Nothing is invented.

WLTP cycle:
    The Worldwide Harmonised Light Vehicle Test Procedure (WLTP) is defined
    in UN GTR No. 15 (2015, amended 2018). It replaces NEDC and is now
    mandatory for all new cars sold in EU, UK, India (BIS IS 17874).
    The cycle is 1800 seconds, 23.266 km, with four phases:
        Low    : 0–589s,    v_max = 56.5 km/h  (urban)
        Medium : 589–1022s, v_max = 76.6 km/h  (suburban)
        High   : 1022–1477s,v_max = 97.4 km/h  (rural)
        Extra  : 1477–1800s,v_max = 131.3 km/h (motorway)

    The speed-time profile used here is reconstructed from the published
    GTR No.15 trace data (publicly available, UN UNECE repository).

Reference constants:
    ρ_air       = 1.225 kg/m³   (ICAO standard atmosphere, sea level)
    η_petrol    = 0.35          (thermal efficiency, modern petrol engine)
    η_diesel    = 0.40          (thermal efficiency, modern diesel engine)
    E_petrol    = 34.2 MJ/litre (lower heating value, EN 228 standard)
    E_diesel    = 35.7 MJ/litre (lower heating value, EN 590 standard)
    CO2_petrol  = 2.31 kg/litre (carbon content × stoichiometry, IPCC)
    CO2_diesel  = 2.68 kg/litre (IPCC emission factor)
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import Tuple

from core.panel_solver import solve_car, INDIAN_CARS
from core.modifications import (
    apply_mod_set, mod_underbody_panel, mod_rear_diffuser,
    mod_rear_spoiler, mod_wheel_covers, mod_front_splitter,
    mod_side_skirts, ModSet
)


# ════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ════════════════════════════════════════════════════════════════

RHO_AIR       = 1.225    # kg/m³
GRAVITY       = 9.81     # m/s²

# Engine thermal efficiency (fraction of fuel energy → shaft work)
ETA = {
    "petrol": 0.35,
    "diesel": 0.40,
    "hybrid": 0.42,
}

# Fuel energy density (lower heating value)
E_DENSITY = {
    "petrol": 34.2e6,   # J/litre  (EN 228)
    "diesel": 35.7e6,   # J/litre  (EN 590)
    "hybrid": 34.2e6,   # hybrid petrol base
}

# CO₂ emission factor per litre of fuel burned
CO2_PER_LITRE = {
    "petrol": 2.31,   # kg CO₂/litre  (IPCC 2006 Vol 2, Table 3.2.1)
    "diesel": 2.68,   # kg CO₂/litre
    "hybrid": 2.31,
}

# Rolling resistance coefficient (used to compute TOTAL fuel use, not just aero)
# Source: Hucho (1998), Table 1.2 — typical road car on tarmac
CR = 0.013


# ════════════════════════════════════════════════════════════════
# WLTP CLASS 3 DRIVE CYCLE
#   Reconstructed from UN GTR No.15 published speed-time trace.
#   Resolution: 1 Hz (one data point per second).
#   Total: 1800 seconds, 23.266 km.
# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════
# PUBLISHED WLTC CLASS 3b SPECIFICATION
#   UN GTR No.15, Annex 1. These are the numbers the reconstruction below
#   is held to. They are the ground truth, not the waypoint list.
# ════════════════════════════════════════════════════════════════

WLTC_SPEC = {
    #  phase        t_start t_end  distance_km  v_max_kmh
    "low":        (0,     589,   3.095,       56.5),
    "medium":     (589,   1022,  4.756,       76.6),
    "high":       (1022,  1477,  7.158,       97.4),
    "extra_high": (1477,  1800,  8.254,       131.3),
}
WLTC_TOTAL_KM = 23.263

# Path to the official 1 Hz trace. If this file is populated, it is used
# verbatim and the reconstruction below is bypassed entirely.
_TRACE_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "wltp_cycle.csv")


def load_official_trace(path: str = _TRACE_CSV):
    """
    Load the official 1 Hz WLTC trace if present.

    Expected format: two columns, `time_s,speed_kmh`, 1800 rows, with or
    without a header. Download from the UNECE WLTP repository and drop it in
    data/wltp_cycle.csv — the whole pipeline will pick it up automatically and
    the reconstruction error disappears.

    Returns (t_s, v_ms) or None if the file is missing or empty.
    """
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return None
    try:
        raw = np.genfromtxt(path, delimiter=",", skip_header=0)
        if raw.ndim != 2 or raw.shape[0] < 100:
            return None
        if np.isnan(raw[0]).any():                 # header row present
            raw = raw[1:]
        t_s = raw[:, 0].astype(float)
        v_ms = raw[:, 1].astype(float) / 3.6       # km/h -> m/s
        return t_s, v_ms
    except Exception:
        return None


def _shape_exponent(v_phase: np.ndarray, v_max: float,
                    target_km: float, tol: float = 1e-4) -> float:
    """
    Find p such that reshaping v -> v_max * (v/v_max)**p makes the phase cover
    its published distance.

    Why a shape exponent and not a scale factor: scaling every speed by a
    constant would fix the distance but destroy the phase's peak speed, and
    aerodynamic power goes as v^3, so the peak is exactly what must be
    preserved. This transform is a fixed point at v = 0 and at v = v_max, so
    idle stays idle and the peak stays the peak. Only the mid-range is bent,
    which is where the hand-typed waypoint list is actually wrong.

    p > 1 pulls mid-range speeds down (shorter distance); p < 1 pushes them up.
    """
    if v_max <= 0:
        return 1.0
    u = np.clip(v_phase / v_max, 0.0, 1.0)

    def dist_km(p):
        return float(np.trapezoid(v_max * u ** p, dx=1.0)) / 1000.0

    lo, hi = 0.2, 5.0
    if dist_km(hi) > target_km or dist_km(lo) < target_km:
        return 1.0                                  # target unreachable; leave as-is
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if dist_km(mid) > target_km:
            lo = mid                                # need larger p to shrink further
        else:
            hi = mid
        if abs(hi - lo) < tol:
            break
    return 0.5 * (lo + hi)


def build_wltp_cycle(calibrate: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the WLTC Class 3b speed-time profile at 1 Hz.

    Resolution order:
      1. The official 1 Hz trace from data/wltp_cycle.csv, if provided.
      2. Otherwise the waypoint reconstruction below, calibrated per phase so
         that each phase covers its PUBLISHED distance and retains its
         PUBLISHED peak speed.

    Honesty note. The raw waypoint list is a hand-entered approximation of the
    official trace and it is not accurate: uncalibrated it covers 28.4 km
    against a 23.26 km specification, a 22% overshoot, because it lacks much of
    the cycle's low-speed and idle content. Fuel consumption per 100 km is
    proportional to the ratio of integral(v^3) to integral(v), so that error
    does NOT cancel between baseline and modified — it inflates absolute
    L/100km and every rupee figure downstream. The per-phase calibration below
    removes the distance error. For publication-grade absolute numbers, supply
    the official trace; call cycle_report() to see exactly where you stand.

    Returns:
        t_s  : time, seconds, shape (1800,)
        v_ms : speed, m/s,   shape (1800,)
    """
    official = load_official_trace()
    if official is not None:
        return official

    # Waypoint reconstruction (time_s, speed_kmh), linear interpolation between.
    waypoints = [
        # ── LOW phase (urban, 0–589s) ─────────────────────────────
        (0,    0),
        (11,   0),
        (12,   16.7),
        (23,   26.0),
        (39,   26.0),
        (40,   0),
        (57,   0),
        (58,   6.0),
        (68,   21.0),
        (82,   21.0),
        (88,   0),
        (109,  0),
        (110,  7.0),
        (119,  26.0),
        (130,  26.0),
        (134,  0),
        (163,  0),
        (164,  26.0),
        (176,  26.0),
        (178,  38.0),
        (188,  38.0),
        (193,  38.0),
        (203,  0),
        (228,  0),
        (229,  17.0),
        (237,  23.0),
        (249,  23.0),
        (254,  0),
        (274,  0),
        (275,  26.0),
        (290,  38.0),
        (309,  38.0),
        (319,  0),
        (339,  0),
        (340,  26.0),
        (351,  26.0),
        (356,  38.0),
        (369,  38.0),
        (376,  0),
        (396,  0),
        (397,  26.0),
        (413,  26.0),
        (421,  38.0),
        (433,  38.0),
        (438,  0),
        (452,  0),
        (453,  36.0),
        (463,  56.5),
        (472,  56.5),
        (488,  0),
        (513,  0),
        (514,  21.0),
        (524,  38.0),
        (544,  38.0),
        (549,  0),
        (566,  0),
        (567,  21.0),
        (578,  21.0),
        (585,  0),
        (589,  0),

        # ── MEDIUM phase (suburban, 589–1022s) ────────────────────
        (590,  0),
        (600,  38.0),
        (614,  56.0),
        (636,  56.0),
        (644,  38.0),
        (663,  38.0),
        (664,  56.0),
        (687,  56.0),
        (688,  76.6),
        (705,  76.6),
        (706,  56.0),
        (721,  56.0),
        (726,  76.6),
        (745,  76.6),
        (750,  56.0),
        (763,  56.0),
        (764,  76.6),
        (776,  76.6),
        (786,  0),
        (806,  0),
        (807,  56.0),
        (840,  56.0),
        (855,  76.6),
        (871,  76.6),
        (876,  56.0),
        (896,  56.0),
        (904,  76.6),
        (917,  76.6),
        (919,  56.0),
        (940,  56.0),
        (946,  76.6),
        (960,  76.6),
        (972,  38.0),
        (982,  38.0),
        (993,  56.0),
        (1002, 56.0),
        (1019, 0),
        (1022, 0),

        # ── HIGH phase (rural, 1022–1477s) ────────────────────────
        (1023, 0),
        (1035, 56.0),
        (1052, 76.6),
        (1074, 97.4),
        (1106, 97.4),
        (1111, 76.6),
        (1130, 76.6),
        (1135, 97.4),
        (1151, 97.4),
        (1156, 76.6),
        (1171, 76.6),
        (1173, 97.4),
        (1199, 97.4),
        (1204, 76.6),
        (1219, 76.6),
        (1222, 56.0),
        (1232, 56.0),
        (1250, 97.4),
        (1270, 97.4),
        (1275, 76.6),
        (1293, 76.6),
        (1297, 97.4),
        (1307, 97.4),
        (1311, 76.6),
        (1329, 76.6),
        (1333, 97.4),
        (1352, 97.4),
        (1360, 56.0),
        (1378, 56.0),
        (1385, 0),
        (1400, 0),
        (1407, 56.0),
        (1434, 76.6),
        (1451, 76.6),
        (1460, 56.0),
        (1467, 56.0),
        (1474, 0),
        (1477, 0),

        # ── EXTRA HIGH phase (motorway, 1477–1800s) ───────────────
        (1478, 0),
        (1490, 97.4),
        (1520, 131.3),
        (1560, 131.3),
        (1571, 97.4),
        (1590, 97.4),
        (1596, 131.3),
        (1617, 131.3),
        (1626, 97.4),
        (1646, 97.4),
        (1649, 131.3),
        (1677, 131.3),
        (1682, 97.4),
        (1700, 97.4),
        (1703, 131.3),
        (1724, 131.3),
        (1726, 97.4),
        (1742, 97.4),
        (1744, 131.3),
        (1762, 131.3),
        (1769, 97.4),
        (1780, 97.4),
        (1785, 56.0),
        (1795, 0),
        (1800, 0),
    ]

    t_way = np.array([w[0] for w in waypoints], dtype=float)
    v_way = np.array([w[1] for w in waypoints], dtype=float) / 3.6  # km/h -> m/s

    t_s = np.arange(0, 1800, dtype=float)
    v_ms = np.interp(t_s, t_way, v_way)

    if not calibrate:
        return t_s, v_ms

    # --- Per-phase calibration to the published distances -------------------
    v_cal = v_ms.copy()
    for _name, (t0, t1, dist_km, vmax_kmh) in WLTC_SPEC.items():
        mask = (t_s >= t0) & (t_s < t1)
        v_phase = v_ms[mask]
        if v_phase.size == 0 or v_phase.max() <= 0:
            continue
        v_max = vmax_kmh / 3.6
        p = _shape_exponent(v_phase, v_max, dist_km)
        v_cal[mask] = v_max * np.clip(v_phase / v_max, 0.0, 1.0) ** p

    return t_s, v_cal


def cycle_report(verbose: bool = True) -> dict:
    """
    Compare the cycle actually in use against the published WLTC 3b spec.

    This exists so the reconstruction error is always visible rather than
    buried. Call it before quoting any absolute fuel figure.
    """
    t_s, v_ms = build_wltp_cycle()
    using_official = load_official_trace() is not None

    rows = {}
    if verbose:
        src = "official 1 Hz trace" if using_official else "calibrated reconstruction"
        print(f"\n  WLTC Class 3b — source: {src}")
        print(f"  {'phase':<12} {'dist km':>8} {'spec':>8} {'err':>7} "
              f"{'v_max':>7} {'spec':>7}")
        print("  " + "-" * 52)

    total = 0.0
    for name, (t0, t1, dist_spec, vmax_spec) in WLTC_SPEC.items():
        mask = (t_s >= t0) & (t_s < t1)
        v = v_ms[mask]
        dist = float(np.trapezoid(v, dx=1.0)) / 1000.0
        vmax = float(v.max() * 3.6)
        total += dist
        err = (dist - dist_spec) / dist_spec * 100.0
        rows[name] = dict(distance_km=dist, distance_spec=dist_spec,
                          error_pct=err, v_max_kmh=vmax, v_max_spec=vmax_spec)
        if verbose:
            print(f"  {name:<12} {dist:8.3f} {dist_spec:8.3f} {err:+6.1f}% "
                  f"{vmax:7.1f} {vmax_spec:7.1f}")

    err_tot = (total - WLTC_TOTAL_KM) / WLTC_TOTAL_KM * 100.0
    rows["total"] = dict(distance_km=total, distance_spec=WLTC_TOTAL_KM,
                         error_pct=err_tot)
    if verbose:
        print("  " + "-" * 52)
        print(f"  {'TOTAL':<12} {total:8.3f} {WLTC_TOTAL_KM:8.3f} {err_tot:+6.1f}%")
        if not using_official:
            print("  Note: drop the official 1 Hz trace into data/wltp_cycle.csv")
            print("        to remove the remaining reconstruction error.\n")
        else:
            print()
    return rows


# Phase boundaries (seconds) — used for urban/highway subsetting
WLTP_PHASES = {
    "urban":    (0,    589),
    "suburban": (589,  1022),
    "rural":    (1022, 1477),
    "motorway": (1477, 1800),
    "full":     (0,    1800),
    # Composite contexts for user selection
    "city":     (0,    589),          # alias for urban
    "highway":  (1022, 1800),         # rural + motorway
    "mixed":    (0,    1800),         # full cycle
}


# ════════════════════════════════════════════════════════════════
# FUEL CONSUMPTION CALCULATOR
# ════════════════════════════════════════════════════════════════

@dataclass
class FuelResult:
    """Fuel consumption and CO₂ output for a given Cd and drive context."""
    Cd:               float
    context:          str
    distance_km:      float
    fuel_aero_L:      float    # fuel used purely for aerodynamic drag
    fuel_total_L:     float    # total fuel (aero + rolling resistance)
    L_per_100km_aero: float    # aero drag contribution to fuel consumption
    L_per_100km:      float    # total fuel consumption
    CO2_per_100km_kg: float    # CO₂ emitted per 100 km
    fuel_type:        str


def compute_fuel_consumption(Cd: float,
                              A_frontal_m2: float,
                              mass_kg: float,
                              fuel_type: str = "petrol",
                              context: str = "mixed") -> FuelResult:
    """
    Integrate drag force over the WLTP drive cycle to compute fuel consumption.

    Args:
        Cd            : drag coefficient
        A_frontal_m2  : frontal area in m²
        mass_kg       : vehicle kerb weight in kg
        fuel_type     : "petrol" | "diesel" | "hybrid"
        context       : "city" | "highway" | "mixed" | "urban" |
                        "suburban" | "rural" | "motorway"

    Returns:
        FuelResult with L/100km aero and total
    """
    t_s, v_ms = build_wltp_cycle()

    # Subset to driving context
    if context not in WLTP_PHASES:
        context = "mixed"
    t_start, t_end = WLTP_PHASES[context]
    mask = (t_s >= t_start) & (t_s < t_end)

    v     = v_ms[mask]
    t_sub = t_s[mask]

    if len(v) == 0:
        raise ValueError(f"No data for context '{context}'")

    # ── Aerodynamic drag force and power ─────────────────────────
    # F_drag = ½ρv²CdA  [N]
    # P_drag = F_drag × v  [W]
    F_drag = 0.5 * RHO_AIR * v**2 * Cd * A_frontal_m2
    P_drag = F_drag * v

    # ── Rolling resistance force and power ────────────────────────
    # F_roll = Cr × m × g  (constant — independent of speed)
    F_roll = CR * mass_kg * GRAVITY
    P_roll = F_roll * v

    # ── Integration over cycle (trapezoidal rule, dt=1s) ─────────
    dt = 1.0   # seconds (1 Hz sampling)

    E_aero_J = float(np.trapezoid(P_drag, dx=dt))   # joules from aero drag
    E_roll_J = float(np.trapezoid(P_roll, dx=dt))   # joules from rolling
    E_total_J = E_aero_J + E_roll_J

    # ── Distance covered in this context ─────────────────────────
    distance_m  = float(np.trapezoid(v, dx=dt))
    distance_km = distance_m / 1000.0

    if distance_km < 0.01:
        raise ValueError(f"Effectively zero distance in context '{context}'")

    # ── Convert energy to fuel volume ────────────────────────────
    # Fuel energy needed = mechanical energy / η_engine
    # (engine converts only η fraction of fuel energy to shaft work)
    eta = ETA.get(fuel_type, 0.35)
    e_d = E_DENSITY.get(fuel_type, 34.2e6)

    fuel_aero_L  = (E_aero_J  / eta) / e_d
    fuel_total_L = (E_total_J / eta) / e_d

    # ── Normalise to L/100km ──────────────────────────────────────
    L_per_100km_aero  = fuel_aero_L  / distance_km * 100
    L_per_100km_total = fuel_total_L / distance_km * 100

    # ── CO₂ ───────────────────────────────────────────────────────
    co2_factor = CO2_PER_LITRE.get(fuel_type, 2.31)
    CO2_per_100km = fuel_total_L / distance_km * 100 * co2_factor

    return FuelResult(
        Cd=Cd,
        context=context,
        distance_km=distance_km,
        fuel_aero_L=fuel_aero_L,
        fuel_total_L=fuel_total_L,
        L_per_100km_aero=L_per_100km_aero,
        L_per_100km=L_per_100km_total,
        CO2_per_100km_kg=CO2_per_100km,
        fuel_type=fuel_type,
    )


# ════════════════════════════════════════════════════════════════
# SAVINGS CALCULATOR
# ════════════════════════════════════════════════════════════════

@dataclass
class SavingsResult:
    """Fuel and CO₂ savings from a drag reduction."""
    car_name:           str
    context:            str
    Cd_baseline:        float
    Cd_modified:        float
    delta_Cd:           float
    delta_L_per_100km:  float    # fuel saved per 100 km
    delta_pct:          float    # % fuel reduction (aero component)
    annual_fuel_L:      float    # fuel saved per year (15,000 km)
    annual_cost_INR:    float    # money saved per year (petrol price India)
    lifetime_CO2_kg:    float    # CO₂ avoided over car lifetime
    lifetime_CO2_t:     float    # same in tonnes


# Average petrol price in India 2024 (₹/litre) — Indian Oil Corp bulletin
PETROL_PRICE_INR = 103.0
# Average annual mileage India — Ministry of Road Transport survey 2023
ANNUAL_KM = 15000
# Average car lifetime — SIAM (Society of Indian Automobile Manufacturers)
CAR_LIFETIME_YR = 12


def compute_savings(car_key: str,
                    Cd_baseline: float,
                    Cd_modified: float,
                    context: str = "mixed") -> SavingsResult:
    """
    Compute fuel savings from reducing Cd_baseline → Cd_modified.

    All figures are for Indian driving conditions:
        Annual km   : 15,000 km (MoRTH survey 2023)
        Petrol price: ₹103/litre (Indian Oil Corp, 2024 average)
        Lifetime    : 12 years (SIAM average)
    """
    car_info  = INDIAN_CARS[car_key]
    fuel_type = car_info.get("fuel_type", "petrol")
    A_ref     = car_info["frontal_area_m2"]
    mass_kg   = car_info["kerb_weight_kg"]
    car_name  = car_info["display_name"]

    base = compute_fuel_consumption(Cd_baseline, A_ref, mass_kg,
                                    fuel_type, context)
    mod  = compute_fuel_consumption(Cd_modified, A_ref, mass_kg,
                                    fuel_type, context)

    delta_L_100km = base.L_per_100km - mod.L_per_100km
    delta_pct     = delta_L_100km / base.L_per_100km_aero * 100

    annual_fuel_L = delta_L_100km / 100 * ANNUAL_KM
    annual_cost   = annual_fuel_L * PETROL_PRICE_INR

    # CO₂ avoided over lifetime
    co2_factor    = CO2_PER_LITRE.get(fuel_type, 2.31)
    lifetime_fuel = annual_fuel_L * CAR_LIFETIME_YR
    lifetime_co2  = lifetime_fuel * co2_factor

    return SavingsResult(
        car_name=car_name,
        context=context,
        Cd_baseline=Cd_baseline,
        Cd_modified=Cd_modified,
        delta_Cd=Cd_baseline - Cd_modified,
        delta_L_per_100km=delta_L_100km,
        delta_pct=delta_pct,
        annual_fuel_L=annual_fuel_L,
        annual_cost_INR=annual_cost,
        lifetime_CO2_kg=lifetime_co2,
        lifetime_CO2_t=lifetime_co2 / 1000,
    )


# ════════════════════════════════════════════════════════════════
# FULL PIPELINE: car + modifications → fuel savings
# ════════════════════════════════════════════════════════════════

def full_pipeline(car_key: str,
                  mod_list: list,
                  context: str = "mixed") -> dict:
    """
    Run the complete pipeline for one car and one modification set.

    Args:
        car_key  : key from INDIAN_CARS (e.g. "maruti_swift")
        mod_list : list of (function, kwargs) tuples for apply_mod_set
        context  : "city" | "highway" | "mixed"

    Returns:
        dict with baseline, mod_set, savings, and per-context breakdown
    """
    baseline = solve_car(car_key)
    ms       = apply_mod_set(baseline, mod_list)

    savings  = compute_savings(car_key,
                               baseline['Cd'], ms.Cd_final,
                               context)

    # Per-context breakdown (always compute all three for comparison)
    contexts = {}
    for ctx in ["city", "highway", "mixed"]:
        contexts[ctx] = compute_savings(car_key,
                                        baseline['Cd'], ms.Cd_final, ctx)

    return dict(baseline=baseline, mod_set=ms,
                savings=savings, contexts=contexts)


# ════════════════════════════════════════════════════════════════
# PRINT HELPERS
# ════════════════════════════════════════════════════════════════

def print_fuel_result(fr: FuelResult):
    print(f"  Context         : {fr.context}")
    print(f"  Distance        : {fr.distance_km:.2f} km")
    print(f"  Cd              : {fr.Cd:.4f}")
    print(f"  Aero fuel use   : {fr.L_per_100km_aero:.3f} L/100km")
    print(f"  Total fuel use  : {fr.L_per_100km:.3f} L/100km")
    print(f"  CO₂             : {fr.CO2_per_100km_kg:.3f} kg/100km")


def print_savings(s: SavingsResult):
    print(f"\n  {'─'*60}")
    print(f"  {s.car_name}  [{s.context}]")
    print(f"  {'─'*60}")
    print(f"  Baseline Cd       : {s.Cd_baseline:.4f}")
    print(f"  Modified Cd       : {s.Cd_modified:.4f}  (ΔCd = -{s.delta_Cd:.4f})")
    print(f"  Fuel saved        : {s.delta_L_per_100km:.3f} L/100km")
    print(f"  Aero fuel saved   : {s.delta_pct:.1f}% of aero component")
    print(f"  Annual saving     : {s.annual_fuel_L:.1f} L/yr  "
          f"(≈ ₹{s.annual_cost_INR:,.0f}/year)")
    print(f"  Lifetime CO₂ saved: {s.lifetime_CO2_t:.2f} tonnes  "
          f"({s.lifetime_CO2_kg:.0f} kg over {CAR_LIFETIME_YR} years)")


def print_context_comparison(contexts: dict, car_name: str):
    print(f"\n  Context comparison — {car_name}")
    print(f"  {'Context':<12} {'ΔL/100km':>10} {'Annual L':>10} "
          f"{'Annual ₹':>12} {'CO₂ (t)':>10}")
    print(f"  {'─'*58}")
    for ctx, s in contexts.items():
        print(f"  {ctx:<12} {s.delta_L_per_100km:>10.3f} "
              f"{s.annual_fuel_L:>10.1f} "
              f"{s.annual_cost_INR:>12,.0f} "
              f"{s.lifetime_CO2_t:>10.2f}")


# ════════════════════════════════════════════════════════════════
# DEMO
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n  LAYER 3 — WLTP DRIVE CYCLE FUEL CONSUMPTION")
    print("  =" * 35)

    # Build and inspect the cycle
    t_s, v_ms = build_wltp_cycle()
    distance_total = np.trapezoid(v_ms, dx=1.0) / 1000
    v_avg = v_ms.mean() * 3.6
    print(f"\n  WLTP cycle loaded:")
    print(f"    Duration        : {len(t_s)} seconds (1800s = 30 min)")
    print(f"    Total distance  : {distance_total:.2f} km  (spec: 23.27 km)")
    print(f"    Average speed   : {v_avg:.1f} km/h")
    print(f"    Max speed       : {v_ms.max()*3.6:.1f} km/h  (spec: 131.3)")

    # Standard modification set
    STANDARD_MODS = [
        (mod_underbody_panel, dict(coverage_fraction=0.70)),
        (mod_rear_diffuser,   dict(angle_deg=5.0, length_norm=0.12)),
        (mod_wheel_covers,    dict(n_wheels_covered=4)),
        (mod_rear_spoiler,    dict(chord_m=0.20, angle_deg=7.0)),
    ]

    print("\n" + "═" * 62)
    print("  INDIAN CAR CASE STUDIES — WLTP fuel savings")
    print("═" * 62)

    for car_key in ["maruti_swift", "hyundai_i20", "tata_nexon",
                    "hyundai_creta"]:
        result = full_pipeline(car_key, STANDARD_MODS, context="mixed")

        ms   = result['mod_set']
        base = result['baseline']
        s    = result['savings']

        print_savings(s)
        print_context_comparison(result['contexts'],
                                 INDIAN_CARS[car_key]['display_name'])
        print()