"""
Coastdown analysis — measure your car's real CdA with a phone.
Road Car Aerodynamic Fuel Efficiency Engine
============================================

Why this module is the most important one in the project
---------------------------------------------------------
Everything else here is a model. This is a measurement. Coast a car in neutral
on a flat, windless road while logging speed against time, and the shape of the
deceleration curve hands you the drag equation of the actual car:

    m_eff * dv/dt = -( F0  +  F1*v  +  F2*v^2 )

    F0  : rolling resistance (tyres, bearings)         ~ constant with speed
    F1  : driveline drag and tyre hysteresis           ~ linear in speed
    F2  : AERODYNAMIC drag = 0.5 * rho * CdA           ~ quadratic in speed

The v-squared coefficient IS the aerodynamics. Fit the curve, read off F2, and

    CdA = 2 * F2 / rho

is a measured property of your car — no panel method, no calibration constant,
no belief required. This is not a hack: it is SAE J2263, and road-load
coefficients determined exactly this way are what feed the official WLTP
procedure this project integrates over.

The killer use: run a coastdown BEFORE and AFTER fitting a modification
(wheel covers are the easy first test — ~Rs 2,000 and ten minutes to fit).
The difference in measured CdA is the modification's real effect on your car,
and it either confirms this tool's prediction or corrects it. Either outcome
is worth more than any simulation in this repository.

Protocol: see docs/COASTDOWN.md. Short version — flat straight road, no wind,
no traffic, warm tyres, coast from ~90 km/h to ~20 km/h in neutral, log speed
at 1 Hz (a GPS logger app or an OBD-II dongle), repeat in BOTH directions and
average (this cancels road slope and any steady wind).

Input format: CSV with two columns, time_s, speed. Speed may be km/h or m/s —
autodetected from magnitude. Header row optional.
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

RHO_AIR = 1.225      # kg/m^3
GRAVITY = 9.81       # m/s^2

ROTATING_MASS_FACTOR = 1.04
"""Effective-mass factor: wheels, brake discs and driveline store rotational
kinetic energy, so the decelerating inertia exceeds the kerb mass by 3-8% in
top gear / neutral. 1.04 is the standard flat assumption for a coastdown in
neutral (SAE J1263 guidance). Stated here rather than buried, because it scales
every fitted force by the same factor."""


# ════════════════════════════════════════════════════════════════
# LOADING AND FITTING
# ════════════════════════════════════════════════════════════════

def load_run(path: str) -> tuple:
    """Load one coastdown CSV -> (t_s, v_ms). Autodetects km/h vs m/s."""
    raw = np.genfromtxt(path, delimiter=",")
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError(f"{path}: expected two columns time_s,speed")
    if np.isnan(raw[0]).any():          # header row
        raw = raw[1:]
    t, v = raw[:, 0].astype(float), raw[:, 1].astype(float)

    # A car coasting from highway speed peaks near 90-110. In m/s that would be
    # a 350 km/h car; in km/h it is a normal one.
    if np.nanmax(v) > 45.0:
        v = v / 3.6
    order = np.argsort(t)
    return t[order], v[order]


@dataclass
class CoastdownFit:
    """Result of fitting one or more coastdown runs."""
    CdA_m2: float                 # measured drag area
    CdA_unc_m2: float             # 1-sigma from the regression
    Crr: float                    # rolling-resistance coefficient
    F0_N: float                   # constant force term
    F1_Ns_per_m: float            # linear force term
    F2_Ns2_per_m2: float          # quadratic term = 0.5*rho*CdA
    mass_kg: float
    n_points: int
    v_range_kmh: tuple
    per_run_CdA: List[float] = field(default_factory=list)

    def Cd(self, frontal_area_m2: float) -> float:
        """Split the measured CdA with a known frontal area."""
        return self.CdA_m2 / frontal_area_m2


def _smooth_derivative(t: np.ndarray, v: np.ndarray, half_window: int = 4):
    """
    dv/dt from noisy GPS speed, by local straight-line fits.

    Differentiating raw 1 Hz GPS speed (noise ~0.1-0.3 km/h) point-to-point
    amplifies the noise catastrophically. A short moving least-squares slope is
    the standard cure; the window is +/-4 s, short against the ~60 s of a run.
    """
    n = len(t)
    a = np.full(n, np.nan)
    vm = np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i - half_window), min(n, i + half_window + 1)
        if hi - lo < 3:
            continue
        ts, vs = t[lo:hi], v[lo:hi]
        A = np.column_stack([ts - ts.mean(), np.ones_like(ts)])
        coef, *_ = np.linalg.lstsq(A, vs, rcond=None)
        a[i] = coef[0]
        vm[i] = vs.mean()
    ok = ~np.isnan(a)
    return vm[ok], a[ok]


def fit_coastdown(runs: List[tuple], mass_kg: float,
                  v_min_kmh: float = 15.0, v_max_kmh: float = 120.0) -> CoastdownFit:
    """
    Fit F = F0 + F1*v + F2*v^2 to one or more coastdown runs and extract CdA.

    Args:
        runs     : list of (t_s, v_ms) tuples — ideally an even number, run in
                   alternating directions so slope and steady wind cancel
        mass_kg  : vehicle kerb weight + driver + fuel (weigh or estimate)

    EACH RUN IS FITTED SEPARATELY and the per-run CdA values are averaged.
    This is not a style choice. A road slope adds a constant force, which
    belongs in that run's own F0 — an uphill run and a downhill run have
    genuinely different F0s. An earlier version pooled every run into one
    regression with a single shared F0; forced to compromise between the two
    true values, the fit leaked the slope error into the v^2 term (the two
    directions cover different speed ranges, so the leak does not cancel), and
    the aero coefficient came back biased. Fitted per run, the slope lands
    harmlessly in F0 and the average of the per-run CdA values is clean.
    """
    m_eff = mass_kg * ROTATING_MASS_FACTOR

    per_coef, per_sigma, per_n = [], [], []
    v_lo, v_hi = np.inf, -np.inf
    for t, v in runs:
        vm, a = _smooth_derivative(np.asarray(t, float), np.asarray(v, float))
        keep = (vm * 3.6 >= v_min_kmh) & (vm * 3.6 <= v_max_kmh) & (a < 0)
        if keep.sum() < 8:
            continue
        coef, cov = _regress(vm[keep], a[keep], m_eff, want_cov=True)
        per_coef.append(coef)
        per_sigma.append(float(np.sqrt(max(cov[2, 2], 0.0))))
        per_n.append(int(keep.sum()))
        v_lo = min(v_lo, float(vm[keep].min() * 3.6))
        v_hi = max(v_hi, float(vm[keep].max() * 3.6))

    if not per_coef:
        raise ValueError("not enough usable coastdown points after filtering "
                         "(need a run from ~90 down to ~20 km/h)")

    coefs = np.array(per_coef)                       # (runs, 3)
    weights = np.array(per_n, float)
    F0, F1, F2 = (coefs * weights[:, None]).sum(axis=0) / weights.sum()

    per_run_cda = list(2.0 * coefs[:, 2] / RHO_AIR)

    # Uncertainty: the spread ACROSS runs (captures wind, slope and traffic
    # differences the per-run regression cannot see) when there are several
    # runs, never reported below the mean per-run regression sigma.
    sigma_reg = 2.0 * float(np.mean(per_sigma)) / RHO_AIR
    if len(per_run_cda) >= 2:
        sigma_spread = float(np.std(per_run_cda, ddof=1) / np.sqrt(len(per_run_cda)))
        cda_unc = max(sigma_spread, sigma_reg)
    else:
        cda_unc = sigma_reg

    return CoastdownFit(
        CdA_m2=2.0 * F2 / RHO_AIR,
        CdA_unc_m2=cda_unc,
        Crr=F0 / (mass_kg * GRAVITY),
        F0_N=float(F0), F1_Ns_per_m=float(F1), F2_Ns2_per_m2=float(F2),
        mass_kg=mass_kg, n_points=int(weights.sum()),
        v_range_kmh=(v_lo, v_hi),
        per_run_CdA=per_run_cda,
    )


def _regress(v, a, m_eff, want_cov: bool = False):
    """Least squares for m_eff * (-a) = F0 + F1 v + F2 v^2."""
    X = np.column_stack([np.ones_like(v), v, v ** 2])
    y = -a * m_eff
    coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    if not want_cov:
        return coef
    dof = max(len(v) - 3, 1)
    s2 = float(res[0]) / dof if len(res) else float(np.sum((y - X @ coef) ** 2)) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    return coef, cov


# ════════════════════════════════════════════════════════════════
# SYNTHETIC RUNS — for the demo and for the test suite
# ════════════════════════════════════════════════════════════════

def synthesize_run(CdA_m2: float = 0.70, Crr: float = 0.012,
                   mass_kg: float = 1000.0, v0_kmh: float = 95.0,
                   noise_kmh: float = 0.25, slope_pct: float = 0.0,
                   seed: int = 0) -> tuple:
    """
    Simulate a coastdown with KNOWN CdA and GPS-like noise. Used by the test
    suite to prove the fitter recovers the truth it was given — the same
    closed-loop honesty test the panel solver gets from the sphere.
    """
    rng = np.random.default_rng(seed)
    m_eff = mass_kg * ROTATING_MASS_FACTOR
    F2 = 0.5 * RHO_AIR * CdA_m2
    F0 = Crr * mass_kg * GRAVITY + mass_kg * GRAVITY * slope_pct / 100.0
    F1 = 0.9                                     # small driveline term

    dt = 0.1
    v = v0_kmh / 3.6
    ts, vs = [0.0], [v]
    t = 0.0
    while v > 4.0 and t < 300.0:
        acc = -(F0 + F1 * v + F2 * v * v) / m_eff
        v += acc * dt
        t += dt
        ts.append(t)
        vs.append(v)

    # sample at 1 Hz like a phone GPS, with noise
    t_arr, v_arr = np.array(ts), np.array(vs)
    t1 = np.arange(0.0, t_arr[-1], 1.0)
    v1 = np.interp(t1, t_arr, v_arr)
    v1 = v1 + rng.normal(0.0, noise_kmh / 3.6, size=len(v1))
    return t1, np.maximum(v1, 0.0)


def write_demo_csv(path: str, **kwargs):
    t, v = synthesize_run(**kwargs)
    header = "time_s,speed_kmh"
    np.savetxt(path, np.column_stack([t, v * 3.6]),
               delimiter=",", header=header, comments="", fmt="%.2f")
    return path


# ════════════════════════════════════════════════════════════════
# REPORT
# ════════════════════════════════════════════════════════════════

def print_report(fit: CoastdownFit, frontal_area_m2: Optional[float] = None,
                 predicted_Cd: Optional[float] = None):
    print("\n" + "=" * 68)
    print("  COASTDOWN RESULT — measured, not modelled")
    print("=" * 68)
    print(f"  Points used      : {fit.n_points}  "
          f"({fit.v_range_kmh[0]:.0f}-{fit.v_range_kmh[1]:.0f} km/h)")
    print(f"  Vehicle mass     : {fit.mass_kg:.0f} kg "
          f"(x{ROTATING_MASS_FACTOR} rotating-inertia factor)")
    print(f"  F0 (rolling)     : {fit.F0_N:.1f} N   ->  Crr = {fit.Crr:.4f}")
    print(f"  F2 (aero)        : {fit.F2_Ns2_per_m2:.4f} N/(m/s)^2")
    print(f"  MEASURED CdA     : {fit.CdA_m2:.3f} ± {fit.CdA_unc_m2:.3f} m^2")
    if fit.per_run_CdA:
        runs = ", ".join(f"{c:.3f}" for c in fit.per_run_CdA)
        print(f"  Per-run CdA      : {runs}")
    if frontal_area_m2:
        cd = fit.Cd(frontal_area_m2)
        print(f"  With A = {frontal_area_m2:.2f} m^2 ->  Cd = {cd:.3f}")
        if predicted_Cd:
            err = (predicted_Cd - cd) / cd * 100.0
            print(f"  Model predicted  : Cd = {predicted_Cd:.3f}  "
                  f"({err:+.1f}% vs your car)")
    print("=" * 68)
    print("  Before/after a modification, the DIFFERENCE in measured CdA is")
    print("  that modification's real effect on your car — the number every")
    print("  simulation in this project is trying to estimate.")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    import sys
    import argparse
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(
        description="Fit coastdown logs -> measured CdA (SAE J2263-style)")
    ap.add_argument("csv", nargs="*", help="coastdown CSV files (time_s,speed)")
    ap.add_argument("--mass", type=float, default=1010.0,
                    help="vehicle mass incl. driver, kg")
    ap.add_argument("--area", type=float, default=None,
                    help="frontal area m^2, to split CdA into Cd")
    ap.add_argument("--demo", action="store_true",
                    help="synthesize a Swift-like run and analyse it")
    args = ap.parse_args()

    if args.demo or not args.csv:
        print("  [demo] synthesizing two opposite-direction runs for a "
              "Swift-like car (true CdA = 0.702 m^2)...")
        runs = [synthesize_run(CdA_m2=0.702, mass_kg=args.mass,
                               slope_pct=+0.3, seed=1),
                synthesize_run(CdA_m2=0.702, mass_kg=args.mass,
                               slope_pct=-0.3, seed=2)]
    else:
        runs = [load_run(p) for p in args.csv]

    fit = fit_coastdown(runs, mass_kg=args.mass)
    print_report(fit, frontal_area_m2=args.area)