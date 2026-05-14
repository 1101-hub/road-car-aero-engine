"""
Layer 4: Pareto Frontier Optimizer
Road Car Aerodynamic Fuel Efficiency Engine
============================================

Finds the optimal combination and sizing of aerodynamic modifications
for a given car, producing a Pareto frontier of solutions where no
solution can improve fuel savings without increasing modification
complexity or cost.

What a Pareto frontier is:
    A set of solutions where you cannot improve one objective
    (drag reduction) without worsening another (complexity/cost).
    Each point on the frontier is a legitimate "best" answer for a
    different user preference — the engine presents all of them and
    lets the user choose.

Three tiers emerge naturally:
    Tier 1 — Minimal: highest return-on-effort modifications only
    Tier 2 — Moderate: balanced set, good real-world feasibility
    Tier 3 — Aggressive: maximum drag reduction within physical limits

Optimizer approach:
    We use a grid search over modification parameter space rather than
    gradient-based optimization. This is the right choice because:
      1. The parameter space is small (6 modifications, 2-3 params each)
      2. Physical constraints create discontinuities (stall angles, etc.)
         that break gradient methods
      3. Grid search is fully transparent — every solution evaluated is
         visible, nothing is a black box
      4. For a design-exploration tool, coverage matters more than speed

    Total evaluations: ~2000-5000 per car. Runs in seconds.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from itertools import product
from typing import List, Optional

from core.panel_solver import solve_car, INDIAN_CARS
from core.modifications import (
    apply_mod_set,
    mod_underbody_panel, mod_rear_diffuser, mod_rear_spoiler,
    mod_front_splitter, mod_side_skirts, mod_wheel_covers,
    ModSet
)
from core.wltp import compute_savings, full_pipeline, PETROL_PRICE_INR


# ════════════════════════════════════════════════════════════════
# SOLUTION DATACLASS
# ════════════════════════════════════════════════════════════════

@dataclass
class Solution:
    """One point in the design space — a specific modification set."""
    mod_list:         list             # [(function, kwargs), ...]
    mod_names:        list             # human-readable mod names
    delta_Cd:         float            # drag reduction achieved
    Cd_final:         float            # absolute Cd after mods
    delta_L_100km:    float            # fuel saved, L/100km (mixed)
    delta_L_highway:  float            # fuel saved, L/100km (highway)
    annual_saving_INR: float           # ₹/year saved
    lifetime_CO2_t:   float            # tonnes CO₂ avoided
    n_modifications:  int              # number of distinct mods applied
    complexity_score: float            # 0–1, weighted difficulty of install
    explanation:      str              # plain-language summary
    tier:             int = 0          # assigned after Pareto analysis


# ════════════════════════════════════════════════════════════════
# COMPLEXITY SCORING
#   Encodes how difficult each modification is to install on a
#   road car. Used as the second Pareto objective.
#   Scale: 0.0 (trivial) → 1.0 (requires professional fabrication)
# ════════════════════════════════════════════════════════════════

COMPLEXITY = {
    "wheel_covers":    0.05,   # bolt-on, off-the-shelf
    "underbody_panel": 0.20,   # requires jack, basic tools
    "front_splitter":  0.30,   # moderate fabrication
    "rear_spoiler":    0.35,   # bonding + alignment
    "side_skirts":     0.45,   # body-length fitment
    "rear_diffuser":   0.55,   # underbody geometry modification
}


def complexity_score(mod_names: list) -> float:
    """
    Weighted complexity score for a set of modifications.
    Combination penalty: each additional mod adds 5% overhead
    (fitment interactions, alignment).
    """
    if not mod_names:
        return 0.0
    base = sum(COMPLEXITY.get(n, 0.3) for n in mod_names)
    interaction_penalty = 0.05 * max(0, len(mod_names) - 1)
    return min(base + interaction_penalty, 1.0)


# ════════════════════════════════════════════════════════════════
# PARAMETER GRID
#   Each modification is swept over physically meaningful values.
#   "None" means the modification is not applied.
# ════════════════════════════════════════════════════════════════

PARAM_GRID = {
    "wheel_covers": [
        None,
        {"n_wheels_covered": 4},
    ],
    "underbody_panel": [
        None,
        {"coverage_fraction": 0.50},
        {"coverage_fraction": 0.70},
        {"coverage_fraction": 1.00},
    ],
    "front_splitter": [
        None,
        {"depth_mm": 40.0},
        {"depth_mm": 60.0},
    ],
    "rear_spoiler": [
        None,
        {"chord_m": 0.15, "angle_deg": 6.0, "span_fraction": 0.80},
        {"chord_m": 0.25, "angle_deg": 8.0, "span_fraction": 0.85},
    ],
    "side_skirts": [
        None,
        {"height_mm": 50.0, "coverage_fraction": 0.75},
    ],
    "rear_diffuser": [
        None,
        {"angle_deg": 4.0, "length_norm": 0.10},
        {"angle_deg": 5.5, "length_norm": 0.12},
        {"angle_deg": 7.0, "length_norm": 0.14},
    ],
}

MOD_FUNCTIONS = {
    "wheel_covers":    mod_wheel_covers,
    "underbody_panel": mod_underbody_panel,
    "front_splitter":  mod_front_splitter,
    "rear_spoiler":    mod_rear_spoiler,
    "side_skirts":     mod_side_skirts,
    "rear_diffuser":   mod_rear_diffuser,
}


# ════════════════════════════════════════════════════════════════
# GRID SEARCH
# ════════════════════════════════════════════════════════════════

def run_grid_search(car_key: str,
                    verbose: bool = False) -> List[Solution]:
    """
    Enumerate all feasible modification combinations for a car and
    compute fuel savings for each.

    The grid has 2×4×3×3×2×4 = 576 combinations.
    After removing all-None and infeasible sets: ~200-400 evaluated.

    Returns:
        List of Solution objects, one per feasible combination.
    """
    baseline   = solve_car(car_key)
    car_info   = INDIAN_CARS[car_key]
    fuel_type  = car_info.get("fuel_type", "petrol")

    mod_names  = list(PARAM_GRID.keys())
    param_opts = list(PARAM_GRID.values())

    solutions = []
    n_total   = 1
    for opts in param_opts:
        n_total *= len(opts)

    if verbose:
        print(f"  Grid search: {n_total} combinations to evaluate...")

    for combo in product(*param_opts):
        # combo is a tuple of (None or kwargs dict) per modification
        active_mods = [
            (mod_names[i], combo[i])
            for i in range(len(mod_names))
            if combo[i] is not None
        ]

        if not active_mods:
            continue   # skip the all-None case

        # Build mod_list for apply_mod_set
        mod_list = [
            (MOD_FUNCTIONS[name], kwargs)
            for name, kwargs in active_mods
        ]
        active_names = [name for name, _ in active_mods]

        # Apply modifications
        ms = apply_mod_set(baseline, mod_list)

        # Skip if no feasible modifications in set
        if not any(r.feasible for r in ms.modifications):
            continue

        # Only count feasible mods
        feasible_names = [
            r.mod_name for r in ms.modifications if r.feasible
        ]

        if not feasible_names:
            continue

        # Compute fuel savings (mixed context)
        try:
            s_mix = compute_savings(car_key, baseline['Cd'],
                                    ms.Cd_final, "mixed")
            s_hwy = compute_savings(car_key, baseline['Cd'],
                                    ms.Cd_final, "highway")
        except Exception:
            continue

        if s_mix.delta_L_per_100km <= 0:
            continue   # no improvement — skip

        # Build explanation
        mod_strs = []
        for r in ms.modifications:
            if r.feasible:
                p = r.params_used
                if r.mod_name == "wheel_covers":
                    mod_strs.append(
                        f"wheel covers ({p['n_wheels_covered']} wheels)")
                elif r.mod_name == "underbody_panel":
                    mod_strs.append(
                        f"underbody panel ({p['coverage_fraction']*100:.0f}% coverage)")
                elif r.mod_name == "rear_diffuser":
                    mod_strs.append(
                        f"rear diffuser ({p['angle_deg']}°, "
                        f"{p['length_norm']*100:.0f}% length)")
                elif r.mod_name == "rear_spoiler":
                    mod_strs.append(
                        f"rear spoiler ({p['chord_m']*1000:.0f}mm chord, "
                        f"{p['angle_deg']}°)")
                elif r.mod_name == "front_splitter":
                    mod_strs.append(f"front splitter ({p['depth_mm']:.0f}mm)")
                elif r.mod_name == "side_skirts":
                    mod_strs.append(f"side skirts ({p['height_mm']:.0f}mm)")

        explanation = (
            f"{' + '.join(mod_strs)}. "
            f"Saves {s_mix.delta_L_per_100km:.3f} L/100km "
            f"(mixed), {s_hwy.delta_L_per_100km:.3f} L/100km (highway). "
            f"≈₹{s_mix.annual_cost_INR:,.0f}/year saved."
        )

        sol = Solution(
            mod_list=mod_list,
            mod_names=feasible_names,
            delta_Cd=ms.delta_Cd_total,
            Cd_final=ms.Cd_final,
            delta_L_100km=s_mix.delta_L_per_100km,
            delta_L_highway=s_hwy.delta_L_per_100km,
            annual_saving_INR=s_mix.annual_cost_INR,
            lifetime_CO2_t=s_mix.lifetime_CO2_t,
            n_modifications=len(feasible_names),
            complexity_score=complexity_score(feasible_names),
            explanation=explanation,
        )
        solutions.append(sol)

    if verbose:
        print(f"  Feasible solutions found: {len(solutions)}")

    return solutions


# ════════════════════════════════════════════════════════════════
# PARETO FILTERING
#   Objective 1 (maximise): delta_L_100km  (fuel savings)
#   Objective 2 (minimise): complexity_score (install difficulty)
#
#   A solution is Pareto-dominated if another solution has:
#     BOTH higher fuel savings AND lower or equal complexity
# ════════════════════════════════════════════════════════════════

def pareto_filter(solutions: List[Solution]) -> List[Solution]:
    """
    Return only the Pareto-optimal solutions.

    A solution A dominates solution B if:
        A.delta_L_100km >= B.delta_L_100km   (at least as good savings)
        AND
        A.complexity_score <= B.complexity_score  (at least as simple)
        AND at least one inequality is strict.
    """
    pareto = []
    for i, sol_a in enumerate(solutions):
        dominated = False
        for j, sol_b in enumerate(solutions):
            if i == j:
                continue
            # Does sol_b dominate sol_a?
            if (sol_b.delta_L_100km >= sol_a.delta_L_100km and
                sol_b.complexity_score <= sol_a.complexity_score and
                (sol_b.delta_L_100km > sol_a.delta_L_100km or
                 sol_b.complexity_score < sol_a.complexity_score)):
                dominated = True
                break
        if not dominated:
            pareto.append(sol_a)

    # Sort by fuel savings ascending
    pareto.sort(key=lambda s: s.delta_L_100km)
    return pareto


def assign_tiers(pareto: List[Solution]) -> List[Solution]:
    """
    Assign tier labels (1/2/3) to Pareto-optimal solutions.
    Splits the frontier into thirds by fuel savings.
    """
    if not pareto:
        return pareto
    savings = [s.delta_L_100km for s in pareto]
    lo, hi  = min(savings), max(savings)
    span    = hi - lo if hi > lo else 1.0

    for s in pareto:
        frac = (s.delta_L_100km - lo) / span
        if frac < 0.33:
            s.tier = 1   # Minimal
        elif frac < 0.67:
            s.tier = 2   # Moderate
        else:
            s.tier = 3   # Aggressive

    return pareto


# ════════════════════════════════════════════════════════════════
# REPRESENTATIVE SOLUTIONS
#   Pick the best representative from each tier for user output.
# ════════════════════════════════════════════════════════════════

def pick_representatives(pareto: List[Solution]) -> dict:
    """
    From the Pareto frontier, pick one representative per tier.
    Within each tier, pick the solution with lowest complexity
    (easiest to achieve).
    """
    reps = {}
    for tier in [1, 2, 3]:
        tier_sols = [s for s in pareto if s.tier == tier]
        if tier_sols:
            # Lowest complexity in this tier
            reps[tier] = min(tier_sols, key=lambda s: s.complexity_score)
    return reps


# ════════════════════════════════════════════════════════════════
# VISUALISATION
# ════════════════════════════════════════════════════════════════

TIER_COLORS = {1: "#4CAF50", 2: "#FF9800", 3: "#F44336"}
TIER_LABELS = {1: "Tier 1 — Minimal", 2: "Tier 2 — Moderate",
               3: "Tier 3 — Aggressive"}


def plot_pareto(car_key: str, all_solutions: List[Solution],
                pareto: List[Solution], reps: dict,
                save_path: str = None):
    """
    Three-panel figure:
      Left   : Full Pareto frontier (complexity vs fuel savings)
      Centre : Tier representatives — bar chart of savings and CO2
      Right  : Context comparison (city / highway / mixed) for each tier
    """
    car_name = INDIAN_CARS[car_key]["display_name"]
    baseline = solve_car(car_key)

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(
        f"Pareto Frontier — Aerodynamic Modifications: {car_name}",
        fontsize=13, fontweight='bold'
    )
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── Left: Pareto frontier scatter ─────────────────────────────
    ax0 = fig.add_subplot(gs[0])

    # All evaluated solutions (background)
    xs_all = [s.complexity_score for s in all_solutions]
    ys_all = [s.delta_L_100km    for s in all_solutions]
    ax0.scatter(xs_all, ys_all, c='lightgray', s=8, alpha=0.5,
                label='All evaluated', zorder=1)

    # Pareto frontier (colored by tier)
    for tier in [1, 2, 3]:
        pts = [s for s in pareto if s.tier == tier]
        if pts:
            ax0.scatter(
                [s.complexity_score for s in pts],
                [s.delta_L_100km    for s in pts],
                c=TIER_COLORS[tier], s=40, zorder=3,
                label=TIER_LABELS[tier]
            )

    # Draw frontier line
    pf_x = [s.complexity_score for s in pareto]
    pf_y = [s.delta_L_100km    for s in pareto]
    if pf_x:
        order = np.argsort(pf_x)
        ax0.plot(np.array(pf_x)[order], np.array(pf_y)[order],
                 'k--', lw=1.0, alpha=0.5, zorder=2)

    # Star the representatives
    for tier, sol in reps.items():
        ax0.scatter(sol.complexity_score, sol.delta_L_100km,
                    marker='*', s=200, c=TIER_COLORS[tier],
                    edgecolors='black', linewidth=0.8,
                    zorder=5)

    ax0.set_xlabel("Complexity score  (0 = trivial, 1 = major fabrication)")
    ax0.set_ylabel("Fuel saved  (L / 100 km, mixed WLTP)")
    ax0.set_title("Pareto Frontier\n★ = recommended per tier")
    ax0.legend(fontsize=8, loc='lower right')
    ax0.grid(True, alpha=0.3)

    # ── Centre: savings bar chart ─────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    tier_keys  = sorted(reps.keys())
    labels     = [f"Tier {t}" for t in tier_keys]
    savings    = [reps[t].delta_L_100km    for t in tier_keys]
    co2_tonnes = [reps[t].lifetime_CO2_t   for t in tier_keys]
    colors     = [TIER_COLORS[t]            for t in tier_keys]

    x = np.arange(len(tier_keys))
    w = 0.38
    bars1 = ax1.bar(x - w/2, savings, w, label='L/100km saved',
                    color=colors, alpha=0.85)
    ax1_r = ax1.twinx()
    bars2 = ax1_r.bar(x + w/2, co2_tonnes, w, label='Lifetime CO₂ (t)',
                      color=colors, alpha=0.45, hatch='//')

    for bar, val in zip(bars1, savings):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f"{val:.2f}", ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, co2_tonnes):
        ax1_r.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.05,
                   f"{val:.1f}t", ha='center', va='bottom', fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Fuel saved  (L / 100 km)")
    ax1_r.set_ylabel("Lifetime CO₂ avoided  (tonnes)")
    ax1.set_title("Tier Representatives\n(solid = fuel, hatch = CO₂)")
    ax1.grid(True, axis='y', alpha=0.3)

    # ── Right: context comparison ─────────────────────────────────
    ax2  = fig.add_subplot(gs[2])
    ctxs = ["city", "mixed", "highway"]
    ctx_labels = ["City\n(<56 km/h)", "Mixed\n(WLTP full)", "Highway\n(>97 km/h)"]
    x2   = np.arange(len(ctxs))
    width = 0.22
    offsets = np.linspace(-width, width, len(tier_keys))

    for i, tier in enumerate(tier_keys):
        sol = reps[tier]
        ctx_savings = []
        for ctx in ctxs:
            try:
                s = compute_savings(car_key, baseline['Cd'],
                                    sol.Cd_final, ctx)
                ctx_savings.append(s.delta_L_per_100km)
            except Exception:
                ctx_savings.append(0.0)
        ax2.bar(x2 + offsets[i], ctx_savings, width,
                label=f"Tier {tier}", color=TIER_COLORS[tier], alpha=0.85)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(ctx_labels)
    ax2.set_ylabel("Fuel saved  (L / 100 km)")
    ax2.set_title("Savings by Driving Context\n(drag matters more at speed)")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis='y', alpha=0.3)

    path = save_path or f"output/pareto_{car_key}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved → {path}")
    return path


# ════════════════════════════════════════════════════════════════
# PRINT HELPERS
# ════════════════════════════════════════════════════════════════

def print_pareto_summary(car_key: str, reps: dict):
    car_name = INDIAN_CARS[car_key]["display_name"]
    baseline = solve_car(car_key)

    print(f"\n{'═'*68}")
    print(f"  PARETO SUMMARY — {car_name}")
    print(f"  Baseline Cd: {baseline['Cd']:.4f}")
    print(f"{'═'*68}")

    tier_names = {1: "MINIMAL  (easy wins)",
                  2: "MODERATE (balanced)",
                  3: "AGGRESSIVE (maximum reduction)"}

    for tier in sorted(reps.keys()):
        sol = reps[tier]
        print(f"\n  ── Tier {tier}: {tier_names[tier]} ──")
        print(f"  Modifications : {', '.join(sol.mod_names)}")
        print(f"  ΔCd           : -{sol.delta_Cd:.4f}  →  Cd = {sol.Cd_final:.4f}")
        print(f"  Fuel saved    : {sol.delta_L_100km:.3f} L/100km (mixed WLTP)")
        print(f"  Highway saving: {sol.delta_L_highway:.3f} L/100km")
        print(f"  Annual saving : ₹{sol.annual_saving_INR:,.0f}/year")
        print(f"  Lifetime CO₂  : {sol.lifetime_CO2_t:.2f} tonnes avoided")
        print(f"  Complexity    : {sol.complexity_score:.2f}/1.00")
        print(f"  Detail        : {sol.explanation}")

    print(f"\n{'═'*68}\n")


# ════════════════════════════════════════════════════════════════
# MAIN OPTIMIZER FUNCTION
# ════════════════════════════════════════════════════════════════

def optimize(car_key: str,
             plot: bool = True,
             verbose: bool = True) -> dict:
    """
    Full Pareto optimization for one car.

    Returns:
        dict with all_solutions, pareto, reps, car_key
    """
    if verbose:
        print(f"\n  Optimizing: {INDIAN_CARS[car_key]['display_name']}")

    all_sols = run_grid_search(car_key, verbose=verbose)
    pareto   = pareto_filter(all_sols)
    pareto   = assign_tiers(pareto)
    reps     = pick_representatives(pareto)

    if verbose:
        print(f"  Pareto-optimal solutions: {len(pareto)}")
        print_pareto_summary(car_key, reps)

    fig_path = None
    if plot:
        import os
        os.makedirs("output", exist_ok=True)
        fig_path = plot_pareto(car_key, all_sols, pareto, reps)

    return dict(all_solutions=all_sols, pareto=pareto,
                reps=reps, car_key=car_key, fig_path=fig_path)


# ════════════════════════════════════════════════════════════════
# DEMO
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n  LAYER 4 — PARETO OPTIMIZER")
    print("  Running on Maruti Swift and Tata Nexon\n")

    for car_key in ["maruti_swift", "tata_nexon"]:
        result = optimize(car_key, plot=True, verbose=True)