"""
Road Car Aerodynamic Fuel Efficiency Engine
============================================
A physics-first computational tool for identifying aerodynamic
modifications that reduce fuel consumption in road cars.

Run this file to execute the full pipeline:
    python main.py

Or target a specific car and context:
    python main.py --car maruti_swift --context highway
    python main.py --car tata_nexon --context city
    python main.py --list

Available car keys:
    maruti_swift, hyundai_i20, tata_altroz
    honda_city, maruti_dzire
    tata_nexon, hyundai_creta, mahindra_scorpio_n

Available contexts:
    city      — urban driving (<56 km/h, WLTP low phase)
    highway   — rural + motorway (>97 km/h)
    mixed     — full WLTP cycle (default, most realistic)

Author      : Amulya
Methodology : 2D source panel method + separation model + WLTP integration
              + Pareto optimisation over modification parameter space
Physics refs: Hucho (1998), Katz (1995), Ahmed et al. SAE 840300,
              Senior & Zhang SAE 2000-01-0354, UN GTR No.15 (WLTP)
"""

import sys
import os
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np

# The banner and the physics notation use box-drawing characters and Greek
# letters. On Windows the console defaults to cp1252, which cannot encode them,
# and the program died with UnicodeEncodeError before printing a single line.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── make sure core/ is importable regardless of working directory ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.panel_solver  import (validate_all, print_dalembert_report,
                                 plot_results, solve_car, get_car_params,
                                 INDIAN_CARS)
from core.compliance    import check_compliance, print_compliance
from core.modifications import (apply_mod_set, mod_underbody_panel,
                                 mod_rear_diffuser, mod_rear_spoiler,
                                 mod_front_splitter, mod_side_skirts,
                                 mod_wheel_covers, print_mod_set)
from core.wltp          import (compute_fuel_consumption, compute_savings,
                                 print_savings, print_context_comparison,
                                 build_wltp_cycle, cycle_report)
from core.optimizer     import (optimize, print_pareto_summary,
                                 plot_pareto, run_grid_search,
                                 pareto_filter, assign_tiers,
                                 pick_representatives)


# ════════════════════════════════════════════════════════════════
# OUTPUT DIRECTORY
# ════════════════════════════════════════════════════════════════

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════
# BANNER
# ════════════════════════════════════════════════════════════════

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║     Road Car Aerodynamic Fuel Efficiency Engine              ║
║     Physics-first · No dataset · No ML                       ║
║     2D Panel Method → WLTP Integration → Pareto Optimiser    ║
╚══════════════════════════════════════════════════════════════╝
"""


# ════════════════════════════════════════════════════════════════
# PIPELINE STAGES
# ════════════════════════════════════════════════════════════════

def stage_1_validate():
    """
    Stage 1 — Prove the solver is correct, then validate it against real cars.

    The d'Alembert check comes FIRST and is not optional. A closed body in
    attached potential flow must produce exactly zero drag. If it does not, the
    panel method is broken and every Cd it reports downstream is meaningless,
    however plausible the number looks. This check is what the original solver
    was missing, and it is what a pair of sign errors had been hiding behind.
    """
    print("\n" + "─" * 64)
    print("  STAGE 1 — Solver correctness, then validation")
    print("─" * 64)

    print_dalembert_report()
    results = validate_all()
    cycle_report()

    path = os.path.join(OUTPUT_DIR, "validation_archetypes.png")
    plot_results(results, save_path=path)
    return results


def stage_2_modifications(car_key: str):
    """
    Stage 2 — Show individual modification physics for one car.
    """
    print("\n" + "─" * 64)
    print(f"  STAGE 2 — Modification Physics: "
          f"{INDIAN_CARS[car_key]['display_name']}")
    print("─" * 64)

    baseline = solve_car(car_key)
    print(f"\n  Baseline Cd: {baseline['Cd']:.4f}  "
          f"(ref: {baseline['Cd_reference']:.3f})")

    mod_list = [
        (mod_underbody_panel, dict(coverage_fraction=0.70)),
        (mod_rear_diffuser,   dict(angle_deg=5.0, length_norm=0.12)),
        (mod_wheel_covers,    dict(n_wheels_covered=4)),
        (mod_rear_spoiler,    dict(chord_m=0.20, angle_deg=7.0)),
        (mod_front_splitter,  dict(depth_mm=50.0)),
        (mod_side_skirts,     dict(height_mm=55.0)),
    ]

    ms = apply_mod_set(baseline, mod_list)
    print_mod_set(ms, baseline['Cd'])
    return baseline, ms


def stage_3_wltp(car_key: str, Cd_modified: float, context: str = "mixed"):
    """
    Stage 3 — Compute WLTP fuel savings for a given Cd reduction.
    """
    print("\n" + "─" * 64)
    print(f"  STAGE 3 — WLTP Fuel Consumption: "
          f"{INDIAN_CARS[car_key]['display_name']}")
    print("─" * 64)

    baseline = solve_car(car_key)
    savings  = compute_savings(car_key, baseline['Cd'], Cd_modified, context)
    print_savings(savings)

    contexts = {}
    for ctx in ["city", "mixed", "highway"]:
        contexts[ctx] = compute_savings(car_key, baseline['Cd'],
                                        Cd_modified, ctx)
    print_context_comparison(contexts, INDIAN_CARS[car_key]['display_name'])
    return savings


def stage_4_optimize(car_key: str, context: str = "mixed",
                     legal_only: bool = False) -> dict:
    """
    Stage 4 — Run the Pareto optimiser for one car.
    """
    print("\n" + "─" * 64)
    print(f"  STAGE 4 — Pareto Optimiser: "
          f"{INDIAN_CARS[car_key]['display_name']}")
    print("─" * 64)

    all_sols = run_grid_search(car_key, verbose=True, legal_only=legal_only)
    pareto   = pareto_filter(all_sols)
    pareto   = assign_tiers(pareto)
    reps     = pick_representatives(pareto)

    print_pareto_summary(car_key, reps)

    fig_path = os.path.join(OUTPUT_DIR, f"pareto_{car_key}.png")
    plot_pareto(car_key, all_sols, pareto, reps, save_path=fig_path)

    return dict(all_solutions=all_sols, pareto=pareto,
                reps=reps, car_key=car_key)


# ════════════════════════════════════════════════════════════════
# WLTP CYCLE VISUALISATION
# ════════════════════════════════════════════════════════════════

def plot_wltp_cycle():
    """Plot the WLTP drive cycle with phase annotations."""
    t, v = build_wltp_cycle()
    v_kmh = v * 3.6

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, v_kmh, color='#1565C0', lw=1.2, label='Speed (km/h)')
    ax.fill_between(t, v_kmh, alpha=0.15, color='#1565C0')

    # Phase bands
    phases = [
        (0,    589,  "#4CAF50", "Low (Urban)"),
        (589,  1022, "#FF9800", "Medium (Suburban)"),
        (1022, 1477, "#F44336", "High (Rural)"),
        (1477, 1800, "#9C27B0", "Extra High (Motorway)"),
    ]
    for t0, t1, col, label in phases:
        ax.axvspan(t0, t1, alpha=0.07, color=col)
        ax.text((t0 + t1) / 2, 138, label, ha='center',
                fontsize=8, color=col, fontweight='bold')

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("WLTP Class 3b Drive Cycle — UN GTR No.15")
    ax.set_xlim(0, 1800)
    ax.set_ylim(0, 150)
    ax.grid(True, alpha=0.3)

    path = os.path.join(OUTPUT_DIR, "wltp_cycle.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  WLTP cycle figure saved → {path}")
    return path


# ════════════════════════════════════════════════════════════════
# FULL CASE STUDY — one car, all stages, clean output
# ════════════════════════════════════════════════════════════════

def run_case_study(car_key: str, context: str = "mixed",
                   legal_only: bool = False):
    """
    Run the complete pipeline for one car and print a
    self-contained case study report.
    """
    car_info = INDIAN_CARS[car_key]
    print(f"\n{'═'*68}")
    print(f"  CASE STUDY: {car_info['display_name']}")
    print(f"  Archetype : {car_info['archetype'].upper()}")
    print(f"  Engine    : {car_info['engine_cc']}cc {car_info['fuel_type']}")
    print(f"  Weight    : {car_info['kerb_weight_kg']} kg")
    print(f"  ARAI rated: city {car_info['city_kmpl']} kmpl / "
          f"highway {car_info['highway_kmpl']} kmpl")
    print(f"{'═'*68}\n")

    # Stages 2 → 3 → 4
    baseline, ms = stage_2_modifications(car_key)
    stage_3_wltp(car_key, ms.Cd_final, context)
    result = stage_4_optimize(car_key, context, legal_only=legal_only)

    # Pick the Tier 2 (moderate) recommendation for the summary
    reps = result['reps']
    rec  = reps.get(2) or reps.get(1) or list(reps.values())[0]

    print(f"\n{'─'*68}")
    print(f"  RECOMMENDED MODIFICATION SET (Tier 2 — Moderate)")
    print(f"{'─'*68}")
    print(f"  Car     : {car_info['display_name']}")
    print(f"  Mods    : {', '.join(rec.mod_names)}")
    print(f"  ΔCd     : -{rec.delta_Cd:.4f} ± {rec.delta_Cd_unc:.4f}")
    print(f"  Savings : {rec.delta_L_100km:.3f} ± {rec.delta_L_unc:.3f} L/100km  "
          f"(highway: {rec.delta_L_highway:.3f} L/100km)")
    print(f"  Annual  : ₹{rec.annual_saving_INR:,.0f} ± "
          f"{rec.annual_saving_unc:,.0f}/year")
    print(f"  CO₂     : {rec.lifetime_CO2_t:.2f} tonnes over 12 years")
    if rec.payback:
        print(f"  Cost    : {rec.payback.cost_str()}  →  pays back in "
              f"{rec.payback.payback_str()}")
    print(f"  Why     : {rec.explanation}")
    print(f"{'─'*68}\n")

    # Stage 5 — can the owner actually fit this, legally and on Indian roads?
    # A recommendation that ignores this is not a recommendation, it is a wish.
    params, _ = get_car_params(car_key)
    rep = check_compliance(car_info, params, rec.mod_names)
    print_compliance(rep)

    return result


# ════════════════════════════════════════════════════════════════
# FULL RUN — all cars
# ════════════════════════════════════════════════════════════════

def run_full(context: str = "mixed", legal_only: bool = False):
    """
    Run the complete pipeline across all Indian cars.
    This is what you run to generate everything for the portfolio.
    """
    print(BANNER)
    t_start = time.time()

    # Stage 1 — validation (run once)
    stage_1_validate()
    plot_wltp_cycle()

    # Case studies — prioritised Indian cars
    priority_cars = [
        "maruti_swift",
        "hyundai_i20",
        "tata_nexon",
        "hyundai_creta",
    ]

    all_results = {}
    for car_key in priority_cars:
        all_results[car_key] = run_case_study(car_key, context, legal_only)

    # ── Fleet summary table ───────────────────────────────────────
    print(f"\n{'═'*68}")
    print("  FLEET SUMMARY — Tier 2 (Moderate) Recommendations")
    print(f"{'═'*68}")
    print(f"  {'Car':<28} {'ΔCd':>8} {'L/100km':>9} "
          f"{'₹/yr':>10} {'CO₂ (t)':>9}")
    print(f"  {'─'*66}")

    for car_key in priority_cars:
        reps    = all_results[car_key]['reps']
        rec     = reps.get(2) or reps.get(1) or list(reps.values())[0]
        name    = INDIAN_CARS[car_key]['display_name']
        short   = name.split('(')[0].strip()
        print(f"  {short:<28} {rec.delta_Cd:>8.4f} "
              f"{rec.delta_L_100km:>9.3f} "
              f"{rec.annual_saving_INR:>10,.0f} "
              f"{rec.lifetime_CO2_t:>9.2f}")

    elapsed = time.time() - t_start
    print(f"\n  All outputs saved to: {OUTPUT_DIR}/")
    print(f"  Total runtime: {elapsed:.1f}s")
    print(f"{'═'*68}\n")


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

def list_cars():
    print(f"\n  Available cars:\n")
    for key, info in INDIAN_CARS.items():
        print(f"    {key:<25} {info['display_name']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Road Car Aerodynamic Fuel Efficiency Engine"
    )
    parser.add_argument(
        "--car", type=str, default=None,
        help="Car key (e.g. maruti_swift). Default: runs all priority cars."
    )
    parser.add_argument(
        "--context", type=str, default="mixed",
        choices=["city", "highway", "mixed"],
        help="Driving context for fuel savings (default: mixed)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available car keys"
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Run solver correctness check and Layer 1 validation only"
    )
    parser.add_argument(
        "--legal-only", action="store_true",
        help="Only recommend modifications an Indian owner can fit WITHOUT RTO "
             "endorsement (Motor Vehicles Act s.52). Excludes every external "
             "body modification. See core/compliance.py."
    )

    args = parser.parse_args()

    if args.list:
        list_cars()
        return

    if args.validate_only:
        print(BANNER)
        stage_1_validate()
        return

    if args.car:
        if args.car not in INDIAN_CARS:
            print(f"\n  Error: '{args.car}' not found. "
                  f"Run with --list to see available cars.\n")
            sys.exit(1)
        print(BANNER)
        run_case_study(args.car, args.context, args.legal_only)
    else:
        run_full(args.context, args.legal_only)


if __name__ == "__main__":
    main()