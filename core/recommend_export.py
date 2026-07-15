"""
Recommendation-page data export.
Road Car Aerodynamic Fuel Efficiency Engine
============================================

Precomputes every car's Pareto recommendations — savings per driving context,
error bars, costs, payback and the legal verdict — and splices them into
web/recommend.html as a JSON payload.

Why static precompute instead of a server (Streamlit et al.): everything a
user can adjust on the page is LINEAR in what is precomputed here. Fuel saved
scales with annual km and with fuel price; payback is a division. A page of
multiplications needs no Python runtime, no hosting account, no cold starts —
it is a file, it works from GitHub Pages forever, and the physics stays in
this repository where the tests are.
"""

import json

from core.panel_solver import solve_car, get_car_params, INDIAN_CARS
from core.optimizer import (run_grid_search, pareto_filter, assign_tiers,
                            pick_representatives)
from core.wltp import compute_savings, PETROL_PRICE_INR, ANNUAL_KM
from core.costs import MOD_COST_INR, CAR_LIFETIME_YR
from core.compliance import check_compliance, MOD_LEGALITY, Legality
from core.uncertainty import confidence_of

MOD_LABELS = {
    "wheel_covers": "Wheel covers",
    "underbody_panel": "Underbody panel",
    "front_splitter": "Front splitter",
    "side_skirts": "Side skirts",
    "rear_spoiler": "Rear spoiler",
    "rear_diffuser": "Rear diffuser",
}

TIER_NAMES = {1: "Start here", 2: "Best value", 3: "Maximum"}


def _tier_payload(car_key: str, sol, baseline_Cd: float) -> dict:
    """One tier -> everything the page needs, per driving context."""
    contexts = {}
    for ctx in ("city", "mixed", "highway"):
        s = compute_savings(car_key, baseline_Cd, sol.Cd_final, ctx)
        rel = (sol.delta_Cd_unc / sol.delta_Cd) if sol.delta_Cd > 0 else 1.0
        contexts[ctx] = dict(
            dL100=round(s.delta_L_per_100km, 4),
            dL100_unc=round(s.delta_L_per_100km * rel, 4),
        )

    mods = []
    for name in sol.mod_names:
        legality, why = MOD_LEGALITY[name]
        tag, reason = confidence_of(name)
        lo, hi, buys = MOD_COST_INR[name]
        mods.append(dict(
            id=name, label=MOD_LABELS[name],
            legal=("ok" if legality is Legality.OK else "rto"),
            legal_why=why,
            confidence=tag, confidence_why=reason,
            cost_lo=lo, cost_hi=hi, cost_buys=buys,
        ))

    return dict(
        name=TIER_NAMES.get(sol.tier, f"Tier {sol.tier}"),
        tier=sol.tier,
        mods=mods,
        dCd=round(sol.delta_Cd, 4),
        dCd_unc=round(sol.delta_Cd_unc, 4),
        Cd_final=round(sol.Cd_final, 4),
        co2_t=round(sol.lifetime_CO2_t, 2),
        contexts=contexts,
        cost_lo=sol.payback.cost_low_INR,
        cost_hi=sol.payback.cost_high_INR,
        legal_as_is=all(m["legal"] == "ok" for m in mods),
    )


def export_recommend(html_path: str = "web/recommend.html",
                     verbose: bool = True) -> dict:
    payload = dict(
        defaults=dict(annual_km=ANNUAL_KM, petrol_inr=PETROL_PRICE_INR,
                      lifetime_yr=CAR_LIFETIME_YR),
        cars=[],
    )

    for key, info in INDIAN_CARS.items():
        baseline = solve_car(key)
        reps = pick_representatives(assign_tiers(pareto_filter(
            run_grid_search(key))))
        legal_reps = pick_representatives(assign_tiers(pareto_filter(
            run_grid_search(key, legal_only=True))))

        # speed-breaker sanity for the card
        params, car = get_car_params(key)
        comp = check_compliance(car, params, [])

        tiers = [_tier_payload(key, reps[t], baseline["Cd"])
                 for t in sorted(reps)]
        # the best fully-legal set, shown as its own headline
        legal_best = max(legal_reps.values(), key=lambda s: s.delta_L_100km)

        payload["cars"].append(dict(
            id=key,
            name=info["display_name"],
            archetype=info["archetype"],
            Cd=round(baseline["Cd"], 3),
            Cd_published=info["reference_Cd"],
            fuel=info.get("fuel_type", "petrol"),
            kerb_kg=info["kerb_weight_kg"],
            belly_ok=bool(comp.belly_actual_mm >= comp.belly_required_mm),
            tiers=tiers,
            legal_best=_tier_payload(key, legal_best, baseline["Cd"]),
        ))
        if verbose:
            lb = "+".join(legal_best.mod_names)
            print(f"  {info['display_name']:32s} tiers={len(tiers)} "
                  f"legal_best={lb}")

    blob = json.dumps(payload, separators=(",", ":"))
    with open(html_path, encoding="utf-8") as f:
        html = f.read()
    start = html.index("/*RECO_DATA_START*/")
    end = html.index("/*RECO_DATA_END*/")
    html = (html[:start] + "/*RECO_DATA_START*/const RECO="
            + blob + ";" + html[end:])
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    if verbose:
        print(f"  spliced {len(blob)/1024:.0f} KB into {html_path}")
    return payload


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    export_recommend()