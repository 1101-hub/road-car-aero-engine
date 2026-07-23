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
from core.wltp import (compute_savings, compute_fuel_consumption,
                       PETROL_PRICE_INR, ANNUAL_KM)
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

# Which drag-BUDGET line each modification actually draws from. This is what
# makes the before/after airflow honest: only pressure-line mods (diffuser,
# spoiler) visibly shrink the WAKE; the rest cut friction/wheel/underbody drag
# that the streamlines barely show, so their effect must be read off the budget
# bar instead of faked as a flow change.
MOD_COMPONENT = {
    "wheel_covers": "Cd_wheels",
    "underbody_panel": "Cd_underbody",
    "side_skirts": "Cd_underbody",
    "front_splitter": "Cd_underbody",
    "rear_diffuser": "Cd_pressure",
    "rear_spoiler": "Cd_pressure",
}
BUDGET_KEYS = ["Cd_pressure", "Cd_friction", "Cd_underbody",
               "Cd_wheels", "Cd_cooling", "Cd_mirrors"]

TIER_NAMES = {1: "Start here", 2: "Best value", 3: "Maximum"}


def _budget_after(base_solve, sol) -> dict:
    """The drag budget after a tier's modifications: subtract each mod's real
    delta from the budget line it draws from. Re-applies the mod set to recover
    the per-mod deltas (the grid search discarded them)."""
    from core.modifications import apply_mod_set
    ms = apply_mod_set(base_solve, sol.mod_list)
    after = {k: float(base_solve[k]) for k in BUDGET_KEYS}
    for r in ms.modifications:
        if r.feasible:
            comp = MOD_COMPONENT.get(r.mod_name)
            if comp:
                after[comp] = max(after[comp] - r.delta_Cd, 0.0)
    return after


def _tier_payload(car_key: str, sol, baseline_Cd: float,
                  base_solve=None) -> dict:
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

    payload = dict(
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
    if base_solve is not None:
        payload["budget_after"] = {k: round(v, 4)
                                   for k, v in _budget_after(base_solve, sol).items()}
    return payload


def _archetype_flow(car_key, nx=200, ny=100):
    """Precompute a particle-advection velocity field + outline for one car,
    so the recommendation page can show live airflow without a solver in the
    browser. Reuses the flow-visualisation engine (core/flowviz)."""
    import numpy as np
    from matplotlib.path import Path as MplPath
    from core import flowviz
    from core.panel_solver import get_car_params, solve, V_REF

    params, car = get_car_params(car_key)
    r = solve(params, n_panels=400)
    pg, sigma, coords, meta, sep = (r["pg"], r["sigma"], r["coords"],
                                    r["meta"], r["sep_idx"])
    g = dict(x_lo=-0.5, x_hi=2.05, y_lo=-0.40, y_hi=0.60, nx=nx, ny=ny)
    gx = np.linspace(g["x_lo"], g["x_hi"], nx)
    gy = np.linspace(g["y_lo"], g["y_hi"], ny)
    GX, GY = np.meshgrid(gx, gy)
    u, v = flowviz.field_velocity(pg, sigma, V_REF, GX.ravel(), GY.ravel())
    U, V = u / V_REF, v / V_REF
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    body = MplPath(coords)
    wake_poly = flowviz._wake_polygon(coords, meta, sep, pg, g["x_hi"])
    mask = np.zeros(len(pts), np.uint8)
    mask[MplPath(wake_poly).contains_points(pts)] = 2
    mask[body.contains_points(pts)] = 1
    U[mask == 1] = 0.0
    V[mask == 1] = 0.0
    return dict(grid=g, qscale=flowviz.Q_SCALE,
                u=flowviz._b64_i16(U), v=flowviz._b64_i16(V),
                mask=flowviz._b64_u8(mask),
                body=[[round(float(x), 4), round(float(y), 4)]
                      for x, y in coords[::3]],
                sep=[round(float(pg["xc"][sep]), 4), round(float(pg["yc"][sep]), 4)])


def export_recommend(html_path: str = "web/recommend.html",
                     verbose: bool = True) -> dict:
    # Fuel-saving slope: litres per 100 km saved, per unit of drag area
    # (CdA, m^2) removed, per driving context and fuel type. The aero fuel
    # burn is exactly linear in CdA, so one number per (context, fuel) is the
    # ENTIRE model a custom car needs — the page multiplies, the physics
    # stays here where the tests are.
    slopes = {}
    for ctx in ("city", "mixed", "highway"):
        slopes[ctx] = {}
        for fuel in ("petrol", "diesel"):
            fr = compute_fuel_consumption(Cd=1.0, A_frontal_m2=1.0,
                                          mass_kg=1000.0, fuel_type=fuel,
                                          context=ctx)
            slopes[ctx][fuel] = round(fr.L_per_100km_aero, 5)

    payload = dict(
        defaults=dict(annual_km=ANNUAL_KM, petrol_inr=PETROL_PRICE_INR,
                      lifetime_yr=CAR_LIFETIME_YR),
        slopes=slopes,
        # Custom-car mode borrows each archetype's representative: the shape
        # (and therefore the modification deltas) comes from the archetype,
        # only the DIMENSIONS are the user's. The page adds an extra 15%
        # uncertainty for that borrowed shape — see the JS.
        arch_reps=dict(hatchback="maruti_swift", sedan="honda_city",
                       suv="tata_nexon"),
        cars=[],
    )

    # Live particle-flow fields, one per body type — the recommendation page
    # shows the user's car's CLASS airflow (all a 2-D model can resolve anyway).
    payload["flow"] = {}
    for arch, rep in payload["arch_reps"].items():
        payload["flow"][arch] = _archetype_flow(rep)
        if verbose:
            print(f"  flow field: {arch}")

    for key, info in INDIAN_CARS.items():
        baseline = solve_car(key)
        reps = pick_representatives(assign_tiers(pareto_filter(
            run_grid_search(key))))
        legal_reps = pick_representatives(assign_tiers(pareto_filter(
            run_grid_search(key, legal_only=True))))

        # speed-breaker sanity for the card
        params, car = get_car_params(key)
        comp = check_compliance(car, params, [])

        tiers = [_tier_payload(key, reps[t], baseline["Cd"], baseline)
                 for t in sorted(reps)]
        # the best fully-legal set, shown as its own headline
        legal_best = max(legal_reps.values(), key=lambda s: s.delta_L_100km)

        payload["cars"].append(dict(
            id=key,
            name=info["display_name"],
            archetype=info["archetype"],
            Cd=round(baseline["Cd"], 3),
            Cd_published=info["reference_Cd"],
            Cpb=round(baseline["Cpb"], 3),
            budget={k: round(float(baseline[k]), 4) for k in BUDGET_KEYS},
            fuel=info.get("fuel_type", "petrol"),
            kerb_kg=info["kerb_weight_kg"],
            belly_ok=bool(comp.belly_actual_mm >= comp.belly_required_mm),
            tiers=tiers,
            legal_best=_tier_payload(key, legal_best, baseline["Cd"], baseline),
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