<img src="output/readme_banner.png" alt="Physics-First Aerodynamic Fuel Optimizer — point a phone at any car and find the parts that cut its fuel bill; real fluid physics, computer vision, every number tested" width="100%">

# Physics-First Aerodynamic Fuel Optimizer

![Python](https://img.shields.io/badge/python-3.10+-836035?style=flat-square&labelColor=161009) ![Physics](https://img.shields.io/badge/method-panel_method-E68A2B?style=flat-square&labelColor=161009) ![Computer vision](https://img.shields.io/badge/AI-computer_vision-E68A2B?style=flat-square&labelColor=161009) ![Tests](https://img.shields.io/badge/tests-408_passing-AEBB74?style=flat-square&labelColor=161009) ![License](https://img.shields.io/badge/license-MIT-5A4227?style=flat-square&labelColor=161009)

**Point a phone at any car and find the bolt-on parts that cut its fuel bill** — which modification, at what geometry, on that specific car, actually reduces drag, by how much, at what cost, and whether it's legal to fit. A computer-vision front-end reads the car's shape from a photo; a first-principles fluid-physics engine does the rest. No black-box models — every number traces to a fluid-dynamics equation, a measured dimension, or a public standard, and every claim is enforced by a test.

**Try it (no install):** [the recommender](web/recommend.html) · [the flow explorer](web/flow_explorer.html) — self-contained web apps that run in the browser.

---

## Why it exists

A car's aerodynamics is fixed at the factory and never revisited, yet aerodynamic drag burns 40–50% of its fuel at highway speed (Hucho, *Aerodynamics of Road Vehicles*, 1998). India runs one of the world's largest vehicle fleets — 300M+ registered vehicles ([MoRTH, 2023](https://morth.nic.in/)) — and a large aftermarket sells spoilers, diffusers and body kits with **no physics guidance**; fitted wrongly they *increase* drag and fuel use. This tool works out which change actually helps, before anyone spends money or fabricates anything.

It's free, runs in any phone or laptop browser, needs no login, and works offline once loaded — for owners who have a smartphone but no engineering knowledge and no access to CFD or a wind tunnel. Even a 3–5% aerodynamic saving across a fraction of the fleet avoids millions of tonnes of CO₂ and real money per household each year (SDG 13, SDG 12).

Built to be **checkable, not trusted**: photos are processed on-device and never uploaded; every output carries an uncertainty band; a physical sanity bound blocks impossible claims (no bolt-on part can beat the most aerodynamic production car ever built); legal guidance is flagged *"unverified — confirm with your RTO."* It ships with a road test so you can catch it being wrong.

---

## What it computes

Maruti Suzuki Swift (2024), predicted Cd **0.344** against a published 0.320:

| Tier | Modifications | ΔCd | Fuel saved | Annual | Complexity |
|---|---|---|---|---|---|
| 1 — Minimal | Wheel covers | 0.011 ± 0.003 | 0.06 ± 0.02 L/100km | ₹976 ± 309/yr | 0.05 |
| 2 — Moderate | Wheel covers + underbody panel | 0.035 ± 0.010 | 0.21 ± 0.06 L/100km | ₹3,233 ± 902/yr | 0.30 |
| 3 — Aggressive | + rear diffuser | 0.057 ± 0.015 | 0.34 ± 0.09 L/100km | ₹5,215 ± 1,401/yr | 0.90 |

The headline pick is the **underbody panel**: no moving parts, no angle to tune, the best return on effort in the catalogue — and, with wheel covers, one of only two modifications an owner can fit without RTO approval. The physics answer and the legal answer coincide.

Every figure carries an error bar, propagated from the published ranges behind each modification plus the drag model's own 10% validation error. ("₹12,010/year" to five significant figures was this project's original headline, and it was produced by a bug — false precision reads as authority, so no user-facing number ships without its uncertainty.)

---

## How the physics works

### It has to be correct first

**A closed body in attached potential flow has exactly zero drag** — d'Alembert's paradox, a theorem, not an approximation. If the solver reports drag on a closed body before any separation model is applied, the solver is wrong, however plausible the Cd it prints.

```
  SOLVER CORRECTNESS — d'Alembert's paradox on a closed body
  Archetype     closure gap   net source  Cd_attached   Cp_max
  sedan            0.00e+00      -0.0018      +0.0057    1.000
  hatchback        0.00e+00      +0.0353      -0.0289    1.000
  suv              0.00e+00      +0.0547      -0.0392    1.000
```

`Cp_max = 1.000` confirms the stagnation point is resolved exactly, and the residual drag converges to zero under mesh refinement. Real drag then comes from exactly one statement: **the flow separates, and the wake sits at a lower pressure than potential flow predicts.** Nothing else creates pressure drag here.

### The drag budget

Total Cd is a sum of named, physically separate components — and **every modification must draw from one of them, and cannot take more than that component contains.** Wheel covers cannot remove more drag than the wheels produce. This single rule is what keeps the model honest.

| Component | Physics | Source |
|---|---|---|
| `Cd_pressure` | Wake / base pressure, from the panel solution | Hoerner Ch.3; Roshko |
| `Cd_friction` | Turbulent flat plate, Cf = 0.074/Re^0.2 | Prandtl |
| `Cd_underbody` | Rough channel: exhaust, sump, suspension | Hucho Ch.4 |
| `Cd_wheels` | Four rotating wheels, from their real frontal area | Cogotti (1983) |
| `Cd_cooling` | Radiator and engine-bay through-flow | Hucho Ch.4 |
| `Cd_mirrors` | Mirrors, roof rails, seals, gaps | Hucho Table 4.1 |

### The modifications

| Modification | Draws from | Mechanism | Constraint |
|---|---|---|---|
| Wheel covers | `Cd_wheels` | Removes spoke turbulence + ventilation drag | — |
| Underbody panel | `Cd_underbody` | Turns a rough channel back into a flat plate | — |
| Side skirts | `Cd_underbody` | Seals the lateral pressure leak under the sills | Ground gap ≥ 50mm |
| Front splitter | `Cd_underbody` | Diverts flow around the car, not under it | Depth ≤ 80mm |
| Rear diffuser | `Cd_pressure` | Bernoulli recovery raises the wake pressure | Angle ≤ 7° ; clearance ≥ 120mm |
| Rear spoiler | `Cd_pressure` | Turns the shear layer in, minus its own drag | Stall angle ≤ 15° |

Every ΔCd is checked against published wind-tunnel ranges in `test/test_modifications.py`; a model that drifts outside what has been measured on a real car fails the suite. Each carries a **confidence tag** (HIGH/MEDIUM/LOW) with the reason — a spoiler is LOW by structure, not taste: its net effect is the small difference of two competing terms, which amplifies relative error.

### From drag to rupees

Drag is integrated over the WLTP drive cycle to get fuel: `F = ½ρv²CdA → P = Fv → E = ∫P dt → litres → L/100km`, over the WLTC Class 3b cycle at 1 Hz, plus rolling resistance → ₹/year → CO₂. A **Pareto optimiser** then evaluates 576 modification combinations against two objectives — maximise fuel saved, minimise installation complexity — and splits the frontier into three tiers.

### Can you legally fit it?

Every recommendation is checked on two axes ([core/compliance.py](core/compliance.py)). **Statutory:** the Motor Vehicles Act 1988, s.52 — a vehicle may not be altered away from its registered specification; splitters, skirts, spoilers and diffusers are flagged **NEEDS RTO APPROVAL**, while wheel covers and the concealed underbody panel pass. **Practical:** a standard Indian speed breaker (IRC:99, 3.7 m × 0.10 m) is a ~17 m arc; the belly clearance to straddle it is the sagitta over half the wheelbase, and the ramp angle bounds a splitter's approach — geometry, no statute needed.

Every legal citation carries a **verification status**: the Motor Vehicles Act reading ships marked `NEEDS_VERIFICATION`, and a test fails if anyone upgrades it without reading the primary source. (An earlier version cited "CMVR Rule 95(1)" for a clearance minimum; the rule number couldn't be verified, so it was replaced by the speed-breaker geometry, which needs no citation to be true.) **None of this is legal advice.**

### Check it against your real car

Everything above is a model; the coastdown kit is a measurement ([core/coastdown.py](core/coastdown.py)). Coast in neutral from ~90 km/h on a flat road while a phone logs GPS speed — the v² term of the deceleration curve **is** the aerodynamics:

```
F = F0 + F1·v + F2·v²        CdA = 2·F2 / ρ
```

This is SAE J2263, the same road-load procedure behind the official WLTP figures. The fitter handles GPS noise, cancels road slope and steady wind, and reports CdA with an honest error bar. In the self-test it plants CdA = 0.702 m² under noise and opposing slopes and recovers **0.704 ± 0.038**. Measure before and after fitting a part and the difference is its real effect — the number every estimate here is only approximating. Protocol and safety notes: [docs/COASTDOWN.md](docs/COASTDOWN.md).

---

## Read any car from a photo

![Photo to geometry to drag, on the Maruti Swift](output/vision_demo.png)

Point a phone at your car, tap seven dots (wheels, roof, nose, tail), and the tool reconstructs its aerodynamic geometry and runs the whole pipeline — for *any* make, not just the ten in the database.

Why this works when "reconstruct a car from a photo" is normally a hard, model-hungry problem: **the solver is parametric.** A car's whole aerodynamic shape is about eight physically-meaningful numbers, so the vision task collapses from "rebuild 3-D geometry" to "measure eight numbers off a silhouette" — classical geometry, no neural network, no training data, no black box. And because [core/geometry.py](core/geometry.py) runs *forward* (parameters → silhouette), the vision layer ([core/vision.py](core/vision.py)) is just its *inverse*, so it validates against ground truth for free:

```
python -m core.vision
  Cd recovered to mean 2.7%, max 6.7% across 10 cars   (body type: 10/10 correct)
```

The vision step adds *less* error than the drag model already carries against published figures. Segmenting a car out of a messy photo — the genuinely hard step — is kept out of the tested core; the robust path is the seven tapped landmarks, which never fail on a cluttered background and port to a few lines of browser JavaScript.

---

## Try it

This is what the solver actually computes — not stock CFD footage, but the same source strengths that produce the Cd numbers:

![Computed flow field around the Maruti Swift](output/flow_maruti_swift.png)

The **[flow explorer](web/flow_explorer.html)** animates it live: ten switchable cars, a hover probe reading local speed and pressure, and the drag budget per car. Keys: <kbd>1</kbd>–<kbd>9</kbd> switch car · <kbd>space</kbd> pause · <kbd>A</kbd> annotations · <kbd>F</kbd> pressure field. The hatched region is drawn honestly — past the separation point, potential flow stops being true, so the picture stops pretending and labels the wake as the base-pressure zone.

The **[recommender](web/recommend.html)** is the owner-facing side: pick a car, enter dimensions, or scan a photo; get the modification set with fuel saved (± error bars), parts cost, payback, and the RTO verdict; then toggle individual parts and watch the airflow, budget and money respond. Both pages are static and self-contained — the solved physics is embedded, so they run with no server, offline once loaded. To publish: repo Settings → Pages → `main` / root.

*(Every image here regenerates from the solver: `python -m core.flowviz`. The banner is the Swift's computed pressure field, not artwork — the aesthetic is the data.)*

---

## Validation & honest limitations

| Car | Predicted Cd | Published Cd | Error |
|---|---|---|---|
| Maruti Suzuki Swift (2024) | 0.344 | 0.320 | +7.4% |
| Hyundai i20 (2023) | 0.335 | 0.300 | +11.7% |
| Tata Altroz (2023) | 0.339 | 0.310 | +9.4% |
| Honda City (2023) | 0.266 | 0.280 | −5.0% |
| Maruti Suzuki Dzire (2024) | 0.282 | 0.300 | −6.1% |
| Tata Nexon (2023) | 0.309 | 0.350 | −11.7% |
| Hyundai Creta (2024) | 0.311 | 0.360 | −13.5% |
| Mahindra Scorpio-N (2023) | 0.307 | 0.420 | −26.9% * |

**RMS error 9.7%, max 13.5%.** \* Scorpio-N is excluded: its "reference" 0.42 is itself a geometry estimate, not a measurement, and it is the only body-on-frame ladder SUV — the boxiest shape, furthest from the archetype set.

- **It is a 2D model, and that is its dominant error.** A side cross-section can't see the plan view: a Nexon and a Swift have nearly the same silhouette, yet the SUV's real Cd is 10–15% higher (roof rails, square corners, flared arches, bigger mirrors, A-pillar vortices). Those are carried as explicit, separately-sourced components rather than a fudge factor, but a 3D solver would compute them directly — which is why SUVs are consistently under-predicted.
- **Three constants are fitted** — `K_3D` (3-D relief), the base-pressure coefficients, and `K_ROUGH` (underbody roughness) — each physically bounded and documented at its definition. Everything else is a measured dimension or a textbook constant.
- **The WLTP cycle is a reconstruction**, calibrated so every phase covers its published distance and peak speed (integrated distance exact at 23.263 km). Drop the official 1 Hz trace into `data/wltp_cycle.csv` and the pipeline uses it.
- **It is a design-exploration tool**, not a substitute for a wind tunnel — but it ships with the coastdown kit so you can check it against your own car.

---

## The 3D engine (`core/aero_3d.py`)

A full 3D source-panel solver for triangulated meshes: exact Hess–Smith constant-strength panels (verified term-by-term against brute-force integration of the source kernel), a **ground plane by the method of images** — the thing that makes car aerodynamics different from aircraft aerodynamics — and a wake model driven by one rule: flow cannot round a convex edge sharper than ~20° (Katz 1995). Meshes are re-oriented programmatically, and the solver refuses to run on an unclosed or inconsistently wound surface.

**Validated where exact validation exists:** potential flow past a sphere (Cp = 1 − 2.25 sin²θ) is reproduced to a mean |ΔCp| of 0.0055, net force at machine zero — d'Alembert to 10⁻¹⁵. **And honest where it isn't:** on the Ahmed body (SAE 840300), the Cd sweep is flat across 25–40° where the experiment has its famous 30° peak (0.378) and collapse — that peak is made by C-pillar vortices, and a source method has no circulation to reproduce it. A test (`test_the_30_degree_miss_is_present_and_documented`) *fails if the sweep ever starts matching at 30°*, because that could only mean a constant had been quietly bent into a lie. Shedding a vortex sheet off the slant edges is the stated next step.

---

## Run it

```bash
pip install -r requirements.txt          # Python 3.10+, NumPy 2.0+

python main.py                           # all cars
python main.py --car maruti_swift        # one car
python main.py --car tata_nexon --legal-only   # only mods needing no RTO approval
python main.py --validate-only           # solver correctness + validation
python -m core.aero_3d                   # 3D solver: sphere + Ahmed sweep
python -m core.coastdown --demo          # measure CdA from a (synthetic) coastdown
python -m core.vision                    # photo → geometry closed-loop check

pytest test/ -q                          # 408 tests
```

```
core/
  panel_solver.py   # source panel method + drag budget          (Layer 1)
  geometry.py       # closed 2D silhouette, arc-length resampled
  modifications.py  # modification physics, budget-constrained    (Layer 2)
  wltp.py           # WLTP drive-cycle integration                (Layer 3)
  optimizer.py      # Pareto frontier, with error bars            (Layer 4)
  compliance.py     # statutory (MVA s.52) + speed-breaker geometry (Layer 0)
  costs.py          # part costs + payback-range arithmetic
  uncertainty.py    # error-bar propagation + confidence tags
  aero_3d.py        # 3D panels + ground effect + wake            (Layer 5)
  vision.py         # photo → car geometry (computer vision)
  coastdown.py      # measure real CdA from a phone GPS log
  flowviz.py        # flow figures, GIF, banner, embedded fields
  recommend_export.py  # precomputes the recommendation-page data
web/   recommend.html · flow_explorer.html   # static, self-contained
docs/  COASTDOWN.md      test/  408 tests      data/  drop the WLTP trace here
```

---

## References

- Katz, J. & Plotkin, A. *Low-Speed Aerodynamics*, 2nd ed. Cambridge (2001) — source panel method
- Stratford, B.S. Prediction of separation of the turbulent boundary layer. *J. Fluid Mech.* 5(1) (1959)
- Hoerner, S.F. *Fluid-Dynamic Drag*, Ch.3 — base pressure · Roshko, A. — bluff-body wake width
- Ahmed, S.R. et al. Time-averaged ground vehicle wake. SAE 840300 (1984)
- Senior, A.E. & Zhang, X. Diffuser-equipped bluff body in ground effect. SAE 2000-01-0354 (2000)
- Hucho, W.H. *Aerodynamics of Road Vehicles*, 4th ed. SAE (1998) — drag breakdown
- Cogotti, A. Aerodynamic characteristics of car wheels. *Int. J. Vehicle Design* (1983)
- UN GTR No.15 (2015, amended 2018) — WLTP cycle · IPCC 2006 Guidelines Vol 2 Table 3.2.1 — CO₂ factors
