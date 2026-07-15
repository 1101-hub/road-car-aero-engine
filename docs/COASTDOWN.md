# Coastdown protocol — measure your car's real CdA

This is the one afternoon that turns this project from a simulation into an
experiment. The method is the industry's own (SAE J2263 road-load
determination — the same procedure that generates the coefficients behind the
official WLTP figures), reduced to what a phone can do.

## Safety first — read before driving

- **Empty road only.** A long, straight, flat stretch with no traffic, no
  pedestrians, no junctions. Early morning on a bypass service road or a
  private test area. You will be decelerating slowly from ~90 km/h with no
  brake lights showing — do not do this anywhere someone can run into you.
- A passenger handles the phone. The driver drives. Nothing else.
- Coasting in neutral means **no engine braking and longer stopping
  distances** — keep your foot over the brake pedal.
- Do not do this in rain, in wind you can feel, or at night.
- You are responsible for complying with local traffic law.

## What you need

- A GPS speed logger app (1 Hz logging, exports CSV), or an OBD-II Bluetooth
  dongle with a logging app. GPS speed is accurate to ~0.2 km/h once moving —
  good enough.
- The car's mass with you in it: kerb weight (from the registration papers)
  + your weight + fuel. Accuracy to ±20 kg is fine.
- A calm day. Wind is the biggest error source; below ~8 km/h wind is usable,
  still air at dawn is best.

## Procedure

1. **Warm up** — 15 minutes of driving so tyres and driveline reach working
   temperature (cold tyres read ~20% higher rolling resistance).
2. **Tyre pressures** to placard values, checked warm.
3. Accelerate to **~95 km/h**, settle, then **shift to neutral** and let the
   car coast. Touch nothing — no brakes, no steering beyond keeping straight,
   windows closed, HVAC off.
4. Log until you're below **~20 km/h**, then pull away safely.
5. **Repeat in the opposite direction** on the same stretch. This is not
   optional: averaging opposite directions cancels road slope and any steady
   wind. Do at least 2 runs each way; 3+ each way is better.
6. Export each run as CSV: two columns, `time_s, speed_kmh` (or m/s — the
   loader autodetects).

## Analysis

```bash
python -m core.coastdown run_north1.csv run_south1.csv run_north2.csv run_south2.csv \
       --mass 1010 --area 2.04
```

The tool fits the road-load equation `F = F0 + F1·v + F2·v²` to your logs and
reports:

- **CdA ± uncertainty** — your car's measured drag area (the v² coefficient is
  the aerodynamics; nothing else grows with speed squared),
- **Crr** — rolling resistance, from the constant term,
- **Cd**, if you pass the frontal area.

No CSV handy? `python -m core.coastdown --demo` synthesizes a noisy Swift-like
run with a known planted CdA and shows the fitter recovering it — the same
closed-loop honesty check the panel solver gets from the analytic sphere.

## The experiment that matters

1. Baseline: 4–6 runs, get CdA₁.
2. Fit **wheel covers** (the cheapest modification, and the only one — with an
   underbody panel — that needs no RTO approval; see `core/compliance.py`).
3. Same day, same road, same tyre pressures: 4–6 runs, get CdA₂.
4. **CdA₁ − CdA₂ is the measured effect of the modification on your car.**

The model in this repository predicts that difference is worth about
ΔCd ≈ 0.011 (≈ 0.022 m² on a Swift) — with the honest caveat that a single
modification's effect sits near the resolution limit of a phone-GPS coastdown,
so take the full set of runs and mind the error bars. Whether the measurement
confirms or corrects the prediction, you'll know something no simulation could
have told you.

## Error sources, honestly

| Source | Size | Mitigation |
|---|---|---|
| Wind | Large — biggest single error | Calm day; both directions; average |
| Road slope | Large if one-way | Both directions, same stretch |
| GPS speed noise | ~0.2 km/h | Windowed slope fitting (done by the tool) |
| Mass estimate | ±2% | Scales all forces equally — ±2% on CdA |
| Rotating inertia | ±3% | Standard 1.04 factor applied and stated |
| Temperature/density | ±2% per 6 °C | Note air temp; ρ = 1.225 assumed |
