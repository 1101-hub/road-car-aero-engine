"""
Uncertainty propagation for user-facing savings figures.
Road Car Aerodynamic Fuel Efficiency Engine
============================================

Why this module exists
----------------------
"₹12,010/year" was this project's original headline number, and it was produced
by a bug. Part of what let that number survive was its five significant figures:
false precision reads as authority. An honest model states how wrong it might
be, on every number a user might act on.

What feeds the band
-------------------
The fuel and money savings are LINEAR in DeltaCd (drag power is 0.5*rho*v^3*Cd*A
and everything downstream divides or multiplies by constants), so the relative
uncertainty on DeltaCd passes straight through to the rupee figure. Two sources
are combined in quadrature, treated as independent:

  1. PER-MODIFICATION MODEL UNCERTAINTY. Each modification's effectiveness
     constant is anchored to a published range, and the half-width of that range
     relative to its centre is the honest error bar on the model. A spoiler is
     worse than a floor panel here for a structural reason: its DeltaCd is the
     small difference of two competing terms (wake-pressure gain minus its own
     profile drag), and differences of similar numbers amplify relative error.

  2. BASELINE SCALE ERROR. The whole drag budget is calibrated against published
     Cd figures with a validation RMS near 10%, and every modification draws a
     fraction of a budget component, so a 10% scale error on the budget is a 10%
     scale error on everything taken from it.

What the band does NOT include: real-world fitment quality, vehicle condition,
driving style, and fuel-price movement. Those belong to the user, not the model.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


BASELINE_SCALE_UNCERTAINTY = 0.10
"""Relative error on the calibrated drag budget — the validation RMS against
published Cd across the car database (9.7% at last calibration, rounded up)."""


# Relative (1-sigma-ish) uncertainty on each modification's DeltaCd, from the
# half-width of its published range relative to the range centre, and a
# confidence tag a user can read without statistics.
MOD_UNCERTAINTY = {
    #                     rel. unc.  confidence  why
    "wheel_covers":    (0.30, "HIGH",
                        "directly measured on production cars (e.g. smooth "
                        "covers on EVs); the physics is simple occlusion"),
    "underbody_panel": (0.35, "HIGH",
                        "well-replicated wind-tunnel range 0.02-0.045 Cd; "
                        "no angles to get wrong at install time"),
    "front_splitter":  (0.50, "MEDIUM",
                        "underbody-flow diversion is modelled, not measured, "
                        "and depends on bumper shape"),
    "side_skirts":     (0.50, "MEDIUM",
                        "seal effectiveness depends on how close to the road "
                        "the skirt actually runs"),
    "rear_diffuser":   (0.50, "MEDIUM",
                        "recovery efficiency and the base-feed fraction are "
                        "modelled; both are geometry-sensitive"),
    "rear_spoiler":    (0.60, "LOW",
                        "net of two competing terms of similar size (wake gain "
                        "minus own drag) — small differences amplify error"),
}


@dataclass
class SavingsBand:
    """A user-facing number with its honest error bar."""
    value: float
    uncertainty: float          # absolute, same units as value

    @property
    def low(self) -> float:
        return max(self.value - self.uncertainty, 0.0)

    @property
    def high(self) -> float:
        return self.value + self.uncertainty

    def fmt(self, unit: str = "", prec: int = 0) -> str:
        return (f"{self.value:,.{prec}f} ± {self.uncertainty:,.{prec}f}{unit}")


def delta_cd_uncertainty(mod_results: list) -> float:
    """
    Absolute uncertainty on the combined DeltaCd of a modification set.

    Args:
        mod_results : list of ModResult (only feasible ones contribute)

    Per-mod terms combine in quadrature (independent models), then the baseline
    scale error is applied to the total, also in quadrature.
    """
    deltas = [(r.mod_name, r.delta_Cd) for r in mod_results if r.feasible]
    if not deltas:
        return 0.0

    per_mod_sq = 0.0
    total = 0.0
    for name, d in deltas:
        rel = MOD_UNCERTAINTY.get(name, (0.50,))[0]
        per_mod_sq += (rel * d) ** 2
        total += d

    scale_sq = (BASELINE_SCALE_UNCERTAINTY * total) ** 2
    return float(np.sqrt(per_mod_sq + scale_sq))


def confidence_of(mod_name: str) -> Tuple[str, str]:
    """(tag, reason) for one modification."""
    _, tag, why = MOD_UNCERTAINTY.get(mod_name, (0.5, "MEDIUM", "no entry"))
    return tag, why


def band(value: float, delta_Cd: float, delta_Cd_unc: float) -> SavingsBand:
    """
    Attach the DeltaCd error bar to any quantity that is linear in DeltaCd
    (L/100km saved, litres per year, rupees per year, lifetime CO2).
    """
    if delta_Cd <= 0:
        return SavingsBand(value=value, uncertainty=abs(value))
    rel = delta_Cd_unc / delta_Cd
    return SavingsBand(value=value, uncertainty=abs(value) * rel)