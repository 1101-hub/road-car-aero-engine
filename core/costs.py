"""
Modification costs and payback — the missing decision number.
Road Car Aerodynamic Fuel Efficiency Engine
============================================

An owner does not think in complexity scores; they think "what does it cost,
and when does it pay for itself." This module prices each modification and
computes the payback period against the fuel it saves.

HONESTY, SAME RULES AS THE PHYSICS
----------------------------------
These are MARKET ESTIMATES, not measurements. They are aftermarket / DIY
price ranges for the Indian market (2024-25), deliberately wide, and every
range assumes the owner does the installation themselves except where noted.
Local prices vary by city, brand and workshop; check yours before spending.
The one thing this module refuses to do is print a single confident number:
costs ship as a range, so payback ships as a range.

Payback arithmetic (deliberately pessimistic on both ends):
    best case  = cheapest parts / (annual saving + its uncertainty)
    worst case = priciest parts / (annual saving - its uncertainty)

If the worst case exceeds a car's remaining life (~12 years), the honest
verdict is "may never pay back" and that is what gets printed.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

CAR_LIFETIME_YR = 12.0     # SIAM average — same constant Layer 3 uses


# (low_INR, high_INR, what the money buys)
# DIY installation assumed; a workshop adds Rs 500-2000 per item.
MOD_COST_INR = {
    "wheel_covers": (1200, 3500,
                     "set of 4 smooth/aero wheel covers, aftermarket"),
    "underbody_panel": (2500, 9000,
                        "ABS / aluminium composite sheet + fasteners + "
                        "heat-resistant section near the exhaust; DIY fit"),
    "front_splitter": (2000, 6000,
                       "ABS lip splitter or fabricated ply/composite"),
    "side_skirts": (3000, 9000,
                    "universal or model-specific skirt pair"),
    "rear_spoiler": (2500, 8000,
                     "universal lip spoiler or OEM-style unit, bonded"),
    "rear_diffuser": (4000, 12000,
                      "fabricated underbody diffuser section"),
}


@dataclass
class Payback:
    cost_low_INR: float
    cost_high_INR: float
    payback_low_yr: Optional[float]     # best case; None if saving is zero
    payback_high_yr: Optional[float]    # worst case; None if it never pays back
    pays_back_within_life: bool

    def cost_str(self) -> str:
        return f"₹{self.cost_low_INR:,.0f}–{self.cost_high_INR:,.0f}"

    def payback_str(self) -> str:
        if self.payback_low_yr is None:
            return "no fuel saving — never pays back"
        lo = _fmt_years(self.payback_low_yr)
        if not self.pays_back_within_life:
            return (f"{lo} in the best case; may NOT pay back within the "
                    f"car's remaining life in the worst")
        hi = _fmt_years(self.payback_high_yr)
        return f"{lo}–{hi}"


def _fmt_years(y: float) -> str:
    if y < 1.0:
        return f"{y * 12:.0f} months"
    return f"{y:.1f} yr"


def mod_set_cost(mod_names: list) -> Tuple[float, float]:
    """Total parts cost range for a set of modifications."""
    lo = sum(MOD_COST_INR.get(m, (0, 0, ""))[0] for m in mod_names)
    hi = sum(MOD_COST_INR.get(m, (0, 0, ""))[1] for m in mod_names)
    return float(lo), float(hi)


def compute_payback(mod_names: list,
                    annual_saving_INR: float,
                    annual_saving_unc_INR: float = 0.0) -> Payback:
    """
    Payback period range for a modification set.

    The range is built from the corners that matter to a buyer: the best case
    pairs the cheapest parts with the optimistic end of the saving band; the
    worst case pairs the priciest parts with the pessimistic end. If the
    pessimistic saving is zero or negative, the worst case is "never", and
    pays_back_within_life reports whether even the WORST case clears the
    car's remaining life — the honest bar for "is this worth my money".
    """
    lo, hi = mod_set_cost(mod_names)
    save_hi = annual_saving_INR + annual_saving_unc_INR
    save_lo = annual_saving_INR - annual_saving_unc_INR

    if save_hi <= 0:
        return Payback(lo, hi, None, None, False)

    pb_low = lo / save_hi
    pb_high = (hi / save_lo) if save_lo > 0 else None
    within = pb_high is not None and pb_high <= CAR_LIFETIME_YR
    return Payback(lo, hi, pb_low, pb_high, within)