"""
Layer 0: Legal and Practical Compliance
Road Car Aerodynamic Fuel Efficiency Engine
============================================

A tool that tells an Indian car owner to bolt a splitter onto their Swift has an
obligation to tell them whether that is legal, and whether it will survive the
first speed breaker. This layer does both.

READ THIS BEFORE TRUSTING ANY CITATION IN THIS FILE
---------------------------------------------------
Legal citations are marked with a VerificationStatus. Nothing here is legal
advice, and the author of this code is not a lawyer.

  VERIFIED          — the author has checked the primary source.
  NEEDS_VERIFICATION— stated from secondary knowledge. CHECK THE PRIMARY SOURCE
                      BEFORE RELYING ON IT OR PUBLISHING IT.

Everything in this file currently ships as NEEDS_VERIFICATION except the
constraints derived from geometry, which are computed from first principles and
do not depend on a citation being right.

This matters. An earlier version of this project asserted a 120 mm road-legal
minimum ground clearance sourced to "CMVR Rule 95(1)". That rule number could
not be verified, and a physics tool that invents a statute to justify a
constraint is doing exactly the thing this project claims not to do. The
constraint itself was sound; the citation was not. It has been replaced by a
requirement DERIVED from the geometry of an Indian speed breaker, which needs no
citation to be true — you can measure it yourself.

The two kinds of limit
----------------------
STATUTORY  — what the law permits. The controlling provision is the Motor
             Vehicles Act 1988, Section 52. Its effect is that a vehicle may not
             be altered so that its particulars no longer match the manufacturer's
             specification recorded at registration. In practice this means most
             external body modifications need RTO endorsement, and some are not
             permitted at all. This is the single largest real-world obstacle to
             everything this tool recommends, and it must be surfaced, not buried.

PRACTICAL  — what the road permits. A diffuser that is legal but grounds out on
             every speed breaker in the city is not a viable modification. These
             limits are computed from the geometry of the obstacle and the car,
             so they are as trustworthy as the dimensions you feed them.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import numpy as np


class VerificationStatus(Enum):
    VERIFIED = "verified against primary source"
    NEEDS_VERIFICATION = "UNVERIFIED — check the primary source before relying on this"
    DERIVED = "derived from geometry — needs no citation"


class Legality(Enum):
    OK = "no statutory obstacle identified"
    NEEDS_RTO_APPROVAL = "likely requires RTO endorsement under MVA s.52"
    LIKELY_PROHIBITED = "likely not permitted without type approval"


@dataclass
class Citation:
    text: str
    source: str
    status: VerificationStatus

    def __str__(self):
        flag = "" if self.status is not VerificationStatus.NEEDS_VERIFICATION else "  [UNVERIFIED]"
        return f"{self.text} ({self.source}){flag}"


# ════════════════════════════════════════════════════════════════
# STATUTORY POSITION
# ════════════════════════════════════════════════════════════════

MVA_SECTION_52 = Citation(
    text=("A motor vehicle may not be altered such that its particulars differ "
          "from those recorded in the certificate of registration, i.e. from the "
          "manufacturer's original specification. External body modifications "
          "therefore generally require RTO endorsement; some are not permitted "
          "at all. Enforcement was tightened by a 2019 Supreme Court ruling."),
    source="Motor Vehicles Act 1988, s.52 (as amended 2019)",
    status=VerificationStatus.NEEDS_VERIFICATION,
)

# How each modification is likely to be treated. This is an ENGINEERING
# JUDGEMENT about how the statute applies, not a legal determination.
MOD_LEGALITY = {
    # Fits within the existing wheel; does not alter a registered particular.
    "wheel_covers": (
        Legality.OK,
        "A wheel cover sits within the existing wheel and changes no dimension "
        "recorded at registration. This is the only modification here with no "
        "obvious statutory obstacle — which is a large part of why it is the "
        "recommended starting point."),

    # Under the car, adds no external dimension.
    "underbody_panel": (
        Legality.OK,
        "A flat floor panel is concealed beneath the vehicle and adds no external "
        "dimension. It does not change the registered particulars. Note it may "
        "affect the manufacturer's warranty and can trap heat around the exhaust "
        "— use heat-resistant material and leave the catalytic converter clear."),

    # These change the external body.
    "front_splitter": (
        Legality.NEEDS_RTO_APPROVAL,
        "A splitter projects forward of the registered front overhang, changing "
        "the vehicle's recorded length. It is also an external projection with "
        "pedestrian-safety implications."),
    "side_skirts": (
        Legality.NEEDS_RTO_APPROVAL,
        "Skirts alter the external body profile below the sills."),
    "rear_spoiler": (
        Legality.NEEDS_RTO_APPROVAL,
        "A spoiler alters the external body and may change the registered height "
        "or length. Factory-fitted or manufacturer-approved accessory spoilers "
        "are generally acceptable; fabricated ones are not."),
    "rear_diffuser": (
        Legality.NEEDS_RTO_APPROVAL,
        "A diffuser alters the underbody and rear bumper geometry, and reduces "
        "the effective ground clearance and departure angle."),
}


# ════════════════════════════════════════════════════════════════
# PRACTICAL LIMIT — THE INDIAN SPEED BREAKER
#
# This is the constraint that actually decides whether a modification survives
# contact with an Indian road, and it is computed, not cited.
# ════════════════════════════════════════════════════════════════

SPEED_BREAKER = Citation(
    text=("Round-topped hump, 3.7 m chord and 0.10 m height. This is the standard "
          "Indian speed-breaker geometry; real ones are frequently worse."),
    source="IRC:99 (Indian Roads Congress), speed-breaker guidelines",
    status=VerificationStatus.NEEDS_VERIFICATION,
)

HUMP_CHORD_M = 3.7
HUMP_HEIGHT_M = 0.10

# Safety margin on every computed clearance: real humps are built to no standard
# at all, and a car carries passengers and luggage that compress the suspension.
CLEARANCE_MARGIN_MM = 25.0

SKIRT_DEBRIS_GAP_MM = 50.0
"""Minimum skirt-to-road gap for debris, potholes and kerbs. An engineering
judgement, not a statute."""


def hump_radius_m() -> float:
    """
    Radius of the circular arc that a standard speed breaker approximates.

    For a circular segment of chord c and height h:
        R = c^2 / (8h) + h/2
    """
    c, h = HUMP_CHORD_M, HUMP_HEIGHT_M
    return c ** 2 / (8.0 * h) + h / 2.0


def belly_clearance_required_mm(wheelbase_m: float) -> float:
    """
    Ground clearance a car needs at mid-wheelbase to straddle a speed breaker
    without grounding its floor.

    With both axles on the hump, the car's belly must clear the crown. The rise
    of the arc over half a wheelbase is the sagitta:

        sag = R - sqrt(R^2 - (wheelbase/2)^2)

    This is pure geometry. It needs no citation and you can verify it with a
    tape measure. It is why a flat underbody panel or a diffuser on a low car is
    a practical problem long before it is a legal one.
    """
    R = hump_radius_m()
    half = wheelbase_m / 2.0
    sag_m = R - np.sqrt(max(R ** 2 - half ** 2, 0.0))
    return float(sag_m * 1000.0 + CLEARANCE_MARGIN_MM)


def hump_ramp_angle_deg() -> float:
    """
    Steepest slope of the speed-breaker face, at its leading edge.

        tan(theta) = (c/2) / sqrt(R^2 - (c/2)^2)

    A car's approach and departure angles must exceed this or the front splitter
    / rear diffuser strikes the hump face.
    """
    R = hump_radius_m()
    half_c = HUMP_CHORD_M / 2.0
    return float(np.degrees(np.arctan2(half_c, np.sqrt(max(R ** 2 - half_c ** 2, 0.0)))))


def approach_angle_deg(overhang_m: float, edge_height_m: float) -> float:
    """Angle from the tyre contact patch up to the lowest point of a front
    (or rear) overhang. Must exceed the hump's ramp angle."""
    if overhang_m <= 0:
        return 90.0
    return float(np.degrees(np.arctan2(edge_height_m, overhang_m)))


# ════════════════════════════════════════════════════════════════
# REPORT
# ════════════════════════════════════════════════════════════════

def legally_unrestricted_mods() -> List[str]:
    """
    Modifications with no identified statutory obstacle — fittable without RTO
    endorsement.

    This is what `--legal-only` restricts the optimiser to, and it is a short
    list: wheel covers and an underbody panel. Everything else alters the
    external body.

    That is not a weakness of the tool, it is the finding. The two modifications
    an Indian owner can actually fit today happen to include the underbody panel,
    which is also the best return on effort in the whole catalogue. The physics
    and the law agree, which is a genuinely useful thing to be able to say.
    """
    return [name for name, (legality, _) in MOD_LEGALITY.items()
            if legality is Legality.OK]


@dataclass
class ComplianceIssue:
    mod_name: str
    kind: str                     # "statutory" | "practical"
    severity: str                 # "blocker" | "approval" | "advisory"
    message: str
    citation: Optional[Citation] = None


@dataclass
class ComplianceReport:
    car_name: str
    issues: List[ComplianceIssue] = field(default_factory=list)
    belly_required_mm: float = 0.0
    belly_actual_mm: float = 0.0
    ramp_angle_deg: float = 0.0

    @property
    def blockers(self):
        return [i for i in self.issues if i.severity == "blocker"]

    @property
    def needs_approval(self):
        return [i for i in self.issues if i.severity == "approval"]

    @property
    def road_legal_as_is(self) -> bool:
        return not self.blockers and not self.needs_approval


def check_compliance(car: dict, params: dict, mod_names: List[str]) -> ComplianceReport:
    """
    Check a set of fitted modifications against statutory and practical limits.

    Args:
        car       : entry from INDIAN_CARS (needs wheelbase_m)
        params    : merged geometry params (needs underbody_clearance_norm, height_m)
        mod_names : names of the modifications actually FITTED (feasible ones only)
    """
    rep = ComplianceReport(car_name=car["display_name"])

    wb = car.get("wheelbase_m")
    clearance_mm = params["underbody_clearance_norm"] * params["height_m"] * 1000.0

    rep.ramp_angle_deg = hump_ramp_angle_deg()
    rep.belly_actual_mm = clearance_mm
    if wb:
        rep.belly_required_mm = belly_clearance_required_mm(wb)

        if clearance_mm < rep.belly_required_mm:
            rep.issues.append(ComplianceIssue(
                mod_name="(vehicle)", kind="practical", severity="advisory",
                message=(f"Ground clearance {clearance_mm:.0f}mm is already below the "
                         f"{rep.belly_required_mm:.0f}mm needed to straddle a standard "
                         f"speed breaker without grounding. Any underbody work makes "
                         f"this worse."),
                citation=SPEED_BREAKER))

    for name in mod_names:
        legality, why = MOD_LEGALITY.get(name, (Legality.NEEDS_RTO_APPROVAL, ""))

        if legality is Legality.NEEDS_RTO_APPROVAL:
            rep.issues.append(ComplianceIssue(
                mod_name=name, kind="statutory", severity="approval",
                message=why, citation=MVA_SECTION_52))
        elif legality is Legality.LIKELY_PROHIBITED:
            rep.issues.append(ComplianceIssue(
                mod_name=name, kind="statutory", severity="blocker",
                message=why, citation=MVA_SECTION_52))
        elif why:
            rep.issues.append(ComplianceIssue(
                mod_name=name, kind="statutory", severity="advisory",
                message=why, citation=None))

        # --- practical: things that hang below the car ---
        if name == "rear_diffuser" and wb:
            rep.issues.append(ComplianceIssue(
                mod_name=name, kind="practical", severity="advisory",
                message=(f"A diffuser reduces the departure angle. The rear edge must "
                         f"stay clear of a {hump_ramp_angle_deg():.1f}-degree hump face "
                         f"and of ramps."),
                citation=SPEED_BREAKER))

        if name == "front_splitter":
            L, wb_ = params["length_m"], (wb or 0.0)
            front_overhang_m = 0.55 * (L - wb_) if wb_ else 0.8
            edge_h = clearance_mm / 1000.0
            appr = approach_angle_deg(front_overhang_m, edge_h)
            ramp = hump_ramp_angle_deg()
            sev = "blocker" if appr < ramp else "advisory"
            rep.issues.append(ComplianceIssue(
                mod_name=name, kind="practical", severity=sev,
                message=(f"Approach angle {appr:.1f} deg against a {ramp:.1f} deg hump "
                         f"face (front overhang ~{front_overhang_m*1000:.0f}mm). "
                         + ("The splitter WILL strike speed breakers."
                            if sev == "blocker" else
                            "Clears a standard hump, but expect contact on steep ramps.")),
                citation=SPEED_BREAKER))

        if name == "side_skirts":
            gap = clearance_mm  # skirt height is checked in modifications.py
            rep.issues.append(ComplianceIssue(
                mod_name=name, kind="practical", severity="advisory",
                message=(f"Skirts must leave at least {SKIRT_DEBRIS_GAP_MM:.0f}mm to the "
                         f"road for debris and kerbs (current ride height {gap:.0f}mm)."),
                citation=None))

    return rep


# ════════════════════════════════════════════════════════════════
# PRINTING
# ════════════════════════════════════════════════════════════════

DISCLAIMER = """
  ------------------------------------------------------------------
  NOT LEGAL ADVICE. This tool computes aerodynamics, not law. The
  statutory position below is an engineering reading of the Motor
  Vehicles Act and is marked UNVERIFIED: confirm it with your RTO
  before modifying a registered vehicle. Fitting an unapproved body
  modification can invalidate your insurance and your registration.
  ------------------------------------------------------------------"""


def print_compliance(rep: ComplianceReport):
    print("\n" + "=" * 72)
    print(f"  COMPLIANCE — {rep.car_name}")
    print("=" * 72)
    print(DISCLAIMER)

    if rep.belly_required_mm:
        ok = "OK" if rep.belly_actual_mm >= rep.belly_required_mm else "TIGHT"
        print(f"\n  Speed breaker (IRC:99, 3.7m x 0.10m hump):")
        print(f"    Ramp face angle        : {rep.ramp_angle_deg:.1f} deg")
        print(f"    Belly clearance needed : {rep.belly_required_mm:.0f} mm  "
              f"(incl. {CLEARANCE_MARGIN_MM:.0f}mm margin)")
        print(f"    This car has           : {rep.belly_actual_mm:.0f} mm   [{ok}]")

    blockers = rep.blockers
    approvals = rep.needs_approval
    advisories = [i for i in rep.issues if i.severity == "advisory"]

    if blockers:
        print(f"\n  BLOCKERS — do not fit:")
        for i in blockers:
            print(f"    [{i.mod_name}] {i.message}")

    if approvals:
        print(f"\n  REQUIRES RTO APPROVAL ({len(approvals)} modification(s)):")
        for i in approvals:
            print(f"    [{i.mod_name}] {i.message}")
        print(f"\n    Basis: {MVA_SECTION_52.source}")
        print(f"    {MVA_SECTION_52.status.value}")

    if advisories:
        print(f"\n  ADVISORY:")
        for i in advisories:
            print(f"    [{i.mod_name}] {i.message}")

    verdict = ("FIT AS-IS — no statutory obstacle identified"
               if rep.road_legal_as_is else
               "NOT fit-and-forget — see above before spending money")
    print(f"\n  Verdict: {verdict}")
    print("=" * 72)
