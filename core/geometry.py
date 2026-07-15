"""
Car silhouette geometry — 2D longitudinal cross-section builder.
Road Car Aerodynamic Fuel Efficiency Engine
============================================

Separated out of panel_solver so that the geometry is built from a car's
ACTUAL dimensions rather than from a generic archetype. The previous version
meshed the archetype and then relabelled the result with the car's numbers,
so every car in a class produced an identical flow solution.

The silhouette is a CLOSED loop traversed clockwise:

    nose --(upper surface)--> base top
                                  |
                            (rear face)      <- the blunt base
                                  |
    nose <--(underbody)------ base bottom

Two things matter for the physics downstream:

  * The loop is exactly closed. A source-panel method conserves mass only on
    a closed contour; an open contour silently violates d'Alembert's paradox
    and produces drag out of nothing.

  * The rear face is BLUNT and explicit. Real cars do not have pointed tails.
    Modelling the base as real geometry means the base area — the single
    biggest driver of car drag — is a derived quantity, not a fitted
    correlation.

Corners are filleted with true circular arcs. A sharp convex corner is a
velocity singularity in potential flow: it produces Cp spikes of -40 that are
pure discretisation artefact. Real cars have radiused A-pillars and roof
trailing edges, and modelling them keeps the pressure field physical.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
from typing import Tuple, Dict


R_BASE_CORNER = 0.025
"""Radius where the body wraps into the rear face, normalised by car length —
about 95 mm on a 4 m car, which is what a real tailgate edge plus bumper wrap
measures.

This is not cosmetic. Left as a mathematically sharp corner, the potential-flow
velocity there is singular. The damage is not local: it destroys the d'Alembert
residual the solver is validated against, and it leaves Cd drifting 15% as the
mesh is refined, because each refinement resolves a sharper and more unphysical
suction spike on the last panels before the corner. A physical radius converges."""


def _fillet(P0, P1, P2, r: float, n: int = 16) -> np.ndarray:
    """
    Circular arc of radius r tangent to segment P0->P1 and segment P1->P2.

    Returns the arc points replacing the sharp corner at P1. If the corner is
    degenerate (collinear or a perfect reversal) the corner is returned as-is.
    """
    P0, P1, P2 = (np.asarray(p, dtype=float) for p in (P0, P1, P2))
    v1, v2 = P0 - P1, P2 - P1
    L1, L2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if L1 < 1e-9 or L2 < 1e-9:
        return np.array([P1])

    d1, d2 = v1 / L1, v2 / L2
    half = np.arccos(np.clip(np.dot(d1, d2), -1.0, 1.0)) / 2.0
    if half < 1e-6 or abs(half - np.pi / 2) < 1e-6:
        return np.array([P1])

    # Shrink the fillet if it would overrun either adjacent segment.
    tan_len = min(r / np.tan(half), 0.45 * L1, 0.45 * L2)
    r_eff = tan_len * np.tan(half)

    T1, T2 = P1 + d1 * tan_len, P1 + d2 * tan_len
    bisector = d1 + d2
    bisector /= np.linalg.norm(bisector)
    centre = P1 + bisector * (r_eff / np.sin(half))

    a1 = np.arctan2(T1[1] - centre[1], T1[0] - centre[0])
    a2 = np.arctan2(T2[1] - centre[1], T2[0] - centre[0])
    if a2 - a1 > np.pi:
        a2 -= 2 * np.pi
    if a1 - a2 > np.pi:
        a2 += 2 * np.pi

    ang = np.linspace(a1, a2, n)
    return np.column_stack([centre[0] + r_eff * np.cos(ang),
                            centre[1] + r_eff * np.sin(ang)])


def _dedupe(pts: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Drop consecutive duplicate vertices. A zero-length panel is a division by
    zero in the influence matrix."""
    keep = [0]
    for i in range(1, len(pts)):
        if np.hypot(*(pts[i] - pts[keep[-1]])) > tol:
            keep.append(i)
    return pts[keep]


def _arclen(pts: np.ndarray) -> np.ndarray:
    """Cumulative arc length along a polyline, starting at zero."""
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    return np.concatenate([[0.0], np.cumsum(seg)])


def _resample(pts: np.ndarray, n: int, cluster: bool = True) -> np.ndarray:
    """
    Resample a polyline to n points, spaced evenly in ARC LENGTH.

    With cluster=True the spacing is cosine-weighted, putting panels where the
    flow needs them — at the stagnation point and at the tail — rather than
    where the x-axis happens to be convenient.
    """
    s = _arclen(pts)
    total = s[-1]
    if total <= 0:
        return pts[:1].repeat(n, axis=0)
    if cluster:
        t = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
    else:
        t = np.linspace(0.0, 1.0, n)
    s_target = t * total
    return np.column_stack([np.interp(s_target, s, pts[:, 0]),
                            np.interp(s_target, s, pts[:, 1])])


def _trim_after(pts: np.ndarray, stop: np.ndarray) -> np.ndarray:
    """Truncate a polyline at the vertex nearest `stop`, then land exactly on it."""
    d = np.hypot(pts[:, 0] - stop[0], pts[:, 1] - stop[1])
    i = int(np.argmin(d))
    return np.vstack([pts[:max(i, 1)], stop])


def _trim_before(pts: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Drop the head of a polyline up to the vertex nearest `start`."""
    d = np.hypot(pts[:, 0] - start[0], pts[:, 1] - start[1])
    i = int(np.argmin(d))
    return np.vstack([start, pts[min(i + 1, len(pts) - 1):]])


def build_profile(params: dict, n_panels: int = 500) -> Tuple[np.ndarray, Dict]:
    """
    Build the closed 2D silhouette from a car's own parameters.

    Args:
        params    : merged parameter dict (car dimensions + archetype shape).
                    Must contain length_m, height_m, windshield_angle_deg,
                    rear_window_angle_deg, trunk_height_norm,
                    underbody_clearance_norm, diffuser_angle_deg,
                    diffuser_length_norm.
        n_panels  : total panel count around the loop.

    Returns:
        coords : (N+1, 2) closed loop of vertices, normalised by car length.
        meta   : derived geometry (body height, base extents, panel indices).
    """
    H = params["height_m"] / params["length_m"]          # normalised height
    y_nose = 0.42 * H                                    # nose stagnation height
    y_hood = 0.62 * H

    wsa = np.radians(params["windshield_angle_deg"])
    rwa = np.radians(params["rear_window_angle_deg"])

    # --- Base (rear face) extents -----------------------------------------
    y_base_top = params["trunk_height_norm"] * H
    clearance = params["underbody_clearance_norm"] * H
    diff_rise = (params["diffuser_length_norm"]
                 * np.tan(np.radians(params["diffuser_angle_deg"])))
    y_base_bot = clearance + diff_rise

    # --- Where the roof starts and ends ------------------------------------
    # Windshield climbs from the hood to roof height at its own rake angle.
    ws_run = (H - y_hood) / np.tan(wsa)
    roof_start = 0.32 + ws_run

    # The backlight drops from roof height to the base top at its rake angle,
    # arriving exactly at the tail. The roof fills whatever is left between.
    rw_run = (H - y_base_top) / np.tan(rwa)
    roof_end = float(np.clip(1.0 - rw_run, roof_start + 0.06, 0.99))

    # ── Build the outline as a POLYLINE, then resample it by ARC LENGTH ────
    #
    # The earlier version sampled the surface as y(x) on a cosine grid in x.
    # That fails on exactly the body this project cares most about: an SUV's
    # backlight rakes at 75 degrees, so dy/dx is nearly infinite there and an
    # x-grid puts almost no panels on it. The SUV came back with Cp = -30 on a
    # surface that should have read about -1.
    #
    # Arc-length sampling is indifferent to how steep a surface is, so a
    # vertical tailgate is resolved exactly as well as a flat roof.

    r_fillet = 0.06 * H / 0.375        # A-pillar / roof-edge radius
    r_base = min(R_BASE_CORNER, 0.30 * (y_base_top - y_base_bot))

    hood_end = (0.32, y_hood)
    roof_a = (roof_start, H)
    roof_b = (roof_end, H)
    tail_top = (1.0, y_base_top)
    tail_bot = (1.0, y_base_bot)

    # Nose and hood: a gentle curve, densely sampled.
    nose_x = np.array([0.0, 0.03, 0.09, 0.19, 0.32])
    nose_y = np.array([y_nose, 0.50 * H, 0.575 * H, 0.605 * H, y_hood])
    f_nose = PchipInterpolator(nose_x, nose_y)   # shape-preserving: no overshoot
    xs_nose = np.linspace(0.0, 0.32, 60)
    nose_pts = np.column_stack([xs_nose, f_nose(xs_nose)])

    # Corner fillets. _fillet() clamps its radius to the adjacent segments, so a
    # short backlight (the SUV's is 0.5% of chord) simply gets a smaller radius
    # instead of a fillet that cuts across it and creates a kink.
    arc_ws = _fillet(hood_end, roof_a, roof_b, r_fillet)
    arc_bl = _fillet(roof_a, roof_b, tail_top, r_fillet)
    arc_top = _fillet(roof_b, tail_top, tail_bot, r_base, n=12)
    arc_bot = _fillet(tail_top, tail_bot, (1.0 - 0.10, y_base_bot), r_base, n=12)

    # Underbody, nose-to-tail then reversed.
    diff_start = 1.0 - params["diffuser_length_norm"]
    x_lo = np.array([0.0, 0.04, 0.10, 0.16, diff_start, 1.0])
    y_lo = np.array([y_nose, 0.14 * H, clearance * 1.05, clearance,
                     clearance, y_base_bot])
    f_lo = PchipInterpolator(x_lo, y_lo)
    xs_lo = np.linspace(0.0, 1.0, 80)
    lower_pts = np.column_stack([xs_lo, f_lo(xs_lo)])[::-1]   # tail -> nose

    # arc_top ends tangent to the rear face, arc_bot starts tangent to it, so the
    # straight face between them is implicit in the polyline. Re-inserting the
    # sharp corner apex between the two arcs makes the path run down the fillet,
    # back UP to the apex and down again — a 180-degree fold that the panel
    # method sees as a real surface.
    upper_poly = _dedupe(np.vstack([nose_pts, arc_ws, arc_bl, [tail_top]]))
    base_poly = _dedupe(np.vstack([arc_top, arc_bot]))
    lower_poly = _dedupe(lower_pts)

    # Trim the upper polyline where the base fillet takes over, and the lower
    # polyline where it hands back.
    upper_poly = _trim_after(upper_poly, base_poly[0])
    lower_poly = _trim_before(lower_poly, base_poly[-1])

    # --- Resample each run by arc length ----------------------------------
    n_base = max(10, int(round(n_panels * 0.12)))
    n_rest = n_panels - n_base
    L_up = _arclen(upper_poly)[-1]
    L_lo = _arclen(lower_poly)[-1]
    n_up = max(20, int(round(n_rest * L_up / (L_up + L_lo))))
    n_lo = max(20, n_rest - n_up)

    # Cosine clustering in arc length: dense at the stagnation point and the tail.
    up_pts = _resample(upper_poly, n_up + 1, cluster=True)
    base_pts = _resample(base_poly, n_base + 1, cluster=False)
    lo_pts = _resample(lower_poly, n_lo + 1, cluster=True)

    coords = _dedupe(np.vstack([up_pts, base_pts[1:], lo_pts[1:]]))
    coords[-1] = coords[0]                       # close the loop exactly

    n_upper = len(up_pts) - 1
    n_base_total = len(base_pts) - 1

    meta = dict(
        H=H,
        y_base_top=y_base_top,
        y_base_bot=y_base_bot,
        base_height=y_base_top - y_base_bot,
        base_height_ratio=(y_base_top - y_base_bot) / H,
        n_upper=n_upper,
        n_base=n_base_total,
        roof_start=roof_start,
        roof_end=roof_end,
        clearance_norm=clearance,
        r_base_corner=r_base,
    )
    return coords, meta


def panel_geometry(coords: np.ndarray) -> dict:
    """
    Per-panel geometry from the closed vertex loop.

    Clockwise traversal means the outward normal is at phi + pi/2:
        upper surface  phi ~ 0     -> beta ~ +pi/2  -> normal points up    (out)
        rear face      phi = -pi/2 -> beta = 0      -> normal points aft   (out)
        underbody      phi ~ pi    -> beta ~ 3pi/2  -> normal points down  (out)
    """
    xa, ya = coords[:-1, 0], coords[:-1, 1]
    xb, yb = coords[1:, 0], coords[1:, 1]
    phi = np.arctan2(yb - ya, xb - xa)
    return dict(
        n=len(xa), xa=xa, ya=ya, xb=xb, yb=yb,
        xc=(xa + xb) / 2.0, yc=(ya + yb) / 2.0,
        ds=np.hypot(xb - xa, yb - ya),
        phi=phi,
        beta=phi + np.pi / 2.0,
    )
