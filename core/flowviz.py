"""
Flow visualisation — show people what the solver actually computes.
Road Car Aerodynamic Fuel Efficiency Engine
============================================

The rest of this project prints numbers. This module draws the flow, because
"the wake is where your fuel goes" lands in one picture and in zero equations.

Everything drawn here comes from THIS project's own panel solution — the
velocity field is evaluated from the same source strengths that produce the
Cd numbers, not from stock footage or a canned CFD image. That matters: the
picture is evidence, not decoration.

Honesty note, visible in the figure itself: potential flow is only valid where
the flow is attached. Downstream of the separation point the panel solution
would show tidy attached streamlines that do not exist on a real car, so the
wake region is masked, hatched, and labelled as the zone where the solver
switches to its base-pressure model. Streamlines are drawn only where the
mathematics is actually meaningful.

Outputs:
    output/flow_<car>.png    — annotated flow field (README hero image)
    output/flow_<car>.gif    — animated particles (the moving version)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from matplotlib.path import Path as MplPath

from core.geometry import build_profile, panel_geometry
from core.panel_solver import (get_car_params, solve, INDIAN_CARS, V_REF)


# ── palette ──────────────────────────────────────────────────────────────────
INK = "#1F2430"           # body fill + primary text
INK_MUTED = "#5A6472"     # secondary text
WAKE_FACE = "#AAB2BD"     # neutral grey — the wake carries no "speed" meaning
ACCENT = "#C2410C"        # single accent for the separation marker only
SURFACE = "#FFFFFF"

# The pressure field is a TRUE POLARITY — positive pressure pushes the panel,
# suction pulls it, and zero is physically meaningful — so it gets a diverging
# two-hue palette with a neutral midpoint (orange = pressure, blue = suction).
# This is what makes the "coloured CFD look" legitimate here: every colour
# carries a sign, unlike the rainbow maps it superficially resembles.
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
CP_CMAP = LinearSegmentedColormap.from_list(
    "cp_div", ["#2F6DB3", "#7FB8E8", "#F4F2EE", "#F0A268", "#D8622B"])


def field_velocity(pg: dict, sigma: np.ndarray, V_inf: float,
                   px: np.ndarray, py: np.ndarray):
    """
    Velocity induced at arbitrary field points by the solved source sheet,
    plus the freestream. Same Hess-Smith formulas the solver itself uses on
    the surface — evaluated off-body, so the streamlines ARE the solution.
    """
    dx = px[:, None] - pg["xa"][None, :]
    dy = py[:, None] - pg["ya"][None, :]
    cos_p = np.cos(pg["phi"])[None, :]
    sin_p = np.sin(pg["phi"])[None, :]
    ds = pg["ds"][None, :]

    X = dx * cos_p + dy * sin_p
    Y = -dx * sin_p + dy * cos_p
    r1_sq = X ** 2 + Y ** 2
    r2_sq = (X - ds) ** 2 + Y ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        u_loc = np.log(np.sqrt(np.maximum(r1_sq, 1e-12)
                               / np.maximum(r2_sq, 1e-12))) / (2 * np.pi)
    v_loc = (np.arctan2(Y, X - ds) - np.arctan2(Y, X)) / (2 * np.pi)

    u_glo = u_loc * cos_p - v_loc * sin_p
    v_glo = u_loc * sin_p + v_loc * cos_p

    u = V_inf + np.nan_to_num(u_glo) @ sigma
    v = np.nan_to_num(v_glo) @ sigma
    return u, v


def _wake_polygon(coords, meta, sep_idx, pg, x_max):
    """
    The separated region, drawn honestly: it starts at the separation point,
    covers the base, and relaxes downstream. The boundary slope (~7 degrees)
    is the classic near-wake spreading angle of a turbulent free shear layer —
    indicative, and labelled as a model region, not as computed flow.
    """
    x_sep, y_sep = pg["xc"][sep_idx], pg["yc"][sep_idx]
    y_bot = meta["y_base_bot"]
    spread = np.tan(np.radians(7.0))

    top = [(x_sep, y_sep)]
    xs = np.linspace(1.0, x_max, 12)
    for x in xs:
        top.append((x, y_sep + (x - x_sep) * spread * 0.6))
    bot = [(x, max(y_bot - (x - 1.0) * spread * 0.4, 0.02)) for x in xs[::-1]]
    bot.append((1.0, y_bot))

    # close along the rear of the body: base face then up the tail of the
    # upper surface to the separation point
    n_up = meta["n_upper"]
    tail = [(pg["xc"][i], pg["yc"][i])
            for i in range(n_up + meta["n_base"] - 1, sep_idx, -1)
            if pg["xc"][i] >= x_sep - 1e-9]
    return np.array(top + bot + tail)


def render_flow(car_key: str = "maruti_swift", n_panels: int = 500,
                save_png: str = None, save_gif: str = None,
                gif_seconds: float = 4.0, verbose: bool = True):
    """Render the annotated flow field (PNG) and particle animation (GIF)."""
    params, car = get_car_params(car_key)
    result = solve(params, n_panels=n_panels)
    coords, meta, pg = result["coords"], result["meta"], result["pg"]
    sigma, sep = result["sigma"], result["sep_idx"]

    # ── grid over the domain (normalised by car length) ─────────────────────
    x_lo, x_hi, y_lo, y_hi = -0.55, 2.1, -0.42, 0.85
    nx, ny = 260, 130
    gx = np.linspace(x_lo, x_hi, nx)
    gy = np.linspace(y_lo, y_hi, ny)
    GX, GY = np.meshgrid(gx, gy)

    u, v = field_velocity(pg, sigma, V_REF, GX.ravel(), GY.ravel())
    U = u.reshape(GY.shape) / V_REF
    V = v.reshape(GY.shape) / V_REF
    speed = np.hypot(U, V)

    # masks: inside the body, and inside the wake (where potential flow is
    # not the truth and must not be drawn as if it were)
    body_path = MplPath(coords)
    wake_poly = _wake_polygon(coords, meta, sep, pg, x_hi)
    wake_path = MplPath(wake_poly)
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    in_body = body_path.contains_points(pts).reshape(GY.shape)
    in_wake = wake_path.contains_points(pts).reshape(GY.shape)
    U_draw = np.where(in_body | in_wake, np.nan, U)
    V_draw = np.where(in_body | in_wake, np.nan, V)

    # The stagnation marker: the global argmax of Cp can land on the windshield
    # root (a genuine secondary stagnation region on a real car), which makes a
    # label that says "the nose" point at the glass. Restrict to the front 25%
    # of the body so the marker names the primary stagnation point.
    front = np.where(pg["xc"] < 0.25)[0]
    i_stag = int(front[np.argmax(result["Cp"][front])])
    x_stag = float(pg["xc"][i_stag])
    y_stag = float(pg["yc"][i_stag])
    x_sep, y_sep = float(pg["xc"][sep]), float(pg["yc"][sep])

    def draw_scene(ax):
        ax.set_facecolor(SURFACE)
        ax.fill(wake_poly[:, 0], wake_poly[:, 1], facecolor=WAKE_FACE,
                alpha=0.30, hatch="///", edgecolor=WAKE_FACE, linewidth=0.0,
                zorder=2)
        ax.fill(coords[:, 0], coords[:, 1], color=INK, zorder=4)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    # ═════════════════════════ static PNG ═══════════════════════════════════
    if save_png:
        fig, ax = plt.subplots(figsize=(12.5, 6.2))
        draw_scene(ax)

        # pressure field: Cp = 1 - (v/V)^2, masked where the mathematics
        # is not valid (body interior, separated wake)
        Cp_field = 1.0 - (U ** 2 + V ** 2)
        Cp_masked = np.ma.masked_where(in_body | in_wake, Cp_field)
        norm = TwoSlopeNorm(vmin=-1.2, vcenter=0.0, vmax=1.0)
        pc = ax.pcolormesh(GX, GY, Cp_masked, cmap=CP_CMAP, norm=norm,
                           shading="gouraud", zorder=1, alpha=0.9)

        # streamlines now carry DIRECTION only: colouring them by speed would
        # re-encode the same information as the pressure field (Cp = 1 - v^2),
        # and two ramps for one quantity is one ramp too many.
        ax.streamplot(gx, gy, U_draw, V_draw, color="#5A6472",
                      density=1.5, linewidth=0.7, arrowsize=0.8, zorder=3)

        cbar = fig.colorbar(pc, ax=ax, shrink=0.75, pad=0.01)
        cbar.set_label("Cp — orange: pressure (pushes) · blue: suction (pulls)",
                       color=INK_MUTED, fontsize=9)
        cbar.ax.tick_params(labelsize=8, colors=INK_MUTED)
        cbar.outline.set_visible(False)

        # annotations — each names one piece of physics in plain words. The
        # backing box exists because the text now sits on a coloured field.
        box = dict(facecolor=SURFACE, alpha=0.78, edgecolor="none",
                   boxstyle="round,pad=0.35")
        kw = dict(fontsize=9.5, color=INK, zorder=6, bbox=box,
                  arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.9))
        ax.plot([x_stag], [y_stag], "o", ms=7, color=INK, mec=SURFACE,
                mew=1.2, zorder=6)
        ax.annotate("STAGNATION\nair hits the nose and stops —\norange = pressure pushing back",
                    xy=(x_stag, y_stag), xytext=(-0.50, 0.62), **kw)
        ax.annotate("air SPEEDS UP over the roof —\nblue = suction (pulls the car up)",
                    xy=(0.45, meta["H"] + 0.06), xytext=(0.18, 0.74), **kw)
        ax.plot([x_sep], [y_sep], "o", ms=7, color=ACCENT, mec=SURFACE,
                mew=1.2, zorder=6)
        ax.annotate("SEPARATION\nthe flow lets go of the body here\n(Stratford criterion, computed)",
                    xy=(x_sep, y_sep), xytext=(1.28, 0.70),
                    fontsize=9.5, color=ACCENT, zorder=6, bbox=box,
                    arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.9))
        # keep the wake label clear of the colorbar on the right edge:
        # short lines, anchored just aft of the base
        ax.text(1.06, 0.10,
                "THE WAKE\n"
                "low-pressure dead air the car\n"
                "drags along — ~60% of the fuel\n"
                "you burn on aero.\n"
                "(hatched = beyond potential flow;\n"
                "modelled at base pressure, not\n"
                "drawn as fake streamlines)",
                fontsize=9.5, color=INK_MUTED, zorder=6)
        ax.annotate("underbody: rough channel,\nwhere panels and skirts work",
                    xy=(0.5, meta["clearance_norm"] - 0.02),
                    xytext=(0.30, -0.33), **kw)

        ax.set_title(
            f"What the solver sees — {car['display_name']}   "
            f"(computed flow field, Cd = {result['Cd']:.3f}, "
            f"Cpb = {result['Cpb']:.2f})",
            fontsize=12, color=INK, pad=12)
        fig.text(0.02, 0.015,
                 f"Streamlines: this project's own 2D panel solution "
                 f"({n_panels} source panels). Every drag number in the "
                 f"README descends from this field.",
                 fontsize=8, color=INK_MUTED)
        plt.tight_layout()
        fig.savefig(save_png, dpi=150, bbox_inches="tight",
                    facecolor=SURFACE)
        plt.close(fig)
        if verbose:
            print(f"  flow figure -> {save_png}")

    # ═════════════════════════ animated GIF ═════════════════════════════════
    if save_gif:
        from scipy.interpolate import RegularGridInterpolator
        Ui = RegularGridInterpolator((gy, gx), np.where(in_body, 0.0, U),
                                     bounds_error=False, fill_value=1.0)
        Vi = RegularGridInterpolator((gy, gx), np.where(in_body, 0.0, V),
                                     bounds_error=False, fill_value=0.0)

        fps, n_frames = 18, int(gif_seconds * 18)
        dt = 0.012
        rng = np.random.default_rng(3)
        n_part = 130

        def spawn(n):
            return np.column_stack([
                np.full(n, x_lo + 0.01),
                rng.uniform(y_lo + 0.02, y_hi - 0.02, n)])

        # stagger starts so the stream looks continuous from frame one
        pos = spawn(n_part)
        for _ in range(rng.integers(5, 60)):
            pass
        warm = rng.integers(0, 140, n_part)
        for i in range(n_part):
            for _ in range(int(warm[i])):
                p = pos[i]
                vel = np.array([Ui((p[1], p[0])), Vi((p[1], p[0]))])
                pos[i] = p + vel * dt

        trail_len = 7
        trails = [pos.copy() for _ in range(trail_len)]

        fig, ax = plt.subplots(figsize=(10, 5))
        draw_scene(ax)
        ax.set_title(f"Air over a {car['display_name'].split('(')[0].strip()} "
                     f"— computed by this repository", fontsize=11, color=INK)
        scats = [ax.scatter([], [], s=(3.5 * (k + 1) / trail_len) ** 2,
                            color=SPEED_CMAP(0.45 + 0.5 * k / trail_len),
                            zorder=5, edgecolors="none")
                 for k in range(trail_len)]

        def step(_frame):
            nonlocal pos
            vel = np.column_stack([Ui(np.column_stack([pos[:, 1], pos[:, 0]])),
                                   Vi(np.column_stack([pos[:, 1], pos[:, 0]]))])
            pos = pos + vel * dt
            # recycle particles that leave, stall in the body, or hit the wake
            gone = ((pos[:, 0] > x_hi - 0.02)
                    | body_path.contains_points(pos)
                    | wake_path.contains_points(pos))
            if gone.any():
                pos[gone] = spawn(int(gone.sum()))
            trails.pop(0)
            trails.append(pos.copy())
            for k, sc in enumerate(scats):
                sc.set_offsets(trails[k])
            return scats

        anim = manimation.FuncAnimation(fig, step, frames=n_frames, blit=True)
        anim.save(save_gif, writer=manimation.PillowWriter(fps=fps),
                  savefig_kwargs=dict(facecolor=SURFACE))
        plt.close(fig)
        if verbose:
            print(f"  flow animation -> {save_gif}")

    return result


# ════════════════════════════════════════════════════════════════
# WEB EXPORT — the interactive explorer
#
# The browser cannot run the panel solver, but it does not need to: the
# SOLUTION (source strengths + geometry) is tiny, and the velocity field it
# implies can be precomputed here on a grid, quantised to int16, and shipped
# inside a single self-contained HTML file. The page then advects particles
# through bilinear interpolation of that field at 60 fps. Every pixel of
# motion in the browser descends from this repository's solver — the page is
# the solver's output, not an artist's impression of it.
# ════════════════════════════════════════════════════════════════

import base64
import json

WEB_GRID = dict(x_lo=-0.60, x_hi=2.20, y_lo=-0.50, y_hi=0.90, nx=240, ny=120)
Q_SCALE = 8000.0     # int16 quantisation: v/V_inf in [-4, 4] at 1.25e-4 steps

# Every car in the database, so the explorer offers the same fleet as the
# recommender. Grid is trimmed slightly (240x120) to keep ten embedded fields
# to a sensible page weight.
#
# Note the honest limitation this makes visible: the 2D field can barely tell
# cars WITHIN a class apart (the Swift, i20, Altroz and Baleno fields look
# nearly identical), because a side silhouette does not carry the plan-view
# detail that separates them. That sameness is not a bug in the viewer; it is
# the documented boundary of a 2D method, shown rather than hidden.
def _web_configs():
    from core.panel_solver import INDIAN_CARS
    short = {"hatchback": "hatch", "sedan": "sedan", "suv": "SUV"}
    out = []
    for key, info in INDIAN_CARS.items():
        name = info["display_name"].split("(")[0].strip()
        out.append((key, f"{name} ({short[info['archetype']]})", key, {}))
    return out


WEB_CONFIGS = _web_configs()


def _b64_i16(arr: np.ndarray) -> str:
    return base64.b64encode(
        np.clip(np.round(arr * Q_SCALE), -32767, 32767)
        .astype("<i2").tobytes()).decode("ascii")


def _b64_u8(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.uint8).tobytes()).decode("ascii")


def export_web(html_path: str = "web/flow_explorer.html",
               n_panels: int = 500, verbose: bool = True) -> dict:
    # 500 panels = the same resolution the validation table is computed at,
    # so the Cd on the interactive page equals the Cd in the README. A draft
    # exported at 400 and the Swift read 0.350 on screen against 0.344 in the
    # validation table — two "official" numbers for one car is how trust dies.
    """
    Solve every WEB_CONFIG, precompute its velocity field on the WEB_GRID, and
    splice the data into html_path between the FLOW_DATA markers.

    The HTML file is the interactive deliverable; this function only replaces
    the payload, so the page's code and its data stay in one reviewable file.
    """
    g = WEB_GRID
    gx = np.linspace(g["x_lo"], g["x_hi"], g["nx"])
    gy = np.linspace(g["y_lo"], g["y_hi"], g["ny"])
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([GX.ravel(), GY.ravel()])

    payload = dict(grid=g, qscale=Q_SCALE, configs=[])

    for cid, label, car_key, overrides in WEB_CONFIGS:
        params, car = get_car_params(car_key)
        params = dict(params, **overrides)
        result = solve(params, n_panels=n_panels)
        pg, meta, coords = result["pg"], result["meta"], result["coords"]
        sep = result["sep_idx"]

        u, v = field_velocity(pg, result["sigma"], V_REF,
                              GX.ravel(), GY.ravel())
        U, V = u / V_REF, v / V_REF

        body_path = MplPath(coords)
        wake_poly = _wake_polygon(coords, meta, sep, pg, g["x_hi"])
        mask = np.zeros(len(pts), dtype=np.uint8)
        mask[MplPath(wake_poly).contains_points(pts)] = 2
        mask[body_path.contains_points(pts)] = 1
        U[mask == 1] = 0.0
        V[mask == 1] = 0.0

        payload["configs"].append(dict(
            id=cid, label=label,
            display=car["display_name"],
            u=_b64_i16(U), v=_b64_i16(V), mask=_b64_u8(mask),
            body=[[round(float(x), 4), round(float(y), 4)]
                  for x, y in coords[::2]],
            wake=[[round(float(x), 4), round(float(y), 4)]
                  for x, y in wake_poly],
            stag=[round(float(pg["xc"][int(np.argmax(result["Cp"]))]), 4),
                  round(float(pg["yc"][int(np.argmax(result["Cp"]))]), 4)],
            sep=[round(float(pg["xc"][sep]), 4),
                 round(float(pg["yc"][sep]), 4)],
            Cd=round(result["Cd"], 4),
            Cpb=round(result["Cpb"], 3),
            budget={k.replace("Cd_", ""): round(result[k], 4)
                    for k in ("Cd_pressure", "Cd_friction", "Cd_underbody",
                              "Cd_wheels", "Cd_cooling", "Cd_mirrors")},
        ))
        if verbose:
            print(f"  exported {label:20s} Cd={result['Cd']:.3f} "
                  f"Cpb={result['Cpb']:+.2f} sep_x={pg['xc'][sep]:.2f}")

    blob = json.dumps(payload, separators=(",", ":"))
    with open(html_path, encoding="utf-8") as f:
        html = f.read()
    start = html.index("/*FLOW_DATA_START*/")
    end = html.index("/*FLOW_DATA_END*/")
    html = (html[:start] + "/*FLOW_DATA_START*/const FLOW_DATA="
            + blob + ";" + html[end:])
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    if verbose:
        print(f"  spliced {len(blob)/1024:.0f} KB of solver output "
              f"into {html_path}")
    return payload


def render_banner(car_key: str = "maruti_swift",
                  save_path: str = "output/banner.png",
                  n_panels: int = 500, verbose: bool = True):
    """
    The repository banner — and it is not decoration: it is the Swift's actual
    computed pressure field, the same array the drag numbers come from,
    rendered in an iridescent palette on black. Suction runs cyan-to-blue,
    pressure runs magenta-to-orange, and the wisps are real streamlines.
    A banner that IS the physics beats any stock artwork, and nobody else's
    repo can have it.
    """
    params, car = get_car_params(car_key)
    result = solve(params, n_panels=n_panels)
    coords, meta, pg = result["coords"], result["meta"], result["pg"]
    sigma, sep = result["sigma"], result["sep_idx"]

    x_lo, x_hi, y_lo, y_hi = -0.85, 2.55, -0.28, 0.62
    nx, ny = 640, 172
    gx = np.linspace(x_lo, x_hi, nx)
    gy = np.linspace(y_lo, y_hi, ny)
    GX, GY = np.meshgrid(gx, gy)
    u, v = field_velocity(pg, sigma, V_REF, GX.ravel(), GY.ravel())
    U, V = (u / V_REF).reshape(GY.shape), (v / V_REF).reshape(GY.shape)
    Cp = 1.0 - (U ** 2 + V ** 2)

    body_path = MplPath(coords)
    wake_poly = _wake_polygon(coords, meta, sep, pg, x_hi)
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    in_body = body_path.contains_points(pts).reshape(GY.shape)
    in_wake = MplPath(wake_poly).contains_points(pts).reshape(GY.shape)

    # iridescent-on-black: cyan/blue suction pole, magenta/orange pressure pole
    iri = LinearSegmentedColormap.from_list("iri", [
        "#1FD4E8", "#2E8FE0", "#123B8C", "#0A0716",
        "#5B1E7A", "#C22B8A", "#FF6B3D"])
    norm = TwoSlopeNorm(vmin=-1.3, vcenter=0.0, vmax=1.0)
    Cp_draw = np.ma.masked_where(in_body, np.where(in_wake, -0.05, Cp))

    fig, ax = plt.subplots(figsize=(16, 4.0))
    fig.patch.set_facecolor("#07060C")
    ax.set_facecolor("#07060C")
    ax.pcolormesh(GX, GY, Cp_draw, cmap=iri, norm=norm, shading="gouraud")
    U_s = np.where(in_body | in_wake, np.nan, U)
    V_s = np.where(in_body | in_wake, np.nan, V)
    ax.streamplot(gx, gy, U_s, V_s, color="#F4F1FA", density=1.3,
                  linewidth=0.35, arrowsize=0.0)
    ax.fill(coords[:, 0], coords[:, 1], color="#07060C", zorder=4)
    ax.plot(coords[:, 0], coords[:, 1], color="#4A4458", lw=1.0, zorder=5)
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("auto"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(save_path, dpi=160, facecolor="#07060C")
    plt.close(fig)
    if verbose:
        print(f"  banner -> {save_path}")


if __name__ == "__main__":
    import os
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs("output", exist_ok=True)
    render_flow("maruti_swift",
                save_png="output/flow_maruti_swift.png",
                save_gif="output/flow_maruti_swift.gif")
    render_banner()
    if os.path.isfile("web/flow_explorer.html"):
        export_web()