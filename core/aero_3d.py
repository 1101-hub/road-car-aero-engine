"""
Layer 5: 3D Source Panel Method with Ground Effect and a Wake Model
Road Car Aerodynamic Fuel Efficiency Engine
============================================

What this replaces
------------------
The previous version of this file was a foundation that could not do the job.
It used a POINT-SOURCE approximation for every panel including its immediate
neighbours (badly wrong in the near field), it had no ground plane (which is the
one thing that distinguishes car aerodynamics from aircraft aerodynamics), it ran
O(N^2) in interpreted Python loops (25 s at 1600 triangles, extrapolating to an
hour and 3 GB for a real car mesh), and — decisively — it had NO WAKE MODEL, so
by d'Alembert's paradox it produced exactly zero drag. It computed a pressure
field and could not compute the one number the project exists for.

What this does
--------------
    1. Exact constant-strength source panels on triangles (Hess-Smith)
    2. A ground plane, by the method of images
    3. Separation detection on the rear surface
    4. A base-pressure wake model, so drag is real
    5. Vectorised: the influence matrix is built with NumPy broadcasting

Honest scope — READ THIS
------------------------
This is a SOURCE panel method. Sources model thickness and displacement; they
cannot model circulation. That has one consequence that matters enormously here,
and it must not be hidden:

    This solver CANNOT reproduce the Ahmed body's drag peak at 30 degrees.

The Ahmed body's Cd rises steeply to a maximum near a 30-degree rear slant and
then collapses. That peak is produced by a pair of counter-rotating streamwise
(C-pillar) vortices that roll up along the slant edges and pull the flow down
onto the slant. They are a vortex phenomenon. A source panel method has no
vortices, so it sees a 30-degree slant as simply a less-bluff 0-degree slant and
predicts a smooth, monotonic fall in drag.

`validate_ahmed()` prints this failure side by side with the published data
rather than tuning it away. Reproducing that peak needs a shed vortex sheet off
the slant side edges, which is the next piece of work — and it is exactly the
same physics that the 2D engine's documentation names as the thing a 2D
cross-section cannot see.

So: this layer gives correct inviscid pressure, correct ground effect, and a
believable base drag for a square-backed body. It is not yet a general car-drag
solver, and it says so.

References
----------
    Hess, J.L. & Smith, A.M.O. "Calculation of potential flow about arbitrary
        bodies", Prog. Aerospace Sci. 8 (1967) — the panel influence formulae
    Katz & Plotkin, "Low-Speed Aerodynamics" 2nd ed., App. D
    Ahmed, S.R., Ramm, G., Faltin, G. SAE 840300 (1984) — the benchmark
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

EPS = 1e-10

TURN_LIMIT_DEG = 20.0
"""Maximum convex surface turning angle a turbulent boundary layer will follow
before it separates, applied per edge in find_attached_region().

FIXED FROM LITERATURE, NOT TUNED. Katz ("Race Car Aerodynamics", 1995) puts the
tolerance of an attached turbulent boundary layer to a convex body kink at
roughly 15-20 degrees at automotive Reynolds numbers — the same physics behind
this project's 15-degree spoiler stall limit. 20 degrees is the permissive end
of that range.

Why this is a constant with a citation and not a dial: with this value the
slant stays attached below 20 degrees and separates above, which is the correct
qualitative structure. The quantitative sweep (see validate_ahmed) lands in the
right range everywhere but does not reproduce the experiment's 30-degree
peak-and-collapse — that structure is made by C-pillar vortices, which a
source-only method excludes by construction. Nudging this number until the
numbers 'passed' would manufacture agreement out of vortex physics the solver
does not contain, which is how this project went wrong the first time."""


# ════════════════════════════════════════════════════════════════
# MESH
# ════════════════════════════════════════════════════════════════

@dataclass
class Mesh:
    """A triangulated closed surface."""
    verts: np.ndarray       # (V, 3)
    tris: np.ndarray        # (T, 3) int indices

    @property
    def n(self) -> int:
        return len(self.tris)

    def properties(self):
        """Centroids, unit normals, areas. Normals follow the winding order."""
        p = self.verts[self.tris]                     # (T, 3, 3)
        centroids = p.mean(axis=1)
        cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        norm = np.linalg.norm(cross, axis=1)
        areas = 0.5 * norm
        safe = np.maximum(norm, EPS)[:, None]
        normals = cross / safe
        return centroids, normals, areas

    def check_closed(self) -> dict:
        """
        A source panel method is only valid on a CLOSED, CONSISTENTLY WOUND
        surface. If the mesh has holes or flipped triangles, the solve is
        garbage and there is no way to tell from the answer alone.

        Every interior edge must be traversed exactly twice, once in each
        direction. Anything else means a hole or an inconsistent winding.
        """
        edges = {}
        for tri in self.tris:
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                key = (min(a, b), max(a, b))
                edges.setdefault(key, []).append(1 if a < b else -1)

        boundary = sum(1 for v in edges.values() if len(v) == 1)
        nonmanifold = sum(1 for v in edges.values() if len(v) > 2)
        flipped = sum(1 for v in edges.values() if len(v) == 2 and sum(v) != 0)

        # Signed volume: positive for outward-facing normals (divergence theorem)
        c, nrm, a = self.properties()
        volume = float(np.sum(np.einsum("ij,ij->i", c, nrm) * a) / 3.0)

        return dict(boundary_edges=boundary, nonmanifold_edges=nonmanifold,
                    flipped_edges=flipped, volume=volume,
                    closed=(boundary == 0 and nonmanifold == 0 and flipped == 0),
                    outward=(volume > 0))

    def oriented(self) -> "Mesh":
        """
        Repair the winding so every triangle is consistent and every normal
        points outward. Returns a new Mesh.

        Why this exists: the Ahmed mesher writes its quad windings by hand, one
        face at a time, and hand-written windings are exactly the kind of thing
        a human gets subtly wrong — the first version shipped with 116 flipped
        edges and a NEGATIVE enclosed volume, meaning a whole face's normals
        pointed into the body. On such a mesh the boundary condition has the
        wrong sign on the flipped panels and every downstream number is garbage,
        with nothing in the output to say so. Orientation is therefore repaired
        programmatically and verified by check_closed(), never trusted.

        Method: breadth-first walk across shared edges. Two consistently wound
        neighbours traverse their shared edge in OPPOSITE directions; if a
        neighbour agrees in direction, it is flipped. The walk makes the surface
        globally consistent; the signed-volume test (divergence theorem) then
        decides whether that consistent orientation points out or in, and flips
        everything if it points in.
        """
        tris = self.tris.copy()

        # edge -> list of (triangle, direction it traverses the edge)
        edge_use = {}
        for t, tri in enumerate(tris):
            for k in range(3):
                a, b = tri[k], tri[(k + 1) % 3]
                edge_use.setdefault((min(a, b), max(a, b)), []).append(
                    (t, 1 if a < b else -1))

        neighbours = [[] for _ in range(len(tris))]
        for uses in edge_use.values():
            if len(uses) == 2:
                (t1, _), (t2, _) = uses
                neighbours[t1].append(t2)
                neighbours[t2].append(t1)

        def edge_dir(t, a, b):
            """+1 if triangle t traverses vertex a -> b, -1 if b -> a, 0 if absent."""
            tri = tris[t]
            for k in range(3):
                if tri[k] == a and tri[(k + 1) % 3] == b:
                    return 1
                if tri[k] == b and tri[(k + 1) % 3] == a:
                    return -1
            return 0

        visited = np.zeros(len(tris), bool)
        for root in range(len(tris)):
            if visited[root]:
                continue
            visited[root] = True
            queue = [root]
            while queue:
                t = queue.pop()
                for u in neighbours[t]:
                    if visited[u]:
                        continue
                    # find the shared edge
                    shared = set(tris[t]) & set(tris[u])
                    if len(shared) == 2:
                        a, b = tuple(shared)
                        if edge_dir(t, a, b) == edge_dir(u, a, b):
                            tris[u] = tris[u][::-1]      # same direction -> flip
                    visited[u] = True
                    queue.append(u)

        fixed = Mesh(verts=self.verts.copy(), tris=tris)
        if fixed.check_closed()["volume"] < 0:
            fixed = Mesh(verts=fixed.verts, tris=tris[:, ::-1].copy())
        return fixed


# ════════════════════════════════════════════════════════════════
# PANEL INFLUENCE — exact constant-strength source on a planar polygon
# ════════════════════════════════════════════════════════════════

def _panel_frames(mesh: Mesh):
    """Build an orthonormal local frame on each triangle."""
    p = mesh.verts[mesh.tris]
    c, n, a = mesh.properties()

    # local x along the first edge, y = n x x
    ex = p[:, 1] - p[:, 0]
    ex /= np.maximum(np.linalg.norm(ex, axis=1), EPS)[:, None]
    ey = np.cross(n, ex)
    ey /= np.maximum(np.linalg.norm(ey, axis=1), EPS)[:, None]

    # vertices in the local frame (z = 0 by construction)
    rel = p - c[:, None, :]                          # (T, 3, 3)
    lx = np.einsum("tvk,tk->tv", rel, ex)            # (T, 3)
    ly = np.einsum("tvk,tk->tv", rel, ey)
    return c, n, a, ex, ey, lx, ly


def _source_velocity(field_pts, c, n, a, ex, ey, lx, ly,
                     mesh_verts, mesh_tris, chunk=256):
    """
    Velocity induced at every field point by a UNIT-strength constant source on
    every panel. Returns (F, T, 3) in global coordinates.

    In-plane components: Hess-Smith edge sums,

        u += (dy_k / d_k) * ln( (r_k + r_k1 - d_k) / (r_k + r_k1 + d_k) )
        v += (-dx_k / d_k) * ln( same )

    Perpendicular component: the exact signed solid angle of the triangle at the
    field point (Van Oosterom & Strackee form — see below for why not the
    textbook atan2 sum). All times 1/(4*pi), with a global sign fixed by
    brute-force integration of the source kernel (see the return statement).

    This is the near-field-correct result. The old code used sigma*A/(4*pi*r^2)
    — a point source — for EVERY pair including adjacent panels, where it is
    simply wrong.
    """
    F = len(field_pts)
    T = len(c)
    out = np.zeros((F, T, 3))

    for lo in range(0, F, chunk):
        hi = min(lo + chunk, F)
        P = field_pts[lo:hi]                          # (f, 3)
        rel = P[:, None, :] - c[None, :, :]           # (f, T, 3)

        x = np.einsum("ftk,tk->ft", rel, ex)
        y = np.einsum("ftk,tk->ft", rel, ey)
        z = np.einsum("ftk,tk->ft", rel, n)

        u = np.zeros_like(x)
        v = np.zeros_like(x)

        for k in range(3):
            k1 = (k + 1) % 3
            x1, y1 = lx[None, :, k], ly[None, :, k]
            x2, y2 = lx[None, :, k1], ly[None, :, k1]

            dx, dy = x2 - x1, y2 - y1
            d = np.maximum(np.sqrt(dx ** 2 + dy ** 2), EPS)

            r1 = np.sqrt((x - x1) ** 2 + (y - y1) ** 2 + z ** 2)
            r2 = np.sqrt((x - x2) ** 2 + (y - y2) ** 2 + z ** 2)

            # log term, guarded on the edge itself
            num = np.maximum(r1 + r2 - d, EPS)
            den = np.maximum(r1 + r2 + d, EPS)
            log = np.log(num / den)

            u += (dy / d) * log
            v += (-dx / d) * log

        u /= (4 * np.pi)
        v /= (4 * np.pi)

        # --- perpendicular component: the EXACT signed solid angle -----------
        # The textbook form of this term is a sum of atan2(m*e - h, z*r) with
        # m = dy/dx, which divides by zero on any edge parallel to the local y
        # axis. On an Ahmed body — a box — most edges are exactly that, and the
        # whole solve came back NaN.
        #
        # The solid angle a triangle subtends at a point is exactly what this
        # term IS, and Van Oosterom & Strackee give it in a form with no such
        # singularity:
        #
        #   tan(omega/2) = a . (b x c)
        #                  ---------------------------------------------------
        #                  |a||b||c| + (a.b)|c| + (b.c)|a| + (c.a)|b|
        #
        # with a, b, c the vectors from the field point to the three vertices.
        # w = omega / (4*pi), which tends to +1/2 as the point approaches the
        # panel from its outward side — exactly the self-influence we want.
        tri = mesh_verts[mesh_tris]                          # (T, 3, 3)
        a_v = tri[None, :, 0, :] - P[:, None, :]             # (f, T, 3)
        b_v = tri[None, :, 1, :] - P[:, None, :]
        c_v = tri[None, :, 2, :] - P[:, None, :]

        la = np.linalg.norm(a_v, axis=2)
        lb = np.linalg.norm(b_v, axis=2)
        lc = np.linalg.norm(c_v, axis=2)

        triple = np.einsum("ftk,ftk->ft", a_v, np.cross(b_v, c_v))
        denom = (la * lb * lc
                 + np.einsum("ftk,ftk->ft", a_v, b_v) * lc
                 + np.einsum("ftk,ftk->ft", b_v, c_v) * la
                 + np.einsum("ftk,ftk->ft", c_v, a_v) * lb)

        omega = 2.0 * np.arctan2(triple, denom)
        w = omega / (4 * np.pi)

        # back to global
        out[lo:hi] = (u[..., None] * ex[None, :, :]
                      + v[..., None] * ey[None, :, :]
                      + w[..., None] * n[None, :, :])

    # Global sign. The Hess-Smith expressions above are written for a potential
    # of one sign convention; the physical source sheet used here is the one that
    # blows fluid OUTWARD,
    #
    #     v(P) = 1/(4 pi) * integral (P - Q) / |P - Q|^3 dA
    #
    # so that a unit source just outside a panel induces +1/2 along its outward
    # normal. Verified component-by-component against brute-force numerical
    # integration of that kernel in test_aero_3d.py::test_influence_matches_
    # brute_force_integration — which is the only way to catch a global sign
    # flip, because the flipped field still satisfies the boundary condition and
    # still integrates to zero drag. It just gets the pressure exactly backwards.
    return -out


def find_attached_region(mesh: Mesh, Cp: np.ndarray,
                         vel: Optional[np.ndarray] = None,
                         turn_limit_deg: float = TURN_LIMIT_DEG) -> np.ndarray:
    """
    Find which panels the flow stays attached to, and by elimination, the wake.

    The physical statement is the same one the 2D engine uses: FLOW CANNOT ROUND
    A SHARP CONVEX EDGE. Past some turning angle the boundary layer cannot follow
    the surface and it separates. So: flood-fill outward from the front
    stagnation region, refusing to cross any edge where the FLOW is asked to turn
    convexly by more than `turn_limit_deg`. Everything the fill cannot reach is
    in the wake.

    Three details, each of which produced spectacular garbage when got wrong:

    * THE WAKE IS NOT "EVERY PANEL FACING BACKWARDS". Blanketing all rear-facing
      panels with base pressure destroys the pressure recovery on attached rear
      surfaces — the recovery that, by d'Alembert, cancels the forebody's
      stagnation drag. Done that way, the Ahmed body returned Cd = 1.06 against
      a measured 0.25: the solver handed over the entire forebody pressure as
      drag.

    * THE SEED MUST BE THE FRONT STAGNATION. Potential flow has TWO stagnation
      points — the flow re-stagnates at the tail (that is d'Alembert's whole
      point) — so argmax(Cp) is as likely to sit on the base as on the nose.
      Seeded at the rear, the fill grew the "attached" region backwards from the
      tail, put the NOSE in the wake, and wake pressure on forward-facing panels
      is thrust: the solver reported negative drag. The seeds are therefore the
      high-pressure panels that FACE THE FLOW (Cp > 0.5 and n_x < 0).

    * THE TURN IS WEIGHTED BY WHETHER THE FLOW CROSSES THE EDGE. Separation is
      about the flow being asked to turn, not about surface geometry in the
      abstract. A box's roof-side edges turn 90 degrees, but the flow there runs
      PARALLEL to the edge and crosses nothing, and the real Ahmed body's
      longitudinal edges are rounded precisely so the flow hugs them. An
      unweighted criterion cut the roof off at those edges and marked the whole
      roof as wake at every slant angle. The turn is scaled by the sine of the
      angle between the local velocity and the edge: full weight when the flow
      crosses the edge squarely, zero when it runs along it.

    What this criterion buys and what it does not: it DOES capture a gently
    raked slant staying attached while a steep one separates. It does NOT
    capture the Ahmed body's 30-degree drag peak, because that is C-pillar
    vortices holding the flow onto a slant it could never follow unaided — a
    vortex effect, and there are no vortices in a source panel method.
    """
    T = mesh.n
    c, n, a = mesh.properties()

    # --- panel adjacency across shared edges --------------------------------
    edge_map = {}
    for t, tri in enumerate(mesh.tris):
        for k in range(3):
            i, j = tri[k], tri[(k + 1) % 3]
            edge_map.setdefault((min(i, j), max(i, j)), []).append(t)

    adj = [[] for _ in range(T)]
    for (i, j), tl in edge_map.items():
        if len(tl) != 2:
            continue
        t1, t2 = tl

        cos_turn = np.clip(np.dot(n[t1], n[t2]), -1.0, 1.0)
        turn = np.degrees(np.arccos(cos_turn))

        # convex if the neighbour's centroid sits behind this panel's plane
        convex = np.dot(c[t2] - c[t1], n[t1]) < 0.0

        # weight by how squarely the flow crosses this edge
        crossing = 1.0
        if vel is not None:
            e = mesh.verts[j] - mesh.verts[i]
            e_len = np.linalg.norm(e)
            v_mean = vel[t1] + vel[t2]
            v_len = np.linalg.norm(v_mean)
            if e_len > EPS and v_len > EPS:
                along = abs(np.dot(v_mean / v_len, e / e_len))
                crossing = float(np.sqrt(max(1.0 - along ** 2, 0.0)))

        blocked = convex and (turn * crossing > turn_limit_deg)
        if not blocked:
            adj[t1].append(t2)
            adj[t2].append(t1)

    # --- flood fill from the FRONT stagnation region -------------------------
    seeds = np.where((Cp > 0.5) & (n[:, 0] < -0.1))[0]
    if seeds.size == 0:
        # fall back: the most upstream-facing high-pressure panel
        facing = np.where(n[:, 0] < -0.1)[0]
        seeds = np.array([facing[np.argmax(Cp[facing])]] if facing.size
                         else [int(np.argmax(Cp))])

    attached = np.zeros(T, bool)
    attached[seeds] = True
    stack = list(seeds)
    while stack:
        t = stack.pop()
        for u in adj[t]:
            if not attached[u]:
                attached[u] = True
                stack.append(u)

    return attached


def _mirror(mesh: Mesh, z_ground: float) -> Mesh:
    """
    Reflect the mesh in the ground plane, flipping the winding so the image body
    keeps outward normals.

    THE GROUND PLANE IS NOT OPTIONAL FOR A CAR. It is the whole reason road-vehicle
    aerodynamics differs from aircraft aerodynamics: the underbody flow is confined
    between the car and a wall moving at freestream speed, and that confinement
    drives underbody acceleration, ground effect and the diffuser's entire reason
    for existing. The previous version had no ground plane at all.

    The method of images enforces zero normal velocity on z = z_ground exactly, for
    free, by superposing a mirrored copy of the body.
    """
    v = mesh.verts.copy()
    v[:, 2] = 2 * z_ground - v[:, 2]
    t = mesh.tris[:, ::-1].copy()          # reverse winding to keep normals out
    return Mesh(verts=v, tris=t)


# ════════════════════════════════════════════════════════════════
# SOLVER
# ════════════════════════════════════════════════════════════════

@dataclass
class Solution3D:
    Cp: np.ndarray                  # (T,) pressure coefficient, wake imposed
    Cp_potential: np.ndarray        # (T,) before the wake model
    velocity: np.ndarray            # (T, 3)
    sigma: np.ndarray               # (T,) source strengths
    wake: np.ndarray                # (T,) bool — panels in the separated region
    Cpb: float                      # base pressure in the wake
    Cd: float                       # total drag coefficient
    Cd_pressure: float
    Cd_friction: float
    Cl: float                       # lift coefficient (ground effect makes it real)
    A_ref: float
    centroids: np.ndarray
    normals: np.ndarray
    areas: np.ndarray


def solve_3d(mesh: Mesh,
             V_inf: float = 27.8,
             A_ref: Optional[float] = None,
             ground: bool = True,
             z_ground: float = 0.0,
             wake_model: bool = True,
             Cpb: float = -0.20,
             turn_limit_deg: float = TURN_LIMIT_DEG,
             L_ref: float = 1.0,
             nu: float = 1.5e-5) -> Solution3D:
    """
    Solve the 3D source-panel system and return a full Solution3D.

    Args:
        mesh         : closed, consistently wound triangulated surface
        V_inf        : freestream speed along +x
        A_ref        : reference area for the coefficients; if None, the mesh's
                       projected frontal area is COMPUTED from the geometry
        ground       : enforce a ground plane by the method of images
        wake_model   : impose base pressure on the separated region. Without this
                       the solver returns EXACTLY ZERO DRAG (d'Alembert), which is
                       correct potential flow and useless engineering.
        Cpb          : base pressure coefficient in the wake. -0.20 is the
                       middle of Ahmed's measured base-pressure range.
        turn_limit_deg : maximum convex surface turn the flow will follow before
                       separating. FIXED FROM LITERATURE, NOT TUNED — see
                       TURN_LIMIT_DEG for the anchor and for what happened the
                       last time this project treated a constant as a free dial.
    """
    # A panel method on an unclosed or inconsistently wound mesh returns
    # confident garbage — flipped normals silently negate the boundary condition
    # panel by panel, and nothing in the output reveals it. Refuse to run.
    chk = mesh.check_closed()
    if not (chk["closed"] and chk["outward"]):
        raise ValueError(
            f"mesh is not a closed, outward-oriented surface: {chk}. "
            f"Run Mesh.oriented() first.")

    c, n, a, ex, ey, lx, ly = _panel_frames(mesh)
    T = mesh.n

    # --- reference area: computed from the mesh, not asserted ---------------
    if A_ref is None:
        # projected frontal area = 1/2 * sum |n_x| * A  over a closed body
        A_ref = float(0.5 * np.sum(np.abs(n[:, 0]) * a))
    A_ref = max(A_ref, EPS)

    # --- influence of the real body on itself -------------------------------
    Vind = _source_velocity(c, c, n, a, ex, ey, lx, ly,
                            mesh.verts, mesh.tris)              # (T, T, 3)

    # --- plus the influence of the mirrored body, if there is a ground -------
    if ground:
        m = _mirror(mesh, z_ground)
        mc, mn, ma, mex, mey, mlx, mly = _panel_frames(m)
        Vind = Vind + _source_velocity(c, mc, mn, ma, mex, mey, mlx, mly,
                                       m.verts, m.tris)

    # Self-influence. A panel's own centroid lies exactly IN its plane, where the
    # signed solid angle is degenerate and flips sign on floating-point noise —
    # measured across a sphere it came out 0.10 +/- 0.49 instead of a uniform
    # +0.5. The analytic limit approaching from the outward side is +1/2 normal
    # and zero in-plane, so impose that directly.
    #
    # Crucially it must be imposed in the INFLUENCE ARRAY, not just in the matrix
    # A. If the matrix assumes +0.5 while the velocity reconstruction uses the
    # degenerate value, the solve satisfies a boundary condition that the
    # reconstruction then violates.
    idx = np.arange(T)
    Vind[idx, idx] = 0.5 * n

    A = np.einsum("ftk,fk->ft", Vind, n)

    # --- boundary condition: no flow through the surface --------------------
    U = np.array([V_inf, 0.0, 0.0])
    rhs = -(n @ U)
    sigma = np.linalg.solve(A, rhs)

    # --- recover the velocity field -----------------------------------------
    # vel[i] = U + sum_j Vind[i, j, :] * sigma[j]   — summed over the SOURCE
    # index j. An earlier version transposed Vind here, which summed sigma over
    # the FIELD index instead: the boundary condition came out violated by
    # |v.n| = 0.9, and the error grew as the mesh was refined.
    vel = U[None, :] + np.einsum("ftk,t->fk", Vind, sigma)
    # remove any residual normal component (discretisation) before taking Cp
    vn = np.einsum("fk,fk->f", vel, n)
    vt = vel - vn[:, None] * n
    Cp_pot = 1.0 - np.einsum("fk,fk->f", vt, vt) / V_inf ** 2

    # --- wake --------------------------------------------------------------
    # Sources alone give zero drag on a closed body (d'Alembert). Drag appears
    # only when the separated surface is held at the base pressure instead of the
    # much higher pressure that potential flow puts there.
    if wake_model:
        attached = find_attached_region(mesh, Cp_pot, vel=vel,
                                        turn_limit_deg=turn_limit_deg)
        wake = ~attached
    else:
        wake = np.zeros(T, bool)

    Cp = Cp_pot.copy()
    Cp[wake] = Cpb

    # --- forces -------------------------------------------------------------
    # F = -integral Cp * n dA
    F = -np.einsum("f,fk,f->k", Cp, n, a)
    Cd_pressure = float(F[0] / A_ref)
    Cl = float(F[2] / A_ref)

    # --- skin friction (turbulent flat plate over the wetted area) ----------
    Re = V_inf * L_ref / nu
    Cf = 0.074 / Re ** 0.2
    S_wet = float(np.sum(a))
    Cd_friction = float(Cf * S_wet / A_ref)

    return Solution3D(
        Cp=Cp, Cp_potential=Cp_pot, velocity=vel, sigma=sigma,
        wake=wake, Cpb=Cpb,
        Cd=Cd_pressure + Cd_friction,
        Cd_pressure=Cd_pressure, Cd_friction=Cd_friction, Cl=Cl,
        A_ref=A_ref, centroids=c, normals=n, areas=a,
    )


# ════════════════════════════════════════════════════════════════
# GEOMETRY — sphere (for exact validation) and the Ahmed body
# ════════════════════════════════════════════════════════════════

def make_sphere(radius: float = 1.0, subdiv: int = 3) -> Mesh:
    """Icosphere. The only body with an exact analytic potential-flow solution,
    so it is the only way to check the influence coefficients are right."""
    t = (1 + 5 ** 0.5) / 2
    v = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    v /= np.linalg.norm(v, axis=1)[:, None]
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
         [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
         [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
         [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]

    verts = [tuple(x) for x in v]
    index = {p: i for i, p in enumerate(verts)}

    def midpoint(i, j):
        p = (v[i] + v[j]) if False else None
        return None

    V = list(v)
    for _ in range(subdiv):
        cache = {}
        newf = []

        def mid(i, j):
            key = (min(i, j), max(i, j))
            if key not in cache:
                p = V[i] + V[j]
                p = p / np.linalg.norm(p)
                V.append(p)
                cache[key] = len(V) - 1
            return cache[key]

        for A_, B_, C_ in f:
            ab, bc, ca = mid(A_, B_), mid(B_, C_), mid(C_, A_)
            newf += [[A_, ab, ca], [B_, bc, ab], [C_, ca, bc], [ab, bc, ca]]
        f = newf

    return Mesh(verts=np.array(V) * radius,
                tris=np.array(f, dtype=int)).oriented()


def make_ahmed_body(slant_deg: float = 25.0, n_len: int = 22,
                    n_wid: int = 10, n_hgt: int = 10,
                    scale: float = 1.0) -> Mesh:
    """
    The Ahmed reference body (SAE 840300) — the standard automotive bluff-body
    benchmark, parametric in one variable: the rear slant angle.

    Nominal dimensions (mm): 1044 long, 389 wide, 288 tall, 222 slant length,
    with ALL leading edges rounded at R100. The rounding is not cosmetic and it
    cannot be approximated with a chamfer: an earlier version used a single flat
    46-degree chamfer for the nose, the separation criterion (correctly) refused
    to carry attached flow over a 44-degree kink, the flood fill never escaped
    the front face, and the whole body was blanketed in wake — Cd came out at
    0.78 against a measured 0.25. The real body is rounded precisely so the
    flow stays attached around the nose, and the model must be too. The nose is
    therefore a quarter-ellipse arc, tangent to the front face below and to the
    roof above, faceted finely enough (~11 degrees per facet) for the flood
    fill to walk over it.

    (The plan-view nose rounding is omitted: the side panels carry n_x = 0 and
    contribute nothing to drag either way.)

    Its published Cd-vs-slant curve is the standard test of whether a wake model
    is real. See validate_ahmed() — and read the module docstring for what this
    solver can and cannot reproduce.
    """
    L, W, H = 1.044 * scale, 0.389 * scale, 0.288 * scale
    clearance = 0.050 * scale
    slant_len = 0.222 * scale
    r_nose = 0.100 * scale          # leading-edge radius, SAE 840300

    phi = np.radians(slant_deg)
    # The 222 mm in SAE 840300 is the slant LENGTH along the surface (the
    # hypotenuse), so the vertical drop is sin(phi) and the horizontal run is
    # cos(phi). An earlier version used tan(phi) for the drop while keeping
    # cos(phi) for the run — inconsistent triangles: the facet built for
    # '40 degrees' actually sloped at 47.5, so the whole sweep was compared
    # against the wrong abscissa.
    drop = min(slant_len * np.sin(phi), H * 0.95)
    slant_run = slant_len * np.cos(phi) if slant_deg > 0 else 0.0

    z0, z1 = clearance, clearance + H
    y0, y1 = -W / 2, W / 2
    x_slant = L - slant_run
    x_tail = L

    def z_top(x):
        """Upper surface height: nose arc -> roof -> slant."""
        if x < r_nose:
            # quarter ellipse: vertical tangent at x=0 (blends into the front
            # face), horizontal tangent at x=r_nose (blends into the roof)
            f = (r_nose - x) / r_nose
            return z1 - r_nose + r_nose * np.sqrt(max(1.0 - f * f, 0.0))
        if x <= x_slant or slant_run <= 0:
            return z1
        return z1 - (x - x_slant) / max(slant_run, EPS) * drop

    def z_bot(x):
        """Lower surface height: nose arc -> flat floor."""
        if x < r_nose:
            f = (r_nose - x) / r_nose
            return z0 + r_nose - r_nose * np.sqrt(max(1.0 - f * f, 0.0))
        return z0

    # Stations: uniform in ARC ANGLE over the nose (uniform curvature -> uniform
    # turn per facet, ~11 degrees each), then linear over body and slant.
    theta = np.linspace(0.0, np.pi / 2, 9)
    xs_nose = r_nose * (1.0 - np.cos(theta))
    xs_body = np.linspace(r_nose, x_slant, max(n_len - 10, 4))[1:]
    xs_slant = (np.linspace(x_slant, x_tail, 5)[1:]
                if slant_run > 0 else np.array([x_tail]))
    xs = np.unique(np.round(np.concatenate([xs_nose, xs_body, xs_slant]), 9))

    ys = np.linspace(y0, y1, n_wid + 1)
    verts = []
    vid = {}

    def V(x, y, z):
        key = (round(x, 7), round(y, 7), round(z, 7))
        if key not in vid:
            vid[key] = len(verts)
            verts.append([x, y, z])
        return vid[key]

    tris = []

    def quad(a, b, c, d):
        if a != b and b != c and a != c:
            tris.append([a, b, c])
        if a != c and c != d and a != d:
            tris.append([a, c, d])

    # roof + floor (normals: roof up = +z, floor down = -z)
    for i in range(len(xs) - 1):
        xa, xb = xs[i], xs[i + 1]
        za_hi, zb_hi = z_top(xa), z_top(xb)
        za_lo, zb_lo = z_bot(xa), z_bot(xb)
        for j in range(len(ys) - 1):
            ya, yb = ys[j], ys[j + 1]
            quad(V(xa, ya, za_hi), V(xb, ya, zb_hi), V(xb, yb, zb_hi), V(xa, yb, za_hi))
            quad(V(xa, ya, za_lo), V(xa, yb, za_lo), V(xb, yb, zb_lo), V(xb, ya, zb_lo))

    # sides (y = y0 outward -y ; y = y1 outward +y)
    for i in range(len(xs) - 1):
        xa, xb = xs[i], xs[i + 1]
        zs_a = np.linspace(z_bot(xa), z_top(xa), n_hgt + 1)
        zs_b = np.linspace(z_bot(xb), z_top(xb), n_hgt + 1)
        for k in range(n_hgt):
            quad(V(xa, y0, zs_a[k]), V(xa, y0, zs_a[k + 1]),
                 V(xb, y0, zs_b[k + 1]), V(xb, y0, zs_b[k]))
            quad(V(xa, y1, zs_a[k]), V(xb, y1, zs_b[k]),
                 V(xb, y1, zs_b[k + 1]), V(xa, y1, zs_a[k + 1]))

    # front face (outward = -x), spans the reduced height between the nose arcs
    zs_f = np.linspace(z_bot(0.0), z_top(0.0), n_hgt + 1)
    for j in range(len(ys) - 1):
        for k in range(n_hgt):
            quad(V(0.0, ys[j], zs_f[k]), V(0.0, ys[j + 1], zs_f[k]),
                 V(0.0, ys[j + 1], zs_f[k + 1]), V(0.0, ys[j], zs_f[k + 1]))

    # rear face / base (outward = +x)
    zs_r = np.linspace(z0, z_top(x_tail), n_hgt + 1)
    for j in range(len(ys) - 1):
        for k in range(n_hgt):
            quad(V(x_tail, ys[j], zs_r[k]), V(x_tail, ys[j], zs_r[k + 1]),
                 V(x_tail, ys[j + 1], zs_r[k + 1]), V(x_tail, ys[j + 1], zs_r[k]))

    return Mesh(verts=np.array(verts, float),
                tris=np.array(tris, dtype=int)).oriented()


# ════════════════════════════════════════════════════════════════
# VALIDATION
# ════════════════════════════════════════════════════════════════

def validate_sphere(subdiv: int = 3, verbose: bool = True) -> dict:
    """
    The only exact check available.

    Potential flow past a sphere has the closed-form surface pressure

        Cp(theta) = 1 - (9/4) sin^2(theta)

    so Cp runs from +1 at the stagnation point to -1.25 at the equator, and the
    net force is exactly zero. If the influence coefficients are wrong, this
    fails immediately and unambiguously — which is why it is worth having.
    """
    mesh = make_sphere(1.0, subdiv)
    chk = mesh.check_closed()
    sol = solve_3d(mesh, V_inf=1.0, ground=False, wake_model=False, A_ref=np.pi)

    c = sol.centroids
    r = np.linalg.norm(c, axis=1)
    cos_t = c[:, 0] / np.maximum(r, EPS)             # +x is the freestream
    sin2 = 1.0 - cos_t ** 2
    Cp_exact = 1.0 - 2.25 * sin2

    err = np.abs(sol.Cp_potential - Cp_exact)
    force = float(np.linalg.norm(
        -np.einsum("f,fk,f->k", sol.Cp_potential, sol.normals, sol.areas)))

    out = dict(n_tris=mesh.n, closed=chk["closed"], outward=chk["outward"],
               Cp_max=float(sol.Cp_potential.max()),
               Cp_min=float(sol.Cp_potential.min()),
               mean_abs_err=float(err.mean()), max_abs_err=float(err.max()),
               net_force=force)

    if verbose:
        print("=" * 72)
        print("  3D SOLVER VALIDATION — sphere against the analytic solution")
        print("  Cp(theta) = 1 - 2.25 sin^2(theta):  +1.000 at stagnation, -1.250 at the equator")
        print("=" * 72)
        print(f"  triangles           : {out['n_tris']}")
        print(f"  closed / outward    : {out['closed']} / {out['outward']}")
        print(f"  Cp max (exact +1.000): {out['Cp_max']:+.3f}")
        print(f"  Cp min (exact -1.250): {out['Cp_min']:+.3f}")
        print(f"  mean |error|        : {out['mean_abs_err']:.4f}")
        print(f"  net force (exact 0) : {out['net_force']:.2e}   <- d'Alembert")
        print("=" * 72 + "\n")
    return out


# Published Ahmed-body drag, SAE 840300 (Re = 4.29e6, ground-fixed).
# The peak at 30 degrees and the collapse just past it are the whole point of
# the experiment, and the reason this body is the standard benchmark.
AHMED_PUBLISHED_CD = {
    0.0: 0.250,
    12.5: 0.230,
    25.0: 0.285,
    30.0: 0.378,      # the peak — C-pillar vortices at their strongest
    35.0: 0.260,      # vortices burst, the slant fully separates, drag collapses
    40.0: 0.255,
}


def validate_ahmed(angles=(0.0, 12.5, 25.0, 30.0, 35.0, 40.0),
                   verbose: bool = True) -> dict:
    """
    Run the Ahmed body across its slant-angle sweep and compare with SAE 840300.

    This function exists to show a FAILURE as much as a success, and it does not
    tune it away. See the module docstring: a source panel method has no
    circulation, so it cannot produce the C-pillar vortex system that drives the
    drag peak at 30 degrees. Expect agreement at a square back and a growing
    error as the slant approaches the critical angle.

    That is the honest boundary of this solver, and knowing exactly where a model
    stops working is worth more than a curve that has been massaged to fit.
    """
    results = {}
    for phi in angles:
        mesh = make_ahmed_body(slant_deg=phi)
        sol = solve_3d(mesh, V_inf=27.8, ground=True, z_ground=0.0,
                       wake_model=True, Cpb=-0.20, L_ref=1.044)
        results[phi] = sol

    if verbose:
        print("=" * 72)
        print("  3D SOLVER — Ahmed body (SAE 840300) slant-angle sweep")
        print("=" * 72)
        print(f"  {'slant':>6} {'Cd_pred':>8} {'Cd_pub':>8} {'error':>8}   note")
        print("  " + "-" * 62)
        for phi, sol in results.items():
            pub = AHMED_PUBLISHED_CD.get(phi)
            err = (sol.Cd - pub) / pub * 100 if pub else float("nan")
            note = ""
            if phi in (25.0, 30.0):
                note = "<- C-pillar vortices; NOT modelled by sources"
            print(f"  {phi:>5.1f} {sol.Cd:>8.3f} {pub:>8.3f} {err:>+7.1f}%   {note}")
        print("  " + "-" * 62)
        print("  A source panel method has no circulation and therefore cannot")
        print("  reproduce the 30-degree drag peak, which is created by a pair of")
        print("  counter-rotating streamwise vortices off the slant side edges.")
        print("  Reproducing it needs a shed vortex sheet — the next piece of work.")
        print("  The square-backed case (0 deg), which is a pure base-drag problem,")
        print("  is what this solver is currently good for.")
        print("=" * 72 + "\n")

    return {phi: dict(Cd=s.Cd, Cd_pressure=s.Cd_pressure,
                      Cd_friction=s.Cd_friction, Cl=s.Cl, A_ref=s.A_ref,
                      published=AHMED_PUBLISHED_CD.get(phi))
            for phi, s in results.items()}


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    validate_sphere()
    validate_ahmed()
