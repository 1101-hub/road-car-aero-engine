"""
Tests for Layer 5 — the 3D source panel method.

The ordering here mirrors how this solver actually failed while being built,
and each failure earned a test:

  * The influence formula had a GLOBAL SIGN FLIP. It still satisfied the
    boundary condition and still gave zero net force — the two obvious checks —
    while getting every pressure exactly backwards. The only test that catches
    it is comparing against brute-force numerical integration of the source
    kernel, so that test exists now.

  * The velocity reconstruction transposed the field/source axes of the
    influence array. Symptom: the boundary condition it had just solved for
    came out violated by |v.n| ~ 0.9. Hence the boundary-condition test.

  * The Ahmed mesher shipped 116 flipped edges and a negative enclosed volume
    (inward normals), which silently negated the boundary condition panel by
    panel. Hence the closure/orientation tests and the guard in solve_3d.

  * The separation flood-fill seeded at the REAR stagnation point (potential
    flow has two — that is d'Alembert's whole point) and grew the attached
    region backwards; wake pressure landed on the nose and the solver reported
    NEGATIVE drag. Hence the no-forward-facing-wake test.
"""

import numpy as np
import pytest

from core.aero_3d import (
    Mesh, make_sphere, make_ahmed_body, solve_3d, find_attached_region,
    _panel_frames, _source_velocity, validate_sphere,
    AHMED_PUBLISHED_CD, TURN_LIMIT_DEG, EPS,
)


# ── module-scope fixtures: each full solve is expensive, run it once ────────

@pytest.fixture(scope="module")
def sphere():
    mesh = make_sphere(1.0, subdiv=2)          # 320 triangles
    sol = solve_3d(mesh, V_inf=1.0, ground=False, wake_model=False,
                   A_ref=np.pi)
    return mesh, sol


def _coarse_ahmed(slant_deg):
    return make_ahmed_body(slant_deg=slant_deg, n_len=14, n_wid=6, n_hgt=6)


@pytest.fixture(scope="module")
def ahmed_0():
    mesh = _coarse_ahmed(0.0)
    return mesh, solve_3d(mesh, V_inf=27.8, ground=True, wake_model=True,
                          L_ref=1.044)


@pytest.fixture(scope="module")
def ahmed_30():
    mesh = _coarse_ahmed(30.0)
    return mesh, solve_3d(mesh, V_inf=27.8, ground=True, wake_model=True,
                          L_ref=1.044)


# ════════════════════════════════════════════════════════════════
# THE INFLUENCE FORMULA ITSELF
# ════════════════════════════════════════════════════════════════

def test_influence_matches_brute_force_integration():
    """
    THE definitive test. The velocity of a unit constant-strength source sheet is

        v(P) = 1/(4 pi) * integral over the panel of (P - Q)/|P - Q|^3 dA

    Integrate that numerically over one triangle and the analytic formula must
    match, component by component, at several field points. This is the only
    test that can catch a global sign flip: a fully negated influence field
    still satisfies the boundary condition and still produces zero net force —
    it just reports every pressure backwards, which is exactly what happened.
    """
    tri = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mesh = Mesh(verts=tri, tris=np.array([[0, 1, 2]]))
    c, n, a, ex, ey, lx, ly = _panel_frames(mesh)

    def brute(P, N=600):
        # Cell-centred barycentric grid. The parametrisation Q = A + u*AB + v*AC
        # over the region u+v <= 1 has |Jacobian| = 2*Area = 1 for this unit
        # right triangle, so the integral is just sum(f) / N^2. (A first draft
        # used .mean() times the area, with 1% grid margins — the margins
        # under-covered the triangle and the mean mis-weighted the clipped
        # region, so the reference itself was biased by several percent.)
        g = (np.arange(N) + 0.5) / N
        u, v = np.meshgrid(g, g)
        u, v = u.ravel(), v.ravel()
        keep = u + v <= 1.0
        u, v = u[keep], v[keep]
        Q = tri[0] + u[:, None] * (tri[1] - tri[0]) + v[:, None] * (tri[2] - tri[0])
        d = P[None, :] - Q
        r = np.linalg.norm(d, axis=1)[:, None]
        return (d / r ** 3).sum(axis=0) / N ** 2 / (4 * np.pi)

    for P in [np.array([0.3, 0.3, 0.5]),      # above the panel
              np.array([1.5, 0.2, 0.3]),      # off to the side
              np.array([0.3, 0.3, -0.5])]:    # below the panel
        got = _source_velocity(P[None, :], c, n, a, ex, ey, lx, ly,
                               mesh.verts, mesh.tris)[0, 0]
        want = brute(P)
        assert np.allclose(got, want, rtol=0.02, atol=1e-4), \
            f"at P={P}: formula {got} vs integral {want}"


def test_self_influence_is_half_normal(sphere):
    """A source panel induces exactly +1/2 along its own outward normal, and
    nothing in-plane. Indirect check: a wrong self-term shifts every diagonal
    of the influence matrix and the stagnation Cp drifts far from +1.

    Tolerance note: at 320 triangles no panel CENTROID sits exactly on the
    stagnation pole, so the max sampled Cp is 0.93 by geometry alone (0.983 at
    1280 triangles, 1.000 in the limit). The tolerance covers that sampling
    offset, not solver error."""
    mesh, sol = sphere
    assert sol.Cp_potential.max() == pytest.approx(1.0, abs=0.10)


# ════════════════════════════════════════════════════════════════
# EXACT ANALYTIC VALIDATION — THE SPHERE
# ════════════════════════════════════════════════════════════════

def test_sphere_matches_analytic_solution(sphere):
    """Potential flow past a sphere: Cp = 1 - 2.25 sin^2(theta). The only body
    with an exact closed-form answer, so the only exact test of the solver."""
    mesh, sol = sphere
    c = sol.centroids
    cos_t = c[:, 0] / np.linalg.norm(c, axis=1)
    Cp_exact = 1.0 - 2.25 * (1.0 - cos_t ** 2)
    err = np.abs(sol.Cp_potential - Cp_exact)
    assert err.mean() < 0.05, f"mean |Cp error| = {err.mean():.4f}"
    assert sol.Cp_potential.min() == pytest.approx(-1.25, abs=0.15)
    # 0.10 abs on the max: no centroid sits exactly on the stagnation pole at
    # this resolution — see test_self_influence_is_half_normal.
    assert sol.Cp_potential.max() == pytest.approx(1.0, abs=0.10)


def test_sphere_dalembert(sphere):
    """Closed body, attached potential flow: net pressure force must vanish."""
    mesh, sol = sphere
    F = -np.einsum("f,fk,f->k", sol.Cp_potential, sol.normals, sol.areas)
    assert np.linalg.norm(F) < 1e-8


def test_boundary_condition_is_satisfied(sphere):
    """
    v . n = 0 on every panel of the solved flow. This catches the axis-transpose
    bug: the influence array was flipped between its field and source indices,
    and the reconstructed velocity violated the very condition the linear solve
    had just enforced, by |v.n| up to 0.9.
    """
    mesh, sol = sphere
    vn = np.einsum("fk,fk->f", sol.velocity, sol.normals)
    assert np.abs(vn).max() < 1e-8


# ════════════════════════════════════════════════════════════════
# MESH INTEGRITY
# ════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("slant", [0.0, 12.5, 30.0])
def test_ahmed_mesh_is_closed_and_outward(slant):
    chk = _coarse_ahmed(slant).check_closed()
    assert chk["closed"], f"slant {slant}: {chk}"
    assert chk["outward"], f"slant {slant}: inward normals, volume {chk['volume']}"


def test_oriented_repairs_scrambled_winding():
    """Flip a third of the triangles at random; oriented() must fully repair it."""
    mesh = make_sphere(1.0, subdiv=1)
    tris = mesh.tris.copy()
    rng = np.random.default_rng(7)
    bad = rng.choice(len(tris), size=len(tris) // 3, replace=False)
    tris[bad] = tris[bad, ::-1]
    broken = Mesh(verts=mesh.verts, tris=tris)
    assert not broken.check_closed()["closed"]

    fixed = broken.oriented()
    chk = fixed.check_closed()
    assert chk["closed"] and chk["outward"]


def test_solver_refuses_a_broken_mesh():
    """Flipped normals silently negate the boundary condition panel by panel;
    the solver must refuse rather than return confident garbage."""
    mesh = make_sphere(1.0, subdiv=1)
    tris = mesh.tris.copy()
    tris[0] = tris[0, ::-1]
    with pytest.raises(ValueError, match="closed"):
        solve_3d(Mesh(verts=mesh.verts, tris=tris), wake_model=False)


# ════════════════════════════════════════════════════════════════
# SEPARATION / WAKE
# ════════════════════════════════════════════════════════════════

def test_no_forward_facing_panel_in_wake(ahmed_0, ahmed_30):
    """
    Wake pressure on a forward-facing panel is THRUST. When the flood fill
    seeded itself at the rear stagnation point (potential flow has two), the
    nose ended up in the wake and the solver reported negative drag.
    """
    for mesh, sol in (ahmed_0, ahmed_30):
        fwd_wake_area = sol.areas[sol.wake & (sol.normals[:, 0] < -0.3)].sum()
        assert fwd_wake_area / sol.areas.sum() < 0.01, \
            "forward-facing surface has been marked as wake"


def test_base_is_wake_and_nose_is_attached(ahmed_0):
    """On a square-back body the base must be in the wake and the front face,
    roof and floor must be attached."""
    mesh, sol = ahmed_0
    n, c = sol.normals, sol.centroids
    base = n[:, 0] > 0.9
    front = n[:, 0] < -0.9
    roof = (n[:, 2] > 0.9) & (c[:, 0] > 0.3) & (c[:, 0] < 0.7)
    assert sol.wake[base].all(), "square-back base must be separated"
    assert not sol.wake[front].any(), "the nose cannot be in the wake"
    assert not sol.wake[roof].any(), "mid-roof must stay attached"


def test_slant_attached_below_limit_separated_above():
    """The turn criterion's qualitative content: a 12.5-degree slant (< 20-deg
    limit) keeps attached flow; a 30-degree slant separates at the roof edge."""
    m_shallow = _coarse_ahmed(12.5)
    s_shallow = solve_3d(m_shallow, ground=True, wake_model=True, L_ref=1.044)
    m_steep = _coarse_ahmed(30.0)
    s_steep = solve_3d(m_steep, ground=True, wake_model=True, L_ref=1.044)

    def slant_wake_fraction(sol):
        n = sol.normals
        slant = (n[:, 0] > 0.2) & (n[:, 2] > 0.5)      # up-and-backward facing
        if not slant.any():
            return 0.0
        return sol.areas[slant & sol.wake].sum() / sol.areas[slant].sum()

    assert slant_wake_fraction(s_shallow) < 0.2, "12.5-deg slant should stay attached"
    assert slant_wake_fraction(s_steep) > 0.8, "30-deg slant should be separated"


def test_dalembert_residual_shrinks_with_refinement():
    """
    With the wake model off, a closed body in potential flow has zero drag —
    but the Ahmed body's base edges are genuinely SHARP (unlike its rounded
    nose), and a sharp convex edge is a velocity singularity, so at finite
    resolution the residual is large (measured: -0.50 at 672 triangles) and
    converges slowly (-0.38 at 1920). This is the exact 3D analogue of the 2D
    engine's base-corner story.

    So the correctness demand here is CONVERGENCE — the residual must shrink
    as the mesh refines — not a small value at coarse resolution. The sphere
    (smooth, no corners) is where near-zero is demanded outright, and it
    delivers ~1e-15. Note also that the with-wake Cd is protected from this
    corner corruption: the wake model overwrites precisely the corrupted rear
    panels with Cpb, and the corrupted roof-edge panels carry n_x = 0 and
    cannot contribute drag.
    """
    coarse = solve_3d(_coarse_ahmed(0.0), ground=False, wake_model=False,
                      L_ref=1.044).Cd_pressure
    fine = solve_3d(make_ahmed_body(0.0, n_len=22, n_wid=10, n_hgt=10),
                    ground=False, wake_model=False, L_ref=1.044).Cd_pressure
    assert abs(fine) < abs(coarse), \
        f"residual grew under refinement: {coarse:.4f} -> {fine:.4f}"


# ════════════════════════════════════════════════════════════════
# THE NUMBERS ARE PHYSICAL (AND THE MISS IS THE DOCUMENTED ONE)
# ════════════════════════════════════════════════════════════════

def test_ahmed_square_back_cd_is_in_range(ahmed_0):
    """Square back = pure base-drag problem = what this solver is for.
    Published: 0.250. Demand the right neighbourhood, not the third decimal."""
    mesh, sol = ahmed_0
    assert 0.15 < sol.Cd < 0.50, f"Cd = {sol.Cd:.3f}"


def test_ahmed_drag_is_never_negative(ahmed_0, ahmed_30):
    for mesh, sol in (ahmed_0, ahmed_30):
        assert sol.Cd > 0, f"negative drag: {sol.Cd:.3f}"
        assert sol.Cd_pressure > 0


def test_reference_area_is_computed_from_geometry(ahmed_0):
    """A_ref must come from the mesh, not from an assertion. Frontal area of the
    Ahmed body is about 0.389 x 0.288 = 0.112 m^2, slightly less with the nose
    rounding."""
    mesh, sol = ahmed_0
    assert 0.08 < sol.A_ref < 0.13, f"A_ref = {sol.A_ref:.4f}"


def test_ground_effect_changes_the_solution():
    """The ground plane must actually do something: the same body in free air
    and in ground effect must not report the same lift."""
    mesh = _coarse_ahmed(0.0)
    free = solve_3d(mesh, ground=False, wake_model=True, L_ref=1.044)
    ground = solve_3d(mesh, ground=True, wake_model=True, L_ref=1.044)
    assert abs(free.Cl - ground.Cl) > 0.01, \
        "ground plane had no measurable effect on lift"


def test_the_30_degree_miss_is_present_and_documented():
    """
    The anti-overfitting guard, inverted: this solver must NOT reproduce the
    Ahmed 30-degree drag peak, because it contains no vortices. If some future
    change makes the sweep match at 30 degrees WITHOUT a vortex model, that is
    not a triumph — it means a constant has been quietly tuned into a lie, and
    this test is here to make that visible.
    """
    m25 = _coarse_ahmed(25.0)
    s25 = solve_3d(m25, ground=True, wake_model=True, L_ref=1.044)
    m30 = _coarse_ahmed(30.0)
    s30 = solve_3d(m30, ground=True, wake_model=True, L_ref=1.044)
    # experiment: steep RISE from 25 to 30 (0.285 -> 0.378, +33%)
    # source method: flat, both slants fully separated
    rise = (s30.Cd - s25.Cd) / s25.Cd
    assert abs(rise) < 0.15, (
        "the sweep now shows vortex-peak structure between 25 and 30 degrees; "
        "a source-only method cannot honestly produce that — check what was tuned")