"""
Closed-loop tests for Layer 6 — photo -> geometry.

The whole credibility of the vision layer rests on one idea: core/geometry.py
runs FORWARD (parameters -> silhouette), so vision is its INVERSE, and the
inverse can be checked against ground truth for free. Render each known car,
recover its geometry from the pixels, solve, and confirm the recovered Cd
matches the car the solver started from.

This is the same honesty pattern as the analytic sphere (test_aero_3d) and the
synthetic coastdown (test_coastdown): the tool is only trusted on real input
because it demonstrably recovers the truth on synthetic input.
"""

import numpy as np
import pytest

from core.panel_solver import INDIAN_CARS, solve_car, get_car_params, ARCHETYPES
from core.vision import (extract_shape, estimate_car, estimate_from_landmarks,
                         classify_archetype, synthesize_side_view, clean_mask,
                         LANDMARK_NAMES)

ALL = list(INDIAN_CARS)


# ════════════════════════════════════════════════════════════════
# THE CORE CLOSED LOOP: render -> recover -> solve
# ════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("key", ALL)
def test_recovers_cd_from_rendered_silhouette(key):
    """
    Render the car, recover its geometry from the image, solve, and the Cd must
    land within 10% of the real car — i.e. the vision step adds no more error
    than the drag model already carries against published figures.
    """
    mask, truth = synthesize_side_view(key)
    est = estimate_car(mask, wheelbase_m=truth["car"]["wheelbase_m"], n_panels=500)
    cd_true = solve_car(key)["Cd"]
    err = abs(est.Cd - cd_true) / cd_true
    assert err < 0.10, f"{key}: recovered Cd {est.Cd:.3f} vs {cd_true:.3f} ({err*100:.1f}%)"


@pytest.mark.parametrize("key", ALL)
def test_classifies_the_body_type(key):
    """The recovered archetype must match the car's real class."""
    mask, truth = synthesize_side_view(key)
    est = estimate_car(mask, wheelbase_m=truth["car"]["wheelbase_m"])
    assert est.archetype == INDIAN_CARS[key]["archetype"], \
        f"{key}: called it a {est.archetype}"


@pytest.mark.parametrize("key", ALL)
def test_recovers_the_shape_angles(key):
    """Windshield and backlight rake within a few degrees of the truth."""
    params, _ = get_car_params(key)
    mask, _ = synthesize_side_view(key)
    shape = extract_shape(mask)
    assert abs(shape.windshield_angle_deg - params["windshield_angle_deg"]) < 5
    assert abs(shape.rear_window_angle_deg - params["rear_window_angle_deg"]) < 8


@pytest.mark.parametrize("key", ALL)
def test_recovers_scale_from_wheelbase(key):
    """Wheelbase-anchored length within 3% of the real car."""
    mask, truth = synthesize_side_view(key)
    est = estimate_car(mask, wheelbase_m=truth["car"]["wheelbase_m"])
    err = abs(est.length_m - truth["car"]["length_m"]) / truth["car"]["length_m"]
    assert err < 0.03, f"{key}: length {est.length_m:.2f} vs {truth['car']['length_m']:.2f}"


# ════════════════════════════════════════════════════════════════
# ROBUSTNESS — the mask a real segmentation actually hands you
# ════════════════════════════════════════════════════════════════

def test_survives_a_ragged_mask():
    """A real segmentation has speckle and holes. clean_mask must recover the
    car, and the recovered Cd must barely move."""
    mask, truth = synthesize_side_view("maruti_swift")
    rng = np.random.default_rng(0)
    noisy = mask.copy()
    # punch holes (window/reflection dropouts) and sprinkle stray blobs
    for _ in range(40):
        r = rng.integers(0, mask.shape[0] - 8)
        c = rng.integers(0, mask.shape[1] - 8)
        noisy[r:r + 6, c:c + 6] = rng.random() > 0.5
    clean_cd = estimate_car(mask, wheelbase_m=truth["car"]["wheelbase_m"]).Cd
    noisy_cd = estimate_car(noisy, wheelbase_m=truth["car"]["wheelbase_m"]).Cd
    assert abs(noisy_cd - clean_cd) / clean_cd < 0.08


def test_orientation_is_normalised():
    """A photo of the car facing the other way must give the same answer."""
    mask, truth = synthesize_side_view("tata_nexon")
    flipped = mask[:, ::-1]
    a = estimate_car(mask, wheelbase_m=truth["car"]["wheelbase_m"]).Cd
    b = estimate_car(flipped, wheelbase_m=truth["car"]["wheelbase_m"]).Cd
    assert abs(a - b) / a < 0.05, "facing direction changed the result"


def test_empty_mask_is_rejected():
    with pytest.raises(ValueError):
        extract_shape(np.zeros((100, 200), bool))


# ════════════════════════════════════════════════════════════════
# THE LANDMARK PATH (what the web scanner runs) survives tap noise
# ════════════════════════════════════════════════════════════════

def _true_landmarks(key, width_px=1000, pad=60):
    """Where the seven landmarks actually sit on the rendered car — used to
    simulate a user tapping them (with human imprecision added by the test)."""
    from core.geometry import build_profile
    params, car = get_car_params(key)
    coords, meta = build_profile(params, 500)
    L = car["length_m"]
    x, y = coords[:, 0], coords[:, 1]
    s = (width_px - 2 * pad) / (x.max() - x.min())
    ymax = y.max()
    from core.vision import WHEEL_RADIUS_M
    rw, wb = WHEEL_RADIUS_M / L, car["wheelbase_m"] / L
    H = params["height_m"] / L

    def X(xn): return (xn - x.min()) * s + pad
    def Yimg(yn): return (ymax - yn) * s + pad     # image y (down)

    return {
        "front_wheel": (X(0.16), Yimg(0.0)),
        "rear_wheel": (X(0.16 + wb), Yimg(0.0)),
        "nose": (X(0.0), Yimg(0.42 * H)),
        "windshield_base": (X(0.32), Yimg(0.62 * H)),
        "roof_front": (X(meta["roof_start"]), Yimg(H)),
        "roof_rear": (X(meta["roof_end"]), Yimg(H)),
        "tail_top": (X(1.0), Yimg(params["trunk_height_norm"] * H)),
    }


@pytest.mark.parametrize("key", ["maruti_swift", "honda_city", "tata_nexon"])
def test_landmark_path_recovers_cd(key):
    """Seven exact taps -> Cd within 10% (this is the JS path's algorithm)."""
    pts = _true_landmarks(key)
    est = estimate_from_landmarks(pts, wheelbase_m=INDIAN_CARS[key]["wheelbase_m"])
    cd_true = solve_car(key)["Cd"]
    assert abs(est.Cd - cd_true) / cd_true < 0.10


@pytest.mark.parametrize("key", ["maruti_swift", "tata_nexon"])
def test_landmark_path_tolerates_shaky_taps(key):
    """
    Humans do not tap exactly. Add pixel jitter of ~1% of the car's length to
    every landmark and the Cd must still land within 12% — the guarantee that
    the web scanner works for a real, imprecise user.
    """
    pts = _true_landmarks(key)
    rng = np.random.default_rng(3)
    jitter = 0.01 * 1000                    # ~1% of the 1000-px car length
    errs = []
    for _ in range(6):
        shaky = {k: (x + rng.normal(0, jitter), y + rng.normal(0, jitter))
                 for k, (x, y) in pts.items()}
        est = estimate_from_landmarks(shaky, wheelbase_m=INDIAN_CARS[key]["wheelbase_m"])
        errs.append(abs(est.Cd - solve_car(key)["Cd"]) / solve_car(key)["Cd"])
    assert np.mean(errs) < 0.12, f"{key}: mean Cd error under tap jitter {np.mean(errs)*100:.1f}%"


# ════════════════════════════════════════════════════════════════
# CLASSIFIER + HONEST BOUNDS
# ════════════════════════════════════════════════════════════════

def test_classifier_on_canonical_rakes():
    assert classify_archetype(30, 0.66, 0.31) == "sedan"
    assert classify_archetype(55, 0.87, 0.39) == "hatchback"
    assert classify_archetype(75, 0.95, 0.40) == "suv"


@pytest.mark.parametrize("key", ALL)
def test_recovered_car_is_physically_plausible(key):
    """A car reconstructed from a photo must still be a believable car — the
    vision layer cannot manufacture an impossible Cd."""
    mask, truth = synthesize_side_view(key)
    est = estimate_car(mask, wheelbase_m=truth["car"]["wheelbase_m"])
    assert 0.22 < est.Cd < 0.48
    assert est.Cd_uncertainty > 0
