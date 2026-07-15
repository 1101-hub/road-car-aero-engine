"""
Tests for uncertainty propagation.

"₹12,010/year" to five significant figures is how this project's original bug
survived review — false precision reads as authority. These tests pin down the
error-bar machinery so no user-facing number can quietly drop its uncertainty.
"""

import numpy as np
import pytest

from core.panel_solver import solve_car
from core.modifications import (make_budget, apply_mod_set,
                                mod_wheel_covers, mod_underbody_panel,
                                mod_rear_spoiler)
from core.uncertainty import (delta_cd_uncertainty, band, confidence_of,
                              MOD_UNCERTAINTY, BASELINE_SCALE_UNCERTAINTY)
from core.optimizer import run_grid_search


def test_quadrature_combination():
    """Two mods with known deltas must combine in quadrature plus the baseline
    scale term — checked against the closed form."""
    base = solve_car("maruti_swift")
    ms = apply_mod_set(base, [
        (mod_wheel_covers, dict(n_wheels_covered=4)),
        (mod_underbody_panel, dict(coverage_fraction=1.0)),
    ])
    d = {r.mod_name: r.delta_Cd for r in ms.modifications if r.feasible}
    expected = np.sqrt(
        (MOD_UNCERTAINTY["wheel_covers"][0] * d["wheel_covers"]) ** 2
        + (MOD_UNCERTAINTY["underbody_panel"][0] * d["underbody_panel"]) ** 2
        + (BASELINE_SCALE_UNCERTAINTY * ms.delta_Cd_total) ** 2)
    assert delta_cd_uncertainty(ms.modifications) == pytest.approx(expected)


def test_uncertainty_is_meaningful_but_not_absurd():
    """The error bar must be a real fraction of the value — neither the old
    false precision (~0) nor wider than the value itself for HIGH-confidence
    modifications."""
    base = solve_car("maruti_swift")
    ms = apply_mod_set(base, [
        (mod_wheel_covers, dict(n_wheels_covered=4)),
        (mod_underbody_panel, dict(coverage_fraction=1.0)),
    ])
    unc = delta_cd_uncertainty(ms.modifications)
    rel = unc / ms.delta_Cd_total
    assert 0.15 < rel < 0.60, f"relative uncertainty {rel:.2f} is not credible"


def test_infeasible_mods_contribute_no_uncertainty():
    """A modification that was not fitted has neither effect nor error bar."""
    base = solve_car("maruti_swift")
    ms_with = apply_mod_set(base, [
        (mod_wheel_covers, dict(n_wheels_covered=4)),
        (mod_rear_spoiler, dict(angle_deg=25.0)),          # stalled -> infeasible
    ])
    ms_without = apply_mod_set(base, [
        (mod_wheel_covers, dict(n_wheels_covered=4)),
    ])
    assert (delta_cd_uncertainty(ms_with.modifications)
            == pytest.approx(delta_cd_uncertainty(ms_without.modifications)))


def test_band_scales_linearly():
    """Savings are linear in delta-Cd, so the band must carry the same relative
    width whatever the unit (litres, rupees, tonnes)."""
    b_rupees = band(3000.0, delta_Cd=0.03, delta_Cd_unc=0.009)
    b_litres = band(0.21, delta_Cd=0.03, delta_Cd_unc=0.009)
    assert b_rupees.uncertainty / b_rupees.value == pytest.approx(0.3)
    assert b_litres.uncertainty / b_litres.value == pytest.approx(0.3)
    assert b_rupees.low >= 0.0


def test_every_catalogue_mod_has_a_confidence_tag():
    for name in ["wheel_covers", "underbody_panel", "front_splitter",
                 "side_skirts", "rear_diffuser", "rear_spoiler"]:
        tag, why = confidence_of(name)
        assert tag in ("HIGH", "MEDIUM", "LOW")
        assert len(why) > 10, f"{name}: confidence must come with a reason"


def test_spoiler_is_least_certain():
    """Structural, not aesthetic: the spoiler's delta is the small difference
    of two competing terms, so it must carry the widest relative bar."""
    rels = {n: MOD_UNCERTAINTY[n][0] for n in MOD_UNCERTAINTY}
    assert rels["rear_spoiler"] == max(rels.values())
    assert MOD_UNCERTAINTY["rear_spoiler"][1] == "LOW"


def test_optimizer_solutions_carry_bands():
    """Every solution the user can see must ship with its error bars filled."""
    sols = run_grid_search("maruti_swift", legal_only=True)
    assert sols
    for s in sols:
        assert s.delta_Cd_unc > 0
        assert s.annual_saving_unc > 0
        rel = s.annual_saving_unc / s.annual_saving_INR
        assert 0.1 < rel < 0.8, f"{s.mod_names}: band {rel:.2f} not credible"