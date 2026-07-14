"""docs/fdtd_module_plan.md Section 3 -- excitation profile and probe
placement."""
import numpy as np
import pytest

from cavity_perturbation.cavity import ModeIndex, RectangularCavity
from cavity_perturbation.fdtd.grid.yee import E_COMPONENTS, YeeGrid
from cavity_perturbation.fdtd.source import (
    build_modal_source,
    choose_probe_point,
    gaussian_modulated_pulse,
    gaussian_pulse_sigma_t,
)
from cavity_perturbation.fields import AnalyticalField


def _te011_cavity():
    return RectangularCavity(0.03, 0.02, 0.025, ModeIndex("TE", (0, 1, 1)))


def test_build_modal_source_returns_real_arrays_of_grid_shape():
    cav = _te011_cavity()
    field = AnalyticalField(cav)
    grid = YeeGrid(shape=(4, 4, 4), cell_size=(cav.a / 4, cav.b / 4, cav.c / 4))

    profile = build_modal_source(grid, field)

    assert set(profile.keys()) == set(E_COMPONENTS)
    for component, arr in profile.items():
        assert arr.shape == grid.shape
        assert np.isrealobj(arr)
        assert np.all(np.isfinite(arr))


def test_build_modal_source_ez_component_is_identically_zero_for_te011():
    # TE_0,1,1: kx=0 -> Ey ~ sin(kx x)=0 identically, Ez=0 identically (TE_z
    # has no Ez by construction) -- a real-valued closed-form check the
    # extraction shouldn't disturb.
    cav = _te011_cavity()
    field = AnalyticalField(cav)
    grid = YeeGrid(shape=(3, 3, 3), cell_size=(cav.a / 3, cav.b / 3, cav.c / 3))
    profile = build_modal_source(grid, field)
    assert np.allclose(profile["Ez"], 0.0)
    assert np.allclose(profile["Ey"], 0.0)
    assert not np.allclose(profile["Ex"], 0.0)


def test_build_modal_source_recovers_correct_magnitude_after_phase_removal():
    cav = _te011_cavity()
    field = AnalyticalField(cav)
    grid = YeeGrid(shape=(4, 4, 4), cell_size=(cav.a / 4, cav.b / 4, cav.c / 4))
    profile = build_modal_source(grid, field)

    coords = grid.component_coords("Ex")
    complex_ex = field.E(coords)[..., 0].reshape(grid.shape)
    # magnitude must be preserved exactly by the phase-removal step
    assert np.allclose(np.abs(profile["Ex"]), np.abs(complex_ex))


def test_gaussian_pulse_sigma_t_formula_and_validation():
    bandwidth = 5e7
    sigma_t = gaussian_pulse_sigma_t(bandwidth)
    assert sigma_t == pytest.approx(1.0 / (2.0 * np.pi * bandwidth))
    with pytest.raises(ValueError):
        gaussian_pulse_sigma_t(0.0)
    with pytest.raises(ValueError):
        gaussian_pulse_sigma_t(-1.0)


def test_gaussian_modulated_pulse_peaks_at_t0_with_unit_envelope():
    f0, t0, sigma_t = 1e9, 5e-9, 1e-9
    value_at_t0 = gaussian_modulated_pulse(np.array([t0]), f0, t0, sigma_t)[0]
    assert value_at_t0 == pytest.approx(1.0)


def test_gaussian_modulated_pulse_decays_away_from_t0():
    f0, t0, sigma_t = 1e9, 5e-9, 1e-9
    t = np.array([t0, t0 + 3 * sigma_t, t0 + 10 * sigma_t])
    values = gaussian_modulated_pulse(t, f0, t0, sigma_t)
    assert abs(values[1]) < abs(values[0])
    assert abs(values[2]) < 1e-8


def test_choose_probe_point_lands_inside_cavity_at_dominant_component():
    cav = _te011_cavity()
    field = AnalyticalField(cav)
    point, component = choose_probe_point(cav, field)
    assert cav.contains(point)[0]
    assert component == "Ex"  # TE_011's only nonzero field component


def test_choose_probe_point_near_the_known_analytic_maximum():
    # TE_0,1,1's Ex ~ sin(ky y) sin(kz z), independent of x -- maximized at
    # y=b/2, z=c/2 for any x. The coarse structured search should land
    # close to that plane.
    cav = _te011_cavity()
    field = AnalyticalField(cav)
    point, _ = choose_probe_point(cav, field, n_per_axis=21)
    assert point[1] == pytest.approx(cav.b / 2.0, abs=cav.b / 20.0)
    assert point[2] == pytest.approx(cav.c / 2.0, abs=cav.c / 20.0)
