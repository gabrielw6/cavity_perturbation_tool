"""docs/fdtd_module_plan.md Section 7.2 (long-run boundedness) and general
leapfrog update-loop tests."""
import numpy as np
import pytest
from scipy import constants

from cavity_perturbation.cavity import CylindricalCavity, ModeIndex, RectangularCavity
from cavity_perturbation.fdtd import stepper as stepper_module
from cavity_perturbation.fdtd.grid.rasterize import rasterize_all
from cavity_perturbation.fdtd.grid.yee import E_COMPONENTS, H_COMPONENTS, YeeGrid
from cavity_perturbation.fdtd.materials import assemble_e_coefficients
from cavity_perturbation.fdtd.source import (
    build_modal_source,
    choose_probe_point,
    gaussian_modulated_pulse,
    gaussian_pulse_sigma_t,
)
from cavity_perturbation.fdtd.stability import stable_time_step
from cavity_perturbation.fdtd.stepper import FDTDStepper
from cavity_perturbation.fields import AnalyticalField


def test_forward_diff_matches_manual_calculation_with_zero_boundary():
    arr = np.array([1.0, 2.0, 5.0]).reshape(3, 1, 1)
    result = stepper_module._forward_diff(arr, axis=0).reshape(-1)
    # [2-1, 5-2, 0-5] -- last entry uses the fictitious zero neighbor
    assert np.allclose(result, [1.0, 3.0, -5.0])


def test_backward_diff_matches_manual_calculation_with_zero_boundary():
    arr = np.array([1.0, 2.0, 5.0]).reshape(3, 1, 1)
    result = stepper_module._backward_diff(arr, axis=0).reshape(-1)
    # [1-0, 2-1, 5-2] -- first entry uses the fictitious zero neighbor
    assert np.allclose(result, [1.0, 1.0, 3.0])


def _build_empty_cavity_stepper(cav):
    field = AnalyticalField(cav)
    rmin, rmax = cav.bounding_box()
    n_per_axis = 12
    cell_size = tuple((rmax - rmin) / n_per_axis)
    grid = YeeGrid(shape=(n_per_axis, n_per_axis, n_per_axis), cell_size=cell_size, origin=rmin)

    masks = rasterize_all(grid, cavity_mode=cav, sample_region=None)
    dt = stable_time_step(cell_size, cav.epsilon_bg, cav.mu_bg)
    coeffs = assemble_e_coefficients(grid, dt, cav.epsilon_bg, masks, f0=cav.f0, sample_material=None)
    stepper = FDTDStepper(grid, dt, cav.mu_bg, coeffs, masks)
    return stepper, field, dt


def test_single_step_from_rest_reproduces_the_source_exactly():
    cav = RectangularCavity(0.03, 0.02, 0.025, ModeIndex("TE", (0, 1, 1)))
    stepper, field, dt = _build_empty_cavity_stepper(cav)
    profile = build_modal_source(stepper.grid, field)

    stepper.step(e_source=profile)

    for component in E_COMPONENTS:
        expected = profile[component].copy()
        expected[~stepper.masks[component].cavity_interior] = 0.0
        assert np.allclose(stepper.E[component], expected)
    for component in H_COMPONENTS:
        assert np.allclose(stepper.H[component], 0.0)  # H hasn't seen a nonzero E yet


def test_long_run_boundedness_empty_lossless_cavity_does_not_blow_up():
    cav = RectangularCavity(0.03, 0.02, 0.025, ModeIndex("TE", (0, 1, 1)))
    stepper, field, dt = _build_empty_cavity_stepper(cav)
    profile = build_modal_source(stepper.grid, field)

    sigma_t = gaussian_pulse_sigma_t(bandwidth_hz=cav.f0 / 20.0)
    t0 = 5.0 * sigma_t
    n_pulse_steps = int(2.0 * t0 / dt) + 1

    peak_during_pulse = 0.0
    t = 0.0
    for _ in range(n_pulse_steps):
        pulse_val = float(gaussian_modulated_pulse(np.array([t]), cav.f0, t0, sigma_t)[0])
        source = {c: pulse_val * profile[c] for c in E_COMPONENTS}
        stepper.step(e_source=source)
        t += dt
        peak_during_pulse = max(peak_during_pulse, max(np.max(np.abs(stepper.E[c])) for c in E_COMPONENTS))

    assert peak_during_pulse > 0.0

    # Long free-running phase, no further source -- many periods of the mode.
    n_free_steps = int(50.0 / cav.f0 / dt)
    max_after_pulse = 0.0
    for _ in range(n_free_steps):
        stepper.step(e_source=None)
        max_after_pulse = max(max_after_pulse, max(np.max(np.abs(stepper.E[c])) for c in E_COMPONENTS))

    assert np.isfinite(max_after_pulse)
    assert max_after_pulse <= 2.0 * peak_during_pulse


def test_pec_mask_holds_field_at_zero_outside_cavity_interior():
    # A cylindrical cavity's bounding box has corners outside the round
    # cross-section -- those cells must stay exactly zero throughout.
    cav = CylindricalCavity(0.01, 0.02, ModeIndex("TM", (0, 1, 0)))
    field = AnalyticalField(cav)
    rmin, rmax = cav.bounding_box()
    n_per_axis = 10
    cell_size = tuple((rmax - rmin) / n_per_axis)
    grid = YeeGrid(shape=(n_per_axis, n_per_axis, n_per_axis), cell_size=cell_size, origin=rmin)
    masks = rasterize_all(grid, cavity_mode=cav, sample_region=None)
    dt = stable_time_step(cell_size, cav.epsilon_bg, cav.mu_bg)
    coeffs = assemble_e_coefficients(grid, dt, cav.epsilon_bg, masks, f0=cav.f0, sample_material=None)
    stepper = FDTDStepper(grid, dt, cav.mu_bg, coeffs, masks)
    profile = build_modal_source(grid, field)

    for _ in range(20):
        stepper.step(e_source=profile)

    for component in E_COMPONENTS:
        outside = ~masks[component].cavity_interior
        assert np.all(stepper.E[component][outside] == 0.0)
    for component in H_COMPONENTS:
        outside = ~masks[component].cavity_interior
        assert np.all(stepper.H[component][outside] == 0.0)


def test_stepper_rejects_masks_missing_a_component():
    cav = RectangularCavity(0.03, 0.02, 0.025, ModeIndex("TE", (0, 1, 1)))
    grid = YeeGrid(shape=(4, 4, 4), cell_size=(0.03 / 4, 0.02 / 4, 0.025 / 4))
    masks = rasterize_all(grid, cavity_mode=cav, sample_region=None)
    del masks["Ex"]
    dt = stable_time_step(grid.cell_size, cav.epsilon_bg, cav.mu_bg)
    coeffs = assemble_e_coefficients(grid, dt, cav.epsilon_bg, masks | {"Ex": masks["Ey"]}, f0=cav.f0)
    with pytest.raises(ValueError):
        FDTDStepper(grid, dt, cav.mu_bg, coeffs, masks)
