"""docs/gui_module_plan.md Section 8 -- adapters/field_sampling.py."""
import numpy as np
import pytest

from cavity_perturbation_gui.adapters.cavity_adapter import CavityParams
from cavity_perturbation_gui.adapters.fdtd_runner import run_fdtd
from cavity_perturbation_gui.adapters.field_sampling import (
    build_plane_grid,
    plane_through_point,
    sample_closed_form_field,
    sample_fdtd_snapshot,
    sample_ritz_field,
)
from cavity_perturbation_gui.adapters.perturbation_runner import run_perturbation
from cavity_perturbation_gui.adapters.ritz_runner import run_ritz
from cavity_perturbation_gui.adapters.sample_adapter import SampleParams

_CAVITY_PARAMS = CavityParams("rectangular", (0.03, 0.02, 0.025), "TE", (0, 1, 1))
_SAMPLE_PARAMS = SampleParams(shape="sphere", radius=1.5e-3, eps_r=4.5, tan_delta_e=0.01)


def test_plane_through_point_spans_cavity_bounding_box():
    run_result = run_perturbation(_CAVITY_PARAMS, _SAMPLE_PARAMS)
    plane = plane_through_point(run_result.cavity, run_result.sample.region.center, ("x", "y"))
    rmin, rmax = run_result.cavity.bounding_box()
    assert plane.extent == ((rmin[0], rmax[0]), (rmin[1], rmax[1]))
    assert plane.fixed_axis == "z"
    assert plane.fixed_value == pytest.approx(run_result.sample.region.center[2])


def test_build_plane_grid_masks_points_outside_cavity():
    run_result = run_perturbation(_CAVITY_PARAMS, _SAMPLE_PARAMS)
    cav = run_result.cavity
    # A plane deliberately extending past the cavity's own extent.
    plane = plane_through_point(cav, run_result.sample.region.center, ("x", "y"), resolution=(10, 10))
    from dataclasses import replace

    (lo1, hi1), (lo2, hi2) = plane.extent
    plane = replace(plane, extent=((lo1 - 0.01, hi1), (lo2 - 0.01, hi2)))
    grid = build_plane_grid(cav, plane)
    assert not np.all(grid.inside_mask)
    assert np.any(grid.inside_mask)


def test_sample_closed_form_field_masks_outside_points_to_nan():
    run_result = run_perturbation(_CAVITY_PARAMS, _SAMPLE_PARAMS)
    plane = plane_through_point(run_result.cavity, run_result.sample.region.center, ("x", "y"), resolution=(20, 20))
    grid, values = sample_closed_form_field(run_result.cavity, run_result.field_provider, plane, field="E")
    assert values.shape == (20, 20, 3)
    assert np.all(np.isnan(values[~grid.inside_mask]))
    assert not np.any(np.isnan(values[grid.inside_mask]))


def test_sample_closed_form_field_h_differs_from_e():
    run_result = run_perturbation(_CAVITY_PARAMS, _SAMPLE_PARAMS)
    plane = plane_through_point(run_result.cavity, run_result.sample.region.center, ("x", "y"), resolution=(10, 10))
    _, e_values = sample_closed_form_field(run_result.cavity, run_result.field_provider, plane, field="E")
    _, h_values = sample_closed_form_field(run_result.cavity, run_result.field_provider, plane, field="H")
    finite = ~np.isnan(e_values) & ~np.isnan(h_values)
    assert not np.allclose(e_values[finite], h_values[finite])


def test_sample_ritz_field_matches_manual_reconstruction():
    run_result = run_ritz(_CAVITY_PARAMS, _SAMPLE_PARAMS, n_basis=3, conductivity=5.8e7)
    plane = plane_through_point(run_result.cavity, run_result.sample.region.center, ("x", "z"), resolution=(8, 8))
    grid, values = sample_ritz_field(run_result.cavity, run_result.diagnostics, plane)

    point = grid.points[3, 3]
    expected = sum(
        coeff * mode.E(point)
        for coeff, mode in zip(run_result.diagnostics.coefficients, run_result.diagnostics.basis_modes)
    )
    if grid.inside_mask[3, 3]:
        assert np.allclose(values[3, 3], expected)


def test_sample_fdtd_snapshot_rejects_unknown_component():
    run_result = run_fdtd(_CAVITY_PARAMS, _SAMPLE_PARAMS, cells_per_wavelength=8, min_cells_per_axis=6)
    plane = plane_through_point(run_result.cavity, run_result.sample.region.center, ("x", "y"), resolution=(10, 10))
    with pytest.raises(ValueError):
        sample_fdtd_snapshot(run_result.cavity, run_result.diagnostics, plane, component="Ew")


def test_sample_fdtd_snapshot_returns_real_scalar_plane():
    run_result = run_fdtd(_CAVITY_PARAMS, _SAMPLE_PARAMS, cells_per_wavelength=8, min_cells_per_axis=6)
    plane = plane_through_point(run_result.cavity, run_result.sample.region.center, ("x", "y"), resolution=(10, 10))
    grid, values = sample_fdtd_snapshot(run_result.cavity, run_result.diagnostics, plane, component="Ex")
    assert values.shape == (10, 10)
    assert not np.iscomplexobj(values)
