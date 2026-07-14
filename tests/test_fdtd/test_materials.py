"""materials.py -- not a named file in docs/fdtd_module_plan.md Section 8's
test list, but its own conductivity-match and Ca/Cb-assembly formulas
deserve a direct unit test, isolating an assembly bug from a stepping bug
(same rationale as ritz.py's N=1-reduction test)."""
import numpy as np
import pytest
from scipy import constants

from cavity_perturbation.fdtd.grid.rasterize import ComponentMask
from cavity_perturbation.fdtd.grid.yee import E_COMPONENTS, YeeGrid
from cavity_perturbation.fdtd.materials import assemble_e_coefficients, matched_conductivity
from cavity_perturbation.sample import Material


def test_matched_conductivity_zero_for_lossless_material():
    mat = Material.from_loss_tangent(eps_r=4.0, tan_delta_e=0.0)
    assert matched_conductivity(mat.eps, f0=1e9) == pytest.approx(0.0)


def test_matched_conductivity_matches_closed_form():
    eps_r_real, tan_delta = 4.0, 0.02
    mat = Material.from_loss_tangent(eps_r=eps_r_real, tan_delta_e=tan_delta)
    f0 = 2.45e9
    sigma = matched_conductivity(mat.eps, f0)
    expected = 2.0 * np.pi * f0 * constants.epsilon_0 * eps_r_real * tan_delta
    assert sigma == pytest.approx(expected, rel=1e-12)


def test_matched_conductivity_rejects_nonpositive_eps_real():
    with pytest.raises(ValueError):
        matched_conductivity(eps_r=complex(0.0, -0.1), f0=1e9)


def _uniform_masks(grid: YeeGrid, sample_interior: np.ndarray) -> dict[str, ComponentMask]:
    cavity_interior = np.ones(grid.shape, dtype=bool)
    return {c: ComponentMask(cavity_interior=cavity_interior, sample_interior=sample_interior) for c in E_COMPONENTS}


def test_no_sample_gives_lossless_background_coefficients_everywhere():
    grid = YeeGrid(shape=(2, 2, 2), cell_size=(1e-3, 1e-3, 1e-3))
    dt = 1e-13
    eps_bg = constants.epsilon_0
    masks = _uniform_masks(grid, np.zeros(grid.shape, dtype=bool))

    coeffs = assemble_e_coefficients(grid, dt, eps_bg, masks, f0=1e9, sample_material=None)

    for component in E_COMPONENTS:
        assert np.allclose(coeffs.Ca[component], 1.0)
        assert np.allclose(coeffs.Cb[component], dt / eps_bg)


def test_lossy_sample_cells_get_matched_coefficients():
    grid = YeeGrid(shape=(2, 2, 2), cell_size=(1e-3, 1e-3, 1e-3))
    dt = 1e-13
    eps_bg = constants.epsilon_0
    f0 = 1e9
    eps_r_real, tan_delta = 5.0, 0.05
    material = Material.from_loss_tangent(eps_r=eps_r_real, tan_delta_e=tan_delta)

    sample_interior = np.zeros(grid.shape, dtype=bool)
    sample_interior[0, 0, 0] = True
    masks = _uniform_masks(grid, sample_interior)

    coeffs = assemble_e_coefficients(grid, dt, eps_bg, masks, f0=f0, sample_material=material)

    eps_sample_abs = eps_r_real * constants.epsilon_0
    sigma = matched_conductivity(material.eps, f0)
    half_loss = sigma * dt / (2.0 * eps_sample_abs)
    expected_Ca = (1.0 - half_loss) / (1.0 + half_loss)
    expected_Cb = (dt / eps_sample_abs) / (1.0 + half_loss)

    for component in E_COMPONENTS:
        assert coeffs.Ca[component][0, 0, 0] == pytest.approx(expected_Ca)
        assert coeffs.Cb[component][0, 0, 0] == pytest.approx(expected_Cb)
        # a cell outside the sample keeps the lossless background coefficients
        assert coeffs.Ca[component][1, 1, 1] == pytest.approx(1.0)
        assert coeffs.Cb[component][1, 1, 1] == pytest.approx(dt / eps_bg)


def test_lossless_dielectric_sample_still_gives_ca_equal_one():
    grid = YeeGrid(shape=(2, 2, 2), cell_size=(1e-3, 1e-3, 1e-3))
    dt = 1e-13
    eps_bg = constants.epsilon_0
    material = Material.from_loss_tangent(eps_r=9.0, tan_delta_e=0.0)  # lossless, eps_r != 1
    sample_interior = np.ones(grid.shape, dtype=bool)
    masks = _uniform_masks(grid, sample_interior)

    coeffs = assemble_e_coefficients(grid, dt, eps_bg, masks, f0=1e9, sample_material=material)

    for component in E_COMPONENTS:
        assert np.allclose(coeffs.Ca[component], 1.0)
        assert np.allclose(coeffs.Cb[component], dt / (9.0 * constants.epsilon_0))
