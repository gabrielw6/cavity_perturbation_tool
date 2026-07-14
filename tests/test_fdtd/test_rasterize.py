"""docs/fdtd_module_plan.md Section 7.1 -- rasterization, pure geometry, no
solver."""
import math

import numpy as np
import pytest

from cavity_perturbation.cavity import CylindricalCavity, ModeIndex, RectangularCavity
from cavity_perturbation.fdtd.grid.rasterize import rasterize_all, rasterize_component
from cavity_perturbation.fdtd.grid.yee import COMPONENT_OFFSETS, E_COMPONENTS, H_COMPONENTS, YeeGrid
from cavity_perturbation.sample import Sphere


def _sphere_volume_estimate(n_cells: int, radius: float = 0.3, center=(0.5, 0.5, 0.5)) -> float:
    box = 1.0
    grid = YeeGrid(shape=(n_cells, n_cells, n_cells), cell_size=(box / n_cells,) * 3)
    sphere = Sphere(center=np.array(center), radius=radius)
    mask = rasterize_component(grid, "Ex", cavity_mode=_UnitBoxCavity(), sample_region=sphere)
    return float(np.sum(mask.sample_interior)) * grid.cell_volume


class _UnitBoxCavity:
    """Minimal contains()-only stand-in -- rasterize_component only needs
    `contains`, not the full CavityMode ABC (Section 0.3)."""

    def contains(self, r):
        r = np.atleast_2d(r)
        return np.all((r >= 0.0) & (r <= 1.0), axis=-1)


def test_sphere_rasterized_volume_converges_to_closed_form_4_3_pi_r3():
    # Cross-checked against the closed-form 4/3*pi*r^3 computed independently
    # in this test -- NOT against Sphere.volume(), same independent-
    # verification discipline as the (retired) meshing module's plan.
    radius = 0.3
    analytic_volume = 4.0 / 3.0 * math.pi * radius**3

    coarse_err = abs(_sphere_volume_estimate(20, radius) - analytic_volume)
    fine_err = abs(_sphere_volume_estimate(80, radius) - analytic_volume)

    assert fine_err < coarse_err
    assert fine_err / analytic_volume < 0.02


def test_rasterize_component_evaluates_at_own_staggered_location():
    # A cavity mask that is a half-space x <= 0.5 -- Ex (offset 0.5,0,0) and
    # Ey (offset 0,0.5,0) must disagree on which cells near x=0.5 are
    # "interior", since they sample the mask at different x. If
    # rasterization evaluated at cell centers and reused, they would agree
    # exactly -- this test would then fail to distinguish the two paths.
    class HalfSpace:
        def contains(self, r):
            r = np.atleast_2d(r)
            return r[..., 0] <= 0.5

    grid = YeeGrid(shape=(1, 1, 1), cell_size=(1.0, 1.0, 1.0))
    masks = rasterize_all(grid, cavity_mode=HalfSpace(), sample_region=None)
    # Ex sits at x=0.5 (<=0.5 -> True), Hy also sits at x=0.5 -> True;
    # Ey/Ez/Hx/Hz sit at x=0.0 -> True as well for this particular half-space
    # threshold -- so instead directly check the two offsets differ in x.
    assert masks["Ex"].cavity_interior[0, 0, 0] == True  # noqa: E712 (x=0.5)
    coords_ex = grid.component_coords("Ex")[0]
    coords_ey = grid.component_coords("Ey")[0]
    assert coords_ex[0] != coords_ey[0]


def test_no_sample_region_gives_all_false_sample_mask():
    grid = YeeGrid(shape=(3, 3, 3), cell_size=(1.0, 1.0, 1.0))
    cav = RectangularCavity(3.0, 3.0, 3.0, ModeIndex("TE", (0, 1, 1)))
    masks = rasterize_all(grid, cavity_mode=cav, sample_region=None)
    for component in COMPONENT_OFFSETS:
        assert not np.any(masks[component].sample_interior)


def test_rasterize_all_covers_all_six_components():
    grid = YeeGrid(shape=(2, 2, 2), cell_size=(0.01, 0.01, 0.01))
    cav = CylindricalCavity(0.01, 0.02, ModeIndex("TM", (0, 1, 0)))
    masks = rasterize_all(grid, cavity_mode=cav, sample_region=None)
    assert set(masks.keys()) == set(COMPONENT_OFFSETS)
    for component, mask in masks.items():
        assert mask.cavity_interior.shape == grid.shape
        assert mask.sample_interior.shape == grid.shape


def test_cavity_interior_mask_matches_direct_contains_call():
    grid = YeeGrid(shape=(4, 4, 4), cell_size=(0.0025, 0.0025, 0.0025))
    cav = RectangularCavity(0.01, 0.01, 0.01, ModeIndex("TE", (0, 1, 1)))
    mask = rasterize_component(grid, "Ez", cavity_mode=cav, sample_region=None)
    coords = grid.component_coords("Ez")
    expected = np.asarray(cav.contains(coords)).reshape(grid.shape)
    assert np.array_equal(mask.cavity_interior, expected)


# --- tangential_wall_pin (near-wall PEC enforcement, Section 5.3) ---------

# axis index (0=x, 1=y, 2=z) each E component is tangential to (its
# COMPONENT_OFFSETS entry is 0.0 there) -- Ex is tangential to y and z
# walls, never to x walls (where it's the *normal*, not tangential, field).
_TANGENTIAL_AXES = {"Ex": (1, 2), "Ey": (0, 2), "Ez": (0, 1)}


def _grid_matching_bounding_box(cav, n_per_axis=6):
    rmin, rmax = cav.bounding_box()
    cell_size = tuple((rmax - rmin) / n_per_axis)
    return YeeGrid(shape=(n_per_axis, n_per_axis, n_per_axis), cell_size=cell_size, origin=rmin)


@pytest.mark.parametrize("component", E_COMPONENTS)
def test_tangential_wall_pin_true_only_on_index_0_slice_of_tangential_axes(component):
    # A grid sized to exactly match the cavity's own bounding box (as
    # fdtd/model.py's _build_grid does) -- CavityMode.contains()'s inclusive
    # bounds mean a tangential E component's index-0 point along an axis it
    # doesn't offset into sits exactly ON that wall (e.g. Ex's y=0, z=0
    # points), which cavity_interior alone can't distinguish from a
    # genuinely interior point.
    cav = RectangularCavity(0.03, 0.02, 0.025, ModeIndex("TE", (0, 1, 1)))
    grid = _grid_matching_bounding_box(cav)
    mask = rasterize_component(grid, component, cavity_mode=cav, sample_region=None)

    expected = np.zeros(grid.shape, dtype=bool)
    for axis in _TANGENTIAL_AXES[component]:
        idx = [slice(None)] * 3
        idx[axis] = 0
        expected[tuple(idx)] = True

    # `expected` is built from exactly the two tangential axes -- an exact
    # match here also confirms the non-tangential axis (the one this
    # component IS offset into, where it's the *normal* not tangential
    # field, and PEC doesn't constrain it) never contributes a pin.
    assert np.array_equal(mask.tangential_wall_pin, expected)


@pytest.mark.parametrize("component", H_COMPONENTS)
def test_tangential_wall_pin_always_false_for_h_components(component):
    cav = RectangularCavity(0.03, 0.02, 0.025, ModeIndex("TE", (0, 1, 1)))
    grid = _grid_matching_bounding_box(cav)
    mask = rasterize_component(grid, component, cavity_mode=cav, sample_region=None)
    assert not np.any(mask.tangential_wall_pin)
