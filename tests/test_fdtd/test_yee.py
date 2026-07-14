"""docs/fdtd_module_plan.md Section 7.1 -- Yee offsets, pure arithmetic, no
solver."""
import numpy as np
import pytest

from cavity_perturbation.fdtd.grid.yee import COMPONENT_OFFSETS, YeeGrid


def test_component_offsets_match_section_1_table():
    expected = {
        "Ex": (0.5, 0.0, 0.0),
        "Ey": (0.0, 0.5, 0.0),
        "Ez": (0.0, 0.0, 0.5),
        "Hx": (0.0, 0.5, 0.5),
        "Hy": (0.5, 0.0, 0.5),
        "Hz": (0.5, 0.5, 0.0),
    }
    assert COMPONENT_OFFSETS == expected


@pytest.mark.parametrize("component,expected_offset", list(COMPONENT_OFFSETS.items()))
def test_component_coords_places_first_cell_at_half_cell_offset(component, expected_offset):
    grid = YeeGrid(shape=(3, 4, 5), cell_size=(0.1, 0.2, 0.3))
    coords = grid.component_coords(component)
    assert coords.shape == (3 * 4 * 5, 3)
    first = coords[0]
    expected = np.array(expected_offset) * np.array([0.1, 0.2, 0.3])
    assert np.allclose(first, expected)


def test_component_coords_general_cell_index_arithmetic():
    grid = YeeGrid(shape=(4, 4, 4), cell_size=(1.0, 1.0, 1.0))
    coords = grid.component_coords("Hz").reshape(4, 4, 4, 3)
    # cell (2, 1, 3): corner at (2,1,3), Hz offset (0.5, 0.5, 0.0)
    assert np.allclose(coords[2, 1, 3], [2.5, 1.5, 3.0])


def test_origin_shifts_all_components_uniformly():
    origin = np.array([1.0, -2.0, 0.5])
    grid = YeeGrid(shape=(2, 2, 2), cell_size=(1.0, 1.0, 1.0), origin=origin)
    coords = grid.component_coords("Ex")
    assert np.allclose(coords[0], origin + [0.5, 0.0, 0.0])


def test_cell_volume():
    grid = YeeGrid(shape=(2, 3, 4), cell_size=(0.1, 0.2, 0.5))
    assert grid.cell_volume == pytest.approx(0.1 * 0.2 * 0.5)


def test_rejects_nonpositive_shape_or_cell_size():
    with pytest.raises(ValueError):
        YeeGrid(shape=(0, 1, 1), cell_size=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        YeeGrid(shape=(1, 1, 1), cell_size=(-1.0, 1.0, 1.0))


def test_unknown_component_raises():
    grid = YeeGrid(shape=(2, 2, 2), cell_size=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        grid.component_coords("Ew")
