"""FDTD Module -- grid/rasterize.py: material rasterization.

For each of the six field-component grids, builds the boolean membership
masks the update-coefficient assembly (materials.py) needs, by evaluating
`contains()` at that component's own staggered location -- per
docs/fdtd_module_plan.md Section 2, never once at cell centers and reused,
since that would bias the effective sample volume by a half-cell.

No Maxwell's-equations or time-stepping knowledge -- only geometry.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...cavity import CavityMode
from ...sample import SampleRegion
from .yee import COMPONENT_OFFSETS, YeeGrid

Array = np.ndarray


@dataclass(frozen=True)
class ComponentMask:
    """Boolean membership, shape `grid.shape`, for one field component."""

    cavity_interior: Array
    sample_interior: Array


def rasterize_component(
    grid: YeeGrid,
    component: str,
    cavity_mode: CavityMode,
    sample_region: SampleRegion | None,
) -> ComponentMask:
    """Rasterize `component`'s own staggered grid against `cavity_mode` and
    (if given) `sample_region`, both consumed only through their `contains()`
    method (Section 0.3: no mesh, no CAD)."""
    coords = grid.component_coords(component)
    cavity_interior = np.asarray(cavity_mode.contains(coords)).reshape(grid.shape)
    if sample_region is None:
        sample_interior = np.zeros(grid.shape, dtype=bool)
    else:
        sample_interior = np.asarray(sample_region.contains(coords)).reshape(grid.shape)
    return ComponentMask(cavity_interior=cavity_interior, sample_interior=sample_interior)


def rasterize_all(
    grid: YeeGrid,
    cavity_mode: CavityMode,
    sample_region: SampleRegion | None,
) -> dict[str, ComponentMask]:
    """`rasterize_component` for all six field components."""
    return {
        component: rasterize_component(grid, component, cavity_mode, sample_region)
        for component in COMPONENT_OFFSETS
    }
