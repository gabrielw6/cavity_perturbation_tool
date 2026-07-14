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
from .yee import COMPONENT_OFFSETS, E_COMPONENTS, YeeGrid

Array = np.ndarray


@dataclass(frozen=True)
class ComponentMask:
    """Boolean membership, shape `grid.shape`, for one field component.

    `tangential_wall_pin` is the near-wall (index-0) half of Section 5.3's
    PEC enforcement: True where this E component is tangential to a cavity
    wall it sits exactly on (always all-False for H components -- PEC only
    forces tangential *E* to zero, never H). See `_tangential_wall_pin` for
    why this can't be folded into `cavity_interior`."""

    cavity_interior: Array
    sample_interior: Array
    tangential_wall_pin: Array


def _tangential_wall_pin(
    grid: YeeGrid,
    component: str,
    cavity_mode: CavityMode,
    cavity_interior: Array,
    coords: Array,
) -> Array:
    """Section 5.3 PEC enforcement, the near-wall (index-0) half.

    `CavityMode.contains()` uses inclusive bounds (e.g. 0<=x<=a), so a
    wall-coincident grid point reports `cavity_interior=True` -- the
    existing `cavity_interior` mask alone never flags it. But
    `stepper.py`'s `_backward_diff` (the curl(H)->E update) has no notion
    of "this point sits on the wall": at index 0 along a tangential axis it
    is driven by a genuine neighboring H value like any other interior
    point, and evolves freely instead of staying pinned at zero, as PEC
    requires for a *tangential* E component. (The analogous far-wall case
    is already handled correctly by `_forward_diff`'s own zero-padding, and
    this grid layout's offset-0 axes never place a real point exactly on
    the far wall in the first place -- see `YeeGrid.component_coords`.)

    Detected geometrically, not from cavity-type-specific bounds: a point
    is "on the wall along axis `ax`" if it is itself `contains()==True` but
    a point half a cell further in the -ax direction is `contains()==False`
    -- i.e. it is the very first interior layer along that axis. OR'd
    across every axis this component is tangential to (an integer,
    zero-valued `COMPONENT_OFFSETS` entry along that axis; H components
    have none checked at all -- see `rasterize_component`).

    TODO: only pins wall points reachable by stepping -0.5 cell along a
    single Cartesian axis -- correct for the flat (rectangular, and
    cylindrical/coaxial end-cap) walls every cavity type here has along at
    least one axis, but a curved radial wall (e.g. a cylindrical cavity's
    rho=a side) is not axis-aligned and would need a more general surface-
    normal neighbor check; out of scope for this fix.
    """
    offsets = COMPONENT_OFFSETS[component]
    pin = np.zeros(grid.shape, dtype=bool)
    for axis in range(3):
        if offsets[axis] != 0.0:
            continue  # not tangential along this axis -- no pin needed here
        shift = np.zeros(3)
        shift[axis] = -0.5 * grid.cell_size[axis]
        outside_along_axis = ~np.asarray(cavity_mode.contains(coords + shift)).reshape(grid.shape)
        pin |= cavity_interior & outside_along_axis
    return pin


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

    if component in E_COMPONENTS:
        tangential_wall_pin = _tangential_wall_pin(grid, component, cavity_mode, cavity_interior, coords)
    else:
        tangential_wall_pin = np.zeros(grid.shape, dtype=bool)

    return ComponentMask(
        cavity_interior=cavity_interior,
        sample_interior=sample_interior,
        tangential_wall_pin=tangential_wall_pin,
    )


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
