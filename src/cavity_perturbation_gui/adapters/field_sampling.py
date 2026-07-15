"""adapters/field_sampling.py -- plane-of-points -> E/H values, shared
across all four forward tabs (docs/gui_module_plan.md Section 4). Points
outside `cavity.contains()` are masked to NaN so a plot never renders
extrapolated nonsense from outside the cavity's valid domain."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from cavity_perturbation.cavity import CavityMode
from cavity_perturbation.fdtd.diagnostics import FDTDDiagnostics
from cavity_perturbation.fdtd.grid.yee import COMPONENT_OFFSETS
from cavity_perturbation.fields import FieldProvider
from cavity_perturbation.ritz import RitzDiagnostics

Array = np.ndarray
Axis = Literal["x", "y", "z"]
_AXIS_INDEX: dict[Axis, int] = {"x": 0, "y": 1, "z": 2}
_ALL_AXES: tuple[Axis, Axis, Axis] = ("x", "y", "z")


@dataclass(frozen=True)
class PlaneSpec:
    """Two free axes + a fixed third coordinate, spanning `extent` at
    `resolution` points per free axis (docs/gui_module_plan.md Section 4:
    "two axes + a fixed third coordinate")."""

    free_axes: tuple[Axis, Axis]
    fixed_axis: Axis
    fixed_value: float
    extent: tuple[tuple[float, float], tuple[float, float]]
    resolution: tuple[int, int] = (120, 120)


@dataclass(frozen=True)
class PlaneGrid:
    """The sampled point grid + coordinate axes for one `PlaneSpec`, shared
    by every field-sampling function below (built once, reused)."""

    points: Array  # (n1, n2, 3)
    axis1_values: Array  # (n1,)
    axis2_values: Array  # (n2,)
    inside_mask: Array  # (n1, n2) bool -- cavity.contains()


def plane_through_point(
    cavity: CavityMode,
    point: Array,
    free_axes: tuple[Axis, Axis],
    resolution: tuple[int, int] = (120, 120),
) -> PlaneSpec:
    """'Through the sample center' convenience (Section 4): fixes the one
    axis not in `free_axes` at `point`'s own coordinate there, and spans
    the two free axes across the cavity's own bounding box."""
    fixed_axis = next(a for a in _ALL_AXES if a not in free_axes)
    rmin, rmax = cavity.bounding_box()
    i1, i2 = _AXIS_INDEX[free_axes[0]], _AXIS_INDEX[free_axes[1]]
    extent = ((float(rmin[i1]), float(rmax[i1])), (float(rmin[i2]), float(rmax[i2])))
    fixed_value = float(np.asarray(point)[_AXIS_INDEX[fixed_axis]])
    return PlaneSpec(
        free_axes=free_axes, fixed_axis=fixed_axis, fixed_value=fixed_value, extent=extent, resolution=resolution
    )


def build_plane_grid(cavity: CavityMode, plane: PlaneSpec) -> PlaneGrid:
    (lo1, hi1), (lo2, hi2) = plane.extent
    n1, n2 = plane.resolution
    axis1_values = np.linspace(lo1, hi1, n1)
    axis2_values = np.linspace(lo2, hi2, n2)
    A1, A2 = np.meshgrid(axis1_values, axis2_values, indexing="ij")

    points = np.empty((n1, n2, 3))
    i1, i2 = _AXIS_INDEX[plane.free_axes[0]], _AXIS_INDEX[plane.free_axes[1]]
    i_fixed = _AXIS_INDEX[plane.fixed_axis]
    points[..., i1] = A1
    points[..., i2] = A2
    points[..., i_fixed] = plane.fixed_value

    inside = np.asarray(cavity.contains(points.reshape(-1, 3))).reshape(n1, n2)
    return PlaneGrid(points=points, axis1_values=axis1_values, axis2_values=axis2_values, inside_mask=inside)


def sample_closed_form_field(
    cavity: CavityMode,
    field_provider: FieldProvider,
    plane: PlaneSpec,
    field: Literal["E", "H"] = "E",
) -> tuple[PlaneGrid, Array]:
    """Analytical/Perturbational tabs (Section 5): the unperturbed
    closed-form field, evaluated directly from `field_provider.E`/`.H`.
    Perturbational's own field plot is explicitly this, not a resolved
    perturbed field -- Section 5's honesty note, carried through to the
    widget layer's label, not hidden here."""
    grid = build_plane_grid(cavity, plane)
    func = field_provider.E if field == "E" else field_provider.H
    values = func(grid.points.reshape(-1, 3)).reshape(*plane.resolution, 3)
    values = np.where(grid.inside_mask[..., None], values, np.nan)
    return grid, values


def sample_ritz_field(cavity: CavityMode, diagnostics: RitzDiagnostics, plane: PlaneSpec) -> tuple[PlaneGrid, Array]:
    """Variational tab: the genuine Ritz-mixed reconstruction
    E(r) = sum_i c_i E_i(r) over `diagnostics.basis_modes`/`.coefficients`
    (docs/gui_module_plan.md Section 2.3) -- the reconstruction arithmetic
    itself lives here, not in `ritz.py` (Section 0.1: no physics in the GUI
    package, but this is arithmetic on already-computed physics, a view of
    it, not new physics)."""
    grid = build_plane_grid(cavity, plane)
    pts = grid.points.reshape(-1, 3)
    accumulated: Array = np.zeros((pts.shape[0], 3), dtype=complex)
    for coeff, mode in zip(diagnostics.coefficients, diagnostics.basis_modes):
        accumulated = accumulated + coeff * mode.E(pts)
    values: Array = accumulated.reshape(*plane.resolution, 3)
    values = np.where(grid.inside_mask[..., None], values, np.nan)
    return grid, values


def sample_fdtd_snapshot(
    cavity: CavityMode,
    diagnostics: FDTDDiagnostics,
    plane: PlaneSpec,
    component: str = "Ex",
) -> tuple[PlaneGrid, Array]:
    """FDTD tab: a nearest-grid-point slice of the single end-of-excitation
    field snapshot (Section 0.3), one Cartesian component at a time (the
    snapshot stores each component on its own staggered grid, unlike the
    other three tabs' closed-form vector field -- Section 5's per-tab
    honesty applies here too)."""
    if component not in COMPONENT_OFFSETS:
        raise ValueError(f"unknown component {component!r}, expected one of {list(COMPONENT_OFFSETS)}")
    if component not in diagnostics.field_snapshot:
        raise ValueError(f"component {component!r} not present in this snapshot")

    grid = build_plane_grid(cavity, plane)
    snap_grid = diagnostics.snapshot_grid
    field_array = diagnostics.field_snapshot[component]
    offsets = COMPONENT_OFFSETS[component]
    cell_size = snap_grid.cell_size

    pts = grid.points.reshape(-1, 3)
    idx = np.empty((pts.shape[0], 3), dtype=int)
    for axis in range(3):
        raw = (pts[:, axis] - snap_grid.origin[axis]) / cell_size[axis] - offsets[axis]
        idx[:, axis] = np.clip(np.round(raw).astype(int), 0, snap_grid.shape[axis] - 1)

    values = field_array[idx[:, 0], idx[:, 1], idx[:, 2]].reshape(*plane.resolution)
    values = np.where(grid.inside_mask, values, np.nan)
    return grid, values
