"""FDTD Module -- stepper.py: leapfrog E/H update loop, per
docs/fdtd_module_plan.md Section 5.

Standard Yee leapfrog: H updates from curl(E) (Section 5.1), advance a half
step, E updates from curl(H) with the conductivity term (materials.py's
Ca/Cb coefficients), advance a half step. Cells outside a component's own
`cavity_interior` mask (grid/rasterize.py) are held at zero after every
update -- Section 5.3's PEC enforcement, applied uniformly to every field
component, not only E.
"""
from __future__ import annotations

import numpy as np

from .grid.rasterize import ComponentMask
from .grid.yee import COMPONENT_OFFSETS, E_COMPONENTS, H_COMPONENTS, YeeGrid
from .materials import EFieldCoefficients

Array = np.ndarray


def _forward_diff(arr: Array, axis: int) -> Array:
    """(arr[i+1]-arr[i])/1 along `axis`, treating the fictitious arr[n] (one
    past the last index) as zero -- the correct curl(E)->H update behavior
    exactly when the grid's own outer boundary coincides with a PEC cavity
    wall (Section 5.3): the missing forward E neighbor IS the wall, where
    tangential E is zero by definition."""
    pad_shape = list(arr.shape)
    pad_shape[axis] = 1
    extended = np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=axis)
    return np.diff(extended, axis=axis)


def _backward_diff(arr: Array, axis: int) -> Array:
    """(arr[i]-arr[i-1])/1 along `axis`, treating the fictitious arr[-1]
    (one before the first index) as zero -- the curl(H)->E analogue of
    `_forward_diff`."""
    pad_shape = list(arr.shape)
    pad_shape[axis] = 1
    extended = np.concatenate([np.zeros(pad_shape, dtype=arr.dtype), arr], axis=axis)
    return np.diff(extended, axis=axis)


def _curl_e_for_h(E: dict[str, Array], cell_size: tuple[float, float, float]) -> dict[str, Array]:
    dx, dy, dz = cell_size
    return {
        "Hx": _forward_diff(E["Ez"], axis=1) / dy - _forward_diff(E["Ey"], axis=2) / dz,
        "Hy": _forward_diff(E["Ex"], axis=2) / dz - _forward_diff(E["Ez"], axis=0) / dx,
        "Hz": _forward_diff(E["Ey"], axis=0) / dx - _forward_diff(E["Ex"], axis=1) / dy,
    }


def _curl_h_for_e(H: dict[str, Array], cell_size: tuple[float, float, float]) -> dict[str, Array]:
    dx, dy, dz = cell_size
    return {
        "Ex": _backward_diff(H["Hz"], axis=1) / dy - _backward_diff(H["Hy"], axis=2) / dz,
        "Ey": _backward_diff(H["Hx"], axis=2) / dz - _backward_diff(H["Hz"], axis=0) / dx,
        "Ez": _backward_diff(H["Hy"], axis=0) / dx - _backward_diff(H["Hx"], axis=1) / dy,
    }


class FDTDStepper:
    """Owns the six field-component arrays and advances them one leapfrog
    step at a time. `dt` is always the caller's already-CFL-validated time
    step (stability.py) -- this class never computes or second-guesses it."""

    def __init__(
        self,
        grid: YeeGrid,
        dt: float,
        mu_bg: complex,
        e_coefficients: EFieldCoefficients,
        masks: dict[str, ComponentMask],
    ) -> None:
        self.grid = grid
        self.dt = dt
        self._mu_bg = float(np.real(mu_bg))
        self._h_coeff = dt / self._mu_bg  # Section 4.2: mu_r=1 -> single global H coefficient
        self.e_coefficients = e_coefficients
        self.masks = masks
        self.E: dict[str, Array] = {c: np.zeros(grid.shape, dtype=float) for c in E_COMPONENTS}
        self.H: dict[str, Array] = {c: np.zeros(grid.shape, dtype=float) for c in H_COMPONENTS}
        self.time = 0.0

        for component in (*E_COMPONENTS, *H_COMPONENTS):
            if component not in masks:
                raise ValueError(f"masks missing entry for component {component!r}")

    def step(self, e_source: dict[str, Array] | None = None) -> None:
        """One leapfrog step (Section 5.1). `e_source`, if given, is an
        additive soft-source term (Section 3) added to E *after* the
        natural update -- not a hard-set field value."""
        curl_e = _curl_e_for_h(self.E, self.grid.cell_size)
        for component in H_COMPONENTS:
            updated = self.H[component] - self._h_coeff * curl_e[component]
            updated[~self.masks[component].cavity_interior] = 0.0
            self.H[component] = updated

        curl_h = _curl_h_for_e(self.H, self.grid.cell_size)
        for component in E_COMPONENTS:
            Ca = self.e_coefficients.Ca[component]
            Cb = self.e_coefficients.Cb[component]
            updated = Ca * self.E[component] + Cb * curl_h[component]
            if e_source is not None and component in e_source:
                updated = updated + e_source[component]
            updated[~self.masks[component].cavity_interior] = 0.0
            self.E[component] = updated

        self.time += self.dt

    def probe_index(self, point: Array, component: str) -> tuple[int, int, int]:
        """Nearest grid index of `component`'s own staggered array to
        `point`, clamped to the array's own bounds."""
        ox, oy, oz = COMPONENT_OFFSETS[component]
        dx, dy, dz = self.grid.cell_size
        point = np.asarray(point, dtype=float)
        i = int(round((point[0] - self.grid.origin[0]) / dx - ox))
        j = int(round((point[1] - self.grid.origin[1]) / dy - oy))
        k = int(round((point[2] - self.grid.origin[2]) / dz - oz))
        nx, ny, nz = self.grid.shape
        return (
            int(np.clip(i, 0, nx - 1)),
            int(np.clip(j, 0, ny - 1)),
            int(np.clip(k, 0, nz - 1)),
        )

    def probe_value(self, point: Array, component: str) -> float:
        """Section 3 probing: the value of `component` at the grid point
        nearest `point`."""
        i, j, k = self.probe_index(point, component)
        return float(self.E[component][i, j, k])
