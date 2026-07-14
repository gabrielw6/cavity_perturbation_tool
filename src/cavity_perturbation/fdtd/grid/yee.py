"""FDTD Module -- grid/yee.py: Yee staggered-grid coordinate maps.

Pure staggering arithmetic -- no Maxwell's equations, no time-stepping, no
material/geometry knowledge. See docs/fdtd_module_plan.md Section 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Array = np.ndarray

# Section 1 table: cell-corner (i,j,k) plus this half-cell offset (in
# cell-size units, not meters) gives each component's physical location
# within cell (i,j,k).
COMPONENT_OFFSETS: dict[str, tuple[float, float, float]] = {
    "Ex": (0.5, 0.0, 0.0),
    "Ey": (0.0, 0.5, 0.0),
    "Ez": (0.0, 0.0, 0.5),
    "Hx": (0.0, 0.5, 0.5),
    "Hy": (0.5, 0.0, 0.5),
    "Hz": (0.5, 0.5, 0.0),
}

E_COMPONENTS: tuple[str, str, str] = ("Ex", "Ey", "Ez")
H_COMPONENTS: tuple[str, str, str] = ("Hx", "Hy", "Hz")


@dataclass(frozen=True)
class YeeGrid:
    """A regular Cartesian Yee grid of `shape` cells (nx, ny, nz), each cell
    sized `cell_size` (dx, dy, dz) meters, with cell-corner (i,j,k)=(0,0,0)
    located at `origin`.

    Every field component (Ex..Hz) is stored on its own (nx, ny, nz)-shaped
    array -- one uniform shape for every component, boundary cells included.
    PEC walls (Section 5.3) are enforced via a per-component interior mask
    (grid/rasterize.py), not by shrinking the component arrays near
    boundaries.
    """

    shape: tuple[int, int, int]
    cell_size: tuple[float, float, float]
    origin: Array = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        if any(n < 1 for n in self.shape):
            raise ValueError(f"grid shape must be >= 1 per axis, got {self.shape}")
        if any(d <= 0 for d in self.cell_size):
            raise ValueError(f"cell_size must be > 0 per axis, got {self.cell_size}")
        object.__setattr__(self, "origin", np.asarray(self.origin, dtype=float))

    def component_coords(self, component: str) -> Array:
        """Physical (x, y, z) coordinates of every grid point of
        `component`, shape (nx*ny*nz, 3), flattened in 'ij' (C) order --
        i.e. `coords.reshape(*self.shape, 3)` recovers the per-cell layout
        used everywhere else in this sub-package."""
        if component not in COMPONENT_OFFSETS:
            raise ValueError(
                f"unknown component {component!r}, expected one of {list(COMPONENT_OFFSETS)}"
            )
        nx, ny, nz = self.shape
        dx, dy, dz = self.cell_size
        ox, oy, oz = COMPONENT_OFFSETS[component]
        I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        x = self.origin[0] + (I + ox) * dx
        y = self.origin[1] + (J + oy) * dy
        z = self.origin[2] + (K + oz) * dz
        return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)

    @property
    def cell_volume(self) -> float:
        dx, dy, dz = self.cell_size
        return dx * dy * dz
