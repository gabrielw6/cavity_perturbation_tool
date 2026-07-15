"""adapters/cavity_adapter.py -- sidebar params -> CavityMode
(docs/gui_module_plan.md Section 3). No Qt import (Section 1.4)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from scipy import constants

from cavity_perturbation.cavity import (
    CavityMode,
    CoaxialCavity,
    CylindricalCavity,
    ModeIndex,
    RectangularCavity,
)

CavityType = Literal["rectangular", "cylindrical", "coaxial"]


@dataclass(frozen=True)
class CavityParams:
    """Plain, Qt-free description of a cavity -- what the sidebar widget
    collects and every runner consumes (Section 4). `dimensions` meaning
    depends on `cavity_type`: (a, b, c) / (radius, length) /
    (r_inner, r_outer, length)."""

    cavity_type: CavityType
    dimensions: tuple[float, ...]
    mode_kind: str
    mode_indices: tuple[int, ...]
    bg_eps_r: float = 1.0
    bg_mu_r: float = 1.0


def build_cavity(params: CavityParams) -> CavityMode:
    """Section 4: the one place sidebar parameters turn into a `CavityMode`
    -- no widget constructs one directly."""
    eps = params.bg_eps_r * constants.epsilon_0
    mu = params.bg_mu_r * constants.mu_0
    mode = ModeIndex(params.mode_kind, params.mode_indices)

    if params.cavity_type == "rectangular":
        a, b, c = params.dimensions
        return RectangularCavity(a, b, c, mode, eps=eps, mu=mu)
    if params.cavity_type == "cylindrical":
        radius, length = params.dimensions
        return CylindricalCavity(radius, length, mode, eps=eps, mu=mu)
    if params.cavity_type == "coaxial":
        r_inner, r_outer, length = params.dimensions
        return CoaxialCavity(r_inner, r_outer, length, mode, eps=eps, mu=mu)
    raise ValueError(f"unknown cavity_type {params.cavity_type!r}")


def cavity_constructor_and_args(
    params: CavityParams,
) -> tuple[Callable[..., CavityMode], tuple[float, ...], ModeIndex]:
    """(constructor, positional geometry args, mode) -- for `ritz_runner.py`'s
    `nearest_basis_modes` call, which needs to build several instances of
    the same geometry at different mode indices (same shape as
    scripts/simulate_perturbation.py's own `cavity_type_and_args`)."""
    mode = ModeIndex(params.mode_kind, params.mode_indices)
    if params.cavity_type == "rectangular":
        return RectangularCavity, params.dimensions, mode
    if params.cavity_type == "cylindrical":
        return CylindricalCavity, params.dimensions, mode
    if params.cavity_type == "coaxial":
        return CoaxialCavity, params.dimensions, mode
    raise ValueError(f"unknown cavity_type {params.cavity_type!r}")


def resolve_rs(cav: CavityMode, *, rs: float | None, conductivity: float | None) -> float | None:
    """Same precedence as scripts/simulate_perturbation.py's own
    resolve_rs: an explicit surface resistance wins, otherwise derive one
    from wall conductivity (skin-effect formula), otherwise None (no wall
    loss -- the sidebar's own "Rs source" choice, Section 5)."""
    if rs is not None:
        return rs
    if conductivity is not None:
        return float((constants.pi * cav.f0 * constants.mu_0 / conductivity) ** 0.5)
    return None
