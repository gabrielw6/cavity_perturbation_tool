"""FDTD Module -- materials.py: single-frequency-matched conductivity and
per-cell E-update coefficient arrays, per docs/fdtd_module_plan.md Section 4.

mu_r = 1 scope (Section 4.2): only epsilon/conductivity vary by cell: the H
update stays a single global coefficient (assembled in stepper.py directly
from `epsilon_bg`/`mu_bg`, no array needed here).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import constants

from ..sample import Material
from .grid.rasterize import ComponentMask
from .grid.yee import E_COMPONENTS, YeeGrid

Array = np.ndarray


def matched_conductivity(eps_r: complex, f0: float) -> float:
    """Section 4: sigma = omega0*eps0*eps_r'*tan_delta_e, matching the
    sample's own loss tangent at the mode frequency f0. `eps_r` is Module
    3's relative (vacuum-referenced) `Material.eps` -- always multiplied by
    the vacuum `epsilon_0`, never a background permittivity, per this
    project's absolute-vs-relative convention (CLAUDE.md): a *relative*
    permittivity is always referenced to vacuum, regardless of what medium
    actually fills the cavity."""
    eps_r_real = eps_r.real
    if eps_r_real <= 0:
        raise ValueError(f"Re(eps_r)={eps_r_real!r} must be > 0 to define a conductivity match")
    tan_delta_e = -eps_r.imag / eps_r_real
    omega0 = 2.0 * np.pi * f0
    return omega0 * constants.epsilon_0 * eps_r_real * tan_delta_e


@dataclass(frozen=True)
class EFieldCoefficients:
    """Per-E-component update coefficients (Section 5.1 leapfrog E update),
    each an array of shape `grid.shape`. `Ca`/`Cb` reduce to the lossless
    C_a=1, C_b=dt/eps everywhere sigma=0 (background and any lossless
    sample), a single code path rather than a special-cased lossy branch."""

    Ca: dict[str, Array]
    Cb: dict[str, Array]


def assemble_e_coefficients(
    grid: YeeGrid,
    dt: float,
    epsilon_bg: complex,
    masks: dict[str, ComponentMask],
    f0: float,
    sample_material: Material | None = None,
) -> EFieldCoefficients:
    """Builds Ca/Cb for each E component from that component's own
    rasterized `sample_interior` mask (grid/rasterize.py) -- `epsilon_bg`
    is already absolute SI (CavityMode's own convention, no vacuum-eps0
    multiply needed); `sample_material.eps` is relative and is converted
    explicitly via `matched_conductivity`/an eps0 multiply, the same seam
    Module 4's `perturbation.py` crosses (module4 doc Section 0.1)."""
    eps_bg_abs = float(np.real(epsilon_bg))
    if sample_material is not None:
        eps_sample_abs = float(sample_material.eps.real) * constants.epsilon_0
        sigma_sample = matched_conductivity(sample_material.eps, f0)

    Ca: dict[str, Array] = {}
    Cb: dict[str, Array] = {}
    for component in E_COMPONENTS:
        mask = masks[component]
        eps_arr = np.full(grid.shape, eps_bg_abs, dtype=float)
        sigma_arr = np.zeros(grid.shape, dtype=float)
        if sample_material is not None:
            eps_arr[mask.sample_interior] = eps_sample_abs
            sigma_arr[mask.sample_interior] = sigma_sample

        half_loss = sigma_arr * dt / (2.0 * eps_arr)
        Ca[component] = (1.0 - half_loss) / (1.0 + half_loss)
        Cb[component] = (dt / eps_arr) / (1.0 + half_loss)

    return EFieldCoefficients(Ca=Ca, Cb=Cb)
