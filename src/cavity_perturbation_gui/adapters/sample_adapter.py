"""adapters/sample_adapter.py -- sidebar params -> Sample / Material /
SampleRegion (docs/gui_module_plan.md Section 3). No Qt import (Section 1.4).

Position/orientation resolution mirrors scripts/simulate_perturbation.py's
own logic (same project convention: default sample position is the field
maximum inside the cavity's valid domain, not the bounding-box center,
which for e.g. CoaxialCavity sits on the excluded inner conductor) --
reimplemented against a plain `SampleParams` dataclass instead of
`argparse.Namespace` so this package has no dependency on the script.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from cavity_perturbation.cavity import CavityMode
from cavity_perturbation.fields import FieldProvider
from cavity_perturbation.sample import (
    Cylinder,
    Material,
    Sample,
    SampleRegion,
    Slab,
    Sphere,
    orthonormal_frame,
    real_field_direction,
)

Array = np.ndarray
SampleShape = Literal["sphere", "rod", "disk"]
Orientation = Literal["aligned", "perpendicular"]
FieldName = Literal["E", "H"]

_POSITION_SEARCH_N = 25


@dataclass(frozen=True)
class SampleParams:
    """Plain, Qt-free description of a sample -- the sidebar's sample panel
    (Section 5). `position=None`/`axis=None` fall back to the same
    field-extremum search `simulate_perturbation.py` uses."""

    shape: SampleShape
    position: tuple[float, float, float] | None = None
    radius: float | None = None  # sphere / rod
    length: float | None = None  # rod (None -> 16x radius)
    extent: tuple[float, float] | None = None  # disk
    thickness: float | None = None  # disk (None -> 0.05x min(extent))
    axis: tuple[float, float, float] | None = None  # explicit rod axis / disk normal
    orientation: Orientation = "aligned"
    orient_field: FieldName = "E"
    eps_r: float = 2.5
    tan_delta_e: float = 1e-3
    mu_r: float = 1.0
    tan_delta_m: float = 0.0


def _margin(params: SampleParams) -> float:
    if params.shape in ("sphere", "rod"):
        return params.radius or 0.0
    if params.shape == "disk":
        return 0.5 * max(params.extent) if params.extent else 0.0
    return 0.0


def resolve_position(cav: CavityMode, field: FieldProvider, params: SampleParams) -> Array:
    """Default sample position: the E-field maximum within the cavity's
    actual valid domain, excluding a `margin`-wide band near the boundary
    (a mode flat along one axis would otherwise tie across it and an
    argmax would pick a boundary point) -- see
    scripts/simulate_perturbation.py's `resolve_position` for the full
    rationale, reproduced here without an `argparse.Namespace` dependency.
    """
    if params.position is not None:
        return np.array(params.position, dtype=float)

    margin = _margin(params)
    rmin, rmax = cav.bounding_box()
    axes = [np.linspace(lo, hi, _POSITION_SEARCH_N) for lo, hi in zip(rmin, rmax)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    candidates = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    safe = cav.contains(candidates)
    if margin > 0:
        for axis in range(3):
            offset = np.zeros(3)
            offset[axis] = margin
            safe = safe & cav.contains(candidates + offset) & cav.contains(candidates - offset)
    eligible = candidates[safe] if np.any(safe) else candidates[cav.contains(candidates)]
    if eligible.shape[0] == 0:
        raise ValueError(
            "could not find any point inside the cavity to place the sample -- "
            "specify an explicit sample position"
        )
    with np.errstate(all="ignore"):
        e_mag2 = np.sum(np.abs(field.E(eligible)) ** 2, axis=-1)
    e_mag2 = np.where(np.isfinite(e_mag2), e_mag2, 0.0)
    return eligible[np.argmax(e_mag2)]


def resolve_axis(field: FieldProvider, position: Array, params: SampleParams) -> Array:
    """Cylinder axis / Slab normal, per `params.orientation` -- 'aligned'
    means parallel to the reference field at the sample's center, matching
    Module 3's N=0 rod-axial / N=1 disk-normal canonical case;
    'perpendicular' the N=1/2 rod-transverse / N=0 disk-tangential case."""
    if params.axis is not None:
        v = np.array(params.axis, dtype=float)
        return v / np.linalg.norm(v)

    ref_field = field.E(position) if params.orient_field == "E" else field.H(position)
    ref_dir = real_field_direction(ref_field)
    if ref_dir is None:
        raise ValueError(
            f"the {params.orient_field} field is (numerically) zero at the sample position "
            f"{position.tolist()} -- pick a different position, orient relative to the other "
            "field, or give an explicit axis"
        )
    if params.orientation == "aligned":
        return ref_dir
    e1, _e2, _n = orthonormal_frame(ref_dir)
    return e1


def build_sample_region(cav: CavityMode, field: FieldProvider, params: SampleParams) -> SampleRegion:
    position = resolve_position(cav, field, params)

    if params.shape == "sphere":
        if params.radius is None:
            raise ValueError("radius is required for shape='sphere'")
        return Sphere(center=position, radius=params.radius)

    axis = resolve_axis(field, position, params)

    if params.shape == "rod":
        if params.radius is None:
            raise ValueError("radius is required for shape='rod'")
        length = params.length if params.length is not None else 16.0 * params.radius
        return Cylinder(center=position, axis=axis, radius=params.radius, height=length)

    if params.shape == "disk":
        if params.extent is None:
            raise ValueError("extent is required for shape='disk'")
        thickness = params.thickness if params.thickness is not None else 0.05 * min(params.extent)
        return Slab(center=position, normal=axis, thickness=thickness, extent=params.extent)

    raise ValueError(f"unknown shape {params.shape!r}")


def build_material(params: SampleParams) -> Material:
    return Material.from_loss_tangent(params.eps_r, params.tan_delta_e, params.mu_r, params.tan_delta_m)


def build_sample(cav: CavityMode, field: FieldProvider, params: SampleParams) -> Sample:
    region = build_sample_region(cav, field, params)
    material = build_material(params)
    return Sample(region=region, material=material)
