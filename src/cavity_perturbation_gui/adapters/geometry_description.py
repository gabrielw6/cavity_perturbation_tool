"""adapters/geometry_description.py -- CavityMode + SampleRegion -> plain
geometric primitives for the 3D view (docs/gui_module_plan.md Section 4).
`geometry_view3d.py` only ever draws these dataclasses, never a
`cavity_perturbation` type directly."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from cavity_perturbation.cavity import (
    CavityMode,
    CoaxialCavity,
    CylindricalCavity,
    RectangularCavity,
    ToroidalCavity,
)
from cavity_perturbation.sample import Cylinder, SampleRegion, Slab, Sphere

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class BoxPrimitive:
    corner_min: Vec3
    corner_max: Vec3
    kind: Literal["box"] = "box"


@dataclass(frozen=True)
class CylinderPrimitive:
    center: Vec3
    axis: Vec3
    radius: float
    length: float
    kind: Literal["cylinder"] = "cylinder"


@dataclass(frozen=True)
class AnnulusPrimitive:
    center: Vec3
    axis: Vec3
    inner_radius: float
    outer_radius: float
    length: float
    kind: Literal["annulus"] = "annulus"


@dataclass(frozen=True)
class SpherePrimitive:
    center: Vec3
    radius: float
    kind: Literal["sphere"] = "sphere"


@dataclass(frozen=True)
class SlabPrimitive:
    center: Vec3
    normal: Vec3
    thickness: float
    extent: tuple[float, float]
    kind: Literal["slab"] = "slab"


@dataclass(frozen=True)
class TorusPrimitive:
    """`ToroidalCavity`'s ring always lies in the x-y plane by construction
    (`theta = atan2(y, x)`, docs/toroidal_cavity_plan.md Section 1) -- unlike
    the other cavity primitives, no `axis`/orientation field, since there's
    nothing to configure."""

    center: Vec3
    major_radius: float
    minor_radius: float
    kind: Literal["torus"] = "torus"


CavityPrimitive = BoxPrimitive | CylinderPrimitive | AnnulusPrimitive | TorusPrimitive
SamplePrimitive = SpherePrimitive | CylinderPrimitive | SlabPrimitive


def describe_cavity(cavity: CavityMode) -> CavityPrimitive:
    if isinstance(cavity, RectangularCavity):
        return BoxPrimitive(corner_min=(0.0, 0.0, 0.0), corner_max=(cavity.a, cavity.b, cavity.c))
    if isinstance(cavity, CylindricalCavity):
        return CylinderPrimitive(
            center=(0.0, 0.0, cavity.d / 2.0), axis=(0.0, 0.0, 1.0), radius=cavity.a, length=cavity.d
        )
    if isinstance(cavity, CoaxialCavity):
        return AnnulusPrimitive(
            center=(0.0, 0.0, cavity.L / 2.0),
            axis=(0.0, 0.0, 1.0),
            inner_radius=cavity.a,
            outer_radius=cavity.b,
            length=cavity.L,
        )
    if isinstance(cavity, ToroidalCavity):
        return TorusPrimitive(center=(0.0, 0.0, 0.0), major_radius=cavity.R, minor_radius=cavity.a)
    raise ValueError(f"unknown CavityMode subtype {type(cavity)!r}")


def _vec3(arr: NDArray[np.float64]) -> Vec3:
    values = np.asarray(arr, dtype=float)
    return float(values[0]), float(values[1]), float(values[2])


def describe_sample(region: SampleRegion) -> SamplePrimitive:
    if isinstance(region, Sphere):
        return SpherePrimitive(center=_vec3(region.center), radius=region.radius)
    if isinstance(region, Cylinder):
        return CylinderPrimitive(
            center=_vec3(region.center), axis=_vec3(region.axis), radius=region.radius, length=region.height
        )
    if isinstance(region, Slab):
        return SlabPrimitive(
            center=_vec3(region.center), normal=_vec3(region.normal), thickness=region.thickness, extent=region.extent
        )
    raise ValueError(f"unknown SampleRegion subtype {type(region)!r}")
