"""docs/gui_module_plan.md Section 8 -- adapters/geometry_description.py."""
import pytest

from cavity_perturbation.cavity import CoaxialCavity, CylindricalCavity, ModeIndex, RectangularCavity
from cavity_perturbation.sample import Cylinder, Slab, Sphere
from cavity_perturbation_gui.adapters.geometry_description import (
    AnnulusPrimitive,
    BoxPrimitive,
    CylinderPrimitive,
    SlabPrimitive,
    SpherePrimitive,
    describe_cavity,
    describe_sample,
)


def test_describe_cavity_rectangular():
    cav = RectangularCavity(0.03, 0.02, 0.025, ModeIndex("TE", (0, 1, 1)))
    prim = describe_cavity(cav)
    assert isinstance(prim, BoxPrimitive)
    assert prim.corner_min == (0.0, 0.0, 0.0)
    assert prim.corner_max == (0.03, 0.02, 0.025)


def test_describe_cavity_cylindrical():
    cav = CylindricalCavity(0.02, 0.03, ModeIndex("TM", (0, 1, 0)))
    prim = describe_cavity(cav)
    assert isinstance(prim, CylinderPrimitive)
    assert prim.radius == 0.02
    assert prim.length == 0.03


def test_describe_cavity_coaxial():
    cav = CoaxialCavity(0.01, 0.023, 0.5)
    prim = describe_cavity(cav)
    assert isinstance(prim, AnnulusPrimitive)
    assert prim.inner_radius == 0.01
    assert prim.outer_radius == 0.023
    assert prim.length == 0.5


def test_describe_sample_sphere():
    region = Sphere(center=[0.01, 0.02, 0.03], radius=1e-3)
    prim = describe_sample(region)
    assert isinstance(prim, SpherePrimitive)
    assert prim.center == pytest.approx((0.01, 0.02, 0.03))
    assert prim.radius == 1e-3


def test_describe_sample_cylinder():
    region = Cylinder(center=[0.0, 0.0, 0.0], axis=[0.0, 0.0, 1.0], radius=2e-4, height=3e-3)
    prim = describe_sample(region)
    assert isinstance(prim, CylinderPrimitive)
    assert prim.radius == 2e-4
    assert prim.length == 3e-3


def test_describe_sample_slab():
    region = Slab(center=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0], thickness=1e-4, extent=(2e-3, 2e-3))
    prim = describe_sample(region)
    assert isinstance(prim, SlabPrimitive)
    assert prim.thickness == 1e-4
    assert prim.extent == (2e-3, 2e-3)
