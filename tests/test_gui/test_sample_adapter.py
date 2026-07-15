"""docs/gui_module_plan.md Section 8 -- adapters/sample_adapter.py."""
import numpy as np
import pytest

from cavity_perturbation.cavity import ModeIndex, RectangularCavity
from cavity_perturbation.fields import AnalyticalField
from cavity_perturbation.sample import Cylinder, Slab, Sphere
from cavity_perturbation_gui.adapters.sample_adapter import (
    SampleParams,
    build_material,
    build_sample,
    build_sample_region,
    resolve_position,
)

_CAV = RectangularCavity(0.03, 0.02, 0.025, ModeIndex("TE", (0, 1, 1)))
_FIELD = AnalyticalField(_CAV)


def test_resolve_position_explicit_wins():
    params = SampleParams(shape="sphere", position=(0.01, 0.01, 0.01), radius=1e-3)
    pos = resolve_position(_CAV, _FIELD, params)
    assert np.allclose(pos, [0.01, 0.01, 0.01])


def test_resolve_position_default_is_inside_cavity_field_maximum():
    params = SampleParams(shape="sphere", radius=1e-3)
    pos = resolve_position(_CAV, _FIELD, params)
    assert _CAV.contains(pos)[0]


def test_build_sample_region_sphere():
    params = SampleParams(shape="sphere", position=(0.015, 0.01, 0.0125), radius=1.5e-3)
    region = build_sample_region(_CAV, _FIELD, params)
    assert isinstance(region, Sphere)
    assert region.radius == 1.5e-3


def test_build_sample_region_sphere_requires_radius():
    params = SampleParams(shape="sphere", position=(0.015, 0.01, 0.0125))
    with pytest.raises(ValueError):
        build_sample_region(_CAV, _FIELD, params)


def test_build_sample_region_rod_default_length():
    params = SampleParams(shape="rod", position=(0.015, 0.01, 0.0125), radius=2e-4)
    region = build_sample_region(_CAV, _FIELD, params)
    assert isinstance(region, Cylinder)
    assert region.height == pytest.approx(16.0 * 2e-4)


def test_build_sample_region_disk_default_thickness():
    params = SampleParams(shape="disk", position=(0.015, 0.01, 0.0125), extent=(2e-3, 2e-3))
    region = build_sample_region(_CAV, _FIELD, params)
    assert isinstance(region, Slab)
    assert region.thickness == pytest.approx(0.05 * 2e-3)


def test_build_sample_region_unknown_shape_raises():
    params = SampleParams(shape="triangle", position=(0.015, 0.01, 0.0125))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_sample_region(_CAV, _FIELD, params)


def test_build_material_from_loss_tangent():
    params = SampleParams(shape="sphere", radius=1e-3, eps_r=4.5, tan_delta_e=0.01, mu_r=1.0)
    material = build_material(params)
    assert material.eps == pytest.approx(4.5 * (1 - 0.01j))
    assert material.mu == pytest.approx(1.0 + 0.0j)


def test_build_sample_combines_region_and_material():
    params = SampleParams(shape="sphere", position=(0.015, 0.01, 0.0125), radius=1.5e-3, eps_r=3.0)
    sample = build_sample(_CAV, _FIELD, params)
    assert isinstance(sample.region, Sphere)
    assert sample.material.eps.real == pytest.approx(3.0)
