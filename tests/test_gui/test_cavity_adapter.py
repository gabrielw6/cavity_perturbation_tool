"""docs/gui_module_plan.md Section 8 -- adapters/, plain pytest, real
cavity_perturbation calls, no Qt."""
import pytest
from scipy import constants

from cavity_perturbation.cavity import CoaxialCavity, CylindricalCavity, RectangularCavity
from cavity_perturbation_gui.adapters.cavity_adapter import (
    CavityParams,
    build_cavity,
    cavity_constructor_and_args,
    resolve_rs,
)


def test_build_cavity_rectangular():
    params = CavityParams("rectangular", (0.03, 0.02, 0.025), "TE", (0, 1, 1))
    cav = build_cavity(params)
    assert isinstance(cav, RectangularCavity)
    assert (cav.a, cav.b, cav.c) == (0.03, 0.02, 0.025)


def test_build_cavity_cylindrical():
    params = CavityParams("cylindrical", (0.02, 0.03), "TM", (0, 1, 0))
    cav = build_cavity(params)
    assert isinstance(cav, CylindricalCavity)
    assert (cav.a, cav.d) == (0.02, 0.03)


def test_build_cavity_coaxial():
    params = CavityParams("coaxial", (0.01, 0.023, 0.5), "TEM", (1,))
    cav = build_cavity(params)
    assert isinstance(cav, CoaxialCavity)
    assert (cav.a, cav.b, cav.L) == (0.01, 0.023, 0.5)


def test_build_cavity_applies_background_permittivity():
    params = CavityParams("rectangular", (0.03, 0.02, 0.025), "TE", (0, 1, 1), bg_eps_r=2.0, bg_mu_r=1.0)
    cav = build_cavity(params)
    assert cav.epsilon_bg == pytest.approx(2.0 * constants.epsilon_0)


def test_build_cavity_rejects_unknown_type():
    params = CavityParams("nonagonal", (1.0,), "TE", (0, 1, 1))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_cavity(params)


def test_cavity_constructor_and_args_round_trips_through_build_cavity():
    params = CavityParams("cylindrical", (0.02, 0.03), "TM", (0, 1, 0))
    ctor, args, mode = cavity_constructor_and_args(params)
    cav_direct = ctor(*args, mode)
    cav_adapter = build_cavity(params)
    assert cav_direct.f0 == pytest.approx(cav_adapter.f0)


def test_resolve_rs_explicit_value_wins():
    cav = build_cavity(CavityParams("rectangular", (0.03, 0.02, 0.025), "TE", (0, 1, 1)))
    assert resolve_rs(cav, rs=0.05, conductivity=5.8e7) == 0.05


def test_resolve_rs_derives_from_conductivity():
    cav = build_cavity(CavityParams("rectangular", (0.03, 0.02, 0.025), "TE", (0, 1, 1)))
    rs = resolve_rs(cav, rs=None, conductivity=5.8e7)
    assert rs is not None and rs > 0.0


def test_resolve_rs_none_means_no_wall_loss():
    cav = build_cavity(CavityParams("rectangular", (0.03, 0.02, 0.025), "TE", (0, 1, 1)))
    assert resolve_rs(cav, rs=None, conductivity=None) is None
