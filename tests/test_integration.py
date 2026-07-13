"""Whole-pipeline (Module 1->5) synthetic-data recovery test.

Per CLAUDE.md's Testing philosophy: "the regression guard for the whole
pipeline now that Modules 1-5 are all spec-complete" -- fit known eps, mu
from model-generated f_meas, Q_meas, driving every module for real (no test
doubles): CavityMode -> AnalyticalField -> Sample/Material -> PerturbationModel
-> InverseSolver. Covers the architecture doc's "Cross-module integration
checklist" last bullet: (a) lossless-ish dielectric at E-max, (b) lossy
dielectric at E-max, (c) two stacked measurements with fit_mu=True.
"""
import numpy as np
import pytest

from cavity_perturbation.cavity import ModeIndex, RectangularCavity
from cavity_perturbation.fields import AnalyticalField
from cavity_perturbation.inverse import InverseSolver, Measurement
from cavity_perturbation.perturbation import PerturbationModel
from cavity_perturbation.sample import Material, Sample, Sphere


def _synthetic_measurement(model, region, material, sigma_f=1e-4, sigma_Q=1e-2):
    sample = Sample(region=region, material=material)
    r = model.evaluate(sample)
    return Measurement(
        model=model, region=region, f_meas=r.f_calc, Q_meas=r.Q_calc,
        sigma_f=sigma_f, sigma_Q=sigma_Q,
    )


def test_recovery_lossless_dielectric_at_e_max():
    a = b = c = 0.03
    cav = RectangularCavity(a, b, c, ModeIndex("TE", (0, 1, 1)))
    field = AnalyticalField(cav)
    model = PerturbationModel(field, Rs_walls=0.02)  # finite wall loss so Q_meas is finite

    region = Sphere(center=[a / 2, b / 2, c / 2], radius=1e-3)  # TE011 E-max is at the center
    true_material = Material.from_loss_tangent(4.5, 0.0)
    meas = _synthetic_measurement(model, region, true_material)

    result = InverseSolver([meas], fit_mu=False).fit()

    assert result.success
    assert result.eps.real == pytest.approx(true_material.eps.real, rel=1e-3)
    assert result.eps.imag == pytest.approx(0.0, abs=1e-6)
    assert result.mu == pytest.approx(1.0 - 0j)


def test_recovery_lossy_dielectric_at_e_max():
    a = b = c = 0.03
    cav = RectangularCavity(a, b, c, ModeIndex("TE", (0, 1, 1)))
    field = AnalyticalField(cav)
    model = PerturbationModel(field, Rs_walls=0.02)

    region = Sphere(center=[a / 2, b / 2, c / 2], radius=1e-3)
    true_material = Material.from_loss_tangent(4.5, 0.01)
    meas = _synthetic_measurement(model, region, true_material)

    result = InverseSolver([meas], fit_mu=False).fit()

    assert result.success
    assert result.eps.real == pytest.approx(true_material.eps.real, rel=1e-3)
    assert result.eps.imag == pytest.approx(true_material.eps.imag, rel=1e-2)
    assert result.mu == pytest.approx(1.0 - 0j)


def test_recovery_fit_mu_true_two_stacked_measurements():
    """Same mode, two sample placements with different E:H character
    (TE011's field ratio varies with position off-axis) -- the realistic
    version of "two measurements at different field ratios" the
    architecture doc's identifiability discussion calls for."""
    a, b, c = 0.03, 0.025, 0.04
    cav = RectangularCavity(a, b, c, ModeIndex("TE", (0, 1, 1)))
    field = AnalyticalField(cav)
    model = PerturbationModel(field, Rs_walls=None)

    region1 = Sphere(center=[a / 2, b / 2, c / 2], radius=1e-3)
    region2 = Sphere(center=[a / 2, b / 4, c / 4], radius=1e-3)
    true_material = Material(eps=4.5 * (1.0 - 0.01j), mu=1.2 - 0.005j)

    meas1 = _synthetic_measurement(model, region1, true_material)
    meas2 = _synthetic_measurement(model, region2, true_material)

    result = InverseSolver([meas1, meas2], fit_mu=True).fit()

    assert result.success
    assert result.eps.real == pytest.approx(true_material.eps.real, rel=1e-3)
    assert result.eps.imag == pytest.approx(true_material.eps.imag, rel=1e-2)
    assert result.mu.real == pytest.approx(true_material.mu.real, rel=1e-3)
    assert result.mu.imag == pytest.approx(true_material.mu.imag, rel=1e-2)
    # (Not asserting condition_number here: the tight sigma_f (1e-4) vs.
    # loose sigma_Q (1e-2) residual weighting alone inflates it past 1e6 in
    # this configuration even though recovery is accurate to ~1e-11 --
    # tests/test_inverse.py's degenerate-multi-mode test isolates the
    # diagnostic itself from that weighting-scale effect.)
