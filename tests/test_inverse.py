"""Validation suite for Module 5 (docs/module5_inverse_equations.md Section 6)."""
import pathlib

import numpy as np
import pytest
from scipy import constants

from cavity_perturbation.cavity import ModeIndex, RectangularCavity
from cavity_perturbation.fields import AnalyticalField, FieldProvider
from cavity_perturbation.inverse import (
    FitResult,
    InverseSolver,
    Measurement,
    point_dipole_filling_factors,
)
from cavity_perturbation.perturbation import PerturbationModel
from cavity_perturbation.sample import Cylinder, Material, Sample, Sphere


class ConstantFieldProvider(FieldProvider):
    """Test double: spatially uniform E, H, with fixed f0/Q_wall/W/epsilon_bg/
    mu_bg -- lets a test control the E:H field ratio (and hence p_E:p_H)
    directly, independent of any real Module 1 cavity mode."""

    def __init__(self, E_vec, H_vec, f0=1e9, Q_wall_value=1e4, W=1e-12,
                 epsilon_bg=constants.epsilon_0, mu_bg=constants.mu_0):
        self._E_vec = np.asarray(E_vec, dtype=complex)
        self._H_vec = np.asarray(H_vec, dtype=complex)
        self._f0 = f0
        self._Q_wall_value = Q_wall_value
        self._W = W
        self._epsilon_bg = epsilon_bg
        self._mu_bg = mu_bg

    def E(self, r):
        r = np.asarray(r, dtype=float)
        if r.ndim == 1:
            return self._E_vec.copy()
        return np.tile(self._E_vec, (r.shape[0], 1))

    def H(self, r):
        r = np.asarray(r, dtype=float)
        if r.ndim == 1:
            return self._H_vec.copy()
        return np.tile(self._H_vec, (r.shape[0], 1))

    def total_stored_energy(self):
        return self._W

    @property
    def f0(self):
        return self._f0

    @property
    def epsilon_bg(self):
        return self._epsilon_bg

    @property
    def mu_bg(self):
        return self._mu_bg

    def Q_wall(self, Rs):
        return self._Q_wall_value


def _small_cavity_model(Rs_walls=None):
    a = b = c = 0.03
    cav = RectangularCavity(a, b, c, ModeIndex("TE", (0, 1, 1)))
    field = AnalyticalField(cav)
    return cav, field, PerturbationModel(field, Rs_walls=Rs_walls)


def _synthetic_measurement(model, region, material, sigma_f=1e-4, sigma_Q=1e-2):
    """Runs the forward model to generate a noiseless (f_meas, Q_meas) for a
    known material -- the standard synthetic-recovery setup (Section 6)."""
    sample = Sample(region=region, material=material)
    r = model.evaluate(sample)
    return Measurement(
        model=model, region=region, f_meas=r.f_calc, Q_meas=r.Q_calc,
        sigma_f=sigma_f, sigma_Q=sigma_Q,
    )


# --- Synthetic-data recovery (the whole-pipeline regression guard) ---------

def test_synthetic_recovery_fit_mu_false():
    _, _, model = _small_cavity_model(Rs_walls=0.02)
    region = Sphere(center=[0.015, 0.015, 0.015], radius=1e-3)
    true_material = Material.from_loss_tangent(4.5, 0.01)
    meas = _synthetic_measurement(model, region, true_material)

    solver = InverseSolver([meas], fit_mu=False)
    result = solver.fit()

    assert result.success
    assert result.eps.real == pytest.approx(true_material.eps.real, rel=1e-3)
    assert result.eps.imag == pytest.approx(true_material.eps.imag, rel=1e-2)
    assert result.mu == pytest.approx(1.0 - 0j)


def test_synthetic_recovery_fit_mu_true_two_measurements():
    """Uses 'generic'-shaped regions (kappa_E=kappa_H=1 identically, no
    depolarization saturation) so the two-measurement system is exactly
    linear in (eps_r, mu_r) and well-conditioned by construction -- a
    Sphere's N=1/3 saturation (kappa_E->3 as eps_r->inf) makes eps' poorly
    constrained on its own at moderate-to-large eps_r even when the two
    measurements' E:H ratios differ, which is a separate, real identifiability
    effect and not what this test targets (see the degenerate-multi-mode test
    below for deliberately-similar-ratio ill-conditioning)."""
    fp1 = ConstantFieldProvider(E_vec=[1.0, 0, 0], H_vec=[0.2, 0, 0])
    fp2 = ConstantFieldProvider(E_vec=[0.2, 0, 0], H_vec=[1.0, 0, 0])
    model1 = PerturbationModel(fp1, Rs_walls=None)
    model2 = PerturbationModel(fp2, Rs_walls=None)

    region1 = Cylinder(center=[0, 0, 0], axis=[0, 0, 1], radius=1e-4, height=1e-4)
    region2 = Cylinder(center=[0, 0, 0], axis=[0, 0, 1], radius=1e-4, height=1e-4)
    assert region1.shape_kind == "generic"
    true_material = Material(eps=6.0 - 0.05j, mu=2.0 - 0.02j)

    meas1 = _synthetic_measurement(model1, region1, true_material)
    meas2 = _synthetic_measurement(model2, region2, true_material)

    solver = InverseSolver([meas1, meas2], fit_mu=True)
    result = solver.fit()

    assert result.success
    assert result.eps.real == pytest.approx(true_material.eps.real, rel=1e-3)
    assert result.eps.imag == pytest.approx(true_material.eps.imag, rel=1e-2)
    assert result.mu.real == pytest.approx(true_material.mu.real, rel=1e-3)
    assert result.mu.imag == pytest.approx(true_material.mu.imag, rel=1e-2)
    # (Not asserting condition_number here: this configuration's seed is
    # exact, so least_squares terminates in a single evaluation at zero
    # residual, and its finite-difference Jacobian there is not a
    # meaningful identifiability signal -- the degenerate-multi-mode test
    # below is the real test of that diagnostic.)


# --- Closed-form seed accuracy (isolates seed bugs from optimizer bugs) ----

def test_closed_form_seed_matches_known_material_in_point_dipole_limit():
    fp = ConstantFieldProvider(E_vec=[1.0, 0, 0], H_vec=[0, 0, 0])
    model = PerturbationModel(fp, Rs_walls=None)
    # Aspect ratio in the 'generic' range -> kappa_E=kappa_H=1 always
    # (Module 3's point-dipole fallback, independent of sample size).
    region = Cylinder(center=[0, 0, 0], axis=[0, 0, 1], radius=1e-4, height=1e-4)
    assert region.shape_kind == "generic"

    true_material = Material(eps=6.0 - 0.3j, mu=1.0 - 0j)
    meas = _synthetic_measurement(model, region, true_material)

    solver = InverseSolver([meas], fit_mu=False)
    seed = solver._initial_guess_vector(None)
    seed_material = solver._unpack(seed)

    # kappa=1 everywhere in this configuration makes the seed formula exact
    # (no small-sample approximation error), so this should match to near
    # machine precision, not just "a few percent."
    assert seed_material.eps.real == pytest.approx(true_material.eps.real, rel=1e-6)
    assert seed_material.eps.imag == pytest.approx(true_material.eps.imag, rel=1e-6)


# --- Degenerate multi-mode / identifiability -------------------------------

def test_degenerate_multi_mode_seed_and_condition_number_flag_ill_conditioning():
    """Two measurements of (effectively) the same mode share nearly the same
    p_E:p_H ratio -- both the Section 2.4 seed matrix and the Section 4.2
    condition number should flag this as ill-conditioned, not just silently
    converge to a poorly-constrained answer."""
    a = b = c = 0.03
    cav = RectangularCavity(a, b, c, ModeIndex("TE", (0, 1, 1)))
    field = AnalyticalField(cav)
    model = PerturbationModel(field, Rs_walls=None)

    center = np.array([a / 2, b / 2, c / 2])
    region1 = Sphere(center=center, radius=1e-3)
    # TE(0,1,1) has no x-dependence, so the offset must be in y/z (the
    # directions this mode actually varies in) to be near- rather than
    # exactly-degenerate.
    region2 = Sphere(center=center + np.array([0.0, 1e-6, 1e-6]), radius=1e-3)

    true_material = Material(eps=6.0 - 0.05j, mu=2.0 - 0.02j)
    meas1 = _synthetic_measurement(model, region1, true_material)
    meas2 = _synthetic_measurement(model, region2, true_material)

    p_E1, p_H1 = point_dipole_filling_factors(model, region1)
    p_E2, p_H2 = point_dipole_filling_factors(model, region2)
    seed_matrix = np.array([[p_E1, p_H1], [p_E2, p_H2]], dtype=complex)
    assert np.linalg.cond(seed_matrix) > 1e6

    solver = InverseSolver([meas1, meas2], fit_mu=True)
    result = solver.fit()
    assert result.condition_number > 1e6


# --- Bounds enforcement -----------------------------------------------------

def test_bounds_enforcement_keeps_fit_within_default_bounds():
    _, _, model = _small_cavity_model(Rs_walls=None)
    region = Sphere(center=[0.015, 0.015, 0.015], radius=1e-3)
    # eps' < 1 violates the default fitting prior but is still passive
    # (eps'>0, eps''>=0) -- PerturbationModel.evaluate accepts it fine.
    true_material = Material(eps=0.5 - 0.01j, mu=1.0 - 0j)
    meas = _synthetic_measurement(model, region, true_material)

    solver = InverseSolver([meas], fit_mu=False)
    result = solver.fit()

    assert result.success
    assert result.eps.real >= 1.0 - 1e-9
    assert result.eps.imag <= 0.0 + 1e-9


def test_constructor_rejects_single_measurement_with_fit_mu_true():
    _, _, model = _small_cavity_model(Rs_walls=None)
    region = Sphere(center=[0.015, 0.015, 0.015], radius=1e-3)
    meas = _synthetic_measurement(model, region, Material.from_loss_tangent(4.5, 0.01))
    with pytest.raises(ValueError):
        InverseSolver([meas], fit_mu=True)


def test_constructor_rejects_empty_measurement_list():
    with pytest.raises(ValueError):
        InverseSolver([], fit_mu=False)


# --- Q_calc = infinity robustness (Section 1.3) -----------------------------

def test_qcalc_infinite_does_not_crash_residuals_or_fit():
    _, _, model = _small_cavity_model(Rs_walls=None)
    region = Sphere(center=[0.015, 0.015, 0.015], radius=1e-3)
    lossy_material = Material.from_loss_tangent(4.5, 0.01)
    meas = _synthetic_measurement(model, region, lossy_material)

    solver = InverseSolver([meas], fit_mu=False)
    p_lossless = np.array([4.5, 0.0])  # eps''=0 -> Q_calc=inf for this trial point
    residuals = solver._residuals(p_lossless)
    assert np.all(np.isfinite(residuals))

    result = solver.fit()
    assert result.success
    assert np.isfinite(result.residual_norm)


# --- Measurement validation --------------------------------------------------

def test_measurement_rejects_nonpositive_f_meas():
    _, _, model = _small_cavity_model(Rs_walls=None)
    region = Sphere(center=[0.015, 0.015, 0.015], radius=1e-3)
    with pytest.raises(ValueError):
        Measurement(model=model, region=region, f_meas=0.0, Q_meas=1000.0)


def test_measurement_rejects_infinite_q_meas():
    _, _, model = _small_cavity_model(Rs_walls=None)
    region = Sphere(center=[0.015, 0.015, 0.015], radius=1e-3)
    with pytest.raises(ValueError):
        Measurement(model=model, region=region, f_meas=1e9, Q_meas=float("inf"))


# --- Rename check (Section 6, sigma_invQ -> sigma_Q) ------------------------

def test_no_lingering_sigma_invq_references():
    stale_name = "sigma" + "_invQ"  # built dynamically so this file's own text doesn't self-match
    root = pathlib.Path(__file__).resolve().parents[1]
    for subdir in ("src", "tests", "scripts"):
        for path in (root / subdir).rglob("*.py"):
            if path == pathlib.Path(__file__).resolve():
                continue
            assert stale_name not in path.read_text(), f"stale {stale_name} reference in {path}"
