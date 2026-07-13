"""Validation suite for Module 4 (docs/module4_perturbation_equations.md Section 5)."""
import gc
import weakref

import numpy as np
import pytest
from scipy import constants

from cavity_perturbation.cavity import ModeIndex, RectangularCavity
from cavity_perturbation.fields import AnalyticalField, FieldProvider
from cavity_perturbation.perturbation import PerturbationModel, PerturbationResult
from cavity_perturbation.sample import Material, Sample, Sphere

from .conftest import assert_passive_material_never_improves_q


class ConstantFieldProvider(FieldProvider):
    """Test double: spatially uniform E, H, with fixed f0/Q_wall/W/epsilon_bg/
    mu_bg. Decouples Module 4's own formula from any real Module 1 cavity
    physics -- used specifically for the background-medium-sensitivity check,
    where epsilon_bg must be varied in isolation from everything else."""

    def __init__(self, E_vec, H_vec, f0, Q_wall_value, W, epsilon_bg, mu_bg):
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


def _delta_from_result(result: PerturbationResult, f0: float) -> complex:
    omega0 = 2.0 * np.pi * f0
    return result.omega_tilde / omega0 - 1.0


def _small_cavity_field():
    a, b, c = 0.03, 0.03, 0.03
    cav = RectangularCavity(a, b, c, ModeIndex("TE", (0, 1, 1)))
    return cav, AnalyticalField(cav)


# --- Small-sample limit (point-dipole convergence) -------------------------

def test_small_sample_limit_converges_to_point_dipole_formula():
    a = b = c = 0.03
    cav, field = _small_cavity_field()
    model = PerturbationModel(field)  # Rs_walls=None -> sample-only

    center = np.array([a / 2, b / 2, c / 2])
    eps_r = 4.5
    material = Material(eps=eps_r - 0j, mu=1.0 - 0j)
    E0_center = field.E(center)
    E0_mag2 = float(np.sum(np.abs(E0_center) ** 2))
    W = field.total_stored_energy()
    N_sphere = 1.0 / 3.0
    kappa_E_expected = 1.0 / (1.0 + N_sphere * (eps_r - 1.0))

    errors = []
    for radius in (2e-3, 1e-3, 5e-4, 2.5e-4):
        region = Sphere(center=center, radius=radius)
        sample = Sample(region=region, material=material)
        result = model.evaluate(sample)
        delta = _delta_from_result(result, cav.f0)
        p_E = -2.0 * delta / np.conj(eps_r - 1.0)  # eps_r is real here, so conj is a no-op

        p_E_expected = cav.epsilon_bg * kappa_E_expected * E0_mag2 * region.volume() / W
        errors.append(abs(p_E - p_E_expected) / abs(p_E_expected))

    assert errors[-1] < errors[0]
    assert errors[-1] < 1e-2


# --- Passivity => Q can only degrade ---------------------------------------

def test_passive_material_never_improves_q():
    a = b = c = 0.03
    cav, field = _small_cavity_field()
    Rs = 0.02
    model = PerturbationModel(field, Rs_walls=Rs)
    region = Sphere(center=[a / 2, b / 2, c / 2], radius=1e-3)
    assert_passive_material_never_improves_q(model, region, eps_r=4.5, Q_wall=cav.Q_wall(Rs))


# --- Reciprocal-Q additivity -------------------------------------------------

def test_reciprocal_q_additivity():
    """1/Q_wall + 1/Q_sample_only == 1/Q_combined holds only to first order in
    the sample perturbation (both are independent *small* perturbations to
    the same base state, per Section 2.1) -- the omitted higher-order cross
    term is O(Re(delta)), so this needs a small-enough sample, not an exact
    tolerance at any sample size (verified: relative error shrinks with
    sample volume, same limiting behavior as the small-sample-limit test)."""
    a = b = c = 0.03
    cav, field = _small_cavity_field()
    Rs = 0.02
    model_combined = PerturbationModel(field, Rs_walls=Rs)
    model_sample_only = PerturbationModel(field, Rs_walls=None)

    region = Sphere(center=[a / 2, b / 2, c / 2], radius=1e-5)
    material = Material.from_loss_tangent(4.5, 0.01)
    sample = Sample(region=region, material=material)

    combined = model_combined.evaluate(sample)
    sample_only = model_sample_only.evaluate(sample)
    Q_wall = cav.Q_wall(Rs)

    expected_inv_Q = 1.0 / Q_wall + 1.0 / sample_only.Q_calc
    assert 1.0 / combined.Q_calc == pytest.approx(expected_inv_Q, rel=1e-6)


# --- Background-medium sensitivity (catches the Section 0.1 bug class) ----

def test_background_medium_sensitivity():
    E_vec = [1.0 + 0j, 0, 0]
    H_vec = [0, 1.0 + 0j, 0]
    W = 1e-12
    fp1 = ConstantFieldProvider(
        E_vec, H_vec, f0=1e9, Q_wall_value=1e4, W=W,
        epsilon_bg=constants.epsilon_0, mu_bg=constants.mu_0,
    )
    fp2 = ConstantFieldProvider(
        E_vec, H_vec, f0=1e9, Q_wall_value=1e4, W=W,
        epsilon_bg=2.0 * constants.epsilon_0, mu_bg=constants.mu_0,
    )

    region = Sphere(center=[0.0, 0.0, 0.0], radius=1e-4)
    material = Material.from_loss_tangent(4.5, 0.0)  # mu_r=1 isolates the E-term
    sample = Sample(region=region, material=material)

    r1 = PerturbationModel(fp1).evaluate(sample)
    r2 = PerturbationModel(fp2).evaluate(sample)

    delta1 = _delta_from_result(r1, fp1.f0)
    delta2 = _delta_from_result(r2, fp2.f0)
    assert delta2 == pytest.approx(2.0 * delta1, rel=1e-9)


# --- Scale invariance --------------------------------------------------------

def test_scale_invariance():
    a = b = c = 0.03
    mode = ModeIndex("TE", (0, 1, 1))
    cav1 = RectangularCavity(a, b, c, mode, amplitude=1.0)
    cav2 = RectangularCavity(a, b, c, mode, amplitude=3.0 - 2.0j)
    Rs = 0.02
    model1 = PerturbationModel(AnalyticalField(cav1), Rs_walls=Rs)
    model2 = PerturbationModel(AnalyticalField(cav2), Rs_walls=Rs)

    region = Sphere(center=[a / 2, b / 2, c / 2], radius=1e-3)
    material = Material.from_loss_tangent(4.5, 0.01)
    sample = Sample(region=region, material=material)

    r1 = model1.evaluate(sample)
    r2 = model2.evaluate(sample)
    assert r1.f_calc == pytest.approx(r2.f_calc, rel=1e-9)
    assert r1.Q_calc == pytest.approx(r2.Q_calc, rel=1e-9)


# --- Cache correctness (Section 0.3 fix) ------------------------------------

def test_cache_holds_strong_reference_preventing_id_reuse():
    a = b = c = 0.03
    _, field = _small_cavity_field()
    model = PerturbationModel(field)

    region = Sphere(center=[a / 2, b / 2, c / 2], radius=1e-3)
    ref = weakref.ref(region)
    material = Material.from_loss_tangent(4.5, 0.0)
    sample = Sample(region=region, material=material)
    model.evaluate(sample)

    del region, sample
    gc.collect()
    assert ref() is not None  # model's cache holds region alive


def test_cache_distinguishes_two_simultaneously_alive_regions():
    a = b = c = 0.03
    _, field = _small_cavity_field()
    model = PerturbationModel(field)

    region1 = Sphere(center=[a / 2, b / 2, c / 2], radius=1e-3)
    region2 = Sphere(center=[a / 2, b / 2, c / 2], radius=2e-3)
    I_E1, I_H1 = model._shape_integrals(region1)
    I_E2, I_H2 = model._shape_integrals(region2)
    assert I_E1 != pytest.approx(I_E2)
    # re-fetching from cache returns the same values, not a collision
    I_E1_again, I_H1_again = model._shape_integrals(region1)
    assert I_E1_again == I_E1 and I_H1_again == I_H1


# --- Edge cases --------------------------------------------------------------

def test_lossless_material_no_wallloss_gives_infinite_q():
    a = b = c = 0.03
    _, field = _small_cavity_field()
    model = PerturbationModel(field, Rs_walls=None)
    region = Sphere(center=[a / 2, b / 2, c / 2], radius=1e-3)
    material = Material(eps=4.5 - 0j, mu=1.0 - 0j)
    sample = Sample(region=region, material=material)
    result = model.evaluate(sample)
    assert result.Q_calc == float("inf")


def test_evaluate_raises_for_nonpassive_material():
    a = b = c = 0.03
    _, field = _small_cavity_field()
    model = PerturbationModel(field)
    region = Sphere(center=[a / 2, b / 2, c / 2], radius=1e-3)
    material = Material(eps=4.5 + 0.1j, mu=1.0 - 0j)  # eps'' < 0, non-passive
    sample = Sample(region=region, material=material)
    with pytest.raises(ValueError):
        model.evaluate(sample)
