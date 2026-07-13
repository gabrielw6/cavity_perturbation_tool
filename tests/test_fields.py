"""Validation suite for Module 2 (docs/module2_fields_equations.md Section 8).

Module 3 (`sample.py`) is not yet built beyond the `SampleRegion` interface
(see CLAUDE.md Status), so this file supplies its own SampleRegion test
doubles -- Gauss-Legendre tensor-product quadrature over a box or a
cylindrical/annular volume -- purely to exercise Module 2 against realistic
(region, field) pairs. These are not Module 3 deliverables.
"""
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from cavity_perturbation import fields
from cavity_perturbation.cavity import (
    CoaxialCavity,
    CylindricalCavity,
    ModeIndex,
    RectangularCavity,
)
from cavity_perturbation.fields import AnalyticalField, FieldProvider, RitzField
from cavity_perturbation.sample import SampleRegion


def _gauss_legendre_1d(lo: float, hi: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    xi, wi = leggauss(max(n, 1))
    x = 0.5 * (hi - lo) * xi + 0.5 * (hi + lo)
    w = 0.5 * (hi - lo) * wi
    return x, w


class BoxRegion(SampleRegion):
    """Axis-aligned box, Gauss-Legendre tensor-product quadrature."""

    def __init__(self, xlo, xhi, ylo, yhi, zlo, zhi):
        self.bounds = (xlo, xhi, ylo, yhi, zlo, zhi)

    def contains(self, r: np.ndarray) -> np.ndarray:
        r = np.atleast_2d(r)
        xlo, xhi, ylo, yhi, zlo, zhi = self.bounds
        x, y, z = r[..., 0], r[..., 1], r[..., 2]
        return (xlo <= x) & (x <= xhi) & (ylo <= y) & (y <= yhi) & (zlo <= z) & (z <= zhi)

    def volume(self) -> float:
        xlo, xhi, ylo, yhi, zlo, zhi = self.bounds
        return (xhi - xlo) * (yhi - ylo) * (zhi - zlo)

    def quadrature_points(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        xlo, xhi, ylo, yhi, zlo, zhi = self.bounds
        m = max(1, round(n ** (1 / 3)))
        xu, xw = _gauss_legendre_1d(xlo, xhi, m)
        yu, yw = _gauss_legendre_1d(ylo, yhi, m)
        zu, zw = _gauss_legendre_1d(zlo, zhi, m)
        X, Y, Z = np.meshgrid(xu, yu, zu, indexing="ij")
        WX, WY, WZ = np.meshgrid(xw, yw, zw, indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        w = (WX * WY * WZ).ravel()
        return pts, w

    @property
    def shape_kind(self) -> str:
        return "generic"


class CylinderVolumeRegion(SampleRegion):
    """Full cylinder (rho_lo=0) or annulus (rho_lo>0), Gauss-Legendre
    tensor-product quadrature in (rho, phi, z)."""

    def __init__(self, rho_lo: float, rho_hi: float, length: float):
        self.rho_lo, self.rho_hi, self.length = rho_lo, rho_hi, length

    def contains(self, r: np.ndarray) -> np.ndarray:
        r = np.atleast_2d(r)
        x, y, z = r[..., 0], r[..., 1], r[..., 2]
        rho = np.hypot(x, y)
        return (self.rho_lo <= rho) & (rho <= self.rho_hi) & (0 <= z) & (z <= self.length)

    def volume(self) -> float:
        return np.pi * (self.rho_hi**2 - self.rho_lo**2) * self.length

    def quadrature_points(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        m = max(1, round(n ** (1 / 3)))
        rho_u, rho_w = _gauss_legendre_1d(self.rho_lo, self.rho_hi, m)
        phi_u, phi_w = _gauss_legendre_1d(0.0, 2 * np.pi, m)
        z_u, z_w = _gauss_legendre_1d(0.0, self.length, m)
        RHO, PHI, Z = np.meshgrid(rho_u, phi_u, z_u, indexing="ij")
        WR, WP, WZ = np.meshgrid(rho_w, phi_w, z_w, indexing="ij")
        X, Y = RHO * np.cos(PHI), RHO * np.sin(PHI)
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        w = (RHO * WR * WP * WZ).ravel()  # extra rho Jacobian factor
        return pts, w

    @property
    def shape_kind(self) -> str:
        return "generic"


class BrokenWeightsRegion(SampleRegion):
    """Wraps a real region but returns deliberately wrong weights, to test
    that Module 2's volume-consistency check (Section 1.4) actually fires."""

    def __init__(self, inner: SampleRegion):
        self._inner = inner

    def contains(self, r: np.ndarray) -> np.ndarray:
        return self._inner.contains(r)

    def volume(self) -> float:
        return self._inner.volume()

    def quadrature_points(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        pts, w = self._inner.quadrature_points(n)
        return pts, w * 0.5

    @property
    def shape_kind(self) -> str:
        return self._inner.shape_kind


# --- hermitian_density (Section 1.2) ---------------------------------------

def test_hermitian_density_hand_picked_vectors():
    vals = np.array([[1 + 2j, 0, 0], [1 + 1j, 2 + 0j, -3j]])
    density = fields.hermitian_density(vals)
    assert density == pytest.approx([5.0, 1 + 1 + 4 + 9])


def test_hermitian_density_is_real_nonnegative():
    rng = np.random.default_rng(0)
    vals = rng.uniform(-1, 1, size=(20, 3)) + 1j * rng.uniform(-1, 1, size=(20, 3))
    density = fields.hermitian_density(vals)
    assert not np.iscomplexobj(density)
    assert np.all(density >= 0)


# --- converge_by_doubling (Section 1.5) ------------------------------------

def test_converge_by_doubling_recovers_volume():
    """Toy integrand with a known closed-form answer: integrate 1 over a
    region to recover its volume, before pointing the helper at a real field."""
    region = BoxRegion(0, 1, 0, 2, 0, 3)

    def estimate(n):
        _, w = region.quadrature_points(n)
        return float(np.sum(w))

    result = fields.converge_by_doubling(estimate, n_start=8)
    assert result == pytest.approx(region.volume(), rel=1e-10)


def test_converge_by_doubling_raises_when_never_converging():
    # Strictly alternates every call, regardless of n -- never converges.
    state = {"toggle": False}

    def bad_estimator(n):
        state["toggle"] = not state["toggle"]
        return 1.0 if state["toggle"] else 2.0

    with pytest.raises(RuntimeError):
        fields.converge_by_doubling(bad_estimator, n_start=50, tol=1e-4, max_doublings=5)


def test_converge_by_doubling_terminates_quickly_for_smooth_case():
    calls = []

    def estimate(n):
        calls.append(n)
        return 1.0 - 1.0 / n

    result = fields.converge_by_doubling(estimate, n_start=1000, tol=1e-4)
    assert result == pytest.approx(1.0, abs=1e-3)
    # Well short of the max_doublings=10 cap (11 possible calls) -- this is
    # the "converges for a smooth interior region" case, not the pathological one.
    assert len(calls) <= 6


# --- FieldProvider.integrate_field_energy defensive checks (Section 1.4) --

def test_volume_consistency_check_fires_on_broken_region():
    a, b, c = 0.03, 0.03, 0.03
    cav = RectangularCavity(a, b, c)
    field = AnalyticalField(cav)
    good_region = BoxRegion(0, a, 0, b, 0, c)
    broken_region = BrokenWeightsRegion(good_region)
    with pytest.raises(ValueError):
        field.integrate_field_energy(broken_region, "E")


def test_volume_consistency_check_passes_on_good_region():
    a, b, c = 0.03, 0.03, 0.03
    cav = RectangularCavity(a, b, c)
    field = AnalyticalField(cav)
    region = BoxRegion(0, a, 0, b, 0, c)
    # should not raise
    field.integrate_field_energy(region, "E")


# --- AnalyticalField / whole-cavity consistency (Sections 3, 8) -----------

@pytest.mark.parametrize("kind,indices", [("TE", (0, 1, 1)), ("TM", (1, 1, 1))])
def test_rectangular_whole_cavity_energy_consistency(kind, indices):
    a, b, c = 0.03, 0.04, 0.05
    cav = RectangularCavity(a, b, c, ModeIndex(kind, indices))
    field = AnalyticalField(cav)
    region = BoxRegion(0, a, 0, b, 0, c)

    I_E = field.integrate_field_energy(region, "E")
    I_H = field.integrate_field_energy(region, "H")
    W = field.total_stored_energy()

    assert I_E == pytest.approx(2.0 * W / cav.eps, rel=1e-6)
    assert I_H == pytest.approx(2.0 * W / cav.mu, rel=1e-6)


@pytest.mark.parametrize("kind,indices", [("TM", (0, 1, 0)), ("TE", (1, 1, 1))])
def test_cylindrical_whole_cavity_energy_consistency(kind, indices):
    radius, length = 0.02, 0.03
    cav = CylindricalCavity(radius, length, ModeIndex(kind, indices))
    field = AnalyticalField(cav)
    region = CylinderVolumeRegion(0.0, radius, length)

    I_E = field.integrate_field_energy(region, "E")
    W = field.total_stored_energy()
    assert I_E == pytest.approx(2.0 * W / cav.eps, rel=1e-5)


def test_coaxial_whole_cavity_energy_consistency():
    r_inner, r_outer, length = 0.01, 0.023, 0.2
    cav = CoaxialCavity(r_inner, r_outer, length)
    field = AnalyticalField(cav)
    region = CylinderVolumeRegion(r_inner, r_outer, length)

    I_E = field.integrate_field_energy(region, "E")
    W = field.total_stored_energy()
    assert I_E == pytest.approx(2.0 * W / cav.eps, rel=1e-6)


def test_analytical_field_passes_through_f0_qwall_energy_unchanged():
    a, b, c = 0.03, 0.04, 0.05
    cav = RectangularCavity(a, b, c)
    field = AnalyticalField(cav)
    assert field.f0 == cav.f0
    assert field.Q_wall(1e-3) == cav.Q_wall(1e-3)
    assert field.total_stored_energy() == cav.total_stored_energy()


def test_analytical_field_is_field_provider_subclass():
    assert issubclass(AnalyticalField, FieldProvider)


# --- epsilon_bg / mu_bg passthrough (Module 4 doc Section 0.1) ------------

def test_analytical_field_epsilon_bg_mu_bg_passthrough():
    import scipy.constants as constants

    eps, mu = 2.5 * constants.epsilon_0, 1.3 * constants.mu_0
    cav = RectangularCavity(0.03, 0.04, 0.05, eps=eps, mu=mu)
    field = AnalyticalField(cav)
    assert field.epsilon_bg == eps
    assert field.mu_bg == mu


def test_ritz_field_epsilon_bg_mu_bg_raise_not_implemented():
    # __init__ always raises, so exercise the methods directly on the class
    # (mirrors how test_ritz_field_stub_raises_not_implemented handles this).
    with pytest.raises(NotImplementedError):
        RitzField.epsilon_bg.fget(object.__new__(RitzField))
    with pytest.raises(NotImplementedError):
        RitzField.mu_bg.fget(object.__new__(RitzField))


# --- Scale invariance (Section 1.4 / 8) ------------------------------------

@pytest.mark.parametrize("field_name", ["E", "H"])
def test_scale_invariance_of_integrate_field_energy(field_name):
    a, b, c = 0.03, 0.04, 0.05
    mode = ModeIndex("TE", (0, 1, 1))
    cav1 = RectangularCavity(a, b, c, mode, amplitude=1.0)
    cav2 = RectangularCavity(a, b, c, mode, amplitude=2.0 - 1.5j)
    region = BoxRegion(0, a, 0, b, 0, c)

    I1 = AnalyticalField(cav1).integrate_field_energy(region, field_name)
    I2 = AnalyticalField(cav2).integrate_field_energy(region, field_name)

    assert I2 == pytest.approx(abs(2.0 - 1.5j) ** 2 * I1, rel=1e-6)


# --- RitzField stub ---------------------------------------------------------

def test_ritz_field_stub_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        RitzField(basis_functions=[], coefficients=np.array([]))


def test_ritz_field_is_field_provider_subclass():
    assert issubclass(RitzField, FieldProvider)
