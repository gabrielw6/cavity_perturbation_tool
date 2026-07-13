"""Standalone tests for the shared Module 1 numerics (docs/module1_cavity_equations.md
Section 0), validated before any concrete cavity class per Section 1.8 step 1."""
import numpy as np
import pytest
from scipy import integrate, special

from cavity_perturbation import numerics as num


@pytest.mark.parametrize("k", [0, 1, 2, 3])
@pytest.mark.parametrize("L", [1.0, 2.5])
def test_cos2_integral_matches_quadrature(k, L):
    expected, _ = integrate.quad(lambda u: np.cos(k * np.pi * u / L) ** 2, 0, L)
    assert num.cos2_integral(k, L) == pytest.approx(expected)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
@pytest.mark.parametrize("L", [1.0, 2.5])
def test_sin2_integral_matches_quadrature(k, L):
    expected, _ = integrate.quad(lambda u: np.sin(k * np.pi * u / L) ** 2, 0, L)
    assert num.sin2_integral(k, L) == pytest.approx(expected)


@pytest.mark.parametrize("n,p", [(0, 1), (0, 2), (1, 1), (2, 3)])
def test_bessel_tm_radial_integral_matches_quadrature(n, p):
    a = 1.7
    X_np = num.bessel_zero_tm(n, p)
    expected, _ = integrate.quad(
        lambda rho: rho * special.jv(n, X_np * rho / a) ** 2, 0, a
    )
    assert num.bessel_tm_radial_integral(n, X_np, a) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("n,p", [(0, 1), (1, 1), (1, 2), (2, 1)])
def test_bessel_te_radial_integral_matches_quadrature(n, p):
    a = 1.7
    Xp_np = num.bessel_zero_te(n, p)
    expected, _ = integrate.quad(
        lambda rho: rho * special.jv(n, Xp_np * rho / a) ** 2, 0, a
    )
    assert num.bessel_te_radial_integral(n, Xp_np, a) == pytest.approx(expected, rel=1e-6)


def test_bessel_zeros_known_values():
    # Table 5-2/5-3-style known values (Section 2.8 step 1)
    assert num.bessel_zero_tm(0, 1) == pytest.approx(2.405, abs=1e-3)
    assert num.bessel_zero_te(1, 1) == pytest.approx(1.841, abs=1e-3)


def test_zhat_cross():
    v = np.array([[1.0, 2.0, 0.0], [3.0, -1.0, 0.0]])
    result = num.zhat_cross(v)
    expected = np.array([[-2.0, 1.0, 0.0], [1.0, 3.0, 0.0]])
    assert np.allclose(result, expected)


def test_tez_tmz_recipe_curl_residual_tm():
    """Section 1.8 step 2: trivial made-up Phi, check curl E = -j*omega*mu*H
    and curl H = j*omega*eps*E numerically before trusting the recipe on any
    real geometry.

    Phi's transverse part cos(x)cos(y) satisfies (grad_t^2 + k_c2)Phi = 0 with
    k_c2 = 2 (both second derivatives contribute -1x each). The dispersion
    relation k_c2 + kz^2 = omega^2*eps*mu must also hold for the fields to
    actually satisfy Maxwell's equations -- pick kz, omega, eps, mu consistent
    with that, not arbitrarily, or the curl residual is a real physics
    violation rather than a recipe bug."""
    k_c2 = 2.0
    kz = 1.0
    eps = 1.0
    mu = 1.0
    omega = np.sqrt(k_c2 + kz**2)  # dispersion: k^2 = k_c2 + kz^2 = omega^2*eps*mu

    def Phi(r):
        x, y, z = r[..., 0], r[..., 1], r[..., 2]
        return np.cos(x) * np.cos(y) * np.cos(kz * z)

    def grad_t_Phi(r):
        x, y, z = r[..., 0], r[..., 1], r[..., 2]
        out = np.zeros(r.shape, dtype=complex)
        out[..., 0] = -np.sin(x) * np.cos(y) * np.cos(kz * z)
        out[..., 1] = -np.cos(x) * np.sin(y) * np.cos(kz * z)
        return out

    def grad_t_dPhi_dz(r):
        x, y, z = r[..., 0], r[..., 1], r[..., 2]
        out = np.zeros(r.shape, dtype=complex)
        out[..., 0] = kz * np.sin(x) * np.cos(y) * np.sin(kz * z)
        out[..., 1] = kz * np.cos(x) * np.sin(y) * np.sin(kz * z)
        return out

    E, H = num.tez_tmz_fields("TM", Phi, grad_t_Phi, grad_t_dPhi_dz, k_c2, omega, eps, mu)

    rng = np.random.default_rng(0)
    pts = rng.uniform(0.2, 1.0, size=(5, 3))

    curl_E = num.curl_fd(E, pts)
    curl_H = num.curl_fd(H, pts)
    assert np.allclose(curl_E, -1j * omega * mu * H(pts), atol=1e-5)
    assert np.allclose(curl_H, 1j * omega * eps * E(pts), atol=1e-5)


def test_tez_tmz_recipe_curl_residual_te():
    k_c2 = 2.0
    kz = 1.0
    eps = 1.0
    mu = 1.0
    omega = np.sqrt(k_c2 + kz**2)

    def Phi(r):
        x, y, z = r[..., 0], r[..., 1], r[..., 2]
        return np.cos(x) * np.cos(y) * np.sin(kz * z)

    def grad_t_Phi(r):
        x, y, z = r[..., 0], r[..., 1], r[..., 2]
        out = np.zeros(r.shape, dtype=complex)
        out[..., 0] = -np.sin(x) * np.cos(y) * np.sin(kz * z)
        out[..., 1] = -np.cos(x) * np.sin(y) * np.sin(kz * z)
        return out

    def grad_t_dPhi_dz(r):
        x, y, z = r[..., 0], r[..., 1], r[..., 2]
        out = np.zeros(r.shape, dtype=complex)
        out[..., 0] = -kz * np.sin(x) * np.cos(y) * np.cos(kz * z)
        out[..., 1] = -kz * np.cos(x) * np.sin(y) * np.cos(kz * z)
        return out

    E, H = num.tez_tmz_fields("TE", Phi, grad_t_Phi, grad_t_dPhi_dz, k_c2, omega, eps, mu)

    rng = np.random.default_rng(1)
    pts = rng.uniform(0.2, 1.0, size=(5, 3))

    curl_E = num.curl_fd(E, pts)
    curl_H = num.curl_fd(H, pts)
    assert np.allclose(curl_E, -1j * omega * mu * H(pts), atol=1e-5)
    assert np.allclose(curl_H, 1j * omega * eps * E(pts), atol=1e-5)
