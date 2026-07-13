"""Validation suite for Module 1 cavity classes (docs/module1_cavity_equations.md).

Rectangular cavity: Section 1.9 validation targets.
"""
import numpy as np
import pytest
from scipy import constants, integrate

from cavity_perturbation import numerics as num
from cavity_perturbation.cavity import (
    CavityMode,
    CoaxialCavity,
    CylindricalCavity,
    ModeIndex,
    RectangularCavity,
)

COPPER_CONDUCTIVITY = 5.8e7  # S/m


def copper_Rs(f0: float) -> float:
    return np.sqrt(np.pi * f0 * constants.mu_0 / COPPER_CONDUCTIVITY)


# --- Basic construction / defaults -----------------------------------------

def test_default_mode_is_te011():
    cav = RectangularCavity(1.0, 2.0, 3.0)
    assert cav.mode == ModeIndex("TE", (0, 1, 1))


def test_invalid_mode_indices_rejected():
    with pytest.raises(ValueError):
        RectangularCavity(1.0, 2.0, 3.0, ModeIndex("TE", (0, 0, 1)))
    with pytest.raises(ValueError):
        RectangularCavity(1.0, 2.0, 3.0, ModeIndex("TM", (0, 1, 0)))


# --- Section 1.9: resonant frequency, dominant mode ------------------------

def test_resonant_frequency_square_base_dominant_mode():
    """a = b/2 = c/2 reduces to f_r = 1/(b*sqrt(2*eps*mu))."""
    a, b, c = 1.0, 2.0, 2.0
    cav = RectangularCavity(a, b, c)
    expected = 1.0 / (b * np.sqrt(2 * constants.epsilon_0 * constants.mu_0))
    assert cav.f0 == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    "kind,indices",
    [("TE", (0, 1, 1)), ("TE", (1, 0, 1)), ("TE", (1, 1, 2)),
     ("TM", (1, 1, 0)), ("TM", (1, 1, 1)), ("TM", (2, 1, 1))],
)
def test_resonant_frequency_matches_closed_form(kind, indices):
    a, b, c = 1.0, 2.0, 3.0
    m, n, p = indices
    cav = RectangularCavity(a, b, c, ModeIndex(kind, indices))
    expected = (
        1.0 / (2 * np.sqrt(constants.epsilon_0 * constants.mu_0))
        * np.sqrt((m / a) ** 2 + (n / b) ** 2 + (p / c) ** 2)
    )
    assert cav.f0 == pytest.approx(expected, rel=1e-12)


def test_mode_ratio_table_matches_independent_calculation():
    """f_mnp/f_011 for several low-order modes at a fixed a:b:c ratio, checked
    against the Section 1.3 formula computed independently of the class."""
    a, b, c = 1.0, 2.0, 3.0

    def f_direct(m, n, p):
        return np.sqrt((m / a) ** 2 + (n / b) ** 2 + (p / c) ** 2)

    f011 = RectangularCavity(a, b, c, ModeIndex("TE", (0, 1, 1))).f0
    for kind, idx in [("TE", (1, 0, 1)), ("TM", (1, 1, 0)), ("TE", (0, 1, 2))]:
        cav = RectangularCavity(a, b, c, ModeIndex(kind, idx))
        ratio = cav.f0 / f011
        expected_ratio = f_direct(*idx) / f_direct(0, 1, 1)
        assert ratio == pytest.approx(expected_ratio, rel=1e-12)


# --- Section 1.9: cubic copper cavity Q ------------------------------------

def test_cubic_copper_cavity_q_order_of_magnitude():
    # Size a cubic cavity for a ~10 GHz TE011 resonance.
    b = c = 1.5 * constants.c / (2 * 10e9) * np.sqrt(2)
    a = b
    cav = RectangularCavity(a, b, c)
    Rs = copper_Rs(cav.f0)
    Q = cav.Q_wall(Rs)
    assert Q > 0
    assert 1e3 < Q < 5e4  # order-of-magnitude sanity check, not exact digits


def test_q_scales_as_inverse_sqrt_f():
    """Q_c ~ 1/sqrt(f) for fixed geometry proportions, varying overall scale."""
    Qs = []
    fs = []
    for scale in (1.0, 2.0, 4.0):
        a = b = c = scale * 0.03
        cav = RectangularCavity(a, b, c)
        Rs = copper_Rs(cav.f0)
        Qs.append(cav.Q_wall(Rs))
        fs.append(cav.f0)
    ratio_Q = Qs[0] / Qs[-1]
    ratio_f = np.sqrt(fs[-1] / fs[0])
    assert ratio_Q == pytest.approx(ratio_f, rel=1e-6)


# --- Scale invariance (Module 0 / CLAUDE.md convention) --------------------

@pytest.mark.parametrize("kind,indices", [("TE", (0, 1, 1)), ("TM", (1, 1, 1))])
def test_scale_invariance(kind, indices):
    a, b, c = 1.0, 1.3, 1.7
    mode = ModeIndex(kind, indices)
    cav1 = RectangularCavity(a, b, c, mode, amplitude=1.0)
    cav2 = RectangularCavity(a, b, c, mode, amplitude=3.7 - 2.1j)

    assert cav1.f0 == pytest.approx(cav2.f0)
    assert cav1.Q_wall(1e-3) == pytest.approx(cav2.Q_wall(1e-3), rel=1e-10)

    r = np.array([[0.3, 0.4, 0.5], [0.6, 0.2, 0.9]])
    e1, e2 = cav1.E(r), cav2.E(r)
    ratio = np.where(np.abs(e1) > 1e-12, e2 / np.where(e1 == 0, 1, e1), np.nan)
    finite_ratios = ratio[np.isfinite(ratio)]
    assert np.allclose(finite_ratios, 3.7 - 2.1j, rtol=1e-8)


# --- Curl residual (permanent regression test) -----------------------------

@pytest.mark.parametrize("kind,indices", [("TE", (0, 1, 1)), ("TM", (1, 1, 1)), ("TE", (1, 2, 1))])
def test_curl_residual(kind, indices):
    a, b, c = 1.0, 1.3, 1.7
    cav = RectangularCavity(a, b, c, ModeIndex(kind, indices))
    rng = np.random.default_rng(42)
    pts = rng.uniform(0.1, 0.9, size=(6, 3)) * np.array([a, b, c])

    curl_E = num.curl_fd(cav.E, pts)
    curl_H = num.curl_fd(cav.H, pts)
    assert np.allclose(curl_E, -1j * cav._omega * cav.mu * cav.H(pts), atol=1e-4, rtol=1e-4)
    assert np.allclose(curl_H, 1j * cav._omega * cav.eps * cav.E(pts), atol=1e-4, rtol=1e-4)


# --- TE_011 field-pattern sanity check (Section 1.4 note) ------------------

def test_te011_dominant_mode_field_pattern():
    a, b, c = 1.0, 1.3, 1.7
    cav = RectangularCavity(a, b, c)  # default TE_011
    r = np.array([[0.3, 0.4, 0.5]])
    e = cav.E(r)[0]
    h = cav.H(r)[0]
    assert abs(e[1]) < 1e-10  # Ey vanishes
    assert abs(h[0]) < 1e-10  # Hx vanishes
    assert abs(e[0]) > 1e-6 and abs(h[1]) > 1e-6 and abs(h[2]) > 1e-6


# --- Volume energy consistency (architecture doc Module 1 test plan) -------

@pytest.mark.parametrize("kind,indices", [("TE", (0, 1, 1)), ("TM", (1, 1, 1))])
def test_total_stored_energy_matches_brute_force_quadrature(kind, indices):
    a, b, c = 1.0, 1.3, 1.7
    cav = RectangularCavity(a, b, c, ModeIndex(kind, indices))

    def integrand(z, y, x):
        e = cav.E(np.array([x, y, z]))
        return np.sum(np.abs(e) ** 2)

    brute, _ = integrate.tplquad(integrand, 0, a, 0, b, 0, c, epsabs=1e-10, epsrel=1e-8)
    expected = brute * cav.eps / 2.0
    assert cav.total_stored_energy() == pytest.approx(expected, rel=1e-6)


# --- contains / bounding_box ------------------------------------------------

def test_contains_and_bounding_box():
    a, b, c = 1.0, 2.0, 3.0
    cav = RectangularCavity(a, b, c)
    rmin, rmax = cav.bounding_box()
    assert np.allclose(rmin, [0, 0, 0])
    assert np.allclose(rmax, [a, b, c])

    inside = np.array([[0.5, 1.0, 1.5], [-0.1, 1.0, 1.5], [0.5, 2.5, 1.5]])
    mask = cav.contains(inside)
    assert list(mask) == [True, False, False]


def test_is_cavity_mode_subclass():
    assert issubclass(RectangularCavity, CavityMode)


# --- epsilon_bg / mu_bg (Module 4 doc Section 0.1 retroactive addition) ---

def test_rectangular_epsilon_bg_mu_bg_default_vacuum():
    cav = RectangularCavity(1.0, 2.0, 3.0)
    assert cav.epsilon_bg == constants.epsilon_0
    assert cav.mu_bg == constants.mu_0


def test_rectangular_epsilon_bg_mu_bg_passthrough_nonvacuum():
    eps, mu = 2.5 * constants.epsilon_0, 1.3 * constants.mu_0
    cav = RectangularCavity(1.0, 2.0, 3.0, eps=eps, mu=mu)
    assert cav.epsilon_bg == eps
    assert cav.mu_bg == mu


def test_cylindrical_epsilon_bg_mu_bg_default_vacuum():
    cav = CylindricalCavity(1.0, 2.0)
    assert cav.epsilon_bg == constants.epsilon_0
    assert cav.mu_bg == constants.mu_0


def test_coaxial_epsilon_bg_mu_bg_default_vacuum():
    cav = CoaxialCavity(0.01, 0.023, 0.5)
    assert cav.epsilon_bg == constants.epsilon_0
    assert cav.mu_bg == constants.mu_0


# ===========================================================================
# Cylindrical cavity -- Section 2.9 validation targets
# ===========================================================================

def test_cyl_default_mode_is_tm010():
    cav = CylindricalCavity(1.0, 2.0)
    assert cav.mode == ModeIndex("TM", (0, 1, 0))


def test_cyl_bessel_zeros_known_values():
    assert num.bessel_zero_tm(0, 1) == pytest.approx(2.405, abs=1e-3)
    assert num.bessel_zero_tm(1, 1) == pytest.approx(3.832, abs=1e-3)
    assert num.bessel_zero_te(1, 1) == pytest.approx(1.841, abs=1e-3)


def test_cyl_tm010_resonant_frequency_independent_of_length():
    a = 0.05
    X01 = num.bessel_zero_tm(0, 1)
    expected = X01 / (2 * np.pi * a * np.sqrt(constants.epsilon_0 * constants.mu_0))
    for d in (0.05, 0.1, 0.5):
        cav = CylindricalCavity(a, d, ModeIndex("TM", (0, 1, 0)))
        assert cav.f0 == pytest.approx(expected, rel=1e-10)


def test_cyl_mode_crossover_near_d_over_a_2():
    a = 0.05
    ds = np.linspace(1.5 * a, 2.5 * a, 21)
    diffs = []
    for d in ds:
        f010 = CylindricalCavity(a, d, ModeIndex("TM", (0, 1, 0))).f0
        f111 = CylindricalCavity(a, d, ModeIndex("TE", (1, 1, 1))).f0
        diffs.append(f010 - f111)
    diffs = np.array(diffs)
    # sign change (TM010 dominant below crossover, TE111 dominant above)
    assert diffs[0] < 0 < diffs[-1]
    crossover_d = np.interp(0.0, diffs, ds)
    assert crossover_d / a == pytest.approx(2.0, abs=0.15)


@pytest.mark.parametrize("kind,indices", [("TM", (0, 1, 0)), ("TE", (1, 1, 1)), ("TM", (1, 1, 1))])
def test_cyl_curl_residual(kind, indices):
    a, d = 0.05, 0.09
    cav = CylindricalCavity(a, d, ModeIndex(kind, indices))
    rng = np.random.default_rng(7)
    rho = rng.uniform(0.1 * a, 0.9 * a, size=6)
    phi = rng.uniform(0, 2 * np.pi, size=6)
    z = rng.uniform(0.1 * d, 0.9 * d, size=6)
    pts = np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=-1)

    curl_E = num.curl_fd(cav.E, pts)
    curl_H = num.curl_fd(cav.H, pts)
    assert np.allclose(curl_E, -1j * cav._omega * cav.mu * cav.H(pts), atol=1e-3, rtol=1e-3)
    assert np.allclose(curl_H, 1j * cav._omega * cav.eps * cav.E(pts), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("kind,indices", [("TM", (0, 1, 0)), ("TE", (1, 1, 1)), ("TM", (1, 1, 1))])
def test_cyl_scale_invariance(kind, indices):
    a, d = 0.05, 0.09
    mode = ModeIndex(kind, indices)
    cav1 = CylindricalCavity(a, d, mode, amplitude=1.0)
    cav2 = CylindricalCavity(a, d, mode, amplitude=2.5 + 1.1j)
    assert cav1.f0 == pytest.approx(cav2.f0)
    assert cav1.Q_wall(1e-3) == pytest.approx(cav2.Q_wall(1e-3), rel=1e-8)


@pytest.mark.parametrize("kind,indices", [("TM", (0, 1, 0)), ("TE", (1, 1, 1)), ("TM", (1, 1, 1))])
def test_cyl_total_stored_energy_matches_brute_force_quadrature(kind, indices):
    a, d = 0.05, 0.09
    cav = CylindricalCavity(a, d, ModeIndex(kind, indices))

    def integrand(phi, rho, z):
        r = np.array([rho * np.cos(phi), rho * np.sin(phi), z])
        e = cav.E(r)
        return np.sum(np.abs(e) ** 2) * rho

    brute, _ = integrate.tplquad(
        integrand, 0, d, 0, a, 0, 2 * np.pi, epsabs=1e-12, epsrel=1e-7
    )
    expected = brute * cav.eps / 2.0
    assert cav.total_stored_energy() == pytest.approx(expected, rel=1e-5)


def test_cyl_rho_axis_no_nan():
    """Section 2.8 step 3: quadrature points landing exactly on the axis
    must not produce NaN, for n=0, 1, or >=2."""
    for n in (0, 1, 2):
        cav = CylindricalCavity(0.05, 0.09, ModeIndex("TM", (n, 1, 0)))
        pts = np.array([[0.0, 0.0, 0.03]])
        e, h = cav.E(pts), cav.H(pts)
        assert np.all(np.isfinite(e))
        assert np.all(np.isfinite(h))


def test_cyl_contains_and_bounding_box():
    a, d = 0.05, 0.09
    cav = CylindricalCavity(a, d)
    rmin, rmax = cav.bounding_box()
    assert np.allclose(rmin, [-a, -a, 0])
    assert np.allclose(rmax, [a, a, d])
    pts = np.array([[0.0, 0.0, 0.03], [0.06, 0.0, 0.03], [0.0, 0.0, -0.01]])
    mask = cav.contains(pts)
    assert list(mask) == [True, False, False]


# ===========================================================================
# Coaxial cavity -- Section 3.8 validation targets
# ===========================================================================

def test_coax_default_mode_is_q1():
    cav = CoaxialCavity(0.01, 0.023, 0.5)
    assert cav.mode == ModeIndex("TEM", (1,))


def test_coax_resonant_frequency_air_filled():
    L = 0.5
    cav = CoaxialCavity(0.01, 0.023, L, ModeIndex("TEM", (1,)))
    expected = constants.c / (2.0 * L)
    assert cav.f0 == pytest.approx(expected, rel=1e-12)


def test_coax_impedance_50_ohm_ratio():
    """Z0 = 50 Ohm reproduces the standard b/a ~ 2.3 ratio for an air line."""
    a = 0.01
    b = a * 2.3
    cav = CoaxialCavity(a, b, 0.5)
    assert cav._Z0 == pytest.approx(50.0, rel=5e-3)


@pytest.mark.parametrize("q", [1, 2, 3])
def test_coax_resonant_frequency_half_wave_spacing(q):
    L = 0.5
    cav = CoaxialCavity(0.01, 0.023, L, ModeIndex("TEM", (q,)))
    expected = q * constants.c / (2.0 * L)
    assert cav.f0 == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("amplitude", [1.0, 3.3 - 1.7j])
def test_coax_scale_invariance(amplitude):
    a, b, L = 0.01, 0.023, 0.5
    cav1 = CoaxialCavity(a, b, L, amplitude=1.0)
    cav2 = CoaxialCavity(a, b, L, amplitude=amplitude)
    assert cav1.f0 == pytest.approx(cav2.f0)
    assert cav1.Q_wall(1e-3) == pytest.approx(cav2.Q_wall(1e-3), rel=1e-10)


def test_coax_total_stored_energy_matches_brute_force_quadrature():
    a, b, L = 0.01, 0.023, 0.5
    cav = CoaxialCavity(a, b, L)

    def integrand(phi, rho, z):
        r = np.array([rho * np.cos(phi), rho * np.sin(phi), z])
        e = cav.E(r)
        return np.sum(np.abs(e) ** 2) * rho

    brute, _ = integrate.tplquad(integrand, 0, L, a, b, 0, 2 * np.pi, epsabs=1e-12, epsrel=1e-8)
    expected = brute * cav.eps / 2.0
    assert cav.total_stored_energy() == pytest.approx(expected, rel=1e-6)


def test_coax_q_wall_positive_and_reasonable():
    a, b, L = 0.01, 0.023, 0.5
    cav = CoaxialCavity(a, b, L)
    sigma_copper = 5.8e7
    Rs = np.sqrt(np.pi * cav.f0 * constants.mu_0 / sigma_copper)
    Q = cav.Q_wall(Rs)
    assert Q > 0
    assert 1e2 < Q < 1e5


def test_coax_contains_and_bounding_box():
    a, b, L = 0.01, 0.023, 0.5
    cav = CoaxialCavity(a, b, L)
    rmin, rmax = cav.bounding_box()
    assert np.allclose(rmin, [-b, -b, 0])
    assert np.allclose(rmax, [b, b, L])
    pts = np.array([[0.015, 0.0, 0.2], [0.0, 0.0, 0.2], [0.03, 0.0, 0.2]])
    mask = cav.contains(pts)
    assert list(mask) == [True, False, False]
