"""Module 1 -- Analytical Cavity Library.

Empty-cavity resonant fields, f0, and Q_wall for Rectangular, Cylindrical, and
Coaxial cavities. See docs/module1_cavity_equations.md for the full derivation
and docs/cavity_perturbation_modules_1-5.md for the CavityMode contract.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import constants, integrate, special

from . import numerics as _num

Array = np.ndarray
_FieldFuncs = tuple[
    Callable[[Array], Array], Callable[[Array], Array], Callable[[Array], Array]
]


@dataclass(frozen=True)
class ModeIndex:
    """Generic mode label.

    Meaning of `indices` depends on cavity type:
    rectangular (m, n, p); circular TE/TM (p, q) with n given separately via
    `kind`... concretely: rectangular (m, n, p); cylindrical (n, p, q);
    coaxial (q,).
    """

    kind: str
    indices: tuple[int, ...]


class CavityMode(ABC):
    """One resonant mode of one empty (unperturbed) cavity."""

    @abstractmethod
    def E(self, r: Array) -> Array:
        """r: (3,) or (N,3) in meters, cavity-local Cartesian frame.
        Returns complex field, same leading shape as r, units V/m
        (arbitrary overall scale -- see normalization convention)."""

    @abstractmethod
    def H(self, r: Array) -> Array:
        """Same contract as E, units A/m, same arbitrary scale as E."""

    @property
    @abstractmethod
    def f0(self) -> float:
        """Resonant frequency of this mode in Hz, closed-form for empty cavity."""

    @property
    @abstractmethod
    def epsilon_bg(self) -> complex:
        """Absolute (SI) permittivity of the cavity's background fill medium
        -- whatever this instance was constructed with, vacuum epsilon_0 by
        default. Retroactive addition for Module 4 (module4 doc Section 0.1):
        this is where absolute (Module 1/2) and relative (Module 3 Material)
        permittivity meet, and the conversion must be explicit."""

    @property
    @abstractmethod
    def mu_bg(self) -> complex:
        """Absolute (SI) permeability of the cavity's background fill medium,
        same rationale as epsilon_bg."""

    @abstractmethod
    def Q_wall(self, Rs: float) -> float:
        """Unloaded Q from finite wall conductivity, given surface resistance
        Rs [Ohm]."""

    @abstractmethod
    def stored_energy_density(self, r: Array) -> tuple[Array, Array]:
        """Returns (w_e(r), w_m(r)): local electric and magnetic energy
        densities (eps/2*|E|^2, mu/2*|H|^2) at the same arbitrary scale as
        E, H above."""

    @abstractmethod
    def total_stored_energy(self) -> float:
        """W = integral over V of (w_e + w_m) dV, closed form for this mode."""

    def bounding_box(self) -> tuple[Array, Array]:
        """(rmin, rmax) axis-aligned box containing the cavity volume."""
        raise NotImplementedError

    @abstractmethod
    def contains(self, r: Array) -> Array:
        """Boolean mask, True where r is inside the cavity volume."""


class RectangularCavity(CavityMode):
    """TE_mnp / TM_mnp modes of an a x b x c rectangular cavity, one corner
    at the origin, per docs/module1_cavity_equations.md Section 1.

    `mode.indices` = (m, n, p). Default mode is TE_011 (dominant for a<b<c
    per Section 1.3).
    """

    def __init__(
        self,
        a: float,
        b: float,
        c: float,
        mode: ModeIndex | None = None,
        amplitude: complex = 1.0,
        eps: float = constants.epsilon_0,
        mu: float = constants.mu_0,
    ) -> None:
        if mode is None:
            mode = ModeIndex("TE", (0, 1, 1))
        if mode.kind not in ("TE", "TM"):
            raise ValueError(f"mode.kind must be 'TE' or 'TM', got {mode.kind!r}")
        m, n, p = mode.indices
        if mode.kind == "TE":
            if m == 0 and n == 0:
                raise ValueError("TE_mnp requires m, n not both zero")
            if p < 1:
                raise ValueError("TE_mnp requires p >= 1")
        else:
            if m < 1 or n < 1:
                raise ValueError("TM_mnp requires m, n >= 1")
            if p < 0:
                raise ValueError("TM_mnp requires p >= 0")

        self.a, self.b, self.c = a, b, c
        self.mode = mode
        self.amplitude = amplitude
        self.eps = eps
        self.mu = mu
        self._m, self._n, self._p = m, n, p

        self._k_c2 = (m * np.pi / a) ** 2 + (n * np.pi / b) ** 2
        self._f0 = (
            1.0
            / (2.0 * np.sqrt(eps * mu))
            * np.sqrt((m / a) ** 2 + (n / b) ** 2 + (p / c) ** 2)
        )
        self._omega = 2.0 * np.pi * self._f0

        Phi, grad_t_Phi, grad_t_dPhi_dz = self._make_mode_function()
        self._E, self._H = _num.tez_tmz_fields(
            mode.kind, Phi, grad_t_Phi, grad_t_dPhi_dz, self._k_c2, self._omega, eps, mu
        )

    def _make_mode_function(self) -> _FieldFuncs:
        a, b, c = self.a, self.b, self.c
        m, n, p = self._m, self._n, self._p
        amp = self.amplitude
        kx, ky, kz = m * np.pi / a, n * np.pi / b, p * np.pi / c

        if self.mode.kind == "TE":
            def Phi(r: Array) -> Array:
                x, y, z = r[..., 0], r[..., 1], r[..., 2]
                return amp * np.cos(kx * x) * np.cos(ky * y) * np.sin(kz * z)

            def grad_t_Phi(r: Array) -> Array:
                x, y, z = r[..., 0], r[..., 1], r[..., 2]
                out = np.zeros(r.shape, dtype=complex)
                out[..., 0] = -amp * kx * np.sin(kx * x) * np.cos(ky * y) * np.sin(kz * z)
                out[..., 1] = -amp * ky * np.cos(kx * x) * np.sin(ky * y) * np.sin(kz * z)
                return out

            def grad_t_dPhi_dz(r: Array) -> Array:
                x, y, z = r[..., 0], r[..., 1], r[..., 2]
                out = np.zeros(r.shape, dtype=complex)
                out[..., 0] = -amp * kx * kz * np.sin(kx * x) * np.cos(ky * y) * np.cos(kz * z)
                out[..., 1] = -amp * ky * kz * np.cos(kx * x) * np.sin(ky * y) * np.cos(kz * z)
                return out
        else:
            def Phi(r: Array) -> Array:
                x, y, z = r[..., 0], r[..., 1], r[..., 2]
                return amp * np.sin(kx * x) * np.sin(ky * y) * np.cos(kz * z)

            def grad_t_Phi(r: Array) -> Array:
                x, y, z = r[..., 0], r[..., 1], r[..., 2]
                out = np.zeros(r.shape, dtype=complex)
                out[..., 0] = amp * kx * np.cos(kx * x) * np.sin(ky * y) * np.cos(kz * z)
                out[..., 1] = amp * ky * np.sin(kx * x) * np.cos(ky * y) * np.cos(kz * z)
                return out

            def grad_t_dPhi_dz(r: Array) -> Array:
                x, y, z = r[..., 0], r[..., 1], r[..., 2]
                out = np.zeros(r.shape, dtype=complex)
                out[..., 0] = -amp * kx * kz * np.cos(kx * x) * np.sin(ky * y) * np.sin(kz * z)
                out[..., 1] = -amp * ky * kz * np.sin(kx * x) * np.cos(ky * y) * np.sin(kz * z)
                return out

        return Phi, grad_t_Phi, grad_t_dPhi_dz

    def E(self, r: Array) -> Array:
        return self._E(r)

    def H(self, r: Array) -> Array:
        return self._H(r)

    @property
    def f0(self) -> float:
        return self._f0

    @property
    def epsilon_bg(self) -> complex:
        return self.eps

    @property
    def mu_bg(self) -> complex:
        return self.mu

    def stored_energy_density(self, r: Array) -> tuple[Array, Array]:
        e = self.E(r)
        h = self.H(r)
        w_e = self.eps / 2.0 * np.sum(np.abs(e) ** 2, axis=-1)
        w_m = self.mu / 2.0 * np.sum(np.abs(h) ** 2, axis=-1)
        return w_e, w_m

    def total_stored_energy(self) -> float:
        """Closed form (Section 1.5): W = eps/2 * integral |E|^2 dV, each
        Cartesian component of E is a single product-of-trig term, so the
        volume integral is a direct product of the three 1-D identities
        (Section 0.4) -- no summation over multiple terms per component."""
        a, b, c = self.a, self.b, self.c
        m, n, p = self._m, self._n, self._p
        amp2 = abs(self.amplitude) ** 2
        k_c2 = self._k_c2
        kx, ky, kz = m * np.pi / a, n * np.pi / b, p * np.pi / c

        Ix_cos, Ix_sin = _num.cos2_integral(m, a), _num.sin2_integral(m, a)
        Iy_cos, Iy_sin = _num.cos2_integral(n, b), _num.sin2_integral(n, b)
        Iz_cos, Iz_sin = _num.cos2_integral(p, c), _num.sin2_integral(p, c)

        if self.mode.kind == "TE":
            # Ex ~ cos(kx x) sin(ky y) sin(kz z), Ey ~ sin(kx x) cos(ky y) sin(kz z), Ez = 0
            int_Ex2 = amp2 * (ky / k_c2) ** 2 * Ix_cos * Iy_sin * Iz_sin
            int_Ey2 = amp2 * (kx / k_c2) ** 2 * Ix_sin * Iy_cos * Iz_sin
            int_E2 = (self._omega * self.mu) ** 2 * (int_Ex2 + int_Ey2)
        else:
            # Ez ~ sin(kx x) sin(ky y) cos(kz z)
            int_Ez2 = amp2 * Ix_sin * Iy_sin * Iz_cos
            # Ex ~ cos(kx x) sin(ky y) sin(kz z), Ey ~ sin(kx x) cos(ky y) sin(kz z)
            int_Ex2 = amp2 * (kx * kz / k_c2) ** 2 * Ix_cos * Iy_sin * Iz_sin
            int_Ey2 = amp2 * (ky * kz / k_c2) ** 2 * Ix_sin * Iy_cos * Iz_sin
            int_E2 = int_Ez2 + int_Ex2 + int_Ey2

        return self.eps / 2.0 * int_E2

    def Q_wall(self, Rs: float) -> float:
        """Closed form (Section 1.6): six wall faces, each contributing the
        tangential-H surface integral via the shared trig identities. The
        x=0/a (resp. y=0/b, z=0/c) face pair contribute equally, since fixing
        the normal coordinate at 0 or its full extent squares away the sign
        difference -- compute one member of each pair and double it."""
        a, b, c = self.a, self.b, self.c
        m, n, p = self._m, self._n, self._p
        amp2 = abs(self.amplitude) ** 2
        k_c2 = self._k_c2
        kx, ky, kz = m * np.pi / a, n * np.pi / b, p * np.pi / c

        Ix_cos, Ix_sin = _num.cos2_integral(m, a), _num.sin2_integral(m, a)
        Iy_cos, Iy_sin = _num.cos2_integral(n, b), _num.sin2_integral(n, b)
        Iz_cos, Iz_sin = _num.cos2_integral(p, c), _num.sin2_integral(p, c)

        if self.mode.kind == "TE":
            # Hz ~ cos(kx x) cos(ky y) sin(kz z)
            # Hx ~ sin(kx x) cos(ky y) cos(kz z), prefactor kx*kz/k_c2
            # Hy ~ cos(kx x) sin(ky y) cos(kz z), prefactor ky*kz/k_c2
            pref_x = amp2 * (kx * kz / k_c2) ** 2
            pref_y = amp2 * (ky * kz / k_c2) ** 2
            pref_z = amp2

            # x=0 face: Hx=0 (sin(0)=0); tangential = Hy, Hz
            int_x_face = pref_y * Iy_sin * Iz_cos + pref_z * Iy_cos * Iz_sin
            # y=0 face: Hy=0 (sin(0)=0); tangential = Hx, Hz
            int_y_face = pref_x * Ix_sin * Iz_cos + pref_z * Ix_cos * Iz_sin
            # z=0 face: Hz=0 (sin(0)=0); tangential = Hx, Hy
            int_z_face = pref_x * Ix_sin * Iy_cos + pref_y * Ix_cos * Iy_sin
        else:
            # Hx ~ sin(kx x) cos(ky y) cos(kz z), prefactor omega*eps*ky/k_c2
            # Hy ~ cos(kx x) sin(ky y) cos(kz z), prefactor omega*eps*kx/k_c2
            pref_x = amp2 * (self._omega * self.eps * ky / k_c2) ** 2
            pref_y = amp2 * (self._omega * self.eps * kx / k_c2) ** 2

            # x=0 face: Hx=0; tangential = Hy only (Hz=0 identically for TM)
            int_x_face = pref_y * Iy_sin * Iz_cos
            # y=0 face: Hy=0; tangential = Hx only
            int_y_face = pref_x * Ix_sin * Iz_cos
            # z=0 face: neither Hx nor Hy vanishes at z=0; tangential = Hx, Hy
            int_z_face = pref_x * Ix_sin * Iy_cos + pref_y * Ix_cos * Iy_sin

        p_loss = Rs / 2.0 * 2.0 * (int_x_face + int_y_face + int_z_face)
        w = self.total_stored_energy()
        return self._omega * w / p_loss

    def bounding_box(self) -> tuple[Array, Array]:
        return np.zeros(3), np.array([self.a, self.b, self.c])

    def contains(self, r: Array) -> Array:
        r = np.atleast_2d(np.asarray(r, dtype=float))
        x, y, z = r[..., 0], r[..., 1], r[..., 2]
        mask = (
            (0 <= x) & (x <= self.a)
            & (0 <= y) & (y <= self.b)
            & (0 <= z) & (z <= self.c)
        )
        return mask


class CylindricalCavity(CavityMode):
    """TM_npq / TE_npq modes of a radius-a, length-d circular cavity, axis
    along z, end caps at z=0 and z=d, per docs/module1_cavity_equations.md
    Section 2. `mode.indices` = (n, p, q): n = azimuthal, p = radial (Bessel
    zero) index, q = axial index. Default mode is TM_010 (dominant for
    d/a <~ 2, Section 2.3).
    """

    _RHO_TOL_FACTOR = 1e-9

    def __init__(
        self,
        radius: float,
        length: float,
        mode: ModeIndex | None = None,
        amplitude: complex = 1.0,
        eps: float = constants.epsilon_0,
        mu: float = constants.mu_0,
    ) -> None:
        if mode is None:
            mode = ModeIndex("TM", (0, 1, 0))
        if mode.kind not in ("TE", "TM"):
            raise ValueError(f"mode.kind must be 'TE' or 'TM', got {mode.kind!r}")
        n, p, q = mode.indices
        if n < 0 or p < 1:
            raise ValueError("n >= 0 and p >= 1 required")
        if mode.kind == "TM" and q < 0:
            raise ValueError("TM_npq requires q >= 0")
        if mode.kind == "TE" and q < 1:
            raise ValueError("TE_npq requires q >= 1")

        self.a = radius
        self.d = length
        self.mode = mode
        self.amplitude = amplitude
        self.eps = eps
        self.mu = mu
        self._n, self._p, self._q = n, p, q

        self._X = (
            _num.bessel_zero_tm(n, p) if mode.kind == "TM" else _num.bessel_zero_te(n, p)
        )
        self._kc = self._X / radius
        self._k_c2 = self._kc**2
        self._f0 = (
            1.0
            / (2.0 * np.pi * np.sqrt(eps * mu))
            * np.sqrt(self._k_c2 + (q * np.pi / length) ** 2)
        )
        self._omega = 2.0 * np.pi * self._f0

        Phi, grad_t_Phi, grad_t_dPhi_dz = self._make_mode_function()
        self._E, self._H = _num.tez_tmz_fields(
            mode.kind, Phi, grad_t_Phi, grad_t_dPhi_dz, self._k_c2, self._omega, eps, mu
        )
        self._I_deriv, self._I_over_rho = self._radial_quadratures()

    def _make_mode_function(self) -> _FieldFuncs:
        a, d = self.a, self.d
        n, q = self._n, self._q
        kc = self._kc
        amp = self.amplitude
        rho_tol = self._RHO_TOL_FACTOR * a

        def cart_to_cyl(r: Array) -> tuple[Array, Array, Array]:
            x, y, z = r[..., 0], r[..., 1], r[..., 2]
            rho = np.hypot(x, y)
            phi = np.arctan2(y, x)
            return rho, phi, z

        def safe_bessel_over_rho(rho: Array) -> Array:
            """J_n(kc*rho)/rho, regularized at rho=0 (Section 2.8 step 3):
            the n/rho * J_n(kc*rho) term is 0/0 there for n>=1, but the true
            limit is finite (kc/2 for n=1, 0 otherwise) -- avoid NaN."""
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = special.jv(n, kc * rho) / rho
            limit = kc / 2.0 if n == 1 else 0.0
            return np.where(rho < rho_tol, limit, ratio)

        if self.mode.kind == "TM":
            def Z(z: Array) -> Array:
                return np.cos(q * np.pi * z / d)

            def dZ_dz(z: Array) -> Array:
                return -(q * np.pi / d) * np.sin(q * np.pi * z / d)
        else:
            def Z(z: Array) -> Array:
                return np.sin(q * np.pi * z / d)

            def dZ_dz(z: Array) -> Array:
                return (q * np.pi / d) * np.cos(q * np.pi * z / d)

        def Phi(r: Array) -> Array:
            rho, phi, z = cart_to_cyl(r)
            return amp * special.jv(n, kc * rho) * np.cos(n * phi) * Z(z)

        def _grad_t(r: Array, z_envelope: Callable[[Array], Array]) -> Array:
            rho, phi, z = cart_to_cyl(r)
            radial = amp * kc * special.jvp(n, kc * rho, 1) * np.cos(n * phi) * z_envelope(z)
            azimuthal = amp * (-n) * safe_bessel_over_rho(rho) * np.sin(n * phi) * z_envelope(z)
            cosphi, sinphi = np.cos(phi), np.sin(phi)
            out = np.zeros(r.shape, dtype=complex)
            out[..., 0] = radial * cosphi - azimuthal * sinphi
            out[..., 1] = radial * sinphi + azimuthal * cosphi
            return out

        def grad_t_Phi(r: Array) -> Array:
            return _grad_t(r, Z)

        def grad_t_dPhi_dz(r: Array) -> Array:
            return _grad_t(r, dZ_dz)

        return Phi, grad_t_Phi, grad_t_dPhi_dz

    def _radial_quadratures(self) -> tuple[float, float]:
        """Section 2.5: closed form for the E_z (or H_z) term via the Bessel
        normalization identity (numerics.bessel_tm/te_radial_integral); the
        J_n' and J_n/rho terms have no such elementary closed form, so they
        are cached 1-D quadratures, computed once here."""
        a, n, kc = self.a, self._n, self._kc
        I_deriv, _ = integrate.quad(lambda rho: rho * special.jvp(n, kc * rho, 1) ** 2, 0, a)
        if n == 0:
            I_over_rho = 0.0
        else:
            I_over_rho, _ = integrate.quad(lambda rho: special.jv(n, kc * rho) ** 2 / rho, 0, a)
        return I_deriv, I_over_rho

    def E(self, r: Array) -> Array:
        return self._E(r)

    def H(self, r: Array) -> Array:
        return self._H(r)

    @property
    def f0(self) -> float:
        return self._f0

    @property
    def epsilon_bg(self) -> complex:
        return self.eps

    @property
    def mu_bg(self) -> complex:
        return self.mu

    def stored_energy_density(self, r: Array) -> tuple[Array, Array]:
        e = self.E(r)
        h = self.H(r)
        w_e = self.eps / 2.0 * np.sum(np.abs(e) ** 2, axis=-1)
        w_m = self.mu / 2.0 * np.sum(np.abs(h) ** 2, axis=-1)
        return w_e, w_m

    def total_stored_energy(self) -> float:
        """Closed form (Section 2.5): phi- and z-integrals are elementary
        (Section 0.4 / phi_cos2_integral), the E_z radial integral is the
        Bessel normalization identity, and the E_rho/E_phi radial integrals
        reuse the cached quadratures from __init__."""
        n, q = self._n, self._q
        d = self.d
        amp2 = abs(self.amplitude) ** 2
        kc, k_c2 = self._kc, self._k_c2
        phi_cos = _num.phi_cos2_integral(n)
        phi_sin = _num.phi_sin2_integral(n)
        Iz_cos = _num.cos2_integral(q, d)
        Iz_sin = _num.sin2_integral(q, d)

        if self.mode.kind == "TM":
            I_bessel = _num.bessel_tm_radial_integral(n, self._X, self.a)
            int_Ez2 = amp2 * I_bessel * phi_cos * Iz_cos
            int_Erho2 = amp2 * (q * np.pi / (d * kc)) ** 2 * self._I_deriv * phi_cos * Iz_sin
            int_Ephi2 = (
                amp2 * (n * q * np.pi / (d * k_c2)) ** 2 * self._I_over_rho * phi_sin * Iz_sin
            )
            int_E2 = int_Ez2 + int_Erho2 + int_Ephi2
        else:
            omega_mu = self._omega * self.mu
            int_Erho2 = amp2 * (omega_mu * n / k_c2) ** 2 * self._I_over_rho * phi_sin * Iz_sin
            int_Ephi2 = amp2 * (omega_mu / kc) ** 2 * self._I_deriv * phi_cos * Iz_sin
            int_E2 = int_Erho2 + int_Ephi2

        return self.eps / 2.0 * int_E2

    def Q_wall(self, Rs: float) -> float:
        """Closed form (Section 2.6): curved wall (rho=a) plus two end caps.
        The curved-wall integral uses J_n'(X_np) = -J_(n+1)(X_np) at a TM
        zero (Section 0.4), or a direct J_n(X'_np) evaluation for TE -- no
        quadrature needed there; the end caps reuse the cached radial
        quadratures from __init__."""
        n, q = self._n, self._q
        a, d = self.a, self.d
        amp2 = abs(self.amplitude) ** 2
        kc, k_c2 = self._kc, self._k_c2
        phi_cos = _num.phi_cos2_integral(n)
        phi_sin = _num.phi_sin2_integral(n)
        Iz_cos = _num.cos2_integral(q, d)
        Iz_sin = _num.sin2_integral(q, d)

        if self.mode.kind == "TM":
            omega_eps = self._omega * self.eps
            J_np1 = special.jv(n + 1, self._X)
            Hphi_a2 = amp2 * (omega_eps / kc) ** 2 * J_np1**2
            curved = a * Hphi_a2 * phi_cos * Iz_cos

            end_cap = (
                amp2 * (omega_eps * n / k_c2) ** 2 * self._I_over_rho * phi_sin
                + amp2 * (omega_eps / kc) ** 2 * self._I_deriv * phi_cos
            )
            p_loss = Rs / 2.0 * (curved + 2.0 * end_cap)
        else:
            pref_rho = amp2 * (q * np.pi / (d * kc)) ** 2
            pref_phi = amp2 * (n * q * np.pi / (d * k_c2)) ** 2
            J_n = special.jv(n, self._X)
            Hphi_a2 = pref_phi * (J_n / a) ** 2
            Hz_a2 = amp2 * J_n**2
            curved = a * (Hphi_a2 * phi_sin * Iz_cos + Hz_a2 * phi_cos * Iz_sin)

            end_cap = pref_rho * self._I_deriv * phi_cos + pref_phi * self._I_over_rho * phi_sin
            p_loss = Rs / 2.0 * (curved + 2.0 * end_cap)

        w = self.total_stored_energy()
        return self._omega * w / p_loss

    def bounding_box(self) -> tuple[Array, Array]:
        return np.array([-self.a, -self.a, 0.0]), np.array([self.a, self.a, self.d])

    def contains(self, r: Array) -> Array:
        r = np.atleast_2d(np.asarray(r, dtype=float))
        x, y, z = r[..., 0], r[..., 1], r[..., 2]
        rho = np.hypot(x, y)
        return (rho <= self.a) & (0 <= z) & (z <= self.d)


class CoaxialCavity(CavityMode):
    """TEM standing-wave modes of a shorted coaxial line (inner conductor
    radius r_inner, outer radius r_outer, length L), per
    docs/module1_cavity_equations.md Section 3.

    Scope decision (Section 3.1): only the TEM family (standing-wave
    transmission-line resonances) is implemented. Coaxial lines also support
    higher-order hybrid TE/TM modes, but these have cutoff frequencies well
    above the TEM family and are never the intended operating mode for a
    perturbation-measurement fixture -- exactly analogous to why ordinary
    coax cable is operated below its first higher-order-mode cutoff. This is
    a deliberate scope boundary, not an oversight.

    `mode.indices` = (q,), the axial standing-wave index (q = 1, 2, 3, ...).
    """

    def __init__(
        self,
        r_inner: float,
        r_outer: float,
        length: float,
        mode: ModeIndex | None = None,
        amplitude: complex = 1.0,
        eps: float = constants.epsilon_0,
        mu: float = constants.mu_0,
    ) -> None:
        if mode is None:
            mode = ModeIndex("TEM", (1,))
        if mode.kind != "TEM":
            raise ValueError(f"mode.kind must be 'TEM', got {mode.kind!r}")
        (q,) = mode.indices
        if q < 1:
            raise ValueError("CoaxialCavity requires q >= 1")

        self.a, self.b, self.L = r_inner, r_outer, length
        self.mode = mode
        self.amplitude = amplitude
        self.eps = eps
        self.mu = mu
        self._q = q

        self._eta = np.sqrt(mu / eps)
        self._ln_ba = np.log(r_outer / r_inner)
        self._Z0 = self._eta / (2.0 * np.pi) * self._ln_ba
        self._f0 = q / (2.0 * length * np.sqrt(eps * mu))
        self._omega = 2.0 * np.pi * self._f0
        self._beta = q * np.pi / length

    def _V(self, z: Array) -> Array:
        return self.amplitude * np.sin(self._beta * z)

    def _I(self, z: Array) -> Array:
        return 1j * (self.amplitude / self._Z0) * np.cos(self._beta * z)

    def E(self, r: Array) -> Array:
        r_arr = np.asarray(r, dtype=float)
        orig_shape = r_arr.shape
        pts = np.atleast_2d(r_arr)
        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
        rho = np.hypot(x, y)
        phi = np.arctan2(y, x)
        E_rho = self._V(z) / (rho * self._ln_ba)
        out = np.zeros(pts.shape, dtype=complex)
        out[..., 0] = E_rho * np.cos(phi)
        out[..., 1] = E_rho * np.sin(phi)
        return out.reshape(orig_shape)

    def H(self, r: Array) -> Array:
        r_arr = np.asarray(r, dtype=float)
        orig_shape = r_arr.shape
        pts = np.atleast_2d(r_arr)
        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
        rho = np.hypot(x, y)
        phi = np.arctan2(y, x)
        H_phi = self._I(z) / (2.0 * np.pi * rho)
        out = np.zeros(pts.shape, dtype=complex)
        out[..., 0] = -H_phi * np.sin(phi)
        out[..., 1] = H_phi * np.cos(phi)
        return out.reshape(orig_shape)

    @property
    def f0(self) -> float:
        return self._f0

    @property
    def epsilon_bg(self) -> complex:
        return self.eps

    @property
    def mu_bg(self) -> complex:
        return self.mu

    def stored_energy_density(self, r: Array) -> tuple[Array, Array]:
        e = self.E(r)
        h = self.H(r)
        w_e = self.eps / 2.0 * np.sum(np.abs(e) ** 2, axis=-1)
        w_m = self.mu / 2.0 * np.sum(np.abs(h) ** 2, axis=-1)
        return w_e, w_m

    def total_stored_energy(self) -> float:
        """Closed form (Section 3.4), using integral_a^b drho/rho = ln(b/a)
        and the trig identity (Section 0.4) for the z-integral (= L/2 since
        q >= 1).

        Note: the doc's Section 3.4 states this proportional to ln(b/a), but
        E_rho = V(z)/(rho*ln(b/a)) makes the volume integral scale as
        1/ln(b/a) (energy diverges, correctly, as the conductors merge and
        ln(b/a) -> 0 for fixed voltage) -- verified against brute-force
        quadrature in tests/test_cavity_coaxial.py."""
        amp2 = abs(self.amplitude) ** 2
        return (
            self.eps
            / 2.0
            * (2.0 * np.pi / self._ln_ba)
            * amp2
            * _num.sin2_integral(self._q, self.L)
        )

    def Q_wall(self, Rs: float) -> float:
        """Closed form (Section 3.5): attenuation-constant route. Drops the
        end-cap surface-loss contribution as negligible relative to the long
        side walls for any practically-proportioned resonator -- an explicit
        approximation, not a silent omission."""
        alpha_c = Rs / (2.0 * self._eta * self._ln_ba) * (1.0 / self.a + 1.0 / self.b)
        return self._beta / (2.0 * alpha_c)

    def bounding_box(self) -> tuple[Array, Array]:
        return np.array([-self.b, -self.b, 0.0]), np.array([self.b, self.b, self.L])

    def contains(self, r: Array) -> Array:
        r = np.atleast_2d(np.asarray(r, dtype=float))
        x, y, z = r[..., 0], r[..., 1], r[..., 2]
        rho = np.hypot(x, y)
        return (self.a <= rho) & (rho <= self.b) & (0 <= z) & (z <= self.L)


class ToroidalCavity(CavityMode):
    """TM_npl / TE_npl modes of a torus (major radius R, minor/tube radius
    a), per docs/toroidal_cavity_plan.md.

    Scope decision: **thin-torus approximation only** (a << R), not an exact
    torus solution -- exactly analogous to `CoaxialCavity`'s TEM-only scope
    (Section 3.1 there): an exact torus eigenproblem has no closed form (it
    doesn't separate in any standard orthogonal coordinate system the way
    the rectangular/cylindrical/coaxial problems do), so this class instead
    treats the tube, at each point around the ring, as a locally straight
    circular waveguide of radius `a` carrying a standing-wave pattern
    `cos(l*theta)`/`sin(l*theta)` around the ring (`theta` = angle around
    the major circle) in place of the cylindrical cavity's `cos`/`sin(q*pi
    z/d)` standing wave along a straight axis. This drops O(a/R) curvature
    ("connection") terms that a rigorous toroidal-coordinate treatment would
    include -- deliberate, and quantified directly (not assumed) by this
    class's own curl-residual convergence test and its FDTD cross-check
    (`contains()` below is exact, so FDTD can simulate the true geometry
    independent of this approximation).

    `mode.indices` = (n, p, l): n = azimuthal index around the tube
    cross-section (same role as `CylindricalCavity`'s n), p = radial
    (Bessel-zero) index (same role as `CylindricalCavity`'s p), l = the
    periodic standing-wave index around the big ring (plays the role of
    `CylindricalCavity`'s axial index q, but with wavenumber l/R -- a
    periodic ring has no end walls to set a q*pi/d-style quantization
    against). Validity mirrors `CylindricalCavity` exactly with l standing
    in for q: TM_npl requires l >= 0, TE_npl requires l >= 1 -- this
    specific choice (not a different indexing scheme) is what lets
    `ritz.py`'s `nearest_basis_modes` enumerate/construct/reject candidate
    ToroidalCavity modes with zero changes to `ritz.py` itself, the same way
    it already does for `CylindricalCavity`.

    Coordinates: `theta = atan2(y, x)` (position around the big ring),
    `rho_big = hypot(x, y)`, then local tube coordinates
    `rho_local = hypot(rho_big - R, z)`, `phi_local = atan2(z, rho_big - R)`.
    This mapping is exact and used unmodified for `contains()`/
    `bounding_box()` regardless of the field approximation above.
    """

    _RHO_TOL_FACTOR = 1e-9

    def __init__(
        self,
        R: float,
        a: float,
        mode: ModeIndex | None = None,
        amplitude: complex = 1.0,
        eps: float = constants.epsilon_0,
        mu: float = constants.mu_0,
    ) -> None:
        if a <= 0 or R <= a:
            raise ValueError(
                "ToroidalCavity requires 0 < a < R (thin-torus regime, "
                "non-self-intersecting geometry)"
            )
        if mode is None:
            mode = ModeIndex("TM", (0, 1, 1))
        if mode.kind not in ("TE", "TM"):
            raise ValueError(f"mode.kind must be 'TE' or 'TM', got {mode.kind!r}")
        n, p, l = mode.indices
        if n < 0 or p < 1:
            raise ValueError("n >= 0 and p >= 1 required")
        if mode.kind == "TM" and l < 0:
            raise ValueError("TM_npl requires l >= 0")
        if mode.kind == "TE" and l < 1:
            raise ValueError("TE_npl requires l >= 1")

        self.R = R
        self.a = a
        self.mode = mode
        self.amplitude = amplitude
        self.eps = eps
        self.mu = mu
        self._n, self._p, self._l = n, p, l

        self._X = (
            _num.bessel_zero_tm(n, p) if mode.kind == "TM" else _num.bessel_zero_te(n, p)
        )
        self._kc = self._X / a
        self._k_c2 = self._kc**2
        self._f0 = (
            1.0
            / (2.0 * np.pi * np.sqrt(eps * mu))
            * np.sqrt(self._k_c2 + (l / R) ** 2)
        )
        self._omega = 2.0 * np.pi * self._f0

        Phi, grad_t_Phi, grad_t_dPhi_ds = self._make_local_mode_function()
        self._E_local, self._H_local = _num.tez_tmz_fields(
            mode.kind, Phi, grad_t_Phi, grad_t_dPhi_ds, self._k_c2, self._omega, eps, mu
        )
        self._I_deriv, self._I_over_rho = self._radial_quadratures()

    def _make_local_mode_function(self) -> _FieldFuncs:
        """Builds (Phi, grad_t_Phi, grad_t_dPhi_ds) in a fictitious *local*
        frame, fed to `numerics.tez_tmz_fields` exactly as `CylindricalCavity`
        does -- component 0/1 of the input play the role of that class's
        (x,y) [i.e. its own (rho,phi) pair], component 2 plays the role of
        its z, with `theta` (not literal length) as the raw value there.
        The role-substitution is (rho_local, phi_local, theta) <-> that
        class's (rho, phi, z): same Bessel cross-section profile
        `J_n(kc*rho_local)*cos(n*phi_local)`, replacing its `cos`/`sin(q*pi
        z/d)` envelope with `cos`/`sin(l*theta)` and its wavenumber `q*pi/d`
        with `l/R` (Section 0 of docs/toroidal_cavity_plan.md) -- `dZ_ds`
        below is the theta-envelope's derivative *with respect to arc
        length* `R*theta`, not raw `theta`, which is what keeps this
        dimensionally consistent with `k_c2` (1/length^2) exactly the way
        `CylindricalCavity`'s own `dZ/dz` is (z already being a length).
        The public `E`/`H` below feed this a synthetic (xi, eta, theta)
        triple (xi = rho_big - R, eta = z) and then rotate the 3-vector this
        produces from that local frame into true Cartesian -- see `E`/`H`."""
        a = self.a
        n, l, R = self._n, self._l, self.R
        kc = self._kc
        amp = self.amplitude
        rho_tol = self._RHO_TOL_FACTOR * a

        def local_to_polar(r_fake: Array) -> tuple[Array, Array, Array]:
            xi, eta, theta = r_fake[..., 0], r_fake[..., 1], r_fake[..., 2]
            rho_local = np.hypot(xi, eta)
            phi_local = np.arctan2(eta, xi)
            return rho_local, phi_local, theta

        def safe_bessel_over_rho(rho: Array) -> Array:
            """J_n(kc*rho)/rho, regularized at rho=0 -- the tube's own core
            circle -- identical reasoning/limit to `CylindricalCavity`'s
            on-axis regularization (module1 doc Section 2.8 step 3)."""
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = special.jv(n, kc * rho) / rho
            limit = kc / 2.0 if n == 1 else 0.0
            return np.where(rho < rho_tol, limit, ratio)

        if self.mode.kind == "TM":
            def Z(theta: Array) -> Array:
                return np.cos(l * theta)

            def dZ_ds(theta: Array) -> Array:
                return -(l / R) * np.sin(l * theta)
        else:
            def Z(theta: Array) -> Array:
                return np.sin(l * theta)

            def dZ_ds(theta: Array) -> Array:
                return (l / R) * np.cos(l * theta)

        def Phi(r_fake: Array) -> Array:
            rho_local, phi_local, theta = local_to_polar(r_fake)
            return amp * special.jv(n, kc * rho_local) * np.cos(n * phi_local) * Z(theta)

        def _grad_t(r_fake: Array, envelope: Callable[[Array], Array]) -> Array:
            rho_local, phi_local, theta = local_to_polar(r_fake)
            radial = amp * kc * special.jvp(n, kc * rho_local, 1) * np.cos(n * phi_local) * envelope(theta)
            azimuthal = amp * (-n) * safe_bessel_over_rho(rho_local) * np.sin(n * phi_local) * envelope(theta)
            cosphi, sinphi = np.cos(phi_local), np.sin(phi_local)
            out = np.zeros(r_fake.shape, dtype=complex)
            out[..., 0] = radial * cosphi - azimuthal * sinphi
            out[..., 1] = radial * sinphi + azimuthal * cosphi
            return out

        def grad_t_Phi(r_fake: Array) -> Array:
            return _grad_t(r_fake, Z)

        def grad_t_dPhi_ds(r_fake: Array) -> Array:
            return _grad_t(r_fake, dZ_ds)

        return Phi, grad_t_Phi, grad_t_dPhi_ds

    def _radial_quadratures(self) -> tuple[float, float]:
        """Identical role/formula to `CylindricalCavity`'s own (module1 doc
        Section 2.5): rho_local ranges over [0, a] exactly as that class's
        rho does, so both cached 1-D quadratures reuse the same integrands."""
        a, n, kc = self.a, self._n, self._kc
        I_deriv, _ = integrate.quad(lambda rho: rho * special.jvp(n, kc * rho, 1) ** 2, 0, a)
        if n == 0:
            I_over_rho = 0.0
        else:
            I_over_rho, _ = integrate.quad(lambda rho: special.jv(n, kc * rho) ** 2 / rho, 0, a)
        return I_deriv, I_over_rho

    def _to_local(self, r: Array) -> tuple[Array, Array, tuple[int, ...]]:
        r_arr = np.asarray(r, dtype=float)
        orig_shape = r_arr.shape
        pts = np.atleast_2d(r_arr)
        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
        theta = np.arctan2(y, x)
        rho_big = np.hypot(x, y)
        xi = rho_big - self.R
        eta = z
        r_fake = np.stack([xi, eta, theta], axis=-1)
        return r_fake, theta, orig_shape

    def _rotate_to_cartesian(self, local_vec: Array, theta: Array) -> Array:
        """Local-frame components (index 0/1/2 <-> xi/eta/theta directions)
        -> true Cartesian. The local frame is (e_rho(theta), e_z,
        e_theta(theta)) with e_rho=(cos,sin,0), e_theta=(-sin,cos,0) --
        `e_rho x e_z = -e_theta`, so the axial (index-2) component must map
        to *minus* e_theta to keep this a right-handed triple (index0 x
        index1 = index2), the same convention `tez_tmz_fields`'s internal
        `zhat_cross` assumes; verified directly via the curl-residual
        convergence test, not just asserted -- getting this sign wrong
        produces curl E ~= +j*omega*mu*H instead of the required -j*omega*
        mu*H, caught immediately as a large (~2x), a/R-independent residual
        rather than one that shrinks with a/R."""
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        out = np.zeros(local_vec.shape, dtype=complex)
        out[..., 0] = local_vec[..., 0] * cos_t + local_vec[..., 2] * sin_t
        out[..., 1] = local_vec[..., 0] * sin_t - local_vec[..., 2] * cos_t
        out[..., 2] = local_vec[..., 1]
        return out

    def E(self, r: Array) -> Array:
        r_fake, theta, orig_shape = self._to_local(r)
        return self._rotate_to_cartesian(self._E_local(r_fake), theta).reshape(orig_shape)

    def H(self, r: Array) -> Array:
        r_fake, theta, orig_shape = self._to_local(r)
        return self._rotate_to_cartesian(self._H_local(r_fake), theta).reshape(orig_shape)

    @property
    def f0(self) -> float:
        return self._f0

    @property
    def epsilon_bg(self) -> complex:
        return self.eps

    @property
    def mu_bg(self) -> complex:
        return self.mu

    def stored_energy_density(self, r: Array) -> tuple[Array, Array]:
        e = self.E(r)
        h = self.H(r)
        w_e = self.eps / 2.0 * np.sum(np.abs(e) ** 2, axis=-1)
        w_m = self.mu / 2.0 * np.sum(np.abs(h) ** 2, axis=-1)
        return w_e, w_m

    def total_stored_energy(self) -> float:
        """Closed form: same cross-section (radial/phi) closed forms as
        `CylindricalCavity.total_stored_energy` (module1 doc Section 2.5),
        times the periodic theta-integral (`numerics.periodic_cos2_integral`/
        `periodic_sin2_integral`) in place of that class's `cos2_integral`/
        `sin2_integral`, times an *explicit extra factor of R*: the
        thin-limit volume element is `dV ~= R*rho_local*d(rho_local)*
        d(phi_local)*d(theta)`, and unlike `CylindricalCavity`'s z-integral
        (already a true-length integral, R-free), the periodic theta-integral
        is dimensionless (an integral over a bare angle) and needs this
        multiplier to become an actual arc-length volume element -- verified
        directly against brute-force Cartesian quadrature
        (test_toroidal_total_stored_energy_matches_brute_force_quadrature),
        not just asserted, per CLAUDE.md's Conventions warning about this
        exact class of silent scale bug."""
        n, l, R = self._n, self._l, self.R
        amp2 = abs(self.amplitude) ** 2
        kc, k_c2 = self._kc, self._k_c2
        phi_cos = _num.phi_cos2_integral(n)
        phi_sin = _num.phi_sin2_integral(n)
        Th_cos = _num.periodic_cos2_integral(l)
        Th_sin = _num.periodic_sin2_integral(l)

        if self.mode.kind == "TM":
            I_bessel = _num.bessel_tm_radial_integral(n, self._X, self.a)
            int_Ez2 = amp2 * I_bessel * phi_cos * Th_cos
            int_Erho2 = amp2 * (l / (R * kc)) ** 2 * self._I_deriv * phi_cos * Th_sin
            int_Ephi2 = amp2 * (n * l / (R * k_c2)) ** 2 * self._I_over_rho * phi_sin * Th_sin
            int_E2 = int_Ez2 + int_Erho2 + int_Ephi2
        else:
            omega_mu = self._omega * self.mu
            int_Erho2 = amp2 * (omega_mu * n / k_c2) ** 2 * self._I_over_rho * phi_sin * Th_sin
            int_Ephi2 = amp2 * (omega_mu / kc) ** 2 * self._I_deriv * phi_cos * Th_sin
            int_E2 = int_Erho2 + int_Ephi2

        return self.eps / 2.0 * R * int_E2

    def Q_wall(self, Rs: float) -> float:
        """Closed form: a *single* curved-surface term (module1 doc's
        general master formula, Section 0.3) -- unlike `CylindricalCavity`,
        there are no end caps (the tube is a closed ring). Surface element
        `dA ~= a*R*d(phi_local)*d(theta)` -- same explicit-extra-R-factor
        reasoning as `total_stored_energy` above (the curved-wall integral
        reuses `CylindricalCavity`'s own `a*d(phi)*d(z)` surface-element
        structure, and needs the same R multiplier for the same
        dimensionless-periodic-integral reason); verified against
        brute-force Cartesian surface quadrature
        (test_toroidal_q_wall_matches_numerical_surface_integral), not just
        asserted."""
        n, l, R, a = self._n, self._l, self.R, self.a
        amp2 = abs(self.amplitude) ** 2
        kc, k_c2 = self._kc, self._k_c2
        phi_cos = _num.phi_cos2_integral(n)
        phi_sin = _num.phi_sin2_integral(n)
        Th_cos = _num.periodic_cos2_integral(l)
        Th_sin = _num.periodic_sin2_integral(l)

        if self.mode.kind == "TM":
            omega_eps = self._omega * self.eps
            J_np1 = special.jv(n + 1, self._X)
            Hphi_a2 = amp2 * (omega_eps / kc) ** 2 * J_np1**2
            curved = a * R * Hphi_a2 * phi_cos * Th_cos
        else:
            pref_phi = amp2 * (n * l / (R * k_c2)) ** 2
            J_n = special.jv(n, self._X)
            Hphi_a2 = pref_phi * (J_n / a) ** 2
            Hz_a2 = amp2 * J_n**2
            curved = a * R * (Hphi_a2 * phi_sin * Th_cos + Hz_a2 * phi_cos * Th_sin)

        p_loss = Rs / 2.0 * curved
        w = self.total_stored_energy()
        return self._omega * w / p_loss

    def bounding_box(self) -> tuple[Array, Array]:
        R, a = self.R, self.a
        return np.array([-(R + a), -(R + a), -a]), np.array([R + a, R + a, a])

    def contains(self, r: Array) -> Array:
        r = np.atleast_2d(np.asarray(r, dtype=float))
        x, y, z = r[..., 0], r[..., 1], r[..., 2]
        rho_big = np.hypot(x, y)
        return (rho_big - self.R) ** 2 + z**2 <= self.a**2
