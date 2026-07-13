"""Module 3 -- Geometry + Material.

`SampleRegion` (`Sphere`, `Cylinder`, `Slab`), `Material`, `Sample`, and the
depolarization-factor formulas, per docs/module3_sample_equations.md. See
that doc's Section 0 for why `shape_kind` is purely geometric while field
alignment is handled separately in `Sample.depolarization_factor`.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np

Array = np.ndarray


class SampleRegion(ABC):
    """Geometric region occupied by the sample, in the cavity-local frame.

    Purely geometric -- never holds or references a field/FieldProvider (see
    module3 doc Section 0). Field alignment is Sample.depolarization_factor's
    concern, not this class's.
    """

    @abstractmethod
    def contains(self, r: Array) -> Array:
        """Boolean mask, r: (N,3) -> (N,)."""

    @abstractmethod
    def volume(self) -> float:
        """Volume of the region, in the same length units as `contains`/
        `quadrature_points`."""

    @abstractmethod
    def quadrature_points(self, n: int) -> tuple[Array, Array]:
        """Returns (points (M,3), weights (M,)) such that
        sum(weights * f(points)) approximates integral_region f dV.
        M need not equal n exactly (e.g. structured grids round up)."""

    @property
    @abstractmethod
    def shape_kind(self) -> str:
        """One of {'sphere', 'thin_rod', 'thin_disk', 'generic'} -- purely
        geometric (aspect-ratio based), never computed from field
        information. 'generic' means: no closed-form depolarization
        correction available, fall back to the point-dipole (small-sample)
        limit (see Sample.depolarization_factor)."""


# ===========================================================================
# Section 1: Material
# ===========================================================================

@dataclass(frozen=True)
class Material:
    """eps = eps' - j*eps'', mu = mu' - j*mu'', eps'',mu'' >= 0 (project-wide
    time convention e^{+j*omega*t}, CLAUDE.md)."""

    eps: complex
    mu: complex

    @property
    def loss_tangent_e(self) -> float:
        if self.eps.real <= 0:
            raise ValueError(f"Re(eps)={self.eps.real!r} must be > 0 to define a loss tangent")
        return -self.eps.imag / self.eps.real

    @property
    def loss_tangent_m(self) -> float:
        if self.mu.real <= 0:
            raise ValueError(f"Re(mu)={self.mu.real!r} must be > 0 to define a loss tangent")
        return -self.mu.imag / self.mu.real

    @classmethod
    def from_loss_tangent(
        cls, eps_r: float, tan_delta_e: float, mu_r: float = 1.0, tan_delta_m: float = 0.0
    ) -> "Material":
        return cls(eps=eps_r * (1 - 1j * tan_delta_e), mu=mu_r * (1 - 1j * tan_delta_m))

    @property
    def is_passive(self) -> bool:
        """Fundamental physical validity check (Section 1.4) -- distinct
        from Module 5's fitting-prior bounds (e.g. eps' >= 1)."""
        return self.eps.imag <= 0 and self.mu.imag <= 0 and self.eps.real > 0 and self.mu.real > 0


# ===========================================================================
# Section 3.1: shared 1-D Gauss-Legendre quadrature
# ===========================================================================

def gauss_legendre(n: int, a: float, b: float) -> tuple[Array, Array]:
    """Nodes/weights for integrating a smooth function over [a, b]."""
    xi, wi = np.polynomial.legendre.leggauss(max(n, 1))
    x = 0.5 * (b - a) * xi + 0.5 * (a + b)
    w = 0.5 * (b - a) * wi
    return x, w


def _periodic_uniform(n: int) -> tuple[Array, Array]:
    """n equally spaced points on [0, 2*pi), uniform weight 2*pi/n each --
    a periodic trapezoid rule, spectrally accurate for smooth periodic
    integrands (Section 3.3)."""
    n = max(n, 1)
    phi = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    w = np.full(n, 2.0 * np.pi / n)
    return phi, w


def _n_per_axis(n: int) -> int:
    """Section 3.6 point-count-splitting rule: n_i ~ n^(1/3) per tensor
    direction, rounded up, never collapsing to zero points."""
    return max(1, math.ceil(n ** (1.0 / 3.0)))


# ===========================================================================
# Section 3.2: local-frame -> cavity-frame rigid transform
# ===========================================================================

def orthonormal_frame(n_hat: Array) -> tuple[Array, Array, Array]:
    """Build an orthonormal basis {e1, e2, n_hat} given a unit vector n_hat
    (the axis or normal of a Cylinder/Slab). The specific in-plane choice of
    e1, e2 is arbitrary -- both consumers are symmetric under rotation about
    n_hat."""
    n_hat = np.asarray(n_hat, dtype=float)
    n_hat = n_hat / np.linalg.norm(n_hat)

    u = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(u, n_hat)) > 0.9:  # near-parallel, degenerate Gram-Schmidt
        u = np.array([0.0, 1.0, 0.0])

    e1 = u - np.dot(u, n_hat) * n_hat
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n_hat, e1)
    return e1, e2, n_hat


def _to_local(r: Array, center: Array, e1: Array, e2: Array, n_hat: Array) -> tuple[Array, Array, Array]:
    rel = np.atleast_2d(r) - center
    return rel @ e1, rel @ e2, rel @ n_hat


def _to_cavity(local_pts: Array, center: Array, e1: Array, e2: Array, n_hat: Array) -> Array:
    """Inverse of _to_local: local (xi1, xi2, xi3) -> cavity-frame (x, y, z)."""
    return (
        center
        + local_pts[..., 0:1] * e1
        + local_pts[..., 1:2] * e2
        + local_pts[..., 2:3] * n_hat
    )


# ===========================================================================
# Section 3.3: Sphere
# ===========================================================================

@dataclass(frozen=True)
class Sphere(SampleRegion):
    center: Array
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", np.asarray(self.center, dtype=float))

    def contains(self, r: Array) -> Array:
        r = np.atleast_2d(r)
        return np.linalg.norm(r - self.center, axis=-1) <= self.radius

    def volume(self) -> float:
        return 4.0 / 3.0 * np.pi * self.radius**3

    def quadrature_points(self, n: int) -> tuple[Array, Array]:
        m = _n_per_axis(n)
        r_nodes, r_w = gauss_legendre(m, 0.0, self.radius)
        r_w = r_w * r_nodes**2  # dV = r^2 dr du dphi
        u_nodes, u_w = gauss_legendre(m, -1.0, 1.0)
        phi_nodes, phi_w = _periodic_uniform(m)

        R, U, PHI = np.meshgrid(r_nodes, u_nodes, phi_nodes, indexing="ij")
        WR, WU, WPHI = np.meshgrid(r_w, u_w, phi_w, indexing="ij")

        z = R * U
        rho = R * np.sqrt(1.0 - U**2)
        x = rho * np.cos(PHI)
        y = rho * np.sin(PHI)
        pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1) + self.center
        w = (WR * WU * WPHI).ravel()
        return pts, w

    @property
    def shape_kind(self) -> str:
        return "sphere"


# ===========================================================================
# Section 3.4: Cylinder
# ===========================================================================

_CYLINDER_ROD_ASPECT = 5.0
_CYLINDER_DISK_ASPECT = 0.2


@dataclass(frozen=True)
class Cylinder(SampleRegion):
    center: Array
    axis: Array
    radius: float
    height: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", np.asarray(self.center, dtype=float))
        axis = np.asarray(self.axis, dtype=float)
        object.__setattr__(self, "axis", axis / np.linalg.norm(axis))

    def contains(self, r: Array) -> Array:
        e1, e2, n_hat = orthonormal_frame(self.axis)
        xi1, xi2, xi3 = _to_local(r, self.center, e1, e2, n_hat)
        rho = np.hypot(xi1, xi2)
        return (rho <= self.radius) & (np.abs(xi3) <= self.height / 2.0)

    def volume(self) -> float:
        return np.pi * self.radius**2 * self.height

    def quadrature_points(self, n: int) -> tuple[Array, Array]:
        m = _n_per_axis(n)
        rho_nodes, rho_w = gauss_legendre(m, 0.0, self.radius)
        rho_w = rho_w * rho_nodes  # dV = rho drho dphi dz
        phi_nodes, phi_w = _periodic_uniform(m)
        z_nodes, z_w = gauss_legendre(m, -self.height / 2.0, self.height / 2.0)

        RHO, PHI, Z = np.meshgrid(rho_nodes, phi_nodes, z_nodes, indexing="ij")
        WR, WP, WZ = np.meshgrid(rho_w, phi_w, z_w, indexing="ij")

        xi1 = RHO * np.cos(PHI)
        xi2 = RHO * np.sin(PHI)
        local_pts = np.stack([xi1.ravel(), xi2.ravel(), Z.ravel()], axis=-1)

        e1, e2, n_hat = orthonormal_frame(self.axis)
        pts = _to_cavity(local_pts, self.center, e1, e2, n_hat)
        w = (WR * WP * WZ).ravel()
        return pts, w

    @property
    def shape_kind(self) -> str:
        aspect = self.height / (2.0 * self.radius)
        if aspect > _CYLINDER_ROD_ASPECT:
            return "thin_rod"
        if aspect < _CYLINDER_DISK_ASPECT:
            return "thin_disk"
        return "generic"


# ===========================================================================
# Section 3.5: Slab
# ===========================================================================

_SLAB_DISK_ASPECT = 0.2


@dataclass(frozen=True)
class Slab(SampleRegion):
    """A rectangular (not circular) disk: a box of side `extent[0]` x
    `extent[1]` in the plane perpendicular to `normal`, and `thickness`
    along `normal`. `extent` gives the two *full* lateral side lengths (not
    half-widths -- `contains`/`volume` halve them internally), measured
    along the arbitrary in-plane axes `orthonormal_frame(normal)` picks;
    since the lateral cross-section only needs to be 'large' for the
    thin-disk depolarization approximation to apply (module3 doc Section
    3.5), which of the two axes `extent[0]` vs. `extent[1]` lands on
    doesn't matter physically."""

    center: Array
    normal: Array
    thickness: float
    extent: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", np.asarray(self.center, dtype=float))
        normal = np.asarray(self.normal, dtype=float)
        object.__setattr__(self, "normal", normal / np.linalg.norm(normal))

    def contains(self, r: Array) -> Array:
        e1, e2, n_hat = orthonormal_frame(self.normal)
        xi1, xi2, xi3 = _to_local(r, self.center, e1, e2, n_hat)
        return (
            (np.abs(xi3) <= self.thickness / 2.0)
            & (np.abs(xi1) <= self.extent[0] / 2.0)
            & (np.abs(xi2) <= self.extent[1] / 2.0)
        )

    def volume(self) -> float:
        return self.thickness * self.extent[0] * self.extent[1]

    def quadrature_points(self, n: int) -> tuple[Array, Array]:
        m = _n_per_axis(n)
        xi1_nodes, xi1_w = gauss_legendre(m, -self.extent[0] / 2.0, self.extent[0] / 2.0)
        xi2_nodes, xi2_w = gauss_legendre(m, -self.extent[1] / 2.0, self.extent[1] / 2.0)
        xi3_nodes, xi3_w = gauss_legendre(m, -self.thickness / 2.0, self.thickness / 2.0)

        XI1, XI2, XI3 = np.meshgrid(xi1_nodes, xi2_nodes, xi3_nodes, indexing="ij")
        W1, W2, W3 = np.meshgrid(xi1_w, xi2_w, xi3_w, indexing="ij")

        local_pts = np.stack([XI1.ravel(), XI2.ravel(), XI3.ravel()], axis=-1)
        e1, e2, n_hat = orthonormal_frame(self.normal)
        pts = _to_cavity(local_pts, self.center, e1, e2, n_hat)
        w = (W1 * W2 * W3).ravel()
        return pts, w

    @property
    def shape_kind(self) -> str:
        aspect = self.thickness / min(self.extent)
        if aspect < _SLAB_DISK_ASPECT:
            return "thin_disk"
        return "generic"


# ===========================================================================
# Section 2: depolarization factor
# ===========================================================================

_N_SPHERE = 1.0 / 3.0
_ANGLE_TOL_DEG = 10.0

# (shape_kind, alignment) -> N. 'aligned' = axial (rod) / normal (disk);
# 'perpendicular' = transverse (rod) / tangential (disk). Section 2.2.
_N_TABLE: dict[tuple[str, str], float] = {
    ("thin_rod", "aligned"): 0.0,
    ("thin_rod", "perpendicular"): 0.5,
    ("thin_disk", "aligned"): 1.0,
    ("thin_disk", "perpendicular"): 0.0,
}


def _depolarization_ratio(N: float, rel_permittivity_or_permeability: complex) -> complex:
    """Master formula (Section 2.1): F_in/F0 = 1 / (1 + N*(chi_r - 1))."""
    return 1.0 / (1.0 + N * (rel_permittivity_or_permeability - 1.0))


def real_field_direction(field_direction: Array) -> Array | None:
    """Extract the real spatial direction of a complex field vector whose
    Cartesian components share one overall complex phase -- true of every
    Module 1 mode (e.g. a TE mode's Ex, Ey both carry the same j*omega*mu
    prefactor, so the field is often *purely imaginary* for a real-valued
    amplitude convention; naively truncating to `.real` would silently
    zero it out). Returns None if the field is (numerically) zero, a
    legitimate degeneracy -- e.g. a sample sitting exactly on a TM_010
    cavity's axis, where H0 vanishes identically by symmetry, not a bug."""
    field_direction = np.asarray(field_direction, dtype=complex)
    idx = int(np.argmax(np.abs(field_direction)))
    if np.abs(field_direction[idx]) < 1e-300:
        return None
    phase = np.angle(field_direction[idx])
    direction = (field_direction * np.exp(-1j * phase)).real
    norm = np.linalg.norm(direction)
    if norm < 1e-300:
        return None
    return direction / norm


def _classify_alignment(n_hat: Array, field_direction: Array) -> Literal["aligned", "perpendicular", "oblique"]:
    """Section 2.3 angle test between the shape's axis/normal and the local
    field direction, with a named tolerance (not a magic number). A
    (numerically) zero field direction has no well-defined alignment --
    falls back to 'oblique' (-> the generic correction), not an error."""
    n_hat = np.asarray(n_hat, dtype=float)
    n_hat = n_hat / np.linalg.norm(n_hat)
    f_hat = real_field_direction(field_direction)
    if f_hat is None:
        return "oblique"

    cos_theta = np.clip(abs(np.dot(n_hat, f_hat)), -1.0, 1.0)
    theta_deg = np.degrees(np.arccos(cos_theta))
    if theta_deg < _ANGLE_TOL_DEG:
        return "aligned"
    if theta_deg > 90.0 - _ANGLE_TOL_DEG:
        return "perpendicular"
    return "oblique"


@dataclass(frozen=True)
class Sample:
    region: SampleRegion
    material: Material

    def depolarization_factor(self, field_type: Literal["E", "H"], field_direction: Array) -> complex:
        """F_in/F0 correction multiplier (Section 2). field_direction is the
        (not necessarily unit) local E0 or H0 vector at the sample's center,
        evaluated and supplied by the caller (Module 4) -- Sample/SampleRegion
        never construct or hold a FieldProvider reference themselves."""
        if field_type not in ("E", "H"):
            raise ValueError(f"field_type must be 'E' or 'H', got {field_type!r}")
        rel = self.material.eps if field_type == "E" else self.material.mu

        shape_kind = self.region.shape_kind
        if shape_kind == "sphere":
            return _depolarization_ratio(_N_SPHERE, rel)
        if shape_kind not in ("thin_rod", "thin_disk"):
            return 1.0 + 0j  # 'generic' fallback (Section 2.4)

        n_hat = getattr(self.region, "axis", None)
        if n_hat is None:
            n_hat = getattr(self.region, "normal", None)
        if n_hat is None:
            return 1.0 + 0j

        alignment = _classify_alignment(n_hat, field_direction)
        if alignment == "oblique":
            return 1.0 + 0j  # misoriented relative to what the closed form assumes
        return _depolarization_ratio(_N_TABLE[(shape_kind, alignment)], rel)
