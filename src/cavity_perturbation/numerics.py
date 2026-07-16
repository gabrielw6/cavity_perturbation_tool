"""Shared numerics for Module 1 (docs/module1_cavity_equations.md, Section 0).

Trig/Bessel integral identities and the TE_z/TM_z field-generation recipe,
implemented once here and reused by every concrete ``CavityMode``.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np
from scipy import special

Array = np.ndarray


# --- Section 2.2: cached Bessel zeros ---------------------------------------

@lru_cache(maxsize=None)
def bessel_zero_tm(n: int, p: int) -> float:
    """p-th zero of J_n (TM family radial eigenvalue X_np), cached."""
    return float(special.jn_zeros(n, p)[-1])


@lru_cache(maxsize=None)
def bessel_zero_te(n: int, p: int) -> float:
    """p-th zero of J_n' (TE family radial eigenvalue X'_np), cached."""
    return float(special.jnp_zeros(n, p)[-1])


# --- Section 0.4: reusable integral identities -----------------------------

def cos2_integral(k: int, L: float) -> float:
    """Integral over [0, L] of cos^2(k*pi*u/L) du."""
    delta_k0 = 1.0 if k == 0 else 0.0
    return L / 2.0 * (1.0 + delta_k0)


def sin2_integral(k: int, L: float) -> float:
    """Integral over [0, L] of sin^2(k*pi*u/L) du."""
    delta_k0 = 1.0 if k == 0 else 0.0
    return L / 2.0 * (1.0 - delta_k0)


def phi_cos2_integral(n: int) -> float:
    """Integral over [0, 2*pi] of cos^2(n*phi) dphi."""
    delta_n0 = 1.0 if n == 0 else 0.0
    return np.pi * (1.0 + delta_n0)


def phi_sin2_integral(n: int) -> float:
    """Integral over [0, 2*pi] of sin^2(n*phi) dphi."""
    delta_n0 = 1.0 if n == 0 else 0.0
    return np.pi * (1.0 - delta_n0)


def periodic_cos2_integral(l: int) -> float:
    """Integral over [0, 2*pi) of cos^2(l*theta) dtheta -- the toroidal
    cavity's periodic-envelope analogue of `cos2_integral`. NOT the same
    formula: `cos2_integral(k, L)` is a *finite-domain* [0, L] integral for a
    standing wave pinned against two end walls (normalized so k=0 gives L,
    not 2L); this is a *full-period* integral of a function that is already
    exactly periodic on [0, 2*pi) with no boundary to pin against. Don't
    substitute one for the other -- same superficial "cos^2 of a linear
    argument" shape, different domain and different l=0 normalization."""
    return 2.0 * np.pi if l == 0 else np.pi


def periodic_sin2_integral(l: int) -> float:
    """Integral over [0, 2*pi) of sin^2(l*theta) dtheta -- see
    `periodic_cos2_integral`'s docstring for why this is not
    `sin2_integral`."""
    return 0.0 if l == 0 else np.pi


def bessel_tm_radial_integral(n: int, X_np: float, a: float) -> float:
    """Integral over [0, a] of rho * J_n(X_np * rho/a)^2 drho.

    X_np must be a zero of J_n (TM family radial eigenvalue).
    """
    return float(a**2 / 2.0 * special.jv(n + 1, X_np) ** 2)


def bessel_te_radial_integral(n: int, Xp_np: float, a: float) -> float:
    """Integral over [0, a] of rho * J_n(Xp_np * rho/a)^2 drho.

    Xp_np must be a zero of J_n' (TE family radial eigenvalue).
    """
    return float(a**2 / 2.0 * (1.0 - n**2 / Xp_np**2) * special.jv(n, Xp_np) ** 2)


# --- Section 0.1: TE_z / TM_z field-generation recipe ----------------------

def zhat_cross(v: Array) -> Array:
    """z_hat cross v, for v of shape (..., 3) with zero z-component."""
    out = np.empty_like(v)
    out[..., 0] = -v[..., 1]
    out[..., 1] = v[..., 0]
    out[..., 2] = 0
    return out


def _as_points(r: Array) -> tuple[Array, tuple[int, ...]]:
    """Normalize r ((3,) or (N,3)) to 2-D (M,3); return it with the original shape."""
    r = np.asarray(r, dtype=float)
    orig_shape = r.shape
    return np.atleast_2d(r), orig_shape


def tez_tmz_fields(
    family: str,
    Phi: Callable[[Array], Array],
    grad_t_Phi: Callable[[Array], Array],
    grad_t_dPhi_dz: Callable[[Array], Array],
    k_c2: float,
    omega: float,
    eps: complex,
    mu: complex,
) -> tuple[Callable[[Array], Array], Callable[[Array], Array]]:
    """Mechanically apply the Module 1 doc Section 0.1 recipe to a scalar mode function.

    Phi, grad_t_Phi, grad_t_dPhi_dz are callables r -> array, taking r of shape
    (M,3) and returning, respectively, shape (M,) and (M,3) (transverse gradient,
    z-component identically zero). amplitude is folded into Phi by the caller.

    Note: the doc's abstract Section 0.1 signs don't survive a literal transcription
    -- TE_z's E_t needs a "+" (not the "-" as written) and TM_z's H_t needs a "-"
    (not the "+" as written) to satisfy curl E = -j*omega*mu*H and
    curl H = j*omega*eps*E. Both signs below were fixed by deriving directly from
    Maxwell's equations and confirmed with the curl-residual regression test
    (Section 1.8 step 2) rather than trusted from the doc text verbatim.
    """
    if family not in ("TE", "TM"):
        raise ValueError(f"family must be 'TE' or 'TM', got {family!r}")

    def _eval(r: Array) -> tuple[Array, Array]:
        pts, orig_shape = _as_points(r)
        phi_val = np.asarray(Phi(pts))
        gphi = np.asarray(grad_t_Phi(pts))
        gdphidz = np.asarray(grad_t_dPhi_dz(pts))

        axial = (gdphidz / k_c2).astype(complex)
        axial[..., 2] = phi_val
        transverse = zhat_cross(gphi).astype(complex)
        return axial.reshape(orig_shape), transverse.reshape(orig_shape)

    if family == "TM":
        def E(r: Array) -> Array:
            axial, _ = _eval(r)
            return axial

        def H(r: Array) -> Array:
            _, transverse = _eval(r)
            return -(1j * omega * eps / k_c2) * transverse
    else:
        def H(r: Array) -> Array:
            axial, _ = _eval(r)
            return axial

        def E(r: Array) -> Array:
            _, transverse = _eval(r)
            return (1j * omega * mu / k_c2) * transverse

    return E, H


# --- curl-residual test harness (Section 1.8 step 2 / Section 5) ----------

def curl_fd(F: Callable[[Array], Array], r: Array, h: float = 1e-6) -> Array:
    """Finite-difference curl of a vector field F at points r, shape (N,3) -> (N,3).

    Used only to regression-test the analytic field recipes against Maxwell's
    equations (curl E = -j*omega*mu*H, curl H = j*omega*eps*E) -- not part of
    the production evaluation path.
    """
    r = np.atleast_2d(np.asarray(r, dtype=float))
    curl = np.zeros((r.shape[0], 3), dtype=complex)
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = h
        plus = F(r + step)
        minus = F(r - step)
        curl_contribution = (plus - minus) / (2 * h)
        # d(F)/d(axis) contributes to curl components other than `axis`
        # curl_x = dFz/dy - dFy/dz, curl_y = dFx/dz - dFz/dx, curl_z = dFy/dx - dFx/dy
        if axis == 0:  # d/dx
            curl[:, 1] -= curl_contribution[:, 2]
            curl[:, 2] += curl_contribution[:, 1]
        elif axis == 1:  # d/dy
            curl[:, 0] += curl_contribution[:, 2]
            curl[:, 2] -= curl_contribution[:, 0]
        else:  # d/dz
            curl[:, 0] -= curl_contribution[:, 1]
            curl[:, 1] += curl_contribution[:, 0]
    return curl
