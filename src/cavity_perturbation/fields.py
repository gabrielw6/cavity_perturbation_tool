"""Module 2 -- Field Provider Abstraction.

Owns no physics of its own -- Module 1 already produced exact, closed-form E,
H. This module turns "E and H evaluated at a point" into "integral over V_s
of |E|^2 dV and |H|^2 dV over an arbitrary sample region," a numerical-
integration problem. See docs/module2_fields_equations.md for the full
derivation and build order.

The `RitzField(FieldProvider)` stub once sketched here has been retired --
see docs/ritz_module_plan.md Section 0 for why a `FieldProvider`-shaped Ritz
class was never the right interface for this project's sample-size-
correction use case; `ritz.py`'s `RitzCorrectedModel` (a `PerturbationModel`
sibling, not a `FieldProvider`) replaces it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Literal, TypeVar

import numpy as np

from .cavity import CavityMode
from .sample import SampleRegion

Array = np.ndarray
_Number = TypeVar("_Number", float, complex)

_CONVERGENCE_TOL = 1e-4
_MAX_DOUBLINGS = 10
_VOLUME_RTOL = 1e-4
_IMAG_REL_TOL = 1e-9
_CROSS_OVERLAP_ATOL_FACTOR = 1e-8  # fraction of the Cauchy-Schwarz bound treated as a noise floor


def hermitian_density(field_values: Array) -> Array:
    """Pointwise |F|^2 = F . F* (Section 1.2), real and >= 0 by construction.

    `np.sum(field_values**2, axis=-1)` (no `abs`) would be a bug: it sums
    F_k^2, which is complex and physically meaningless, not |F_k|^2.
    """
    return np.sum(np.abs(field_values) ** 2, axis=-1)


def converge_by_doubling(
    estimator: Callable[[int], _Number],
    n_start: int,
    tol: float = _CONVERGENCE_TOL,
    max_doublings: int = _MAX_DOUBLINGS,
    atol: float = 0.0,
) -> _Number:
    """Doubling convergence control (Section 1.5): evaluate `estimator` at n
    and 2n, keep doubling until the change is within `tol` relative to the
    current estimate (or within the absolute floor `atol`, whichever is
    looser), up to `max_doublings` doublings, then raise rather than
    silently return an unconverged value. Returns the finer (most recent)
    estimate once converged. `estimator` is independent of what's being
    integrated, so the same helper drives both the E and H integrals without
    duplication -- generic over `float`/`complex` so
    `integrate_field_cross_overlap` (generally complex-valued) can reuse it
    too. `atol` defaults to 0 (pure relative tolerance, the original
    behavior) -- it exists for callers like `integrate_field_cross_overlap`
    whose true value can be exactly or near zero (e.g. two orthogonal
    fields), where a relative-only tolerance never converges on
    floating-point noise fluctuating around that zero.
    """
    n = n_start
    prev = estimator(n)
    for _ in range(max_doublings):
        n *= 2
        curr = estimator(n)
        denom = curr if curr != 0.0 else prev
        if denom == 0.0 or abs(curr - prev) <= max(tol * abs(denom), atol):
            return curr
        prev = curr
    raise RuntimeError(
        f"quadrature did not converge to rtol={tol} (atol={atol}) within {max_doublings} "
        f"doublings (n up to {n})"
    )


def _assert_real(raw_sum: complex) -> float:
    """Section 5: the energy integral is manifestly real and non-negative --
    assert any imaginary part is floating-point noise before dropping it,
    rather than silently propagating a complex value into Module 4."""
    value = complex(raw_sum)
    if value.imag != 0.0:
        tol = _IMAG_REL_TOL * max(abs(value.real), 1e-300)
        if abs(value.imag) > tol:
            raise ValueError(
                f"energy integral has non-negligible imaginary part {value.imag!r} "
                f"(real part {value.real!r}) -- likely a missing conjugate or a "
                "sum(vals**2) instead of sum(abs(vals)**2) bug"
            )
    return value.real


class FieldProvider(ABC):
    """Uniform access to a trial/exact field solution for one cavity mode."""

    @abstractmethod
    def E(self, r: Array) -> Array:
        """r: (3,) or (N,3) in meters, cavity-local Cartesian frame.
        Returns complex field, same leading shape as r, units V/m
        (arbitrary overall scale -- see normalization convention)."""

    @abstractmethod
    def H(self, r: Array) -> Array:
        """Same contract as E, units A/m, same arbitrary scale as E."""

    @abstractmethod
    def total_stored_energy(self) -> float:
        """Denominator of the perturbation formula. Must be on the SAME
        scale as E()/H() returned by this same instance."""

    @property
    @abstractmethod
    def f0(self) -> float:
        """Resonant frequency in Hz."""

    @property
    @abstractmethod
    def epsilon_bg(self) -> complex:
        """Absolute (SI) permittivity of the background fill medium this
        field solution was computed for. Retroactive addition for Module 4
        (module4 doc Section 0.1) -- the units seam where this module's
        absolute eps/mu meets Module 3's relative Material.eps/mu."""

    @property
    @abstractmethod
    def mu_bg(self) -> complex:
        """Absolute (SI) permeability of the background fill medium, same
        rationale as epsilon_bg."""

    @abstractmethod
    def Q_wall(self, Rs: float) -> float:
        """Unloaded Q from finite wall conductivity, given surface
        resistance Rs [Ohm]."""

    def integrate_field_energy(
        self, region: SampleRegion, field: Literal["E", "H"], n_points: int = 2000
    ) -> float:
        """Default (quadrature) implementation of:
            integral over `region` of |E|^2 dV   (field='E')
            integral over `region` of |H|^2 dV   (field='H')
        Concrete subclasses may override with an analytic fast path for
        specific region shapes (Section 4, deferred); callers should never
        need to know which path was taken.
        """
        field_func = self.E if field == "E" else self.H
        volume_checked = False

        def estimate(n: int) -> float:
            nonlocal volume_checked
            pts, w = region.quadrature_points(n)
            if not volume_checked:
                # Section 1.4: a wrong weight-generation formula in Module 3
                # fails this identically at any resolution, so check once.
                total_w = float(np.sum(w))
                vol = region.volume()
                if not np.isclose(total_w, vol, rtol=_VOLUME_RTOL):
                    raise ValueError(
                        f"region.quadrature_points weights sum to {total_w!r}, "
                        f"expected region.volume()={vol!r} (rtol={_VOLUME_RTOL}) -- "
                        "quadrature_points is inconsistent with volume()"
                    )
                volume_checked = True
            vals = field_func(pts)
            density = hermitian_density(vals)
            raw_sum = np.sum(np.asarray(w) * density)
            return _assert_real(raw_sum)

        return converge_by_doubling(estimate, n_points)


class AnalyticalField(FieldProvider):
    """Thin wrapper around a Module-1 CavityMode -- no transformation, no
    re-scaling. The default quadrature path (FieldProvider.integrate_field_energy)
    is correct and sufficient; the analytic fast path (Section 4) is deferred
    until Module 5 profiling justifies it."""

    def __init__(self, mode: CavityMode) -> None:
        self._mode = mode

    def E(self, r: Array) -> Array:
        return self._mode.E(r)

    def H(self, r: Array) -> Array:
        return self._mode.H(r)

    def total_stored_energy(self) -> float:
        return self._mode.total_stored_energy()

    @property
    def f0(self) -> float:
        return self._mode.f0

    @property
    def epsilon_bg(self) -> complex:
        return self._mode.epsilon_bg

    @property
    def mu_bg(self) -> complex:
        return self._mode.mu_bg

    def Q_wall(self, Rs: float) -> float:
        return self._mode.Q_wall(Rs)


def integrate_field_cross_overlap(
    region: SampleRegion,
    field_i: Callable[[Array], Array],
    field_j: Callable[[Array], Array],
    n_points: int = 2000,
) -> complex:
    """integral over `region` of F_i . F_j* dV, for two independently
    supplied vector field functions F_i, F_j (e.g. two different Module 1
    modes' `.E`) -- the cross-overlap primitive needed by the Rayleigh-Ritz
    module's mass-matrix assembly (docs/ritz_module_plan.md Section 2.4),
    which `integrate_field_energy` doesn't cover (that method only handles
    the same-field, same-index case |F|^2). A standalone function rather
    than a `FieldProvider` method, since it operates on two fields, not one
    provider's own E/H pair -- reuses the same quadrature/convergence
    machinery as `integrate_field_energy` directly. Unlike a same-field
    energy integral, this is generally complex (not manifestly real), so no
    `_assert_real` step applies here.

    Two fields can be (near-)exactly orthogonal over `region` -- e.g. two
    distinct Module 1 modes, by symmetry, even over a small sub-volume (this
    is *expected*, not a bug: see `RitzCorrectedModel`'s K matrix, Section
    2.1, which relies on exactly this orthogonality over the *whole*
    cavity). A relative-only tolerance never converges on floating-point
    noise fluctuating around a true value of ~0, so this derives an
    absolute floor from Cauchy-Schwarz (|<F_i,F_j>| <= sqrt(<F_i,F_i><F_j,F_j>))
    up front, using the same quadrature resolution as the first estimate.
    """
    volume_checked = False

    def estimate(n: int) -> complex:
        nonlocal volume_checked
        pts, w = region.quadrature_points(n)
        if not volume_checked:
            total_w = float(np.sum(w))
            vol = region.volume()
            if not np.isclose(total_w, vol, rtol=_VOLUME_RTOL):
                raise ValueError(
                    f"region.quadrature_points weights sum to {total_w!r}, "
                    f"expected region.volume()={vol!r} (rtol={_VOLUME_RTOL}) -- "
                    "quadrature_points is inconsistent with volume()"
                )
            volume_checked = True
        vals_i = field_i(pts)
        vals_j = field_j(pts)
        overlap = np.sum(np.asarray(w) * np.sum(vals_i * np.conj(vals_j), axis=-1))
        return complex(overlap)

    pts0, w0 = region.quadrature_points(n_points)
    w0 = np.asarray(w0)
    scale_i = float(np.sum(w0 * np.sum(np.abs(field_i(pts0)) ** 2, axis=-1)))
    scale_j = float(np.sum(w0 * np.sum(np.abs(field_j(pts0)) ** 2, axis=-1)))
    atol = _CROSS_OVERLAP_ATOL_FACTOR * float(np.sqrt(max(scale_i, 0.0) * max(scale_j, 0.0)))

    return converge_by_doubling(estimate, n_points, atol=atol)
