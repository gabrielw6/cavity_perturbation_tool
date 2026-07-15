"""Module 5 -- Inverse: Nonlinear Least-Squares Fit.

Recovers (eps, mu) from measured (f, Q) pairs by driving Module 4's forward
model with `scipy.optimize.least_squares`, per docs/module5_inverse_equations.md.
Never imports `FieldProvider`, `CavityMode`, or quadrature code directly --
only calls `Measurement.model.evaluate(sample)` and `Measurement.model`'s
public `field_provider`/`Rs_walls` accessors (module5 doc "Contract from
Module 4").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares

from .perturbation import PerturbationModelLike
from .sample import Material, Sample, SampleRegion

Array = np.ndarray

Bounds = tuple[float, float, float, float]  # (lo_re, lo_im, hi_re, hi_im)

_DEFAULT_EPS_BOUNDS: Bounds = (1.0, 0.0, np.inf, np.inf)
_DEFAULT_MU_BOUNDS: Bounds = (1.0, 0.0, np.inf, np.inf)
_NEUTRAL_SEED_EPS = 2.0 - 0j
_NEUTRAL_SEED_MU = 1.0 - 0j


@dataclass(frozen=True)
class Measurement:
    model: PerturbationModelLike  # bound to the mode/geometry this reading came from
    region: SampleRegion  # sample position/shape at time of this reading
    f_meas: float  # Hz
    Q_meas: float
    sigma_f: float = 1e-4  # fractional (relative) frequency uncertainty
    sigma_Q: float = 1e-2  # fractional Q uncertainty (== fractional 1/Q uncertainty, 0.2)

    def __post_init__(self) -> None:
        if not self.f_meas > 0:
            raise ValueError(f"f_meas must be > 0, got {self.f_meas!r}")
        if not (0.0 < self.Q_meas < float("inf")):
            raise ValueError(f"Q_meas must be finite and > 0, got {self.Q_meas!r}")


@dataclass(frozen=True)
class FitResult:
    eps: complex
    mu: complex
    success: bool
    residual_norm: float
    n_measurements: int
    covariance: Array | None
    condition_number: float
    raw: object  # scipy OptimizeResult, kept for diagnostics


def point_dipole_filling_factors(model: PerturbationModelLike, region: SampleRegion) -> tuple[complex, complex]:
    """p_E^(0), p_H^(0) (Section 2.1) -- the material-independent piece of
    Module 4's filling factors in the point-dipole limit (kappa_E=kappa_H=1,
    Module 3's 'generic' fallback). Standalone function of (model, region)
    only, reusable by both the closed-form seed and any future diagnostics."""
    fp = model.field_provider
    I_E = fp.integrate_field_energy(region, "E")
    I_H = fp.integrate_field_energy(region, "H")
    W = fp.total_stored_energy()
    p_E = fp.epsilon_bg * I_E / W
    p_H = fp.mu_bg * I_H / W
    return p_E, p_H


def _delta_from_measurement(model: PerturbationModelLike, f_meas: float, Q_meas: float) -> complex:
    """Invert Module 4's combination formula (Section 2.2) for Delta given
    the measured complex resonance."""
    fp = model.field_provider
    omega0 = 2.0 * np.pi * fp.f0
    omega_meas = 2.0 * np.pi * f_meas * (1.0 - 1j / (2.0 * Q_meas))
    Rs_walls = model.Rs_walls
    if Rs_walls is not None:
        Q_wall = fp.Q_wall(Rs_walls)
        wall_term = 1j / (2.0 * Q_wall)
    else:
        wall_term = 0j
    return omega_meas / omega0 - 1.0 + wall_term


def _closed_form_seed(measurements: Sequence[Measurement], fit_mu: bool) -> Array:
    """Cheap non-iterative estimate to seed least_squares (Sections 2.3-2.5),
    using the first (fit_mu=False) or first two (fit_mu=True) measurements'
    point-dipole filling factors directly -- bypassing PerturbationModel's
    general (depolarization-corrected) path entirely."""
    m0 = measurements[0]
    p_E0, p_H0 = point_dipole_filling_factors(m0.model, m0.region)
    delta0 = _delta_from_measurement(m0.model, m0.f_meas, m0.Q_meas)

    # Doc bug (module5 doc Sections 2.2-2.4 vs. module4_delta_sign_error):
    # the doc's inversion is derived against Module 4's *literal* Delta
    # formula, delta = -0.5*(eps_r-1)*p_E -- but perturbation.py implements
    # the conjugate-corrected formula, delta = -0.5*conj(eps_r-1)*p_E (the
    # fix required for passivity, see PerturbationModel.evaluate). Inverting
    # the formula actually used by the forward model:
    #   delta = -0.5*conj(eps_r-1)*p_E  =>  eps_r-1 = -2*conj(delta/p_E)
    if not fit_mu:
        eps_r0 = 1.0 - 2.0 * np.conj(delta0 / p_E0)
        return np.array([eps_r0.real, -eps_r0.imag])

    if len(measurements) < 2:
        # Defensive fallback (2.5) -- shouldn't happen given __init__'s
        # fit_mu/measurement-count guard (0.6), but a single measurement
        # cannot separate eps and mu, so don't attempt an ill-posed solve.
        eps_r0, mu_r0 = _NEUTRAL_SEED_EPS, _NEUTRAL_SEED_MU
        return np.array([eps_r0.real, -eps_r0.imag, mu_r0.real, -mu_r0.imag])

    m1 = measurements[1]
    p_E1, p_H1 = point_dipole_filling_factors(m1.model, m1.region)
    delta1 = _delta_from_measurement(m1.model, m1.f_meas, m1.Q_meas)

    # Section 2.4: a miniature version of the identifiability check -- if
    # the two measurements share nearly the same p_E:p_H ratio, this matrix
    # is ill-conditioned (or, for exactly-degenerate ratios, exactly
    # singular) and the seed itself will be poor, an early warning before
    # the optimizer ever runs (Section 4.2 applies the same diagnostic to
    # the full nonlinear fit's Jacobian). A singular system falls back to
    # the same neutral prior as the underdetermined case (2.5) rather than
    # letting solve()'s exception propagate out of seed generation.
    A = np.array([[p_E0, p_H0], [p_E1, p_H1]], dtype=complex)
    b = -2.0 * np.array([delta0, delta1], dtype=complex)
    try:
        chi = np.linalg.solve(A, b)  # (conj(eps_r-1), conj(mu_r-1)) -- see conjugate note above
    except np.linalg.LinAlgError:
        eps_r0, mu_r0 = _NEUTRAL_SEED_EPS, _NEUTRAL_SEED_MU
        return np.array([eps_r0.real, -eps_r0.imag, mu_r0.real, -mu_r0.imag])
    eps_r0 = 1.0 + np.conj(chi[0])
    mu_r0 = 1.0 + np.conj(chi[1])
    return np.array([eps_r0.real, -eps_r0.imag, mu_r0.real, -mu_r0.imag])


class InverseSolver:
    """Fits (eps, mu) to one or more Measurements via nonlinear least
    squares. Never imports FieldProvider, CavityMode, or quadrature code --
    only calls Measurement.model.evaluate(sample) and the model's public
    field_provider/Rs_walls accessors, so a later analytic-Jacobian or
    Ritz-backed model swaps in without this class changing shape."""

    def __init__(
        self,
        measurements: Sequence[Measurement],
        fit_mu: bool = False,
        eps_bounds: Bounds = _DEFAULT_EPS_BOUNDS,
        mu_bounds: Bounds = _DEFAULT_MU_BOUNDS,
    ) -> None:
        """fit_mu: if False, mu is held fixed at 1 (typical dielectric
        characterization) and only (eps', eps'') are fit -- reduces the
        unknown vector from 4 to 2 and sidesteps the identifiability
        problem for the common case.

        eps_bounds/mu_bounds: (lo_re, lo_im, hi_re, hi_im) fitting priors
        (Section 0.3), e.g. eps'>=1 for an ordinary dielectric -- distinct
        from Material.is_passive's physical-law check, which is enforced
        independently at the PerturbationModel.evaluate boundary regardless
        of these bounds.
        """
        measurements = list(measurements)
        if not measurements:
            raise ValueError("at least one Measurement is required")
        if fit_mu and len(measurements) < 2:
            raise ValueError(
                "fit_mu=True requires at least two measurements (Section 0.6) "
                "-- a single (f, Q) pair cannot separate eps and mu"
            )
        self._meas = measurements
        self._fit_mu = fit_mu
        self._eps_bounds = eps_bounds
        self._mu_bounds = mu_bounds

    def _unpack(self, p: Array) -> Material:
        if self._fit_mu:
            eps = p[0] - 1j * p[1]
            mu = p[2] - 1j * p[3]
        else:
            eps = p[0] - 1j * p[1]
            mu = 1.0 - 0j
        return Material(eps=eps, mu=mu)

    def _residuals(self, p: Array) -> Array:
        material = self._unpack(p)
        res = []
        for m in self._meas:
            sample = Sample(region=m.region, material=material)
            r = m.model.evaluate(sample)
            res.append((r.f_calc - m.f_meas) / (m.sigma_f * m.f_meas))
            # 1/r.Q_calc is exactly 0.0 in IEEE arithmetic when Q_calc is
            # inf -- no special-casing needed (Section 1.3).
            res.append((1.0 / r.Q_calc - 1.0 / m.Q_meas) / (m.sigma_Q * (1.0 / m.Q_meas)))
        return np.array(res)

    def _bounds(self) -> tuple[list[float], list[float]]:
        lo = [self._eps_bounds[0], self._eps_bounds[1]]
        hi = [self._eps_bounds[2], self._eps_bounds[3]]
        if self._fit_mu:
            lo += [self._mu_bounds[0], self._mu_bounds[1]]
            hi += [self._mu_bounds[2], self._mu_bounds[3]]
        return lo, hi

    def _initial_guess_vector(self, guess: Material | None) -> Array:
        if guess is not None:
            base = [guess.eps.real, -guess.eps.imag]
            if self._fit_mu:
                base += [guess.mu.real, -guess.mu.imag]
            return np.array(base)
        return _closed_form_seed(self._meas, self._fit_mu)

    def fit(self, initial_guess: Material | None = None) -> FitResult:
        p0 = self._initial_guess_vector(initial_guess)
        lo, hi = self._bounds()
        # A closed-form seed (or a user-supplied guess) derived from data
        # that violates the fitting prior would otherwise be infeasible for
        # least_squares (it requires lb <= x0 <= ub) -- clip in rather than
        # let the seed silently violate the bounds it's about to be
        # optimized under (Section 6, "Bounds enforcement").
        p0 = np.clip(p0, lo, hi)
        result = least_squares(
            self._residuals, p0, bounds=(lo, hi), method="trf", jac="2-point", x_scale="jac"
        )
        material = self._unpack(result.x)

        # Section 4: result.jac is the Jacobian of the already-weighted
        # residual vector, so (J^T J)^-1 needs no extra sigma^2 rescaling.
        # (least_squares' return type is broader than what jac='2-point'
        # actually produces here -- always a dense ndarray -- so pin the
        # type explicitly for mypy.)
        J: Array = np.asarray(result.jac, dtype=float)
        JTJ = J.T @ J
        covariance = np.linalg.pinv(JTJ)  # 4.3: pinv, won't raise on singular JTJ
        eigenvalues = np.linalg.eigvalsh(JTJ)
        lambda_min, lambda_max = eigenvalues[0], eigenvalues[-1]
        if lambda_min <= 0.0 or lambda_max <= 0.0:
            condition_number = float("inf")
        else:
            condition_number = float(lambda_max / lambda_min)

        return FitResult(
            eps=material.eps,
            mu=material.mu,
            success=bool(result.success),
            residual_norm=float(np.linalg.norm(result.fun)),
            n_measurements=len(self._meas),
            covariance=covariance,
            condition_number=condition_number,
            raw=result,
        )
