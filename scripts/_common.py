"""Shared plumbing for the scripts/example_*.py demo scripts. Not part of
the public cavity_perturbation API -- just the bits common to every example
(field-max sample placement, forward+inverse round trip, report formatting)
so each example script stays focused on what it's actually varying.
"""
from __future__ import annotations

import numpy as np
from scipy import constants

from cavity_perturbation.cavity import CavityMode
from cavity_perturbation.fields import FieldProvider
from cavity_perturbation.inverse import FitResult, InverseSolver, Measurement
from cavity_perturbation.perturbation import PerturbationModel, PerturbationResult
from cavity_perturbation.sample import Material, Sample, SampleRegion

COPPER_CONDUCTIVITY = 5.8e7  # S/m
FREQ_SHIFT_WARN_THRESHOLD = 0.01  # |df/f0| above this is well outside first-order perturbation theory's assumed small-sample regime


def rs_from_conductivity(f0: float, sigma: float = COPPER_CONDUCTIVITY) -> float:
    """Copper-wall surface resistance at f0 (skin-effect formula)."""
    return float(np.sqrt(np.pi * f0 * constants.mu_0 / sigma))


def field_max_position(cav: CavityMode, field: FieldProvider, margin: float = 0.0, n: int = 25) -> np.ndarray:
    """Grid-search for the E-field maximum inside the cavity's valid domain
    (same rationale as scripts/simulate_perturbation.py's resolve_position):
    the bounding box's own center isn't a safe generic default -- e.g. it
    sits exactly on CoaxialCavity's rho=0 singularity -- and a real
    measurement places the sample at a field extremum for maximum
    sensitivity anyway. `margin` excludes candidates too close to a
    boundary, needed because a mode flat along one axis (e.g. TE_0np) ties
    across that whole axis and a naive argmax would pick a boundary point.
    """
    rmin, rmax = cav.bounding_box()
    axes = [np.linspace(lo, hi, n) for lo, hi in zip(rmin, rmax)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    candidates = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    safe = cav.contains(candidates)
    if margin > 0:
        for axis in range(3):
            offset = np.zeros(3)
            offset[axis] = margin
            safe = safe & cav.contains(candidates + offset) & cav.contains(candidates - offset)
    eligible = candidates[safe] if np.any(safe) else candidates[cav.contains(candidates)]
    if eligible.shape[0] == 0:
        raise ValueError("no valid point found inside the cavity for the sample")

    with np.errstate(all="ignore"):
        e_mag2 = np.sum(np.abs(field.E(eligible)) ** 2, axis=-1)
    e_mag2 = np.where(np.isfinite(e_mag2), e_mag2, 0.0)
    return eligible[np.argmax(e_mag2)]


def round_trip(
    model: PerturbationModel,
    region: SampleRegion,
    true_material: Material,
    sigma_f: float = 1e-4,
    sigma_Q: float = 1e-2,
    fit_mu: bool = False,
) -> tuple[PerturbationResult, FitResult]:
    """Forward-simulate (f, Q) for `true_material` sitting in `region`, then
    feed that single noiseless measurement straight back into Module 5's
    InverseSolver -- a full Module 1(4)->5 round trip. Returns (forward
    result, fit result) so a caller can compare fitted vs. true material and
    report the fit's formal uncertainty."""
    sample = Sample(region=region, material=true_material)
    forward = model.evaluate(sample)
    measurement = Measurement(
        model=model, region=region, f_meas=forward.f_calc, Q_meas=forward.Q_calc,
        sigma_f=sigma_f, sigma_Q=sigma_Q,
    )
    fit = InverseSolver([measurement], fit_mu=fit_mu).fit()
    return forward, fit


def perturbation_validity_warning(
    f0: float, forward: PerturbationResult, threshold: float = FREQ_SHIFT_WARN_THRESHOLD
) -> str | None:
    """Module 4's Delta formula is a first-order (small-sample) perturbation
    result -- docs/module4_perturbation_equations.md repeatedly invokes "the
    small-sample assumption" as the basis for the whole derivation. Nothing
    in evaluate() enforces that assumption, and InverseSolver inverts the
    *same* formula the synthetic measurement was generated with, so a
    round-trip example recovers the input material exactly regardless of
    sample size -- even once the perturbation itself is enormous (e.g. a
    >90% frequency shift) and the small-sample approximation has long since
    broken down. That "exact recovery" reflects code self-consistency
    between Modules 4 and 5, not physical measurement accuracy, which is
    why this check exists: flag it explicitly rather than let a converged
    fit imply the result means anything physically."""
    freq_shift = abs(forward.f_calc - f0) / f0
    if freq_shift <= threshold:
        return None
    return (
        f"WARNING: |df/f0| = {freq_shift:.1%} is well outside first-order perturbation "
        "theory's small-sample regime. The forward model and InverseSolver share the same "
        "formula, so recovery still looks exact -- that no longer reflects what a real "
        "measurement of a sample this large could achieve."
    )


def fmt_complex(z: complex) -> str:
    sign = "+" if z.imag >= 0 else "-"
    return f"{z.real:.6g}{sign}{abs(z.imag):.3g}j"


def eps_sigma(fit: FitResult) -> tuple[float, float]:
    """1-sigma formal uncertainty on (eps', eps'') implied by the
    measurement's sigma_f/sigma_Q, read off the fit's covariance matrix
    (module5 doc Section 4) -- the propagated effect of the declared
    measurement precision on the recovered material, not a comparison
    against the (here, known) true value."""
    if fit.covariance is None:
        return float("nan"), float("nan")
    diag = np.diag(fit.covariance)
    return float(np.sqrt(max(diag[0], 0.0))), float(np.sqrt(max(diag[1], 0.0)))


def relative_error(fitted: complex, true: complex) -> float:
    return float(abs(fitted - true) / abs(true))
