"""adapters/inversion_runner.py -- Inversion tab (docs/gui_module_plan.md
Section 0.4 / Section 4). Thin wrapper over `InverseSolver`/`Measurement`/
`FitResult` -- consumes either pasted-in (f_meas, Q_meas) values or a
"use this result" measurement captured directly from a forward tab
(Section 5.6)."""
from __future__ import annotations

from cavity_perturbation.inverse import FitResult, InverseSolver, Measurement
from cavity_perturbation.perturbation import PerturbationModelLike, PerturbationResult
from cavity_perturbation.sample import Material, SampleRegion

_DEFAULT_SIGMA_F = 1e-4
_DEFAULT_SIGMA_Q = 1e-2


def measurement_from_result(
    model: PerturbationModelLike,
    region: SampleRegion,
    result: PerturbationResult,
    *,
    sigma_f: float = _DEFAULT_SIGMA_F,
    sigma_Q: float = _DEFAULT_SIGMA_Q,
) -> Measurement:
    """Section 5.6's "use this result" action: binds a forward tab's own
    model instance and sample region to its last result as a synthetic
    measurement -- works for a Perturbational-, Ritz-, or FDTD-backed model
    alike via Section 2.4's widened `PerturbationModelLike`."""
    return Measurement(
        model=model,
        region=region,
        f_meas=result.f_calc,
        Q_meas=result.Q_calc,
        sigma_f=sigma_f,
        sigma_Q=sigma_Q,
    )


def run_inversion(
    measurements: list[Measurement],
    *,
    fit_mu: bool = False,
    initial_guess: Material | None = None,
) -> FitResult:
    solver = InverseSolver(measurements, fit_mu=fit_mu)
    return solver.fit(initial_guess=initial_guess)
