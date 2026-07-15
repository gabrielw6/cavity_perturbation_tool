"""adapters/ritz_runner.py -- Variational (Ritz) tab, per
docs/gui_module_plan.md Section 4."""
from __future__ import annotations

from dataclasses import dataclass

from cavity_perturbation.cavity import CavityMode
from cavity_perturbation.fields import AnalyticalField
from cavity_perturbation.perturbation import PerturbationResult
from cavity_perturbation.ritz import (
    RitzCorrectedModel,
    RitzDiagnostics,
    converged_ritz_model,
    nearest_basis_modes,
)
from cavity_perturbation.sample import Sample

from .cavity_adapter import CavityParams, build_cavity, cavity_constructor_and_args, resolve_rs
from .sample_adapter import SampleParams, build_sample

_DEFAULT_N_BASIS = 5
_DEFAULT_MAX_INDEX = 4
_DEFAULT_CONVERGENCE_TOL = 1e-4


@dataclass(frozen=True)
class RitzRunResult:
    cavity: CavityMode
    model: RitzCorrectedModel
    sample: Sample
    result: PerturbationResult
    diagnostics: RitzDiagnostics
    Rs_walls: float | None


def run_ritz(
    cavity_params: CavityParams,
    sample_params: SampleParams,
    *,
    rs: float | None = None,
    conductivity: float | None = None,
    n_basis: int = _DEFAULT_N_BASIS,
    max_index: int = _DEFAULT_MAX_INDEX,
    auto_converge: bool = False,
    convergence_tol: float = _DEFAULT_CONVERGENCE_TOL,
) -> RitzRunResult:
    """`auto_converge=True` grows the basis via `converged_ritz_model`
    (docs/ritz_module_plan.md Section 3.5) instead of using a fixed
    `n_basis` -- the "basis size / convergence tolerance" extra settings
    Section 4 calls out for this tab."""
    cavity = build_cavity(cavity_params)
    field_provider = AnalyticalField(cavity)
    sample = build_sample(cavity, field_provider, sample_params)
    Rs = resolve_rs(cavity, rs=rs, conductivity=conductivity)
    cavity_type, cavity_args, mode = cavity_constructor_and_args(cavity_params)

    if auto_converge:
        model, result = converged_ritz_model(
            cavity_type,
            cavity_args,
            mode,
            sample,
            Rs_walls=Rs,
            eps_bg=cavity.epsilon_bg,
            mu_bg=cavity.mu_bg,
            n_start=max(n_basis, 1),
            tol=convergence_tol,
        )
        _, diagnostics = model.evaluate_with_diagnostics(sample)
    else:
        basis = nearest_basis_modes(
            cavity_type,
            cavity_args,
            mode,
            n_basis=n_basis,
            max_index=max_index,
            eps_bg=cavity.epsilon_bg,
            mu_bg=cavity.mu_bg,
        )
        model = RitzCorrectedModel(basis, Rs_walls=Rs)
        result, diagnostics = model.evaluate_with_diagnostics(sample)

    return RitzRunResult(
        cavity=cavity,
        model=model,
        sample=sample,
        result=result,
        diagnostics=diagnostics,
        Rs_walls=Rs,
    )
