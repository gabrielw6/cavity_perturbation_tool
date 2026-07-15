"""adapters/perturbation_runner.py -- Perturbational tab (Module 4), per
docs/gui_module_plan.md Section 4."""
from __future__ import annotations

from dataclasses import dataclass

from cavity_perturbation.cavity import CavityMode
from cavity_perturbation.fields import AnalyticalField, FieldProvider
from cavity_perturbation.perturbation import PerturbationModel, PerturbationResult
from cavity_perturbation.sample import Sample

from .cavity_adapter import CavityParams, build_cavity, resolve_rs
from .sample_adapter import SampleParams, build_sample


@dataclass(frozen=True)
class PerturbationRunResult:
    cavity: CavityMode
    field_provider: FieldProvider
    model: PerturbationModel
    sample: Sample
    result: PerturbationResult
    Rs_walls: float | None


def run_perturbation(
    cavity_params: CavityParams,
    sample_params: SampleParams,
    *,
    rs: float | None = None,
    conductivity: float | None = None,
) -> PerturbationRunResult:
    cavity = build_cavity(cavity_params)
    field_provider = AnalyticalField(cavity)
    sample = build_sample(cavity, field_provider, sample_params)
    Rs = resolve_rs(cavity, rs=rs, conductivity=conductivity)
    model = PerturbationModel(field_provider, Rs_walls=Rs)
    result = model.evaluate(sample)
    return PerturbationRunResult(
        cavity=cavity,
        field_provider=field_provider,
        model=model,
        sample=sample,
        result=result,
        Rs_walls=Rs,
    )
