"""adapters/fdtd_runner.py -- FDTD tab, per docs/gui_module_plan.md
Section 4."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from cavity_perturbation.cavity import CavityMode
from cavity_perturbation.fdtd import FDTDModel
from cavity_perturbation.fdtd.diagnostics import FDTDDiagnostics
from cavity_perturbation.fields import AnalyticalField
from cavity_perturbation.perturbation import PerturbationResult
from cavity_perturbation.sample import Sample

from .cavity_adapter import CavityParams, build_cavity, resolve_rs
from .sample_adapter import SampleParams, build_sample

_DEFAULT_CELLS_PER_WAVELENGTH = 20.0
_DEFAULT_MIN_CELLS_PER_AXIS = 8
_DEFAULT_RECORD_PERIODS = 10.0


@dataclass(frozen=True)
class FDTDRunResult:
    cavity: CavityMode
    model: FDTDModel
    sample: Sample
    result: PerturbationResult
    diagnostics: FDTDDiagnostics
    Rs_walls: float | None


def run_fdtd(
    cavity_params: CavityParams,
    sample_params: SampleParams,
    *,
    rs: float | None = None,
    conductivity: float | None = None,
    cells_per_wavelength: float = _DEFAULT_CELLS_PER_WAVELENGTH,
    min_cells_per_axis: int = _DEFAULT_MIN_CELLS_PER_AXIS,
    record_periods: float = _DEFAULT_RECORD_PERIODS,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> FDTDRunResult:
    cavity = build_cavity(cavity_params)
    field_provider = AnalyticalField(cavity)
    sample = build_sample(cavity, field_provider, sample_params)
    Rs = resolve_rs(cavity, rs=rs, conductivity=conductivity)
    model = FDTDModel(
        cavity,
        Rs_walls=Rs,
        cells_per_wavelength=cells_per_wavelength,
        min_cells_per_axis=min_cells_per_axis,
        record_periods=record_periods,
    )
    result, diagnostics = model.evaluate_with_diagnostics(
        sample, progress_callback=progress_callback, cancel_check=cancel_check
    )
    return FDTDRunResult(
        cavity=cavity,
        model=model,
        sample=sample,
        result=result,
        diagnostics=diagnostics,
        Rs_walls=Rs,
    )
