"""adapters/analytical_runner.py -- Analytical tab (docs/gui_module_plan.md
Section 4). Module 1/2 only: the cavity's own closed-form empty-cavity
resonance, no sample. Represented as a `PerturbationResult`
(f_calc=f0, Q_calc=Q_wall) purely so `curve_plot.py` can treat every forward
tab's answer uniformly -- Analytical never calls Module 4/Ritz/FDTD."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cavity_perturbation.cavity import CavityMode
from cavity_perturbation.fields import AnalyticalField, FieldProvider
from cavity_perturbation.perturbation import PerturbationResult, omega_tilde_to_result

from .cavity_adapter import CavityParams, build_cavity, resolve_rs


@dataclass(frozen=True)
class AnalyticalRunResult:
    cavity: CavityMode
    field_provider: FieldProvider
    result: PerturbationResult
    Rs_walls: float | None


def run_analytical(
    cavity_params: CavityParams, *, rs: float | None = None, conductivity: float | None = None
) -> AnalyticalRunResult:
    cavity = build_cavity(cavity_params)
    field_provider = AnalyticalField(cavity)
    Rs = resolve_rs(cavity, rs=rs, conductivity=conductivity)

    omega0 = 2.0 * np.pi * cavity.f0
    if Rs is not None:
        Q_wall = cavity.Q_wall(Rs)
        omega_tilde = omega0 * (1.0 - 1j / (2.0 * Q_wall))
    else:
        omega_tilde = complex(omega0)
    result = omega_tilde_to_result(omega_tilde)

    return AnalyticalRunResult(cavity=cavity, field_provider=field_provider, result=result, Rs_walls=Rs)
