"""docs/gui_module_plan.md Section 8 -- adapters/*_runner.py (Analytical,
Perturbational, Variational/Ritz; FDTD is covered separately in
test_fdtd_runner.py, since it's much slower). Confirms each runner's
translation from parameters to solver objects is correct and that results
round-trip through the diagnostics dataclasses correctly."""
import pytest

from cavity_perturbation.perturbation import PerturbationModel, PerturbationResult
from cavity_perturbation.ritz import RitzCorrectedModel, RitzDiagnostics
from cavity_perturbation_gui.adapters.analytical_runner import run_analytical
from cavity_perturbation_gui.adapters.cavity_adapter import CavityParams
from cavity_perturbation_gui.adapters.perturbation_runner import run_perturbation
from cavity_perturbation_gui.adapters.ritz_runner import run_ritz
from cavity_perturbation_gui.adapters.sample_adapter import SampleParams

_CAVITY_PARAMS = CavityParams("rectangular", (0.03, 0.02, 0.025), "TE", (0, 1, 1))
_SAMPLE_PARAMS = SampleParams(shape="sphere", radius=1.5e-3, eps_r=4.5, tan_delta_e=0.01)


# --- Analytical --------------------------------------------------------------

def test_run_analytical_no_wall_loss_gives_infinite_q():
    run_result = run_analytical(_CAVITY_PARAMS)
    assert run_result.result.f_calc == pytest.approx(run_result.cavity.f0)
    assert run_result.result.Q_calc == float("inf")
    assert run_result.Rs_walls is None


def test_run_analytical_with_conductivity_matches_q_wall():
    run_result = run_analytical(_CAVITY_PARAMS, conductivity=5.8e7)
    expected_Q_wall = run_result.cavity.Q_wall(run_result.Rs_walls)
    assert run_result.result.Q_calc == pytest.approx(expected_Q_wall)
    assert run_result.result.f_calc == pytest.approx(run_result.cavity.f0)


# --- Perturbational (Module 4) -----------------------------------------------

def test_run_perturbation_matches_direct_perturbation_model():
    run_result = run_perturbation(_CAVITY_PARAMS, _SAMPLE_PARAMS, conductivity=5.8e7)
    assert isinstance(run_result.model, PerturbationModel)
    assert isinstance(run_result.result, PerturbationResult)

    direct = PerturbationModel(run_result.field_provider, Rs_walls=run_result.Rs_walls).evaluate(run_result.sample)
    assert run_result.result == direct


def test_run_perturbation_no_rs_source_means_no_wall_loss():
    run_result = run_perturbation(_CAVITY_PARAMS, _SAMPLE_PARAMS)
    assert run_result.Rs_walls is None


# --- Variational (Ritz) -------------------------------------------------------

def test_run_ritz_fixed_basis_size():
    run_result = run_ritz(_CAVITY_PARAMS, _SAMPLE_PARAMS, n_basis=4, conductivity=5.8e7)
    assert isinstance(run_result.model, RitzCorrectedModel)
    assert isinstance(run_result.diagnostics, RitzDiagnostics)
    assert run_result.model.basis_size == 4
    assert len(run_result.diagnostics.basis_modes) == 4
    assert run_result.diagnostics.coefficients.shape == (4,)


def test_run_ritz_auto_converge_reaches_a_stable_result():
    run_result = run_ritz(_CAVITY_PARAMS, _SAMPLE_PARAMS, auto_converge=True, n_basis=3, convergence_tol=1e-3)
    assert run_result.result.f_calc > 0.0
    assert run_result.model.basis_size >= 3
