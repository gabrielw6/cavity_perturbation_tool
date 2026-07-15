"""docs/gui_module_plan.md Section 8 -- adapters/inversion_runner.py."""
import pytest

from cavity_perturbation.inverse import FitResult, Measurement
from cavity_perturbation_gui.adapters.cavity_adapter import CavityParams
from cavity_perturbation_gui.adapters.inversion_runner import measurement_from_result, run_inversion
from cavity_perturbation_gui.adapters.perturbation_runner import run_perturbation
from cavity_perturbation_gui.adapters.ritz_runner import run_ritz
from cavity_perturbation_gui.adapters.sample_adapter import SampleParams

_CAVITY_PARAMS = CavityParams("rectangular", (0.03, 0.02, 0.025), "TE", (0, 1, 1))
_SAMPLE_PARAMS = SampleParams(shape="sphere", radius=1.5e-3, eps_r=4.5, tan_delta_e=0.01)


def test_measurement_from_result_binds_model_and_region():
    run_result = run_perturbation(_CAVITY_PARAMS, _SAMPLE_PARAMS, conductivity=5.8e7)
    measurement = measurement_from_result(run_result.model, run_result.sample.region, run_result.result)
    assert isinstance(measurement, Measurement)
    assert measurement.model is run_result.model
    assert measurement.region is run_result.sample.region
    assert measurement.f_meas == run_result.result.f_calc
    assert measurement.Q_meas == run_result.result.Q_calc


def test_run_inversion_recovers_known_material_from_perturbation_model():
    run_result = run_perturbation(_CAVITY_PARAMS, _SAMPLE_PARAMS, conductivity=5.8e7)
    measurement = measurement_from_result(run_result.model, run_result.sample.region, run_result.result)
    fit = run_inversion([measurement])
    assert isinstance(fit, FitResult)
    assert fit.eps.real == pytest.approx(4.5, rel=1e-3)


def test_run_inversion_accepts_a_ritz_backed_measurement():
    # Section 2.4's widened PerturbationModelLike: the same inversion path
    # must work for a Ritz-backed model, not only PerturbationModel.
    run_result = run_ritz(_CAVITY_PARAMS, _SAMPLE_PARAMS, n_basis=4, conductivity=5.8e7)
    measurement = measurement_from_result(run_result.model, run_result.sample.region, run_result.result)
    fit = run_inversion([measurement])
    assert fit.eps.real > 0.0
