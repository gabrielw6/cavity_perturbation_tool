"""docs/gui_module_plan.md Section 8 -- adapters/fdtd_runner.py. Kept to a
single coarse/fast run -- fdtd/'s own test suite already covers accuracy;
this only confirms the adapter's translation and diagnostics wiring."""
from cavity_perturbation.fdtd.diagnostics import FDTDDiagnostics
from cavity_perturbation.fdtd.model import FDTDModel
from cavity_perturbation_gui.adapters.cavity_adapter import CavityParams
from cavity_perturbation_gui.adapters.fdtd_runner import run_fdtd
from cavity_perturbation_gui.adapters.sample_adapter import SampleParams

_CAVITY_PARAMS = CavityParams("rectangular", (0.03, 0.02, 0.025), "TE", (0, 1, 1))
_SAMPLE_PARAMS = SampleParams(shape="sphere", radius=1.5e-3, eps_r=4.5, tan_delta_e=0.01)


def test_run_fdtd_returns_model_and_diagnostics():
    run_result = run_fdtd(
        _CAVITY_PARAMS, _SAMPLE_PARAMS, cells_per_wavelength=8, min_cells_per_axis=6
    )
    assert isinstance(run_result.model, FDTDModel)
    assert isinstance(run_result.diagnostics, FDTDDiagnostics)
    assert run_result.result.f_calc > 0.0
    assert run_result.Rs_walls is None
