"""docs/gui_module_plan.md Section 2.1 -- FDTDModel.evaluate_with_diagnostics.
evaluate() itself must stay exactly as before; these tests only cover the
additive diagnostics path."""
import numpy as np
import pytest

from cavity_perturbation.cavity import ModeIndex, RectangularCavity
from cavity_perturbation.fdtd.diagnostics import FDTDDiagnostics
from cavity_perturbation.fdtd.grid.yee import E_COMPONENTS, H_COMPONENTS, YeeGrid
from cavity_perturbation.fdtd.model import FDTDModel
from cavity_perturbation.sample import Material, Sample, Sphere

_CAV = RectangularCavity(0.03, 0.02, 0.025, ModeIndex("TE", (0, 1, 1)))
_SAMPLE = Sample(
    region=Sphere(center=[0.015, 0.01, 0.0125], radius=1e-9),
    material=Material.from_loss_tangent(eps_r=1.0, tan_delta_e=0.0),
)


def _fast_model(**overrides):
    kwargs = dict(cells_per_wavelength=8, min_cells_per_axis=6)
    kwargs.update(overrides)
    return FDTDModel(_CAV, **kwargs)


def test_evaluate_with_diagnostics_matches_evaluate():
    model = _fast_model()
    result = model.evaluate(_SAMPLE)
    result_with_diag, diagnostics = model.evaluate_with_diagnostics(_SAMPLE)

    assert result_with_diag == result
    assert isinstance(diagnostics, FDTDDiagnostics)


def test_diagnostics_excitation_and_probe_series_shapes_match_times():
    model = _fast_model()
    _, diagnostics = model.evaluate_with_diagnostics(_SAMPLE)

    assert diagnostics.excitation_times.shape == diagnostics.excitation_waveform.shape
    assert diagnostics.excitation_times.size > 0
    assert diagnostics.probe_times.shape == diagnostics.probe_series.shape
    assert diagnostics.probe_times.size > 0
    # ringdown recording starts after excitation ends (Section 3 in
    # fdtd_module_plan.md: the analysis window must start after the source
    # is off)
    assert diagnostics.probe_times[0] > diagnostics.excitation_times[-1]


def test_diagnostics_spectrum_fields_populated_from_extract_fft():
    model = _fast_model()
    _, diagnostics = model.evaluate_with_diagnostics(_SAMPLE)
    assert diagnostics.spectrum_freqs is not None
    assert diagnostics.spectrum_power is not None
    assert diagnostics.spectrum_freqs.shape == diagnostics.spectrum_power.shape


def test_diagnostics_field_snapshot_has_all_six_components_matching_grid_shape():
    model = _fast_model()
    _, diagnostics = model.evaluate_with_diagnostics(_SAMPLE)

    assert isinstance(diagnostics.snapshot_grid, YeeGrid)
    assert set(diagnostics.field_snapshot.keys()) == set(E_COMPONENTS) | set(H_COMPONENTS)
    for component, arr in diagnostics.field_snapshot.items():
        assert arr.shape == diagnostics.snapshot_grid.shape


def test_diagnostics_field_snapshot_taken_at_end_of_excitation_not_zero():
    # The excitation pulse should have actually driven the field by the
    # time the snapshot is taken -- a snapshot of all zeros would mean it
    # was captured before any source was ever injected.
    model = _fast_model()
    _, diagnostics = model.evaluate_with_diagnostics(_SAMPLE)
    assert any(np.any(arr != 0.0) for arr in diagnostics.field_snapshot.values())


def test_evaluate_does_not_require_diagnostics_and_stays_cheap_to_call():
    # Plain evaluate() must still work standalone (no diagnostics plumbing
    # required) -- regression guard for the _run(capture=False) path.
    model = _fast_model()
    result = model.evaluate(_SAMPLE)
    assert result.f_calc > 0.0
