"""docs/gui_module_plan.md Section 8 -- PerturbationTab: adapter mocked,
confirms the field-plane view's per-tab honesty relabeling (Section 5's
Perturbational note) and the "use this result" measurement-capture wiring
(Section 5.6)."""
import cavity_perturbation_gui.widgets.tabs.perturbation_tab as perturbation_tab_module
from cavity_perturbation.cavity import ModeIndex, RectangularCavity
from cavity_perturbation.fields import AnalyticalField
from cavity_perturbation.inverse import Measurement
from cavity_perturbation.perturbation import PerturbationModel, PerturbationResult
from cavity_perturbation.sample import Material, Sample, Sphere
from cavity_perturbation_gui.adapters.perturbation_runner import PerturbationRunResult
from cavity_perturbation_gui.widgets.geometry_view3d import GeometryView3D
from cavity_perturbation_gui.widgets.sidebar import Sidebar
from cavity_perturbation_gui.widgets.tabs.perturbation_tab import PerturbationTab

_CAV = RectangularCavity(0.03, 0.02, 0.025, ModeIndex("TE", (0, 1, 1)))


def _fake_result():
    field = AnalyticalField(_CAV)
    sample = Sample(region=Sphere(center=[0.015, 0.01, 0.0125], radius=1.5e-3), material=Material.from_loss_tangent(4.5, 0.01))
    model = PerturbationModel(field, Rs_walls=0.02)
    return PerturbationRunResult(
        cavity=_CAV,
        field_provider=field,
        model=model,
        sample=sample,
        result=PerturbationResult(f_calc=1e9, Q_calc=500.0, omega_tilde=complex(2 * 3.14159e9, -1e6)),
        Rs_walls=0.02,
    )


def _build_tab(qtbot, monkeypatch, run_fn):
    monkeypatch.setattr(perturbation_tab_module, "run_perturbation", run_fn)
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    geometry_view = GeometryView3D()
    qtbot.addWidget(geometry_view)
    tab = PerturbationTab(sidebar, geometry_view)
    qtbot.addWidget(tab)
    return tab


def test_field_view_shows_the_unperturbed_field_honesty_note(qtbot, monkeypatch):
    tab = _build_tab(qtbot, monkeypatch, lambda *a, **k: _fake_result())
    tab.show()  # isVisible() requires the widget to actually be shown, not just addWidget()-ed
    tab.run_bar.button.click()
    qtbot.waitUntil(lambda: tab.last_result is not None, timeout=2000)

    note = tab.field_view._note_label.text()
    assert "unperturbed" in note.lower()
    assert tab.field_view._note_label.isVisible()


def test_use_as_measurement_emits_a_measurement_bound_to_the_models_own_instance(qtbot, monkeypatch):
    tab = _build_tab(qtbot, monkeypatch, lambda *a, **k: _fake_result())
    tab.run_bar.button.click()
    qtbot.waitUntil(lambda: tab.last_result is not None, timeout=2000)

    captured = []
    tab.measurement_captured.connect(captured.append)
    assert tab.use_as_measurement_button.isEnabled()
    tab.use_as_measurement_button.click()

    assert len(captured) == 1
    (measurement,) = captured
    assert isinstance(measurement, Measurement)
    assert measurement.model is tab.last_result.model
    assert measurement.region is tab.last_result.sample.region
    assert measurement.f_meas == tab.last_result.result.f_calc


def test_use_as_measurement_disabled_before_any_run(qtbot, monkeypatch):
    tab = _build_tab(qtbot, monkeypatch, lambda *a, **k: _fake_result())
    assert not tab.use_as_measurement_button.isEnabled()
