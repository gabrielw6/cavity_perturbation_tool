"""docs/gui_module_plan.md Section 8 -- InversionTab: adapter mocked."""
import cavity_perturbation_gui.widgets.tabs.inversion_tab as inversion_tab_module
from cavity_perturbation.cavity import ModeIndex, RectangularCavity
from cavity_perturbation.fields import AnalyticalField
from cavity_perturbation.inverse import FitResult, Measurement
from cavity_perturbation.perturbation import PerturbationModel
from cavity_perturbation.sample import Sphere
from cavity_perturbation_gui.widgets.sidebar import Sidebar
from cavity_perturbation_gui.widgets.tabs.inversion_tab import InversionTab

_CAV = RectangularCavity(0.03, 0.02, 0.025, ModeIndex("TE", (0, 1, 1)))


def _measurement():
    model = PerturbationModel(AnalyticalField(_CAV), Rs_walls=0.02)
    region = Sphere(center=[0.015, 0.01, 0.0125], radius=1.5e-3)
    return Measurement(model=model, region=region, f_meas=1e9, Q_meas=500.0)


def test_add_measurement_appends_to_list(qtbot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    tab = InversionTab(sidebar)
    qtbot.addWidget(tab)

    tab.add_measurement(_measurement())

    assert tab.measurement_list.count() == 1
    assert len(tab._measurements) == 1


def test_remove_selected_removes_from_both_list_and_backing_store(qtbot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    tab = InversionTab(sidebar)
    qtbot.addWidget(tab)
    tab.add_measurement(_measurement())
    tab.measurement_list.setCurrentRow(0)

    tab._on_remove_clicked()

    assert tab.measurement_list.count() == 0
    assert tab._measurements == []


def test_run_fit_with_no_measurements_reports_error_without_crashing(qtbot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    tab = InversionTab(sidebar)
    qtbot.addWidget(tab)

    tab._on_run_clicked()

    assert "error" in tab.result_label.text()


def test_run_fit_calls_adapter_and_displays_result(qtbot, monkeypatch):
    fake_fit = FitResult(
        eps=4.5 - 0.045j, mu=1.0 + 0j, success=True, residual_norm=1e-8,
        n_measurements=1, covariance=None, condition_number=1.0, raw=None,
    )
    calls = []

    def fake_run_inversion(measurements, *, fit_mu=False, initial_guess=None):
        calls.append((measurements, fit_mu))
        return fake_fit

    monkeypatch.setattr(inversion_tab_module, "run_inversion", fake_run_inversion)

    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    tab = InversionTab(sidebar)
    qtbot.addWidget(tab)
    tab.add_measurement(_measurement())

    tab._on_run_clicked()
    qtbot.waitUntil(lambda: "success=True" in tab.result_label.text(), timeout=2000)

    assert len(calls) == 1
    assert len(calls[0][0]) == 1
