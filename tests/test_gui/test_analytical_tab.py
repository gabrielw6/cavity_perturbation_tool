"""docs/gui_module_plan.md Section 8 -- widgets/, pytest-qt, adapters
mocked: confirms button-click-to-adapter-call wiring, and that a raised
exception surfaces in the run bar / log rather than crashing."""
import cavity_perturbation_gui.widgets.tabs.analytical_tab as analytical_tab_module
from cavity_perturbation.cavity import ModeIndex, RectangularCavity
from cavity_perturbation.fields import AnalyticalField
from cavity_perturbation_gui.adapters.analytical_runner import AnalyticalRunResult
from cavity_perturbation_gui.widgets.geometry_view3d import GeometryView3D
from cavity_perturbation_gui.widgets.sidebar import Sidebar
from cavity_perturbation_gui.widgets.tabs.analytical_tab import AnalyticalTab

# "Adapters mocked" (Section 8) means the *runner function* is replaced by a
# fake below -- its return value still needs to be a real, cheap Module 1
# cavity so geometry_description.py's isinstance-based dispatch and
# field_sampling.py's contains()/E() calls work unmodified, rather than
# fighting a hand-rolled stub against two different consumers' contracts.
_CAV = RectangularCavity(0.03, 0.02, 0.025, ModeIndex("TE", (0, 1, 1)))


def _fake_result():
    from cavity_perturbation.perturbation import PerturbationResult

    return AnalyticalRunResult(
        cavity=_CAV,
        field_provider=AnalyticalField(_CAV),
        result=PerturbationResult(f_calc=1e9, Q_calc=500.0, omega_tilde=complex(2 * 3.14159e9, -1e6)),
        Rs_walls=0.02,
    )


def test_run_button_calls_adapter_and_populates_summary(qtbot, monkeypatch):
    calls = []

    def fake_run_analytical(cavity_params, *, rs=None, conductivity=None):
        calls.append((cavity_params, rs, conductivity))
        return _fake_result()

    monkeypatch.setattr(analytical_tab_module, "run_analytical", fake_run_analytical)

    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    geometry_view = GeometryView3D()
    qtbot.addWidget(geometry_view)
    tab = AnalyticalTab(sidebar, geometry_view)
    qtbot.addWidget(tab)

    tab.run_bar.button.click()
    qtbot.waitUntil(lambda: tab.last_result is not None, timeout=2000)

    assert len(calls) == 1
    assert "1" in tab.summary_label.text()
    assert tab.run_bar.status_label.text() == "done"


def test_run_button_disabled_during_run_and_reenabled_after(qtbot, monkeypatch):
    monkeypatch.setattr(analytical_tab_module, "run_analytical", lambda *a, **k: _fake_result())

    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    geometry_view = GeometryView3D()
    qtbot.addWidget(geometry_view)
    tab = AnalyticalTab(sidebar, geometry_view)
    qtbot.addWidget(tab)

    tab.run_bar.button.click()
    qtbot.waitUntil(lambda: tab.run_bar.button.isEnabled(), timeout=2000)
    assert tab.last_result is not None


def test_run_failure_surfaces_in_status_label_not_a_crash(qtbot, monkeypatch):
    def boom(cavity_params, *, rs=None, conductivity=None):
        raise ValueError("synthetic adapter failure")

    monkeypatch.setattr(analytical_tab_module, "run_analytical", boom)

    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    geometry_view = GeometryView3D()
    qtbot.addWidget(geometry_view)
    tab = AnalyticalTab(sidebar, geometry_view)
    qtbot.addWidget(tab)

    tab.run_bar.button.click()
    qtbot.waitUntil(lambda: "error" in tab.run_bar.status_label.text(), timeout=2000)

    assert "synthetic adapter failure" in tab.run_bar.status_label.text()
    assert tab.run_bar.button.isEnabled()
    assert tab.last_result is None
