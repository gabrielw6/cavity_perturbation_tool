"""widgets/tabs/analytical_tab.py -- Module 1/2 only, no sample
(docs/gui_module_plan.md Section 4/5). Shows the cavity's own closed-form
empty-cavity resonance and field, nothing about a sample's effect."""
from __future__ import annotations

import functools

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from cavity_perturbation_gui.adapters.analytical_runner import AnalyticalRunResult, run_analytical
from cavity_perturbation_gui.adapters.field_sampling import plane_through_point, sample_closed_form_field
from cavity_perturbation_gui.adapters.geometry_description import describe_cavity
from cavity_perturbation_gui.widgets.curve_plot import CurvePlot
from cavity_perturbation_gui.widgets.field_plane_view import FieldPlaneView, vector_magnitude
from cavity_perturbation_gui.widgets.geometry_view3d import GeometryView3D
from cavity_perturbation_gui.widgets.sidebar import Sidebar
from cavity_perturbation_gui.widgets.tabs._common import RunBar


class AnalyticalTab(QWidget):
    def __init__(self, sidebar: Sidebar, geometry_view: GeometryView3D, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sidebar = sidebar
        self._geometry_view = geometry_view
        self.last_result: AnalyticalRunResult | None = None

        self.run_bar = RunBar("Run Analytical")
        self.summary_label = QLabel("")
        self.curve_plot = CurvePlot(title="Empty-cavity resonance", x_label="frequency [Hz]", y_label="normalized response")
        self.field_view = FieldPlaneView()

        layout = QVBoxLayout(self)
        layout.addWidget(self.run_bar)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.curve_plot)
        layout.addWidget(self.field_view)

        self.run_bar.button.clicked.connect(self._on_run_clicked)

    def _on_run_clicked(self) -> None:
        cavity_params = self._sidebar.cavity_params()
        rs = self._sidebar.rs()
        conductivity = self._sidebar.conductivity()
        build_call = lambda: functools.partial(run_analytical, cavity_params, rs=rs, conductivity=conductivity)
        self.run_bar.run(build_call, self._on_result, "Analytical")

    def _on_result(self, result: AnalyticalRunResult) -> None:
        self.last_result = result
        r = result.result
        self.summary_label.setText(f"f0 = {r.f_calc:.6e} Hz    Q_wall = {r.Q_calc:.4e}")

        self.curve_plot.clear()
        self.curve_plot.plot_lorentzian(r.f_calc, r.Q_calc, "empty cavity", "c")

        cavity = result.cavity
        center = (cavity.bounding_box()[0] + cavity.bounding_box()[1]) / 2.0
        plane_xy = plane_through_point(cavity, center, ("x", "y"))
        plane_xz = plane_through_point(cavity, center, ("x", "z"))
        grid1, values1 = sample_closed_form_field(cavity, result.field_provider, plane_xy, field="E")
        grid2, values2 = sample_closed_form_field(cavity, result.field_provider, plane_xz, field="E")
        self.field_view.plane1.set_title("|E| (x-y plane)")
        self.field_view.plane1.show_scalar_field(grid1, vector_magnitude(values1))
        self.field_view.plane2.set_title("|E| (x-z plane)")
        self.field_view.plane2.show_scalar_field(grid2, vector_magnitude(values2))

        self._geometry_view.set_geometry(describe_cavity(cavity), None)
