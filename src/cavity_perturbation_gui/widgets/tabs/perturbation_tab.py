"""widgets/tabs/perturbation_tab.py -- Module 4 (docs/gui_module_plan.md
Section 4/5). Field plot is explicitly labeled as the unperturbed field --
Module 4 corrects the resonance with a single scalar inside the sample, not
a resolved internal field (Section 5's honesty note)."""
from __future__ import annotations

import functools

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from cavity_perturbation.inverse import Measurement
from cavity_perturbation_gui.adapters.field_sampling import plane_through_point, sample_closed_form_field
from cavity_perturbation_gui.adapters.geometry_description import describe_cavity, describe_sample
from cavity_perturbation_gui.adapters.inversion_runner import measurement_from_result
from cavity_perturbation_gui.adapters.perturbation_runner import PerturbationRunResult, run_perturbation
from cavity_perturbation_gui.widgets.curve_plot import CurvePlot
from cavity_perturbation_gui.widgets.field_plane_view import FieldPlaneView, vector_magnitude
from cavity_perturbation_gui.widgets.geometry_view3d import GeometryView3D
from cavity_perturbation_gui.widgets.sidebar import Sidebar
from cavity_perturbation_gui.widgets.tabs._common import RunBar

_FIELD_NOTE = (
    "Field plot shows the unperturbed cavity field -- the perturbation model corrects the "
    "resonance with a single scalar inside the sample, not a resolved internal field."
)


class PerturbationTab(QWidget):
    measurement_captured = Signal(object)  # Measurement

    def __init__(self, sidebar: Sidebar, geometry_view: GeometryView3D, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sidebar = sidebar
        self._geometry_view = geometry_view
        self.last_result: PerturbationRunResult | None = None

        self.run_bar = RunBar("Run Perturbational")
        self.summary_label = QLabel("")
        self.use_as_measurement_button = QPushButton("Use this result as a measurement")
        self.use_as_measurement_button.setEnabled(False)
        self.curve_plot = CurvePlot(title="Loaded resonance (Module 4)", x_label="frequency [Hz]", y_label="normalized response")
        self.field_view = FieldPlaneView()

        layout = QVBoxLayout(self)
        layout.addWidget(self.run_bar)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.use_as_measurement_button)
        layout.addWidget(self.curve_plot)
        layout.addWidget(self.field_view)

        self.run_bar.button.clicked.connect(self._on_run_clicked)
        self.use_as_measurement_button.clicked.connect(self._on_use_as_measurement_clicked)

    def _on_run_clicked(self) -> None:
        cavity_params = self._sidebar.cavity_params()
        sample_params = self._sidebar.sample_params()
        rs = self._sidebar.rs()
        conductivity = self._sidebar.conductivity()
        build_call = lambda: functools.partial(
            run_perturbation, cavity_params, sample_params, rs=rs, conductivity=conductivity
        )
        self.run_bar.run(build_call, self._on_result, "Perturbational")

    def _on_result(self, result: PerturbationRunResult) -> None:
        self.last_result = result
        self.use_as_measurement_button.setEnabled(True)
        r = result.result
        self.summary_label.setText(f"f_calc = {r.f_calc:.6e} Hz    Q_calc = {r.Q_calc:.4e}")

        self.curve_plot.clear()
        self.curve_plot.plot_lorentzian(r.f_calc, r.Q_calc, "loaded", "m")

        cavity = result.cavity
        # SampleRegion doesn't declare `.center` on its ABC (Sphere/Cylinder/
        # Slab each have their own -- same duck-typing convention
        # perturbation.py's own evaluate() uses).
        center = getattr(result.sample.region, "center")
        plane_xy = plane_through_point(cavity, center, ("x", "y"))
        plane_xz = plane_through_point(cavity, center, ("x", "z"))
        grid1, values1 = sample_closed_form_field(cavity, result.field_provider, plane_xy, field="E")
        grid2, values2 = sample_closed_form_field(cavity, result.field_provider, plane_xz, field="E")
        self.field_view.set_note(_FIELD_NOTE)
        self.field_view.plane1.set_title("|E| (x-y plane)")
        self.field_view.plane1.show_scalar_field(grid1, vector_magnitude(values1))
        self.field_view.plane2.set_title("|E| (x-z plane)")
        self.field_view.plane2.show_scalar_field(grid2, vector_magnitude(values2))

        self._geometry_view.set_geometry(describe_cavity(cavity), describe_sample(result.sample.region))

    def _on_use_as_measurement_clicked(self) -> None:
        if self.last_result is None:
            return
        measurement: Measurement = measurement_from_result(
            self.last_result.model, self.last_result.sample.region, self.last_result.result
        )
        self.measurement_captured.emit(measurement)
