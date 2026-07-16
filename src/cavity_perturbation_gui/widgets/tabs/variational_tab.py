"""widgets/tabs/variational_tab.py -- Rayleigh-Ritz (docs/gui_module_plan.md
Section 4/5). Field plot is the genuine Ritz reconstruction
E(r)=sum_i c_i E_i(r) (Section 2.3), not the unperturbed single-mode field
Analytical/Perturbational show."""
from __future__ import annotations

import functools

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget

from cavity_perturbation.inverse import Measurement
from cavity_perturbation_gui.adapters.field_sampling import Axis, plane_through_point, sample_ritz_field
from cavity_perturbation_gui.adapters.geometry_description import describe_cavity, describe_sample
from cavity_perturbation_gui.adapters.inversion_runner import measurement_from_result
from cavity_perturbation_gui.adapters.ritz_runner import RitzRunResult, run_ritz
from cavity_perturbation_gui.widgets.curve_plot import CurvePlot
from cavity_perturbation_gui.widgets.field_plane_view import FieldPlaneView, vector_magnitude
from cavity_perturbation_gui.widgets.geometry_view3d import GeometryView3D
from cavity_perturbation_gui.widgets.sidebar import Sidebar
from cavity_perturbation_gui.widgets.tabs._common import PlotColumn, RunBar


class VariationalTab(QWidget):
    measurement_captured = Signal(object)  # Measurement

    def __init__(self, sidebar: Sidebar, geometry_view: GeometryView3D, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sidebar = sidebar
        self._geometry_view = geometry_view
        self.last_result: RitzRunResult | None = None

        self.n_basis_spin = QSpinBox()
        self.n_basis_spin.setRange(1, 30)
        self.n_basis_spin.setValue(5)

        self.run_bar = RunBar("Run Variational (Ritz)")
        self.refresh_button = QPushButton("Refresh field view")
        self.summary_label = QLabel("")
        self.use_as_measurement_button = QPushButton("Use this result as a measurement")
        self.use_as_measurement_button.setEnabled(False)
        self.curve_plot = CurvePlot(title="Loaded resonance (Ritz)", x_label="frequency [Hz]", y_label="normalized response")
        self.field_view = FieldPlaneView()

        run_row = QHBoxLayout()
        run_row.addWidget(self.run_bar)
        run_row.addWidget(self.refresh_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.n_basis_spin)
        layout.addLayout(run_row)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.use_as_measurement_button)

        plots_row = QHBoxLayout()
        plots_row.addWidget(PlotColumn([self.curve_plot]), 1)
        plots_row.addWidget(self.field_view, 1)
        layout.addLayout(plots_row)

        self.run_bar.button.clicked.connect(self._on_run_clicked)
        self.refresh_button.clicked.connect(self._on_refresh_clicked)
        self.use_as_measurement_button.clicked.connect(self._on_use_as_measurement_clicked)
        self.field_view.plane2_axes_changed.connect(self._redraw_plane2)

    def _on_run_clicked(self) -> None:
        cavity_params = self._sidebar.cavity_params()
        sample_params = self._sidebar.sample_params()
        rs = self._sidebar.rs()
        conductivity = self._sidebar.conductivity()
        n_basis = self.n_basis_spin.value()
        build_call = lambda: functools.partial(
            run_ritz, cavity_params, sample_params, rs=rs, conductivity=conductivity, n_basis=n_basis
        )
        self.run_bar.run(build_call, self._on_result, "Variational")

    def _on_result(self, result: RitzRunResult) -> None:
        self.last_result = result
        self.use_as_measurement_button.setEnabled(True)
        r = result.result
        self.summary_label.setText(
            f"f_calc = {r.f_calc:.6e} Hz    Q_calc = {r.Q_calc:.4e}    basis size = {result.model.basis_size}"
        )

        self.curve_plot.clear()
        self.curve_plot.plot_lorentzian(r.f_calc, r.Q_calc, "loaded (Ritz)", "g")

        cavity = result.cavity
        self.field_view.set_note("Field plot is the genuine Ritz reconstruction E(r) = sum_i c_i E_i(r).")
        self._redraw_plane1()
        self._redraw_plane2(self.field_view.plane2.current_axes())

        self._geometry_view.set_geometry(describe_cavity(cavity), describe_sample(result.sample.region))

    def _on_refresh_clicked(self) -> None:
        """Re-issue both planes' draw calls against the cached last result,
        with no new solver run -- a manual escape hatch for pyqtgraph's own
        occasional stale-repaint quirks (docs/gui_module_plan.md Section 5)."""
        self._redraw_plane1()
        self._redraw_plane2(self.field_view.plane2.current_axes())

    def _redraw_plane1(self) -> None:
        if self.last_result is None:
            return
        cavity = self.last_result.cavity
        # SampleRegion doesn't declare `.center` on its ABC (Sphere/Cylinder/
        # Slab each have their own -- same duck-typing convention
        # perturbation.py's own evaluate() uses).
        center = getattr(self.last_result.sample.region, "center")
        plane_xy = plane_through_point(cavity, center, ("x", "y"))
        grid1, values1 = sample_ritz_field(cavity, self.last_result.diagnostics, plane_xy)
        self.field_view.plane1.set_title("|E| (x-y plane)")
        self.field_view.plane1.show_scalar_field(grid1, vector_magnitude(values1))

    def _redraw_plane2(self, axes: tuple[Axis, Axis]) -> None:
        """Resample plane2 at the currently-selected cross-section from the
        last result -- no new solver run needed, the Ritz coefficients are
        already in hand. A no-op before the first run (nothing to resample
        yet)."""
        if self.last_result is None:
            return
        cavity = self.last_result.cavity
        center = getattr(self.last_result.sample.region, "center")
        plane = plane_through_point(cavity, center, axes)
        grid, values = sample_ritz_field(cavity, self.last_result.diagnostics, plane)
        self.field_view.plane2.set_title(f"|E| ({axes[0]}-{axes[1]} plane)")
        self.field_view.plane2.show_scalar_field(grid, vector_magnitude(values))

    def _on_use_as_measurement_clicked(self) -> None:
        if self.last_result is None:
            return
        measurement: Measurement = measurement_from_result(
            self.last_result.model, self.last_result.sample.region, self.last_result.result
        )
        self.measurement_captured.emit(measurement)
