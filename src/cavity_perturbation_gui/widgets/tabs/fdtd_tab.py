"""widgets/tabs/fdtd_tab.py -- time-domain FDTD (docs/gui_module_plan.md
Section 4/5). Two extra curve_plot.py instances beyond the other three
tabs: excitation waveform (time domain) and its spectrum, reusing
FDTDDiagnostics's own arrays (Section 2.2) rather than recomputing an FFT
in the GUI. Field plot is one Cartesian component of the single
end-of-excitation snapshot (Section 0.3), not a full vector field."""
from __future__ import annotations

import functools

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDoubleSpinBox, QLabel, QPushButton, QVBoxLayout, QWidget

from cavity_perturbation.inverse import Measurement
from cavity_perturbation_gui.adapters.field_sampling import plane_through_point, sample_fdtd_snapshot
from cavity_perturbation_gui.adapters.fdtd_runner import FDTDRunResult, run_fdtd
from cavity_perturbation_gui.adapters.geometry_description import describe_cavity, describe_sample
from cavity_perturbation_gui.adapters.inversion_runner import measurement_from_result
from cavity_perturbation_gui.widgets.curve_plot import CurvePlot
from cavity_perturbation_gui.widgets.field_plane_view import FieldPlaneView
from cavity_perturbation_gui.widgets.geometry_view3d import GeometryView3D
from cavity_perturbation_gui.widgets.sidebar import Sidebar
from cavity_perturbation_gui.widgets.tabs._common import RunBar

_FIELD_NOTE = "Field plot is one Cartesian component of the single end-of-excitation field snapshot."


class FDTDTab(QWidget):
    measurement_captured = Signal(object)  # Measurement

    def __init__(self, sidebar: Sidebar, geometry_view: GeometryView3D, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sidebar = sidebar
        self._geometry_view = geometry_view
        self.last_result: FDTDRunResult | None = None

        self.cells_per_wavelength_spin = QDoubleSpinBox()
        self.cells_per_wavelength_spin.setRange(2.0, 200.0)
        self.cells_per_wavelength_spin.setValue(20.0)

        self.run_bar = RunBar("Run FDTD")
        self.summary_label = QLabel("")
        self.use_as_measurement_button = QPushButton("Use this result as a measurement")
        self.use_as_measurement_button.setEnabled(False)

        self.curve_plot = CurvePlot(title="Loaded resonance (FDTD)", x_label="frequency [Hz]", y_label="normalized response")
        self.excitation_plot = CurvePlot(title="Excitation waveform", x_label="time [s]", y_label="amplitude")
        self.spectrum_plot = CurvePlot(title="Ringdown spectrum", x_label="frequency [Hz]", y_label="power")
        self.field_view = FieldPlaneView()

        layout = QVBoxLayout(self)
        layout.addWidget(self.cells_per_wavelength_spin)
        layout.addWidget(self.run_bar)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.use_as_measurement_button)
        layout.addWidget(self.curve_plot)
        layout.addWidget(self.excitation_plot)
        layout.addWidget(self.spectrum_plot)
        layout.addWidget(self.field_view)

        self.run_bar.button.clicked.connect(self._on_run_clicked)
        self.use_as_measurement_button.clicked.connect(self._on_use_as_measurement_clicked)

    def _on_run_clicked(self) -> None:
        cavity_params = self._sidebar.cavity_params()
        sample_params = self._sidebar.sample_params()
        rs = self._sidebar.rs()
        conductivity = self._sidebar.conductivity()
        cells_per_wavelength = self.cells_per_wavelength_spin.value()
        build_call = lambda: functools.partial(
            run_fdtd,
            cavity_params,
            sample_params,
            rs=rs,
            conductivity=conductivity,
            cells_per_wavelength=cells_per_wavelength,
        )
        self.run_bar.run(build_call, self._on_result, "FDTD")

    def _on_result(self, result: FDTDRunResult) -> None:
        self.last_result = result
        self.use_as_measurement_button.setEnabled(True)
        r = result.result
        self.summary_label.setText(f"f_calc = {r.f_calc:.6e} Hz    Q_calc = {r.Q_calc:.4e}")

        self.curve_plot.clear()
        self.curve_plot.plot_lorentzian(r.f_calc, r.Q_calc, "loaded (FDTD)", "r")

        diag = result.diagnostics
        self.excitation_plot.clear()
        self.excitation_plot.plot_xy(diag.excitation_times, diag.excitation_waveform, "excitation", "y")

        self.spectrum_plot.clear()
        if diag.spectrum_freqs is not None and diag.spectrum_power is not None:
            self.spectrum_plot.plot_xy(diag.spectrum_freqs, diag.spectrum_power, "ringdown spectrum", "y")

        cavity = result.cavity
        # SampleRegion doesn't declare `.center` on its ABC (Sphere/Cylinder/
        # Slab each have their own -- same duck-typing convention
        # perturbation.py's own evaluate() uses).
        center = getattr(result.sample.region, "center")
        plane_xy = plane_through_point(cavity, center, ("x", "y"))
        plane_xz = plane_through_point(cavity, center, ("x", "z"))
        grid1, values1 = sample_fdtd_snapshot(cavity, diag, plane_xy, component="Ex")
        grid2, values2 = sample_fdtd_snapshot(cavity, diag, plane_xz, component="Ex")
        self.field_view.set_note(_FIELD_NOTE)
        self.field_view.plane1.set_title("Ex (x-y plane)")
        self.field_view.plane1.show_scalar_field(grid1, values1)
        self.field_view.plane2.set_title("Ex (x-z plane)")
        self.field_view.plane2.show_scalar_field(grid2, values2)

        self._geometry_view.set_geometry(describe_cavity(cavity), describe_sample(result.sample.region))

    def _on_use_as_measurement_clicked(self) -> None:
        if self.last_result is None:
            return
        measurement: Measurement = measurement_from_result(
            self.last_result.model, self.last_result.sample.region, self.last_result.result
        )
        self.measurement_captured.emit(measurement)
