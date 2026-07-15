"""widgets/tabs/fdtd_tab.py -- time-domain FDTD (docs/gui_module_plan.md
Section 4/5). Two extra curve_plot.py instances beyond the other three
tabs: excitation waveform (time domain) and its spectrum, reusing
FDTDDiagnostics's own arrays (Section 2.2) rather than recomputing an FFT
in the GUI. Field plot is one Cartesian component of the single
end-of-excitation snapshot (Section 0.3), not a full vector field."""
from __future__ import annotations

import functools
import threading

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cavity_perturbation.fdtd import FDTDCancelled
from cavity_perturbation.inverse import Measurement
from cavity_perturbation_gui.adapters.field_sampling import Axis, plane_through_point, sample_fdtd_snapshot
from cavity_perturbation_gui.adapters.fdtd_runner import FDTDRunResult, run_fdtd
from cavity_perturbation_gui.adapters.geometry_description import describe_cavity, describe_sample
from cavity_perturbation_gui.adapters.inversion_runner import measurement_from_result
from cavity_perturbation_gui.widgets.curve_plot import CurvePlot
from cavity_perturbation_gui.widgets.field_plane_view import FieldPlaneView
from cavity_perturbation_gui.widgets.geometry_view3d import GeometryView3D
from cavity_perturbation_gui.widgets.sidebar import Sidebar
from cavity_perturbation_gui.widgets.tabs._common import PlotColumn, RunBar

_FIELD_NOTE = "Field plot is one Cartesian component of the single end-of-excitation field snapshot."


class _ProgressRelay(QObject):
    """Owned by the tab (main thread); `run_fdtd`'s `progress_callback` is
    handed `self.progress.emit` directly, called from the background
    worker thread -- Qt auto-queues the emission to this object's own
    (main-thread) home thread, so no changes to `SolveWorker`/
    `run_in_background` are needed for cross-thread progress updates."""

    progress = Signal(int, int)  # current_step, total_steps


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
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self._progress_relay = _ProgressRelay()
        self._progress_relay.progress.connect(self._on_progress)
        # threading.Event, not a Qt signal -- `cancel_check` is *polled*
        # from the background worker thread inside FDTDModel's step loops
        # (docs/gui_module_plan.md Section 6), a plain thread-safe flag read
        # with no need for Qt's cross-thread queued-signal machinery.
        self._cancel_event = threading.Event()

        self.summary_label = QLabel("")
        self.use_as_measurement_button = QPushButton("Use this result as a measurement")
        self.use_as_measurement_button.setEnabled(False)

        self.curve_plot = CurvePlot(title="Loaded resonance (FDTD)", x_label="frequency [Hz]", y_label="normalized response")
        self.excitation_plot = CurvePlot(title="Excitation waveform", x_label="time [s]", y_label="amplitude")
        self.spectrum_plot = CurvePlot(title="Ringdown spectrum", x_label="frequency [Hz]", y_label="power")
        self.field_view = FieldPlaneView()

        run_row = QHBoxLayout()
        run_row.addWidget(self.run_bar)
        run_row.addWidget(self.stop_button)
        run_row.addWidget(self.progress_bar)

        layout = QVBoxLayout(self)
        layout.addWidget(self.cells_per_wavelength_spin)
        layout.addLayout(run_row)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.use_as_measurement_button)

        # Left: the tab's three 2D plots, stacked one below the other in a
        # scrollable column (Section 5) so they get a sane minimum height
        # each instead of shrinking to nothing; right: the two field planes.
        plots_row = QHBoxLayout()
        plots_row.addWidget(PlotColumn([self.curve_plot, self.excitation_plot, self.spectrum_plot]), 1)
        plots_row.addWidget(self.field_view, 1)
        layout.addLayout(plots_row)

        self.run_bar.button.clicked.connect(self._on_run_clicked)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.use_as_measurement_button.clicked.connect(self._on_use_as_measurement_clicked)
        self.field_view.plane2_axes_changed.connect(self._redraw_plane2)

    def _on_run_clicked(self) -> None:
        cavity_params = self._sidebar.cavity_params()
        sample_params = self._sidebar.sample_params()
        rs = self._sidebar.rs()
        conductivity = self._sidebar.conductivity()
        cells_per_wavelength = self.cells_per_wavelength_spin.value()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self._cancel_event = threading.Event()  # fresh token per run
        build_call = lambda: functools.partial(
            run_fdtd,
            cavity_params,
            sample_params,
            rs=rs,
            conductivity=conductivity,
            cells_per_wavelength=cells_per_wavelength,
            progress_callback=self._progress_relay.progress.emit,
            cancel_check=self._cancel_event.is_set,
        )
        self.run_bar.run(build_call, self._on_result, "FDTD", on_failure=self._on_run_failed)
        self.stop_button.setEnabled(True)

    def _on_stop_clicked(self) -> None:
        self._cancel_event.set()
        self.stop_button.setEnabled(False)

    def _on_run_failed(self, exc: BaseException) -> None:
        self.stop_button.setEnabled(False)
        if isinstance(exc, FDTDCancelled):
            self.summary_label.setText("Run cancelled by user.")

    def _on_progress(self, current_step: int, total_steps: int) -> None:
        self.progress_bar.setRange(0, total_steps)
        self.progress_bar.setValue(current_step)

    def _on_result(self, result: FDTDRunResult) -> None:
        self.last_result = result
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(self.progress_bar.maximum())
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
            self.spectrum_plot.zoom_x_to_peak(diag.spectrum_freqs, diag.spectrum_power)

        cavity = result.cavity
        # SampleRegion doesn't declare `.center` on its ABC (Sphere/Cylinder/
        # Slab each have their own -- same duck-typing convention
        # perturbation.py's own evaluate() uses).
        center = getattr(result.sample.region, "center")
        plane_xy = plane_through_point(cavity, center, ("x", "y"))
        grid1, values1 = sample_fdtd_snapshot(cavity, diag, plane_xy, component="Ex")
        self.field_view.set_note(_FIELD_NOTE)
        self.field_view.plane1.set_title("Ex (x-y plane)")
        self.field_view.plane1.show_scalar_field(grid1, values1)
        self._redraw_plane2(self.field_view.plane2.current_axes())

        self._geometry_view.set_geometry(describe_cavity(cavity), describe_sample(result.sample.region))

    def _redraw_plane2(self, axes: tuple[Axis, Axis]) -> None:
        """Resample plane2 at the currently-selected cross-section from the
        last result's snapshot -- no new solver run needed. A no-op before
        the first run (nothing to resample yet)."""
        if self.last_result is None:
            return
        cavity = self.last_result.cavity
        center = getattr(self.last_result.sample.region, "center")
        plane = plane_through_point(cavity, center, axes)
        grid, values = sample_fdtd_snapshot(cavity, self.last_result.diagnostics, plane, component="Ex")
        self.field_view.plane2.set_title(f"Ex ({axes[0]}-{axes[1]} plane)")
        self.field_view.plane2.show_scalar_field(grid, values)

    def _on_use_as_measurement_clicked(self) -> None:
        if self.last_result is None:
            return
        measurement: Measurement = measurement_from_result(
            self.last_result.model, self.last_result.sample.region, self.last_result.result
        )
        self.measurement_captured.emit(measurement)
