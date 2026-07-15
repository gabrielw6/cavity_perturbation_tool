"""widgets/tabs/inversion_tab.py -- Inversion tab (docs/gui_module_plan.md
Section 0.4/5). Consumes either pasted-in (f_meas, Q_meas) values (bound to
a fresh `PerturbationModel` built from the current sidebar state) or a
"use this result" measurement captured from a forward tab (already bound to
that tab's own model instance -- Section 2.4's widened `PerturbationModelLike`
is what lets a Ritz-/FDTD-backed measurement sit in the same list)."""
from __future__ import annotations

import functools

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cavity_perturbation.fields import AnalyticalField
from cavity_perturbation.inverse import FitResult, Measurement
from cavity_perturbation.perturbation import PerturbationModel
from cavity_perturbation_gui.adapters.cavity_adapter import build_cavity, resolve_rs
from cavity_perturbation_gui.adapters.inversion_runner import run_inversion
from cavity_perturbation_gui.adapters.sample_adapter import build_sample
from cavity_perturbation_gui.logging_bridge import get_logger
from cavity_perturbation_gui.widgets.sidebar import Sidebar
from cavity_perturbation_gui.widgets.tabs._common import RunBar


def _describe(measurement: Measurement) -> str:
    return f"{type(measurement.model).__name__}: f={measurement.f_meas:.6e} Hz, Q={measurement.Q_meas:.4e}"


class InversionTab(QWidget):
    def __init__(self, sidebar: Sidebar, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sidebar = sidebar
        self._measurements: list[Measurement] = []
        self._logger = get_logger()

        self.measurement_list = QListWidget()
        self.remove_button = QPushButton("Remove selected")

        self.f_meas_spin = QDoubleSpinBox()
        self.f_meas_spin.setRange(1.0, 1e15)
        self.f_meas_spin.setDecimals(3)
        self.Q_meas_spin = QDoubleSpinBox()
        self.Q_meas_spin.setRange(1.0, 1e12)
        self.Q_meas_spin.setDecimals(3)
        self.add_manual_button = QPushButton("Add pasted-in measurement")

        self.fit_mu_checkbox = QCheckBox("Fit mu too")

        self.run_bar = RunBar("Run Fit")
        self.result_label = QLabel("")

        manual_row = QHBoxLayout()
        manual_row.addWidget(QLabel("f_meas [Hz]"))
        manual_row.addWidget(self.f_meas_spin)
        manual_row.addWidget(QLabel("Q_meas"))
        manual_row.addWidget(self.Q_meas_spin)
        manual_row.addWidget(self.add_manual_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.measurement_list)
        layout.addWidget(self.remove_button)
        layout.addLayout(manual_row)
        layout.addWidget(self.fit_mu_checkbox)
        layout.addWidget(self.run_bar)
        layout.addWidget(self.result_label)

        self.add_manual_button.clicked.connect(self._on_add_manual_clicked)
        self.remove_button.clicked.connect(self._on_remove_clicked)
        self.run_bar.button.clicked.connect(self._on_run_clicked)

    def add_measurement(self, measurement: Measurement) -> None:
        """Slot for each forward tab's `measurement_captured` signal
        (Section 5.6) -- `main_window.py` connects all four."""
        self._measurements.append(measurement)
        self.measurement_list.addItem(QListWidgetItem(_describe(measurement)))
        self._logger.info("Inversion: measurement added (%s)", _describe(measurement))

    def _on_add_manual_clicked(self) -> None:
        try:
            cavity_params = self._sidebar.cavity_params()
            sample_params = self._sidebar.sample_params()
            cavity = build_cavity(cavity_params)
            field = AnalyticalField(cavity)
            sample = build_sample(cavity, field, sample_params)
            Rs = resolve_rs(cavity, rs=self._sidebar.rs(), conductivity=self._sidebar.conductivity())
            model = PerturbationModel(field, Rs_walls=Rs)
            measurement = Measurement(
                model=model,
                region=sample.region,
                f_meas=self.f_meas_spin.value(),
                Q_meas=self.Q_meas_spin.value(),
            )
        except Exception as exc:  # noqa: BLE001 -- Section 4: never let a bad parameter crash the app
            self._logger.error("Inversion: failed to add pasted-in measurement: %s", exc)
            self.result_label.setText(f"error: {exc}")
            return
        self.add_measurement(measurement)

    def _on_remove_clicked(self) -> None:
        row = self.measurement_list.currentRow()
        if row < 0:
            return
        self.measurement_list.takeItem(row)
        del self._measurements[row]

    def _on_run_clicked(self) -> None:
        if not self._measurements:
            self.result_label.setText("error: no measurements added")
            return
        measurements = list(self._measurements)
        fit_mu = self.fit_mu_checkbox.isChecked()
        build_call = lambda: functools.partial(run_inversion, measurements, fit_mu=fit_mu)
        self.run_bar.run(build_call, self._on_result, "Inversion")

    def _on_result(self, fit: FitResult) -> None:
        self.result_label.setText(
            f"eps = {fit.eps:.4g}    mu = {fit.mu:.4g}    success={fit.success}    "
            f"residual_norm={fit.residual_norm:.4g}    condition_number={fit.condition_number:.4g}"
        )
