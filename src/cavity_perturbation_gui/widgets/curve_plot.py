"""widgets/curve_plot.py -- PyQtGraph 2D curve plotting: resonance curves
(Lorentzian from f_calc/Q_calc, same construction as
scripts/simulate_perturbation.py's `plot_resonance_curves`/`lorentzian`) and,
for the FDTD tab, arbitrary x-y traces (excitation waveform, spectrum) --
docs/gui_module_plan.md Section 5."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

Array = np.ndarray


def lorentzian(f: Array, f_res: float, Q: float) -> Array:
    """Normalized Lorentzian response -- same formula as
    scripts/simulate_perturbation.py's `plot_resonance_curves`."""
    x = 2.0 * Q * (f - f_res) / f_res
    return 1.0 / (1.0 + x**2)


class CurvePlot(QWidget):
    """A single PyQtGraph plot, reused both for overlaid resonance-curve
    Lorentzians (one call to `plot_lorentzian` per solver result) and for
    arbitrary x-y traces (FDTD's excitation waveform / spectrum plots)."""

    def __init__(self, title: str = "", x_label: str = "", y_label: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.plot_widget = pg.PlotWidget(title=title)
        self.plot_widget.setLabel("bottom", x_label)
        self.plot_widget.setLabel("left", y_label)
        self.plot_widget.addLegend()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_widget)
        self._items: list[pg.GraphicsObject] = []

    def clear(self) -> None:
        self.plot_widget.clear()
        self.plot_widget.addLegend()
        self._items.clear()

    def plot_xy(self, x: Array, y: Array, label: str, color: str = "w") -> None:
        curve = self.plot_widget.plot(x, y, pen=pg.mkPen(color, width=2), name=label)
        self._items.append(curve)

    def plot_lorentzian(self, f_res: float, Q: float, label: str, color: str = "w", n_points: int = 2000) -> None:
        """If `Q` is finite, plots a normalized Lorentzian sized to show
        its own full linewidth; if `Q` is infinite (no loss at all), draws
        a vertical line at `f_res` instead -- there is no meaningful finite
        width to draw for a truly lossless resonance. Mirrors
        scripts/simulate_perturbation.py's `plot_resonance_curves` exactly.
        """
        if np.isfinite(Q):
            width = 4.0 * f_res / Q
            f = np.linspace(f_res - width, f_res + width, n_points)
            self.plot_xy(f, lorentzian(f, f_res, Q), f"{label} (Q={Q:.3g})", color)
        else:
            line = pg.InfiniteLine(
                pos=f_res,
                angle=90,
                pen=pg.mkPen(color, width=2, style=Qt.PenStyle.DashLine),
                label=f"{label} (Q=inf)",
            )
            self.plot_widget.addItem(line)
            self._items.append(line)
