"""widgets/main_window.py -- tabs + sidebar + log bar (docs/gui_module_plan.md
Section 5). Wires the four forward tabs' `measurement_captured` signal to
the Inversion tab (Section 5.6) -- the one place this package lets a tab
reach another, kept in the window rather than having tabs reference each
other directly."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QHBoxLayout, QMainWindow, QSplitter, QTabWidget, QWidget

from cavity_perturbation_gui.logging_bridge import install_logging_bridge
from cavity_perturbation_gui.widgets.geometry_view3d import GeometryView3D
from cavity_perturbation_gui.widgets.log_panel import LogPanel
from cavity_perturbation_gui.widgets.sidebar import Sidebar
from cavity_perturbation_gui.widgets.tabs.analytical_tab import AnalyticalTab
from cavity_perturbation_gui.widgets.tabs.fdtd_tab import FDTDTab
from cavity_perturbation_gui.widgets.tabs.geometry_tab import GeometryTab
from cavity_perturbation_gui.widgets.tabs.inversion_tab import InversionTab
from cavity_perturbation_gui.widgets.tabs.perturbation_tab import PerturbationTab
from cavity_perturbation_gui.widgets.tabs.variational_tab import VariationalTab

_WINDOW_TITLE = "Cavity Perturbation Measurement Suite"


class MainWindow(QMainWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_WINDOW_TITLE)

        self.log_handler = install_logging_bridge()
        self.log_panel = LogPanel()
        self.log_handler.record_logged.connect(self.log_panel.append_line)

        self.sidebar = Sidebar()
        self.geometry_view = GeometryView3D()

        self.analytical_tab = AnalyticalTab(self.sidebar, self.geometry_view)
        self.perturbation_tab = PerturbationTab(self.sidebar, self.geometry_view)
        self.variational_tab = VariationalTab(self.sidebar, self.geometry_view)
        self.fdtd_tab = FDTDTab(self.sidebar, self.geometry_view)
        self.geometry_tab = GeometryTab(self.sidebar)
        self.inversion_tab = InversionTab(self.sidebar)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.analytical_tab, "Analytical")
        self.tabs.addTab(self.perturbation_tab, "Perturbational")
        self.tabs.addTab(self.variational_tab, "Variational (Ritz)")
        self.tabs.addTab(self.fdtd_tab, "FDTD")
        self.tabs.addTab(self.geometry_tab, "3D Geometry")
        self.tabs.addTab(self.inversion_tab, "Inversion")

        for tab in (self.perturbation_tab, self.variational_tab, self.fdtd_tab):
            tab.measurement_captured.connect(self.inversion_tab.add_measurement)

        left_right_splitter = QSplitter()
        sidebar_and_geometry = QSplitter()
        sidebar_and_geometry.setOrientation(Qt.Orientation.Vertical)
        sidebar_and_geometry.addWidget(self.sidebar)
        sidebar_and_geometry.addWidget(self.geometry_view)
        left_right_splitter.addWidget(sidebar_and_geometry)
        left_right_splitter.addWidget(self.tabs)
        left_right_splitter.setStretchFactor(1, 1)

        central = QWidget()
        central_layout = QHBoxLayout(central)
        central_layout.addWidget(left_right_splitter)
        self.setCentralWidget(central)

        log_dock = self._make_log_dock()
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, log_dock)

    def _make_log_dock(self) -> QDockWidget:
        dock = QDockWidget("Log")
        dock.setWidget(self.log_panel)
        dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        return dock
