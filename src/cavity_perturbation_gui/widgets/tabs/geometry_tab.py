"""widgets/tabs/geometry_tab.py -- a dedicated, full-size 3D view of the
cavity + sample geometry, live-updating directly from the sidebar's current
parameters. Unlike the four forward-solver tabs, this needs no Run action
or background thread: building the geometry primitives is cheap, closed-form
Module 1/3 construction, no field solve -- so it just redraws on every
`Sidebar.changed` signal, always showing the sample even if no forward tab
has been run yet (the shared sidebar-docked `GeometryView3D` only updates
per solver run, and shows no sample at all for the Analytical tab, which
never takes one)."""
from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget

from cavity_perturbation.fields import AnalyticalField
from cavity_perturbation_gui.adapters.cavity_adapter import build_cavity
from cavity_perturbation_gui.adapters.geometry_description import describe_cavity, describe_sample
from cavity_perturbation_gui.adapters.sample_adapter import build_sample
from cavity_perturbation_gui.widgets.geometry_view3d import GeometryView3D
from cavity_perturbation_gui.widgets.sidebar import Sidebar


class GeometryTab(QWidget):
    def __init__(self, sidebar: Sidebar, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sidebar = sidebar
        self.geometry_view = GeometryView3D()

        layout = QVBoxLayout(self)
        layout.addWidget(self.geometry_view)

        self._sidebar.changed.connect(self.redraw)
        self.redraw()

    def redraw(self) -> None:
        """Rebuild and redraw from the sidebar's current parameters. A
        mid-edit invalid parameter (e.g. a momentarily-empty spin box while
        typing) just leaves the last valid geometry on screen rather than
        clearing the view or raising -- this tab has no Run button/status
        label to surface an error against, and a live view flickering
        empty on every keystroke would be worse than a stale-but-valid one.
        """
        try:
            cavity_params = self._sidebar.cavity_params()
            sample_params = self._sidebar.sample_params()
            cavity = build_cavity(cavity_params)
            field = AnalyticalField(cavity)
            sample = build_sample(cavity, field, sample_params)
        except Exception:
            return
        self.geometry_view.set_geometry(describe_cavity(cavity), describe_sample(sample.region))
