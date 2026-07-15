"""widgets/field_plane_view.py -- PyQtGraph image view, field cross-sections
(docs/gui_module_plan.md Section 5: "two perpendicular planes side by
side"). Only ever renders a real scalar array `adapters/field_sampling.py`
already produced (magnitude/component already selected by the caller) --
this widget has no field-physics knowledge of its own."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from cavity_perturbation_gui.adapters.field_sampling import PlaneGrid

Array = np.ndarray


def vector_magnitude(values: Array) -> Array:
    """(n1, n2, 3) complex or real vector field -> (n1, n2) real magnitude,
    for tabs whose field source is a vector (Analytical/Perturbational/
    Variational) before handing it to `show_scalar_field`. NaN-masked
    points (`adapters/field_sampling.py`'s outside-cavity convention)
    propagate through automatically -- `np.linalg.norm` of a vector with a
    NaN component is NaN, no separate handling needed."""
    return np.linalg.norm(values, axis=-1)


class _SinglePlaneView(QWidget):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title_label = QLabel(title)
        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._title_label)
        layout.addWidget(self.image_view)

    def set_title(self, text: str) -> None:
        self._title_label.setText(text)

    def show_scalar_field(self, grid: PlaneGrid, values: Array) -> None:
        """`values`: a real (n1, n2) array -- NaN-masked points (outside
        the cavity, `adapters/field_sampling.py`'s own convention) render
        as zero, since PyQtGraph's `ImageItem` has no native NaN handling."""
        image = np.nan_to_num(values, nan=0.0)
        n1, n2 = len(grid.axis1_values), len(grid.axis2_values)
        scale1 = (grid.axis1_values[-1] - grid.axis1_values[0]) / max(n1 - 1, 1)
        scale2 = (grid.axis2_values[-1] - grid.axis2_values[0]) / max(n2 - 1, 1)
        self.image_view.setImage(
            image,
            pos=(float(grid.axis1_values[0]), float(grid.axis2_values[0])),
            scale=(float(scale1), float(scale2)),
        )

    def clear(self) -> None:
        self.image_view.clear()


class FieldPlaneView(QWidget):
    """Two perpendicular planes side by side (Section 5), with a shared
    note label above both for Section 5's per-tab honesty callout (e.g.
    Perturbational's "unperturbed cavity field..." note)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._note_label = QLabel("")
        self._note_label.setWordWrap(True)
        self._note_label.setVisible(False)

        self.plane1 = _SinglePlaneView("Plane 1")
        self.plane2 = _SinglePlaneView("Plane 2")
        planes_row = QHBoxLayout()
        planes_row.addWidget(self.plane1)
        planes_row.addWidget(self.plane2)

        layout = QVBoxLayout(self)
        layout.addWidget(self._note_label)
        layout.addLayout(planes_row)

    def set_note(self, text: str) -> None:
        self._note_label.setText(text)
        self._note_label.setVisible(bool(text))

    def clear(self) -> None:
        self.plane1.clear()
        self.plane2.clear()
