"""widgets/field_plane_view.py -- PyQtGraph image view, field cross-sections
(docs/gui_module_plan.md Section 5: two planes stacked one below the other
on the tab's right-hand side). Only ever renders a real scalar array
`adapters/field_sampling.py` already produced (magnitude/component already
selected by the caller) -- this widget has no field-physics knowledge of
its own."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from cavity_perturbation_gui.adapters.field_sampling import Axis, PlaneGrid

Array = np.ndarray

#: The second plane's selectable cross-sections -- always through the same
#: point plane1 already fixes, only which two axes are free changes.
PLANE2_AXIS_OPTIONS: tuple[tuple[str, tuple[Axis, Axis]], ...] = (
    ("x-z plane", ("x", "z")),
    ("y-z plane", ("y", "z")),
)


def vector_magnitude(values: Array) -> Array:
    """(n1, n2, 3) complex or real vector field -> (n1, n2) real magnitude,
    for tabs whose field source is a vector (Analytical/Perturbational/
    Variational) before handing it to `show_scalar_field`. NaN-masked
    points (`adapters/field_sampling.py`'s outside-cavity convention)
    propagate through automatically -- `np.linalg.norm` of a vector with a
    NaN component is NaN, no separate handling needed."""
    return np.linalg.norm(values, axis=-1)


class _SinglePlaneView(QWidget):
    #: Only emitted by a view constructed with `axis_options` -- the
    #: newly-selected `(free_axis_1, free_axis_2)` pair.
    axes_changed = Signal(tuple)

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
        axis_options: tuple[tuple[str, tuple[Axis, Axis]], ...] | None = None,
    ) -> None:
        super().__init__(parent)
        self._title_label = QLabel(title)
        self.axis_combo: QComboBox | None = None

        header = QHBoxLayout()
        header.addWidget(self._title_label)
        if axis_options is not None:
            self.axis_combo = QComboBox()
            for label, axes in axis_options:
                self.axis_combo.addItem(label, userData=axes)
            self.axis_combo.currentIndexChanged.connect(self._on_axis_combo_changed)
            header.addWidget(self.axis_combo)
        header.addStretch(1)

        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        # Before any real data exists, `pg.ImageView()`'s own default state
        # is a blank black image paired with a meaningless (0, 1.0)
        # histogram/scale bar -- easy to mistake for "the field isn't being
        # shown" (a real user bug report). Hide the histogram until the
        # first `show_scalar_field()` call actually has real levels to show.
        self.image_view.ui.histogram.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(header)
        layout.addWidget(self.image_view)

    def _on_axis_combo_changed(self, index: int) -> None:
        assert self.axis_combo is not None
        self.axes_changed.emit(self.axis_combo.itemData(index))

    def current_axes(self) -> tuple[Axis, Axis]:
        """The plane's currently-selected free axes -- `("x", "y")` for a
        view with no selector (plane1 is always the x-y plane)."""
        if self.axis_combo is None:
            return ("x", "y")
        axes: tuple[Axis, Axis] = self.axis_combo.currentData()
        return axes

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
        self.image_view.ui.histogram.show()
        self.image_view.setImage(
            image,
            pos=(float(grid.axis1_values[0]), float(grid.axis2_values[0])),
            scale=(float(scale1), float(scale2)),
        )
        # setImage's own autoRange isn't reliable with a custom pos/scale
        # transform, and a prior manual zoom/pan would otherwise persist
        # into new data -- pin the view to the plane's own extent every time.
        view = self.image_view.getView()
        view.setRange(
            xRange=(float(grid.axis1_values[0]), float(grid.axis1_values[-1])),
            yRange=(float(grid.axis2_values[0]), float(grid.axis2_values[-1])),
            padding=0,
        )

    def clear(self) -> None:
        self.image_view.clear()
        self.image_view.ui.histogram.hide()


class FieldPlaneView(QWidget):
    """Two perpendicular planes stacked one below the other -- meant to sit
    on a forward tab's right-hand side, beside its 2D plots (Section 5) --
    with a shared note label above both for Section 5's per-tab honesty
    callout (e.g. Perturbational's "unperturbed cavity field..." note).
    `plane1` is always the x-y plane; `plane2` additionally carries a
    dropdown letting the user pick x-z or y-z, since one fixed second plane
    can't show every cross-section a differently-shaped/oriented sample
    might need."""

    #: Forwards `plane2`'s own `axes_changed` -- the tab reacts by
    #: resampling and redrawing plane2 from its last result, with no new
    #: solver run needed (the field/diagnostics are already in hand).
    plane2_axes_changed = Signal(tuple)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._note_label = QLabel("")
        self._note_label.setWordWrap(True)
        self._note_label.setVisible(False)

        self.plane1 = _SinglePlaneView("Plane 1 (x-y plane)")
        self.plane2 = _SinglePlaneView("Plane 2", axis_options=PLANE2_AXIS_OPTIONS)
        self.plane2.axes_changed.connect(self.plane2_axes_changed)
        planes_col = QVBoxLayout()
        planes_col.addWidget(self.plane1)
        planes_col.addWidget(self.plane2)

        layout = QVBoxLayout(self)
        layout.addWidget(self._note_label)
        layout.addLayout(planes_col)

    def set_note(self, text: str) -> None:
        self._note_label.setText(text)
        self._note_label.setVisible(bool(text))

    def clear(self) -> None:
        self.plane1.clear()
        self.plane2.clear()
