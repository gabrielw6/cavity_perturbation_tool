"""widgets/sidebar.py -- cavity type + dimensions + mode indices +
background eps_r/mu_r + Rs source; sample shape + position + orientation +
material -- one shared parameter model every tab reads (docs/gui_module_plan.md
Section 5). Emits `changed` on any edit, so `geometry_view3d.py` can redraw
live; each forward tab reads `.cavity_params()`/`.sample_params()`/
`.rs()`/`.conductivity()` on its own Run action rather than subscribing to
every keystroke.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QRadioButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from cavity_perturbation_gui.adapters.cavity_adapter import CavityParams, CavityType
from cavity_perturbation_gui.adapters.sample_adapter import Orientation, SampleParams, SampleShape

_DIM_RANGE = (1e-6, 10.0)  # meters
_DIM_DECIMALS = 6
_EPS_MU_RANGE = (0.01, 1000.0)
_LOSS_TANGENT_RANGE = (0.0, 10.0)
_CONDUCTIVITY_RANGE = (1.0, 1e9)
_RS_RANGE = (0.0, 1e6)


def _dim_spinbox(default: float) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(*_DIM_RANGE)
    box.setDecimals(_DIM_DECIMALS)
    box.setSingleStep(1e-3)
    box.setValue(default)
    return box


def _parse_indices(text: str) -> tuple[int, ...]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


class _RectangularPage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.a = _dim_spinbox(0.03)
        self.b = _dim_spinbox(0.03)
        self.c = _dim_spinbox(0.03)
        layout = QFormLayout(self)
        layout.addRow("a [m]", self.a)
        layout.addRow("b [m]", self.b)
        layout.addRow("c [m]", self.c)

    def dimensions(self) -> tuple[float, ...]:
        return (self.a.value(), self.b.value(), self.c.value())


class _CylindricalPage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.radius = _dim_spinbox(0.02)
        self.length = _dim_spinbox(0.03)
        layout = QFormLayout(self)
        layout.addRow("radius [m]", self.radius)
        layout.addRow("length [m]", self.length)

    def dimensions(self) -> tuple[float, ...]:
        return (self.radius.value(), self.length.value())


class _CoaxialPage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.r_inner = _dim_spinbox(0.01)
        self.r_outer = _dim_spinbox(0.023)
        self.length = _dim_spinbox(0.5)
        layout = QFormLayout(self)
        layout.addRow("r_inner [m]", self.r_inner)
        layout.addRow("r_outer [m]", self.r_outer)
        layout.addRow("length [m]", self.length)

    def dimensions(self) -> tuple[float, ...]:
        return (self.r_inner.value(), self.r_outer.value(), self.length.value())


_CAVITY_PAGES: tuple[tuple[CavityType, str], ...] = (
    ("rectangular", "Rectangular"),
    ("cylindrical", "Cylindrical"),
    ("coaxial", "Coaxial"),
)
_DEFAULT_MODE_KIND: dict[CavityType, str] = {"rectangular": "TE", "cylindrical": "TM", "coaxial": "TEM"}
_DEFAULT_MODE_INDICES: dict[CavityType, str] = {"rectangular": "0, 1, 1", "cylindrical": "0, 1, 0", "coaxial": "1"}


class _CavitySection(QWidget):
    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.type_combo = QComboBox()
        for value, label in _CAVITY_PAGES:
            self.type_combo.addItem(label, userData=value)

        self.pages = QStackedWidget()
        self._rect = _RectangularPage()
        self._cyl = _CylindricalPage()
        self._coax = _CoaxialPage()
        for page in (self._rect, self._cyl, self._coax):
            self.pages.addWidget(page)

        self.mode_kind_edit = QLineEdit(_DEFAULT_MODE_KIND["rectangular"])
        self.mode_indices_edit = QLineEdit(_DEFAULT_MODE_INDICES["rectangular"])
        self.bg_eps_r = QDoubleSpinBox()
        self.bg_eps_r.setRange(*_EPS_MU_RANGE)
        self.bg_eps_r.setValue(1.0)
        self.bg_mu_r = QDoubleSpinBox()
        self.bg_mu_r.setRange(*_EPS_MU_RANGE)
        self.bg_mu_r.setValue(1.0)

        self.rs_none_radio = QRadioButton("no wall loss")
        self.rs_conductivity_radio = QRadioButton("from conductivity")
        self.rs_explicit_radio = QRadioButton("explicit Rs")
        self.rs_conductivity_radio.setChecked(True)
        self.conductivity_box = QDoubleSpinBox()
        self.conductivity_box.setRange(*_CONDUCTIVITY_RANGE)
        self.conductivity_box.setValue(5.8e7)
        self.conductivity_box.setDecimals(0)
        self.rs_box = QDoubleSpinBox()
        self.rs_box.setRange(*_RS_RANGE)
        self.rs_box.setDecimals(6)

        form = QFormLayout()
        form.addRow("Cavity type", self.type_combo)
        form.addRow(self.pages)
        form.addRow("Mode kind", self.mode_kind_edit)
        form.addRow("Mode indices", self.mode_indices_edit)
        form.addRow("bg eps_r", self.bg_eps_r)
        form.addRow("bg mu_r", self.bg_mu_r)
        form.addRow(self.rs_none_radio)
        form.addRow(self.rs_conductivity_radio, self.conductivity_box)
        form.addRow(self.rs_explicit_radio, self.rs_box)

        box = QGroupBox("Cavity")
        box.setLayout(form)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(box)

        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        for widget in (
            self._rect.a, self._rect.b, self._rect.c,
            self._cyl.radius, self._cyl.length,
            self._coax.r_inner, self._coax.r_outer, self._coax.length,
            self.bg_eps_r, self.bg_mu_r, self.conductivity_box, self.rs_box,
        ):
            widget.valueChanged.connect(self.changed)
        self.mode_kind_edit.textChanged.connect(self.changed)
        self.mode_indices_edit.textChanged.connect(self.changed)
        for radio in (self.rs_none_radio, self.rs_conductivity_radio, self.rs_explicit_radio):
            radio.toggled.connect(self.changed)

    def _on_type_changed(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        cavity_type = self.type_combo.itemData(index)
        self.mode_kind_edit.setText(_DEFAULT_MODE_KIND[cavity_type])
        self.mode_indices_edit.setText(_DEFAULT_MODE_INDICES[cavity_type])
        self.changed.emit()

    def cavity_type(self) -> CavityType:
        return self.type_combo.currentData()

    def dimensions(self) -> tuple[float, ...]:
        # Dispatch on cavity_type directly (not QStackedWidget.currentWidget(),
        # which is typed as the generic QWidget, not the specific page class)
        # so each page's own `dimensions()` return type is preserved.
        cavity_type = self.cavity_type()
        if cavity_type == "rectangular":
            return self._rect.dimensions()
        if cavity_type == "cylindrical":
            return self._cyl.dimensions()
        return self._coax.dimensions()

    def cavity_params(self) -> CavityParams:
        return CavityParams(
            cavity_type=self.cavity_type(),
            dimensions=self.dimensions(),
            mode_kind=self.mode_kind_edit.text().strip(),
            mode_indices=_parse_indices(self.mode_indices_edit.text()),
            bg_eps_r=self.bg_eps_r.value(),
            bg_mu_r=self.bg_mu_r.value(),
        )

    def rs(self) -> float | None:
        return self.rs_box.value() if self.rs_explicit_radio.isChecked() else None

    def conductivity(self) -> float | None:
        return self.conductivity_box.value() if self.rs_conductivity_radio.isChecked() else None


class _SpherePage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.radius = _dim_spinbox(1e-3)
        layout = QFormLayout(self)
        layout.addRow("radius [m]", self.radius)


class _RodPage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.radius = _dim_spinbox(2e-4)
        self.length = QDoubleSpinBox()
        self.length.setRange(0.0, 10.0)
        self.length.setDecimals(_DIM_DECIMALS)
        self.length.setSpecialValueText("auto (16x radius)")
        layout = QFormLayout(self)
        layout.addRow("radius [m]", self.radius)
        layout.addRow("length [m]", self.length)


class _DiskPage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.extent_x = _dim_spinbox(2e-3)
        self.extent_y = _dim_spinbox(2e-3)
        self.thickness = QDoubleSpinBox()
        self.thickness.setRange(0.0, 10.0)
        self.thickness.setDecimals(_DIM_DECIMALS)
        self.thickness.setSpecialValueText("auto (0.05x min extent)")
        layout = QFormLayout(self)
        layout.addRow("extent x [m]", self.extent_x)
        layout.addRow("extent y [m]", self.extent_y)
        layout.addRow("thickness [m]", self.thickness)


_SAMPLE_PAGES: tuple[tuple[SampleShape, str], ...] = (("sphere", "Sphere"), ("rod", "Rod"), ("disk", "Disk"))


class _SampleSection(QWidget):
    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.shape_combo = QComboBox()
        for value, label in _SAMPLE_PAGES:
            self.shape_combo.addItem(label, userData=value)

        self.pages = QStackedWidget()
        self._sphere = _SpherePage()
        self._rod = _RodPage()
        self._disk = _DiskPage()
        for page in (self._sphere, self._rod, self._disk):
            self.pages.addWidget(page)

        self.auto_position_label = QLabel("position: auto (field maximum)")
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItem("aligned", userData="aligned")
        self.orientation_combo.addItem("perpendicular", userData="perpendicular")

        self.eps_r = QDoubleSpinBox()
        self.eps_r.setRange(*_EPS_MU_RANGE)
        self.eps_r.setValue(2.5)
        self.tan_delta_e = QDoubleSpinBox()
        self.tan_delta_e.setRange(*_LOSS_TANGENT_RANGE)
        self.tan_delta_e.setDecimals(6)
        self.tan_delta_e.setValue(1e-3)
        self.mu_r = QDoubleSpinBox()
        self.mu_r.setRange(*_EPS_MU_RANGE)
        self.mu_r.setValue(1.0)
        self.tan_delta_m = QDoubleSpinBox()
        self.tan_delta_m.setRange(*_LOSS_TANGENT_RANGE)
        self.tan_delta_m.setDecimals(6)
        self.tan_delta_m.setValue(0.0)

        form = QFormLayout()
        form.addRow("Shape", self.shape_combo)
        form.addRow(self.pages)
        form.addRow(self.auto_position_label)
        form.addRow("Orientation", self.orientation_combo)
        form.addRow("eps_r", self.eps_r)
        form.addRow("tan_delta_e", self.tan_delta_e)
        form.addRow("mu_r", self.mu_r)
        form.addRow("tan_delta_m", self.tan_delta_m)

        box = QGroupBox("Sample")
        box.setLayout(form)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(box)

        self.shape_combo.currentIndexChanged.connect(self._on_shape_changed)
        for widget in (
            self._sphere.radius, self._rod.radius, self._rod.length,
            self._disk.extent_x, self._disk.extent_y, self._disk.thickness,
            self.eps_r, self.tan_delta_e, self.mu_r, self.tan_delta_m,
        ):
            widget.valueChanged.connect(self.changed)
        self.orientation_combo.currentIndexChanged.connect(self.changed)

    def _on_shape_changed(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        self.changed.emit()

    def shape(self) -> SampleShape:
        return self.shape_combo.currentData()

    def orientation(self) -> Orientation:
        return self.orientation_combo.currentData()

    def sample_params(self) -> SampleParams:
        shape = self.shape()
        orientation = self.orientation()
        eps_r = self.eps_r.value()
        tan_delta_e = self.tan_delta_e.value()
        mu_r = self.mu_r.value()
        tan_delta_m = self.tan_delta_m.value()

        radius: float | None = None
        length: float | None = None
        extent: tuple[float, float] | None = None
        thickness: float | None = None

        if shape == "sphere":
            radius = self._sphere.radius.value()
        elif shape == "rod":
            radius = self._rod.radius.value()
            if self._rod.length.value() > 0.0:
                length = self._rod.length.value()
        elif shape == "disk":
            extent = (self._disk.extent_x.value(), self._disk.extent_y.value())
            if self._disk.thickness.value() > 0.0:
                thickness = self._disk.thickness.value()

        return SampleParams(
            shape=shape,
            radius=radius,
            length=length,
            extent=extent,
            thickness=thickness,
            orientation=orientation,
            eps_r=eps_r,
            tan_delta_e=tan_delta_e,
            mu_r=mu_r,
            tan_delta_m=tan_delta_m,
        )


class Sidebar(QWidget):
    """Owns both the cavity and sample sections; `changed` fires whenever
    either does."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.cavity_section = _CavitySection()
        self.sample_section = _SampleSection()
        layout = QVBoxLayout(self)
        layout.addWidget(self.cavity_section)
        layout.addWidget(self.sample_section)
        layout.addStretch(1)

        self.cavity_section.changed.connect(self.changed)
        self.sample_section.changed.connect(self.changed)

    def cavity_params(self) -> CavityParams:
        return self.cavity_section.cavity_params()

    def sample_params(self) -> SampleParams:
        return self.sample_section.sample_params()

    def rs(self) -> float | None:
        return self.cavity_section.rs()

    def conductivity(self) -> float | None:
        return self.cavity_section.conductivity()
