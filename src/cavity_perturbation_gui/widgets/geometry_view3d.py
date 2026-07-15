"""widgets/geometry_view3d.py -- pyqtgraph.opengl GLViewWidget, the cavity +
sample geometry (docs/gui_module_plan.md Section 0.2/5). Redraws live as
sidebar parameters change. Only ever draws plain geometric primitives from
adapters/geometry_description.py -- never touches a cavity_perturbation
type directly (Section 4)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtWidgets import QVBoxLayout, QWidget

from cavity_perturbation_gui.adapters.geometry_description import (
    AnnulusPrimitive,
    BoxPrimitive,
    CavityPrimitive,
    CylinderPrimitive,
    SamplePrimitive,
    SlabPrimitive,
    SpherePrimitive,
)

Array = np.ndarray
Point3 = tuple[float, float, float] | Array

_CAVITY_COLOR = (0.3, 0.5, 0.9, 0.25)
_SAMPLE_COLOR = (0.9, 0.3, 0.2, 0.6)
_INNER_CONDUCTOR_COLOR = (0.6, 0.6, 0.6, 0.9)


def _rotation_matrix_z_to(axis: Point3) -> Array:
    """3x3 rotation matrix mapping the unit +z axis onto unit vector
    `axis` (Rodrigues' rotation formula) -- used so every mesh primitive
    below, natively built along +z, can be pointed along an arbitrary
    sample axis/normal in one deterministic step, rather than composing
    pyqtgraph's incremental (and order-sensitive) rotate()/translate()
    calls."""
    z = np.array([0.0, 0.0, 1.0])
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    v = np.cross(z, axis)
    c = float(np.dot(z, axis))
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        if c > 0:
            return np.eye(3)
        # Antiparallel (axis = -z): 180-degree rotation about any
        # perpendicular vector -- x-axis is perpendicular to z.
        return np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def _transform_matrix(center: Point3, axis: Point3, local_z_offset: float = 0.0) -> Array:
    """4x4 transform placing a +z-aligned, origin-based mesh at `center`,
    pointed along `axis`, after first shifting it by `local_z_offset`
    along its own (pre-rotation) z-axis -- e.g. -length/2, to recenter a
    cylinder built spanning local z in [0, length]."""
    R = _rotation_matrix_z_to(axis)
    local_shift = R @ np.array([0.0, 0.0, local_z_offset])
    translation = np.asarray(center, dtype=float) + local_shift
    matrix = np.eye(4)
    matrix[:3, :3] = R
    matrix[:3, 3] = translation
    return matrix


def _apply_transform(item: gl.GLGraphicsItem.GLGraphicsItem, matrix: Array) -> None:
    from pyqtgraph import Transform3D

    item.setTransform(Transform3D(*matrix.flatten().tolist()))


def _cylinder_item(center: Point3, axis: Point3, radius: float, length: float, color: tuple[float, ...]) -> gl.GLMeshItem:
    length = max(length, 1e-12)
    radius = max(radius, 1e-12)
    md = gl.MeshData.cylinder(rows=2, cols=24, radius=[radius, radius], length=length)
    item = gl.GLMeshItem(meshdata=md, smooth=True, color=color, shader="balloon", glOptions="translucent")
    _apply_transform(item, _transform_matrix(center, axis, local_z_offset=-length / 2.0))
    return item


def _sphere_item(center: Point3, radius: float, color: tuple[float, ...]) -> gl.GLMeshItem:
    md = gl.MeshData.sphere(rows=12, cols=24, radius=max(radius, 1e-12))
    item = gl.GLMeshItem(meshdata=md, smooth=True, color=color, shader="balloon", glOptions="translucent")
    item.translate(*[float(c) for c in center])
    return item


def _box_item(corner_min: Point3, corner_max: Point3, color: tuple[float, ...]) -> gl.GLBoxItem:
    corner_min = np.asarray(corner_min, dtype=float)
    corner_max = np.asarray(corner_max, dtype=float)
    size = corner_max - corner_min
    item = gl.GLBoxItem()
    item.setSize(*[float(s) for s in size])
    item.setColor(tuple(int(255 * c) for c in color))
    item.translate(*[float(c) for c in corner_min])
    return item


class GeometryView3D(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.gl_view = gl.GLViewWidget()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.gl_view)
        self._items: list[gl.GLGraphicsItem.GLGraphicsItem] = []

    def clear(self) -> None:
        for item in self._items:
            self.gl_view.removeItem(item)
        self._items.clear()

    def set_geometry(self, cavity: CavityPrimitive, sample: SamplePrimitive | None) -> None:
        self.clear()
        self._add_cavity(cavity)
        if sample is not None:
            self._add_sample(sample)
        self._frame_camera(cavity)

    def _add(self, item: gl.GLGraphicsItem.GLGraphicsItem) -> None:
        self.gl_view.addItem(item)
        self._items.append(item)

    def _add_cavity(self, prim: CavityPrimitive) -> None:
        if isinstance(prim, BoxPrimitive):
            self._add(_box_item(prim.corner_min, prim.corner_max, _CAVITY_COLOR))
        elif isinstance(prim, CylinderPrimitive):
            self._add(_cylinder_item(prim.center, prim.axis, prim.radius, prim.length, _CAVITY_COLOR))
        elif isinstance(prim, AnnulusPrimitive):
            self._add(_cylinder_item(prim.center, prim.axis, prim.outer_radius, prim.length, _CAVITY_COLOR))
            self._add(_cylinder_item(prim.center, prim.axis, prim.inner_radius, prim.length, _INNER_CONDUCTOR_COLOR))
        else:
            raise ValueError(f"unknown cavity primitive {prim!r}")

    def _add_sample(self, prim: SamplePrimitive) -> None:
        if isinstance(prim, SpherePrimitive):
            self._add(_sphere_item(prim.center, prim.radius, _SAMPLE_COLOR))
        elif isinstance(prim, CylinderPrimitive):
            self._add(_cylinder_item(prim.center, prim.axis, prim.radius, prim.length, _SAMPLE_COLOR))
        elif isinstance(prim, SlabPrimitive):
            half = (prim.extent[0] / 2.0, prim.extent[1] / 2.0, prim.thickness / 2.0)
            # A slab is a box in its own (extent[0], extent[1], thickness)
            # local frame, whose local z is `normal` -- reuse the cylinder
            # transform's rotation, sized as a flat box instead of a disk.
            md_box = gl.MeshData.cylinder(rows=1, cols=4, radius=[half[0] * 1.4142, half[0] * 1.4142], length=prim.thickness)
            item = gl.GLMeshItem(meshdata=md_box, smooth=False, color=_SAMPLE_COLOR, shader="balloon", glOptions="translucent")
            _apply_transform(item, _transform_matrix(prim.center, prim.normal, local_z_offset=-prim.thickness / 2.0))
            self._add(item)
        else:
            raise ValueError(f"unknown sample primitive {prim!r}")

    def _frame_camera(self, cavity: CavityPrimitive) -> None:
        if isinstance(cavity, BoxPrimitive):
            extent = np.asarray(cavity.corner_max) - np.asarray(cavity.corner_min)
            center = (np.asarray(cavity.corner_min) + np.asarray(cavity.corner_max)) / 2.0
        elif isinstance(cavity, CylinderPrimitive):
            extent = np.array([2 * cavity.radius, 2 * cavity.radius, cavity.length])
            center = np.asarray(cavity.center)
        elif isinstance(cavity, AnnulusPrimitive):
            extent = np.array([2 * cavity.outer_radius, 2 * cavity.outer_radius, cavity.length])
            center = np.asarray(cavity.center)
        else:
            return
        distance = float(np.linalg.norm(extent)) * 1.5
        self.gl_view.setCameraPosition(distance=max(distance, 1e-6))
        self.gl_view.opts["center"] = pg_vector(center)


def pg_vector(center: Array) -> Any:
    from pyqtgraph import Vector

    return Vector(*[float(c) for c in center])
