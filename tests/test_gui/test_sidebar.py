"""docs/gui_module_plan.md Section 8 -- Sidebar: the shared parameter model
every tab reads. No adapter mocking needed -- this widget only ever builds
plain dataclasses, never calls a runner."""
from cavity_perturbation_gui.widgets.sidebar import Sidebar


def test_default_cavity_params_are_rectangular_te011(qtbot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    params = sidebar.cavity_params()
    assert params.cavity_type == "rectangular"
    assert params.mode_kind == "TE"
    assert params.mode_indices == (0, 1, 1)
    assert len(params.dimensions) == 3


def test_switching_cavity_type_changes_dimensions_arity(qtbot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    sidebar.cavity_section.type_combo.setCurrentIndex(1)  # cylindrical
    params = sidebar.cavity_params()
    assert params.cavity_type == "cylindrical"
    assert len(params.dimensions) == 2
    assert params.mode_kind == "TM"


def test_changed_signal_fires_on_dimension_edit(qtbot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    with qtbot.waitSignal(sidebar.changed, timeout=1000):
        sidebar.cavity_section._rect.a.setValue(0.05)
    assert sidebar.cavity_params().dimensions[0] == 0.05


def test_rs_source_defaults_to_conductivity(qtbot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    assert sidebar.rs() is None
    assert sidebar.conductivity() is not None


def test_rs_source_explicit_overrides_conductivity(qtbot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    sidebar.cavity_section.rs_explicit_radio.setChecked(True)
    sidebar.cavity_section.rs_box.setValue(0.05)
    assert sidebar.rs() == 0.05
    assert sidebar.conductivity() is None


def test_no_wall_loss_gives_neither_rs_nor_conductivity(qtbot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    sidebar.cavity_section.rs_none_radio.setChecked(True)
    assert sidebar.rs() is None
    assert sidebar.conductivity() is None


def test_default_sample_params_are_sphere(qtbot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    params = sidebar.sample_params()
    assert params.shape == "sphere"
    assert params.radius is not None


def test_switching_sample_shape_to_disk_requires_extent(qtbot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    sidebar.sample_section.shape_combo.setCurrentIndex(2)  # disk
    params = sidebar.sample_params()
    assert params.shape == "disk"
    assert params.extent is not None
