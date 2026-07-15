"""GeometryTab: a dedicated 3D view, live-updating from the sidebar with no
Run action -- exercises the real adapters/geometry_description.py path
(cheap, closed-form, no solver run needed), unlike the four forward tabs'
mocked-adapter tests."""
from cavity_perturbation_gui.widgets.geometry_view3d import GeometryView3D
from cavity_perturbation_gui.widgets.sidebar import Sidebar
from cavity_perturbation_gui.widgets.tabs.geometry_tab import GeometryTab


def test_draws_cavity_and_sample_on_construction_with_no_run_action(qtbot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    tab = GeometryTab(sidebar)
    qtbot.addWidget(tab)

    assert isinstance(tab.geometry_view, GeometryView3D)
    # cavity (1 box) + sample (1 sphere by default) -- no Run button exists
    # on this tab at all, so any drawing here happened purely from __init__.
    assert len(tab.geometry_view._items) == 2


def test_redraws_live_when_sidebar_changes(qtbot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    tab = GeometryTab(sidebar)
    qtbot.addWidget(tab)

    first_items = list(tab.geometry_view._items)
    sidebar.cavity_section._rect.a.setValue(0.05)

    # A fresh set of drawn items, not the same Python objects reused --
    # confirms `redraw()` actually re-ran (clear + rebuild), not a no-op.
    assert tab.geometry_view._items != first_items
    assert len(tab.geometry_view._items) == 2


def test_switching_cavity_type_still_draws_both_primitives(qtbot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    tab = GeometryTab(sidebar)
    qtbot.addWidget(tab)

    sidebar.cavity_section.type_combo.setCurrentIndex(2)  # coaxial -> annulus (2 items) + sample
    assert len(tab.geometry_view._items) == 3  # outer + inner conductor cylinders, plus sample


def test_invalid_sidebar_state_does_not_crash_the_live_view(qtbot):
    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    tab = GeometryTab(sidebar)
    qtbot.addWidget(tab)

    # Switch to 'disk' with a zero extent -- a degenerate but not
    # exception-raising geometry (Slab accepts extent=(0,0)); confirms the
    # redraw path tolerates edge-of-valid sidebar states without raising.
    sidebar.sample_section.shape_combo.setCurrentIndex(2)
    sidebar.sample_section._disk.extent_x.setValue(0.0)
    sidebar.sample_section._disk.extent_y.setValue(0.0)
    tab.redraw()  # must not raise
