"""cavity_perturbation_gui -- interactive PySide6 + PyQtGraph front end for
the four forward solvers (analytical, perturbational, variational/Ritz,
FDTD) and the inverse fit in `cavity_perturbation`, per docs/gui_module_plan.md.

One-way dependency (Section 1.1): this package imports `cavity_perturbation`;
`cavity_perturbation` never imports this package. `tests/test_gui/
test_no_reverse_import.py` enforces that mechanically.
"""
