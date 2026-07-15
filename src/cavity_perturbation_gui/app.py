"""app.py -- application entry point (docs/gui_module_plan.md Section 3)."""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from cavity_perturbation_gui.widgets.main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    app = QApplication(argv if argv is not None else sys.argv)
    window = MainWindow()
    window.resize(1400, 900)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
