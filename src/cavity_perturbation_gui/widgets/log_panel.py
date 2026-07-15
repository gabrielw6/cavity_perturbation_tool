"""widgets/log_panel.py -- bottom-docked log bar (docs/gui_module_plan.md
Section 5: "spans the whole window, always visible regardless of active
tab"). Connected to `logging_bridge.QtLogHandler.record_logged` by
`main_window.py`/`app.py`, not by this widget itself -- keeps this widget
usable/testable without any real logging setup."""
from __future__ import annotations

from PySide6.QtWidgets import QPlainTextEdit, QWidget

_MAX_LINES = 2000


class LogPanel(QPlainTextEdit):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(_MAX_LINES)

    def append_line(self, text: str) -> None:
        self.appendPlainText(text)
