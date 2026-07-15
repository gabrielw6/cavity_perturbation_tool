"""logging_bridge.py -- bridges Python's `logging` module to the Qt log bar
(docs/gui_module_plan.md Section 7). Not stdout capture: a `logging.Handler`
subclass posts formatted records to `widgets/log_panel.py` via a Qt signal,
safe to emit from a worker thread (Qt's signal/slot delivery across threads
is queued automatically). `logging.captureWarnings(True)` picks up
`ritz.py`'s existing degenerate-mode-mixing `warnings.warn` unmodified.
"""
from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal

LOGGER_NAME = "cavity_perturbation_gui"
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


class QtLogHandler(logging.Handler, QObject):
    """A `logging.Handler` that emits a Qt signal instead of writing
    anywhere itself. `widgets/log_panel.py` connects to `record_logged` to
    append formatted text."""

    record_logged = Signal(str)

    def __init__(self, level: int = logging.INFO) -> None:
        logging.Handler.__init__(self, level=level)
        QObject.__init__(self)
        self.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
        # Multiple inheritance collision, not a real LSP violation: QObject
        # has its own unrelated `emit` (Qt's low-level signal-emission
        # primitive). Python's MRO resolves `self.emit` to
        # `logging.Handler`'s version here (Handler listed first above),
        # which is what `logging`'s dispatch machinery actually calls --
        # mypy sees the QObject.emit signature this overrides and flags it,
        # but the two `emit`s are unrelated methods that happen to share a
        # name.
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()
        self.record_logged.emit(message)


def install_logging_bridge(level: int = logging.INFO) -> QtLogHandler:
    """Attaches a `QtLogHandler` to this package's logger and to
    `logging`'s `py.warnings` logger (Section 7), so both ordinary log
    calls and `warnings.warn` (e.g. `ritz.py`'s near-degenerate-mixing
    warning) reach the log bar. Intended to be called once, from `app.py`.
    """
    handler = QtLogHandler(level=level)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.addHandler(handler)

    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.setLevel(level)
    warnings_logger.addHandler(handler)

    return handler


def get_logger() -> logging.Logger:
    """This package's logger -- runners log their own start/parameters/
    finish through this, at INFO (Section 7), so a run's provenance is
    visible in the log bar without instrumenting every widget."""
    return logging.getLogger(LOGGER_NAME)
