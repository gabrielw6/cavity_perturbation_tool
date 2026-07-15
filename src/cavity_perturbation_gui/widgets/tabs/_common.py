"""widgets/tabs/_common.py -- shared plumbing every forward/inversion tab
uses: a Run button that disables itself for the run's duration and routes
the call through `workers/solve_worker.py` (docs/gui_module_plan.md
Section 6), logging start/finish/failure through `logging_bridge.py`
(Section 7). Not one of Section 3's five listed tab files itself -- an
internal implementation detail shared by all of them, the same "small
reusable piece, not a speculative abstraction" rationale as `fdtd/model.py`'s
own `_run(capture)` helper.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from cavity_perturbation_gui.logging_bridge import get_logger
from cavity_perturbation_gui.workers.solve_worker import SolveWorker, run_in_background

Array = np.ndarray


class RunBar(QWidget):
    """A Run button + status label. `run()` disables the button for the
    duration of the call (Section 6: "the button set just disables for the
    duration of that tab's run") and never lets a raised exception escape
    to crash the app (Section 4) -- it surfaces in both the status label
    and the log bar instead."""

    def __init__(self, label: str = "Run", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.button = QPushButton(label)
        self.status_label = QLabel("")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.button)
        layout.addWidget(self.status_label)
        layout.addStretch(1)
        self._logger = get_logger()
        self._thread: QThread | None = None
        self._worker: SolveWorker | None = None

    def run(
        self,
        build_call: Callable[[], Callable[[], Any]],
        on_success: Callable[[Any], None],
        tab_name: str,
    ) -> None:
        """`build_call` runs on the GUI thread (so a bad sidebar value
        raises immediately, before ever touching a background thread) and
        must return a zero-argument callable -- typically
        `functools.partial(run_perturbation, cavity_params, sample_params)`
        -- which is what actually executes in the background."""
        try:
            call = build_call()
        except Exception as exc:  # noqa: BLE001 -- Section 4: never let a bad parameter crash the app
            self._logger.error("%s: failed to prepare run: %s", tab_name, exc)
            self.status_label.setText(f"error: {exc}")
            return

        self.button.setEnabled(False)
        self.status_label.setText("running...")
        self._logger.info("%s: run started", tab_name)
        self._thread, self._worker = run_in_background(call)

        def _on_finished(result: Any) -> None:
            self.button.setEnabled(True)
            self.status_label.setText("done")
            self._logger.info("%s: run finished", tab_name)
            on_success(result)

        def _on_failed(exc: BaseException) -> None:
            self.button.setEnabled(True)
            self.status_label.setText(f"error: {exc}")
            self._logger.error("%s: run failed: %s", tab_name, exc)

        self._worker.finished.connect(_on_finished)
        self._worker.failed.connect(_on_failed)
