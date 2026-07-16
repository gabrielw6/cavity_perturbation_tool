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
from PySide6.QtCore import QThread, Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from cavity_perturbation_gui.logging_bridge import get_logger
from cavity_perturbation_gui.workers.solve_worker import SolveWorker, run_in_background

Array = np.ndarray

_DEFAULT_MIN_PLOT_HEIGHT = 220


class RunBar(QWidget):
    """A Run button + status label. `run()` disables the button for the
    duration of the call (Section 6: "the button set just disables for the
    duration of that tab's run") and never lets a raised exception escape
    to crash the app (Section 4) -- it surfaces in both the status label
    and the log bar instead.

    `SolveWorker.finished`/`.failed` fire from the *background* thread
    (Section 6). Qt's own AutoConnection auto-upgrades a cross-thread
    signal-slot connection to a queued one, but only when it can determine
    the *receiver's* thread affinity -- which it can for a bound method of
    a real `QObject` (this class is a `QWidget`, always constructed on the
    GUI thread), but NOT for a plain closure (no owning `QObject`, so Qt has
    no thread to post the event to). An earlier version of this method
    connected the worker's signals to local closures defined inside `run()`
    -- verified directly (a background thread + real `QThread`, not just
    read from Qt's docs) to execute those closures on the *worker* thread
    regardless of connection type, including an explicit
    `Qt.ConnectionType.QueuedConnection`. Symptoms matched a real user bug
    report exactly: `QLabel`/`CurvePlot` updates happened to survive
    off-thread execution, but pyqtgraph's `ImageView.setImage()` (the field
    plane view) silently failed to render -- caught by a pytest-qt
    regression test asserting the finished-handler's own thread via
    `QThread.currentThread()`, not just by eye. Fixed by making the
    handlers real bound methods of `self` instead, so Qt's own thread-
    affinity detection does the right thing automatically (the explicit
    `QueuedConnection` below is kept anyway, as documentation of intent,
    not because it's load-bearing once the receiver is a real `QObject`)."""

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
        self._on_success: Callable[[Any], None] | None = None
        self._on_failure: Callable[[BaseException], None] | None = None
        self._tab_name: str = ""

    def run(
        self,
        build_call: Callable[[], Callable[[], Any]],
        on_success: Callable[[Any], None],
        tab_name: str,
        on_failure: Callable[[BaseException], None] | None = None,
    ) -> None:
        """`build_call` runs on the GUI thread (so a bad sidebar value
        raises immediately, before ever touching a background thread) and
        must return a zero-argument callable -- typically
        `functools.partial(run_perturbation, cavity_params, sample_params)`
        -- which is what actually executes in the background.

        `on_failure`, if given, additionally runs after this bar's own
        failure handling (status label + log) -- e.g. the FDTD tab's Stop
        button uses it to reset its own state regardless of whether the run
        failed normally or was cancelled (docs/gui_module_plan.md Section 6).

        Only one run is ever in flight per `RunBar` (the button disables
        itself for the duration), so it's safe to stash the per-call
        callbacks as instance state for `_handle_finished`/`_handle_failed`
        (real bound methods, not closures -- see the class docstring) to
        pick up."""
        try:
            call = build_call()
        except Exception as exc:  # noqa: BLE001 -- Section 4: never let a bad parameter crash the app
            self._logger.error("%s: failed to prepare run: %s", tab_name, exc)
            self.status_label.setText(f"error: {exc}")
            return

        self.button.setEnabled(False)
        self.status_label.setText("running...")
        self._logger.info("%s: run started", tab_name)
        self._on_success = on_success
        self._on_failure = on_failure
        self._tab_name = tab_name
        self._thread, self._worker = run_in_background(call)
        self._worker.finished.connect(self._handle_finished, Qt.ConnectionType.QueuedConnection)
        self._worker.failed.connect(self._handle_failed, Qt.ConnectionType.QueuedConnection)

    def _handle_finished(self, result: Any) -> None:
        self.button.setEnabled(True)
        self.status_label.setText("done")
        self._logger.info("%s: run finished", self._tab_name)
        if self._on_success is not None:
            self._on_success(result)

    def _handle_failed(self, exc: BaseException) -> None:
        self.button.setEnabled(True)
        self.status_label.setText(f"error: {exc}")
        self._logger.error("%s: run failed: %s", self._tab_name, exc)
        if self._on_failure is not None:
            self._on_failure(exc)


class PlotColumn(QScrollArea):
    """Left-hand 2D-plot column shared by all four forward tabs (Section 5):
    stacks one or more plot widgets vertically, each given a sane minimum
    height, inside a scroll area -- so a tab with several plots (FDTD's
    resonance/excitation/spectrum trio) gets a scrollbar instead of
    squeezing every plot down to nothing when the window is short."""

    def __init__(
        self,
        plots: list[QWidget],
        min_plot_height: int = _DEFAULT_MIN_PLOT_HEIGHT,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        container = QWidget()
        layout = QVBoxLayout(container)
        for plot in plots:
            plot.setMinimumHeight(min_plot_height)
            layout.addWidget(plot)
        self.setWidget(container)
