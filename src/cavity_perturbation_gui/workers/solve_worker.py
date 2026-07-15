"""workers/solve_worker.py -- QObject + moveToThread around one runner call
(docs/gui_module_plan.md Section 6). Every Run action goes through this,
never the GUI thread. No mid-run cancellation in v1 (Section 11) -- the
caller's button set just disables for the duration of the run.
"""
from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, Signal


class SolveWorker(QObject):
    """Wraps a single zero-argument callable -- typically
    `functools.partial(run_perturbation, cavity_params, sample_params, ...)`
    from one of `adapters/*_runner.py` -- and executes it, catching any
    exception rather than letting it propagate (Section 4: "workers ...
    catch and forward exceptions as log entries, never let one crash the
    app"). `run()` is safe to call directly (e.g. from a test, on the
    calling thread) or via `run_in_background`'s real QThread.

    Signals:
        finished(object): the runner's return value (a `*RunResult`
            dataclass from `adapters/*_runner.py`).
        failed(object): the exception the runner raised.
        log(str): a short provenance message ("run started"/"run
            finished") -- the runner itself logs its own parameters via
            `logging_bridge.get_logger()` (Section 7); this signal is only
            the coarse start/stop bracket around that.
    """

    finished = Signal(object)
    failed = Signal(object)
    log = Signal(str)

    def __init__(self, run: Callable[[], Any]) -> None:
        super().__init__()
        self._run = run

    def run(self) -> None:
        self.log.emit("run started")
        try:
            result = self._run()
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, Section 4/6
            self.log.emit(f"run failed: {exc}")
            self.failed.emit(exc)
            return
        self.log.emit("run finished")
        self.finished.emit(result)


def run_in_background(run: Callable[[], Any]) -> tuple[QThread, SolveWorker]:
    """Builds and starts a `QThread` + `SolveWorker` pair for `run`.
    Returns both so the caller (a tab widget) can connect
    `finished`/`failed`/`log` before the thread starts producing signals,
    and so it owns the objects' lifetime -- kept a plain function rather
    than baked into a widget so it's testable without a real widget
    (Section 8).

    Cleanup wiring follows Qt's own documented worker-thread pattern
    exactly (https://doc.qt.io/qt-6/qthread.html#details) -- each object's
    *own* completion signal drives its *own* deleteLater, not the other
    object's: `worker.finished`/`.failed` fire while the worker thread's
    event loop is still alive, so `worker.deleteLater` there is processed
    safely by that same (still-running) loop; `thread.finished` only fires
    once the worker thread's loop has fully stopped, so `thread.deleteLater`
    is queued back to the thread that *created* `thread` (its own affinity)
    instead. Wiring `thread.finished` to `worker.deleteLater` -- an earlier
    version of this function did -- targets a queued call at a thread whose
    event loop has already stopped by the time `thread.finished` fires,
    which is never safely processed; caught directly via a `Fatal Python
    error: Aborted` crash during the test suite, not a Python-level
    exception."""
    thread = QThread()
    worker = SolveWorker(run)
    worker.moveToThread(thread)

    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    worker.failed.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    thread.start()
    return thread, worker
