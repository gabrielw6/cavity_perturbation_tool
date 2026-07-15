"""docs/gui_module_plan.md Section 8/10 build order item 3 -- the threading
skeleton, testable with a trivial fake runner. `pytest-qt`'s `qtbot` gives
a real Qt event loop without a visible window."""
import pytest

from cavity_perturbation_gui.workers.solve_worker import SolveWorker, run_in_background


def test_run_emits_finished_with_runner_return_value(qtbot):
    worker = SolveWorker(lambda: 42)
    logs = []
    worker.log.connect(logs.append)

    with qtbot.waitSignal(worker.finished, timeout=1000) as blocker:
        worker.run()

    assert blocker.args == [42]
    assert logs == ["run started", "run finished"]


def test_run_emits_failed_when_runner_raises(qtbot):
    def boom():
        raise ValueError("synthetic failure")

    worker = SolveWorker(boom)
    logs = []
    worker.log.connect(logs.append)

    with qtbot.waitSignal(worker.failed, timeout=1000) as blocker:
        worker.run()

    (caught,) = blocker.args
    assert isinstance(caught, ValueError)
    assert str(caught) == "synthetic failure"
    assert logs[0] == "run started"
    assert "synthetic failure" in logs[1]


def test_failed_runner_never_emits_finished(qtbot):
    worker = SolveWorker(lambda: (_ for _ in ()).throw(RuntimeError("nope")))
    finished_calls = []
    worker.finished.connect(finished_calls.append)

    with qtbot.waitSignal(worker.failed, timeout=1000):
        worker.run()

    assert finished_calls == []


def test_run_in_background_executes_on_a_real_thread(qtbot):
    # The thread object itself isn't safe to poll afterward: it's wired to
    # deleteLater on its own `finished` signal (Section 6's "don't leak
    # worker objects across runs"), which can already have fired by the
    # time this signal wait returns. `worker.finished` delivering the
    # correct result is itself the proof the background thread actually
    # ran and completed -- the meaningful, documented contract.
    _thread, worker = run_in_background(lambda: "done")
    with qtbot.waitSignal(worker.finished, timeout=2000) as finished_blocker:
        pass
    assert finished_blocker.args == ["done"]


def test_run_in_background_failure_stops_the_thread_too(qtbot):
    def boom():
        raise KeyError("bad")

    _thread, worker = run_in_background(boom)
    with qtbot.waitSignal(worker.failed, timeout=2000) as failed_blocker:
        pass
    (caught,) = failed_blocker.args
    assert isinstance(caught, KeyError)
