"""docs/gui_module_plan.md Section 7/8 -- the logging bridge, testable with
plain `logging` calls and `pytest-qt`'s `qtbot`."""
import logging
import warnings

import pytest

from cavity_perturbation_gui.logging_bridge import (
    LOGGER_NAME,
    QtLogHandler,
    get_logger,
    install_logging_bridge,
)


@pytest.fixture
def clean_loggers():
    """Logging is global/process-wide state -- strip handlers this test
    added before and after, so tests don't leak handlers into each other."""
    yield
    for name in (LOGGER_NAME, "py.warnings"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
    logging.captureWarnings(False)


def test_qt_log_handler_emits_formatted_record(qtbot):
    handler = QtLogHandler()
    messages = []
    handler.record_logged.connect(messages.append)

    logger = logging.getLogger("test_qt_log_handler")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info("hello from the test")

    assert len(messages) == 1
    assert "hello from the test" in messages[0]
    assert "INFO" in messages[0]


def test_install_logging_bridge_routes_package_logger(qtbot, clean_loggers):
    handler = install_logging_bridge()
    messages = []
    handler.record_logged.connect(messages.append)

    get_logger().info("run started: rectangular cavity")

    assert any("run started: rectangular cavity" in m for m in messages)


def test_install_logging_bridge_captures_warnings(qtbot, clean_loggers):
    handler = install_logging_bridge()
    messages = []
    handler.record_logged.connect(messages.append)

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn("near-degenerate mode mixing", RuntimeWarning)

    assert any("near-degenerate mode mixing" in m for m in messages)


def test_get_logger_returns_the_package_logger():
    assert get_logger().name == LOGGER_NAME
