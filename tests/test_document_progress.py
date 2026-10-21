import io
from threading import Event

import pytest

from document_progress import OperationProgress, report_progress


def test_progress_is_opt_in_and_reports_stages_elapsed_and_outcome(capsys):
    report_progress("not displayed")
    with OperationProgress("ask", "test-model", interval=60) as progress:
        report_progress("planning")
        report_progress("answer: request 2, attempt 2/2")
    report_progress("also not displayed")
    output = capsys.readouterr().err
    assert "not displayed" not in output
    assert "planning" in output and "attempt 2/2" in output
    assert "elapsed=" in output and "model=test-model" in output and "Ctrl+C cancels" in output
    assert "completed" in output
    assert not progress.thread.is_alive()


@pytest.mark.parametrize("error,stage", [
    (KeyboardInterrupt(), "cancelled"), (EOFError(), "cancelled"), (ValueError("SECRET"), "failed"),
])
def test_progress_stops_and_resets_on_interruption_or_failure(error, stage, capsys):
    with pytest.raises(type(error)):
        with OperationProgress("ask", "test-model", interval=60) as progress:
            raise error
    report_progress("after scope")
    output = capsys.readouterr().err
    assert stage in output and "SECRET" not in output and "after scope" not in output
    assert not progress.thread.is_alive()


def test_heartbeat_reports_while_main_thread_waits_and_stops_on_exit(monkeypatch):
    heartbeat = Event()

    class Stream(io.StringIO):
        def write(self, text):
            if "planning" in text:
                heartbeat.set()
            return super().write(text)

    stream = Stream()
    monkeypatch.setattr("sys.stderr", stream)
    with OperationProgress("ask", "model", interval=0.01) as progress:
        # Set the heartbeat stage without an immediate display.
        with progress.lock:
            progress.stage = "planning"
        assert heartbeat.wait(timeout=2)
    assert not progress.thread.is_alive()
    assert "planning" in stream.getvalue()


def test_nested_scopes_restore_outer_progress(capsys):
    with OperationProgress("outer", "model", interval=60):
        with OperationProgress("inner", "model", interval=60):
            report_progress("inside")
        report_progress("outside")
    output = capsys.readouterr().err
    assert "inner | inside" in output
    assert "outer | outside" in output


def test_failed_thread_start_restores_scope_and_surfaces_error(monkeypatch, capsys):
    progress = OperationProgress("ask", "model")

    def fail():
        raise RuntimeError("Cannot start display thread")

    monkeypatch.setattr(progress.thread, "start", fail)
    with pytest.raises(RuntimeError, match="Cannot start"):
        with progress:
            pytest.fail("Operation must not run without progress initialization")
    report_progress("leaked scope")
    assert "leaked scope" not in capsys.readouterr().err
    assert progress.stopped.is_set()
