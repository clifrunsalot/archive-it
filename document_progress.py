"""Scoped terminal progress without document or model-response content."""

import sys
from contextvars import ContextVar, Token
from threading import Event, Lock, Thread
from time import monotonic
from types import TracebackType

_active: ContextVar["OperationProgress | None"] = ContextVar("document_progress", default=None)


def model_name(model: object) -> str:
    bound = getattr(model, "bound", model)
    name = getattr(bound, "model", None)
    return name if isinstance(name, str) else "configured local model"


def report_progress(stage: str) -> None:
    progress = _active.get()
    if progress is not None:
        progress.update(stage)


class OperationProgress:
    def __init__(self, action: str, model: str, *, interval: float = 5.0) -> None:
        self.action, self.model, self.interval = action, model, interval
        self.started = monotonic()
        self.stage = "starting"
        self.stopped = Event()
        self.lock = Lock()
        self.thread = Thread(target=self._heartbeat, daemon=True)
        self.token: Token | None = None

    def _display(self) -> None:
        print(f"Progress: {self.action} | {self.stage} | model={self.model} | "
              f"elapsed={monotonic() - self.started:.0f}s | Ctrl+C cancels",
              file=sys.stderr, flush=True)

    def update(self, stage: str) -> None:
        with self.lock:
            self.stage = stage
            self._display()

    def _heartbeat(self) -> None:
        while not self.stopped.wait(self.interval):
            with self.lock:
                self._display()

    def __enter__(self) -> "OperationProgress":
        self.started = monotonic()
        self.token = _active.set(self)
        try:
            self.update("starting")
            self.thread.start()
        except BaseException:
            self.stopped.set()
            if self.thread.ident is not None:
                self.thread.join()
            _active.reset(self.token)
            raise
        return self

    def __exit__(self, kind: type[BaseException] | None, _error: BaseException | None,
                 _traceback: TracebackType | None) -> None:
        self.stopped.set()
        try:
            self.thread.join()
        finally:
            if self.token is not None:
                _active.reset(self.token)
        self.update("completed" if kind is None else
                    "cancelled" if kind in (KeyboardInterrupt, EOFError) else "failed")
