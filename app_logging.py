"""File-configured application logging without recording personal content."""

import json
import logging
import traceback
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import perf_counter
from typing import Callable, ParamSpec, TypeVar

LOGGER = logging.getLogger("archive_it")
LOGGER.addHandler(logging.NullHandler())
LOGGER.propagate = False
P = ParamSpec("P")
T = TypeVar("T")


def configure_logging(config_path: str | Path) -> Path:
    """Configure one rotating handler; paths are relative to the config file."""
    path = Path(config_path).resolve()
    settings = json.loads(path.read_text(encoding="utf-8"))
    required = {"level", "file", "max_bytes", "retained_files"}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError("Logging config must contain level, file, max_bytes, retained_files.")
    if settings["level"] not in ("INFO", "DEBUG"):
        raise ValueError("Logging level must be INFO or DEBUG.")
    if not isinstance(settings["file"], str) or not settings["file"].strip():
        raise ValueError("Logging file must be a non-empty path.")
    if type(settings["max_bytes"]) is not int or settings["max_bytes"] < 1024:
        raise ValueError("Logging max_bytes must be an integer of at least 1024.")
    if type(settings["retained_files"]) is not int or settings["retained_files"] < 2:
        raise ValueError("Logging retained_files must be an integer of at least 2.")
    log_path = path.parent / settings["file"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_path, maxBytes=settings["max_bytes"],
        backupCount=settings["retained_files"] - 1, encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s pid=%(process)d thread=%(threadName)s %(message)s"
    ))
    for previous in LOGGER.handlers[:]:
        LOGGER.removeHandler(previous)
        previous.close()
    LOGGER.addHandler(handler)
    LOGGER.setLevel(settings["level"])
    LOGGER.info("Logging configured level=%s max_bytes=%d retained_files=%d",
                settings["level"], settings["max_bytes"], settings["retained_files"])
    return log_path


def log_failure(operation: str, error: Exception) -> None:
    """Log exception type and stack locations, never exception text or locals."""
    LOGGER.error("%s failed error_type=%s", operation, type(error).__name__)
    if LOGGER.isEnabledFor(logging.DEBUG):
        for frame in traceback.extract_tb(error.__traceback__):
            LOGGER.debug("Failure frame file=%s line=%d function=%s",
                         frame.filename, frame.lineno, frame.name)


def log_activity(operation: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Record timing and failures without inspecting arguments or return values."""
    def decorate(function: Callable[P, T]) -> Callable[P, T]:
        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            started = perf_counter()
            LOGGER.debug("%s started", operation)
            try:
                result = function(*args, **kwargs)
            except Exception as error:
                log_failure(operation, error)
                raise
            finally:
                LOGGER.debug("%s finished elapsed_ms=%.2f",
                             operation, (perf_counter() - started) * 1000)
            return result
        return wrapped
    return decorate
