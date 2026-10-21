import json
from logging.handlers import RotatingFileHandler
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

import app_logging
import chatbot_tech_doc as app


@pytest.fixture(autouse=True)
def restore_logger():
    logger = app_logging.LOGGER
    handlers, level = logger.handlers[:], logger.level
    logger.handlers = []
    yield
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logger.handlers = handlers
    logger.setLevel(level)


def configure(tmp_path: Path, **changes) -> Path:
    settings = {
        "level": "DEBUG", "file": "logs/test.log",
        "max_bytes": 5242880, "retained_files": 5,
    }
    settings.update(changes)
    path = tmp_path / "logging.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    return app_logging.configure_logging(path)


def test_project_configuration_is_valid():
    settings = json.loads(app.LOG_CONFIG_PATH.read_text(encoding="utf-8"))
    assert set(settings) == {"level", "file", "max_bytes", "retained_files"}
    assert settings["level"] in ("INFO", "DEBUG")
    assert isinstance(settings["file"], str) and settings["file"]
    assert type(settings["max_bytes"]) is int and settings["max_bytes"] >= 1024
    assert type(settings["retained_files"]) is int and settings["retained_files"] >= 2


@pytest.mark.parametrize("level, includes_debug", [("INFO", False), ("DEBUG", True)])
def test_levels_and_format(tmp_path, level, includes_debug):
    path = configure(tmp_path, level=level)
    app_logging.LOGGER.info("Lifecycle event")
    app_logging.LOGGER.debug("Detailed event")
    text = path.read_text(encoding="utf-8")
    assert "INFO archive_it pid=" in text
    assert "thread=" in text
    assert "Lifecycle event" in text
    assert ("Detailed event" in text) is includes_debug


def test_rotation_keeps_five_total_bounded_files(tmp_path):
    path = configure(tmp_path, max_bytes=1024)
    for index in range(60):
        app_logging.LOGGER.info("Record %03d %s", index, "x" * 150)
    files = sorted(path.parent.glob("test.log*"))
    assert {file.name for file in files} == {
        "test.log", "test.log.1", "test.log.2", "test.log.3", "test.log.4",
    }
    assert all(0 < file.stat().st_size <= 1024 for file in files)
    assert "Record 059" in path.read_text()
    assert "Record 000" not in "".join(file.read_text() for file in files)


def test_reconfiguration_replaces_handler(tmp_path):
    path = configure(tmp_path)
    previous = app_logging.LOGGER.handlers[0]
    configure(tmp_path, level="INFO")
    assert len(app_logging.LOGGER.handlers) == 1
    assert isinstance(previous, RotatingFileHandler)
    assert previous.stream is None
    app_logging.LOGGER.info("Single record")
    assert path.read_text().count("Single record") == 1


@pytest.mark.parametrize("changes", [
    {"level": "TRACE"}, {"level": 10}, {"file": ""}, {"file": None},
    {"max_bytes": 0}, {"max_bytes": 1023}, {"max_bytes": True},
    {"retained_files": 1}, {"retained_files": -1}, {"retained_files": 5.0},
    {"extra": "invalid"},
])
def test_invalid_configuration_fails_explicitly(tmp_path, changes):
    previous_handlers = app_logging.LOGGER.handlers[:]
    with pytest.raises(ValueError):
        configure(tmp_path, **changes)
    assert app_logging.LOGGER.handlers == previous_handlers


@pytest.mark.parametrize("contents", ["{", "[]", "{}"])
def test_malformed_or_incomplete_configuration(tmp_path, contents):
    config = tmp_path / "bad.json"
    config.write_text(contents)
    with pytest.raises(ValueError):
        app_logging.configure_logging(config)


def test_missing_configuration_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        app_logging.configure_logging(tmp_path / "missing.json")


def test_log_file_open_failure_is_not_silently_ignored(tmp_path):
    (tmp_path / "logs").write_text("Not a directory")
    with pytest.raises(OSError):
        configure(tmp_path)


def test_index_build_reuse_and_failure_are_logged(tmp_path):
    log = configure(tmp_path)
    source = tmp_path / "private-policy.html"
    source.write_text("<table><tr><td>SECRET-POLICY</td><td>SECRET-CLAUSE</td></tr></table>")
    embedding = FakeEmbeddings(size=10)
    app.get_vector_store(embedding, tmp_path / "index", source_path=source)
    app.get_vector_store(embedding, tmp_path / "index", source_path=source)
    source.write_text("<table><tr><td>Changed</td><td>Updated</td></tr></table>")
    with pytest.raises(app.IndexMismatchError):
        app.get_vector_store(embedding, tmp_path / "index", source_path=source)
    text = log.read_text()
    for event in [
        "Source extraction started", "Source extracted documents=1",
        "Embedding and storing documents=1", "Index build completed",
        "Validated index reused without re-embedding", "error_type=IndexMismatchError",
    ]:
        assert event in text
    for private in ["SECRET-POLICY", "SECRET-CLAUSE", source.name]:
        assert private not in text


def test_query_stages_logged_without_personal_content(tmp_path):
    log = configure(tmp_path)
    docs = [Document(page_content="SECRET-DOCUMENT", metadata={
        "source": "SECRET-FILENAME", "warning": "SECRET-WARNING",
    })]
    model = Mock(side_effect=[
        AIMessage(content="SECRET-REWRITTEN-QUESTION", response_metadata={
            "done_reason": "stop", "eval_count": 12, "prompt_eval_count": 34,
            "total_duration": 1000, "model": "SECRET-MODEL",
        }),
        AIMessage(content="SECRET-ANSWER [1]"),
    ])
    chain = app.create_rag_chain(RunnableLambda(lambda _: docs), RunnableLambda(model))
    result = chain.invoke({
        "question": "SECRET-QUESTION",
        "chat_history": [HumanMessage(content="SECRET-HISTORY"), AIMessage(content="SECRET-OLD-ANSWER")],
    })
    assert result.startswith("SECRET-ANSWER")
    text = log.read_text()
    for event in [
        "Input validation started", "Query rewrite model request started",
        "Retrieval completed", "Evidence documents=1",
        "Answer model request completed", "Source citation formatting",
        "output_chars=25", "completion_reason=stop", "eval_count=12",
        "prompt_eval_count=34", "total_duration=1000", "cleaned_chars=25",
        "elapsed_ms=",
    ]:
        assert event in text
    assert "SECRET-" not in text


def test_untrusted_completion_metadata_is_not_logged(tmp_path):
    log = configure(tmp_path)
    runnable = app.logged_runnable("Answer model request", RunnableLambda(
        lambda _: AIMessage(content="SECRET-TEXT", response_metadata={
            "done_reason": "SECRET-REASON", "eval_count": "SECRET-COUNT",
        })
    ))
    runnable.invoke("SECRET-QUESTION")
    assert "completion_reason=unknown" in log.read_text()
    assert "SECRET-" not in log.read_text()


def test_empty_retrieval_logged_without_model_call(tmp_path):
    log = configure(tmp_path)
    model = Mock()
    chain = app.create_rag_chain(RunnableLambda(lambda _: []), RunnableLambda(model))
    assert chain.invoke("Question") == app.NO_CONTEXT_ANSWER
    model.assert_not_called()
    assert "No retrieved evidence; skipping answer generation" in log.read_text()


def test_failure_logs_safe_stack_and_preserves_exception(tmp_path):
    log = configure(tmp_path)
    error = httpx.ConnectError("SECRET-BACKEND-RESPONSE")

    @app_logging.log_activity("Test operation")
    def fail():
        raise error

    with pytest.raises(httpx.ConnectError) as raised:
        fail()
    assert raised.value is error
    text = log.read_text()
    assert "error_type=ConnectError" in text
    assert "Failure frame" in text
    assert "function=fail" in text
    assert "SECRET-BACKEND-RESPONSE" not in text


def test_cli_activity_and_service_failure(tmp_path, monkeypatch):
    log = configure(tmp_path)
    chain = Mock()
    chain.invoke.side_effect = [httpx.ConnectError("SECRET-ERROR"), "SECRET-ANSWER"]
    monkeypatch.setattr(app, "main", Mock(return_value=chain))
    monkeypatch.setattr("builtins.input", Mock(side_effect=["SECRET-QUESTION", "Retry", "exit"]))
    app.run_cli()
    text = log.read_text()
    assert "CLI session started" in text
    assert "Query turn=1 started" in text
    assert "CLI query service request failed error_type=ConnectError" in text
    assert "Query turn=2 completed" in text
    assert "CLI session ended attempted_turns=2" in text
    assert "SECRET-" not in text


def test_runnable_configuration_forwarded(tmp_path):
    configure(tmp_path)

    def stage(value, config):
        assert config["configurable"]["test_flag"] == 42
        return value

    runnable = app.logged_runnable("Test stage", RunnableLambda(stage))
    assert runnable.invoke("unchanged", config={"configurable": {"test_flag": 42}}) == "unchanged"
