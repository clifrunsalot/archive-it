import json
import logging
import os
import signal
import subprocess
import sys
import time
from contextlib import closing
from unittest.mock import Mock

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from document_facts import list_facts, review_fact, save_fact
from document_library import DocumentLibrary, DocumentQueryError
from document_query import QuerySession
from document_search import DocumentSearch
from document_workspace import run_workspace


@pytest.fixture
def context(tmp_path):
    library = DocumentLibrary(tmp_path / "catalogue.sqlite3")
    source = tmp_path / "sample.txt"
    source.write_text("Rent USD 10 monthly.")
    identifier = library.ingest(source)
    saved = save_fact(library, {
        "document_id": identifier, "version_id": library.snapshots([identifier])[0].version_id,
        "name": "Rent", "kind": "money", "value": "10", "currency": "USD", "period": "monthly",
        "unit": "", "conditions": "", "uncertainty": "", "candidate_id": "",
        "support": [{"segment": 1, "char_start": 0, "char_end": 20, "quote": source.read_text()}],
    })
    review_fact(library, saved["fact_id"], 1, "confirmed")
    return library, identifier


def script(monkeypatch, values):
    sequence = iter(values)

    def answer(message):
        try:
            value = next(sequence)
        except StopIteration:
            pytest.fail("Unexpected prompt: " + message)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr("builtins.input", answer)


def factory_for(context):
    library, identifier = context
    fact = list_facts(library, [identifier])[0]
    responses = iter([
        AIMessage(content=json.dumps({
            "intent": "fact_query", "search_question": "Sum confirmed rent", "document_ids": None,
            "presentation": "prose", "clarification": None, "filters": [],
        })),
        AIMessage(content=json.dumps({
            "operation": "sum", "fact_ids": [fact["fact_id"]],
            "lower": None, "upper": None, "clarification": None,
        })),
    ])
    session = QuerySession(
        DocumentSearch(library, Mock(spec=Embeddings), "test"),
        RunnableLambda(lambda _: next(responses)),
        RunnableLambda(lambda _: pytest.fail("Answer model must not run")), [identifier],
    )
    factory = Mock(return_value=session)
    return factory, session


def test_browsing_and_fact_review_are_model_free(context, monkeypatch, capsys):
    factory = Mock(side_effect=AssertionError("No services needed"))
    script(monkeypatch, ["sample.txt", "facts", "quit", "calculate", "1", "sum", "cancel", "quit"])
    run_workspace(context[0], factory)
    factory.assert_not_called()
    output = capsys.readouterr().out
    assert "current version not indexed" in output and '"confirmed": 1' in output
    assert "Calculation cancelled" in output
    assert '"eligible": 1' in output


def test_explicit_menu_approval_executes_guarded_query(context, monkeypatch, capsys):
    factory, session = factory_for(context)
    script(monkeypatch, ["1", "ask", "Sum confirmed rent", "selection", "1", "approve", "run", "quit"])
    run_workspace(context[0], factory)
    output = capsys.readouterr().out
    assert '"value": "10"' in output
    assert session.pending_fact_query is None
    factory.assert_called_once_with(context[1])


@pytest.mark.parametrize("action", ["cancel-query", "documents", "facts"])
def test_leaving_proposal_discards_approval(context, monkeypatch, action, capsys):
    factory, session = factory_for(context)
    ending = {
        "cancel-query": ["approve", "quit"],
        "documents": ["1", "approve", "quit"],
        "facts": ["quit", "approve", "quit"],
    }[action]
    script(monkeypatch, ["1", "ask", "Sum confirmed rent", action, *ending])
    run_workspace(context[0], factory)
    assert "No pending fact preview" in capsys.readouterr().err
    assert session.pending_fact_query is None


def test_stale_review_prevents_menu_approval(context, monkeypatch, capsys):
    factory, _ = factory_for(context)
    script(monkeypatch, ["1", "ask", "Sum confirmed rent", "approve", "run", "quit"])
    import builtins
    original = builtins.input

    def change(message):
        answer = original(message)
        if answer == "run":
            fact = list_facts(context[0], [context[1]])[0]
            review_fact(context[0], fact["fact_id"], 1, "confirmed")
        return answer

    monkeypatch.setattr("builtins.input", change)
    run_workspace(context[0], factory)
    captured = capsys.readouterr()
    assert "catalogue changed" in captured.err
    assert "- Result:" not in captured.out


def test_duplicate_filenames_require_number(context, tmp_path, monkeypatch, capsys):
    path = tmp_path / "other" / "sample.txt"
    path.parent.mkdir()
    path.write_text("Other document.")
    context[0].ingest(path)
    script(monkeypatch, ["sample.txt", "1", "quit"])
    factory = Mock()
    run_workspace(context[0], factory)
    assert "No unique match" in capsys.readouterr().out
    factory.assert_not_called()


def test_search_index_and_summary_keep_document_scope(context, monkeypatch):
    import document_synthesis
    from document_search import AnswerResult

    factory, session = factory_for(context)
    session.search.index = Mock(return_value={"indexed_versions": 1})
    session.search.search = Mock(return_value={"passages": []})
    synthesis = Mock(return_value=AnswerResult("Summary", [], {}))
    monkeypatch.setattr(document_synthesis, "synthesize", synthesis)
    script(monkeypatch, ["1", "index", "search", "Rent", "summarize", "Terms", "quit"])
    run_workspace(context[0], factory)
    session.search.index.assert_called_once_with([context[1]])
    session.search.search.assert_called_once_with("Rent", [context[1]])
    assert synthesis.call_args.args[-1] == [context[1]]
    factory.assert_called_once()


def test_multi_document_scope_search_summary_and_compare(context, monkeypatch, capsys):
    import document_synthesis
    from document_search import AnswerResult

    library, first_id = context
    second_path = library.database.parent / "addendum.txt"
    second_path.write_text("The rent is USD 20 monthly.")
    second_id = library.ingest(second_path)
    factory, session = factory_for(context)
    record = workspace_evidence(context)
    session.search.search = Mock(return_value={
        "passages": [record], "coverage": {"selected_documents": 2},
    })
    synthesis = Mock(return_value=AnswerResult(
        "The amounts differ. [1]", [record], {"documents": []},
    ))
    monkeypatch.setattr(document_synthesis, "synthesize", synthesis)

    script(monkeypatch, [
        "1", "scope", "1,2", "search", "monthly rent", "summarize", "Summarize rent",
        "compare", "Compare the rent terms", "quit",
    ])
    run_workspace(library, factory)

    scope_ids = [entry["id"] for entry in library.list_documents()]
    assert session.search.search.call_args.args == ("monthly rent", scope_ids)
    assert [call.args[2:] for call in synthesis.call_args_list] == [
        ("summarize", "Summarize rent", scope_ids),
        ("compare", "Compare the rent terms", scope_ids),
    ]
    output = capsys.readouterr().out
    assert "Active scope (2 document(s))" in output
    assert str(second_path) in output
    assert factory.call_count == 1


def test_fact_workbench_requires_single_document_scope(context, monkeypatch, capsys):
    library, first_id = context
    second_path = library.database.parent / "addendum.txt"
    second_path.write_text("The rent is USD 20 monthly.")
    library.ingest(second_path)
    factory = Mock(side_effect=AssertionError("Fact scope check must not create models"))

    script(monkeypatch, ["1", "scope", "1,2", "facts", "scope", "1", "quit"])
    run_workspace(library, factory)

    assert "Fact editing requires a single-document scope" in capsys.readouterr().err
    factory.assert_not_called()


def test_action_error_is_visible_and_returns_to_menu(context, monkeypatch, capsys):
    factory = Mock(side_effect=DocumentQueryError("Test service failure"))
    script(monkeypatch, ["1", "index", "quit"])
    run_workspace(context[0], factory)
    assert "Test service failure" in capsys.readouterr().err


@pytest.mark.parametrize("ending", [EOFError(), KeyboardInterrupt(), "/cancel"])
def test_safe_exit_or_cancel_without_services(context, monkeypatch, ending):
    factory = Mock()
    script(monkeypatch, ["1", ending, "quit"])
    run_workspace(context[0], factory)
    factory.assert_not_called()


def test_empty_catalogue_exits_without_services(tmp_path, capsys):
    factory = Mock()
    run_workspace(DocumentLibrary(tmp_path / "empty.sqlite3"), factory)
    assert "No documents" in capsys.readouterr().out
    factory.assert_not_called()


def test_cli_workspace_uses_selected_database_without_ollama(context):
    result = subprocess.run([
        sys.executable, "document_library.py", "--database", str(context[0].database), "workspace",
    ], input="1\nfacts\nquit\nquit\n", text=True, capture_output=True, check=True)
    assert "sample.txt" in result.stdout and '"confirmed": 1' in result.stdout
    assert "Local fact review; no model calls" in result.stdout


def test_approval_defaults_to_cancel_not_run(context, monkeypatch, capsys):
    factory, session = factory_for(context)
    script(monkeypatch, ["1", "ask", "Sum confirmed rent", "approve", "", "quit"])
    run_workspace(context[0], factory)
    output = capsys.readouterr().out
    assert "Fact query cancelled" in output and "- Result:" not in output
    assert session.pending_fact_query is None


def test_blocked_preview_does_not_offer_run(context, monkeypatch, capsys):
    fact = list_facts(context[0], [context[1]])[0]
    # Confirmed but uncertain facts cannot be calculated.
    changed = save_fact(context[0], {**fact["payload"], "uncertainty": "Check interpretation"},
                        fact_id=fact["fact_id"], expected_revision=1)
    review_fact(context[0], changed["fact_id"], 2, "confirmed")
    factory, session = factory_for(context)
    # Build a blocked preview through the same API used for manual selections.
    from document_facts import preview_fact_query
    pending = {
        "catalogue": list_facts(context[0], [context[1]]),
        "preview": preview_fact_query(context[0], [fact["fact_id"]], "sum"),
        "presentation": "prose",
    }
    def propose(question):
        assert question == "Sum confirmed rent"
        session.pending_fact_query = pending
        return "Blocked preview"

    session.execute = Mock(side_effect=propose)
    script(monkeypatch, ["1", "ask", "Sum confirmed rent", "approve", "quit"])
    run_workspace(context[0], factory)
    assert "Approval unavailable" in capsys.readouterr().out


def test_service_failure_is_reported_without_switching_scope(context, monkeypatch, capsys):
    from ollama import ResponseError

    factory = Mock(side_effect=ResponseError("PRIVATE_SERVER_MESSAGE", status_code=503))
    script(monkeypatch, ["1", "index", "quit"])
    run_workspace(context[0], factory)
    output = capsys.readouterr()
    assert "Local Ollama request failed" in output.err
    assert "PRIVATE_SERVER_MESSAGE" not in output.err
    factory.assert_called_once_with(context[1])


def test_failed_document_is_visible_but_not_queryable(context, monkeypatch, capsys):
    library, _ = context
    with closing(library.connect()) as connection, connection:
        connection.execute("UPDATE documents SET current_version=NULL")
    script(monkeypatch, ["1", ""])
    factory = Mock()
    run_workspace(library, factory)
    assert "no successful extraction" in capsys.readouterr().out
    factory.assert_not_called()


def test_navigation_does_not_log_private_contents(context, monkeypatch, caplog):
    script(monkeypatch, ["1", "facts", "quit", "quit"])
    with caplog.at_level(logging.INFO, logger="archive_it"):
        run_workspace(context[0], Mock())
    assert "sample.txt" not in caplog.text and "Rent USD 10 monthly." not in caplog.text


@pytest.mark.parametrize("override", [None, "qwen3.5:4b"])
def test_workspace_answer_default_and_explicit_model_override(context, monkeypatch, override):
    import chatbot_tech_doc
    import document_library

    observed = []
    monkeypatch.setattr(chatbot_tech_doc, "get_llm", lambda name: observed.append(name) or
                        RunnableLambda(lambda _: AIMessage(content="unused")))
    monkeypatch.setattr(chatbot_tech_doc, "get_embeddings", lambda name: Mock(spec=Embeddings))
    import document_workspace
    monkeypatch.setattr(document_workspace, "run_workspace", lambda library, create: create(context[1]))
    args = ["document_library.py", "--database", str(context[0].database), "workspace"]
    if override:
        args.extend(["--model", override])
    monkeypatch.setattr(sys, "argv", args)
    assert document_library.main() == 0
    assert observed[0] == (override or "deepseek-r1:8b")


def workspace_evidence(context):
    library, identifier = context
    snapshot = library.snapshots([identifier])[0]
    return {
        "label": 1, "document_id": identifier, "version_id": snapshot.version_id,
        "source_path": snapshot.source_path, "segment": 1, "source": "sample.txt",
        "location": snapshot.extraction["segments"][0]["location"],
        "text": snapshot.extraction["segments"][0]["text"],
        "quality": {"status": "extracted", "issues": []},
        "latest_attempt_status": snapshot.latest_attempt_status,
    }


@pytest.mark.parametrize("action", ["search", "ask", "summarize"])
def test_completed_query_sources_are_inspectable_without_new_model_calls(context, monkeypatch, capsys, action):
    from document_search import AnswerResult
    import document_synthesis

    factory, session = factory_for(context)
    record = workspace_evidence(context)
    completed = AnswerResult("Rent is USD 10. [1]", [record], {})
    session.search.search = Mock(return_value={"passages": [record], "coverage": {}})

    def execute(question):
        session.previous_result = completed
        return completed.answer

    session.execute = Mock(side_effect=execute)
    summary = Mock(return_value=completed)
    monkeypatch.setattr(document_synthesis, "synthesize", summary)
    script(monkeypatch, ["1", action, "Rent", "inspect", "1", "", "", "quit"])
    run_workspace(context[0], factory)
    captured = capsys.readouterr()
    assert "Original supplied passage [1]" in captured.out
    assert "Rent USD 10 monthly." in captured.out
    assert f"Progress: {action}" in captured.err and "completed" in captured.err
    assert factory.call_count == 1
    assert session.execute.call_count == int(action == "ask")
    assert summary.call_count == int(action == "summarize")
    assert session.search.search.call_count == int(action == "search")


def test_readable_search_numbers_match_inspector_without_modifying_search_response(context, monkeypatch, capsys):
    factory, session = factory_for(context)
    first = workspace_evidence(context)
    second = dict(first, text="Second displayed candidate.")
    found = {"passages": [first, second], "coverage": {"exhaustive": False}}
    before = json.dumps(found)
    session.search.search = Mock(return_value=found)
    script(monkeypatch, ["1", "search", "Rent", "inspect", "2", "", "", "quit"])
    run_workspace(context[0], factory)
    output = capsys.readouterr().out
    assert "Search results: 2 passage(s)" in output
    assert "[1] sample.txt" in output and "[2] sample.txt" in output
    assert "Original supplied passage [2] (not model-generated)" in output
    assert output.count("Second displayed candidate.") == 2
    assert json.dumps(found) == before
    assert '"passages":' not in output
    session.search.search.assert_called_once_with("Rent", [context[1]])


@pytest.mark.parametrize("action", ["index", "search", "ask", "summarize"])
def test_interrupting_service_returns_to_menu_and_clears_pending_and_partial_state(
    context, monkeypatch, capsys, action,
):
    import document_synthesis
    from document_search import AnswerResult

    factory, session = factory_for(context)

    def interrupted(*args, **kwargs):
        session.pending_fact_query = {"partial": True}
        session.previous_result = AnswerResult("SECRET INCOMPLETE", [], {})
        raise KeyboardInterrupt

    session.execute = Mock(side_effect=interrupted)
    session.search.index = Mock(side_effect=interrupted)
    session.search.search = Mock(side_effect=interrupted)
    monkeypatch.setattr(document_synthesis, "synthesize", interrupted)
    script(monkeypatch, ["1", action, *([] if action == "index" else ["Rent"]),
                        "inspect", "approve", "quit"])
    run_workspace(context[0], factory)
    captured = capsys.readouterr()
    assert "Action interrupted" in captured.out
    assert "No completed evidence result" in captured.err
    assert "No pending fact preview" in captured.err
    assert "SECRET INCOMPLETE" not in captured.out + captured.err
    assert "cancelled" in captured.err
    assert session.pending_fact_query is None and session.previous_result is None
    assert list_facts(context[0], [context[1]])[0]["decision"] == "confirmed"


def test_failed_new_query_cannot_expose_previous_inspection_result(context, monkeypatch, capsys):
    factory, session = factory_for(context)
    session.search.search = Mock(side_effect=[
        {"passages": [workspace_evidence(context)], "coverage": {}},
        DocumentQueryError("Service unavailable"),
    ])
    script(monkeypatch, ["1", "search", "Rent", "search", "Other", "inspect", "quit"])
    run_workspace(context[0], factory)
    captured = capsys.readouterr()
    assert "No completed evidence result" in captured.err
    assert "Original supplied passage" not in captured.out


def test_document_navigation_discards_inspection_result(context, monkeypatch, capsys):
    factory, session = factory_for(context)
    session.search.search = Mock(return_value={"passages": [workspace_evidence(context)], "coverage": {}})
    script(monkeypatch, ["1", "search", "Rent", "documents", "1", "inspect", "quit"])
    run_workspace(context[0], factory)
    assert "No completed evidence result" in capsys.readouterr().err


def test_inspection_with_no_result_remains_model_free(context, monkeypatch, capsys):
    factory = Mock(side_effect=AssertionError("No model client needed"))
    script(monkeypatch, ["1", "inspect", "quit"])
    run_workspace(context[0], factory)
    factory.assert_not_called()
    assert "No completed evidence result" in capsys.readouterr().err


def test_reformatted_result_is_inspectable_but_clarification_does_not_republish_old_sources(
    context, monkeypatch, capsys,
):
    from document_search import AnswerResult

    factory, session = factory_for(context)
    completed = AnswerResult("Rent is USD 10. [1]", [workspace_evidence(context)], {})
    session.search.answer = Mock(return_value=completed)
    base = {
        "search_question": "Rent", "document_ids": None, "presentation": "prose",
        "clarification": None, "filters": [],
    }
    responses = iter([
        AIMessage(content=json.dumps(dict(base, intent="extract_fact"))),
        AIMessage(content=json.dumps(dict(base, intent="reformat", presentation="csv"))),
        AIMessage(content=json.dumps(dict(base, intent="clarify", clarification="Which clause?"))),
    ])
    session.planner = RunnableLambda(lambda _: next(responses))
    script(monkeypatch, ["1", "ask", "Rent", "ask", "Use CSV", "inspect", "1", "", "",
                        "ask", "Ambiguous", "inspect", "quit"])
    run_workspace(context[0], factory)
    captured = capsys.readouterr()
    assert captured.out.count("Original supplied passage [1]") == 1
    assert "Clarification needed: Which clause?" in captured.out
    assert captured.err.count("No completed evidence result") == 1
    assert session.previous_result is completed
    session.search.answer.assert_called_once()


def test_cancel_prompt_discards_prior_query_state_and_inspection(context, monkeypatch, capsys):
    from document_search import AnswerResult

    factory, session = factory_for(context)

    def execute(question):
        session.previous_result = AnswerResult("Rent [1]", [workspace_evidence(context)], {})
        return "Rent [1]"

    session.execute = Mock(side_effect=execute)
    script(monkeypatch, ["1", "ask", "Rent", "ask", "/cancel", "inspect", "quit"])
    run_workspace(context[0], factory)
    assert session.previous_result is None and session.previous_plan is None
    assert session.previous_versions == {}
    assert "No completed evidence result" in capsys.readouterr().err


def test_inspection_cannot_approve_or_discard_pending_fact_preview(context, monkeypatch, capsys):
    factory, _ = factory_for(context)
    script(monkeypatch, ["1", "ask", "Sum confirmed rent", "inspect", "approve", "run", "quit"])
    run_workspace(context[0], factory)
    captured = capsys.readouterr()
    assert "No completed evidence result" in captured.err
    assert '"value": "10"' in captured.out
    assert "No pending fact preview" not in captured.err


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX Ctrl+C signal integration")
def test_real_sigint_interrupts_active_service_and_workspace_remains_usable(context, tmp_path):
    ready = tmp_path / "ready"
    program = """
import sys
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from document_library import DocumentLibrary
from document_workspace import run_workspace

def blocked(*args):
    Path(sys.argv[2]).write_text("ready")
    Event().wait(30)
    raise AssertionError("Active action was not interrupted")

session = SimpleNamespace(model=object(), pending_fact_query=None,
                          search=SimpleNamespace(embedding=object(), index=blocked))
run_workspace(DocumentLibrary(sys.argv[1]), lambda _: session)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", program, str(context[0].database), str(ready)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert process.stdin is not None
        process.stdin.write("1\nindex\ninspect\nquit\n")
        process.stdin.flush()
        deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "Child did not reach active service call"
        os.kill(process.pid, signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0
        assert "Action interrupted" in stdout
        assert "No completed evidence result" in stderr and "cancelled" in stderr
        assert "Traceback" not in stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
