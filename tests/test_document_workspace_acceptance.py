"""Synthetic end-to-end workspace workflows with isolated catalogues."""

import json
import os
from contextlib import closing
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

from chatbot_tech_doc import get_llm
from document_library import DocumentLibrary
from document_query import QuerySession, get_planner
from document_search import AnswerResult, DocumentSearch
from document_workspace import run_workspace
from test_document_acceptance import (
    Case, FlatEmbeddings, check_claims, check_evidence, normalized, write_document,
)
from test_document_workspace import script


PARAGRAPHS = (
    "RENT: Monthly rent is USD 1200. Parking costs USD 50 per month. "
    "The refundable deposit is USD 500, payable once.",
    "UTILITIES: The tenant pays electricity and internet. The landlord pays water.",
    "TERM: The lease begins 2027-01-01 and ends 2027-12-31.",
    "STORAGE: The lease states USD 40 monthly; the addendum states USD 65 monthly. "
    "The amounts conflict. It is unclear which controls.",
)
SUMMARY = "Summarize the rent, parking, deposit, utilities, term dates and conflicting storage fees."
SUMMARY_CASE = Case(
    "workspace-summary", SUMMARY, PARAGRAPHS, PARAGRAPHS,
    tuple((pattern, support) for pattern, support in (
        ("1200", "USD 1200"), ("50", "USD 50"), ("500", "USD 500"),
        ("electricity", "electricity"), ("internet", "internet"), ("water", "water"),
        ("2027-01-01", "2027-01-01"), ("2027-12-31", "2027-12-31"),
        ("40", "USD 40"), ("65", "USD 65"), ("conflict", "conflict"),
        ("unclear", "unclear"),
    )),
    ("addendum controls", "supersedes"),
    qualifiers=((r"deposit", r"refundable"), (r"deposit", r"\bonce\b|\bone[- ]time\b")),
)


@pytest.fixture(params=["txt", "docx", "pdf"])
def workspace_corpus(request, tmp_path):
    library = DocumentLibrary(tmp_path / "catalogue.sqlite3")
    path = tmp_path / ("selected." + request.param)
    write_document(path, PARAGRAPHS)
    identifier = library.ingest(path)
    other = tmp_path / ("unselected." + request.param)
    write_document(other, ("Rent USD 9999. The addendum controls. Lease ends 2099-12-31.",))
    library.ingest(other)
    return library, identifier, path


def planner_response(intent, question):
    return AIMessage(content=json.dumps({
        "intent": intent, "search_question": question, "document_ids": None,
        "presentation": "prose", "clarification": None, "filters": [],
    }))


def summary_response(prompt):
    records = json.loads(prompt.to_messages()[-1].text.split("(JSON):\n", 1)[1])
    return AIMessage(content=json.dumps({
        "status": "findings",
        "findings": [{"text": " ".join(record["text"].split()), "citations": [record["label"]]}
                     for record in records],
    }))


@pytest.mark.parametrize("defect", [
    "missing-date", "lost-condition", "wrong-citation", "false-precedence", "monthly-deposit",
])
def test_summary_oracle_rejects_incomplete_or_unsupported_output(defect):
    rows = [f"{paragraph} [{index}]" for index, paragraph in enumerate(PARAGRAPHS, 1)]
    if defect == "missing-date":
        rows[2] = "TERM: The lease begins 2027-01-01. [3]"
    elif defect == "lost-condition":
        rows[0] = rows[0].replace("refundable", "")
    elif defect == "wrong-citation":
        rows[0] = rows[0].replace("[1]", "[2]")
    elif defect == "false-precedence":
        rows[3] += " The addendum controls. [4]"
    else:
        rows[0] = rows[0].replace("payable once", "payable monthly")
    result = AnswerResult("\n".join(rows), [
        {"label": index, "text": paragraph} for index, paragraph in enumerate(PARAGRAPHS, 1)
    ], {})
    with pytest.raises(AssertionError):
        check_claims(SUMMARY_CASE, result)


@pytest.mark.parametrize("frequency", ["payable once", "a one-time payment", "a one time payment"])
def test_summary_oracle_accepts_equivalent_single_payment_wording(frequency):
    result = AnswerResult("\n".join(
        f"{paragraph.replace('payable once', frequency)} [{index}]"
        for index, paragraph in enumerate(PARAGRAPHS, 1)
    ), [{"label": index, "text": paragraph}
        for index, paragraph in enumerate(PARAGRAPHS, 1)], {})
    check_claims(SUMMARY_CASE, result)


@pytest.mark.parametrize("intent", ["search", "extract_fact", "summarize"])
def test_ask_routes_through_real_services_then_inspects_original_evidence(
    workspace_corpus, monkeypatch, capsys, intent,
):
    library, identifier, path = workspace_corpus
    engine = DocumentSearch(library, FlatEmbeddings(), "workspace-flat-v1")
    engine.index([identifier])
    question = SUMMARY if intent == "summarize" else "What is the monthly rent?"
    planner = Mock(return_value=planner_response(intent, question))

    def answer(prompt):
        if intent == "summarize":
            return summary_response(prompt)
        records, _ = json.JSONDecoder().raw_decode(
            prompt.to_messages()[-1].text.split("Evidence (JSON records):\n", 1)[1])
        source = next(record for record in records if "USD 1200" in record["text"])
        return AIMessage(content=json.dumps({
            "status": "findings",
            "findings": [{"support": [source["label"]], "text": "Monthly rent is USD 1200."}],
        }))

    model = Mock(side_effect=answer)
    session = QuerySession(engine, RunnableLambda(planner), RunnableLambda(model), [identifier])
    script(monkeypatch, [path.name, "ask", question, "inspect", "1", "segment", "", "", "quit"])
    run_workspace(library, lambda _: session)
    output = capsys.readouterr()
    result = session.previous_result
    assert result is not None and session.previous_plan is not None
    assert session.previous_plan.intent == intent
    planner.assert_called_once()
    assert model.call_count == (0 if intent == "search" else 1)
    request = json.loads(planner.call_args.args[0].to_messages()[-1].text)
    assert [entry["id"] for entry in request["scope"]] == [identifier]
    assert "Original supplied passage [1]" in output.out
    assert "NOT evidence used for this answer" in output.out
    assert "USD 9999" not in output.out and "2099-12-31" not in output.out
    assert not output.err.count("failed")
    check_evidence(
        SUMMARY_CASE if intent == "summarize" else Case(
            "rent", question, PARAGRAPHS, ("USD 1200",), (),
        ), result.evidence, library, identifier,
    )
    if intent == "summarize":
        check_claims(SUMMARY_CASE, result)
        assert result.coverage["synthesis_strategy"] == "direct"


def test_workspace_stale_inspection_never_substitutes_new_source(
    workspace_corpus, monkeypatch, capsys,
):
    library, identifier, path = workspace_corpus
    session = QuerySession(
        DocumentSearch(library, FlatEmbeddings(), "workspace-flat-v1"),
        RunnableLambda(lambda _: planner_response("summarize", SUMMARY)),
        RunnableLambda(summary_response), [identifier],
    )
    actions = iter([path.name, "ask", SUMMARY, "inspect", "1", "", "quit"])

    def enter(_):
        value = next(actions)
        if value == "inspect":
            write_document(path, ("REPLACEMENT PRIVATE SOURCE: Rent USD 7777.",))
            library.ingest(path)
        return value

    monkeypatch.setattr("builtins.input", enter)
    run_workspace(library, lambda _: session)
    output = capsys.readouterr()
    assert "Source version is no longer current" in output.out
    assert "nearby navigation unavailable" in output.out
    assert "REPLACEMENT PRIVATE SOURCE" not in output.out
    assert session.previous_result is not None
    assert session.previous_result.evidence[0]["version_id"] != library.snapshots([identifier])[0].version_id


def test_workspace_source_change_during_summary_does_not_publish_result(
    workspace_corpus, monkeypatch, capsys,
):
    library, identifier, path = workspace_corpus

    def changed(prompt):
        response = summary_response(prompt)
        write_document(path, ("Replacement lease terms.",))
        library.ingest(path)
        return response

    session = QuerySession(
        DocumentSearch(library, FlatEmbeddings(), "workspace-flat-v1"),
        RunnableLambda(lambda _: planner_response("summarize", SUMMARY)),
        RunnableLambda(changed), [identifier],
    )
    script(monkeypatch, [path.name, "ask", SUMMARY, "inspect", "quit"])
    run_workspace(library, lambda _: session)
    output = capsys.readouterr()
    assert "Documents changed during synthesis" in output.err
    assert "No completed evidence result" in output.err
    assert session.previous_result is None
    assert "Original supplied passage" not in output.out


def test_workspace_rejects_planner_scope_expansion_before_generation(
    workspace_corpus, monkeypatch, capsys,
):
    library, identifier, path = workspace_corpus
    outside = next(entry["id"] for entry in library.list_documents() if entry["id"] != identifier)
    value = json.loads(planner_response("summarize", SUMMARY).text)
    value["document_ids"] = [identifier, outside]
    model = Mock(side_effect=AssertionError("Scope must be checked before generation"))
    session = QuerySession(
        DocumentSearch(library, FlatEmbeddings(), "workspace-flat-v1"),
        RunnableLambda(lambda _: AIMessage(content=json.dumps(value))),
        RunnableLambda(model), [identifier],
    )
    script(monkeypatch, [path.name, "ask", SUMMARY, "inspect", "quit"])
    run_workspace(library, lambda _: session)
    output = capsys.readouterr()
    assert "attempted to broaden scope" in output.err
    assert "No completed evidence result" in output.err
    assert session.previous_result is None
    model.assert_not_called()


def test_workspace_cancel_and_resume_reuses_only_validated_checkpoints(
    workspace_corpus, monkeypatch, capsys,
):
    library, identifier, path = workspace_corpus
    write_document(path, tuple(f"Section {number}: " + "Rent conditions apply. " * 45
                               for number in range(8)))
    library.ingest(path)
    client = get_llm("qwen3.5:4b")
    assert isinstance(client, ChatOllama)
    calls = 0

    def chat(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        records = json.loads(kwargs["messages"][-1]["content"].split("(JSON):\n", 1)[1])
        labels = [record["label"] for record in records] if "label" in records[0] else [
            label for record in records for finding in record["findings"]
            for label in finding["citations"]
        ]
        return iter([{
            "model": client.model, "message": {"role": "assistant", "content": json.dumps({
                "status": "findings", "findings": [{"text": "Rent conditions apply.", "citations": labels}],
            })}, "done": True, "done_reason": "stop",
        }])

    monkeypatch.setattr(client._client, "chat", chat)
    session = QuerySession(
        DocumentSearch(library, FlatEmbeddings(), "workspace-flat-v1"),
        RunnableLambda(lambda _: planner_response("summarize", SUMMARY)), client, [identifier],
    )

    def verify_cancelled():
        assert session.previous_result is None and session.pending_fact_query is None
        with closing(library.connect()) as connection:
            assert connection.execute("SELECT COUNT(*) FROM synthesis_checkpoints").fetchone()[0] == 1

    actions = iter([path.name, "ask", SUMMARY, "inspect", "ask", SUMMARY, "inspect", "1", "", "", "quit"])

    def enter(_):
        value = next(actions)
        if value == "inspect" and calls == 2:
            verify_cancelled()
        return value

    monkeypatch.setattr("builtins.input", enter)
    run_workspace(library, lambda _: session)
    output = capsys.readouterr()
    assert "Action interrupted" in output.out
    assert "No completed evidence result" in output.err
    result = session.previous_result
    assert result is not None
    assert result.coverage["synthesis_strategy"] == "hierarchical"
    assert result.coverage["reused_checkpoints"] == 1
    assert "Original supplied passage [1]" in output.out


@pytest.mark.skipif(os.getenv("RUN_WORKSPACE_SMOKE_TESTS") != "1",
                    reason="Opt-in serial live workspace smoke suite")
def test_live_workspace_planner_summary_and_inspection(
    workspace_corpus, monkeypatch, capsys, record_property,
):
    library, identifier, path = workspace_corpus
    session = QuerySession(
        DocumentSearch(library, FlatEmbeddings(), "workspace-flat-v1"),
        get_planner(), get_llm("deepseek-r1:8b"), [identifier],
    )
    script(monkeypatch, [path.name, "ask", SUMMARY, "inspect", "1", "", "", "quit"])
    run_workspace(library, lambda _: session)
    output = capsys.readouterr()
    assert session.previous_plan is not None, (
        "Workspace did not publish a successful plan/result.\n"
        f"Workspace errors:\n{output.err}\nWorkspace output:\n{output.out}"
    )
    assert session.previous_plan.intent == "summarize", (
        f"Unexpected route: {session.previous_plan.intent}\n{output.err}\n{output.out}"
    )
    result = session.previous_result
    assert result is not None, output.err
    check_claims(SUMMARY_CASE, result)
    assert result.coverage["synthesis_strategy"] == "direct"
    assert result.coverage["model_requests"] <= 2
    assert "Original supplied passage [1]" in output.out
    assert normalized(result.evidence[0]["text"]) in normalized(output.out)
    assert "USD 9999" not in output.out
    record_property("format", path.suffix)
    record_property("model_requests", result.coverage["model_requests"])
