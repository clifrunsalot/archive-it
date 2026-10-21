import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

import document_query as query
from app_logging import LOGGER, configure_logging
from document_library import DocumentLibrary, DocumentQueryError
from document_search import AnswerResult, DocumentSearch


def plan_message(**changes):
    fields = {
        "intent": "extract_fact", "search_question": "What is the monthly rent?",
        "document_ids": None, "presentation": "prose", "clarification": None, "filters": [],
    }
    fields.update(changes)
    return AIMessage(content=json.dumps(fields, indent=2), response_metadata={"done_reason": "stop"})


@pytest.fixture
def session(tmp_path):
    library = DocumentLibrary(tmp_path / "catalogue.sqlite3")
    source = tmp_path / "lease.txt"
    source.write_text("Rent is $1200 per month. Pets require permission.")
    identifier = library.ingest(source)
    embedding = Mock(spec=Embeddings)
    embedding.embed_documents.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
    embedding.embed_query.return_value = [1.0, 0.0]
    search = DocumentSearch(library, embedding, "synthetic")
    search.index()
    planner = Mock(return_value=plan_message())
    answer = Mock(return_value=AIMessage(content=json.dumps({
        "status": "findings",
        "findings": [{"support": [1],
                      "text": "Monthly rent is $1200"}],
    })))
    session = query.QuerySession(search, RunnableLambda(planner), RunnableLambda(answer), [identifier])
    return session, planner, answer, identifier


def test_multiline_json_plan_and_wrapped_search_intent_are_valid():
    result = query.parse_plan(plan_message(search_question="What is the\nmonthly rent?").text, set())
    assert result.search_question == "What is the monthly rent?"


def test_factual_plan_uses_original_question_if_model_omits_search_intent(session):
    service, planner, _, _ = session
    planner.return_value = plan_message(search_question="")

    result = service.plan("What is the monthly rent?")

    assert result.intent == "extract_fact"
    assert result.search_question == "What is the monthly rent?"


def test_query_session_rejects_empty_scope_instead_of_resetting_to_all_documents(session):
    service, _, _, identifier = session

    with pytest.raises(DocumentQueryError, match="Select at least one document"):
        service.select([])

    assert service.document_ids == [identifier]


@pytest.mark.parametrize("text", [
    "{", "[]", "{}", '{"intent":"extract_fact"}', "x" * 8001,
])
def test_malformed_plans_fail(text):
    with pytest.raises(DocumentQueryError):
        query.parse_plan(text, set())


@pytest.mark.parametrize("changes", [
    {"intent": "execute_sql"}, {"intent": {}}, {"presentation": "html"},
    {"search_question": ""}, {"search_question": 42}, {"search_question": "x" * 513},
    {"document_ids": ["unknown"]}, {"document_ids": []}, {"document_ids": "id"},
    {"clarification": ""}, {"clarification": 42}, {"intent": "clarify"},
    {"filters": "date"}, {"filters": [None]}, {"filters": ["x" * 201]},
])
def test_plan_schema_and_scope_validation(changes):
    with pytest.raises(DocumentQueryError):
        query.parse_plan(plan_message(**changes).text, set())


def test_format_is_separate_from_retrieval_and_ascii_is_deterministic(session):
    service, planner, answer, identifier = session
    planner.return_value = plan_message(presentation="ascii_table")
    result = service.execute("Show monthly rent as an ASCII table.")
    assert result.isascii()
    assert result.startswith("+")
    assert "$1200 [1]" in result
    assert "Coverage" in result
    assert identifier in result.replace("\n", "").replace("|", "").replace(" ", "")
    service.search.embedding.embed_query.assert_called_once_with("What is the monthly rent?")
    messages = answer.call_args.args[0].to_messages()
    assert "ASCII table" not in messages[-1].content


@pytest.mark.parametrize("intent", ["aggregate", "advice"])
def test_future_capabilities_never_retrieve_or_generate(session, intent):
    service, planner, answer, identifier = session
    planner.return_value = plan_message(intent=intent)
    result = service.execute("Complex request")
    assert "Not supported yet" in result
    assert "No query was executed" in result
    service.search.embedding.embed_query.assert_not_called()
    answer.assert_not_called()
    assert service.previous_plan is None


def test_enumeration_routes_to_all_passage_synthesis_without_search(session, monkeypatch):
    import document_synthesis

    service, planner, answer, identifier = session
    expected = AnswerResult(
        "Swimming pool [1]\nFitness room [1]",
        [],
        {"all_extracted_text_processed": True, "exhaustive": False},
    )
    synthesize = Mock(return_value=expected)
    monkeypatch.setattr(document_synthesis, "synthesize", synthesize)
    planner.return_value = plan_message(intent="enumerate", search_question="List all amenities")

    result = service.execute("List all amenities in this document.")

    assert "Swimming pool [1]" in result
    assert "Fitness room [1]" in result
    synthesize.assert_called_once_with(
        service.search.library, service.model, "enumerate", "List all amenities", [identifier],
    )
    service.search.embedding.embed_query.assert_not_called()
    answer.assert_not_called()


def test_enumeration_model_is_independent_from_answer_model(session, monkeypatch):
    import document_synthesis

    service, planner, answer, identifier = session
    enumeration_model = RunnableLambda(lambda _: AIMessage(content="unused"))
    service.enumeration_model = enumeration_model
    synthesize = Mock(return_value=AnswerResult("Pool [1]", [], {}))
    monkeypatch.setattr(document_synthesis, "synthesize", synthesize)
    planner.return_value = plan_message(intent="enumerate")
    service.execute("List all amenities")
    assert synthesize.call_args.args[1] is enumeration_model
    assert service.model is not enumeration_model
    answer.assert_not_called()


@pytest.mark.parametrize("question", [
    "list every amenity in the lease", "List all amenities in this document",
    "Enumerate each renewal condition",
])
def test_explicit_item_inventory_overrides_search_misclassification(session, question):
    service, planner, _, _ = session
    planner.return_value = plan_message(intent="search", search_question="amenities")
    plan = service.plan(question)
    assert plan.intent == "enumerate"
    assert plan.search_question == question


@pytest.mark.parametrize("intent,question", [
    ("search", "List all matching passages about amenities"),
    ("aggregate", "List all amenities and calculate their total fees"),
    ("advice", "List every legal action I should take"),
    ("clarify", "List every undefined category"),
])
def test_enumeration_guard_preserves_other_capabilities(session, intent, question):
    service, planner, _, _ = session
    planner.return_value = plan_message(
        intent=intent, clarification="Which category?" if intent == "clarify" else None,
    )
    assert service.plan(question).intent == intent


def test_summary_routes_without_retrieval_and_can_be_reformatted(session):
    service, planner, answer, identifier = session
    planner.return_value = plan_message(intent="summarize", search_question="Summarize this lease")
    answer.return_value = AIMessage(content=json.dumps({
        "status": "findings",
        "findings": [{"text": "Monthly rent is $1200", "citations": [1]}],
    }))
    result = service.execute("Summarize this lease")
    assert "$1200 [1]" in result
    assert "all_extracted_text_processed" in result
    assert answer.call_count == 1
    service.search.embedding.embed_query.assert_not_called()
    planner.return_value = plan_message(intent="reformat", search_question="", presentation="csv")
    assert service.execute("Put that in CSV").startswith("Type,Content")
    assert answer.call_count == 1


def test_topic_summary_planner_instructions_distinguish_topics_from_explicit_passages(session):
    service, planner, answer, _ = session
    question = "Summarize the rent, parking, deposit, utilities, term dates and conflicting storage fees."
    planner.return_value = plan_message(intent="summarize", search_question=question)
    answer.return_value = AIMessage(content=json.dumps({
        "status": "findings",
        "findings": [{"text": "Monthly rent is $1200", "citations": [1]}],
    }))
    service.execute(question)
    instructions = planner.call_args.args[0].to_messages()[0].text
    assert "Topic selection does not mean passage selection" in instructions
    assert "only when the user explicitly limits it to a particular passage or clause" in instructions
    assert "Focused passage summaries may be explain" not in instructions
    assert service.previous_plan is not None and service.previous_plan.intent == "summarize"
    service.search.embedding.embed_query.assert_not_called()
    answer.assert_called_once()


@pytest.mark.parametrize("bound", [False, True])
def test_planner_request_schema_restricts_ids_without_mutating_client(session, monkeypatch, bound):
    service, _, _, identifier = session
    planner = query.get_planner()
    assert isinstance(planner, ChatOllama)
    chat = Mock(return_value=iter([{
        "model": planner.model,
        "message": {"role": "assistant", "content": plan_message(
            intent="summarize", search_question="Summarize terms",
            document_ids=[identifier],
        ).text},
        "done": True, "done_reason": "stop",
    }]))
    monkeypatch.setattr(planner._client, "chat", chat)
    service.planner = planner.bind(temperature=0.0) if bound else planner
    plan = service.plan("Summarize terms")
    assert plan.document_ids == [identifier]
    selection = chat.call_args.kwargs["format"]["properties"]["document_ids"]["anyOf"][0]
    assert selection["items"]["enum"] == [identifier]
    assert selection["minItems"] == 1
    assert planner.format == query.PLAN_SCHEMA
    assert "enum" not in query.PLAN_SCHEMA["properties"]["document_ids"]["anyOf"][0]["items"]


def test_comparison_needs_two_selected_documents(session):
    service, planner, answer, identifier = session
    planner.return_value = plan_message(intent="compare")
    with pytest.raises(DocumentQueryError, match="2 to 4"):
        service.execute("Compare these leases")
    answer.assert_not_called()


def test_summary_failure_keeps_previous_successful_context(session):
    service, planner, answer, identifier = session
    service.execute("What is the rent?")
    previous = service.previous_result
    planner.return_value = plan_message(intent="summarize", search_question="Summarize lease")
    answer.return_value = AIMessage(content="Uncited summary")
    with pytest.raises(DocumentQueryError, match="invalid_json"):
        service.execute("Summarize entire lease")
    assert service.previous_result is previous
    assert service.previous_plan.intent == "extract_fact"


def test_motivating_ranking_guard_overrides_bad_planner_classification(session):
    service, planner, answer, identifier = session
    planner.return_value = plan_message(intent="explain", presentation="ascii_table")
    result = service.execute("generate an ascii table that summarizes the component with the most warnings")
    assert "aggregate" in result
    assert "No query was executed" in result
    answer.assert_not_called()
    service.search.embedding.embed_query.assert_not_called()


def test_unsupported_filters_cannot_silently_search_whole_library(session):
    service, planner, answer, identifier = session
    planner.return_value = plan_message(filters=["tax year 2025"])
    assert "filters" in service.execute("Find records for tax year 2025")
    service.search.embedding.embed_query.assert_not_called()
    answer.assert_not_called()


def test_ambiguous_reference_requests_clarification(session):
    service, planner, answer, identifier = session
    planner.return_value = plan_message(
        intent="clarify", search_question="", clarification="Which lease do you mean?",
    )
    assert service.execute("What about that lease?") == "Clarification needed: Which lease do you mean?"
    answer.assert_not_called()


def test_format_follow_up_reuses_evidence_without_retrieval_or_generation(session):
    service, planner, answer, identifier = session
    service.execute("What is the rent?")
    planner.return_value = plan_message(intent="reformat", search_question="", presentation="csv")
    result = service.execute("Put that in CSV")
    assert result.startswith("Type,Content")
    assert "Source," in result
    assert answer.call_count == 1
    assert service.search.embedding.embed_query.call_count == 1
    prompt = planner.call_args.args[0].to_messages()[-1].content
    assert '"previous_plan"' in prompt
    assert "Monthly rent is $1200" not in prompt
    assert service.previous_plan.search_question == "What is the monthly rent?"
    assert service.previous_plan.presentation == "csv"


def test_format_only_before_answer_requires_clarification(session):
    service, planner, answer, identifier = session
    planner.return_value = plan_message(intent="reformat", search_question="")
    assert "ask a focused question" in service.execute("Make a table")
    answer.assert_not_called()


def test_follow_up_resolution_and_standalone_question_do_not_blindly_reuse_subject(session):
    service, planner, answer, identifier = session
    service.execute("What is rent?")
    planner.return_value = plan_message(search_question="Does rent include utilities?")
    service.execute("Does it include utilities?")
    assert service.search.embedding.embed_query.call_args.args[0] == "Does rent include utilities?"
    planner.return_value = plan_message(search_question="What is the pet policy?")
    service.execute("What is the pet policy?")
    assert service.search.embedding.embed_query.call_args.args[0] == "What is the pet policy?"


def test_scope_cannot_be_broadened_by_planner(session, tmp_path):
    service, planner, answer, identifier = session
    other = tmp_path / "policy.txt"
    other.write_text("Insurance $500")
    other_id = service.search.library.ingest(other)
    planner.return_value = plan_message(document_ids=[other_id])
    with pytest.raises(DocumentQueryError, match="broaden"):
        service.execute("Search that other document")
    assert service.document_ids == [identifier]
    answer.assert_not_called()


def test_scope_selection_persists_and_reset_clears_context(session, tmp_path):
    service, planner, answer, identifier = session
    other = tmp_path / "policy.txt"
    other.write_text("Insurance $500")
    other_id = service.search.library.ingest(other)
    service.search.index()
    service.select(None)
    planner.return_value = plan_message(document_ids=[identifier])
    service.execute("What is rent in lease.txt?")
    assert service.document_ids == [identifier]
    service.select([other_id])
    assert service.previous_result is None
    assert service.previous_plan is None
    assert service.document_ids == [other_id]
    with pytest.raises(DocumentQueryError):
        service.select(["unknown"])
    assert service.document_ids == [other_id]


def test_reformat_checks_current_versions(session):
    service, planner, answer, identifier = session
    service.execute("Rent?")
    path = Path(service.search.library.snapshots([identifier])[0].source_path)
    path.write_text("Rent changed to $2000.")
    service.search.library.ingest(path)
    planner.return_value = plan_message(intent="reformat", search_question="")
    with pytest.raises(DocumentQueryError, match="versions changed"):
        service.execute("Make that a table")


def test_reformat_cannot_hide_a_new_failed_ingestion(session):
    service, planner, answer, identifier = session
    service.execute("Rent?")
    path = Path(service.search.library.snapshots([identifier])[0].source_path)
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError):
        service.search.library.ingest(path)
    planner.return_value = plan_message(intent="reformat", search_question="")
    with pytest.raises(DocumentQueryError, match="changed"):
        service.execute("Make that a table")


def test_failure_preserves_session_state(session):
    service, planner, answer, identifier = session
    service.execute("Rent?")
    previous = service.previous_result
    answer.side_effect = httpx.ConnectError("SECRET-BACKEND")
    with pytest.raises(httpx.ConnectError):
        service.execute("Another focused question")
    assert service.previous_result is previous


def test_search_intent_returns_passages_without_answer_generation(session):
    service, planner, answer, identifier = session
    planner.return_value = plan_message(intent="search")
    result = service.execute("Find rent clauses")
    assert "Rent is $1200" in result
    assert "Evidence sources" in result
    answer.assert_not_called()


@pytest.mark.parametrize("response", [
    AIMessage(content="{}", response_metadata={"done_reason": "length"}), "not a message",
])
def test_truncated_or_wrong_type_planner_response(session, response):
    service, planner, answer, identifier = session
    planner.return_value = response
    with pytest.raises(DocumentQueryError):
        service.execute("Rent?")
    answer.assert_not_called()


def test_large_scope_prompt_fails_before_model(session, tmp_path):
    service, planner, answer, identifier = session
    for index in range(50):
        path = tmp_path / f"record-{index}.txt"
        path.write_text("Synthetic")
        service.search.library.ingest(path)
    service.select(None)
    with pytest.raises(DocumentQueryError, match="scope is too large"):
        service.execute("Which document?")
    planner.assert_not_called()


def test_render_formats_preserve_citations_and_ascii_escape():
    result = AnswerResult("Tax €500 [1]", [], {"exhaustive": False})
    for presentation in ("list", "ascii_table", "csv"):
        rendered = query.render_answer(result, presentation)
        assert "[1]" in rendered
        assert "exhaustive" in rendered
    assert query.render_answer(result, "ascii_table").isascii()
    assert "'=SUM" in query.format_rows([("Answer", "=SUM(A1) [1]")], "csv")


def test_cli_chat_scope_and_error_recovery(session, monkeypatch, capsys):
    service, planner, answer, identifier = session
    monkeypatch.setattr("builtins.input", Mock(side_effect=[
        f"/select {identifier}", "/scope", "", "Rent?", "/reset", "/all", "exit",
    ]))
    query.run_chat(service)
    output = capsys.readouterr()
    assert "Question must contain" in output.err
    assert "Document selection updated" in output.out
    assert "$1200 [1]" in output.out
    assert output.out.endswith("Goodbye.\n")
    assert service.previous_plan is None


def test_chat_service_failure_keeps_session_alive(session, monkeypatch, capsys):
    service, planner, answer, identifier = session
    planner.side_effect = [httpx.ConnectError("SECRET-ERROR"), plan_message()]
    monkeypatch.setattr("builtins.input", Mock(side_effect=["Rent?", "Rent?", "exit"]))
    query.run_chat(service)
    output = capsys.readouterr()
    assert "Local Ollama request failed" in output.err
    assert "SECRET-" not in output.err
    assert "$1200" in output.out


@pytest.mark.parametrize("ending", [EOFError(), KeyboardInterrupt()])
def test_chat_interruptions_are_clean(session, monkeypatch, capsys, ending):
    service, planner, answer, identifier = session
    monkeypatch.setattr("builtins.input", Mock(side_effect=ending))
    query.run_chat(service)
    assert capsys.readouterr().out.endswith("Goodbye.\n")


def test_logs_do_not_record_plan_question_or_output(session, tmp_path):
    service, planner, answer, identifier = session
    config = tmp_path / "logging.json"
    config.write_text(json.dumps({
        "level": "DEBUG", "file": "test.log", "max_bytes": 100000, "retained_files": 5,
    }))
    previous, level = LOGGER.handlers[:], LOGGER.level
    configure_logging(config)
    try:
        planner.return_value = plan_message(search_question="SECRET-SEARCH")
        answer.return_value = AIMessage(content=json.dumps({
            "status": "findings",
            "findings": [{"support": [1],
                          "text": "SECRET-ANSWER"}],
        }))
        service.execute("SECRET-QUESTION")
        text = (tmp_path / "test.log").read_text()
        assert "Query planned intent=extract_fact" in text
        assert "SECRET-" not in text
        assert identifier not in text
        assert "lease.txt" not in text
    finally:
        for handler in LOGGER.handlers:
            handler.close()
        LOGGER.handlers = previous
        LOGGER.setLevel(level)


def test_factory_sends_structured_non_reasoning_plan(monkeypatch):
    planner = query.get_planner()
    assert isinstance(planner, ChatOllama)
    chat = Mock(return_value={
        "model": "qwen3.5:4b", "message": {"role": "assistant", "content": plan_message().text},
        "done": True, "done_reason": "stop",
    })
    monkeypatch.setattr(planner._client, "chat", chat)
    response = planner.invoke("Plan a query", stream=False)
    assert query.parse_plan(response.text, set()).intent == "extract_fact"
    assert chat.call_args.kwargs["format"] == query.PLAN_SCHEMA
    assert chat.call_args.kwargs["think"] is False
    assert chat.call_args.kwargs["options"]["num_predict"] == 768


@pytest.mark.skipif(os.environ.get("RUN_OLLAMA_TESTS") != "1",
                    reason="Enable local structured planning integration")
def test_live_local_plan_routes_ranking_and_focused_question(tmp_path):
    library = DocumentLibrary(tmp_path / "catalogue.sqlite3")
    source = tmp_path / "lease.txt"
    source.write_text("Monthly rent $1200. Pets require written permission.")
    identifier = library.ingest(source)
    from chatbot_tech_doc import get_embeddings, get_llm
    engine = DocumentSearch(library, get_embeddings(), "nomic-embed-text")
    engine.index()
    service = query.QuerySession(engine, query.get_planner(), get_llm("qwen3.5:4b"), [identifier])
    plan = service.plan("What is the monthly rent? Give the answer as an ASCII table.")
    assert plan.intent in ("extract_fact", "explain")
    assert plan.presentation == "ascii_table"
    assert "ascii" not in plan.search_question.casefold()
    answer = service.execute("What is the monthly rent?", "ascii_table")
    assert "1200" in answer.replace(",", "")
    assert answer.isascii()
    reformatted = service.execute("Put that answer in CSV.")
    assert reformatted.startswith("Type,Content")
    assert "[1]" in reformatted
    assert service.previous_plan.search_question
    assert service.plan("Summarize the entire lease").intent == "summarize"
    assert "Not supported yet" in service.execute(
        "generate an ascii table that summarizes the component with the most warnings"
    )
    config = tmp_path / "logging.json"
    config.write_text(json.dumps({
        "level": "INFO", "file": "test.log", "max_bytes": 100000, "retained_files": 5,
    }))
    result = subprocess.run([
        sys.executable, str(Path(query.__file__).parent / "document_library.py"),
        "--database", str(library.database), "--log-config", str(config),
        "plan", "What is the monthly rent?", "--document", identifier,
    ], capture_output=True, text=True, check=True, timeout=180)
    assert json.loads(result.stdout)["intent"] in ("extract_fact", "explain")
