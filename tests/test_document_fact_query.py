import json
import logging
import os
import shlex
import subprocess
from copy import deepcopy
from unittest.mock import Mock

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

from document_fact_query import FACT_PLAN_SCHEMA, propose_fact_query
from document_facts import list_facts, review_fact, save_fact
from document_library import DocumentLibrary, DocumentQueryError
from document_query import QuerySession, run_chat
from document_search import DocumentSearch


def message(**changes):
    value = {
        "operation": "sum", "fact_ids": [], "lower": None, "upper": None,
        "clarification": None,
    }
    value.update(changes)
    return AIMessage(content=json.dumps(value))


def route(intent="aggregate", **changes):
    value = {
        "intent": intent, "search_question": "Total confirmed monthly charges",
        "document_ids": None, "presentation": "prose", "clarification": None,
        "filters": [],
    }
    value.update(changes)
    return AIMessage(content=json.dumps(value))


@pytest.fixture
def context(tmp_path):
    library = DocumentLibrary(tmp_path / "catalogue.sqlite3")
    path = tmp_path / "lease.txt"
    path.write_text("Rent $10 monthly. Parking $20 monthly. Deposit $30 once.")
    document = library.ingest(path)
    version = library.snapshots([document])[0].version_id
    facts = []
    for name, quote, value, period, decision in (
        ("Rent", "Rent $10 monthly.", "10", "monthly", "confirmed"),
        ("Parking", "Parking $20 monthly.", "20", "monthly", "confirmed"),
        ("Deposit", "Deposit $30 once.", "30", "once", "unreviewed"),
    ):
        start = path.read_text().index(quote)
        record = save_fact(library, {
            "document_id": document, "version_id": version, "name": name, "kind": "money",
            "value": value, "currency": "USD", "period": period, "unit": "",
            "conditions": "", "uncertainty": "", "candidate_id": "",
            "support": [{
                "segment": 1, "char_start": start, "char_end": start + len(quote),
                "quote": quote,
            }],
        })
        if decision == "confirmed":
            record = review_fact(library, record["fact_id"], 1, decision)
        facts.append(record)
    planner = Mock()
    model = Mock(side_effect=AssertionError("Fact query must not call answer model"))
    search = DocumentSearch(library, Mock(spec=Embeddings), "unused")
    session = QuerySession(search, RunnableLambda(planner), RunnableLambda(model), [document])
    return session, planner, model, facts, path


def start(context, **changes):
    session, planner, _, facts, _ = context
    planner.side_effect = [
        route(), message(fact_ids=[facts[0]["fact_id"], facts[1]["fact_id"]], **changes),
    ]
    return session.execute("What is the total of the confirmed monthly charges?")


def test_query_proposes_only_then_requires_exact_explicit_approval(context, monkeypatch):
    import document_query

    session, planner, model, _, _ = context
    execute = Mock(wraps=document_query.query_facts)
    monkeypatch.setattr(document_query, "query_facts", execute)
    output = start(context)
    execute.assert_not_called()
    assert "no calculation performed" in output
    assert "/facts-run " in output and "/facts-select " in output
    assert "--expected-preview" in output and "--database" in output
    assert "not human-confirmed" in output and "Deposit $30 once." in output
    assert "Rent $10 monthly." in output and "Parking $20 monthly." in output
    pending = session.pending_fact_query
    assert pending is not None and pending["preview"]["ready"]
    token = pending["preview"]["preview_token"]
    result = session.execute("/facts-run " + token)
    assert '"value": "30"' in result
    assert "not a document-wide total" in result
    execute.assert_called_once()
    assert planner.call_count == 2
    model.assert_not_called()
    session.search.embedding.embed_query.assert_not_called()
    assert session.pending_fact_query is None
    assert session.previous_result is None
    with pytest.raises(DocumentQueryError, match="No pending"):
        session.execute("/facts-run " + token)


def test_change_selection_repreviews_and_old_token_cannot_run(context):
    session, _, _, facts, _ = context
    start(context)
    original = session.pending_fact_query["preview"]["preview_token"]
    catalogue = session.pending_fact_query["catalogue"]
    number = next(i for i, fact in enumerate(catalogue, 1) if fact["fact_id"] == facts[1]["fact_id"])
    updated = session.execute(f"/facts-select {number}")
    assert "Proposed selection only" in updated
    preview = session.pending_fact_query["preview"]
    assert preview["selected_fact_ids"] == [facts[1]["fact_id"]]
    assert preview["preview_token"] != original
    with pytest.raises(DocumentQueryError, match="current ready preview"):
        session.execute("/facts-run " + original)
    result = session.execute("/facts-run " + preview["preview_token"])
    assert '"value": "20"' in result


def test_user_selection_of_ineligible_record_blocks_no_silent_exclusion(context):
    session, _, _, facts, _ = context
    start(context)
    number = next(i for i, fact in enumerate(session.pending_fact_query["catalogue"], 1)
                  if fact["fact_id"] == facts[2]["fact_id"])
    output = session.execute(f"/facts-select {number}")
    assert "Blocked" in output and "not human-confirmed" in output
    assert "/facts-run " not in output
    with pytest.raises(DocumentQueryError, match="current ready preview"):
        session.execute("/facts-run " + session.pending_fact_query["preview"]["preview_token"])


@pytest.mark.parametrize("command", ["/facts-select 0", "/facts-select 999", "/facts-select x",
                                    "/facts-select 1,1", "/facts-run wrong", "/facts-unknown"])
def test_bad_commands_cannot_execute_or_replace_selection(context, command):
    session = context[0]
    start(context)
    before = deepcopy(session.pending_fact_query)
    with pytest.raises(DocumentQueryError):
        session.execute(command)
    assert session.pending_fact_query == before


def test_cancel_new_question_and_scope_change_clear_pending(context):
    session, planner, _, _, _ = context
    start(context)
    session.execute("/facts-cancel")
    assert session.pending_fact_query is None
    start(context)
    planner.side_effect = [route("clarify", clarification="Which category?")]
    assert "Clarification needed" in session.execute("Different question")
    assert session.pending_fact_query is None
    start(context)
    session.select(session.document_ids)
    assert session.pending_fact_query is None


@pytest.mark.parametrize("changed", ["selected_review", "omitted_review", "new_fact", "source"])
def test_catalogue_changes_invalidate_entire_proposal_even_omitted_records(context, changed):
    session, _, _, facts, path = context
    start(context)
    token = session.pending_fact_query["preview"]["preview_token"]
    library = session.search.library
    if changed in ("selected_review", "omitted_review"):
        index = 0 if changed == "selected_review" else 2
        review_fact(library, facts[index]["fact_id"], 1, "confirmed")
    elif changed == "new_fact":
        save_fact(library, {**facts[0]["payload"], "name": "Newly added charge"})
    else:
        path.write_text("Different contract.")
        library.ingest(path)
    with pytest.raises(DocumentQueryError, match="catalogue changed"):
        session.execute("/facts-run " + token)
    assert session.pending_fact_query is None


def test_ambiguous_proposal_requests_clarification_without_pending_execution(context):
    session, planner, model, _, _ = context
    planner.side_effect = [
        route(), message(operation=None, clarification="Which charges should be included?"),
    ]
    output = session.execute("What is the total?")
    assert output == "Clarification needed: Which charges should be included?"
    assert session.pending_fact_query is None
    model.assert_not_called()


@pytest.mark.parametrize("change", [
    {"operation": "annualize"}, {"operation": None},
    {"fact_ids": ["unknown"]}, {"fact_ids": []}, {"fact_ids": [{}]},
    {"lower": "2026"}, {"upper": 42}, {"extra": True},
    {"clarification": ""}, {"clarification": []},
    {"clarification": "Which fees?", "operation": "sum"},
])
def test_malformed_proposals_fail_closed(context, change):
    session, _, _, facts, _ = context
    value = {
        "operation": "sum", "fact_ids": [facts[0]["fact_id"]],
        "lower": None, "upper": None, "clarification": None,
    }
    value.update(change)
    planner = RunnableLambda(lambda _: AIMessage(content=json.dumps(value)))
    with pytest.raises(DocumentQueryError):
        propose_fact_query("Total charges", list_facts(session.search.library, session.document_ids), planner)


def test_duplicate_and_ineligible_planner_ids_rejected(context):
    session, _, _, facts, _ = context
    catalogue = list_facts(session.search.library, session.document_ids)
    for identifiers in ([facts[0]["fact_id"]] * 2, [facts[2]["fact_id"]]):
        with pytest.raises(DocumentQueryError, match="ineligible or duplicate"):
            propose_fact_query(
                "Total charges", catalogue,
                RunnableLambda(lambda _: message(fact_ids=identifiers)),
            )


def test_planning_budget_does_not_truncate_or_sample_catalogue(context):
    records = list_facts(context[0].search.library, context[0].document_ids)
    planner = Mock()
    with pytest.raises(DocumentQueryError, match="budget"):
        propose_fact_query("Sum", records * 50, RunnableLambda(planner))
    oversized = deepcopy(records)
    oversized[0]["payload"]["conditions"] = "x" * 6000
    with pytest.raises(DocumentQueryError, match="budget"):
        propose_fact_query("Sum", oversized, RunnableLambda(planner))
    planner.assert_not_called()


@pytest.mark.parametrize("response", [
    AIMessage(content="not JSON"), AIMessage(content="[]"),
    AIMessage(content="{}", response_metadata={"done_reason": "length"}), "wrong type",
])
def test_invalid_model_responses_are_explicit(context, response):
    with pytest.raises(DocumentQueryError):
        propose_fact_query(
            "Sum", list_facts(context[0].search.library, context[0].document_ids),
            RunnableLambda(lambda _: response),
        )


def test_no_explicit_scope_never_implicitly_selects_all_documents(context):
    session, planner, _, _, _ = context
    session.select(None)
    planner.side_effect = [route()]
    output = session.execute("Sum the fees")
    assert "select documents explicitly" in output
    assert session.pending_fact_query is None
    assert planner.call_count == 1


def test_incompatible_values_are_previewed_as_blocked_not_reduced_to_subset(context):
    session, planner, _, facts, _ = context
    review_fact(session.search.library, facts[2]["fact_id"], 1, "confirmed")
    planner.side_effect = [route(), message(fact_ids=[fact["fact_id"] for fact in facts])]
    output = session.execute("Sum every charge")
    assert "Blocked" in output and "periods must match" in output
    assert len(session.pending_fact_query["preview"]["selected_fact_ids"]) == 3
    assert "/facts-run " not in output


def test_comparison_approval_preserves_requested_first_second_order(context):
    session, planner, _, facts, _ = context
    identifiers = [facts[1]["fact_id"], facts[0]["fact_id"]]
    planner.side_effect = [route("fact_query"), message(operation="compare", fact_ids=identifiers)]
    session.execute("Compare the confirmed parking charge with rent, in that order")
    preview = session.pending_fact_query["preview"]
    assert preview["selected_fact_ids"] == identifiers
    assert "relation" not in preview
    output = session.execute("/facts-run " + preview["preview_token"])
    assert '"relation": "greater"' in output


@pytest.mark.parametrize("operation", ["count", "sort"])
@pytest.mark.parametrize("mode", [
    "all_dates", "prefilter_dates", pytest.param("local", marks=pytest.mark.skipif(
        os.environ.get("RUN_OLLAMA_TESTS") != "1", reason="Enable local date proposal integration",
    )),
])
def test_natural_language_inclusive_date_filters_use_shared_preview(context, operation, tmp_path, mode):
    session, planner, _, _, _ = context
    path = tmp_path / "dates.txt"
    path.write_text("2026-01-01 and 2027-01-01")
    library = session.search.library
    document = library.ingest(path)
    snapshot = library.snapshots([document])[0]
    records = []
    for value in ("2026-01-01", "2027-01-01"):
        start = path.read_text().index(value)
        record = save_fact(library, {
            "document_id": document, "version_id": snapshot.version_id,
            "name": "Recorded date " + value, "kind": "date", "value": value,
            "unit": "", "currency": "", "period": "", "conditions": "",
            "uncertainty": "", "candidate_id": "",
            "support": [{"segment": 1, "char_start": start, "char_end": start + 10, "quote": value}],
        })
        records.append(review_fact(library, record["fact_id"], 1, "confirmed"))
    session.select([document])
    if mode == "local":
        from document_query import get_planner
        session.planner = get_planner("qwen3.5:4b")
    else:
        identifiers = [record["fact_id"] for record in records]
        if mode == "prefilter_dates":
            identifiers = identifiers[:1]
        planner.side_effect = [
            route("fact_query", filters=["Dates during 2026"]),
            message(operation=operation, fact_ids=identifiers,
                    lower="2026-01-01", upper="2026-12-31"),
        ]
    output = session.execute(
        f"{operation} the confirmed date records from 2026-01-01 through 2026-12-31, ascending.")
    preview = session.pending_fact_query["preview"]
    assert preview["ready"] and not preview["calculation_performed"]
    assert preview["date_bounds"] == {"lower": "2026-01-01", "upper": "2026-12-31"}
    assert preview["included_fact_ids"] == [records[0]["fact_id"]]
    if mode == "all_dates":
        assert preview["exclusions"][0]["fact_id"] == records[1]["fact_id"]
    elif mode == "prefilter_dates":
        assert records[1]["fact_id"] not in preview["selected_fact_ids"]
    assert records[1]["fact_id"] in output and "2027-01-01" in output
    assert "Outside inclusive date bounds" in output
    result = session.execute("/facts-run " + preview["preview_token"])
    if operation == "count":
        assert '"value": 1' in result
    else:
        assert '"operation": "sort"' in result


def test_printed_one_shot_command_runs_in_separate_process_against_same_database(context):
    session = context[0]
    output = start(context)
    command = next(line.removeprefix("- CLI: ") for line in output.splitlines() if line.startswith("- CLI: "))
    result = subprocess.run(shlex.split(command), text=True, capture_output=True, check=True)
    value = json.loads(result.stdout)
    assert value["value"] == "30"
    assert value["selected_fact_ids"] == session.pending_fact_query["preview"]["selected_fact_ids"]
    assert not value["document_total"]


def test_no_eligible_facts_reports_ineligibility_and_evidence(context):
    session, planner, _, facts, _ = context
    for fact in facts[:2]:
        review_fact(session.search.library, fact["fact_id"], 1, "unreviewed")
    planner.side_effect = [route("fact_query")]
    output = session.execute("Count confirmed charges")
    assert "No eligible confirmed facts" in output
    assert "Rent $10 monthly." in output and "not human-confirmed" in output
    assert session.pending_fact_query is None
    assert planner.call_count == 1


def test_fact_request_discards_old_passage_result_so_format_followup_cannot_reuse_it(context):
    from document_search import AnswerResult

    session, planner, _, _, _ = context
    session.previous_result = AnswerResult("Old unrelated answer", [], {})
    start(context)
    planner.side_effect = [route("reformat", presentation="ascii_table")]
    output = session.execute("Put that in a table")
    assert "ask a focused question" in output
    assert "Old unrelated answer" not in output
    assert session.pending_fact_query is None


def test_planner_receives_metadata_not_source_quotes_or_review_notes(context):
    session, _, _, facts, _ = context
    review_fact(session.search.library, facts[0]["fact_id"], 1, "confirmed", "PRIVATE_REVIEW_NOTE")
    captured = []

    def respond(prompt):
        captured.append(prompt.to_string())
        return message(fact_ids=[facts[0]["fact_id"]])

    propose_fact_query(
        "Sum rent", list_facts(session.search.library, session.document_ids), RunnableLambda(respond),
    )
    assert "Rent" in captured[0] and '"eligible":true' in captured[0]
    assert "PRIVATE_REVIEW_NOTE" not in captured[0] and "Rent $10 monthly." not in captured[0]


@pytest.mark.parametrize("intent", ["compare", "search", "explain", "aggregate"])
def test_explicit_confirmed_fact_operations_cannot_route_to_document_comparison(context, intent):
    session, planner, model, _, _ = context
    planner.side_effect = [route(intent)]
    assert session.plan("Compare confirmed rent and parking facts").intent == "fact_query"
    model.assert_not_called()


@pytest.mark.parametrize("intent", ["advice", "clarify"])
def test_fact_route_guard_does_not_override_advice_or_clarification(context, intent):
    session, planner, _, _, _ = context
    planner.side_effect = [route(intent, clarification="Which records?" if intent == "clarify" else None)]
    assert session.plan("Compare confirmed facts and recommend what I should pay").intent == intent


def test_question_fact_values_evidence_and_tokens_are_not_logged(context, caplog):
    session, _, _, facts, _ = context
    with caplog.at_level(logging.INFO, logger="archive_it"):
        start(context)
        token = session.pending_fact_query["preview"]["preview_token"]
        session.execute("/facts-run " + token)
    for private in (
        "What is the total of the confirmed monthly charges?", "Rent $10 monthly.",
        "Parking $20 monthly.", token, facts[0]["fact_id"],
    ):
        assert private not in caplog.text


def test_ollama_fact_schema_and_output_budget_preserve_planner_configuration(context, monkeypatch):
    import document_fact_query
    from document_query import PLAN_SCHEMA, get_planner

    session, _, _, facts, _ = context
    planner = get_planner("qwen3.5:4b")
    assert isinstance(planner, ChatOllama)
    captured = []

    def transport(name, model):
        assert name == "Typed fact query planner"
        captured.append(model)
        return RunnableLambda(lambda _: message(fact_ids=[facts[0]["fact_id"]]))

    monkeypatch.setattr(document_fact_query, "logged_runnable", transport)
    propose_fact_query("Sum rent", list_facts(session.search.library, session.document_ids), planner)
    assert captured[0].format["properties"]["fact_ids"]["items"]["enum"] == [
        record["fact_id"] for record in list_facts(session.search.library, session.document_ids)
        if record["usable_for_queries"]
    ]
    assert "enum" not in FACT_PLAN_SCHEMA["properties"]["fact_ids"]["items"]
    assert captured[0].num_predict == 4096
    assert captured[0].num_ctx == planner.num_ctx and captured[0].temperature == 0
    assert captured[0].reasoning is False
    assert planner.format == PLAN_SCHEMA and planner.num_predict == 768


def test_chat_commands_bypass_planner_and_require_approval(context, monkeypatch, capsys):
    session, planner, _, facts, _ = context
    planner.side_effect = [
        route(), message(fact_ids=[facts[0]["fact_id"], facts[1]["fact_id"]]),
    ]
    questions = iter(["What is the sum of monthly charges?", "/facts-cancel", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(questions))
    run_chat(session)
    output = capsys.readouterr().out
    assert "/facts-run " in output and "cancelled" in output
    assert planner.call_count == 2
    assert session.pending_fact_query is None


@pytest.mark.skipif(os.environ.get("RUN_OLLAMA_TESTS") != "1", reason="Enable local fact proposal integration")
@pytest.mark.parametrize(("question", "expected"), [
    ("What is the total of the confirmed monthly charges?", "sum"),
    ("How many confirmed monthly charges are recorded?", "count"),
    ("Compare the confirmed rent value to the confirmed parking value, rent first.", "compare"),
])
def test_live_local_fact_selection(context, question, expected):
    from document_query import get_planner

    session, _, _, facts, _ = context
    records = list_facts(session.search.library, session.document_ids)
    proposal = propose_fact_query(question, records, get_planner("qwen3.5:4b"))
    assert proposal.clarification is None
    assert proposal.operation == expected
    assert set(proposal.fact_ids) == {facts[0]["fact_id"], facts[1]["fact_id"]}
    if expected == "compare":
        assert proposal.fact_ids == [facts[0]["fact_id"], facts[1]["fact_id"]]


@pytest.mark.parametrize("question", [
    "Annualize the confirmed monthly charges.",
    "Which document has the most charges? Rank the documents.",
    "Sort the confirmed charge values descending.",
    "Convert the confirmed charges into EUR.",
    "Sort recorded dates newest first.",
    "Sort dates in reverse chronological order.",
    "Count records grouped by document.",
])
def test_explicit_unsupported_operations_request_clarification_without_model(context, question):
    session = context[0]
    planner = Mock()
    proposal = propose_fact_query(
        question, list_facts(session.search.library, session.document_ids), RunnableLambda(planner),
    )
    assert proposal.clarification
    assert proposal.operation is None and proposal.fact_ids == []
    planner.assert_not_called()


def test_explicit_compare_operand_order_overrides_reversed_model_order(context):
    session, _, _, facts, _ = context
    records = list_facts(session.search.library, session.document_ids)
    planner = RunnableLambda(lambda _: message(
        operation="compare", fact_ids=[facts[1]["fact_id"], facts[0]["fact_id"]],
    ))

    proposal = propose_fact_query(
        "Compare the confirmed rent value to the confirmed parking value, rent first.",
        records, planner,
    )

    assert proposal.fact_ids == [facts[0]["fact_id"], facts[1]["fact_id"]]


@pytest.mark.skipif(os.environ.get("RUN_OLLAMA_TESTS") != "1", reason="Enable local fact routing integration")
@pytest.mark.parametrize(("question", "expected"), [
    ("What is the total of the confirmed monthly charges?", "sum"),
    ("How many confirmed monthly charge records are there?", "count"),
    ("Compare the confirmed rent and parking values, rent first.", "compare"),
])
def test_live_local_question_routes_to_preview_not_passage_answer(context, question, expected):
    from document_query import get_planner

    session, _, model, facts, _ = context
    session.planner = get_planner("qwen3.5:4b")
    output = session.execute(question)
    assert "no calculation performed" in output
    assert session.pending_fact_query["preview"]["ready"]
    assert session.pending_fact_query["preview"]["operation"] == expected
    assert set(session.pending_fact_query["preview"]["selected_fact_ids"]) == {
        facts[0]["fact_id"], facts[1]["fact_id"],
    }
    model.assert_not_called()
