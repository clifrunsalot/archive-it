import builtins
import json
import sys
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from document_fact_ui import run_fact_workbench
from document_facts import fact_history, list_facts, query_facts, review_fact, save_fact
from document_library import DocumentLibrary
from document_synthesis import synthesize


@pytest.fixture
def library(tmp_path):
    source = tmp_path / "lease.txt"
    source.write_text("Rent $10 monthly. Parking $20 monthly. Rent $10 monthly.")
    catalogue = DocumentLibrary(tmp_path / "catalogue.sqlite3")
    return catalogue, catalogue.ingest(source), source


def input_script(monkeypatch, answers):
    sequence = iter(answers)
    prompts = []

    def respond(message):
        prompts.append(message)
        try:
            value = next(sequence)
        except StopIteration:
            pytest.fail("Interactive script exhausted at: " + message)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr("builtins.input", respond)
    return prompts


def money_fields(value="10", name="Rent"):
    return [name, "money", value, "USD", "monthly", "", ""]


def test_add_fact_select_duplicate_quote_preview_confirm_and_query(library, monkeypatch, capsys):
    catalogue, identifier, _ = library
    prompts = input_script(monkeypatch, [
        "add", "1", "Rent $10 monthly.", "2", "no",
        *money_fields(), "save", "confirm", "Checked source", "quit",
    ])
    run_fact_workbench(catalogue, identifier)
    records = list_facts(DocumentLibrary(catalogue.database), [identifier])
    assert len(records) == 1
    record = records[0]
    assert record["decision"] == "confirmed"
    assert record["payload"]["support"][0]["char_start"] == 39
    assert record["payload"]["support"][0]["quote"] == "Rent $10 monthly."
    assert record["payload"]["value"] == "10"
    assert query_facts(catalogue, [record["fact_id"]], "sum")["value"] == "10"
    output = capsys.readouterr().out
    assert "Draft preview" in output and "Source context:" in output
    assert output.index("Draft preview") < output.index("Saved unreviewed fact")
    assert output.index("Saved unreviewed fact") < output.index("is now confirmed")
    assert any("Quote occurrence" in prompt for prompt in prompts)


@pytest.mark.parametrize("ending", [
    ["cancel", "quit"], ["/cancel", "quit"], [EOFError()], [KeyboardInterrupt()],
])
def test_cancel_or_interrupt_before_save_publishes_nothing(library, monkeypatch, ending):
    catalogue, identifier, _ = library
    input_script(monkeypatch, ["add", "1", "", "no", *money_fields(), *ending])
    run_fact_workbench(catalogue, identifier)
    assert list_facts(catalogue, [identifier]) == []


@pytest.mark.parametrize("decision", ["leave", EOFError(), KeyboardInterrupt(), "/cancel"])
def test_saved_fact_does_not_auto_confirm(library, monkeypatch, decision):
    catalogue, identifier, _ = library
    ending = [decision] if isinstance(decision, BaseException) else [decision, "quit"]
    input_script(monkeypatch, [
        "add", "1", "", "no", *money_fields(), "save", *ending,
    ])
    run_fact_workbench(catalogue, identifier)
    records = list_facts(catalogue, [identifier])
    assert len(records) == 1 and records[0]["decision"] == "unreviewed"
    assert records[0]["review_history"] == []


def test_invalid_number_quote_and_value_are_explicit_and_correctable(library, monkeypatch, capsys):
    catalogue, identifier, _ = library
    input_script(monkeypatch, [
        "add", "not-number", "99", "1", "not in source", "Parking $20 monthly.", "no",
        *money_fields(value="NaN", name="Parking"), "save", "edit",
        "", "", "20", "", "", "", "", "save", "leave", "quit",
    ])
    run_fact_workbench(catalogue, identifier)
    records = list_facts(catalogue, [identifier])
    assert len(records) == 1 and records[0]["payload"]["value"] == "20"
    captured = capsys.readouterr()
    assert "decimal strings" in captured.err
    assert "Enter a number from" in captured.out
    assert "Select an exact substring" in captured.out


def test_candidate_conversion_prefills_source_without_inheriting_inventory_decisions(
    library, monkeypatch, capsys,
):
    from document_inventory_review import review_inventory

    catalogue, identifier, source = library
    source.write_text(
        "Rent $10 monthly. Parking $20 monthly is available for an additional fee. "
        "Rent $10 monthly."
    )
    catalogue.ingest(source)
    model = RunnableLambda(lambda _: AIMessage(content=json.dumps({
        "status": "findings", "findings": [{
            "text": "Parking", "conditions": "additional fee", "assessment": "established",
            "uncertainty": "", "evidence": [{
                "label": 1,
                "quote": "Parking $20 monthly is available for an additional fee.",
            }],
        }],
    })))
    result = synthesize(catalogue, model, "enumerate", "List every amenity", [identifier])
    review_inventory(
        catalogue, "List every amenity", [identifier],
        item_id=result.items[0]["review_id"], decision="accepted",
    )
    input_script(monkeypatch, [
        "candidate", "1", "yes", "", "money", "20", "USD", "monthly",
        "", "", "save", "leave", "quit",
    ])
    run_fact_workbench(catalogue, identifier, "List every amenity")
    records = list_facts(catalogue, [identifier])
    assert records[0]["payload"]["candidate_id"] == result.items[0]["review_id"]
    assert records[0]["payload"]["name"] == "Parking"
    assert records[0]["payload"]["conditions"] == "additional fee"
    assert records[0]["payload"]["support"][0]["char_start"] == 18
    assert records[0]["decision"] == "unreviewed"
    output = capsys.readouterr().out
    assert "Candidate:" in output and "Parking $20 monthly is available" in output


def seed_fact(library):
    catalogue, identifier, _ = library
    record = save_fact(catalogue, {
        "document_id": identifier, "version_id": catalogue.snapshots([identifier])[0].version_id,
        "name": "Rent", "kind": "money", "value": "10", "currency": "USD",
        "period": "monthly", "unit": "", "conditions": "Old condition", "uncertainty": "",
        "candidate_id": "", "support": [{
            "segment": 1, "char_start": 0, "char_end": 17, "quote": "Rent $10 monthly.",
        }],
    })
    return review_fact(catalogue, record["fact_id"], 1, "confirmed")


def test_correction_creates_unreviewed_revision_and_clear_removes_condition(library, monkeypatch):
    catalogue, identifier, _ = library
    original = seed_fact(library)
    input_script(monkeypatch, [
        "edit", "1", "yes", "", "", "12", "", "", "/clear", "",
        "save", "leave", "quit",
    ])
    run_fact_workbench(catalogue, identifier)
    history = fact_history(catalogue, original["fact_id"])
    assert len(history) == 2 and history[0]["decision"] == "confirmed"
    assert history[1]["payload"]["value"] == "12"
    assert history[1]["payload"]["conditions"] == ""
    assert history[1]["decision"] == "unreviewed"


def test_stale_source_requires_fresh_evidence_for_correction(library, monkeypatch):
    catalogue, identifier, source = library
    original = seed_fact(library)
    source.write_text("Rent $30 monthly.")
    catalogue.ingest(source)
    input_script(monkeypatch, [
        "edit", "1", "1", "", "no", "", "", "30", "", "", "", "",
        "save", "confirm", "", "quit",
    ])
    run_fact_workbench(catalogue, identifier)
    history = fact_history(catalogue, original["fact_id"])
    assert history[0]["stale"]
    assert not history[1]["stale"] and history[1]["decision"] == "confirmed"
    assert history[1]["payload"]["support"][0]["quote"] == "Rent $30 monthly."


def test_missing_candidates_and_facts_are_explicit(library, monkeypatch, capsys):
    catalogue, identifier, _ = library
    input_script(monkeypatch, ["candidate", "edit", "quit"])
    run_fact_workbench(catalogue, identifier)
    captured = capsys.readouterr()
    assert "requires --question" in captured.err
    assert "No facts yet" in captured.out


def test_multiple_support_spans_are_retained(library, monkeypatch):
    catalogue, identifier, _ = library
    input_script(monkeypatch, [
        "add", "1", "Rent $10 monthly.", "1", "yes",
        "1", "Parking $20 monthly.", "no",
        *money_fields(), "save", "leave", "quit",
    ])
    run_fact_workbench(catalogue, identifier)
    record = list_facts(catalogue, [identifier])[0]
    assert len(record["payload"]["support"]) == 2


def test_reject_after_save_is_explicit(library, monkeypatch):
    catalogue, identifier, _ = library
    input_script(monkeypatch, [
        "add", "1", "", "no", *money_fields(), "save", "reject", "Wrong interpretation", "quit",
    ])
    run_fact_workbench(catalogue, identifier)
    record = list_facts(catalogue, [identifier])[0]
    assert record["decision"] == "rejected"
    assert record["review_history"][0]["note"] == "Wrong interpretation"


def test_cli_registration_dispatch_and_no_model_calls(library, monkeypatch):
    import chatbot_tech_doc
    import document_library

    catalogue, identifier, _ = library
    model = Mock(side_effect=AssertionError("Workbench cannot invoke a model"))
    monkeypatch.setattr(chatbot_tech_doc, "get_llm", model)
    monkeypatch.setattr(document_library, "configure_logging", lambda _: None)
    monkeypatch.setattr(sys, "argv", [
        "document_library.py", "--database", str(catalogue.database),
        "fact-workbench", "--document", identifier,
    ])
    input_script(monkeypatch, ["quit"])
    assert document_library.main() == 0
    model.assert_not_called()


def test_changed_source_during_preview_is_not_saved(library, monkeypatch, capsys):
    catalogue, identifier, source = library
    input_script(monkeypatch, [
        "add", "1", "", "no", *money_fields(), "save", "quit",
    ])
    scripted = builtins.input

    def change_before_save(message):
        value = scripted(message)
        if message.startswith("Next action"):
            source.write_text("Rent $99 monthly.")
            catalogue.ingest(source)
        return value

    monkeypatch.setattr("builtins.input", change_before_save)
    run_fact_workbench(catalogue, identifier)
    assert list_facts(catalogue, [identifier]) == []
    assert "Source changed during review" in capsys.readouterr().err


def test_changed_fact_revision_during_edit_is_not_overwritten(library, monkeypatch, capsys):
    catalogue, identifier, _ = library
    original = seed_fact(library)
    input_script(monkeypatch, [
        "edit", "1", "yes", "", "", "12", "", "", "", "",
        "save", "cancel", "quit",
    ])
    scripted = builtins.input

    def change_before_save(message):
        value = scripted(message)
        if message.startswith("Next action"):
            corrected = {**original["payload"], "value": "99"}
            save_fact(
                catalogue, corrected, fact_id=original["fact_id"], expected_revision=1,
            )
        return value

    monkeypatch.setattr("builtins.input", change_before_save)
    run_fact_workbench(catalogue, identifier)
    history = fact_history(catalogue, original["fact_id"])
    assert len(history) == 2
    assert history[1]["payload"]["value"] == "99"
    assert "Fact revision changed" in capsys.readouterr().err


def test_changed_source_after_save_blocks_confirmation(library, monkeypatch, capsys):
    catalogue, identifier, source = library
    input_script(monkeypatch, [
        "add", "1", "", "no", *money_fields(), "save", "confirm", "", "quit",
    ])
    scripted = builtins.input

    def change_before_confirmation(message):
        value = scripted(message)
        if message.startswith("Review decision"):
            source.write_text("Rent $99 monthly.")
            catalogue.ingest(source)
        return value

    monkeypatch.setattr("builtins.input", change_before_confirmation)
    run_fact_workbench(catalogue, identifier)
    record = list_facts(catalogue, [identifier])[0]
    assert record["decision"] == "unreviewed" and record["stale"]
    assert "Fact source changed" in capsys.readouterr().err


def test_workbench_privacy_does_not_log_values_notes_or_quotes(library, monkeypatch, caplog):
    import logging

    catalogue, identifier, _ = library
    caplog.set_level(logging.DEBUG)
    input_script(monkeypatch, [
        "add", "1", "", "no", *money_fields(name="PRIVATE FACT"), "save",
        "confirm", "PRIVATE NOTE", "quit",
    ])
    run_fact_workbench(catalogue, identifier)
    assert "PRIVATE FACT" not in caplog.text
    assert "PRIVATE NOTE" not in caplog.text
    assert "Rent $10 monthly." not in caplog.text


def test_cli_script_persists_reviewed_fact_across_processes(library):
    import subprocess
    from pathlib import Path

    catalogue, identifier, _ = library
    command = [
        sys.executable, "document_library.py", "--database", str(catalogue.database),
        "fact-workbench", "--document", identifier,
    ]
    answers = ["add", "1", "Parking $20 monthly.", "no", *money_fields("20", "Parking"),
               "save", "confirm", "", "calculate", "1", "sum", "run", "quit"]
    completed = subprocess.run(
        command, input="\n".join(answers) + "\n", text=True, capture_output=True,
        cwd=Path(__file__).resolve().parents[1], timeout=30, check=True,
    )
    assert "is now confirmed" in completed.stdout
    assert "Calculation preview" in completed.stdout
    assert '"value": "20"' in completed.stdout.split("Calculation result: ", 1)[1]
    records = list_facts(DocumentLibrary(catalogue.database), [identifier])
    assert len(records) == 1 and records[0]["decision"] == "confirmed"
    assert query_facts(catalogue, [records[0]["fact_id"]], "sum")["value"] == "20"


def test_existing_fact_can_be_reviewed_without_creating_a_revision(library, monkeypatch):
    catalogue, identifier, _ = library
    original = seed_fact(library)
    input_script(monkeypatch, ["review", "1", "reject", "Rechecked", "quit"])
    run_fact_workbench(catalogue, identifier)
    history = fact_history(catalogue, original["fact_id"])
    assert len(history) == 1
    assert history[0]["decision"] == "rejected"
    assert [event["decision"] for event in history[0]["review_history"]] == [
        "confirmed", "rejected",
    ]


@pytest.mark.parametrize("format", ["txt", "docx", "pdf"])
def test_representative_ingest_review_confirm_query_flow(tmp_path, monkeypatch, format):
    from docx import Document
    from test_document_library import make_pdf

    text = "Parking costs 20 monthly."
    path = tmp_path / f"representative.{format}"
    if format == "txt":
        path.write_text(text)
    elif format == "docx":
        document = Document()
        document.add_paragraph(text)
        document.save(str(path))
    else:
        path.write_bytes(make_pdf(text))
    catalogue = DocumentLibrary(tmp_path / "catalogue.sqlite3")
    identifier = catalogue.ingest(path)
    input_script(monkeypatch, [
        "add", "1", text, "no", *money_fields("20", "Parking"),
        "save", "confirm", "Checked source", "quit",
    ])
    run_fact_workbench(catalogue, identifier)
    records = list_facts(DocumentLibrary(catalogue.database), [identifier])
    assert len(records) == 1 and records[0]["decision"] == "confirmed"
    result = query_facts(catalogue, [records[0]["fact_id"]], "sum")
    assert result["value"] == "20" and result["period"] == "monthly"
    assert result["records"][0]["evidence"][0]["quote"] == text
    assert result["exhaustive"] is False


def test_workbench_calculation_requires_explicit_run_after_preview(library, monkeypatch, capsys):
    catalogue, identifier, _ = library
    record = seed_fact(library)
    input_script(monkeypatch, ["calculate", "1", "sum", "run", "quit"])
    run_fact_workbench(catalogue, identifier)
    output = capsys.readouterr().out
    assert output.index("Calculation preview") < output.index("Calculation result")
    assert '"calculation_performed": false' in output
    result = json.loads(output.split("Calculation result: ", 1)[1])
    assert result["value"] == "10"
    assert result["selected_fact_ids"] == [record["fact_id"]]
    assert not result["document_total"]


@pytest.mark.parametrize("choice", ["", "cancel", "/cancel", EOFError(), KeyboardInterrupt()])
def test_workbench_calculation_cancellation_never_executes(library, monkeypatch, capsys, choice):
    import document_fact_ui

    catalogue, identifier, _ = library
    seed_fact(library)
    runner = Mock(side_effect=AssertionError("Cancellation must not calculate"))
    monkeypatch.setattr(document_fact_ui, "query_facts", runner)
    ending = [choice] if isinstance(choice, BaseException) else [choice, "quit"]
    input_script(monkeypatch, ["calculate", "1", "sum", *ending])
    run_fact_workbench(catalogue, identifier)
    runner.assert_not_called()
    assert "Calculation result" not in capsys.readouterr().out


def test_workbench_blocked_preview_never_offers_partial_calculation(library, monkeypatch, capsys):
    import document_fact_ui

    catalogue, identifier, _ = library
    record = seed_fact(library)
    review_fact(catalogue, record["fact_id"], 1, "unreviewed")
    runner = Mock(side_effect=AssertionError("Blocked preview cannot run"))
    monkeypatch.setattr(document_fact_ui, "query_facts", runner)
    input_script(monkeypatch, ["calculate", "1,1", "bad", "1", "sum", "quit"])
    run_fact_workbench(catalogue, identifier)
    runner.assert_not_called()
    output = capsys.readouterr().out
    assert "Select distinct numbers" in output
    assert "Calculation blocked" in output
    assert "not human-confirmed" in output


def test_workbench_refuses_changed_records_after_preview(library, monkeypatch, capsys):
    catalogue, identifier, _ = library
    record = seed_fact(library)
    input_script(monkeypatch, ["calculate", "1", "sum", "run", "quit"])
    scripted = builtins.input

    def change_at_run(message):
        value = scripted(message)
        if message.startswith("Execute previewed query?"):
            review_fact(catalogue, record["fact_id"], 1, "confirmed", "New review event")
        return value

    monkeypatch.setattr("builtins.input", change_at_run)
    run_fact_workbench(catalogue, identifier)
    captured = capsys.readouterr()
    assert "changed since preview" in captured.err
    assert "Calculation result" not in captured.out


def test_workbench_calculation_empty_scope_is_explicit(library, monkeypatch, capsys):
    input_script(monkeypatch, ["calculate", "quit"])
    run_fact_workbench(library[0], library[1])
    assert "No facts yet" in capsys.readouterr().out


def test_workbench_date_bounds_show_exclusion_before_count(library, monkeypatch, capsys):
    catalogue, identifier, _ = library
    original = seed_fact(library)
    for value in ("2026-08-01", "2027-07-31"):
        payload = {
            **original["payload"], "name": value, "kind": "date", "value": value,
            "currency": "", "period": "", "conditions": "",
        }
        saved = save_fact(catalogue, payload)
        review_fact(catalogue, saved["fact_id"], 1, "confirmed")
    facts = list_facts(catalogue, [identifier])
    numbers = [str(number) for number, fact in enumerate(facts, 1)
               if fact["payload"]["kind"] == "date"]
    input_script(monkeypatch, [
        "calculate", ",".join(numbers), "count", "2027-01-01", "", "run", "quit",
    ])
    run_fact_workbench(catalogue, identifier)
    output = capsys.readouterr().out
    assert "Outside inclusive date bounds." in output
    assert "Not selected (not part of the query)" in output
    result = json.loads(output.split("Calculation result: ", 1)[1])
    assert result["value"] == 1
    assert result["records"][0]["payload"]["value"] == "2027-07-31"


def test_similar_fact_reuses_only_metadata_and_requires_new_evidence_and_confirmation(
    library, monkeypatch, capsys,
):
    catalogue, identifier, _ = library
    original = seed_fact(library)
    prompts = input_script(monkeypatch, [
        "similar", "1", "1", "Parking $20 monthly.", "no",
        "Parking", "", "20", "", "", "", "", "save", "leave", "quit",
    ])
    run_fact_workbench(catalogue, identifier)
    facts = list_facts(catalogue, [identifier])
    assert len(facts) == 2
    new = next(fact for fact in facts if fact["fact_id"] != original["fact_id"])
    assert new["payload"]["name"] == "Parking"
    assert new["payload"]["value"] == "20"
    assert new["payload"]["currency"] == "USD" and new["payload"]["period"] == "monthly"
    assert new["payload"]["conditions"] == new["payload"]["uncertainty"] == ""
    assert new["payload"]["candidate_id"] == ""
    assert new["payload"]["support"][0]["quote"] == "Parking $20 monthly."
    assert new["decision"] == "unreviewed" and new["review_history"] == []
    assert fact_history(catalogue, original["fact_id"])[0] == original
    assert any("Kind" in prompt and "[money]" in prompt for prompt in prompts)
    assert any("Currency" in prompt and "[USD]" in prompt for prompt in prompts)
    assert any(prompt == "Fact name: " for prompt in prompts)
    assert any(prompt == "Value (ISO date or decimal string for typed values): " for prompt in prompts)
    assert "confirmation are not copied" in capsys.readouterr().out


def test_similar_metadata_can_be_overridden_without_changing_original(library, monkeypatch):
    catalogue, identifier, _ = library
    seed_fact(library)
    input_script(monkeypatch, [
        "similar", "1", "1", "", "no",
        "Text fact", "text", "Rent requires payment", "", "", "save", "leave", "quit",
    ])
    run_fact_workbench(catalogue, identifier)
    record = next(f for f in list_facts(catalogue, [identifier]) if f["payload"]["kind"] == "text")
    assert record["payload"]["currency"] == record["payload"]["period"] == record["payload"]["unit"] == ""
    assert record["decision"] == "unreviewed"


def test_similar_cancel_does_not_create_fact(library, monkeypatch):
    catalogue, identifier, _ = library
    original = seed_fact(library)
    input_script(monkeypatch, ["similar", "1", "/cancel", "quit"])
    run_fact_workbench(catalogue, identifier)
    assert list_facts(catalogue, [identifier]) == [original]


def test_review_queue_requires_individual_decisions_and_skips_confirmed_rejected(
    library, monkeypatch, capsys,
):
    catalogue, identifier, _ = library
    original = seed_fact(library)
    for name, decision in (("A pending", "unreviewed"), ("B pending", "unreviewed"),
                           ("Rejected", "rejected")):
        saved = save_fact(catalogue, {**original["payload"], "name": name})
        if decision != "unreviewed":
            review_fact(catalogue, saved["fact_id"], 1, decision)
    ordered = [f for f in list_facts(catalogue, [identifier]) if f["decision"] == "unreviewed"]
    prompts = input_script(monkeypatch, ["review-pending", "confirm", "Checked first", "", "quit"])
    run_fact_workbench(catalogue, identifier)
    by_id = {f["fact_id"]: f for f in list_facts(catalogue, [identifier])}
    assert by_id[ordered[0]["fact_id"]]["decision"] == "confirmed"
    assert by_id[ordered[1]["fact_id"]]["decision"] == "unreviewed"
    assert by_id[original["fact_id"]] == original
    assert next(f for f in by_id.values() if f["payload"]["name"] == "Rejected")["decision"] == "rejected"
    output = capsys.readouterr().out
    assert "Review queue: 2" in output and "Review 2/2" in output
    assert "Rent $10 monthly." in output
    assert "No calculations were performed" in output
    assert sum(prompt.startswith("Review decision") for prompt in prompts) == 2
    assert not any(prompt.startswith("Fact number") for prompt in prompts)


@pytest.mark.parametrize("ending", ["/cancel", EOFError(), KeyboardInterrupt()])
def test_review_queue_interrupt_preserves_prior_decisions(library, monkeypatch, ending):
    catalogue, identifier, _ = library
    original = seed_fact(library)
    review_fact(catalogue, original["fact_id"], 1, "unreviewed")
    save_fact(catalogue, {**original["payload"], "name": "Second"})
    ordered = list_facts(catalogue, [identifier])
    answers = ["review-pending", "confirm", "", ending]
    if not isinstance(ending, BaseException):
        answers.append("quit")
    input_script(monkeypatch, answers)
    run_fact_workbench(catalogue, identifier)
    by_id = {f["fact_id"]: f for f in list_facts(catalogue, [identifier])}
    assert by_id[ordered[0]["fact_id"]]["decision"] == "confirmed"
    assert by_id[ordered[1]["fact_id"]]["decision"] == "unreviewed"


def test_review_queue_excludes_stale_facts_explicitly(library, monkeypatch, capsys):
    catalogue, identifier, path = library
    record = seed_fact(library)
    review_fact(catalogue, record["fact_id"], 1, "unreviewed")
    path.write_text("Changed terms.")
    catalogue.ingest(path)
    input_script(monkeypatch, ["review-pending", "quit"])
    run_fact_workbench(catalogue, identifier)
    output = capsys.readouterr().out
    assert "Review queue: 0" in output and "Stale fact excluded" in output
    assert list_facts(catalogue, [identifier])[0]["decision"] == "unreviewed"


def test_review_queue_refuses_changed_revision_before_confirmation(library, monkeypatch, capsys):
    catalogue, identifier, _ = library
    record = seed_fact(library)
    review_fact(catalogue, record["fact_id"], 1, "unreviewed")
    input_script(monkeypatch, ["review-pending", "confirm", "", "quit"])
    scripted = builtins.input

    def change_at_confirmation(message):
        answer = scripted(message)
        if message.startswith("Review decision"):
            save_fact(catalogue, {**record["payload"], "value": "99"},
                      fact_id=record["fact_id"], expected_revision=1)
        return answer

    monkeypatch.setattr("builtins.input", change_at_confirmation)
    run_fact_workbench(catalogue, identifier)
    assert list_facts(catalogue, [identifier])[0]["decision"] == "unreviewed"
    assert capsys.readouterr().err


@pytest.mark.parametrize(("selection", "expected"), [
    ("1-3", [1, 2, 3]), ("3,1-2", [3, 1, 2]), (" 1 - 2 , 4", [1, 2, 4]),
    ("2-2", [2]), ("1-128", list(range(1, 129))),
])
def test_range_selection_preserves_explicit_order(monkeypatch, selection, expected):
    from document_fact_ui import select_numbers

    input_script(monkeypatch, [selection])
    assert select_numbers(128) == expected


@pytest.mark.parametrize("invalid", [
    "1-2,2", "2-1", "0-2", "1-129", "1-999999999", "1,1", "all", "", "1,,2",
    "-1", "1-2-3", "١", "1" * 1025,
])
def test_invalid_ranges_never_select_silently(monkeypatch, capsys, invalid):
    from document_fact_ui import select_numbers

    input_script(monkeypatch, [invalid, "2,1"])
    assert select_numbers(128) == [2, 1]
    assert "Select distinct numbers" in capsys.readouterr().out


def test_range_calculation_still_requires_preview_and_run(library, monkeypatch, capsys):
    catalogue, identifier, _ = library
    record = seed_fact(library)
    saved = save_fact(catalogue, {
        **record["payload"], "name": "Parking", "value": "20",
        "support": [{"segment": 1, "char_start": 18, "char_end": 38, "quote": "Parking $20 monthly."}],
    })
    review_fact(catalogue, saved["fact_id"], 1, "confirmed")
    input_script(monkeypatch, ["calculate", "1-2", "sum", "run", "quit"])
    run_fact_workbench(catalogue, identifier)
    output = capsys.readouterr().out
    assert output.index("Calculation preview") < output.index("Calculation result")
    result = json.loads(output.split("Calculation result: ", 1)[1])
    assert result["value"] == "30"


def test_streamlined_workflow_persists_across_cli_processes(library):
    import subprocess

    catalogue, identifier, _ = library
    original = seed_fact(library)
    answers = [
        "similar", "1", "1", "Parking $20 monthly.", "no",
        "Parking", "", "20", "", "", "Old condition", "", "save", "leave",
        "review-pending", "confirm", "Checked parking evidence",
        "calculate", "1-2", "sum", "run", "quit",
    ]
    result = subprocess.run([
        sys.executable, "document_library.py", "--database", str(catalogue.database),
        "fact-workbench", "--document", identifier,
    ], input="\n".join(answers) + "\n", text=True, capture_output=True, check=True)
    assert "Review queue: 1" in result.stdout
    calculation, _ = json.JSONDecoder().raw_decode(result.stdout.split("Calculation result: ", 1)[1])
    assert calculation["value"] == "30" and not calculation["document_total"]
    records = list_facts(DocumentLibrary(catalogue.database), [identifier])
    assert len(records) == 2 and all(record["decision"] == "confirmed" for record in records)
    assert next(record for record in records if record["fact_id"] == original["fact_id"]) == original
