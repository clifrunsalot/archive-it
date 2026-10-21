import json
import logging
import sys
from copy import deepcopy
from unittest.mock import Mock

import pytest

from document_facts import (
    fact_history, list_facts, preview_fact_query, query_facts, review_fact, save_fact,
)
from document_library import DocumentLibrary, DocumentQueryError


@pytest.fixture
def library(tmp_path):
    source = tmp_path / "facts.txt"
    source.write_text(
        "Rent $0.10 monthly. Parking $0.20 monthly. "
        "Term begins 2026-08-01 and ends 2027-07-31. "
        "There are 2 keys. Pets need permission."
    )
    catalogue = DocumentLibrary(tmp_path / "catalogue.sqlite3")
    identifier = catalogue.ingest(source)
    return catalogue, identifier, source


def payload(library, quote="Rent $0.10 monthly.", **changes):
    catalogue, identifier, _ = library
    snapshot = catalogue.snapshots([identifier])[0]
    text = snapshot.extraction["segments"][0]["text"]
    start = text.index(quote)
    value = {
        "document_id": identifier, "version_id": snapshot.version_id,
        "name": "Rent", "kind": "money", "value": "0.10",
        "unit": "", "currency": "USD", "period": "monthly",
        "conditions": "", "uncertainty": "", "candidate_id": "",
        "support": [{
            "segment": 1, "char_start": start, "char_end": start + len(quote), "quote": quote,
        }],
    }
    value.update(changes)
    return value


def confirmed(library, **changes):
    record = save_fact(library[0], payload(library, **changes))
    return review_fact(library[0], record["fact_id"], 1, "confirmed")


@pytest.mark.parametrize("changes", [
    {},
    {"kind": "date", "value": "2026-08-01", "currency": "", "period": ""},
    {"kind": "quantity", "value": "2", "currency": "", "period": "", "unit": "keys"},
    {"kind": "text", "value": "Permission required", "currency": "", "period": ""},
])
def test_add_and_confirm_every_kind_without_a_model(library, changes):
    catalogue, identifier, _ = library
    record = save_fact(catalogue, payload(library, **changes))
    assert record["revision"] == 1 and record["decision"] == "unreviewed"
    assert not record["usable_for_queries"] and not record["stale"]
    assert record["quote_provenance_verified"] and not record["semantic_relevance_verified"]
    assert record["origin"] == "human_authored"
    evidence = record["evidence"][0]
    assert evidence["source_path"] == str(library[2].resolve())
    assert evidence["quote"] == record["payload"]["support"][0]["quote"]
    assert evidence["location"]["char_start"] == record["payload"]["support"][0]["char_start"]
    assert list_facts(DocumentLibrary(catalogue.database), [identifier]) == [record]
    record = review_fact(catalogue, record["fact_id"], 1, "confirmed", "Checked source")
    assert record["usable_for_queries"]
    assert record["review_history"][0]["note"] == "Checked source"
    assert query_facts(catalogue, [record["fact_id"]], "count")["value"] == 1


@pytest.mark.parametrize(("changes", "reason"), [
    ({"name": ""}, "requires a name"),
    ({"kind": "number"}, "supported kind"),
    ({"value": 0.1}, "bounded strings"),
    ({"value": "NaN"}, "decimal strings"),
    ({"value": "Infinity"}, "decimal strings"),
    ({"value": "1e2"}, "decimal strings"),
    ({"value": "1,000"}, "decimal strings"),
    ({"value": "01.2"}, "decimal strings"),
    ({"value": "0.0000001"}, "decimal strings"),
    ({"value": "9" * 25}, "decimal strings"),
    ({"value": True}, "bounded strings"),
    ({"value": " 1.0"}, "bounded strings"),
    ({"value": "x" * 351}, "bounded strings"),
    ({"currency": "usd"}, "currency code"),
    ({"currency": ""}, "currency code"),
    ({"period": ""}, "currency code"),
    ({"period": "per lease"}, "currency code"),
    ({"unit": "dollars"}, "currency code"),
    ({"kind": "date", "value": "08/01/2026", "currency": "", "period": ""}, "ISO"),
    ({"kind": "date", "value": "2026-02-30", "currency": "", "period": ""}, "calendar"),
    ({"kind": "date", "value": "20260801", "currency": "", "period": ""}, "ISO"),
    ({"kind": "date", "value": "2026-08-01"}, "cannot have units"),
    ({"kind": "text", "value": "Rent"}, "cannot have units"),
    ({"kind": "quantity", "value": "2", "currency": "", "period": ""}, "explicit unit"),
    ({"version_id": True}, "stale"),
    ({"version_id": 99999}, "stale"),
    ({"support": []}, "1-16"),
    ({"unexpected": True}, "schema"),
    ({"conditions": "x" * 2001}, "bounded strings"),
    ({"candidate_id": "missing"}, "candidate does not exist"),
])
def test_invalid_values_never_create_facts(library, changes, reason):
    with pytest.raises(DocumentQueryError, match=reason):
        save_fact(library[0], payload(library, **changes))
    assert list_facts(library[0], [library[1]]) == []


@pytest.mark.parametrize("changes", [
    {"char_start": True}, {"char_start": -1}, {"char_end": 999999},
    {"char_end": 0}, {"quote": "Made up evidence"}, {"segment": 0}, {"segment": True},
    {"segment": 999}, {"quote": ""}, {"extra": 0},
])
def test_invalid_exact_spans_never_publish(library, changes):
    fact = payload(library)
    fact["support"][0].update(changes)
    with pytest.raises(DocumentQueryError, match="support"):
        save_fact(library[0], fact)
    assert list_facts(library[0], [library[1]]) == []


def test_support_unicode_offsets_and_duplicate_spans(library):
    catalogue, _, source = library
    source.write_text("Fee €0.10 monthly.\nSecond line.")
    catalogue.ingest(source)
    fact = payload(library, quote="€0.10")
    record = save_fact(catalogue, fact)
    assert record["payload"]["support"][0]["char_start"] == 4
    fact["support"].append(deepcopy(fact["support"][0]))
    with pytest.raises(DocumentQueryError, match="duplicate"):
        save_fact(catalogue, fact)


def test_correction_preserves_history_and_never_inherits_confirmation(library):
    catalogue, identifier, _ = library
    original = confirmed(library)
    corrected = save_fact(
        catalogue, payload(library, value="0.20"),
        fact_id=original["fact_id"], expected_revision=1,
    )
    assert corrected["revision"] == 2 and corrected["decision"] == "unreviewed"
    assert not corrected["usable_for_queries"]
    with pytest.raises(DocumentQueryError, match="human-confirmed"):
        query_facts(catalogue, [original["fact_id"]], "sum")
    with pytest.raises(DocumentQueryError, match="revision changed"):
        review_fact(catalogue, original["fact_id"], 1, "confirmed")
    with pytest.raises(DocumentQueryError, match="revision changed"):
        save_fact(catalogue, payload(library), fact_id=original["fact_id"], expected_revision=1)
    history = fact_history(DocumentLibrary(catalogue.database), original["fact_id"])
    assert len(history) == 2
    assert history[0]["payload"]["value"] == "0.10"
    assert history[0]["review_history"][0]["decision"] == "confirmed"
    assert history[0]["superseded"] and not history[0]["usable_for_queries"]
    assert not history[1]["superseded"]
    assert history[1]["review_history"] == []
    assert list_facts(catalogue, [identifier]) == [corrected]


def test_review_reject_reset_are_audited(library):
    record = confirmed(library)
    for decision in ("rejected", "unreviewed", "confirmed"):
        record = review_fact(library[0], record["fact_id"], 1, decision)
    assert [event["decision"] for event in record["review_history"]] == [
        "confirmed", "rejected", "unreviewed", "confirmed",
    ]
    assert all(event["created_at"] for event in record["review_history"])


def test_stale_version_blocks_query_review_but_allows_corrected_revision(library):
    catalogue, identifier, source = library
    record = confirmed(library)
    source.write_text("Rent $0.20 monthly.")
    catalogue.ingest(source)
    assert list_facts(catalogue, [identifier])[0]["stale"]
    with pytest.raises(DocumentQueryError, match="current and human-confirmed"):
        query_facts(catalogue, [record["fact_id"]], "sum")
    with pytest.raises(DocumentQueryError, match="source changed"):
        review_fact(catalogue, record["fact_id"], 1, "confirmed")
    corrected = save_fact(
        catalogue, payload(library, quote="Rent $0.20 monthly.", value="0.20"),
        fact_id=record["fact_id"], expected_revision=1,
    )
    assert not corrected["stale"] and corrected["decision"] == "unreviewed"
    assert fact_history(catalogue, record["fact_id"])[0]["stale"]


def test_changed_ingestion_status_invalidates_confirmation(library):
    catalogue, identifier, _ = library
    record = confirmed(library)
    with catalogue.connect() as connection:
        connection.execute(
            "INSERT INTO attempts(document_id,status,reason) VALUES (?,'failed','test')",
            (identifier,),
        )
    with pytest.raises(DocumentQueryError, match="human-confirmed"):
        query_facts(catalogue, [record["fact_id"]], "count")


def test_decimal_sum_is_exact_and_reports_narrow_scope(library):
    first = confirmed(library)
    second = confirmed(library, quote="Parking $0.20 monthly.", name="Parking", value="0.20")
    result = query_facts(library[0], [first["fact_id"], second["fact_id"]], "sum")
    assert result["value"] == "0.30"
    assert result["currency"] == "USD" and result["period"] == "monthly"
    assert not result["document_total"] and not result["exhaustive"]
    assert "explicitly selected human-confirmed" in result["scope"]
    assert [record["revision"] for record in result["records"]] == [1, 1]
    assert result["records"][0]["payload"]["support"]
    assert query_facts(library[0], [first["fact_id"], second["fact_id"]], "compare")["relation"] == "less"
    assert [r["payload"]["value"] for r in query_facts(
        library[0], [second["fact_id"], first["fact_id"]], "sort",
    )["records"]] == ["0.10", "0.20"]


@pytest.mark.parametrize("changes", [
    {"currency": "EUR"}, {"period": "yearly"}, {"conditions": "Only if renewed"},
    {"uncertainty": "Fee may change"},
    {"kind": "quantity", "currency": "", "period": "", "unit": "keys"},
])
def test_sum_refuses_incompatible_values_conditions_or_uncertainty(library, changes):
    first = confirmed(library)
    second = confirmed(library, quote="Parking $0.20 monthly.", **changes)
    with pytest.raises(DocumentQueryError):
        query_facts(library[0], [first["fact_id"], second["fact_id"]], "sum")


def test_sum_blocks_duplicate_ids_and_overlapping_source_evidence(library):
    first = confirmed(library)
    with pytest.raises(DocumentQueryError, match="distinct"):
        query_facts(library[0], [first["fact_id"], first["fact_id"]], "sum")
    duplicate = confirmed(library)
    with pytest.raises(DocumentQueryError, match="double-counting"):
        query_facts(library[0], [first["fact_id"], duplicate["fact_id"]], "sum")
    overlap = confirmed(library, quote="$0.10", name="Overlapping evidence")
    with pytest.raises(DocumentQueryError, match="double-counting"):
        query_facts(library[0], [first["fact_id"], overlap["fact_id"]], "sum")


def test_quantity_negative_values_and_large_precise_sums(library):
    first = confirmed(library, kind="quantity", currency="", period="", unit="keys", value="-2.5")
    second = confirmed(
        library, quote="There are 2 keys.", kind="quantity", currency="", period="",
        unit="keys", value="999999999999999999999999.999999",
    )
    result = query_facts(library[0], [first["fact_id"], second["fact_id"]], "sum")
    assert result["value"] == "999999999999999999999997.499999"
    assert result["unit"] == "keys"


def test_date_filters_sort_and_comparison(library):
    first = confirmed(
        library, quote="Term begins 2026-08-01 and ends 2027-07-31.",
        kind="date", value="2026-08-01", currency="", period="",
    )
    second = confirmed(
        library, quote="Term begins 2026-08-01 and ends 2027-07-31.",
        kind="date", value="2027-07-31", currency="", period="",
    )
    ids = [second["fact_id"], first["fact_id"]]
    assert [r["payload"]["value"] for r in query_facts(library[0], ids, "sort")["records"]] == [
        "2026-08-01", "2027-07-31",
    ]
    assert query_facts(library[0], ids, "compare")["relation"] == "greater"
    assert query_facts(library[0], ids, "count", lower="2027-01-01")["value"] == 1
    assert query_facts(library[0], ids, "count", upper="2026-08-01")["value"] == 1
    assert query_facts(library[0], ids, "count", lower="2030-01-01")["value"] == 0
    with pytest.raises(DocumentQueryError, match="Only compatible"):
        query_facts(library[0], ids, "sum")
    for lower, upper in (("2027-02-30", None), ("20270101", None), ("2027-01-01", "2026-01-01")):
        with pytest.raises(DocumentQueryError):
            query_facts(library[0], ids, "count", lower=lower, upper=upper)
    with pytest.raises(DocumentQueryError, match="only for count and sort"):
        query_facts(library[0], ids, "compare", lower="2027-01-01")


def test_never_silently_skip_unconfirmed_rejected_or_unknown_ids(library):
    first = confirmed(library)
    second = save_fact(library[0], payload(library, quote="Parking $0.20 monthly."))
    for decision in ("unreviewed", "rejected"):
        review_fact(library[0], second["fact_id"], 1, decision)
        with pytest.raises(DocumentQueryError, match="none were skipped"):
            query_facts(library[0], [first["fact_id"], second["fact_id"]], "count")
    with pytest.raises(DocumentQueryError, match="does not exist"):
        query_facts(library[0], [first["fact_id"], "missing"], "count")
    with pytest.raises(DocumentQueryError, match="1-128"):
        query_facts(library[0], [], "sum")
    with pytest.raises(DocumentQueryError, match="exactly two"):
        query_facts(library[0], [first["fact_id"]], "compare")
    with pytest.raises(DocumentQueryError, match="only date"):
        query_facts(library[0], [first["fact_id"]], "count", lower="2026-01-01")


def test_notes_values_and_quotes_never_logged(library, caplog):
    caplog.set_level(logging.DEBUG)
    fact = save_fact(library[0], payload(library, name="PRIVATE FACT", conditions="PRIVATE CONDITION"))
    review_fact(library[0], fact["fact_id"], 1, "confirmed", "PRIVATE NOTE")
    query_facts(library[0], [fact["fact_id"]], "sum")
    assert all(text not in caplog.text for text in (
        "PRIVATE FACT", "PRIVATE CONDITION", "PRIVATE NOTE", "Rent $0.10 monthly.",
    ))


def test_cli_add_review_query_and_history_are_model_free(library, tmp_path, monkeypatch, capsys):
    import chatbot_tech_doc
    import document_library

    monkeypatch.setattr(document_library, "configure_logging", lambda _: None)
    model = Mock(side_effect=AssertionError("Typed facts must not invoke a model"))
    monkeypatch.setattr(chatbot_tech_doc, "get_llm", model)
    path = tmp_path / "fact.json"
    path.write_text(json.dumps(payload(library)))

    def run(*args):
        monkeypatch.setattr(sys, "argv", [
            "document_library.py", "--database", str(library[0].database), *args,
        ])
        assert document_library.main() == 0
        return json.loads(capsys.readouterr().out)

    saved = run("fact-save", str(path))
    identifier = saved["fact_id"]
    assert run("facts", "--document", library[1])[0]["fact_id"] == identifier
    reviewed = run("fact-review", identifier, "--revision", "1", "--decision", "confirmed")
    assert reviewed["usable_for_queries"]
    assert run("fact-query", "sum", "--fact", identifier)["value"] == "0.10"
    assert run("fact-history", identifier)[0]["review_history"][0]["decision"] == "confirmed"
    path.write_text(json.dumps(payload(library, value="0.15")))
    corrected = run("fact-save", str(path), "--fact", identifier, "--revision", "1")
    assert corrected["revision"] == 2 and corrected["decision"] == "unreviewed"
    assert len(run("fact-history", identifier)) == 2
    model.assert_not_called()


def test_link_inventory_candidate_without_inheriting_acceptance(library):
    from document_inventory_review import prepare

    catalogue, identifier, _ = library
    candidate = {
        "document_id": identifier, "version_id": payload(library)["version_id"],
    }
    with catalogue.connect() as connection:
        prepare(connection)
        connection.execute(
            "INSERT INTO inventory_review_candidates(id,question,item) VALUES (?,?,?)",
            ("candidate", "List fees", json.dumps(candidate)),
        )
        connection.execute(
            "INSERT INTO inventory_review_events(candidate,decision,note) VALUES (?,?,?)",
            ("candidate", "accepted", ""),
        )
    record = save_fact(catalogue, payload(library, candidate_id="candidate"))
    assert record["payload"]["candidate_id"] == "candidate"
    assert record["decision"] == "unreviewed"
    for field, value in (("document_id", "another"), ("version_id", 999)):
        modified = {**candidate, field: value}
        with catalogue.connect() as connection:
            connection.execute(
                "UPDATE inventory_review_candidates SET item=? WHERE id='candidate'",
                (json.dumps(modified),),
            )
        with pytest.raises(DocumentQueryError, match="another document or version"):
            save_fact(catalogue, payload(library, candidate_id="candidate"))


def test_corrections_cannot_move_documents_and_require_expected_revision(library, tmp_path):
    catalogue, _, _ = library
    record = confirmed(library)
    attempts: list[tuple[str | None, int | None]] = [
        (record["fact_id"], None), (None, 1), (record["fact_id"], True),
        (record["fact_id"], 0),
    ]
    for fact_id, revision in attempts:
        with pytest.raises(DocumentQueryError):
            save_fact(catalogue, payload(library), fact_id=fact_id, expected_revision=revision)
    other = tmp_path / "other.txt"
    other.write_text("Rent $0.10 monthly.")
    identifier = catalogue.ingest(other)
    version = catalogue.snapshots([identifier])[0].version_id
    with pytest.raises(DocumentQueryError, match="another document"):
        save_fact(
            catalogue, payload(library, document_id=identifier, version_id=version),
            fact_id=record["fact_id"], expected_revision=1,
        )
    assert len(fact_history(catalogue, record["fact_id"])) == 1


def test_query_quantity_units_are_not_implicitly_converted(library):
    first = confirmed(library, kind="quantity", currency="", period="", unit="m", value="1")
    second = confirmed(
        library, quote="Parking $0.20 monthly.", kind="quantity", currency="",
        period="", unit="cm", value="100",
    )
    for operation in ("sum", "sort", "compare"):
        with pytest.raises(DocumentQueryError, match="must match"):
            query_facts(library[0], [first["fact_id"], second["fact_id"]], operation)
    # Counting records is not a sum of their heterogeneous values.
    assert query_facts(library[0], [first["fact_id"], second["fact_id"]], "count")["value"] == 2


def test_text_sort_and_equal_comparison_are_deterministic(library):
    first = confirmed(library, kind="text", value="Permission", currency="", period="")
    second = confirmed(
        library, quote="Pets need permission.", kind="text",
        value="Permission", currency="", period="",
    )
    identifiers = [first["fact_id"], second["fact_id"]]
    result = query_facts(library[0], identifiers, "compare")
    assert result["relation"] == "equal"
    assert [record["fact_id"] for record in result["records"]] == sorted(identifiers)
    with pytest.raises(DocumentQueryError, match="Only compatible"):
        query_facts(library[0], identifiers, "sum")


def test_empty_invalid_missing_facts_and_review_input_are_explicit(library):
    assert list_facts(library[0], [library[1]]) == []
    with pytest.raises(DocumentQueryError, match="Select documents explicitly"):
        list_facts(library[0], None)
    with pytest.raises(DocumentQueryError, match="does not exist"):
        fact_history(library[0], "missing")
    with pytest.raises(DocumentQueryError, match="document_id"):
        save_fact(library[0], [])
    with pytest.raises(DocumentQueryError, match="no current"):
        save_fact(library[0], payload(library, document_id="missing"))
    record = confirmed(library)
    for revision, decision, note in (
        (True, "confirmed", ""), (1, "accepted", ""), (1, "confirmed", "x" * 2001),
    ):
        with pytest.raises(DocumentQueryError):
            review_fact(library[0], record["fact_id"], revision, decision, note)


def test_cli_bad_json_fails_without_content_disclosure(library, tmp_path, monkeypatch, capsys):
    import document_library

    monkeypatch.setattr(document_library, "configure_logging", lambda _: None)
    path = tmp_path / "invalid.json"
    path.write_text('{"PRIVATE INVALID PAYLOAD":')
    monkeypatch.setattr(sys, "argv", [
        "document_library.py", "--database", str(library[0].database), "fact-save", str(path),
    ])
    assert document_library.main() == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "valid JSON" in captured.err and "PRIVATE" not in captured.err
    assert list_facts(library[0], [library[1]]) == []


def test_cli_persists_fact_and_confirmation_across_processes(library, tmp_path):
    import subprocess
    from pathlib import Path

    path = tmp_path / "fact.json"
    path.write_text(json.dumps(payload(library)))

    def run(*args):
        completed = subprocess.run(
            [
                sys.executable, "document_library.py", "--database", str(library[0].database),
                *args,
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=30, check=True,
        )
        return json.loads(completed.stdout)

    saved = run("fact-save", str(path))
    reviewed = run(
        "fact-review", saved["fact_id"], "--revision", "1", "--decision", "confirmed",
    )
    assert reviewed["usable_for_queries"]
    preview = run("fact-query", "sum", "--fact", saved["fact_id"], "--preview")
    assert preview["ready"] and preview["calculation_performed"] is False
    result = run(
        "fact-query", "sum", "--fact", saved["fact_id"],
        "--expected-preview", preview["preview_token"],
    )
    assert result["value"] == "0.10"
    assert result["records"][0]["evidence"][0]["quote"] == "Rent $0.10 monthly."


@pytest.mark.parametrize("operation", ["count", "sum", "sort", "compare"])
def test_preview_has_evidence_and_compatibility_but_no_computed_result(library, operation):
    first = confirmed(library)
    second = confirmed(library, quote="Parking $0.20 monthly.", name="Parking", value="0.20")
    identifiers = [second["fact_id"], first["fact_id"]]
    preview = preview_fact_query(library[0], identifiers, operation)
    assert preview["ready"] and not preview["calculation_performed"]
    assert preview["blocking_reasons"] == [] and preview["compatibility"] == "passed"
    assert preview["passed_checks"]
    assert "value" not in preview and "relation" not in preview
    assert preview["included_fact_ids"] == identifiers
    assert preview["exclusions"] == []
    assert [r["fact_id"] for r in preview["selected_records"]] == identifiers
    assert preview["selected_records"][0]["evidence"][0]["quote"] == "Parking $0.20 monthly."
    assert preview["preview_token"] == preview_fact_query(
        library[0], identifiers, operation,
    )["preview_token"]
    result = query_facts(
        library[0], identifiers, operation, expected_preview=preview["preview_token"],
    )
    assert result == query_facts(library[0], identifiers, operation)


@pytest.mark.parametrize("changes", [
    {"currency": "EUR"}, {"period": "yearly"},
    {"conditions": "Only if renewed"}, {"uncertainty": "Unknown fee"},
])
def test_incompatible_preview_is_blocked_without_partial_total(library, changes):
    first = confirmed(library)
    second = confirmed(library, quote="Parking $0.20 monthly.", **changes)
    preview = preview_fact_query(library[0], [first["fact_id"], second["fact_id"]], "sum")
    assert not preview["ready"]
    assert len(preview["selected_records"]) == 2
    assert preview["included_fact_ids"] == [] and preview["exclusions"] == []
    assert preview["blocking_reasons"] and "value" not in preview
    with pytest.raises(DocumentQueryError):
        query_facts(
            library[0], preview["selected_fact_ids"], "sum",
            expected_preview=preview["preview_token"],
        )


def test_preview_reports_every_selected_status_not_silent_exclusions(library):
    first = confirmed(library)
    second = save_fact(library[0], payload(library, quote="Parking $0.20 monthly."))
    third = save_fact(library[0], payload(
        library, quote="There are 2 keys.", uncertainty="Unknown eligibility",
    ))
    review_fact(library[0], third["fact_id"], 1, "rejected")
    preview = preview_fact_query(
        library[0], [first["fact_id"], second["fact_id"], third["fact_id"]], "count",
    )
    assert not preview["ready"] and len(preview["selected_records"]) == 3
    issues = [issue for issue in preview["blocking_reasons"] if issue["fact_id"] is not None]
    assert {issue["fact_id"] for issue in issues} == {second["fact_id"], third["fact_id"]}
    assert any("uncertainty" in issue["reason"] for issue in issues)
    assert preview["exclusions"] == []


def test_date_preview_names_excluded_records_and_can_preview_zero_matches(library):
    first = confirmed(
        library, kind="date", value="2026-08-01", currency="", period="",
    )
    second = confirmed(
        library, quote="Parking $0.20 monthly.", kind="date",
        value="2027-07-31", currency="", period="",
    )
    identifiers = [first["fact_id"], second["fact_id"]]
    preview = preview_fact_query(library[0], identifiers, "count", lower="2027-01-01")
    assert preview["ready"] and preview["included_fact_ids"] == [second["fact_id"]]
    assert preview["exclusions"] == [{
        "fact_id": first["fact_id"], "reason": "Outside inclusive date bounds.",
    }]
    empty = preview_fact_query(library[0], identifiers, "count", lower="2030-01-01")
    assert empty["ready"] and not empty["included_fact_ids"] and len(empty["exclusions"]) == 2
    assert query_facts(
        library[0], identifiers, "count", lower="2030-01-01",
        expected_preview=empty["preview_token"],
    )["value"] == 0


@pytest.mark.parametrize("change", ["revision", "review", "source", "ingestion", "operation", "order", "bounds"])
def test_execution_refuses_any_change_after_preview(library, change):
    catalogue, identifier, source = library
    first = confirmed(library)
    second = confirmed(library, quote="Parking $0.20 monthly.", value="0.20")
    identifiers = [first["fact_id"], second["fact_id"]]
    operation = "sum"
    preview = preview_fact_query(catalogue, identifiers, operation)
    options = {}
    if change == "revision":
        save_fact(catalogue, payload(library, value="0.50"),
                  fact_id=first["fact_id"], expected_revision=1)
    elif change == "review":
        # Even a reconfirmation creates a new audit event and requires a new preview.
        review_fact(catalogue, first["fact_id"], 1, "confirmed", "Rechecked")
    elif change == "source":
        source.write_text("Changed source.")
        catalogue.ingest(source)
    elif change == "ingestion":
        with catalogue.connect() as connection:
            connection.execute(
                "INSERT INTO attempts(document_id,status,reason) VALUES (?,'failed','test')",
                (identifier,),
            )
    elif change == "operation":
        operation = "count"
    elif change == "order":
        identifiers.reverse()
    else:
        options["lower"] = "2027-01-01"
    with pytest.raises(DocumentQueryError, match="changed since preview"):
        query_facts(
            catalogue, identifiers, operation, expected_preview=preview["preview_token"], **options,
        )


def test_preview_duplicate_missing_and_invalid_date_queries_are_explicit(library):
    first = confirmed(library)
    with pytest.raises(DocumentQueryError, match="distinct"):
        preview_fact_query(library[0], [first["fact_id"], first["fact_id"]], "sum")
    with pytest.raises(DocumentQueryError, match="does not exist"):
        preview_fact_query(library[0], ["missing"], "sum")
    preview = preview_fact_query(library[0], [first["fact_id"]], "count", lower="2026-01-01")
    assert not preview["ready"] and "date facts" in preview["blocking_reasons"][0]["reason"]
    overlap = confirmed(library)
    preview = preview_fact_query(library[0], [first["fact_id"], overlap["fact_id"]], "sum")
    assert not preview["ready"] and "double-counting" in preview["blocking_reasons"][0]["reason"]


def test_preview_cli_and_token_guard(library, monkeypatch, capsys):
    import document_library

    record = confirmed(library)
    monkeypatch.setattr(document_library, "configure_logging", lambda _: None)
    argv = [
        "document_library.py", "--database", str(library[0].database),
        "fact-query", "sum", "--fact", record["fact_id"],
    ]
    monkeypatch.setattr(sys, "argv", [*argv, "--preview"])
    assert document_library.main() == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["ready"] and "value" not in preview
    monkeypatch.setattr(sys, "argv", [*argv, "--expected-preview", preview["preview_token"]])
    assert document_library.main() == 0
    assert json.loads(capsys.readouterr().out)["value"] == "0.10"
    review_fact(library[0], record["fact_id"], 1, "unreviewed")
    assert document_library.main() == 1
    assert "changed since preview" in capsys.readouterr().err


def test_sum_preview_validates_without_performing_decimal_arithmetic(library, monkeypatch):
    import document_facts

    first = confirmed(library)
    second = confirmed(library, quote="Parking $0.20 monthly.", value="0.20")
    decimal = Mock(side_effect=AssertionError("Preview must not compute numeric results"))
    monkeypatch.setattr(document_facts, "Decimal", decimal)
    preview = preview_fact_query(library[0], [first["fact_id"], second["fact_id"]], "sum")
    assert preview["ready"] and not preview["calculation_performed"]
    assert any("do not overlap" in check for check in preview["passed_checks"])
    decimal.assert_not_called()
