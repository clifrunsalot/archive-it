import json
import logging
import sys
from copy import deepcopy
from dataclasses import asdict
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from document_inventory_review import (
    candidate_key, persist_inventory, refresh_inventory, review_inventory,
)
from document_library import DocumentLibrary, DocumentQueryError
from document_query import render_answer
from document_synthesis import synthesize

QUESTION = "List every amenity in the lease"


@pytest.fixture
def inventory(tmp_path):
    source = tmp_path / "lease.txt"
    source.write_text("Residents have gym access. Package lockers are available to residents.")
    library = DocumentLibrary(tmp_path / "catalogue.sqlite3")
    identifier = library.ingest(source)

    def respond(prompt):
        records = json.loads(prompt.to_messages()[-1].content.split("(JSON):\n", 1)[1])
        return AIMessage(content=json.dumps({
            "status": "findings", "findings": [{
                "text": name, "conditions": "", "assessment": "established",
                "uncertainty": "", "evidence": [{
                    "label": records[0]["label"], "quote": quote,
                }],
            } for name, quote in (
                ("Gym", "Residents have gym access."),
                ("Package lockers", "Package lockers are available to residents."),
            )],
        }))

    model = Mock(side_effect=respond)
    result = synthesize(library, RunnableLambda(model), "enumerate", QUESTION, [identifier])
    return library, identifier, source, result, model


def test_default_candidates_are_unreviewed_never_autoaccepted(inventory):
    library, identifier, _, result, model = inventory
    assert result.coverage["review_counts"] == {
        "accepted": 0, "rejected": 0, "unreviewed": 2,
    }
    assert all(item["review"]["decision"] == "unreviewed" for item in result.items)
    assert all(item["review_history"] == [] for item in result.items)
    assert result.answer.startswith("Unreviewed candidate: Gym")
    assert "Unreviewed candidate: Package lockers" in result.answer
    listed = review_inventory(library, QUESTION, [identifier])
    assert listed == result
    assert model.call_count == 1


def test_explicit_accept_reject_and_reset_are_persistent_auditable_and_local(inventory, caplog):
    library, identifier, _, result, model = inventory
    gym, deposit = result.items
    caplog.set_level(logging.DEBUG)
    accepted = review_inventory(
        library, QUESTION, [identifier], item_id=gym["review_id"], decision="accepted",
        note="PRIVATE REVIEW NOTE",
    )
    assert "Accepted by reviewer: Gym" in accepted.answer
    assert "Unreviewed candidate: Package lockers" in accepted.answer
    rejected = review_inventory(
        DocumentLibrary(library.database), QUESTION, [identifier],
        item_id=deposit["review_id"], decision="rejected", note="Not an amenity",
    )
    assert "Package lockers" not in rejected.answer
    assert rejected.items[1]["review"]["decision"] == "rejected"
    assert rejected.items[1]["support"] == result.items[1]["support"]
    assert rejected.coverage["review_counts"] == {
        "accepted": 1, "rejected": 1, "unreviewed": 0,
    }
    assert not rejected.coverage["semantic_relevance_verified"]
    assert all(not item["semantic_relevance_verified"] for item in rejected.items)
    reset = review_inventory(
        library, QUESTION, [identifier], item_id=gym["review_id"], decision="unreviewed",
        note="Reconsider",
    )
    assert "Unreviewed candidate: Gym" in reset.answer
    history = reset.items[0]["review_history"]
    assert [event["decision"] for event in history] == ["accepted", "unreviewed"]
    assert history[0]["id"] < history[1]["id"]
    assert history[0]["created_at"] and history[1]["created_at"]
    assert "PRIVATE REVIEW NOTE" not in caplog.text
    assert "Not an amenity" not in caplog.text
    assert model.call_count == 1


def test_all_rejected_is_not_reported_as_absence(inventory):
    library, identifier, _, result, _ = inventory
    for item in result.items:
        final = review_inventory(
            library, QUESTION, [identifier], item_id=item["review_id"], decision="rejected",
        )
    assert "All generated candidates" in final.answer
    assert "rejected by the reviewer" in final.answer
    assert "not proof of absence" in final.answer
    assert "No matching items" not in final.answer
    assert len(final.items) == 2
    assert final.coverage["no_relevant_document_ids"] == []
    assert not final.coverage["exhaustive"]


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"item_id": "item"}, "requires --item"),
    ({"decision": "accepted"}, "requires --item"),
    ({"decision": "automatic", "item_id": "item"}, "requires --item"),
    ({"note": "note"}, "requires --item"),
    ({"decision": "accepted", "item_id": "wrong"}, "does not belong"),
    ({"decision": "accepted", "item_id": "wrong", "note": "x" * 2001}, "2000"),
])
def test_invalid_decisions_do_not_change_history(inventory, kwargs, message):
    library, identifier, _, _, _ = inventory
    with pytest.raises(DocumentQueryError, match=message):
        review_inventory(library, QUESTION, [identifier], **kwargs)
    with library.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM inventory_review_events").fetchone()[0] == 0


def test_wrong_question_scope_or_missing_explicit_scope_cannot_apply_decision(inventory):
    library, identifier, _, result, _ = inventory
    with pytest.raises(DocumentQueryError, match="Select documents explicitly"):
        review_inventory(library, QUESTION, None)
    for question, scope in (("List every fee in the lease", [identifier]), (QUESTION, ["other"])):
        with pytest.raises(DocumentQueryError, match="No saved inventory"):
            review_inventory(
                library, question, scope,
                item_id=result.items[0]["review_id"], decision="accepted",
            )
    with library.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM inventory_review_events").fetchone()[0] == 0


def test_changed_version_blocks_review_and_reuse(inventory):
    library, identifier, source, result, _ = inventory
    gym = result.items[0]
    review_inventory(
        library, QUESTION, [identifier], item_id=gym["review_id"], decision="accepted",
    )
    source.write_text(
        "Residents have gym access. Package lockers are available to residents in the updated lease."
    )
    library.ingest(source)
    for action in (
        lambda: review_inventory(library, QUESTION, [identifier]),
        lambda: review_inventory(
            library, QUESTION, [identifier], item_id=gym["review_id"], decision="rejected",
        ),
        lambda: persist_inventory(library, QUESTION, result, 256000),
        lambda: refresh_inventory(library, QUESTION, result),
    ):
        with pytest.raises(DocumentQueryError, match="versions or ingestion status changed"):
            action()
    changed = deepcopy(result)
    version = library.snapshots([identifier])[0].version_id
    changed.coverage["documents"][0]["version_id"] = version
    for item in changed.items:
        item["version_id"] = version
        for support in item["support"]:
            support["version_id"] = version
    current = persist_inventory(library, QUESTION, changed, 256000)
    assert current.items[0]["review_id"] != gym["review_id"]
    assert all(item["review"]["decision"] == "unreviewed" for item in current.items)
    with library.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM inventory_review_events").fetchone()[0] == 1
        archived = connection.execute(
            "SELECT question,item FROM inventory_review_candidates WHERE id=?",
            (gym["review_id"],),
        ).fetchone()
    assert archived["question"] == QUESTION
    assert json.loads(archived["item"])["version_id"] == gym["version_id"]
    assert json.loads(archived["item"])["support"] == gym["support"]


def test_failed_latest_ingestion_blocks_review(inventory):
    library, identifier, _, _, _ = inventory
    with library.connect() as connection:
        connection.execute(
            "INSERT INTO attempts(document_id,status,reason) VALUES (?,'failed','test')",
            (identifier,),
        )
    with pytest.raises(DocumentQueryError, match="versions or ingestion status changed"):
        review_inventory(library, QUESTION, [identifier])


def test_changed_evidence_or_conditions_require_new_review_but_labels_do_not(inventory):
    library, identifier, _, result, _ = inventory
    original = result.items[0]
    review_inventory(
        library, QUESTION, [identifier], item_id=original["review_id"], decision="accepted",
    )
    relabeled = deepcopy(result)
    relabeled.items[0]["support"][0]["label"] = 99
    saved = persist_inventory(library, QUESTION, relabeled, 256000)
    assert saved.items[0]["review"]["decision"] == "accepted"
    assert saved.items[0]["review_id"] == original["review_id"]
    for field, value in (("name", "Free gym"), ("conditions", "Included"), ("uncertainty", "unknown")):
        changed = deepcopy(result)
        changed.items[0][field] = value
        saved = persist_inventory(library, QUESTION, changed, 256000)
        assert saved.items[0]["review"]["decision"] == "unreviewed"
        assert saved.items[0]["review_id"] != original["review_id"]
    changed = deepcopy(result)
    changed.items[0]["support"][0]["location"]["char_start"] += 1
    assert candidate_key(QUESTION, changed.items[0]) != original["review_id"]
    assert candidate_key("List every fee", original) != original["review_id"]


def test_format_refresh_keeps_original_candidates_after_a_newer_run(inventory):
    library, identifier, _, result, _ = inventory
    original = deepcopy(result)
    newer = deepcopy(result)
    newer.items[0]["conditions"] = "Potentially free"
    persist_inventory(library, QUESTION, newer, 256000)
    # Restore the original candidate to record its decision, then replace the saved model run.
    persist_inventory(library, QUESTION, original, 256000)
    review_inventory(
        library, QUESTION, [identifier],
        item_id=original.items[0]["review_id"], decision="accepted",
    )
    persist_inventory(library, QUESTION, newer, 256000)
    refreshed = refresh_inventory(library, QUESTION, original)
    assert refreshed.items[0]["conditions"] == ""
    assert refreshed.items[0]["review"]["decision"] == "accepted"
    assert refreshed.items[1]["support"] == original.items[1]["support"]


def test_reviewed_presentation_and_json_preserve_decision_distinction(inventory):
    library, identifier, _, result, _ = inventory
    final = review_inventory(
        library, QUESTION, [identifier], item_id=result.items[0]["review_id"], decision="accepted",
    )
    for presentation in ("prose", "list", "ascii_table", "csv"):
        output = render_answer(final, presentation)
        assert "Accepted by reviewer: Gym" in output
        assert "Unreviewed candidate: Package lockers" in output
        normalized = " ".join(output.replace("|", " ").split())
        assert "1 accepted, 0 rejected, 1 unreviewed" in normalized
        assert "not establish completeness" in normalized
    saved = json.loads(json.dumps(asdict(final)))
    assert saved["items"][0]["review_history"][0]["decision"] == "accepted"


def test_inventory_review_cli_does_not_invoke_a_model(inventory, monkeypatch, capsys):
    import chatbot_tech_doc
    import document_library

    library, identifier, _, result, _ = inventory
    monkeypatch.setattr(document_library, "configure_logging", lambda _: None)
    model = Mock(side_effect=AssertionError("Review must not invoke a model"))
    monkeypatch.setattr(chatbot_tech_doc, "get_llm", model)
    argv = [
        "document_library.py", "--database", str(library.database),
        "inventory-review", QUESTION, "--document", identifier,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert document_library.main() == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["items"][0]["review_id"] == result.items[0]["review_id"]
    monkeypatch.setattr(sys, "argv", [
        *argv, "--item", result.items[0]["review_id"],
        "--decision", "accepted", "--note", "Checked original lease",
    ])
    assert document_library.main() == 0
    accepted = json.loads(capsys.readouterr().out)
    assert accepted["items"][0]["review"]["decision"] == "accepted"
    assert model.call_count == 0


def test_repeat_generation_retains_decisions_for_identical_candidates(inventory):
    library, identifier, _, result, model = inventory
    review_inventory(
        library, QUESTION, [identifier],
        item_id=result.items[1]["review_id"], decision="rejected",
    )
    repeated = synthesize(library, RunnableLambda(model), "enumerate", QUESTION, [identifier])
    assert repeated.items[1]["review"]["decision"] == "rejected"
    assert repeated.items[1]["review_id"] == result.items[1]["review_id"]
    assert "Package lockers" not in repeated.answer
    assert model.call_count == 2


def test_empty_inventory_is_reviewable_but_has_no_decisions(inventory):
    library, identifier, _, result, _ = inventory
    result.items = []
    persisted = persist_inventory(library, QUESTION, result, 256000)
    listed = review_inventory(library, QUESTION, [identifier])
    assert listed == persisted
    assert "No matching items" in listed.answer
    assert "not proof of absence" in listed.answer
    assert listed.coverage["review_counts"] == {
        "accepted": 0, "rejected": 0, "unreviewed": 0,
    }


def test_failed_save_does_not_replace_previously_saved_candidates(inventory):
    library, identifier, _, result, _ = inventory
    modified = deepcopy(result)
    modified.items[0]["name"] = "Changed candidate"
    with pytest.raises(DocumentQueryError, match="output budget"):
        persist_inventory(library, QUESTION, modified, 1)
    listed = review_inventory(library, QUESTION, [identifier])
    assert listed.items[0]["name"] == "Gym"


def test_chat_reformat_refreshes_human_decisions_without_model_calls(inventory):
    from langchain_core.embeddings import Embeddings
    from document_query import QueryPlan, QuerySession
    from document_search import DocumentSearch

    library, identifier, _, result, _ = inventory
    search = DocumentSearch(library, Mock(spec=Embeddings), "unused")
    planner = Mock(side_effect=AssertionError("No planner required in this focused test"))
    model = Mock(side_effect=AssertionError("Reformat must not invoke the answer model"))
    session = QuerySession(search, RunnableLambda(planner), RunnableLambda(model), [identifier])
    session.previous_result = result
    session.previous_versions = {
        identifier: (result.coverage["documents"][0]["version_id"],
                     result.coverage["documents"][0]["latest_attempt_status"]),
    }
    plan = QueryPlan(
        intent="reformat", search_question="", document_ids=[identifier], presentation="prose",
        clarification=None, filters=[],
    )
    session.plan = Mock(return_value=plan)
    review_inventory(
        library, QUESTION, [identifier],
        item_id=result.items[1]["review_id"], decision="rejected",
    )
    output = session.execute("Reformat the inventory")
    assert "Unreviewed candidate: Gym" in output
    assert "Package lockers" not in output
    assert "1 rejected" in output
    model.assert_not_called()
    planner.assert_not_called()
