"""Local human decisions on version-bound inventory candidates, never model verdicts."""

import hashlib
import json
import sqlite3
from contextlib import closing
from copy import deepcopy
from dataclasses import asdict

from app_logging import LOGGER
from document_inventory import render_inventory
from document_library import DocumentLibrary, DocumentQueryError
from document_search import AnswerResult, validate_question


def scope_key(question: str, document_ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(
        [question.strip(), sorted(set(document_ids))], ensure_ascii=False,
    ).encode()).hexdigest()


def candidate_key(question: str, item: dict) -> str:
    support = [{
        key: span[key] for key in
        ("document_id", "version_id", "segment", "location", "quote")
    } for span in item["support"]]
    support = sorted(
        {json.dumps(span, sort_keys=True, ensure_ascii=False) for span in support},
    )
    identity = {
        key: item[key] for key in
        ("name", "conditions", "assessment", "uncertainty", "document_id", "version_id")
    }
    return hashlib.sha256(json.dumps(
        {"protocol": "human-review-v1", "question": question.strip(),
         "item": identity, "support": support}, sort_keys=True, ensure_ascii=False,
    ).encode()).hexdigest()


def prepare(connection: sqlite3.Connection) -> None:
    connection.execute("""
        CREATE TABLE IF NOT EXISTS inventory_review_runs (
            scope TEXT PRIMARY KEY, result TEXT NOT NULL
        )
    """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS inventory_review_candidates (
            id TEXT PRIMARY KEY, question TEXT NOT NULL, item TEXT NOT NULL
        )
    """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS inventory_review_events (
            id INTEGER PRIMARY KEY,
            candidate TEXT NOT NULL REFERENCES inventory_review_candidates(id),
            decision TEXT NOT NULL CHECK(decision IN ('accepted','rejected','unreviewed')),
            note TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        )
    """)
    connection.execute("""
        CREATE INDEX IF NOT EXISTS inventory_review_candidate
        ON inventory_review_events(candidate, id)
    """)


def check_current(connection: sqlite3.Connection, result: AnswerResult) -> None:
    for document in result.coverage["documents"]:
        row = connection.execute("""
            SELECT current_version,
                   (SELECT status FROM attempts WHERE document_id=d.id ORDER BY id DESC LIMIT 1)
                   AS latest_attempt_status
            FROM documents d WHERE id=?
        """, (document["document_id"],)).fetchone()
        if row is None or (row["current_version"], row["latest_attempt_status"]) != (
            document["version_id"], document["latest_attempt_status"],
        ):
            raise DocumentQueryError(
                "Inventory source versions or ingestion status changed. Enumerate again before review.")


def archive_candidates(connection: sqlite3.Connection, question: str, result: AnswerResult) -> None:
    for item in result.items:
        candidate = {
            key: value for key, value in item.items()
            if key not in {"review_id", "review", "review_history"}
        }
        connection.execute(
            "INSERT OR IGNORE INTO inventory_review_candidates(id,question,item) VALUES (?,?,?)",
            (candidate_key(question, item), question, json.dumps(candidate, ensure_ascii=False)),
        )


def apply_reviews(connection: sqlite3.Connection, question: str, result: AnswerResult) -> AnswerResult:
    reviewed = deepcopy(result)
    for item in reviewed.items:
        identifier = candidate_key(question, item)
        events = connection.execute("""
            SELECT id,decision,note,created_at FROM inventory_review_events
            WHERE candidate=? ORDER BY id
        """, (identifier,)).fetchall()
        item["review_id"] = identifier
        item["review_history"] = [dict(event) for event in events]
        item["review"] = dict(events[-1]) if events else {
            "decision": "unreviewed", "note": "", "id": None, "created_at": None,
        }
        item["semantic_relevance_verified"] = False
    counts = {
        decision: sum(item["review"]["decision"] == decision for item in reviewed.items)
        for decision in ("accepted", "rejected", "unreviewed")
    }
    reviewed.answer = render_inventory(reviewed.items)
    for document in reviewed.coverage["documents"]:
        candidates = [
            item for item in reviewed.items if item["document_id"] == document["document_id"]
        ]
        if not candidates:
            message = (
                f"No matching items were identified in the reviewed extracted text for document "
                f"{document['document_id']}; this is not proof of absence from the original."
            )
        elif all(item["review"]["decision"] == "rejected" for item in candidates):
            message = (
                "All generated candidates for document "
                f"{document['document_id']} were rejected by the reviewer; "
                "this is not proof of absence from the original."
            )
        else:
            continue
        reviewed.answer += ("\n" if reviewed.answer else "") + message
    reviewed.coverage.update({
        "inventory_question": question,
        "review_protocol": "human-review-v1",
        "review_counts": counts,
        "semantic_relevance_verified": False,
        "human_review_is_not_exhaustive": True,
    })
    return reviewed


def persist_inventory(
    library: DocumentLibrary, question: str, result: AnswerResult, max_chars: int,
) -> AnswerResult:
    identifiers = [document["document_id"] for document in result.coverage["documents"]]
    with closing(library.connect()) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        prepare(connection)
        check_current(connection, result)
        archive_candidates(connection, question, result)
        reviewed = apply_reviews(connection, question, result)
        payload = json.dumps(asdict(reviewed), ensure_ascii=False)
        if len(payload) > max_chars:
            raise DocumentQueryError(
                "Inventory records and review history exceed the output budget. Narrow the request.")
        connection.execute(
            "INSERT OR REPLACE INTO inventory_review_runs(scope,result) VALUES (?,?)",
            (scope_key(question, identifiers), payload),
        )
    LOGGER.info("Inventory candidates saved items=%d", len(reviewed.items))
    return reviewed


def refresh_inventory(
    library: DocumentLibrary, question: str, result: AnswerResult,
) -> AnswerResult:
    """Refresh decisions without replacing a chat result with a newer model run."""
    with closing(library.connect()) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        prepare(connection)
        check_current(connection, result)
        return apply_reviews(connection, question, result)


def review_inventory(
    library: DocumentLibrary, question: str, document_ids: list[str] | None, *,
    item_id: str | None = None, decision: str | None = None, note: str = "",
) -> AnswerResult:
    question = validate_question(question, 6)
    if not document_ids:
        raise DocumentQueryError("Select documents explicitly with --document for inventory review.")
    if (decision not in (None, "accepted", "rejected", "unreviewed")
            or bool(item_id) != (decision is not None) or (note and decision is None)):
        raise DocumentQueryError("A review decision requires --item and --decision together.")
    if not isinstance(note, str) or len(note) > 2000:
        raise DocumentQueryError("Review notes must be text of at most 2000 characters.")
    with closing(library.connect()) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        prepare(connection)
        row = connection.execute(
            "SELECT result FROM inventory_review_runs WHERE scope=?",
            (scope_key(question, document_ids),),
        ).fetchone()
        if row is None:
            raise DocumentQueryError(
                "No saved inventory for this exact question and document scope. Enumerate first.")
        saved = json.loads(row["result"])
        result = AnswerResult(**saved)
        check_current(connection, result)
        archive_candidates(connection, question, result)
        if decision is not None:
            if item_id not in {candidate_key(question, item) for item in result.items}:
                raise DocumentQueryError(
                    "Review item does not belong to the current saved inventory. List candidates again.")
            connection.execute(
                "INSERT INTO inventory_review_events(candidate,decision,note) VALUES (?,?,?)",
                (item_id, decision, note),
            )
        reviewed = apply_reviews(connection, question, result)
    LOGGER.info("Inventory review completed changed=%s", decision is not None)
    return reviewed
