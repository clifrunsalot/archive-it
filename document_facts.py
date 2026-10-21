"""Human-authored typed facts, revision-bound confirmation, and exact-value queries."""

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from datetime import date
from decimal import Decimal, localcontext
from uuid import uuid4

from app_logging import LOGGER
from document_library import DocumentLibrary, DocumentQueryError

KINDS = ("date", "money", "quantity", "text")
DECISIONS = ("confirmed", "rejected", "unreviewed")
PERIODS = ("once", "daily", "weekly", "monthly", "yearly")
MAX_PAYLOAD_CHARS = 32000


def prepare(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS facts (
            id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id)
        );
        CREATE TABLE IF NOT EXISTS fact_revisions (
            fact_id TEXT NOT NULL REFERENCES facts(id), revision INTEGER NOT NULL,
            payload TEXT NOT NULL, ingestion_status TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            PRIMARY KEY(fact_id,revision)
        );
        CREATE TABLE IF NOT EXISTS fact_review_events (
            id INTEGER PRIMARY KEY, fact_id TEXT NOT NULL, revision INTEGER NOT NULL,
            decision TEXT NOT NULL CHECK(decision IN ('confirmed','rejected','unreviewed')),
            note TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            FOREIGN KEY(fact_id,revision) REFERENCES fact_revisions(fact_id,revision)
        );
        CREATE INDEX IF NOT EXISTS fact_review_revision
        ON fact_review_events(fact_id,revision,id);
    """)


def source_row(connection: sqlite3.Connection, document_id: str) -> sqlite3.Row:
    row = connection.execute("""
        SELECT d.current_version,d.source_path,v.extraction,
               (SELECT status FROM attempts WHERE document_id=d.id ORDER BY id DESC LIMIT 1)
               AS ingestion_status
        FROM documents d LEFT JOIN versions v ON v.id=d.current_version WHERE d.id=?
    """, (document_id,)).fetchone()
    if row is None or row["current_version"] is None:
        raise DocumentQueryError("Fact document has no current extracted version.")
    return row


def validate_payload(payload: object, source: sqlite3.Row) -> dict:
    fields = {
        "document_id", "version_id", "name", "kind", "value", "unit", "currency",
        "period", "conditions", "uncertainty", "support", "candidate_id",
    }
    if not isinstance(payload, dict) or set(payload) != fields:
        raise DocumentQueryError("Fact payload fields do not match the documented schema.")
    for field in ("document_id", "name", "kind", "value", "unit", "currency",
                  "period", "conditions", "uncertainty", "candidate_id"):
        value = payload[field]
        limit = 2000 if field in ("conditions", "uncertainty") else 350
        if not isinstance(value, str) or len(value) > limit or value != value.strip():
            raise DocumentQueryError("Fact text fields must be bounded strings without outer whitespace.")
    if not payload["name"] or not payload["value"] or payload["kind"] not in KINDS:
        raise DocumentQueryError("Fact requires a name, value, and supported kind.")
    if (type(payload["version_id"]) is not int
            or payload["version_id"] != source["current_version"]):
        raise DocumentQueryError("Fact source version is stale. Inspect the current extraction first.")
    kind, value = payload["kind"], payload["value"]
    if kind == "date":
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise DocumentQueryError("Date facts require an unambiguous ISO YYYY-MM-DD value.")
        try:
            date.fromisoformat(value)
        except ValueError as error:
            raise DocumentQueryError("Date fact is not a valid calendar date.") from error
    if kind in ("money", "quantity"):
        if not re.fullmatch(r"-?(?:0|[1-9]\d{0,23})(?:\.\d{1,6})?", value):
            raise DocumentQueryError(
                "Numeric facts require decimal strings: at most 24 integer and 6 fractional digits.")
    if kind == "money":
        if (not re.fullmatch(r"[A-Z]{3}", payload["currency"]) or payload["unit"]
                or payload["period"] not in PERIODS):
            raise DocumentQueryError("Money facts require a currency code and period, with no unit.")
    elif kind == "quantity":
        if not payload["unit"] or payload["currency"] or payload["period"]:
            raise DocumentQueryError("Quantity facts require an explicit unit, with no currency or period.")
    elif payload["unit"] or payload["currency"] or payload["period"]:
        raise DocumentQueryError("Date and text facts cannot have units, currency or periods.")
    supports = payload["support"]
    if not isinstance(supports, list) or not 1 <= len(supports) <= 16:
        raise DocumentQueryError("Facts require 1-16 exact supporting source spans.")
    segments = json.loads(source["extraction"])["segments"]
    verified = []
    for span in supports:
        if not isinstance(span, dict) or set(span) != {"segment", "char_start", "char_end", "quote"}:
            raise DocumentQueryError("Fact support requires segment, char_start, char_end and quote.")
        segment, start, end, quote = (span[key] for key in
                                      ("segment", "char_start", "char_end", "quote"))
        if (any(type(number) is not int for number in (segment, start, end))
                or not 1 <= segment <= len(segments)):
            raise DocumentQueryError("Fact support segment or offsets are invalid.")
        text = segments[segment - 1]["text"]
        if (not 0 <= start < end <= len(text) or not isinstance(quote, str)
                or not quote.strip() or len(quote) > 4000 or text[start:end] != quote):
            raise DocumentQueryError("Fact supporting quote does not match its exact source offsets.")
        if span in verified:
            raise DocumentQueryError("Fact contains duplicate supporting spans.")
        verified.append(span)
    if len(json.dumps(payload, ensure_ascii=False)) > MAX_PAYLOAD_CHARS:
        raise DocumentQueryError("Fact payload exceeds its size limit.")
    return payload


def current_revision(connection: sqlite3.Connection, fact_id: str) -> sqlite3.Row:
    row = connection.execute("""
        SELECT r.* FROM fact_revisions r WHERE fact_id=? ORDER BY revision DESC LIMIT 1
    """, (fact_id,)).fetchone()
    if row is None:
        raise DocumentQueryError("Fact ID does not exist.")
    return row


def fact_record(connection: sqlite3.Connection, row: sqlite3.Row) -> dict:
    payload = json.loads(row["payload"])
    source = source_row(connection, payload["document_id"])
    stale = (payload["version_id"], row["ingestion_status"]) != (
        source["current_version"], source["ingestion_status"],
    )
    events = connection.execute("""
        SELECT id,decision,note,created_at FROM fact_review_events
        WHERE fact_id=? AND revision=? ORDER BY id
    """, (row["fact_id"], row["revision"])).fetchall()
    decision = events[-1]["decision"] if events else "unreviewed"
    latest = connection.execute(
        "SELECT MAX(revision) FROM fact_revisions WHERE fact_id=?", (row["fact_id"],),
    ).fetchone()[0]
    superseded = row["revision"] != latest
    extraction = source["extraction"]
    if payload["version_id"] != source["current_version"]:
        original = connection.execute(
            "SELECT extraction FROM versions WHERE id=? AND document_id=?",
            (payload["version_id"], payload["document_id"]),
        ).fetchone()
        if original is None:
            raise DocumentQueryError("Fact source version is unavailable; no result published.")
        extraction = original["extraction"]
    extracted = json.loads(extraction)
    segments = extracted["segments"]
    evidence = []
    for span in payload["support"]:
        segment = segments[span["segment"] - 1]
        location = dict(segment["location"])
        location.update(char_start=span["char_start"], char_end=span["char_end"])
        if "line_start" in location:
            location["line_start"] += segment["text"][:span["char_start"]].count("\n")
            location["line_end"] = location["line_start"] + span["quote"].rstrip("\n").count("\n")
        evidence.append({
            "document_id": payload["document_id"], "version_id": payload["version_id"],
            "segment": span["segment"], "source_path": source["source_path"],
            "location": location, "quote": span["quote"],
        })
    return {
        "fact_id": row["fact_id"], "revision": row["revision"], "payload": payload,
        "created_at": row["created_at"], "decision": decision, "stale": stale,
        "superseded": superseded,
        "usable_for_queries": not stale and not superseded and decision == "confirmed",
        "review_history": [dict(event) for event in events],
        "quote_provenance_verified": True, "semantic_relevance_verified": False,
        "origin": "human_authored", "source_ingestion_status": row["ingestion_status"],
        "evidence": evidence,
        "quality": {key: extracted[key] for key in ("status", "issues", "ocr_used")},
    }


def save_fact(
    library: DocumentLibrary, payload: object, *, fact_id: str | None = None,
    expected_revision: int | None = None,
) -> dict:
    if not isinstance(payload, dict) or not isinstance(payload.get("document_id"), str):
        raise DocumentQueryError("Fact payload requires a document_id.")
    if (fact_id is None) != (expected_revision is None):
        raise DocumentQueryError("Fact corrections require both --fact and --revision.")
    with closing(library.connect()) as connection:
        prepare(connection)
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            source = source_row(connection, payload["document_id"])
            validated = validate_payload(payload, source)
            if validated["candidate_id"]:
                table = connection.execute("""
                    SELECT name FROM sqlite_master WHERE type='table'
                    AND name='inventory_review_candidates'
                """).fetchone()
                candidate = connection.execute(
                    "SELECT item FROM inventory_review_candidates WHERE id=?",
                    (validated["candidate_id"],),
                ).fetchone() if table else None
                if candidate is None:
                    raise DocumentQueryError("Linked inventory candidate does not exist.")
                original = json.loads(candidate["item"])
                if (original["document_id"], original["version_id"]) != (
                    validated["document_id"], validated["version_id"],
                ):
                    raise DocumentQueryError("Linked candidate belongs to another document or version.")
            revision = 1
            if fact_id is not None:
                current = current_revision(connection, fact_id)
                if type(expected_revision) is not int or current["revision"] != expected_revision:
                    raise DocumentQueryError("Fact revision changed. Inspect it before correcting.")
                if json.loads(current["payload"])["document_id"] != validated["document_id"]:
                    raise DocumentQueryError("Fact corrections cannot move to another document.")
                revision = current["revision"] + 1
            else:
                fact_id = uuid4().hex
                connection.execute(
                    "INSERT INTO facts(id,document_id) VALUES (?,?)",
                    (fact_id, validated["document_id"]),
                )
            connection.execute("""
                INSERT INTO fact_revisions(fact_id,revision,payload,ingestion_status) VALUES (?,?,?,?)
            """, (fact_id, revision, json.dumps(validated, ensure_ascii=False), source["ingestion_status"]))
            result = fact_record(connection, current_revision(connection, fact_id))
    LOGGER.info("Typed fact saved revision=%d", revision)
    return result


def review_fact(
    library: DocumentLibrary, fact_id: str, revision: int, decision: str, note: str = "",
) -> dict:
    if decision not in DECISIONS or type(revision) is not int:
        raise DocumentQueryError("Fact review requires a revision and supported decision.")
    if not isinstance(note, str) or len(note) > 2000:
        raise DocumentQueryError("Fact review note must be text of at most 2000 characters.")
    with closing(library.connect()) as connection:
        prepare(connection)
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            row = current_revision(connection, fact_id)
            if row["revision"] != revision:
                raise DocumentQueryError("Fact revision changed. Inspect it before review.")
            result = fact_record(connection, row)
            if result["stale"]:
                raise DocumentQueryError("Fact source changed. Create a corrected revision before review.")
            connection.execute("""
                INSERT INTO fact_review_events(fact_id,revision,decision,note) VALUES (?,?,?,?)
            """, (fact_id, revision, decision, note))
            result = fact_record(connection, row)
    LOGGER.info("Typed fact human review recorded")
    return result


def list_facts(library: DocumentLibrary, document_ids: list[str] | None) -> list[dict]:
    if not document_ids:
        raise DocumentQueryError("Select documents explicitly with --document for facts.")
    library.snapshots(document_ids)
    with closing(library.connect()) as connection:
        prepare(connection)
        with connection:
            connection.execute("BEGIN")
            placeholders = ",".join("?" for _ in document_ids)
            rows = connection.execute(f"""
                SELECT r.* FROM fact_revisions r JOIN facts f ON f.id=r.fact_id
                WHERE f.document_id IN ({placeholders})
                AND r.revision=(SELECT MAX(revision) FROM fact_revisions WHERE fact_id=f.id)
                ORDER BY r.fact_id
            """, document_ids).fetchall()
            return [fact_record(connection, row) for row in rows]


def fact_history(library: DocumentLibrary, fact_id: str) -> list[dict]:
    with closing(library.connect()) as connection:
        prepare(connection)
        with connection:
            connection.execute("BEGIN")
            current_revision(connection, fact_id)
            rows = connection.execute(
                "SELECT * FROM fact_revisions WHERE fact_id=? ORDER BY revision", (fact_id,),
            ).fetchall()
            return [fact_record(connection, row) for row in rows]


def load_query_records(library: DocumentLibrary, fact_ids: list[str], operation: str) -> list[dict]:
    if (not fact_ids or len(fact_ids) > 128 or len(set(fact_ids)) != len(fact_ids)
            or operation not in ("count", "sort", "compare", "sum")):
        raise DocumentQueryError("Fact query requires 1-128 distinct explicit IDs and a supported operation.")
    with closing(library.connect()) as connection:
        prepare(connection)
        with connection:
            connection.execute("BEGIN")
            records = [fact_record(connection, current_revision(connection, identifier))
                       for identifier in fact_ids]
    return records


def plan_fact_query(
    records: list[dict], operation: str, lower: str | None, upper: str | None,
) -> dict:
    if any(not record["usable_for_queries"] for record in records):
        raise DocumentQueryError("Every selected fact must be current and human-confirmed; none were skipped.")
    if any(record["payload"]["uncertainty"] for record in records):
        raise DocumentQueryError("Selected facts have unresolved uncertainty. Correct them before querying.")
    filtered = records
    if lower is not None or upper is not None:
        if operation not in ("count", "sort"):
            raise DocumentQueryError("Date bounds are supported only for count and sort.")
        if any(record["payload"]["kind"] != "date" for record in records):
            raise DocumentQueryError("Date bounds require only date facts.")
        for bound in (lower, upper):
            if bound is not None:
                try:
                    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", bound):
                        raise ValueError
                    date.fromisoformat(bound)
                except ValueError as error:
                    raise DocumentQueryError("Date bounds require valid ISO YYYY-MM-DD dates.") from error
        if lower is not None and upper is not None and lower > upper:
            raise DocumentQueryError("Date lower bound cannot exceed upper bound.")
        filtered = [
            record for record in records
            if (lower is None or record["payload"]["value"] >= lower)
            and (upper is None or record["payload"]["value"] <= upper)
        ]
    result = {
        "operation": operation, "scope": "explicitly selected human-confirmed records",
        "selected_fact_ids": [record["fact_id"] for record in records], "records": filtered,
        "exhaustive": False, "document_total": False,
        "date_bounds": {"lower": lower, "upper": upper},
        "source_review_required": True,
        "selection_is_inclusion_rule": True,
        "no_unit_currency_or_period_conversion": True,
        "checks": [
            "All selected facts are current and human-confirmed.",
            "No selected fact has unresolved uncertainty.",
            "Date bounds, if supplied, are valid and applicable.",
        ],
    }
    if operation == "count":
        return result
    compatible = {
        tuple(record["payload"][key] for key in ("kind", "unit", "currency", "period"))
        for record in records
    }
    if len(compatible) != 1:
        raise DocumentQueryError("Fact kinds, units, currency and periods must match; no conversions are inferred.")
    kind, unit, currency, period = next(iter(compatible))
    result["checks"].append("Selected kinds, units, currency and periods match.")
    if operation == "sum":
        if kind not in ("money", "quantity"):
            raise DocumentQueryError("Only compatible money or quantity facts can be summed.")
        supports = []
        for record in records:
            payload = record["payload"]
            spans = {
                (payload["document_id"], payload["version_id"], span["segment"],
                 span["char_start"], span["char_end"]) for span in payload["support"]
            }
            if any(
                left[:3] == right[:3] and left[3] < right[4] and right[3] < left[4]
                for left in supports for right in spans
            ):
                raise DocumentQueryError("Selected sum records share source spans; resolve possible double-counting.")
            supports.extend(spans)
        if any(record["payload"]["conditions"] != records[0]["payload"]["conditions"]
               for record in records):
            raise DocumentQueryError("Sum records have differing conditions; no inclusion rules are inferred.")
        result.update(
            unit=unit, currency=currency, period=period,
            conditions=records[0]["payload"]["conditions"],
        )
        result["checks"].extend([
            "Only money or quantity values are selected.",
            "Selected source spans do not overlap across records.",
            "Selected records have identical conditions.",
        ])
    elif operation == "compare" and len(records) != 2:
        raise DocumentQueryError("Comparison requires exactly two selected facts.")
    return result


def query_token(records: list[dict], operation: str, lower: str | None, upper: str | None) -> str:
    return hashlib.sha256(json.dumps(
        {"protocol": "fact-preview-v1", "records": records,
         "operation": operation, "lower": lower, "upper": upper},
        sort_keys=True, ensure_ascii=False,
    ).encode()).hexdigest()


def preview_fact_query(
    library: DocumentLibrary, fact_ids: list[str], operation: str, *,
    lower: str | None = None, upper: str | None = None,
) -> dict:
    records = load_query_records(library, fact_ids, operation)
    blocking_reasons = []
    for record in records:
        if record["stale"]:
            blocking_reasons.append({"fact_id": record["fact_id"], "reason": "Source is stale."})
        if record["decision"] != "confirmed":
            blocking_reasons.append({
                "fact_id": record["fact_id"], "reason": "Record is not human-confirmed.",
            })
        if record["payload"]["uncertainty"]:
            blocking_reasons.append({
                "fact_id": record["fact_id"], "reason": "Record has unresolved uncertainty.",
            })
    try:
        plan = plan_fact_query(records, operation, lower, upper)
    except DocumentQueryError as error:
        blocking_reasons.append({"fact_id": None, "reason": str(error)})
        plan = None
    included = {record["fact_id"] for record in plan["records"]} if plan is not None else set()
    return {
        "preview_protocol": "fact-preview-v1", "operation": operation,
        "ready": plan is not None, "calculation_performed": False,
        "scope": "explicitly selected human-confirmed records",
        "selected_fact_ids": fact_ids, "selected_records": records,
        "included_fact_ids": [record["fact_id"] for record in records if record["fact_id"] in included],
        "exclusions": [
            {"fact_id": record["fact_id"], "reason": "Outside inclusive date bounds."}
            for record in records if plan is not None and record["fact_id"] not in included
        ],
        "blocking_reasons": blocking_reasons,
        "date_bounds": {"lower": lower, "upper": upper},
        "preview_token": query_token(records, operation, lower, upper),
        "exhaustive": False, "document_total": False,
        "no_unit_currency_or_period_conversion": True,
        "compatibility": "passed" if plan is not None else "blocked; no records silently skipped",
        "passed_checks": plan["checks"] if plan is not None else [],
    }


def query_facts(
    library: DocumentLibrary, fact_ids: list[str], operation: str, *,
    lower: str | None = None, upper: str | None = None,
    expected_preview: str | None = None,
) -> dict:
    records = load_query_records(library, fact_ids, operation)
    if expected_preview is not None and query_token(records, operation, lower, upper) != expected_preview:
        raise DocumentQueryError(
            "Facts, source status, review decisions or query selection changed since preview. Preview again.")
    result = plan_fact_query(records, operation, lower, upper)
    filtered = result["records"]
    kind = records[0]["payload"]["kind"]
    if operation == "count":
        result["value"] = len(filtered)
    elif operation == "sum":
        with localcontext() as context:
            context.prec = 64
            total = sum((Decimal(record["payload"]["value"]) for record in records), Decimal(0))
        result["value"] = format(total, "f")
    else:
        result["records"] = sorted(
            filtered, key=lambda record: (
                Decimal(record["payload"]["value"]) if kind in ("money", "quantity")
                else record["payload"]["value"], record["fact_id"],
            ),
        )
        result["ordering"] = "ascending by value; ties by fact ID"
        if operation == "compare":
            values = [
                Decimal(record["payload"]["value"]) if kind in ("money", "quantity")
                else record["payload"]["value"] for record in records
            ]
            result["relation"] = "equal" if values[0] == values[1] else (
                "less" if values[0] < values[1] else "greater")
    return result
