"""Interactive local fact review using the same validated, revision-bound APIs."""

import json
import re
import sys
from copy import deepcopy

from app_logging import LOGGER, log_failure
from document_facts import (
    KINDS, PERIODS, list_facts, preview_fact_query, query_facts, review_fact, save_fact,
)
from document_inventory_review import review_inventory
from document_library import DocumentLibrary, DocumentQueryError, DocumentSnapshot


class CancelDraft(Exception):
    """An explicit cancellation before the next write."""


def prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    if value == "/cancel":
        raise CancelDraft
    return default if not value else ("" if value == "/clear" else value)


def choose(label: str, options: tuple[str, ...], default: str = "") -> str:
    while True:
        value = prompt(f"{label} ({'/'.join(options)})", default)
        if value in options:
            return value
        print("Choose one of the listed options.")


def select_number(label: str, count: int) -> int:
    while True:
        value = prompt(label)
        if (len(value) <= len(str(count)) and value.isascii() and value.isdecimal()
                and 1 <= int(value) <= count):
            return int(value)
        print(f"Enter a number from 1 to {count}.")


def select_numbers(count: int) -> list[int]:
    while True:
        selection = prompt("Select 1-128 fact numbers in order (commas or ranges, e.g. 1,3-5)")
        numbers = []
        valid = len(selection) <= 1024
        for part in selection.split(",") if valid else []:
            match = re.fullmatch(r"([0-9]{1,9})(?:\s*-\s*([0-9]{1,9}))?", part.strip())
            if match is None:
                valid = False
                break
            first = int(match[1])
            last = int(match[2]) if match[2] else first
            if not 1 <= first <= last <= count or len(numbers) + last - first + 1 > 128:
                valid = False
                break
            numbers.extend(range(first, last + 1))
        if valid and numbers and len(numbers) == len(set(numbers)):
            return numbers
        print("Select distinct numbers from the displayed facts; use ascending ranges. "
              "No records were chosen automatically.")


def select_support(snapshot: DocumentSnapshot) -> list[dict]:
    segments = snapshot.extraction["segments"]
    if not segments:
        raise DocumentQueryError("Selected document has no extracted segments to use as evidence.")
    for number, segment in enumerate(segments, 1):
        snippet = " ".join(segment["text"].split())[:100]
        print(f"{number}. {json.dumps(segment['location'])} {snippet}")
    support = []
    while True:
        number = select_number("Source segment", len(segments))
        segment = segments[number - 1]
        text = segment["text"]
        print(f"\nSegment {number}, location {json.dumps(segment['location'])}\n{text}")
        while True:
            quote = prompt("Exact substring (Enter selects whole segment; preserves its line breaks)")
            quote = quote or text
            starts = []
            start = text.find(quote)
            while start >= 0:
                starts.append(start)
                start = text.find(quote, start + 1)
            if not quote.strip() or len(quote) > 4000 or not starts:
                print("Select an exact substring of at most 4000 characters from this segment.")
                continue
            if len(starts) > 1:
                for index, offset in enumerate(starts, 1):
                    context = text[max(0, offset - 40):offset + len(quote) + 40]
                    print(f"{index}. offset {offset}: {json.dumps(context, ensure_ascii=False)}")
                start = starts[select_number("Quote occurrence", len(starts)) - 1]
            else:
                start = starts[0]
            span = {
                "segment": number, "char_start": start, "char_end": start + len(quote),
                "quote": quote,
            }
            if span in support:
                print("That exact span has already been selected.")
                continue
            support.append(span)
            break
        if len(support) == 16 or choose("Add another source span?", ("yes", "no"), "no") == "no":
            return support


def show_evidence(snapshot: DocumentSnapshot, support: list[dict]) -> None:
    print(f"Source: {snapshot.source_path}; document={snapshot.document_id}; version={snapshot.version_id}")
    print("Extraction:", json.dumps({
        key: snapshot.extraction[key] for key in ("status", "issues", "ocr_used")
    }))
    for span in support:
        segment = snapshot.extraction["segments"][span["segment"] - 1]
        print(f"Segment {span['segment']}, location={json.dumps(segment['location'])}, "
              f"characters [{span['char_start']}, {span['char_end']})")
        print("Quote:", span["quote"])
        print("Source context:", segment["text"])


def edit_fields(draft: dict) -> None:
    draft["name"] = prompt("Fact name", draft["name"])
    draft["kind"] = choose("Kind", KINDS, draft["kind"])
    draft["value"] = prompt("Value (ISO date or decimal string for typed values)", draft["value"])
    if draft["kind"] == "money":
        draft["currency"] = prompt("Currency (three uppercase letters)", draft["currency"])
        draft["period"] = choose("Period", PERIODS, draft["period"])
        draft["unit"] = ""
    elif draft["kind"] == "quantity":
        draft["unit"] = prompt("Unit", draft["unit"])
        draft["currency"] = draft["period"] = ""
    else:
        draft["unit"] = draft["currency"] = draft["period"] = ""
    draft["conditions"] = prompt("Conditions (/clear removes)", draft["conditions"])
    draft["uncertainty"] = prompt("Uncertainty (/clear removes)", draft["uncertainty"])


def review_saved(library: DocumentLibrary, saved: dict) -> None:
    print("Saved record:", json.dumps(saved, ensure_ascii=False, indent=2))
    print("Confirmation is a human judgment; it does not establish document completeness.")
    decision = choose("Review decision", ("confirm", "reject", "leave"), "leave")
    if decision != "leave":
        note = prompt("Private review note (optional)")
        reviewed = review_fact(
            library, saved["fact_id"], saved["revision"],
            "confirmed" if decision == "confirm" else "rejected", note,
        )
        print(f"Revision {reviewed['revision']} is now {reviewed['decision']}.")


def edit_fact(
    library: DocumentLibrary, document_id: str, *, candidate: dict | None = None,
    existing: dict | None = None, metadata: dict | None = None,
) -> None:
    snapshot = library.snapshots([document_id])[0]
    draft = deepcopy(existing["payload"]) if existing else {
        "document_id": document_id, "version_id": snapshot.version_id,
        "name": "", "kind": "", "value": "", "currency": "", "unit": "", "period": "",
        "conditions": "", "uncertainty": "", "support": [], "candidate_id": "",
    }
    if metadata is not None:
        if existing is not None or candidate is not None:
            raise DocumentQueryError("Metadata reuse is only supported for a new human-authored fact.")
        draft.update({key: metadata[key] for key in ("kind", "unit", "currency", "period")})
        print("Reusing kind/unit/currency/period only; verify these defaults. "
              "Name, value, evidence, conditions, uncertainty and confirmation are not copied.")
    if candidate is not None:
        draft.update(
            name=candidate["name"], conditions=candidate["conditions"],
            uncertainty=candidate["uncertainty"], candidate_id=candidate["review_id"],
            support=[{
                "segment": span["segment"], "char_start": span["location"]["char_start"],
                "char_end": span["location"]["char_end"], "quote": span["quote"],
            } for span in candidate["support"]],
        )
        # A quote may have several possible occurrences; keep them all unless replaced explicitly.
        draft["support"] = list({json.dumps(span, sort_keys=True): span
                                 for span in draft["support"]}.values())
    if existing:
        print("Existing revision (corrections never inherit confirmation):")
        print(json.dumps(existing, ensure_ascii=False, indent=2))
    if draft["version_id"] != snapshot.version_id:
        print("Source version changed. Select fresh evidence; the old candidate link is cleared.")
        draft.update(version_id=snapshot.version_id, support=[], candidate_id="")
    if draft["support"]:
        show_evidence(snapshot, draft["support"])
        if choose("Keep source spans?", ("yes", "no"), "yes") == "no":
            draft["support"] = []
    if not draft["support"]:
        draft["support"] = select_support(snapshot)
    while True:
        edit_fields(draft)
        print("\nDraft preview (not saved or confirmed):")
        print(json.dumps(draft, ensure_ascii=False, indent=2))
        show_evidence(snapshot, draft["support"])
        action = choose("Next action", ("save", "edit", "evidence", "cancel"), "cancel")
        if action == "cancel":
            print("Draft cancelled; no fact was saved.")
            return
        if action == "edit":
            continue
        if action == "evidence":
            draft["support"] = select_support(snapshot)
            continue
        current = library.snapshots([document_id])[0]
        if (current.version_id, current.latest_attempt_status) != (
            snapshot.version_id, snapshot.latest_attempt_status,
        ):
            raise DocumentQueryError("Source changed during review. Restart with the current extraction.")
        try:
            saved = save_fact(
                library, draft, fact_id=existing["fact_id"] if existing else None,
                expected_revision=existing["revision"] if existing else None,
            )
        except DocumentQueryError as error:
            log_failure("Interactive fact validation", error)
            print(str(error), file=sys.stderr)
            if choose("Validation failed", ("edit", "cancel"), "cancel") == "edit":
                continue
            print("Draft cancelled; no fact was saved.")
            return
        print(f"Saved unreviewed fact {saved['fact_id']} revision {saved['revision']}.")
        review_saved(library, saved)
        return


def calculate_facts(library: DocumentLibrary, document_id: str) -> None:
    facts = list_facts(library, [document_id])
    if not facts:
        print("No facts yet. Add and confirm facts before calculating.")
        return
    for number, fact in enumerate(facts, 1):
        payload = fact["payload"]
        print(f"{number}. {payload['name']} = {payload['value']} "
              f"{payload['unit'] or payload['currency']} {payload['period']} | "
              f"{fact['decision']} stale={fact['stale']} | ID={fact['fact_id']}")
    identifiers = [facts[number - 1]["fact_id"] for number in select_numbers(len(facts))]
    operation = choose("Operation", ("count", "sort", "compare", "sum"))
    lower = upper = None
    if operation in ("count", "sort"):
        lower = prompt("Inclusive from-date (ISO YYYY-MM-DD; Enter for none)") or None
        upper = prompt("Inclusive through-date (ISO YYYY-MM-DD; Enter for none)") or None
    preview = preview_fact_query(library, identifiers, operation, lower=lower, upper=upper)
    print("\nCalculation preview (no calculation performed):")
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    unselected = [fact["fact_id"] for fact in facts if fact["fact_id"] not in identifiers]
    print("Not selected (not part of the query):", json.dumps(unselected))
    print("This concerns selected records, not a document-wide total or completeness guarantee.")
    if not preview["ready"]:
        print("Calculation blocked. Resolve the listed issues and preview again; nothing was skipped.")
        return
    if choose("Execute previewed query?", ("run", "cancel"), "cancel") != "run":
        print("Calculation cancelled; no result was computed.")
        return
    result = query_facts(
        library, identifiers, operation, lower=lower, upper=upper,
        expected_preview=preview["preview_token"],
    )
    print("Calculation result:", json.dumps(result, ensure_ascii=False, indent=2))


def review_pending(library: DocumentLibrary, document_id: str) -> None:
    facts = list_facts(library, [document_id])
    pending = [fact for fact in facts if fact["decision"] == "unreviewed" and not fact["stale"]]
    print(f"Review queue: {len(pending)} current unreviewed facts. "
          "Each requires its own decision; Enter leaves it unchanged.")
    stale = [fact for fact in facts if fact["stale"]]
    for fact in stale:
        print(f"Stale fact excluded from queue: {fact['payload']['name']} | ID={fact['fact_id']}. "
              "Edit with fresh evidence before review.")
    for number, fact in enumerate(pending, 1):
        print(f"\nReview {number}/{len(pending)}: {fact['payload']['name']}")
        review_saved(library, fact)
    print("Review queue finished. No calculations were performed.")


def run_fact_workbench(
    library: DocumentLibrary, document_id: str, question: str | None = None,
) -> None:
    library.snapshots([document_id])
    print("Local fact review; no model calls. /cancel cancels the current draft.")
    print("Enter keeps a displayed default; /clear empties a text field. Source text is untrusted data.")
    try:
        while True:
            try:
                action = choose(
                    "Action", ("add", "similar", "candidate", "edit", "review",
                               "review-pending", "calculate", "quit"), "quit",
                )
                if action == "quit":
                    return
                if action == "add":
                    edit_fact(library, document_id)
                elif action == "review-pending":
                    review_pending(library, document_id)
                elif action == "calculate":
                    calculate_facts(library, document_id)
                elif action == "candidate":
                    if question is None:
                        raise DocumentQueryError("Candidate browsing requires --question with the exact inventory question.")
                    inventory = review_inventory(library, question, [document_id])
                    candidates = inventory.items
                    if not candidates:
                        print("No saved candidates. Use add to record a missed fact.")
                        continue
                    for number, item in enumerate(candidates, 1):
                        print(f"{number}. {item['name']} | {item['review']['decision']} | "
                              f"{item['conditions']} | {item['uncertainty']}")
                    item = candidates[select_number("Candidate number", len(candidates)) - 1]
                    print("Candidate:", json.dumps(item, ensure_ascii=False, indent=2))
                    edit_fact(library, document_id, candidate=item)
                else:
                    facts = list_facts(library, [document_id])
                    if not facts:
                        print("No facts yet. Use add or candidate.")
                        continue
                    for number, fact in enumerate(facts, 1):
                        print(f"{number}. {fact['payload']['name']} | {fact['decision']} | "
                              f"revision={fact['revision']} stale={fact['stale']}")
                    fact = facts[select_number("Fact number", len(facts)) - 1]
                    if action == "similar":
                        edit_fact(library, document_id, metadata=fact["payload"])
                    elif action == "review":
                        review_saved(library, fact)
                    else:
                        edit_fact(library, document_id, existing=fact)
            except CancelDraft:
                print("Current action cancelled. Previously saved revisions and decisions are retained.")
            except DocumentQueryError as error:
                log_failure("Interactive fact review", error)
                print(str(error), file=sys.stderr)
    except (EOFError, KeyboardInterrupt):
        LOGGER.info("Interactive fact review interrupted")
        print("\nReview stopped. Explicitly saved revisions and decisions are retained.")
