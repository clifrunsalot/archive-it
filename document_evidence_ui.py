"""Read-only inspection of the most recent workspace result."""

import json
import textwrap
from pathlib import Path

from document_fact_ui import prompt
from document_library import DocumentLibrary, DocumentQueryError
from document_answers import cited_labels
from document_search import AnswerResult


def render_search_results(result: AnswerResult) -> str:
    lines = [
        f"Search results: {len(result.evidence)} passage(s), in retrieval order.",
        "Candidates only, not a generated answer or verified support; not guaranteed exhaustive.",
    ]
    if not result.evidence:
        lines.append("No matching passages returned.")
    for record in result.evidence:
        location = ", ".join(
            f"{key.replace('_', ' ')} {value}"
            for key, value in record["location"].items()
            if key not in ("char_start", "char_end")
        )
        lines.extend([
            "",
            f"[{record['label']}] {Path(record['source_path']).name} | "
            f"version {record['version_id']} | segment {record['segment']}"
            + (f" | {location}" if location else ""),
        ])
        for paragraph in record["text"].splitlines():
            lines.extend(textwrap.wrap(
                paragraph, width=88, initial_indent="    ", subsequent_indent="    ",
                break_long_words=False, break_on_hyphens=False,
            ) or [""])
        quality = record["quality"]
        if quality["status"] != "extracted" or quality["issues"]:
            lines.append(f"    Extraction: {quality['status']}; "
                         f"issues: {', '.join(quality['issues']) or 'none reported'}")
        if record["latest_attempt_status"] == "failed":
            lines.append("    Warning: latest ingestion failed; this is the retained successful version.")
    lines.extend(["", "Coverage: " + json.dumps(result.coverage)])
    if result.evidence:
        lines.append("Use inspect, then a passage number, for exact text, full provenance and nearby context.")
    return "\n".join(lines)


def inspect_evidence(library: DocumentLibrary, result: AnswerResult) -> None:
    if not result.evidence:
        raise DocumentQueryError("No evidence to inspect. Complete a search, answer or summary first.")
    records = {record["label"]: record for record in result.evidence}
    cited = cited_labels(result.answer)
    print("Model interpretation / search output (not verified source truth):")
    print(result.answer)
    for label, record in records.items():
        print(f"[{label}] {record['source_path']} | version={record['version_id']} | "
              f"{'referenced in output' if label in cited else 'supplied evidence; not cited'}")
    while True:
        value = prompt("Citation number (Enter returns)")
        if not value:
            return
        if not value.isascii() or not value.isdecimal() or len(value) > 6 or int(value) not in records:
            print("Choose a displayed citation number.")
            continue
        record = records[int(value)]
        print(f"\nOriginal supplied passage [{record['label']}] (not model-generated):")
        print("Source:", record["source_path"])
        print("Version:", record["version_id"], "| Segment:", record["segment"])
        print("Location:", json.dumps(record["location"], ensure_ascii=False))
        print("Extraction:", json.dumps(record["quality"], ensure_ascii=False))
        print("Latest attempt at query time:", record["latest_attempt_status"])
        print(record["text"])
        current = next((entry for entry in library.list_documents()
                        if entry["id"] == record["document_id"]), None)
        snapshot = (library.snapshots([record["document_id"]])[0]
                    if current is not None and current["current_version"] is not None else None)
        if snapshot is None or snapshot.version_id != record["version_id"]:
            print("Source version is no longer current. Showing retained query evidence only; "
                  "nearby navigation unavailable. Ask again for current evidence.")
            continue
        if snapshot.latest_attempt_status != record["latest_attempt_status"]:
            print("Latest ingestion status changed since this result:",
                  snapshot.latest_attempt_status)
        segments = snapshot.extraction["segments"]
        original = record["segment"] - 1
        position = original
        while True:
            action = prompt("Context: segment / previous / next (Enter returns)")
            if not action:
                break
            if action not in ("segment", "previous", "next"):
                print("Choose segment, previous or next.")
                continue
            target = original if action == "segment" else position + (-1 if action == "previous" else 1)
            if not 0 <= target < len(segments):
                print("No further source segment in that direction.")
                continue
            position = target
            segment = segments[position]
            print("Additional source context — NOT evidence used for this answer. "
                  "No new model call or verification.")
            print(f"Version {snapshot.version_id} | Segment {position + 1} | "
                  f"Location: {json.dumps(segment['location'], ensure_ascii=False)}")
            print(segment["text"])
