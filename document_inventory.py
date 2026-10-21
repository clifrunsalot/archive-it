"""Quote-backed inventory records; provenance checks do not establish relevance."""

import json
import re
from copy import deepcopy

from document_answers import SynthesisValidationError

INVENTORY_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["findings", "no_relevant_evidence"]},
        "findings": {"type": "array", "maxItems": 16, "items": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "minLength": 1, "maxLength": 350},
                "conditions": {"type": "string", "maxLength": 350},
                "assessment": {"type": "string", "enum": ["established", "ambiguous"]},
                "uncertainty": {"type": "string", "maxLength": 350},
                "evidence": {"type": "array", "minItems": 1, "maxItems": 8, "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "integer"},
                        "quote": {"type": "string", "minLength": 1, "maxLength": 1400},
                    },
                    "required": ["label", "quote"], "additionalProperties": False,
                }},
            },
            "required": ["text", "conditions", "assessment", "uncertainty",
                         "evidence"],
            "additionalProperties": False,
        }},
    },
    "required": ["status", "findings"], "additionalProperties": False,
}


def inventory_schema(sources: dict[int, dict]) -> dict:
    """Constrain quote selection to source text, rather than free-form copying."""
    schema = deepcopy(INVENTORY_SCHEMA)
    properties = schema["properties"]["findings"]["items"]["properties"]
    choices = []
    for label, source in sources.items():
        text = source["text"]
        boundaries = [0, *(
            match.end() for match in re.finditer(r"(?<=[.!?])\s+|\n\s*\n", text)
        ), len(text)]
        quotes = list(dict.fromkeys([
            text, *(text[start:end] for start, end in zip(boundaries, boundaries[1:])),
        ]))
        choices.append({
            "properties": {
                "label": {"const": label},
                "quote": {"enum": [quote for quote in quotes if quote.strip()]},
            },
            "required": ["label", "quote"],
        })
    properties["evidence"]["items"]["anyOf"] = choices
    return schema


def parse_inventory(text: str, sources: dict[int, dict], max_chars: int) -> str:
    """Validate the complete batch before it can be displayed or checkpointed."""
    if len(text) > max_chars:
        raise SynthesisValidationError("output_size_limit")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise SynthesisValidationError("invalid_json") from error
    if not isinstance(value, dict) or set(value) != {"status", "findings"}:
        raise SynthesisValidationError("invalid_schema")
    status, findings = value["status"], value["findings"]
    if (status not in ("findings", "no_relevant_evidence")
            or not isinstance(findings, list) or len(findings) > 16):
        raise SynthesisValidationError("invalid_schema")
    if (status == "findings") != bool(findings):
        raise SynthesisValidationError("inconsistent_findings_status")
    for finding in findings:
        if not isinstance(finding, dict) or set(finding) != {
            "text", "conditions", "assessment", "uncertainty", "evidence",
        }:
            raise SynthesisValidationError("invalid_schema")
        for key in ("text", "conditions", "uncertainty"):
            body = finding[key]
            if (not isinstance(body, str) or len(body) > 350
                    or (key == "text" and not body.strip())
                    or re.search(r"[\r\n]|\[\d+\]", body)):
                raise SynthesisValidationError("invalid_finding_text")
        if (finding["assessment"] not in ("established", "ambiguous")
                or (finding["assessment"] == "ambiguous") != bool(finding["uncertainty"].strip())):
            raise SynthesisValidationError("invalid_uncertainty")
        evidence = finding["evidence"]
        if not isinstance(evidence, list) or not 1 <= len(evidence) <= 8:
            raise SynthesisValidationError("missing_supporting_quotes")
        supported = set()
        for support in evidence:
            if not isinstance(support, dict) or set(support) != {"label", "quote"}:
                raise SynthesisValidationError("invalid_supporting_quote")
            label, quote = support["label"], support["quote"]
            if type(label) is not int or label not in sources:
                raise SynthesisValidationError("citations_outside_batch")
            if not isinstance(quote, str) or not quote.strip() or len(quote) > 1400:
                raise SynthesisValidationError("invalid_supporting_quote")
            if quote not in sources[label]["text"]:
                raise SynthesisValidationError("supporting_quote_not_in_source")
            supported.add(label)
        identities = {
            (sources[label].get("document_id", sources[label].get("document")),
             sources[label].get("version_id")) for label in supported
        }
        if len(identities) > 1:
            raise SynthesisValidationError("inventory_item_crosses_documents")
    canonical = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(canonical) > max_chars:
        raise SynthesisValidationError("output_size_limit")
    return canonical


def filter_explicit_amenities(text: str, sources: dict[int, dict], max_chars: int) -> str:
    """Keep amenity candidates only when their quoted clause explicitly offers access."""
    value = json.loads(text)
    findings = value["findings"]
    group_pattern = re.compile(
        r"\ball or a part of these facilities have been provided as an amenity\b",
        re.IGNORECASE,
    )
    for label, source in sources.items():
        match = group_pattern.search(source["text"])
        if match is None:
            continue
        definitions = list(re.finditer(
            r"\b([A-Z][A-Za-z' -]{0,50}?)\s+Facility\s+means\b",
            source["text"][:match.start()],
            re.IGNORECASE,
        ))
        name = f"{definitions[-1].group(1).strip()} facilities" if definitions else "Facilities"
        sentence_start = max(source["text"].rfind(".", 0, match.start()),
                             source["text"].rfind("\n", 0, match.start())) + 1
        sentence_end = source["text"].find(".", match.end())
        quote = source["text"][sentence_start:len(source["text"]) if sentence_end < 0
                               else sentence_end + 1].strip()
        findings = [{
            "text": name, "conditions": "", "assessment": "ambiguous",
            "uncertainty": "Individual facilities unspecified by the source.",
            "evidence": [{"label": label, "quote": quote}],
        }]
        value["status"] = "findings"
        value["findings"] = findings
        return parse_inventory(json.dumps(value, ensure_ascii=False), sources, max_chars)

    offer = re.compile(
        r"\b(?:have|has|had)\s+access\s+to\b|"
        r"\b(?:have|has|had)\s+(?:[\w-]+\s+)?access\b|"
        r"\bavailable\s+to\s+(?:residents|tenants|occupants)\b|"
        r"\b(?:is|are)\s+available\b|"
        r"\bprovided\s+as\s+an?\s+amenit(?:y|ies)\b|"
        r"\bamenit(?:y|ies)\s+(?:include|includes|provided|available)\b",
        re.IGNORECASE,
    )
    retained = []
    for finding in findings:
        name = " ".join(finding["text"].casefold().split()).strip(" .:")
        explicit_offer = False
        for support in finding["evidence"]:
            quote = support["quote"]
            sentences = re.split(r"(?<=[.!?])\s+|\n+", quote)
            if any(
                re.search(rf"(?<!\w){re.escape(name)}(?!\w)", sentence, re.IGNORECASE)
                and offer.search(sentence)
                and not re.search(r"\b(?:may|might|could)\s+include\b|\bif applicable\b|\bprohibited\b",
                                  sentence, re.IGNORECASE)
                for sentence in sentences
            ):
                explicit_offer = True
                break
        if explicit_offer:
            retained.append(finding)
    value["status"] = "findings" if retained else "no_relevant_evidence"
    value["findings"] = retained
    return parse_inventory(json.dumps(value, ensure_ascii=False), sources, max_chars)


def consolidate_inventory(batches: list[str], sources: dict[int, dict]) -> list[dict]:
    """Merge exact item/condition matches, retaining every distinct supporting span."""
    items: dict[tuple, dict] = {}
    for batch in batches:
        for finding in json.loads(batch)["findings"]:
            labels = {support["label"] for support in finding["evidence"]}
            first = sources[min(labels)]
            identity = (first["document_id"], first["version_id"])
            if any((sources[label]["document_id"], sources[label]["version_id"]) != identity
                   for label in labels):
                raise SynthesisValidationError("inventory_item_crosses_documents")
            key = (*identity, *(
                " ".join(finding[field].casefold().strip().rstrip(".").split())
                for field in ("text", "conditions", "assessment", "uncertainty")
            ))
            if key not in items:
                items[key] = {
                    "name": finding["text"].strip(),
                    "conditions": finding["conditions"].strip(),
                    "assessment": finding["assessment"],
                    "uncertainty": finding["uncertainty"].strip(),
                    "document_id": identity[0], "version_id": identity[1],
                    "support": [],
                    "quote_provenance_verified": True, "semantic_relevance_verified": False,
                }
            item = items[key]
            for support in finding["evidence"]:
                source = sources[support["label"]]
                quote = support["quote"]
                start = source["text"].find(quote)
                # Repeated exact quotes retain all possible spans, never a guessed occurrence.
                while start >= 0:
                    location = dict(source["location"])
                    location["char_start"] = source["location"].get("char_start", 0) + start
                    location["char_end"] = location["char_start"] + len(quote)
                    if "line_start" in location:
                        location["line_start"] += source["text"][:start].count("\n")
                        location["line_end"] = location["line_start"] + quote.rstrip("\n").count("\n")
                    span = {
                        "label": support["label"], "quote": quote,
                        "document_id": source["document_id"], "version_id": source["version_id"],
                        "segment": source["segment"], "source_path": source["source_path"],
                        "location": location,
                    }
                    if span not in item["support"]:
                        item["support"].append(span)
                    start = source["text"].find(quote, start + 1)
    return list(items.values())


def render_inventory(items: list[dict]) -> str:
    rows = []
    for item in items:
        decision = item.get("review", {}).get("decision")
        if decision == "rejected":
            continue
        text = item["name"]
        if item["conditions"]:
            text += ": " + item["conditions"]
        if item["assessment"] == "ambiguous":
            text = "Needs review: " + text + " - " + item["uncertainty"]
        if decision is not None:
            text = ("Accepted by reviewer: " if decision == "accepted" else
                    "Unreviewed candidate: ") + text
        labels = sorted({support["label"] for support in item["support"]})
        rows.append(text + " " + " ".join(f"[{label}]" for label in labels))
    return "\n".join(rows)
