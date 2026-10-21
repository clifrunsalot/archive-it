import json
import logging
from dataclasses import asdict
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from document_answers import SynthesisValidationError
from document_inventory import (
    INVENTORY_SCHEMA, consolidate_inventory, filter_explicit_amenities, inventory_schema,
    parse_inventory, render_inventory,
)
from document_library import DocumentLibrary, DocumentQueryError
from document_synthesis import SynthesisRun, prompt_for, records_for, synthesize


@pytest.fixture
def sources():
    text = "Residents may use the gym for $10 monthly.\nAvailability may vary."
    return {
        label: {
            "label": label, "document_id": document, "version_id": version,
            "segment": 2, "source_path": "/private/lease.txt",
            "text": text, "location": {"page": 4, "char_start": 50, "char_end": 50 + len(text)},
        }
        for label, document, version in ((1, "a", 18), (2, "a", 18), (3, "b", 19), (4, "a", 20))
    }


def finding(label=1, **overrides):
    return {
        "text": "Gym", "conditions": "$10 monthly",
        "assessment": "established", "uncertainty": "",
        "evidence": [{"label": label, "quote": "Residents may use the gym for $10 monthly."}],
        **overrides,
    }


def response(*findings):
    return json.dumps({
        "status": "findings" if findings else "no_relevant_evidence",
        "findings": list(findings),
    })


def test_exact_quotes_resolve_to_segment_offsets_and_preserve_metadata(sources):
    batch = parse_inventory(response(finding()), sources, 24000)
    items = consolidate_inventory([batch], sources)
    item = items[0]
    support = item["support"][0]
    assert support["quote"] == sources[1]["text"][:42]
    assert support["location"] == {"page": 4, "char_start": 50, "char_end": 92}
    assert support["document_id"] == "a" and support["version_id"] == 18
    assert support["segment"] == 2
    assert support["source_path"] == "/private/lease.txt"
    assert item["quote_provenance_verified"]
    assert not item["semantic_relevance_verified"]
    assert render_inventory(items) == "Gym: $10 monthly [1]"


def test_quote_schema_contains_only_exact_source_choices_and_correlated_labels(sources):
    sources[1]["text"] = "Gym access costs €10.\r\nResidents may cancel."
    schema = inventory_schema({1: sources[1], 3: sources[3]})
    fields = schema["properties"]["findings"]["items"]["properties"]
    choices = fields["evidence"]["items"]["anyOf"]
    assert len(choices) == 2
    for choice in choices:
        label = choice["properties"]["label"]["const"]
        quotes = choice["properties"]["quote"]["enum"]
        assert sources[label]["text"] in quotes
        assert all(quote.strip() and quote in sources[label]["text"] for quote in quotes)
    assert "anyOf" not in INVENTORY_SCHEMA["properties"]["findings"]["items"][
        "properties"]["evidence"]["items"]


def test_quote_provenance_does_not_claim_category_relevance(sources):
    sources[1]["text"] = "The security deposit is $500."
    batch = parse_inventory(response(finding(
        text="Security deposit", conditions="$500",
        evidence=[{"label": 1, "quote": "The security deposit is $500."}],
    )), sources, 24000)
    item = consolidate_inventory([batch], sources)[0]
    assert item["quote_provenance_verified"]
    assert not item["semantic_relevance_verified"]


def test_amenity_filter_retains_only_explicitly_offered_items(sources):
    sources[1]["text"] = (
        "Residents have access to the fitness center. "
        "Reserved parking is available for an additional fee. "
        "Security deposit $500. "
        "Pets are prohibited in common spaces which may include a pool or sauna, if applicable."
    )
    findings = [
        finding(text="Fitness center", conditions="", evidence=[{
            "label": 1, "quote": "Residents have access to the fitness center.",
        }]),
        finding(text="Reserved parking", conditions="available for an additional fee", evidence=[{
            "label": 1, "quote": "Reserved parking is available for an additional fee.",
        }]),
        finding(text="Security deposit", conditions="$500", evidence=[{
            "label": 1, "quote": "Security deposit $500.",
        }]),
        finding(text="Pool", conditions="if applicable", evidence=[{
            "label": 1,
            "quote": "Pets are prohibited in common spaces which may include a pool or sauna, if applicable.",
        }]),
    ]

    filtered = json.loads(filter_explicit_amenities(
        response(*findings), {1: sources[1]}, 24000,
    ))

    assert [item["text"] for item in filtered["findings"]] == [
        "Fitness center", "Reserved parking",
    ]


def test_amenity_filter_keeps_ambiguous_provided_group_not_its_members(sources):
    text = (
        "Recreational Facility means swimming pool, sauna, exercise areas, tennis court. "
        "All or a part of these facilities have been provided as an amenity by Landlord."
    )
    sources[1]["text"] = text

    filtered = json.loads(filter_explicit_amenities(
        response(), {1: sources[1]}, 24000,
    ))

    assert filtered["status"] == "findings"
    assert filtered["findings"][0]["text"] == "Recreational facilities"
    assert filtered["findings"][0]["assessment"] == "ambiguous"
    assert "individual facilities unspecified" in filtered["findings"][0]["uncertainty"].casefold()
    assert filtered["findings"][0]["evidence"][0]["quote"] == (
        "All or a part of these facilities have been provided as an amenity by Landlord.")


@pytest.mark.parametrize(("overrides", "reason"), [
    ({"text": ""}, "invalid_finding_text"),
    ({"text": "Gym [1]"}, "invalid_finding_text"),
    ({"conditions": "fee\nincluded"}, "invalid_finding_text"),
    ({"conditions": "x" * 351}, "invalid_finding_text"),
    ({"assessment": "verified"}, "invalid_uncertainty"),
    ({"assessment": "ambiguous"}, "invalid_uncertainty"),
    ({"uncertainty": "Availability unknown"}, "invalid_uncertainty"),
    ({"evidence": [{"label": True, "quote": "gym"}]}, "citations_outside_batch"),
    ({"evidence": []}, "missing_supporting_quotes"),
    ({"evidence": [{"label": 999, "quote": "gym"}]}, "citations_outside_batch"),
    ({"evidence": [{"label": 1, "quote": "Residents have free gym access."}]},
     "supporting_quote_not_in_source"),
    ({"evidence": [{"label": 1, "quote": "   "}]}, "invalid_supporting_quote"),
    ({"evidence": [{"label": 1, "quote": "gym", "char_start": 0}]}, "invalid_supporting_quote"),
    ({"unexpected": True}, "invalid_schema"),
])
def test_invalid_records_fail_with_fixed_content_free_reasons(sources, overrides, reason):
    with pytest.raises(SynthesisValidationError, match=f"^{reason}$"):
        parse_inventory(response(finding(**overrides)), sources, 24000)


def test_quote_matching_is_exact_without_unicode_or_whitespace_rewriting(sources):
    sources[1]["text"] = "Fee: €10\nmonthly."
    valid = finding(evidence=[{"label": 1, "quote": "€10\nmonthly"}])
    parse_inventory(response(valid), sources, 24000)
    invalid = finding(evidence=[{"label": 1, "quote": "€10 monthly"}])
    with pytest.raises(SynthesisValidationError, match="supporting_quote_not_in_source"):
        parse_inventory(response(invalid), sources, 24000)


def test_repeated_quotes_retain_all_matching_offsets_and_line_numbers(sources):
    sources[1]["text"] = "gym\ngym"
    sources[1]["location"] = {"line_start": 8, "line_end": 9, "char_start": 100}
    item = finding(evidence=[{"label": 1, "quote": "gym"}])
    items = consolidate_inventory([parse_inventory(response(item), sources, 24000)], sources)
    assert [support["location"] for support in items[0]["support"]] == [
        {"line_start": 8, "line_end": 8, "char_start": 100, "char_end": 103},
        {"line_start": 9, "line_end": 9, "char_start": 104, "char_end": 107},
    ]


def test_exact_consolidation_keeps_conditions_uncertainty_versions_and_documents(sources):
    findings = [
        finding(), finding(2, text=" gym. "), finding(3), finding(4),
        finding(conditions="Fee waived"),
        finding(assessment="ambiguous", uncertainty="Availability may vary"),
    ]
    batches = [parse_inventory(response(item), sources, 24000) for item in findings]
    items = consolidate_inventory(batches + batches, sources)
    assert len(items) == 5
    assert {support["label"] for support in items[0]["support"]} == {1, 2}
    assert len(items[0]["support"]) == 2
    assert "Needs review: Gym: $10 monthly - Availability may vary [1]" in render_inventory(items)
    assert len({(item["document_id"], item["version_id"]) for item in items}) == 3


def test_items_cannot_merge_support_from_different_documents(sources):
    item = finding(evidence=[
        {"label": label, "quote": "Residents may use the gym for $10 monthly."}
        for label in (1, 3)
    ])
    with pytest.raises(SynthesisValidationError, match="inventory_item_crosses_documents"):
        consolidate_inventory([parse_inventory(response(item), sources, 24000)], sources)


@pytest.mark.parametrize("text", [
    "not JSON", "[]", '{"status":"findings","findings":[]}',
    '{"status":"no_relevant_evidence","findings":[{}]}',
    response(*[finding() for _ in range(17)]),
])
def test_invalid_batch_shapes_are_rejected(sources, text):
    with pytest.raises(SynthesisValidationError):
        parse_inventory(text, sources, 24000)


def test_structured_size_budget_is_enforced(sources):
    with pytest.raises(SynthesisValidationError, match="output_size_limit"):
        parse_inventory(response(finding()), sources, 30)


def test_quote_validation_failure_retries_then_fails_without_checkpoint_or_content_logs(
    sources, tmp_path, caplog,
):
    library = DocumentLibrary(tmp_path / "catalogue.sqlite3")
    run = SynthesisRun(library, "quote-protocol-test", 60, 3)
    bad_quote = "PRIVATE fabricated quote"
    model = Mock(return_value=AIMessage(content=response(finding(
        evidence=[{"label": 1, "quote": bad_quote}],
    ))))
    prompt = prompt_for("enumerate", "List every facility", "input", [sources[1]])
    caplog.set_level(logging.DEBUG, logger="archive_it")
    with pytest.raises(DocumentQueryError, match="supporting_quote_not_in_source"):
        run.generate(
            RunnableLambda(model), prompt, {1}, 24000, stage="input", batch=1,
            stats={"requests": 0}, schema=INVENTORY_SCHEMA,
            response_parser=lambda text: parse_inventory(text, sources, 24000),
        )
    assert model.call_count == 2
    assert bad_quote not in caplog.text
    with library.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM synthesis_checkpoints").fetchone()[0] == 0


def test_quote_checkpoints_are_revalidated_on_resume(sources, tmp_path):
    library = DocumentLibrary(tmp_path / "catalogue.sqlite3")
    run = SynthesisRun(library, "quote-protocol-test", 60, 3)
    model = Mock(return_value=AIMessage(content=response(finding())))
    prompt = prompt_for("enumerate", "List every facility", "input", [sources[1]])

    def call(run, stats):
        return run.generate(
            RunnableLambda(model), prompt, {1}, 24000, stage="input", batch=1, stats=stats,
            schema=INVENTORY_SCHEMA,
            response_parser=lambda text: parse_inventory(text, sources, 24000),
        )

    expected = call(run, {"requests": 0})
    resumed = SynthesisRun(DocumentLibrary(library.database), "quote-protocol-test", 60, 3)
    stats = {"requests": 0}
    assert call(resumed, stats) == expected
    assert model.call_count == 1 and resumed.reused == 1 and stats["requests"] == 0
    with library.connect() as connection:
        connection.execute("UPDATE synthesis_checkpoints SET text=?", (response(finding(
            evidence=[{"label": 1, "quote": "fabricated"}],
        )),))
    with pytest.raises(DocumentQueryError, match="checkpoint is invalid"):
        call(resumed, {"requests": 0})
    assert model.call_count == 1


def test_ambiguous_result_survives_formatting_and_json_serialization(tmp_path):
    library = DocumentLibrary(tmp_path / "catalogue.sqlite3")
    source = tmp_path / "lease.txt"
    source.write_text("Residents may use the gym for $10 monthly. Availability may vary.")
    identifier = library.ingest(source)
    item = finding(assessment="ambiguous", uncertainty="Availability may vary")
    result = synthesize(
        library, RunnableLambda(lambda _: AIMessage(content=response(item))),
        "enumerate", "List every facility", [identifier],
    )
    assert result.coverage["ambiguous_items"] == 1
    assert "Needs review:" in result.answer
    assert result.items[0]["support"][0]["location"]["char_start"] == 0
    assert json.loads(json.dumps(asdict(result)))["items"] == result.items
    from document_query import render_answer
    for presentation in ("prose", "list", "ascii_table", "csv"):
        output = render_answer(result, presentation)
        assert "Needs review:" in output
        assert "not verified" in output


def test_sentence_boundary_packing_retains_every_character_and_exact_offsets(tmp_path):
    library = DocumentLibrary(tmp_path / "catalogue.sqlite3")
    source = tmp_path / "lease.txt"
    text = ("A complete clause with a fee. " * 100) + "Final exception."
    source.write_text(text)
    records = records_for(library.snapshots([library.ingest(source)]))
    assert "".join(record["text"] for record in records) == text
    assert all(len(record["text"].encode()) <= 1400 for record in records)
    assert all(record["text"].endswith(". ") for record in records[:-1])
    for record in records:
        location = record["location"]
        assert text[location["char_start"]:location["char_end"]] == record["text"]


def test_cli_records_json_exposes_support_without_changing_default_output(tmp_path, monkeypatch, capsys):
    import sys
    import document_library
    import document_synthesis
    import chatbot_tech_doc
    from document_search import AnswerResult

    result = AnswerResult("Gym [1]", [], {"operation": "enumerate"}, items=[{
        "name": "Gym", "support": [{"quote": "Residents have gym access."}],
        "quote_provenance_verified": True, "semantic_relevance_verified": False,
    }])
    monkeypatch.setattr(document_library, "configure_logging", lambda _: None)
    monkeypatch.setattr(document_synthesis, "synthesize", Mock(return_value=result))
    monkeypatch.setattr(chatbot_tech_doc, "get_llm", Mock())
    monkeypatch.setattr(sys, "argv", [
        "document_library.py", "--database", str(tmp_path / "catalogue.sqlite3"),
        "enumerate", "List every amenity", "--document", "document", "--records-json",
    ])
    assert document_library.main() == 0
    data = json.loads(capsys.readouterr().out)
    assert data == asdict(result)
    assert data["items"][0]["support"][0]["quote"] == "Residents have gym access."
    monkeypatch.setattr(sys, "argv", sys.argv + ["--format", "prose"])
    assert document_library.main() == 1
    assert "cannot be combined" in capsys.readouterr().err
