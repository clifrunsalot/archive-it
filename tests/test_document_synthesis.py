import json
import logging
import os
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

import document_synthesis as synthesis
from document_answers import SynthesisValidationError, parse_findings
from document_library import DocumentLibrary, DocumentQueryError


@pytest.fixture
def documents(tmp_path):
    library = DocumentLibrary(tmp_path / "catalogue.sqlite3")
    ids = []
    for index in range(2):
        path = tmp_path / f"lease-{index}.txt"
        path.write_text(f"Monthly rent ${1200 + index * 100}. Pets require written consent.")
        ids.append(library.ingest(path))
    return library, ids


def responding(prompt):
    records = json.loads(prompt.to_messages()[-1].content.split("(JSON):\n", 1)[1])
    labels = sorted({
        label for record in records
        for label in ([record["label"]] if "label" in record else
                      [label for finding in record["findings"] for label in finding["citations"]])
    })
    return findings_message("Terms and uncertainties", labels)


def findings_message(text, labels):
    return AIMessage(content=json.dumps({
        "status": "findings", "findings": [{"text": text, "citations": labels}],
    }))


def inventory_message(records, names=None):
    if names is None:
        pairs = [(record, record["text"][:60]) for record in records]
    else:
        pairs = [(records[0], name) for name in names]
    return AIMessage(content=json.dumps({
        "status": "findings", "findings": [{
            "text": name, "conditions": "", "assessment": "established", "uncertainty": "",
            "evidence": [{"label": record["label"], "quote": record["text"][:100]}],
        } for record, name in pairs],
    }))


def test_comparison_reads_selected_extractions_without_index(documents):
    library, ids = documents
    model = Mock(side_effect=responding)
    result = synthesis.synthesize(library, RunnableLambda(model), "compare", "Compare terms", ids)
    assert model.call_count == 3
    assert {r["document_id"] for r in result.evidence} == set(ids)
    assert result.coverage["all_extracted_text_processed"]
    assert not result.coverage["exhaustive"]
    assert not result.coverage["original_completeness_verified"]
    assert len(result.coverage["documents"]) == 2
    assert all(len(call.args[0].to_string().encode()) <= synthesis.MAX_PROMPT_BYTES
               for call in model.call_args_list)


def test_final_synthesis_accepts_eight_findings_but_input_notes_remain_bounded(documents):
    library, ids = documents
    calls = []

    def respond(prompt):
        calls.append(prompt)
        records = json.loads(prompt.to_messages()[-1].text.split("(JSON):\n", 1)[1])
        if "label" in records[0]:
            return responding(prompt)
        labels = [label for record in records for finding in record["findings"]
                  for label in finding["citations"]]
        return AIMessage(content=json.dumps({
            "status": "findings",
            "findings": [{"text": f"Compared rule {number}", "citations": labels} for number in range(8)],
        }))

    result = synthesis.synthesize(library, RunnableLambda(respond), "compare", "Rules", ids)
    assert len(calls) == 3 and len(result.answer.splitlines()) == 8


def test_synthesis_notes_separate_text_from_original_citation_ids():
    records = [{
        "document": "original", "text": "Rent USD 1200.\nParking USD 50. [4] [10]\nTerm starts August. [5] [11]",
        "relevant": True,
    }]
    for operation in ("summarize", "compare"):
        prompt = synthesis.prompt_for(operation, "Terms", "Combine notes", records)
        supplied = json.loads(prompt.to_messages()[-1].text.split("(JSON):\n", 1)[1])
        assert supplied == [{
            "document": "D1", "relevant": True, "findings": [
                {"text": "Rent USD 1200.\nParking USD 50.", "citations": [4, 10]},
                {"text": "Term starts August.", "citations": [5, 11]},
            ],
        }]
        assert "Valid citation IDs: [4, 5, 10, 11]" in prompt.to_string()
    assert records[0]["text"].endswith("[5] [11]")


def test_synthesis_instructions_require_all_supporting_sources_and_separate_conditions():
    prompt = synthesis.prompt_for("summarize", "Summarize payments and services", "final", [
        {"label": 1, "text": "A deposit is payable once."},
        {"label": 2, "text": "Electricity is paid monthly."},
    ])
    instructions = prompt.to_messages()[0].text
    assert "Separate topics; cite ALL sources for every claim" in instructions
    assert "Keep conditions/frequencies with their original obligations" in instructions


def test_comparison_can_reuse_note_text_without_copying_inline_citations(documents):
    library, ids = documents

    def copy_notes(prompt):
        records = json.loads(prompt.to_messages()[-1].text.split("(JSON):\n", 1)[1])
        if "label" in records[0]:
            return findings_message("Monthly rent is stated.", [record["label"] for record in records])
        return AIMessage(content=json.dumps({
            "status": "findings",
            "findings": [finding for record in records for finding in record["findings"]],
        }))

    model = Mock(side_effect=copy_notes)
    result = synthesis.synthesize(
        library, RunnableLambda(model), "compare", "Compare charges and dates", ids,
    )
    assert model.call_count == 3
    assert {record["document_id"] for record in result.evidence} == set(ids)
    assert result.answer.count("Monthly rent is stated.") == 2


def test_enumeration_keeps_each_item_and_reads_all_extracted_text(documents):
    library, ids = documents
    source = Path(library.snapshots([ids[0]])[0].source_path)
    Path(source).write_text(
        "Amenities include a swimming pool, fitness room, package lockers, and a dog park."
    )
    library.ingest(source)
    model = Mock()

    def enumerate_items(prompt):
        prompt_text = prompt.to_string()
        assert "one finding per item" in prompt_text
        records = json.loads(prompt.to_messages()[-1].content.split("(JSON):\n", 1)[1])
        return inventory_message(records, ("Swimming pool", "Fitness room", "Package lockers", "Dog park"))

    model.side_effect = enumerate_items
    result = synthesis.synthesize(
        library, RunnableLambda(model), "enumerate", "List every amenity", [ids[0]],
    )

    assert model.call_count == 1
    assert all(f"{item} [1]" in result.answer for item in (
        "Swimming pool", "Fitness room", "Package lockers", "Dog park",
    ))
    assert result.coverage["all_extracted_text_processed"]
    assert result.coverage["enumeration_scope"] == "all_current_extracted_text"
    assert result.coverage["enumeration_item_limit_per_batch"] == 16
    assert result.coverage["model_generated_matches_may_be_incomplete"]
    assert not result.coverage["exhaustive"]
    assert {record["document_id"] for record in result.evidence} == {ids[0]}
    assert len(result.items) == 4
    assert all(item["support"][0]["quote"] in source.read_text() for item in result.items)
    assert result.coverage["quote_provenance_verified"]
    assert not result.coverage["semantic_relevance_verified"]


def test_enumeration_no_match_is_not_reported_as_document_absence(documents):
    library, ids = documents

    def no_match(_prompt):
        return AIMessage(content=json.dumps({
            "status": "no_relevant_evidence", "findings": [],
        }))

    result = synthesis.synthesize(
        library, RunnableLambda(no_match), "enumerate", "List swimming pool hours", [ids[0]],
    )

    assert "No matching items were identified" in result.answer
    assert "not proof of absence from the original" in result.answer
    assert result.evidence == []
    assert result.coverage["no_relevant_input_batches"] == 1
    assert result.coverage["no_relevant_document_ids"] == ids[:1]


def test_amenity_enumeration_filters_model_items_through_explicit_source_language(documents):
    library, ids = documents
    source = Path(library.snapshots([ids[0]])[0].source_path)
    Path(source).write_text(
        "Residents have access to the fitness center. "
        "Reserved parking is available for an additional fee. "
        "Security deposit $500. "
        "Pets are prohibited in common spaces which may include a pool or sauna, if applicable."
    )
    library.ingest(source)
    text = Path(source).read_text()

    def model(_prompt):
        candidates = [
            ("Fitness center", "", "Residents have access to the fitness center."),
            ("Reserved parking", "available for an additional fee",
             "Reserved parking is available for an additional fee."),
            ("Security deposit", "$500", "Security deposit $500."),
            ("Pool", "if applicable",
             "Pets are prohibited in common spaces which may include a pool or sauna, if applicable."),
        ]
        return AIMessage(content=json.dumps({
            "status": "findings",
            "findings": [{
                "text": name, "conditions": conditions,
                "assessment": "established", "uncertainty": "",
                "evidence": [{"label": 1, "quote": quote}],
            } for name, conditions, quote in candidates if quote in text],
        }))

    result = synthesis.synthesize(
        library, RunnableLambda(model), "enumerate", "List every amenity", [ids[0]],
    )

    assert [item["name"] for item in result.items] == ["Fitness center", "Reserved parking"]
    assert "Security deposit" not in result.answer
    assert "Pool" not in result.answer and "Sauna" not in result.answer


def test_amenity_enumeration_recovers_ambiguous_provided_group_from_source(documents):
    library, ids = documents
    source = Path(library.snapshots([ids[0]])[0].source_path)
    Path(source).write_text(
        "Recreational Facility means swimming pool, sauna, exercise areas, tennis court. "
        "All or a part of these facilities have been provided as an amenity by Landlord."
    )
    library.ingest(source)
    model = Mock(return_value=AIMessage(content=json.dumps({
        "status": "no_relevant_evidence", "findings": [],
    })))

    result = synthesis.synthesize(
        library, RunnableLambda(model), "enumerate", "List every amenity", [ids[0]],
    )

    assert len(result.items) == 1
    assert result.items[0]["name"] == "Recreational facilities"
    assert "individual facilities unspecified" in result.items[0]["uncertainty"].casefold()
    assert "swimming pool" not in result.answer.casefold()


def test_enumeration_prompt_is_separate_from_summary_instructions():
    records = [{"label": 7, "document": "lease", "text": "Resident access to fitness room."}]
    enumeration = synthesis.prompt_for(
        "enumerate", "List every amenity", "input", records,
    ).to_string()
    summary = synthesis.prompt_for(
        "summarize", "Summarize the lease", "input", records,
    ).to_string()
    assert "strict category filter" in enumeration
    assert "never discard supported facts" not in enumeration
    assert "individual facilities unspecified" in enumeration
    assert "stated fees and optional enrollment" in enumeration
    assert "available for an additional fee" in enumeration
    assert "do not stop after the first match" in enumeration
    assert "do not list its possible members or return no matches" in enumeration
    assert "never discard supported facts" in summary.lower()
    assert "strict category filter" not in summary


def test_enumeration_presentation_uses_compact_sources_and_item_rows(documents):
    from document_query import render_answer

    library, ids = documents
    result = synthesis.synthesize(
        library, RunnableLambda(lambda prompt: inventory_message(
            json.loads(prompt.to_messages()[-1].content.split("(JSON):\n", 1)[1]),
            ["Fitness center"],
        )),
        "enumerate", "List every item", ids[:1],
    )
    for presentation in ("prose", "list", "ascii_table", "csv"):
        output = render_answer(result, presentation)
        assert "Fitness center [1]" in output
        assert "lease-0.txt" in output
        assert library.snapshots(ids[:1])[0].source_path not in output
        assert "characters_processed" not in output
        assert "exhaustive" in output
        assert len(output) < 2000
    assert "document_id" in result.coverage["documents"][0]


def test_enumeration_splits_truncated_batch_and_preserves_citations(documents, monkeypatch, caplog):
    library, ids = documents
    snapshot = library.snapshots([ids[0]])[0]
    records = [{
        "label": label, "document_id": snapshot.document_id, "version_id": snapshot.version_id,
        "segment": label, "source_path": snapshot.source_path, "source": "lease.txt",
        "location": {"page": label}, "text": f"Amenity item number {label} is available.",
        "quality": {key: snapshot.extraction[key] for key in ("status", "issues", "ocr_used")},
        "latest_attempt_status": snapshot.latest_attempt_status,
    } for label in range(1, 5)]
    monkeypatch.setattr(synthesis, "records_for", lambda _snapshots: records)
    model = Mock()

    def respond(prompt):
        records = json.loads(prompt.to_messages()[-1].content.split("(JSON):\n", 1)[1])
        if len(records) > 2:
            return AIMessage(content="", response_metadata={"done_reason": "length"})
        return inventory_message(records)

    model.side_effect = respond
    result = synthesis.synthesize(
        library, RunnableLambda(model), "enumerate", "List every amenity", [ids[0]],
    )

    assert model.call_count == 3
    assert result.coverage["model_requests"] == 3
    for label in range(1, 5):
        assert f"[{label}]" in result.answer
    assert "batch split after truncated output" in caplog.text


def test_enumeration_single_passage_truncation_fails_once(documents):
    library, ids = documents
    model = Mock(return_value=AIMessage(
        content="", response_metadata={"done_reason": "length"},
    ))

    with pytest.raises(DocumentQueryError, match="single source passage"):
        synthesis.synthesize(
            library, RunnableLambda(model), "enumerate", "List every amenity", [ids[0]],
        )

    assert model.call_count == 1


def test_enumeration_rejects_hidden_reasoning_without_retry_or_checkpoint(documents, monkeypatch):
    from chatbot_tech_doc import get_llm

    library, ids = documents
    client = get_llm("deepseek-r1:8b")
    chat = Mock(return_value=iter([{
        "model": "deepseek-r1:8b",
        "message": {"role": "assistant", "content": findings_message("Pool", [1]).text,
                    "thinking": "PRIVATE REASONING"},
        "done": True, "done_reason": "stop",
    }]))
    monkeypatch.setattr(client._client, "chat", chat)
    with pytest.raises(DocumentQueryError, match="despite disabled thinking"):
        synthesis.synthesize(library, client, "enumerate", "List amenities", ids[:1])
    assert chat.call_count == 1
    assert chat.call_args.kwargs["think"] is False
    assert client.reasoning is True
    with library.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM synthesis_checkpoints").fetchone()[0] == 0


@pytest.mark.parametrize("stage", ["input", "reduce", "final"])
@pytest.mark.parametrize("name,budgets", [
    ("deepseek-r1:8b", [2048, 4096]),
    ("qwen3.5:4b", [2048, 2048]),
])
def test_synthesis_truncation_retry_preserves_schema_and_uses_effective_budget(
    documents, monkeypatch, stage, name, budgets, caplog,
):
    from chatbot_tech_doc import get_llm

    library, _ = documents
    client = get_llm(name)
    schema = synthesis.FINAL_SCHEMA if stage == "final" else synthesis.OUTPUT_SCHEMA
    chat = Mock(side_effect=[
        [{"model": name, "message": {"role": "assistant", "content": "",
                                   "thinking": "SECRET reasoning"},
          "done": True, "done_reason": "length"}],
        [{"model": name, "message": {"role": "assistant", "content":
                                   findings_message("Rent USD 1200.", [7]).text,
                                   "thinking": "SECRET reasoning"},
          "done": True, "done_reason": "stop"}],
    ])
    monkeypatch.setattr(client._client, "chat", chat)
    model = client.bind(reasoning=name.startswith("deepseek-r1:"), format=schema)
    run = synthesis.SynthesisRun(library, None, 600, 80)
    stats = {"requests": 0}
    with caplog.at_level(logging.DEBUG, logger="archive_it"):
        answer = run.generate(
            model, synthesis.prompt_for("summarize", "Rent?", stage,
                                        [{"label": 7, "text": "Rent USD 1200."}]),
            {7}, 1800, stage=stage, batch=1, stats=stats, schema=schema,
        )
    assert answer == "Rent USD 1200. [7]"
    assert stats["requests"] == run.calls == 2
    assert [call.kwargs["options"]["num_predict"] for call in chat.call_args_list] == budgets
    expected = json.loads(json.dumps(schema))
    expected["properties"]["findings"]["items"]["properties"]["citations"]["items"] = {
        "type": "integer", "enum": [7],
    }
    assert all(call.kwargs["format"] == expected for call in chat.call_args_list)
    assert stats.get("reasoning_responses", 0) == (2 if name.startswith("deepseek-r1:") else 0)
    assert "SECRET" not in caplog.text and "SECRET" not in answer
    assert client.num_predict == 2048 and client.reasoning is True


@pytest.mark.skipif(os.getenv("RUN_OLLAMA_TESTS") != "1", reason="Opt-in local Ollama inference")
def test_live_deepseek_manual_lease_summary(tmp_path):
    from chatbot_tech_doc import get_llm

    source = tmp_path / "manual-lease.txt"
    source.write_text(
        "MANUAL TEST LEASE\n\n"
        "The landlord is Example Property Management.\n"
        "The landlord's mailing address is 123 Example Street, Springfield, IL 62704.\n\n"
        "The tenant pays monthly rent of USD 1200.\n"
        "Parking costs an additional USD 50 per month.\n"
        "The refundable security deposit is USD 500, payable once.\n\n"
        "The tenant pays electricity and internet charges.\n"
        "The landlord pays water charges.\n\n"
        "The lease starts on January 1, 2027 and ends on December 31, 2027.\n\n"
        "Administrative contact listing:\n"
        "Example Property Management. No mailing address is recorded in this contact listing."
    )
    library = DocumentLibrary(tmp_path / "catalogue.sqlite3")
    identifier = library.ingest(source)
    result = synthesis.synthesize(
        library, get_llm("deepseek-r1:8b"), "summarize",
        "summarize the rent, parking, deposit, utilities, and lease dates.",
        [identifier], resume=False,
    )
    answer = result.answer.lower()
    for fact in ("1200", "50", "500", "electricity", "internet", "water", "2027"):
        assert fact in answer, f"Missing {fact}: {result.answer}"
    assert "january" in answer or "2027-01-01" in answer
    assert "december" in answer or "2027-12-31" in answer
    assert "refundable" in answer and ("once" in answer or "one-time" in answer)
    assert result.coverage["all_extracted_text_processed"]
    assert result.coverage["model_requests"] == 1
    assert result.coverage["synthesis_strategy"] == "direct"
    assert synthesis.cited_labels(result.answer)


def test_short_summary_uses_original_sources_in_one_call(documents, tmp_path):
    library, ids = documents
    source = tmp_path / "lease-0.txt"
    source.write_text("Monthly rent $1200.\n\nThe lease starts on January 1, 2027.")
    library.ingest(source)
    prompts = []

    def respond(prompt):
        prompts.append(prompt.to_string())
        records = json.loads(prompt.to_messages()[-1].text.split("(JSON):\n", 1)[1])
        assert all("label" in record for record in records)
        assert all("document" not in record for record in records)
        assert "January 1, 2027" in records[1]["text"]
        return findings_message("Lease starts January 1, 2027.", [2])

    result = synthesis.synthesize(
        library, RunnableLambda(respond), "summarize", "Summarize rent and dates", ids[:1],
    )
    assert "Lease starts January 1, 2027. [2]" == result.answer
    assert [record["label"] for record in result.evidence] == [2]
    assert result.evidence[0]["document_id"] == ids[0]
    assert len(prompts) == 1
    assert result.coverage["synthesis_strategy"] == "direct"
    assert all(len(prompt.encode()) <= synthesis.MAX_PROMPT_BYTES for prompt in prompts)


@pytest.mark.parametrize("claim", [
    "The lease starts 2027-01-01.", "The lease starts January 1, 2027.",
    "The lease starts\nJanuary 1 2027.",
])
def test_summary_date_requires_the_cited_passage_not_just_any_supplied_source(claim):
    from document_answers import validate_cited_dates

    sources = {1: {"text": "Rent USD 1200."}, 3: {"text": "Starts 2027-01-01."}}
    with pytest.raises(SynthesisValidationError, match="date_not_in_cited_sources"):
        validate_cited_dates(claim + " [1]", sources)
    validate_cited_dates(claim + " [3]", sources)
    validate_cited_dates(claim + " [1] [3]", sources)


@pytest.mark.parametrize("repair", [True, False])
def test_direct_summary_corrects_misplaced_date_citation_or_publishes_nothing(
    documents, tmp_path, monkeypatch, repair, caplog,
):
    from chatbot_tech_doc import get_llm

    library, ids = documents
    source = tmp_path / "lease-0.txt"
    source.write_text("Rent USD 1200.\n\nLease begins 2027-01-01 and ends 2027-12-31.")
    library.ingest(source)
    client = get_llm("deepseek-r1:8b")
    assert isinstance(client, ChatOllama)
    text = "Lease begins 2027-01-01 and ends 2027-12-31."
    chat = Mock(side_effect=[
        [{"model": client.model, "message": {"role": "assistant",
          "content": findings_message(text, [label]).text},
          "done": True, "done_reason": "stop"}]
        for label in (1, 2 if repair else 1)
    ])
    monkeypatch.setattr(client._client, "chat", chat)
    if repair:
        result = synthesis.synthesize(library, client, "summarize", "Lease dates?", ids[:1])
        assert result.answer == text + " [2]"
        assert result.coverage["model_requests"] == 2
        again = synthesis.synthesize(library, client, "summarize", "Lease dates?", ids[:1])
        assert again.coverage["model_requests"] == 0
    else:
        with pytest.raises(DocumentQueryError, match="date_not_in_cited_sources"):
            synthesis.synthesize(library, client, "summarize", "Lease dates?", ids[:1])
    assert chat.call_count == 2
    assert "date_not_in_cited_sources" in caplog.text
    with closing(library.connect()) as connection:
        assert connection.execute("SELECT COUNT(*) FROM synthesis_checkpoints").fetchone()[0] == int(repair)


def test_date_validation_rejects_tampered_checkpoint(documents):
    library, _ = documents
    run = synthesis.SynthesisRun(library, "date-profile", 600, 80)
    sources = {1: {"text": "Rent USD 1200."}, 2: {"text": "Lease starts 2027-01-01."}}
    prompt = synthesis.prompt_for("summarize", "Dates?", "final", [
        {"label": label, "text": record["text"]} for label, record in sources.items()
    ])
    model = RunnableLambda(lambda _: findings_message("Starts 2027-01-01.", [2]))
    run.generate(model, prompt, {1, 2}, 1800, stage="final", batch=1,
                 stats={"requests": 0}, date_sources=sources)
    with closing(library.connect()) as connection, connection:
        connection.execute("UPDATE synthesis_checkpoints SET text='Starts 2027-01-01. [1]'")
    with pytest.raises(DocumentQueryError, match="checkpoint is invalid.*date_not_in_cited_sources"):
        run.generate(model, prompt, {1, 2}, 1800, stage="final", batch=1,
                     stats={"requests": 0}, date_sources=sources)


@pytest.mark.parametrize("text", ["Bounded terms. ", "Unicode terms: \u20ac \u65e5\u672c\u8a9e. "])
@pytest.mark.parametrize("extra_bytes,strategy", [(0, "direct"), (-1, "hierarchical")])
def test_direct_summary_threshold_counts_utf8_and_reserves_retry_space(
    documents, tmp_path, monkeypatch, text, extra_bytes, strategy,
):
    library, ids = documents
    source = tmp_path / "lease-0.txt"
    source.write_text(text * 10)
    library.ingest(source)
    question = "Summarize terms"
    records = synthesis.records_for(library.snapshots(ids[:1]))
    inputs = [{key: record[key] for key in ("label", "location", "text")}
              | {"document": record["document_id"]} for record in records]
    final_stage = ("Produce the requested summary. Cite passage label IDs, never document aliases. "
                   "At most 4000 characters.")
    size = len(synthesis.prompt_for("summarize", question, final_stage, inputs).to_string().encode())
    monkeypatch.setattr(synthesis, "MAX_PROMPT_BYTES", size + synthesis.RETRY_RESERVE_BYTES + extra_bytes)
    model = Mock(side_effect=responding)
    result = synthesis.synthesize(library, RunnableLambda(model), "summarize", question, ids[:1])
    assert result.coverage["synthesis_strategy"] == strategy
    assert model.call_count == (1 if strategy == "direct" else 2)
    first = model.call_args_list[0].args[0]
    supplied = json.loads(first.to_messages()[-1].text.split("(JSON):\n", 1)[1])
    assert "".join(record["text"] for record in supplied) == text * 10
    assert all(len(call.args[0].to_string().encode()) <=
               synthesis.MAX_PROMPT_BYTES - synthesis.RETRY_RESERVE_BYTES
               for call in model.call_args_list)


def test_direct_summary_no_evidence_keeps_per_document_notices(documents):
    library, ids = documents
    model = Mock(return_value=AIMessage(content='{"status":"no_relevant_evidence","findings":[]}'))
    result = synthesis.synthesize(library, RunnableLambda(model), "summarize", "Pool rules", ids)
    assert model.call_count == 1
    assert result.answer.count("not proof of absence") == 2
    assert result.coverage["no_relevant_input_batches"] == 2
    assert set(result.coverage["no_relevant_document_ids"]) == set(ids)
    assert {record["document_id"] for record in result.evidence} == set(ids)


def test_direct_summary_must_cite_each_selected_document(documents):
    library, ids = documents
    model = Mock(return_value=findings_message("Rent", [1]))
    with pytest.raises(DocumentQueryError, match="selected document"):
        synthesis.synthesize(library, RunnableLambda(model), "summarize", "Rent", ids)
    assert model.call_count == 1


def test_direct_summary_cancellation_persists_no_partial_checkpoint(documents, monkeypatch):
    from chatbot_tech_doc import get_llm

    library, ids = documents
    model = get_llm("deepseek-r1:8b")
    assert isinstance(model, ChatOllama)
    chat = Mock(side_effect=KeyboardInterrupt)
    monkeypatch.setattr(model._client, "chat", chat)
    with pytest.raises(KeyboardInterrupt):
        synthesis.synthesize(library, model, "summarize", "Rent", ids[:1])
    assert chat.call_count == 1
    with closing(library.connect()) as connection:
        assert connection.execute("SELECT COUNT(*) FROM synthesis_checkpoints").fetchone()[0] == 0


def test_direct_summary_retry_obeys_invocation_budget(documents):
    library, ids = documents
    model = Mock(return_value=findings_message("Invalid", [999]))
    with pytest.raises(DocumentQueryError, match="budget reached"):
        synthesis.synthesize(library, RunnableLambda(model), "summarize", "Rent", ids[:1], max_calls=1)
    assert model.call_count == 1


@pytest.mark.parametrize("configured_budget", [2048, 8192])
@pytest.mark.parametrize("reason", ["length", "stop"])
def test_deepseek_summary_final_retry_keeps_larger_budget_and_only_expands_on_truncation(
    documents, monkeypatch, configured_budget, reason, caplog,
):
    from chatbot_tech_doc import get_llm

    library, ids = documents
    client = get_llm("deepseek-r1:8b").model_copy(update={"num_predict": configured_budget})
    assert isinstance(client, ChatOllama)

    def response(content, done="stop"):
        return [{"model": client.model, "message": {
            "role": "assistant", "content": content, "thinking": "SECRET reasoning",
        }, "done": True, "done_reason": done}]

    chat = Mock(side_effect=[
        response("{}", reason),
        response(findings_message("Rent USD 1200.", [1]).text),
    ])
    monkeypatch.setattr(client._client, "chat", chat)
    result = synthesis.synthesize(library, client, "summarize", "Rent?", ids[:1])
    expected_retry = max(configured_budget, 4096) if reason == "length" else configured_budget
    assert [call.kwargs["options"]["num_predict"] for call in chat.call_args_list] == [
        configured_budget, expected_retry,
    ]
    assert all(call.kwargs["think"] is True for call in chat.call_args_list)
    assert result.coverage["model_reasoning_responses"] == 2
    assert "SECRET" not in result.answer and "SECRET" not in caplog.text
    assert client.num_predict == configured_budget


def test_deepseek_repeated_direct_truncation_persists_no_checkpoint(documents, monkeypatch):
    from chatbot_tech_doc import get_llm

    library, ids = documents
    client = get_llm("deepseek-r1:8b")
    assert isinstance(client, ChatOllama)
    chat = Mock(side_effect=[
        *[[{"model": client.model, "message": {"role": "assistant", "content": ""},
            "done": True, "done_reason": "length"}]] * 2,
    ])
    monkeypatch.setattr(client._client, "chat", chat)
    with pytest.raises(DocumentQueryError, match="stage=final.*truncated_output.*one corrective retry"):
        synthesis.synthesize(library, client, "summarize", "Rent?", ids[:1])
    assert [call.kwargs["options"]["num_predict"] for call in chat.call_args_list] == [2048, 4096]
    with closing(library.connect()) as connection:
        assert connection.execute("SELECT COUNT(*) FROM synthesis_checkpoints").fetchone()[0] == 0


def test_enumeration_does_not_use_summary_output_limit(documents, monkeypatch):
    library, ids = documents
    original_records = synthesis.records_for(library.snapshots(ids[:1]))
    records = [
        dict(original_records[0], label=index, text=f"Amenity {index}")
        for index in range(1, 17)
    ]
    monkeypatch.setattr(synthesis, "records_for", lambda _: records)
    monkeypatch.setattr(
        synthesis, "pack", lambda records, *_: [[record] for record in records],
    )

    def respond(prompt):
        record = json.loads(prompt.to_messages()[-1].content.split("(JSON):\n", 1)[1])[0]
        return inventory_message([record], [
            f"Item {record['label']}-{index}: " + "x" * 300 for index in range(3)
        ])

    result = synthesis.synthesize(
        library, RunnableLambda(respond), "enumerate", "List all items", ids[:1],
    )
    assert synthesis.MAX_ANSWER_CHARS < len(result.answer) < synthesis.MAX_ENUMERATION_ANSWER_CHARS
    assert result.coverage["all_extracted_text_processed"]
    monkeypatch.setattr(synthesis, "MAX_ENUMERATION_ANSWER_CHARS", 100)
    with pytest.raises(DocumentQueryError, match="output budget"):
        synthesis.synthesize(
            library, RunnableLambda(respond), "enumerate", "List all items", ids[:1],
        )


def test_long_unicode_input_processed_once_and_end_included(documents, tmp_path):
    library, _ = documents
    text = "Lease € 日本語 terms. " * 1300 + "FINAL DEADLINE 2030-01-01"
    source = tmp_path / "long.txt"
    source.write_text(text)
    identifier = library.ingest(source)
    def verbose_notes(prompt):
        response = responding(prompt)
        data = json.loads(response.text)
        data["findings"][0]["text"] = "Relevant terms and unresolved exceptions. " * 15
        return AIMessage(content=json.dumps(data))
    model = Mock(side_effect=verbose_notes)
    result = synthesis.synthesize(library, RunnableLambda(model), "summarize", "Summarize", [identifier])
    seen = []
    for call in model.call_args_list:
        content = call.args[0].to_messages()[-1].content
        assert len(call.args[0].to_string().encode()) <= synthesis.MAX_PROMPT_BYTES
        records = json.loads(content.split("(JSON):\n", 1)[1])
        seen.extend(record["text"] for record in records if "label" in record)
    assert "".join(seen) == text
    assert result.coverage["reduction_levels"] > 0
    assert result.coverage["documents"][0]["characters_processed"] == len(text)


@pytest.mark.parametrize("operation,selected,reason", [
    ("summarize", None, "explicitly"), ("compare", [0], "2 to 4"),
])
def test_scope_gates_before_model(documents, operation, selected, reason):
    library, ids = documents
    model = Mock()
    with pytest.raises(DocumentQueryError, match=reason):
        synthesis.synthesize(library, RunnableLambda(model), operation, "Request",
                             [ids[i] for i in selected] if selected else None)
    model.assert_not_called()


@pytest.mark.parametrize("response", [
    AIMessage(content="No citations"), AIMessage(content="Wrong [999]"),
    AIMessage(content="Partial [1]", response_metadata={"done_reason": "length"}),
    AIMessage(content="x" * 1801 + "[1]"), "not a message",
])
def test_invalid_model_outputs_never_publish(documents, response):
    library, ids = documents
    with pytest.raises(DocumentQueryError):
        synthesis.synthesize(library, RunnableLambda(lambda _: response), "summarize",
                             "Summarize", ids[:1])


def test_cross_batch_citation_rejected(documents):
    library, ids = documents
    model = Mock(side_effect=[findings_message("Terms", [1]),
                              findings_message("Wrong", [1]), findings_message("Wrong", [1])])
    with pytest.raises(DocumentQueryError, match="citations_outside_batch"):
        synthesis.synthesize(library, RunnableLambda(model), "compare", "Compare", ids)


def test_final_must_cite_every_selected_document(documents):
    library, ids = documents
    calls = 0
    def respond(prompt):
        nonlocal calls
        calls += 1
        return findings_message("Only one", [1]) if calls == 3 else responding(prompt)
    with pytest.raises(DocumentQueryError, match="selected document"):
        synthesis.synthesize(library, RunnableLambda(respond), "compare", "Compare", ids)


def test_input_limits_preflight_without_model(documents, monkeypatch):
    library, ids = documents
    monkeypatch.setattr(synthesis, "MAX_BATCHES", 1)
    model = Mock()
    with pytest.raises(DocumentQueryError, match="input batches"):
        synthesis.synthesize(library, RunnableLambda(model), "compare", "Compare", ids)
    model.assert_not_called()


def test_nonshrinking_notes_fail_instead_of_dropping_input(documents, monkeypatch):
    library, ids = documents
    original = synthesis.pack
    def separate_notes(records, operation, question, stage):
        if records and "label" not in records[0]:
            return [[record] for record in records]
        return original(records, operation, question, stage)
    monkeypatch.setattr(synthesis, "pack", separate_notes)
    with pytest.raises(DocumentQueryError, match="without losing coverage"):
        synthesis.synthesize(library, RunnableLambda(responding), "compare", "Compare", ids)


def test_oversized_prompt_fails_before_generation(documents, monkeypatch):
    library, ids = documents
    monkeypatch.setattr(synthesis, "MAX_PROMPT_BYTES", 100)
    model = Mock()
    with pytest.raises(DocumentQueryError, match="prompt budget"):
        synthesis.synthesize(library, RunnableLambda(model), "summarize", "Summarize", ids[:1])
    model.assert_not_called()


@pytest.mark.parametrize("operation", ["summarize", "compare"])
def test_direct_cli_routes_without_index(documents, tmp_path, monkeypatch, capsys, operation):
    import document_library
    import chatbot_tech_doc

    library, ids = documents
    config = tmp_path / "logging.json"
    config.write_text(json.dumps({
        "level": "DEBUG", "file": "synthesis.log", "max_bytes": 100000, "retained_files": 5,
    }))
    monkeypatch.setattr(chatbot_tech_doc, "get_llm", lambda _: RunnableLambda(responding))
    arguments = ["document_library.py", "--database", str(library.database),
                 "--log-config", str(config), operation, "Summarize or compare terms", "--format", "csv"]
    for identifier in ids:
        arguments.extend(["--document", identifier])
    monkeypatch.setattr("sys.argv", arguments)
    from app_logging import LOGGER
    previous, level = LOGGER.handlers[:], LOGGER.level
    try:
        assert document_library.main() == 0
        output = capsys.readouterr()
        assert output.out.startswith("Type,Content")
        assert "all_extracted_text_processed" in output.out
        assert not output.err
        log_text = (tmp_path / "synthesis.log").read_text()
        assert ("Synthesis progress stage=final completed=0 total=1" if operation == "summarize"
                else "Synthesis progress stage=input completed=0 total=2") in log_text
        assert "Synthesis progress stage=completed completed=1 total=1" in log_text
    finally:
        for handler in LOGGER.handlers:
            if handler not in previous:
                handler.close()
        LOGGER.handlers = previous
        LOGGER.setLevel(level)


def test_quality_and_failed_latest_attempt_retained(documents, tmp_path):
    library, ids = documents
    source = tmp_path / "lease-0.txt"
    source.write_bytes(b"\xff")
    with pytest.raises(ValueError):
        library.ingest(source)
    result = synthesis.synthesize(library, RunnableLambda(responding), "summarize", "Summarize", ids[:1])
    assert result.coverage["documents"][0]["latest_attempt_status"] == "failed"
    assert result.evidence[0]["latest_attempt_status"] == "failed"


def test_document_changes_during_generation_fail(documents, tmp_path):
    library, ids = documents
    def respond(prompt):
        source = tmp_path / "lease-0.txt"
        source.write_text("New terms")
        library.ingest(source)
        return responding(prompt)
    with pytest.raises(DocumentQueryError, match="changed during"):
        synthesis.synthesize(library, RunnableLambda(respond), "summarize", "Summarize", ids[:1])


def test_logs_exclude_contents_and_paths(documents, caplog):
    library, ids = documents
    with caplog.at_level(logging.DEBUG, logger="archive_it"):
        synthesis.synthesize(library, RunnableLambda(responding), "summarize", "SECRET request", ids[:1])
    assert "Whole document synthesis completed" in caplog.text
    assert "SECRET" not in caplog.text and "Monthly rent" not in caplog.text
    assert "lease-0.txt" not in caplog.text


def test_no_relevant_evidence_is_explicit_and_still_covers_documents(documents):
    library, ids = documents
    model = RunnableLambda(lambda _: AIMessage(content=json.dumps({
        "status": "no_relevant_evidence", "findings": [],
    })))
    result = synthesis.synthesize(library, model, "compare", "Compare swimming pools", ids)
    assert "No relevant evidence identified in the processed evidence" in result.answer
    assert "not proof of absence" in result.answer
    assert {record["document_id"] for record in result.evidence} == set(ids)
    assert result.coverage["all_extracted_text_processed"]
    assert result.coverage["no_relevant_results"] == 2
    assert result.coverage["model_requests"] == 2
    assert result.coverage["no_relevant_input_batches"] == 2


def test_final_cannot_discard_known_relevant_notes(documents):
    library, ids = documents
    calls = 0
    def respond(prompt):
        nonlocal calls
        calls += 1
        if calls > 2:
            return AIMessage(content='{"status":"no_relevant_evidence","findings":[]}')
        return responding(prompt)
    with pytest.raises(DocumentQueryError, match="relevant_notes_discarded"):
        synthesis.synthesize(library, RunnableLambda(respond), "compare", "Compare", ids)
    assert calls == 4


@pytest.mark.parametrize("seconds,calls", [(0, 80), (float("nan"), 80), (10, 0), (10, 1.5)])
def test_invalid_synthesis_budgets(documents, seconds, calls):
    library, _ = documents
    with pytest.raises(DocumentQueryError, match="budgets"):
        synthesis.SynthesisRun(library, None, seconds, calls)


def test_call_budget_counts_retries(documents):
    library, _ = documents
    run = synthesis.SynthesisRun(library, "test-profile", 600, 1)
    model = Mock(return_value=findings_message("Wrong", [999]))
    prompt = synthesis.prompt_for("summarize", "Summarize", "input", [{"label": 1, "text": "Terms"}])
    with pytest.raises(DocumentQueryError, match="budget reached"):
        run.generate(RunnableLambda(model), prompt, {1}, 1800,
                     stage="input", batch=1, stats={"requests": 0})
    assert model.call_count == 1


def test_elapsed_budget_checks_before_request(documents, monkeypatch):
    library, _ = documents
    monkeypatch.setattr(synthesis, "monotonic", Mock(side_effect=[10, 610]))
    run = synthesis.SynthesisRun(library, None, 600, 80)
    with pytest.raises(DocumentQueryError, match="budget reached"):
        run.before_call()
    assert run.calls == 0


def test_checkpoint_reopens_and_invalidates_by_prompt_or_profile(documents):
    library, _ = documents
    prompt = synthesis.prompt_for("summarize", "Terms", "input", [{"label": 7, "text": "SECRET terms"}])
    model = Mock(return_value=findings_message("SECRET note", [7]))
    first = synthesis.SynthesisRun(library, "model-profile", 600, 80)
    text = first.generate(RunnableLambda(model), prompt, {7}, 1800,
                          stage="input", batch=1, stats={"requests": 0})
    reopened = DocumentLibrary(library.database)
    resumed = synthesis.SynthesisRun(reopened, "model-profile", 600, 80)
    stats = {"requests": 0}
    assert resumed.generate(RunnableLambda(model), prompt, {7}, 1800,
                            stage="input", batch=1, stats=stats) == text
    assert stats["requests"] == 0 and resumed.reused == 1
    assert model.call_count == 1
    changed = synthesis.SynthesisRun(reopened, "different-model", 600, 80)
    changed.generate(RunnableLambda(model), prompt, {7}, 1800,
                     stage="input", batch=1, stats={"requests": 0})
    assert model.call_count == 2
    new_prompt = synthesis.prompt_for("summarize", "Different request", "input",
                                     [{"label": 7, "text": "SECRET terms"}])
    resumed.generate(RunnableLambda(model), new_prompt, {7}, 1800,
                     stage="input", batch=1, stats={"requests": 0})
    assert model.call_count == 3


def test_cancelled_call_retains_only_completed_checkpoints_and_reports_safe_progress(documents, capsys):
    from document_progress import OperationProgress

    library, _ = documents
    prompt = synthesis.prompt_for("summarize", "SECRET question", "input",
                                  [{"label": 7, "text": "SECRET terms"}])
    interrupted_prompt = synthesis.prompt_for("summarize", "Another SECRET question", "input",
                                              [{"label": 7, "text": "SECRET terms"}])
    run = synthesis.SynthesisRun(library, "model-profile", 600, 80)
    model = Mock(side_effect=[findings_message("SECRET note", [7]), KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        with OperationProgress("summarize", "test-model", interval=60):
            run.generate(RunnableLambda(model), prompt, {7}, 1800,
                         stage="input", batch=1, stats={"requests": 0})
            run.progress("input", 1, 2)
            run.generate(RunnableLambda(model), interrupted_prompt, {7}, 1800,
                         stage="input", batch=2, stats={"requests": 1})
    with closing(library.connect()) as connection:
        assert connection.execute("SELECT COUNT(*) FROM synthesis_checkpoints").fetchone()[0] == 1
    resumed = synthesis.SynthesisRun(DocumentLibrary(library.database), "model-profile", 600, 80)
    resumed.generate(RunnableLambda(model), prompt, {7}, 1800,
                     stage="input", batch=1, stats={"requests": 0})
    assert model.call_count == 2 and resumed.reused == 1
    output = capsys.readouterr().err
    assert "1/2 batches" in output and "request 2" in output and "cancelled" in output
    assert "SECRET" not in output


def test_invalid_checkpoint_is_explicit(documents):
    library, _ = documents
    run = synthesis.SynthesisRun(library, "profile", 600, 80)
    prompt = synthesis.prompt_for("summarize", "Terms", "input", [{"label": 1, "text": "Terms"}])
    run.generate(RunnableLambda(lambda _: findings_message("Terms", [1])),
                 prompt, {1}, 1800, stage="input", batch=1, stats={"requests": 0})
    with library.connect() as connection:
        connection.execute("UPDATE synthesis_checkpoints SET text='Wrong [999]'")
    with pytest.raises(DocumentQueryError, match="checkpoint is invalid"):
        run.generate(RunnableLambda(responding), prompt, {1}, 1800,
                     stage="input", batch=1, stats={"requests": 0})


def test_real_client_direct_resume_and_version_change(documents, tmp_path, monkeypatch):
    from chatbot_tech_doc import get_llm
    from langchain_ollama import ChatOllama
    library, ids = documents
    client = get_llm("qwen3.5:4b")
    assert isinstance(client, ChatOllama)
    chat = Mock(side_effect=lambda **kwargs: iter([{
        "model": "qwen3.5:4b", "message": {
            "role": "assistant", "content": findings_message("Terms", [1]).text,
        }, "done": True, "done_reason": "stop",
    }]))
    monkeypatch.setattr(client._client, "chat", chat)
    first = synthesis.synthesize(library, client, "summarize", "Terms", ids[:1], max_calls=1)
    assert first.coverage["model_requests"] == 1
    assert chat.call_count == 1
    result = synthesis.synthesize(DocumentLibrary(library.database), client, "summarize", "Terms",
                                  ids[:1], max_calls=1)
    assert result.coverage["model_requests"] == 0
    assert result.coverage["reused_checkpoints"] == 1
    again = synthesis.synthesize(library, client, "summarize", "Terms", ids[:1])
    assert again.coverage["model_requests"] == 0 and again.coverage["reused_checkpoints"] == 1
    (tmp_path / "lease-0.txt").write_text("Changed rent")
    library.ingest(tmp_path / "lease-0.txt")
    changed = synthesis.synthesize(library, client, "summarize", "Terms", ids[:1])
    assert changed.coverage["reused_checkpoints"] == 0
    assert changed.coverage["model_requests"] == 1


def test_irrelevant_batches_never_enter_reduction(documents, tmp_path):
    library, _ = documents
    path = tmp_path / "many.txt"
    path.write_text("Unrelated text " * 3000)
    identifier = library.ingest(path)
    calls = []
    def respond(prompt):
        records = json.loads(prompt.to_messages()[-1].content.split("(JSON):\n", 1)[1])
        calls.append(records)
        assert all("label" in record for record in records)
        return AIMessage(content='{"status":"no_relevant_evidence","findings":[]}')
    result = synthesis.synthesize(library, RunnableLambda(respond), "summarize", "Rent", [identifier])
    assert result.coverage["model_requests"] == result.coverage["input_batches"] == len(calls)
    assert result.coverage["reduction_levels"] == 0


def test_compact_prompt_measurable_batch_capacity():
    records = [{"label": index + 1, "document": "d" * 64,
                "location": {"page": index + 1}, "text": "x" * 1400} for index in range(119)]
    batches = synthesis.pack(records, "summarize", "Summarize rent",
                             "Read every supplied passage. Produce cited notes, at most 900 characters.")
    assert len(batches) <= 40
    assert [r for batch in batches for r in batch] == records
    assert all(len(synthesis.prompt_for("summarize", "Summarize rent",
               "Read every supplied passage. Produce cited notes, at most 900 characters.",
               batch).to_string().encode()) + synthesis.RETRY_RESERVE_BYTES
               <= synthesis.MAX_PROMPT_BYTES for batch in batches)


def test_no_findings_document_preserved_without_reducing_its_notices(documents):
    library, ids = documents
    calls = 0
    def respond(prompt):
        nonlocal calls
        calls += 1
        if calls == 2:
            return AIMessage(content='{"status":"no_relevant_evidence","findings":[]}')
        return responding(prompt)
    result = synthesis.synthesize(library, RunnableLambda(respond), "compare", "Compare rent", ids)
    assert calls == 3
    assert len(result.coverage["no_relevant_document_ids"]) == 1
    assert {record["document_id"] for record in result.evidence} == set(ids)
    assert "not proof of absence" in result.answer


def test_no_resume_bypasses_persistent_checkpoints(documents, monkeypatch):
    from chatbot_tech_doc import get_llm
    from langchain_ollama import ChatOllama
    library, ids = documents
    model = get_llm("qwen3.5:4b")
    assert isinstance(model, ChatOllama)
    chat = Mock(side_effect=lambda **kwargs: iter([{
        "model": "qwen3.5:4b", "message": {
            "role": "assistant", "content": findings_message("Terms", [1]).text,
        }, "done": True, "done_reason": "stop",
    }]))
    monkeypatch.setattr(model._client, "chat", chat)
    synthesis.synthesize(library, model, "summarize", "Terms", ids[:1])
    second = synthesis.synthesize(library, model, "summarize", "Terms", ids[:1], resume=False)
    assert second.coverage["model_requests"] == 1
    assert second.coverage["reused_checkpoints"] == 0
    assert chat.call_count == 2


@pytest.mark.parametrize("value,reason", [
    ({"status": "findings", "findings": [{"text": "Rent", "citations": []}]}, "missing_citations"),
    ({"status": "findings", "findings": [{"text": "Rent", "citations": [True]}]},
     "citations_outside_batch"),
    ({"status": "findings", "findings": [{"text": "Rent [1]", "citations": [1]}]},
     "invalid_finding_text"),
    ({"status": "no_relevant_evidence", "findings": [{"text": "Rent", "citations": [1]}]},
     "inconsistent_no_relevant_evidence"),
    ({"status": "findings", "findings": []}, "missing_findings"),
])
def test_structured_validation_exact_reasons(value, reason):
    with pytest.raises(SynthesisValidationError, match=reason):
        parse_findings(json.dumps(value), {1}, 1800)


def test_one_corrective_retry_preserves_budget_and_counts(documents, caplog):
    library, ids = documents
    calls = []
    def respond(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return findings_message("SECRET rejected text", [999])
        return responding(prompt)
    with caplog.at_level(logging.DEBUG, logger="archive_it"):
        result = synthesis.synthesize(library, RunnableLambda(respond), "summarize", "Summarize", ids[:1])
    assert len(calls) == 2
    assert result.coverage["model_requests"] == 2
    assert "Corrective retry" in calls[1].to_string()
    assert "SECRET" not in calls[1].to_string() and "SECRET" not in caplog.text
    assert "stage=final batch=1 attempt=1 reason=citations_outside_batch" in caplog.text
    assert all(len(prompt.to_string().encode()) <= synthesis.MAX_PROMPT_BYTES for prompt in calls)


def test_retry_exhaustion_does_not_weaken_validation(documents, caplog):
    library, ids = documents
    model = Mock(return_value=findings_message("Unsupported", [999]))
    with caplog.at_level(logging.WARNING, logger="archive_it"):
        with pytest.raises(DocumentQueryError, match="after one corrective retry"):
            synthesis.synthesize(library, RunnableLambda(model), "summarize", "Summarize", ids[:1])
    assert model.call_count == 2
    assert "stage=final batch=1 attempt=2 reason=citations_outside_batch" in caplog.text


@pytest.mark.parametrize("status", [-1, 500, 502, 503, 504])
def test_transient_server_error_uses_same_bounded_retry(documents, status):
    from ollama import ResponseError
    library, ids = documents
    calls = 0
    def respond(prompt):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ResponseError("SECRET service text", status_code=status)
        return responding(prompt)
    result = synthesis.synthesize(library, RunnableLambda(respond), "summarize", "Summarize", ids[:1])
    assert result.coverage["model_requests"] == 2


def test_nontransient_service_error_is_not_retried(documents):
    from ollama import ResponseError
    library, ids = documents
    model = Mock(side_effect=ResponseError("missing model", status_code=404))
    with pytest.raises(ResponseError):
        synthesis.synthesize(library, RunnableLambda(model), "summarize", "Summarize", ids[:1])
    assert model.call_count == 1


def test_record_offsets_preserve_lines_and_ocr_quality(documents):
    library, ids = documents
    snapshots = library.snapshots(ids[:1])
    snapshots[0].extraction["ocr_used"] = True
    snapshots[0].extraction["status"] = "partial"
    snapshots[0].extraction["issues"] = ["ocr_text_requires_review"]
    snapshots[0].extraction["segments"][0]["text"] = "line\n" * 1000
    records = synthesis.records_for(snapshots)
    assert "".join(r["text"] for r in records) == "line\n" * 1000
    assert records[1]["location"]["line_start"] > 1
    assert all(r["quality"]["ocr_used"] for r in records)


@pytest.mark.skipif(os.environ.get("RUN_OLLAMA_TESTS") != "1", reason="Enable local synthesis integration")
@pytest.mark.parametrize(("text", "expected", "excluded"), [
    (
        "CONTENTS: Fitness Center Addendum; Window Guard Addendum. "
        "Security deposit $500. Maximum occupancy: two people. "
        "Pets are prohibited in common spaces, which may include a pool, sauna, "
        "playground or business center, if applicable.",
        (),
        ("fitness", "window", "deposit", "occupancy", "pool", "sauna", "playground"),
    ),
    (
        "Residents have access to the fitness center. Reserved parking is available for "
        "an additional fee. Security deposit $500. Maximum occupancy two people. "
        "Pets are prohibited in common spaces which may include a pool or sauna, if applicable.",
        ("fitness", "reserved parking", "fee"),
        ("deposit", "occupancy", "pool", "sauna"),
    ),
    (
        "Recreational Facility means swimming pool, sauna, exercise areas, tennis court, "
        "playground, resident lounge, barbecue grills, etc. All or a part of these "
        "facilities have been provided as an amenity by Landlord for recreational enjoyment.",
        ("recreational facilities",),
        ("swimming pool", "sauna", "tennis", "playground", "grills"),
    ),
])
def test_live_enumeration_filters_irrelevant_mentions(tmp_path, text, expected, excluded):
    from chatbot_tech_doc import get_llm

    library = DocumentLibrary(tmp_path / "catalogue.sqlite3")
    source = tmp_path / "amenities.txt"
    source.write_text(text)
    document = library.ingest(source)
    result = synthesis.synthesize(
        library, get_llm("qwen3.5:4b"), "enumerate",
        "List every amenity in the lease", [document],
    )
    answer = result.answer.casefold()
    assert all(term in answer for term in expected), result.answer
    assert all(term not in answer for term in excluded), result.answer
    assert len(result.answer) < 1000
    if not expected:
        assert "No matching items were identified" in result.answer
    assert result.coverage["all_extracted_text_processed"]
    assert not result.coverage["exhaustive"]


@pytest.mark.skipif(os.environ.get("RUN_OLLAMA_TESTS") != "1", reason="Enable local synthesis integration")
def test_live_budget_resume(documents):
    from chatbot_tech_doc import get_llm
    library, ids = documents
    model = get_llm("qwen3.5:4b")
    with pytest.raises(DocumentQueryError, match="budget reached"):
        synthesis.synthesize(library, model, "compare", "Compare rent", ids, max_calls=1)
    result = synthesis.synthesize(DocumentLibrary(library.database), model, "compare",
                                  "Compare rent", ids)
    assert {record["document_id"] for record in result.evidence} == set(ids)
    assert result.coverage["reused_checkpoints"] == 1
    assert result.coverage["model_requests"] == 2


@pytest.mark.skipif(os.environ.get("RUN_OLLAMA_TESTS") != "1", reason="Enable local synthesis integration")
def test_live_summary_and_comparison(documents):
    from chatbot_tech_doc import get_embeddings, get_llm
    from document_query import QuerySession, get_planner
    from document_search import DocumentSearch

    library, ids = documents
    model = get_llm("qwen3.5:4b")
    service = QuerySession(DocumentSearch(library, get_embeddings(), "nomic-embed-text"),
                           get_planner(), model, ids[:1])
    service.execute("Summarize the entire selected lease, including rent and pet obligations.")
    assert service.previous_plan.intent == "summarize"
    summary = service.previous_result
    assert "1200" in summary.answer.replace(",", "")
    service.select(ids)
    service.execute("Compare the rent and pet obligations in both selected leases.")
    assert service.previous_plan.intent == "compare"
    comparison = service.previous_result
    assert "1200" in comparison.answer.replace(",", "")
    assert "1300" in comparison.answer.replace(",", "")
    assert {r["document_id"] for r in comparison.evidence} == set(ids)
    assert service.execute("Reformat that comparison as CSV.").startswith("Type,Content")
