import json

import pytest

from document_evidence_ui import inspect_evidence, render_search_results
from document_library import DocumentLibrary, DocumentQueryError, ExtractionError
from document_search import AnswerResult


@pytest.fixture
def evidence(tmp_path):
    library = DocumentLibrary(tmp_path / "catalogue.sqlite3")
    source = tmp_path / "source.txt"
    source.write_text("Earlier context.\n\nSECRET original passage.\n\nLater context.")
    identifier = library.ingest(source)
    snapshot = library.snapshots([identifier])[0]
    record = {
        "label": 1, "document_id": identifier, "version_id": snapshot.version_id,
        "source_path": str(source), "segment": 2,
        "location": {**snapshot.extraction["segments"][1]["location"], "char_start": 0, "char_end": 6},
        "text": snapshot.extraction["segments"][1]["text"][:6],
        "quality": {"status": "partial", "issues": ["layout_not_verified"]},
        "latest_attempt_status": snapshot.latest_attempt_status,
    }
    return library, source, AnswerResult("Model interpretation [1]", [record], {})


def inputs(monkeypatch, sequence):
    values = iter(sequence)
    monkeypatch.setattr("builtins.input", lambda _: next(values))


def test_inspection_shows_exact_original_metadata_and_distinguishes_nearby_context(evidence, monkeypatch, capsys, caplog):
    library, _, result = evidence
    inputs(monkeypatch, ["99", "1", "segment", "previous", "previous", "next", "next", "next", "", ""])
    inspect_evidence(library, result)
    output = capsys.readouterr().out
    assert "Choose a displayed" in output
    assert "Model interpretation / search output" in output
    assert "Original supplied passage [1] (not model-generated)" in output
    assert "layout_not_verified" in output
    assert json.dumps(result.evidence[0]["location"]) in output
    assert "NOT evidence used for this answer" in output
    assert "Earlier context." in output and "Later context." in output
    assert "No further source segment" in output
    assert "SECRET" not in caplog.text
    assert result.evidence[0]["text"] == "SECRET"


def test_changed_version_keeps_retained_passage_but_refuses_nearby_navigation(evidence, monkeypatch, capsys):
    library, source, result = evidence
    source.write_text("NEW PRIVATE CONTENT")
    library.ingest(source)
    inputs(monkeypatch, ["1", ""])
    inspect_evidence(library, result)
    output = capsys.readouterr().out
    assert "no longer current" in output and "SECRET" in output
    assert "NEW PRIVATE CONTENT" not in output
    assert "nearby navigation unavailable" in output


def test_uncited_supplied_passages_are_not_presented_as_cited(evidence, monkeypatch, capsys):
    library, _, result = evidence
    result.answer = "Insufficient evidence"
    inputs(monkeypatch, [""])
    inspect_evidence(library, result)
    assert "supplied evidence; not cited" in capsys.readouterr().out


def test_empty_evidence_is_an_explicit_error(evidence):
    with pytest.raises(DocumentQueryError, match="No evidence"):
        inspect_evidence(evidence[0], AnswerResult("No answer", [], {}))


def test_new_failed_ingestion_is_visible_without_mixing_source_versions(evidence, monkeypatch, capsys):
    library, source, result = evidence
    source.write_text("")
    with pytest.raises(ExtractionError, match="no_extractable_text"):
        library.ingest(source)
    inputs(monkeypatch, ["1", "segment", "", ""])
    inspect_evidence(library, result)
    output = capsys.readouterr().out
    assert "Latest ingestion status changed since this result: failed" in output
    assert "SECRET original passage." in output


@pytest.mark.parametrize("location,expected", [
    ({"page": 2}, "page 2"),
    ({"block": 3, "row": 4}, "block 3, row 4"),
    ({"line_start": 5, "line_end": 7}, "line start 5, line end 7"),
    ({}, "segment 2"),
])
def test_search_presentation_retains_order_labels_locations_and_original_data(evidence, location, expected, caplog):
    _, _, result = evidence
    result.evidence[0]["location"] = dict(location, char_start=0, char_end=6)
    result.evidence.append(dict(result.evidence[0], label=2, text="Second passage."))
    before = json.dumps(result.evidence)
    output = render_search_results(result)
    assert "[1] source.txt | version 1 | segment 2" in output
    assert expected in output
    assert output.index("[1]") < output.index("SECRET") < output.index("[2]")
    assert "Second passage." in output
    assert "Candidates only" in output and "not guaranteed exhaustive" in output
    assert "layout_not_verified" in output and "Extraction: partial" in output
    assert "Use inspect" in output
    assert "char_start" not in output
    assert json.dumps(result.evidence) == before
    assert "SECRET" not in caplog.text


def test_search_presentation_wraps_without_truncating_and_flags_failed_ingestion(evidence):
    _, _, result = evidence
    result.evidence[0]["text"] = ("Original passage words. " * 50) + "\nFINAL QUALIFICATION."
    result.evidence[0]["latest_attempt_status"] = "failed"
    output = render_search_results(result)
    assert " ".join(result.evidence[0]["text"].split()) in " ".join(output.split())
    assert "latest ingestion failed" in output
    assert max(len(line) for line in output.splitlines() if line.startswith("    Original")) <= 88


def test_empty_search_is_explicit_and_retains_coverage():
    output = render_search_results(AnswerResult("", [], {"selected_documents": 1, "exhaustive": False}))
    assert "No matching passages returned." in output
    assert '"selected_documents": 1' in output
    assert "Use inspect" not in output
