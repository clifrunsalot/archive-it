import json
import math
import os
import runpy
from pathlib import Path
from unittest.mock import Mock

import chromadb
import httpx
import langchain_ollama
import pytest
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings, FakeEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

import chatbot_tech_doc as app


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "manual.html"
    path.write_text(
        "<table><tr><th>Warning Message</th><th>Procedure</th></tr>"
        "<tr><td>Airbag Fault</td><td>Airbag detected a fault."
        "<p>Contact an authorised repairer.</p></td></tr>"
        "<tr><td>Low Oil Pressure</td><td>Check engine oil level.</td></tr></table>",
        encoding="utf-8",
    )
    return path


def test_row_ingestion_preserves_warning_and_procedure(source: Path):
    docs = app.load_and_split_html(source)
    assert len(docs) == 2
    assert docs[0].page_content == (
        "Warning: Airbag Fault\nProcedure: Airbag detected a fault. "
        "Contact an authorised repairer."
    )
    assert "Low Oil Pressure" not in docs[0].page_content
    assert docs[0].metadata == {"source": "manual.html", "warning": "Airbag Fault"}


@pytest.mark.parametrize("html", [
    "<p>No warning table</p>",
    "<table><tr><td>Fault</td></tr></table>",
    "<table><tr><td>Fault</td><td></td></tr></table>",
])
def test_invalid_source_is_rejected(source: Path, html: str):
    source.write_text(html, encoding="utf-8")
    with pytest.raises(ValueError):
        app.load_and_split_html(source)


def test_real_source_rows():
    docs = app.load_and_split_html(app.SOURCE_PATH)
    for warning in ["Airbag Fault", "Low Oil Pressure"]:
        matches = [doc for doc in docs if doc.metadata["warning"] == warning]
        assert len(matches) == 1
        assert "Procedure:" in matches[0].page_content


def test_stable_ids_and_duplicate_input(tmp_path: Path):
    doc = Document(page_content="Warning", metadata={"source": "manual"})
    store = app.create_vector_store([doc, doc], FakeEmbeddings(size=10), tmp_path)
    ids = store.get()["ids"]
    reopened = app.create_vector_store([doc], FakeEmbeddings(size=10), tmp_path)
    assert reopened.get()["ids"] == ids
    assert len(ids) == 1


def test_index_reuse_does_not_embed_again(source: Path, tmp_path: Path):
    path = tmp_path / "index"
    store = app.get_vector_store(FakeEmbeddings(size=10), path, source_path=source)
    original_ids = store.get()["ids"]
    embedding = Mock(spec=Embeddings)
    embedding.embed_documents.side_effect = AssertionError("Must not re-embed")
    reopened = app.get_vector_store(embedding, path, source_path=source)
    assert reopened.get()["ids"] == original_ids
    embedding.embed_documents.assert_not_called()
    collection = chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False)
    ).get_collection(app.COLLECTION_NAME)
    hnsw = collection.configuration.get("hnsw")
    assert hnsw is not None and hnsw.get("space") == "cosine"
    assert "index_manifest" in collection.metadata


def test_archive_statistics_report_source_when_index_is_missing(source: Path, tmp_path: Path):
    index_path = tmp_path / "not-created"

    statistics = app.get_archive_statistics(index_path, source_path=source)

    assert statistics == {
        "source_bytes": source.stat().st_size,
        "source_documents": 2,
        "source_warnings": 2,
        "indexed_documents": 0,
        "indexed_warnings": 0,
        "index_status": "missing",
        "manifest_matches": None,
        "metric_matches": None,
        "document_ids_match": None,
    }
    assert not index_path.exists()
    report = app.format_archive_statistics(statistics)
    assert "Manifest (source/schema/model): not checked (index missing)" in report
    json_report = json.loads(app.format_archive_statistics_json(statistics))
    assert json_report == statistics


def test_archive_statistics_check_index_against_source(source: Path, tmp_path: Path):
    index_path = tmp_path / "index"
    app.get_vector_store(FakeEmbeddings(size=10), index_path, source_path=source)

    statistics = app.get_archive_statistics(index_path, source_path=source)

    assert statistics["indexed_documents"] == 2
    assert statistics["indexed_warnings"] == 2
    assert statistics["index_status"] == "current"
    assert statistics["manifest_matches"] is True
    assert statistics["metric_matches"] is True
    assert statistics["document_ids_match"] is True
    report = app.format_archive_statistics(statistics)
    assert "Source size:" in report
    assert "Unique source warnings: 2" in report
    assert "Index status: current" in report
    assert "Manifest (source/schema/model): match" in report
    assert "Distance metric (cosine): match" in report
    assert "Document IDs: match" in report

    source.write_text(source.read_text().replace("oil level", "oil level now"))
    changed = app.get_archive_statistics(index_path, source_path=source)
    assert changed["index_status"] == "stale or incompatible"
    assert changed["manifest_matches"] is False
    assert changed["metric_matches"] is True
    assert changed["document_ids_match"] is False


@pytest.mark.parametrize(
    ("change", "expected_checks"),
    [
        ("manifest", (False, True, True)),
        ("records", (True, True, False)),
    ],
)
def test_archive_statistics_identify_index_mismatch(
    source: Path, tmp_path: Path, change: str, expected_checks: tuple[bool, bool, bool],
):
    index_path = tmp_path / "index"
    store = app.get_vector_store(FakeEmbeddings(size=10), index_path, source_path=source)
    client = chromadb.PersistentClient(
        path=str(index_path), settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(app.COLLECTION_NAME)
    if change == "manifest":
        collection.modify(metadata={"index_manifest": "old"})
    else:
        store.delete(ids=[store.get()["ids"][0]])

    statistics = app.get_archive_statistics(index_path, source_path=source)

    assert statistics["index_status"] == "stale or incompatible"
    assert (
        statistics["manifest_matches"],
        statistics["metric_matches"],
        statistics["document_ids_match"],
    ) == expected_checks


@pytest.mark.parametrize(
    ("index_status", "fail_on_stale", "expected_exit"),
    [
        ("current", True, 0),
        ("missing", True, 1),
        ("stale or incompatible", True, 1),
        ("stale or incompatible", False, 0),
    ],
)
def test_stats_cli_exit_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    index_status: str,
    fail_on_stale: bool,
    expected_exit: int,
):
    statistics = {
        "source_bytes": 10,
        "source_documents": 1,
        "source_warnings": 1,
        "indexed_documents": 1,
        "indexed_warnings": 1,
        "index_status": index_status,
        "manifest_matches": None,
        "metric_matches": None,
        "document_ids_match": None,
    }
    monkeypatch.setattr(app, "get_archive_statistics", lambda _: statistics)

    exit_status = app.run_stats_cli(as_json=True, fail_on_stale=fail_on_stale)

    assert exit_status == expected_exit
    assert json.loads(capsys.readouterr().out) == statistics


@pytest.mark.parametrize("change", ["source", "model", "records", "manifest"])
def test_index_mismatch_requires_explicit_rebuild(
    source: Path, tmp_path: Path, change: str
):
    path = tmp_path / "index"
    store = app.get_vector_store(FakeEmbeddings(size=10), path, source_path=source)
    kwargs = {}
    if change == "source":
        source.write_text(source.read_text().replace("oil level", "oil level now"))
    elif change == "model":
        kwargs["embedding_model"] = "different-model"
    elif change == "records":
        store.delete(ids=[store.get()["ids"][0]])
    else:
        client = chromadb.PersistentClient(
            path=str(path), settings=Settings(anonymized_telemetry=False)
        )
        client.get_collection(app.COLLECTION_NAME).modify(metadata={"index_manifest": "old"})
    with pytest.raises(app.IndexMismatchError, match="--rebuild-index"):
        app.get_vector_store(FakeEmbeddings(size=10), path, source_path=source, **kwargs)


def test_rebuild_preserves_unrelated_collection(source: Path, tmp_path: Path):
    path = tmp_path / "index"
    app.get_vector_store(FakeEmbeddings(size=10), path, source_path=source)
    client = chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False)
    )
    client.create_collection("unrelated")
    source.write_text(source.read_text().replace("oil level", "oil level now"))
    store = app.get_vector_store(
        FakeEmbeddings(size=10), path, source_path=source, rebuild=True
    )
    assert len(store.get()["ids"]) == 2
    assert "unrelated" in {collection.name for collection in client.list_collections()}
    assert any("oil level now" in text for text in store.get()["documents"])


def test_legacy_l2_collection_is_rejected(source: Path, tmp_path: Path):
    app.create_vector_store(
        app.load_and_split_html(source), FakeEmbeddings(size=10), tmp_path,
        distance_metric="l2",
    )
    with pytest.raises(app.IndexMismatchError):
        app.get_vector_store(FakeEmbeddings(size=10), tmp_path, source_path=source)


def test_import_creates_no_clients(monkeypatch: pytest.MonkeyPatch):
    embeddings = Mock(side_effect=AssertionError("Import created client"))
    chat = Mock(side_effect=AssertionError("Import created client"))
    monkeypatch.setattr(langchain_ollama, "OllamaEmbeddings", embeddings)
    monkeypatch.setattr(langchain_ollama, "ChatOllama", chat)
    runpy.run_path(app.__file__, run_name="import_test")
    embeddings.assert_not_called()
    chat.assert_not_called()


def test_model_factories_use_local_limits(monkeypatch: pytest.MonkeyPatch):
    chat = Mock()
    embeddings = Mock()
    monkeypatch.setattr(app, "ChatOllama", chat)
    monkeypatch.setattr(app, "OllamaEmbeddings", embeddings)
    assert app.get_llm("qwen3.5:4b") is chat.return_value
    assert app.get_embeddings() is embeddings.return_value
    options = chat.call_args.kwargs
    assert options["model"] == "qwen3.5:4b"
    assert options["base_url"] == app.OLLAMA_BASE_URL
    assert options["num_predict"] == 2048
    assert options["reasoning"] is True
    assert "extract_reasoning" not in options
    assert options["temperature"] == 0.0
    for factory in [chat, embeddings]:
        assert factory.call_args.kwargs["client_kwargs"] == {
            "timeout": app.REQUEST_TIMEOUT_SECONDS, "trust_env": False,
        }


def test_real_chat_client_sends_thinking_and_separates_reasoning(monkeypatch):
    model = app.get_llm("qwen3.5:4b")
    assert isinstance(model, langchain_ollama.ChatOllama)
    chat = Mock(return_value={
        "model": "qwen3.5:4b",
        "message": {
            "role": "assistant",
            "content": "Grounded answer [1]",
            "thinking": "Internal reasoning",
        },
        "done": True,
    })
    monkeypatch.setattr(model._client, "chat", chat)
    response = model.invoke("Airbag fault?", stream=False)
    assert response.content == "Grounded answer [1]"
    assert response.additional_kwargs["reasoning_content"] == "Internal reasoning"
    assert chat.call_args.kwargs["think"] is True
    assert chat.call_args.kwargs["options"]["num_ctx"] == 8192
    assert chat.call_args.kwargs["options"]["num_predict"] == 2048
    assert chat.call_args.kwargs["model"] == "qwen3.5:4b"


@pytest.mark.parametrize("text, expected", [
    ("Normal answer", "Normal answer"),
    ("<think>reasoning\nhere</think> Answer ", "Answer"),
    ("<think>one</think><think>two</think>Answer", "Answer"),
])
def test_output_cleanup(text: str, expected: str):
    assert app.clean_deepseek_output(text) == expected


@pytest.mark.parametrize("text", ["", "  ", "<think>unfinished", "</think>Answer",
                                  "<think>only reasoning</think>", "<THINK>unfinished"])
def test_invalid_output_is_rejected(text: str):
    with pytest.raises(app.ModelOutputError):
        app.clean_deepseek_output(text)


@pytest.mark.parametrize("question", ["", " ", "x" * (app.MAX_QUESTION_CHARS + 1)])
def test_invalid_questions_are_rejected(question: str):
    with pytest.raises(app.InputValidationError):
        app.normalize_chat_input(question)


def test_history_limits_and_roles():
    history = []
    for index in range(8):
        history.extend([HumanMessage(content=str(index)), AIMessage(content="Answer")])
    result = app.normalize_chat_input({"question": "Question", "chat_history": history})
    assert len(result["chat_history"]) == 10
    assert result["chat_history"][0].content == "3"
    oversized = [HumanMessage(content="x" * app.MAX_HISTORY_CHARS),
                 AIMessage(content="Answer")]
    assert app.normalize_chat_input({
        "question": "Question", "chat_history": oversized,
    })["chat_history"] == []
    with pytest.raises(app.InputValidationError):
        app.normalize_chat_input({"question": "Question", "chat_history": [AIMessage(content="x")]})

def test_resource_limits(source: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(app, "MAX_SOURCE_BYTES", 10)
    with pytest.raises(ValueError, match="size limit"):
        app.load_and_split_html(source)
    with pytest.raises(app.ModelOutputError, match="context size limit"):
        app.format_docs([Document(page_content="x" * app.MAX_CONTEXT_CHARS)])
    with pytest.raises(app.ModelOutputError, match="size limit"):
        app.validate_search_question("x" * (app.MAX_QUESTION_CHARS + 1))


def test_main_wires_factories_and_rebuild(monkeypatch: pytest.MonkeyPatch):
    embedding_factory = Mock()
    store_factory = Mock()
    model_factory = Mock()
    chain_factory = Mock()
    monkeypatch.setattr(app, "get_embeddings", embedding_factory)
    monkeypatch.setattr(app, "get_vector_store", store_factory)
    monkeypatch.setattr(app, "get_llm", model_factory)
    monkeypatch.setattr(app, "create_rag_chain", chain_factory)
    assert app.main(
        rebuild=True, model="answer-model", rewrite_model="rewrite-model",
    ) is chain_factory.return_value
    store_factory.assert_called_once_with(
        embedding=embedding_factory.return_value,
        persist_directory=app.PERSIST_DIRECTORY, rebuild=True,
    )
    assert model_factory.call_args_list == [
        (("answer-model",), {}),
        (("rewrite-model",), {"rewrite": True}),
    ]
    store_factory.return_value.as_retriever.assert_called_once_with(**app.RETRIEVER_CONFIG)
    assert chain_factory.call_args.kwargs["rewrite_llm"] is model_factory.return_value
    catalogue = chain_factory.call_args.kwargs["warning_catalogue"]
    load_catalogue = Mock()
    monkeypatch.setattr(app, "load_warning_catalogue", load_catalogue)
    catalogue()
    load_catalogue.assert_called_once_with(store_factory.return_value)


def test_rewrite_client_sends_short_non_reasoning_request(monkeypatch):
    model = app.get_llm("deepseek-r1:8b", rewrite=True)
    assert isinstance(model, langchain_ollama.ChatOllama)
    chat = Mock(return_value={
        "model": "deepseek-r1:8b",
        "message": {"role": "assistant", "content": "What should I do about Airbag Fault?"},
        "done": True, "done_reason": "stop", "eval_count": 12,
    })
    monkeypatch.setattr(model._client, "chat", chat)
    response = model.invoke("Rewrite a question", stream=False)
    assert response.content == "What should I do about Airbag Fault?"
    assert chat.call_args.kwargs["think"] is False
    assert chat.call_args.kwargs["options"]["num_predict"] == app.REWRITE_MAX_TOKENS
    assert response.response_metadata["done_reason"] == "stop"


@pytest.mark.parametrize("question", [
    "/warnings", "list all of the possible warnings", "List all warnings!",
    "what are the possible warnings?", "show me all warning messages",
    "please list all faults in the manual", "list warnings",
])
def test_catalogue_requests_return_all_records_without_models(source, tmp_path, question):
    store = app.get_vector_store(FakeEmbeddings(size=10), tmp_path, source_path=source)
    retrieve, answer, rewrite = Mock(), Mock(), Mock()
    chain = app.create_rag_chain(
        RunnableLambda(retrieve), RunnableLambda(answer),
        rewrite_llm=RunnableLambda(rewrite),
        warning_catalogue=lambda: app.load_warning_catalogue(store),
    )
    result = chain.invoke(question)
    assert result.startswith("All indexed warnings (2 entries):")
    assert "1. Airbag Fault - Source: manual.html" in result
    assert "2. Low Oil Pressure - Source: manual.html" in result
    retrieve.assert_not_called()
    answer.assert_not_called()
    rewrite.assert_not_called()


@pytest.mark.parametrize("question", [
    "There are other faults. list them too.", "list the other faults",
    "show the others", "list the remaining warnings", "list them too",
])
def test_catalogue_follow_up_bypasses_rewrite(question):
    retrieve, answer, rewrite = Mock(), Mock(), Mock()
    catalogue = Mock(return_value=[Document(page_content="", metadata={
        "warning": "Airbag Fault", "source": "manual.html",
    })])
    chain = app.create_rag_chain(
        RunnableLambda(retrieve), RunnableLambda(answer),
        rewrite_llm=RunnableLambda(rewrite), warning_catalogue=catalogue,
    )
    result = chain.invoke({
        "question": question, "chat_history": [
            HumanMessage(content="list all of the possible warnings"),
            AIMessage(content="Only two retrieved warnings"),
        ],
    })
    assert "Airbag Fault" in result
    catalogue.assert_called_once()
    retrieve.assert_not_called()
    answer.assert_not_called()
    rewrite.assert_not_called()


@pytest.mark.parametrize("question", [
    "list all engine faults", "list warnings about oil",
    "what should I do about an airbag fault?", "list insurance policies",
])
def test_filtered_or_unrelated_questions_do_not_route_to_catalogue(question):
    assert not app.wants_warning_catalogue({"question": question, "chat_history": []})


def test_unrelated_list_follow_up_not_misrouted():
    assert not app.wants_warning_catalogue({
        "question": "list them too", "chat_history": [
            HumanMessage(content="What are the insurance policies?"),
            AIMessage(content="Two policies"),
        ],
    })


def test_catalogue_metadata_validation_and_empty_index():
    for result in [
        {"ids": ["1"], "metadatas": None},
        {"ids": ["1"], "metadatas": []},
        {"ids": ["1"], "metadatas": [None]},
        {"ids": ["1"], "metadatas": [{"warning": "Fault", "source": ""}]},
        {"ids": ["1"], "metadatas": [{"warning": 42, "source": "manual"}]},
    ]:
        with pytest.raises(app.IndexMismatchError):
            app.load_warning_catalogue(Mock(get=Mock(return_value=result)))
    assert app.format_warning_catalogue([]) == "No warnings are indexed."


def test_catalogue_preserves_same_warning_from_different_sources():
    docs = [
        Document(page_content="", metadata={"warning": "Fault", "source": source})
        for source in ["b.html", "a.html", "a.html"]
    ]
    result = app.format_warning_catalogue(docs)
    assert result.startswith("All indexed warnings (2 entries):")
    assert result.count("Source: a.html") == 1
    assert result.count("Source: b.html") == 1


def test_separate_rewrite_model_and_bounded_validation():
    retrieve = Mock(return_value=[Document(page_content="Procedure")])
    rewrite = Mock(return_value=AIMessage(content="Airbag fault procedure?"))
    answer = Mock(return_value=AIMessage(content="Contact a repairer"))
    chain = app.create_rag_chain(
        RunnableLambda(retrieve), RunnableLambda(answer), rewrite_llm=RunnableLambda(rewrite),
    )
    result = chain.invoke({
        "question": "What should I do?", "chat_history": [
            HumanMessage(content="Airbag fault?"), AIMessage(content="Fault detected"),
        ],
    })
    assert result.startswith("Contact a repairer")
    rewrite.assert_called_once()
    answer.assert_called_once()
    retrieve.assert_called_once_with("Airbag fault procedure?")


@pytest.mark.parametrize("content, metadata", [
    ("x" * (app.MAX_REWRITE_CHARS + 1), {}),
    ("Question?\nHere is another question.", {}),
    ("A plausible but truncated question", {"done_reason": "length"}),
    ("", {"done_reason": "length"}),
])
def test_invalid_rewrite_stops_before_retrieval(content, metadata):
    retrieve, answer = Mock(), Mock()
    rewrite = RunnableLambda(lambda _: AIMessage(content=content, response_metadata=metadata))
    chain = app.create_rag_chain(
        RunnableLambda(retrieve), RunnableLambda(answer), rewrite_llm=rewrite,
    )
    with pytest.raises(app.ModelOutputError):
        chain.invoke({
            "question": "What about it?", "chat_history": [
                HumanMessage(content="Airbag fault?"), AIMessage(content="Fault detected"),
            ],
        })
    retrieve.assert_not_called()
    answer.assert_not_called()


def test_rewrite_boundary_and_reasoning_cleanup():
    assert app.validate_search_question("x" * app.MAX_REWRITE_CHARS) == "x" * app.MAX_REWRITE_CHARS
    assert app.validate_search_question(
        "<think>" + "r" * 2500 + "</think>Airbag fault procedure?"
    ) == "Airbag fault procedure?"


def test_complete_real_warning_catalogue(tmp_path):
    docs = app.load_and_split_html(app.SOURCE_PATH)
    store = app.create_vector_store(docs, FakeEmbeddings(size=10), tmp_path)
    chain = app.create_rag_chain(
        RunnableLambda(Mock(side_effect=AssertionError("Must not retrieve"))),
        RunnableLambda(Mock(side_effect=AssertionError("Must not generate"))),
        warning_catalogue=lambda: app.load_warning_catalogue(store),
    )
    question = "list all of the possible warnings"
    answer = chain.invoke(question)
    expected = {(doc.metadata["warning"], doc.metadata["source"]) for doc in docs}
    assert len(expected) == 32
    assert answer.startswith("All indexed warnings (32 entries):")
    for warning, source in expected:
        assert f"{warning} - Source: {source}" in answer
    assert chain.invoke({
        "question": "There are other faults. list them too.",
        "chat_history": [HumanMessage(content=question), AIMessage(content=answer)],
    }) == answer


@pytest.mark.skipif(os.environ.get("RUN_OLLAMA_TESTS") != "1",
                    reason="Set RUN_OLLAMA_TESTS=1 for live local rewrite evaluation")
def test_live_local_bounded_rewrite():
    retrieve = Mock(return_value=[Document(page_content="Contact an authorised repairer.")])
    chain = app.create_rag_chain(
        RunnableLambda(retrieve), RunnableLambda(lambda _: AIMessage(content="Contact a repairer")),
        rewrite_llm=app.get_llm(app.REWRITE_MODEL, rewrite=True),
    )
    answer = chain.invoke({
        "question": "What should I do about it?",
        "chat_history": [
            HumanMessage(content="What does Airbag Fault mean?"),
            AIMessage(content="It indicates an airbag fault."),
        ],
    })
    assert answer.startswith("Contact a repairer")
    rewritten = retrieve.call_args.args[0]
    assert 0 < len(rewritten) <= app.MAX_REWRITE_CHARS
    assert len(rewritten.splitlines()) == 1
    assert "airbag" in rewritten.casefold()


@pytest.mark.parametrize("error", [
    app.IndexMismatchError("Run with --rebuild-index"),
    httpx.ConnectError("private backend response"),
])
def test_startup_errors_are_reported_without_prompting(monkeypatch, capsys, error):
    monkeypatch.setattr(app, "main", Mock(side_effect=error))
    user_input = Mock()
    monkeypatch.setattr("builtins.input", user_input)
    app.run_cli()
    user_input.assert_not_called()
    captured = capsys.readouterr()
    assert captured.err
    assert "private backend response" not in captured.err


def test_first_question_bypasses_rewrite_and_includes_sources():
    docs = [Document(page_content="Manual procedure",
                     metadata={"source": "manual.html", "warning": "Airbag Fault"})]
    retrieve = Mock(return_value=docs)
    generate = Mock(return_value=AIMessage(content="Grounded answer [1]"))
    chain = app.create_rag_chain(RunnableLambda(retrieve), RunnableLambda(generate))
    answer = chain.invoke("Airbag fault?")
    assert answer.startswith("Grounded answer [1]")
    assert "[1] manual.html - Airbag Fault" in answer
    generate.assert_called_once()
    retrieve.assert_called_once_with("Airbag fault?")
    messages = generate.call_args.args[0].to_messages()
    assert [message.type for message in messages] == ["system", "human"]
    assert app.format_docs(docs) in messages[-1].content


def test_empty_context_skips_generation():
    generate = Mock()
    chain = app.create_rag_chain(RunnableLambda(lambda _: []), RunnableLambda(generate))
    assert chain.invoke("Unrelated question?") == app.NO_CONTEXT_ANSWER
    generate.assert_not_called()


def test_follow_up_preserves_question_and_retrieves_rewritten_query():
    history = [HumanMessage(content="Airbag fault?"), AIMessage(content="Fault detected")]
    generate = Mock(side_effect=[
        AIMessage(content="<think>rewrite</think>Airbag fault procedure?"),
        AIMessage(content="Contact a repairer [1]"),
    ])
    retrieve = Mock(return_value=[Document(page_content="Contact a repairer")])
    chain = app.create_rag_chain(RunnableLambda(retrieve), RunnableLambda(generate))
    assert chain.invoke({
        "question": "What should I do about it?", "chat_history": history,
    }).startswith("Contact a repairer")
    retrieve.assert_called_once_with("Airbag fault procedure?")
    for call in generate.call_args_list:
        assert call.args[0].to_messages()[1:-1] == history
    assert "What should I do about it?" in generate.call_args.args[0].to_messages()[-1].content


def test_empty_rewrite_stops_before_retrieval():
    retrieve = Mock()
    chain = app.create_rag_chain(
        RunnableLambda(retrieve),
        RunnableLambda(lambda _: AIMessage(content="<think>only reasoning</think>")),
    )
    with pytest.raises(app.ModelOutputError):
        chain.invoke({"question": "What about it?", "chat_history": [
            HumanMessage(content="Airbag"), AIMessage(content="Fault"),
        ]})
    retrieve.assert_not_called()


THRESHOLD_DOCUMENT = "Low oil pressure indicates that engine oil pressure is too low."
THRESHOLD_QUESTIONS = {
    "What does low oil pressure mean?": 0.0,
    "What does the engine oil pressure warning indicate?": 0.5,
    "What does the low oil pressure warning mean?": math.sqrt(0.3092 * math.sqrt(2)),
    "What is the weather tomorrow?": 1.0,
}


class ThresholdEmbeddings(Embeddings):
    """Synthetic distances test filtering, not real semantic relevance."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, THRESHOLD_QUESTIONS[text]]


@pytest.fixture
def threshold_store(tmp_path: Path):
    return app.create_vector_store(
        [Document(page_content=THRESHOLD_DOCUMENT)], ThresholdEmbeddings(), tmp_path,
        distance_metric="l2",
    )


@pytest.mark.parametrize("question, threshold, match", [
    ("What does low oil pressure mean?", 0.65, True),
    ("What does low oil pressure mean?", 0.95, True),
    ("What does the engine oil pressure warning indicate?", 0.7, True),
    ("What does the engine oil pressure warning indicate?", 0.85, False),
    ("What does the low oil pressure warning mean?", 0.65, True),
    ("What does the low oil pressure warning mean?", 0.7, False),
    ("What is the weather tomorrow?", 0.2, True),
    ("What is the weather tomorrow?", 0.35, False),
    ("What is the weather tomorrow?", 0.65, False),
    ("What is the weather tomorrow?", 0.7, False),
])
def test_threshold_context(threshold_store, question, threshold, match):
    score = threshold_store.similarity_search_with_relevance_scores(question, k=1)[0][1]
    assert score == pytest.approx(
        1 - THRESHOLD_QUESTIONS[question] ** 2 / math.sqrt(2), abs=1e-6
    )
    retrieve = threshold_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 1, "score_threshold": threshold},
    )
    assert bool(retrieve.invoke(question)) == match
    generate = Mock(return_value=AIMessage(content="Test answer"))
    answer = app.create_rag_chain(retrieve, RunnableLambda(generate)).invoke(question)
    assert answer.startswith("Test answer") if match else answer == app.NO_CONTEXT_ANSWER
    assert generate.call_count == int(match)


@pytest.mark.parametrize("offset, match", [(-1e-6, True), (0.0, True), (1e-6, False)])
def test_threshold_boundary(threshold_store, offset, match):
    question = "What does the low oil pressure warning mean?"
    score = threshold_store.similarity_search_with_relevance_scores(question, k=1)[0][1]
    retrieve = threshold_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 1, "score_threshold": score + offset},
    )
    assert bool(retrieve.invoke(question)) == match


@pytest.mark.parametrize("ending", ["exit", " QuIt ", EOFError(), KeyboardInterrupt()])
def test_cli_loop_and_exit(monkeypatch, capsys, ending):
    chain = Mock()
    chain.invoke.side_effect = ["First answer", "Second answer"]
    build = Mock(return_value=chain)
    monkeypatch.setattr(app, "main", build)
    monkeypatch.setattr("builtins.input", Mock(side_effect=["", "First question", "Follow-up", ending]))
    app.run_cli()
    build.assert_called_once_with(
        rebuild=False, model=app.CHAT_MODEL, rewrite_model=app.REWRITE_MODEL,
    )
    assert chain.invoke.call_count == 2
    assert chain.invoke.call_args_list[0].args[0]["chat_history"] == []
    assert chain.invoke.call_args_list[1].args[0]["chat_history"] == [
        HumanMessage(content="First question"), AIMessage(content="First answer"),
    ]
    captured = capsys.readouterr()
    assert "non-empty question" in captured.err
    assert captured.out.endswith("Goodbye.\n")


@pytest.mark.parametrize("phase", ["startup", "generation"])
@pytest.mark.parametrize("interrupt", [EOFError, KeyboardInterrupt])
def test_cli_interruptions(monkeypatch, capsys, phase, interrupt):
    chain = Mock()
    chain.invoke.side_effect = interrupt()
    build = Mock(side_effect=interrupt()) if phase == "startup" else Mock(return_value=chain)
    monkeypatch.setattr(app, "main", build)
    monkeypatch.setattr("builtins.input", Mock(return_value="Question"))
    app.run_cli()
    assert capsys.readouterr().out.endswith("Goodbye.\n")


@pytest.mark.parametrize("failure", [
    httpx.ConnectError("backend details"), app.ModelOutputError("No final answer"),
])
def test_failed_request_keeps_history_and_session(monkeypatch, capsys, failure):
    chain = Mock()
    chain.invoke.side_effect = ["Good answer", failure, "Next answer"]
    monkeypatch.setattr(app, "main", Mock(return_value=chain))
    monkeypatch.setattr("builtins.input", Mock(side_effect=["First", "Second", "Third", "exit"]))
    app.run_cli()
    assert chain.invoke.call_args_list[1].args[0]["chat_history"] == (
        chain.invoke.call_args_list[2].args[0]["chat_history"]
    )
    assert "backend details" not in capsys.readouterr().err


def test_unexpected_errors_propagate(monkeypatch):
    chain = Mock()
    chain.invoke.side_effect = RuntimeError("Programming bug")
    monkeypatch.setattr(app, "main", Mock(return_value=chain))
    monkeypatch.setattr("builtins.input", Mock(return_value="Question"))
    with pytest.raises(RuntimeError, match="Programming bug"):
        app.run_cli()


def test_cli_sessions_are_isolated_and_history_bounded(monkeypatch):
    chain = Mock()
    chain.invoke.return_value = "Answer"
    monkeypatch.setattr(app, "main", Mock(return_value=chain))
    monkeypatch.setattr("builtins.input", Mock(
        side_effect=[f"Question {index}" for index in range(7)] + ["exit", "New", "exit"]
    ))
    app.run_cli()
    app.run_cli()
    assert len(chain.invoke.call_args_list[6].args[0]["chat_history"]) == 10
    assert chain.invoke.call_args_list[7].args[0]["chat_history"] == []


@pytest.mark.skipif(
    __import__("os").getenv("RUN_OLLAMA_TESTS") != "1",
    reason="Set RUN_OLLAMA_TESTS=1 for live local embedding evaluation",
)
@pytest.mark.parametrize("question, warning", [
    ("What does the low oil pressure warning mean?", "Low Oil Pressure"),
    ("What does the airbag fault light mean?", "Airbag Fault"),
])
def test_live_local_retrieval(tmp_path, question, warning):
    store = app.get_vector_store(app.get_embeddings(), tmp_path)
    results = store.similarity_search(question, k=2)
    assert warning in {doc.metadata["warning"] for doc in results}
