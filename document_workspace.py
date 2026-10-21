"""Guided document-scope navigation over existing review and query services."""

import json
import sys
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

import httpx
from ollama import ResponseError

from app_logging import log_failure
from document_fact_ui import (
    CancelDraft, calculate_facts, prompt, run_fact_workbench, select_numbers,
)
from document_facts import list_facts
from document_library import DocumentLibrary, DocumentQueryError
from document_query import QuerySession, render_answer
from document_evidence_ui import inspect_evidence, render_search_results
from document_progress import OperationProgress, model_name
from document_search import AnswerResult


def menu(options: tuple[str, ...]) -> str:
    for number, option in enumerate(options, 1):
        print(f"{number}. {option}")
    while True:
        value = prompt("Action (number or name; Enter returns)", options[-1])
        if value in options:
            return value
        if value.isascii() and value.isdecimal() and len(value) <= 2:
            number = int(value)
            if 1 <= number <= len(options):
                return options[number - 1]
        print("Choose a displayed number or action name.")


def select_document(library: DocumentLibrary) -> str | None:
    documents = library.list_documents()
    if not documents:
        print("No documents. Use ingest first, then reopen the workspace.")
        return None
    for number, document in enumerate(documents, 1):
        print(f"{number}. {document['source_path']} | {document['latest_attempt_status']}")
    while True:
        value = prompt("Document number or exact filename (Enter exits)")
        if not value:
            return None
        matches = [entry for entry in documents if Path(entry["source_path"]).name == value]
        if value.isascii() and value.isdecimal() and len(value) <= len(str(len(documents))):
            number = int(value)
            if 1 <= number <= len(documents):
                matches = [documents[number - 1]]
        if len(matches) == 1:
            return matches[0]["id"]
        print("No unique match. Use a displayed number; duplicate filenames require a number.")


def select_scope(library: DocumentLibrary) -> list[str] | None:
    documents = library.list_documents()
    if not documents:
        print("No documents are available for selection.")
        return None
    for number, document in enumerate(documents, 1):
        print(f"{number}. {document['source_path']} | {document['latest_attempt_status']}")
    while True:
        value = prompt("Document numbers for this scope (comma-separated; Enter cancels)")
        if not value:
            return None
        parts = value.split(",")
        if (len(value) <= 1024 and parts
                and all(part.strip().isascii() and part.strip().isdecimal()
                        and len(part.strip()) <= len(str(len(documents)))
                        and 1 <= int(part.strip()) <= len(documents) for part in parts)
                and len({int(part.strip()) for part in parts}) == len(parts)):
            selected = [documents[int(part.strip()) - 1] for part in parts]
            unavailable = [entry["source_path"] for entry in selected
                           if entry["current_version"] is None]
            if unavailable:
                print("Every selected document must have a successful extraction. "
                      "Remove unavailable entries and try again.")
                continue
            return [entry["id"] for entry in selected]
        print("Choose distinct displayed document numbers separated by commas.")


def show_status(library: DocumentLibrary, identifier: str) -> None:
    document = next(entry for entry in library.list_documents() if entry["id"] == identifier)
    print("\nSelected:", document["source_path"])
    print("Extraction:", json.dumps({
        "version": document["current_version"], "latest_attempt": document["latest_attempt_status"],
        "reason": document["latest_attempt_reason"], "quality": document["quality"],
    }, ensure_ascii=False))
    with closing(library.connect()) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='search_versions'",
        ).fetchone()
        indexed = connection.execute(
            "SELECT passage_count FROM search_versions WHERE version_id=?",
            (document["current_version"],),
        ).fetchone() if exists else None
    print("Index:", f"stored for current version ({indexed[0]} passages); compatibility checked on search"
          if indexed else "current version not indexed")
    facts = list_facts(library, [identifier]) if document["current_version"] is not None else []
    print("Facts:", json.dumps({
        "total": len(facts),
        **{decision: sum(fact["decision"] == decision for fact in facts)
           for decision in ("confirmed", "unreviewed", "rejected")},
        "stale": sum(fact["stale"] for fact in facts),
        "eligible": sum(fact["usable_for_queries"] and not fact["payload"]["uncertainty"] for fact in facts),
    }))
    print("Automatic enumeration is experimental; reviewed facts do not prove completeness.")


def run_workspace(
    library: DocumentLibrary, create_session: Callable[[str], QuerySession],
) -> None:
    identifier = None
    scope_ids: list[str] = []
    session = None
    result: AnswerResult | None = None
    print("Document workspace. Local private output; /cancel abandons a prompt; "
          "Ctrl+C cancels an active action.")
    try:
        while True:
            try:
                if identifier is None:
                    identifier = select_document(library)
                    if identifier is None:
                        return
                    scope_ids = [identifier]
                documents = {entry["id"]: entry for entry in library.list_documents()}
                unavailable = [documents[selected_id]["source_path"] for selected_id in scope_ids
                               if documents[selected_id]["current_version"] is None]
                if unavailable:
                    print("A selected document has no successful extraction or no longer has one; "
                          "choose another scope.")
                    identifier = None
                    scope_ids = []
                    session = None
                    continue
                print(f"\nActive scope ({len(scope_ids)} document(s)):")
                for selected_id in scope_ids:
                    show_status(library, selected_id)
                action = menu(("index", "search", "ask", "summarize", "compare", "scope",
                               "facts", "calculate", "approve", "selection", "cancel-query",
                               "documents", "inspect", "quit"))
                if action == "quit":
                    return
                if action == "inspect":
                    if result is None:
                        raise DocumentQueryError(
                            "No completed evidence result. Choose search, ask, summarize or compare first.")
                    inspect_evidence(library, result)
                    continue
                if action == "scope":
                    selected_scope = select_scope(library)
                    if selected_scope is None:
                        continue
                    identifier = selected_scope[0]
                    scope_ids = selected_scope
                    if session is not None:
                        session.select(scope_ids)
                    result = None
                    continue
                if action == "documents":
                    if session is not None:
                        session.pending_fact_query = None
                    identifier = None
                    scope_ids = []
                    session = None
                    result = None
                    continue
                if action in ("approve", "selection", "cancel-query"):
                    if session is None or session.pending_fact_query is None:
                        raise DocumentQueryError("No pending fact preview. Choose ask first.")
                    pending = session.pending_fact_query
                    if action == "cancel-query":
                        print(session.fact_command("/facts-cancel"))
                    elif action == "selection":
                        numbers = select_numbers(len(pending["catalogue"]))
                        print(session.fact_command("/facts-select " + ",".join(map(str, numbers))))
                    else:
                        print(session.render_fact_preview())
                        if not pending["preview"]["ready"]:
                            print("Approval unavailable: resolve blocking reasons or change selection.")
                            continue
                        if menu(("run", "cancel")) == "run":
                            print(session.fact_command("/facts-run " + pending["preview"]["preview_token"]))
                        else:
                            print(session.fact_command("/facts-cancel"))
                    continue
                if session is not None:
                    session.pending_fact_query = None
                result = None
                if action == "facts":
                    if len(scope_ids) != 1:
                        raise DocumentQueryError("Fact editing requires a single-document scope. "
                                                 "Use scope to select one document.")
                    run_fact_workbench(library, identifier)
                elif action == "calculate":
                    if len(scope_ids) != 1:
                        raise DocumentQueryError("Manual calculation requires a single-document scope. "
                                                 "Use scope to select one document.")
                    calculate_facts(library, identifier)
                else:
                    if session is None:
                        session = create_session(identifier)
                        if len(scope_ids) > 1:
                            session.select(scope_ids)
                    elif session.document_ids != scope_ids:
                        session.select(scope_ids)
                    question = prompt("Search question" if action == "search" else
                                      "Question" if action == "ask" else
                                      "Comparison request" if action == "compare" else
                                      "Summary request") if action != "index" else ""
                    if action == "ask" and question.startswith("/"):
                        raise DocumentQueryError("Use workspace actions for approval and document selection.")
                    active_model = session.search.embedding if action in ("index", "search") else session.model
                    with OperationProgress(action, model_name(active_model)):
                        if action == "index":
                            print(json.dumps(session.search.index(scope_ids)))
                        elif action == "search":
                            found = session.search.search(question, scope_ids)
                            result = AnswerResult(
                                "Retrieved passages (search output, not a generated answer).",
                                [dict(record, label=index) for index, record in
                                 enumerate(found["passages"], 1)], found.get("coverage", {}),
                            )
                            print(render_search_results(result))
                        elif action == "ask":
                            previous = session.previous_result
                            previous_plan = session.previous_plan
                            print(session.execute(question))
                            if (session.previous_result is not previous
                                    or session.previous_plan is not previous_plan):
                                result = session.previous_result
                        else:
                            from document_synthesis import synthesize
                            completed = synthesize(
                                library, session.model, action, question, scope_ids,
                            )
                            print(render_answer(completed, "prose"))
                            result = completed
            except (KeyboardInterrupt, EOFError) as error:
                if session is not None:
                    session.pending_fact_query = None
                    session.previous_result = None
                    session.previous_plan = None
                    session.previous_versions = {}
                result = None
                print("\nAction interrupted. No incomplete result published. Saved changes and "
                      "validated checkpoints retained; pending approval cleared. "
                      "The local server may still finish an already submitted request.")
                if isinstance(error, EOFError):
                    return
            except CancelDraft:
                if session is not None:
                    session.pending_fact_query = None
                    session.previous_result = None
                    session.previous_plan = None
                    session.previous_versions = {}
                result = None
                print("Action cancelled. Explicitly saved changes are retained; pending approval cleared.")
            except DocumentQueryError as error:
                log_failure("Document workspace action", error)
                print(str(error), file=sys.stderr)
            except (httpx.HTTPError, ConnectionError, ResponseError) as error:
                log_failure("Document workspace service", error)
                print("Local Ollama request failed. Check service, models and timeout.", file=sys.stderr)
    except (EOFError, KeyboardInterrupt):
        print("\nWorkspace closed. Explicitly saved changes are retained; pending approval discarded.")
    finally:
        if session is not None:
            session.pending_fact_query = None
