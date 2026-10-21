"""Local, versioned document catalogue and provenance-preserving extraction."""

import argparse
import hashlib
import io
import json
import logging
import sqlite3
import sys
import warnings
import zipfile
from contextlib import closing
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import httpx
from docx import Document as WordDocument
from docx.oxml.exceptions import InvalidXmlError
from docx.table import Table
from lxml.etree import XMLSyntaxError
from pypdf import PdfReader
from pypdf.errors import PyPdfError
from ollama import ResponseError

from app_logging import LOGGER, configure_logging, log_failure
from document_ocr import LocalExtractionError, engine_version, extract_local

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATABASE = PROJECT_DIR / "data/document_library/catalogue.sqlite3"
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_PDF_FILE_BYTES = 160 * 1024 * 1024
MAX_EXPANDED_BYTES = 100 * 1024 * 1024
MAX_TEXT_CHARS = 2_000_000
MAX_PAGES = 1000
IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
FORMATS = {".pdf", ".docx", ".txt"} | IMAGE_FORMATS
SCHEMA_VERSION = 1


class ExtractionError(ValueError):
    """An expected extraction failure with a safe, fixed reason code."""


class DocumentQueryError(ValueError):
    """A fixed, actionable search error safe to show without personal content."""


@dataclass
class Segment:
    text: str
    location: dict[str, str | int]
    kind: str = "paragraph"


@dataclass
class Extraction:
    segments: list[Segment]
    issues: list[str]
    status: str
    ocr_used: bool = False


@dataclass
class DocumentSnapshot:
    document_id: str
    version_id: int
    source_path: str
    sha256: str
    extraction: dict
    latest_attempt_status: str


def extractor_version(suffix: str) -> str:
    dependency = {".pdf": "pypdf", ".docx": "python-docx"}.get(suffix)
    profile = f"extraction-v3:{suffix}:{version(dependency) if dependency else 'utf8-v1'}"
    if suffix == ".pdf" or suffix in IMAGE_FORMATS:
        profile += (f":pdfium-{version('pypdfium2')}:pillow-{version('Pillow')}"
                    f":ocr-eng-{engine_version()}")
    if suffix == ".pdf":
        profile += f":fonts-{version('fonttools')}"
    return profile


class ParserWarnings(logging.Handler):
    """Count parser diagnostics without exposing arbitrary source text."""

    def __init__(self) -> None:
        super().__init__()
        self.counts: dict[str, int] = {}

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith("Superfluous whitespace found in object header"):
            code = "pdf_object_header_whitespace"
        elif "fontTools" in message:
            code = "pdf_font_decoding_requires_review"
        else:
            code = "parser_diagnostics_require_review"
        self.counts[code] = self.counts.get(code, 0) + 1


def repeated_page_text(segments: list[Segment]) -> bool:
    texts = [" ".join(segment.text.split()) for segment in segments]
    return len(texts) != len(set(texts))


def repeated_lines_across_pages(segments: list[Segment]) -> bool:
    page_lines: dict[int, set[str]] = {}
    page_text: dict[int, list[str]] = {}
    for segment in segments:
        page = segment.location.get("page")
        if not isinstance(page, int):
            continue
        lines = page_lines.setdefault(page, set())
        page_text.setdefault(page, []).append(segment.text.casefold())
        lines.update(
            normalized for line in segment.text.splitlines()
            if len(normalized := " ".join(line.casefold().split())) >= 8
        )
    occurrences: dict[str, int] = {}
    for lines in page_lines.values():
        for line in lines:
            occurrences[line] = occurrences.get(line, 0) + 1
    if any(count >= 2 for count in occurrences.values()):
        return True
    # Some PDF text extractors concatenate separately positioned header/body
    # text. Retain it, but flag a repeated page-edge phrase for source review.
    prefixes: set[str] = set()
    suffixes: set[str] = set()
    for fragments in page_text.values():
        text = " ".join(fragments)
        if len(text) >= 12:
            if text[:12] in prefixes or text[-12:] in suffixes:
                return True
            prefixes.add(text[:12])
            suffixes.add(text[-12:])
    return False


def extract_document(data: bytes, suffix: str) -> Extraction:
    segments: list[Segment] = []
    issues: list[str] = []
    total_chars = 0
    ocr_used = False

    def add(text: str, location: dict[str, str | int], kind: str = "paragraph") -> None:
        nonlocal total_chars
        if not text.strip():
            return
        total_chars += len(text)
        if total_chars > MAX_TEXT_CHARS:
            raise ExtractionError("extracted_text_limit")
        segments.append(Segment(text, location, kind))

    if suffix == ".txt":
        try:
            text = data.decode("utf-8-sig")
        except UnicodeError as error:
            raise ExtractionError("invalid_utf8") from error
        if "\x00" in text:
            raise ExtractionError("binary_text")
        lines = text.splitlines()
        start = 0
        for index in range(len(lines) + 1):
            if index == len(lines) or not lines[index].strip():
                if index > start:
                    add("\n".join(lines[start:index]), {
                        "line_start": start + 1, "line_end": index,
                    })
                start = index + 1
    elif suffix == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise ExtractionError("invalid_pdf_signature")
        reader = PdfReader(io.BytesIO(data), strict=True)
        if reader.is_encrypted:
            raise ExtractionError("encrypted_pdf")
        if len(reader.pages) > MAX_PAGES:
            raise ExtractionError("page_limit")
        missing_text = False
        native_overflow = False
        for number, page in enumerate(reader.pages, 1):
            stream = page.get_contents()
            if stream is not None and len(stream.get_data()) > MAX_EXPANDED_BYTES:
                raise ExtractionError("pdf_page_content_limit")
            text = page.extract_text() or ""
            if not text.strip():
                missing_text = True
            if total_chars + len(text) > MAX_TEXT_CHARS:
                native_overflow = True
            else:
                add(text, {"page": number}, "page")
        repeated = repeated_page_text(segments)
        if repeated or missing_text or native_overflow:
            LOGGER.info("PDF page-bounded extraction requested repeated_text=%s missing_text=%s "
                        "native_text_limit=%s", repeated, missing_text, native_overflow)
            repaired = extract_local(data, suffix, native_pdf=True)
            if len(repaired["pages"]) != len(reader.pages):
                raise ExtractionError("pdf_page_count_mismatch")
            segments.clear()
            total_chars = 0
            for number, extracted in enumerate(repaired["pages"], 1):
                if not extracted["text"].strip():
                    issues.append(f"page_{number}_no_extractable_text")
                location: dict[str, str | int] = {"page": number}
                if extracted["ocr"]:
                    ocr_used = True
                    location["method"] = "ocr_tesseract_eng"
                add(extracted["text"], location, "page")
            if repeated or native_overflow:
                issues.append("pdf_page_bounds_reextracted")
            if repeated_page_text(segments):
                issues.append("pdf_repeated_page_text_requires_review")
        if (not any(issue == "pdf_repeated_page_text_requires_review" for issue in issues)
                and repeated_lines_across_pages(segments)):
            issues.append("pdf_repeated_page_lines_requires_review")
        issues.append("pdf_layout_and_tables_not_verified")
    elif suffix in IMAGE_FORMATS:
        result = extract_local(data, suffix)
        add(result["pages"][0]["text"], {"image": 1, "method": "ocr_tesseract_eng"}, "image")
        ocr_used = True
    elif suffix == ".docx":
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = archive.infolist()
            if len(members) > 10000 or sum(item.file_size for item in members) > MAX_EXPANDED_BYTES:
                raise ExtractionError("docx_expansion_limit")
            if "word/document.xml" not in archive.namelist():
                raise ExtractionError("invalid_docx_package")
            if any(name.endswith("vbaProject.bin") for name in archive.namelist()):
                raise ExtractionError("macro_document_not_supported")
            if any(name.startswith("word/media/") for name in archive.namelist()):
                issues.append("images_not_extracted")
        document = WordDocument(io.BytesIO(data))
        heading = ""
        for number, block in enumerate(document.iter_inner_content(), 1):
            if isinstance(block, Table):
                for row_number, row in enumerate(block.rows, 1):
                    add(json.dumps([cell.text for cell in row.cells], ensure_ascii=False),
                        {"block": number, "row": row_number, "heading": heading}, "table_row")
            else:
                if block.style is not None and (block.style.name or "").startswith("Heading"):
                    heading = block.text
                add(block.text, {"block": number, "heading": heading})
        seen_parts: set[str] = set()
        for section_number, section in enumerate(document.sections, 1):
            areas = [("header", section.header), ("footer", section.footer)]
            if section.different_first_page_header_footer:
                areas.extend([
                    ("first_page_header", section.first_page_header),
                    ("first_page_footer", section.first_page_footer),
                ])
            if document.settings.odd_and_even_pages_header_footer:
                areas.extend([
                    ("even_page_header", section.even_page_header),
                    ("even_page_footer", section.even_page_footer),
                ])
            for area, container in areas:
                part_name = str(container.part.partname)
                if part_name in seen_parts:
                    continue
                seen_parts.add(part_name)
                for block_number, block in enumerate(container.iter_inner_content(), 1):
                    location = {
                        "section": section_number, "area": area, "block": block_number,
                    }
                    if isinstance(block, Table):
                        for row_number, row in enumerate(block.rows, 1):
                            add(json.dumps([cell.text for cell in row.cells], ensure_ascii=False),
                                {**location, "row": row_number}, "table_row")
                    else:
                        add(block.text, location)
        issues.append("docx_textboxes_revisions_nested_tables_not_verified")
    else:
        raise ExtractionError("unsupported_format")
    if not segments:
        raise ExtractionError("no_extractable_text_ocr_or_review_required")
    if ocr_used:
        issues.append("ocr_text_requires_review")
    status = "partial" if any("no_extractable_text" in issue or issue == "images_not_extracted"
                              or issue.endswith("requires_review") for issue in issues) else "extracted"
    LOGGER.debug("Extraction text_chars=%d ocr_used=%s", total_chars, ocr_used)
    return Extraction(segments, issues, status, ocr_used)


def safely_extract(data: bytes, suffix: str) -> Extraction:
    # Library parser warnings may quote document text. Count them, never log them.
    parser_logger = logging.getLogger("pypdf")
    counter = ParserWarnings()
    handlers, propagate = parser_logger.handlers[:], parser_logger.propagate
    parser_logger.handlers = [counter]
    parser_logger.propagate = False
    started = perf_counter()
    LOGGER.debug("Document extraction started")
    try:
        with warnings.catch_warnings(record=True) as caught:
            result = extract_document(data, suffix)
        result.issues.extend(code for code in counter.counts if code not in result.issues)
        if caught and "parser_diagnostics_require_review" not in result.issues:
            result.issues.append("parser_diagnostics_require_review")
        if caught or any(code != "pdf_object_header_whitespace" for code in counter.counts):
            result.status = "partial"
        LOGGER.info("Document extracted status=%s segments=%d issues=%d",
                    result.status, len(result.segments), len(result.issues))
        LOGGER.debug("Parser diagnostics categories=%s python_warning_count=%d",
                     json.dumps(counter.counts), len(caught))
        return result
    finally:
        LOGGER.debug("Document extraction finished elapsed_ms=%.2f",
                     (perf_counter() - started) * 1000)
        parser_logger.handlers = handlers
        parser_logger.propagate = propagate


class DocumentLibrary:
    """Single-writer SQLite storage; failed attempts never publish a version."""

    def __init__(self, database: Path = DEFAULT_DATABASE) -> None:
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as connection, connection:
            schema = connection.execute("PRAGMA user_version").fetchone()[0]
            if schema not in (0, SCHEMA_VERSION):
                raise ValueError("Unsupported document catalogue schema.")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, source_path TEXT UNIQUE NOT NULL,
                    current_version INTEGER
                );
                CREATE TABLE IF NOT EXISTS versions (
                    id INTEGER PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id),
                    sha256 TEXT NOT NULL, format TEXT NOT NULL, extractor TEXT NOT NULL,
                    original BLOB NOT NULL, extraction TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                    UNIQUE(document_id, sha256, extractor)
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id),
                    status TEXT NOT NULL, reason TEXT,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );
                PRAGMA user_version = 1;
            """)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def ingest(self, path: Path) -> str:
        started = perf_counter()
        source = path.resolve()
        document_id = hashlib.sha256(str(source).encode("utf-8")).hexdigest()
        LOGGER.info("Document ingestion started document_id=%s", document_id)
        with closing(self.connect()) as connection, connection:
            connection.execute("INSERT OR IGNORE INTO documents(id, source_path) VALUES (?, ?)",
                               (document_id, str(source)))
        try:
            suffix = source.suffix.casefold()
            if suffix not in FORMATS:
                raise ExtractionError("unsupported_format")
            limit = MAX_PDF_FILE_BYTES if suffix == ".pdf" else MAX_FILE_BYTES
            with source.open("rb") as stream:
                data = stream.read(limit + 1)
            if len(data) > limit:
                raise ExtractionError("file_size_limit")
            fingerprint = hashlib.sha256(data).hexdigest()
            extractor = extractor_version(suffix)
            with closing(self.connect()) as connection, connection:
                existing = connection.execute(
                    "SELECT id FROM versions WHERE document_id=? AND sha256=? AND extractor=?",
                    (document_id, fingerprint, extractor),
                ).fetchone()
                if existing is not None:
                    connection.execute("UPDATE documents SET current_version=? WHERE id=?",
                                       (existing["id"], document_id))
                    attempt = connection.execute(
                        "INSERT INTO attempts(document_id,status) VALUES (?, 'unchanged')", (document_id,))
                    LOGGER.info("Document version reused without extraction document_id=%s attempt_id=%d "
                                "version_id=%d", document_id, attempt.lastrowid, existing["id"])
                    LOGGER.debug("Document ingestion finished document_id=%s elapsed_ms=%.2f",
                                 document_id, (perf_counter() - started) * 1000)
                    return document_id
            extraction = safely_extract(data, suffix)
        except (ExtractionError, LocalExtractionError, OSError, PyPdfError, zipfile.BadZipFile,
                XMLSyntaxError, InvalidXmlError, KeyError, ValueError) as error:
            reason = str(error) if isinstance(error, (ExtractionError, LocalExtractionError)) \
                else "unreadable_or_corrupt_document"
            with closing(self.connect()) as connection, connection:
                attempt = connection.execute(
                    "INSERT INTO attempts(document_id,status,reason) VALUES (?, 'failed', ?)",
                    (document_id, reason))
            log_failure(f"Document ingestion document_id={document_id} "
                        f"attempt_id={attempt.lastrowid} reason={reason}", error)
            LOGGER.debug("Document ingestion finished document_id=%s elapsed_ms=%.2f",
                         document_id, (perf_counter() - started) * 1000)
            raise ExtractionError(reason) from error
        with closing(self.connect()) as connection, connection:
            cursor = connection.execute(
                "INSERT INTO versions(document_id,sha256,format,extractor,original,extraction) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (document_id, fingerprint, suffix, extractor, data, json.dumps(asdict(extraction))),
            )
            connection.execute("UPDATE documents SET current_version=? WHERE id=?",
                               (cursor.lastrowid, document_id))
            attempt = connection.execute("INSERT INTO attempts(document_id,status) VALUES (?, ?)",
                                         (document_id, extraction.status))
        LOGGER.info("Document version published status=%s document_id=%s attempt_id=%d version_id=%d",
                    extraction.status, document_id, attempt.lastrowid, cursor.lastrowid)
        LOGGER.debug("Document ingestion finished document_id=%s elapsed_ms=%.2f",
                     document_id, (perf_counter() - started) * 1000)
        return document_id

    def list_documents(self) -> list[dict]:
        with closing(self.connect()) as connection:
            rows = connection.execute("""
                SELECT d.id, d.source_path, d.current_version, v.sha256, v.format,
                       v.extractor, v.created_at, v.extraction,
                       a.status AS latest_attempt_status, a.reason AS latest_attempt_reason
                FROM documents d LEFT JOIN versions v ON v.id=d.current_version
                LEFT JOIN attempts a ON a.id=(
                    SELECT MAX(id) FROM attempts WHERE document_id=d.id
                ) ORDER BY d.source_path
            """).fetchall()
        result = []
        for row in rows:
            entry = dict(row)
            extracted = json.loads(entry.pop("extraction")) if row["extraction"] else None
            entry["quality"] = {
                key: extracted[key] for key in ("status", "issues", "ocr_used")
            } if extracted else None
            result.append(entry)
        LOGGER.info("Catalogue listed documents=%d", len(result))
        return result

    def show(self, document_id: str, version_id: int | None = None) -> dict:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT v.* FROM versions v JOIN documents d ON d.id=v.document_id "
                "WHERE d.id=? AND v.id=COALESCE(?, d.current_version)",
                (document_id, version_id),
            ).fetchone()
        if row is None:
            raise ValueError("Document version not found.")
        entry = dict(row)
        entry.pop("original")
        entry["extraction"] = json.loads(entry["extraction"])
        return entry

    def versions(self, document_id: str) -> list[dict]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT id, sha256, format, extractor, created_at FROM versions "
                "WHERE document_id=? ORDER BY id", (document_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def snapshots(self, document_ids: list[str] | None = None) -> list[DocumentSnapshot]:
        """Read selected current extracted versions without loading original blobs."""
        with closing(self.connect()) as connection:
            rows = connection.execute("""
                SELECT d.id, d.source_path, d.current_version, v.sha256, v.extraction,
                       (SELECT status FROM attempts WHERE document_id=d.id ORDER BY id DESC LIMIT 1)
                       AS latest_attempt_status
                FROM documents d LEFT JOIN versions v ON v.id=d.current_version ORDER BY d.id
            """).fetchall()
        selected = {row["id"] for row in rows} if document_ids is None else set(document_ids)
        available = {row["id"] for row in rows}
        if selected - available:
            raise DocumentQueryError("Selected document ID does not exist.")
        result = []
        for row in rows:
            if row["id"] not in selected:
                continue
            if row["current_version"] is None:
                if document_ids:
                    raise DocumentQueryError("Selected document has no successfully extracted version.")
                continue
            result.append(DocumentSnapshot(
                row["id"], row["current_version"], row["source_path"], row["sha256"],
                json.loads(row["extraction"]), row["latest_attempt_status"],
            ))
        return result

    def export_original(self, document_id: str, destination: Path, version_id: int | None = None) -> None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT v.original FROM versions v JOIN documents d ON d.id=v.document_id "
                "WHERE d.id=? AND v.id=COALESCE(?, d.current_version)",
                (document_id, version_id),
            ).fetchone()
        if row is None:
            raise ValueError("Document version not found.")
        with destination.open("xb") as stream:
            stream.write(row["original"])
        LOGGER.info("Original document exported")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--log-config", type=Path, default=PROJECT_DIR / "logging.json")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Ingest files or immediate files in a folder")
    ingest.add_argument("paths", type=Path, nargs="+")
    commands.add_parser("list", help="List documents and latest extraction status as JSON")
    review = commands.add_parser(
        "inventory-review", help="Inspect saved candidates or record an explicit human decision",
    )
    review.add_argument("question", help="The exact enumeration question used to generate candidates")
    review.add_argument("--document", action="append", dest="document_ids", required=True)
    review.add_argument("--item", dest="item_id", help="Candidate review_id from the saved records")
    review.add_argument("--decision", choices=["accepted", "rejected", "unreviewed"])
    review.add_argument("--note", default="", help="Private reviewer note; stored locally, not logged")
    save = commands.add_parser("fact-save", help="Add or correct an evidence-backed typed fact")
    save.add_argument("payload", type=Path, help="JSON file matching the typed fact payload schema")
    save.add_argument("--fact", dest="fact_id", help="Existing fact ID for a correction")
    save.add_argument("--revision", type=int, help="Expected latest revision for a correction")
    facts = commands.add_parser("facts", help="List latest typed fact revisions for selected documents")
    facts.add_argument("--document", action="append", dest="document_ids", required=True)
    history = commands.add_parser("fact-history", help="Inspect immutable revisions and human decisions")
    history.add_argument("fact_id")
    fact_review = commands.add_parser("fact-review", help="Explicitly confirm, reject or reset a fact revision")
    fact_review.add_argument("fact_id")
    fact_review.add_argument("--revision", type=int, required=True)
    fact_review.add_argument("--decision", choices=["confirmed", "rejected", "unreviewed"], required=True)
    fact_review.add_argument("--note", default="")
    fact_query = commands.add_parser("fact-query", help="Query explicit human-confirmed fact records without a model")
    fact_query.add_argument("operation", choices=["count", "sort", "compare", "sum"])
    fact_query.add_argument("--fact", action="append", dest="fact_ids", required=True)
    fact_query.add_argument("--from-date", dest="lower")
    fact_query.add_argument("--through-date", dest="upper")
    preview_mode = fact_query.add_mutually_exclusive_group()
    preview_mode.add_argument(
        "--preview", action="store_true", help="Show evidence and validation without calculating",
    )
    preview_mode.add_argument(
        "--expected-preview", help="Run only if records and options still match this preview token",
    )
    workbench = commands.add_parser(
        "fact-workbench", help="Add similar facts, review pending records and preview calculations",
    )
    workbench.add_argument("--document", dest="document_id", required=True)
    workbench.add_argument("--question", help="Exact question for browsing a saved single-document inventory")
    workspace = commands.add_parser("workspace", help="Guided document selection, queries and fact review")
    workspace.add_argument("--model", default="deepseek-r1:8b",
                           help="Answer model (default: deepseek-r1:8b; explicit override supported)")
    workspace.add_argument("--planner-model", default="qwen3.5:4b")
    workspace.add_argument("--embedding-model", default="nomic-embed-text")
    workspace.add_argument("--enumeration-model", default="qwen3.5:4b")
    for command in ("index", "search", "ask", "plan", "chat", "summarize", "compare", "enumerate"):
        descriptions = {
            "ask": "Answer a question; calculations propose a reviewed-fact preview only",
            "chat": "Interactive questions and explicit approval of reviewed-fact previews",
            "plan": "Show the routing plan only; does not select facts or calculate",
        }
        subparser = commands.add_parser(command, description=descriptions.get(command))
        subparser.add_argument("--document", action="append", dest="document_ids",
                               help="Restrict to a document ID; repeat for multiple documents")
        subparser.add_argument("--embedding-model", default="nomic-embed-text")
        if command == "index":
            subparser.add_argument("--rebuild", action="store_true",
                                   help="Replace only search data, not originals or catalogue versions")
        else:
            if command != "chat":
                subparser.add_argument("question")
            subparser.add_argument("--limit", type=int, default=6)
            if command in ("ask", "plan", "chat", "summarize", "compare", "enumerate"):
                subparser.add_argument(
                    "--model",
                    default="qwen3.5:4b" if command == "enumerate" else "deepseek-r1:8b",
                )
                subparser.add_argument("--planner-model", default="qwen3.5:4b")
                if command in ("ask", "chat", "plan"):
                    subparser.add_argument(
                        "--enumeration-model", default="qwen3.5:4b",
                        help="Independent model for full-document enumeration",
                    )
                if command != "chat":
                    subparser.add_argument("--format", choices=["prose", "list", "ascii_table", "csv"],
                                           dest="presentation")
            if command in ("summarize", "compare", "enumerate"):
                if command == "enumerate":
                    subparser.add_argument(
                        "--records-json", action="store_true",
                        help="Emit full quote-backed records and coverage instead of compact output",
                    )
                subparser.add_argument("--max-seconds", type=float, default=600,
                                       help="Stop before starting a model call after this elapsed budget")
                subparser.add_argument("--max-calls", type=int, default=80)
                subparser.add_argument("--no-resume", action="store_true",
                                       help="Bypass stored synthesis notes, e.g. after replacing a model tag")
    for command in ("show", "export", "versions"):
        subparser = commands.add_parser(command)
        subparser.add_argument("document_id")
        subparser.add_argument("--version", type=int)
        if command == "export":
            subparser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        configure_logging(args.log_config)
        library = DocumentLibrary(args.database)
        if args.command == "workspace":
            from document_workspace import run_workspace
            from document_query import QuerySession, get_planner
            from document_search import DocumentSearch
            from chatbot_tech_doc import get_embeddings, get_llm

            def create_session(identifier: str) -> QuerySession:
                return QuerySession(
                    DocumentSearch(library, get_embeddings(args.embedding_model), args.embedding_model),
                    get_planner(args.planner_model), get_llm(args.model), [identifier],
                    enumeration_model=get_llm(args.enumeration_model),
                )

            run_workspace(library, create_session)
            return 0
        if args.command == "fact-workbench":
            from document_fact_ui import run_fact_workbench
            run_fact_workbench(library, args.document_id, args.question)
            return 0
        if args.command in ("fact-save", "facts", "fact-history", "fact-review", "fact-query"):
            from document_facts import (
                MAX_PAYLOAD_CHARS, fact_history, list_facts, preview_fact_query, query_facts,
                review_fact, save_fact,
            )
            if args.command == "fact-save":
                with args.payload.open(encoding="utf-8") as stream:
                    text = stream.read(MAX_PAYLOAD_CHARS + 1)
                if len(text) > MAX_PAYLOAD_CHARS:
                    raise DocumentQueryError("Fact payload exceeds its size limit.")
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as error:
                    raise DocumentQueryError("Fact payload file must contain valid JSON.") from error
                result = save_fact(
                    library, payload, fact_id=args.fact_id, expected_revision=args.revision,
                )
            elif args.command == "facts":
                result = list_facts(library, args.document_ids)
            elif args.command == "fact-history":
                result = fact_history(library, args.fact_id)
            elif args.command == "fact-review":
                result = review_fact(library, args.fact_id, args.revision, args.decision, args.note)
            else:
                if args.preview:
                    result = preview_fact_query(
                        library, args.fact_ids, args.operation, lower=args.lower, upper=args.upper,
                    )
                else:
                    result = query_facts(
                        library, args.fact_ids, args.operation, lower=args.lower, upper=args.upper,
                        expected_preview=args.expected_preview,
                    )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "inventory-review":
            from document_inventory_review import review_inventory
            result = review_inventory(
                library, args.question, args.document_ids,
                item_id=args.item_id, decision=args.decision, note=args.note,
            )
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
            return 0
        if args.command in ("summarize", "compare", "enumerate"):
            if args.command == "enumerate" and args.records_json and args.presentation:
                raise DocumentQueryError("--records-json cannot be combined with --format.")
            from chatbot_tech_doc import get_llm
            from document_query import render_answer
            from document_synthesis import synthesize
            result = synthesize(library, get_llm(args.model), args.command,
                                args.question, args.document_ids,
                                max_seconds=args.max_seconds, max_calls=args.max_calls,
                                resume=not args.no_resume)
            if args.command == "enumerate" and args.records_json:
                print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
            else:
                print(render_answer(result, args.presentation or "prose"))
            return 0
        if args.command in ("index", "search", "ask", "plan", "chat"):
            from document_search import DocumentSearch
            from chatbot_tech_doc import get_embeddings, get_llm
            search = DocumentSearch(library, get_embeddings(args.embedding_model), args.embedding_model)
            if args.command == "index":
                print(json.dumps(search.index(args.document_ids, rebuild=args.rebuild)))
            elif args.command == "search":
                print(json.dumps(search.search(
                    args.question, args.document_ids, limit=args.limit,
                ), indent=2))
            else:
                from document_query import QuerySession, get_planner, run_chat
                session = QuerySession(
                    search, get_planner(args.planner_model), get_llm(args.model),
                    args.document_ids, limit=args.limit,
                    enumeration_model=get_llm(args.enumeration_model),
                )
                if args.command == "chat":
                    run_chat(session)
                elif args.command == "plan":
                    print(json.dumps(asdict(session.plan(args.question, args.presentation)), indent=2))
                else:
                    print(session.execute(args.question, args.presentation))
            return 0
        if args.command == "ingest":
            failed = False
            for path in args.paths:
                paths = sorted(item for item in path.iterdir() if item.is_file()) if path.is_dir() else [path]
                if not paths:
                    print("Folder contains no files.", file=sys.stderr)
                    failed = True
                for item in paths:
                    if item.name == ".DS_Store" or item.name.startswith("._"):
                        LOGGER.info("System metadata skipped reason=system_metadata")
                        continue
                    try:
                        identifier = library.ingest(item)
                        print(json.dumps({"document_id": identifier}))
                    except ExtractionError as error:
                        print(f"Ingestion failed: {error}", file=sys.stderr)
                        failed = True
            return 1 if failed else 0
        if args.command == "list":
            print(json.dumps(library.list_documents(), indent=2))
        elif args.command == "versions":
            print(json.dumps(library.versions(args.document_id), indent=2))
        elif args.command == "show":
            print(json.dumps(library.show(args.document_id, args.version), indent=2))
        else:
            library.export_original(args.document_id, args.destination, args.version)
        return 0
    except DocumentQueryError as error:
        log_failure("Document query", error)
        print(str(error), file=sys.stderr)
        return 1
    except (httpx.HTTPError, ConnectionError, ResponseError) as error:
        log_failure("Document library local service", error)
        print("Local Ollama request failed. Check the service, installed models, and timeout.",
              file=sys.stderr)
        return 1
    except (OSError, ValueError, sqlite3.Error) as error:
        log_failure("Document library command", error)
        print(f"Document library command failed ({type(error).__name__}). "
              "Check configuration, paths, permissions, and document/version ID.", file=sys.stderr)
        return 1
    except Exception as error:
        log_failure("Document library unexpected error", error)
        raise


if __name__ == "__main__":
    # Keep catalogue and search exception identities in the same module namespace.
    from document_library import main as run_main
    sys.exit(run_main())
