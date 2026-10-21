# Archive It

Archive It is a local-first command-line tool for organizing documents, finding
relevant passages, and asking questions with inspectable source references. It
keeps versioned originals and extracted text in a local catalogue and uses
[Ollama](https://ollama.com/) for embeddings and model-assisted queries.

The repository also includes a separate chatbot for the bundled vehicle-warning
documentation.

**Status:** Experimental personal-use prototype. Extraction, citations, and model
answers are not guarantees of accuracy or completeness. Check important results
against the original documents.

## What you can do

- Import PDF, DOCX, UTF-8 text, PNG, JPEG, and single-frame TIFF documents.
- Retain original files, extraction history, and page, block, or line references.
- Search with keyword and semantic retrieval, then inspect the evidence.
- Ask focused questions, summarize documents, and compare selected documents.
- Author and review evidence-backed facts, then count, sort, compare, or sum
  explicitly selected confirmed records without model calculations.
- Use a guided terminal workspace for document selection and review.

OCR uses local Tesseract with English language data. Automatic inventory
enumeration is experimental; it produces candidates for human review, not a
verified or exhaustive inventory.

## Quick start

### 1. Set up Python

Use Python 3.13, the runtime documented in the project's validation history.
From the repository root, create a virtual environment and install the pinned
dependencies:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The examples use a macOS/Linux shell. If `.venv` already exists, use its
interpreter rather than recreating it.

For scanned PDFs and images, install Tesseract with English language data. On
macOS:

```bash
brew install tesseract
tesseract --list-langs
```

See [OCR setup and limits](./TECHNICAL_GUIDE.md#local-ocr-setup-and-limits)
for supported inputs and extraction constraints.

### 2. Prepare local models

Install and start Ollama, then download the models used by default:

```bash
ollama pull nomic-embed-text
ollama pull deepseek-r1:8b
ollama pull qwen3.5:4b
```

The application connects to `http://127.0.0.1:11434`. Native Ollama is preferred
on Apple Silicon for GPU acceleration; a
[Docker Compose option](./TECHNICAL_GUIDE.md#ollama-in-docker) is also available.
Model downloads require network access, but document queries use the local
service. The application does not download models automatically.

### 3. Import and explore a document

```bash
.venv/bin/python document_library.py ingest /path/to/document.pdf
.venv/bin/python document_library.py list
.venv/bin/python document_library.py workspace
```

In the workspace, select a document and choose **index** before using **search**
or focused **ask** questions. Importing and extracting documents does not require
Ollama; indexing and model-assisted queries do.

For direct commands, replace `DOCUMENT_ID` with an ID returned by `ingest` or
`list`:

```bash
.venv/bin/python document_library.py index --document DOCUMENT_ID
.venv/bin/python document_library.py search 'payment terms' --document DOCUMENT_ID
.venv/bin/python document_library.py ask 'What are the payment terms?' --document DOCUMENT_ID
```

Without `--document`, direct search and indexing commands use all successfully
extracted documents. Run `index` again after importing changes.

### Bundled warning chatbot

To query the bundled warning documentation instead of your personal catalogue:

```bash
.venv/bin/python chatbot_tech_doc.py
```

This uses a separate persistent index. See the
[warning chatbot guide](./TECHNICAL_GUIDE.md#run-and-rebuild) for index validation,
rebuild commands, and follow-up questions.

## Privacy and limitations

- Documents, questions, and extracted evidence are processed locally. The
  application disables LangSmith tracing and optional Chroma telemetry.
- The catalogue stores **unencrypted original documents, extracted text,
  embeddings, and historical versions**. Protect local storage, backups, and
  terminal output with appropriate permissions and disk encryption. There is
  no document-deletion command yet.
- Ingest only trusted files. Parser limits are not a complete sandbox.
- Keep Ollama on loopback; the application provides no API authentication.
- The [Git ignore rules](./.gitignore) exclude common private-data locations and
  formats, but do not protect already tracked files or arbitrary personal notes.
  Review staged content before publishing.
- Citations identify source passages; they do not establish that a model's
  interpretation is correct. This tool does not provide professional advice.
- The pinned Chroma dependency has a documented
  [temporary security exception](./TECHNICAL_GUIDE.md#validated-dependency-upgrade-and-temporary-chroma-exception).
  Keep it embedded and local; do not treat this prototype as security-certified.

## Development

Install the test runner and run the regression suite:

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest -q
```

Default tests use synthetic documents and isolated databases. Some extraction
tests need Tesseract; live-model and other opt-in tests have separate setup
requirements described in the [testing guide](./TECHNICAL_GUIDE.md#tests).

## Documentation

The [technical guide](./TECHNICAL_GUIDE.md) contains the full command reference,
architecture and retrieval details, fact-review workflows, extraction limits,
logging configuration, acceptance benchmarks, and dependency validation history.
