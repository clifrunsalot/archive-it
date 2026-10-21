# Local warning-documentation chatbot

Install dependencies into the project virtual environment using `requirements.txt`.
The manifest now contains the local pipeline dependencies only. Previously
installed OpenAI/Unstructured packages are not automatically uninstalled from an
existing environment; use a fresh environment when checking the reduced set.
The script uses Ollama at `127.0.0.1:11434`; proxy environment settings are ignored.
Source text and questions are sent to that local service, not a hosted model API.
LangSmith tracing and optional Chroma telemetry are disabled in this application.

## Personal document catalogue and querying (phases 1-5)

### Keeping personal material out of Git

The project `.gitignore` excludes private data, backups, exports, logs, databases,
common document/image formats, structured JSON/CSV exports, local editor settings,
and credentials. Source code, `requirements.txt`, `logging.json`, and the bundled
`data/mg-zs-warning-messages.html` remain eligible for version control. Put personal
notes and references in `private/`, `personal/`, or `references/`, regardless of
their format; Markdown and arbitrary filenames elsewhere are not automatically
recognized as private. These excluded folder names apply at any directory level.

Ignore rules are not access controls or encryption. They do not protect files
already tracked by Git, and `git add -f` can bypass them. Check `git status` and
the staged diff before committing. New fixtures/configuration in ignored formats
must be reviewed as synthetic/non-sensitive before adding a narrow exception;
do not broadly unignore private storage. `logging.json` and `.env.example` are
exceptions for non-sensitive settings/examples only, never personal paths or secrets.

`document_library.py` provides a separate, local document catalogue and extraction
workflow, with hybrid search and cited passage questions. Ingestion/extraction
does not call Ollama; the explicit `index`, `search`, and `ask` commands use the
local models described below. The separate warning index/chatbot is unchanged.

Install the pinned dependencies from `requirements.txt`, then:

```bash
.venv/bin/python document_library.py ingest /path/to/lease.pdf /path/to/receipt.docx /path/to/notes.txt
.venv/bin/python document_library.py ingest /path/to/documents-folder
.venv/bin/python document_library.py list
.venv/bin/python document_library.py show DOCUMENT_ID
.venv/bin/python document_library.py versions DOCUMENT_ID
.venv/bin/python document_library.py show DOCUMENT_ID --version VERSION_ID
.venv/bin/python document_library.py export DOCUMENT_ID /path/to/new-original-copy.pdf
```

### Index, search, and ask

After ingestion, build the persistent keyword/vector index:

```bash
.venv/bin/python document_library.py index
.venv/bin/python document_library.py search 'policy ZX-789'
.venv/bin/python document_library.py search 'Are pets permitted?' --document DOCUMENT_ID
.venv/bin/python document_library.py ask 'What is the monthly rent?' --document DOCUMENT_ID
.venv/bin/python document_library.py ask 'What is the deductible?' --model qwen3.5:4b
```

`search` outputs JSON passages with source path, document ID, version, hash,
segment, source locations, extraction issues, and scope coverage. `ask` outputs an
answer with numbered citations and those source references. Repeat `--document`
to select several catalogue IDs. Without selection, current successfully
extracted versions of all documents are searched. Unknown IDs or explicitly
selected documents without successful extraction fail rather than broadening
scope. Original source files are not needed after ingestion.

Run `index` again after ingesting changes. Unchanged extracted versions reuse
their stored vectors. Queries reject a selected scope containing current versions
that have not been indexed; they never silently search older versions. A failed
extraction keeps the prior successful version searchable, but coverage and source
labels explicitly report the failed latest attempt. Failed-only documents are
excluded from an unrestricted search and counted in coverage.

`index --document DOCUMENT_ID` builds only selected documents. `--embedding-model`
on `index`, `search`, and `ask` defaults to `nomic-embed-text`; the same model tag
must be used across these commands. Model/configuration changes require
`index --rebuild`. Rebuild replaces only search data, not originals or catalogue
versions; a failed build rolls back both keyword and vector changes. Rebuild with
a document selection leaves other documents unindexed. Model tags are recorded,
not model digests; explicitly rebuild if you replace a model behind the same tag.
No models are automatically downloaded.

Both keyword records (SQLite FTS5) and normalized embeddings are stored in the
catalogue database and commit atomically. Semantic retrieval uses an exact cosine
scan of selected current-version vectors, not the warning Chroma collection.
This is appropriate for an initial small personal library but takes linear time
and memory as passage counts grow. A larger library will need an indexed vector
backend with equivalent version-consistency guarantees. Use a single writer;
index transactions can remain open during local embedding requests.

Long extracted segments are split into overlapping, bounded passages without
crossing their source segment. Locations retain page/block/row/line references
and character offsets. DOCX heading and first-table-row context are included in
embedding input (the first row is not asserted to be a verified semantic header).
Input is conservatively bounded below the default embedding model's context;
models with smaller input limits may require different chunking. Keyword
retrieval uses literal query tokens and BM25; semantic retrieval handles
paraphrases. Reciprocal rank fusion merges/deduplicates the candidates, with exact
punctuated identifiers (such as `ZX-789`) prioritized over semantic matches. `--limit`
controls returned passages (default 6, range 1-10), not corpus completeness.
For field-style questions beginning with what/which/who/where, passages with a
line-start field heading matching a substantive question word are prioritized
over incidental mentions (for example a `Supplier:` contact block versus a rule
about notifying a supplier of address changes). This is a general label-based
retrieval heuristic, not an address extractor or semantic guarantee. Basic
singular/plural normalization matches labels such as `UTILITY` to “utilities”.
Less common matching heading terms are weighted above repeated generic terms.
Possessive suffixes (straight or curly apostrophes) are removed before matching;
“landlord's address” and “address of the landlord” use the same field terms.
Matching scores account for the full heading length, favoring a direct
`LANDLORD:` block over a partial match such as `Landlord or Agent's Name:`.
For an address question, when retrieved evidence also contains a direct matching
party heading with a street address, name-only absence notes and addresses under
other labelled fields are excluded from model input. This is a narrow
disambiguation safeguard, not a general address extractor or semantic guarantee.
The next overlapping passage in the same source segment receives a smaller
heading preference so a list continuing beyond the heading's passage can be
included. This is bounded context expansion, not whole-section coverage.
For example, “which specific utilities is the tenant responsible for paying?”
can retrieve the utilities clause and its named list instead of only generic
payment-duration rules. Answers must retain conditional scope (such as optional
charging activity) and distinguish named items from generic obligations. Exact
identifier and term-date priorities remain ahead of this preference. It uses
the existing stored passages; no re-ingestion or index rebuild is required.

Answer prompts include only selected evidence within a conservative UTF-8 byte
budget plus a character budget for the configured model context. If fewer
passages fit, only those supplied to the model receive evidence labels.
The model receives stable document numbers, passage labels, source text,
locations and extraction-quality flags. Random catalogue IDs, content hashes
and filesystem paths stay in the returned provenance records, not in the model
prompt. Separate document versions still receive distinct model-visible document
numbers. Moving identical sources between temporary catalogues no longer changes
their model-visible identities. The metadata character budget still bounds the
full returned evidence.
Personal `ask`, `chat` and `workspace` default to `deepseek-r1:8b`; `--model` can select another installed answer
model. One-shot `ask` invocations are standalone; `chat` retains session-local
document scope and a successful query plan as described below.
Empty scopes skip model generation. Missing/unknown citations, truncated answers,
or invalid output fail explicitly. Citations identify evidence candidates, not
proof that each generated claim is supported. Prompt instructions cannot
guarantee factual grounding or prevent document prompt injection.

Focused `ask` answers now share structured findings validation and deterministic
citation rendering with whole-document synthesis (`document_answers.py`).
Focused answers use a source-ID protocol: every finding includes a nonempty
`support` array of integer labels from the passages actually supplied, followed
by its prose `text`. The application retains the original text, version,
location and quality metadata in the returned evidence records and renders the
selected labels as citations. The model no longer has to reproduce quotes.
This replaces the quote-first protocol, whose copy step repeatedly produced
altered quotes and consumed output budget. Missing or unknown labels fail
validation and receive the existing single corrective retry. Ollama's schema
also constrains support IDs to the actual supplied labels.
Synthesis retains its existing `{text, citations}` findings protocol.
Custom focused-answer model adapters must use `{support: [1], text: "..."}`;
the former `{citation, quote}` objects are rejected rather than silently trusted.

A conservative focused-only wording guard also rejects common unscoped claims
that information is not recorded/provided/stated. Such absence must be scoped
to a field, passage or clause, or use the explicit insufficient-evidence status.
Verbatim supported negative rules (such as “No warranty is provided.”) are also
allowed unless they assert document-wide absence.
This is a wording heuristic, not a semantic absence checker. A name-only field's
local omission must not override a direct value elsewhere. Prompts also require
preserving genuine conflicting values without inventing precedence.
Coverage reports `support_protocol=source-passages-v1` and
`source_references_verified` for findings with validated labels. The former
`support_quotes_verified` field is no longer emitted. `semantic_support_verified`
remains false: valid labels and unchanged source text prove neither that the
prose follows the passage nor that the passage answers the question.
It does not certify relevance, completeness, correct scope or interpretation.
The guard can turn previously published misleading answers into explicit
validation errors; the model's general answer fidelity is not certified.

Ollama answers request schema-constrained JSON. The full schema is passed only
through the API, not repeated in the prose prompt; the prompt describes the
fields briefly. A raw local response demonstrated that DeepSeek spent its
entire 512-token probe reasoning about the repeated schema even with `think=false`.
DeepSeek-R1 is therefore run with explicit reasoning extraction; other focused
models retain disabled thinking. Reasoning shares the output-token budget and
is neither published nor logged. Coverage counts responses containing extracted
reasoning as `model_reasoning_responses`, including rejected attempts.
The configured client's defaults are not mutated. Each finding requires valid integer
source IDs from the passages actually supplied to the model. One corrective retry
is allowed for invalid output; fixed diagnostics are logged without answer text.
Focused answers and final summaries/comparisons accept up to eight findings,
matching the schema advertised to the model; intermediate synthesis notes remain
limited to three. Valid four-to-eight-finding answers do not trigger a corrective
retry. Truncated output is still refused, never published as a partial answer.
Focused findings are rendered as one whitespace-normalized row each so multiline
addresses and other wrapped claims remain on the same row as their citations.
This does not change the source text or synthesis rendering.
When a focused DeepSeek-R1 response, summary or comparison stage hits its token limit, the existing single
corrective retry uses a 4096-token output budget (or preserves a larger configured
budget). Ordinary first requests retain the configured limit, normally 2048.
Other validation errors and other models do not receive this increase. The
original client configuration is unchanged; there is still no third call, no
model fallback and no publication of truncated output. This may increase latency
and does not guarantee that every response fits.
Conflict questions explicitly require the discrepancy and any supported
uncertainty about precedence, rather than just listing the two amounts.
Focused-answer instructions are kept compact to leave room for source passages
and reduce redundant output. Named lists should normally be one finding rather
than a separate finding for every item. A conflict conclusion
must cite all passages it uses and retain any stated precedence uncertainty in
the prose. Source-ID checks, the eight-finding limit, prompt budget and bounded
truncation retry remain in effect. These
instructions reduce failure opportunities; they do not verify semantic support.
The installed DeepSeek model can still exhaust its output budget. Earlier attempts
to disable its thinking did not reliably do so. Qwen completed the lease bed-bug question in one request; DeepSeek
truncated twice. At the user's selection, personal answers now default to DeepSeek;
this model change is not an accuracy certification and may increase latency or
produce explicit truncation errors. There is no silent fallback. Planner and
enumeration defaults remain Qwen. Restart an already-open workspace to use the new
default, or launch it explicitly with `--model deepseek-r1:8b`. Use
`--model qwen3.5:4b` to override the answer model. No re-ingestion or
re-indexing is needed. The separate technical chatbot defaults are unchanged.
The same single retry allowance covers transient local server errors.
An explicit insufficient-evidence response identifies reviewed passages and does
not assert the information is absent from the full document. Coverage includes
the supplied passage count, actual model requests, and `insufficient_evidence`.
Prompt packing reserves space for a corrective retry. This change fixes response
validation, not retrieval recall: the selected passages may still omit the
requested lease terms, dates, or other information.

Hybrid retrieval supports **focused passage questions**, not whole-document
coverage. Whole-document summaries/comparisons use the separate pipeline below.
Full-corpus lists, financial totals, rankings, and legal/tax advice remain unsupported.
Semantic search can return irrelevant passages even for unrelated questions;
there is no calibrated relevance cutoff. Treat insufficient evidence as unknown
and inspect original sources, especially for financial/legal decisions. Extraction
quality flags and non-exhaustive scope coverage remain visible in the output.

### Query planning, presentation, and interactive sessions

`ask` now goes through a structured planner before retrieval. It separates the
operation, factual search question, document scope, presentation, clarification,
and requested filters. The planner uses the installed `qwen3.5:4b` model with
thinking disabled, JSON-schema output, and a bounded generation budget. Override
with `--planner-model`; it is independent of `--model`, the answer model.
No additional dependencies or model downloads are required.

```bash
# Inspect a proposal without searching or generating an answer
.venv/bin/python document_library.py plan 'What is the monthly rent? Answer as an ASCII table.' --document DOCUMENT_ID

# Choose presentation explicitly or request it in the question
.venv/bin/python document_library.py ask 'What is the monthly rent?' --document DOCUMENT_ID --format ascii_table
.venv/bin/python document_library.py ask 'What is the deductible?' --document DOCUMENT_ID --format csv

# Start an interactive personal-document session
.venv/bin/python document_library.py chat --document DOCUMENT_ID
.venv/bin/python document_library.py chat --model qwen3.5:4b
```

Supported operations are search for passages, explanation, extraction of
facts from selected passages, whole-document summaries and comparisons, and
explicit enumeration from selected documents. Calculations/rankings and
professional legal/tax advice are never executed by the planner. Supported
calculations now route to reviewed-fact **proposals**, requiring explicit human
approval as described below; grouped rankings and professional advice remain
unsupported. Date/tag/party/historical-version filters on passage queries stop
execution with guidance to select documents explicitly; reviewed date facts
separately support inclusive date bounds. Undefined categories and ambiguous
document references trigger a clarification rather than invented assignments.
The motivating "component with the most warnings" request is explicitly gated as
aggregation, even when asked as an ASCII table. Enumeration does not implement
calculations, rankings, or technical-component classifications.

Plans are validated before use: unknown fields/operations, invalid JSON, unknown
document IDs, scope broadening, oversized questions, and truncated output are
explicit errors with no fallback query. Multi-line JSON and harmless line wrapping
inside a search question are accepted. Model-proposed plans are not guarantees
of intent accuracy; inspect `plan` for complex questions. The planner never receives
document contents or answer text, only the latest question, permitted catalogue
IDs/filenames, and the previous successful plan. A conservative prompt budget may
require selecting fewer documents for a large catalogue.

Formatting is deterministic, not a request to the search rewriter or answer model:
`prose`, `list`, `ascii_table`, or `csv`. Tables/CSV use Type/Content rows for the
answer, evidence sources, and coverage; they do not invent domain-specific columns
or structured facts. ASCII output escapes non-ASCII characters and wraps rows.
CSV output quotes values and guards leading spreadsheet formulas. Citations and
quality/coverage remain attached in every format. `--format` overrides the
planner's proposed presentation.

Interactive commands:

| Command | Effect |
| --- | --- |
| `/select ID [ID ...]` | Select documents and clear prior query context |
| `/all` | Restore all current successful documents and clear context |
| `/scope` | Inspect selected document IDs (`null` means all) |
| `/reset` | Clear prior query/answer context, retaining document selection |
| `exit` / `quit` / Ctrl+D / Ctrl+C | End the session |

A successful plan's scope persists for follow-ups. The planner resolves references
using the prior search intent, not by treating previous answers as factual evidence.
Unrelated standalone questions should not inherit the previous topic. A pure
formatting follow-up re-renders the cached result without new retrieval or answer
generation, after checking scope, current versions, and latest ingestion status.
Changed documents require a fresh question. Failed/unsupported turns do not replace
successful context. To resolve a document ambiguity, use `/select` and restate the
question explicitly; clarification replies are not maintained as a separate dialogue.
Sessions are in memory only and do not persist between processes.

The existing `search` command remains a direct low-level hybrid passage lookup
without a planner or answer generation. It cannot establish corpus-wide completeness.
Programmatic `DocumentSearch.ask` also remains the focused low-level API;
`QuerySession` in `document_query.py` is the planned, conversational interface.

### Whole-document summaries and comparisons (step 4)

Select documents explicitly. These operations read **all stored extracted text**
for the current selected versions, not the top search hits; a search index and
embeddings are not required. They do not re-read the original files or certify
that extraction captured the entire original.

```bash
# Direct operations bypass model intent classification
.venv/bin/python document_library.py summarize 'Summarize obligations and deadlines.' --document DOCUMENT_ID --model qwen3.5:4b
.venv/bin/python document_library.py compare 'Compare rent, exceptions, and pet obligations.' --document FIRST_ID --document SECOND_ID --model qwen3.5:4b --format ascii_table

# The planner also routes these operations through ask and chat
.venv/bin/python document_library.py ask 'Summarize the entire lease.' --document DOCUMENT_ID --model qwen3.5:4b
.venv/bin/python document_library.py chat --document FIRST_ID --document SECOND_ID --model qwen3.5:4b
```

Summaries accept 1-4 documents; comparisons require 2-4 distinct documents.
Direct summary/comparison commands still default to DeepSeek; `--model qwen3.5:4b` is a
locally tested alternative. Summaries and comparisons explicitly extract
DeepSeek-R1 reasoning so its presence is counted without logging or displaying
it. Other models still receive disabled optional reasoning; models may differ
in honoring this. Experimental enumeration retains its separate disabled-thinking
guard and does not use the extended summary retry budget.

The pipeline divides extracted segments into non-overlapping UTF-8-bounded
passages with exact page/block/line and character locations. When all selected
extracted evidence fits the final prompt budget, a summary goes directly from
original passages to the final cited answer in one generation request, without
intermediate notes. The fit check includes the full prompt in UTF-8 bytes and
reserves 640 bytes for a corrective retry. This avoids both the intermediate
model call and the risk of losing requested details during note compression.
Larger summaries and all comparisons still read every passage into cited notes,
combine notes hierarchically per document when needed, then generate the final
answer. No source characters are silently truncated or sampled to enable the
direct path.
Every model prompt is limited to 6,000 UTF-8 bytes. A request allows at most
128 input batches and eight reduction levels; oversized or non-shrinking notes
fail with guidance to narrow the request. Large documents can require many local
model calls and take substantially longer than a passage answer.
Coverage reports `synthesis_strategy` as `direct`, `hierarchical`, or `inventory`.
`input_batches` counts source partitions, not model requests; a direct summary
skips input-note generations and reports final-stage progress immediately.
Validation, the existing single corrective retry, cancellation, version checks
and checkpoint reuse still apply. A direct summary can therefore require two
requests after a rejected response, or zero when reusing a validated checkpoint.

Compact instructions and batch-local document aliases leave more space for
source text without increasing the context budget or dropping source characters.
Schema constraints remain enforced by Ollama and validated by the application.
Input batches reporting no relevant evidence are tracked deterministically and
excluded from model reduction. If an entire selected document has no findings,
its reviewed references and uncertainty are appended explicitly; it is never
silently dropped from comparison coverage. If no document has findings, no
reduction/final model calls are needed.

Synthesis progress is recorded through the configured application logger;
direct-command prose/CSV output remains clean. The guided workspace additionally
shows scoped progress and heartbeats on stderr. Progress includes
input/reduction/final stage, counts, elapsed seconds, actual calls and reused
checkpoints. Input ETA is approximate and excludes reduction/final work; it is
omitted when checkpoint reuse makes the estimate misleading.

The default budget is **600 seconds or 80 actual model calls per invocation**,
including retries. It is checked before every call, not a hard interrupt of an
in-flight generation: one call may finish after the elapsed budget. A budget
failure does not publish a partial answer. Direct commands accept overrides:

```bash
.venv/bin/python document_library.py summarize 'Summarize the rent terms.' --document DOCUMENT_ID --model qwen3.5:4b --max-seconds 600 --max-calls 80
```

Validated intermediate results are committed individually to the existing
private catalogue's `synthesis_checkpoints` table for Ollama clients. Re-running
the identical command resumes completed calls, with a fresh invocation budget;
cancelled/failed calls are never cached. Checkpoints include input, reduction,
and final generations. Reuse requires the same request, current versions,
extraction hashes/latest attempt status, prompt, schema, allowed citations,
and model configuration/endpoint. Coverage separates new model calls from reused
checkpoints. Non-Ollama programmatic runnables do not persist checkpoints because
they have no reliable model identity. Changed model contents behind the same tag
cannot be detected: use `--no-resume` to bypass reuse after replacing a model tag.
Historical checkpoints remain stored; they are **unencrypted sensitive text**,
like the retained original/extraction versions, and inherit database/backup
privacy requirements. A malformed checkpoint fails explicitly.

Intermediate and final outputs reject truncation, excessive size, missing
citations and unknown/out-of-batch labels. Final answers must cite every selected
document. Returned source labels resolve to original extracted passages, not just
generated notes. The coverage report records processed segments/characters,
versions, quality issues, OCR status, latest attempt status, and model-call counts.
Synthesis requests schema-constrained JSON containing findings with integer
citation IDs, rather than depending on model-formatted prose citations. Citation
markers are rendered deterministically after validation. A batch with no relevant
information uses an explicit `no_relevant_evidence` result; its reviewed-source
references are preserved without inventing factual findings or claiming absence
from the original. This also supports targeted requests such as "describe the
amenities" while still reading every selected extracted passage.
Invalid outputs receive at most one corrective retry using fixed diagnostics,
never echoing rejected output. Both attempts obey the prompt budget; a second
failure stops processing. Logs distinguish invalid JSON/schema, missing citations,
out-of-batch IDs, and truncation, with stage/batch/attempt numbers but no source or
model text. Coverage counts actual model requests including retries and
no-relevant-evidence responses.
For DeepSeek-R1 summary/comparison input, reduction and final stages, a truncated
first response receives the same bounded retry policy as a focused answer:
4096 output tokens, or the configured limit if larger. First requests and retries
for other validation failures retain their configured limit. There is still only
one retry per generation, and a second truncation fails without publishing a
partial result or caching the failed generation. Coverage also records
`model_reasoning_responses`; reasoning content is never published.

The synthetic manual-workspace summary has an opt-in live regression that checks
the requested amounts, utilities and lease dates, and requires one model request
with the direct strategy:

```bash
RUN_OLLAMA_TESTS=1 .venv/bin/python -m pytest tests/test_document_synthesis.py \
  -k live_deepseek_manual_lease_summary -v
```

Transient Ollama HTTP 500/502/503/504 errors and in-stream server errors (status
`-1`) share the same single retry allowance
per generation; other service errors stop immediately. Service retries keep the
original prompt and log only the status code and stage, not server error text.
Known extraction gaps remain explicit. Documents changing during processing
invalidate the result; failed requests do not replace successful chat context.
Format-only follow-ups reuse a successful result with the existing version checks.

Reduction and final comparison/summary prompts pass intermediate findings as
separate text and numeric citation arrays, rather than inline-cited prose. This
avoids encouraging the model to copy rendered citation markers into finding
text. Inline markers in model finding text remain invalid; the application
validates source IDs and renders citations itself. Existing validated input
checkpoints can be reused; changed note prompts automatically get new cache keys.

**Processing all extracted text is not an exhaustive or verified summary.**
Generated notes are selective and may omit or misinterpret details; citation
validation checks references, not entailment. Comparison prompts distinguish
conflicts, missing evidence and uncertainty without assuming one document
supersedes another. Important financial/legal findings need source review.
This synthesis workflow adds no verified calculations, guarantees of exhaustive
inventories, automatic revision pairing, or new domain classifications.
The original technical warning chatbot remains separate and retains its prior behavior.

### Explicit enumeration (step 5)

Enumeration is for requests such as "list every amenity in this lease" or
"enumerate the stated renewal conditions." It examines all stored extracted text
for the explicitly selected current document versions, without relying on the
top-k search results. The model identifies candidate items separately, keeps
citations on each item, and does not reduce batch results into a summary.
Enumeration uses category-filtering instructions separate from summary prompts.
For amenity requests, an application-side gate keeps a candidate only when its
cited source sentence names it and explicitly states access, availability or
provision as an amenity. Hypothetical, conditional and prohibited mentions do
not establish availability. Explicit optional/paid access retains its
conditions. A source saying that all or part of a facility group is provided as
an amenity is reported only as an ambiguous group, never as its possible
members. These checks address the known relevance failures in the synthetic
regression cases; they do not establish general category relevance or
completeness.
Identical names, conditions, assessments and uncertainty (ignoring case,
whitespace and a trailing period) are combined within each document version,
retaining their supporting spans. Differing conditions, separate documents and
different versions remain separate. This is not semantic or factual verification.
Terminal output uses compact item bullets, filename/page references and one
coverage warning rather than repeating full paths and raw coverage metadata.
Programmatic results retain the full provenance and coverage fields.
The direct command bypasses query planning; `ask` and `chat` can route recognized
enumeration requests through the same operation. Explicit "list/enumerate
all/every/each" item requests are guarded against accidental snippet-search
routing; literal requests for matching passages, files or documents stay search:

```bash
.venv/bin/python document_library.py enumerate 'List every amenity stated in the lease.' --document DOCUMENT_ID
.venv/bin/python document_library.py ask 'List every renewal condition in the lease.' --document DOCUMENT_ID
```

Select 1-4 documents explicitly. Enumeration works from extracted text and does
not require the search index. Direct `enumerate` defaults to `qwen3.5:4b`;
override it with `--model`. In `ask`/`chat`, `--enumeration-model` independently
selects this model (also Qwen by default), separately from the personal answer
model (now also Qwen by default). No model is silently substituted for an explicit override.
The locally installed DeepSeek model was observed generating hidden reasoning
despite `think=false`, taking minutes per batch on CPU. Enumeration captures
unexpected reasoning and stops after that response with guidance to select Qwen,
rather than repeating this cost throughout the document. Reasoning text is not logged.
Each input batch is processed and validated; if
the request exceeds the configured time/call budget or output budget, it fails
instead of silently truncating the list. The structured response permits up to
16 individually cited items per input batch; this limit and the non-exhaustive
coverage flag are reported. If a model reaches its output-token limit, the
truncated response is discarded and the passage batch is split recursively;
if a single passage still truncates, the request stops with an explicit error.
Itemized results and their serialized supporting records have a separate
256,000-character output limit, rather than the 12,000-character summary limit;
no generated items are dropped to fit it.
Ollama calls can resume from validated
checkpoints in the same way as summaries.

Enumeration coverage reports that all extracted text was processed, but
`exhaustive` remains false: the model can overlook or misinterpret matching
items, and extraction may omit content from the original. A "no matching items
identified" result is not proof of absence. Review cited source passages and the
original document before relying on the result. Other categories do not receive this amenity-specific relevance gate.
Repeat the local model relevance regression with
`RUN_OLLAMA_TESTS=1 .venv/bin/python -m pytest tests/test_document_synthesis.py -k live_enumeration -q`.
It checks inclusion of stated amenities and exclusion of unrelated clauses,
contents headings and hypothetical facilities; it does not certify all
documents. Enumeration remains experimental: model omissions, extraction
limitations and unsupported relevance outside these narrow amenity rules remain
possible. A bounded local reasoning experiment exhausted 512 tokens without a
final answer, so production reasoning settings remain unchanged.

### Evidence-backed inventory foundation (step 6, experimental)

Enumeration now produces structured candidate records, rather than only free-form
cited lines. `AnswerResult.items` retains each item's name, conditions, model
assessment (`established` or `ambiguous`), uncertainty, document/version identity,
and supporting evidence. An `established` assessment is still a model judgment,
not an application-verified finding.

For local Ollama calls, the response schema constrains supporting quotes to
exact source passages or sentence/paragraph spans. The model selects from those
strings instead of freely rewriting a quote. A separate validator checks each
quote against its cited input passage before publication or checkpointing.
Citations are derived from the validated support, not generated as a second,
potentially inconsistent label list. Invalid quotes or labels fail after one
corrective retry; no completed result is published.

Each supporting record contains the original quote, source path, document and
version IDs, segment, location and zero-based, end-exclusive character offsets
within the stored extraction segment. Offsets count Python Unicode characters,
not bytes or PDF coordinates. Repeated quotes retain all matching spans rather
than guessing an occurrence. Input splitting prefers sentence/paragraph boundaries
when they fit the byte budget, preserves every character, and retains offsets.
Very long clauses and page boundaries can still separate relevant context.

Records are available through the API and through an explicit JSON CLI option:

```bash
.venv/bin/python document_library.py enumerate 'List every amenity in the lease.' \
  --document DOCUMENT_ID --records-json
```

`--records-json` cannot be combined with `--format`. It includes private source
text and paths, so treat exported JSON as personal document data. Default output
remains compact, is labeled experimental, and marks model-reported ambiguity
with `Needs review`. Full quote-backed batch records are checkpointed in the
private catalogue; resumed records are validated again against the source.
Changing the protocol/schema invalidates older free-form checkpoints without
deleting them.

**Verified quote provenance is not verified relevance, entailment, availability,
completeness, or numerical accuracy.** Coverage and individual records explicitly
report `semantic_relevance_verified=false`. Full-lease testing verified every
support span and zero-call checkpoint replay, but still found unrelated items
and incorrect interpretations. The live relevance regression remains a release
gate for a trustworthy inventory, not a passing certification of this foundation.
No totals, professional conclusions, domain profiles or automatic semantic
deduplication are enabled by this change.

### Human acceptance of inventory candidates

A completed enumeration saves its candidate records locally. Each starts as
`unreviewed`; the application does not automatically accept any model output.
Compact output distinguishes `Unreviewed candidate` from `Accepted by reviewer`.
`Needs review` additionally marks uncertainty reported by the extraction model,
not the human decision. Accepted ambiguous items retain that uncertainty.

Inspect saved candidates, exact quotes, offsets and their `review_id` without
running a model:

```bash
.venv/bin/python document_library.py inventory-review \
  'List every amenity in the lease.' --document DOCUMENT_ID
```

Use the **same question and selected documents** as the enumeration. This command
prints full JSON records, including private source text and paths. Re-enumerate
once to populate review records for results produced before this feature.
After checking the evidence and original document, record your decision:

```bash
.venv/bin/python document_library.py inventory-review \
  'List every amenity in the lease.' --document DOCUMENT_ID \
  --item REVIEW_ID --decision accepted --note 'Checked the source'

.venv/bin/python document_library.py inventory-review \
  'List every amenity in the lease.' --document DOCUMENT_ID \
  --item REVIEW_ID --decision rejected --note 'Unrelated to the requested category'
```

Use `--decision unreviewed` to reset a decision. Notes are optional, limited to
2,000 characters, stored locally and not logged. Every explicit change appends
a timestamped event in `review_history`; prior decisions are retained. This is
a local reviewer workflow, not an authenticated multi-user approval system.
No personal lease candidates are accepted or rejected automatically.

Rerunning enumeration applies saved decisions only when the exact question,
document/version, candidate name, conditions, uncertainty and supporting spans
still match. Citation renumbering does not invalidate a matching decision.
Changed candidates require a new review; decisions for previous candidates
remain in the private audit table. Review commands refuse stale source versions
or changed ingestion status, requiring a new enumeration first.
The private audit tables retain the original candidate payload and question
alongside decision events, even after a newer run replaces the displayed set.

Rejected candidates are excluded from compact item output but remain in JSON
records with their supporting evidence and history. Coverage reports accepted,
rejected and unreviewed counts. If every candidate is rejected, the output says
so rather than claiming the document contains no matches. Cached reruns and
format-only chat follow-ups refresh decisions without new extraction calls;
format-only follow-ups retain their original candidate set.

**Human acceptance is not a completeness guarantee.** This workflow addresses
which generated candidates a person accepts; it cannot recover items the model
missed, correct extraction gaps or automatically prove relevance. The automated
semantic acceptance tests remain unresolved, and `semantic_relevance_verified`
stays false even for human-accepted candidates. No calculations or professional
conclusions are enabled by inventory review itself. The separate typed-fact
workflow below requires its own confirmation before deterministic calculations.

### Typed facts, corrections and deterministic record queries

Typed facts are explicitly authored by a person, not automatically inferred from
inventory candidates. They support dates, money, quantities and text. Creating a
fact or correcting its value does **not** confirm it. Inventory acceptance does
not transfer to typed facts; an optional `candidate_id` records the link to an
archived candidate from the same document/version.

First inspect the current stored extraction with `show`, or the supporting
records from enumeration. Save a JSON payload with **all** these fields:

```json
{
  "document_id": "DOCUMENT_ID",
  "version_id": 18,
  "name": "Monthly rent",
  "kind": "money",
  "value": "1200.00",
  "unit": "",
  "currency": "USD",
  "period": "monthly",
  "conditions": "",
  "uncertainty": "",
  "candidate_id": "",
  "support": [
    {
      "segment": 1,
      "char_start": 0,
      "char_end": 22,
      "quote": "Rent $1200.00 monthly."
    }
  ]
}
```

This is a schema example, not a fact about your documents: substitute the real
document/version and an exact quote at the stated offsets. Segments are one-based;
character offsets are zero-based and end-exclusive within the extraction segment.
At least one exact supporting span is required. Quotes and offsets are checked
against the current extraction; the application does not prove that the quoted
text entails the typed value. A human must check that interpretation.

| Kind | Value format | Required metadata |
| --- | --- | --- |
| `date` | Valid ISO `YYYY-MM-DD` string | Empty unit, currency and period |
| `money` | Decimal string, never a JSON float | Three-uppercase-letter currency code; period `once`, `daily`, `weekly`, `monthly`, or `yearly`; empty unit |
| `quantity` | Decimal string | Explicit unit; empty currency and period |
| `text` | Nonempty string | Empty unit, currency and period |

Numeric strings allow an optional minus sign, up to 24 integer digits and 6 fractional digits.
No thousands separators, exponents, NaN or Infinity are accepted. Currency codes
are syntax-checked, not checked against a currency registry. No currency, unit
or period conversions are inferred. Conditions and unresolved uncertainty are
retained explicitly; facts with uncertainty cannot be used in deterministic
queries even after confirmation.

```bash
# Add a proposed fact from a private JSON file.
.venv/bin/python document_library.py fact-save fact.json

# Inspect current facts and their returned fact_id/revision.
.venv/bin/python document_library.py facts --document DOCUMENT_ID

# Explicitly confirm this revision only after checking the source and value.
.venv/bin/python document_library.py fact-review FACT_ID \
  --revision 1 --decision confirmed --note 'Checked value and source'

# Correct a fact using a complete replacement payload, with optimistic revision checking.
.venv/bin/python document_library.py fact-save corrected-fact.json \
  --fact FACT_ID --revision 1

# Inspect every immutable revision and its separate review history.
.venv/bin/python document_library.py fact-history FACT_ID
```

Corrections create a new `unreviewed` revision and preserve previous payloads and
decisions. They cannot move the fact to a different document. Concurrent/stale
revision numbers are refused. `fact-review` also supports `rejected` and
`unreviewed`; each change appends a timestamped event, never overwriting history.
Changed source versions or ingestion status make a fact stale. To use it again,
save an updated revision against the current extraction and confirm that revision.
Prior/superseded revisions remain inspectable but are not queryable.

The following deterministic queries make **no model calls** and use only the
latest revision of each explicitly named fact:

```bash
.venv/bin/python document_library.py fact-query count --fact FACT_A --fact FACT_B
.venv/bin/python document_library.py fact-query sort --fact DATE_A --fact DATE_B \
  --from-date 2026-01-01 --through-date 2026-12-31
.venv/bin/python document_library.py fact-query compare --fact FACT_A --fact FACT_B
.venv/bin/python document_library.py fact-query sum --fact MONEY_A --fact MONEY_B
```

- Every selected record must be current, confirmed and free of unresolved
  uncertainty. An invalid record fails the entire query; nothing is silently skipped.
- `count` counts records, not distinct real-world items or their numeric values.
  Different kinds may be counted together. Repeated IDs are refused.
- `sort` orders compatible values ascending, with fact IDs breaking ties.
  ISO dates sort chronologically, numbers numerically, and text lexicographically.
- `compare` requires exactly two compatible facts and reports `less`, `equal` or
  `greater` for the first selected value relative to the second. Its records are
  also sorted for presentation; `selected_fact_ids` preserves comparison order.
- `sum` uses exact decimal arithmetic, not floating point. It supports only money
  and quantity records with identical kinds, units, currency, periods and
  conditions. Overlapping source spans across selected records are refused to
  guard against obvious double-counting. It does not detect every semantic duplicate.
- Date bounds are inclusive, apply only to date `count`/`sort`, and are reported
  in the result. No matches gives an explicit count of zero or an empty sorted list.

JSON results include the selected IDs, contributing records with exact source
support and locations, extraction quality flags, revision and review history,
units/periods, conditions and scope.
The selected IDs are the inclusion rule: the application does not decide which
charges belong together, resolve alternative clauses or infer eligibility.
Results explicitly say they concern selected confirmed **records**, not a
document-wide total or an exhaustive inventory. Human confirmation is not a
legal, financial or completeness guarantee.

Natural-language aggregate requests in `ask`/`chat` can propose a reviewed-fact
selection and calculation preview, described below. They never calculate without
an explicit execution command. Automatic semantic extraction remains unreliable.
Fact payloads, notes and supporting quotes are private local catalogue data and
are not logged. Protect any JSON files or terminal exports containing them.

### Guided document workspace

Use one terminal menu instead of copying document IDs between commands:

```bash
.venv/bin/python document_library.py workspace --model qwen3.5:4b

# Use an isolated acceptance-test catalogue (no shell helper required):
.venv/bin/python document_library.py \
  --database "$HOME/Documents/archive-it-test/catalogue.sqlite3" \
  workspace --model qwen3.5:4b

# Equivalent, if you already defined the archive_test shell function:
archive_test workspace --model qwen3.5:4b
```

Select an initial document by its displayed number or exact filename. Duplicate
filenames require a number; full source paths are shown to distinguish them. Use
**scope** to replace or expand the active scope with one or more displayed
document numbers (for example `1,3`). The workspace lists status and warnings for
every active document, whether each current version has a stored index, and its
confirmed, unreviewed, rejected, stale and eligible fact counts. Stale counts
overlap review counts. A stored index is not a health guarantee; search still
validates model/profile compatibility and index consistency. Every selected
document must have a successful extraction; unavailable documents are not
silently excluded.

Actions accept a number or name:

- **index**, **search**, and **ask** use the active scope. Indexing is explicit,
  never automatically triggered by a query.
- **summarize** processes the extracted text in the active scope. **compare**
  requires at least two selected documents; both actions cite sources by
  document and passage. The synthesis service supports at most four selected
  documents. It reports supported differences and conflicts without assuming
  which document supersedes another; citations identify the evidence, not which
  agreement controls.
- **scope** selects one or more documents by their displayed numbers. A scope
  change clears prior query/inspection state and pending approval.
- **facts** opens the existing model-free fact workbench; **quit** there returns
  to the workspace. Fact editing and **calculate** require a single-document
  scope; use **scope** to select one document first. The calculation preview
  remains manual and does not use a model.
- **selection** changes a pending natural-language proposal using catalogue
  numbers/ranges; it produces a fresh preview.
- **approve** displays the pending preview again and requires explicit **run**.
  Enter defaults to cancel. Blocked previews cannot be approved. Execution uses
  the existing catalogue and selected-record consistency guards.
- **cancel-query** discards a pending proposal.
- **documents** returns to the initial document chooser; **quit** exits.
- **inspect** opens the most recent completed search, answer, summary or
  comparison without
  calling a model. Choose a displayed citation number to read its original
  supplied passage, source path, version, location and extraction warnings,
  separately from the model's interpretation. Supplied but uncited passages are
  identified as such. This is source inspection, not semantic verification.

Workspace **search** now displays numbered passages in retrieval order instead
of raw JSON. Each header shows the filename, version, segment and available
page/block/row/line location. Numbers match **inspect** exactly. Passage text is
wrapped for readability without omission; exact original whitespace, full paths,
IDs, hashes and character offsets remain available through inspection. Extraction
issues, failed latest ingestion and retrieval coverage remain visible. Results
are candidate passages, not generated answers or verified support; lower-ranked
passages may not answer the question and retrieval is not guaranteed exhaustive.
This changes only the workspace **search** action: the standalone
`document_library.py search` command still returns the original JSON shape with
`passages` and `coverage`, suitable for scripts.

Within **inspect**, use **segment**, **previous** or **next** to read the full
extracted segment or adjacent segments for that evidence's document. This
additional context is explicitly marked as *not evidence used for the answer*.
If that document's current version has changed, only the retained query evidence
is displayed; context navigation is disabled. A later failed ingestion is also
flagged. Press Enter to leave context, then Enter to return to the workspace.
Beginning another action, changing scope, or cancelling clears the inspection
result; approval actions do not generate new evidence. Inspection results are
kept only for this workspace session, not saved as an answer history.

Service actions show stage/model information and elapsed time on stderr, with a
heartbeat every five seconds while waiting. Generation reports batch, request and
retry numbers; synthesis also reports completed batches, model calls and reused
checkpoints. No question, source text, answer or hidden reasoning is included in
these progress messages.

Press **Ctrl+C** during an action to return to the menu. Pending approval and
incomplete inspection/query state are discarded; explicitly saved facts, review
decisions and validated synthesis checkpoints remain. Rerun an identical synthesis
request with the same model/profile to reuse its valid checkpoints. Cancellation
interrupts the local client: **Ollama may still finish an already submitted
request**. It does not stop the shared server or promise server-side cancellation.
EOF exits the workspace; `/cancel` abandons the current prompt.

#### Menu examples

The menu is a convenience layer over the existing services, not a new processing
engine. Enter the action's name or its displayed number. The examples below use
names; type them at the workspace's **Action** prompt, not at the shell prompt.
Questions and fact values are illustrative, not claims about your documents.

| Action | Purpose | Example interaction |
| --- | --- | --- |
| **index** | Prepare the active scope for hybrid search | Enter `index`. Check the returned indexed/reused version counts. Do this before the first passage search or focused answer, and after ingesting a new version. |
| **search** | Show relevant source passages without generating an answer | Enter `search`, then enter `monthly parking` at **Search question**. Inspect the returned passages and source locations. |
| **ask** | Ask a natural-language question using the active scope and existing planner | Enter `ask`, then `What is the monthly rent?`. Inspect the cited answer. A calculation request instead produces a fact-selection preview, not an immediate result. |
| **summarize** | Summarize extracted text in the active scope | Enter `summarize`, then `Summarize the charges and term dates.` at **Summary request**. Check the citations and coverage limitations. This action does not require a search index. |
| **compare** | Compare selected documents and surface supported differences or conflicts | Enter `compare`, then `Compare the rent and term dates.` at **Comparison request**. Select at least two documents with **scope** first; cited evidence does not establish which document controls. |
| **scope** | Replace the active document set | Enter `scope`, then enter displayed numbers such as `1,3`. The status banner confirms all selected documents. |
| **facts** | Open the fact workbench to add, correct or review facts | Enter `facts`, then `add` to author a fact with source evidence, or `review-pending` to review saved drafts. Saving does not confirm a fact. Enter `quit` in the workbench to return to the workspace. |
| **calculate** | Manually select facts and preview a deterministic query | Enter `calculate`, select the displayed rent and parking fact numbers (for example `1,2`), then choose `sum`. Inspect the preview and explicitly enter `run`, or press Enter to cancel. No model is used. |
| **selection** | Replace the facts in a pending natural-language proposal | After an `ask` calculation preview, enter `selection`, then the proposal catalogue numbers, such as `1,3` or `1-3`. Inspect the new preview. Selection order matters for comparison. |
| **approve** | Execute a ready natural-language proposal only after approval | After checking an `ask` preview, enter `approve`. The preview is displayed again. Enter `run` to calculate; Enter or `cancel` discards it. No token copying is needed in this menu. |
| **cancel-query** | Discard a pending natural-language proposal without calculating | After an `ask` preview, enter `cancel-query`. To calculate later, ask again for a fresh preview. |
| **documents** | Return to the catalogue and select another initial document | Enter `documents`, then a displayed number or exact filename such as `sample.pdf`. Any pending proposal for the previous scope is discarded. |
| **inspect** | Inspect the last completed result's original source evidence | After `search`, `ask`, `summarize` or `compare`, enter `inspect`, then a citation number such as `1`. Use `segment`, `previous` or `next` for clearly separated additional context. No model is called. |
| **quit** | Close the workspace | Enter `quit`. Explicitly saved facts and decisions remain; pending proposals do not. |

#### Example: review facts and calculate monthly charges

For a synthetic document stating rent of USD 1200 monthly, parking of USD 50
monthly and a USD 500 one-time deposit:

1. Start the workspace using one of the commands above. At **Document number or
   exact filename**, enter the displayed number or `sample.txt`. No document ID
   needs to be copied. The document must already have been ingested.
2. Choose **index** if you want to search or ask focused passage questions.
   Then choose **search** and enter `monthly parking` to inspect the source.
3. Choose **facts**. Use **add** to create rent, parking and deposit money facts,
   each with its own exact evidence. Enter values `1200`, `50` and `500`,
   currency `USD`, and periods `monthly`, `monthly` and `once`, respectively.
   Check conditions against the source. Use **similar** to reuse type/currency/
   period defaults where appropriate, but change the deposit period to `once`.
4. Explicitly **save** each fact, then **confirm** it after checking its evidence.
   Alternatively choose **leave** after saving and use **review-pending** later.
   Enter **quit** to return to the workspace with the same document selected.
5. Choose **ask** and enter `What is the total of the confirmed monthly charges?`
   Check that rent and parking are proposed, and that the one-time deposit is
   omitted. Inspect values, evidence, periods, conditions and blocking reasons.
   If necessary, choose **selection** and enter the displayed catalogue numbers
   for rent and parking. Do not assume they are always numbered 1 and 2.
6. Choose **approve**, inspect the displayed preview, and enter **run**.
   For these exact synthetic facts, the expected result is USD 1250 monthly over
   the selected records, not a guaranteed complete document-wide total.
7. To test cancellation instead, make a fresh request with **ask**, then choose
   **cancel-query**, or choose **approve** and press Enter at its run/cancel menu.
8. Choose **documents** to work on another file, or **quit** to exit.

For the model-free route, replace steps 5-6 with **calculate**: select rent and
parking by their displayed numbers, choose **sum**, inspect the preview and
explicitly choose **run**. Selecting the one-time deposit as well must block
the sum because the periods differ; it must not silently omit that record.

**approve**, **selection** and **cancel-query** require a pending proposal from
**ask**. Manual **calculate** has its own preview/run prompt and does not create
such a proposal. If there is no pending preview, choose **ask** first. A missing
index error calls for **index**; incompatible or damaged indexes require the
explicit CLI `index --rebuild` operation, not an automatic workspace rebuild.

The document stays selected across actions. New questions, indexing, search,
summaries, fact workbench entry, manual calculation, document changes and exit
discard pending natural-language approvals. `/cancel` abandons the current
workspace action and clears pending approval. EOF/Ctrl-C exits the workspace;
nested workbench/calculation cancellation retains its existing behavior.
Explicitly saved facts and review events are never rolled back by navigation.
The workspace does not remember document selection or proposals across processes.

Browsing, status, fact review and manual calculation require no model calls.
Query clients are created lazily for indexing/search/ask/summarize/compare.
Existing model defaults are retained; override `--model`, `--planner-model`,
`--embedding-model` or `--enumeration-model` as needed. Expected
service/validation failures are reported and return to the menu without silently
selecting another document.
There is no automatic rebuild, ingestion, model download or bulk confirmation.
The workspace is a guided scope selector, not a completeness guarantee.
Automatic enumeration remains experimental, including when an `ask` question
routes to it. Scope selection and comparison do not infer that the documents
belong together or establish precedence when their terms conflict.

The terminal shows private paths, source text and facts. Workspace navigation
does not log those contents; protect terminal captures just like CLI exports.

### Interactive fact workbench

Use the local, model-free workbench instead of writing fact payload JSON or
copying character offsets manually:

```bash
.venv/bin/python document_library.py fact-workbench --document DOCUMENT_ID

# Also browse the candidates from a saved single-document inventory.
.venv/bin/python document_library.py fact-workbench --document DOCUMENT_ID \
  --question 'list every amenity in the lease'
```

The workbench operates on one explicitly selected document at a time:

- **add**: browse extracted segments and select evidence for a missed fact.
- **similar**: select an existing fact by number and start a **new** fact with
  its kind, unit, currency and period as editable defaults. Press Enter to keep
  those defaults. Name, value, evidence, conditions, uncertainty, candidate link
  and confirmation are never copied. Select fresh evidence and explicitly save
  and review the new fact; the original remains unchanged.
- **candidate**: browse saved candidates with their review status, conditions,
  uncertainty and source support, then prefill a new typed fact. The question
  must match the saved single-document inventory exactly. Conversion does not
  accept the inventory candidate or confirm the fact.
- **edit**: inspect an existing fact and create a corrected revision. Previously
  confirmed revisions never transfer approval to the new revision.
- **review**: inspect an existing fact and explicitly confirm or reject its
  current revision without creating a correction.
- **review-pending**: walk through the current unreviewed facts, showing evidence
  and asking for a separate confirm/reject/leave decision for each. No repeated
  fact-number selection or ID copying is needed. Enter leaves a record unchanged;
  already confirmed/rejected records are not queued. Stale facts are listed
  separately and require correction with fresh evidence. `/cancel` ends the
  queue, preserving decisions already explicitly recorded.
- **calculate**: explicitly select fact numbers, preview evidence and compatibility,
  then choose whether to execute a deterministic query.
- **quit**: leave the workbench.

For new evidence, choose a segment and read its full source text. Enter an exact
substring, or press Enter to select the whole segment (up to 4,000 characters).
If the substring occurs more than once, choose the occurrence from its offsets
and surrounding context. The workbench computes the source offsets; you do not
type them. You can select multiple supporting spans. Evidence containing line
breaks can be selected as a whole segment; the existing JSON interface remains
available for finer multiline spans.

Enter the name, kind, value and applicable units/currency/period. Conditions
and uncertainty remain explicit. Enter keeps a displayed default; `/clear`
empties a text field. Read the draft preview and source context, then explicitly
choose **save**. The default is **cancel**, which writes no fact.
Invalid values produce an explicit validation message and may be edited or
cancelled; no invalid revision is published.

Saving creates an **unreviewed** revision. A separate **confirm/reject/leave**
prompt follows, defaulting to **leave**. Confirmation is never inferred from
saving, candidate acceptance or pressing Enter. Optional private review notes
are stored with the decision but not logged.

Use `/cancel` to abandon the current action. EOF or Ctrl-C exits safely:
unsaved drafts are discarded; previously saved revisions and decisions remain.
If you exit after saving but before confirming, the fact stays unreviewed.
Source-version and revision checks prevent edits or confirmations from silently
overwriting changed data. Stale facts require fresh evidence against the current
extraction before a corrected revision can be saved.

For repeated entry, add the first fact normally, then use **similar** for the
next facts with the same type and units. You can choose **leave** after saving
each fact and use **review-pending** later for a sequential evidence review.
There is no bulk confirmation, automatic value inference or implicit approval.

The terminal displays private source text, paths, values and notes. Do not share
terminal captures containing personal data. The workbench makes no model calls
and does not improve automatic semantic extraction or establish completeness.
Natural-language selection proposals are available separately in `ask`/`chat`;
the workbench itself remains model-free.

### Calculation preview and explicit execution

In the workbench, choose **calculate**, select comma-separated fact numbers in
the desired order, then choose `count`, `sort`, `compare`, or `sum`. For date
count/sort, optional inclusive ISO date bounds can be supplied.
Ascending number ranges are also supported: `1-3,5` selects 1, 2, 3, then 5;
`3,1-2` selects 3, 1, then 2. These are catalogue-number ranges, not value/date
filters. Duplicate, overlapping, descending or out-of-range selections are
refused, with a maximum of 128 selected records. Ineligible selections still
block the preview; ranges never silently select only confirmed records.

The preview shows:

- Every explicitly selected record, including its value, revision, confirmation,
  conditions, uncertainty, extraction quality and exact supporting evidence.
- Passed validation checks when the query is ready; blocking reasons otherwise.
  Invalid selected facts block the query instead of being dropped.
- Which selected IDs fall outside the inclusive date bounds. Unselected IDs
  from the workbench document are listed separately, not treated as query inputs.
- The precise operation, bounds and selected-record scope. There is no
  document-wide total, completeness claim, unit conversion or inferred inclusion rule.

The preview does **not** compute a sum, count result, comparison relation or sorted
result. Choose **run** to calculate; Enter defaults to **cancel**. `/cancel`, EOF
and Ctrl-C also stop execution. A blocked preview never offers a run option.
This operation does not save facts or change review decisions.

Scripted callers can use the same preview and guard:

```bash
.venv/bin/python document_library.py fact-query sum \
  --fact FACT_A --fact FACT_B --preview

# Copy preview_token from that JSON result.
.venv/bin/python document_library.py fact-query sum \
  --fact FACT_A --fact FACT_B --expected-preview PREVIEW_TOKEN
```

`--preview` and `--expected-preview` are mutually exclusive. The token binds the
operation, ordered fact IDs, date bounds, current revisions, source status and
review history. Workbench execution always uses this guard. If those inputs
change, execution refuses the old preview and requires a fresh one; even a new
confirmation event requires a new preview. Calculation uses one consistent
database snapshot read at execution, not a promise that the database will never
change afterward. A token is a consistency check, not authentication or an
approval credential.

A blocked CLI preview returns JSON with `ready=false` and diagnostic reasons;
it is a successful inspection, not a successful calculation. Unknown IDs,
duplicate selections or malformed query shapes fail explicitly. Existing
`fact-query` calls without preview options continue to calculate directly with
the same compatibility and confirmation safeguards.

### Natural-language reviewed-fact proposals

After authoring and confirming typed facts, select documents explicitly:

```bash
.venv/bin/python document_library.py ask \
  'What is the total of the confirmed monthly charges?' --document DOCUMENT_ID
.venv/bin/python document_library.py chat --document DOCUMENT_ID
```

The local planner proposes an operation and an ordered subset of eligible facts,
not an answer. Initial operations are record count, ascending compatible-value
sort (including dates), two-value comparison and compatible money/quantity sums.
Inclusive ISO date bounds are supported only for date count/sort. Ambiguous or
unsupported interpretations should produce a clarification instead. No inferred
conversion, annualization, professional judgment, grouped ranking, entity count,
or completeness claim is supported.
Explicit conversion, annualization, ranking and descending-sort wording is
conservatively guarded before fact planning; other semantic interpretation still
depends on the local model and requires human inspection.
The model can prefilter date candidates itself instead of leaving all date
filtering to the calculator. Out-of-range dates remain visible in the omitted
catalogue with a deterministic bounds note; check both the selected set and the
date bounds rather than interpreting a small selection as a complete inventory.

The preview shows selected records, exact evidence, conditions, revisions and
checks, plus a numbered catalogue of proposed and omitted records. Omitted
records include their evidence and an explicit reason: ineligible (stale,
unconfirmed or uncertain) or simply not proposed. A model's selection is **not
authoritative**; inspect omissions and inclusion rules yourself. Invented,
duplicate or ineligible planner IDs fail explicitly. Incompatible selections
produce a blocked preview rather than silently dropping facts.

In chat:

- `/facts-select 1,2` replaces the selection with these catalogue numbers,
  in that order, and produces a new preview. First/second order matters for
  comparison. Choosing an ineligible record blocks execution.
- `/facts-run TOKEN` explicitly approves the current ready preview. Copy the
  exact token from that preview; ordinary replies like “yes” never approve it.
- `/facts-cancel` discards the proposal without calculating.

A fresh question, `/select`, `/all` or `/reset` discards the pending preview.
Changing the operation or bounds requires a fresh question (or an explicit
`fact-query` preview). Any catalogue change, including new or omitted records
and review events, invalidates the chat proposal; execution additionally rechecks
the selected-record consistency token. Proposals are session-only and single-use;
they do not save facts or review decisions.

One-shot `ask` prints the preview and a shell-quoted `fact-query` command using
the same database and `--expected-preview` token. It does not execute that
command. To edit the selection, use chat, the workbench, or a new explicit
`fact-query --preview`. The one-shot execution command checks selected-record
consistency, not subsequent changes to omitted records; ask again if the
catalogue changes. `plan` continues to show only the initial routing plan,
without a fact-selection model call.

Fact planning uses the local planner model and sends it typed names, values,
units, conditions and eligibility, not source quotes or review notes. The
complete catalogue is bounded to 128 facts and a 6,000-byte prompt; larger scopes
fail explicitly instead of sampling or truncating. Narrow the document scope
or select explicit IDs. Preview and execution do not call the answer model or
retrieval service. Keep terminal output private; personal question/fact contents
are not logged. Model selection errors and incomplete human-authored records
remain possible: approval does not establish a document-wide total.

The default catalogue is `data/document_library/catalogue.sqlite3`, ignored by Git.
Use `--database /path/to/catalogue.sqlite3` or `--log-config /path/to/logging.json`
**before** the command. `list`, `show`, and `versions` output JSON. `show` includes
extracted text and provenance; `export` writes the original bytes and refuses to
overwrite a file. Folder ingestion is non-recursive and attempts every immediate
file except macOS `.DS_Store`/`._*` metadata (logged as skipped), reporting
unsupported files rather than silently skipping them. Batch
ingestion continues after expected per-file errors and exits nonzero if any fail.

### Supported extraction and provenance

| Format | Extracted representation | Source location |
| --- | --- | --- |
| UTF-8 TXT (optional BOM) | Paragraphs retaining internal line breaks | Original line start/end |
| Text PDF | Text per page, retaining extracted values | One-based page number |
| Scanned PDF | Local English OCR for textless pages | One-based page number and OCR method |
| PNG/JPEG/single-frame TIFF | Local English OCR | Image number and OCR method |
| DOCX | Body paragraphs/headings and table rows in body order | One-based block/row and heading |

DOCX table rows contain JSON arrays of cells, preserving cell boundaries rather
than flattening amounts into prose. DOCX locations are not rendered page numbers.
No automatic classification, inferred dates, or domain-specific fields are
published as facts.

Document identity is based on the resolved source path, not the filename alone.
Each successful version records the exact original bytes, SHA-256 hash, format,
extractor/parser version, UTC ingestion timestamp, extracted segments, issues, and
OCR status. Moving a file creates a separate document; identical bytes at different
paths remain separate sources. Timestamps are ingestion times, not document
effective dates. Extraction v3 records PDFium, Pillow, fontTools and installed
Tesseract versions where relevant; profile changes trigger re-extraction.

Unchanged versions are reused without extraction. Changed files create a new
version; old versions remain inspectable/exportable. Parser behavior changes
advance the extractor profile, so re-ingesting unchanged bytes after an upgrade
publishes a newly extracted version. Returning to previously
ingested bytes makes that version current again. SQLite commits the new snapshot,
segments, and current-version pointer together. Failed updates record a failed
attempt and leave the previous successful version intact. `list` distinguishes
the current extraction quality from the latest attempt status; a failed document
with no successful version has no current content. Missing source files do not
delete retained records. Use one ingestion process at a time.

### Quality and limitations

`extracted` means the supported extraction completed, **not verified completeness
or accuracy**. `partial` marks known gaps such as PDF pages without text, DOCX
images, repeated PDF page text or repeated page-edge lines, font/unknown parser diagnostics, or OCR
that requires review. Both retain specific issue flags for inspection.
PDF object-header whitespace warnings are recorded but do not alone downgrade
quality. Parser messages themselves are never logged.
Repeated PDF page text triggers page-bounded PDFium re-extraction rather than
blind deduplication: embedded content outside the visible page is excluded.
Identical text that remains after this correction is retained with a review flag.
Repeated page-edge lines or phrases are retained and flagged for review, not
automatically removed; repeated clauses can therefore also trigger this warning.
PDF layout/table fidelity is not verified. Extraction retains page-level
provenance, not column coordinates or reliable reading order for multi-column
layouts.

DOCX extraction includes main-body paragraphs and tables plus text/rows from
default, first-page and enabled even-page headers/footers, with section/area/block
locations. Linked header/footer parts are included once rather than repeated for
each section that inherits them. Text boxes, tracked revisions, nested-table
fidelity and embedded objects remain unverified; embedded images are flagged
instead of silently OCRed. Source references allow comparison with retained
originals; they do not prove that every meaningful field was extracted.

### Local OCR setup and limits

Install the pinned Python dependencies and a local Tesseract engine with English
language data. On macOS:

```bash
.venv/bin/python -m pip install -r requirements.txt
brew install tesseract
tesseract --list-langs
.venv/bin/python document_library.py ingest /path/to/scanned.pdf /path/to/receipt.png
```

No document contents are sent to an external OCR service. Textless PDF pages use
local rendering/OCR automatically; images use OCR directly. OCR text is always
`partial`, with `ocr_used=true`, an `ocr_text_requires_review` issue, and per-segment
`method=ocr_tesseract_eng` provenance. Native PDF text is preferred when available;
images embedded on otherwise text-bearing pages are not automatically OCRed.
OCR is not a guarantee of correct amounts, identifiers, tables, handwriting, or
signatures. Compare important fields against originals before relying on them.
An entirely empty result still fails explicitly. Missing engines/language data,
timeouts, and resource limits produce fixed failure reasons, not silent omissions.
Encrypted PDF passwords, legacy `.doc`, multi-frame images, HEIC, and macro
documents are not supported.

Input limits are 25 MiB for non-PDF files, **160 MiB for PDFs**, 1,000 PDF pages,
100 MiB expanded DOCX/package or PDF page content, and two million extracted
characters. The higher PDF limit accommodates large scanned originals without
raising limits for other formats. Page-bounded extraction/rendering runs in a
separate process with a 300-second document deadline; Tesseract has a 60-second
per-page deadline and at most 50 OCR pages per document. Rendering is capped at
200 DPI, 16 million pixels and 6,000 pixels per side. Source images are limited to
40 million pixels and downscaled, never upscaled. Temporary files use a private
directory and are removed after completion/failure; temporary storage also needs
disk encryption. A worker timeout stops its own process group, including OCR.
These are application limits, not a full sandbox or a hard memory limit for every
native parser; only ingest trusted files.

Re-ingest sources to publish corrected v2 versions; historical originals and
extractions remain available. Run `index` afterward to index current versions.

The SQLite database contains **unencrypted original personal documents and their
text and search embeddings**, including historical versions. Protect it and backups using appropriate
OS permissions and disk encryption. Original sources are never modified.
Catalogue metadata and text shown on the terminal are also sensitive. Logging
records stages, counts, statuses, timings, and safe failures, not contents or
source names; safe document/attempt/version IDs correlate ingestion events.
Failed attempts log a fixed reason once; extraction status is logged only after
parser diagnostics are classified. No deletion command
is provided yet, so retained versions and backups persist.

## Ollama in Docker

On Apple Silicon, prefer the native Ollama app for Metal GPU acceleration.
This workspace was switched to native Ollama with user approval; the Docker
container was stopped and its existing models copied to the native store.
Both use the same loopback API. Check `ollama ps` for GPU offloading.
To return to Docker, quit native Ollama first; the Docker volume is retained.

Stop native Ollama before starting a container on the same port:

```bash
docker compose up -d
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull deepseek-r1:8b
docker compose exec ollama ollama pull qwen3.5:4b
```

The API is published only on host loopback, not all network interfaces. It has no
application authentication: do not expose it publicly. Docker volumes are
persistent storage, not encrypted storage. Do not mount credentials or the Docker
socket into the container. On macOS, Docker inference may be CPU-only and slower
than native Ollama. Increase the explicit request timeout if necessary.

For long synthesis jobs on Apple Silicon, check `ollama ps` (or
`docker exec ollama ollama ps`) for the PROCESSOR column. `100% CPU` means model
inference is not GPU-offloaded. Native macOS Ollama can use Metal; switching
requires stopping the container occupying port 11434 before starting the native
service and ensuring its separate model store has the required models. Do not run
both services on the same port. This application does not switch services,
download models, or add parallel inference automatically. GPU acceleration is an
execution/deployment choice, independent of synthesis checkpointing and budgets.

For repeatable deployments, set `OLLAMA_IMAGE` to a verified image tag or digest
before running Compose; its development default is `ollama/ollama:latest`.

## Run and rebuild

```bash
.venv/bin/python chatbot_tech_doc.py
.venv/bin/python chatbot_tech_doc.py --model qwen3.5:4b
.venv/bin/python chatbot_tech_doc.py --rewrite-model qwen3.5:4b
.venv/bin/python chatbot_tech_doc.py --rebuild-index
.venv/bin/python chatbot_tech_doc.py --stats
.venv/bin/python chatbot_tech_doc.py --stats-json
.venv/bin/python chatbot_tech_doc.py --stats-json --fail-on-stale
```

`--stats` reports source size, source and indexed document counts, unique warning
counts, and index validation checks for the source/schema/model manifest,
distance metric, and document IDs. Missing indexes show these checks as not
checked. Use `--stats-json` for the same fields as machine-readable JSON, with
boolean validation results and `null` for checks unavailable when the index is
missing. Neither statistics option starts chat, calls Ollama, creates a missing
index, or changes an existing index. Add `--fail-on-stale` with either stats
option to return exit status 1 if the index is missing or any validation check
fails; it returns 0 for a current index. The flag is only valid with a stats
option.

Each table row becomes one complete warning/procedure document. The new index is
`data/chroma_warnings_v1`; the old `data/chroma_db_nomic` is not changed. Its manifest
checks the source hash, row schema, embedding model, cosine metric, and document
IDs. A mismatch requires explicit rebuild. `--rebuild-index` replaces only the
`warning_rows` collection in the configured database, not other collections or
folders. Use a single writer for initialization/rebuild; ingestion is not a
multi-process transaction. A failed build requires another explicit rebuild.

Chat-model changes do not require re-indexing. Embedding-model, parser, source, or
distance-metric changes do. Configuration constants are at the top of the script.

The session retains at most five complete exchanges and a bounded character
budget. Follow-ups use a separate query-rewriting call. Answers include retrieved
source labels; these labels identify candidates, not proof that every generated
claim is correct. Review the manual for safety-critical instructions. Top-K
retrieval can return irrelevant documents; prompt rules cannot guarantee
abstention or prevent prompt injection. Empty retrieval skips answer generation.

### Complete warning lists and follow-ups

Type `/warnings` or ask `list all of the possible warnings` for a complete,
alphabetically sorted list with source filenames. This reads all warning metadata
from the validated index directly; it does not use similarity search, embeddings,
or answer generation. A contextual follow-up such as
`There are other faults. list them too.` also returns the full catalogue.
Recognition is intentionally limited to unfiltered warning-list requests; use
`/warnings` if a phrasing is not recognized. Filtered questions such as
`list warnings about oil` still use similarity retrieval and are not exhaustive.
The catalogue covers this project's indexed warning rows, not every possible
vehicle warning or a future general personal-document library.

Normal questions still retrieve the top two passages. The answer prompt identifies
these as a subset, not an exhaustive list. Follow-up rewriting uses a separate
`qwen3.5:4b` client with thinking disabled and an output budget of
128 tokens, instead of the answer client's 2048 tokens. Rewrites must be one
non-empty line of at most 512 characters after reasoning cleanup. Truncated,
oversized, empty, or multiline rewrites produce an explicit error and skip
retrieval/answer generation; ask a standalone question to proceed. There is no
silent truncation or fallback search with an unresolved follow-up.
Answer generation still defaults to `deepseek-r1:8b`. Override the rewrite model
with `--rewrite-model`; use a model that can reliably emit short responses without
thinking. DeepSeek-R1 exhausted the short rewrite budget in local validation even
with thinking disabled, whereas the installed Qwen model completed the tested
rewrite in 15 tokens. No automatic model download or fallback occurs. The rewrite
model must be installed for normal follow-ups, but complete catalogue queries do
not invoke it.

Type `exit`/`quit` or use Ctrl+C/Ctrl+D to stop. Expected service/output failures
are reported without adding an exchange to history; unexpected bugs propagate.
Interrupting the client does not guarantee immediate server-side cancellation.

## Logging

The CLI reads `logging.json` before initialization and writes UTF-8 logs to
`logs/archive-it.log`. No extra dependency is required. To use another config:

```bash
.venv/bin/python chatbot_tech_doc.py --log-config /path/to/logging.json
```

Configuration fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `level` | `INFO` | Set `DEBUG` for detailed activity and timings |
| `file` | `logs/archive-it.log` | Absolute path or path relative to the config file |
| `max_bytes` | `5242880` | Rotate at approximately 5 MiB per file; minimum 1024 bytes |
| `retained_files` | `5` | Total files including the active log; minimum 2 |

The default retains the active file and four archives (`.1` newest through `.4`
oldest), approximately 25 MiB total. Rotation happens before a record would
exceed the limit; an individual oversized record can exceed it. Settings apply
on restart. Lowering retention does not remove old archives from a previous
larger setting. Use one process per log file; rotation is not multi-process safe.
Missing or invalid configuration and file-opening failures stop CLI startup
with an error rather than silently disabling logging. Logs are ignored by Git.

INFO records lifecycle events, extraction/index counts, index reuse/rebuild,
retrieval/model-request completion, and query outcomes. DEBUG adds stage starts,
elapsed times, input/context sizes, history counts, index validation decisions,
and safe exception stack locations. Failures remain ERROR severity at both
settings and preserve existing error handling.
DEBUG also records model output character counts, cleaned rewrite lengths,
recognized completion reasons (`stop`, `length`, or `unknown`), and numeric token
counts/durations when supplied by Ollama, without logging response text or
arbitrary completion metadata.

Neither level records document contents, source filenames/metadata, question or
history text, prompts, model responses, embeddings, or exception messages.
Exception diagnostics include the exception type; DEBUG includes stack file
paths, function names, and line numbers without source lines or local variables.
Third-party HTTP/model/database debug logs are deliberately not captured because
they can expose personal content. This logs application activity, not Ollama's
server-side internals. Treat logs as private operational data; they are not
encrypted. Programmatic callers can opt in using
`app_logging.configure_logging(config_path)`; importing modules creates no log
files or model clients.

## Tests

### Retrieval and answer acceptance benchmark

The synthetic benchmark separates **evidence selection/prompt packing** from
**live answer fidelity**. It generates temporary TXT, DOCX and text-PDF documents,
ingests and indexes them, and runs nine question cases in each format:

| Case | Required behavior |
| --- | --- |
| Landlord address, straight possessive | Find the contact address, not the agent-name-only disclosure or tenant address |
| Landlord address, curly possessive | Same evidence and answer expectation with a curly apostrophe |
| Landlord address, paraphrase | “Address of the landlord” resolves to the same contact block |
| Specific utilities | Supply and name the utility list, not just the generic payment obligation; any optional charging fee must retain its conditional scope |
| Term dates | Answer with the beginning/end dates, not the signature date |
| Conditional payment | When explicitly asked, retain the optional-use condition, USD 12 amount and per-use billing basis |
| Conflicting lease/addendum fees | Preserve both USD 40 and USD 65 and the conflict/uncertainty; do not invent addendum precedence |
| Missing pool hours | Return insufficient evidence rather than substituting office hours or claiming absence from the original |
| Non-lease supplier address | Retrieve the supplier contact address rather than an address-change rule or delivery location |

Run the deterministic benchmark without Ollama:

```bash
.venv/bin/python -m pytest tests/test_document_acceptance.py -q
```

This runs 27 retrieval/prompt-packing cases, seven negative-control tests and
three positive citation-rendering/equivalent-wording controls.
Uninformative synthetic vectors deliberately test keyword/heading ranking
without assuming good embedding matches. These tests inspect the evidence
**actually packed into the bounded answer prompt**, not only the search results.
They check required evidence, document scope, current version, exact source
offsets, provenance and extraction-quality flags. Answer output is stubbed here;
passing this stage does not measure model accuracy.

Run the same benchmark with installed local models:

```bash
RUN_OLLAMA_TESTS=1 .venv/bin/python -m pytest tests/test_document_acceptance.py -v
```

For independent repetitions of every live case, without rerunning only failures:

```bash
RUN_OLLAMA_TESTS=1 RAG_ACCEPTANCE_REPEATS=3 \
  .venv/bin/python -m pytest tests/test_document_acceptance.py -v
```

`RAG_ACCEPTANCE_REPEATS` must be a positive integer (default 1). Repetitions have
separate `run-N` pytest IDs and report properties; any failed repetition fails
the suite. Repeating a passing deterministic configuration is a stability check,
not a statistical accuracy estimate.

This adds 27 live cases using `nomic-embed-text` and `qwen3.5:4b`. Each must pass
its evidence checks, expected answer/qualification checks and claim-linked
citation checks. The benchmark also rejects selected distractors and checks
explicit insufficient-evidence behavior. Seven deliberately wrong answers test
that the acceptance checks detect missing qualifications, wrong cited evidence,
invented hours and unsupported conclusions. All required cases must pass; there
is no averaged score that hides a failed scenario, no expected-failure marker
and no retries beyond the application's existing corrective retry.

The benchmark keeps Qwen as its historical baseline independently of the
application's answer default. To evaluate the selected DeepSeek answer model:

```bash
RUN_OLLAMA_TESTS=1 RAG_ACCEPTANCE_MODEL=deepseek-r1:8b \
  .venv/bin/python -m pytest tests/test_document_acceptance.py -v
```

The selected answer model is recorded in each live test's report properties.

An earlier run was **60 passed** (33 deterministic checks plus 27 live cases),
but subsequent runs exposed address-answer failures despite correct packed
evidence. With the initial quote-first protocol, a Qwen full run was **51 passed,
9 failed**: the address cases still fail through unscoped absence claims or
unrelated tenant-address content. A prior three-repetition scope-guard-only run
was **84 passed, 30 failed**. These are unresolved fidelity failures, not expected
enumeration failures, and no assertions have been weakened or marked expected.
These measurements are not a general accuracy estimate: acceptance uses explicit normalized text/regex checks,
not a proof of entailment or an LLM grading another LLM. Equivalent unrecognized
wording can need an oracle review; never remove a required fact merely to make
a test pass. Model output may vary across runs, versions and machines.
Pytest IDs identify format, scenario and repetition. Live tests also record case name,
evidence-passage count and model-request count as pytest report properties.

After the DeepSeek-focused fixes, the reported TXT conflict, TXT supplier-contact
and DOCX curly-possessive address scenarios passed **three repetitions each
(9 passed)**. The conflict oracle now accepts “does not clarify if/whether”
alongside “does not state/specify”; simply listing two fees still fails a negative
control. Multiline finding rendering has a deterministic claim-linked-citation
regression. This targeted result is not a full cross-format DeepSeek certification
or proof that future runs cannot truncate or misinterpret evidence.

The subsequent compact-prompt revision was checked on conflict and named-utility
questions across TXT, DOCX and PDF, three repetitions per format/scenario.
All **18 checks passed across two runs**: the first was interrupted after 14
passes because inference was taking unusually long; the remaining four PDF
checks completed separately. The DOCX conflict omission was reproduced before
the change. The PDF truncation was intermittent and did not reproduce in the
initial single-case check. Acceptance assertions, sources, models and token
limits were not relaxed for this revision. These targeted checks do not replace
a full live benchmark or guarantee bounded latency/semantic correctness.

Those earlier targeted passes did not hold on later complete runs. The
source-ID/stable-prompt revision described above was subsequently verified with
DeepSeek on the **entire benchmark**: **64 passed** with one live repetition,
then **118 passed** with three live repetitions (81 live answers plus 37
deterministic/control checks). The three-repetition run took approximately
59 minutes. All original answer claims, qualifications, forbidden conclusions
and claim-linked-citation checks were retained. No token limit was increased
for this revision. The default regression suite passed **698 tests**, with
46 opt-in tests skipped. These results measure this synthetic corpus and
installed model only, not general semantic accuracy or future-run reliability.

The benchmark uses isolated temporary catalogues and fictional values only.
It makes no personal-catalogue changes, downloads no models, and changes no
production prompts or ranking logic. A relevant-looking unselected document
checks scope isolation. It tests the focused retrieval/answer service directly,
not planner routing, exhaustive enumeration, OCR scans, complex tables, or
whole-document synthesis; those require separate tests. Existing live enumeration
accuracy failures remain visible and are not resolved by this benchmark.

### Workspace workflow acceptance

The workspace acceptance suite exercises the actual menu, planner protocol,
query session, retrieval/synthesis services and source inspector with isolated
synthetic TXT, DOCX and PDF catalogues. Default tests replace only model/embedding
responses, rather than mocking the query or synthesis service. They check:

- `ask` routing to search, focused answers and single-pass summaries.
- Original source spans, selected-document isolation and model-free inspection.
- Summary amounts, dates, payment qualifications, conflicts and claim-linked
  citations, with negative controls that deliberately remove or miscite facts.
- Rejection of planner scope expansion and source changes during generation.
- Stale inspection without substituting newer source text.
- Cancellation of hierarchical synthesis followed by reuse of validated
  checkpoints, with no partial result or pending approval published.

Planner instructions distinguish topic-based document summaries from explanations
of explicitly selected passages or clauses. Naming several topics to summarize
does not require a retrieval index. Ollama planning requests constrain selected
document IDs to the current scope in a per-request schema; application validation
still rejects unknown or out-of-scope IDs, without changing the planner's stored
configuration. Synthesis instructions require all supporting source labels for
combined claims and retain conditions/frequencies with their original obligations.
Single-document direct summaries omit redundant document aliases from model-visible
evidence and explicitly request passage label IDs, avoiding confusion between a
document alias and a passage citation. Returned evidence still retains full
document/version provenance; multi-document prompts retain document distinctions.
Synthesis also checks that ISO `YYYY-MM-DD` dates and full English month-name
dates in each finding occur in that finding's cited original passages, accepting
equivalent forms such as `2027-01-01` and `January 1, 2027`. A date cited only to
an unrelated passage is rejected before publication/checkpointing and receives
the existing single corrective retry. A second failure remains explicit.
Input notes, reductions, final answers and reused checkpoints receive this check;
previous checkpoints without the date-support validation key are not reused.
This is a date-presence guard, not semantic entailment: it does not verify which
event a date belongs to, amounts, utility responsibilities, or other date formats.
Live failures include captured workspace errors/output when no successful plan
is published, rather than only reporting a missing result. Qualification checks
accept equivalent single-payment wording such as "once" and "one-time", while
negative controls still reject a monthly deposit.

Run the fast deterministic checks without invoking Ollama:

```bash
.venv/bin/python -m pytest tests/test_document_workspace_acceptance.py -q
```

A separate opt-in smoke suite uses the actual default planner and
`deepseek-r1:8b` answer model for one summary/inspection workflow per format.
It needs the installed `qwen3.5:4b` and `deepseek-r1:8b` models and a running
local Ollama service, but no retrieval index or embedding model: summary routing
does not use retrieval. Run it serially on memory-constrained machines:

```bash
RUN_WORKSPACE_SMOKE_TESTS=1 .venv/bin/python -m pytest \
  tests/test_document_workspace_acceptance.py -k live_workspace -q --durations=3
```

This flag is independent of `RUN_OLLAMA_TESTS`, so the longer focused-answer
benchmark does not also launch the workspace smoke suite. The smoke suite checks
actual planner routing, required summary facts/qualifications, supporting
citations, direct strategy, request bounds and source inspection. It does not
certify general model accuracy or exercise live retrieval, OCR or complex tables.
All fixtures are fictional and temporary; no personal catalogue is modified and
no models are downloaded.

### Other regression and integration tests

```bash
.venv/bin/python -m pytest tests/test_vector_store.py -q --basetemp=.pytest_cache/test-data
.venv/bin/python -m pytest tests/test_logging.py -q --basetemp=.pytest_cache/log-test-data
.venv/bin/python -m pytest tests/test_document_library.py -q --basetemp=.pytest_cache/document-test-data
.venv/bin/python -m pytest tests/test_document_search.py -q --basetemp=.pytest_cache/search-test-data
.venv/bin/python -m pytest tests/test_document_query.py -q --basetemp=.pytest_cache/query-test-data
RUN_OLLAMA_TESTS=1 .venv/bin/python -m pytest tests/test_document_query.py -k live_local -q --basetemp=.pytest_cache/live-query-data
RUN_OLLAMA_TESTS=1 .venv/bin/python -m pytest tests/test_document_search.py -k live_local -q --basetemp=.pytest_cache/live-document-data
RUN_OLLAMA_TESTS=1 .venv/bin/python -m pytest tests/test_vector_store.py -k live_local -q --basetemp=.pytest_cache/live-test-data
```

Default tests use isolated databases and synthetic embeddings. Optional local
tests evaluate retrieval for oil-pressure and airbag questions against the actual
downloaded embedding model, and a short follow-up rewrite using the installed
`qwen3.5:4b` model, without generating answers. Review more paraphrases,
ambiguous questions, and unrelated questions before adopting a score threshold.

Use `python -m pip check` for dependency consistency and `pip-audit` for known
dependency advisories. Advisory checks send package/version metadata to advisory
services, not source documents. Review findings and upgrade compatible dependency
sets together; exact version pins alone do not establish safety.

## Validated dependency upgrade and temporary Chroma exception

The October 14, 2026 upgrade pins `langchain-core==1.6.10`,
`langchain-ollama==1.1.0`, and `langchain-chroma==1.1.0`. ChatOllama now uses
`reasoning=True`, not the removed `extract_reasoning` parameter, to separate
reasoning from answer text. Models, loopback endpoints, source labels, and
conversation history behavior are unchanged.

Validate or run the upgraded set separately from the existing environment:

```bash
.venv/bin/python -m venv .venv-upgrade
.venv-upgrade/bin/python -m pip install --upgrade 'pip>=26.2'
.venv-upgrade/bin/python -m pip install -r requirements.txt pytest pip-audit
RUN_OLLAMA_TESTS=1 .venv-upgrade/bin/python -m pytest tests/test_vector_store.py -q --basetemp=.pytest_cache/upgrade-test-data
.venv-upgrade/bin/python -m pip check
.venv-upgrade/bin/python -m pip_audit --progress-spinner off
.venv-upgrade/bin/python chatbot_tech_doc.py
```

Do not recreate an existing `.venv-upgrade` just to rerun validation; activate or
use its interpreter directly. The upgrade leaves `.venv`, existing database
directories, and the running Ollama container untouched. Validation uses separate
test databases, not an existing index migration. Back up persistent data before
opening it with upgraded dependencies; compatibility with existing on-disk
databases was not tested by this upgrade.

Validation on Python 3.13.12 passed all 64 tests, including both optional live
retrieval cases using the already-installed `nomic-embed-text` model, and
`pip check` reported no broken requirements. No models were downloaded.
`pip-audit` 2.10.1, after upgrading the isolated environment's bundled pip from
25.3 to 26.2.1, reported only ChromaDB findings: five entries representing four
unique advisories (one duplicate). No LangChain advisories were reported in that
environment. This is a point-in-time advisory check, not a security certification.
The old `.venv` was not patched or audited again and may retain the previously
reported unused-package and installer vulnerabilities.

**User-approved temporary exception:** retain `chromadb==1.5.9` for embedded-only
local storage despite these unresolved advisories, for which the audit records
no fixed version:

- [GHSA-f4j7-r4q5-qw2c](https://github.com/advisories/GHSA-f4j7-r4q5-qw2c):
  server collection-creation code execution via `trust_remote_code`.
- [GHSA-36p7-vc44-83pf](https://github.com/advisories/GHSA-36p7-vc44-83pf):
  server collection-update code execution via `trust_remote_code`.
- [GHSA-xph7-9rjv-w5fr](https://github.com/advisories/GHSA-xph7-9rjv-w5fr):
  RBAC permission scope bypass across tenants/databases/collections.
- [GHSA-2wm9-hf6c-p5cr](https://github.com/advisories/GHSA-2wm9-hf6c-p5cr):
  missing server tenant authorization validation.

This application uses an embedded `PersistentClient`, not a Chroma HTTP server
or HTTP client, and does not use Chroma server APIs, RBAC providers, remote model
repositories, or `trust_remote_code`. Embeddings come from the configured local
Ollama client. These constraints reduce exposure to the described paths; they
are **not proof that exploitation is absent**. No patch was available in the
verified release/advisory data at review time. Keep the database local and
trusted, recheck advisories when dependencies change, and revisit this exception
**before any Chroma server, multi-tenant, or remote-code deployment**. Do not
treat this prototype as dependency-security certified.
