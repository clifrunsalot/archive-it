# Archive It technical guide

Start with the [README](./README.md) for installation and a quick introduction.
Use this guide for commands, workflows, limits, and implementation details.

All shell examples run from the repository root. Replace uppercase placeholders
such as `DOCUMENT_ID` with values from your catalogue.
`VERSION_ID` must be an integer returned by `versions`. Examples use a
macOS/Linux shell.

## Contents

- [Local models and dependencies](#local-model-configuration)
- [Import, index, search, and ask](#personal-document-catalogue-and-querying-phases-1-5)
- [Query planning and chat](#query-planning-presentation-and-interactive-sessions)
- [Summaries and comparisons](#whole-document-summaries-and-comparisons-step-4)
- [Experimental enumeration](#explicit-enumeration-step-5)
- [Quote-backed inventory records](#evidence-backed-inventory-foundation-step-6-experimental)
- [Review inventory candidates](#human-acceptance-of-inventory-candidates)
- [Typed facts and deterministic queries](#typed-facts-corrections-and-deterministic-record-queries)
- [Guided workspace](#guided-document-workspace)
- [Fact workbench](#interactive-fact-workbench)
- [Calculation previews](#calculation-preview-and-explicit-execution)
- [Natural-language fact proposals](#natural-language-reviewed-fact-proposals)
- [Extraction and provenance](#supported-extraction-and-provenance)
- [Quality and limitations](#quality-and-limitations)
- [OCR setup and limits](#local-ocr-setup-and-limits)
- [Ollama in Docker](#ollama-in-docker)
- [Warning chatbot](#run-and-rebuild)
- [Logging](#logging)
- [Tests](#tests)
- [Dependency validation and Chroma exception](#validated-dependency-upgrade-and-temporary-chroma-exception)

## Local model configuration

- Install pinned dependencies from `requirements.txt` into a virtual environment.
- Use a fresh environment when checking the dependency set. Installing the
  manifest does not remove older OpenAI or Unstructured packages.
- The application connects to Ollama at `127.0.0.1:11434` and ignores proxy
  environment settings.
- Source text and questions go to that local service, not a hosted model API.
- LangSmith tracing and optional Chroma telemetry are disabled.
- Models must already be installed. The application never downloads them.

| Role | Default model | Override |
| --- | --- | --- |
| Personal answers, summaries, and comparisons | `deepseek-r1:8b` | `--model` |
| Personal query planner | `qwen3.5:4b` | `--planner-model` |
| Personal embeddings | `nomic-embed-text` | `--embedding-model` |
| Direct enumeration | `qwen3.5:4b` | `--model` |
| Enumeration routed through `ask` or `chat` | `qwen3.5:4b` | `--enumeration-model` |
| Warning chatbot answers | `deepseek-r1:8b` | `--model` |
| Warning follow-up rewriting | `qwen3.5:4b` | `--rewrite-model` |

## Personal document catalogue and querying (phases 1-5)

`document_library.py` manages a local, versioned document catalogue. It is
separate from the bundled warning chatbot and its Chroma index.

### Keeping personal material out of Git

The `.gitignore` excludes:

- Private-data folders, backups, exports, logs, and databases.
- Common document and image formats.
- Structured JSON/CSV exports.
- Credentials, local editor settings, and Python environments.

Source code, `requirements.txt`, `logging.json`, and the bundled
`data/mg-zs-warning-messages.html` remain eligible for version control.

Before committing:

1. Put personal notes in `private/`, `personal/`, or `references/`, regardless of
   format. Those folder names are ignored at any directory level.
2. Check `git status` and the staged diff. Arbitrary filenames and Markdown notes
   elsewhere are not automatically recognized as private.
3. Review new fixtures before adding narrow ignore exceptions. Do not broadly
   unignore private storage.
4. Keep `logging.json` and `.env.example` free of secrets and personal paths.

**Ignore rules are not access controls or encryption.** They do not protect
already tracked files, and `git add -f` bypasses them.

### Import and inspect documents

Ingestion and extraction do not call Ollama.

```bash
.venv/bin/python document_library.py ingest /path/to/lease.pdf /path/to/receipt.docx /path/to/notes.txt
.venv/bin/python document_library.py ingest /path/to/documents-folder
.venv/bin/python document_library.py list
.venv/bin/python document_library.py show DOCUMENT_ID
.venv/bin/python document_library.py versions DOCUMENT_ID
.venv/bin/python document_library.py show DOCUMENT_ID --version VERSION_ID
.venv/bin/python document_library.py export DOCUMENT_ID /path/to/new-original-copy.pdf
```

Command behavior:

- `list`, `show`, and `versions` return JSON.
- `show` includes extracted text and provenance.
- `export` writes the retained original bytes. It refuses to overwrite a file.
- Folder ingestion is non-recursive. It attempts every immediate file except
  macOS `.DS_Store` and `._*` metadata.
- Unsupported files produce errors rather than being silently skipped.
- Batch ingestion continues after expected per-file errors. It exits nonzero
  if any file fails.

The default database is `data/document_library/catalogue.sqlite3`, ignored by Git.
Put global options **before** the command:

```bash
.venv/bin/python document_library.py \
  --database /path/to/catalogue.sqlite3 \
  --log-config /path/to/logging.json list
```

### Index, search, and ask

After importing documents:

1. Run `index` to build keyword and vector search data.
2. Use `search` to inspect candidate passages.
3. Use `ask` for a generated answer with citations.
4. Run `index` again after importing changes.

```bash
.venv/bin/python document_library.py index
.venv/bin/python document_library.py search 'policy ZX-789'
.venv/bin/python document_library.py search 'Are pets permitted?' --document DOCUMENT_ID
.venv/bin/python document_library.py ask 'What is the monthly rent?' --document DOCUMENT_ID
.venv/bin/python document_library.py ask 'What is the deductible?' --model qwen3.5:4b
```

**Scope and output**

- Repeat `--document` to select multiple IDs.
- Without a selection, queries search all current, successfully extracted
  documents.
- Unknown IDs and selected documents without successful extraction fail. The
  query does not broaden its scope.
- `search` returns JSON passages with paths, IDs, versions, hashes, locations,
  extraction issues, and coverage.
- `ask` returns an answer, numbered citations, and source references.
- Original source files are not needed after ingestion.

**Version handling**

- Unchanged versions reuse stored vectors.
- Selected current versions must be indexed. Queries never silently use an
  older indexed version.
- A failed extraction retains the prior successful version. Source labels and
  coverage report the failed latest attempt.
- Documents with no successful version are excluded from unrestricted searches
  and counted in coverage.

**Models and rebuilds**

- `index --document DOCUMENT_ID` indexes only that selection.
- Use the same `--embedding-model` tag for indexing and querying.
- Model or configuration changes require `index --rebuild`.
- A rebuild replaces search data only, not originals or catalogue versions.
- A failed build rolls back keyword and vector changes together.
- A rebuild with a document selection leaves other documents unindexed.
- Model tags are recorded, not digests. Rebuild explicitly if you replace a
  model behind the same tag.

#### How retrieval works

| Part | Behavior |
| --- | --- |
| Storage | SQLite FTS5 keyword records and normalized embeddings commit atomically |
| Semantic search | Exact cosine scan over selected current-version vectors |
| Keyword search | Literal query tokens ranked with BM25 |
| Fusion | Reciprocal rank fusion merges and deduplicates candidates |
| Passage limit | `--limit` defaults to 6; allowed range is 1-10 |
| Scaling | Linear time and memory as passage counts grow |

Use a single writer. Index transactions can remain open during local embedding
requests. Larger libraries will need an indexed vector backend with equivalent
version-consistency guarantees.

Passage construction:

- Long segments become bounded, overlapping passages. They never cross source
  segments.
- Locations retain page/block/row/line references and character offsets.
- DOCX headings and first-table-row context are included in embedding input.
  The first row is not a verified semantic header.
- Input fits conservatively below the default embedding context. Smaller
  models may need different chunking.

Ranking preferences:

1. Exact punctuated identifiers, such as `ZX-789`.
2. Term-date evidence for date questions.
3. Matching field headings for questions beginning with what/which/who/where.
4. Remaining keyword and semantic candidates.

The field-heading heuristic:

- Prefers a `Supplier:` contact block over an incidental supplier mention.
- Normalizes basic plurals, such as `UTILITY` and "utilities".
- Removes straight or curly possessive suffixes.
- Gives uncommon heading terms more weight than repeated generic terms.
- Accounts for heading length. `LANDLORD:` beats a partial match such as
  `Landlord or Agent's Name:`.
- Gives the immediately following overlapping passage a smaller preference
  when a passage ranks, preserving nearby clause context and helping lists or
  conditions continue beyond a chunk boundary.

For address questions, a matching party heading with a street address can
exclude name-only absence notes and other parties' address fields from model
input.

These are narrow heuristics, not address extraction or semantic guarantees.
They use existing passages; no rebuild is needed. A passage limit is not a
completeness measure, and the following passage may still not contain every
related term or condition.

Broad focused questions asking for document "terms", "conditions", or
"requirements" add targeted fee-and-amount searches alongside the main query
(including an "ends before expiration" variant for termination questions).
Related overlapping chunks from termination-fee results are retained ahead of
unrelated source segments, while the broader results still favor distinct
segments. The model is also
asked to cover notice periods, duration, conditions, and exceptions where
supported, and to identify requested details that are not established by the
supplied passages. This improves discovery and makes gaps more explicit, but
does not make retrieval exhaustive. The focused answer prompt allows up to
8,000 bytes to fit more adjacent clause passages while remaining bounded.
The answer is requested as five labeled fields (fee/amount, notice period,
duration, conditions, and exceptions). If validation rejects the default
answer model for a broad terms question, the configured alternate
(`--enumeration-model`, Qwen by default) gets one attempt; failed validation
still blocks publication.

#### Focused-answer evidence and validation

Personal `ask`, `chat`, and `workspace` use DeepSeek by default. Override with
`--model qwen3.5:4b` or another installed model.

- One-shot `ask` requests are standalone.
- `chat` retains scope and a successful plan in memory.
- Empty scopes skip generation.
- Hybrid retrieval answers focused passage questions, not whole-document
  coverage. Use synthesis for summaries and comparisons.

Prompt packing:

- Evidence must fit both a UTF-8 byte budget and a character budget.
- Only passages actually supplied receive labels.
- Prompts include stable document numbers, passage labels, text, locations, and
  quality flags.
- Random catalogue IDs, hashes, and paths remain in returned provenance records,
  not model prompts.
- Different document versions receive distinct model-visible document numbers.
- Full evidence metadata still counts toward the returned-evidence character
  budget.
- Space is reserved for a corrective retry.

Focused model adapters must return findings in this form:

```json
{"status": "findings", "findings": [{"support": [1], "text": "Supported answer."}]}
```

- `support` must contain valid integer labels from the supplied passages.
- The application retains original text and provenance, then renders citations.
- Findings are whitespace-normalized so wrapped addresses stay with citations.
- The former `{citation, quote}` protocol is rejected.
- Synthesis uses its separate `{text, citations}` finding protocol.

Validation rules:

- Reject missing/unknown labels, malformed output, and truncation.
- Allow up to eight findings in focused answers and final synthesis answers.
- Limit intermediate synthesis notes to three findings.
- Allow at most one corrective retry; never publish truncated answers.
- Reject common unscoped claims that information is not recorded or provided.
  Absence must be scoped to a field, passage, or clause, or use the explicit
  insufficient-evidence status.
- Permit verbatim supported negative rules, such as "No warranty is provided",
  unless they assert document-wide absence.
- Require conflict conclusions to cite all used passages and retain uncertainty
  about precedence.
- Keep named lists together where practical, rather than repeating one finding
  per item.

Coverage fields:

| Field | Meaning |
| --- | --- |
| `support_protocol=source-passages-v1` | Source-ID protocol is in use |
| `source_references_verified` | Finding labels passed validation |
| `semantic_support_verified=false` | Interpretation is not verified |
| `insufficient_evidence` | Supplied passages did not support an answer |
| `model_reasoning_responses` | Responses with extracted reasoning, including rejected attempts |

`support_quotes_verified` is no longer emitted. Valid labels do not prove
relevance, entailment, correct scope, or completeness. The absence guard is also
a wording heuristic, not a semantic checker.

#### Reasoning and retry behavior

- Ollama receives the JSON schema through the API, not repeated in prose.
- DeepSeek-R1 uses explicit reasoning extraction. Reasoning is never displayed
  or logged, but shares the output-token budget.
- Other focused models request disabled thinking.
- First requests normally use 2048 output tokens.
- If a DeepSeek-R1 response truncates, its single retry uses 4096 tokens, or a
  larger configured limit.
- Other models and other validation failures do not receive that increase.
- Transient local server errors share the same retry allowance.
- There is no third call, silent model fallback, or mutation of client defaults.

DeepSeek can still exhaust its budget or take longer than Qwen. Earlier local
probes found that disabling thinking was unreliable; a 512-token schema probe
produced only reasoning. Qwen answered one tested bed-bug question in one call,
while DeepSeek truncated twice. These observations are not accuracy guarantees.

Restart an open workspace to pick up changed defaults. Changing the answer model
does not require re-ingestion or re-indexing.

**Always inspect important sources.** Semantic search can return unrelated
passages, and there is no calibrated relevance cutoff. Prompt rules cannot
guarantee grounding or prevent document prompt injection. Insufficient evidence
means unknown, not absent from the original. Full-corpus totals, rankings, and
professional legal/tax advice are not supported by passage answers.

### Query planning, presentation, and interactive sessions

`ask` runs a structured planner before retrieval. The plan separates:

- Operation and factual search question.
- Document scope.
- Presentation format.
- Clarification needs and requested filters.

The planner uses Qwen with thinking disabled, schema-constrained JSON, and a
bounded generation budget. `--planner-model` is independent of the answer
`--model`.

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

| Request | Handling |
| --- | --- |
| Find passages, explain, or extract a value | Focused retrieval |
| Summarize, compare, or explicitly enumerate | Whole-extraction pipeline |
| Supported calculation | Reviewed-fact proposal; explicit approval required |
| Grouped ranking or professional advice | Unsupported |
| Passage date/tag/party/historical-version filter | Stop with guidance to select documents explicitly |
| Undefined category or ambiguous document | Request clarification |

Reviewed date facts separately support inclusive date bounds. A request such as
"component with the most warnings" remains aggregation even when formatted as
a table. Enumeration does not add component classifications or calculations.

Planner safeguards:

- Validate JSON, fields, operations, scope, IDs, sizes, and completion status.
- Accept multiline JSON and harmless search-question wrapping.
- Never fall back to a broader query after invalid output.
- Send only the latest question, permitted IDs/filenames, and prior successful
  plan. Do not send document contents or answer text.
- Require fewer selected documents if the planning prompt exceeds its budget.

A valid plan can still misinterpret intent. Inspect `plan` for complex requests.

#### Presentation

Use `prose`, `list`, `ascii_table`, or `csv`. `--format` overrides the proposed
format.

- Formatting is deterministic, not another model task.
- Tables and CSV use Type/Content rows for answers, sources, and coverage.
- They do not invent domain-specific columns or typed facts.
- ASCII tables escape non-ASCII characters and wrap rows.
- CSV quotes values and guards leading spreadsheet formulas.
- Citations and quality information remain attached.

#### Chat commands and follow-ups

| Command | Effect |
| --- | --- |
| `/select ID [ID ...]` | Select documents and clear prior context |
| `/all` | Restore all current successful documents and clear context |
| `/scope` | Show selected IDs; `null` means all |
| `/reset` | Clear context but keep document selection |
| `exit`, `quit`, Ctrl+D, Ctrl+C | End the session |

- Successful scope selections persist for follow-ups.
- References use prior search intent, not prior answers as evidence.
- Unrelated questions should not inherit the prior topic.
- A format-only follow-up re-renders the cached result without retrieval or
  answer generation. It first checks scope, versions, and ingestion status.
- Changed documents require a fresh question.
- Failed or unsupported turns do not replace successful context.
- To resolve ambiguity, use `/select` and restate the question. Clarification
  replies are not stored as a separate dialogue.
- Sessions do not persist between processes.

The direct `search` command bypasses planning and answer generation.
Programmatically, `DocumentSearch.ask` is the focused API; `QuerySession` is the
planned conversational interface.

### Whole-document summaries and comparisons (step 4)

These operations read **all stored extracted text** for selected current
versions. They do not re-read originals or certify extraction completeness.

| Operation | Required scope |
| --- | --- |
| Summary | 1-4 explicitly selected documents |
| Comparison | 2-4 distinct, explicitly selected documents |

No search index or embeddings are required.

```bash
# Direct operations bypass model intent classification
.venv/bin/python document_library.py summarize 'Summarize obligations and deadlines.' --document DOCUMENT_ID --model qwen3.5:4b
.venv/bin/python document_library.py compare 'Compare rent, exceptions, and pet obligations.' --document FIRST_ID --document SECOND_ID --model qwen3.5:4b --format ascii_table

# The planner also routes these operations through ask and chat
.venv/bin/python document_library.py ask 'Summarize the entire lease.' --document DOCUMENT_ID --model qwen3.5:4b
.venv/bin/python document_library.py chat --document FIRST_ID --document SECOND_ID --model qwen3.5:4b
```

DeepSeek is the default. Qwen is a locally tested alternative. DeepSeek reasoning
is extracted and counted, not displayed or logged. Other models request disabled
thinking, which they may not honor.

#### Processing steps

1. Split extracted segments into non-overlapping UTF-8-bounded passages.
   Preserve every character and exact source offsets.
2. For a summary that fits the final prompt, generate directly from the original
   passages. Do not create intermediate notes.
3. For larger summaries and all comparisons, read every passage into cited notes.
4. Reduce notes per document when needed, then generate the final answer.
5. Validate output and confirm that source versions have not changed.

Limits and coverage:

| Setting or field | Value or behavior |
| --- | --- |
| Prompt limit | 6,000 UTF-8 bytes |
| Corrective-retry reserve | 640 bytes |
| Input batches | At most 128 |
| Reduction levels | At most 8 |
| Invocation budget | 600 seconds or 80 actual calls, including retries |
| `synthesis_strategy` | `direct`, `hierarchical`, or `inventory` |
| `input_batches` | Source partitions, not model-call count |

- Oversized or non-shrinking notes fail with guidance to narrow the request.
- Direct summaries skip input-note calls. They may use two calls after a rejected
  response, or zero when reusing a checkpoint.
- No source text is sampled or truncated to enable the direct path.
- Batches with no relevant evidence are excluded from reduction and tracked.
- A document with no findings is still reported with reviewed references and
  uncertainty. It is not silently dropped from comparisons.
- If no document has findings, reduction and final calls are skipped.

#### Budgets, progress, and resuming

Override direct-command budgets when needed:

```bash
.venv/bin/python document_library.py summarize 'Summarize the rent terms.' --document DOCUMENT_ID --model qwen3.5:4b --max-seconds 600 --max-calls 80
```

- Budgets are checked before each call. They do not interrupt a call in progress.
- Budget failures publish no partial answer.
- Progress goes to the application log. Workspace heartbeats also use stderr,
  leaving direct-command prose/CSV output clean.
- Progress includes stage, batches, elapsed time, calls, and reused checkpoints.
- Input ETA excludes reduction/final work. It is omitted when reuse makes it
  misleading.

For Ollama clients, validated input, reduction, and final results are saved
individually in `synthesis_checkpoints`.

To resume:

1. Rerun the identical request with the same model configuration.
2. Completed calls reuse validated checkpoints.
3. The new invocation gets a fresh budget.
4. Use `--no-resume` after replacing model contents behind the same tag.

Checkpoint rules:

- Failed and cancelled calls are never cached.
- Reuse requires matching requests, versions, hashes, ingestion status, prompts,
  schemas, allowed citations, and model configuration/endpoint.
- Non-Ollama runnables do not persist checkpoints: they lack reliable identity.
- Malformed checkpoints fail explicitly.
- Historical checkpoints remain stored as **unencrypted sensitive text**.

#### Validation and failure handling

- Outputs use schema-constrained findings and integer citation IDs.
- Reject truncation, excessive size, missing citations, and out-of-batch labels.
- Final answers must cite every selected document.
- Returned citations resolve to original passages, not just generated notes.
- Intermediate notes carry separate text and citation arrays. Inline citation
  markers in model finding text remain invalid.
- A `no_relevant_evidence` batch preserves reviewed references without asserting
  absence from the original.
- Invalid output gets one corrective retry using fixed diagnostics, not echoed
  rejected text.
- A second failure stops processing. Failed generations are not cached.
- DeepSeek truncation retries use the focused-answer 4096-token policy.
- HTTP 500/502/503/504 and in-stream status `-1` errors share the same retry
  allowance. Other service errors stop immediately.
- Service retries keep the original prompt. Logs include status/stage, not
  server error text.
- Source changes invalidate the result.
- Failed requests do not replace successful chat context.
- Format-only follow-ups reuse results with version checks.

Coverage reports processed segments/characters, versions, quality, OCR status,
latest ingestion status, actual calls, and reasoning-response counts.

An opt-in regression checks the synthetic manual-workspace summary for amounts,
utilities, lease dates, and one-call direct processing:

```bash
RUN_OLLAMA_TESTS=1 .venv/bin/python -m pytest tests/test_document_synthesis.py \
  -k live_deepseek_manual_lease_summary -v
```

**Processing every extracted passage is not a verified or exhaustive summary.**
Notes can omit or misinterpret details. Comparisons must preserve conflicts and
uncertainty, not assume precedence. This pipeline adds no verified calculations,
automatic revision pairing, exhaustive inventories, or domain classifications.
The separate warning chatbot is unchanged.

### Explicit enumeration (step 5)

Use enumeration for requests such as "list every amenity" or "enumerate the
stated renewal conditions".

```bash
.venv/bin/python document_library.py enumerate 'List every amenity in the lease.' --document DOCUMENT_ID
.venv/bin/python document_library.py ask 'List every renewal condition in the lease.' --document DOCUMENT_ID
```

- Select 1-4 documents explicitly.
- Enumeration reads all stored extracted text, not top-k search results.
- No search index is required.
- Candidate items remain separate; they are not reduced into a summary.
- Direct `enumerate` uses Qwen unless `--model` overrides it.
- In `ask`/`chat`, use `--enumeration-model` independently of the answer model.
- There is no silent model substitution.

Routing guards send explicit "list/enumerate all/every/each" item requests to
enumeration. Literal requests for matching passages, files, or documents remain
search requests.

#### Amenity-specific checks

A candidate must have a cited sentence that names it and explicitly states
access, availability, or provision as an amenity.

- Hypothetical, conditional-only, or prohibited mentions do not establish access.
- Explicit optional or paid offers retain their conditions.
- "All or part" of a facility group becomes an ambiguous group, not a list of
  every possible member.
- Other categories do not receive this narrow relevance gate.

Items merge only within a document version when names, conditions, assessments,
and uncertainty match after basic case/whitespace/trailing-period normalization.
Different conditions, documents, or versions stay separate.

#### Output and limits

- Terminal output uses compact bullets, filename/page references, and one
  coverage warning.
- API results retain full provenance.
- Each input batch allows up to 16 cited items.
- Time, call, and output limits fail explicitly; items are not dropped to fit.
- Truncated output is discarded and the passage batch splits recursively.
- If one passage still truncates, the request fails.
- Itemized output and serialized support have a 256,000-character limit,
  separate from the 12,000-character summary limit.
- Validated Ollama calls can resume from checkpoints.
- Unexpected reasoning stops enumeration with guidance to use Qwen. Reasoning
  text is not logged.

Local DeepSeek probes produced hidden reasoning despite `think=false`, sometimes
taking minutes on CPU. A bounded 512-token probe produced no final answer.
Enumeration therefore retains its disabled-thinking guard rather than using the
extended summary retry budget.

Run the amenity relevance regression:

```bash
RUN_OLLAMA_TESTS=1 .venv/bin/python -m pytest tests/test_document_synthesis.py -k live_enumeration -q
```

**Enumeration remains experimental.** `exhaustive` stays false even when all
extracted text was processed. Models can miss or misinterpret items; extraction
can omit original content. "No matching items identified" is not proof of
absence. The narrow regression checks are not general relevance certification.

### Evidence-backed inventory foundation (step 6, experimental)

`AnswerResult.items` stores structured candidates:

- Name and conditions.
- Model assessment: `established` or `ambiguous`.
- Uncertainty.
- Document/version identity.
- Supporting evidence.

`established` is a model judgment, not an application-verified fact.

#### Quote validation

1. Ollama selects exact source passages or sentence/paragraph strings through
   the response schema.
2. A validator checks each quote against the cited input passage.
3. Citations are derived from validated support.
4. Invalid quotes/labels get one corrective retry. A second failure publishes
   no completed result.

Supporting records include:

- Original quote and source path.
- Document/version IDs, segment, and location.
- Zero-based, end-exclusive offsets within the extraction segment.

Offsets count Python Unicode characters, not bytes or PDF coordinates. Repeated
quotes retain all matching spans. Input splitting preserves characters and
prefers sentence/paragraph boundaries, but long clauses and page boundaries can
still separate context.

Get full records as JSON:

```bash
.venv/bin/python document_library.py enumerate 'List every amenity in the lease.' \
  --document DOCUMENT_ID --records-json
```

- `--records-json` cannot be combined with `--format`.
- JSON includes private text and paths; protect exported files.
- Compact output is labeled experimental. `Needs review` marks model ambiguity.
- Quote-backed batches are checkpointed privately and revalidated on resume.
- Protocol/schema changes stop reuse of older free-form checkpoints without
  deleting them.

**Quote provenance does not verify relevance, entailment, availability,
completeness, or numerical accuracy.** Records report
`semantic_relevance_verified=false`.

Full-lease testing verified spans and zero-call replay but still found unrelated
items and wrong interpretations. The live relevance regression remains a release
gate, not a passing certification. No totals, professional conclusions, domain
profiles, or automatic semantic deduplication are enabled.

### Human acceptance of inventory candidates

Completed enumerations save candidates locally. New candidates start
`unreviewed`; matching candidates can retain earlier human decisions.

| Output label | Meaning |
| --- | --- |
| `Unreviewed candidate` | No human acceptance |
| `Accepted by reviewer` | Explicit human decision |
| `Needs review` | Model-reported uncertainty; independent of the human decision |

Accepting an ambiguous item does not remove its uncertainty.

1. Inspect saved records without a model call:

   ```bash
   .venv/bin/python document_library.py inventory-review \
     'List every amenity in the lease.' --document DOCUMENT_ID
   ```

2. Use the **same question and selected documents** as the enumeration.
3. Check exact quotes, offsets, `review_id`, and the original document.
4. Record a decision:

   ```bash
   .venv/bin/python document_library.py inventory-review \
     'List every amenity in the lease.' --document DOCUMENT_ID \
     --item REVIEW_ID --decision accepted --note 'Checked the source'

   .venv/bin/python document_library.py inventory-review \
     'List every amenity in the lease.' --document DOCUMENT_ID \
     --item REVIEW_ID --decision rejected --note 'Unrelated to the requested category'
   ```

5. Use `--decision unreviewed` to reset a decision.

Review behavior:

- Full JSON includes private source text and paths.
- Older results may need one new enumeration to populate review records.
- Notes are optional, limited to 2,000 characters, stored locally, and not logged.
- Changes append timestamped `review_history` events; they never erase history.
- This is a local workflow, not authenticated multi-user approval.
- Saved decisions reapply only to matching questions, versions, candidate
  details, and supporting spans. Citation renumbering alone does not invalidate
  a match.
- Changed candidates need new review. Old payloads, questions, and decisions
  remain in private audit tables.
- Stale versions or changed ingestion status require a new enumeration.
- Rejected candidates disappear from compact output but remain in JSON.
- Coverage counts accepted, rejected, and unreviewed records.
- If all are rejected, output says so rather than claiming no matches exist.
- Cached reruns and format-only follow-ups refresh decisions without extraction
  calls. Format-only follow-ups keep their original candidate set.

Human acceptance cannot recover missed items or correct extraction gaps.
`semantic_relevance_verified` stays false, even after acceptance. Inventory review
does not enable calculations; typed facts require separate confirmation.

### Typed facts, corrections and deterministic record queries

Typed facts are human-authored dates, money, quantities, or text.

- Saving or correcting a fact does **not** confirm it.
- Inventory acceptance does not transfer to a fact.
- An optional `candidate_id` links to an archived candidate from the same
  document/version.

#### Create a fact

1. Inspect the current extraction with `show`, or inspect enumeration support.
2. Prepare a payload containing **all** fields below.
3. Replace the example identity and support with real values.
4. Check that the quote supports your interpretation.

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

This is a schema example, not a fact about your documents.

- Segments are one-based.
- Character offsets are zero-based and end-exclusive.
- At least one exact supporting span is required.
- Quotes and offsets are checked against the current extraction.
- Matching quotes do not prove that a typed value follows from the source.

| Kind | Value | Metadata |
| --- | --- | --- |
| `date` | Valid ISO `YYYY-MM-DD` | Empty unit, currency, and period |
| `money` | Decimal string, never a JSON float | Three-uppercase-letter currency; supported period; empty unit |
| `quantity` | Decimal string | Explicit unit; empty currency and period |
| `text` | Nonempty string | Empty unit, currency, and period |

Value rules:

- Money periods: `once`, `daily`, `weekly`, `monthly`, or `yearly`.
- Numbers allow an optional minus sign, 24 integer digits, and 6 fractional
  digits.
- Thousands separators, exponents, NaN, and Infinity are rejected.
- Currency codes are syntax-checked, not registry-checked.
- No currency, unit, or period conversions are inferred.
- Unresolved uncertainty blocks deterministic queries even after confirmation.

#### Save, confirm, and correct

```bash
# Save a proposed fact from a private JSON file
.venv/bin/python document_library.py fact-save fact.json

# Inspect the returned fact_id and revision
.venv/bin/python document_library.py facts --document DOCUMENT_ID

# Confirm only after checking the value and source
.venv/bin/python document_library.py fact-review FACT_ID \
  --revision 1 --decision confirmed --note 'Checked value and source'

# Correct with a complete replacement payload
.venv/bin/python document_library.py fact-save corrected-fact.json \
  --fact FACT_ID --revision 1

# Inspect immutable revisions and review history
.venv/bin/python document_library.py fact-history FACT_ID
```

- Corrections create a new `unreviewed` revision.
- Previous payloads and decisions remain inspectable.
- Facts cannot move to another document.
- Stale revision numbers are refused.
- Review decisions also include `rejected` and `unreviewed`.
- Source-version or ingestion-status changes make a fact stale.
- To use a stale fact, save a corrected revision against current evidence and
  confirm it.
- Superseded revisions cannot be queried.

#### Run deterministic queries

These commands make **no model calls**:

```bash
.venv/bin/python document_library.py fact-query count --fact FACT_A --fact FACT_B
.venv/bin/python document_library.py fact-query sort --fact DATE_A --fact DATE_B \
  --from-date 2026-01-01 --through-date 2026-12-31
.venv/bin/python document_library.py fact-query compare --fact FACT_A --fact FACT_B
.venv/bin/python document_library.py fact-query sum --fact MONEY_A --fact MONEY_B
```

Every selected fact must be current, confirmed, and free of uncertainty.
Invalid selections fail together; nothing is silently skipped.

| Operation | Rules |
| --- | --- |
| `count` | Counts records, not entities or numeric values; mixed kinds allowed; duplicate IDs rejected |
| `sort` | Compatible values only; ascending; ties by fact ID |
| `compare` | Exactly two compatible facts; reports first value as `less`, `equal`, or `greater` |
| `sum` | Exact decimals; money/quantity only; matching kind, unit, currency, period, and conditions |

- ISO dates sort chronologically, numbers numerically, and text lexicographically.
- Comparison records are sorted for display; `selected_fact_ids` preserves
  comparison order.
- Sums reject overlapping source spans to reduce double-counting. This does not
  detect every semantic duplicate.
- Inclusive date bounds apply only to date count/sort. No matches produces zero
  or an empty list.
- JSON includes selected IDs, contributing records, evidence, locations, quality,
  revisions, review history, units, periods, conditions, and scope.

The selected IDs are the inclusion rule. The application does not decide which
charges belong together, resolve alternatives, or infer eligibility. Results
describe selected confirmed **records**, not complete document totals.

Protect fact payloads, notes, quotes, and exports. They are private data and are
not logged. Natural-language queries can propose selections, but still require
explicit execution.

### Guided document workspace

Use a terminal menu instead of copying document IDs:

```bash
.venv/bin/python document_library.py workspace --model qwen3.5:4b

# Use an isolated acceptance-test catalogue
.venv/bin/python document_library.py \
  --database "$HOME/Documents/archive-it-test/catalogue.sqlite3" \
  workspace --model qwen3.5:4b

# Equivalent if you already defined the archive_test shell function
archive_test workspace --model qwen3.5:4b
```

#### Select documents

1. Choose an initial document by displayed number or exact filename.
2. Use numbers for duplicate filenames; full paths distinguish them.
3. Use **scope** to replace the active set, for example `1,3`.
4. Check each document's extraction, index, warnings, and fact counts.

- Every selected document needs a successful extraction.
- Stale fact counts overlap confirmed/unreviewed/rejected counts.
- A stored index is not a health guarantee. Search still validates compatibility
  and consistency.
- Document selection persists across actions, not across processes.

#### Menu examples

Enter an action name or its displayed number at the **Action** prompt, not the
shell. The examples are illustrative.

| Action | Use it to | Example |
| --- | --- | --- |
| **index** | Prepare the active scope for search | Run before first search and after importing a version |
| **search** | View candidate passages without an answer | `monthly parking` |
| **ask** | Ask a planned question or request a fact proposal | `What is the monthly rent?` |
| **summarize** | Read all selected extracted text | `Summarize the charges and term dates.` |
| **compare** | Compare 2-4 selected documents | `Compare the rent and term dates.` |
| **scope** | Replace selected documents | `1,3` |
| **facts** | Open the model-free workbench | `add`, then `review-pending` |
| **calculate** | Manually preview a fact query | Select `1,2`, choose `sum`, inspect, then `run` |
| **selection** | Replace facts in a pending proposal | `1,3` or `1-3`; comparison order matters |
| **approve** | Review and run a ready proposal | Enter `run`; Enter alone cancels |
| **cancel-query** | Discard a pending proposal | Ask again for a fresh preview later |
| **documents** | Return to document selection | Choose a number or exact filename |
| **inspect** | Read evidence from the last completed result | Choose citation `1`, then `segment` |
| **quit** | Exit | Saved facts and decisions remain |

Action rules:

- Index/search/ask use the active scope. Indexing is never automatic.
- Summaries and comparisons need no search index and allow at most four
  documents. Citations do not establish which agreement controls.
- Fact editing and manual calculation need a single-document scope.
- **selection**, **approve**, and **cancel-query** need a pending **ask**
  proposal. Manual **calculate** has its own preview/run flow.
- Blocked previews cannot be approved.
- Scope changes clear prior query/inspection state and pending approval.
- Missing indexes require **index**. Incompatible or damaged indexes require the
  explicit CLI `index --rebuild`; the workspace does not rebuild automatically.

#### Read search results and inspect evidence

Workspace **search** displays passages in retrieval order:

- Numbered headers identify filename, version, segment, and available locations.
- Passage text is wrapped without omission.
- Extraction issues, failed latest ingestion, and coverage remain visible.
- Numbers match **inspect**.
- The standalone CLI `search` still returns `passages`/`coverage` JSON.

To inspect:

1. Complete a search, answer, summary, or comparison.
2. Choose **inspect**, then a citation number.
3. Read the supplied source passage separately from the model interpretation.
4. Use **segment**, **previous**, or **next** for additional extracted context.
5. Press Enter to leave context, then Enter to return to the workspace.

Inspection details:

- Shows exact text, full paths, version, segment, location, quality, and
  character offsets.
- Use the standalone `search` JSON output for document/passage IDs and hashes.
- Marks supplied-but-uncited passages.
- Labels nearby context as **not evidence used for the answer**.
- Disables nearby navigation if the source version changed.
- Flags later failed ingestion.
- Makes no model calls and does not verify interpretation.
- Keeps results only for the workspace session.
- New actions, scope changes, or cancellation clear inspection results.
  Approval actions do not generate new evidence.

#### Progress, cancellation, and privacy

- Service actions show stage, model, and elapsed time on stderr.
- Heartbeats appear every five seconds while waiting.
- Generation shows batch/request/retry counts; synthesis adds checkpoints.
- Progress contains no source text, questions, answers, or hidden reasoning.
- Ctrl+C during an action returns to the menu. EOF exits the workspace.
- `/cancel` abandons the current prompt.
- Cancellation clears pending approval and incomplete query/inspection state.
- Saved facts, decisions, and validated checkpoints remain.
- Rerun identical synthesis requests to reuse checkpoints.
- Ollama may still finish an already submitted request. Client cancellation
  does not stop the shared server.

Browsing, status, fact review, and manual calculations need no model calls.
Query clients are created only when needed. Override models with `--model`,
`--planner-model`, `--embedding-model`, or `--enumeration-model`.

Expected service/validation failures return to the menu without changing
documents. There is no automatic ingestion, rebuild, model download, or bulk
confirmation.

Terminal output contains private paths, text, and facts. Protect captures as
carefully as exports. Scope selection does not prove that documents belong
together or establish precedence.

#### Example: review facts and calculate monthly charges

Suppose a synthetic document states:

- Rent: USD 1200 monthly.
- Parking: USD 50 monthly.
- Deposit: USD 500 once.

1. Start the workspace and select the already-ingested document.
2. Choose **index**, then **search** for `monthly parking` if you want to inspect
   retrieval.
3. Choose **facts** and **add** each money fact with its exact evidence.
4. Set currency to `USD`; use `monthly`, `monthly`, and `once` respectively.
5. **Save**, then **confirm** each fact after checking it. Alternatively, choose
   **leave** and review later with **review-pending**.
6. Return to the workspace. **Ask**:
   `What is the total of the confirmed monthly charges?`
7. Check that rent and parking are selected and the deposit is omitted.
   Inspect evidence, periods, conditions, and blocking reasons.
8. If needed, use **selection** with the displayed rent/parking numbers. Do not
   assume they are always 1 and 2.
9. Choose **approve**, inspect the preview again, then enter **run**.
10. Expect USD 1250 monthly over those selected records, not a verified complete
    document total.

For a model-free route, use **calculate**, select rent and parking, choose
**sum**, inspect, and **run**. Including the one-time deposit must block the sum
because periods differ; it must not silently omit the deposit.

To cancel, use **cancel-query** or press Enter at the approval menu. New
questions, processing actions, workbench entry, manual calculations, document
changes, and exit discard pending natural-language approvals. Saved facts and
review events are never rolled back by navigation.

### Interactive fact workbench

The workbench is local and model-free. It avoids manual JSON and offset entry.

```bash
.venv/bin/python document_library.py fact-workbench --document DOCUMENT_ID

# Also browse a saved single-document inventory
.venv/bin/python document_library.py fact-workbench --document DOCUMENT_ID \
  --question 'List every amenity in the lease.'
```

It operates on one document at a time.

| Action | Behavior |
| --- | --- |
| **add** | Author a fact with selected source evidence |
| **similar** | Start a new fact using editable kind/unit/currency/period defaults |
| **candidate** | Prefill from a saved candidate; question must match exactly |
| **edit** | Create a corrected, unreviewed revision |
| **review** | Confirm or reject the current revision |
| **review-pending** | Review each unreviewed fact in sequence |
| **calculate** | Select facts, preview compatibility, then choose whether to run |
| **quit** | Exit |

**similar** copies only type/unit/currency/period defaults. It never copies name,
value, evidence, conditions, uncertainty, candidate link, or confirmation. Select
fresh evidence and explicitly save/review the new fact.

**candidate** does not accept the inventory candidate or confirm the new fact.

**review-pending**:

- Shows evidence and asks confirm/reject/leave separately for each record.
- Enter leaves the record unchanged.
- Confirmed/rejected facts are not queued.
- Stale facts need correction with fresh evidence.
- `/cancel` stops the queue but preserves recorded decisions.

#### Add or correct a fact

1. Choose a segment and read its full text.
2. Enter an exact substring, or press Enter for the whole segment, up to
   4,000 characters.
3. If the substring repeats, choose its occurrence using offsets and context.
4. Add more supporting spans if needed. The workbench computes offsets.
5. Enter name, kind, value, and applicable units/currency/period.
6. Enter conditions and uncertainty explicitly.
7. Inspect the draft and source context.
8. Choose **save**. The default is **cancel**, which writes nothing.
9. Separately choose **confirm**, **reject**, or **leave**. The default is
   **leave**.

Input tips:

- Enter keeps a displayed default.
- `/clear` empties a text field.
- Select a whole segment for multiline evidence, or use JSON for finer spans.
- Invalid values can be edited or cancelled; no invalid revision is published.
- Optional private review notes are stored, not logged.

Cancellation and consistency:

- `/cancel` abandons an action.
- EOF/Ctrl+C exits the workbench; unsaved drafts are discarded.
- Saved revisions and decisions remain.
- Saving without confirmation leaves a fact unreviewed.
- Version/revision checks prevent silent overwrite of changed data.
- Stale facts need fresh evidence and a corrected revision.
- There is no bulk confirmation, inferred value, or implicit approval.

For repeated entry, **add** the first fact, use **similar** for the next, then
review drafts with **review-pending**.

Keep terminal captures private. The workbench does not improve automatic
semantic extraction or prove completeness.

### Calculation preview and explicit execution

In the workbench:

1. Choose **calculate**.
2. Select comma-separated fact numbers in the desired order.
3. Choose `count`, `sort`, `compare`, or `sum`.
4. For date count/sort, optionally supply inclusive ISO bounds.
5. Inspect the preview.
6. Choose **run**, or press Enter to **cancel**.

Selection rules:

- `1-3,5` selects 1, 2, 3, 5.
- `3,1-2` selects 3, 1, 2. Order matters for comparison.
- These are catalogue-number ranges, not date/value filters.
- Duplicate, overlapping, descending, or out-of-range selections are rejected.
- At most 128 records may be selected.
- Ineligible records block the preview; ranges do not silently filter them.

The preview shows:

- Selected values, revisions, confirmation, conditions, uncertainty, and quality.
- Exact supporting evidence.
- Passed checks or blocking reasons.
- Selected IDs excluded by date bounds.
- Unselected document facts separately from query inputs.
- Operation, bounds, and selected-record scope.

The preview does **not** compute results. It makes no completeness claim and
infers no conversions or inclusion rules.

- Blocked previews offer no run option.
- `/cancel`, EOF, and Ctrl+C stop execution.
- Calculations do not save facts or change review decisions.

#### Scripted previews

```bash
.venv/bin/python document_library.py fact-query sum \
  --fact FACT_A --fact FACT_B --preview

# Copy preview_token from the result
.venv/bin/python document_library.py fact-query sum \
  --fact FACT_A --fact FACT_B --expected-preview PREVIEW_TOKEN
```

- `--preview` and `--expected-preview` are mutually exclusive.
- The token binds operation, ordered IDs, bounds, revisions, source status, and
  review history.
- Any change, even a new confirmation event, requires a fresh preview.
- Workbench execution always uses this guard.
- Calculation reads one consistent execution-time database snapshot.
- The token is a consistency check, not authentication or an approval credential.
- A blocked CLI preview returns `ready=false` with reasons. That is successful
  inspection, not a successful calculation.
- Unknown IDs, duplicates, and malformed query shapes fail explicitly.
- Plain `fact-query` calls still calculate directly, with confirmation and
  compatibility checks.

### Natural-language reviewed-fact proposals

First author and confirm typed facts, then select documents explicitly:

```bash
.venv/bin/python document_library.py ask \
  'What is the total of the confirmed monthly charges?' --document DOCUMENT_ID
.venv/bin/python document_library.py chat --document DOCUMENT_ID
```

The planner proposes an operation and ordered eligible facts, **not an answer**.

Supported:

- Record count.
- Ascending compatible-value sort, including dates.
- Two-value comparison.
- Compatible money/quantity sums.
- Inclusive ISO bounds for date count/sort only.

Unsupported:

- Conversion or annualization.
- Professional judgment.
- Grouped rankings or entity counts.
- Descending sorting.
- Completeness claims.

Explicit conversion, annualization, ranking, and descending-sort wording is
guarded before planning. Other interpretation still depends on the model.
Ambiguous requests should produce clarification.

#### Check both selections and omissions

The preview includes selected evidence, conditions, revisions, checks, and a
numbered catalogue.

- Omitted records retain evidence and reasons: stale, unconfirmed, uncertain,
  or simply not proposed.
- Invented, duplicate, or ineligible planner IDs fail.
- Incompatible facts block the preview rather than being dropped.
- A model can prefilter date candidates incorrectly. Out-of-range omitted dates
  have deterministic bounds notes, but you must still inspect the whole set.
- Small selections do not prove a complete inventory.

#### Approve in chat

| Command | Effect |
| --- | --- |
| `/facts-select 1,2` | Replace selection in that order and create a new preview |
| `/facts-run TOKEN` | Run the exact current ready preview |
| `/facts-cancel` | Discard without calculating |

- First/second order matters for comparison.
- Choosing an ineligible record blocks execution.
- Ordinary replies such as "yes" never approve.
- A new question, `/select`, `/all`, or `/reset` discards the proposal.
- To change operation or bounds, ask again or use `fact-query --preview`.
- Any catalogue change, including omitted/new facts or review events, invalidates
  the chat proposal.
- Execution also checks the selected-record token.
- Proposals are session-only and single-use. They save no facts or decisions.

#### One-shot requests and planner limits

- One-shot `ask` prints a preview and shell-quoted `fact-query` command.
- The command uses the same database and `--expected-preview` token.
- It is **not executed automatically**.
- Edit selections through chat, the workbench, or a new explicit preview.
- The printed command checks selected records, not later changes to omitted
  facts. Ask again if the catalogue changes.
- `plan` shows routing only; it does not call the fact-selection model.
- Fact planning sends typed names, values, units, conditions, and eligibility to
  the local planner, not source quotes or review notes.
- Limits are 128 facts and a 6,000-byte prompt. Larger scopes fail, not sample.
- Narrow the document scope or use explicit fact IDs.
- Preview/execution do not call the answer model or retrieval service.

Keep output private. Selection errors and incomplete human-authored facts remain
possible; approval does not establish a document-wide total.

### Supported extraction and provenance

| Format | Representation | Location |
| --- | --- | --- |
| UTF-8 TXT, optional BOM | Paragraphs with internal line breaks | Original line start/end |
| Text PDF | Per-page text | One-based page |
| Scanned PDF | Local English OCR for textless pages | Page and OCR method |
| PNG/JPEG/single-frame TIFF | Local English OCR | Image and OCR method |
| DOCX | Body paragraphs/headings and table rows | Block/row and heading |

- DOCX table rows are JSON cell arrays, not flattened prose.
- DOCX locations are not rendered page numbers.
- No automatic classifications, inferred dates, or domain fields become facts.

#### Identity and history

Each successful version retains:

- Exact original bytes and SHA-256 hash.
- Format and extractor/parser version.
- UTC ingestion timestamp.
- Extracted segments, issues, and OCR status.

Version rules:

- Identity uses the resolved path, not just the filename.
- Moving a file creates another document.
- Identical bytes at different paths remain separate sources.
- Timestamps are ingestion times, not effective dates.
- Unchanged versions reuse extraction.
- Changed bytes create a new version; old versions remain inspectable/exportable.
- Extractor-profile changes trigger re-extraction even for unchanged bytes.
- Extraction v3 records relevant PDFium, Pillow, fontTools, and Tesseract versions.
- Returning to previously ingested bytes makes that version current again.
- Snapshot, segments, and current-version pointer commit atomically.
- Failed updates preserve the prior successful version and record the failure.
- `list` separates current quality from latest-attempt status.
- Missing source files do not delete retained records.
- Use one ingestion process at a time.

### Quality and limitations

| Status | Meaning |
| --- | --- |
| `extracted` | Supported extraction completed; accuracy/completeness not verified |
| `partial` | Known gaps or review requirements exist |

Partial issues can include:

- PDF pages without text.
- DOCX images.
- Repeated PDF page text or page-edge lines.
- Font or unknown parser diagnostics.
- OCR requiring review.

Both statuses retain issue flags. PDF object-header whitespace warnings alone
do not downgrade quality. Parser messages are not logged.

**PDF**

- Repeated page text triggers page-bounded PDFium re-extraction, not blind
  deduplication.
- Content outside the visible page is excluded.
- Remaining duplicates and repeated page-edge phrases are retained and flagged.
  Repeated clauses can also trigger the warning.
- Layout, tables, column coordinates, and multi-column reading order are not
  verified.

**DOCX**

- Includes body paragraphs/tables and default, first-page, and enabled even-page
  headers/footers.
- Records section/area/block locations.
- Linked header/footer parts are included once.
- Text boxes, tracked revisions, nested-table fidelity, and embedded objects
  remain unverified.
- Embedded images are flagged, not silently OCRed.

Source references help compare extraction with retained originals. They do not
prove that every meaningful field was captured.

### Local OCR setup and limits

Install Tesseract with English language data. On macOS:

```bash
.venv/bin/python -m pip install -r requirements.txt
brew install tesseract
tesseract --list-langs
.venv/bin/python document_library.py ingest /path/to/scanned.pdf /path/to/receipt.png
```

OCR behavior:

- Runs locally; no external OCR service receives content.
- Textless PDF pages use rendering/OCR automatically.
- Images use OCR directly.
- Native PDF text is preferred.
- Images on otherwise text-bearing PDF pages are not automatically OCRed.
- OCR results are `partial`, with `ocr_used=true`,
  `ocr_text_requires_review`, and `method=ocr_tesseract_eng`.
- Empty extraction, missing engines/language data, timeouts, and resource limits
  fail explicitly with fixed reasons.
- Important amounts, identifiers, tables, handwriting, and signatures still
  require comparison with originals.

Unsupported inputs include password-protected PDFs, legacy `.doc`, multi-frame
images, HEIC, and macro documents.

#### Resource limits

| Resource | Limit |
| --- | --- |
| Non-PDF input | 25 MiB |
| PDF input | 160 MiB |
| PDF pages | 1,000 |
| Expanded DOCX/package or PDF page content | 100 MiB |
| Extracted characters | 2,000,000 |
| Page-bounded worker deadline | 300 seconds per document |
| Tesseract deadline | 60 seconds per page |
| OCR pages | 50 per document |
| Render resolution | At most 200 DPI |
| Render size | 16 million pixels; 6,000 pixels per side |
| Source image size | 40 million pixels |

- Images are downscaled, never upscaled.
- Extraction/rendering uses a separate worker for page-bounded processing.
- Worker timeouts stop its process group, including OCR.
- Private temporary files are removed after completion/failure.
- Temporary storage also needs disk encryption.
- These are application limits, not a full sandbox or hard memory limits for
  every native parser. **Ingest trusted files only.**

Re-ingest to obtain corrected extraction, then run `index`. Historical originals
and extractions remain available.

#### Storage privacy

- The SQLite catalogue contains **unencrypted originals, text, embeddings, and
  historical versions**.
- Protect the database, backups, and temporary storage with permissions and disk
  encryption.
- Original source files are never modified.
- Terminal text and metadata are sensitive.
- Logs use stages, counts, timings, statuses, and safe IDs, not content or names.
- Failed attempts log a fixed reason. Extraction status is logged after
  diagnostics are classified.
- There is no deletion command yet. Retained versions and backups persist.

## Ollama in Docker

On Apple Silicon, prefer native Ollama for Metal acceleration. Native and Docker
services use the same loopback API but separate model stores.

1. Stop native Ollama before binding a container to port 11434.
2. Start Compose and install the required models:

   ```bash
   docker compose up -d
   docker compose exec ollama ollama pull nomic-embed-text
   docker compose exec ollama ollama pull deepseek-r1:8b
   docker compose exec ollama ollama pull qwen3.5:4b
   ```

3. Check `ollama ps`, or `docker compose exec ollama ollama ps`, for the
   PROCESSOR column. `100% CPU` means inference is not GPU-offloaded.
4. To switch back to native Ollama, stop the container first and ensure the
   native store has the required models.

Deployment notes:

- Never run both services on the same port.
- The Compose API binds to host loopback only. It has no application
  authentication; do not expose it publicly.
- Docker volumes persist data but do not encrypt it.
- Do not mount credentials or the Docker socket.
- Docker on macOS may use CPU inference and run more slowly. Adjust request
  timeouts if needed.
- The application does not switch services, download models, or add parallel
  inference automatically.
- GPU acceleration is independent of checkpoints and budgets.
- For repeatable deployments, set `OLLAMA_IMAGE` to a verified tag or digest.
  The development default is `ollama/ollama:latest`.

## Run and rebuild

These commands run the **separate warning chatbot**, not the personal catalogue.

```bash
.venv/bin/python chatbot_tech_doc.py
.venv/bin/python chatbot_tech_doc.py --model qwen3.5:4b
.venv/bin/python chatbot_tech_doc.py --rewrite-model qwen3.5:4b
.venv/bin/python chatbot_tech_doc.py --rebuild-index
.venv/bin/python chatbot_tech_doc.py --stats
.venv/bin/python chatbot_tech_doc.py --stats-json
.venv/bin/python chatbot_tech_doc.py --stats-json --fail-on-stale
```

#### Statistics and index validation

- `--stats` reports source size, document counts, warning counts, and checks.
- `--stats-json` returns the same information as JSON.
- Checks cover the source/schema/model manifest, metric, and document IDs.
- Missing indexes report unavailable checks as `null` in JSON.
- Statistics never start chat, call Ollama, create an index, or alter one.
- `--fail-on-stale` is valid only with a stats option. It returns 1 for a missing
  or mismatched index and 0 for a current one.

Index rules:

- Each warning-table row becomes one complete warning/procedure document.
- Storage is `data/chroma_warnings_v1`; `data/chroma_db_nomic` is untouched.
- The manifest checks source hash, row schema, embedding model, cosine metric,
  and document IDs.
- Mismatches require explicit rebuild.
- `--rebuild-index` replaces only `warning_rows`, not other collections/folders.
- Use one writer for initialization/rebuild. This is not a multi-process
  ingestion transaction.
- Failed builds need another explicit rebuild.
- Answer-model changes need no re-indexing.
- Embedding, parser, source, or metric changes do.
- Configuration constants are at the top of `chatbot_tech_doc.py`.

Normal questions retrieve two passages. Source labels identify candidates, not
verified claims. Empty retrieval skips generation. Check the original manual
for safety-critical instructions; prompts cannot guarantee abstention or prevent
prompt injection.

### Complete warning lists and follow-ups

To list the complete indexed warning catalogue:

- Enter `/warnings`.
- Or ask `list all of the possible warnings`.
- A recognized contextual request, such as
  `There are other faults. list them too.`, also returns the full list.

The list is alphabetically sorted and includes source filenames. It reads
validated warning metadata directly, with no embeddings, similarity search,
rewrite call, or answer generation.

Limits:

- Recognition covers unfiltered warning-list requests. Use `/warnings` if your
  wording is not recognized.
- `list warnings about oil` uses similarity retrieval and is not exhaustive.
- The catalogue covers indexed rows, not every possible vehicle warning.

Follow-up rewriting:

| Setting | Value |
| --- | --- |
| History | At most five complete exchanges, plus a character budget |
| Rewrite model | Qwen, thinking disabled |
| Rewrite output budget | 128 tokens |
| Rewrite shape | One nonempty line, at most 512 characters after cleanup |
| Answer output budget | 2048 tokens by default |

- Truncated, oversized, empty, or multiline rewrites fail before retrieval.
- Ask a standalone question to proceed; there is no unresolved fallback search.
- Use `--rewrite-model` for another model that reliably emits short responses.
- Local validation found DeepSeek exhausted the rewrite budget; Qwen completed
  the tested rewrite in 15 tokens.
- Normal follow-ups need the rewrite model installed. Catalogue lists do not.
- No automatic model download or fallback occurs.
- `exit`, `quit`, Ctrl+C, or Ctrl+D stop chat.
- Expected service/output failures do not enter history; unexpected bugs
  propagate.
- Client interruption does not guarantee server-side cancellation.

## Logging

The CLI reads `logging.json` before initialization. Default output is UTF-8 at
`logs/archive-it.log`. No extra dependency is needed.

```bash
.venv/bin/python chatbot_tech_doc.py --log-config /path/to/logging.json
```

| Field | Default | Meaning |
| --- | --- | --- |
| `level` | `DEBUG` | Shipped setting; use `INFO` for less detail |
| `file` | `logs/archive-it.log` | Absolute path or relative to the config file |
| `max_bytes` | `5242880` | About 5 MiB per file; minimum 1024 |
| `retained_files` | `5` | Active file plus archives; minimum 2 |

#### Rotation and startup

- Defaults retain the active log plus `.1` through `.4`, about 25 MiB total.
- Rotation occurs before a record exceeds the limit; one oversized record can
  still exceed it.
- Settings apply on restart.
- Lower retention does not remove archives from a previous larger setting.
- Use one process per log file. Rotation is not multi-process safe.
- Invalid config or file-opening failures stop startup rather than disable logs.
- Logs are ignored by Git but are not encrypted.

#### What is recorded

| Level | Content |
| --- | --- |
| INFO | Lifecycle, extraction/index counts, reuse/rebuild, retrieval/request completion, outcomes |
| DEBUG | Stage starts, timing, input/context sizes, history counts, index decisions, safe stack locations |
| ERROR | Failures at either configured level |

DEBUG also records output lengths, cleaned rewrite lengths, recognized completion
reasons, and numeric token/duration metadata when supplied by Ollama.

Never recorded:

- Document contents, source filenames/metadata, or embeddings.
- Questions, history, prompts, responses, or exception messages.
- Source lines or local variables from stack traces.

DEBUG stack records include code paths, functions, and line numbers. Third-party
HTTP/model/database debug logs are not captured because they can expose content.
Application logs do not cover Ollama server internals.

Programmatic callers can use `app_logging.configure_logging(config_path)`.
Importing modules creates no log files or model clients.

## Tests

Default tests use isolated catalogues and synthetic data. Live tests require
installed local models and explicit opt-in flags.

Install the test runner first if needed:

```bash
.venv/bin/python -m pip install pytest
```

### Retrieval and answer acceptance benchmark

The benchmark separates evidence selection from live answer quality. It creates
fictional TXT, DOCX, and text-PDF documents and tests nine scenarios per format.

| Case | Required behavior |
| --- | --- |
| Landlord address, straight possessive | Use the landlord contact, not agent-name-only or tenant fields |
| Landlord address, curly possessive | Same evidence and answer |
| Landlord address, paraphrase | Resolve "address of the landlord" to the contact block |
| Specific utilities | Name the list and preserve optional charging conditions |
| Term dates | Use beginning/end dates, not signatures |
| Conditional payment | Keep optional use, USD 12, and per-use billing |
| Conflicting fees | Keep USD 40 and USD 65 plus precedence uncertainty |
| Missing pool hours | Report insufficient evidence, not office hours or document-wide absence |
| Supplier address | Use the supplier contact, not delivery or address-change text |

#### Run deterministic checks

```bash
.venv/bin/python -m pytest tests/test_document_acceptance.py -q
```

- Runs 27 retrieval/prompt cases, seven negative controls, and three positive
  citation/equivalent-wording controls.
- Uses uninformative synthetic vectors to exercise keyword/heading ranking.
- Checks evidence actually packed into the bounded prompt, not just search hits.
- Checks scope, versions, offsets, provenance, and quality.
- Stubs answer generation; it does not measure model accuracy.

#### Run live checks

```bash
RUN_OLLAMA_TESTS=1 .venv/bin/python -m pytest tests/test_document_acceptance.py -v

# Repeat every live case independently
RUN_OLLAMA_TESTS=1 RAG_ACCEPTANCE_REPEATS=3 \
  .venv/bin/python -m pytest tests/test_document_acceptance.py -v

# Evaluate DeepSeek instead of the historical Qwen baseline
RUN_OLLAMA_TESTS=1 RAG_ACCEPTANCE_MODEL=deepseek-r1:8b \
  .venv/bin/python -m pytest tests/test_document_acceptance.py -v
```

- Live cases use `nomic-embed-text` and Qwen unless overridden.
- `RAG_ACCEPTANCE_REPEATS` is a positive integer; default 1.
- Repetitions get separate `run-N` IDs. Any failure fails the suite.
- Each case checks evidence, expected claims/qualifications, linked citations,
  distractors, and insufficient-evidence behavior.
- Wrong-answer controls detect missing qualifications, incorrect citations,
  invented hours, and unsupported conclusions.
- There is no averaged score, expected-failure marker, or retry beyond the
  application's existing corrective retry.
- Reports include model, case, evidence count, and model-request count.

Passing repetitions are stability checks, not statistical accuracy estimates.
The oracle uses normalized text/regex checks, not entailment proof or another
LLM. Equivalent wording may need oracle review; do not remove required facts to
force a pass.

#### Historical validation results

These are recorded development runs, not promises for new environments.

| Stage | Recorded result | Interpretation |
| --- | --- | --- |
| Earlier baseline | 60 passed: 33 deterministic + 27 live | Later runs exposed failures despite correct evidence |
| Qwen quote-first full run | 51 passed, 9 failed | Address omissions and unrelated tenant-address answers |
| Scope-guard-only, three repetitions | 84 passed, 30 failed | Fidelity failures remained unresolved at that stage |
| Targeted DeepSeek fixes | 9 passed | Three repeats each: TXT conflict, TXT supplier, DOCX curly possessive |
| Compact-prompt checks | 18 passed across two runs | Conflict/utilities across three formats; interrupted after 14, then four completed |
| Source-ID/stable prompts, full DeepSeek | 64 passed | One live repetition |
| Source-ID/stable prompts, repeated DeepSeek | 118 passed | 81 live + 37 deterministic/control checks; about 59 minutes |
| Regression suite at that revision | 698 passed, 46 skipped | Historical count, not the current suite size |

Additional notes:

- Targeted fixes did not initially hold on full runs.
- The conflict oracle accepts "does not clarify if/whether"; merely listing two
  fees still fails a negative control.
- Multiline findings have deterministic claim-linked-citation coverage.
- DOCX conflict omission was reproduced before the compact-prompt change.
- PDF truncation was intermittent.
- Required claims, qualifiers, forbidden conclusions, and citation checks were
  retained. The compact/source-ID revisions did not relax token limits.
- Live output can vary with model, version, machine, and run.

Benchmark boundaries:

- Fictional temporary data only; no personal catalogue changes.
- No model downloads or production prompt/ranking modifications.
- An unselected distractor checks scope isolation.
- Tests focused retrieval/answers directly, not planner routing, OCR, complex
  tables, enumeration, or whole-document synthesis.
- Enumeration accuracy failures remain separate and unresolved by this suite.

### Workspace workflow acceptance

This suite exercises actual menus, planner protocols, query sessions,
retrieval/synthesis, and source inspection with synthetic TXT/DOCX/PDF catalogues.
Default tests replace model/embedding responses, not query/synthesis services.

Checks include:

- Search, focused-answer, and direct-summary routing.
- Exact source spans, scope isolation, and model-free inspection.
- Amounts, dates, conditions, conflicts, and linked citations.
- Negative controls that remove or miscite facts.
- Rejection of planner scope expansion and source changes.
- Stale inspection without substituting newer text.
- Cancelled hierarchical synthesis and checkpoint reuse without partial results
  or pending approval.

Protocol safeguards covered:

- Topic-based summaries read the document without requiring an index.
- Explicit clause/passage explanations remain distinct.
- Per-request planner schemas constrain IDs to the current scope.
- Application validation still rejects unknown/out-of-scope IDs.
- Synthesis keeps conditions/frequencies attached to obligations and cites all
  supporting passages.
- Direct single-document summaries omit redundant document aliases. Multiple
  documents remain distinct.
- Returned evidence retains full provenance.

**Date-presence guard**

- Each finding's ISO dates and full English month-name dates must occur in its
  cited originals.
- Equivalent forms such as `2027-01-01` and `January 1, 2027` are accepted.
- Dates supported only by unrelated passages fail before publication/caching.
- Input notes, reductions, final answers, and checkpoints receive this check.
- Old checkpoints without its validation key are not reused.
- Failures share the single corrective retry.
- This checks presence, not event identity or entailment. It does not verify
  amounts, utility responsibilities, or other date formats.

Live failures report captured workspace output/errors. Qualification checks
accept "once" and "one-time"; negative controls still reject monthly deposits.

Run deterministic checks:

```bash
.venv/bin/python -m pytest tests/test_document_workspace_acceptance.py -q
```

Run the separate live summary/inspection smoke suite:

```bash
RUN_WORKSPACE_SMOKE_TESTS=1 .venv/bin/python -m pytest \
  tests/test_document_workspace_acceptance.py -k live_workspace -q --durations=3
```

- Needs running Ollama, Qwen, and DeepSeek.
- Needs no index or embedding model because summaries bypass retrieval.
- Run serially on memory-constrained machines.
- `RUN_WORKSPACE_SMOKE_TESTS` is independent of `RUN_OLLAMA_TESTS`.
- Checks routing, facts/qualifiers, citations, direct strategy, request bounds,
  and inspection.
- Does not certify general accuracy or test live retrieval, OCR, or complex
  tables.
- Uses fictional temporary fixtures; downloads no models.

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

- Default tests isolate databases and use synthetic embeddings.
- Optional tests cover oil-pressure/airbag retrieval and short Qwen follow-up
  rewriting without answer generation.
- Test more paraphrases, ambiguity, and unrelated questions before choosing a
  relevance threshold.
- Use `python -m pip check` for dependency consistency.
- Use `pip-audit` for known advisories. It sends package/version metadata to
  advisory services, not source documents.
- Review compatible upgrades together. Exact pins alone do not establish safety.

## Validated dependency upgrade and temporary Chroma exception

This section records the **October 14, 2026** upgrade and audit. It is not a
current advisory scan or security certification.

Upgrade details:

- `langchain-core==1.6.10`
- `langchain-ollama==1.1.0`
- `langchain-chroma==1.1.0`
- ChatOllama uses `reasoning=True`, replacing removed `extract_reasoning`.
- Models, loopback endpoints, labels, and history behavior were unchanged.

### Validate an isolated environment

```bash
.venv/bin/python -m venv .venv-upgrade
.venv-upgrade/bin/python -m pip install --upgrade 'pip>=26.2'
.venv-upgrade/bin/python -m pip install -r requirements.txt pytest pip-audit
RUN_OLLAMA_TESTS=1 .venv-upgrade/bin/python -m pytest tests/test_vector_store.py -q --basetemp=.pytest_cache/upgrade-test-data
.venv-upgrade/bin/python -m pip check
.venv-upgrade/bin/python -m pip_audit --progress-spinner off
.venv-upgrade/bin/python chatbot_tech_doc.py
```

- Reuse an existing `.venv-upgrade`; do not recreate it just to rerun checks.
- Isolated validation leaves `.venv`, database directories, and the running
  Ollama container untouched.
- Tests use separate databases, not an existing-index migration.
- Back up persistent data before opening it with new dependencies. Existing
  on-disk compatibility was not tested by this upgrade.

### Recorded results

| Check | Result |
| --- | --- |
| Runtime | Python 3.13.12 |
| Tests | 64 passed, including two optional live retrieval cases |
| Models | Existing `nomic-embed-text`; no downloads |
| `pip check` | No broken requirements |
| Auditor | `pip-audit` 2.10.1 |
| Isolated pip upgrade | 25.3 to 26.2.1 |
| Audit findings | ChromaDB only: five entries, four unique advisories |
| LangChain advisories | None reported in that environment |

The old `.venv` was not patched or re-audited. It may retain earlier unused-package
or installer vulnerabilities.

### Temporary Chroma exception

**User-approved exception:** retain `chromadb==1.5.9` for embedded-only local
storage. At that audit, no fixed version was recorded for these advisories:

| Advisory | Reported issue |
| --- | --- |
| [GHSA-f4j7-r4q5-qw2c](https://github.com/advisories/GHSA-f4j7-r4q5-qw2c) | Server collection-creation code execution via `trust_remote_code` |
| [GHSA-36p7-vc44-83pf](https://github.com/advisories/GHSA-36p7-vc44-83pf) | Server collection-update code execution via `trust_remote_code` |
| [GHSA-xph7-9rjv-w5fr](https://github.com/advisories/GHSA-xph7-9rjv-w5fr) | RBAC scope bypass across tenants/databases/collections |
| [GHSA-2wm9-hf6c-p5cr](https://github.com/advisories/GHSA-2wm9-hf6c-p5cr) | Missing server tenant authorization validation |

Deployment constraints:

- Use embedded `PersistentClient`, not a Chroma HTTP server or client.
- Do not use Chroma server APIs, RBAC providers, remote model repositories, or
  `trust_remote_code`.
- Obtain embeddings from local Ollama.
- Keep databases local and trusted.
- Recheck advisories when dependencies change.
- Revisit the exception **before any server, multi-tenant, or remote-code
  deployment**.

These constraints reduce exposure; they do not prove exploitation is impossible.
Do not treat the prototype as dependency-security certified.
