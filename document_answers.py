"""Shared structured evidence validation, rendering, and bounded generation."""

import json
import re
from datetime import datetime
from collections.abc import Callable

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import Runnable, RunnableBinding
from langchain_ollama import ChatOllama
from ollama import ResponseError

from app_logging import LOGGER
from chatbot_tech_doc import ModelOutputError, clean_deepseek_output, logged_runnable
from document_library import DocumentQueryError
from document_progress import model_name, report_progress

MAX_PROMPT_BYTES = 6000

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["findings", "no_relevant_evidence"]},
        "findings": {"type": "array", "maxItems": 3, "items": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "minLength": 1, "maxLength": 350},
                "citations": {"type": "array", "minItems": 1, "maxItems": 12,
                              "items": {"type": "integer"}},
            },
            "required": ["text", "citations"], "additionalProperties": False,
        }},
    },
    "required": ["status", "findings"], "additionalProperties": False,
}

FINAL_SCHEMA = json.loads(json.dumps(OUTPUT_SCHEMA))

FINAL_SCHEMA["properties"]["findings"]["maxItems"] = 8

FINAL_SCHEMA["properties"]["findings"]["items"]["properties"]["text"]["maxLength"] = 1000

FINAL_SCHEMA["properties"]["findings"]["items"]["properties"]["citations"]["maxItems"] = 32

FOCUSED_SCHEMA = json.loads(json.dumps(FINAL_SCHEMA))
FOCUSED_SCHEMA["properties"]["findings"]["items"] = {
    "type": "object",
    "properties": {
        "support": {"type": "array", "minItems": 1, "maxItems": 32, "items": {
            "type": "integer",
        }},
        "text": {"type": "string", "minLength": 1, "maxLength": 1000},
    },
    "required": ["support", "text"], "additionalProperties": False,
}


def focused_schema(evidence: list[dict]) -> dict:
    schema = json.loads(json.dumps(FOCUSED_SCHEMA))
    schema["properties"]["findings"]["items"]["properties"]["support"]["items"]["enum"] = [
        record["label"] for record in evidence
    ]
    return schema


def truncation_retry_model(model: Runnable) -> Runnable | None:
    if (not isinstance(model, RunnableBinding) or not isinstance(model.bound, ChatOllama)
            or model.bound.model.split(":")[0] != "deepseek-r1"):
        return None
    # num_predict must be a client field; binding it does not change Ollama options.
    client = model.bound.model_copy(update={
        "num_predict": max(model.bound.num_predict or 0, 4096),
    })
    return model.model_copy(update={"bound": client})


def cited_labels(text: str) -> set[int]:
    return {int(label) for label in re.findall(r"\[(\d+)\]", text)}

class SynthesisValidationError(DocumentQueryError):
    """Fixed validation reason safe for diagnostics and corrective requests."""

class SynthesisTruncatedError(DocumentQueryError):
    """A model response was cut off at its output-token limit."""

def parse_findings(
    text: str, allowed: set[int], max_chars: int, *, max_findings: int = 3,
) -> str:
    if len(text) > max_chars * 4:
        raise SynthesisValidationError("output_size_limit")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise SynthesisValidationError("invalid_json") from error
    if not isinstance(value, dict) or set(value) != {"status", "findings"}:
        raise SynthesisValidationError("invalid_schema")
    status, findings = value["status"], value["findings"]
    if (status not in ("findings", "no_relevant_evidence")
            or not isinstance(findings, list) or len(findings) > max_findings):
        raise SynthesisValidationError("invalid_schema")
    if status == "no_relevant_evidence":
        if findings:
            raise SynthesisValidationError("inconsistent_no_relevant_evidence")
        rendered = ("No relevant evidence identified in this reviewed batch; this is not "
                    "proof of absence from the original document. Reviewed sources: "
                    + " ".join(f"[{label}]" for label in sorted(allowed)))
    else:
        if not findings:
            raise SynthesisValidationError("missing_findings")
        rows = []
        for finding in findings:
            if not isinstance(finding, dict) or set(finding) != {"text", "citations"}:
                raise SynthesisValidationError("invalid_schema")
            body, citations = finding["text"], finding["citations"]
            if not isinstance(body, str) or not body.strip() or cited_labels(body):
                raise SynthesisValidationError("invalid_finding_text")
            if not isinstance(citations, list) or not citations:
                raise SynthesisValidationError("missing_citations")
            if any(type(label) is not int or label not in allowed for label in citations):
                raise SynthesisValidationError("citations_outside_batch")
            rows.append(body.strip() + " " + " ".join(
                f"[{label}]" for label in dict.fromkeys(citations)))
        rendered = "\n".join(rows)
    if len(rendered) > max_chars:
        raise SynthesisValidationError("output_size_limit")
    return rendered


CALENDAR_DATE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|"
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s+\d{4}\b", re.IGNORECASE,
)


def calendar_dates(text: str) -> set[str]:
    dates = set()
    for match in CALENDAR_DATE.finditer(text):
        value = match.group()
        if value[0].isdigit():
            dates.add(value)
        else:
            try:
                dates.add(datetime.strptime(
                    " ".join(value.replace(",", "").split()), "%B %d %Y",
                ).date().isoformat())
            except ValueError as error:
                raise SynthesisValidationError("invalid_calendar_date") from error
    return dates


def validate_cited_dates(rendered: str, sources: dict[int, dict]) -> None:
    for finding in re.split(r"(?<=\])\n", rendered):
        claimed = calendar_dates(finding)
        if not claimed:
            continue
        labels = cited_labels(finding)
        supported = set().union(*(calendar_dates(sources[label]["text"]) for label in labels))
        if not claimed <= supported:
            raise SynthesisValidationError("date_not_in_cited_sources")


INFORMATION_ABSENCE = re.compile(
    r"\bno\b[^.!?\n]{0,100}\b(?:recorded|provided|specified|stated|listed|mentioned|"
    r"documented)\b|"
    r"\bnot\s+(?:(?:explicitly|clearly|directly)\s+)?(?:recorded|provided|specified|"
    r"stated|listed|mentioned|documented)\b|"
    r"\bno\b[^.!?\n]{0,80}\b(?:information|details|data)\b"
    r"[^.!?\n]{0,20}\b(?:available|found)\b|"
    r"\bdoes\s+not\s+(?:(?:explicitly|clearly|directly)\s+)?"
    r"(?:record|provide|specify|state|list|mention)\b", re.IGNORECASE,
)
LOCAL_SCOPE = re.compile(
    r"\b(?:field|section|block|passage|addendum|clause)\b|\bsupplied evidence\b|"
    r"\b(?:here|in this (?:contact|name|mailing) (?:listing|block|field))\b",
    re.IGNORECASE,
)
GLOBAL_SCOPE = re.compile(
    r"\b(?:entire|whole)\s+(?:document|source|file)\b|\banywhere in\b|"
    r"\b(?:document|source|file)\s+(?:does not|never)\s+"
    r"(?:record|provide|specify|state|list|mention)\b",
    re.IGNORECASE,
)


def parse_focused_findings(text: str, evidence: list[dict], max_chars: int) -> str:
    if len(text) > max_chars * 4:
        raise SynthesisValidationError("output_size_limit")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise SynthesisValidationError("invalid_json") from error
    if not isinstance(value, dict) or set(value) != {"status", "findings"}:
        raise SynthesisValidationError("invalid_schema")
    findings = value["findings"]
    if not isinstance(findings, list) or len(findings) > 8:
        raise SynthesisValidationError("invalid_schema")
    sources = {record["label"]: record["text"] for record in evidence}
    canonical = []
    for finding in findings:
        if not isinstance(finding, dict) or set(finding) != {"support", "text"}:
            raise SynthesisValidationError("missing_support_references")
        if not isinstance(finding["text"], str) or len(finding["text"]) > 1000:
            raise SynthesisValidationError("invalid_finding_text")
        support = finding["support"]
        if not isinstance(support, list) or not support or len(support) > 32:
            raise SynthesisValidationError("missing_support_references")
        for label in support:
            if type(label) is not int or label not in sources:
                raise SynthesisValidationError("citations_outside_batch")
        canonical.append({"text": " ".join(finding["text"].split()), "citations": support})
    rendered = parse_findings(
        json.dumps({"status": value["status"], "findings": canonical}),
        set(sources), max_chars, max_findings=8,
    )
    for finding in value["findings"]:
        # Selected passages cannot establish document-wide absence of information.
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", finding["text"]):
            if INFORMATION_ABSENCE.search(sentence) and (
                GLOBAL_SCOPE.search(sentence) or (
                    not LOCAL_SCOPE.search(sentence)
                    and not any(" ".join(sentence.split()) in " ".join(sources[label].split())
                                for label in finding["support"])
                )
            ):
                raise SynthesisValidationError("unscoped_information_absence")
    return rendered


def generate(
    model: Runnable, prompt, allowed: set[int], max_chars: int, *,
    stage: str, batch: int | str, stats: dict[str, int], allow_no_relevant: bool = True,
    operation: str = "Synthesis", max_prompt_bytes: int = MAX_PROMPT_BYTES,
    before_call: Callable[[], None] | None = None, max_findings: int = 3,
    retry_truncated: bool = True,
    reject_reasoning: bool = False,
    response_parser: Callable[[str], str] | None = None,
    truncated_retry_model: Runnable | None = None,
) -> str:
    if len(prompt.to_string().encode("utf-8")) > max_prompt_bytes:
        raise DocumentQueryError(f"{operation} request exceeds prompt budget.")
    for attempt in (1, 2):
        report_progress(f"{stage}: batch {batch}, request {stats['requests'] + 1}, "
                        f"attempt {attempt}/2; active model={model_name(model)}")
        if before_call is not None:
            before_call()
        stats["requests"] += 1
        LOGGER.debug("%s generation stage=%s batch=%s attempt=%d allowed_citations=%d",
                     operation, stage, batch, attempt, len(allowed))
        try:
            response = logged_runnable(f"{operation} model request", model).invoke(prompt)
        except ResponseError as error:
            LOGGER.warning("%s service failure stage=%s batch=%s attempt=%d status=%d",
                           operation, stage, batch, attempt, error.status_code)
            # Ollama uses -1 for an error delivered inside a successful HTTP stream.
            if attempt == 1 and error.status_code in (-1, 500, 502, 503, 504):
                continue
            raise
        try:
            if not isinstance(response, AIMessage):
                raise SynthesisValidationError("invalid_message_type")
            has_reasoning = bool(response.additional_kwargs.get("reasoning_content"))
            stats["reasoning_responses"] = stats.get("reasoning_responses", 0) + int(has_reasoning)
            LOGGER.debug("%s response reasoning_present=%s", operation, has_reasoning)
            if reject_reasoning and response.additional_kwargs.get("reasoning_content"):
                LOGGER.error("%s model ignored disabled thinking stage=%s batch=%s",
                             operation, stage, batch)
                raise DocumentQueryError(
                    "Enumeration model generated reasoning despite disabled thinking. "
                    "Use --model qwen3.5:4b for enumerate or --enumeration-model "
                    "qwen3.5:4b for ask/chat. No completed result published."
                )
            if response.response_metadata.get("done_reason") == "length":
                raise SynthesisValidationError("truncated_output")
            try:
                text = clean_deepseek_output(response.text)
            except ModelOutputError as error:
                raise SynthesisValidationError("invalid_model_text") from error
            rendered = (response_parser(text) if response_parser is not None else
                        parse_findings(text, allowed, max_chars, max_findings=max_findings))
            no_relevant = json.loads(text)["status"] == "no_relevant_evidence"
            if no_relevant and not allow_no_relevant:
                raise SynthesisValidationError("relevant_notes_discarded")
            stats["last_no_relevant"] = int(no_relevant)
            if no_relevant:
                stats["no_relevant_results"] = stats.get("no_relevant_results", 0) + 1
                LOGGER.debug("%s relevance stage=%s batch=%s status=no_relevant_evidence",
                             operation, stage, batch)
                if operation == "Focused answer":
                    rendered = rendered.replace(
                        "No relevant evidence identified in this reviewed batch",
                        "Insufficient evidence in the supplied passages to answer the question")
                if stage == "final":
                    rendered = rendered.replace("this reviewed batch", "the processed evidence")
            return rendered
        except SynthesisValidationError as error:
            LOGGER.warning("%s validation stage=%s batch=%s attempt=%d reason=%s",
                           operation, stage, batch, attempt, str(error))
            if str(error) == "truncated_output" and not retry_truncated:
                raise SynthesisTruncatedError(
                    f"{operation} output was truncated stage={stage} batch={batch}."
                ) from error
            if attempt == 2:
                raise DocumentQueryError(
                    f"{operation} validation failed stage={stage} batch={batch} "
                    f"reason={error} after one corrective retry. No completed result published."
                ) from error
            if str(error) == "truncated_output" and truncated_retry_model is not None:
                model = truncated_retry_model
                LOGGER.info("%s corrective retry uses extended output budget", operation)
            messages = prompt.to_messages()
            correction = (
                f"\nCorrective retry: previous response rejected reason={error}. "
                "Return schema-valid JSON only; no inline citations. "
                "Use only exact source IDs supplied in this batch or its cited notes. "
                "If no relevant evidence exists in this batch, use "
                f'status="no_relevant_evidence", findings=[]. Keep the total rendered '
                f"text and citations below {max_chars} characters; prefer fewer, shorter findings."
            )
            if str(error) == "missing_citations":
                correction += " Keep supported findings; remove uncited findings, not all relevant evidence."
            if str(error) == "date_not_in_cited_sources":
                correction += (
                    " A date is missing from its cited passages. Keep the stated dates, but cite "
                    "the passage labels containing those exact dates, not another topic's labels."
                )
            if str(error) == "missing_support_references":
                correction += " Select supplied integer source labels in support; do not copy or generate quotes."
            if str(error) == "unscoped_information_absence":
                correction += (
                    " Use direct stated values; scope any absence explicitly to its exact field "
                    "or clause. If another supplied passage gives the requested value, answer "
                    "with that value and do not infer that it is missing from a different field. "
                    "Return no_relevant_evidence only if supplied passages cannot answer."
                )
            if not allow_no_relevant:
                correction += " Relevant input notes exist; preserve their findings, not a no-relevant result."
            # Append only fixed diagnostics, never the rejected model output.
            messages[0] = SystemMessage(content=messages[0].text + correction)
            prompt = ChatPromptValue(messages=messages)
            if len(prompt.to_string().encode()) > max_prompt_bytes:
                raise DocumentQueryError(
                    f"{operation} corrective retry exceeds prompt budget. Narrow the request.") from error
    raise AssertionError("Unreachable synthesis retry state")
