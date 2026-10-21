# Code review

**Review date:** 2026-10-21  
**Scope:** Current working tree; the repository has no commits to use as a diff baseline.  
**Validation before correction:** `.venv/bin/python -m pytest -q` — 814 passed, 50 skipped.  
**Correction validation:** Targeted document tests — 194 passed, 3 skipped.

## Findings

| # | Severity | Location | Issue | Corrective action | Status |
|---|---|---|---|---|---|
| 1 | Medium | [`document_library.py`](./document_library.py#L468), [`document_search.py`](./document_search.py#L405), [`document_query.py`](./document_query.py#L207) | Empty document selections were treated as unrestricted in catalogue snapshots and query-session scope setup, allowing an empty explicit scope to expand to the full catalogue. | `None` is now the only unrestricted-scope value. Empty snapshot/search scopes return no documents or passages; indexing and query-session selection reject an empty scope. Regression tests cover snapshots, search, index, and query sessions. | Resolved |

## Summary

No other actionable correctness findings were confirmed during this review.
