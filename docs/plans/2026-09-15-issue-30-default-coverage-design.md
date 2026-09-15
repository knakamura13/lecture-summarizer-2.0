# Issue #30 Default-Configuration Coverage Design

## Goal

Add offline, deterministic tests that exercise default pipeline capacity,
overlap sizing, finalization runtime resolution, and audit-link failures that
previous coverage omitted.

## Design

Tests remain in the subsystem that owns the public behavior. `test_pipeline.py`
gets realistic default-tokenizer hierarchy and overlap-capacity cases.
`test_finalization.py` covers enabled verification without an injected runtime.
Audit-link invariant failures live with the existing audit tests.

The realistic hierarchy provider will return every source identifier found in a
merge request, including grounding-preserved references, instead of returning a
single arbitrary identifier. The real tokenizer case skips only if its cached
vocabulary is unavailable. It makes no network calls.

No production code is planned. A new failing test would instead become evidence
for a separate behavior defect rather than being hidden by test scaffolding.
