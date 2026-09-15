# Issue #35 Verification Diagnostics and Escalation Batching Design

## Goal

Make closed verification-reduction diagnostics visible in audit artifacts and
bound contradiction escalation provider calls by token-sized batches rather
than one call per omitted segment.

## Decision

Keep `VerificationResult.diagnostic_codes` as the canonical producer-side
field. During audit projection, combine it with explicit `warning_codes` in
stable order with duplicates removed. This preserves the audit schema's
closed-warning-code contract without adding a second producer-side routing
responsibility.

For each contradicted claim with omitted legal provenance, represent each
omitted segment as a work item. Use `pack_work_items` with a renderer that
combines a packed batch's passages into one `EvidenceBundle`, then issue one
classification request for that combined bundle. Parse every response against
all passages included in that request. The final escalation bundle continues
to record all examined passages and complete retrieval.

## Alternatives Considered

1. Populate `warning_codes` at every `VerificationResult` construction site.
   This duplicates routing logic and is easy to miss on terminal paths.
2. Remove the audit warning-code promise and unused field. This loses
   diagnostics already emitted by reduction and would make the audit less
   useful.

## Safety and Compatibility

The audit schema remains unchanged. It stores only closed codes, never prose.
The batching implementation reuses the existing request-capacity contract and
terminal failure behavior. A batch containing one passage retains the current
semantic result.

## Verification

Focused tests will prove diagnostic projection and that three fitting omitted
segments result in one escalation provider call. Existing verification and
audit suites, followed by the complete offline suite, will guard compatibility.
