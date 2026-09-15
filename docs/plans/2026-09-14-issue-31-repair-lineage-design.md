# Issue #31: committed repair lineage

## Decision

When a verification result returns repaired text, `repairs` records the
locally validated repair operations on that returned text's committed lineage.
It is not a history of attempted repairs and is not a byte-for-byte final-text
diff.

Consequently:

- A repair made on a successful path remains in `repairs`, including when a
  later pass reworks the same span. The audit intentionally retains no repair
  prose or replacement hash, so it cannot and need not model a replacement
  diff.
- A repair made in a branch whose text is discarded must be omitted. In
  particular, a post-repair verifier failure that returns the pre-repair draft
  has no repair events.
- If a recursive continuation fails and the caller returns its already repaired
  draft, only the caller's events remain. Events produced solely by the failed
  continuation are discarded with that continuation's text.
- If a recursive continuation succeeds, its events are appended to the caller's
  events because both repair sets contribute to the returned text's lineage.

This is the narrowest interpretation that fixes the Issue #31 mismatch: the
result and audit never describe a repair from a discarded text branch, while
the audit stays a privacy-preserving record of applied repair operations rather
than a second copy of summary prose.

## Data flow

Each `_verify_and_repair` invocation owns one candidate transaction:

1. Verify its input draft, locally validate and apply its repair proposals, and
   re-verify the repaired candidate.
2. If that candidate or a successful recursive continuation is returned, retain
   the events that constructed it.
3. If an error, exhausted contradiction, or failed continuation causes the
   function to return an earlier draft, return only the events that constructed
   that earlier draft. Never carry events from the abandoned candidate.

`AuditArtifact` already projects `VerificationResult.repairs` directly, so the
result tuple is the single contract boundary. No audit schema or public model
expansion is required.

## Failure handling

Verification remains fail-closed. A failed re-verification returns the safe
earlier draft and preserves redacted pass and generation diagnostics, but omits
repair events belonging only to the rejected candidate. This keeps failure
diagnostics visible without presenting attempted repairs as applied output.

## Verification matrix

- Pass 1 repairs claim A and a later pass repairs independent claim B; the
  result contains both fixes and records both events.
- A post-repair provider failure returns the original draft with no repairs;
  the audit projects the same empty repair list.
- A recursive continuation failure returns the outer repaired draft and only
  the outer repair events.
- The multi-pass regression fails against `174f54b`, where recursion starts
  again from the original draft.
