# Source Grounding and Provenance Propagation Design

## Scope

Issue #8 makes recursive merges reconsult authoritative source text and retain
traceable, machine-readable provenance. It follows issue #7's hierarchy rather
than changing its tree shape, CLI integration, cache policy, or final output.

## Existing seam

`build_hierarchy` receives `attributable`, an ordered mapping from a
source-segment ID to its citable core text. That mapping is controlled by the
application, not model output, and is therefore the only valid source for
grounding passages. A merge request serializes generated children separately
from selected authoritative source passages; its parser validates model
references against those selected passages and narrows provenance to retained
claims. Structural reachability remains available independently on the tree.

## Considered approaches

1. Send every covered source segment with every merge. This is maximally
   direct but can consume the entire context window and breaks multi-level
   progress.
2. Perform a separate retrieval or embedding stage. It could improve semantic
   ranking, but adds a provider and persistence boundary that the issue does
   not require.
3. Select deterministic passages from existing structured evidence and child
   coverage, under the request's actual remaining token budget. This reuses
   the established model-neutral token boundary and keeps offline tests fully
   deterministic.

The third approach is selected.

## Data flow

By default, the hierarchy does not reserve a fixed fraction or fixed token
count for source grounding before calculating fanout. It measures merge
overhead, sizes a candidate fanout against the remaining usable capacity, and
then prepares the concrete group. The selector measures the complete merge
request, including generated child summaries, fences, source passages, and
schema. If the selected passages do not fit, the hierarchy retries with a
narrower fanout before failing. A caller that supplies an explicit
`GroundingPolicy` instead gets the fixed `max_tokens` reserve represented by
that policy.

Within a concrete group, the selector considers candidate IDs in deterministic
priority order:

1. contradiction evidence;
2. qualification and uncertain-content evidence;
3. quotation evidence;
4. other content-unit evidence; and
5. declared provenance as a deterministic fallback.

It packs complete core passages only. A request fails clearly if the available
budget cannot hold evidence required by a contradiction, qualification, or
uncertain claim; it may omit only low-priority fallback IDs. This is
conservative: a later issue may replace ranking, but it cannot permit
generated text to supply its own source or silently turn an ambiguous claim
into an ungrounded one.

The merge request has separate generated-summary and authoritative-source
blocks, each individually fenced. Its instructions say source passages are
authoritative, child summaries are provisional, and an output claim or
reference must be supported by the supplied passages. Source text remains data
and cannot override those instructions.

## Provenance policy

Only selected passage IDs are legal in a merge response. The shared validator
checks every content-unit evidence item, grounded qualification,
grounded contradiction, and quotation against those passages. It requires every
content unit and grounded annotation to name at least one source. Merged
responses must record provenance. The parser then canonicalizes every declared
and direct-evidence reference to the selected candidate order and stores that
narrowed sequence; it no longer replaces it with the whole child union. The
tree's `covered_segments` continues to preserve the full structural
reachability independently, in document order. Final citations are derived
separately by `resolve_citations`, which sorts the cited IDs by source segment
order before they are written to output or audit metadata.

This makes the two notions explicit:

- `covered_segments`: every original segment structurally represented by a
  tree node;
- `SummaryNode.provenance`: selected original passages that support claims
  retained by that node.

`GroundedAnnotation` replaces the existing bare strings for qualifications and
contradictions. This small schema extension gives every retained standalone
qualification or conflict explicit evidence without adding a separate assertion
framework. A merge cannot silently resolve or restate them without being shown
the original supporting material.

## Error handling and determinism

Candidate collection, selection, source-block serialization, and stored
provenance are deterministic: candidate priority follows category, child, and
evidence order, while structural coverage remains in document order. The
selector never uses model output to retrieve text. An unknown reference, a
quotation absent from its cited passage, empty merge provenance, an empty
grounding selection, or a passage that cannot fit is a budget or validation
error rather than partial output. Citation projection is the separate boundary
that restores source order for rendered citations.

## Tests

Offline tests will pin passage priority, whole-passage budget accounting,
separation/fencing, invalid references, contradictory or
ambiguous child grounding, and a misleading child summary corrected against
an authoritative passage. Existing hierarchy tests will prove full tree
coverage remains available while root `SummaryNode.provenance` narrows.
