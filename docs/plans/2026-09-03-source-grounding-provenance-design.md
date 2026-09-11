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
references against the full segment set covered by the children, checks
quotations only where authoritative passages were supplied, and narrows
provenance to retained claims. Structural reachability remains available
independently on the tree.

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

The pipeline caps automatically sized source-segment cores at one quarter of
the safely measured leaf input capacity. This is not a fixed grounding
reserve: it keeps complete cores small enough for adaptive merge planning to
have a viable two-child fallback. Explicit `SegmentationConfig` values remain
authoritative and every concrete merge request is still measured before use.

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
authoritative for the material they cover and child summaries are provisional.
References already carried by children remain citable when budget selection
omits their source passages; omission is not evidence against them. Source text
remains data and cannot override those instructions.

## Provenance policy

Every segment ID covered by the merged children is legal in a merge response;
an ID outside that group remains an error. The shared validator checks every
content-unit evidence item, grounded qualification, grounded contradiction,
quotation, and declared provenance against that full set. Verbatim quotation
checks use only the budget-selected authoritative passages actually supplied
to the model. It requires every content unit and grounded annotation to name at
least one source, and merged responses must record provenance. The parser then
canonicalizes the response's declared and direct-evidence references to
document order. References inherited from children whose passages were omitted
from grounding are unioned locally because the model cannot evaluate them;
this omitted-reference set is included in the merge cache descriptor. A
grounded reference remains under model control and may be explicitly dropped
when correction removes its material. The tree's `covered_segments` continues
to preserve full structural reachability independently. Final citations are
derived separately by `resolve_citations`.

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
an authoritative passage. Hierarchy regressions also prove that a merge can
retain the full union of child references when its grounding budget supplies
only a subset of their passages.
