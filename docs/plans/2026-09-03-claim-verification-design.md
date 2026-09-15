# Claim Verification and Bounded Repair Design

## Scope

Issue #10 adds an optional safeguard after final editorial writing and before
citations and audit materialization. It checks independently reviewable claims
against bounded original-source evidence, repairs problematic spans at most a
configured number of times, and exposes every limitation in audit data.

Verification is disabled by default. The default reader-facing result remains
plain summary text. This feature is a library boundary until issue #12 replaces
the transitional legacy CLI workflow.

## Delivered boundary

Tasks 1–6 are implemented. `PipelineConfig` carries a disabled-by-default
`VerificationConfig`; `finalize_summary` invokes verification immediately after
editorial writing and before citation rendering or successful audit
materialization. The pipeline constructs the default runtime from the original
summarization provider, model, counter, timeout, and selected context window.
A caller may replace all of those dependencies with one complete
`VerificationRuntime`. A model-only override is deliberately absent.

Verification is a best-effort, evidence-scoped safeguard. A `supported` verdict
means that the selected source evidence supported a validated claim under this
bounded process; it is not a statement of factual perfection. Conversely,
`insufficiently_supported` records a bounded-retrieval limitation rather than a
finding that the source lacks support.

## Runtime decision

Verification uses the summarization provider, model, counter, and timeout by
default. A different verifier is not represented by only a model string because
it also needs a compatible provider, counter, context window, endpoint, and
credentials. Verification APIs therefore accept their runtime dependencies
explicitly, allowing a complete verifier runtime to be injected later without
changing the verification domain.

This avoids premature configuration while preserving a clean high-assurance
path for model diversity. A second model may reduce correlated mistakes, but its
judgments still remain fallible assessments rather than proof. A runtime whose
reserve and safety margin leave no usable request capacity becomes a terminal
verification failure; when an audit path is configured, that failure is
recorded before finalization withholds reader-facing output.

## Pipeline

The selected hybrid design avoids both one oversized full-draft request and one
provider call per claim:

1. Write the normal editorial draft.
2. Split it locally into immutable sentence spans with pass-scoped identifiers
   and character offsets.
3. Ask the model for exact draft anchors that represent atomic claims. Anchors
   may overlap when claims share context; classification sees each anchor in its
   full sentence span. Local code assigns identifiers and designates an exact
   full-span anchor as the fallback, adding one only when absent. Decomposition
   therefore cannot exempt or omit a span, and only classification may return
   `not_meaningfully_verifiable`.
4. Rank every root-provenance segment through a precomputed lexical index using
   deterministic overlap scoring and stable source-order tie breaking.
5. Pack complete evidence passages under configured token budgets and record
   every examined and omitted candidate plus whether retrieval was complete.
6. Pack complete decomposition, classification, and repair work items into
   independently budget-bounded requests.
7. Validate one verdict for every claim: `supported`, `contradicted`,
   `insufficiently_supported`, or `not_meaningfully_verifiable`.
8. If a material claim is contradicted, escalate deterministically through its
   omitted legal evidence and reduce all batch findings conservatively. Perform
   one targeted span-level repair by default only after all legal provenance has
   been examined and the containing whole-span fallback is also contradicted.
   Repair may
   qualify, replace, or remove a span but may not alter a span whose hash no
   longer matches or discard a supported sibling claim.
9. Re-split, re-decompose, and re-verify the entire repaired draft. The repair
   call never certifies its own output. Each further repair pass continues from
   the most recently repaired draft, never the original input, so consecutive
   passes accumulate fixes instead of re-deriving the same one.
10. Stop at the configured limit. Any remaining material contradiction fails
    closed. Insufficient evidence remains an audit-visible limitation.

Budget-derived batching reduces duplicated prompt overhead while keeping every
request bounded for extremely long artifacts.

## Domain model

`VerificationConfig` records whether verification is enabled, separate evidence
and request token budgets, output reserve, safety margin, and
`max_repair_passes`, which defaults to one and may be zero. It rejects
nonpositive budgets and negative limits.

`VerificationRuntime` contains a provider, counter, model, timeout, context
window. By default these are the summarization runtime's dependencies. Injection
replaces the complete runtime, never only its model. Every request must satisfy
`measured_input + output_reserve + safety_margin <= context_window`; the request
budget is an additional lower cap on measured input.

`DraftSpan` contains a pass-scoped identifier such as `V01S000001`, source
order, draft character range, text, and a content hash. Offsets are computed
locally and never accepted from model output.

`Claim` contains a pass-scoped identifier such as `V01C000001`, its draft span
identifier, ordinal within that span, an exact draft anchor, and whether it is
the whole-span fallback. Anchors may overlap but must be exact substrings; no
model-authored paraphrase becomes a claim. Local code treats every claim as
material; model output cannot choose materiality, identifiers, offsets, or
whether text bypasses classification.

`EvidenceSelection` records selected, examined, and omitted segment identifiers,
token cost, a versioned retrieval method, and whether retrieval was complete.
Original passage text is ephemeral.

`BatchFinding` records one classification batch's verdict and selected evidence
identifiers. `ClaimAssessment` records the single reduced verdict, its batch
findings, verifier identity, prompt version, and pass index. Supported and
contradicted findings require selected evidence. Evidence identifiers must be
known and selected. Exact quotes may be used transiently for substring checks
but are not persisted.

`RepairEvent` records the target span identifier, original span hash, triggering
claim identifiers, and action. Repaired prose remains transient.

`VerificationResult` contains final text, assessments by pass, repair events,
provider generations, warnings, limitations, and exhaustion state.

## Evidence selection

The merge-specific `select_source_passages` policy is not reused because its
ranking favors ambiguity and qualifications across merge children. Claim
verification instead uses root provenance as the legal evidence set and
resolves those identifiers to canonical segment cores. One precomputed lexical
index ranks all legal segments by normalized term overlap, then stable source
order. No unrecorded claim-to-source lineage is assumed.

Only complete segment cores are eligible. The selector never accepts passage
text or identifiers from the model. It deduplicates candidates, applies a
per-claim evidence budget, records examined and omitted identifiers, and fails
clearly if no complete passage can fit. Retrieval is recall-bounded, so an
insufficient verdict means only that selected evidence did not establish the
claim. An initial contradiction triggers bounded classification batches over
the omitted legal segments. It is eligible for automatic repair only after all
legal provenance has been examined; otherwise it remains an explicit unresolved
limitation.

Batch findings reduce deterministically. Any supported and contradicted mixture
becomes `insufficiently_supported` with a closed conflicting-evidence limitation
and never repairs. Final `contradicted` requires at least one contradicted
finding, no supported finding, complete examination, and no inconsistent
meaningfulness finding. Final `supported` requires support and no contradiction.
All other evidence-dependent combinations reduce to
`insufficiently_supported`. `not_meaningfully_verifiable` is final only when
every finding agrees. Batch findings remain separately auditable.

## Structured provider calls

Decomposition, classification, and repair each use a strict Pydantic schema and
a deterministic, source-specific fence. The counter measures the full input,
including instructions, schema, and data. Each request must leave the configured
output reserve and safety margin inside the runtime context window and remain
under the lower configured input cap. Each phase supports deterministic
batching, and an oversized indivisible work item fails before any provider call.
Source, draft, and claim content are delimited as inert data. Responses reject
extra fields, duplicate or missing identifiers, unknown spans or segments,
invalid verdict/evidence combinations, and non-exact anchors. Every span also
receives a locally created whole-span fallback claim.

Malformed verification output fails closed when verification is enabled. Normal
provider transport retries remain the provider layer's responsibility. There is
no unbounded format-correction or repair loop.

## Repair policy

Only contradicted material claims whose containing whole-span fallback is also
contradicted trigger repair by default. A mixed atomic/fallback result becomes a
closed limitation and never repairs. Insufficiently supported claims are
reported rather than silently rewritten because bounded retrieval may have
missed support. Non-verifiable statements are not failures.

Repair operates on whole local sentence spans. The request includes every claim
from the target span. Local validation requires each stored exact anchor of a
supported sibling to remain in the replacement and rejects stale hashes. After
a repair, the entire draft is re-split, re-decomposed, and re-verified.

The repaired span also has a post-repair signal-preservation invariant. Every
previously supported sibling remains traceably represented, and every new or
materially changed checkable claim must be supported or non-verifiable in the
new pass. If a sibling disappears or a new claim is contradicted or
insufficiently supported, the repair is rejected and the prior draft is
restored. The repair call never certifies these conditions itself.

## Audit schema

Verification advances every newly written artifact to `audit/2`, including
runs where verification is disabled. Disabled operation preserves summary text,
provider call count, and citation behavior but is represented explicitly in the
new schema. The audit verification record contains:

- whether verification ran;
- verifier provider, model, and prompt versions;
- pass count and exhaustion state;
- claim identifiers, span hashes, verdicts, and evidence segment identifiers;
- selected and omitted evidence identifiers plus retrieval method;
- repair actions and triggering claim identifiers;
- closed warning, failure, and limitation codes with identifier or numeric
  metadata only;
- usage metadata for decomposition, classification, and repair generations.

It stores no claim text, draft text, replacement text, source passages, exact
quotes, free-form warning text, prompts, requests, endpoints, credentials, or
paths. Runtime configuration is serialized through an explicit allowlist of
provider name, model name, prompt versions, limits, counts, and usage. The
verification records are nested by pass, and claim and span identifiers resolve
only within their owning pass. Serialization remains canonical, validated,
redacted, and atomic.

The audit states that model verdicts are assessments, exact-quote validation
establishes traceability rather than truth, and evidence retrieval was bounded.

## Errors and completion semantics

Configuration, parsing, evidence resolution, and repair application use focused
verification failures with bounded, redacted codes. `verify_and_repair` returns
a terminal failed result with all metadata instead of raising before it can be
recorded. When an audit path is configured, finalization validates and writes
the failure audit atomically, then withholds the summary and raises a focused
error. Citations are not resolved or rendered on that terminal path. Malformed
output, capacity failures, and unresolved material contradictions therefore
never return a summary labeled as verified. Pass exhaustion is explicit and
finite.

`repairs` records the returned text's committed lineage, not a history of
attempted repairs. A pass that proposes a repair and fails its own
re-verification (or errors outright) reverts to the draft that was current
before that attempt, with that attempt's events excluded. A pass whose own
repair *is* re-verified successfully, but whose deeper continuation
subsequently fails closed, keeps its own repaired text and its own events;
only the events belonging to the discarded continuation are dropped. Either
way, a repair is reported only when it is present in the returned text.

Insufficiently supported claims may return the unchanged or repaired summary
only with audit-visible warnings and limitations. This preserves signal without
turning incomplete retrieval into an unsupported deletion.

## Testing

Offline scripted providers and deterministic counters cover disabled zero-call
behavior, sentence spans, overlapping atomic anchors, whole-span fallback
claims, compound claims, every verdict, mixed evidence, stable ranking,
retrieval escalation, and batching for decomposition, classification, and
repair. They cover exact fits, oversized indivisible items, malformed output,
invented identifiers, quote mismatch, supported-sibling deletion, unsupported
new repair claims, stale-span refusal, failed repair, pass exhaustion,
full-draft reverification, pass-scoped identifier collisions,
prompt-injection-like source data, audit link resolution, allowlisted config,
closed diagnostic codes, failure-audit writing for malformed output and pass
exhaustion, deterministic serialization, usage accounting, and source and
secret redaction.

## Alternatives rejected

One full-draft verification call minimizes calls but grows with all evidence,
weakens claim isolation, and hides local failures. One call per claim maximizes
isolation but repeats prompt overhead and scales poorly. A separately configured
verifier model now would create an incomplete seam without its required provider
and token-budget runtime. The hybrid bounded-batch design provides the strongest
current balance of faithfulness, cost, and future extensibility.
