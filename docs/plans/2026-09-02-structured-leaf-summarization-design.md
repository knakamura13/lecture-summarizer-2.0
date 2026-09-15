# Structured Leaf Summarization Design

## Scope

Issue #5 turns each source segment into validated structured data instead of an opaque paragraph. It consumes the `SourceSegment` records produced by issue #4 and produces one validated leaf record per segment, in source order.

It is deliberately library-only. No acceptance criterion requires a command-line change, and issue #4 left the legacy flat workflow intact so that each PR stays independently verifiable. The CLI keeps running the legacy path until a later issue switches it over.

Three boundaries matter, because each is owned by a different issue:

- **Issue #4 owns segmentation.** `SourceSegment` already exists, with disjoint core ranges, a context range, token counts, and a stable `S000001`-style identifier. Issue #5 consumes it unchanged and adds only leaf-side models.
- **Issue #6 owns budget arithmetic.** Usable input capacity, prompt and structured-output overhead, output reservations, and the safety margin are all its concern. Issue #5 never decides how many tokens a leaf request may spend; it accepts a segment that is already budget-compliant.
- **Issues #7 and #8 own the hierarchy.** Merging, branching factor, and provenance propagation through merge levels are later work. Issue #5's provenance duty is narrower: a leaf's evidence must resolve to a known segment.

## Considered approaches

### Prompt and parse only

Ask for JSON in the instructions and parse whatever comes back. This works against any model or tag and needs no provider change, but it turns validation into a large, fragile parser that must tolerate prose preambles, fenced code blocks, and trailing commentary. It also weakens the determinism criterion, because the shape of the response is unconstrained.

### Schema-constrained decoding through a pydantic type

Use the OpenAI SDK's `responses.parse` with a `text_format` model. This is the least code on the OpenAI side, but it leaks a pydantic class through the provider protocol and has no Ollama equivalent — the Ollama client accepts a JSON Schema mapping, not a model class. It would make one provider's SDK shape the contract for all providers.

### A provider-neutral schema on the request, plus a tolerant parser

Add an optional response schema to `GenerationRequest` as a plain JSON Schema mapping. Each adapter maps it to its own native mechanism. Parsing and validation stay above the provider boundary.

This is the selected approach. A JSON Schema mapping is the one representation both installed clients accept natively: the OpenAI Responses API takes `text={"format": {"type": "json_schema", ...}}`, and the Ollama chat client takes `format=<schema>`. It keeps `ModelProvider` a single-method protocol, keeps orchestration free of provider types, and leaves room for a future adapter that ignores the schema entirely.

The parser must stay tolerant even so. Ollama's `format` constrains decoding on a best-effort basis, not a guarantee, and small local tags still emit preambles. The parser locates the outermost JSON object, tolerating a surrounding code fence, and then validates strictly. Constrained decoding reduces slop; it does not license trusting it.

## Domain model

Three records, nested, mirroring the names the epic suggests:

- **`EvidenceItem`** — a `segment_id`, and optionally a verbatim `quote` drawn from that segment.
- **`ContentUnit`** — one claim or point: its text, a kind, its supporting `EvidenceItem` values, and any qualification or uncertainty attached to it.
- **`SummaryNode`** — a concise local summary, its `ContentUnit` values, relevant entities or subjects, material qualifications, contradictions, salient quotations, the provenance set it draws on, and a `level`.

`SummaryNode` carries `level` from the start, set to `0` for leaves. Issue #7 needs a record that can be the input to another merge level, and naming that type now avoids renaming the hierarchy's central record one issue later.

Contradictions and quotations must be representable without being mandatory. Under OpenAI's strict schema mode every property is required and nullable fields become an explicit `anyOf` with `null`, so "optional" here means a required, possibly-empty array rather than an absent key. The validator accepts both an empty array and `null` for those fields, because the non-strict paths — Ollama, or prompt-and-parse against a model that ignores the schema — will produce both.

### Validation library

The leaf records are pydantic models rather than frozen dataclasses, which departs from the house style used by `SourceDocument`, `SourceSegment`, and the provider records.

The reason is that these records are the only ones in the project built from untrusted model output rather than from local computation. Two things follow: the JSON Schema sent to the provider must be derived from the same definition that validates the response, or the two drift silently; and validation errors need to be structured enough to report which field of which segment's response failed. Pydantic is already a declared dependency, and the Ollama adapter already handles `pydantic.ValidationError` at its boundary.

Every model is frozen, so immutability is preserved. This choice sets a precedent for issues #7 through #9 and is listed under open decisions for that reason.

## Provenance

A leaf's legal reference set is computed by the caller from the segments it was given, never from model output. This is what makes the fourth acceptance criterion enforceable and is also the defence against an injected citation: source text instructing the model to cite `S999999` produces a validation failure, not a dangling reference.

Evidence is restricted to the leaf's own `segment_id`. Issue #4's design states that later stages should attribute evidence to core ranges and treat overlap as context only, so text that arrives through a segment's overlap is available as context but is not attributable. The prompt says so explicitly.

Quotations are stored verbatim and located by matching, not by character offsets supplied by the model — a model asked to count characters will get it wrong, whereas a quote it copied can be checked. The match runs against the segment's **core** range rather than its context range: a segment's text spans its whole context, so matching against that would let a leaf claim a neighbouring segment's core through the overlap window and produce two attributions for one sentence. No offsets are stored; a later stage that wants them can recover them from the core range and the quotation.

A leaf must also record provenance for itself. Checking only that no *unknown* identifier appears would let a leaf with no references at all pass, satisfying the requirement for resolvable evidence vacuously.

### A conflict to resolve first

Both provider adapters currently collapse whitespace in the response with `re.sub(r"\s+", " ", ...)`. For JSON this is mostly harmless, since whitespace between tokens is insignificant and an escaped newline inside a string literal is two characters that the collapse does not touch. It is not harmless for quotations: a quote copied exactly out of a segment containing a newline or a double space comes back single-spaced, and an exact match against the source then fails.

The collapse exists to tidy prose summaries, so the fix is to stop applying it to structured responses: when a request carries a response schema, the adapter returns the payload unmodified. The alternative — making quote matching whitespace-insensitive — hides the problem and quietly weakens what a verbatim quote means.

## Prompt and the injection boundary

Issue #4 committed to treating every byte of source text as inert data, and generation is where that commitment is actually tested. The leaf prompt therefore:

- puts task instructions only in the request's `instructions` slot, and source text only in `input_text`, never interpolating source into the instructions;
- fences the source and states that everything inside the fence is data to be summarized rather than instructions to follow, with explicit precedence if the two conflict;
- assumes the source can forge the fence, so the delimiter is chosen not to collide and the precedence rule does not depend on the fence being unforgeable;
- stays genre-neutral. The legacy prompt in `summarizer/text.py` is tuned for dense technical prose and is asserted verbatim by characterization tests; this is a new prompt, not an edit to that one.

Structural labels such as `boundary_kind` are derived locally by segmentation and never taken from model output, so they cannot be poisoned.

Error messages must not echo source text. The provider tests already assert that source content and credentials stay out of failure messages, and a malformed-response error follows the same rule: it names the segment and the validation failure, not the payload.

## Errors and invariants

Failures use focused application exceptions and never become summary text:

- a response that is not parseable JSON, or that fails schema validation, raises an actionable error naming the segment and the reason;
- an evidence reference outside the legal set fails validation;
- a quotation that cannot be located in its segment fails validation.

The first invalid segment fails the stage. Nothing in issue #5 requires surviving a bad segment, the project's existing rule is that provider failures are never converted into output, and a half-populated hierarchy reaching issue #7 is worse than a clear failure. The stage returns an ordered immutable sequence rather than a mapping, so per-segment outcomes can be added later for issue #11's resume support without changing the success path.

There is no retry on a schema violation. `ProviderResponseError` is deliberately not a transient error, so the existing retry decorator will not re-ask, and a bounded re-ask would be new machinery. Deferring it is a decision, not an oversight.

## Determinism

The testable claim is request-level: the same segment and the same prompt and schema version produce a byte-identical `GenerationRequest`. That is also the precondition for issue #11's cache keys, which are specified in terms of a stage, prompt, and schema version, so the version constant is emitted now even though nothing caches on it yet.

Output determinism is claimed only for a deterministic provider, which is how the acceptance criterion is worded, so a deterministic fake is sufficient to test it. Real-provider reproducibility is a separate matter: the installed OpenAI Responses API exposes no seed parameter, while Ollama supports both a seed and a temperature through its options. Plumbing those through belongs with the later concurrency and evaluation work.

Source order is preserved by construction — segments are consumed in `order` and results are emitted in the same sequence.

## Module layout

- `summarizer/summaries.py` — the records above, their schema, and the version constant. Issues #7 and #8 add merge levels that reuse these records, so they do not live in a leaf-specific module.
- `summarizer/leaf.py` — the prompt, the request builder, the parser and validator, and the leaf stage.

`summarizer/text.py` is left alone. It is explicitly transitional and its constants are pinned by characterization tests.

## Testing

All tests are offline. The autouse fixtures that block outbound sockets and isolate logging apply unchanged, and provider access is faked through the existing `client_factory` seam rather than by intercepting HTTP.

Coverage follows the acceptance criteria: a valid structured response; a malformed one; an incomplete one missing required fields; one carrying contradictions; one carrying uncertainty; one citing an unknown segment; one whose quotation does not appear in the source; and injection-like source text, for which the existing fixture string and the genre fixture corpus are reused.

Two assertions deserve naming because they are easy to omit. Error messages are checked negatively, for the absence of source text and credentials. And the request built for a given segment is asserted byte-identical across runs, which is the determinism criterion stated as something a test can fail.

A mocked suite cannot prove that structured output works against a real provider, and this design does not claim otherwise; that gap is closed by manual verification against both providers, recorded on the pull request.

## Delivery boundary

Issue #5 delivers the leaf records, their schema, the provider-neutral schema field and its two adapter mappings, the leaf prompt, the parser and validator, the leaf stage, and their tests. The CLI, the legacy workflow, and the budget arithmetic are untouched.

It depends on issue #4, which is open in PR #14. The `SourceSegment` field list is settled, but until #4 merges this design is written against an unmerged API.

## Decisions taken

Implemented as designed, with the selections below confirmed rather than revisited:

1. **Pydantic for the leaf records.** They are the only records built from untrusted model output, and generating the provider's schema from the same definition that validates the response is what stops the two from drifting. Two settings carry that: forbidding extra fields emits `additionalProperties: false`, and declaring no field defaults marks every property required — so the generated schema is already an OpenAI strict schema and no private SDK helper is needed.
2. **`GenerationRequest` extended**, with both new fields appended and defaulted.
3. **One `SummaryNode` with a `level`**, zero for leaves.
4. **The whitespace collapse is now conditional** rather than removed. A schema-carrying response is returned unmodified; a prose response is still collapsed, so the legacy workflow is untouched.
5. **Quotation limits** were left to the prompt and to post-validation rather than encoded as schema constraints; a cap remains a product decision. Resolved in issue #36: a per-quote length cap (`MAX_QUOTE_CHARS`) and a per-node count cap on the `quotations` field (`MAX_QUOTATIONS_PER_NODE`), both in `summarizer/summaries.py`, enforced by pydantic validators on `SummaryNode` and `EvidenceItem` and stated in both the leaf and merge prompts. Post-validation rather than JSON Schema keywords, because OpenAI's strict mode does not support `maxLength`/`maxItems` — the same reason the existing blank-text and negative-level checks are validators rather than schema constraints. The merge prompt states the same limits so a merge is less likely to propose an over-limit quote in the first place, rather than relying solely on post-hoc rejection.

Two details emerged during implementation and are worth recording:

- The fence delimiting source text is derived per segment from the source digest and the segment identifier. That keeps a request byte-identical across runs while making the marker unguessable from source text alone. Precedence is still stated in the instructions, because a fence that merely looks unguessable is not a boundary on its own.
- Where a segment carries overlap, inner markers identify the part that belongs to it, and the instructions declare the surrounding context unattributable. Without that, an instruction to summarize only the core would not be actionable, because a segment's text spans its whole context range.

## Open decisions

These change the shape of the code and set precedent beyond this issue:

1. **Pydantic or frozen dataclasses for the leaf records.** Selected above as pydantic, for schema derivation and untrusted-input validation, at the cost of consistency with every other record in the project. It sets the pattern for the whole hierarchy.
2. **Extending `GenerationRequest`.** Selected above, which touches the provider contract established by issue #3 and both adapters. Keeping structured output entirely above the provider boundary would leave issue #5 self-contained but weaken the parsing and determinism criteria. Any new field is appended with a default, since both request and result records are constructed positionally in the existing tests.
3. **One `SummaryNode` for leaves and merges, or a leaf-only record that issue #7 generalizes.** Selected above as one record with a `level`.
4. **Quotation limits.** Resolved in issue #36 - see decision 5 above.
5. **Whether the whitespace collapse changes now.** The fix above alters observable behaviour of both adapters for schema-carrying requests only. If that is unwelcome in this issue, verbatim quotations should be dropped from the leaf record rather than validated unreliably.
