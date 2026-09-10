# Recursive Text Summarizer

This project is being rebuilt as a generalized, source-grounded hierarchical summarization pipeline for extremely long artifacts. The current application provides a provider-neutral foundation while preserving the legacy flat, sentence-chunked workflow.

Canonical ingestion, token-aware segmentation, structured leaf summarization, token-budget arithmetic, whole-document direct summarization, and multi-level hierarchical merging are now available as library components in `summarizer.ingestion`, `summarizer.tokenization`, `summarizer.segmentation`, `summarizer.summaries`, `summarizer.leaf`, `summarizer.budget`, `summarizer.direct`, `summarizer.merge`, and `summarizer.hierarchy`. The command-line workflow still runs the legacy flat, sentence-chunked path and does not consume them yet.

Source grounding, claim verification, audit/2 and audit/3 with reliability
metadata, opt-in caching with resumable execution, bounded concurrency, and a
recoverable final-output publication protocol are available at the library
boundary. The command line still uses the transitional legacy workflow; its
migration to the hierarchical pipeline is separate work.

## Requirements

- Python 3.10 or newer
- An OpenAI API key for hosted generation, or a running Ollama service for
  local generation

Create a virtual environment and install the dependencies:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Set the credential through the environment. The application never accepts credentials through command-line configuration:

```sh
export OPENAI_API_KEY="your-api-key"
```

For local generation, install [Ollama](https://ollama.com/download), start its
service, and pull a model. A local Ollama service does not require an API key:

```sh
ollama serve
```

The Ollama desktop application normally starts the service itself. Run
`ollama serve` in a separate terminal when operating Ollama without the desktop
application. After the service is available, pull the model you want to use:

```sh
ollama pull gemma3:4b
```

## Usage

The no-argument workflow reads `input.txt` and writes `output.txt` in the current directory:

```sh
python main.py
```

Paths and foundational settings can be overridden:

```sh
python main.py --input source.txt --output summary.txt
python main.py --model gpt-4o-mini --timeout 180 --max-retries 5
python main.py --chunk-size 1000 --max-chunks 10
```

Select Ollama and any model tag already installed in that service:

```sh
python main.py --provider ollama --model gemma3:4b
python main.py --provider ollama --model qwen3.8
```

The standard Ollama host is `http://localhost:11434`. Override it when the
service is reachable elsewhere:

```sh
python main.py --provider ollama --ollama-host http://ollama.internal:11434 --model gemma3:4b
```

The application uses Ollama's native non-streaming chat API. It does not pull or
manage models automatically, so a missing tag produces an actionable error.

Run the file workflow without constructing or calling a provider:

```sh
python main.py --dry-run
```

Dry-run mode still reads, chunks, normalizes, and writes the source. It is intended for configuration and file-workflow checks, not summary-quality evaluation.

Run `python main.py --help` for the complete option reference.

## Defaults and failures

The default provider is `openai`, and the default model is `gpt-4o-mini`. This intentionally replaces the legacy `gpt-4-1106-preview` mapping with a currently supported, inexpensive model compatible with the OpenAI Responses API. Model selection will be evaluated separately as the hierarchical pipeline develops.

Configuration errors are reported by the argument parser. Missing files, provider failures, and exhausted retries produce an actionable stderr message and a nonzero process exit. Provider errors are never written as summary text, and output is written only after all configured chunks succeed.

## Architecture

`main.py` is an import-safe entry point over `summarizer.cli`. The CLI constructs immutable configuration, the selected OpenAI or native Ollama adapter, a provider-independent retry decorator, and the transitional legacy workflow.

The provider contract returns structured generation metadata, including the resolved model, token usage, finish status, and request ID when available. This data will support later cost, throughput, and compression evaluation without coupling pipeline orchestration to provider response objects.

The current workflow still uses NLTK sentence tokenization, fixed character-size chunks, independent summaries, and newline concatenation. These are compatibility behaviors scheduled for replacement by later issues.

## Ingestion and segmentation

`summarizer.ingestion` reads a file into an immutable `SourceDocument` holding canonical text and a SHA-256 `source_id` derived from it. Canonicalization is narrow: it strips a leading byte order mark, converts CRLF and CR line endings to LF, removes trailing spaces and tabs from each line, and trims outer blank lines. Headings, indentation, list markers, and internal blank lines survive unchanged.

Every character offset produced by segmentation resolves against that canonical text, not against the original bytes, so `document.text[segment.context_start:segment.context_end] == segment.text` always holds. Concatenating the core ranges of all segments in order reproduces the canonical document.

`summarizer.segmentation` splits a document by structure first — headings, then paragraphs and lists, then sentences, then a token-safe character fallback — and packs consecutive units up to the configured token budget. Each segment owns a disjoint *core* range. Overlap is configured in tokens, defaults to zero, and only ever widens the *context* range around a core; it never moves a core boundary, changes a segment identifier, or transfers evidence ownership. When a core already fills the budget, overlap is reduced, to zero if necessary.

Token accounting is injected through the `TokenCounter` protocol, and segmentation never contacts a provider:

- OpenAI models recognized by `tiktoken` get exact counts for the selected encoding. Constructing that counter can download an uncached vocabulary; counting afterwards is local.
- Arbitrary Ollama tags and otherwise unsupported models fall back to a conservative estimator that treats every UTF-8 byte as a token. It deliberately under-packs, and reports `exact` as false. Injecting an exact counter for such a model recovers that capacity.

Boundary searches are verified rather than assumed: a returned boundary is always recounted against the budget. Because byte-pair encoders are not monotonic in text length — adding a character can *lower* a token count — a search is not guaranteed to find the largest boundary that would have fit. Segments may therefore be slightly smaller than the budget allows, which is safe; no segment ever exceeds it.

## Structured leaf summarization

`summarizer.leaf` turns each segment into a validated record rather than a paragraph of prose. A record carries a local summary, the content units it asserts, the evidence supporting each one, entities, qualifications, uncertainty, contradictions, and quotations. Qualifications and contradictions are grounded annotations: each has its own source evidence, rather than being an untraceable string. Every retained content unit and annotation must likewise record at least one supporting segment. The records live in `summarizer.summaries` and are shared with later merge levels, which is why each one carries a `level` — zero for a leaf.

Evidence must resolve to a segment identifier the caller supplied. The legal set is built from the segments passed in and never from the model's response, so a citation that arrives because the source text asked for one fails validation instead of entering the hierarchy as a dangling reference. A leaf must record provenance for itself, and it may cite nothing else: text that reached it through overlap is available as context to interpret the segment, but is not attributable, and a quotation drawn only from that context is rejected.

Quotations must occur character for character in the part of the segment the leaf owns — its core range, not its whole context range. Character offsets are not stored; a quotation is verified by matching, and later stages that want offsets can recover them from the core range and the quotation itself.

When a provider can constrain decoding, the request carries a JSON Schema — a strict `json_schema` text format for OpenAI, and the native `format` argument for Ollama. Parsing stays defensive regardless, because constraining decoding is best effort on the Ollama side rather than a guarantee, so a response may still arrive fenced or behind a preamble. A structured response is *not* whitespace-collapsed at the provider boundary; a prose response still is, which is what keeps verbatim quotations locatable in their source.

Source text is placed only in the request's input slot, never in its instructions, and the instructions state that fenced content is data rather than instructions. The stage fails on the first segment whose response cannot be validated, naming the segment and the failing field without echoing the payload or the source.

The command line does not consume leaf records yet.

## Strategy selection and token budgets

`summarizer.budget` decides how a document should be executed before any provider is called:

```
usable input = context window − prompt and schema overhead − reserved output − safety margin
```

Every term is measured or configured, not guessed. Overhead is measured by building a real request and counting it, because a hard-coded figure would rot the moment the prompt or the record changed — it is 887 tokens for `gpt-4o-mini`, of which the schema alone is 610, and it rises to 1,007 when overlap is configured. The safety margin is the larger of a fixed floor and a fraction of the window, so a small window keeps a usable floor while a very large one is not charged thousands of tokens for nothing.

A configuration that leaves no usable capacity raises an error naming every term, rather than returning a meaningless number. That case is reachable rather than theoretical: the conservative estimator charges roughly four times a real tokenizer on this project's own prompt text, which exhausts a small window on its own.

`--strategy auto` selects direct only when the document provably fits, the context window is *known* rather than assumed, and any configured direct cap is respected; otherwise it selects hierarchical. `--strategy direct` fails before a provider call when the document does not fit, reporting the same arithmetic the decision used. `--strategy hierarchical` always splits.

Context windows come from a hand-maintained table, because there is no offline source of truth and, for OpenAI, no online one either — the client exposes no window, and `tiktoken` maps model names to encodings rather than to sizes. An unrecognized model therefore resolves to an assumed window that is reported as assumed, and `auto` routes it to hierarchical rather than gambling a direct request on a guess. Pass `--context-window` to state a window explicitly and recover the direct path.

Two provider behaviours are worth knowing, because they differ in a way that matters. OpenAI rejects an oversized request outright. Ollama silently truncates the prompt and returns a plausible answer, so on the local path the budget arithmetic is the only thing standing between a too-large document and a confidently ungrounded summary.

The reserved output size is accounted for when sizing a request but not yet enforced on the provider, and the command line accepts strategy configuration without yet running the new pipeline — it still executes the legacy workflow.

## Hierarchical merging

`summarizer.hierarchy` reduces ordered leaf summaries to a single root, through as many merge levels as the budget requires. `summarizer.merge` owns the prompt that combines several children into one node.

Group sizes are measured rather than fixed. Each child is serialized, its delimiters are added, and the number that fits a merge request is derived from the largest child in that level — sized from the largest rather than the average, so a group is never assembled that only fits on average. Groups are then *balanced* instead of greedily packed, because a ragged final group would compress the end of a document less than the beginning.

What makes the recursion terminate is that a merge level is never narrower than a pair: a capacity that cannot hold two children fails with the arithmetic rather than being rounded up to two, so the node count always falls. The node-count reduction is additionally asserted as a defensive invariant. The merge request's own instructions, schema, and delimiters are subtracted from the budget before children are measured, because the budget calculator measures a *leaf* request and passing that figure through would under-reserve. A fixed, token-measured part of each merge budget is also reserved for original source passages before fanout is calculated; the fully assembled request is checked against the usable budget before it reaches a provider.

On the default local configuration this means hierarchical execution is refused outright: with an assumed context window and the conservative estimator, a merge request's own overhead exceeds the usable capacity, so no children fit. Supplying `--context-window`, or an exact token counter, is what makes it possible.

**Generated summaries and authoritative source passages are deliberately separate.** Child provenance is excluded from serialized merge children, because an ever-growing union would eventually prevent a pair of children from fitting. Instead, each merge deterministically selects complete original source cores from child evidence, prioritizing contradictions, qualifications, uncertain claims, quotations, and then ordinary claims. The source passages occupy their own fenced block and are the authority when they conflict with a generated child summary. Source text is still data, never instructions.

The tree keeps two related facts. `TreeNode.covered_segments` is the full ordered structural coverage of a branch, so the root remains traceable through its children to every original segment. `SummaryNode.provenance` is narrower: it records only the legal source references retained by that node's generated assertions, canonicalized to source order. A merge cannot introduce a reference outside the source passages it received, and every quoted string is checked against its cited core. These checks make grounding a best-effort safeguard, not a claim that a model-produced summary is factually perfect.

The merge prompt requires deduplication that keeps every supporting reference, preserves qualifications and uncertainty, and records disagreements rather than reconciling them. It also states that the children arrive in document order and that order alone is not evidence one caused or preceded another — a model will otherwise read adjacency as causation. Children are themselves model-written text that may carry an instruction laundered out of the source, so each is fenced separately with a derived delimiter and the same precedence rule applies to them as to source.

Legal identifiers are not enumerated in the prompt. At the second level a legal set can run to thousands of identifiers and would spend a third of the request on a list, so the prompt refers to the authoritative source passages it received and the validator enforces the real set.

Three merge levels are not reachable from document size alone at a hosted model's capacity — it would take roughly twenty million source tokens — so multi-level behaviour is exercised by configuring a narrower ceiling on children per merge. That ceiling is unset by default, leaving measurement in charge.

## Final editorial output and audit artifacts

`summarizer.pipeline` is the library-level path from a canonical document to a
direct or grounded hierarchical root and then one dedicated final editorial
call. The editor consumes the root's structured record, preserves material
qualifications and contradictions, and returns readable plain text by default.
It deliberately is not connected to the legacy command line yet; that remains
the later end-to-end integration work.

Callers may opt into a compact source list. Those citations are not generated
by the final writer: they are a deterministic, source-ordered rendering of the
root's already validated provenance, and each identifier resolves to recorded
segment metadata. The default output has neither citations nor audit data.

An optional `audit/2` artifact records safe run configuration, strategy,
source-segment metadata, tree/evidence links, warnings or failures, available
usage metadata, citation mappings, and an explicit verification record. It does
not copy raw source text, generated prose, quotations, provider request IDs, or
provider authentication data. Values matching common credential forms are
redacted; JSON is canonical, validated after serialization, and atomically
written only after validation.

## Optional claim verification

`PipelineConfig.verification` is disabled by default, so existing library
callers keep the same readable final text, provider calls, and citation behavior.
Verification is currently a library option; issue #12 owns CLI exposure when it
replaces the transitional legacy workflow.

When enabled, finalization verifies the editorial draft before citations are
rendered. By default it uses the same provider, model, token counter, timeout,
and context window as summarization. A caller may instead inject a complete
dedicated `VerificationRuntime`; supplying only a model is intentionally not a
configuration path.

Verification decomposes locally derived sentence spans, retrieves only bounded
complete source cores from the root's recorded provenance, and classifies each
claim with validated structured results. A contradicted eligible span can be
qualified, replaced, or removed once within the default repair budget; the
complete repaired draft is then decomposed and verified again. Retrieval is
bounded: `insufficiently_supported` means the selected evidence did not settle
the claim, not that the source lacks support. Likewise, `supported` is an
evidence-scoped verifier assessment, not proof of factual perfection.

Malformed verification, a capacity failure, or an unresolved material
contradiction fails closed. If auditing is configured, finalization writes a
validated terminal `audit/2` record before withholding the reader-facing
summary; citations are rendered only after successful verification. Audit
metadata keeps pass identifiers, hashes, verdicts, evidence links, repairs,
usage, and closed limitation/failure codes, while excluding claim, draft,
replacement, quote, and source prose.

## Reliability, cache, and resume

Reliability is opt-in for library callers. Set `PipelineConfig.cache.enabled`,
provide a stable `ReliabilityConfig.run_id`, and select `run_mode="new"` or
`run_mode="resume"`. The default root is `.summarizer-cache/`, which is ignored
by Git. `max_in_flight` defaults to `1`; values above one bound concurrent leaf
calls and independent merge groups without changing their source order or merge
level barriers.

The cache is JSON-only. It stores immutable, content-addressed stage objects
under `objects/` and a versioned JSON manifest per run under `runs/`. A new run
may adopt any compatible validated object already in the cache and immediately
records that reference in its manifest. A resumed run trusts only its manifest's
references. Missing, corrupt, wrong-version, or incompatible objects are safe
misses and are recomputed.

Only parsed and locally validated terminal results are reusable. That includes
validated segmentation, grounded direct/leaf/merge records, a parsed editorial
draft, and verification only when `VerificationResult.failed` is false. Raw
provider responses, exceptions, ambiguous or failed work items, and failed
verification never become cache references. A successful sibling observed while
a failed leaf or merge batch drains is validated and checkpointed independently.
Cache files can contain source-derived or generated
text, so directories are created with mode `0700` and files with mode `0600`.
Descriptors and audit projections exclude application credentials, hosts,
paths, prompts, request bodies, and raw provider errors. Cached payloads can
still reproduce credential-like text found in the source artifact, so the cache
root remains sensitive local data and is not encrypted.

Reliability-enabled `audit/3` records closed cache hit/miss reasons and retry
counts without raw provider errors. Entries follow the manifest's stable work
order even when calls finish concurrently. Verification decomposition,
classification, repair, and later passes aggregate under the stable `V01` work
identifier rather than exposing their provider completion order.

When cache reliability and `PipelineConfig.audit_path` are both enabled,
`run_pipeline` publishes a validated `audit/3` artifact to that path before it
atomically replaces `AppConfig.output_path` with the readable summary. The run
manifest records both SHA-256 digests and moves through `audit_staged` to
`complete`; `read_published_summary` accepts the summary only when both files
match a complete manifest. Resume can finish a missing completion marker or
republish a damaged pair. This is a recoverable protocol, not cross-path
atomicity. Calls sharing an output pair are serialized within one process, so
separate processes must not publish different runs to the same output paths.

Without this opt-in combination, `run_pipeline` returns its final result in
memory and retains the existing standalone audit behavior. The CLI exposes
neither cache/resume nor reliable paired publication yet.

Historical scripts under `omscs-ml-lectures/` remain available but are not part of the modern application entry point.

## Tests

Install development dependencies and run the offline suite:

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m pytest -q --import-mode=importlib
```

The suite blocks outbound socket connections and uses deterministic provider
fakes. It does not require an OpenAI credential, a running Ollama service, or a
locally installed model.

## Contributing

For major changes, open an issue before submitting a pull request.
