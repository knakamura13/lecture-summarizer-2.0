# Recursive Text Summarizer

Recursive Text Summarizer turns a normalized UTF-8 text or Markdown document into one source-grounded summary. It is provider-neutral and makes no assumptions about the document's subject or genre. Small documents can be summarized directly; larger documents are segmented and reduced through a balanced, source-grounded hierarchy before a final editorial pass.

The project supports OpenAI and locally served Ollama models through the same pipeline. The default reader-facing output is plain text. Optional citations and a validated JSON audit artifact expose the source and tree metadata without copying raw source or generated prose into the audit file.

## Supported input

The application reads one non-empty UTF-8 text file. Markdown is supported as text; headings, paragraphs, lists, indentation, and internal blank lines remain meaningful to segmentation. Ingestion removes a leading BOM, normalizes CRLF/CR to LF, removes trailing spaces and tabs on each line, and trims blank lines at the document edges. It does not OCR images, transcribe audio, fetch external facts, or interpret a document as a lecture, textbook, or other special domain.

Source text is treated as untrusted data. Instructions inside the source cannot replace the summarization instructions. An empty file, an invalid UTF-8 file, or an unreadable path fails before a final output is published.

## Installation and provider configuration

Requirements are Python 3.10 or newer and either an OpenAI API key or a running Ollama service.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For OpenAI, export the key before a run:

```sh
export OPENAI_API_KEY="your-api-key"
```

The OpenAI provider reads `OPENAI_API_KEY` when the provider is first used. The key is not accepted as a CLI value and is not written to logs, cache descriptors, audit artifacts, or output. The application does not automatically load a `.env` file; export environment variables explicitly or configure them through your process manager.

For Ollama, install it from [ollama.com/download](https://ollama.com/download), start the service, and pull a model. A local Ollama provider does not require an API key:

```sh
ollama serve
ollama pull gemma3:4b
```

The default Ollama endpoint is `http://localhost:11434`. Use `--ollama-host` for another endpoint. The application does not pull models automatically. A missing model and an unavailable service produce actionable provider errors.

## Basic usage

With no arguments, the CLI reads `input.txt` and writes `output.txt` in the current directory:

```sh
python main.py
```

Choose paths and a hosted model explicitly:

```sh
python main.py --input source.txt --output summary.txt --model gpt-4o-mini
```

Use Ollama with any model tag already installed in the service:

```sh
python main.py --provider ollama --model gemma3:4b
python main.py --provider ollama --model qwen3.8 --ollama-host http://localhost:11434
```

A successful run publishes one final editorial summary only after all required stages succeed. Provider failures, invalid configuration, missing input, and exhausted retries return a nonzero exit status and do not replace the final output with an error message.

### Dry run

`--dry-run` reads and normalizes the source and reports the measured budget and selected strategy without constructing a provider, making a model request, or writing the final output:

```sh
python main.py --dry-run
```

An explicit-source example is:

```sh
python main.py --input source.md --strategy auto --dry-run
```

Dry run is for configuration and budget inspection. It is not a summary-quality evaluation.

## Strategies and token budgets

The strategy decision measures the real request overhead, reserved output, and safety margin before selecting a path:

- `auto` (default) chooses `direct` only when the complete document provably fits a known context window and any direct-size cap. Otherwise it chooses `hierarchical`.
- `direct` sends the complete document in one structured summarization call and fails before a provider call if the budget cannot fit it.
- `hierarchical` uses token-aware, structure-aware segments, structured leaf summaries, balanced merge levels, original-source passages for grounding, and one final editorial call.

A model whose context window is not known uses an assumed window for reporting; `auto` does not gamble on a direct request in that case. `--context-window` supplies an explicit window. OpenAI models with a registered `tiktoken` encoding use exact counts; Ollama and non-OpenAI providers use a conservative UTF-8-byte estimator by default. Library callers can supply `encoding_name` to `resolve_token_counter` to select an exact `tiktoken` counter for any provider. An unknown OpenAI model reports an actionable token-accounting error rather than silently falling back. Constructing a `tiktoken` counter may download an uncached vocabulary; counting afterwards is local.

Segmentation prefers headings, paragraphs/lists, and sentences, then uses a token-safe character fallback for an oversized unit. A segment has a stable identifier (`S000001`, etc.), a disjoint core range, and optional overlap context. Overlap provides context only; it does not duplicate evidence ownership or move core boundaries. By default, source-segment cores are capped at one quarter of the safely measured leaf input capacity so adaptive merge planning retains a viable two-child grounding fallback; an explicit `--chunk-tokens` value remains authoritative. Merge fan-out is measured from the complete request, and the default hierarchy narrows child groups when selected source passages need more room. An explicit internal `GroundingPolicy` can instead provide a fixed source-token reserve. `--max-merge-children` can impose a smaller ceiling to force a deeper hierarchy.

## CLI reference

Run `python main.py --help` for parser-generated help. The complete options are:

| Option | Default | Purpose |
| --- | --- | --- |
| `--input PATH` | `input.txt` | UTF-8 source text or Markdown file. |
| `--output PATH` | `output.txt` | Final plain-text summary path. |
| `--provider {openai,ollama}` | `openai` | Provider adapter. |
| `--model MODEL` | `gpt-4o-mini` | Provider model identifier. |
| `--ollama-host URL` | `http://localhost:11434` | Ollama service endpoint. |
| `--timeout SECONDS` | `180` | Per-provider request timeout. |
| `--max-retries N` | `5` | Maximum attempts for retryable provider failures. |
| `--strategy {auto,direct,hierarchical}` | `auto` | Execution strategy. |
| `--context-window TOKENS` | unset | Explicit total context window; otherwise use the model table/assumed value. |
| `--max-output-tokens TOKENS` | `1024` | Output tokens reserved during budget calculations. |
| `--safety-margin-tokens TOKENS` | `256` | Fixed budget safety floor. |
| `--safety-margin-fraction FRACTION` | `0.02` | Fractional safety margin; the larger margin applies. |
| `--max-direct-tokens TOKENS` | unset | Optional direct-path cap used by `auto`. |
| `--target-words N` | `300` | Target size for final editorial writing. |
| `--chunk-tokens TOKENS` | unset | Maximum leaf segment tokens; unset uses measured hierarchical capacity. |
| `--overlap-tokens TOKENS` | `0` | Context overlap around adjacent segment cores. |
| `--max-merge-children N` | unset | Optional upper bound on children per merge. |
| `--verify` | off | Verify final-draft claims against bounded source evidence. |
| `--max-repair-passes N` | `1` | Maximum verification repair passes (`0` disables repairs while verification remains enabled). |
| `--citations` | off | Append a deterministic, source-ordered `Sources:` list. |
| `--audit PATH` | unset | Write validated audit JSON (`audit/2`, `audit/3` with direct-run reliability metadata, or `audit/4` for hierarchical merge grounding). |
| `--cache-dir PATH` | unset | Opt into the local JSON cache and resumable run manifest. An actual cached run also requires `--run-id` and `--audit`. |
| `--run-id ID` | unset | Stable identifier required when cache is enabled. |
| `--resume` | off | Resume the manifest named by `--run-id`; requires `--cache-dir` and `--run-id`. |
| `--max-concurrency N` | `1` | Maximum in-flight leaf/merge calls; values above `1` require `--cache-dir`; output order remains deterministic. |
| `--dry-run` | off | Report budget/strategy without provider construction or final-output writes. |

`--chunk-size` and `--max-chunks` are removed. They are not aliases: the pipeline no longer exposes the old character-chunk and prefix-truncation controls.

## Output and audit artifacts

The output path contains the final editorial text. Without `--citations`, it contains only that text. With citations, a source-ordered list such as `Sources: [S000001, S000004]` is appended from validated root provenance. Merge references remain valid across all segments covered by their children even when the grounding budget supplies only a subset of source passages: omitted child references are preserved locally, while grounded references may be explicitly dropped after correction. Verbatim quotation checks apply only to passages actually supplied. Citations are not invented by the editorial model.

`--audit PATH` writes a canonical, validated JSON artifact. The top-level audit fields are:

- `schema_version`: `audit/2` for ordinary direct output, `audit/3` for direct output with reliability metadata, or `audit/4` when a hierarchy executes a merge;
- `source_id`, `strategy`, and `model`;
- safe `configuration` and budget metadata;
- `source_segments` with identifiers, source order, core/context ranges, token counts, overlap counts, and boundary kind;
- `tree_nodes` and `root_node_id`, including levels, child links, structural covered segments, narrowed provenance, content-unit classifications, and evidence links. `audit/4` additionally records each executed merge's selected IDs, budget omissions, fixed reserve or adaptive request capacity, and reason;
- source-ordered `citations` and provider `usage` metadata when available;
- closed-code `warnings` and `failures`;
- `verification`, including pass/claim/evidence links, verdicts, repair actions, usage, warnings, limitations, and failures;
- `reliability` in `audit/3` and reliable `audit/4` output, including cache outcomes, retry categories, resume state, reuse count, and recomputation count.

Audit artifacts deliberately do not contain raw source text, generated summary prose, quotations, prompts, request bodies, provider request IDs, or authentication data. Source-derived text may still exist in cache payloads, so cache directories are sensitive local data even though descriptors and audit projections are secret-safe.

## Cache, resume, retries, and concurrency

Caching is opt-in. `--cache-dir` enables a JSON object store and run manifests; an actual cached run also requires `--run-id` and `--audit` for witnessed paired publication. `--resume` requires cache and run ID. Dry-run may inspect the budget without opening the cache or writing artifacts. A new run may reuse compatible, validated objects already in the cache. Missing, corrupt, wrong-version, or incompatible objects are safe misses and are recomputed.

Only parsed and locally validated terminal results are reusable. Raw provider responses, exceptions, failed work items, and failed verification do not become cache references. Cache directories are created with restrictive permissions, but the cache is not encrypted. Do not commit it.

Retryable timeout, rate-limit, connection, and server failures use bounded exponential backoff; non-retryable authentication, request, response, and configuration errors fail immediately. `--max-retries` controls the attempt limit. Concurrent independent work is bounded by `--max-concurrency` when cache reliability is enabled, while manifest order, source order, merge-level barriers, and audit entries remain deterministic. Final paired publication (summary plus reliable audit/3 or audit/4) uses a manifest witness so an incomplete run is not accepted as complete; separate processes must not publish different runs to the same output pair.

## Verification limitations

`--verify` decomposes the editorial draft into claims, retrieves bounded complete source cores, classifies claims as `supported`, `contradicted`, `insufficiently_supported`, or `not_meaningfully_verifiable`, and may qualify, replace, or remove a problematic span within the repair budget. A `supported` verdict is evidence-scoped, not proof of factual perfection. Retrieval can be incomplete, and malformed verification, capacity failures, or unresolved material contradictions fail closed. Verification uses the selected provider by default and can consume additional requests and budget.

The offline evaluator uses a deterministic, source-sensitive fake provider. It tests orchestration, provenance, source mutation behavior, and rubric evidence; it does not measure live-model coherence or quality. Live OpenAI/Ollama quality must be inspected separately and is not guaranteed by this project.

## Migration from the original workflow

The no-argument contract remains:

```sh
python main.py   # reads input.txt, writes output.txt
```

The migration intentionally changes the internals and several controls:

- fixed character-size sentence chunks, independent summaries, and newline concatenation are replaced by token-aware segmentation, direct or hierarchical execution, and a final editorial synthesis;
- `--chunk-size` and `--max-chunks` are removed rather than accepted as misleading aliases; use `--chunk-tokens`, `--strategy`, `--max-direct-tokens`, or `--max-merge-children` for the new controls;
- the old sentence-tokenizer injection seam is replaced by a token-counter seam for the pipeline;
- dry-run no longer writes the source or a final output and does not construct a provider; it reports budget/strategy only;
- model/provider settings are runtime options, with `OPENAI_API_KEY` supplied through the environment and Ollama usable without a key;
- failures use nonzero exit status and never become summary text or misleading partial output;
- the historical `omscs-ml-lectures/` scripts and input data remain untouched and are not the modern entry point.

No `.env` file is loaded automatically. Existing automation should export its credentials and migrate removed flags before invoking the CLI.

## Tests and offline evaluation

Install development dependencies and run the offline suite:

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m pytest -q --import-mode=importlib
```

The suite blocks outbound network access and uses deterministic provider fakes. It does not require an API key, an Ollama service, or a downloaded model. Relevant coverage includes `tests/test_ingestion.py`, `tests/test_tokenization.py`, `tests/test_segmentation.py`, `tests/test_budget.py`, `tests/test_hierarchy.py`, `tests/test_pipeline.py`, `tests/test_cli.py`, `tests/test_documented_cli.py`, the provider tests under `tests/providers/`, reliability tests (`tests/test_cache.py`, `tests/test_checkpoint.py`, `tests/test_pipeline_reliability.py`, `tests/test_scheduler.py`), audit/publication tests, and verification tests.

The repeatable five-genre integration evaluator is:

```sh
python -m tests.support.evaluation --output-dir <temporary-directory>
```

It writes one `evaluation.json` plus per-case audit artifacts in the requested temporary directory. `tests/test_evaluation.py` validates the deterministic fake, direct/auto paths, multi-level hierarchy, citations, provenance, audit links, and source-sensitive mutation behavior. See [docs/evaluation.md](docs/evaluation.md) for the rubric, output schema, manual inspection procedure, regression dispositions, and parent definition-of-done evidence map. Do not commit evaluator output, caches, credentials, or temporary files.

## Contributing

Before closing an implementation issue, the merged diff must pass an adversarial review against the issue's acceptance criteria. Green CI is not sufficient on its own.

**Reviewer checklist before closing:**

- **Diff review**: Read the actual merged diff against each acceptance criterion. Confirm the implementation satisfies them, not merely that tests pass.
- **Real-dependency checks**: Where behavior depends on a real provider, file format, or external integration, verify against those real dependencies—not only mock-based coverage.
- **Documentation vs. executable behavior**: Run the documented commands (CLI flags, example invocations, README snippets). Confirm the documented behavior matches the code path reached by those commands.
- **Test mutation**: Where practical, mutate or negate new tests (remove an assertion, invert a condition) to confirm they fail when the implementation is broken. A test that passes under mutation is not providing coverage.
- **Source-tree verification**: Check the actual merged source tree for the fix—locate the changed function or module and read it. Do not rely on CI artifact reports as a proxy for reading the code.

**Closing comment requirements:**

The closing issue comment must record:
1. The merge evidence (PR number, merge commit SHA).
2. Which acceptance criteria were verified and how (commands run, files read, tests mutated).
3. Any findings or limitations discovered during the review.

## Limitations

The system is an orchestration and grounding implementation, not a guarantee of model truth. Conservative token estimates can reduce packing efficiency; context-window tables are maintained metadata and may require `--context-window`; large or unusual blocks can be split at a hard fallback boundary. Ollama and OpenAI differ in transport behavior, so provider errors remain possible. Verification is best effort and bounded. OCR, audio, external research, GUI operation, and cross-process shared publication are outside the supported scope.
