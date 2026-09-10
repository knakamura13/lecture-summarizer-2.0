# Legacy Summarizer Baseline

This document records the observable behavior of the legacy summarizer before the generalized hierarchical pipeline is introduced. The characterization suite protects compatibility where useful and makes known defects visible. These findings describe the current implementation; they are not the target behavior for the rebuilt pipeline.

## Running the offline suite

From the repository root, create and activate a virtual environment, install the development dependencies, and run the full suite:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -v
```

The suite blocks connection attempts through `socket.create_connection` and `socket.socket.connect`. It replaces OpenAI, NLTK, dotenv, tqdm, and requests behavior with deterministic test doubles, so the characterization tests do not need network access, credentials, or downloaded tokenizer data.

## Compatibility constraints

The executable workflow currently provides these observable guarantees:

- `python main.py` reads UTF-8 text from `input.txt` in the current working directory.
- A successful run writes UTF-8 text to `output.txt` in the current working directory.
- The workflow performs one summarization operation for each source chunk in source order. A positive `MAX_CHUNKS` value keeps only that many chunks from the start of the source, while the default `-1` processes every chunk.
- Provider responses are joined with exactly one newline between adjacent responses.

## Provider prompt and logging shape

Each executable-path summarization passes the alias `gpt-4`, which resolves to `gpt-4-1106-preview`, and sets a 180-second provider timeout. The request contains exactly two messages in this order:

- The `system` message is `You are a writing assistant, skilled in revising and summarizing complex technical writing with accuracy and precision.`
- The `user` message is the following fixed text and chunk template. Each delimiter is a newline, three double-quote characters, and another newline (`\n"""\n`).

```text
Provide an executive summary of the following text (delimited by triple quotes). Present the key ideas and findings directly, without bullet points, as if for a busy professional who needs to grasp the essential points quickly. Ignore complete sentences and grammatical correctness. Abbreviate long and repetitive words.
"""
<source chunk>
"""
```

Successful response text has whitespace collapsed before it is added to `output.txt`. Every successful chunk makes one write attempt to `gpt_logs/<timestamp>_gpt.txt`, where `<timestamp>` is the value returned by `time.time()`. The artifact contains the original chunk under `PROMPT:`, a line of ten equals signs, and the normalized response under `RESPONSE:`. If two chunks receive equal timestamp values, the later write uses the same filename and overwrites the earlier artifact, so a multi-chunk run is not guaranteed to retain one distinct artifact per chunk.

At import, `logging.basicConfig` requests an INFO-level file handler at `summarizer.log` in the current working directory. Python only applies that request when the root logger has no handlers already configured. File read and write failures are logged as errors, provider failures are logged for each attempt, exhausted retries are logged as errors, and an executable workflow failure is logged as critical. The executable catches that final exception, so it does not use a nonzero process exit to signal failure.

Provider operations make at most five attempts. A failed attempt waits 1, 2, 4, then 8 seconds before the next attempt. Five `requests.exceptions.RequestException` failures return `GPT error: Unknown error`. Five generic exceptions expose the legacy `UnboundLocalError` defect instead of returning error text. A successful attempt returns immediately and does not perform later retries.

## Known legacy limitations

The characterization suite preserves evidence for these limitations:

- The chunk character budget can exceed the configured limit because the separator added before a sentence is not included in the budget check.
- A sentence longer than the configured chunk size remains unsplit.
- Importing `main.py` downloads the NLTK `punkt` resource and constructs an OpenAI client.
- Prompts and model aliases are fixed in source code.
- Intermediate summaries are concatenated directly, without recursive synthesis.
- Request-related provider errors can become output text, while a terminal generic provider failure raises `UnboundLocalError`.
- Fatal executable errors are logged but do not produce a nonzero process exit.
- Module constants require source edits rather than runtime configuration.

These are findings about legacy behavior, not requirements or target behavior for the new pipeline.

## Issue #3 replacement behavior

Issue #3 replaces the legacy executable seams while retaining the characterized flat workflow for transition. `main.py` becomes an import-safe CLI entry point. Configuration moves to frozen validated values, OpenAI calls use the Responses API behind an injectable provider, and retry behavior moves to a provider-neutral decorator.

The following legacy limitations are intentionally fixed:

- importing application modules no longer downloads NLTK data, constructs an OpenAI client, configures file logging, calls the network, or writes files;
- credentials are read by the official client from `OPENAI_API_KEY` only when a live provider is first used;
- provider failures raise typed exceptions and cannot become summary text;
- exhausted retries preserve the final transient failure as their cause;
- CLI failures use nonzero exit codes;
- input, output, model, timeout, retry count, chunk size, maximum chunks, and dry-run behavior are configurable and validated.

Sentence packing, fixed character budgets, independent chunk summaries, and newline concatenation remain transitional compatibility behavior. Later issues replace those policies with token-aware segmentation, hierarchy, grounding, and evaluation.

## Fixture corpus

The offline corpus supplies five compact, fictional source genres for future evaluation:

- `article.txt` is a municipal news article with dated actions, attributed claims, and an unsigned funding qualification.
- `report.txt` is a quarterly operations report with metrics, causes, recommendations, and preliminary figures.
- `transcript.txt` is a meeting exchange with corrections, proposals, and a decision that remains pending.
- `structured.md` is a migration plan with headings, nested lists, scoped work, rollback details, and an unresolved warning.
- `narrative.txt` is a chronological field account whose observations do not establish a final cause.

The existing `omscs-ml-lectures/` historical lecture data remains unchanged by this baseline harness.

## Historical status

This file is a characterization of the pre-hierarchy and transitional Issue #3 behavior. After the Issue #12 integration cutover, statements above that say the legacy workflow is “current” or “transitional” describe the historical snapshot, not the modern CLI. The final CLI keeps the no-argument `input.txt` to `output.txt` contract but uses token-aware direct/hierarchical execution, removes `--chunk-size` and `--max-chunks`, and makes dry-run a budget/strategy report with no provider construction or final-output write. The regression dispositions and final evidence map are maintained in [docs/evaluation.md](evaluation.md); the historical fixture and lecture data remain unchanged.
