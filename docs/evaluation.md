# End-to-end evaluation and migration evidence

This document defines the offline integration evaluation for issue #12 and records how its results map to the original baseline and parent issue #1. It is a procedure and evidence map, not a live-model quality claim. The evaluator uses a deterministic, source-sensitive fake provider so that orchestration, provenance, source mutation, and failure behavior are repeatable without network access. It cannot establish how a live OpenAI or Ollama model writes, selects, or verifies content.

## Reproduction command

From the repository root, run the evaluator with a disposable output directory:

```sh
python -m tests.support.evaluation --output-dir <temporary-directory>
```

The runner executes five original-genre cases on the `auto` path, one article case on the `direct` path, and one expanded article case on `auto`. The expanded case is configured to exercise a genuine hierarchy with more than one merge level. It writes only the requested directory:

- `evaluation.json`, one JSON object containing all cases;
- `{case_id}.audit.json` for each case (`article-auto`, `article-direct`, and `article-auto-hierarchy`, for example), produced by the pipeline's audit writer.

Do not commit that directory. The evaluator blocks network access and identifies the provider as `deterministic-source-sensitive`. A test runner may use temporary directories and inspect the JSON without changing the fixture corpus.

The repository's automated gate is `tests/test_evaluation.py`; component and integration evidence remains in `tests/test_pipeline.py`, `tests/test_hierarchy.py`, `tests/test_audit.py`, `tests/test_provenance_validation.py`, `tests/test_publication.py`, `tests/test_cache.py`, `tests/test_checkpoint.py`, `tests/test_pipeline_reliability.py`, `tests/test_scheduler.py`, and the verification test modules. The exact executed pass/fail result and any observed runtime values belong to the centralized verification run; this document does not invent scores, timings, token counts, or provider-quality results.

## Fixtures

| Case | Source | What it exercises |
| --- | --- | --- |
| `article` | `tests/fixtures/article.txt` | Municipal news prose, dated actions, attributed claims, and an unsigned funding qualification. |
| `report` | `tests/fixtures/report.txt` | Metrics, causes, recommendations, and preliminary figures. |
| `transcript` | `tests/fixtures/transcript.txt` | Speaker turns, corrections, proposals, and a decision that remains pending. |
| `structured` | `tests/fixtures/structured.md` | Headings, nested lists, scoped work, rollback details, and an unresolved warning. |
| `narrative` | `tests/fixtures/narrative.txt` | Chronological observations where sequence does not prove a final cause. |

`tests/test_fixture_corpus.py::test_representative_fixture_is_nonempty_utf8` protects the corpus's non-empty UTF-8 contract. Historical lecture data under `omscs-ml-lectures/` is not an evaluation fixture and must remain unchanged.

## Output schema: `evaluation/1`

The evaluator's top-level object has this shape (field names are contractual; values are observed per run):

```json
{
  "schema_version": "evaluation/1",
  "offline": true,
  "provider": "deterministic-source-sensitive",
  "fixture_corpus": ["article.txt", "report.txt", "transcript.txt", "structured.md", "narrative.txt"],
  "rubric_dimensions": [
    "coherence",
    "salient_content_coverage",
    "provenance_resolution",
    "qualification_contradiction_retention",
    "source_faithfulness"
  ],
  "cases": [
    {
      "case_id": "article-auto",
      "genre": "article",
      "fixture": "article.txt",
      "strategy_requested": "auto",
      "strategy_selected": "direct",
      "strategy_reason": "...",
      "root_level": 0,
      "document_tokens": 0,
      "provider_calls": 0,
      "final_text": "...",
      "citations": [{"segment_id": "S000001", "source_id": "...", "order": 0}],
      "audit_path": ".../article-auto.audit.json",
      "rubric": {
        "coherence": {"passed": false, "observed": "", "evidence": []},
        "salient_content_coverage": {
          "passed": false,
          "observed": "",
          "evidence": [],
          "claims": [{"id": "article-scale", "evidence_span": "", "observed": false}]
        },
        "provenance_resolution": {"passed": false, "observed": "", "evidence": []},
        "qualification_contradiction_retention": {"passed": false, "observed": "", "evidence": []},
        "source_faithfulness": {"passed": false, "observed": "", "evidence": []}
      },
      "heuristic_vs_human": {
        "deterministic_fake_evidence": true,
        "heuristic_checks": ["..."],
        "human_inspection": ["..."],
        "live_model_evaluation": false
      },
      "passed": false
    }
  ],
  "passed": false
}
```

The example shows types and required keys, not a result. The report records requested and selected strategies, measured document tokens, provider call count, root depth, final text, source-resolving citations, five rubric dimensions, and an aggregate `passed` value. `rubric.*.observed` and `.evidence` are explicit observations rather than hidden model judgments; coverage additionally records curated claim IDs and source spans. `heuristic_vs_human` keeps automated signals separate from human review and explicitly sets `live_model_evaluation` to false.

Each audit artifact uses the pipeline audit contract: `audit/2` for an ordinary direct case, `audit/3` for direct output with reliability metadata, or `audit/4` when a hierarchy executes a merge. Its source segment identifiers, tree nodes, citations, evidence links, and root identifier must resolve internally. The audit intentionally omits raw source prose, generated prose, quotations, prompts, request bodies, and credentials. See the README's [audit section](../README.md#output-and-audit-artifacts) for the field-level contract.

## Five-dimension rubric

A reviewer should apply the same procedure to every case. Read the fixture, the final text, `evaluation.json`, and the corresponding audit artifact. Record observations in the evaluator's explicit strings and keep human judgments separate from heuristics. A pass requires the stated criterion; an unobserved or ambiguous condition is not a pass.

### 1. Coherence

**Pass when:** the final text is one standalone summary with an intelligible order, consistent names, no chunk-boundary seams, no repeated leaf paragraphs, and no unsupported lecture-specific framing. The summary should be appropriate to the source's genre and target length.

**Procedure:** read the final text once without the fixture, identify its organization and referents, then compare it with the source order. For an `auto` long case, inspect `root_level` and audit `tree_nodes` to confirm that the final text was produced after recursive reduction rather than concatenating independent responses. Heuristic checks may detect empty text, repeated exact spans, or leaked JSON; they cannot judge writing quality.

### 2. Salient-content coverage

**Pass when:** all curated must-retain claims for the fixture are represented in the final text at an appropriate level of detail, with material numbers, decisions, actors, and outcomes retained. Omitted detail that is not salient is acceptable; omission of a curated claim is not.

**Procedure:** use the case's `rubric.salient_content_coverage.claims` entries. For every claim ID, locate the stated source/evidence span in the fixture and a corresponding observation in `final_text`; confirm its audit evidence links include the source segment. A claim may be summarized or paraphrased, but the observation must not depend on a model outside the source. The evaluator records coverage heuristically; a reviewer decides whether the retained detail is materially adequate.

### 3. Provenance resolution

**Pass when:** every citation and every evidence reference used by the root resolves to an existing source segment with the same source identity, and citations appear in source order without duplicates.

**Procedure:** load the audit JSON; build maps from `source_segments.segment_id` and `tree_nodes.node_id`; resolve `root_node_id`, child links, `covered_segments`, summary provenance, content-unit evidence, annotation evidence, and citations. Confirm citation `order` matches the referenced source segment. `tests/test_audit.py`, `tests/test_provenance_validation.py`, and the evaluator's citation checks provide machine evidence; the reviewer should inspect a sample link back to the fixture.

### 4. Qualification and contradiction retention

**Pass when:** the result retains material uncertainty, preliminary status, attribution, unresolved warnings, corrections, disagreements, and contradictions rather than upgrading them to settled facts. A transcript correction must not be silently treated as the original assertion; a narrative sequence must not be turned into causation.

**Procedure:** inspect the fixture's marked qualifiers/corrections and the curated rubric observations. Search the final text for the relevant qualification and inspect the root/audit content-unit flags and qualification/contradiction evidence. For a case with no contradiction, the criterion still requires that uncertainty or attribution not be strengthened. Verification verdicts are supporting evidence only, not proof of truth.

### 5. Source faithfulness

**Pass when:** factual content in the final text is entailed by the fixture and its linked source cores, with no external facts, invented causal links, or claims created by prompt-injection-like source text. The source-sensitive mutation check must change the relevant deterministic output or fail the expected assertion; a fixed canned answer is not acceptable.

**Procedure:** inspect every curated claim's evidence span, compare key names/numbers/relationships with the fixture, and review source-sensitive mutation evidence. Confirm the fake provider receives source-derived input and that changing a source claim changes the corresponding result. This demonstrates grounding and pipeline sensitivity only. It does not predict live-provider faithfulness, detect every hallucination, or establish factual correctness beyond the supplied fixture.

## Manual review record

For each case, record:

1. fixture and strategy, including the observed strategy decision and root depth;
2. one or more exact observations for each rubric dimension;
3. any unresolved limitation or ambiguity;
4. whether the observation came from a heuristic, an audit invariant, or human inspection;
5. whether all citations/evidence links resolved;
6. whether the case is suitable for a live-provider follow-up (optional, outside this offline gate).

Do not convert a heuristic pass into a claim that a live model is coherent or faithful. A deterministic fake can make the same answer repeatably while still being unlike a production model.

## Baseline regression and disposition matrix

The baseline below is derived from `docs/legacy-baseline.md`. It distinguishes compatibility retained intentionally from behavior replaced by issue #1/#12. “Evidence” names implementation or tests that a reviewer can inspect; it is not an assertion that a centralized run has already passed.

| Historical behavior or constraint | Final disposition | Evidence to inspect |
| --- | --- | --- |
| No-argument run reads UTF-8 `input.txt` | Retained | `main.py`; `summarizer.ingestion.read_source`; `tests/test_cli.py` default workflow. |
| Successful run writes UTF-8 `output.txt` | Retained, with atomic publication when reliability is enabled | `summarizer.finalization.publish_final_output`; `tests/test_cli.py`; `tests/test_publication.py`. |
| Fixed character chunks and sentence packing | Replaced by token/structure-aware segmentation | `summarizer.segmentation`; `tests/test_segmentation.py`; `tests/test_tokenization.py`. |
| `MAX_CHUNKS` prefix truncation | Removed; no misleading replacement alias | final CLI parser; `tests/test_cli.py`; help contract in `tests/test_documented_cli.py`. |
| Exactly one newline between independent chunk responses | Replaced by one final editorial result | `summarizer.pipeline._run_pipeline`; `summarizer.editorial`; `tests/test_pipeline.py`. |
| Legacy `gpt-4` alias to `gpt-4-1106-preview` | Replaced by configurable model, default `gpt-4o-mini` | `AppConfig`; provider tests; CLI defaults. |
| Fixed legacy prompt and two-message request shape | Replaced by provider-neutral structured leaf/merge/editorial requests | `summarizer.leaf`, `summarizer.merge`, `summarizer.editorial`; provider adapter tests. |
| Per-chunk `gpt_logs/<timestamp>_gpt.txt` artifacts | Removed; optional validated audit replaces them | `summarizer.audit`; `tests/test_audit.py`; README audit contract. |
| Import-time NLTK/client/logging/file/network side effects | Removed | import-safe `main.py`/CLI; `tests/test_entrypoint.py`; `tests/test_offline_guard.py`. |
| Legacy retry timings and generic exception defect | Replaced by typed provider errors and bounded retry policy | `summarizer.providers.retrying`; `tests/providers/test_retrying.py`; `tests/test_pipeline_reliability.py`. |
| Fatal executable errors logged but exit code remained zero | Replaced by actionable nonzero CLI failures | `summarizer.cli.main`; `tests/test_cli.py`. |
| Runtime configuration required source edits | Replaced by validated CLI/config objects | `summarizer.config`; `tests/test_config.py`; `tests/test_strategy_config.py`. |
| Dry run still wrote source/output under the transition | Changed: reports budget/strategy without provider construction or final-output writes | CLI `--dry-run`; dry-run coverage in `tests/test_cli.py`. |
| Credentials supplied by provider environment | Retained and tightened: `OPENAI_API_KEY` only; Ollama needs no key | `summarizer.providers.openai`; `summarizer.providers.ollama`; provider tests. |
| Historical lecture files and scripts | Preserved and outside modern entry point | `omscs-ml-lectures/`; `tests/test_fixture_corpus.py`; repository diff review. |

This matrix is a disposition, not a claim that all rows have passed a particular run. The centralized verification record supplies execution evidence.

## Parent #1 definition-of-done evidence map

The parent has 14 definition-of-done items. The map records the strongest inspectable evidence and the remaining review boundary. It deliberately does not declare the parent ready to close.

| # | Parent definition of done | Evidence / disposition |
| ---: | --- | --- |
| 1 | Arbitrary long-form text works without domain assumptions | Ingestion, segmentation, leaf, merge, and editorial prompts are generic; five distinct fixtures exercise genres. Evaluation rubric checks source-specific content. |
| 2 | Small inputs use direct summarization when appropriate | `summarizer.budget.select_strategy`; `tests/test_strategy_selection.py`; evaluator direct cases. |
| 3 | Large inputs use a genuine recursive multi-level hierarchy | `summarizer.hierarchy.build_hierarchy`; `tests/test_hierarchy.py`; evaluator `auto` long case checks root depth/tree nodes. |
| 4 | Merge stages retain traceable original-source links | `summarizer.grounding`, `summarizer.audit`; `tests/test_provenance_validation.py`; audit resolution procedure above. |
| 5 | Final editorial stage produces one cohesive summary | `summarizer.editorial`, `summarizer.finalization`; `tests/test_pipeline.py`; coherence review. |
| 6 | Optional verification identifies and repairs/removes unsupported claims | `summarizer.verification`; `tests/test_verification_*`; CLI `--verify`; evidence remains best effort, not a truth guarantee. |
| 7 | Interrupted work can be cached and resumed safely | `summarizer.cache`, `summarizer.checkpoint`, `summarizer.reliability`; `tests/test_cache.py`, `tests/test_checkpoint.py`, `tests/test_pipeline_reliability.py`, publication tests. |
| 8 | Default `input.txt` to `output.txt` workflow still works | `main.py`, `summarizer.cli`, `tests/test_cli.py`; no-argument example in README. |
| 9 | Same CLI/pipeline supports OpenAI and local Ollama without provider orchestration forks | `summarizer.providers.openai`, `summarizer.providers.ollama`, shared `ModelProvider`; `tests/providers/`; CLI provider selection tests. |
| 10 | Automated tests pass without external API access | `tests/support/network_guard.py`, `tests/test_offline_guard.py`, deterministic fakes, full-suite command in README; execution result must come from centralized verification. |
| 11 | Representative fixtures evaluated for coherence, coverage, and faithfulness | `tests/support/evaluation.py`, `tests/test_evaluation.py`, this five-dimension rubric; generated `evaluation.json` is the run evidence. |
| 12 | README accurately documents final behavior | README sections for CLI, flags/defaults, providers, audit, reliability, verification, migration, tests, and limitations. |
| 13 | Historical repository data remains intact | `omscs-ml-lectures/` unchanged; fixture test and repository diff inspection. |
| 14 | No credentials or generated caches are committed | `.gitignore`, secret-safe audit/cache projections, and final repository status review. |

## Issue #12 acceptance-criteria map

The nine issue-specific criteria are mapped separately so a reviewer can distinguish this integration gate from the broader parent definition of done.

| # | Issue #12 acceptance criterion | Evidence / disposition |
| ---: | --- | --- |
| 1 | Run deterministic end-to-end suite across article, report, transcript, structured Markdown, and narrative fixtures without network | `tests.support.evaluation` command, `tests/test_evaluation.py`, `checks.network_blocked`; actual execution output supplied by Main. |
| 2 | Demonstrate direct for small inputs and genuine multi-level hierarchy for sufficiently large inputs | evaluator direct/auto cases; `root_level`, `strategy_reason`, and audit tree fields. |
| 3 | Evaluate coherence, salient coverage, provenance, qualifications/contradictions, and source faithfulness with repeatable rubric | Five rubric sections above and explicit JSON rubric fields. |
| 4 | Resolve or document every regression against accepted baseline and parent DoD | Baseline disposition matrix and parent evidence map above; intentional CLI/dry-run removals are explicit. |
| 5 | Verify `python main.py` reads `input.txt`, writes `output.txt`, documents credentials, and avoids misleading partial output | CLI implementation/tests, README migration and failure sections, atomic publication path. |
| 6 | Update README with purpose, inputs, installation, environment, CLI, strategies, output/audit, cache/resume, verification limits, migration, and tests | README section inventory; no `.env` auto-loading is claimed. |
| 7 | Confirm automated tests pass without real API calls or external network | network guard, deterministic fakes, test commands; pass result is intentionally left to centralized verification. |
| 8 | Confirm historical input data is intact and no credentials, caches, temporary files, or generated evaluation artifacts are committed | fixture/history disposition, cache/secret rules, final repository status. |
| 9 | Record evidence against every parent definition-of-done item | 14-row parent map above; review and centralized run remain required before closing #1. |

The tables make the evidence auditable without pretending that inspected code or a deterministic fake is a live-provider quality guarantee. Parent closure requires review of the centralized execution evidence and repository state.
