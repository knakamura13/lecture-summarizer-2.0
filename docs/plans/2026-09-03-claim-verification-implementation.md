# Claim Verification and Bounded Repair Implementation Plan

> **Status:** Tasks 1–6 are complete and committed. Task 7 records the final
> local documentation and acceptance evidence; issue #10 remains open until
> its pull request is merged.

**Goal:** Add optional, source-grounded claim verification and finite repair between editorial drafting and final audit materialization.

**Architecture:** A new `summarizer.verification` module owns immutable draft spans, strict structured claim and verdict parsing, deterministic evidence ranking, bounded verification batches, and repair orchestration. `finalize_summary` invokes it only when enabled, while `audit/2` stores linkable verification metadata without durable source-derived prose.

**Tech Stack:** Python 3, frozen dataclasses, Pydantic, NLTK Punkt spans, existing provider and token-counter protocols, pytest

---

## Completed implementation record

| Task | Delivered commit(s) |
| --- | --- |
| 1. Local spans and strict parsing | `22d894a`, `0863d24`, `613b10b` |
| 2. Deterministic bounded evidence | `869803c` |
| 3. Bounded decomposition and classification | `4e040de` |
| 4. Finite repair and reverification | `2387c01` |
| 5. `audit/2` verification metadata | `6854e0a` |
| 6. Finalization and pipeline integration | `ee7c361` |

The final implementation retains claim, source, draft, replacement, and quote
prose only in transient verification work. `audit/2` projects identifiers,
hashes, closed codes, safe runtime metadata, and phase usage instead. The
library path is opt-in; issue #12 owns CLI exposure.

### Task 1: Model local spans, claims, and strict structured responses

**Files:**
- Create: `summarizer/verification.py`
- Create: `tests/test_verification_models.py`
- Create: `tests/test_verification_parsing.py`

**Step 1: Write failing span and model tests**

Test deterministic sentence spans with pass-scoped `V01S000001` identifiers,
exact offsets, hashes, Unicode, repeated sentences, and instruction-like draft
text. Test frozen `VerificationConfig`, `VerificationRuntime`, `Claim`,
`ClaimVerdict`, `BatchFinding`, `ClaimAssessment`, `EvidenceSelection`, `RepairEvent`,
and `VerificationResult` invariants.

`VerificationConfig` defaults to disabled, one repair pass, positive evidence
and request budgets, an output reserve, and a safety margin. Reject nonpositive
budgets and negative limits. `VerificationRuntime` owns the context window.

**Step 2: Run the focused tests and verify RED**

Run:

```bash
uv run --with-requirements requirements.txt --with pytest python -m pytest -q tests/test_verification_models.py
```

Expected: import failure because `summarizer.verification` does not exist.

**Step 3: Implement local spans and immutable domain records**

Use a resource-free `PunktSentenceTokenizer().span_tokenize(text)`. Attach
inter-sentence whitespace to the preceding span, retain exact draft slices, and
hash each span from its UTF-8 bytes. Keep source-derived text only in transient
runtime records, never audit records.

**Step 4: Write failing decomposition and verdict parser tests**

Test strict single-object Pydantic responses, stable locally assigned
`V01C000001` claim IDs, known span IDs, ordinals, all four verdicts, exactly one
result per claim, duplicate/missing claims, unknown evidence IDs, unselected
evidence, unsupported verdicts without evidence, exact quote substring
validation, extra fields, multiple JSON objects, and bounded redacted errors.
Atomic claim anchors must be exact span substrings and may overlap. Each is
classified in its full sentence context; no model-authored paraphrase becomes a
claim. Local code reuses an exact full-span atomic anchor as the fallback or
creates one when absent. Every claim is material and classified; the model
cannot choose coverage, materiality, or a non-verifiable decomposition kind.

**Step 5: Implement request schemas and parsers**

Reuse the existing `_extract_json_object`, `_describe`, and `_sanitize` error
helpers. Model output supplies exact anchors but cannot choose claim prose,
local offsets, materiality, final claim IDs, or fallback status. Assign
pass-scoped identifiers locally in span and ordinal order, reuse or append the
whole-span fallback, and validate every model-supplied identifier against
explicit legal sets. Test fallback deduplication.

**Step 6: Run focused and full tests**

Run:

```bash
uv run --with-requirements requirements.txt --with pytest python -m pytest -q tests/test_verification_models.py tests/test_verification_parsing.py
uv run --with-requirements requirements.txt --with pytest python -m pytest -q
```

Expected: all tests pass.

**Step 7: Obtain a fresh-eyes review and commit**

Stage only Task 1 files. Ask a read-only 5.6 Terra reviewer to check invariant
gaps, parser trust boundaries, simplification, and comment length. Address valid
findings and rerun tests before committing:

```bash
git commit -m "feat(verification): model claims and strict verdicts"
```

### Task 2: Select and batch authoritative evidence deterministically

**Files:**
- Modify: `summarizer/verification.py`
- Create: `tests/test_verification_evidence.py`

**Step 1: Write failing evidence-selection tests**

Use canonical segment cores and deterministic counters. Cover all legal
candidates from root provenance, a precomputed lexical index, case-folded
overlap, punctuation and Unicode, stable source-order tie breaking,
deduplication, complete-passage packing, selected, examined, and
omitted identifiers, retrieval completeness, exact token cost, a passage too
large to fit, unknown provenance, and repeated text at distinct segment IDs.

**Step 2: Verify RED**

Run the focused file and confirm failure because claim evidence selection is
missing.

**Step 3: Implement claim-specific retrieval**

Resolve root provenance through a local segment map and rank every legal segment
from one precomputed lexical index by overlap descending and source order
ascending. Serialize complete `SourcePassage` records with the existing compact
serializer and recount each candidate selection under the configured evidence
budget. Record whether all legal provenance was examined. Never accept source
text or candidate identifiers from model output.

**Step 4: Write failing bounded-batch tests**

Cover stable work-item order, passage deduplication within a batch, complete
input cost including instructions, schema, and data, exact-fit capacity with
output reserve and safety margin, oversized indivisible items, and multiple
batches without item loss or duplication. Exercise the generic packer with
decomposition, classification, and repair-shaped requests.

**Step 5: Implement deterministic budget packing**

Pack complete work items greedily while
`measured_input + output_reserve + safety_margin <= context_window` and measured
input remains under the configured request budget. Fail before any provider
call when one work item cannot fit. Recount the complete serialized candidate
because tokenization is not additive.

**Step 6: Run focused and full tests, review, and commit**

Use a read-only Terra reviewer because retrieval and packing affect both
faithfulness and long-artifact performance. Commit only after findings are
addressed:

```bash
git commit -m "feat(verification): select and batch source evidence"
```

### Task 3: Decompose and classify claims in bounded calls

**Files:**
- Modify: `summarizer/verification.py`
- Create: `tests/test_verification_requests.py`
- Create: `tests/test_verification_stage.py`

**Step 1: Write failing request tests**

Assert deterministic source-specific fences, genre-neutral instructions,
source/draft content treated as inert data, no outside knowledge, strict schemas,
stable operation IDs, configured model and timeout, and absence of provider
credentials or endpoint data.

**Step 2: Verify RED, then implement request builders**

Create versioned decomposition and classification prompts. Build decomposition
batches from complete spans under the capacity invariant and require exact
anchors for atomic claims. Append a whole-span fallback claim locally.
Classification requests contain only the claims in that batch and locally
selected authoritative passages. They state that insufficient evidence is
scoped to the supplied selection and that verdicts are assessments rather than
proof.

**Step 3: Write failing stage tests**

Use scripted providers for every verdict, overlapping atomic anchors,
whole-span fallback deduplication, multiple decomposition and classification
batches, oversized single spans and claims, malformed output, unknown IDs,
missing claims, contradictory verdict metadata, usage collection, deterministic
reruns, and disabled zero-call behavior.

**Step 4: Implement one verification pass**

Add an internal pass function that splits the draft, decomposes claims, selects
evidence, builds batches, and classifies every claim. Retain each batch finding
and reduce them deterministically: conflicting support and contradiction becomes
insufficient with a closed conflict limitation; contradiction requires complete
examination with no support or inconsistent meaningfulness; support requires no
contradiction; non-verifiable requires unanimous findings; all other
combinations are insufficient. Return reduced assessments, selections,
generations, warnings, and limitations. Do not repair or loop in this task.

**Step 5: Run focused and full tests, review, and commit**

Use a read-only Terra reviewer for prompt trust boundaries, batching, failure
semantics, simplification, and comment length:

```bash
git commit -m "feat(verification): classify claims against bounded evidence"
```

### Task 4: Add finite span repair and complete reverification

**Files:**
- Modify: `summarizer/verification.py`
- Create: `tests/test_verification_repair.py`

**Step 1: Write failing repair-model and application tests**

Cover whole-span replacement, qualification, removal, original-hash matching,
unknown spans, duplicate repairs, stable application order, supported sibling
claims preserved in request data, locally stored sibling anchors retained in the
replacement, missing anchors, and replacement text redaction.

**Step 2: Verify RED, then implement repair request and local application**

An initial contradiction triggers bounded classification batches over omitted
legal segments. Only claims reduced to contradicted after all legal provenance
is examined and corroborated by a contradicted whole-span fallback trigger
repair; otherwise record a closed limitation code. Build
complete, budgeted repair work items containing target spans, triggering claim
IDs, selected evidence, and all sibling claims from those spans. Require each
locally stored exact anchor of a supported sibling to occur in the replacement.
Apply replacements locally only when the current span hash matches.

**Step 3: Write failing orchestration tests**

Cover disabled behavior, no-repair supported results, one successful repair,
multiple repair batches, an oversized repair item, contradiction escalation
across omitted legal evidence, mixed supported/contradicted batches, inconsistent
meaningfulness, mixed atomic/fallback verdicts, full repaired-draft decomposition
and reverification, supported-sibling deletion, a new unsupported repair claim,
failed repair, incomplete retrieval, unresolved material contradictions,
insufficient claims retained with limitations,
`max_repair_passes=0`, and exhaustion at one or more configured passes.

**Step 4: Implement `verify_and_repair`**

Run a complete verification pass, repair only when needed and permitted, then
repeat on the entire new draft. Reject and restore the prior draft if a
previously supported sibling disappears or any new or materially changed claim
is contradicted or insufficiently supported. Stop after the configured repair
count. Return a terminal failed result with complete metadata for remaining
material contradictions and expose exhaustion. Never let repair output serve as
its own verification.

**Step 5: Run focused and full tests, review, and commit**

Use a read-only Terra reviewer for termination, state replacement, source
grounding, signal preservation, complexity, and comment length:

```bash
git commit -m "feat(verification): add bounded repair and reverification"
```

### Task 5: Version audit artifacts with verification metadata

**Files:**
- Modify: `summarizer/audit.py`
- Modify: `tests/test_audit.py`
- Create: `tests/test_verification_audit.py`

**Step 1: Write failing `audit/2` schema tests**

Cover explicit disabled and completed verification records, pass-scoped
claim/span link resolution, known evidence segment IDs, pass ordering, repair
triggers, exhaustion, stable verifier metadata, retrieval omissions, closed
warning/failure/limitation codes, and rejection of duplicate, missing, or
unknown identifiers across or within passes.

Assert serialized output contains no claim, draft, replacement, source passage,
quote, free-form diagnostic, secret, prompt, request, endpoint, credential, or
path prose. Assert runtime configuration uses an explicit safe allowlist.

**Step 2: Verify RED, then implement versioned records**

Bump every newly written artifact to `audit/2`, including disabled verification
runs. Add verification, claim, selection, assessment, and repair records nested
by pass and containing only identifiers, hashes, enums, counts, actions,
allowlisted runtime identity, and closed diagnostic codes. Validate all links in
`AuditArtifact._links_resolve` within their owning pass.

**Step 3: Include verifier usage and preserve canonical output**

Feed all decomposition, classification, and repair `GenerationResult` values
into existing usage conversion. Replace broad application-config serialization
with an allowlist that excludes hosts, endpoints, paths, credentials, prompts,
and requests. Keep sorted compact JSON, validation before write, and atomic
replacement unchanged.

**Step 4: Run focused and full tests, review, and commit**

Use a read-only Terra reviewer for durable secret/source leakage, schema
compatibility, referential integrity, simplification, and comments:

```bash
git commit -m "feat(audit): record claim verification metadata"
```

### Task 6: Integrate optional verification into finalization and pipeline

**Files:**
- Modify: `summarizer/finalization.py`
- Modify: `summarizer/pipeline.py`
- Modify: `tests/test_pipeline.py`
- Create: `tests/test_verification_integration.py`

**Step 1: Write failing disabled-path tests**

Prove default pipeline output, provider call count, and citations remain
unchanged when verification is disabled. Prove newly written audits use
`audit/2` with an explicit disabled verification record.

**Step 2: Write failing enabled-path tests**

Cover direct and multi-level hierarchical flows, verification after editorial,
repaired final text before citations, all verifier generations in usage, audit
metadata, same default complete provider/counter/model/timeout/context runtime,
injected complete verifier runtime, runtime/budget mismatch, and fail-closed
unresolved contradictions. With an audit path, malformed output and exhausted
contradictions must write a validated failure audit before raising.

**Step 3: Implement the integration seam**

Add `verification: VerificationConfig` to `PipelineConfig` with a disabled
default. Extend `finalize_summary` with counter, root-provenance source cores,
verification config, and optional complete verifier runtime. Call
`verify_and_repair` immediately after `write_editorial`. Materialize the audit
from its success or terminal failure result, then raise for a failure before
writing or returning the reader-facing summary. Apply citations only to a
successful final text. Keep the transitional CLI unchanged for issue #12.

**Step 4: Run focused and full tests, review, and commit**

Use a read-only Terra reviewer for call ordering, default compatibility,
budget/runtime mismatches, error propagation, simplification, and comments:

```bash
git commit -m "feat(pipeline): verify final claims before output"
```

### Task 7: Document limitations and complete issue #10 evidence

**Files:**
- Modify: `README.md`
- Modify: `docs/plans/2026-09-03-claim-verification-design.md`
- Modify: `docs/plans/2026-09-03-claim-verification-implementation.md`

**Step 1: Document safe interpretation and migration boundary**

Explain opt-in behavior, same-runtime default, dedicated-runtime injection,
bounded retrieval, finite repair, audit/2, fail-closed contradictions, and why a
supported verdict is not factual proof. State that issue #12 owns CLI exposure.

**Step 2: Run static and complete verification**

Run:

```bash
uv run --with-requirements requirements.txt python -m compileall -q summarizer tests
uv run --with-requirements requirements.txt --with pytest python -m pytest -q
uv run --with-requirements requirements.txt --with pytest python -m pytest -q --import-mode=importlib
git diff --check
git status --short
```

Expected: both complete suites pass, compilation succeeds, and only intended
documentation remains unstaged.

**Step 3: Obtain final fresh-eyes review and commit**

Use a read-only 5.6 Terra reviewer for accuracy, concision, limitation language,
and stale commands. Address findings and rerun documentation tests:

```bash
git commit -m "chore: document claim verification and repair"
```

**Step 4: Update issue #10 acceptance evidence**

Post final test counts and map every acceptance criterion to implementation and
test paths. Do not close issue #10 until its PR is merged.

## Local acceptance evidence matrix

The following is local evidence for issue #10's acceptance criteria. It does
not claim a live-provider factual-quality result: the offline suite uses
deterministic fakes and verifies contracts, boundaries, and fail-closed paths.

| Issue #10 criterion | Implementation | Offline evidence |
| --- | --- | --- |
| Library opt-in preserves default readability; CLI remains #12 | `summarizer.pipeline.PipelineConfig`, `summarizer.finalization.finalize_summary` | `tests/test_pipeline.py`, `tests/test_verification_integration.py::test_pipeline_verification_is_explicitly_disabled_by_default` |
| Local claim decomposition, including non-verifiable statements | `summarizer.verification.split_draft_spans`, `parse_claim_anchors`, `verify_draft_once` | `tests/test_verification_models.py`, `tests/test_verification_parsing.py`, `tests/test_verification_stage.py` |
| Provenance-scoped deterministic bounded evidence | `build_source_lexical_index`, `select_claim_evidence` | `tests/test_verification_evidence.py` |
| Four validated verdict classes | `ClaimVerdict`, `parse_claim_findings`, `reduce_batch_findings` | `tests/test_verification_parsing.py`, `tests/test_verification_stage.py` |
| Bounded qualify/replace/remove repair without unsupported additions | `build_repair_request`, `apply_repairs`, `verify_and_repair` | `tests/test_verification_repair.py` |
| Full repaired-draft decomposition and reverification | `verify_and_repair` | `tests/test_verification_repair.py::test_verify_and_repair_reverifies_the_complete_repaired_draft` |
| Finite configurable repair and clear exhaustion | `VerificationConfig.max_repair_passes`, `verify_and_repair` | `tests/test_verification_repair.py`, `tests/test_verification_integration.py::test_exhausted_contradiction_writes_terminal_audit_before_raising` |
| Versioned no-prose audit record with findings, omissions, repairs, usage, and limits | `summarizer.audit.AuditArtifact`, `_audit_verification` | `tests/test_audit.py`, `tests/test_verification_audit.py` |
| Documentation/output never equates verifier approval with factual perfection | `README.md`, `docs/plans/2026-09-03-claim-verification-design.md` | Documentation review; `tests/test_verification_audit.py` verifies no reader/source prose enters audit output |
| Offline coverage of classifications, mixed evidence, malformed output, repair success/failure, stale spans, and exhaustion | `summarizer.verification` | `tests/test_verification_stage.py`, `tests/test_verification_repair.py`, `tests/test_verification_audit.py`, `tests/test_verification_integration.py` |

Latest local verification (2026-09-06): `compileall` succeeded; both complete
offline pytest modes passed with **509 passed, 2 skipped**. The final issue
comment should reproduce this matrix and those exact command results after the
documentation commit is reviewed.
