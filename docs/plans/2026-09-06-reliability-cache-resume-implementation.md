# Reliability, Cache, and Resume Implementation Plan

**Status:** Implemented and acceptance-tested.

This file preserves the TDD execution sequence. Its RED expectations describe
the pre-implementation state, not current behavior.

**Goal:** Add opt-in local cache/resume, bounded concurrency, typed retry,
`audit/3`, and recoverable publication to the library pipeline.

**Architecture:** `summarizer.cache` persists immutable descriptor-keyed JSON
objects only after existing parsers validate a stage result. `checkpoint` owns
local run manifests and locks; a coordinator composes cache, checkpoint, and a
bounded scheduler around current direct, hierarchy, editorial, and verification
seams. Audit-first and summary-last writes use a manifest completion witness,
not an impossible multi-path atomic rename.

**Tech Stack:** Python 3.10+, stdlib JSON/hashlib/pathlib/os/fsync/concurrent
futures, Pydantic, existing provider/token protocols, pytest temporary dirs and
deterministic fakes.

---

## Shared key and review rules

Canonical descriptors are compact sorted UTF-8 JSON. Keys hash the descriptor
before provider calls and enumerate source ID; canonical input hash; stage/work
ID; prompt/schema version; provider/model; counter identity/exactness; context
window; segmentation bounds/overlap; strategy/budget/target words; merge fanout
and grounding policy; editorial version; verification/repair configuration; and
all other output-affecting options. They exclude credentials, hosts/endpoints,
paths (including cache root), prompts, requests, raw source prose, and run IDs.
Cached source-derived content is `0600`; cache directories are `0700`.

Every task is TDD: one failing focused test, observe RED, smallest GREEN change,
focused regression, then a fresh Terra review of that exact staged diff for
bugs, simplification, and comment concision before its commit. Do not use Sol.
Stage only named files. Use
`uv run --with-requirements requirements.txt --with pytest python -m pytest`.
Do not push, edit a PR, or alter the legacy CLI without explicit authorization.

### Task 1: Descriptor models and sharded object store

**Files:**

- Create: `summarizer/cache.py`
- Create: `tests/test_cache.py`
- Modify: `summarizer/config.py`
- Modify: `tests/test_config.py`
- Modify: `.gitignore`

**Step 1: Write failing tests.** Add exact-key tests for every shared input and
every excluded unsafe field; sharding, immutable first-writer behavior,
`0600`/`0700`, validate-before-write, `fsync` path, and typed missing/corrupt/
wrong-version/incompatible misses.

**Step 2: Verify RED.**

```bash
uv run --with-requirements requirements.txt --with pytest python -m pytest -q tests/test_cache.py
```

Expected: `summarizer.cache` and `CacheConfig` do not exist.

**Step 3: Implement minimum code.** Add frozen opt-in
`CacheConfig(enabled=False, root=Path(".summarizer-cache"))`; add
`.summarizer-cache/` to `.gitignore`. Implement canonical descriptor/envelope
models, descriptor SHA-256, `objects/<prefix>/<key>.json`, narrow per-key local
lock, temp write/flush/`fsync`/atomic replace/parent `fsync`, and typed misses.

**Step 4: Verify GREEN.**

```bash
uv run --with-requirements requirements.txt --with pytest python -m pytest -q tests/test_cache.py tests/test_config.py
```

**Step 5: Terra review and commit.**

```bash
git add .gitignore summarizer/cache.py summarizer/config.py tests/test_cache.py tests/test_config.py
git commit -m "feat(cache): store validated stage objects"
```

### Task 2: Run manifests, local locks, and compatible resume

**Files:**

- Create: `summarizer/checkpoint.py`
- Create: `tests/test_checkpoint.py`
- Modify: `summarizer/config.py`
- Modify: `tests/test_config.py`

**Step 1: Write failing tests.** Cover a versioned `runs/<run-id>.json` with
run descriptor/source hash/ordered work IDs/completed refs/safe metadata/
publication state; one active local holder; atomic restrictive checkpoints; and
reuse only of descriptor-compatible, validated refs.

**Step 2: Verify RED.**

```bash
uv run --with-requirements requirements.txt --with pytest python -m pytest -q tests/test_checkpoint.py
```

Expected: `summarizer.checkpoint` import fails.

**Step 3: Implement minimum code.** Add `ReliabilityConfig(max_in_flight=1)`
and explicit run/resume choice; versioned Pydantic manifest, local advisory
`runs/<run-id>.lock`, validate+temp+fsync+replace, stable references, and
closed incompatible/corrupt non-reusable reasons. No distributed or stale-lock
coordination.

**Step 4: Verify GREEN.**

```bash
uv run --with-requirements requirements.txt --with pytest python -m pytest -q tests/test_cache.py tests/test_checkpoint.py tests/test_config.py
```

**Step 5: Terra review and commit.**

```bash
git add summarizer/checkpoint.py summarizer/config.py tests/test_checkpoint.py tests/test_config.py
git commit -m "feat(cache): checkpoint compatible runs"
```

### Task 3: Evolve typed retry with bounded injectable jitter

**Files:**

- Modify: `summarizer/config.py`
- Modify: `summarizer/providers/base.py`
- Modify: `summarizer/providers/retrying.py`
- Modify: `tests/test_config.py`
- Modify: `tests/providers/test_retrying.py`

**Step 1: Write failing tests.** Prove bounded exponential jitter through
injected sleeper/clock/RNG, safe attempt records, exhaustion, and immediate
authentication/request/response/config failures. Pin existing `RetryPolicy()`
count and deterministic no-jitter compatibility.

**Step 2: Verify RED.**

```bash
uv run --with-requirements requirements.txt --with pytest python -m pytest -q tests/providers/test_retrying.py tests/test_config.py
```

Expected: policy has no bounds/jitter and attempts are unobservable.

**Step 3: Implement minimum code.** Add finite delay/jitter policy fields with
compatibility defaults, a closed attempt record, injected time/RNG dependencies,
and catch only `TransientProviderError`. Retain no provider message, request,
host, or credential in metadata.

**Step 4: Verify GREEN.**

```bash
uv run --with-requirements requirements.txt --with pytest python -m pytest -q tests/providers/test_retrying.py tests/providers/test_base.py tests/test_config.py
```

**Step 5: Terra review and commit.**

```bash
git add summarizer/config.py summarizer/providers/base.py summarizer/providers/retrying.py tests/test_config.py tests/providers/test_retrying.py
git commit -m "feat(retry): record bounded transient attempts"
```

### Task 4: Deterministic scheduler and exact failure latch

**Files:**

- Create: `summarizer/scheduler.py`
- Create: `tests/test_scheduler.py`
- Modify: `summarizer/checkpoint.py`
- Modify: `tests/test_checkpoint.py`

**Step 1: Write failing tests.** Pin max in-flight, stable result ordering,
level barriers, and no submissions after the first error. Add the required two
in-flight leaf test: one fails while a sibling succeeds during drain; that
sibling is validated/stored/checkpointed by stable work ID and resume does not
call it again. Cover cancelled/unobservable/unknown non-reusable records.

**Step 2: Verify RED.**

```bash
uv run --with-requirements requirements.txt --with pytest python -m pytest -q tests/test_scheduler.py tests/test_checkpoint.py
```

Expected: `summarizer.scheduler` import fails.

**Step 3: Implement minimum code.** Use a bounded local executor. Latch first
failure, stop submission, drain every observable future, invoke validator/store
for successful drain results, sort durable refs by work ID, record other items
non-reusable, checkpoint all state, then raise terminal failure.

**Step 4: Verify GREEN.**

```bash
uv run --with-requirements requirements.txt --with pytest python -m pytest -q tests/test_scheduler.py tests/test_checkpoint.py tests/test_cache.py
```

**Step 5: Terra review and commit.**

```bash
git add summarizer/scheduler.py summarizer/checkpoint.py tests/test_scheduler.py tests/test_checkpoint.py
git commit -m "feat(pipeline): checkpoint drained parallel work"
```

### Task 5: Reuse only parsed and validated pipeline intermediates

**Files:**

- Modify: `summarizer/direct.py`
- Modify: `summarizer/leaf.py`
- Modify: `summarizer/hierarchy.py`
- Modify: `summarizer/editorial.py`
- Modify: `summarizer/segmentation.py`
- Modify: `summarizer/verification.py`
- Modify: `summarizer/pipeline.py`
- Create: `tests/test_pipeline_reliability.py`
- Modify: `tests/test_direct.py`
- Modify: `tests/test_leaf_stage.py`
- Modify: `tests/test_hierarchy.py`
- Modify: `tests/test_editorial.py`
- Modify: `tests/test_segmentation.py`
- Modify: `tests/test_verification_integration.py`

**Step 1: Write failing tests.** Use counting providers to show exact compatible
direct reuse; ordered leaves and same-level merges reuse; editorial and only
frozen independent verification batches reuse; every shared key input
invalidates; malformed raw provider data is never cached and is called again.
On the hierarchical path, use an injectable counting segmentation seam to prove
that the first compatible run stores a validated segmentation object and
manifest ref before leaf work, while a resumed run does not call
`segment_document` and retains downstream leaf order. Corrupt or invalid
segmentation payloads are explained misses that recompute segmentation.
Add a terminal-verification regression with schema-valid responses that reduce
to a material contradiction under `max_repair_passes=0`: its
`VerificationResult.failed` is true, no cache object or manifest reference is
written, and a resumed run invokes the verifier provider again.

**Step 2: Verify RED.**

```bash
uv run --with-requirements requirements.txt --with pytest python -m pytest -q tests/test_pipeline_reliability.py
```

Expected: pipeline has no coordinator or validated-stage cache seam.

**Step 3: Implement minimum code.** Thread an optional coordinator through
direct/leaf/hierarchy/editorial/verification. Build descriptor before call; on
hit deserialize+validate typed record; on miss call provider, run existing
parser/validator, then store only a terminal-success domain result:

- hierarchical segmentation, before any leaf call: a list of `SourceSegment`
  values keyed by canonical source hash, segmentation schema/version, complete
  `SegmentationConfig`, and tokenizer/counter-relevant behavior identity. On
  every hit, validate stable IDs and order, source identity, contiguous core
  coverage, offsets/context bounds, and core/full token invariants before use;
  do not include source text, filesystem paths, or secrets in the descriptor;
- direct and leaf: a returned `SummaryNode` after existing parser and
  provenance validation;
- merge: a returned merged `SummaryNode` after parse/provenance validation;
- editorial: a returned nonblank `EditorialResult` after `FinalDraft` parsing;
- verification: only `VerificationResult` where `failed is False`.

Exceptions, raw malformed provider responses, semantic terminal failures,
failure sentinels, and `VerificationResult.failed is True` are never stored or
referenced by a manifest. Schedule only leaves, preformed same-level merge
groups, and frozen independent verification batches; preserve escalation,
repair, and re-verification barriers.

**Step 4: Verify GREEN.**

```bash
uv run --with-requirements requirements.txt --with pytest python -m pytest -q tests/test_pipeline_reliability.py tests/test_direct.py tests/test_leaf_stage.py tests/test_hierarchy.py tests/test_editorial.py tests/test_segmentation.py tests/test_verification_integration.py
```

**Step 5: Terra review and commit.**

```bash
git add summarizer/direct.py summarizer/leaf.py summarizer/hierarchy.py summarizer/editorial.py summarizer/segmentation.py summarizer/verification.py summarizer/pipeline.py tests/test_pipeline_reliability.py tests/test_direct.py tests/test_leaf_stage.py tests/test_hierarchy.py tests/test_editorial.py tests/test_segmentation.py tests/test_verification_integration.py
git commit -m "feat(pipeline): reuse validated stage results"
```

### Task 6: Version-discriminated `audit/3` reliability metadata

**Files:**

- Modify: `summarizer/audit.py`
- Modify: `summarizer/finalization.py`
- Modify: `tests/test_audit.py`
- Modify: `tests/test_verification_audit.py`
- Create: `tests/test_audit_reliability.py`

**Step 1: Write failing tests.** Pin read/validation of `audit/2` fixtures.
Add `audit/3` cases for closed cache hit/miss/invalidation codes, resume state,
and retry attempts; reject unknown/cross-version fields and all paths, roots,
source/generated prose, prompts, requests, hosts/endpoints, and secrets.

**Step 2: Verify RED.**

```bash
uv run --with-requirements requirements.txt --with pytest python -m pytest -q tests/test_audit_reliability.py tests/test_audit.py tests/test_verification_audit.py
```

Expected: no `audit/3` discriminated model exists.

**Step 3: Implement minimum code.** Leave strict `audit/2` unchanged. Add an
`audit/3` model and `schema_version`-discriminated read/serialize union; new
reliability-enabled artifacts materialize as `audit/3` and project only safe
closed reliability metadata.

**Step 4: Verify GREEN.**

```bash
uv run --with-requirements requirements.txt --with pytest python -m pytest -q tests/test_audit.py tests/test_verification_audit.py tests/test_audit_reliability.py
```

**Step 5: Terra review and commit.**

```bash
git add summarizer/audit.py summarizer/finalization.py tests/test_audit.py tests/test_verification_audit.py tests/test_audit_reliability.py
git commit -m "feat(audit): record reliability metadata in v3"
```

### Task 7: Audit-first publication, summary-last witness, and resume

**Files:**

- Modify: `summarizer/finalization.py`
- Modify: `summarizer/pipeline.py`
- Modify: `summarizer/checkpoint.py`
- Create: `tests/test_publication.py`
- Modify: `tests/test_pipeline_reliability.py`

**Step 1: Write failing tests.** Inject audit-write, summary-replace, and
marker-write failures. Assert no summary after audit failure; incomplete
manifest after summary failure; digest-checked recovery after marker failure;
and reader rejection without matching completion marker. Add interrupted
end-to-end resume using Task 4's late-sibling checkpoint.

**Step 2: Verify RED.**

```bash
uv run --with-requirements requirements.txt --with pytest python -m pytest -q tests/test_publication.py tests/test_pipeline_reliability.py
```

Expected: no completion-witness publication protocol exists.

**Step 3: Implement minimum code.** Build/validate `audit/3` and summary,
atomically write audit, checkpoint `audit_staged`, atomically replace summary
last, then checkpoint `complete` with both digests. Resume verifies digests
before marking complete or republishing. Preserve claim-verification's existing
fail-closed no-summary/no-citation terminal path.

**Step 4: Verify GREEN.**

```bash
uv run --with-requirements requirements.txt --with pytest python -m pytest -q tests/test_publication.py tests/test_pipeline_reliability.py tests/test_pipeline.py tests/test_verification_integration.py
```

**Step 5: Terra review and commit.**

```bash
git add summarizer/finalization.py summarizer/pipeline.py summarizer/checkpoint.py tests/test_publication.py tests/test_pipeline_reliability.py
git commit -m "feat(pipeline): publish resumable final output"
```

### Task 8: Documentation and acceptance evidence

**Files:**

- Modify: `README.md`
- Modify: `docs/plans/2026-09-06-reliability-cache-resume-design.md`
- Modify: `docs/plans/2026-09-06-reliability-cache-resume-implementation.md`

**Step 1: Write the acceptance matrix.** Map retry, cache, resume, bounded
concurrency, atomic-output witness, and `audit/3` metadata to exact tests.
Explain opt-in `.summarizer-cache/`, sensitive `0600` persistence, descriptor
exclusions, local-only locking, `audit/2` compatibility, no cross-machine
guarantee, and Issue #12 CLI ownership.

**Step 2: Verify documentation evidence.**

```bash
uv run --with-requirements requirements.txt python -m compileall -q summarizer tests
uv run --with-requirements requirements.txt --with pytest python -m pytest -q
uv run --with-requirements requirements.txt --with pytest python -m pytest -q --import-mode=importlib
git diff --check
```

**Step 3: Terra review and commit.**

```bash
git add README.md docs/plans/2026-09-06-reliability-cache-resume-design.md docs/plans/2026-09-06-reliability-cache-resume-implementation.md
git commit -m "chore: document reliable pipeline behavior"
```

### Acceptance evidence

| Contract | Executable evidence |
| --- | --- |
| Canonical descriptors, JSON sharding, restrictive modes, immutable writes, and typed safe misses | `tests/test_cache.py` |
| Compatible manifests, stable work prefixes, run locking, and referenced-only resume | `tests/test_checkpoint.py` |
| Transient-only bounded retry, deterministic jitter, successful/exhausted audit counts, stable `V01` verification aggregation, and immediate terminal errors | `tests/providers/test_retrying.py`, `tests/providers/test_base.py`, `tests/test_pipeline_reliability.py` |
| Bounded leaf/merge work, stable result order, drained-success checkpoints, and scheduler failure latching | `tests/test_scheduler.py`, `tests/test_pipeline_reliability.py` |
| Validated segmentation/direct/leaf/merge/editorial reuse, new-run adoption, resume call counts, and failed-verification exclusion | `tests/test_pipeline_reliability.py`, `tests/test_verification_integration.py` |
| Strict audit/2 compatibility, version-discriminated audit/3, and manifest-ordered closed reliability metadata | `tests/test_audit.py`, `tests/test_audit_reliability.py`, `tests/test_verification_audit.py`, `tests/test_pipeline_reliability.py` |
| Audit-first and summary-last publication, digest witness recovery, reader rejection, and same-process output-pair serialization | `tests/test_publication.py`, `tests/test_pipeline_reliability.py` |

The implementation retains the scope boundaries in the design: JSON rather
than SQL, local filesystem storage, no eviction or encryption, no distributed
coordination, and no cache/resume CLI flags. Publication locking for a shared
summary/audit path pair is process-local; the manifest remains the durable
digest witness and recovery mechanism.

## Final verification

After the individually reviewed task commits:

```bash
uv run --with-requirements requirements.txt python -m compileall -q summarizer tests
uv run --with-requirements requirements.txt --with pytest python -m pytest -q
uv run --with-requirements requirements.txt --with pytest python -m pytest -q --import-mode=importlib
git diff --check
git status --short
```

Record exact results in the Issue #11 matrix. Keep all remote mutation behind
explicit authorization; Issue #12 owns CLI exposure and its migration docs.
