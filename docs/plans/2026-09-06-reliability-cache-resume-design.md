# Reliability, Cache, and Resume Design

## Scope

Issue #11 adds local reliability primitives around the library pipeline: a
content-addressed cache for validated successful intermediates, per-run resume
checkpoints, bounded parallel work, and typed retry observability. It does not
change the transitional CLI. Existing callers retain conservative behavior until
they explicitly supply reliability configuration.

This design uses JSON files, not SQLite. It targets one local filesystem and
makes no cross-machine portability guarantee.

## Defaults and non-goals

`CacheConfig` is opt-in at the library boundary and defaults its root to
`.summarizer-cache/` when enabled; callers may configure a root outside the
repository. `ReliabilityConfig.max_in_flight` defaults to `1`. Cache/resume are
not wired into the legacy CLI, so current callers retain provider-call behavior
until they explicitly opt in. A caller must also select or resume a run manifest
to continue an interrupted run.

This issue does not add a database, distributed cache or locks, eviction daemon,
encryption, a cross-machine cache guarantee, or a new provider timeout layer.
Provider request timeouts remain the existing `GenerationRequest` boundary.

## Storage model

The default cache root is `.summarizer-cache/`; a configured external root has
the same layout:

```text
<cache-root>/
  objects/<first-two-hex>/<sha256>.json
  runs/<run-id>.json
  runs/<run-id>.lock
```

An object is immutable and content addressed by its canonical input descriptor.
Its file name is the SHA-256 of that descriptor, which can be derived before a
provider call; the envelope separately carries the validated payload digest.
The shard is the first two hexadecimal key characters. Object envelopes carry
an object-format version, descriptor, payload kind, payload, and payload digest.
The descriptor is canonical JSON with sorted keys and UTF-8 encoding.

Every descriptor includes the source and input-content hashes; stage identity;
prompt and schema versions; provider and model identities; token-counter and
context identities where behavior depends on them; and every stage behavior
setting that can affect output (segmentation, strategy, verification, repair,
and relevant retry-independent generation settings). It excludes credentials,
hosts or endpoints, filesystem paths, request text, and operational run IDs.
Changing any included field produces a different key rather than an ambiguous
reuse.

Objects hold only successful, locally validated intermediates: segmentation
records; direct, leaf, and merge summaries; editorial drafts; and successful
terminal verification results. Their schemas are checked at both write and
read. Failed calls, raw provider responses, ambiguous or unsuccessful work
items, failed verification, and reader-facing final publication never become
cache objects. A successful sibling observed while a failed concurrent batch
drains is validated and stored independently.

Cached summaries and verification data can contain source-derived text. New
object and manifest files are created with mode `0600` (directories `0700`),
and the `.gitignore` entry `.summarizer-cache/` covers objects,
manifests, locks, and temporary files. Descriptor, manifest, and audit
projections exclude application credentials supplied outside the source, along
with cache roots, paths, hosts, endpoints, prompts, requests, and raw provider
errors. Cached source-derived payloads may still contain credential-like source
text. The cache is not encrypted and must be treated as sensitive local data.

## Validation and atomicity

`summarizer.cache` owns canonical encoding, descriptor hashing, schema
validation, and filesystem operations. A write creates a same-directory
temporary file with restrictive permissions, writes the complete validated
payload, flushes and `fsync`s the file, atomically replaces the destination,
then `fsync`s the containing directory where supported. A narrow local lock on
that object key protects the final existence check: the first valid writer uses
atomic replace, while a concurrent writer validates and reuses the existing
object instead of replacing it. A corrupt pre-existing file is replaced only
with a newly validated object under the same lock. This keeps valid objects
immutable and makes cross-run writes idempotent without a global or distributed
lock.

Reads recompute the descriptor and payload hashes and validate the envelope and
payload schema. A missing object is a normal miss. A malformed, unreadable,
hash-mismatched, wrong-version, or descriptor-incompatible object is explained
as a typed miss reason, never reused. Implementations may quarantine corrupt
files only after a separately approved retention policy; the initial behavior
does not delete them during a read.

## Runs and resume

`summarizer.checkpoint` owns a versioned per-run manifest. It records the
run descriptor, source hash, planned ordered work identifiers, completed object
references, retry/cache metadata, and publication state. It records no raw
source, prompt, credential, endpoint, or request data. Each checkpoint rewrite
uses the same validate, temporary-file, `fsync`, and atomic-replace discipline
as an object write.

A process holds an exclusive local advisory lock for its exact
`runs/<run-id>.lock` while it reads or advances that manifest. A second process
for the same run fails clearly with `run_active`; it must not duplicate in-flight
provider calls. This is intentionally a local-process coordination rule, not a
distributed lock.

Resume first validates that the manifest descriptor exactly matches the current
run descriptor. It then reuses only referenced objects that pass validation and
whose descriptors match the planned work. Missing, corrupt, or incompatible
entries become explicit misses and are recomputed. Thus an interrupted direct,
hierarchical, or verification run repeats no compatible completed provider call,
while a behavior change cannot silently consume stale output.

## Pipeline data flow and concurrency

`run_pipeline` remains the orchestration entry point. Reliability adds a
pipeline-local coordinator rather than teaching prompts or providers about
files:

1. Derive the run descriptor and open or create its manifest under the run lock.
2. Reuse or compute segmentation and persist only validated completed records.
3. Submit independent leaves with at most `max_in_flight` work items. Results are
   collected by their stable segment order before the next stage sees them.
4. At each hierarchy level, form the deterministic merge groups first, then run
   independent groups with the same cap and restore group order before building
   the next level.
5. Reuse or compute editorial and verification work. Verification, escalation,
   repair, and re-verification remain ordered barriers; only a terminal
   successful verification result is cache eligible.
6. Build and validate the audit artifact, then execute the publication protocol.

The scheduler uses a bounded local executor around the existing synchronous
provider protocol. It never changes request inputs, response schemas, or result
ordering. Its failure latch is exact: on the first failure it stops submitting
new work, drains every observable in-flight future, validates and stores each
successful drained result, then checkpoints all successful references in stable
work-ID order. It marks cancelled, unobservable, and unknown work explicitly
non-reusable; only after those records are durable does it persist the terminal
failure. Thus a sibling that succeeds while the failed batch drains is reusable
on resume, while no ambiguous in-flight work is reused. `max_in_flight=1` is the
conservative default.

## Retry boundary

`RetryingProvider` evolves rather than being replaced. It retries only
`TransientProviderError` subclasses (timeout, rate limit, connection, and
server errors). Configuration, authentication, request, and response errors
remain immediate terminal failures. Retry delay is bounded exponential backoff
with optional bounded jitter; the clock/sleeper and random source are injected
so tests can make timing deterministic. Compatibility defaults preserve the
current retry count and deterministic delay behavior unless jitter is explicitly
enabled.

Each failed attempt produces safe metadata: attempt number, closed error
category, planned delay, observation time, and exhaustion state. Provider
messages, request data, endpoint details, and application credentials are not
retained. Audit/3 projects successful and exhausted retry activity into attempt
counts and closed failure codes rather than copying raw attempt data. Cache and
retry entries follow manifest work order, independent of concurrent completion
order; all verification phases and passes aggregate under stable work ID `V01`.

## Audit version evolution

Issue #11 introduces `audit/3` for cache, resume, and retry metadata. It does
not mutate the strict `audit/2` contract: existing `audit/2` artifacts remain
readable and valid under their current model, while reliability-enabled
artifacts use `audit/3`. Serialization and reads select a version-discriminated
artifact model (or equivalent union) by `schema_version`; migration never
reinterprets an `audit/2` payload as `audit/3`.

`audit/3` adds only closed cache outcome codes, resume state, and retry attempt
metadata. It excludes cache roots and other paths, source or generated prose,
prompts, request data, hosts, endpoints, raw errors, and application secrets.
Issue #12's CLI migration documentation must describe the version choice and
continue to accept historical `audit/2` records.

## Final publication protocol

Two arbitrary output paths cannot be atomically committed together. The design
therefore does not claim cross-path atomicity. It provides an ordered,
recoverable protocol instead:

1. Build and validate the final audit artifact and final summary candidate.
2. Atomically stage/write the audit and record `publication: audit_staged` in
   the run manifest.
3. Atomically replace the readable summary last.
4. Atomically mark the manifest `publication: complete` after the summary
   replace succeeds.

If audit staging fails, no final summary is written. If summary replacement
fails, the manifest remains incomplete and callers must not treat a prior or
partial summary as this run's completion. If the summary write succeeds but the
completion marker does not, resume verifies the summary digest and completes the
marker or republishes safely. Protocol-aware readers accept a final summary only
when the matching manifest completion marker and digests agree. Each individual
file replacement is atomic; the manifest is the commit witness for the pair.

A process-local lock serializes callers using the same resolved summary/audit
path pair. The per-run manifest lock also prevents two local processes from
advancing the same run. Neither lock coordinates different run IDs publishing
to shared output paths across processes; callers must avoid that configuration.

## Failure semantics

Cache corruption and incompatibility are safe misses with closed reasons.
Scheduler-managed leaf and merge failures drain observable siblings, checkpoint
validated successes plus safe non-reusable state, and leave final publication
incomplete. Sequential direct, editorial, and verification computation failures
are not recorded in the manifest's scheduler failure fields; their unsuccessful
results remain uncached and publication remains incomplete. Existing
claim-verification terminal failures retain their fail-closed behavior: a
validated failure audit may be written, but no reader-facing final summary or
citations are published for that run.

## Testing and migration

Tests use temporary directories, deterministic providers, injected
clock/RNG/sleeper, and fault-injected filesystem calls. Coverage includes:

- canonical keys, sharding, permissions, validation before write, idempotent
  concurrent object writes, and corrupt/incompatible miss reasons;
- exact descriptor invalidation for source, prompt/schema, model, and behavior
  configuration changes, with credentials, hosts, and paths absent from keys;
- manifest locking, checkpoint recovery, and resume call counts for completed
  direct, leaf, merge, and eligible verification batches;
- maximum in-flight enforcement, stable request/result ordering, level barriers,
  and failure checkpoints under bounded concurrency, including two in-flight
  leaves where one fails and its sibling succeeds during drain; resume must not
  repeat that sibling call and must retain stable work ordering;
- transient-only retries, deterministic bounded jitter, immediate terminal
  errors, successful and exhausted attempt metadata, manifest-ordered audit
  projection, and stable `V01` verification aggregation;
- `audit/2` read/validation compatibility, `audit/3` version discrimination,
  and audit metadata redaction with closed cache/resume/retry codes; and
- audit-first, summary-last publication failures and completion-marker recovery.

Migration is additive. Existing callers keep cache/resume off and serial work by
default; existing `RetryPolicy` construction remains valid. Reliability is
available through `PipelineConfig.cache` and `PipelineConfig.reliability` at the
library boundary. Enabling the cache requires a run ID. Reliable paired
publication additionally requires `PipelineConfig.audit_path`; it writes the
summary to `AppConfig.output_path`. The legacy CLI does not expose these controls
and remains unchanged for Issue #12's migration.
