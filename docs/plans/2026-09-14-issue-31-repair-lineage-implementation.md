# Issue #31 Committed Repair Lineage Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Continue multi-pass verification from the repaired draft and ensure result/audit repair records describe only the repair lineage of the returned text.

**Architecture:** Keep `VerificationResult.repairs` as the single repair-record contract. A successful recursive continuation appends its committed events; a failed continuation or failed post-repair verification returns an earlier draft and drops events that belong only to the rejected candidate. `AuditArtifact` continues to project that tuple without a schema change.

**Tech Stack:** Python 3.10+, dataclasses, Pydantic audit models, pytest, `uv`.

---

### Task 1: Encode the returned-text repair contract in verification tests

**Files:**
- Modify: `tests/test_verification_repair.py:392-504`
- Test: `tests/test_verification_repair.py`

**Step 1: Write the failing independent-claim regression**

Replace the single-span `42 → 41 → 40` multi-pass fixture with a scripted
two-sentence draft. Pass 1 must repair only claim A; pass 2 must verify that
repaired text and identify claim B; pass 3 must repair B; pass 4 must verify
the combined result. Assert the final text contains both fixes and the event
span IDs are `V01S000001` and `V03S000002`.

Use complete decomposition responses for every locally split span, and hash the
V03 second-span replacement against the text produced by pass 1. This must fail
against `174f54b`, where the continuation receives the original draft.

**Step 2: Run the new regression to verify it fails**

Run:

```bash
uv run --with-requirements requirements.txt --with pytest python -m pytest -q \
  tests/test_verification_repair.py::test_verify_and_repair_keeps_independent_repairs_in_returned_text
```

Expected: FAIL on current `main` before the implementation change because the
second repair is evaluated against the original draft and is rejected or loses
the first fix.

**Step 3: Strengthen rejected-branch tests**

In `test_verify_and_repair_closes_post_repair_provider_failure_without_retrying`,
add `assert result.repairs == ()`, because the function returns `draft`.

Add a separate scripted recursive-continuation failure test:

```python
assert result.failed
assert result.text == outer_repaired
assert [event.span_id for event in result.repairs] == ["V01S000001"]
```

The provider should apply an outer repair, enter a nested repair attempt, and
fail during the nested re-verification. That isolates events from the discarded
nested candidate.

**Step 4: Run the focused tests to verify current failures**

Run:

```bash
uv run --with-requirements requirements.txt --with pytest python -m pytest -q \
  tests/test_verification_repair.py
```

Expected: the new independent-claim and repair-event assertions fail before the
logic change; existing unrelated repair tests remain green.

**Step 5: Commit the tests**

```bash
git add tests/test_verification_repair.py
git commit -m "test: cover committed repair lineage"
```

### Task 2: Make recursive repair events follow the returned text

**Files:**
- Modify: `summarizer/verification.py:1844-1958`
- Test: `tests/test_verification_repair.py`

**Step 1: Implement the minimal branch-local changes**

1. Recurse with `repaired`, not `draft`.
2. When the immediate re-verification fails and the function returns `draft`,
   pass `repairs=()` to `_terminal_result`.
3. When the repair budget is exhausted and the function returns `draft`, pass
   `repairs=()`.
4. When a recursive continuation fails, return the outer `repaired` text and
   only the outer `events`; discard `continued.repairs`.
5. When a recursive continuation succeeds, retain `(*events,
   *continued.repairs)` because both sets are on the returned lineage.

Do not change `RepairEvent`, the audit schema, prompt contracts, or redaction
behavior.

**Step 2: Run the focused test module**

Run:

```bash
uv run --with-requirements requirements.txt --with pytest python -m pytest -q \
  tests/test_verification_repair.py
```

Expected: PASS.

**Step 3: Commit the implementation**

```bash
git add summarizer/verification.py tests/test_verification_repair.py
git commit -m "fix: retain committed multi-pass repair lineage"
```

### Task 3: State and verify the audit projection semantics

**Files:**
- Modify: `docs/plans/2026-09-03-claim-verification-design.md:73-78`
- Modify: `tests/test_verification_audit.py:153-200`
- Test: `tests/test_verification_audit.py`

**Step 1: Align the enduring verification design document**

Extend item 9 to say that every additional pass verifies the prior repaired
draft, and that `repairs` records operations in the returned draft's committed
lineage—not attempted repairs from a discarded branch or a textual diff.

**Step 2: Add an audit projection assertion**

Build an artifact from a `VerificationResult` with an empty repair tuple and
assert serialized `verification.repairs` is `[]`. Pair it with the
verification-level rejected-branch test so the tests prove both sides of the
contract: execution clears discarded events, and audit faithfully projects the
cleared tuple.

**Step 3: Run the focused contract tests**

Run:

```bash
uv run --with-requirements requirements.txt --with pytest python -m pytest -q \
  tests/test_verification_repair.py tests/test_verification_audit.py
```

Expected: PASS.

**Step 4: Commit the contract documentation and audit test**

```bash
git add docs/plans/2026-09-03-claim-verification-design.md tests/test_verification_audit.py
git commit -m "docs: clarify verification repair lineage"
```

### Task 4: Verify the regression and branch quality

**Files:**
- Verify only: `summarizer/verification.py`
- Verify only: `tests/test_verification_repair.py`

**Step 1: Confirm the regression rejects the historical implementation**

Export `174f54b` to a temporary directory, copy the new focused regression
there without modifying the repository, and run it with the project test
command. Record the expected failure caused by recursion from the original
draft.

**Step 2: Run the full suite on the implementation branch**

Run:

```bash
uv run --with-requirements requirements.txt --with pytest python -m pytest -q
```

Expected: PASS with no collection, audit-schema, or verification regressions.

**Step 3: Inspect the final diff**

Run:

```bash
git diff --check origin/main...HEAD
git status --short
git log --oneline origin/main..HEAD
```

Expected: a focused documentation, tests, and repair-control-flow diff; no
unrelated files or generated artifacts.
