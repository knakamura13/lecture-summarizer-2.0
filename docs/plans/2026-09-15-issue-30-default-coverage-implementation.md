# Issue #30 Default-Configuration Coverage Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Cover realistic default pipeline behavior and missing finalization/audit invariants offline.

**Architecture:** Add deterministic test fixtures only. Preserve public production APIs and use real tokenizer coverage only when its local vocabulary is available.

**Tech Stack:** Python, pytest, tiktoken, deterministic fake providers.

---

### Task 1: Exercise default hierarchical capacity

**Files:**
- Modify: `tests/test_pipeline.py`

1. Write a failing real-tokenizer test that exceeds default usable capacity and retains all merge-request references.
2. Run it and verify it fails against the prior fixture behavior.
3. Add the minimum fake-provider behavior needed for valid grounded merge records.
4. Verify it passes or skips only for an unavailable tokenizer vocabulary.

### Task 2: Exercise overlap-derived capacity

**Files:**
- Modify: `tests/test_pipeline.py`

1. Write a failing overlap-enabled pipeline test.
2. Assert the result and source segments reflect capacity recomputed with overlap.
3. Run focused pipeline coverage.

### Task 3: Cover default finalization runtime resolution

**Files:**
- Create: `tests/test_finalization.py`

1. Write a failing direct-finalization test with enabled verification and no injected runtime.
2. Use a deterministic provider and counter to assert the resolved runtime performs verification.
3. Run the new file.

### Task 4: Cover audit-link rejections

**Files:**
- Modify: `tests/test_audit.py`

1. Add a parameterized negative test for each required link invariant.
2. Verify every malformed artifact is rejected without storing prose.
3. Run focused audit coverage and the full offline suite.
