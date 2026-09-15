# Issue #35 Verification Diagnostics Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Surface closed verification diagnostics in audit artifacts and batch omitted-evidence contradiction escalation.

**Architecture:** Audit projection merges producer diagnostics into its existing warning-code representation. Escalation uses the existing token packer to group omitted passages for one claim, constructing one combined evidence bundle and provider request per packed batch.

**Tech Stack:** Python 3.11, Pydantic, pytest, deterministic scripted providers.

---

### Task 1: Cover audit diagnostic projection

**Files:**
- Modify: `tests/test_verification_audit.py`
- Modify: `summarizer/audit.py`

**Step 1:** Write a test that supplies `diagnostic_codes=("retrieval_bounded", "conflicting_evidence")` and asserts the audit verification warning codes contain both closed codes.

**Step 2:** Run the focused test and verify it fails because diagnostic codes are not projected.

**Step 3:** Project diagnostic codes with explicit warning codes in stable, deduplicated order.

**Step 4:** Re-run the focused audit test and `tests/test_verification_audit.py`.

### Task 2: Cover batched contradiction escalation

**Files:**
- Modify: `tests/test_verification_repair.py`
- Modify: `summarizer/verification.py`

**Step 1:** Write a scripted-provider test with three omitted segments that fit one request and assert a single escalation classification call containing all three segment identifiers.

**Step 2:** Run the focused test and verify it fails under per-segment escalation.

**Step 3:** Use `pack_work_items` to group omitted passages, build one combined evidence bundle per packed batch, and parse each result against the batch's selected segment map.

**Step 4:** Re-run the focused test and `tests/test_verification_repair.py`.

### Task 3: Validate the issue scope

**Files:**
- Verify: `summarizer/audit.py`
- Verify: `summarizer/verification.py`
- Verify: `tests/test_verification_audit.py`
- Verify: `tests/test_verification_repair.py`

**Step 1:** Run the focused verification/audit suites.

**Step 2:** Run `uv run --with-requirements requirements.txt --with pytest python -m pytest -q`.

**Step 3:** Inspect the diff and confirm no prose, prompt, or source content is newly persisted in the audit.

**Step 4:** Commit the scoped implementation with a conventional commit message.
