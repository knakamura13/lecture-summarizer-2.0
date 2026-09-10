# Reconcile Issue 37 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the grounding documentation, safety module, CLI contract, and repository hygiene match the behavior shipped on `main`.

**Architecture:** Keep the current default hierarchy behavior: merge fanout is sized against the complete request and narrows when selected source passages do not fit. A caller-supplied `GroundingPolicy` remains a fixed reserve path. Keep model-selected provenance in deterministic candidate-priority order, while the final citation projection sorts by source order. Do not implement the separate open #26, #29, #34, or #35 features as part of this reconciliation.

**Tech Stack:** Python 3.12, argparse, Pydantic records, pytest, Markdown documentation, Git ignore rules.

---

### Task 1: Reconcile grounding and citation documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/plans/2026-09-03-source-grounding-provenance-design.md`
- Modify: `docs/plans/2026-09-03-source-grounding-provenance-implementation.md`

**Step 1: Write the documentation assertions**

Record the shipped distinctions: default adaptive grounding measures the complete merge request and retries a narrower fanout; explicit policies reserve `max_tokens`; candidate selection follows category/child/evidence priority; `covered_segments` remains structural document order; merged `SummaryNode.provenance` retains selected priority order; `resolve_citations` produces source-ordered citations.

**Step 2: Update the documents**

Replace the quarter-reserve and all-source-ordered-provenance claims with the behavior above. Preserve the source-grounding safety, validation, and offline-evaluation claims that remain true.

**Step 3: Review literal consistency**

Run `rg -n -i 'quarter|reserve|source order|document order|provenance|citation|adaptive' README.md docs/plans/2026-09-03-source-grounding-provenance-*.md` and inspect every matching statement.

### Task 2: Remove dead safety code

**Files:**
- Modify: `summarizer/safety.py`

**Step 1: Verify the dead-code seam**

Confirm `safe_json_value` has no callers with the graph trace and a literal search.

**Step 2: Remove the unused function and imports**

Keep `redact_text` and its credential-pattern behavior; remove the unused recursive serializer and symbols that existed only for it.

**Step 3: Run focused safety/audit tests**

Run `uv run --with-requirements requirements.txt --with pytest python -m pytest tests/test_verification_audit.py -q`.

### Task 3: Pin current CLI wiring at the public entrypoint

**Files:**
- Modify: `tests/test_documented_cli.py` or `tests/test_cli.py`

**Step 1: Write the failing regression test**

Exercise `main.py --help` and a no-provider `--dry-run` invocation with an explicit strategy/context window. Assert that help exposes the wired budget controls and that the entrypoint reports the supplied strategy and context window.

**Step 2: Run the focused test**

Run `uv run --with-requirements requirements.txt --with pytest python -m pytest tests/test_documented_cli.py tests/test_cli.py -q`.

**Step 3: Make only the minimal CLI/help adjustment if needed**

Keep existing wired behavior and clarify help text only if a test finds a stale statement; do not reject currently supported flags.

### Task 4: Apply issue-37 repository hygiene and stale-fixture cleanup

**Files:**
- Modify: `.gitignore`
- Remove from index: `summarizer.log`
- Modify: `summarizer/budget.py`
- Modify: `tests/test_verification_audit.py`

**Step 1: Correct the two budget comments**

Describe exact lookup followed by longest known prefix only for listed OpenAI families, and state that other providers use the assumed window path.

**Step 2: Align the verification fixture with production fields**

Use `diagnostic_codes` for the emitted reduction code rather than constructing an impossible `warning_codes` runtime result; update the expected audit projection to match the still-open #35 boundary.

**Step 3: Ignore and untrack the historical log**

Add `summarizer.log` to `.gitignore` and remove only its Git index entry, preserving any working-tree file.

### Task 5: Verify the complete issue boundary

**Step 1: Run focused tests and the full offline suite**

Run the affected tests, then `uv run --with-requirements requirements.txt --with pytest python -m pytest -q`.

**Step 2: Verify CLI and repository invariants**

Run `uv run --with-requirements requirements.txt python main.py --help`, the intentionally invalid dry-run budget command, `git ls-files summarizer.log`, and `git diff --check`.

**Step 3: Review the diff against the acceptance criteria**

Confirm no unrelated worktree changes were touched, document the open #26/#29/#34/#35 boundaries, and report the exact verification evidence.
