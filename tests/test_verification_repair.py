import hashlib
import json
from dataclasses import fields, is_dataclass

import pytest

from summarizer.grounding import SourcePassage
from summarizer.providers.base import GenerationResult
from summarizer.providers.base import ProviderError
from summarizer.tokenization import ConservativeUtf8TokenCounter
from summarizer.verification import (
    Claim,
    ClaimVerdict,
    DraftSpan,
    RepairAction,
    RepairProposal,
    RepairWorkItem,
    VerificationConfig,
    VerificationResponseError,
    VerificationRuntime,
    apply_repairs,
    build_source_lexical_index,
    build_repair_request,
    split_draft_spans,
    verify_and_repair,
    verify_draft_once,
)


class Provider:
    def generate(self, request):  # pragma: no cover - request construction only
        raise AssertionError("not called")


def runtime() -> VerificationRuntime:
    return VerificationRuntime(
        provider=Provider(),
        counter=ConservativeUtf8TokenCounter(),
        model="repair-model",
        timeout_seconds=30,
        context_window_tokens=8192,
    )


def contradicted_atomic_and_fallback(
    pass_index: int, quote: str, *, anchor: str = "42"
) -> tuple[str, str]:
    prefix = f"V{pass_index:02d}"
    return (
        json.dumps({"spans": [{"span_id": f"{prefix}S000001", "anchors": [anchor]}]}),
        json.dumps(
            {
                "findings": [
                    {
                        "claim_id": f"{prefix}C000001",
                        "verdict": "contradicted",
                        "evidence": [{"segment_id": "S000001", "exact_quote": quote}],
                    },
                    {
                        "claim_id": f"{prefix}C000002",
                        "verdict": "contradicted",
                        "evidence": [{"segment_id": "S000001", "exact_quote": quote}],
                    },
                ]
            }
        ),
    )


def retained_strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if is_dataclass(value):
        return tuple(
            string
            for field in fields(value)
            for string in retained_strings(getattr(value, field.name))
        )
    if isinstance(value, dict):
        return tuple(
            string
            for item in (*value.keys(), *value.values())
            for string in retained_strings(item)
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return tuple(string for item in value for string in retained_strings(item))
    return ()


def test_repair_request_preserves_supported_sibling_anchors_without_secrets() -> None:
    span = split_draft_spans("Measured value is 42.", pass_index=1)[0]
    trigger = Claim("V01C000001", span.span_id, 1, "42", False)
    sibling = Claim("V01C000002", span.span_id, 2, "Measured value", False)
    item = RepairWorkItem(
        span=span,
        triggering_claim_ids=(trigger.claim_id,),
        evidence=(SourcePassage("S000001", "The measured value is 41."),),
        preserved_anchors=(sibling.anchor,),
    )

    request = build_repair_request((item,), source_id="a" * 64, runtime=runtime())

    assert request.operation_id == "verification-repair:V01"
    assert request.response_schema is not None
    assert request.schema_name == "verification_repairs"
    assert sibling.anchor in request.input_text
    assert span.content_hash in request.input_text
    assert "endpoint" not in request.input_text.lower()
    assert "credential" not in request.input_text.lower()


def test_apply_repairs_replaces_spans_in_stable_source_order() -> None:
    draft = "First wrong. Second wrong."
    first, second = split_draft_spans(draft, pass_index=1)
    repairs = (
        RepairProposal(second.span_id, second.content_hash, RepairAction.REPLACE, "Second fixed."),
        RepairProposal(first.span_id, first.content_hash, RepairAction.QUALIFY, "First qualified."),
    )

    repaired, events = apply_repairs(
        draft,
        spans=(first, second),
        repairs=repairs,
        triggering_claim_ids={
            first.span_id: ("V01C000001",),
            second.span_id: ("V01C000002",),
        },
    )

    assert repaired == "First qualified.Second fixed."
    assert [event.span_id for event in events] == [first.span_id, second.span_id]
    assert [event.action for event in events] == [RepairAction.QUALIFY, RepairAction.REPLACE]


def test_apply_repairs_rejects_stale_or_missing_supported_anchor() -> None:
    span = split_draft_spans("Value is 42.", pass_index=1)[0]
    stale = RepairProposal(span.span_id, "a" * 64, RepairAction.REPLACE, "Value is 41.")

    with pytest.raises(VerificationResponseError, match="stale"):
        apply_repairs(
            "Value is 42.",
            spans=(span,),
            repairs=(stale,),
            triggering_claim_ids={span.span_id: ("V01C000001",)},
        )

    proposal = RepairProposal(span.span_id, span.content_hash, RepairAction.REPLACE, "Value is 41.")
    with pytest.raises(VerificationResponseError, match="supported anchor"):
        apply_repairs(
            "Value is 42.",
            spans=(span,),
            repairs=(proposal,),
            triggering_claim_ids={span.span_id: ("V01C000001",)},
            preserved_anchors={span.span_id: ("42",)},
        )


def test_apply_repairs_requires_supported_full_fallback_to_survive_unchanged() -> None:
    span = split_draft_spans("The value is 42 in the study.", pass_index=1)[0]
    proposal = RepairProposal(
        span.span_id, span.content_hash, RepairAction.QUALIFY, "The value is 42."
    )

    with pytest.raises(VerificationResponseError, match="supported anchor"):
        apply_repairs(
            span.text,
            spans=(span,),
            repairs=(proposal,),
            triggering_claim_ids={span.span_id: ("V01C000001",)},
            preserved_anchors={span.span_id: (span.text,)},
        )


def test_apply_repairs_rejects_duplicate_and_unknown_span_targets() -> None:
    span = split_draft_spans("Value is 42.", pass_index=1)[0]
    proposal = RepairProposal(span.span_id, span.content_hash, RepairAction.REMOVE, "")

    with pytest.raises(VerificationResponseError, match="duplicate"):
        apply_repairs(
            "Value is 42.",
            spans=(span,),
            repairs=(proposal, proposal),
            triggering_claim_ids={span.span_id: ("V01C000001",)},
        )

    unknown = RepairProposal("V01S000099", hashlib.sha256(b"x").hexdigest(), RepairAction.REMOVE, "")
    with pytest.raises(VerificationResponseError, match="unknown"):
        apply_repairs(
            "Value is 42.",
            spans=(span,),
            repairs=(unknown,),
            triggering_claim_ids={span.span_id: ("V01C000001",)},
        )


def test_verify_and_repair_is_disabled_without_provider_calls() -> None:
    provider = Provider()

    result = verify_and_repair(
        "Value is 42.",
        source_id="a" * 64,
        source_index=build_source_lexical_index(
            provenance_ids=("S000001",), source={"S000001": "Value is 42."}
        ),
        runtime=VerificationRuntime(
            provider=provider,
            counter=ConservativeUtf8TokenCounter(),
            model="repair-model",
            timeout_seconds=30,
            context_window_tokens=8192,
        ),
        config=VerificationConfig(),
    )

    assert result.text == "Value is 42."
    assert not result.failed
    assert result.pass_results == ()


def test_verify_once_escalates_raw_contradictions_through_omitted_evidence() -> None:
    class ScriptedProvider:
        def __init__(self) -> None:
            self.requests = []
            self.responses = iter(
                (
                    '{"spans":[{"span_id":"V01S000001","anchors":[]}]}',
                    '{"findings":[{"claim_id":"V01C000001","verdict":"contradicted","evidence":[{"segment_id":"S000001","exact_quote":"value is 42"}]}]}',
                    '{"findings":[{"claim_id":"V01C000001","verdict":"contradicted","evidence":[{"segment_id":"S000002","exact_quote":"value is 43"}]}]}',
                    '{"findings":[{"claim_id":"V01C000001","verdict":"contradicted","evidence":[{"segment_id":"S000003","exact_quote":"value is 44"}]}]}',
                    '{"findings":[{"claim_id":"V01C000001","verdict":"contradicted","evidence":[{"segment_id":"S000004","exact_quote":"value is 45"}]}]}',
                )
            )

        def generate(self, request):
            self.requests.append(request)
            return GenerationResult(next(self.responses), "scripted", "model")

    provider = ScriptedProvider()
    result = verify_draft_once(
        "The value is 42.",
        source_id="a" * 64,
        source_index=build_source_lexical_index(
            provenance_ids=("S000001", "S000002", "S000003", "S000004"),
            source={
                "S000001": "The value is 42." * 3,
                "S000002": "The value is 43." * 3,
                "S000003": "The value is 44." * 3,
                "S000004": "The value is 45." * 3,
            },
        ),
        runtime=VerificationRuntime(provider, ConservativeUtf8TokenCounter(), "model", 30, 10_000),
        config=VerificationConfig(enabled=True, evidence_tokens=120),
        pass_index=1,
    )

    assert result.assessments[0].verdict is ClaimVerdict.CONTRADICTED
    assert result.selections[0].retrieval_complete
    assert [request.operation_id for request in provider.requests] == [
        "verification-decompose:V01",
        "verification-classify:V01",
        "verification-classify:V01",
    ]
    escalation_request = provider.requests[-1]
    assert all(segment_id in escalation_request.input_text for segment_id in ("S000002", "S000003", "S000004"))


def test_verify_and_repair_reverifies_the_complete_repaired_draft() -> None:
    class ScriptedProvider:
        def __init__(self) -> None:
            self.responses = iter(
                (
                    *contradicted_atomic_and_fallback(1, "value is 41"),
                    '{"repairs":[{"span_id":"V01S000001","original_hash":"%s","action":"replace","replacement":"The value is 41."}]}' % hashlib.sha256(b"The value is 42.").hexdigest(),
                    '{"spans":[{"span_id":"V02S000001","anchors":[]}]}',
                    '{"findings":[{"claim_id":"V02C000001","verdict":"supported","evidence":[{"segment_id":"S000001","exact_quote":"value is 41"}]}]}',
                )
            )

        def generate(self, request):
            return GenerationResult(next(self.responses), "scripted", "model")

    result = verify_and_repair(
        "The value is 42.",
        source_id="a" * 64,
        source_index=build_source_lexical_index(
            provenance_ids=("S000001",), source={"S000001": "The value is 41."}
        ),
        runtime=VerificationRuntime(ScriptedProvider(), ConservativeUtf8TokenCounter(), "model", 30, 10_000),
        config=VerificationConfig(enabled=True),
    )

    assert result.text == "The value is 41."
    assert not result.failed
    assert [generation.phase for generation in result.phase_generations] == [
        "decomposition", "classification", "repair", "decomposition", "classification",
    ]
    assert len(result.pass_results) == 2


def test_verify_and_repair_retains_attempt_metadata_on_malformed_repair() -> None:
    class ScriptedProvider:
        def __init__(self) -> None:
            self.responses = iter(
                (
                    *contradicted_atomic_and_fallback(1, "value is 41"),
                    "not-json",
                )
            )

        def generate(self, request):
            return GenerationResult(next(self.responses), "scripted", "model")

    draft = "The value is 42."
    result = verify_and_repair(
        draft,
        source_id="a" * 64,
        source_index=build_source_lexical_index(
            provenance_ids=("S000001",), source={"S000001": "The value is 41."}
        ),
        runtime=VerificationRuntime(ScriptedProvider(), ConservativeUtf8TokenCounter(), "model", 30, 10_000),
        config=VerificationConfig(enabled=True),
    )

    assert result.failed
    assert result.text == draft
    assert result.failure_codes == ("repair_failed",)
    assert [generation.phase for generation in result.phase_generations] == [
        "decomposition", "classification", "repair",
    ]


def test_verify_and_repair_terminalizes_malformed_initial_decomposition() -> None:
    class ScriptedProvider:
        def generate(self, request):
            return GenerationResult("not-json", "scripted", "model")

    draft = "The value is 42."
    result = verify_and_repair(
        draft,
        source_id="a" * 64,
        source_index=build_source_lexical_index(
            provenance_ids=("S000001",), source={"S000001": "The value is 42."}
        ),
        runtime=VerificationRuntime(ScriptedProvider(), ConservativeUtf8TokenCounter(), "model", 30, 10_000),
        config=VerificationConfig(enabled=True),
    )

    assert result.failed
    assert result.text == draft
    assert result.failure_codes == ("decomposition_failed",)
    assert [(item.phase, item.prompt_version) for item in result.phase_generations] == [
        ("decomposition", "verification-decomposition/1")
    ]


def test_verify_and_repair_terminalizes_invalid_span_anchor_result_after_repair() -> None:
    """A structurally valid but mismatched anchor response is a closed failure."""
    draft = "The value is 42."

    class ScriptedProvider:
        def __init__(self) -> None:
            self.responses = iter(
                (
                    *contradicted_atomic_and_fallback(1, "value is 41"),
                    '{"repairs":[{"span_id":"V01S000001","original_hash":"%s","action":"replace","replacement":"The value is 41."}]}'
                    % hashlib.sha256(draft.encode()).hexdigest(),
                    '{"spans":[{"span_id":"V02S000001","anchors":["not in the draft"]}]}',
                )
            )

        def generate(self, request):
            return GenerationResult(next(self.responses), "scripted", "model")

    result = verify_and_repair(
        draft,
        source_id="a" * 64,
        source_index=build_source_lexical_index(
            provenance_ids=("S000001",), source={"S000001": "The value is 41."}
        ),
        runtime=VerificationRuntime(ScriptedProvider(), ConservativeUtf8TokenCounter(), "model", 30, 10_000),
        config=VerificationConfig(enabled=True, max_repair_passes=2),
    )

    assert result.failed
    assert result.text == draft
    assert result.failure_codes == ("anchor_failed",)
    assert [item.phase for item in result.phase_generations] == [
        "decomposition", "classification", "repair", "decomposition",
    ]


def test_verify_and_repair_closes_post_repair_provider_failure_without_retrying() -> None:
    draft = "The value is 42."

    class ScriptedProvider:
        def __init__(self) -> None:
            self.calls = 0
            self.responses = iter(
                (
                    *contradicted_atomic_and_fallback(1, "value is 41"),
                    '{"repairs":[{"span_id":"V01S000001","original_hash":"%s","action":"replace","replacement":"The value is 41."}]}'
                    % hashlib.sha256(draft.encode()).hexdigest(),
                )
            )

        def generate(self, request):
            self.calls += 1
            if self.calls == 4:
                raise ProviderError("offline")
            return GenerationResult(next(self.responses), "scripted", "model")

    result = verify_and_repair(
        draft,
        source_id="a" * 64,
        source_index=build_source_lexical_index(
            provenance_ids=("S000001",), source={"S000001": "The value is 41."}
        ),
        runtime=VerificationRuntime(ScriptedProvider(), ConservativeUtf8TokenCounter(), "model", 30, 10_000),
        config=VerificationConfig(enabled=True, max_repair_passes=2),
    )

    assert result.failed
    assert result.text == draft
    assert result.repairs == ()
    assert result.failure_codes == ("decomposition_provider_failed",)
    assert len(result.phase_generations) == 4
    assert result.phase_generations[-1].generation is None


def test_verification_config_bounds_repair_passes_to_available_pass_identifiers() -> None:
    with pytest.raises(ValueError, match="between 0 and 49"):
        VerificationConfig(max_repair_passes=50)


def test_post_repair_malformed_pass_does_not_consume_another_repair_attempt() -> None:
    draft = "The value is 42."

    class ScriptedProvider:
        def __init__(self) -> None:
            self.calls = 0
            self.responses = iter(
                (
                    *contradicted_atomic_and_fallback(1, "value is 41"),
                    '{"repairs":[{"span_id":"V01S000001","original_hash":"%s","action":"replace","replacement":"The value is 41."}]}' % hashlib.sha256(draft.encode()).hexdigest(),
                    "not-json",
                )
            )

        def generate(self, request):
            self.calls += 1
            return GenerationResult(next(self.responses), "scripted", "model")

    provider = ScriptedProvider()
    result = verify_and_repair(
        draft,
        source_id="a" * 64,
        source_index=build_source_lexical_index(
            provenance_ids=("S000001",), source={"S000001": "The value is 41."}
        ),
        runtime=VerificationRuntime(provider, ConservativeUtf8TokenCounter(), "model", 30, 10_000),
        config=VerificationConfig(enabled=True, max_repair_passes=2),
    )

    assert result.failed
    assert result.text == draft
    assert provider.calls == 4
    assert result.failure_codes == ("decomposition_failed",)


def test_verify_and_repair_uses_each_configured_repair_pass_at_most_once() -> None:
    """Pass 2 continues from pass 1's repaired draft, keeping both fixes in the final text."""
    first = "The value is 42."
    second = "The value is 41."
    third = "The value is 40."

    class ScriptedProvider:
        def __init__(self) -> None:
            self.responses = iter(
                (
                    *contradicted_atomic_and_fallback(1, "value is 40"),
                    '{"repairs":[{"span_id":"V01S000001","original_hash":"%s","action":"replace","replacement":"%s"}]}' % (hashlib.sha256(first.encode()).hexdigest(), second),
                    *contradicted_atomic_and_fallback(2, "value is 40", anchor="41"),
                    *contradicted_atomic_and_fallback(3, "value is 40", anchor="41"),
                    '{"repairs":[{"span_id":"V03S000001","original_hash":"%s","action":"replace","replacement":"%s"}]}' % (hashlib.sha256(second.encode()).hexdigest(), third),
                    '{"spans":[{"span_id":"V04S000001","anchors":[]}]}',
                    '{"findings":[{"claim_id":"V04C000001","verdict":"supported","evidence":[{"segment_id":"S000001","exact_quote":"value is 40"}]}]}',
                )
            )

        def generate(self, request):
            return GenerationResult(next(self.responses), "scripted", "model")

    result = verify_and_repair(
        first,
        source_id="a" * 64,
        source_index=build_source_lexical_index(
            provenance_ids=("S000001",), source={"S000001": third}
        ),
        runtime=VerificationRuntime(ScriptedProvider(), ConservativeUtf8TokenCounter(), "model", 30, 10_000),
        config=VerificationConfig(enabled=True, max_repair_passes=2),
    )

    assert result.text == third
    assert not result.failed
    assert [event.span_id for event in result.repairs] == ["V01S000001", "V03S000001"]
    assert [item.pass_index for item in result.phase_generations if item.phase == "repair"] == [1, 3]


def test_verify_and_repair_keeps_independent_repairs_in_returned_text() -> None:
    """Pass 1 fixes claim A and pass 3 fixes independent claim B; both survive."""
    draft = "Claim A is wrong. Claim B is wrong."
    fully_repaired = "Claim A is fixed. Claim B is fixed."

    def decomposition(pass_index: int, first_anchor: str, second_anchor: str) -> str:
        return json.dumps(
            {
                "spans": [
                    {"span_id": f"V{pass_index:02d}S000001", "anchors": [first_anchor]},
                    {"span_id": f"V{pass_index:02d}S000002", "anchors": [second_anchor]},
                ]
            }
        )

    def findings(pass_index: int, first_verdict: str, second_verdict: str) -> str:
        return json.dumps(
            {
                "findings": [
                    *[
                        {
                            "claim_id": f"V{pass_index:02d}C{claim_number:06d}",
                            "verdict": first_verdict,
                            "evidence": [
                                {"segment_id": "S000001", "exact_quote": "Claim A is fixed."}
                            ],
                        }
                        for claim_number in (1, 2)
                    ],
                    *[
                        {
                            "claim_id": f"V{pass_index:02d}C{claim_number:06d}",
                            "verdict": second_verdict,
                            "evidence": [
                                {"segment_id": "S000001", "exact_quote": "Claim B is fixed."}
                            ],
                        }
                        for claim_number in (3, 4)
                    ],
                ]
            }
        )

    class ScriptedProvider:
        def __init__(self) -> None:
            self.responses = iter(
                (
                    decomposition(1, "Claim A is wrong", "Claim B is wrong"),
                    findings(1, "contradicted", "supported"),
                    json.dumps(
                        {
                            "repairs": [
                                {
                                    "span_id": "V01S000001",
                                    "original_hash": hashlib.sha256(
                                        "Claim A is wrong. ".encode()
                                    ).hexdigest(),
                                    "action": "replace",
                                    "replacement": "Claim A is fixed. ",
                                }
                            ]
                        }
                    ),
                    decomposition(2, "Claim A is fixed", "Claim B is wrong"),
                    findings(2, "supported", "contradicted"),
                    decomposition(3, "Claim A is fixed", "Claim B is wrong"),
                    findings(3, "supported", "contradicted"),
                    json.dumps(
                        {
                            "repairs": [
                                {
                                    "span_id": "V03S000002",
                                    "original_hash": hashlib.sha256(
                                        "Claim B is wrong.".encode()
                                    ).hexdigest(),
                                    "action": "replace",
                                    "replacement": "Claim B is fixed.",
                                }
                            ]
                        }
                    ),
                    decomposition(4, "Claim A is fixed", "Claim B is fixed"),
                    findings(4, "supported", "supported"),
                )
            )

        def generate(self, request):
            return GenerationResult(next(self.responses), "scripted", "model")

    result = verify_and_repair(
        draft,
        source_id="a" * 64,
        source_index=build_source_lexical_index(
            provenance_ids=("S000001",), source={"S000001": fully_repaired}
        ),
        runtime=VerificationRuntime(ScriptedProvider(), ConservativeUtf8TokenCounter(), "model", 30, 10_000),
        config=VerificationConfig(enabled=True, max_repair_passes=2),
    )

    assert "Claim A is fixed." in result.text
    assert "Claim B is fixed." in result.text
    assert not result.failed
    assert [event.span_id for event in result.repairs] == ["V01S000001", "V03S000002"]
    assert [item.pass_index for item in result.phase_generations if item.phase == "repair"] == [1, 3]


def test_verify_and_repair_discards_nested_repairs_when_continuation_fails() -> None:
    """The outer pass's own committed repair survives a deeper pass that fails closed."""
    draft = "Claim A is wrong. Claim B is wrong."
    outer_repaired = "Claim A is fixed. Claim B is wrong."

    def decomposition(pass_index: int, first_anchor: str, second_anchor: str) -> str:
        return json.dumps(
            {
                "spans": [
                    {"span_id": f"V{pass_index:02d}S000001", "anchors": [first_anchor]},
                    {"span_id": f"V{pass_index:02d}S000002", "anchors": [second_anchor]},
                ]
            }
        )

    def findings(pass_index: int, first_verdict: str, second_verdict: str) -> str:
        return json.dumps(
            {
                "findings": [
                    *[
                        {
                            "claim_id": f"V{pass_index:02d}C{claim_number:06d}",
                            "verdict": first_verdict,
                            "evidence": [
                                {"segment_id": "S000001", "exact_quote": "Claim A is fixed."}
                            ],
                        }
                        for claim_number in (1, 2)
                    ],
                    *[
                        {
                            "claim_id": f"V{pass_index:02d}C{claim_number:06d}",
                            "verdict": second_verdict,
                            "evidence": [
                                {"segment_id": "S000001", "exact_quote": "Claim B is fixed."}
                            ],
                        }
                        for claim_number in (3, 4)
                    ],
                ]
            }
        )

    class ScriptedProvider:
        def __init__(self) -> None:
            self.responses = iter(
                (
                    decomposition(1, "Claim A is wrong", "Claim B is wrong"),
                    findings(1, "contradicted", "supported"),
                    json.dumps(
                        {
                            "repairs": [
                                {
                                    "span_id": "V01S000001",
                                    "original_hash": hashlib.sha256(
                                        "Claim A is wrong. ".encode()
                                    ).hexdigest(),
                                    "action": "replace",
                                    "replacement": "Claim A is fixed. ",
                                }
                            ]
                        }
                    ),
                    decomposition(2, "Claim A is fixed", "Claim B is wrong"),
                    findings(2, "supported", "contradicted"),
                    decomposition(3, "Claim A is fixed", "Claim B is wrong"),
                    findings(3, "supported", "contradicted"),
                    json.dumps(
                        {
                            "repairs": [
                                {
                                    "span_id": "V03S000002",
                                    "original_hash": hashlib.sha256(
                                        "Claim B is wrong.".encode()
                                    ).hexdigest(),
                                    "action": "replace",
                                    "replacement": "Claim B is fixed.",
                                }
                            ]
                        }
                    ),
                    "not-json",
                )
            )

        def generate(self, request):
            return GenerationResult(next(self.responses), "scripted", "model")

    result = verify_and_repair(
        draft,
        source_id="a" * 64,
        source_index=build_source_lexical_index(
            provenance_ids=("S000001",), source={"S000001": "Claim A is fixed. Claim B is fixed."}
        ),
        runtime=VerificationRuntime(ScriptedProvider(), ConservativeUtf8TokenCounter(), "model", 30, 10_000),
        config=VerificationConfig(enabled=True, max_repair_passes=2),
    )

    assert result.failed
    assert result.text == outer_repaired
    assert [event.span_id for event in result.repairs] == ["V01S000001"]


def test_verify_and_repair_never_repairs_a_contradicted_fallback_by_itself() -> None:
    class ScriptedProvider:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, request):
            self.calls += 1
            return GenerationResult(
                (
                    '{"spans":[{"span_id":"V01S000001","anchors":[]}]}'
                    if self.calls == 1
                    else '{"findings":[{"claim_id":"V01C000001","verdict":"contradicted","evidence":[{"segment_id":"S000001","exact_quote":"value is 41"}]}]}'
                ),
                "scripted",
                "model",
            )

    draft = "The value is 42."
    provider = ScriptedProvider()
    result = verify_and_repair(
        draft,
        source_id="a" * 64,
        source_index=build_source_lexical_index(
            provenance_ids=("S000001",), source={"S000001": "The value is 41."}
        ),
        runtime=VerificationRuntime(provider, ConservativeUtf8TokenCounter(), "model", 30, 10_000),
        config=VerificationConfig(enabled=True),
    )

    assert result.failed
    assert result.text == draft
    assert result.limitation_codes == ("repair_not_eligible",)
    assert provider.calls == 2


def test_verify_and_repair_refuses_conflicting_atomic_and_fallback_assessments() -> None:
    class ScriptedProvider:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, request):
            self.calls += 1
            return GenerationResult(
                (
                    '{"spans":[{"span_id":"V01S000001","anchors":["42"]}]}'
                    if self.calls == 1
                    else '{"findings":[{"claim_id":"V01C000001","verdict":"contradicted","evidence":[{"segment_id":"S000001","exact_quote":"value is 41"}]},{"claim_id":"V01C000002","verdict":"supported","evidence":[{"segment_id":"S000001","exact_quote":"value is 41"}]}]}'
                ),
                "scripted",
                "model",
            )

    draft = "The value is 42."
    result = verify_and_repair(
        draft,
        source_id="a" * 64,
        source_index=build_source_lexical_index(
            provenance_ids=("S000001",), source={"S000001": "The value is 41."}
        ),
        runtime=VerificationRuntime(ScriptedProvider(), ConservativeUtf8TokenCounter(), "model", 30, 10_000),
        config=VerificationConfig(enabled=True),
    )

    assert result.failed
    assert result.text == draft
    assert result.limitation_codes == ("repair_conflicting_assessment",)


def test_verify_and_repair_terminalizes_provider_response_errors_with_redacted_attempt() -> None:
    class FailingProvider:
        def generate(self, request):
            raise VerificationResponseError("response boundary")

    result = verify_and_repair(
        "The value is 42.",
        source_id="a" * 64,
        source_index=build_source_lexical_index(
            provenance_ids=("S000001",), source={"S000001": "The value is 42."}
        ),
        runtime=VerificationRuntime(FailingProvider(), ConservativeUtf8TokenCounter(), "model", 30, 10_000),
        config=VerificationConfig(enabled=True),
    )

    assert result.failed
    assert result.failure_codes == ("decomposition_provider_failed",)
    assert [(item.phase, item.generation) for item in result.phase_generations] == [
        ("decomposition", None)
    ]


@pytest.mark.parametrize(
    ("failing_call", "phase", "pass_index", "failure_code"),
    (
        (2, "classification", 1, "classification_provider_failed"),
        (3, "repair", 1, "repair_provider_failed"),
        (4, "decomposition", 2, "decomposition_provider_failed"),
        (5, "classification", 2, "classification_provider_failed"),
    ),
)
def test_verify_and_repair_terminalizes_provider_response_errors_in_every_phase(
    failing_call: int, phase: str, pass_index: int, failure_code: str
) -> None:
    draft = "The value is 42."
    decomposition, findings = contradicted_atomic_and_fallback(1, "value is 41")

    class ScriptedProvider:
        def __init__(self) -> None:
            self.calls = 0
            self.responses = iter(
                (
                    decomposition,
                    findings,
                    '{"repairs":[{"span_id":"V01S000001","original_hash":"%s","action":"replace","replacement":"The value is 41."}]}'
                    % hashlib.sha256(draft.encode()).hexdigest(),
                    '{"spans":[{"span_id":"V02S000001","anchors":[]}]}',
                    '{"findings":[{"claim_id":"V02C000001","verdict":"supported","evidence":[{"segment_id":"S000001","exact_quote":"value is 41"}]}]}',
                )
            )

        def generate(self, request):
            self.calls += 1
            if self.calls == failing_call:
                raise VerificationResponseError("provider response boundary")
            return GenerationResult(next(self.responses), "scripted", "model")

    result = verify_and_repair(
        draft,
        source_id="a" * 64,
        source_index=build_source_lexical_index(
            provenance_ids=("S000001",), source={"S000001": "The value is 41."}
        ),
        runtime=VerificationRuntime(ScriptedProvider(), ConservativeUtf8TokenCounter(), "model", 30, 10_000),
        config=VerificationConfig(enabled=True),
    )

    assert result.failed
    assert result.text == draft
    assert result.failure_codes == (failure_code,)
    assert (result.phase_generations[-1].phase, result.phase_generations[-1].pass_index) == (
        phase,
        pass_index,
    )
    assert result.phase_generations[-1].generation is None


def test_verify_and_repair_redacts_source_passages_from_terminal_results() -> None:
    sentinel = "secret source"
    class ScriptedProvider:
        def __init__(self) -> None:
            self.responses = iter(
                (
                    '{"spans":[{"span_id":"V01S000001","anchors":[]}]}',
                    '{"findings":[{"claim_id":"V01C000001","verdict":"contradicted","evidence":[{"segment_id":"S000001","exact_quote":"secret source"}]}]}',
                    "not-json",
                )
            )

        def generate(self, request):
            return GenerationResult(next(self.responses), "scripted", "model")

    result = verify_and_repair(
        "The value is 42.",
        source_id="a" * 64,
        source_index=build_source_lexical_index(
            provenance_ids=("S000001",), source={"S000001": "secret source value is 41."}
        ),
        runtime=VerificationRuntime(ScriptedProvider(), ConservativeUtf8TokenCounter(), "model", 30, 10_000),
        config=VerificationConfig(enabled=True),
    )

    assert result.failed
    assert result.pass_results[0].bundles[0].passages == ()
    assert result.pass_results[0].bundles[0].selection.selected_ids == ("S000001",)
    assert sentinel not in retained_strings(result)
    assert all(generation.text == "[redacted]" for generation in result.generations)


def test_verify_and_repair_terminalizes_evidence_capacity_failures() -> None:
    class ScriptedProvider:
        def generate(self, request):
            return GenerationResult(
                '{"spans":[{"span_id":"V01S000001","anchors":[]}]}', "scripted", "model"
            )

    result = verify_and_repair(
        "The value is 42.",
        source_id="a" * 64,
        source_index=build_source_lexical_index(
            provenance_ids=("S000001",), source={"S000001": "The value is 42."}
        ),
        runtime=VerificationRuntime(ScriptedProvider(), ConservativeUtf8TokenCounter(), "model", 30, 10_000),
        config=VerificationConfig(enabled=True, evidence_tokens=1),
    )

    assert result.failed
    assert result.failure_codes == ("evidence_capacity_failed",)
    assert [item.phase for item in result.phase_generations] == [
        "decomposition", "classification",
    ]
    assert result.phase_generations[-1].generation is None
