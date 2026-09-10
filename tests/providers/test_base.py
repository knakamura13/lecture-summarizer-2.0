from dataclasses import FrozenInstanceError

import pytest

from summarizer.providers.base import (
    GenerationRequest,
    GenerationResult,
    ModelProvider,
    ProviderRetriesExhaustedError,
)


def test_generation_values_capture_provider_metadata() -> None:
    request = GenerationRequest(
        model="gpt-4o-mini",
        instructions="Summarize accurately.",
        input_text='\n"""\nSource\n"""\n',
        timeout_seconds=30,
        operation_id="leaf-17",
    )
    result = GenerationResult(
        text="Summary",
        provider="openai",
        model="gpt-4o-mini-2024-07-18",
        input_tokens=20,
        output_tokens=4,
        finish_status="completed",
        request_id="req_123",
    )

    assert request.operation_id == "leaf-17"
    assert result.input_tokens == 20
    assert result.output_tokens == 4
    assert result.request_id == "req_123"


def test_generation_values_are_immutable() -> None:
    request = GenerationRequest("model", "instructions", "input", 30)
    result = GenerationResult("text", "provider", "model")

    with pytest.raises(FrozenInstanceError):
        request.model = "changed"
    with pytest.raises(FrozenInstanceError):
        result.text = "changed"


def test_generation_request_preserves_existing_positional_schema_arguments() -> None:
    schema = {"type": "object"}

    request = GenerationRequest(
        "model",
        "instructions",
        "input",
        30,
        "operation-1",
        schema,
        "summary",
    )

    assert request.operation_id == "operation-1"
    assert request.response_schema == schema
    assert request.schema_name == "summary"
    assert request.audit_work_id is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": ""},
        {"instructions": " "},
        {"input_text": ""},
        {"timeout_seconds": 0},
    ],
)
def test_generation_request_rejects_invalid_values(kwargs: dict) -> None:
    values = {
        "model": "model",
        "instructions": "instructions",
        "input_text": "input",
        "timeout_seconds": 30,
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        GenerationRequest(**values)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"text": ""},
        {"provider": " "},
        {"model": ""},
        {"input_tokens": -1},
        {"output_tokens": -1},
    ],
)
def test_generation_result_rejects_invalid_values(kwargs: dict) -> None:
    values = {"text": "text", "provider": "provider", "model": "model"}
    values.update(kwargs)

    with pytest.raises(ValueError):
        GenerationResult(**values)


def test_provider_protocol_accepts_structural_implementation() -> None:
    class Provider:
        def generate(self, request: GenerationRequest) -> GenerationResult:
            return GenerationResult("text", "fake", request.model)

    assert isinstance(Provider(), ModelProvider)


def test_retry_exhaustion_records_attempt_count() -> None:
    error = ProviderRetriesExhaustedError(3)

    assert error.attempts == 3
    assert "3" in str(error)
