import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import ollama
import pytest
from pydantic import ValidationError

from summarizer.providers.base import (
    GenerationRequest,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
)
from summarizer.providers.ollama import OllamaProvider

REQUEST = GenerationRequest(
    model="gemma3:4b",
    instructions="Summarize accurately.",
    input_text="Source material",
    timeout_seconds=42,
)


class FakeClient:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, object]] = []

    def chat(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def response(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "message": SimpleNamespace(content=" local\n summary "),
        "model": "gemma3:4b-q4_K_M",
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 42,
        "eval_count": 11,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_adapts_native_chat_request_and_response_lazily() -> None:
    client = FakeClient(response())
    constructions: list[dict[str, object]] = []

    def client_factory(**kwargs: object) -> FakeClient:
        constructions.append(kwargs)
        return client

    provider = OllamaProvider(
        host="http://ollama.internal:11434",
        client_factory=client_factory,
    )

    assert constructions == []

    result = provider.generate(REQUEST)

    assert constructions == [
        {"host": "http://ollama.internal:11434", "timeout": 42}
    ]
    assert client.calls == [
        {
            "model": "gemma3:4b",
            "messages": [
                {"role": "system", "content": "Summarize accurately."},
                {"role": "user", "content": "Source material"},
            ],
            "stream": False,
            "think": False,
        }
    ]
    assert result.text == "local summary"
    assert result.provider == "ollama"
    assert result.model == "gemma3:4b-q4_K_M"
    assert result.input_tokens == 42
    assert result.output_tokens == 11
    assert result.finish_status == "stop"
    assert result.request_id is None

    provider.generate(REQUEST)
    assert len(constructions) == 1


def test_uses_a_client_with_each_distinct_request_timeout() -> None:
    constructions: list[dict[str, object]] = []

    def client_factory(**kwargs: object) -> FakeClient:
        constructions.append(kwargs)
        return FakeClient(response())

    provider = OllamaProvider(client_factory=client_factory)

    provider.generate(REQUEST)
    provider.generate(
        GenerationRequest(
            model=REQUEST.model,
            instructions=REQUEST.instructions,
            input_text=REQUEST.input_text,
            timeout_seconds=90,
        )
    )
    provider.generate(REQUEST)

    assert constructions == [
        {"host": "http://localhost:11434", "timeout": 42},
        {"host": "http://localhost:11434", "timeout": 90},
    ]


def test_concurrent_first_calls_construct_one_client_per_timeout_without_deadlock() -> None:
    constructions: list[dict[str, object]] = []
    factory_barrier = threading.Barrier(2)
    start = threading.Barrier(3)

    def client_factory(**kwargs: object) -> FakeClient:
        constructions.append(kwargs)
        try:
            factory_barrier.wait(timeout=0.1)
        except threading.BrokenBarrierError:
            pass
        return FakeClient(response())

    provider = OllamaProvider(client_factory=client_factory)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(lambda: (start.wait(), provider.generate(REQUEST))[1]) for _ in range(2)]
        start.wait()
        results = [future.result(timeout=2) for future in futures]

    provider.generate(
        GenerationRequest(
            model=REQUEST.model,
            instructions=REQUEST.instructions,
            input_text=REQUEST.input_text,
            timeout_seconds=90,
        )
    )

    assert constructions == [
        {"host": "http://localhost:11434", "timeout": 42},
        {"host": "http://localhost:11434", "timeout": 90},
    ]
    assert [result.text for result in results] == ["local summary", "local summary"]


@pytest.mark.parametrize(
    "bad_response",
    [
        response(message=SimpleNamespace(content=" ")),
        response(message=None),
        response(done=False),
        response(done=None),
    ],
)
def test_rejects_nonterminal_or_missing_content(bad_response: object) -> None:
    provider = OllamaProvider(
        client_factory=lambda **_kwargs: FakeClient(bad_response)
    )

    with pytest.raises(ProviderResponseError):
        provider.generate(REQUEST)


@pytest.mark.parametrize(
    ("error", "expected_type", "message"),
    [
        (
            httpx.ReadTimeout("secret source"),
            ProviderTimeoutError,
            "timed out",
        ),
        (
            ConnectionError("secret source"),
            ProviderConnectionError,
            "connection failed",
        ),
        (
            ollama.ResponseError("model secret-model not found", 404),
            ProviderRequestError,
            "model was not found",
        ),
        (
            ollama.ResponseError("secret source", 429),
            ProviderRateLimitError,
            "rate limited",
        ),
        (
            ollama.ResponseError("secret source", 500),
            ProviderServerError,
            "server request failed",
        ),
        (
            ollama.ResponseError("secret source", 502),
            ProviderServerError,
            "server request failed",
        ),
        (
            ollama.ResponseError("secret source", 400),
            ProviderRequestError,
            "rejected",
        ),
    ],
)
def test_translates_native_errors_without_sensitive_content(
    error: Exception,
    expected_type: type[Exception],
    message: str,
) -> None:
    provider = OllamaProvider(
        client_factory=lambda **_kwargs: FakeClient(error)
    )

    with pytest.raises(expected_type, match=message) as exc_info:
        provider.generate(REQUEST)

    assert exc_info.value.__cause__ is error
    assert "secret" not in str(exc_info.value)


def test_rejects_invalid_metadata() -> None:
    provider = OllamaProvider(
        client_factory=lambda **_kwargs: FakeClient(
            response(prompt_eval_count=-1)
        )
    )

    with pytest.raises(ProviderResponseError, match="metadata"):
        provider.generate(REQUEST)


def test_rejects_non_string_model_metadata() -> None:
    provider = OllamaProvider(
        client_factory=lambda **_kwargs: FakeClient(response(model=123))
    )

    with pytest.raises(ProviderResponseError, match="metadata"):
        provider.generate(REQUEST)


@pytest.mark.parametrize(
    "error",
    [
        json.JSONDecodeError("secret response body", "secret response body", 0),
        ValidationError.from_exception_data("secret schema", []),
    ],
)
def test_translates_malformed_successful_response_without_leaking_body(
    error: Exception,
) -> None:
    provider = OllamaProvider(
        client_factory=lambda **_kwargs: FakeClient(error)
    )

    with pytest.raises(ProviderResponseError, match="malformed") as exc_info:
        provider.generate(REQUEST)

    assert exc_info.value.__cause__ is error
    assert "secret" not in str(exc_info.value)


SCHEMA_REQUEST = GenerationRequest(
    model="gemma3:4b",
    instructions="Summarize accurately.",
    input_text="Source material",
    timeout_seconds=42,
    response_schema={"type": "object", "properties": {}},
    schema_name="leaf_summary",
)


def _provider_for(client: FakeClient) -> OllamaProvider:
    return OllamaProvider(client_factory=lambda **kwargs: client)


def test_omits_format_when_no_schema_is_requested() -> None:
    client = FakeClient(response())

    _provider_for(client).generate(REQUEST)

    assert "format" not in client.calls[0]


def test_passes_a_requested_schema_as_the_native_format_argument() -> None:
    client = FakeClient(response(message=SimpleNamespace(content="{}")))

    _provider_for(client).generate(SCHEMA_REQUEST)

    assert client.calls[0]["format"] == {"type": "object", "properties": {}}


def test_preserves_response_whitespace_only_for_structured_requests() -> None:
    payload = '{\n  "summary": "a  b",\n  "quote": "line\\nbreak"\n}'
    client = FakeClient(response(message=SimpleNamespace(content=payload)))

    structured = _provider_for(client).generate(SCHEMA_REQUEST)

    assert structured.text == payload

    client = FakeClient(response())
    prose = _provider_for(client).generate(REQUEST)

    assert prose.text == "local summary"
