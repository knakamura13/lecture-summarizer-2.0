import importlib
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx2
import openai
import pytest

from summarizer.providers.base import (
    GenerationRequest,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
)
from summarizer.providers.openai import OpenAIProvider

REQUEST = GenerationRequest(
    model="gpt-4o-mini",
    instructions="Summarize accurately.",
    input_text='\n"""\nSensitive source\n"""\n',
    timeout_seconds=30,
    operation_id="leaf-17",
)


class FakeResponses:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def test_adapts_request_response_and_constructs_client_lazily() -> None:
    response = SimpleNamespace(
        output_text="  concise\nsummary  ",
        model="gpt-4o-mini-2024-07-18",
        status="completed",
        usage=SimpleNamespace(input_tokens=100, output_tokens=12),
        _request_id="req_123",
    )
    responses = FakeResponses(response)
    factory_calls: list[dict[str, object]] = []

    def client_factory(**kwargs: object) -> object:
        factory_calls.append(kwargs)
        return SimpleNamespace(responses=responses)

    provider = OpenAIProvider(client_factory=client_factory)
    assert factory_calls == []

    first = provider.generate(REQUEST)
    second = provider.generate(REQUEST)

    assert factory_calls == [{"max_retries": 0}]
    assert responses.calls == [
        {
            "model": "gpt-4o-mini",
            "instructions": "Summarize accurately.",
            "input": '\n"""\nSensitive source\n"""\n',
            "timeout": 30,
        },
        {
            "model": "gpt-4o-mini",
            "instructions": "Summarize accurately.",
            "input": '\n"""\nSensitive source\n"""\n',
            "timeout": 30,
        },
    ]
    assert first.text == "concise summary"
    assert first.provider == "openai"
    assert first.model == "gpt-4o-mini-2024-07-18"
    assert first.input_tokens == 100
    assert first.output_tokens == 12
    assert first.finish_status == "completed"
    assert first.request_id == "req_123"
    assert second == first


def test_concurrent_first_calls_construct_one_client_without_deadlock() -> None:
    response = SimpleNamespace(output_text="summary", model="gpt-4o-mini")
    factory_calls: list[dict[str, object]] = []
    factory_barrier = threading.Barrier(2)
    start = threading.Barrier(3)

    def client_factory(**kwargs: object) -> object:
        factory_calls.append(kwargs)
        try:
            factory_barrier.wait(timeout=0.1)
        except threading.BrokenBarrierError:
            pass
        return SimpleNamespace(responses=FakeResponses(response))

    provider = OpenAIProvider(client_factory=client_factory)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(lambda: (start.wait(), provider.generate(REQUEST))[1]) for _ in range(2)]
        start.wait()
        results = [future.result(timeout=2) for future in futures]

    assert factory_calls == [{"max_retries": 0}]
    assert [result.text for result in results] == ["summary", "summary"]


@pytest.mark.parametrize("output_text", [None, "", "   ", 42])
def test_rejects_missing_or_invalid_output_text(output_text: object) -> None:
    provider = OpenAIProvider(
        client_factory=lambda **_kwargs: SimpleNamespace(
            responses=FakeResponses(SimpleNamespace(output_text=output_text))
        )
    )

    with pytest.raises(ProviderResponseError, match="text"):
        provider.generate(REQUEST)


def test_allows_absent_optional_response_metadata() -> None:
    provider = OpenAIProvider(
        client_factory=lambda **_kwargs: SimpleNamespace(
            responses=FakeResponses(SimpleNamespace(output_text="summary"))
        )
    )

    result = provider.generate(REQUEST)

    assert result.model == REQUEST.model
    assert result.input_tokens is None
    assert result.output_tokens is None
    assert result.finish_status is None
    assert result.request_id is None


def test_missing_environment_credential_is_an_actionable_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAIProvider()

    with pytest.raises(
        ProviderAuthenticationError,
        match="OPENAI_API_KEY",
    ):
        provider.generate(REQUEST)


@pytest.mark.parametrize("status", ["incomplete", "failed"])
def test_rejects_nonempty_terminal_partial_responses(status: str) -> None:
    provider = OpenAIProvider(
        client_factory=lambda **_kwargs: SimpleNamespace(
            responses=FakeResponses(
                SimpleNamespace(output_text="partial summary", status=status)
            )
        )
    )

    with pytest.raises(ProviderResponseError, match=status):
        provider.generate(REQUEST)


def test_import_does_not_construct_an_openai_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def fail_if_called(**kwargs: object) -> object:
        calls.append(kwargs)
        raise AssertionError("OpenAI client constructed during import")

    monkeypatch.setattr(openai, "OpenAI", fail_if_called)
    module = importlib.import_module("summarizer.providers.openai")

    importlib.reload(module)

    assert calls == []


def _sdk_error(name: str) -> BaseException:
    request = httpx2.Request("POST", "https://api.openai.com/v1/responses")
    if name == "timeout":
        return openai.APITimeoutError(request)
    if name == "connection":
        return openai.APIConnectionError(request=request)
    status_by_name = {
        "rate_limit": 429,
        "authentication": 401,
        "request": 400,
        "server": 500,
    }
    response = httpx2.Response(status_by_name[name], request=request)
    class_by_name = {
        "rate_limit": openai.RateLimitError,
        "authentication": openai.AuthenticationError,
        "request": openai.BadRequestError,
        "server": openai.InternalServerError,
    }
    return class_by_name[name](
        "sentinel credential and Sensitive source",
        response=response,
        body=None,
    )


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("timeout", ProviderTimeoutError),
        ("rate_limit", ProviderRateLimitError),
        ("connection", ProviderConnectionError),
        ("authentication", ProviderAuthenticationError),
        ("request", ProviderRequestError),
        ("server", ProviderServerError),
    ],
)
def test_translates_sdk_errors_without_sensitive_content(
    name: str,
    expected_type: type[BaseException],
) -> None:
    sdk_error = _sdk_error(name)
    provider = OpenAIProvider(
        client_factory=lambda **_kwargs: SimpleNamespace(
            responses=FakeResponses(sdk_error)
        )
    )

    with pytest.raises(expected_type) as exc_info:
        provider.generate(REQUEST)

    message = str(exc_info.value)
    assert "Sensitive source" not in message
    assert "sentinel credential" not in message
    assert exc_info.value.__cause__ is sdk_error


SCHEMA_REQUEST = GenerationRequest(
    model="gpt-4o-mini",
    instructions="Summarize accurately.",
    input_text="Source material",
    timeout_seconds=30,
    response_schema={"type": "object", "properties": {}},
    schema_name="leaf_summary",
)


def _provider_for(responses: FakeResponses) -> OpenAIProvider:
    return OpenAIProvider(
        client_factory=lambda **kwargs: SimpleNamespace(responses=responses)
    )


def test_omits_response_format_when_no_schema_is_requested() -> None:
    responses = FakeResponses(
        SimpleNamespace(output_text="summary", model="gpt-4o-mini", status="completed")
    )

    _provider_for(responses).generate(REQUEST)

    assert "text" not in responses.calls[0]


def test_passes_a_requested_schema_as_a_strict_json_schema_format() -> None:
    responses = FakeResponses(
        SimpleNamespace(output_text="{}", model="gpt-4o-mini", status="completed")
    )

    _provider_for(responses).generate(SCHEMA_REQUEST)

    assert responses.calls[0]["text"] == {
        "format": {
            "type": "json_schema",
            "name": "leaf_summary",
            "schema": {"type": "object", "properties": {}},
            "strict": True,
        }
    }


def test_preserves_response_whitespace_only_for_structured_requests() -> None:
    payload = '{\n  "summary": "a  b",\n  "quote": "line\\nbreak"\n}'
    responses = FakeResponses(
        SimpleNamespace(output_text=payload, model="gpt-4o-mini", status="completed")
    )

    structured = _provider_for(responses).generate(SCHEMA_REQUEST)

    # A quotation copied verbatim out of the source keeps its whitespace, so it
    # can still be located in the segment it came from.
    assert structured.text == payload

    responses = FakeResponses(
        SimpleNamespace(
            output_text="  concise\nsummary  ", model="gpt-4o-mini", status="completed"
        )
    )

    prose = _provider_for(responses).generate(REQUEST)

    assert prose.text == "concise summary"
