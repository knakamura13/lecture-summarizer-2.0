from __future__ import annotations

import os
from collections.abc import Callable
from threading import Lock
from typing import Any

import openai

from summarizer.providers.base import (
    GenerationRequest,
    GenerationResult,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
    normalize_output_text,
)


def _create_client(**kwargs: object) -> object:
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise ProviderAuthenticationError("OPENAI_API_KEY is not set")
    return openai.OpenAI(**kwargs)


class OpenAIProvider:
    def __init__(
        self,
        client_factory: Callable[..., object] = _create_client,
    ) -> None:
        self._client_factory = client_factory
        self._client: Any | None = None
        self._client_lock = Lock()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        arguments: dict[str, Any] = {
            "model": request.model,
            "instructions": request.instructions,
            "input": request.input_text,
            "timeout": request.timeout_seconds,
        }
        if request.response_schema is not None:
            arguments["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": request.schema_name,
                    "schema": request.response_schema,
                    "strict": True,
                }
            }

        try:
            response = self._get_client().responses.create(**arguments)
        except openai.APITimeoutError as error:
            raise ProviderTimeoutError("OpenAI request timed out") from error
        except openai.RateLimitError as error:
            raise ProviderRateLimitError("OpenAI request was rate limited") from error
        except openai.APIConnectionError as error:
            raise ProviderConnectionError("OpenAI connection failed") from error
        except openai.AuthenticationError as error:
            raise ProviderAuthenticationError(
                "OpenAI authentication failed"
            ) from error
        except openai.APIStatusError as error:
            request_id = getattr(error, "request_id", None)
            detail = f" (request {request_id})" if request_id else ""
            if error.status_code >= 500:
                raise ProviderServerError(
                    f"OpenAI server request failed{detail}"
                ) from error
            raise ProviderRequestError(
                f"OpenAI rejected the request{detail}"
            ) from error
        except openai.OpenAIError as error:
            raise ProviderRequestError(
                "OpenAI client or request failed"
            ) from error

        status = getattr(response, "status", None)
        if status not in (None, "completed"):
            raise ProviderResponseError(
                f"OpenAI response status was {status}"
            )

        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise ProviderResponseError(
                "OpenAI response did not contain valid text"
            )

        usage = getattr(response, "usage", None)
        return GenerationResult(
            text=normalize_output_text(output_text, request),
            provider="openai",
            model=getattr(response, "model", None) or request.model,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            finish_status=status,
            request_id=getattr(response, "_request_id", None),
        )

    def _get_client(self) -> Any:
        client = self._client
        if client is not None:
            return client
        with self._client_lock:
            if self._client is None:
                self._client = self._client_factory(max_retries=0)
            return self._client
