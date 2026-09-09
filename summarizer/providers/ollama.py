from __future__ import annotations

import json
from collections.abc import Callable
from threading import Lock
from typing import Any

import httpx
import ollama
from pydantic import ValidationError

from summarizer.providers.base import (
    GenerationRequest,
    GenerationResult,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
    normalize_output_text,
)


def _create_client(**kwargs: object) -> object:
    return ollama.Client(**kwargs)


class OllamaProvider:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        client_factory: Callable[..., object] = _create_client,
    ) -> None:
        self._host = host
        self._client_factory = client_factory
        self._clients: dict[float, Any] = {}
        self._clients_lock = Lock()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        arguments: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {
                    "role": "system",
                    "content": request.instructions,
                },
                {"role": "user", "content": request.input_text},
            ],
            "stream": False,
            "think": False,
        }
        if request.response_schema is not None:
            # The native client takes a JSON Schema directly. It constrains
            # decoding on a best-effort basis rather than guaranteeing it, so
            # callers still parse defensively.
            arguments["format"] = request.response_schema

        try:
            response = self._get_client(request.timeout_seconds).chat(**arguments)
        except httpx.TimeoutException as error:
            raise ProviderTimeoutError("Ollama request timed out") from error
        except (ConnectionError, httpx.TransportError) as error:
            raise ProviderConnectionError(
                "Ollama connection failed; confirm the service is running"
            ) from error
        except ollama.ResponseError as error:
            status_code = error.status_code
            if status_code == 404:
                raise ProviderRequestError(
                    "Ollama model was not found; pull it before retrying"
                ) from error
            if status_code == 429:
                raise ProviderRateLimitError(
                    "Ollama request was rate limited"
                ) from error
            if status_code >= 500:
                raise ProviderServerError(
                    "Ollama server request failed"
                ) from error
            raise ProviderRequestError(
                "Ollama rejected the request"
            ) from error
        except ollama.RequestError as error:
            raise ProviderRequestError(
                "Ollama request was invalid"
            ) from error
        except (json.JSONDecodeError, ValidationError) as error:
            raise ProviderResponseError(
                "Ollama returned a malformed response"
            ) from error

        if getattr(response, "done", None) is not True:
            raise ProviderResponseError(
                "Ollama response did not complete"
            )
        message = getattr(response, "message", None)
        output_text = getattr(message, "content", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise ProviderResponseError(
                "Ollama response did not contain valid text"
            )

        model = getattr(response, "model", None) or request.model
        if not isinstance(model, str) or not model.strip():
            raise ProviderResponseError(
                "Ollama response contained invalid metadata"
            )

        try:
            return GenerationResult(
                text=normalize_output_text(output_text, request),
                provider="ollama",
                model=model,
                input_tokens=getattr(response, "prompt_eval_count", None),
                output_tokens=getattr(response, "eval_count", None),
                finish_status=getattr(response, "done_reason", None),
                request_id=None,
            )
        except (TypeError, ValueError) as error:
            raise ProviderResponseError(
                "Ollama response contained invalid metadata"
            ) from error

    def _get_client(self, timeout_seconds: float) -> Any:
        client = self._clients.get(timeout_seconds)
        if client is not None:
            return client
        with self._clients_lock:
            client = self._clients.get(timeout_seconds)
            if client is None:
                client = self._client_factory(
                    host=self._host,
                    timeout=timeout_seconds,
                )
                self._clients[timeout_seconds] = client
            return client
