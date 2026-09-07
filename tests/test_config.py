from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from summarizer.config import AppConfig, CacheConfig, LegacyWorkflowConfig, RetryPolicy


def test_configuration_defaults_are_legacy_compatible() -> None:
    app = AppConfig()
    cache = CacheConfig()
    retry = RetryPolicy()
    workflow = LegacyWorkflowConfig()

    assert app.input_path == Path("input.txt")
    assert app.output_path == Path("output.txt")
    assert app.model == "gpt-4o-mini"
    assert app.provider == "openai"
    assert app.ollama_host == "http://localhost:11434"
    assert app.timeout_seconds == 180
    assert cache.enabled is False
    assert cache.root == Path(".summarizer-cache")
    assert retry.max_attempts == 5
    assert retry.initial_delay_seconds == 1
    assert retry.backoff_multiplier == 2
    assert workflow.chunk_size == 1000
    assert workflow.max_chunks == -1
    assert workflow.dry_run is False


@pytest.mark.parametrize(
    ("configuration", "field_name"),
    [
        (AppConfig(), "model"),
        (CacheConfig(), "enabled"),
        (RetryPolicy(), "max_attempts"),
        (LegacyWorkflowConfig(), "chunk_size"),
    ],
)
def test_configuration_is_immutable(
    configuration: object,
    field_name: str,
) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(configuration, field_name, "changed")


def test_cache_configuration_rejects_a_symlinked_final_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "cache"
    root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="root"):
        CacheConfig(root=root)


@pytest.mark.parametrize(
    ("factory", "field_name"),
    [
        (lambda: AppConfig(model=" "), "model"),
        (lambda: AppConfig(provider="other"), "provider"),
        (lambda: AppConfig(ollama_host=" "), "ollama_host"),
        (lambda: AppConfig(input_path=Path("")), "input_path"),
        (lambda: AppConfig(output_path=Path("")), "output_path"),
        (
            lambda: AppConfig(
                input_path=Path("same.txt"),
                output_path=Path("./same.txt"),
            ),
            "output_path",
        ),
        (lambda: AppConfig(timeout_seconds=0), "timeout_seconds"),
        (lambda: AppConfig(timeout_seconds=float("nan")), "timeout_seconds"),
        (lambda: AppConfig(timeout_seconds=float("inf")), "timeout_seconds"),
        (lambda: CacheConfig(root=Path("/")), "root"),
        (lambda: CacheConfig(root=Path("..")), "root"),
        (lambda: RetryPolicy(max_attempts=0), "max_attempts"),
        (
            lambda: RetryPolicy(initial_delay_seconds=0),
            "initial_delay_seconds",
        ),
        (
            lambda: RetryPolicy(initial_delay_seconds=float("nan")),
            "initial_delay_seconds",
        ),
        (lambda: RetryPolicy(backoff_multiplier=0), "backoff_multiplier"),
        (
            lambda: RetryPolicy(backoff_multiplier=float("inf")),
            "backoff_multiplier",
        ),
        (lambda: LegacyWorkflowConfig(chunk_size=0), "chunk_size"),
        (lambda: LegacyWorkflowConfig(max_chunks=0), "max_chunks"),
        (lambda: LegacyWorkflowConfig(max_chunks=-2), "max_chunks"),
    ],
)
def test_configuration_rejects_invalid_values(
    factory: Callable[[], object],
    field_name: str,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        factory()
