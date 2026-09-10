import json
import re
from pathlib import Path

import pytest

from summarizer import cli
from summarizer.cli import main, parse_args
from summarizer.config import AppConfig, StrategyConfig
from summarizer.providers.base import (
    GenerationRequest,
    GenerationResult,
    ModelProvider,
    ProviderConnectionError,
    ProviderRequestError,
)


class CharacterCounter:
    identity = "test:characters"
    exact = True
    monotonic = True

    def count(self, text: str) -> int:
        return len(text)


class RecordingProvider:
    def __init__(self, outcome: object = "summary") -> None:
        self.outcome = outcome
        self.calls: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        if request.operation_id == "editorial-final":
            text = json.dumps({"text": str(self.outcome)})
        elif request.operation_id == "D000001":
            text = json.dumps(
                {
                    "summary": str(self.outcome),
                    "content_units": [],
                    "entities": [],
                    "qualifications": [],
                    "contradictions": [],
                    "quotations": [],
                    "provenance": ["D000001"],
                    "level": 0,
                }
            )
        else:
            identifiers = re.findall(r'"segment_id":"([SD]\d+)"', request.input_text)
            text = json.dumps(
                {
                    "summary": str(self.outcome),
                    "content_units": [],
                    "entities": [],
                    "qualifications": [],
                    "contradictions": [],
                    "quotations": [],
                    "provenance": identifiers[-1:] or ["S000001"],
                    "level": 0,
                }
            )
        return GenerationResult(text, "fake", request.model)


def counter_factory(_config: AppConfig) -> CharacterCounter:
    return CharacterCounter()


def test_parse_args_returns_exact_defaults() -> None:
    parsed = parse_args([])

    assert parsed.app.input_path == Path("input.txt")
    assert parsed.app.output_path == Path("output.txt")
    assert parsed.app.model == "gpt-4o-mini"
    assert parsed.app.provider == "openai"
    assert parsed.app.timeout_seconds == 180
    assert parsed.retry.max_attempts == 5
    assert parsed.pipeline.target_words == 300
    assert parsed.pipeline.segmentation is None
    assert parsed.pipeline.include_citations is False
    assert parsed.pipeline.cache.enabled is False
    assert parsed.dry_run is False


def test_parse_args_supports_pipeline_overrides() -> None:
    parsed = parse_args(
        [
            "--input", "source.txt", "--output", "summary.txt",
            "--model", "qwen3.8", "--provider", "ollama",
            "--ollama-host", "http://ollama.internal:11434", "--timeout", "42.5",
            "--max-retries", "3", "--target-words", "120", "--chunk-tokens", "2048",
            "--overlap-tokens", "64", "--max-merge-children", "7", "--verify",
            "--max-repair-passes", "2", "--citations", "--audit", "audit.json",
            "--cache-dir", "cache", "--run-id", "run-one", "--resume",
            "--max-concurrency", "3", "--dry-run",
        ]
    )

    assert parsed.app.provider == "ollama"
    assert parsed.app.timeout_seconds == 42.5
    assert parsed.pipeline.target_words == 120
    assert parsed.pipeline.segmentation is not None
    assert parsed.pipeline.segmentation.max_tokens == 2048
    assert parsed.pipeline.segmentation.overlap_tokens == 64
    assert parsed.pipeline.max_merge_children == 7
    assert parsed.pipeline.verification.enabled is True
    assert parsed.pipeline.verification.max_repair_passes == 2
    assert parsed.pipeline.include_citations is True
    assert parsed.pipeline.audit_path == Path("audit.json")
    assert parsed.pipeline.cache.enabled is True
    assert parsed.pipeline.reliability.run_mode == "resume"
    assert parsed.pipeline.reliability.max_in_flight == 3
    assert parsed.dry_run is True


@pytest.mark.parametrize(
    "arguments",
    [
        ["--timeout", "0"],
        ["--max-retries", "0"],
        ["--target-words", "0"],
        ["--chunk-tokens", "0"],
        ["--overlap-tokens", "-1", "--chunk-tokens", "100"],
        ["--max-merge-children", "0"],
        ["--max-repair-passes", "50"],
        ["--max-concurrency", "0"],
        ["--max-concurrency", "2"],
        ["--input", "same.txt", "--output", "same.txt"],
        ["--run-id", "run-one"],
        ["--resume"],
        ["--cache-dir", "cache"],
    ],
)
def test_parse_args_reports_invalid_configuration(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_args(arguments)
    assert exc_info.value.code == 2


def test_parse_args_explains_concurrency_cache_requirement(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--max-concurrency", "2"])

    assert "--max-concurrency > 1 requires --cache-dir" in capsys.readouterr().err


def test_parse_args_rejects_path_conflicts() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--audit", "input.txt"])
    with pytest.raises(SystemExit):
        parse_args(["--cache-dir", "input.txt", "--run-id", "run", "--audit", "audit.json"])


def test_main_runs_default_pipeline_without_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input.txt").write_text("Source.", encoding="utf-8")
    provider = RecordingProvider("concise summary")

    exit_code = main(
        [], provider_factory=lambda _config: provider, counter_factory=counter_factory
    )

    assert exit_code == 0
    assert (tmp_path / "output.txt").read_text(encoding="utf-8") == "concise summary"
    assert [call.operation_id for call in provider.calls] == ["D000001", "editorial-final"]


def test_main_reports_provider_failure_and_preserves_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input.txt").write_text("Source.", encoding="utf-8")
    (tmp_path / "output.txt").write_text("previous", encoding="utf-8")
    provider = RecordingProvider(ProviderRequestError("invalid request"))

    exit_code = main(
        [], provider_factory=lambda _config: provider, counter_factory=counter_factory
    )

    assert exit_code == 1
    assert "invalid request" in capsys.readouterr().err
    assert (tmp_path / "output.txt").read_text(encoding="utf-8") == "previous"


def test_main_reports_missing_input_without_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    exit_code = main([], provider_factory=lambda _config: RecordingProvider(), counter_factory=counter_factory)
    assert exit_code == 1
    assert "input.txt" in capsys.readouterr().err
    assert not (tmp_path / "output.txt").exists()


def test_dry_run_does_not_construct_provider_or_write_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input.txt").write_text("Source.", encoding="utf-8")
    constructions: list[object] = []

    def provider_factory(_config: AppConfig) -> ModelProvider:
        constructions.append(object())
        raise AssertionError("provider constructed during dry run")

    exit_code = main(
        ["--dry-run"], provider_factory=provider_factory, counter_factory=counter_factory
    )
    assert exit_code == 0
    assert constructions == []
    assert not (tmp_path / "output.txt").exists()


def test_main_reports_missing_openai_credential_without_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / "input.txt").write_text("Source.", encoding="utf-8")

    exit_code = main([], counter_factory=counter_factory)
    stderr = capsys.readouterr().err
    assert exit_code == 1
    assert "OPENAI_API_KEY" in stderr
    assert "Traceback" not in stderr
    assert not (tmp_path / "output.txt").exists()


def test_build_provider_selects_openai_without_ollama_construction() -> None:
    constructions: list[tuple[str, object]] = []
    expected = RecordingProvider()
    provider = cli.build_provider(
        AppConfig(provider="openai"),
        openai_factory=lambda: expected,
        ollama_factory=lambda **kwargs: constructions.append(("ollama", kwargs)),
    )
    assert provider is expected
    assert constructions == []


def test_build_provider_selects_ollama_with_configured_host() -> None:
    constructions: list[tuple[str, object]] = []
    expected = RecordingProvider()
    provider = cli.build_provider(
        AppConfig(provider="ollama", model="qwen3.8", ollama_host="http://ollama.internal:11434"),
        openai_factory=lambda: constructions.append(("openai", None)),
        ollama_factory=lambda **kwargs: constructions.append(("ollama", kwargs)) or expected,
    )
    assert provider is expected
    assert constructions == [("ollama", {"host": "http://ollama.internal:11434"})]


def test_main_reports_unavailable_selected_ollama_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input.txt").write_text("Source.", encoding="utf-8")
    selected: list[AppConfig] = []

    def provider_factory(config: AppConfig) -> ModelProvider:
        selected.append(config)
        return RecordingProvider(ProviderConnectionError("Ollama connection failed"))

    exit_code = main(
        ["--provider", "ollama", "--max-retries", "1"],
        provider_factory=provider_factory,
        counter_factory=counter_factory,
    )
    assert exit_code == 1
    assert "Ollama connection failed" in capsys.readouterr().err
    assert selected[0].provider == "ollama"
    assert not (tmp_path / "output.txt").exists()


@pytest.mark.parametrize("strategy", ["auto", "direct", "hierarchical"])
def test_each_strategy_is_accepted(strategy: str) -> None:
    assert parse_args(["--strategy", strategy]).strategy.strategy == strategy


def test_budget_flags_reach_the_strategy_config() -> None:
    parsed = parse_args(
        ["--strategy", "direct", "--context-window", "32768", "--max-output-tokens", "2048", "--safety-margin-tokens", "512", "--safety-margin-fraction", "0.05", "--max-direct-tokens", "5000"]
    )
    assert parsed.strategy == StrategyConfig(strategy="direct", context_window=32768, max_output_tokens=2048, safety_margin_tokens=512, safety_margin_fraction=0.05, max_direct_tokens=5000)


def test_ollama_uses_offline_conservative_counter() -> None:
    counter = cli.build_counter(
        AppConfig(provider="ollama", model="unknown-local-model")
    )
    assert counter.identity == "estimate:utf8-bytes"
