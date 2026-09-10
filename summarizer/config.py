from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal

ProviderName = Literal["openai", "ollama"]


@dataclass(frozen=True)
class ReliabilityConfig:
    """Conservative library defaults for opt-in cache/resume execution."""

    max_in_flight: int = 1
    run_mode: Literal["new", "resume"] = "new"
    run_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.max_in_flight, int) or isinstance(
            self.max_in_flight, bool
        ) or self.max_in_flight <= 0:
            raise ValueError("max_in_flight must be positive")
        if self.run_mode not in ("new", "resume"):
            raise ValueError("run_mode must be new or resume")
        if self.run_mode == "resume" and not (self.run_id or "").strip():
            raise ValueError("resume requires run_id")


@dataclass(frozen=True)
class CacheConfig:
    """Opt-in local cache configuration for library callers."""

    enabled: bool = False
    root: Path = Path(".summarizer-cache")

    def __post_init__(self) -> None:
        if (
            self.root == Path(".")
            or self.root == Path(self.root.anchor)
            or ".." in self.root.parts
            or self.root.is_symlink()
        ):
            raise ValueError("root must name a contained cache directory")


@dataclass(frozen=True)
class AppConfig:
    input_path: Path = Path("input.txt")
    output_path: Path = Path("output.txt")
    model: str = "gpt-4o-mini"
    timeout_seconds: float = 180
    provider: ProviderName = "openai"
    ollama_host: str = "http://localhost:11434"

    def __post_init__(self) -> None:
        if self.input_path == Path("."):
            raise ValueError("input_path must name a file")
        if self.output_path == Path("."):
            raise ValueError("output_path must name a file")
        if self.input_path.resolve(strict=False) == self.output_path.resolve(
            strict=False
        ):
            raise ValueError("output_path must differ from input_path")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.provider not in ("openai", "ollama"):
            raise ValueError("provider must be openai or ollama")
        if not self.ollama_host.strip():
            raise ValueError("ollama_host must not be empty")
        if not isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    initial_delay_seconds: float = 1
    backoff_multiplier: float = 2
    max_delay_seconds: float = 60
    jitter_fraction: float = 0

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if (
            not isfinite(self.initial_delay_seconds)
            or self.initial_delay_seconds <= 0
        ):
            raise ValueError("initial_delay_seconds must be positive")
        if (
            not isfinite(self.backoff_multiplier)
            or self.backoff_multiplier <= 0
        ):
            raise ValueError("backoff_multiplier must be positive")
        if not isfinite(self.max_delay_seconds) or self.max_delay_seconds <= 0:
            raise ValueError("max_delay_seconds must be positive")
        if not isfinite(self.jitter_fraction) or not 0 <= self.jitter_fraction <= 1:
            raise ValueError("jitter_fraction must be between 0 and 1")


@dataclass(frozen=True)
class LegacyWorkflowConfig:
    chunk_size: int = 1000
    max_chunks: int = -1
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.max_chunks == 0 or self.max_chunks < -1:
            raise ValueError("max_chunks must be -1 or a positive integer")


StrategyName = Literal["auto", "direct", "hierarchical"]


@dataclass(frozen=True)
class StrategyConfig:
    """How to choose an execution path, and what to reserve when sizing it.

    Kept separate from `AppConfig` rather than added to it. `AppConfig` is
    constructed positionally in existing tests, and a commit exists in this
    repository's history solely to repair that after a field moved, so budget
    settings live in their own record instead of re-opening that hazard.
    """

    strategy: StrategyName = "auto"
    context_window: int | None = None
    max_output_tokens: int = 1_024
    safety_margin_tokens: int = 256
    safety_margin_fraction: float = 0.02
    # Unset means capacity alone decides. A cap exists because a window-only
    # rule sends very large documents in one call, which is a quality question
    # this project has not answered; setting it makes hierarchical reachable
    # without a code change.
    max_direct_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.strategy not in ("auto", "direct", "hierarchical"):
            raise ValueError(
                "strategy must be one of auto, direct, or hierarchical"
            )
        if self.context_window is not None and self.context_window <= 0:
            raise ValueError("context_window must be positive when provided")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.safety_margin_tokens < 0:
            raise ValueError("safety_margin_tokens must not be negative")
        if not isfinite(self.safety_margin_fraction) or not (
            0 <= self.safety_margin_fraction < 1
        ):
            raise ValueError("safety_margin_fraction must be at least 0 and below 1")
        if self.max_direct_tokens is not None and self.max_direct_tokens <= 0:
            raise ValueError("max_direct_tokens must be positive when provided")
