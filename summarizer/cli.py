from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from summarizer.budget import BudgetReport, select_strategy
from summarizer.config import (
    AppConfig,
    CacheConfig,
    ReliabilityConfig,
    RetryPolicy,
    StrategyConfig,
)
from summarizer.finalization import (
    FinalizationVerificationError,
    PublicationError,
    _atomic_replace,
)
from summarizer.ingestion import read_source
from summarizer.pipeline import PipelineConfig, run_pipeline
from summarizer.providers.base import ModelProvider, ProviderError
from summarizer.providers.openai import OpenAIProvider
from summarizer.providers.ollama import OllamaProvider
from summarizer.providers.retrying import RetryingProvider
from summarizer.segmentation import SegmentationConfig
from summarizer.tokenization import TokenCounter, resolve_token_counter
from summarizer.verification import VerificationConfig


@dataclass(frozen=True)
class ParsedConfig:
    app: AppConfig
    retry: RetryPolicy
    strategy: StrategyConfig
    pipeline: PipelineConfig
    dry_run: bool = False

def build_provider(
    config: AppConfig,
    *,
    openai_factory: Callable[..., ModelProvider] = OpenAIProvider,
    ollama_factory: Callable[..., ModelProvider] = OllamaProvider,
) -> ModelProvider:
    if config.provider == "ollama":
        return ollama_factory(host=config.ollama_host)
    return openai_factory()


def build_counter(config: AppConfig) -> TokenCounter:
    """Resolve the provider's local token accounting boundary."""
    return resolve_token_counter(provider=config.provider, model=config.model)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize a UTF-8 text file")
    parser.add_argument("--input", type=Path, default=Path("input.txt"))
    parser.add_argument("--output", type=Path, default=Path("output.txt"))
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--provider", choices=("openai", "ollama"), default="openai"
    )
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--target-words", type=int, default=300)
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=None,
        help="maximum tokens in each hierarchical source segment",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=0,
        help="tokens of context repeated between hierarchical segments",
    )
    parser.add_argument("--max-merge-children", type=int, default=None)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--max-repair-passes", type=int, default=1)
    parser.add_argument("--citations", action="store_true")
    parser.add_argument("--audit", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument(
        "--strategy",
        choices=("auto", "direct", "hierarchical"),
        default="auto",
        help=(
            "how to execute: auto picks direct when the document provably "
            "fits, direct requires that it fits, hierarchical always splits"
        ),
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=None,
        help=(
            "the model's total context size in tokens; required to use direct "
            "with a model this project has no table entry for"
        ),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1024,
        help="tokens reserved for the response when sizing a request",
    )
    parser.add_argument(
        "--safety-margin-tokens",
        type=int,
        default=256,
        help="minimum tokens held back from the context window",
    )
    parser.add_argument(
        "--safety-margin-fraction",
        type=float,
        default=0.02,
        help=(
            "fraction of the context window held back; the larger of this and "
            "--safety-margin-tokens applies"
        ),
    )
    parser.add_argument(
        "--max-direct-tokens",
        type=int,
        default=None,
        help=(
            "with --strategy auto, refuse direct above this document size "
            "even when it fits, to force hierarchical execution"
        ),
    )
    return parser


def _conflicts_with_cache(path: Path, cache_root: Path) -> bool:
    path = path.resolve(strict=False)
    cache_root = cache_root.resolve(strict=False)
    return path == cache_root or cache_root in path.parents or path in cache_root.parents


def parse_args(argv: list[str] | None = None) -> ParsedConfig:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        app = AppConfig(
            input_path=args.input,
            output_path=args.output,
            model=args.model,
            provider=args.provider,
            ollama_host=args.ollama_host,
            timeout_seconds=args.timeout,
        )
        retry = RetryPolicy(max_attempts=args.max_retries)
        strategy = StrategyConfig(
            strategy=args.strategy,
            context_window=args.context_window,
            max_output_tokens=args.max_output_tokens,
            safety_margin_tokens=args.safety_margin_tokens,
            safety_margin_fraction=args.safety_margin_fraction,
            max_direct_tokens=args.max_direct_tokens,
        )
        if args.chunk_tokens is None:
            if args.overlap_tokens:
                raise ValueError("--overlap-tokens requires --chunk-tokens")
            segmentation = None
        else:
            segmentation = SegmentationConfig(
                max_tokens=args.chunk_tokens, overlap_tokens=args.overlap_tokens
            )
        verification = VerificationConfig(
            enabled=args.verify, max_repair_passes=args.max_repair_passes
        )
        if args.max_merge_children is not None and args.max_merge_children <= 0:
            raise ValueError("max_merge_children must be positive when provided")
        cache_enabled = args.cache_dir is not None
        if args.max_concurrency > 1 and not cache_enabled:
            raise ValueError("--max-concurrency > 1 requires --cache-dir")
        if args.resume and not cache_enabled:
            raise ValueError("--resume requires --cache-dir")
        if args.run_id is not None and not cache_enabled:
            raise ValueError("--run-id requires --cache-dir")
        if cache_enabled and not (args.run_id or "").strip():
            raise ValueError("--cache-dir requires --run-id")
        if cache_enabled and args.audit is None and not args.dry_run:
            raise ValueError("--cache-dir requires --audit for paired publication")
        cache_root = args.cache_dir or Path(".summarizer-cache")
        cache = CacheConfig(enabled=cache_enabled, root=cache_root)
        reliability = ReliabilityConfig(
            max_in_flight=args.max_concurrency,
            run_mode="resume" if args.resume else "new",
            run_id=args.run_id,
        )
        if args.audit is not None:
            audit_resolved = args.audit.resolve(strict=False)
            if audit_resolved in {
                app.input_path.resolve(strict=False),
                app.output_path.resolve(strict=False),
            }:
                raise ValueError("audit path must differ from input and output paths")
        if cache_enabled and any(
            _conflicts_with_cache(path, cache.root)
            for path in (app.input_path, app.output_path, args.audit)
            if path is not None
        ):
            raise ValueError("cache directory must not overlap input, output, or audit paths")
        pipeline = PipelineConfig(
            target_words=args.target_words,
            segmentation=segmentation,
            max_merge_children=args.max_merge_children,
            include_citations=args.citations,
            audit_path=args.audit,
            verification=verification,
            cache=cache,
            reliability=reliability,
        )
        return ParsedConfig(
            app=app,
            retry=retry,
            strategy=strategy,
            pipeline=pipeline,
            dry_run=args.dry_run,
        )
    except ValueError as error:
        parser.error(str(error))




def _report_dry_run(report: BudgetReport) -> None:
    print(f"Strategy: {report.strategy}")
    print(f"Context window: {report.context_window_tokens} tokens")
    print(f"Usable input capacity: {report.usable_input_capacity} tokens")


def main(
    argv: list[str] | None = None,
    *,
    provider_factory: Callable[[AppConfig], ModelProvider] = build_provider,
    counter_factory: Callable[[AppConfig], TokenCounter] = build_counter,
) -> int:
    config = parse_args(argv)
    try:
        document = read_source(config.app.input_path)
        counter = counter_factory(config.app)
        if config.dry_run:
            report = select_strategy(
                document,
                counter,
                provider=config.app.provider,
                model=config.app.model,
                config=config.strategy,
            )
            _report_dry_run(report)
            return 0
        provider = RetryingProvider(provider_factory(config.app), config.retry)
        result = run_pipeline(
            document,
            provider,
            counter,
            app=config.app,
            strategy=config.strategy,
            config=config.pipeline,
        )
        # Reliable mode publishes the summary and audit together inside the
        # pipeline. The ordinary audit remains an independent atomic file.
        if not config.pipeline.cache.enabled:
            _atomic_replace(config.app.output_path, result.final.text.encode("utf-8"))
        return 0
    except (
        OSError,
        ProviderError,
        ValueError,
        FinalizationVerificationError,
        PublicationError,
    ) as error:
        print(f"Summarization failed: {error}", file=sys.stderr)
        return 1
