import json
from dataclasses import replace
from pathlib import Path
import socket

import pytest

from summarizer.audit import Citation
from summarizer.providers.base import GenerationRequest
from tests.support.evaluation import (
    CASE_SPECS,
    CURATED_CLAIMS,
    FIXTURES,
    DeterministicSourceProvider,
    _offline_network_guard,
    _rubric,
    evaluate_case,
    execute_case,
    run_evaluation,
)
from tests.support.network_guard import OfflineNetworkError


ORIGINAL_AUTO = tuple(spec for spec in CASE_SPECS if spec.expansion == 1 and spec.strategy == "auto")
HIERARCHICAL_AUTO = next(spec for spec in CASE_SPECS if spec.expansion > 1)


def test_curated_claims_are_exact_spans_of_unchanged_fixtures() -> None:
    for genre, claims in CURATED_CLAIMS.items():
        fixture = next(spec.fixture for spec in ORIGINAL_AUTO if spec.genre == genre)
        source = (FIXTURES / fixture).read_text(encoding="utf-8")
        assert all(claim.quote in source for claim in claims)


def test_source_sensitive_provider_consumes_request_source_not_genre_defaults() -> None:
    provider = DeterministicSourceProvider()
    source_claim = CURATED_CLAIMS["article"][0].quote

    def request(source: str) -> GenerationRequest:
        return GenerationRequest(
            model="test-model",
            instructions="return grounded JSON",
            input_text=f"-----BEGIN-----\n{source}\n-----END-----",
            timeout_seconds=1,
            operation_id="D000001",
        )

    grounded = json.loads(provider.generate(request(source_claim)).text)
    unrelated = json.loads(provider.generate(request("A different source has no such finding.")).text)

    assert source_claim in grounded["summary"]
    assert source_claim not in unrelated["summary"]
    assert grounded != unrelated


def test_original_genres_auto_select_direct_and_keep_all_rubric_dimensions(tmp_path: Path) -> None:
    report = run_evaluation(tmp_path)
    # The public report intentionally keeps one case per original genre; do not
    # infer semantic success from provider call counts or a mock echo.
    originals = [case for case in report["cases"] if case["document_tokens"] < 7_000]
    assert {case["genre"] for case in originals} >= set(CURATED_CLAIMS)
    for case in originals:
        assert case["strategy_selected"] == "direct"
        assert case["root_level"] == 0
        assert case["passed"] is True
        assert all(value["passed"] for value in case["rubric"].values())


def test_automatic_large_case_builds_a_real_multi_level_tree(tmp_path: Path) -> None:
    case = evaluate_case(HIERARCHICAL_AUTO, output_dir=tmp_path)

    assert case["strategy_requested"] == "auto"
    assert case["strategy_selected"] == "hierarchical"
    assert case["root_level"] >= 2
    assert case["passed"] is True
    assert case["provider_calls"] > 4


def test_rubric_rejects_dropped_qualification_or_correction(tmp_path: Path) -> None:
    spec = next(spec for spec in ORIGINAL_AUTO if spec.genre == "transcript")
    document, result, _ = execute_case(spec, output_dir=tmp_path)
    dropped = CURATED_CLAIMS["transcript"][2].quote
    bad_final = replace(result.final, text=result.final.text.replace(dropped, ""))
    bad_result = replace(result, final=bad_final)

    rubric = _rubric(document=document, result=bad_result, claims=CURATED_CLAIMS["transcript"])

    assert rubric["salient_content_coverage"]["passed"] is False
    assert rubric["qualification_contradiction_retention"]["passed"] is False


def test_rubric_rejects_unfaithful_output_and_dangling_citation(tmp_path: Path) -> None:
    spec = next(spec for spec in ORIGINAL_AUTO if spec.genre == "article")
    document, result, _ = execute_case(spec, output_dir=tmp_path)
    dangling = Citation("S999999", document.source_id, 999)
    bad_final = replace(result.final, text="The moon contains an invented unrelated fact.", citations=(dangling,))
    bad_result = replace(result, final=bad_final)

    rubric = _rubric(document=document, result=bad_result, claims=CURATED_CLAIMS["article"])

    assert rubric["salient_content_coverage"]["passed"] is False
    assert rubric["provenance_resolution"]["passed"] is False
    assert rubric["source_faithfulness"]["passed"] is False


def test_standalone_evaluator_blocks_network_and_writes_only_explicit_output(tmp_path: Path) -> None:
    with pytest.raises(OfflineNetworkError):
        with _offline_network_guard():
            socket.create_connection(("example.invalid", 443))

    report = run_evaluation(tmp_path)

    assert report["offline"] is True
    assert (tmp_path / "evaluation.json").is_file()
    assert all(Path(case["audit_path"]).parent == tmp_path for case in report["cases"])
    assert not list(tmp_path.parent.glob("*.audit.json"))
