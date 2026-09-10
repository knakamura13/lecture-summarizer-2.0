"""Small, deterministic end-to-end evaluation of the library pipeline.

This is deliberately an evaluation *harness*, not a benchmark.  It uses the
repository fixtures and a source-sensitive provider which consumes the source
or grounded child/source records present in each real request.  It never calls
an external model.  Generated JSON belongs in the caller supplied output
directory and is never written to the repository.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import socket
from typing import Iterator, Mapping, Sequence

from summarizer.budget import measure_overhead
from summarizer.config import AppConfig, StrategyConfig
from summarizer.ingestion import SourceDocument, ingest_text
from summarizer.pipeline import PipelineConfig, PipelineResult, run_pipeline
from summarizer.providers.base import GenerationRequest, GenerationResult, ModelProvider
from summarizer.segmentation import SegmentationConfig

from tests.support.network_guard import deny_network_access


SCHEMA_VERSION = "evaluation/1"
MODEL = "deterministic-source-sensitive"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@dataclass(frozen=True)
class SalientClaim:
    """A source span whose retention is required by the evaluation rubric."""

    claim_id: str
    quote: str
    kind: str = "claim"
    qualification: bool = False
    correction: bool = False


# These are exact, copied spans from the five checked-in fixtures.  Keeping the
# spans here makes the evaluator fail closed when a fake starts inventing prose
# or when a fixture changes without its expectations being reconsidered.
CURATED_CLAIMS: Mapping[str, tuple[SalientClaim, ...]] = {
    "article": (
        SalientClaim(
            "article-scale",
            "began planting the first of 2,400 trees on March 18, 2027",
        ),
        SalientClaim(
            "article-survey-limit",
            "He cautioned that the survey covered only clear afternoons and should not be treated as a complete measure of residents' heat exposure.",
            qualification=True,
        ),
        SalientClaim(
            "article-grant",
            "the final award has not yet been signed",
            qualification=True,
        ),
    ),
    "report": (
        SalientClaim(
            "report-performance",
            "On-time performance was 87.6 percent, up from 84.1 percent last quarter but below the 90 percent target.",
        ),
        SalientClaim(
            "report-preliminary",
            "The figures are preliminary because forty-two operator logs remain under review.",
            qualification=True,
        ),
        SalientClaim(
            "report-cause",
            "Signal work near Alder Junction accounted for an estimated 31 percent of recorded delay minutes.",
            qualification=True,
        ),
    ),
    "transcript": (
        SalientClaim("transcript-scope-correction", "No, only Tuesdays and Thursdays.", correction=True),
        SalientClaim(
            "transcript-date-correction",
            "June 11.",
            correction=True,
        ),
        SalientClaim(
            "transcript-not-approved",
            "Not yet. Today's discussion may produce a recommendation, but it is not a decision.",
            qualification=True,
        ),
    ),
    "structured": (
        SalientClaim(
            "structured-out-of-scope",
            "archived purchase orders remain out of scope.",
            qualification=True,
        ),
        SalientClaim(
            "structured-rehearsal-warning",
            "The cause appears to be a status filter, but that explanation is not yet confirmed.",
            qualification=True,
        ),
        SalientClaim(
            "structured-stop-condition",
            "Do not start production migration until the corrected export has been reviewed.",
            qualification=True,
        ),
    ),
    "narrative": (
        SalientClaim(
            "narrative-analysis",
            "The result did not establish where the dust originated.",
            qualification=True,
        ),
        SalientClaim(
            "narrative-follow-up",
            "They agreed to check it daily for one week before drawing any conclusion.",
            qualification=True,
        ),
        SalientClaim(
            "narrative-findings",
            "the jar contained mostly clay and salt with a small amount of plant fiber",
        ),
    ),
}

FIXTURE_NAMES: Mapping[str, str] = {
    "article": "article.txt",
    "report": "report.txt",
    "transcript": "transcript.txt",
    "structured": "structured.md",
    "narrative": "narrative.txt",
}


@dataclass(frozen=True)
class CharacterCounter:
    """Exact, monotonic counter suitable for deterministic local budgeting."""

    identity: str = "evaluation:characters"
    exact: bool = True
    monotonic: bool = True

    def count(self, text: str) -> int:
        return len(text)


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    genre: str
    fixture: str
    strategy: str
    expansion: int = 1


CASE_SPECS: tuple[EvaluationCase, ...] = (
    *(EvaluationCase(f"{genre}-auto", genre, filename, "auto") for genre, filename in FIXTURE_NAMES.items()),
    EvaluationCase("article-direct", "article", "article.txt", "direct"),
    # This is intentionally a repeated, source-consistent expansion.  It is
    # large enough to exceed the measured direct capacity, while each repeated
    # copy remains a real source passage with real evidence.
    EvaluationCase("article-auto-hierarchy", "article", "article.txt", "auto", expansion=20),
)


class DeterministicSourceProvider(ModelProvider):
    """Bounded provider that derives every response from the request payload.

    The implementation understands only the three request shapes used by the
    pipeline: leaf/direct, grounded merge, and editorial.  It does not know a
    fixture's answer in advance.  Removing a claim from the request removes it
    from the response, which makes source-faithfulness tests meaningful.
    """

    def __init__(self, claims: Mapping[str, Sequence[SalientClaim]] = CURATED_CLAIMS) -> None:
        self.claims = tuple(claim for values in claims.values() for claim in values)
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        operation = request.operation_id or ""
        if operation == "editorial-final":
            response = {"text": self._editorial_text(request)}
        elif operation == "D000001" or operation.startswith("S"):
            source_id, source_text = self._leaf_source(request)
            response = self._node_from_sources(source_id, ((source_id, source_text),), level=0)
        elif operation.startswith("merge-L"):
            level = int(operation.rsplit("L", 1)[1])
            sources = self._merge_sources(request)
            if not sources:
                raise ValueError("deterministic merge received no authoritative source passage")
            response = self._node_from_sources(
                sources[0][0], tuple(sources), level=level
            )
        else:
            raise AssertionError(f"unexpected deterministic evaluation operation {operation!r}")
        return GenerationResult(
            text=json.dumps(response, separators=(",", ":"), sort_keys=True),
            provider=MODEL,
            model=request.model,
            input_tokens=len(request.input_text),
            output_tokens=len(json.dumps(response)),
            finish_status="stop",
        )

    @staticmethod
    def _body_lines(request: GenerationRequest) -> list[str]:
        lines = request.input_text.splitlines()
        return lines[1:-1] if len(lines) >= 2 else []

    def _leaf_source(self, request: GenerationRequest) -> tuple[str, str]:
        lines = self._body_lines(request)
        operation = request.operation_id or "S000001"
        if any("CORE-BEGIN" in line for line in lines):
            begin = next(index for index, line in enumerate(lines) if "CORE-BEGIN" in line)
            end = next(index for index, line in enumerate(lines) if "CORE-END" in line)
            text = "\n".join(lines[begin + 1 : end])
        else:
            text = "\n".join(line for line in lines if "BEGIN" not in line and "END" not in line)
        return operation, text

    @staticmethod
    def _json_lines(request: GenerationRequest) -> Iterator[dict[str, object]]:
        for line in request.input_text.splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value

    def _merge_sources(self, request: GenerationRequest) -> list[tuple[str, str]]:
        sources: list[tuple[str, str]] = []
        for value in self._json_lines(request):
            identifier = value.get("segment_id")
            text = value.get("text")
            if isinstance(identifier, str) and isinstance(text, str):
                sources.append((identifier, text))
        return sources

    def _matching_claims(self, text: str) -> list[SalientClaim]:
        return [claim for claim in self.claims if claim.quote in text]

    def _node_from_sources(
        self, fallback_id: str, sources: Sequence[tuple[str, str]], *, level: int
    ) -> dict[str, object]:
        units: list[dict[str, object]] = []
        qualifications: list[dict[str, object]] = []
        contradictions: list[dict[str, object]] = []
        quotations: list[dict[str, object]] = []
        provenance: list[str] = []
        summaries: list[str] = []

        for segment_id, source in sources:
            for claim in self._matching_claims(source):
                evidence = [{"segment_id": segment_id, "quote": claim.quote}]
                unit = {
                    "text": claim.quote,
                    "kind": claim.kind,
                    "evidence": evidence,
                    "qualification": claim.quote if claim.qualification else None,
                    "uncertain": claim.qualification,
                }
                if not any(item["text"] == claim.quote for item in units):
                    units.append(unit)
                    summaries.append(claim.quote)
                if claim.qualification and not any(item["text"] == claim.quote for item in qualifications):
                    qualifications.append({"text": claim.quote, "evidence": evidence})
                if claim.correction and not any(item["text"] == claim.quote for item in contradictions):
                    contradictions.append({"text": claim.quote, "evidence": evidence})
                if claim.quote not in [item["quote"] for item in quotations]:
                    quotations.append({"segment_id": segment_id, "quote": claim.quote})
                if segment_id not in provenance:
                    provenance.append(segment_id)

            if not summaries:
                sentence = self._first_sentence(source)
                if sentence:
                    evidence = [{"segment_id": segment_id, "quote": sentence}]
                    units.append(
                        {
                            "text": sentence,
                            "kind": "other",
                            "evidence": evidence,
                            "qualification": None,
                            "uncertain": False,
                        }
                    )
                    summaries.append(sentence)
                    quotations.append({"segment_id": segment_id, "quote": sentence})
                    provenance.append(segment_id)

        if not provenance:
            # This is reachable only for an empty/malformed request and keeps
            # the failure obvious to the pipeline's provenance validator.
            provenance.append(fallback_id)
        if not summaries:
            summaries.append("No supported source assertion was supplied.")

        return {
            "summary": " ".join(summaries[:8]),
            "content_units": units[:12],
            "entities": [],
            "qualifications": qualifications[:8],
            "contradictions": contradictions[:8],
            "quotations": quotations[:8],
            "provenance": provenance,
            "level": level,
        }

    @staticmethod
    def _first_sentence(source: str) -> str:
        source = source.strip()
        if not source:
            return ""
        match = re.search(r".+?(?:[.!?](?:\s|$)|$)", source, flags=re.S)
        sentence = (match.group(0) if match else source).strip()
        return sentence[:360]

    def _editorial_text(self, request: GenerationRequest) -> str:
        root: dict[str, object] | None = None
        for value in self._json_lines(request):
            if "summary" in value and "content_units" in value:
                root = value
                break
        if root is None:
            raise ValueError("editorial request did not contain a grounded root")

        parts: list[str] = []
        summary = root.get("summary")
        if isinstance(summary, str) and summary.strip():
            parts.append(summary.strip())
        for field in ("qualifications", "contradictions"):
            values = root.get(field, ())
            if isinstance(values, list):
                for item in values:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        text = item["text"].strip()
                        if text and text not in parts:
                            parts.append(text)
        return " ".join(parts) or "The supplied grounded record contained no text."


@contextmanager
def _offline_network_guard() -> Iterator[None]:
    """Block both common socket entry points for standalone module execution."""

    original_create = socket.create_connection
    original_connect = socket.socket.connect
    socket.create_connection = deny_network_access  # type: ignore[assignment]
    socket.socket.connect = deny_network_access  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.create_connection = original_create  # type: ignore[assignment]
        socket.socket.connect = original_connect  # type: ignore[assignment]


def _fixture_document(spec: EvaluationCase) -> SourceDocument:
    path = FIXTURES / spec.fixture
    text = path.read_text(encoding="utf-8")
    if spec.expansion > 1:
        text = "\n\n".join(text for _ in range(spec.expansion))
    return ingest_text(text, path=path)


def _strategy_for(document: SourceDocument, strategy: str, counter: CharacterCounter) -> StrategyConfig:
    overhead = measure_overhead(counter, with_overlap=False).total
    if strategy == "direct":
        context_window = overhead + len(document.text) + 1_000
    else:
        # A 14k measured capacity leaves room for two grounded child records
        # and merge overhead; the twenty-copy source remains larger than it.
        context_window = overhead + (14_000 if len(document.text) > 7_000 else len(document.text) + 1_000)
    return StrategyConfig(
        strategy=strategy,  # type: ignore[arg-type]
        context_window=context_window,
        max_output_tokens=128,
        safety_margin_tokens=0,
        safety_margin_fraction=0,
    )


def _rubric(
    *, document: SourceDocument, result: PipelineResult, claims: Sequence[SalientClaim]
) -> dict[str, object]:
    final_text = result.final.text.split("\n\nSources:", 1)[0]
    matched = [claim for claim in claims if claim.quote in final_text]
    expected_qualified = [claim for claim in claims if claim.qualification or claim.correction]
    retained_qualified = [claim for claim in expected_qualified if claim.quote in final_text]

    audit = result.final.audit
    audit_ids = {segment.segment_id for segment in audit.source_segments} if audit else set()
    citation_ids = {citation.segment_id for citation in result.final.citations}
    provenance_ids = set(result.root.summary.provenance)
    segment_cores = (
        {
            segment.segment_id: document.text[segment.core_start : segment.core_end]
            for segment in audit.source_segments
        }
        if audit
        else {}
    )
    evidence_resolves = True
    for field in ("content_units", "qualifications", "contradictions", "quotations"):
        for item in getattr(result.root.summary, field):
            evidence_items = getattr(item, "evidence", (item,))
            for evidence in evidence_items:
                core = segment_cores.get(evidence.segment_id)
                if core is None or (
                    evidence.quote is not None and evidence.quote not in core
                ):
                    evidence_resolves = False
    provenance_ok = (
        bool(audit)
        and bool(citation_ids)
        and citation_ids == provenance_ids
        and citation_ids <= audit_ids
        and provenance_ids <= audit_ids
        and evidence_resolves
    )
    root_text = " ".join(
        (
            result.root.summary.summary,
            *(unit.text for unit in result.root.summary.content_units),
            *(item.text for item in result.root.summary.qualifications),
            *(item.text for item in result.root.summary.contradictions),
        )
    )
    output_sentences_grounded = all(
        sentence.strip() in root_text
        for sentence in re.split(r"(?<=[.!?])\s+", final_text)
        if sentence.strip()
    )
    source_terms = set(re.findall(r"[A-Za-z0-9']+", document.text.casefold()))
    final_terms = set(re.findall(r"[A-Za-z0-9']+", final_text.casefold()))
    faithfulness_overlap = len(source_terms & final_terms) >= 4
    all_units_grounded = all(
        item.evidence and all(evidence.segment_id in audit_ids for evidence in item.evidence)
        for item in result.root.summary.content_units
    )

    return {
        "coherence": {
            "passed": bool(final_text.strip()) and len(final_text.split()) >= 4,
            "observed": f"final editorial contains {len(final_text.split())} words across {len(re.findall(r'[.!?]', final_text))} sentence boundaries",
            "evidence": [final_text[:240]],
        },
        "salient_content_coverage": {
            "passed": len(matched) == len(claims),
            "observed": f"matched {len(matched)}/{len(claims)} curated exact evidence spans",
            "evidence": [f"{claim.claim_id}: {claim.quote!r}" for claim in matched],
            "claims": [
                {"id": claim.claim_id, "evidence_span": claim.quote, "observed": claim.quote in final_text}
                for claim in claims
            ],
        },
        "provenance_resolution": {
            "passed": provenance_ok and all_units_grounded,
            "observed": f"audit segments={len(audit_ids)}, citations={len(citation_ids)}, root provenance={len(provenance_ids)}, evidence_resolves={evidence_resolves}",
            "evidence": [
                f"citation {identifier} resolves={identifier in audit_ids}" for identifier in sorted(citation_ids)
            ],
        },
        "qualification_contradiction_retention": {
            "passed": len(retained_qualified) == len(expected_qualified),
            "observed": f"retained {len(retained_qualified)}/{len(expected_qualified)} required qualification/correction spans",
            "evidence": [
                f"{claim.claim_id}: {claim.quote!r}" for claim in retained_qualified
            ],
        },
        "source_faithfulness": {
            "passed": (
                faithfulness_overlap
                and output_sentences_grounded
                and all_units_grounded
                and not any(
                    phrase in final_text.casefold()
                    for phrase in ("the moon", "unrelated fact", "invented")
                )
            ),
            "observed": f"source/final lexical overlap={len(source_terms & final_terms)}; grounded units={all_units_grounded}; output sentences grounded={output_sentences_grounded}",
            "evidence": [
                "deterministic provider output is derived from request source passages and grounded root",
                f"source_id={document.source_id}",
            ],
        },
    }


def execute_case(
    spec: EvaluationCase, *, output_dir: str | Path
) -> tuple[SourceDocument, PipelineResult, DeterministicSourceProvider]:
    """Execute one case and return live pipeline objects for focused tests."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    document = _fixture_document(spec)
    counter = CharacterCounter()
    provider = DeterministicSourceProvider()
    audit_path = destination / f"{spec.case_id}.audit.json"
    pipeline_config = PipelineConfig(
        target_words=220,
        segmentation=(SegmentationConfig(max_tokens=500) if spec.expansion > 1 else None),
        max_merge_children=2 if spec.expansion > 1 else None,
        include_citations=True,
        audit_path=audit_path,
    )
    result = run_pipeline(
        document,
        provider,
        counter,
        app=AppConfig(model="gpt-4o-mini", timeout_seconds=30),
        strategy=_strategy_for(document, spec.strategy, counter),
        config=pipeline_config,
    )
    return document, result, provider


def evaluate_case(spec: EvaluationCase, *, output_dir: Path) -> dict[str, object]:
    document, result, provider = execute_case(spec, output_dir=output_dir)
    rubric = _rubric(document=document, result=result, claims=CURATED_CLAIMS[spec.genre])
    passed = all(bool(value["passed"]) for value in rubric.values())
    audit_path = output_dir / f"{spec.case_id}.audit.json"
    return {
        "case_id": spec.case_id,
        "genre": spec.genre,
        "fixture": spec.fixture,
        "strategy_requested": spec.strategy,
        "strategy_selected": result.strategy.strategy,
        "strategy_reason": result.strategy.reason,
        "root_level": result.root.level,
        "document_tokens": result.strategy.document_tokens,
        "provider_calls": len(provider.requests),
        "final_text": result.final.text,
        "citations": [asdict(citation) for citation in result.final.citations],
        "audit_path": str(audit_path),
        "rubric": rubric,
        "heuristic_vs_human": {
            "deterministic_fake_evidence": True,
            "heuristic_checks": [
                "exact curated evidence-span matching",
                "audit/citation/tree link resolution",
                "lexical source-overlap and grounded-unit checks",
            ],
            "human_inspection": [
                "coherence and ordering remain a reader-facing judgment",
                "a deterministic fake is not evidence of live-model quality",
            ],
            "live_model_evaluation": False,
        },
        "passed": passed,
    }



def run_evaluation(output_dir: str | Path) -> dict[str, object]:
    """Run the bounded offline suite and write results only beneath output_dir."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with _offline_network_guard():
        cases = [evaluate_case(spec, output_dir=destination) for spec in CASE_SPECS]
    report = {
        "schema_version": SCHEMA_VERSION,
        "offline": True,
        "provider": MODEL,
        "fixture_corpus": list(FIXTURE_NAMES.values()),
        "rubric_dimensions": [
            "coherence",
            "salient_content_coverage",
            "provenance_resolution",
            "qualification_contradiction_retention",
            "source_faithfulness",
        ],
        "cases": cases,
        "passed": all(bool(case["passed"]) for case in cases),
    }
    (destination / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    report = run_evaluation(args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "passed": report["passed"]}, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
