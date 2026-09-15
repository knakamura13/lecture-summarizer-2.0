import json

from summarizer.direct import whole_document_segment
from summarizer.finalization import finalize_summary
from summarizer.hierarchy import TreeNode
from summarizer.ingestion import ingest_text
from summarizer.providers.base import GenerationResult
from summarizer.summaries import SummaryNode
from summarizer.tokenization import ConservativeUtf8TokenCounter
from summarizer.verification import VerificationConfig


class ScriptedProvider:
    def __init__(self) -> None:
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        if request.operation_id == "editorial-final":
            payload = {"text": "The source fact is correct."}
        elif request.operation_id == "verification-decompose:V01":
            payload = {"spans": [{"span_id": "V01S000001", "anchors": []}]}
        else:
            payload = {
                "findings": [
                    {
                        "claim_id": "V01C000001",
                        "verdict": "supported",
                        "evidence": [
                            {"segment_id": "D000001", "exact_quote": "source fact"}
                        ],
                    }
                ]
            }
        return GenerationResult(json.dumps(payload), "fake", request.model)


def test_finalize_summary_resolves_verification_runtime_without_injection() -> None:
    counter = ConservativeUtf8TokenCounter()
    document = ingest_text("The source fact is correct.")
    segment = whole_document_segment(document, counter)
    summary = SummaryNode.model_validate(
        {
            "summary": "The source fact is correct.",
            "content_units": [],
            "entities": [],
            "qualifications": [],
            "contradictions": [],
            "quotations": [],
            "provenance": [segment.segment_id],
            "level": 0,
        }
    )
    root = TreeNode("L0N0001", 0, 0, summary, (), (segment.segment_id,))
    provider = ScriptedProvider()

    result = finalize_summary(
        summary,
        provider,
        source_id=document.source_id,
        model="test-model",
        timeout_seconds=30,
        target_words=20,
        strategy="direct",
        segments=(segment,),
        nodes=(root,),
        root_node_id=root.node_id,
        counter=counter,
        source_cores={segment.segment_id: segment.text},
        verification=VerificationConfig(enabled=True),
        verification_context_window_tokens=10_000,
    )

    assert result.text == "The source fact is correct."
    assert [request.operation_id for request in provider.requests] == [
        "editorial-final",
        "verification-decompose:V01",
        "verification-classify:V01",
    ]
