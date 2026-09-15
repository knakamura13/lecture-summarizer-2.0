import pytest
from pydantic import ValidationError

from summarizer.summaries import (
    LEAF_SCHEMA_VERSION,
    MAX_QUOTE_CHARS,
    ContentKind,
    ContentUnit,
    EvidenceItem,
    GroundedAnnotation,
    SummaryNode,
    leaf_summary_schema,
)


def evidence(segment_id: str = "S000001", quote: str | None = None) -> EvidenceItem:
    return EvidenceItem(segment_id=segment_id, quote=quote)


def content_unit(text: str = "The archive was moved in March.") -> ContentUnit:
    return ContentUnit(
        text=text,
        kind=ContentKind.FACT,
        evidence=(evidence(),),
        qualification=None,
        uncertain=False,
    )


def summary_node(**overrides: object) -> SummaryNode:
    fields: dict[str, object] = {
        "summary": "The archive moved and the index was rebuilt.",
        "content_units": (content_unit(),),
        "entities": ("archive",),
        "qualifications": (),
        "contradictions": (),
        "quotations": (),
        "provenance": ("S000001",),
        "level": 0,
    }
    fields.update(overrides)
    return SummaryNode(**fields)  # type: ignore[arg-type]


def test_accepts_a_minimal_valid_leaf_node() -> None:
    node = summary_node()

    assert node.level == 0
    assert node.content_units[0].kind is ContentKind.FACT
    assert node.content_units[0].evidence[0].segment_id == "S000001"


def test_leaf_records_are_immutable() -> None:
    node = summary_node()

    with pytest.raises(ValidationError):
        node.summary = "rewritten"  # type: ignore[misc]


def test_rejects_blank_required_text() -> None:
    with pytest.raises(ValidationError):
        summary_node(summary="   ")

    with pytest.raises(ValidationError):
        EvidenceItem(segment_id="", quote=None)

    with pytest.raises(ValidationError):
        ContentUnit(
            text="",
            kind=ContentKind.CLAIM,
            evidence=(),
            qualification=None,
            uncertain=False,
        )


def test_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        summary_node(recommended_action="delete the archive")


def test_rejects_a_negative_level() -> None:
    with pytest.raises(ValidationError):
        summary_node(level=-1)


def test_contradictions_and_quotations_may_be_empty() -> None:
    node = summary_node(contradictions=(), quotations=())

    assert node.contradictions == ()
    assert node.quotations == ()


def test_qualifications_and_contradictions_retain_evidence() -> None:
    qualification = GroundedAnnotation(
        text="The source only reports this once.", evidence=(evidence(),)
    )
    contradiction = GroundedAnnotation(
        text="Another passage gives a different date.",
        evidence=(evidence("S000002"),),
    )

    node = summary_node(
        qualifications=(qualification,), contradictions=(contradiction,)
    )

    assert node.qualifications[0].evidence[0].segment_id == "S000001"
    assert node.contradictions[0].evidence[0].segment_id == "S000002"


def test_null_collections_are_accepted_as_empty() -> None:
    """Only the strict OpenAI path guarantees a present key.

    Ollama's format argument is best effort and a model may emit null for a
    field it has nothing to say about, so null and empty must mean the same
    thing rather than failing validation.
    """
    node = summary_node(contradictions=None, quotations=None, entities=None)

    assert node.contradictions == ()
    assert node.quotations == ()
    assert node.entities == ()


def test_uncertainty_and_qualification_are_representable() -> None:
    unit = ContentUnit(
        text="The index may have been rebuilt twice.",
        kind=ContentKind.CLAIM,
        evidence=(evidence(quote="rebuilt twice"),),
        qualification="The source hedges this.",
        uncertain=True,
    )

    assert unit.uncertain is True
    assert unit.qualification == "The source hedges this."


def test_schema_is_strict_at_every_level() -> None:
    """The schema is sent to the provider, so strictness is a wire contract.

    OpenAI's strict mode requires every property to be required and every
    object to forbid additional properties, including nested definitions.
    """
    schema = leaf_summary_schema()

    def assert_strict(definition: dict[str, object], label: str) -> None:
        assert definition.get("additionalProperties") is False, label
        properties = definition.get("properties", {})
        assert isinstance(properties, dict)
        assert set(definition.get("required", [])) == set(properties), label

    assert_strict(schema, "root")
    definitions = schema.get("$defs", {})
    assert isinstance(definitions, dict)
    assert definitions, "nested records should appear as definitions"
    for name, definition in definitions.items():
        if definition.get("type") == "object":
            assert_strict(definition, name)


def test_schema_matches_what_the_openai_sdk_would_generate() -> None:
    """Pins the claim that the generated schema is already strict-mode ready.

    The design rests on pydantic's own output being usable as an OpenAI
    `strict: true` schema with no post-processing. That claim would otherwise
    only fail against a live endpoint, so it is checked here against the SDK's
    own converter. A private module path is used deliberately: it is the
    oracle, and openai is already a declared dependency.

    It does not cover every strict-mode rule - unsupported keywords such as
    `minLength` are not stripped by this transform - so it guards drift in
    `additionalProperties`, `required`, nullable handling and `$ref` siblings
    rather than proving live acceptance.
    """
    to_strict = pytest.importorskip("openai.lib._pydantic").to_strict_json_schema

    assert leaf_summary_schema() == to_strict(SummaryNode)


def test_schema_version_is_recorded_for_cache_keys() -> None:
    assert LEAF_SCHEMA_VERSION
    assert isinstance(LEAF_SCHEMA_VERSION, str)

def test_rejects_overly_long_quotes() -> None:
    with pytest.raises(ValidationError, match="quote must not exceed 500 characters"):
        evidence(quote="a" * 501)

    # 500 should be accepted
    evidence(quote="a" * 500)


def test_rejects_too_many_quotations() -> None:
    quotes = tuple(evidence(quote="quote") for _ in range(6))
    with pytest.raises(ValidationError, match="must not exceed 5 quotations"):
        summary_node(quotations=quotes)

    # 5 should be accepted
    quotes_5 = tuple(evidence(quote="quote") for _ in range(5))
    summary_node(quotations=quotes_5)


def test_oversized_quote_is_rejected_wherever_it_appears() -> None:
    """The length cap applies to every `EvidenceItem`, not only `quotations`.

    A content unit's or annotation's own evidence quote inflates a node's
    serialized size exactly as a top-level quotation does, so the same cap
    has to reach both. Payloads are raw dicts, as a provider response would
    be, since an already-validated `EvidenceItem` could not carry the
    oversized quote in the first place.
    """
    oversized_evidence = {
        "segment_id": "S000001",
        "quote": "x" * (MAX_QUOTE_CHARS + 1),
    }

    with pytest.raises(ValidationError):
        ContentUnit.model_validate(
            {
                "text": "The archive was moved in March.",
                "kind": "fact",
                "evidence": [oversized_evidence],
                "qualification": None,
                "uncertain": False,
            }
        )

    with pytest.raises(ValidationError):
        GroundedAnnotation.model_validate(
            {"text": "A hedge.", "evidence": [oversized_evidence]}
        )
