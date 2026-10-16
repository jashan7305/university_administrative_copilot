"""Structured input and output of the NMIMS administrative NLP component (pydantic models, JSON-serialisable).

Every output carries `schema_version`; a field is only ever added within a major version, never renamed or removed.
`python -m nlp.component schema` writes the JSON Schema of every model to docs/schemas/.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "1.0"
MAX_TEXT = 1000
MAX_BATCH = 64

ConfidenceLevel = Literal["high", "medium", "low"]
Action = Literal["auto_route", "ask_for_information", "confirm_with_student", "refer_to_staff", "out_of_scope"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- input

class Options(_Model):
    top_k: int = Field(3, ge=1, le=13, description="Ranked alternatives returned for intent and sub-intent.")
    k_sources: int = Field(3, ge=1, le=10, description="Knowledge-base passages retrieved as evidence.")
    routing_review_below: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Refer to staff when the sub-intent confidence is below this value (Week 5: chosen on validation "
                    "for >= 90% routing accuracy on the questions answered automatically). 0 disables it.")
    include_timings: bool = Field(True, description="Return per-stage latency in milliseconds.")


class TextInput(_Model):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT, examples=["I lost my ID card. What should I do?"])
    request_id: str | None = Field(None, max_length=128, description="Caller's id, echoed back unchanged.")

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must contain non-whitespace characters")
        return v.strip()


class AnalyzeInput(TextInput):
    options: Options = Field(default_factory=Options)


class IntentInput(TextInput):
    top_k: int = Field(3, ge=1, le=13)


class RetrieveInput(TextInput):
    k: int = Field(3, ge=1, le=10)
    intent: str | None = Field(None, description="Optional intent whose passages are boosted (else predicted).")


class BatchInput(_Model):
    items: list[AnalyzeInput] = Field(..., min_length=1, max_length=MAX_BATCH)


# --------------------------------------------------------------------------- output parts

class Confidence(_Model):
    score: float = Field(..., ge=0.0, le=1.0, description="Model probability of the returned label.")
    level: ConfidenceLevel = Field(..., description="Band of `score` given the component's decision thresholds.")
    method: str = Field(..., description="How `score` is computed.")


class ScoredLabel(_Model):
    label: str
    score: float = Field(..., ge=0.0, le=1.0)


class IntentResult(_Model):
    intent: str = Field(..., description="One of the 12 NMIMS intents, or 'out_of_scope'.")
    in_scope: bool
    confidence: Confidence
    rejection_threshold: float = Field(..., description="Below this top-class probability the query is out of scope.")
    alternatives: list[ScoredLabel] = Field(..., description="Top-k intents by probability (before rejection).")


class RoutingResult(_Model):
    sub_intent: str = Field(..., description="Specific service, e.g. 'id_card_lost'.")
    department: str
    confidence: Confidence
    alternatives: list[ScoredLabel]


class EntityResult(_Model):
    label: str = Field(..., description="Entity type, e.g. DOCUMENT, DATE, STUDENT_ID.")
    text: str
    start: int = Field(..., ge=0, description="Character offset in the input text (inclusive).")
    end: int = Field(..., ge=0, description="Character offset in the input text (exclusive).")
    source: Literal["rule", "crf", "transformer"]
    confidence: float | None = Field(
        None, ge=0.0, le=1.0,
        description="Mean per-token tag probability of the learned tagger; null for regex-rule entities, which are "
                    "deterministic pattern matches.")


class SourceResult(_Model):
    rank: int = Field(..., ge=1)
    doc_id: str
    title: str
    page: str
    url: str
    score: float = Field(..., description="Retriever score (fused rank score; comparable within one query only).")
    relative_score: float = Field(..., ge=0.0, le=1.0, description="score / top score of this query.")


class ServiceRequestResult(_Model):
    intent: str
    sub_intent: str
    department: str
    required_documents: list[str]
    next_action: str
    extracted_details: dict[str, list[str]]
    missing_information: list[str] = Field(..., description="Questions to ask before the request can be actioned.")
    priority: Literal["high", "normal"]
    status: Literal["ready", "needs_information", "needs_confirmation"]
    supporting_facts: list[str] = Field(..., description="Ids of the verified NMIMS facts used.")
    source_urls: list[str]


class Decision(_Model):
    action: Action
    reason: str


class ModelInfo(_Model):
    component_version: str
    schema_version: str = SCHEMA_VERSION
    pipeline: str = Field(..., description="'v3' (Week 4 fine-tuned models) or 'v2' (Week 3 models).")


# --------------------------------------------------------------------------- outputs

class AnalyzeOutput(_Model):
    request_id: str | None
    query: str
    intent: IntentResult
    routing: RoutingResult | None = Field(..., description="Null when the query is out of scope.")
    entities: list[EntityResult]
    answer: str
    answer_generator: str
    sources: list[SourceResult]
    service_request: ServiceRequestResult | None
    decision: Decision
    timings_ms: dict[str, float] | None
    model: ModelInfo


class IntentOutput(_Model):
    request_id: str | None
    query: str
    intent: IntentResult
    routing: RoutingResult | None
    model: ModelInfo


class EntitiesOutput(_Model):
    request_id: str | None
    query: str
    entities: list[EntityResult]
    model: ModelInfo


class RetrieveOutput(_Model):
    request_id: str | None
    query: str
    intent_used: str | None
    sources: list[SourceResult]
    model: ModelInfo


class BatchOutput(_Model):
    results: list[AnalyzeOutput]
    count: int


class Thresholds(_Model):
    intent_rejection: float
    intent_confirm_below: float
    routing_review_below: float


class ComponentInfo(_Model):
    name: str
    model: ModelInfo
    intents: list[str]
    sub_intents: int
    entity_labels: list[str]
    knowledge_base_passages: int
    thresholds: Thresholds
    confidence_levels: dict[str, str]
    limits: dict[str, int]


class ErrorOutput(_Model):
    error: Literal["validation_error", "models_unavailable", "internal_error"]
    detail: str


ALL_SCHEMAS = [AnalyzeInput, IntentInput, RetrieveInput, BatchInput, AnalyzeOutput, IntentOutput, EntitiesOutput,
               RetrieveOutput, BatchOutput, ComponentInfo, ErrorOutput]
