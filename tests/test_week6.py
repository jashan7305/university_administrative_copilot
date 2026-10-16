"""Week 6: the reusable NLP component - structured input/output, confidence scores, decisions, API and CLI.

The contract tests run on a stub copilot (fixed probabilities, real knowledge base and service catalogue), so they
need no trained models; the last tests run the real models when they are on disk."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.api import v1
from backend.main import app
from nlp.component import AdminNLP, AnalyzeInput, BatchInput, ModelsUnavailable, Options
from nlp.component.schemas import MAX_BATCH, ALL_SCHEMAS
from nlp.knowledge.store import load_chunks, load_services
from nlp.ner.extractor import Entity, RuleExtractor
from nlp.retrieval.search import Hit

SERVICES = load_services()
ID_SUBS = sorted(s for s, v in SERVICES.items() if v.intent == "ID_CARDS")
INTENTS = sorted({v.intent for v in SERVICES.values()} | {"out_of_scope"})


class Proba:
    """sklearn-like model returning fixed probabilities chosen by keyword."""

    def __init__(self, classes, rules, default):
        self.classes_ = np.array(classes)
        self.rules, self.default = rules, default

    def predict_proba(self, X):
        out = []
        for x in X:
            want = next((p for kw, p in self.rules if kw in x.lower()), self.default)
            row = np.full(len(self.classes_), (1 - sum(want.values())) / max(len(self.classes_) - len(want), 1))
            for c, p in want.items():
                row[list(self.classes_).index(c)] = p
            out.append(row)
        return np.array(out)


class StubNER:
    id2label = {0: "O", 1: "B-DOCUMENT", 2: "I-DOCUMENT"}

    def extract(self, text):
        ents = RuleExtractor().extract(text)
        i = text.lower().find("id card")
        if i >= 0:
            ents.append(Entity("DOCUMENT", text[i:i + 7], i, i + 7, "crf", 0.91))
        return sorted(ents, key=lambda e: e.start)


class StubRetriever:
    def search(self, query, k=3, intent=None, sub_intent=None):
        chunks = sorted(load_chunks(), key=lambda c: c.intent != intent)   # passages of the intent first
        return [Hit(c.doc_id, 0.05 / (i + 1), c) for i, c in enumerate(chunks[:k])]


def stub_component() -> AdminNLP:
    from nlp.pipeline.copilot import Copilot

    intent = Proba(INTENTS, [("movie", {"out_of_scope": 0.9}), ("vague", {"ID_CARDS": 0.45, "HOSTEL": 0.3}),
                             ("murky", {"ID_CARDS": 0.2})], {"ID_CARDS": 0.92})
    sub = Proba(sorted(SERVICES), [("unsure", {ID_SUBS[0]: 0.3, ID_SUBS[1]: 0.25})], {ID_SUBS[0]: 0.85})
    cp = Copilot({"model": intent, "threshold": 0.35}, {"model": sub, "strategy": "hierarchical"}, StubNER(),
                 StubRetriever())
    return AdminNLP(copilot=cp)


@pytest.fixture(scope="module")
def nlp():
    return stub_component()


@pytest.fixture(scope="module")
def client(nlp):
    app.dependency_overrides[v1.nlp] = lambda: nlp
    yield TestClient(app)
    app.dependency_overrides.clear()


# --------------------------------------------------------------------------- python API

def test_analyze_confident_query(nlp):
    out = nlp.analyze("I lost my ID card, roll no J054, call 9876543210. It is urgent")
    assert out.intent.intent == "ID_CARDS" and out.intent.in_scope
    assert out.intent.confidence.score == pytest.approx(0.92) and out.intent.confidence.level == "high"
    assert out.routing.sub_intent == ID_SUBS[0] and out.routing.confidence.level == "high"
    assert out.routing.department == SERVICES[ID_SUBS[0]].department
    assert [a.score for a in out.intent.alternatives] == sorted((a.score for a in out.intent.alternatives), reverse=True)
    assert out.service_request.priority == "high" and out.decision.action in {"auto_route", "ask_for_information"}
    assert out.sources[0].rank == 1 and out.sources[0].relative_score == 1.0
    assert all(0 <= s.relative_score <= 1 for s in out.sources)
    assert out.model.schema_version == "1.0" and out.model.pipeline == "v2"


def test_entity_confidence(nlp):
    ents = {e.label: e for e in nlp.extract_entities("lost id card, roll J054, paid Rs 500").entities}
    assert ents["STUDENT_ID"].source == "rule" and ents["STUDENT_ID"].confidence is None
    assert ents["DOCUMENT"].source == "crf" and ents["DOCUMENT"].confidence == pytest.approx(0.91)
    assert ents["AMOUNT"].text == "Rs 500"


def test_out_of_scope(nlp):
    out = nlp.analyze("recommend a good movie")
    assert out.intent.intent == "out_of_scope" and not out.intent.in_scope
    assert out.routing is None and out.service_request is None and out.sources == []
    assert out.decision.action == "out_of_scope"


def test_low_intent_confidence_is_rejected(nlp):
    out = nlp.analyze("something murky")
    assert out.intent.intent == "out_of_scope" and out.intent.confidence.level == "low"
    assert "rejection threshold" in out.decision.reason


def test_medium_intent_confidence_asks_to_confirm(nlp):
    out = nlp.analyze("a vague question about my card")
    assert out.intent.confidence.level == "medium" and out.decision.action == "confirm_with_student"


def test_low_routing_confidence_refers_to_staff(nlp):
    out = nlp.analyze("unsure which id card form")
    assert out.routing.confidence.level == "low" and out.decision.action == "refer_to_staff"
    off = nlp.analyze(AnalyzeInput(text="unsure which id card form", options=Options(routing_review_below=0)))
    assert off.decision.action != "refer_to_staff"


def test_options_and_request_id(nlp):
    out = nlp.analyze(AnalyzeInput(text="lost ID card", request_id="abc-1",
                                   options=Options(top_k=5, k_sources=2, include_timings=False)))
    assert out.request_id == "abc-1" and len(out.intent.alternatives) == 5 and len(out.sources) == 2
    assert out.timings_ms is None


def test_classify_and_retrieve(nlp):
    c = nlp.classify("lost my ID card")
    assert c.intent.intent == "ID_CARDS" and c.routing.sub_intent == ID_SUBS[0] and len(c.intent.alternatives) == 3
    r = nlp.retrieve("lost my ID card")
    assert r.intent_used == "ID_CARDS" and len(r.sources) == 3 and all(s.url.startswith("http") for s in r.sources)


def test_batch_and_info(nlp):
    b = nlp.analyze_batch(["lost ID card", "recommend a movie"])
    assert b.count == 2 and [r.intent.in_scope for r in b.results] == [True, False]
    info = nlp.info()
    assert "ID_CARDS" in info.intents and {"DOCUMENT", "STUDENT_ID", "DATE"} <= set(info.entity_labels)
    assert info.thresholds.intent_rejection == 0.35 and info.knowledge_base_passages > 700


@pytest.mark.parametrize("bad", [{"text": ""}, {"text": "   "}, {"text": "x" * 1001}, {"text": "ok", "extra": 1},
                                 {"text": "ok", "options": {"top_k": 0}}])
def test_input_validation(bad):
    with pytest.raises(ValidationError):
        AnalyzeInput(**bad)


def test_batch_limit():
    with pytest.raises(ValidationError):
        BatchInput(items=[{"text": "q"}] * (MAX_BATCH + 1))


def test_output_is_json_serialisable(nlp):
    d = json.loads(nlp.analyze("lost ID card").model_dump_json())
    assert {"intent", "routing", "entities", "answer", "sources", "service_request", "decision", "model"} <= set(d)


def test_json_schemas(tmp_path):
    from nlp.component.__main__ import write_schemas

    paths = write_schemas(tmp_path)
    assert len(paths) == len(ALL_SCHEMAS)
    schema = json.loads((tmp_path / "AnalyzeOutput.json").read_text())
    assert "decision" in schema["properties"] and "Confidence" in schema["$defs"]


def test_missing_models_raise_clear_error(monkeypatch):
    from nlp.pipeline.copilot import Copilot

    def missing(*a, **k):
        raise FileNotFoundError("models/intent_v2.joblib")

    monkeypatch.setattr(Copilot, "from_disk", missing)
    with pytest.raises(ModelsUnavailable, match="Train the models"):
        AdminNLP().analyze("lost ID card")


# --------------------------------------------------------------------------- HTTP API

def test_api_endpoints(client):
    assert client.get("/v1/health").json()["status"] == "ok"
    r = client.post("/v1/analyze", json={"text": "I lost my ID card", "request_id": "r1"})
    assert r.status_code == 200 and r.json()["request_id"] == "r1" and r.json()["intent"]["confidence"]["score"] > 0
    assert client.post("/v1/intent", json={"text": "lost ID card", "top_k": 2}).json()["routing"]["sub_intent"] == ID_SUBS[0]
    assert client.post("/v1/entities", json={"text": "roll J054"}).json()["entities"][0]["label"] == "STUDENT_ID"
    assert len(client.post("/v1/retrieve", json={"text": "lost ID card", "k": 2}).json()["sources"]) == 2
    assert client.post("/v1/analyze/batch", json={"items": [{"text": "a"}, {"text": "b"}]}).json()["count"] == 2
    assert client.get("/v1/info").json()["limits"]["max_batch_items"] == MAX_BATCH


def test_api_validation_error_shape(client):
    r = client.post("/v1/analyze", json={"text": ""})
    assert r.status_code == 422 and r.json()["error"] == "validation_error" and "text" in r.json()["detail"]


def test_api_models_unavailable(monkeypatch):
    from nlp.pipeline.copilot import Copilot

    monkeypatch.setattr(Copilot, "from_disk", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no models")))
    app.dependency_overrides[v1.nlp] = lambda: AdminNLP()
    try:
        r = TestClient(app).post("/v1/analyze", json={"text": "lost ID card"})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 503 and r.json()["error"] == "models_unavailable"


def test_openapi_documents_v1(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/v1/analyze", "/v1/intent", "/v1/entities", "/v1/retrieve", "/v1/info"} <= set(paths)


# --------------------------------------------------------------------------- real models

@pytest.fixture(scope="module")
def real():
    from nlp.pipeline.copilot import INTENT_PATH, NER_PATH, SUBINTENT_PATH
    if not (INTENT_PATH.exists() and NER_PATH.exists() and SUBINTENT_PATH.exists()):
        pytest.skip("models not trained; run python -m nlp.pipeline.train_eval")
    return AdminNLP()


def test_real_models_end_to_end(real):
    out = real.analyze("I lost my ID card, roll no J054. What should I do? It is urgent")
    assert out.intent.intent == "ID_CARDS" and 0 < out.intent.confidence.score <= 1
    assert out.routing and 0 < out.routing.confidence.score <= 1
    assert out.service_request.priority == "high" and out.sources and out.answer
    learned = [e for e in out.entities if e.source != "rule"]
    assert all(e.confidence is not None and 0 < e.confidence <= 1 for e in learned)
    assert real.analyze("can you recommend a good movie to watch tonight").decision.action == "out_of_scope"
