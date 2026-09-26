"""Checks for the Week 3 pipeline: KB, NER, retrieval and the end-to-end service request."""
import pytest

from logic.knowledge.store import load_chunks, load_procedures
from logic.ner.extractor import RuleExtractor, bio_to_entities, spans_to_bio, tokenize


def test_kb_covers_all_intents():
    procs = load_procedures()
    assert len(procs) == 27
    assert len({p.intent for p in procs.values()}) == 9
    assert all(p.department and p.next_action and p.required_documents for p in procs.values())
    assert len(load_chunks()) > len(procs)


def test_bio_roundtrip():
    text = "I need a bonafide certificate for a bank loan"
    spans = [{"start": 9, "end": 29, "label": "DOCUMENT"}, {"start": 34, "end": 45, "label": "PURPOSE"}]
    toks = tokenize(text)
    ents = bio_to_entities(text, toks, spans_to_bio(toks, spans))
    assert [(e.label, e.text) for e in ents] == [("DOCUMENT", "bonafide certificate"), ("PURPOSE", "a bank loan")]


def test_rules():
    found = {(e.label, e.text) for e in RuleExtractor().extract("Paid Rs 45,000 on 12/08/2026, roll J054, call 9876543210, 68%")}
    assert {("MONEY", "Rs 45,000"), ("DATE", "12/08/2026"), ("STUDENT_ID", "J054"), ("PHONE", "9876543210"), ("PERCENTAGE", "68%")} <= found


@pytest.fixture(scope="module")
def copilot():
    from logic.pipeline.copilot import INTENT_PATH, NER_PATH, get_copilot
    if not (INTENT_PATH.exists() and NER_PATH.exists()):
        pytest.skip("models not trained; run python -m logic.pipeline.train_v1")
    return get_copilot()


def test_end_to_end_request(copilot):
    r = copilot.run("I lost my ID card, roll no J054. What should I do? It is urgent")
    assert r.intent == "id_cards"
    assert r.service_request["procedure_id"].startswith("id_")
    assert r.service_request["priority"] == "high"
    assert any(e["label"] == "STUDENT_ID" for e in r.entities)


def test_out_of_scope(copilot):
    r = copilot.run("can you recommend a good movie to watch tonight")
    assert r.intent == "out_of_scope" and r.service_request is None
