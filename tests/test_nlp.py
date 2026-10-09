"""NLP component checks: preprocessing, knowledge base, NER helpers and the end-to-end copilot."""
import pytest

from nlp.intent.preprocess import DEFAULT, Preprocessor, SpellCorrector
from nlp.knowledge.store import load_chunks, load_services
from nlp.pipeline.summarize import MMRSummarizer
from nlp.ner.extractor import RuleExtractor, bio_to_entities, spans_to_bio, tokenize


@pytest.fixture(scope="module")
def prep():
    return Preprocessor(DEFAULT, SpellCorrector(["certificate certificate hostel hostel attendance attendance"]))


@pytest.mark.parametrize("text, expected", [
    ("I can't pay", "i cannot pay"),
    ("pls send NOC", "please send no objection certificate"),
    ("attendance is 68%", "attendance is <percent>"),
    ("paid Rs. 45,000 on 12/08/2026", "pay <money> on <date>"),
    ("call 9876543210", "call <phone>"),
    ("roll no J054", "roll no <rollno>"),
    ("hostle certficate", "hostel certificate"),
    ("has it been credited as promised", "has it been credit as promise"),
])
def test_preprocessing(prep, text, expected):
    assert prep(text) == expected


def test_spell_corrector_leaves_valid_english(prep):
    assert prep("call") == "call"


def test_knowledge_base_and_services():
    chunks, services = load_chunks(), load_services()
    assert len(chunks) > 700 and all(c.url.startswith("http") for c in chunks)
    assert len({s.intent for s in services.values()}) == 12
    assert all(s.department and s.fact_ids and s.sources for s in services.values())


def test_summarizer_is_extractive():
    texts = ["Revaluation is applied for online on the SAP portal. A photocopy must be requested first. "
             "The window closes at 4 p.m. on the third day."]
    summary = MMRSummarizer().summarize("how do i apply for revaluation", texts)
    assert summary and all(sent in texts[0] for sent in summary.split(". ") if sent)


def test_bio_roundtrip():
    text = "I need a bonafide certificate for a bank loan"
    spans = [{"start": 9, "end": 29, "label": "DOCUMENT"}, {"start": 34, "end": 45, "label": "PURPOSE"}]
    toks = tokenize(text)
    ents = bio_to_entities(text, toks, spans_to_bio(toks, spans))
    assert [(e.label, e.text) for e in ents] == [("DOCUMENT", "bonafide certificate"), ("PURPOSE", "a bank loan")]


def test_rules():
    found = {(e.label, e.text) for e in RuleExtractor().extract("Paid Rs 45,000 on 12/08/2026, roll J054, call 9876543210, 68%")}
    assert {("AMOUNT", "Rs 45,000"), ("DATE", "12/08/2026"), ("STUDENT_ID", "J054"), ("PHONE", "9876543210"), ("PERCENTAGE", "68%")} <= found


@pytest.fixture(scope="module")
def copilot():
    from nlp.pipeline.copilot import INTENT_PATH, NER_PATH, SUBINTENT_PATH, get_copilot
    if not (INTENT_PATH.exists() and NER_PATH.exists() and SUBINTENT_PATH.exists()):
        pytest.skip("models not trained; run python -m nlp.pipeline.train_eval")
    return get_copilot()


def test_end_to_end_request(copilot):
    r = copilot.run("I lost my ID card, roll no J054. What should I do? It is urgent")
    assert r.intent == "ID_CARDS"
    sr = r.service_request
    assert sr["priority"] == "high" and sr["department"] and sr["supporting_facts"] and sr["source_urls"]
    assert any(e["label"] == "STUDENT_ID" for e in r.entities)
    assert r.sources and r.answer


def test_out_of_scope(copilot):
    r = copilot.run("can you recommend a good movie to watch tonight")
    assert r.intent == "out_of_scope" and r.service_request is None
