"""Copilot NLP pipeline, version 2 (NMIMS data).

    query ──► intent classification (12 intents + out_of_scope, confidence rejection) ──► out of scope? ──► fallback
                    │
                    ├─► sub-intent routing (95 sub-intents, restricted to the predicted intent)
                    ├─► entity extraction (CRF trained on NMIMS annotations + regex rules)
                    ├─► hybrid retrieval over 727 official NMIMS passages (BM25 + embeddings, intent-boosted)
                    ├─► answer = query-focused extractive summary (MMR) of verified facts + retrieved passages,
                    │   with source citations (optional Claude rewrite, grounded in the same evidence)
                    └─► structured service request: department, documents, next action, extracted details,
                        missing information, priority, status, sources
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import joblib
import numpy as np

from nlp import MODELS
from nlp.knowledge.store import Service, load_services
from nlp.ner.extractor import EntityExtractor
from nlp.pipeline.summarize import MMRSummarizer
from nlp.retrieval.search import Hit, HybridRetriever

INTENT_PATH = MODELS / "intent_v2.joblib"
SUBINTENT_PATH = MODELS / "subintent_v2.joblib"
NER_PATH = MODELS / "ner_crf_v2.joblib"
V3_DIR = MODELS / "week4"

# Details a request needs before it can be actioned (entity types from the NMIMS NER schema), and how to ask.
REQUIRED_SLOTS: dict[str, list[tuple[str, str]]] = {
    "CERTIFICATES": [("CERTIFICATE|DOCUMENT", "Which certificate or document do you need?")],
    "TRANSCRIPTS": [("DOCUMENT|CERTIFICATE", "Which document do you need (transcript, grade sheet, percentage letter)?")],
    "EXAMINATIONS": [("EXAMINATION|SUBJECT", "Which examination or subject is this about?")],
    "FEES": [("FEE", "Which fee or deposit is this about?")],
    "SCHOLARSHIPS": [("SCHOLARSHIP", "Which scholarship are you asking about?")],
    "ATTENDANCE": [("SUBJECT|PERCENTAGE|SEMESTER", "Which subject or semester is affected, and what is your attendance?")],
    "ACADEMIC_REGISTRATION": [("PROGRAM|SEMESTER", "Which programme and semester are you in?")],
}
URGENT = re.compile(r"\b(urgent|urgently|asap|immediately|today|tomorrow|tonight|as soon as possible|emergency)\b", re.I)
OOS_REPLY = ("I can help with NMIMS administrative requests: certificates, transcripts, ID cards, hostel, examinations, "
             "fees, scholarships, attendance, registration, library, student services and welfare. Your question seems "
             "to be outside these areas. Could you rephrase it, or contact the student helpdesk?")


@dataclass
class ServiceRequest:
    intent: str
    sub_intent: str
    department: str
    required_documents: list[str]
    next_action: str
    extracted_details: dict[str, list[str]]
    missing_information: list[str]
    priority: str
    status: str            # ready | needs_information | needs_confirmation
    supporting_facts: list[str]
    source_urls: list[str]


@dataclass
class CopilotResponse:
    query: str
    intent: str
    sub_intent: str
    confidence: float
    top_intents: list[dict]
    entities: list[dict]
    answer: str
    sources: list[dict]
    service_request: dict | None
    generator: str
    timings_ms: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- answer generation

class ExtractiveGenerator:
    """Grounded answer: MMR summary of the service's verified facts and the retrieved passages."""

    name = "extractive_mmr"

    def __init__(self):
        self.summarizer = MMRSummarizer()

    def generate(self, query: str, service: Service, hits: list[Hit]) -> str:
        summary = self.summarizer.summarize(query, list(service.facts) + [h.chunk.text for h in hits])
        docs = "; ".join(service.required_documents) or "none stated in the sources"
        return (f"{summary}\n\nDepartment: {service.department}.\nRequired documents: {docs}.\n"
                f"Next step: {service.next_action}.")


class ClaudeGenerator:
    """Optional abstractive answer (COPILOT_GENERATOR=claude + Anthropic credentials), limited to the same evidence.
    Any API error or refusal falls back to the extractive answer."""

    name = "claude"
    SYSTEM = ("You are the NMIMS University Administrative Copilot. Answer using ONLY the verified facts and source "
              "excerpts provided. Be concise (at most 120 words), name the department, list required documents and "
              "the next step. If the evidence does not answer the question, say so. Never invent fees, dates or rules.")

    def __init__(self, model: str = "claude-opus-5"):
        import anthropic  # optional dependency

        self.client = anthropic.Anthropic()
        self.model = model

    def generate(self, query: str, service: Service, hits: list[Hit]) -> str:
        facts = "\n".join(f"- {f}" for f in service.facts)
        excerpts = "\n\n".join(f"[{h.chunk.title}, {h.chunk.section}]\n{h.chunk.text}" for h in hits[:3])
        meta = (f"Department: {service.department}\nRequired documents: {'; '.join(service.required_documents)}\n"
                f"Next action: {service.next_action}")
        response = self.client.beta.messages.create(
            model=self.model, max_tokens=1024, betas=["server-side-fallback-2026-06-01"],
            fallbacks=[{"model": "claude-opus-4-8"}], output_config={"effort": "low"}, system=self.SYSTEM,
            messages=[{"role": "user", "content": f"{meta}\n\nVerified facts:\n{facts}\n\nExcerpts:\n{excerpts}\n\nQuestion: {query}"}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("generation refused")
        return "".join(b.text for b in response.content if b.type == "text").strip()


# --------------------------------------------------------------------------- pipeline

class Copilot:
    def __init__(self, intent_bundle: dict, subintent_bundle: dict, extractor: EntityExtractor,
                 retriever: HybridRetriever, generator=None, confirm_below: float = 0.5):
        self.intent_model, self.threshold = intent_bundle["model"], intent_bundle["threshold"]
        self.sub_model = subintent_bundle["model"]
        self.route_strategy = subintent_bundle.get("strategy", "hierarchical")
        self.extractor, self.retriever = extractor, retriever
        self.services = load_services()
        self.extractive = ExtractiveGenerator()
        self.generator = generator or self.extractive
        self.confirm_below = confirm_below

    @classmethod
    def from_disk(cls, version: str | None = None) -> "Copilot":
        """v2 = Week 3 pipeline (TF-IDF/embedding ensemble, CRF, hybrid retrieval);
        v3 = Week 4 improved pipeline (fine-tuned transformer router, transformer NER, hybrid retrieval with the
        fine-tuned vector model; cross-encoder re-ranking was evaluated but scored lower, so it is not used). Default: v3 if its models exist, else v2 (override with COPILOT_VERSION)."""
        generator = ClaudeGenerator() if os.getenv("COPILOT_GENERATOR", "").lower() == "claude" else None
        version = version or os.getenv("COPILOT_VERSION") or ("v3" if (V3_DIR / "router").exists() else "v2")
        if version == "v2":
            return cls(joblib.load(INTENT_PATH), joblib.load(SUBINTENT_PATH), EntityExtractor.load(NER_PATH),
                       HybridRetriever(), generator)
        from dataset.config import SUB_TO_INTENT
        from nlp.intent.finetune import JointIntentAdapter, TransformerClassifier
        from nlp.ner.transformer_ner import TransformerNER
        from nlp.retrieval.search import BM25Retriever, DenseRetriever

        from nlp.intent.finetune import ProbabilityBlend

        meta = json.loads((V3_DIR / "meta.json").read_text())
        transformer = TransformerClassifier.load(V3_DIR / "router")
        v2i, v2s = joblib.load(INTENT_PATH)["model"], joblib.load(SUBINTENT_PATH)["model"]
        router = ProbabilityBlend([transformer, v2s], [meta["w_sub"], 1 - meta["w_sub"]])
        intent_model = ProbabilityBlend([JointIntentAdapter(transformer, SUB_TO_INTENT), v2i], [meta["w_int"], 1 - meta["w_int"]])
        dense = DenseRetriever(model=str(V3_DIR / "dense"))
        retriever = HybridRetriever(BM25Retriever(dense.chunks), dense)
        obj = cls({"model": intent_model, "threshold": meta["threshold"]},
                  {"model": router, "strategy": "flat"}, TransformerNER.load(V3_DIR / "ner"), retriever, generator)
        obj.version = "v3"
        return obj

    def classify(self, text: str) -> tuple[str, float, list[dict]]:
        proba = self.intent_model.predict_proba([text])[0]
        order = np.argsort(-proba)
        intent, conf = self.intent_model.classes_[order[0]], float(proba[order[0]])
        if conf < self.threshold:
            intent = "out_of_scope"
        return intent, conf, [{"intent": self.intent_model.classes_[i], "score": round(float(proba[i]), 4)} for i in order[:3]]

    def route(self, text: str, intent: str) -> str:
        """Most probable sub-intent; with the hierarchical strategy only those under the predicted intent count.
        The strategy (flat or hierarchical) is chosen on validation data in train_eval.py."""
        proba = self.sub_model.predict_proba([text])[0]
        classes = self.sub_model.classes_
        known = [i for i, s in enumerate(classes) if s in self.services]
        allowed = [i for i in known if self.services[classes[i]].intent == intent]
        pool = allowed if (self.route_strategy == "hierarchical" and allowed) else known
        return classes[max(pool, key=lambda i: proba[i])]

    def _service_request(self, text: str, conf: float, service: Service, entities) -> ServiceRequest:
        details: dict[str, list[str]] = {}
        for e in entities:
            details.setdefault(e.label, [])
            if e.text not in details[e.label]:
                details[e.label].append(e.text)
        missing = [q for labels, q in REQUIRED_SLOTS.get(service.intent, [])
                   if not any(l in details for l in labels.split("|"))]
        priority = "high" if URGENT.search(text) else "normal"
        status = "needs_confirmation" if conf < self.confirm_below else ("needs_information" if missing else "ready")
        return ServiceRequest(service.intent, service.sub_intent, service.department, list(service.required_documents),
                              service.next_action, details, missing, priority, status, list(service.fact_ids),
                              list(service.sources))

    def run(self, text: str, k: int = 3) -> CopilotResponse:
        t, t0 = {}, time.perf_counter()
        intent, conf, top = self.classify(text)
        t["intent"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        entities = self.extractor.extract(text)
        t["ner"] = (time.perf_counter() - t0) * 1000
        if intent == "out_of_scope":
            t["total"] = sum(t.values())
            return CopilotResponse(text, intent, "", round(conf, 4), top, [e.as_dict() for e in entities], OOS_REPLY,
                                   [], None, "fallback", {a: round(b, 2) for a, b in t.items()})

        t0 = time.perf_counter()
        sub = self.route(text, intent)
        service = self.services[sub]
        t["routing"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        hits = self.retriever.search(text, k=k, intent=intent, sub_intent=sub)
        t["retrieval"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        used = self.generator.name
        try:
            answer = self.generator.generate(text, service, hits)
        except Exception:  # an external generator failure must not break the request
            answer, used = self.extractive.generate(text, service, hits), "extractive_mmr (fallback)"
        t["generation"] = (time.perf_counter() - t0) * 1000

        request = self._service_request(text, conf, service, entities)
        t["total"] = sum(t.values())
        sources = [{"doc_id": h.doc_id, "title": h.chunk.title, "page": h.chunk.section, "url": h.chunk.url,
                    "score": round(h.score, 4)} for h in hits]
        return CopilotResponse(text, intent, sub, round(conf, 4), top, [e.as_dict() for e in entities], answer, sources,
                               asdict(request), used, {a: round(b, 2) for a, b in t.items()})


@lru_cache(maxsize=1)
def get_copilot() -> Copilot:
    return Copilot.from_disk()
