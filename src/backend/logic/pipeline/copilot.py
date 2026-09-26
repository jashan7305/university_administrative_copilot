"""Copilot NLP pipeline, version 1.

    query ──► intent classification ──► out of scope? ──► polite fallback
                    │
                    ├─► entity extraction (CRF + rules)
                    ├─► hybrid retrieval over the procedure KB (intent-boosted)
                    ├─► answer generation (extractive by default; Claude optional, grounded in retrieved text)
                    └─► structured service request (department, documents, fee, next action, missing info)
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import joblib

from logic.knowledge.store import Procedure, load_procedures
from logic.ner.extractor import EntityExtractor
from logic.retrieval.search import Hit, HybridRetriever

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
INTENT_PATH = MODEL_DIR / "intent_v1.joblib"
NER_PATH = MODEL_DIR / "ner_crf.joblib"

# Which extracted entities each intent needs before a request can be actioned, and how to ask for them.
REQUIRED_SLOTS: dict[str, list[tuple[str, str]]] = {
    "certificates": [("DOCUMENT", "Which certificate do you need (for example bonafide, NOC or migration)?"),
                     ("PURPOSE", "What is the certificate for (for example a bank loan, visa or internship)?")],
    "transcripts": [("DOCUMENT", "Which document do you need (official transcript, duplicate marksheet or CGPA certificate)?")],
    "id_cards": [],
    "hostel": [],
    "examinations": [("EXAM_TYPE|SUBJECT", "Which exam or subject is this about?")],
    "fees": [("FEE_TYPE", "Which fee is this about (tuition, hostel, exam and so on)?")],
    "scholarships": [("SCHOLARSHIP", "Which scholarship are you asking about (merit, government, need-based and so on)?")],
    "attendance": [("SUBJECT|PERCENTAGE", "Which subject is affected, and what is your current attendance?")],
    "academic_registration": [("COURSE_TYPE|TERM", "Which course or semester is this for?")],
}
URGENT = re.compile(r"\b(urgent|urgently|asap|immediately|today|tomorrow|as soon as possible|emergency)\b", re.I)

OOS_REPLY = ("I can help with university administrative requests: certificates, transcripts, ID cards, hostel, "
             "examinations, fees, scholarships, attendance and course registration. Your question seems to be "
             "outside these areas. Could you rephrase it, or contact the student helpdesk?")


@dataclass
class ServiceRequest:
    intent: str
    department: str
    procedure_id: str
    procedure_title: str
    required_documents: list[str]
    fee: str
    processing_time: str
    next_action: str
    extracted_details: dict[str, list[str]]
    missing_information: list[str]
    priority: str
    status: str  # ready | needs_information | needs_confirmation


@dataclass
class CopilotResponse:
    query: str
    intent: str
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
    """Deterministic, fully grounded answer: procedure metadata plus the best-matching passage."""

    name = "extractive"

    def generate(self, query: str, proc: Procedure, hits: list[Hit]) -> str:
        passage = next((h.chunk.text for h in hits if h.doc_id == proc.id), "")
        sentences = re.split(r"(?<=[.!?])\s+", passage)
        excerpt = " ".join(sentences[:3])
        docs = "; ".join(proc.required_documents)
        return (f"{proc.title} is handled by the {proc.department}. {excerpt}\n\n"
                f"Required documents: {docs}.\nFee: {proc.fee}. Processing time: {proc.processing_time}.\n"
                f"Next step: {proc.next_action}.")


class ClaudeGenerator:
    """Optional abstractive generator (set COPILOT_GENERATOR=claude and Anthropic credentials).

    The model may only use the retrieved procedure text; on any API error or refusal the pipeline falls
    back to the extractive answer, so the service never depends on the external call.
    """

    name = "claude"
    SYSTEM = ("You are the University Administrative Copilot. Answer the student's question using ONLY the "
              "procedure excerpts provided. Be concise (at most 120 words), list required documents and the next "
              "step, and name the department. If the excerpts do not answer the question, say so and suggest "
              "contacting the named department. Do not invent fees, dates or rules.")

    def __init__(self, model: str = "claude-opus-5"):
        import anthropic  # optional dependency

        self.anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def generate(self, query: str, proc: Procedure, hits: list[Hit]) -> str:
        context = "\n\n".join(f"[{h.doc_id} / {h.chunk.section}]\n{h.chunk.text}" for h in hits[:3])
        meta = (f"Procedure: {proc.title}\nDepartment: {proc.department}\nRequired documents: "
                f"{'; '.join(proc.required_documents)}\nFee: {proc.fee}\nProcessing time: {proc.processing_time}\n"
                f"Next action: {proc.next_action}")
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=1024,
            betas=["server-side-fallback-2026-06-01"],
            fallbacks=[{"model": "claude-opus-4-8"}],
            output_config={"effort": "low"},
            system=self.SYSTEM,
            messages=[{"role": "user", "content": f"{meta}\n\nExcerpts:\n{context}\n\nQuestion: {query}"}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("generation refused")
        return "".join(b.text for b in response.content if b.type == "text").strip()


# --------------------------------------------------------------------------- pipeline

class Copilot:
    def __init__(self, intent_bundle: dict, extractor: EntityExtractor, retriever: HybridRetriever,
                 generator=None, confirm_below: float = 0.5):
        self.intent_model = intent_bundle["model"]
        self.threshold = intent_bundle["threshold"]
        self.extractor = extractor
        self.retriever = retriever
        self.procedures = load_procedures()
        self.extractive = ExtractiveGenerator()
        self.generator = generator or self.extractive
        self.confirm_below = confirm_below

    @classmethod
    def from_disk(cls) -> "Copilot":
        generator = None
        if os.getenv("COPILOT_GENERATOR", "").lower() == "claude":
            generator = ClaudeGenerator()
        return cls(joblib.load(INTENT_PATH), EntityExtractor.load(NER_PATH), HybridRetriever(), generator)

    def _classify(self, text: str):
        proba = self.intent_model.predict_proba([text])[0]
        ranked = sorted(zip(self.intent_model.classes_, proba), key=lambda x: -x[1])
        intent, conf = ranked[0]
        if conf < self.threshold:
            intent = "out_of_scope"
        return intent, float(conf), [{"intent": i, "score": round(float(p), 4)} for i, p in ranked[:3]]

    def _service_request(self, text, intent, conf, proc: Procedure, entities) -> ServiceRequest:
        details: dict[str, list[str]] = {}
        for e in entities:
            details.setdefault(e.label, [])
            if e.text not in details[e.label]:
                details[e.label].append(e.text)
        missing = [q for labels, q in REQUIRED_SLOTS.get(intent, []) if not any(l in details for l in labels.split("|"))]
        dated_urgent = any(re.search(r"today|tomorrow", v, re.I) for v in details.get("DATE", []))
        priority = "high" if URGENT.search(text) or dated_urgent else "normal"
        status = "needs_confirmation" if conf < self.confirm_below else ("needs_information" if missing else "ready")
        return ServiceRequest(intent, proc.department, proc.id, proc.title, list(proc.required_documents), proc.fee,
                              proc.processing_time, proc.next_action, details, missing, priority, status)

    def run(self, text: str, k: int = 3) -> CopilotResponse:
        t = {}
        t0 = time.perf_counter()
        intent, conf, top = self._classify(text)
        t["intent"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        entities = self.extractor.extract(text)
        t["ner"] = (time.perf_counter() - t0) * 1000

        if intent == "out_of_scope":
            t["total"] = sum(t.values())
            return CopilotResponse(text, intent, round(conf, 4), top, [e.as_dict() for e in entities], OOS_REPLY, [],
                                   None, "fallback", {k2: round(v, 2) for k2, v in t.items()})

        t0 = time.perf_counter()
        hits = self.retriever.search(text, k=k, intent=intent)
        t["retrieval"] = (time.perf_counter() - t0) * 1000
        # Prefer the best document of the predicted intent; fall back to the overall best hit.
        proc = self.procedures[next((h.doc_id for h in hits if h.chunk.intent == intent), hits[0].doc_id)]

        t0 = time.perf_counter()
        generator_used = self.generator.name
        try:
            answer = self.generator.generate(text, proc, hits)
        except Exception:  # external generator failure must not break the request
            answer, generator_used = self.extractive.generate(text, proc, hits), "extractive (fallback)"
        t["generation"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        request = self._service_request(text, intent, conf, proc, entities)
        t["service_request"] = (time.perf_counter() - t0) * 1000
        t["total"] = sum(t.values())

        sources = [{"doc_id": h.doc_id, "title": h.chunk.title, "section": h.chunk.section, "score": round(h.score, 4)}
                   for h in hits]
        return CopilotResponse(text, intent, round(conf, 4), top, [e.as_dict() for e in entities], answer, sources,
                               asdict(request), generator_used, {k2: round(v, 2) for k2, v in t.items()})


@lru_cache(maxsize=1)
def get_copilot() -> Copilot:
    return Copilot.from_disk()
