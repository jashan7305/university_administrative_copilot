"""Dependency-light fallback used by the browser demo when trained artefacts are absent.

The demo still uses the repository's verified service catalogue and knowledge
base. It is intentionally not used by the reusable component or CLI, which
continue to report missing model artefacts explicitly.
"""
from __future__ import annotations

import re
import time

from nlp.knowledge.store import load_chunks, load_services
from nlp.ner.extractor import RuleExtractor
from nlp.pipeline.copilot import CopilotResponse, OOS_REPLY, ServiceRequest
from nlp.retrieval.search import Hit


INTENT_KEYWORDS = {
    "ID_CARDS": ("id card", "identity card", "lost card", "duplicate card", "card not working"),
    "CERTIFICATES": ("certificate", "bonafide", "migration", "degree"),
    "TRANSCRIPTS": ("transcript", "grade sheet", "marksheet", "statement of marks", "percentage letter"),
    "HOSTEL": ("hostel", "residence", "room allocation"),
    "EXAMINATIONS": ("exam", "examination", "revaluation", "backlog", "re-exam", "hall ticket"),
    "FEES": ("fee", "fees", "payment", "refund", "receipt"),
    "SCHOLARSHIPS": ("scholarship", "financial aid"),
    "ATTENDANCE": ("attendance", "absent", "medical leave"),
    "ACADEMIC_REGISTRATION": ("register", "registration", "enrol", "enroll", "elective", "semester"),
    "LIBRARY": ("library", "book", "borrow", "renew"),
    "STUDENT_SERVICES": ("portal", "placement", "international", "facility"),
    "STUDENT_WELFARE": ("harassment", "grievance", "counselling", "counseling", "ragging", "welfare"),
}

SUBINTENT_KEYWORDS = {
    "DUPLICATE_ID_CARD": ("lost", "stolen", "duplicate", "damaged", "replacement"),
    "ID_CARD_NOT_WORKING": ("not working", "blocked", "access", "biometric"),
    "TRANSCRIPT": ("transcript",),
    "COURSE_REGISTRATION": ("register", "registration", "enrol", "enroll"),
    "ELECTIVE_SELECTION": ("elective", "choose"),
    "LOW_ATTENDANCE": ("low", "below", "%", "attendance"),
    "FEE_PAYMENT": ("pay", "payment", "fee"),
    "SCHOLARSHIP_APPLICATION": ("apply", "application", "scholarship"),
    "HOSTEL_APPLICATION": ("apply", "application", "hostel"),
}


class DemoCopilot:
    threshold = 0.35
    confirm_below = 0.5
    version = "demo"

    def __init__(self):
        self.services = load_services()
        self.extractor = RuleExtractor()

    @staticmethod
    def _score(text: str, words: tuple[str, ...]) -> int:
        return sum(1 for word in words if word in text)

    def _intent(self, text: str) -> tuple[str, float]:
        scores = {intent: self._score(text, words) for intent, words in INTENT_KEYWORDS.items()}
        intent, score = max(scores.items(), key=lambda item: item[1])
        return (intent, min(0.94, 0.68 + score * 0.08)) if score else ("out_of_scope", 0.9)

    def _subintent(self, text: str, intent: str) -> tuple[str | None, float]:
        candidates = {key: value for key, value in self.services.items() if value.intent == intent}
        if not candidates:
            return None, 0.0
        scored = sorted(((self._score(text, SUBINTENT_KEYWORDS.get(key, ())) , key) for key in candidates),
                        reverse=True)
        score, key = scored[0]
        if score == 0:
            key = next(iter(candidates))
        return key, min(0.9, 0.64 + score * 0.1)

    def _hits(self, text: str, intent: str) -> list[dict]:
        terms = set(re.findall(r"[a-z0-9]+", text))
        ranked = []
        for chunk in load_chunks():
            if chunk.intent != intent:
                continue
            overlap = len(terms & set(re.findall(r"[a-z0-9]+", chunk.text.lower())))
            ranked.append((overlap, chunk))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [{"doc_id": chunk.doc_id, "title": chunk.title, "page": chunk.section, "url": chunk.url,
                 "score": float(max(score, 1)), "rank": i}
                for i, (score, chunk) in enumerate(ranked[:3], 1)]

    def run(self, text: str, k: int = 3, top_k: int = 3) -> CopilotResponse:
        started = time.perf_counter()
        clean = text.lower()
        intent, intent_conf = self._intent(clean)
        entities = [entity.as_dict() for entity in self.extractor.extract(text)]
        if intent == "out_of_scope":
            return CopilotResponse(text, intent, "", intent_conf, [{"intent": intent, "score": intent_conf}],
                                   entities, OOS_REPLY, [], None, "demo_rules",
                                   {"total": round((time.perf_counter() - started) * 1000, 2)})

        sub, sub_conf = self._subintent(clean, intent)
        service = self.services[sub]
        details = {}
        for entity in entities:
            details.setdefault(entity["label"], []).append(entity["text"])
        missing = []
        status = "ready"
        request = ServiceRequest(service.intent, service.sub_intent, service.department,
                                 list(service.required_documents), service.next_action, details, missing,
                                 "high" if re.search(r"\b(urgent|asap|immediately|today)\b", clean) else "normal",
                                 status, list(service.fact_ids), list(service.sources))
        hits = self._hits(clean, intent)[:k]
        answer = (f"Based on the official NMIMS guidance, contact {service.department}. "
                  f"{service.next_action}. Required documents: "
                  f"{'; '.join(service.required_documents) if service.required_documents else 'confirm with the department'}.")
        return CopilotResponse(
            text, intent, sub, intent_conf, [{"intent": intent, "score": intent_conf}],
            entities, answer, hits, request.__dict__, "demo_rules",
            {"total": round((time.perf_counter() - started) * 1000, 2)},
            sub_conf, [{"sub_intent": sub, "score": sub_conf}],
        )
