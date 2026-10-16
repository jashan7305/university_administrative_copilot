"""AdminNLP: the reusable entry point to the NMIMS administrative NLP pipeline.

Wraps the trained copilot (intent classification, sub-intent routing, NER, hybrid retrieval, grounded answer,
service request) behind four calls with validated, typed input and output:

    nlp = AdminNLP()                            # models load lazily on first use
    nlp.analyze("I lost my ID card")            # full pipeline -> AnalyzeOutput
    nlp.classify("...") / nlp.extract_entities("...") / nlp.retrieve("...")
    nlp.analyze_batch([...]) / nlp.info()

Every call also accepts the matching pydantic input model; plain strings get default options. On top of the
copilot it adds one thing: a `decision` that turns the confidence scores into what the caller should do next.
"""
from __future__ import annotations

import threading
from functools import lru_cache

from nlp.component.schemas import (
    MAX_BATCH, MAX_TEXT, SCHEMA_VERSION, AnalyzeInput, AnalyzeOutput, BatchInput, BatchOutput, ComponentInfo,
    Confidence, Decision, EntitiesOutput, EntityResult, IntentInput, IntentOutput, IntentResult, ModelInfo, Options,
    RetrieveInput, RetrieveOutput, RoutingResult, ScoredLabel, ServiceRequestResult, SourceResult, TextInput,
    Thresholds,
)

COMPONENT_VERSION = "1.0.0"
OUT_OF_SCOPE = "out_of_scope"
ROUTING_REVIEW_BELOW = Options.model_fields["routing_review_below"].default

INTENT_METHOD = {
    "v3": "top-class probability of the weighted blend of the fine-tuned transformer and the embedding ensemble",
    "v2": "top-class probability of the sentence-embedding + TF-IDF ensemble (logistic regression)",
}
ROUTING_METHOD = "routing model probability of the chosen sub-intent (flat or hierarchical, chosen on validation)"


class ModelsUnavailable(RuntimeError):
    """The trained models are not on disk."""


class AdminNLP:
    def __init__(self, copilot=None, pipeline: str | None = None):
        """copilot: an already-built nlp.pipeline.copilot.Copilot (tests, custom models); else loaded from models/.
        pipeline: 'v3' or 'v2' (default: v3 if its models exist, else v2; or the COPILOT_VERSION variable)."""
        self._copilot, self._pipeline = copilot, pipeline
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ loading
    @property
    def loaded(self) -> bool:
        return self._copilot is not None

    @property
    def copilot(self):
        if self._copilot is None:
            with self._lock:
                if self._copilot is None:
                    from nlp.pipeline.copilot import Copilot

                    try:
                        self._copilot = Copilot.from_disk(self._pipeline)
                    except FileNotFoundError as e:
                        raise ModelsUnavailable(f"{e}. Train the models first: python -m nlp.pipeline.train_eval "
                                                "(v2) and python -m nlp.pipeline.compare_week4 (v3)") from e
        return self._copilot

    def model_info(self) -> ModelInfo:
        return ModelInfo(component_version=COMPONENT_VERSION, schema_version=SCHEMA_VERSION,
                         pipeline=getattr(self.copilot, "version", "v2"))

    # ------------------------------------------------------------------ confidence
    def _intent_conf(self, score: float) -> Confidence:
        cp = self.copilot
        level = "low" if score < cp.threshold else ("medium" if score < cp.confirm_below else "high")
        return Confidence(score=round(score, 4), level=level, method=INTENT_METHOD[self.model_info().pipeline])

    @staticmethod
    def _routing_conf(score: float) -> Confidence:
        return Confidence(score=round(score, 4), level="high" if score >= ROUTING_REVIEW_BELOW else "low",
                          method=ROUTING_METHOD)

    def _intent_result(self, intent: str, conf: float, top: list[dict]) -> IntentResult:
        return IntentResult(intent=intent, in_scope=intent != OUT_OF_SCOPE, confidence=self._intent_conf(conf),
                            rejection_threshold=round(float(self.copilot.threshold), 4),
                            alternatives=[ScoredLabel(label=t["intent"], score=t["score"]) for t in top])

    def _routing_result(self, sub: str, score: float, alternatives: list[dict]) -> RoutingResult:
        return RoutingResult(sub_intent=sub, department=self.copilot.services[sub].department,
                             confidence=self._routing_conf(score),
                             alternatives=[ScoredLabel(label=a["sub_intent"], score=a["score"]) for a in alternatives])

    @staticmethod
    def _sources(raw: list[dict]) -> list[SourceResult]:
        top = max((s["score"] for s in raw), default=0.0)
        return [SourceResult(rank=i, doc_id=s["doc_id"], title=s["title"], page=s["page"], url=s["url"],
                             score=s["score"], relative_score=round(s["score"] / top, 4) if top > 0 else 0.0)
                for i, s in enumerate(raw, 1)]

    def _decide(self, intent: IntentResult, routing: RoutingResult | None, request: ServiceRequestResult | None,
                review_below: float) -> Decision:
        c = intent.confidence.score
        if not intent.in_scope:
            if c < intent.rejection_threshold:
                why = f"intent confidence {c:.2f} is below the rejection threshold {intent.rejection_threshold:.2f}"
            else:
                why = f"the intent model classified the query as out of scope (p = {c:.2f})"
            return Decision(action="out_of_scope", reason=f"Not an NMIMS administrative request: {why}.")
        if request.status == "needs_confirmation":
            alts = ", ".join(a.label for a in intent.alternatives[:2])
            return Decision(action="confirm_with_student",
                            reason=f"Intent confidence {c:.2f} is below {self.copilot.confirm_below:.2f}; ask the "
                                   f"student which service they mean (most likely: {alts}).")
        if routing.confidence.score < review_below:
            return Decision(action="refer_to_staff",
                            reason=f"Sub-intent confidence {routing.confidence.score:.2f} is below {review_below:.2f}; "
                                   f"a staff member at {routing.department} should check the routing.")
        if request.status == "needs_information":
            return Decision(action="ask_for_information",
                            reason="Routed, but details are missing: " + " ".join(request.missing_information))
        return Decision(action="auto_route", reason=f"Confident routing to {routing.department}; all required "
                                                    "details are present.")

    # ------------------------------------------------------------------ public calls
    def analyze(self, inp: AnalyzeInput | str) -> AnalyzeOutput:
        """Full pipeline: intent, routing, entities, retrieval, grounded answer, service request and decision."""
        inp = AnalyzeInput(text=inp) if isinstance(inp, str) else inp
        o = inp.options
        r = self.copilot.run(inp.text, k=o.k_sources, top_k=o.top_k)
        intent = self._intent_result(r.intent, r.confidence, r.top_intents)
        routing = self._routing_result(r.sub_intent, r.sub_intent_confidence, r.top_sub_intents) if intent.in_scope else None
        request = ServiceRequestResult(**r.service_request) if r.service_request else None
        return AnalyzeOutput(
            request_id=inp.request_id, query=inp.text, intent=intent, routing=routing,
            entities=[EntityResult(**e) for e in r.entities], answer=r.answer, answer_generator=r.generator,
            sources=self._sources(r.sources), service_request=request,
            decision=self._decide(intent, routing, request, o.routing_review_below),
            timings_ms=r.timings_ms if o.include_timings else None, model=self.model_info())

    def analyze_batch(self, inp: BatchInput | list[str | AnalyzeInput]) -> BatchOutput:
        if not isinstance(inp, BatchInput):
            inp = BatchInput(items=[AnalyzeInput(text=x) if isinstance(x, str) else x for x in inp])
        results = [self.analyze(x) for x in inp.items]
        return BatchOutput(results=results, count=len(results))

    def classify(self, inp: IntentInput | str) -> IntentOutput:
        """Intent (with out-of-scope rejection) and, when in scope, the sub-intent routing with alternatives."""
        inp = IntentInput(text=inp) if isinstance(inp, str) else inp
        name, conf, top = self.copilot.classify(inp.text, inp.top_k)
        intent = self._intent_result(name, conf, top)
        routing = None
        if intent.in_scope:
            ranked = self.copilot.route_scores(inp.text, name)
            routing = self._routing_result(ranked[0][0], ranked[0][1],
                                           [{"sub_intent": s, "score": round(p, 4)} for s, p in ranked[:inp.top_k]])
        return IntentOutput(request_id=inp.request_id, query=inp.text, intent=intent, routing=routing,
                            model=self.model_info())

    def extract_entities(self, inp: TextInput | str) -> EntitiesOutput:
        inp = TextInput(text=inp) if isinstance(inp, str) else inp
        ents = [EntityResult(**e.as_dict()) for e in self.copilot.extractor.extract(inp.text)]
        return EntitiesOutput(request_id=inp.request_id, query=inp.text, entities=ents, model=self.model_info())

    def retrieve(self, inp: RetrieveInput | str) -> RetrieveOutput:
        """Evidence passages from the NMIMS knowledge base; boosted by `intent` if given, else by the predicted one."""
        inp = RetrieveInput(text=inp) if isinstance(inp, str) else inp
        intent, sub = inp.intent, None
        if intent is None:
            predicted, _, _ = self.copilot.classify(inp.text)
            if predicted != OUT_OF_SCOPE:
                intent, sub = predicted, self.copilot.route(inp.text, predicted)
        hits = self.copilot.retriever.search(inp.text, k=inp.k, intent=intent, sub_intent=sub)
        raw = [{"doc_id": h.doc_id, "title": h.chunk.title, "page": h.chunk.section, "url": h.chunk.url,
                "score": round(h.score, 4)} for h in hits]
        return RetrieveOutput(request_id=inp.request_id, query=inp.text, intent_used=intent, sources=self._sources(raw),
                              model=self.model_info())

    def info(self) -> ComponentInfo:
        from nlp.knowledge.store import load_chunks
        from nlp.ner.extractor import RULES

        cp = self.copilot
        ner = cp.extractor
        learned = ner.id2label.values() if hasattr(ner, "id2label") else ner.crf.crf.classes_
        labels = sorted({t.split("-", 1)[1] for t in learned if t != "O"} | {label for label, _ in RULES})
        return ComponentInfo(
            name="NMIMS University Administrative NLP", model=self.model_info(),
            intents=sorted(str(c) for c in cp.intent_model.classes_), sub_intents=len(cp.services),
            entity_labels=labels, knowledge_base_passages=len(load_chunks()),
            thresholds=Thresholds(intent_rejection=round(float(cp.threshold), 4), intent_confirm_below=cp.confirm_below,
                                  routing_review_below=ROUTING_REVIEW_BELOW),
            confidence_levels={
                "intent.high": f">= {cp.confirm_below}: routed without confirmation",
                "intent.medium": f"[{cp.threshold:.2f}, {cp.confirm_below}): in scope, but the student is asked to confirm",
                "intent.low": f"< {cp.threshold:.2f}: rejected as out of scope",
                "routing.high": f">= {ROUTING_REVIEW_BELOW}: Week 5 validation, >= 90% routing accuracy at this level",
                "routing.low": f"< {ROUTING_REVIEW_BELOW}: refer to staff (decision.action = refer_to_staff)",
            },
            limits={"max_text_chars": MAX_TEXT, "max_batch_items": MAX_BATCH})


@lru_cache(maxsize=1)
def get_component() -> AdminNLP:
    """Process-wide shared instance (models are loaded once, on first use)."""
    return AdminNLP()
