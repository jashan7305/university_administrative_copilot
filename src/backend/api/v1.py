"""Versioned HTTP API of the NLP component (/v1). Request and response bodies are the pydantic models in
nlp.component.schemas, so the OpenAPI page (/docs) documents every field. Errors use ErrorOutput."""
from fastapi import APIRouter, Depends

from nlp.component import COMPONENT_VERSION, SCHEMA_VERSION, AdminNLP, get_component
from nlp.component.schemas import (
    AnalyzeInput, AnalyzeOutput, BatchInput, BatchOutput, ComponentInfo, EntitiesOutput, ErrorOutput, IntentInput,
    IntentOutput, RetrieveInput, RetrieveOutput, TextInput,
)

ERRORS = {422: {"model": ErrorOutput, "description": "Invalid input"},
          503: {"model": ErrorOutput, "description": "Models not trained / not on disk"}}

router = APIRouter(prefix="/v1", tags=["nlp component v1"], responses=ERRORS)


def nlp() -> AdminNLP:
    """Dependency, so tests and other deployments can inject their own component."""
    return get_component()


@router.get("/health")
def health(component: AdminNLP = Depends(nlp)) -> dict:
    """Liveness; does not load the models (`models_loaded` turns true after the first request)."""
    return {"status": "ok", "component_version": COMPONENT_VERSION, "schema_version": SCHEMA_VERSION,
            "models_loaded": component.loaded}


@router.get("/info", response_model=ComponentInfo)
def info(component: AdminNLP = Depends(nlp)):
    """Labels, thresholds, confidence bands and limits of the loaded models."""
    return component.info()


@router.post("/analyze", response_model=AnalyzeOutput)
def analyze(body: AnalyzeInput, component: AdminNLP = Depends(nlp)):
    """Full pipeline: intent, routing, entities, evidence, grounded answer, service request, decision."""
    return component.analyze(body)


@router.post("/analyze/batch", response_model=BatchOutput)
def analyze_batch(body: BatchInput, component: AdminNLP = Depends(nlp)):
    return component.analyze_batch(body)


@router.post("/intent", response_model=IntentOutput)
def intent(body: IntentInput, component: AdminNLP = Depends(nlp)):
    """Intent with out-of-scope rejection, plus sub-intent routing when in scope."""
    return component.classify(body)


@router.post("/entities", response_model=EntitiesOutput)
def entities(body: TextInput, component: AdminNLP = Depends(nlp)):
    return component.extract_entities(body)


@router.post("/retrieve", response_model=RetrieveOutput)
def retrieve(body: RetrieveInput, component: AdminNLP = Depends(nlp)):
    """Evidence passages from the official NMIMS knowledge base."""
    return component.retrieve(body)
