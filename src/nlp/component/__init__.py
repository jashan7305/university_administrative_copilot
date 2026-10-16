"""Reusable NLP component for NMIMS administrative requests (Week 6).

    from nlp.component import AdminNLP
    out = AdminNLP().analyze("I need a bonafide certificate for a bank loan, urgently")
    out.intent.intent, out.intent.confidence.score, out.decision.action, out.model_dump_json()

Documentation: docs/nlp_component.md. Also served over HTTP (src/backend, /v1/*) and as a CLI (python -m nlp.component).
"""
from nlp.component.engine import COMPONENT_VERSION, AdminNLP, ModelsUnavailable, get_component
from nlp.component.schemas import (
    SCHEMA_VERSION, AnalyzeInput, AnalyzeOutput, BatchInput, BatchOutput, ComponentInfo, EntitiesOutput, IntentInput,
    IntentOutput, Options, RetrieveInput, RetrieveOutput, TextInput,
)

__version__ = COMPONENT_VERSION
__all__ = ["AdminNLP", "ModelsUnavailable", "get_component", "COMPONENT_VERSION", "SCHEMA_VERSION", "AnalyzeInput",
           "AnalyzeOutput", "BatchInput", "BatchOutput", "ComponentInfo", "EntitiesOutput", "IntentInput",
           "IntentOutput", "Options", "RetrieveInput", "RetrieveOutput", "TextInput"]
