"""NMIMS knowledge base for retrieval and the service catalogue for structured requests.

  load_chunks()    page-level passages from data/final/nmims_knowledge_base.csv (built by dataset phase 1 from
                   official NMIMS pages and PDFs); each chunk keeps its source URL, page and the ids of the
                   verified facts that cite it
  load_services()  one Service per sub-intent, assembled from the verified facts in data/annotations/facts.yaml:
                   routing department, documents named by the sources, next action, and the supporting facts
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from nlp import DATA


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str            # source_id + page: the unit retrieval results are aggregated to
    intent: str            # top-level intent of the facts cited on this page (else the source category)
    sub_intent: str
    title: str
    section: str
    text: str
    url: str
    fact_ids: tuple[str, ...] = ()
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def indexed_text(self) -> str:
        """Text the retrievers see: the source title gives each passage its context."""
        return f"{self.title}. {self.text}"


@dataclass(frozen=True)
class Service:
    sub_intent: str
    intent: str
    department: str
    required_documents: tuple[str, ...]
    next_action: str
    fact_ids: tuple[str, ...]
    facts: tuple[str, ...]
    sources: tuple[str, ...]


@lru_cache(maxsize=1)
def load_chunks() -> tuple[Chunk, ...]:
    kb = pd.read_csv(DATA / "final" / "nmims_knowledge_base.csv", keep_default_na=False)
    return tuple(
        Chunk(chunk_id=r.document_id, doc_id=f"{r.source_id}-P{int(r.page):03d}", intent=r.category,
              sub_intent=r.sub_category, title=r.title, section=f"p. {r.page}", text=r.content, url=r.source_url,
              fact_ids=tuple(f for f in str(r.fact_ids).split("|") if f),
              meta={"current_or_historical": r.current_or_historical, "academic_year": r.academic_year})
        for r in kb.itertuples())


@lru_cache(maxsize=1)
def load_services() -> dict[str, Service]:
    from dataset.config import SOURCES, load_facts, next_actions

    facts, actions = load_facts(), next_actions()
    by_sub: dict[str, list[dict]] = {}
    for f in facts.values():
        if not f["historical"]:
            by_sub.setdefault(f["sub_intent"], []).append(f)
    services = {}
    for sub, fs in by_sub.items():
        dept = Counter(f["department"] for f in fs).most_common(1)[0][0]
        docs = tuple(dict.fromkeys(d for f in fs for d in f["documents"]))
        services[sub] = Service(sub, fs[0]["intent"], dept, docs, actions.get(sub, ""),
                                tuple(f["fact_id"] for f in fs), tuple(f["fact"] for f in fs),
                                tuple(dict.fromkeys(SOURCES[f["source_id"]].url for f in fs)))
    return services
