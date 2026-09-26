"""Knowledge base of administrative procedures: loading, metadata and chunking.

Each procedure is a Markdown file with YAML front matter (department, required documents, fee,
processing time, next action). Front matter feeds the structured service request; the body is
chunked by section for retrieval.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

KB_DIR = Path(__file__).resolve().parent / "kb"


@dataclass(frozen=True)
class Procedure:
    id: str
    intent: str
    title: str
    department: str
    required_documents: tuple[str, ...]
    fee: str
    processing_time: str
    next_action: str
    body: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    intent: str
    title: str
    section: str
    text: str
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def indexed_text(self) -> str:
        """Text the retrievers see: title and section heading give each chunk its context."""
        return f"{self.title}. {self.section}. {self.text}"


def _parse(path: Path) -> Procedure:
    raw = path.read_text()
    _, fm, body = raw.split("---", 2)
    m = yaml.safe_load(fm)
    return Procedure(
        id=m["id"], intent=m["intent"], title=m["title"], department=m["department"],
        required_documents=tuple(m.get("required_documents", [])), fee=str(m.get("fee", "")),
        processing_time=str(m.get("processing_time", "")), next_action=m.get("next_action", ""), body=body.strip(),
    )


@lru_cache(maxsize=1)
def load_procedures() -> dict[str, Procedure]:
    procs = {p.id: p for p in (_parse(f) for f in sorted(KB_DIR.glob("*.md")))}
    if not procs:
        raise FileNotFoundError(f"No procedures found in {KB_DIR}")
    return procs


def chunk_procedure(p: Procedure, max_words: int = 120) -> list[Chunk]:
    """Split on '## ' sections; the preamble becomes an 'Overview' chunk; long sections are windowed."""
    parts = re.split(r"^## +(.+)$", p.body, flags=re.M)
    sections = [("Overview", re.sub(r"^# .+\n", "", parts[0]).strip())]
    sections += [(parts[i].strip(), parts[i + 1].strip()) for i in range(1, len(parts), 2)]
    chunks = []
    for s_idx, (heading, text) in enumerate(sections):
        words = text.split()
        if not words:
            continue
        step = max_words - 20  # 20-word overlap between windows of a long section
        for w_idx, start in enumerate(range(0, max(1, len(words) - 20), step)):
            piece = " ".join(words[start:start + max_words])
            chunks.append(Chunk(f"{p.id}#{s_idx}.{w_idx}", p.id, p.intent, p.title, heading, piece))
    return chunks


@lru_cache(maxsize=1)
def load_chunks() -> tuple[Chunk, ...]:
    return tuple(c for p in load_procedures().values() for c in chunk_procedure(p))
