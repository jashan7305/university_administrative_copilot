"""Query-focused extractive summarization of retrieved NMIMS passages.

The copilot's answer is a short summary of the evidence: candidate sentences come from the verified facts of the
routed service and the top retrieved passages, and a summarizer picks the few sentences that best answer the query.
Extractive only - every sentence in an answer is copied from an official source or a verified fact.

  LeadSummarizer      first sentences of the best passage (baseline)
  TextRankSummarizer  sentence centrality (PageRank over a cosine-similarity graph), query-agnostic
  MMRSummarizer       Maximal Marginal Relevance: relevance to the query minus redundancy with chosen sentences
"""
from __future__ import annotations

import re

import numpy as np

from nlp.retrieval.search import get_encoder

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
MAX_WORDS = 70


def sentences(texts: list[str]) -> list[str]:
    out = []
    for t in texts:
        t = re.sub(r"\s+", " ", str(t)).strip()
        out += [s.strip() for s in _SENT.split(t) if 6 <= len(s.split()) <= 60]
    return list(dict.fromkeys(out))


def _budget(sents: list[str], order: list[int]) -> str:
    chosen, words = [], 0
    for i in order:
        n = len(sents[i].split())
        if chosen and words + n > MAX_WORDS:
            break
        chosen.append(sents[i])
        words += n
    return " ".join(chosen)


class LeadSummarizer:
    name = "lead"

    def summarize(self, query: str, texts: list[str]) -> str:
        sents = sentences(texts[:1])
        return _budget(sents, list(range(len(sents))))


class TextRankSummarizer:
    name = "textrank"

    def summarize(self, query: str, texts: list[str]) -> str:
        sents = sentences(texts)
        if len(sents) <= 1:
            return " ".join(sents)
        e = get_encoder().encode(sents, normalize_embeddings=True, show_progress_bar=False)
        sim = np.clip(e @ e.T, 0, None)
        np.fill_diagonal(sim, 0)
        trans = sim / np.maximum(sim.sum(1, keepdims=True), 1e-9)
        r = np.full(len(sents), 1 / len(sents))
        for _ in range(30):
            r = 0.15 / len(sents) + 0.85 * trans.T @ r
        return _budget(sents, list(np.argsort(-r)))


class MMRSummarizer:
    name = "mmr"

    def __init__(self, lam: float = 0.75):
        self.lam = lam

    def summarize(self, query: str, texts: list[str]) -> str:
        sents = sentences(texts)
        if not sents:
            return ""
        enc = get_encoder()
        e = enc.encode(sents, normalize_embeddings=True, show_progress_bar=False)
        q = enc.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        rel, order = e @ q, []
        cand = list(range(len(sents)))
        while cand and len(order) < 4:
            red = (e[cand] @ e[order].T).max(1) if order else np.zeros(len(cand))
            best = cand[int(np.argmax(self.lam * rel[cand] - (1 - self.lam) * red))]
            order.append(best)
            cand.remove(best)
        return _budget(sents, order)


# --------------------------------------------------------------------------- Week 4: LLM-assisted summarization

LLM_PROMPT = ("You are the NMIMS University Administrative Copilot. Using ONLY the evidence below, answer the student's "
              "question in at most 2 sentences (under 60 words). Use only evidence that answers this question and copy fees, "
              "dates, emails and form names exactly as written. "
              "If the evidence does not answer the question, say which office to contact.\n\nEvidence:\n{evidence}\n\n"
              "Question: {query}\nAnswer:")


class LLMSummarizer:
    """LLM-assisted summarization: MMR first selects the most relevant evidence sentences, then an instruction-tuned
    LLM rewrites them into a short answer. Default is a local open model (Qwen2.5-Instruct, runs on the Apple GPU);
    set backend="claude" to use Claude via the Anthropic API instead (needs credentials)."""

    def __init__(self, model: str = "Qwen/Qwen2.5-1.5B-Instruct", backend: str = "local", max_new_tokens: int = 120):
        self.name = f"llm_{backend}"
        self.backend, self.model_name, self.max_new_tokens = backend, model, max_new_tokens
        if backend == "local":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            self.tok = AutoTokenizer.from_pretrained(model)
            self.llm = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float16 if dev == "mps" else torch.float32).to(dev).eval()
            self.dev = dev
        else:
            import anthropic

            self.client = anthropic.Anthropic()

    def evidence(self, query: str, texts: list[str]) -> str:
        sents = sentences(texts)
        if not sents:
            return ""
        enc = get_encoder()
        e = enc.encode(sents, normalize_embeddings=True, show_progress_bar=False)
        q = enc.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        top = np.argsort(-(e @ q))[:6]
        return "\n".join(f"- {sents[i]}" for i in top)

    def summarize(self, query: str, texts: list[str]) -> str:
        ev = self.evidence(query, texts)
        if not ev:
            return ""
        prompt = LLM_PROMPT.format(evidence=ev, query=query)
        if self.backend == "local":
            import torch

            msgs = [{"role": "user", "content": prompt}]
            ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.dev)
            with torch.no_grad():
                out = self.llm.generate(**ids, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=self.tok.eos_token_id)
            return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        resp = self.client.messages.create(model="claude-opus-5", max_tokens=400, output_config={"effort": "low"},
                                           messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in resp.content if b.type == "text").strip()
