"""Week 4 improved NER: fine-tuned transformer token classification (vs rule-based and CRF baselines).

The query is tokenised into word pieces with character offsets; each word piece gets the BIO tag of the span it
falls in (continuation pieces of a word share the word's tag type). At inference, contiguous B/I pieces of the same
type are merged back into character spans. Regex rules for pattern-shaped entities (dates, amounts, phone numbers,
student IDs) are merged in exactly as for the CRF, so the two NER models differ only in the learned tagger.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from seqeval.metrics import f1_score as seq_f1

from nlp.intent.finetune import BACKBONE, device, seed_all
from nlp.ner.extractor import Entity, RuleExtractor


class TransformerNER:
    def __init__(self, backbone: str = BACKBONE, epochs: int = 8, lr: float = 5e-5, batch_size: int = 16, max_len: int = 96):
        self.backbone, self.epochs, self.lr, self.batch_size, self.max_len = backbone, epochs, lr, batch_size, max_len

    # ------------------------------------------------------------------ alignment
    def _encode(self, texts: list[str], spans: list[list[dict]] | None = None):
        enc = self.tok(texts, truncation=True, max_length=self.max_len, padding=True, return_offsets_mapping=True,
                       return_tensors="pt")
        offsets = enc.pop("offset_mapping").tolist()
        if spans is None:
            return enc, offsets, None
        labels = []
        for off, sp in zip(offsets, spans):
            row, prev = [], None
            for s, e in off:
                if s == e:                                  # special / padding token
                    row.append(-100)
                    continue
                tag = "O"
                for x in sp:
                    if s >= x["start"] and e <= x["end"]:
                        tag = ("I-" if prev == x["start"] else "B-") + x["label"]
                        prev = x["start"]
                        break
                else:
                    prev = None
                row.append(self.label2id.get(tag, 0))
            labels.append(row)
        return enc, offsets, torch.tensor(labels)

    def _decode(self, text: str, offsets, pred_ids) -> list[Entity]:
        ents, cur = [], None
        for (s, e), pid in zip(offsets, pred_ids):
            if s == e:
                continue
            tag = self.id2label[pid]
            if tag == "O":
                if cur:
                    ents.append(cur)
                cur = None
                continue
            kind, label = tag.split("-", 1)
            # continuation of the same word (no gap) or an I- tag of the same type extends the current span
            if cur and cur[0] == label and (kind == "I" or s == cur[2]):
                cur[2] = e
            else:
                if cur:
                    ents.append(cur)
                cur = [label, s, e]
        if cur:
            ents.append(cur)
        return [Entity(l, text[s:e], s, e, "transformer") for l, s, e in ents]

    # ------------------------------------------------------------------ train / predict
    def fit(self, records: list[dict], val: list[dict] | None = None) -> "TransformerNER":
        from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

        seed_all()
        self.dev = device()
        types = sorted({e["label"] for r in records for e in r["entities"]})
        tags = ["O"] + [f"{p}-{t}" for t in types for p in ("B", "I")]
        self.label2id = {t: i for i, t in enumerate(tags)}
        self.id2label = dict(enumerate(tags))
        self.tok = AutoTokenizer.from_pretrained(self.backbone)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.backbone, num_labels=len(tags), id2label=self.id2label, label2id=self.label2id).to(self.dev)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        steps = self.epochs * ((len(records) + self.batch_size - 1) // self.batch_size)
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        best, best_state, self.history = -1.0, None, []
        for ep in range(self.epochs):
            self.model.train()
            idx = list(range(len(records)))
            random.shuffle(idx)
            total = 0.0
            for i in range(0, len(idx), self.batch_size):
                batch = [records[j] for j in idx[i:i + self.batch_size]]
                enc, _, labels = self._encode([r["text"] for r in batch], [r["entities"] for r in batch])
                out = self.model(**{k: v.to(self.dev) for k, v in enc.items()}, labels=labels.to(self.dev))
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step(), sched.step(), opt.zero_grad()
                total += out.loss.item()
            score = self.score(val) if val else -total
            self.history.append({"epoch": ep + 1, "train_loss": round(total, 3), "val_f1": round(score, 4)})
            print(f"      epoch {ep + 1}: loss={total:.2f} val_span_f1={score:.4f}")
            if score > best:
                best, best_state = score, {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def extract_learned(self, texts: list[str]) -> list[list[Entity]]:
        self.model.eval()
        out = []
        for i in range(0, len(texts), 64):
            chunk = texts[i:i + 64]
            enc, offsets, _ = self._encode(chunk)
            pred = self.model(**{k: v.to(self.dev) for k, v in enc.items()}).logits.argmax(-1).cpu().tolist()
            out += [self._decode(t, o, p) for t, o, p in zip(chunk, offsets, pred)]
        return out

    def score(self, records: list[dict]) -> float:
        """Strict span F1 (seqeval over character-aligned word tokens) on records with gold spans."""
        from nlp.ner.extractor import spans_to_bio, tokenize

        preds = self.extract_learned([r["text"] for r in records])
        Y, P = [], []
        for r, ents in zip(records, preds):
            t = tokenize(r["text"])
            Y.append(spans_to_bio(t, r["entities"]))
            P.append(spans_to_bio(t, [{"start": e.start, "end": e.end, "label": e.label} for e in ents]))
        return float(seq_f1(Y, P))

    def extract(self, text: str) -> list[Entity]:
        rule_ents = RuleExtractor().extract(text)
        out = list(rule_ents)
        for e in self.extract_learned([text])[0]:
            if not any(e.start < r.end and r.start < e.end for r in rule_ents):
                out.append(e)
        return sorted(out, key=lambda e: e.start)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path), self.tok.save_pretrained(path)

    @classmethod
    def load(cls, path: Path) -> "TransformerNER":
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        obj = cls(backbone=str(path))
        obj.dev = device()
        obj.tok = AutoTokenizer.from_pretrained(path)
        obj.model = AutoModelForTokenClassification.from_pretrained(path).to(obj.dev).eval()
        obj.id2label = {int(k): v for k, v in obj.model.config.id2label.items()}
        obj.label2id = {v: k for k, v in obj.id2label.items()}
        return obj
