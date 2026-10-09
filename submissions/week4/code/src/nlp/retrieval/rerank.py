"""Week 4 improved retrieval: fine-tuned vector retrieval and cross-encoder re-ranking (vs BM25 keyword search).

  finetune_dense()     fine-tunes the BGE-small bi-encoder on (training question, relevant NMIMS passage) pairs
                       with in-batch negatives (MultipleNegativesRankingLoss); only TRAIN-split questions are used
  RerankRetriever      first stage = any retriever (top-N candidates), second stage = a cross-encoder that reads
                       query and passage together and re-scores them (ms-marco MiniLM, optionally fine-tuned)
"""
from __future__ import annotations

from pathlib import Path

from nlp.intent.finetune import BACKBONE, device, seed_all
from nlp.retrieval.search import Hit

CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def finetune_dense(pairs: list[tuple[str, str]], out_dir: Path, epochs: int = 2, batch_size: int = 16,
                   max_seq_length: int = 256) -> Path:
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.base.sampler import BatchSamplers

    seed_all()
    model = SentenceTransformer(BACKBONE, device=device())
    model.max_seq_length = max_seq_length          # passages are long; 256 tokens keeps GPU memory bounded
    ds = Dataset.from_dict({"anchor": [q for q, _ in pairs], "positive": [p for _, p in pairs]})
    args = SentenceTransformerTrainingArguments(
        output_dir="/tmp/dense_ft", num_train_epochs=epochs, per_device_train_batch_size=batch_size, learning_rate=2e-5,
        warmup_ratio=0.1, batch_sampler=BatchSamplers.NO_DUPLICATES, report_to="none", save_strategy="no", logging_steps=1000)
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=MultipleNegativesRankingLoss(model)).train()
    model.save(str(out_dir))
    return out_dir


def finetune_cross_encoder(pairs: list[tuple[str, str, float]], out_dir: Path, epochs: int = 1, batch_size: int = 16) -> Path:
    """Fine-tune the cross-encoder on (question, passage, 1/0) pairs (positives + mined hard negatives)."""
    from datasets import Dataset
    from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    seed_all()
    model = CrossEncoder(CROSS_ENCODER, device=device(), max_length=256)
    ds = Dataset.from_dict({"query": [a for a, _, _ in pairs], "passage": [b for _, b, _ in pairs],
                            "label": [float(c) for _, _, c in pairs]})
    args = CrossEncoderTrainingArguments(output_dir="/tmp/ce_ft", num_train_epochs=epochs, per_device_train_batch_size=batch_size,
                                         learning_rate=2e-5, warmup_ratio=0.1, report_to="none", save_strategy="no", logging_steps=1000)
    CrossEncoderTrainer(model=model, args=args, train_dataset=ds, loss=BinaryCrossEntropyLoss(model)).train()
    model.save(str(out_dir))
    return out_dir


class RerankRetriever:
    name = "rerank"

    def __init__(self, first_stage, model: str | Path = CROSS_ENCODER, candidates: int = 20):
        from sentence_transformers.cross_encoder import CrossEncoder

        self.first, self.candidates = first_stage, candidates
        self.ce = CrossEncoder(str(model), device=device(), max_length=256)
        self.chunks = getattr(first_stage, "chunks", None) or first_stage.bm25.chunks

    def search(self, query: str, k: int = 5, **kw) -> list[Hit]:
        cands = self.first.search(query, k=self.candidates, **kw)
        if not cands:
            return []
        scores = self.ce.predict([(query, h.chunk.indexed_text) for h in cands], show_progress_bar=False)
        ranked = sorted(zip(cands, scores), key=lambda x: -x[1])
        return [Hit(h.doc_id, float(s), h.chunk) for h, s in ranked[:k]]
