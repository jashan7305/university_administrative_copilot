"""Week 4 improved classifiers: fine-tuned transformers (vs the TF-IDF and frozen-embedding baselines).

  TransformerClassifier  end-to-end fine-tuning of a sentence-transformer backbone (BGE-small, 33M params) with a
                         classification head. Trained jointly on 95 sub-intents + out_of_scope, so one model gives
                         routing (sub-intent) and, through the taxonomy, the intent.
  SetFitClassifier       contrastive fine-tuning of the sentence embedding (batch-all triplet loss on the labels),
                         then a logistic-regression head - the "SetFit" recipe for few-shot text classification.

Both follow the sklearn interface (fit / predict / predict_proba / classes_) used by the rest of the pipeline and
select the best epoch on the validation split.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

BACKBONE = "BAAI/bge-small-en-v1.5"


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TransformerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, backbone: str = BACKBONE, epochs: int = 10, lr: float = 8e-5, head_lr: float = 2e-3,
                 batch_size: int = 32, max_len: int = 64, class_weighting: str = "sqrt", seed: int = 42):
        self.backbone, self.epochs, self.lr, self.head_lr, self.batch_size = backbone, epochs, lr, head_lr, batch_size
        self.max_len, self.class_weighting, self.seed = max_len, class_weighting, seed

    def _batches(self, texts, labels=None, shuffle=False):
        idx = list(range(len(texts)))
        if shuffle:
            random.shuffle(idx)
        for i in range(0, len(idx), self.batch_size):
            b = idx[i:i + self.batch_size]
            enc = self.tok_([texts[j] for j in b], padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
            enc = {k: v.to(self.dev_) for k, v in enc.items()}
            yield enc, (torch.tensor([labels[j] for j in b], device=self.dev_) if labels is not None else None)

    def fit(self, X, y, X_val=None, y_val=None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

        seed_all(self.seed)
        self.dev_ = device()
        X, y = list(X), list(y)
        self.classes_ = np.array(sorted(set(y)))
        lab = {c: i for i, c in enumerate(self.classes_)}
        yi = [lab[v] for v in y]
        self.tok_ = AutoTokenizer.from_pretrained(self.backbone)
        self.model_ = AutoModelForSequenceClassification.from_pretrained(
            self.backbone, num_labels=len(self.classes_), id2label=dict(enumerate(self.classes_)),
            label2id=lab).to(self.dev_)
        # sub-intents range from a handful to hundreds of examples: inverse-sqrt-frequency weights soften the imbalance
        counts = np.maximum(np.bincount(yi, minlength=len(self.classes_)), 1)
        w = (counts.max() / counts) ** (0.5 if self.class_weighting == "sqrt" else 1.0) if self.class_weighting != "none" else np.ones_like(counts)
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(w / w.mean(), dtype=torch.float, device=self.dev_))
        # the new classification head learns from scratch, so it gets a much higher learning rate than the encoder
        head = [p for n, p in self.model_.named_parameters() if n.startswith("classifier")]
        body = [p for n, p in self.model_.named_parameters() if not n.startswith("classifier")]
        opt = torch.optim.AdamW([{"params": body, "lr": self.lr}, {"params": head, "lr": self.head_lr}], weight_decay=0.01)
        steps = self.epochs * ((len(X) + self.batch_size - 1) // self.batch_size)
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        best, best_state, self.history_ = -1.0, None, []
        for ep in range(self.epochs):
            self.model_.train()
            total = 0.0
            for enc, lbl in self._batches(X, yi, shuffle=True):
                loss = loss_fn(self.model_(**enc).logits, lbl)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                opt.step(), sched.step(), opt.zero_grad()
                total += loss.item()
            score = f1_score(y_val, self.predict(X_val), average="macro", zero_division=0) if X_val is not None else -total
            self.history_.append({"epoch": ep + 1, "train_loss": round(total, 3), "val_macro_f1": round(float(score), 4)})
            print(f"      epoch {ep + 1}: loss={total:.2f} val_macro_f1={score:.4f}")
            if score > best:
                best, best_state = score, {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        self.model_.eval()
        out = [torch.softmax(self.model_(**enc).logits, -1).float().cpu().numpy() for enc, _ in self._batches(list(X))]
        return np.concatenate(out) if out else np.zeros((0, len(self.classes_)))

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(1)]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model_.save_pretrained(path), self.tok_.save_pretrained(path)
        (path / "classes.json").write_text(json.dumps(self.classes_.tolist()))

    @classmethod
    def load(cls, path: Path) -> "TransformerClassifier":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        obj = cls(backbone=str(path))
        obj.dev_ = device()
        obj.tok_ = AutoTokenizer.from_pretrained(path)
        obj.model_ = AutoModelForSequenceClassification.from_pretrained(path).to(obj.dev_).eval()
        obj.classes_ = np.array(json.loads((path / "classes.json").read_text()))
        return obj


class SetFitClassifier(BaseEstimator, ClassifierMixin):
    """Contrastively fine-tuned sentence embeddings + logistic regression head."""

    def __init__(self, backbone: str = BACKBONE, epochs: int = 2, batch_size: int = 64, C: float = 10.0, seed: int = 42):
        self.backbone, self.epochs, self.batch_size, self.C, self.seed = backbone, epochs, batch_size, C, seed

    def fit(self, X, y):
        from datasets import Dataset
        from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
        from sentence_transformers.losses import BatchAllTripletLoss
        from sentence_transformers.base.sampler import BatchSamplers

        seed_all(self.seed)
        X, y = list(X), list(y)
        labels = {c: i for i, c in enumerate(sorted(set(y)))}
        self.encoder_ = SentenceTransformer(self.backbone, device=device())
        ds = Dataset.from_dict({"text": X, "label": [labels[v] for v in y]})
        args = SentenceTransformerTrainingArguments(
            output_dir="/tmp/setfit_run", num_train_epochs=self.epochs, per_device_train_batch_size=self.batch_size,
            learning_rate=3e-5, warmup_ratio=0.1, batch_sampler=BatchSamplers.GROUP_BY_LABEL, report_to="none",
            save_strategy="no", logging_steps=1000, seed=self.seed)
        SentenceTransformerTrainer(model=self.encoder_, args=args, train_dataset=ds,
                                   loss=BatchAllTripletLoss(self.encoder_)).train()
        self.head_ = LogisticRegression(C=self.C, max_iter=5000, class_weight="balanced").fit(self._embed(X), y)
        self.classes_ = self.head_.classes_
        return self

    def _embed(self, X):
        return self.encoder_.encode(list(X), normalize_embeddings=True, batch_size=64, show_progress_bar=False)

    def predict_proba(self, X) -> np.ndarray:
        return self.head_.predict_proba(self._embed(X))

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(1)]


class JointIntentAdapter:
    """Exposes a joint sub-intent model as an intent classifier: intent probability = sum of its sub-intents'."""

    def __init__(self, joint, sub_to_intent: dict[str, str]):
        self.joint = joint
        self.map = {s: sub_to_intent.get(s, "out_of_scope") for s in joint.classes_}
        self.classes_ = np.array(sorted(set(self.map.values())))

    def predict_proba(self, X) -> np.ndarray:
        p = self.joint.predict_proba(X)
        out = np.zeros((len(p), len(self.classes_)))
        idx = {c: i for i, c in enumerate(self.classes_)}
        for j, s in enumerate(self.joint.classes_):
            out[:, idx[self.map[s]]] += p[:, j]
        return out

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(1)]


class ProbabilityBlend:
    """Weighted average of class probabilities from models with possibly different label sets (labels are aligned
    by name; a model contributes 0 for labels it does not have). Weights are chosen on validation data."""

    def __init__(self, members: list, weights: list[float]):
        self.members, self.weights = members, weights
        self.classes_ = np.array(sorted(set().union(*[set(m.classes_) for m in members])))

    def predict_proba(self, X) -> np.ndarray:
        idx = {c: i for i, c in enumerate(self.classes_)}
        out = np.zeros((len(X), len(self.classes_)))
        for m, w in zip(self.members, self.weights):
            p = m.predict_proba(X)
            for j, c in enumerate(m.classes_):
                out[:, idx[c]] += w * p[:, j]
        return out / sum(self.weights)

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(1)]
