"""Inference wrapper around the trained baseline (models/intent_baseline.joblib)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "intent_baseline.joblib"


@lru_cache(maxsize=1)
def _load() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not trained yet. Run: python -m logic.intent.experiments")
    return joblib.load(MODEL_PATH)


def classify(text: str, top_k: int = 3) -> dict:
    bundle = _load()
    model, threshold = bundle["model"], bundle["threshold"]
    proba = model.predict_proba([text])[0]
    ranked = sorted(zip(model.classes_, proba), key=lambda x: -x[1])
    intent, confidence = ranked[0]
    return {
        "intent": intent if confidence >= threshold else "out_of_scope",
        "confidence": round(float(confidence), 4),
        "rejected_low_confidence": bool(confidence < threshold),
        "top_k": [{"intent": i, "score": round(float(p), 4)} for i, p in ranked[:top_k]],
        "normalised_text": model.named_steps["prep"].transform([text])[0],
    }
