"""Baseline intent classifier: word+char TF-IDF + logistic regression.

Run from src/backend:  python -m logic.intent.baseline
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
        ])),
        ("clf", LogisticRegression(max_iter=1000, C=10)),
    ])


def evaluate(model, frame: pd.DataFrame, name: str) -> dict:
    pred = model.predict(frame["clean_text"])
    acc = accuracy_score(frame["intent"], pred)
    macro_f1 = f1_score(frame["intent"], pred, average="macro")
    print(f"\n=== {name} ({len(frame)} examples) ===")
    print(f"accuracy={acc:.3f}  macro-F1={macro_f1:.3f}")
    print(classification_report(frame["intent"], pred, zero_division=0))
    labels = sorted(frame["intent"].unique())
    print("confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(confusion_matrix(frame["intent"], pred, labels=labels), index=labels, columns=labels).to_string())
    errors = frame.assign(pred=pred)[lambda d: d["intent"] != d["pred"]]
    if len(errors):
        print("\nmisclassified:")
        print(errors[["text", "intent", "pred"]].to_string(index=False))
    return {"accuracy": acc, "macro_f1": macro_f1}


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    tests = {n: pd.read_csv(DATA_DIR / f"{n}.csv") for n in ("test", "test_noisy", "test_clinc_oos")}

    majority = DummyClassifier(strategy="most_frequent").fit(train["clean_text"], train["intent"])
    model = build_pipeline().fit(train["clean_text"], train["intent"])

    results = {"majority_class": {}, "tfidf_logreg": {}}
    for name, frame in tests.items():
        results["majority_class"][name] = evaluate(majority, frame, f"majority baseline / {name}")
        results["tfidf_logreg"][name] = evaluate(model, frame, f"TF-IDF + LogReg / {name}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "intent_baseline.joblib")
    (DATA_DIR / "baseline_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
