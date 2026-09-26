"""Week 2 experiments: EDA, preprocessing ablation, model comparison, threshold tuning, final evaluation.

Protocol
  * Model selection uses only train + val: 5-fold GroupKFold cross-validation on train (groups = sentence
    frame, so folds never share a template) and the frame-disjoint validation split.
  * The test splits are touched once, by the selected configuration, at the end.
  * Metrics: accuracy and macro-F1 (primary, classes are imbalanced), with 95% bootstrap intervals.

Run from src/backend:  python -m logic.intent.experiments
Outputs: reports/week2/{results.json, *.csv, figures/*.png}, models/intent_baseline.joblib
"""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, cross_val_score

from logic.intent.models import MODEL_NAMES, TextPreprocessor, make_model, predict_with_rejection
from logic.intent.preprocess import DEFAULT, RAW, PreprocessConfig, Preprocessor

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "week2"
FIG = OUT / "figures"
MODEL_DIR = ROOT / "models"
SEED = 42
TEST_SPLITS = ("test", "test_noisy", "test_seed", "test_clinc_oos")

sns.set_theme(style="whitegrid", context="paper")


def load() -> dict[str, pd.DataFrame]:
    return {s: pd.read_csv(DATA / f"{s}.csv") for s in ("train", "val", *TEST_SPLITS)}


def macro_f1(y, p) -> float:
    return f1_score(y, p, average="macro", zero_division=0)


def bootstrap_ci(y, p, metric, n: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    y, p = np.asarray(y), np.asarray(p)
    scores = [metric(y[idx], p[idx]) for idx in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=160)
    plt.close()


# --------------------------------------------------------------------------- 1. EDA

def eda(d: dict[str, pd.DataFrame]) -> dict:
    train = d["train"]
    counts = pd.DataFrame({s: d[s]["intent"].value_counts() for s in ("train", "val", "test")}).fillna(0).astype(int)
    counts.to_csv(OUT / "class_distribution.csv")

    ax = counts.plot.barh(figsize=(7, 4.5), width=0.8)
    ax.set_xlabel("utterances"), ax.set_ylabel(""), ax.set_title("Class distribution by split")
    savefig("class_distribution.png")

    plt.figure(figsize=(7, 3.5))
    sns.boxplot(data=train, x="n_tokens", y="intent", color="#9db4d6", fliersize=2)
    plt.title("Utterance length (tokens) in train"), plt.xlabel("tokens"), plt.ylabel("")
    savefig("length_distribution.png")

    prep = Preprocessor(replace(DEFAULT, spell_correct=False))
    train_tok = [prep.tokens(t) for t in train["text"]]
    vocab = Counter(t for toks in train_tok for t in toks)
    oov = {}
    for s in ("val", "test", "test_noisy", "test_seed"):
        toks = [t for x in d[s]["text"] for t in prep.tokens(x)]
        oov[s] = round(sum(t not in vocab for t in toks) / len(toks), 4)

    # Most discriminative terms per intent (chi-squared, one-vs-rest)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3)
    X = vec.fit_transform([" ".join(t) for t in train_tok])
    names = np.array(vec.get_feature_names_out())
    top_terms = {}
    for intent in sorted(train["intent"].unique()):
        scores, _ = chi2(X, train["intent"] == intent)
        top_terms[intent] = names[np.argsort(np.nan_to_num(scores))[::-1][:8]].tolist()

    stats = {
        "rows": {s: len(df) for s, df in d.items()},
        "train_vocab_size": len(vocab),
        "train_mean_tokens": round(train["n_tokens"].mean(), 2),
        "train_median_tokens": float(train["n_tokens"].median()),
        "train_max_tokens": int(train["n_tokens"].max()),
        "imbalance_ratio_max_min": round(counts["train"].max() / counts["train"].min(), 2),
        "oov_rate_vs_train": oov,
        "top_chi2_terms": top_terms,
    }
    return stats


# --------------------------------------------------------------------------- 2. preprocessing ablation

ABLATION_STEPS: list[tuple[str, PreprocessConfig]] = [
    ("raw (tokenise only)", RAW),
    ("+ lowercase", replace(RAW, lowercase=True)),
    ("+ contractions", replace(RAW, lowercase=True, expand_contractions=True)),
    ("+ abbreviations", replace(RAW, lowercase=True, expand_contractions=True, expand_abbreviations=True)),
    ("+ entity masking", replace(RAW, lowercase=True, expand_contractions=True, expand_abbreviations=True, mask_entities=True)),
    ("+ spelling correction", replace(DEFAULT, lemmatize=False)),
    ("+ lemmatisation (full pipeline)", DEFAULT),
    ("full, stemming instead of lemma", replace(DEFAULT, lemmatize=False, stem=True)),
    ("full + stopword removal", replace(DEFAULT, remove_stopwords=True)),
]


def ablation(d: dict[str, pd.DataFrame], model_name: str) -> pd.DataFrame:
    rows = []
    for label, cfg in ABLATION_STEPS:
        m = make_model(model_name, cfg).fit(d["train"]["text"], d["train"]["intent"])
        row = {"step": label}
        for s in ("val", "test_noisy"):  # test_noisy here is only reported, never used to choose
            row[f"{s}_macro_f1"] = round(macro_f1(d[s]["intent"], m.predict(d[s]["text"])), 4)
        rows.append(row)
        print(f"  {label:34s} val={row['val_macro_f1']:.4f} noisy={row['test_noisy_macro_f1']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "preprocessing_ablation.csv", index=False)

    plt.figure(figsize=(7, 3.8))
    long = df.melt(id_vars="step", var_name="split", value_name="macro-F1")
    long["split"] = long["split"].str.replace("_macro_f1", "")
    sns.barplot(data=long, y="step", x="macro-F1", hue="split", palette=["#4c72b0", "#dd8452"])
    plt.xlim(max(0, long["macro-F1"].min() - 0.05), 1), plt.ylabel(""), plt.title("Preprocessing ablation (LogReg, word+char TF-IDF)")
    savefig("preprocessing_ablation.png")
    return df


# --------------------------------------------------------------------------- 3. model comparison

def compare_models(d: dict[str, pd.DataFrame], cfg: PreprocessConfig) -> pd.DataFrame:
    train = d["train"]
    cv = GroupKFold(n_splits=5)
    rows = []
    for name in MODEL_NAMES:
        t0 = time.perf_counter()
        cv_scores = cross_val_score(make_model(name, cfg), train["text"], train["intent"], groups=train["group"],
                                    cv=cv, scoring="f1_macro", n_jobs=-1)
        m = make_model(name, cfg).fit(train["text"], train["intent"])
        pv = m.predict(d["val"]["text"])
        rows.append({"model": name, "cv_macro_f1_mean": round(cv_scores.mean(), 4), "cv_macro_f1_std": round(cv_scores.std(), 4),
                     "val_accuracy": round(accuracy_score(d["val"]["intent"], pv), 4),
                     "val_macro_f1": round(macro_f1(d["val"]["intent"], pv), 4),
                     "fit_cv_seconds": round(time.perf_counter() - t0, 1)})
        print(f"  {name:30s} cv={cv_scores.mean():.4f}±{cv_scores.std():.4f} val={rows[-1]['val_macro_f1']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "model_comparison.csv", index=False)

    plt.figure(figsize=(7, 3.4))
    order = df.sort_values("val_macro_f1")["model"]
    plt.barh(order, df.set_index("model").loc[order, "cv_macro_f1_mean"], xerr=df.set_index("model").loc[order, "cv_macro_f1_std"],
             color="#4c72b0", alpha=0.8, label="5-fold GroupKFold CV")
    plt.scatter(df.set_index("model").loc[order, "val_macro_f1"], order, color="#dd8452", zorder=3, label="validation")
    plt.xlabel("macro-F1"), plt.title("Model comparison"), plt.legend(loc="lower right")
    savefig("model_comparison.png")
    return df


def tune_C(d: dict[str, pd.DataFrame], name: str, cfg: PreprocessConfig) -> tuple[float, list]:
    train = d["train"]
    grid = []
    for C in (0.3, 1, 3, 10, 30, 100):
        s = cross_val_score(make_model(name, cfg, C=C), train["text"], train["intent"], groups=train["group"],
                            cv=GroupKFold(5), scoring="f1_macro", n_jobs=-1)
        grid.append({"C": C, "cv_macro_f1": round(s.mean(), 4), "std": round(s.std(), 4)})
        print(f"  C={C:<6} cv={s.mean():.4f}±{s.std():.4f}")
    best = max(grid, key=lambda g: g["cv_macro_f1"])["C"]
    return best, grid


# --------------------------------------------------------------------------- 4. OOS threshold

def tune_threshold(model, d: dict[str, pd.DataFrame]) -> tuple[float, list]:
    val = d["val"]
    curve = []
    for t in np.round(np.arange(0.0, 0.81, 0.05), 2):
        p = predict_with_rejection(model, val["text"], t)
        in_scope = val["intent"] != "out_of_scope"
        curve.append({"threshold": float(t), "val_macro_f1": round(macro_f1(val["intent"], p), 4),
                      "in_scope_accuracy": round(accuracy_score(val["intent"][in_scope], p[in_scope]), 4),
                      "oos_recall": round(float((p[~in_scope] == "out_of_scope").mean()), 4)})
    best = max(curve, key=lambda c: (c["val_macro_f1"], -c["threshold"]))["threshold"]
    df = pd.DataFrame(curve)
    df.to_csv(OUT / "threshold_curve.csv", index=False)
    plt.figure(figsize=(6, 3.4))
    for col, lab in (("val_macro_f1", "macro-F1"), ("in_scope_accuracy", "in-scope accuracy"), ("oos_recall", "OOS recall")):
        plt.plot(df["threshold"], df[col], marker="o", ms=3, label=lab)
    plt.axvline(best, color="grey", ls="--", lw=1)
    plt.xlabel("confidence threshold"), plt.title("Out-of-scope rejection threshold (validation)"), plt.legend()
    savefig("threshold_curve.png")
    return best, curve


# --------------------------------------------------------------------------- 5. final evaluation

def final_eval(model, threshold: float, d: dict[str, pd.DataFrame]) -> dict:
    results, labels = {}, sorted(d["train"]["intent"].unique())
    errors = []
    for s in TEST_SPLITS:
        df = d[s]
        p = predict_with_rejection(model, df["text"], threshold)
        y = df["intent"].to_numpy()
        r = {"n": len(df), "accuracy": round(accuracy_score(y, p), 4)}
        if s == "test_clinc_oos":
            r["oos_recall"] = r["accuracy"]
            r["accuracy_ci95"] = [round(x, 4) for x in bootstrap_ci(y, p, accuracy_score)]
        else:
            r["macro_f1"] = round(macro_f1(y, p), 4)
            r["accuracy_ci95"] = [round(x, 4) for x in bootstrap_ci(y, p, accuracy_score)]
            r["macro_f1_ci95"] = [round(x, 4) for x in bootstrap_ci(y, p, macro_f1)]
            r["per_class"] = {k: {m: round(v[m], 3) for m in ("precision", "recall", "f1-score", "support")}
                              for k, v in classification_report(y, p, output_dict=True, zero_division=0).items() if k in labels}
            if s in ("test", "test_seed"):
                cm = confusion_matrix(y, p, labels=labels)
                pd.DataFrame(cm, index=labels, columns=labels).to_csv(OUT / f"confusion_{s}.csv")
                plt.figure(figsize=(7, 5.8))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False)
                plt.xlabel("predicted"), plt.ylabel("true"), plt.title(f"Confusion matrix: {s}")
                plt.xticks(rotation=45, ha="right")
                savefig(f"confusion_{s}.png")
        errors += [{"split": s, "text": t, "true": a, "pred": b} for t, a, b in zip(df["text"], y, p) if a != b]
        results[s] = r
    pd.DataFrame(errors).to_csv(OUT / "errors.csv", index=False)

    texts = d["test"]["text"].tolist()
    t0 = time.perf_counter()
    for t in texts:
        model.predict_proba([t])
    results["latency_ms_per_query"] = round((time.perf_counter() - t0) / len(texts) * 1000, 2)
    return results


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    d = load()

    print("[1] EDA")
    stats = eda(d)
    print("[2] Preprocessing ablation")
    abl = ablation(d, "logreg_word_char")
    best_step = abl.loc[abl["val_macro_f1"].idxmax(), "step"]
    cfg = dict(ABLATION_STEPS)[best_step]
    print(f"    selected preprocessing: {best_step} -> {cfg.name()}")

    print("[3] Model comparison")
    comp = compare_models(d, cfg)
    best_model = comp.loc[comp["val_macro_f1"].idxmax(), "model"]
    print(f"    selected model: {best_model}")

    print("[4] Regularisation (C) tuning")
    C, grid = tune_C(d, best_model, cfg) if best_model.startswith(("logreg", "linear_svm")) else (10.0, [])
    model = make_model(best_model, cfg, C=C)
    if not hasattr(model, "predict_proba") or best_model == "linear_svm":
        best_model = "logreg_word_char"  # rejection needs calibrated probabilities
        model = make_model(best_model, cfg, C=C)
    model.fit(d["train"]["text"], d["train"]["intent"])

    print("[5] OOS threshold tuning")
    threshold, curve = tune_threshold(model, d)

    print("[6] Final evaluation (test splits, touched once)")
    final = final_eval(model, threshold, d)
    baseline_majority = make_model("majority", cfg).fit(d["train"]["text"], d["train"]["intent"])
    final["majority_reference"] = {s: round(accuracy_score(d[s]["intent"], baseline_majority.predict(d[s]["text"])), 4)
                                   for s in ("test", "test_seed")}

    joblib.dump({"model": model, "threshold": threshold, "preprocessing": cfg.__dict__}, MODEL_DIR / "intent_baseline.joblib")
    results = {"eda": stats, "ablation": abl.to_dict("records"), "selected_preprocessing": cfg.__dict__,
               "model_comparison": comp.to_dict("records"), "selected_model": best_model, "C_grid": grid, "C": C,
               "threshold_curve": curve, "threshold": threshold, "final": final}
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: {m: v for m, v in r.items() if m != "per_class"} if isinstance(r, dict) else r
                      for k, r in final.items()}, indent=2))


if __name__ == "__main__":
    main()
