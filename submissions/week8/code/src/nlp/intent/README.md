# Intent classification

| File | Purpose |
|---|---|
| `preprocess.py` | Configurable preprocessing (contractions, NMIMS abbreviations, entity masking, spelling correction, stopwords, lemmatisation) |
| `models.py` | TF-IDF model zoo (Naive Bayes, logistic regression, linear SVM) behind a leakage-safe preprocessing transformer |
| `embedding_model.py` | Sentence-embedding classifiers (MiniLM, BGE-small) and probability-averaging ensembles |
| `experiments.py` | Week 2 / Phase 5: EDA, ablation, model comparison, C tuning, out-of-scope threshold, final evaluation, sub-intent baseline |
| `predict.py` | Inference for `POST /intent/classify` |
| `load_clinc150.py` | Real out-of-scope data from CLINC150 (official splits kept apart) |

Data: `data/final/{train,validation,test}.csv` (NMIMS) + CLINC150 out-of-scope. Results: `reports/week2/`.
Run from the repo root: `python -m nlp.intent.experiments`.
