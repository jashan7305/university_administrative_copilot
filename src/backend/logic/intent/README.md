# Intent classification (Week 2: dataset and baseline)

This module classifies an administrative query into one of 10 classes: the 9 intents from the problem
statement plus `out_of_scope`.

## Layout

| File | Purpose |
|---|---|
| `../../data/processed/` | Cleaned, split dataset; cleaning audit; entity annotations |
| `preprocess.py` | Configurable text preprocessing pipeline and domain spelling corrector |
| `models.py` | Model zoo (NB, LogReg, linear SVM) behind a leakage-safe preprocessing transformer |
| `experiments.py` | EDA, preprocessing ablation, model comparison, C tuning, OOS threshold, final evaluation |
| `predict.py` | Inference used by `POST /intent/classify` |

## Data pipeline

**Sources**
- **Seed set:** 200 utterances written by the team.
- **Frames:** 195 sentence frames filled with slot values (documents, fee types, subjects, programs and so on). The span of every slot is recorded, which gives 2,715 entity-annotated sentences (21 entity types) for the NER stage.
- **Real out-of-scope data:** CLINC150 queries (Larson et al., 2019), taken from the [Few-Shot-Intent-Detection](https://github.com/jianguoz/Few-Shot-Intent-Detection) repo.
- **Hard negatives:** 40 queries that are university-related but not administrative, such as the canteen menu or assignment help.

**Cleaning** (audited in `data/processed/cleaning_report.json`)
- Unicode NFKC normalisation and whitespace cleanup.
- Removal of empty rows and of rows that are too short (<2 tokens) or too long (>60 tokens).
- Label-conflict detection: the same normalised text under two intents.
- Removal of exact and near duplicates.
- A cross-split leakage check.

**Split (leakage-controlled)**
- **Frame-disjoint split:** each intent's frames are split 70/15/15 into train, val and test, so no test sentence shares a template with a training sentence.
- **Held-out slot values:** 20% of slot values appear only in val and test.
- **Seeds:** split with stratification.
- **CLINC150:** its official test split is kept for evaluation only.

**Augmentation (train only, after the split):** typos (swap, drop, double, keyboard replace), chat-speak ("pls", "u", "sem"), casing changes and dropped punctuation.

| Split | Rows | Contents |
|---|---|---|
| train | 3,101 | Frames, seeds, CLINC150, hard negatives, augmented copies |
| val | 309 | Unseen frames; used for model selection |
| test | 348 | Unseen frames and unseen slot values; touched once |
| test_noisy | 348 | `test` with heavy typos and chat-speak |
| test_seed | 49 | Held-out hand-written seeds |
| test_clinc_oos | 999 | CLINC150 out-of-scope test set |

## Preprocessing

The pipeline runs these steps in order:
1. NFKC normalisation
2. Lowercasing
3. Squeezing repeated characters
4. Contraction expansion
5. Expansion of chat-speak and domain abbreviations (NOC, TC, sem, dept, reval, ATKT and so on)
6. Entity masking (`<email> <url> <phone> <date> <money> <percent> <rollno> <num>`)
7. Tokenisation
8. Spelling correction against the domain vocabulary, with an English dictionary guard
9. Stopword removal that keeps wh-words and negations
10. WordNet lemmatisation (function words protected)

Every step can be switched on or off through `PreprocessConfig`. The spelling vocabulary is fitted on the training fold only.

## Results

**Model selection:** 5-fold GroupKFold cross-validation on train (grouped by frame) plus the validation split. Test sets are used once.

| Model (full preprocessing) | CV macro-F1 | Val macro-F1 |
|---|---|---|
| Majority class | 0.013 | 0.018 |
| Complement Naive Bayes | 0.843 ± 0.055 | 0.722 |
| LogReg, word TF-IDF | 0.916 ± 0.067 | 0.898 |
| Linear SVM, word+char TF-IDF | 0.919 ± 0.079 | 0.918 |
| LogReg, word+char, unweighted | 0.919 ± 0.066 | 0.915 |
| **LogReg, word+char, class-balanced (C=10)** | **0.922 ± 0.067** | **0.931** |

**Final model on the test sets** (95% bootstrap confidence intervals):

| Test set | Accuracy | Macro-F1 |
|---|---|---|
| test (348) | 0.968 [0.951, 0.986] | 0.970 [0.953, 0.986] |
| test_noisy (348) | 0.954 [0.931, 0.974] | 0.954 [0.931, 0.975] |
| test_seed (49) | 0.898 [0.796, 0.980] | 0.903 [0.787, 0.972] |
| CLINC150 OOS (999) | 0.960 out-of-scope recall | – |

The majority-class reference scores 0.10 accuracy. Inference latency is about 0.4 ms per query on CPU.

**Main error:** 7 of 29 examination queries in `test` come from one unseen frame ("I failed X, how do I reappear?"), and all 7 are rejected as out_of_scope. Wording that appears in no training frame is the model's main weakness.

## Reproduce

```bash
cd src/backend
python -m logic.intent.experiments     # reports/week2/*, models/intent_baseline.joblib (~1.5 min)
python -m pytest tests                 # preprocessing + split-integrity tests
uvicorn main:app --reload              # POST /intent/classify {"text": "I lost my ID card"}
```
