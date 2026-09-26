# Intent classification: dataset and baseline (Week 2)

## Data
- **Collection:** no public dataset covers university administrative queries, so we collected
  20 utterances for each of the 9 intents in the problem statement, plus an `out_of_scope`
  class (10 classes, 200 utterances) in `seed_data.py`. The seed set is collected by a survey.
- **Split:** stratified 75/25 on the seed utterances, done *before* augmentation (150 train / 50 test).
- **External data:** the `out_of_scope` class also uses real queries from CLINC150 (Larson et al., 2019),
  taken from the [Few-Shot-Intent-Detection](https://github.com/jianguoz/Few-Shot-Intent-Detection)
  repo (`data/external/clinc150_oos/`). 70 of its train+valid queries go into training; its 1,000 test
  queries are used only for evaluation (`test_clinc_oos.csv`). We did not check the licence, so verify it
  before redistributing. Other public student-chatbot datasets were reviewed and not used (too generic).
- **Augmentation (train only):** 5 variants per utterance with greeting/closing phrases and
  random typos, giving 918 training rows after dedup, including the CLINC150 queries. Test sentences never appear in training.
- **Preprocessing:** lowercase, strip punctuation, collapse whitespace (`preprocess.py`).
- **Test sets:** `test.csv` (clean) and `test_noisy.csv` (same sentences with heavy typos).
- **Annotation:** labels come from the seed grouping; no extra annotation was needed.

## Baseline
Word (1-2) + character (2-5) TF-IDF features into logistic regression, in `baseline.py`.
A majority-class dummy is included for reference.

| Model | test acc | test macro-F1 | noisy acc | noisy macro-F1 | CLINC150 OOS recall (1000) |
|---|---|---|---|---|---|
| Majority class | 0.100 | 0.018 | 0.100 | 0.018 | (predicts scholarships) |
| TF-IDF + LogReg | 0.740 | 0.757 | 0.720 | 0.737 | 0.965 |

Character n-grams helped over word-only TF-IDF. The CLINC150 misses are mostly queries that share
words with real intents, such as "credit card" -> id_cards or "fee for a cash advance" -> fees.

**Caveats:** the test set is only 50 sentences, so one example moves accuracy by 2 points. The
sentences are written by the same authors, so real student queries will probably score lower.
Most errors are between neighbouring intents such as certificates, transcripts and fees.

## Reproduce
```bash
cd src/backend
python -m logic.intent.build_dataset   # writes data/train.csv, test.csv, test_noisy.csv
python -m logic.intent.baseline        # trains, evaluates, saves models/intent_baseline.joblib
```
