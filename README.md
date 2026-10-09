# University Administrative Copilot (NMIMS)

Group 10 (J054, J055, J056, J057, J059). An NLP component that understands NMIMS administrative requests
(certificates, transcripts, ID cards, hostel, examinations, fees, scholarships, attendance, academic registration,
library, student services, student welfare) and routes them to the right department with the documents and next step.

## Repository layout

```
├── src/
│   ├── backend/     FastAPI app (main.py, api/routes.py)
│   ├── nlp/         NLP components: intent/, ner/, retrieval/, knowledge/, pipeline/
│   └── dataset/     NMIMS dataset pipeline, one module per phase (see below)
├── data/
│   ├── annotations/ facts.yaml, queries.yaml - curated dataset inputs from official NMIMS sources
│   ├── sources/     downloaded NMIMS pages/PDFs (gitignored; re-download with phase 1)
│   ├── interim/     extracted text and fetch/extraction logs
│   ├── final/       released dataset (queries, splits, NER, knowledge base, source inventory)
│   ├── external/    CLINC150 out-of-scope queries
│   └── eval/        Week 3 retrieval evaluation queries
├── reports/         dataset/ (quality report, taxonomy, inventory), week2/, week3/
├── tests/           test_dataset.py, test_nlp.py, test_api.py
├── submissions/     submitted reports per week
├── models/          trained models (gitignored)
└── archive/         superseded Week 2 generated-data code (gitignored, local only)
```

## Setup

```bash
uv sync                      # creates .venv with all dependencies and installs src/ packages
source .venv/bin/activate
```
Without uv: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pip install -e .`

## Dataset pipeline (phases)

| Phase | Command | Produces |
|---|---|---|
| 1. Sources | `python -m dataset.phase1_sources [--step collect\|extract\|build]` | `data/sources/`, `data/interim/`, `data/final/source_inventory.csv`, `data/final/nmims_knowledge_base.csv` |
| 2. Annotate | `python -m dataset.phase2_annotate` | core queries + NER from `data/annotations/*.yaml` |
| 3. Expand + split | `python -m dataset.phase3_expand` | `data/final/nmims_admin_queries.csv`, `train/validation/test.csv`, `test_core.csv`, `nmims_ner_annotations.csv`, `split_statistics.csv` |
| 4. Validate | `python -m dataset.phase4_validate` | `reports/dataset/*.md`; exits non-zero if any check fails |

How the data is made: facts are extracted from official NMIMS sources (each with a verbatim quote that phase 4
re-checks); core queries are **written by the project team** from those facts; variants are **rule-generated**
rewordings; NER labels are rule-based. Details and statistics: `reports/dataset/dataset_quality_report.md`.

## Running the API

```bash
uvicorn backend.main:app --reload     # http://127.0.0.1:8000  (GET /health, POST /intent/classify, POST /copilot/query)
```
Train and evaluate on the NMIMS dataset, then rebuild the weekly reports from the saved results:

```bash
python -m nlp.intent.experiments      # Phase 5 / Week 2: preprocessing ablation, baselines, intent + routing models
python -m nlp.pipeline.train_eval     # Phase 6 / Week 3: classification, NER, similarity, RAG, summarization, end-to-end
python -m nlp.reports                 # submissions/week2 and week3 reports (Word)
```

## Tests

```bash
pytest
```

## Status

| Phase | State |
|---|---|
| 0 Restructure | done |
| 1-4 NMIMS dataset (sources, annotations, expansion + split, validation) | done - 5,863 queries, 19/19 checks pass |
| 5 Week 2 baselines on NMIMS data | done - `reports/week2/` |
| 6 Week 3 pipeline v2 on NMIMS data | done - `reports/week3/` |
| Week 4 improved models (fine-tuned transformer router + NER, fine-tuned vector retrieval, LLM summarization, gold NER set) | done - `reports/week4/`, `python -m nlp.pipeline.compare_week4` |
| Week 5 evaluation & error analysis (CIs, significance, calibration, robustness, challenge set, failure cases, fixes) | done - `reports/week5/`, `python -m nlp.pipeline.evaluate_week5` |
