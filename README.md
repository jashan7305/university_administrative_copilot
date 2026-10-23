# University Administrative Copilot (NMIMS)

Group 10 (J054, J055, J056, J057, J059). An NLP component that understands NMIMS administrative requests
(certificates, transcripts, ID cards, hostel, examinations, fees, scholarships, attendance, academic registration,
library, student services, student welfare) and routes them to the right department with the documents and next step.

## Repository layout

```
├── src/
│   ├── backend/     FastAPI app (main.py, api/), demo UI (static/, served at /ui/)
│   ├── nlp/         NLP components: intent/, ner/, retrieval/, knowledge/, pipeline/, component/ (reusable asset),
│   │                electives/ (elective advisor), integrations/ (other groups' components)
│   └── dataset/     NMIMS dataset pipeline, one module per phase (see below)
├── data/
│   ├── annotations/ facts.yaml, queries.yaml - curated dataset inputs from official NMIMS sources
│   ├── sources/     downloaded NMIMS pages/PDFs (gitignored; re-download with phase 1)
│   ├── interim/     extracted text and fetch/extraction logs
│   ├── final/       released dataset (queries, splits, NER, knowledge base, source inventory)
│   ├── external/    CLINC150 out-of-scope queries
│   └── eval/        Week 3 retrieval evaluation queries
├── docs/            elective_advisor.md, integration.md, responsible_ai.md, nlp_component.md, schemas/
├── reports/         dataset/ (quality report, taxonomy, inventory), week2/ ... week6/
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

## Demo: elective advisor (Week 7)

```bash
uvicorn backend.main:app      # open http://127.0.0.1:8000/ui/?demo
```
Several syllabi on a topic (including a new course) plus the student's academic identity -> the elective the
student can enrol in, with every course checked against the official MPSTME programme structures and rules.
See [docs/elective_advisor.md](docs/elective_advisor.md), [docs/integration.md](docs/integration.md) and
[docs/responsible_ai.md](docs/responsible_ai.md).

## Using the NLP component

The whole pipeline is packaged as one reusable component with typed input/output and confidence scores
(documentation: [docs/nlp_component.md](docs/nlp_component.md)):

```python
from nlp.component import AdminNLP
out = AdminNLP().analyze("I lost my ID card, roll no J054. It's urgent")
out.intent.confidence.score, out.routing.department, out.decision.action
```
```bash
python -m nlp.component analyze "I lost my ID card"      # CLI, prints JSON
uvicorn backend.main:app                                  # HTTP: POST /v1/analyze, /v1/intent, /v1/entities, /v1/retrieve
```

## Running the API

```bash
uvicorn backend.main:app --reload     # http://127.0.0.1:8000/docs  (/v1/* component API; Week 3 /intent/classify, /copilot/query)
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
| Week 6 reusable NLP asset (module + HTTP API + CLI, structured I/O, confidence scores, decisions, docs) | done - `src/nlp/component/`, `docs/nlp_component.md`, `python -m nlp.pipeline.evaluate_week6` |
| Week 7 UI, API integration, responsible AI, architecture integration, partner components (elective advisor demo) | done - `src/nlp/electives/`, `src/backend/static/`, `docs/`, `python -m nlp.pipeline.evaluate_week7` |
