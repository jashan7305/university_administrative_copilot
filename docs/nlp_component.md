# NMIMS Administrative NLP component (`nlp.component`)

A reusable component that understands a student's administrative request at NMIMS and returns, as one typed JSON
object:

- the **intent** (12 areas such as certificates, examinations, hostel, or `out_of_scope`) and the specific
  **service** (sub-intent) with the department that handles it;
- the **entities** in the text (documents, dates, amounts, student IDs, ...), with character offsets;
- **evidence passages** from official NMIMS sources, a grounded **answer** and a structured **service request**
  (required documents, next step, missing details, priority);
- a **confidence score** for every prediction, and a **decision** that says what the caller should do next.

The component wraps the trained models of the copilot (Week 4 `v3`, or the Week 3 `v2` models if `v3` is not
installed). It adds the interface, the schemas, the confidence levels and the decision policy. It does not change
any model.

| Version | Value |
|---|---|
| Component | `1.0.0` (`nlp.component.COMPONENT_VERSION`) |
| Output schema | `1.0` (`model.schema_version` in every response) |

## Contents

1. [Setup](#1-setup)
2. [Python](#2-python)
3. [HTTP API](#3-http-api)
4. [Command line](#4-command-line)
5. [Input](#5-input)
6. [Output](#6-output)
7. [Confidence scores](#7-confidence-scores)
8. [Decisions](#8-decisions)
9. [Errors](#9-errors)
10. [Reusing it elsewhere](#10-reusing-it-elsewhere)
11. [Limitations](#11-limitations)

## 1. Setup

```bash
uv sync                                  # or: pip install -r requirements.txt && pip install -e .
python -m nlp.pipeline.train_eval        # trains the v2 models into models/   (CPU is fine)
python -m nlp.pipeline.compare_week4     # trains the v3 models into models/week4/  (GPU recommended)
```

Only one pipeline is needed. By default the component uses `v3` when `models/week4/router` exists and `v2`
otherwise. Override it with `COPILOT_VERSION=v2` or `AdminNLP(pipeline="v2")`. Set `COPILOT_GENERATOR=claude` and
Anthropic credentials to get abstractive answers, which use the same evidence. Otherwise answers are extractive.

## 2. Python

```python
from nlp.component import AdminNLP, AnalyzeInput, Options

nlp = AdminNLP()                                   # models load on the first call, then stay in memory

out = nlp.analyze("I lost my ID card, my roll no is J054. What should I do? It's urgent")
out.intent.intent                  # 'ID_CARDS'
out.intent.confidence.score        # 0.8829   (level 'high')
out.routing.sub_intent             # 'DUPLICATE_ID_CARD'  (confidence 0.781)
out.decision.action                # 'auto_route'
out.service_request.priority       # 'high'   ("urgent")
out.model_dump()                   # plain dict;  out.model_dump_json() -> JSON string

# options and a caller id
out = nlp.analyze(AnalyzeInput(text="lost my id card", request_id="ticket-118",
                               options=Options(top_k=5, k_sources=2, routing_review_below=0.6)))

# single stages
nlp.classify("revaluation of my TEE paper")        # intent + routing only (no NER, retrieval or answer)
nlp.extract_entities("paid Rs 45,000 on 12/08/2026, roll J054")
nlp.retrieve("hostel refund rules")                # evidence passages
nlp.analyze_batch(["query 1", "query 2"])          # up to 64
nlp.info()                                         # labels, thresholds, confidence bands, limits
```

Every call takes either a plain string, which gets the default options, or the matching input model. Every call
returns a pydantic model. A shared instance is available as `nlp.component.get_component()`.

## 3. HTTP API

```bash
uvicorn backend.main:app            # interactive documentation: http://127.0.0.1:8000/docs
```

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/v1/health` | - | status, versions, `models_loaded` (does not load the models) |
| GET | `/v1/info` | - | `ComponentInfo` |
| POST | `/v1/analyze` | `AnalyzeInput` | `AnalyzeOutput` |
| POST | `/v1/analyze/batch` | `BatchInput` | `BatchOutput` |
| POST | `/v1/intent` | `IntentInput` | `IntentOutput` |
| POST | `/v1/entities` | `TextInput` | `EntitiesOutput` |
| POST | `/v1/retrieve` | `RetrieveInput` | `RetrieveOutput` |

```bash
curl -s localhost:8000/v1/analyze -H 'content-type: application/json' \
     -d '{"text": "I lost my ID card. What should I do?", "request_id": "t-1"}'
```

The Week 3 endpoints `/intent/classify` and `/copilot/query` are unchanged.

## 4. Command line

```bash
python -m nlp.component analyze "I lost my ID card, roll no J054"
python -m nlp.component intent "revaluation form last date" --k 5
python -m nlp.component entities "paid Rs 500 on 12/08/2026"
python -m nlp.component retrieve "hostel refund" --k 3
python -m nlp.component batch queries.txt          # one query per line ('-' = stdin), prints JSON lines
python -m nlp.component info
python -m nlp.component schema                     # writes docs/schemas/*.json
```

The exit code is 0 on success, 2 for invalid input and 3 if the models are not trained. Errors are printed as JSON
on stderr.

## 5. Input

`AnalyzeInput`. Unknown fields are rejected, so a typo in a field name is reported instead of silently ignored.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `text` | string, 1-1000 chars, not blank | required | The request; leading and trailing whitespace is stripped |
| `request_id` | string <= 128 chars | `null` | Echoed back unchanged |
| `options.top_k` | int 1-13 | 3 | Ranked alternatives for intent and sub-intent |
| `options.k_sources` | int 1-10 | 3 | Evidence passages |
| `options.routing_review_below` | float 0-1 | 0.5 | Refer to staff below this routing confidence; 0 turns it off |
| `options.include_timings` | bool | true | Per-stage latency in `timings_ms` |

The other inputs follow the same pattern: `IntentInput` (`text`, `request_id`, `top_k`), `TextInput` (`text`,
`request_id`), `RetrieveInput` (`text`, `request_id`, `k`, optional `intent` to boost instead of the predicted one)
and `BatchInput` (`items`: 1-64 `AnalyzeInput`).

## 6. Output

`AnalyzeOutput`:

| Field | Content |
|---|---|
| `request_id`, `query` | Echo of the input (the query after whitespace stripping) |
| `intent` | `intent`, `in_scope`, `confidence`, `rejection_threshold`, `alternatives` (top-k `{label, score}`) |
| `routing` | `sub_intent`, `department`, `confidence`, `alternatives`; `null` when out of scope |
| `entities[]` | `label`, `text`, `start`, `end` (character offsets, end exclusive), `source`, `confidence` |
| `answer`, `answer_generator` | Grounded answer and what wrote it (`extractive_mmr`, `claude`, `fallback`) |
| `sources[]` | `rank`, `doc_id`, `title`, `page`, `url`, `score`, `relative_score` |
| `service_request` | `department`, `required_documents`, `next_action`, `extracted_details`, `missing_information`, `priority` (`high`/`normal`), `status` (`ready`/`needs_information`/`needs_confirmation`), `supporting_facts` (fact ids), `source_urls`; `null` when out of scope |
| `decision` | `action` and a one-sentence `reason` (section 8) |
| `timings_ms` | Per-stage latency, or `null` if `include_timings` is false |
| `model` | `component_version`, `schema_version`, `pipeline` |

Example: the actual output for one query, produced by `python -m nlp.pipeline.evaluate_week6` and saved in
`reports/week6/example_analyze.json`.

```json
{
  "request_id": "example-1",
  "query": "I lost my ID card, my roll no is J054. What should I do? It's urgent",
  "intent": {
    "intent": "ID_CARDS",
    "in_scope": true,
    "confidence": {
      "score": 0.8829,
      "level": "high",
      "method": "top-class probability of the sentence-embedding + TF-IDF ensemble (logistic regression)"
    },
    "rejection_threshold": 0.25,
    "alternatives": [
      {
        "label": "ID_CARDS",
        "score": 0.8829
      },
      {
        "label": "out_of_scope",
        "score": 0.0992
      },
      {
        "label": "TRANSCRIPTS",
        "score": 0.0046
      }
    ]
  },
  "routing": {
    "sub_intent": "DUPLICATE_ID_CARD",
    "department": "School Academic Office (Course Coordinator / AR / DR)",
    "confidence": {
      "score": 0.781,
      "level": "high",
      "method": "routing model probability of the chosen sub-intent (flat or hierarchical, chosen on validation)"
    },
    "alternatives": [
      {
        "label": "DUPLICATE_ID_CARD",
        "score": 0.781
      },
      {
        "label": "ID_CARD_ACCESS",
        "score": 0.035
      },
      {
        "label": "ID_CARD_NOT_WORKING",
        "score": 0.0275
      }
    ]
  },
  "entities": [
    {
      "label": "ISSUE",
      "text": "lost",
      "start": 2,
      "end": 6,
      "source": "crf",
      "confidence": 0.9989
    },
    {
      "label": "DOCUMENT",
      "text": "ID card",
      "start": 10,
      "end": 17,
      "source": "crf",
      "confidence": 0.9995
    },
    {
      "label": "STUDENT_ID",
      "text": "J054",
      "start": 33,
      "end": 37,
      "source": "rule",
      "confidence": null
    }
  ],
  "answer": "Any grievances related to ID Card should be raised to biometricregistration@svkm.ac.in. If a student misplaces the original ID card, a duplicate ID card is issued by the school on payment of a prescribed fee. 2.12 In case a student forgets to carry an ID card, they should approach security and complete the process to get the requested access. 9440/- | | 5. | All Urgent Cases | Rs.\n\nDepartment: School Academic Office (Course Coordinator / AR / DR).\nRequired documents: none stated in the sources.\nNext step: Apply to the school for a duplicate ID card and pay the prescribed fee.",
  "answer_generator": "extractive_mmr",
  "sources": [
    {
      "rank": 1,
      "doc_id": "SRC025-P001",
      "title": "Proforma of Indemnity Bond",
      "page": "p. 1",
      "url": "https://www.nmims.edu/docs/proforma-indemnity-bond new.pdf",
      "score": 0.0291,
      "relative_score": 1.0
    },
    {
      "rank": 2,
      "doc_id": "SRC006-P001",
      "title": "Procedure for Education Verification",
      "page": "p. 1",
      "url": "https://nmims.edu/education-verification",
      "score": 0.0284,
      "relative_score": 0.9759
    },
    {
      "rank": 3,
      "doc_id": "SRC059-P005",
      "title": "SRB Part I - University 2024-25",
      "page": "p. 5",
      "url": "https://navimumbai.nmims.edu/docs/SRB-Part-I-University-2024-25.pdf",
      "score": 0.028,
      "relative_score": 0.9622
    }
  ],
  "service_request": {
    "intent": "ID_CARDS",
    "sub_intent": "DUPLICATE_ID_CARD",
    "department": "School Academic Office (Course Coordinator / AR / DR)",
    "required_documents": [],
    "next_action": "Apply to the school for a duplicate ID card and pay the prescribed fee",
    "extracted_details": {
      "ISSUE": [
        "lost"
      ],
      "DOCUMENT": [
        "ID card"
      ],
      "STUDENT_ID": [
        "J054"
      ]
    },
    "missing_information": [],
    "priority": "high",
    "status": "ready",
    "supporting_facts": [
      "F0002"
    ],
    "source_urls": [
      "https://engineering.nmims.edu/wp-content/uploads/2026/08/SRB_2026_compressed.pdf"
    ]
  },
  "decision": {
    "action": "auto_route",
    "reason": "Confident routing to School Academic Office (Course Coordinator / AR / DR); all required details are present."
  },
  "timings_ms": {
    "intent": 69.85,
    "ner": 4.01,
    "routing": 45.83,
    "retrieval": 47.06,
    "generation": 617.48,
    "total": 784.23
  },
  "model": {
    "component_version": "1.0.0",
    "schema_version": "1.0",
    "pipeline": "v2"
  }
}
```

The machine-readable JSON Schema of every input and output is in [`docs/schemas/`](schemas/). It is generated with
`python -m nlp.component schema` and also served at `/openapi.json`.

## 7. Confidence scores

| Where | Score | How it is computed | Range |
|---|---|---|---|
| `intent.confidence` | top-class probability | v3: weighted blend of the fine-tuned transformer and the embedding ensemble; v2: embedding + TF-IDF ensemble (logistic regression) | 0-1 |
| `routing.confidence` | probability of the chosen sub-intent | routing model; flat or hierarchical (restricted to the predicted intent), chosen on validation | 0-1 |
| `entities[].confidence` | mean tag probability over the entity's tokens | CRF: posterior marginals; transformer: softmax | 0-1, `null` for regex-rule entities (`source = "rule"`), which are deterministic pattern matches |
| `sources[].score` | fused retriever score | reciprocal-rank fusion of BM25 and dense retrieval, intent-boosted | not a probability; compare within one query only |
| `sources[].relative_score` | `score / best score` of the query | | 0-1, the top passage is 1 |
| `alternatives[].score` | probability of each ranked label | same model as the main score | 0-1 |

Each `confidence` object also gives a `level` and the `method` used to compute it. The levels come from the
thresholds the pipeline uses to act:

| Score | Level | Range | Effect |
|---|---|---|---|
| intent | `high` | >= 0.5 | Route |
| intent | `medium` | rejection threshold to 0.5 | Ask the student to confirm the service |
| intent | `low` | < rejection threshold (from training, see `/v1/info`) | Out of scope |
| routing | `high` | >= 0.5 | Route |
| routing | `low` | < 0.5 | Refer to staff |

**How to read the scores.** They are model probabilities, not calibrated probabilities. The Week 5 evaluation
measured an expected calibration error of 0.10 for intent and 0.17 for sub-intent (v3), so 0.8 does not mean
exactly an 80% chance of being right. They do rank well: the routing threshold of 0.5 was chosen on the validation
set as the lowest at which automatically routed questions are at least 90% correct. On the test set that kept
90.7% routing accuracy for the 64% of questions answered automatically. `reports/week6/results.json` gives accuracy
per level on held-out data for the installed pipeline. With v2 on test_core, intent accuracy is 94.0% at level
`high` and 58.3% at `medium`, and routing accuracy is 90.0% at `high` and 58.9% at `low`.

**Example of a low score doing its job.** For "I need a bonafide certificate for a bank loan, urgently" the
intent is right (`CERTIFICATES`, 0.94). However, the knowledge base has no verified source for bonafide
certificates, so no service exists for it. Routing falls back to `MIGRATION_CERTIFICATE` with confidence 0.02.
The decision is `refer_to_staff`, not a wrong automatic request.

## 8. Decisions

`decision.action` is the first rule that matches:

| Order | Action | When | What the caller should do |
|---|---|---|---|
| 1 | `out_of_scope` | intent rejected (below threshold, or predicted out of scope) | Show the fallback answer / helpdesk |
| 2 | `confirm_with_student` | intent level `medium` | Ask which of `intent.alternatives` they mean |
| 3 | `refer_to_staff` | routing confidence < `options.routing_review_below` | Queue for a staff member at `routing.department` |
| 4 | `ask_for_information` | routed, but `service_request.missing_information` is not empty | Ask those questions |
| 5 | `auto_route` | otherwise | Create the service request |

`service_request.status` is the copilot's own status and is unchanged from Week 4. `decision` adds the
routing-confidence check from the Week 5 error analysis on top of it.

## 9. Errors

| HTTP | `error` | When | Python | CLI exit |
|---|---|---|---|---|
| 422 | `validation_error` | blank or too-long text, unknown field, option out of range, more than 64 batch items | `pydantic.ValidationError` | 2 |
| 503 | `models_unavailable` | models are not in `models/` (the message gives the training command) | `nlp.component.ModelsUnavailable` | 3 |

Error bodies are always `{"error": ..., "detail": ...}` (`ErrorOutput`).

## 10. Reusing it elsewhere

- **Other Python code**: install the project (`pip install -e .`) and `from nlp.component import AdminNLP`. The
  only state is the models in `models/` and the knowledge base in `data/final/`.
- **Other languages**: run the HTTP API and generate a client from `/openapi.json`, or validate responses against
  `docs/schemas/AnalyzeOutput.json`.
- **Your own models**: `AdminNLP(copilot=my_copilot)` accepts any `nlp.pipeline.copilot.Copilot`, for example one
  built with different classifiers, a different retriever or another knowledge base. The interface and schemas
  stay the same. `tests/test_week6.py` builds one from stub models.
- **Your own API**: `backend.api.v1.nlp` is a FastAPI dependency. Override it to serve a different `AdminNLP`.
- **Compatibility**: within schema version 1.x fields are only added, never renamed or removed. Check
  `model.schema_version` if you store outputs.

## 11. Limitations

- Confidence scores are not calibrated (section 7). Temperature scaling on the validation set is the next step.
- English (with common Indian-English abbreviations and typos) only. The knowledge base is NMIMS 2026-27; time-sensitive
  facts (fees, dates) should be checked against the linked source.
- The routing threshold was chosen for v3. The v2 models use the same value.
- No authentication or rate limiting. Deploy behind a gateway, with one model copy per worker process.
