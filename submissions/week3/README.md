# Week 3 - Core NLP Pipeline (v1)

Reports: `Group10_Week3_NLP_Pipeline_v2_Report.docx` (current) and `Group10_Week3_NLP_Pipeline_v1_Report.docx` (first submission)

Code for this week lives in the main tree:
- Intent (embeddings, ensemble): `src/nlp/intent/embedding_model.py`
- NER (CRF + rules): `src/nlp/ner/extractor.py`
- Knowledge base (727 NMIMS passages) and retrieval (BM25, dense, hybrid): `src/nlp/knowledge/store.py`, `src/nlp/retrieval/search.py`
- Pipeline, service request, training/evaluation: `src/nlp/pipeline/copilot.py`, `train_v1.py`
- API: `src/backend/api/routes.py` (`POST /copilot/query`); results: `reports/week3/`
