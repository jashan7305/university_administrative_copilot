# Week 4 - Improved Model (baseline vs improved)

Report: `Group10_Week4_Improved_Model_Report.docx`  |  Code: `Group10_Week4_code.zip` and `code/` (same files as in `src/`)

| Component | Baseline | Improved | Code |
|---|---|---|---|
| Classification / routing | TF-IDF + LogReg; frozen-embedding ensemble | SetFit; fine-tuned transformer; transformer + ensemble blend | `src/nlp/intent/finetune.py` |
| NER | Rule-based labeller; CRF | Fine-tuned transformer token classifier (scored on a hand-annotated gold set) | `src/nlp/ner/transformer_ner.py`, `data/annotations/ner_gold.yaml` |
| Retrieval | BM25 keyword search | Hybrid with fine-tuned vector retrieval (cross-encoder re-ranking evaluated, scored lower, not used) | `src/nlp/retrieval/rerank.py` |
| Summarization | Lead / MMR extractive | LLM-assisted (local Qwen2.5-Instruct; Claude optional) | `src/nlp/pipeline/summarize.py` |
| End to end | Copilot v2 | Copilot v3 | `src/nlp/pipeline/copilot.py`, `compare_week4.py` |

Reproduce: `python -m nlp.pipeline.compare_week4` then `python -m nlp.reports week4 code`.
